"""Joint model of the Phase 4 spike's ``BART × SV`` composition for Geweke and SBC (SPEC §13 Phase
4 question 2; ``docs/phase4/SPIKE.md``). **Prototype, not API.**

:class:`BartJoint` implements ``validation.geweke.JointModel`` for

    A y_t = g(x_t) + Λ_t^{½} ε_t,   g_i = Σ_{m=1}^{M} tree_{i,m}(bin(x_t)),   x_t = (y_{t−1}', …, y_{t−p}')'

with the recursive structure and SV core of the ``zero_sv`` composition (``joint.joint_spec``,
its test prior and fixed design; ``a_ij ~ N(0, 0.7²)``) and the tree prior of
``components.mean._bart_spike.BartPrior`` (Chipman, George & McCulloch 2010, in bartz's
convention: a node at depth δ splits with probability ``alpha/(1 + δ)**beta`` when a decision rule
is available, the variable is uniform over the variables with one, the cutpoint uniform over the
available ones; leaves ``N(0, (g_sd/√M)²)``) on a **fixed** cutpoint grid.

Two-implementation rule (DESIGN D-22): the tree prior sampler, the forest evaluator used by the
simulator and the test functions, and the tree log-prior are written here in box form (a node owns
the bins ``[L_v, H_v)`` of every variable; a cutpoint ``c`` is available iff ``L_v < c < H_v``),
independently of bartz's heap descent and of the kernel's ``traverse``. ``step`` is the production
sweep (``compose`` with the BART mean in the ``mean`` slot → ``make_sweep``): BART step, then
``a_i | g_i`` (``Recursive.contemp_step``), then the SV block, with the frozen blocks of a block
configuration restored afterwards (``joint.hold_frozen``).

Test functions: the ``zero_sv`` reference functions that do not involve the mean (``a``, SV
parameters and states, data functions) plus, per equation, ``g`` at three fixed probe points and
their squares, the mean and mean square of ``g(x_t)`` over the sample, ``g`` at three dates, the
leaf count and its square, the mean tree depth, split counts per variable, the leaf sum of squares,
and the tree log-prior; the log-likelihood, ``Σ e²/λ`` and ``log p(θ)`` are recomputed with the
BART residuals ``e = A y_t − g(x_t)``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.linalg import solve_triangular

from rai_bvar.components.mean._bart_spike import BartPrior, BartSpikeMean, template_state
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc import data as engine_data
from rai_bvar.engines.mcmc.state import ChainState, Empty, EngineData
from rai_bvar.validation import testfns
from rai_bvar.validation.joint import ALL, JointSpec, hold_frozen, joint_spec
from rai_bvar.validation.log_joint import LOG_2PI, log_prior
from rai_bvar.validation.reference import effective_log_var
from rai_bvar.validation.simulate import unit_lower

Array = jax.Array
_F64 = jnp.float64

__all__ = [
    "BartJoint",
    "Forest",
    "bart_joint",
    "forest_boxes",
    "forest_eval",
    "forest_prior_sample",
]

#: base test functions whose value involves the mean; :class:`BartJoint` recomputes them
_MEAN_DEPENDENT = ("loglik", "log_prior")


# --------------------------------------------------------------------------- path (b) experiment


@contextlib.contextmanager
def bartz_x64_patch() -> Iterator[None]:
    """**Experiment only** (memo §Q2, path b): the two one-line fixes that let bartz 0.13.0 trace
    ``step`` under ``jax_enable_x64`` with float64 residuals and precision scales, applied as a
    monkeypatch while the context is open (and while anything traced in it is compiled):

    1. ``mcmcstep._moves._propose_moves``: cast ``Moves.lrt_nodes`` to int32 (under x64,
       ``node << jnp.array([1, 1, 0])`` is int64 and ``accept_move_and_sample_leaves`` asserts
       int32);
    2. ``mcmcstep._reduction.BatchedReduction._reduce``: cast floating ``values`` to the
       accumulator dtype before the scatter (float64 values into the float32 per-leaf sums raise
       JAX's "scatter inputs have incompatible types" FutureWarning, an error in future JAX).

    Upstream these are ``.astype(jnp.int32)`` on ``lrt_nodes`` and ``.add(values.astype(dtype))``.
    """
    from bartz.mcmcstep import _moves, _reduction

    propose, reduce = _moves._propose_moves, _reduction.BatchedReduction._reduce

    def propose_int32(*args: Any, **kwargs: Any) -> Any:
        moves = propose(*args, **kwargs)
        return dataclasses.replace(moves, lrt_nodes=moves.lrt_nodes.astype(jnp.int32))

    def reduce_cast(self: Any, values: Any, indices: Array, /, **kwargs: Any) -> Array:
        values = jnp.asarray(values)
        if jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(kwargs["dtype"])
        return reduce(self, values, indices, **kwargs)

    _moves._propose_moves = propose_int32
    _reduction.BatchedReduction._reduce = reduce_cast
    try:
        yield
    finally:
        _moves._propose_moves = propose
        _reduction.BatchedReduction._reduce = reduce


# --------------------------------------------------------------------------- reference trees


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Forest:
    """One equation's forest in heap layout (node 1 the root, children ``2k, 2k+1``):
    ``var``/``split`` ``i32[M, half]`` (``split = 0``: leaf or unused), ``leaf`` ``f64[M, 2·half]``
    in data units (meaningful at actual leaves only)."""

    var: Array
    split: Array
    leaf: Array


def _heap_depth(size: int) -> np.ndarray:
    return np.floor(np.log2(np.maximum(np.arange(size), 1))).astype(np.int64)


def forest_boxes(var: Array, split: Array, p_x: int, max_split: int) -> tuple[Array, Array, Array]:
    """``(L, H, exists)`` of every heap slot (``i32[..., 2·half, p_x]`` twice, ``bool[..., 2·half]``):
    node ``k`` owns bins ``X_v ∈ [L_v, H_v)``; the root owns ``[0, max_split + 1)``; a split
    ``(v, c)`` gives the left child ``H_v = c`` and the right child ``L_v = c``."""
    half = var.shape[-1]
    size = 2 * half
    batch = var.shape[:-1]
    L = jnp.zeros((*batch, size, p_x), jnp.int32)
    H = jnp.full((*batch, size, p_x), max_split + 1, jnp.int32)
    exists = jnp.zeros((*batch, size), bool).at[..., 1].set(True)
    onehot_v = jnp.arange(p_x)
    for node in range(2, size):  # parents come first in heap order
        par = node >> 1
        v = var[..., par][..., None]
        c = split[..., par][..., None]
        hit = onehot_v == v
        if node & 1:
            L = L.at[..., node, :].set(jnp.where(hit, c, L[..., par, :]))
            H = H.at[..., node, :].set(H[..., par, :])
        else:
            H = H.at[..., node, :].set(jnp.where(hit, c, H[..., par, :]))
            L = L.at[..., node, :].set(L[..., par, :])
        exists = exists.at[..., node].set(exists[..., par] & (split[..., par] > 0))
    return L, H, exists


def forest_prior_sample(key: Array, prior: BartPrior, p_x: int) -> tuple[Forest, Array]:
    """One forest from the tree prior (module docstring) and its growable-leaf mask
    (``bool[M, half]``: an existing leaf above the bottom level with an available rule, bartz's
    ``affluence_tree``). Nodes are visited level by level; each node's box comes from its parent."""
    M, d, S = prior.num_trees, prior.max_depth, prior.max_split
    half = 2 ** (d - 1)
    pnt = jnp.asarray(prior.p_nonterminal(), _F64)

    def one(k: Array) -> tuple[Forest, Array]:
        var = jnp.zeros(half, jnp.int32)
        split = jnp.zeros(half, jnp.int32)
        L = jnp.zeros((half, p_x), jnp.int32)
        H = jnp.full((half, p_x), S + 1, jnp.int32)
        exists = jnp.zeros(half, bool).at[1].set(True)
        for depth in range(d - 1):
            nodes = np.arange(2**depth, 2 ** (depth + 1))
            if depth:
                par, right = nodes >> 1, (nodes & 1).astype(bool)
                hit = jnp.arange(p_x)[None, :] == var[par][:, None]
                c = split[par][:, None]
                L = L.at[nodes].set(jnp.where(hit & right[:, None], c, L[par]))
                H = H.at[nodes].set(jnp.where(hit & ~right[:, None], c, H[par]))
                exists = exists.at[nodes].set(split[par] > 0)
            lo, hi = L[nodes], H[nodes]
            avail = hi - lo > 1  # some cutpoint c with lo < c < hi
            kd = random.fold_in(k, depth)
            u = random.uniform(random.fold_in(kd, 0), (nodes.size,), _F64)
            grow = exists[nodes] & jnp.any(avail, axis=1) & (u < pnt[depth])
            v = random.categorical(random.fold_in(kd, 1), jnp.where(avail, 0.0, -jnp.inf), axis=1)
            v = v.astype(jnp.int32)
            lo_v = jnp.take_along_axis(lo, v[:, None], 1)[:, 0]
            hi_v = jnp.take_along_axis(hi, v[:, None], 1)[:, 0]
            c = random.randint(
                random.fold_in(kd, 2),
                (nodes.size,),
                lo_v + 1,
                jnp.maximum(hi_v, lo_v + 2),
                jnp.int32,
            )
            var = var.at[nodes].set(jnp.where(grow, v, 0))
            split = split.at[nodes].set(jnp.where(grow, c, 0))
        leaf = prior.leaf_sd * random.normal(random.fold_in(k, d), (2 * half,), _F64)
        growable = exists & (split == 0) & jnp.any(H - L > 1, axis=1)
        return Forest(var, split, leaf), growable.at[0].set(False)

    return jax.vmap(one)(random.split(key, M))


def forest_eval(forest: Forest, Xbin: Array, max_split: int) -> Array:
    """``g`` at the columns of ``Xbin`` (``i32[p_x, T]`` bins) by box membership: ``f64[T]``."""
    p_x = Xbin.shape[0]
    L, H, exists = forest_boxes(forest.var, forest.split, p_x, max_split)
    size = forest.leaf.shape[-1]
    is_leaf = exists & (jnp.concatenate([forest.split, jnp.zeros_like(forest.split)], -1) == 0)
    x = Xbin.T[None, None, :, :]  # (1, 1, T, p_x)
    inside = jnp.all((L[:, :, None, :] <= x) & (x < H[:, :, None, :]), axis=-1)  # (M, size, T)
    w = jnp.where(is_leaf, forest.leaf, 0.0)
    assert w.shape[-1] == size
    return jnp.einsum("mkt,mk->t", inside.astype(_F64), w)


def tree_log_prior(forest: Forest, prior: BartPrior, p_x: int, *, leaves: bool = True) -> Array:
    """``log p(trees) + log p(leaves)`` of one forest (structure: split/no-split at every existing
    node above the bottom level that has an available rule, times the uniform variable and
    cutpoint probabilities; leaves: Gaussian at the actual leaves; ``leaves=False``: structure
    only)."""
    S = prior.max_split
    half = forest.var.shape[-1]
    L, H, exists = forest_boxes(forest.var, forest.split, p_x, S)
    is_leaf = exists & (jnp.concatenate([forest.split, jnp.zeros_like(forest.split)], -1) == 0)
    L, H, ex = L[:, :half], H[:, :half], exists[:, :half]
    pnt = jnp.asarray(np.concatenate([prior.p_nonterminal(), [0.0]]), _F64)[_heap_depth(half)]
    n_cut = H - L - 1  # available cutpoints per variable
    n_var = jnp.sum(n_cut > 0, axis=-1)
    internal = ex & (forest.split > 0)
    leaf_avail = ex & (forest.split == 0) & (n_var > 0)
    n_cut_v = jnp.take_along_axis(n_cut, forest.var[..., None], -1)[..., 0]
    node_split = jnp.where(
        internal,
        jnp.log(pnt) - jnp.log(jnp.maximum(n_var, 1)) - jnp.log(jnp.maximum(n_cut_v, 1)),
        0.0,
    )
    node_stop = jnp.where(leaf_avail, jnp.log1p(-pnt), 0.0)
    s2 = prior.leaf_sd**2
    leaf_lp = jnp.where(is_leaf, -0.5 * (LOG_2PI + jnp.log(s2) + forest.leaf**2 / s2), 0.0)
    out = jnp.sum(node_split) + jnp.sum(node_stop)
    return out + jnp.sum(leaf_lp) if leaves else out


def forest_from_state(ms: Any) -> Forest:
    """The reference :class:`Forest` of a bartz ``State`` (one equation): leaves in data units."""
    f = ms.forest
    leaf = jnp.asarray(f.leaf_tree, _F64) * jnp.asarray(f.leaf_unit, _F64)
    return Forest(f.var_tree.astype(jnp.int32), f.split_tree.astype(jnp.int32), leaf)


def embed_forest(tmpl: Any, forest: Forest, growable: Array) -> Any:
    """``tmpl`` (a bartz ``State``) with the trees of ``forest``: leaves rounded to the stored
    float32 units (``leaf_unit`` is a power of two, so the conversion is exact)."""
    f = tmpl.forest
    unit = jnp.asarray(f.leaf_unit, _F64)
    forest_new = dataclasses.replace(
        f,
        var_tree=forest.var.astype(f.var_tree.dtype),
        split_tree=forest.split.astype(f.split_tree.dtype),
        leaf_tree=(forest.leaf / unit).astype(f.leaf_tree.dtype),
        affluence_tree=growable.astype(f.affluence_tree.dtype),
    )
    return dataclasses.replace(tmpl, forest=forest_new)


# --------------------------------------------------------------------------- joint model


@dataclass(frozen=True, eq=False)
class BartJoint:
    """``BART × SV`` (or ``× Constant``) joint model through the production sweep (module
    docstring). ``base`` is the ``zero_sv`` (or ``zero``-mean) :class:`JointSpec` supplying the
    structure, the volatility block, their test prior and the fixed design."""

    name: str
    base: JointSpec
    bart: BartPrior
    live: frozenset[str] = ALL
    n_probe: int = 3
    probe_seed: int = 20261025

    def __post_init__(self) -> None:
        _ = self.kernel  # builds the bartz template eagerly (never first inside a trace)

    # ----------------------------------------------------------------- statics
    @property
    def static(self):
        return self.base.static

    @property
    def p_x(self) -> int:
        s = self.static.shapes
        return s.n * s.lags

    @cached_property
    def kernel(self) -> BartSpikeMean:
        return BartSpikeMean.for_static(self.static, self.bart)

    @cached_property
    def probes(self) -> np.ndarray:
        """Fixed probe points (bins): the middle bin, a ramp, and random bins (``i32[p_x, P]``)."""
        S = self.bart.max_split
        rng = np.random.default_rng(self.probe_seed)
        mid = np.full(self.p_x, (S + 1) // 2)
        ramp = np.round(np.linspace(S / 4, 3 * S / 4, self.p_x)).astype(int)
        rand = rng.integers(0, S + 1, size=(self.n_probe, self.p_x))
        pts = np.vstack([mid, ramp, rand])[: self.n_probe]
        return pts.T.astype(np.int32)

    @cached_property
    def _program(self):
        from rai_bvar.engines.mcmc.compose import compose
        from rai_bvar.engines.mcmc.sweep import make_sweep

        program = compose(self.static, overrides={"mean": self.kernel})
        return program, make_sweep(program)

    @property
    def plan(self):
        return self._program[1].plan

    def prepare(self, data: EngineData) -> Any:
        base = engine_data.base_cache(data, None, self.static)
        return self.kernel.prepare(data, self.base.prior, base)

    # ----------------------------------------------------------------- JointModel
    def prior_sample(self, key: Array) -> ChainState:
        st = self.base.prior_sample(random.fold_in(key, 0))
        n = self.static.shapes.n
        tmpl = jax.tree.map(
            jnp.asarray, template_state(self.bart, self.p_x, self.static.shapes.T_pad)
        )
        keys = jax.vmap(lambda i: random.fold_in(random.fold_in(key, 1), i))(
            jnp.arange(n, dtype=jnp.uint32)
        )
        forests, growable = jax.vmap(lambda k: forest_prior_sample(k, self.bart, self.p_x))(keys)
        mean = jax.vmap(lambda f, g: embed_forest(tmpl, f, g))(forests, growable)
        return dataclasses.replace(st, eq=dataclasses.replace(st.eq, mean=mean))

    def forests(self, st: ChainState) -> Forest:
        return jax.vmap(forest_from_state)(st.eq.mean)

    def simulate(self, key: Array, st: ChainState) -> EngineData:
        """``y ~ p(y | θ)``, recursively in t with the current latents (Geweke error 2)."""
        s = self.static.shapes
        n, p, T_pad = s.n, s.lags, s.T_pad
        fixed = self.base.fixed
        tmask = (jnp.arange(T_pad) < fixed.T_eff).astype(_F64)
        eps = random.normal(key, (T_pad, n), _F64)
        shocks = jnp.exp(0.5 * effective_log_var(st.eq.vol, self.static)).T * eps
        A = unit_lower(st.eq.struct.a)
        forests = self.forests(st)
        S = self.bart.max_split
        L, H, exists = jax.vmap(lambda f: forest_boxes(f.var, f.split, self.p_x, S))(forests)
        zeros = jnp.zeros_like(forests.split)
        is_leaf = exists & (jnp.concatenate([forests.split, zeros], -1) == 0)
        w = jnp.where(is_leaf, forests.leaf, 0.0)  # (n, M, size)
        edges = jnp.asarray(self.bart.cuts, _F64)

        def body(window: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
            e_t, m_t = inp
            xb = jnp.searchsorted(edges, window[::-1].reshape(-1), side="right").astype(jnp.int32)
            inside = jnp.all((L <= xb) & (xb < H), axis=-1)  # (n, M, size)
            g = jnp.sum(jnp.where(inside, w, 0.0), axis=(1, 2))
            y = solve_triangular(A, g + e_t, lower=True, unit_diagonal=True)
            y = jnp.where(m_t > 0, y, 0.0)
            return jnp.concatenate([window[1:], y[None]], axis=0), y

        window0 = jnp.asarray(fixed.presample, _F64).reshape(p, n)
        _, Y = lax.scan(body, window0, (shocks, tmask))
        return engine_data.build_engine_data(
            Y, window0, jnp.asarray(fixed.T_eff, jnp.int32), n_det=s.n_det, t0=fixed.t0
        )

    def step(self, key: Array, it: Array, st: ChainState, data: EngineData) -> ChainState:
        _, sweep = self._program
        cache = self.prepare(data)
        new = sweep(key, it, st, data, self.base.prior, cache)
        return hold_frozen(new, st, self.live)

    # ----------------------------------------------------------------- test functions
    def g_values(self, st: ChainState, data: EngineData) -> Array:
        """``g_i(x_t)`` by the reference evaluator, ``f64[n, T_pad]`` (0 on padded rows)."""
        Xbin = jnp.searchsorted(
            jnp.asarray(self.bart.cuts, _F64), data.X[:, self.static.shapes.n_det :], side="right"
        ).T.astype(jnp.int32)
        g = jax.vmap(lambda f: forest_eval(f, Xbin, self.bart.max_split))(self.forests(st))
        return g * data.tmask[None, :]

    def _functions(self, st: ChainState, data: EngineData) -> testfns.TestFns:
        s = self.static.shapes
        n, T_eff = s.n, self.base.fixed.T_eff
        zero_view = dataclasses.replace(st, eq=dataclasses.replace(st.eq, mean=Empty()))
        base = testfns.test_functions(
            zero_view, data, self.base.prior, self.static, tails=self.base.tails
        )
        h_ok = self.base.tails.h2 > 1.0
        out: testfns.TestFns = {}
        for k, v in base.items():  # keep the mean-free functions in their order
            if k in _MEAN_DEPENDENT or k.endswith(".joint.rss_std") or k.endswith(".joint.loglik"):
                continue
            out[k] = v
        forests = self.forests(st)
        g = self.g_values(st, data)
        m = data.tmask
        probes = jnp.asarray(self.probes)
        S = self.bart.max_split
        half = forests.var.shape[-1]
        for i in range(n):
            f = jax.tree.map(lambda x, i=i: x[i], forests)
            gp = forest_eval(f, probes, S)
            for j in range(self.n_probe):
                out[f"eq{i}.mean.g_probe[{j}]"] = (gp[j], True)
                out[f"eq{i}.mean.g_probe_sq[{j}]"] = (gp[j] ** 2, True)
            out[f"eq{i}.mean.g_mean"] = (jnp.sum(g[i] * m) / jnp.sum(m), True)
            out[f"eq{i}.mean.g_ms"] = (jnp.sum(g[i] ** 2 * m) / jnp.sum(m), True)
            for t in (0, T_eff // 2, T_eff - 1):
                out[f"eq{i}.mean.g_t[{t}]"] = (g[i, t], True)
            _, _, exists = forest_boxes(f.var, f.split, self.p_x, S)
            is_leaf = exists & (jnp.concatenate([f.split, jnp.zeros_like(f.split)], -1) == 0)
            n_leaves = jnp.sum(is_leaf).astype(_F64)
            out[f"eq{i}.mean.n_leaves"] = (n_leaves, True)
            out[f"eq{i}.mean.n_leaves_sq"] = (n_leaves**2, True)
            depth = jnp.asarray(_heap_depth(2 * half), _F64)
            tree_depth = jnp.max(jnp.where(is_leaf, depth, 0.0), axis=-1)
            out[f"eq{i}.mean.depth_mean"] = (jnp.mean(tree_depth), True)
            internal = exists[:, :half] & (f.split > 0)
            for v in range(self.p_x):
                cnt = jnp.sum(internal & (f.var == v)).astype(_F64)
                out[f"eq{i}.mean.n_split_var[{v}]"] = (cnt, True)
            out[f"eq{i}.mean.leaf_ss"] = (jnp.sum(jnp.where(is_leaf, f.leaf**2, 0.0)), True)
            out[f"eq{i}.mean.tree_log_prior"] = (tree_log_prior(f, self.bart, self.p_x), True)
        # mean-dependent functions with the BART residuals e = A y_t − g(x_t)
        A = unit_lower(st.eq.struct.a)
        e = (data.Y @ A.T * m[:, None]).T - g
        lv = effective_log_var(st.eq.vol, self.static)
        live = m[None, :] > 0
        lv = jnp.where(live, lv, 0.0)
        terms = jnp.where(live, -0.5 * (LOG_2PI + lv + e**2 * jnp.exp(-lv)), 0.0)
        rss = jnp.sum(jnp.where(live, e**2 * jnp.exp(-lv), 0.0), axis=1)
        out["loglik"] = (jnp.sum(terms), h_ok)
        lp_tree = sum(out[f"eq{i}.mean.tree_log_prior"][0] for i in range(n))
        out["log_prior"] = (log_prior(zero_view, self.base.prior, self.static, m) + lp_tree, True)
        for i in range(n):
            out[f"eq{i}.joint.rss_std"] = (rss[i], True)
            if n > 1:
                out[f"eq{i}.joint.loglik"] = (jnp.sum(terms[i]), h_ok)
        return {k: (jnp.asarray(v, _F64), bool(fin)) for k, (v, fin) in out.items()}

    @cached_property
    def function_flags(self) -> dict[str, bool]:
        flags: dict[str, bool] = {}

        def run(key: Array) -> dict[str, Array]:
            st = self.prior_sample(key)
            d = self._functions(st, self.simulate(key, st))
            flags.update({k: f for k, (_, f) in d.items()})
            return {k: v for k, (v, _) in d.items()}

        jax.eval_shape(run, random.key(0))
        return flags

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(k for k, fin in self.function_flags.items() if fin)

    @property
    def all_names(self) -> tuple[str, ...]:
        return tuple(self.function_flags)

    @property
    def sbc_names(self) -> tuple[str, ...]:
        """SBC quantities (DESIGN §9.6, as ``sbc_models.PredictiveJoint``): every test function
        except data functions (constant given ``y``), squares and cross-products."""
        skip = ("^2", "_sq", "*")
        return tuple(
            k for k in self.all_names if not k.startswith("data.") and not any(s in k for s in skip)
        )

    def sbc_quantities(self, st: ChainState, data: EngineData) -> Array:
        d = self._functions(st, data)
        return jnp.stack([d[k][0] for k in self.sbc_names])

    def test_functions(self, st: ChainState, data: EngineData) -> Array:
        d = self._functions(st, data)
        return jnp.stack([d[k][0] for k in self.names])

    def quantities(self, st: ChainState, data: EngineData) -> Array:
        d = self._functions(st, data)
        return jnp.stack([d[k][0] for k in self.all_names])


@functools.cache
def _bart_joint_cached(name: str, key: tuple) -> BartJoint:
    kw = dict(key)
    bart = kw.pop("bart")
    live = frozenset(kw.pop("live", ("all",)))
    base_name = kw.pop("base_name", "zero_sv")
    size = kw.pop("size", "small")
    if "prior_kwargs" in kw:
        kw["prior_kwargs"] = dict(kw["prior_kwargs"])
    base = joint_spec(base_name, size, **kw)
    return BartJoint(name=name, base=base, bart=bart, live=live)


def bart_joint(
    name: str,
    *,
    bart: BartPrior,
    n: int,
    T_eff: int,
    lags: int = 1,
    size: Literal["small", "full"] = "small",
    live: frozenset[str] | tuple[str, ...] = ALL,
    base_name: str = "zero_sv",
    prior_kwargs: dict[str, float] | None = None,
) -> BartJoint:
    """A cached :class:`BartJoint`: ``base_name`` (``zero_sv``) at ``n``, ``lags``, ``T_eff`` with
    the reference test prior (``prior_kwargs`` passed to ``joint.reference_prior``)."""
    key = {
        "bart": bart, "n": n, "T_eff": T_eff, "lags": lags, "size": size,
        "live": tuple(sorted(live)), "base_name": base_name,
    }  # fmt: skip
    if prior_kwargs:
        key["prior_kwargs"] = tuple(sorted(prior_kwargs.items()))
    return _bart_joint_cached(name, tuple(sorted(key.items())))
