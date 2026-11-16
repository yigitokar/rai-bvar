"""Phase 4 week-1 spike (SPEC §13): a BART mean block on ``bartz`` 0.13.0. **Prototype, not API.**

Not registered, not exported (no ``rb.BART``): tests compose it through the kernel-slot test seam,
``compose(static, overrides={"mean": BartSpikeMean(...)})`` with a ``MeanStatic("zero")``
placeholder, so the production sweep runs ``mean → contemp → residual → vol`` (DESIGN §4.3; the
non-fusing, stateful-mean plan of ``plan.build_plan``). The decision memo is
``docs/phase4/SPIKE.md``.

Model of equation ``i`` (SPEC §4.1, recursive structural form)::

    target_t = ỹ_{i,t} + Σ_{j<i} a_ij ỹ_{j,t} = g_i(x_t) + √λ_{i,t} ε_t,   g_i = offset + Σ_m tree_m

with ``x_t`` the lag block of the design row (``data.X[:, n_det:]``, no deterministic columns: a
constant column has no cutpoint) binned on a **fixed** cutpoint grid (:class:`BartPrior.cuts`), so
the tree prior does not depend on the data (Geweke and SBC need that; production may bin per
window, R9). The weights ``w_t = m_t/λ_eff,i,t`` of the ``EqContext`` are the per-observation
precisions; bartz's own error precision is held fixed at 1 (``Wishart(None, None, 1)``), so its
σ update is skipped and volatility is left entirely to the SV block (SPEC §13 Phase 4, question 3).

**The adapter** (:func:`set_data`, the setter the memo proposes upstream) swaps the data into the
bartz ``State`` before every ``bartz.mcmcstep.step``: the binned covariates and the leaf indices
(re-traversed, because Geweke's successive-conditional simulator redraws ``y`` and hence the lags
every sweep), the residuals ``target − forest(X)`` recomputed from the trees, the five
error-scale fields that ``init`` derives from ``error_scale`` (``inv_sdev_scale``, ``prec_scale``,
``inv_sdev_unit``, ``n_non_missing``, ``sum_diag_prec_scale``) and the per-node precision cache
``forest.prec_tree``, which ``step`` otherwise only updates at the nodes a move touches.

**float64 (SPEC R4, §8).** bartz 0.13.0 cannot trace ``step`` with ``jax_enable_x64`` on: under x64
``_propose_moves`` builds ``lrt_nodes`` as int64 and ``accept_move_and_sample_leaves`` asserts
int32. Two paths, chosen by :attr:`BartPrior.x64`:

* ``x64=False`` (default; unmodified bartz): the bartz state is built and stepped with x64
  switched off for the duration of bartz's trace, so **every** float in the bartz state is float32
  (residuals and precision scales as well as leaves; wider than the one exception SPEC §8 names).
  The adapter computes in float64 and rounds at the boundary; the fitted values handed back to the
  sweep are float64 sums of the float32 leaves.
* ``x64=True``: bartz traced under x64 with float64 residuals and precision scales and float32
  leaves (SPEC §8's stated exception). Needs two one-line fixes in bartz (the int32 cast and a
  cast before the scatter in ``mcmcstep/_reduction.py``); the tests apply them as a monkeypatch
  (``tests/phase4/bartz_x64_patch.py``). Without the patch ``step`` raises ``AssertionError``.

bartz is imported lazily (optional extra ``bart``); importing this module does not import it.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import MeanStatic, StaticConfig
from rai_bvar.engines.mcmc.keys import Key
from rai_bvar.engines.mcmc.protocols import Batching, KernelBase
from rai_bvar.engines.mcmc.state import (
    Coupling,
    Empty,
    EngineData,
    EqContext,
    FitCache,
    GaussianDiag,
    GaussianPrec,
    GlobalState,
    GlobDraw,
    PriorArrays,
    SharedPrior,
    StepInfo,
    _pytree,
)

Array = jax.Array
PyTree = Any

__all__ = [
    "BARTZ_VERSION",
    "BartCache",
    "BartPrior",
    "BartSpikeMean",
    "bin_lags",
    "forest_values",
    "node_sums",
    "set_data",
    "template_state",
]

BARTZ_VERSION = "0.13.0"
"""The bartz release the adapter was written against (pinned in the ``bart`` extra)."""

#: ``State`` fields the adapter writes (the upstream setter's contract; memo §Q1 churn table).
SET_FIELDS: tuple[str, ...] = (
    "X", "y", "resid", "resid_inexact_integral", "error_scale", "prec_scale", "inv_sdev_scale",
    "inv_sdev_unit", "n_non_missing", "sum_diag_prec_scale", "forest.leaf_indices",
    "forest.to_prune", "forest.prec_tree",
)  # fmt: skip

#: ``State`` fields the adapter reads but never writes.
READ_FIELDS: tuple[str, ...] = (
    "resid_unit", "forest.var_tree", "forest.split_tree", "forest.leaf_tree", "forest.leaf_unit",
    "forest.offset", "error_cov_inv",
)  # fmt: skip


def _default_cuts() -> tuple[float, ...]:
    return tuple(float(c) for c in np.linspace(-3.0, 3.0, 31))


@dataclass(frozen=True)
class BartPrior:
    """Static BART choices (hashable). ``max_depth`` is bartz's ``d``: heaps of ``2**d`` slots,
    nodes at depth ``d − 1`` are always leaves; ``P(node at depth δ splits | a rule is available) =
    alpha/(1 + δ)**beta`` (Chipman, George & McCulloch 2010). Leaves are ``N(0, (g_sd/√m)²)``, so
    ``g(x)`` has prior s.d. ``g_sd`` at every ``x``. ``cuts`` is the fixed cutpoint grid (at most
    255 cutpoints: ``uint8`` bins)."""

    num_trees: int = 200
    max_depth: int = 6
    alpha: float = 0.95
    beta: float = 2.0
    g_sd: float = 0.5
    cuts: tuple[float, ...] = dataclasses.field(default_factory=_default_cuts)
    x64: bool = False

    def __post_init__(self) -> None:
        if self.num_trees < 1 or self.max_depth < 2:
            raise ValueError("num_trees >= 1 and max_depth >= 2 are required")
        if not 1 <= len(self.cuts) <= 255 or list(self.cuts) != sorted(set(self.cuts)):
            raise ValueError("cuts must be 1..255 strictly increasing cutpoints")
        if not (0.0 < self.alpha < 1.0 and self.beta >= 0.0 and self.g_sd > 0.0):
            raise ValueError("need 0 < alpha < 1, beta >= 0 and g_sd > 0")

    @property
    def max_split(self) -> int:
        return len(self.cuts)

    @property
    def tree_size(self) -> int:
        return 2**self.max_depth

    @property
    def leaf_sd(self) -> float:
        return self.g_sd / math.sqrt(self.num_trees)

    def p_nonterminal(self) -> np.ndarray:
        """``alpha/(1 + δ)**beta`` for ``δ = 0..d−2`` (bartz's ``make_p_nonterminal``), float32."""
        depth = np.arange(self.max_depth - 1, dtype=np.float64)
        return (self.alpha / (1.0 + depth) ** self.beta).astype(np.float32)


@_pytree
class BartCache:
    """``FitCache.eq`` of the BART mean: the binned lag block, one copy per equation."""

    Xbin: Array  # u8[n, p_x, T_pad]


def bin_lags(X: Array, n_det: int, cuts: tuple[float, ...]) -> Array:
    """``Xbin[v, t] = #{c ∈ cuts : c ≤ x_{t, n_det + v}}`` (``u8[p_x, T_pad]``), so that
    ``Xbin < s ⇔ x < cuts[s − 1]``: bartz's left-child rule ``X < split`` on a fixed grid."""
    lags = X[:, n_det:]
    edges = jnp.asarray(cuts, jnp.float64)
    return jnp.searchsorted(edges, lags, side="right").astype(jnp.uint8).T


@functools.cache
def template_state(prior: BartPrior, p_x: int, T_pad: int) -> Any:
    """A root-only bartz ``State`` with the prior's static choices, its dtypes normalised to what
    ``step`` returns (so the state is a valid ``lax.scan`` carry). Must first be called outside any
    trace (``BartSpikeMean`` builds it on construction). Built by ``mcmcstep.init`` on
    placeholder data under the prior's x64 setting; the adapter replaces every data field.

    ``init`` rejects a prior-free error precision for continuous outcomes (``assert rate is not
    None``), although ``step`` skips the σ update exactly when ``error_cov_inv.nu is None``; the
    fixed precision 1 is therefore swapped in after ``init``."""
    from bartz import mcmcstep

    f_store = jnp.float64 if prior.x64 else jnp.float32
    with jax.enable_x64(prior.x64):
        st = mcmcstep.init(
            X=np.zeros((p_x, T_pad), np.uint8),
            y=np.zeros(T_pad, np.float32),
            offset=0.0,
            max_split=np.full(p_x, prior.max_split, np.uint8),
            num_trees=prior.num_trees,
            p_nonterminal=prior.p_nonterminal(),
            leaf_prior_cov_inv=mcmcstep.Wishart(None, None, value=1.0 / prior.leaf_sd**2),
            error_cov_inv=mcmcstep.Wishart(nu=1.0, rate=1.0, value=1.0),
            error_scale=np.ones(T_pad, np.float32),
            leaf_dtype=jnp.float32,
            prec_scale_dtype=f_store,
            resid_dtype=f_store,
        )
        st = dataclasses.replace(st, error_cov_inv=mcmcstep.Wishart(None, None, value=1.0))
        out = jax.eval_shape(mcmcstep.step, jax.random.key(0), st)
    # host (NumPy) leaves: every use builds fresh device arrays, so the donation of an eagerly
    # called ``step`` can never delete the cached template
    return jax.tree.map(lambda a, o: np.asarray(a, o.dtype), st, out)


def float_leaf_paths(prior: BartPrior, p_x: int, T_pad: int) -> frozenset[str]:
    """Dotted paths (under ``eq.mean``) of the template's float32 leaves: the exemptions from the
    float64 rule (``MeanBlock.float32_leaves``, ``state.assert_float64``)."""
    st = template_state(prior, p_x, T_pad)
    return frozenset(
        "eq.mean." + jax.tree_util.keystr(path, simple=True, separator=".")
        for path, leaf in jax.tree_util.tree_flatten_with_path(st)[0]
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != jnp.float64
    )


def _round_pow2(x: Array) -> Array:
    return 2.0 ** jnp.round(jnp.log2(x))


def traverse(X: Array, var_tree: Array, split_tree: Array, out_dtype: Any) -> Array:
    """Leaf index of every column of ``X`` in every tree (``u[m, T]``); bartz's rule: left iff
    ``X[var] < split``, a node is a leaf iff ``split == 0``."""
    depth = round(math.log2(var_tree.shape[-1])) + 1  # 2·half slots ⇒ d levels

    def one(var: Array, split: Array) -> Array:
        idx = jnp.ones(X.shape[1], jnp.int32)
        done = jnp.zeros(X.shape[1], bool)
        for _ in range(depth - 1):
            s = split[idx].astype(jnp.int32)
            v = var[idx].astype(jnp.int32)
            done = done | (s == 0)
            xv = jnp.take_along_axis(X, v[None, :], axis=0)[0].astype(jnp.int32)
            idx = jnp.where(done, idx, 2 * idx + (xv >= s).astype(jnp.int32))
        return idx.astype(out_dtype)

    return jax.vmap(one)(var_tree, split_tree)


def forest_values(forest: Any, leaf_indices: Array) -> Array:
    """``g_t = offset + leaf_unit · Σ_m leaf_m[idx_{m,t}]`` in float64 (``f64[T]``). Valid for the
    not-yet-pruned indices bartz keeps between steps (pruned nodes mirror their value onto the
    dangling children)."""
    leaves = jnp.take_along_axis(forest.leaf_tree, leaf_indices.astype(jnp.int32), axis=-1)
    total = jnp.sum(leaves.astype(jnp.float64), axis=0)
    return jnp.float64(forest.offset) + jnp.float64(forest.leaf_unit) * total


def node_sums(values: Array, leaf_indices: Array, tree_size: int) -> Array:
    """``f32[m, tree_size]``: per tree, the sum of ``values`` over the datapoints below each heap
    node (float32 accumulation, like bartz's ``prec_tree``). Valid at every node, so in particular
    at the leaves of the pre-move trees, which is what ``step`` requires."""
    v = values.astype(jnp.float32)
    s = jax.vmap(lambda ix: jnp.zeros(tree_size, jnp.float32).at[ix.astype(jnp.int32)].add(v))(
        leaf_indices
    )
    level = tree_size // 2
    while level > 1:  # parents of the nodes at heap level [level, 2·level)
        par = np.arange(level // 2, level)
        s = s.at[:, par].add(s[:, 2 * par] + s[:, 2 * par + 1])
        level //= 2
    return s


def set_data(st: Any, X: Array, target: Array, weights: Array) -> Any:
    """Swap one equation's data into a bartz ``State`` (the setter proposed upstream; memo §Q1).

    ``X``: ``u8[p_x, T]`` binned covariates; ``target``: ``f64[T]`` (bartz's ``y``); ``weights``:
    ``f64[T]`` per-observation precisions ``w_t = 1/λ_t`` (0 on padded rows, which then drop out
    like bartz's ``missing``). The error precision ``error_cov_inv.value`` (1 here) multiplies
    them, as in bartz (``precision_t = error_cov_inv / error_scale_t²``).

    Written fields (:data:`SET_FIELDS`): ``X``; ``forest.leaf_indices`` by re-traversal with
    ``forest.to_prune`` cleared (``split_tree`` already reflects a pending prune, so the fold at the
    start of the next ``step`` becomes a no-op); ``resid = (y − forest(X))/resid_unit`` with
    bartz's sanitised ``y`` (``target``, ``offset`` on zero-weight rows), computed from the trees
    (no running-update drift; ``resid_inexact_integral`` reset to 0); the error-scale fields as
    ``init``'s ``compute_scale_related_attrs`` derives them; ``prec_tree``
    rebuilt for every node; ``y`` and ``error_scale`` for consistency (the continuous ``step`` does
    not read them)."""
    f = st.forest
    idx = traverse(X, f.var_tree, f.split_tree, f.leaf_indices.dtype)
    g = forest_values(f, idx)
    live = weights > 0
    offset = jnp.float64(f.offset)
    y = jnp.where(live, target, offset)  # bartz's sanitised y: offset at missing rows
    resid = ((y - g) / jnp.float64(st.resid_unit)).astype(st.resid.dtype)
    inv_sd = jnp.sqrt(jnp.where(live, weights, 0.0))
    n_live = jnp.sum(live).astype(st.n_non_missing.dtype)
    sum_w = jnp.sum(jnp.where(live, weights, 0.0))
    unit = _round_pow2(jnp.sqrt(sum_w / jnp.maximum(n_live, 1)))
    unit = jnp.where(unit > 0, unit, 1.0)
    scaled = inv_sd / unit
    prec = (scaled * scaled).astype(st.prec_scale.dtype)
    forest = dataclasses.replace(
        f,
        leaf_indices=idx,
        to_prune=jnp.zeros_like(f.to_prune),
        prec_tree=node_sums(prec, idx, f.leaf_tree.shape[-1]),
    )
    return dataclasses.replace(
        st,
        X=X.astype(st.X.dtype),
        y=y.astype(st.y.dtype),
        resid=resid,
        resid_inexact_integral=jnp.zeros_like(st.resid_inexact_integral),
        error_scale=jnp.where(live, 1.0 / jnp.where(live, inv_sd, 1.0), 1.0).astype(
            st.error_scale.dtype
        ),
        prec_scale=prec,
        inv_sdev_scale=scaled.astype(st.inv_sdev_scale.dtype),
        inv_sdev_unit=unit.astype(st.inv_sdev_unit.dtype),
        n_non_missing=n_live,
        sum_diag_prec_scale=sum_w.astype(st.sum_diag_prec_scale.dtype),
        forest=forest,
    )


def bartz_step(key: Key, st: Any, x64: bool) -> Any:
    """``bartz.mcmcstep.step`` traced under the x64 setting the state was built with. Called
    eagerly (outside ``jit``), bartz donates ``st``: its arrays are deleted."""
    from bartz import mcmcstep

    with jax.enable_x64(x64):
        return mcmcstep.step(key, st)


@dataclass(frozen=True)
class BartSpikeMean(KernelBase):
    """``MeanBlock`` prototype: ``g_i`` is a bartz forest (module docstring). Non-fusing
    (``Recursive.contemp_step`` draws ``a_i | g_i``) and stateful (``eq.mean`` is the bartz
    ``State``, batched over equations by ``vmap``)."""

    static: MeanStatic
    n: int
    n_det: int
    lags: int
    T_pad: int
    prior: BartPrior = dataclasses.field(default_factory=BartPrior)

    fuses_contemp: ClassVar[bool] = False
    has_state: ClassVar[bool] = True
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()  # per instance: float_leaf_paths

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.static.kind != "zero":
            raise ValueError("the spike composes with a MeanStatic('zero') placeholder")
        template_state(self.prior, self.p_x, self.T_pad)  # eager: never first built in a trace

    @classmethod
    def for_static(cls, static: StaticConfig, prior: BartPrior) -> BartSpikeMean:
        s = static.shapes
        return cls(static.mean, n=s.n, n_det=s.n_det, lags=s.lags, T_pad=s.T_pad, prior=prior)

    @property
    def p_x(self) -> int:
        return self.n * self.lags

    def template(self) -> Any:
        return template_state(self.prior, self.p_x, self.T_pad)

    def float32_paths(self) -> frozenset[str]:
        return float_leaf_paths(self.prior, self.p_x, self.T_pad)

    # ----------------------------------------------------------------- fit-level
    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache:
        """``cache.eq = BartCache(Xbin)``: the binned lags, one copy per equation."""
        Xbin = bin_lags(data.X, self.n_det, self.prior.cuts)
        Xbin = jnp.where(data.tmask[None, :] > 0, Xbin, 0).astype(jnp.uint8)
        return dataclasses.replace(
            base, eq=BartCache(Xbin=jnp.broadcast_to(Xbin, (self.n, *Xbin.shape)))
        )

    def fitted_reduced(self, glob: GlobalState, data: EngineData) -> None:
        return None

    # ----------------------------------------------------------------- per equation
    def init_eq(
        self,
        key: Key,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[Any, Array]:
        """Root-only trees with zero leaves, data swapped in."""
        st = jax.tree.map(jnp.asarray, self.template())
        return set_data(st, cache_i.Xbin, ctx.target, ctx.weights), a_i

    def sample_eq(
        self,
        key: Key,
        ms: Any,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[Any, Array, Coupling, StepInfo]:
        """One bartz sweep over the trees given ``a_i`` (``ctx.target`` already has ``Z a_i``
        subtracted, ``sweep.mean_context``) and the weights ``m_t/λ_eff,t``; ``a_i`` unchanged."""
        st = set_data(ms, cache_i.Xbin, ctx.target, ctx.weights)
        st = bartz_step(key, st, self.prior.x64)
        ok = jnp.isfinite(jnp.sum(st.resid)) & jnp.isfinite(jnp.sum(st.forest.leaf_tree))
        return st, a_i, Coupling.zeros(), StepInfo(ok, jnp.full((), jnp.inf))

    def conditional_eq(self, *args: Any, **kwargs: Any) -> GaussianPrec:
        """No Gaussian conditional (tree moves are Metropolis–Hastings): the empty Gaussian."""
        return GaussianPrec(K=jnp.zeros((0, 0), jnp.float64), b=jnp.zeros((0,), jnp.float64))

    def fitted_eq(
        self, ms: Any, a_i: Array, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> Array:
        """``g_i(x_t)`` (``f64[T_pad]``, 0 on padded rows) from the leaf indices of ``ms``: valid
        after :meth:`sample_eq` on the same data, which the plan guarantees (the mean step runs
        first in every sweep)."""
        return jnp.where(ctx.tmask > 0, forest_values(ms.forest, ms.forest.leaf_indices), 0.0)

    def coefficients_eq(self, ms: Any, a_i: Array, shared: SharedPrior) -> Array:
        return jnp.zeros((0,), jnp.float64)

    def log_prior_eq(self, *args: Any, **kwargs: Any) -> Array:
        """Not used by the spike (the validation reference computes the tree prior)."""
        return jnp.zeros((), jnp.float64)

    def extract_eq(self, ms: Any, a_i: Array, shared: SharedPrior) -> Empty:
        return Empty()

    def extract_global(self, glob: GlobalState) -> GlobDraw:
        return GlobDraw(mean=Empty(), coef=Empty(), vol=Empty(), struct=Empty())

    def future_eq(self, key: Key, md: PyTree, H: int) -> Empty:
        return Empty()

    def predict_eq(self, md: PyTree, fut: PyTree, x: Array, h: Array, gfut: PyTree) -> Array:
        raise NotImplementedError("forecasting is outside the Phase 4 week-1 spike")

    def predict_reduced(self, gd: GlobDraw, x: Array, h: Array, gfut: PyTree) -> Array:
        return jnp.zeros((self.n,), jnp.float64)
