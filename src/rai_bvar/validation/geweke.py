"""Geweke (2004) joint-distribution tests with parallel successive-conditional chains
(N06 §3; N07 §13.2; DESIGN §9.2, §9.5).

**Simulators** (Geweke 2004 eqs 3–5). Marginal-conditional (MC), iid::

    θ^(m) ~ p(θ),  y^(m) ~ p(y | θ^(m)),  g^(m) = g(θ^(m), y^(m)),          m = 1..M₁
    ḡ_MC = M₁⁻¹ Σ g^(m),   σ̂²_g = sample variance of g^(m)

Successive-conditional (SC): ``C`` independent chains, each started at a prior draw and run with
**no burn-in** (under H0 every chain is stationary from its first sweep; N06 §3.1)::

    θ^(0) ~ p(θ);  for m = 1..M_c:  y^(m) ~ p(y | θ^(m−1)),  θ^(m) ~ q(θ | θ^(m−1), y^(m)),
                                     g^(m) = g(θ^(m), y^(m))

**Statistic** (N06 §3.2, D-2). With chain means ``ḡ_c``, ``ḡ_SC = C⁻¹Σ_c ḡ_c`` and
``s²_C = (C−1)⁻¹Σ_c (ḡ_c − ḡ_SC)²``::

    z_g = (ḡ_MC − ḡ_SC) / √(σ̂²_g/M₁ + s²_C/C),   p_g = 2·P(t_{C−1} > |z_g|)

valid for any ``M_c ≥ 1`` and needing only invariance, not ergodicity (N06 §6 item 14). The test
passes iff no ``p_g ≤ α/G`` (Bonferroni, ``t_{C−1}`` critical value ``t_crit``); Holm lists the
failing functions (N06 §3.5). ``RNE_g = σ̂²_g/τ̂²_g`` with ``τ̂²_g = M_c·s²_C``. Batch means within
chains (``b = ⌊√M_c⌋``, ``τ̂² = b·Var(batch means)``) are a non-gating cross-check (N06 §3.2 item 2).
Every test function must have finite variance under ``p(θ, y)`` (Geweke eq. 2; N06 §3.4).

**Keys** (DESIGN §5.6). ``validation_key(seed, Tier.G, composition, r)`` with ``r`` the chain index
(SC) or the draw index (MC), then a :class:`Part` sub-stream; the composition id is the CRC-32 of
``model.name`` (0 without a name). Each chain's draws depend only on ``(seed, composition, c)``, so
results do not depend on the block sizes used to vectorise.

A non-finite test-function value fails that function (``p = 0``) and is counted in
``n_nonfinite``: the model signals sampler failures by returning NaN (no silent dropping; SPEC §8).
"""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from scipy import stats

from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc.keys import Tier, validation_key
from rai_bvar.validation.sequential import (
    SequentialResult,
    bonferroni,
    gandy_scott,
    holm_reject,
    stage_multiplier,
    stage_seed,
)

__all__ = [
    "GewekeConfig",
    "GewekeResult",
    "JointModel",
    "Part",
    "composition_id",
    "geweke",
    "geweke_sequential",
    "rep_keys",
]

PyTree = Any


@runtime_checkable
class JointModel(Protocol):
    """The joint model ``p(θ, y)`` under test and one sweep of its posterior sampler (§9.2).

    - ``prior_sample(key)``: ``θ ~ p(θ)``, including latent states (never auxiliary indicators).
    - ``simulate(key, state)``: ``y ~ p(y | θ)`` using the state's CURRENT latents (Geweke error 2).
    - ``step(key, it, state, data)``: one sweep ``θ′ ~ q(θ | θ, y)``. ``key`` is constant along a
      chain and ``it`` is the uint32 absolute sweep index, which ``step`` folds in (as the
      production sweep does: ``fold_in(ckey, it)``; no key is carried, D-16).
    - ``test_functions(state, data)``: ``f64[G]`` with ``G = len(names)``, finite variance under
      ``p(θ, y)``; NaN signals a sampler failure.

    All four are pure JAX functions (traced under ``jit``/``vmap``); results are float64.
    """

    names: tuple[str, ...]

    def prior_sample(self, key) -> PyTree: ...

    def simulate(self, key, state) -> PyTree: ...

    def step(self, key, it, state, data) -> PyTree: ...

    def test_functions(self, state, data) -> jax.Array: ...


class Part(IntEnum):
    """Sub-streams below one validation key (append-only: renumbering changes every stream)."""

    PRIOR = 0  # Geweke SC chain start θ⁽⁰⁾; SBC and block-SBC truth θ̃
    SIM = 1  # Geweke SC simulator (then the sweep index); SBC data ỹ
    STEP = 2  # Geweke SC chain key handed to step
    MC_PRIOR = 3  # Geweke MC prior draw
    MC_SIM = 4  # Geweke MC data
    PILOT_INIT = 5  # SBC pilot chain: independent prior draw (never θ̃)
    PILOT_STEP = 6  # SBC pilot chain key handed to step
    CHAIN_INIT = 7  # SBC production chain: fresh independent prior draw
    CHAIN_STEP = 8  # SBC production chain key handed to step
    TIES = 9  # one uniform per quantity for rank tie-breaking
    DRAWS = 10  # block SBC conditional draws (then the draw index)


def composition_id(model: object) -> int:
    """CRC-32 of ``model.name`` (0 when the model has no string name)."""
    name = getattr(model, "name", "")
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF if isinstance(name, str) else 0


def rep_keys(seed: int, tier: int, composition: int, reps) -> jax.Array:
    """``validation_key(seed, tier, composition, r)`` for every index ``r`` in ``reps``."""
    reps = np.asarray(reps, dtype=np.int64).ravel()
    if reps.size and (reps.min() < 0 or reps.max() >= 2**32):
        raise ValueError("replication indices must lie in [0, 2**32)")
    r = jnp.asarray(reps.astype(np.uint32))
    return jax.vmap(lambda i: validation_key(seed, int(tier), composition, i))(r)


def part(key: jax.Array, p: Part) -> jax.Array:
    """``fold_in(key, p)``."""
    return random.fold_in(key, int(p))


# --- compiled-function cache (hashable models only; unhashable ones are re-traced per call) ----

_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 32


def compiled(model: object, tag: tuple, build: Callable[[], Callable]) -> Callable:
    """``build()`` memoised on ``(tag, model)`` when the model is hashable."""
    try:
        key = (tag, model)
        hash(key)
    except TypeError:
        return build()
    fn = _CACHE.get(key)
    if fn is None:
        fn = build()
        _CACHE[key] = fn
        if len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return fn


def check_output(fn: Callable, key: jax.Array, G: int, what: str) -> None:
    """Abstract-evaluate ``fn(key)`` and require ``f64[G]`` (R4: float64 everywhere)."""
    out = jax.eval_shape(fn, key)
    if tuple(out.shape) != (G,):
        raise ValueError(f"{what} must return shape ({G},) to match names, got {tuple(out.shape)}")
    if out.dtype != jnp.float64:
        raise TypeError(f"{what} must return float64, got {out.dtype} (SPEC §8)")


# --- configuration and result ------------------------------------------------------------------


def _positive_int(value: object, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer ≥ {minimum}, got {value!r}")


@dataclass(frozen=True)
class GewekeConfig:
    """Sizes of one Geweke test (DESIGN §9.5): G-small ``C = 1000, M_c = 100, M1 = 2·10⁵,
    α = 1e-3``; G-full ``C = 4000, M_c = 500, M1 = 10⁶, α = 1e-2``. ``chain_block`` and
    ``draw_block`` only bound memory (results do not depend on them)."""

    C: int
    M_c: int
    M1: int
    alpha: float
    seed: int
    batch_means: bool = True
    chain_block: int = 1000
    draw_block: int = 10000

    def __post_init__(self) -> None:
        _positive_int(self.C, "C", 2)
        _positive_int(self.M_c, "M_c")
        _positive_int(self.M1, "M1", 2)
        _positive_int(self.chain_block, "chain_block")
        _positive_int(self.draw_block, "draw_block")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int | np.integer):
            raise TypeError(f"seed must be an integer, got {self.seed!r}")
        if self.C > 2**32 or self.M1 > 2**32 or self.M_c >= 2**32:
            raise ValueError("C, M1 and M_c must fit the uint32 counters (DESIGN §5.6)")


@dataclass(frozen=True)
class GewekeResult:
    """Per test function ``g``: ``z``, two-sided ``t_{C−1}`` p-value ``p``, ``rne``; the test
    passes iff ``failing`` (Holm at ``alpha``, equivalently Bonferroni) is empty."""

    names: tuple[str, ...]
    z: np.ndarray
    p: np.ndarray
    t_crit: float
    passed: bool
    rne: np.ndarray
    failing: tuple[str, ...]
    alpha: float = 0.0
    C: int = 0
    M_c: int = 0
    M1: int = 0
    mean_mc: np.ndarray | None = None
    mean_sc: np.ndarray | None = None
    var_mc: np.ndarray | None = None
    var_chain_means: np.ndarray | None = None
    n_nonfinite: np.ndarray | None = None
    z_batch: np.ndarray | None = None
    p_batch: np.ndarray | None = None
    rne_batch: np.ndarray | None = None

    @property
    def p_bonferroni(self) -> float:
        """``min(1, G·min p)``: the stage p-value for :func:`~rai_bvar.validation.sequential.gandy_scott`."""
        return bonferroni(self.p)

    def table(self):
        """A ``pandas.DataFrame`` of the per-function statistics, largest ``|z|`` first."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "z": self.z,
                "p": self.p,
                "rne": self.rne,
                "mean_mc": self.mean_mc,
                "mean_sc": self.mean_sc,
                "z_batch": self.z_batch,
                "n_nonfinite": self.n_nonfinite,
                "failing": [n in self.failing for n in self.names],
            },
            index=pd.Index(self.names, name="g"),
        )
        return frame.reindex(frame["z"].abs().sort_values(ascending=False).index)


# --- the two simulators ------------------------------------------------------------------------


def _mc_block_fn(model: JointModel) -> Callable:
    def one(key):
        th = model.prior_sample(part(key, Part.MC_PRIOR))
        y = model.simulate(part(key, Part.MC_SIM), th)
        return model.test_functions(th, y)

    def block(keys, valid):
        g = jax.vmap(one)(keys)  # (B, G)
        ok = jnp.isfinite(g) & valid[:, None]
        n = jnp.sum(ok, axis=0)
        mean = jnp.sum(jnp.where(ok, g, 0.0), axis=0) / jnp.maximum(n, 1)
        m2 = jnp.sum(jnp.where(ok, (g - mean) ** 2, 0.0), axis=0)
        bad = jnp.sum(~jnp.isfinite(g) & valid[:, None], axis=0)
        return n, mean, m2, bad

    return jax.jit(block)


def _sc_block_fn(model: JointModel, M_c: int, batch: int) -> Callable:
    """Chain statistics: mean of ``g``, non-finite count, and (``batch > 0``) the sum and sum of
    squares of the within-chain batch means of ``g − shift`` over the last ``⌊M_c/b⌋·b`` sweeps."""
    n_batches = M_c // batch if batch else 0
    offset = M_c - n_batches * batch

    def chain(key, shift):
        th0 = model.prior_sample(part(key, Part.PRIOR))
        sim_key, step_key = part(key, Part.SIM), part(key, Part.STEP)
        zeros = jnp.zeros_like(shift)

        def body(carry, it):
            th, total, bad, cur, bsum, bsq = carry
            y = model.simulate(random.fold_in(sim_key, it), th)
            th = model.step(step_key, it, th, y)
            g = model.test_functions(th, y)
            ok = jnp.isfinite(g)
            d = jnp.where(ok, g - shift, 0.0)
            total = total + d
            bad = bad + (~ok).astype(jnp.int32)
            if batch:
                pos = it.astype(jnp.int32) - offset  # position inside the batched tail
                cur = cur + jnp.where(pos >= 0, d, 0.0)
                close = (pos >= 0) & ((pos + 1) % batch == 0)
                bmean = cur / batch
                bsum = bsum + jnp.where(close, bmean, 0.0)
                bsq = bsq + jnp.where(close, bmean**2, 0.0)
                cur = jnp.where(close, 0.0, cur)
            return (th, total, bad, cur, bsum, bsq), None

        init = (th0, zeros, jnp.zeros(shift.shape, jnp.int32), zeros, zeros, zeros)
        (_, total, bad, _, bsum, bsq), _ = lax.scan(body, init, jnp.arange(M_c, dtype=jnp.uint32))
        return total / M_c + shift, bad, bsum, bsq

    return jax.jit(jax.vmap(chain, in_axes=(0, None)))


def _merge(n_a, mean_a, m2_a, n_b, mean_b, m2_b):
    """Chan et al. pairwise merge of (count, mean, sum of squared deviations)."""
    n = n_a + n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = mean_b - mean_a
        mean = np.where(n > 0, mean_a + delta * np.where(n > 0, n_b / n, 0.0), 0.0)
        m2 = m2_a + m2_b + np.where(n > 0, delta**2 * n_a * n_b / np.where(n > 0, n, 1), 0.0)
    return n, mean, m2


def _zstat(diff: np.ndarray, var: np.ndarray) -> np.ndarray:
    """``diff/√var`` with ``0/0 = 0`` (a constant function carries no evidence) and ``x/0 = ±∞``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = diff / np.sqrt(var)
        z = np.where((var == 0) & (diff != 0), np.sign(diff) * np.inf, z)
    return np.where((var == 0) & (diff == 0), 0.0, z)


def geweke(model: JointModel, cfg: GewekeConfig) -> GewekeResult:
    """Run the MC and parallel SC simulators of ``model`` and test every function in ``names``."""
    names = tuple(model.names)
    G = len(names)
    if G == 0:
        raise ValueError("model.names is empty")
    comp = composition_id(model)
    with _jax.x64():
        probe = rep_keys(cfg.seed, Tier.G, comp, [0])[0]

        def mc_one(k):
            th = model.prior_sample(part(k, Part.MC_PRIOR))
            return model.test_functions(th, model.simulate(part(k, Part.MC_SIM), th))

        check_output(mc_one, probe, G, "test_functions")

        # MC: iid draws, accumulated block by block (Chan merge in float64)
        mc_block = compiled(model, ("geweke_mc",), lambda: _mc_block_fn(model))
        B = min(cfg.draw_block, cfg.M1)
        n = np.zeros(G)
        mean = np.zeros(G)
        m2 = np.zeros(G)
        bad_mc = np.zeros(G, dtype=np.int64)
        for start in range(0, cfg.M1, B):
            keys = rep_keys(cfg.seed, Tier.G, comp, np.arange(start, start + B))
            valid = jnp.arange(start, start + B) < cfg.M1
            nb, mb, m2b, badb = (np.asarray(a) for a in mc_block(keys, valid))
            n, mean, m2 = _merge(n, mean, m2, nb.astype(np.float64), mb, m2b)
            bad_mc += badb.astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_mc = np.where(n > 1, m2 / np.where(n > 1, n - 1, 1), np.nan)

        # SC: C chains of M_c sweeps, no burn-in
        b = int(np.floor(np.sqrt(cfg.M_c))) if cfg.batch_means else 0
        sc_block = compiled(
            model, ("geweke_sc", cfg.M_c, b), lambda: _sc_block_fn(model, cfg.M_c, b)
        )
        shift = jnp.asarray(np.where(np.isfinite(mean), mean, 0.0))
        CB = min(cfg.chain_block, cfg.C)
        chain_means, bad_sc, bsum, bsq = [], np.zeros(G, dtype=np.int64), np.zeros(G), np.zeros(G)
        for start in range(0, cfg.C, CB):
            keys = rep_keys(cfg.seed, Tier.G, comp, np.arange(start, start + CB))
            cm, cb, bs, bq = (np.asarray(a) for a in sc_block(keys, shift))
            keep = min(CB, cfg.C - start)
            chain_means.append(cm[:keep])
            bad_sc += cb[:keep].sum(axis=0)
            bsum += bs[:keep].sum(axis=0)
            bsq += bq[:keep].sum(axis=0)
        chain_means = np.concatenate(chain_means, axis=0)  # (C, G)

    mean_sc = chain_means.mean(axis=0)
    var_chain = chain_means.var(axis=0, ddof=1)
    n_bad = bad_mc + bad_sc
    z = _zstat(mean - mean_sc, var_mc / cfg.M1 + var_chain / cfg.C)
    p = 2.0 * stats.t.sf(np.abs(z), cfg.C - 1)
    broken = (n_bad > 0) | ~np.isfinite(z)
    p = np.where(broken, 0.0, p)
    t_crit = float(stats.t.isf(cfg.alpha / (2 * G), cfg.C - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rne = var_mc / (cfg.M_c * var_chain)

    z_b = p_b = rne_b = None
    n_batches_total = cfg.C * (cfg.M_c // b) if b else 0
    if n_batches_total >= 2:
        bm_mean = bsum / n_batches_total
        bm_var = np.maximum(bsq - n_batches_total * bm_mean**2, 0.0) / (n_batches_total - 1)
        tau2 = b * bm_var
        z_b = _zstat(mean - mean_sc, var_mc / cfg.M1 + tau2 / (cfg.C * cfg.M_c))
        p_b = np.where(broken, 0.0, 2.0 * stats.norm.sf(np.abs(z_b)))
        with np.errstate(divide="ignore", invalid="ignore"):
            rne_b = var_mc / tau2

    reject = holm_reject(p, cfg.alpha)
    failing = tuple(name for name, r in zip(names, reject, strict=True) if r)
    return GewekeResult(
        names=names,
        z=z,
        p=p,
        t_crit=t_crit,
        passed=not failing,
        rne=rne,
        failing=failing,
        alpha=float(cfg.alpha),
        C=int(cfg.C),
        M_c=int(cfg.M_c),
        M1=int(cfg.M1),
        mean_mc=mean,
        mean_sc=mean_sc,
        var_mc=var_mc,
        var_chain_means=var_chain,
        n_nonfinite=n_bad,
        z_batch=z_b,
        p_batch=p_b,
        rne_batch=rne_b,
    )


def geweke_sequential(
    model: JointModel, cfg: GewekeConfig, k: int = 3, delta: int = 4
) -> tuple[SequentialResult, list[GewekeResult]]:
    """:func:`geweke` inside the Gandy–Scott wrapper at family level ``cfg.alpha`` (DESIGN §9.5):
    stage ``s`` reruns with :func:`~rai_bvar.validation.sequential.stage_seed` and ``C``, ``M1``
    multiplied by ``Δ`` after stage 1 (``M_c`` unchanged), so a marginal p-value triggers a fresh
    larger run instead of a red build."""
    results: list[GewekeResult] = []

    def run(stage: int) -> float:
        m = stage_multiplier(stage, delta)
        stage_cfg = replace(cfg, seed=stage_seed(cfg.seed, stage), C=cfg.C * m, M1=cfg.M1 * m)
        results.append(geweke(model, stage_cfg))
        return results[-1].p_bonferroni

    return gandy_scott(run, cfg.alpha, k, delta), results
