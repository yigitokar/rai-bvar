"""Simulation-based calibration: full-model SBC with per-replication thinning and block-conditional
SBC (Talts et al. 2018; Modrák et al. 2025; N06 §4, §5.2; N07 §13.3; DESIGN §9.4, §9.6).

**Full SBC** (:func:`sbc`; Talts et al. Algorithm 1 with the clean two-pass version of their
Algorithm 2, N06 §4.2). For replication ``i``:

1. ``θ̃ ~ p(θ)``, ``ỹ ~ p(y | θ̃)``. Data-only filter ``finite(ỹ) ∧ max|ỹ| < max_abs_data`` on every
   floating leaf of the data (valid: ranks are uniform conditionally on ``y``; Modrák et al. eq. 3
   and the text after Theorem 5). Filtering never looks at ``θ̃`` or at sampler output.
2. **Pilot**: a chain started at an independent prior draw (never at ``θ̃``: Gandy & Scott 2021
   §2.2), ``burn`` sweeps, then ``pilot`` recorded sweeps. ``IAT_i = max_f`` of the Geyer
   initial-monotone IAT (Vehtari et al. 2021 §3.2 eq. 13) of ``f`` and of the tail indicators
   ``1{f ≤ q̂_0.05}``, ``1{f ≤ q̂_0.95}`` (tail-ESS, §4.3). Thinning ``t_i = min(⌈iat_factor·IAT_i⌉,
   t_max)``; it depends on ``(ỹ, pilot noise)`` only, so uniformity is preserved.
3. **Production**: a FRESH chain (new independent prior start and key), ``burn`` sweeps, then ``L``
   draws kept every ``t_i`` sweeps (``thin`` is traced; replications are sorted by ``t_i`` before
   blocking, which bounds the idle work of the vectorised ``while`` loops).
4. ``rank = n_less + ⌊u(n_tie + 1)⌋`` per quantity (one ``u`` per quantity).

**Decision** (DESIGN §9.6). The test passes iff all three hold:

1. *γ test*: p-values from the exact Säilynoja recursion, Bonferroni over the ``Q`` quantities at
   ``alpha`` (fail iff ``min p ≤ alpha/Q``; Holm lists the failing quantities);
2. *no failed replication*: any replication with a non-finite quantity (pilot, truth or kept
   draws) fails the test; it is reported, never dropped;
3. *the ESS rule*, the single rule of DESIGN §9.6: the per-replication thinned ESS reaches
   ``min_ess·L`` (0.8 L) for at least ``ess_coverage`` (99%) of the replications.

The per-replication thinned ESS is the ESS of replication ``i``'s production chain thinned by
``t_i``, computed from that replication's pilot IAT with the AR(1) form
``ρ_i = (IAT_i − 1)/(IAT_i + 1)``, ``ESS_i = L(1 − ρ_i^{t_i})/(1 + ρ_i^{t_i})`` (``ess_thin``).
Uncapped, ``t_i = ⌈2·IAT_i⌉`` gives ``ESS_i ≥ 0.96L``, so only replications whose thinning hit
``t_max`` can fall short: ``ESS_i ≥ 0.8L`` iff ``ρ_i^{t_i} ≤ 1/9``, i.e. at ``t_max = 50`` iff
``IAT_i ≤ 45.5`` (the remedy is a larger ``t_max``). The rule takes the IAT from the pilot because
``L = 99`` kept draws cannot certify ``0.8L`` for one replication: with iid draws the Geyer ESS of
the kept draws reaches ``0.8L`` in only ~76% of replications for one quantity, so gating on it
would fail every correct sampler. For the same reason the rule cannot see a pilot that
underestimates the IAT (a pilot much shorter than the IAT) unless the underestimate still exceeds
the cap's limit (45.5 at ``t_max = 50``).

Two kept-draw ESS values are reported. ``ess_kept``, the per-replication Geyer ESS of the kept
draws (minimum over quantities), is a diagnostic only. ``ess_pooled`` is the **second gate**
(orchestrator erratum to DESIGN §9.6, 2026-10-22): per
quantity ``L/max_a τ̄_a``, where ``τ̄_a`` is the Geyer IAT of the autocovariances, pooled over
replications, of the ECDF indicators ``1{f ≤ q̂_{i,a}}`` at each replication's own empirical
quantiles, ``a ∈`` :data:`KEPT_LEVELS` (Talts et al. §5.1: the rank is a CDF statistic). For
continuous ``f`` every replication's indicator has variance ``≈ a(1 − a)``, so replications weigh
equally (N06 §4.2 rejects the pooled lag-1 autocorrelation of ``f`` itself, which a few
large-variance replications dominate); a degenerate discrete sequence weighs ``≈ 0``. A level is
used only when Kish's effective number of replications ``(Σγ̂_{i,0})²/Σγ̂_{i,0}²`` reaches
``pooled_min_reps`` (``ess_pooled`` is NaN when no level qualifies). With iid draws, ``n = 40``
and ``L = 99``, the minimum over 2·10⁴ simulations was ``0.836L``, so ``ess_pooled < min_ess·L``
for any quantity fails the run: it points to a pilot that underestimates the IAT, the blind spot of
the per-replication rule; raise ``pilot`` (or ``burn``, ``iat_factor``).

**Block-conditional SBC** (:func:`block_sbc`; N06 §5.2): ``(θ̃, ỹ)`` from the joint, ``L`` iid draws
of the block from its exact conditional given ``θ̃_−b``, ranks of the block statistics; the same
decision without the ESS rule (the draws are iid).

**Sequential wrapper** (:func:`sbc_sequential`, :func:`block_sbc_sequential`; DESIGN §9.4, §9.6):
the stage value is :attr:`SBCResult.p_bonferroni`, which is ``0`` whenever the run failed for a
reason other than the γ test (failed replications are not a marginal result).

**Keys**: ``validation_key(seed, Tier.S | Tier.B, composition, i)`` then :class:`Part` sub-streams;
sweep counters are uint32 (``burn + j·t + r``, DESIGN §5.6).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc.keys import Tier
from rai_bvar.validation.geweke import (
    JointModel,
    Part,
    check_output,
    compiled,
    composition_id,
    part,
    rep_keys,
)
from rai_bvar.validation.sequential import (
    SequentialResult,
    bonferroni,
    gandy_scott,
    holm_reject,
    stage_multiplier,
    stage_seed,
)
from rai_bvar.validation.uniformity import gamma_test, rank_with_ties

__all__ = [
    "SBCConfig",
    "SBCResult",
    "block_sbc",
    "block_sbc_sequential",
    "iat",
    "sbc",
    "sbc_sequential",
]

PyTree = Any


def _int_at_least(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer ≥ {minimum}, got {value!r}")


@dataclass(frozen=True)
class SBCConfig:
    """SBC sizes and rules (DESIGN §9.6 defaults). ``rep_block`` only bounds memory; ``min_ess``
    and ``ess_coverage`` set the ESS rule; ``pooled_min_reps`` only sets the non-gating
    ``ess_pooled`` diagnostic (module doc)."""

    N: int = 1000
    L: int = 99
    burn: int = 300
    pilot: int = 1000
    t_max: int = 50
    alpha: float = 1e-2
    seed: int = 0
    iat_factor: float = 2.0
    max_abs_data: float | None = 1e3
    min_ess: float = 0.8
    ess_coverage: float = 0.99
    rep_block: int = 100
    pooled_min_reps: int = 40

    def __post_init__(self) -> None:
        _int_at_least(self.N, "N", 1)
        _int_at_least(self.L, "L", 1)
        _int_at_least(self.burn, "burn", 0)
        _int_at_least(self.pilot, "pilot", 4)
        _int_at_least(self.t_max, "t_max", 1)
        _int_at_least(self.rep_block, "rep_block", 1)
        _int_at_least(self.pooled_min_reps, "pooled_min_reps", 1)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int | np.integer):
            raise TypeError(f"seed must be an integer, got {self.seed!r}")
        if not self.iat_factor > 0.0:
            raise ValueError("iat_factor must be positive")
        if self.max_abs_data is not None and not self.max_abs_data > 0.0:
            raise ValueError("max_abs_data must be positive or None")
        if not 0.0 <= self.min_ess <= 1.0 or not 0.0 <= self.ess_coverage <= 1.0:
            raise ValueError("min_ess and ess_coverage are fractions in [0, 1]")
        if self.N > 2**32 or max(self.burn + self.pilot, self.burn + self.L * self.t_max) >= 2**32:
            raise ValueError("replication and sweep counters must fit uint32 (DESIGN §5.6)")


@dataclass(frozen=True)
class SBCResult:
    """Ranks and the γ uniformity test per quantity. ``passed`` combines the Bonferroni γ test,
    the absence of failed replications and (full SBC) the ESS rule (module doc); ``reasons``
    says why a test failed. ``checks_ok`` is True when everything but the γ test held (no failed
    replication, at least one replication kept, the ESS rule met): only then is the result a
    γ-test p-value, and ``passed == checks_ok and not failing == (p_bonferroni > alpha)``."""

    names: tuple[str, ...]
    ranks: np.ndarray  # (N_kept, Q) in 0..L
    L: int
    gamma: np.ndarray  # (Q,)
    p: np.ndarray  # (Q,) exact γ p-values
    alpha: float
    passed: bool
    failing: tuple[str, ...]  # Holm at alpha
    reasons: tuple[str, ...] = ()
    n_total: int = 0  # replications drawn (before the data-only filter)
    kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    failed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    thin: np.ndarray | None = None  # (N_kept,) per-replication thinning
    iat: np.ndarray | None = None  # (N_kept,) pilot IAT (max over quantities and tails)
    ess_thin: np.ndarray | None = None  # (N_kept,) per-replication thinned ESS (the ESS rule)
    ess_ok: float = 1.0  # fraction of replications with ess_thin ≥ min_ess·L
    ess_kept: np.ndarray | None = None  # (N_kept,) empirical min-over-quantities ESS (diagnostic)
    ess_pooled: np.ndarray | None = None  # (Q,) pooled kept-draw ESS (diagnostic)
    checks_ok: bool = False  # all non-γ checks held (fail-safe default)

    @property
    def N(self) -> int:
        return int(self.ranks.shape[0])

    @property
    def p_bonferroni(self) -> float:
        """The stage value for :func:`~rai_bvar.validation.sequential.gandy_scott`: the
        Bonferroni-adjusted γ p-value ``min(1, Q·min p)`` when ``checks_ok``, otherwise ``0.0``
        (failed replications, an empty data-only filter or a failed ESS rule fail the test; they
        are never a marginal result that a larger rerun could overturn). ``bonferroni(self.p)``
        gives the γ value alone."""
        return bonferroni(self.p) if self.checks_ok else 0.0

    def table(self):
        """A ``pandas.DataFrame`` of γ and p per quantity, smallest p first."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "gamma": self.gamma,
                "p": self.p,
                "failing": [n in self.failing for n in self.names],
            },
            index=pd.Index(self.names, name="quantity"),
        )
        return frame.sort_values("p")


# --- IAT (Geyer initial monotone sequence, single chain) ----------------------------------------


def _acov(x: np.ndarray) -> np.ndarray:
    """Sample autocovariances ``γ̂_k = n⁻¹Σ_{t<n−k} x̃_t x̃_{t+k}`` (``x̃`` centred; FFT) along the
    last axis."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(xc, n=size, axis=-1)
    return np.fft.irfft(spec * np.conj(spec), n=size, axis=-1)[..., :n] / n


def _geyer_tau(acov: np.ndarray) -> np.ndarray:
    """Geyer's initial monotone sequence estimate from autocovariances ``acov[..., k]``
    (``k = 0..n−1``): with ``ρ_k = γ_k/γ_0``, ``τ = −1 + 2Σ_{k<K} P̂_k``,
    ``P̂_k = min_{j≤k}(ρ_{2j} + ρ_{2j+1})``, summed up to the first non-positive pair (Vehtari et
    al. 2021 §3.2 eq. 13), floored at 1; ``τ = 1`` when ``γ_0 = 0`` (a constant sequence)."""
    var = acov[..., 0]
    rho = acov / np.where(var > 0, var, 1.0)[..., None]
    K = rho.shape[-1] // 2
    pairs = rho[..., 0 : 2 * K : 2] + rho[..., 1 : 2 * K : 2]
    positive = pairs > 0
    stop = np.where(positive.all(axis=-1), K, np.argmin(positive, axis=-1))
    mono = np.minimum.accumulate(pairs, axis=-1)
    keep = np.arange(K) < stop[..., None]
    tau = -1.0 + 2.0 * np.sum(np.where(keep, mono, 0.0), axis=-1)
    return np.where(var > 0, np.maximum(tau, 1.0), 1.0)


def iat(x: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation time along the last axis: :func:`_geyer_tau` of the sample
    autocovariances (Vehtari et al. 2021 §3.2 eq. 13). Floored at 1 (thinning never goes below 1);
    a constant sequence gives 1. Input must be finite."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] < 4:
        raise ValueError("iat needs at least 4 draws")
    return _geyer_tau(_acov(x))


def _indicators(draws: np.ndarray, levels) -> np.ndarray:
    """``draws[..., n, Q]`` → ``f64[..., A, Q, n]``: ``1{f ≤ q̂_a}`` at the sequence's own empirical
    quantiles ``q̂_a``, ``a ∈ levels`` (Talts et al. §5.1; tail-ESS, Vehtari et al. 2021 §4.3)."""
    q = np.quantile(draws, levels, axis=-2, keepdims=True)  # (A, ..., 1, Q)
    return np.swapaxes(np.moveaxis(draws <= q, 0, -3), -1, -2).astype(np.float64)


def _pilot_iat(draws: np.ndarray) -> float:
    """Max over quantities of the IAT of ``f``, ``1{f ≤ q̂_0.05}`` and ``1{f ≤ q̂_0.95}``;
    ``draws`` is ``(pilot, Q)`` and finite (N06 §4.2 pass 1)."""
    tails = _indicators(draws, [0.05, 0.95])
    return float(max(iat(draws.T).max(), iat(tails).max()))


def _thinned_ess(tau: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """``L(1 − ρ^t)/(1 + ρ^t)`` with ``ρ = (τ − 1)/(τ + 1)`` (AR(1) with integrated time τ)."""
    rho_t = ((tau - 1.0) / (tau + 1.0)) ** t
    return L * (1.0 - rho_t) / (1.0 + rho_t)


#: ECDF levels of the ``ess_pooled`` diagnostic: tails (tail-ESS) and quartiles (bulk)
KEPT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _pooled_ess(acov_sum: np.ndarray, var_sq: np.ndarray, L: int, min_reps: float) -> np.ndarray:
    """``(Q,)`` ESS of the kept draws, ``L/max_a τ̄_a``: ``τ̄_a`` is :func:`_geyer_tau` of the
    replication-pooled autocovariances ``Σ_i γ̂_{i,k}`` of ``1{f ≤ q̂_{i,a}}`` (``acov_sum`` is
    ``(A, Q, L)``), over the levels with ``n_eff = (Σ_i γ̂_{i,0})²/Σ_i γ̂_{i,0}² ≥ min_reps``
    (``var_sq = Σ_i γ̂_{i,0}²``; Kish's effective number of replications: the s.e. of the pooled
    ``ρ̂_k`` is ``≈ 1/√(n_eff·L)``). NaN when no level qualifies (e.g. a quantity that is
    constant in almost every replication)."""
    var = acov_sum[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(var_sq > 0, var**2 / var_sq, 0.0)
    tau = np.where(n_eff >= min_reps, _geyer_tau(acov_sum), -np.inf).max(axis=0)
    return np.where(np.isfinite(tau), L / np.where(np.isfinite(tau), tau, 1.0), np.nan)


# --- JAX pieces ------------------------------------------------------------------------------


def _data_ok(data: PyTree, max_abs: float | None) -> jax.Array:
    """``finite ∧ |·| < max_abs`` over every floating leaf of the simulated data (data only)."""
    ok = jnp.array(True)
    for leaf in jax.tree.leaves(data):
        leaf = jnp.asarray(leaf)
        if not jnp.issubdtype(leaf.dtype, jnp.inexact):
            continue
        ok = ok & jnp.all(jnp.isfinite(leaf))
        if max_abs is not None:
            ok = ok & jnp.all(jnp.abs(leaf) < max_abs)
    return ok


def _truth(model: JointModel, key):
    th = model.prior_sample(part(key, Part.PRIOR))
    return th, model.simulate(part(key, Part.SIM), th)


def _run(model: JointModel, init_key, step_key, data, n: int, it0: int):
    """``n`` sweeps from an independent prior start (``n`` static); sweep indices ``it0..``."""
    th = model.prior_sample(init_key)
    return lax.fori_loop(
        0,
        n,
        lambda j, th: model.step(step_key, jnp.uint32(it0) + j.astype(jnp.uint32), th, data),
        th,
    )


def _filter_fn(model: JointModel, max_abs: float | None) -> Callable:
    return jax.jit(jax.vmap(lambda k: _data_ok(_truth(model, k)[1], max_abs)))


def _pilot_fn(model: JointModel, quantities: Callable, burn: int, pilot: int) -> Callable:
    def one(key):
        _, y = _truth(model, key)
        step_key = part(key, Part.PILOT_STEP)
        th = _run(model, part(key, Part.PILOT_INIT), step_key, y, burn, 0)

        def body(th, it):
            th = model.step(step_key, it, th, y)
            return th, quantities(th, y)

        _, qs = lax.scan(body, th, jnp.arange(burn, burn + pilot, dtype=jnp.uint32))
        return qs

    return jax.jit(jax.vmap(one))


def _production_fn(model: JointModel, quantities: Callable, burn: int, L: int, Q: int) -> Callable:
    def one(key, t):
        th_true, y = _truth(model, key)
        truth = quantities(th_true, y)
        step_key = part(key, Part.CHAIN_STEP)
        th = _run(model, part(key, Part.CHAIN_INIT), step_key, y, burn, 0)
        t_u = t.astype(jnp.uint32)

        def keep(th, j):
            start = jnp.uint32(burn) + j * t_u
            th = lax.fori_loop(
                0, t, lambda r, th: model.step(step_key, start + r.astype(jnp.uint32), th, y), th
            )
            return th, quantities(th, y)

        _, draws = lax.scan(keep, th, jnp.arange(L, dtype=jnp.uint32))
        u = random.uniform(part(key, Part.TIES), (Q,), dtype=jnp.float64)
        return rank_with_ties(truth, draws, u), truth, draws

    return jax.jit(jax.vmap(one))


def _blocks(indices: np.ndarray, size: int):
    """Fixed-size blocks of ``indices`` (the last one padded with its final index)."""
    for start in range(0, indices.size, size):
        chunk = indices[start : start + size]
        pad = size - chunk.size
        yield start, chunk.size, np.concatenate([chunk, np.repeat(chunk[-1:], pad)])


def _decide(
    names: tuple[str, ...], ranks: np.ndarray, L: int, alpha: float, failed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[str], list[str]]:
    """γ test (Bonferroni over ``Q``; Holm lists the failures) and the failed-replication check:
    ``(gamma, p, failing, γ reasons, other reasons)``."""
    if ranks.shape[0] == 0:
        empty = "no replication survived the data-only filter"
        return np.full(len(names), np.nan), np.zeros(len(names)), names, [], [empty]
    gammas, pvals = gamma_test(ranks, L)
    reject = holm_reject(pvals, alpha)
    failing = tuple(n for n, r in zip(names, reject, strict=True) if r)
    gamma_reasons, other = [], []
    if failing:
        gamma_reasons.append(f"γ test (Bonferroni over {len(names)} at α = {alpha:g}): {failing}")
    if failed.any():
        other.append(f"{int(failed.sum())} replication(s) failed (non-finite quantities)")
    return gammas, pvals, failing, gamma_reasons, other


def sbc(
    model: JointModel, quantities: Callable, names: tuple[str, ...], cfg: SBCConfig
) -> SBCResult:
    """Full-model SBC of ``model``'s sampler with test quantities ``quantities(state, data) ->
    f64[Q]`` (parameters, latent states, total and per-equation log-likelihood, ``log p(θ)``, …)."""
    names = tuple(names)
    Q = len(names)
    if Q == 0:
        raise ValueError("names is empty")
    comp = composition_id(model)
    tag = (cfg.burn, cfg.pilot, cfg.L, cfg.max_abs_data, quantities)
    with _jax.x64():
        probe = rep_keys(cfg.seed, Tier.S, comp, [0])[0]
        check_output(lambda k: quantities(*_truth(model, k)), probe, Q, "quantities")
        B = min(cfg.rep_block, cfg.N)

        # 1. data-only filter
        filt = compiled(
            model, ("sbc_filter", cfg.max_abs_data), lambda: _filter_fn(model, cfg.max_abs_data)
        )
        keep = np.zeros(cfg.N, dtype=bool)
        for start, size, idx in _blocks(np.arange(cfg.N), B):
            keep[start : start + size] = np.asarray(filt(rep_keys(cfg.seed, Tier.S, comp, idx)))[
                :size
            ]
        kept = np.flatnonzero(keep)
        n_kept = kept.size

        # 2. pilot chains → thinning
        pilot = compiled(
            model, ("sbc_pilot", *tag), lambda: _pilot_fn(model, quantities, cfg.burn, cfg.pilot)
        )
        tau = np.ones(n_kept)
        failed = np.zeros(n_kept, dtype=bool)
        for start, size, idx in _blocks(kept, B):
            qs = np.asarray(pilot(rep_keys(cfg.seed, Tier.S, comp, idx)))[:size]  # (size, pilot, Q)
            for j in range(size):
                if np.all(np.isfinite(qs[j])):
                    tau[start + j] = _pilot_iat(qs[j])
                else:
                    failed[start + j] = True
        thin = np.clip(np.ceil(cfg.iat_factor * tau), 1, cfg.t_max).astype(np.int64)
        thin = np.where(failed, cfg.t_max, thin)

        # 3. production chains, sorted by thinning
        prod = compiled(
            model,
            ("sbc_prod", *tag, Q),
            lambda: _production_fn(model, quantities, cfg.burn, cfg.L, Q),
        )
        order = np.argsort(thin, kind="stable")
        ranks = np.zeros((n_kept, Q), dtype=np.int64)
        ess_kept = np.full(n_kept, np.nan)
        acov_sum, var_sq = np.zeros((len(KEPT_LEVELS), Q, cfg.L)), np.zeros((len(KEPT_LEVELS), Q))
        for _, size, pos in _blocks(order, B):
            keys = rep_keys(cfg.seed, Tier.S, comp, kept[pos])
            r, truth, draws = prod(keys, jnp.asarray(thin[pos], dtype=jnp.int32))
            r, truth, draws = (np.asarray(a)[:size] for a in (r, truth, draws))
            where = pos[:size]
            ranks[where] = r
            finite = np.isfinite(truth).all(axis=1) & np.isfinite(draws).all(axis=(1, 2))
            failed[where] |= ~finite
            if cfg.L >= 4 and finite.any():
                good = draws[finite]  # (n, L, Q)
                ess_kept[where[finite]] = cfg.L / iat(np.swapaxes(good, 1, 2)).max(axis=1)
                for lo in range(0, good.shape[0], 25):  # bounds the FFT workspace
                    acov = _acov(_indicators(good[lo : lo + 25], KEPT_LEVELS))  # (n, A, Q, L)
                    acov_sum += acov.sum(axis=0)
                    var_sq += (acov[..., 0] ** 2).sum(axis=0)

    gammas, pvals, failing, gamma_reasons, other = _decide(names, ranks, cfg.L, cfg.alpha, failed)
    # the ESS rule (DESIGN §9.6): per-replication thinned ESS ≥ min_ess·L for ≥ ess_coverage
    ess_thin = _thinned_ess(tau, thin, cfg.L)
    ess_ok = float(np.mean(ess_thin >= cfg.min_ess * cfg.L)) if n_kept else 0.0
    if n_kept and ess_ok < cfg.ess_coverage:
        other.append(
            f"thinned ESS ≥ {cfg.min_ess:g}·L in only {100 * ess_ok:.1f}% of replications "
            f"(need {100 * cfg.ess_coverage:g}%; raise t_max or pilot)"
        )
    # second ESS gate (module doc): catches pilots that underestimate the IAT
    ess_pooled = _pooled_ess(acov_sum, var_sq, cfg.L, cfg.pooled_min_reps) if cfg.L >= 4 else None
    if ess_pooled is not None:
        low = np.isfinite(ess_pooled) & (ess_pooled < cfg.min_ess * cfg.L)
        if low.any():
            other.append(
                f"pooled kept-draw ESS below {cfg.min_ess:g}·L for {[names[q] for q in np.flatnonzero(low)]}: "
                "the pilot underestimates the IAT (raise pilot, burn or iat_factor)"
            )
    reasons = gamma_reasons + other
    return SBCResult(
        names=names,
        ranks=ranks,
        L=int(cfg.L),
        gamma=gammas,
        p=pvals,
        alpha=float(cfg.alpha),
        passed=not reasons,
        failing=failing,
        reasons=tuple(reasons),
        n_total=int(cfg.N),
        kept=kept,
        failed=failed,
        thin=thin,
        iat=tau,
        ess_thin=ess_thin,
        ess_ok=ess_ok,
        ess_kept=ess_kept,
        ess_pooled=ess_pooled,
        checks_ok=not other,
    )


def sbc_sequential(
    model: JointModel,
    quantities: Callable,
    names: tuple[str, ...],
    cfg: SBCConfig,
    k: int = 3,
    delta: int = 4,
) -> tuple[SequentialResult, list[SBCResult]]:
    """:func:`sbc` inside the Gandy–Scott wrapper at ``cfg.alpha`` (DESIGN §9.6): stage ``s`` uses
    :func:`~rai_bvar.validation.sequential.stage_seed` and ``N`` multiplied by ``Δ`` after stage 1.
    The stage value is :attr:`SBCResult.p_bonferroni`, so a stage whose replications failed or
    whose ESS rule failed returns ``q = 0`` (failed fits fail the test; they are not a marginal
    result)."""
    results: list[SBCResult] = []

    def run(stage: int) -> float:
        m = stage_multiplier(stage, delta)
        res = sbc(
            model, quantities, names, replace(cfg, seed=stage_seed(cfg.seed, stage), N=cfg.N * m)
        )
        results.append(res)
        return res.p_bonferroni

    return gandy_scott(run, cfg.alpha, k, delta), results


def _block_fn(
    joint_sample: Callable, conditional_draws: Callable, stat: Callable, L: int, Q: int,
    max_abs: float | None,
) -> Callable:  # fmt: skip
    def one(key):
        th, y = joint_sample(part(key, Part.PRIOR))
        truth = stat(th, y)
        dkey = part(key, Part.DRAWS)
        draws = jax.vmap(lambda d: stat(conditional_draws(random.fold_in(dkey, d), th, y), y))(
            jnp.arange(L, dtype=jnp.uint32)
        )
        u = random.uniform(part(key, Part.TIES), (Q,), dtype=jnp.float64)
        ok = _data_ok(y, max_abs)
        finite = jnp.all(jnp.isfinite(truth)) & jnp.all(jnp.isfinite(draws))
        return rank_with_ties(truth, draws, u), ok, finite

    return jax.jit(jax.vmap(one))


def block_sbc(
    joint_sample: Callable,
    conditional_draws: Callable,
    stat: Callable,
    N: int = 200,
    L: int = 99,
    seed: int = 0,
    *,
    names: tuple[str, ...] | None = None,
    alpha: float = 1e-3,
    composition: int = 0,
    max_abs_data: float | None = None,
    rep_block: int = 200,
) -> SBCResult:
    """Block-conditional SBC (N06 §5.2): ``joint_sample(key) -> (θ̃, ỹ)`` draws from the joint,
    ``conditional_draws(key, θ̃, ỹ) -> θ`` returns ``θ̃`` with block ``b`` redrawn from its exact
    conditional (one iid draw per key), ``stat(θ, ỹ) -> f64[Q]`` the block statistics (include
    the block log-likelihood). Tier B: ``N = 200``, ``L = 99``, Bonferroni at ``α = 1e-3``."""
    _int_at_least(N, "N", 1)
    _int_at_least(L, "L", 1)
    _int_at_least(rep_block, "rep_block", 1)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    with _jax.x64():
        probe = rep_keys(seed, Tier.B, composition, [0])[0]
        shape = jax.eval_shape(lambda k: stat(*joint_sample(k)), probe)
        if len(shape.shape) != 1 or shape.dtype != jnp.float64:
            raise TypeError(f"stat must return f64[Q], got {shape.dtype}{list(shape.shape)}")
        Q = int(shape.shape[0])
        names = tuple(names) if names is not None else tuple(f"q{j}" for j in range(Q))
        if len(names) != Q:
            raise ValueError(f"names has {len(names)} entries, stat returns {Q}")
        fn = compiled(
            joint_sample,
            ("block_sbc", conditional_draws, stat, L, Q, max_abs_data),
            lambda: _block_fn(joint_sample, conditional_draws, stat, L, Q, max_abs_data),
        )
        B = min(rep_block, N)
        ranks = np.zeros((N, Q), dtype=np.int64)
        keep = np.zeros(N, dtype=bool)
        finite = np.zeros(N, dtype=bool)
        for start, size, idx in _blocks(np.arange(N), B):
            r, ok, fin = (
                np.asarray(a)[:size] for a in fn(rep_keys(seed, Tier.B, composition, idx))
            )
            ranks[start : start + size], keep[start : start + size] = r, ok
            finite[start : start + size] = fin
    kept = np.flatnonzero(keep)
    failed = ~finite[kept]
    gammas, pvals, failing, gamma_reasons, other = _decide(names, ranks[kept], L, alpha, failed)
    reasons = gamma_reasons + other
    return SBCResult(
        names=names,
        ranks=ranks[kept],
        L=int(L),
        gamma=gammas,
        p=pvals,
        alpha=float(alpha),
        passed=not reasons,
        failing=failing,
        reasons=tuple(reasons),
        n_total=int(N),
        kept=kept,
        failed=failed,
        checks_ok=not other,
    )


def block_sbc_sequential(
    joint_sample: Callable,
    conditional_draws: Callable,
    stat: Callable,
    N: int = 200,
    L: int = 99,
    seed: int = 0,
    *,
    names: tuple[str, ...] | None = None,
    alpha: float = 1e-3,
    composition: int = 0,
    max_abs_data: float | None = None,
    rep_block: int = 200,
    k: int = 3,
    delta: int = 4,
) -> tuple[SequentialResult, list[SBCResult]]:
    """:func:`block_sbc` inside the Gandy–Scott wrapper at family level ``alpha`` (Tier B,
    DESIGN §9.4: ``k = 3``, ``Δ = 4``, fresh keys per stage): stage ``s`` uses
    :func:`~rai_bvar.validation.sequential.stage_seed` ``(seed, s)`` and ``N`` multiplied by ``Δ``
    after stage 1. The stage value is :attr:`SBCResult.p_bonferroni`, so a stage with failed
    replications returns ``q = 0`` and fails at once."""
    results: list[SBCResult] = []

    def run(stage: int) -> float:
        res = block_sbc(
            joint_sample,
            conditional_draws,
            stat,
            N=N * stage_multiplier(stage, delta),
            L=L,
            seed=stage_seed(seed, stage),
            names=names,
            alpha=alpha,
            composition=composition,
            max_abs_data=max_abs_data,
            rep_block=rep_block,
        )
        results.append(res)
        return res.p_bonferroni

    return gandy_scott(run, alpha, k, delta), results
