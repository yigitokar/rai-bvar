"""Rank uniformity for simulation-based calibration (N06 §4.1, §4.3; DESIGN §9.4, §9.6).

**Ranks with ties** (Modrák et al. 2025 §1.2; N07 §13.3). For one quantity with true value ``f̃``
and ``L`` draws ``f_l``, ``rank = n_less + ⌊u·(n_tie + 1)⌋`` with ``n_less = #{f_l < f̃}``,
``n_tie = #{f_l = f̃}`` and ONE uniform ``u`` per quantity: the rank is uniform over the tie
block, so discrete quantities (indicator counts, ν on a grid, repeated MH values) keep uniform
ranks. A coin flip per tied draw would not (it gives a Binomial offset; N07 §13.3).

**γ statistic** (Säilynoja, Bürkner & Vehtari 2022 eq. 13 = Modrák et al. 2025 eq. 7). For ``N``
ranks in ``{0..L}``, with ``z_i = i/(L+1)`` and ``R_i = #{ranks < i}``::

    γ = 2 · min_{i=1..L} c_i(R_i),    c_i(r) = min{ F_Bin(r; N, z_i), 1 − F_Bin(r − 1; N, z_i) }

Small γ means the rank ECDF leaves its pointwise binomial envelope somewhere.

**Exact null distribution** (Säilynoja et al. 2022 §2.3, eqs 15–23; N06 §4.3). Discrete ranks
uniform on ``{0..L}`` give ``R_0 = 0`` and ``R_{i+1} − R_i | R_i ~ Bin(N − R_i, 1/(L+1−i))``, so
the law of the path ``(R_1..R_L)`` is a Markov chain. ``P(γ_null > g)`` is the probability that
the path stays in the interiors ``I_i(g) = {r : 2c_i(r) > g}`` (each an interval, since
``F_Bin(r)`` increases and ``1 − F_Bin(r − 1)`` decreases in ``r``), and the p-value

    p(γ_obs) = P(γ_null ≤ γ_obs) = Σ_i P(path inside I_1..I_i, R_{i+1} ∉ I_{i+1})

is accumulated as the mass leaving the interior at each step, which keeps far-tail p-values
(Bonferroni at ``α/Q``) accurate to relative precision instead of computing ``1 − P(stay)``. The
envelope values ``c_i(r)`` come from one cached table per ``(N, L)``, used both for the statistic
and for the interiors, so ``γ_obs`` is exactly an atom of the null.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from scipy import special, stats

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)

__all__ = [
    "envelope_table",
    "gamma_pvalue",
    "gamma_statistic",
    "gamma_test",
    "rank_with_ties",
]


def rank_with_ties(truth, draws, u):
    """Ranks of ``truth`` (``f64[Q]``) among ``draws`` (``f64[L, Q]``) with uniform tie-breaking.

    ``rank_q = n_less + ⌊u_q·(n_tie + 1)⌋ ∈ {0..L}``, one uniform ``u_q ∈ [0, 1)`` per quantity
    (N07 §13.3). Pure ``jnp``: usable under ``jit``/``vmap``. Returns ``i32[Q]``.
    """
    truth = jnp.asarray(truth)
    draws = jnp.asarray(draws)
    u = jnp.asarray(u)
    n_less = jnp.sum(draws < truth[None, :], axis=0)
    n_tie = jnp.sum(draws == truth[None, :], axis=0)
    offset = jnp.floor(u * (n_tie + 1)).astype(n_less.dtype)
    return (n_less + jnp.clip(offset, 0, n_tie)).astype(jnp.int32)


@lru_cache(maxsize=16)
def _table(N: int, L: int) -> np.ndarray:
    z = np.arange(1, L + 1, dtype=np.float64)[:, None] / (L + 1)
    r = np.arange(N + 1, dtype=np.float64)[None, :]
    lower = stats.binom.cdf(r, N, z)  # F_Bin(r; N, z_i)
    upper = stats.binom.sf(r - 1.0, N, z)  # 1 − F_Bin(r − 1; N, z_i) = P(X ≥ r)
    table = np.minimum(lower, upper)
    table.setflags(write=False)
    return table


def _check_sizes(N: int, L: int) -> tuple[int, int]:
    if isinstance(N, bool) or int(N) != N or N < 1:
        raise ValueError(f"N must be a positive integer, got {N!r}")
    if isinstance(L, bool) or int(L) != L or L < 1:
        raise ValueError(f"L must be a positive integer, got {L!r}")
    return int(N), int(L)


def envelope_table(N: int, L: int) -> np.ndarray:
    """``c_i(r)`` for ``i = 1..L`` (rows) and ``r = 0..N`` (columns); read-only, cached."""
    return _table(*_check_sizes(N, L))


def _cumulative_counts(ranks: np.ndarray, L: int) -> np.ndarray:
    """``R_i = #{ranks < i}`` for ``i = 1..L`` along axis 0 (ranks of shape ``(N,)`` or ``(N, Q)``)."""
    ranks = np.asarray(ranks)
    if ranks.ndim not in (1, 2) or ranks.shape[0] < 1:
        raise ValueError(f"ranks must have shape (N,) or (N, Q) with N ≥ 1, got {ranks.shape}")
    if not np.issubdtype(ranks.dtype, np.integer):
        if not np.all(np.isfinite(ranks)) or np.any(ranks != np.round(ranks)):
            raise ValueError("ranks must be integers")
        ranks = ranks.astype(np.int64)
    if ranks.size and (ranks.min() < 0 or ranks.max() > L):
        raise ValueError(f"ranks must lie in 0..L = 0..{L}")
    flat = ranks.reshape(ranks.shape[0], -1)
    counts = np.stack([np.bincount(col, minlength=L + 1) for col in flat.T], axis=1)
    R = np.cumsum(counts, axis=0)[:L]  # row i−1 holds R_i
    return R.reshape((L, *ranks.shape[1:]))


def gamma_statistic(ranks: np.ndarray, L: int) -> float:
    """γ of ``N`` ranks in ``{0..L}`` (Säilynoja et al. 2022 eq. 13); ``ranks`` has shape ``(N,)``."""
    ranks = np.asarray(ranks)
    if ranks.ndim != 1:
        raise ValueError("gamma_statistic takes one quantity (shape (N,)); use gamma_test")
    N, L = _check_sizes(ranks.shape[0], L)
    R = _cumulative_counts(ranks, L)
    return float(2.0 * _table(N, L)[np.arange(L), R].min())


def gamma_pvalue(gamma: float, N: int, L: int) -> float:
    """Exact ``P(γ_null ≤ γ)`` for ``N`` iid uniform ranks in ``{0..L}`` (Säilynoja eqs 15–23).

    Forward recursion over ``R_0 = 0 → R_1 → … → R_L`` restricted to the interiors
    ``{r : 2c_i(r) > γ}``; the p-value is the total mass leaving them. Cost ``O(Σ_i w_i w_{i+1})``
    with ``w_i`` the interior widths (at most ``O(L·N²)``).
    """
    N, L = _check_sizes(N, L)
    gamma = float(gamma)
    if np.isnan(gamma):
        raise ValueError("gamma is NaN")
    if gamma < 0.0:
        return 0.0
    table = _table(N, L)
    mass = np.ones(1)  # law of R_i on its support [a, a + len(mass)); R_0 = 0
    a = 0
    lost = 0.0
    for i in range(L):  # step R_i → R_{i+1}; interior of R_{i+1} is row i of the table
        inside = np.flatnonzero(2.0 * table[i] > gamma)
        if inside.size == 0:  # every path leaves here
            lost += float(mass.sum())
            mass = np.zeros(0)
            break
        lo, hi = int(inside[0]), int(inside[-1])
        prob = 1.0 / (L + 1 - i)
        r = a + np.arange(mass.size, dtype=np.int64)
        n_rem = (N - r).astype(np.float64)
        leave = stats.binom.cdf(lo - r - 1, n_rem, prob) + stats.binom.sf(hi - r, n_rem, prob)
        lost += float(mass @ leave)
        k = np.arange(lo, hi + 1, dtype=np.int64)[None, :] - r[:, None]  # increments
        valid = (k >= 0) & (k <= (N - r)[:, None])
        kc = np.clip(k, 0, (N - r)[:, None]).astype(np.float64)
        nr = n_rem[:, None]
        logpmf = (
            special.gammaln(nr + 1.0)
            - special.gammaln(kc + 1.0)
            - special.gammaln(nr - kc + 1.0)
            + kc * np.log(prob)
            + (nr - kc) * np.log1p(-prob)
        )
        mass = mass @ np.where(valid, np.exp(np.where(valid, logpmf, 0.0)), 0.0)
        a = lo
    return float(min(1.0, max(0.0, lost)))


def gamma_test(ranks: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """γ and its exact p-value per column of ``ranks`` (shape ``(N, Q)``): ``(f64[Q], f64[Q])``."""
    ranks = np.asarray(ranks)
    if ranks.ndim == 1:
        ranks = ranks[:, None]
    N, L = _check_sizes(ranks.shape[0], L)
    R = _cumulative_counts(ranks, L)  # (L, Q)
    table = _table(N, L)
    gammas = 2.0 * table[np.arange(L)[:, None], R].min(axis=0)
    cache: dict[float, float] = {}
    pvals = np.empty_like(gammas)
    for q, g in enumerate(gammas):
        key = float(g)
        if key not in cache:
            cache[key] = gamma_pvalue(key, N, L)
        pvals[q] = cache[key]
    return gammas, pvals
