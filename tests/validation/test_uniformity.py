"""Rank uniformity (N06 §4.1, §4.3; N07 §13.3): ties, the γ statistic and its exact null law."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from rai_bvar.validation.uniformity import (
    envelope_table,
    gamma_pvalue,
    gamma_statistic,
    gamma_test,
    rank_with_ties,
)

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

# N06 §0: null quantiles of γ for L = 99 from the exact recursion (cross-checked by simulation).
N06_QUANTILES = {
    100: {0.05: 4.1e-3, 0.01: 6.8e-4, 0.001: 6.6e-5, 0.0001: 5.5e-6},
    200: {0.05: 3.5e-3, 0.01: 5.7e-4, 0.001: 5.2e-5, 0.0001: 4.6e-6},
    500: {0.05: 3.0e-3, 0.01: 4.9e-4, 0.001: 4.1e-5, 0.0001: 3.7e-6},
    1000: {0.05: 2.7e-3, 0.01: 4.6e-4, 0.001: 3.7e-5, 0.0001: 3.4e-6},
}


def gamma_direct(ranks: np.ndarray, L: int) -> float:
    """Säilynoja et al. eq. 13 evaluated straight from scipy (no table)."""
    N = ranks.size
    out = np.inf
    for i in range(1, L + 1):
        z = i / (L + 1)
        R = np.sum(ranks < i)
        out = min(out, stats.binom.cdf(R, N, z), 1.0 - stats.binom.cdf(R - 1, N, z))
    return 2.0 * out


def simulate_gamma(N: int, L: int, draws: int, seed: int) -> np.ndarray:
    """``draws`` null γ values: counts of each rank value are Multinomial(N, uniform)."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(N, np.full(L + 1, 1.0 / (L + 1)), size=draws)
    R = np.cumsum(counts, axis=1)[:, :L]  # R_i = #{ranks < i}, i = 1..L
    table = envelope_table(N, L)
    return 2.0 * table[np.arange(L)[None, :], R].min(axis=1)


# --- ranks --------------------------------------------------------------------------------------


def test_rank_without_ties_counts_smaller_draws():
    draws = jnp.array([[0.1, 5.0], [0.3, 1.0], [0.7, 2.0], [0.9, 3.0]])
    truth = jnp.array([0.5, 2.5])
    r = rank_with_ties(truth, draws, jnp.array([0.99, 0.0]))
    assert r.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(r), [2, 2])


def test_rank_ties_are_uniform_over_the_tie_block():
    # truth tied with 3 draws, 1 draw below: rank uniform on {1, 2, 3, 4}
    draws = jnp.array([[0.0], [1.0], [1.0], [1.0], [2.0]])
    u = jnp.linspace(0.0, 1.0, 4000, endpoint=False)
    ranks = jax.vmap(lambda ui: rank_with_ties(jnp.array([1.0]), draws, ui[None])[0])(u)
    values, counts = np.unique(np.asarray(ranks), return_counts=True)
    np.testing.assert_array_equal(values, [1, 2, 3, 4])
    np.testing.assert_array_equal(counts, [1000] * 4)
    # u just below 1 never exceeds the block
    top = rank_with_ties(jnp.array([1.0]), draws, jnp.array([np.nextafter(1.0, 0.0)]))
    assert int(top[0]) == 4


def test_tie_breaking_keeps_discrete_ranks_uniform():
    # N07 §13.3: exchangeable draws on {0, 1, 2}, L = 9. One uniform per quantity gives uniform
    # ranks; a coin flip per tied draw would not (edge bins 0.066 vs 0.1).
    rng = np.random.default_rng(11)
    reps, L = 40000, 9
    values = rng.integers(0, 3, size=(reps, L + 1)).astype(np.float64)
    u = rng.uniform(size=reps)
    ranks = jax.jit(jax.vmap(lambda v, ui: rank_with_ties(v[:1], v[1:, None], ui[None])[0]))(
        jnp.asarray(values), jnp.asarray(u)
    )
    counts = np.bincount(np.asarray(ranks), minlength=L + 1)
    assert stats.chisquare(counts).pvalue > 1e-3
    # the wrong rule for contrast: n_less + Binomial(n_tie, 1/2)
    n_less = (values[:, 1:] < values[:, :1]).sum(axis=1)
    n_tie = (values[:, 1:] == values[:, :1]).sum(axis=1)
    coin = n_less + rng.binomial(n_tie, 0.5)
    assert stats.chisquare(np.bincount(coin, minlength=L + 1)).pvalue < 1e-6


# --- γ statistic --------------------------------------------------------------------------------


@pytest.mark.parametrize(("N", "L"), [(50, 9), (200, 99), (37, 20)])
def test_gamma_statistic_matches_the_direct_formula(N, L):
    rng = np.random.default_rng(N + L)
    for ranks in (rng.integers(0, L + 1, N), np.minimum(rng.integers(0, L + 1, N), L // 2)):
        assert gamma_statistic(ranks, L) == pytest.approx(gamma_direct(ranks, L), rel=1e-12)


def test_gamma_test_is_the_columnwise_statistic_and_pvalue():
    rng = np.random.default_rng(3)
    ranks = rng.integers(0, 20, size=(80, 4))
    ranks[:, 3] = np.minimum(ranks[:, 3], 5)  # strongly non-uniform column
    gammas, pvals = gamma_test(ranks, 19)
    for q in range(4):
        assert gammas[q] == gamma_statistic(ranks[:, q], 19)
        assert pvals[q] == gamma_pvalue(gammas[q], 80, 19)
    assert pvals[3] < 1e-10 < pvals[:3].min()


def test_rank_validation():
    with pytest.raises(ValueError, match=r"0\.\.L"):
        gamma_statistic(np.array([0, 1, 10]), 9)
    with pytest.raises(ValueError, match="integers"):
        gamma_statistic(np.array([0.5, 1.0]), 9)
    with pytest.raises(ValueError, match="one quantity"):
        gamma_statistic(np.zeros((3, 2), dtype=int), 9)
    with pytest.raises(ValueError, match="positive integer"):
        gamma_pvalue(0.1, 0, 9)


# --- exact null law -----------------------------------------------------------------------------


@pytest.mark.parametrize(("N", "L"), [(3, 2), (4, 3), (5, 4), (2, 6)])
def test_pvalue_equals_brute_force_enumeration(N, L):
    # Every rank vector in {0..L}^N is equally likely under H0.
    gammas = np.array(
        [gamma_statistic(np.array(r), L) for r in itertools.product(range(L + 1), repeat=N)]
    )
    for g in np.unique(gammas):
        assert gamma_pvalue(g, N, L) == pytest.approx(np.mean(gammas <= g), rel=1e-12, abs=1e-15)
    assert gamma_pvalue(gammas.max() * 1.5, N, L) == 1.0
    assert gamma_pvalue(-1.0, N, L) == 0.0


@pytest.mark.parametrize("N", [100, 200, 500, 1000])
def test_pvalues_at_the_n06_quantiles(N):
    # γ is discrete and the table has two significant digits: p(q_α) ≈ α to within 7%.
    for alpha, q in N06_QUANTILES[N].items():
        assert gamma_pvalue(q, N, 99) == pytest.approx(alpha, rel=0.07)


@pytest.mark.parametrize("N", [100, 200, 1000])
def test_exact_recursion_agrees_with_simulation(N):
    # DESIGN §13 WP7: exact recursion vs 10⁵ simulated null γ values (N06 §0 table).
    L, draws = 99, 100_000
    sim = simulate_gamma(N, L, draws, seed=N)
    for alpha in (0.05, 0.01, 0.001):
        g = np.quantile(sim, alpha)
        p_exact = gamma_pvalue(g, N, L)
        p_sim = np.mean(sim <= g)
        se = np.sqrt(p_exact * (1 - p_exact) / draws)
        assert abs(p_sim - p_exact) < 4 * se + 1e-5, (alpha, p_sim, p_exact)
    # the exact quantiles reproduce the N06 table
    for alpha, q in N06_QUANTILES[N].items():
        if alpha >= 0.001:
            assert np.quantile(sim, alpha) == pytest.approx(q, rel=0.25)


def test_pvalue_extremes_and_monotonicity():
    N, L = 120, 19
    rng = np.random.default_rng(5)
    uniform = np.repeat(np.arange(L + 1), N // (L + 1))
    assert gamma_pvalue(gamma_statistic(uniform, L), N, L) > 0.99
    assert gamma_pvalue(gamma_statistic(np.zeros(N, dtype=int), L), N, L) < 1e-50
    gs = np.sort(rng.uniform(0, 1, 30))
    ps = [gamma_pvalue(g, N, L) for g in gs]
    assert np.all(np.diff(ps) >= 0)


def test_null_pvalues_are_super_uniform():
    # P(p ≤ a) ≤ a for ranks drawn under H0 (discrete statistic: conservative at atoms)
    rng = np.random.default_rng(9)
    N, L = 60, 19
    _, p = gamma_test(rng.integers(0, L + 1, size=(N, 3000)), L)  # atoms repeat: p cached
    for a in (0.01, 0.05, 0.2, 0.5):
        assert np.mean(p <= a) <= a + 4 * np.sqrt(a * (1 - a) / p.size)
    assert np.mean(p <= 0.5) > 0.4  # and not trivially conservative
