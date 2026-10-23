"""Convergence diagnostics (SPEC §10.6; Vehtari et al. 2021): rank-normalised split R-hat, bulk ESS.

Reference values come from ArviZ: ``az.rhat(x, method="rank")`` / ``az.ess(x, method="bulk")``
from ``arviz`` 0.23.4 and ``arviz_stats.base.array_stats.rhat/ess`` from ``arviz-stats`` 1.3.3,
which agree to the last digit on every case below. They were computed once, in a throwaway
virtualenv (numpy 2.5.4, scipy 1.18.1), on the exact arrays the builders in this file produce,
and are hard-coded: ArviZ is not a test dependency. Both functions are deterministic
re-implementations of the same estimator, so the comparison tolerance only allows for
floating-point summation order and FFT length (``rtol=1e-10``).

Statistical tolerances (iid / AR(1) ESS, shifted and rescaled chains) were calibrated from 200–300
replications of the same designs; each bound sits at least ~3.5 sampling s.d. from the observed
mean, and the corresponding replication extremes are quoted next to each bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rai_bvar.diagnostics import ess_bulk, rhat

RHAT_THRESHOLD = 1.01  # SPEC §10.6 default; the analytic engine's convergence flag uses it


# --------------------------------------------------------------------------- chain builders
# (the reference script exec'd exactly this block in the ArviZ venv; keep it self-contained)


def ar1_chains(rng: np.random.Generator, rho: float, chains: int, draws: int) -> np.ndarray:
    """Stationary Gaussian AR(1) chains with unit innovation variance."""
    e = rng.standard_normal((chains, draws))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1.0 - rho**2)
    for t in range(1, draws):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def rwm_chains(rng: np.random.Generator, starts, draws: int, step: float = 2.4) -> np.ndarray:
    """Random-walk Metropolis chains targeting N(0, 1): rejections produce repeated values."""
    cur = np.asarray(starts, dtype=np.float64).copy()
    out = np.empty((cur.size, draws))
    for t in range(draws):
        prop = cur + step * rng.standard_normal(cur.size)
        accept = np.log(rng.random(cur.size)) < 0.5 * (cur**2 - prop**2)
        cur = np.where(accept, prop, cur)
        out[:, t] = cur
    return out


CASE_SEEDS = {
    "iid": 101,
    "iid_odd": 102,
    "shifted": 103,
    "scaled": 104,
    "ar_0.5": 105,
    "ar_0.9": 106,
    "ar_-0.3": 107,
    "ar_-0.9": 108,
    "cauchy": 109,
    "ties": 110,
    "rwm": 111,
    "stuck_at_median": 112,
    "short": 113,
    "single_chain": 114,
    "many_chains": 115,
    "infinities": 116,
}


def build(name: str) -> np.ndarray:
    """The ``(chains, draws)`` array of reference case ``name``."""
    rng = np.random.default_rng(CASE_SEEDS[name])
    if name == "iid":
        return rng.standard_normal((4, 1000))
    if name == "iid_odd":  # odd length: the split drops the middle draw
        return rng.standard_normal((4, 1001))
    if name == "shifted":  # one chain shifted by one s.d.
        return rng.standard_normal((4, 1000)) + np.array([[0.0], [0.0], [0.0], [1.0]])
    if name == "scaled":  # same location, one chain with three times the s.d.
        return rng.standard_normal((4, 1000)) * np.array([[1.0], [1.0], [1.0], [3.0]])
    if name.startswith("ar_"):
        return ar1_chains(rng, float(name[3:]), 4, 2000)
    if name == "cauchy":  # infinite variance: only rank normalisation makes R-hat meaningful
        return rng.standard_cauchy((4, 1000))
    if name == "ties":  # discrete draws: average ranks
        return rng.poisson(2.0, (4, 500)).astype(np.float64)
    if name == "rwm":  # dispersed starts, no burn-in
        return rwm_chains(rng, [-3.0, -1.0, 1.0, 3.0], 1000)
    if name == "stuck_at_median":  # one chain never moves, sitting at the pooled median
        x = rng.standard_normal((4, 1000))
        x[3] = 0.0
        return x
    if name == "short":  # the minimum length ArviZ accepts for R-hat
        return rng.standard_normal((4, 4))
    if name == "single_chain":
        return rng.standard_normal((1, 1000))
    if name == "many_chains":  # 16 short chains with small chain-specific offsets
        return rng.standard_normal((16, 64)) + 0.2 * rng.standard_normal((16, 1))
    if name == "infinities":
        x = rng.standard_normal((4, 100))
        x[1, 5], x[2, 50] = np.inf, -np.inf
        return x
    raise KeyError(name)


# --------------------------------------------------------------------------- end of builders

# ArviZ 0.23.4 == arviz-stats 1.3.3 on every case (printed with repr, i.e. full precision).
# ``single_chain`` has no R-hat reference: ArviZ requires two chains (it returns nan).
ARVIZ_RHAT = {
    "iid": 1.0005552941690128,
    "iid_odd": 1.0002313366184798,
    "shifted": 1.101023132050445,
    "scaled": 1.1398545374036384,
    "ar_0.5": 1.0005177588765888,
    "ar_0.9": 1.0104143847907605,
    "ar_-0.3": 1.0000339400959062,
    "ar_-0.9": 1.0029910749501787,
    "cauchy": 1.0004429954824954,
    "ties": 1.0001147257529313,
    "rwm": 1.0060929783740487,
    "stuck_at_median": 1.5243948555995526,
    "short": 1.3197748051678635,
    "many_chains": 1.0348602681949974,
    "infinities": 1.0014268762050047,
}
# ``short`` has no ESS reference: with split halves of two draws ArviZ returns the cap
# ``N log10 N`` (19.27), a placeholder rather than an estimate (see test_too_short_for_ess).
ARVIZ_ESS_BULK = {
    "iid": 3879.9901276918727,
    "iid_odd": 4137.442500570096,
    "shifted": 26.53068065208921,
    "scaled": 3803.473763145608,
    "ar_0.5": 2618.1316826736397,
    "ar_0.9": 443.20932061428044,
    "ar_-0.3": 14896.530665872835,
    "ar_-0.9": 31224.71989593555,
    "cauchy": 4184.000211070284,
    "ties": 1889.1391756286828,
    "rwm": 824.5371789835953,
    "stuck_at_median": 4271.17115697426,
    "single_chain": 952.7351896171286,
    "many_chains": 625.8995307199685,
    "infinities": 485.25310854218543,
}
REF_RTOL = 1e-10


# --------------------------------------------------------------------------- independent oracles


def _split(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.vstack([x[:, :half], x[:, x.shape[1] - half :]])


def _gelman_rubin(chains: np.ndarray) -> float:
    """Classic potential scale reduction factor of the given (already split) chains."""
    n = chains.shape[1]
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def _normal_scores(x: np.ndarray) -> np.ndarray:
    """Blom (1958) normal scores of the pooled draws."""
    ranks = stats.rankdata(x, method="average").reshape(x.shape)
    return stats.norm.ppf((ranks - 3 / 8) / (x.size + 1 / 4))


def _bulk_rhat(x: np.ndarray) -> float:
    """Rank-normalised split R-hat without the folded (tail) half."""
    return _gelman_rubin(_normal_scores(_split(x)))


# --------------------------------------------------------------------------- ArviZ parity


@pytest.mark.parametrize("case", sorted(ARVIZ_RHAT))
def test_rhat_matches_arviz(case):
    value = rhat(build(case))
    assert isinstance(value, float)
    np.testing.assert_allclose(value, ARVIZ_RHAT[case], rtol=REF_RTOL)


@pytest.mark.parametrize("case", sorted(ARVIZ_ESS_BULK))
def test_ess_bulk_matches_arviz(case):
    value = ess_bulk(build(case))
    assert isinstance(value, float)
    np.testing.assert_allclose(value, ARVIZ_ESS_BULK[case], rtol=REF_RTOL)


# --------------------------------------------------------------------------- R-hat behaviour


@pytest.mark.parametrize(("chains", "draws", "seed"), [(4, 1000, 1), (8, 500, 2), (2, 2000, 3)])
def test_rhat_close_to_one_for_iid_chains(chains, draws, seed):
    """iid chains pass the convergence threshold (300 replications of 4x1000: range 0.9993–1.0021)."""
    value = rhat(np.random.default_rng(seed).standard_normal((chains, draws)))
    assert 1.0 - 0.005 < value < 1.0 + 0.005
    assert value < RHAT_THRESHOLD


def test_rhat_flags_a_shifted_chain():
    """One chain shifted by one s.d. (300 replications: R-hat ≥ 1.081)."""
    x = np.random.default_rng(4).standard_normal((4, 1000))
    x[2] += 1.0
    assert rhat(x) > 1.05


def test_rhat_increases_with_the_shift():
    base = np.random.default_rng(31).standard_normal((4, 1000))
    values = [rhat(base + np.array([[0.0], [0.0], [0.0], [s]])) for s in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert np.all(np.diff(values) > 0), values
    assert values[0] < RHAT_THRESHOLD < values[2]


def test_folded_rhat_catches_a_variance_difference():
    """Same location, s.d. ratio 3: invisible to the classic and the bulk R-hat, caught by the
    folded one (300 replications: rank R-hat ≥ 1.119, classic split R-hat ≤ 1.005)."""
    x = np.random.default_rng(5).standard_normal((4, 1000)) * np.array([[1.0], [3.0], [1.0], [1.0]])
    assert _gelman_rubin(_split(x)) < RHAT_THRESHOLD
    assert _bulk_rhat(x) < RHAT_THRESHOLD
    assert rhat(x) > 1.05


def test_folded_rhat_catches_a_chain_stuck_at_the_median():
    x = build("stuck_at_median")
    assert _bulk_rhat(x) < RHAT_THRESHOLD  # the stuck chain's location is right
    assert rhat(x) > 1.2


def test_split_rhat_detects_drift_within_chains():
    """Identical drifting chains agree with each other; only splitting them reveals the drift."""
    x = np.random.default_rng(21).standard_normal((4, 1000)) + np.linspace(-1.0, 1.0, 1000)
    assert _gelman_rubin(x) < RHAT_THRESHOLD  # unsplit chains look converged
    assert rhat(x) > 1.1


def test_single_chain_is_split_into_halves():
    """One chain still gives a split R-hat (ArviZ ≥ 1 returns nan; Stan's posterior does not)."""
    rng = np.random.default_rng(22)
    assert abs(rhat(rng.standard_normal((1, 2000))) - 1.0) < 0.005
    assert rhat(rng.standard_normal((1, 2000)) + np.linspace(-1.0, 1.0, 2000)) > 1.1


def test_rhat_is_fine_with_heavy_tails():
    """Rank normalisation: iid Cauchy chains (infinite variance) still look converged."""
    for seed in range(3):
        value = rhat(np.random.default_rng(60 + seed).standard_cauchy((4, 1000)))
        assert abs(value - 1.0) < 0.005


# --------------------------------------------------------------------------- ESS behaviour


def test_ess_iid_close_to_total_draws():
    """200 replications of 4x10000: ESS/N mean 0.993, s.d. 0.015, range 0.93–1.02."""
    for seed in range(3):
        value = ess_bulk(np.random.default_rng(70 + seed).standard_normal((4, 10_000)))
        np.testing.assert_allclose(value / 40_000, 1.0, atol=0.06)


def test_ess_iid_short_chains_close_to_total_draws():
    """300 replications of 4x1000: ESS/N mean 0.984, s.d. 0.043, range 0.84–1.08."""
    value = ess_bulk(np.random.default_rng(80).standard_normal((4, 1000)))
    np.testing.assert_allclose(value / 4000, 1.0, atol=0.15)


@pytest.mark.parametrize(
    ("rho", "atol"),
    [
        (0.5, 0.12),  # 200 replications: ratio mean 0.997, s.d. 0.031
        (0.9, 0.25),  # mean 0.987, s.d. 0.065
        (-0.3, 0.10),  # antithetic, ESS > N: mean 1.000, s.d. 0.027
    ],
)
def test_ess_ar1_matches_theory(rho, atol):
    """ESS ≈ N (1 − ρ) / (1 + ρ) for a Gaussian AR(1) (normal scores keep the autocorrelation)."""
    chains, draws = 4, 10_000
    x = ar1_chains(np.random.default_rng(90), rho, chains, draws)
    theory = chains * draws * (1.0 - rho) / (1.0 + rho)
    np.testing.assert_allclose(ess_bulk(x) / theory, 1.0, atol=atol)


def test_ess_decreases_with_autocorrelation():
    values = [
        ess_bulk(ar1_chains(np.random.default_rng(41), r, 4, 2000)) for r in (0, 0.3, 0.6, 0.9)
    ]
    assert np.all(np.diff(values) < 0), values


def test_ess_is_capped_at_n_log10_n():
    """Strongly antithetic chains hit the Vehtari et al. cap ``N log10 N`` exactly."""
    x = build("ar_-0.9")
    size = x.size
    np.testing.assert_allclose(ess_bulk(x), size * np.log10(size), rtol=1e-12)


def test_ess_of_chains_stuck_at_distinct_values_is_tiny():
    x = np.repeat(np.array([[0.0], [1.0], [2.0], [3.0]]), 500, axis=1)
    assert ess_bulk(x) < 10.0


# --------------------------------------------------------------------------- invariances


@pytest.mark.parametrize("case", ["iid", "ar_0.5", "scaled", "rwm"])
def test_invariant_to_increasing_affine_maps(case):
    """Exact for power-of-two scalings (exact in floating point). For a general affine map the
    folded half is invariant only up to rounding: the two draws adjacent to an even-sized median
    are tied in ``|x − median|`` in exact arithmetic, and ``3x − 7`` can break or create that tie
    (a two-rank change, which moved R-hat by 2e-6 on ``iid``)."""
    x = build(case)
    for y in (4.0 * x, x / 8.0):
        np.testing.assert_allclose(rhat(y), rhat(x), rtol=1e-12)
        np.testing.assert_allclose(ess_bulk(y), ess_bulk(x), rtol=1e-12)
    y = 3.0 * x - 7.0
    np.testing.assert_allclose(ess_bulk(y), ess_bulk(x), rtol=1e-12)
    np.testing.assert_allclose(rhat(y), rhat(x), rtol=0, atol=1e-5)


@pytest.mark.parametrize("case", ["iid", "ar_0.5", "scaled"])
def test_ess_bulk_invariant_to_monotone_transforms(case):
    """Bulk ESS depends on the draws only through their pooled ranks."""
    x = build(case)
    np.testing.assert_allclose(ess_bulk(np.exp(x)), ess_bulk(x), rtol=1e-12)
    np.testing.assert_allclose(ess_bulk(np.arctan(x)), ess_bulk(x), rtol=1e-12)


@pytest.mark.parametrize("case", ["iid", "shifted", "scaled", "ar_0.9"])
def test_invariant_to_negation(case):
    x = build(case)
    np.testing.assert_allclose(rhat(-x), rhat(x), rtol=1e-9)
    np.testing.assert_allclose(ess_bulk(-x), ess_bulk(x), rtol=1e-9)


@pytest.mark.parametrize("case", ["shifted", "scaled", "many_chains"])
def test_invariant_to_chain_order(case):
    x = build(case)
    perm = np.random.default_rng(0).permutation(x.shape[0])
    np.testing.assert_allclose(rhat(x[perm]), rhat(x), rtol=1e-12)
    np.testing.assert_allclose(ess_bulk(x[perm]), ess_bulk(x), rtol=1e-12)


# --------------------------------------------------------------------------- inputs and edge cases


def test_inputs_are_not_modified_and_lists_and_integers_are_accepted():
    x = build("rwm")
    before = x.copy()
    rhat(x), ess_bulk(x)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(rhat(x.tolist()), rhat(x), rtol=0)
    counts = np.random.default_rng(1).poisson(3.0, (4, 200))
    np.testing.assert_allclose(rhat(counts), rhat(counts.astype(np.float64)), rtol=0)
    np.testing.assert_allclose(ess_bulk(counts), ess_bulk(counts.astype(np.float64)), rtol=0)


@pytest.mark.parametrize("draws", [0, 1, 2, 3])
def test_too_short_for_rhat(draws):
    """Fewer than four draws (split halves of one draw): undefined, as in ArviZ."""
    x = np.random.default_rng(draws).standard_normal((4, draws))
    assert np.isnan(rhat(x))
    assert np.isnan(ess_bulk(x))


@pytest.mark.parametrize("draws", [4, 5, 6, 7])
def test_too_short_for_ess(draws):
    """Split halves of 2–3 draws carry no autocorrelation information: ESS is nan (ArviZ returns
    its cap ``N log10 N`` instead), while R-hat is already defined."""
    x = np.random.default_rng(draws).standard_normal((4, draws))
    assert np.isnan(ess_bulk(x))
    assert np.isfinite(rhat(x))


@pytest.mark.parametrize("value", [0.0, 1.0, 1 / 3, -7.3, 1e10, np.pi])
@pytest.mark.parametrize(("chains", "draws"), [(4, 1000), (4, 1001), (3, 999), (2, 4), (1, 50)])
def test_constant_chains_do_not_raise_or_warn(value, chains, draws):
    x = np.full((chains, draws), value)
    r, e = rhat(x), ess_bulk(x)  # RuntimeWarning is an error in this suite
    assert np.isnan(r) or r == 1.0
    assert np.isnan(e) or e == x.size  # nan here; ArviZ returns the number of draws


@pytest.mark.parametrize("value", [0.0, 0.25, -3.0])
def test_constant_chains_never_count_as_converged(value):
    """Chains that never move carry no evidence of mixing. ArviZ (SPEC §10.6) and Stan's
    ``posterior`` return nan/NA, which the analytic engine's flag (``isfinite(r) and r < 1.01``)
    treats as not converged; a never-accepting MH run from a common start must not pass."""
    x = np.full((4, 1000), value)
    assert not rhat(x) < RHAT_THRESHOLD


def test_chains_stuck_at_distinct_values_are_not_converged():
    x = np.repeat(np.array([[0.0], [1.0], [2.0], [3.0]]), 500, axis=1)
    assert not rhat(x) < RHAT_THRESHOLD


def test_one_stuck_chain_among_mixing_chains_is_not_converged():
    x = np.random.default_rng(23).standard_normal((4, 1000))
    x[0] = x[0, 0]  # stuck at its starting value
    assert rhat(x) > RHAT_THRESHOLD


def test_nan_draws_give_nan():
    x = build("iid")
    x[2, 17] = np.nan
    assert np.isnan(rhat(x))
    assert np.isnan(ess_bulk(x))


# --------------------------------------------------------------------------- nightly (slow)


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, -0.3])
def test_ess_ar1_unbiased_over_replications(rho):
    """Mean of ESS / theory over 200 replications of 4x2000 (observed: 0.986–1.014, s.e. ≤ 0.009)."""
    reps, chains, draws = 200, 4, 2000
    theory = chains * draws * (1.0 - rho) / (1.0 + rho)
    ratios = [
        ess_bulk(ar1_chains(np.random.default_rng(10_000 + r), rho, chains, draws)) / theory
        for r in range(reps)
    ]
    assert abs(np.mean(ratios) - 1.0) < 0.03


@pytest.mark.slow
def test_rhat_false_alarm_rate_and_power():
    """200 replications of 4x1000: iid chains never fail the threshold; a 0.5-s.d. shift in one
    chain or a doubled s.d. is always flagged (300 replications: minima 1.016 and 1.050)."""
    reps = 200
    iid = shift = scale = 0
    for r in range(reps):
        x = np.random.default_rng(20_000 + r).standard_normal((4, 1000))
        iid += rhat(x) >= RHAT_THRESHOLD
        shift += rhat(x + np.array([[0.0], [0.0], [0.0], [0.5]])) >= RHAT_THRESHOLD
        scale += rhat(x * np.array([[1.0], [1.0], [1.0], [2.0]])) >= RHAT_THRESHOLD
    assert iid <= 0.01 * reps
    assert shift >= 0.99 * reps
    assert scale >= 0.99 * reps
