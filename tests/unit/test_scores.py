"""Scores for one forecast against realised outcomes (SPEC §9.2)."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scoringrules as sr
from scipy import integrate, stats

from rai_bvar.evaluation.scores import QW_WEIGHTS, quantile_weighted_crps, score_forecast
from rai_bvar.forecast import ForecastResult
from rai_bvar.models import OLSVAR

PER_VARIABLE = {
    "lpd_marginal",
    "crps",
    "qwcrps_center",
    "qwcrps_left",
    "qwcrps_right",
    "sq_err",
    "abs_err",
    "pit",
    "cover50",
    "cover90",
}
JOINT = {"lpd", "lpd_mcse", "energy", "variogram"}


def gaussian_crps(y, mu, sigma):
    """Closed form (Gneiting & Raftery 2007): σ[z(2Φ(z) − 1) + 2φ(z) − 1/√π]."""
    z = (np.asarray(y) - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def fair_crps(y: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """scoringrules' fair CRPS, one forecast at a time (its NumPy backend needs S² memory)."""
    return np.array(
        [sr.crps_ensemble(yi, ei, estimator="fair") for yi, ei in zip(y, ens, strict=True)]
    )


def stratified_normal(size: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """An ensemble at the quantiles (i − ½)/S of N(μ, σ²): a near-exact representation."""
    return mu + sigma * stats.norm.ppf((np.arange(size) + 0.5) / size)


def gaussian_result(mu: np.ndarray, sigma: np.ndarray, draws: int, seed: int = 0) -> ForecastResult:
    """Independent Gaussian marginals with stratified, randomly paired draws; mu is (H, n)."""
    rng = np.random.default_rng(seed)
    H, n = mu.shape
    paths = np.empty((draws, H, n))
    for h in range(H):
        for j in range(n):
            paths[:, h, j] = rng.permutation(stratified_normal(draws, mu[h, j], sigma[j]))
    return ForecastResult.from_arrays(
        paths=paths,
        cond_mean=np.broadcast_to(mu, (draws, H, n)),
        cond_chol=np.broadcast_to(np.diag(sigma), (draws, H, n, n)),
        variables=[f"v{j}" for j in range(n)],
        origin=pd.Period("2010Q4", "Q"),
    )


@pytest.fixture(scope="module")
def var_forecast(sim_data_stationary):
    """An OLS-VAR(2) forecast from 2015Q4 with the realised values that followed."""
    origin = pd.Period("2015Q4", "Q")
    fr = OLSVAR(2).fit(sim_data_stationary.upto(origin)).forecast(8, draws=600, seed=4)
    return fr, sim_data_stationary.realized(origin, 8)


# --------------------------------------------------------------------------- quantile-weighted CRPS


@pytest.mark.parametrize(
    "sampler",
    [
        lambda r, s: r.normal(1.0, 2.0, s),
        lambda r, s: r.lognormal(0.0, 0.8, s),
        lambda r, s: r.standard_t(4, s),
    ],
    ids=["normal", "lognormal", "t4"],
)
def test_unweighted_qwcrps_matches_fair_crps(sampler):
    """With w ≡ 1 the quantile-score integral is the CRPS.

    The 99 midpoint levels stop at α = 0.005, which costs up to ~2e-3 (relative) for t₄ tails.
    """
    draws = sampler(np.random.default_rng(0), 3000)
    ys = np.array([-4.0, -1.0, 0.0, 0.5, 1.0, 2.5, 7.0])
    ens = np.broadcast_to(draws, (ys.size, draws.size))
    ours = quantile_weighted_crps(ens, ys)
    np.testing.assert_allclose(ours, fair_crps(ys, ens), rtol=5e-3)
    np.testing.assert_allclose(
        quantile_weighted_crps(ens, ys, lambda a: np.ones_like(a)), ours, rtol=1e-14
    )


def test_crps_against_gaussian_closed_form():
    mu, sigma = 0.5, 1.5
    ens = stratified_normal(20_000, mu, sigma)
    ys = np.array([-5.0, -1.0, 0.5, 1.2, 3.0, 6.0])
    exact = gaussian_crps(ys, mu, sigma)
    np.testing.assert_allclose(
        exact, sr.crps_normal(ys, mu, sigma), rtol=1e-12
    )  # the oracle itself
    batch = np.broadcast_to(ens, (ys.size, ens.size))
    np.testing.assert_allclose(quantile_weighted_crps(batch, ys), exact, rtol=3e-3)
    small = np.broadcast_to(stratified_normal(3000, mu, sigma), (ys.size, 3000))
    np.testing.assert_allclose(fair_crps(ys, small), exact, rtol=1e-3)


@pytest.mark.parametrize("weight", ["center", "left", "right"])
def test_weighted_qwcrps_against_numerical_integral(weight):
    """2∫ QS_α(F⁻¹(α), y) w(α) dα with the exact Gaussian quantile function."""
    mu, sigma = 1.0, 2.0
    w = QW_WEIGHTS[weight]
    ens = stratified_normal(20_000, mu, sigma)
    for y in (-4.0, 0.0, 1.0, 2.5, 7.0):

        def integrand(a, y=y):
            q = mu + sigma * stats.norm.ppf(a)
            return 2.0 * ((y <= q) - a) * (q - y) * w(a)

        exact, _ = integrate.quad(
            integrand, 0.0, 1.0, points=[stats.norm.cdf((y - mu) / sigma)], limit=200
        )
        ours = quantile_weighted_crps(ens, np.array(y), weight)
        np.testing.assert_allclose(ours, exact, rtol=5e-3)


def test_tail_weights_decompose_the_crps():
    """(1 − α)² + α² + 2α(1 − α) = 1: left + right + 2·centre equals the unweighted score."""
    rng = np.random.default_rng(1)
    ens = rng.normal(size=(4, 3, 500))
    y = rng.normal(size=(4, 3))
    parts = {k: quantile_weighted_crps(ens, y, k) for k in QW_WEIGHTS}
    np.testing.assert_allclose(
        parts["left"] + parts["right"] + 2 * parts["center"], quantile_weighted_crps(ens, y)
    )
    assert all(np.all(p >= 0) for p in parts.values())


def test_tail_weights_emphasise_their_tail():
    ens = stratified_normal(5001)  # symmetric about 0

    def score(y, weight):
        return quantile_weighted_crps(ens, np.array(y), weight)

    assert score(-3.0, "left") > 1.5 * score(-3.0, "right")
    assert score(3.0, "right") > 1.5 * score(3.0, "left")
    # mirror symmetry: the left-tail score of -y equals the right-tail score of y
    assert score(-3.0, "left") == pytest.approx(score(3.0, "right"), rel=1e-10)
    assert score(-0.7, "center") == pytest.approx(score(0.7, "center"), rel=1e-10)


# --------------------------------------------------------------------------- score_forecast: layout


def test_output_layout(var_forecast):
    fr, realized = var_forecast
    ds = score_forecast(fr, realized, rng=np.random.default_rng(0))
    assert set(ds.data_vars) == PER_VARIABLE | JOINT
    for name in PER_VARIABLE:
        assert ds[name].dims == ("horizon", "variable")
    for name in JOINT:
        assert ds[name].dims == ("horizon",)
    np.testing.assert_array_equal(ds["horizon"].values, np.arange(1, 9))
    assert list(ds["variable"].values) == ["gdp", "prices", "rate"]
    for name in ds.data_vars:
        assert ds[name].dtype == np.float64
        assert np.isfinite(ds[name].values).all(), name


def test_realized_shape_is_checked(var_forecast):
    fr, realized = var_forecast
    with pytest.raises(ValueError, match="realized must have shape"):
        score_forecast(fr, realized[:4])
    with pytest.raises(ValueError, match="realized must have shape"):
        score_forecast(fr, realized[:, :2])


def test_scores_match_their_definitions(var_forecast):
    fr, realized = var_forecast
    ds = score_forecast(fr, realized, rng=np.random.default_rng(0))
    draws = fr.stacked()  # (S, H, n)
    ens = np.moveaxis(draws, 0, -1)
    np.testing.assert_allclose(
        ds["crps"].values, sr.crps_ensemble(realized, ens, estimator="fair"), rtol=1e-12
    )
    np.testing.assert_allclose(
        ds["sq_err"].values, (draws.mean(axis=0) - realized) ** 2, rtol=1e-12
    )
    np.testing.assert_allclose(
        ds["abs_err"].values, np.abs(np.median(draws, axis=0) - realized), rtol=1e-12
    )
    np.testing.assert_allclose(ds["pit"].values, (draws <= realized).mean(axis=0))
    for name, (lo, hi) in {"cover50": (0.25, 0.75), "cover90": (0.05, 0.95)}.items():
        q = np.quantile(draws, [lo, hi], axis=0)
        np.testing.assert_array_equal(
            ds[name].values, ((q[0] <= realized) & (realized <= q[1])).astype(float)
        )
    for k in QW_WEIGHTS:
        np.testing.assert_allclose(
            ds[f"qwcrps_{k}"].values, quantile_weighted_crps(ens, realized, k), rtol=1e-12
        )
    for h in range(fr.horizon):
        lpd, mcse = fr.log_pred_density(realized[h], h + 1, return_mcse=True)
        assert ds["lpd"].values[h] == pytest.approx(lpd, rel=1e-12)
        assert ds["lpd_mcse"].values[h] == pytest.approx(mcse, rel=1e-12)
        for j, var in enumerate(fr.variables):
            expected = fr.log_pred_density(realized[h, j : j + 1], h + 1, [var])
            assert ds["lpd_marginal"].values[h, j] == pytest.approx(expected, rel=1e-12)
    # with every draw used, the multivariate scores are the scoringrules ensemble scores
    sub = np.moveaxis(draws, 0, 1)  # (H, S, n)
    np.testing.assert_allclose(
        ds["energy"].values, sr.es_ensemble(realized, sub, estimator="fair"), rtol=1e-10
    )
    np.testing.assert_allclose(
        ds["variogram"].values, sr.vs_ensemble(realized, sub, p=0.5), rtol=1e-10
    )


def test_gaussian_forecast_scores_are_exact():
    """Stratified Gaussian draws: LPD, CRPS and PIT agree with their closed forms."""
    mu = np.array([[0.0, 2.0], [0.5, 1.0], [1.0, -1.0]])
    sigma = np.array([1.0, 0.5])
    fr = gaussian_result(mu, sigma, draws=4000)
    realized = np.array([[0.3, 2.2], [-1.0, 1.6], [3.5, -1.1]])
    ds = score_forecast(fr, realized, rng=np.random.default_rng(0))
    z = (realized - mu) / sigma
    np.testing.assert_allclose(
        ds["lpd_marginal"].values, stats.norm.logpdf(realized, mu, sigma), rtol=1e-12
    )
    np.testing.assert_allclose(
        ds["lpd"].values, stats.norm.logpdf(realized, mu, sigma).sum(axis=1), rtol=1e-12
    )
    np.testing.assert_allclose(ds["lpd_mcse"].values, 0.0, atol=1e-12)
    np.testing.assert_allclose(ds["crps"].values, gaussian_crps(realized, mu, sigma), rtol=2e-3)
    np.testing.assert_allclose(ds["pit"].values, stats.norm.cdf(z), atol=1e-3)
    np.testing.assert_array_equal(ds["cover90"].values, (np.abs(z) < 1.6449).astype(float))
    np.testing.assert_array_equal(ds["cover50"].values, (np.abs(z) < 0.6745).astype(float))


# --------------------------------------------------------------------------- missing outcomes


def test_ragged_realized(var_forecast):
    fr, realized = var_forecast
    ragged = realized.copy()
    ragged[2, 1] = np.nan  # one variable missing at h = 3
    ragged[6:] = np.nan  # not yet observed at h = 7, 8
    ds = score_forecast(fr, ragged, rng=np.random.default_rng(0))
    full = score_forecast(fr, realized, rng=np.random.default_rng(0))
    observed = np.isfinite(ragged)
    complete = observed.all(axis=1)
    for name in PER_VARIABLE:
        values = ds[name].values
        assert np.isnan(values[~observed]).all(), name
        np.testing.assert_allclose(
            values[observed], full[name].values[observed], rtol=1e-12, err_msg=name
        )
    for name in JOINT:
        values = ds[name].values
        assert np.isnan(values[~complete]).all(), name
        np.testing.assert_allclose(
            values[complete], full[name].values[complete], rtol=1e-12, err_msg=name
        )
    # the marginal LPD of an observed variable does not need the missing one
    assert ds["lpd_marginal"].values[2, 0] == pytest.approx(
        fr.log_pred_density(ragged[2, :1], 3, ["gdp"])
    )


def test_all_missing_gives_all_nan(var_forecast):
    fr, realized = var_forecast
    ds = score_forecast(fr, np.full_like(realized, np.nan))
    for name in ds.data_vars:
        assert np.isnan(ds[name].values).all(), name


# --------------------------------------------------------------------------- individual scores


def test_energy_score_in_one_dimension_equals_crps(sim_data_stationary):
    data = sim_data_stationary.select(["rate"]).upto("2012Q2")
    fr = OLSVAR(2).fit(data).forecast(5, draws=800, seed=6)
    realized = sim_data_stationary.select(["rate"]).realized("2012Q2", 5)
    ds = score_forecast(fr, realized, rng=np.random.default_rng(0))
    np.testing.assert_allclose(ds["energy"].values, ds["crps"].values[:, 0], rtol=1e-10)
    np.testing.assert_allclose(ds["lpd"].values, ds["lpd_marginal"].values[:, 0], rtol=1e-12)
    np.testing.assert_allclose(ds["variogram"].values, 0.0, atol=1e-15)  # no pairs of variables


def test_multivariate_scores_subsample_reproducibly(var_forecast):
    fr, realized = var_forecast
    a = score_forecast(fr, realized, rng=np.random.default_rng(1), multivariate_max_samples=100)
    b = score_forecast(fr, realized, rng=np.random.default_rng(1), multivariate_max_samples=100)
    full = score_forecast(fr, realized, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a["energy"].values, b["energy"].values)
    np.testing.assert_array_equal(a["crps"].values, full["crps"].values)  # univariate: all draws
    assert not np.allclose(a["energy"].values, full["energy"].values, rtol=1e-6)
    every = score_forecast(
        fr, realized, rng=np.random.default_rng(2), multivariate_max_samples=fr.n_samples
    )
    np.testing.assert_array_equal(every["energy"].values, full["energy"].values)


def test_pit_is_within_unit_interval(var_forecast):
    fr, realized = var_forecast
    shifted = realized.copy()
    shifted[0] = -1e6  # below every draw
    shifted[1] = 1e6  # above every draw
    pit = score_forecast(fr, shifted, rng=np.random.default_rng(0))["pit"].values
    assert np.all((pit >= 0) & (pit <= 1))
    np.testing.assert_array_equal(pit[0], 0.0)
    np.testing.assert_array_equal(pit[1], 1.0)
    assert np.all((pit[2:] > 0) & (pit[2:] < 1))


def test_twcrps_only_with_bounds(var_forecast):
    fr, realized = var_forecast
    assert "twcrps" not in score_forecast(fr, realized, rng=np.random.default_rng(0))
    ds = score_forecast(
        fr, realized, rng=np.random.default_rng(0), tw_bounds={"gdp": (-np.inf, 0.005)}
    )
    assert ds["twcrps"].dims == ("horizon", "variable")
    # unbounded variables: the threshold weight is 1 everywhere, so twCRPS = CRPS
    np.testing.assert_allclose(ds["twcrps"].values[:, 1:], ds["crps"].values[:, 1:], rtol=1e-12)
    ens = np.moveaxis(fr.stacked()[..., 0], 0, -1)
    expected = sr.twcrps_ensemble(realized[:, 0], ens, b=0.005, estimator="fair")
    np.testing.assert_allclose(ds["twcrps"].values[:, 0], expected, rtol=1e-12)
    assert not np.allclose(ds["twcrps"].values[:, 0], ds["crps"].values[:, 0])


def test_twcrps_respects_missing_values(var_forecast):
    fr, realized = var_forecast
    ragged = realized.copy()
    ragged[3, 0] = np.nan
    ds = score_forecast(fr, ragged, rng=np.random.default_rng(0), tw_bounds={"gdp": (0.0, np.inf)})
    assert np.isnan(ds["twcrps"].values[3, 0])
    assert np.isfinite(ds["twcrps"].values[3, 1:]).all()


def test_memory_is_not_quadratic_in_the_number_of_draws():
    """CRPS and twCRPS sort the ensemble (O(S log S), SPEC §9.2). Only the energy score has a
    pairwise term, on at most ``multivariate_max_samples`` draws and one horizon at a time."""
    fr = gaussian_result(np.zeros((3, 2)), np.ones(2), draws=6000)
    tracemalloc.start()
    try:
        score_forecast(
            fr, np.zeros((3, 2)), rng=np.random.default_rng(0), tw_bounds={"v0": (0.0, np.inf)}
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100e6  # an S x S temporary for 6 series would need ~1.7 GB


# --------------------------------------------------------------------------- calibration (nightly)


@pytest.mark.slow
def test_flat_prior_forecasts_are_calibrated(simulate_var):
    """Over many fresh samples from the true VAR(1), PITs are uniform and coverage is nominal."""
    from rai_bvar.data import VARData

    reps, T, H = 300, 100, 4
    pits, cover90, cover50 = [], [], []
    index = pd.period_range("1980Q1", periods=T, freq="Q")
    for r in range(reps):
        y = simulate_var(T + H, seed=1000 + r)
        data = VARData.from_frame(
            pd.DataFrame(y[:T], index=index, columns=["gdp", "prices", "rate"])
        )
        fr = OLSVAR(1).fit(data).forecast(H, draws=500, seed=r)
        ds = score_forecast(fr, y[T:], rng=np.random.default_rng(r))
        pits.append(ds["pit"].values)
        cover90.append(ds["cover90"].values)
        cover50.append(ds["cover50"].values)
    pits, cover90, cover50 = np.array(pits), np.array(cover90), np.array(cover50)  # (reps, H, n)
    for h in range(H):
        for j in range(3):
            assert stats.kstest(pits[:, h, j], "uniform").pvalue > 1e-4
    # binomial standard error of a coverage rate over reps × variables
    for rate, nominal in ((cover90, 0.9), (cover50, 0.5)):
        se = np.sqrt(nominal * (1 - nominal) / (reps * 3))
        assert np.all(np.abs(rate.mean(axis=(0, 2)) - nominal) < 4 * se)
