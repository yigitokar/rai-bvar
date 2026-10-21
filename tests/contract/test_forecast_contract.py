"""The predictive contract (SPEC §7), checked for every registered model composition.

Every model in ``REGISTERED_MODELS`` must produce a ``ForecastResult`` whose paths are joint
draws consistent with the stored conditional Gaussian moments. Analytic checks compare the
Rao-Blackwellised log predictive density with closed forms where they exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rai_bvar import __version__
from rai_bvar.forecast import ContractError, ForecastResult, check_structure, simulation_consistency
from rai_bvar.models import OLSVAR, FittedModel, Model, RandomWalk, spec_hash

REGISTERED_MODELS = [
    RandomWalk(),
    RandomWalk(drift=True),
    OLSVAR(1),
    OLSVAR(2),
    OLSVAR(2, uncertainty="none"),
    OLSVAR(1, deterministic="ct"),
]

H = 12  # > number of regressors of every registered model (needed by the joint-path check)
DRAWS = 1000
SEED = 20261021


def model_id(model) -> str:
    spec = model.spec()
    extras = [f"{k}={v}" for k, v in spec.items() if k not in ("class", "lags")]
    return "-".join([model.name, *extras])


def _is_plug_in(model) -> bool:
    return isinstance(model, RandomWalk) or (
        isinstance(model, OLSVAR) and model.uncertainty == "none"
    )


def _is_flat_ols(model) -> bool:
    return isinstance(model, OLSVAR) and model.uncertainty == "flat"


PLUG_IN_MODELS = [m for m in REGISTERED_MODELS if _is_plug_in(m)]
FLAT_MODELS = [m for m in REGISTERED_MODELS if _is_flat_ols(m)]  # Jeffreys-prior posterior draws


# --------------------------------------------------------------------------- independent oracles


def _design_terms(lags: int, deterministic: str) -> tuple[int, int]:
    return lags, {"n": 0, "c": 1, "ct": 2}[deterministic]


def _regressor(lagged: list[np.ndarray], t: float, n_det: int) -> np.ndarray:
    """``(1, t)[:n_det]`` followed by the lag-1 … lag-p values (each ``(..., n)``)."""
    lead = lagged[0].shape[:-1]
    det = np.broadcast_to(np.array([1.0, t])[:n_det], (*lead, n_det))
    return np.concatenate([det, *lagged], axis=-1)


def _ols(values: np.ndarray, lags: int, deterministic: str):
    """Least squares written out directly (no rai_bvar code): ``Y``, ``X``, ``B̂``, residual SSCP."""
    p, n_det = _design_terms(lags, deterministic)
    T = values.shape[0]
    X = np.stack(
        [_regressor([values[t - j] for j in range(1, p + 1)], t + 1.0, n_det) for t in range(p, T)]
    )
    Y = values[p:]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    return Y, X, B, E.T @ E


def _forecast_regressors(
    values: np.ndarray, paths: np.ndarray, lags: int, deterministic: str
) -> np.ndarray:
    """``x_{s,T+h}`` built from the data and the simulated path up to ``T+h−1``: ``(S, H, k)``."""
    p, n_det = _design_terms(lags, deterministic)
    S, horizon, n = paths.shape
    T = values.shape[0]
    full = np.concatenate([np.broadcast_to(values[-p:], (S, p, n)), paths], axis=1)
    return np.stack(
        [
            _regressor([full[:, p + h - j] for j in range(1, p + 1)], T + h + 1.0, n_det)
            for h in range(horizon)
        ],
        axis=1,
    )


def _model_design(model, fitted) -> tuple[int, str]:
    """Lags and deterministic terms of a linear conditional mean (skip for other models)."""
    if isinstance(model, RandomWalk):
        return 1, "c" if model.drift else "n"
    if isinstance(model, OLSVAR):
        return model.lags, model.deterministic
    lags, deterministic = getattr(fitted, "lags", None), getattr(fitted, "deterministic", None)
    if lags is None or deterministic is None:
        pytest.skip("the joint-path check needs a VAR(p) conditional mean")
    return lags, deterministic


def _exact_plug_in_moments(model, data):
    """Mean and covariance of ``y_{T+1}`` for the plug-in models."""
    values = data.values
    if isinstance(model, RandomWalk):
        dy = np.diff(values, axis=0)
        mean = values[-1] + (dy.mean(axis=0) if model.drift else 0.0)
        return mean, np.cov(dy, rowvar=False)
    _, X, B, S = _ols(values, model.lags, model.deterministic)
    p, n_det = _design_terms(model.lags, model.deterministic)
    x = _regressor([values[-j] for j in range(1, p + 1)], len(values) + 1.0, n_det)
    return B.T @ x, S / (X.shape[0] - X.shape[1])


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module", params=REGISTERED_MODELS, ids=model_id)
def model(request):
    return request.param


@pytest.fixture(scope="module")
def fitted(model, sim_data):
    return model.fit(sim_data)


@pytest.fixture(scope="module")
def forecast(fitted) -> ForecastResult:
    return fitted.forecast(H, draws=DRAWS, seed=SEED)


# --------------------------------------------------------------------------- invariants for every model


def test_model_protocol(model, fitted):
    assert isinstance(model, Model)
    assert isinstance(fitted, FittedModel)
    assert spec_hash(model) == spec_hash(
        type(model)(**{k: v for k, v in model.spec().items() if k != "class"})
    )
    assert len({spec_hash(m) for m in REGISTERED_MODELS}) == len(REGISTERED_MODELS)


def test_check_structure_passes(forecast):
    check_structure(forecast)


def test_coordinates_and_attributes(forecast, model, sim_data):
    ds = forecast.dataset
    assert forecast.variables == list(sim_data.variables)
    assert forecast.horizon == H
    assert forecast.origin == sim_data.last
    assert forecast.n_samples == DRAWS
    assert list(ds["date"].values) == [str(sim_data.last + h) for h in range(1, H + 1)]
    assert ds["date"].values[0] == "2020Q1"
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        assert ds[name].dtype == np.float64
    assert ds.attrs["model"] == model.name
    assert ds.attrs["engine"]
    if isinstance(model, RandomWalk | OLSVAR):
        assert ds.attrs["engine"] == ("plug-in" if _is_plug_in(model) else "analytic")
    assert ds.attrs["rai_bvar_version"] == __version__
    assert '"gdp": 5' in ds.attrs["tcodes"]


def test_simulation_consistency(forecast):
    """Invariant 1: L⁻¹(paths − cond_mean) are iid N(0, 1)."""
    assert simulation_consistency(forecast) > 1e-3


def test_stacked_shapes(forecast):
    """Invariant 3: stacking (chain, draw) into one sample axis gives (samples, H, n)."""
    assert forecast.stacked().shape == (DRAWS, H, 3)
    assert forecast.stacked("cond_mean").shape == (DRAWS, H, 3)
    assert forecast.stacked("cond_chol").shape == (DRAWS, H, 3, 3)
    assert forecast.stacked("paths_level").shape == (DRAWS, H, 3)


def test_same_seed_gives_identical_paths(fitted, forecast):
    again = fitted.forecast(H, draws=DRAWS, seed=SEED)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(again.dataset[name].values, forecast.dataset[name].values)
    from_generator = fitted.forecast(H, draws=DRAWS, seed=np.random.default_rng(SEED))
    np.testing.assert_array_equal(from_generator.stacked(), forecast.stacked())
    other = fitted.forecast(H, draws=DRAWS, seed=SEED + 1)
    assert not np.array_equal(other.stacked(), forecast.stacked())


def test_paths_level_is_the_pathwise_inverse(forecast, sim_data, sim_raw):
    np.testing.assert_allclose(
        forecast.stacked("paths_level"), sim_data.inverse_paths(forecast.stacked()), rtol=1e-14
    )
    # spelled out for the first step: Δlog for gdp/prices, level for the rate
    paths, level = forecast.stacked()[:, 0], forecast.stacked("paths_level")[:, 0]
    last = sim_raw.iloc[-1].to_numpy()
    np.testing.assert_allclose(level[:, :2], last[:2] * np.exp(paths[:, :2]), rtol=1e-13)
    np.testing.assert_array_equal(level[:, 2], paths[:, 2])


def test_joint_paths(model, fitted, forecast, sim_data):
    """Invariant 2: cond_mean at T+h is a fixed function of the draw and the simulated path to T+h−1.

    For linear VAR(p) means, ``cond_mean[s, h] = B_s' x_{s,h}`` with ``x_{s,h}`` built from the
    simulated path; recovering ``B_s`` from all ``H > k`` horizons must leave no residual. Paths
    assembled from marginals, or a mean that ignores the simulated path, fail this check.
    """
    lags, deterministic = _model_design(model, fitted)
    paths, mean = forecast.stacked(), forecast.stacked("cond_mean")
    X = _forecast_regressors(sim_data.values, paths, lags, deterministic)  # (S, H, k)
    assert X.shape[2] < H
    coefs = np.linalg.pinv(X) @ mean  # per-draw least squares, (S, k, n)
    scale = np.abs(mean).max(axis=(0, 1))  # per variable
    np.testing.assert_allclose(np.abs(X @ coefs - mean).max(axis=(0, 1)) / scale, 0.0, atol=1e-10)

    if isinstance(model, RandomWalk) and not model.drift:
        np.testing.assert_array_equal(mean[:, 0], np.broadcast_to(sim_data.values[-1], (DRAWS, 3)))
        np.testing.assert_array_equal(mean[:, 1:], paths[:, :-1])
    elif isinstance(model, RandomWalk):
        drift = np.diff(sim_data.values, axis=0).mean(axis=0)
        np.testing.assert_allclose(mean[:, 1:], paths[:, :-1] + drift, rtol=1e-12, atol=1e-15)
    elif _is_plug_in(model):
        _, _, B, _ = _ols(sim_data.values, lags, deterministic)
        np.testing.assert_allclose(mean, X @ B, rtol=1e-9, atol=1e-12)
    elif _is_flat_ols(model):
        # flat prior: B_s varies over draws, centred on the OLS estimate
        _, _, B, _ = _ols(sim_data.values, lags, deterministic)
        se = coefs.std(axis=0) / np.sqrt(DRAWS)
        assert np.all(coefs.std(axis=0) > 0)
        assert np.all(np.abs(coefs.mean(axis=0) - B) < 5 * se)


@pytest.mark.parametrize(
    "plug_in",
    [
        OLSVAR(1, deterministic="ct", uncertainty="none"),
        OLSVAR(3, deterministic="n", uncertainty="none"),
    ],
    ids=model_id,
)
def test_plug_in_paths_recomputed_from_ols(plug_in, sim_data):
    """Exact recomputation, including the trend continuing past the sample end (T+1, T+2, …)."""
    fr = plug_in.fit(sim_data).forecast(6, draws=50, seed=1)
    _, _, B, _ = _ols(sim_data.values, plug_in.lags, plug_in.deterministic)
    X = _forecast_regressors(sim_data.values, fr.stacked(), plug_in.lags, plug_in.deterministic)
    np.testing.assert_allclose(fr.stacked("cond_mean"), X @ B, rtol=1e-9, atol=1e-12)


def test_flat_prior_coefficient_spread(sim_data):
    """Posterior sd of B under the flat prior: Var(B_ij) = E[Σ_jj] (X'X)⁻¹_ii, E[Σ] = S/(T−k−n−1)."""
    model = OLSVAR(1)
    draws = 4000
    fr = model.fit(sim_data).forecast(H, draws=draws, seed=5)
    X = _forecast_regressors(sim_data.values, fr.stacked(), 1, "c")
    coefs = np.linalg.pinv(X) @ fr.stacked("cond_mean")
    _, Xs, _, S = _ols(sim_data.values, 1, "c")
    T, k = Xs.shape
    expected = np.outer(np.diag(np.linalg.inv(Xs.T @ Xs)), np.diag(S) / (T - k - 3 - 1))
    ratio = coefs.var(axis=0) / expected
    # the sampling sd of each variance ratio is about √(2/4000) ≈ 0.022
    assert np.all(np.abs(ratio - 1.0) < 0.12), ratio


# --------------------------------------------------------------------------- analytic predictive densities


@pytest.mark.parametrize("model", PLUG_IN_MODELS, ids=model_id)
def test_plug_in_one_step_lpd_is_exact_gaussian(model, sim_data):
    fr = model.fit(sim_data).forecast(3, draws=200, seed=3)
    mean, cov = _exact_plug_in_moments(model, sim_data)
    sd = np.sqrt(np.diag(cov))
    for y in (mean, mean + sd * np.array([1.0, -2.0, 0.5]), mean - 3.0 * sd):
        exact = stats.multivariate_normal(mean, cov).logpdf(y)
        np.testing.assert_allclose(fr.log_pred_density(y, 1), exact, rtol=1e-10)
        for subset in (["rate"], ["prices", "gdp"]):
            idx = [sim_data.variables.index(v) for v in subset]
            marginal = stats.multivariate_normal(mean[idx], cov[np.ix_(idx, idx)]).logpdf(y[idx])
            np.testing.assert_allclose(fr.log_pred_density(y[idx], 1, subset), marginal, rtol=1e-10)
            np.testing.assert_allclose(fr.log_pred_density(y, 1, subset), marginal, rtol=1e-10)
    _, mcse = fr.log_pred_density(mean, 1, return_mcse=True)
    assert mcse == pytest.approx(0.0, abs=1e-12)  # every draw has the same density


@pytest.mark.parametrize("window", [None, 40], ids=["full sample", "last 40 obs"])
@pytest.mark.parametrize("model", FLAT_MODELS, ids=model_id)
def test_flat_prior_one_step_lpd_is_multivariate_t(model, sim_data, window):
    """y_{T+1} | Y ~ t_ν(B̂'x, S(1 + x'(X'X)⁻¹x)/ν) with ν = T − k − n + 1 (Jeffreys prior).

    The Rao-Blackwellised estimate must match within 4 Monte Carlo standard errors (the delta-method
    MCSE returned by ``log_pred_density``; its bias is O(1/S) and negligible at 20k draws).
    """
    data = sim_data if window is None else sim_data.tail(window)
    fr = model.fit(data).forecast(1, draws=20_000, seed=11)
    _, X, B, S = _ols(data.values, model.lags, model.deterministic)
    T, k = X.shape
    n = data.n_vars
    nu = T - k - n + 1
    p, n_det = _design_terms(model.lags, model.deterministic)
    x = _regressor([data.values[-j] for j in range(1, p + 1)], len(data) + 1.0, n_det)
    loc = B.T @ x
    shape = S * (1.0 + x @ np.linalg.solve(X.T @ X, x)) / nu
    sd = np.sqrt(np.diag(shape))
    exact_dist = stats.multivariate_t(loc=loc, shape=shape, df=nu)
    for y in (loc, loc + sd * np.array([1.0, -0.5, 1.5]), loc + 2.5 * sd):
        lpd, mcse = fr.log_pred_density(y, 1, return_mcse=True)
        assert 0 < mcse < 0.05
        assert abs(lpd - exact_dist.logpdf(y)) < 4 * mcse, (lpd, exact_dist.logpdf(y), mcse)
    # marginal of a subset: Student-t with the same ν on the sub-block
    idx = [0, 2]
    y = loc + sd * np.array([-1.0, 0.0, 2.0])
    lpd, mcse = fr.log_pred_density(y[idx], 1, ["gdp", "rate"], return_mcse=True)
    marginal = stats.multivariate_t(loc=loc[idx], shape=shape[np.ix_(idx, idx)], df=nu).logpdf(
        y[idx]
    )
    assert abs(lpd - marginal) < 4 * mcse
    # power: the tolerance is far tighter than the gap to the plug-in Gaussian N(B̂'x, S/(T−k)),
    # which ignores parameter uncertainty
    y = loc + 2.5 * sd
    lpd, mcse = fr.log_pred_density(y, 1, return_mcse=True)
    plug_in = stats.multivariate_normal(loc, S / (T - k)).logpdf(y)
    assert abs(lpd - plug_in) > 10 * mcse


# --------------------------------------------------------------------------- ForecastResult API


def _toy_result(chains: int = 2, draws: int = 4, horizon: int = 3, n: int = 2) -> ForecastResult:
    rng = np.random.default_rng(0)
    paths = rng.standard_normal((chains, draws, horizon, n))
    chol = np.broadcast_to(np.eye(n), (chains, draws, horizon, n, n))
    return ForecastResult.from_arrays(
        paths=paths,
        cond_mean=np.zeros_like(paths),
        cond_chol=chol,
        paths_level=np.exp(paths),
        variables=["a", "b"][:n],
        origin=pd.Period("2001Q4", "Q"),
        attrs={"model": "toy", "tcodes": {"a": 1, "b": 1}, "seed": 5},
    )


def test_stacking_order_is_chain_major():
    fr = _toy_result()
    stacked = fr.stacked()
    paths = fr.paths.values
    assert stacked.shape == (8, 3, 2)
    np.testing.assert_array_equal(stacked[:4], paths[0])
    np.testing.assert_array_equal(stacked[4:], paths[1])
    assert fr.n_samples == 8


def test_from_arrays_without_chain_axis():
    fr = _toy_result()
    single = ForecastResult.from_arrays(
        paths=fr.stacked(),
        cond_mean=fr.stacked("cond_mean"),
        cond_chol=fr.stacked("cond_chol"),
        variables=fr.variables,
        origin=fr.origin,
    )
    assert single.dataset.sizes["chain"] == 1 and single.dataset.sizes["draw"] == 8
    assert single.paths_level is None
    with pytest.raises(ValueError, match="no level paths"):
        single.mean(level=True)
    assert single.origin == pd.Period("2001Q4", "Q")
    assert list(single.dataset["date"].values) == ["2002Q1", "2002Q2", "2002Q3"]


def test_attrs_are_serialised():
    fr = _toy_result()
    assert fr.dataset.attrs["seed"] == 5
    assert fr.dataset.attrs["tcodes"] == '{"a": 1, "b": 1}'
    assert fr.dataset.attrs["origin"] == "2001Q4"


def test_summaries():
    fr = _toy_result()
    stacked = fr.stacked()
    np.testing.assert_allclose(fr.mean().values, stacked.mean(axis=0))
    np.testing.assert_allclose(fr.median().values, np.median(stacked, axis=0))
    np.testing.assert_allclose(fr.mean(level=True).values, np.exp(stacked).mean(axis=0))
    q = fr.quantiles([0.1, 0.9])
    np.testing.assert_allclose(q.values, np.quantile(stacked, [0.1, 0.9], axis=0))
    frame = fr.to_dataframe(q=(0.1, 0.5, 0.9))
    assert frame.shape == (6, 3)
    assert frame.loc[("2002Q2", "b"), 0.5] == pytest.approx(np.median(stacked[:, 1, 1]))


def test_log_pred_density_against_the_mixture():
    fr = _toy_result()
    y = np.array([0.3, -1.2])
    expected = stats.multivariate_normal(np.zeros(2), np.eye(2)).logpdf(y)
    assert fr.log_pred_density(y, 2) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="horizon"):
        fr.log_pred_density(y, 0)
    with pytest.raises(ValueError, match="horizon"):
        fr.log_pred_density(y, 4)
    with pytest.raises(ValueError, match="one value per selected variable"):
        fr.log_pred_density(np.zeros(3), 1)
    with pytest.raises(KeyError):
        fr.log_pred_density(y[:1], 1, ["c"])


def test_log_pred_density_variable_selection_and_order():
    """Subsets marginalise the Gaussian; any order of the selected variables gives the same density."""
    cov = np.array([[1.0, 0.8, 0.2], [0.8, 2.0, -0.5], [0.2, -0.5, 0.7]])
    mu = np.array([0.1, -0.3, 1.0])
    fr = ForecastResult.from_arrays(
        paths=np.zeros((5, 1, 3)),
        cond_mean=np.broadcast_to(mu, (5, 1, 3)),
        cond_chol=np.broadcast_to(np.linalg.cholesky(cov), (5, 1, 3, 3)),
        variables=["a", "b", "c"],
        origin=pd.Period("2000Q1", "Q"),
    )
    y = np.array([0.5, 0.2, 0.3])
    joint = stats.multivariate_normal(mu, cov).logpdf(y)
    for order in (["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"]):
        idx = ["abc".index(v) for v in order]
        assert fr.log_pred_density(y[idx], 1, order) == pytest.approx(joint, rel=1e-12)
    for subset in (["b", "a"], ["c"], ["a", "c"]):
        idx = ["abc".index(v) for v in subset]
        marginal = stats.multivariate_normal(mu[idx], cov[np.ix_(idx, idx)]).logpdf(y[idx])
        assert fr.log_pred_density(y[idx], 1, subset) == pytest.approx(marginal, rel=1e-12)
        assert fr.log_pred_density(y, 1, subset) == pytest.approx(marginal, rel=1e-12)


def test_log_pred_density_mixes_draws():
    """Two equally likely draws with different means: log of the average density."""
    paths = np.zeros((2, 1, 1))
    fr = ForecastResult.from_arrays(
        paths=paths,
        cond_mean=np.array([[[-1.0]], [[2.0]]]),
        cond_chol=np.array([[[[1.0]]], [[[0.5]]]]),
        variables=["a"],
        origin=pd.Period("2000-01", "M"),
    )
    y = 0.4
    expected = np.log(0.5 * stats.norm(-1.0, 1.0).pdf(y) + 0.5 * stats.norm(2.0, 0.5).pdf(y))
    lpd, mcse = fr.log_pred_density([y], 1, return_mcse=True)
    assert lpd == pytest.approx(expected, rel=1e-12)
    assert mcse > 0


def test_invalid_results_are_rejected():
    fr = _toy_result()
    with pytest.raises(ValueError, match="cond_chol"):
        ForecastResult(fr.dataset.drop_vars("cond_chol"))
    bad = fr.dataset.copy(deep=True)
    bad["cond_chol"].values[0, 0, 0, 0, 1] = 0.5  # upper triangle
    with pytest.raises(ContractError, match="lower triangular"):
        check_structure(ForecastResult(bad))
    bad = fr.dataset.copy(deep=True)
    bad["cond_chol"].values[0, 1, 2, 1, 1] = -1.0
    with pytest.raises(ContractError, match="positive diagonal"):
        check_structure(ForecastResult(bad))
    bad = fr.dataset.copy(deep=True)
    bad["paths"].values[1, 1, 1, 1] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        check_structure(ForecastResult(bad))


def test_simulation_consistency_detects_wrong_moments(fitted):
    fr = fitted.forecast(4, draws=500, seed=2)
    ds = fr.dataset.copy(deep=True)
    ds["cond_chol"] = ds["cond_chol"] * 0.8  # stored scale too small for the simulated paths
    assert simulation_consistency(ForecastResult(ds)) < 1e-6
