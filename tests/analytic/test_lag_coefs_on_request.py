"""``lag_coefs`` on request on analytic forecasts (P3 DESIGN D-49 as amended by E120; WP33).

* the default analytic forecast is the Phase 1 one: no ``lag_coefs`` variable, the zero-stride
  ``cond_chol`` view (``strides[2] == 0``) and the retained memory (``tracemalloc`` current) of its
  physical arrays only — :func:`test_default_forecast_carries_no_lag_coefs_and_keeps_its_memory`;
* ``forecast(..., lag_coefs=True)`` adds ``lag_coefs[0, s, l−1, i, j] = B[n_det + (l−1)·n + j, i]``
  of each path's coefficient draw, from the draws the forecast already makes (no new random
  numbers): ``paths``, ``cond_mean`` and ``cond_chol`` are bitwise the default's, and the lag
  matrices are those of ``draw_parameters`` on the same seed —
  :func:`test_lag_coefs_on_request_adds_no_random_numbers` (``hierarchy`` fixed and ``mh``, and
  more draws than one chunk of ``_DRAW_BLOCK``); their orientation end to end, with one draw the
  integrated estimator is the exact two-step density — :func:`test_lag_coefs_orientation_end_to_end`;
* the benchmarks accept the keyword and ignore it (the forecast protocol, ``models.base``), and
  ``models.base.forecast_options`` passes it only to forecasts that take it —
  :func:`test_benchmarks_accept_and_ignore_the_keyword`, :func:`test_forecast_options`.
"""

from __future__ import annotations

import gc
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import rai_bvar as rb
from rai_bvar.forecast.result import LAG_DIMS
from rai_bvar.models import OLSVAR
from rai_bvar.models.base import forecast_options

N, LAGS, H = 5, 4, 3


def var_data(n: int = N, T: int = 90, seed: int = 7) -> rb.VARData:
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.1 + A @ y[t - 1] + rng.standard_normal(n)
    idx = pd.period_range("1980Q1", periods=T, freq="Q")
    return rb.VARData.from_frame(
        pd.DataFrame(y, index=idx, columns=[f"v{i}" for i in range(n)]), tcodes=1
    )


@pytest.fixture(scope="module")
def data() -> rb.VARData:
    return var_data()


def fitted(data: rb.VARData, hierarchy: str = "fixed"):
    mh = rb.MHSettings(draws=60, burn=40, chains=2, seed=4)
    prior = rb.Minnesota(hierarchy=hierarchy, mh=mh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return rb.BVAR(mean=rb.Linear(prior=prior), lags=LAGS).fit(data)


def _lag_matrices(coefs: np.ndarray, lags: int, n_det: int) -> np.ndarray:
    S, _, n = coefs.shape
    return coefs[:, n_det:].reshape(S, lags, n, n).swapaxes(-1, -2)


def test_default_forecast_carries_no_lag_coefs_and_keeps_its_memory(data):
    fit = fitted(data)
    S = 2000
    fit.forecast(H, 16, seed=0)  # warm the lazy imports outside the measurement
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        default = fit.forecast(H, S, seed=1)
        gc.collect()
        retained_default = tracemalloc.get_traced_memory()[0] - before
        requested = fit.forecast(H, S, seed=1, lag_coefs=True)
        gc.collect()
        retained_both = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    ds = default.dataset
    assert "lag_coefs" not in ds.data_vars
    assert ds["cond_chol"].values.strides[2] == 0  # the zero-stride view over the horizon
    path_bytes = S * H * N * 8  # paths, cond_mean, paths_level: one each
    physical = 3 * path_bytes + S * N * N * 8  # + the (S, n, n) store behind cond_chol
    lag_bytes = S * LAGS * N * N * 8  # 1.6 MB: what an always-on lag_coefs would add
    slack = 64 * 1024  # coordinates, attrs (the spec JSON), xarray bookkeeping
    assert physical <= retained_default <= physical + slack, (retained_default, physical)
    # the requested forecast retains the same arrays plus lag_coefs and nothing else
    extra = retained_both - retained_default
    assert lag_bytes <= extra - physical <= lag_bytes + slack, (extra - physical, lag_bytes)
    assert requested.dataset["lag_coefs"].values.nbytes == lag_bytes


@pytest.mark.parametrize("hierarchy", ["fixed", "mh"])
def test_lag_coefs_on_request_adds_no_random_numbers(data, hierarchy):
    fit = fitted(data, hierarchy)
    draws = 37  # three blocks of _DRAW_BLOCK = 16, the last one partial
    plain = fit.forecast(H, draws, seed=5).dataset
    ds = fit.forecast(H, draws, seed=5, lag_coefs=True).dataset
    assert set(ds.data_vars) == set(plain.data_vars) | {"lag_coefs"}
    for name in plain.data_vars:
        np.testing.assert_array_equal(ds[name].values, plain[name].values, err_msg=name)
    assert ds.attrs == plain.attrs
    da = ds["lag_coefs"]
    assert da.dims == LAG_DIMS and da.shape == (1, draws, LAGS, N, N)
    coefs, _ = fit.draw_parameters(draws, 5)  # the forecast's parameter stream on the same seed
    np.testing.assert_array_equal(da.values[0], _lag_matrices(coefs, LAGS, 1))
    # the integrated estimator now runs at h > 1 and coincides with the default at h = 1
    fr = rb.ForecastResult(ds)
    y = data.values[-1]
    assert np.isfinite(fr.log_pred_density(y, 2, lpd_method="integrated"))
    assert fr.log_pred_density(y, 1, lpd_method="integrated") == fr.log_pred_density(y, 1)


def test_lag_coefs_orientation_end_to_end(data):
    """One draw, constant volatility: the integrated estimator is the exact density
    ``y_{T+2} | θ ~ N(m_2 − A_1 u_1, A_1ΣA_1ᵀ + Σ)`` (``u_1 = paths_1 − m_1``), with ``A_1`` built
    here by index from the draw's coefficients (``A_1[i, j]`` = coefficient of lag 1 of variable
    ``j`` in equation ``i``, regressor row ``1 + j`` after the constant)."""
    fit = fitted(data)
    ds = fit.forecast(2, 1, seed=8, lag_coefs=True).dataset
    coefs, chol = fit.draw_parameters(1, 8)
    A1 = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            A1[i, j] = coefs[0, 1 + j, i]
    np.testing.assert_array_equal(ds["lag_coefs"].values[0, 0, 0], A1)
    sigma = chol[0] @ chol[0].T
    paths, means = ds["paths"].values[0, 0], ds["cond_mean"].values[0, 0]
    mean2 = means[1] - A1 @ (paths[0] - means[0])
    y = data.values[-1] + 0.3
    want = stats.multivariate_normal(mean2, A1 @ sigma @ A1.T + sigma).logpdf(y)
    got = rb.ForecastResult(ds).log_pred_density(y, 2, lpd_method="integrated")
    assert got == pytest.approx(want, abs=1e-10)


def test_benchmarks_accept_and_ignore_the_keyword(data):
    fit = OLSVAR(2).fit(data)
    plain = fit.forecast(H, 30, seed=2).dataset
    asked = fit.forecast(H, 30, seed=2, lag_coefs=True).dataset
    assert "lag_coefs" not in asked.data_vars
    for name in plain.data_vars:
        np.testing.assert_array_equal(asked[name].values, plain[name].values, err_msg=name)


def test_forecast_options():
    class Phase1Fit:  # a model written to the Phase 1 protocol
        def forecast(self, horizon, draws=1000, seed=None): ...

    class Keyword:
        def forecast(self, horizon, draws=1000, seed=None, *, lag_coefs=False): ...

    class Anything:
        def forecast(self, horizon, draws=1000, seed=None, **options): ...

    for fit in (Phase1Fit(), Keyword(), Anything()):
        assert forecast_options(fit, lag_coefs=False) == {}
    assert forecast_options(Phase1Fit(), lag_coefs=True) == {}
    assert forecast_options(Keyword(), lag_coefs=True) == {"lag_coefs": True}
    assert forecast_options(Anything(), lag_coefs=True) == {"lag_coefs": True}


def test_lag_coefs_must_be_a_bool(data):
    with pytest.raises(TypeError, match="lag_coefs must be a bool"):
        fitted(data).forecast(H, 10, seed=0, lag_coefs="yes")
