"""``score_forecast(…, lpd_method=…)`` (SPEC §9.2; DESIGN §7.3, D-34).

On synthetic forecasts of a VAR(1) with one coefficient matrix per draw (no JAX): the default
output is the Phase 1 layout and values whether or not the forecast carries ``lag_coefs`` (the
exact layout itself is pinned by ``test_scores.py``); ``"integrated"`` adds exactly the four
documented variables and ``attrs["lpd_method"]``, its values are the direct
``log_pred_density`` calls (bitwise: same calls on the same arrays), the conditioned columns are
the default run's LPD bitwise, every other score is unchanged, and missing truths give the same
NaN pattern; ``h = 1`` needs no ``lag_coefs``, ``h > 1`` raises without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rai_bvar.evaluation.scores import score_forecast
from rai_bvar.forecast import ForecastResult
from rai_bvar.forecast.result import LAG_DIMS
from rai_bvar.models import OLSVAR

VARIABLES = ["gdp", "prices", "rate"]
N, H = 3, 4
LPD = {"lpd", "lpd_mcse", "lpd_marginal"}
EXTRA_MARGINAL = {"lpd_marginal_mcse", "lpd_marginal_conditioned"}
EXTRA_JOINT = {"lpd_conditioned", "lpd_mcse_conditioned"}


def var1_forecast(lag: bool = True, chains: int = 2, draws: int = 150, seed: int = 0):
    """``y_t = c + A_s y_{t−1} + L ε_t`` with one ``A_s`` per draw; paths, conditional moments
    and (``lag``) the ``lag_coefs`` of every draw, ``(chain, draw)`` shaped."""
    rng = np.random.default_rng(seed)
    S = chains * draws
    A = 0.7 * np.eye(N) + rng.normal(0.0, 0.08, (S, N, N))
    c = np.array([0.1, -0.2, 0.05])
    L = np.linalg.cholesky(np.array([[1.0, 0.3, -0.2], [0.3, 0.8, 0.1], [-0.2, 0.1, 0.5]]))
    y = np.broadcast_to(rng.normal(0.0, 1.0, N), (S, N))
    paths, means = np.empty((S, H, N)), np.empty((S, H, N))
    for h in range(H):
        means[:, h] = c + np.einsum("sij,sj->si", A, y)
        paths[:, h] = means[:, h] + rng.standard_normal((S, N)) @ L.T
        y = paths[:, h]
    shape = (chains, draws)
    extra = {"lag_coefs": (LAG_DIMS, A.reshape(*shape, 1, N, N))} if lag else None
    return ForecastResult.from_arrays(
        paths=paths.reshape(*shape, H, N), cond_mean=means.reshape(*shape, H, N),
        cond_chol=np.broadcast_to(L, (*shape, H, N, N)), variables=VARIABLES,
        origin=pd.Period("2010Q4", "Q"), extra=extra,
    )  # fmt: skip


@pytest.fixture(scope="module")
def case():
    fr = var1_forecast()
    sd = fr.paths.std(("chain", "draw")).values  # (H, n)
    realized = fr.paths.mean(("chain", "draw")).values + sd * np.array([2.5, -1.0, 0.5])
    return fr, realized


def _scores(fr, realized, **kw):
    return score_forecast(fr, realized, rng=np.random.default_rng(0), **kw)


def test_default_scores_ignore_lag_coefs(case):
    fr, realized = case
    plain = var1_forecast(lag=False)
    a = _scores(fr, realized)
    for b in (_scores(plain, realized), _scores(fr, realized, lpd_method="conditioned")):
        assert set(b.data_vars) == set(a.data_vars) and set(a.data_vars) >= LPD
        assert a.attrs == b.attrs == {}
        for name in a.data_vars:
            np.testing.assert_array_equal(b[name].values, a[name].values, err_msg=name)
    assert not ({*a.data_vars} & (EXTRA_MARGINAL | EXTRA_JOINT))


def test_integrated_layout_and_values(case):
    fr, realized = case
    base = _scores(fr, realized)
    ds = _scores(fr, realized, lpd_method="integrated")
    assert set(ds.data_vars) == set(base.data_vars) | EXTRA_MARGINAL | EXTRA_JOINT
    assert ds.attrs == {"lpd_method": "integrated"}
    for name in EXTRA_MARGINAL:
        assert ds[name].dims == ("horizon", "variable") and ds[name].dtype == np.float64
    for name in EXTRA_JOINT:
        assert ds[name].dims == ("horizon",) and ds[name].dtype == np.float64
    # every other score is the default run's (the estimator uses no random numbers)
    for name in set(base.data_vars) - LPD:
        np.testing.assert_array_equal(ds[name].values, base[name].values, err_msg=name)
    # the conditioned columns are the default estimator on the same forecast
    for new, old in (("lpd_conditioned", "lpd"), ("lpd_mcse_conditioned", "lpd_mcse"),
                     ("lpd_marginal_conditioned", "lpd_marginal")):  # fmt: skip
        np.testing.assert_array_equal(ds[new].values, base[old].values, err_msg=new)
    # the primary columns are the direct integrated calls
    for h in range(1, H + 1):
        lpd, mcse = fr.log_pred_density(
            realized[h - 1], h, return_mcse=True, lpd_method="integrated"
        )
        assert ds["lpd"].sel(horizon=h).item() == lpd
        assert ds["lpd_mcse"].sel(horizon=h).item() == mcse
        for j, var in enumerate(VARIABLES):
            lj, mj = fr.log_pred_density(
                realized[h - 1, j : j + 1], h, [var], return_mcse=True, lpd_method="integrated"
            )
            assert ds["lpd_marginal"].sel(horizon=h, variable=var).item() == lj
            assert ds["lpd_marginal_mcse"].sel(horizon=h, variable=var).item() == mj
    # h = 1: the estimators coincide bitwise; h > 1: they differ
    for new, old in (("lpd", "lpd_conditioned"), ("lpd_mcse", "lpd_mcse_conditioned"),
                     ("lpd_marginal", "lpd_marginal_conditioned")):  # fmt: skip
        one, later = ds[new].sel(horizon=1), ds[new].sel(horizon=slice(2, None))
        np.testing.assert_array_equal(one.values, ds[old].sel(horizon=1).values)
        assert np.all(later.values != ds[old].sel(horizon=slice(2, None)).values), new
    for name in ds.data_vars:
        assert np.isfinite(ds[name].values).all(), name


def test_integrated_missing_truths_follow_the_default_nan_pattern(case):
    fr, realized = case
    ragged = realized.copy()
    ragged[1, 2] = np.nan  # one variable missing at h = 2
    ragged[3] = np.nan  # not yet observed at h = 4
    base = _scores(fr, ragged)
    ds = _scores(fr, ragged, lpd_method="integrated")
    observed, complete = np.isfinite(ragged), np.isfinite(ragged).all(axis=1)
    for name in ds.data_vars:
        want = observed if ds[name].dims == ("horizon", "variable") else complete
        if name in base:
            np.testing.assert_array_equal(np.isfinite(base[name].values), want, err_msg=name)
        np.testing.assert_array_equal(np.isfinite(ds[name].values), want, err_msg=name)
    full = _scores(fr, realized, lpd_method="integrated")
    np.testing.assert_array_equal(
        ds["lpd_marginal"].values[observed], full["lpd_marginal"].values[observed]
    )
    nothing = score_forecast(fr, np.full_like(realized, np.nan), lpd_method="integrated")
    for name in nothing.data_vars:
        assert np.isnan(nothing[name].values).all(), name


def test_integrated_needs_lag_coefs_beyond_horizon_one(sim_data_stationary):
    origin = pd.Period("2015Q4", "Q")
    fit = OLSVAR(2).fit(sim_data_stationary.upto(origin))
    truth = sim_data_stationary.realized(origin, 3)
    one = fit.forecast(1, draws=300, seed=1)  # a benchmark forecast: no lag_coefs
    a = score_forecast(one, truth[:1], rng=np.random.default_rng(0))
    b = score_forecast(one, truth[:1], rng=np.random.default_rng(0), lpd_method="integrated")
    for name in a.data_vars:
        np.testing.assert_array_equal(b[name].values, a[name].values, err_msg=name)
    with pytest.raises(ValueError, match="lag_coefs"):
        score_forecast(fit.forecast(3, draws=300, seed=1), truth, lpd_method="integrated")


def test_unknown_lpd_method_raises(case):
    fr, realized = case
    for bad in ("Integrated", "path", None):
        with pytest.raises(ValueError, match="lpd_method must be one of"):
            score_forecast(fr, realized, lpd_method=bad)
    with pytest.raises(ValueError, match="lpd_method must be one of"):  # even with nothing to score
        score_forecast(fr, np.full_like(realized, np.nan), lpd_method="exact")
