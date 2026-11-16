"""The ordering check with Phase 3 priors (P3 DESIGN §3.8, §4.2; WP33).

* ``permute_model`` moves a global-local prior's custom ``groups`` array with the variables
  (equation and regressor), and an ``AsymmetricConjugate`` ψ tuple; named groups and ``psi="ar"``
  are left alone — :func:`test_permute_model_moves_a_custom_groups_array`,
  :func:`test_permute_model_moves_an_asymmetric_conjugate_psi_tuple`;
* the prior mode draws the NIG closed form's prior for ``AsymmetricConjugate``: at
  ``cross = contemp = 1`` it is the Phase 1 ``IW(diag ψ, n + 2)`` (two-sample KS against the
  conjugate model's draws, family-wise ``α = 1e-3`` over ``m = 6`` statistics, D-60); a smaller
  ``contemp`` shrinks the correlations — :func:`test_prior_mode_draws_the_nig_prior`;
* forecast mode reads ``StructuralNIGFit``'s exact posterior means: at ``cross = contemp = 1``
  with fixed hyperparameters they are order-invariant (the NIW posterior, ``1e-10``), at
  ``cross ≠ 1`` they are not — :func:`test_forecast_mode_reads_the_structural_nig_posterior`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import rai_bvar as rb
from rai_bvar.structural.ordering import (
    ordering_sensitivity,
    permute_model,
    prior_covariance_draws,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def var_data(n: int = 3, T: int = 80, seed: int = 21) -> rb.VARData:
    rng = np.random.default_rng(seed)
    A = 0.4 * np.eye(n) + 0.15 * np.eye(n, k=1)
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.1 + A @ y[t - 1] + rng.standard_normal(n) * np.linspace(0.5, 2.0, n)
    idx = pd.period_range("1980Q1", periods=T, freq="Q")
    return rb.VARData.from_frame(pd.DataFrame(y, index=idx, columns=list("abc")[:n]), tcodes=1)


@pytest.fixture(scope="module")
def data() -> rb.VARData:
    return var_data()


def ac(**kwargs) -> rb.BVAR:
    return rb.BVAR(mean=rb.Linear(rb.AsymmetricConjugate(**kwargs)), lags=1)


def test_permute_model_moves_a_custom_groups_array():
    n, lags = 3, 2
    # group id = 100·equation + 10·lag + variable: every entry names its own cell
    groups = [
        [100 * i + 10 * (lag + 1) + j for lag in range(lags) for j in range(n)] for i in range(n)
    ]
    m = rb.BVAR(mean=rb.Linear(rb.Horseshoe(groups=groups)), vol=rb.SV(), lags=lags)
    moved = permute_model(m, ["c", "a", "b"], variables=["a", "b", "c"])
    perm = [2, 0, 1]
    got = np.asarray(moved.mean.prior.groups)
    for i in range(n):
        for lag in range(lags):
            for j in range(n):
                assert got[i, lag * n + j] == 100 * perm[i] + 10 * (lag + 1) + perm[j]
    assert permute_model(m, perm) == moved
    assert permute_model(moved, ["a", "b", "c"], variables=["c", "a", "b"]) == m
    with pytest.raises(ValueError, match="custom groups array"):
        permute_model(m, ["b", "a", "c"])
    named = rb.BVAR(mean=rb.Linear(rb.Horseshoe()), vol=rb.SV(), lags=lags)
    assert permute_model(named, ["b", "a", "c"], variables=["a", "b", "c"]) is named
    both = rb.BVAR(mean=rb.Linear(rb.SSVS(groups=groups, psi=(1.0, 2.0, 3.0))), lags=lags)
    moved = permute_model(both, perm).mean.prior
    assert moved.psi == (3.0, 1.0, 2.0) and np.asarray(moved.groups)[0, 0] == 210 + 2


def test_permute_model_moves_an_asymmetric_conjugate_psi_tuple():
    m = ac(psi=(1.0, 2.0, 3.0), mean={"a": 0.5})
    moved = permute_model(m, ["c", "a", "b"], variables=["a", "b", "c"])
    assert moved.mean.prior.psi == (3.0, 1.0, 2.0) and moved.mean.prior.mean == {"a": 0.5}
    assert moved.engine == "analytic"
    assert permute_model(ac(), ["b", "a", "c"]) == ac()


def test_prior_mode_draws_the_nig_prior(data):
    nig = prior_covariance_draws(ac(cross=1.0, contemp=1.0, hierarchy="fixed"), data, 20_000, 1)
    iw = prior_covariance_draws(
        rb.BVAR(mean=rb.Linear(rb.Minnesota(hierarchy="fixed")), lags=1), data, 20_000, 2
    )
    assert nig.variables == iw.variables == ("a", "b", "c")
    pvalues = []
    for draws in (nig, iw):
        assert np.isfinite(draws.sigma).all() and draws.shock.shape == (20_000, 3)
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)):
        pvalues.append(stats.ks_2samp(nig.sigma[:, i, j], iw.sigma[:, i, j]).pvalue)
    assert min(pvalues) > 1e-3 / len(pvalues), pvalues
    tight = prior_covariance_draws(ac(contemp=0.1, hierarchy="fixed"), data, 20_000, 1)
    corr = lambda s: s[:, 2, 0] / np.sqrt(s[:, 0, 0] * s[:, 2, 2])  # noqa: E731
    assert np.mean(np.abs(corr(tight.sigma))) < 0.5 * np.mean(np.abs(corr(nig.sigma)))
    # the ordering check's prior mode runs on it
    res = ordering_sensitivity(ac(hierarchy="fixed"), data, n_perm=2, mode="prior", draws=500)
    assert res.dataset.attrs["engine"] == "analytic" and len(res.orderings) == 2


def test_forecast_mode_reads_the_structural_nig_posterior(data):
    psi = (0.3, 1.2, 3.5)
    kw = dict(n_perm=3, noise_reps=0, horizons=(1, 2), draws=200)
    for cross, invariant in ((1.0, True), (0.4, False)):
        model = ac(cross=cross, contemp=1.0, psi=psi, hierarchy="fixed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = ordering_sensitivity(model, data, **kw).dataset
        assert {"B_post", "sigma_post", "hyper_post"} <= set(ds.data_vars)
        assert set(ds["hyper"].values) >= {"lambda", "cross", "contemp"}
        B, sigma = ds["B_post"].values, ds["sigma_post"].values
        # the user ordering's values are the fit's exact posterior means
        fit = model.fit(data)
        B0, sigma0 = fit.posterior_moments()
        np.testing.assert_allclose(B[0], B0, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(sigma[0], sigma0, rtol=1e-12)
        spread = max(np.abs(B - B[0]).max() / np.abs(B0).max(), np.abs(sigma - sigma[0]).max())
        assert (spread < 1e-10) == invariant, spread
