"""Log predictive densities of MCMC forecasts (DESIGN §13 WP17; §7.3, D-20; N03 §2.2, §5.3).

``log_pred_density`` keeps its SPEC §7 definition. Tested here: the factored ``(cond_A,
cond_scale)`` density equals the ``cond_chol`` density (1e-12) for the full vector and leading
blocks; leading-block slicing ``L[:j, :j]`` equals ``marginal_chol`` (1e-12; MCMC and analytic
forecasts); the ESS-adjusted MCSE is at least the iid MCSE on autocorrelated draws; the opt-in
transitory Rao-Blackwellisation (``f`` equal to the kernels' ``TransitoryScale.log_density``)
matches Monte Carlo over drawn ``(o, τ)`` (the N03 §5.3 cross-check) and falls back with a
warning where it does not apply; the accelerator variables are not contract fields.
"""

from __future__ import annotations

import jax.numpy as jnp
import mcmc_fits as mf
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import logsumexp

from rai_bvar import BVAR, Linear, Minnesota
from rai_bvar.components.vol import transitory as TR
from rai_bvar.diagnostics import ess_mean
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.state import Empty, OutlierDraw, OutlierTDraw, TDraw
from rai_bvar.forecast import ForecastResult, check_structure
from rai_bvar.forecast.result import transitory_log_density
from rai_bvar.linalg import marginal_chol, mvn_logpdf_chol

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

EXTRAS = ("cond_A", "cond_scale", "cond_scale_persistent", "log_p", "log1m_p", "nu")


@pytest.fixture(scope="module")
def fr(sim_data):
    return mf.shared_fit(sim_data).forecast(4, draws=120, seed=8)


def _plain(fr: ForecastResult) -> ForecastResult:
    """The same forecast without accelerator variables."""
    return ForecastResult(fr.dataset.drop_vars([v for v in EXTRAS if v in fr.dataset]))


def _reference(fr: ForecastResult, y, h: int, idx) -> float:
    """``logsumexp_s log N(y; m_s[idx], marginal_chol(L_s, idx)) − log S`` (QR factor)."""
    mean = fr.stacked("cond_mean")[:, h - 1][:, idx]
    chol = marginal_chol(fr.stacked("cond_chol")[:, h - 1], np.asarray(idx))
    logp = mvn_logpdf_chol(np.asarray(y), mean, chol)
    return float(logsumexp(logp) - np.log(logp.size))


def _points(fr: ForecastResult, h: int) -> list[np.ndarray]:
    m = fr.stacked("cond_mean")[:, h - 1].mean(axis=0)
    sd = fr.stacked("paths")[:, h - 1].std(axis=0)
    return [m, m + sd * np.array([1.5, -2.0, 0.5]), m - 3.0 * sd]


def test_mcmc_forecasts_carry_the_accelerator_variables(fr):
    ds = fr.dataset
    for name in EXTRAS[:5]:
        assert name in ds, name
    assert ds["cond_A"].dims == ("chain", "draw", "variable", "variable_")
    A = fr.stacked("cond_A")
    np.testing.assert_array_equal(np.diagonal(A, axis1=1, axis2=2), 1.0)
    assert np.all(np.triu(A, 1) == 0)
    # cond_chol = A⁻¹ diag(cond_scale) at every horizon
    L = fr.stacked("cond_chol")
    ell = fr.stacked("cond_scale")
    np.testing.assert_allclose(
        np.einsum("sij,shjk->shik", A, L), ell[..., None] * np.eye(3), rtol=1e-12, atol=1e-14
    )
    check_structure(fr)  # extras are ignored by the contract checks


@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize(
    "subset", [None, ["gdp"], ["gdp", "prices"], ["prices", "gdp"], ["rate", "gdp", "prices"]]
)
def test_factored_lpd_equals_cond_chol_lpd(fr, subset, h):
    """Full vector and leading blocks (in any order): factored ``(A, ℓ)`` density = ``cond_chol``
    density = the ``marginal_chol`` reference (1e-12); the MCSE agrees too."""
    plain = _plain(fr)
    idx = np.arange(3) if subset is None else np.array([fr.variables.index(v) for v in subset])
    for y in _points(fr, h):
        yy = y[idx]
        a, mcse_a = fr.log_pred_density(yy, h, subset, return_mcse=True)
        b, mcse_b = plain.log_pred_density(yy, h, subset, return_mcse=True)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mcse_a, mcse_b, rtol=1e-9)
        np.testing.assert_allclose(b, _reference(fr, yy, h, idx), rtol=1e-12, atol=1e-12)


def test_non_leading_subsets_use_the_qr_factor(fr):
    plain = _plain(fr)
    for subset in (["prices"], ["rate"], ["prices", "rate"], ["rate", "prices"]):
        idx = [fr.variables.index(v) for v in subset]
        y = _points(fr, 2)[1][idx]
        assert fr.log_pred_density(y, 2, subset) == plain.log_pred_density(y, 2, subset)
        assert plain.log_pred_density(y, 2, subset) == _reference(fr, y, 2, idx)


def test_leading_block_slicing_equals_marginal_chol_for_analytic_forecasts(sim_data):
    model = BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed")), lags=2)
    fr = model.fit(sim_data).forecast(3, draws=300, seed=4)
    assert fr.dataset.attrs["engine"] == "analytic"
    for subset in (["gdp"], ["gdp", "prices"], ["prices", "gdp"], ["rate", "prices", "gdp"]):
        idx = [fr.variables.index(v) for v in subset]
        for y in _points(fr, 3):
            got = fr.log_pred_density(y[idx], 3, subset)
            np.testing.assert_allclose(got, _reference(fr, y[idx], 3, idx), rtol=1e-12, atol=1e-12)


def _ar_forecast(rho: float, *, engine: str | None, chains=4, draws=500, seed=0) -> ForecastResult:
    """One-variable forecasts whose conditional means follow an AR(1) across draws."""
    rng = np.random.default_rng(seed)
    m = np.empty((chains, draws))
    m[:, 0] = rng.standard_normal(chains)
    for d in range(1, draws):
        m[:, d] = rho * m[:, d - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(chains)
    mean = m[:, :, None, None]
    attrs = {} if engine is None else {"engine": engine}
    return ForecastResult.from_arrays(
        paths=mean + 1.0, cond_mean=mean, cond_chol=np.ones((chains, draws, 1, 1, 1)),
        variables=["x"], origin=pd.Period("2000Q1", "Q"), attrs=attrs,
    )  # fmt: skip


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_ess_adjusted_mcse(rho):
    """``engine == "mcmc"``: the iid delta-method MCSE times ``√(S/ESS)``, ``ESS =
    ess_mean(w)`` of the weights on the (chain, draw) array; ≥ iid MCSE when autocorrelated."""
    y = np.array([1.2])
    iid = _ar_forecast(rho, engine=None)
    mc = _ar_forecast(rho, engine="mcmc")
    lpd_a, se_iid = iid.log_pred_density(y, 1, return_mcse=True)
    lpd_b, se_mc = mc.log_pred_density(y, 1, return_mcse=True)
    assert lpd_a == lpd_b
    logp = mvn_logpdf_chol(y, mc.stacked("cond_mean")[:, 0], mc.stacked("cond_chol")[:, 0])
    w = np.exp(logp - logp.max()).reshape(4, 500)
    np.testing.assert_allclose(se_mc, se_iid * np.sqrt(w.size / ess_mean(w)), rtol=1e-12)
    if rho > 0:
        assert se_mc > 2.0 * se_iid  # τ ≈ (1 + ρ²)/(1 − ρ²)-ish for these weights
    else:
        assert 0.8 < se_mc / se_iid < 1.25


def test_mcse_falls_back_to_iid_without_an_ess():
    """Fewer than 10 draws per chain: the ESS is undefined and the iid MCSE is kept."""
    a = _ar_forecast(0.5, engine=None, draws=8)
    b = _ar_forecast(0.5, engine="mcmc", draws=8)
    assert a.log_pred_density([0.3], 1, return_mcse=True) == b.log_pred_density(
        [0.3], 1, return_mcse=True
    )


# --------------------------------------------------------------------------- transitory RB


def _kernel(kind: str, t_scale: str = "standardized", K: int = 20):
    vs = VolStatic(persistent="sv", transitory=kind, outlier_K=K, nu_grid=(5, 7, 9),
                   t_scale=t_scale)  # fmt: skip
    cls = {"none": TR.NoTransitory, "outlier": TR.OutlierScale, "t": TR.TScale,
           "outlier_t": TR.OutlierTScale}[kind]  # fmt: skip
    return cls(vs, 8)


@pytest.mark.parametrize(
    ("kind", "t_scale", "K"),
    [("none", "standardized", 20), ("outlier", "standardized", 20), ("outlier", "standardized", 4),
     ("t", "standardized", 20), ("t", "unit", 20), ("outlier_t", "standardized", 20),
     ("outlier_t", "unit", 6)],
)  # fmt: skip
def test_transitory_log_density_equals_the_kernels(kind, t_scale, K):
    """NumPy ``f`` of N03 §2.2 = ``TransitoryScale.log_density`` (1e-12)."""
    z = np.array([-30.0, -6.0, -1.3, -1e-3, 0.0, 0.4, 2.5, 11.0])
    log_p, log1m_p, nu = np.log(0.07), np.log1p(-0.07), 7.0
    td = {
        "none": Empty(),
        "outlier": OutlierDraw(log_p=jnp.asarray(log_p), log1m_p=jnp.asarray(log1m_p)),
        "t": TDraw(nu=jnp.asarray(nu)),
        "outlier_t": OutlierTDraw(
            log_p=jnp.asarray(log_p), log1m_p=jnp.asarray(log1m_p), nu=jnp.asarray(nu)
        ),
    }[kind]
    ref = np.asarray(_kernel(kind, t_scale, K).log_density(jnp.log(z * z), td))
    got = transitory_log_density(z, kind, log_p=log_p, log1m_p=log1m_p, nu=nu, K=K, t_scale=t_scale)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def _rb_problem(S: int, *, seed: int = 5, p: float = 0.05, nu: float = 6.0, K: int = 20):
    """N03 §5.3 cross-check: ``n = 3``, random unit-lower ``A``, standardized SVO-t, ``S`` drawn
    ``(o, τ)`` configurations around one parameter draw. Returns the forecast whose draws carry
    the drawn effective scales, and the accelerator variables of the persistent part."""
    rng = np.random.default_rng(seed)
    n = 3
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = rng.normal(0.0, 0.6, 3)
    m = rng.normal(0.0, 1.0, n)
    ell_p = np.array([0.8, 1.3, 0.6])
    u = rng.uniform(size=(S, n))
    o = np.where(u < p, 2.0 + np.floor(u / p * (K - 1)), 1.0)
    tau = ((nu - 2.0) / 2.0) / rng.gamma(nu / 2.0, 1.0, size=(S, n))
    ell = ell_p * o * np.sqrt(tau)
    chol = np.linalg.solve(A, np.eye(n))[None] * ell[:, None, :]  # A⁻¹ diag(ℓ)
    paths = m + np.einsum("sij,sj->si", chol, rng.standard_normal((S, n)))
    extra = {
        "cond_A": (("chain", "draw", "variable", "variable_"), np.broadcast_to(A, (1, S, n, n))),
        "cond_scale": (("chain", "draw", "horizon", "variable"), ell[None, :, None, :]),
        "cond_scale_persistent": (
            ("chain", "draw", "horizon", "variable"),
            np.broadcast_to(ell_p, (1, S, 1, n)),
        ),
        "log_p": (("chain", "draw", "variable"), np.full((1, S, n), np.log(p))),
        "log1m_p": (("chain", "draw", "variable"), np.full((1, S, n), np.log1p(-p))),
        "nu": (("chain", "draw", "variable"), np.full((1, S, n), nu)),
    }
    fr = ForecastResult.from_arrays(
        paths=paths[None, :, None, :], cond_mean=np.broadcast_to(m, (1, S, 1, n)),
        cond_chol=chol[None, :, None], variables=["a", "b", "c"],
        origin=pd.Period("2020Q1", "Q"),
        attrs={"engine": "mcmc", "transitory": "outlier_t", "outlier_K": K,
               "t_scale": "standardized"},
        extra=extra,
    )  # fmt: skip
    return fr, A, m, ell_p


def test_rb_transitory_matches_monte_carlo():
    """RB ``Σ_{i<j}[−log ℓ^pers_i + log f_i(η_i)]`` vs the drawn-scale estimator over 200k
    drawn ``(o, τ)`` (N03 §5.3: full vector and the leading block ``j = 2``, an extreme residual
    in equation 2), within 4 MCSE; and the RB value written out with ``f`` directly."""
    fr, A, m, ell_p = _rb_problem(200_000)
    for y, subset in (
        (m + np.array([0.3, -6.0, 0.5]), None),
        (m[:2] + np.array([0.3, -6.0]), ["a", "b"]),
        (m + np.array([-0.4, 0.2, 1.0]), None),
    ):
        rb = fr.log_pred_density(y, 1, subset, rb_transitory=True)
        mc, se = fr.log_pred_density(y, 1, subset, return_mcse=True)
        assert abs(rb - mc) < 4.0 * se + 1e-3, (rb, mc, se)
        j = y.size
        eta = (A[:j, :j] @ (y - m[:j])) / ell_p[:j]
        exact = np.sum(
            transitory_log_density(eta, "outlier_t", log_p=np.log(0.05),
                                   log1m_p=np.log1p(-0.05), nu=6.0)
            - np.log(ell_p[:j])
        )  # fmt: skip
        np.testing.assert_allclose(rb, exact, rtol=1e-12)


def test_rb_without_a_transitory_scale_is_the_gaussian_density():
    fr, _, m, ell_p = _rb_problem(50)
    ds = fr.dataset.copy()
    ds.attrs["transitory"] = "none"
    ds["cond_scale"] = ds["cond_scale_persistent"]
    S = ds.sizes["draw"]
    A = ds["cond_A"].values[0, 0]
    chol = np.linalg.solve(A, np.eye(3)) * ell_p
    ds["cond_chol"] = (ds["cond_chol"].dims, np.broadcast_to(chol, (1, S, 1, 3, 3)).copy())
    g = ForecastResult(ds)
    y = m + np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(
        g.log_pred_density(y, 1, rb_transitory=True), g.log_pred_density(y, 1), rtol=1e-12
    )


def test_rb_on_an_mcmc_forecast(fr):
    """Full vector and leading blocks are finite; other subsets and forecasts without the
    extras warn and fall back to the drawn-scale estimator."""
    y = _points(fr, 1)[1]
    full = fr.log_pred_density(y, 1, rb_transitory=True)
    lead = fr.log_pred_density(y[:2], 1, ["gdp", "prices"], rb_transitory=True)
    assert np.isfinite(full) and np.isfinite(lead)
    assert abs(full - fr.log_pred_density(y, 1)) < 1.0  # the same quantity, two estimators
    with pytest.warns(UserWarning, match="leading block"):
        got = fr.log_pred_density(y[1:2], 1, ["prices"], rb_transitory=True)
    assert got == fr.log_pred_density(y[1:2], 1, ["prices"])
    with pytest.warns(UserWarning, match="cond_A"):
        got = _plain(fr).log_pred_density(y, 1, rb_transitory=True)
    assert got == _plain(fr).log_pred_density(y, 1)


# --------------------------------------------------------------------------- from_arrays extras


def test_from_arrays_extras_validation():
    paths = np.zeros((2, 3, 1, 2))
    chol = np.broadcast_to(np.eye(2), (2, 3, 1, 2, 2))
    base = dict(paths=paths, cond_mean=paths, cond_chol=chol, variables=["a", "b"],
                origin=pd.Period("2000Q1", "Q"))  # fmt: skip
    fr = ForecastResult.from_arrays(**base, extra={"cond_A": (("draw", "variable", "variable_"),
                                                               np.zeros((3, 2, 2)))})  # fmt: skip
    assert fr.dataset["cond_A"].dims == ("draw", "variable", "variable_")
    single = {**base, "paths": paths[0], "cond_mean": paths[0], "cond_chol": chol[0]}
    fr = ForecastResult.from_arrays(
        **single, extra={"nu": (("chain", "draw", "variable"), np.ones((3, 2)))}
    )
    assert fr.dataset["nu"].shape == (1, 3, 2)  # chain axis added, as for the contract fields
    with pytest.raises(ValueError, match="clashes"):
        ForecastResult.from_arrays(**base, extra={"paths": (("chain",), np.zeros(2))})
    with pytest.raises(ValueError, match="axes"):
        ForecastResult.from_arrays(**base, extra={"nu": (("chain", "draw"), np.zeros((2, 3, 2)))})
    assert isinstance(fr.dataset, xr.Dataset)
    check_structure(fr)


def test_transitory_log_density_errors():
    with pytest.raises(ValueError, match="kind"):
        transitory_log_density(np.zeros(2), "cauchy")
    with pytest.raises(ValueError, match="K >= 2"):
        transitory_log_density(np.zeros(2), "outlier", log_p=-1.0, log1m_p=-0.5, K=1)
