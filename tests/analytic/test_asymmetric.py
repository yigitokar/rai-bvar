"""A9 and the ``StructuralNIGFit`` contract (P3 DESIGN §4, §9.7 A9 as amended by E127; WP33).

``BVAR(Linear(AsymmetricConjugate(...)))`` resolves to the analytic engine and is fitted by
``engines.structural_nig.StructuralNIGFit``: Chan's (2022) per-equation NIG closed form with iid
draws and the exact log marginal likelihood, hyperparameters ``fixed``, ``eb`` (free ``cross``,
``contemp``: flat on their log over ``[1e-3, 10]``) or ``mh``.

**A9** (PR tier unless marked):

* at ``cross = contemp = 1`` the prior is the Phase 1 NIW prior, so ``StructuralNIGFit`` equals
  ``AnalyticFit``: log ML at several hyperparameter points (``1e-10``), the posterior means
  ``E[B] = (E[A]⁻¹E[Γ])ᵀ`` (exact by row independence) and ``E[Σ]`` from the full-covariance
  recursion over each equation's NIG posterior second moments (``posterior_moments``) against
  ``NIWPosterior.mean`` and ``sigma_mean`` (``1e-10``), and the EB ``λ`` (``1e-6``);
* EB with free ``cross``: the projected gradient (the box boundary) of the EB objective at the
  returned mode, by finite differences written here from the design's objective (``Σ_i log
  p(y_i | h)`` + the GLP log density of ``λ``; nothing for the flat ``log cross``), below
  ``1e-6``;
* MH: quantiles of the MH draws of ``(λ, cross)`` against the posterior on a hyperparameter grid
  by quadrature (written from the design's statement: the natural-scale GLP Gamma prior of ``λ``
  from SciPy, ``cross`` flat on its log over ``[1e-3, 10]``), under the Phase 2 parity rule in the
  CDF form of ``test_mcmc_twin.py``: ``z = (F̂_MH(x_q) − q)/MCSE``, ``|z| ≤ max(4, z_{1−0.01/(2m)})``
  with the MCSE of the indicator mean (``diagnostics.mcse_mean_many``) and ``N_ref = ∞``;
* at ``cross ≠ 1`` (weekly, ``slow``): forecast quantiles against MCMC Mode II
  (``Minnesota(form="structural", cross=…) × Constant("iw")``, the same posterior, P2 rule 4)
  under the same rule, with the analytic forecast's iid draws as the reference (``N_ref``).

Besides A9: the fit contract (dispatch, attrs, the forecast structure checks, ``lag_coefs`` on
request; no JAX loaded, §1.1), Monte Carlo moments of ``draw_parameters`` against
``posterior_moments``, ``psi`` free
only at ``cross = contemp = 1`` (the twin's mode), the R8 note for ``n ≥ 4`` when ``cross ≠ 1``
or ``contemp ≠ 1``, and the MH seed (``fit(seed=…)``, E131).
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, interpolate, stats

import rai_bvar as rb
from rai_bvar import diagnostics
from rai_bvar.engines import structural_nig as nig
from rai_bvar.forecast import check_structure, simulation_consistency
from rai_bvar.priors.asymmetric import LOG_FLAT_BOX

LAGS, DET = 2, "c"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
ORDERING_NOTE = "this composition is not order-invariant"


def var_data(n: int = 3, T: int = 80, seed: int = 2033) -> rb.VARData:
    """A stationary VAR(1) with cross-variable dynamics and correlated shocks (tcode 1)."""
    rng = np.random.default_rng(seed)
    A = 0.4 * np.eye(n) + 0.2 * np.eye(n, k=1) + 0.15 * np.eye(n, k=-1)
    corr = 0.3 * np.ones((n, n)) + 0.7 * np.eye(n)
    chol = np.linalg.cholesky(corr) * np.linspace(0.6, 1.4, n)[:, None]
    y, out = np.zeros(n), np.empty((T, n))
    for t in range(T + 50):
        y = 0.2 + A @ y + chol @ rng.standard_normal(n)
        if t >= 50:
            out[t - 50] = y
    index = pd.period_range("1980Q1", periods=T, freq="Q")
    frame = pd.DataFrame(out, index=index, columns=[f"v{j}" for j in range(n)])
    return rb.VARData.from_frame(frame, tcodes=1)


@pytest.fixture(scope="module")
def data() -> rb.VARData:
    return var_data()


def ac(**kwargs) -> rb.BVAR:
    return rb.BVAR(mean=rb.Linear(rb.AsymmetricConjugate(**kwargs)), lags=LAGS, deterministic=DET)


def conjugate(**kwargs) -> rb.BVAR:
    prior = rb.Minnesota(form="conjugate", **kwargs)
    return rb.BVAR(mean=rb.Linear(prior), lags=LAGS, deterministic=DET)


def fitted(model: rb.BVAR, data: rb.VARData, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model.fit(data, **kwargs)


def bonferroni(m: int) -> float:
    """``max(4, z_{1−0.01/(2m)})``: the Phase 2 parity rule over ``m`` comparisons."""
    return max(4.0, float(stats.norm.ppf(1.0 - 0.01 / (2.0 * m))))


# =========================================================================== the fit contract


def test_bvar_fit_is_a_structural_nig_fit(data):
    model = ac()
    assert model.engine == "analytic" and model.vol == rb.Constant("iw")
    assert not model.uses_fit_seed and not model.supports_warm_start  # E131
    fit = fitted(model, data)
    assert isinstance(fit, nig.StructuralNIGFit) and fit.engine == "analytic"
    assert fit.converged and fit.mh is None and fit.optimization["success"]
    assert set(fit.hyper) == {"lambda", "alpha", "soc", "sur", "cross", "contemp", "psi"}
    assert fit.hyper["cross"] != 0.5 and fit.hyper["contemp"] == 1.0  # cross free, contemp not
    assert list(fit.summary().index) == ["lambda", "cross"]
    fc = fit.forecast(3, 400, seed=3)
    check_structure(fc)
    assert simulation_consistency(fc) > 1e-3  # KS p-value of the standardised residuals
    attrs = fc.dataset.attrs
    assert attrs["engine"] == "analytic" and attrs["model"] == model.name
    assert attrs["ordering"] == '["v0", "v1", "v2"]' and attrs["mh_seed"] == -1
    assert "lag_coefs" not in fc.dataset.data_vars
    asked = fit.forecast(3, 400, seed=3, lag_coefs=True).dataset
    for name in fc.dataset.data_vars:
        np.testing.assert_array_equal(asked[name].values, fc.dataset[name].values)
    coefs, chol = fit.draw_parameters(400, 3)
    lag = coefs[:, 1:].reshape(400, LAGS, 3, 3).swapaxes(-1, -2)
    np.testing.assert_array_equal(asked["lag_coefs"].values[0], lag)
    np.testing.assert_array_equal(fc.dataset["cond_chol"].values[0, :, 0], chol)
    irf = fit.irf(4, 50, seed=1)
    assert irf.data.shape == (1, 50, 5, 3, 3)
    # the posterior is the module's closed form at the fit's hyperparameters
    posts = nig.structural_posteriors(data, model.mean.prior.minnesota(), fit.hyper, LAGS, DET)
    for got, want in zip(fit.posterior(), posts, strict=True):
        np.testing.assert_array_equal(got.theta_hat, want.theta_hat)
        np.testing.assert_array_equal(got.R, want.R)
    assert fit.log_ml() == sum(p.log_ml for p in posts)


def test_2020_warning_does_not_advise_lenza_primiceri():
    """Phase 1's 2020 advice (``outliers='lenza-primiceri'``) is refused by AsymmetricConjugate at
    construction (E-8), so its fit advises what the model accepts (the R6 review of PR #22)."""
    rng = np.random.default_rng(7)
    index = pd.period_range("2005Q1", periods=72, freq="Q")
    frame = pd.DataFrame(rng.standard_normal((72, 3)), index=index, columns=["v0", "v1", "v2"])
    with pytest.warns(UserWarning) as record:
        ac(hierarchy="fixed").fit(rb.VARData.from_frame(frame, tcodes=1))
    notes = [str(w.message) for w in record if "covers 2020" in str(w.message)]
    assert notes and all("consider outliers='lenza-primiceri'" not in m for m in notes)
    assert all("not available with AsymmetricConjugate" in m for m in notes)


def test_fixed_hierarchy_uses_the_prior_values(data):
    fit = fitted(ac(hierarchy="fixed", lambda_=0.3, cross=0.4, contemp=0.6), data)
    assert fit.optimization is None and fit.converged and fit.summary().empty
    assert (fit.hyper["lambda"], fit.hyper["cross"], fit.hyper["contemp"]) == (0.3, 0.4, 0.6)
    prior = rb.Minnesota(form="structural", lambda_=0.3, cross=0.4, contemp=0.6)
    assert fit.log_ml() == nig.structural_log_ml(data, prior, {}, LAGS, DET)


def test_draws_match_the_exact_posterior_moments(data):
    """Monte Carlo means of ``draw_parameters`` against ``posterior_moments`` (``|z| ≤ 4.5`` over
    ``k·n + n(n+1)/2 = 27`` quantities; iid draws)."""
    fit = fitted(ac(hierarchy="fixed", cross=0.4, contemp=0.6), data)
    B, sigma = fit.posterior_moments()
    coefs, chol = fit.draw_parameters(20_000, 11)
    draws_sigma = chol @ np.swapaxes(chol, -1, -2)
    for draws, mean in ((coefs, B), (draws_sigma, sigma)):
        z = (draws.mean(axis=0) - mean) / (draws.std(axis=0, ddof=1) / np.sqrt(len(draws)))
        assert np.nanmax(np.abs(z)) < 4.5, z


# =========================================================================== A9 at cross = contemp = 1


A9_CASES = {
    "plain": {},
    "soc+sur": dict(soc=1.5, sur=2.0),
    "means+trendless": dict(mean={"v0": 0.9, "v2": 0.5}, soc=0.8, soc_scope="all"),
}


@pytest.mark.parametrize("name", list(A9_CASES))
def test_a9_log_ml_and_posterior_moments_equal_analytic_fit(data, name):
    case = dict(lambda_=0.3, alpha=1.7, var_const=10.0, hierarchy="fixed", **A9_CASES[name])
    structural = fitted(ac(cross=1.0, contemp=1.0, **case), data)
    phase1 = fitted(conjugate(**case), data)
    psi = phase1.hyper["psi"]
    hypers = [{}, {"lambda": 0.6, "alpha": 1.2}, {"psi": psi * np.linspace(0.7, 1.4, len(psi))}]
    if "soc" in case:
        hypers.append({"soc": 4.0})
    for hyper in hypers:
        assert structural.log_ml(hyper) == pytest.approx(phase1.log_ml(hyper), abs=1e-10), hyper
        B, sigma = structural.posterior_moments(hyper)
        niw = phase1.posterior(hyper or None)
        assert_allclose(B, niw.mean, rtol=1e-10, atol=1e-10 * np.abs(niw.mean).max())
        assert_allclose(sigma, niw.sigma_mean, rtol=1e-10, atol=1e-10 * np.abs(sigma).max())


@pytest.mark.parametrize(
    ("hyper", "dummy"), [(("lambda",), None), (("lambda", "soc"), 1.0)], ids=["lambda", "soc"]
)
def test_a9_eb_lambda_equals_analytic_fit(data, hyper, dummy):
    structural = fitted(ac(cross=1.0, contemp=1.0, hyper=hyper, soc=dummy), data)
    phase1 = fitted(conjugate(hyper=hyper, soc=dummy), data)
    assert structural.converged and phase1.converged
    for name in hyper:
        assert abs(structural.hyper[name] - phase1.hyper[name]) <= 1e-6, name
    assert structural.log_ml() == pytest.approx(phase1.log_ml(), abs=1e-8)


# =========================================================================== A9: EB with free cross


def _eb_objective(data: rb.VARData, model: rb.BVAR, free: tuple[str, ...], fixed: dict):
    """The EB objective of P3 §4.3 in ``z = log h`` over ``free``: ``Σ_i log p(y_i | h)`` plus
    the GLP Gamma log density of ``λ`` (natural scale, SciPy; mode 0.2, s.d. 0.4); the flat prior
    on ``log cross``/``log contemp`` adds a constant."""
    prior = model.mean.prior.minnesota()
    theta = (-0.2 + np.sqrt(0.2**2 + 4 * 0.4**2)) / 2.0  # Gamma(k, θ): mode (k−1)θ, var kθ²
    gamma = stats.gamma(a=0.2 / theta + 1.0, scale=theta)

    def f(z: np.ndarray) -> float:
        h = {**fixed, **{name: float(np.exp(v)) for name, v in zip(free, z, strict=True)}}
        value = nig.structural_log_ml(data, prior, h, LAGS, DET)
        return value + (gamma.logpdf(h["lambda"]) if "lambda" in free else 0.0)

    return f


def _projected_gradient(f, z: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: float = 1e-4):
    """``max_i |g_i|`` of the gradient of the maximised ``f`` with the components that point out
    of the box at an active bound set to 0; Richardson-extrapolated differences (one-sided at a
    bound)."""
    out = 0.0
    for i in range(z.size):
        e = np.zeros(z.size)
        e[i] = 1.0
        at_lo, at_hi = z[i] - lo[i] < 1e-9, hi[i] - z[i] < 1e-9

        def d(step: float, i=i, e=e, at_lo=at_lo, at_hi=at_hi) -> float:
            if at_lo or at_hi:
                s = step if at_lo else -step
                return (-3 * f(z) + 4 * f(z + s * e) - f(z + 2 * s * e)) / (2 * s)
            return (f(z + step * e) - f(z - step * e)) / (2 * step)

        g = (4 * d(h / 2) - d(h)) / 3
        if (at_lo and g < 0) or (at_hi and g > 0):
            g = 0.0
        out = max(out, abs(g))
    return out


EB_FREE = {
    "lambda+cross": dict(),
    "lambda+cross+contemp": dict(hyper=("lambda", "cross", "contemp")),
}


@pytest.mark.parametrize("name", list(EB_FREE))
def test_a9_free_cross_eb_passes_phase1s_stationarity_test(data, name):
    fit = fitted(ac(**EB_FREE[name]), data)
    assert fit.converged and fit.optimization["predicted_gain"] < 1e-6  # Phase 1's rule
    lo, hi = LOG_FLAT_BOX
    assert lo <= fit.hyper["cross"] <= hi


# A9's criterion: the projected gradient at the returned mode is below 1e-6. eb_structural's free-scale
# path (cross or contemp free) runs L-BFGS-B with tightened tolerances for it (P3 Q38, owner
# 2026-10-25); with the Phase 2 settings it stopped at 4.5e-6 (lambda+cross) and 1.4e-6 (+contemp).
@pytest.mark.parametrize("name", list(EB_FREE))
def test_a9_free_cross_eb_projected_gradient(data, name):
    model = ac(**EB_FREE[name])
    fit = fitted(model, data)
    free = tuple(n for n in fit.summary().index)
    hp = model.mean.prior.hyperpriors.lambda_
    bounds = {"lambda": (hp.lower, hp.upper), "cross": LOG_FLAT_BOX, "contemp": LOG_FLAT_BOX}
    lo = np.log([bounds[n][0] for n in free])
    hi = np.log([bounds[n][1] for n in free])
    z = np.clip(np.log([fit.hyper[n] for n in free]), lo, hi)
    fixed = {k: v for k, v in fit.hyper.items() if k not in free}
    pg = _projected_gradient(_eb_objective(data, model, free, fixed), z, lo, hi)
    assert pg < 1e-6, f"projected gradient {pg:.3g} at {dict(zip(free, np.exp(z), strict=True))}"


# =========================================================================== A9: MH vs quadrature


def _quadrature_quantiles(data: rb.VARData, model: rb.BVAR) -> dict[str, dict[str, np.ndarray]]:
    """Quantiles of ``λ`` and ``cross`` under ``p(λ, cross | y) ∝ p(y | λ, cross) p(λ) p(cross)``:
    ``p(λ)`` the GLP Gamma (mode 0.2, s.d. 0.4) on ``[1e-4, 5]``, ``p(cross) ∝ 1/cross`` on
    ``[1e-3, 10]`` (flat in ``log cross``, the design's prior: key ``"design"``), on a tensor grid
    in ``(log λ, log cross)``: a coarse 41 × 41 grid over the box locates the mass, an 81 × 81
    grid over the region within 30 log units of its maximum integrates it (Simpson, the CDF
    inverted by monotone cubic interpolation; against a 481 × 481 grid the quantiles agree to
    2e-4 in log, about 1% of their MH MCSE). Key
    ``"flat-cross"``: the same with ``p(cross)`` flat on the natural scale (the negative control:
    the other face of the flat-log prior)."""
    prior = model.mean.prior.minnesota()
    window = nig._Window(data, prior, LAGS, DET)
    base = window.complete({})
    theta = (-0.2 + np.sqrt(0.2**2 + 4 * 0.4**2)) / 2.0
    gamma = stats.gamma(a=0.2 / theta + 1.0, scale=theta)

    def log_density(zl: np.ndarray, zc: np.ndarray) -> np.ndarray:
        out = np.empty((zl.size, zc.size))
        for a, u in enumerate(zl):
            lam = float(np.exp(u))
            for b, v in enumerate(zc):
                h = {**base, "lambda": lam, "cross": float(np.exp(v))}
                out[a, b] = window.log_ml(h)
            out[a] += gamma.logpdf(lam) + u  # density of log λ; log cross: flat
        return out

    box = np.log([[1e-4, 5.0], LOG_FLAT_BOX])
    zl, zc = np.linspace(*box[0], 41), np.linspace(*box[1], 41)
    coarse = log_density(zl, zc)
    keep = coarse > coarse.max() - 30.0
    rows, cols = np.flatnonzero(keep.any(axis=1)), np.flatnonzero(keep.any(axis=0))
    step_l, step_c = zl[1] - zl[0], zc[1] - zc[0]
    fl = np.linspace(
        max(box[0, 0], zl[rows[0]] - step_l), min(box[0, 1], zl[rows[-1]] + step_l), 81
    )
    fc = np.linspace(
        max(box[1, 0], zc[cols[0]] - step_c), min(box[1, 1], zc[cols[-1]] + step_c), 81
    )
    fine = log_density(fl, fc)
    out = {}
    for key, log_dens in (("design", fine), ("flat-cross", fine + fc[None, :])):
        dens = np.exp(log_dens - log_dens.max())
        out[key] = {}
        for name, grid, marginal in (
            ("lambda", fl, integrate.simpson(dens, x=fc, axis=1)),
            ("cross", fc, integrate.simpson(dens, x=fl, axis=0)),
        ):
            cdf = integrate.cumulative_simpson(marginal, x=grid, initial=0.0)
            cdf /= cdf[-1]
            out[key][name] = np.exp(interpolate.PchipInterpolator(cdf, grid)(QUANTILES))
    return out


def test_a9_mh_against_the_posterior_by_quadrature(data):
    model = ac(hierarchy="mh")  # free: lambda and cross (the "auto" set); default MHSettings
    fit = fitted(model, data)
    assert fit.converged and fit.mh.names == ("lambda", "cross")
    chains = fit.mh.chains  # (chains, draws, 2), natural scale
    zmax = {}
    for key, reference in _quadrature_quantiles(data, model).items():
        indicators = np.stack(
            [chains[:, :, j, None] <= reference[name][None, None, :]
             for j, name in enumerate(fit.mh.names)], axis=2,
        ).astype(np.float64)  # (chains, draws, 2, quantiles)  # fmt: skip
        f_hat = indicators.mean(axis=(0, 1))
        mcse = diagnostics.mcse_mean_many(indicators)
        z = (f_hat - np.array(QUANTILES)[None, :]) / mcse
        assert np.all(np.isfinite(z))
        zmax[key] = float(np.abs(z).max())
    assert zmax["design"] <= bonferroni(10), zmax
    assert zmax["flat-cross"] > bonferroni(10), zmax  # the comparison has power against it


# =========================================================================== A9 at cross ≠ 1 (weekly)


@pytest.mark.slow
def test_a9_forecasts_against_mcmc_mode_ii(data):
    """The NIG closed form at ``cross = 0.5``, ``contemp = 0.7`` against MCMC Mode II with the
    same fixed hyperparameters and ψ: forecast quantiles of every variable at ``h = 1..4``
    (``m = 60``), the analytic forecast's 40,000 iid draws as the reference."""
    psi = tuple(float(v) for v in fitted(ac(hierarchy="fixed"), data).psi)
    settings = dict(lambda_=0.3, cross=0.5, contemp=0.7, psi=psi, hierarchy="fixed")
    analytic = fitted(ac(**settings), data)
    mcmc_model = rb.BVAR(
        mean=rb.Linear(rb.Minnesota(form="structural", **settings)),
        vol=rb.Constant("iw"),
        lags=LAGS,
        deterministic=DET,
        mcmc=rb.MCMCSettings(chains=4, draws=5000, burn=1000, seed=17),
    )
    mcmc = fitted(mcmc_model, data)
    H, N_ref = 4, 40_000
    ref = analytic.forecast(H, N_ref, seed=5).dataset["paths"].values[0]  # (N_ref, H, n)
    paths = mcmc.forecast(H, 20_000, seed=6).dataset["paths"].values  # (chains, draws, H, n)
    x = np.quantile(ref, QUANTILES, axis=0)  # (Q, H, n)
    indicators = (paths[:, :, None] <= x[None, None]).astype(np.float64)
    f_hat = indicators.mean(axis=(0, 1))
    q = np.array(QUANTILES)[:, None, None]
    se = np.sqrt(diagnostics.mcse_mean_many(indicators) ** 2 + q * (1 - q) / N_ref)
    z = (f_hat - q) / se
    assert np.abs(z).max() <= bonferroni(z.size), np.abs(z).max()


# =========================================================================== other rules


def test_psi_is_free_only_at_cross_and_contemp_one(data):
    """At ``cross = contemp = 1`` the conjugate twin's EB mode (ψ included) is the structural one
    (A1); with ``contemp ≠ 1`` ψ cannot be estimated (E-15 covers ``cross`` at construction)."""
    structural = fitted(ac(cross=1.0, hyper=("lambda", "psi")), data)
    phase1 = fitted(conjugate(hyper=("lambda", "psi")), data)
    assert structural.converged
    assert structural.hyper["lambda"] == phase1.hyper["lambda"]
    np.testing.assert_array_equal(structural.psi, phase1.psi)
    assert structural.log_ml() == pytest.approx(phase1.log_ml(), abs=1e-10)
    assert list(structural.summary().index) == ["lambda", "psi_v0", "psi_v1", "psi_v2"]
    with pytest.raises(ValueError, match="empirical Bayes over psi needs cross = contemp = 1"):
        fitted(ac(cross=1.0, contemp=0.5, hyper=("lambda", "psi")), data)


def test_r8_note_from_four_variables_when_not_the_niw_prior():
    data4 = var_data(n=4, T=70, seed=5)
    note = [
        any(ORDERING_NOTE in w for w in fitted(ac(hierarchy="fixed", **kw), data4).warnings)
        for kw in (dict(cross=1.0), dict(cross=0.5), dict(cross=1.0, contemp=0.5))
    ]
    assert note == [False, True, True]
    assert not any(ORDERING_NOTE in w for w in fitted(ac(hierarchy="fixed"), var_data()).warnings)


def test_the_structural_nig_fit_never_loads_jax():
    """P3 §1.1: the analytic engine for AsymmetricConjugate is NumPy only (fit with EB and MH,
    forecast with ``lag_coefs``, IRFs, posterior moments, the ordering check's prior mode)."""
    code = (
        "import sys\n"
        "import numpy as np, pandas as pd\n"
        "import rai_bvar as rb\n"
        "from rai_bvar.structural.ordering import prior_covariance_draws\n"
        "rng = np.random.default_rng(0)\n"
        "idx = pd.period_range('1980Q1', periods=60, freq='Q')\n"
        "d = rb.VARData.from_frame(pd.DataFrame(rng.standard_normal((60, 3)), index=idx,\n"
        "                                       columns=list('abc')))\n"
        "mh = rb.MHSettings(draws=20, burn=10, chains=2)\n"
        "for prior in (rb.AsymmetricConjugate(soc=1.0), rb.AsymmetricConjugate(hierarchy='mh', mh=mh)):\n"
        "    m = rb.BVAR(mean=rb.Linear(prior), lags=2)\n"
        "    fit = m.fit(d)\n"
        "    fit.forecast(2, 10, seed=1, lag_coefs=True); fit.irf(2, 5, seed=1)\n"
        "    fit.posterior_moments(); fit.summary(); prior_covariance_draws(m, d, 50)\n"
        "assert 'jax' not in sys.modules, 'the structural NIG fit loaded JAX'\n"
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)


def test_mh_seed_comes_from_the_fit_or_the_settings(data):
    mh = rb.MHSettings(draws=200, burn=100, chains=2, seed=9)
    a = fitted(ac(hierarchy="mh", mh=mh), data)
    b = fitted(ac(hierarchy="mh", mh=mh), data, seed=9)
    c = fitted(ac(hierarchy="mh", mh=mh), data, seed=10)
    assert a.mh_seed == b.mh_seed == 9 and c.mh_seed == 10
    np.testing.assert_array_equal(a.mh.chains, b.mh.chains)
    assert not np.array_equal(a.mh.chains, c.mh.chains)
    assert c.forecast(2, 20, seed=1).dataset.attrs["mh_seed"] == 10
    coefs, _ = c.draw_parameters(40, 2)  # one draw per pooled MH draw picked
    assert np.isfinite(coefs).all() and not np.allclose(coefs[0], coefs[-1])
