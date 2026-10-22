"""The GLP (2015) hierarchy of the analytic engine (SPEC §4.6, §10.3, §10.6).

* Hyperprior densities against SciPy: Gamma and Beta priors given by mode and s.d. must
  reproduce that mode and s.d.; Inverse-Gamma and Pareto densities must be normalised.
* ``log_prior`` sums the hyperprior terms of exactly the free hyperparameters.
* ``hierarchy="eb"`` returns the maximiser of ``log p(Y | h) + log p(h)``: checked against a dense
  grid in ``λ`` and a coarse grid in ``(λ, μ, δ)`` polished by Nelder-Mead, and against the bound
  when the mode lies on it.
* ``hierarchy="mh"`` samples ``p(λ | Y) ∝ p(Y | λ) p(λ)`` (natural-scale density, so the
  log-scale random walk must carry the log-Jacobian): MH quantiles against the quadrature
  posterior of ``λ`` within Monte Carlo error, also when a bound truncates the posterior.
* Sampler bookkeeping: proposals outside the bounds are rejected, the step size adapts only during
  burn-in, acceptance counts only the sampling phase, seeds reproduce chains bitwise, and the
  ``converged`` flag follows the SPEC §10.6 defaults (more than one chain, R-hat < 1.01,
  bulk ESS > 400).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, stats

import rai_bvar as rb
from rai_bvar.data.outliers import COVID_NAMES
from rai_bvar.diagnostics import ess_bulk, rhat
from rai_bvar.engines.analytic import ConjugateProblem

FIXTURES = Path(__file__).resolve().parents[1] / "parity" / "fixtures"
LAGS = 4
PROBS = (0.05, 0.25, 0.5, 0.75, 0.95)
MC_TOLERANCE = 4.5  # Monte Carlo standard errors (5 correlated comparisons per chain set)


# --------------------------------------------------------------------------- data and helpers


def _load(name: str, start: str | None = None) -> rb.VARData:
    frame = pd.read_csv(FIXTURES / name, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    if start is not None:
        frame.index = pd.period_range(start, periods=len(frame), freq="Q")
    return rb.VARData.from_frame(frame, tcodes=1)


@pytest.fixture(scope="module")
def qd() -> rb.VARData:
    """GDP, prices (100·log levels) and the Fed funds rate, 1960Q1–2019Q4."""
    return _load("qd_small_levels.csv")


@pytest.fixture(scope="module")
def sim3() -> rb.VARData:
    """Simulated persistent VAR(3 variables), 1970Q1–2019Q4."""
    return _load("sim_var3.csv")


@pytest.fixture(scope="module")
def covid_data() -> rb.VARData:
    """The simulated sample shifted to 1972Q1–2021Q4, so the sample reaches 2020Q1."""
    return _load("sim_var3.csv", start="1972Q1")


def _bvar(**prior) -> rb.BVAR:
    return rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(mean=1.0, **prior)), lags=LAGS)


def _lambda_prior(lower: float = 1e-4, upper: float = 5.0) -> rb.GLPHyperpriors:
    """GLP's Gamma(mode 0.2, s.d. 0.4) on λ with the given hard bounds."""
    return rb.GLPHyperpriors(lambda_=rb.GammaPrior(mode=0.2, sd=0.4, lower=lower, upper=upper))


def _lambda_quadrature(fit, lower: float, upper: float, size: int = 8001):
    """Exact posterior of λ on ``[lower, upper]``: grid, CDF and mean by the trapezoid rule."""
    grid = np.geomspace(lower, upper, size)
    log_post = np.array([fit.log_posterior({"lambda": float(x)}) for x in grid])
    density = np.exp(log_post - log_post.max())
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0.0)
    mass = cdf[-1]
    return grid, cdf / mass, float(integrate.trapezoid(grid * density, grid) / mass)


def _mcse(x: np.ndarray, batches: int = 20) -> float:
    """Batch-means Monte Carlo s.e. of ``x.mean()`` for ``x`` of shape (chains, draws).

    Floored at the iid standard error: positively autocorrelated chains cannot do better.
    """
    chains, draws = x.shape
    size = draws // batches
    means = x[:, : size * batches].reshape(chains, batches, size).mean(axis=2).ravel()
    batch = means.std(ddof=1) / np.sqrt(means.size)
    return float(max(batch, np.sqrt(x.var() / x.size)))


def _assert_matches_quadrature(fit, lower: float, upper: float) -> None:
    grid, cdf, mean = _lambda_quadrature(fit, lower, upper)
    draws = fit.mh.chains[:, :, 0]
    report = []
    for p in PROBS:
        q = float(np.interp(p, cdf, grid))
        below = (draws <= q).astype(np.float64)
        z = (below.mean() - p) / _mcse(below)
        report.append((f"P(λ ≤ q{p:g})", below.mean(), p, z))
    report.append(("mean", draws.mean(), mean, (draws.mean() - mean) / _mcse(draws)))
    bad = [r for r in report if not abs(r[3]) < MC_TOLERANCE]
    assert not bad, "MH vs quadrature (statistic, MH, exact, z): " + repr(bad)


# --------------------------------------------------------------------------- hyperprior densities


@pytest.mark.parametrize(
    ("mode", "sd"), [(0.2, 0.4), (1.0, 1.0), (2.0, 0.25), (0.05, 2.0), (5.0, 0.01)]
)
def test_gamma_prior_mode_sd_round_trip(mode, sd):
    prior = rb.GammaPrior(mode=mode, sd=sd, lower=1e-8, upper=1e8)
    k, theta = prior.shape, prior.scale
    assert k > 1.0  # a positive mode needs shape > 1
    np.testing.assert_allclose((k - 1.0) * theta, mode, rtol=1e-12)
    reference = stats.gamma(a=k, scale=theta)
    np.testing.assert_allclose(reference.std(), sd, rtol=1e-12)
    found = optimize.minimize_scalar(
        lambda x: -prior.logpdf(x),
        bounds=(mode / 10, mode * 10),
        method="bounded",
        options={"xatol": 1e-10 * mode},
    )
    np.testing.assert_allclose(found.x, mode, rtol=1e-6)
    xs = mode + sd * np.array([-0.5, 0.0, 0.5, 1.0, 3.0])
    xs = xs[xs > 0]
    # atol: at shape 2.5e5 the log-density is a difference of terms of size ~3e6 (float64 ~1e-9)
    np.testing.assert_allclose(
        [prior.logpdf(x) for x in xs], reference.logpdf(xs), rtol=1e-10, atol=1e-8
    )
    lo, hi = reference.ppf([1e-12, 1.0 - 1e-12])
    mass, _ = integrate.quad(lambda x: np.exp(prior.logpdf(x)), lo, hi, points=[mode], limit=200)
    np.testing.assert_allclose(mass, 1.0, atol=1e-8)


@pytest.mark.parametrize(
    ("scale", "shape"), [(1.0, 2.0), (0.5, 3.5), (2.0, 0.7), (0.02**2, 0.02**2)]
)
def test_inverse_gamma_prior_matches_scipy_and_is_normalised(scale, shape):
    prior = rb.InvGammaPrior(scale=scale, shape=shape)
    reference = stats.invgamma(a=shape, scale=scale)
    xs = np.array([1e-4, 0.01, 0.3, 1.0, 4.0, 100.0])
    np.testing.assert_allclose([prior.logpdf(x) for x in xs], reference.logpdf(xs), rtol=1e-10)
    found = optimize.minimize_scalar(
        lambda u: -prior.logpdf(np.exp(u)), bracket=(np.log(scale) - 1, np.log(scale) + 1)
    )
    np.testing.assert_allclose(np.exp(found.x), scale / (shape + 1.0), rtol=1e-6)
    if shape >= 0.5:  # the GLP default (shape 4e-4) has too heavy a tail for quadrature
        mass, _ = integrate.quad(
            lambda u: np.exp(prior.logpdf(np.exp(u)) + u),
            -60.0,
            60.0,
            points=[np.log(scale)],
            limit=400,
        )
        np.testing.assert_allclose(mass, 1.0, atol=1e-8)


@pytest.mark.parametrize(("mode", "sd"), [(0.8, 0.2), (0.5, 0.1), (0.2, 0.05), (0.95, 0.03)])
def test_beta_prior_mode_sd_solve(mode, sd):
    prior = rb.BetaPrior(mode=mode, sd=sd)
    a, b = prior.a, prior.b
    assert a > 1.0 and b > 1.0  # interior mode
    np.testing.assert_allclose((a - 1.0) / (a + b - 2.0), mode, rtol=1e-10)
    reference = stats.beta(a, b)
    np.testing.assert_allclose(reference.std(), sd, rtol=1e-8)
    xs = np.array([0.01, 0.2, 0.5, mode, 0.9, 0.99])
    np.testing.assert_allclose([prior.logpdf(x) for x in xs], reference.logpdf(xs), rtol=1e-10)
    found = optimize.minimize_scalar(
        lambda x: -prior.logpdf(x),
        bounds=(1e-6, 1 - 1e-6),
        method="bounded",
        options={"xatol": 1e-12},
    )
    np.testing.assert_allclose(found.x, mode, rtol=1e-6)
    mass, _ = integrate.quad(lambda x: np.exp(prior.logpdf(x)), 0.0, 1.0, points=[mode])
    np.testing.assert_allclose(mass, 1.0, atol=1e-8)


@pytest.mark.parametrize(
    ("mode", "sd"),
    [(0.0, 0.1), (1.0, 0.1), (1.2, 0.1), (0.5, 0.3)],  # sd > 1/√12: wider than the uniform
)
def test_beta_prior_rejects_impossible_mode_or_sd(mode, sd):
    with pytest.raises(ValueError):
        rb.BetaPrior(mode=mode, sd=sd)


@pytest.mark.parametrize(("scale", "shape"), [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)])
def test_pareto_prior_matches_scipy_and_is_normalised(scale, shape):
    prior = rb.ParetoPrior(scale=scale, shape=shape)
    assert prior.lower == scale  # the support starts at x_m: the hard lower bound
    reference = stats.pareto(b=shape, scale=scale)
    xs = scale * np.array([1.0, 1.5, 3.0, 40.0])
    np.testing.assert_allclose([prior.logpdf(x) for x in xs], reference.logpdf(xs), rtol=1e-12)
    mass, _ = integrate.quad(lambda x: np.exp(prior.logpdf(x)), scale, np.inf)
    np.testing.assert_allclose(mass, 1.0, atol=1e-8)


def test_glp_and_lenza_primiceri_default_hyperpriors():
    hp = rb.GLPHyperpriors()
    assert (hp.lambda_.mode, hp.lambda_.sd) == (0.2, 0.4)
    assert (hp.soc.mode, hp.soc.sd) == (1.0, 1.0)
    assert (hp.sur.mode, hp.sur.sd) == (1.0, 1.0)
    assert (hp.psi.scale, hp.psi.shape) == (0.02**2, 0.02**2)
    lp = rb.LenzaPrimiceri()
    assert (lp.scale_prior.scale, lp.scale_prior.shape) == (1.0, 1.0)
    assert (lp.rho_prior.mode, lp.rho_prior.sd) == (0.8, 0.2)
    (s_lo, _), _, _, (r_lo, r_hi) = lp.bounds()
    assert s_lo == 1.0 and 0.0 < r_lo < r_hi < 1.0


# --------------------------------------------------------------------------- log_prior


def _expected_log_prior(problem: ConjugateProblem, h: dict) -> float:
    """Hyperprior terms written out with SciPy for the free blocks of ``problem``."""
    hp = problem.prior.hyperpriors
    free = problem.free.names
    total = 0.0
    for name in ("lambda", "alpha", "soc", "sur"):
        if name in free:
            g = getattr(hp, "lambda_" if name == "lambda" else name)
            total += stats.gamma(a=g.shape, scale=g.scale).logpdf(h[name])
    if any(nm.startswith("psi_") for nm in free):
        c = problem.d - problem.n - 1.0  # ψ_j / (d − n − 1) ~ IG(scale, shape)
        ig = stats.invgamma(a=hp.psi.shape, scale=hp.psi.scale)
        total += float(np.sum(ig.logpdf(np.asarray(h["psi"]) / c) - np.log(c)))
    if COVID_NAMES[0] in free:
        lp = problem.outliers
        s0, s1, s2, rho = h["covid"]
        par = stats.pareto(b=lp.scale_prior.shape, scale=lp.scale_prior.scale)
        total += float(np.sum(par.logpdf([s0, s1, s2])))
        total += stats.beta(lp.rho_prior.a, lp.rho_prior.b).logpdf(rho)
    return float(total)


LOG_PRIOR_CASES = {
    "auto-lambda": (dict(hierarchy="eb"), ("lambda",)),
    "auto-soc": (dict(hierarchy="eb", soc=1.0), ("lambda", "soc")),
    "auto-soc-sur": (dict(hierarchy="mh", soc=1.0, sur=1.0), ("lambda", "soc", "sur")),
    "lambda-only-dummies-fixed": (
        dict(hierarchy="eb", soc=1.0, sur=1.0, hyper=("lambda",)),
        ("lambda",),
    ),
    "lambda-alpha-psi": (
        dict(hierarchy="eb", hyper=("psi", "alpha", "lambda")),
        ("lambda", "alpha", "psi_GDPC1", "psi_GDPCTPI", "psi_FEDFUNDS"),
    ),
    "fixed": (dict(hierarchy="fixed", soc=1.0, sur=1.0), ()),
}


@pytest.mark.parametrize("case", list(LOG_PRIOR_CASES))
def test_log_prior_covers_exactly_the_free_hyperparameters(qd, case):
    kwargs, names = LOG_PRIOR_CASES[case]
    problem = ConjugateProblem(qd, rb.Minnesota(mean=1.0, **kwargs), LAGS, "c", None)
    assert problem.free.names == names
    values = {"lambda": 0.37, "alpha": 2.3, "psi": problem.default["psi"] * 1.3}
    values |= {d: v for d, v in (("soc", 0.7), ("sur", 1.9)) if kwargs.get(d) is not None}
    h = problem.complete(values)
    expected = _expected_log_prior(problem, h)
    np.testing.assert_allclose(problem.log_prior(h), expected, rtol=1e-12, atol=1e-12)
    if not names:
        assert problem.log_prior(h) == 0.0


def test_log_prior_ignores_hyperparameters_that_are_not_free(qd):
    prior = rb.Minnesota(mean=1.0, soc=1.0, sur=1.0, hierarchy="eb", hyper=("lambda",))
    problem = ConjugateProblem(qd, prior, LAGS, "c", None)
    base = problem.complete({"lambda": 0.3})
    others = [
        {"soc": 0.2},
        {"sur": 7.0},
        {"alpha": 1.3},
        {"psi": base["psi"] * 3.0},
    ]
    for change in others:
        h = problem.complete({"lambda": 0.3, **change})
        assert problem.log_prior(h) == problem.log_prior(base), change
        # ... while each of them does move the likelihood (so the check is not vacuous)
        assert problem.evaluate(h)[0] != pytest.approx(problem.evaluate(base)[0], abs=1e-6)
    lam = problem.complete({"lambda": 0.6})
    assert problem.log_prior(lam) != problem.log_prior(base)


def test_fit_log_posterior_is_log_ml_plus_log_prior_of_free_block(qd):
    fit = _bvar(hierarchy="eb", soc=1.0, sur=1.0).fit(qd)
    h = {"lambda": 0.31, "soc": 0.8, "sur": 2.5}
    hp = rb.GLPHyperpriors()
    expected = sum(
        stats.gamma(a=g.shape, scale=g.scale).logpdf(h[k])
        for k, g in (("lambda", hp.lambda_), ("soc", hp.soc), ("sur", hp.sur))
    )
    np.testing.assert_allclose(fit.log_posterior(h) - fit.log_ml(h), expected, rtol=1e-9)
    fixed = _bvar(hierarchy="fixed", soc=1.0, sur=1.0).fit(qd)
    assert fixed.log_posterior(h) == fixed.log_ml(h)


def test_log_prior_includes_the_covid_block_only_when_estimated(covid_data):
    prior = rb.Minnesota(mean=1.0, hierarchy="eb")
    problem = ConjugateProblem(covid_data, prior, LAGS, "c", rb.LenzaPrimiceri())
    assert problem.free.names == ("lambda", *COVID_NAMES)
    h = problem.complete({"lambda": 0.25, "covid": np.array([4.0, 2.5, 1.7, 0.6])})
    np.testing.assert_allclose(problem.log_prior(h), _expected_log_prior(problem, h), rtol=1e-12)
    fixed = rb.LenzaPrimiceri(fixed=(4.0, 2.5, 1.7, 0.6))
    problem = ConjugateProblem(covid_data, prior, LAGS, "c", fixed)
    assert problem.free.names == ("lambda",)
    h = problem.complete({"lambda": 0.25})
    gamma = rb.GLPHyperpriors().lambda_
    np.testing.assert_allclose(
        problem.log_prior(h),
        stats.gamma(a=gamma.shape, scale=gamma.scale).logpdf(0.25),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="fixed"):
        ConjugateProblem(
            covid_data, replace(prior, hierarchy="fixed"), LAGS, "c", rb.LenzaPrimiceri()
        )


# --------------------------------------------------------------------------- EB mode


@pytest.mark.parametrize("dataset", ["qd", "sim3"])
def test_eb_lambda_is_the_maximiser_on_a_dense_grid(request, dataset):
    data = request.getfixturevalue(dataset)
    fit = _bvar(hierarchy="eb").fit(data)
    assert fit.problem.free.names == ("lambda",)
    assert fit.optimization["success"], fit.optimization
    lam = fit.hyper["lambda"]
    np.testing.assert_allclose(fit.optimization["log_posterior"], fit.log_posterior(), rtol=1e-12)
    grid = np.geomspace(1e-4, 5.0, 4001)  # the default bounds of λ
    values = np.array([fit.log_posterior({"lambda": float(x)}) for x in grid])
    best = int(values.argmax())
    assert 0 < best < grid.size - 1  # interior mode
    assert fit.log_posterior() >= values.max() - 1e-9
    assert grid[best - 1] <= lam <= grid[best + 1]
    refined = optimize.minimize_scalar(
        lambda x: -fit.log_posterior({"lambda": x}),
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    np.testing.assert_allclose(lam, refined.x, rtol=1e-4)
    np.testing.assert_allclose(fit.log_posterior(), -refined.fun, rtol=0, atol=1e-7)


@pytest.mark.parametrize("dataset", ["qd", "sim3"])
def test_eb_lambda_soc_sur_matches_a_coarse_grid(request, dataset):
    data = request.getfixturevalue(dataset)
    fit = _bvar(hierarchy="eb", soc=1.0, sur=1.0).fit(data)
    names = ("lambda", "soc", "sur")
    assert fit.problem.free.names == names
    assert fit.optimization["success"], fit.optimization
    mode = fit.problem.free.vector(fit.hyper)
    lam_grid = np.geomspace(0.03, 3.0, 13)
    dummy_grid = np.geomspace(0.03, 30.0, 13)
    best_value, best_x = -np.inf, None
    for x in itertools.product(lam_grid, dummy_grid, dummy_grid):
        value = fit.log_posterior(dict(zip(names, map(float, x), strict=True)))
        if value > best_value:
            best_value, best_x = value, np.array(x)
    assert fit.log_posterior() >= best_value - 1e-9
    polished = optimize.minimize(
        lambda z: -fit.problem.log_posterior_vector(np.exp(z)),
        np.log(best_x),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000},
    )
    assert polished.success, polished.message
    np.testing.assert_allclose(fit.log_posterior(), -polished.fun, rtol=0, atol=1e-5)
    np.testing.assert_allclose(mode, np.exp(polished.x), rtol=2e-3)


@pytest.fixture(scope="module")
def qd_lambda_mode(qd) -> float:
    return float(_bvar(hierarchy="eb").fit(qd).hyper["lambda"])


@pytest.mark.parametrize(
    ("side", "bound", "start"),
    [
        ("upper", 0.15, 0.2),  # the default start λ = 0.2 lies outside the bounds
        ("upper", 0.15, 0.05),
        ("upper", 0.1, 0.2),
        ("upper", 0.1, 0.05),
        ("lower", 0.5, 0.2),
        ("lower", 0.5, 2.0),
        ("lower", 0.35, 2.0),
    ],
)
def test_eb_mode_on_a_bound(qd, qd_lambda_mode, side, bound, start):
    """The unconstrained mode is λ ≈ 0.30; a bound on the wrong side of it is the constrained mode."""
    lower, upper = (1e-4, bound) if side == "upper" else (bound, 5.0)
    assert (qd_lambda_mode > bound) if side == "upper" else (qd_lambda_mode < bound)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = _bvar(hierarchy="eb", lambda_=start, hyperpriors=_lambda_prior(lower, upper)).fit(qd)
    lam = fit.hyper["lambda"]
    assert lower <= lam <= upper
    inside = np.linspace(lower, upper, 201)[1:-1]
    assert fit.log_posterior() > max(fit.log_posterior({"lambda": float(x)}) for x in inside)
    np.testing.assert_allclose(lam, bound, rtol=1e-6)
    assert fit.optimization["success"], fit.optimization
    assert not [str(w.message) for w in caught]


def test_eb_mode_with_one_coordinate_on_its_bound(qd):
    """With δ capped below its unconstrained mode (≈ 0.96), δ sits on the cap and (λ, μ) equal
    the EB mode with δ fixed at the cap."""
    hp = rb.GLPHyperpriors(sur=rb.GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=0.5))
    fit = _bvar(hierarchy="eb", soc=1.0, sur=1.0, hyperpriors=hp).fit(qd)
    assert fit.optimization["success"], fit.optimization
    np.testing.assert_allclose(fit.hyper["sur"], 0.5, rtol=1e-6)
    pinned = _bvar(hierarchy="eb", soc=1.0, sur=0.5, hyper=("lambda", "soc")).fit(qd)
    np.testing.assert_allclose(
        [fit.hyper["lambda"], fit.hyper["soc"]],
        [pinned.hyper["lambda"], pinned.hyper["soc"]],
        rtol=1e-3,
    )
    # the optimal values differ by the δ hyperprior term only (δ is free only in ``fit``);
    # atol: L-BFGS-B stops at a relative reduction of ~2e-9 of |log posterior| ≈ 585
    gamma = hp.sur
    np.testing.assert_allclose(
        fit.log_posterior() - pinned.log_posterior(),
        stats.gamma(a=gamma.shape, scale=gamma.scale).logpdf(0.5),
        rtol=0,
        atol=1e-5,
    )


# --------------------------------------------------------------------------- MH vs quadrature

FAST_MH = rb.MHSettings(draws=3000, burn=500, chains=2)
SLOW_MH = rb.MHSettings(draws=20000, burn=2000, chains=4)


@pytest.mark.parametrize("upper", [5.0, 0.3], ids=["glp-bounds", "truncated-at-0.3"])
def test_mh_lambda_matches_quadrature_posterior(qd, upper):
    """Fast version (2 × 3000 draws). With ``upper = 0.3`` the bound cuts the posterior near its
    median: the sampler must reject (not clip or reflect) proposals beyond it."""
    hp = rb.GLPHyperpriors() if upper == 5.0 else _lambda_prior(upper=upper)
    fit = _bvar(hierarchy="mh", mh=FAST_MH, hyperpriors=hp).fit(qd)
    assert fit.mh.names == ("lambda",)
    assert fit.mh.chains.shape == (FAST_MH.chains, FAST_MH.draws, 1)
    _assert_matches_quadrature(fit, 1e-4, upper)


@pytest.mark.slow
@pytest.mark.parametrize("upper", [5.0, 0.3], ids=["glp-bounds", "truncated-at-0.3"])
def test_mh_lambda_matches_quadrature_posterior_long_chains(qd, upper):
    hp = rb.GLPHyperpriors() if upper == 5.0 else _lambda_prior(upper=upper)
    fit = _bvar(hierarchy="mh", mh=SLOW_MH, hyperpriors=hp).fit(qd)
    assert fit.converged
    _assert_matches_quadrature(fit, 1e-4, upper)


@pytest.mark.slow
def test_mh_lambda_matches_quadrature_posterior_simulated_data(sim3):
    fit = _bvar(hierarchy="mh", mh=SLOW_MH).fit(sim3)
    _assert_matches_quadrature(fit, 1e-4, 5.0)


# --------------------------------------------------------------------------- MH bookkeeping


def test_mh_draws_respect_bounds(qd):
    """EB mode on qd is (λ, μ, δ) ≈ (0.44, 0.45, 0.96); all three bounds below cut into it."""
    hp = rb.GLPHyperpriors(
        lambda_=rb.GammaPrior(mode=0.2, sd=0.4, lower=1e-4, upper=0.4),
        soc=rb.GammaPrior(mode=1.0, sd=1.0, lower=0.6, upper=50.0),
        sur=rb.GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=0.8),
    )
    settings = rb.MHSettings(draws=1000, burn=300, chains=2)
    fit = _bvar(hierarchy="mh", soc=1.0, sur=1.0, hyperpriors=hp, mh=settings).fit(qd)
    lower, upper = fit.problem.free.lower, fit.problem.free.upper
    draws = fit.mh.chains.reshape(-1, 3)
    assert np.all(draws >= lower) and np.all(draws <= upper)
    # the active bounds are actually visited (otherwise the check is vacuous)
    assert (draws[:, 0] > 0.98 * 0.4).mean() > 0.01
    assert (draws[:, 1] < 1.02 * 0.6).mean() > 0.01
    assert (draws[:, 2] > 0.98 * 0.8).mean() > 0.01
    summary = fit.summary()
    assert np.all(summary[["q05", "q50", "q95"]].to_numpy().T >= lower)
    assert np.all(summary[["q05", "q50", "q95"]].to_numpy().T <= upper)


def test_mh_acceptance_counts_only_the_sampling_phase(qd):
    settings = rb.MHSettings(draws=2000, burn=500, chains=2)
    fit = _bvar(hierarchy="mh", mh=settings).fit(qd)
    for c in range(settings.chains):
        chain = fit.mh.chains[c, :, 0]
        moves = int(np.count_nonzero(chain[1:] != chain[:-1]))
        accepted = fit.mh.acceptance[c] * settings.draws
        # the first kept draw may also be a move away from the last burn-in state
        assert moves <= round(accepted) <= moves + 1, (c, moves, accepted)
    thinned = replace(settings, draws=500, thin=3)
    fit = _bvar(hierarchy="mh", mh=thinned).fit(qd)
    for c in range(thinned.chains):
        chain = fit.mh.chains[c, :, 0]
        moves = np.count_nonzero(chain[1:] != chain[:-1])
        assert 0.0 < fit.mh.acceptance[c] < 1.0
        assert moves <= fit.mh.acceptance[c] * thinned.draws * thinned.thin + 1


@pytest.mark.parametrize("target", [0.2, 0.5])
def test_mh_burn_in_adaptation_reaches_the_target_acceptance(qd, target):
    settings = rb.MHSettings(draws=2000, burn=1000, chains=2, target_acceptance=target)
    fit = _bvar(hierarchy="mh", mh=settings).fit(qd)
    assert abs(fit.mh.acceptance.mean() - target) < 0.1, fit.mh.acceptance


def _recorded_mh(monkeypatch, data, settings, seed=0):
    """Fit with ``hierarchy="mh"`` while recording every point at which the target is evaluated.

    Returns the fit and the λ proposed at each sampling-phase iteration (one evaluation per
    in-bounds proposal; the bounds here are wide enough that no proposal leaves them).
    """
    calls: list[float] = []
    original = ConjugateProblem.log_posterior_vector

    def recording(self, x):
        calls.append(float(np.asarray(x)[0]))
        return original(self, x)

    monkeypatch.setattr(ConjugateProblem, "log_posterior_vector", recording)
    hp = _lambda_prior(lower=1e-6, upper=1e3)
    fit = _bvar(hierarchy="mh", mh=settings, hyperpriors=hp).fit(data, seed=seed)
    monkeypatch.undo()
    return fit, np.array(calls[-settings.draws :])


def _proposal_sizes(fit, proposals: np.ndarray) -> np.ndarray:
    """``|log λ_proposed − log λ_current|`` for sampling iterations 1, …, draws − 1."""
    chain = fit.mh.chains[0, :, 0]
    moved = chain[1:] != chain[:-1]
    # alignment of the recording: every accepted move lands on that iteration's proposal
    np.testing.assert_array_equal(chain[1:][moved], proposals[1:][moved])
    return np.abs(np.log(proposals[1:]) - np.log(chain[:-1]))


def test_mh_step_adapts_during_burn_in_only(qd, monkeypatch):
    """Target acceptance 0.9 is far above the un-tuned acceptance (≈ 0.45), so burn-in adaptation
    must shrink the random-walk step. After burn-in the step must stay fixed: the proposal sizes
    are then ``step · |F z_t|`` with iid ``z_t``, whose median is the same in every window. Had the
    adaptation continued (acceptance after 100 burn-in draws is still ≈ 0.8 < 0.9) the step would
    keep shrinking window after window."""
    adaptive = rb.MHSettings(draws=3000, burn=100, chains=1, target_acceptance=0.9)
    fit, proposals = _recorded_mh(monkeypatch, qd, adaptive)
    sizes = _proposal_sizes(fit, proposals)
    fit0, proposals0 = _recorded_mh(monkeypatch, qd, replace(adaptive, burn=0))
    sizes0 = _proposal_sizes(fit0, proposals0)
    # positive control: burn-in adaptation changed the step, in the direction of the target
    assert np.median(sizes) / np.median(sizes0) < 0.6
    assert fit.mh.acceptance[0] > fit0.mh.acceptance[0]
    for s in (sizes, sizes0):
        windows = np.array([np.median(w) for w in np.array_split(s, 6)]) / np.median(s)
        assert np.all((windows > 0.8) & (windows < 1.25)), windows


# --------------------------------------------------------------------------- seeds


def test_mh_seed_reproduces_chains_bitwise(qd):
    settings = rb.MHSettings(draws=300, burn=100, chains=2)
    model = _bvar(hierarchy="mh", mh=settings)
    a, b = model.fit(qd, seed=11), model.fit(qd, seed=11)
    np.testing.assert_array_equal(a.mh.chains, b.mh.chains)
    np.testing.assert_array_equal(a.mh.acceptance, b.mh.acceptance)
    # fit(seed=s) overrides MHSettings.seed; seed=None uses MHSettings.seed
    c = _bvar(hierarchy="mh", mh=replace(settings, seed=11)).fit(qd)
    np.testing.assert_array_equal(a.mh.chains, c.mh.chains)
    default = model.fit(qd)
    np.testing.assert_array_equal(default.mh.chains, model.fit(qd, seed=settings.seed).mh.chains)
    other = model.fit(qd, seed=12)
    assert not np.array_equal(a.mh.chains, other.mh.chains)
    assert not np.array_equal(a.mh.chains[0], a.mh.chains[1])  # chains are distinct
    # downstream draws given the chains are reproducible too
    np.testing.assert_array_equal(a.draw_parameters(50, rng=3)[0], b.draw_parameters(50, rng=3)[0])
    fa, fb = a.forecast(4, draws=50, seed=5), b.forecast(4, draws=50, seed=5)
    np.testing.assert_array_equal(fa.paths.values, fb.paths.values)


# --------------------------------------------------------------------------- converged flag


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (rb.MHSettings(draws=1500, burn=500, chains=4), True),
        (rb.MHSettings(draws=3000, burn=500, chains=1), False),  # one chain: cannot be assessed
        (rb.MHSettings(draws=3, burn=0, chains=4), False),  # R-hat undefined (NaN)
        (rb.MHSettings(draws=10, burn=0, chains=4), False),  # dispersed starts, R-hat > 1.01
    ],
    ids=["4x1500", "1-chain", "3-draws", "10-draws"],
)
def test_converged_flag(qd, settings, expected):
    fit = _bvar(hierarchy="mh", mh=settings).fit(qd)
    chains = fit.mh.chains[:, :, 0]
    expected_rhat = rhat(chains)
    np.testing.assert_equal(fit.mh.rhat["lambda"], expected_rhat)
    np.testing.assert_equal(fit.mh.ess["lambda"], ess_bulk(chains))
    rule = settings.chains > 1 and bool(np.isfinite(expected_rhat)) and expected_rhat < 1.01
    if expected:
        assert rule
    assert fit.mh.converged is expected
    assert fit.converged is expected
    forecast = fit.forecast(2, draws=20, seed=1)
    assert forecast.dataset.attrs["converged"] == int(expected)


def test_converged_flag_requires_bulk_ess_above_400(qd):
    """SPEC §10.6 defaults: 4 chains, R-hat < 1.01 *and* bulk ESS > 400."""
    settings = rb.MHSettings(draws=400, burn=500, chains=4)
    fit = _bvar(hierarchy="mh", mh=settings).fit(qd, seed=2)  # a seed where R-hat alone passes
    assert fit.mh.rhat["lambda"] < 1.01  # precondition: R-hat alone would pass
    assert fit.mh.ess["lambda"] < 400  # ... but the effective sample is too small
    assert not fit.mh.converged
    assert not fit.converged


@pytest.mark.parametrize("hierarchy", ["eb", "fixed"])
def test_converged_without_mh(qd, hierarchy):
    fit = _bvar(hierarchy=hierarchy).fit(qd)
    assert fit.mh is None
    assert fit.converged is True
    assert fit.forecast(2, draws=20, seed=1).dataset.attrs["converged"] == 1
