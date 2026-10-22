"""Hyperparameter MH under the Lenza-Primiceri COVID scaling (SPEC §4.6, §5.6, §10.6).

The scaling has four parameters ``(s0, s1, s2, ρ)``. With ``j = t − t*`` a parameter is
*informed* when some period of the estimation sample depends on it: ``s0`` needs ``j = 0``,
``s1`` needs ``j = 1``, ``s2`` needs ``j ≥ 2`` and ``ρ`` needs ``j ≥ 3``. Regression tests:

* Forecast origins at ``t*`` and ``t* + 1`` (2020-03 / 2020-04 monthly, 2020Q1 / 2020Q2
  quarterly): the uninformed parameters are neither free nor sampled (their likelihood is flat,
  checked directly), and the chains of the informed ones converge: max R-hat < 1.05 and min bulk
  ESS > 100.
* Rolling windows starting after ``t*`` (``s0``, ``s1`` uninformed): ``ρ`` is sampled on the
  logit scale (checked exactly through the random-walk increments) and the chains converge.
* The MH draws follow the exact posterior: quantiles against quadrature of
  ``p(Y | h) p(h)`` on unconstrained scales (1-D fast, 2-D slow), with a power check that a
  sampler without the log-Jacobian would be caught.
* Forecasts integrate the uninformed parameters over their *truncated* priors under MH (KS
  against the truncated Pareto and Beta written out with SciPy, both for the parameter draws and
  for the scale ratios visible in ``cond_chol``) and keep the prior modes under EB.
* ``ParetoPrior.sample``, ``BetaPrior.sample`` and ``LenzaPrimiceri.sample_prior`` are exact
  (KS and tail quantiles against SciPy) and respect their bounds.
* Adaptation happens during burn-in only: after burn-in the random-walk increments are one fixed
  linear map of the generator's normal draws; chains are reproducible for a seed, thinning is
  pure bookkeeping, and the number of kept draws does not change a chain.

The simulated data are a persistent VAR(1) in levels (tcode 1, so the default own-lag prior
mean of 1 fits) whose shocks are multiplied by the Lenza-Primiceri ``s_t``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, stats
from scipy.special import expit, logit

import rai_bvar as rb
from rai_bvar.data.outliers import COVID_NAMES
from rai_bvar.engines.analytic import ConjugateProblem

T_STAR = {"M": pd.Period("2020-03", freq="M"), "Q": pd.Period("2020Q1", freq="Q")}
TRUTH = {"M": (10.0, 6.0, 4.0, 0.7), "Q": (3.0, 8.0, 4.0, 0.6)}  # (s0, s1, s2, ρ)
FIRST = {"M": "2014-01", "Q": "1995Q1"}  # fast tests
FIRST_LONG = {"M": "2012-01", "Q": "1985Q1"}  # slow tests
ROLL_FIRST, ROLL_LAST = "2010-01", "2023-12"
LAGS = 2
RHAT_MAX, ESS_MIN = 1.05, 100.0
MC_TOLERANCE = 4.5  # Monte Carlo standard errors
KS_ALPHA = 1e-3
PROBS = (0.05, 0.25, 0.5, 0.75, 0.95)

FAST_MH = rb.MHSettings(draws=1000, burn=500, chains=4)
SMALL_MH = rb.MHSettings(draws=400, burn=200, chains=2)
LONG_MH = rb.MHSettings(draws=10000, burn=1000, chains=4)

# origin -> (frequency, informed (s0, s1, s2, ρ))
ORIGINS = {
    "2020-03": ("M", (True, False, False, False)),
    "2020-04": ("M", (True, True, False, False)),
    "2020Q1": ("Q", (True, False, False, False)),
    "2020Q2": ("Q", (True, True, False, False)),
}
CUSTOM_LP = rb.LenzaPrimiceri(
    scale_prior=rb.ParetoPrior(scale=1.0, shape=2.0, upper=50.0),
    rho_prior=rb.BetaPrior(mode=0.6, sd=0.15, lower=0.2, upper=0.95),
)


# --------------------------------------------------------------------------- data and helpers


def oracle_scales(periods: pd.PeriodIndex, t_star: pd.Period, params) -> np.ndarray:
    """``s_t`` written out period by period from the model description."""
    s0, s1, s2, rho = (float(v) for v in params)
    out = []
    for period in periods:
        j = period.ordinal - t_star.ordinal
        if j < 0:
            out.append(1.0)
        elif j <= 2:
            out.append((s0, s1, s2)[j])
        else:
            out.append(1.0 + (s2 - 1.0) * rho ** (j - 2))
    return np.array(out)


def lp_frame(first: str, last: str, freq: str, seed: int) -> pd.DataFrame:
    """Persistent VAR(1) in levels (n = 3) with shocks scaled by ``s_t`` (``t*`` = default)."""
    index = pd.period_range(first, last, freq=freq)
    s = oracle_scales(index, T_STAR[freq], TRUTH[freq])
    rng = np.random.default_rng(seed)
    mu = np.array([2.0, -1.0, 5.0])
    A = np.diag([0.9, 0.8, 0.95]) + 0.05 * (np.ones((3, 3)) - np.eye(3))
    chol = np.diag([0.5, 0.3, 0.8]) @ np.linalg.cholesky(np.full((3, 3), 0.3) + 0.7 * np.eye(3))
    y, prev = np.empty((len(index), 3)), mu.copy()
    for t in range(len(index)):
        prev = mu + A @ (prev - mu) + s[t] * chol @ rng.standard_normal(3)
        y[t] = prev
    return pd.DataFrame(y, index=index, columns=["y1", "y2", "y3"])


def to_data(frame: pd.DataFrame) -> rb.VARData:
    return rb.VARData.from_frame(frame, tcodes=1)


def origin_data(label: str, seed: int = 0, first: dict[str, str] = FIRST) -> rb.VARData:
    """A sample ending at the forecast origin ``label``."""
    freq = ORIGINS[label][0] if label in ORIGINS else "M"
    return to_data(lp_frame(first[freq], label, freq, seed))


def window_data(start: str, seed: int, last: str = ROLL_LAST) -> rb.VARData:
    """A rolling window ``start .. last`` cut from the monthly sample 2010-01 .. 2023-12."""
    frame = lp_frame(ROLL_FIRST, ROLL_LAST, "M", seed)
    return to_data(frame.loc[pd.Period(start, freq="M") : pd.Period(last, freq="M")])


def bvar(hierarchy: str = "mh", outliers="lenza-primiceri", **prior) -> rb.BVAR:
    minnesota = rb.Minnesota(hierarchy=hierarchy, **prior)
    return rb.BVAR(mean=rb.Linear(prior=minnesota), lags=LAGS, outliers=outliers)


def fit_quietly(model: rb.BVAR, data: rb.VARData, seed: int | None = None):
    """Fit, recording the expected UserWarnings (psi fallback note, the library's stricter
    R-hat < 1.01 / ESS > 400 convergence rule). RuntimeWarnings stay errors."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", UserWarning)
        return model.fit(data, seed=seed)


def expected_free(informed) -> tuple[str, ...]:
    return ("lambda", *(name for name, i in zip(COVID_NAMES, informed, strict=True) if i))


def mixing_failures(fit) -> dict[str, tuple[float, float]]:
    """Free hyperparameters failing max R-hat < 1.05 or min bulk ESS > 100."""
    return {
        name: (round(fit.mh.rhat[name], 4), round(fit.mh.ess[name], 1))
        for name in fit.mh.names
        if not (fit.mh.rhat[name] < RHAT_MAX and fit.mh.ess[name] > ESS_MIN)
    }


def mcse(x: np.ndarray, batches: int = 10) -> float:
    """Batch-means Monte Carlo s.e. of ``x.mean()`` for ``x`` of shape (chains, draws), floored
    at the iid standard error."""
    chains, draws = x.shape
    size = draws // batches
    means = x[:, : size * batches].reshape(chains, batches, size).mean(axis=2).ravel()
    return float(max(means.std(ddof=1) / np.sqrt(means.size), np.sqrt(x.var() / x.size)))


def quantile_z_scores(draws: np.ndarray, grid: np.ndarray, cdf: np.ndarray) -> list[tuple]:
    """``(p, P̂(x ≤ q_p), z)`` for the exact quantiles ``q_p`` of a gridded CDF."""
    out = []
    for p in PROBS:
        below = (draws <= float(np.interp(p, cdf, grid))).astype(np.float64)
        out.append((p, round(float(below.mean()), 4), round((below.mean() - p) / mcse(below), 2)))
    return out


def gridded_cdf(grid: np.ndarray, log_density: np.ndarray) -> np.ndarray:
    density = np.exp(log_density - log_density.max())
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0.0)
    return cdf / cdf[-1]


def beta_ab(mode: float, sd: float) -> tuple[float, float]:
    """Beta(a, b) with mode ``(a − 1)/(a + b − 2)`` and the given s.d. (oracle; solves in
    ``log(a + b − 2)``, where the variance falls monotonically)."""

    def excess(log_excess: float) -> float:
        c = 2.0 + np.exp(log_excess)
        a = mode * (c - 2.0) + 1.0
        return a * (c - a) / (c**2 * (c + 1.0)) - sd**2

    c = 2.0 + np.exp(optimize.brentq(excess, -40.0, 80.0, xtol=1e-14))
    a = mode * (c - 2.0) + 1.0
    return a, c - a


def truncated_pareto(prior: rb.ParetoPrior):
    """SciPy's Pareto(shape) on ``[scale, upper]``."""
    return stats.truncpareto(b=prior.shape, c=prior.upper / prior.scale, scale=prior.scale)


def truncated_beta_cdf(prior: rb.BetaPrior):
    dist = stats.beta(*beta_ab(prior.mode, prior.sd))
    lo, hi = dist.cdf([prior.lower, prior.upper])

    def cdf(x):
        return np.clip((dist.cdf(x) - lo) / (hi - lo), 0.0, 1.0)

    return cdf


def truncated_beta_rvs(prior: rb.BetaPrior, size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference draws by rejection from SciPy's Beta (independent of the library's sampler)."""
    dist = stats.beta(*beta_ab(prior.mode, prior.sd))
    out = np.empty(0)
    while out.size < size:
        x = dist.rvs(size=2 * size, random_state=rng)
        out = np.concatenate([out, x[(x >= prior.lower) & (x <= prior.upper)]])
    return out[:size]


def covid_draws(fit, size: int, seed: int) -> np.ndarray:
    """The ``(s0, s1, s2, ρ)`` behind each parameter draw used by forecasts, ``(size, 4)``."""
    rng = np.random.default_rng(seed)
    return np.concatenate([covid for *_, covid in fit._iter_draws(size, rng)])


# --------------------------------------------------------------------------- free set


FREE_SET_CASES = [
    pytest.param("2014-01", "2020-03", "M", (True, False, False, False), id="origin-2020-03"),
    pytest.param("2014-01", "2020-04", "M", (True, True, False, False), id="origin-2020-04"),
    pytest.param("2014-01", "2020-05", "M", (True, True, True, False), id="origin-2020-05"),
    pytest.param("1995Q1", "2020Q1", "Q", (True, False, False, False), id="origin-2020Q1"),
    pytest.param("1995Q1", "2020Q2", "Q", (True, True, False, False), id="origin-2020Q2"),
    pytest.param("2020-02", "2023-12", "M", (False, True, True, True), id="window-2020-02"),
    pytest.param("2020-05", "2023-12", "M", (False, False, True, True), id="window-2020-05"),
    pytest.param("2022-01", "2023-12", "M", (False, False, True, True), id="window-2022-01"),
    pytest.param("2020Q2", "2023Q4", "Q", (False, False, True, True), id="window-2020Q2"),
]


@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
@pytest.mark.parametrize(("first", "last", "freq", "informed"), FREE_SET_CASES)
def test_free_set_holds_exactly_the_informed_covid_components(
    first, last, freq, informed, hierarchy
):
    """With ``p = 2`` lags the estimation sample starts two periods after ``first``. A component
    is free iff the likelihood depends on it: moving an uninformed one leaves ``log p(Y | h)``
    bitwise unchanged, moving an informed one changes it. ρ is the only logit coordinate."""
    data = to_data(lp_frame(first, last, freq, seed=0))
    problem = ConjugateProblem(
        data, rb.Minnesota(hierarchy=hierarchy), LAGS, "c", rb.LenzaPrimiceri()
    )
    informed = np.array(informed)
    names = expected_free(informed)
    assert problem.free.names == names
    np.testing.assert_array_equal(problem.covid_informed, informed)
    np.testing.assert_array_equal(problem.covid_prior_only, ~informed)
    assert problem.free.logit.tolist() == [name == "covid_rho" for name in names]
    # uninformed components start (and, under EB, stay) at the prior mode
    np.testing.assert_array_equal(
        problem.default["covid"][~informed], np.array([1.0, 1.0, 1.0, 0.8])[~informed]
    )
    base = np.array([4.0, 3.0, 2.0, 0.9])  # ρ^22 ≈ 0.1 still reaches a window from 2022-03
    reference = problem.evaluate(problem.complete({"covid": base}))[0]
    for i in range(4):
        moved = base.copy()
        moved[i] = 0.95 if i == 3 else 1.5 * moved[i]
        value = problem.evaluate(problem.complete({"covid": moved}))[0]
        if informed[i]:
            assert abs(value - reference) > 1e-6, COVID_NAMES[i]
        else:
            assert value == reference, COVID_NAMES[i]


# --------------------------------------------------------------------------- MH at the origins


@pytest.mark.parametrize("label", list(ORIGINS))
def test_mh_at_covid_origins_samples_informed_components_and_mixes(label):
    _, informed = ORIGINS[label]
    fit = fit_quietly(bvar(mh=FAST_MH), origin_data(label), seed=1)
    names = expected_free(informed)
    assert fit.mh.names == names
    assert fit.mh.chains.shape == (FAST_MH.chains, FAST_MH.draws, len(names))
    assert list(fit.summary().index) == list(names)
    draws = fit.mh.chains.reshape(-1, len(names))
    assert np.all(draws >= fit.problem.free.lower) and np.all(draws <= fit.problem.free.upper)
    assert np.all(fit.mh.acceptance > 0.1), fit.mh.acceptance
    assert not mixing_failures(fit), (mixing_failures(fit), fit.mh.rhat, fit.mh.ess)


@pytest.mark.slow
@pytest.mark.parametrize("data_seed", [0, 1, 2])
@pytest.mark.parametrize("label", list(ORIGINS))
def test_mh_at_covid_origins_mixes_with_default_settings(label, data_seed):
    """Default MHSettings (4 × 2500 after 1000 burn-in), longer samples, two MH seeds each."""
    data = origin_data(label, seed=data_seed, first=FIRST_LONG)
    failures = {}
    for mh_seed in (1, 2):
        fit = fit_quietly(bvar(), data, seed=mh_seed)
        assert fit.mh.names == expected_free(ORIGINS[label][1])
        if mixing_failures(fit):
            failures[mh_seed] = mixing_failures(fit)
    assert not failures, failures


# --------------------------------------------------------------------------- rolling windows


def test_mh_rolling_window_after_t_star_samples_rho_on_the_logit_scale_and_mixes():
    """Window 2020-05 .. 2022-12: the sample starts at j = 4, so s0 and s1 are uninformed and
    (s2, ρ) are identified only through the decay ``1 + (s2 − 1) ρ^{j−2}``."""
    data = window_data("2020-05", seed=2, last="2022-12")
    fit = fit_quietly(bvar(mh=FAST_MH), data, seed=1)
    assert fit.mh.names == ("lambda", "covid_s2", "covid_rho")
    free = fit.problem.free
    assert free.logit.tolist() == [False, False, True]
    lo, hi = free.z_bounds()
    np.testing.assert_allclose(lo[2], logit(free.lower[2]), rtol=1e-12)
    np.testing.assert_allclose(hi[2], logit(free.upper[2]), rtol=1e-12)
    rho = fit.mh.chains[..., 2]
    assert np.all((rho > free.lower[2]) & (rho < free.upper[2]))
    assert not mixing_failures(fit), (mixing_failures(fit), fit.mh.rhat, fit.mh.ess)


@pytest.mark.slow
@pytest.mark.parametrize("start", ["2020-05", "2021-01", "2022-01"])
@pytest.mark.parametrize("data_seed", [0, 1, 2])
def test_mh_rolling_windows_after_t_star_mix_with_default_settings(data_seed, start):
    """Windows ``start .. 2023-12`` with the default MHSettings and ψ (pre-t* fallback), three
    MH seeds each: every run must reach max R-hat < 1.05 and min bulk ESS > 100."""
    data = window_data(start, seed=data_seed)
    failures = {}
    for mh_seed in (1, 2, 3):
        fit = fit_quietly(bvar(), data, seed=mh_seed)
        assert fit.mh.names == ("lambda", "covid_s2", "covid_rho")
        assert fit.problem.free.logit.tolist() == [False, False, True]
        if mixing_failures(fit):
            failures[mh_seed] = (mixing_failures(fit), np.round(fit.mh.acceptance, 2).tolist())
    assert not failures, f"EB mode covid={np.round(fit.hyper['covid'], 3)}; failures {failures}"


def test_unconstrained_transforms_and_log_jacobian():
    """``z = (log λ, log s2, logit ρ)``; the MH target carries ``log |dx/dz|`` =
    ``log λ + log s2 + log ρ(1 − ρ)``, checked against central differences of ``from_z``."""
    problem = ConjugateProblem(
        window_data("2020-05", seed=2),
        rb.Minnesota(hierarchy="mh"),
        LAGS,
        "c",
        rb.LenzaPrimiceri(),
    )
    free = problem.free
    assert free.names == ("lambda", "covid_s2", "covid_rho")
    for x in (
        np.array([0.3, 2.5, 0.97]),
        np.array([1e-3, 400.0, 1e-4]),
        np.array([2.0, 1.01, 0.5]),
    ):
        z = free.to_z(x)
        np.testing.assert_allclose(z, [np.log(x[0]), np.log(x[1]), logit(x[2])], rtol=1e-13)
        np.testing.assert_allclose(free.from_z(z), x, rtol=1e-13)
        step = 1e-6
        numeric = [
            (free.from_z(z + step * e)[i] - free.from_z(z - step * e)[i]) / (2 * step)
            for i, e in enumerate(np.eye(3))
        ]
        expected = np.log(x[0]) + np.log(x[1]) + np.log(x[2] * (1.0 - x[2]))
        np.testing.assert_allclose(free.log_jacobian(z), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(free.log_jacobian(z), np.log(numeric).sum(), atol=1e-6)


# --------------------------------------------------------------------------- MH vs quadrature


def test_mh_s0_matches_quadrature_at_t_star():
    """Origin 2020-03 with λ fixed (``hyper=()``): only s0 is free and one month informs it. The
    exact posterior of ``log s0`` is ``p(Y | s0) p(s0) s0`` on ``[0, log 1000]``. Power: the same
    comparison rejects the density without the ``s0`` Jacobian (a sampler that forgot it)."""
    fit = fit_quietly(
        bvar(hyper=(), mh=rb.MHSettings(draws=1500, burn=500, chains=4)),
        origin_data("2020-03"),
        seed=1,
    )
    assert fit.mh.names == ("covid_s0",)
    lower, upper = fit.problem.free.lower[0], fit.problem.free.upper[0]
    assert (lower, upper) == (1.0, 1000.0)
    grid = np.linspace(np.log(lower), np.log(upper), 1201)
    log_post = np.array(
        [fit.log_posterior({"covid": np.array([np.exp(u), 1.0, 1.0, 0.8])}) for u in grid]
    )
    draws = np.log(fit.mh.chains[:, :, 0])
    report = quantile_z_scores(draws, grid, gridded_cdf(grid, log_post + grid))
    assert all(abs(z) < MC_TOLERANCE for *_, z in report), report
    wrong = quantile_z_scores(draws, grid, gridded_cdf(grid, log_post))
    assert max(abs(z) for *_, z in wrong) > MC_TOLERANCE, wrong


def _transform(kind: str):
    """``(x(z), log |dx/dz|)`` for a log or logit coordinate."""
    if kind == "log":
        return np.exp, lambda z: z
    return expit, lambda z: -np.logaddexp(0.0, z) - np.logaddexp(0.0, -z)


QUADRATURE_CASES = {
    "origin-2020-04-s0-s1": (
        lambda: origin_data("2020-04"),
        (),
        ("covid_s0", "covid_s1"),
        ("log", "log"),
        lambda x: {"covid": np.array([x[0], x[1], 1.0, 0.8])},
    ),
    "origin-2020-03-lambda-s0": (
        lambda: origin_data("2020-03"),
        "auto",
        ("lambda", "covid_s0"),
        ("log", "log"),
        lambda x: {"lambda": x[0], "covid": np.array([x[1], 1.0, 1.0, 0.8])},
    ),
    "window-2020-05-s2-rho": (
        lambda: window_data("2020-05", seed=2),
        (),
        ("covid_s2", "covid_rho"),
        ("log", "logit"),
        lambda x: {"covid": np.array([1.0, 1.0, x[0], x[1]])},
    ),
    "window-2021-01-s2-rho": (
        lambda: window_data("2021-01", seed=1),
        (),
        ("covid_s2", "covid_rho"),
        ("log", "logit"),
        lambda x: {"covid": np.array([1.0, 1.0, x[0], x[1]])},
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(QUADRATURE_CASES))
def test_mh_two_hyperparameters_match_2d_quadrature(case):
    """Marginal quantiles of the MH draws against a two-stage tensor grid of the exact posterior
    on unconstrained scales (coarse 41² grid over the whole box to locate the mass, then a 121²
    grid over the cells within e^-12 of the maximum; measured to move the quantiles by < 2e-3)."""
    make_data, hyper, names, kinds, hyper_of = QUADRATURE_CASES[case]
    fit = fit_quietly(bvar(hyper=hyper, mh=LONG_MH), make_data(), seed=1)
    assert fit.mh.names == names
    transforms = [_transform(kind) for kind in kinds]
    to_z = [np.log if kind == "log" else logit for kind in kinds]
    lo = [f(b) for f, b in zip(to_z, fit.problem.free.lower, strict=True)]
    hi = [f(b) for f, b in zip(to_z, fit.problem.free.upper, strict=True)]

    def log_density(za: float, zb: float) -> float:
        x = [t[0](z) for t, z in zip(transforms, (za, zb), strict=True)]
        jac = sum(t[1](z) for t, z in zip(transforms, (za, zb), strict=True))
        return fit.log_posterior(hyper_of(x)) + jac

    def tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([[log_density(u, v) for v in b] for u in a])

    a, b = np.linspace(lo[0], hi[0], 41), np.linspace(lo[1], hi[1], 41)
    coarse = tensor(a, b)
    keep = coarse > coarse.max() - 12.0
    ia, ib = np.flatnonzero(keep.any(axis=1)), np.flatnonzero(keep.any(axis=0))
    a = np.linspace(a[max(ia[0] - 1, 0)], a[min(ia[-1] + 1, 40)], 121)
    b = np.linspace(b[max(ib[0] - 1, 0)], b[min(ib[-1] + 1, 40)], 121)
    fine = tensor(a, b)
    density = np.exp(fine - fine.max())
    marginals = (
        (a, integrate.trapezoid(density, b, axis=1)),
        (b, integrate.trapezoid(density, a, axis=0)),
    )
    report = {}
    for i, (grid, marginal) in enumerate(marginals):
        draws = to_z[i](fit.mh.chains[:, :, i])
        cdf = integrate.cumulative_trapezoid(marginal, grid, initial=0.0)
        report[names[i]] = quantile_z_scores(draws, grid, cdf / cdf[-1])
    bad = {k: v for k, v in report.items() if any(abs(z) >= MC_TOLERANCE for *_, z in v)}
    assert not bad, report


# --------------------------------------------------------------------------- forecasts


@pytest.mark.parametrize("lp", [rb.LenzaPrimiceri(), CUSTOM_LP], ids=["default", "custom-priors"])
def test_mh_forecast_draws_uninformed_parameters_from_the_truncated_prior(lp):
    """Origin 2020-03: s0 comes from the chains; s1, s2 (truncated Pareto) and ρ (truncated
    Beta) are uninformed and must be drawn afresh from the configured priors for every draw."""
    fit = fit_quietly(bvar(outliers=lp, mh=SMALL_MH), origin_data("2020-03"), seed=3)
    assert fit.mh.names == ("lambda", "covid_s0")
    S = 4000
    covid = covid_draws(fit, S, seed=7)
    assert covid.shape == (S, 4)
    assert set(covid[:, 0]) <= set(fit.mh.chains[:, :, 1].ravel())
    pareto, prior = truncated_pareto(lp.scale_prior), lp.scale_prior
    for i in (1, 2):
        s = covid[:, i]
        assert np.all((s >= prior.scale) & (s <= prior.upper)), COVID_NAMES[i]
        assert stats.kstest(s, pareto.cdf).pvalue > KS_ALPHA, COVID_NAMES[i]
    rho = covid[:, 3]
    assert np.all((rho >= lp.rho_prior.lower) & (rho <= lp.rho_prior.upper))
    assert stats.kstest(rho, truncated_beta_cdf(lp.rho_prior)).pvalue > KS_ALPHA
    corr = stats.spearmanr(covid[:, 1:]).statistic
    assert np.all(np.abs(corr[np.triu_indices(3, k=1)]) < 4.5 / np.sqrt(S)), corr


@pytest.mark.parametrize("lp", [rb.LenzaPrimiceri(), CUSTOM_LP], ids=["default", "custom-priors"])
def test_mh_forecast_scale_ratios_follow_the_truncated_priors(lp):
    """Black-box view through the forecast factors at origin 2020-03:
    ``cond_chol[T+2]/cond_chol[T+1] = s2/s1`` and ``cond_chol[T+3]/cond_chol[T+2] =
    (1 + (s2 − 1)ρ)/s2``, against reference draws from SciPy (two-sample KS)."""
    fit = fit_quietly(bvar(outliers=lp, mh=SMALL_MH), origin_data("2020-03"), seed=3)
    S = 4000
    factor = fit.forecast(3, draws=S, seed=11).stacked("cond_chol")[:, :, 0, 0]
    rng = np.random.default_rng(2026)
    N = 200_000
    s1, s2 = truncated_pareto(lp.scale_prior).rvs(size=(2, N), random_state=rng)
    rho = truncated_beta_rvs(lp.rho_prior, N, rng)
    observed = (np.log(factor[:, 1] / factor[:, 0]), np.log(factor[:, 2] / factor[:, 1]))
    reference = (np.log(s2 / s1), np.log((1.0 + (s2 - 1.0) * rho) / s2))
    for k, (x, ref) in enumerate(zip(observed, reference, strict=True)):
        assert stats.ks_2samp(x, ref).pvalue > KS_ALPHA, k


def test_mh_forecast_draws_rho_from_its_prior_when_only_rho_is_uninformed():
    """Origin 2020-05 (j = 2): (s0, s1, s2) come jointly from the chain rows, ρ from its prior."""
    fit = fit_quietly(bvar(mh=SMALL_MH), origin_data("2020-05"), seed=3)
    assert fit.mh.names == ("lambda", "covid_s0", "covid_s1", "covid_s2")
    covid = covid_draws(fit, 4000, seed=8)
    rows = {tuple(r) for r in fit.mh.chains[:, :, 1:].reshape(-1, 3)}
    assert all(tuple(r) in rows for r in covid[:, :3])
    prior = rb.LenzaPrimiceri().rho_prior
    assert np.all((covid[:, 3] >= prior.lower) & (covid[:, 3] <= prior.upper))
    assert stats.kstest(covid[:, 3], truncated_beta_cdf(prior)).pvalue > KS_ALPHA


@pytest.mark.parametrize("lp", [rb.LenzaPrimiceri(), CUSTOM_LP], ids=["default", "custom-priors"])
def test_eb_forecast_keeps_uninformed_parameters_at_the_prior_mode(lp):
    """Origin 2020-03 under EB: (s1, s2, ρ) stay at the prior mode, every parameter draw uses it,
    and with s1 = s2 = 1 the forecast factor is the same at every horizon."""
    mode = np.array([lp.scale_prior.scale] * 3 + [lp.rho_prior.mode])
    fit = fit_quietly(bvar("eb", outliers=lp), origin_data("2020-03"))
    covid = np.asarray(fit.hyper["covid"])
    np.testing.assert_array_equal(covid[1:], mode[1:])
    assert covid[0] > 2.0  # s0 is estimated from the burst month
    np.testing.assert_array_equal(covid_draws(fit, 500, seed=1), np.tile(covid, (500, 1)))
    factor = fit.forecast(4, draws=500, seed=5).stacked("cond_chol")
    for h in range(1, 4):
        np.testing.assert_allclose(factor[:, h], factor[:, 0], rtol=1e-14, atol=0)


def test_eb_forecast_decays_from_the_estimated_s2_with_the_prior_mode_rho():
    """Origin 2020-05 under EB: ρ is uninformed and stays at 0.8, so ``s_{T+h} = 1 + (ŝ2 − 1)
    0.8^h`` and ``cond_chol[T+h]/cond_chol[T+1]`` follows from ŝ2 alone. Data seed 1 has ŝ2 ≈ 2.5
    (in several other seeds the high-leverage 2020-05 row, whose lags are the burst months,
    leaves ŝ2 on its bound 1, which would make the check vacuous)."""
    fit = fit_quietly(bvar("eb"), origin_data("2020-05", seed=1))
    s2, rho = np.asarray(fit.hyper["covid"])[2:]
    assert rho == 0.8 and s2 > 1.5
    factor = fit.forecast(4, draws=300, seed=5).stacked("cond_chol")[:, :, 0, 0]
    expected = 1.0 + (s2 - 1.0) * 0.8 ** np.arange(1, 5)
    np.testing.assert_allclose(
        factor / factor[:, :1], np.tile(expected / expected[0], (300, 1)), rtol=1e-12
    )


# --------------------------------------------------------------------------- prior samplers

PARETO_CASES = [(1.0, 1.0, 1000.0), (1.0, 2.0, 50.0), (0.5, 0.7, 20.0), (2.0, 3.0, 2.5)]
PARETO_CASES += [(1.0, 1.0, 1.001)]


@pytest.mark.parametrize(("scale", "shape", "upper"), PARETO_CASES)
def test_pareto_prior_sample_is_the_exact_truncated_pareto(scale, shape, upper):
    prior = rb.ParetoPrior(scale=scale, shape=shape, upper=upper)
    reference = truncated_pareto(prior)
    N = 40_000
    x = prior.sample(N, np.random.default_rng(17))
    assert x.shape == (N,)
    assert np.all((x >= scale) & (x <= upper))
    assert not np.any(x == upper)  # no atom: truncation is not clipping
    assert stats.kstest(x, reference.cdf).pvalue > KS_ALPHA
    for p in (0.001, 0.01, 0.99, 0.999):  # tails, where KS has little power
        below = float(np.mean(x <= reference.ppf(p)))
        assert abs(below - p) < MC_TOLERANCE * np.sqrt(p * (1 - p) / N), (p, below)
    np.testing.assert_array_equal(x, prior.sample(N, np.random.default_rng(17)))


BETA_CASES = [
    pytest.param(dict(mode=0.8, sd=0.2), id="lp-default"),
    pytest.param(dict(mode=0.6, sd=0.15, lower=0.2, upper=0.95), id="custom-box"),
    pytest.param(dict(mode=0.95, sd=0.03, lower=0.9, upper=1 - 1e-6), id="near-one"),
    pytest.param(dict(mode=0.8, sd=0.2, lower=0.79, upper=0.81), id="narrow-box"),
    pytest.param(dict(mode=0.3, sd=0.1, lower=0.3, upper=0.31), id="mode-on-bound"),
    pytest.param(dict(mode=0.5, sd=1e-4), id="tiny-sd"),
    pytest.param(dict(mode=0.8, sd=0.2, lower=0.0, upper=1.0), id="untruncated"),
]


@pytest.mark.parametrize("kwargs", BETA_CASES)
def test_beta_prior_sample_is_the_exact_truncated_beta(kwargs):
    prior = rb.BetaPrior(**kwargs)
    np.testing.assert_allclose((prior.a, prior.b), beta_ab(prior.mode, prior.sd), rtol=1e-8)
    N = 40_000
    x = prior.sample(N, np.random.default_rng(23))
    assert x.shape == (N,)
    assert np.all((x >= prior.lower) & (x <= prior.upper))
    assert not np.any((x == prior.lower) | (x == prior.upper))  # no atoms from clipping
    cdf = truncated_beta_cdf(prior)
    assert stats.kstest(x, cdf).pvalue > KS_ALPHA
    for p in (0.001, 0.01, 0.99, 0.999):  # tails, where KS has little power
        q = optimize.brentq(lambda v, p=p: cdf(v) - p, prior.lower, prior.upper, xtol=1e-15)
        below = float(np.mean(x <= q))
        assert abs(below - p) < MC_TOLERANCE * np.sqrt(p * (1 - p) / N), (p, below)
    np.testing.assert_array_equal(x, prior.sample(N, np.random.default_rng(23)))


@pytest.mark.parametrize("lp", [rb.LenzaPrimiceri(), CUSTOM_LP], ids=["default", "custom-priors"])
def test_lenza_primiceri_sample_prior_draws_four_independent_columns(lp):
    N = 20_000
    draws = lp.sample_prior(N, np.random.default_rng(29))
    assert draws.shape == (N, 4)
    pareto = truncated_pareto(lp.scale_prior)
    for i in range(3):
        assert stats.kstest(draws[:, i], pareto.cdf).pvalue > KS_ALPHA, COVID_NAMES[i]
    assert stats.kstest(draws[:, 3], truncated_beta_cdf(lp.rho_prior)).pvalue > KS_ALPHA
    corr = stats.spearmanr(draws).statistic
    assert np.all(np.abs(corr[np.triu_indices(4, k=1)]) < 4.5 / np.sqrt(N)), corr
    lower, upper = np.array(lp.bounds()).T
    assert np.all((draws >= lower) & (draws <= upper))


# --------------------------------------------------------------------------- adaptation and seeds


@pytest.fixture(scope="module")
def roll() -> rb.VARData:
    """Window 2020-05 .. 2023-12: free (λ, s2, ρ)."""
    return window_data("2020-05", seed=2)


def test_mh_chains_are_reproducible_for_a_fixed_seed(roll):
    settings = rb.MHSettings(draws=300, burn=300, chains=2)
    model = bvar(mh=settings)
    a, b = fit_quietly(model, roll, seed=4), fit_quietly(model, roll, seed=4)
    np.testing.assert_array_equal(a.mh.chains, b.mh.chains)
    np.testing.assert_array_equal(a.mh.acceptance, b.mh.acceptance)
    assert not np.array_equal(a.mh.chains, fit_quietly(model, roll, seed=5).mh.chains)
    fa, fb = a.forecast(3, draws=200, seed=6), b.forecast(3, draws=200, seed=6)
    for name in ("paths", "cond_chol"):
        np.testing.assert_array_equal(fa.stacked(name), fb.stacked(name))


def test_thinning_is_bookkeeping_only(roll):
    """``thin = 3`` keeps every third state of the very same chains as ``thin = 1`` with three
    times the draws, and counts acceptance over the same iterations."""
    dense = fit_quietly(bvar(mh=rb.MHSettings(draws=900, burn=300, chains=2)), roll, seed=4)
    thinned = fit_quietly(
        bvar(mh=rb.MHSettings(draws=300, burn=300, chains=2, thin=3)), roll, seed=4
    )
    np.testing.assert_array_equal(thinned.mh.chains, dense.mh.chains[:, ::3])
    np.testing.assert_array_equal(thinned.mh.acceptance, dense.mh.acceptance)


def test_number_of_kept_draws_does_not_change_the_chain(roll):
    """Chain 0 of a shorter run is a prefix of chain 0 of a longer one (later chains start from
    a different point of the random stream, so only chain 0 is comparable)."""
    long = fit_quietly(bvar(mh=rb.MHSettings(draws=900, burn=300, chains=2)), roll, seed=4)
    short = fit_quietly(bvar(mh=rb.MHSettings(draws=450, burn=300, chains=2)), roll, seed=4)
    np.testing.assert_array_equal(short.mh.chains[0], long.mh.chains[0, :450])


def replayed_normals(seed: int, settings: rb.MHSettings, m: int) -> np.ndarray:
    """The N(0, I_m) vectors behind every proposal, ``(chains, iterations, m)``.

    The sampler draws from ``default_rng(seed)``: per chain one vector to disperse the start,
    then per iteration one proposal vector followed by one uniform for the accept step.
    """
    rng = np.random.default_rng(seed)
    total = settings.burn + settings.draws * settings.thin
    out = np.empty((settings.chains, total, m))
    for c in range(settings.chains):
        rng.standard_normal(m)
        for it in range(total):
            out[c, it] = rng.standard_normal(m)
            rng.uniform()
    return out


def increment_maps(fit, rho_scale=logit) -> tuple[list[np.ndarray], list[float]]:
    """Per chain, the least-squares ``M`` with ``Δz = M ε`` over the accepted post-burn-in moves
    (``z = (log λ, log s2, rho_scale(ρ))``) and its max relative residual."""
    settings = fit.problem.prior.mh
    m = len(fit.mh.names)
    normals = replayed_normals(fit.mh_seed, settings, m)
    chains = fit.mh.chains
    z = np.concatenate([np.log(chains[..., :2]), rho_scale(chains[..., 2:])], axis=-1)
    maps, residuals = [], []
    for c in range(settings.chains):
        moved = np.any(np.diff(chains[c], axis=0) != 0.0, axis=1)
        assert moved.sum() > 50, moved.sum()
        eps = normals[c, settings.burn + 1 :][moved]
        dz = np.diff(z[c], axis=0)[moved]
        M = np.linalg.lstsq(eps, dz, rcond=None)[0].T
        maps.append(M)
        residuals.append(float(np.abs(eps @ M.T - dz).max() / np.abs(dz).max()))
    return maps, residuals


def test_random_walk_proposal_is_frozen_after_burn_in(roll):
    """Exact check, with the normal draws replayed from the seed: every accepted post-burn-in
    move is ``Δz = step · F ε`` for one fixed matrix per chain, so neither the step nor the
    proposal covariance adapts after burn-in. Without burn-in nothing adapts at all (the same
    map in every chain); with burn-in each chain has re-tuned its own map. The moves are linear
    in ε on the logit scale of ρ and not on its log scale."""
    settings = rb.MHSettings(draws=600, burn=400, chains=2)
    adapted = fit_quietly(bvar(mh=settings), roll, seed=3)
    maps, residuals = increment_maps(adapted)
    assert max(residuals) < 1e-9, residuals
    untuned = fit_quietly(bvar(mh=rb.MHSettings(draws=600, burn=0, chains=2)), roll, seed=3)
    maps0, residuals0 = increment_maps(untuned)
    assert max(residuals0) < 1e-9, residuals0
    np.testing.assert_allclose(maps0[1], maps0[0], rtol=1e-8, atol=1e-12)
    scale = np.abs(maps0[0]).max()
    assert np.abs(maps[1] - maps[0]).max() > 1e-2 * scale  # burn-in tuned each chain
    assert all(np.abs(M - maps0[0]).max() > 1e-2 * scale for M in maps)
    _, log_residuals = increment_maps(adapted, rho_scale=np.log)
    assert min(log_residuals) > 1e-3, log_residuals
