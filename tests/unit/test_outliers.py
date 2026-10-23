"""Lenza & Primiceri (2022) COVID residual scaling (SPEC §4.6, §5.6).

Model: ``y_t = B'x_t + s_t u_t`` with one scalar ``s_t`` common to all variables. ``s_t = 1``
before ``t*``; ``s_{t*+j}`` is the free ``s_j`` for ``j = 0, 1, 2``; afterwards
``s_{t*+j} = 1 + (s_2 − 1) ρ^{j−2}``. ``t*`` defaults to March 2020 (monthly) and 2020Q1
(quarterly). Priors: Pareto(1, 1) on the scales, Beta(mode 0.8, s.d. 0.2) on ρ.

The oracles below are written out from that description (loops over periods, SciPy densities);
they do not call rai_bvar. The engine-level tests check that the scaling is inert when the
sample stops before ``t*``, that EB recovers known scales, and that forecasts scale the shock
at ``T + h`` by ``s_{T+h}`` while keeping the predictive contract (SPEC §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import optimize, stats

import rai_bvar as rb
from rai_bvar.data.outliers import COVID_NAMES, LenzaPrimiceri
from rai_bvar.forecast import ForecastResult, check_structure, simulation_consistency
from rai_bvar.priors.hyperpriors import BetaPrior, ParetoPrior

QD_SMALL = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "qd_small_levels.csv"
T_STAR = {"M": pd.Period("2020-03", freq="M"), "Q": pd.Period("2020Q1", freq="Q")}
FIRST = {"M": "1996-01", "Q": "1985Q1"}
TINY_MH = rb.MHSettings(draws=60, burn=60, chains=2)


# --------------------------------------------------------------------------- oracles and data


def oracle_scales(periods, t_star: pd.Period, params) -> np.ndarray:
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
    return np.array(out, dtype=np.float64)


def beta_by_mode_sd(mode: float, sd: float):
    """Beta(a, b) with the given mode ``(a − 1)/(a + b − 2)`` and standard deviation.

    Independent of the library's parametrisation: ``b`` follows from the mode as a function of
    ``a``, and the variance equation is solved for ``a`` by bracketing (no fsolve, whose
    convergence warnings differ across platforms).
    """

    def b_of(a: float) -> float:
        return (a - 1.0) * (1.0 - mode) / mode + 1.0

    def excess(a: float) -> float:
        b = b_of(a)
        return a * b / ((a + b) ** 2 * (a + b + 1.0)) - sd**2

    a = optimize.brentq(excess, 1.0 + 1e-12, 1e12, xtol=1e-14, rtol=1e-15)
    return stats.beta(a, b_of(a))


def burst_frame(
    truth: tuple[float, float, float, float] | None,
    end: str,
    *,
    freq: str = "M",
    seed: int = 0,
    n: int = 3,
    start: str | None = None,
    normalise: bool = True,
) -> pd.DataFrame:
    """A persistent VAR(1) in levels whose shocks are scaled by the Lenza-Primiceri ``s_t``.

    With ``normalise=True`` every standardised shock from ``t*`` on is rescaled to squared norm
    exactly ``n`` (its expectation), so each burst month carries ``q_t = u_t'Σ⁻¹u_t = n s_t²``
    and the information about ``s_t`` does not depend on a chi-square draw.
    """
    t_star = T_STAR[freq] if start is None else pd.Period(start, freq=freq)
    index = pd.period_range(FIRST[freq], end, freq=freq)
    s = np.ones(len(index)) if truth is None else oracle_scales(index, t_star, truth)
    rng = np.random.default_rng(seed)
    mu = np.array([2.0, -1.0, 5.0])[:n]
    A = np.diag([0.9, 0.8, 0.95])[:n, :n] + 0.05 * (np.ones((n, n)) - np.eye(n))
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    chol = np.diag([0.5, 0.3, 0.8])[:n, :n] @ np.linalg.cholesky(corr)
    y, prev = np.empty((len(index), n)), mu.copy()
    for t, period in enumerate(index):
        z = rng.standard_normal(n)
        if normalise and period >= t_star:
            z *= np.sqrt(n) / np.linalg.norm(z)
        prev = mu + A @ (prev - mu) + s[t] * chol @ z
        y[t] = prev
    return pd.DataFrame(y, index=index, columns=["y1", "y2", "y3"][:n])


def to_data(frame: pd.DataFrame) -> rb.VARData:
    return rb.VARData.from_frame(frame, tcodes=1)


def qd_small() -> rb.VARData:
    df = pd.read_csv(QD_SMALL, index_col="date")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return rb.VARData.from_frame(df, tcodes=1)


def bvar(hierarchy: str = "fixed", outliers=None, lags: int = 3, **prior) -> rb.BVAR:
    minnesota = rb.Minnesota(hierarchy=hierarchy, **prior)
    return rb.BVAR(mean=rb.Linear(prior=minnesota), lags=lags, outliers=outliers)


# --------------------------------------------------------------------------- t* and s_t


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("M", pd.Period("2020-03", freq="M")),
        ("Q", pd.Period("2020Q1", freq="Q")),
        (pd.period_range("2000-01", periods=2, freq="M").freq, pd.Period("2020-03", freq="M")),
        (pd.period_range("2000Q1", periods=2, freq="Q").freq, pd.Period("2020Q1", freq="Q")),
    ],
    ids=["M", "Q", "MonthEnd", "QuarterEnd"],
)
def test_default_start_is_march_2020_or_2020q1(freq, expected):
    start = LenzaPrimiceri().start_period(freq)
    assert isinstance(start, pd.Period)
    assert start == expected
    assert start.freqstr == expected.freqstr


@pytest.mark.parametrize(
    ("start", "freq", "expected"),
    [
        ("2019-11", "M", pd.Period("2019-11", freq="M")),
        ("2008-09", "M", pd.Period("2008-09", freq="M")),
        ("2021Q2", "Q", pd.Period("2021Q2", freq="Q")),
        ("2008Q4", "Q", pd.Period("2008Q4", freq="Q")),
    ],
)
def test_custom_start(start, freq, expected):
    assert LenzaPrimiceri(start=start).start_period(freq) == expected


PARAMS = [
    pytest.param((5.0, 3.0, 2.0, 0.6), id="decaying"),
    pytest.param((8.0, 1.0, 1.5, 0.3), id="s1-at-one"),
    pytest.param((4.0, 2.0, 3.0, 0.0), id="rho-zero"),
    pytest.param((2.0, 5.0, 1.0, 0.9), id="s2-at-one"),
    pytest.param((3.0, 2.0, 6.0, 0.999), id="slow-decay"),
    pytest.param((1.0, 1.0, 1.0, 0.5), id="all-ones"),
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize(
    ("freq", "start"), [("M", None), ("Q", None), ("M", "2019-11"), ("Q", "2008Q4")]
)
def test_scales_formula_for_every_j(freq, start, params):
    """``j < 0`` → 1, ``j = 0, 1, 2`` → ``s0, s1, s2``, ``j ≥ 3`` → ``1 + (s2 − 1)ρ^{j−2}``."""
    lp = LenzaPrimiceri(start=start)
    t_star = T_STAR[freq] if start is None else pd.Period(start, freq=freq)
    periods = pd.period_range(t_star - 7, t_star + 40, freq=freq)
    s = lp.scales(periods, params)
    assert s.shape == (len(periods),)
    assert s.dtype == np.float64
    np.testing.assert_allclose(s, oracle_scales(periods, t_star, params), rtol=1e-14, atol=0)
    j = np.array([p.ordinal - t_star.ordinal for p in periods])
    np.testing.assert_array_equal(s[j < 0], 1.0)
    for i in range(3):  # the free scales enter exactly
        assert s[j == i][0] == params[i]
    s2, rho = params[2], params[3]
    assert s[j == 3][0] == pytest.approx(1.0 + (s2 - 1.0) * rho, rel=1e-15)
    assert np.all(s[j >= 3] >= 1.0)
    if s2 > 1.0 and rho > 0.0:  # the excess over one decays geometrically
        excess = s[j >= 2] - 1.0
        big = excess[:-1] > 1e-6  # 1 + tiny − 1 loses digits; keep excesses above 1e-6
        np.testing.assert_allclose(excess[1:][big] / excess[:-1][big], rho, rtol=1e-9)


def test_scales_on_non_contiguous_and_empty_periods():
    lp = LenzaPrimiceri()
    params = (5.0, 3.0, 2.0, 0.6)
    periods = pd.PeriodIndex(["2019-01", "2020-03", "2020-05", "2021-03"], freq="M")
    np.testing.assert_allclose(lp.scales(periods, params), [1.0, 5.0, 2.0, 1.0 + 0.6**10])
    empty = periods[:0]
    assert lp.scales(empty, params).shape == (0,)
    assert lp.scales(empty, np.tile(params, (3, 1))).shape == (3, 0)


def test_scales_broadcast_over_parameter_draws():
    """``params`` of shape ``(..., 4)`` gives ``(..., T)``: one scale path per draw."""
    lp = LenzaPrimiceri()
    periods = pd.period_range("2019-10", "2021-06", freq="M")
    rng = np.random.default_rng(3)
    params = np.concatenate(
        [1.0 + 9.0 * rng.uniform(size=(2, 5, 3)), rng.uniform(size=(2, 5, 1))], axis=-1
    )
    s = lp.scales(periods, params)
    assert s.shape == (2, 5, len(periods))
    for idx in np.ndindex(2, 5):
        np.testing.assert_allclose(s[idx], oracle_scales(periods, T_STAR["M"], params[idx]))
    s[0, 0, 0] = -1.0  # a fresh, writable array (not a broadcast view)
    assert lp.scales(periods, params)[0, 0, 0] == 1.0
    ints = lp.scales(periods, [5, 3, 2, 0])  # integer input is promoted to float64
    assert ints.dtype == np.float64
    np.testing.assert_allclose(ints, oracle_scales(periods, T_STAR["M"], (5, 3, 2, 0)))


def test_custom_start_shifts_the_whole_path():
    periods = pd.period_range("2018-01", "2023-12", freq="M")
    params = (6.0, 4.0, 3.0, 0.7)
    default = LenzaPrimiceri().scales(periods, params)
    early = LenzaPrimiceri(start="2019-11").scales(periods, params)  # four months earlier
    np.testing.assert_array_equal(early[:-4], default[4:])


# --------------------------------------------------------------------------- active()


@pytest.mark.parametrize(
    ("first", "last", "freq", "start", "active"),
    [
        ("2019-01", "2020-02", "M", None, False),  # ends one period before t*
        ("2019-01", "2020-03", "M", None, True),  # ends exactly at t*
        ("2019-01", "2023-12", "M", None, True),
        ("2021-01", "2021-06", "M", None, True),  # starts after t*
        ("1960Q1", "2019Q4", "Q", None, False),
        ("1960Q1", "2020Q1", "Q", None, True),
        ("2000-01", "2019-10", "M", "2019-11", False),
        ("2000-01", "2019-11", "M", "2019-11", True),
        ("1990Q1", "2008Q3", "Q", "2008Q4", False),
        ("1990Q1", "2008Q4", "Q", "2008Q4", True),
    ],
)
def test_active_iff_the_sample_reaches_t_star(first, last, freq, start, active):
    periods = pd.period_range(first, last, freq=freq)
    assert LenzaPrimiceri(start=start).active(periods) is active


def test_empty_sample_is_inactive():
    assert LenzaPrimiceri().active(pd.PeriodIndex([], freq="M")) is False


# --------------------------------------------------------------------------- fixed=


@pytest.mark.parametrize(
    "fixed",
    [(1.0, 1.0, 1.0, 0.0), (5.0, 3.0, 2.0, 0.6), [5, 3, 2, 0], np.array([9.0, 1.0, 4.0, 0.999])],
    ids=["boundary", "tuple", "list-of-ints", "ndarray"],
)
def test_fixed_accepts_valid_values_as_a_float_tuple(fixed):
    lp = LenzaPrimiceri(fixed=fixed)
    assert isinstance(lp.fixed, tuple)
    assert all(type(v) is float for v in lp.fixed)
    assert lp.fixed == tuple(float(v) for v in fixed)
    assert lp == LenzaPrimiceri(fixed=tuple(float(v) for v in fixed))
    hash(lp)  # frozen and hashable (it is part of the model spec)


@pytest.mark.parametrize(
    "fixed",
    [
        pytest.param((5.0, 3.0, 2.0), id="three-values"),
        pytest.param((5.0, 3.0, 2.0, 0.5, 0.1), id="five-values"),
        pytest.param((0.99, 3.0, 2.0, 0.5), id="s0<1"),
        pytest.param((5.0, 0.5, 2.0, 0.5), id="s1<1"),
        pytest.param((5.0, 3.0, 0.9, 0.5), id="s2<1"),
        pytest.param((5.0, 3.0, 2.0, 1.0), id="rho=1"),
        pytest.param((5.0, 3.0, 2.0, 1.5), id="rho>1"),
        pytest.param((5.0, 3.0, 2.0, -0.1), id="rho<0"),
        pytest.param((5.0, 3.0, 2.0, np.nan), id="rho-nan"),
    ],
)
def test_fixed_rejects_invalid_values(fixed):
    with pytest.raises(ValueError, match="fixed must be"):
        LenzaPrimiceri(fixed=fixed)


@pytest.mark.parametrize(
    "fixed",
    [
        pytest.param((np.nan, 3.0, 2.0, 0.5), id="s0-nan"),
        pytest.param((5.0, np.nan, 2.0, 0.5), id="s1-nan"),
        pytest.param((5.0, 3.0, np.nan, 0.5), id="s2-nan"),
        pytest.param((np.inf, 3.0, 2.0, 0.5), id="s0-inf"),
        pytest.param((5.0, 3.0, np.inf, 0.5), id="s2-inf"),
    ],
)
def test_fixed_rejects_non_finite_scales(fixed):
    """A NaN scale does not satisfy ``s ≥ 1``; an infinite one zeroes the row and sends the
    Jacobian ``−n Σ log s_t`` to −∞. Both must be rejected like any other invalid value."""
    with pytest.raises(ValueError, match="fixed must be"):
        LenzaPrimiceri(fixed=fixed)


# --------------------------------------------------------------------------- log_prior, bounds


def test_beta_oracle_has_the_requested_mode_and_sd():
    dist = beta_by_mode_sd(0.8, 0.2)
    assert dist.std() == pytest.approx(0.2, rel=1e-10)
    mode = optimize.minimize_scalar(
        lambda x: -dist.logpdf(x), bounds=(0.01, 0.99), method="bounded"
    )
    assert mode.x == pytest.approx(0.8, abs=1e-5)


@pytest.mark.parametrize(
    "params",
    [(1.0, 1.0, 1.0, 0.8), (5.0, 3.0, 2.0, 0.6), (40.0, 1.5, 7.0, 0.05), (2.0, 900.0, 1.01, 0.97)],
)
def test_log_prior_is_pareto_on_scales_and_beta_on_rho(params):
    expected = sum(stats.pareto(b=1.0, scale=1.0).logpdf(s) for s in params[:3])
    expected += beta_by_mode_sd(0.8, 0.2).logpdf(params[3])
    assert LenzaPrimiceri().log_prior(np.array(params)) == pytest.approx(expected, rel=1e-12)


def test_log_prior_uses_custom_priors():
    lp = LenzaPrimiceri(
        scale_prior=ParetoPrior(scale=1.0, shape=2.5, upper=100.0),
        rho_prior=BetaPrior(mode=0.6, sd=0.15),
    )
    params = np.array([4.0, 2.0, 1.5, 0.45])
    expected = sum(stats.pareto(b=2.5, scale=1.0).logpdf(s) for s in params[:3])
    expected += beta_by_mode_sd(0.6, 0.15).logpdf(params[3])
    assert lp.log_prior(params) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param((0.5, 3.0, 2.0, 0.6), id="s0<1"),
        pytest.param((5.0, 3.0, 0.99, 0.6), id="s2<1"),
        pytest.param((5.0, 3.0, 2.0, 1.2), id="rho>1"),
        pytest.param((5.0, 3.0, 2.0, -0.1), id="rho<0"),
    ],
)
def test_log_prior_is_minus_infinity_outside_the_support(params):
    """Pareto(1, 1) has support ``s ≥ 1`` and the Beta prior ``0 < ρ < 1``: the prior density
    is zero outside, so the log prior is −∞ (``AnalyticFit.log_posterior`` adds it as is)."""
    assert LenzaPrimiceri().log_prior(np.array(params)) == -np.inf


def test_bounds():
    assert LenzaPrimiceri().bounds() == [(1.0, 1e3)] * 3 + [(1e-6, 1.0 - 1e-6)]
    custom = LenzaPrimiceri(
        scale_prior=ParetoPrior(scale=1.0, shape=2.0, upper=50.0),
        rho_prior=BetaPrior(mode=0.5, sd=0.1, lower=0.01, upper=0.99),
    )
    assert custom.bounds() == [(1.0, 50.0)] * 3 + [(0.01, 0.99)]


# --------------------------------------------------------------------------- the model façade


def test_outliers_string_resolves_to_the_default_scaling():
    model = bvar("eb", outliers="lenza-primiceri")
    assert model.outliers == LenzaPrimiceri()
    assert model.axes["outliers"] == "lenza-primiceri"
    assert "COVID" in model.name
    assert bvar("eb").axes["outliers"] == "none"
    with pytest.raises(ValueError, match="lenza-primiceri"):
        bvar("eb", outliers="exclude-covid")


# --------------------------------------------------------------------------- inactive scaling


def _assert_same_fit(fit, base, mh: bool) -> None:
    assert fit.problem.outliers is None
    assert fit.hyper["covid"] is None
    assert fit.problem.free.names == base.problem.free.names
    assert not set(COVID_NAMES) & set(fit.problem.free.names)
    assert fit.log_ml() == base.log_ml()
    assert fit.log_posterior() == base.log_posterior()
    for key, value in base.hyper.items():
        np.testing.assert_array_equal(np.asarray(fit.hyper[key]), np.asarray(value))
    post, ref = fit.posterior(), base.posterior()
    for name in ("mean", "row_upper", "scale_chol"):
        np.testing.assert_array_equal(getattr(post, name), getattr(ref, name))
    assert post.df == ref.df
    pd.testing.assert_frame_equal(fit.summary(), base.summary())
    if mh:
        np.testing.assert_array_equal(fit.mh.chains, base.mh.chains)
    a, b = fit.forecast(8, draws=200, seed=4), base.forecast(8, draws=200, seed=4)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(a.dataset[name].values, b.dataset[name].values)
    np.testing.assert_array_equal(
        fit.irf(8, draws=50, seed=5).data.values, base.irf(8, draws=50, seed=5).data.values
    )


INACTIVE_OUTLIERS = [
    pytest.param("lenza-primiceri", id="string"),
    pytest.param(LenzaPrimiceri(), id="default"),
    pytest.param(LenzaPrimiceri(fixed=(5.0, 3.0, 2.0, 0.6)), id="fixed-values"),
]


@pytest.mark.parametrize("outliers", INACTIVE_OUTLIERS)
@pytest.mark.parametrize("hierarchy", ["fixed", "eb", "mh"])
def test_inactive_scaling_equals_no_outliers_quarterly(hierarchy, outliers):
    """qd_small ends in 2019Q4: the scaling never binds, so the fit is bit-identical to
    ``outliers=None``. ``hierarchy="fixed"`` needs ``fixed=`` values at construction."""
    data = qd_small()
    assert data.last == pd.Period("2019Q4", freq="Q")
    prior = dict(mean=1.0, soc=1.0, sur=1.0, mh=TINY_MH)
    if hierarchy == "fixed" and not (isinstance(outliers, LenzaPrimiceri) and outliers.fixed):
        with pytest.raises(ValueError, match=r"fixed=\(s0, s1, s2, rho\)"):
            bvar(hierarchy, outliers, lags=4, **prior)
        return
    base = bvar(hierarchy, None, lags=4, **prior).fit(data)
    fit = bvar(hierarchy, outliers, lags=4, **prior).fit(data)
    _assert_same_fit(fit, base, hierarchy == "mh")


@pytest.mark.parametrize(
    ("outliers", "end"),
    [
        pytest.param("lenza-primiceri", "2020-02", id="default-start"),
        pytest.param(LenzaPrimiceri(start="2008-09"), "2008-08", id="custom-start"),
    ],
)
@pytest.mark.parametrize("hierarchy", ["fixed", "eb"])
def test_inactive_scaling_equals_no_outliers_monthly(hierarchy, outliers, end):
    data = to_data(burst_frame(None, end))
    if hierarchy == "fixed":
        with pytest.raises(ValueError, match=r"fixed=\(s0, s1, s2, rho\)"):
            bvar(hierarchy, outliers)
        return
    base = bvar(hierarchy, None).fit(data)
    fit = bvar(hierarchy, outliers).fit(data)
    _assert_same_fit(fit, base, mh=False)


# --------------------------------------------------------------------------- fixed / free values


@pytest.mark.parametrize(
    "outliers",
    ["lenza-primiceri", LenzaPrimiceri(), LenzaPrimiceri(start="2019-06")],
    ids=["string", "default", "custom-start"],
)
def test_fixed_hierarchy_requires_fixed_values_when_active(outliers):
    with pytest.raises(ValueError, match=r"fixed=\(s0, s1, s2, rho\)"):
        bvar("fixed", outliers)


def test_fixed_hierarchy_uses_the_given_values():
    fixed = (5.0, 3.0, 2.0, 0.6)
    fit = bvar("fixed", LenzaPrimiceri(fixed=fixed)).fit(to_data(burst_frame(fixed, "2021-12")))
    assert fit.problem.free.names == ()
    np.testing.assert_array_equal(fit.hyper["covid"], fixed)


@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_fixed_values_are_not_estimated_by_the_hierarchy(hierarchy):
    fixed = (5.0, 3.0, 2.0, 0.6)
    data = to_data(burst_frame((9.0, 5.0, 3.0, 0.8), "2021-12"))
    fit = bvar(hierarchy, LenzaPrimiceri(fixed=fixed), mh=TINY_MH).fit(data)
    assert fit.problem.free.names == ("lambda",)
    np.testing.assert_array_equal(fit.hyper["covid"], fixed)
    assert list(fit.summary().index) == ["lambda"]
    if hierarchy == "mh":
        assert fit.mh.names == ("lambda",)


def test_free_scaling_joins_the_hyperparameters_with_its_bounds():
    custom = LenzaPrimiceri(
        scale_prior=ParetoPrior(scale=1.0, shape=1.0, upper=50.0),
        rho_prior=BetaPrior(mode=0.8, sd=0.2, lower=0.01, upper=0.99),
    )
    fit = bvar("eb", custom).fit(to_data(burst_frame((5.0, 3.0, 2.0, 0.6), "2021-12")))
    free = fit.problem.free
    assert free.names == ("lambda", *COVID_NAMES)
    np.testing.assert_array_equal(free.lower[1:], [1.0, 1.0, 1.0, 0.01])
    np.testing.assert_array_equal(free.upper[1:], [50.0, 50.0, 50.0, 0.99])
    assert list(fit.summary().index) == ["lambda", *COVID_NAMES]


@pytest.mark.parametrize(
    ("freq", "end", "start"),
    [("M", "2021-06", None), ("M", "2021-06", "2019-11"), ("Q", "2021Q4", None)],
)
def test_row_scales_align_with_the_estimation_sample(freq, end, start):
    """Row ``t`` of ``(Y, X)`` is the observation dated ``data.index[p + t]`` and is divided
    by ``s_t`` for that date."""
    fixed = (5.0, 3.0, 2.0, 0.6)
    lags = 3
    data = to_data(burst_frame(fixed, end, freq=freq, start=start))
    fit = bvar("fixed", LenzaPrimiceri(start=start, fixed=fixed), lags=lags).fit(data)
    periods = data.index[lags:]
    assert fit.problem.periods.equals(periods)
    t_star = T_STAR[freq] if start is None else pd.Period(start, freq=freq)
    np.testing.assert_allclose(
        fit.problem.row_scales(fit.hyper), oracle_scales(periods, t_star, fixed), rtol=1e-14
    )


# --------------------------------------------------------------------------- EB recovery

TRUTH = (10.0, 6.0, 4.0, 0.75)  # (s0, s1, s2, ρ) of the simulated burst
RECOVERY_END = "2023-12"
# Tolerances, calibrated by refitting seeds 0..199 of the same design (the slow test below
# re-checks them on every seed): ratio of the EB mode to the one-month target
# s·√(n/(n+2)) for s0 had mean 1.01, s.d. 0.04, range [0.90, 1.12]; for s1 mean 0.99, s.d.
# 0.07, range [0.74, 1.17]. ŝ2/s2 had mean 0.87, s.d. 0.05, range [0.73, 1.00]; ρ̂ − ρ had
# mean 0.03, s.d. 0.015, max |·| 0.07. Each tolerance is ≥ 4 s.d. (about the reference value
# used: 1, 1, 0.87, ρ) and covers every seed.
TOL_S0, TOL_S1, S2_CENTRE, TOL_S2, TOL_RHO = 0.16, 0.3, 0.87, 0.2, 0.1


def _recovery_errors(fit, n: int) -> np.ndarray:
    """Distances of the EB mode from the recovery targets, in units of the tolerances."""
    s0, s1, s2, rho = np.asarray(fit.hyper["covid"])
    target = np.array(TRUTH[:2]) * np.sqrt(n / (n + 2.0))
    return np.array(
        [
            abs(s0 / target[0] - 1.0) / TOL_S0,
            abs(s1 / target[1] - 1.0) / TOL_S1,
            abs(s2 / TRUTH[2] - S2_CENTRE) / TOL_S2,
            abs(rho - TRUTH[3]) / TOL_RHO,
        ]
    )


def test_eb_recovers_known_scales_from_a_monthly_volatility_burst():
    """Monthly n = 3 VAR, 1996-01..2023-12, burst at the default t* = 2020-03.

    s0 and s1 are identified by one month each. With ``q = u'Σ⁻¹u = n s²`` (normalised shocks)
    the log posterior in that month's scale is ``−n log s`` (Jacobian) ``− 2 log s`` (Pareto)
    ``− q/(2s²)``, maximised at ``s·√(n/(n+2))``; that is the target for s0 and s1. s2 is also
    informed by the decay months, so ŝ2/s2 sits between √(n/(n+2)) ≈ 0.77 and 1. Plausible bugs
    are far outside the tolerances: a missing Jacobian sends the scales to the upper bound, a
    Jacobian of ``−Σ log s_t`` (no ``n``) or a missing prior moves ŝ0 by √(5/3) − 1 ≈ 29 %, and
    moving t* one month early or late on this sample gives ŝ0 = 1.0 or 4.0 (measured), 5 and 3
    tolerances away.
    """
    n = 3
    data = to_data(burst_frame(TRUTH, RECOVERY_END, seed=0, n=n))
    fit = bvar("eb", "lenza-primiceri").fit(data)
    assert fit.optimization["success"], fit.optimization
    errors = _recovery_errors(fit, n)
    assert np.all(errors < 1.0), (fit.hyper["covid"], errors)


def test_eb_mode_is_stationary_with_the_jacobian_curvature():
    """At the EB mode the log posterior is flat in every log COVID hyperparameter, and its
    curvature in ``z0 = log s0`` is the one implied by ``−n log s`` (Jacobian) and ``−2 log s``
    (Pareto prior): for one row, ``f(z) = −(n+2)z − (ν̄/2) log(1 + c e^{−2z})`` with
    ``ν̄ = T + d`` gives ``−f''(ẑ) = 2(n + 2)(1 − (n + 2)/ν̄)`` at the mode. Over seeds 0..99
    the measured curvature deviated from this by at most 0.2 %; dropping the ``n`` of the
    Jacobian or the Pareto term would change it by 40 %, a ``−log s`` prior by 20 %."""
    n = 3
    data = to_data(burst_frame(TRUTH, RECOVERY_END, seed=0, n=n))
    fit = bvar("eb", "lenza-primiceri").fit(data)
    z0, step = np.log(np.asarray(fit.hyper["covid"])), 1e-4

    def target(z):
        return fit.log_posterior({"covid": np.exp(z)})

    centre = target(z0)
    curvature = np.empty(4)
    for i in range(4):
        e = np.zeros(4)
        e[i] = step
        up, down = target(z0 + e), target(z0 - e)
        grad = (up - down) / (2 * step)
        curvature[i] = -(up - 2.0 * centre + down) / step**2
        assert curvature[i] > 0
        assert abs(grad) / np.sqrt(curvature[i]) < 0.01, (COVID_NAMES[i], grad, curvature[i])
    nu = fit.posterior().df
    assert nu == len(data) - 3 + n + 2.0
    expected = 2.0 * (n + 2.0) * (1.0 - (n + 2.0) / nu)
    assert curvature[0] == pytest.approx(expected, rel=0.01)


def test_eb_unidentified_scaling_parameters_sit_at_their_prior_modes():
    """Sample ending 2020-04 (j = 1): nothing in the likelihood involves s2 or ρ, so the EB mode
    is the prior mode, s2 = 1 (Pareto(1, 1), at its lower bound) and ρ = 0.8 (Beta mode)."""
    data = to_data(burst_frame(TRUTH, "2020-04"))
    fit = bvar("eb", "lenza-primiceri").fit(data)
    s0, s1, s2, rho = np.asarray(fit.hyper["covid"])
    assert s2 == pytest.approx(1.0, abs=1e-6)
    assert rho == pytest.approx(0.8, abs=1e-3)
    assert s0 > 5.0 and s1 > 3.0  # the two observed burst months are still picked up


@pytest.mark.parametrize("upper", [3.0, 5.0])
def test_eb_respects_a_binding_scale_upper_bound(upper):
    """The unconstrained mode has ŝ0 ≈ 7.8: with the scales capped below it, ŝ0 sits on the cap."""
    capped = LenzaPrimiceri(scale_prior=ParetoPrior(scale=1.0, shape=1.0, upper=upper))
    fit = bvar("eb", capped).fit(to_data(burst_frame(TRUTH, RECOVERY_END)))
    s = np.asarray(fit.hyper["covid"])
    assert np.all(s[:3] <= upper * (1.0 + 1e-12))
    assert s[0] == pytest.approx(upper, rel=1e-8), s


@pytest.mark.parametrize("upper", [30.0, 100.0, 200.0, 500.0])
def test_eb_mode_does_not_depend_on_a_non_binding_upper_bound(upper):
    """The default-bound mode (ŝ0 ≈ 7.8, every scale < 30) is interior to all these boxes, so the
    EB mode must be the same up to optimiser tolerance (measured: 5e-4 relative, 2e-6 in log
    posterior, for the bounds where the optimiser runs normally)."""
    data = to_data(burst_frame(TRUTH, RECOVERY_END))
    ref = bvar("eb", "lenza-primiceri").fit(data)
    assert np.all(np.asarray(ref.hyper["covid"])[:3] < 30.0)
    wide = LenzaPrimiceri(scale_prior=ParetoPrior(scale=1.0, shape=1.0, upper=upper))
    fit = bvar("eb", wide).fit(data)
    np.testing.assert_allclose(fit.hyper["covid"], ref.hyper["covid"], rtol=2e-3)
    assert fit.hyper["lambda"] == pytest.approx(ref.hyper["lambda"], rel=2e-3)
    assert fit.log_posterior() == pytest.approx(ref.log_posterior(), abs=1e-3)


@pytest.mark.slow
def test_eb_recovery_tolerances_hold_across_seeds():
    """The calibration behind the tolerances of the recovery test: every one of 200 simulated
    samples is recovered within them, and the s0 target is unbiased to within 3 %."""
    n = 3
    errors, ratio_s0 = [], []
    for seed in range(200):
        fit = bvar("eb", "lenza-primiceri").fit(
            to_data(burst_frame(TRUTH, RECOVERY_END, seed=seed))
        )
        assert fit.optimization["success"], (seed, fit.optimization)
        errors.append(_recovery_errors(fit, n))
        ratio_s0.append(fit.hyper["covid"][0] / (TRUTH[0] * np.sqrt(n / (n + 2.0))))
    errors = np.array(errors)
    assert errors.max() < 1.0, (errors.argmax(axis=0), errors.max(axis=0))
    assert abs(np.mean(ratio_s0) - 1.0) < 0.03


# --------------------------------------------------------------------------- forecasts

FIXED = (5.0, 3.0, 2.0, 0.6)


def _forecast_regressors(values: np.ndarray, paths: np.ndarray, lags: int) -> np.ndarray:
    """``x_{s,T+h} = (1, y_{T+h−1}, …, y_{T+h−p})`` from the data and the simulated path."""
    S, H, n = paths.shape
    full = np.concatenate([np.broadcast_to(values[-lags:], (S, lags, n)), paths], axis=1)
    cols = [np.ones((S, H, 1))]
    cols += [full[:, lags - lag : lags - lag + H] for lag in range(1, lags + 1)]
    return np.concatenate(cols, axis=-1)


def _future_scales(data: rb.VARData, horizon: int, lp: LenzaPrimiceri, params) -> np.ndarray:
    freq = data.index.freqstr[0]
    t_star = T_STAR[freq] if lp.start is None else pd.Period(lp.start, freq=freq)
    future = pd.period_range(data.last + 1, periods=horizon, freq=data.index.freq)
    return oracle_scales(future, t_star, params)


def _assert_sigma_draws(factors: np.ndarray, posterior) -> None:
    """``L_s`` are factors of iid draws from ``IW(S̄, ν̄)``: the Monte Carlo mean of ``L L'`` is
    within 5 standard errors of ``S̄/(ν̄ − n − 1)``, with the exact inverse-Wishart variance
    ``Var Σ_ij = ((ν−n+1) S̄_ij² + (ν−n−1) S̄_ii S̄_jj) / ((ν−n)(ν−n−1)²(ν−n−3))``."""
    S, n = factors.shape[0], factors.shape[-1]
    psi, nu = posterior.scale, posterior.df
    var = ((nu - n + 1) * psi**2 + (nu - n - 1) * np.outer(np.diag(psi), np.diag(psi))) / (
        (nu - n) * (nu - n - 1) ** 2 * (nu - n - 3)
    )
    mc_mean = np.einsum("sij,skj->ik", factors, factors) / S
    assert np.all(np.abs(mc_mean - psi / (nu - n - 1)) < 5.0 * np.sqrt(var / S))


@pytest.mark.parametrize(
    ("freq", "end", "start"),
    [
        pytest.param("M", "2020-03", None, id="M-ends-at-t*"),
        pytest.param("Q", "2020Q1", None, id="Q-ends-at-t*"),
        pytest.param("M", "2020-03", "2020-02", id="M-custom-start"),
        pytest.param("M", "2020-05", None, id="M-ends-at-t*+2"),
        pytest.param("Q", "2021Q2", None, id="Q-decay"),
    ],
)
def test_forecast_cond_chol_is_scaled_by_s_at_t_plus_h(freq, end, start):
    """``cond_chol[s, h] = s_{T+h} L_s``: one Σ draw per path, its factor multiplied by the
    scale of the forecast date (s1, s2 and the decay all appear across the cases)."""
    H, S = 8, 2000
    lp = LenzaPrimiceri(start=start, fixed=FIXED)
    data = to_data(burst_frame(FIXED, end, freq=freq, start=start))
    fit = bvar("fixed", lp, lags=2).fit(data)
    fr = fit.forecast(H, draws=S, seed=11)
    s = _future_scales(data, H, lp, FIXED)
    assert np.any(s > 1.0)
    chol = fr.stacked("cond_chol")
    base = chol / s[None, :, None, None]
    # the same Σ draw at every horizon: the per-horizon factor is exactly s_{T+h}
    np.testing.assert_allclose(base, np.broadcast_to(base[:, :1], base.shape), rtol=1e-13)
    # and that draw is a posterior draw of Σ (Monte Carlo check of E[LL'] = S̄/(ν̄ − n − 1))
    _assert_sigma_draws(base[:, 0], fit.posterior())
    check_structure(fr)
    assert simulation_consistency(fr) > 1e-3


def test_eb_forecast_uses_the_estimated_scales():
    lp = LenzaPrimiceri()
    data = to_data(burst_frame(TRUTH, "2020-06"))
    fit = bvar("eb", lp).fit(data)
    H = 6
    fr = fit.forecast(H, draws=300, seed=2)
    s = _future_scales(data, H, lp, fit.hyper["covid"])
    chol = fr.stacked("cond_chol")
    base = chol / s[None, :, None, None]
    np.testing.assert_allclose(base, np.broadcast_to(base[:, :1], base.shape), rtol=1e-13)
    assert simulation_consistency(fr) > 1e-3


def test_forecast_contract_invariants_under_covid_scaling():
    """SPEC §7 for a COVID-scaled model: structure, simulation consistency (with power against
    the unscaled factor), joint paths, shapes, dtypes, coordinates and reproducibility."""
    lags, H, S = 2, 12, 1000
    lp = LenzaPrimiceri(fixed=FIXED)
    data = to_data(burst_frame(FIXED, "2020-03"))
    model = bvar("fixed", lp, lags=lags)
    fit = model.fit(data)
    fr = fit.forecast(H, draws=S, seed=7)
    ds = fr.dataset

    check_structure(fr)
    assert simulation_consistency(fr) > 1e-3
    # power: stored factors without s_{T+h} do not describe the simulated paths
    s = _future_scales(data, H, lp, FIXED)
    np.testing.assert_allclose(s[:3], [3.0, 2.0, 1.6])
    unscaled = ds.copy(deep=True)
    s_h = xr.DataArray(s, dims=["horizon"], coords={"horizon": ds["horizon"].values})
    unscaled["cond_chol"] = ds["cond_chol"] / s_h
    assert simulation_consistency(ForecastResult(unscaled)) < 1e-6

    # joint paths: cond_mean[s, h] = B_s' x_{s,h}, x built from the simulated path to T+h−1
    paths, mean = fr.stacked(), fr.stacked("cond_mean")
    X = _forecast_regressors(data.values, paths, lags)
    assert X.shape[2] < H
    coefs = np.linalg.pinv(X) @ mean
    scale = np.abs(mean).max(axis=(0, 1))
    np.testing.assert_allclose(np.abs(X @ coefs - mean).max(axis=(0, 1)) / scale, 0.0, atol=1e-10)

    assert fr.stacked().shape == (S, H, 3)
    assert fr.stacked("cond_chol").shape == (S, H, 3, 3)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        assert ds[name].dtype == np.float64
    assert fr.origin == pd.Period("2020-03", freq="M")
    assert ds["date"].values[0] == "2020-04"
    assert ds.attrs["engine"] == "analytic"
    assert "COVID" in ds.attrs["model"]
    np.testing.assert_allclose(
        fr.stacked("paths_level"), data.inverse_paths(fr.stacked()), rtol=1e-14
    )
    again = fit.forecast(H, draws=S, seed=7)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(again.dataset[name].values, ds[name].values)


def test_irf_uses_pre_covid_shocks():
    """Impulse responses are to one-s.d. shocks of Σ itself, not of ``s_t² Σ``: the impact
    matrix is the lower-triangular factor of a Σ draw, with the Σ posterior mean on average."""
    data = to_data(burst_frame(FIXED, "2020-03"))
    fit = bvar("fixed", LenzaPrimiceri(fixed=FIXED), lags=2).fit(data)
    S = 2000
    impact = fit.irf(4, draws=S, seed=3).data.values[0, :, 0]  # (draw, response, shock)
    np.testing.assert_array_equal(np.triu(impact, k=1), 0.0)
    _assert_sigma_draws(impact, fit.posterior())


def test_mh_forecast_scales_come_from_the_mh_draws():
    """With ``hierarchy="mh"`` every forecast draw carries the scales of one MH draw: the ratios
    ``cond_chol[s, h]/cond_chol[s, 1] = s_{T+h}/s_{T+1}`` must equal those implied by some row
    ``(s0, s1, s2, ρ)`` of the MH chains, and more than one row must be used."""
    lp = LenzaPrimiceri()
    data = to_data(burst_frame(TRUTH, "2020-06"))  # forecasts start at j = 4, in the decay
    fit = bvar("mh", lp, mh=TINY_MH).fit(data, seed=5)
    H, S = 4, 300
    fr = fit.forecast(H, draws=S, seed=9)
    check_structure(fr)
    assert simulation_consistency(fr) > 1e-3
    chol = fr.stacked("cond_chol")
    observed = chol[:, :, 0, 0] / chol[:, :1, 0, 0]  # (S, H)
    pooled = fit.mh.chains.reshape(-1, len(fit.mh.names))
    covid = pooled[:, [fit.mh.names.index(name) for name in COVID_NAMES]]
    implied = np.stack([_future_scales(data, H, lp, row) for row in covid])
    implied = implied / implied[:, :1]
    match = np.all(np.isclose(observed[:, None], implied[None], rtol=1e-10, atol=0), axis=-1)
    assert np.all(match.any(axis=1)), np.flatnonzero(~match.any(axis=1))
    assert len(np.unique(observed.round(12), axis=0)) > 1


@pytest.mark.slow
@pytest.mark.parametrize(
    "end",
    [pytest.param("2023-12", id="burst-identified"), pytest.param("2020-04", id="ends-at-t*+1")],
)
def test_mh_chains_mix_with_default_settings(end):
    """A VAR estimated in April 2020 (the Lenza-Primiceri use case) has no data on s2 and ρ, so
    their posterior is the prior: log s2 ~ Exp(1) truncated at log 1000 and ρ ~ Beta. That is an
    easy target for random-walk MH on log hyperparameters; with the default 4 × 2500 draws every
    chain must at least have mixed, R-hat < 1.1 (the identified-burst sample gives ≈ 1.01)."""
    fit = bvar("mh", "lenza-primiceri").fit(to_data(burst_frame(TRUTH, end)), seed=1)
    assert all(np.isfinite(v) for v in fit.mh.rhat.values())
    assert max(fit.mh.rhat.values()) < 1.1, fit.mh.rhat
    assert np.all(fit.mh.acceptance > 0.1), fit.mh.acceptance
