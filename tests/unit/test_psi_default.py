"""The default IW scale ``psi="ar"`` of the conjugate Minnesota prior (SPEC §4.6).

``ψ_j`` is the residual variance ``SSR / (rows − p − 1)`` of a univariate AR(p) with constant,
by least squares. The rules checked here:

* Laziness: an explicit ``psi=(...)`` never runs the AR regressions, so constant series, exactly
  fitted series and samples too short for an AR(p) fit as long as ψ is given.
* With the Lenza-Primiceri scaling active, ψ comes from the periods before ``t*`` only (where
  ``s_t = 1``): it equals the AR variances of those rows and ignores everything from ``t*`` on.
* Fallback: with fewer than ``2p + 2`` pre-``t*`` periods (rolling windows near or after
  ``t*``), the AR regression runs on the whole window without the rows whose dependent period
  or any of its ``p`` lags falls in ``t* … t*+2``, with a warning.
* Failures name the variable: too few observations, a series constant or collinear with its
  lags, and a series fitted exactly by the AR(p) all raise a ``ValueError`` (never a bare
  ``LinAlgError``, which is itself a ``ValueError`` subclass).
* A note in ``fit.warnings`` when the AR regressions rest on fewer than 10 degrees of freedom.
* With ψ free, its default bounds are the AR variance /100 and ×100; a lone user bound that
  leaves the interval empty raises a ``ValueError`` naming the variables.
* Rolling backtests whose windows cross ``t*`` complete instead of aborting at the first window
  with too few pre-``t*`` periods.

The oracles below build the AR regressions themselves (explicit loops, ``np.linalg.lstsq``) and
do not call rai_bvar.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import rai_bvar as rb
from rai_bvar.engines import analytic
from rai_bvar.engines.analytic import ConjugateProblem

VARS = ("y1", "y2", "y3")
T_STAR = {"M": pd.Period("2020-03", freq="M"), "Q": pd.Period("2020Q1", freq="Q")}
FIRST = {"M": "2012-01", "Q": "2000Q1"}
LAST = {"M": "2025-12", "Q": "2025Q4"}
MU = np.array([2.0, -1.0, 5.0])
COEF = np.diag([0.6, 0.5, 0.7]) + 0.05 * (np.ones((3, 3)) - np.eye(3))
SD = np.array([0.5, 0.3, 0.8])  # innovation s.d. before t* (diagonal Σ)
BURST = (25.0, 20.0, 15.0, 0.3)  # (s0, s1, s2, ρ): a COVID-sized volatility burst
GIVEN_PSI = (0.3, 0.1, 0.5)

FALLBACK = "residual variances use the window without the rows touching"
SMALL_DF = "rest on fewer than 10 degrees of freedom"


# --------------------------------------------------------------------------- oracles and data


def oracle_scale(period: pd.Period, t_star: pd.Period, truth) -> float:
    s0, s1, s2, rho = truth
    j = (period - t_star).n
    if j < 0:
        return 1.0
    if j <= 2:
        return (s0, s1, s2)[j]
    return 1.0 + (s2 - 1.0) * rho ** (j - 2)


def burst_frame(
    first: str | None = None,
    last: str | None = None,
    *,
    freq: str = "M",
    truth=BURST,
    seed: int = 0,
    t_star: str | None = None,
) -> pd.DataFrame:
    """A stationary VAR(1) whose shocks are multiplied by the Lenza-Primiceri ``s_t``.

    The same seed gives the same draws whatever ``truth`` is, so frames that differ only in
    the burst agree exactly before ``t*``.
    """
    index = pd.period_range(first or FIRST[freq], last or LAST[freq], freq=freq)
    star = T_STAR[freq] if t_star is None else pd.Period(t_star, freq=freq)
    rng = np.random.default_rng(seed)
    y, prev = np.empty((len(index), 3)), MU.copy()
    for t, period in enumerate(index):
        shock = oracle_scale(period, star, truth) * SD * rng.standard_normal(3)
        prev = MU + COEF @ (prev - MU) + shock
        y[t] = prev
    return pd.DataFrame(y, index=index, columns=list(VARS))


def noise_frame(T: int, *, seed: int = 1, first: str = "2005-01", freq: str = "M") -> pd.DataFrame:
    """Pre-2020 data (no COVID warning) from the same VAR without any burst."""
    index = pd.period_range(first, periods=T, freq=freq)
    assert index[-1] < T_STAR[freq]
    frame = burst_frame(first, str(index[-1]), freq=freq, truth=(1.0, 1.0, 1.0, 0.0), seed=seed)
    return frame


def oracle_ar_variance(x: np.ndarray, index: pd.PeriodIndex, p: int, drop=frozenset()) -> float:
    """``SSR / (rows − p − 1)`` of ``x_t = c + Σ_l a_l x_{t−l} + e_t`` by ``np.linalg.lstsq``.

    A regression row ``t`` is dropped when its dependent period or one of its ``p`` lags is in
    ``drop`` (the lags themselves always come from the contiguous series).
    """
    rows = [t for t in range(p, len(x)) if not any(index[t - lag] in drop for lag in range(p + 1))]
    X = np.array([[1.0, *(x[t - lag] for lag in range(1, p + 1))] for t in rows])
    y = np.asarray(x, dtype=np.float64)[rows]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return float(resid @ resid) / (len(rows) - p - 1)


def oracle_psi(frame: pd.DataFrame, p: int, drop=frozenset()) -> np.ndarray:
    return np.array(
        [oracle_ar_variance(frame[c].to_numpy(), frame.index, p, drop) for c in frame.columns]
    )


def burst_periods(t_star: pd.Period) -> frozenset:
    return frozenset({t_star, t_star + 1, t_star + 2})


def to_data(frame: pd.DataFrame) -> rb.VARData:
    return rb.VARData.from_frame(frame, tcodes=1)


def bvar(hierarchy: str = "eb", *, lags: int = 2, outliers=None, **prior) -> rb.BVAR:
    minnesota = rb.Minnesota(hierarchy=hierarchy, **prior)
    return rb.BVAR(mean=rb.Linear(prior=minnesota), lags=lags, outliers=outliers)


def lp_bvar(hierarchy: str, *, lags: int = 2, start: str | None = None, **prior) -> rb.BVAR:
    """Lenza-Primiceri scaling: estimated under ``eb``, fixed at the true burst under ``fixed``."""
    fixed = BURST if hierarchy == "fixed" else None
    return bvar(hierarchy, lags=lags, outliers=rb.LenzaPrimiceri(start=start, fixed=fixed), **prior)


def fit_recording(model: rb.BVAR, data: rb.VARData):
    """Fit, checking that every UserWarning emitted is recorded in ``fit.warnings`` (in order)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        fit = model.fit(data)
    user = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    for w in caught:  # anything else passes through
        if not issubclass(w.category, UserWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    assert user == fit.warnings
    return fit


def psi_notes(fit) -> list[str]:
    return [m for m in fit.warnings if m.startswith("psi='ar'")]


def assert_value_error(excinfo) -> None:
    """``np.linalg.LinAlgError`` subclasses ValueError: insist on a plain, explained ValueError."""
    assert type(excinfo.value) is ValueError, type(excinfo.value)
    assert "pass psi=(...) explicitly" in str(excinfo.value)


# --------------------------------------------------------------------------- laziness


@pytest.fixture
def no_ar_fits(monkeypatch):
    """Make any AR residual-variance computation fail loudly."""

    def forbidden(*args, **kwargs):
        raise AssertionError("AR residual variances computed although psi was given")

    monkeypatch.setattr(analytic, "_ar_residual_variances", forbidden)


def _with_column(frame: pd.DataFrame, column: str, values) -> pd.DataFrame:
    out = frame.copy()
    out[column] = values
    return out


LAZY_CASES = {
    # name: (frame factory, lags, error raised by psi="ar" on the same data)
    "regular": (lambda: noise_frame(60), 2, None),
    "constant series": (
        lambda: _with_column(noise_frame(60), "y2", 3.0),
        2,
        "'y2' is constant (or collinear with its lags)",
    ),
    "exactly fitted series": (
        lambda: _with_column(noise_frame(60), "y2", np.arange(60.0)),
        1,
        "'y2' is fitted (numerically) exactly by an AR(1)",
    ),
    "T = p + 1": (lambda: noise_frame(5), 4, "needs at least 10 usable observations of 'y1'"),
    "T = 2p + 1": (lambda: noise_frame(9), 4, "needs at least 10 usable observations of 'y1'"),
    "p = 1, T = 2": (lambda: noise_frame(2), 1, "needs at least 4 usable observations of 'y1'"),
}


@pytest.mark.parametrize("hierarchy", ["fixed", "eb"])
@pytest.mark.parametrize("case", list(LAZY_CASES))
def test_explicit_psi_never_runs_the_ar_regressions(no_ar_fits, case, hierarchy):
    make, lags, _ = LAZY_CASES[case]
    data = to_data(make())
    fit = fit_recording(bvar(hierarchy, lags=lags, psi=GIVEN_PSI), data)
    np.testing.assert_array_equal(fit.psi, GIVEN_PSI)
    assert np.isfinite(fit.log_ml())
    assert np.all(np.isfinite(fit.forecast(3, draws=50, seed=0).paths.values))
    assert psi_notes(fit) == []


@pytest.mark.parametrize("case", [c for c, v in LAZY_CASES.items() if v[2] is not None])
def test_psi_ar_fails_with_a_named_error_where_explicit_psi_succeeds(case):
    make, lags, message = LAZY_CASES[case]
    with pytest.raises(ValueError, match=re.escape(message)) as excinfo:
        bvar("fixed", lags=lags).fit(to_data(make()))
    assert_value_error(excinfo)


def test_the_laziness_guard_catches_an_ar_fit(no_ar_fits):
    """Control: with ``psi="ar"`` the patched helper is reached, so the guard is effective."""
    with pytest.raises(AssertionError, match="AR residual variances computed"):
        bvar("fixed").fit(to_data(noise_frame(60)))


@pytest.mark.parametrize("hierarchy", ["fixed", "eb"])
@pytest.mark.parametrize(
    ("first", "last"),
    [
        pytest.param("2020-05", "2021-06", id="window-after-t*"),
        pytest.param("2020-01", "2020-08", id="too-few-rows-even-after-the-mask"),
        pytest.param("2019-12", "2020-03", id="sample-ends-at-t*"),
    ],
)
def test_explicit_psi_never_runs_the_ar_regressions_under_covid_scaling(
    no_ar_fits, hierarchy, first, last
):
    data = to_data(burst_frame(first, last))
    fit = fit_recording(lp_bvar(hierarchy, psi=GIVEN_PSI), data)
    assert fit.problem.outliers is not None  # the scaling is active
    np.testing.assert_array_equal(fit.psi, GIVEN_PSI)
    assert np.isfinite(fit.log_ml())
    assert np.all(np.isfinite(fit.forecast(3, draws=50, seed=0).paths.values))
    assert psi_notes(fit) == []


def test_explicit_free_psi_is_the_mode_and_sets_the_bounds(no_ar_fits):
    """ψ free with an explicit starting value: the bounds come from it, not from AR fits."""
    prior = rb.Minnesota(hierarchy="eb", psi=GIVEN_PSI, hyper=("lambda", "psi"))
    problem = ConjugateProblem(to_data(noise_frame(6)), prior, 4, "c", None)
    free = problem.free
    idx = [free.names.index(f"psi_{v}") for v in VARS]
    np.testing.assert_allclose(free.lower[idx], np.array(GIVEN_PSI) / 100.0, rtol=1e-15)
    np.testing.assert_allclose(free.upper[idx], np.array(GIVEN_PSI) * 100.0, rtol=1e-15)
    np.testing.assert_array_equal(problem.default["psi"], GIVEN_PSI)


# --------------------------------------------------------------------------- full sample


def test_default_psi_without_scaling_is_the_full_sample_ar_variance():
    frame = noise_frame(80)
    fit = fit_recording(bvar("fixed", lags=3), to_data(frame))
    np.testing.assert_allclose(fit.psi, oracle_psi(frame, 3), rtol=1e-10)
    assert psi_notes(fit) == []


@pytest.mark.parametrize(("lags", "T"), [(4, 9), (4, 8), (4, 5), (1, 3), (1, 2), (2, 5), (3, 7)])
def test_psi_ar_on_too_short_a_sample_raises_the_documented_error(lags, T):
    """``T = 2p + 1`` used to divide by zero; anything below ``2p + 2`` must name the minimum."""
    with pytest.raises(ValueError, match=re.escape(f"needs at least {2 * lags + 2} usable")) as e:
        bvar("fixed", lags=lags).fit(to_data(noise_frame(T)))
    assert_value_error(e)
    assert f"'y1' over the sample for AR({lags})" in str(e.value)


@pytest.mark.parametrize("lags", [1, 2, 4])
def test_psi_ar_at_exactly_2p_plus_2_observations_works_with_a_note(lags):
    frame = noise_frame(2 * lags + 2)
    with pytest.warns(UserWarning, match=SMALL_DF):
        fit = bvar("fixed", lags=lags).fit(to_data(frame))
    np.testing.assert_allclose(fit.psi, oracle_psi(frame, lags), rtol=1e-8)
    assert np.all(np.isfinite(fit.psi) & (fit.psi > 0))


# --------------------------------------------------------------------------- pre-t* estimation

PRE_CASES = [
    pytest.param("M", None, 2, id="monthly"),
    pytest.param("Q", None, 2, id="quarterly"),
    pytest.param("M", "2019-11", 2, id="monthly-custom-start"),
    pytest.param("Q", "2019Q3", 1, id="quarterly-custom-start"),
    pytest.param("M", None, 4, id="monthly-p4"),
]


@pytest.mark.parametrize("hierarchy", ["eb", "fixed"])
@pytest.mark.parametrize(("freq", "start", "lags"), PRE_CASES)
def test_covid_default_psi_is_the_pre_t_star_ar_variance(hierarchy, freq, start, lags):
    frame = burst_frame(freq=freq, t_star=start)
    t_star = T_STAR[freq] if start is None else pd.Period(start, freq=freq)
    fit = fit_recording(lp_bvar(hierarchy, lags=lags, start=start), to_data(frame))
    assert fit.problem.outliers is not None
    pre = oracle_psi(frame[frame.index < t_star], lags)
    np.testing.assert_allclose(fit.psi, pre, rtol=1e-10)
    assert psi_notes(fit) == []
    # the test discriminates: the burst inflates the full-sample AR variances
    assert np.all(oracle_psi(frame, lags) > 2.0 * pre)


def test_covid_default_psi_ignores_everything_from_t_star_on():
    """Same pre-``t*`` data, different bursts and sample ends: the same ψ."""
    t_star = T_STAR["M"]
    psis = []
    for truth, last in [(BURST, "2023-12"), ((3.0, 2.0, 1.5, 0.9), "2023-12"), (BURST, "2020-03")]:
        frame = burst_frame(last=last, truth=truth)
        psis.append(fit_recording(lp_bvar("eb"), to_data(frame)).psi)
    big, small = burst_frame(truth=BURST), burst_frame(truth=(3.0, 2.0, 1.5, 0.9))
    assert not np.allclose(big.loc[t_star:].to_numpy(), small.loc[t_star:].to_numpy())
    np.testing.assert_array_equal(psis[0], psis[1])
    np.testing.assert_array_equal(psis[0], psis[2])
    np.testing.assert_allclose(psis[0], oracle_psi(big[big.index < t_star], 2), rtol=1e-10)


def test_covid_default_psi_sets_the_free_psi_bounds():
    """With ψ free under the scaling, its mode and default bounds are the pre-``t*`` values."""
    frame = burst_frame()
    prior = rb.Minnesota(hierarchy="eb", hyper=("lambda", "psi"))
    problem = ConjugateProblem(to_data(frame), prior, 2, "c", rb.LenzaPrimiceri())
    pre = oracle_psi(frame[frame.index < T_STAR["M"]], 2)
    free = problem.free
    idx = [free.names.index(f"psi_{v}") for v in VARS]
    np.testing.assert_allclose(problem.default["psi"], pre, rtol=1e-10)
    np.testing.assert_allclose(free.lower[idx], pre / 100.0, rtol=1e-10)
    np.testing.assert_allclose(free.upper[idx], pre * 100.0, rtol=1e-10)


def test_inactive_scaling_uses_the_full_sample():
    """A sample ending before ``t*`` never reaches the scaling: ψ is the full-sample value."""
    frame = burst_frame(last="2020-02")
    fit = fit_recording(lp_bvar("eb"), to_data(frame))
    assert fit.problem.outliers is None
    np.testing.assert_allclose(fit.psi, oracle_psi(frame, 2), rtol=1e-10)
    plain = fit_recording(bvar("eb"), to_data(frame))
    np.testing.assert_array_equal(fit.psi, plain.psi)


@pytest.mark.parametrize("lags", [1, 2, 4])
def test_exactly_2p_plus_2_pre_t_star_periods_use_the_pre_t_star_rule(lags):
    t_star = T_STAR["M"]
    frame = burst_frame(str(t_star - (2 * lags + 2)), "2023-12")
    fit = fit_recording(lp_bvar("eb", lags=lags), to_data(frame))
    np.testing.assert_allclose(fit.psi, oracle_psi(frame[frame.index < t_star], lags), rtol=1e-8)
    notes = psi_notes(fit)
    assert len(notes) == 1  # fit_recording checked that it was emitted as a UserWarning
    assert SMALL_DF in notes[0] and f"over the periods before t* = {t_star}" in notes[0]
    assert FALLBACK not in notes[0]


# --------------------------------------------------------------------------- fallback

FALLBACK_CASES = [
    # (freq, first, last, lags, custom start)
    pytest.param("M", "2019-10", "2023-12", 2, None, id="M-2p+1-pre-periods"),
    pytest.param("M", "2020-01", "2023-12", 2, None, id="M-two-pre-periods"),
    pytest.param("M", "2020-03", "2023-12", 2, None, id="M-starts-at-t*"),
    pytest.param("M", "2020-05", "2023-12", 2, None, id="M-starts-after-t*"),
    pytest.param("M", "2021-01", "2023-12", 2, None, id="M-burst-outside-window"),
    pytest.param("M", "2019-12", "2023-12", 4, None, id="M-p4"),
    pytest.param("Q", "2019Q3", "2025Q4", 1, None, id="Q-p1"),
    pytest.param("Q", "2020Q2", "2025Q4", 1, None, id="Q-after-t*"),
    pytest.param("M", "2019-09", "2023-12", 2, "2019-11", id="M-custom-start"),
]


def fallback_message(lags: int, t_star: pd.Period) -> str:
    return (
        f"psi='ar': fewer than {2 * lags + 2} observations before t* = {t_star}; AR({lags}) "
        f"residual variances use the window without the rows touching {t_star}..{t_star + 2}"
    )


@pytest.mark.parametrize("hierarchy", ["eb", "fixed"])
@pytest.mark.parametrize(("freq", "first", "last", "lags", "start"), FALLBACK_CASES)
def test_fallback_psi_is_the_regression_without_the_burst_rows(
    hierarchy, freq, first, last, lags, start
):
    t_star = T_STAR[freq] if start is None else pd.Period(start, freq=freq)
    frame = burst_frame(first, last, freq=freq, t_star=start)
    assert (frame.index < t_star).sum() < 2 * lags + 2
    message = fallback_message(lags, t_star)
    fit = fit_recording(lp_bvar(hierarchy, lags=lags, start=start), to_data(frame))
    assert message in fit.warnings  # fit_recording checked that it was emitted as a UserWarning
    expected = oracle_psi(frame, lags, drop=burst_periods(t_star))
    np.testing.assert_allclose(fit.psi, expected, rtol=1e-10)


def test_fallback_psi_does_not_depend_on_the_burst_values():
    """The dropped rows are exactly those touching ``t* … t*+2``: changing those values leaves
    ψ unchanged, while changing ``t*+3`` (a lag of a kept row when ``p = 2``) changes it."""
    t_star = T_STAR["M"]
    frame = burst_frame("2019-10", "2023-12")
    base = fit_recording(lp_bvar("fixed"), to_data(frame)).psi
    shocked = frame.copy()
    shocked.loc[t_star : t_star + 2] *= 1000.0
    np.testing.assert_array_equal(fit_recording(lp_bvar("fixed"), to_data(shocked)).psi, base)
    later = frame.copy()
    later.loc[t_star + 3] += 50.0
    assert np.all(fit_recording(lp_bvar("fixed"), to_data(later)).psi != base)


def test_fallback_psi_is_not_inflated_by_the_burst():
    frame = burst_frame("2019-12", "2024-12", truth=(60.0, 40.0, 30.0, 0.2), seed=4)
    fit = fit_recording(lp_bvar("eb"), to_data(frame))
    ratio = fit.psi / SD**2
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio
    assert np.all(oracle_psi(frame, 2) / SD**2 > 20.0)  # what the unmasked window would give


def test_fallback_with_too_few_rows_left_raises_naming_the_window():
    t_star = T_STAR["M"]
    frame = burst_frame("2020-01", "2020-08")  # one regression row survives the mask
    with pytest.raises(ValueError, match=re.escape("needs at least 6 usable observations")) as e:
        lp_bvar("eb").fit(to_data(frame))
    assert_value_error(e)
    assert f"of 'y1' over the window without the rows touching {t_star}..{t_star + 2}" in str(
        e.value
    )


def test_fallback_with_few_rows_left_adds_the_small_df_note():
    t_star = T_STAR["M"]
    frame = burst_frame("2020-01", "2020-12")  # rows for 2020-08 … 2020-12 survive: df = 2
    fit = fit_recording(lp_bvar("eb"), to_data(frame))
    np.testing.assert_allclose(fit.psi, oracle_psi(frame, 2, burst_periods(t_star)), rtol=1e-10)
    notes = psi_notes(fit)
    assert notes[0] == fallback_message(2, t_star)
    assert any(
        SMALL_DF in m and f"over the window without the rows touching {t_star}..{t_star + 2}" in m
        for m in notes
    )


# --------------------------------------------------------------------------- named failures


def _series(kind: str, T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    return {
        "constant": np.full(T, 3.0),
        "period-2": np.where(t % 2 == 0, 1.0, 4.0),
        "trend": t,
        "affine-trend": 10.0 * t + 1000.0,
        "geometric": 0.9**t,
        "alternating": (-1.0) ** t,
        "exact-AR(1)": 2.0 + 5.0 * 0.7**t,
    }[kind]


COLLINEAR = "is constant (or collinear with its lags) over the {window} used for the AR({p})"
EXACT = "is fitted (numerically) exactly by an AR({p}) with constant over the {window}"
NAMED_CASES = [
    pytest.param("constant", 1, 60, COLLINEAR, id="constant-p1"),
    pytest.param("constant", 3, 60, COLLINEAR, id="constant-p3"),
    pytest.param("period-2", 2, 60, COLLINEAR, id="period-2-p2"),
    pytest.param("trend", 2, 60, COLLINEAR, id="trend-p2"),
    pytest.param("trend", 1, 8, EXACT, id="trend-p1-T8"),
    pytest.param("trend", 1, 30, EXACT, id="trend-p1-T30"),
    pytest.param("trend", 1, 120, EXACT, id="trend-p1-T120"),
    pytest.param("affine-trend", 1, 60, EXACT, id="affine-trend-p1"),
    pytest.param("geometric", 1, 60, EXACT, id="geometric-p1"),
    pytest.param("alternating", 1, 60, EXACT, id="alternating-p1"),
    pytest.param("exact-AR(1)", 1, 40, EXACT, id="exact-ar1"),
]


def _named_frame(kind: str, T: int) -> pd.DataFrame:
    frame = noise_frame(T).rename(columns={"y2": "rate"})
    frame["rate"] = _series(kind, T)
    return frame


@pytest.mark.parametrize(("kind", "lags", "T", "template"), NAMED_CASES)
def test_degenerate_series_raise_a_value_error_naming_the_variable(kind, lags, T, template):
    with pytest.raises(ValueError) as e:
        bvar("fixed", lags=lags).fit(to_data(_named_frame(kind, T)))
    assert_value_error(e)
    message = str(e.value)
    assert "'rate'" in message
    assert template.format(window="sample", p=lags) in message
    assert "variable 1" not in message


def test_near_exact_but_noisy_series_is_not_rejected():
    """The exact-fit test is relative to round-off: a trend with tiny noise is real data."""
    frame = _named_frame("trend", 60)
    frame["rate"] += 1e-4 * np.random.default_rng(7).standard_normal(60)
    fit = fit_recording(bvar("fixed", lags=1), to_data(frame))
    assert 1e-9 < fit.psi[1] < 1e-7


def _policy_rate_frame() -> pd.DataFrame:
    """Monthly 2016-04 … 2023-12; ``rate`` sits at exactly 0 until 2022-06, then rises."""
    frame = burst_frame("2016-04", "2023-12").rename(columns={"y3": "rate"})
    hikes = np.cumsum(np.random.default_rng(3).uniform(0.1, 0.5, len(frame)))
    frame["rate"] = np.where(frame.index < pd.Period("2022-07", freq="M"), 0.0, hikes)
    return frame


def test_series_constant_before_t_star_raises_naming_variable_and_window():
    frame = _policy_rate_frame()
    with pytest.raises(ValueError) as e:
        lp_bvar("eb").fit(to_data(frame))
    assert_value_error(e)
    window = f"periods before t* = {T_STAR['M']}"
    assert f"'rate' {COLLINEAR.format(window=window, p=2)}" in str(e.value)
    # the same panel fits without the scaling (the full sample includes the hikes) …
    with pytest.warns(UserWarning, match="covers 2020"):
        assert np.all(np.isfinite(bvar("eb").fit(to_data(frame)).psi))
    # … and with the scaling once psi is given
    fit = fit_recording(lp_bvar("eb", psi=GIVEN_PSI), to_data(frame))
    np.testing.assert_array_equal(fit.psi, GIVEN_PSI)


@pytest.mark.parametrize(
    ("kind", "lags", "template"),
    [
        pytest.param("trend", 1, EXACT, id="exact-trend"),
        pytest.param("period-2", 2, COLLINEAR, id="collinear"),
        pytest.param("geometric", 1, EXACT, id="exact-geometric"),
    ],
)
def test_series_degenerate_before_t_star_raises_naming_variable_and_window(kind, lags, template):
    frame = burst_frame("2015-01", "2023-12").rename(columns={"y1": "ip"})
    pre = frame.index < T_STAR["M"]
    frame.loc[pre, "ip"] = _series(kind, int(pre.sum()))
    with pytest.raises(ValueError) as e:
        lp_bvar("eb", lags=lags).fit(to_data(frame))
    assert_value_error(e)
    window = f"periods before t* = {T_STAR['M']}"
    assert f"'ip' {template.format(window=window, p=lags)}" in str(e.value)


def test_series_constant_over_the_fallback_rows_raises_naming_variable_and_window():
    t_star = T_STAR["M"]
    frame = burst_frame("2020-01", "2022-12").rename(columns={"y2": "rate"})
    frame.loc[t_star + 3 :, "rate"] = 0.25  # every row the fallback keeps is constant
    with pytest.raises(ValueError) as e:
        lp_bvar("eb").fit(to_data(frame))
    assert_value_error(e)
    window = f"window without the rows touching {t_star}..{t_star + 2}"
    assert f"'rate' {COLLINEAR.format(window=window, p=2)}" in str(e.value)


# --------------------------------------------------------------------------- small-df note


@pytest.mark.parametrize(
    ("lags", "T", "noted"),
    [(2, 14, True), (2, 15, False), (1, 12, True), (1, 13, False), (4, 18, True), (4, 19, False)],
)
def test_small_df_note_below_ten_degrees_of_freedom(lags, T, noted):
    """``df = T − 2p − 1``: a note when ``df < 10``, recorded in fit.warnings and forecasts."""
    frame = noise_frame(T)
    fit = fit_recording(bvar("fixed", lags=lags), to_data(frame))
    expected = (
        f"psi='ar': AR({lags}) residual variances of ['y1', 'y2', 'y3'] rest on fewer than 10 "
        "degrees of freedom over the sample; consider passing psi=(...)"
    )
    assert (expected in fit.warnings) is noted
    assert psi_notes(fit) == ([expected] if noted else [])
    attrs = fit.forecast(2, draws=20, seed=0).dataset.attrs["warnings"]
    assert (expected in attrs.split("\n")) is noted


def test_small_df_note_before_t_star():
    t_star = T_STAR["M"]
    frame = burst_frame(str(t_star - 8), "2023-12")  # 8 pre-t* periods with p = 2: df = 3
    fit = fit_recording(lp_bvar("eb"), to_data(frame))
    np.testing.assert_allclose(fit.psi, oracle_psi(frame[frame.index < t_star], 2), rtol=1e-8)
    assert psi_notes(fit) == [
        f"psi='ar': AR(2) residual variances of ['y1', 'y2', 'y3'] rest on fewer than 10 degrees "
        f"of freedom over the periods before t* = {t_star}; consider passing psi=(...)"
    ]


def test_no_small_df_note_with_explicit_psi():
    fit = fit_recording(bvar("fixed", lags=2, psi=GIVEN_PSI), to_data(noise_frame(14)))
    assert fit.warnings == []


# --------------------------------------------------------------------------- psi bounds


def _psi_hyperpriors(lower=None, upper=None) -> rb.GLPHyperpriors:
    return rb.GLPHyperpriors(
        psi=rb.InvGammaPrior(scale=0.02**2, shape=0.02**2, lower=lower, upper=upper)
    )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [pytest.param(30.0, None, id="lower"), pytest.param(None, 0.005, id="upper")],
)
def test_a_lone_psi_bound_that_empties_the_interval_names_the_variables(lower, upper):
    frame = noise_frame(120)
    mode = oracle_psi(frame, 2)
    lo = mode / 100.0 if lower is None else np.full(3, lower)
    hi = mode * 100.0 if upper is None else np.full(3, upper)
    inverted = [v for v, a, b in zip(VARS, lo, hi, strict=True) if not a < b]
    assert 0 < len(inverted) < 3  # the case separates variables
    prior = dict(hyper=("lambda", "psi"), hyperpriors=_psi_hyperpriors(lower, upper))
    with pytest.raises(ValueError, match=re.escape(f"psi bounds are empty for {inverted}")) as e:
        bvar("eb", **prior).fit(to_data(frame))
    assert type(e.value) is ValueError
    assert "set both InvGammaPrior bounds" in str(e.value)


def test_psi_bounds_with_both_ends_given_do_not_depend_on_the_ar_mode():
    frame = noise_frame(120)
    prior = rb.Minnesota(hyper=("lambda", "psi"), hyperpriors=_psi_hyperpriors(30.0, 40.0))
    free = ConjugateProblem(to_data(frame), prior, 2, "c", None).free
    idx = [free.names.index(f"psi_{v}") for v in VARS]
    np.testing.assert_array_equal(free.lower[idx], 30.0)
    np.testing.assert_array_equal(free.upper[idx], 40.0)


def test_psi_bound_check_under_covid_scaling_uses_the_pre_t_star_mode():
    """A lone lower bound above every pre-``t*`` mode × 100 empties all three intervals under
    the scaling, but only some of them with the (burst-inflated) full-sample mode."""
    frame = burst_frame()
    pre = oracle_psi(frame[frame.index < T_STAR["M"]], 2)
    full = oracle_psi(frame, 2)
    lower = 101.0 * pre.max()
    inverted_full = [v for v, m in zip(VARS, full, strict=True) if not lower < 100.0 * m]
    assert len(inverted_full) < 3  # the case discriminates between the two modes
    prior = rb.Minnesota(hyper=("lambda", "psi"), hyperpriors=_psi_hyperpriors(lower=lower))
    with pytest.raises(ValueError, match=re.escape(f"psi bounds are empty for {list(VARS)}")):
        ConjugateProblem(to_data(frame), prior, 2, "c", rb.LenzaPrimiceri())
    if inverted_full:
        with pytest.raises(
            ValueError, match=re.escape(f"psi bounds are empty for {inverted_full}")
        ):
            ConjugateProblem(to_data(frame), prior, 2, "c", None)
    else:
        free = ConjugateProblem(to_data(frame), prior, 2, "c", None).free
        idx = [free.names.index(f"psi_{v}") for v in VARS]
        np.testing.assert_allclose(free.upper[idx], 100.0 * full, rtol=1e-10)


# --------------------------------------------------------------------------- rolling backtests

ROLLING_ORIGINS = {  # Rolling(60), p = 2: pre-t* periods in the window
    "2023-06": 20,  # ≥ 2p + 11: no note
    "2024-06": 8,  # 2p + 2 … 2p + 10: small-df note
    "2024-12": 2,  # < 2p + 2: fallback
    "2025-06": 0,  # window starts after t*: fallback
}


@pytest.fixture(scope="module")
def rolling_data() -> rb.VARData:
    return to_data(burst_frame(seed=3))


def _run_rolling(model, data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = rb.backtest(
            model, data, origins=list(ROLLING_ORIGINS), window=rb.Rolling(60), horizon=3, draws=200
        )
    return result, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_rolling_backtest_across_t_star_completes_with_per_origin_notes(rolling_data):
    result, _ = _run_rolling(lp_bvar("eb"), rolling_data)
    assert result.origins == list(ROLLING_ORIGINS)
    assert np.all(np.isfinite(result.scores["lpd"].values))
    fallback = [m for m in result.warnings if FALLBACK in m]
    small = [m for m in result.warnings if SMALL_DF in m]
    assert fallback == [fallback_message(2, T_STAR["M"])]
    assert result.warnings[fallback[0]] == ["2024-12", "2025-06"]
    assert len(small) == 1 and result.warnings[small[0]] == ["2024-06"]


def test_rolling_compare_across_t_star_completes(rolling_data):
    models = {"ar": lp_bvar("eb"), "given": lp_bvar("eb", psi=GIVEN_PSI)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = rb.compare(
            models,
            rolling_data,
            origins=list(ROLLING_ORIGINS),
            window=rb.Rolling(60),
            horizons=(1,),
            draws=100,
        )
    for name in models:
        assert result.results[name].origins == list(ROLLING_ORIGINS)
        assert np.all(np.isfinite(result.scores(name)["lpd"].values))
    assert not any(m.startswith("psi='ar'") for m in result.results["given"].warnings)


def test_rolling_backtest_reports_each_psi_note_with_its_own_origins(rolling_data):
    """The emitted summary must not hide the fallback behind another ψ note, nor attribute the
    other note to origins where it did not occur."""
    _, emitted = _run_rolling(lp_bvar("eb"), rolling_data)
    fallback = [m for m in emitted if FALLBACK in m]
    small = [m for m in emitted if SMALL_DF in m]
    assert len(fallback) == 1, emitted
    assert "at 2 of 4 origins" in fallback[0], fallback
    assert len(small) == 1, emitted
    assert "at 1 of 4 origins" in small[0], small


@pytest.mark.slow
@pytest.mark.parametrize(
    ("freq", "lags", "size", "first_origin", "last_origin"),
    [
        pytest.param("M", 4, 48, "2022-10", "2025-10", id="monthly-p4"),
        pytest.param("M", 2, 60, "2024-01", "2025-10", id="monthly-p2"),
        pytest.param("Q", 2, 24, "2023Q1", "2025Q2", id="quarterly-p2"),
    ],
)
def test_rolling_backtest_never_aborts_at_any_origin_across_t_star(
    freq, lags, size, first_origin, last_origin
):
    data = to_data(burst_frame(freq=freq, seed=5))
    origins = rb.Origins(first_origin, last_origin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = rb.backtest(
            lp_bvar("eb", lags=lags),
            data,
            origins=origins,
            window=rb.Rolling(size),
            horizon=2,
            draws=200,
        )
    resolved = origins.resolve(data.index)
    assert result.origins == [str(o) for o in resolved]
    assert np.all(np.isfinite(result.scores["lpd"].values))
    t_star = T_STAR[freq]
    pre = {str(o): max(0, (t_star - (o - size + 1)).n) for o in resolved}
    fallback = result.warnings.get(fallback_message(lags, t_star), [])
    assert fallback == [o for o, k in pre.items() if k < 2 * lags + 2]
    small = sorted(
        {o for m, os in result.warnings.items() if SMALL_DF in m and "before t*" in m for o in os}
    )
    assert small == [o for o, k in pre.items() if 2 * lags + 2 <= k < 2 * lags + 11]
    assert fallback and small  # the origins cover both regimes


@pytest.mark.slow
def test_rolling_backtest_across_t_star_completes_under_mh():
    data = to_data(burst_frame(seed=3))
    mh = rb.MHSettings(draws=150, burn=150, chains=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = rb.backtest(
            lp_bvar("mh", mh=mh),
            data,
            origins=list(ROLLING_ORIGINS),
            window=rb.Rolling(60),
            horizon=2,
            draws=200,
        )
    assert result.origins == list(ROLLING_ORIGINS)
    assert np.all(np.isfinite(result.scores["lpd"].values))
    assert result.warnings[fallback_message(2, T_STAR["M"])] == ["2024-12", "2025-06"]


@pytest.mark.slow
def test_fallback_psi_tracks_the_innovation_variance_across_seeds():
    """Over many samples the fallback ψ stays near the pre-burst innovation variance, while the
    unmasked window AR variance is inflated by the burst."""
    fallback, unmasked = [], []
    for seed in range(40):
        frame = burst_frame("2019-12", "2024-12", truth=(30.0, 20.0, 15.0, 0.3), seed=100 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fallback.append(lp_bvar("fixed").fit(to_data(frame)).psi / SD**2)
        unmasked.append(oracle_psi(frame, 2) / SD**2)
    median = np.median(fallback, axis=0)
    assert np.all((median > 0.8) & (median < 1.25)), median
    assert np.all(np.median(unmasked, axis=0) > 10.0)
