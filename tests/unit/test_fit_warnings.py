"""Warnings raised by ``fit()`` and their aggregation across backtest origins (SPEC §5.3, §5.6,
§10.6).

* §5.3: ``fit()`` warns when user-specified own-lag prior means contradict the transformation
  codes: a persistent code (1, 4) with mean < 0.5, or a differenced code with mean ≥ 0.5.
  The warning names exactly the offending variables; ``mean=None`` and agreeing means are silent.
* §5.6: ``fit()`` warns when the estimation sample overlaps March–December 2020 and no outlier
  treatment is set, at any frequency (monthly, quarterly, annual incl. fiscal years, weekly).
  Nothing is said when the sample ends before March 2020 or starts after 2020, or when
  Lenza-Primiceri scaling is on.
* MH and EB: a non-converged run warns with its reasons (failing parameters, single chain,
  optimiser message) and is flagged ``converged = False``.
* Every warning is kept on ``fit.warnings``, copied into the forecast attrs (``"warnings"``),
  and points at the user's ``fit()`` call.
* §10.6: ``backtest`` records each origin's user warnings, emits each distinct message once as
  "... [at k of N origins: first..last]" (number-bearing MH/EB messages grouped by kind), keeps
  ``BacktestResult.warnings`` (message → origins), keeps them on cache hits and in the process
  pool, and lets other categories (RuntimeWarning, an error under this suite's filters) through.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

import rai_bvar as rb
from rai_bvar.engines import analytic
from rai_bvar.models import OLSVAR

QD_SMALL = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "qd_small_levels.csv"
TCODE_WARNING = "contradict the transformation codes"
COVID_WARNING = "no outlier treatment"
MH_WARNING = "hyperparameter MH did not converge"
EB_WARNING = "hyperparameter optimisation did not converge"
NOISY_WARNING = "noisy model: a user warning raised twice per fit"
RUNTIME_WARNING = "noisy model: a numerical problem"
LP_FIXED = (3.0, 2.0, 1.5, 0.7)
MONTHS_2020H1 = rb.Origins("2020-01", "2020-06")  # COVID rows enter from the 2020-03 origin on
QUARTERS_2010 = rb.Origins("2010Q1", "2010Q3")  # nothing about COVID


# --------------------------------------------------------------------------- data and helpers


def _frame(freq: str, start: str, periods: int, seed: int = 0) -> pd.DataFrame:
    """Two stationary series, ``y_t = 0.3 + 0.5 y_{t−1} + e_t``."""
    rng = np.random.default_rng(seed)
    y = np.zeros((periods, 2))
    for t in range(1, periods):
        y[t] = 0.3 + 0.5 * y[t - 1] + rng.standard_normal(2)
    index = pd.period_range(start, periods=periods, freq=freq)
    return pd.DataFrame(y, index=index, columns=["x", "z"])


def _window(data: rb.VARData, first: str | None, last: str | None) -> rb.VARData:
    """``data`` cut to ``first..last`` (``None``: keep that end). ``first`` is the first
    observation, a lag only: the estimation rows start one period later (lags=1)."""
    if last is not None:
        data = data.upto(last)
    if first is not None:
        data = data.tail((data.last - pd.Period(first, freq=data.index.freq)).n + 1)
    return data


def fixed_bvar(outliers=None, lags: int = 1, **prior) -> rb.BVAR:
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(hierarchy="fixed", **prior)),
        lags=lags,
        outliers=outliers,
    )


def mh_bvar(lags: int = 1, prior: dict | None = None, **settings) -> rb.BVAR:
    minnesota = rb.Minnesota(hierarchy="mh", mh=rb.MHSettings(**settings), **(prior or {}))
    return rb.BVAR(mean=rb.Linear(prior=minnesota), lags=lags)


def _record(fn):
    """Run ``fn`` recording every UserWarning (other categories keep the suite's filters)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        out = fn()
    return out, caught


def _user_messages(caught) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _fit_recording(model, data, **kwargs):
    fit, caught = _record(lambda: model.fit(data, **kwargs))
    return fit, _user_messages(caught)


def _only(messages: list[str], fragment: str) -> str:
    matching = [m for m in messages if fragment in m]
    assert len(matching) == 1, messages
    return matching[0]


@dataclass(frozen=True)
class Noisy:
    """OLS-VAR(1) whose fit raises the same UserWarning twice and, optionally, a RuntimeWarning.

    Module level (picklable) so the process-pool backtest can run it in its workers.
    """

    runtime: bool = False

    @property
    def name(self) -> str:
        return "noisy"

    @property
    def axes(self) -> dict[str, str]:
        return {"model": "noisy"}

    def spec(self) -> dict:
        return {"class": "Noisy", "runtime": self.runtime}

    def fit(self, data: rb.VARData):
        for _ in range(2):
            warnings.warn(NOISY_WARNING, UserWarning, stacklevel=2)
        if self.runtime:
            warnings.warn(RUNTIME_WARNING, RuntimeWarning, stacklevel=2)
        return OLSVAR(1).fit(data)


def _no_progress(fun, x0, **kwargs):
    """Stand-in for ``scipy.optimize.minimize`` that stops at the start without iterating."""
    x0 = np.asarray(x0, dtype=np.float64)
    return optimize.OptimizeResult(
        x=x0, fun=fun(x0), success=False, message="stopped by the test before iterating", nit=0
    )


@pytest.fixture(scope="module")
def qd() -> rb.VARData:
    """Quarterly 100·log GDP, 100·log prices and the policy rate, 1960Q1–2019Q4, tcode 1."""
    frame = pd.read_csv(QD_SMALL, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return rb.VARData.from_frame(frame, tcodes=1)


@pytest.fixture(scope="module")
def mixed() -> rb.VARData:
    """Quarterly 1970Q1–2019Q4 with one variable per tcode class: LVL1 (1) and LOG4 (4) are
    persistent, DIF2 (2) and GRO5 (5) differenced."""
    rng = np.random.default_rng(11)
    T = 200
    e = rng.standard_normal((T, 4))
    ar = np.zeros((T, 2))
    for t in range(1, T):
        ar[t] = 0.7 * ar[t - 1] + e[t, :2]
    frame = pd.DataFrame(
        {
            "LVL1": 2.0 + ar[:, 0],
            "LOG4": 50.0 * np.exp(0.05 * ar[:, 1]),
            "DIF2": np.cumsum(0.1 + e[:, 2]),
            "GRO5": 100.0 * np.exp(np.cumsum(0.005 + 0.01 * e[:, 3])),
        },
        index=pd.period_range("1970Q1", periods=T, freq="Q"),
    )
    return rb.VARData.from_frame(frame, {"LVL1": 1, "LOG4": 4, "DIF2": 2, "GRO5": 5})


@pytest.fixture(scope="module")
def samples() -> dict[str, rb.VARData]:
    """Simulated stationary samples (tcode 1) that run well past 2020, by frequency."""
    specs = {
        "M": ("2015-01", 120),  # 2015-01..2024-12
        "Q": ("2000Q1", 120),  # 2000Q1..2029Q4
        "Y": ("1950", 100),  # 1950..2049
        "Y-JUN": ("1950", 100),  # fiscal years ending in June: FY2020 = 2019-07..2020-06
        "W": ("2015-01-05", 600),  # weeks ending Sunday, 2015-01..2026-07
    }
    return {
        freq: rb.VARData.from_frame(_frame(freq, start, periods), 1)
        for freq, (start, periods) in specs.items()
    }


# =========================================================================== §5.3 tcode rule

TCODE_CASES = [
    (1.0, {"DIF2", "GRO5"}),
    (0.99, {"DIF2", "GRO5"}),
    (0.9, {"DIF2", "GRO5"}),
    (1.2, {"DIF2", "GRO5"}),
    (0.5, {"DIF2", "GRO5"}),  # boundary: a differenced code with mean ≥ 0.5
    (0.4999, {"LVL1", "LOG4"}),  # boundary: a persistent code with mean < 0.5
    (0.1, {"LVL1", "LOG4"}),
    (0.0, {"LVL1", "LOG4"}),
    (-0.5, {"LVL1", "LOG4"}),
    ({"GRO5": 0.99}, {"GRO5"}),
    ({"GRO5": 0.97, "DIF2": 0.97}, {"GRO5", "DIF2"}),
    ({"DIF2": 1.000000000001}, {"DIF2"}),
    ({"LVL1": 0.1}, {"LVL1"}),
    ({"LOG4": 0.0}, {"LOG4"}),
    ({"LOG4": -0.5, "GRO5": 1.2, "LVL1": 0.9}, {"LOG4", "GRO5"}),
]


@pytest.mark.parametrize(("mean", "clash"), TCODE_CASES, ids=[repr(m) for m, _ in TCODE_CASES])
def test_tcode_warning_names_exactly_the_contradicting_variables(mixed, mean, clash):
    fit, messages = _fit_recording(fixed_bvar(mean=mean), mixed)
    message = _only(messages, TCODE_WARNING)
    assert {v for v in mixed.variables if v in message} == clash, message
    assert fit.warnings == messages == [message]


SILENT_TCODE_CASES = [
    ("all", None),
    ("all", {"LVL1": 0.9, "LOG4": 1.0, "DIF2": 0.0, "GRO5": 0.2}),
    ("all", {"GRO5": 0.0}),
    ("all", {"DIF2": 0.4999}),
    ("all", {"LOG4": 0.5}),
    ("levels", 0.9),
    ("levels", 0.5),
    ("levels", 1.0),
    ("differences", 0.0),
    ("differences", 0.2),
    ("differences", 0.4999),
]


@pytest.mark.parametrize(
    ("subset", "mean"), SILENT_TCODE_CASES, ids=[f"{s}-{m!r}" for s, m in SILENT_TCODE_CASES]
)
def test_tcode_warning_is_silent_when_the_means_agree(mixed, subset, mean):
    variables = {
        "all": mixed.variables,
        "levels": ("LVL1", "LOG4"),
        "differences": ("DIF2", "GRO5"),
    }[subset]
    fit, messages = _fit_recording(fixed_bvar(mean=mean), mixed.select(variables))
    assert messages == [] and fit.warnings == []


def test_tcode_warning_also_fires_under_empirical_bayes(mixed):
    fit, messages = _fit_recording(rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(mean=1.0))), mixed)
    assert {v for v in mixed.variables if v in _only(messages, TCODE_WARNING)} == {"DIF2", "GRO5"}
    assert fit.warnings == messages


# =========================================================================== §5.6 COVID sample

COVID_CASES = [
    # (frequency, first observation, last observation, warns)
    ("M", None, "2020-02", False),
    ("M", None, "2020-03", True),
    ("M", None, "2020-12", True),
    ("M", None, None, True),
    ("M", "2020-06", None, True),
    ("M", "2021-01", None, False),
    ("Q", None, "2019Q4", False),
    ("Q", None, "2020Q1", True),
    ("Q", "2020Q3", None, True),
    ("Q", "2021Q1", None, False),
    ("Y", None, "2019", False),
    ("Y", None, "2020", True),
    ("Y", "2019", None, True),
    ("Y", "2021", None, False),
    ("Y-JUN", None, "2019", False),  # FY2019 ends 2019-06
    ("Y-JUN", None, "2020", True),  # FY2020 = 2019-07..2020-06 contains March 2020
    ("Y-JUN", "2020", None, True),
    ("Y-JUN", "2022", None, False),  # FY2022 starts 2021-07
    ("W", None, "2020-02-23", False),  # last week 2020-02-17..2020-02-23
    ("W", None, "2020-03-08", True),
    ("W", "2021-01-04", None, False),
]


@pytest.mark.parametrize(
    ("freq", "first", "last", "warns"),
    COVID_CASES,
    ids=[f"{f}:{a or 'start'}..{b or 'end'}" for f, a, b, _ in COVID_CASES],
)
def test_covid_warning_follows_the_estimation_sample(samples, freq, first, last, warns):
    data = _window(samples[freq], first, last)
    fit, messages = _fit_recording(fixed_bvar(), data)
    if warns:
        _only(messages, COVID_WARNING)
        assert messages == fit.warnings and len(messages) == 1, messages
    else:
        assert messages == [] and fit.warnings == [], (data, messages)


@pytest.mark.parametrize("freq", ["M", "Q"])
@pytest.mark.parametrize("hierarchy", ["fixed", "eb"])
def test_no_covid_warning_with_lenza_primiceri_scaling(samples, freq, hierarchy):
    data = _window(samples[freq], None, "2022-12" if freq == "M" else "2022Q4")
    model = (
        fixed_bvar(outliers=rb.LenzaPrimiceri(fixed=LP_FIXED))
        if hierarchy == "fixed"
        else rb.BVAR(outliers="lenza-primiceri")
    )
    fit, messages = _fit_recording(model, data)
    assert not [m for m in messages if COVID_WARNING in m], messages
    assert not [m for m in fit.warnings if COVID_WARNING in m]
    # the same sample without a treatment does warn
    _only(_fit_recording(fixed_bvar(), data)[1], COVID_WARNING)


@pytest.mark.parametrize("freq", ["Y", "W"])
def test_covid_warning_recommends_lenza_primiceri_only_where_it_is_available(samples, freq):
    """Lenza-Primiceri scaling supports monthly and quarterly data only, so at other
    frequencies the warning must not tell the user to set ``outliers='lenza-primiceri'``."""
    data = samples[freq]
    with pytest.raises(ValueError):
        fixed_bvar(outliers=rb.LenzaPrimiceri(fixed=LP_FIXED)).fit(data)
    message = _only(_fit_recording(fixed_bvar(), data)[1], COVID_WARNING)
    assert "outliers='lenza-primiceri'" not in message, message


# =========================================================================== fit.warnings, attrs


def test_fit_records_its_warnings_and_every_forecast_carries_them(samples):
    data = samples["M"].upto("2020-06")
    fit, messages = _fit_recording(fixed_bvar(mean=0.0), data)
    assert len(messages) == 2
    _only(messages, TCODE_WARNING)
    _only(messages, COVID_WARNING)
    assert fit.warnings == messages
    for horizon, seed in [(2, 0), (1, 5)]:
        attrs = fit.forecast(horizon, 20, seed=seed).dataset.attrs
        assert isinstance(attrs["warnings"], str)
        assert attrs["warnings"].split("\n") == fit.warnings


def test_clean_fit_has_no_warnings_and_an_empty_attr(samples):
    fit, messages = _fit_recording(fixed_bvar(), samples["M"].upto("2019-12"))
    assert messages == [] and fit.warnings == []
    attrs = fit.forecast(2, 20, seed=0).dataset.attrs
    assert attrs["warnings"] == "" and attrs["converged"] == 1


def test_warnings_are_recorded_even_when_the_user_ignores_them(samples):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", UserWarning)
        fit = fixed_bvar(mean=0.0).fit(samples["M"].upto("2020-06"))
    assert _user_messages(caught) == []
    assert len(fit.warnings) == 2
    assert fit.forecast(1, 10, seed=0).dataset.attrs["warnings"].split("\n") == fit.warnings


def test_each_fit_keeps_its_own_warnings(samples):
    noisy, _ = _fit_recording(fixed_bvar(mean=0.0), samples["M"].upto("2020-06"))
    clean, _ = _fit_recording(fixed_bvar(), samples["M"].upto("2019-06"))
    assert len(noisy.warnings) == 2 and clean.warnings == []


def test_setting_warnings_point_at_the_users_fit_call(samples):
    with pytest.warns(UserWarning) as record:
        fixed_bvar(mean=0.0).fit(samples["M"].upto("2020-06"))
    assert len(record) == 2
    assert all(w.filename == __file__ for w in record), [(w.filename, w.lineno) for w in record]


# =========================================================================== MH non-convergence


def test_single_chain_warning_explains_why_without_an_empty_dict(qd):
    """Thresholds every chain passes: the single chain is the only reason for the flag."""
    fit, messages = _fit_recording(
        mh_bvar(draws=200, burn=200, chains=1, rhat_max=10.0, ess_min=0.0), qd
    )
    message = _only(messages, MH_WARNING)
    assert not fit.converged and not fit.mh.converged
    assert "{}" not in message, message
    assert re.search(r"single chain", message, re.IGNORECASE), message
    assert re.search(r"chains\s*(≥|>=|>)\s*[12]", message), message
    assert "lambda" not in message, message  # no parameter failed a threshold
    assert fit.warnings == messages


def test_single_chain_warning_also_lists_failing_parameters(qd):
    fit, messages = _fit_recording(mh_bvar(draws=200, burn=200, chains=1), qd)
    message = _only(messages, MH_WARNING)
    assert fit.mh.ess["lambda"] < 400  # premise: the default ESS threshold fails
    assert "lambda" in message and re.search(r"single chain", message, re.IGNORECASE)
    assert "{}" not in message


def test_multi_chain_warning_lists_exactly_the_failing_parameters(qd):
    """Same seed, so identical chains; only the ESS threshold changes between the two fits."""
    prior = {"soc": 1.0, "sur": 1.0}
    lenient, messages = _fit_recording(
        mh_bvar(prior=prior, draws=300, burn=200, chains=2, rhat_max=10.0, ess_min=0.0), qd
    )
    assert lenient.converged and messages == [] and lenient.warnings == []
    ess = lenient.mh.ess
    assert set(ess) == {"lambda", "soc", "sur"}
    assert max(ess.values()) > min(ess.values())  # premise: the threshold separates them
    threshold = 0.5 * (max(ess.values()) + min(ess.values()))
    strict, messages = _fit_recording(
        mh_bvar(prior=prior, draws=300, burn=200, chains=2, rhat_max=10.0, ess_min=threshold), qd
    )
    np.testing.assert_array_equal(strict.mh.chains, lenient.mh.chains)
    message = _only(messages, MH_WARNING)
    failing = {name for name, value in ess.items() if not value > threshold}
    assert {name for name in ess if name in message} == failing, (message, ess, threshold)
    assert "{}" not in message and not re.search(r"single chain", message, re.IGNORECASE)
    assert not strict.converged


def test_mh_warning_reaches_fit_warnings_and_forecast_attrs(qd):
    fit, messages = _fit_recording(mh_bvar(draws=100, burn=0, chains=2), qd)
    message = _only(messages, MH_WARNING)
    assert fit.warnings == [message]
    attrs = fit.forecast(2, 50, seed=0).dataset.attrs
    assert attrs["converged"] == 0 and message in attrs["warnings"].split("\n")
    assert attrs["mh_seed"] == fit.mh_seed


def test_mh_warning_points_at_the_users_fit_call(qd):
    with pytest.warns(UserWarning, match=MH_WARNING) as record:
        mh_bvar(draws=100, burn=0, chains=2).fit(qd)
    assert [w.filename for w in record] == [__file__]


def test_converged_mh_raises_no_warning(qd):
    fit, messages = _fit_recording(
        mh_bvar(draws=300, burn=200, chains=2, rhat_max=10.0, ess_min=0.0), qd
    )
    assert fit.converged and messages == [] and fit.warnings == []
    assert fit.forecast(1, 20, seed=0).dataset.attrs["warnings"] == ""


# =========================================================================== EB failure


def test_eb_failure_warns_once_and_marks_the_fit_unconverged(qd, monkeypatch):
    monkeypatch.setattr(analytic, "optimize", SimpleNamespace(minimize=_no_progress))
    with pytest.warns(UserWarning, match=EB_WARNING) as record:
        fit = rb.BVAR().fit(qd)
    messages = [str(w.message) for w in record]
    message = _only(messages, EB_WARNING)
    assert "stopped by the test" in message  # the optimiser's own reason is reported
    assert [w.filename for w in record] == [__file__]
    assert not fit.converged and not fit.optimization["success"]
    assert fit.warnings == [message]
    attrs = fit.forecast(1, 20, seed=0).dataset.attrs
    assert attrs["converged"] == 0 and attrs["warnings"] == message


@pytest.mark.parametrize(
    "case", ["qd-p1", "qd-p4-soc-sur", "monthly-lenza-primiceri"], ids=lambda c: c
)
def test_converged_eb_fit_raises_no_warning(qd, samples, case):
    if case == "qd-p1":
        model, data = rb.BVAR(), qd
    elif case == "qd-p4-soc-sur":
        model, data = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(soc=1.0, sur=1.0)), lags=4), qd
    else:
        model, data = rb.BVAR(outliers="lenza-primiceri"), samples["M"].upto("2022-12")
    fit, messages = _fit_recording(model, data)
    assert fit.converged, fit.optimization
    assert messages == [] and fit.warnings == []


# =========================================================================== backtest aggregation


def test_backtest_emits_each_distinct_warning_once_with_its_origins(samples):
    data, model = samples["M"], fixed_bvar(mean=0.0)
    expected, _ = _record(lambda: model.fit(data.upto("2020-06")).warnings)
    tcode, covid = _only(expected, TCODE_WARNING), _only(expected, COVID_WARNING)
    result, caught = _record(
        lambda: rb.backtest(model, data, origins=MONTHS_2020H1, horizon=1, draws=20)
    )
    emitted = _user_messages(caught)
    assert len(emitted) == 2, emitted
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    assert result.warnings == {tcode: months, covid: months[2:]}
    line = _only(emitted, TCODE_WARNING)
    assert line.startswith(model.name) and tcode in line
    assert "at 6 of 6 origins: 2020-01..2020-06" in line, line
    line = _only(emitted, COVID_WARNING)
    assert covid in line and "at 4 of 6 origins: 2020-03..2020-06" in line, line
    assert all(w.filename == __file__ for w in caught)
    assert "warnings" not in result.scores.attrs


def test_backtest_without_warnings_records_none(samples):
    result, caught = _record(
        lambda: rb.backtest(
            fixed_bvar(), samples["M"], origins=["2019-01", "2019-06"], horizon=1, draws=20
        )
    )
    assert _user_messages(caught) == [] and result.warnings == {}


def test_backtest_counts_a_repeated_warning_once_per_origin(samples):
    result, caught = _record(
        lambda: rb.backtest(Noisy(), samples["Q"], origins=QUARTERS_2010, horizon=1, draws=20)
    )
    assert result.warnings == {NOISY_WARNING: ["2010Q1", "2010Q2", "2010Q3"]}
    line = _only(_user_messages(caught), NOISY_WARNING)
    assert "at 3 of 3 origins: 2010Q1..2010Q3" in line, line


def test_backtest_groups_number_bearing_mh_warnings(samples):
    model = mh_bvar(draws=60, burn=0, chains=2)
    result, caught = _record(
        lambda: rb.backtest(model, samples["Q"], origins=QUARTERS_2010, horizon=1, draws=20)
    )
    # premise: at least two origins report different diagnostics (distinct texts, one kind)
    assert len(result.warnings) >= 2, result.warnings
    assert all(MH_WARNING in m for m in result.warnings)
    assert sorted(o for os_ in result.warnings.values() for o in os_) == [
        "2010Q1",
        "2010Q2",
        "2010Q3",
    ]
    line = _only(_user_messages(caught), MH_WARNING)
    assert "at 3 of 3 origins: 2010Q1..2010Q3" in line, line
    np.testing.assert_array_equal(result.scores["converged"].values, 0.0)


def test_backtest_aggregates_eb_failures(samples, monkeypatch):
    monkeypatch.setattr(analytic, "optimize", SimpleNamespace(minimize=_no_progress))
    result, caught = _record(
        lambda: rb.backtest(rb.BVAR(), samples["Q"], origins=QUARTERS_2010, horizon=1, draws=20)
    )
    assert result.warnings and all(EB_WARNING in m for m in result.warnings)
    assert sorted({o for os_ in result.warnings.values() for o in os_}) == [
        "2010Q1",
        "2010Q2",
        "2010Q3",
    ]
    line = _only(_user_messages(caught), EB_WARNING)
    assert "at 3 of 3 origins: 2010Q1..2010Q3" in line, line
    np.testing.assert_array_equal(result.scores["converged"].values, 0.0)


def test_backtest_records_warnings_the_user_ignores(samples):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", UserWarning)
        result = rb.backtest(
            fixed_bvar(mean=0.0), samples["M"], origins=MONTHS_2020H1, horizon=1, draws=20
        )
    assert _user_messages(caught) == []
    assert len(result.warnings) == 2


def test_cache_hits_keep_their_warnings(samples, tmp_path):
    kwargs = dict(origins=MONTHS_2020H1, horizon=1, draws=20, cache=tmp_path)
    model = fixed_bvar(mean=0.0)
    first, caught_first = _record(lambda: rb.backtest(model, samples["M"], **kwargs))
    second, caught_second = _record(lambda: rb.backtest(model, samples["M"], **kwargs))
    assert (first.cached, second.cached) == (0, 6)
    assert len(first.warnings) == 2  # non-ASCII text (§, ≥) survives the netCDF cache
    assert second.warnings == first.warnings
    assert _user_messages(caught_second) == _user_messages(caught_first)
    assert len(_user_messages(caught_second)) == 2


def test_cache_hits_without_warnings_stay_silent(samples, tmp_path):
    kwargs = dict(origins=["2019-01", "2019-06"], horizon=1, draws=20, cache=tmp_path)
    for expected_hits in (0, 2):
        result, caught = _record(lambda: rb.backtest(fixed_bvar(), samples["M"], **kwargs))
        assert result.cached == expected_hits
        assert result.warnings == {} and _user_messages(caught) == []


def test_runtime_warnings_still_raise_under_the_error_filter(samples):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match=RUNTIME_WARNING):
            rb.backtest(
                Noisy(runtime=True), samples["Q"], origins=QUARTERS_2010, horizon=1, draws=20
            )


def test_runtime_warnings_pass_through_unaggregated(samples):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rb.backtest(
            Noisy(runtime=True), samples["Q"], origins=QUARTERS_2010, horizon=1, draws=20
        )
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime and all(str(w.message) == RUNTIME_WARNING for w in runtime)
    assert RUNTIME_WARNING not in result.warnings
    assert list(result.warnings) == [NOISY_WARNING]


def test_compare_aggregates_per_model(samples):
    noisy, clean = fixed_bvar(mean=0.0), fixed_bvar()
    result, caught = _record(
        lambda: rb.compare(
            [noisy, clean], samples["M"], origins=MONTHS_2020H1, horizons=(1,), draws=20
        )
    )
    emitted = _user_messages(caught)
    assert len(emitted) == 3, emitted  # noisy: tcode + COVID; clean: COVID
    assert sum(m.startswith(f"{noisy.name}: ") for m in emitted) == 2
    assert sum(m.startswith(f"{clean.name}: ") for m in emitted) == 1
    assert all("of 6 origins" in m for m in emitted)
    assert len(result.results[noisy.name].warnings) == 2
    (covid,) = result.results[clean.name].warnings
    assert COVID_WARNING in covid
    assert result.results[clean.name].warnings[covid] == [f"2020-{m:02d}" for m in range(3, 7)]


# =========================================================================== process pool


def test_process_pool_aggregates_like_serial(samples):
    kwargs = dict(origins=MONTHS_2020H1, horizon=1, draws=20)
    model = fixed_bvar(mean=0.0)
    serial, caught_serial = _record(lambda: rb.backtest(model, samples["M"], **kwargs))
    pooled, caught_pooled = _record(lambda: rb.backtest(model, samples["M"], n_jobs=2, **kwargs))
    assert pooled.warnings == serial.warnings and len(pooled.warnings) == 2
    assert _user_messages(caught_pooled) == _user_messages(caught_serial)
    assert len(_user_messages(caught_pooled)) == 2


def test_process_pool_runtime_warnings_still_raise_under_the_error_filter(samples):
    """A RuntimeWarning raised in a worker must reach the parent's filters, as it does with
    ``n_jobs=1`` (above); printing it in the worker would bypass ``error::RuntimeWarning``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match=RUNTIME_WARNING):
            rb.backtest(
                Noisy(runtime=True),
                samples["Q"],
                origins=QUARTERS_2010,
                horizon=1,
                draws=20,
                n_jobs=2,
            )


# =========================================================================== slow


@pytest.mark.slow
def test_default_mh_settings_converge_without_warnings(qd):
    model = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(hierarchy="mh", soc=1.0, sur=1.0)), lags=4)
    fit, messages = _fit_recording(model, qd)
    assert fit.converged, (fit.mh.rhat, fit.mh.ess)
    assert messages == [] and fit.warnings == []
    attrs = fit.forecast(4, 200, seed=0).dataset.attrs
    assert attrs["warnings"] == "" and attrs["converged"] == 1


@pytest.mark.slow
def test_monthly_mh_backtest_through_2020_emits_one_warning_per_kind(samples):
    """Twelve 2020 origins, MH with too few draws to converge, no outlier treatment and an
    own-lag mean of 0 on level (tcode 1) data: three kinds of warning, each emitted once."""
    model = rb.BVAR(
        mean=rb.Linear(
            prior=rb.Minnesota(
                hierarchy="mh", mean=0.0, mh=rb.MHSettings(draws=100, burn=100, chains=2)
            )
        )
    )
    origins = rb.Origins("2020-01", "2020-12")
    result, caught = _record(
        lambda: rb.backtest(model, samples["M"], origins=origins, horizon=2, draws=50)
    )
    emitted = _user_messages(caught)
    assert len(emitted) == 3, emitted
    assert "at 12 of 12 origins: 2020-01..2020-12" in _only(emitted, TCODE_WARNING)
    assert "at 10 of 12 origins: 2020-03..2020-12" in _only(emitted, COVID_WARNING)
    _only(emitted, MH_WARNING)
    mh_origins = {o for m, os_ in result.warnings.items() if MH_WARNING in m for o in os_}
    assert f"at {len(mh_origins)} of 12 origins" in _only(emitted, MH_WARNING)


@pytest.mark.slow
def test_process_pool_compare_matches_serial_warnings(samples):
    models = [fixed_bvar(mean=0.0), rb.BVAR(outliers="lenza-primiceri"), fixed_bvar()]
    kwargs = dict(origins=rb.Origins("2019-10", "2020-09"), horizons=(1,), draws=20)
    serial, caught_serial = _record(lambda: rb.compare(models, samples["M"], **kwargs))
    pooled, caught_pooled = _record(lambda: rb.compare(models, samples["M"], n_jobs=2, **kwargs))
    for name, result in serial.results.items():
        assert pooled.results[name].warnings == result.warnings
    assert sorted(_user_messages(caught_pooled)) == sorted(_user_messages(caught_serial))
    lp = rb.BVAR(outliers="lenza-primiceri").name
    covid = [m for m in _user_messages(caught_serial) if COVID_WARNING in m]
    assert len(covid) == 2 and not [m for m in covid if m.startswith(f"{lp}: ")]
