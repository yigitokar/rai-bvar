"""``VARData``: construction, trimming, sub-samples, realised values and inversion (SPEC §5)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rai_bvar.data import VARData, transforms


def _quarterly(values: dict[str, np.ndarray], start: str = "2000Q1") -> pd.DataFrame:
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.period_range(start, periods=n, freq="Q"))


def _positive(n: int, seed: int = 0, scale: float = 100.0) -> np.ndarray:
    return scale * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.01, 0.02, n)))


# --------------------------------------------------------------------------- fixtures themselves


def test_sim_fixtures_reproduce_the_transformed_var(sim_data, sim_data_stationary, sim_transformed):
    assert sim_data.variables == ("gdp", "prices", "rate")
    assert sim_data.tcodes == {"gdp": 5, "prices": 5, "rate": 1}
    assert len(sim_data) == 240
    assert (sim_data.first, sim_data.last) == (pd.Period("1960Q1", "Q"), pd.Period("2019Q4", "Q"))
    np.testing.assert_allclose(sim_data.values, sim_transformed.to_numpy(), rtol=0, atol=1e-13)
    np.testing.assert_array_equal(sim_data_stationary.values, sim_transformed.to_numpy())
    assert set(sim_data_stationary.tcodes.values()) == {1}
    assert sim_data_stationary.index.equals(sim_data.index)


def test_sim_data_follows_the_true_var(sim_data, sim_truth):
    """OLS on the fixture recovers the documented VAR(1) within sampling error."""
    from rai_bvar.data import lagged_design
    from rai_bvar.linalg import qr_least_squares

    assert np.all(np.abs(np.linalg.eigvals(sim_truth.coefs)) < 1)
    np.testing.assert_allclose(sim_truth.const, (np.eye(3) - sim_truth.coefs) @ sim_truth.mean)
    Y, X = lagged_design(sim_data.values, 1, "c")
    B, _, resid = qr_least_squares(X, Y)
    sigma2 = (resid**2).sum(axis=0) / (X.shape[0] - X.shape[1])
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ X)), sigma2))
    assert np.all(np.abs(B - sim_truth.B) < 4 * se)
    # residual sds within ~3 standard errors (sd/√(2T) ≈ 5%), correlations within ~3 × 0.065
    sd_true = np.sqrt(np.diag(sim_truth.sigma))
    np.testing.assert_allclose(resid.std(axis=0), sd_true, rtol=0.15)
    corr_true = sim_truth.sigma / np.outer(sd_true, sd_true)
    np.testing.assert_allclose(np.corrcoef(resid, rowvar=False), corr_true, atol=0.2)


# --------------------------------------------------------------------------- from_frame: indexes


@pytest.mark.parametrize("with_freq", [True, False])
def test_from_frame_monthly_month_start(with_freq):
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    if not with_freq:
        idx = pd.DatetimeIndex(list(idx))  # e.g. read from CSV: frequency must be inferred
        assert idx.freq is None
    d = VARData.from_frame(pd.DataFrame({"a": np.arange(1.0, 25.0)}, index=idx))
    assert isinstance(d.index, pd.PeriodIndex)
    assert d.index.freqstr == "M"
    assert d.first == pd.Period("2000-01", "M")
    assert d.last == pd.Period("2001-12", "M")
    assert len(d) == 24


def test_from_frame_monthly_month_end():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    d = VARData.from_frame(pd.DataFrame({"a": np.arange(6.0)}, index=idx))
    assert d.first == pd.Period("2000-01", "M") and d.last == pd.Period("2000-06", "M")


@pytest.mark.parametrize(
    "dates",
    [
        ["3/1/1959", "6/1/1959", "9/1/1959", "12/1/1959", "3/1/1960"],  # FRED-QD convention
        ["1/1/1959", "4/1/1959", "7/1/1959", "10/1/1959", "1/1/1960"],  # quarter start
        ["3/31/1959", "6/30/1959", "9/30/1959", "12/31/1959", "3/31/1960"],  # quarter end
    ],
)
def test_from_frame_quarterly_dates_map_to_calendar_quarters(dates):
    idx = pd.to_datetime(dates, format="%m/%d/%Y")
    d = VARData.from_frame(pd.DataFrame({"a": np.arange(1.0, 6.0)}, index=idx))
    expected = pd.period_range("1959Q1", "1960Q1", freq="Q")
    assert d.index.equals(expected)
    assert d.freq == "Q-DEC"


def test_from_frame_annual_dates():
    idx = pd.date_range("2000-01-01", periods=5, freq="YS")
    d = VARData.from_frame(pd.DataFrame({"a": np.arange(5.0)}, index=idx))
    assert d.first == pd.Period("2000", "Y") and d.last == pd.Period("2004", "Y")


def test_from_frame_period_index_is_kept():
    frame = _quarterly({"a": np.arange(1.0, 9.0), "b": np.arange(9.0, 17.0)}, start="1999Q3")
    d = VARData.from_frame(frame)
    assert d.index.equals(frame.index)
    np.testing.assert_array_equal(d.values, frame.to_numpy())
    assert d.variables == ("a", "b") and d.n_vars == 2


def test_from_frame_casts_column_names_and_values():
    frame = pd.DataFrame(
        {0: [1, 2, 3, 4], 1: [5, 6, 7, 9]}, index=pd.period_range("2000Q1", periods=4, freq="Q")
    )
    d = VARData.from_frame(frame)
    assert d.variables == ("0", "1")
    assert d.values.dtype == np.float64


def test_from_frame_does_not_modify_input():
    frame = _quarterly({"a": np.arange(1.0, 6.0)})
    before = frame.copy()
    VARData.from_frame(frame, 5)
    pd.testing.assert_frame_equal(frame, before)


def test_datetime_index_with_gap_raises():
    idx = pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-04-01", "2000-05-01"])
    with pytest.raises(ValueError, match="frequency"):
        VARData.from_frame(pd.DataFrame({"a": np.arange(4.0)}, index=idx))


def test_period_index_with_gap_raises():
    idx = pd.PeriodIndex([pd.Period(p, "Q") for p in ["2000Q1", "2000Q2", "2000Q4", "2001Q1"]])
    with pytest.raises(ValueError, match="gaps"):
        VARData.from_frame(pd.DataFrame({"a": np.arange(4.0)}, index=idx))


@pytest.mark.parametrize("order", [[0, 2, 1, 3], [0, 1, 1, 2]])
def test_unordered_or_duplicated_index_raises(order):
    periods = pd.period_range("2000Q1", periods=4, freq="Q")
    idx = pd.PeriodIndex([periods[i] for i in order])
    with pytest.raises(ValueError, match="increasing"):
        VARData.from_frame(pd.DataFrame({"a": np.arange(4.0)}, index=idx))


def test_non_time_index_raises():
    with pytest.raises(TypeError):
        VARData.from_frame(pd.DataFrame({"a": np.arange(4.0)}))


def test_direct_construction_needs_period_index():
    with pytest.raises(TypeError, match="PeriodIndex"):
        VARData(
            raw=pd.DataFrame(
                {"a": np.arange(4.0)}, index=pd.date_range("2000", periods=4, freq="MS")
            ),
            tcodes={"a": 1},
        )


# --------------------------------------------------------------------------- from_frame: tcodes


def test_tcode_forms_are_equivalent():
    frame = _quarterly({"a": _positive(12, 1), "b": _positive(12, 2)})
    by_map = VARData.from_frame(frame, {"b": 1, "a": 5})
    by_seq = VARData.from_frame(frame, [5, 1])
    assert by_map.tcodes == by_seq.tcodes == {"a": 5, "b": 1}
    assert by_map.fingerprint() == by_seq.fingerprint()
    assert VARData.from_frame(frame).tcodes == {"a": 1, "b": 1}
    assert VARData.from_frame(frame, 4).tcodes == {"a": 4, "b": 4}


@pytest.mark.parametrize(
    ("tcodes", "match"),
    [
        ({"a": 1}, "tcodes missing"),
        ([1], "one entry per column"),
        (9, "unknown transformation"),
        ({"a": 1, "b": 0}, "unknown"),
    ],
)
def test_bad_tcodes_raise(tcodes, match):
    frame = _quarterly({"a": _positive(6, 1), "b": _positive(6, 2)})
    with pytest.raises(ValueError, match=match):
        VARData.from_frame(frame, tcodes)


def test_duplicate_or_missing_columns_raise():
    frame = _quarterly({"a": np.arange(1.0, 5.0)})
    with pytest.raises(ValueError, match="duplicate"):
        VARData.from_frame(pd.concat([frame, frame], axis=1))
    with pytest.raises(ValueError, match="no columns"):
        VARData.from_frame(frame[[]])


def test_log_transform_of_non_positive_raw_values_raises():
    frame = _quarterly({"a": np.array([1.0, 2.0, -1.0, 3.0, 4.0])})
    with pytest.raises(ValueError, match="non-positive"):
        VARData.from_frame(frame, 5)


def test_infinite_values_raise():
    frame = _quarterly({"a": np.array([1.0, 2.0, np.inf, 3.0, 4.0])})
    with pytest.raises(ValueError, match="infinite"):
        VARData.from_frame(frame)


# --------------------------------------------------------------------------- missing values


def test_interior_missing_value_raises():
    a = _positive(12, 3)
    a[6] = np.nan
    frame = _quarterly({"a": a, "b": _positive(12, 4)})
    with pytest.raises(
        ValueError, match=r"missing values inside the sample at \['2001Q3'\] in \['a'\]"
    ):
        VARData.from_frame(frame, 1)


def test_interior_missing_value_after_differencing_raises():
    a = _positive(12, 3)
    a[6] = np.nan  # Δlog makes periods 6 and 7 missing
    with pytest.raises(ValueError, match="missing values inside the sample"):
        VARData.from_frame(_quarterly({"a": a}), 5)


def test_transformation_nans_are_dropped_silently(sim_raw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = VARData.from_frame(sim_raw, {"gdp": 6, "prices": 7, "rate": 3})
    assert d.first == sim_raw.index[2]


def _ragged_frame() -> pd.DataFrame:
    a, b = _positive(16, 5), _positive(16, 6)
    a[:3] = np.nan  # leading: a starts late
    b[-2:] = np.nan  # trailing ragged edge: b ends early
    return _quarterly({"a": a, "b": b})


def test_leading_and_trailing_missing_values_are_trimmed_with_warnings():
    frame = _ragged_frame()
    with pytest.warns(UserWarning) as record:
        d = VARData.from_frame(frame, {"a": 1, "b": 5})
    messages = [str(w.message) for w in record]
    assert any("leading" in m and "2000Q4" in m for m in messages)
    assert any("trailing" in m and "2003Q2" in m for m in messages)
    assert d.first == pd.Period("2000Q4", "Q")
    assert d.last == pd.Period("2003Q2", "Q")
    assert np.isfinite(d.values).all()
    np.testing.assert_allclose(d.values[:, 0], frame["a"].loc["2000Q4":"2003Q2"].to_numpy())


def test_derived_samples_do_not_warn_again():
    with pytest.warns(UserWarning):
        d = VARData.from_frame(_ragged_frame(), {"a": 1, "b": 5})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.upto("2002Q1")
        d.tail(5)
        d.select(["b", "a"])
        d.select(["a"])
        d.upto("2003Q1").tail(3).select(["b"])


def test_realized_includes_the_observed_part_of_a_ragged_edge():
    frame = _ragged_frame()
    with pytest.warns(UserWarning):
        d = VARData.from_frame(frame, {"a": 1, "b": 5})
    out = d.realized(d.last, 3)
    np.testing.assert_allclose(out[:2, 0], frame["a"].iloc[-2:].to_numpy())  # a is still observed
    assert np.isnan(out[:, 1]).all()  # b is not
    assert np.isnan(out[2]).all()  # beyond the raw data


# --------------------------------------------------------------------------- sub-samples


@pytest.mark.parametrize("origin", ["1985Q2", pd.Period("2001Q1", "Q"), "1960Q1", "2019Q4"])
def test_upto_shows_nothing_after_origin(sim_data, origin):
    sub = sim_data.upto(origin)
    origin = pd.Period(origin, "Q")
    assert sub.last == origin
    assert sub.first == sim_data.first
    assert sub.raw.index[-1] == origin
    np.testing.assert_array_equal(sub.values, sim_data.sample.loc[:origin].to_numpy())
    assert np.isnan(sub.realized(origin, 4)).all()


def test_upto_is_unaffected_by_future_data(sim_raw, sim_data):
    origin = pd.Period("1990Q4", "Q")
    future_changed = sim_raw.copy()
    future_changed.loc[origin + 1 :] *= 1.5
    other = VARData.from_frame(future_changed, sim_data.tcodes)
    a, b = sim_data.upto(origin), other.upto(origin)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.fingerprint() == b.fingerprint()
    paths = np.full((2, 3, 3), 0.01)
    np.testing.assert_array_equal(a.inverse_paths(paths), b.inverse_paths(paths))
    assert sim_data.fingerprint() != other.fingerprint()


@pytest.mark.parametrize("origin", ["1959Q4", "1950Q1", "2020Q1", "2030Q1"])
def test_upto_outside_the_sample_raises(sim_data, origin):
    with pytest.raises(ValueError, match="outside the sample"):
        sim_data.upto(origin)


def test_tail_keeps_the_last_observations(sim_data):
    window = sim_data.tail(40)
    assert len(window) == 40
    assert window.last == sim_data.last
    assert window.first == sim_data.index[-40]
    np.testing.assert_array_equal(window.values, sim_data.values[-40:])
    paths = np.random.default_rng(0).normal(0, 0.01, (5, 4, 3))
    np.testing.assert_array_equal(window.inverse_paths(paths), sim_data.inverse_paths(paths))


def test_tail_then_upto_is_a_rolling_window(sim_data):
    window = sim_data.upto("2000Q4").tail(20)
    assert (window.first, window.last, len(window)) == (
        pd.Period("1996Q1", "Q"),
        pd.Period("2000Q4", "Q"),
        20,
    )
    shrunk = window.upto("1999Q4")  # the start is kept
    assert shrunk.first == window.first and len(shrunk) == 16


@pytest.mark.parametrize("n_obs", [0, -1, 241])
def test_tail_with_invalid_length_raises(sim_data, n_obs):
    with pytest.raises(ValueError):
        sim_data.tail(n_obs)


def test_select_reorders_and_subsets(sim_data):
    sel = sim_data.select(["rate", "gdp"])
    assert sel.variables == ("rate", "gdp")
    assert sel.tcodes == {"rate": 1, "gdp": 5}
    np.testing.assert_array_equal(sel.values, sim_data.values[:, [2, 0]])
    paths = np.random.default_rng(1).normal(0, 0.01, (6, 2, 3))
    np.testing.assert_allclose(
        sel.inverse_paths(paths[..., [2, 0]]), sim_data.inverse_paths(paths)[..., [2, 0]]
    )
    assert sel.prior_means() == {"rate": 1.0, "gdp": 0.0}


def test_select_unknown_variable_raises(sim_data):
    with pytest.raises(KeyError, match="unknown variables"):
        sim_data.select(["gdp", "unemployment"])


# --------------------------------------------------------------------------- realised values


def test_realized_inside_the_sample(sim_data):
    origin = pd.Period("2000Q1", "Q")
    pos = sim_data.index.get_loc(origin)
    np.testing.assert_array_equal(sim_data.realized(origin, 4), sim_data.values[pos + 1 : pos + 5])
    np.testing.assert_array_equal(
        sim_data.realized("2000Q1", 1), sim_data.values[pos + 1 : pos + 2]
    )


def test_realized_is_nan_beyond_the_end(sim_data):
    out = sim_data.realized(sim_data.last - 2, 5)
    assert out.shape == (5, 3)
    np.testing.assert_array_equal(out[:2], sim_data.values[-2:])
    assert np.isnan(out[2:]).all()
    assert np.isnan(sim_data.realized(sim_data.last, 3)).all()


# --------------------------------------------------------------------------- inversion


@pytest.mark.parametrize("origin", ["1970Q3", "2005Q1", "2017Q4"])
def test_inverse_paths_recovers_true_raw_levels(sim_data, sim_raw, origin):
    origin = pd.Period(origin, "Q")
    H = 8
    sub = sim_data.upto(origin)
    future = sim_data.realized(origin, H)
    level = sub.inverse_paths(future)
    np.testing.assert_allclose(level, sim_raw.loc[origin + 1 : origin + H].to_numpy(), rtol=1e-12)


def test_inverse_paths_all_tcodes():
    T = 40
    raw = _quarterly(
        {f"x{t}": _positive(T, seed=t, scale=10.0 * t) for t in sorted(transforms.TCODES)}, "1990Q1"
    )
    full = VARData.from_frame(raw, sorted(transforms.TCODES))
    origin = full.index[-7]
    level = full.upto(origin).inverse_paths(full.realized(origin, 6))
    np.testing.assert_allclose(level, raw.iloc[-6:].to_numpy(), rtol=1e-11)


def test_inverse_paths_broadcasts_over_draws(sim_data):
    paths = np.random.default_rng(2).normal(0.0, 0.01, (2, 7, 5, 3))  # (chain, draw, H, n)
    out = sim_data.inverse_paths(paths)
    assert out.shape == paths.shape
    np.testing.assert_allclose(out[1, 3], sim_data.inverse_paths(paths[1, 3]))


def test_inverse_paths_checks_the_variable_axis(sim_data):
    with pytest.raises(ValueError, match="number of variables"):
        sim_data.inverse_paths(np.zeros((4, 2)))


def test_raw_history(sim_data, sim_raw):
    np.testing.assert_array_equal(sim_data.raw_history(3), sim_raw.iloc[-3:].to_numpy())
    with pytest.raises(ValueError):
        sim_data.raw_history(len(sim_raw) + 1)


# --------------------------------------------------------------------------- identity


def test_fingerprint_is_stable(sim_data, sim_raw):
    assert sim_data.fingerprint() == sim_data.fingerprint()
    assert (
        VARData.from_frame(sim_raw.copy(), dict(sim_data.tcodes)).fingerprint()
        == sim_data.fingerprint()
    )
    assert len(sim_data.fingerprint()) == 64


def test_fingerprint_changes_with_content(sim_data, sim_raw):
    base = sim_data.fingerprint()
    changed = sim_raw.copy()
    changed.iloc[100, 2] += 1e-9
    assert VARData.from_frame(changed, sim_data.tcodes).fingerprint() != base
    assert VARData.from_frame(sim_raw, {"gdp": 5, "prices": 4, "rate": 1}).fingerprint() != base
    assert sim_data.select(["prices", "gdp", "rate"]).fingerprint() != base
    assert sim_data.upto("2010Q1").fingerprint() != base
    assert sim_data.tail(100).fingerprint() != base


def test_fingerprint_covers_exactly_the_window_and_its_history(sim_data, sim_raw):
    window = sim_data.tail(100)
    pos = sim_raw.index.get_loc(window.first)

    before_history = sim_raw.copy()
    before_history.iloc[pos - 2] *= 1.1  # only feeds a transformed value outside the window
    other = VARData.from_frame(before_history, sim_data.tcodes).tail(100)
    assert other.fingerprint() == window.fingerprint()

    history = sim_raw.copy()
    history.iloc[pos - 1, 0] *= 1.1  # the first Δlog value of the window uses it
    other = VARData.from_frame(history, sim_data.tcodes).tail(100)
    assert other.fingerprint() != window.fingerprint()


def test_prior_means(sim_data, sim_data_stationary):
    assert sim_data.prior_means() == {"gdp": 0.0, "prices": 0.0, "rate": 1.0}
    assert sim_data_stationary.prior_means() == {"gdp": 1.0, "prices": 1.0, "rate": 1.0}


def test_repr_and_basic_properties(sim_data):
    text = repr(sim_data)
    assert "T=240" in text and "1960Q1..2019Q4" in text
    assert sim_data.freq == "Q-DEC"
    assert sim_data.n_vars == 3
    values = sim_data.values
    values[0, 0] = 1e6  # a copy: the dataset is not modified
    assert sim_data.values[0, 0] != 1e6
