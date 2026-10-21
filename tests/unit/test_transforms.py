"""FRED-MD/QD transformation codes: forward maps, pathwise inverses and metadata (SPEC §5.2-5.3)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rai_bvar.data import transforms

ALL_TCODES = sorted(transforms.TCODES)
LOG_TCODES = (4, 5, 6)
EXPECTED_HISTORY = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 2}


def _reference_forward(x: np.ndarray, tcode: int) -> np.ndarray:
    """Textbook definitions, written independently of the implementation."""
    out = np.full(x.shape, np.nan)
    lx = np.log(x) if tcode in LOG_TCODES else x
    if tcode in (1, 4):
        out[:] = lx
    elif tcode in (2, 5):
        out[1:] = lx[1:] - lx[:-1]
    elif tcode in (3, 6):
        out[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]
    else:
        out[2:] = x[2:] / x[1:-1] - x[1:-1] / x[:-2]
    return out


def _positive_series(steps: list[float], scale: float) -> np.ndarray:
    return scale * np.exp(np.cumsum(steps))


positive_series = st.builds(
    _positive_series,
    st.lists(st.floats(-0.2, 0.2, allow_nan=False), min_size=5, max_size=40),
    st.floats(0.01, 1e4),
)
real_series = st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=5, max_size=40).map(np.array)


# --------------------------------------------------------------------------- metadata


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_history_length(tcode):
    assert transforms.history_length(tcode) == EXPECTED_HISTORY[tcode]


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_history_length_equals_leading_nans_of_forward(tcode):
    y = transforms.forward(np.linspace(1.0, 2.0, 10) ** 2, tcode)
    lost = transforms.history_length(tcode)
    assert np.isnan(y[:lost]).all()
    assert np.isfinite(y[lost:]).all()


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_prior_mean_own_lag(tcode):
    expected = 1.0 if tcode in (1, 4) else 0.0
    assert transforms.prior_mean_own_lag(tcode) == expected


@pytest.mark.parametrize("bad", [0, 8, -1])
def test_unknown_tcode_raises(bad):
    with pytest.raises(ValueError, match="unknown transformation code"):
        transforms.validate_tcode(bad)
    with pytest.raises(ValueError):
        transforms.history_length(bad)
    with pytest.raises(ValueError):
        transforms.prior_mean_own_lag(bad)
    with pytest.raises(ValueError):
        transforms.forward(np.ones(5), bad)
    with pytest.raises(ValueError):
        transforms.inverse_paths(np.ones(3), np.ones(3), bad)


# --------------------------------------------------------------------------- forward


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_forward_matches_definition(tcode):
    x = np.array([2.0, 2.5, 2.2, 3.1, 3.0, 3.7, 4.4])
    np.testing.assert_allclose(
        transforms.forward(x, tcode),
        _reference_forward(x, tcode),
        rtol=1e-13,
        atol=1e-15,
        equal_nan=True,
    )


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_forward_keeps_length_and_does_not_mutate(tcode):
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    before = x.copy()
    assert transforms.forward(x, tcode).shape == x.shape
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("tcode", LOG_TCODES)
@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_log_codes_reject_non_positive_values(tcode, bad_value):
    x = np.array([1.0, 2.0, bad_value, 3.0, 4.0])
    with pytest.raises(ValueError, match="non-positive"):
        transforms.forward(x, tcode)


@pytest.mark.parametrize("tcode", LOG_TCODES)
def test_log_codes_ignore_missing_values(tcode):
    x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    y = transforms.forward(x, tcode)
    assert np.isnan(y[0]) and np.isnan(y[-1])


@pytest.mark.parametrize("tcode", (1, 2, 3))
def test_level_codes_accept_negative_values(tcode):
    x = np.array([-1.0, 0.0, -2.0, 3.0, -4.0])
    np.testing.assert_allclose(
        transforms.forward(x, tcode), _reference_forward(x, tcode), equal_nan=True
    )


def test_percent_change_code_rejects_zeros():
    with pytest.raises(ValueError, match="zeros"):
        transforms.forward(np.array([1.0, 2.0, 0.0, 3.0, 4.0]), 7)


# --------------------------------------------------------------------------- round trips


def _check_round_trip(x: np.ndarray, tcode: int, split: int) -> None:
    """forward → inverse recovers the raw future, and inverse → forward recovers the path."""
    y = transforms.forward(x, tcode)
    history, future = x[:split], x[split:]
    level = transforms.inverse_paths(y[split:], history, tcode)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(x))))
    np.testing.assert_allclose(level, future, rtol=1e-9, atol=tol)

    # the other direction: arbitrary transformed paths → levels → transformed again
    paths = (
        np.linspace(-0.05, 0.05, 6) if tcode in LOG_TCODES or tcode == 7 else np.linspace(-3, 3, 6)
    )
    level = transforms.inverse_paths(paths, history, tcode)
    again = transforms.forward(np.concatenate([history, level]), tcode)[split:]
    np.testing.assert_allclose(again, paths, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_round_trip_fixed_series(tcode):
    x = 100.0 * np.exp(np.cumsum(np.random.default_rng(tcode).normal(0.01, 0.02, 30)))
    for split in range(max(transforms.history_length(tcode), 1), 25, 6):
        _check_round_trip(x, tcode, split)


@pytest.mark.parametrize("tcode", ALL_TCODES)
@settings(max_examples=40, deadline=None)
@given(x=positive_series, data=st.data())
def test_round_trip_property_positive(tcode, x, data):
    split = data.draw(st.integers(max(transforms.history_length(tcode), 1), len(x) - 1))
    _check_round_trip(x, tcode, split)


@pytest.mark.parametrize("tcode", (1, 2, 3))
@settings(max_examples=40, deadline=None)
@given(x=real_series, data=st.data())
def test_round_trip_property_real_valued(tcode, x, data):
    split = data.draw(st.integers(max(transforms.history_length(tcode), 1), len(x) - 1))
    _check_round_trip(x, tcode, split)


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_inverse_uses_only_the_most_recent_history(tcode):
    x = np.array([5.0, 4.0, 6.0, 7.0, 6.5, 8.0])
    paths = np.array([0.01, -0.02, 0.03])
    need = max(transforms.history_length(tcode), 1)
    short = transforms.inverse_paths(paths, x[-need:], tcode)
    np.testing.assert_allclose(transforms.inverse_paths(paths, x, tcode), short)


@pytest.mark.parametrize("tcode", (1, 4))
def test_inverse_of_level_codes_needs_no_history(tcode):
    paths = np.array([0.5, 1.0, 1.5])
    expected = np.exp(paths) if tcode == 4 else paths
    np.testing.assert_allclose(transforms.inverse_paths(paths, np.empty(0), tcode), expected)


@pytest.mark.parametrize("tcode", [t for t in ALL_TCODES if EXPECTED_HISTORY[t] > 0])
def test_inverse_with_too_little_history_raises(tcode):
    history = np.ones(EXPECTED_HISTORY[tcode] - 1)
    with pytest.raises(ValueError, match="raw observations of history"):
        transforms.inverse_paths(np.zeros(3), history, tcode)


@pytest.mark.parametrize("tcode", ALL_TCODES)
def test_inverse_broadcasts_over_leading_draw_axes(tcode):
    rng = np.random.default_rng(10 + tcode)
    paths = rng.normal(0.0, 0.01, size=(2, 5, 4))  # (chain, draw, H)
    history = np.array([3.0, 3.1, 3.3])
    batched = transforms.inverse_paths(paths, history, tcode)
    assert batched.shape == paths.shape
    for c in range(2):
        for d in range(5):
            np.testing.assert_allclose(
                batched[c, d], transforms.inverse_paths(paths[c, d], history, tcode), rtol=1e-14
            )
    # history with its own leading axis (one history per draw) broadcasts too
    histories = np.stack([history * (1 + 0.1 * d) for d in range(5)])  # (draw, L)
    per_draw = transforms.inverse_paths(paths[0], histories, tcode)
    for d in range(5):
        np.testing.assert_allclose(
            per_draw[d], transforms.inverse_paths(paths[0, d], histories[d], tcode), rtol=1e-14
        )


def test_inverse_is_pathwise_not_applied_to_means():
    """exp of the mean path differs from the mean of exp paths (SPEC §5.2): inversion is pathwise."""
    rng = np.random.default_rng(0)
    paths = rng.normal(0.0, 0.3, size=(4000, 3))
    history = np.array([1.0])
    levels = transforms.inverse_paths(paths, history, 5)
    np.testing.assert_allclose(levels, np.exp(np.cumsum(paths, axis=-1)), rtol=1e-12)
    naive = transforms.inverse_paths(paths.mean(axis=0), history, 5)
    assert np.all(levels.mean(axis=0) > naive)  # Jensen
