"""Lagged design matrices and forecast regressors (SPEC §4.1, §5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from rai_bvar.data.design import (
    deterministic_terms,
    lagged_design,
    n_deterministic,
    regressor_names,
    regressors_from_history,
)

DETERMINISTIC = ["n", "c", "ct"]


def _coded_values(T: int, n: int) -> np.ndarray:
    """values[t, j] = 100·(t+1) + j, so every entry identifies its period and variable."""
    return 100.0 * np.arange(1, T + 1)[:, None] + np.arange(n)[None, :]


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize("lags", [1, 2, 4])
def test_column_layout(lags, deterministic):
    T, n = 12, 3
    values = _coded_values(T, n)
    Y, X = lagged_design(values, lags, deterministic)
    n_d = n_deterministic(deterministic)
    assert Y.shape == (T - lags, n)
    assert X.shape == (T - lags, n_d + n * lags)
    np.testing.assert_array_equal(Y, values[lags:])
    for row in range(T - lags):
        t = row + lags  # 0-based period of the dependent variable
        expected = [1.0, t + 1.0][:n_d]  # trend = 1-based position in the sample
        for lag in range(1, lags + 1):
            expected += [100.0 * (t - lag + 1) + j for j in range(n)]
        np.testing.assert_array_equal(X[row], expected)


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_names_follow_the_layout(deterministic):
    names = regressor_names(["gdp", "rate"], 2, deterministic)
    det = {"n": [], "c": ["const"], "ct": ["const", "trend"]}[deterministic]
    assert names == [*det, "gdp.L1", "rate.L1", "gdp.L2", "rate.L2"]
    _, X = lagged_design(_coded_values(6, 2), 2, deterministic)
    assert len(names) == X.shape[1]


def test_trend_values():
    _, X = lagged_design(np.zeros((10, 1)), 3, "ct")
    np.testing.assert_array_equal(X[:, 0], 1.0)
    np.testing.assert_array_equal(X[:, 1], np.arange(4.0, 11.0))
    np.testing.assert_array_equal(
        deterministic_terms(np.array([5, 6]), "ct"), [[1.0, 5.0], [1.0, 6.0]]
    )
    assert deterministic_terms(np.array([5, 6]), "n").shape == (2, 0)


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize("lags", [1, 3])
def test_regressors_from_history_match_design_rows(lags, deterministic):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((15, 2))
    _, X = lagged_design(values, lags, deterministic)
    for row in range(X.shape[0]):
        t = row + lags
        x = regressors_from_history(values[t - lags : t], t + 1, deterministic)
        np.testing.assert_array_equal(x, X[row])


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_regressors_continue_the_trend_into_the_forecast(deterministic):
    """The regressor for T+1 equals the last design row of the sample extended by one period."""
    rng = np.random.default_rng(1)
    values = rng.standard_normal((20, 3))
    lags = 2
    extended = np.vstack([values, np.full((1, 3), np.nan)])
    _, X_ext = lagged_design(extended, lags, deterministic)
    x_next = regressors_from_history(values[-lags:], len(values) + 1, deterministic)
    np.testing.assert_array_equal(x_next, X_ext[-1])
    if deterministic == "ct":
        assert x_next[1] == 21.0


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_regressors_from_history_batched(deterministic):
    rng = np.random.default_rng(2)
    history = rng.standard_normal((4, 5, 2, 3))  # (chain, draw, p, n)
    out = regressors_from_history(history, 31, deterministic)
    assert out.shape == (4, 5, n_deterministic(deterministic) + 6)
    np.testing.assert_array_equal(
        out[2, 3], regressors_from_history(history[2, 3], 31, deterministic)
    )


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="lags"):
        lagged_design(np.zeros((5, 2)), 0)
    with pytest.raises(ValueError, match="more than"):
        lagged_design(np.zeros((3, 2)), 3)
    with pytest.raises(ValueError, match="deterministic"):
        lagged_design(np.zeros((5, 2)), 1, "trend")
    with pytest.raises(ValueError, match="deterministic"):
        n_deterministic("ctt")
