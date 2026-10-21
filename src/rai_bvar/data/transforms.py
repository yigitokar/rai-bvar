"""FRED-MD/QD transformation codes (McCracken & Ng 2016) with pathwise inverses.

Inverse transforms are applied to whole simulated paths, never to quantiles or means:
nonlinear transforms do not commute with expectations (SPEC §5.2).
"""

from __future__ import annotations

import numpy as np

TCODES: dict[int, str] = {
    1: "level",
    2: "first difference",
    3: "second difference",
    4: "log level",
    5: "first difference of log",
    6: "second difference of log",
    7: "first difference of percent change",
}

# Raw observations needed before the first transformed value (and to invert a path).
_HISTORY = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 2}
_LOGGED = frozenset({4, 5, 6})

# Series whose transformed values are persistent levels: Minnesota prior mean 1 on the own first lag.
PERSISTENT_TCODES = frozenset({1, 4})


def validate_tcode(tcode: int) -> int:
    if tcode not in TCODES:
        raise ValueError(f"unknown transformation code {tcode!r}; expected one of {sorted(TCODES)}")
    return int(tcode)


def history_length(tcode: int) -> int:
    """Number of raw observations lost at the start, and needed to invert a path."""
    return _HISTORY[validate_tcode(tcode)]


def prior_mean_own_lag(tcode: int) -> float:
    """Transform-aware Minnesota prior mean for the own first lag (SPEC §5.3)."""
    return 1.0 if validate_tcode(tcode) in PERSISTENT_TCODES else 0.0


def forward(x: np.ndarray, tcode: int) -> np.ndarray:
    """Apply a transformation to a raw series. The output has the input's length, NaN-padded."""
    tcode = validate_tcode(tcode)
    x = np.asarray(x, dtype=np.float64)
    if tcode in _LOGGED:
        finite = x[np.isfinite(x)]
        if np.any(finite <= 0):
            raise ValueError(f"tcode {tcode} takes logs but the series has non-positive values")
        x = np.log(x)
    if tcode == 7 and np.any(x[np.isfinite(x)] == 0):
        raise ValueError("tcode 7 divides by the previous value but the series has zeros")
    if tcode in (1, 4):
        return x.copy()
    if tcode in (2, 5):
        return _diff(x, 1)
    if tcode in (3, 6):
        return _diff(_diff(x, 1), 1)
    # tcode 7: Δ(x_t / x_{t−1} − 1)
    pct = np.full_like(x, np.nan)
    pct[1:] = x[1:] / x[:-1] - 1.0
    return _diff(pct, 1)


def inverse_paths(paths: np.ndarray, history: np.ndarray, tcode: int) -> np.ndarray:
    """Map transformed future paths back to raw levels.

    Parameters
    ----------
    paths
        Transformed values for periods ``T+1, …, T+H`` along the last axis, shape ``(..., H)``.
    history
        Raw values ending at ``T`` along the last axis, shape ``(..., L)`` broadcastable
        against ``paths`` with ``L >= history_length(tcode)``.
    """
    tcode = validate_tcode(tcode)
    paths = np.asarray(paths, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    need = _HISTORY[tcode]
    if need and history.shape[-1] < need:
        raise ValueError(f"tcode {tcode} needs {need} raw observations of history")
    if tcode in _LOGGED and need:
        history = np.log(history)

    if tcode in (1, 4):
        level = paths.copy()
    elif tcode in (2, 5):
        level = history[..., -1:] + np.cumsum(paths, axis=-1)
    elif tcode in (3, 6):
        last_diff = history[..., -1:] - history[..., -2:-1]
        diffs = last_diff + np.cumsum(paths, axis=-1)
        level = history[..., -1:] + np.cumsum(diffs, axis=-1)
    else:  # tcode 7
        last_pct = history[..., -1:] / history[..., -2:-1] - 1.0
        pct = last_pct + np.cumsum(paths, axis=-1)
        return history[..., -1:] * np.cumprod(1.0 + pct, axis=-1)
    return np.exp(level) if tcode in _LOGGED else level


def _diff(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out
