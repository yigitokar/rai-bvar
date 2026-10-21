"""Lagged design matrices: ``x_t = (d_t', y_{t-1}', …, y_{t-p}')'`` (SPEC §4.1).

Deterministic terms come first, then lags in order 1..p, each lag block in variable order.
The trend is the 1-based position in the estimation sample and continues into the forecast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DETERMINISTIC_KINDS = {"n": 0, "c": 1, "ct": 2}


def n_deterministic(kind: str) -> int:
    if kind not in DETERMINISTIC_KINDS:
        raise ValueError(
            f"deterministic must be one of {sorted(DETERMINISTIC_KINDS)}, got {kind!r}"
        )
    return DETERMINISTIC_KINDS[kind]


def deterministic_terms(t: np.ndarray, kind: str) -> np.ndarray:
    """Deterministic regressors for 1-based sample positions ``t``; shape ``(*t.shape, n_d)``."""
    t = np.asarray(t, dtype=np.float64)
    cols = [np.ones_like(t), t][: n_deterministic(kind)]
    return np.stack(cols, axis=-1) if cols else np.zeros((*t.shape, 0))


def lagged_design(
    values: np.ndarray, lags: int, deterministic: str = "c"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Y`` of shape ``(T−p, n)`` and ``X`` of shape ``(T−p, n_d + n·p)``."""
    values = np.asarray(values, dtype=np.float64)
    if lags < 1:
        raise ValueError("lags must be >= 1")
    T = values.shape[0]
    if T <= lags:
        raise ValueError(f"need more than {lags} observations, got {T}")
    rows = np.arange(lags, T)
    blocks = [deterministic_terms(rows + 1.0, deterministic)]
    blocks += [values[rows - lag] for lag in range(1, lags + 1)]
    return values[rows], np.concatenate(blocks, axis=1)


def regressors_from_history(history: np.ndarray, t_next: float, deterministic: str) -> np.ndarray:
    """Regressor vector(s) for the next period from the last ``p`` observations.

    ``history`` has shape ``(..., p, n)`` with the most recent observation last; the result has
    shape ``(..., n_d + n·p)``.
    """
    history = np.asarray(history, dtype=np.float64)
    lead = history.shape[:-2]
    det = np.broadcast_to(
        deterministic_terms(np.asarray(t_next), deterministic),
        (*lead, n_deterministic(deterministic)),
    )
    lagged = history[..., ::-1, :].reshape(*lead, -1)  # lag 1 first
    return np.concatenate([det, lagged], axis=-1)


def regressor_names(variables: Sequence[str], lags: int, deterministic: str) -> list[str]:
    names = ["const", "trend"][: n_deterministic(deterministic)]
    names += [f"{v}.L{lag}" for lag in range(1, lags + 1) for v in variables]
    return names
