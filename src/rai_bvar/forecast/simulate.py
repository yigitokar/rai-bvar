"""Joint path simulation for conditionally Gaussian VARs (SPEC §6.1)."""

from __future__ import annotations

import numpy as np

from rai_bvar.data.design import regressors_from_history


def simulate_linear_var(
    coefs: np.ndarray,
    chol: np.ndarray,
    history: np.ndarray,
    horizon: int,
    deterministic: str,
    t_last: int,
    rng: np.random.Generator,
    shock_scale: np.ndarray | None = None,
    shocks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ``y_{T+h} = B'x_{T+h} + Lε`` jointly over horizons, one path per parameter draw.

    Parameters
    ----------
    coefs
        Coefficient draws ``B``, shape ``(S, k, n)``, rows ordered as in :func:`lagged_design`.
    chol
        Lower Cholesky factors of the error covariance, shape ``(S, n, n)``.
    history
        Last ``p`` observations of the estimation sample, shape ``(p, n)``, most recent last.
    t_last
        1-based sample position of the last observation (continues the trend).
    shock_scale
        Optional multiplier of the shocks, broadcastable to ``(S, H)`` (e.g. the COVID residual
        scaling ``s_{T+h}``); the conditional Cholesky factors are scaled accordingly.
    shocks
        Optional pre-drawn standard normal shocks of shape ``(S, H, n)``; ``rng`` is then unused.

    Returns
    -------
    paths, cond_mean, cond_chol
        Shapes ``(S, H, n)``, ``(S, H, n)`` and ``(S, H, n, n)``. ``cond_mean[s, h]`` is the mean
        of ``y_{T+h}`` given draw ``s`` and the simulated path up to ``T+h−1``.
    """
    S, _, n = coefs.shape
    scale = None if shock_scale is None else np.broadcast_to(shock_scale, (S, horizon))
    window = np.broadcast_to(history, (S, *history.shape)).copy()
    paths = np.empty((S, horizon, n))
    means = np.empty((S, horizon, n))
    for h in range(horizon):
        x = regressors_from_history(window, t_last + h + 1, deterministic)
        mean = np.einsum("sk,skn->sn", x, coefs)
        z = rng.standard_normal((S, n)) if shocks is None else shocks[:, h]
        shock = np.einsum("sij,sj->si", chol, z)
        draw = mean + (shock if scale is None else scale[:, h, None] * shock)
        means[:, h], paths[:, h] = mean, draw
        window = np.concatenate([window[:, 1:], draw[:, None, :]], axis=1)
    cond_chol = np.broadcast_to(chol[:, None], (S, horizon, n, n))
    if scale is not None:
        cond_chol = scale[:, :, None, None] * cond_chol
    return paths, means, cond_chol
