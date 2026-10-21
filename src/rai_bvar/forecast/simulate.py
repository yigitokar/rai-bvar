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

    Returns
    -------
    paths, cond_mean, cond_chol
        Shapes ``(S, H, n)``, ``(S, H, n)`` and ``(S, H, n, n)``. ``cond_mean[s, h]`` is the mean
        of ``y_{T+h}`` given draw ``s`` and the simulated path up to ``T+h−1``.
    """
    S, _, n = coefs.shape
    window = np.broadcast_to(history, (S, *history.shape)).copy()
    paths = np.empty((S, horizon, n))
    means = np.empty((S, horizon, n))
    for h in range(horizon):
        x = regressors_from_history(window, t_last + h + 1, deterministic)
        mean = np.einsum("sk,skn->sn", x, coefs)
        draw = mean + np.einsum("sij,sj->si", chol, rng.standard_normal((S, n)))
        means[:, h], paths[:, h] = mean, draw
        window = np.concatenate([window[:, 1:], draw[:, None, :]], axis=1)
    return paths, means, np.broadcast_to(chol[:, None], (S, horizon, n, n))
