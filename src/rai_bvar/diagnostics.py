"""MCMC convergence diagnostics (Vehtari et al. 2021): rank-normalised split R-hat and bulk ESS.

Self-contained NumPy implementations so the analytic engine's hyperparameter MH chains can be
checked without an ArviZ dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _split(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=np.float64)
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, chains.shape[1] - half :]], axis=0)


def _rank_normalise(chains: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(chains, method="average").reshape(chains.shape)
    return stats.norm.ppf((ranks - 0.375) / (chains.size + 0.25))


def _rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chains.mean(axis=1).var(ddof=1)
    if within <= 0:  # chains that never move are not evidence of convergence
        return np.nan
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def rhat(chains: np.ndarray) -> float:
    """Rank-normalised split R-hat: the max of the bulk and the folded (tail) versions."""
    split = _split(chains)
    if split.shape[1] < 2:
        return np.nan
    bulk = _rhat(_rank_normalise(split))
    folded = _rhat(_rank_normalise(np.abs(split - np.median(split))))
    return np.nan if np.isnan(bulk) or np.isnan(folded) else float(max(bulk, folded))


def ess_bulk(chains: np.ndarray) -> float:
    """Bulk effective sample size of rank-normalised split chains.

    Geyer's initial positive and monotone sequences with the "improved estimate" term and the
    lag bound ``n − 3``, as in Stan and ArviZ (Vehtari et al. 2021). Constant chains give NaN.
    """
    z = _rank_normalise(_split(chains))
    m, n = z.shape
    if n < 5:  # split halves of ≤ 4 draws carry no usable autocorrelation information
        return np.nan
    centred = z - z.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=1)[:, :n] / n
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + (z.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if not var_plus > 0:
        return np.nan

    def rho(t: int) -> float:
        return 1.0 - (mean_var - acov[:, t].mean()) / var_plus

    rho_hat = np.zeros(n)
    rho_even, rho_odd = 1.0, rho(1)
    rho_hat[0], rho_hat[1] = rho_even, rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:  # initial positive sequence
        rho_even, rho_odd = rho(t + 1), rho(t + 2)
        if rho_even + rho_odd >= 0:
            rho_hat[t + 1], rho_hat[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:  # improved estimate
        rho_hat[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:  # initial monotone sequence
        if rho_hat[t + 1] + rho_hat[t + 2] > rho_hat[t - 1] + rho_hat[t]:
            rho_hat[t + 1] = rho_hat[t + 2] = (rho_hat[t - 1] + rho_hat[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * rho_hat[: max_t + 1].sum() + rho_hat[max_t + 1 : max_t + 2].sum()
    total = m * n
    return float(total / max(tau, 1.0 / np.log10(total)))
