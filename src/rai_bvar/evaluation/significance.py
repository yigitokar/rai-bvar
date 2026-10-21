"""Tests of equal predictive ability (SPEC §9.3).

Formal validity: DM-type statistics are asymptotically standard only in the Giacomini-White
(2006) framework, i.e. with rolling or fixed estimation windows. With expanding windows and
nested models they are descriptive (Clark & McCracken 2001); callers label them accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    mean_diff: float
    n: int
    horizon: int
    alternative: str


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    *,
    hln: bool = True,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> DMResult:
    """Diebold-Mariano test on ``d_t = loss_a − loss_b`` (pairs with a NaN are dropped).

    The long-run variance uses the rectangular kernel with ``h − 1`` autocovariances, the choice
    the Harvey-Leybourne-Newbold correction is derived for; if that estimate is not positive it
    falls back to Newey-West (Bartlett) weights. With ``hln`` the statistic is scaled by
    ``√((P + 1 − 2h + h(h−1)/P) / P)`` and compared with Student-t(P−1), else with N(0, 1).
    ``alternative="less"`` tests whether A has the smaller expected loss.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    P = d.size
    if P < 3:
        return DMResult(np.nan, np.nan, float(d.mean()) if P else np.nan, P, horizon, alternative)
    dbar = d.mean()
    lrv = _long_run_variance(d - dbar, horizon - 1, bartlett=False)
    if lrv <= 0:
        lrv = _long_run_variance(d - dbar, horizon - 1, bartlett=True)
    if lrv <= 0:
        return DMResult(np.nan, np.nan, float(dbar), P, horizon, alternative)
    stat = dbar / np.sqrt(lrv / P)
    if hln:
        stat *= np.sqrt(max(P + 1 - 2 * horizon + horizon * (horizon - 1) / P, 0.0) / P)
        dist = stats.t(df=P - 1)
    else:
        dist = stats.norm()
    if alternative == "two-sided":
        p = 2 * dist.sf(abs(stat))
    elif alternative == "less":
        p = dist.cdf(stat)
    elif alternative == "greater":
        p = dist.sf(stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    return DMResult(float(stat), float(p), float(dbar), P, horizon, alternative)


def _long_run_variance(e: np.ndarray, lags: int, *, bartlett: bool) -> float:
    P = e.size
    lrv = e @ e / P
    for k in range(1, min(lags, P - 1) + 1):
        weight = 1.0 - k / (lags + 1) if bartlett else 1.0
        lrv += 2.0 * weight * (e[k:] @ e[:-k]) / P
    return float(lrv)


@dataclass(frozen=True)
class MCSResult:
    pvalues: np.ndarray  # MCS p-value per model (column of the loss matrix)
    elimination_order: tuple[int, ...]

    def included(self, alpha: float) -> np.ndarray:
        """Boolean mask of the models in the ``1 − alpha`` model confidence set."""
        return self.pvalues >= alpha


def circular_block_bootstrap(
    n: int, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Index matrix ``(n_boot, n)`` of circular block-bootstrap resamples."""
    n_blocks = -(-n // block_length)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def model_confidence_set(
    losses: np.ndarray,
    *,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int | np.random.Generator | None = 0,
) -> MCSResult:
    """Model Confidence Set of Hansen, Lunde & Nason (2011) with the ``T_max`` statistic.

    ``losses`` has shape ``(P, M)`` (time × models; rows with any NaN are dropped). At each step
    the model with the largest standardised loss relative to the average of the surviving models
    is eliminated; MCS p-values are the running maximum of the bootstrap p-values.
    """
    L = np.asarray(losses, dtype=np.float64)
    L = L[np.isfinite(L).all(axis=1)]
    P, M = L.shape
    pvalues = np.ones(M)
    if M < 2 or P < 3:
        return MCSResult(pvalues, ())
    rng = np.random.default_rng(seed)
    idx = circular_block_bootstrap(P, block_length or max(1, round(P ** (1 / 3))), n_boot, rng)
    remaining = list(range(M))
    order: list[int] = []
    running = 0.0
    while len(remaining) > 1:
        d = L[:, remaining] - L[:, remaining].mean(axis=1, keepdims=True)
        dbar = d.mean(axis=0)
        boot = d[idx].mean(axis=1) - dbar  # (n_boot, m), centred
        scale = np.sqrt(np.mean(boot**2, axis=0))
        scale = np.where(scale > 0, scale, np.inf)
        t = dbar / scale
        p = float(np.mean((boot / scale).max(axis=1) >= t.max()))
        worst = remaining[int(np.argmax(t))]
        running = max(running, p)
        pvalues[worst] = running
        order.append(worst)
        remaining.remove(worst)
    return MCSResult(pvalues, tuple(order))
