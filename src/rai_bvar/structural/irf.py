"""Recursive impulse responses and forecast-error variance decompositions (SPEC §6.3).

A linear VAR ``y_t = B'x_t + u_t`` with lag matrices ``A_1, …, A_p`` has the moving-average
representation ``y_t = … + Σ_h Φ_h u_{t−h}`` with ``Φ_0 = I`` and
``Φ_h = Σ_{l=1}^{min(h,p)} A_l Φ_{h−l}``. Recursive (Cholesky) identification writes the
reduced-form errors as ``u_t = L ε_t`` with ``LL' = Σ`` and orthogonal unit-variance structural
shocks ``ε_t``, so the response of ``y_{t+h}`` to a one-standard-deviation shock is
``Θ_h = Φ_h L``. ``L`` is lower triangular: shock ``j`` moves only variables ``j, j+1, …`` on
impact.

**Recursive identification depends on the variable ordering (SPEC R8).** Reordering the variables
changes ``L``, and therefore the impulse responses and the variance decomposition, even though
the reduced form is unchanged. The ordering used is the column order of the data; every
:class:`IRFResult` records it (the ``response``/``shock`` coordinates and the ``ordering``
attribute). Use the ordering-sensitivity check (SPEC §6.6) before reading much into the
responses to shocks of variables whose position is not pinned down by economic reasoning.

In the recursive structural form ``L = A⁻¹Λ^{1/2}`` with ``A`` unit lower triangular, so the
unit-impact normalisation ``Φ_h L diag(L)⁻¹ = Φ_h A⁻¹`` does not depend on the volatilities:
responses to unit structural shocks are time-invariant in shape, while one-standard-deviation
responses depend on the date at which the impact matrix is evaluated (stochastic volatility,
COVID residual scaling). Pass a date-specific ``chol``, or a ``scale`` for a common scalar
volatility factor, to obtain them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

DIMS = ("chain", "draw", "horizon", "response", "shock")
SHOCKS = ("sd", "unit")


def _lag_matrices(coefs: np.ndarray, lags: int, n_det: int) -> np.ndarray:
    """Lag matrices ``A[..., l−1, i, j] = coefs[..., n_det + (l−1)·n + j, i]``, ``(..., p, n, n)``."""
    if coefs.ndim < 2:
        raise ValueError(f"coefs must have shape (..., k, n), got {coefs.shape}")
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if n_det < 0:
        raise ValueError("n_det must be >= 0")
    k, n = coefs.shape[-2:]
    if k != n_det + n * lags:
        raise ValueError(
            f"coefs has {k} rows, expected n_det + n·lags = {n_det} + {n}·{lags} = "
            f"{n_det + n * lags}"
        )
    blocks = coefs[..., n_det:, :].reshape(*coefs.shape[:-2], lags, n, n)  # [..., l, j, i]
    return np.swapaxes(blocks, -1, -2)


def ma_coefficients(coefs: np.ndarray, lags: int, n_det: int, horizon: int) -> np.ndarray:
    """Moving-average coefficients ``Φ_0, …, Φ_H`` of a linear VAR.

    Parameters
    ----------
    coefs
        Coefficients ``B`` of shape ``(..., k, n)`` with ``k = n_det + n·lags``, rows ordered as
        in :func:`rai_bvar.data.design.lagged_design`. Deterministic rows are ignored.
    lags, n_det
        Lag order ``p`` and number of deterministic regressors.
    horizon
        Largest horizon ``H``.

    Returns
    -------
    np.ndarray
        Shape ``(..., H+1, n, n)``; ``[..., h, i, j]`` is the response of ``y_{i,t+h}`` to
        ``u_{j,t}``.
    """
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    A = _lag_matrices(np.asarray(coefs, dtype=np.float64), lags, n_det)
    *lead, p, n, _ = A.shape
    phi = np.zeros((*lead, horizon + 1, n, n))
    phi[..., 0, :, :] = np.eye(n)
    for h in range(1, horizon + 1):
        m = min(h, p)
        previous = phi[..., h - m : h, :, :][..., ::-1, :, :]  # Φ_{h−1}, …, Φ_{h−m}
        phi[..., h, :, :] = (A[..., :m, :, :] @ previous).sum(axis=-3)
    return phi


def recursive_irf(
    coefs: np.ndarray,
    chol: np.ndarray,
    lags: int,
    n_det: int,
    horizon: int,
    shock: Literal["sd", "unit"] = "sd",
    scale: float | np.ndarray | None = None,
) -> np.ndarray:
    """Impulse responses to recursively identified structural shocks.

    Parameters
    ----------
    coefs
        Coefficients ``B``, shape ``(..., k, n)`` (see :func:`ma_coefficients`).
    chol
        Lower Cholesky factors ``L`` of the error covariance, shape ``(..., n, n)``; leading axes
        broadcast against those of ``coefs``.
    shock
        ``"sd"``: one-standard-deviation shocks, ``Θ_h = Φ_h L``. ``"unit"``: each shock has a
        unit impact on its own variable, ``Θ_h = Φ_h L diag(L)⁻¹``.
    scale
        Optional scalar or array multiplying the whole impact matrix ``L`` (e.g. a COVID residual
        scale ``s_t``). It broadcasts against the leading (draw) axes of ``chol``, so an array
        with extra leading axes (e.g. dates) adds them to the output. Unit-shock responses are
        invariant to it.

    Returns
    -------
    np.ndarray
        Shape ``(..., H+1, n, n)`` indexed ``[..., horizon, response, shock]``.
    """
    if shock not in SHOCKS:
        raise ValueError(f"shock must be one of {SHOCKS}, got {shock!r}")
    phi = ma_coefficients(coefs, lags, n_det, horizon)
    n = phi.shape[-1]
    impact = np.asarray(chol, dtype=np.float64)
    if impact.ndim < 2 or impact.shape[-2:] != (n, n):
        raise ValueError(f"chol must have shape (..., {n}, {n}), got {impact.shape}")
    if scale is not None:
        impact = impact * np.asarray(scale, dtype=np.float64)[..., None, None]
    if shock == "unit":
        impact = impact / np.diagonal(impact, axis1=-2, axis2=-1)[..., None, :]
    return phi @ impact[..., None, :, :]


def fevd(irf: np.ndarray) -> np.ndarray:
    """Forecast-error variance decomposition from one-standard-deviation responses.

    ``irf`` has shape ``(..., H+1, n, n)`` indexed ``[..., h, response, shock]`` and must start
    at ``h = 0`` with orthogonal unit-variance shocks (``shock="sd"``). Entry
    ``[..., h, i, j]`` of the result is the share of the ``(h+1)``-step forecast-error variance
    of variable ``i`` due to shock ``j``, ``Σ_{s≤h} Θ_s[i, j]² / Σ_{s≤h} Σ_m Θ_s[i, m]²``; rows
    sum to one.
    """
    irf = np.asarray(irf, dtype=np.float64)
    if irf.ndim < 3 or irf.shape[-1] != irf.shape[-2]:
        raise ValueError(f"irf must have shape (..., H+1, n, n), got {irf.shape}")
    contributions = np.cumsum(irf**2, axis=-3)
    return contributions / contributions.sum(axis=-1, keepdims=True)


class IRFResult:
    """Posterior draws of impulse responses with dims ``(chain, draw, horizon, response, shock)``.

    Each ``(chain, draw)`` slice holds the responses implied by one parameter draw; the horizon
    coordinate runs from 0 (impact) to ``H``.
    """

    def __init__(self, data: xr.DataArray):
        if data.dims != DIMS:
            raise ValueError(f"IRFResult needs a DataArray with dims {DIMS}, got {data.dims}")
        self._da = data

    @classmethod
    def from_arrays(
        cls,
        irf: np.ndarray,
        variables: Sequence[str],
        *,
        shock: Literal["sd", "unit"] = "sd",
        attrs: Mapping[str, Any] | None = None,
    ) -> IRFResult:
        """Build from responses shaped ``(draw, H+1, n, n)`` or ``(chain, draw, H+1, n, n)``."""
        if shock not in SHOCKS:
            raise ValueError(f"shock must be one of {SHOCKS}, got {shock!r}")
        irf = np.asarray(irf, dtype=np.float64)
        irf = irf[None] if irf.ndim == 4 else irf
        variables = [str(v) for v in variables]
        n = len(variables)
        if irf.ndim != 5 or irf.shape[-2:] != (n, n):
            raise ValueError(
                f"irf must have shape ([chain,] draw, H+1, {n}, {n}) for {n} variables, "
                f"got {irf.shape}"
            )
        coords = {
            "chain": np.arange(irf.shape[0]),
            "draw": np.arange(irf.shape[1]),
            "horizon": np.arange(irf.shape[2]),
            "response": variables,
            "shock": variables,
        }
        meta: dict[str, Any] = {
            "identification": "recursive",
            "shock_size": shock,
            "ordering": json.dumps(variables),
        }
        for key, value in (attrs or {}).items():
            meta[key] = (
                value if isinstance(value, str | int | float) else json.dumps(value, default=str)
            )
        return cls(xr.DataArray(irf, dims=DIMS, coords=coords, name="irf", attrs=meta))

    # ------------------------------------------------------------------ accessors

    @property
    def data(self) -> xr.DataArray:
        return self._da

    @property
    def variables(self) -> list[str]:
        return [str(v) for v in self._da["response"].values]

    @property
    def horizon(self) -> int:
        """Largest horizon ``H`` (the horizon coordinate runs 0..H)."""
        return int(self._da["horizon"].values[-1])

    # ------------------------------------------------------------------ summaries

    def quantiles(self, q: Sequence[float] | float) -> xr.DataArray:
        return self._da.quantile(q, dim=("chain", "draw"))

    def median(self) -> xr.DataArray:
        return self._da.median(dim=("chain", "draw"))

    def mean(self) -> xr.DataArray:
        return self._da.mean(dim=("chain", "draw"))

    def fevd(self) -> xr.DataArray:
        """FEVD shares per draw, same dims and coordinates as :attr:`data`."""
        if self._da.attrs.get("shock_size", "sd") != "sd":
            raise ValueError("FEVD needs one-standard-deviation responses (shock='sd')")
        if not np.array_equal(self._da["horizon"].values, np.arange(self._da.sizes["horizon"])):
            raise ValueError("FEVD needs every horizon from 0 (impact) onwards")
        return self._da.copy(data=fevd(self._da.values)).rename("fevd")

    def to_dataframe(self, q: Sequence[float] = (0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Quantiles in wide form: rows ``(horizon, response, shock)``, one column per quantile."""
        return self.quantiles(list(q)).to_series().unstack("quantile")

    def __repr__(self) -> str:
        sizes = self._da.sizes
        return (
            f"IRFResult(horizon={self.horizon}, variables={self.variables}, "
            f"shock_size={self._da.attrs.get('shock_size', 'sd')!r}, "
            f"samples={sizes['chain'] * sizes['draw']})"
        )
