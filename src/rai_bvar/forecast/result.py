"""``ForecastResult``: the predictive contract shared by every model (SPEC §7).

Each ``(chain, draw)`` slice of ``paths`` is one joint path over horizons and variables,
generated from one parameter draw. ``cond_mean`` and ``cond_chol`` hold the Gaussian moments of
``y_{T+h}`` given that draw and the simulated path up to ``T+h−1``, which makes the log
predictive density computable by Rao-Blackwellisation instead of density estimation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

from rai_bvar.linalg import marginal_chol, mvn_logpdf_chol

DIMS = ("chain", "draw", "horizon", "variable")
CHOL_DIMS = (*DIMS, "variable_")


class ForecastResult:
    """Container for joint predictive draws and their conditional Gaussian moments."""

    def __init__(self, dataset: xr.Dataset):
        for name, dims in (("paths", DIMS), ("cond_mean", DIMS), ("cond_chol", CHOL_DIMS)):
            if name not in dataset or dataset[name].dims != dims:
                raise ValueError(f"ForecastResult needs {name!r} with dims {dims}")
        self._ds = dataset

    @classmethod
    def from_arrays(
        cls,
        *,
        paths: np.ndarray,
        cond_mean: np.ndarray,
        cond_chol: np.ndarray,
        variables: Sequence[str],
        origin: pd.Period,
        paths_level: np.ndarray | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> ForecastResult:
        """Build from arrays shaped ``(draw, H, n)`` or ``(chain, draw, H, n)``."""

        def with_chain(a: np.ndarray, core: int) -> np.ndarray:
            a = np.asarray(a, dtype=np.float64)
            return a[None] if a.ndim == core + 1 else a

        paths = with_chain(paths, 2)
        H = paths.shape[2]
        variables = [str(v) for v in variables]
        coords = {
            "chain": np.arange(paths.shape[0]),
            "draw": np.arange(paths.shape[1]),
            "horizon": np.arange(1, H + 1),
            "variable": variables,
            "variable_": variables,
            "date": ("horizon", [str(origin + h) for h in range(1, H + 1)]),
        }
        data = {
            "paths": (DIMS, paths),
            "cond_mean": (DIMS, with_chain(cond_mean, 2)),
            "cond_chol": (CHOL_DIMS, with_chain(cond_chol, 3)),
        }
        if paths_level is not None:
            data["paths_level"] = (DIMS, with_chain(paths_level, 2))
        meta = {"origin": str(origin), "freq": origin.freqstr}
        for key, value in (attrs or {}).items():
            meta[key] = (
                value if isinstance(value, str | int | float) else json.dumps(value, default=str)
            )
        return cls(xr.Dataset(data, coords=coords, attrs=meta))

    # ------------------------------------------------------------------ accessors

    @property
    def dataset(self) -> xr.Dataset:
        return self._ds

    @property
    def paths(self) -> xr.DataArray:
        return self._ds["paths"]

    @property
    def paths_level(self) -> xr.DataArray | None:
        return self._ds.get("paths_level")

    @property
    def cond_mean(self) -> xr.DataArray:
        return self._ds["cond_mean"]

    @property
    def cond_chol(self) -> xr.DataArray:
        return self._ds["cond_chol"]

    @property
    def variables(self) -> list[str]:
        return [str(v) for v in self._ds["variable"].values]

    @property
    def horizon(self) -> int:
        return self._ds.sizes["horizon"]

    @property
    def origin(self) -> pd.Period:
        return pd.Period(self._ds.attrs["origin"], freq=self._ds.attrs["freq"])

    @property
    def n_samples(self) -> int:
        return self._ds.sizes["chain"] * self._ds.sizes["draw"]

    def stacked(self, name: str = "paths") -> np.ndarray:
        """Array with ``(chain, draw)`` flattened into one sample axis (v1 shape for paths)."""
        values = self._ds[name].values
        return values.reshape(-1, *values.shape[2:])

    # ------------------------------------------------------------------ densities

    def log_pred_density(
        self,
        y: np.ndarray,
        horizon: int,
        variables: Sequence[str] | None = None,
        *,
        return_mcse: bool = False,
    ):
        """Rao-Blackwellised log predictive density of ``y_{T+h}`` (all or a subset of variables).

        ``log p̂ = logsumexp_s log N(y; m_s, Σ_s) − log S``. With ``return_mcse`` the delta-method
        Monte Carlo standard error of ``log p̂`` (independent draws) is returned as well.
        """
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be in 1..{self.horizon}")
        idx = self._indices(variables)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.size == self._ds.sizes["variable"] and idx.size != y.size:
            y = y[idx]
        if y.size != idx.size:
            raise ValueError("y must have one value per selected variable")
        mean = self.stacked("cond_mean")[:, horizon - 1][:, idx]
        chol = self.stacked("cond_chol")[:, horizon - 1]
        # a reordering of all variables also needs a new factor, not just a subset
        chol = chol if np.array_equal(idx, np.arange(chol.shape[-1])) else marginal_chol(chol, idx)
        logp = mvn_logpdf_chol(y, mean, chol)
        lpd = float(logsumexp(logp) - np.log(logp.size))
        if not return_mcse:
            return lpd
        w = np.exp(logp - logp.max())
        mcse = float(w.std(ddof=1) / (np.sqrt(w.size) * w.mean())) if w.size > 1 else np.nan
        return lpd, mcse

    # ------------------------------------------------------------------ summaries

    def quantiles(self, q: Sequence[float] | float, *, level: bool = False) -> xr.DataArray:
        da = self._source(level)
        return da.quantile(q, dim=("chain", "draw"))

    def mean(self, *, level: bool = False) -> xr.DataArray:
        return self._source(level).mean(dim=("chain", "draw"))

    def median(self, *, level: bool = False) -> xr.DataArray:
        return self._source(level).median(dim=("chain", "draw"))

    def to_dataframe(
        self, q: Sequence[float] = (0.05, 0.5, 0.95), *, level: bool = False
    ) -> pd.DataFrame:
        """Quantiles in wide form: rows ``(date, variable)``, one column per quantile."""
        qs = self.quantiles(list(q), level=level)
        frame = qs.to_dataframe(name="value").reset_index()
        frame["date"] = frame["horizon"].map(
            dict(zip(self._ds["horizon"].values, self._ds["date"].values, strict=True))
        )
        return frame.pivot_table(index=["date", "variable"], columns="quantile", values="value")

    def __repr__(self) -> str:
        return (
            f"ForecastResult(origin={self.origin}, horizon={self.horizon}, "
            f"variables={self.variables}, samples={self.n_samples})"
        )

    # ------------------------------------------------------------------ helpers

    def _indices(self, variables: Sequence[str] | None) -> np.ndarray:
        names = self.variables
        if variables is None:
            return np.arange(len(names))
        missing = [v for v in variables if v not in names]
        if missing:
            raise KeyError(f"unknown variables {missing}")
        return np.array([names.index(v) for v in variables])

    def _source(self, level: bool) -> xr.DataArray:
        if not level:
            return self.paths
        if self.paths_level is None:
            raise ValueError("this forecast has no level paths")
        return self.paths_level
