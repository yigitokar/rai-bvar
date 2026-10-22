"""``ForecastResult``: the predictive contract shared by every model (SPEC §7).

Each ``(chain, draw)`` slice of ``paths`` is one joint path over horizons and variables,
generated from one parameter draw. ``cond_mean`` and ``cond_chol`` hold the Gaussian moments of
``y_{T+h}`` given that draw and the simulated path up to ``T+h−1``, which makes the log
predictive density computable by Rao-Blackwellisation instead of density estimation.

Optional accelerator variables (DESIGN §7.3, D-20; never contract fields, ignored by
``check_structure`` and every other consumer) may ride along in the dataset; MCMC forecasts
attach them:

* ``cond_A (chain, draw, variable, variable_)``, unit lower triangular, and ``cond_scale (chain,
  draw, horizon, variable)`` with ``cond_chol = A⁻¹ diag(cond_scale)``: the factored density
  ``log N = −½ j log 2π − Σ_{i<j} log ℓ_i − ½‖(A(y − m))_{:j} ⊘ ℓ_{:j}‖²`` of a leading block
  ``{0..j−1}`` (``(A⁻¹)₁₁ = (A₁₁)⁻¹``), O(n²) per draw without a triangular solve;
* ``cond_scale_persistent (chain, draw, horizon, variable)`` (``exp(½ log λ^pers)``) and the
  transitory parameters ``log_p``, ``log1m_p``, ``nu (chain, draw, variable)`` for the opt-in
  transitory Rao-Blackwellisation (``rb_transitory=True``; N03 §2.2, §5.3).
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import gammaln, logsumexp

from rai_bvar.linalg import marginal_chol, mvn_logpdf_chol

DIMS = ("chain", "draw", "horizon", "variable")
CHOL_DIMS = (*DIMS, "variable_")
_LOG_2PI = math.log(2.0 * math.pi)
_TRANSITORY = ("none", "outlier", "t", "outlier_t")


def transitory_log_density(
    z: np.ndarray,
    kind: str,
    *,
    log_p: np.ndarray | None = None,
    log1m_p: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    K: int = 20,
    t_scale: str = "standardized",
) -> np.ndarray:
    """``log f(z)`` of the transitory-integrated standardised shock (N03 §2.2), elementwise:

    =========  ==============================================================================
    none       ``log φ(z)``
    outlier    ``log Σ_k w_k k⁻¹ φ(z/k)``, ``w_1 = 1 − p``, ``w_k = p/(K − 1)`` (``k = 2..K``)
    t          ``log t_{c,ν}(z)``, ``c = ν − 2`` (standardized) or ``ν`` (unit)
    outlier_t  ``log Σ_k w_k k⁻¹ t_{c,ν}(z/k)``
    =========  ==============================================================================

    with ``log t_{c,ν}(z) = lnΓ((ν+1)/2) − lnΓ(ν/2) − ½ log(πc) − ((ν+1)/2) log(1 + z²/c)``.
    ``log_p``, ``log1m_p`` and ``nu`` broadcast against ``z``.
    """
    if kind not in _TRANSITORY:
        raise ValueError(f"kind must be one of {_TRANSITORY}, got {kind!r}")
    z = np.asarray(z, dtype=np.float64)
    if kind in ("t", "outlier_t"):
        nu = np.asarray(nu, dtype=np.float64)
        c = nu - 2.0 if t_scale == "standardized" else nu
        const = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * c)

        def base(x: np.ndarray) -> np.ndarray:
            return const - 0.5 * (nu + 1.0) * np.log1p(x * x / c)

    else:

        def base(x: np.ndarray) -> np.ndarray:
            return -0.5 * (_LOG_2PI + x * x)

    if kind in ("none", "t"):
        return base(z)
    if K < 2:
        raise ValueError(f"the outlier grid needs K >= 2, got {K}")
    log_p = np.asarray(log_p, dtype=np.float64)
    log1m_p = np.asarray(log1m_p, dtype=np.float64)
    terms = []
    for k in range(1, K + 1):
        log_w = log1m_p if k == 1 else log_p - math.log(K - 1)
        if kind == "outlier_t":
            terms.append(log_w - math.log(k) + base(z / k))
        else:
            terms.append(log_w - math.log(k) - 0.5 * (_LOG_2PI + (z / k) ** 2))
    return logsumexp(np.stack(np.broadcast_arrays(*terms)), axis=0)


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
        extra: Mapping[str, tuple[tuple[str, ...], np.ndarray]] | None = None,
    ) -> ForecastResult:
        """Build from arrays shaped ``(draw, H, n)`` or ``(chain, draw, H, n)``.

        ``extra`` maps names to ``(dims, array)`` of optional accelerator variables (module
        docstring); dims use the dataset's dimension names, and an array without the leading
        ``chain`` axis that ``dims`` names gets one. They are never contract fields.
        """

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
        for name, (dims, array) in (extra or {}).items():
            if name in data or name in coords:
                raise ValueError(f"extra variable {name!r} clashes with a contract field")
            dims = tuple(str(d) for d in dims)
            array = np.asarray(array)
            if dims and dims[0] == "chain" and array.ndim == len(dims) - 1:
                array = array[None]
            if array.ndim != len(dims):
                raise ValueError(f"extra variable {name!r}: {array.ndim} axes for dims {dims}")
            data[name] = (dims, array)
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
        rb_transitory: bool = False,
    ):
        """Rao-Blackwellised log predictive density of ``y_{T+h}`` (all or a subset of variables).

        ``log p̂ = logsumexp_s log N(y; m_s, Σ_s) − log S``. With ``return_mcse`` the delta-method
        Monte Carlo standard error of ``log p̂`` is returned as well: ``sd(w)/(√S·mean(w))`` for
        the weights ``w_s = exp(log p_s − max)`` of independent draws, multiplied by ``√(S/ESS)``
        with ``ESS = diagnostics.ess_mean(w)`` on the ``(chain, draw)`` array when
        ``attrs["engine"] == "mcmc"`` (autocorrelated draws; DESIGN §7.3).

        A subset that is a leading block ``{0..j−1}`` of the ordering (in any order: the density
        is invariant to permuting ``y``) uses the exact marginal factor ``L[:j, :j]`` (``L``
        is lower triangular), or the factored ``(cond_A, cond_scale)`` variables when present;
        other subsets factor ``L[idx]`` by QR (``linalg.marginal_chol``).

        ``rb_transitory=True`` (opt-in; DESIGN §7.3, N03 §5.3 item 2, CCMM S.11–S.16) integrates
        the transitory scales of ``T+h`` out analytically: ``log p(y_{1:j}) = Σ_{i<j}[−½ log
        λ^pers_i + log f_i(η_i)]``, ``η_i = e^{−½ log λ^pers_i}(A(y − m))_i``, ``f_i`` from
        :func:`transitory_log_density`. It needs the extras ``cond_A``,
        ``cond_scale_persistent`` and the transitory parameters (MCMC forecasts) and a full or
        leading-block subset; otherwise it warns and falls back to the drawn-scale estimator.
        """
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be in 1..{self.horizon}")
        idx = self._indices(variables)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.size == self._ds.sizes["variable"] and idx.size != y.size:
            y = y[idx]
        if y.size != idx.size:
            raise ValueError("y must have one value per selected variable")
        j = idx.size
        leading = bool(np.array_equal(np.sort(idx), np.arange(j)))
        if leading:  # reorder to the ordering: same joint density, leading-block factor
            order = np.argsort(idx)
            idx, y = idx[order], y[order]
        h = horizon - 1
        logp = None
        if rb_transitory and self._ds.attrs.get("transitory") == "none":
            rb_transitory = False  # no transitory scale: the estimator below is already exact
        if rb_transitory:
            missing = self._rb_missing() if leading else "a leading block of the ordering"
            if missing is None:
                logp = self._rb_logp(y, h, j)
            else:
                warnings.warn(
                    f"rb_transitory=True needs {missing}; using the drawn-scale estimator",
                    UserWarning,
                    stacklevel=2,
                )
        if logp is None and leading and "cond_A" in self._ds and "cond_scale" in self._ds:
            logp = self._factored_logp(y, h, j)
        if logp is None:
            mean = self.stacked("cond_mean")[:, h][:, idx]
            chol = self.stacked("cond_chol")[:, h]
            chol = chol[..., :j, :j] if leading else marginal_chol(chol, idx)
            logp = mvn_logpdf_chol(y, mean, chol)
        lpd = float(logsumexp(logp) - np.log(logp.size))
        if not return_mcse:
            return lpd
        w = np.exp(logp - logp.max())
        mcse = float(w.std(ddof=1) / (np.sqrt(w.size) * w.mean())) if w.size > 1 else np.nan
        if self._ds.attrs.get("engine") == "mcmc" and mcse > 0:
            mcse *= self._ess_factor(w)
        return lpd, mcse

    def _ess_factor(self, w: np.ndarray) -> float:
        """``√(S/ESS)`` of the weights on the ``(chain, draw)`` array; 1 when the ESS is
        undefined (fewer than 10 draws per chain, or constant weights)."""
        from rai_bvar.diagnostics import ess_mean

        chains = w.reshape(self._ds.sizes["chain"], self._ds.sizes["draw"])
        ess = ess_mean(chains)
        return math.sqrt(w.size / ess) if np.isfinite(ess) and ess > 0 else 1.0

    def _factored_logp(self, y: np.ndarray, h: int, j: int) -> np.ndarray:
        """``log N`` of the leading block ``{0..j−1}`` from ``(cond_A, cond_scale)`` (module
        docstring), per stacked draw."""
        A = self.stacked("cond_A")[:, :j, :j]
        ell = self.stacked("cond_scale")[:, h, :j]
        r = np.einsum("sij,sj->si", A, y - self.stacked("cond_mean")[:, h, :j]) / ell
        return -0.5 * j * _LOG_2PI - np.log(ell).sum(axis=-1) - 0.5 * np.einsum("si,si->s", r, r)

    def _rb_missing(self) -> str | None:
        """What the transitory Rao-Blackwellisation lacks, or ``None``."""
        kind = self._ds.attrs.get("transitory")
        if kind not in _TRANSITORY:
            return "the attribute 'transitory' (MCMC forecasts)"
        need = ["cond_A", "cond_scale_persistent"]
        need += {
            "outlier": ["log_p", "log1m_p"],
            "t": ["nu"],
            "outlier_t": ["log_p", "log1m_p", "nu"],
        }.get(kind, [])
        absent = [name for name in need if name not in self._ds]
        return f"the variables {absent}" if absent else None

    def _rb_logp(self, y: np.ndarray, h: int, j: int) -> np.ndarray:
        """``Σ_{i<j}[−½ log λ^pers_i + log f_i(η_i)]`` per stacked draw (N03 §5.3)."""
        attrs = self._ds.attrs
        kind = attrs["transitory"]
        A = self.stacked("cond_A")[:, :j, :j]
        ell = self.stacked("cond_scale_persistent")[:, h, :j]
        eta = np.einsum("sij,sj->si", A, y - self.stacked("cond_mean")[:, h, :j]) / ell
        params = {
            name: self.stacked(name)[:, :j]
            for name in ("log_p", "log1m_p", "nu")
            if name in self._ds
        }
        logf = transitory_log_density(
            eta, kind, K=int(attrs.get("outlier_K", 20)),
            t_scale=str(attrs.get("t_scale", "standardized")), **params,
        )  # fmt: skip
        return (logf - np.log(ell)).sum(axis=-1)

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
