"""Benchmark models used to exercise the pipeline end to end (SPEC §13, Phase 0).

Both are conditionally Gaussian linear VARs on the transformed data and share one simulator:

* ``RandomWalk``: ``y_t = y_{t−1} (+ μ) + e_t`` with ``Σ`` estimated from first differences
  (plug-in).
* ``OLSVAR``: least-squares VAR(p). ``uncertainty="flat"`` draws ``(B, Σ)`` from the posterior
  under the Jeffreys prior ``p(B, Σ) ∝ |Σ|^{−(n+1)/2}``: ``Σ ~ IW(S, T−k)`` and
  ``B | Σ ~ MN(B̂, (X'X)⁻¹, Σ)``. ``uncertainty="none"`` plugs in ``B̂`` and ``S/(T−k)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from rai_bvar._version import __version__
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import lagged_design, n_deterministic
from rai_bvar.forecast.result import ForecastResult
from rai_bvar.forecast.simulate import simulate_linear_var
from rai_bvar.linalg import (
    gram_cholesky,
    qr_least_squares,
    sample_inverse_wishart_chol,
    sample_matrix_normal,
)
from rai_bvar.models.base import SeedLike, spec_hash_of

ParamSampler = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class RandomWalk:
    drift: bool = False

    @property
    def name(self) -> str:
        return "RW+drift" if self.drift else "RW"

    @property
    def axes(self) -> dict[str, str]:
        return {"model": self.name}

    def spec(self) -> dict[str, Any]:
        return {"class": "RandomWalk", **asdict(self)}

    def fit(self, data: VARData) -> LinearGaussianFit:
        values = data.values
        if len(values) < 3:
            raise ValueError("RandomWalk needs at least 3 observations")
        dy = np.diff(values, axis=0)
        n = values.shape[1]
        drift = dy.mean(axis=0) if self.drift else np.zeros(n)
        chol = gram_cholesky((dy - dy.mean(axis=0)) / np.sqrt(len(dy) - 1))
        deterministic = "c" if self.drift else "n"
        coefs = np.vstack([drift[None, :], np.eye(n)]) if self.drift else np.eye(n)

        def sampler(size: int, rng: np.random.Generator):
            return np.broadcast_to(coefs, (size, *coefs.shape)), np.broadcast_to(chol, (size, n, n))

        return LinearGaussianFit(data, 1, deterministic, sampler, self.name, self.spec(), "plug-in")


@dataclass(frozen=True)
class OLSVAR:
    lags: int = 1
    deterministic: str = "c"
    uncertainty: Literal["flat", "none"] = "flat"

    def __post_init__(self) -> None:
        if self.uncertainty not in ("flat", "none"):
            raise ValueError("uncertainty must be 'flat' or 'none'")
        n_deterministic(self.deterministic)

    @property
    def name(self) -> str:
        return f"OLS-VAR({self.lags})"

    @property
    def axes(self) -> dict[str, str]:
        return {"model": self.name}

    def spec(self) -> dict[str, Any]:
        return {"class": "OLSVAR", **asdict(self)}

    def fit(self, data: VARData) -> LinearGaussianFit:
        Y, X = lagged_design(data.values, self.lags, self.deterministic)
        T, k = X.shape
        n = Y.shape[1]
        df = T - k
        if df < n + 1:
            raise ValueError(
                f"OLS-VAR({self.lags}) needs at least {k + n + 1} usable observations, got {T}"
            )
        B, R, resid = qr_least_squares(X, Y)
        scale_chol = gram_cholesky(resid)  # Cholesky factor of S = resid'resid

        if self.uncertainty == "none":
            chol = scale_chol / np.sqrt(df)

            def sampler(size: int, rng: np.random.Generator):
                return np.broadcast_to(B, (size, k, n)), np.broadcast_to(chol, (size, n, n))
        else:

            def sampler(size: int, rng: np.random.Generator):
                chols = sample_inverse_wishart_chol(scale_chol, df, size, rng)
                return sample_matrix_normal(B, R, chols, rng), chols

        engine = "analytic" if self.uncertainty == "flat" else "plug-in"
        return LinearGaussianFit(
            data, self.lags, self.deterministic, sampler, self.name, self.spec(), engine
        )


class LinearGaussianFit:
    """A fitted linear VAR whose parameter draws come from ``sampler(size, rng)``."""

    def __init__(
        self,
        data: VARData,
        lags: int,
        deterministic: str,
        sampler: ParamSampler,
        name: str,
        spec: dict[str, Any],
        engine: str,
    ):
        self.data, self.lags, self.deterministic = data, lags, deterministic
        self._sampler, self.name, self.spec, self.engine = sampler, name, spec, engine

    def forecast(self, horizon: int, draws: int = 1000, seed: SeedLike = None) -> ForecastResult:
        if horizon < 1 or draws < 1:
            raise ValueError("horizon and draws must be positive")
        rng = np.random.default_rng(seed)
        coefs, chol = self._sampler(draws, rng)
        paths, mean, cond_chol = simulate_linear_var(
            coefs,
            chol,
            self.data.values[-self.lags :],
            horizon,
            self.deterministic,
            len(self.data),
            rng,
        )
        return ForecastResult.from_arrays(
            paths=paths,
            cond_mean=mean,
            cond_chol=cond_chol,
            paths_level=self.data.inverse_paths(paths),
            variables=self.data.variables,
            origin=self.data.last,
            attrs={
                "model": self.name,
                "engine": self.engine,
                "spec": self.spec,
                "spec_hash": spec_hash_of(self.spec),
                "tcodes": self.data.tcodes,
                "rai_bvar_version": __version__,
            },
        )
