"""Checks every model's ``ForecastResult`` must pass (SPEC §7, contract test layer)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular

from rai_bvar.forecast.result import ForecastResult


class ContractError(AssertionError):
    """A forecast violates the predictive contract."""


def check_structure(fr: ForecastResult) -> None:
    """Shapes, finiteness and triangular structure of the conditional Cholesky factors."""
    ds = fr.dataset
    n, H = ds.sizes["variable"], ds.sizes["horizon"]
    if ds.sizes["variable_"] != n:
        raise ContractError("cond_chol must be square in the variable dimensions")
    if not np.array_equal(ds["horizon"].values, np.arange(1, H + 1)):
        raise ContractError("horizon coordinate must be 1..H")
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        if name in ds and not np.all(np.isfinite(ds[name].values)):
            raise ContractError(f"{name} has non-finite values")
    if "paths_level" in ds and ds["paths_level"].shape != ds["paths"].shape:
        raise ContractError("paths_level must have the same shape as paths")
    chol = fr.stacked("cond_chol")
    if np.any(np.triu(chol, k=1) != 0):
        raise ContractError("cond_chol must be lower triangular")
    if np.any(np.diagonal(chol, axis1=-2, axis2=-1) <= 0):
        raise ContractError("cond_chol must have a positive diagonal")
    if fr.stacked().shape != (fr.n_samples, H, n):
        raise ContractError("stacked paths must have shape (samples, horizon, variables)")


def standardized_residuals(fr: ForecastResult) -> np.ndarray:
    """``L⁻¹(paths − cond_mean)`` for every sample and horizon, shape ``(samples, H, n)``."""
    diff = fr.stacked("paths") - fr.stacked("cond_mean")
    return solve_triangular(fr.stacked("cond_chol"), diff[..., None], lower=True)[..., 0]


def simulation_consistency(fr: ForecastResult, max_values: int = 200_000) -> float:
    """KS p-value that the standardised residuals are iid N(0, 1) (SPEC §7, invariant 1)."""
    z = standardized_residuals(fr).reshape(-1)
    return float(stats.kstest(z[:max_values], "norm").pvalue)
