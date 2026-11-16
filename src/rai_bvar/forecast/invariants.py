"""Checks every model's ``ForecastResult`` must pass (SPEC §7, contract test layer).

A ``ConditionalForecastResult`` (P3 DESIGN §6.3) has singular conditioned cells, so
:func:`check_conditional_structure` and :func:`conditional_simulation_consistency` replace
:func:`check_structure` and :func:`simulation_consistency` for it.
"""

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


def check_conditional_structure(fr: ForecastResult) -> None:
    """The structure of a ``ConditionalForecastResult`` (P3 DESIGN §6.3), replacing
    :func:`check_structure`: shapes, coordinates and finiteness as there; ``condition``,
    ``conditioned`` (``= isfinite(condition)``, with at least one conditioned and one free cell)
    and a finite ``log_weight (chain, draw)``; the conditioned cells of ``paths`` equal the
    conditions exactly in every draw; the conditioned rows and columns of ``cond_chol`` are
    exactly 0; the free block of each horizon is lower triangular with a positive diagonal; none
    of the dropped accelerator variables is present."""
    from rai_bvar.forecast.conditional import DROPPED, ConditionalForecastResult

    if not isinstance(fr, ConditionalForecastResult):
        raise ContractError(f"expected a ConditionalForecastResult, got {type(fr).__name__}")
    ds = fr.dataset
    n, H = ds.sizes["variable"], ds.sizes["horizon"]
    if ds.sizes["variable_"] != n:
        raise ContractError("cond_chol must be square in the variable dimensions")
    if not np.array_equal(ds["horizon"].values, np.arange(1, H + 1)):
        raise ContractError("horizon coordinate must be 1..H")
    for name in ("paths", "cond_mean", "cond_chol", "paths_level", "log_weight"):
        if name in ds and not np.all(np.isfinite(ds[name].values)):
            raise ContractError(f"{name} has non-finite values")
    if "paths_level" in ds and ds["paths_level"].shape != ds["paths"].shape:
        raise ContractError("paths_level must have the same shape as paths")
    present = [name for name in DROPPED if name in ds]
    if present:
        raise ContractError(f"a conditional forecast must not carry {present}")
    table = ds["condition"].values
    held = ds["conditioned"].values
    if held.dtype != bool or not np.array_equal(held, np.isfinite(table)):
        raise ContractError("conditioned must be the boolean mask of the finite conditions")
    if not held.any() or held.all():
        raise ContractError("a conditional forecast needs a conditioned and a free cell")
    paths = fr.stacked("paths")
    if not np.all(paths[:, held] == table[held]):
        raise ContractError("conditioned paths must equal the conditions exactly")
    chol = fr.stacked("cond_chol")
    for h in range(H):
        cond_h, free_h = np.flatnonzero(held[h]), np.flatnonzero(~held[h])
        L = chol[:, h]
        if np.any(L[:, cond_h, :] != 0) or np.any(L[:, :, cond_h] != 0):
            raise ContractError(
                f"cond_chol must have exactly zero conditioned rows and columns (horizon {h + 1})"
            )
        block = L[:, free_h[:, None], free_h[None, :]]
        if np.any(np.triu(block, k=1) != 0):
            raise ContractError(f"the free block of cond_chol must be lower triangular ({h + 1})")
        if np.any(np.diagonal(block, axis1=-2, axis2=-1) <= 0):
            raise ContractError(f"the free block of cond_chol needs a positive diagonal ({h + 1})")
    if fr.stacked().shape != (fr.n_samples, H, n):
        raise ContractError("stacked paths must have shape (samples, horizon, variables)")


def conditional_simulation_consistency(fr: ForecastResult, max_values: int = 200_000) -> np.ndarray:
    """Per-horizon KS p-values that the standardised free residuals ``(L^c_ff)⁻¹(paths −
    cond_mean)_free`` are iid N(0, 1) (P3 §6.3), ``NaN`` for a horizon without a free variable.
    The horizons of one forecast are one family (D-60, E124): it passes at level ``α`` iff every
    finite p-value is ``≥ α/m``, ``m`` the number of finite p-values."""
    held = fr.dataset["conditioned"].values
    diff = fr.stacked("paths") - fr.stacked("cond_mean")
    chol = fr.stacked("cond_chol")
    out = np.full(fr.horizon, np.nan)
    for h in range(fr.horizon):
        free = np.flatnonzero(~held[h])
        if not free.size:
            continue
        L = chol[:, h][:, free[:, None], free[None, :]]
        z = solve_triangular(L, diff[:, h, free][..., None], lower=True)[..., 0].reshape(-1)
        out[h] = float(stats.kstest(z[:max_values], "norm").pvalue)
    return out
