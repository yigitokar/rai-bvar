"""Scores for one forecast against realised outcomes (SPEC §9.2).

Lower is better for every score except the log predictive density (LPD), where higher is
better. CRPS, threshold-weighted CRPS, energy and variogram scores come from ``scoringrules``;
the quantile-weighted CRPS (Gneiting & Ranjan 2011) is implemented here. The LPD is
``ForecastResult.log_pred_density``: by default the SPEC §7 estimator, with
``lpd_method="integrated"`` the opt-in path-integrated one (DESIGN §7.3, D-34), reported next to
the default estimator on the same forecast (:func:`score_forecast`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import scoringrules as sr
import xarray as xr

from rai_bvar.forecast.result import LPD_METHODS, ForecastResult

_FAIR_CRPS = "pwm"  # equals the fair (unbiased) estimator exactly; sorting-based

QW_WEIGHTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "center": lambda a: a * (1.0 - a),
    "left": lambda a: (1.0 - a) ** 2,
    "right": lambda a: a**2,
}


def quantile_weighted_crps(
    draws: np.ndarray,
    y: np.ndarray,
    weight: str | Callable[[np.ndarray], np.ndarray] | None = None,
    n_levels: int = 99,
) -> np.ndarray:
    """``2 ∫₀¹ QS_α(F⁻¹(α), y) w(α) dα`` by the midpoint rule over ``n_levels`` quantile levels.

    ``QS_α(q, y) = (1{y ≤ q} − α)(q − y)`` is the quantile score; with ``w ≡ 1`` this is the
    CRPS. ``draws`` has the ensemble on its last axis; ``y`` matches the leading axes.
    """
    alpha = (np.arange(n_levels) + 0.5) / n_levels
    q = np.moveaxis(np.quantile(draws, alpha, axis=-1), 0, -1)
    y = np.asarray(y, dtype=np.float64)[..., None]
    qs = ((y <= q) - alpha) * (q - y)
    w = (
        np.ones_like(alpha)
        if weight is None
        else (QW_WEIGHTS[weight] if isinstance(weight, str) else weight)(alpha)
    )
    return 2.0 * np.mean(qs * w, axis=-1)


def score_forecast(
    fr: ForecastResult,
    realized: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    multivariate_max_samples: int = 1000,
    qw_levels: int = 99,
    tw_bounds: Mapping[str, tuple[float, float]] | None = None,
    lpd_method: Literal["conditioned", "integrated"] = "conditioned",
) -> xr.Dataset:
    """All scores of one forecast. ``realized`` has shape ``(H, n)``; NaN marks unobserved values.

    Energy and variogram scores cost ``O(S²)`` per horizon, so they use a random subsample of at
    most ``multivariate_max_samples`` draws. ``tw_bounds`` maps variables to the interval
    ``(a, b)`` that the threshold-weighted CRPS focuses on (e.g. a recession threshold for
    output growth); without it no ``twcrps`` is computed.

    ``lpd_method`` goes to every ``log_pred_density`` call (joint with its MCSE, and per
    variable). With the default the output is the Phase 1 layout. With ``"integrated"`` (DESIGN
    §7.3, D-34) ``lpd``, ``lpd_mcse`` and ``lpd_marginal`` hold the path-integrated estimator,
    and the dataset also has ``lpd_marginal_mcse`` (its per-variable MCSE) and the default
    estimator on the same forecast, ``lpd_conditioned``, ``lpd_mcse_conditioned`` and
    ``lpd_marginal_conditioned``, with ``attrs["lpd_method"] = "integrated"``. At ``h = 1`` the
    two coincide bitwise; at ``h > 1`` a forecast without ``lag_coefs`` raises ``ValueError``.

    A conditional forecast (``ConditionalForecastResult``, or any forecast whose attrs carry
    ``"conditional"``) is a scenario, not a forecast to score (P3 DESIGN §6.3, D-50): it raises
    ``TypeError``, and so does the backtest, which scores through this function.
    """
    from rai_bvar.forecast.conditional import ConditionalForecastResult

    attrs = getattr(getattr(fr, "dataset", None), "attrs", None) or {}
    if isinstance(fr, ConditionalForecastResult) or "conditional" in attrs:
        raise TypeError(
            "a conditional forecast is a per-draw scenario under hard conditions (P3 DESIGN "
            "§6.3, D-50), not a forecast of the data: its equal-weight mixture is not "
            "p(y | data, conditions), so it is not scored; score the unconditional forecast"
        )
    if lpd_method not in LPD_METHODS:
        raise ValueError(f"lpd_method must be one of {LPD_METHODS}, got {lpd_method!r}")
    realized = np.asarray(realized, dtype=np.float64)
    H, n = fr.horizon, len(fr.variables)
    if realized.shape != (H, n):
        raise ValueError(f"realized must have shape {(H, n)}, got {realized.shape}")
    integrated = lpd_method == "integrated"
    rng = np.random.default_rng(rng)
    draws = fr.stacked()  # (S, H, n)
    ens = np.moveaxis(draws, 0, -1)  # (H, n, S)
    observed = np.isfinite(realized)
    complete = observed.all(axis=1)

    lpd = np.full(H, np.nan)
    lpd_mcse = np.full(H, np.nan)
    lpd_marginal = np.full((H, n), np.nan)
    # integrated only: its per-variable MCSE and the default estimator on the same forecast
    marginal_mcse = np.full((H, n), np.nan)
    cond, cond_mcse, cond_marginal = np.full(H, np.nan), np.full(H, np.nan), np.full((H, n), np.nan)
    for h in range(H):
        if complete[h]:
            lpd[h], lpd_mcse[h] = fr.log_pred_density(
                realized[h], h + 1, return_mcse=True, lpd_method=lpd_method
            )
            if integrated:
                cond[h], cond_mcse[h] = fr.log_pred_density(realized[h], h + 1, return_mcse=True)
        for j, var in enumerate(fr.variables):
            if not observed[h, j]:
                continue
            y = realized[h, j : j + 1]
            if integrated:
                lpd_marginal[h, j], marginal_mcse[h, j] = fr.log_pred_density(
                    y, h + 1, [var], return_mcse=True, lpd_method=lpd_method
                )
                cond_marginal[h, j] = fr.log_pred_density(y, h + 1, [var])
            else:
                lpd_marginal[h, j] = fr.log_pred_density(y, h + 1, [var])

    obs = np.where(observed, realized, 0.0)  # placeholders; masked again below
    sub = (
        draws
        if draws.shape[0] <= multivariate_max_samples
        else draws[rng.choice(draws.shape[0], multivariate_max_samples, replace=False)]
    )
    sub = np.moveaxis(sub, 0, 1)  # (H, S', n)
    # one horizon at a time: the pairwise term needs an (S', S', n) temporary
    es = np.array([sr.es_ensemble(obs[h], sub[h], estimator="fair") for h in range(H)])
    energy = np.where(complete, es, np.nan)
    variogram = np.where(complete, sr.vs_ensemble(obs, sub, p=0.5), np.nan)

    q = np.quantile(ens, [0.05, 0.25, 0.5, 0.75, 0.95], axis=-1)
    mean = ens.mean(axis=-1)
    # The fair CRPS. scoringrules' "fair" estimator builds all S² pairwise differences (NumPy
    # backend); the probability-weighted-moment form is algebraically identical and O(S log S).
    per_variable = {
        "lpd_marginal": lpd_marginal,
        "crps": sr.crps_ensemble(obs, ens, estimator=_FAIR_CRPS),
        "qwcrps_center": quantile_weighted_crps(ens, obs, "center", qw_levels),
        "qwcrps_left": quantile_weighted_crps(ens, obs, "left", qw_levels),
        "qwcrps_right": quantile_weighted_crps(ens, obs, "right", qw_levels),
        "sq_err": (mean - obs) ** 2,
        "abs_err": np.abs(q[2] - obs),
        "pit": (ens <= obs[..., None]).mean(axis=-1),
        "cover50": ((q[1] <= obs) & (obs <= q[3])).astype(np.float64),
        "cover90": ((q[0] <= obs) & (obs <= q[4])).astype(np.float64),
    }
    if integrated:
        per_variable |= {
            "lpd_marginal_mcse": marginal_mcse,
            "lpd_marginal_conditioned": cond_marginal,
        }
    if tw_bounds is not None:
        lower = np.array([tw_bounds.get(v, (-np.inf, np.inf))[0] for v in fr.variables])
        upper = np.array([tw_bounds.get(v, (-np.inf, np.inf))[1] for v in fr.variables])
        per_variable["twcrps"] = np.stack(
            [
                sr.twcrps_ensemble(
                    obs[:, j], ens[:, j], a=lower[j], b=upper[j], estimator=_FAIR_CRPS
                )
                for j in range(n)
            ],
            axis=1,
        )
    data = {
        k: (("horizon", "variable"), np.where(observed, v, np.nan)) for k, v in per_variable.items()
    }
    data |= {
        "lpd": ("horizon", lpd),
        "lpd_mcse": ("horizon", lpd_mcse),
        "energy": ("horizon", energy),
        "variogram": ("horizon", variogram),
    }
    if integrated:
        data |= {
            "lpd_conditioned": ("horizon", cond),
            "lpd_mcse_conditioned": ("horizon", cond_mcse),
        }
    coords = {"horizon": np.arange(1, H + 1), "variable": fr.variables}
    return xr.Dataset(data, coords=coords, attrs={"lpd_method": lpd_method} if integrated else None)
