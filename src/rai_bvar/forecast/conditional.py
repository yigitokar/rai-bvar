"""Conditional forecasts with hard conditions: per-draw scenarios (SPEC §6.2; P3 DESIGN §6, D-50).

**Convention** (D-50, E43). The conditions ``y_{T+h, j} = r_{h,j}`` (model scale) are imposed on
each ``(chain, draw)`` slice of an unconditional forecast separately: given the slice's parameter
draw ``θ_s`` and its drawn volatility path ``λ_s`` the joint path is Gaussian, and the slice is
replaced by one exact draw from that Gaussian conditioned on the cells ``RY = r`` (Waggoner & Zha
1999 §3, the per-draw step; Bańbura, Giannone & Lenza 2015). The conditions update neither
parameters nor volatilities, and draws are not reweighted by how likely they make the
conditions, so the equal-weight mixture over draws is a **scenario**, not ``p(y | data, RY = r)``;
the log weights ``log w_s = log N(r; Rμ_s, R₁ᵀR₁)`` that would turn it into the latter are stored
(Q25). No random number is drawn: the result is a deterministic function of the unconditional
draws, so the same seed reproduces it.

**The exact update** (§6.2). Per slice, with ``u_k = paths_k − cond_mean_k``, ``L_k = cond_chol_k``
and the moving-average coefficients ``Φ_0 = I``, ``Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}`` of the
draw's ``lag_coefs``: the zero-shock path ``μ_h = cond_mean_h − Σ_{k<h} Φ_{h−k} u_k`` and the
block lower triangular ``M`` (``nH × nH``, ``M_{h,k} = Φ_{h−k} L_k``), so that ``Y = μ + Mε``
exactly with ``ε_k = L_k⁻¹ u_k`` (:func:`zero_shock_mean`, :func:`impact_matrix`). ``M`` is lower
triangular with a positive diagonal in the stacked order ``(h − 1)·n + j``, so any selection
``RM`` has full row rank. With the QR ``(RM)ᵀ = Q₁R₁`` (``R₁`` upper, positive diagonal)::

    Y_c  = Y + M Q₁ R₁⁻ᵀ (r − RY)                       one exact draw from Y | θ_s, λ_s, RY = r
    E_c  = μ + M Q₁ R₁⁻ᵀ (r − Rμ)                        the conditional mean
    Cov_c,h = N_hᵀ N_h,   N_h = (I − Q₁Q₁ᵀ) M_hᵀ          M_h: the n rows of horizon h
    L^c_h = Rᵀ of the QR of N_h[:, free_h], embedded at (free_h, free_h) of an n × n zero matrix
    log w = −(m/2) log 2π − Σ log R₁,ii − ½ ‖R₁⁻ᵀ (r − Rμ)‖²

(:func:`exact_update`). The conditioned rows and columns of ``Cov_c,h`` are exactly zero in exact
arithmetic, so factoring the free columns only is exact (E32); the embedded factor is lower
triangular because ``free_h`` is sorted. The conditioned cells of ``Y_c`` and ``E_c`` are set to
``r`` (rounding). An ``R₁`` whose diagonal has an entry below ``1e-12·max|R₁|`` (near-dependent
conditions) raises; nothing is jittered.

**The result** (§6.3). :class:`ConditionalForecastResult` keeps the ``ForecastResult`` contract
fields with ``paths = Y_c``, ``cond_mean = E_c`` and ``cond_chol = L^c`` (the moments of
``y_{T+h}`` given the draw, its scale path and the conditions, integrated over the other shocks;
not the one-step moments of SPEC §7, which condition on the path), ``paths_level`` the pathwise
inverse transform of ``Y_c`` (SPEC §5.2), and adds ``condition (horizon, variable)`` (``NaN`` =
free), ``conditioned (horizon, variable)`` and ``log_weight (chain, draw)``. It drops the
accelerator variables of the unconditional factors (``cond_A``, ``cond_scale``,
``cond_scale_persistent``, ``log_p``, ``log1m_p``, ``nu``, ``lag_coefs``; E33). Its attrs are the
base forecast's (``seed`` is the base forecast's seed) plus ``conditional = "hard-per-draw-v1"``
and ``weight_ess``, the Kish effective sample size ``(Σw)²/Σw²`` of the stored weights.
``evaluation.scores.score_forecast`` (and so the backtest) refuses it, and
``forecast.invariants.check_conditional_structure`` replaces ``check_structure`` for it.

This module is NumPy-only at import (P3 §1.1, E137): it imports no JAX and no engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar.forecast.result import CHOL_DIMS, DIMS, LAG_DIMS, ForecastResult
from rai_bvar.linalg import positive_diagonal, solve_lower

CONDITIONAL_KIND = "hard-per-draw-v1"  # attrs["conditional"] (P3 §6.3)
R1_RTOL = 1e-12  # near-dependent conditions: diag(R₁) below R1_RTOL·max|R₁| raises (§6.1, E-16)
DROPPED = ("cond_A", "cond_scale", "cond_scale_persistent", "log_p", "log1m_p", "nu", "lag_coefs")
_CHUNK_ELEMENTS = 4_000_000  # elements of M (S_c·(nH)²) per chunk of draws (~32 MB)
_LOG_2PI = math.log(2.0 * math.pi)


class DependentConditionsError(ValueError):
    """Near-dependent conditions (E-16): ``diag(R₁)`` below ``R1_RTOL·max|R₁|`` for the draw at
    position ``draw`` of the batch :func:`exact_update` was given."""

    def __init__(self, draw: int, low: float, floor: float):
        self.draw, self.low, self.floor = int(draw), float(low), float(floor)
        super().__init__(
            f"the conditions are (nearly) linearly dependent for draw {self.draw}: R₁ has a "
            f"diagonal entry {self.low:.3g} below {R1_RTOL:g}·max|R₁| = {self.floor:.3g} "
            "(P3 §6.1, E-16); drop or merge the dependent conditions (nothing is jittered)"
        )


class ConditionalDraws(NamedTuple):
    """Output of :func:`exact_update` for ``S`` draws (stacked index ``(h − 1)·n + j``)."""

    paths: np.ndarray  # (S, H, n) Y_c
    mean: np.ndarray  # (S, H, n) E_c
    chol: np.ndarray  # (S, H, n, n) L^c (conditioned rows and columns exactly 0)
    log_weight: np.ndarray  # (S,) log N(r; Rμ, R₁ᵀR₁)
    cov: np.ndarray | None  # (S, nH, nH) the joint Cov_c with full_cov=True, else None


# --------------------------------------------------------------------------- conditions


class Conditions(NamedTuple):
    """Parsed conditions of an ``H × n`` forecast (:func:`parse_conditions`)."""

    table: np.ndarray  # (H, n) values, NaN = free
    index: np.ndarray  # (m,) stacked indices (h − 1)·n + j of the conditioned cells, ascending
    values: np.ndarray  # (m,) r, in the order of index

    @property
    def conditioned(self) -> np.ndarray:
        return np.isfinite(self.table)


def parse_conditions(
    conditions: Any,
    variables: list[str] | tuple[str, ...],
    horizon: int,
    origin: pd.Period | None = None,
) -> Conditions:
    """The conditions as an ``(H, n)`` table (P3 §6.1).

    ``conditions`` is a ``pandas.DataFrame`` indexed by horizon ``1..H`` (ints) or by the forecast
    periods (``origin + 1 … origin + H``), with columns a subset of ``variables``, or a mapping
    ``{variable: sequence}`` whose sequences start at horizon 1 and are padded with ``NaN``.
    ``NaN`` marks a free cell; values are on the model's (transformed) scale. Errors (E-16):
    ``KeyError`` for an unknown variable; ``ValueError`` for a horizon or period outside the
    forecast, a duplicated horizon or variable, a non-finite condition, no condition, or every
    cell conditioned.
    """
    variables = [str(v) for v in variables]
    n, H = len(variables), int(horizon)
    if isinstance(conditions, pd.DataFrame):
        frame = conditions
    elif isinstance(conditions, Mapping):
        columns = {}
        for var, seq in conditions.items():
            values = np.asarray(seq, dtype=np.float64).reshape(-1)
            if values.size > H:
                raise ValueError(
                    f"conditions for {var!r} have {values.size} values; the forecast has horizons "
                    f"1..{H}"
                )
            columns[var] = np.concatenate([values, np.full(H - values.size, np.nan)])
        frame = pd.DataFrame(columns, index=np.arange(1, H + 1))
    else:
        raise TypeError(
            "conditions must be a pandas.DataFrame indexed by horizon (1..H) or forecast period, "
            f"or a mapping {{variable: sequence}}; got {type(conditions).__name__}"
        )
    names = [str(c) for c in frame.columns]
    unknown = [c for c in names if c not in variables]
    if unknown:
        raise KeyError(
            f"conditions name unknown variables {unknown}; the variables are {variables}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"conditions name a variable twice: {names}")
    rows = [_horizon_of(label, H, origin) for label in frame.index]
    if len(set(rows)) != len(rows):
        raise ValueError(f"conditions give a horizon twice: {sorted(rows)}")
    table = np.full((H, n), np.nan)
    values = frame.to_numpy(dtype=np.float64)
    for r, h in enumerate(rows):
        for c, name in enumerate(names):
            table[h - 1, variables.index(name)] = values[r, c]
    bad = np.isinf(table)
    if bad.any():
        h, j = map(int, np.argwhere(bad)[0])
        raise ValueError(
            f"the condition on {variables[j]!r} at horizon {h + 1} is not finite "
            f"({table[h, j]}); use NaN for a free cell"
        )
    held = np.isfinite(table)
    if not held.any():
        raise ValueError("conditions hold no value: give at least one (NaN marks a free cell)")
    if held.all():
        raise ValueError(
            f"every cell of the {H} × {n} forecast is conditioned; leave at least one free (NaN)"
        )
    index = np.flatnonzero(held.reshape(-1))
    return Conditions(table=table, index=index, values=table.reshape(-1)[index].copy())


def _horizon_of(label: Any, H: int, origin: pd.Period | None) -> int:
    """Horizon ``1..H`` of an index label: an integer horizon or a forecast period."""
    if isinstance(label, pd.Period):
        if origin is None:
            raise ValueError("conditions indexed by period need the forecast origin")
        if label.freqstr != origin.freqstr:
            raise ValueError(
                f"condition period {label} has frequency {label.freqstr}; the forecast is "
                f"{origin.freqstr}"
            )
        h = (label - origin).n
        if not 1 <= h <= H:
            raise ValueError(
                f"condition period {label} is outside the forecast periods {origin + 1} … "
                f"{origin + H} (horizons 1..{H})"
            )
        return int(h)
    if isinstance(label, bool | np.bool_) or not isinstance(label, int | np.integer):
        raise ValueError(
            f"condition index label {label!r} is neither a horizon (an int in 1..{H}) nor a "
            "forecast period"
        )
    if not 1 <= int(label) <= H:
        raise ValueError(f"condition horizon {int(label)} is outside 1..{H}")
    return int(label)


# --------------------------------------------------------------------------- the update


def ma_matrices(lag_coefs: np.ndarray, H: int) -> np.ndarray:
    """``Φ_0 … Φ_{H−1}`` ``(S, H, n, n)``: ``Φ_0 = I``, ``Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}``,
    from ``lag_coefs`` ``(S, p, n, n)`` (``[s, l − 1] = A_l``)."""
    A = np.asarray(lag_coefs, dtype=np.float64)
    S, p, n = A.shape[0], A.shape[1], A.shape[2]
    phi = np.zeros((S, H, n, n))
    phi[:, 0] = np.eye(n)
    for j in range(1, H):
        for lag in range(1, min(j, p) + 1):
            phi[:, j] += A[:, lag - 1] @ phi[:, j - lag]
    return phi


def zero_shock_mean(paths: np.ndarray, cond_mean: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """``μ_h = cond_mean_h − Σ_{k<h} Φ_{h−k} u_k`` ``(S, H, n)``, ``u = paths − cond_mean``."""
    u = paths - cond_mean
    mu = np.array(cond_mean, dtype=np.float64)
    for h in range(1, mu.shape[1]):
        for k in range(h):
            mu[:, h] -= (phi[:, h - k] @ u[:, k, :, None])[..., 0]
    return mu


def impact_matrix(phi: np.ndarray, cond_chol: np.ndarray) -> np.ndarray:
    """``M`` ``(S, nH, nH)``, block lower triangular with ``M_{h,k} = Φ_{h−k} L_k`` (``k ≤ h``)."""
    S, H, n = phi.shape[0], phi.shape[1], phi.shape[2]
    M = np.zeros((S, n * H, n * H))
    for h in range(H):
        for k in range(h + 1):
            M[:, h * n : (h + 1) * n, k * n : (k + 1) * n] = phi[:, h - k] @ cond_chol[:, k]
    return M


def exact_update(
    mu: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    index: np.ndarray,
    values: np.ndarray,
    *,
    full_cov: bool = False,
) -> ConditionalDraws:
    """The hard-condition update of ``S`` draws (module docstring; P3 §6.2).

    ``mu`` and ``Y`` are ``(S, H, n)`` (the zero-shock path and the unconditional path), ``M`` is
    ``(S, nH, nH)`` with ``Y = μ + Mε``, ``index`` the ascending stacked indices of the
    conditioned cells and ``values`` their values ``r``. This is the function the keyword-only
    ``_update`` of :func:`conditional_forecast` replaces (the test seam of mutant M33). Raises
    :class:`DependentConditionsError` (a ``ValueError``) when ``diag(R₁)`` has an entry below
    ``R1_RTOL·max|R₁|`` for some draw of the batch; ``full_cov=True`` also returns the joint
    ``Cov_c`` (tests).
    """
    S, H, n = Y.shape
    nH = n * H
    index = np.asarray(index, dtype=np.intp)
    r = np.asarray(values, dtype=np.float64)
    m = index.size
    Yf, muf = Y.reshape(S, nH), mu.reshape(S, nH)
    Q, R1 = np.linalg.qr(np.swapaxes(M[:, index, :], 1, 2))  # (RM)ᵀ = Q₁R₁: (S, nH, m), (S, m, m)
    signs = np.sign(np.diagonal(R1, axis1=1, axis2=2))
    signs = np.where(signs == 0, 1.0, signs)
    Q, R1 = Q * signs[:, None, :], R1 * signs[:, :, None]
    diag = np.diagonal(R1, axis1=1, axis2=2)
    floor = R1_RTOL * np.abs(R1).max(axis=(1, 2))
    low = np.flatnonzero((diag <= floor[:, None]).any(axis=1))
    if low.size:
        s = int(low[0])
        raise DependentConditionsError(s, diag[s].min(), floor[s])
    R1T = np.swapaxes(R1, 1, 2)  # lower triangular
    x_mu = solve_lower(R1T, r - muf[:, index])  # R₁⁻ᵀ(r − Rμ)
    x_y = solve_lower(R1T, r - Yf[:, index])  # R₁⁻ᵀ(r − RY)
    MQ = M @ Q
    paths = Yf + (MQ @ x_y[..., None])[..., 0]
    mean = muf + (MQ @ x_mu[..., None])[..., 0]
    paths[:, index] = r
    mean[:, index] = r
    log_w = -0.5 * m * _LOG_2PI - np.log(diag).sum(axis=1) - 0.5 * np.einsum("si,si->s", x_mu, x_mu)
    free = np.ones(nH, dtype=bool)
    free[index] = False
    free = free.reshape(H, n)
    chol = np.zeros((S, H, n, n))
    QT = np.swapaxes(Q, 1, 2)
    for h in range(H):
        f = np.flatnonzero(free[h])
        if not f.size:
            continue
        rows = np.swapaxes(M[:, h * n + f, :], 1, 2)  # M_h[free]ᵀ: (S, nH, n_f)
        N = rows - Q @ (QT @ rows)  # (I − Q₁Q₁ᵀ) M_h[free]ᵀ
        Rh = positive_diagonal(np.linalg.qr(N, mode="r"), upper=True)
        chol[:, h, f[:, None], f[None, :]] = np.swapaxes(Rh, 1, 2)
    cov = None
    if full_cov:
        MT = np.swapaxes(M, 1, 2)
        N = MT - Q @ (QT @ MT)
        cov = np.swapaxes(N, 1, 2) @ N
    return ConditionalDraws(
        paths=paths.reshape(S, H, n), mean=mean.reshape(S, H, n), chol=chol, log_weight=log_w,
        cov=cov,
    )  # fmt: skip


UpdateFn = Callable[..., ConditionalDraws]


def update_draws(
    paths: np.ndarray,
    cond_mean: np.ndarray,
    cond_chol: np.ndarray,
    lag_coefs: np.ndarray,
    cond: Conditions,
    *,
    _update: UpdateFn | None = None,
) -> ConditionalDraws:
    """:func:`exact_update` of stacked draws (``(S, H, n)``, ``(S, H, n, n)``, ``(S, p, n, n)``),
    in chunks of bounded memory (each chunk builds ``μ`` and ``M`` of its draws)."""
    update = exact_update if _update is None else _update
    S, H, n = paths.shape
    chunk = max(1, _CHUNK_ELEMENTS // (n * H) ** 2)
    out_paths, out_mean = np.empty((S, H, n)), np.empty((S, H, n))
    out_chol, out_lw = np.empty((S, H, n, n)), np.empty(S)
    for lo in range(0, S, chunk):
        sl = slice(lo, min(S, lo + chunk))
        Y = np.asarray(paths[sl], dtype=np.float64)
        L = np.asarray(cond_chol[sl], dtype=np.float64)
        phi = ma_matrices(lag_coefs[sl], H)
        mu = zero_shock_mean(Y, np.asarray(cond_mean[sl], dtype=np.float64), phi)
        try:
            res = update(mu, impact_matrix(phi, L), Y, cond.index, cond.values)
        except DependentConditionsError as err:  # name the stacked draw, not the chunk position
            raise DependentConditionsError(sl.start + err.draw, err.low, err.floor) from None
        out_paths[sl], out_mean[sl], out_chol[sl], out_lw[sl] = res[:4]
    return ConditionalDraws(out_paths, out_mean, out_chol, out_lw, None)


# --------------------------------------------------------------------------- entry points


def condition_forecast(
    base: ForecastResult,
    conditions: Any,
    *,
    data: Any = None,
    _update: UpdateFn | None = None,
) -> ConditionalForecastResult:
    """Condition an unconditional ``ForecastResult`` that carries ``lag_coefs`` (P3 §6.2).

    ``conditions`` as :func:`parse_conditions` (or a parsed :class:`Conditions`); ``data`` (a
    ``VARData``, optional) gives ``paths_level`` by its pathwise inverse transform. ``base`` needs
    the accelerator variable ``lag_coefs`` (``ValueError`` otherwise, naming why it is missing).
    """
    if isinstance(base, ConditionalForecastResult):
        raise TypeError(
            "base is already a ConditionalForecastResult; condition an unconditional forecast"
        )
    if not isinstance(base, ForecastResult):
        raise TypeError(f"base must be a ForecastResult, got {type(base).__name__}")
    ds = base.dataset
    if "lag_coefs" not in ds:
        raise ValueError(_missing_lag_coefs(base))
    if ds["lag_coefs"].dims != LAG_DIMS:
        raise ValueError(f"'lag_coefs' must have dims {LAG_DIMS}, got {ds['lag_coefs'].dims}")
    cond = (
        conditions
        if isinstance(conditions, Conditions)
        else parse_conditions(conditions, base.variables, base.horizon, base.origin)
    )
    H, n = base.horizon, len(base.variables)
    if cond.table.shape != (H, n):
        raise ValueError(f"conditions must be a ({H}, {n}) table, got {cond.table.shape}")
    C, D = ds.sizes["chain"], ds.sizes["draw"]
    try:
        res = update_draws(
            base.stacked("paths"), base.stacked("cond_mean"), base.stacked("cond_chol"),
            base.stacked("lag_coefs"), cond, _update=_update,
        )  # fmt: skip
    except DependentConditionsError as err:
        c, d = divmod(err.draw, D)
        raise ValueError(f"{err} [chain {c}, draw {d}]") from None
    paths = res.paths.reshape(C, D, H, n)
    data_vars = {
        "paths": (DIMS, paths),
        "cond_mean": (DIMS, res.mean.reshape(C, D, H, n)),
        "cond_chol": (CHOL_DIMS, res.chol.reshape(C, D, H, n, n)),
        "condition": (("horizon", "variable"), cond.table.copy()),
        "conditioned": (("horizon", "variable"), cond.conditioned),
        "log_weight": (("chain", "draw"), res.log_weight.reshape(C, D)),
    }
    if data is not None and hasattr(data, "inverse_paths"):
        data_vars["paths_level"] = (DIMS, data.inverse_paths(paths))
    lw = res.log_weight
    w = np.exp(lw - lw.max())
    attrs = {
        **ds.attrs,
        "conditional": CONDITIONAL_KIND,
        "weight_ess": float(w.sum() ** 2 / (w * w).sum()),
    }
    kept = set(CHOL_DIMS)  # coordinates of dims only the dropped variables had are dropped too
    coords = {name: ds.coords[name] for name in ds.coords if set(ds.coords[name].dims) <= kept}
    return ConditionalForecastResult(xr.Dataset(data_vars, coords=coords, attrs=attrs))


def _missing_lag_coefs(base: ForecastResult) -> str:
    """The E-16 message for a base forecast without ``lag_coefs``, with the reason."""
    attrs = base.dataset.attrs
    engine = attrs.get("engine")
    head = (
        "conditional forecasts need the forecast's accelerator variable 'lag_coefs' (the lag "
        "matrices of each draw, P3 §6.2), and this forecast has none: "
    )
    if engine == "mcmc" and int(attrs.get("in_loop", 0)) == 1:
        why = (
            "it is an in-loop forecast (FitPlan) whose draws are not all stored; ask for a "
            "different (horizon, draws, seed) or store every draw the plan forecasts from"
        )
    elif engine == "mcmc":
        why = "its mean is not linear (only linear means have lag matrices)"
    elif engine == "analytic" and "mh_seed" in attrs:
        why = "the analytic forecast was made without them; call forecast(..., lag_coefs=True)"
    else:
        why = (
            f"model {attrs.get('model', '?')!r} (engine {engine!r}) does not attach them "
            "(benchmark and ordering-pooled forecasts carry none)"
        )
    return head + why


def conditional_forecast(
    fit: Any,
    horizon: int,
    conditions: Any,
    draws: int = 1000,
    seed: Any = None,
    base: ForecastResult | None = None,
    *,
    _update: UpdateFn | None = None,
) -> ConditionalForecastResult:
    """Hard-condition scenarios from a fitted linear BVAR (P3 §6; D-50; module docstring).

    The base is ``fit.forecast(horizon, draws, seed, lag_coefs=True)`` unless ``base`` (an
    unconditional ``ForecastResult`` of ``fit`` with the same ``horizon``; ``draws`` and ``seed``
    are then not used) is given. ``conditions``: a ``DataFrame`` indexed by horizon ``1..H`` or by
    the forecast periods, columns a subset of the variables, ``NaN`` = free, values on the
    model's transformed scale; or a mapping ``{variable: sequence}`` (``NaN`` padding). Every
    ``(chain, draw)`` slice is conditioned exactly given its parameter draw and its drawn
    volatility path; the result is a :class:`ConditionalForecastResult`, a scenario that
    ``score_forecast`` refuses. ``_update`` (keyword-only, tests only) replaces
    :func:`exact_update` (mutant M33).
    """
    for name, value in (("horizon", horizon), ("draws", draws)):
        if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    data = getattr(fit, "data", None)
    if base is not None:
        if isinstance(base, ConditionalForecastResult) or not isinstance(base, ForecastResult):
            raise TypeError(
                "base must be an unconditional ForecastResult of this fit, got "
                f"{type(base).__name__}"
            )
        if base.horizon != int(horizon):
            raise ValueError(
                f"base has horizon {base.horizon}; conditional_forecast was asked for horizon "
                f"{int(horizon)} (they must be the same)"
            )
        variables, origin = base.variables, base.origin
        if data is not None and hasattr(data, "last") and origin != data.last:
            raise ValueError(
                f"base forecasts from {origin}; this fit's sample ends at {data.last} (base must be "
                "a forecast of this fit: the levels are inverted against its history)"
            )
        expected = _fit_variables(fit)
        if expected is not None and list(expected) != list(variables):
            raise ValueError(
                f"base forecasts the variables {variables}; this fit's ordering is {list(expected)}"
            )
    else:
        variables = _fit_variables(fit)
        origin = data.last if data is not None and hasattr(data, "last") else None
    cond = None
    if variables is not None:  # check the conditions before the (possibly costly) forecast
        cond = parse_conditions(conditions, variables, int(horizon), origin)
    if base is None:
        from rai_bvar.models.base import forecast_options

        base = fit.forecast(int(horizon), int(draws), seed, **forecast_options(fit, lag_coefs=True))
    if cond is None:
        cond = parse_conditions(conditions, base.variables, base.horizon, base.origin)
    return condition_forecast(base, cond, data=data, _update=_update)


def _fit_variables(fit: Any) -> list[str] | None:
    """The variable ordering of ``fit``'s forecasts: ``fit.ordering`` (MCMC) or
    ``fit.data.variables`` (analytic engines); ``None`` when neither is known."""
    ordering = getattr(fit, "ordering", None)
    if ordering is not None:
        return [str(v) for v in ordering]
    data = getattr(fit, "data", None)
    if data is not None and hasattr(data, "variables"):
        return [str(v) for v in data.variables]
    return None


# --------------------------------------------------------------------------- the result


class ConditionalForecastResult(ForecastResult):
    """Per-draw hard-condition scenarios (P3 §6.3): a ``ForecastResult`` (same fields; inherited
    ``quantiles``, ``mean``, ``median``, ``to_dataframe``) whose dataset adds ``condition``,
    ``conditioned`` and ``log_weight`` and drops the unconditional accelerator variables (module
    docstring). It is a scenario, not a forecast to score: ``score_forecast`` and the backtest
    refuse it."""

    def __init__(self, dataset: xr.Dataset):
        super().__init__(dataset)
        for name, dims in (
            ("condition", ("horizon", "variable")),
            ("conditioned", ("horizon", "variable")),
            ("log_weight", ("chain", "draw")),
        ):
            if name not in dataset or dataset[name].dims != dims:
                raise ValueError(f"ConditionalForecastResult needs {name!r} with dims {dims}")
        if dataset.attrs.get("conditional") != CONDITIONAL_KIND:
            raise ValueError(
                f"ConditionalForecastResult needs attrs['conditional'] = {CONDITIONAL_KIND!r}"
            )
        present = [name for name in DROPPED if name in dataset]
        if present:
            raise ValueError(f"a ConditionalForecastResult carries no {present} (P3 §6.3, E33)")

    @property
    def condition(self) -> xr.DataArray:
        """``(horizon, variable)``: the condition values, ``NaN`` = free."""
        return self._ds["condition"]

    @property
    def conditioned(self) -> xr.DataArray:
        """``(horizon, variable)`` bool: which cells are conditioned."""
        return self._ds["conditioned"]

    @property
    def log_weight(self) -> xr.DataArray:
        """``(chain, draw)``: ``log N(r; Rμ_s, R₁ᵀR₁)``, the density of the conditions under each
        draw's unconditional path; ``exp`` of it reweights the scenario towards ``p(y | data, RY
        = r)`` (Q25), which the equal-weight mixture is not."""
        return self._ds["log_weight"]

    def log_pred_density(
        self,
        y: np.ndarray,
        horizon: int,
        variables: Any = None,
        *,
        return_mcse: bool = False,
        rb_transitory: bool = False,
        lpd_method: Literal["conditioned", "integrated"] = "conditioned",
    ):
        """``log (1/S) Σ_s N(y_free; E_c,s, Cov_c,s)`` for free variables of ``horizon``: the
        equal-weight mixture of the per-draw scenario densities, given each draw's parameters,
        its drawn volatility path and the conditions, integrated over the other shocks. It is not
        ``p(y_free | data, RY = r)`` (that needs the weights ``log_weight``; Q25) and is no
        forecast score. Raises ``ValueError`` for a conditioned ``(horizon, variable)`` (also when
        ``variables=None`` and some variable of that horizon is conditioned) and for
        ``lpd_method="integrated"`` (the path-integrated estimator needs the unconditional lag
        matrices, which this result drops). Otherwise as ``ForecastResult.log_pred_density``."""
        if lpd_method == "integrated":
            raise ValueError(
                "lpd_method='integrated' is not available for a conditional forecast (a per-draw "
                "scenario carries no unconditional lag matrices; P3 §6.3)"
            )
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be in 1..{self.horizon}")
        idx = self._indices(variables)
        held = self._ds["conditioned"].values[horizon - 1, idx]
        if held.any():
            names = [self.variables[i] for i in idx[held]]
            raise ValueError(
                f"{names} are conditioned at horizon {horizon}: a hard condition has no "
                "predictive density; select free variables only"
            )
        return super().log_pred_density(
            y, horizon, variables, return_mcse=return_mcse, rb_transitory=rb_transitory,
            lpd_method=lpd_method,
        )  # fmt: skip

    def __repr__(self) -> str:
        m = int(self._ds["conditioned"].values.sum())
        return (
            f"ConditionalForecastResult(origin={self.origin}, horizon={self.horizon}, "
            f"variables={self.variables}, samples={self.n_samples}, conditions={m})"
        )
