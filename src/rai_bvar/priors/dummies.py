"""SOC/SUR dummy rows of the MCMC coefficient prior (DESIGN §13 WP25; N05 §7.7; N04 §3.1). NumPy
leaves on the host.

**Rows** (Phase 1 convention, ``engines.analytic.ConjugateProblem.dummies`` and
``engines.structural_nig._Window.dummies``; a test pins all three bit for bit). With ``ȳ₀`` the mean
of the first ``p`` observations of the window (GLP 2015), ``μ = soc`` and ``δ = sur``:

* sum-of-coefficients: one row per SOC variable ``j`` (every variable with ``soc_scope="all"``,
  else those with a non-zero own-lag prior mean), ``y_d = e_j ȳ₀_j/μ``, ``x_d = (0_det, y_d, …, y_d)``;
* single-unit-root: one row ``y_d = ȳ₀/δ``, ``x_d = (1/δ [constant only; a trend gets 0], y_d, …, y_d)``.

``x_d`` follows the ``data/design.py`` regressor order ``(d', y_{−1}', …, y_{−p}')``.

**Weights** (``CoefPriorArrays.dummy_w``, λ-free): the rows are prior rows, appended to every
equation's regression with a weight that never depends on a volatility state unless the volatility
block's conditional sees them too (N05 §7.7):

* structural Mode I and ``iw_equivalent``: the fixed ``1/σ̄_i² = ι/ψ_i`` (DESIGN §6.1), which makes
  the rows part of the Gaussian prior of ``θ_i = [δ_i | a_i]``;
* structural Mode II: ``1`` — the kernel weights the rows ``1/λ_i`` exactly like data rows and the
  ``λ_i`` step counts them (coupling), which is the Phase 1 NIW-plus-dummies posterior;
* reduced form: ``1/ψ_j``, rows with fixed variance ``ψ_j`` in equation ``j`` of ``Π`` (never scaled
  by ``Σ_t``: that would make the ``Π`` prior depend on ``A``; N04 §3.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from rai_bvar.engines.mcmc.state import DummyRows, Empty
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import WindowScales, _check_dims, _check_scales

DummyMode = Literal["I", "II", "iw_equivalent", "reduced"]

__all__ = ["dummy_arrays", "dummy_hyper", "dummy_rows", "dummy_weights", "soc_rows"]


def soc_rows(own_means: np.ndarray, soc_scope: str) -> np.ndarray:
    """Variables with a SOC row: all (``soc_scope="all"``) or those with a non-zero own-lag mean."""
    own_means = np.asarray(own_means, dtype=np.float64)
    if soc_scope == "all":
        return np.arange(own_means.size)
    if soc_scope != "persistent":
        raise ValueError(f"soc_scope must be 'persistent' or 'all', got {soc_scope!r}")
    return np.flatnonzero(own_means != 0.0)


def dummy_rows(
    ybar0: np.ndarray,
    rows: np.ndarray,
    soc: float | None,
    sur: float | None,
    *,
    lags: int,
    n_det: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(Y_d (m, n), X_d (m, k))``: SOC rows ``diag(ȳ₀)/μ`` restricted to ``rows``, then the SUR row
    ``ȳ₀'/δ`` (module docstring; the Phase 1 operation order, so the rows are bitwise equal)."""
    ybar0 = np.asarray(ybar0, dtype=np.float64)
    n = ybar0.size
    _check_dims(n, lags, n_det)
    k = n_det + n * lags
    rows_y, rows_x = [np.zeros((0, n))], [np.zeros((0, k))]
    if soc is not None:
        if np.asarray(rows).size == 0:
            raise ValueError(
                "the SOC dummy needs at least one variable with a non-zero own-lag prior mean"
            )
        yd = (np.diag(ybar0) / soc)[rows]
        rows_y.append(yd)
        rows_x.append(np.hstack([np.zeros((len(yd), n_det)), np.tile(yd, lags)]))
    if sur is not None:
        yd = ybar0[None, :] / sur
        det = np.zeros((1, n_det))
        if n_det:
            det[0, 0] = 1.0 / sur  # constant only; a trend gets no SUR weight
        rows_y.append(yd)
        rows_x.append(np.hstack([det, np.tile(yd, lags)]))
    return np.vstack(rows_y), np.vstack(rows_x)


def dummy_weights(scales: WindowScales, n: int, mode: DummyMode) -> np.ndarray:
    """λ-free weights ``(n,)`` of the dummy rows (module docstring): ``ι/ψ_i`` (structural Mode I
    and ``iw_equivalent``), ``1`` (Mode II), ``1/ψ_j`` (reduced form)."""
    _check_scales(scales, n)
    if mode in ("I", "iw_equivalent"):
        return 1.0 / scales.sigma2_bar
    if mode == "II":
        return np.ones(n)
    if mode == "reduced":
        return 1.0 / scales.psi
    raise ValueError(f"mode must be 'I', 'II', 'iw_equivalent' or 'reduced', got {mode!r}")


def dummy_hyper(prior: Minnesota, hyper: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``soc``/``sur`` in use: ``hyper`` values (EB, DESIGN §6.5) override the prior's; a dummy that
    is off (``None`` in the prior) stays off."""
    hyper = hyper or {}
    out: dict[str, Any] = {}
    for name in ("soc", "sur"):
        value = getattr(prior, name)
        if value is not None and hyper.get(name) is not None:
            value = hyper[name]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int | float | np.number):
                raise TypeError(f"hyperparameter {name!r} must be a number, got {value!r}")
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
            value = float(value)
        out[name] = value
    return out


def dummy_arrays(
    values: np.ndarray,
    scales: WindowScales,
    prior: Minnesota,
    hyper: Mapping[str, Any] | None,
    *,
    lags: int,
    n_det: int,
    mode: DummyMode,
) -> tuple[DummyRows | Empty, np.ndarray | Empty]:
    """``(SharedPrior.dummies, CoefPriorArrays.dummy_w)`` of one window: the rows from the window's
    first ``lags`` observations ``values[:lags]`` (R9) and the prior's ``soc``/``sur`` (or the EB
    values in ``hyper``), and the weights of ``mode``; ``(Empty(), Empty())`` without dummies."""
    values = np.asarray(values, dtype=np.float64)
    n = scales.n
    if values.ndim != 2 or values.shape[1] != n or values.shape[0] <= lags:
        raise ValueError(f"values must be (T, {n}) with T > lags = {lags}, got {values.shape}")
    h = dummy_hyper(prior, hyper)
    if h["soc"] is None and h["sur"] is None:
        return Empty(), Empty()
    ybar0 = values[:lags].mean(axis=0)
    Y, X = dummy_rows(
        ybar0, soc_rows(scales.own_means, prior.soc_scope), h["soc"], h["sur"],
        lags=lags, n_det=n_det,
    )  # fmt: skip
    return DummyRows(Y=Y, X=X), dummy_weights(scales, n, mode)
