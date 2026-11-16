"""Host builders of the global-local and SSVS prior arrays (P3 DESIGN §3.6, D-37, D-38, D-44).

NumPy only on the host; imported by ``engines/mcmc/resolve.py`` (so ``import rai_bvar`` loads no
JAX through it, P3 §1.1). Everything a GL prior does not govern (ψ, the own-lag means, the
deterministic prior and the ``a_i`` arrays) comes from the Phase 2 Minnesota builders through the
prior's internal adapter ``GlobalLocalPrior.minnesota()`` (P3 §3.6); this module adds, per
estimation window (R9):

* the GL mask (1 on the lag slots ``r ≥ n_det`` of the ``data/design.py`` layout, D-36), the group
  ids of D-38 (``GlobalLocalPrior.group_ids``, which raises E-13 for a custom array of the wrong
  shape) placed on the lag slots, and ``G``;
* ``log r_j`` (D-37, P3 §3.1): ``log ψ_i − log ι − log ψ_m`` (structural), ``log ψ_i − log ψ_m``
  (reduced) for the coefficient of equation ``i`` (``ι = i + 1``) on a lag of variable ``m``, from
  the window's ψ; exactly 0 under ``scale="none"`` and on the deterministic slots;
* the group-level hyperparameters (``GLShared``; unused fields 0, NG's tied ``c = a/2`` encoded as
  ``c = −1``, a fixed ``a`` as ``log_a_fixed`` with a one-point zero grid);
* SSVS ``log τ₀, log τ₁`` (D-44): ``log c + log ŝ_j`` with the semi-automatic ``ŝ_j`` of
  :func:`ssvs_shat`, else ``log c + ½ log r_j``.

The semi-automatic scale is the flat-prior posterior s.d. of the coefficient on the window's real
rows, one QR per equation, ``[(FᵀF)⁻¹]_jj`` as the squared row norms of ``R⁻¹`` (triangular
solves; no inverse of ``FᵀF``)::

    structural, equation i:  ŝ²_j = RSS_i/(T_eff − q_i − 2) · [(F_iᵀF_i)⁻¹]_jj,   F_i = [X | X M_c[:, :i] − Y_{:, :i}]
    reduced, equation j:     ŝ²_{rj} = Ŝ_jj/(T_eff − k − n − 1) · [(XᵀX)⁻¹]_rr

with ``q_i = k + i``; it needs ``T_eff > k + n + 1`` in both forms (the structural condition
``T_eff > q_i + 2`` for every ``i`` binds at ``i = n − 1``), else E-12 at fit.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import solve_triangular

from rai_bvar.engines.mcmc.state import Empty, GLArrays, GLShared, SSVSArrays
from rai_bvar.priors.global_local import (
    R2D2,
    SSVS,
    DirichletLaplace,
    DiscreteGrid,
    GlobalLocalPrior,
    Horseshoe,
    NormalGamma,
)
from rai_bvar.priors.scales import WindowScales, prior_mean_matrix, regressor_layout
from rai_bvar.priors.sv import Beta

__all__ = ["check_coef_groups", "gl_arrays", "gl_log_r", "shat_error_message", "ssvs_shat"]

Form = Literal["structural", "reduced"]


def _form(form: str) -> Form:
    out = {"structural": "structural", "linear_structural": "structural", "reduced": "reduced",
           "linear_reduced": "reduced"}.get(form)  # fmt: skip
    if out is None:
        raise ValueError(f"form must be 'structural' or 'reduced', got {form!r}")
    return out  # type: ignore[return-value]


def check_coef_groups(prior: GlobalLocalPrior, n: int, lags: int) -> tuple[np.ndarray, int]:
    """The ``(n, n·lags)`` group ids and ``G`` of ``prior`` for ``n`` variables (D-38); a custom
    array of another shape raises E-13 (at fit, P3 §7.3)."""
    ids = prior.group_ids(n, lags)
    return ids, int(ids.max()) + 1


def gl_log_r(
    psi: np.ndarray, n: int, lags: int, n_det: int, form: str, scale: str = "psi"
) -> np.ndarray:
    """``log r_j`` (``n × k``, row = equation) of P3 D-37 (module docstring)."""
    form = _form(form)
    k = n_det + n * lags
    out = np.zeros((n, k))
    if scale == "none":
        return out
    if scale != "psi":
        raise ValueError(f"scale must be 'psi' or 'none', got {scale!r}")
    psi = np.asarray(psi, dtype=np.float64)
    _, var_of = regressor_layout(n, lags, n_det)
    m = var_of[n_det:]
    log_psi = np.log(psi)
    lag = log_psi[:, None] - log_psi[m][None, :]
    if form == "structural":
        lag = lag - np.log(np.arange(1.0, n + 1.0))[:, None]
    out[:, n_det:] = lag
    return out


def shat_error_message(T_eff: int, needed: int) -> str:
    """E-12 (P3 §7.3), raised at fit."""
    return (
        "SSVS(semiautomatic=True) scales by the flat-prior posterior s.d., which needs more "
        f"observations than coefficients (T = {T_eff}, needed > {needed}); use "
        "semiautomatic=False or fewer lags"
    )


def _real_rows(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(Y, X)`` of the window's real rows from an ``EngineData`` (padded) or a pair."""
    if isinstance(data, tuple):
        Y, X = data
        return np.asarray(Y, dtype=np.float64), np.asarray(X, dtype=np.float64)
    T = int(np.asarray(data.T_eff))
    return np.asarray(data.Y, np.float64)[:T], np.asarray(data.X, np.float64)[:T]


def _ols(F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """``(diag((FᵀF)⁻¹), RSS)`` by one QR of ``F`` (``R⁻¹`` from a triangular solve)."""
    Q, R = np.linalg.qr(F, mode="reduced")
    Rinv = solve_triangular(R, np.eye(R.shape[0]), lower=False)
    diag = np.sum(Rinv * Rinv, axis=1)
    resid = y - Q @ (Q.T @ y)
    return diag, float(resid @ resid)


def ssvs_shat(
    Y: np.ndarray, X: np.ndarray, form: str, *, Mc: np.ndarray | None = None
) -> np.ndarray:
    """Semi-automatic SSVS scales ``ŝ`` (``n × k``, row = equation) on the real rows ``Y``
    (``T_eff × n``) and ``X`` (``T_eff × k``) (D-44; module docstring). ``Mc`` (``k × (n − 1)``):
    the D3 centring columns of the structural δ-design. Raises E-12 when ``T_eff ≤ k + n + 1``."""
    form = _form(form)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    T, n = Y.shape
    k = X.shape[1]
    if T <= k + n + 1:
        raise ValueError(shat_error_message(T, k + n + 1))
    out = np.empty((n, k))
    if form == "reduced":
        diag, _ = _ols(X, Y[:, 0])
        for j in range(n):
            _, rss = _ols(X, Y[:, j])
            out[j] = np.sqrt(rss / (T - k - n - 1) * diag)
        return out
    if Mc is None:
        raise ValueError("the structural ŝ needs the D3 centring columns Mc")
    Mc = np.asarray(Mc, dtype=np.float64)
    for i in range(n):
        F = np.concatenate([X, X @ Mc[:, :i] - Y[:, :i]], axis=1)
        diag, rss = _ols(F, Y[:, i])
        q = k + i
        out[i] = np.sqrt(rss / (T - q - 2) * diag[:k])
    return out


def _shared(prior: GlobalLocalPrior, A: int) -> GLShared:
    zero = np.zeros(())

    def f(x: float) -> np.ndarray:
        return np.asarray(float(x), dtype=np.float64)

    b = c = global_scale = log_a = p_fixed = s1 = s2 = zero
    support, logw = np.zeros(max(A, 1)), np.zeros(max(A, 1))
    a = getattr(prior, "a", None)
    if isinstance(a, DiscreteGrid):
        support = np.asarray(a.support, dtype=np.float64)
        logw = np.log(np.asarray(a.weights, dtype=np.float64))
    elif a is not None:
        log_a = f(np.log(a))
    if isinstance(prior, Horseshoe):
        global_scale = f(prior.global_scale)
    if isinstance(prior, NormalGamma | R2D2):
        b = f(prior.b)
    if isinstance(prior, NormalGamma):
        c = f(-1.0) if prior.c == "a/2" else f(prior.c)  # c < 0 encodes the tied c = a/2
    if isinstance(prior, SSVS):
        if isinstance(prior.p, Beta):
            s1, s2 = f(prior.p.a), f(prior.p.b)
        else:
            p_fixed = f(prior.p)
    return GLShared(
        b=b, c=c, global_scale=global_scale, log_a_fixed=log_a, a_support=support, a_logw=logw,
        p_fixed=p_fixed, s1=s1, s2=s2,
    )  # fmt: skip


def gl_arrays(
    prior: GlobalLocalPrior,
    scales: WindowScales,
    *,
    n: int,
    lags: int,
    n_det: int,
    form: str,
    data: Any = None,
) -> tuple[GLArrays, GLShared, int]:
    """``(GLArrays, GLShared, G)`` of ``prior`` on one window (module docstring; P3 §3.6).

    ``scales``: the window's ψ and own-lag means (``priors.scales.window_scales`` of the
    prior's internal Minnesota); ``data``: the window's ``EngineData`` (or a pair ``(Y, X)`` of
    its real rows), read only by the semi-automatic SSVS scales."""
    if not isinstance(prior, GlobalLocalPrior):
        raise TypeError(f"prior must be a global-local prior, got {type(prior).__name__}")
    form = _form(form)
    k = n_det + n * lags
    ids, G = check_coef_groups(prior, n, lags)
    mask = np.zeros((n, k))
    mask[:, n_det:] = 1.0
    group = np.zeros((n, k), dtype=np.int32)
    group[:, n_det:] = ids
    log_r = gl_log_r(scales.psi, n, lags, n_det, form, prior.scale)
    ssvs: SSVSArrays | Empty = Empty()
    if isinstance(prior, SSVS):
        if prior.semiautomatic:
            if data is None:
                raise ValueError("SSVS(semiautomatic=True) needs the window's data")
            Y, X = _real_rows(data)
            Mc = (
                prior_mean_matrix(scales, n, lags, n_det)[:, : n - 1]
                if form == "structural"
                else None
            )
            log_s = np.log(ssvs_shat(Y, X, form, Mc=Mc))
        else:
            log_s = 0.5 * log_r
        lag = mask > 0
        ssvs = SSVSArrays(
            log_tau0=np.where(lag, np.log(prior.c0) + log_s, 0.0),
            log_tau1=np.where(lag, np.log(prior.c1) + log_s, 0.0),
        )
    arrays = GLArrays(mask=mask, group=group, log_r=log_r, ssvs=ssvs)
    return arrays, _shared(prior, prior.a_grid), G


def is_global_local(prior: Any) -> bool:
    """Whether ``prior`` is one of the five global-local / SSVS families."""
    return isinstance(prior, Horseshoe | NormalGamma | R2D2 | DirichletLaplace | SSVS)
