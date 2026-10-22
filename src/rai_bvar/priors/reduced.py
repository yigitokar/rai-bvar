"""Reduced-form coefficient prior for the corrected triangular algorithm (CCCM 2022) and the
explicit-array parity hook (DESIGN §2.6, §6.2, §9.9; D-11; N04 §3.1–3.2). NumPy leaves.

``π_j ~ N(m_j, V_j)`` independently over equations ``j``, ``V_j`` diagonal, moment-matched to the
NIW marginal at ``cross = 1`` (``E[Σ_jj] = ψ_j``, ``d = n + 2``):

* own lag ``l``: ``κ²/l^α``;
* lag ``l`` of variable ``m ≠ j``: ``κ²κ_c² ψ_j/(l^α ψ_m)`` (``κ_c`` scales the s.d.);
* deterministic terms: ``κ_d ψ_j = κ_d ψ_j/(d − n − 1)`` (the NIW marginal; N04 §3.1 writes the
  raw ``var_const``, immaterial at ``1e7`` but recorded as a decision, D-11);

i.e. ``V_j = ψ_j Ω^{(j)}`` with the conjugate pattern ``Ω`` of ``priors.scales.conjugate_omega``.
Means: ``m_j`` on the own first lag (transformation codes, SPEC §5.3), 0 elsewhere. The
``(A, λ)`` prior is the structural one (``priors.structural.contemp_arrays`` and the IG ``λ``, or
``iw_equivalent``), so the two forms differ only in the coefficient prior (N04 §3.2). There is no
D3 centring in reduced form: ``SharedPrior.Mc`` is zero.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rai_bvar.engines.mcmc.state import CoefPriorArrays, ContempPriorArrays, SharedPrior
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import (
    WindowScales,
    _check_dims,
    coefficient_groups,
    conjugate_omega,
    lambda_alpha,
    prior_mean_matrix,
)


def reduced_arrays(
    scales: WindowScales,
    prior: Minnesota,
    hyper: Mapping[str, float],
    *,
    n: int,
    lags: int,
    n_det: int,
) -> tuple[CoefPriorArrays, SharedPrior]:
    """``π_j`` prior arrays (rows ``j``): ``prec0_j = 1/(ψ_j Ω^{(j)})``, ``mean0_j = m_j``;
    ``SharedPrior.Mc = 0`` (k, n−1). ``hyper`` overrides ``lambda``/``alpha`` (EB values)."""
    _check_dims(n, lags, n_det)
    lam, alpha = lambda_alpha(prior, hyper, scales)
    var = conjugate_omega(scales, prior, lam, alpha, n, lags, n_det) * scales.psi[:, None]
    M = prior_mean_matrix(scales, n, lags, n_det)
    coef = CoefPriorArrays(
        mean0=np.ascontiguousarray(M.T),
        prec0=1.0 / var,
        group=coefficient_groups(n, lags, n_det),
    )
    return coef, SharedPrior(Mc=np.zeros((M.shape[0], n - 1)))


def explicit_reduced(
    mean0: np.ndarray,
    var0: np.ndarray,
    a_mean0: np.ndarray,
    a_var0: np.ndarray,
    *,
    n_det: int = 1,
) -> tuple[CoefPriorArrays, ContempPriorArrays]:
    """Arrays from explicit numbers (engine-level parity hook, e.g. bayesianVARs; DESIGN §9.9).

    ``mean0`` (n, k): prior means of ``π_j`` (row ``j``); ``var0``: prior variances, broadcast to
    (n, k); ``a_mean0``, ``a_var0``: prior means and variances of ``a_ij``, broadcast to (n, n−1),
    used on the active slots ``j < i`` only (inactive slots get mean 0, precision 1). The
    contemporaneous prior is unscaled (``contemp_scaling="none"``). ``n_det`` (default 1, a
    constant) only labels ``group`` (0 deterministic, 1 own lag, 2 cross lag); it must satisfy
    ``k = n_det + n·p``.
    """
    mean = np.array(mean0, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[0] < 1:
        raise ValueError(f"mean0 must be an (n, k) array, got shape {mean.shape}")
    n, k = mean.shape
    if isinstance(n_det, bool) or not isinstance(n_det, int | np.integer) or n_det < 0:
        raise ValueError(f"n_det must be a non-negative integer, got {n_det!r}")
    lags, rest = divmod(k - n_det, n)
    if rest or lags < 1:
        raise ValueError(f"k = {k} regressors do not fit n_det + n·p with n = {n}, n_det = {n_det}")
    var = _broadcast("var0", var0, (n, k))
    a_mean = _broadcast("a_mean0", a_mean0, (n, n - 1))
    a_var = _broadcast("a_var0", a_var0, (n, n - 1))
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean0 must be finite")
    if not np.all(np.isfinite(var) & (var > 0)):
        raise ValueError("var0 must be positive and finite")
    cmask = (np.arange(n - 1)[None, :] < np.arange(n)[:, None]).astype(np.float64)
    active = cmask > 0
    if not (np.all(np.isfinite(a_mean[active])) and np.all(np.isfinite(a_var[active]))):
        raise ValueError("a_mean0 and a_var0 must be finite on the active slots j < i")
    if not np.all(a_var[active] > 0):
        raise ValueError("a_var0 must be positive on the active slots j < i")
    coef = CoefPriorArrays(
        mean0=mean,
        prec0=1.0 / var,
        group=coefficient_groups(n, lags, n_det),
    )
    contemp = ContempPriorArrays(
        mean0=np.where(active, a_mean, 0.0),
        prec0=np.where(active, 1.0 / np.where(active, a_var, 1.0), 1.0),
        cmask=cmask,
    )
    return coef, contemp


def _broadcast(name: str, value: object, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    try:
        return np.array(np.broadcast_to(arr, shape))
    except ValueError:
        raise ValueError(f"{name} must broadcast to {shape}, got shape {arr.shape}") from None


__all__ = ["explicit_reduced", "reduced_arrays"]
