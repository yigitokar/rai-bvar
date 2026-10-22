"""Structural-form prior arrays in δ-coordinates and their implied reduced-form moments
(DESIGN §2.6, §4.7, §6.1; D-10, D-30; N05 §4.1–4.4). NumPy leaves on the host.

Equation ``i`` (0-based, ``ι = i + 1``) of ``A y_t = Γ x_t + Λ_t^{½} ε_t`` has ``θ_i = [δ_i | a_i]``
with ``δ_i = γ_i − M_c a_i`` (unit Jacobian), ``M_c = M[:, :n−1]`` the reduced-form prior mean
columns, so ``E[γ_i | a_i] = M ã_i`` (D3) and ``δ_i ⊥ a_i`` a priori with diagonal precisions
(N05 §4.1). With ``κ = lambda_``, ``α = alpha``, ``κ_c = cross`` (scales the s.d.), ``κ_d =
var_const``, ``κ_a = contemp``, ``d = n + 2``:

* **Mode I** (independent, moment-matched; every vol block by default): ``σ̄_i² = ψ_i/ι``,
  ``a_ij ~ N(0, κ_a ψ_i/(ι ψ_j))`` (``j < i``), ``δ_i ~ N(m_i, σ̄_i² Ω^{(i)})``: own lag ``l``
  ``κ²/(ι l^α)``, lag ``l`` of variable ``m ≠ i`` ``κ²κ_c² ψ_i/(ι l^α ψ_m)``, deterministic
  ``κ_d ψ_i/ι``.
* **Mode II** (``Constant("iw")`` × structural): ``a_i | λ_i ~ N(0, λ_i κ_a diag(1/ψ_j))`` and
  ``δ_i | λ_i ~ N(m_i, λ_i Ω^{(i)})``, ``Ω^{(i)}_{(l,m)} = κ² c_im (d−n−1)/(l^α ψ_m)``,
  ``Ω^{(i)}_det = κ_d``. At ``κ_c = κ_a = 1`` it is the Phase 1 NIW prior (N05 §4.2).
* **IW-equivalent** (``Constant("iw_equivalent")``; Chan 2022 Prop. 1): Mode I ``δ`` with the
  Mode II ``a`` prior; at ``κ_a = 1`` the implied ``Σ ~ IW(diag ψ, n + 2)`` (N08 §0, §B5).

Arrays hold λ-free precisions ``P̃₀`` (DESIGN §2.6): which blocks the kernels divide by ``λ_i``
is static (``StructStatic.contemp_scaling``, ``MeanStatic.coef_scaling``). Inactive contemporaneous
slots ``j ≥ i`` have mean 0, precision 1 and ``cmask = 0`` (DESIGN §2.3, D-15).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rai_bvar.engines.mcmc.config import Scaling
from rai_bvar.engines.mcmc.state import CoefPriorArrays, ContempPriorArrays, SharedPrior
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import (
    WindowScales,
    _check_dims,
    _check_scales,
    coefficient_groups,
    conjugate_omega,
    lambda_alpha,
    prior_mean_matrix,
)

ImpliedMode = Literal["I", "II", "iw_equivalent"]
_MC_CHUNK = 4_000_000  # draw elements (draws · n · k) simulated at a time


def contemp_arrays(
    scales: WindowScales, prior: Minnesota, n: int, *, scaling: Scaling
) -> ContempPriorArrays:
    """``a_i`` prior arrays (n, n−1), shared by the structural and reduced forms (D-11).

    ``scaling`` is required and must be ``StructStatic.contemp_scaling`` of the composition, since
    the kernels divide ``prec0`` by ``λ_i`` only when it is ``"lambda"`` (DESIGN §3.7.1).
    ``scaling="none"`` (Mode I, SV, ``Constant("ig")``): ``prec0_ij = ι ψ_j/(κ_a ψ_i)``.
    ``scaling="lambda"`` (Mode II ``Constant("iw")`` and ``Constant("iw_equivalent")``): the
    λ-free ``P̃_a = ψ_j/κ_a`` of ``a_ij | λ_i ~ N(0, λ_i κ_a/ψ_j)``. Means are 0; slots ``j ≥ i``
    are inactive.
    """
    _check_scales(scales, n)
    if scaling not in ("none", "lambda"):
        raise ValueError(f"scaling must be 'none' or 'lambda', got {scaling!r}")
    i = np.arange(n)[:, None]
    j = np.arange(n - 1)[None, :]
    cmask = (j < i).astype(np.float64)
    psi_j = scales.psi[:-1][None, :]
    if scaling == "lambda":
        prec = np.broadcast_to(psi_j / prior.contemp, (n, n - 1))
    else:
        prec = (i + 1.0) * psi_j / (prior.contemp * scales.psi[:, None])
    return ContempPriorArrays(
        mean0=np.zeros((n, n - 1)),
        prec0=np.where(cmask > 0, prec, 1.0),
        cmask=cmask,
    )


def structural_arrays(
    scales: WindowScales,
    prior: Minnesota,
    hyper: Mapping[str, float],
    *,
    n: int,
    lags: int,
    n_det: int,
    mode: Literal["I", "II"],
) -> tuple[CoefPriorArrays, SharedPrior]:
    """δ prior (``CoefPriorArrays``, rows ``i``) and the D3 centring columns ``M_c`` (k, n−1).

    ``mode="I"``: ``prec0_i = 1/(σ̄_i² Ω^{(i)})`` (``iw_equivalent`` uses these too and is accepted
    as an alias); ``mode="II"``: the λ-free ``1/Ω^{(i)}``. ``mean0_i = m_i`` (own first lag from the
    transformation codes, 0 elsewhere); ``hyper`` overrides ``lambda``/``alpha`` (EB values).
    """
    if mode not in ("I", "II", "iw_equivalent"):
        raise ValueError(f"mode must be 'I' or 'II', got {mode!r}")
    _check_dims(n, lags, n_det)
    lam, alpha = lambda_alpha(prior, hyper, scales)
    omega = conjugate_omega(scales, prior, lam, alpha, n, lags, n_det)  # (n, k)
    var = omega if mode == "II" else omega * scales.sigma2_bar[:, None]
    M = prior_mean_matrix(scales, n, lags, n_det)
    coef = CoefPriorArrays(
        mean0=np.ascontiguousarray(M.T),
        prec0=1.0 / var,
        group=coefficient_groups(n, lags, n_det),
    )
    return coef, SharedPrior(Mc=np.ascontiguousarray(M[:, : n - 1]))


# --------------------------------------------------------------------------- implied moments


@dataclass(frozen=True)
class ImpliedMoments:
    """Implied reduced-form prior moments of a structural prior (N05 §4.3; R8).

    ``mean`` (k, n) ``E[B]`` (``= M`` exactly under D3 centring, for every ``A``); ``var`` (k, n)
    the marginal prior variances of ``B``; ``sigma_mean`` (n, n) ``E[Σ]`` (``E[Σ_t]`` with SV);
    ``shares`` (n, n) ``s_jm = E[(A⁻¹)²_jm λ_m]/E[Σ_jj]``, the prior share of structural shock ``m``
    in ``Var(u_j)`` (rows sum to 1). ``method`` is ``"closed_form"`` or ``"monte_carlo"``.
    """

    mean: np.ndarray
    var: np.ndarray
    sigma_mean: np.ndarray
    shares: np.ndarray
    method: Literal["closed_form", "monte_carlo"]
    draws: int = 0

    def ratio_to(self, reference_var: np.ndarray) -> np.ndarray:
        """``var / reference_var`` (k, n). Against the NIW marginal ``κ²ψ_j/(l^αψ_m)`` (the
        variances ``1/prec0.T`` of ``reduced_arrays`` with ``cross = 1``) this is the leak ratio
        ``κ_c² + (1 − κ_c²)s_jm`` of N05 §4.3 for the lag coefficients of the default Mode I/II
        priors (1 at ``κ_c = 1``)."""
        ref = np.asarray(reference_var, dtype=np.float64)
        if ref.shape != self.var.shape:
            raise ValueError(f"reference_var must have shape {self.var.shape}, got {ref.shape}")
        return self.var / ref


def implied_reduced_form_moments(
    coef: CoefPriorArrays,
    contemp: ContempPriorArrays,
    shared: SharedPrior,
    vol_mean: np.ndarray,
    *,
    mode: ImpliedMode,
    draws: int = 0,
    rng: np.random.Generator | None = None,
) -> ImpliedMoments:
    """Prior moments of ``B = M + (Δ − M')'A⁻¹'`` and ``Σ = A⁻¹ΛA⁻¹'`` (N05 §4.3, §8).

    ``vol_mean`` (n,) is ``E[λ_i]`` (``ψ_i/ι`` for ``IG(1 + ι/2, ψ_i/2)``; the stationary
    ``E[exp h]`` with SV). ``mode`` names the λ-scaled blocks (``config.scaling_mode``): ``"I"``
    none, ``"II"`` both, ``"iw_equivalent"`` ``a`` only; a scaled block has prior variance
    ``E[λ_i]/P̃₀``. ``C = A⁻¹`` has column ``i`` driven by rows ``> i`` only, which are independent
    of ``λ_i`` and ``δ_i``, so with ``ε_i = δ_i − m_i``

        Var(B_rj) = Σ_i E[C²_ji] E[Var ε_ir],     E[Σ_jl] = Σ_i E[C_ji C_li] E[λ_i],

    and the second moments of each column of ``C`` follow exactly from ``C_ii = 1``,
    ``C_ji = −a_j'C_{<j,i}`` with ``E[a_j a_j'] = diag(Var a_j) + m_a m_a'`` (any contemporaneous
    means). This closed form is exact for the diagonal arrays built here; it reproduces the
    leak ratio ``κ_c² + (1 − κ_c²)s_jm`` and the Mode III ``(1 + κ_a)^{i−1}`` growth of ``E[Σ_ii]``.
    ``draws > 0`` instead estimates the same moments by Monte Carlo over ``(A, δ)`` with each
    block's variance at its ``E[λ]`` value (exact for these second moments, which are linear in
    each row's second moment); ``rng=None`` uses a fixed seed (0) so reports are reproducible.
    """
    if mode not in ("I", "II", "iw_equivalent"):
        raise ValueError(f"mode must be 'I', 'II' or 'iw_equivalent', got {mode!r}")
    mean0 = np.asarray(coef.mean0, dtype=np.float64)
    n, k = mean0.shape
    prec_d = np.asarray(coef.prec0, dtype=np.float64)
    a_mean = np.asarray(contemp.mean0, dtype=np.float64)
    prec_a = np.asarray(contemp.prec0, dtype=np.float64)
    cmask = np.asarray(contemp.cmask, dtype=np.float64)
    Mc = np.asarray(shared.Mc, dtype=np.float64)
    lam = np.asarray(vol_mean, dtype=np.float64)
    if prec_d.shape != (n, k) or a_mean.shape != (n, n - 1) or prec_a.shape != (n, n - 1):
        raise ValueError("coef arrays must be (n, k) and contemp arrays (n, n−1)")
    if lam.shape != (n,) or not np.all(np.isfinite(lam) & (lam > 0)):
        raise ValueError(f"vol_mean must be {n} positive finite values (E[λ_i])")
    if Mc.shape != (k, n - 1) or not np.allclose(Mc, mean0[: n - 1].T, rtol=1e-12, atol=0.0):
        raise ValueError(
            "implied moments need D3 centring: shared.Mc must equal the δ prior means of the "
            "first n−1 equations (M[:, :n−1])"
        )
    var_d = (lam[:, None] if mode == "II" else 1.0) / prec_d  # E[Var ε_i] (n, k)
    var_a = (lam[:, None] if mode != "I" else 1.0) / prec_a * cmask  # E[Var a_ij] (n, n−1)
    a_mean = a_mean * cmask
    if isinstance(draws, bool) or not isinstance(draws, int | np.integer) or draws < 0:
        raise ValueError(f"draws must be a non-negative integer, got {draws!r}")
    if draws == 0:
        S = _column_second_moments(var_a, a_mean)  # S[i] = E[C_{:,i} C_{:,i}'] (n, n, n)
        diag = np.einsum("ijj->ji", S)  # E[C²_ji] (j, i)
        var_B = (diag @ var_d).T  # (k, n): Σ_i E[C²_ji] var_d[i, r]
        sigma = np.einsum("ijl,i->jl", S, lam)
        contrib = diag * lam[None, :]
        method: Literal["closed_form", "monte_carlo"] = "closed_form"
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        var_B, sigma, contrib = _monte_carlo(var_a, a_mean, var_d, lam, int(draws), rng)
        method = "monte_carlo"
    shares = contrib / np.diag(sigma)[:, None]
    return ImpliedMoments(
        mean=np.ascontiguousarray(mean0.T),
        var=var_B,
        sigma_mean=sigma,
        shares=shares,
        method=method,
        draws=int(draws),
    )


def _column_second_moments(var_a: np.ndarray, a_mean: np.ndarray) -> np.ndarray:
    """``S[i] = E[C_{:,i} C_{:,i}']`` for ``C = A⁻¹``, rows of ``A`` independent with means
    ``a_mean`` and variances ``var_a`` (inactive slots 0)."""
    n = var_a.shape[0]
    S = np.zeros((n, n, n))
    for i in range(n):
        Si = S[i]
        Si[i, i] = 1.0
        for j in range(i + 1, n):
            m = a_mean[j, :j]
            block = Si[:j, :j]
            Si[j, :j] = Si[:j, j] = -(m @ block)  # E[C_ji C_li] = −m_j'E[C_{<j,i} C_li]
            Si[j, j] = var_a[j, :j] @ np.diag(block) + m @ block @ m
    return S


def _monte_carlo(
    var_a: np.ndarray,
    a_mean: np.ndarray,
    var_d: np.ndarray,
    lam: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo of the closed-form moments: ``A`` rows ``N(m_a, diag var_a)``,
    ``ε_i ~ N(0, diag var_d_i)``, ``C = A⁻¹`` by forward substitution (``A`` unit lower
    triangular: ``C_j = e_j − Σ_{l<j} a_jl C_l``, row by row, no inverse)."""
    n, k = var_d.shape
    sum_B = np.zeros((k, n))
    sum_sigma = np.zeros((n, n))
    sum_c2 = np.zeros((n, n))
    chunk = max(1, _MC_CHUNK // (n * max(k, n)))
    done = 0
    eye = np.eye(n)
    rows, cols = np.tril_indices(n, -1)
    while done < draws:
        m = min(chunk, draws - done)
        a = np.zeros((m, n, n))
        z = rng.standard_normal((m, rows.size))
        a[:, rows, cols] = a_mean[rows, cols] + np.sqrt(var_a[rows, cols]) * z
        C = np.broadcast_to(eye, (m, n, n)).copy()
        for j in range(1, n):
            C[:, j, :] -= np.einsum("sl,slc->sc", a[:, j, :j], C[:, :j, :])
        eps = rng.standard_normal((m, n, k)) * np.sqrt(var_d)[None]
        dev = np.einsum("sji,sir->srj", C, eps)  # B − M
        sum_B += np.einsum("srj,srj->rj", dev, dev)
        c2 = C**2
        sum_c2 += c2.sum(axis=0)
        sum_sigma += np.einsum("sji,sli,i->jl", C, C, lam)
        done += m
    contrib = sum_c2 / draws * lam[None, :]
    return sum_B / draws, sum_sigma / draws, contrib


__all__ = [
    "ImpliedMoments",
    "contemp_arrays",
    "implied_reduced_form_moments",
    "structural_arrays",
]
