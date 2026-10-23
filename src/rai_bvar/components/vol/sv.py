"""Stochastic-volatility persistent core ``SVCore`` (AR and RW; DESIGN §3.7.4, D-1…D-7; N01 §8, §10;
N02 §3–§9).

**Model** (one equation; ``h`` is the log variance, state index ``j = t + 1`` ↔ data row ``t``,
``j = 0`` is ``h_0``; N02 §2.1). AR: ``h_j = μ + φ(h_{j−1} − μ) + ση_j``, ``h_0 ~ N(μ, σ²/ω₀)`` with
``ω₀ = 1 − φ²`` (``h0="stationary"``) or ``1/B₀`` (``"fixed_b0"``). RW (``rw=True``, N02 §8.1):
``h_j = h_{j−1} + ση_j``, ``h_0 ~ N(m₀, V₀)``. Priors (D-13): ``μ ~ N(b_μ, B_μ)``,
``(φ+1)/2 ~ Beta(a_φ, b_φ)``, ``σ² ~ B_σ·χ²₁`` (⇔ ``σ ~ |N(0, B_σ)|``) or ``σ² ~ IG(c₀, C₀)``.
Observation: ``ẽ_t = exp(h_{t+1}/2)ε_t`` with ``log ẽ_t² = log_e2c_t`` handed in by the vol block.

**One update** (:meth:`SVCore.sample`; order fixed, N02 §3.2, D-5)::

    y*_t  = logaddexp(log ẽ_t², log c)  on observed rows, 0 elsewhere                          D-2
    (c)   s_t | y*, h       inverse CDF on ℓ_tj = logw_j − (y*_t − h_{t+1} − m_j)²/(2v_j)       D-4
    (a)   h | y*, s, θ      AWOL tridiagonal draw in scaled form (AR: h = μ + σx̃; RW: h = σx̃);
                            exact=True: independence MH, log α = w(h′) − w(h)                  D-1
    (b)   (γ, φ) | h, σ²    independence MH, centred cross-products, B₀ = diag(1e12, 1e8)       D-6
          σ² | h, μ, φ      IG(T/2, C_T) independence MH (B_σχ²₁) or exact IG (InvGamma)
    (b**) φ | h̃            NC independence MH; then (μ, σ) | h̃, s, y* by the 2×2 regression;
                            one accept with log α_aux + w(μ′ + σ′h̃) − w(h) (exact mode, E6)
          fold (σ, h̃) ↦ (−σ, −h̃) when σ′ < 0, so h is unchanged                               D-7

with the exact-mode weight (N01 §10.2, Gaussian-``ẽ`` form, valid for any offset ``c ≥ 0``)::

    w(h) = Σ_t m_t [ −½(log 2π + h_{t+1} + exp(log ẽ_t² − h_{t+1})) − log g(y*_t − h_{t+1}) ]

``log g`` at the current ``h`` is the ``logsumexp`` of the indicator logits (reused, DESIGN §10.4
lever 6); the NC move compares against ``w`` of the post-``h``-step state (pilot §3.4: a stale ``w``
is a bug). The centred step does not move ``h`` and ``p̃(s | h, y)`` does not involve θ, so it needs
no correction (N01 §10.3). The indicators are local to one call (never state; DESIGN §2.7).

**AWOL systems** (N02 §5.2–§5.3, §8.1). Writing ``D_t = m_t/v_{s_t}`` (``m_t`` the observation mask):

* AR, non-centred (``Q̃ = σ²Q``, ``h = μ + σx̃``): ``Q̃_00 = ω₀ + φ²``,
  ``Q̃_jj = σ²D_{j−1} + 1 + φ²`` (``1 ≤ j < T_eff``), ``Q̃_{T_eff,T_eff} = σ²D + 1``,
  ``Q̃_{j,j−1} = −φ`` (``j ≤ T_eff``), ``b̃_0 = 0``, ``b̃_j = σD_{j−1}(y*_{j−1} − m_{s} − μ)``.
* RW, centred and scaled (``Q̃ = σ²Q``, ``b̃ = σc``, ``h = σx̃``; exact change of variables that
  avoids ``1/σ²``): ``Q̃_00 = σ²/V₀ + 1``, ``Q̃_jj = σ²D + 2`` (``1 ≤ j < T_eff``),
  ``Q̃_{T_eff,T_eff} = σ²D + 1``, ``Q̃_{j,j−1} = −1``, ``b̃_0 = σm₀/V₀``, ``b̃_j = σD(y* − m_s)``.
  This is a path-graph Laplacian plus ``diag(σ²/V₀, σ²D)``: step (a) factors it in that form
  (:func:`awol_rw_laplacian`), because assembling ``Q̃`` rounds ``σ²D`` away once it is below the
  ulp of 2 (``σ ≲ 1e-8``) and the level direction's pivot ``≈ σ²(1/V₀ + ΣD)`` is then lost
  (WP25 follow-up, the Clark–Ravazzolo ordering-stage crash).
* States ``j > T_eff`` are decoupled dummies (``Q̃ = 1``, off-diagonal 0, ``b̃ = 0``; D-15). They are
  never read by any step (every sum is masked) and are rewritten to ``μ + σz`` (RW: ``σz``) with
  the step's own AWOL normals at the end of each update, so they stay finite and never drift.

**NC regression** (KFS eq. 10; N02 §7): rows ``t`` with ``m_t = 1``,
``y*_t − m_{s_t} = μ + σh̃_{t+1} + N(0, v_{s_t})``, prior ``N((b_μ, 0), diag(B_μ, B_σ))`` (B_σχ²₁ ⇔
``±σ ~ N(0, B_σ)``, an exact Gibbs draw for the auxiliary target). InvGamma: flat ``σ`` prior in
the regression and ``log α_aux = log p_ext(σ′) − log p_ext(σ)``, ``p_ext(s) ∝ |s|^{−2c₀−1}e^{−C₀/s²}``.
RW: ``(h_0, σ)`` with ``h̃_j = (h_j − h_0)/σ`` and prior ``N((m₀, 0), diag(V₀, ·))``.

**Masks.** ``tmask`` defines the states and transitions (``T_eff``); the observation mask
``m_t = tmask_t`` except in ``zero_offset`` parity mode, where rows with ``log ẽ² = −∞`` are dropped
from every observation term (missing observations, N02 §2.4, §10.3 item 7) and counted in
``acc_floor``. Transition sums use ``tmask``; ``T_eff`` enters the σ² shapes.

**Keys** (DESIGN §5.6; one consumer each): ``Sub.S`` indicator uniforms; ``H`` AWOL normals;
``H_ACC``; ``C_GP`` (2 normals); ``C_GP_ACC``; ``C_S2`` (log-gamma); ``C_S2_ACC``; ``NC_PHI``;
``NC_PHI_ACC``; ``NC_MS`` (2 normals); ``NC_MS_ACC``. Accept iff ``log u < log α`` with
``u ∈ [tiny, 1)``. Noise is injected (D-17): :meth:`SVCore.draw_noise` then :meth:`SVCore.sample_core`.

**Failures** are values (D-18): ``StepInfo.ok`` requires a positive AWOL pivot, ``C_T > 0``,
``σ > 0``, ``|φ| < 1`` (AR), a positive-definite NC regression and a finite new state (pilot
§3.1). No jitter.

**Cold start** (DESIGN §5.4 plus the pilot §3.1 amendment): AR ``μ = log(mean_t ẽ²) + N(0, 0.5²)``,
``φ`` = prior mean, ``σ = σ̄·exp(N(0, 0.25²))`` (``σ̄ = √B_σ``, InvGamma: ``√(C₀/(c₀+1))``, the IG mode);
RW ``h_0`` level ``log(mean_t ẽ²)``; then indicators at ``h ≡`` level and **one unconditional AWOL
draw of ``h``**, so ``C_T > 0`` almost surely (``h ≡ μ`` makes σ = 0 absorbing).

Exposed for tests (DESIGN §3.7.4): :func:`indicator_logits`, :func:`awol_system`,
:func:`awol_rw_laplacian`, :func:`w_weight`, :func:`h_log_accept`, :func:`c_step`,
:func:`c_step_log_ratios`, :func:`nc_phi_log_ratio`, :func:`nc_regression`, :func:`nc_step`,
:func:`nc_log_accept`, :func:`nc_fold`, :func:`aux_log_likelihood`; the stages :meth:`SVCore.h_move` and
:meth:`SVCore.nc_move` (one-block tests). Overridable internals (DESIGN §4.9): ``mixture_table``,
``indicator_logits``, ``h_accept``, ``c_step``, ``nc_step``, ``nc_accept``, ``nc_fold``.

**Documented choices** (N02 §12). Joint ``(T+1)``-state AWOL system, not KFS's ``T``-system plus
``h_0 | h_1``; rotated GIS-C order (s first); 2-block centred step with centred cross-products;
measure-preserving fold (stochvol takes ``|σ|`` without the flip, which fails Geweke, N02 §7);
exact correction on by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jr
from jax.scipy.special import gammaln

from rai_bvar.components.vol.mixture import (
    LOG_2PI,
    MixtureTable,
    indicator_logits,
    log_g,
    log_g_from_logits,
    mixture_table,
)
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig, VolStatic
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.linalg import (
    laplacian_precision_sample,
    stable_ar1_regression,
    tridiag_precision_sample,
)
from rai_bvar.engines.mcmc.protocols import KernelBase
from rai_bvar.engines.mcmc.random import categorical_inverse_cdf_lse, loggamma, uniform_tiny
from rai_bvar.engines.mcmc.state import (
    EqPrior,
    GaussianPrec,
    StepInfo,
    SVDraw,
    SVEq,
    SVPrior,
    SVRWDraw,
    SVRWEq,
)

Array = jax.Array
_F64 = jnp.float64

B0_GAMMA = 1e12  # KFS auxiliary prior variance of γ (proposal only; N02 §6, D-6)
B0_PHI = 1e8  # ... and of φ

__all__ = [
    "B0_GAMMA",
    "B0_PHI",
    "CStep",
    "NCMove",
    "SVCore",
    "SVNoise",
    "aux_log_likelihood",
    "awol_rw_laplacian",
    "awol_system",
    "c_gp_log_target",
    "c_step",
    "c_step_log_ratios",
    "draw_indicators",
    "h_log_accept",
    "indicator_logits",
    "log_p_ext_sigma",
    "log_phi_prior",
    "nc_fold",
    "nc_log_accept",
    "nc_phi_log_ratio",
    "nc_regression",
    "nc_step",
    "observation_mask",
    "sigma2_proposal_shape",
    "w_weight",
    "y_star",
]


# --------------------------------------------------------------------------- pytrees


def _pytree[T](cls: type[T]) -> type[T]:
    cls = dataclass(frozen=True, eq=False)(cls)
    names = list(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return jax.tree_util.register_dataclass(cls, data_fields=names, meta_fields=[])


@_pytree
class SVNoise:
    """Injected noise of one SV update (D-17); one key consumer per field (module docstring)."""

    u_s: Array  # f64[T_pad]: indicator uniforms U[0, 1)                      Sub.S
    z_h: Array  # f64[T_pad+1]: AWOL normals                                   Sub.H
    u_h: Array  # f64[]: exact-h accept, U[tiny, 1)                            Sub.H_ACC
    z_gp: Array  # f64[2]: centred (γ, φ) proposal normals                     Sub.C_GP
    u_gp: Array  # f64[]                                                       Sub.C_GP_ACC
    lg_s2: Array  # f64[]: log G, G ~ Gamma(sigma2_proposal_shape, 1)          Sub.C_S2
    u_s2: Array  # f64[]                                                       Sub.C_S2_ACC
    z_phi: Array  # f64[]: NC φ proposal normal                                Sub.NC_PHI
    u_phi: Array  # f64[]                                                      Sub.NC_PHI_ACC
    z_ms: Array  # f64[2]: NC (μ, σ) (RW: (h_0, σ)) normals                    Sub.NC_MS
    u_ms: Array  # f64[]                                                       Sub.NC_MS_ACC


class CStep(NamedTuple):
    """Result of the centred step (:func:`c_step`). RW: ``mu``/``phi`` are passed through."""

    mu: Array
    phi: Array
    sigma: Array
    acc_gp: Array  # bool[]
    acc_s2: Array  # bool[]
    C_T: Array  # f64[]: σ²-conditional scale (> 0 required)
    proposal: Array  # f64[2]: the (γ′, φ′) proposal (AR; zeros for RW), for diagnostics and tests


class NCMove(NamedTuple):
    """Proposal of the non-centred step (:func:`nc_step`), before the accept and the fold."""

    phi: Array  # φ after the NC φ move (kept even if (μ, σ) is rejected)
    acc_phi: Array  # bool[]
    h_tilde: Array  # f64[T_pad+1]: (h − μ)/σ (RW: (h − h_0)/σ)
    mu: Array  # proposed μ′ (RW: h_0′)
    sigma: Array  # proposed σ′ on ℝ (folded afterwards)
    log_alpha_aux: Array  # 0 (B_σχ²₁) or log p_ext(σ′) − log p_ext(σ) (InvGamma)
    info: StepInfo  # 2×2 regression factorisation


# --------------------------------------------------------------------------- small helpers


def _prior(prior_i: EqPrior | SVPrior) -> SVPrior:
    return prior_i if isinstance(prior_i, SVPrior) else prior_i.vol_core


def _masked_sum(mask: Array, x: Array) -> Array:
    return jnp.sum(jnp.where(mask > 0, x, 0.0))


def _take(table_col: Any, s: Array) -> Array:
    return jnp.asarray(table_col, dtype=_F64)[s]


def log_phi_prior(phi: Array, a: Array, b: Array) -> Array:
    """``(a − 1)log(1 + φ) + (b − 1)log(1 − φ)``: ``(φ+1)/2 ~ Beta(a, b)`` up to a constant."""
    return (a - 1.0) * jnp.log1p(phi) + (b - 1.0) * jnp.log1p(-phi)


def log_p_ext_sigma(s: Array, c0: Array, C0: Array) -> Array:
    """``log p_ext(s) = −(2c₀ + 1)log|s| − C₀/s²`` + const: the density of σ implied by
    ``σ² ~ IG(c₀, C₀)``, extended symmetrically to ℝ (N02 §7)."""
    return -(2.0 * c0 + 1.0) * jnp.log(jnp.abs(s)) - C0 / (s * s)


def _two_by_two_draw(K: Array, b: Array, z: Array) -> tuple[Array, StepInfo]:
    """``θ ~ N(K⁻¹b, K⁻¹)`` for a 2×2 SPD ``K`` in closed form with Jacobi scaling (D-14):
    ``s = diag(K)^{−½}``, ``sKs = [[1, ρ], [ρ, 1]] = LLᵀ``, ``θ = s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z)``; pivot
    ``1 − ρ² = (1 − ρ)(1 + ρ)``."""
    s1, s2 = 1.0 / jnp.sqrt(K[0, 0]), 1.0 / jnp.sqrt(K[1, 1])
    rho = K[0, 1] * s1 * s2
    piv = (1.0 - rho) * (1.0 + rho)
    l22 = jnp.sqrt(piv)
    v1 = s1 * b[0]
    v2 = (s2 * b[1] - rho * v1) / l22
    x2 = (v2 + z[1]) / l22
    x1 = v1 + z[0] - rho * x2
    theta = jnp.stack([s1 * x1, s2 * x2])
    ok = jnp.isfinite(piv) & (piv > 0.0) & jnp.all(jnp.isfinite(theta))
    return theta, StepInfo(ok, jnp.where(jnp.isnan(piv), 0.0, piv))


# --------------------------------------------------------------------------- observation, indicators


def observation_mask(log_e2c: Array, tmask: Array, zero_offset: bool) -> tuple[Array, Array]:
    """``(m, n_floor)``: the observation mask (``tmask``; in ``zero_offset`` mode also 0 where
    ``log ẽ² = −∞``, N02 §2.4) and the number of such floor rows (float)."""
    real = tmask > 0
    if not zero_offset:
        return real.astype(_F64), jnp.zeros((), _F64)
    floor = real & jnp.isneginf(log_e2c)
    return (real & ~floor).astype(_F64), jnp.sum(floor.astype(_F64))


def y_star(log_e2c: Array, log_c: Array, obs: Array) -> Array:
    """``y*_t = log(ẽ_t² + c) = logaddexp(log ẽ_t², log c)`` on observed rows, 0 elsewhere (D-2)."""
    on = obs > 0
    return jnp.where(on, jnp.logaddexp(jnp.where(on, log_e2c, 0.0), log_c), 0.0)


def draw_indicators(
    u: Array, r: Array, table: MixtureTable, logits_fn: Any = indicator_logits
) -> tuple[Array, Array]:
    """``(s, log g(r))``: one inverse-CDF draw per row from ``softmax(logits_fn(r, table))`` (D-4)
    and the row ``log g`` from the same ``logsumexp`` (reused by the exact correction)."""
    s, lse = categorical_inverse_cdf_lse(u, logits_fn(r, table))
    return s, lse - 0.5 * LOG_2PI


# --------------------------------------------------------------------------- (a) AWOL


def awol_system(
    y_star: Array,
    s: Array,
    obs: Array,
    T_eff: Array,
    mu: Array,
    phi: Array,
    sigma: Array,
    prior_i: EqPrior | SVPrior,
    table: MixtureTable,
    *,
    h0: str = "stationary",
    rw: bool = False,
) -> tuple[Array, Array, Array]:
    """``(diag, off, b̃)`` of the scaled AWOL system (module docstring; N02 §5.3, §8.1):
    ``x̃ ~ N(Q̃⁻¹b̃, Q̃⁻¹)``, then ``h = μ + σx̃`` (AR) or ``h = σx̃`` (RW). ``off[j] = Q̃[j+1, j]``;
    ``obs`` is the observation mask (``m_t``); states ``j > T_eff`` are decoupled.
    ``Q̃ = σ²Q`` exactly, ``Q`` the precision of ``h | y*, s, θ`` (RW: ``b̃ = σc``)."""
    sp = _prior(prior_i)
    T = y_star.shape[0]
    j = jnp.arange(1, T + 1)
    D = obs / _take(table.v, s)
    resid = y_star - _take(table.m, s)
    s2 = sigma * sigma
    interior, last, live = j < T_eff, j == T_eff, j <= T_eff
    if rw:
        rows = jnp.where(interior, s2 * D + 2.0, jnp.where(last, s2 * D + 1.0, 1.0))
        d0 = s2 / sp.V0 + 1.0
        b0 = sigma * sp.m0 / sp.V0
        off = jnp.where(live, -1.0, 0.0)
        b_rows = jnp.where(live, sigma * D * resid, 0.0)
    else:
        rows = jnp.where(interior, s2 * D + 1.0 + phi * phi, jnp.where(last, s2 * D + 1.0, 1.0))
        d0 = jnp.ones_like(phi) if h0 == "stationary" else 1.0 / sp.B0 + phi * phi
        b0 = jnp.zeros_like(mu)
        off = jnp.where(live, -phi, 0.0)
        b_rows = jnp.where(live, sigma * D * (resid - mu), 0.0)
    diag = jnp.concatenate([jnp.reshape(d0, (1,)), rows])
    b = jnp.concatenate([jnp.reshape(b0, (1,)), b_rows])
    return diag, off, b


def awol_rw_laplacian(
    y_star: Array,
    s: Array,
    obs: Array,
    T_eff: Array,
    sigma: Array,
    prior_i: EqPrior | SVPrior,
    table: MixtureTable,
) -> tuple[Array, Array, Array]:
    """The RW system of :func:`awol_system` as ``(δ, w, b̃)``, ``Q̃ = Lap_w + diag(δ)`` (the form
    :func:`~rai_bvar.engines.mcmc.linalg.laplacian_precision_sample` factors without
    cancellation): unit edges on the live transitions (``w[j−1] = 1``, ``j ≤ T_eff``),
    ``δ_0 = σ²/V₀``, ``δ_j = σ²D_j`` (live) and ``δ_j = 1`` (padded); ``b̃`` is
    :func:`awol_system`'s. The step (a) system for ``rw=True`` (module docstring)."""
    sp = _prior(prior_i)
    T = y_star.shape[0]
    j = jnp.arange(1, T + 1)
    D = obs / _take(table.v, s)
    resid = y_star - _take(table.m, s)
    s2 = sigma * sigma
    live = j <= T_eff
    delta = jnp.concatenate([jnp.reshape(s2 / sp.V0, (1,)), jnp.where(live, s2 * D, 1.0)])
    w = jnp.where(live, 1.0, 0.0)
    b0 = sigma * sp.m0 / sp.V0
    b = jnp.concatenate([jnp.reshape(b0, (1,)), jnp.where(live, sigma * D * resid, 0.0)])
    return delta, w, b


def aux_log_likelihood(
    y_star: Array,
    s: Array,
    obs: Array,
    T_eff: Array,
    mu: Array,
    phi: Array,
    sigma: Array,
    prior_i: EqPrior | SVPrior,
    table: MixtureTable,
    *,
    h0: str = "stationary",
    rw: bool = False,
) -> Array:
    """``log p̃(y* | s, θ)`` with ``h`` integrated out (N02 §5.6; CJ 2009), from the scaled system:
    with ``y′ = y* − m_s − μ`` (RW: ``− m₀``), ``Q̃``, ``b̃′`` its deviation-form right-hand side,

        −½ T_obs log 2π − ½Σ m log v_s − ½Σ m y′²/v_s + ½ log|K̃_x| − ½ log|Q̃| + ½ b̃′ᵀQ̃⁻¹b̃′

    where ``log|K̃_x| = log ω₀`` (AR) or ``log(σ²/V₀)`` (RW) is the prior part. Deterministic oracle
    (T6) and an SBC quantity. RW factors ``Q̃`` in the Laplacian form of step (a)
    (:func:`awol_rw_laplacian`), so it stays exact as ``σ → 0``."""
    sp = _prior(prior_i)
    v_s, m_s = _take(table.v, s), _take(table.m, s)
    level = sp.m0 if rw else mu
    yp = jnp.where(obs > 0, y_star - m_s - level, 0.0)
    D = obs / v_s
    j = jnp.arange(1, y_star.shape[0] + 1)
    b = jnp.concatenate([jnp.zeros((1,), _F64), jnp.where(j <= T_eff, sigma * D * yp, 0.0)])
    if rw:
        delta, w, _ = awol_rw_laplacian(y_star, s, obs, T_eff, sigma, sp, table)
        xm, logdet, _ = laplacian_precision_sample(delta, w, b, jnp.zeros_like(b))
    else:
        diag, off, _ = awol_system(y_star, s, obs, T_eff, mu, phi, sigma, sp, table, h0=h0, rw=rw)
        xm, logdet, _ = tridiag_precision_sample(diag, off, b, jnp.zeros_like(b))
    if rw:
        log_kx = jnp.log(sigma * sigma / sp.V0)
    elif h0 == "stationary":
        log_kx = jnp.log1p(-phi) + jnp.log1p(phi)
    else:
        log_kx = -jnp.log(sp.B0)
    T_obs = jnp.sum(obs)
    quad = _masked_sum(obs, yp * yp / v_s)
    return (
        -0.5 * T_obs * LOG_2PI
        - 0.5 * _masked_sum(obs, jnp.log(v_s))
        - 0.5 * quad
        + 0.5 * log_kx
        - 0.5 * logdet
        + 0.5 * jnp.dot(b, xm)
    )


# --------------------------------------------------------------------------- exact correction


def w_weight(
    h: Array,
    log_e2c: Array,
    y_star: Array,
    obs: Array,
    table: MixtureTable,
    log_g_cur: Array | None = None,
) -> Array:
    """``w(h) = Σ_t m_t[−½(log 2π + h_{t+1} + exp(log ẽ_t² − h_{t+1})) − log g(y*_t − h_{t+1})]``
    (N01 §10.2). ``log_g_cur`` reuses ``log g`` at this ``h`` (from the indicator step)."""
    on = obs > 0
    hd = jnp.where(on, h[1:], 0.0)
    lec = jnp.where(on, log_e2c, 0.0)
    lg = log_g(y_star - hd, table) if log_g_cur is None else log_g_cur
    val = -0.5 * (LOG_2PI + hd + jnp.exp(lec - hd)) - lg
    return jnp.sum(jnp.where(on, val, 0.0))


def h_log_accept(w_prop: Array, w_cur: Array) -> Array:
    """Exact-``h`` log acceptance ``log α_h = w(h′) − w(h)`` (DNP 2015 Alg. 3; N01 §10.2)."""
    return w_prop - w_cur


def nc_log_accept(log_alpha_aux: Array, w_prop: Array, w_cur: Array, exact: bool) -> Array:
    """NC ``(μ, σ)`` log acceptance: ``log α_aux + w(μ′ + σ′h̃) − w(h)`` in exact mode (one
    accept/reject, N01 §10.3, E6), else ``log α_aux``. ``w_cur`` is ``w`` of the post-``h``-step
    state (the centred step does not move ``h``)."""
    return log_alpha_aux + (w_prop - w_cur) if exact else log_alpha_aux


def nc_fold(sigma: Array, h_tilde: Array) -> tuple[Array, Array]:
    """Measure-preserving fold ``(σ, h̃) ↦ (−σ, −h̃)`` when ``σ < 0`` (D-7; N02 §7): ``μ + σh̃`` is
    unchanged bit for bit."""
    neg = sigma < 0.0
    return jnp.where(neg, -sigma, sigma), jnp.where(neg, -h_tilde, h_tilde)


# --------------------------------------------------------------------------- (b) centred step


def sigma2_proposal_shape(T_eff: Array, prior_i: EqPrior | SVPrior, vol: VolStatic) -> Array:
    """Gamma shape of the σ² step: AR ``T/2`` (B_σχ²₁ proposal) or ``c₀ + (T+1)/2`` (exact IG);
    RW ``(T−1)/2`` or ``c₀ + T/2`` (N02 §6.1 steps 3–5, §8.1)."""
    sp = _prior(prior_i)
    T = jnp.asarray(T_eff).astype(_F64)
    chi2 = vol.sigma2_prior == "scaled_chi2"
    if vol.rw:
        return 0.5 * (T - 1.0) if chi2 else sp.c0 + 0.5 * T
    return 0.5 * T if chi2 else sp.c0 + 0.5 * (T + 1.0)


def c_gp_log_target(
    gamma: Array, phi: Array, h0: Array, sigma2: Array, prior_i: EqPrior | SVPrior, *, h0_kind: str
) -> Array:
    """Centred ``(γ, φ)`` independence-MH log target over the auxiliary proposal (N02 §6.1 step 2):
    ``ℓ₀(μ, φ, σ²) + log p(γ | φ) + log p_φ(φ) + (γ²/B₀¹¹ + φ²/B₀²²)/(2σ²)``, ``μ = γ/(1 − φ)``,
    with ``log p(γ | φ) = −log(1 − φ) − (μ − b_μ)²/(2B_μ)`` (Jacobian of ``μ ↦ γ``) and
    ``ℓ₀ = log N(h_0; μ, σ²/ω₀)``. Valid for ``|φ| < 1``."""
    sp = _prior(prior_i)
    one_m = 1.0 - phi
    mu = gamma / one_m
    if h0_kind == "stationary":
        om = one_m * (1.0 + phi)
        l0 = 0.5 * (jnp.log(om) - jnp.log(sigma2)) - 0.5 * om * (h0 - mu) ** 2 / sigma2
    else:
        var0 = sp.B0 * sigma2
        l0 = -0.5 * jnp.log(var0) - 0.5 * (h0 - mu) ** 2 / var0
    lgam = -jnp.log(one_m) - 0.5 * (mu - sp.b_mu) ** 2 / sp.B_mu
    aux = 0.5 * (gamma * gamma / B0_GAMMA + phi * phi / B0_PHI) / sigma2
    return l0 + lgam + log_phi_prior(phi, sp.a_phi, sp.b_phi) + aux


def _ct_ar(h: Array, mu: Array, phi: Array, tmask: Array, prior: SVPrior, h0_kind: str) -> Array:
    res = h[1:] - mu - phi * (h[:-1] - mu)
    om = (1.0 - phi) * (1.0 + phi) if h0_kind == "stationary" else 1.0 / prior.B0
    return 0.5 * (_masked_sum(tmask, res * res) + om * (h[0] - mu) ** 2)


def _ct_rw(h: Array, tmask: Array) -> Array:
    d = h[1:] - h[:-1]
    return 0.5 * _masked_sum(tmask, d * d)


def c_step_log_ratios(
    h: Array,
    mu: Array,
    phi: Array,
    sigma: Array,
    gamma_p: Array,
    phi_p: Array,
    sigma2_p: Array,
    prior_i: EqPrior | SVPrior,
    *,
    h0_kind: str = "stationary",
) -> tuple[Array, Array]:
    """``(log R_γφ, log R_σ²)`` of the centred MH moves for given proposals (B_σχ²₁ prior): the
    ``(γ, φ)`` ratio at the current σ² (``−∞`` if ``|φ′| ≥ 1``) and ``(σ² − σ²′)/(2B_σ)``."""
    sp = _prior(prior_i)
    s2 = sigma * sigma
    valid = jnp.abs(phi_p) < 1.0
    phi_s = jnp.where(valid, phi_p, 0.0)
    lr = c_gp_log_target(gamma_p, phi_s, h[0], s2, sp, h0_kind=h0_kind) - c_gp_log_target(
        mu * (1.0 - phi), phi, h[0], s2, sp, h0_kind=h0_kind
    )
    return jnp.where(valid, lr, -jnp.inf), (s2 - sigma2_p) / (2.0 * sp.B_sigma)


def c_step(
    h: Array,
    mu: Array,
    phi: Array,
    sigma: Array,
    tmask: Array,
    T_eff: Array,
    prior_i: EqPrior | SVPrior,
    noise: SVNoise,
    *,
    vol: VolStatic,
) -> CStep:
    """Centred θ step given ``h`` (N02 §6.1, D-6; RW: σ² only, N02 §8.1).

    AR: ``(γ′, φ′) = b_T + σ·chol(B_T) z`` from :func:`~rai_bvar.engines.mcmc.linalg.stable_ar1_regression`
    on the masked transitions (centred cross-products, ``B₀ = diag(1e12, 1e8)``), accepted with
    :func:`c_step_log_ratios`; then with ``C_T = ½[Σ m(h_{j} − μ − φ(h_{j−1} − μ))² + ω₀(h_0 − μ)²]``:
    B_σχ²₁: ``σ²′ = C_T/G``, ``G ~ Gamma(T/2)``, ``log R = (σ² − σ²′)/(2B_σ)``; InvGamma:
    ``σ²′ ~ IG(c₀ + (T+1)/2, C₀ + C_T)`` (Gibbs). RW: ``C_T = ½Σ m(h_j − h_{j−1})²`` with shapes
    ``(T−1)/2`` (proposal) or ``c₀ + T/2`` (Gibbs)."""
    sp = _prior(prior_i)
    s2 = sigma * sigma
    if vol.rw:
        mu1, phi1 = mu, phi
        acc_gp = jnp.zeros((), bool)
        prop = jnp.zeros((2,), _F64)
        C_T = _ct_rw(h, tmask)
    else:
        bT, cB = stable_ar1_regression(h[:-1], h[1:], tmask, 1.0 / B0_GAMMA, 1.0 / B0_PHI)
        prop = bT + sigma * (cB @ noise.z_gp)
        gam_p, phi_p = prop[0], prop[1]
        lr_gp, _ = c_step_log_ratios(h, mu, phi, sigma, gam_p, phi_p, s2, sp, h0_kind=vol.h0)
        acc_gp = jnp.log(noise.u_gp) < lr_gp  # never for |φ′| ≥ 1 (log R = −∞)
        phi1 = jnp.where(acc_gp, phi_p, phi)
        mu1 = jnp.where(acc_gp, gam_p / (1.0 - phi1), mu)
        C_T = _ct_ar(h, mu1, phi1, tmask, sp, vol.h0)
    if vol.sigma2_prior == "scaled_chi2":
        log_s2p = jnp.log(C_T) - noise.lg_s2
        s2p = jnp.exp(log_s2p)
        acc_s2 = jnp.log(noise.u_s2) < (s2 - s2p) / (2.0 * sp.B_sigma)
    else:
        log_s2p = jnp.log(sp.C0 + C_T) - noise.lg_s2
        acc_s2 = jnp.ones((), bool)
    sigma1 = jnp.where(acc_s2, jnp.exp(0.5 * log_s2p), sigma)
    return CStep(mu1, phi1, sigma1, acc_gp, acc_s2, C_T, prop)


# --------------------------------------------------------------------------- (b**) non-centred step


def nc_regression(
    h_tilde: Array,
    y_star: Array,
    m_s: Array,
    v_s: Array,
    obs: Array,
    prior_i: EqPrior | SVPrior,
    *,
    vol: VolStatic,
) -> GaussianPrec:
    """Exact Gaussian conditional of ``(μ, σ)`` (RW: ``(h_0, σ)``) in the NC regression (KFS eq. 10;
    N02 §7): weights ``w_t = m_t/v_{s_t}``, regressors ``(1, h̃_{t+1})``, response ``y*_t − m_{s_t}``;
    ``K = P₀ + Σ w XXᵀ``, ``b = P₀β₀ + Σ w X(y* − m_s)`` with ``P₀ = diag(1/B_μ, 1/B_σ)``,
    ``β₀ = (b_μ, 0)`` (RW: ``1/V₀``, ``m₀``); InvGamma: the σ precision is 0 (flat)."""
    sp = _prior(prior_i)
    on = obs > 0
    w = jnp.where(on, obs / v_s, 0.0)
    x = jnp.where(on, h_tilde[1:], 0.0)
    yt = jnp.where(on, y_star - m_s, 0.0)
    if vol.rw:
        p1, m1 = 1.0 / sp.V0, sp.m0
    else:
        p1, m1 = 1.0 / sp.B_mu, sp.b_mu
    p2 = 1.0 / sp.B_sigma if vol.sigma2_prior == "scaled_chi2" else jnp.zeros_like(p1)
    k11 = p1 + jnp.sum(w)
    k12 = jnp.sum(w * x)
    k22 = p2 + jnp.sum(w * x * x)
    K = jnp.stack([jnp.stack([k11, k12]), jnp.stack([k12, k22])])
    b = jnp.stack([p1 * m1 + jnp.sum(w * yt), jnp.sum(w * x * yt)])
    return GaussianPrec(K, b)


def nc_phi_log_ratio(
    h_tilde: Array,
    phi: Array,
    phi_p: Array,
    prior_i: EqPrior | SVPrior,
    *,
    h0_kind: str = "stationary",
) -> Array:
    """NC ``φ`` independence-MH log ratio for the proposal ``N(Σh̃_{j−1}h̃_j/Σh̃²_{j−1}, 1/Σh̃²_{j−1})``
    (flat auxiliary prior; N02 §7): ``log N(h̃_0; 0, 1/ω₀(φ′)) + log p_φ(φ′) − [same at φ]``; the
    ``h̃_0`` term is φ-free (cancels) for ``h0="fixed_b0"``; ``−∞`` if ``|φ′| ≥ 1``."""
    sp = _prior(prior_i)
    valid = jnp.abs(phi_p) < 1.0

    def target(ph: Array) -> Array:
        lt = log_phi_prior(ph, sp.a_phi, sp.b_phi)
        if h0_kind == "stationary":
            om = (1.0 - ph) * (1.0 + ph)
            lt = lt + 0.5 * jnp.log(om) - 0.5 * om * h_tilde[0] ** 2
        return lt

    lr = target(jnp.where(valid, phi_p, 0.0)) - target(phi)
    return jnp.where(valid, lr, -jnp.inf)


def nc_step(
    h: Array,
    mu: Array,
    phi: Array,
    sigma: Array,
    y_star: Array,
    m_s: Array,
    v_s: Array,
    obs: Array,
    tmask: Array,
    prior_i: EqPrior | SVPrior,
    noise: SVNoise,
    *,
    vol: VolStatic,
) -> NCMove:
    """The non-centred (ASIS) proposal given the centred-step output (N02 §7; KFS §2.5).

    AR: ``h̃ = (h − μ)/σ``; ``φ′ ~ N(Σh̃_{j−1}h̃_j/Σh̃²_{j−1}, 1/Σh̃²_{j−1})`` over the masked
    transitions, accepted with ``log N(h̃_0; 0, 1/ω₀(φ)) + log p_φ(φ)`` (``−∞`` if ``|φ′| ≥ 1``);
    then ``(μ′, σ′)`` from :func:`nc_regression`. RW: ``h̃ = (h − h_0)/σ`` and ``(h_0′, σ′)``.
    ``log_alpha_aux`` is 0 (B_σχ²₁: exact Gibbs) or the InvGamma ``p_ext`` ratio."""
    sp = _prior(prior_i)
    if vol.rw:
        ht = (h - h[0]) / sigma
        phi1, acc_phi = phi, jnp.zeros((), bool)
    else:
        ht = (h - mu) / sigma
        on = tmask > 0
        x = jnp.where(on, ht[:-1], 0.0)
        z = jnp.where(on, ht[1:], 0.0)
        sxx, sxy = jnp.sum(x * x), jnp.sum(x * z)
        phi_p = sxy / sxx + noise.z_phi / jnp.sqrt(sxx)
        lr = nc_phi_log_ratio(ht, phi, phi_p, sp, h0_kind=vol.h0)
        acc_phi = jnp.log(noise.u_phi) < lr
        phi1 = jnp.where(acc_phi, phi_p, phi)
    gp = nc_regression(ht, y_star, m_s, v_s, obs, sp, vol=vol)
    theta, info = _two_by_two_draw(gp.K, gp.b, noise.z_ms)
    mu_p, sig_p = theta[0], theta[1]
    if vol.sigma2_prior == "scaled_chi2":
        laux = jnp.zeros_like(sig_p)
    else:
        laux = log_p_ext_sigma(sig_p, sp.c0, sp.C0) - log_p_ext_sigma(sigma, sp.c0, sp.C0)
    return NCMove(phi1, acc_phi, ht, mu_p, sig_p, laux, info)


# --------------------------------------------------------------------------- the kernel


@dataclass(frozen=True)
class SVCore(KernelBase):
    """SV persistent core (``PersistentCore``; DESIGN §3.5, §3.7.4) for ``VolStatic.persistent ==
    "sv"``: AR (``rw=False``, state ``SVEq``) or RW (``rw=True``, state ``SVRWEq``). Statics read
    from ``static``: ``rw``, ``h0``, ``sigma2_prior``, ``mixture``, ``exact``, ``asis``,
    ``zero_offset``. One equation per call (batched by the program's ``eq_map``).

    Construction follows the vol-kernel convention ``SVCore(vol_static, T_pad)`` (``T_pad``
    optional; when given, inputs are checked against it at trace time) or
    :meth:`from_static`."""

    static: VolStatic
    T_pad: int | None = None

    # overridable internals (DESIGN §4.9): called through self.impl(name)
    mixture_table = staticmethod(mixture_table)
    indicator_logits = staticmethod(indicator_logits)
    h_accept = staticmethod(h_log_accept)
    c_step = staticmethod(c_step)
    nc_step = staticmethod(nc_step)
    nc_accept = staticmethod(nc_log_accept)
    nc_fold = staticmethod(nc_fold)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.static, VolStatic):
            raise TypeError(f"SVCore needs a VolStatic, got {type(self.static).__name__}")
        if self.static.persistent != "sv":
            raise ValueError(f"SVCore needs persistent='sv', got {self.static.persistent!r}")
        if self.T_pad is not None:
            if isinstance(self.T_pad, bool) or not isinstance(self.T_pad, int) or self.T_pad < 1:
                raise ValueError(f"T_pad must be a positive int or None, got {self.T_pad!r}")

    @classmethod
    def from_static(cls, static: StaticConfig | VolStatic, **kw: Any) -> SVCore:
        """``SVCore(static.vol, static.shapes.T_pad)`` (or ``SVCore(static)`` for a ``VolStatic``)."""
        if isinstance(static, StaticConfig):
            return cls(static.vol, static.shapes.T_pad, **kw)
        return cls(static, **kw)

    def _check_rows(self, rows: int) -> None:
        if self.T_pad is not None and rows != self.T_pad:
            raise ValueError(f"SVCore built for T_pad={self.T_pad}, got {rows} data rows")

    @property
    def rw(self) -> bool:
        return self.static.rw

    def table(self) -> MixtureTable:
        """The mixture table in use (``mixture_table`` override applied)."""
        return self.impl("mixture_table")(self.static.mixture)

    def _log_g(self, r: Array, table: MixtureTable) -> Array:
        return log_g_from_logits(self.impl("indicator_logits")(r, table))

    # ----------------------------------------------------------------- noise
    def draw_noise(self, key: Key, T_pad: int, T_eff: Array, prior_i: EqPrior | SVPrior) -> SVNoise:
        """Every random number of one update, one sub-key per consumer (module docstring)."""
        k = lambda sub: sub_key(key, sub)  # noqa: E731
        shape = sigma2_proposal_shape(T_eff, prior_i, self.static)
        return SVNoise(
            u_s=jr.uniform(k(Sub.S), (T_pad,), dtype=_F64),
            z_h=jr.normal(k(Sub.H), (T_pad + 1,), dtype=_F64),
            u_h=uniform_tiny(k(Sub.H_ACC)),
            z_gp=jr.normal(k(Sub.C_GP), (2,), dtype=_F64),
            u_gp=uniform_tiny(k(Sub.C_GP_ACC)),
            lg_s2=loggamma(k(Sub.C_S2), shape),
            u_s2=uniform_tiny(k(Sub.C_S2_ACC)),
            z_phi=jr.normal(k(Sub.NC_PHI), (), dtype=_F64),
            u_phi=uniform_tiny(k(Sub.NC_PHI_ACC)),
            z_ms=jr.normal(k(Sub.NC_MS), (2,), dtype=_F64),
            u_ms=uniform_tiny(k(Sub.NC_MS_ACC)),
        )

    # ----------------------------------------------------------------- stages
    def h_move(
        self,
        noise: SVNoise,
        h: Array,
        mu: Array,
        phi: Array,
        sigma: Array,
        log_e2c: Array,
        ys: Array,
        obs: Array,
        T_eff: Array,
        prior_i: EqPrior | SVPrior,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Steps (c) and (a): ``(h, acc_h, w(h), m_s, v_s, min_pivot)`` — indicators, AWOL
        proposal and (exact mode) the MH accept; ``w`` is 0 when ``exact=False``."""
        vol, table = self.static, self.table()
        s, lg_cur = draw_indicators(noise.u_s, ys - h[1:], table, self.impl("indicator_logits"))
        if vol.rw:
            delta, w, b = awol_rw_laplacian(ys, s, obs, T_eff, sigma, prior_i, table)
            x, _, piv = laplacian_precision_sample(delta, w, b, noise.z_h)
        else:
            diag, off, b = awol_system(
                ys, s, obs, T_eff, mu, phi, sigma, prior_i, table, h0=vol.h0, rw=vol.rw
            )
            x, _, piv = tridiag_precision_sample(diag, off, b, noise.z_h)
        h_prop = sigma * x if vol.rw else mu + sigma * x
        if vol.exact:
            w_cur = w_weight(h, log_e2c, ys, obs, table, lg_cur)
            lg_prop = self._log_g(ys - jnp.where(obs > 0, h_prop[1:], 0.0), table)
            w_prop = w_weight(h_prop, log_e2c, ys, obs, table, lg_prop)
            acc = jnp.log(noise.u_h) < self.impl("h_accept")(w_prop, w_cur)
            h = jnp.where(acc, h_prop, h)
            w_h = jnp.where(acc, w_prop, w_cur)
        else:
            acc, h, w_h = jnp.ones((), bool), h_prop, jnp.zeros((), _F64)
        return h, acc, w_h, _take(table.m, s), _take(table.v, s), piv

    def nc_move(
        self,
        noise: SVNoise,
        h: Array,
        mu: Array,
        phi: Array,
        sigma: Array,
        w_h: Array,
        log_e2c: Array,
        ys: Array,
        m_s: Array,
        v_s: Array,
        obs: Array,
        tmask: Array,
        prior_i: EqPrior | SVPrior,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, StepInfo]:
        """Step (b**)–(b***): ``(h, μ, φ, σ, acc_phi, acc_ms, folded, info)``. One accept/reject of
        ``(μ′, σ′)`` (exact mode adds ``w(μ′ + σ′h̃) − w_h``); on accept fold and map back
        ``h = μ + σh̃`` (RW: ``h_0 + σh̃``); on reject keep ``(μ, σ, h)`` and the φ draw."""
        vol, table = self.static, self.table()
        mv = self.impl("nc_step")(
            h, mu, phi, sigma, ys, m_s, v_s, obs, tmask, prior_i, noise, vol=vol
        )
        h_prop = mv.mu + mv.sigma * mv.h_tilde
        if vol.exact:
            lg = self._log_g(ys - jnp.where(obs > 0, h_prop[1:], 0.0), table)
            w_prop = w_weight(h_prop, log_e2c, ys, obs, table, lg)
        else:
            w_prop = jnp.zeros((), _F64)
        la = self.impl("nc_accept")(mv.log_alpha_aux, w_prop, w_h, vol.exact)
        acc = jnp.log(noise.u_ms) < la
        sig_f, ht_f = self.impl("nc_fold")(mv.sigma, mv.h_tilde)
        folded = acc & (mv.sigma < 0.0)
        mu1 = jnp.where(acc, mv.mu, mu)
        sigma1 = jnp.where(acc, sig_f, sigma)
        h1 = jnp.where(acc, mv.mu + sig_f * ht_f, h)
        return h1, mu1, mv.phi, sigma1, mv.acc_phi, acc, folded, mv.info

    # ----------------------------------------------------------------- PersistentCore
    def sample_core(
        self,
        noise: SVNoise,
        cs: SVEq | SVRWEq,
        log_e2c: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior | SVPrior,
        *,
        obs: Array | None = None,
    ) -> tuple[SVEq | SVRWEq, StepInfo]:
        """One SV update with injected noise (module docstring). ``obs`` overrides the observation
        mask (tests: prior recovery with every observation masked)."""
        vol = self.static
        sp = _prior(prior_i)
        self._check_rows(log_e2c.shape[0])
        if obs is None:
            obs, n_floor = observation_mask(log_e2c, tmask, vol.zero_offset)
        else:
            n_floor = jnp.zeros((), _F64)
        ys = y_star(log_e2c, sp.log_c, obs)
        rw = vol.rw
        mu = jnp.zeros((), _F64) if rw else cs.mu
        phi = jnp.ones((), _F64) if rw else cs.phi
        h, acc_h, w_h, m_s, v_s, piv = self.h_move(
            noise, cs.h, mu, phi, cs.sigma, log_e2c, ys, obs, T_eff, sp
        )
        cst = self.impl("c_step")(h, mu, phi, cs.sigma, tmask, T_eff, sp, noise, vol=vol)
        mu, phi, sigma = cst.mu, cst.phi, cst.sigma
        info = StepInfo(jnp.isfinite(piv) & (piv > 0.0), piv)
        acc_phi = acc_ms = folded = jnp.zeros((), bool)
        if vol.asis:
            level = h[0] if rw else mu
            h, level, phi, sigma, acc_phi, acc_ms, folded, nc_info = self.nc_move(
                noise, h, level, phi, sigma, w_h, log_e2c, ys, m_s, v_s, obs, tmask, sp
            )
            info = info.merge(nc_info)
            mu = mu if rw else level
        # padded states: decoupled dummies μ + σz (RW: σz) from this step's AWOL normals
        live = jnp.arange(h.shape[0]) <= T_eff
        h = jnp.where(live, h, sigma * noise.z_h if rw else mu + sigma * noise.z_h)
        state_ok = (
            jnp.all(jnp.isfinite(h))
            & jnp.isfinite(sigma)
            & (sigma > 0.0)
            & jnp.isfinite(cst.C_T)
            & (cst.C_T > 0.0)
        )
        if not rw:
            state_ok = state_ok & jnp.isfinite(mu) & (jnp.abs(phi) < 1.0)
        info = StepInfo(info.ok & state_ok, info.min_pivot)
        f = lambda b: b.astype(_F64)  # noqa: E731
        run = (cs.acc_rej_run + 1) * (1 - acc_h.astype(jnp.int32))
        common = dict(
            h=h,
            sigma=sigma,
            acc_h=cs.acc_h + f(acc_h),
            acc_cs2=cs.acc_cs2 + f(cst.acc_s2),
            acc_ncms=cs.acc_ncms + f(acc_ms),
            acc_fold=cs.acc_fold + f(folded),
            acc_floor=cs.acc_floor + n_floor,
            acc_rej_run=run.astype(jnp.int32),
            acc_rej_run_max=jnp.maximum(cs.acc_rej_run_max, run).astype(jnp.int32),
            rb_h_sum=cs.rb_h_sum + h,
            rb_h_sq=cs.rb_h_sq + h * h,
        )
        if rw:
            return SVRWEq(**common), info
        return SVEq(
            mu=mu,
            phi=phi,
            acc_cgp=cs.acc_cgp + f(cst.acc_gp),
            acc_ncphi=cs.acc_ncphi + f(acc_phi),
            **common,
        ), info

    def sample(
        self,
        key: Key,
        cs: SVEq | SVRWEq,
        log_e2c: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior | SVPrior,
        cpl: Any = None,
    ) -> tuple[SVEq | SVRWEq, StepInfo]:
        """``PersistentCore.sample``: :meth:`draw_noise` then :meth:`sample_core`. ``log_e2c`` is
        ``log ẽ²`` after the transitory step (0 on padded rows). ``cpl`` is ignored (coupling is a
        Constant-core channel, D-30)."""
        noise = self.draw_noise(key, log_e2c.shape[0], T_eff, prior_i)
        return self.sample_core(noise, cs, log_e2c, tmask, T_eff, prior_i)

    def init(
        self, key: Key, log_e2: Array, tmask: Array, T_eff: Array, prior_i: EqPrior | SVPrior
    ) -> SVEq | SVRWEq:
        """Cold start (DESIGN §5.4 + pilot §3.1; module docstring). Keys: ``Sub.S`` indicators,
        ``Sub.H`` AWOL normals, ``Sub.NC_MS`` the two jitter normals (μ, log σ)."""
        vol, table = self.static, self.table()
        sp = _prior(prior_i)
        T_pad = log_e2.shape[0]
        self._check_rows(T_pad)
        obs, _ = observation_mask(log_e2, tmask, vol.zero_offset)
        ys = y_star(log_e2, sp.log_c, obs)
        n_obs = jnp.sum(obs)
        lse = jax.scipy.special.logsumexp(jnp.where(obs > 0, ys, -jnp.inf))
        lm = lse - jnp.log(jnp.maximum(n_obs, 1.0))
        fallback = sp.m0 if vol.rw else sp.b_mu
        lm = jnp.where((n_obs > 0) & jnp.isfinite(lm), lm, fallback)
        zj = jr.normal(sub_key(key, Sub.NC_MS), (2,), dtype=_F64)
        if vol.sigma2_prior == "scaled_chi2":
            sbar = jnp.sqrt(sp.B_sigma)
        else:
            sbar = jnp.sqrt(sp.C0 / (sp.c0 + 1.0))
        sigma = sbar * jnp.exp(0.25 * zj[1])
        if vol.rw:
            mu, phi, level = jnp.zeros((), _F64), jnp.ones((), _F64), lm
        else:
            mu = lm + 0.5 * zj[0]
            phi = 2.0 * sp.a_phi / (sp.a_phi + sp.b_phi) - 1.0
            level = mu
        u = jr.uniform(sub_key(key, Sub.S), (T_pad,), dtype=_F64)
        s, _ = draw_indicators(u, ys - level, table, self.impl("indicator_logits"))
        diag, off, b = awol_system(
            ys, s, obs, T_eff, mu, phi, sigma, sp, table, h0=vol.h0, rw=vol.rw
        )
        z = jr.normal(sub_key(key, Sub.H), (T_pad + 1,), dtype=_F64)
        x, _, _ = tridiag_precision_sample(diag, off, b, z)
        h = sigma * x if vol.rw else mu + sigma * x
        zf = jnp.zeros((), _F64)
        zi = jnp.zeros((), jnp.int32)
        zh = jnp.zeros((T_pad + 1,), _F64)
        common = dict(
            h=h, sigma=sigma, acc_h=zf, acc_cs2=zf, acc_ncms=zf, acc_fold=zf, acc_floor=zf,
            acc_rej_run=zi, acc_rej_run_max=zi, rb_h_sum=zh, rb_h_sq=zh,
        )  # fmt: skip
        if vol.rw:
            return SVRWEq(**common)
        return SVEq(mu=mu, phi=phi, acc_cgp=zf, acc_ncphi=zf, **common)

    def log_var(self, cs: SVEq | SVRWEq) -> Array:
        """Persistent log variance per data row: ``h[1:]`` (f64[T_pad])."""
        return cs.h[1:]

    def log_scale(self, cs: SVEq | SVRWEq) -> Array:
        """0: SV never feeds the coupling channel (D-30)."""
        return jnp.zeros((), _F64)

    def extract(self, cs: SVEq | SVRWEq, T_eff: Array) -> SVDraw | SVRWDraw:
        """``h_T = h[T_eff]`` (dynamic index, never ``h[-1]``) and the parameters."""
        h_T = lax.dynamic_index_in_dim(cs.h, jnp.asarray(T_eff, jnp.int32), keepdims=False)
        if self.static.rw:
            return SVRWDraw(h_T=h_T, sigma=cs.sigma)
        return SVDraw(h_T=h_T, mu=cs.mu, phi=cs.phi, sigma=cs.sigma)

    def ahead(self, key: Key, cd: SVDraw | SVRWDraw, H: int) -> Array:
        """``h_{T+j} = μ + φ(h_{T+j−1} − μ) + ση_j`` (RW: ``h_{T+j−1} + ση_j``), ``j = 1..H``
        (f64[H]; one normal vector from ``key``)."""
        eta = jr.normal(key, (int(H),), dtype=_F64)
        if self.static.rw:
            return cd.h_T + cd.sigma * jnp.cumsum(eta)

        def step(prev: Array, e: Array) -> tuple[Array, Array]:
            nxt = cd.mu + cd.phi * (prev - cd.mu) + cd.sigma * e
            return nxt, nxt

        return lax.scan(step, cd.h_T, eta)[1]

    def transition(self, key: Key, log_lam_prev: Array, cd: SVDraw | SVRWDraw) -> Array:
        """One persistent transition from ``log_lam_prev`` (the particle-filter hook)."""
        e = jr.normal(key, (), dtype=_F64)
        if self.static.rw:
            return log_lam_prev + cd.sigma * e
        return cd.mu + cd.phi * (log_lam_prev - cd.mu) + cd.sigma * e

    def log_prior(self, cs: SVEq | SVRWEq, prior_i: EqPrior | SVPrior, tmask: Array) -> Array:
        """``log p(θ, h_{0:T_eff})`` w.r.t. Lebesgue measure on ``(μ, φ, σ, h)`` (RW: ``(σ, h)``):
        ``N(μ)``, Beta on ``(φ+1)/2`` with its ``½`` Jacobian, the density of ``σ > 0``
        (``|N(0, B_σ)|`` or ``IG`` on σ² times ``2σ``), ``h_0`` and the masked transitions."""
        vol = self.static
        sp = _prior(prior_i)
        h, sigma = cs.h, cs.sigma

        def lnorm(x: Array, m: Array, v: Array) -> Array:
            return -0.5 * (LOG_2PI + jnp.log(v) + (x - m) ** 2 / v)

        if vol.sigma2_prior == "scaled_chi2":
            lp = math.log(2.0) + lnorm(sigma, 0.0, sp.B_sigma)
        else:
            s2 = sigma * sigma
            lp = (
                sp.c0 * jnp.log(sp.C0)
                - gammaln(sp.c0)
                - (sp.c0 + 1.0) * jnp.log(s2)
                - sp.C0 / s2
                + jnp.log(2.0 * sigma)
            )
        if vol.rw:
            lp = lp + lnorm(h[0], sp.m0, sp.V0)
            trans = lnorm(h[1:], h[:-1], sigma * sigma)
        else:
            mu, phi = cs.mu, cs.phi
            lp = lp + lnorm(mu, sp.b_mu, sp.B_mu)
            log_b = gammaln(sp.a_phi) + gammaln(sp.b_phi) - gammaln(sp.a_phi + sp.b_phi)
            lp = lp + (  # jax's betaln is ~6e-7 off in float64 at (10, 2) (jax 0.12): not used
                (sp.a_phi - 1.0) * (jnp.log1p(phi) - math.log(2.0))
                + (sp.b_phi - 1.0) * (jnp.log1p(-phi) - math.log(2.0))
                - log_b
                - math.log(2.0)
            )
            if vol.h0 == "stationary":
                v0 = sigma * sigma / ((1.0 - phi) * (1.0 + phi))
            else:
                v0 = sp.B0 * sigma * sigma
            lp = lp + lnorm(h[0], mu, v0)
            trans = lnorm(h[1:], mu + phi * (h[:-1] - mu), sigma * sigma)
        return lp + _masked_sum(tmask, trans)
