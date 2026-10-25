"""Pytrees of the MCMC engine (DESIGN §2.5–§2.9) and the helpers every block shares.

Conventions (DESIGN §2.1):

* every class is a frozen dataclass registered with ``jax.tree_util.register_dataclass``, and
  every field is a data leaf. Static flags never live in leaves (E19): they are meta fields of
  the kernel dataclasses or of ``StaticConfig``;
* an absent part is :class:`Empty` (no leaves), never ``None``, so the tree structure is a
  function of the ``StaticConfig`` alone (:func:`state_template`, :func:`prior_template`,
  :func:`draw_template`);
* shapes are written for one chain with a leading equation axis ``n``; a kernel sees one
  equation (the ``_i`` slice without that axis) and the runner adds a leading chain axis ``C``;
* floats are float64; indicator and grid indices int32; error codes and run lengths int32; the
  sweep counter uint32 (DESIGN §2.2). Leaves named ``acc_*`` and ``rb_*`` are accumulators owned
  by the component that writes them; :func:`zero_accumulators` resets exactly those at the
  burn-in → sampling boundary (D-27);
* ``h`` has ``T_pad + 1`` states: index 0 is ``h_0`` and index ``t + 1`` belongs to data row
  ``t``; ``h_T = h[T_eff]`` is read with a dynamic index, never ``h[-1]`` (DESIGN §2.3).

Draw containers (:class:`ParamDraw` and its parts) are the per-draw parameters that the runner
stores and the forecast core consumes (DESIGN §3.2–§3.5, §5.3, §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.keys import Block

Array = jax.Array
PyTree = Any

_ERROR_BASE = 1000  # code = 1 + 1000·block + eq (DESIGN §5.7)


def _pytree[T](cls: type[T]) -> type[T]:
    """Frozen dataclass (identity equality: leaves are arrays) with every field a data leaf."""
    cls = dataclass(frozen=True, eq=False)(cls)
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    return jax.tree_util.register_dataclass(cls, data_fields=names, meta_fields=[])


@dataclass(frozen=True)
class Empty:
    """An absent part of a pytree: no leaves, and every instance is equal."""


jax.tree_util.register_dataclass(Empty, data_fields=[], meta_fields=[])


# --------------------------------------------------------------------------- data and cache (§2.5)


@_pytree
class EngineData:
    """One estimation window, shared by the chains (``in_axes=None``)."""

    Y: Array  # f64[T_pad, n]; rows ≥ T_eff are 0
    X: Array  # f64[T_pad, k]; x_t = (d_t', y_{t−1}', …, y_{t−p}')' (data/design.py layout)
    tmask: Array  # f64[T_pad]; m_t = 1{t < T_eff}
    T_eff: Array  # i32[]
    hist: Array  # f64[p, n]; last p observations, most recent last
    t_last: Array  # f64[]; 1-based sample position of the last observation (trend)


@_pytree
class EigenCache:
    """Mode I × Constant: ``G̃ = d^{−½}(cmaskθ⊗cmaskθ ⊙ G_const)d^{−½} = Q diag(g) Qᵀ`` (§3.7.2)."""

    Q: Array  # f64[n, q, q]
    g: Array  # f64[n, q]
    d_half: Array  # f64[n, q]
    u1: Array  # f64[n, q]
    u2: Array  # f64[n, q]


@_pytree
class FixedFactorCache:
    """Mode II × Constant: ``θ = θ̂ + √λ_i · s ⊙ Lc⁻ᵀz`` (§3.7.2)."""

    Lc: Array  # f64[n, q, q]
    s: Array  # f64[n, q]
    theta_hat: Array  # f64[n, q]


@_pytree
class DummyCache:
    """SOC/SUR dummy rows of the coefficient prior, per fit (WP25; N05 §7.7, N04 §3.1).

    Structural mean: ``F = [X_d | X_d M_c − Y_d[:, :n−1]]`` (``m × q``, the δ-design of the rows);
    reduced mean: ``F = X_d`` (``m × k``). ``G = FᵀF`` (exactly symmetric), ``FtY = FᵀY_d``, and
    ``w`` the λ-free row weights of every equation (a copy of ``CoefPriorArrays.dummy_w`` for the
    kernels, which see one equation's ``EqContext`` and not its prior arrays). The rows are prior
    rows: they never enter ``EngineData``, ``tmask``/``T_eff``, the likelihood or the LPD."""

    F: Array  # f64[m, q] (structural) | f64[m, k] (reduced)
    G: Array  # f64[q, q] | f64[k, k]
    FtY: Array  # f64[q, n] | f64[k, n]
    w: Array  # f64[n]


@_pytree
class FitCache:
    """Computed once per fit by the jitted ``prepare()`` (per Geweke iteration in validation)."""

    F: Array | Empty  # f64[T_pad, q]: structural δ-design [X | X M_c − Y_{:, :n−1}]
    FF: (
        Array | Empty
    )  # f64[T_pad, P]: packed products F[:, iu]·F[:, ju] (gram="packed", const_path="weighted")
    XX: Array | Empty  # f64[T_pad, P_x]: packed products of X (reduced form)
    G_const: Array | Empty  # f64[q, q]: Fᵀ D_m F (Constant paths)
    FtY: Array | Empty  # f64[q, n]: Fᵀ D_m Y (Constant paths)
    eq: EigenCache | FixedFactorCache | Empty  # per-equation leaves (leading n)
    dummy: DummyCache | Empty = Empty()  # SOC/SUR dummy rows (MeanStatic.n_dummy > 0; WP25)

    @classmethod
    def empty(cls) -> FitCache:
        return cls(Empty(), Empty(), Empty(), Empty(), Empty(), Empty(), Empty())


# --------------------------------------------------------------------------- prior arrays (§2.6)


@_pytree
class CoefPriorArrays:
    """δ_i (structural) or π_j (reduced) prior; λ-free precisions.

    ``dummy_w`` (WP25; ``MeanStatic.n_dummy > 0``, else ``Empty``): the λ-free weight of equation
    ``i``'s SOC/SUR dummy rows (``SharedPrior.dummies``): ``1/σ̄_i² = ι/ψ_i`` in structural Mode I
    and ``iw_equivalent`` (fixed weights, N05 §7.7), ``1`` in Mode II (the kernel divides by
    ``λ_i``), ``1/ψ_j`` in reduced form (fixed variance ``ψ_j``, N04 §3.1)."""

    mean0: Array  # f64[n, k]
    prec0: Array  # f64[n, k]
    group: Array  # i32[n, k]: 0 deterministic, 1 own lag, 2 cross lag
    dummy_w: Array | Empty = Empty()  # f64[n]: λ-free dummy-row weight (WP25)
    gl: GLArrays | Empty = Empty()  # global-local / SSVS arrays (P3 §2.3; CoefPriorStatic GL kinds)


@_pytree
class SSVSArrays:
    """SSVS spike and slab log standard deviations of every coefficient (P3 D-44)."""

    log_tau0: Array  # f64[n, k]
    log_tau1: Array  # f64[n, k]


@_pytree
class GLArrays:
    """Per-coefficient arrays of a global-local or SSVS prior (P3 §2.3, §3.1, §3.6)."""

    mask: Array  # f64[n, k]: 1 on GL-governed slots (lag coefficients), 0 elsewhere
    group: Array  # i32[n, k]: group id g(j) on GL slots (0 elsewhere, masked)
    log_r: Array  # f64[n, k]: log scale r_j (P3 D-37); 0 for scale="none"
    ssvs: SSVSArrays | Empty = Empty()


@_pytree
class GLShared:
    """Group-level hyperparameters of a global-local or SSVS prior (shared by every equation).

    Unused fields are 0 for a family that has no such parameter (P3 §3.2): ``b``, ``c`` (NG,
    R2D2; ``c`` < 0 encodes NG's tied ``c = a/2``), ``global_scale`` (HS), ``log_a_fixed`` (a fixed
    ``a``), ``a_support``/``a_logw`` (the discrete hyperprior grid, ``CoefPriorStatic.a_grid``
    points; shape ``(1,)`` zeros when ``a`` is fixed), ``p_fixed``, ``s1``, ``s2`` (SSVS)."""

    b: Array  # f64[]
    c: Array  # f64[]
    global_scale: Array  # f64[]
    log_a_fixed: Array  # f64[]
    a_support: Array  # f64[A]
    a_logw: Array  # f64[A]
    p_fixed: Array  # f64[]
    s1: Array  # f64[]
    s2: Array  # f64[]


@_pytree
class CommonSVPrior:
    """``ρ ~ N(ρ₀, V_ρ)·1(|ρ|<1)`` or ``(ρ+1)/2 ~ Beta(a, b)``; ``σ_h² ~ IG(c₀, C₀)`` or
    ``B·χ²₁`` (P3 D-46); ``n_eq`` the number of equations sharing ``h``."""

    rho_mean: Array  # f64[]
    rho_var: Array  # f64[]
    rho_a: Array  # f64[]
    rho_b: Array  # f64[]
    s2_shape: Array  # f64[]
    s2_scale: Array  # f64[]
    s2_B: Array  # f64[]
    n_eq: Array  # f64[]


@_pytree
class ContempPriorArrays:
    """``a_i`` prior; λ-free precisions; inactive slots ``j ≥ i`` have mean 0, precision 1."""

    mean0: Array  # f64[n, n−1]
    prec0: Array  # f64[n, n−1]
    cmask: Array  # f64[n, n−1]: 1{j < i}


@_pytree
class ConstantPrior:
    """``λ_i ~ IG(shape_i, scale_i)``."""

    shape: Array  # f64[n]
    scale: Array  # f64[n]


@_pytree
class SVPrior:
    """``μ ~ N(b_μ, B_μ)``, ``(φ+1)/2 ~ Beta(a_φ, b_φ)``, ``σ² ~ B_σ·χ²₁`` or ``IG(c0, C0)``."""

    b_mu: Array  # f64[n]
    B_mu: Array  # f64[n]: a variance
    a_phi: Array  # f64[n]
    b_phi: Array  # f64[n]
    B_sigma: Array  # f64[n]: sigma2_prior="scaled_chi2"
    c0: Array  # f64[n]: sigma2_prior="inv_gamma"
    C0: Array  # f64[n]
    B0: Array  # f64[n]: h0="fixed_b0": h_0 ~ N(μ, B0 σ²)
    m0: Array  # f64[n]: RW: h_0 ~ N(m0, V0)
    V0: Array  # f64[n]
    log_c: Array  # f64[n]: log offset (−inf only in zero_offset parity mode)


@_pytree
class TransitoryPrior:
    """``p ~ Beta(a_p, b_p)`` and log prior weights on the static ν grid."""

    a_p: Array  # f64[n]
    b_p: Array  # f64[n]
    log_nu_w: Array  # f64[n, G]


@_pytree
class EqPrior:
    """Per-equation prior blocks; every leaf has the leading axis ``n`` (vmapped)."""

    coef: CoefPriorArrays
    contemp: ContempPriorArrays
    vol_core: ConstantPrior | SVPrior
    transitory: TransitoryPrior | Empty


@_pytree
class DummyRows:
    """SOC/SUR dummy observations ``(y_d', x_d')`` in the reduced-form layout of Phase 1
    (``ConjugateProblem.dummies``; ``x_d`` in the ``data/design.py`` regressor order). Shared by
    every equation; their weights are per equation (``CoefPriorArrays.dummy_w``). WP25."""

    Y: Array  # f64[m, n]
    X: Array  # f64[m, k]


@_pytree
class SharedPrior:
    """Broadcast prior quantities (``in_axes=None``)."""

    Mc: Array  # f64[k, n−1]: D3 centring columns; zeros for reduced and zero means
    dummies: DummyRows | Empty = Empty()  # SOC/SUR rows (MeanStatic.n_dummy > 0; WP25)
    gl: GLShared | Empty = Empty()  # global-local / SSVS group-level hyperparameters (P3)
    shared_prec: Array | Empty = (
        Empty()
    )  # f64[q]: λ-free Mode II precision (shared_factor, P3 §5.3)
    csv: CommonSVPrior | Empty = Empty()  # CommonSV priors (VolStatic.common, P3 §5)


@_pytree
class PriorArrays:
    eq: EqPrior
    shared: SharedPrior


# --------------------------------------------------------------------------- chain state (§2.7)


@_pytree
class RecursiveEq:
    a: Array  # f64[n, n−1]: row i holds a_ij for j < i and exactly 0 for j ≥ i


@_pytree
class LinearEq:
    delta: Array  # f64[n, k]; γ_i = δ_i + M_c a_i is derived, never stored
    dev_log: Array | Empty = Empty()  # f64[n, k]: log|δ_i − m_i| (P3 §3.4; MeanStatic.coef_log)


@_pytree
class ConstantEq:
    log_lam: Array  # f64[n]


@_pytree
class SVEq:
    """AR SV (stationary or fixed-B0 ``h_0``); σ > 0 is stored, not σ²."""

    h: Array  # f64[n, T_pad+1]
    mu: Array  # f64[n]
    phi: Array  # f64[n]
    sigma: Array  # f64[n]
    acc_h: Array  # f64[n]
    acc_cgp: Array  # f64[n]
    acc_cs2: Array  # f64[n]
    acc_ncphi: Array  # f64[n]
    acc_ncms: Array  # f64[n]
    acc_fold: Array  # f64[n]
    acc_floor: Array  # f64[n]: zero_offset parity mode, masked exact zeros
    acc_rej_run: Array  # i32[n]: current run of exact-h rejections
    acc_rej_run_max: Array  # i32[n]
    rb_h_sum: Array  # f64[n, T_pad+1]
    rb_h_sq: Array  # f64[n, T_pad+1]


@_pytree
class SVRWEq:
    """Random-walk SV: ``h_0 = h[:, 0]``; no μ, φ."""

    h: Array  # f64[n, T_pad+1]
    sigma: Array  # f64[n]
    acc_h: Array  # f64[n]
    acc_cs2: Array  # f64[n]
    acc_ncms: Array  # f64[n]
    acc_fold: Array  # f64[n]
    acc_floor: Array  # f64[n]
    acc_rej_run: Array  # i32[n]
    acc_rej_run_max: Array  # i32[n]
    rb_h_sum: Array  # f64[n, T_pad+1]
    rb_h_sq: Array  # f64[n, T_pad+1]


@_pytree
class OutlierEq:
    """Outlier scale ``o = o_idx + 1 ∈ {1..K}``; padded rows hold ``o_idx = 0``."""

    o_idx: Array  # i32[n, T_pad]
    log_p: Array  # f64[n]
    log1m_p: Array  # f64[n]
    rb_out: Array  # f64[n, T_pad]: Σ P(o_t > 1 | ·)


@_pytree
class TEq:
    """Student-t scale ``τ`` and ``ν = nu_grid[nu_idx]``; padded rows hold ``log τ = 0``."""

    log_tau: Array  # f64[n, T_pad]
    nu_idx: Array  # i32[n]
    acc_nu_move: Array  # f64[n]


@_pytree
class OutlierTEq:
    o_idx: Array  # i32[n, T_pad]
    log_p: Array  # f64[n]
    log1m_p: Array  # f64[n]
    log_tau: Array  # f64[n, T_pad]
    nu_idx: Array  # i32[n]
    rb_out: Array  # f64[n, T_pad]
    acc_nu_move: Array  # f64[n]


@_pytree
class VolEq:
    core: ConstantEq | SVEq | SVRWEq
    trans: OutlierEq | TEq | OutlierTEq | Empty


# ---- Phase 3 coefficient-prior states (P3 §2.2): every scale is stored as its log; leaves are 0
#      on non-GL slots (deterministic terms), which the prior's mask excludes from every sum.


@_pytree
class HSEq:
    log_theta: Array  # f64[n, k]: local ϑ
    log_nu: Array  # f64[n, k]: auxiliary ν (Makalic–Schmidt)


@_pytree
class HSGlobal:
    log_zeta: Array  # f64[G]: group ζ_g
    log_tau: Array  # f64[G]: auxiliary τ_g


@_pytree
class NGEq:
    log_w: Array  # f64[n, k]: ϖ


@_pytree
class NGGlobal:
    log_xi: Array  # f64[G]: rate ξ_g
    a_idx: Array  # i32[G]: grid index of a_g (0 with a fixed a)


@_pytree
class R2D2Eq:
    log_varrho: Array  # f64[n, k]: exponential mixing ϱ
    log_w: Array  # f64[n, k]: ϖ


@_pytree
class R2D2Global:
    log_xi: Array  # f64[G]
    a_idx: Array  # i32[G]


@_pytree
class DLEq:
    log_theta: Array  # f64[n, k]: Laplace scale θ̃
    log_varrho: Array  # f64[n, k]: exponential mixing ϱ


@_pytree
class DLGlobal:
    a_idx: Array  # i32[G]


@_pytree
class SSVSEq:
    gamma: Array  # i32[n, k]: inclusion indicator
    rb_incl: Array  # f64[n, k]: Σ P(γ = 1 | ·) over sampling sweeps


@_pytree
class SSVSGlobal:
    log_p: Array  # f64[G]
    log1m_p: Array  # f64[G]


CoefEqState = HSEq | NGEq | R2D2Eq | DLEq | SSVSEq
CoefGlobalState = HSGlobal | NGGlobal | R2D2Global | DLGlobal | SSVSGlobal


@_pytree
class CommonSVGlobal:
    """Common log-volatility (P3 §5): index ``t`` is data row ``t`` (no presample state), 0 for
    ``t ≥ T_eff``; ``σ_h`` is stored as its log."""

    h: Array  # f64[T_pad]
    rho: Array  # f64[]
    log_sigma: Array  # f64[]
    acc_h: Array  # f64[]: accepted fraction of the non-empty h blocks, summed over sweeps
    acc_rho: Array  # f64[]
    acc_newton: Array  # f64[]: Newton iterations that hit the cap
    rb_ch_sum: Array  # f64[T_pad]
    rb_ch_sq: Array  # f64[T_pad]


@_pytree
class EqState:
    struct: RecursiveEq
    mean: LinearEq | Empty
    coef: CoefEqState | Empty
    vol: VolEq


@_pytree
class CCCMGlobal:
    Pi: Array  # f64[k, n]
    dev_log: Array | Empty = Empty()  # f64[n, k]: log|π_j − m_j|, row j (P3 §3.4; coef_log)


@_pytree
class GlobalState:
    mean: CCCMGlobal | Empty
    coef: CoefGlobalState | Empty
    vol: CommonSVGlobal | Empty
    struct: Empty


@_pytree
class Status:
    """First failure of the chain: ``code = 1 + 1000·block + eq`` (0 = none) at sweep ``it``."""

    code: Array  # i32[]
    it: Array  # u32[]
    min_pivot: Array  # f64[]: smallest factorisation pivot seen (+inf before the first)

    @classmethod
    def initial(cls) -> Status:
        return cls(jnp.zeros((), jnp.int32), jnp.zeros((), jnp.uint32), jnp.full((), jnp.inf))


@_pytree
class ChainState:
    eq: EqState
    glob: GlobalState
    status: Status


# --------------------------------------------------------------------------- views (§2.9)


@_pytree
class EqContext:
    """Built by the structure inside the equation body (one equation)."""

    i: Array  # i32[]: position in the ordering (keys, masks)
    target: Array  # f64[T_pad]: ỹ_i (minus cols·a_i when handed to a non-fusing mean)
    cols: Array  # f64[T_pad, n−1]: −ỹ_{·, :n−1}, shared and unmasked
    cmask: Array  # f64[n−1]: 1{j < i}
    weights: Array  # f64[T_pad]: m_t · exp(−log λ_eff,i,t)
    log_scale: Array  # f64[]: log λ_i if a prior block is λ-scaled, else 0
    tmask: Array  # f64[T_pad]
    common_w: Array | Empty = Empty()  # f64[T_pad]: m_t e^{−h_t}, unbatched (CommonSV, P3 §5.3)


@_pytree
class GaussianDiag:
    """A block prior ``N(mean, diag(1/prec))`` with its scaling applied; ``cmask`` marks active slots."""

    mean: Array  # f64[m]
    prec: Array  # f64[m]
    cmask: Array  # f64[m]


@_pytree
class GaussianLogDiag:
    """A block prior ``N(mean, diag(exp(−log_prec)))`` of a varying coefficient prior (P3 §2.4,
    §3.3): the mean takes the deviation path of P3 §3.4 (``MeanStatic.coef_log``)."""

    mean: Array  # f64[m]
    log_prec: Array  # f64[m]
    cmask: Array  # f64[m]


@_pytree
class GaussianPrec:
    """An exact Gaussian conditional ``N(K⁻¹b, K⁻¹)`` in precision form."""

    K: Array  # f64[m, m]
    b: Array  # f64[m]


@_pytree
class Coupling:
    """Coupling channel (D-30): ``λ_i | θ_i ~ IG(c + (T_eff + n_active)/2, C + ½e'D_m e + quad)``."""

    n_active: Array  # f64[]
    quad: Array  # f64[]

    @classmethod
    def zeros(cls) -> Coupling:
        return cls(jnp.zeros(()), jnp.zeros(()))

    def __add__(self, other: Coupling) -> Coupling:
        return Coupling(self.n_active + other.n_active, self.quad + other.quad)


@_pytree
class StepInfo:
    """Outcome of one block: failures are values, never exceptions (DESIGN §3.1 rule 4)."""

    ok: Array  # bool[]
    min_pivot: Array  # f64[]

    @classmethod
    def success(cls) -> StepInfo:
        """A step without factorisation: ``ok`` and pivot ``+inf``."""
        return cls(jnp.ones((), bool), jnp.full((), jnp.inf))

    def merge(self, other: StepInfo) -> StepInfo:
        return StepInfo(
            jnp.logical_and(self.ok, other.ok), jnp.minimum(self.min_pivot, other.min_pivot)
        )


@_pytree
class EqOut:
    """Equation-phase outputs handed to post-phase global steps."""

    e: Array  # f64[n, T_pad]
    log_lam: Array  # f64[n, T_pad]: effective log variance


@_pytree
class SweepCtx:
    data: EngineData
    prior: PriorArrays
    cache: FitCache
    eq_out: EqOut | Empty


# --------------------------------------------------------------------------- draws (§3, §5.3, §7.2)


@_pytree
class StructDraw:
    a: Array  # f64[n, n−1]


@_pytree
class LinearDraw:
    gamma: Array  # f64[n, k] (one equation: f64[k]); γ_i = δ_i + M_c a_i


@_pytree
class CCCMDraw:
    Pi: Array  # f64[k, n]


@_pytree
class CommonSVDraw:
    h_T: Array  # f64[]: h[T_eff − 1], the last data row (P3 §5.5)
    rho: Array  # f64[]
    sigma: Array  # f64[]


@_pytree
class GlobDraw:
    """Global parameters of one draw, one part per owner (mirrors :class:`GlobalState`)."""

    mean: CCCMDraw | Empty
    coef: Empty
    vol: CommonSVDraw | Empty
    struct: Empty


@_pytree
class ConstantDraw:
    log_lam: Array  # f64[n]


@_pytree
class SVDraw:
    h_T: Array  # f64[n]: h[T_eff]
    mu: Array  # f64[n]
    phi: Array  # f64[n]
    sigma: Array  # f64[n]


@_pytree
class SVRWDraw:
    h_T: Array  # f64[n]
    sigma: Array  # f64[n]


@_pytree
class OutlierDraw:
    log_p: Array  # f64[n]
    log1m_p: Array  # f64[n]


@_pytree
class TDraw:
    nu: Array  # f64[n]: ν as a float


@_pytree
class OutlierTDraw:
    log_p: Array  # f64[n]
    log1m_p: Array  # f64[n]
    nu: Array  # f64[n]


@_pytree
class VolDraw:
    core: ConstantDraw | SVDraw | SVRWDraw
    trans: OutlierDraw | TDraw | OutlierTDraw | Empty


@_pytree
class ParamDraw:
    """Everything forecasting needs from one stored draw (DESIGN §7.2)."""

    glob: GlobDraw
    mean: LinearDraw | Empty
    struct: StructDraw
    vol: VolDraw


@_pytree
class PathDraw:
    """Latent paths (``store="paths"``); rows beyond ``T_eff`` are trimmed on the host."""

    h: Array  # f64[n, T_pad+1]
    log_scale2: Array  # f64[n, T_pad]: 2 log o + log τ


# --------------------------------------------------------------------------- templates


def _f(*shape: int) -> np.ndarray:
    return np.zeros(shape, np.float64)


def _i(*shape: int) -> np.ndarray:
    return np.zeros(shape, np.int32)


def state_template(static: StaticConfig) -> ChainState:
    """A zero-filled one-chain :class:`ChainState` with the structure, shapes and dtypes that
    ``static`` implies (NumPy leaves; ``status.min_pivot = +inf``). The values are placeholders
    (σ = 0 is not a valid state); the template is the contract for structure and shapes."""
    s, vol = static.shapes, static.vol
    n, k, T = s.n, s.k, s.T_pad
    if vol.persistent == "constant":
        core: ConstantEq | SVEq | SVRWEq = ConstantEq(log_lam=_f(n))
    elif vol.rw:
        core = SVRWEq(
            h=_f(n, T + 1), sigma=_f(n), acc_h=_f(n), acc_cs2=_f(n), acc_ncms=_f(n),
            acc_fold=_f(n), acc_floor=_f(n), acc_rej_run=_i(n), acc_rej_run_max=_i(n),
            rb_h_sum=_f(n, T + 1), rb_h_sq=_f(n, T + 1),
        )  # fmt: skip
    else:
        core = SVEq(
            h=_f(n, T + 1), mu=_f(n), phi=_f(n), sigma=_f(n), acc_h=_f(n), acc_cgp=_f(n),
            acc_cs2=_f(n), acc_ncphi=_f(n), acc_ncms=_f(n), acc_fold=_f(n), acc_floor=_f(n),
            acc_rej_run=_i(n), acc_rej_run_max=_i(n), rb_h_sum=_f(n, T + 1), rb_h_sq=_f(n, T + 1),
        )  # fmt: skip
    trans: OutlierEq | TEq | OutlierTEq | Empty
    if vol.transitory == "outlier":
        trans = OutlierEq(o_idx=_i(n, T), log_p=_f(n), log1m_p=_f(n), rb_out=_f(n, T))
    elif vol.transitory == "t":
        trans = TEq(log_tau=_f(n, T), nu_idx=_i(n), acc_nu_move=_f(n))
    elif vol.transitory == "outlier_t":
        trans = OutlierTEq(
            o_idx=_i(n, T), log_p=_f(n), log1m_p=_f(n), log_tau=_f(n, T), nu_idx=_i(n),
            rb_out=_f(n, T), acc_nu_move=_f(n),
        )  # fmt: skip
    else:
        trans = Empty()
    coef_log = static.mean.coef_log
    dev = _f(n, k) if coef_log else Empty()
    coef_eq, coef_glob = _coef_templates(static)
    eq = EqState(
        struct=RecursiveEq(a=_f(n, n - 1)),
        mean=(
            LinearEq(delta=_f(n, k), dev_log=dev)
            if static.mean.kind == "linear_structural"
            else Empty()
        ),
        coef=coef_eq,
        vol=VolEq(core=core, trans=trans),
    )
    glob = GlobalState(
        mean=(
            CCCMGlobal(Pi=_f(k, n), dev_log=dev)
            if static.mean.kind == "linear_reduced"
            else Empty()
        ),
        coef=coef_glob,
        vol=_common_sv_template(static) if vol.common else Empty(),
        struct=Empty(),
    )
    status = Status(np.zeros((), np.int32), np.zeros((), np.uint32), np.full((), np.inf))
    return ChainState(eq=eq, glob=glob, status=status)


def _coef_templates(static: StaticConfig) -> tuple[PyTree, PyTree]:
    """The coefficient prior's per-equation and global state (``Empty`` for ``fixed_diagonal``)."""
    c, s = static.coef, static.shapes
    n, k, G = s.n, s.k, c.n_groups
    if c.kind == "horseshoe":
        return HSEq(_f(n, k), _f(n, k)), HSGlobal(_f(G), _f(G))
    if c.kind == "normal_gamma":
        return NGEq(_f(n, k)), NGGlobal(_f(G), _i(G))
    if c.kind == "r2d2":
        return R2D2Eq(_f(n, k), _f(n, k)), R2D2Global(_f(G), _i(G))
    if c.kind == "dirichlet_laplace":
        return DLEq(_f(n, k), _f(n, k)), DLGlobal(_i(G))
    if c.kind == "ssvs":
        return SSVSEq(_i(n, k), _f(n, k)), SSVSGlobal(_f(G), _f(G))
    return Empty(), Empty()


def _common_sv_template(static: StaticConfig) -> CommonSVGlobal:
    T = static.shapes.T_pad
    return CommonSVGlobal(
        h=_f(T), rho=_f(), log_sigma=_f(), acc_h=_f(), acc_rho=_f(), acc_newton=_f(),
        rb_ch_sum=_f(T), rb_ch_sq=_f(T),
    )  # fmt: skip


def prior_template(static: StaticConfig) -> PriorArrays:
    """A zero-filled :class:`PriorArrays` with the structure and shapes ``static`` implies."""
    s, vol = static.shapes, static.vol
    n, k = s.n, s.k
    if vol.persistent == "constant":
        core: ConstantPrior | SVPrior = ConstantPrior(shape=_f(n), scale=_f(n))
    else:
        core = SVPrior(*(_f(n) for _ in range(len(fields(SVPrior)))))
    trans: TransitoryPrior | Empty = (
        Empty()
        if vol.transitory == "none"
        else TransitoryPrior(a_p=_f(n), b_p=_f(n), log_nu_w=_f(n, len(vol.nu_grid)))
    )
    m = static.mean.n_dummy
    coef = static.coef
    gl: GLArrays | Empty = Empty()
    gl_shared: GLShared | Empty = Empty()
    if coef.varies:
        ssvs = SSVSArrays(_f(n, k), _f(n, k)) if coef.kind == "ssvs" else Empty()
        gl = GLArrays(mask=_f(n, k), group=_i(n, k), log_r=_f(n, k), ssvs=ssvs)
        A = max(coef.a_grid, 1)
        gl_shared = GLShared(
            b=_f(), c=_f(), global_scale=_f(), log_a_fixed=_f(), a_support=_f(A), a_logw=_f(A),
            p_fixed=_f(), s1=_f(), s2=_f(),
        )  # fmt: skip
    eq = EqPrior(
        coef=CoefPriorArrays(
            mean0=_f(n, k),
            prec0=_f(n, k),
            group=_i(n, k),
            dummy_w=_f(n) if m else Empty(),
            gl=gl,
        ),
        contemp=ContempPriorArrays(mean0=_f(n, n - 1), prec0=_f(n, n - 1), cmask=_f(n, n - 1)),
        vol_core=core,
        transitory=trans,
    )
    dummies = DummyRows(Y=_f(m, n), X=_f(m, k)) if m else Empty()
    shared_prec = _f(s.q) if static.mean.const_path == "shared_factor" else Empty()
    csv = (
        CommonSVPrior(*(_f() for _ in range(len(fields(CommonSVPrior))))) if vol.common else Empty()
    )
    shared = SharedPrior(
        Mc=_f(k, n - 1), dummies=dummies, gl=gl_shared, shared_prec=shared_prec, csv=csv
    )
    return PriorArrays(eq=eq, shared=shared)


def draw_template(static: StaticConfig) -> ParamDraw:
    """A zero-filled one-draw :class:`ParamDraw` with the structure and shapes ``static`` implies."""
    s, vol = static.shapes, static.vol
    n, k = s.n, s.k
    if vol.persistent == "constant":
        core: ConstantDraw | SVDraw | SVRWDraw = ConstantDraw(log_lam=_f(n))
    elif vol.rw:
        core = SVRWDraw(h_T=_f(n), sigma=_f(n))
    else:
        core = SVDraw(h_T=_f(n), mu=_f(n), phi=_f(n), sigma=_f(n))
    trans: OutlierDraw | TDraw | OutlierTDraw | Empty = {
        "none": Empty,
        "outlier": lambda: OutlierDraw(log_p=_f(n), log1m_p=_f(n)),
        "t": lambda: TDraw(nu=_f(n)),
        "outlier_t": lambda: OutlierTDraw(log_p=_f(n), log1m_p=_f(n), nu=_f(n)),
    }[vol.transitory]()
    reduced = static.mean.kind == "linear_reduced"
    return ParamDraw(
        glob=GlobDraw(
            mean=CCCMDraw(Pi=_f(k, n)) if reduced else Empty(),
            coef=Empty(),
            vol=CommonSVDraw(h_T=_f(), rho=_f(), sigma=_f()) if vol.common else Empty(),
            struct=Empty(),
        ),
        mean=LinearDraw(gamma=_f(n, k)) if static.mean.kind == "linear_structural" else Empty(),
        struct=StructDraw(a=_f(n, n - 1)),
        vol=VolDraw(core=core, trans=trans),
    )


# --------------------------------------------------------------------------- helpers


def leaf_path(path: tuple[Any, ...]) -> str:
    """Dotted name of a pytree path, e.g. ``"eq.vol.core.h"``."""
    return jax.tree_util.keystr(path, simple=True, separator=".")


def _allowed(name: str, allow: frozenset[str]) -> bool:
    return any(name == a or name.endswith("." + a) for a in allow)


def _is_key(dtype: Any) -> bool:
    return jnp.issubdtype(dtype, jax.dtypes.prng_key)


def assert_float64(tree: PyTree, allow: frozenset[str] = frozenset()) -> None:
    """Raise ``TypeError`` if a floating leaf is not float64 (complex: not complex128).

    ``allow`` lists leaves exempt from the rule (a component's declared ``float32_leaves``,
    empty in Phase 2); an entry matches a dotted path or any dotted suffix of it.
    """
    bad = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            dtype = np.result_type(leaf)
        if _is_key(dtype):
            continue
        name = leaf_path(path)
        if jnp.issubdtype(dtype, jnp.floating) and dtype != np.float64:
            if not _allowed(name, allow):
                bad.append(f"{name}: {dtype}")
        elif jnp.issubdtype(dtype, jnp.complexfloating) and dtype != np.complex128:
            bad.append(f"{name}: {dtype}")
    if bad:
        raise TypeError("float leaves must be float64 (R4): " + ", ".join(bad))


def tree_where(pred: Array, a: PyTree, b: PyTree) -> PyTree:
    """Leafwise ``where(pred, a, b)`` for a scalar predicate; dtypes and structure are kept."""
    pred = jnp.asarray(pred, dtype=bool)
    return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def zero_accumulators(st: PyTree) -> PyTree:
    """Zero every leaf named ``acc_*`` or ``rb_*`` (dtype kept); every other leaf is returned
    unchanged (the same object). Works on device and host (NumPy) trees."""

    def zero(path: tuple[Any, ...], leaf: Any) -> Any:
        name = getattr(path[-1], "name", None) if path else None
        if not (isinstance(name, str) and name.startswith(("acc_", "rb_"))):
            return leaf
        return np.zeros_like(leaf) if isinstance(leaf, np.ndarray) else jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(zero, st)


def encode_error(block: Block, eq: Any) -> Array:
    """``1 + 1000·block + eq`` as int32 (``eq`` is the equation; 0 for global steps)."""
    if isinstance(block, IntEnum) and not isinstance(block, Block):
        raise TypeError(f"block must be a keys.Block, got {type(block).__name__}.{block.name}")
    if isinstance(block, int):
        block = Block(block)
    if not isinstance(block, Block):
        raise TypeError(f"block must be a keys.Block, got {block!r}")
    if isinstance(eq, int) and not 0 <= eq < _ERROR_BASE:
        raise ValueError(f"equation index must lie in [0, {_ERROR_BASE}), got {eq}")
    return jnp.int32(1 + _ERROR_BASE * int(block)) + jnp.asarray(eq).astype(jnp.int32)


def decode_error(code: int) -> tuple[Block, int]:
    """Inverse of :func:`encode_error` for a non-zero code."""
    code = int(code)
    if code <= 0:
        raise ValueError(f"error code must be positive (0 means no failure), got {code}")
    block, eq = divmod(code - 1, _ERROR_BASE)
    try:
        return Block(block), eq
    except ValueError:
        raise ValueError(f"error code {code} names no keys.Block ({block})") from None


def check_state_invariants(st: ChainState, static: StaticConfig, T_eff: int) -> list[str]:
    """Host-side check of the DESIGN §2.8 invariants; returns the violations (empty if none).

    Accepts one chain or any leading batch axes (chains). Checks: structure, trailing shapes and
    dtypes against :func:`state_template` (float64 except ``static.float32_leaves``; no keys);
    finiteness (``status.min_pivot`` may be +inf); ``status.code == 0``; ``A`` unit lower
    triangular (``a ⊙ (1 − cmask) == 0`` exactly); inert padding (``o_idx``, ``log τ`` zero on
    rows ``≥ T_eff``); index ranges; ``σ > 0``; ``|φ| < 1`` (AR); accumulators ``≥ 0``.
    """
    out: list[str] = []
    shapes = static.shapes
    if not 1 <= int(T_eff) <= shapes.T_pad:
        out.append(f"T_eff={T_eff} outside [1, T_pad={shapes.T_pad}]")
        return out
    T_eff = int(T_eff)
    tmpl = state_template(static)
    got, treedef = jax.tree_util.tree_flatten_with_path(st)
    if treedef != jax.tree.structure(tmpl):
        return [f"state structure does not match the StaticConfig: {treedef}"]
    batch: tuple[int, ...] | None = None
    arrays: dict[str, np.ndarray] = {}
    for (path, leaf), ref in zip(got, jax.tree.leaves(tmpl), strict=True):
        name = leaf_path(path)
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or _is_key(dtype):
            out.append(f"{name}: not an array of numbers (keys are never state)")
            continue
        float_ok = dtype == ref.dtype or (
            jnp.issubdtype(ref.dtype, jnp.floating)
            and _allowed(name, static.float32_leaves)
            and dtype in (np.float32, np.float64)  # exempt from float64 only, not from being float
        )
        if not float_ok:
            out.append(f"{name}: dtype {dtype}, expected {ref.dtype}")
        lshape = tuple(np.shape(leaf))
        nb = len(lshape) - ref.ndim
        if nb < 0 or lshape[nb:] != ref.shape:
            out.append(f"{name}: shape {lshape}, expected (..., {', '.join(map(str, ref.shape))})")
            continue
        if batch is None:
            batch = lshape[:nb]
        elif lshape[:nb] != batch:
            out.append(f"{name}: batch axes {lshape[:nb]} differ from {batch}")
        arrays[name] = np.asarray(leaf)
    if out:  # structural problems: the value checks below would be meaningless
        return out
    for name, arr in arrays.items():
        if jnp.issubdtype(arr.dtype, jnp.floating):
            finite = np.isfinite(arr)
            if name == "status.min_pivot":  # +inf is the "no pivot seen yet" sentinel; −inf is not
                finite |= arr == np.inf
            if not finite.all():
                out.append(f"{name}: {int((~finite).sum())} non-finite values")
        if name.split(".")[-1].startswith("acc_") and (arr < 0).any():
            out.append(f"{name}: negative accumulator")
    if (arrays["status.code"] != 0).any():
        for c in sorted({int(c) for c in np.unique(arrays["status.code"]) if c != 0}):
            try:
                block, eq = decode_error(c)
                out.append(f"status.code = {c}: failure in {block.name}, equation {eq}")
            except ValueError:
                out.append(f"status.code = {c}: invalid error code")
    n = shapes.n
    upper = np.arange(n - 1)[None, :] >= np.arange(n)[:, None]  # j ≥ i: inactive slots
    if (arrays["eq.struct.a"][..., upper] != 0).any():
        out.append("eq.struct.a: A is not unit lower triangular (non-zero a_ij with j >= i)")
    vol = static.vol
    for name, hi in (
        ("eq.vol.trans.o_idx", vol.outlier_K),
        ("eq.vol.trans.nu_idx", len(vol.nu_grid)),
    ):
        if name in arrays and ((arrays[name] < 0) | (arrays[name] >= hi)).any():
            out.append(f"{name}: index outside [0, {hi})")
    for name in ("eq.vol.trans.o_idx", "eq.vol.trans.log_tau"):
        if name in arrays and (arrays[name][..., T_eff:] != 0).any():
            out.append(f"{name}: padded rows t >= T_eff={T_eff} are not inert")
    if "eq.vol.core.sigma" in arrays and (arrays["eq.vol.core.sigma"] <= 0).any():
        out.append("eq.vol.core.sigma: σ must be > 0")
    if "eq.vol.core.phi" in arrays and (np.abs(arrays["eq.vol.core.phi"]) >= 1).any():
        out.append("eq.vol.core.phi: |φ| must be < 1")
    # Phase 3 (P3 §2.2): GL leaves are exactly 0 off the GL slots (the deterministic columns)
    if static.coef.varies:
        n_det = shapes.n_det
        for name, arr in arrays.items():
            if name.startswith("eq.coef.") and (arr[..., :n_det] != 0).any():
                out.append(f"{name}: non-zero on the deterministic slots (not GL coordinates)")
    a_hi = max(static.coef.a_grid, 1)
    if "glob.coef.a_idx" in arrays:
        idx = arrays["glob.coef.a_idx"]
        if ((idx < 0) | (idx >= a_hi)).any():
            out.append(f"glob.coef.a_idx: index outside [0, {a_hi})")
    if "eq.coef.gamma" in arrays and not np.isin(arrays["eq.coef.gamma"], (0, 1)).all():
        out.append("eq.coef.gamma: SSVS indicators must be 0 or 1")
    if "glob.vol.rho" in arrays and (np.abs(arrays["glob.vol.rho"]) >= 1).any():
        out.append("glob.vol.rho: |ρ| must be < 1")
    if "glob.vol.h" in arrays and (arrays["glob.vol.h"][..., T_eff:] != 0).any():
        out.append(f"glob.vol.h: padded states t >= T_eff={T_eff} must be 0")
    return out
