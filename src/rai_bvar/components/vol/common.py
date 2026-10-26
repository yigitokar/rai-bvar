"""``CommonSV``: one stochastic-volatility factor shared by every variance (P3 DESIGN §5; D-46…D-48;
N08 §A4–A5.3; WP32).

**Model** (P3 §5.1; dates 0-based, ``h_t`` belongs to data row ``t``, ``T = T_eff``)::

    λ_{i,t} = λ_i e^{h_t},   h_0 ~ N(0, σ_h²/(1 − ρ²)),   h_t = ρ h_{t−1} + σ_h η_t  (1 ≤ t < T)
    ρ ~ N(ρ₀, V_ρ)·1(|ρ| < 1)  or  (ρ + 1)/2 ~ Beta(a, b);   σ_h² ~ IG(c₀, C₀)  or  B·χ²₁

The per-equation cores are the Constant ones (``ConstantCore`` with ``uses_global_vol``); this
module holds the post step :class:`CommonSVStep` (``Block.G_COMMON_SV``, ``reads = {eq.vol,
glob.vol}``, ``writes = {glob.vol}``) that draws ``h, ρ, σ_h`` after the equation phase. Given the
coefficients and the ``λ_i`` the data enter only through ``q_t = Σ_i e²_{it}/λ_i`` (N08 §A4):
``log q_t = logsumexp_i(log e²_{it} − log λ_i)`` (:func:`csv_q`) from ``SweepCtx.eq_out.e`` and the
cores' ``log λ_i``. The conditional of ``h`` is (*derived here* as N08 §A5.2 does)::

    ℓ(h) = −(n/2) Σ_{t<T} h_t − ½ Σ_{t<T} exp(log q_t − h_t) − ½ hᵀK_h h,
    K_h = σ_h⁻² tridiag(diag = [1, 1+ρ², …, 1+ρ², 1], off = −ρ) on t < T     (T = 1: (1 − ρ²)/σ_h²)

(``−2 log p(h) = (1 − ρ²)h_0²/σ_h² + Σ_{t≥1}(h_t − ρh_{t−1})²/σ_h²`` + const, expanded.) Padded
states ``t ≥ T`` are held at exactly 0 and enter no sum (decoupled: unit diagonal, zero
off-diagonals).

**The h step** (P3 §5.4, D-48): an offset ``o ~ U{0, …, block − 1}`` (``Sub.CSV_OFFSET``) lays out
the blocks ``B_0 = [0, o)``, ``B_b = [o + (b−1)·block, o + b·block)`` for ``b = 1 … ⌈T_pad/block⌉``,
each intersected with ``[0, T)`` (a static count ``1 + ⌈T_pad/block⌉``; every real date lies in
exactly one block, empty blocks are no-ops). A ``lax.scan`` visits them **sequentially**, so block
``B`` conditions on the current ``h`` outside it (the updated left neighbour). Each block lives in a
window of static length ``W = min(block, T_pad)``; its target is

    φ_B(x) = Σ_{t∈B} [−(n/2) x_t − ½ exp(log q_t − x_t)] − ½ xᵀK_BB x + linᵀx,
    lin = −K_{B,B^c} h_{B^c}  (only ``ρ/σ_h²·h_{lo−1}`` at the first date and ``ρ/σ_h²·h_hi`` at the last)

``linalg.tridiag_newton`` finds its mode ``ĥ`` from the start ``log q_t − log n`` (the internal
``csv_newton_start``, which returns the start and the iteration cap: a function of ``q``, the
neighbours and ``(ρ, σ_h)`` only, never of the current block, so the proposal is an independence
proposal whatever Newton's accuracy), with the negative Hessian ``G_B = K_BB + diag(½ q e^{−ĥ})``.
The proposal is ``h* = ĥ + L_B⁻ᵀz`` (``G_B = L_B L_Bᵀ``, ``z`` from ``Sub.CSV_H`` folded with the block
index), accepted iff ``log U < log α`` (``Sub.CSV_H_ACC``, block index) with

    log α = φ_B(h*) − φ_B(h_B) + ½(h* − ĥ)ᵀG_B(h* − ĥ) − ½(h_B − ĥ)ᵀG_B(h_B − ĥ).

**ρ | h, σ_h** (§5.4, E117, E158): with ``S_xx = Σ_{t=1}^{T−1} h_{t−1}²`` and ``S_xy = Σ h_{t−1}h_t``,
``K_ρ = S_xx/σ_h² (+ 1/V_ρ under TruncatedNormal)``, ``ρ̂ = K_ρ⁻¹(S_xy/σ_h² (+ ρ₀/V_ρ))``. The
proposal is ``TN(ρ̂, K_ρ⁻¹; −1, 1)`` (``Sub.CSV_RHO``; inverse CDF in log space,
``random.truncated_std_normal``), except under the Beta prior when ``K_ρ < 1e-8``, where it is
``U(−1, 1)`` (the choice depends on ``h`` and ``σ_h`` only, so the kernel is valid). With
``g(ρ) = ½ log(1 − ρ²) − (1 − ρ²)h_0²/(2σ_h²)`` (the stationary-start factor)::

    log α = g(ρ*) − g(ρ) [+ (a−1)Δlog(1+ρ) + (b−1)Δlog(1−ρ) under Beta]
                         [+ (S_xy(ρ* − ρ) − ½S_xx(ρ*² − ρ²))/σ_h² for the uniform proposal]

(internal ``csv_rho_accept``); a draw with ``|ρ*| ≥ 1`` rejects (``Sub.CSV_RHO_ACC``).

**σ_h² | h, ρ** (exact; ``Sub.CSV_S2``): ``S = (1 − ρ²)h_0² + Σ_{t=1}^{T−1}(h_t − ρh_{t−1})²``;
``InvGamma(c₀, C₀)``: ``IG(c₀ + T/2, C₀ + S/2)``; ``ScaledChi2(B)``: ``GIG(½ − T/2, 1/B, S)``
(``random.log_gig``). The state stores ``log σ_h``.

**Accumulators** (E132): ``acc_csv_h`` the accepted fraction of the non-empty blocks of a sweep,
``acc_csv_rho`` the ρ acceptance, ``acc_csv_newton`` the number of blocks whose Newton hit the
cap or stalled (the draw is exact either way), all summed over sampling sweeps; ``rb_ch_sum``,
``rb_ch_sq`` the sums of ``h`` and ``h²``. **Init** (§5.5, E133, E175): ``ρ`` the prior centre
clipped to ``(−0.99, 0.99)``, ``σ_h²`` the prior mean (the IG mode when ``c₀ ≤ 1``) with ``log σ_h``
jittered by ``N(0, 0.25²)`` per chain, and ``h`` on the real dates one draw of its stationary AR(1)
prior given that ``(ρ, σ_h)`` (0 on the padded rows). Not ``h = 0``: if the first ``h`` step rejects
every block, ``S = 0`` and the ``ScaledChi2`` conditional ``GIG(½ − T/2, 1/B, 0)`` is improper. **Forecast** (:meth:`CommonSVStep.future`): ``h_{T+j} =
ρh_{T+j−1} + σ_hη_j`` from ``h_T = h[T − 1]``.

Overridable internals (P3 §2.4, §9.6; test seam only): ``csv_newton_start`` (M28),
``csv_q`` (M29), ``csv_rho_accept`` (M30); their defaults are the module functions of the same
names. The module computes nothing with JAX at import time (P3 §1.1, E163).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax import random as jr
from jax.scipy.special import gammaln, log_ndtr, logsumexp

from rai_bvar.components.vol.robust import log_e2
from rai_bvar.components.vol.transitory import resolve_vol_static
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc import random as R
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.keys import Block, Key, Sub, sub_key
from rai_bvar.engines.mcmc.linalg import (
    NEWTON_MAX_ITER,
    TridiagNewton,
    tridiag_newton,
    tridiag_precision_sample,
)
from rai_bvar.engines.mcmc.protocols import EqMap, KernelBase
from rai_bvar.engines.mcmc.state import (
    ChainState,
    CommonSVDraw,
    CommonSVGlobal,
    CommonSVPrior,
    EngineData,
    FitCache,
    GlobalState,
    GlobDraw,
    PriorArrays,
    StepInfo,
    SweepCtx,
)

Array = jax.Array
PyTree = Any
_F64 = jnp.float64

__all__ = [
    "RHO_UNIFORM_THRESHOLD",
    "CommonSVStep",
    "HBlock",
    "RhoProposal",
    "Sigma2Conditional",
    "ar1_precision",
    "block_system",
    "csv_newton_start",
    "csv_q",
    "csv_rho_accept",
    "g_rho",
    "h_block_log_target",
    "h_log_accept",
    "rho_proposal",
    "rho_sufficient",
    "sigma2_conditional",
    "stationary_ar1_path",
    "tridiag_quadratic",
]

RHO_UNIFORM_THRESHOLD = 1e-8  # Beta prior: U(−1, 1) proposal when K_ρ is below this (E117, E158)
INIT_RHO_BOUND = 0.99  # ρ starts at the prior centre clipped to (−0.99, 0.99) (P3 §5.5)
INIT_LOG_SIGMA_SD = 0.25  # per-chain N(0, 0.25²) jitter of log σ_h at the cold start
_LOG_2 = math.log(2.0)
_LOG_2PI = math.log(2.0 * math.pi)


def stationary_ar1_path(key: Key, rho: Array, log_sigma: Array, T: int) -> Array:
    """One path ``h_0 ~ N(0, σ_h²/(1 − ρ²))``, ``h_t = ρh_{t−1} + σ_hη_t`` of length ``T`` (the
    cold start's ``h``, E175)."""
    eta = jnp.exp(log_sigma) * jr.normal(key, (T,), dtype=_F64)
    eta = eta.at[0].divide(jnp.sqrt(1.0 - rho * rho))

    def step(prev: Array, e: Array) -> tuple[Array, Array]:
        h = rho * prev + e
        return h, h

    _, h = lax.scan(step, jnp.zeros((), _F64), eta)
    return h


RhoKind = Literal["truncated_normal", "beta"]
S2Kind = Literal["inv_gamma", "scaled_chi2"]


# --------------------------------------------------------------------------- the data statistic


def csv_q(log_e2: Array, log_lam: Array, tmask: Array) -> Array:
    """``log q_t = logsumexp_i(log e²_{it} − log λ_i)`` (f64[T_pad]) on real rows, 0 on padded
    rows (N08 §A4; overridable, mutant M29 uses ``+ log λ_i``). ``log_e2`` is ``(n, T_pad)``
    (``robust.log_e2``: ``−∞`` for an exact zero residual), ``log_lam`` the cores' ``(n,)``."""
    live = jnp.asarray(tmask) > 0
    terms = jnp.where(live[None, :], log_e2 - jnp.asarray(log_lam, _F64)[:, None], -jnp.inf)
    return jnp.where(live, logsumexp(terms, axis=0), 0.0)


# --------------------------------------------------------------------------- the h block (§5.4)


def ar1_precision(rho: Array, log_sigma: Array, T_eff: Array, length: int) -> tuple[Array, Array]:
    """``K_h`` on ``t < T`` in a window of static ``length`` (``diag`` f64[length], ``off``
    f64[length − 1] with ``off[t] = K[t+1, t]``): ``σ_h⁻²·[1, 1 + ρ², …, 1 + ρ², 1]`` and
    ``−ρ/σ_h²``; ``(1 − ρ²)/σ_h²`` when ``T = 1`` (E152); decoupled padded states (diagonal 1,
    no off-diagonal)."""
    t = jnp.arange(length, dtype=jnp.int32)
    T = jnp.asarray(T_eff, jnp.int32)
    inv_s2 = jnp.exp(-2.0 * jnp.asarray(log_sigma, _F64))
    rho = jnp.asarray(rho, _F64)
    interior = (t >= 1) & (t <= T - 2)
    base = jnp.where(T == 1, (1.0 - rho) * (1.0 + rho), jnp.where(interior, 1.0 + rho * rho, 1.0))
    diag = jnp.where(t < T, base * inv_s2, 1.0)
    off = jnp.where(t[:-1] + 1 < T, -rho * inv_s2, 0.0)
    return diag, off


def tridiag_quadratic(diag: Array, off: Array, v: Array) -> Array:
    """``vᵀQv`` for the tridiagonal ``Q`` (``diag``, ``off[t] = Q[t+1, t]``)."""
    return jnp.sum(diag * v * v) + 2.0 * jnp.sum(off * v[:-1] * v[1:])


def h_block_log_target(
    x: Array, log_q: Array, n: Array, diag: Array, off: Array, lin: Array, mask: Array
) -> Array:
    """``φ_B(x) = Σ_B [−(n/2)x_t − ½exp(log q_t − x_t)] − ½xᵀK_BBx + linᵀx`` (module docstring)
    on a window: ``mask`` marks the block's dates; ``diag``/``off``/``lin`` as :func:`block_system`
    builds them (masked slots decoupled), so masked slots of ``x`` contribute nothing."""
    on = mask > 0
    x = jnp.where(on, x, 0.0)
    e = jnp.exp(jnp.where(on, log_q - x, -jnp.inf))
    like = jnp.sum(jnp.where(on, -0.5 * n * x - 0.5 * e, 0.0))
    return like - 0.5 * tridiag_quadratic(jnp.where(on, diag, 0.0), off, x) + jnp.sum(lin * x)


class HBlock(NamedTuple):
    """One block of the ``h`` step in its window (:func:`block_system`)."""

    start: Array  # i32[]: window start in the padded path (= lo when the block is non-empty)
    mask: Array  # f64[W]: 1 on the block's dates
    log_q: Array  # f64[W]
    h: Array  # f64[W]: the current values
    diag: Array  # f64[W]: K_BB diagonal (1 on masked slots)
    off: Array  # f64[W−1]: K_BB off-diagonal (0 outside pairs of block dates)
    lin: Array  # f64[W]: −K_{B,B^c} h_{B^c}


def block_system(
    hf: Array, lqf: Array, kd: Array, ko: Array, lo: Array, hi: Array, T_eff: Array,
    T_pad: int, W: int,
) -> HBlock:  # fmt: skip
    """Block ``[lo, hi) ∩ [0, T)`` in a window of length ``W`` read from the padded path ``hf``,
    ``log q`` ``lqf`` (length ``T_pad + W``) and the full ``K_h`` (``kd``, ``ko`` of
    :func:`ar1_precision` on that length). A block starting at or beyond ``T_pad`` is empty and
    reads the zero padding."""
    start = jnp.minimum(jnp.asarray(lo, jnp.int32), T_pad)
    idx = start + jnp.arange(W, dtype=jnp.int32)
    T = jnp.asarray(T_eff, jnp.int32)
    on = (idx >= lo) & (idx < hi) & (idx < T)
    mask = on.astype(_F64)
    h_w = lax.dynamic_slice(hf, (start,), (W,))
    lq_w = lax.dynamic_slice(lqf, (start,), (W,))
    diag = jnp.where(on, lax.dynamic_slice(kd, (start,), (W,)), 1.0)
    pair = on[1:] & on[:-1]
    off = jnp.where(pair, lax.dynamic_slice(ko, (start,), (W - 1,)), 0.0)
    L = hf.shape[0]
    left = jnp.where(idx >= 1, hf[jnp.clip(idx - 1, 0, L - 1)], 0.0)
    right = hf[jnp.clip(idx + 1, 0, L - 1)]
    k_left = jnp.where(idx >= 1, ko[jnp.clip(idx - 1, 0, L - 2)], 0.0)  # K[idx, idx − 1]
    k_right = ko[jnp.clip(idx, 0, L - 2)]  # K[idx, idx + 1] (0 beyond the real dates)
    in_left = jnp.concatenate([jnp.zeros((1,), bool), on[:-1]])
    in_right = jnp.concatenate([on[1:], jnp.zeros((1,), bool)])
    lin = jnp.where(in_left, 0.0, -k_left * left) + jnp.where(in_right, 0.0, -k_right * right)
    return HBlock(start, mask, lq_w, h_w, diag, off, jnp.where(on, lin, 0.0))


def csv_newton_start(log_q: Array, h: Array, mask: Array, n: Array) -> tuple[Array, int]:
    """``(x0, max_iter)`` of the block's Newton (overridable; E152: the seam sets both, as
    mutant M28 changes both): ``log q_t − log n`` on the block, a function of ``q`` alone, and
    the cap ``linalg.NEWTON_MAX_ITER``. ``h`` (the current block) is unused by design."""
    del h
    return jnp.where(mask > 0, log_q - jnp.log(jnp.asarray(n, _F64)), 0.0), NEWTON_MAX_ITER


def h_log_accept(x_new: Array, x_old: Array, prop: TridiagNewton, blk: HBlock, n: Array) -> Array:
    """``log α = φ_B(h*) − φ_B(h) + ½(h* − ĥ)ᵀG(h* − ĥ) − ½(h − ĥ)ᵀG(h − ĥ)`` of the block's
    independence MH (module docstring), ``G`` the negative Hessian at the mode ``ĥ``."""
    on = blk.mask > 0

    def target(x: Array) -> Array:
        return h_block_log_target(x, blk.log_q, n, blk.diag, blk.off, blk.lin, blk.mask)

    def half_q(x: Array) -> Array:
        v = jnp.where(on, x - prop.mode, 0.0)
        return 0.5 * tridiag_quadratic(jnp.where(on, prop.diag, 0.0), prop.off, v)

    return target(x_new) - target(x_old) + half_q(x_new) - half_q(x_old)


# --------------------------------------------------------------------------- ρ and σ_h² (§5.4)


def rho_sufficient(h: Array, T_eff: Array) -> tuple[Array, Array, Array]:
    """``(S_xy, S_xx, h_0)``: ``Σ_{t=1}^{T−1} h_{t−1}h_t``, ``Σ_{t=1}^{T−1} h_{t−1}²`` and ``h_0``."""
    t = jnp.arange(1, h.shape[0], dtype=jnp.int32)
    on = t < jnp.asarray(T_eff, jnp.int32)
    prev, cur = h[:-1], h[1:]
    return jnp.sum(jnp.where(on, prev * cur, 0.0)), jnp.sum(jnp.where(on, prev * prev, 0.0)), h[0]


class RhoProposal(NamedTuple):
    """``ρ* ~ TN(mean, 1/prec; −1, 1)``, or ``U(−1, 1)`` when ``uniform`` (P3 §5.4)."""

    prec: Array
    mean: Array
    uniform: Array  # bool[]


def rho_proposal(
    s_xy: Array, s_xx: Array, log_sigma: Array, csv: CommonSVPrior, kind: RhoKind
) -> RhoProposal:
    """``K_ρ``, ``ρ̂`` and the branch of the ρ proposal (module docstring; E117, E158): the
    uniform branch only under the Beta prior when ``K_ρ < 1e-8``; under TruncatedNormal always
    the TN (``K_ρ ≥ 1/V_ρ``)."""
    inv_s2 = jnp.exp(-2.0 * jnp.asarray(log_sigma, _F64))
    prec = s_xx * inv_s2
    num = s_xy * inv_s2
    if kind == "truncated_normal":
        prec = prec + 1.0 / csv.rho_var
        num = num + csv.rho_mean / csv.rho_var
        uniform = jnp.zeros((), bool)
    else:
        uniform = prec < RHO_UNIFORM_THRESHOLD
    mean = jnp.where(prec > 0.0, num / jnp.where(prec > 0.0, prec, 1.0), 0.0)
    return RhoProposal(prec, mean, uniform)


def g_rho(rho: Array, h0: Array, log_sigma: Array) -> Array:
    """The stationary-start factor ``log g(ρ) = ½ log(1 − ρ²) − (1 − ρ²)h_0²/(2σ_h²)``."""
    one_m = (1.0 - rho) * (1.0 + rho)
    return 0.5 * (jnp.log1p(-rho) + jnp.log1p(rho)) - 0.5 * one_m * h0 * h0 * jnp.exp(
        -2.0 * log_sigma
    )


def csv_rho_accept(
    rho_new: Array, rho_old: Array, h0: Array, log_sigma: Array, s_xy: Array, s_xx: Array,
    csv: CommonSVPrior, uniform: Array, kind: RhoKind,
) -> Array:  # fmt: skip
    """``log α`` of the ρ independence MH (module docstring; overridable, mutant M30 drops the
    stationary-start factor): ``g(ρ*) − g(ρ)``, plus the Beta log-density ratio under the Beta
    prior, plus the Gaussian factor's ratio for the uniform proposal; ``−∞`` for ``|ρ*| ≥ 1``."""
    inside = jnp.abs(rho_new) < 1.0
    rn = jnp.where(inside, rho_new, 0.0)
    la = g_rho(rn, h0, log_sigma) - g_rho(rho_old, h0, log_sigma)
    if kind == "beta":
        la = la + (csv.rho_a - 1.0) * (jnp.log1p(rn) - jnp.log1p(rho_old))
        la = la + (csv.rho_b - 1.0) * (jnp.log1p(-rn) - jnp.log1p(-rho_old))
        gauss = (s_xy * (rn - rho_old) - 0.5 * s_xx * (rn * rn - rho_old * rho_old)) * jnp.exp(
            -2.0 * log_sigma
        )
        la = la + jnp.where(uniform, gauss, 0.0)
    return jnp.where(inside, la, -jnp.inf)


class Sigma2Conditional(NamedTuple):
    """``σ_h² | h, ρ``: ``IG(shape, scale)`` (``inv_gamma``) or ``GIG(lam, ψ, χ)`` with ``ψ``, ``χ``
    in logs (``scaled_chi2``); the unused fields are inert."""

    shape: Array
    log_scale: Array
    lam: Array
    log_psi: Array
    log_chi: Array


def sigma2_conditional(
    h: Array, rho: Array, T_eff: Array, csv: CommonSVPrior, kind: S2Kind
) -> Sigma2Conditional:
    """The exact ``σ_h² | h, ρ`` (module docstring): ``S = (1 − ρ²)h_0² + Σ(h_t − ρh_{t−1})²``."""
    t = jnp.arange(1, h.shape[0], dtype=jnp.int32)
    T = jnp.asarray(T_eff, jnp.int32)
    resid = jnp.where(t < T, h[1:] - rho * h[:-1], 0.0)
    S = (1.0 - rho) * (1.0 + rho) * h[0] * h[0] + jnp.sum(resid * resid)
    half_T = 0.5 * T.astype(_F64)
    zero = jnp.zeros((), _F64)
    if kind == "inv_gamma":
        return Sigma2Conditional(
            csv.s2_shape + half_T, jnp.log(csv.s2_scale + 0.5 * S), zero, zero, zero
        )
    return Sigma2Conditional(zero, zero, 0.5 - half_T, -jnp.log(csv.s2_B), jnp.log(S))


# --------------------------------------------------------------------------- the step


@dataclass(frozen=True)
class CommonSVStep(KernelBase):
    """The post-phase global step of ``CommonSV`` (``Block.G_COMMON_SV``; module docstring).

    Built as ``CommonSVStep(VolStatic, T_pad)`` (the volatility convention; the volatility block's
    :meth:`~rai_bvar.components.vol.robust.VolBlockImpl.global_steps` does so) or with a
    ``StaticConfig``."""

    static: VolStatic
    T_pad: int | None = None

    block: ClassVar[Block] = Block.G_COMMON_SV
    phase: ClassVar[Literal["pre", "post"]] = "post"
    reads: ClassVar[frozenset[str]] = frozenset({"eq.vol", "glob.vol"})
    writes: ClassVar[frozenset[str]] = frozenset({"glob.vol"})
    marginal_of_indicators: ClassVar[bool] = True
    name: ClassVar[str] = "common_sv"

    csv_newton_start = staticmethod(csv_newton_start)
    csv_q = staticmethod(csv_q)
    csv_rho_accept = staticmethod(csv_rho_accept)

    def __post_init__(self) -> None:
        super().__post_init__()
        resolve_vol_static(self)
        if not self.static.common:
            raise ValueError("CommonSVStep needs VolStatic(common=True)")

    # ------------------------------------------------------------------ static layout
    @property
    def window(self) -> int:
        """``W = min(block, T_pad)``: the static window of one block."""
        return int(min(self.static.csv_block, self.T_pad))

    @property
    def n_blocks(self) -> int:
        """``1 + ⌈T_pad/block⌉`` blocks (``B_0`` and ``B_1 … B_⌈T_pad/block⌉``)."""
        block = self.static.csv_block
        return 1 + -(-int(self.T_pad) // block)

    def block_bounds(self, b: Array, o: Array) -> tuple[Array, Array]:
        """``(lo, hi)`` of block ``b`` at offset ``o`` (before the intersection with ``[0, T)``)."""
        block = self.static.csv_block
        b = jnp.asarray(b, jnp.int32)
        lo = jnp.where(b == 0, 0, o + (b - 1) * block).astype(jnp.int32)
        hi = jnp.where(b == 0, o, lo + block).astype(jnp.int32)
        return lo, hi

    def padded(self, h: Array, log_q: Array, rho: Array, log_sigma: Array, T_eff: Array):
        """``(hf, lqf, kd, ko)``: the path, ``log q`` and ``K_h`` padded to ``T_pad + W``."""
        W = self.window
        z = jnp.zeros((W,), _F64)
        hf = jnp.concatenate([jnp.asarray(h, _F64), z])
        lqf = jnp.concatenate([jnp.asarray(log_q, _F64), z])
        kd, ko = ar1_precision(rho, log_sigma, T_eff, self.T_pad + W)
        return hf, lqf, kd, ko

    def block_of(
        self, h: Array, log_q: Array, rho: Array, log_sigma: Array, T_eff: Array,
        lo: Array, hi: Array,
    ) -> HBlock:  # fmt: skip
        """The :class:`HBlock` of ``[lo, hi)`` at the state ``(h, ρ, σ_h)`` (tests and the step)."""
        hf, lqf, kd, ko = self.padded(h, log_q, rho, log_sigma, T_eff)
        return block_system(hf, lqf, kd, ko, lo, hi, T_eff, int(self.T_pad), self.window)

    def proposal(self, blk: HBlock, n: Array) -> TridiagNewton:
        """The block's proposal ``N(ĥ, G_B⁻¹)``: Newton from the ``csv_newton_start`` start."""
        x0, max_iter = self.impl("csv_newton_start")(blk.log_q, blk.h, blk.mask, n)
        return tridiag_newton(
            blk.log_q, n, blk.diag, blk.off, blk.lin, blk.mask, x0, max_iter=int(max_iter)
        )

    def draw_block(
        self, prop: TridiagNewton, blk: HBlock, z: Array, u: Array, n: Array
    ) -> tuple[Array, Array, Array, Array]:
        """``(h_B new, accepted, log α, ok)`` of one block for injected noise ``z`` (f64[W]) and
        uniform ``u`` (D-17): ``h* = ĥ + L⁻ᵀz`` on the block, accepted iff ``log u < log α``."""
        on = blk.mask > 0
        dev, _, _ = tridiag_precision_sample(prop.diag, prop.off, jnp.zeros_like(z), z)
        x_new = jnp.where(on, prop.mode + dev, blk.h)
        log_alpha = h_log_accept(x_new, blk.h, prop, blk, n)
        ok = prop.info.ok & jnp.isfinite(log_alpha) & jnp.all(jnp.isfinite(x_new))
        accept = ok & (jnp.log(u) < log_alpha)
        return jnp.where(accept & on, x_new, blk.h), accept, log_alpha, ok

    # ------------------------------------------------------------------ the three moves
    def h_step(
        self, key: Key, h: Array, log_q: Array, rho: Array, log_sigma: Array, n: Array,
        T_eff: Array,
    ) -> tuple[Array, Array, Array, StepInfo]:  # fmt: skip
        """The blocked sequential independence MH of ``h`` (module docstring). Returns the new
        ``h`` (f64[T_pad], exactly 0 on padded states), the accepted fraction of the non-empty
        blocks, the number of blocks whose Newton did not converge and the step info."""
        T_pad, W = int(self.T_pad), self.window
        o = jr.randint(sub_key(key, Sub.CSV_OFFSET), (), 0, self.static.csv_block, jnp.int32)
        hf, lqf, kd, ko = self.padded(h, log_q, rho, log_sigma, T_eff)
        k_h, k_u = sub_key(key, Sub.CSV_H), sub_key(key, Sub.CSV_H_ACC)

        def body(carry, b):
            hf, acc, cnt, cap, ok, piv = carry
            lo, hi = self.block_bounds(b, o)
            blk = block_system(hf, lqf, kd, ko, lo, hi, T_eff, T_pad, W)
            prop = self.proposal(blk, n)
            bu = jnp.asarray(b, jnp.uint32)
            z = jr.normal(jr.fold_in(k_h, bu), (W,), dtype=_F64)
            u = R.uniform_tiny(jr.fold_in(k_u, bu), ())
            new, accepted, _, ok_b = self.draw_block(prop, blk, z, u, n)
            nonempty = jnp.any(blk.mask > 0)
            hf = lax.dynamic_update_slice(hf, new, (blk.start,))
            acc = acc + jnp.where(nonempty & accepted, 1.0, 0.0)
            cnt = cnt + jnp.where(nonempty, 1.0, 0.0)
            cap = cap + jnp.where(nonempty & ~prop.converged, 1.0, 0.0)
            ok = ok & jnp.where(nonempty, ok_b, True)
            piv = jnp.minimum(piv, jnp.where(nonempty, prop.info.min_pivot, jnp.inf))
            return (hf, acc, cnt, cap, ok, piv), None

        zero = jnp.zeros((), _F64)
        init = (hf, zero, zero, zero, jnp.ones((), bool), jnp.full((), jnp.inf, _F64))
        (hf, acc, cnt, cap, ok, piv), _ = lax.scan(
            body, init, jnp.arange(self.n_blocks, dtype=jnp.int32)
        )
        frac = acc / jnp.maximum(cnt, 1.0)
        return hf[:T_pad], frac, cap, StepInfo(ok, piv)

    def rho_step(
        self, key: Key, h: Array, rho: Array, log_sigma: Array, T_eff: Array, csv: CommonSVPrior
    ) -> tuple[Array, Array]:
        """``ρ | h, σ_h`` (module docstring): ``(ρ new, accepted)``."""
        kind = self.static.csv_rho_prior
        s_xy, s_xx, h0 = rho_sufficient(h, T_eff)
        prop = rho_proposal(s_xy, s_xx, log_sigma, csv, kind)
        u = R.uniform_tiny(sub_key(key, Sub.CSV_RHO), ())
        rho_star = self.rho_draw(prop, u)
        la = self.impl("csv_rho_accept")(
            rho_star, rho, h0, log_sigma, s_xy, s_xx, csv, prop.uniform, kind
        )
        u_acc = R.uniform_tiny(sub_key(key, Sub.CSV_RHO_ACC), ())
        accept = jnp.log(u_acc) < la
        return jnp.where(accept, rho_star, rho), accept

    @staticmethod
    def rho_draw(prop: RhoProposal, u: Array) -> Array:
        """``ρ*`` for the injected uniform ``u``: the ``u``-quantile of ``TN(ρ̂, K_ρ⁻¹; −1, 1)``
        (log-space inverse CDF), or ``−1 + 2u`` on the uniform branch."""
        sd = 1.0 / jnp.sqrt(jnp.where(prop.prec > 0.0, prop.prec, 1.0))
        z = R.truncated_std_normal(u, (-1.0 - prop.mean) / sd, (1.0 - prop.mean) / sd)
        return jnp.where(prop.uniform, -1.0 + 2.0 * u, prop.mean + sd * z)

    def sigma_step(
        self, key: Key, h: Array, rho: Array, T_eff: Array, csv: CommonSVPrior
    ) -> tuple[Array, Array]:
        """``σ_h² | h, ρ`` (exact): ``(log σ_h, ok)``."""
        c = sigma2_conditional(h, rho, T_eff, csv, self.static.csv_sigma2_prior)
        k = sub_key(key, Sub.CSV_S2)
        if self.static.csv_sigma2_prior == "inv_gamma":
            log_s2 = c.log_scale - R.loggamma(k, c.shape)
            ok = jnp.isfinite(log_s2)
        else:
            log_s2, ok = R.log_gig(k, c.lam, c.log_psi, c.log_chi)
            ok = ok & jnp.isfinite(log_s2)
        return 0.5 * log_s2, ok

    # ------------------------------------------------------------------ GlobalStep protocol
    def init(
        self, key: Key, st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache
    ) -> ChainState:
        """Cold start (P3 §5.5, E133; module docstring)."""
        csv = prior.shared.csv
        if self.static.csv_rho_prior == "truncated_normal":
            centre = jnp.asarray(csv.rho_mean, _F64)
        else:
            centre = 2.0 * csv.rho_a / (csv.rho_a + csv.rho_b) - 1.0
        rho = jnp.clip(centre, -INIT_RHO_BOUND, INIT_RHO_BOUND)
        if self.static.csv_sigma2_prior == "inv_gamma":
            c0, C0 = csv.s2_shape, csv.s2_scale
            s2 = jnp.where(c0 > 1.0, C0 / jnp.where(c0 > 1.0, c0 - 1.0, 1.0), C0 / (c0 + 1.0))
        else:
            s2 = jnp.asarray(csv.s2_B, _F64)
        jitter = jr.normal(sub_key(key, Sub.CSV_S2), (), dtype=_F64)
        log_sigma = 0.5 * jnp.log(s2) + INIT_LOG_SIGMA_SD * jitter
        T = int(self.T_pad)
        zf, zh = jnp.zeros((), _F64), jnp.zeros((T,), _F64)
        h = stationary_ar1_path(sub_key(key, Sub.CSV_H), rho, log_sigma, T)
        h = jnp.where(jnp.asarray(data.tmask, _F64) > 0.0, h, 0.0)
        gv = CommonSVGlobal(
            h=h, rho=rho, log_sigma=log_sigma, acc_csv_h=zf, acc_csv_rho=zf, acc_csv_newton=zf,
            rb_ch_sum=zh, rb_ch_sq=zh,
        )  # fmt: skip
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, vol=gv))

    def log_q(self, st: ChainState, ctx: SweepCtx) -> Array:
        """``log q_t`` of the state's cores and the equation phase's residuals."""
        e = ctx.eq_out.e
        tmask = jnp.asarray(ctx.data.tmask, _F64)
        return self.impl("csv_q")(log_e2(e, tmask[None, :]), st.eq.vol.core.log_lam, tmask)

    def sample(
        self, key: Key, st: ChainState, ctx: SweepCtx, eq_map: EqMap
    ) -> tuple[ChainState, StepInfo]:
        """``h`` (blocked MH) → ``ρ`` (MH) → ``σ_h²`` (exact) given ``q`` (module docstring)."""
        del eq_map
        csv = ctx.prior.shared.csv
        gv = st.glob.vol
        T_eff = ctx.data.T_eff
        log_q = self.log_q(st, ctx)
        h, frac, cap, info = self.h_step(key, gv.h, log_q, gv.rho, gv.log_sigma, csv.n_eq, T_eff)
        rho, acc_rho = self.rho_step(key, h, gv.rho, gv.log_sigma, T_eff, csv)
        log_sigma, ok_s = self.sigma_step(key, h, rho, T_eff, csv)
        ok = info.ok & ok_s & jnp.isfinite(rho) & (jnp.abs(rho) < 1.0) & jnp.isfinite(log_sigma)
        new = CommonSVGlobal(
            h=h,
            rho=rho,
            log_sigma=log_sigma,
            acc_csv_h=gv.acc_csv_h + frac,
            acc_csv_rho=gv.acc_csv_rho + acc_rho.astype(_F64),
            acc_csv_newton=gv.acc_csv_newton + cap,
            rb_ch_sum=gv.rb_ch_sum + h,
            rb_ch_sq=gv.rb_ch_sq + h * h,
        )
        st = dataclasses.replace(st, glob=dataclasses.replace(st.glob, vol=new))
        return st, StepInfo(ok, info.min_pivot)

    def log_prior(self, glob: GlobalState, prior: PriorArrays, tmask: Array | None = None) -> Array:
        """``log p(h, ρ, σ_h)`` w.r.t. Lebesgue measure on ``(h_{0:T−1}, ρ, σ_h)``: the ρ prior
        (the truncated normal normalised on ``(−1, 1)``, or the Beta on ``(ρ + 1)/2`` with its
        ``½`` Jacobian), the ``σ_h`` density (the prior on ``σ_h²`` times ``2σ_h``), the stationary
        ``h_0`` and the transitions of the real dates. The real dates need the data mask: the
        protocol's two-argument call raises (the monitor calls :meth:`log_prior_data`)."""
        if tmask is None:
            raise ValueError(
                "CommonSVStep.log_prior needs the data mask tmask (the real dates of h); "
                "use log_prior_data(glob, prior, data)"
            )
        gv, csv = glob.vol, prior.shared.csv
        h, rho, log_sigma = gv.h, gv.rho, gv.log_sigma
        tmask = jnp.asarray(tmask, _F64)
        T = jnp.sum(tmask).astype(jnp.int32)
        if self.static.csv_rho_prior == "truncated_normal":
            m, v = csv.rho_mean, csv.rho_var
            sd = jnp.sqrt(v)
            log_mass = _log_diff_ndtr(
                (1.0 - m) / sd, (-1.0 - m) / sd
            )  # P(|ρ| < 1) before truncation
            lp = -0.5 * (_LOG_2PI + jnp.log(v) + (rho - m) ** 2 / v) - log_mass
        else:
            a, b = csv.rho_a, csv.rho_b
            log_b = gammaln(a) + gammaln(b) - gammaln(a + b)
            lp = (
                (a - 1.0) * (jnp.log1p(rho) - _LOG_2)
                + (b - 1.0) * (jnp.log1p(-rho) - _LOG_2)
                - log_b
                - _LOG_2
            )
        sigma = jnp.exp(log_sigma)
        if self.static.csv_sigma2_prior == "inv_gamma":
            c0, C0 = csv.s2_shape, csv.s2_scale
            lp = lp + (
                c0 * jnp.log(C0) - gammaln(c0) - (c0 + 1.0) * 2.0 * log_sigma
                - C0 * jnp.exp(-2.0 * log_sigma) + _LOG_2 + log_sigma
            )  # fmt: skip
        else:
            B = csv.s2_B
            lp = lp + _LOG_2 - 0.5 * (_LOG_2PI + jnp.log(B) + sigma * sigma / B)
        s2 = jnp.exp(2.0 * log_sigma)
        v0 = s2 / ((1.0 - rho) * (1.0 + rho))
        lp = lp - 0.5 * (_LOG_2PI + jnp.log(v0) + h[0] * h[0] / v0)
        t = jnp.arange(1, h.shape[0], dtype=jnp.int32)
        resid = h[1:] - rho * h[:-1]
        trans = -0.5 * (_LOG_2PI + 2.0 * log_sigma + resid * resid / s2)
        return lp + jnp.sum(jnp.where(t < T, trans, 0.0))

    def log_prior_data(self, glob: GlobalState, prior: PriorArrays, data: EngineData) -> Array:
        """:meth:`log_prior` with the data mask of ``data`` (the monitor's call)."""
        return self.log_prior(glob, prior, data.tmask)

    def extract(self, gv: CommonSVGlobal, T_eff: Array) -> CommonSVDraw:
        """``CommonSVDraw(h_T = h[T_eff − 1], ρ, σ_h)`` (a dynamic index, never ``h[-1]``)."""
        t = jnp.asarray(T_eff, jnp.int32) - 1
        h_T = lax.dynamic_index_in_dim(gv.h, t, keepdims=False)
        return CommonSVDraw(h_T=h_T, rho=gv.rho, sigma=jnp.exp(gv.log_sigma))

    def future(self, key: Key, gd: GlobDraw, H: int) -> Array:
        """``h_{T+j} = ρh_{T+j−1} + σ_hη_j``, ``j = 1..H`` (f64[H]) from ``h_T`` of ``gd.vol``; the
        normals from ``fold_in(key, Sub.CSV_H)``."""
        cd = gd.vol
        eta = jr.normal(sub_key(key, Sub.CSV_H), (int(H),), dtype=_F64)

        def step(prev: Array, e: Array) -> tuple[Array, Array]:
            nxt = cd.rho * prev + cd.sigma * e
            return nxt, nxt

        return lax.scan(step, jnp.asarray(cd.h_T, _F64), eta)[1]


def _log_diff_ndtr(a: Array, b: Array) -> Array:
    """``log(Φ(a) − Φ(b))`` for ``a > b``, in whichever tail keeps it accurate."""
    upper = b > 0.0  # both in the upper tail: Φ(a) − Φ(b) = Φ(−b) − Φ(−a)
    hi = jnp.where(upper, -b, a)
    lo = jnp.where(upper, -a, b)
    lh, ll = log_ndtr(hi, series_order=10), log_ndtr(lo, series_order=10)
    return lh + jnp.log1p(-jnp.exp(ll - lh))
