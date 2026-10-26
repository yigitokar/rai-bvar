"""Global-local and selection coefficient priors and their scale step (P3 DESIGN §3; D-36…D-44).

Five ``CoefPrior`` kernels (:class:`Horseshoe`, :class:`NormalGamma`, :class:`R2D2`,
:class:`DirichletLaplace`, :class:`SSVS`) govern the **deviations** ``d_j = coefficient − m_j`` of
the lag coefficients (P3 §3.1): ``d_j | v_j ~ N(0, r_j v_j)`` (SSVS ``N(0, τ²_{γ_j, j})``), with
``r_j`` from ``CoefPriorArrays.gl.log_r`` (D-37) and the family's mixing variance ``v_j``.
Deterministic slots keep the fixed Normal prior ``N(mean0, 1/prec0)`` (D-36). Every density and
conditional reads ``ℓ_j = log|d_j|``, which the mean kernels deliver exactly (P3 §3.4).

**Moments** (P3 §3.3). :meth:`moments_eq` returns ``GaussianLogDiag(mean0, log_prec, 1)`` with
``log_prec_j = −log r_j − log v_j`` on GL slots (``log v``: HS ``log ϑ + log ζ_g``, NG ``log ϖ``,
R2D2 ``log ϱ + log ϖ − log 2``, DL ``log ϱ + 2 log θ̃``; SSVS ``−2 log τ_{γ_j j}``) and
``log prec0`` elsewhere, which selects the mean's deviation path (``MeanStatic.coef_log``).

**Scale step** (:class:`GLScaleStep`, ``Block.G_COEF_GLOBAL``, post phase, D-40). Given the
coefficients the scales are independent of the data, the volatility and the indicators, so one
step draws every local, group and hyper parameter by the lines of P3 §3.2, **in the printed
order** (*derived here*; checked by the Tier U density-ratio identities):

    HS     ϑ_j | ν_j, ζ_g, d_j  ~ IG(1, 1/ν_j + d_j²/(2 r_j ζ_g))
           ν_j | ϑ_j            ~ IG(1, 1 + 1/ϑ_j)
           ζ_g | τ_g, ϑ, d      ~ IG((n_g + 1)/2, 1/τ_g + Σ_{j∈g} d_j²/(2 r_j ϑ_j))
           τ_g | ζ_g            ~ IG(1, 1/s² + 1/ζ_g)
    NG     ϖ_j | d_j, ξ_g       ~ GIG(a_g − ½, 2ξ_g, d_j²/r_j)
           a_g | ϖ (ξ integrated),  then  ξ_g | a_g, ϖ ~ G(n_g a_g + b, 2c/a_g + W_g)
    R2D2   ϱ_j | d_j, ϖ_j       ~ GIG(½, 1, 2d_j²/(r_j ϖ_j))
           ϖ_j | d_j, ϱ_j, ξ_g  ~ GIG(a_g − ½, 2ξ_g, 2d_j²/(r_j ϱ_j))
           a_g | ϖ (ξ integrated),  then  ξ_g | a_g, ϖ ~ G(n_g a_g + b, 1 + W_g)
    DL     θ̃_j | d_j (ϱ integrated) ~ GIG(a_g − 1, 1, 2|d_j|/√r_j),  ϱ_j | d_j, θ̃_j ~ GIG(½, 1, d_j²/(r_j θ̃_j²))
           a_g | θ̃
    SSVS   logit P(γ_j = 1 | ·) = log p_g − log(1 − p_g) + log τ_0j − log τ_1j − ½ (d_j²/τ_1j² − d_j²/τ_0j²)
           p_g | γ ~ Beta(s₁ + N1_g, s₂ + n_g − N1_g)

with ``S_g = Σ_{j∈g} log ϖ_j``, ``W_g = Σ_{j∈g} ϖ_j`` (``log W_g`` a segment logsumexp) and the
collapsed ``a``-weights of :func:`a_log_weights`. Every scale is drawn on the log scale
(``log x = log β − LG(α)`` for ``IG(α, β)``, ``LG(α) − log β`` for ``G(α, β)``,
``random.log_gig`` for the GIG); no floor, truncation or jitter is ever added (D-41).

**Keys** (P3 §2.4): the step key ``fold_in(it_key, Block.G_COEF_GLOBAL)`` → ``Sub.GL_*`` → the flat
coordinate index ``i·k + r`` (uint32) for per-coordinate draws, or the group id for per-group
draws. Lines: ``GL_LOCAL`` the first local line (HS ϑ, NG ϖ, R2D2 ϱ, DL θ̃, SSVS γ), ``GL_LOCAL_AUX``
the second (HS ν, R2D2 ϖ, DL ϱ), ``GL_GROUP`` the group scale (HS ζ, NG/R2D2 ξ, SSVS p),
``GL_GROUP_AUX`` the HS auxiliary τ, ``GL_A`` the hyperprior index ``a_g``.

**Overridable internals** (P3 §2.4, §9.6; through ``KernelBase.impl``, test seam only):
``gl_local_conditional``, ``gl_group_conditional``, ``gl_a_conditional`` (the whole collapsed
hyper block, so a mutant may reorder it), ``ssvs_logodds`` and ``gig_sampler``. Their default
implementations are the module functions of the same names.

**Initialisation** (P3 §3.7; :meth:`init_global`, run before every pre-phase init and the mean's
cold draw). The local scales start at their unit multipliers and the group scales so that
``r_j v_j`` is the Mode I Minnesota variance at ``λ = 0.2`` (own lag 1): HS ``ϑ = ν = 1``,
``ζ_g = 0.04``, ``τ_g = 1/s²``; NG and R2D2 the matching rate ``ξ_g = a_g/0.04`` with ``ϖ_j =
a_g/ξ_g`` (R2D2 ``ϱ_j = 2``); DL ``θ̃_j = ϱ_j = 2``; ``a_g`` at the grid point nearest 0.1; SSVS
``p_g = 0.5`` (Beta hyperprior; else the fixed ``p``) and ``γ_j = 1``. Each group log scale
(``log ζ_g``, ``log τ_g``, ``log ξ_g``) is jittered by ``N(0, 0.5²)`` per chain (keys
``Sub.GL_GROUP``/``GL_GROUP_AUX`` → group id of the init stream).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc import random as R
from rai_bvar.engines.mcmc.config import CoefPriorStatic, StaticConfig
from rai_bvar.engines.mcmc.keys import Block, Key, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import EqMap, KernelBase
from rai_bvar.engines.mcmc.state import (
    ChainState,
    DLEq,
    DLGlobal,
    Empty,
    EngineData,
    EqPrior,
    GaussianLogDiag,
    GlobalState,
    GlobDraw,
    HSEq,
    HSGlobal,
    NGEq,
    NGGlobal,
    PriorArrays,
    R2D2Eq,
    R2D2Global,
    SSVSEq,
    SSVSGlobal,
    StepInfo,
    SweepCtx,
)

Array = jax.Array
PyTree = Any

__all__ = [
    "R2D2",
    "SSVS",
    "BetaParams",
    "DirichletLaplace",
    "GIGParams",
    "GLScaleStep",
    "GammaParams",
    "GroupInfo",
    "Horseshoe",
    "IGParams",
    "NormalGamma",
    "SSVSDraw",
    "a_log_weights",
    "a_values",
    "gig_sampler",
    "gl_a_conditional",
    "gl_group_conditional",
    "gl_local_conditional",
    "group_logsumexp",
    "group_sum",
    "ssvs_logodds",
    "xi_params",
]

LOG2 = math.log(2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_LGAMMA_HALF = float(gammaln(0.5))
INIT_LOG_VAR = math.log(0.04)  # λ² at λ = 0.2: the Mode I own-lag-1 variance in units of r_j
INIT_A = 0.1  # a_g starts at the grid point nearest this value (P3 §3.7)
INIT_JITTER_SD = 0.5  # per-chain N(0, 0.5²) jitter of every group log scale

Family = Literal["horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"]
_HAS_XI = ("normal_gamma", "r2d2")
_HAS_A = ("normal_gamma", "r2d2", "dirichlet_laplace")


# --------------------------------------------------------------------------- conditional parameters


class IGParams(NamedTuple):
    """``x ~ IG(shape, scale)`` (shape ``α``, scale ``β``): ``log x = log β − LG(α)``."""

    shape: Array
    log_scale: Array


class GammaParams(NamedTuple):
    """``x ~ G(shape, rate)``: ``log x = LG(α) − log β``."""

    shape: Array
    log_rate: Array


class GIGParams(NamedTuple):
    """``x ~ GIG(λ, ψ, χ)``, density ``∝ x^{λ−1} exp(−(ψx + χ/x)/2)``; ψ and χ in logs."""

    lam: Array
    log_psi: Array
    log_chi: Array


class BetaParams(NamedTuple):
    """``p ~ Beta(a, b)``."""

    a: Array
    b: Array


class GroupInfo(NamedTuple):
    """The static group layout of one prior (P3 §3.1, D-38): ``group`` and ``mask`` (``n × k``,
    the GL arrays), the number of groups ``G`` and the hyperprior grid size ``a_grid`` (Python
    ints, static), and ``n_g`` (``G``) the GL coordinates of each group."""

    group: Array
    mask: Array
    G: int
    a_grid: int
    n_g: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class SSVSDraw:
    """The SSVS indicators of one stored draw (``ParamDraw.glob.coef``; int8, ``n × k``, 0 off the
    lag slots), stored with the parameter draws (``store_thin``) for the inclusion MCSE (E123)."""

    incl: Array


# --------------------------------------------------------------------------- group helpers


def group_sum(x: Array, info: GroupInfo) -> Array:
    """``Σ_{j∈g} x_j`` over the GL coordinates of every group (``G``)."""
    v = jnp.where(info.mask > 0, x, 0.0)
    return jax.ops.segment_sum(v.ravel(), info.group.ravel(), num_segments=info.G)


def group_logsumexp(x: Array, info: GroupInfo) -> Array:
    """``log Σ_{j∈g} exp(x_j)`` over the GL coordinates of every group (``G``; ``−∞`` for an
    empty group): the segment maximum is subtracted first, so no term overflows."""
    v = jnp.where(info.mask > 0, x, -jnp.inf).ravel()
    g = info.group.ravel()
    mx = jax.ops.segment_max(v, g, num_segments=info.G)
    safe = jnp.where(jnp.isfinite(mx), mx, 0.0)
    s = jax.ops.segment_sum(jnp.where(v > -jnp.inf, jnp.exp(v - safe[g]), 0.0), g, info.G)
    return safe + jnp.log(s)


def _bcast(v: Array, info: GroupInfo) -> Array:
    """A group-level value at every coordinate (``n × k``; off the GL slots: group 0's value)."""
    return jnp.take(v, info.group)


def a_values(glob: PyTree, gl: Any, info: GroupInfo) -> Array:
    """``a_g`` (``G``): the grid value at ``a_idx``, or the fixed ``exp(log_a_fixed)``."""
    if info.a_grid:
        return jnp.take(gl.a_support, glob.a_idx)
    return jnp.broadcast_to(jnp.exp(gl.log_a_fixed), (info.G,))


def _ng_log_beta(a: Array, gl: Any) -> Array:
    """``log(2c/a)``, the rate of ``ξ_g | a`` under NG; 0 when ``c`` is tied (``c < 0`` encodes
    ``c = a/2``, ``GLShared`` docstring)."""
    c = gl.c
    return jnp.where(c < 0.0, 0.0, LOG2 + jnp.log(jnp.where(c > 0.0, c, 1.0)) - jnp.log(a))


# --------------------------------------------------------------------------- the lines of P3 §3.2


def gl_local_conditional(
    family: str, line: str, ell: Array, log_r: Array, loc: PyTree, grp: dict[str, Array], gl: Any
) -> IGParams | GIGParams:
    """Parameters of the local line ``line`` of ``family`` at every coordinate (``n × k``).

    ``ell = log|d|``, ``log_r`` the scales, ``loc`` the local state with the lines already drawn in
    this step, ``grp`` the group-level values broadcast to the coordinates (``log_zeta``,
    ``log_tau``; ``log_xi``, ``a``), ``gl`` the ``GLShared`` hyperparameters. Overridable
    (mutants M20, M21, M23, M24, M27)."""
    if family == "horseshoe":
        one = jnp.ones_like(ell)
        if line == "theta":  # IG(1, 1/ν + d²/(2 r ζ))
            return IGParams(
                one, jnp.logaddexp(-loc.log_nu, 2.0 * ell - LOG2 - log_r - grp["log_zeta"])
            )
        if line == "nu":  # IG(1, 1 + 1/ϑ)
            return IGParams(one, jnp.logaddexp(0.0, -loc.log_theta))
    elif family == "normal_gamma":
        if line == "w":  # GIG(a − ½, 2ξ, d²/r)
            return GIGParams(grp["a"] - 0.5, LOG2 + grp["log_xi"], 2.0 * ell - log_r)
    elif family == "r2d2":
        if line == "varrho":  # GIG(½, 1, 2d²/(r ϖ))
            return GIGParams(
                jnp.full_like(ell, 0.5), jnp.zeros_like(ell), LOG2 + 2.0 * ell - log_r - loc.log_w
            )
        if line == "w":  # GIG(a − ½, 2ξ, 2d²/(r ϱ))
            return GIGParams(
                grp["a"] - 0.5, LOG2 + grp["log_xi"], LOG2 + 2.0 * ell - log_r - loc.log_varrho
            )
    elif family == "dirichlet_laplace":
        if line == "theta":  # GIG(a − 1, 1, 2|d|/√r)  (ϱ integrated)
            return GIGParams(grp["a"] - 1.0, jnp.zeros_like(ell), LOG2 + ell - 0.5 * log_r)
        if line == "varrho":  # GIG(½, 1, d²/(r θ̃²))
            return GIGParams(
                jnp.full_like(ell, 0.5),
                jnp.zeros_like(ell),
                2.0 * ell - log_r - 2.0 * loc.log_theta,
            )
    raise ValueError(f"no local line {line!r} for the {family!r} prior")


def xi_params(family: str, a: Array, loc: PyTree, info: GroupInfo, gl: Any) -> GammaParams:
    """``ξ_g | a_g, ϖ ~ G(n_g a_g + b, β_g + W_g)``: NG ``β_g = 2c/a_g``, R2D2 ``β_g = 1``."""
    log_W = group_logsumexp(loc.log_w, info)
    log_beta = _ng_log_beta(a, gl) if family == "normal_gamma" else jnp.zeros_like(a)
    return GammaParams(info.n_g * a + gl.b, jnp.logaddexp(log_beta, log_W))


def gl_group_conditional(
    family: str,
    line: str,
    ell: Array,
    log_r: Array,
    loc: PyTree,
    glob: PyTree,
    info: GroupInfo,
    gl: Any,
) -> IGParams | GammaParams | BetaParams:
    """Parameters of the group line ``line`` of ``family`` (``G``) given the coordinates' ``ell``,
    ``log_r`` and the drawn local state ``loc``, the group state ``glob`` and the layout ``info``.
    Overridable (mutant M19)."""
    if family == "horseshoe":
        if line == "zeta":  # IG((n_g + 1)/2, 1/τ + Σ d²/(2 r ϑ))
            terms = 2.0 * ell - LOG2 - log_r - loc.log_theta
            return IGParams(
                0.5 * (info.n_g + 1.0),
                jnp.logaddexp(-glob.log_tau, group_logsumexp(terms, info)),
            )
        if line == "tau":  # IG(1, 1/s² + 1/ζ)
            return IGParams(
                jnp.ones((info.G,)),
                jnp.logaddexp(-2.0 * jnp.log(gl.global_scale), -glob.log_zeta),
            )
    elif family in _HAS_XI and line == "xi":
        return xi_params(family, a_values(glob, gl, info), loc, info, gl)
    elif family == "ssvs" and line == "p":
        N1 = group_sum(loc.gamma.astype(jnp.float64), info)
        return BetaParams(gl.s1 + N1, gl.s2 + info.n_g - N1)
    raise ValueError(f"no group line {line!r} for the {family!r} prior")


def a_log_weights(family: str, loc: PyTree, info: GroupInfo, gl: Any) -> Array:
    """Unnormalised log weights ``log P(a_g = ã | ·)`` (``G × A``) over the hyperprior grid
    (P3 §3.2; *derived here*):

    NG    ``log π(ã) + (ã − 1)S_g − n_g lnΓ(ã) + b log β(ã) + lnΓ(n_g ã + b) − (n_g ã + b) log(β(ã) + W_g)``,
          ``β(ã) = 2c/ã`` (1 when tied), ``ξ_g`` integrated
    R2D2  the same with ``β = 1``
    DL    ``log π(ã) − n_g ã log 2 + (ã − 1) Σ_{j∈g} log θ̃_j − n_g lnΓ(ã)``
    """
    sup = gl.a_support[None, :]
    lw = gl.a_logw[None, :]
    n = info.n_g[:, None]
    if family == "dirichlet_laplace":
        S = group_sum(loc.log_theta, info)[:, None]
        return lw - n * sup * LOG2 + (sup - 1.0) * S - n * gammaln(sup)
    S = group_sum(loc.log_w, info)[:, None]
    log_W = group_logsumexp(loc.log_w, info)[:, None]
    log_beta = _ng_log_beta(sup, gl) if family == "normal_gamma" else jnp.zeros_like(sup)
    shape = n * sup + gl.b
    return (
        lw + (sup - 1.0) * S - n * gammaln(sup) + gl.b * log_beta + gammaln(shape)
        - shape * jnp.logaddexp(log_beta, log_W)
    )  # fmt: skip


def gl_a_conditional(
    family: str,
    keys_a: Array,
    keys_xi: Array,
    loc: PyTree,
    glob: PyTree,
    info: GroupInfo,
    gl: Any,
) -> tuple[Array, Array | None]:
    """The collapsed hyper block (P3 §3.2): ``a_g`` from its ``ξ``-marginal (one inverse-CDF
    categorical on :func:`a_log_weights`, key ``keys_a[g]``) and **then** ``log ξ_g | a_g`` from
    :func:`xi_params` (key ``keys_xi[g]``; NG, R2D2); DL draws ``a_g | θ̃`` only (``None``).
    Returns ``(a_idx, log_xi)``. Overridable (mutants M22, M35)."""
    logw = a_log_weights(family, loc, info, gl)
    a_idx = jax.vmap(R.categorical)(keys_a, logw).astype(jnp.int32)
    if family not in _HAS_XI:
        return a_idx, None
    p = xi_params(family, jnp.take(gl.a_support, a_idx), loc, info, gl)
    return a_idx, _gamma(keys_xi, p)


def ssvs_logodds(
    ell: Array, log_tau0: Array, log_tau1: Array, log_p: Array, log1m_p: Array
) -> Array:
    """``logit P(γ = 1 | ·) = log p − log(1 − p) + log τ₀ − log τ₁ − ½(d²/τ₁² − d²/τ₀²)``.

    The quadratic term is ``½ s e^{2ℓ − 2 log τ_<}(1 − e^{−2|log τ₁ − log τ₀|})`` with ``τ_<`` the
    smaller scale and ``s = sign(log τ₁ − log τ₀)``: one exponential, so ``d²/τ²`` never forms
    ``∞ − ∞`` (0 when ``τ₀ = τ₁``). Overridable (mutant M25)."""
    diff = log_tau1 - log_tau0
    lo = jnp.minimum(log_tau0, log_tau1)
    quad = 0.5 * jnp.sign(diff) * jnp.exp(2.0 * ell - 2.0 * lo) * (-jnp.expm1(-2.0 * jnp.abs(diff)))
    return log_p - log1m_p - diff + quad


def gig_sampler(key: Key, lam: Array, log_psi: Array, log_chi: Array) -> tuple[Array, Array]:
    """One GIG draw on the log scale (``random.log_gig``; D-42). Overridable (mutant M34)."""
    return R.log_gig(key, lam, log_psi, log_chi)


# --------------------------------------------------------------------------- draws


def _coord_keys(key: Key, sub: Sub, n_coords: int) -> Array:
    """``fold_in(sub_key(key, sub), j)`` for the flat coordinate index ``j`` (uint32)."""
    base = sub_key(key, sub)
    return jax.vmap(lambda j: random.fold_in(base, j))(jnp.arange(n_coords, dtype=jnp.uint32))


def _group_keys(key: Key, sub: Sub, G: int) -> Array:
    """``fold_in(sub_key(key, sub), g)`` for the group id ``g`` (uint32)."""
    base = sub_key(key, sub)
    return jax.vmap(lambda g: random.fold_in(base, g))(jnp.arange(G, dtype=jnp.uint32))


def _ig(keys: Array, p: IGParams, on: Array | None = None) -> Array:
    shape = jnp.broadcast_to(p.shape, p.log_scale.shape)
    if on is not None:
        shape = jnp.where(on, shape, 1.0)
    lg = jax.vmap(R.loggamma)(keys, shape.ravel()).reshape(shape.shape)
    return p.log_scale - lg


def _gamma(keys: Array, p: GammaParams) -> Array:
    shape = jnp.broadcast_to(p.shape, p.log_rate.shape)
    lg = jax.vmap(R.loggamma)(keys, shape.ravel()).reshape(shape.shape)
    return lg - p.log_rate


def _gig(sampler: Any, keys: Array, p: GIGParams, on: Array) -> tuple[Array, Array]:
    """GIG draws on the GL coordinates; the other lanes get the finite stand-in ``GIG(1, 1, 1)``
    so they accept at once and never report ``ok = False``."""
    shape = jnp.broadcast_shapes(p.lam.shape, p.log_psi.shape, p.log_chi.shape)

    def prep(x: Array, fill: float) -> Array:
        return jnp.where(on, jnp.broadcast_to(x, shape), fill).ravel()

    out, ok = jax.vmap(sampler)(keys, prep(p.lam, 1.0), prep(p.log_psi, 0.0), prep(p.log_chi, 0.0))
    return out.reshape(shape), ok.reshape(shape)


def _finite_on(x: Array, on: Array) -> Array:
    return jnp.all(jnp.where(on, jnp.isfinite(x), True))


# --------------------------------------------------------------------------- log densities


def _log_ig(x_log: Array, shape: Array, log_scale: Array) -> Array:
    """``log IG(x; α, β)`` w.r.t. ``x`` from ``log x``."""
    return shape * log_scale - gammaln(shape) - (shape + 1.0) * x_log - jnp.exp(log_scale - x_log)


def _log_gamma(x_log: Array, shape: Array, log_rate: Array) -> Array:
    """``log G(x; α, β)`` (rate) w.r.t. ``x`` from ``log x``."""
    return shape * log_rate - gammaln(shape) + (shape - 1.0) * x_log - jnp.exp(log_rate + x_log)


# --------------------------------------------------------------------------- the scale step


@dataclass(frozen=True)
class GLScaleStep(KernelBase):
    """``GlobalStep`` of every GL coefficient prior: the lines of P3 §3.2 in the printed order
    (module docstring). Reads ``ℓ`` from ``eq.mean.dev_log`` (structural) or
    ``glob.mean.dev_log`` (reduced form), never the data or ``eq.vol``."""

    block: ClassVar[Block] = Block.G_COEF_GLOBAL
    phase: ClassVar[Literal["pre", "post"]] = "post"
    reads: ClassVar[frozenset[str]] = frozenset({"eq.mean", "glob.mean", "eq.coef", "glob.coef"})
    writes: ClassVar[frozenset[str]] = frozenset({"eq.coef", "glob.coef"})
    marginal_of_indicators: ClassVar[bool] = True
    name: ClassVar[str] = "gl_scales"

    static: CoefPriorStatic
    form: Literal["structural", "reduced"] = "structural"

    gl_local_conditional = staticmethod(gl_local_conditional)
    gl_group_conditional = staticmethod(gl_group_conditional)
    gl_a_conditional = staticmethod(gl_a_conditional)
    ssvs_logodds = staticmethod(ssvs_logodds)
    gig_sampler = staticmethod(gig_sampler)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.static.varies:
            raise ValueError(f"GLScaleStep needs a global-local prior, got {self.static.kind!r}")
        if self.form not in ("structural", "reduced"):
            raise ValueError(f"form must be 'structural' or 'reduced', got {self.form!r}")

    @property
    def family(self) -> str:
        return self.static.kind

    # ----------------------------------------------------------------- helpers
    def ell(self, st: ChainState) -> Array:
        """``ℓ = log|d|`` (``n × k``, row = equation) from the mean's state (P3 §3.3)."""
        mean = st.eq.mean if self.form == "structural" else st.glob.mean
        if isinstance(mean.dev_log, Empty):
            raise ValueError("the scale step needs the deviation path's dev_log (coef_log=True)")
        return mean.dev_log

    def info(self, prior: PriorArrays) -> GroupInfo:
        arr = prior.eq.coef.gl
        mask = jnp.asarray(arr.mask, jnp.float64)
        group = jnp.asarray(arr.group, jnp.int32)
        G = self.static.n_groups
        n_g = jax.ops.segment_sum(mask.ravel(), group.ravel(), num_segments=G)
        return GroupInfo(group=group, mask=mask, G=G, a_grid=self.static.a_grid, n_g=n_g)

    # ----------------------------------------------------------------- GlobalStep
    def init(
        self, key: Key, st: ChainState, data: EngineData, prior: PriorArrays, cache: Any
    ) -> ChainState:
        """Identity: the coefficient prior's ``init_global`` set the scales (P3 §2.4)."""
        return st

    def sample(
        self, key: Key, st: ChainState, ctx: SweepCtx, eq_map: EqMap
    ) -> tuple[ChainState, StepInfo]:
        prior = ctx.prior
        arr, gl = prior.eq.coef.gl, prior.shared.gl
        info = self.info(prior)
        on = info.mask > 0
        ell = jnp.where(on, self.ell(st), 0.0)
        log_r = jnp.where(on, arr.log_r, 0.0)
        loc, glob, ok = self.draw(key, ell, log_r, st.eq.coef, st.glob.coef, info, gl, arr)
        st = dataclasses.replace(
            st,
            eq=dataclasses.replace(st.eq, coef=loc),
            glob=dataclasses.replace(st.glob, coef=glob),
        )
        return st, StepInfo(ok, jnp.full((), jnp.inf))

    def draw(
        self,
        key: Key,
        ell: Array,
        log_r: Array,
        loc: PyTree,
        glob: PyTree,
        info: GroupInfo,
        gl: Any,
        arr: Any,
    ) -> tuple[PyTree, PyTree, Array]:
        """The scale step on arrays: new local state, new group state and ``ok`` (every new value
        finite on the GL slots, every GIG accepted)."""
        fam = self.family
        on = info.mask > 0
        N = int(ell.size)
        G = info.G
        local = self.impl("gl_local_conditional")
        group = self.impl("gl_group_conditional")
        sampler = self.impl("gig_sampler")

        def mask(x: Array) -> Array:
            return jnp.where(on, x, 0.0)

        def ig_local(sub: Sub, p: IGParams) -> Array:
            return mask(_ig(_coord_keys(key, sub, N), p, on).reshape(ell.shape))

        def gig_local(sub: Sub, p: GIGParams) -> tuple[Array, Array]:
            x, ok_x = _gig(sampler, _coord_keys(key, sub, N), p, on)
            return mask(x), jnp.all(jnp.where(on, ok_x, True))

        oks = []
        if fam == "horseshoe":
            grp = {"log_zeta": _bcast(glob.log_zeta, info), "log_tau": _bcast(glob.log_tau, info)}
            loc = dataclasses.replace(
                loc, log_theta=ig_local(Sub.GL_LOCAL, local(fam, "theta", ell, log_r, loc, grp, gl))
            )
            loc = dataclasses.replace(
                loc, log_nu=ig_local(Sub.GL_LOCAL_AUX, local(fam, "nu", ell, log_r, loc, grp, gl))
            )
            pz = group(fam, "zeta", ell, log_r, loc, glob, info, gl)
            glob = dataclasses.replace(glob, log_zeta=_ig(_group_keys(key, Sub.GL_GROUP, G), pz))
            pt = group(fam, "tau", ell, log_r, loc, glob, info, gl)
            glob = dataclasses.replace(glob, log_tau=_ig(_group_keys(key, Sub.GL_GROUP_AUX, G), pt))
            oks += [_finite_on(loc.log_theta, on), _finite_on(loc.log_nu, on)]
            oks += [jnp.all(jnp.isfinite(glob.log_zeta)), jnp.all(jnp.isfinite(glob.log_tau))]
        elif fam in ("normal_gamma", "r2d2", "dirichlet_laplace"):
            grp = {"a": _bcast(a_values(glob, gl, info), info)}
            if fam in _HAS_XI:
                grp["log_xi"] = _bcast(glob.log_xi, info)
            first, second = {
                "normal_gamma": ("w", None),
                "r2d2": ("varrho", "w"),
                "dirichlet_laplace": ("theta", "varrho"),
            }[fam]
            leaf = {"w": "log_w", "varrho": "log_varrho", "theta": "log_theta"}
            x, ok1 = gig_local(Sub.GL_LOCAL, local(fam, first, ell, log_r, loc, grp, gl))
            loc = dataclasses.replace(loc, **{leaf[first]: x})
            oks += [ok1, _finite_on(x, on)]
            if second is not None:
                x, ok2 = gig_local(Sub.GL_LOCAL_AUX, local(fam, second, ell, log_r, loc, grp, gl))
                loc = dataclasses.replace(loc, **{leaf[second]: x})
                oks += [ok2, _finite_on(x, on)]
            if info.a_grid:
                a_idx, log_xi = self.impl("gl_a_conditional")(
                    fam, _group_keys(key, Sub.GL_A, G), _group_keys(key, Sub.GL_GROUP, G),
                    loc, glob, info, gl,
                )  # fmt: skip
                glob = dataclasses.replace(glob, a_idx=jnp.asarray(a_idx, jnp.int32))
                if fam in _HAS_XI:
                    glob = dataclasses.replace(glob, log_xi=log_xi)
            elif fam in _HAS_XI:
                p = group(fam, "xi", ell, log_r, loc, glob, info, gl)
                glob = dataclasses.replace(
                    glob, log_xi=_gamma(_group_keys(key, Sub.GL_GROUP, G), p)
                )
            if fam in _HAS_XI:
                oks.append(jnp.all(jnp.isfinite(glob.log_xi)))
        elif fam == "ssvs":
            ssvs = arr.ssvs
            logit = self.impl("ssvs_logodds")(
                ell, ssvs.log_tau0, ssvs.log_tau1, _bcast(glob.log_p, info),
                _bcast(glob.log1m_p, info),
            )  # fmt: skip
            keys_u = _coord_keys(key, Sub.GL_LOCAL, N)
            u = jax.vmap(R.uniform_tiny)(keys_u).reshape(ell.shape)
            gamma = jnp.where(on, R.bernoulli_logit_core(u, logit), 0).astype(jnp.int32)
            prob = jnp.exp(-jnp.logaddexp(0.0, -logit))  # σ(logit), exact for ±∞
            loc = SSVSEq(gamma=gamma, rb_incl=mask(loc.rb_incl + prob))
            oks.append(jnp.all(jnp.where(on, ~jnp.isnan(logit), True)))
            if self.static.p_beta:
                pb = group(fam, "p", ell, log_r, loc, glob, info, gl)
                lp, l1mp = jax.vmap(R.log_beta)(_group_keys(key, Sub.GL_GROUP, G), pb.a, pb.b)
                glob = SSVSGlobal(log_p=lp, log1m_p=l1mp)
                oks.append(jnp.all(jnp.isfinite(lp) & jnp.isfinite(l1mp)))
        else:  # pragma: no cover - CoefPriorStatic rejects other kinds
            raise ValueError(f"unknown global-local family {fam!r}")
        ok = jnp.all(jnp.stack([jnp.asarray(o, bool) for o in oks]))
        return loc, glob, ok

    def log_prior(self, glob: GlobalState, prior: PriorArrays) -> Array:
        """``log p`` of the group-level state (natural coordinates; counting measure on
        ``a_idx``): HS ``ζ_g | τ_g ~ IG(½, 1/τ_g)``, ``τ_g ~ IG(½, 1/s²)``; NG ``ξ_g | a_g ~
        G(b, 2c/a_g)``; R2D2 ``ξ_g ~ G(b, 1)``; the grid's log weights of ``a_idx``; SSVS
        ``p_g ~ Beta(s₁, s₂)`` under a Beta hyperprior."""
        g, gl = glob.coef, prior.shared.gl
        info = self.info(prior)
        fam = self.family
        out = jnp.zeros((), jnp.float64)
        if fam == "horseshoe":
            half = jnp.full_like(g.log_zeta, 0.5)
            out = out + jnp.sum(_log_ig(g.log_zeta, half, -g.log_tau))
            out = out + jnp.sum(_log_ig(g.log_tau, half, -2.0 * jnp.log(gl.global_scale)))
        if fam in _HAS_XI:
            a = a_values(g, gl, info)
            log_rate = _ng_log_beta(a, gl) if fam == "normal_gamma" else jnp.zeros_like(a)
            out = out + jnp.sum(_log_gamma(g.log_xi, jnp.broadcast_to(gl.b, a.shape), log_rate))
        if fam in _HAS_A and info.a_grid:
            logw = gl.a_logw - jax.scipy.special.logsumexp(gl.a_logw)
            out = out + jnp.sum(jnp.take(logw, g.a_idx))
        if fam == "ssvs" and self.static.p_beta:
            lbeta = gammaln(gl.s1) + gammaln(gl.s2) - gammaln(gl.s1 + gl.s2)
            out = out + jnp.sum((gl.s1 - 1.0) * g.log_p + (gl.s2 - 1.0) * g.log1m_p - lbeta)
        return out

    def local_log_prior(self, loc: PyTree, glob: PyTree, prior: PriorArrays) -> Array:
        """``log p`` of the local state given the group state (natural coordinates; counting
        measure on ``γ``), ``n × k`` and 0 off the GL slots: HS ``ϑ_j | ν_j ~ IG(½, 1/ν_j)``,
        ``ν_j ~ IG(½, 1)``; NG ``ϖ_j | ξ_g, a_g ~ G(a_g, ξ_g)``; R2D2 that and ``ϱ_j ~ Exp(½)``;
        DL ``θ̃_j ~ G(a_g, ½)``, ``ϱ_j ~ Exp(½)``; SSVS ``γ_j ~ Bernoulli(p_g)``."""
        gl = prior.shared.gl
        info = self.info(prior)
        on = info.mask > 0
        fam = self.family
        if fam == "horseshoe":
            half = jnp.full_like(loc.log_theta, 0.5)
            lp = _log_ig(loc.log_theta, half, -loc.log_nu) + _log_ig(
                loc.log_nu, half, jnp.zeros_like(half)
            )
        elif fam == "ssvs":
            lp = jnp.where(loc.gamma == 1, _bcast(glob.log_p, info), _bcast(glob.log1m_p, info))
        else:
            a = _bcast(a_values(glob, gl, info), info)
            if fam == "dirichlet_laplace":
                lp = _log_gamma(loc.log_theta, a, jnp.full_like(a, -LOG2))
            else:
                lp = _log_gamma(loc.log_w, a, _bcast(glob.log_xi, info))
            if fam in ("r2d2", "dirichlet_laplace"):  # ϱ ~ Exp(½) = G(1, ½)
                one = jnp.ones_like(a)
                lp = lp + _log_gamma(loc.log_varrho, one, jnp.full_like(a, -LOG2))
        return jnp.where(on, lp, 0.0)

    def log_prior_state(
        self, st: ChainState, prior: PriorArrays, eq_map: EqMap | None = None
    ) -> Array:
        """``log p`` of every scale of the prior from the full chain state (P3 §3.7): the local
        densities (:meth:`local_log_prior`) plus the group level (:meth:`log_prior`)."""
        loc = jnp.sum(self.local_log_prior(st.eq.coef, st.glob.coef, prior))
        return loc + self.log_prior(st.glob, prior)

    def future(self, key: Key, gd: GlobDraw, H: int) -> Empty:
        """``Empty()``: the scales do not enter forecasts."""
        return Empty()


# --------------------------------------------------------------------------- the CoefPrior kernels


@dataclass(frozen=True)
class _GlobalLocal(KernelBase):
    """Shared implementation of the five GL ``CoefPrior`` kernels (module docstring)."""

    static: CoefPriorStatic = field(default_factory=CoefPriorStatic)
    form: Literal["structural", "reduced"] = "structural"

    varies: ClassVar[bool] = True
    has_local_state: ClassVar[bool] = False  # the local scales are drawn by GLScaleStep (D-40)
    family: ClassVar[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.static.kind != self.family:
            raise ValueError(
                f"{type(self).__name__} needs CoefPriorStatic(kind={self.family!r}), "
                f"got {self.static.kind!r}"
            )
        if self.form not in ("structural", "reduced"):
            raise ValueError(f"form must be 'structural' or 'reduced', got {self.form!r}")

    @classmethod
    def from_static(cls, static: StaticConfig) -> _GlobalLocal:
        """The prior of a composition: ``static.coef`` and the form of ``static.mean``."""
        form = "reduced" if static.mean.kind == "linear_reduced" else "structural"
        return cls(static.coef, form=form)

    # ----------------------------------------------------------------- state
    def _zeros(self, k_shape: tuple[int, ...]) -> PyTree:
        z = jnp.zeros(k_shape, jnp.float64)
        return {
            "horseshoe": lambda: HSEq(z, z),
            "normal_gamma": lambda: NGEq(z),
            "r2d2": lambda: R2D2Eq(z, z),
            "dirichlet_laplace": lambda: DLEq(z, z),
            "ssvs": lambda: SSVSEq(jnp.zeros(k_shape, jnp.int32), z),
        }[self.family]()

    def init_eq(self, key: Key, i: Array, prior_i: EqPrior) -> PyTree:
        """Zeros (the local scales are set by :meth:`init_global`, which runs next)."""
        return self._zeros(jnp.shape(prior_i.coef.mean0))

    def scale_step(self) -> GLScaleStep:
        return GLScaleStep(self.static, form=self.form, overrides=self.overrides)

    def global_steps(self) -> tuple[GLScaleStep]:
        """The scale step (``Block.G_COEF_GLOBAL``, post phase)."""
        return (self.scale_step(),)

    # ----------------------------------------------------------------- moments
    def log_kernel_var(self, cs: PyTree, arr: Any, glob_coef: PyTree) -> Array:
        """``log(r_j v_j)`` (SSVS ``2 log τ_{γ_j, j}``) at every coordinate of ``arr``'s shape;
        0 off the GL slots."""
        on = arr.mask > 0
        fam = self.family
        if fam == "ssvs":
            v = 2.0 * jnp.where(cs.gamma == 1, arr.ssvs.log_tau1, arr.ssvs.log_tau0)
        elif fam == "horseshoe":
            v = arr.log_r + cs.log_theta + jnp.take(glob_coef.log_zeta, arr.group)
        elif fam == "normal_gamma":
            v = arr.log_r + cs.log_w
        elif fam == "r2d2":
            v = arr.log_r + cs.log_varrho + cs.log_w - LOG2
        else:
            v = arr.log_r + cs.log_varrho + 2.0 * cs.log_theta
        return jnp.where(on, v, 0.0)

    def moments_eq(
        self, cs_i: PyTree, i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> GaussianLogDiag:
        """``GaussianLogDiag(mean0, log_prec, 1)``: ``log_prec = −log(r v)`` on the GL slots,
        ``log prec0`` on the deterministic slots (P3 §3.3)."""
        c = prior_i.coef
        mean0 = jnp.asarray(c.mean0, jnp.float64)
        on = c.gl.mask > 0
        log_prec = jnp.where(
            on, -self.log_kernel_var(cs_i, c.gl, glob_coef), jnp.log(jnp.where(on, 1.0, c.prec0))
        )
        return GaussianLogDiag(mean=mean0, log_prec=log_prec, cmask=jnp.ones_like(mean0))

    def sample_local_eq(
        self, key: Key, cs_i: PyTree, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> PyTree:
        """Identity: :class:`GLScaleStep` draws the local scales (D-40)."""
        return cs_i

    def log_prior_eq(
        self, cs_i: PyTree, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> Array:
        """0. The scales' density couples the equations through their groups and needs the shared
        hyperprior grid, so it is evaluated from the full chain state by
        :meth:`GLScaleStep.log_prior_state` (P3 §3.7); the coefficients' Gaussian density is
        counted by the mean (structural) or the CCCM step (reduced form)."""
        return jnp.zeros((), jnp.float64)

    # ----------------------------------------------------------------- initialisation
    def init_global(
        self, key: Key, st: ChainState, data: EngineData, prior: PriorArrays
    ) -> ChainState:
        """The cold start of the scales (P3 §3.7; module docstring). ``key`` is the init stream's
        ``… → Block.G_COEF_GLOBAL`` key."""
        arr, gl = prior.eq.coef.gl, prior.shared.gl
        on = jnp.asarray(arr.mask) > 0
        group = jnp.asarray(arr.group, jnp.int32)
        G = self.static.n_groups

        def jitter(sub: Sub) -> Array:
            keys = _group_keys(key, sub, G)
            return INIT_JITTER_SD * jax.vmap(lambda kk: random.normal(kk, (), jnp.float64))(keys)

        def at(v: Array) -> Array:
            return jnp.where(on, v, 0.0)

        z = jnp.zeros(on.shape, jnp.float64)
        fam = self.family
        if self.static.a_grid:
            a_idx = jnp.full((G,), jnp.argmin(jnp.abs(gl.a_support - INIT_A)), jnp.int32)
            a = jnp.take(gl.a_support, a_idx)
        else:
            a_idx = jnp.zeros((G,), jnp.int32)
            a = jnp.broadcast_to(jnp.exp(gl.log_a_fixed), (G,))
        if fam == "horseshoe":
            log_zeta = INIT_LOG_VAR + jitter(Sub.GL_GROUP)
            log_tau = -2.0 * jnp.log(gl.global_scale) + jitter(Sub.GL_GROUP_AUX)
            loc, glob = HSEq(z, z), HSGlobal(log_zeta=log_zeta, log_tau=log_tau)
        elif fam in _HAS_XI:
            log_xi = jnp.log(a) - INIT_LOG_VAR + jitter(Sub.GL_GROUP)
            log_w = at(jnp.take(jnp.log(a) - log_xi, group))  # ϖ_j = a_g/ξ_g: unit multiplier
            if fam == "normal_gamma":
                loc, glob = NGEq(log_w=log_w), NGGlobal(log_xi=log_xi, a_idx=a_idx)
            else:
                loc = R2D2Eq(log_varrho=at(jnp.full(on.shape, LOG2)), log_w=log_w)
                glob = R2D2Global(log_xi=log_xi, a_idx=a_idx)
        elif fam == "dirichlet_laplace":
            two = at(jnp.full(on.shape, LOG2))
            loc, glob = DLEq(log_theta=two, log_varrho=two), DLGlobal(a_idx=a_idx)
        else:
            p = jnp.asarray(0.5) if self.static.p_beta else jnp.asarray(gl.p_fixed)
            loc = SSVSEq(gamma=jnp.where(on, 1, 0).astype(jnp.int32), rb_incl=z)
            glob = SSVSGlobal(
                log_p=jnp.broadcast_to(jnp.log(p), (G,)),
                log1m_p=jnp.broadcast_to(jnp.log1p(-p), (G,)),
            )
        return dataclasses.replace(
            st,
            eq=dataclasses.replace(st.eq, coef=loc),
            glob=dataclasses.replace(st.glob, coef=glob),
        )

    # ----------------------------------------------------------------- storage
    def extract_draw(self, eq_coef: PyTree, glob_coef: PyTree) -> PyTree:
        """``ParamDraw.glob.coef``: the SSVS indicators (int8), else ``Empty()``."""
        return Empty()


@dataclass(frozen=True)
class Horseshoe(_GlobalLocal):
    """Horseshoe (Carvalho, Polson & Scott 2010; Makalic & Schmidt 2016 auxiliaries):
    ``v_j = ϑ_j ζ_g``, ``√ϑ_j ~ C⁺(0, 1)``, ``√ζ_g ~ C⁺(0, s)``."""

    family: ClassVar[str] = "horseshoe"


@dataclass(frozen=True)
class NormalGamma(_GlobalLocal):
    """Normal-Gamma (Griffin & Brown 2010; GK's parametrisation): ``v_j = ϖ_j | ξ_g ~ G(a_g, ξ_g)``,
    ``ξ_g ~ G(b, 2c/a_g)`` (``c`` tied: rate 1)."""

    family: ClassVar[str] = "normal_gamma"


@dataclass(frozen=True)
class R2D2(_GlobalLocal):
    """R2D2 (Zhang et al. 2022; GK's gamma form): ``v_j = ϱ_j ϖ_j/2``, ``ϱ_j ~ Exp(½)``,
    ``ϖ_j | ξ_g ~ G(a_g, ξ_g)``, ``ξ_g ~ G(b, 1)``."""

    family: ClassVar[str] = "r2d2"


@dataclass(frozen=True)
class DirichletLaplace(_GlobalLocal):
    """Dirichlet–Laplace (Bhattacharya et al. 2015; gamma form): ``v_j = ϱ_j θ̃_j²``,
    ``ϱ_j ~ Exp(½)``, ``θ̃_j ~ G(a_g, ½)``."""

    family: ClassVar[str] = "dirichlet_laplace"


@dataclass(frozen=True)
class SSVS(_GlobalLocal):
    """Stochastic search variable selection (George & McCulloch 1993; George, Sun & Ni 2008):
    ``d_j | γ_j ~ N(0, τ²_{γ_j, j})``, ``γ_j | p_g ~ Bernoulli(p_g)``, ``p_g`` fixed or Beta."""

    family: ClassVar[str] = "ssvs"

    def extract_draw(self, eq_coef: PyTree, glob_coef: PyTree) -> SSVSDraw:
        """The indicators ``γ`` as int8 (``n × k``; P3 E123 as amended: stored per stored draw)."""
        return SSVSDraw(incl=jnp.asarray(eq_coef.gamma).astype(jnp.int8))
