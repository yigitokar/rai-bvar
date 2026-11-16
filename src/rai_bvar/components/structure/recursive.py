"""Recursive covariance structure: unit lower-triangular ``A`` (DESIGN §3.2, §3.7.1; D-28).

The structure owns ``a_i``, its prior and the residual algebra of every Phase 2 composition
(N05 §3; N04 §2, §6.2)::

    A y_t = Γ x_t + Λ_t^{½} ε_t,   A = I + strict-lower(a),   |A| = 1 (no Jacobian)
    e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i = target − (cols ⊙ cmask)·a_i − g_i

with ``ỹ = Y`` and ``g_i = Xγ_i`` (structural Linear, D3) or ``ỹ = Y − XΠ`` and ``g_i = 0`` (reduced
form, ``e = UAᵀ``). ``cols = −ỹ_{·, :n−1}`` is shared by every equation; row ``i`` uses the slots
``j < i`` (``cmask``), so the inactive slots are trailing (DESIGN §2.3).

Prior (DESIGN §6.1, §4.7): ``a_ij ~ N(m, 1/P̃)`` on active slots; with ``contemp_scaling="lambda"``
(Mode II, ``Constant("iw_equivalent")``) ``a_i | λ_i ~ N(m, λ_i P̃⁻¹)``, so :meth:`contemp_moments`
divides ``P̃`` by ``λ_i = exp(log_scale)``. Inactive slots carry mean 0 and precision 1 (D-15).

Contemporaneous conditional for a non-fusing mean (N04 §6.2 with its minus sign; N05 §5.4):
``ỹ_i − g_i = −ỹ_{<i} a_i + e_i`` is a weighted regression on ``Z = cols ⊙ cmask``::

    K_a = diag(prec) + Zᵀ diag(w) Z,   b_a = prec ⊙ mean + Zᵀ(w ⊙ (target − g_i))

Maps back to the reduced form use unit-triangular solves only (N05 §8; no inverses):
``B = (A⁻¹Γ)ᵀ``, ``chol(Σ_t) = A⁻¹ Λ_t^{½}`` (lower, positive diagonal), ``E[y] = A⁻¹g + f``.

CommonSV (P3 §5.3, E87): when the global volatility state is a ``CommonSVGlobal``,
:meth:`Recursive.context` also sets ``EqContext.common_w = m_t e^{−h_t}`` from that state and
``tmask`` alone. Both are closed over by the vmapped equation body (``SweepCtx``/``_Sweep``), so
``common_w`` is **unbatched** and everything the shared-factor path builds from it (its Gram,
scaling and Cholesky factor) is computed once per chain and sweep, outside the equation batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import lax, random

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig, StructStatic
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.linalg import lstsq_draw, precision_draw, solve_unit_lower
from rai_bvar.engines.mcmc.protocols import KernelBase, VolBlock
from rai_bvar.engines.mcmc.state import (
    CommonSVGlobal,
    Coupling,
    Empty,
    EngineData,
    EqContext,
    EqPrior,
    EqState,
    GaussianDiag,
    GaussianPrec,
    GlobalState,
    RecursiveEq,
    StepInfo,
    StructDraw,
)

Array = jax.Array
PyTree = Any

_HIGHEST = lax.Precision.HIGHEST  # full float64 products on every backend (CPU ignores it)

__all__ = ["Recursive", "contemp_prior", "slot_mask", "unit_lower"]


# --------------------------------------------------------------------------- helpers


def unit_lower(a: Array) -> Array:
    """``A = I + strict-lower(a)`` (``n × n``) from the state layout ``a[i, j] = a_ij`` (``n × n−1``);
    entries with ``j ≥ i`` are not read."""
    n = a.shape[-2]
    full = jnp.concatenate([a, jnp.zeros((*a.shape[:-1], 1), a.dtype)], axis=-1)
    return jnp.eye(n, dtype=a.dtype) + jnp.tril(full, -1)


def slot_mask(i: Array, n: int) -> Array:
    """``1{j < i}`` for the ``n − 1`` contemporaneous slots of equation ``i`` (float64)."""
    return (jnp.arange(n - 1, dtype=jnp.int32) < jnp.asarray(i, jnp.int32)).astype(jnp.float64)


def contemp_prior(prior_i: EqPrior, cmask: Array, log_scale: Array | float = 0.0) -> GaussianDiag:
    """``a_i`` prior on the active slots ``cmask ⊙ prior.cmask``: mean ``m``, precision
    ``P̃·exp(−log_scale)``; inactive slots mean 0, precision 1 (D-15). ``log_scale = 0`` gives the
    λ-free prior (Mode I, and the per-fit fast-path caches of ``LinearStructural``)."""
    c = prior_i.contemp
    active = cmask * c.cmask
    on = active > 0
    return GaussianDiag(
        mean=jnp.where(on, c.mean0, 0.0),
        prec=jnp.where(on, c.prec0 * jnp.exp(-jnp.asarray(log_scale, jnp.float64)), 1.0),
        cmask=active,
    )


# --------------------------------------------------------------------------- kernel


@dataclass(frozen=True)
class Recursive(KernelBase):
    """``CovarianceStructure`` with unit lower-triangular ``A`` (DESIGN §3.2, §3.7.1).

    ``static.contemp_scaling == "lambda"`` (Mode II, ``iw_equivalent``) λ-scales the ``a_i`` prior
    and makes :meth:`contemp_step` return the coupling of DESIGN §4.7. The overridable internal is
    ``contemp_conditional`` (DESIGN §4.9; mutant M14 flips its sign).
    """

    static: StructStatic = field(default_factory=StructStatic)

    has_contemp: ClassVar[bool] = True
    supports_reduced_mean: ClassVar[bool] = True

    @classmethod
    def from_static(cls, static: StaticConfig) -> Recursive:
        """The structure of a composition (``static.struct``)."""
        return cls(static.struct)

    @property
    def scaled(self) -> bool:
        """``a_i | λ_i ~ N(m, λ_i P̃⁻¹)``: Mode II and ``iw_equivalent`` (DESIGN §4.7)."""
        return self.static.contemp_scaling == "lambda"

    # ----------------------------------------------------------------- state and residuals

    def init_eq(self, key: Key, i: Array, prior_i: EqPrior) -> RecursiveEq:
        """``a_i = 0`` (DESIGN §5.4; the structural mean's first draw then replaces it)."""
        return RecursiveEq(a=jnp.zeros_like(prior_i.contemp.mean0, dtype=jnp.float64))

    def pre_residual(self, glob: GlobalState, data: EngineData, f_reduced: Array | None) -> Array:
        """``ỹ = Y`` (structural and zero means) or ``Y − XΠ`` (reduced form), zero on padded rows."""
        Y = jnp.asarray(data.Y, jnp.float64)
        ytil = Y if f_reduced is None else Y - f_reduced
        return jnp.where(jnp.asarray(data.tmask)[:, None] > 0, ytil, 0.0)

    def context(
        self, i: Array, ytil: Array, eq_i: EqState, vol: VolBlock, data: EngineData, gvol: PyTree
    ) -> EqContext:
        """Equation ``i``'s view: ``target = ỹ_i``, ``cols = −ỹ_{·,:n−1}``, ``cmask = 1{j < i}``,
        ``weights = m_t·exp(−log λ_eff,i,t)`` (masked before the ``exp``, never ``1/exp``) and
        ``log_scale = log λ_i`` when the ``a`` prior is λ-scaled (any λ-scaling implies it,
        ``config.validate_composition`` rule 2), else 0. Under CommonSV (``gvol`` a
        ``CommonSVGlobal``) also ``common_w = m_t e^{−h_t}`` from ``gvol`` and ``tmask`` only, so
        it is unbatched inside the equation body (P3 §5.3, E87)."""
        n = ytil.shape[1]
        i = jnp.asarray(i, jnp.int32)
        tmask = jnp.asarray(data.tmask, jnp.float64)
        live = tmask > 0
        lv = vol.log_var_eq(eq_i.vol, gvol)
        weights = jnp.where(live, jnp.exp(-jnp.where(live, lv, 0.0)), 0.0)
        log_scale = vol.log_scale_eq(eq_i.vol) if self.scaled else 0.0
        common_w: Array | Empty = Empty()
        if isinstance(gvol, CommonSVGlobal):  # closed-over glob.vol: unbatched (E87)
            h = jnp.asarray(gvol.h, jnp.float64)
            common_w = jnp.where(live, jnp.exp(-jnp.where(live, h, 0.0)), 0.0)
        return EqContext(
            i=i,
            target=jnp.take(ytil, i, axis=1),
            cols=-ytil[:, : n - 1],
            cmask=slot_mask(i, n),
            weights=weights,
            log_scale=jnp.asarray(log_scale, jnp.float64),
            tmask=tmask,
            common_w=common_w,
        )

    def residual(self, a_i: Array, ctx: EqContext, g_i: Array) -> Array:
        """``e_i = target − (cols ⊙ cmask)·a_i − g_i``, exactly 0 on padded rows."""
        contemp = jnp.matmul(ctx.cols * ctx.cmask[None, :], a_i, precision=_HIGHEST)
        return jnp.where(ctx.tmask > 0, ctx.target - contemp - g_i, 0.0)

    def log_jacobian(self, glob: GlobalState, T_eff: Array) -> Array:
        """``T_eff·log|det A| = 0`` for unit-triangular ``A``."""
        return jnp.zeros((), jnp.float64)

    # ----------------------------------------------------------------- contemporaneous step

    def contemp_moments(self, ctx: EqContext, prior_i: EqPrior) -> GaussianDiag:
        """The ``a_i`` prior (DESIGN §3.7.1): precision ``P̃·exp(−log_scale)`` on active slots when
        λ-scaled, ``P̃`` otherwise, 1 on inactive slots."""
        return contemp_prior(prior_i, ctx.cmask, ctx.log_scale if self.scaled else 0.0)

    def contemp_conditional(self, ctx: EqContext, g_i: Array, cm: GaussianDiag) -> GaussianPrec:
        """``a_i | g_i, λ`` (N04 §6.2 with the minus sign; N05 §5.4): with ``Z = cols ⊙ cmask``,
        ``K_a = diag(prec) + Zᵀdiag(w)Z``, ``b_a = prec ⊙ mean + Zᵀ(w ⊙ (target − g_i))``."""
        Z = ctx.cols * cm.cmask[None, :]
        Zw = Z * ctx.weights[:, None]
        K = jnp.matmul(Zw.T, Z, precision=_HIGHEST) + jnp.diag(cm.prec)
        K = jnp.tril(K) + jnp.tril(K, -1).T  # exactly symmetric
        b = cm.prec * cm.mean + jnp.matmul(Zw.T, ctx.target - g_i, precision=_HIGHEST)
        return GaussianPrec(K=K, b=b)

    def contemp_step(
        self, key: Key, s_i: RecursiveEq, ctx: EqContext, g_i: Array, cm: GaussianDiag
    ) -> tuple[RecursiveEq, Coupling, StepInfo]:
        """``a_i ~ N(K_a⁻¹b_a, K_a⁻¹)`` (precision draw D-14, ``z`` from ``Sub.Z``), masked to
        exact zeros on inactive slots; non-fusing means only (DESIGN §4.5). When λ-scaled it
        returns ``Coupling(Σ cmask, ½ Σ cmask·P̃·(a − m)²)`` with the λ-free ``P̃ =
        prec·exp(log_scale)`` (DESIGN §4.7), else zeros."""
        m = cm.mean.shape[-1]
        if m == 0:  # n = 1: no contemporaneous slots
            return s_i, Coupling.zeros(), StepInfo.success()
        z = random.normal(sub_key(key, Sub.Z), (m,), dtype=jnp.float64)
        return self.contemp_core(z, ctx, g_i, cm)

    def contemp_core(
        self, z: Array, ctx: EqContext, g_i: Array, cm: GaussianDiag
    ) -> tuple[RecursiveEq, Coupling, StepInfo]:
        """:meth:`contemp_step` for injected noise ``z ~ N(0, I_{n−1})`` (D-17).

        The draw is the whitened least-squares form of :meth:`contemp_conditional`
        (``linalg.lstsq_draw`` on ``A = [diag(√prec); √w ⊙ Z]``, ``rhs = [√prec ⊙ mean;
        √w ⊙ (target − g_i)]``, so ``AᵀA = K_a`` and ``Aᵀrhs = b_a``): the same function of ``z``
        as ``precision_draw(K_a, b_a, z)`` up to rounding, but ``Zᵀdiag(w)Z`` is never formed, so
        a regressor ~1e8 times the others (an SV state ``h ≈ 35`` next to ``h ≈ 0``) no longer
        makes ``K_a`` singular in float64 (the ``test_g_full[reduced_sv]`` finding). An overridden
        ``contemp_conditional`` (mutant M14, tests) is drawn from its ``(K, b)`` as before."""
        if self._overridden("contemp_conditional"):
            cond = self.impl("contemp_conditional")(ctx, g_i, cm)
            a, info = precision_draw(cond.K, cond.b, z)
        else:
            a, info = self.contemp_lstsq_draw(z, ctx, g_i, cm)
        a = jnp.where(cm.cmask > 0, a, 0.0)
        return RecursiveEq(a=a), self.coupling(a, ctx, cm), info

    def contemp_lstsq_draw(
        self, z: Array, ctx: EqContext, g_i: Array, cm: GaussianDiag
    ) -> tuple[Array, StepInfo]:
        """``a_i`` (unmasked) from the whitened rows of the conditional (:meth:`contemp_core`)."""
        Z = ctx.cols * cm.cmask[None, :]
        sw = jnp.sqrt(ctx.weights)  # weights m_t·exp(−log λ) ≥ 0: padded rows are zero rows
        sp = jnp.sqrt(cm.prec)
        A = jnp.concatenate([jnp.diag(sp), sw[:, None] * Z], axis=0)
        rhs = jnp.concatenate([sp * cm.mean, sw * (ctx.target - g_i)])
        return lstsq_draw(A, rhs, z)

    def _overridden(self, name: str) -> bool:
        return any(key == name for key, _ in self.overrides)

    def coupling(self, a_i: Array, ctx: EqContext, cm: GaussianDiag) -> Coupling:
        """Coupling of the ``a_i`` block (DESIGN §4.7): zeros unless λ-scaled."""
        if not self.scaled:
            return Coupling.zeros()
        prec0 = cm.prec * jnp.exp(ctx.log_scale)  # λ-free P̃ on active slots
        quad = 0.5 * jnp.sum(jnp.where(cm.cmask > 0, prec0 * (a_i - cm.mean) ** 2, 0.0))
        return Coupling(n_active=jnp.sum(cm.cmask), quad=quad)

    # ----------------------------------------------------------------- draws and maps

    def extract(self, eq_struct: RecursiveEq, glob: GlobalState) -> StructDraw:
        """``StructDraw(a)`` (``a`` f64[n, n−1], exact zeros on inactive slots)."""
        return StructDraw(a=eq_struct.a)

    def mean_map(self, sd: StructDraw, g: Array, f: Array) -> Array:
        """``A⁻¹g + f`` (one unit-triangular solve)."""
        return solve_unit_lower(unit_lower(sd.a), g) + f

    def chol(self, sd: StructDraw, log_lam: Array) -> Array:
        """``A⁻¹ diag(exp(½ log λ))``: lower triangular (exact zeros above the diagonal) with
        diagonal ``exp(½ log λ) > 0``, so ``chol·cholᵀ = A⁻¹ΛA⁻ᵀ = Σ`` (N05 §8)."""
        return solve_unit_lower(unit_lower(sd.a), jnp.diag(jnp.exp(0.5 * log_lam)))

    def reduced_form(self, sd: StructDraw, gamma: Array) -> Array:
        """``B = (A⁻¹Γ)ᵀ`` (``k × n``) for structural coefficients ``Γ`` (``n × k``)."""
        return solve_unit_lower(unit_lower(sd.a), gamma).T

    def realign(self, s_host: PyTree, drop_front: int, append: int) -> PyTree:
        """Identity: ``a`` has no time index (warm starts, DESIGN §5.5)."""
        return s_host
