"""Structural ``Linear`` mean: the fused ``(δ_i, a_i)`` draw (DESIGN §3.7.2; D-10, D-14, D-25, D-26;
N05 §4.1, §5.1, §7.1–7.4).

Equation ``i`` of ``A y_t = Γ x_t + Λ_t^{½} ε_t`` in δ-coordinates ``θ_i = [δ_i | a_i]``
(``q = k + n − 1`` slots, ``a``-slot ``k + j`` active iff ``j < i``; DESIGN §2.3)::

    y_i = F θ_i + e_i,   F = [X | X M_c − Y_{:, :n−1}],   γ_i = δ_i + M_c a_i   (D3: E[γ_i | a_i] = M ã_i)
    θ_i ~ N(meanθ, diag(1/precθ)),   meanθ = [pm.mean | cm.mean],   precθ = [pm.prec | cm.prec]

``pm`` is the λ-free coefficient prior of ``CoefPrior.moments_eq`` (divided here by
``λ_i = exp(log_scale)`` in Mode II, ``coef_scaling="lambda"``); ``cm`` comes from the structure
with its λ-scaling applied (Mode II, ``iw_equivalent``). With weights ``w_t = m_t/λ_eff,i,t`` and
``cθ`` the active-slot mask, the exact conditional is (N05 §5.1)::

    K = (cθ ⊗ cθ) ⊙ Fᵀdiag(w)F + diag(precθ),   b = precθ ⊙ meanθ + cθ ⊙ Fᵀ(w ⊙ y_i),   θ ~ N(K⁻¹b, K⁻¹)

Inactive slots have mean 0, precision 1 and zero data columns, so ``K`` is block diagonal and the
active draw equals the unpadded draw with ``z[:k+i]`` (D-15); the draw is masked afterwards.

Four static paths (``MeanStatic.const_path``, resolved in ``resolve.py``; all draw the same law):

* ``weighted`` (any time-varying weights): ``Fᵀdiag(w)F = weighted_gram(w, FF | F)`` (packed
  Khatri–Rao by default, D-25), Jacobi-scaled Cholesky and two triangular solves (D-14);
* ``cholesky_const`` (Constant core, any scaling; the oracle of the two fast paths):
  ``Fᵀdiag(w)F = G_const/λ_i``, ``Fᵀ(w ⊙ y_i) = FtY[:, i]/λ_i``;
* ``eigen`` (Mode I × Constant): per fit ``G̃ = d^{−½}(cθ⊗cθ ⊙ G_const)d^{−½} = Q diag(g) Qᵀ``
  with ``d = precθ``; per sweep ``θ = d^{−½}Q(v ⊙ (u1 + u2/λ) + √v ⊙ z)``, ``v = 1/(1 + g/λ)``
  (``O(q²)``, N05 §7.4);
* ``fixed_factor`` (Mode II × Constant): ``K = K̃/λ_i`` with ``K̃`` λ-free, so per fit
  ``Lc = chol(sK̃s)``, ``θ̂ = K̃⁻¹r̃``; per sweep ``θ = θ̂ + √λ_i·s ⊙ Lc⁻ᵀz`` (N05 §5.1).

On the Constant paths the weights are ``m_t/λ_i``, so ``1/λ_i = w_0`` (row 0 is always real).

Coupling (D-30, DESIGN §4.7): Mode II returns ``Coupling(k + i, ½ Σ_active P̃θ(θ − meanθ)²)`` with
the λ-free ``P̃θ``; ``iw_equivalent`` the same sums over the active ``a``-slots; Mode I zeros.

SOC/SUR dummy rows (WP25; ``MeanStatic.n_dummy = m > 0``; N05 §7.7): ``m`` prior rows with the
δ-design ``F_d = [X_d | X_d M_c − Y_d[:, :n−1]]`` (``FitCache.dummy``) and targets ``y_{d,i} =
Y_d[:, i]`` join every equation's regression, ``K += w_d (cθ⊗cθ ⊙ F_dᵀF_d)``,
``b += w_d cθ ⊙ F_dᵀy_{d,i}``, on every path (``fixed_factor``: in ``K̃``, λ-free; ``eigen``: in the
dense prior precision ``P = diag(precθ) + w_d(cθ⊗cθ ⊙ F_dᵀF_d)`` of ``linalg.eigen_prepare_dense``,
N05 §7.7); the weight is the fixed ``w_d = dummy_w_i = 1/σ̄_i²`` in Mode I and ``iw_equivalent``
(the rows are part of the Gaussian prior), and ``w_d = dummy_w_i/λ_i`` in Mode II, where the rows
are like data rows and the coupling counts them: ``n_active += m``,
``quad += ½ dummy_w_i Σ_d (y_{d,i} − F_dθ_i)²``. Without rows (``m = 0``) none of this is traced.

Deviation path (P3 §3.4, D-41; ``MeanStatic.coef_log``): a varying coefficient prior returns
``GaussianLogDiag`` moments, and the draw is taken in deviation coordinates ``θ_i = c_θ + d`` about
the prior mean ``c_θ = [pm.mean | cm.mean]`` with the Jacobi scaling in log space
(``linalg.precision_draw_log``)::

    K = diag(P) + G,   log P = [pm.log_prec | log cm.prec],   G = (cθ ⊗ cθ) ⊙ Fᵀ diag(w) F
    b̃ = cθ ⊙ Fᵀ(w ⊙ (y_i − F c_θ)),   d ~ N(K⁻¹b̃, K⁻¹),   dev_log = log|d|

This is the same law as the weighted path (it is D-14's draw with ``b̃ = b − K c_θ``), and it
returns ``LinearEq.dev_log = log|δ_i − m_i|`` exactly even when ``d`` underflows. The cold start
(:meth:`init_eq`) takes it too, so ``dev_log`` is finite from the first state on. GL priors carry
no SOC/SUR rows (D-36). The internal ``gl_deviation(mean) -> centre`` returns the centre of the
deviation (the prior mean; mutant M26 centres at 0). With ``GaussianDiag`` moments nothing of this
is traced: the Phase 2 paths are unchanged.

Shared-factor path (P3 §5.3, D-47; ``const_path="shared_factor"``: the conjugate Minnesota prior
× ``CommonSV``, structural Mode II at ``cross = contemp = 1``). The λ-free prior precision ``P̃``
(``SharedPrior.shared_prec``) is the same for every equation on its active (leading) slots, the
weights ``W = diag(m_t e^{−h_t})`` (``EqContext.common_w``) do not depend on ``i`` and the SOC/SUR
rows have the λ-free Mode II weight, so ``K̃_i = P̃ + F_iᵀWF_i + F_{d,i}ᵀF_{d,i}`` is the leading
``q_i × q_i`` block of one matrix and its Jacobi-scaled Cholesky factor the leading block of one
factor (``linalg.shared_factor_*``)::

    G = (√w ⊙ F)ᵀ(√w ⊙ F) + F_dᵀF_d,   s = diag(P̃ + G)^{−½},   L = chol(s(P̃ + G)s)   once per chain
    b_i = c_i ⊙ (P̃μ_i + Fᵀ(w ⊙ y_i) + F_dᵀy_{d,i})                                     per equation
    θ_i = c_i ⊙ s ⊙ L⁻ᵀ(c_i ⊙ L⁻¹(s ⊙ b_i) + √λ_i·(c_i ⊙ z_i))    ~ N(K̃_i⁻¹b_i, λ_i K̃_i⁻¹)

``common_w``, ``F``, the rows and ``P̃`` are closed over by the vmapped equation body (unbatched,
E87), so ``G``, ``s`` and ``L`` are computed once per chain and sweep. This is the weighted
path's law (``K = K̃_i/λ_i``, ``b = b_i/λ_i``) and the same function of ``z`` up to rounding. The
SOC/SUR rows are prior rows: weight 1, **not** weighted by ``h`` (E83, E88; internal
``shared_factor_rows``, mutant M32). The coupling is the Mode II one with the rows (E88:
``n_active = q_i + m``, the rows' sum of squares in ``quad``); for the coupled common core it is
formed by the internal ``csv_lambda_conditional(θ part, row part) -> Coupling`` (mutants M31, M36;
the ``Coupling`` contract carries totals only, so the parts are separated here). The exposed
conditional (``linear_conditional``, ``conditional_eq``) takes the weighted branch (E95), and an
overridden internal sends the draw there as well.

Overridable internals (DESIGN §4.9): ``structural_gamma(δ, a, M_c) -> γ`` and
``linear_conditional(ctx, pm, cm, cache, shared) -> GaussianPrec``. Overriding either switches the
draw to the Cholesky path through ``linear_conditional``; overriding ``structural_gamma`` (which must
be affine with ``∂γ/∂δ = I``) also rebuilds the design in the kernel as
``[X | X·∂γ/∂a − Y_{:, :n−1}]``, so draws, fitted values, residuals and stored ``γ`` all follow the
override (mutant M8, "D3 centring dropped").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from rai_bvar.components.structure.recursive import contemp_prior, slot_mask
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import MeanStatic, Scaling, StaticConfig
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.linalg import (
    eigen_draw,
    eigen_prepare,
    eigen_prepare_dense,
    fixed_factor_draw,
    fixed_factor_prepare,
    precision_draw,
    precision_draw_log,
    shared_factor_draw,
    shared_factor_gram,
    shared_factor_prepare,
    shared_factor_rhs,
    weighted_gram,
)
from rai_bvar.engines.mcmc.protocols import Batching, KernelBase
from rai_bvar.engines.mcmc.state import (
    Coupling,
    Empty,
    EngineData,
    EqContext,
    EqPrior,
    FitCache,
    GaussianDiag,
    GaussianLogDiag,
    GaussianPrec,
    GlobalState,
    GlobDraw,
    LinearDraw,
    LinearEq,
    PriorArrays,
    SharedPrior,
    StepInfo,
)

Array = jax.Array
PyTree = Any

_HIGHEST = lax.Precision.HIGHEST  # full float64 products on every backend (CPU ignores it)
_LOG_2PI = 1.8378770664093453
_FAST_PATHS = frozenset({"eigen", "fixed_factor"})

__all__ = ["DeviationConditional", "LinearStructural", "theta_prior_arrays"]


def theta_prior_arrays(prior_i: EqPrior, i: Array, n: int) -> GaussianDiag:
    """λ-free prior of ``θ_i = [δ_i | a_i]`` from the arrays (the ``FixedDiagonal`` moments and the
    ``Recursive`` ``a`` prior without scaling): the per-fit caches of the ``eigen`` (Mode I) and
    ``fixed_factor`` (Mode II, λ-free ``P̃θ``) paths."""
    c = prior_i.coef
    cm = contemp_prior(prior_i, slot_mask(i, n))
    mean0 = jnp.asarray(c.mean0, jnp.float64)
    cmask = jnp.concatenate([jnp.ones_like(mean0), cm.cmask])
    on = cmask > 0
    return GaussianDiag(
        mean=jnp.where(on, jnp.concatenate([mean0, cm.mean]), 0.0),
        prec=jnp.where(on, jnp.concatenate([jnp.asarray(c.prec0, jnp.float64), cm.prec]), 1.0),
        cmask=cmask,
    )


class DeviationConditional(NamedTuple):
    """The exact conditional of ``θ_i`` on the deviation path (P3 §3.4): ``θ_i = centre + d``,
    ``d ~ N(K⁻¹b, K⁻¹)``, ``K = diag(exp(log_prec)) + G`` (``G`` masked, ``b = b̃`` masked)."""

    centre: Array
    log_prec: Array
    G: Array
    b: Array
    cmask: Array


@dataclass(frozen=True)
class LinearStructural(KernelBase):
    """``MeanBlock`` of the structural Linear mean with D3 centring (DESIGN §3.7.2).

    Static fields: ``static`` (``MeanStatic``: path, Gram, ``coef_scaling``), ``contemp_scaling``
    (``StructStatic.contemp_scaling``: whether the ``a`` prior is λ-scaled, which decides the
    coupling of ``iw_equivalent``) and ``n`` (equations). Build it with :meth:`from_static`.
    """

    static: MeanStatic
    contemp_scaling: Scaling
    n: int
    common: bool = False  # CommonSV volatility (VolStatic.common; P3 §5): the coupled common core

    fuses_contemp: ClassVar[bool] = True
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.static.kind != "linear_structural":
            raise ValueError(
                f"LinearStructural needs MeanStatic(kind='linear_structural'), got {self.static.kind!r}"
            )
        if self.contemp_scaling not in ("none", "lambda"):
            raise ValueError(
                f"contemp_scaling must be 'none' or 'lambda', got {self.contemp_scaling!r}"
            )
        if self.static.coef_scaling == "lambda" and self.contemp_scaling != "lambda":
            raise ValueError(
                "Mode II (coef_scaling='lambda') scales a_i too: contemp_scaling must be 'lambda' "
                "(build the kernel with LinearStructural.from_static)"
            )
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")

    @classmethod
    def from_static(cls, static: StaticConfig) -> LinearStructural:
        """The mean of a composition: ``static.mean``, ``static.struct.contemp_scaling``,
        ``static.shapes.n`` and whether the volatility is ``CommonSV`` (``static.vol.common``)."""
        return cls(
            static.mean,
            contemp_scaling=static.struct.contemp_scaling,
            n=static.shapes.n,
            common=bool(static.vol.common),
        )

    # ----------------------------------------------------------------- static helpers

    @property
    def coef_scaled(self) -> bool:
        """Mode II: the δ prior is λ-scaled."""
        return self.static.coef_scaling == "lambda"

    @property
    def contemp_scaled(self) -> bool:
        """Mode II or ``iw_equivalent``: the ``a`` prior is λ-scaled."""
        return self.contemp_scaling == "lambda"

    def _overridden(self, name: str) -> bool:
        return any(key == name for key, _ in self.overrides)

    @property
    def draw_path(self) -> str:
        """The path :meth:`sample_eq` takes: ``const_path``, except that an overridden internal
        sends the fast paths through ``linear_conditional`` (``cholesky_const``)."""
        path = self.static.const_path
        custom = self._overridden("structural_gamma") or self._overridden("linear_conditional")
        if path in _FAST_PATHS and custom:
            return "cholesky_const"
        if path == "shared_factor" and custom:  # P3 E95: the general (weighted) conditional
            return "weighted"
        return path

    # ----------------------------------------------------------------- prior and conditional

    def theta_prior(self, ctx: EqContext, pm: GaussianDiag, cm: GaussianDiag) -> GaussianDiag:
        """Prior of ``θ_i`` with the λ-scaling applied: ``pm.prec·exp(−log_scale)`` in Mode II,
        ``cm`` as given; inactive slots mean 0, precision 1."""
        pprec = pm.prec * jnp.exp(-ctx.log_scale) if self.coef_scaled else pm.prec
        cmask = jnp.concatenate([pm.cmask, cm.cmask])
        on = cmask > 0
        return GaussianDiag(
            mean=jnp.where(on, jnp.concatenate([pm.mean, cm.mean]), 0.0),
            prec=jnp.where(on, jnp.concatenate([pprec, cm.prec]), 1.0),
            cmask=cmask,
        )

    def structural_gamma(self, delta: Array, a: Array, Mc: Array) -> Array:
        """``γ_i = δ_i + M_c a_i`` (D3; N05 §4.1)."""
        return delta + jnp.matmul(Mc, a, precision=_HIGHEST)

    def _gamma_jacobian(self, shared: SharedPrior) -> Array:
        """``∂γ/∂a`` (k × n−1) of the (overridden) ``structural_gamma``; ``M_c`` for the D3 map."""
        k, m = shared.Mc.shape
        sg = self.impl("structural_gamma")
        return jax.jacfwd(lambda a: sg(jnp.zeros((k,), jnp.float64), a, shared.Mc))(
            jnp.zeros((m,), jnp.float64)
        )

    def _design(
        self, ctx: EqContext, cache: FitCache, shared: SharedPrior, *, batched: bool = False
    ) -> tuple[Array, Array | None, str, int]:
        """``(F, FF, gram mode, tile)``: the cached δ-design, or the in-kernel design
        ``[X | X·∂γ/∂a + cols]`` when ``structural_gamma`` is overridden (``cols = −ỹ_{<}``)."""
        if self._overridden("structural_gamma"):
            k = shared.Mc.shape[0]
            X = cache.F[:, :k]
            J = self._gamma_jacobian(shared)
            F = jnp.concatenate([X, jnp.matmul(X, J, precision=_HIGHEST) + ctx.cols], axis=1)
            return F, None, "batched", 0
        if batched or self.static.gram != "packed":
            mode = "batched" if batched else self.static.gram
            return cache.F, None, mode, self.static.gram_tile
        return cache.F, cache.FF, "packed", 0

    def _conditional(
        self,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache: FitCache,
        shared: SharedPrior,
        *,
        batched: bool,
    ) -> GaussianPrec:
        pr = self.theta_prior(ctx, pm, cm)
        c = pr.cmask
        w = ctx.weights
        shared_factor = self.static.const_path == "shared_factor"  # P3 E95: the weighted branch
        const = self.static.const_path != "weighted" and not batched and not shared_factor
        if const and not self._overridden("structural_gamma"):
            inv_lam = w[0]  # weights m_t/λ_i: row 0 is always real (T_eff ≥ 1)
            G = cache.G_const * inv_lam
            fty = jnp.take(cache.FtY, ctx.i, axis=1) * inv_lam
        else:  # the shared-factor cache has no packed products FF: the Gram is formed from F
            F, FF, mode, tile = self._design(ctx, cache, shared, batched=batched or shared_factor)
            G = weighted_gram(w, F=F, FF=FF, mode=mode, q=F.shape[1], tile=tile)
            fty = jnp.matmul(F.T, w * ctx.target, precision=_HIGHEST)
        if self.static.n_dummy:  # SOC/SUR prior rows (WP25); closed path without dummies
            Gd, fty_d = self.dummy_terms(ctx, cache, shared)
            G = G + Gd
            fty = fty + fty_d
        # exactly symmetric: the packed Gram is v[pos] and G_const is mirrored (WP4, WP5)
        K = c[:, None] * G * c[None, :] + jnp.diag(pr.prec)
        return GaussianPrec(K=K, b=pr.prec * pr.mean + c * fty)

    # ----------------------------------------------------------------- SOC/SUR dummy rows (WP25)

    def _dummy_rows(
        self, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> tuple[Array, Array, Array]:
        """``(F_d, y_d, w_d)`` of equation ``ctx.i``: the rows' δ-design (``m × q``; rebuilt as
        ``[X_d | X_d·∂γ/∂a − Y_d[:, :n−1]]`` when ``structural_gamma`` is overridden), their
        targets ``Y_d[:, i]`` and the weight in use: the λ-free ``dummy_w_i``, divided by
        ``λ_i = exp(log_scale)`` in Mode II (N05 §7.7)."""
        d = cache.dummy
        if isinstance(d, Empty) or isinstance(shared.dummies, Empty):
            raise ValueError(
                f"MeanStatic.n_dummy = {self.static.n_dummy} needs FitCache.dummy and "
                "SharedPrior.dummies (prepare with the prior's dummy rows)"
            )
        Yd = shared.dummies.Y
        if self._overridden("structural_gamma"):
            J = self._gamma_jacobian(shared)
            Xd = shared.dummies.X
            Fd = jnp.concatenate(
                [Xd, jnp.matmul(Xd, J, precision=_HIGHEST) - Yd[:, : self.n - 1]], axis=1
            )
        else:
            Fd = d.F
        w = jnp.take(d.w, ctx.i)
        if self.coef_scaled:
            w = w * jnp.exp(-ctx.log_scale)
        return Fd, jnp.take(Yd, ctx.i, axis=1), w

    def dummy_terms(
        self, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> tuple[Array, Array]:
        """The dummy rows' contribution ``(w_d F_dᵀF_d, w_d F_dᵀy_d)`` to the Gram and to
        ``Fᵀ(w ⊙ y_i)`` of equation ``ctx.i`` (unmasked, like the data part). Fixed weight
        ``1/σ̄_i²`` in Mode I and ``iw_equivalent``, ``1/λ_i`` in Mode II (N05 §7.7)."""
        if self._overridden("structural_gamma"):
            Fd, yd, w = self._dummy_rows(ctx, cache, shared)
            G = jnp.matmul(Fd.T, Fd, precision=_HIGHEST)
            return w * G, w * jnp.matmul(Fd.T, yd, precision=_HIGHEST)
        _, _, w = self._dummy_rows(ctx, cache, shared)
        return w * cache.dummy.G, w * jnp.take(cache.dummy.FtY, ctx.i, axis=1)

    def dummy_log_density_eq(
        self,
        ms: LinearEq,
        a_i: Array,
        ctx: EqContext,
        cache: FitCache,
        shared: SharedPrior,
    ) -> Array:
        """``Σ_d log N(y_{d,i}; F_d θ_i, 1/w_d)``: the dummy rows' factor of the prior of
        ``θ_i = [δ_i | a_i]`` (with ``λ_i`` in Mode II), up to its normalising constant; 0 without
        dummies. The monitor adds it to ``log_prior_eq`` (DESIGN §8.1)."""
        if not self.static.n_dummy:
            return jnp.zeros((), jnp.float64)
        Fd, yd, w = self._dummy_rows(ctx, cache, shared)
        r = yd - jnp.matmul(Fd, jnp.concatenate([ms.delta, a_i]), precision=_HIGHEST)
        return 0.5 * jnp.sum(jnp.log(w) - _LOG_2PI - w * r * r)

    def linear_conditional(
        self,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache: FitCache,
        shared: SharedPrior,
    ) -> GaussianPrec:
        """The exact conditional ``θ_i | λ_eff,i, y ~ N(K⁻¹b, K⁻¹)`` (module docstring; N05 §5.1)
        with the path's Gram: ``weighted_gram(w, FF | F)`` on the weighted path, ``G_const·w_0`` and
        ``FtY[:, i]·w_0`` on the Constant paths. ``pm`` is λ-free (Mode II scales it here)."""
        return self._conditional(ctx, pm, cm, cache, shared, batched=False)

    # ----------------------------------------------------------------- caches and init

    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache:
        """Per-equation caches of the fast paths (DESIGN §2.5): ``EigenCache`` (``eigen``,
        ``d = precθ`` λ-free) or ``FixedFactorCache`` (``fixed_factor``, ``K̃ = cθ⊗cθ ⊙ G_const +
        diag(P̃θ)``, ``r̃ = P̃θ ⊙ meanθ + cθ ⊙ FtY[:, i]``); otherwise ``base`` unchanged."""
        path = self.static.const_path
        if path not in _FAST_PATHS:
            return base
        n = self.n
        G = base.G_const

        def one(i: Array, prior_i: EqPrior) -> PyTree:
            pr = theta_prior_arrays(prior_i, i, n)
            c = pr.cmask
            fty = jnp.take(base.FtY, i, axis=1)
            if path == "eigen" and self.static.n_dummy:  # dense prior precision (WP25)
                d = base.dummy
                w = jnp.take(d.w, i)
                P = jnp.diag(pr.prec) + w * (c[:, None] * d.G * c[None, :])
                b0 = pr.prec * pr.mean + w * c * jnp.take(d.FtY, i, axis=1)
                return eigen_prepare_dense(G, P, b0, fty, c)
            if path == "eigen":
                return eigen_prepare(G, pr.prec, pr.mean, fty, c)
            Gi = G
            if self.static.n_dummy:  # Mode II: λ-free rows (weight w_i/λ_i like the data; WP25)
                d = base.dummy
                w = jnp.take(d.w, i)
                Gi = G + w * d.G
                fty = fty + w * jnp.take(d.FtY, i, axis=1)
            K_tilde = c[:, None] * Gi * c[None, :] + jnp.diag(pr.prec)
            return fixed_factor_prepare(K_tilde, pr.prec * pr.mean + c * fty)

        eq = jax.vmap(one)(jnp.arange(n, dtype=jnp.int32), prior.eq)
        return dataclasses.replace(base, eq=eq)

    def fitted_reduced(self, glob: GlobalState, data: EngineData) -> None:
        return None

    def init_eq(
        self,
        key: Key,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[LinearEq, Array]:
        """One exact draw of ``(δ_i, a_i)`` with the weights of ``ctx`` (the runner passes
        ``m_t·ι/ψ_ι``, DESIGN §5.4), on any path: Gram ``Fᵀ(w ⊙ F)`` from ``cache.F``."""
        if isinstance(pm, GaussianLogDiag):  # deviation path (P3 §3.4): sets dev_log
            dc = self.deviation_conditional(ctx, pm, cm, cache, shared, batched=True)
            ms, a, _, _ = self._draw_log(self._noise(key, pm, cm), dc, pm, cm)
            return ms, a
        cond = self._conditional(ctx, pm, cm, cache, shared, batched=True)
        theta, _ = precision_draw(cond.K, cond.b, self._noise(key, pm, cm))
        return self._split(theta, pm, cm)

    # ----------------------------------------------------------------- the equation step

    def _noise(self, key: Key, pm: GaussianDiag, cm: GaussianDiag) -> Array:
        q = pm.mean.shape[-1] + cm.mean.shape[-1]
        return random.normal(sub_key(key, Sub.Z), (q,), dtype=jnp.float64)

    def _split(self, theta: Array, pm: GaussianDiag, cm: GaussianDiag) -> tuple[LinearEq, Array]:
        k = pm.mean.shape[-1]
        theta = jnp.where(jnp.concatenate([pm.cmask, cm.cmask]) > 0, theta, 0.0)
        return LinearEq(delta=theta[:k]), theta[k:]

    def draw_core(
        self,
        z: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[Array, StepInfo]:
        """Unmasked ``θ_i`` for injected noise ``z ~ N(0, I_q)`` on :attr:`draw_path` (D-17)."""
        path = self.draw_path
        if path == "eigen":
            return eigen_draw(cache_i, 1.0 / ctx.weights[0], z)
        if path == "fixed_factor":
            theta = fixed_factor_draw(cache_i, jnp.exp(ctx.log_scale), z)
            return theta, StepInfo(jnp.all(jnp.isfinite(theta)), jnp.full((), jnp.inf))
        if path == "shared_factor":
            return self.shared_factor_eq(z, ctx, pm, cm, cache, shared)
        cond = self.impl("linear_conditional")(ctx, pm, cm, cache, shared)
        return precision_draw(cond.K, cond.b, z)

    # ----------------------------------------------------------------- shared factor (P3 §5.3)

    def shared_factor_rows(self, w: Array, Fd: Array, Yd: Array) -> tuple[Array, Array]:
        """The SOC/SUR rows ``(F_d, Y_d)`` that join the shared factor: unchanged, i.e. weight 1
        and not weighted by ``h`` (P3 E83, E88; overridable, mutant M32 weights them by
        ``e^{−h_0}`` = ``w[0]``). ``w`` is ``EqContext.common_w``."""
        del w
        return Fd, Yd

    def shared_factor_parts(
        self, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> tuple[Array, Array, Array | None, Array | None]:
        """``(L, s, F_d, Y_d)``: the factor shared by every equation (unbatched inside the
        equation body: it reads only ``common_w``, ``F``, the rows and ``P̃``) and the rows
        (``None`` without SOC/SUR rows), scaled by the square root of their λ-free weight
        (``dummy_w``, equal across equations and 1 in Mode II; ``resolve.py`` checks it)."""
        w = ctx.common_w
        if isinstance(w, Empty) or isinstance(shared.shared_prec, Empty):
            raise ValueError(
                "const_path='shared_factor' needs EqContext.common_w (CommonSV) and "
                "SharedPrior.shared_prec (the equation-invariant λ-free precision)"
            )
        Fd = Yd = None
        if self.static.n_dummy:
            d = cache.dummy
            if isinstance(d, Empty) or isinstance(shared.dummies, Empty):
                raise ValueError("MeanStatic.n_dummy > 0 needs FitCache.dummy and the rows")
            sw = jnp.sqrt(d.w[0])
            Fd, Yd = self.impl("shared_factor_rows")(w, sw * d.F, sw * shared.dummies.Y)
        G = shared_factor_gram(w, cache.F, Fd)
        L, s, _ = shared_factor_prepare(shared.shared_prec, G)
        return L, s, Fd, Yd

    def shared_factor_eq(
        self,
        z: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[Array, StepInfo]:
        """Unmasked ``θ_i`` of the shared-factor path for injected ``z`` (module docstring):
        ``θ_i ~ N(K̃_i⁻¹b_i, λ_i K̃_i⁻¹)`` with ``λ_i = exp(log_scale)`` (Mode II, coupled)."""
        L, s, Fd, Yd = self.shared_factor_parts(ctx, cache, shared)
        cmask = jnp.concatenate([pm.cmask, cm.cmask])
        on = cmask > 0
        mean = jnp.where(on, jnp.concatenate([pm.mean, cm.mean]), 0.0)
        yd = None if Yd is None else jnp.take(Yd, ctx.i, axis=1)
        prec = shared.shared_prec
        b = shared_factor_rhs(prec, mean, cache.F, ctx.common_w, ctx.target, cmask, Fd, yd)
        return shared_factor_draw(L, s, b, cmask, jnp.exp(ctx.log_scale), z)

    def csv_lambda_conditional(self, theta: Coupling, dummy: Coupling) -> Coupling:
        """The coefficient-side terms of the coupled common core's ``λ_i | θ_i, e_i, h``
        (P3 §5.3): ``(n_active, quad)`` of the θ block plus those of the SOC/SUR rows
        (``n_active = q_i + m``; E88). Overridable (mutants M31: no ``n_active/2`` shape term,
        M36: no rows); production code returns the sum."""
        return theta + dummy

    def sample_eq(
        self,
        key: Key,
        ms: LinearEq,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[LinearEq, Array, Coupling, StepInfo]:
        """Joint draw of ``(δ_i, a_i) | λ_eff,i`` (DESIGN §4.2 S1; ``z`` from ``Sub.Z``); returns
        the new state, the new ``a_i``, the coupling and the step info. ``ok`` requires a finite
        draw and a positive factorisation (eigen: ``g ≥ −1e-10·max g``)."""
        if cm is None:
            raise ValueError("LinearStructural fuses the contemporaneous step: cm is required")
        if isinstance(pm, GaussianLogDiag):  # deviation path (P3 §3.4)
            return self._sample_eq_log(key, ctx, pm, cm, cache, shared)
        theta, info = self.draw_core(self._noise(key, pm, cm), ctx, pm, cm, cache_i, cache, shared)
        new, a = self._split(theta, pm, cm)
        return new, a, self.coupling(new.delta, a, ctx, pm, cm, cache=cache, shared=shared), info

    def conditional_eq(
        self,
        ms: LinearEq,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> GaussianPrec:
        """The exposed exact conditional of ``θ_i = [δ_i | a_i]`` (DESIGN §3.1 rule 3)."""
        if cm is None:
            raise ValueError("LinearStructural fuses the contemporaneous step: cm is required")
        if isinstance(pm, GaussianLogDiag):  # θ coordinates of the deviation path's conditional
            dc = self.deviation_conditional(ctx, pm, cm, cache, shared, batched=False)
            K = jnp.diag(jnp.exp(dc.log_prec)) + dc.G
            return GaussianPrec(K=K, b=dc.b + jnp.matmul(K, dc.centre, precision=_HIGHEST))
        return self.impl("linear_conditional")(ctx, pm, cm, cache, shared)

    def coupling(
        self,
        delta: Array,
        a: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag,
        *,
        cache: FitCache | None = None,
        shared: SharedPrior | None = None,
    ) -> Coupling:
        """DESIGN §4.7: λ-free quadratic forms over the λ-scaled active coordinates.

        Mode II with SOC/SUR dummy rows (WP25; N05 §7.7): the rows are weighted ``w_i/λ_i`` like
        data rows, so the ``λ_i`` conditional counts them too: ``n_active += m`` and ``quad +=
        ½ w_i Σ_d (y_{d,i} − F_d θ_i)²`` (``cache`` and ``shared`` are then required)."""
        if not (self.coef_scaled or self.contemp_scaled):
            return Coupling.zeros()
        prec_a = cm.prec * jnp.exp(ctx.log_scale)  # λ-free P̃ of the a-slots
        quad = jnp.sum(jnp.where(cm.cmask > 0, prec_a * (a - cm.mean) ** 2, 0.0))
        n_active = jnp.sum(cm.cmask)
        if self.coef_scaled and self.common:  # the coupled common core (P3 §5.3): parts apart
            quad = quad + jnp.sum(jnp.where(pm.cmask > 0, pm.prec * (delta - pm.mean) ** 2, 0.0))
            theta = Coupling(n_active=n_active + jnp.sum(pm.cmask), quad=0.5 * quad)
            rows = Coupling.zeros()
            if self.static.n_dummy:
                if cache is None or shared is None:
                    raise ValueError("Mode II coupling with dummy rows needs cache and shared")
                Fd, yd, _ = self._dummy_rows(ctx, cache, shared)
                r = yd - jnp.matmul(Fd, jnp.concatenate([delta, a]), precision=_HIGHEST)
                rows = Coupling(
                    n_active=jnp.asarray(float(self.static.n_dummy), jnp.float64),
                    quad=0.5 * jnp.take(cache.dummy.w, ctx.i) * jnp.sum(r * r),
                )
            return self.impl("csv_lambda_conditional")(theta, rows)
        if self.coef_scaled:
            quad = quad + jnp.sum(jnp.where(pm.cmask > 0, pm.prec * (delta - pm.mean) ** 2, 0.0))
            n_active = n_active + jnp.sum(pm.cmask)
            if self.static.n_dummy:
                if cache is None or shared is None:
                    raise ValueError("Mode II coupling with dummy rows needs cache and shared")
                Fd, yd, _ = self._dummy_rows(ctx, cache, shared)
                r = yd - jnp.matmul(Fd, jnp.concatenate([delta, a]), precision=_HIGHEST)
                quad = quad + jnp.take(cache.dummy.w, ctx.i) * jnp.sum(r * r)
                n_active = n_active + self.static.n_dummy
        return Coupling(n_active=n_active, quad=0.5 * quad)

    # ----------------------------------------------------------------- deviation path (P3 §3.4)

    def gl_deviation(self, mean: Array) -> Array:
        """The centre of the deviation coordinates: the prior mean (overridable; mutant M26)."""
        return mean

    def theta_prior_log(self, pm: GaussianLogDiag, cm: GaussianDiag) -> GaussianLogDiag:
        """Prior of ``θ_i`` in log precisions: ``[pm.log_prec | log cm.prec]`` on the active slots
        (``cm`` carries its λ-scaling), mean 0 and log precision 0 on the inactive ones."""
        cmask = jnp.concatenate([pm.cmask, cm.cmask])
        on = cmask > 0
        log_pa = jnp.log(jnp.where(cm.cmask > 0, cm.prec, 1.0))
        return GaussianLogDiag(
            mean=jnp.where(on, jnp.concatenate([pm.mean, cm.mean]), 0.0),
            log_prec=jnp.where(on, jnp.concatenate([pm.log_prec, log_pa]), 0.0),
            cmask=cmask,
        )

    def deviation_conditional(
        self,
        ctx: EqContext,
        pm: GaussianLogDiag,
        cm: GaussianDiag,
        cache: FitCache,
        shared: SharedPrior,
        *,
        batched: bool,
    ) -> DeviationConditional:
        """``(centre, log P, G, b̃)`` of the deviation draw (module docstring): the weighted Gram
        ``(cθ ⊗ cθ) ⊙ Fᵀdiag(w)F`` (``batched``: from ``F``, the cold start) and
        ``b̃ = cθ ⊙ Fᵀ(w ⊙ (y_i − F centre))``."""
        if self.static.n_dummy:
            raise ValueError("global-local priors carry no SOC/SUR dummy rows (P3 D-36)")
        pr = self.theta_prior_log(pm, cm)
        c = pr.cmask
        centre = self.impl("gl_deviation")(pr.mean)
        F, FF, mode, tile = self._design(ctx, cache, shared, batched=batched)
        w = ctx.weights
        G = weighted_gram(w, F=F, FF=FF, mode=mode, q=F.shape[1], tile=tile)
        resid = ctx.target - jnp.matmul(F, centre, precision=_HIGHEST)
        b = c * jnp.matmul(F.T, w * resid, precision=_HIGHEST)
        return DeviationConditional(
            centre=centre, log_prec=pr.log_prec, G=c[:, None] * G * c[None, :], b=b, cmask=c
        )

    def _draw_log(
        self, z: Array, dc: DeviationConditional, pm: GaussianLogDiag, cm: GaussianDiag
    ) -> tuple[LinearEq, Array, Array, StepInfo]:
        """``θ_i = centre + d`` for injected ``z``: ``(LinearEq(δ, dev_log), a, d, info)``."""
        d, dev_log, info = precision_draw_log(dc.log_prec, dc.G, dc.b, z)
        new, a = self._split(dc.centre + d, pm, cm)
        k = pm.mean.shape[-1]
        return LinearEq(delta=new.delta, dev_log=dev_log[:k]), a, d, info

    def _sample_eq_log(
        self,
        key: Key,
        ctx: EqContext,
        pm: GaussianLogDiag,
        cm: GaussianDiag,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[LinearEq, Array, Coupling, StepInfo]:
        dc = self.deviation_conditional(ctx, pm, cm, cache, shared, batched=False)
        new, a, _, info = self._draw_log(self._noise(key, pm, cm), dc, pm, cm)
        return new, a, self.coupling(new.delta, a, ctx, pm, cm, cache=cache, shared=shared), info

    # ----------------------------------------------------------------- views and densities

    def fitted_eq(
        self, ms: LinearEq, a_i: Array, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> Array:
        """``g_i = Xγ_i`` over rows (D3 term included; 0 on padded rows), ``X = F[:, :k]``."""
        k = ms.delta.shape[-1]
        gamma = self.impl("structural_gamma")(ms.delta, a_i, shared.Mc)
        return jnp.matmul(cache.F[:, :k], gamma, precision=_HIGHEST)

    def coefficients_eq(self, ms: LinearEq, a_i: Array, shared: SharedPrior) -> Array:
        """``δ_i``: what the coefficient prior governs."""
        return ms.delta

    def log_prior_eq(
        self,
        ms: LinearEq,
        a_i: Array,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        shared: SharedPrior,
        *,
        log_scale: Array | float = 0.0,
    ) -> Array:
        """``Σ_active log N(θ; meanθ, 1/precθ)``: ``pm`` λ-free (divided by ``exp(log_scale)`` in
        Mode II, as in :meth:`sample_eq`), ``cm`` as the structure returns it (scaled); ``cm=None``
        gives the δ part only."""
        if isinstance(pm, GaussianLogDiag):  # from dev_log (P3 §3.7): exact at any scale
            lp = 0.5 * (pm.log_prec - _LOG_2PI) - 0.5 * jnp.exp(2.0 * ms.dev_log + pm.log_prec)
            out = jnp.sum(jnp.where(pm.cmask > 0, lp, 0.0))
            if cm is not None:
                la = 0.5 * (jnp.log(cm.prec) - _LOG_2PI - cm.prec * (a_i - cm.mean) ** 2)
                out = out + jnp.sum(jnp.where(cm.cmask > 0, la, 0.0))
            return out
        prec_d = (
            pm.prec * jnp.exp(-jnp.asarray(log_scale, jnp.float64)) if self.coef_scaled else pm.prec
        )

        def part(x: Array, mean: Array, prec: Array, mask: Array) -> Array:
            lp = 0.5 * (jnp.log(prec) - _LOG_2PI - prec * (x - mean) ** 2)
            return jnp.sum(jnp.where(mask > 0, lp, 0.0))

        out = part(ms.delta, pm.mean, prec_d, pm.cmask)
        if cm is not None:
            out = out + part(a_i, cm.mean, cm.prec, cm.cmask)
        return out

    def extract_eq(self, ms: LinearEq, a_i: Array, shared: SharedPrior) -> LinearDraw:
        """``LinearDraw(gamma = δ_i + M_c a_i)``."""
        return LinearDraw(gamma=self.impl("structural_gamma")(ms.delta, a_i, shared.Mc))

    def extract_global(self, glob: GlobalState) -> GlobDraw:
        return GlobDraw(mean=Empty(), coef=Empty(), vol=Empty(), struct=Empty())

    def future_eq(self, key: Key, md: LinearDraw, H: int) -> Empty:
        return Empty()

    def predict_eq(self, md: LinearDraw, fut: PyTree, x: Array, h: Array, gfut: PyTree) -> Array:
        """``g_i(x) = γ_i'x``."""
        return jnp.dot(md.gamma, x, precision=_HIGHEST)

    def predict_reduced(self, gd: GlobDraw, x: Array, h: Array, gfut: PyTree) -> Array:
        """``f64[n]`` zeros (no reduced-form part)."""
        return jnp.zeros((self.n,), jnp.float64)
