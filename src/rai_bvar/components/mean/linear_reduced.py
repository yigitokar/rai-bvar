"""Reduced-form mean (CCCM 2022): the CCCM pre-phase global step and the per-equation no-op mean
(DESIGN §3.7.3, §4.3; D-11; N04 §2, §5.3, §6.1–6.4).

Model (N04 §2): ``y_t = Π'x_t + A⁻¹Λ_t^{½}ε_t`` with ``A = I + strict-lower(a)`` unit lower
triangular and the coefficient prior ``π_j ~ N(m_j, diag(1/p_j))`` independent over equations
``j`` (``Π = [π_0 … π_{n−1}]``, ``k × n``). Equation ``i`` of ``Ay_t`` contains ``π_l`` for every
``l ≤ i``, so ``π_j`` cannot be drawn from equation ``j`` alone: the CCM (2019) step that did so
is exact only for ``j = n−1`` (N04 §4).

**Π step** (:func:`cccm_core`, the corrected triangular algorithm): one Gibbs scan over the full
conditionals ``π_j | π_{−j}, A, Λ, y`` (CCCM 2022 eqs. 13–15) in the combined-weight form of N04
§5.3, with ``inv_lam_{i,t} = m_t exp(−log λ_eff,i,t)`` (zero on padded rows)::

    W   = inv_lamᵀ (A ∘ A)                   W_tj = Σ_i a_ij²/λ_it = (Σ_t⁻¹)_jj       (T_pad, n)
    K_j = Xᵀ diag(W_·j) X + diag(p_j)        independent of Π: all j factored first (precompute)
    U = Y − XΠ,  E = UAᵀ                      from scratch every sweep (no drift; N04 §7)
    for j = 0..n−1 (lax.fori_loop, dynamic j):
        G_j = (E ⊙ inv_lamᵀ) A_·j            (Σ_t⁻¹u_t)_j
        b_j = p_j ⊙ m_j + Xᵀ(G_j + W_·j ⊙ Xπ_j)
        π_j ← s ⊙ L_j⁻ᵀ(L_j⁻¹(s ⊙ b_j) + z_j) ~ N(K_j⁻¹b_j, K_j⁻¹)       (D-14, L_j = chol(sK_js))
        d = X(π_j^new − π_j^old);  E −= d A_·jᵀ                          (rank-1 in time)

This is the textbook SUR Gibbs conditional ``K_j = P_j + Σ_t ω_jj,t x_t x_t'``,
``b_j = P_j m_j + Σ_t x_t(ω_jj,t y_jt + Σ_{l≠j} ω_jl,t u_lt)`` with ``Ω_t = A'Λ_t⁻¹A`` (N04 §5.3),
so the scan is block Gauss–Seidel on the joint precision of ``vec Π`` (N04 §9.2; Fox & Parker
2017) and leaves ``N(Q⁻¹b, Q⁻¹)`` invariant. ``A`` is never inverted: the needed entries of
``Σ_t⁻¹`` come from ``A`` and ``Λ_t`` (N04 §7). The ``K_j`` Gram follows ``MeanStatic.gram``:
``packed`` reads the packed products ``XX`` of the fit cache (one GEMM ``(n, T_pad) × (T_pad, P_x)``
for all ``j``; D-25), ``packed_tiled`` and ``batched`` form them from ``X``.

**The rest of the reduced-form sweep** (DESIGN §4.3) lives elsewhere: ``a_j | Π, λ_j`` is
``Recursive.contemp_step`` on ``ỹ = U`` (the regression of ``u_j`` on ``−u_{<j}``; N04 §6.2), and
the volatility blocks receive ``e_j = (UAᵀ)_j``. :class:`ReducedEq` is the per-equation mean of
that sweep: no state, ``g_i ≡ 0``, ``fitted_reduced = XΠ``.

**Construction** (``compose``, WP14): ``CCCMStep(static.mean, vol=VolBlock, coef=CoefPrior)``,
or ``CCCMStep.from_static(static, vol=..., coef=...)``, with the program's kernels (DESIGN §3.6:
global steps get the kernels at construction). ``coef=None`` reads the moments straight from
``prior.eq.coef`` (the ``FixedDiagonal`` moments). ``ReducedEq(static.mean)`` or
``ReducedEq.from_static(static)``. Both accept ``overrides`` (``KernelBase``); the overridable
internal is ``cccm_core`` (mutant M1 and the ``contemp_only`` block configuration).

**Keys** (DESIGN §5.6): the step key ``fold_in(it_key, Block.G_CCCM)`` → equation ``j`` →
``Sub.Z``: ``z_j ~ N(0, I_k)`` (:func:`cccm_noise`). Failures are values (``StepInfo``; D-18).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp
from jax import lax, random

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import MeanStatic, StaticConfig
from rai_bvar.engines.mcmc.data import packed_products
from rai_bvar.engines.mcmc.keys import Block, Key, Sub, eq_key, sub_key
from rai_bvar.engines.mcmc.linalg import precision_draw_factored, precision_factor, weighted_gram
from rai_bvar.engines.mcmc.protocols import Batching, EqMap, KernelBase
from rai_bvar.engines.mcmc.state import (
    CCCMDraw,
    CCCMGlobal,
    ChainState,
    ConstantPrior,
    Coupling,
    Empty,
    EngineData,
    EqContext,
    FitCache,
    GaussianDiag,
    GaussianPrec,
    GlobalState,
    GlobDraw,
    PriorArrays,
    SharedPrior,
    StepInfo,
    SVPrior,
    SweepCtx,
)

Array = jax.Array

_LOG_2PI = math.log(2.0 * math.pi)

__all__ = [
    "CCCMStep",
    "ReducedEq",
    "cccm_conditional_j",
    "cccm_core",
    "cccm_noise",
    "cccm_precisions",
    "cccm_weights",
    "masked_inv_lam",
    "unit_lower",
]


# --------------------------------------------------------------------------- pure kernels


def unit_lower(a: Array) -> Array:
    """``A = I + strict-lower(a)`` from the state layout ``a`` (``n × (n−1)``, ``a[i, j] = a_ij``
    for ``j < i``); entries with ``j ≥ i`` are ignored. Shape ``(n, n)``."""
    a = jnp.asarray(a, dtype=jnp.float64)
    n = a.shape[0]
    full = jnp.concatenate([a, jnp.zeros((n, 1), a.dtype)], axis=1)
    return jnp.eye(n, dtype=a.dtype) + jnp.tril(full, k=-1)


def masked_inv_lam(log_lam: Array, tmask: Array) -> Array:
    """``inv_lam_{i,t} = m_t·exp(−log λ_{i,t})`` (``n × T_pad``), the log variance masked before the
    ``exp``: padded rows are exactly 0 whatever ``log_lam`` holds there (E23; never ``1/exp``)."""
    live = jnp.asarray(tmask)[None, :] > 0
    return jnp.where(live, jnp.exp(-jnp.where(live, log_lam, 0.0)), 0.0)


def cccm_weights(a: Array, inv_lam: Array) -> Array:
    """Combined weights ``W = inv_lamᵀ(A ∘ A)``, ``W_tj = Σ_i a_ij²/λ_it = (Σ_t⁻¹)_jj`` (N04 §5.3);
    shape ``(T_pad, n)``."""
    A = unit_lower(a)
    return inv_lam.T @ (A * A)


def _gram(w: Array, X: Array, XX: Array | None, gram: str, tile: int) -> Array:
    """``Xᵀ diag(w) X`` by the static Gram mode (``packed`` without ``XX`` forms the products)."""
    k = X.shape[1]
    if gram == "packed":
        FF = packed_products(X) if XX is None else XX
        return weighted_gram(w, FF=FF, mode="packed", q=k)
    if gram == "packed_tiled":
        return weighted_gram(w, F=X, mode="packed_tiled", q=k, tile=tile)
    return weighted_gram(w, F=X, mode=gram, q=k)  # type: ignore[arg-type]


def cccm_precisions(
    W: Array,
    X: Array,
    XX: Array | Empty | None,
    prec0: Array,
    *,
    gram: str,
    tile: int = 0,
) -> Array:
    """All ``K_j = Xᵀ diag(W_·j) X + diag(p_j)``, shape ``(n, k, k)`` (exactly symmetric).

    ``gram="packed"``: ``(Wᵀ XX)[:, pos] + diag(p_j)``, one GEMM for all ``j`` (D-25)."""
    xx = None if isinstance(XX, Empty) else XX
    G = jax.vmap(lambda w: _gram(w, X, xx, gram, tile))(W.T)
    return G + jax.vmap(jnp.diag)(prec0)


def cccm_core(
    Pi: Array,
    a: Array,
    inv_lam: Array,
    X: Array,
    Y: Array,
    XX: Array | Empty,
    mean0: Array,
    prec0: Array,
    Z: Array,
    *,
    gram: str,
    precompute: bool,
    tile: int = 0,
) -> tuple[Array, StepInfo]:
    """One CTA scan ``j = 0..n−1`` (module docstring; N04 §5.3, §6.1, §6.4) with injected noise.

    ``Pi`` (k, n) current; ``a`` (n, n−1) state layout; ``inv_lam`` (n, T_pad), zero on padded
    rows (:func:`masked_inv_lam`); ``X`` (T_pad, k), ``Y`` (T_pad, n) zero on padded rows; ``XX``
    the packed products of ``X`` (``gram="packed"``; ``Empty`` otherwise); ``mean0``, ``prec0``
    (n, k) the moments of ``π_j`` (row ``j``); ``Z`` (n, k) standard normals. ``precompute``: every
    ``(L_j, s_j)`` before the loop (``n·k²`` floats) or inside it (identical algebra, less memory;
    DESIGN §10.5). ``tile``: columns per tile for ``gram="packed_tiled"``.

    Returns ``(Π_new, info)``: ``info.ok`` iff every factorisation succeeded and ``Π_new`` is
    finite; ``info.min_pivot`` the smallest pivot of the Jacobi-scaled ``K_j`` (in ``(0, 1]``).
    """
    n = Pi.shape[1]
    A = unit_lower(a)
    W = inv_lam.T @ (A * A)
    xx = None if isinstance(XX, Empty) else XX
    if gram == "packed" and xx is None:  # no cached products: form them once per sweep
        xx = packed_products(X)
    E0 = (Y - X @ Pi) @ A.T
    WI = inv_lam.T  # (T_pad, n)

    def factor(j: Array) -> tuple[Array, Array, StepInfo]:
        K = _gram(W[:, j], X, xx, gram, tile) + jnp.diag(prec0[j])
        return precision_factor(K)

    if precompute:
        Ks = cccm_precisions(W, X, xx, prec0, gram=gram, tile=tile)
        Ls, ss, infos = jax.vmap(precision_factor)(Ks)
        info0 = StepInfo(jnp.all(infos.ok), jnp.min(infos.min_pivot))
    else:
        info0 = StepInfo.success()

    def body(j: Array, carry: tuple[Array, Array, StepInfo]) -> tuple[Array, Array, StepInfo]:
        Pi, E, info = carry
        if precompute:
            L, s = Ls[j], ss[j]
        else:
            L, s, info_j = factor(j)
            info = info.merge(info_j)
        A_j = A[:, j]
        pi_old = Pi[:, j]
        f = X @ pi_old
        G = (E * WI) @ A_j
        b = prec0[j] * mean0[j] + X.T @ (G + W[:, j] * f)
        pi_new = precision_draw_factored(L, s, b, Z[j])
        d = X @ (pi_new - pi_old)
        E = E - d[:, None] * A_j[None, :]
        return Pi.at[:, j].set(pi_new), E, info

    Pi_new, _, info = lax.fori_loop(0, n, body, (Pi, E0, info0))
    ok = info.ok & jnp.all(jnp.isfinite(Pi_new))
    return Pi_new, StepInfo(ok, info.min_pivot)


def cccm_conditional_j(
    Pi: Array,
    a: Array,
    inv_lam: Array,
    X: Array,
    Y: Array,
    mean0: Array,
    prec0: Array,
    j: int | Array,
) -> GaussianPrec:
    """The exact full conditional ``π_j | π_{−j}, A, Λ, y = N(K_j⁻¹b_j, K_j⁻¹)`` at the current
    ``Π`` (N04 §5.3; DESIGN §3.1 rule 3), from scratch: ``E = (Y − XΠ)Aᵀ``,
    ``K_j = Xᵀ diag(W_·j) X + diag(p_j)``, ``b_j = p_j ⊙ m_j + Xᵀ((E ⊙ inv_lamᵀ)A_·j + W_·j ⊙ Xπ_j)``.
    ``j`` may be traced. Arguments as in :func:`cccm_core`."""
    k = X.shape[1]
    A = unit_lower(a)
    W = inv_lam.T @ (A * A)
    E = (Y - X @ Pi) @ A.T
    w = W[:, j]
    K = weighted_gram(w, F=X, mode="batched", q=k) + jnp.diag(prec0[j])
    b = prec0[j] * mean0[j] + X.T @ ((E * inv_lam.T) @ A[:, j] + w * (X @ Pi[:, j]))
    return GaussianPrec(K=K, b=b)


def cccm_noise(key: Key, n: int, k: int) -> Array:
    """``Z`` (n, k): row ``j`` from ``sub_key(eq_key(key, j), Sub.Z)`` (DESIGN §5.6: CCCM folds
    ``j`` before ``Sub.Z``); ``key`` is the step key ``fold_in(it_key, Block.G_CCCM)``."""

    def one(j: Array) -> Array:
        return random.normal(sub_key(eq_key(key, j), Sub.Z), (k,), dtype=jnp.float64)

    return jax.vmap(one)(jnp.arange(n, dtype=jnp.uint32))


def _prior_log_lam(vol_core: ConstantPrior | SVPrior, rw: bool) -> Array:
    """Prior centre of ``log λ_j`` (n,) used by the cold start (DESIGN §5.4: ``λ_j = ψ_ι/ι``),
    the same centre as the structural cold start (``runner.initial_log_variance``): Constant
    ``IG(c, C)`` mean ``C/(c−1)`` (``C/c`` when ``c ≤ 1``, where the mean does not exist); SV
    ``b_μ`` (AR) or ``m_0`` (RW), both ``log(ψ_ι/ι)`` under the default priors (D-13)."""
    if isinstance(vol_core, ConstantPrior):
        c = jnp.asarray(vol_core.shape, jnp.float64)
        return jnp.log(vol_core.scale) - jnp.log(jnp.where(c > 1.0, c - 1.0, c))
    return jnp.asarray(vol_core.m0 if rw else vol_core.b_mu, jnp.float64)


def _vol_rw(vol: Any) -> bool:
    """``rw`` of the program's ``VolBlock``: its ``static`` is the ``VolStatic``
    (``VolBlockImpl``) or a ``StaticConfig`` (``.vol``); a bare core exposes ``rw``."""
    s = getattr(vol, "static", None)
    s = getattr(s, "vol", s)
    return bool(getattr(s, "rw", getattr(vol, "rw", False)))


# --------------------------------------------------------------------------- kernels


def _check_static(static: Any, owner: str) -> None:
    if not isinstance(static, MeanStatic):
        raise TypeError(f"{owner}.static must be a MeanStatic, got {type(static).__name__}")
    if static.kind != "linear_reduced":
        raise ValueError(f"{owner} needs MeanStatic(kind='linear_reduced'), got {static.kind!r}")


@dataclass(frozen=True)
class CCCMStep(KernelBase):
    """Pre-phase global step ``Π | A, λ_eff, y`` (``Block.G_CCCM``; DESIGN §3.6, §3.7.3, §4.3).

    Reads ``glob.mean.Pi``, ``eq.struct.a`` and ``eq.vol`` (through ``vol.log_var_eq``); writes
    ``glob.mean.Pi``. Marginal of the mixture indicators: it conditions on the exact Gaussian
    likelihood given ``λ_eff`` (DESIGN §4.6 item 2). ``vol``: the program's ``VolBlock``
    (required by :meth:`sample`); ``coef``: its ``CoefPrior`` (``moments_eq`` through ``eq_map``,
    so Phase 3 global-local priors plug in unchanged), or ``None`` for the prior arrays' moments.
    """

    block: ClassVar[Block] = Block.G_CCCM
    phase: ClassVar[Literal["pre", "post"]] = "pre"
    reads: ClassVar[frozenset[str]] = frozenset({"glob.mean.Pi", "eq.struct.a", "eq.vol"})
    writes: ClassVar[frozenset[str]] = frozenset({"glob.mean.Pi"})
    marginal_of_indicators: ClassVar[bool] = True

    static: MeanStatic
    vol: Any = None
    coef: Any = None

    cccm_core = staticmethod(cccm_core)  # overridable internal (DESIGN §4.9)

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_static(self.static, "CCCMStep")

    @classmethod
    def from_static(cls, static: StaticConfig, *, vol: Any = None, coef: Any = None) -> CCCMStep:
        """The step of a composition (``static.mean``) with the program's ``VolBlock`` and
        ``CoefPrior`` kernels."""
        return cls(static.mean, vol=vol, coef=coef)

    # ----------------------------------------------------------------- helpers
    def _core_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"gram": self.static.gram, "precompute": self.static.cccm_precompute}
        if self.static.gram == "packed_tiled":
            kw["tile"] = self.static.gram_tile
        return kw

    def moments(
        self, st: ChainState, prior: PriorArrays, eq_map: EqMap | None = None
    ) -> GaussianDiag:
        """``(m_j, p_j)`` of every ``π_j`` (leading axis ``n``): ``coef.moments_eq`` batched by
        ``eq_map`` (default ``vmap``), or ``prior.eq.coef`` when no ``CoefPrior`` was given."""
        if self.coef is None:
            c = prior.eq.coef
            return GaussianDiag(mean=c.mean0, prec=c.prec0, cmask=jnp.ones_like(c.prec0))
        eq_map = eq_map or jax.vmap
        n = prior.eq.coef.mean0.shape[0]
        gcoef = st.glob.coef

        def one(cs_i: Any, i: Array, prior_i: Any) -> GaussianDiag:
            return self.coef.moments_eq(cs_i, i, prior_i, gcoef)

        return eq_map(one)(st.eq.coef, jnp.arange(n, dtype=jnp.int32), prior.eq)

    def inv_lam(self, st: ChainState, tmask: Array, eq_map: EqMap | None = None) -> Array:
        """``m_t exp(−log λ_eff,i,t)`` (n, T_pad) from ``vol.log_var_eq`` batched by ``eq_map``."""
        if self.vol is None:
            raise ValueError("CCCMStep needs the program's VolBlock: CCCMStep(static, vol=...)")
        eq_map = eq_map or jax.vmap
        gvol = st.glob.vol
        log_lam = eq_map(lambda vs: self.vol.log_var_eq(vs, gvol))(st.eq.vol)
        return masked_inv_lam(log_lam, tmask)

    def _draw(
        self, key: Key, Pi: Array, a: Array, inv_lam: Array, data: EngineData, XX: Any,
        pm: GaussianDiag,
    ) -> tuple[Array, StepInfo]:  # fmt: skip
        n, k = data.Y.shape[1], data.X.shape[1]
        Z = cccm_noise(key, n, k)
        return self.impl("cccm_core")(
            Pi, a, inv_lam, data.X, data.Y, XX, pm.mean, pm.prec, Z, **self._core_kwargs()
        )

    # ----------------------------------------------------------------- GlobalStep
    def init(
        self, key: Key, st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache
    ) -> ChainState:
        """Cold start (DESIGN §5.4): one CCCM draw with ``A = I`` and ``λ_j`` at its prior
        centre (equations decouple, so the draw is exact and ignores the old ``Π``; N04 §8).
        ``key`` is the step key of the init stream (``… → Block.G_CCCM``)."""
        (T_pad, n), k = data.Y.shape, data.X.shape[1]
        log_lam0 = _prior_log_lam(prior.eq.vol_core, _vol_rw(self.vol))
        inv_lam = masked_inv_lam(jnp.broadcast_to(log_lam0[:, None], (n, T_pad)), data.tmask)
        pm = self.moments(st, prior)
        Pi, _ = self._draw(
            key, jnp.zeros((k, n)), jnp.zeros((n, n - 1)), inv_lam, data, cache.XX, pm
        )
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=Pi)))

    def sample(
        self, key: Key, st: ChainState, ctx: SweepCtx, eq_map: EqMap
    ) -> tuple[ChainState, StepInfo]:
        """``Π ← cccm_core(Π, a, inv_lam, …)`` with ``inv_lam`` from the current vol state."""
        data = ctx.data
        inv_lam = self.inv_lam(st, data.tmask, eq_map)
        pm = self.moments(st, ctx.prior, eq_map)
        Pi, info = self._draw(key, st.glob.mean.Pi, st.eq.struct.a, inv_lam, data, ctx.cache.XX, pm)
        st = dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=Pi)))
        return st, info

    def conditional_j(
        self, st: ChainState, ctx: SweepCtx, j: int | Array, eq_map: EqMap | None = None
    ) -> GaussianPrec:
        """The exact conditional of ``π_j`` at the state (:func:`cccm_conditional_j`)."""
        data = ctx.data
        pm = self.moments(st, ctx.prior, eq_map)
        return cccm_conditional_j(
            st.glob.mean.Pi, st.eq.struct.a, self.inv_lam(st, data.tmask, eq_map), data.X,
            data.Y, pm.mean, pm.prec, j,
        )  # fmt: skip

    def log_prior(self, glob: GlobalState, prior: PriorArrays) -> Array:
        """``Σ_j log N(π_j; m_j, diag(1/p_j))`` from ``prior.eq.coef`` (fixed moments)."""
        c = prior.eq.coef
        r = glob.mean.Pi.T - c.mean0
        return 0.5 * jnp.sum(jnp.log(c.prec0) - _LOG_2PI - c.prec0 * r * r)

    def future(self, key: Key, gd: GlobDraw, H: int) -> Empty:
        """``Empty()``: ``Π`` is constant over the forecast horizon."""
        return Empty()


@dataclass(frozen=True)
class ReducedEq(KernelBase):
    """The per-equation mean of the reduced form (DESIGN §3.3, §3.7.3): no state, ``g_i ≡ 0``;
    the coefficients ``Π`` are global (:class:`CCCMStep`) and enter through
    ``fitted_reduced = XΠ`` (``ỹ = Y − XΠ``) and ``predict_reduced = Π'x``. Not fusing: the
    structure's ``contemp_step`` draws ``a_i`` (N04 §6.2)."""

    fuses_contemp: ClassVar[bool] = False
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()

    static: MeanStatic

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_static(self.static, "ReducedEq")

    @classmethod
    def from_static(cls, static: StaticConfig) -> ReducedEq:
        """The mean of a composition (``static.mean``)."""
        return cls(static.mean)

    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache:
        """``base`` with the packed products ``XX`` filled for ``gram="packed"`` (``base_cache``
        normally built them already)."""
        if self.static.gram == "packed" and isinstance(base.XX, Empty):
            return dataclasses.replace(base, XX=packed_products(data.X))
        return base

    def fitted_reduced(self, glob: GlobalState, data: EngineData) -> Array:
        """``XΠ`` (T_pad, n); zero on padded rows (``X`` is)."""
        return data.X @ glob.mean.Pi

    def init_eq(
        self, key: Key, a_i: Array, ctx: EqContext, pm: GaussianDiag, cm: GaussianDiag,
        cache_i: Any, cache: FitCache, shared: SharedPrior,
    ) -> tuple[Empty, Array]:  # fmt: skip
        return Empty(), a_i

    def sample_eq(
        self, key: Key, ms: Any, a_i: Array, ctx: EqContext, pm: GaussianDiag,
        cm: GaussianDiag | None, cache_i: Any, cache: FitCache, shared: SharedPrior,
    ) -> tuple[Empty, Array, Coupling, StepInfo]:  # fmt: skip
        """No-op: ``(Empty(), a_i, 0, ok)``; the reduced-form coefficients are drawn globally."""
        return Empty(), a_i, Coupling.zeros(), StepInfo.success()

    def conditional_eq(
        self, ms: Any, a_i: Array, ctx: EqContext, pm: GaussianDiag, cm: GaussianDiag | None,
        cache_i: Any, cache: FitCache, shared: SharedPrior,
    ) -> GaussianPrec:  # fmt: skip
        """Empty (no per-equation coefficients): ``K`` is ``0 × 0``."""
        return GaussianPrec(K=jnp.zeros((0, 0)), b=jnp.zeros((0,)))

    def fitted_eq(
        self, ms: Any, a_i: Array, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> Array:
        """``g_i ≡ 0`` (the fit ``XΠ`` is already removed from ``ỹ``)."""
        return jnp.zeros_like(ctx.target)

    def coefficients_eq(self, ms: Any, a_i: Array, shared: SharedPrior) -> Array:
        """Zeros (k): ``π_i`` lives in ``glob.mean.Pi``, outside the equation body. The Phase 2
        ``CoefPrior`` ignores it (identity local step); a Phase 3 reduced-form local-scale prior
        reads ``Π`` in its post-phase global step instead."""
        return jnp.zeros((shared.Mc.shape[0],), dtype=jnp.float64)

    def log_prior_eq(
        self, ms: Any, a_i: Array, pm: GaussianDiag, cm: GaussianDiag | None, shared: SharedPrior
    ) -> Array:
        """0: the ``Π`` prior is global (:meth:`CCCMStep.log_prior`)."""
        return jnp.zeros((), dtype=jnp.float64)

    def extract_eq(self, ms: Any, a_i: Array, shared: SharedPrior) -> Empty:
        return Empty()

    def extract_global(self, glob: GlobalState) -> GlobDraw:
        """``GlobDraw(mean=CCCMDraw(Pi))``, the other parts ``Empty()``."""
        return GlobDraw(mean=CCCMDraw(Pi=glob.mean.Pi), coef=Empty(), vol=Empty(), struct=Empty())

    def future_eq(self, key: Key, md: Any, H: int) -> Empty:
        return Empty()

    def predict_eq(self, md: Any, fut: Any, x: Array, h: Array, gfut: Any) -> Array:
        """``g_i(x_{T+h}) = 0``."""
        return jnp.zeros((), dtype=jnp.float64)

    def predict_reduced(self, gd: GlobDraw, x: Array, h: Array, gfut: Any) -> Array:
        """``Π'x`` (n,)."""
        return x @ gd.mean.Pi
