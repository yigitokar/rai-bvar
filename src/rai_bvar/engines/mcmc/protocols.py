"""Block protocols of the MCMC engine (DESIGN §3). The only place the block interfaces live.

Rules every block obeys (DESIGN §3.1):

1. **Pure.** No Python branching on traced values, no host callbacks, no state outside the
   arguments. Static choices are attributes of the frozen kernel dataclass; ``lax.cond``/``switch``
   on traced flags are forbidden inside vmapped bodies (they lower to ``select``; N07 §2.1).
2. **Noise injection.** ``sample*(key, …) = *_core(noise, …) ∘ draw_noise(key, …)`` (D-17).
3. **Conditional parameters exposed.** Every exact Gibbs draw has a ``*_conditional(...)``
   returning ``GaussianPrec``, IG ``(shape, scale)`` or categorical logits; every MH step exposes
   its log acceptance ratio as a pure function.
4. **Failures are values** (``StepInfo.ok``), never exceptions; no jitter (D-18).
5. **Keys.** One key per equation; sub-keys only by ``fold_in(key, Sub.X)`` (``keys.sub_key``).
6. **Overridable internals.** A kernel calls its named internals through
   :meth:`KernelBase.impl`; the names are frozen in :data:`OVERRIDABLE` (DESIGN §4.9).

Naming: a type with suffix ``_i`` in the docs (``RecursiveEq_i``, ``EqPrior_i``, …) is one
equation's slice of the class, i.e. the same class without the leading ``n`` axis.

Mapping from the SPEC §4.2 sketch (DESIGN §3, Q12): ``MeanBlock.sample`` → ``sample_eq`` with an
``EqContext``; ``VolBlock.sample`` → ``sample_eq`` (atomic); ``VolBlock.variances`` →
``log_var_eq``; ``VolBlock.simulate_ahead`` → ``draw_ahead_eq``; ``GlobalBlock.sample`` →
``GlobalStep.sample``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar, Literal, Protocol, Self, runtime_checkable

import jax

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.keys import Block, Key
from rai_bvar.engines.mcmc.state import (
    ChainState,
    Coupling,
    EngineData,
    EqContext,
    EqPrior,
    EqState,
    FitCache,
    GaussianDiag,
    GaussianPrec,
    GlobalState,
    GlobDraw,
    LinearDraw,
    PriorArrays,
    RecursiveEq,
    SharedPrior,
    StepInfo,
    StructDraw,
    SweepCtx,
    VolDraw,
)

Array = jax.Array
PyTree = Any
MeanEq = Any  # a mean's per-equation state: LinearEq_i or Empty in Phase 2
CoefEq = Any  # a coefficient prior's per-equation state: Empty in Phase 2
CoreEq = Any  # ConstantEq_i | SVEq_i | SVRWEq_i
TransEq = Any  # OutlierEq_i | TEq_i | OutlierTEq_i | Empty
CoreDraw = Any  # ConstantDraw_i | SVDraw_i | SVRWDraw_i
TransDraw = Any  # OutlierDraw_i | TDraw_i | OutlierTDraw_i | Empty
MeanDraw = Any  # LinearDraw_i | Empty
Future = Any  # mean or global futures: Empty in Phase 2 (TVP paths, regimes later)

EqMap = Callable[[Callable[..., Any]], Callable[..., Any]]
"""The program's batching of a per-equation function: ``jax.vmap`` or a ``lax.map`` wrapper."""


class Batching(IntEnum):
    """How equations are batched; ``Program.batching = max(component.batching)`` (DESIGN §3.8)."""

    VMAP = 0
    MAP = 1  # lax.map, for kernels that do not vmap (Phase 4 bartz)

    @property
    def label(self) -> Literal["vmap", "map"]:
        return "vmap" if self is Batching.VMAP else "map"


KERNEL_SLOTS: frozenset[str] = frozenset({"structure", "mean", "coef", "vol", "global_pre"})
"""Whole kernel slots replaceable through ``compose(static, overrides=...)``."""

INTERNALS: frozenset[str] = frozenset(
    {
        "structural_gamma",
        "linear_conditional",
        "cccm_core",
        "contemp_conditional",
        "indicator_logits",
        "mixture_table",
        "h_accept",
        "c_step",
        "nc_step",
        "nc_accept",
        "nc_fold",
        "constant_conditional",
        "outlier_logits",
        "p_conditional",
        "nu_logits",
        "tau_conditional",
    }
)
"""Named kernel internals replaceable through :meth:`KernelBase.impl` (DESIGN §4.9)."""

OVERRIDABLE: frozenset[str] = KERNEL_SLOTS | INTERNALS
"""Every key accepted by ``compose(..., overrides=...)`` and ``make_sweep(..., _overrides=...)``.
Only tests (the mutation suite, block-level validation configurations) pass overrides."""


@dataclass(frozen=True)
class KernelBase:
    """Base of every kernel dataclass: a frozen, hashable object whose static choices are fields.

    ``overrides`` (keyword-only, so subclasses keep positional fields) maps names of
    :data:`INTERNALS` to replacement callables; it is part of equality and hashing, so a kernel
    with overrides never shares a cache entry with the production kernel. Production code never
    sets it.
    """

    overrides: tuple[tuple[str, Callable[..., Any]], ...] = field(
        default=(), kw_only=True, repr=False
    )

    def __post_init__(self) -> None:
        names = [name for name, _ in self.overrides]
        unknown = sorted(set(names) - INTERNALS)
        if unknown:
            raise ValueError(f"not overridable kernel internals: {unknown}; see INTERNALS")
        if len(set(names)) != len(names):
            raise ValueError("duplicate override names")

    def impl(self, name: str) -> Callable[..., Any]:
        """The override registered for ``name``, else the kernel's own attribute ``name``."""
        if name not in OVERRIDABLE:
            raise KeyError(f"{name!r} is not an overridable name; see protocols.OVERRIDABLE")
        for key, fn in self.overrides:
            if key == name:
                return fn
        default = getattr(self, name, None)
        if default is None or not callable(default):
            raise NotImplementedError(f"{type(self).__name__} has no internal {name!r}")
        return default

    def with_overrides(self, overrides: Mapping[str, Callable[..., Any]]) -> Self:
        """A copy with ``overrides`` merged into the existing ones (test seam only)."""
        merged = dict(self.overrides)
        merged.update(overrides)
        return dataclasses.replace(
            self, overrides=tuple(sorted(merged.items(), key=lambda kv: kv[0]))
        )


class MCMCNumericalError(RuntimeError):
    """A block reported a numerical failure (D-18). Raised on the host after deterministic replay
    of the failing chunk, with diagnostics recomputed from the failing block's inputs (DESIGN
    §5.7): e.g. ``diag_ratio`` (lower bound on cond₂ from ``max/min diag(L)²``), ``weight_range``,
    ``min_pivot``, ``prior_precision_range``. No jitter is ever added and no chain is dropped."""

    def __init__(
        self,
        block: str,
        equation: int,
        iteration: int,
        detail: Mapping[str, Any] | None = None,
        chain: int | None = None,
    ) -> None:
        self.block = str(block)
        self.equation = int(equation)
        self.iteration = int(iteration)
        self.detail = dict(detail or {})
        self.chain = chain
        where = f"chain {chain}, " if chain is not None else ""
        info = "; ".join(f"{k}={v}" for k, v in self.detail.items())
        super().__init__(
            f"numerical failure in block {self.block}, equation {self.equation} "
            f"({where}sweep {self.iteration})" + (f": {info}" if info else "")
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.block, self.equation, self.iteration, self.detail, self.chain))


@runtime_checkable
class CovarianceStructure(Protocol):
    """Owns ``a_i``, its prior and the residual algebra (D-28; DESIGN §3.2):
    ``e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i = target − (cols ⊙ cmask)·a_i − g_i``, with ``ỹ = Y``
    (structural means) or ``Y − XΠ`` (reduced form)."""

    static: StructStatic
    has_contemp: ClassVar[bool]  # Recursive: True; Phase 5 Factor: False
    supports_reduced_mean: ClassVar[bool]  # Recursive: True

    def init_eq(self, key: Key, i: Array, prior_i: EqPrior) -> RecursiveEq:
        """``a_i = 0``."""
        ...

    def pre_residual(self, glob: GlobalState, data: EngineData, f_reduced: Array | None) -> Array:
        """``ỹ = Y`` or ``Y − XΠ``, times ``tmask``; f64[T_pad, n]."""
        ...

    def context(
        self, i: Array, ytil: Array, eq_i: EqState, vol: VolBlock, data: EngineData, gvol: PyTree
    ) -> EqContext: ...

    def contemp_moments(self, ctx: EqContext, prior_i: EqPrior) -> GaussianDiag:
        """The ``a_i`` prior (precision ÷ λ_i when ``contemp_scaling="lambda"``)."""
        ...

    def contemp_conditional(self, ctx: EqContext, g_i: Array, cm: GaussianDiag) -> GaussianPrec:
        """``K_a = diag(prec) + Zᵀ diag(w) Z``, ``b_a = prec ⊙ mean + Zᵀ(w ⊙ (target − g_i))``,
        ``Z = cols ⊙ cmask`` (N04 §6.2, minus sign)."""
        ...

    def contemp_step(
        self, key: Key, s_i: RecursiveEq, ctx: EqContext, g_i: Array, cm: GaussianDiag
    ) -> tuple[RecursiveEq, Coupling, StepInfo]:
        """``a_i | g_i`` for non-fusing means only."""
        ...

    def residual(self, a_i: Array, ctx: EqContext, g_i: Array) -> Array:
        """``e_i``, zero on padded rows; f64[T_pad]."""
        ...

    def log_jacobian(self, glob: GlobalState, T_eff: Array) -> Array:
        """0 for a unit-triangular ``A``."""
        ...

    def extract(self, eq_struct: RecursiveEq, glob: GlobalState) -> StructDraw: ...

    def mean_map(self, sd: StructDraw, g: Array, f: Array) -> Array:
        """``A⁻¹g + f``; f64[n]."""
        ...

    def chol(self, sd: StructDraw, log_lam: Array) -> Array:
        """``A⁻¹ diag(exp(½ log λ))``: lower triangular, positive diagonal; f64[n, n]."""
        ...

    def reduced_form(self, sd: StructDraw, gamma: Array) -> Array:
        """``B = (A⁻¹Γ)'``; f64[k, n]."""
        ...

    def realign(self, s_host: PyTree, drop_front: int, append: int) -> PyTree:
        """Host-side warm-start realignment of time-indexed state (identity for Recursive)."""
        ...


@runtime_checkable
class MeanBlock(Protocol):
    """The conditional mean ``g_i(x_t)`` (DESIGN §3.3). A fusing mean draws ``(δ_i, a_i)`` jointly
    and returns the new ``a_i``; a non-fusing mean returns ``a_i`` unchanged."""

    static: MeanStatic
    fuses_contemp: ClassVar[bool]  # LinearStructural True; ReducedEq, ZeroMean False
    batching: ClassVar[Batching]  # Batching.VMAP in Phase 2
    float32_leaves: ClassVar[frozenset[str]]

    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache:
        """Fill the mean's caches (eigen / fixed-factor) on top of the base cache."""
        ...

    def fitted_reduced(self, glob: GlobalState, data: EngineData) -> Array | None:
        """Reduced form: ``XΠ`` (f64[T_pad, n]); structural and zero means: ``None``."""
        ...

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
    ) -> tuple[MeanEq, Array]: ...

    def sample_eq(
        self,
        key: Key,
        ms: MeanEq,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[MeanEq, Array, Coupling, StepInfo]: ...

    def conditional_eq(
        self,
        ms: MeanEq,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> GaussianPrec: ...

    def fitted_eq(
        self, ms: MeanEq, a_i: Array, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> Array:
        """``g_i(x_t)`` over rows (the D3 term included); f64[T_pad]."""
        ...

    def coefficients_eq(self, ms: MeanEq, a_i: Array, shared: SharedPrior) -> Array:
        """What the ``CoefPrior`` governs (Linear: ``δ_i``); f64[k]."""
        ...

    def log_prior_eq(
        self, ms: MeanEq, a_i: Array, pm: GaussianDiag, cm: GaussianDiag | None, shared: SharedPrior
    ) -> Array:
        """Active coordinates only."""
        ...

    def extract_eq(self, ms: MeanEq, a_i: Array, shared: SharedPrior) -> LinearDraw | Any:
        """Linear: ``LinearDraw(gamma f64[k])``; reduced and zero means: ``Empty()``."""
        ...

    def extract_global(self, glob: GlobalState) -> GlobDraw:
        """The draw's global parameters: the mean fills ``GlobDraw.mean`` (reduced:
        ``CCCMDraw(Pi f64[k, n])``) and leaves the other parts ``Empty()``."""
        ...

    def future_eq(self, key: Key, md: MeanDraw, H: int) -> Future:
        """``Empty()`` in Phase 2 (TVP γ paths later)."""
        ...

    def predict_eq(self, md: MeanDraw, fut: Future, x: Array, h: Array, gfut: Future) -> Array:
        """``g_i(x_{T+h})``; f64[]."""
        ...

    def predict_reduced(self, gd: GlobDraw, x: Array, h: Array, gfut: Future) -> Array:
        """``Π'x`` (reduced form) or zeros; f64[n]."""
        ...


@runtime_checkable
class SupportsCollapsedQuadratic(Protocol):
    """Optional mean capability for Phase 5 order-invariant SV (DESIGN §3.9)."""

    def collapsed_quadratic_eq(self, ctx: EqContext, pm: GaussianDiag) -> Array:
        """f64[n, n]."""
        ...


@runtime_checkable
class CoefPrior(Protocol):
    """Prior of the coefficients a mean exposes (DESIGN §3.4). Phase 3 global-local priors are
    siblings: local scales in ``eq.coef``, global scales in ``glob.coef`` via a post GlobalStep."""

    static: CoefPriorStatic
    varies: ClassVar[bool]  # False for FixedDiagonal: Constant fast paths allowed

    def init_eq(self, key: Key, i: Array, prior_i: EqPrior) -> CoefEq: ...

    def moments_eq(
        self, cs_i: CoefEq, i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> GaussianDiag:
        """λ-free ``(mean0, prec0)`` of the coefficients."""
        ...

    def sample_local_eq(
        self, key: Key, cs_i: CoefEq, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> CoefEq:
        """Identity in Phase 2."""
        ...

    def log_prior_eq(
        self, cs_i: CoefEq, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> Array: ...

    def global_steps(self) -> tuple[GlobalStep, ...]:
        """``()`` in Phase 2."""
        ...


@runtime_checkable
class VolBlock(Protocol):
    """Volatility of one equation: transitory scale ∘ persistent core (DESIGN §3.5, §4.4).

    ``sample_eq`` is atomic and its internal order is fixed: transitory ``o → p → ν → τ``, then
    ``log ẽ² = log e² − 2 log o − log τ``, then ``s → h → θ_SV`` (centred, then non-centred).
    ``log e²`` is ``where(m_t > 0, log(where(m_t > 0, e_t², 1)), 0)``: padded rows never see
    ``log 0`` (E23)."""

    static: VolStatic

    def constant_weights(self) -> bool:
        """Constant core, no transitory scale (and no global volatility)."""
        ...

    def init_eq(
        self, key: Key, resid0: Array, tmask: Array, T_eff: Array, prior_i: EqPrior
    ) -> Any: ...

    def sample_eq(
        self,
        key: Key,
        vs: Any,
        resid: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
        cpl: Coupling,
        gvol: PyTree,
    ) -> tuple[Any, StepInfo]: ...

    def log_var_eq(
        self, vs: Any, gvol: PyTree, which: Literal["effective", "persistent"] = "effective"
    ) -> Array:
        """Per-row log variance; f64[T_pad]."""
        ...

    def log_scale_eq(self, vs: Any) -> Array:
        """``log λ_i`` for a coupled Constant core, else 0."""
        ...

    def loglik_eq(self, vs: Any, resid: Array, tmask: Array, gvol: PyTree) -> Array:
        """``Σ_t m_t log N(e_t; 0, λ_eff,t)``, exact given the latents."""
        ...

    def obs_logdensity_eq(self, e: Array, log_lam_pers: Array, vd: VolDraw) -> Array:
        """Transitory-integrated ``−½ log λ + log f(e·λ^{−½})`` (N03 §2.2)."""
        ...

    def transition_eq(self, key: Key, log_lam_prev: Array, vd: VolDraw) -> Array:
        """One persistent transition step."""
        ...

    def extract_eq(self, vs: Any, T_eff: Array) -> VolDraw:
        """``h_T``, ``(μ, φ, σ)`` | ``σ`` (RW) | ``log λ``; ``log p``, ``ν``."""
        ...

    def draw_ahead_eq(
        self, key: Key, vd: VolDraw, prior_i: EqPrior, H: int, gfut: Future
    ) -> tuple[Array, Array]:
        """``(log λ_pers, log λ_eff)`` for ``T+1..T+H``."""
        ...

    def log_prior_eq(self, vs: Any, prior_i: EqPrior, tmask: Array) -> Array: ...


@runtime_checkable
class PersistentCore(Protocol):
    """ConstantCore | SVCore (AR) | SVCore (RW)."""

    def init(
        self, key: Key, log_e2: Array, tmask: Array, T_eff: Array, prior_i: EqPrior
    ) -> CoreEq: ...

    def sample(
        self,
        key: Key,
        cs: CoreEq,
        log_e2c: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
        cpl: Coupling,
    ) -> tuple[CoreEq, StepInfo]: ...

    def log_var(self, cs: CoreEq) -> Array:
        """Persistent log variance per data row; f64[T_pad]."""
        ...

    def log_scale(self, cs: CoreEq) -> Array: ...

    def extract(self, cs: CoreEq, T_eff: Array) -> CoreDraw: ...

    def ahead(self, key: Key, cd: CoreDraw, H: int) -> Array:
        """f64[H]."""
        ...

    def transition(self, key: Key, log_lam_prev: Array, cd: CoreDraw) -> Array: ...

    def log_prior(self, cs: CoreEq, prior_i: EqPrior, tmask: Array) -> Array: ...


@runtime_checkable
class TransitoryScale(Protocol):
    """none | outlier | t | outlier_t (DESIGN §3.7.6)."""

    def init(self, tmask: Array, prior_i: EqPrior) -> TransEq: ...

    def sample(
        self,
        key: Key,
        ts: TransEq,
        log_e2: Array,
        log_var_pers: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
    ) -> TransEq: ...

    def log_scale2(self, ts: TransEq) -> Array:
        """``2 log o_t + log τ_t`` (0 for none and on padded rows); f64[T_pad]."""
        ...

    def extract(self, ts: TransEq) -> TransDraw:
        """``log p``, ``ν`` (as float)."""
        ...

    def ahead(self, key: Key, td: TransDraw, H: int) -> Array:
        """Future ``2 log o + log τ``; f64[H]."""
        ...

    def log_density(self, log_z2: Array, td: TransDraw) -> Array:
        """``log f(z)`` of N03 §2.2, with ``z²`` given in logs."""
        ...

    def log_prior(self, ts: TransEq, prior_i: EqPrior, tmask: Array) -> Array: ...


@runtime_checkable
class GlobalStep(Protocol):
    """A step outside the equation body, in the pre or post phase (D-29; DESIGN §3.6). It must be
    an exact conditional of the indicator-free target (``marginal_of_indicators``)."""

    block: ClassVar[Block]
    phase: ClassVar[Literal["pre", "post"]]
    reads: ClassVar[frozenset[str]]
    writes: ClassVar[frozenset[str]]  # ⊆ glob.* (and eq.coef.* for post steps)
    marginal_of_indicators: ClassVar[bool]

    def init(
        self, key: Key, st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache
    ) -> ChainState: ...

    def sample(
        self, key: Key, st: ChainState, ctx: SweepCtx, eq_map: EqMap
    ) -> tuple[ChainState, StepInfo]: ...

    def log_prior(self, glob: GlobalState, prior: PriorArrays) -> Array: ...

    def future(self, key: Key, gd: GlobDraw, H: int) -> Future:
        """``Empty()`` in Phase 2."""
        ...
