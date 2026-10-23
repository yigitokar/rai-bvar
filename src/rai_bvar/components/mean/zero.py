"""``ZeroMean``: ``g_i ≡ 0``, no coefficients (DESIGN §3.7.7).

``y_t = A⁻¹Λ_t^{½}ε_t``. The mean does not fuse the contemporaneous step (``fuses_contemp = False``):
the sweep hands ``a_i`` to ``Recursive.contemp_step`` with ``g_i = 0`` (DESIGN §4.5). With ``n = 1``
it runs the block-level stochvol parity through the production runner (N07 §13.4; DESIGN §9.9)
and the harness toys of WP14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import MeanStatic, StaticConfig
from rai_bvar.engines.mcmc.keys import Key
from rai_bvar.engines.mcmc.protocols import Batching, KernelBase
from rai_bvar.engines.mcmc.state import (
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
)

Array = jax.Array
PyTree = Any

__all__ = ["ZeroMean"]


@dataclass(frozen=True)
class ZeroMean(KernelBase):
    """``MeanBlock`` with ``g ≡ 0``; ``n`` is the number of equations (shape of
    :meth:`predict_reduced`). Build it with :meth:`from_static`."""

    static: MeanStatic
    n: int

    fuses_contemp: ClassVar[bool] = False
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.static.kind != "zero":
            raise ValueError(f"ZeroMean needs MeanStatic(kind='zero'), got {self.static.kind!r}")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")

    @classmethod
    def from_static(cls, static: StaticConfig) -> ZeroMean:
        """The mean of a composition (``static.mean``, ``static.shapes.n``)."""
        return cls(static.mean, n=static.shapes.n)

    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache:
        """No caches."""
        return base

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
    ) -> tuple[Empty, Array]:
        return Empty(), a_i

    def sample_eq(
        self,
        key: Key,
        ms: PyTree,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> tuple[Empty, Array, Coupling, StepInfo]:
        """Nothing to draw: ``a_i`` is returned unchanged."""
        return Empty(), a_i, Coupling.zeros(), StepInfo.success()

    def conditional_eq(
        self,
        ms: PyTree,
        a_i: Array,
        ctx: EqContext,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        cache_i: PyTree,
        cache: FitCache,
        shared: SharedPrior,
    ) -> GaussianPrec:
        """The empty Gaussian (no coefficients)."""
        return GaussianPrec(K=jnp.zeros((0, 0), jnp.float64), b=jnp.zeros((0,), jnp.float64))

    def fitted_eq(
        self, ms: PyTree, a_i: Array, ctx: EqContext, cache: FitCache, shared: SharedPrior
    ) -> Array:
        """``g_i ≡ 0``."""
        return jnp.zeros_like(ctx.target)

    def coefficients_eq(self, ms: PyTree, a_i: Array, shared: SharedPrior) -> Array:
        """No coefficients: ``f64[0]``."""
        return jnp.zeros((0,), jnp.float64)

    def log_prior_eq(
        self,
        ms: PyTree,
        a_i: Array,
        pm: GaussianDiag,
        cm: GaussianDiag | None,
        shared: SharedPrior,
        *,
        log_scale: Array | float = 0.0,
    ) -> Array:
        return jnp.zeros((), jnp.float64)

    def extract_eq(self, ms: PyTree, a_i: Array, shared: SharedPrior) -> Empty:
        return Empty()

    def extract_global(self, glob: GlobalState) -> GlobDraw:
        return GlobDraw(mean=Empty(), coef=Empty(), vol=Empty(), struct=Empty())

    def future_eq(self, key: Key, md: PyTree, H: int) -> Empty:
        return Empty()

    def predict_eq(self, md: PyTree, fut: PyTree, x: Array, h: Array, gfut: PyTree) -> Array:
        """``g_i(x) = 0``."""
        return jnp.zeros((), jnp.float64)

    def predict_reduced(self, gd: GlobDraw, x: Array, h: Array, gfut: PyTree) -> Array:
        """``f64[n]`` zeros."""
        return jnp.zeros((self.n,), jnp.float64)
