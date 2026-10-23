"""``FixedDiagonal`` coefficient prior (DESIGN §3.4, §3.7.7).

The coefficients a mean exposes (structural Linear: ``δ_i``; reduced form: ``π_j``) have the fixed
independent prior ``N(mean0, diag(1/prec0))`` of ``CoefPriorArrays`` (λ-free; DESIGN §2.6). There
is no state, no local step and no global step, and the moments never change (``varies = False``),
so the Constant-volatility fast paths of ``LinearStructural`` are allowed (DESIGN §3.7.2).

Phase 3 global-local priors (HS, NG, DL, R2D2, SSVS) are siblings with local scales in ``eq.coef``
(:meth:`sample_local_eq`), global scales in ``glob.coef`` (a post ``GlobalStep``) and per-coordinate
precisions from :meth:`moments_eq`; no mean kernel changes (DESIGN §3.4, §3.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import CoefPriorStatic, StaticConfig
from rai_bvar.engines.mcmc.keys import Key
from rai_bvar.engines.mcmc.protocols import GlobalStep, KernelBase
from rai_bvar.engines.mcmc.state import Empty, EqPrior, GaussianDiag

Array = jax.Array
PyTree = Any

__all__ = ["FixedDiagonal"]


@dataclass(frozen=True)
class FixedDiagonal(KernelBase):
    """``CoefPrior`` with fixed diagonal moments ``(prior_i.coef.mean0, prior_i.coef.prec0)``."""

    static: CoefPriorStatic = field(default_factory=CoefPriorStatic)

    varies: ClassVar[bool] = False

    @classmethod
    def from_static(cls, static: StaticConfig) -> FixedDiagonal:
        """The coefficient prior of a composition (``static.coef``)."""
        return cls(static.coef)

    def init_eq(self, key: Key, i: Array, prior_i: EqPrior) -> Empty:
        """No state."""
        return Empty()

    def moments_eq(
        self, cs_i: PyTree, i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> GaussianDiag:
        """λ-free ``(mean0, prec0)`` with every coordinate active; the mean applies any λ-scaling
        (Mode II, DESIGN §2.6)."""
        c = prior_i.coef
        mean0 = jnp.asarray(c.mean0, jnp.float64)
        return GaussianDiag(
            mean=mean0,
            prec=jnp.asarray(c.prec0, jnp.float64),
            cmask=jnp.ones_like(mean0),
        )

    def sample_local_eq(
        self, key: Key, cs_i: PyTree, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> PyTree:
        """Identity (no local scales)."""
        return cs_i

    def log_prior_eq(
        self, cs_i: PyTree, coef_i: Array, prior_i: EqPrior, glob_coef: PyTree
    ) -> Array:
        """``log p`` of the prior's own state: 0 (it has none). The Gaussian density of the
        coefficients under :meth:`moments_eq` is counted once, by the mean that owns them
        (``MeanBlock.log_prior_eq``; reduced form: the CCCM step's ``log_prior``)."""
        return jnp.zeros((), jnp.float64)

    def global_steps(self) -> tuple[GlobalStep, ...]:
        """None in Phase 2."""
        return ()
