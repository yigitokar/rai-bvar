"""Constant persistent volatility ``λ_i`` with the coupling channel (DESIGN §3.7.5, §4.7; D-30;
N05 §5.2; N03 §4.5).

One equation. ``λ_i ~ IG(c_i, C_i)`` a priori (shape, scale; DESIGN §2.6, §6.1). Given the
cleaned residuals ``ẽ_t = e_t/(o_t √τ_t)`` (``log ẽ² = log e² − 2 log o − log τ``, from the
transitory step) and the coupling terms of the λ-scaled prior blocks::

    λ_i | · ~ IG(c_i + T_eff/2 + n_active/2,  C_i + ½ Σ_t m_t ẽ²_t + quad)

* Mode I / ``Constant("ig")``: uncoupled, ``Coupling = 0`` (N05 §5.2 row I);
* Mode II (``Constant("iw")`` × structural): ``n_active = q_i = k + i``,
  ``quad = ½(θ_i − μ₀)ᵀP̃₀(θ_i − μ₀)`` over the active coordinates (N05 §5.2 row II, verified in
  N05 §13.2 check 3);
* ``Constant("iw_equivalent")``: the same sums over the active ``a``-slots only (Chan 2022
  Prop. 1; N08 §0).

The coupling terms are consumed only when ``VolStatic.coupled`` (a static branch; plan invariant 5
of DESIGN §4.8 guarantees ``Coupling = 0`` otherwise). The draw is
``log λ = log(scale′) − loggamma(shape′)`` (N07 §9.1) with the key ``Sub.H`` of the equation's vol
key (the persistent-level draw, the Constant analogue of ``h``). Padded rows are masked before the
exponential; an exact zero residual (``log ẽ² = −∞``) contributes 0. ``log_var`` broadcasts
``log λ_i`` over the ``T_pad`` rows; ``ahead`` and ``transition`` are constant.

Overridable internal (DESIGN §4.9): :func:`constant_conditional` (mutant M13: ``T_pad`` for
``T_eff``), called as ``self.impl("constant_conditional")(log_e2c, tmask, T_eff, prior, cpl)``.

**CommonSV** (P3 §5.2; ``VolStatic.common``): the same class serves the per-equation cores of
``CommonSV``, ``λ_{i,t} = λ_i e^{h_t}`` with the common ``h`` of ``glob.vol``, so
:attr:`ConstantCore.uses_global_vol` is ``static.common`` and ``VolBlockImpl`` then passes the
global volatility state ``gvol`` (``CommonSVGlobal``) to :meth:`~ConstantCore.sample` and
:meth:`~ConstantCore.log_var` and the future of the common ``h`` to :meth:`~ConstantCore.ahead`.
Given ``h`` the conditional is the Constant one on the residuals rescaled by ``e^{−h_t/2}``::

    λ_i | e_i, h, cpl ~ IG(c_i + T_eff/2 + n_active/2,  C_i + ½ Σ_t m_t exp(log ẽ²_t − h_t) + quad)

so :func:`constant_conditional` is called on ``log ẽ²_t − h_t`` (masked before the subtraction).
``log_var`` is ``log λ_i + h_t`` on real rows (``h`` is exactly 0 on padded rows) and ``ahead``
adds the future common ``h_{T+j}``. Under the conjugate prior (Mode II, coupled) the coupling
``(n_active, quad)`` carries the coefficient and SOC/SUR-row terms of P3 §5.3 (formed by the mean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from rai_bvar.components.vol.transitory import resolve_vol_static
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc import random as R
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import KernelBase
from rai_bvar.engines.mcmc.state import (
    ConstantDraw,
    ConstantEq,
    ConstantPrior,
    Coupling,
    EqPrior,
    StepInfo,
)

Array = jax.Array
_F64 = jnp.float64

__all__ = ["ConstantCore", "constant_conditional", "log_ig_density"]


def constant_conditional(
    log_e2c: Array, tmask: Array, T_eff: Array, cp: ConstantPrior, cpl: Coupling
) -> tuple[Array, Array]:
    """IG ``(shape′, scale′)`` of ``λ_i | ·``: ``shape′ = c + T_eff/2 + n_active/2``,
    ``scale′ = C + ½ Σ_t m_t exp(log ẽ²_t) + quad`` (N05 §5.2; DESIGN §3.7.5)."""
    live = tmask > 0
    ssr = jnp.sum(jnp.where(live, jnp.exp(jnp.where(live, log_e2c, -jnp.inf)), 0.0))
    shape = cp.shape + 0.5 * jnp.asarray(T_eff, _F64) + 0.5 * cpl.n_active
    return shape, cp.scale + 0.5 * ssr + cpl.quad


def log_ig_density(log_x: Array, shape: Array, scale: Array) -> Array:
    """``log IG(x; shape, scale)`` at ``x = exp(log_x)``: the density of ``x`` (not of ``log x``),
    ``shape·log scale − lnΓ(shape) − (shape + 1) log x − scale/x``."""
    return shape * jnp.log(scale) - gammaln(shape) - (shape + 1.0) * log_x - scale * jnp.exp(-log_x)


@dataclass(frozen=True)
class ConstantCore(KernelBase):
    """``PersistentCore`` with a time-constant variance ``λ_i`` (state ``ConstantEq.log_lam``).

    Construct with the ``StaticConfig`` or with ``(VolStatic, T_pad)``; ``T_pad`` sizes
    :meth:`log_var`."""

    static: VolStatic
    T_pad: int | None = None

    constant_conditional = staticmethod(constant_conditional)

    def __post_init__(self) -> None:
        super().__post_init__()
        resolve_vol_static(self)
        if self.static.persistent != "constant":
            raise ValueError(
                f"ConstantCore needs VolStatic.persistent='constant', got {self.static.persistent!r}"
            )

    def _coupling(self, cpl: Coupling) -> Coupling:
        return cpl if self.static.coupled else Coupling.zeros()

    @property
    def uses_global_vol(self) -> bool:
        """Whether the core reads the common ``h`` of ``glob.vol`` (CommonSV, P3 §2.4, §5.2)."""
        return bool(self.static.common)

    def _common_h(self, gvol: Any) -> Array:
        """The common ``h`` (f64[T_pad]) of ``gvol``; required when :attr:`uses_global_vol`."""
        h = getattr(gvol, "h", None)
        if h is None:
            raise ValueError(
                "the CommonSV Constant core needs the global volatility state glob.vol "
                "(CommonSVGlobal) as gvol"
            )
        return jnp.asarray(h, _F64)

    def common_log_e2(self, log_e2c: Array, tmask: Array, gvol: Any) -> Array:
        """``log ẽ²_t − h_t`` on real rows, 0 on padded rows (CommonSV; masked before use)."""
        live = tmask > 0
        h = self._common_h(gvol)
        return jnp.where(live, log_e2c - jnp.where(live, h, 0.0), 0.0)

    def conditional(
        self,
        log_e2c: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
        cpl: Coupling,
        gvol: Any = None,
    ) -> tuple[Array, Array]:
        """The exposed exact conditional ``(shape′, scale′)`` (DESIGN §3.1 rule 3); under
        CommonSV the residuals are rescaled by the common ``e^{−h_t/2}`` first (module
        docstring), with ``gvol`` the ``CommonSVGlobal`` state."""
        if self.uses_global_vol:
            log_e2c = self.common_log_e2(log_e2c, tmask, gvol)
        return self.impl("constant_conditional")(
            log_e2c, tmask, T_eff, prior_i.vol_core, self._coupling(cpl)
        )

    def init(
        self, key: Key, log_e2: Array, tmask: Array, T_eff: Array, prior_i: EqPrior
    ) -> ConstantEq:
        """``log λ = log(Σ_t m_t e²_t/T_eff)`` from the initial residuals (DESIGN §5.4); if every
        real residual is exactly zero, the prior mode ``log(C/(c + 1))`` instead."""
        live = tmask > 0
        lse = logsumexp(jnp.where(live, log_e2, -jnp.inf))
        log_lam = lse - jnp.log(jnp.asarray(T_eff, _F64))
        cp = prior_i.vol_core
        mode = jnp.log(cp.scale) - jnp.log(cp.shape + 1.0)
        return ConstantEq(log_lam=jnp.where(jnp.isfinite(log_lam), log_lam, mode))

    def sample(
        self,
        key: Key,
        cs: ConstantEq,
        log_e2c: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
        cpl: Coupling,
        gvol: Any = None,
    ) -> tuple[ConstantEq, StepInfo]:
        """One exact draw ``log λ = log(scale′) − loggamma(shape′)`` (``Sub.H``); ``gvol`` is
        read only under CommonSV (:attr:`uses_global_vol`)."""
        shape, scale = self.conditional(log_e2c, tmask, T_eff, prior_i, cpl, gvol)
        log_lam = jnp.log(scale) - R.loggamma(sub_key(key, Sub.H), shape)
        ok = jnp.isfinite(log_lam) & (shape > 0.0) & (scale > 0.0)
        return ConstantEq(log_lam=log_lam), StepInfo(ok, jnp.full((), jnp.inf))

    def log_var(self, cs: ConstantEq, gvol: Any = None) -> Array:
        """``log λ_i`` on every row; under CommonSV ``log λ_i + h_t`` (``h`` is 0 on padded
        rows)."""
        lv = jnp.broadcast_to(cs.log_lam, (self.T_pad,))
        return lv + self._common_h(gvol) if self.uses_global_vol else lv

    def log_scale(self, cs: ConstantEq) -> Array:
        """``log λ_i`` when coupled (it scales the λ-scaled prior blocks), else 0."""
        return cs.log_lam if self.static.coupled else jnp.zeros((), _F64)

    def extract(self, cs: ConstantEq, T_eff: Array) -> ConstantDraw:
        return ConstantDraw(log_lam=cs.log_lam)

    def ahead(self, key: Key, cd: ConstantDraw, H: int, gfut: Any = None) -> Array:
        """``log λ_i`` for ``T+1..T+H``; under CommonSV plus the future common ``h_{T+j}``
        (``gfut``, f64[H], the ``CommonSVStep.future`` of the draw)."""
        out = jnp.broadcast_to(cd.log_lam, (H,))
        if self.uses_global_vol:
            if gfut is None:
                raise ValueError("the CommonSV Constant core needs the future common h (gfut)")
            out = out + jnp.asarray(gfut, _F64)
        return out

    def transition(self, key: Key, log_lam_prev: Array, cd: ConstantDraw) -> Array:
        return jnp.asarray(cd.log_lam, _F64)

    def log_prior(self, cs: ConstantEq, prior_i: EqPrior, tmask: Array) -> Array:
        """``log IG(λ; c, C)`` (density of ``λ``)."""
        cp = prior_i.vol_core
        return log_ig_density(cs.log_lam, cp.shape, cp.scale)
