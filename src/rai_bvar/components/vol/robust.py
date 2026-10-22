"""``VolBlockImpl``: the volatility block = transitory scale ∘ persistent core (DESIGN §3.5,
§3.7.7, §4.4; N03 §5.1; N01 §9). The only ``VolBlock`` of Phase 2.

``sample_eq`` is atomic and its internal order is fixed (not configurable)::

    log e²_t = where(m_t > 0, log(where(m_t > 0, e²_t, 1)), 0)                (E23: no log 0)
    transitory:  o → p → ν → τ  given log e² and the CURRENT persistent log λ   (N03 §5.1)
    log ẽ²_t = where(m_t > 0, log e²_t − 2 log o_t − log τ_t, 0)               (N03 §4.6)
    core:        Constant λ | ẽ, coupling   |   SV: s → h → θ_C → θ_NC        (DNP 2015)

Nothing else runs in between, so the mixture indicators (local to the SV core) are drawn
immediately before ``h`` (DESIGN §4.8 invariant 2). Padded rows never see ``log 0``; an exact
zero residual in a real row gives ``log e² = −∞``, which the transitory logits handle
(``z² = 0``), the Constant core turns into a zero contribution, and the SV core absorbs with its
offset (``y* = log c``) or, in ``zero_offset`` parity mode, masks and counts in ``acc_floor``
(N02 §2.4). One equation key serves both parts in ``init_eq``/``sample_eq``; each derives its own
``keys.Sub`` sub-keys (transitory ``O, P, NU, TAU``; Constant ``H``; SV ``S … NC_MS_ACC``). In
``draw_ahead_eq`` and ``transition_eq`` the core gets ``fold_in(key, Sub.H)``, the persistent-level
sub-key, and the transitory scale gets ``key`` (it folds ``O`` and ``TAU`` itself). A core that
consumes its key directly (``jr.normal(key, (H,))``, as ``SVCore.ahead`` does) would otherwise
read the word ``threefry(key, (0, j))``, which is the key data of ``fold_in(key, j)``: the
``Sub.O`` key once ``H ≥ 12`` and the ``Sub.TAU`` key once ``H ≥ 15`` (partitionable threefry).
That would break one consumer per key (DESIGN §3.1 rule 5, §5.6).

Views: ``log_var_eq(which="effective") = core.log_var + 2 log o + log τ`` (the mean-block
weights are ``m_t exp(−log λ_eff,t)``); ``which="persistent"`` drops the transitory part (one-s.d.
IRF shocks, N03 §5.4). ``obs_logdensity_eq = −½ log λ^pers + log f(e·λ^{−½})`` with the
transitory-integrated ``f`` of N03 §2.2 (RB LPD, particle filters). ``draw_ahead_eq`` returns
``(log λ^pers, log λ^eff)`` for ``T+1..T+H`` (N03 §5.2).

Construction: ``VolBlockImpl(static_config)`` builds the core and the transitory scale through
``engines.mcmc.registry`` with the same convention ``cls(vol_static, T_pad)``;
``VolBlockImpl(vol_static, T_pad, core=…, transitory=…)`` takes them explicitly (``compose`` and
tests). Overrides given to the block (``with_overrides``) are pushed down to both parts, so the
names of DESIGN §4.9 (``constant_conditional``, ``outlier_logits``, …, ``indicator_logits``,
``h_accept``, …) reach the kernel that calls them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp

from rai_bvar.components.vol.transitory import resolve_vol_static
from rai_bvar.engines.mcmc import (
    _jax,  # noqa: F401  (enables float64 before any computation)
    registry,
)
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import KernelBase
from rai_bvar.engines.mcmc.state import (
    Coupling,
    EqPrior,
    StepInfo,
    VolDraw,
    VolEq,
)

Array = jax.Array
PyTree = Any
_F64 = jnp.float64
_LOG_2PI = math.log(2.0 * math.pi)

__all__ = ["VolBlockImpl", "log_e2"]


def log_e2(resid: Array, tmask: Array) -> Array:
    """``log e²_t`` masked before the log: 0 on padded rows (whatever they hold), ``−∞`` for an
    exact zero in a real row (DESIGN §3.5; E23)."""
    live = tmask > 0
    return jnp.where(live, jnp.log(jnp.where(live, resid * resid, 1.0)), 0.0)


def _build(kind: str, static: VolStatic, T_pad: int) -> Any:
    """The registry kernel of ``kind`` built as ``cls(VolStatic, T_pad)`` (the vol convention)."""
    cls = registry.load(kind)
    try:
        return cls(static, T_pad)
    except TypeError as err:  # a kernel with another constructor: compose passes it explicitly
        raise TypeError(
            f"cannot build the {kind!r} kernel as {cls.__name__}(VolStatic, T_pad) ({err}); "
            f"pass it to VolBlockImpl explicitly"
        ) from err


@dataclass(frozen=True)
class VolBlockImpl(KernelBase):
    """Transitory scale ∘ persistent core for one equation (``VolBlock`` protocol)."""

    static: VolStatic
    T_pad: int | None = None
    core: Any = None
    transitory: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        resolve_vol_static(self)
        core = self.core
        if core is None:
            core = _build(self.static.persistent, self.static, self.T_pad)
        trans = self.transitory
        if trans is None:
            trans = _build(self.static.transitory, self.static, self.T_pad)
        for label, part in (("core", core), ("transitory", trans)):
            own = getattr(part, "static", None)
            if isinstance(own, VolStatic) and own != self.static:
                raise ValueError(f"the {label} kernel was built for another VolStatic")
            own_T = getattr(part, "T_pad", None)
            if isinstance(own_T, int) and own_T != self.T_pad:
                raise ValueError(f"the {label} kernel was built for T_pad={own_T}")
        if self.overrides:
            ov = dict(self.overrides)
            core = core.with_overrides(ov) if isinstance(core, KernelBase) else core
            trans = trans.with_overrides(ov) if isinstance(trans, KernelBase) else trans
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "transitory", trans)

    # ------------------------------------------------------------------ static properties
    def constant_weights(self) -> bool:
        """Constant core, no transitory scale (Phase 2 has no global volatility)."""
        return self.static.constant_weights

    # ------------------------------------------------------------------ state
    def init_eq(
        self, key: Key, resid0: Array, tmask: Array, T_eff: Array, prior_i: EqPrior
    ) -> VolEq:
        """Transitory ``o ≡ 1, τ ≡ 1, p`` = prior mean, ``ν`` = prior median; then the core's
        cold start from the initial residuals (DESIGN §5.4)."""
        trans = self.transitory.init(tmask, prior_i)
        le2 = log_e2(resid0, tmask)
        le2c = jnp.where(tmask > 0, le2 - self.transitory.log_scale2(trans), 0.0)
        return VolEq(core=self.core.init(key, le2c, tmask, T_eff, prior_i), trans=trans)

    def sample_eq(
        self,
        key: Key,
        vs: VolEq,
        resid: Array,
        tmask: Array,
        T_eff: Array,
        prior_i: EqPrior,
        cpl: Coupling,
        gvol: PyTree,
    ) -> tuple[VolEq, StepInfo]:
        """Transitory step given the current persistent log variance, then the core on the
        cleaned residuals; atomic (module docstring)."""
        le2 = log_e2(resid, tmask)
        trans = self.transitory.sample(
            key, vs.trans, le2, self.core.log_var(vs.core), tmask, T_eff, prior_i
        )
        scale2 = self.transitory.log_scale2(trans)
        le2c = jnp.where(tmask > 0, le2 - scale2, 0.0)
        core, info = self.core.sample(key, vs.core, le2c, tmask, T_eff, prior_i, cpl)
        ok = info.ok
        for leaf in jax.tree.leaves(trans):  # failures are values (D-18): non-finite p, τ
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                ok = ok & jnp.all(jnp.isfinite(leaf))
        return VolEq(core=core, trans=trans), StepInfo(ok, info.min_pivot)

    # ------------------------------------------------------------------ views
    def log_var_eq(
        self, vs: VolEq, gvol: PyTree, which: Literal["effective", "persistent"] = "effective"
    ) -> Array:
        """Per-row log variance ``log λ^pers_t (+ 2 log o_t + log τ_t)``; f64[T_pad]."""
        if which not in ("effective", "persistent"):
            raise ValueError(f"which must be 'effective' or 'persistent', got {which!r}")
        lv = self.core.log_var(vs.core)
        return lv if which == "persistent" else lv + self.transitory.log_scale2(vs.trans)

    def log_scale2_eq(self, vs: VolEq) -> Array:
        """``2 log o_t + log τ_t`` (0 without a transitory scale and on padded rows)."""
        return self.transitory.log_scale2(vs.trans)

    def log_scale_eq(self, vs: VolEq) -> Array:
        """``log λ_i`` for a coupled Constant core, else 0."""
        return self.core.log_scale(vs.core)

    def loglik_eq(self, vs: VolEq, resid: Array, tmask: Array, gvol: PyTree) -> Array:
        """``Σ_t m_t log N(e_t; 0, λ_eff,t)``, exact given the latents (masked before use)."""
        live = tmask > 0
        lv = jnp.where(live, self.log_var_eq(vs, gvol), 0.0)
        e = jnp.where(live, resid, 0.0)
        return jnp.sum(jnp.where(live, -0.5 * (_LOG_2PI + lv + e * e * jnp.exp(-lv)), 0.0))

    def obs_logdensity_eq(self, e: Array, log_lam_pers: Array, vd: VolDraw) -> Array:
        """``−½ log λ^pers + log f(z)``, ``log z² = log e² − log λ^pers`` (N03 §2.2, §5.3)."""
        e = jnp.asarray(e, _F64)
        log_z2 = jnp.log(e * e) - log_lam_pers
        return -0.5 * log_lam_pers + self.transitory.log_density(log_z2, vd.trans)

    def transition_eq(self, key: Key, log_lam_prev: Array, vd: VolDraw) -> Array:
        """One step of the persistent law (Constant: unchanged); the core's key is
        ``fold_in(key, Sub.H)`` as in ``draw_ahead_eq``."""
        return self.core.transition(sub_key(key, Sub.H), log_lam_prev, vd.core)

    def extract_eq(self, vs: VolEq, T_eff: Array) -> VolDraw:
        return VolDraw(
            core=self.core.extract(vs.core, T_eff), trans=self.transitory.extract(vs.trans)
        )

    def draw_ahead_eq(
        self, key: Key, vd: VolDraw, prior_i: EqPrior, H: int, gfut: PyTree
    ) -> tuple[Array, Array]:
        """``(log λ^pers, log λ^eff)`` for ``T+1..T+H``: the core's persistent path (key
        ``fold_in(key, Sub.H)``) plus iid transitory scales ``2 log o + log τ`` (key ``key``; its
        sub-keys ``O``, ``TAU``) (N03 §5.2; one consumer per key, module docstring)."""
        pers = self.core.ahead(sub_key(key, Sub.H), vd.core, H)
        return pers, pers + self.transitory.ahead(key, vd.trans, H)

    def log_prior_eq(self, vs: VolEq, prior_i: EqPrior, tmask: Array) -> Array:
        """``log p`` of the core parameters and latents plus the transitory ones."""
        return self.core.log_prior(vs.core, prior_i, tmask) + self.transitory.log_prior(
            vs.trans, prior_i, tmask
        )
