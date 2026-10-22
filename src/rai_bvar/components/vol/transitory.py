"""Transitory scales of the volatility block: none, outliers, Student-t, outliers + t
(DESIGN §3.5, §3.7.6, §4.4; D-8, D-9; N03 §2–§6).

One equation, conditional on the residual ``e`` and the persistent log variance ``log λ^pers``
(``h``, or ``log λ`` of a Constant core); with ``log z²_t = log e²_t − log λ^pers_t``::

    e_t = exp(½ log λ^pers_t) · o_t · √τ_t · ε_t,     ε_t ~ N(0, 1)                  (N03 §2.1)
    o_t ∈ {1..K}:  w_1 = 1 − p,  w_k = p/(K − 1)  (k ≥ 2),  p ~ Beta(a_p, b_p)        (D-9)
    τ_t ~ IG(ν/2, c(ν)/2),  c(ν) = ν − 2 (standardized, E τ = 1) | ν (unit)        (D-8)
    ν ~ π on the static grid (prior weights ``log_nu_w`` are data)

Steps, in this order inside ``VolBlock.sample_eq`` (N03 §5.1; exact conditionals given ``h``,
marginal of the mixture indicators):

=============  =================================================================================
``outlier``    ``o | e,h,p`` (§4.1) → ``p | o`` (§4.2)
``t``          ``ν | e,h`` with τ integrated (§4.3a) → ``τ | e,h,ν`` (§4.4)
``outlier_t``  ``o | e,h,ν,p`` (τ integrated, §4.1) → ``p | o`` → ``ν | e,h,o`` (τ integrated)
               → ``τ | e,h,o,ν``
=============  =================================================================================

The τ-collapsed steps are partially collapsed Gibbs blocks whose trimmed τ is never conditioned on
before it is redrawn (van Dyk & Park 2008; N03 §5.1 item 2). Conditionals (all exact):

    o:  ℓ_tk = log w_k − log k − ½ z²_t/k²                          (Gaussian: ``outlier``)
        ℓ_tk = log w_k − log k − ((ν+1)/2) log1p(z²_t/(c k²))       (τ integrated: ``outlier_t``)
    p:  Beta(a_p + N₁, b_p + T_eff − N₁),  N₁ = Σ_t m_t 1{o_t > 1}
    ν:  ℓ_g = log π_g + T_eff·A_g − ((ν_g+1)/2) Σ_t m_t log1p(r²_t/c_g),  r² = z²/o²,
        A_g = lnΓ((ν_g+1)/2) − lnΓ(ν_g/2) − ½ log(π c_g)            (JPR 2004 eq. 7 for c = ν)
    τ:  IG((ν+1)/2, (c(ν) + r²_t)/2), drawn as log τ = log scale − loggamma(shape) (N07 §9.1)

Numerics (pilot §5, decision 12–13): the ν sums use ``log(c_g + r²) − log c_g`` with ``r²``
computed once per row (rows with ``log r² > 700`` use ``log r²``, exact in float64); the Gaussian
outlier logits clamp ``log z²`` at 700 before the exponential (adjacent logits then differ by
more than 1e290, so ``o = K`` with probability one either way). Every sum over ``t`` is masked
with ``m_t``; padded rows hold ``o = 1``, ``τ = 1`` (``o_idx = 0``, ``log τ = 0``) and are
excluded from ``N₁``, the ν log-likelihood and every sum (DESIGN §2.3). Exact zero residuals
(``log z² = −∞``) are harmless here (``z² = 0``; N03 §6).

Overridable internals (DESIGN §4.9, test seam only): :func:`outlier_logits`,
:func:`p_conditional`, :func:`nu_logits`, :func:`tau_conditional`. Each kernel calls them through
``self.impl(name)`` with exactly the signatures below.

Sub-keys of the equation's vol key (``keys.Sub``): ``O`` (one uniform per row), ``P`` (two
log-gamma draws), ``NU`` (one uniform), ``TAU`` (one log-gamma per row). ``ahead`` uses ``O`` and
``TAU`` below the forecast key it receives.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr
from jax.scipy.special import gammaln, logsumexp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc import random as R
from rai_bvar.engines.mcmc.config import StaticConfig, VolStatic
from rai_bvar.engines.mcmc.keys import Key, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import KernelBase
from rai_bvar.engines.mcmc.state import (
    Empty,
    EqPrior,
    OutlierDraw,
    OutlierEq,
    OutlierTDraw,
    OutlierTEq,
    TDraw,
    TEq,
    TransitoryPrior,
)

Array = jax.Array
_F64 = jnp.float64
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)
_EXP_CLAMP = 700.0  # log z² clamp of the Gaussian outlier logits; log r² pass-through of the ν sums

__all__ = [
    "NoTransitory",
    "OutlierScale",
    "OutlierTScale",
    "TScale",
    "log_t_density",
    "masked_log_z2",
    "nu_constants",
    "nu_logits",
    "outlier_log_weights",
    "outlier_logits",
    "p_conditional",
    "resolve_vol_static",
    "tau_conditional",
]


# --------------------------------------------------------------------------- static constants


@functools.cache
def nu_constants(
    nu_grid: tuple[int, ...], t_scale: Literal["standardized", "unit"]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ν_g, log c(ν_g), A_g)`` on the static grid as read-only NumPy float64 arrays, with
    ``c(ν) = ν − 2`` (standardized) or ``ν`` (unit) and
    ``A_g = lnΓ((ν_g+1)/2) − lnΓ(ν_g/2) − ½ log(π c_g)`` (N03 §4.3a, §6). They become constants
    at trace time (never module-level ``jnp`` arrays; DESIGN §1.1 rule 3)."""
    nu = np.asarray(nu_grid, dtype=np.float64)
    c = nu - 2.0 if t_scale == "standardized" else nu.copy()
    if np.any(c <= 0.0):
        raise ValueError(f"c(ν) must be positive on the ν grid ({t_scale}); got grid {nu_grid}")
    A = np.array(
        [math.lgamma((v + 1.0) / 2.0) - math.lgamma(v / 2.0) - 0.5 * math.log(math.pi * cv)
         for v, cv in zip(nu, c, strict=True)]
    )  # fmt: skip
    out = (nu, np.log(c), A)
    for a in out:
        a.setflags(write=False)
    return out


def _log_k(K: int) -> np.ndarray:
    return np.log(np.arange(1, K + 1, dtype=np.float64))


# --------------------------------------------------------------------------- pure conditionals


def masked_log_z2(log_e2: Array, log_var_pers: Array, tmask: Array) -> Array:
    """``log z²_t = log e²_t − log λ^pers_t`` on real rows, 0 on padded rows (masked before use;
    ``−∞`` on a real row with an exact zero residual)."""
    return jnp.where(tmask > 0, log_e2 - log_var_pers, 0.0)


def outlier_log_weights(log_p: Array, log1m_p: Array, K: int) -> Array:
    """``log w_k`` of the outlier grid ``o ∈ {1..K}``: ``[log(1 − p), log p − log(K − 1), …]``
    (D-9); shape ``(..., K)``."""
    rest = jnp.broadcast_to(
        jnp.asarray(log_p, _F64)[..., None] - math.log(K - 1), (*jnp.shape(log_p), K - 1)
    )
    return jnp.concatenate([jnp.asarray(log1m_p, _F64)[..., None], rest], axis=-1)


def outlier_logits(
    log_z2: Array,
    log_p: Array,
    log1m_p: Array,
    K: int,
    nu: Array | None = None,
    log_c: Array | None = None,
) -> Array:
    """Unnormalised log pmf of ``o_t`` on ``{1..K}``; shape ``(T, K)`` (N03 §4.1).

    Gaussian (``nu is None``, transitory ``"outlier"``): ``ℓ_tk = log w_k − log k − ½ z²/k²``
    with ``log z²`` clamped at 700 before the exponential. τ integrated (``nu``, ``log_c`` given,
    transitory ``"outlier_t"``): ``ℓ_tk = log w_k − log k − ((ν+1)/2) log1p(z²/(c k²))``. The
    ``−log k`` term is the Gaussian normalising factor (the M9 mutant drops it)."""
    log_k = jnp.asarray(_log_k(K))
    logw = outlier_log_weights(log_p, log1m_p, K)
    log_z2 = jnp.asarray(log_z2, _F64)
    if nu is None:
        quad = 0.5 * jnp.exp(jnp.minimum(log_z2, _EXP_CLAMP)[..., None] - 2.0 * log_k)
    else:
        if log_c is None:
            raise ValueError("outlier_logits: log_c is required with nu (τ-collapsed form)")
        quad = 0.5 * (nu + 1.0) * jnp.logaddexp(0.0, log_z2[..., None] - 2.0 * log_k - log_c)
    return logw - log_k - quad


def p_conditional(
    o_idx: Array, tmask: Array, T_eff: Array, a_p: Array, b_p: Array
) -> tuple[Array, Array]:
    """``p | o ~ Beta(a_p + N₁, b_p + T_eff − N₁)``, ``N₁ = Σ_t m_t 1{o_t > 1}`` (N03 §4.2)."""
    n1 = jnp.sum(jnp.where(tmask > 0, (o_idx > 0).astype(_F64), 0.0))
    return a_p + n1, b_p + jnp.asarray(T_eff, _F64) - n1


def nu_logits(
    log_r2: Array,
    tmask: Array,
    T_eff: Array,
    log_nu_w: Array,
    nu_grid: tuple[int, ...],
    t_scale: Literal["standardized", "unit"],
) -> Array:
    """τ-collapsed log pmf of ν on the static grid (N03 §4.3a); shape ``(G,)``::

        ℓ_g = log π_g + T_eff·A_g − ((ν_g+1)/2) Σ_t m_t [log(c_g + r²_t) − log c_g]

    ``r²_t = exp(log r²_t)`` once per row; rows with ``log r² > 700`` contribute
    ``log r² − log c_g`` (exact in float64). Grid points with ``log π_g = −∞`` keep ``−∞``."""
    nu, log_c, A = (jnp.asarray(a) for a in nu_constants(tuple(nu_grid), t_scale))
    log_r2 = jnp.asarray(log_r2, _F64)
    big = log_r2 > _EXP_CLAMP
    r2 = jnp.exp(jnp.where(big, 0.0, log_r2))
    lsum = jnp.log(jnp.exp(log_c)[None, :] + r2[:, None])
    val = jnp.where(big[:, None], log_r2[:, None], lsum) - log_c[None, :]
    S = jnp.sum(jnp.where(tmask[:, None] > 0, val, 0.0), axis=0)
    return log_nu_w + jnp.asarray(T_eff, _F64) * A - 0.5 * (nu + 1.0) * S


def tau_conditional(log_r2: Array, nu: Array, log_c: Array) -> tuple[Array, Array]:
    """``τ_t | e,h,o,ν ~ IG((ν+1)/2, (c(ν) + r²_t)/2)`` (N03 §4.4): returns ``(shape, log
    scale_t)`` with ``log scale_t = logaddexp(log c, log r²_t) − log 2`` (M4 mutant: ``ν/2``)."""
    return 0.5 * (nu + 1.0), jnp.logaddexp(log_c, log_r2) - _LOG_2


def log_t_density(log_x2: Array, nu: Array, log_c: Array) -> Array:
    """``log t_{c,ν}(x)`` from ``log x²``: ``lnΓ((ν+1)/2) − lnΓ(ν/2) − ½ log(π c) −
    ((ν+1)/2) log1p(x²/c)``, the density of ``√τ ε`` with ``τ ~ IG(ν/2, c/2)`` (N03 §1)."""
    return (
        gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (math.log(math.pi) + log_c)
        - 0.5 * (nu + 1.0) * jnp.logaddexp(0.0, log_x2 - log_c)
    )  # fmt: skip


def _log_normal_density(log_x2: Array) -> Array:
    """``log φ(x)`` from ``log x²``."""
    return -0.5 * _LOG_2PI - 0.5 * jnp.exp(log_x2)


def _log_beta_density(log_p: Array, log1m_p: Array, a: Array, b: Array) -> Array:
    """``log Beta(p; a, b)`` from ``log p`` and ``log(1 − p)`` (lnΓ sums: ``betaln`` is inexact
    in jax float64)."""
    return (a - 1.0) * log_p + (b - 1.0) * log1m_p - (gammaln(a) + gammaln(b) - gammaln(a + b))


def _log_ig_density(log_x: Array, shape: Array, scale: Array) -> Array:
    """``log IG(x; shape, scale)`` at ``x = exp(log_x)`` (density of ``x``)."""
    return shape * jnp.log(scale) - gammaln(shape) - (shape + 1.0) * log_x - scale * jnp.exp(-log_x)


# --------------------------------------------------------------------------- kernel base


def resolve_vol_static(obj: Any) -> None:
    """Accept ``(StaticConfig)`` or ``(VolStatic, T_pad)`` and store ``(VolStatic, T_pad)``."""
    static = obj.static
    if isinstance(static, StaticConfig):
        if obj.T_pad is not None and int(obj.T_pad) != static.shapes.T_pad:
            raise ValueError(
                f"T_pad={obj.T_pad} contradicts StaticConfig T_pad={static.shapes.T_pad}"
            )
        object.__setattr__(obj, "T_pad", static.shapes.T_pad)
        object.__setattr__(obj, "static", static.vol)
    elif not isinstance(static, VolStatic):
        raise TypeError(
            f"{type(obj).__name__}.static must be a VolStatic (with T_pad) or a StaticConfig"
        )
    T_pad = obj.T_pad
    if isinstance(T_pad, bool) or not isinstance(T_pad, int | np.integer) or T_pad < 1:
        raise TypeError(
            f"{type(obj).__name__} needs T_pad, a positive integer (or pass the StaticConfig)"
        )
    object.__setattr__(obj, "T_pad", int(T_pad))


@dataclass(frozen=True)
class _Transitory(KernelBase):
    """Shared fields and steps of the transitory scales. ``static`` is the ``VolStatic`` (or a
    ``StaticConfig``, unpacked at construction); ``T_pad`` the compiled number of rows."""

    static: VolStatic
    T_pad: int | None = None
    kind: ClassVar[str] = ""

    outlier_logits = staticmethod(outlier_logits)
    p_conditional = staticmethod(p_conditional)
    nu_logits = staticmethod(nu_logits)
    tau_conditional = staticmethod(tau_conditional)

    def __post_init__(self) -> None:
        super().__post_init__()
        resolve_vol_static(self)
        vol = self.static
        if vol.transitory != self.kind:
            raise ValueError(
                f"{type(self).__name__} implements transitory={self.kind!r}, "
                f"but VolStatic.transitory={vol.transitory!r}"
            )
        if vol.has_outliers and vol.outlier_K < 2:
            raise ValueError(f"outlier_K must be >= 2 (w_k = p/(K−1)), got {vol.outlier_K}")
        if vol.has_t:
            nu_constants(vol.nu_grid, vol.t_scale)  # raises for c(ν) ≤ 0 (standardized, ν ≤ 2)

    # ------------------------------------------------------------------ static helpers
    @property
    def K(self) -> int:
        return self.static.outlier_K

    def _nu(self) -> tuple[Array, Array, Array]:
        return tuple(jnp.asarray(a) for a in nu_constants(self.static.nu_grid, self.static.t_scale))

    def nu_value(self, nu_idx: Array) -> Array:
        """``ν = nu_grid[nu_idx]`` as float64."""
        return self._nu()[0][nu_idx]

    def _zeros(self) -> Array:
        return jnp.zeros((self.T_pad,), _F64)

    # ------------------------------------------------------------------ steps (keys: one per eq)
    def o_step(
        self,
        key: Key,
        log_z2: Array,
        tmask: Array,
        log_p: Array,
        log1m_p: Array,
        nu_idx: Array | None = None,
    ) -> tuple[Array, Array]:
        """``o | e,h,p`` (Gaussian) or ``o | e,h,ν,p`` (τ integrated, given ``nu_idx``).

        One float64 uniform per row (``Sub.O``), inverse-CDF categorical (D-4). Returns
        ``(o_idx, π)`` with ``o_idx = o − 1`` (0 on padded rows) and the Rao-Blackwellised
        ``π_t = P(o_t > 1 | ·)`` (0 on padded rows)."""
        if nu_idx is None:
            logits = self.impl("outlier_logits")(log_z2, log_p, log1m_p, self.K)
        else:
            nu, log_c, _ = self._nu()
            logits = self.impl("outlier_logits")(
                log_z2, log_p, log1m_p, self.K, nu[nu_idx], log_c[nu_idx]
            )
        u = jr.uniform(sub_key(key, Sub.O), log_z2.shape, dtype=_F64)
        o_idx, lse = R.categorical_inverse_cdf_lse(u, logits)
        live = tmask > 0
        pi = jnp.exp(logsumexp(logits[..., 1:], axis=-1) - lse)
        return jnp.where(live, o_idx, 0), jnp.where(live, pi, 0.0)

    def p_step(
        self, key: Key, o_idx: Array, tmask: Array, T_eff: Array, tp: TransitoryPrior
    ) -> tuple[Array, Array]:
        """``p | o`` (``Sub.P``): ``(log p, log(1 − p))`` from two log-gamma draws (WP4)."""
        a, b = self.impl("p_conditional")(o_idx, tmask, T_eff, tp.a_p, tp.b_p)
        return R.log_beta(sub_key(key, Sub.P), a, b)

    def nu_step(
        self, key: Key, log_r2: Array, tmask: Array, T_eff: Array, tp: TransitoryPrior
    ) -> Array:
        """``ν | e,h,o`` with τ integrated (``Sub.NU``): grid index, int32."""
        logits = self.impl("nu_logits")(
            log_r2, tmask, T_eff, tp.log_nu_w, self.static.nu_grid, self.static.t_scale
        )
        u = jr.uniform(sub_key(key, Sub.NU), (), dtype=_F64)
        return R.categorical_inverse_cdf(u, logits)

    def tau_step(self, key: Key, log_r2: Array, tmask: Array, nu_idx: Array) -> Array:
        """``τ | e,h,o,ν`` (``Sub.TAU``): ``log τ_t = log scale_t − loggamma(shape)``, 0 on
        padded rows."""
        nu, log_c, _ = self._nu()
        shape, log_scale = self.impl("tau_conditional")(log_r2, nu[nu_idx], log_c[nu_idx])
        lg = R.loggamma(sub_key(key, Sub.TAU), shape, log_r2.shape)
        return jnp.where(tmask > 0, log_scale - lg, 0.0)

    # ------------------------------------------------------------------ prior pieces
    def _p_init(self, tp: TransitoryPrior) -> tuple[Array, Array]:
        """Prior mean of ``p`` in logs: ``(log a − log(a+b), log b − log(a+b))`` (DESIGN §5.4)."""
        tot = jnp.log(tp.a_p + tp.b_p)
        return jnp.log(tp.a_p) - tot, jnp.log(tp.b_p) - tot

    def _nu_init(self, tp: TransitoryPrior) -> Array:
        """Prior median of ν on the grid (the grid median for a uniform prior; DESIGN §5.4)."""
        cdf = jnp.cumsum(jnp.exp(tp.log_nu_w - logsumexp(tp.log_nu_w)))
        G = len(self.static.nu_grid)
        return jnp.minimum(jnp.sum(cdf < 0.5 - 1e-12), G - 1).astype(jnp.int32)

    def _outlier_log_prior(
        self, o_idx: Array, log_p: Array, log1m_p: Array, tp: TransitoryPrior, tmask: Array
    ) -> Array:
        lw = jnp.where(o_idx == 0, log1m_p, log_p - math.log(self.K - 1))
        return _log_beta_density(log_p, log1m_p, tp.a_p, tp.b_p) + jnp.sum(
            jnp.where(tmask > 0, lw, 0.0)
        )

    def _t_log_prior(
        self, log_tau: Array, nu_idx: Array, tp: TransitoryPrior, tmask: Array
    ) -> Array:
        nu, log_c, _ = self._nu()
        lt = _log_ig_density(log_tau, 0.5 * nu[nu_idx], 0.5 * jnp.exp(log_c[nu_idx]))
        return tp.log_nu_w[nu_idx] - logsumexp(tp.log_nu_w) + jnp.sum(jnp.where(tmask > 0, lt, 0.0))

    # ------------------------------------------------------------------ ahead pieces
    def _o_ahead(self, key: Key, log_p: Array, H: int) -> Array:
        """``2 log o_{T+j}``: ``o = 1`` if ``u ≥ p``, else ``2 + ⌊(u/p)(K − 1)⌋`` (N03 §5.2)."""
        u = jr.uniform(sub_key(key, Sub.O), (H,), dtype=_F64)
        p = jnp.exp(log_p)
        hit = u < p
        ratio = jnp.where(hit, u / jnp.where(p > 0.0, p, 1.0), 0.0)
        o = jnp.where(hit, jnp.minimum(2.0 + jnp.floor(ratio * (self.K - 1)), float(self.K)), 1.0)
        return 2.0 * jnp.log(o)

    def _tau_ahead(self, key: Key, nu: Array, H: int) -> Array:
        """``log τ_{T+j} = log(c(ν)/2) − loggamma(ν/2)`` (N03 §5.2)."""
        c = nu - 2.0 if self.static.t_scale == "standardized" else nu
        return jnp.log(0.5 * c) - R.loggamma(sub_key(key, Sub.TAU), 0.5 * nu, (H,))

    def _t_log_density(self, log_x2: Array, nu: Array) -> Array:
        c = nu - 2.0 if self.static.t_scale == "standardized" else nu
        return log_t_density(log_x2, nu, jnp.log(c))

    def _mix_log_density(self, log_z2: Array, log_p: Array, log1m_p: Array, comp) -> Array:
        """``log Σ_k w_k k⁻¹ f(z/k)`` with ``comp(log (z/k)²) = log f``."""
        log_k = jnp.asarray(_log_k(self.K))
        logw = outlier_log_weights(log_p, log1m_p, self.K)
        dens = comp(jnp.asarray(log_z2, _F64)[..., None] - 2.0 * log_k)
        return logsumexp(logw - log_k + dens, axis=-1)


# --------------------------------------------------------------------------- the four scales


@dataclass(frozen=True)
class NoTransitory(_Transitory):
    """No transitory scale: ``o ≡ 1``, ``τ ≡ 1``; ``f = φ`` (N03 §2.2)."""

    kind: ClassVar[str] = "none"

    def init(self, tmask: Array, prior_i: EqPrior) -> Empty:
        return Empty()

    def sample(
        self, key: Key, ts: Empty, log_e2: Array, log_var_pers: Array, tmask: Array,
        T_eff: Array, prior_i: EqPrior,
    ) -> Empty:  # fmt: skip
        return Empty()

    def log_scale2(self, ts: Empty) -> Array:
        return self._zeros()

    def extract(self, ts: Empty) -> Empty:
        return Empty()

    def ahead(self, key: Key, td: Empty, H: int) -> Array:
        return jnp.zeros((H,), _F64)

    def log_density(self, log_z2: Array, td: Empty) -> Array:
        return _log_normal_density(jnp.asarray(log_z2, _F64))

    def log_prior(self, ts: Empty, prior_i: EqPrior, tmask: Array) -> Array:
        return jnp.zeros((), _F64)


@dataclass(frozen=True)
class OutlierScale(_Transitory):
    """Outliers (SVO; CCMM 2024): ``o | e,h,p`` → ``p | o``; ``f(z) = Σ_k w_k k⁻¹ φ(z/k)``."""

    kind: ClassVar[str] = "outlier"

    def init(self, tmask: Array, prior_i: EqPrior) -> OutlierEq:
        log_p, log1m_p = self._p_init(prior_i.transitory)
        z = jnp.zeros(tmask.shape, _F64)
        return OutlierEq(o_idx=jnp.zeros(tmask.shape, jnp.int32), log_p=log_p, log1m_p=log1m_p,
                         rb_out=z)  # fmt: skip

    def sample(
        self, key: Key, ts: OutlierEq, log_e2: Array, log_var_pers: Array, tmask: Array,
        T_eff: Array, prior_i: EqPrior,
    ) -> OutlierEq:  # fmt: skip
        log_z2 = masked_log_z2(log_e2, log_var_pers, tmask)
        o_idx, pi = self.o_step(key, log_z2, tmask, ts.log_p, ts.log1m_p)
        log_p, log1m_p = self.p_step(key, o_idx, tmask, T_eff, prior_i.transitory)
        return OutlierEq(o_idx=o_idx, log_p=log_p, log1m_p=log1m_p, rb_out=ts.rb_out + pi)

    def log_scale2(self, ts: OutlierEq) -> Array:
        return 2.0 * jnp.log(ts.o_idx.astype(_F64) + 1.0)

    def extract(self, ts: OutlierEq) -> OutlierDraw:
        return OutlierDraw(log_p=ts.log_p, log1m_p=ts.log1m_p)

    def ahead(self, key: Key, td: OutlierDraw, H: int) -> Array:
        return self._o_ahead(key, td.log_p, H)

    def log_density(self, log_z2: Array, td: OutlierDraw) -> Array:
        return self._mix_log_density(log_z2, td.log_p, td.log1m_p, _log_normal_density)

    def log_prior(self, ts: OutlierEq, prior_i: EqPrior, tmask: Array) -> Array:
        return self._outlier_log_prior(ts.o_idx, ts.log_p, ts.log1m_p, prior_i.transitory, tmask)


@dataclass(frozen=True)
class TScale(_Transitory):
    """Student-t (SV-t; D-8): ``ν | e,h`` (τ integrated) → ``τ | e,h,ν``; ``f = t_{c(ν),ν}``.
    ``acc_nu_move`` counts sweeps with ``ν′ ≠ ν`` (escape probability, N03 §6)."""

    kind: ClassVar[str] = "t"

    def init(self, tmask: Array, prior_i: EqPrior) -> TEq:
        return TEq(log_tau=jnp.zeros(tmask.shape, _F64), nu_idx=self._nu_init(prior_i.transitory),
                   acc_nu_move=jnp.zeros((), _F64))  # fmt: skip

    def sample(
        self, key: Key, ts: TEq, log_e2: Array, log_var_pers: Array, tmask: Array, T_eff: Array,
        prior_i: EqPrior,
    ) -> TEq:  # fmt: skip
        log_r2 = masked_log_z2(log_e2, log_var_pers, tmask)
        nu_idx = self.nu_step(key, log_r2, tmask, T_eff, prior_i.transitory)
        log_tau = self.tau_step(key, log_r2, tmask, nu_idx)
        moved = (nu_idx != ts.nu_idx).astype(_F64)
        return TEq(log_tau=log_tau, nu_idx=nu_idx, acc_nu_move=ts.acc_nu_move + moved)

    def log_scale2(self, ts: TEq) -> Array:
        return ts.log_tau

    def extract(self, ts: TEq) -> TDraw:
        return TDraw(nu=self.nu_value(ts.nu_idx))

    def ahead(self, key: Key, td: TDraw, H: int) -> Array:
        return self._tau_ahead(key, td.nu, H)

    def log_density(self, log_z2: Array, td: TDraw) -> Array:
        return self._t_log_density(jnp.asarray(log_z2, _F64), td.nu)

    def log_prior(self, ts: TEq, prior_i: EqPrior, tmask: Array) -> Array:
        return self._t_log_prior(ts.log_tau, ts.nu_idx, prior_i.transitory, tmask)


@dataclass(frozen=True)
class OutlierTScale(_Transitory):
    """Outliers and Student-t (SVO-t): ``o | e,h,ν,p`` (τ integrated) → ``p | o`` →
    ``ν | e,h,o`` (τ integrated) → ``τ | e,h,o,ν``; ``f(z) = Σ_k w_k k⁻¹ t_{c(ν),ν}(z/k)``."""

    kind: ClassVar[str] = "outlier_t"

    def init(self, tmask: Array, prior_i: EqPrior) -> OutlierTEq:
        tp = prior_i.transitory
        log_p, log1m_p = self._p_init(tp)
        z = jnp.zeros(tmask.shape, _F64)
        return OutlierTEq(
            o_idx=jnp.zeros(tmask.shape, jnp.int32), log_p=log_p, log1m_p=log1m_p, log_tau=z,
            nu_idx=self._nu_init(tp), rb_out=z, acc_nu_move=jnp.zeros((), _F64),
        )  # fmt: skip

    def sample(
        self, key: Key, ts: OutlierTEq, log_e2: Array, log_var_pers: Array, tmask: Array,
        T_eff: Array, prior_i: EqPrior,
    ) -> OutlierTEq:  # fmt: skip
        tp = prior_i.transitory
        log_z2 = masked_log_z2(log_e2, log_var_pers, tmask)
        o_idx, pi = self.o_step(key, log_z2, tmask, ts.log_p, ts.log1m_p, ts.nu_idx)
        log_p, log1m_p = self.p_step(key, o_idx, tmask, T_eff, tp)
        log_r2 = jnp.where(tmask > 0, log_z2 - 2.0 * jnp.log(o_idx.astype(_F64) + 1.0), 0.0)
        nu_idx = self.nu_step(key, log_r2, tmask, T_eff, tp)
        log_tau = self.tau_step(key, log_r2, tmask, nu_idx)
        moved = (nu_idx != ts.nu_idx).astype(_F64)
        return OutlierTEq(
            o_idx=o_idx, log_p=log_p, log1m_p=log1m_p, log_tau=log_tau, nu_idx=nu_idx,
            rb_out=ts.rb_out + pi, acc_nu_move=ts.acc_nu_move + moved,
        )  # fmt: skip

    def log_scale2(self, ts: OutlierTEq) -> Array:
        return 2.0 * jnp.log(ts.o_idx.astype(_F64) + 1.0) + ts.log_tau

    def extract(self, ts: OutlierTEq) -> OutlierTDraw:
        return OutlierTDraw(log_p=ts.log_p, log1m_p=ts.log1m_p, nu=self.nu_value(ts.nu_idx))

    def ahead(self, key: Key, td: OutlierTDraw, H: int) -> Array:
        return self._o_ahead(key, td.log_p, H) + self._tau_ahead(key, td.nu, H)

    def log_density(self, log_z2: Array, td: OutlierTDraw) -> Array:
        return self._mix_log_density(
            log_z2, td.log_p, td.log1m_p, lambda lx: self._t_log_density(lx, td.nu)
        )

    def log_prior(self, ts: OutlierTEq, prior_i: EqPrior, tmask: Array) -> Array:
        tp = prior_i.transitory
        return self._outlier_log_prior(
            ts.o_idx, ts.log_p, ts.log1m_p, tp, tmask
        ) + self._t_log_prior(ts.log_tau, ts.nu_idx, tp, tmask)
