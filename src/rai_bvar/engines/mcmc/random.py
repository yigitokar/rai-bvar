"""Random-variate kernels of the MCMC engine (DESIGN §3.1 rule 2, D-4, D-8; N07 §9; pilot §5).

Elementwise and batched by shape or ``jax.vmap``; float64 throughout. Keys are typed threefry
keys (``keys.py``); a function that needs several streams derives them from its one key with
``fold_in`` (never reuses it), so it is one consumer in the sense of DESIGN §3.1 rule 5.

* :func:`uniform_tiny`: ``U[tiny, 1)``, for every uniform whose log is taken (N07 §9.1).
* :func:`categorical_inverse_cdf` (D-4, N07 §9.2 S3): one uniform per row, max-subtracted
  weights, ``≤`` comparison, clamp to ``J − 1``. Zero-weight categories are never chosen.
* :func:`loggamma`: ``log G``, ``G ~ Gamma(a, 1)``, by Marsaglia & Tsang (2000) with two
  pre-drawn candidates per element and a ``while_loop`` for the rare element that rejected
  both (the first accepted candidate of an i.i.d. sequence is the rejection sampler, so it is
  exact). For ``a < 1``: ``log G_a = log G_{a+1} + log(U)/a`` in log space, which stays finite at
  ``a = 1e-3`` where ``G`` itself underflows to 0 in half the draws (N07 §9.1). Pilot §5: 5.8×
  faster than ``jax.random.loggamma`` under ``vmap`` (one batched rejection loop over all
  elements).
* :func:`log_inv_gamma`, :func:`log_beta`: IG and Beta draws in log space from :func:`loggamma`.
* :func:`truncated_std_normal` (N07 §9.3 S6): log-space inverse CDF with Newton refinement,
  inverted in the tail where the quantile lies; accurate to rounding in the far tails, where
  ``jax.random.truncated_normal`` returns ``1.797e308``. ``log Φ`` is ``log_ndtr`` with
  ``series_order = 10``: the default order 3 of its ``x ≤ −20`` asymptotic series is off by
  ``4e-9`` at ``x = −20`` (``2e-11`` at ``−40``), order 10 by less than the rounding error.
* :func:`bernoulli_logit` (P3 §3.2, SSVS): ``γ = 1{log u < log σ(logit)}`` with one uniform per
  element, ``log σ(x) = −softplus(−x)``, so ``|logit|`` in the hundreds is exact.
* :func:`log_gig` (P3 D-42, §3.5): ``log X``, ``X ~ GIG(λ, ψ, χ)``, by Devroye (2014) §6 on the
  log scale, with the gamma and inverse-gamma limits exact; one problem, batched by ``vmap``.

Gamma is shape–rate ``Gamma(a, 1)``; ``IG(a, b)`` has density ``∝ x^{−a−1}e^{−b/x}``;
``GIG(λ, ψ, χ)`` has density ``∝ x^{λ−1}exp(−(ψx + χ/x)/2)`` on ``x > 0`` (P3 conventions).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import random as jr
from jax.scipy.special import log_ndtr, ndtri

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.keys import Key

Array = jax.Array

TINY = float(np.finfo(np.float64).tiny)
_LOG_HALF = float(np.log(0.5))
_LOG_SQRT_2PI = float(0.5 * np.log(2.0 * np.pi))
_MT_CANDIDATES = 2  # pre-drawn Marsaglia–Tsang candidates per element (pilot §5: fastest)
_MT_MAX_ROUNDS = 1000  # loop guard: P(an element needs it) < 0.05^1000; such elements return NaN
_NEWTON_STEPS = 4  # log-space inverse normal CDF (N07 §9.3)
_LOG_NDTR_ORDER = 10  # asymptotic-series order of log_ndtr for x ≤ −20 (the default 3 is ~4e-9 off)

GIG_MAX_ATTEMPTS = 512  # P3 D-42: P(ok = False) ≤ (1 − 1/3.46)^512 ≈ 1.6e-76 per call
_LOG_2 = float(np.log(2.0))
_LOG_4 = float(np.log(4.0))
_LOG_HALF_F = float(np.log(0.5))  # log ½ and log 2: the bounds of Devroye's design interval
_LOG_COSH1 = float(np.log(np.cosh(1.0)))
_LOG_COSH1_M1 = float(np.log(np.cosh(1.0) - 1.0))
_LOG_E_M2 = float(np.log(np.e - 2.0))
_ASINH_SWITCH = float(np.log(1e8))  # asinh(e^r) = log 2 + r to rounding beyond r = log 1e8
_LE_SERIES = 1e-3  # le(x) by its Taylor series below |x| = 1e-3
_LE_LARGE = 30.0  # le(x) = x + log1p(−(1 + x)e^{−x}) above x = 30

__all__ = [
    "GIG_MAX_ATTEMPTS",
    "TINY",
    "GIGHat",
    "bernoulli_logit",
    "bernoulli_logit_core",
    "categorical",
    "categorical_inverse_cdf",
    "categorical_inverse_cdf_lse",
    "gig_hat",
    "log_beta",
    "log_gig",
    "log_gig_attempts",
    "log_inv_gamma",
    "loggamma",
    "truncated_std_normal",
    "uniform_tiny",
]


def _shape(shape) -> tuple[int, ...]:
    return (int(shape),) if np.ndim(shape) == 0 else tuple(int(s) for s in shape)


# --------------------------------------------------------------------------- uniforms, categorical


def uniform_tiny(key: Key, shape=()) -> Array:
    """``u ~ U[tiny, 1)`` (float64), so ``log u ≥ −708.4`` is always finite (N07 §9.1)."""
    return jr.uniform(key, _shape(shape), dtype=jnp.float64, minval=TINY)


def categorical_inverse_cdf_lse(u: Array, logw: Array) -> tuple[Array, Array]:
    """:func:`categorical_inverse_cdf` plus the row ``logsumexp(logw)`` (``max + log Σ w``), which
    the exact SV correction reuses as ``log g`` (DESIGN §3.7.4, §10.4 lever 6)."""
    J = logw.shape[-1]
    mx = jnp.max(logw, axis=-1, keepdims=True)
    cw = jnp.cumsum(jnp.exp(logw - mx), axis=-1)
    tot = cw[..., -1]
    k = jnp.sum(cw <= (u * tot)[..., None], axis=-1)
    return jnp.minimum(k, J - 1).astype(jnp.int32), mx[..., 0] + jnp.log(tot)


def categorical_inverse_cdf(u: Array, logw: Array) -> Array:
    """Category ``k = min(#{j : cw_j ≤ u·cw_J}, J − 1)`` (int32), ``cw = cumsum(exp(logw − max))``
    (D-4; N07 §9.2 S3). ``u ∈ [0, 1)`` has the batch shape of ``logw[..., 0]``; ``logw`` holds
    unnormalised log weights (``−inf`` allowed, at least one finite per row). The largest weight
    is exactly 1, so the total is ``≥ 1``: shifts by ±700 are harmless."""
    return categorical_inverse_cdf_lse(u, logw)[0]


def categorical(key: Key, logw: Array) -> Array:
    """One draw per row of ``logw`` (``..., J``) with one float64 uniform per row (D-4)."""
    u = jr.uniform(key, logw.shape[:-1], dtype=jnp.float64)
    return categorical_inverse_cdf(u, logw)


# --------------------------------------------------------------------------- gamma family


def _mt_candidate(key: Key, d: Array, c: Array) -> tuple[Array, Array]:
    """One Marsaglia–Tsang candidate per element: ``(accepted, log v)`` with ``v = (1 + cx)³``;
    accept iff ``v > 0`` and ``log u < x²/2 + d − dv + d log v`` (``u ≥ tiny``)."""
    x = jr.normal(jr.fold_in(key, 0), d.shape, dtype=jnp.float64)
    u = uniform_tiny(jr.fold_in(key, 1), d.shape)
    t = 1.0 + c * x
    pos = t > 0.0
    logv = 3.0 * jnp.log(jnp.where(pos, t, 1.0))
    v = jnp.exp(logv)
    ok = pos & (jnp.log(u) < 0.5 * x * x + d - d * v + d * logv)
    return ok, logv


def loggamma(key: Key, a: Array, shape=None) -> Array:
    """``log G`` with ``G ~ Gamma(a, 1)`` (float64), exact for every ``a > 0``.

    ``shape`` defaults to ``a.shape`` (``a`` broadcasts to it). Marsaglia & Tsang (2000) on
    ``a' = a + 1{a < 1}``: ``d = a' − 1/3``, ``c = 1/√(9d)``, ``log G_{a'} = log d + log v`` at the
    first accepted candidate; then ``log G_a = log G_{a'} + log(U)/a`` when ``a < 1``.
    Non-finite or non-positive ``a`` gives NaN (never an endless loop).
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    shp = a.shape if shape is None else _shape(shape)
    a = jnp.broadcast_to(a, shp)
    valid = jnp.isfinite(a) & (a > 0.0)
    a_safe = jnp.where(valid, a, 1.0)
    boost = a_safe < 1.0
    a1 = jnp.where(boost, a_safe + 1.0, a_safe)
    d = a1 - 1.0 / 3.0
    c = 1.0 / jnp.sqrt(9.0 * d)
    k_cand, k_boost = jr.fold_in(key, 0), jr.fold_in(key, 1)

    oks, logvs = jax.vmap(lambda m: _mt_candidate(jr.fold_in(k_cand, m), d, c))(
        jnp.arange(_MT_CANDIDATES, dtype=jnp.uint32)
    )
    done = jnp.any(oks, axis=0)
    logv = jnp.take_along_axis(logvs, jnp.argmax(oks, axis=0)[None], axis=0)[0]

    def cond(s):
        it, dn, _ = s
        return (it < _MT_MAX_ROUNDS) & ~jnp.all(dn)

    def body(s):
        it, dn, lv = s
        ok, lv_new = _mt_candidate(jr.fold_in(k_cand, it + _MT_CANDIDATES), d, c)
        return it + 1, dn | ok, jnp.where(~dn & ok, lv_new, lv)

    _, done, logv = lax.while_loop(cond, body, (jnp.uint32(0), done | ~valid, logv))
    out = jnp.log(d) + logv
    log_u = jnp.log(uniform_tiny(k_boost, shp))
    out = jnp.where(boost, out + log_u / a_safe, out)
    return jnp.where(valid & done, out, jnp.nan)


def log_inv_gamma(key: Key, shape: Array, scale: Array) -> Array:
    """``log X``, ``X ~ IG(shape, scale)``: ``log(scale) − loggamma(shape)`` (N07 §9.1), on the
    broadcast shape of the arguments."""
    shape = jnp.asarray(shape, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    out_shape = jnp.broadcast_shapes(shape.shape, scale.shape)
    return jnp.log(scale) - loggamma(key, shape, out_shape)


def log_beta(key: Key, a: Array, b: Array) -> tuple[Array, Array]:
    """``(log p, log(1 − p))``, ``p = G_a/(G_a + G_b) ~ Beta(a, b)``, from two log-gamma draws:
    with ``δ = log G_b − log G_a``, ``log p = −softplus(δ)`` and ``log(1 − p) = −softplus(−δ)``.
    Stable for ``a, b ≪ 1`` (``p`` within rounding of 0 or 1), and ``p + (1 − p) = 1`` to rounding
    whatever the size of the log-gamma draws."""
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    out_shape = jnp.broadcast_shapes(a.shape, b.shape)
    la = loggamma(jr.fold_in(key, 0), a, out_shape)
    lb = loggamma(jr.fold_in(key, 1), b, out_shape)
    delta = lb - la
    return -jnp.logaddexp(0.0, delta), -jnp.logaddexp(0.0, -delta)


# --------------------------------------------------------------------------- truncated normal (S6)


def _log_ndtr(x: Array) -> Array:
    """``log Φ(x)`` to rounding for every ``x`` (``log_ndtr`` with ``series_order = 10``)."""
    return log_ndtr(x, series_order=_LOG_NDTR_ORDER)


def _log_ndtri(lp: Array) -> Array:
    """``x`` with ``log Φ(x) = lp`` for ``lp ≤ log ½``: start at ``ndtri(e^lp)`` when ``e^lp`` is
    representable, else at the Mills-ratio asymptote ``−(t − (log t + log √2π)/t)``,
    ``t = √(−2 lp)``; then Newton on ``log Φ(x) − lp`` (derivative ``φ(x)/Φ(x)``)."""
    t = jnp.sqrt(-2.0 * lp)
    x_asym = -(t - (jnp.log(t) + _LOG_SQRT_2PI) / t)
    p = jnp.exp(lp)
    x = jnp.where(p > 1e-300, ndtri(jnp.clip(p, 1e-300, 0.5)), x_asym)
    for _ in range(_NEWTON_STEPS):
        lphi = -0.5 * x * x - _LOG_SQRT_2PI
        lcdf = _log_ndtr(x)
        x = x - (lcdf - lp) * jnp.exp(lcdf - lphi)
    return x


def truncated_std_normal(u: Array, lo: Array, hi: Array) -> Array:
    """The ``u``-quantile of ``N(0, 1)`` restricted to ``(lo, hi)`` (N07 §9.3 S6; pure core,
    increasing in ``u``; ``u ∈ (0, 1)``, e.g. from :func:`uniform_tiny`; infinite bounds allowed).

    Both tail targets are formed in log space with positive terms only:
    ``log Φ(x) = log Φ(hi) + logaddexp(log u, log(1 − u) + log Φ(lo) − log Φ(hi))`` and
    ``log Φ(−x) = log Φ(−lo) + logaddexp(log(1 − u), log u + log Φ(−hi) − log Φ(−lo))`` (no
    ratio of masses is exponentiated, so none underflows to a flushed subnormal). The quantile is
    inverted in whichever tail it lies (``x = Φ⁻¹`` of the first when ``Φ(x) < ½``, else ``−Φ⁻¹``
    of the second) by :func:`_log_ndtri`, so it is accurate in both far tails and for intervals
    that straddle zero (S6 reflects whole intervals instead, which loses the near tail of
    ``(−40, 50)``-type intervals). The result is clipped to ``[lo, hi]``.
    """
    u, lo, hi = jnp.broadcast_arrays(*(jnp.asarray(v, dtype=jnp.float64) for v in (u, lo, hi)))
    la, lb = _log_ndtr(lo), _log_ndtr(hi)  # log Φ(lo) ≤ log Φ(hi)
    qa, qb = _log_ndtr(-lo), _log_ndtr(-hi)  # log Φ(−lo) ≥ log Φ(−hi)
    log_u, log_1mu = jnp.log(u), jnp.log1p(-u)
    lp = lb + jnp.logaddexp(log_u, log_1mu + (la - lb))
    lq = qa + jnp.logaddexp(log_1mu, log_u + (qb - qa))
    lower = lp < _LOG_HALF
    y = _log_ndtri(jnp.minimum(jnp.where(lower, lp, lq), _LOG_HALF))
    return jnp.clip(jnp.where(lower, y, -y), lo, hi)


# --------------------------------------------------------------------------- Bernoulli (P3 §3.2)


def bernoulli_logit_core(u: Array, logit: Array) -> Array:
    """``γ = 1{log u < log σ(logit)}`` (int32; pure core, ``u ∈ (0, 1)``, e.g. from
    :func:`uniform_tiny`), so ``P(γ = 1) = σ(logit)``. ``log σ(x) = −softplus(−x)`` is formed in
    log space, so a logit of ±700 is exact (``σ`` is never exponentiated); ``+inf`` gives 1 and
    ``−inf`` gives 0."""
    u, logit = jnp.broadcast_arrays(*(jnp.asarray(v, dtype=jnp.float64) for v in (u, logit)))
    return (jnp.log(u) < -jnp.logaddexp(0.0, -logit)).astype(jnp.int32)


def bernoulli_logit(key: Key, logit: Array) -> Array:
    """One Bernoulli draw per element of ``logit`` (int32 0/1) with one float64 uniform per
    element (P3 §3.2: the SSVS indicator ``γ_j``)."""
    logit = jnp.asarray(logit, dtype=jnp.float64)
    return bernoulli_logit_core(uniform_tiny(key, logit.shape), logit)


# --------------------------------------------------------------------------- GIG (P3 D-42, §3.5)


class GIGHat(NamedTuple):
    """Devroye's (2014, §4, §6) three-piece hat of the transformed density ``exp(f)`` (P3 §3.5
    step 3), for ``λ ≥ 0``. ``f(x) = −α(cosh x − 1) − λ(eˣ − x − 1)``; flat on ``[−s', t']``,
    ``exp(−η_D − ζ(x − t))`` right of ``t'``, ``exp(−θ + ξ(x + s))`` left of ``−s'``; masses
    ``p = 1/ξ`` (left), ``q = t' + s'`` (centre), ``r = 1/ζ`` (right)."""

    lam: Array  # λ ≥ 0
    log_lam: Array  # log λ (−inf at λ = 0)
    log_alpha: Array  # log α, α = √(ω² + λ²) − λ
    mode: Array  # asinh(λ/ω): log y = mode + X
    t: Array
    s: Array
    eta: Array  # −f(t)
    zeta: Array  # −f'(t)
    theta: Array  # −f(−s)
    xi: Array  # f'(−s)
    tp: Array  # t' = t − rη_D
    sp: Array  # s' = s − pθ
    p: Array
    q: Array
    r: Array


def _lsh(y: Array) -> Array:
    """``log sinh y`` for ``y ≥ 0`` (``−inf`` at 0): ``y − log 2 + log(−expm1(−2y))``."""
    return y - _LOG_2 + jnp.log(-jnp.expm1(-2.0 * y))


def _le(x: Array) -> Array:
    """``log(eˣ − x − 1)`` (``−inf`` at 0): the series ``log(x²/2 + x³/6 + x⁴/24)`` for
    ``|x| < 1e-3``, ``x + log1p(−(1 + x)e^{−x})`` for ``x > 30``, ``log(expm1(x) − x)`` otherwise
    (P3 §3.5). Branches are selected elementwise; an unselected branch may be non-finite."""
    x2 = x * x
    series = jnp.log(x2 * (0.5 + x / 6.0 + x2 / 24.0))
    large = x + jnp.log1p(-(1.0 + x) * jnp.exp(-jnp.maximum(x, _LE_LARGE)))
    mid = jnp.log(jnp.expm1(jnp.minimum(x, _LE_LARGE)) - x)
    return jnp.where(jnp.abs(x) < _LE_SERIES, series, jnp.where(x > _LE_LARGE, large, mid))


def _gig_log_neg_f(x: Array, log_alpha: Array, log_lam: Array, lam_pos: Array) -> Array:
    """``log(−f(x)) = log(α(cosh x − 1) + λ(eˣ − x − 1))`` in log space (P3 §3.5):
    ``α(cosh x − 1) = exp(log α + log 2 + 2 lsh(|x|/2))``, ``λ(eˣ − x − 1) = exp(log λ + le(x))``,
    the ``λ`` term only when ``λ > 0``. Never forms a product of an underflowed and an overflowed
    factor; a result above ``log(max float)`` makes ``f = −∞``."""
    a_term = log_alpha + _LOG_2 + 2.0 * _lsh(0.5 * jnp.abs(x))
    l_term = jnp.where(lam_pos, log_lam + _le(x), -jnp.inf)
    return jnp.logaddexp(a_term, l_term)


def gig_hat(lam: Array, log_omega: Array) -> GIGHat:
    """The hat of Devroye (2014) §6 for ``λ ≥ 0``, ``ω = exp(log ω) > 0``, set up entirely in logs
    (P3 §3.5 step 3; E89):

    * ``Λ = logaddexp(½·logaddexp(2 log ω, 2 log λ), log λ) = log(√(ω² + λ²) + λ)``,
      ``log α = 2 log ω − Λ``; ``asinh(λ/ω) = log 2 + log λ − log ω`` beyond ``log 1e8``.
    * ``t``: 1 if ``−f(1) ∈ [½, 2]``, ``√(2/(α + λ))`` if above, ``log 4 − logaddexp(log α,
      log 2 + log λ)`` if below; ``s``: 1 if ``−f(−1) ∈ [½, 2]``, ``√(4/(α cosh 1 + λ))`` if
      above, ``min(1/λ, logaddexp(0, log1p(√(1 + 2α)) − log α))`` if below (``1/λ = ∞`` at 0).
    * ``(η_D, ζ, θ, ξ) = (−f(t), −f'(t), −f(−s), f'(−s))``, each from its log.
    """
    lam = jnp.asarray(lam, dtype=jnp.float64)
    log_omega = jnp.asarray(log_omega, dtype=jnp.float64)
    lam_pos = lam > 0.0
    log_lam = jnp.log(lam)  # −inf at 0
    lam_term = jnp.where(lam_pos, log_lam, -jnp.inf)
    Lam = jnp.logaddexp(0.5 * jnp.logaddexp(2.0 * log_omega, 2.0 * lam_term), lam_term)
    log_alpha = 2.0 * log_omega - Lam
    ratio = lam_term - log_omega
    mode = jnp.where(
        ratio > _ASINH_SWITCH,
        _LOG_2 + ratio,
        jnp.arcsinh(jnp.exp(jnp.minimum(ratio, _ASINH_SWITCH))),
    )

    # t from −f(1) = α(cosh 1 − 1) + λ(e − 2); s from −f(−1) = α(cosh 1 − 1) + λ/e (in logs)
    l_f1 = jnp.logaddexp(log_alpha + _LOG_COSH1_M1, lam_term + _LOG_E_M2)
    l_fm1 = jnp.logaddexp(log_alpha + _LOG_COSH1_M1, lam_term - 1.0)
    t_hi = jnp.exp(0.5 * (_LOG_2 - jnp.logaddexp(log_alpha, lam_term)))
    t_lo = _LOG_4 - jnp.logaddexp(log_alpha, _LOG_2 + lam_term)
    t = jnp.where(l_f1 > _LOG_2, t_hi, jnp.where(l_f1 < _LOG_HALF_F, t_lo, 1.0))
    s_hi = jnp.exp(0.5 * (_LOG_4 - jnp.logaddexp(log_alpha + _LOG_COSH1, lam_term)))
    alpha_lo = jnp.exp(jnp.minimum(log_alpha, 0.0))  # the s_lo branch needs −f(−1) < ½: α < 1
    s_lo = jnp.minimum(
        jnp.exp(-lam_term),
        jnp.logaddexp(0.0, jnp.log1p(jnp.sqrt(1.0 + 2.0 * alpha_lo)) - log_alpha),
    )
    s = jnp.where(l_fm1 > _LOG_2, s_hi, jnp.where(l_fm1 < _LOG_HALF_F, s_lo, 1.0))

    # the tangents at t and −s: η_D = −f(t), ζ = α sinh t + λ expm1(t), θ = −f(−s),
    # ξ = α sinh s + λ(−expm1(−s))
    log_eta = _gig_log_neg_f(t, log_alpha, log_lam, lam_pos)
    log_zeta = jnp.logaddexp(
        log_alpha + _lsh(t), jnp.where(lam_pos, log_lam + jnp.log(jnp.expm1(t)), -jnp.inf)
    )
    log_theta = _gig_log_neg_f(-s, log_alpha, log_lam, lam_pos)
    log_xi = jnp.logaddexp(
        log_alpha + _lsh(s), jnp.where(lam_pos, log_lam + jnp.log(-jnp.expm1(-s)), -jnp.inf)
    )
    p = jnp.exp(-log_xi)
    r = jnp.exp(-log_zeta)
    tp = t - jnp.exp(log_eta - log_zeta)
    sp = s - jnp.exp(log_theta - log_xi)
    return GIGHat(
        lam=lam, log_lam=log_lam, log_alpha=log_alpha, mode=mode, t=t, s=s,
        eta=jnp.exp(log_eta), zeta=jnp.exp(log_zeta), theta=jnp.exp(log_theta),
        xi=jnp.exp(log_xi), tp=tp, sp=sp, p=p, q=tp + sp, r=r,
    )  # fmt: skip


def _gig_candidate(h: GIGHat, u: Array) -> tuple[Array, Array]:
    """One rejection attempt with ``u = (U, V, W)`` uniform on ``(0, 1]``: ``(X, accepted)``.
    Centre with probability ``q/(p + q + r)`` (``X = −s' + qV``), right with ``r/(p + q + r)``
    (``X = t' − r log V``), left otherwise (``X = −s' + p log V``); accept iff
    ``log W ≤ f(X) − log hat(X)``."""
    U, V, W = u[0], u[1], u[2]
    tot_u = U * (h.p + h.q + h.r)
    log_v = jnp.log(V)
    centre = tot_u < h.q
    right = ~centre & (tot_u < h.q + h.r)
    X = jnp.where(
        centre, -h.sp + h.q * V, jnp.where(right, h.tp - h.r * log_v, -h.sp + h.p * log_v)
    )
    log_hat = jnp.where(
        centre,
        0.0,
        jnp.where(right, -h.eta - h.zeta * (X - h.t), -h.theta + h.xi * (X + h.s)),
    )
    f = -jnp.exp(_gig_log_neg_f(X, h.log_alpha, h.log_lam, h.lam > 0.0))
    return X, jnp.log(W) <= f - log_hat


def log_gig_attempts(
    key: Key, lam: Array, log_psi: Array, log_chi: Array, *, hat=gig_hat
) -> tuple[Array, Array, Array]:
    """:func:`log_gig` plus the number of rejection attempts it used (int32; 0 for the limits
    and for improper input). ``hat`` builds the envelope from ``(|λ|, log ω)`` (the Devroye
    hat by default; a test seam for mutant M34, P3 §9.6)."""
    lam = jnp.asarray(lam, dtype=jnp.float64)
    log_psi = jnp.asarray(log_psi, dtype=jnp.float64)
    log_chi = jnp.asarray(log_chi, dtype=jnp.float64)
    fin_lam = jnp.isfinite(lam)
    regular = fin_lam & jnp.isfinite(log_psi) & jnp.isfinite(log_chi)
    gamma_lim = fin_lam & (lam > 0.0) & (log_chi == -jnp.inf) & jnp.isfinite(log_psi)
    invg_lim = fin_lam & (lam < 0.0) & (log_psi == -jnp.inf) & jnp.isfinite(log_chi)

    # 1. limits: χ = 0, λ > 0: log x = LG(λ) − log(ψ/2); ψ = 0, λ < 0: log x = log(χ/2) − LG(−λ)
    a = jnp.where(gamma_lim, lam, jnp.where(invg_lim, -lam, 1.0))
    lg = loggamma(jr.fold_in(key, GIG_MAX_ATTEMPTS), a)
    log_lim = jnp.where(gamma_lim, lg - (log_psi - _LOG_2), (log_chi - _LOG_2) - lg)

    # 2. reduction: log η = ½(log χ − log ψ), log ω = ½(log ψ + log χ); |λ| and the sign
    log_eta = 0.5 * (log_chi - log_psi)
    log_omega = jnp.where(regular, 0.5 * (log_psi + log_chi), 0.0)
    sign = jnp.where(lam < 0.0, -1.0, 1.0)
    h = hat(jnp.where(regular, jnp.abs(lam), 1.0), log_omega)

    # 3. rejection: attempt t uses fold_in(key, t); irregular lanes start done (no attempts)
    def cond(st):
        it, done, _ = st
        return (it < GIG_MAX_ATTEMPTS) & ~done

    def body(st):
        it, done, x = st
        u = 1.0 - jr.uniform(jr.fold_in(key, it), (3,), dtype=jnp.float64)  # (0, 1]
        X, acc = _gig_candidate(h, u)
        return it + 1, done | acc, jnp.where(done, x, X)

    it, done, X = lax.while_loop(
        cond, body, (jnp.zeros((), jnp.int32), ~regular, jnp.zeros((), jnp.float64))
    )
    # 4. log x = log η + sign·log y,  log y = asinh(λ/ω) + X
    log_dev = log_eta + sign * (h.mode + X)
    ok_dev = regular & done & jnp.isfinite(log_dev)
    ok = ok_dev | gamma_lim | invg_lim
    out = jnp.where(regular, log_dev, log_lim)
    return jnp.where(ok & jnp.isfinite(out), out, jnp.nan), ok & jnp.isfinite(out), it


def log_gig(key: Key, lam: Array, log_psi: Array, log_chi: Array) -> tuple[Array, Array]:
    """``(log x, ok)`` for ``x ~ GIG(λ, ψ, χ)`` (density ``∝ x^{λ−1}exp(−(ψx + χ/x)/2)``) from
    ``λ``, ``log ψ`` and ``log χ`` (P3 D-42, §3.5): one problem, batched by ``vmap``; float64.

    1. **Limits.** ``log χ = −∞``, ``λ > 0``: ``log x = LG(λ) − log(ψ/2)`` (gamma);
       ``log ψ = −∞``, ``λ < 0``: ``log x = log(χ/2) − LG(−λ)`` (inverse gamma), with
       :func:`loggamma` (exact for tiny shapes) on the stream ``fold_in(key, 512)``. Any other
       infinite or NaN input is improper: ``ok = False`` and ``log x = NaN`` (P2 D-18).
    2. **Reduction.** ``log η = ½(log χ − log ψ)``, ``log ω = ½(log ψ + log χ)``; ``y = x/η`` has
       density ``∝ y^{λ−1}exp(−ω(y + 1/y)/2)``; for ``λ < 0`` the draw uses ``|λ|`` and
       ``sign = −1`` (``1/Y ~ GIG(−λ, ω)``), applied once in step 4.
    3. **Devroye (2014) §6** for ``λ ≥ 0`` (:func:`gig_hat`): ``log y = asinh(λ/ω) + X``,
       ``X ∝ exp(f(x))``, ``f(x) = −α(cosh x − 1) − λ(eˣ − x − 1)``, by rejection from the
       three-piece hat; every term in logs. Attempt ``t`` draws ``(U, V, W)`` uniform on
       ``(0, 1]`` from ``fold_in(key, t)``; a ``lax.while_loop`` runs until every lane accepts or
       512 attempts pass (then ``ok = False``: probability ≤ 1.6e-76 per call, by Theorem 2's
       uniform bound of 3.4597 expected attempts).
    4. ``log x = log η + sign·log y``.
    """
    log_x, ok, _ = log_gig_attempts(key, lam, log_psi, log_chi)
    return log_x, ok
