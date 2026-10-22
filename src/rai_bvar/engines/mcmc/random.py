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

Gamma is shape–rate ``Gamma(a, 1)``; ``IG(a, b)`` has density ``∝ x^{−a−1}e^{−b/x}``.
"""

from __future__ import annotations

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

__all__ = [
    "TINY",
    "categorical",
    "categorical_inverse_cdf",
    "categorical_inverse_cdf_lse",
    "log_beta",
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
