"""Independent reference of the SOC/SUR dummy rows of the coefficient prior (DESIGN §9.1, §13 WP25;
N05 §7.7; N04 §3.1). Two-implementation rule (D-22): written from the model, never from the
kernels, and in a different algebra — **covariance form** (``m × m`` systems, Matheron's rule and
Bayes' rule), where the kernels use the precision form (``q × q`` Gram updates).

**Model.** The dummy rows ``(Y_d, X_d)`` (``SharedPrior.dummies``, ``m`` rows) are pseudo-observations
that update the base prior of the coefficients (Phase 1 convention: the analytic engine's prior is
the NIW posterior after the rows; its marginal likelihood is ``ML(rows + data) − ML(rows)``):

* structural equation ``i`` (δ-coordinates ``θ_i = [δ_i | a_i]``, N05 §4.1, §7.7): target
  ``y_d = Y_d[:, i]``, design ``F_d = [X_d | X_d M_c − Y_d[:, :n−1]]`` on the active slots, noise
  variance ``1/w_i`` (Mode I, ``iw_equivalent``) or ``λ_i/w_i`` (Mode II);
* reduced form, equation ``j`` of ``Π`` (N04 §3.1): ``y_d = Y_d[:, j] = X_d π_j + ε``,
  ``ε ~ N(0, I/w_j)``.

With the base prior ``θ ~ N(μ₀, V₀)`` (``V₀`` diagonal; ``λ_i``-scaled blocks where static) and the
rows' noise variance ``R``, the prior *with* the rows is the conditional law given the rows:

    p(θ, λ | d) = p(λ) N(θ; μ₀, V₀) N(y_d; F_d θ, R I) / p(d)          (Bayes' rule)

* Mode I and reduced form (``λ``-free ``V₀``, ``R``): ``p(d) = N(y_d; F_d μ₀, F_d V₀ F_dᵀ + R I)``.
* Mode II (``V₀ = λ Ṽ₀``, ``R = λ/w``): ``p(d | λ) = N(y_d; F_d μ₀, λ S̃)``, ``S̃ = F_d Ṽ₀ F_dᵀ + I/w``,
  so ``λ | d ~ IG(c + m/2, C + ½ r'S̃⁻¹r)`` with ``r = y_d − F_d μ₀`` and
  ``p(d) = Γ(c + m/2)/Γ(c) · C^c/(C + ½ r'S̃⁻¹r)^{c+m/2} · (2π)^{−m/2} |S̃|^{−½}``.
* ``iw_equivalent`` (only ``a`` λ-scaled, rows with fixed weight ``w``): ``p(d | λ) = N(r; 0,
  S(λ))`` with ``S(λ) = S₀ + λUUᵀ``, ``S₀ = F_δV_δF_δᵀ + I/w``, ``U = F_aṼ_a^{½}``, which is not of IG
  form: ``log p(d)`` has no closed form, so the log prior is exact **up to a constant** (enough for
  every density-ratio identity and every Geweke/SBC statistic). ``λ | d`` is drawn exactly by
  rejection from the base prior ``IG(c, C)`` with acceptance probability

      g(λ) = (|S₀|/|S(λ)|)^{½} exp(−½(q(λ) − q∞)) ≤ 1,   q(λ) = r'S(λ)⁻¹r,
      q∞ = lim_{λ→∞} q(λ) = min_u (r − Uu)'S₀⁻¹(r − Uu)

  (``|S(λ)| ≥ |S₀|`` and ``q(λ) ≥ q∞`` because ``S(λ) ⪰ S₀`` grows with λ; ``p(λ | d) ∝ IG(λ; c,
  C) g(λ)``). Proposals come in batches; the first accepted proposal of a batch is the sequential
  rejection draw, so the sampler is exact.

Draws use **Matheron's rule** (exact for Gaussian conditioning): ``θ₀ ~ N(μ₀, V₀)``, ``ε ~ N(0, R I)``,
``θ = θ₀ + V₀F_dᵀ(F_d V₀ F_dᵀ + R I)⁻¹(y_d − F_d θ₀ − ε)``; Mode II and ``iw_equivalent`` first draw
``λ | d``.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cho_solve
from jax.scipy.special import gammaln

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    ConstantEq,
    EqPrior,
    LinearEq,
    PriorArrays,
    RecursiveEq,
    SharedPrior,
)

Array = jax.Array
_F64 = jnp.float64
LOG_2PI = math.log(2.0 * math.pi)

__all__ = [
    "has_dummies",
    "log_rows_evidence",
    "log_rows_likelihood",
    "reduced_dummy_log_factor",
    "reduced_dummy_prior_sample",
    "rows_condition",
    "structural_dummy_log_factor_eq",
    "structural_dummy_prior_sample_eq",
    "structural_rows_eq",
]


def has_dummies(static: StaticConfig) -> bool:
    """The composition has SOC/SUR dummy rows (``MeanStatic.n_dummy > 0``)."""
    return static.mean.n_dummy > 0


def _mode(static: StaticConfig) -> str:
    if static.mean.coef_scaling == "lambda":
        return "II"
    return "iw_equivalent" if static.struct.contemp_scaling == "lambda" else "I"


# --------------------------------------------------------------------------- Gaussian algebra


def _chol_rows(F: Array, v0: Array, r_var: Array) -> Array:
    """Lower Cholesky factor of ``S = F diag(v0) Fᵀ + r_var I`` (``m × m``)."""
    S = (F * v0[None, :]) @ F.T + r_var * jnp.eye(F.shape[0], dtype=_F64)
    return jnp.linalg.cholesky(0.5 * (S + S.T))


def log_rows_likelihood(F: Array, y: Array, theta: Array, r_var: Array) -> Array:
    """``Σ_d log N(y_d; F_d θ, r_var)``."""
    r = y - F @ theta
    return -0.5 * jnp.sum(LOG_2PI + jnp.log(r_var) + r * r / r_var)


def _quad_logdet(F: Array, y: Array, mu0: Array, v0: Array, r_var: Array) -> tuple[Array, Array]:
    """``(r'S⁻¹r, log|S|)`` with ``S = F diag(v0) Fᵀ + r_var I`` and ``r = y − Fμ₀``."""
    L = _chol_rows(F, v0, r_var)
    r = y - F @ mu0
    u = jax.scipy.linalg.solve_triangular(L, r, lower=True)
    return u @ u, 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def log_rows_evidence(F: Array, y: Array, mu0: Array, v0: Array, r_var: Array | float) -> Array:
    """``log N(y; Fμ₀, F diag(v0) Fᵀ + r_var I)``: the rows' prior predictive (λ-free case)."""
    quad, logdet = _quad_logdet(F, y, mu0, v0, jnp.asarray(r_var, _F64))
    return -0.5 * (F.shape[0] * LOG_2PI + logdet + quad)


def _log_rows_evidence_ig(
    F: Array, y: Array, mu0: Array, v0_tilde: Array, w: Array, shape: Array, scale: Array
) -> Array:
    """``log ∫ IG(λ; c, C) N(y; Fμ₀, λ S̃) dλ`` with ``S̃ = F Ṽ₀ Fᵀ + I/w`` (Mode II evidence)."""
    m = F.shape[0]
    quad, logdet = _quad_logdet(F, y, mu0, v0_tilde, 1.0 / w)
    a = shape + 0.5 * m
    return (
        gammaln(a) - gammaln(shape) + shape * jnp.log(scale) - a * jnp.log(scale + 0.5 * quad)
        - 0.5 * m * LOG_2PI - 0.5 * logdet
    )  # fmt: skip


def rows_condition(
    key: Array, F: Array, y: Array, mu0: Array, v0: Array, r_var: Array | float
) -> Array:
    """One draw of ``θ ~ N(μ₀, diag v0)`` conditioned on ``y = Fθ + ε``, ``ε ~ N(0, r_var I)``
    (Matheron's rule; coordinates with ``v0 = 0`` stay at ``μ₀``)."""
    r_var = jnp.asarray(r_var, _F64)
    th0 = mu0 + jnp.sqrt(v0) * random.normal(random.fold_in(key, 0), mu0.shape, dtype=_F64)
    eps = jnp.sqrt(r_var) * random.normal(random.fold_in(key, 1), y.shape, dtype=_F64)
    L = _chol_rows(F, v0, r_var)
    u = cho_solve((L, True), y - F @ th0 - eps)
    return th0 + v0 * (F.T @ u)


# --------------------------------------------------------------------------- structural form


def structural_rows_eq(
    i: Array, prior_i: EqPrior, shared: SharedPrior, static: StaticConfig, log_lam: Array | float
) -> dict[str, Array]:
    """Equation ``i``'s base prior and rows on ``θ_i = [δ_i | a_i]`` (inactive ``a``-slots: zero
    columns, zero variance): ``mu0``, ``v0`` (with λ where static), ``v0_tilde`` (λ-free), ``F``,
    ``y``, the λ-free weight ``w`` and the rows' noise variance ``r_var``."""
    n = static.shapes.n
    c, ct = prior_i.coef, prior_i.contemp
    mask_a = ct.cmask
    lam = jnp.exp(jnp.asarray(log_lam, _F64))
    var_d = 1.0 / c.prec0
    var_a = jnp.where(mask_a > 0, 1.0 / ct.prec0, 0.0)
    mode = _mode(static)
    v0_tilde = jnp.concatenate([var_d, var_a])
    if mode == "II":
        v0 = v0_tilde * lam
    elif mode == "iw_equivalent":
        v0 = jnp.concatenate([var_d, var_a * lam])
    else:
        v0 = v0_tilde
    Yd, Xd = shared.dummies.Y, shared.dummies.X
    Fa = (Xd @ shared.Mc - Yd[:, : n - 1]) * mask_a[None, :]
    w = jnp.asarray(c.dummy_w, _F64)
    return {
        "mu0": jnp.concatenate([c.mean0, jnp.where(mask_a > 0, ct.mean0, 0.0)]),
        "v0": v0,
        "v0_tilde": v0_tilde,
        "F": jnp.concatenate([Xd, Fa], axis=1),
        "y": jnp.take(Yd, i, axis=1),
        "w": w,
        "r_var": lam / w if mode == "II" else 1.0 / w,
    }


def structural_dummy_prior_sample_eq(
    key: Array,
    i: Array,
    prior_i: EqPrior,
    shared: SharedPrior,
    static: StaticConfig,
    core: Any,
) -> tuple[Any, RecursiveEq, LinearEq]:
    """``(core, a_i, δ_i)`` from the prior with the rows (module docstring): Mode II redraws
    ``λ_i | d ~ IG(c + m/2, C + ½ r'S̃⁻¹r)`` and ``iw_equivalent`` draws ``λ_i | d`` by rejection
    (``core`` is replaced), then ``θ_i | λ_i, d`` by Matheron's rule; Mode I keeps ``core`` (λ does
    not enter the rows' law)."""
    mode = _mode(static)
    k = static.shapes.k
    if mode == "iw_equivalent":
        base = structural_rows_eq(i, prior_i, shared, static, 0.0)
        log_lam = _iweq_log_lambda(random.fold_in(key, 0), base, prior_i, k)
        core = ConstantEq(log_lam=log_lam)
    elif mode == "II":
        cp = prior_i.vol_core
        base = structural_rows_eq(i, prior_i, shared, static, 0.0)
        quad, _ = _quad_logdet(base["F"], base["y"], base["mu0"], base["v0_tilde"], 1.0 / base["w"])
        shape = cp.shape + 0.5 * static.mean.n_dummy
        log_lam = jnp.log(cp.scale + 0.5 * quad) - random.loggamma(
            random.fold_in(key, 0), shape, dtype=_F64
        )
        core = ConstantEq(log_lam=log_lam)
    else:
        log_lam = 0.0
    p = structural_rows_eq(i, prior_i, shared, static, log_lam)
    theta = rows_condition(random.fold_in(key, 1), p["F"], p["y"], p["mu0"], p["v0"], p["r_var"])
    mask_a = prior_i.contemp.cmask
    a = jnp.where(mask_a > 0, theta[k:], 0.0)
    return core, RecursiveEq(a=a), LinearEq(delta=theta[:k])


_REJECT_BATCH = 64  # rejection proposals per while-loop iteration (iw_equivalent λ | d)


def _iweq_log_lambda(key: Array, base: dict[str, Array], prior_i: EqPrior, k: int) -> Array:
    """``log λ`` with ``λ ~ p(λ | d) ∝ IG(λ; c, C)·N(r; 0, S₀ + λUUᵀ)`` by exact rejection from
    ``IG(c, C)`` (module docstring). ``base`` is :func:`structural_rows_eq` at ``λ = 1`` (so its
    ``v0`` holds the λ-free variances)."""
    F, y, mu0, v0, w = base["F"], base["y"], base["mu0"], base["v0_tilde"], base["w"]
    m = F.shape[0]
    Fd, Fa = F[:, :k], F[:, k:]
    S0 = (Fd * v0[None, :k]) @ Fd.T + jnp.eye(m, dtype=_F64) / w
    L0 = jnp.linalg.cholesky(0.5 * (S0 + S0.T))
    Ut = jax.scipy.linalg.solve_triangular(L0, Fa * jnp.sqrt(v0[None, k:]), lower=True)  # L0⁻¹U
    rt = jax.scipy.linalg.solve_triangular(L0, y - F @ mu0, lower=True)  # L0⁻¹r
    coef = jnp.linalg.lstsq(Ut, rt)[0]
    q_inf = jnp.sum((rt - Ut @ coef) ** 2)  # min_u ‖r̃ − Ũu‖²
    eye = jnp.eye(m, dtype=_F64)

    def log_accept(lam: Array) -> Array:
        # S(λ) = L0(I + λŨŨᵀ)L0ᵀ: |S(λ)|/|S₀| = |I + λŨŨᵀ|, q(λ) = r̃ᵀ(I + λŨŨᵀ)⁻¹r̃
        Lm = jnp.linalg.cholesky(eye + lam * (Ut @ Ut.T))
        v = jax.scipy.linalg.solve_triangular(Lm, rt, lower=True)
        log_g = -jnp.sum(jnp.log(jnp.diag(Lm))) - 0.5 * (v @ v - q_inf)
        return jnp.minimum(log_g, 0.0)  # q(λ) ≥ q∞ exactly; clip rounding above 0

    cp = prior_i.vol_core

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        key, log_lam, done = carry
        key, k1, k2 = random.split(key, 3)
        props = jnp.log(cp.scale) - random.loggamma(k1, cp.shape, (_REJECT_BATCH,), dtype=_F64)
        log_u = jnp.log(random.uniform(k2, (_REJECT_BATCH,), dtype=_F64, minval=1e-300))
        acc = log_u < jax.vmap(lambda lp: log_accept(jnp.exp(lp)))(props)
        hit = jnp.any(acc)
        log_lam = jnp.where(done | ~hit, log_lam, props[jnp.argmax(acc)])
        return key, log_lam, done | hit

    _, log_lam, _ = jax.lax.while_loop(
        lambda c: ~c[2], body, (key, jnp.zeros((), _F64), jnp.zeros((), bool))
    )
    return log_lam


def structural_dummy_log_factor_eq(
    i: Array,
    delta: Array,
    a: Array,
    log_lam: Array | float,
    prior_i: EqPrior,
    shared: SharedPrior,
    static: StaticConfig,
) -> Array:
    """``log p(d | θ_i, λ_i) − log p(d)``: what the rows add to ``log p(λ_i) + log p(θ_i | λ_i)``
    (Bayes' rule); ``log p(d)`` is omitted (a constant) for ``iw_equivalent``."""
    p = structural_rows_eq(i, prior_i, shared, static, log_lam)
    theta = jnp.concatenate([delta, a])
    out = log_rows_likelihood(p["F"], p["y"], theta, p["r_var"])
    mode = _mode(static)
    if mode == "I":
        out = out - log_rows_evidence(p["F"], p["y"], p["mu0"], p["v0"], p["r_var"])
    elif mode == "II":
        cp = prior_i.vol_core
        out = out - _log_rows_evidence_ig(
            p["F"], p["y"], p["mu0"], p["v0_tilde"], p["w"], cp.shape, cp.scale
        )
    return out


# --------------------------------------------------------------------------- reduced form


def _reduced_parts(prior: PriorArrays) -> tuple[Array, Array, Array, Array, Array]:
    c = prior.eq.coef
    d = prior.shared.dummies
    return c.mean0, 1.0 / c.prec0, jnp.asarray(c.dummy_w, _F64), d.X, d.Y


def reduced_dummy_prior_sample(key: Array, prior: PriorArrays) -> CCCMGlobal:
    """``Π`` with ``π_j ~ N(m_j, diag v_j)`` conditioned on ``Y_d[:, j] = X_d π_j + ε``,
    ``ε ~ N(0, I/w_j)``, independently over ``j`` (N04 §3.1)."""
    mean0, var0, w, Xd, Yd = _reduced_parts(prior)
    n = mean0.shape[0]

    def one(j: Array, m_j: Array, v_j: Array, w_j: Array) -> Array:
        return rows_condition(random.fold_in(key, j), Xd, Yd[:, j], m_j, v_j, 1.0 / w_j)

    pis = jax.vmap(one)(jnp.arange(n, dtype=jnp.uint32), mean0, var0, w)
    return CCCMGlobal(Pi=pis.T)


def reduced_dummy_log_factor(Pi: Array, prior: PriorArrays) -> Array:
    """``Σ_j [log p(d_j | π_j) − log p(d_j)]`` (Bayes' rule, Gaussian evidence)."""
    mean0, var0, w, Xd, Yd = _reduced_parts(prior)

    def one(pi_j, y_j, m_j, v_j, w_j):
        r_var = 1.0 / w_j
        return log_rows_likelihood(Xd, y_j, pi_j, r_var) - log_rows_evidence(
            Xd, y_j, m_j, v_j, r_var
        )

    return jnp.sum(jax.vmap(one)(Pi.T, Yd.T, mean0, var0, w))
