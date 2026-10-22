"""Reference log densities: ``log p(θ)``, ``log p(y | θ)`` and the joint (DESIGN §9.2–§9.3).

Independent of the kernels (D-22; N06 §5.1 "requirement"): written as prior log densities plus
the likelihood, never assembled from a block's posterior formulas, so the density-ratio
identities (Grosse & Duvenaud 2014 eq. 1)

    log q_b(θ′_b | ·) − log q_b(θ_b | ·) = log p(θ′_b, θ_−b, y) − log p(θ_b, θ_−b, y)

are not circular. Coordinates and padding conventions are those of ``reference.py``: densities of
the natural parameters (λ, σ, φ, μ, h, p, τ; counting measure on ``o`` and ν); sums over data rows
masked with ``m_t`` *before* any ``log`` or ``exp`` of a padded entry (E23).

Likelihood (exact, given the latents; N05 §3, N04 §2; ``|A| = 1``, no Jacobian):

    log p(Y | θ) = Σ_i Σ_t m_t [−½ log(2π λ_eff,i,t) − ½ e_{i,t}² / λ_eff,i,t]
    e_t = A y_t − Γ x_t (structural, γ_i = δ_i + M_c a_i)  |  A(y_t − Π'x_t) (reduced)  |  A y_t (zero)

Augmented joint with mixture indicators ``s`` (``s`` given; N01 §8, §10.2; N02 §9), with
``y*_t = logaddexp(log ẽ²_t, log c)``, ``log ẽ² = log e² − 2 log o − log τ`` and ``ξ_t = y*_t − h_t``:

    exact=True  (DNP 2015 Alg. 3):  log p(θ) + log p(Y | θ) + Σ m_t [log p_s + log N(ξ_t; m_s, v_s) − log g(ξ_t)]
    exact=False (auxiliary model):  log p(θ) + Σ m_t [log p_s + log N(ξ_t; m_s, v_s)]

The first is the target of the exact sampler extended by ``q(s | ξ) = p_s N(ξ; m_s, v_s)/g(ξ)``;
the second is the density of ``(θ, s, y*)`` under the auxiliary model, with the Jacobian of
``e ↦ y*`` (free of ``s`` and ``h``) omitted. Use them for the ``s``- and ``h | s``-step
identities only; every other step uses the non-augmented joint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import ChainState, EngineData, EqPrior, EqState, PriorArrays
from rai_bvar.validation.reference import (
    LOG_2PI,
    coupling_log_scale,
    effective_log_var,
    log_mixture_components,
    log_mixture_density,
    mixture_table,
    references_for,
)
from rai_bvar.validation.simulate import structural_gamma, unit_lower

Array = jax.Array
_F64 = jnp.float64


def log_prior_eq(eq_i: EqState, prior_i: EqPrior, static: StaticConfig, tmask: Array) -> Array:
    """``log p`` of one equation's parameters and latents (all parts)."""
    refs = references_for(static)
    scale = coupling_log_scale(eq_i.vol.core, static)
    return (
        refs.core.log_prior_eq(eq_i.vol.core, prior_i, static, tmask)
        + refs.trans.log_prior_eq(eq_i.vol.trans, prior_i, static, tmask)
        + refs.structure.log_prior_eq(eq_i.struct, prior_i, static, tmask, log_scale=scale)
        + refs.mean.log_prior_eq(eq_i.mean, prior_i, static, tmask, log_scale=scale)
        + refs.coef.log_prior_eq(eq_i.coef, prior_i, static, tmask, log_scale=scale)
    )


def log_prior(st: ChainState, prior: PriorArrays, static: StaticConfig, tmask: Array) -> Array:
    """``log p(θ)``: every equation's parts plus the global parts (reduced-form ``Π``)."""
    refs = references_for(static)
    per_eq = jax.vmap(lambda e, p: log_prior_eq(e, p, static, tmask))(st.eq, prior.eq)
    glob = (
        refs.mean.log_prior_global(st.glob.mean, prior, static)
        + refs.coef.log_prior_global(st.glob.coef, prior, static)
        + refs.core.log_prior_global(st.glob.vol, prior, static)
        + refs.structure.log_prior_global(st.glob.struct, prior, static)
    )
    return jnp.sum(per_eq) + glob


def residuals(st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig) -> Array:
    """Structural residuals ``e`` (``n × T_pad``; 0 on padded rows)."""
    A = unit_lower(st.eq.struct.a)
    kind = static.mean.kind
    if kind == "linear_structural":
        U = data.Y @ A.T - data.X @ structural_gamma(st, prior).T
    elif kind == "linear_reduced":
        U = (data.Y - data.X @ st.glob.mean.Pi) @ A.T
    else:
        U = data.Y @ A.T
    return (U * data.tmask[:, None]).T


def loglik_terms(
    st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig
) -> Array:
    """Per ``(i, t)`` Gaussian log densities ``m_t log N(e_{i,t}; 0, λ_eff,i,t)``; ``n × T_pad``."""
    e = residuals(st, data, prior, static)
    lv = effective_log_var(st.eq.vol, static)
    live = data.tmask[None, :] > 0
    lv = jnp.where(live, lv, 0.0)
    terms = -0.5 * (LOG_2PI + lv + e**2 * jnp.exp(-lv))
    return jnp.where(live, terms, 0.0)


def loglik(st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig) -> Array:
    """``log p(Y | θ)``, exact given the latents (``h``, ``o``, ``τ``)."""
    return jnp.sum(loglik_terms(st, data, prior, static))


def log_e2_clean(
    st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig
) -> Array:
    """``log ẽ² = log e² − 2 log o − log τ`` (``n × T_pad``), 0 on padded rows (masked before
    the log); an exact zero in a real row gives ``−∞`` (absorbed by ``log c``)."""
    e = residuals(st, data, prior, static)
    live = data.tmask[None, :] > 0
    log_e2 = jnp.where(live, jnp.log(jnp.where(live, e**2, 1.0)), 0.0)
    scale2 = effective_log_var(st.eq.vol, static) - effective_log_var(
        st.eq.vol, static, "persistent"
    )
    return jnp.where(live, log_e2 - scale2, 0.0)


def y_star(st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig) -> Array:
    """``y* = logaddexp(log ẽ², log c_i)`` (D-2); ``n × T_pad``, 0 on padded rows."""
    if static.vol.persistent != "sv":
        raise ValueError("y* and mixture indicators belong to the SV core")
    log_c = prior.eq.vol_core.log_c[:, None]
    ys = jnp.logaddexp(log_e2_clean(st, data, prior, static), log_c)
    return jnp.where(data.tmask[None, :] > 0, ys, 0.0)


def mixture_terms(
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    static: StaticConfig,
    s: Array,
    *,
    exact: bool,
) -> Array:
    """Per ``(i, t)`` augmentation terms of :func:`log_joint` (``n × T_pad``, 0 on padded rows):
    ``log p_s + log N(ξ; m_s, v_s)`` (minus ``log g(ξ)`` when ``exact``)."""
    table = mixture_table(static.vol.mixture)
    xi = y_star(st, data, prior, static) - st.eq.vol.core.h[:, 1:]
    comps = log_mixture_components(xi, table)  # (n, T_pad, J)
    picked = jnp.take_along_axis(comps, s[..., None].astype(jnp.int32), axis=-1)[..., 0]
    if exact:
        picked = picked - log_mixture_density(xi, table)
    return jnp.where(data.tmask[None, :] > 0, picked, 0.0)


def log_joint(
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    static: StaticConfig,
    s: Array | None = None,
    *,
    exact: bool | None = None,
) -> Array:
    """``log p(θ, y)``; with indicators ``s`` (``i32[n, T_pad]``, components in descending-mean
    order) the augmented joint of the mixture model (module docstring). ``exact`` defaults to
    ``static.vol.exact``."""
    lp = log_prior(st, prior, static, data.tmask)
    if s is None:
        return lp + loglik(st, data, prior, static)
    exact = static.vol.exact if exact is None else bool(exact)
    aug = jnp.sum(mixture_terms(st, data, prior, static, s, exact=exact))
    return lp + aug + (loglik(st, data, prior, static) if exact else 0.0)


__all__ = [
    "log_e2_clean",
    "log_joint",
    "log_prior",
    "log_prior_eq",
    "loglik",
    "loglik_terms",
    "mixture_terms",
    "residuals",
    "y_star",
]
