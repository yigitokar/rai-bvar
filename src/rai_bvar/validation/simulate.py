"""Prior sampler and observable simulators for the validation harness (DESIGN §9.2; N06 §2.1).

Both simulators are written independently of the posterior sampler (two-implementation rule,
D-22): the prior sampler composes the per-kind references of ``reference.py``; the observable
simulator runs the model forward from the fixed presample, **recursively in t**, with the
**current** latent states of the chain state (Geweke 2004 error 2; N06 §6 pitfall 2):

    structural:  y_t = A⁻¹(Γ x_t + e_t),   γ_i = δ_i + M_c a_i         (D3; N05 §3, §4.1)
    reduced:     y_t = Π'x_t + A⁻¹ e_t                                  (CCCM 2022; N04 §2)
    zero mean:   y_t = A⁻¹ e_t
    e_{i,t} = exp(½ log λ_eff,i,t) ε_{i,t},   log λ_eff = log λ^pers + 2 log o + log τ,   ε ~ N(0, I)

with ``A = I + strict-lower(a)`` and ``x_t = (d_t', y_{t−1}', …, y_{t−p}')'`` (``data/design.py``
layout, trend ``t0 + t``). The returned :class:`EngineData` is rebuilt from the simulated
targets by the production jnp builder ``engines.mcmc.data.build_engine_data`` (itself tested
against ``data/design.py``), so every cache the sampler derives from ``y`` is recomputed.

The auxiliary-model simulator (block configuration ``aux_sv``; N01 §12.3(1), N06 §0 D-8) draws
``y*_t = h_t + m_{s_t} + √v_{s_t} z_t`` with fresh indicators ``s_t ~ p`` (``s`` is never model
state, so this is a draw from ``p(y* | θ)`` of the auxiliary model) and sets
``e_t = ±exp(½(y*_t + 2 log o_t + log τ_t))``.

Noise is injected (D-17): ``*_core`` functions take the normal and uniform draws explicitly, so
tests compare them with NumPy loops on the same noise.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.linalg import solve_triangular

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc import data as engine_data
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import (
    ChainState,
    EngineData,
    EqPrior,
    EqState,
    GlobalState,
    PriorArrays,
    Status,
    VolEq,
)
from rai_bvar.validation.reference import (
    coupling_log_scale,
    effective_log_var,
    mixture_table,
    references_for,
)

Array = jax.Array
_F64 = jnp.float64

# fold_in tags of the prior sampler (validation stream; independent of keys.Sub)
_EQ, _GLOB = 0, 1
_CORE, _TRANS, _STRUCT, _MEAN, _COEF = 0, 1, 2, 3, 4


class Design(Protocol):
    """What the simulators need from a fixed design (``joint.FixedDesign``)."""

    presample: Any  # (p, n) fixed y_{1−p:0}
    T_eff: int
    t0: float


def tmask_of(T_eff: Array | int, T_pad: int) -> Array:
    """``m_t = 1{t < T_eff}`` as float64."""
    return (jnp.arange(T_pad, dtype=jnp.int32) < T_eff).astype(_F64)


def prior_sample(
    key: Array, prior: PriorArrays, static: StaticConfig, T_eff: Array | int
) -> ChainState:
    """θ ~ p(θ): every parameter and latent state (``h``, ``o``, ``τ``, ``p``, ``ν``; never ``s``).

    Per equation, in this order (λ-scaled blocks need ``λ_i``; DESIGN §4.7): persistent core,
    transitory scale, ``a_i`` (structure), the mean's ``δ_i``, the coefficient prior's own state;
    then the global parts (reduced form ``Π``). Accumulators are zero and ``status`` is
    :meth:`Status.initial`. Works under ``jit``/``vmap`` with a traced ``T_eff``.
    """
    refs = references_for(static)
    n, T_pad = static.shapes.n, static.shapes.T_pad
    tmask = tmask_of(T_eff, T_pad)
    T_eff = jnp.asarray(T_eff, jnp.int32)
    k_eq, k_glob = random.fold_in(key, _EQ), random.fold_in(key, _GLOB)

    def one(i: Array, prior_i: EqPrior) -> EqState:
        k = random.fold_in(k_eq, i.astype(jnp.uint32))
        args = (prior_i, static, tmask, T_eff)
        core = refs.core.prior_sample_eq(random.fold_in(k, _CORE), i, *args)
        trans = refs.trans.prior_sample_eq(random.fold_in(k, _TRANS), i, *args)
        scale = coupling_log_scale(core, static)
        struct = refs.structure.prior_sample_eq(
            random.fold_in(k, _STRUCT), i, *args, log_scale=scale
        )
        mean = refs.mean.prior_sample_eq(random.fold_in(k, _MEAN), i, *args, log_scale=scale)
        coef = refs.coef.prior_sample_eq(random.fold_in(k, _COEF), i, *args, log_scale=scale)
        return EqState(struct=struct, mean=mean, coef=coef, vol=VolEq(core=core, trans=trans))

    eq = jax.vmap(one)(jnp.arange(n, dtype=jnp.int32), prior.eq)
    glob = GlobalState(
        mean=refs.mean.prior_sample_global(random.fold_in(k_glob, _MEAN), prior, static),
        coef=refs.coef.prior_sample_global(random.fold_in(k_glob, _COEF), prior, static),
        vol=refs.core.prior_sample_global(random.fold_in(k_glob, _CORE), prior, static),
        struct=refs.structure.prior_sample_global(random.fold_in(k_glob, _STRUCT), prior, static),
    )
    return ChainState(eq=eq, glob=glob, status=Status.initial())


def unit_lower(a: Array) -> Array:
    """``A = I + strict-lower(a)`` from the state layout ``a[i, j] = a_ij`` (``j < i``), shape
    ``(n, n−1)`` → ``(n, n)``. Entries with ``j ≥ i`` are ignored."""
    n = a.shape[0]
    full = jnp.concatenate([a, jnp.zeros((n, 1), a.dtype)], axis=1)
    return jnp.eye(n, dtype=_F64) + jnp.tril(full, k=-1)


def structural_gamma(st: ChainState, prior: PriorArrays) -> Array:
    """``Γ`` with rows ``γ_i = δ_i + M_c a_i`` (D3; N05 §4.1); shape ``(n, k)``."""
    return st.eq.mean.delta + st.eq.struct.a @ prior.shared.Mc.T


def _det_terms(t: Array, n_det: int) -> Array:
    return jnp.stack([jnp.ones_like(t), t][:n_det]) if n_det else jnp.zeros((0,), _F64)


def simulate_targets(
    shocks: Array, st: ChainState, static: StaticConfig, prior: PriorArrays, fixed: Design
) -> tuple[Array, Array]:
    """Forward recursion given the structural shocks ``e`` (``T_pad × n``, already scaled).

    Returns ``(Y, X)`` (``T_pad × n``, ``T_pad × k``), zero on rows ``t ≥ T_eff``; ``X`` is the
    regressor matrix the recursion used (equal to the builder's ``X``; tested)."""
    s = static.shapes
    n, p, n_det, T_pad = s.n, s.lags, s.n_det, s.T_pad
    tmask = tmask_of(fixed.T_eff, T_pad)
    A = unit_lower(st.eq.struct.a)
    kind = static.mean.kind
    if kind == "linear_structural":
        G = structural_gamma(st, prior)
    elif kind == "linear_reduced":
        G = st.glob.mean.Pi.T
    else:
        G = jnp.zeros((n, s.k), _F64)
    t0 = jnp.asarray(fixed.t0, _F64)

    def body(window: Array, inp: tuple[Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        e_t, m_t, t = inp
        x_t = jnp.concatenate([_det_terms(t0 + t, n_det), window[::-1].reshape(-1)])
        if kind == "linear_reduced":
            y = G @ x_t + solve_triangular(A, e_t, lower=True, unit_diagonal=True)
        else:
            y = solve_triangular(A, G @ x_t + e_t, lower=True, unit_diagonal=True)
        y = jnp.where(m_t > 0, y, 0.0)
        window = jnp.concatenate([window[1:], y[None]], axis=0)
        return window, (y, jnp.where(m_t > 0, x_t, 0.0))

    window0 = jnp.asarray(fixed.presample, _F64).reshape(p, n)
    steps = jnp.arange(T_pad, dtype=_F64)
    _, (Y, X) = lax.scan(body, window0, (shocks, tmask, steps))
    return Y, X


def _engine_data(Y: Array, static: StaticConfig, fixed: Design) -> EngineData:
    s = static.shapes
    return engine_data.build_engine_data(
        Y,
        jnp.asarray(fixed.presample, _F64).reshape(s.lags, s.n),
        jnp.asarray(fixed.T_eff, jnp.int32),
        n_det=s.n_det,
        t0=fixed.t0,
    )


def gaussian_shocks(eps: Array, st: ChainState, static: StaticConfig) -> Array:
    """``e_{t,i} = exp(½ log λ_eff,i,t) ε_{t,i}`` from standard normals ``eps`` (``T_pad × n``)."""
    log_lam = effective_log_var(st.eq.vol, static)  # (n, T_pad)
    return jnp.exp(0.5 * log_lam).T * eps


def simulate_obs_core(
    eps: Array, st: ChainState, static: StaticConfig, prior: PriorArrays, fixed: Design
) -> EngineData:
    """:func:`simulate_obs` with injected standard normals ``eps`` (``T_pad × n``)."""
    Y, _ = simulate_targets(gaussian_shocks(eps, st, static), st, static, prior, fixed)
    return _engine_data(Y, static, fixed)


def simulate_obs(
    key: Array, st: ChainState, static: StaticConfig, prior: PriorArrays, fixed: Design
) -> EngineData:
    """``y ~ p(y | θ)`` with the state's current latents; ``X`` rebuilt by
    ``data.build_engine_data`` (DESIGN §9.2)."""
    s = static.shapes
    eps = random.normal(key, (s.T_pad, s.n), dtype=_F64)
    return simulate_obs_core(eps, st, static, prior, fixed)


def aux_shocks(
    s_idx: Array, z: Array, u: Array, st: ChainState, static: StaticConfig
) -> tuple[Array, Array]:
    """Auxiliary-model shocks from injected indicators ``s_idx`` (``T_pad × n``, components in
    descending-mean order), normals ``z`` and uniforms ``u`` (signs). Returns ``(e, y*)``.

    ``y*_{t,i} = h_{i,t+1} + m_{s} + √v_s z``;  ``e = sign(u − ½)·exp(½(y* + 2 log o + log τ))``.
    Requires an SV persistent core (the auxiliary model is an SV device)."""
    if static.vol.persistent != "sv":
        raise ValueError("the auxiliary mixture model needs the SV persistent core")
    table = mixture_table(static.vol.mixture)
    m, v = jnp.asarray(table.m)[s_idx], jnp.asarray(table.v)[s_idx]
    h = st.eq.vol.core.h[:, 1:].T  # (T_pad, n): data row t ↔ h[t+1]
    y_star = h + m + jnp.sqrt(v) * z
    log_scale2 = (
        effective_log_var(st.eq.vol, static) - effective_log_var(st.eq.vol, static, "persistent")
    ).T
    sign = jnp.where(u < 0.5, -1.0, 1.0)
    return sign * jnp.exp(0.5 * (y_star + log_scale2)), y_star


def simulate_aux_obs(
    key: Array, st: ChainState, static: StaticConfig, prior: PriorArrays, fixed: Design
) -> EngineData:
    """``y`` from the auxiliary mixture model (block configuration ``aux_sv``)."""
    s = static.shapes
    shape = (s.T_pad, s.n)
    table = mixture_table(static.vol.mixture)
    s_idx = random.categorical(
        random.fold_in(key, 0), jnp.log(jnp.asarray(table.p)), shape=shape
    ).astype(jnp.int32)
    z = random.normal(random.fold_in(key, 1), shape, dtype=_F64)
    u = random.uniform(random.fold_in(key, 2), shape, dtype=_F64)
    e, _ = aux_shocks(s_idx, z, u, st, static)
    Y, _ = simulate_targets(e, st, static, prior, fixed)
    return _engine_data(Y, static, fixed)


__all__ = [
    "aux_shocks",
    "gaussian_shocks",
    "prior_sample",
    "simulate_aux_obs",
    "simulate_obs",
    "simulate_obs_core",
    "simulate_targets",
    "structural_gamma",
    "tmask_of",
    "unit_lower",
]
