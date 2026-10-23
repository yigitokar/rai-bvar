"""Mutation suite M1–M18: the mutants, how each is injected, and the tier checks that catch it
(DESIGN §9.7, §13 WP19; N06 §9.5). M17–M18 are the WP25 mutants of the SOC/SUR dummy rows.

Every mutant is injected through the production seams only (DESIGN §4.9): a named kernel internal
(``compose(static, overrides={name: fn})`` via ``joint_spec(..., kernel_overrides=...)``), a kernel
slot (M2: a mutant volatility block), the sampler's prior arrays (M5), the simulator (M3) or a
configuration (M16). The tier checks:

* **U** — deterministic identities at a joint draw ``(θ, y)`` (DESIGN §9.3): density ratios of the
  mutated conditional against the independent ``validation.log_joint`` (Grosse & Duvenaud 2014
  eq. 1), Bayes consistency of the indicator pmf against the second mixture transcription, and
  closed forms (mixture mean, the D3 map ``γ = δ + M_c a``, the fold and ASIS maps, the exact-mode
  acceptance). Each U check returns an error that the production internal keeps below its
  tolerance (checked on the same draw).
* **B** — block-conditional SBC of the ``π_0`` draw of the program's ``cccm_core`` (M1; N06 §5.2,
  ``N = 200``, ``L = 99``, ranks of ``π_0`` and of the log-likelihood) or one-block invariance of
  the volatility block through the production sweep (N06 §5.3; ``L = 20``, ``N = 1000``,
  two-sample KS per statistic against independent joint draws); Bonferroni at ``α = 1e-3``,
  Gandy–Scott. (One-block invariance of the whole CCCM scan does not see M1: after 20 scans the
  marginals of ``Π`` and of the log-likelihood over datasets stay within KS power at
  ``N = 1000`` — the "parameter ranks are blind" effect of N06 §7.3.)
* **G** — ``geweke_sequential`` at G-small sizes (DESIGN §9.5) of the mutant against the same
  configuration without the mutation, with common seeds.
* **S** — SBC (``sbc_sequential``, short configuration) of the mutant against the same
  configuration without the mutation.

A mutant is caught by a tier when the tier's decision rule fails on the mutant **and** passes on
its control. :data:`MUTANTS` records, per mutant, the catching tiers confirmed by running
``test_mutants.py`` (with the evidence observed on 2026-10-22, M1 Pro, and reproduced on x86-64
Linux); DESIGN §9.7 requires at least one per mutant.

Notes found while confirming the tiers:

* M11 (indicator pmf without ``σ_j⁻¹``) and M12 (double −1.2704 shift) are **not** errors of the
  exact-mode sampler (``exact=True``, the default): the exact correction uses ``log g`` of the same
  (mutated) logits and table, so the augmented chain stays exact for any indicator law and any
  table (N01 §10.2). They are bugs of the auxiliary-mixture sampler (``exact=False``), which is what
  the B and G checks run on (the ``aux_sv`` configuration, where that sampler is exact).
* M7 (uncorrected NC move) is invisible with OCSN-10 at test sizes (the correction is tiny); with
  the one-component table ``COARSE1`` injected through ``mixture_table`` (exact mode is exact for
  any table) the control passes and M7 fails.
* M2 keeps the stale indicators in the ``rb_h_sq[1:]`` accumulator of the SV state (never read by
  any test function; the runner, not used by the harness, would zero it after burn-in) and runs
  with ``exact=False``: Primiceri's Algorithm 1 is an auxiliary-mixture sampler.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from scipy import stats

from rai_bvar.components.mean.linear_reduced import cccm_core
from rai_bvar.components.vol import constant as CN
from rai_bvar.components.vol import mixture as MX
from rai_bvar.components.vol import sv as SV
from rai_bvar.components.vol import transitory as TR
from rai_bvar.components.vol.robust import VolBlockImpl
from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc.keys import Tier
from rai_bvar.engines.mcmc.linalg import precision_draw, tridiag_precision_sample
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    ChainState,
    Coupling,
    Empty,
    GaussianPrec,
    StepInfo,
    SVPrior,
)
from rai_bvar.validation.geweke import GewekeConfig, Part, geweke_sequential, part, rep_keys
from rai_bvar.validation.joint import JointSpec, composition_id, joint_spec
from rai_bvar.validation.log_joint import log_joint, loglik, residuals
from rai_bvar.validation.reference import (
    effective_log_var,
    log_beta_from_logs,
    log_inv_gamma_from_log,
    log_mixture_components,
    nu_grid,
    t_scale_constant,
)
from rai_bvar.validation.reference import mixture_table as reference_table
from rai_bvar.validation.sbc import SBCConfig, block_sbc_sequential, sbc_sequential
from rai_bvar.validation.sequential import (
    bonferroni,
    gandy_scott,
    stage_multiplier,
    stage_seed,
)
from rai_bvar.validation.simulate import structural_gamma, unit_lower

Array = jax.Array
_F64 = jnp.float64
LOGCHI2_MEAN = -1.2703628454614782  # ψ(½) + log 2 (N01 §4)

G_SMALL: dict[str, Any] = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}
SEED_G = 20261022  # the G-small seed of tests/geweke: the unmutated controls are those PR runs
SEED_B = 20261028
SEED_S = 20261029
SEED_U = 20261030


# =========================================================================== mutant internals


def ccm2019_cccm_core(
    Pi: Array, a: Array, inv_lam: Array, X: Array, Y: Array, XX: Any, mean0: Array,
    prec0: Array, Z: Array, *, gram: str, precompute: bool, tile: int = 0,
) -> tuple[Array, StepInfo]:  # fmt: skip
    """M1: the CCM (2019) triangular step — ``π_j`` from equation ``j`` alone,
    ``y_j + Σ_{l<j} a_jl u_l = Xπ_j + e_j`` with weights ``1/λ_j`` (N04 §4): exact only for the
    last equation."""
    n = Pi.shape[1]
    A = unit_lower(a)

    def body(j: Array, carry: tuple[Array, StepInfo]) -> tuple[Array, StepInfo]:
        Pi, info = carry
        U = Y - X @ Pi
        target = Y[:, j] + U @ A[j] - U[:, j]  # y_j + Σ_{l<j} a_jl u_l
        w = inv_lam[j]
        K = X.T @ (w[:, None] * X) + jnp.diag(prec0[j])
        b = prec0[j] * mean0[j] + X.T @ (w * target)
        pi_j, info_j = precision_draw(K, b, Z[j])
        return Pi.at[:, j].set(pi_j), info.merge(info_j)

    return lax.fori_loop(0, n, body, (Pi, StepInfo.success()))


def m17_linear_conditional(ctx: Any, pm: Any, cm: Any, cache: Any, shared: Any) -> GaussianPrec:
    """M17 (WP25): Mode I SOC/SUR rows weighted by the *current* ``1/λ_i`` instead of the fixed
    ``1/σ̄_i²`` — the error N05 §7.7 warns against (the ``λ`` step does not see the rows, so the
    sampler targets the wrong posterior). Mode I, Constant core: the data weights are ``1/λ_i``."""
    c = jnp.concatenate([pm.cmask, cm.cmask])
    on = c > 0
    mean = jnp.where(on, jnp.concatenate([pm.mean, cm.mean]), 0.0)
    prec = jnp.where(on, jnp.concatenate([pm.prec, cm.prec]), 1.0)
    w = ctx.weights
    F = cache.F
    G = F.T @ (w[:, None] * F) + w[0] * cache.dummy.G
    fty = F.T @ (w * ctx.target) + w[0] * jnp.take(cache.dummy.FtY, ctx.i, axis=1)
    K = c[:, None] * G * c[None, :] + jnp.diag(prec)
    return GaussianPrec(K=K, b=prec * mean + c * fty)


def m18_cccm_core(*args: Any, dummy: Any = None, **kw: Any) -> tuple[Array, StepInfo]:
    """M18 (WP25): the SOC/SUR rows dropped from the CCCM ``Π`` step (the prior sampler and the
    reference keep them)."""
    return cccm_core(*args, **kw)


def m4_tau_conditional(log_r2: Array, nu: Array, log_c: Array) -> tuple[Array, Array]:
    """M4: ``τ | ·`` with shape ``ν/2`` instead of ``(ν + 1)/2``."""
    return 0.5 * nu, jnp.logaddexp(log_c, log_r2) - math.log(2.0)


def m6_nc_fold(sigma: Array, h_tilde: Array) -> tuple[Array, Array]:
    """M6: stochvol's ``|σ|`` without flipping ``h̃`` (N02 T16)."""
    return jnp.abs(sigma), h_tilde


def m7_nc_accept(log_alpha_aux: Array, w_prop: Array, w_cur: Array, exact: bool) -> Array:
    """M7: the NC ``(μ, σ)`` move accepted without the exact-mode ``w`` correction."""
    return log_alpha_aux


def m8_structural_gamma(delta: Array, a: Array, Mc: Array) -> Array:
    """M8: the D3 centring dropped (``γ_i = δ_i``)."""
    return delta


def m9_outlier_logits(
    log_z2: Array, log_p: Array, log1m_p: Array, K: int, nu: Array | None = None,
    log_c: Array | None = None,
) -> Array:  # fmt: skip
    """M9: the outlier pmf without the Gaussian normalising factor ``1/k``."""
    out = TR.outlier_logits(log_z2, log_p, log1m_p, K, nu, log_c)
    return out + jnp.log(jnp.arange(1, K + 1, dtype=_F64))


def m10_p_conditional(
    o_idx: Array, tmask: Array, T_eff: Array, a_p: Array, b_p: Array
) -> tuple[Array, Array]:
    """M10: ``p | o ~ Beta(a_p + N₁, b_p + T)`` (``T`` instead of ``T − N₁``)."""
    a, _ = TR.p_conditional(o_idx, tmask, T_eff, a_p, b_p)
    return a, b_p + jnp.asarray(T_eff, _F64)


def m11_indicator_logits(r: Array, table: MX.MixtureTable) -> Array:
    """M11: indicator logits without the ``σ_j⁻¹ = v_j^{−½}`` factor."""
    d = jnp.asarray(r, _F64)[..., None] - table.m
    return np.log(table.p) - d * d * table.half_inv_v


def m12_mixture_table(name: str) -> MX.MixtureTable:
    """M12: the effective means shifted by −1.2704 a second time."""
    return MX.mixture_table(name).shifted(-1.2704)


def m13_constant_conditional(
    log_e2c: Array, tmask: Array, T_eff: Array, cp: Any, cpl: Coupling
) -> tuple[Array, Array]:
    """M13: the IG shape with ``T_pad`` instead of ``T_eff`` (the sum of squares stays masked)."""
    return CN.constant_conditional(log_e2c, tmask, jnp.asarray(log_e2c.shape[0]), cp, cpl)


def m14_contemp_conditional(ctx: Any, g_i: Array, cm: Any) -> GaussianPrec:
    """M14: the A-row regression on ``+u_{<i}`` instead of ``−u_{<i}`` (N04 §9.4)."""
    Z = -ctx.cols * cm.cmask[None, :]
    Zw = Z * ctx.weights[:, None]
    K = Zw.T @ Z + jnp.diag(cm.prec)
    K = jnp.tril(K) + jnp.tril(K, -1).T
    b = cm.prec * cm.mean + Zw.T @ (ctx.target - g_i)
    return GaussianPrec(K=K, b=b)


def m15_nc_step(h: Array, mu: Array, phi: Array, sigma: Array, *args: Any, **kw: Any) -> Any:
    """M15: the ASIS map without the ``h̃`` rescale (``h̃ = h − μ``); exact for the default
    ``σ² ~ B_σχ²₁`` prior, where ``σ`` enters ``nc_step`` only through ``h̃``."""
    return SV.nc_step(h, mu, phi, jnp.ones_like(sigma), *args, **kw)


def coarse_table(name: str) -> MX.MixtureTable:
    """Validation device for M7 (not a mutant): the one-component table ``COARSE1``."""
    return MX.COARSE1


@dataclass(frozen=True)
class StaleIndicatorSV(SV.SVCore):
    """M2: Primiceri (2005) Algorithm 1 — ``h | s_old, y*(current residuals)`` with the indicators
    of the previous sweep, then ``θ_SV``, then ``s | h`` carried to the next sweep (DNP 2015 show
    this order does not target the posterior; N02 §3.3: it diverges). The stale indicators live in
    ``rb_h_sq[1:]``; the ``h`` move reads them from ``noise.u_s`` (set by :meth:`sample_core`)."""

    def h_move(self, noise, h, mu, phi, sigma, log_e2c, ys, obs, T_eff, prior_i):
        vol, table = self.static, self.table()
        s = noise.u_s.astype(jnp.int32)
        diag, off, b = SV.awol_system(
            ys, s, obs, T_eff, mu, phi, sigma, prior_i, table, h0=vol.h0, rw=vol.rw
        )
        x, _, piv = tridiag_precision_sample(diag, off, b, noise.z_h)
        h_new = sigma * x if vol.rw else mu + sigma * x
        m, v = jnp.asarray(table.m)[s], jnp.asarray(table.v)[s]
        return h_new, jnp.ones((), bool), jnp.zeros((), _F64), m, v, piv

    def sample_core(self, noise, cs, log_e2c, tmask, T_eff, prior_i, *, obs=None):
        J = self.table().J
        stale = jnp.clip(jnp.round(cs.rb_h_sq[1:]), 0, J - 1)
        new, info = super().sample_core(
            dataclasses.replace(noise, u_s=stale), cs, log_e2c, tmask, T_eff, prior_i, obs=obs
        )
        sp = prior_i if isinstance(prior_i, SVPrior) else prior_i.vol_core
        mask, _ = SV.observation_mask(log_e2c, tmask, self.static.zero_offset)
        ys = SV.y_star(log_e2c, sp.log_c, mask)
        s_new, _ = SV.draw_indicators(
            noise.u_s, ys - new.h[1:], self.table(), self.impl("indicator_logits")
        )
        stash = cs.rb_h_sq.at[1:].set(s_new.astype(_F64))
        return dataclasses.replace(new, rb_h_sq=stash), info


def stale_indicator_vol(spec: JointSpec) -> VolBlockImpl:
    """The M2 volatility block (``vol`` slot) for ``spec``'s static configuration."""
    vol, T_pad = spec.static.vol, spec.static.shapes.T_pad
    return VolBlockImpl(vol, T_pad, core=StaleIndicatorSV(vol, T_pad))


# =========================================================================== joint draws, U checks


def joint_draw(spec: JointSpec, seed: int) -> tuple[ChainState, Any]:
    """``(θ, y)`` from the references (prior sampler, simulator)."""
    key = random.key(seed, impl="threefry2x32")
    st = spec.prior_sample(random.fold_in(key, 0))
    return st, spec.simulate(random.fold_in(key, 1), st)


def program_of(spec: JointSpec) -> Any:
    """The composed production program of ``spec`` (its kernel overrides included)."""
    from rai_bvar.engines.mcmc.compose import compose

    return compose(spec.static, overrides=dict(spec.overrides) or None)


def _rel(lhs: float, rhs: float) -> float:
    return abs(float(lhs) - float(rhs)) / (1.0 + abs(float(rhs)))


def _slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def _set_leaf(st: ChainState, path: str, value: Array) -> ChainState:
    """``st`` with the leaf at dotted ``path`` replaced (``eq.vol.trans.log_tau`` …)."""
    from rai_bvar.engines.mcmc.state import leaf_path

    def pick(p: tuple[Any, ...], x: Array) -> Array:
        return value if leaf_path(p) == path else x

    return jax.tree_util.tree_map_with_path(pick, st)


def _lj(spec: JointSpec, st: ChainState, data: Any) -> float:
    return float(log_joint(st, data, spec.prior, spec.static))


def _masked_log_e2(e: Array, tmask: Array) -> Array:
    live = tmask > 0
    return jnp.where(live, jnp.log(jnp.where(live, e * e, 1.0)), 0.0)


def u_cccm(spec: JointSpec, seed: int = SEED_U) -> float:
    """M1 (U, density ratio): the ``π_0`` draw of the program's ``cccm_core`` is affine in
    ``Z[0]``, ``π_0 = μ + Cz``; ``log N(x; μ, CCᵀ)`` differences must equal ``log p(θ, y)``
    differences (the exact full conditional; N04 §5.3)."""
    st, data = joint_draw(spec, seed)
    static, prior = spec.static, spec.prior
    core = program_of(spec).global_pre[0].impl("cccm_core")
    n, k = static.shapes.n, static.shapes.k
    lv = effective_log_var(st.eq.vol, static)
    live = data.tmask[None, :] > 0
    inv_lam = jnp.where(live, jnp.exp(-jnp.where(live, lv, 0.0)), 0.0)
    c = prior.eq.coef

    def col0(Z: Array) -> np.ndarray:
        Pi, _ = core(st.glob.mean.Pi, st.eq.struct.a, inv_lam, data.X, data.Y, Empty(), c.mean0,
                     c.prec0, Z, gram="batched", precompute=True)  # fmt: skip
        return np.asarray(Pi[:, 0])

    Z0 = jnp.zeros((n, k))
    mu = col0(Z0)
    C = np.stack([col0(Z0.at[0, r].set(1.0)) - mu for r in range(k)], axis=1)
    dist = stats.multivariate_normal(mu, C @ C.T)
    r = np.random.default_rng(seed)
    Pi = np.asarray(st.glob.mean.Pi)
    errs = []
    for _ in range(4):
        x1, x2 = (mu + C @ r.standard_normal(k) for _ in range(2))
        sts = []
        for x in (x1, x2):
            P = Pi.copy()
            P[:, 0] = x
            sts.append(dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=jnp.asarray(P)))))  # fmt: skip
        lq = dist.logpdf(x1) - dist.logpdf(x2)
        errs.append(_rel(lq, _lj(spec, sts[0], data) - _lj(spec, sts[1], data)))
    return max(errs)


def _production_cache(spec: JointSpec, data: Any) -> Any:
    from rai_bvar.validation.joint import production_sweep

    prepare, _ = production_sweep(spec.static, spec.overrides)
    return prepare(data, spec.prior)


def u_cccm_dummies(spec: JointSpec, seed: int = SEED_U) -> float:
    """M18 (U, density ratio): as :func:`u_cccm`, with the SOC/SUR rows the production step hands
    the program's ``cccm_core`` (``dummy=(X_dᵀX_d, X_dᵀY_d, w)`` of the fit cache)."""
    st, data = joint_draw(spec, seed)
    static, prior = spec.static, spec.prior
    d = _production_cache(spec, data).dummy
    core = program_of(spec).global_pre[0].impl("cccm_core")
    n, k = static.shapes.n, static.shapes.k
    lv = effective_log_var(st.eq.vol, static)
    live = data.tmask[None, :] > 0
    inv_lam = jnp.where(live, jnp.exp(-jnp.where(live, lv, 0.0)), 0.0)
    c = prior.eq.coef

    def col0(Z: Array) -> np.ndarray:
        Pi, _ = core(st.glob.mean.Pi, st.eq.struct.a, inv_lam, data.X, data.Y, Empty(), c.mean0,
                     c.prec0, Z, gram="batched", precompute=True, dummy=(d.G, d.FtY, d.w))  # fmt: skip
        return np.asarray(Pi[:, 0])

    Z0 = jnp.zeros((n, k))
    mu = col0(Z0)
    C = np.stack([col0(Z0.at[0, r].set(1.0)) - mu for r in range(k)], axis=1)
    dist = stats.multivariate_normal(mu, C @ C.T)
    r = np.random.default_rng(seed)
    Pi = np.asarray(st.glob.mean.Pi)
    errs = []
    for _ in range(4):
        x1, x2 = (mu + C @ r.standard_normal(k) for _ in range(2))
        sts = []
        for x in (x1, x2):
            P = Pi.copy()
            P[:, 0] = x
            sts.append(dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=jnp.asarray(P)))))  # fmt: skip
        lq = dist.logpdf(x1) - dist.logpdf(x2)
        errs.append(_rel(lq, _lj(spec, sts[0], data) - _lj(spec, sts[1], data)))
    return max(errs)


def u_dummy_equation(spec: JointSpec, seed: int = SEED_U) -> float:
    """M17 (U, density ratio): the last equation's ``θ_i = [δ_i | a_i]`` conditional of the
    program's ``linear_conditional`` (context, moments and fit cache built by the production
    kernels) vs ``log p(θ, y)`` with the SOC/SUR rows (``validation.dummies``)."""
    st, data = joint_draw(spec, seed)
    program = program_of(spec)
    S, M = program.structure, program.mean
    cache = _production_cache(spec, data)
    n, k = spec.static.shapes.n, spec.static.shapes.k
    i = n - 1
    eq_i, prior_i = _slice(st.eq, i), _slice(spec.prior.eq, i)
    ytil = S.pre_residual(st.glob, data, M.fitted_reduced(st.glob, data))
    ctx = S.context(jnp.int32(i), ytil, eq_i, program.vol, data, st.glob.vol)
    cm = S.contemp_moments(ctx, prior_i)
    pm = program.coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
    cond = M.impl("linear_conditional")(ctx, pm, cm, cache, spec.prior.shared)
    K, b = np.asarray(cond.K), np.asarray(cond.b)
    mask = np.asarray(jnp.concatenate([pm.cmask, cm.cmask])) > 0
    th0 = np.concatenate([np.asarray(st.eq.mean.delta[i]), np.asarray(st.eq.struct.a[i])])
    r = np.random.default_rng(seed)
    errs = []
    for _ in range(4):
        xs = [np.where(mask, th0 + 0.3 * r.standard_normal(th0.size), 0.0) for _ in range(2)]
        lq = [-0.5 * x @ K @ x + b @ x for x in xs]
        lj = []
        for x in xs:
            s1 = _set_leaf(st, "eq.mean.delta", st.eq.mean.delta.at[i].set(jnp.asarray(x[:k])))
            lj.append(_lj(spec, _set_leaf(s1, "eq.struct.a", s1.eq.struct.a.at[i].set(jnp.asarray(x[k:]))), data))  # fmt: skip
        errs.append(_rel(lq[0] - lq[1], lj[0] - lj[1]))
    return max(errs)


def u_tau(spec: JointSpec, seed: int = SEED_U) -> float:
    """M4 (U, density ratio): ``τ_t | ·`` of the program's ``tau_conditional`` vs ``log p(θ, y)``."""
    st, data = joint_draw(spec, seed)
    static = spec.static
    fn = program_of(spec).vol.transitory.impl("tau_conditional")
    e = residuals(st, data, spec.prior, static)
    lp = effective_log_var(st.eq.vol, static, "persistent")
    i, t = 0, 3
    log_r2 = jnp.where(data.tmask > 0, _masked_log_e2(e[i], data.tmask) - lp[i], 0.0)
    nu = nu_grid(static)[st.eq.vol.trans.nu_idx[i]]
    shape, log_scale = fn(log_r2, nu, jnp.log(t_scale_constant(nu, static)))
    log_tau = np.asarray(st.eq.vol.trans.log_tau)
    errs = []
    for d1, d2 in ((-0.4, 0.3), (0.2, 0.9), (-1.0, -0.1)):
        lts = (log_tau[i, t] + d1, log_tau[i, t] + d2)
        lq = [float(log_inv_gamma_from_log(x, shape, jnp.exp(log_scale[t]))) for x in lts]
        lj = []
        for x in lts:
            L = log_tau.copy()
            L[i, t] = x
            lj.append(_lj(spec, _set_leaf(st, "eq.vol.trans.log_tau", jnp.asarray(L)), data))
        errs.append(_rel(lq[0] - lq[1], lj[0] - lj[1]))
    return max(errs)


def u_outlier(spec: JointSpec, seed: int = SEED_U) -> float:
    """M9 (U, density ratio): the program's ``outlier_logits`` differences vs ``log p(θ, y)``."""
    st, data = joint_draw(spec, seed)
    static = spec.static
    tr = st.eq.vol.trans
    fn = program_of(spec).vol.transitory.impl("outlier_logits")
    e = residuals(st, data, spec.prior, static)
    lp = effective_log_var(st.eq.vol, static, "persistent")
    i, t = 0, 2
    log_z2 = jnp.where(data.tmask > 0, _masked_log_e2(e[i], data.tmask) - lp[i], 0.0)
    logits = np.asarray(fn(log_z2, tr.log_p[i], tr.log1m_p[i], static.vol.outlier_K))
    o = np.asarray(tr.o_idx)
    errs = []
    for k1, k2 in ((0, 1), (0, 4), (2, 9), (1, 19)):
        lj = []
        for k in (k1, k2):
            occ = o.copy()
            occ[i, t] = k
            lj.append(_lj(spec, _set_leaf(st, "eq.vol.trans.o_idx", jnp.asarray(occ)), data))
        errs.append(_rel(logits[t, k1] - logits[t, k2], lj[0] - lj[1]))
    return max(errs)


def u_p(spec: JointSpec, seed: int = SEED_U) -> float:
    """M10 (U, density ratio): the program's ``p_conditional`` Beta vs ``log p(θ, y)``."""
    st, data = joint_draw(spec, seed)
    i = 0
    # three outliers in equation i (the identity holds at any state; N₁ > 0 makes T − N₁ ≠ T)
    st = _set_leaf(
        st,
        "eq.vol.trans.o_idx",
        st.eq.vol.trans.o_idx.at[i, :3].set(jnp.array([1, 4, 0], jnp.int32)),
    )
    tr = st.eq.vol.trans
    tp = spec.prior.eq.transitory
    fn = program_of(spec).vol.transitory.impl("p_conditional")
    a, b = fn(tr.o_idx[i], data.tmask, data.T_eff, tp.a_p[i], tp.b_p[i])
    errs = []
    for p1, p2 in ((0.05, 0.2), (0.11, 0.31), (0.02, 0.5)):
        lq, lj = [], []
        for p in (p1, p2):
            lq.append(float(log_beta_from_logs(math.log(p), math.log1p(-p), a, b)))
            s2 = _set_leaf(st, "eq.vol.trans.log_p", tr.log_p.at[i].set(math.log(p)))
            s2 = _set_leaf(s2, "eq.vol.trans.log1m_p", tr.log1m_p.at[i].set(math.log1p(-p)))
            lj.append(_lj(spec, s2, data))
        errs.append(_rel(lq[0] - lq[1], lj[0] - lj[1]))
    return max(errs)


def u_indicator(spec: JointSpec, seed: int = SEED_U) -> float:
    """M11 (U, Bayes consistency; N01 §12.2): ``softmax`` of the program's ``indicator_logits`` vs
    ``p_j N(r; m_j, v_j)/Σ`` from the second transcription of the table."""
    core = program_of(spec).vol.core
    fn = core.impl("indicator_logits")
    r = jnp.linspace(-14.0, 4.0, 61)
    got = jax.nn.softmax(fn(r, core.table()), axis=-1)
    ref = jax.nn.softmax(log_mixture_components(r, reference_table(spec.static.vol.mixture)), -1)
    return float(jnp.max(jnp.abs(got - ref)))


def u_table(spec: JointSpec, seed: int = SEED_U) -> float:
    """M12 (U, closed form; N01 §12.1): the program's table has mean ``ψ(½) + log 2 ± 1e-4`` and
    the means of the second transcription. Returns ``max(|mean − ψ(½) − log 2|, max|Δm|)``."""
    table = program_of(spec).vol.core.table()
    ref = reference_table(spec.static.vol.mixture)
    return max(abs(table.mean - LOGCHI2_MEAN), float(np.max(np.abs(table.m - ref.m))))


def u_constant(spec: JointSpec, seed: int = SEED_U) -> float:
    """M13 (U, density ratio): the program's ``constant_conditional`` IG vs ``log p(θ, y)``."""
    st, data = joint_draw(spec, seed)
    static = spec.static
    fn = program_of(spec).vol.core.impl("constant_conditional")
    e = residuals(st, data, spec.prior, static)
    i = 0
    cp = _slice(spec.prior.eq.vol_core, i)
    shape, scale = fn(
        _masked_log_e2(e[i], data.tmask), data.tmask, data.T_eff, cp, Coupling.zeros()
    )
    ll = np.asarray(st.eq.vol.core.log_lam)
    errs = []
    for d1, d2 in ((-0.3, 0.4), (0.1, 1.0), (-1.2, 0.0)):
        lq, lj = [], []
        for d in (d1, d2):
            lq.append(float(log_inv_gamma_from_log(ll[i] + d, shape, scale)))
            lj.append(_lj(spec, _set_leaf(st, "eq.vol.core.log_lam", jnp.asarray(ll).at[i].add(d)), data))  # fmt: skip
        errs.append(_rel(lq[0] - lq[1], lj[0] - lj[1]))
    return max(errs)


def u_contemp(spec: JointSpec, seed: int = SEED_U) -> float:
    """M14 (U, density ratio; N04 §9.4): ``a_i | Π, λ, y`` of the program's
    ``contemp_conditional`` (context built by the production structure) vs ``log p(θ, y)``."""
    st, data = joint_draw(spec, seed)
    program = program_of(spec)
    S = program.structure
    n = spec.static.shapes.n
    i = n - 1
    ytil = S.pre_residual(st.glob, data, data.X @ st.glob.mean.Pi)
    ctx = S.context(jnp.int32(i), ytil, _slice(st.eq, i), program.vol, data, st.glob.vol)
    cm = S.contemp_moments(ctx, _slice(spec.prior.eq, i))
    cond = S.impl("contemp_conditional")(ctx, jnp.zeros_like(ctx.target), cm)
    K, b = np.asarray(cond.K), np.asarray(cond.b)
    a = np.asarray(st.eq.struct.a)
    r = np.random.default_rng(seed)
    errs = []
    for _ in range(4):
        xs = [np.where(np.asarray(cm.cmask) > 0, a[i] + 0.5 * r.standard_normal(n - 1), 0.0)
              for _ in range(2)]  # fmt: skip
        lq = [-0.5 * x @ K @ x + b @ x for x in xs]
        lj = [
            _lj(spec, _set_leaf(st, "eq.struct.a", jnp.asarray(a).at[i].set(x)), data) for x in xs
        ]
        errs.append(_rel(lq[0] - lq[1], lj[0] - lj[1]))
    return max(errs)


def u_gamma(spec: JointSpec, seed: int = SEED_U) -> float:
    """M8 (U, closed form): the program's stored ``γ_i`` (``extract_eq``) equals the reference D3
    map ``δ_i + M_c a_i`` (N05 §4.1)."""
    st, _ = joint_draw(spec, seed)
    mean = program_of(spec).mean
    shared = spec.prior.shared
    got = jax.vmap(lambda ms, a: mean.extract_eq(ms, a, shared).gamma)(st.eq.mean, st.eq.struct.a)
    return float(jnp.max(jnp.abs(got - structural_gamma(st, spec.prior))))


def u_fold(spec: JointSpec, seed: int = SEED_U) -> float:
    """M6 (U, closed form; D-7): the fold leaves ``h = μ + σh̃`` unchanged and returns ``σ > 0``."""
    fn = program_of(spec).vol.core.impl("nc_fold")
    r = np.random.default_rng(seed)
    sigma = -np.abs(r.standard_normal())
    ht = jnp.asarray(r.standard_normal(spec.static.shapes.T_pad + 1))
    s2, ht2 = fn(jnp.asarray(sigma), ht)
    return float(jnp.max(jnp.abs(s2 * ht2 - sigma * ht))) + float(s2 <= 0.0)


def u_nc_accept(spec: JointSpec, seed: int = SEED_U) -> float:
    """M7 (U, closed form; N01 §10.3, E6): exact-mode NC log acceptance
    ``log α_aux + w(h′) − w(h)``."""
    fn = program_of(spec).vol.core.impl("nc_accept")
    r = np.random.default_rng(seed)
    la, wp, wc = (float(x) for x in r.standard_normal(3))
    return abs(float(fn(jnp.asarray(la), jnp.asarray(wp), jnp.asarray(wc), True)) - (la + wp - wc))


def u_asis_map(spec: JointSpec, seed: int = SEED_U) -> float:
    """M15 (U, closed form; N02 §7): the NC map is a bijection, ``μ + σh̃ = h`` on every state."""
    st, data = joint_draw(spec, seed)
    core = program_of(spec).vol.core
    fn = core.impl("nc_step")
    cs = _slice(st.eq.vol.core, 0)
    prior_i = _slice(spec.prior.eq, 0)
    T_pad = spec.static.shapes.T_pad
    noise = core.draw_noise(random.key(seed, impl="threefry2x32"), T_pad, data.T_eff, prior_i)
    z = jnp.zeros((T_pad,), _F64)
    mv = fn(cs.h, cs.mu, cs.phi, cs.sigma, z, z, jnp.ones((T_pad,)), data.tmask, data.tmask,
            prior_i, noise, vol=spec.static.vol)  # fmt: skip
    live = jnp.arange(T_pad + 1) <= data.T_eff
    return float(jnp.max(jnp.where(live, jnp.abs(cs.mu + cs.sigma * mv.h_tilde - cs.h), 0.0)))


# =========================================================================== B, G and S checks


def _invariance_fn(spec: JointSpec, L: int, stat: Callable) -> Callable:
    def one(key: Array) -> tuple[Array, Array]:
        st = spec.prior_sample(part(key, Part.PRIOR))
        y = spec.simulate(part(key, Part.SIM), st)
        step_key = part(key, Part.STEP)
        out = lax.fori_loop(0, L, lambda j, s: spec.step(step_key, j.astype(jnp.uint32), s, y), st)
        ref = spec.prior_sample(part(key, Part.MC_PRIOR))
        y_ref = spec.simulate(part(key, Part.MC_SIM), ref)
        return stat(out, y), stat(ref, y_ref)

    return jax.jit(jax.vmap(one))


def invariance(spec: JointSpec, stat: Callable, *, N: int = 1000, L: int = 20) -> dict[str, Any]:
    """One-block invariance (module docstring, B) inside Gandy–Scott at ``α = 1e-3``."""
    fn = _invariance_fn(spec, L, stat)
    comp = composition_id(spec.name)
    qs: list[float] = []

    def run(stage: int) -> float:
        with _jax.x64():
            keys = rep_keys(stage_seed(SEED_B, stage), Tier.B, comp,
                            np.arange(N * stage_multiplier(stage)))  # fmt: skip
            after, ref = (np.asarray(a) for a in fn(keys))
        ps = [
            stats.ks_2samp(after[:, g], ref[:, g]).pvalue
            if np.all(np.isfinite(after[:, g])) and np.all(np.isfinite(ref[:, g])) else 0.0
            for g in range(after.shape[1])
        ]  # fmt: skip
        qs.append(bonferroni(ps))
        return qs[-1]

    seq = gandy_scott(run, 1e-3)
    return {"passed": seq.passed, "q": list(seq.q)}


def _vol_stat(spec: JointSpec) -> Callable:
    """Statistics of the volatility block: its test functions and the exact log-likelihood."""
    names = tuple(g for g in spec.all_names if ".core." in g or ".trans." in g or g == "loglik")
    idx = jnp.asarray([spec.all_names.index(g) for g in names])
    return lambda st, data: spec.quantities(st, data)[idx]


def cccm_column_sbc(spec: JointSpec) -> dict[str, Any]:
    """Block-conditional SBC (N06 §5.2, E1b) of the ``π_0`` draw of the program's ``cccm_core``:
    ``L = 99`` iid draws of ``π_0 | π_{−0}, A, λ, y`` (column 0 of one core call with fresh
    ``Z``; the other columns restored), ranks of ``π_0`` and of the exact log-likelihood
    (``N = 200``, Bonferroni at ``α = 1e-3``, Gandy–Scott). Parameter ranks alone are blind to the
    CCM-2019 error in the prior-dominated regime; the log-likelihood is not (N06 §7.3)."""
    static, prior = spec.static, spec.prior
    core = program_of(spec).global_pre[0].impl("cccm_core")
    n, k = static.shapes.n, static.shapes.k
    c = prior.eq.coef

    def joint_sample(key: Array) -> tuple[ChainState, Any]:
        st = spec.prior_sample(random.fold_in(key, 0))
        return st, spec.simulate(random.fold_in(key, 1), st)

    def conditional_draws(key: Array, st: ChainState, data: Any) -> ChainState:
        lv = effective_log_var(st.eq.vol, static)
        live = data.tmask[None, :] > 0
        inv_lam = jnp.where(live, jnp.exp(-jnp.where(live, lv, 0.0)), 0.0)
        Z = random.normal(key, (n, k), dtype=_F64)
        Pi, _ = core(st.glob.mean.Pi, st.eq.struct.a, inv_lam, data.X, data.Y, Empty(), c.mean0,
                     c.prec0, Z, gram="batched", precompute=True)  # fmt: skip
        P = st.glob.mean.Pi.at[:, 0].set(Pi[:, 0])
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=P)))

    def stat(st: ChainState, data: Any) -> Array:
        ll = loglik(st, data, prior, static)
        return jnp.concatenate([st.glob.mean.Pi[:, 0], ll[None]])

    seq, results = block_sbc_sequential(
        joint_sample, conditional_draws, stat, N=200, L=99, seed=SEED_B, alpha=1e-3,
        composition=composition_id(spec.name),
    )  # fmt: skip
    last = results[-1]
    return {"passed": seq.passed, "q": list(seq.q), "min_p": float(np.min(last.p)),
            "reasons": list(last.reasons)[:2]}  # fmt: skip


def geweke_check(spec: JointSpec) -> dict[str, Any]:
    """``geweke_sequential`` at G-small sizes; the evidence of the last stage."""
    seq, results = geweke_sequential(spec, GewekeConfig(seed=SEED_G, **G_SMALL))
    last = results[-1]
    j = int(np.argmax(np.abs(last.z)))
    return {"passed": seq.passed, "q": list(seq.q), "max_abs_z": float(np.abs(last.z[j])),
            "argmax": last.names[j], "t_crit": last.t_crit, "failing": list(last.failing)[:8],
            "n_nonfinite": int(np.sum(last.n_nonfinite))}  # fmt: skip


def sbc_check(spec: JointSpec, *, N: int = 200) -> dict[str, Any]:
    """``sbc_sequential`` (the S tier's decision rule: γ test, Bonferroni at ``α = 1e-2``,
    Gandy–Scott) with a short configuration: ``N = 200`` (``×4`` after stage 1), ``L = 99``,
    burn 100, pilot 300, ``t_max = 30``; quantities as in ``tests/sbc``."""
    names = tuple(k for k in spec.all_names if not k.startswith("data.") and "^2" not in k
                  and "*" not in k)  # fmt: skip
    idx = jnp.asarray([spec.all_names.index(k) for k in names])
    cfg = SBCConfig(N=N, L=99, burn=100, pilot=300, t_max=30, alpha=1e-2, seed=SEED_S)
    seq, results = sbc_sequential(spec, lambda st, data: spec.quantities(st, data)[idx], names, cfg)
    last = results[-1]
    j = int(np.argmin(last.p))
    return {"passed": seq.passed, "q": list(seq.q), "min_p": float(last.p[j]),
            "argmin": names[j], "reasons": list(last.reasons)[:3]}  # fmt: skip


# =========================================================================== the registry


@dataclass(frozen=True)
class Mutant:
    """One mutant: where it is injected, its configuration, its control and the confirmed catching
    tiers with their cadence (``weekly`` for the whole suite, DESIGN §9.10)."""

    id: str
    title: str
    injected: str
    design: str  # the DESIGN §9.7 "expected catcher" column
    composition: str
    mutant_kw: Callable[[], dict[str, Any]]
    control_kw: Callable[[], dict[str, Any]] = dict
    u_check: Callable[[JointSpec], float] | None = None
    u_tol: float = 1e-8
    block: str | None = None  # B: "cccm0" (π_0 block SBC) or "vol" (one-block invariance)
    catchers: tuple[str, ...] = ()  # confirmed tiers (U, B, G, S)
    evidence: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def spec(self, control: bool = False, **more: Any) -> JointSpec:
        kw = self.control_kw() if control else self.mutant_kw()
        return joint_spec(self.composition, **{**kw, **more})


def _ov(**kw: Callable[..., Any]) -> Callable[[], dict[str, Any]]:
    return lambda: {"kernel_overrides": dict(kw)}


def _m2_kw() -> dict[str, Any]:
    base = joint_spec("structural_sv", exact=False)
    return {"exact": False, "kernel_overrides": {"vol": stale_indicator_vol(base)}}


_M16_PRIOR = {"b_mu": math.log(0.01)}

_ALL: tuple[Mutant, ...] = (
    Mutant("M1", "CCM-2019 triangular step instead of CTA", "cccm_core",
           "U (density ratio), B, G", "reduced_constant", _ov(cccm_core=ccm2019_cccm_core),
           u_check=u_cccm, block="cccm0", catchers=("U", "B", "G")),
    Mutant("M2", "stale indicators (Primiceri Alg. 1: s carried across sweeps, drawn after h)",
           "vol slot (StaleIndicatorSV)", "G / divergence; tier to confirm", "structural_sv",
           _m2_kw, lambda: {"exact": False}, catchers=("G",)),
    Mutant("M3", "simulator draws fresh latents", "JointSpec(fresh_latents=True)",
           "G (Geweke error 2)", "structural_sv", lambda: {"fresh_latents": True},
           catchers=("G",)),
    Mutant("M4", "τ conditional with shape ν/2 instead of (ν+1)/2", "tau_conditional",
           "U, G", "structural_sv_t", _ov(tau_conditional=m4_tau_conditional),
           u_check=u_tau, catchers=("U", "G")),
    Mutant("M5", "B_σ × 2 in the sampler only", "sampler prior arrays", "G",
           "structural_sv", lambda: {"sampler_prior_updates": {"eq.vol_core.B_sigma": 0.2}},
           catchers=("G",)),
    Mutant("M6", "NC |σ| without flipping h̃", "nc_fold", "G (N02 T16)", "zero_sv",
           _ov(nc_fold=m6_nc_fold), u_check=u_fold, catchers=("U", "G")),
    Mutant("M7", "uncorrected NC move in exact mode", "nc_accept",
           "G (N01 §10.3 toy); tier to confirm", "zero_sv",
           _ov(nc_accept=m7_nc_accept, mixture_table=coarse_table),
           _ov(mixture_table=coarse_table), u_check=u_nc_accept, catchers=("U", "G")),
    Mutant("M8", "D3 centring dropped (γ_i = δ_i)", "structural_gamma",
           "G (a·γ cross moment), S (‖Γ − AM′‖²)", "structural_constant",
           _ov(structural_gamma=m8_structural_gamma), u_check=u_gamma, u_tol=1e-12,
           catchers=("U", "G", "S")),
    Mutant("M9", "outlier pmf without 1/k", "outlier_logits", "U, G (N03: z ≈ 392)",
           "structural_sv_outliers", _ov(outlier_logits=m9_outlier_logits),
           u_check=u_outlier, catchers=("U", "G")),
    Mutant("M10", "b_p + T in the p update", "p_conditional", "U, G",
           "structural_sv_outliers", _ov(p_conditional=m10_p_conditional), u_check=u_p,
           catchers=("U", "G")),
    Mutant("M11", "indicator pmf without σ_j⁻¹", "indicator_logits",
           "U (Bayes consistency), B", "aux_sv", _ov(indicator_logits=m11_indicator_logits),
           u_check=u_indicator, u_tol=1e-12, block="vol", catchers=("U", "B", "G")),
    Mutant("M12", "double −1.2704 shift", "mixture_table", "U (mixture mean), B", "aux_sv",
           _ov(mixture_table=m12_mixture_table), u_check=u_table, u_tol=1e-4, block="vol",
           catchers=("U", "B", "G")),
    Mutant("M13", "IG shape with T_pad instead of T_eff", "constant_conditional",
           "G (masks exercised by design)", "structural_constant",
           _ov(constant_conditional=m13_constant_conditional), u_check=u_constant,
           catchers=("U", "G")),
    Mutant("M14", "A-row regression with +u_{<j}", "contemp_conditional", "U (N04 §9.4), G",
           "reduced_constant", _ov(contemp_conditional=m14_contemp_conditional),
           u_check=u_contemp, catchers=("U", "G")),
    Mutant("M15", "ASIS map without the h̃ rescale", "nc_step",
           "B (one-block invariance), G", "zero_sv", _ov(nc_step=m15_nc_step),
           u_check=u_asis_map, u_tol=1e-10, block="vol", catchers=("U", "B", "G")),
    Mutant("M16", "offset c = 0.1 at λ ≈ 0.01 with exact=False", "SV(offset=0.1, exact=False)",
           "G (defined with exact=False only)", "zero_sv",
           lambda: {"exact": False, "prior_kwargs": {**_M16_PRIOR, "offset_rel": 0.1}},
           lambda: {"exact": False, "prior_kwargs": dict(_M16_PRIOR)}, catchers=("G",)),
    Mutant("M17", "Mode I SOC/SUR rows weighted by the current 1/λ_i, not the fixed 1/σ̄_i²",
           "linear_conditional", "WP25 (N05 §7.7): U (density ratio), G",
           "structural_constant_dummies", _ov(linear_conditional=m17_linear_conditional),
           u_check=u_dummy_equation, catchers=("U", "G")),
    Mutant("M18", "SOC/SUR rows dropped from the CCCM Π step", "cccm_core",
           "WP25 (N04 §3.1): U (density ratio), G", "reduced_constant_dummies",
           _ov(cccm_core=m18_cccm_core), u_check=u_cccm_dummies, catchers=("U", "G")),
)  # fmt: skip
#: Observed on 2026-10-22 (M1 Pro): U errors (mutant / control), G-small max |z| (mutant /
#: control; t_crit 4.1–4.5), B and S Bonferroni q or min p (mutant / control). Re-run
#: independently on x86-64 Linux (4 CPUs) the same day: all 33 mutant × tier checks caught, every
#: control passed, the statistics equal to the digits shown (M2: G 705 / 2.47). M17–M18 (WP25)
#: add four checks, recorded on the M1 Pro only.
EVIDENCE: dict[str, str] = {
    "M1": "U 1.70 / 1.6e-14; B (π_0 block SBC) loglik p 2.8e-11 / min p 0.10; G 20.8 / 3.03; "
    "one-block invariance of the whole CCCM scan: q 0.29 (not caught)",
    "M2": "G 702 (loglik) / 2.48, exact=False",
    "M3": "G 23.5 (mean |h|) / 2.59",
    "M4": "U 0.163 / 5e-15; G 281 / 2.12",
    "M5": "G 50.3 (log p(θ)) / 2.59",
    "M6": "U 1.94 / 0; G 7.4 / 2.34",
    "M7": "U 2.87 / 0; G with COARSE1 77.6 / 2.28 (OCSN-10: correction too small to see)",
    "M8": "U 0.585 / 0; G 101 / 2.77; S q 1.6e-13 / pass at stage 3 (q 0.0069, 0.065, 0.95)",
    "M9": "U 0.699 / 1e-15; G 1474 / 2.51",
    "M10": "U 0.120 / 1e-14; G 30.1 / 2.51",
    "M11": "U 0.112 / 6e-16; B q 4.9e-19 / 0.93; G (aux_sv) 86.9 / 2.18; exact mode (zero_sv): "
    "G passes, 2.39 (not a bug there)",
    "M12": "U 1.2704 / 8.3e-5 (tol 1e-4); B q 2.5e-162 / 0.93; G (aux_sv) 444 / 2.18; exact "
    "mode (zero_sv): G passes, 2.25 (not a bug there)",
    "M13": "U 1.64 / 3e-15; G 429 (loglik) / 2.77",
    "M14": "U 14.9 / 9e-15; G 325 / 3.03",
    "M15": "U 0.457 / 6e-17; B q 1.8e-14 / 1.0; G 50.5 / 2.34",
    "M16": "G 264 / 1.99 (λ ≈ 0.01, exact=False; control offset 1e-12)",
    "M17": "U 0.060 / 2.4e-15; G 23.1 (eq0.mean.delta1^2) / 2.44 (2026-10-22, WP25, M1 Pro)",
    "M18": "U 0.405 / 2.9e-14; G 99.7 (glob.mean.Pi[1,0]) / 2.61 (2026-10-22, WP25, M1 Pro)",
}
MUTANTS: dict[str, Mutant] = {m.id: dataclasses.replace(m, evidence=EVIDENCE[m.id]) for m in _ALL}

TIERS = ("U", "B", "G", "S")


def available_tiers(m: Mutant) -> tuple[str, ...]:
    """Tiers with a check for ``m`` (U needs a U check, B a block statistic; G always; S only where
    the registry lists it: SBC is the expensive tier)."""
    out = []
    if m.u_check is not None:
        out.append("U")
    if m.block is not None:
        out.append("B")
    out.append("G")
    if "S" in m.catchers:
        out.append("S")
    return tuple(out)


def run_tier(m: Mutant, tier: str) -> dict[str, Any]:
    """Run tier ``tier`` on the mutant and its control; ``caught`` iff the control passes and the
    mutant fails (U: error above / below ``u_tol``)."""
    t0 = time.perf_counter()
    if tier == "U":
        err_m, err_c = m.u_check(m.spec()), m.u_check(m.spec(control=True))
        out = {"mutant_error": err_m, "control_error": err_c, "tol": m.u_tol,
               "caught": err_c <= m.u_tol < err_m}  # fmt: skip
    elif tier == "B":
        spec_m, spec_c = m.spec(), m.spec(control=True)
        if m.block == "cccm0":
            res_m, res_c = cccm_column_sbc(spec_m), cccm_column_sbc(spec_c)
        else:
            res_m, res_c = (
                invariance(spec_m, _vol_stat(spec_m)),
                invariance(spec_c, _vol_stat(spec_c)),
            )
        out = {"mutant": res_m, "control": res_c,
               "caught": res_c["passed"] and not res_m["passed"]}  # fmt: skip
    elif tier == "G":
        res_m, res_c = geweke_check(m.spec()), geweke_check(m.spec(control=True))
        out = {"mutant": res_m, "control": res_c,
               "caught": res_c["passed"] and not res_m["passed"]}  # fmt: skip
    elif tier == "S":
        res_m, res_c = sbc_check(m.spec()), sbc_check(m.spec(control=True))
        out = {"mutant": res_m, "control": res_c,
               "caught": res_c["passed"] and not res_m["passed"]}  # fmt: skip
    else:
        raise ValueError(f"unknown tier {tier!r}")
    out.update(id=m.id, tier=tier, composition=m.composition, seconds=time.perf_counter() - t0)
    return out


__all__ = [
    "MUTANTS",
    "TIERS",
    "Mutant",
    "StaleIndicatorSV",
    "available_tiers",
    "ccm2019_cccm_core",
    "joint_draw",
    "run_tier",
    "stale_indicator_vol",
]
