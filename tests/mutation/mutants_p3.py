"""Mutation suite of Phase 3, M19–M36: the mutants, their seams and the tier checks that catch
them (docs/phase3/DESIGN.md §9.6, §9.2; WP35). Phase 2's suite (M1–M18) is ``mutants.py``.

Every mutant is injected through the seam §9.6 names, never by editing a kernel: a named kernel
internal of ``protocols.INTERNALS`` (``compose(static, overrides={name: fn})``, through
``joint_spec(..., kernel_overrides=...)``), the keyword-only ``_update`` of the conditional
forecast (M33), or the GIG sampler internal ``gig_sampler`` of the global-local step with a mutant
hat (M34, built on ``random.log_gig_attempts``' ``hat`` argument). The checks, written from the
design and the independent references of WP30 (``validation.log_joint``, ``validation.reference``):

* **U** — deterministic identities at joint draws ``(θ, y)`` of the row (P3 §9.2): density
  ratios of the conditional the program's internal returns against ``log p(θ, y)`` (Grosse &
  Duvenaud 2014 eq. 1; ``validation.identities``) for the global-local local and group lines, the
  collapsed ``(a_g, ξ_g)`` block, the DL ``(θ̃, ϱ)`` block, the SSVS indicator log odds, the
  CommonSV ``h`` blocks (``q`` from the internal ``csv_q``) and the common core ``λ_i``; the MH
  identity of the ``ρ`` move; the proposal invariance of the ``h`` blocks (§9.2 item 1: the
  Newton proposal is bitwise unchanged when the current block changes); the shared-factor draw
  against the weighted path on the same noise (``2e-12 + 5εκ(1 + t)``); A10's oracle
  (``conditional_forecast_reference``, 1e-10); the §3.5 KS family of the GIG sampler at fixed
  grid cells (D-60: ``min p > 1e-3/m``). Each check returns an error that the production
  internal keeps at or below the check's tolerance on the same draws.
* **G** — ``geweke_sequential`` at G-small sizes (``C = 1000, M_c = 100, M₁ = 2·10⁵,
  α = 1e-3``) of the row named by §9.6 on the PR design (``size="small"``), the mutant against
  the same configuration without it, with common seeds (the G-small seed of ``tests/geweke``, so
  the controls of rows 301, 311, 320, 321 are the PR runs); on the TruncatedNormal-ρ rows with
  E170(c)'s SBC-only functions (``tests/geweke/geweke_p3.py``).

A mutant is caught by a check when the check's rule fails on the mutant **and** passes on the
control. :data:`MUTANTS` records the checks confirmed by running ``test_mutants_p3.py`` and their
statistics (:data:`EVIDENCE`); §9.6 requires at least one catching tier per mutant. A11, which
§9.6 also lists for M28 and M31, lives in ``tests/analytic`` (WP32) and is not re-run here (an
open choice of WP35: U and G are run for both).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy import stats
from scipy.special import gammaln, logsumexp

from rai_bvar.components.coef import global_local as GLK
from rai_bvar.components.vol import common as CS
from rai_bvar.components.vol.robust import log_e2
from rai_bvar.engines.mcmc import random as R
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.state import ChainState, Coupling
from rai_bvar.forecast import conditional as CF
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation.geweke import GewekeConfig, geweke_sequential
from rai_bvar.validation.identities import density_ratio_error, mh_ratio_error
from rai_bvar.validation.joint import JointSpec, joint_spec
from rai_bvar.validation.reference import companion_impulse, conditional_forecast_reference

Array = jax.Array
_F64 = jnp.float64
LOG2 = math.log(2.0)
EPS = float(np.finfo(np.float64).eps)

G_SMALL: dict[str, Any] = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}
SEED_G = 20261022  # the G-small seed of tests/geweke: the controls of the PR rows are those runs
SEED_U = 20261030  # Phase 2's U seed (tests/mutation/mutants.py)
N_STATES = 3  # joint draws per U identity


def _geweke_p3() -> Any:
    """``tests/geweke/geweke_p3.py`` (the E170(c) flags of the G checks), loaded once."""
    if "geweke_p3" in sys.modules:
        return sys.modules["geweke_p3"]
    path = Path(__file__).resolve().parents[1] / "geweke" / "geweke_p3.py"
    spec = importlib.util.spec_from_file_location("geweke_p3", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["geweke_p3"] = module
    spec.loader.exec_module(module)
    return module


# =========================================================================== mutant internals


def m19_group(family, line, ell, log_r, loc, glob, info, gl):
    """M19: HS ``ζ_g | ·`` with shape ``n_g/2`` (the ``+1`` of ``(n_g + 1)/2`` dropped)."""
    p = GLK.gl_group_conditional(family, line, ell, log_r, loc, glob, info, gl)
    if family == "horseshoe" and line == "zeta":
        return GLK.IGParams(0.5 * info.n_g, p.log_scale)
    return p


def m20_local(family, line, ell, log_r, loc, grp, gl):
    """M20: HS ``ϑ_j`` scale with ``τ_g`` where ``ζ_g`` belongs (GK App. B as printed)."""
    if family == "horseshoe" and line == "theta":
        return GLK.IGParams(
            jnp.ones_like(ell),
            jnp.logaddexp(-loc.log_nu, 2.0 * ell - LOG2 - log_r - grp["log_tau"]),
        )
    return GLK.gl_local_conditional(family, line, ell, log_r, loc, grp, gl)


def m21_local(family, line, ell, log_r, loc, grp, gl):
    """M21: NG ``ϖ_j`` GIG with ``λ = a − 1`` instead of ``a − ½``."""
    p = GLK.gl_local_conditional(family, line, ell, log_r, loc, grp, gl)
    if family == "normal_gamma" and line == "w":
        return GLK.GIGParams(grp["a"] - 1.0, p.log_psi, p.log_chi)
    return p


def m22_a_block(family, keys_a, keys_xi, loc, glob, info, gl):
    """M22: NG collapsed ``a``-weights without the ``b log(2c/ã)`` term."""
    logw = GLK.a_log_weights(family, loc, info, gl)
    if family == "normal_gamma":
        logw = logw - gl.b * GLK._ng_log_beta(gl.a_support[None, :], gl)
    a_idx = jax.vmap(R.categorical)(keys_a, logw).astype(jnp.int32)
    p = GLK.xi_params(family, jnp.take(gl.a_support, a_idx), loc, info, gl)
    return a_idx, GLK._gamma(keys_xi, p), logw


def m23_local(family, line, ell, log_r, loc, grp, gl):
    """M23: R2D2 ``ϱ_j`` GIG with ``χ = d²/(rϖ)`` (the factor 2 of ``2d²/(rϖ)`` dropped)."""
    p = GLK.gl_local_conditional(family, line, ell, log_r, loc, grp, gl)
    if family == "r2d2" and line == "varrho":
        return GLK.GIGParams(p.lam, p.log_psi, p.log_chi - LOG2)
    return p


def m24_local(family, line, ell, log_r, loc, grp, gl):
    """M24: DL ``θ̃_j`` GIG with ``λ = a − ½`` instead of ``a − 1``."""
    p = GLK.gl_local_conditional(family, line, ell, log_r, loc, grp, gl)
    if family == "dirichlet_laplace" and line == "theta":
        return GLK.GIGParams(grp["a"] - 0.5, p.log_psi, p.log_chi)
    return p


def m25_logodds(ell, log_tau0, log_tau1, log_p, log1m_p):
    """M25: SSVS log odds without ``log τ₀ − log τ₁``."""
    return GLK.ssvs_logodds(ell, log_tau0, log_tau1, log_p, log1m_p) + (log_tau1 - log_tau0)


def m26_deviation(mean):
    """M26: deviation coordinates about 0 instead of the prior mean."""
    return jnp.zeros_like(mean)


def m27_local(family, line, ell, log_r, loc, grp, gl):
    """M27: the scale step's local lines with ``r_j = 1`` while the mean uses ``r_j``."""
    return GLK.gl_local_conditional(family, line, ell, jnp.zeros_like(log_r), loc, grp, gl)


def m28_newton_start(log_q, h, mask, n):
    """M28: the block's Newton started at the current block, one iteration (a state-dependent
    proposal used as an independence proposal; E152: the seam sets the start and the cap)."""
    return jnp.where(mask > 0, h, 0.0), 1


def m29_q(log_e2_, log_lam, tmask):
    """M29: ``q_t`` with ``λ_i`` instead of ``1/λ_i``."""
    return CS.csv_q(log_e2_, -jnp.asarray(log_lam, _F64), tmask)


def m30_rho_accept(rho_new, rho_old, h0, log_sigma, s_xy, s_xx, csv, uniform, kind):
    """M30: the ``ρ`` acceptance without the stationary-start factor ``g(ρ*) − g(ρ)``."""
    la = CS.csv_rho_accept(rho_new, rho_old, h0, log_sigma, s_xy, s_xx, csv, uniform, kind)
    inside = jnp.abs(rho_new) < 1.0
    rn = jnp.where(inside, rho_new, 0.0)
    g = CS.g_rho(rn, h0, log_sigma) - CS.g_rho(rho_old, h0, log_sigma)
    return jnp.where(inside, la - g, la)


def m31_lambda(theta: Coupling, rows: Coupling) -> Coupling:
    """M31: the coupled common core without the ``n_active/2`` shape term."""
    tot = theta + rows
    return Coupling(jnp.zeros_like(tot.n_active), tot.quad)


def m32_rows(w, Fd, Yd):
    """M32: the shared factor's SOC/SUR rows weighted by ``e^{−h_0}`` (``w[0]``, the first real
    row's weight) instead of the λ-free 1."""
    s = jnp.sqrt(w[0])
    return s * Fd, s * Yd


def m33_update(mu, M, Y, index, values, **kw):
    """M33: the conditional update with ``Mᵀ`` where ``M`` belongs."""
    return CF.exact_update(mu, np.swapaxes(M, 1, 2), Y, index, values, **kw)


def m34_hat(lam, log_omega):
    """M34: Devroye's tails from secants instead of tangents (a hat below ``f``; E124): the right
    tail is the chord of ``f`` through ``(t, f(t))`` and ``(t + 1, f(t + 1))`` extended beyond
    ``t'``, the left the chord through ``(−s, f(−s))`` and ``(−s − 1, f(−s − 1))`` beyond
    ``−s'``; the flat centre and the masses follow the new slopes."""
    h = R.gig_hat(lam, log_omega)

    def f(x):
        return -jnp.exp(R._gig_log_neg_f(x, h.log_alpha, h.log_lam, h.lam > 0.0))

    zeta = f(h.t) - f(h.t + 1.0)  # −(slope of the right chord)
    xi = f(-h.s) - f(-h.s - 1.0)  # slope of the left chord
    tp, sp = h.t - h.eta / zeta, h.s - h.theta / xi
    return h._replace(zeta=zeta, xi=xi, r=1.0 / zeta, p=1.0 / xi, tp=tp, sp=sp, q=tp + sp)


def m34_gig_sampler(key, lam, log_psi, log_chi):
    """M34 through the ``gig_sampler`` internal: ``random.log_gig`` with :func:`m34_hat`."""
    log_x, ok, _ = R.log_gig_attempts(key, lam, log_psi, log_chi, hat=m34_hat)
    return log_x, ok


def m35_a_block(family, keys_a, keys_xi, loc, glob, info, gl):
    """M35: ``ξ_g | a_old`` drawn before ``a_g`` (the reversed collapsed order, E30)."""
    logw = GLK.a_log_weights(family, loc, info, gl)
    log_xi = None
    if family in ("normal_gamma", "r2d2"):
        p = GLK.xi_params(family, GLK.a_values(glob, gl, info), loc, info, gl)
        log_xi = GLK._gamma(keys_xi, p)
    a_idx = jax.vmap(R.categorical)(keys_a, logw).astype(jnp.int32)
    return a_idx, log_xi, logw


def m36_lambda(theta: Coupling, rows: Coupling) -> Coupling:
    """M36: the coupled common core without the SOC/SUR rows (``n_active = q_i``, no dummy sum
    of squares; E88)."""
    return theta


# =========================================================================== helpers


def program_of(spec: JointSpec) -> Any:
    from rai_bvar.engines.mcmc.compose import compose

    return compose(spec.static, overrides=dict(spec.overrides) or None)


def global_step(program: Any, block: Block) -> Any:
    (step,) = [g for g in program.global_steps if int(g.block) == int(block)]
    return step


def joint_draws(spec: JointSpec, n: int = N_STATES, seed: int = SEED_U):
    """``n`` joint draws ``(θ, y)`` from the references (prior sampler, simulator)."""
    key = random.key(seed, impl="threefry2x32")
    out = []
    for r in range(n):
        k = random.fold_in(key, r)
        st = spec.prior_sample(random.fold_in(k, 0))
        out.append((st, spec.simulate(random.fold_in(k, 1), st)))
    return out


_LJ: dict[int, Callable[..., Array]] = {}


def log_joint_fn(spec: JointSpec) -> Callable[[ChainState, Any], float]:
    """WP30's ``log p(θ, y)`` of ``spec`` under ``jit`` (one trace per spec)."""
    if id(spec) not in _LJ:
        _LJ[id(spec)] = jax.jit(lambda st, data: spec.log_joint(st, data))
    fn = _LJ[id(spec)]
    return lambda st, data: float(fn(st, data))


def with_coef(st: ChainState, loc: Any = None, glob: Any = None) -> ChainState:
    eq = st.eq if loc is None else dataclasses.replace(st.eq, coef=loc)
    gl = st.glob if glob is None else dataclasses.replace(st.glob, coef=glob)
    return dataclasses.replace(st, eq=eq, glob=gl)


def with_vol(st: ChainState, **leaves: Any) -> ChainState:
    gv = dataclasses.replace(st.glob.vol, **{k: jnp.asarray(v) for k, v in leaves.items()})
    return dataclasses.replace(st, glob=dataclasses.replace(st.glob, vol=gv))


def log_ig(log_x, shape, scale):
    """``log IG(x; shape, scale)`` (natural scale) at ``log x``."""
    return shape * np.log(scale) - gammaln(shape) - (shape + 1.0) * log_x - scale * np.exp(-log_x)


def log_gamma(log_x, shape, log_rate):
    return shape * log_rate - gammaln(shape) + (shape - 1.0) * log_x - np.exp(log_rate + log_x)


def log_gig_kernel(log_x, lam, log_psi, log_chi):
    """``log`` of the GIG kernel ``x^{λ−1} exp(−(ψx + χ/x)/2)`` (no normaliser) at ``log x``."""
    return (lam - 1.0) * log_x - 0.5 * (np.exp(log_psi + log_x) + np.exp(log_chi - log_x))


def log_gig_half(log_x, log_psi, log_chi):
    """The normalised ``log GIG(x; ½, ψ, χ)`` (``K_½(z) = √(π/(2z)) e^{−z}``, ``z = √(ψχ)``)."""
    log_z = 0.5 * (log_psi + log_chi)
    log_norm = 0.25 * (log_psi - log_chi) - LOG2 - 0.5 * (math.log(math.pi / 2.0) - log_z)
    return log_norm + np.exp(log_z) + log_gig_kernel(log_x, 0.5, log_psi, log_chi)


def _log_density(p: Any, log_x: np.ndarray) -> np.ndarray:
    if isinstance(p, GLK.IGParams):
        return log_ig(log_x, np.asarray(p.shape), np.exp(np.asarray(p.log_scale)))
    return log_gig_kernel(log_x, np.asarray(p.lam), np.asarray(p.log_psi), np.asarray(p.log_chi))


def gl_views(spec: JointSpec, step: Any, st: ChainState):
    """``(info, on, ell, log_r, gl, grp)`` of the scale step at ``st`` (the inputs the step hands
    its internals; ``grp`` the group values broadcast to the coordinates)."""
    prior = spec.prior
    info = step.info(prior)
    on = np.asarray(info.mask) > 0
    ell = jnp.where(on, step.ell(st), 0.0)
    log_r = jnp.where(on, prior.eq.coef.gl.log_r, 0.0)
    gl = prior.shared.gl
    glob = st.glob.coef
    fam = spec.static.coef.kind
    if fam == "horseshoe":
        grp = {"log_zeta": jnp.take(glob.log_zeta, info.group),
               "log_tau": jnp.take(glob.log_tau, info.group)}  # fmt: skip
    elif fam == "ssvs":
        grp = {}
    else:
        grp = {"a": jnp.take(GLK.a_values(glob, gl, info), info.group)}
        if fam in ("normal_gamma", "r2d2"):
            grp["log_xi"] = jnp.take(glob.log_xi, info.group)
    return info, on, ell, log_r, gl, grp


_LEAF = {"theta": "log_theta", "nu": "log_nu", "w": "log_w", "varrho": "log_varrho"}


# =========================================================================== U checks


def u_gl_local(line: str) -> Callable[[JointSpec], float]:
    """U (density ratio): the local line ``line`` of every GL coordinate jointly, its IG or GIG
    from the program's ``gl_local_conditional``, against ``log p(θ, y)``."""

    def check(spec: JointSpec) -> float:
        step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
        lj = log_joint_fn(spec)
        fam = spec.static.coef.kind
        rng = np.random.default_rng(SEED_U)
        worst = 0.0
        for st, data in joint_draws(spec):
            _, on, ell, log_r, gl, grp = gl_views(spec, step, st)
            loc = st.eq.coef
            p = step.impl("gl_local_conditional")(fam, line, ell, log_r, loc, grp, gl)
            leaf = _LEAF[line]
            x0 = np.asarray(getattr(loc, leaf))
            x1 = np.where(on, x0 + 0.4 * rng.standard_normal(x0.shape), x0)

            def log_q(x, p=p, on=on):
                return float(np.sum(_log_density(p, x)[on]))

            def joint(x, st=st, data=data, leaf=leaf):
                new = dataclasses.replace(st.eq.coef, **{leaf: jnp.asarray(x)})
                return lj(with_coef(st, loc=new), data)

            worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
        return worst

    return check


def u_gl_group(line: str) -> Callable[[JointSpec], float]:
    """U (density ratio): the group line ``line`` of every group jointly
    (``gl_group_conditional``) against ``log p(θ, y)``."""
    leaf = {"zeta": "log_zeta", "tau": "log_tau", "xi": "log_xi"}[line]

    def check(spec: JointSpec) -> float:
        step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
        lj = log_joint_fn(spec)
        fam = spec.static.coef.kind
        rng = np.random.default_rng(SEED_U)
        worst = 0.0
        for st, data in joint_draws(spec):
            info, _, ell, log_r, gl, _ = gl_views(spec, step, st)
            p = step.impl("gl_group_conditional")(
                fam, line, ell, log_r, st.eq.coef, st.glob.coef, info, gl
            )
            x0 = np.asarray(getattr(st.glob.coef, leaf))
            x1 = x0 + 0.4 * rng.standard_normal(x0.shape)

            def log_q(x, p=p):
                if isinstance(p, GLK.IGParams):
                    return float(np.sum(_log_density(p, x)))
                return float(np.sum(log_gamma(x, np.asarray(p.shape), np.asarray(p.log_rate))))

            def joint(x, st=st, data=data):
                new = dataclasses.replace(st.glob.coef, **{leaf: jnp.asarray(x)})
                return lj(with_coef(st, glob=new), data)

            worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
        return worst

    return check


def u_gl_a_block(spec: JointSpec) -> float:
    """U (density ratio): the collapsed block ``(a_g, ξ_g) | ϖ`` (NG, R2D2 with a grid ``a``):
    ``P(a | ϖ)`` from the log weights the program's ``gl_a_conditional`` exposes times
    ``G(ξ; n_g a + b, β(a) + W_g)``, both moved for every group, against ``log p(θ, y)``; this
    pins the ``a``-weights, NG's ``b log(2c/ã)`` included (M22, with ``c`` fixed)."""
    step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
    lj = log_joint_fn(spec)
    fam = spec.static.coef.kind
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for r, (st, data) in enumerate(joint_draws(spec)):
        info, _, _, _, gl, _ = gl_views(spec, step, st)
        G = info.G
        key = random.key(r, impl="threefry2x32")
        keys = jax.vmap(lambda g, key=key: random.fold_in(key, g))(jnp.arange(G, dtype=jnp.uint32))
        _, _, logw = step.impl("gl_a_conditional")(fam, keys, keys, st.eq.coef, st.glob.coef,
                                                   info, gl)  # fmt: skip
        logw = np.asarray(logw)
        logw = logw - logsumexp(logw, axis=1, keepdims=True)  # log P(a_g = ã | ϖ)
        A = logw.shape[1]
        a0 = np.asarray(st.glob.coef.a_idx)
        a1 = (a0 + rng.integers(1, A, size=G)) % A
        x0 = (a0, np.asarray(st.glob.coef.log_xi))
        x1 = (a1, x0[1] + 0.4 * rng.standard_normal(G))

        def log_q(x, logw=logw, st=st, info=info, gl=gl, G=G):
            a_idx, lxi = x
            p = GLK.xi_params(fam, jnp.take(gl.a_support, jnp.asarray(a_idx)), st.eq.coef, info,
                              gl)  # fmt: skip
            lg = log_gamma(lxi, np.asarray(p.shape), np.asarray(p.log_rate))
            return float(np.sum(logw[np.arange(G), a_idx]) + np.sum(lg))

        def joint(x, st=st, data=data):
            a_idx, lxi = x
            new = dataclasses.replace(st.glob.coef, a_idx=jnp.asarray(a_idx, jnp.int32),
                                      log_xi=jnp.asarray(lxi))  # fmt: skip
            return lj(with_coef(st, glob=new), data)

        worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
    return worst


def u_dl_block(spec: JointSpec) -> float:
    """U (density ratio): the DL block ``(θ̃_j, ϱ_j) | d_j``: ``θ̃ | d`` (``ϱ`` integrated; the
    program's ``theta`` line) times the normalised ``ϱ | d, θ̃`` (its ``varrho`` line), both moved
    at every GL coordinate, against ``log p(θ, y)``."""
    step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
    lj = log_joint_fn(spec)
    fam = spec.static.coef.kind
    local = step.impl("gl_local_conditional")
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for st, data in joint_draws(spec):
        _, on, ell, log_r, gl, grp = gl_views(spec, step, st)
        loc = st.eq.coef
        p_theta = local(fam, "theta", ell, log_r, loc, grp, gl)
        x0 = (np.asarray(loc.log_theta), np.asarray(loc.log_varrho))
        x1 = tuple(np.where(on, x + 0.4 * rng.standard_normal(x.shape), x) for x in x0)

        def log_q(x, p_theta=p_theta, on=on, ell=ell, log_r=log_r, grp=grp, loc=loc, gl=gl):
            lt, lv = x
            moved = dataclasses.replace(loc, log_theta=jnp.asarray(lt))
            p_v = local(fam, "varrho", ell, log_r, moved, grp, gl)
            a = _log_density(p_theta, lt)
            b = log_gig_half(lv, np.asarray(p_v.log_psi), np.asarray(p_v.log_chi))
            return float(np.sum((a + b)[on]))

        def joint(x, st=st, data=data):
            lt, lv = x
            new = dataclasses.replace(st.eq.coef, log_theta=jnp.asarray(lt),
                                      log_varrho=jnp.asarray(lv))  # fmt: skip
            return lj(with_coef(st, loc=new), data)

        worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
    return worst


def u_ssvs(spec: JointSpec) -> float:
    """U (density ratio): the SSVS indicator of single coordinates, ``logit P(γ_j = 1 | ·)`` from
    the program's ``ssvs_logodds`` against ``log p(γ_j = 1, ·) − log p(γ_j = 0, ·)``."""
    step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
    lj = log_joint_fn(spec)
    worst = 0.0
    for st, data in joint_draws(spec):
        info, on, ell, _, _, _ = gl_views(spec, step, st)
        ssvs, glob = spec.prior.eq.coef.gl.ssvs, st.glob.coef
        logit = np.asarray(step.impl("ssvs_logodds")(
            ell, ssvs.log_tau0, ssvs.log_tau1, jnp.take(glob.log_p, info.group),
            jnp.take(glob.log1m_p, info.group),
        ))  # fmt: skip
        gamma = np.asarray(st.eq.coef.gamma)
        for i, r in list(zip(*np.nonzero(on), strict=True))[:4]:
            g0, g1 = gamma.copy(), gamma.copy()
            g0[i, r], g1[i, r] = 0, 1

            def log_q(g, i=i, r=r, logit=logit):
                return float(g[i, r] * logit[i, r])

            def joint(g, st=st, data=data):
                new = dataclasses.replace(st.eq.coef, gamma=jnp.asarray(g, jnp.int32))
                return lj(with_coef(st, loc=new), data)

            worst = max(worst, density_ratio_error(log_q, joint, g0, g1))
    return worst


def _csv_views(spec: JointSpec, step: Any, st: ChainState, data: Any):
    """``log q_t`` from WP30's residuals and the state's ``λ_i`` through the program's ``csv_q``."""
    e = LJ.residuals(st, data, spec.prior, spec.static)
    tm = jnp.asarray(data.tmask)
    return step.impl("csv_q")(log_e2(e, tm[None, :]), st.eq.vol.core.log_lam, tm)


_BLOCKS = ((2, 9), (9, 15), (0, 2), (15, 20))  # [lo, hi) inside T_eff = 20 (the PR design)


def u_newton_invariance(spec: JointSpec) -> float:
    """U (§9.2 item 1, deterministic): the ``h`` block proposal ``N(ĥ, G⁻¹)`` is bitwise unchanged
    when the current values inside the block change (the neighbours fixed). Returns the largest
    change of the mode, diagonal or off-diagonal (0 in production)."""
    step = global_step(program_of(spec), Block.G_COMMON_SV)
    n = jnp.float64(spec.static.shapes.n)
    block_of, proposal = jax.jit(step.block_of), jax.jit(step.proposal)
    worst = 0.0
    for st, data in joint_draws(spec):
        lq = _csv_views(spec, step, st, data)
        gv = st.glob.vol
        h0 = np.asarray(gv.h)
        for lo, hi in _BLOCKS:
            moved = h0.copy()
            moved[lo:hi] += np.linspace(-1.5, 2.0, hi - lo)
            props = [proposal(block_of(jnp.asarray(h), lq, gv.rho, gv.log_sigma, data.T_eff,
                                       lo, hi), n) for h in (h0, moved)]  # fmt: skip
            for f in ("mode", "diag", "off"):
                a, b = (np.asarray(getattr(p, f)) for p in props)
                worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def u_h_blocks(spec: JointSpec) -> float:
    """U (density ratio): each ``h`` block's target ``φ_B`` (P3 §5.4) with ``log q`` from the
    program's ``csv_q`` against ``log p(θ, y)`` when ``h_B`` moves."""
    step = global_step(program_of(spec), Block.G_COMMON_SV)
    lj = log_joint_fn(spec)
    n = jnp.float64(spec.static.shapes.n)
    block_of = jax.jit(step.block_of)
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for st, data in joint_draws(spec):
        lq = _csv_views(spec, step, st, data)
        gv = st.glob.vol
        for lo, hi in _BLOCKS:
            blk = block_of(gv.h, lq, gv.rho, gv.log_sigma, data.T_eff, lo, hi)
            m = np.asarray(blk.mask) > 0
            idx = int(blk.start) + np.nonzero(m)[0]
            x0 = np.asarray(blk.h)
            x1 = x0 + np.where(m, 0.4 * rng.standard_normal(x0.shape), 0.0)

            def log_q(x, blk=blk):
                return float(CS.h_block_log_target(jnp.asarray(x), blk.log_q, n, blk.diag,
                                                   blk.off, blk.lin, blk.mask))  # fmt: skip

            def joint(x, st=st, data=data, idx=idx, m=m):
                h = np.asarray(st.glob.vol.h).copy()
                h[idx] = np.asarray(x)[m]
                return lj(with_vol(st, h=h), data)

            worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
    return worst


def u_rho(spec: JointSpec) -> float:
    """U (MH identity): the ``ρ`` move's ``log α`` from the program's ``csv_rho_accept`` equals
    ``log π(ρ*) − log π(ρ) + log q(ρ) − log q(ρ*)`` with ``π`` the reference joint and ``q`` the
    ``TN(ρ̂, K_ρ⁻¹; −1, 1)`` independence proposal (three proposals per state)."""
    step = global_step(program_of(spec), Block.G_COMMON_SV)
    lj = log_joint_fn(spec)
    kind = spec.static.vol.csv_rho_prior
    csv = spec.prior.shared.csv
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for st, data in joint_draws(spec):
        gv = st.glob.vol
        s_xy, s_xx, h0 = CS.rho_sufficient(gv.h, data.T_eff)
        prop = CS.rho_proposal(s_xy, s_xx, gv.log_sigma, csv, kind)
        prec, mean, uniform = float(prop.prec), float(prop.mean), bool(prop.uniform)

        def log_alpha(r_old, r_new, h0=h0, gv=gv, s_xy=s_xy, s_xx=s_xx, prop=prop):
            return float(step.impl("csv_rho_accept")(
                jnp.float64(r_new), jnp.float64(r_old), h0, gv.log_sigma, s_xy, s_xx, csv,
                prop.uniform, kind,
            ))  # fmt: skip

        def target(r, st=st, data=data):
            return lj(with_vol(st, rho=jnp.float64(r)), data)

        def log_prop(to, frm, prec=prec, mean=mean, uniform=uniform):
            return 0.0 if uniform else -0.5 * prec * (float(to) - mean) ** 2

        for _ in range(3):
            r_new = float(step.rho_draw(prop, jnp.float64(rng.uniform(0.01, 0.99))))
            worst = max(worst, mh_ratio_error(log_alpha, target, log_prop, float(gv.rho), r_new))
    return worst


def _eq_views(program: Any, st: ChainState, data: Any, prior: Any, cache: Any, i: int):
    """One equation's ``(ctx, pm, cm, eq_i, prior_i, cache_i, e)`` as the sweep builds them."""
    s, mean, coef, vol = program.structure, program.mean, program.coef, program.vol
    take = lambda t: jax.tree.map(lambda x: x[i], t)  # noqa: E731
    eq_i, prior_i, cache_i = take(st.eq), take(prior.eq), take(cache.eq)
    ytil = s.pre_residual(st.glob, data, mean.fitted_reduced(st.glob, data))
    ctx = s.context(jnp.int32(i), ytil, eq_i, vol, data, st.glob.vol)
    cm = s.contemp_moments(ctx, prior_i)
    pm = coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
    g = mean.fitted_eq(eq_i.mean, eq_i.struct.a, ctx, cache, prior.shared)
    e = s.residual(eq_i.struct.a, ctx, g)
    return ctx, pm, cm, eq_i, prior_i, cache_i, e


def u_common_core(spec: JointSpec) -> float:
    """U (density ratio): ``λ_i | e_i, h, θ_i, rows`` of the coupled common core (the mean's
    coupling through ``csv_lambda_conditional`` and the core's IG with the common ``h``) against
    ``log p(θ, y)``, every equation (``m > 0`` SOC/SUR rows on row 320: M31's and M36's catcher)."""
    from rai_bvar.engines.mcmc.runner import _prepare_fn

    program = program_of(spec)
    lj = log_joint_fn(spec)
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for st, data in joint_draws(spec):
        cache = _prepare_fn(program)(data, spec.prior)
        tm = jnp.asarray(data.tmask)
        for i in range(spec.static.shapes.n):
            ctx, pm, cm, eq_i, prior_i, _, e = _eq_views(program, st, data, spec.prior, cache, i)
            cpl = program.mean.coupling(eq_i.mean.delta, eq_i.struct.a, ctx, pm, cm, cache=cache,
                                        shared=spec.prior.shared)  # fmt: skip
            shape, scale = program.vol.core.conditional(log_e2(e, tm), tm, data.T_eff, prior_i,
                                                        cpl, st.glob.vol)  # fmt: skip
            a, b = float(shape), float(scale)

            def log_q(ll, a=a, b=b):
                return float(log_ig(ll, a, b))

            def joint(ll, st=st, data=data, i=i):
                lam = np.asarray(st.eq.vol.core.log_lam).copy()
                lam[i] = ll
                core = dataclasses.replace(st.eq.vol.core, log_lam=jnp.asarray(lam))
                eq = dataclasses.replace(st.eq, vol=dataclasses.replace(st.eq.vol, core=core))
                return lj(dataclasses.replace(st, eq=eq), data)

            x0 = float(st.eq.vol.core.log_lam[i])
            worst = max(worst, density_ratio_error(log_q, joint, x0, x0 + 0.3 * rng.normal()))
    return worst


def u_shared_factor(spec: JointSpec) -> float:
    """U (against the weighted path; P3 §9.2 item 1): the shared-factor draw of ``(δ_i, a_i)`` on
    the same noise as the weighted path (Mode II, weights ``m_t e^{−h_t}/λ_i``, rows ``1/λ_i``),
    every equation, with ``h_t = 1.5`` on the data rows (so ``e^{−h_0} ≠ 1``). Returns the largest
    error in posterior s.d. over the tolerance ``2e-12 + 5εκ(1 + t)`` (``≤ 1`` in production)."""
    from rai_bvar.engines.mcmc.compose import compose
    from rai_bvar.engines.mcmc.runner import _prepare_fn

    program = program_of(spec)
    w_static = dataclasses.replace(
        spec.static, mean=dataclasses.replace(spec.static.mean, const_path="weighted")
    )
    w_program = compose(w_static, overrides=dict(spec.overrides) or None)
    rng = np.random.default_rng(SEED_U)
    worst = 0.0
    for st, data in joint_draws(spec):
        T_pad = st.glob.vol.h.shape[0]
        st = with_vol(st, h=np.where(np.arange(T_pad) < int(data.T_eff), 1.5, 0.0))
        cache = _prepare_fn(program)(data, spec.prior)
        w_cache = _prepare_fn(w_program)(data, spec.prior)
        for i in range(spec.static.shapes.n):
            ctx, pm, cm, _, _, cache_i, _ = _eq_views(program, st, data, spec.prior, cache, i)
            z = jnp.asarray(rng.standard_normal(spec.static.shapes.q))
            th_s, _ = program.mean.draw_core(z, ctx, pm, cm, cache_i, cache, spec.prior.shared)
            th_w, _ = w_program.mean.draw_core(z, ctx, pm, cm, cache_i, w_cache, spec.prior.shared)
            cond = w_program.mean.linear_conditional(ctx, pm, cm, w_cache, spec.prior.shared)
            act = np.asarray(jnp.concatenate([pm.cmask, cm.cmask])) > 0
            Ka = np.asarray(cond.K)[np.ix_(act, act)]
            C = np.linalg.inv(Ka)
            sd = np.sqrt(np.diag(C))
            sc = 1.0 / np.sqrt(np.diag(Ka))
            kappa = float(np.linalg.cond(sc[:, None] * Ka * sc[None, :]))
            t = float(np.max(np.abs(C @ np.asarray(cond.b)[act]) / sd))
            err = np.max(np.abs(np.asarray(th_s)[act] - np.asarray(th_w)[act]) / sd)
            worst = max(worst, float(err / (2e-12 + 5.0 * EPS * kappa * (1.0 + t))))
    return worst


def u_conditional(update: Callable[..., Any] | None) -> float:
    """U (A10): the conditional update of ``conditional.update_draws`` (the ``_update`` seam)
    against WP30's stacked companion-form oracle ``conditional_forecast_reference`` on the same
    shocks: ``E_c`` and ``Y_c`` of 4 random draws (``n = 3, p = 2, H = 5``; a conditioned
    variable ordered before a free one); the largest relative error (1e-10 in production)."""
    rng = np.random.default_rng(SEED_U)
    n, p, H, S = 3, 2, 5, 4
    lag = 0.3 * rng.standard_normal((S, p, n, n)) / n
    L = np.tril(0.3 * rng.standard_normal((S, H, n, n)), -1)
    L += np.eye(n) * rng.uniform(0.5, 1.5, (S, H, 1, n))
    mu = rng.standard_normal((S, H, n))
    eps = rng.standard_normal((S, H, n))
    table = np.full((H, n), np.nan)
    table[0, 0], table[1, 2], table[3, 1] = 0.4, -0.7, 1.1
    index = np.flatnonzero(np.isfinite(table.ravel()))
    cond = CF.Conditions(table, index, table.ravel()[index])
    cells = [(int(k // n) + 1, int(k % n), float(table.ravel()[k])) for k in index]
    paths, cmean = np.empty((S, H, n)), np.empty((S, H, n))
    for s in range(S):  # Y = μ + Mε: cond_mean_h = μ_h + Σ_{k<h} Φ_{h−k}u_k, u_k = L_kε_k
        phi = companion_impulse(lag[s], H)
        u = np.einsum("hij,hj->hi", L[s], eps[s])
        for h in range(H):
            cmean[s, h] = mu[s, h] + sum(phi[h - k] @ u[k] for k in range(h))
        paths[s] = cmean[s] + u
    ours = CF.update_draws(paths, cmean, L, lag, cond, _update=update)
    worst = 0.0
    for s in range(S):
        ref = conditional_forecast_reference(lag[s], L[s], mu[s], cells, eps=eps[s])
        for a, b in ((ours.mean[s], ref.mean), (ours.paths[s], ref.path)):
            worst = max(worst, float(np.max(np.abs(a - b) / (np.abs(b) + 1.0))))
    return worst


#: the §3.5 grid cells of M34's KS check: ``(λ, ω)`` with ``ψ = χ = ω`` (so ``x = y``)
GIG_CELLS = ((0.0, 0.5), (0.5, 1.0), (2.0, 2.0), (-1.5, 1.0))
GIG_DRAWS = 100_000


def _log_y_cdf(lam: float, omega: float) -> Callable[[np.ndarray], np.ndarray]:
    """The CDF of ``u = log y`` for ``y ~ GIG(λ, ω)`` (density ``∝ exp(λu − ω cosh u)``) by the
    trapezoid rule on a fine grid in ``u`` (log space; error far below the KS resolution)."""
    centre = math.asinh(lam / omega) if lam else 0.0
    width = 40.0 / math.sqrt(omega + abs(lam)) + 10.0
    u = np.linspace(centre - width, centre + width, 400_001)
    logf = lam * u - omega * np.cosh(np.clip(u, -700, 700))
    f = np.exp(logf - logf.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(u))])
    cdf /= cdf[-1]
    return lambda x: np.interp(x, u, cdf)


def u_gig(spec: JointSpec) -> float:
    """U (the §3.5 KS family at fixed grid cells; D-60): ``10⁵`` draws of the program's
    ``gig_sampler`` per cell of :data:`GIG_CELLS`, a KS test of ``log X`` against the log-space
    numerical CDF; the family passes iff ``min p > 1e-3/m``. Returns ``(1e-3/m)/min p`` (``≤ 1``
    passes)."""
    step = global_step(program_of(spec), Block.G_COEF_GLOBAL)
    sampler = jax.jit(jax.vmap(step.impl("gig_sampler"), in_axes=(0, None, None, None)))
    key = random.key(SEED_U, impl="threefry2x32")
    ps = []
    for c, (lam, omega) in enumerate(GIG_CELLS):
        keys = jax.vmap(lambda i, c=c: random.fold_in(random.fold_in(key, c), i))(
            jnp.arange(GIG_DRAWS, dtype=jnp.uint32)
        )
        lo = jnp.float64(math.log(omega))
        log_x, ok = sampler(keys, jnp.float64(lam), lo, lo)
        x = np.asarray(log_x)
        if not bool(np.all(np.asarray(ok))) or not np.all(np.isfinite(x)):
            ps.append(0.0)
            continue
        ps.append(stats.kstest(x, _log_y_cdf(lam, omega)).pvalue)
    m = len(GIG_CELLS)
    return (1e-3 / m) / max(min(ps), 1e-300)


# =========================================================================== G checks


def g_spec(composition: str, **kw: Any) -> JointSpec:
    """The G check's joint model: the row on the PR design with E170(c)'s SBC-only functions."""
    return _geweke_p3().p3_spec(composition, "small", **kw)


def geweke_check(spec: JointSpec) -> dict[str, Any]:
    """``geweke_sequential`` at G-small sizes; the evidence of the last stage."""
    seq, results = geweke_sequential(spec, GewekeConfig(seed=SEED_G, **G_SMALL))
    last = results[-1]
    j = int(np.argmax(np.abs(last.z)))
    return {"passed": seq.passed, "q": list(seq.q), "max_abs_z": float(np.abs(last.z[j])),
            "argmax": last.names[j], "t_crit": last.t_crit, "G": len(last.names),
            "failing": list(last.failing)[:8],
            "n_nonfinite": int(np.sum(last.n_nonfinite))}  # fmt: skip


# =========================================================================== the registry


@dataclass(frozen=True)
class Check:
    """One catching check of a mutant: tier ``U`` (``u_check`` on the row ``composition``, or on
    nothing when ``composition`` is ``None``) or ``G`` (G-small of ``composition``)."""

    tier: str
    composition: str | None = None
    u_check: Callable[..., float] | None = None
    u_tol: float = 1e-8

    @property
    def label(self) -> str:
        return self.tier if self.composition is None else f"{self.tier}:{self.composition}"


@dataclass(frozen=True)
class Mutant:
    """One mutant: its seam, the §9.6 expected catcher, the internals it replaces and the checks
    confirmed to catch it (:data:`EVIDENCE` records their statistics)."""

    id: str
    title: str
    injected: str
    design: str  # the §9.6 "expected catcher" column
    internals: dict[str, Callable[..., Any]] = field(default_factory=dict)
    checks: tuple[Check, ...] = ()
    evidence: str = ""

    def spec(self, composition: str, control: bool = False, *, g: bool = False) -> JointSpec:
        kw = {} if control or not self.internals else {"kernel_overrides": dict(self.internals)}
        return g_spec(composition, **kw) if g else joint_spec(composition, "small", **kw)


def _u(composition: str | None, check: Callable[..., float], tol: float = 1e-8) -> Check:
    return Check("U", composition, check, tol)


def _g(composition: str) -> Check:
    return Check("G", composition)


#: the rows of the checks (P3 §9.4 names; §9.6's catching rows)
R301, R303 = "structural_constant_hs", "structural_constant_ng"
R304, R305 = "structural_constant_ng_grid", "structural_constant_r2d2"
R306, R307 = "structural_constant_r2d2_grid", "structural_constant_dl"
R309, R326 = "structural_constant_ssvs", "structural_constant_ng_grid_c1"
R320, R321 = "conjugate_commonsv_dummies", "structural_commonsv"

_ALL: tuple[Mutant, ...] = (
    Mutant("M19", "HS ζ_g shape n_g/2 (the +1 dropped)", "gl_group_conditional",
           "U (density ratio), G", {"gl_group_conditional": m19_group},
           (_u(R301, u_gl_group("zeta")), _g(R301))),
    Mutant("M20", "HS ϑ_j scale with τ_g where ζ_g belongs (GK App. B as printed)",
           "gl_local_conditional", "U, G", {"gl_local_conditional": m20_local},
           (_u(R301, u_gl_local("theta")), _g(R301))),
    Mutant("M21", "NG ϖ_j GIG with λ = a − 1", "gl_local_conditional", "U, G",
           {"gl_local_conditional": m21_local}, (_u(R303, u_gl_local("w")), _g(R303))),
    Mutant("M22", "NG collapsed a-weight without b log(2c/ã)", "gl_a_conditional",
           "U (fixed c), G (row 326)", {"gl_a_conditional": m22_a_block},
           (_u(R326, u_gl_a_block), _g(R326))),
    Mutant("M23", "R2D2 ϱ_j GIG with χ = d²/(rϖ) (factor 2 dropped)", "gl_local_conditional",
           "U, G", {"gl_local_conditional": m23_local},
           (_u(R305, u_gl_local("varrho")), _g(R305))),
    Mutant("M24", "DL θ̃_j GIG with λ = a − ½", "gl_local_conditional", "U, G",
           {"gl_local_conditional": m24_local}, (_u(R307, u_dl_block), _g(R307))),
    Mutant("M25", "SSVS log odds without log τ₀ − log τ₁", "ssvs_logodds", "U, G",
           {"ssvs_logodds": m25_logodds}, (_u(R309, u_ssvs), _g(R309))),
    Mutant("M26", "deviation about 0 instead of the prior mean", "gl_deviation",
           "G (own-lag mean 0.4)", {"gl_deviation": m26_deviation}, (_g(R301),)),
    Mutant("M27", "scale step with r_j = 1 while the mean uses r_j", "gl_local_conditional",
           "U, G", {"gl_local_conditional": m27_local},
           (_u(R301, u_gl_local("theta")), _g(R301))),
    Mutant("M28", "CommonSV Newton started at the current block, one iteration "
           "(state-dependent proposal)", "csv_newton_start",
           "U (the proposal-invariance test of §9.2 item 1; required, deterministic), A11, G",
           {"csv_newton_start": m28_newton_start},
           (_u(R321, u_newton_invariance, 0.0), _g(R321))),
    Mutant("M29", "q_t with λ_i instead of 1/λ_i", "csv_q", "U, G", {"csv_q": m29_q},
           (_u(R321, u_h_blocks), _g(R321))),
    Mutant("M30", "ρ acceptance without the stationary-start factor", "csv_rho_accept", "U, G",
           {"csv_rho_accept": m30_rho_accept}, (_u(R321, u_rho, 1e-10), _g(R321))),
    Mutant("M31", "coupled common core without the n_active/2 shape term",
           "csv_lambda_conditional (program.mean.impl, E171)", "U, G (row 320), A11",
           {"csv_lambda_conditional": m31_lambda}, (_u(R320, u_common_core), _g(R320))),
    Mutant("M32", "shared-factor dummy rows weighted by e^{−h_0}", "shared_factor_rows",
           "U (against the weighted path), G (row 320)", {"shared_factor_rows": m32_rows},
           (_u(R320, u_shared_factor, 1.0), _g(R320))),
    Mutant("M33", "conditional update with Mᵀ where M belongs", "_update", "U (A10)", {},
           (Check("U", None, lambda: u_conditional(m33_update), 1e-10),)),
    Mutant("M34", "GIG tails from secants instead of tangents (a hat below f)", "gig_sampler",
           "U (the §3.5 KS family or E log X at some grid cell)",
           {"gig_sampler": m34_gig_sampler}, (_u(R303, u_gig, 1.0),)),
    Mutant("M35", "ξ_g | a_old drawn before a_g (the reversed collapsed order, E30)",
           "gl_a_conditional", "G (joint test function a_g·log ξ_g, rows 304, 306)",
           {"gl_a_conditional": m35_a_block}, (_g(R304), _g(R306))),
    Mutant("M36", "coupled common core without the dummy rows (n_active = q_i, no dummy sum "
           "of squares; E88)", "csv_lambda_conditional (program.mean.impl, E171)",
           "U (density ratio at m > 0), G (row 320)", {"csv_lambda_conditional": m36_lambda},
           (_u(R320, u_common_core), _g(R320))),
)  # fmt: skip

#: the statistics of every confirmed check (mutant / control), filled from the runs of
#: ``test_mutants_p3.py`` (U: error / control error against the tolerance; G: max |z| / control)
EVIDENCE: dict[str, str] = {m.id: "" for m in _ALL}
MUTANTS: dict[str, Mutant] = {m.id: dataclasses.replace(m, evidence=EVIDENCE[m.id]) for m in _ALL}


_G_CONTROLS: dict[str, dict[str, Any]] = {}


def run_check(m: Mutant, k: int) -> dict[str, Any]:
    """Run check ``k`` of ``m`` on the mutant and its control; ``caught`` iff the control passes
    and the mutant fails (U: error at or below / above ``u_tol``)."""
    c = m.checks[k]
    t0 = time.perf_counter()
    if c.tier == "U":
        if c.composition is None:  # M33: the seam is a function argument, not a kernel internal
            err_m, err_c = c.u_check(), u_conditional(None)
        else:
            err_m = c.u_check(m.spec(c.composition))
            err_c = c.u_check(m.spec(c.composition, control=True))
        out = {"mutant_error": err_m, "control_error": err_c, "tol": c.u_tol,
               "caught": bool(err_c <= c.u_tol < err_m)}  # fmt: skip
    elif c.tier == "G":
        res_m = geweke_check(m.spec(c.composition, g=True))
        if c.composition not in _G_CONTROLS:  # deterministic: one control run per row and process
            _G_CONTROLS[c.composition] = geweke_check(m.spec(c.composition, control=True, g=True))
        res_c = _G_CONTROLS[c.composition]
        out = {"mutant": res_m, "control": res_c,
               "caught": res_c["passed"] and not res_m["passed"]}  # fmt: skip
    else:
        raise ValueError(f"unknown tier {c.tier!r}")
    out.update(id=m.id, check=c.label, tier=c.tier, composition=c.composition,
               seconds=time.perf_counter() - t0)  # fmt: skip
    return out


__all__ = [
    "EVIDENCE",
    "MUTANTS",
    "Check",
    "Mutant",
    "geweke_check",
    "run_check",
]
