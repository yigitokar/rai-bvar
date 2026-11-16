"""Global-local and SSVS coefficient priors and their scale step (P3 DESIGN §3; WP31).

Tier U (P3 §9.2):

* item 2: **density-ratio identities** at 20 random states for every line of §3.2 (local, group,
  the collapsed ``(a_g, ξ_g)`` block and ``ξ | a`` alone, DL's ``(θ̃, ϱ)`` block and ``ϱ`` alone,
  SSVS ``γ`` and ``p``) and for the deviation draw (structural ``(δ_i, a_i)`` and CCCM ``π_j``):
  ``log q(x₁) − log q(x₀) = log p(x₁, y) − log p(x₀, y)`` against WP30's ``validation.log_joint``
  (relative 1e-8; natural coordinates on both sides). The conditional densities below are written
  from the distributions the kernels' parameters name (IG, GIG, gamma, Beta, categorical), never
  from the kernels;
* item 3: the **closed forms**: the collapsed ``a``-weights against numerical integration over ξ
  (adaptive quadrature, ``c`` fixed and tied; 1e-9), DL ``θ̃ | d`` against quadrature of the joint over ϱ
  (1e-8), the SSVS log odds against brute force (1e-12);
* infrastructure: protocols, keys, the cold start (P3 §3.7), plan invariants with the write-set
  tree diff, and the exact zeros off the GL slots.

Tier B (P3 §9.3): per-PR block Geweke of 401–405 (``kernel="global_step"``) and 406
(``kernel="sweep"``) through ``validation.joint`` at the P2 §9.5 G-small sizes.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from jax import random
from jax.scipy.special import gammaln

from rai_bvar.components.coef import global_local as GLK
from rai_bvar.components.mean.linear_reduced import CCCMStep
from rai_bvar.engines.mcmc import keys as K
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.keys import Block, Sub
from rai_bvar.engines.mcmc.plan import build_plan, check_invariants
from rai_bvar.engines.mcmc.protocols import GlobalStep
from rai_bvar.engines.mcmc.runner import _prepare_fn
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    LinearEq,
    RecursiveEq,
    SweepCtx,
    check_state_invariants,
)
from rai_bvar.engines.mcmc.sweep import sweep_values
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.identities import density_ratio_error
from rai_bvar.validation.joint import joint_spec

KEY = random.key(3141)
N_STATES = 20
LOG2 = math.log(2.0)
G_SMALL = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}

ROWS = {
    "horseshoe": "structural_constant_hs",
    "ng_fixed": "structural_constant_ng",
    "ng_grid": "structural_constant_ng_grid",
    "ng_grid_c1": "structural_constant_ng_grid_c1",
    "r2d2_fixed": "structural_constant_r2d2",
    "r2d2_grid": "structural_constant_r2d2_grid",
    "dl_fixed": "structural_constant_dl",
    "dl_grid": "structural_constant_dl_grid",
    "ssvs_fixed": "structural_constant_ssvs",
    "ssvs_beta": "structural_constant_ssvs_beta",
}


# --------------------------------------------------------------------------- reference densities
# Log densities in the natural coordinates, from the distributions' definitions (up to constants
# that cancel in a ratio at fixed parameters; normalised where the parameters move with the block).


def log_ig(x, shape, scale):
    return shape * np.log(scale) - gammaln(shape) - (shape + 1.0) * np.log(x) - scale / x


def log_gamma(x, shape, rate):
    return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * np.log(x) - rate * x


def log_gig_kernel(x, lam, psi, chi):
    return (lam - 1.0) * np.log(x) - 0.5 * (psi * x + chi / x)


def log_gig_half(x, psi, chi):
    """Normalised ``GIG(½, ψ, χ)``: ``K_{½}(z) = √(π/(2z)) e^{−z}``, ``z = √(ψχ)``."""
    z = np.sqrt(psi * chi)
    log_k = 0.5 * np.log(np.pi / (2.0 * z)) - z
    return 0.25 * np.log(psi / chi) - LOG2 - log_k + log_gig_kernel(x, 0.5, psi, chi)


def log_beta_pdf(p, a, b):
    return (
        (a - 1.0) * np.log(p)
        + (b - 1.0) * np.log1p(-p)
        - (gammaln(a) + gammaln(b) - gammaln(a + b))
    )


# --------------------------------------------------------------------------- helpers


def spec_of(case: str, size: str = "full"):
    return joint_spec(ROWS.get(case, case), size)


def step_of(spec) -> GLK.GLScaleStep:
    """The production scale step of ``spec``; the tests read its conditionals through the
    overridable internals (``impl``), so a mutant injected there (P3 §9.6) reaches them."""
    form = "reduced" if spec.static.mean.kind == "linear_reduced" else "structural"
    return GLK.GLScaleStep(spec.static.coef, form=form, overrides=spec.overrides)


def draw_state(spec, rep: int):
    k = random.fold_in(KEY, 1000 + rep)
    st = spec.prior_sample(random.fold_in(k, 0))
    data = spec.simulate(random.fold_in(k, 1), st)
    return st, data


def views(spec, st):
    """``(info, on, ell, log_r, gl)`` of the scale step at the state ``st``."""
    step = step_of(spec)
    prior = spec.prior
    info = step.info(prior)
    on = np.asarray(info.mask) > 0
    ell = np.where(on, np.asarray(step.ell(st)), 0.0)
    log_r = np.where(on, np.asarray(prior.eq.coef.gl.log_r), 0.0)
    return info, on, ell, log_r, prior.shared.gl


def with_coef(st, loc=None, glob=None):
    eq = st.eq if loc is None else dataclasses.replace(st.eq, coef=loc)
    gl = st.glob if glob is None else dataclasses.replace(st.glob, coef=glob)
    return dataclasses.replace(st, eq=eq, glob=gl)


def perturb(rng, x, on, sd=0.4):
    return np.where(on, x + sd * rng.standard_normal(np.shape(x)), x)


def check_ratio(log_q, joint, x0, x1, worst):
    err = density_ratio_error(log_q, joint, x0, x1)
    return max(worst, err)


def bcast(v, info):
    return np.asarray(v)[np.asarray(info.group)]


def exact_dev(x, x_state, dev_state, mean):
    """``log|x − mean|``, the state's exact ``dev_log`` where ``x`` is the state's value (a
    deviation below the resolution of ``mean + d`` is exact only there, P3 §3.4)."""
    x, x_state = np.asarray(x), np.asarray(x_state)
    with np.errstate(divide="ignore"):
        fresh = np.log(np.abs(x - mean))
    return np.where(x == x_state, np.asarray(dev_state), fresh)


def a_of(glob, gl, info):
    return np.asarray(GLK.a_values(glob, gl, info))


# --------------------------------------------------------------------------- protocol and keys


@pytest.mark.parametrize("kind", ["horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"])
def test_kernels_and_scale_step_protocol(kind):
    """Registry classes, ``varies``, ``has_local_state = False`` (D-40), and the scale step's
    block, phase, reads/writes (P3 §3.3) and the overridable internals (P3 §2.4)."""
    from rai_bvar.engines.mcmc import registry

    cls = registry.load(kind)
    assert cls.varies is True and cls.has_local_state is False
    case = {"horseshoe": "horseshoe", "normal_gamma": "ng_grid", "r2d2": "r2d2_grid",
            "dirichlet_laplace": "dl_grid", "ssvs": "ssvs_beta"}[kind]  # fmt: skip
    spec = spec_of(case)
    prior = cls.from_static(spec.static)
    (step,) = prior.global_steps()
    assert isinstance(step, GLK.GLScaleStep) and isinstance(step, GlobalStep)
    assert registry.load("gl_scales") is GLK.GLScaleStep
    assert step.block is Block.G_COEF_GLOBAL and step.phase == "post"
    assert step.reads == frozenset({"eq.mean", "glob.mean", "eq.coef", "glob.coef"})
    assert step.writes == frozenset({"eq.coef", "glob.coef"})
    assert step.marginal_of_indicators is True
    for name in ("gl_local_conditional", "gl_group_conditional", "gl_a_conditional",
                 "ssvs_logodds", "gig_sampler"):  # fmt: skip
        assert callable(step.impl(name))
    seen = prior.with_overrides({"ssvs_logodds": GLK.ssvs_logodds}).global_steps()[0]
    assert dict(seen.overrides)["ssvs_logodds"] is GLK.ssvs_logodds


def test_cccm_reads_include_the_coefficient_prior():
    """``CCCMStep.reads`` gains ``eq.coef`` and ``glob.coef`` (P3 §2.4, E70); its writes add
    ``glob.mean.dev_log`` only on the deviation path."""
    spec = spec_of("reduced_sv_hs_global")
    step = CCCMStep.from_static(spec.static)
    assert {"eq.coef", "glob.coef"} <= step.reads
    assert step.writes == frozenset({"glob.mean.Pi", "glob.mean.dev_log"})
    st, _ = draw_state(spec, 0)
    with pytest.raises(ValueError, match="log_prior_state"):  # never the fixed-moment form
        step.log_prior(st.glob, spec.prior)


PLAN_CASES = ["horseshoe", "ng_grid", "r2d2_grid", "dl_grid", "ssvs_beta",
              "reduced_sv_hs_global", "reduced_sv_ssvs_global"]  # fmt: skip


@pytest.mark.parametrize("case", PLAN_CASES)
def test_plan_invariants_and_write_set(case):
    """The plan (no ``coef_local`` step; ``gl_scales`` last) and the write-set tree diff of every
    step at a prior state (P2 §4.8 item 4; P3 §9.2 item 4)."""
    spec = spec_of(case, "small")
    program = compose(spec.static)
    plan = build_plan(program)
    assert "coef_local" not in plan.names and plan.names[-1] == "gl_scales"
    st, data = draw_state(spec, 0)
    cache = _prepare_fn(program)(data, spec.prior)
    check_invariants(plan, program, probe=(KEY, jnp.uint32(7), st, data, spec.prior, cache))


def _scale_step_run(spec, st, key, overrides=()):
    step = step_of(spec)
    if overrides:
        step = step.with_overrides(dict(overrides))
    ctx = SweepCtx(None, spec.prior, None, None)
    return step.sample(key, st, ctx, jax.vmap)


@pytest.mark.parametrize("case", ["horseshoe", "ng_grid", "r2d2_grid", "dl_grid", "ssvs_beta"])
def test_scale_step_keys_invariants_and_padding(case):
    """Determinism by key, a key change moves every family's draws, the local scales and ``γ``
    stay exactly 0 off the GL slots (P3 §2.2), ``a_idx`` in range, ``ok``."""
    spec = spec_of(case)
    st, _ = draw_state(spec, 1)
    run = jax.jit(lambda s, k: _scale_step_run(spec, s, k))
    a, info_a = run(st, random.fold_in(KEY, 1))
    b, _ = run(st, random.fold_in(KEY, 1))
    c, _ = run(st, random.fold_in(KEY, 2))
    assert bool(info_a.ok)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    moved = [
        not np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(a.eq.coef), jax.tree.leaves(c.eq.coef), strict=True)
    ]
    assert any(moved)
    assert check_state_invariants(a, spec.static, int(spec.fixed.T_eff)) == []
    n_det = spec.static.shapes.n_det
    for leaf in jax.tree.leaves(a.eq.coef):
        assert not np.asarray(leaf)[:, :n_det].any()


def test_per_coordinate_and_per_group_key_folding():
    """Per-coordinate draws fold the flat coordinate index ``i·k + r`` into ``Sub.GL_*`` of the
    step key; per-group draws the group id (P3 §2.4)."""
    base = K.sub_key(KEY, Sub.GL_LOCAL)
    ks = GLK._coord_keys(KEY, Sub.GL_LOCAL, 6)
    for j in (0, 5):
        np.testing.assert_array_equal(
            random.key_data(ks[j]), random.key_data(random.fold_in(base, np.uint32(j)))
        )
    gk = GLK._group_keys(KEY, Sub.GL_A, 3)
    np.testing.assert_array_equal(
        random.key_data(gk[2]),
        random.key_data(random.fold_in(K.sub_key(KEY, Sub.GL_A), np.uint32(2))),
    )


# --------------------------------------------------------------------------- initialisation


@pytest.mark.parametrize("case", ["horseshoe", "ng_grid", "ng_fixed", "r2d2_grid", "dl_grid",
                                  "ssvs_beta", "ssvs_fixed"])  # fmt: skip
def test_init_global(case):
    """P3 §3.7: ``r_j v_j`` at the Mode I own-lag-1 variance ``0.04 r_j`` (before the group
    jitter; HS, NG, R2D2), ``a_g`` at the grid point nearest 0.1, the locals at their unit
    multipliers, DL ``θ̃ = ϱ = 2``, SSVS ``γ = 1``, ``p = 0.5`` (Beta) or the fixed ``p``; each
    group log scale jittered by ``N(0, 0.5²)``; exact zeros off the GL slots."""
    spec = spec_of(case)
    prior = spec.prior
    kern = GLK.__dict__[{"h": "Horseshoe", "n": "NormalGamma", "r": "R2D2", "d": "DirichletLaplace",
                        "s": "SSVS"}[case[0]]].from_static(spec.static)  # fmt: skip
    st0, _ = draw_state(spec, 2)
    st = jax.jit(lambda s: kern.init_global(random.fold_in(KEY, 9), s, None, prior))(st0)
    info = step_of(spec).info(prior)
    on = np.asarray(info.mask) > 0
    gl, g, loc = prior.shared.gl, st.glob.coef, st.eq.coef
    G = spec.static.coef.n_groups
    if spec.static.coef.a_grid:
        sup = np.asarray(gl.a_support)
        assert np.all(np.asarray(g.a_idx) == np.argmin(np.abs(sup - 0.1)))
    log_v = np.asarray(kern.log_kernel_var(loc, prior.eq.coef.gl, g)) - np.asarray(
        prior.eq.coef.gl.log_r
    )
    fam = spec.static.coef.kind
    if fam == "horseshoe":
        jit = np.asarray(g.log_zeta) - math.log(0.04)
        np.testing.assert_allclose(log_v[on], (math.log(0.04) + bcast(jit, info))[on], atol=1e-14)
        assert np.all(np.asarray(loc.log_nu)[on] == 0.0)
        jit_tau = np.asarray(g.log_tau) + 2.0 * np.log(float(gl.global_scale))
        assert 0.0 < np.std(np.concatenate([jit, jit_tau])) < 1.5 and G == jit.size
    elif fam in ("normal_gamma", "r2d2"):
        a = a_of(g, gl, info)
        jit = np.asarray(g.log_xi) - (np.log(a) - math.log(0.04))
        np.testing.assert_allclose(log_v[on], (math.log(0.04) - bcast(jit, info))[on], atol=1e-13)
        if fam == "r2d2":
            np.testing.assert_allclose(np.asarray(loc.log_varrho)[on], LOG2, atol=0)
    elif fam == "dirichlet_laplace":
        assert np.all(np.asarray(loc.log_theta)[on] == LOG2)
        assert np.all(np.asarray(loc.log_varrho)[on] == LOG2)
    else:
        assert np.all(np.asarray(loc.gamma)[on] == 1)
        p = 0.5 if spec.static.coef.p_beta else float(gl.p_fixed)
        np.testing.assert_allclose(np.exp(np.asarray(g.log_p)), p, rtol=1e-15)
        np.testing.assert_allclose(np.exp(np.asarray(g.log1m_p)), 1.0 - p, rtol=1e-15)
    for leaf in jax.tree.leaves(loc):
        assert not np.asarray(leaf)[~on].any()


# --------------------------------------------------------------------------- density ratios (item 2)


def _local_ratio_cases():
    return [
        ("horseshoe", "theta"), ("horseshoe", "nu"), ("ng_fixed", "w"), ("ng_grid", "w"),
        ("r2d2_fixed", "varrho"), ("r2d2_grid", "w"), ("dl_grid", "varrho"),
    ]  # fmt: skip


_LEAF = {"theta": "log_theta", "nu": "log_nu", "w": "log_w", "varrho": "log_varrho"}


@pytest.mark.parametrize("case,line", _local_ratio_cases())
def test_density_ratio_local_lines(case, line):
    """Every local line: the conditional (IG or GIG, from the kernel's parameters) of all GL
    coordinates jointly against the reference joint (P3 §9.2 item 2; 20 states, 1e-8)."""
    spec = spec_of(case)
    worst = 0.0
    rng = np.random.default_rng(11)
    fam = spec.static.coef.kind
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        info, on, ell, log_r, gl = views(spec, st)
        loc, glob = st.eq.coef, st.glob.coef
        grp = {"a": bcast(a_of(glob, gl, info), info)}
        if fam == "horseshoe":
            grp = {"log_zeta": bcast(glob.log_zeta, info), "log_tau": bcast(glob.log_tau, info)}
        elif fam in ("normal_gamma", "r2d2"):
            grp["log_xi"] = bcast(glob.log_xi, info)
        p = step_of(spec).impl("gl_local_conditional")(fam, line, ell, log_r, loc, grp, gl)
        leaf = _LEAF[line]
        x0 = np.asarray(getattr(loc, leaf))
        x1 = perturb(rng, x0, on)

        def log_q(x, p=p, on=on):
            v = np.exp(x[on])
            if isinstance(p, GLK.IGParams):
                return np.sum(
                    log_ig(v, np.asarray(p.shape)[on], np.exp(np.asarray(p.log_scale)[on]))
                )
            lam = np.broadcast_to(np.asarray(p.lam), on.shape)[on]
            psi, chi = np.exp(np.asarray(p.log_psi)[on]), np.exp(np.asarray(p.log_chi)[on])
            return np.sum(log_gig_kernel(v, lam, psi, chi))

        def joint(x, st=st, data=data, leaf=leaf):
            s = with_coef(st, loc=dataclasses.replace(st.eq.coef, **{leaf: jnp.asarray(x)}))
            return float(spec.log_joint(s, data))

        worst = check_ratio(log_q, joint, x0, x1, worst)
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


@pytest.mark.parametrize("case,line", [("horseshoe", "zeta"), ("horseshoe", "tau"),
                                       ("ng_fixed", "xi"), ("ng_grid", "xi"),
                                       ("ng_grid_c1", "xi"), ("r2d2_grid", "xi")])  # fmt: skip
def test_density_ratio_group_lines(case, line):
    """Every group line (HS ``ζ``, ``τ``; ``ξ | a`` with ``c`` fixed, tied and 1) of all groups
    jointly against the reference joint (20 states, 1e-8)."""
    spec = spec_of(case)
    fam = spec.static.coef.kind
    worst = 0.0
    rng = np.random.default_rng(12)
    leaf = {"zeta": "log_zeta", "tau": "log_tau", "xi": "log_xi"}[line]
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        info, _on, ell, log_r, gl = views(spec, st)
        p = step_of(spec).impl("gl_group_conditional")(
            fam, line, ell, log_r, st.eq.coef, st.glob.coef, info, gl
        )
        x0 = np.asarray(getattr(st.glob.coef, leaf))
        x1 = x0 + 0.4 * rng.standard_normal(x0.shape)

        def log_q(x, p=p):
            v = np.exp(x)
            if isinstance(p, GLK.IGParams):
                return np.sum(log_ig(v, np.asarray(p.shape), np.exp(np.asarray(p.log_scale))))
            return np.sum(log_gamma(v, np.asarray(p.shape), np.exp(np.asarray(p.log_rate))))

        def joint(x, st=st, data=data):
            s = with_coef(st, glob=dataclasses.replace(st.glob.coef, **{leaf: jnp.asarray(x)}))
            return float(spec.log_joint(s, data))

        worst = check_ratio(log_q, joint, x0, x1, worst)
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


@pytest.mark.parametrize("case", ["ng_grid", "ng_grid_c1", "r2d2_grid"])
def test_density_ratio_collapsed_a_xi_block(case):
    """The collapsed block ``(a_g, ξ_g) | ϖ``: ``P(a | ϖ, ξ integrated)·p(ξ | a, ϖ)`` is the exact
    joint conditional of the pair, so moving both against the reference joint pins the
    ``a``-weights including NG's ``b log(2c/ã)`` (mutant M22) (20 states, 1e-8)."""
    spec = spec_of(case)
    fam = spec.static.coef.kind
    worst = 0.0
    rng = np.random.default_rng(13)
    A = spec.static.coef.a_grid
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        info, _on, _ell, _log_r, gl = views(spec, st)
        loc, glob = st.eq.coef, st.glob.coef
        G = info.G
        _, _, logits = step_of(spec).impl("gl_a_conditional")(
            fam, GLK._group_keys(KEY, Sub.GL_A, G), GLK._group_keys(KEY, Sub.GL_GROUP, G), loc,
            glob, info, gl,
        )  # fmt: skip
        logits = np.asarray(logits)
        logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
        sup = np.asarray(gl.a_support)
        idx0, xi0 = np.asarray(glob.a_idx), np.asarray(glob.log_xi)
        idx1 = (idx0 + rng.integers(1, A, idx0.shape)) % A
        xi1 = xi0 + 0.4 * rng.standard_normal(xi0.shape)

        def log_q(x, loc=loc, info=info, gl=gl, logp=logp, sup=sup):
            idx, lxi = x
            p = GLK.xi_params(fam, jnp.asarray(sup[idx]), loc, info, gl)
            g = np.arange(idx.size)
            return float(np.sum(logp[g, idx]) + np.sum(
                log_gamma(np.exp(lxi), np.asarray(p.shape), np.exp(np.asarray(p.log_rate)))
            ))  # fmt: skip

        def joint(x, st=st, data=data):
            idx, lxi = x
            g = dataclasses.replace(
                st.glob.coef, a_idx=jnp.asarray(idx, jnp.int32), log_xi=jnp.asarray(lxi)
            )
            return float(spec.log_joint(with_coef(st, glob=g), data))

        worst = check_ratio(log_q, joint, (idx0, xi0), (idx1, xi1), worst)
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


def test_density_ratio_dl_block_and_a():
    """DL: the block ``(θ̃_j, ϱ_j) | d`` (``θ̃ | d`` with ϱ integrated, then the normalised
    ``GIG(½, 1, d²/(rθ̃²))``), ``ϱ`` alone, and ``a_g | θ̃`` (20 states each, 1e-8)."""
    spec = spec_of("dl_grid")
    fam = "dirichlet_laplace"
    rng = np.random.default_rng(14)
    worst = {"block": 0.0, "a": 0.0}
    A = spec.static.coef.a_grid
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        info, on, ell, log_r, gl = views(spec, st)
        loc, glob = st.eq.coef, st.glob.coef
        grp = {"a": bcast(a_of(glob, gl, info), info)}
        local = step_of(spec).impl("gl_local_conditional")
        pt = local(fam, "theta", ell, log_r, loc, grp, gl)
        th0, vr0 = np.asarray(loc.log_theta), np.asarray(loc.log_varrho)
        th1, vr1 = perturb(rng, th0, on), perturb(rng, vr0, on)

        def log_q(x, pt=pt, ell=ell, log_r=log_r, loc=loc, grp=grp, gl=gl, on=on, local=local):
            th, vr = x
            lq = log_gig_kernel(np.exp(th[on]), np.asarray(pt.lam)[on], 1.0,
                                np.exp(np.asarray(pt.log_chi)[on]))  # fmt: skip
            pv = local(
                fam, "varrho", ell, log_r, dataclasses.replace(loc, log_theta=jnp.asarray(th)),
                grp, gl,
            )  # fmt: skip
            lq2 = log_gig_half(np.exp(vr[on]), 1.0, np.exp(np.asarray(pv.log_chi)[on]))
            return float(np.sum(lq) + np.sum(lq2))

        def joint(x, st=st, data=data):
            th, vr = x
            loc = dataclasses.replace(
                st.eq.coef, log_theta=jnp.asarray(th), log_varrho=jnp.asarray(vr)
            )
            return float(spec.log_joint(with_coef(st, loc=loc), data))

        worst["block"] = check_ratio(log_q, joint, (th0, vr0), (th1, vr1), worst["block"])
        G = info.G
        _, _, logits = step_of(spec).impl("gl_a_conditional")(
            fam, GLK._group_keys(KEY, Sub.GL_A, G), GLK._group_keys(KEY, Sub.GL_GROUP, G), loc,
            glob, info, gl,
        )  # fmt: skip
        logits = np.asarray(logits)
        idx0 = np.asarray(glob.a_idx)
        idx1 = (idx0 + rng.integers(1, A, idx0.shape)) % A

        def log_qa(idx, logits=logits):
            return float(np.sum(logits[np.arange(idx.size), idx]))

        def joint_a(idx, st=st, data=data):
            g = dataclasses.replace(st.glob.coef, a_idx=jnp.asarray(idx, jnp.int32))
            return float(spec.log_joint(with_coef(st, glob=g), data))

        worst["a"] = check_ratio(log_qa, joint_a, idx0, idx1, worst["a"])
    print(f"density ratio: worst scaled errors {worst}")
    assert max(worst.values()) <= 1e-8, worst


@pytest.mark.parametrize("case", ["ssvs_fixed", "ssvs_beta"])
def test_density_ratio_ssvs(case):
    """SSVS: ``γ_j`` (one coordinate flipped at a time: ``log q(1) − log q(0)`` is the log odds)
    and ``p_g | γ ~ Beta(s₁ + N1_g, s₂ + n_g − N1_g)`` (20 states, 1e-8)."""
    spec = spec_of(case)
    rng = np.random.default_rng(15)
    worst = {"gamma": 0.0, "p": 0.0}
    arr = spec.prior.eq.coef.gl
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        info, on, ell, log_r, gl = views(spec, st)
        loc, glob = st.eq.coef, st.glob.coef
        logit = np.asarray(step_of(spec).impl("ssvs_logodds")(
            ell, arr.ssvs.log_tau0, arr.ssvs.log_tau1, bcast(glob.log_p, info),
            bcast(glob.log1m_p, info),
        ))  # fmt: skip
        coords = np.argwhere(on)
        i, r = coords[rng.integers(len(coords))]
        g0 = np.asarray(loc.gamma)
        g1 = g0.copy()
        g1[i, r] = 1 - g1[i, r]

        def log_q(g, logit=logit, i=i, r=r):
            return float(logit[i, r]) if g[i, r] == 1 else 0.0

        def joint(g, st=st, data=data):
            loc = dataclasses.replace(st.eq.coef, gamma=jnp.asarray(g, jnp.int32))
            return float(spec.log_joint(with_coef(st, loc=loc), data))

        worst["gamma"] = check_ratio(log_q, joint, g0, g1, worst["gamma"])
        if spec.static.coef.p_beta:
            pb = step_of(spec).impl("gl_group_conditional")(
                "ssvs", "p", ell, log_r, loc, glob, info, gl
            )
            p0 = np.exp(np.asarray(glob.log_p))
            p1 = 1.0 / (1.0 + np.exp(-(np.log(p0) - np.log1p(-p0) + rng.normal(0, 0.5, p0.shape))))

            def log_qp(p, pb=pb):
                return float(np.sum(log_beta_pdf(p, np.asarray(pb.a), np.asarray(pb.b))))

            def joint_p(p, st=st, data=data):
                g = dataclasses.replace(
                    st.glob.coef, log_p=jnp.asarray(np.log(p)), log1m_p=jnp.asarray(np.log1p(-p))
                )
                return float(spec.log_joint(with_coef(st, glob=g), data))

            worst["p"] = check_ratio(log_qp, joint_p, p0, p1, worst["p"])
    print(f"density ratio: worst scaled errors {worst}")
    assert max(worst.values()) <= 1e-8, worst


def _program_views(spec, st, data):
    program = compose(spec.static)
    cache = _prepare_fn(program)(data, spec.prior)
    sh = sweep_values(program, KEY, jnp.uint32(0), st, data, spec.prior, cache)
    return program, cache, sh


@pytest.mark.parametrize("case", ["horseshoe", "ng_grid", "r2d2_fixed", "dl_grid", "ssvs_beta",
                                  "structural_iw_equivalent_hs"])  # fmt: skip
def test_density_ratio_structural_deviation_draw(case):
    """The deviation draw ``θ_i = [δ_i | a_i] | scales, λ, y`` (P3 §3.4): the exposed conditional
    in θ coordinates against the reference joint (20 states, 1e-8)."""
    spec = spec_of(case)
    n, k = spec.static.shapes.n, spec.static.shapes.k
    rng = np.random.default_rng(16)
    worst = 0.0
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        program, cache, sh = _program_views(spec, st, data)
        m0 = np.asarray(spec.prior.eq.coef.mean0)
        for i in range(n):
            eq_i = jax.tree.map(lambda x, i=i: x[i], st.eq)
            prior_i = jax.tree.map(lambda x, i=i: x[i], spec.prior.eq)
            cache_i = jax.tree.map(lambda x, i=i: x[i], cache.eq)
            ctx = program.structure.context(jnp.int32(i), sh.ytil, eq_i, program.vol, data,
                                            st.glob.vol)  # fmt: skip
            cm = program.structure.contemp_moments(ctx, prior_i)
            pm = program.coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
            cond = program.mean.conditional_eq(
                eq_i.mean, eq_i.struct.a, ctx, pm, cm, cache_i, cache, spec.prior.shared
            )
            Kc, bc = np.asarray(cond.K), np.asarray(cond.b)
            mask = np.concatenate([np.ones(k), np.asarray(cm.cmask)]) > 0
            th0 = np.concatenate([np.asarray(st.eq.mean.delta[i]), np.asarray(st.eq.struct.a[i])])
            th1 = np.where(mask, th0 + 0.05 * rng.standard_normal(th0.shape), 0.0)

            def log_q(th, Kc=Kc, bc=bc):
                return float(-0.5 * th @ Kc @ th + bc @ th)

            def joint(th, st=st, data=data, i=i, m0=m0):
                delta = st.eq.mean.delta.at[i].set(th[:k])
                dev = st.eq.mean.dev_log.at[i].set(exact_dev(th[:k], st.eq.mean.delta[i],
                                                             st.eq.mean.dev_log[i], m0[i]))  # fmt: skip
                eq = dataclasses.replace(
                    st.eq,
                    mean=LinearEq(delta=delta, dev_log=dev),
                    struct=RecursiveEq(a=st.eq.struct.a.at[i].set(th[k:])),
                )
                return float(spec.log_joint(dataclasses.replace(st, eq=eq), data))

            worst = check_ratio(log_q, joint, th0, th1, worst)
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


@pytest.mark.parametrize("case", ["reduced_sv_hs_global", "reduced_sv_ng_global",
                                  "reduced_sv_ssvs_global", "reduced_sv_dl_global"])  # fmt: skip
def test_density_ratio_cccm_deviation_draw(case):
    """The CCCM deviation draw ``π_j | π_{−j}, A, Λ, scales, y`` (P3 §3.4) against the reference
    joint (20 states, every ``j``, 1e-8)."""
    spec = spec_of(case)
    n = spec.static.shapes.n
    rng = np.random.default_rng(17)
    worst = 0.0
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        program, cache, _ = _program_views(spec, st, data)
        step = program.global_pre[0]
        ctx = SweepCtx(data, spec.prior, cache, None)
        m0 = np.asarray(spec.prior.eq.coef.mean0)
        for j in range(n):
            cond = step.conditional_j(st, ctx, j, jax.vmap)
            Kc, bc = np.asarray(cond.K), np.asarray(cond.b)
            p0 = np.asarray(st.glob.mean.Pi[:, j])
            p1 = p0 + 0.05 * rng.standard_normal(p0.shape)

            def log_q(p, Kc=Kc, bc=bc):
                return float(-0.5 * p @ Kc @ p + bc @ p)

            def joint(p, st=st, data=data, j=j, m0=m0):
                Pi = st.glob.mean.Pi.at[:, j].set(p)
                dev = st.glob.mean.dev_log.at[j].set(
                    exact_dev(p, st.glob.mean.Pi[:, j], st.glob.mean.dev_log[j], m0[j])
                )
                glob = dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=Pi, dev_log=dev))
                return float(spec.log_joint(dataclasses.replace(st, glob=glob), data))

            worst = check_ratio(log_q, joint, p0, p1, worst)
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


# --------------------------------------------------------------------------- closed forms (item 3)


def _quad_log_marginal(n_g, S, logW, a, b, log_beta):
    """``log ∫ Π_j G(ϖ_j; a, ξ) G(ξ; b, β) dξ`` up to the ``a``-free constant, by numerical
    integration over ``u = log ξ`` (``scipy.integrate.quad`` on ``[m − 60/√s, m + 40/√s]`` around
    the mode ``m``, where the shifted integrand peaks at 1): ``(a − 1)S − n lnΓ(a) + b log β −
    lnΓ(b) + log ∫ exp((n a + b)u − e^u (β + W)) du``."""
    from scipy.integrate import quad

    log_rate = np.logaddexp(log_beta, logW)
    shape = n_g * a + b
    mode = math.log(shape) - log_rate  # e^m (β + W) = s at the mode

    def f(u):
        return math.exp(shape * (u - mode) - math.exp(u + log_rate) + shape)

    w = 1.0 / math.sqrt(shape)
    val, err = quad(f, mode - 60.0 * w, mode + 40.0 * w, points=[mode], epsabs=0.0,
                    epsrel=1e-13, limit=400)  # fmt: skip
    assert err <= 1e-11 * val
    log_int = math.log(val) + shape * mode - shape
    return (a - 1.0) * S - n_g * math.lgamma(a) + b * log_beta - math.lgamma(b) + log_int


@pytest.mark.parametrize("case", ["ng_grid", "ng_grid_c1", "r2d2_grid"])
def test_collapsed_a_weights_against_quadrature(case):
    """The collapsed ``a``-weights (``ξ`` integrated) against numerical integration over ``ξ`` with
    ``c`` tied (``ng_grid``), fixed at 1 (``ng_grid_c1``) and R2D2's ``β = 1`` (P3 §9.2 item 3;
    normalised log weights, 1e-9)."""
    spec = spec_of(case)
    fam = spec.static.coef.kind
    worst = 0.0
    for rep in range(3):
        st, _ = draw_state(spec, rep)
        info, _on, _ell, _log_r, gl = views(spec, st)
        loc = st.eq.coef
        logits = np.asarray(GLK.a_log_weights(fam, loc, info, gl))
        logp = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
        S = np.asarray(GLK.group_sum(loc.log_w, info))
        logW = np.asarray(GLK.group_logsumexp(loc.log_w, info))
        n_g = np.asarray(info.n_g)
        sup, lw = np.asarray(gl.a_support), np.asarray(gl.a_logw)
        c = float(gl.c)
        for g in range(info.G):
            ref = []
            for a, w in zip(sup, lw, strict=True):
                if fam == "r2d2":
                    log_beta = 0.0
                else:
                    log_beta = 0.0 if c < 0 else math.log(2.0 * c / a)
                ref.append(w + _quad_log_marginal(n_g[g], S[g], logW[g], a, float(gl.b), log_beta))
            ref = np.asarray(ref)
            ref -= np.logaddexp.reduce(ref)
            worst = max(worst, float(np.max(np.abs(ref - logp[g]))))
    print(f"closed form: worst {worst:.2e}")
    assert worst <= 1e-9, worst


def test_dl_theta_given_d_against_quadrature():
    """DL ``θ̃ | d ~ GIG(a − 1, 1, 2|d|/√r)`` (ϱ integrated) against mpmath quadrature of the joint
    ``G(θ̃; a, ½) ∫ N(d; 0, rϱθ̃²) Exp(ϱ; ½) dϱ``: log density differences at five points
    (several ``a``, ``d``, ``r``; 1e-8 relative)."""
    mpmath.mp.dps = 40
    worst = 0.0
    for a, d, r in [(0.5, 0.3, 1.0), (0.1, 2.0, 0.25), (1.0, 0.01, 4.0), (0.75, 1.5, 0.5)]:
        ell = np.full((1, 1), math.log(abs(d)))
        grp = {"a": np.full((1, 1), a)}
        p = GLK.gl_local_conditional(
            "dirichlet_laplace", "theta", ell, np.full((1, 1), math.log(r)), None, grp, None
        )
        lam, psi, chi = float(p.lam[0, 0]), math.exp(p.log_psi[0, 0]), math.exp(p.log_chi[0, 0])

        def joint(th, a=a, d=d, r=r):
            th = mpmath.mpf(th)

            def inner(rho):
                v = r * rho * th**2
                return (
                    mpmath.e ** (-(d**2) / (2 * v))
                    / mpmath.sqrt(2 * mpmath.pi * v)
                    * 0.5
                    * mpmath.e ** (-rho / 2)
                )

            dens = mpmath.quad(inner, [0, 1, 10, mpmath.inf])
            prior = th ** (a - 1) * mpmath.e ** (-th / 2)
            return float(mpmath.log(prior * dens))

        pts = [0.05, 0.4, 1.0, 3.0, 9.0]
        ref = [joint(t) for t in pts]
        got = [float(log_gig_kernel(t, lam, psi, chi)) for t in pts]
        for k in range(1, len(pts)):
            lhs, rhs = got[k] - got[0], ref[k] - ref[0]
            worst = max(worst, abs(lhs - rhs) / (1.0 + abs(rhs)))
    print(f"density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8, worst


def test_ssvs_logodds_against_brute_force():
    """The SSVS log odds against ``log p − log(1 − p) + log N(d; 0, τ₁²) − log N(d; 0, τ₀²)`` in
    50-digit arithmetic, with ``τ₀ < τ₁``, ``τ₀ > τ₁`` (engine-level arrays) and ``τ₀ = τ₁``
    (P3 §9.2 item 3; 1e-12, relative to ``1 + |logit|``)."""
    mpmath.mp.dps = 50
    rng = np.random.default_rng(18)
    m = 400
    ell = rng.normal(-2.0, 2.0, m)
    lt0 = rng.normal(-3.0, 1.5, m)
    lt1 = np.where(np.arange(m) % 5 == 0, lt0, lt0 + rng.normal(1.5, 1.5, m))
    lp = np.log(rng.uniform(0.05, 0.95, m))
    l1mp = np.log1p(-np.exp(lp))
    got = np.asarray(GLK.ssvs_logodds(ell, lt0, lt1, lp, l1mp))
    worst = 0.0
    for j in range(m):
        d2 = mpmath.e ** (2 * mpmath.mpf(ell[j]))
        t0, t1 = mpmath.e ** (2 * mpmath.mpf(lt0[j])), mpmath.e ** (2 * mpmath.mpf(lt1[j]))
        ref = (mpmath.mpf(lp[j]) - mpmath.mpf(l1mp[j]) - mpmath.log(t1) / 2 - d2 / (2 * t1)
               + mpmath.log(t0) / 2 + d2 / (2 * t0))  # fmt: skip
        worst = max(worst, abs(got[j] - float(ref)) / (1.0 + abs(float(ref))))
    print(f"closed form: worst {worst:.2e}")
    assert worst <= 1e-12, worst


# --------------------------------------------------------------------------- block Geweke (Tier B)

BLOCKS = [
    "gl_scales_only_horseshoe",
    "gl_scales_only_normal_gamma",
    "gl_scales_only_r2d2",
    "gl_scales_only_dirichlet_laplace",
    "gl_scales_only_ssvs",
    "gl_mean_only",
]


@pytest.mark.geweke
@pytest.mark.parametrize("name", BLOCKS)
def test_block_geweke(name):
    """Block Geweke 401–406 (P3 §9.3) at G-small (``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``;
    Bonferroni over the configuration's test functions): 401–405 run the production scale step
    alone (``kernel="global_step"``, coefficients redrawn from ``p(d | scales)``), 406 the
    production sweep with the scales held (``kernel="sweep"``)."""
    spec = joint_spec(name)
    assert spec.kernel == ("sweep" if name == "gl_mean_only" else "global_step")
    res = geweke(spec, GewekeConfig(seed=401 + BLOCKS.index(name), **G_SMALL))
    print(f"{name}: {len(res.names)} functions, max |z| = {np.max(np.abs(res.z)):.3f}, "
          f"min p = {np.min(res.p):.2e} (Bonferroni: p > {G_SMALL['alpha']}/{len(res.names)})")  # fmt: skip
    assert res.passed, res.table().head(8)


# --------------------------------------------------------------------------- fit level (§3.7, D-52)


def _sample_data(T: int = 70, n: int = 3, seed: int = 4):
    import pandas as pd

    from rai_bvar.data import VARData

    rng = np.random.default_rng(seed)
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(n) * np.array([1.0, 0.3, 3.0])[:n]
    df = pd.DataFrame(y, index=pd.period_range("1990Q1", periods=T, freq="Q"),
                      columns=[f"y{j}" for j in range(n)])  # fmt: skip
    return VARData.from_frame(df, dict.fromkeys(df.columns, 1))


def _static(prior, vol=None, lags=2, **mcmc):
    import rai_bvar as rb
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.settings import Runtime

    settings = rb.MCMCSettings(chains=2, draws=20, burn=10, **mcmc)
    model = rb.BVAR(mean=rb.Linear(prior), vol=vol or rb.Constant(), lags=lags, mcmc=settings)
    return model, resolve(model, _sample_data(), Runtime())


def test_monitor_fields_follow_d52():
    """D-52 (as amended): group log scales primary iff ``G ≤ max(2p, n)``, SSVS ``p_g`` primary
    under a Beta prior (absent when fixed), ``τ_g`` and ``a_idx`` secondary, no local scales;
    Phase 2 fields unchanged with a fixed prior."""
    import rai_bvar as rb
    from rai_bvar.engines.mcmc import monitor as M
    from rai_bvar.priors.sv import Beta

    def fields(prior, **kw):
        _, r = _static(prior, **kw)
        s = r.static
        out = M.monitor_fields(s.mean.kind, s.vol, s.shapes.n, s.shapes.n_det, coef=s.coef,
                               lags=s.shapes.lags)  # fmt: skip
        assert (
            out[: len(M.monitor_fields(s.mean.kind, s.vol, s.shapes.n, 1)) - 1]
            == (M.monitor_fields(s.mean.kind, s.vol, s.shapes.n, 1)[:-1])
        )
        spec = M.monitor_spec(s, ("y0", "y1", "y2"))
        return {f.name: f.primary for f in out if f.name.startswith("gl_")}, spec

    f, spec = fields(rb.Horseshoe())  # olcl-lagwise: G = 4 = 2p
    assert f == {"gl_log_zeta": True, "gl_log_tau": False}
    assert "gl_log_zeta[3]" in spec.primary and "gl_log_tau[0]" in spec.secondary
    assert fields(rb.Horseshoe(groups="covariate-wise"))[0]["gl_log_zeta"] is False  # G = 6
    assert fields(rb.NormalGamma(groups="equation-wise"))[0] == {"gl_log_xi": True,
                                                                 "gl_a_idx": False}  # fmt: skip
    assert fields(rb.R2D2(a=0.4))[0] == {"gl_log_xi": True}
    assert fields(rb.DirichletLaplace())[0] == {"gl_a_idx": False}
    assert fields(rb.SSVS(p=Beta(2.0, 3.0)))[0] == {"gl_p": True}
    assert fields(rb.SSVS())[0] == {}
    assert not any(label.startswith(("gl_log_theta", "gl_log_nu")) for label in spec.labels)


@pytest.mark.parametrize("case", ["horseshoe", "ng_grid_c1", "r2d2_grid", "dl_grid", "ssvs_beta",
                                  "reduced_sv_hs_global", "reduced_sv_ssvs_global",
                                  "reduced_sv_r2d2_global"])  # fmt: skip
def test_monitor_log_prior_equals_the_reference(case):
    """The monitored ``log p(θ)`` (mean and CCCM parts from ``dev_log``, the scales'
    densities from the full state, P3 §3.7) equals WP30's ``log_joint.log_prior`` at prior
    states (1e-10 relative): the kernels' densities and the references agree on every
    normalising constant and coordinate."""
    from rai_bvar.engines.mcmc.monitor import monitor_values
    from rai_bvar.validation.log_joint import log_prior

    spec = spec_of(case)
    program = compose(spec.static)
    for rep in range(3):
        st, data = draw_state(spec, rep)
        cache = _prepare_fn(program)(data, spec.prior)
        got = float(monitor_values(st, data, spec.prior, cache, program)["log_prior"])
        ref = float(log_prior(st, spec.prior, spec.static, data.tmask))
        assert abs(got - ref) <= 1e-10 * (1.0 + abs(ref)), (got, ref)


def test_fit_storage_results_and_warnings():
    """A structural SSVS × Constant fit with a Beta ``p`` and a reduced NG fit: the indicators γ
    stored as int8 with the parameter draws (``store_thin``; P3 E123 as amended), zero off the
    lag slots; ``latent_moments/incl_prob (variable, regressor)`` in ``[0, 1]``; the ``group``
    monitor fields with their coordinate; implied reduced-form moments "not applicable" (D-62)."""
    import warnings

    import rai_bvar as rb
    from rai_bvar.priors.sv import Beta

    data = _sample_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = rb.BVAR(mean=rb.Linear(rb.SSVS(p=Beta(2.0, 3.0))), vol=rb.Constant(), lags=2,
                      mcmc=rb.MCMCSettings(chains=2, draws=40, burn=20, store_max=20)).fit(data)  # fmt: skip
        red = rb.BVAR(mean=rb.Linear(rb.NormalGamma(form="reduced")), vol=rb.SV(), lags=2,
                      mcmc=rb.MCMCSettings(chains=2, draws=20, burn=20)).fit(data)  # fmt: skip
    post = fit.posterior["posterior"]
    g = post["ssvs_gamma"]
    assert g.dtype == np.int8 and g.dims == ("chain", "draw", "variable", "regressor")
    assert g.sizes["draw"] == 20 and set(np.unique(g.values)) <= {0, 1}
    assert not g.values[..., 0].any()
    incl = fit.posterior["latent_moments"]["incl_prob"]
    assert incl.dims == ("chain", "variable", "regressor")
    assert np.all((incl.values >= 0) & (incl.values <= 1)) and not incl.values[..., 0].any()
    mon = fit.posterior["monitor"]
    assert mon["gl_p"].dims == ("chain", "draw", "group") and mon.sizes["group"] == 4
    assert np.all((mon["gl_p"].values > 0) & (mon["gl_p"].values < 1))
    assert fit.implied_prior_moments() is None
    assert fit.summary().attrs["implied_reduced_form"].startswith("not applicable")
    assert "gl_p[0]" in fit.monitor_spec.primary
    rmon = red.posterior["monitor"]
    assert {"gl_log_xi", "gl_a_idx"} <= set(rmon.data_vars)
    assert np.isfinite(rmon["log_prior"].values).all()
    assert red.hyper["a"] == "gruber_kastner" and red.hyper["n_groups"] == 4
    assert red.summary().attrs["implied_reduced_form"].startswith("not applicable")


def test_resolve_branches_for_global_local_priors():
    """Resolution of a GL prior (P3 §3.6, §2.1, E65): the internal Minnesota for ψ and the fixed
    blocks, ``coef_log``, no dummy rows, ``G`` from ``n``, and ``const_path="cholesky"`` resolving
    to the weighted path (the varying-prior check comes first)."""
    import rai_bvar as rb
    from rai_bvar.engines.mcmc.resolve import check_coef_groups

    model, r = _static(rb.Horseshoe(groups="equation-wise"), const_path="cholesky")
    s = r.static
    assert s.mean.coef_log and s.mean.const_path == "weighted" and s.mean.n_dummy == 0
    assert s.coef.kind == "horseshoe" and s.coef.n_groups == 3 == check_coef_groups(model, 3)
    assert r.hyper["groups"] == "equation-wise" and "lambda" not in r.hyper
    gl = r.prior.eq.coef.gl
    assert np.asarray(gl.mask)[:, 0].sum() == 0 and np.asarray(gl.mask)[:, 1:].all()
    _, rr = _static(rb.SSVS(form="reduced"), vol=rb.Constant("iw_equivalent"))
    assert rr.static.mean.kind == "linear_reduced" and rr.static.mean.coef_log
    _, rm = _static(rb.Minnesota(form="structural"), const_path="cholesky")
    assert rm.static.mean.const_path == "cholesky_const" and not rm.static.mean.coef_log


class _Compiles:
    """Backend compilations of the runner's executables (``jax.monitoring`` events), as in the
    Phase 2 ``tests/mcmc/runner/test_recompile.py``."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def __call__(self, event: str, duration: float, **kw) -> None:
        if event.endswith("backend_compile_duration"):
            self.names.append(str(kw.get("fun_name", "")))

    @property
    def runner(self) -> list[str]:
        return [n for n in self.names if "mcmc_" in n]


def test_zero_recompiles_across_windows_of_one_bucket(caplog):
    """P3 §9.2 item 4 (the review R6 of PR #23; D-23): a global-local composition compiles once
    per ``T_pad`` bucket. Reduced semi-automatic SSVS × SV on two windows of one bucket
    (``T_eff`` 68 and 78, ``T_pad`` 96) whose GL prior arrays differ (the per-window ``ŝ`` and
    ``log r``): the same ``StaticConfig`` and array shapes (as WP32's one-bucket check in
    ``test_common_sv.py``), one runner, and no ``mcmc_*`` executable compiles on the second
    window (the Phase 2 listener and compile log)."""
    import logging

    import rai_bvar as rb
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.runner import make_runner, run_chains
    from rai_bvar.settings import MCMCSettings, Runtime

    rt = Runtime(chunk=5)
    prior = rb.SSVS(form="reduced")
    assert prior.semiautomatic
    model = rb.BVAR(mean=rb.Linear(prior), vol=rb.SV(), lags=2,
                    mcmc=rb.MCMCSettings(chains=2, draws=8, burn=4))  # fmt: skip
    r1, r2 = (resolve(model, _sample_data(T=T), rt) for T in (70, 80))
    static = r1.static
    assert r2.static == static and static.shapes.T_pad == 96 and static.coef.kind == "ssvs"
    assert (int(r1.data.T_eff), int(r2.data.T_eff)) == (68, 78)
    for a_, b_ in ((r1.prior, r2.prior), (r1.data, r2.data)):
        assert jax.tree.structure(a_) == jax.tree.structure(b_)
        assert [np.shape(x) for x in jax.tree.leaves(a_)] == [
            np.shape(x) for x in jax.tree.leaves(b_)
        ]
    gl1, gl2 = (jax.tree.leaves(r.prior.eq.coef.gl) for r in (r1, r2))
    assert any(not np.array_equal(x, y) for x, y in zip(gl1, gl2, strict=True))

    # fresh executables of the production kernels (an uncached runner), so window 1 compiles
    fns = make_runner(static, overrides={"vol": type(compose(static).vol)})
    counter = _Compiles()
    jax.monitoring.register_event_duration_secs_listener(counter)
    try:
        run_chains(fns, static, r1.data, r1.prior, MCMCSettings(chains=2, draws=7, burn=4), rt,
                   seed=1, salt=r1.salt)  # fmt: skip
        first = set(counter.runner)
        assert {
            "jit(mcmc_prepare)",
            "jit(mcmc_init)",
            "jit(mcmc_burn)",
            "jit(mcmc_sample_chunk)",
        } <= first, first
        counter.names.clear()
        jax.config.update("jax_log_compiles", True)
        with caplog.at_level(logging.WARNING, logger="jax._src.interpreters.pxla"):
            # window 2 of the bucket: new data and prior arrays, burn, thin, draws and seeds
            settings = MCMCSettings(chains=2, draws=12, burn=9, thin=3)
            run_chains(fns, static, r2.data, r2.prior, settings, rt, seed=2, salt=r2.salt)
    finally:
        jax.config.update("jax_log_compiles", False)
        jax.monitoring.unregister_event_duration_listener(counter)
    assert counter.runner == []
    logged = [
        r.getMessage()
        for r in caplog.records
        if "Compiling" in r.getMessage() and "mcmc_" in r.getMessage()
    ]
    assert logged == []


@pytest.mark.parametrize("case,block", [("horseshoe", Block.EQ_MEAN),
                                        ("reduced_sv_hs_global", Block.G_CCCM)])  # fmt: skip
def test_failure_detail_is_log_aware(case, block):
    """D-62: the replay diagnostics read ``GaussianLogDiag`` moments: the log precision range and
    the conditioning of the Jacobi-scaled matrix formed in log space."""
    from rai_bvar.engines.mcmc.runner import failure_detail

    spec = spec_of(case, "small")
    program = compose(spec.static)
    st, data = draw_state(spec, 0)
    cache = _prepare_fn(program)(data, spec.prior)
    out = failure_detail(program, st, data, spec.prior, cache, KEY, 3, block, 0)
    assert "prior_log_precision_range" in out and "prior_precision_range" not in out
    assert out["log_space"] is True and out["cholesky_ok"] is True


def test_kernel_module_imports_inside_a_trace():
    """The registry may import the kernel module for the first time inside a traced function
    (``validation.joint`` composes lazily in its step), so the module computes nothing with JAX
    at import time: a fresh interpreter steps a block configuration under ``jit`` first."""
    import subprocess
    import sys

    code = (
        "import jax, jax.numpy as jnp\n"
        "from rai_bvar.validation.joint import joint_spec\n"
        "s = joint_spec('gl_scales_only_ssvs')\n"
        "st = s.prior_sample(jax.random.key(0)); d = s.simulate(jax.random.key(1), st)\n"
        "out = jax.jit(lambda a, b: s.step(jax.random.key(2), jnp.uint32(0), a, b))(st, d)\n"
        "assert int(out.status.code) == 0\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]


def test_gl_warnings():
    """P3 §8 warnings: secondary group scales (``G > max(2p, n)``) failing R-hat or ESS are named
    (informational) and a hyperprior index at a grid end for more than half the draws says the
    grid may be too narrow; neither fires on well-mixed draws inside the grid."""
    import rai_bvar as rb
    from rai_bvar.engines.mcmc import monitor as M
    from rai_bvar.settings import MCMCSettings

    _, r = _static(rb.NormalGamma(groups="covariate-wise"))  # G = n·p = 6 > max(2p, n) = 4
    s = r.static
    spec = M.monitor_spec(s, ("y0", "y1", "y2"))
    rng = np.random.default_rng(0)
    C, D, G, A = 4, 500, s.coef.n_groups, s.coef.a_grid
    good = {"gl_log_xi": rng.standard_normal((C, D, G)),
            "gl_a_idx": rng.integers(1, A - 1, (C, D, G)).astype(float)}  # fmt: skip
    settings = MCMCSettings(chains=C, draws=D)
    assert M._gl_warnings(good, spec, s, settings) == []
    bad_xi = good["gl_log_xi"] + np.arange(C)[:, None, None]  # chains disagree
    idx = good["gl_a_idx"].copy()
    idx[..., 2] = A - 1  # group 2 always at the upper end
    out = M._gl_warnings({"gl_log_xi": bad_xi, "gl_a_idx": idx}, spec, s, settings)
    assert len(out) == 2
    assert out[0].startswith("secondary (informational") and "gl_log_xi[" in out[0]
    assert "group(s) [2]" in out[1] and "the grid may be too narrow" in out[1]
