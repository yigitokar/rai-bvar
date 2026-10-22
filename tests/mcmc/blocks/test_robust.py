"""``VolBlockImpl`` = transitory scale ∘ persistent core (DESIGN §13 WP13; §3.5, §3.7.7, §4.4;
N03 §5).

The internal order of ``sample_eq`` and its data flow (spy through the override seam); ``log e²``
masking (padded rows never ``−∞``; exact zeros reach the SV core as ``−∞`` for its zero-offset
floor masking, checked with a stub SV core); the views (``log_var_eq``, ``loglik_eq``,
``log_prior_eq``) against the independent references; ``obs_logdensity_eq`` against exact sums and
Monte Carlo (N03 §5.3, §7.1 test 11a); ``draw_ahead_eq`` with ``E λ = E[e^h](1 + 150p)`` (N03 §5.3
item 3, §7.1 test 9); N03 §7.1 test 6 (unknown ``λ̄`` with t: quadrature over ``λ̄`` per grid ν),
test 8 (``p = 0`` limit) and test 12 (scale equivariance); state invariants after sweeps; block
Geweke ``transitory_known_lambda`` and ``constant_transitory`` (DESIGN §9.2, G-small sizes).

``StubSV`` is a local test double of WP12's ``SVCore`` (DESIGN §13 rule: code against the
protocol until the real kernel lands): its ``sample`` records the cleaned ``log ẽ²`` it receives in
``rb_h_sum[1:]`` so the spy survives ``jit``, and its ``ahead``/``transition`` simulate the AR(1)
law from the key they receive, consumed directly as the real core does. The key split of
``draw_ahead_eq``/``transition_eq`` is checked against both: the core's random words never equal
the key data of a transitory sub-key (DESIGN §3.1 rule 5, §5.6).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax import random as jr
from scipy import special as sc
from scipy import stats

from rai_bvar.components.vol import constant as CN
from rai_bvar.components.vol import robust as RB
from rai_bvar.components.vol import transitory as TR
from rai_bvar.engines.mcmc import keys as KY
from rai_bvar.engines.mcmc import protocols as P
from rai_bvar.engines.mcmc import registry
from rai_bvar.engines.mcmc.config import StaticConfig, VolStatic
from rai_bvar.engines.mcmc.keys import Block, Sub, sub_key
from rai_bvar.engines.mcmc.state import (
    ConstantDraw,
    ConstantEq,
    ConstantPrior,
    Coupling,
    Empty,
    EqPrior,
    OutlierTDraw,
    OutlierTEq,
    StepInfo,
    SVDraw,
    SVEq,
    TransitoryPrior,
    VolDraw,
    VolEq,
    check_state_invariants,
)
from rai_bvar.validation import reference as REF
from rai_bvar.validation.geweke import GewekeConfig, geweke_sequential
from rai_bvar.validation.joint import JointSpec, hold_frozen, joint_spec
from rai_bvar.validation.log_joint import log_prior_eq, loglik_terms, residuals

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = jr.key(20261025, impl="threefry2x32")
GRID = tuple(range(3, 41))


def key(i: int):
    return jr.fold_in(KEY, i)


def tmask_of(T_eff: int, T_pad: int) -> jax.Array:
    return (jnp.arange(T_pad) < T_eff).astype(jnp.float64)


def eq_prior(shape=5.0, scale=4.0, a_p=2.0, b_p=18.0, grid=GRID) -> EqPrior:
    G = len(grid)
    tp = TransitoryPrior(jnp.float64(a_p), jnp.float64(b_p), jnp.full((G,), -math.log(G)))
    return EqPrior(Empty(), Empty(), ConstantPrior(jnp.float64(shape), jnp.float64(scale)), tp)


def block(transitory="outlier_t", T_pad=64, coupled=False, grid=GRID, K=20) -> RB.VolBlockImpl:
    st = VolStatic("constant", transitory=transitory, coupled=coupled, nu_grid=grid, outlier_K=K)
    return RB.VolBlockImpl(st, T_pad)


# --------------------------------------------------------------------------- stub SV core (WP12)


@dataclass(frozen=True)
class StubSV(P.KernelBase):
    """AR SV test double: state ``SVEq``; ``log_var = h[1:]``; ``sample`` stores the received
    ``log ẽ²`` in ``rb_h_sum[1:]`` (a jit-proof spy) and leaves ``h, μ, φ, σ`` unchanged."""

    static: VolStatic
    T_pad: int

    def init(self, key, log_e2, tmask, T_eff, prior_i):
        z, T = jnp.zeros(()), self.T_pad
        return SVEq(
            h=jnp.full((T + 1,), -0.5), mu=jnp.float64(-0.5), phi=jnp.float64(0.9),
            sigma=jnp.float64(0.3), acc_h=z, acc_cgp=z, acc_cs2=z, acc_ncphi=z, acc_ncms=z,
            acc_fold=z, acc_floor=z, acc_rej_run=jnp.int32(0), acc_rej_run_max=jnp.int32(0),
            rb_h_sum=jnp.zeros(T + 1), rb_h_sq=jnp.zeros(T + 1),
        )  # fmt: skip

    def sample(self, key, cs, log_e2c, tmask, T_eff, prior_i, cpl):
        return dataclasses.replace(cs, rb_h_sum=cs.rb_h_sum.at[1:].set(log_e2c)), StepInfo.success()

    def log_var(self, cs):
        return cs.h[1:]

    def log_scale(self, cs):
        return jnp.zeros(())

    def extract(self, cs, T_eff):
        return SVDraw(h_T=cs.h[T_eff], mu=cs.mu, phi=cs.phi, sigma=cs.sigma)

    def ahead(self, key, cd, H):
        eta = jr.normal(key, (H,), dtype=jnp.float64)  # consumes its key directly, as SVCore does

        def step(h, e):
            h = cd.mu + cd.phi * (h - cd.mu) + cd.sigma * e
            return h, h

        return lax.scan(step, cd.h_T, eta)[1]

    def transition(self, key, log_lam_prev, cd):
        e = jr.normal(key, (), dtype=jnp.float64)
        return cd.mu + cd.phi * (log_lam_prev - cd.mu) + cd.sigma * e

    def log_prior(self, cs, prior_i, tmask):
        return jnp.zeros(())


def sv_block(transitory="outlier_t", T_pad=64, zero_offset=False, grid=GRID) -> RB.VolBlockImpl:
    st = VolStatic("sv", transitory=transitory, zero_offset=zero_offset, nu_grid=grid)
    return RB.VolBlockImpl(st, T_pad, core=StubSV(st, T_pad))


# --------------------------------------------------------------------------- construction


def test_construction_from_registry_and_explicit_parts():
    assert registry.load("vol_block") is RB.VolBlockImpl
    spec = joint_spec("constant_transitory")
    v = RB.VolBlockImpl(spec.static)
    assert isinstance(v, P.VolBlock) and v.static == spec.static.vol and v.T_pad == 32
    assert isinstance(v.core, CN.ConstantCore) and isinstance(v.transitory, TR.OutlierTScale)
    w = RB.VolBlockImpl(spec.static.vol, 32, core=CN.ConstantCore(spec.static),
                        transitory=TR.OutlierTScale(spec.static))  # fmt: skip
    assert v == w and hash(v) == hash(w)
    assert not v.constant_weights() and block("none").constant_weights()
    with pytest.raises(ValueError, match="another VolStatic"):
        RB.VolBlockImpl(spec.static.vol, 32, core=CN.ConstantCore(VolStatic("constant"), 32))
    with pytest.raises(ValueError, match="T_pad"):
        RB.VolBlockImpl(spec.static.vol, 32, core=CN.ConstantCore(spec.static.vol, 64))
    try:
        registry.load("sv")
    except NotImplementedError:  # WP12 not landed yet: the default build says so
        with pytest.raises(NotImplementedError):
            RB.VolBlockImpl(VolStatic("sv"), 32)


def test_overrides_are_pushed_down_to_both_parts():
    v = block()

    def fake_conditional(*a):
        return CN.constant_conditional(*a)

    def fake_nu(*a):
        return TR.nu_logits(*a)

    w = v.with_overrides({"constant_conditional": fake_conditional, "nu_logits": fake_nu})
    assert w != v
    assert w.core.impl("constant_conditional") is fake_conditional
    assert w.transitory.impl("nu_logits") is fake_nu
    assert v.core.impl("constant_conditional") is CN.constant_conditional
    with pytest.raises(ValueError, match="not overridable"):
        v.with_overrides({"bogus": fake_nu})


# --------------------------------------------------------------------------- the order of sample_eq


def test_sample_eq_order_and_data_flow_spy():
    """o → p → ν → τ → λ, each step seeing the outputs of the previous ones (DESIGN §4.4)."""
    T_eff, T_pad = 30, 32
    tm = tmask_of(T_eff, T_pad)
    rng = np.random.default_rng(1)
    e = rng.normal(size=T_pad) * 0.8
    e[7] = 9.0
    e[T_eff:] = 0.0
    seen: list[tuple[str, tuple[Any, ...]]] = []

    def spy(name, fn):
        def wrapped(*a):
            seen.append((name, a))
            return fn(*a)

        return wrapped

    names = (
        "outlier_logits",
        "p_conditional",
        "nu_logits",
        "tau_conditional",
        "constant_conditional",
    )
    mods = {n: (TR if n != "constant_conditional" else CN) for n in names}
    v = block(T_pad=T_pad).with_overrides({n: spy(n, getattr(mods[n], n)) for n in names})
    pr = eq_prior()
    vs = v.init_eq(key(1), jnp.asarray(e), tm, jnp.int32(T_eff), pr)
    vs = dataclasses.replace(vs, core=ConstantEq(jnp.float64(-0.3)))
    new, info = v.sample_eq(
        key(2), vs, jnp.asarray(e), tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty()
    )
    assert bool(info.ok)
    assert [n for n, _ in seen] == list(names)
    args = dict(seen)
    le2 = np.where(np.asarray(tm) > 0, np.log(np.where(np.asarray(tm) > 0, e * e, 1.0)), 0.0)
    # o step: log z² = log e² − log λ^pers of the CURRENT (pre-step) core state
    np.testing.assert_allclose(
        np.asarray(args["outlier_logits"][0]), np.where(tm > 0, le2 + 0.3, 0.0)
    )
    # p step sees the new o; ν step sees r² = z²/o² with that o
    o_new = np.asarray(new.trans.o_idx)
    np.testing.assert_array_equal(np.asarray(args["p_conditional"][0]), o_new)
    lr2 = np.where(np.asarray(tm) > 0, le2 + 0.3 - 2 * np.log(o_new + 1.0), 0.0)
    np.testing.assert_allclose(np.asarray(args["nu_logits"][0]), lr2, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(args["tau_conditional"][0]), lr2, rtol=1e-15)
    # the core sees log ẽ² = log e² − 2 log o − log τ of the NEW transitory state
    le2c = np.where(np.asarray(tm) > 0, le2 - np.asarray(v.log_scale2_eq(new)), 0.0)
    np.testing.assert_allclose(np.asarray(args["constant_conditional"][0]), le2c, rtol=1e-15)
    assert o_new[7] > 0  # the planted outlier


def test_sv_core_receives_the_cleaned_residual_after_the_transitory_step():
    T_eff, T_pad = 20, 32
    tm = tmask_of(T_eff, T_pad)
    v = sv_block(T_pad=T_pad)
    pr = eq_prior()
    e = jnp.asarray(np.random.default_rng(2).normal(size=T_pad)) * tm
    vs = v.init_eq(key(3), e, tm, jnp.int32(T_eff), pr)
    new, _ = jax.jit(
        lambda k, vs: v.sample_eq(k, vs, e, tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty())
    )(key(4), vs)
    got = np.asarray(new.core.rb_h_sum[1:])
    le2 = np.asarray(RB.log_e2(e, tm))
    np.testing.assert_allclose(
        got, np.where(np.asarray(tm) > 0, le2 - np.asarray(v.log_scale2_eq(new)), 0.0)
    )
    # h is untouched by the stub, so log_var_eq(persistent) is the old h
    np.testing.assert_array_equal(
        np.asarray(v.log_var_eq(new, Empty(), "persistent")), np.asarray(vs.core.h[1:])
    )


# --------------------------------------------------------------------------- log e² masking


def test_log_e2_never_minus_inf_on_padded_rows():
    tm = tmask_of(4, 8)
    resid = jnp.asarray([0.5, 0.0, -2.0, 1e-200, np.nan, np.inf, 0.0, 3.0])
    out = np.asarray(RB.log_e2(resid, tm))
    assert np.all(out[4:] == 0.0)  # padded rows: masked BEFORE the log (E23)
    assert np.isneginf(out[1]) and np.isneginf(out[3])  # exact zero (and underflow) in real rows
    assert out[0] == pytest.approx(2 * math.log(0.5)) and out[2] == pytest.approx(2 * math.log(2.0))


@pytest.mark.parametrize("transitory", ["none", "outlier", "t", "outlier_t"])
def test_zero_offset_rows_reach_the_sv_core_as_minus_inf(transitory):
    """Exact zeros in real rows pass through the transitory step as ``−∞`` (finite transitory
    state) so the SV core's zero-offset floor masking can count them (N02 §2.4; DESIGN §3.5)."""
    T_eff, T_pad = 21, 32
    tm = tmask_of(T_eff, T_pad)
    v = sv_block(transitory, T_pad, zero_offset=True)
    e = np.random.default_rng(4).normal(size=T_pad)
    e[[3, 12]] = 0.0
    e[T_eff:] = np.nan
    pr = eq_prior()
    vs = v.init_eq(key(5), jnp.asarray(e), tm, jnp.int32(T_eff), pr)
    f = jax.jit(
        lambda k, vs: v.sample_eq(
            k, vs, jnp.asarray(e), tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty()
        )
    )
    for it in range(5):
        vs, info = f(key(10 + it), vs)
        assert bool(info.ok)
        seen = np.asarray(vs.core.rb_h_sum[1:])
        assert np.isneginf(seen[[3, 12]]).all()
        assert np.all(seen[T_eff:] == 0.0) and np.all(np.isfinite(np.delete(seen, [3, 12])))
        for leaf in jax.tree.leaves(vs.trans):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.all(np.asarray(v.log_scale2_eq(vs))[T_eff:] == 0.0)


def test_constant_core_with_exact_zero_residuals_is_finite():
    T_eff, T_pad = 21, 32
    tm = tmask_of(T_eff, T_pad)
    v = block(T_pad=T_pad)
    e = np.random.default_rng(6).normal(size=T_pad)
    e[[0, 5]] = 0.0
    e[T_eff:] = np.inf
    pr = eq_prior()
    vs = v.init_eq(key(6), jnp.asarray(e), tm, jnp.int32(T_eff), pr)
    vs, info = v.sample_eq(
        key(7), vs, jnp.asarray(e), tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty()
    )
    assert bool(info.ok) and np.isfinite(float(vs.core.log_lam))
    assert np.isfinite(float(v.loglik_eq(vs, jnp.asarray(e), tm, Empty())))


# --------------------------------------------------------------------------- views vs references


@pytest.mark.parametrize("transitory", ["none", "outlier", "t", "outlier_t"])
def test_views_loglik_and_log_prior_match_the_references(transitory):
    spec = joint_spec("constant_transitory", transitory=transitory, n=1)
    s = spec.static
    v = RB.VolBlockImpl(s)
    refs = REF.references_for(s)
    prior_sample, simulate = jax.jit(spec.prior_sample), jax.jit(spec.simulate)
    for r in range(5):
        st = prior_sample(key(20 + 2 * r))
        data = simulate(key(21 + 2 * r), st)
        e = residuals(st, data, spec.prior, s)
        vs = jax.tree.map(lambda x: x[0], st.eq.vol)
        pr0 = jax.tree.map(lambda x: x[0], spec.prior.eq)
        eff = np.asarray(REF.effective_log_var(st.eq.vol, s))[0]
        pers = np.asarray(REF.effective_log_var(st.eq.vol, s, "persistent"))[0]
        np.testing.assert_allclose(np.asarray(v.log_var_eq(vs, Empty())), eff, rtol=1e-15)
        np.testing.assert_allclose(
            np.asarray(v.log_var_eq(vs, Empty(), "persistent")), pers, rtol=1e-15
        )
        ll = float(v.loglik_eq(vs, e[0], data.tmask, Empty()))
        assert ll == pytest.approx(
            float(jnp.sum(loglik_terms(st, data, spec.prior, s)[0])), rel=1e-12
        )
        lp = float(v.log_prior_eq(vs, pr0, data.tmask))
        want = float(refs.core.log_prior_eq(vs.core, pr0, s, data.tmask)
                     + refs.trans.log_prior_eq(vs.trans, pr0, s, data.tmask))  # fmt: skip
        assert lp == pytest.approx(want, rel=1e-12, abs=1e-10)
        eq0 = jax.tree.map(lambda x: x[0], st.eq)
        assert float(log_prior_eq(eq0, pr0, s, data.tmask)) == pytest.approx(
            lp, rel=1e-12, abs=1e-10
        )
    with pytest.raises(ValueError, match="which"):
        v.log_var_eq(vs, Empty(), "total")


def test_log_scale_eq_only_for_a_coupled_constant_core():
    cs = VolEq(ConstantEq(jnp.float64(0.7)), Empty())
    assert float(block("none", coupled=True).log_scale_eq(cs)) == 0.7
    assert float(block("none").log_scale_eq(cs)) == 0.0


# --------------------------------------------------------------------------- obs_logdensity_eq


def _exact_obs_density(e, log_lam, p, nu, K=20):
    """N03 §2.2 with ``scipy.stats``: ``log Σ_k w_k t(e; ν, scale = k √(λ(ν−2)/ν))``."""
    k = np.arange(1, K + 1)
    w = np.where(k == 1, 1 - p, p / (K - 1))
    scale = k * math.sqrt(math.exp(log_lam) * (nu - 2) / nu)
    return float(sc.logsumexp(np.log(w) + stats.t.logpdf(e, df=nu, scale=scale)))


@pytest.mark.parametrize("z", [1.3, -6.0])
def test_obs_logdensity_matches_exact_sum_and_monte_carlo(z):
    """Standardized SVO-t, monthly SVO prior mean ``p = 1/48``, ν = 5 (N03 §5.3, §7.1 test 11a)."""
    v = block(T_pad=8)
    log_lam, p, nu = math.log(0.6), 1 / 48, 5.0
    e = z * math.exp(0.5 * log_lam)
    vd = VolDraw(
        ConstantDraw(jnp.float64(log_lam)), OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(nu))
    )
    got = float(v.obs_logdensity_eq(jnp.float64(e), jnp.float64(log_lam), vd))
    assert got == pytest.approx(_exact_obs_density(e, log_lam, p, nu), rel=1e-12)
    # Monte Carlo over (o, τ), drawn in NumPy (independent of the kernels): 2·10⁶ draws
    rng = np.random.default_rng(7)
    N = 2_000_000
    o = np.where(rng.random(N) < p, rng.integers(2, 21, N), 1)
    tau = (nu - 2) / 2 / rng.gamma(nu / 2, size=N)
    dens = stats.norm.pdf(e, scale=np.sqrt(math.exp(log_lam) * o**2 * tau))
    mc, se = dens.mean(), dens.std() / math.sqrt(N)
    assert abs(math.exp(got) - mc) < 4 * se
    if abs(z) < 2:
        assert abs(math.log(mc) - got) < 1e-3  # N03 §7.1 test 11a
    # the same with the block's own ahead draws: log_density and ahead are consistent
    _, eff = jax.jit(jax.vmap(lambda k: v.draw_ahead_eq(k, vd, eq_prior(), 1, Empty())))(
        jr.split(key(30), 400_000)
    )
    d2 = stats.norm.pdf(e, scale=np.exp(0.5 * np.asarray(eff)[:, 0]))
    assert abs(math.exp(got) - d2.mean()) < 4 * d2.std() / math.sqrt(d2.size)


def test_obs_logdensity_without_transitory_is_gaussian():
    v = block("none", T_pad=8)
    vd = VolDraw(ConstantDraw(jnp.float64(0.2)), Empty())
    for e in (0.0, 0.4, -3.0):
        got = float(v.obs_logdensity_eq(jnp.float64(e), jnp.float64(0.2), vd))
        assert got == pytest.approx(stats.norm.logpdf(e, scale=math.exp(0.1)), rel=1e-13)


# --------------------------------------------------------------------------- draw_ahead_eq


def test_draw_ahead_constant_core_mean_is_lambda_times_one_plus_150p():
    v = block(T_pad=8)
    p, nu, log_lam, H = 0.02, 10.0, math.log(1.7), 3
    vd = VolDraw(
        ConstantDraw(jnp.float64(log_lam)), OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(nu))
    )
    pers, eff = jax.jit(jax.vmap(lambda k: v.draw_ahead_eq(k, vd, eq_prior(), H, Empty())))(
        jr.split(key(40), 300_000)
    )
    pers, eff = np.asarray(pers), np.asarray(eff)
    assert pers.shape == eff.shape == (300_000, H) and np.all(pers == log_lam)
    lam = np.exp(eff).ravel()
    assert abs(lam.mean() - 1.7 * (1 + 150 * p)) < 4 * lam.std() / math.sqrt(lam.size)
    # the effective part is exactly the core path plus the transitory draw of the same key
    k = key(41)
    a, b = v.draw_ahead_eq(k, vd, eq_prior(), H, Empty())
    np.testing.assert_array_equal(
        np.asarray(b), np.asarray(a) + np.asarray(v.transitory.ahead(k, vd.trans, H))
    )


def test_draw_ahead_sv_core_mean_matches_n03_identity():
    """``E[λ_{T+1} | θ, h_T] = exp(μ + φ(h_T − μ) + σ²/2)·(1 + 150p)`` (standardized t;
    N03 §5.3 item 3), with the stub AR SV core."""
    v = sv_block(T_pad=8)
    mu, phi, sigma, h_T, p = -0.5, 0.9, 0.3, 0.4, 0.03
    vd = VolDraw(SVDraw(jnp.float64(h_T), jnp.float64(mu), jnp.float64(phi), jnp.float64(sigma)),
                 OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(9.0)))  # fmt: skip
    _, eff = jax.jit(jax.vmap(lambda k: v.draw_ahead_eq(k, vd, eq_prior(), 1, Empty())))(
        jr.split(key(42), 400_000)
    )
    lam = np.exp(np.asarray(eff)[:, 0])
    ref = math.exp(mu + phi * (h_T - mu) + sigma**2 / 2) * (1 + 150 * p)
    assert abs(lam.mean() - ref) < 4 * lam.std() / math.sqrt(lam.size)


def _sv_core_block(core: str, transitory="outlier_t", T_pad=32) -> RB.VolBlockImpl:
    """The stub AR SV core, or WP12's registered ``SVCore`` (skip until it lands)."""
    if core == "stub":
        return sv_block(transitory, T_pad)
    try:
        registry.load("sv")
    except NotImplementedError:
        pytest.skip("SVCore (WP12) is not available yet")
    return RB.VolBlockImpl(VolStatic("sv", transitory=transitory, nu_grid=GRID), T_pad)


def _key_word(k) -> np.uint64:
    """The 64-bit key data of a threefry2x32 key."""
    kd = np.asarray(jr.key_data(k)).astype(np.uint64)
    return (kd[0] << np.uint64(32)) | kd[1]


@pytest.mark.parametrize("core", ["stub", "sv"])
def test_ahead_and_transition_give_the_core_its_own_sub_key(core):
    """One consumer per key (DESIGN §3.1 rule 5, §5.6): the core gets ``fold_in(key, Sub.H)``
    in ``draw_ahead_eq`` and ``transition_eq``. A core that draws ``jr.normal(key, (H,))``
    reads the words ``threefry(key, (0, j))``, and these are the key data of
    ``fold_in(key, j)``. Had it been given the block key, ``η_12`` would sit on the ``Sub.O``
    key and ``η_15`` on the ``Sub.TAU`` key of the transitory ahead draws. ``debug_key_reuse``
    cannot see this, because ``fold_in`` is not a consuming use."""
    v = _sv_core_block(core)
    H = 24
    vd = VolDraw(SVDraw(jnp.float64(0.1), jnp.float64(-0.2), jnp.float64(0.9), jnp.float64(0.3)),
                 OutlierTDraw(jnp.log(0.02), jnp.log1p(-0.02), jnp.float64(7.0)))  # fmt: skip
    trans_keys = {int(_key_word(sub_key(key(140 + r), s))) for r in range(8)
                  for s in (Sub.O, Sub.P, Sub.NU, Sub.TAU)}  # fmt: skip
    for r in range(8):
        k = key(140 + r)
        ck = sub_key(k, Sub.H)
        pers, eff = v.draw_ahead_eq(k, vd, eq_prior(), H, Empty())
        # the core consumed fold_in(k, Sub.H); the transitory scale got k (it folds O and TAU)
        np.testing.assert_array_equal(np.asarray(pers), np.asarray(v.core.ahead(ck, vd.core, H)))
        np.testing.assert_array_equal(
            np.asarray(eff), np.asarray(pers + v.transitory.ahead(k, vd.trans, H))
        )
        x = jnp.float64(0.25)
        assert float(v.transition_eq(k, x, vd)) == float(v.core.transition(ck, x, vd.core))
        words = np.asarray(jr.bits(ck, (H,), dtype=jnp.uint64))  # what jr.normal(ck, (H,)) reads
        assert not trans_keys & {int(w) for w in words}
        if jax.config.jax_threefry_partitionable:  # the hazard is real: the block key's words
            raw = np.asarray(jr.bits(k, (H,), dtype=jnp.uint64))
            assert raw[int(Sub.O)] == _key_word(sub_key(k, Sub.O))
            assert raw[int(Sub.TAU)] == _key_word(sub_key(k, Sub.TAU))


@pytest.mark.parametrize("core", ["stub", "sv"])
def test_draw_ahead_sv_core_long_horizon_mean(core):
    """``E λ^eff_{T+j} = exp(μ + φ^j (h_T − μ) + ½σ² (1 − φ^{2j})/(1 − φ²))·(1 + 150p)`` at
    ``j = 1, 12, 15, 24`` (N03 §5.3 item 3), past the ``Sub.O``/``Sub.TAU`` indices."""
    v = _sv_core_block(core)
    mu, phi, sigma, h_T, p, H = -0.5, 0.9, 0.3, 0.4, 0.03, 24
    vd = VolDraw(SVDraw(jnp.float64(h_T), jnp.float64(mu), jnp.float64(phi), jnp.float64(sigma)),
                 OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(9.0)))  # fmt: skip
    _, eff = jax.jit(jax.vmap(lambda k: v.draw_ahead_eq(k, vd, eq_prior(), H, Empty())))(
        jr.split(key(150), 400_000)
    )
    lam = np.exp(np.asarray(eff))
    for j in (1, 12, 15, 24):
        var_h = sigma**2 * (1 - phi ** (2 * j)) / (1 - phi**2)
        ref = math.exp(mu + phi**j * (h_T - mu) + var_h / 2) * (1 + 150 * p)
        col = lam[:, j - 1]
        assert abs(col.mean() - ref) < 4 * col.std() / math.sqrt(col.size), j


def test_extract_and_transition():
    v = block(T_pad=16)
    vs = VolEq(ConstantEq(jnp.float64(0.3)), OutlierTEq(
        jnp.zeros(16, jnp.int32), jnp.log(0.1), jnp.log(0.9), jnp.zeros(16), jnp.int32(4), jnp.zeros(16), jnp.float64(0)))  # fmt: skip
    vd = v.extract_eq(vs, jnp.int32(10))
    assert isinstance(vd.core, ConstantDraw) and isinstance(vd.trans, OutlierTDraw)
    assert float(vd.trans.nu) == 7.0 and float(vd.core.log_lam) == 0.3
    assert float(v.transition_eq(key(43), jnp.float64(0.3), vd)) == 0.3
    sv = sv_block(T_pad=16)
    svs = sv.init_eq(key(44), jnp.ones(16), jnp.ones(16), jnp.int32(12), eq_prior())
    svs = dataclasses.replace(svs, core=dataclasses.replace(svs.core, h=jnp.arange(17.0)))
    assert float(sv.extract_eq(svs, jnp.int32(12)).core.h_T) == 12.0  # h[T_eff], never h[-1]


# --------------------------------------------------------------------------- N03 §7.1 test 8 limit


def test_p_zero_limit_equals_the_no_transitory_block():
    """With ``p = 0`` the outlier step returns ``o ≡ 1``, so the Constant draw equals the one of
    the block without a transitory scale, bitwise (same ``Sub.H`` key)."""
    T_eff, T_pad = 25, 32
    tm = tmask_of(T_eff, T_pad)
    e = jnp.asarray(np.random.default_rng(8).normal(size=T_pad) * 3.0) * tm
    v_out, v_none = block("outlier", T_pad), block("none", T_pad)
    pr = eq_prior()
    vs_out = VolEq(ConstantEq(jnp.float64(0.1)), dataclasses.replace(
        v_out.transitory.init(tm, pr), log_p=-jnp.inf, log1m_p=jnp.float64(0.0)))  # fmt: skip
    a, _ = v_out.sample_eq(key(50), vs_out, e, tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty())
    b, _ = v_none.sample_eq(key(50), VolEq(ConstantEq(jnp.float64(0.1)), Empty()), e, tm,
                            jnp.int32(T_eff), pr, Coupling.zeros(), Empty())  # fmt: skip
    assert np.all(np.asarray(a.trans.o_idx) == 0)
    assert float(a.core.log_lam) == float(b.core.log_lam)


# --------------------------------------------------------------------------- N03 §7.1 test 6


def test_unknown_lambda_with_t_matches_quadrature():
    """Constant ``λ̄ ~ IG(3, 2)`` with standardized t, ν ∈ {3..20}, T = 40: the exact joint
    posterior by quadrature over ``log λ̄`` per grid ν (N03 §7.1 test 6) against Gibbs chains of
    ``VolBlockImpl`` (ν | e, λ̄ with τ integrated → τ → λ̄)."""
    grid = tuple(range(3, 21))
    T_eff, T_pad, a0, b0 = 40, 64, 3.0, 2.0
    rng = np.random.default_rng(9)
    z = rng.standard_t(4, size=T_eff) * math.sqrt(2 / 4)
    e_np = 0.8 * z
    # quadrature: p(λ, ν | e) ∝ IG(λ; a0, b0) π(ν) Π_t t_{ν−2,ν}(e_t/√λ)/√λ
    ll = np.linspace(-6.0, 4.0, 8001)
    lam = np.exp(ll)
    nu = np.asarray(grid, float)
    lp = np.empty((len(grid), ll.size))
    for g, v_ in enumerate(nu):
        sc_t = np.sqrt(lam * (v_ - 2) / v_)
        lp[g] = stats.t.logpdf(e_np[:, None], df=v_, scale=sc_t[None, :]).sum(0)
    lp += stats.invgamma(a0, scale=b0).logpdf(lam)[None, :] + ll[None, :]  # Jacobian dλ = λ dlogλ
    w = np.exp(lp - lp.max())
    w /= w.sum()  # uniform grid in log λ̄ (the Jacobian is in lp)
    p_nu = w.sum(1)
    E_nu, P_le5 = float(p_nu @ nu), float(p_nu[nu <= 5].sum())
    E_ll = float((w.sum(0) * ll).sum())

    v = block("t", T_pad, grid=grid)
    pr = eq_prior(a0, b0, grid=grid)
    tm = tmask_of(T_eff, T_pad)
    e = jnp.asarray(np.concatenate([e_np, np.zeros(T_pad - T_eff)]))
    T = jnp.int32(T_eff)
    burn, keep, C = 100, 500, 400

    def chain(ck):
        vs0 = v.init_eq(jr.fold_in(ck, 2**31), e, tm, T, pr)

        def sweep(vs, it):
            vs, _ = v.sample_eq(jr.fold_in(ck, it), vs, e, tm, T, pr, Coupling.zeros(), Empty())
            g = vs.trans.nu_idx
            f = jnp.stack(
                [jnp.asarray(grid, jnp.float64)[g], (g <= 2).astype(jnp.float64), vs.core.log_lam]
            )
            return vs, f

        _, f = lax.scan(sweep, vs0, jnp.arange(burn + keep, dtype=jnp.uint32))
        return f[burn:].mean(0)

    m = np.asarray(jax.jit(jax.vmap(chain))(jr.split(key(60), C)))
    exact = np.array([E_nu, P_le5, E_ll])
    zsc = (m.mean(0) - exact) / (m.std(0, ddof=1) / math.sqrt(C))
    assert np.all(np.abs(zsc) < 4.0), (zsc, m.mean(0), exact)


# --------------------------------------------------------------------------- N03 §7.1 test 12


def test_scale_equivariance():
    """``e → c·e`` with the IG scale × c² and the start ``log λ + 2 log c``: the same keys give the
    same ``o``, ``ν`` and (to rounding) ``p``, ``τ``, and ``log λ`` shifted by ``2 log c``."""
    T_eff, T_pad, c = 33, 64, 7.3
    tm = tmask_of(T_eff, T_pad)
    rng = np.random.default_rng(10)
    e = rng.standard_t(4, size=T_pad) * 0.5
    e[[4, 17]] = [6.0, -5.0]
    e = jnp.asarray(e) * tm
    v = block(T_pad=T_pad)
    pr1, prc = eq_prior(4.0, 1.5), eq_prior(4.0, 1.5 * c * c)
    T = jnp.int32(T_eff)
    f = jax.jit(lambda k, vs, e, pr: v.sample_eq(k, vs, e, tm, T, pr, Coupling.zeros(), Empty())[0])
    vs1 = v.init_eq(key(70), e, tm, T, pr1)
    vsc = v.init_eq(key(70), c * e, tm, T, prc)
    assert float(vsc.core.log_lam) == pytest.approx(
        float(vs1.core.log_lam) + 2 * math.log(c), abs=1e-12
    )
    for it in range(40):
        vs1, vsc = f(key(100 + it), vs1, e, pr1), f(key(100 + it), vsc, c * e, prc)
        np.testing.assert_array_equal(np.asarray(vs1.trans.o_idx), np.asarray(vsc.trans.o_idx))
        assert int(vs1.trans.nu_idx) == int(vsc.trans.nu_idx)
        np.testing.assert_allclose(
            np.asarray(vsc.trans.log_tau), np.asarray(vs1.trans.log_tau), atol=1e-9
        )
        assert float(vsc.trans.log_p) == pytest.approx(float(vs1.trans.log_p), abs=1e-9)
        assert float(vsc.core.log_lam) - 2 * math.log(c) == pytest.approx(
            float(vs1.core.log_lam), abs=1e-9
        )


# --------------------------------------------------------------------------- invariants under vmap


@pytest.mark.parametrize("transitory", ["none", "outlier", "t", "outlier_t"])
def test_batched_sweeps_keep_the_state_invariants(transitory):
    spec = joint_spec("constant_transitory", transitory=transitory, n=3)
    s = spec.static
    v = RB.VolBlockImpl(s)
    st = jax.jit(spec.prior_sample)(key(80))
    data = jax.jit(spec.simulate)(key(81), st)
    e = residuals(st, data, spec.prior, s)
    n = s.shapes.n

    def one(k, vs, e_i, p_i):
        return v.sample_eq(k, vs, e_i, data.tmask, data.T_eff, p_i, Coupling.zeros(), Empty())

    f = jax.jit(jax.vmap(one))
    vs = jax.vmap(lambda e_i, p_i: v.init_eq(key(82), e_i, data.tmask, data.T_eff, p_i))(
        e, spec.prior.eq
    )
    for it in range(10):
        vs, info = f(jr.split(key(90 + it), n), vs, e, spec.prior.eq)
        assert np.all(np.asarray(info.ok))
    new = dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vs))
    assert check_state_invariants(jax.tree.map(np.asarray, new), s, spec.fixed.T_eff) == []
    # batched lanes equal per-equation calls bitwise
    ks = jr.split(key(99), n)
    b, _ = f(ks, vs, e, spec.prior.eq)
    for i in range(n):
        a, _ = jax.jit(one)(ks[i], jax.tree.map(lambda x, i=i: x[i], vs), e[i],
                            jax.tree.map(lambda x, i=i: x[i], spec.prior.eq))  # fmt: skip
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y)[i])


# --------------------------------------------------------------------------- block Geweke (§9.2)


@dataclass(frozen=True, eq=False)
class VolBlockJoint:
    """``JointModel`` of a zero-mean block configuration whose posterior kernel is one
    ``VolBlockImpl.sample_eq`` per equation (the production sweep of WP14 restricted to the vol
    step, which is all a zero-mean composition has); frozen blocks restored by ``hold_frozen``."""

    spec: JointSpec
    vol: RB.VolBlockImpl

    @property
    def name(self) -> str:
        return f"wp13.{self.spec.name}"

    @property
    def names(self) -> tuple[str, ...]:
        return self.spec.names

    def prior_sample(self, key):
        return self.spec.prior_sample(key)

    def simulate(self, key, st):
        return self.spec.simulate(key, st)

    def test_functions(self, st, data):
        return self.spec.test_functions(st, data)

    def step(self, key, it, st, data):
        s = self.spec.static
        e = residuals(st, data, self.spec.prior, s)
        bk = KY.step_key(key, it, Block.EQ_VOL)

        def one(i, vs, e_i, p_i):
            return self.vol.sample_eq(KY.eq_key(bk, i), vs, e_i, data.tmask, data.T_eff, p_i,
                                      Coupling.zeros(), Empty())  # fmt: skip

        vol, _ = jax.vmap(one)(
            jnp.arange(s.shapes.n, dtype=jnp.int32), st.eq.vol, e, self.spec.prior.eq
        )
        new = dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vol))
        return hold_frozen(new, st, self.spec.live)


@pytest.mark.geweke
@pytest.mark.parametrize("name", ["transitory_known_lambda", "constant_transitory"])
def test_block_geweke(name):
    spec = joint_spec(name)
    assert isinstance(spec.static, StaticConfig) and spec.static.mean.kind == "zero"
    model = VolBlockJoint(spec, RB.VolBlockImpl(spec.static))
    cfg = GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=20261022)  # G-small (§9.5)
    seq, results = geweke_sequential(model, cfg)
    assert seq.passed, results[-1].table().head(10)


# --------------------------------------------------------------------------- keys and the SV core


@pytest.mark.parametrize("core", ["constant", "sv"])
@pytest.mark.parametrize("transitory", ["none", "outlier", "t", "outlier_t"])
def test_no_key_reuse(transitory, core):
    """One consumer per key (DESIGN §5.6): ``sample_eq`` and ``draw_ahead_eq`` under
    ``jax.debug_key_reuse``, with the Constant core and with WP12's ``SVCore`` (fold-in overlaps
    are invisible to it: ``test_ahead_and_transition_give_the_core_its_own_sub_key``)."""
    T_eff, T_pad = 20, 32
    tm = tmask_of(T_eff, T_pad)
    if core == "constant":
        v, pr = block(transitory, T_pad), eq_prior()
    else:
        _sv_core_block("sv")  # skips until WP12's SVCore is registered
        spec = joint_spec("zero_sv", transitory=transitory)
        assert spec.static.shapes.T_pad == T_pad
        v, pr = RB.VolBlockImpl(spec.static), jax.tree.map(lambda x: x[0], spec.prior.eq)
    e = jnp.asarray(np.random.default_rng(11).normal(size=T_pad)) * tm
    with jax.debug_key_reuse(True):
        vs = jax.jit(lambda k: v.init_eq(k, e, tm, jnp.int32(T_eff), pr))(key(110))
        out, info = jax.jit(
            lambda k: v.sample_eq(k, vs, e, tm, jnp.int32(T_eff), pr, Coupling.zeros(), Empty())
        )(key(111))
        jax.jit(lambda k: v.draw_ahead_eq(k, v.extract_eq(out, jnp.int32(T_eff)), pr, 4, Empty()))(
            key(112)
        )
    assert bool(info.ok)


@pytest.mark.parametrize("transitory", ["none", "outlier_t"])
def test_with_the_sv_core_of_wp12(transitory):
    """Integration with the real ``SVCore`` once it is registered (skipped before WP12 lands):
    default construction through the registry, finite sweeps, inert padding, and the cleaned
    residual identity ``log_var_eq(effective) − log_var_eq(persistent) = 2 log o + log τ``."""
    try:
        sv_cls = registry.load("sv")
    except NotImplementedError:
        pytest.skip("SVCore (WP12) is not available yet")
    spec = joint_spec("zero_sv", transitory=transitory)
    s = spec.static
    v = RB.VolBlockImpl(s)
    assert isinstance(v.core, sv_cls)
    st = jax.jit(spec.prior_sample)(key(120))
    data = jax.jit(spec.simulate)(key(121), st)
    e = residuals(st, data, spec.prior, s)

    def one(k, vs, e_i, p_i):
        return v.sample_eq(k, vs, e_i, data.tmask, data.T_eff, p_i, Coupling.zeros(), Empty())

    f = jax.jit(jax.vmap(one))
    vs = jax.vmap(lambda e_i, p_i: v.init_eq(key(122), e_i, data.tmask, data.T_eff, p_i))(
        e, spec.prior.eq
    )
    for it in range(10):
        vs, info = f(jr.split(key(130 + it), s.shapes.n), vs, e, spec.prior.eq)
        assert np.all(np.asarray(info.ok))
    new = dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vs))
    assert check_state_invariants(jax.tree.map(np.asarray, new), s, spec.fixed.T_eff) == []
    vs0 = jax.tree.map(lambda x: x[0], vs)
    np.testing.assert_allclose(
        np.asarray(v.log_var_eq(vs0, Empty()) - v.log_var_eq(vs0, Empty(), "persistent")),
        np.asarray(v.log_scale2_eq(vs0)),
        rtol=0,
        atol=1e-12,
    )
