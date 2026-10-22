"""Constant persistent core with the coupling channel (DESIGN §13 WP13; §3.7.5, §4.7; D-30;
N05 §5.2; N03 §4.5).

The IG conditional ``λ_i | · ~ IG(c + T_eff/2 + n_active/2, C + ½ Σ m_t ẽ²_t + quad)`` against hand
sums (masks, exact zeros, coupling consumed only when ``coupled``); draws against
``scipy.stats.invgamma``; density-ratio identities (Grosse & Duvenaud 2014 eq. 1; DESIGN §9.3
item 2) at 20 joint draws against the independent ``validation.log_joint`` for Mode I, Mode II,
``iw_equivalent`` (coupling sums computed here from the state and the prior arrays) and Constant
with outliers + t; the cold start; the reference log prior; the M13 override seam.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random as jr
from scipy import special as sc
from scipy import stats

from rai_bvar.components.vol import constant as CN
from rai_bvar.components.vol.transitory import masked_log_z2
from rai_bvar.engines.mcmc import protocols as P
from rai_bvar.engines.mcmc import registry
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.keys import Sub, sub_key
from rai_bvar.engines.mcmc.state import (
    ConstantDraw,
    ConstantEq,
    ConstantPrior,
    Coupling,
    Empty,
    EqPrior,
)
from rai_bvar.validation import reference as REF
from rai_bvar.validation.joint import joint_spec
from rai_bvar.validation.log_joint import log_joint, loglik_terms, residuals
from rai_bvar.validation.sbc import block_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = jr.key(20261024, impl="threefry2x32")


def key(i: int):
    return jr.fold_in(KEY, i)


def eq_prior(shape=5.0, scale=4.0) -> EqPrior:
    return EqPrior(Empty(), Empty(), ConstantPrior(jnp.float64(shape), jnp.float64(scale)), Empty())


def tmask_of(T_eff: int, T_pad: int) -> jax.Array:
    return (jnp.arange(T_pad) < T_eff).astype(jnp.float64)


def core(coupled=False, T_pad=64, transitory="none") -> CN.ConstantCore:
    return CN.ConstantCore(VolStatic("constant", coupled=coupled, transitory=transitory), T_pad)


# --------------------------------------------------------------------------- construction


def test_registry_protocol_and_constructors():
    assert registry.load("constant") is CN.ConstantCore
    spec = joint_spec("structural_constant")
    a = CN.ConstantCore(spec.static)  # the StaticConfig is unpacked
    b = CN.ConstantCore(spec.static.vol, spec.static.shapes.T_pad)
    assert a == b and hash(a) == hash(b) and a.T_pad == 32 and isinstance(a, P.PersistentCore)
    with pytest.raises(TypeError, match="T_pad"):
        CN.ConstantCore(VolStatic("constant"))
    with pytest.raises(ValueError, match="persistent='constant'"):
        CN.ConstantCore(VolStatic("sv"), 8)
    with pytest.raises(ValueError, match="contradicts"):
        CN.ConstantCore(spec.static, 64)


# --------------------------------------------------------------------------- the IG conditional


@pytest.mark.parametrize("coupled", [False, True])
def test_conditional_formula_masks_exact_zeros_and_coupling(coupled):
    T_eff, T_pad = 37, 64
    rng = np.random.default_rng(1)
    e = rng.normal(size=T_pad) * 1.7
    e[[4, 11]] = 0.0  # exact zeros: log ẽ² = −∞ contributes 0
    with np.errstate(divide="ignore"):
        le2 = np.log(e * e)
    le2[T_eff:] = rng.choice([np.nan, np.inf, 800.0], size=T_pad - T_eff)  # padded garbage
    tm = tmask_of(T_eff, T_pad)
    cpl = Coupling(jnp.float64(7.0), jnp.float64(2.3))
    k = core(coupled)
    shape, scale = k.conditional(jnp.asarray(le2), tm, jnp.int32(T_eff), eq_prior(2.5, 0.8), cpl)
    ssr = float(np.sum(e[:T_eff] ** 2))
    n_act, quad = (7.0, 2.3) if coupled else (0.0, 0.0)  # D-30: consumed only when coupled
    assert float(shape) == pytest.approx(2.5 + T_eff / 2 + n_act / 2, rel=1e-15)
    assert float(scale) == pytest.approx(0.8 + ssr / 2 + quad, rel=1e-14)
    # the pure internal always applies what it is given
    s2, c2 = CN.constant_conditional(jnp.asarray(le2), tm, jnp.int32(T_eff),
                                     ConstantPrior(2.5, 0.8), cpl)  # fmt: skip
    assert float(s2) == pytest.approx(2.5 + T_eff / 2 + 3.5) and float(c2) == pytest.approx(
        0.8 + ssr / 2 + 2.3
    )


def test_draws_are_inverse_gamma():
    T_eff, T_pad = 21, 32
    rng = np.random.default_rng(2)
    e = rng.normal(size=T_pad) * 0.6
    tm = tmask_of(T_eff, T_pad)
    le2 = jnp.where(tm > 0, jnp.log(jnp.asarray(e) ** 2), 0.0)
    k = core(T_pad=T_pad)
    pr = eq_prior()
    cs = ConstantEq(jnp.float64(0.0))
    out, info = jax.jit(
        jax.vmap(lambda kk: k.sample(kk, cs, le2, tm, jnp.int32(T_eff), pr, Coupling.zeros()))
    )(jr.split(key(3), 200_000))
    assert np.all(np.asarray(info.ok)) and np.all(np.isinf(np.asarray(info.min_pivot)))
    shape, scale = 5.0 + T_eff / 2, 4.0 + 0.5 * np.sum(e[:T_eff] ** 2)
    ll = np.asarray(out.log_lam)
    assert stats.kstest(np.exp(ll), stats.invgamma(shape, scale=scale).cdf).pvalue > 1e-4
    se = math.sqrt(sc.polygamma(1, shape) / ll.size)
    assert abs(ll.mean() - (math.log(scale) - sc.digamma(shape))) < 4 * se


def test_sample_uses_sub_key_H_and_reports_failures():
    k = core(T_pad=8)
    tm, pr = jnp.ones(8), eq_prior()
    le2 = jnp.zeros(8)
    out, _ = k.sample(
        key(4), ConstantEq(jnp.float64(0.0)), le2, tm, jnp.int32(8), pr, Coupling.zeros()
    )
    shape, scale = k.conditional(le2, tm, jnp.int32(8), pr, Coupling.zeros())
    from rai_bvar.engines.mcmc import random as R

    ref = jnp.log(scale) - R.loggamma(sub_key(key(4), Sub.H), shape)
    assert float(out.log_lam) == float(ref)
    _, info = k.sample(key(4), ConstantEq(jnp.float64(0.0)), le2, tm, jnp.int32(8),
                       eq_prior(-10.0, 4.0), Coupling.zeros())  # shape′ < 0: NaN draw  # fmt: skip
    assert not bool(info.ok)  # failures are values (D-18)


def test_m13_mutant_changes_the_shape():
    T_eff, T_pad = 20, 32
    tm = tmask_of(T_eff, T_pad)
    k = core(T_pad=T_pad)
    m13 = k.with_overrides(
        {"constant_conditional": lambda le2, tm, T, cp, cpl: CN.constant_conditional(
            le2, tm, jnp.float64(tm.shape[0]), cp, cpl)}
    )  # fmt: skip
    assert m13 != k and hash(m13) != hash(k)
    a = k.conditional(jnp.zeros(T_pad), tm, jnp.int32(T_eff), eq_prior(), Coupling.zeros())
    b = m13.conditional(jnp.zeros(T_pad), tm, jnp.int32(T_eff), eq_prior(), Coupling.zeros())
    assert float(b[0]) - float(a[0]) == pytest.approx((T_pad - T_eff) / 2)


# --------------------------------------------------------------------------- density ratios


def _coupling_by_hand(spec, st, i):
    """``(n_active, quad)`` of equation ``i`` from the state and the λ-free prior arrays
    (DESIGN §4.7): Mode II every active coordinate, ``iw_equivalent`` the ``a``-slots only."""
    s = spec.static
    pr = jax.tree.map(lambda x: x[i], spec.prior.eq)
    a = st.eq.struct.a[i]
    cm = pr.contemp.cmask > 0
    n_act, quad = jnp.zeros(()), jnp.zeros(())
    if s.struct.contemp_scaling == "lambda":
        n_act += jnp.sum(cm)
        quad += 0.5 * jnp.sum(jnp.where(cm, pr.contemp.prec0 * (a - pr.contemp.mean0) ** 2, 0.0))
    if s.mean.coef_scaling == "lambda":
        d = st.eq.mean.delta[i]
        n_act += d.size
        quad += 0.5 * jnp.sum(pr.coef.prec0 * (d - pr.coef.mean0) ** 2)
    return Coupling(n_act.astype(jnp.float64), quad)


def _with_log_lam(st, i, value):
    ll = st.eq.vol.core.log_lam.at[i].set(value)
    vol = dataclasses.replace(st.eq.vol, core=ConstantEq(log_lam=ll))
    return dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vol))


@pytest.mark.parametrize(
    ("name", "overrides", "expect_coupled"),
    [
        ("structural_constant", {}, False),  # Mode I
        ("structural_mode2_constant", {}, True),  # Mode II: n_active = k + i
        ("structural_iw_equivalent", {}, True),  # a-slots only
        ("constant_transitory", {}, False),  # λ | e, o, τ (N03 §4.5)
        ("constant_transitory", {"transitory": "outlier"}, False),
    ],
)
def test_density_ratio_identity_against_log_joint(name, overrides, expect_coupled):
    """``log q(λ′) − log q(λ) = log p(λ′, ·) − log p(λ, ·)`` per equation at 20 joint draws."""
    spec = joint_spec(name, **overrides)
    s = spec.static
    assert s.vol.coupled == expect_coupled
    k = CN.ConstantCore(s)
    trans = registry.load(s.vol.transitory)(s)

    def pair(kk, st, data, i):
        tm = data.tmask
        e = residuals(st, data, spec.prior, s)[i]
        pr_i = jax.tree.map(lambda x: x[i], spec.prior.eq)
        trans_i = jax.tree.map(lambda x: x[i], st.eq.vol.trans)
        le2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, e * e, 1.0)), 0.0)
        le2c = jnp.where(tm > 0, le2 - trans.log_scale2(trans_i), 0.0)
        cpl = _coupling_by_hand(spec, st, i)
        shape, scale = k.conditional(le2c, tm, data.T_eff, pr_i, cpl)
        old = st.eq.vol.core.log_lam[i]
        new, info = k.sample(kk, ConstantEq(old), le2c, tm, data.T_eff, pr_i, cpl)
        lhs = CN.log_ig_density(new.log_lam, shape, scale) - CN.log_ig_density(old, shape, scale)
        lj = log_joint(_with_log_lam(st, i, new.log_lam), data, spec.prior, s) - log_joint(
            st, data, spec.prior, s
        )
        return lhs, lj, info.ok

    n = s.shapes.n
    fn = jax.jit(jax.vmap(pair, in_axes=(0, None, None, 0)))
    prior_sample, simulate = jax.jit(spec.prior_sample), jax.jit(spec.simulate)
    for r in range(20):
        st = prior_sample(key(100 + 2 * r))
        data = simulate(key(101 + 2 * r), st)
        lhs, rhs, ok = (np.asarray(x) for x in fn(jr.split(key(5000 + r), n), st, data,
                                                   jnp.arange(n)))  # fmt: skip
        assert ok.all()
        assert np.all(np.abs(lhs - rhs) / (1 + np.abs(rhs)) < 1e-8), (name, r, lhs, rhs)


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("structural_constant", {}),
        ("structural_mode2_constant", {}),
        ("structural_iw_equivalent", {}),
        ("constant_transitory", {}),
    ],
)
def test_block_conditional_sbc_of_lambda(name, overrides):
    """Tier B (DESIGN §9.4; N06 §5.2): ``(θ̃, ỹ)`` from the joint, 99 iid draws of every ``λ_i``
    from its conditional given ``θ̃_−λ`` (coupling from the state), ranks of ``log λ_i`` and of the
    log-likelihood; N = 200, Bonferroni at 1e-3."""
    spec = joint_spec(name, **overrides)
    s = spec.static
    k = CN.ConstantCore(s)
    trans = registry.load(s.vol.transitory)(s)
    n = s.shapes.n

    def joint_sample(kk):
        st = spec.prior_sample(jr.fold_in(kk, 0))
        return st, spec.simulate(jr.fold_in(kk, 1), st)

    def draws(kk, st, data):
        tm = data.tmask
        e = residuals(st, data, spec.prior, s)

        def one(k_i, i):
            pr_i = jax.tree.map(lambda x: x[i], spec.prior.eq)
            trans_i = jax.tree.map(lambda x: x[i], st.eq.vol.trans)
            le2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, e[i] ** 2, 1.0)), 0.0)
            le2c = jnp.where(tm > 0, le2 - trans.log_scale2(trans_i), 0.0)
            cs = ConstantEq(st.eq.vol.core.log_lam[i])
            out, _ = k.sample(k_i, cs, le2c, tm, data.T_eff, pr_i, _coupling_by_hand(spec, st, i))
            return out.log_lam

        ll = jax.vmap(one)(jr.split(kk, n), jnp.arange(n))
        vol = dataclasses.replace(st.eq.vol, core=ConstantEq(log_lam=ll))
        return dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vol))

    def stat(st, data):
        ll = jnp.sum(loglik_terms(st, data, spec.prior, s))
        return jnp.concatenate([st.eq.vol.core.log_lam, ll[None]])

    res = block_sbc(joint_sample, draws, stat, N=200, L=99, seed=17, alpha=1e-3)
    assert res.passed, res.table()


# --------------------------------------------------------------------------- init, views, prior


def test_init_is_the_log_mean_square_with_a_prior_fallback():
    T_eff, T_pad = 25, 32
    rng = np.random.default_rng(5)
    e = rng.normal(size=T_pad) * 2.0
    tm = tmask_of(T_eff, T_pad)
    le2 = jnp.where(tm > 0, jnp.log(jnp.asarray(e) ** 2), 123.0)  # padded garbage ignored
    k = core(T_pad=T_pad)
    cs = k.init(key(6), le2, tm, jnp.int32(T_eff), eq_prior())
    assert float(cs.log_lam) == pytest.approx(math.log(np.mean(e[:T_eff] ** 2)), rel=1e-13)
    zeros = jnp.where(tm > 0, -jnp.inf, 0.0)  # every real residual exactly 0
    cs0 = k.init(key(6), zeros, tm, jnp.int32(T_eff), eq_prior(5.0, 4.0))
    assert float(cs0.log_lam) == pytest.approx(math.log(4.0 / 6.0))  # IG mode C/(c+1)


def test_views_extract_ahead_transition_and_log_prior():
    k_unc, k_cpl = core(False, 16), core(True, 16)
    cs = ConstantEq(jnp.float64(-0.4))
    lv = np.asarray(k_unc.log_var(cs))
    assert lv.shape == (16,) and np.all(lv == -0.4)
    assert float(k_unc.log_scale(cs)) == 0.0 and float(k_cpl.log_scale(cs)) == -0.4
    d = k_unc.extract(cs, jnp.int32(10))
    assert isinstance(d, ConstantDraw) and float(d.log_lam) == -0.4
    assert np.all(np.asarray(k_unc.ahead(key(7), d, 5)) == -0.4)
    assert float(k_unc.transition(key(8), jnp.float64(-0.4), d)) == -0.4
    spec = joint_spec("structural_constant")
    ref = REF.references_for(spec.static).core
    pr0 = jax.tree.map(lambda x: x[0], spec.prior.eq)
    for v in (-3.0, 0.1, 2.5):
        got = float(k_unc.log_prior(ConstantEq(jnp.float64(v)), pr0, spec.tmask))
        want = float(ref.log_prior_eq(ConstantEq(jnp.float64(v)), pr0, spec.static, spec.tmask))
        assert got == pytest.approx(want, rel=1e-13)
        assert got == pytest.approx(stats.invgamma(5.0, scale=4.0).logpdf(math.exp(v)), rel=1e-12)


def test_vmap_over_equations_equals_per_equation_calls():
    n, T_pad, T_eff = 4, 32, 30
    k = core(T_pad=T_pad)
    tm = tmask_of(T_eff, T_pad)
    rng = np.random.default_rng(9)
    le2 = jnp.asarray(rng.normal(size=(n, T_pad)))
    pr = EqPrior(Empty(), Empty(), ConstantPrior(jnp.arange(1.0, n + 1), jnp.full(n, 2.0)), Empty())
    cs = ConstantEq(jnp.zeros(n))
    keys = jr.split(key(10), n)

    def one(kk, c, l2, p):
        return k.sample(kk, c, l2, tm, jnp.int32(T_eff), p, Coupling.zeros())

    batched, _ = jax.jit(jax.vmap(one))(keys, cs, le2, pr)
    for i in range(n):
        single, _ = jax.jit(one)(keys[i], ConstantEq(cs.log_lam[i]), le2[i],
                                 jax.tree.map(lambda x, i=i: x[i], pr))  # fmt: skip
        assert float(single.log_lam) == float(batched.log_lam[i])


def test_masked_log_z2_helper_shared_with_transitory():
    tm = tmask_of(3, 5)
    out = np.asarray(
        masked_log_z2(jnp.asarray([1.0, -jnp.inf, 2.0, 9.0, 9.0]), jnp.full(5, 0.5), tm)
    )
    assert out[0] == 0.5 and np.isneginf(out[1]) and out[2] == 1.5 and np.all(out[3:] == 0.0)
