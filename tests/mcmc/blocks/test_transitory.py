"""Transitory scales: outliers, Student-t, outliers + t (DESIGN §13 WP13; §3.7.6; D-8, D-9; N03).

N03 §7.1 tests 1–4 (conditionals against brute force: the ``o`` pmf against ``w_k N(e; 0,
e^h k² τ)`` and ``w_k k⁻¹ t_{c,ν}(z/k)`` from ``scipy.stats``, ``p | o``, ``τ``, the ν grid
against ``scipy.stats.t.logpdf``), test 5 (SVO with known ``λ̄``: quadrature posterior of ``p`` and
of every ``P(o_t > 1 | e)``), test 7 (partially collapsed vs uncollapsed ν: same stationary law,
ESS ratio > 3), test 8 (limits), test 9 (``ahead`` draws: binomial frequency, uniform ``o | o > 1``,
IG ``τ``, ``E o² = 1 + 150p``); density-ratio identities of every step against the independent
references of ``validation`` (D-22); block-conditional SBC of the exact blocks (tier B), with
statistics on the large-``|z|`` rows and a heavy-tailed configuration for the τ-collapsed ``o``
step, whose power against two wrong collapsed-``o`` conditionals (τ ignored; ``c = ν``) is itself
tested, as is the χ² of the ``o_step`` draws; the override seam (mutants M4, M9, M10); padding and
exact zeros. Test 6 (unknown ``λ̄``) and test 12 (scale equivariance) need the Constant core and
live in ``test_robust.py``.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax import random as jr
from scipy import special as sc
from scipy import stats

from rai_bvar.components.vol import transitory as TR
from rai_bvar.engines.mcmc import protocols as P
from rai_bvar.engines.mcmc import random as RND
from rai_bvar.engines.mcmc import registry
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.keys import Sub, sub_key
from rai_bvar.engines.mcmc.state import (
    ConstantPrior,
    Empty,
    EqPrior,
    OutlierDraw,
    OutlierEq,
    OutlierTDraw,
    OutlierTEq,
    TDraw,
    TEq,
    TransitoryPrior,
)
from rai_bvar.validation import reference as REF
from rai_bvar.validation.joint import joint_spec
from rai_bvar.validation.log_joint import log_joint, residuals
from rai_bvar.validation.sbc import block_sbc, iat

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = jr.key(20261023, impl="threefry2x32")
GRID = tuple(range(3, 41))
K20 = 20


def key(i: int):
    return jr.fold_in(KEY, i)


def vstatic(transitory: str, *, K: int = K20, grid=GRID, t_scale="standardized") -> VolStatic:
    return VolStatic("constant", transitory=transitory, outlier_K=K, nu_grid=grid, t_scale=t_scale)


def eq_prior(a_p=2.0, b_p=18.0, grid=GRID, log_nu_w=None) -> EqPrior:
    G = len(grid)
    w = np.full(G, -math.log(G)) if log_nu_w is None else np.asarray(log_nu_w, np.float64)
    tp = TransitoryPrior(a_p=jnp.float64(a_p), b_p=jnp.float64(b_p), log_nu_w=jnp.asarray(w))
    return EqPrior(Empty(), Empty(), ConstantPrior(jnp.float64(5.0), jnp.float64(4.0)), tp)


def tmask_of(T_eff: int, T_pad: int) -> jax.Array:
    return (jnp.arange(T_pad) < T_eff).astype(jnp.float64)


def c_of(nu, t_scale="standardized"):
    return nu - 2.0 if t_scale == "standardized" else nu


def norm_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x - sc.logsumexp(x, axis=-1, keepdims=True)


def chi2_pvalue(counts: np.ndarray, probs: np.ndarray, min_expected: float = 5.0) -> float:
    """χ² goodness of fit with bins of expected count below ``min_expected`` pooled."""
    counts = np.asarray(counts, np.float64)
    exp = probs * counts.sum()
    order = np.argsort(exp)
    c_s, e_s = counts[order], exp[order]
    pooled_c, pooled_e, acc_c, acc_e = [], [], 0.0, 0.0
    for c, e in zip(c_s, e_s, strict=True):
        acc_c += c
        acc_e += e
        if acc_e >= min_expected:
            pooled_c.append(acc_c)
            pooled_e.append(acc_e)
            acc_c = acc_e = 0.0
    if acc_e > 0 and pooled_e:
        pooled_c[-1] += acc_c
        pooled_e[-1] += acc_e
    pooled_c, pooled_e = np.array(pooled_c), np.array(pooled_e)
    if len(pooled_e) < 2:
        return 1.0
    stat = np.sum((pooled_c - pooled_e) ** 2 / pooled_e)
    return float(stats.chi2.sf(stat, len(pooled_e) - 1))


# --------------------------------------------------------------------------- construction


@pytest.mark.parametrize(
    ("kind", "cls"),
    [("none", TR.NoTransitory), ("outlier", TR.OutlierScale), ("t", TR.TScale),
     ("outlier_t", TR.OutlierTScale)],
)  # fmt: skip
def test_registry_construction_and_protocol(kind, cls):
    assert registry.load(kind) is cls
    k = cls(vstatic(kind), 32)
    assert isinstance(k, P.TransitoryScale)
    assert k == cls(vstatic(kind), 32) and hash(k) == hash(cls(vstatic(kind), 32))
    with pytest.raises(ValueError, match="implements transitory"):
        cls(vstatic("none" if kind != "none" else "t"), 32)
    with pytest.raises(TypeError, match="T_pad"):
        cls(vstatic(kind))


def test_invalid_grids_and_outlier_K_raise():
    with pytest.raises(ValueError, match="outlier_K"):
        TR.OutlierScale(vstatic("outlier", K=1), 8)  # w_k = p/(K−1) undefined (N03 §2.1)
    with pytest.raises(ValueError, match="c\\(ν\\)"):
        TR.TScale(vstatic("t", grid=(2, 3)), 8)  # standardized needs ν > 2
    TR.TScale(vstatic("t", grid=(2, 3), t_scale="unit"), 8)  # unit scale allows ν = 2


def test_nu_constants_are_numpy_and_exact():
    nu, log_c, A = TR.nu_constants(GRID, "standardized")
    assert isinstance(nu, np.ndarray) and not nu.flags.writeable
    c = nu - 2.0
    np.testing.assert_allclose(log_c, np.log(c), rtol=0, atol=0)
    ref = sc.gammaln((nu + 1) / 2) - sc.gammaln(nu / 2) - 0.5 * np.log(np.pi * c)
    np.testing.assert_allclose(A, ref, rtol=1e-14, atol=1e-14)


# --------------------------------------------------------------------------- N03 §7.1 test 1: o


def _o_brute_gaussian(e, h, tau, p, K):
    k = np.arange(1, K + 1)
    w = np.where(k == 1, 1 - p, p / (K - 1))
    sd = np.sqrt(np.exp(h) * k**2 * tau)
    return norm_log(np.log(w) + stats.norm.logpdf(e, scale=sd))


def _o_brute_collapsed(e, h, p, nu, K, t_scale="standardized"):
    k = np.arange(1, K + 1)
    w = np.where(k == 1, 1 - p, p / (K - 1))
    scale = k * np.sqrt(c_of(nu, t_scale) / nu) * np.exp(h / 2)  # k⁻¹ t_{c,ν}(z/k) in e
    return norm_log(np.log(w) + stats.t.logpdf(e, df=nu, scale=scale))


CASES = [(0.3, 0.0, 1.0), (-2.5, -1.0, 0.7), (8.0, 0.5, 1.3), (40.0, -2.0, 1.0), (0.0, 0.0, 1.0)]


@pytest.mark.parametrize(("e", "h", "tau"), CASES)
def test_outlier_pmf_gaussian_matches_brute_force(e, h, tau):
    p = 0.07
    log_z2 = np.log(e * e) - h - np.log(tau) if e != 0 else -np.inf
    lg = TR.outlier_logits(jnp.asarray([log_z2]), jnp.log(p), jnp.log1p(-p), K20)
    got = norm_log(np.asarray(lg))[0]
    ref = _o_brute_gaussian(e, h, tau, p, K20)
    np.testing.assert_allclose(np.exp(got), np.exp(ref), rtol=0, atol=1e-14)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
@pytest.mark.parametrize(("e", "h"), [(0.3, 0.0), (-2.5, -1.0), (8.0, 0.5), (40.0, -2.0), (0, 0)])
def test_outlier_pmf_collapsed_matches_scipy_t(e, h, t_scale):
    p, nu = 0.05, 4.0
    log_z2 = np.log(e * e) - h if e != 0 else -np.inf
    lg = TR.outlier_logits(
        jnp.asarray([log_z2]), jnp.log(p), jnp.log1p(-p), K20, jnp.float64(nu),
        jnp.log(c_of(nu, t_scale)),
    )  # fmt: skip
    got = norm_log(np.asarray(lg))[0]
    np.testing.assert_allclose(got, _o_brute_collapsed(e, h, p, nu, K20, t_scale), rtol=1e-12,
                               atol=1e-12)  # fmt: skip


def test_outlier_logits_extreme_residual_and_dropped_one_over_k():
    # log z² far beyond exp overflow: finite logits, o = K with probability one (clamp, pilot §5)
    lg = np.asarray(TR.outlier_logits(jnp.asarray([2000.0]), jnp.log(0.02), jnp.log1p(-0.02), K20))
    assert np.all(np.isfinite(lg)) and np.argmax(lg[0]) == K20 - 1
    lg_t = np.asarray(
        TR.outlier_logits(jnp.asarray([2000.0]), jnp.log(0.02), jnp.log1p(-0.02), K20,
                          jnp.float64(5.0), jnp.log(3.0))
    )  # fmt: skip
    assert np.all(np.isfinite(lg_t)) and np.argmax(lg_t[0]) == K20 - 1
    # p = 0: only o = 1 has mass
    lg0 = np.asarray(TR.outlier_logits(jnp.asarray([3.0]), -jnp.inf, 0.0, K20))
    assert np.isfinite(lg0[0, 0]) and np.all(lg0[0, 1:] == -np.inf)


def test_o_step_draws_follow_the_pmf():
    kern = TR.OutlierScale(vstatic("outlier"), 1)
    e, h, p = 5.0, 0.0, 0.1
    N = 200_000
    log_z2 = jnp.full((N,), math.log(e * e) - h)
    o, pi = jax.jit(lambda k: kern.o_step(k, log_z2, jnp.ones(N), jnp.log(p), jnp.log1p(-p)))(
        key(1)
    )
    probs = np.exp(_o_brute_gaussian(e, h, 1.0, p, K20))
    counts = np.bincount(np.asarray(o), minlength=K20)
    assert chi2_pvalue(counts, probs) > 1e-4
    np.testing.assert_allclose(np.asarray(pi), 1 - probs[0], rtol=1e-12)
    # collapsed form
    kern_t = TR.OutlierTScale(vstatic("outlier_t"), 1)
    nu_idx = jnp.int32(GRID.index(4))
    o2, pi2 = jax.jit(
        lambda k: kern_t.o_step(k, log_z2, jnp.ones(N), jnp.log(p), jnp.log1p(-p), nu_idx)
    )(key(2))
    probs2 = np.exp(_o_brute_collapsed(e, h, p, 4.0, K20))
    assert chi2_pvalue(np.bincount(np.asarray(o2), minlength=K20), probs2) > 1e-4
    np.testing.assert_allclose(np.asarray(pi2), 1 - probs2[0], rtol=1e-12)


Z_ROWS = (0.5, 2.5, 4.0, 8.0, 30.0)


def _collapsed_o_draws(kern, nu, p, N, k):
    """``o_step`` (τ integrated) on ``N`` copies of each row of ``Z_ROWS`` in one call; draws
    ``(rows, N)``."""
    lz = jnp.repeat(jnp.log(jnp.asarray(Z_ROWS) ** 2), N)
    nu_idx = jnp.int32(GRID.index(nu))
    f = jax.jit(lambda k: kern.o_step(k, lz, jnp.ones(lz.size), jnp.log(p), jnp.log1p(-p), nu_idx))
    o, _ = f(k)
    return np.asarray(o).reshape(len(Z_ROWS), N)


@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
@pytest.mark.parametrize("nu", [3, 12])
def test_collapsed_o_step_draws_follow_the_t_mixture_pmf(nu, t_scale):
    """The sampler (not only the logits): ``OutlierTScale.o_step`` draws on rows from the bulk to
    far tails against ``w_k k⁻¹ t_{c,ν}(z/k)`` from ``scipy.stats`` (N03 §4.1, §7.1 test 1), χ²
    per row; and the two collapsed-``o`` mutants of ``COLLAPSED_O_MUTANTS`` fail it."""
    kern = TR.OutlierTScale(vstatic("outlier_t", t_scale=t_scale), 1)
    p, N = 0.1, 40_000
    probs = [np.exp(_o_brute_collapsed(z, 0.0, p, float(nu), K20, t_scale)) for z in Z_ROWS]
    o = _collapsed_o_draws(kern, nu, p, N, key(3 + nu))
    for r, z in enumerate(Z_ROWS):
        assert chi2_pvalue(np.bincount(o[r], minlength=K20), probs[r]) > 1e-4, z
    for name, fn in COLLAPSED_O_MUTANTS.items():
        if name == "c_is_nu" and t_scale == "unit":
            continue  # c(ν) = ν is the exact unit-scale conditional
        om = _collapsed_o_draws(kern.with_overrides({"outlier_logits": fn}), nu, p, N, key(9))
        pv = [chi2_pvalue(np.bincount(om[r], minlength=K20), probs[r]) for r in range(len(Z_ROWS))]
        assert min(pv) < 1e-10, (name, pv)


# --------------------------------------------------------------------------- test 2: p | o


def test_p_conditional_counts_real_rows_only_and_draws_beta():
    T_eff, T_pad = 37, 64
    rng = np.random.default_rng(3)
    o = np.zeros(T_pad, np.int32)
    o[rng.choice(T_eff, 6, replace=False)] = rng.integers(1, K20, 6)
    o[T_eff + 2] = 5  # garbage in a padded row is never counted
    tm = tmask_of(T_eff, T_pad)
    a, b = TR.p_conditional(jnp.asarray(o), tm, jnp.int32(T_eff), 2.5, 117.5)
    assert float(a) == 2.5 + 6 and float(b) == 117.5 + T_eff - 6
    kern = TR.OutlierScale(vstatic("outlier"), T_pad)
    tp = eq_prior(2.5, 117.5).transitory
    draws = jax.jit(jax.vmap(lambda k: kern.p_step(k, jnp.asarray(o), tm, jnp.int32(T_eff), tp)))(
        jr.split(key(4), 100_000)
    )
    lp, l1p = (np.asarray(x) for x in draws)
    np.testing.assert_allclose(np.exp(lp) + np.exp(l1p), 1.0, rtol=0, atol=1e-14)
    ref = stats.beta(8.5, 117.5 + T_eff - 6)
    x = np.exp(lp)
    se = ref.std() / math.sqrt(x.size)
    assert abs(x.mean() - ref.mean()) < 4 * se
    assert abs(x.var() / ref.var() - 1) < 4 * math.sqrt(2 / x.size) * 1.5
    assert stats.kstest(x, ref.cdf).pvalue > 1e-4


# --------------------------------------------------------------------------- test 3: τ


@pytest.mark.parametrize(("nu", "r2", "t_scale"), [(8, 2.3, "standardized"), (3, 40.0, "unit"),
                                                    (3, 0.0, "standardized")])  # fmt: skip
def test_tau_draws_are_inverse_gamma(nu, r2, t_scale):
    st = vstatic("t", t_scale=t_scale)
    kern = TR.TScale(st, 50_000)
    c = c_of(float(nu), t_scale)
    log_r2 = jnp.full((50_000,), math.log(r2) if r2 > 0 else -np.inf)
    shape, log_scale = TR.tau_conditional(log_r2[:1], jnp.float64(nu), jnp.log(c))
    assert float(shape) == (nu + 1) / 2
    np.testing.assert_allclose(np.exp(np.asarray(log_scale)), (c + r2) / 2, rtol=1e-14)
    lt = np.asarray(
        jax.jit(lambda k: kern.tau_step(k, log_r2, jnp.ones(50_000), jnp.int32(GRID.index(nu))))(
            key(5)
        )
    )
    alpha, beta = (nu + 1) / 2, (c + r2) / 2
    ref = stats.invgamma(alpha, scale=beta)
    assert stats.kstest(np.exp(lt), ref.cdf).pvalue > 1e-4
    # E log τ = log β − ψ(α) (finite at every ν)
    se = math.sqrt(sc.polygamma(1, alpha) / lt.size)
    assert abs(lt.mean() - (math.log(beta) - sc.digamma(alpha))) < 4 * se
    if alpha > 2:  # IG variance finite only for α > 2 (N03 §7.1 test 3)
        x = np.exp(lt)
        assert abs(x.mean() - ref.mean()) < 4 * ref.std() / math.sqrt(x.size)


# --------------------------------------------------------------------------- test 4: ν grid


@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
def test_nu_logits_match_scipy_t_product(t_scale):
    rng = np.random.default_rng(7)
    T_eff, T_pad = 37, 64
    r = rng.standard_t(4, size=T_pad)
    r[3], r[4], r[5] = 0.0, 30.0, math.exp(300.0)  # exact zero, outlier, huge (r² ≈ 1e260)
    with np.errstate(divide="ignore"):
        log_r2 = np.where(np.arange(T_pad) < T_eff, np.log(r * r), 7.0)  # garbage on padded rows
    log_nu_w = np.log(rng.dirichlet(np.ones(len(GRID))))
    got = TR.nu_logits(jnp.asarray(log_r2), tmask_of(T_eff, T_pad), jnp.int32(T_eff),
                       jnp.asarray(log_nu_w), GRID, t_scale)  # fmt: skip
    nu = np.asarray(GRID, np.float64)
    scale = np.sqrt(c_of(nu, t_scale) / nu)
    ref = log_nu_w + np.array(
        [stats.t.logpdf(r[:T_eff], df=v, scale=s).sum() for v, s in zip(nu, scale, strict=True)]
    )
    np.testing.assert_allclose(norm_log(np.asarray(got)), norm_log(ref), rtol=0, atol=1e-10)


def test_nu_logits_large_log_r2_pass_through_and_zero_prior_mass():
    # log r² = 750 (r² overflows float64): the row contributes log r² − log c_g exactly
    nu, log_c, A = TR.nu_constants(GRID, "standardized")
    lg = np.asarray(TR.nu_logits(jnp.asarray([750.0, -np.inf]), jnp.ones(2), jnp.int32(2),
                                 jnp.zeros(len(GRID)), GRID, "standardized"))  # fmt: skip
    ref = 2 * A - 0.5 * (nu + 1) * (750.0 - log_c)
    np.testing.assert_allclose(lg, ref, rtol=1e-14)
    w = np.full(len(GRID), -np.inf)
    w[GRID.index(5)] = 0.0  # DiscreteUniform(5, 5): ν fixed (stochvol SV-t parity)
    lg2 = np.asarray(TR.nu_logits(jnp.asarray([0.3, 1.0]), jnp.ones(2), jnp.int32(2),
                                  jnp.asarray(w), GRID, "standardized"))  # fmt: skip
    assert np.isfinite(lg2[GRID.index(5)]) and np.sum(np.isfinite(lg2)) == 1
    kern = TR.TScale(vstatic("t"), 2)
    tp = eq_prior(log_nu_w=w).transitory
    g = jax.vmap(lambda k: kern.nu_step(k, jnp.asarray([0.3, 1.0]), jnp.ones(2), jnp.int32(2), tp))(
        jr.split(key(8), 200)
    )
    assert np.all(np.asarray(g) == GRID.index(5))


def test_nu_step_draws_follow_the_exact_pmf():
    rng = np.random.default_rng(9)
    grid = tuple(range(3, 13))
    st = vstatic("t", grid=grid)
    T = 40
    r = rng.standard_t(5, size=T) * math.sqrt(3 / 5)
    log_r2 = jnp.asarray(np.log(r * r))
    kern = TR.TScale(st, T)
    tp = eq_prior(grid=grid).transitory
    g = jax.jit(jax.vmap(lambda k: kern.nu_step(k, log_r2, jnp.ones(T), jnp.int32(T), tp)))(
        jr.split(key(10), 100_000)
    )
    nu = np.asarray(grid, np.float64)
    logp = np.array([stats.t.logpdf(r, df=v, scale=math.sqrt((v - 2) / v)).sum() for v in nu])
    probs = np.exp(norm_log(logp))
    assert chi2_pvalue(np.bincount(np.asarray(g), minlength=len(grid)), probs) > 1e-4


# --------------------------------------------------------------------------- test 5: SVO, known λ̄


def _svo_exact_posterior(z, a, b, K, grid_n=40_001):
    """Quadrature over logit(p) (N03 §7.1 test 5): weights, p, π_t(p)."""
    u = np.linspace(-14.0, 4.0, grid_n)
    p = sc.expit(u)
    k = np.arange(2, K + 1)
    phi1 = stats.norm.logpdf(z)  # (T,)
    phik = sc.logsumexp(stats.norm.logpdf(z[:, None] / k) - np.log(k), axis=1) - math.log(K - 1)
    l_out = np.log(p)[:, None] + phik[None, :]
    l_reg = np.log1p(-p)[:, None] + phi1[None, :]
    lik = np.logaddexp(l_out, l_reg)  # (Ngrid, T)
    logpost = (a - 1) * np.log(p) + (b - 1) * np.log1p(-p) + lik.sum(1) + np.log(p) + np.log1p(-p)
    w = np.exp(logpost - logpost.max())
    w /= np.trapezoid(w, u)
    pi_t = np.exp(l_out - lik)  # P(o_t > 1 | p, e)
    return u, p, w, pi_t


def test_svo_known_lambda_matches_quadrature():
    """Gibbs (o, p) with known λ̄, T = 50, three planted outliers (N03 §7.1 test 5)."""
    rng = np.random.default_rng(11)
    T_eff, T_pad, lam = 50, 64, 0.5
    z = rng.normal(size=T_eff)
    z[[5, 20, 41]] = [8.0, -11.0, 14.0]
    a, b = 2.0, 18.0
    u, p, w, pi_t = _svo_exact_posterior(z, a, b, K20)

    def integ(f):
        return float(np.trapezoid(w * f, u))

    Ep, Ep2 = integ(p), integ(p**2)
    cdf = np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(u))
    qs = [float(np.interp(q, cdf, p[1:])) for q in (0.05, 0.5, 0.95)]
    Epi = np.array([integ(pi_t[:, t]) for t in range(T_eff)])

    kern = TR.OutlierScale(vstatic("outlier"), T_pad)
    prior_i = eq_prior(a, b)
    tm = tmask_of(T_eff, T_pad)
    e = np.zeros(T_pad)
    e[:T_eff] = math.sqrt(lam) * z
    log_e2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, jnp.asarray(e) ** 2, 1.0)), 0.0)
    lvp = jnp.full((T_pad,), math.log(lam))
    T_eff_a = jnp.int32(T_eff)
    burn, keep, C = 100, 400, 500
    q_arr = jnp.asarray(qs)

    def chain(ck):
        ts0 = kern.init(tm, prior_i)

        def sweep(ts, it):
            return kern.sample(jr.fold_in(ck, it), ts, log_e2, lvp, tm, T_eff_a, prior_i), None

        ts, _ = lax.scan(sweep, ts0, jnp.arange(burn, dtype=jnp.uint32))
        ts = dataclasses.replace(ts, rb_out=jnp.zeros_like(ts.rb_out))  # the burn-in boundary

        def sweep2(carry, it):
            ts, s = carry
            ts = kern.sample(jr.fold_in(ck, it), ts, log_e2, lvp, tm, T_eff_a, prior_i)
            pp = jnp.exp(ts.log_p)
            s = s + jnp.concatenate([jnp.stack([pp, pp * pp]), (pp <= q_arr).astype(jnp.float64)])
            return (ts, s), None

        (ts, s), _ = lax.scan(
            sweep2, (ts, jnp.zeros(5)), jnp.arange(burn, burn + keep, dtype=jnp.uint32)
        )
        return s / keep, ts.rb_out / keep

    s, rb = jax.jit(jax.vmap(chain))(jr.split(key(12), C))
    s, rb = np.asarray(s), np.asarray(rb)
    exact = np.array([Ep, Ep2, 0.05, 0.5, 0.95])
    mean, mcse = s.mean(0), s.std(0, ddof=1) / math.sqrt(C)
    z_scores = (mean - exact) / mcse
    assert np.all(np.abs(z_scores) < 4.0), z_scores
    # posterior s.d. of p
    sd_mc = math.sqrt(mean[1] - mean[0] ** 2)
    assert abs(sd_mc / math.sqrt(Ep2 - Ep**2) - 1) < 0.02
    # every Rao-Blackwellised P(o_t > 1 | e), within 4.5 MCSE (50 comparisons)
    rb_mean, rb_se = rb.mean(0)[:T_eff], rb.std(0, ddof=1)[:T_eff] / math.sqrt(C)
    zt = (rb_mean - Epi) / np.maximum(rb_se, 1e-12)
    assert np.all(np.abs(zt) < 4.5), zt
    assert np.all(rb[:, T_eff:] == 0.0)  # padded rows never accumulate
    assert Epi[[5, 20, 41]].min() > 0.9 and np.median(Epi) < 0.1  # the planted outliers


# --------------------------------------------------------------------------- test 7: PCG


def test_collapsed_nu_matches_uncollapsed_and_mixes_better():
    """Known λ̄, standardized t, ν ∈ {3..40}: collapsed ν | e then τ vs plain Gibbs τ | ν,
    ν | τ (N03 §7.1 test 7). Same stationary law; ESS ratio > 3."""
    rng = np.random.default_rng(13)
    T = 40
    r = rng.standard_t(5, size=T) * math.sqrt(3 / 5)
    log_r2 = jnp.asarray(np.log(r * r))
    tm, T_eff = jnp.ones(T), jnp.int32(T)
    st = vstatic("t")
    kern = TR.TScale(st, T)
    prior_i = eq_prior()
    tp = prior_i.transitory
    nu_g = jnp.asarray(np.asarray(GRID, np.float64))
    exact = np.exp(
        norm_log(np.asarray(TR.nu_logits(log_r2, tm, T_eff, tp.log_nu_w, GRID, st.t_scale)))
    )
    E_nu, P_le5 = float(exact @ np.asarray(GRID)), float(exact[:3].sum())

    def unc_logits(log_tau):
        nu = nu_g
        ll = T * (0.5 * nu * jnp.log(0.5 * (nu - 2.0)) - sc_gammaln(0.5 * nu))
        return (
            tp.log_nu_w
            + ll
            - 0.5 * nu * jnp.sum(log_tau)
            - 0.5 * (nu - 2.0) * jnp.sum(jnp.exp(-log_tau))
        )

    burn, M, C = 200, 2000, 100

    def chain(ck, collapsed: bool):
        ts0 = kern.init(tm, prior_i)

        def sweep(ts, it):
            k = jr.fold_in(ck, it)
            if collapsed:
                ts = kern.sample(k, ts, log_r2, jnp.zeros(T), tm, T_eff, prior_i)
            else:
                log_tau = kern.tau_step(k, log_r2, tm, ts.nu_idx)  # τ | ν
                u = jr.uniform(sub_key(k, Sub.NU), (), dtype=jnp.float64)
                g = jnp.minimum(
                    jnp.sum(jnp.cumsum(jax.nn.softmax(unc_logits(log_tau))) <= u), len(GRID) - 1
                )
                ts = TEq(log_tau=log_tau, nu_idx=g.astype(jnp.int32), acc_nu_move=ts.acc_nu_move)
            return ts, ts.nu_idx

        _, trace = lax.scan(sweep, ts0, jnp.arange(burn + M, dtype=jnp.uint32))
        return trace[burn:]

    tr_c = np.asarray(jax.jit(jax.vmap(lambda k: chain(k, True)))(jr.split(key(14), C)))
    tr_u = np.asarray(jax.jit(jax.vmap(lambda k: chain(k, False)))(jr.split(key(15), C)))
    for tr in (tr_c, tr_u):
        nu = np.asarray(GRID)[tr]
        for f, ref in ((nu, E_nu), ((nu <= 5).astype(float), P_le5)):
            cm = f.mean(1)
            assert abs(cm.mean() - ref) < 4 * cm.std(ddof=1) / math.sqrt(C)
    # collapsed ν draws are iid here (λ̄ known): ν pmf by χ² on the pooled draws
    assert chi2_pvalue(np.bincount(tr_c.ravel(), minlength=len(GRID)), exact) > 1e-4
    iat_c = np.median(iat(np.asarray(GRID, float)[tr_c]))
    iat_u = np.median(iat(np.asarray(GRID, float)[tr_u]))
    assert iat_u / iat_c > 3.0, (iat_c, iat_u)


def sc_gammaln(x):
    return jax.scipy.special.gammaln(x)


# --------------------------------------------------------------------------- test 8: limits


def test_limit_p_zero_gives_no_outliers():
    T_pad, T_eff = 32, 25
    tm = tmask_of(T_eff, T_pad)
    kern = TR.OutlierScale(vstatic("outlier"), T_pad)
    rng = np.random.default_rng(15)
    log_z2 = jnp.asarray(np.log(rng.normal(size=T_pad) ** 2) + 30.0 * (np.arange(T_pad) == 4))
    ts = OutlierEq(jnp.zeros(T_pad, jnp.int32), -jnp.inf, jnp.float64(0.0), jnp.zeros(T_pad))
    o, pi = kern.o_step(key(16), log_z2, tm, ts.log_p, ts.log1m_p)
    assert np.all(np.asarray(o) == 0) and np.all(np.asarray(pi) == 0.0)
    assert np.all(np.asarray(kern.log_scale2(dataclasses.replace(ts, o_idx=o))) == 0.0)
    a, b = TR.p_conditional(o, tm, jnp.int32(T_eff), 2.0, 18.0)
    assert (float(a), float(b)) == (2.0, 18.0 + T_eff)


def test_limit_large_nu_is_gaussian():
    big = (10_000,)
    st = vstatic("t", grid=big)
    kern = TR.TScale(st, 4000)
    lt = np.asarray(
        kern.tau_step(key(17), jnp.full((4000,), math.log(1.7)), jnp.ones(4000), jnp.int32(0))
    )
    assert abs(lt.mean()) < 5e-3 and lt.std() < 0.02  # τ → 1 (sd ≈ √(2/ν))
    lz = jnp.linspace(-6.0, 1.5, 41)  # z² ≤ 4.5: the O(z⁴/ν) gap stays below 1e-3
    dens = np.asarray(kern.log_density(lz, TDraw(nu=jnp.float64(10_000.0))))
    np.testing.assert_allclose(
        dens, -0.5 * np.log(2 * np.pi) - 0.5 * np.exp(np.asarray(lz)), atol=2e-3
    )
    g = np.asarray(TR.outlier_logits(lz, jnp.log(0.1), jnp.log(0.9), K20))
    gt = np.asarray(
        TR.outlier_logits(lz, jnp.log(0.1), jnp.log(0.9), K20, jnp.float64(1e4), jnp.log(1e4 - 2))
    )
    np.testing.assert_allclose(norm_log(gt), norm_log(g), atol=2e-3)


# --------------------------------------------------------------------------- test 9: ahead


def test_outlier_ahead_frequency_uniformity_and_second_moment():
    H, p = 400_000, 0.03
    kern = TR.OutlierScale(vstatic("outlier"), 8)
    x = np.asarray(
        jax.jit(lambda k: kern.ahead(k, OutlierDraw(jnp.log(p), jnp.log1p(-p)), H))(key(18))
    )
    o = np.rint(np.exp(x / 2)).astype(int)
    np.testing.assert_allclose(np.exp(x / 2), o, rtol=1e-12)  # integer scales
    n_out = int((o > 1).sum())
    assert stats.binomtest(n_out, H, p).pvalue > 1e-4
    counts = np.bincount(o[o > 1], minlength=K20 + 1)[2:]
    assert chi2_pvalue(counts, np.full(K20 - 1, 1 / (K20 - 1))) > 1e-4
    o2 = o.astype(float) ** 2
    assert abs(o2.mean() - (1 + 150 * p)) < 4 * o2.std() / math.sqrt(H)  # E o² = 1 + 150p (K = 20)
    assert np.all(
        np.asarray(kern.ahead(key(19), OutlierDraw(-jnp.inf, jnp.float64(0.0)), 100)) == 0
    )


def test_general_K_second_moment():
    H, p, K = 400_000, 0.05, 7
    kern = TR.OutlierScale(vstatic("outlier", K=K), 8)
    o2 = np.exp(np.asarray(kern.ahead(key(20), OutlierDraw(jnp.log(p), jnp.log1p(-p)), H)))
    ref = 1 + p * ((2 * K * K + 5 * K + 6) / 6 - 1)  # N03 §2.3, general K
    assert abs(o2.mean() - ref) < 4 * o2.std() / math.sqrt(H)
    assert o2.max() == pytest.approx(K * K, rel=1e-14)


@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
def test_t_ahead_is_inverse_gamma(t_scale):
    nu = 7.0
    kern = TR.TScale(vstatic("t", t_scale=t_scale), 8)
    lt = np.asarray(kern.ahead(key(21), TDraw(nu=jnp.float64(nu)), 100_000))
    ref = stats.invgamma(nu / 2, scale=c_of(nu, t_scale) / 2)
    assert stats.kstest(np.exp(lt), ref.cdf).pvalue > 1e-4


def test_outlier_t_ahead_mean_is_one_plus_150p():
    H, p, nu = 400_000, 0.02, 10.0
    kern = TR.OutlierTScale(vstatic("outlier_t"), 8)
    lam = np.exp(
        np.asarray(kern.ahead(key(22), OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(nu)), H))
    )
    assert abs(lam.mean() - (1 + 150 * p)) < 4 * lam.std() / math.sqrt(H)


# --------------------------------------------------------------------------- densities, priors


@pytest.mark.parametrize("kind", ["none", "outlier", "t", "outlier_t"])
@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
def test_log_density_matches_reference_and_scipy(kind, t_scale):
    st = vstatic(kind, grid=tuple(range(5, 13)), t_scale=t_scale)
    kern = registry.load(kind)(st, 8)
    p, nu_idx = 0.04, 3
    nu = float(st.nu_grid[nu_idx])
    td = {
        "none": Empty(),
        "outlier": OutlierDraw(jnp.log(p), jnp.log1p(-p)),
        "t": TDraw(jnp.float64(nu)),
        "outlier_t": OutlierTDraw(jnp.log(p), jnp.log1p(-p), jnp.float64(nu)),
    }[kind]
    trans_ref = {
        "none": Empty(),
        "outlier": OutlierEq(jnp.zeros(1, jnp.int32), jnp.log(p), jnp.log1p(-p), jnp.zeros(1)),
        "t": TEq(jnp.zeros(1), jnp.int32(nu_idx), jnp.float64(0.0)),
        "outlier_t": OutlierTEq(jnp.zeros(1, jnp.int32), jnp.log(p), jnp.log1p(-p), jnp.zeros(1),
                                jnp.int32(nu_idx), jnp.zeros(1), jnp.float64(0.0)),
    }[kind]  # fmt: skip
    spec = joint_spec("constant_transitory", transitory=kind, nu_grid=st.nu_grid, t_scale=t_scale)
    z = np.array([0.0, 0.2, -1.1, 3.0, -9.0, 35.0])
    with np.errstate(divide="ignore"):
        lz = jnp.asarray(np.log(z * z))
    got = np.asarray(kern.log_density(lz, td))
    ref = np.asarray(REF.transitory_log_density(lz, trans_ref, spec.static))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    # independent scipy forms (N03 §2.2)
    k = np.arange(1, K20 + 1)
    w = np.where(k == 1, 1 - p, p / (K20 - 1))
    sc_t = math.sqrt(c_of(nu, t_scale) / nu)
    f = {
        "none": stats.norm.logpdf(z),
        "outlier": sc.logsumexp(np.log(w) + stats.norm.logpdf(z[:, None], scale=k), axis=1),
        "t": stats.t.logpdf(z, df=nu, scale=sc_t),
        "outlier_t": sc.logsumexp(
            np.log(w) + stats.t.logpdf(z[:, None], df=nu, scale=k * sc_t), axis=1
        ),
    }[kind]
    np.testing.assert_allclose(got, f, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ["none", "outlier", "t", "outlier_t"])
def test_log_prior_matches_reference(kind):
    spec = joint_spec("constant_transitory", transitory=kind)
    st_ = spec.static
    kern = registry.load(kind)(st_)
    ref = REF.references_for(st_).trans
    tm = spec.tmask
    pr = jax.tree.map(lambda x: x[0], spec.prior.eq)
    for i in range(5):
        st = spec.prior_sample(key(30 + i))
        ts = jax.tree.map(lambda x: x[0], st.eq.vol.trans)
        got = float(kern.log_prior(ts, pr, tm))
        want = float(ref.log_prior_eq(ts, pr, st_, tm))
        assert got == pytest.approx(want, rel=1e-12, abs=1e-10)
        np.testing.assert_allclose(
            np.asarray(kern.log_scale2(ts)),
            np.asarray(ref.latent_log_var_eq(ts, "effective", T_pad=st_.shapes.T_pad)),
            rtol=0, atol=0,
        )  # fmt: skip


# --------------------------------------------------------------------------- init, padding, zeros


def test_init_values():
    T_pad, T_eff = 32, 20
    tm = tmask_of(T_eff, T_pad)
    pr = eq_prior(2.5, 37.5)
    o = TR.OutlierTScale(vstatic("outlier_t"), T_pad).init(tm, pr)
    assert o.o_idx.dtype == jnp.int32 and np.all(np.asarray(o.o_idx) == 0)
    assert float(jnp.exp(o.log_p)) == pytest.approx(2.5 / 40, rel=1e-14)
    assert float(jnp.exp(o.log_p) + jnp.exp(o.log1m_p)) == pytest.approx(1.0, abs=1e-15)
    assert GRID[int(o.nu_idx)] == 21  # grid median of {3..40} (uniform prior; DESIGN §5.4)
    assert np.all(np.asarray(o.log_tau) == 0) and float(o.acc_nu_move) == 0
    w = np.full(len(GRID), -np.inf)
    w[GRID.index(5)] = 0.0
    t = TR.TScale(vstatic("t"), T_pad).init(tm, eq_prior(log_nu_w=w))
    assert GRID[int(t.nu_idx)] == 5  # prior median when the prior is a point mass


@pytest.mark.parametrize("kind", ["outlier", "t", "outlier_t"])
def test_sample_keeps_padding_inert_and_handles_exact_zeros(kind):
    T_pad, T_eff = 32, 21
    tm = tmask_of(T_eff, T_pad)
    kern = registry.load(kind)(vstatic(kind), T_pad)
    pr = eq_prior()
    rng = np.random.default_rng(23)
    e = rng.normal(size=T_pad) * np.exp(rng.normal(size=T_pad))
    e[[2, 9]] = 0.0  # exact zeros on real rows
    e[T_eff:] = np.nan  # anything on padded rows
    log_e2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, jnp.asarray(e) ** 2, 1.0)), 0.0)
    assert np.isneginf(np.asarray(log_e2)[[2, 9]]).all()
    ts = kern.init(tm, pr)
    f = jax.jit(
        lambda k, ts: kern.sample(k, ts, log_e2, jnp.full(T_pad, -0.3), tm, jnp.int32(T_eff), pr)
    )
    for it in range(20):
        ts = f(key(40 + it), ts)
        for leaf in jax.tree.leaves(ts):
            assert np.all(np.isfinite(np.asarray(leaf)))
        s2 = np.asarray(kern.log_scale2(ts))
        assert np.all(s2[T_eff:] == 0.0)
        if hasattr(ts, "o_idx"):
            assert np.all(np.asarray(ts.o_idx)[T_eff:] == 0) and np.all(
                np.asarray(ts.rb_out)[T_eff:] == 0
            )
        if hasattr(ts, "log_tau"):
            assert np.all(np.asarray(ts.log_tau)[T_eff:] == 0)


def _row(tree, i: int):
    return jax.tree.map(lambda x: x[i], tree)


def test_extract_and_vmap_over_equations():
    n, T_pad, T_eff = 3, 32, 30
    kern = TR.OutlierTScale(vstatic("outlier_t"), T_pad)
    pr = jax.tree.map(lambda x: jnp.stack([x] * n), eq_prior())
    tm = tmask_of(T_eff, T_pad)
    ts = jax.vmap(lambda p: kern.init(tm, p))(pr)
    log_e2 = jnp.asarray(np.random.default_rng(1).normal(size=(n, T_pad)))
    keys = jr.split(key(50), n)
    zeros = jnp.zeros(T_pad)

    def one_eq(k, t, le2, p):
        return kern.sample(k, t, le2, zeros, tm, jnp.int32(T_eff), p)

    batched = jax.jit(jax.vmap(one_eq))(keys, ts, log_e2, pr)
    for i in range(n):
        one = jax.jit(one_eq)(keys[i], _row(ts, i), log_e2[i], _row(pr, i))
        for a, b in zip(jax.tree.leaves(one), jax.tree.leaves(_row(batched, i)), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14, atol=0)
    d = jax.vmap(kern.extract)(batched)
    assert isinstance(d, OutlierTDraw) and d.nu.dtype == jnp.float64 and d.nu.shape == (n,)
    np.testing.assert_array_equal(
        np.asarray(d.nu), np.asarray(GRID, float)[np.asarray(batched.nu_idx)]
    )


# --------------------------------------------------------------------------- sub-keys (DESIGN §5.6)


def _o_draw(kk, logits, live):
    """``o_step``'s draw from the key ``kk``: one uniform per row, inverse CDF (D-4)."""
    u = jr.uniform(kk, live.shape, dtype=jnp.float64)
    return jnp.where(live, RND.categorical_inverse_cdf(u, logits), 0)


def _p_draw(kk, a, b):
    """``p_step``'s ``(log p, log(1 − p))`` from the key ``kk`` (two log-gamma draws)."""
    return jnp.stack(RND.log_beta(kk, a, b))


def _nu_draw(kk, logits):
    """``nu_step``'s grid index from the key ``kk`` (one uniform)."""
    return RND.categorical_inverse_cdf(jr.uniform(kk, (), dtype=jnp.float64), logits)


def _tau_draw(kk, shape, log_scale, live):
    """``tau_step``'s ``log τ`` from the key ``kk`` (one log-gamma per row)."""
    return jnp.where(live, log_scale - RND.loggamma(kk, shape, log_scale.shape), 0.0)


def _over_keys(fn):
    """``fn`` jitted and mapped over a stack of keys (the other arguments shared)."""
    return jax.jit(lambda keys, *args: jax.vmap(lambda kk: fn(kk, *args))(keys))


@pytest.mark.parametrize("kind", ["outlier", "t", "outlier_t"])
def test_each_step_draws_from_its_own_sub_key(kind):
    """One consumer per key (DESIGN §3.1 rule 5, §5.6; module docstring). From the equation key
    ``k`` that ``sample`` receives, ``o`` reads ``sub_key(k, Sub.O)`` (one uniform per row), ``p``
    ``Sub.P`` (two log-gamma draws), ``ν`` ``Sub.NU`` (one uniform) and ``τ`` ``Sub.TAU`` (one
    log-gamma per row): each step is reproduced exactly from its own sub-key and the conditionals
    of this module. On the same inputs, the step fed any other key (another transitory step's
    sub-key, a core's — the SV core's ``S … NC_MS_ACC``, the Constant core's ``H`` — or ``k``
    itself) gives a different draw for at least one of the six keys, so a step that shares a key
    fails here. ``jax.debug_key_reuse`` cannot see such a mutant: every ``fold_in`` returns a
    fresh key."""
    T_pad, T_eff = 48, 44
    tm, Te = tmask_of(T_eff, T_pad), jnp.int32(T_eff)
    live = tm > 0
    kern = registry.load(kind)(vstatic(kind), T_pad)
    pr = eq_prior()
    tp = pr.transitory
    e = np.random.default_rng(31).standard_t(5, size=T_pad) * math.sqrt(3 / 5)
    e[[4, 17, 30]] = [7.0, -9.0, 12.0]  # planted outliers: rows with an uncertain o_t
    log_e2 = jnp.where(live, jnp.log(jnp.asarray(e) ** 2), 0.0)
    lvp = jnp.full(T_pad, -0.2)  # the current persistent log variance
    log_z2 = TR.masked_log_z2(log_e2, lvp, tm)
    ts0 = kern.init(tm, pr)  # p at its prior mean, ν at its prior median
    nu, log_c, _ = (jnp.asarray(a) for a in TR.nu_constants(GRID, "standardized"))
    run = jax.jit(lambda k: kern.sample(k, ts0, log_e2, lvp, tm, Te, pr))
    draws = {"o": _over_keys(_o_draw), "p": _over_keys(_p_draw), "nu": _over_keys(_nu_draw),
             "tau": _over_keys(_tau_draw)}  # fmt: skip
    power: dict[tuple[str, str], bool] = {}

    def check(step, own, k, args, got):
        """``got`` is the step's draw from ``sub_key(k, own)``; record which of the wrong keys
        would have changed it."""
        wrong = {sub.name: sub_key(k, sub) for sub in Sub if sub != own} | {"eq_key": k}
        drawn = np.asarray(draws[step](jnp.stack([sub_key(k, own), *wrong.values()]), *args))
        got = np.asarray(got)
        if got.dtype.kind == "i":
            np.testing.assert_array_equal(got, drawn[0], err_msg=step)
        else:
            np.testing.assert_allclose(got, drawn[0], rtol=1e-13, atol=0.0, err_msg=step)
        for name, alt in zip(wrong, drawn[1:], strict=True):
            hit = not np.allclose(alt.astype(np.float64), got, rtol=1e-12, atol=0.0)
            power[(step, name)] = power.get((step, name), False) or hit

    for r in range(6):
        k = key(300 + r)
        out = run(k)
        log_r2 = log_z2
        if kind != "t":
            nu_c = (nu[ts0.nu_idx], log_c[ts0.nu_idx]) if kind == "outlier_t" else ()
            lg_o = TR.outlier_logits(log_z2, ts0.log_p, ts0.log1m_p, K20, *nu_c)
            check("o", Sub.O, k, (lg_o, live), out.o_idx)
            ab = TR.p_conditional(out.o_idx, tm, Te, tp.a_p, tp.b_p)
            check("p", Sub.P, k, ab, jnp.stack([out.log_p, out.log1m_p]))
            log_o2 = 2.0 * jnp.log(out.o_idx.astype(jnp.float64) + 1.0)
            log_r2 = jnp.where(live, log_z2 - log_o2, 0.0)
        if kind != "outlier":
            lg_nu = TR.nu_logits(log_r2, tm, Te, tp.log_nu_w, GRID, "standardized")
            check("nu", Sub.NU, k, (lg_nu,), out.nu_idx)
            shape, log_scale = TR.tau_conditional(log_r2, nu[out.nu_idx], log_c[out.nu_idx])
            check("tau", Sub.TAU, k, (shape, log_scale, live), out.log_tau)
    assert power and all(power.values()), sorted(sk for sk, hit in power.items() if not hit)


def test_ahead_draws_o_and_tau_from_their_own_sub_keys():
    """``OutlierTScale.ahead`` (N03 §5.2; module docstring): ``2 log o`` from
    ``sub_key(k, Sub.O)`` (one uniform per step: ``o = 1`` if ``u ≥ p``, else
    ``2 + ⌊(u/p)(K − 1)⌋``) plus ``log τ = log(c(ν)/2) − loggamma(ν/2)`` from
    ``sub_key(k, Sub.TAU)``, reproduced exactly: the two parts never share a key."""
    H, nu = 64, 7.0
    kern = TR.OutlierTScale(vstatic("outlier_t"), 8)
    td = OutlierTDraw(jnp.log(0.3), jnp.log1p(-0.3), jnp.float64(nu))
    p = float(jnp.exp(td.log_p))
    ahead = jax.jit(lambda k: kern.ahead(k, td, H))
    parts = jax.jit(
        lambda k: (
            jr.uniform(sub_key(k, Sub.O), (H,), dtype=jnp.float64),
            RND.loggamma(sub_key(k, Sub.TAU), jnp.float64(nu / 2), (H,)),
        )
    )
    for r in range(3):
        k = key(320 + r)
        u, lg = (np.asarray(x) for x in parts(k))
        o = np.where(u < p, np.minimum(2.0 + np.floor(u / p * (K20 - 1)), K20), 1.0)
        assert 0 < np.sum(o > 1) < H  # both branches of the o draw are exercised
        ref = 2.0 * np.log(o) + math.log((nu - 2.0) / 2.0) - lg
        np.testing.assert_allclose(np.asarray(ahead(k)), ref, rtol=1e-13, atol=1e-13)


# --------------------------------------------------------------------------- override seam


def test_overrides_reach_the_named_internals():
    T_pad, T_eff = 16, 16
    tm = jnp.ones(T_pad)
    pr = eq_prior()
    base = TR.OutlierTScale(vstatic("outlier_t"), T_pad)
    calls = []

    def spy(name, fn):
        def wrapped(*a, **k):
            calls.append(name)  # recorded at trace time, in program order
            return fn(*a, **k)

        return wrapped

    names = ("outlier_logits", "p_conditional", "nu_logits", "tau_conditional")
    spied = base.with_overrides({n: spy(n, getattr(TR, n)) for n in names})
    assert spied != base and spied.impl("nu_logits") is not TR.nu_logits
    rng = np.random.default_rng(2)
    log_e2 = jnp.asarray(rng.normal(size=T_pad) + 6.0 * (np.arange(T_pad) == 3))
    ts0 = base.init(tm, pr)

    def run(kern):
        f = jax.jit(
            lambda k: kern.sample(k, ts0, log_e2, jnp.zeros(T_pad), tm, jnp.int32(T_eff), pr)
        )
        return jax.tree.leaves(f(key(60)))

    ref = run(base)
    for x, y in zip(ref, run(spied), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert calls == list(names)  # D-9 order o → p → ν → τ
    # mutants change the conditionals they target (M9 no 1/k, M10 b_p + T, M4 dof ν)
    mutants = {
        "outlier_logits": lambda lz, lp, l1p, K, nu=None, lc=None: (
            TR.outlier_logits(lz, lp, l1p, K, nu, lc) + jnp.log(jnp.arange(1.0, K + 1))
        ),
        "p_conditional": lambda o, tm, T, a, b: (a + jnp.sum(o > 0), b + T),
        "tau_conditional": lambda lr, nu, lc: (0.5 * nu, TR.tau_conditional(lr, nu, lc)[1]),
    }
    for name, fn in mutants.items():
        got = run(base.with_overrides({name: fn}))
        assert any(
            not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(got, ref, strict=True)
        ), name


# --------------------------------------------------------------------------- density-ratio identities


def _states(spec, n_states=20, seed=70):
    """``n_states`` draws ``(θ, y)`` from the joint (the independent WP9 references)."""
    prior_sample, simulate = jax.jit(spec.prior_sample), jax.jit(spec.simulate)
    out = []
    for r in range(n_states):
        st = prior_sample(key(seed + 2 * r))
        out.append((st, simulate(key(seed + 2 * r + 1), st)))
    return out


def _joint_fn(spec):
    return jax.jit(lambda st, data: log_joint(st, data, spec.prior, spec.static))


def _eq0(tree):
    return jax.tree.map(lambda x: x[0], tree)


def _with_trans(st, ts):
    ts_b = jax.tree.map(lambda x: x[None], ts)
    return dataclasses.replace(
        st, eq=dataclasses.replace(st.eq, vol=dataclasses.replace(st.eq.vol, trans=ts_b))
    )


def _log_z2(spec, st, data):
    e = residuals(st, data, spec.prior, spec.static)[0]
    tm = data.tmask
    le2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, e * e, 1.0)), 0.0)
    return TR.masked_log_z2(le2, st.eq.vol.core.log_lam[0], tm)


def _rows_logq(lq_rows, tm, o):
    """``Σ_t m_t log q(o_t)`` from row-normalised log pmfs ``(T, K)``."""
    picked = jnp.take_along_axis(lq_rows, o[:, None], axis=1)[:, 0]
    return jnp.sum(jnp.where(tm > 0, picked, 0.0))


def _row_normalise(lg):
    return lg - jax.scipy.special.logsumexp(lg, axis=1, keepdims=True)


def _rel_err(lhs, rhs):
    return np.abs(lhs - rhs) / (1.0 + np.abs(rhs))


def _check_pairs(pair, spec, seed):
    """``pair(k, st, data) -> (lhs, rhs)`` (jitted) at 20 joint draws; relative error 1e-8
    (N06 §5.1; DESIGN §9.3 item 2)."""
    fn = jax.jit(pair)
    for r, (st, data) in enumerate(_states(spec, seed=seed)):
        lhs, rhs = (np.asarray(x) for x in fn(key(seed + 1000 + r), st, data))
        assert np.all(_rel_err(lhs, rhs) < 1e-8), (r, lhs, rhs)


def test_density_ratio_o_gaussian_and_p():
    spec = joint_spec("constant_transitory", transitory="outlier")
    kern = TR.OutlierScale(spec.static)
    tp = _eq0(spec.prior.eq).transitory
    lj = _joint_fn(spec)

    def pair(k, st, data):
        ts, tm = _eq0(st.eq.vol.trans), data.tmask
        lz = _log_z2(spec, st, data)
        # o | e, λ, p (Gaussian): Σ_t log q(o_t) against the joint
        lq_rows = _row_normalise(TR.outlier_logits(lz, ts.log_p, ts.log1m_p, K20))
        o_new, _ = kern.o_step(k, lz, tm, ts.log_p, ts.log1m_p)
        ts1 = dataclasses.replace(ts, o_idx=o_new)
        lhs_o = _rows_logq(lq_rows, tm, o_new) - _rows_logq(lq_rows, tm, ts.o_idx)
        rhs_o = lj(_with_trans(st, ts1), data) - lj(_with_trans(st, ts), data)
        # p | o: log Beta(p; a′, b′) (density of p, as the reference prior)
        a, b = TR.p_conditional(ts.o_idx, tm, data.T_eff, tp.a_p, tp.b_p)
        lp1, l1p1 = kern.p_step(jr.fold_in(k, 1), ts.o_idx, tm, data.T_eff, tp)
        ts2 = dataclasses.replace(ts, log_p=lp1, log1m_p=l1p1)
        lhs_p = TR._log_beta_density(lp1, l1p1, a, b) - TR._log_beta_density(
            ts.log_p, ts.log1m_p, a, b
        )
        rhs_p = lj(_with_trans(st, ts2), data) - lj(_with_trans(st, ts), data)
        return jnp.stack([lhs_o, lhs_p]), jnp.stack([rhs_o, rhs_p])

    _check_pairs(pair, spec, seed=70)


def test_density_ratio_tau_given_o_nu():
    spec = joint_spec("constant_transitory", transitory="outlier_t")
    kern = TR.OutlierTScale(spec.static)
    lj = _joint_fn(spec)

    def pair(k, st, data):
        ts, tm = _eq0(st.eq.vol.trans), data.tmask
        lr2 = jnp.where(tm > 0, _log_z2(spec, st, data) - 2 * jnp.log(ts.o_idx + 1.0), 0.0)
        lt1 = kern.tau_step(k, lr2, tm, ts.nu_idx)
        nu, log_c, _ = kern._nu()
        shape, lsc = TR.tau_conditional(lr2, nu[ts.nu_idx], log_c[ts.nu_idx])

        def lq(lt):  # Σ_t m_t log IG(τ_t; shape, scale_t), density of τ
            return jnp.sum(jnp.where(tm > 0, TR._log_ig_density(lt, shape, jnp.exp(lsc)), 0.0))

        ts1 = dataclasses.replace(ts, log_tau=lt1)
        return lq(lt1) - lq(ts.log_tau), lj(_with_trans(st, ts1), data) - lj(
            _with_trans(st, ts), data
        )

    _check_pairs(pair, spec, seed=400)


def _tau_marginal_joint(spec, st_t, st, data, ts):
    """``log p(o, p, ν, λ, e)`` with τ integrated (up to terms free of o, p, ν), from the
    independent references: Beta prior + Σ_t m_t [log w_{o_t} − log o_t + log t_{c,ν}(z_t/o_t)]
    + log π(ν); ``st_t`` is a ``"t"`` StaticConfig with the same grid (for the t density)."""
    tm = data.tmask
    lz = _log_z2(spec, st, data)
    log_o = jnp.log(ts.o_idx + 1.0)
    lw = REF.outlier_log_weights(ts.log_p, ts.log1m_p, K20)[ts.o_idx]
    dens = REF.transitory_log_density(lz - 2 * log_o, TEq(jnp.zeros(1), ts.nu_idx, 0.0), st_t)
    tp = _eq0(spec.prior.eq).transitory
    prior_nu = tp.log_nu_w[ts.nu_idx] - jax.scipy.special.logsumexp(tp.log_nu_w)
    prior_p = REF.log_beta_from_logs(ts.log_p, ts.log1m_p, tp.a_p, tp.b_p)
    return prior_nu + prior_p + jnp.sum(jnp.where(tm > 0, lw - log_o + dens, 0.0))


def test_density_ratio_collapsed_o_and_nu():
    spec = joint_spec("constant_transitory", transitory="outlier_t")
    kern = TR.OutlierTScale(spec.static)
    grid = spec.static.vol.nu_grid
    tp = _eq0(spec.prior.eq).transitory
    st_t = joint_spec("constant_transitory", transitory="t", nu_grid=grid).static

    def pair(k, st, data):
        ts, tm = _eq0(st.eq.vol.trans), data.tmask
        lz = _log_z2(spec, st, data)
        nu, log_c, _ = kern._nu()
        g = ts.nu_idx
        # o | e, λ, ν, p with τ integrated
        lq_rows = _row_normalise(TR.outlier_logits(lz, ts.log_p, ts.log1m_p, K20, nu[g], log_c[g]))
        o_new, _ = kern.o_step(k, lz, tm, ts.log_p, ts.log1m_p, g)
        ts1 = dataclasses.replace(ts, o_idx=o_new)
        lhs_o = _rows_logq(lq_rows, tm, o_new) - _rows_logq(lq_rows, tm, ts.o_idx)
        rhs_o = _tau_marginal_joint(spec, st_t, st, data, ts1) - _tau_marginal_joint(
            spec, st_t, st, data, ts
        )
        # ν | e, λ, o with τ integrated (a different grid point than the current one)
        lr2 = jnp.where(tm > 0, lz - 2 * jnp.log(ts.o_idx + 1.0), 0.0)
        lnu = TR.nu_logits(lr2, tm, data.T_eff, tp.log_nu_w, grid, "standardized")
        g1 = kern.nu_step(jr.fold_in(k, 1), lr2, tm, data.T_eff, tp)
        g1 = jnp.where(g1 == g, (g + 3) % len(grid), g1).astype(jnp.int32)
        ts2 = dataclasses.replace(ts, nu_idx=g1)
        lhs_nu = lnu[g1] - lnu[g]
        rhs_nu = _tau_marginal_joint(spec, st_t, st, data, ts2) - _tau_marginal_joint(
            spec, st_t, st, data, ts
        )
        return jnp.stack([lhs_o, lhs_nu]), jnp.stack([rhs_o, rhs_nu])

    _check_pairs(pair, spec, seed=600)


# --------------------------------------------------------------------------- tier B: block SBC

# Wrong collapsed ``o`` conditionals of ``outlier_t``, through the override seam (DESIGN §4.9):
# τ ignored (the Gaussian logits; the step then conditions on τ ≡ 1, not on τ integrated out) and
# ``c(ν) = ν`` in place of ``ν − 2``. They differ from the exact pmf mainly on large-``|z|`` rows.
COLLAPSED_O_MUTANTS = {
    "tau_ignored": lambda lz, lp, l1p, K, nu=None, lc=None: TR.outlier_logits(lz, lp, l1p, K),
    "c_is_nu": lambda lz, lp, l1p, K, nu=None, lc=None: TR.outlier_logits(
        lz, lp, l1p, K, nu, jnp.log(nu)
    ),
}
# Heavy tails and frequent outliers (ν ∈ {3..6}, p ~ Beta(3, 9)): the regime where integrating τ
# out of the o step matters most.
HEAVY = {"nu_grid": (3, 4, 5, 6), "prior_kwargs": {"a_p": 3.0, "b_p": 9.0}}
O_STAT_NAMES = ("sum_log_o_x_log1p_z2", "n1_big_z", "sum_log_o_big_z")


def _stat_names(transitory: str, block: str) -> tuple[str, ...]:
    names = ["loglik"]
    if transitory in ("outlier", "outlier_t"):
        names += ["sum_log_o", "log_p"]
    if transitory in ("t", "outlier_t"):
        names += ["sum_log_tau", "nu"]
    return (*names, *(O_STAT_NAMES if block in ("o", "o_tau") else ()))


def _block_case(transitory: str, block: str, *, heavy: bool = False, overrides=None):
    spec = joint_spec("constant_transitory", transitory=transitory, **(HEAVY if heavy else {}))
    kern = registry.load(transitory)(spec.static)
    if overrides:
        kern = kern.with_overrides(overrides)
    pr = _eq0(spec.prior.eq)
    st_ = spec.static
    redraws_o = block in ("o", "o_tau")

    def joint_sample(k):
        st = spec.prior_sample(jr.fold_in(k, 0))
        return st, spec.simulate(jr.fold_in(k, 1), st)

    def draws(k, st, data):
        ts = _eq0(st.eq.vol.trans)
        tm = data.tmask
        lz = _log_z2(spec, st, data)
        if block == "o":  # o | e, λ, p (Gaussian)
            o, _ = kern.o_step(k, lz, tm, ts.log_p, ts.log1m_p)
            ts = dataclasses.replace(ts, o_idx=o)
        elif block == "p":
            lp, l1p = kern.p_step(k, ts.o_idx, tm, data.T_eff, pr.transitory)
            ts = dataclasses.replace(ts, log_p=lp, log1m_p=l1p)
        elif block == "nu_tau":  # (ν, τ) | e, λ, o: collapsed ν then τ
            lr2 = (
                jnp.where(tm > 0, lz - 2 * jnp.log(ts.o_idx + 1.0), 0.0)
                if hasattr(ts, "o_idx")
                else lz
            )
            g = kern.nu_step(k, lr2, tm, data.T_eff, pr.transitory)
            ts = dataclasses.replace(ts, nu_idx=g, log_tau=kern.tau_step(k, lr2, tm, g))
        elif block == "o_tau":  # (o, τ) | e, λ, ν, p: collapsed o then τ
            o, _ = kern.o_step(k, lz, tm, ts.log_p, ts.log1m_p, ts.nu_idx)
            lr2 = jnp.where(tm > 0, lz - 2 * jnp.log(o + 1.0), 0.0)
            ts = dataclasses.replace(ts, o_idx=o, log_tau=kern.tau_step(k, lr2, tm, ts.nu_idx))
        return _with_trans(st, ts)

    def stat(st, data):
        """Block log-likelihood and the redrawn quantities. For the blocks that redraw ``o``, also
        statistics on the rows where the ``o`` conditional is informative: ``z² = e²/λ`` is fixed
        within the block, so ``Σ log o · log1p(z²)``, ``N₁`` and ``Σ log o`` over ``z² > 4``
        compare the drawn ``o`` with the true one row by row (the sums over all rows had no
        power against ``COLLAPSED_O_MUTANTS`` at N = 200)."""
        tm = data.tmask
        ts = _eq0(st.eq.vol.trans)
        lv = st.eq.vol.core.log_lam[0] + kern.log_scale2(ts)
        e = residuals(st, data, spec.prior, st_)[0]
        ll = jnp.sum(jnp.where(tm > 0, -0.5 * (lv + e * e * jnp.exp(-lv)), 0.0))
        out = [ll]
        if hasattr(ts, "o_idx"):
            out += [jnp.sum(jnp.where(tm > 0, jnp.log(ts.o_idx + 1.0), 0.0)), ts.log_p]
        if hasattr(ts, "log_tau"):
            out += [jnp.sum(jnp.where(tm > 0, ts.log_tau, 0.0)), kern.nu_value(ts.nu_idx)]
        if redraws_o:
            lz = _log_z2(spec, st, data)
            log_o = jnp.where(tm > 0, jnp.log(ts.o_idx + 1.0), 0.0)
            big = (tm > 0) & (lz > math.log(4.0))
            out += [
                jnp.sum(log_o * jnp.where(tm > 0, jnp.logaddexp(0.0, lz), 0.0)),
                jnp.sum(jnp.where(big, (ts.o_idx > 0).astype(jnp.float64), 0.0)),
                jnp.sum(jnp.where(big, log_o, 0.0)),
            ]
        return jnp.stack(out).astype(jnp.float64)

    return joint_sample, draws, stat, _stat_names(transitory, block)


def _sbc_N(block: str) -> int:
    """Tier B uses N = 200 (DESIGN §9.4); the blocks that redraw ``o`` use N = 1000 (cheap: the
    conditional is per-row categorical) for power against a wrong ``o`` conditional."""
    return 1000 if block in ("o", "o_tau") else 200


@pytest.mark.parametrize(
    ("transitory", "block", "heavy"),
    [("outlier", "o", False), ("outlier", "p", False), ("t", "nu_tau", False),
     ("outlier_t", "nu_tau", False), ("outlier_t", "o_tau", False), ("outlier_t", "o_tau", True)],
)  # fmt: skip
def test_block_conditional_sbc(transitory, block, heavy):
    joint_sample, draws, stat, names = _block_case(transitory, block, heavy=heavy)
    res = block_sbc(
        joint_sample, draws, stat, N=_sbc_N(block), L=99, seed=13, alpha=1e-3, names=names
    )
    assert res.passed, res.table()


@pytest.mark.parametrize("heavy", [False, True])
@pytest.mark.parametrize("mutant", sorted(COLLAPSED_O_MUTANTS))
def test_block_sbc_detects_a_wrong_collapsed_o_conditional(mutant, heavy):
    """Power of the ``(o, τ)`` block SBC of ``outlier_t``: both mutants fail tier B (same seed and
    sizes as the passing case above)."""
    joint_sample, draws, stat, names = _block_case(
        "outlier_t", "o_tau", heavy=heavy, overrides={"outlier_logits": COLLAPSED_O_MUTANTS[mutant]}
    )
    res = block_sbc(
        joint_sample, draws, stat, N=_sbc_N("o_tau"), L=99, seed=13, alpha=1e-3, names=names
    )
    assert not res.passed, res.table()
