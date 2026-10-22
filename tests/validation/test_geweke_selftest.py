"""Geweke joint-distribution test on a conjugate scalar-normal toy (N06 §3; DESIGN §9.5, §13 WP7).

Toy: θ ~ N(μ₀, τ²), y_t | θ ~ N(θ, s²), t = 1..T. The exact Gibbs step draws θ | y ~ N(m, v);
with τ = s = 1 and T = 4, r = Var_prior/Var_post = 5, so the SC chain of θ is AR(1) with
κ = 1 − 1/r = 0.8 and RNE ≈ 1/(2r − 1) (N06 §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from rai_bvar.engines.mcmc.keys import Tier, validation_key
from rai_bvar.validation.geweke import (
    GewekeConfig,
    GewekeResult,
    JointModel,
    Part,
    composition_id,
    geweke,
    geweke_sequential,
    rep_keys,
)
from rai_bvar.validation.sequential import gandy_scott, stage_multiplier, stage_seed

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

NAMES = ("theta", "theta2", "ybar", "ybar2", "theta_ybar", "loglik")


@dataclass(frozen=True)
class NormalToy:
    mu0: float = 0.5
    tau: float = 1.0
    s: float = 1.0
    T: int = 4
    var_mult: float = 1.0  # deliberately wrong posterior variance when ≠ 1
    sampler_mu0: float | None = None  # deliberately wrong prior mean inside the sampler
    nan_above: float | None = None  # ybar is NaN when θ exceeds this (a "failure" signal)
    name: str = "normal_toy"
    names: tuple[str, ...] = NAMES

    def prior_sample(self, key):
        return self.mu0 + self.tau * random.normal(key, (), dtype=jnp.float64)

    def simulate(self, key, th):
        return th + self.s * random.normal(key, (self.T,), dtype=jnp.float64)

    def step(self, key, it, th, y):
        mu0 = self.mu0 if self.sampler_mu0 is None else self.sampler_mu0
        v = 1.0 / (1.0 / self.tau**2 + self.T / self.s**2)
        m = v * (mu0 / self.tau**2 + jnp.sum(y) / self.s**2)
        return m + jnp.sqrt(v * self.var_mult) * random.normal(
            random.fold_in(key, it), (), jnp.float64
        )

    def test_functions(self, th, y):
        yb = jnp.mean(y)
        if self.nan_above is not None:
            yb = jnp.where(th > self.nan_above, jnp.nan, yb)
        ll = -0.5 * self.T * jnp.log(2 * jnp.pi * self.s**2) - jnp.sum((y - th) ** 2) / (
            2 * self.s**2
        )
        return jnp.stack([th, th**2, yb, yb**2, th * yb, ll])


SMALL = GewekeConfig(C=400, M_c=40, M1=20_000, alpha=0.05, seed=0)


def test_toy_implements_the_protocol():
    assert isinstance(NormalToy(), JointModel)


def test_correct_toy_holds_the_nominal_level_over_20_seeds():
    results = [geweke(NormalToy(), replace(SMALL, seed=seed)) for seed in range(20)]
    failures = sum(not r.passed for r in results)
    # Bonferroni at α = 0.05 per run: P(Bin(20, 0.05) ≥ 4) = 0.016, and Bonferroni is conservative
    assert failures <= 3
    # per function, the 20 z values are iid t_{C−1} under H0 (seeds give independent runs)
    lo, hi = stats.chi2(20).ppf([1e-3, 1 - 1e-3])
    for j in (NAMES.index("theta"), NAMES.index("loglik")):
        z = np.array([r.z[j] for r in results])
        assert stats.kstest(z, stats.t(SMALL.C - 1).cdf).pvalue > 1e-3
        assert lo < np.sum(z**2) < hi
    assert all(r.t_crit == pytest.approx(stats.t.isf(0.05 / 12, SMALL.C - 1)) for r in results)


@pytest.mark.parametrize(
    ("toy", "expected"),
    [
        (NormalToy(var_mult=1.5), {"theta2", "loglik"}),  # wrong conditional variance
        (NormalToy(sampler_mu0=1.0), {"theta"}),  # prior mismatch (Geweke's error 1 class)
    ],
)
def test_deliberately_wrong_toys_fail(toy, expected):
    res = geweke(toy, GewekeConfig(C=1000, M_c=50, M1=20_000, alpha=1e-3, seed=7))
    assert not res.passed
    assert expected <= set(res.failing)
    assert np.max(np.abs(res.z)) > 2 * res.t_crit
    assert res.p_bonferroni <= 1e-3


def test_rne_matches_the_closed_form():
    # AR(1) with κ = 0.8: Var(chain mean of M) = σ²/M·[(1+κ)/(1−κ) − 2κ(1−κ^M)/(M(1−κ)²)]
    M, kappa = 200, 0.8
    long_run = (1 + kappa) / (1 - kappa) - 2 * kappa * (1 - kappa**M) / (M * (1 - kappa) ** 2)
    res = geweke(NormalToy(), GewekeConfig(C=2000, M_c=M, M1=50_000, alpha=1e-3, seed=3))
    assert res.passed
    assert res.rne[0] == pytest.approx(1 / long_run, rel=0.12)
    # batch means of size b = ⌊√M_c⌋ = 14: b·Var(batch mean) = σ²·[the same bracket at M = b]
    b = int(np.floor(np.sqrt(M)))
    batch = (1 + kappa) / (1 - kappa) - 2 * kappa * (1 - kappa**b) / (b * (1 - kappa) ** 2)
    assert res.rne_batch[0] == pytest.approx(1 / batch, rel=0.1)
    assert np.all(np.isfinite(res.z_batch)) and np.all((res.p_batch > 0) & (res.p_batch <= 1))


def test_statistics_follow_the_documented_keys_and_formulas():
    toy = NormalToy()
    cfg = GewekeConfig(C=3, M_c=4, M1=5, alpha=0.05, seed=11, batch_means=False)
    res = geweke(toy, cfg)
    comp = composition_id(toy)

    # MC: iid draws keyed by validation_key(seed, Tier.G, comp, m) → Part.MC_PRIOR / MC_SIM
    g_mc = []
    for m in range(cfg.M1):
        k = validation_key(cfg.seed, int(Tier.G), comp, m)
        th = toy.prior_sample(random.fold_in(k, int(Part.MC_PRIOR)))
        g_mc.append(toy.test_functions(th, toy.simulate(random.fold_in(k, int(Part.MC_SIM)), th)))
    g_mc = np.asarray(g_mc)
    np.testing.assert_allclose(res.mean_mc, g_mc.mean(axis=0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(res.var_mc, g_mc.var(axis=0, ddof=1), rtol=1e-10)

    # SC: chain c from Part.PRIOR, simulator fold(Part.SIM, it), step(fold(Part.STEP), it)
    means = []
    for c in range(cfg.C):
        k = validation_key(cfg.seed, int(Tier.G), comp, c)
        th = toy.prior_sample(random.fold_in(k, int(Part.PRIOR)))
        gs = []
        for it in range(cfg.M_c):
            u = jnp.uint32(it)
            y = toy.simulate(random.fold_in(random.fold_in(k, int(Part.SIM)), u), th)
            th = toy.step(random.fold_in(k, int(Part.STEP)), u, th, y)
            gs.append(toy.test_functions(th, y))
        means.append(np.mean(np.asarray(gs), axis=0))
    means = np.asarray(means)
    np.testing.assert_allclose(res.mean_sc, means.mean(axis=0), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(res.var_chain_means, means.var(axis=0, ddof=1), rtol=1e-8)
    z = (g_mc.mean(0) - means.mean(0)) / np.sqrt(g_mc.var(0, ddof=1) / 5 + means.var(0, ddof=1) / 3)
    np.testing.assert_allclose(res.z, z, rtol=1e-8)
    np.testing.assert_allclose(res.p, 2 * stats.t.sf(np.abs(z), 2), rtol=1e-8)
    assert res.z_batch is None and res.rne_batch is None


def test_batch_means_follow_the_documented_formulas():
    # M_c = 7: b = ⌊√7⌋ = 2 and 3 batches over the LAST 6 sweeps (sweeps 1–2, 3–4, 5–6; sweep 0
    # is left out), pooled over the C chains
    toy = NormalToy()
    cfg = GewekeConfig(C=3, M_c=7, M1=5, alpha=0.05, seed=11)
    res = geweke(toy, cfg)
    comp = composition_id(toy)
    g_mc = []
    for m in range(cfg.M1):
        k = validation_key(cfg.seed, int(Tier.G), comp, m)
        th = toy.prior_sample(random.fold_in(k, int(Part.MC_PRIOR)))
        g_mc.append(toy.test_functions(th, toy.simulate(random.fold_in(k, int(Part.MC_SIM)), th)))
    g_mc = np.asarray(g_mc)
    chains = []
    for c in range(cfg.C):
        k = validation_key(cfg.seed, int(Tier.G), comp, c)
        th = toy.prior_sample(random.fold_in(k, int(Part.PRIOR)))
        gs = []
        for it in range(cfg.M_c):
            u = jnp.uint32(it)
            y = toy.simulate(random.fold_in(random.fold_in(k, int(Part.SIM)), u), th)
            th = toy.step(random.fold_in(k, int(Part.STEP)), u, th, y)
            gs.append(toy.test_functions(th, y))
        chains.append(np.asarray(gs))
    chains = np.asarray(chains)  # (C, M_c, G)
    b = 2
    batch_means = chains[:, 1:].reshape(cfg.C, 3, b, -1).mean(axis=2).reshape(cfg.C * 3, -1)
    tau2 = b * batch_means.var(axis=0, ddof=1)
    diff = g_mc.mean(0) - chains.mean(axis=(0, 1))
    z_b = diff / np.sqrt(g_mc.var(0, ddof=1) / cfg.M1 + tau2 / (cfg.C * cfg.M_c))
    np.testing.assert_allclose(res.z_batch, z_b, rtol=1e-7)
    np.testing.assert_allclose(res.p_batch, 2 * stats.norm.sf(np.abs(z_b)), rtol=1e-7)
    np.testing.assert_allclose(res.rne_batch, g_mc.var(0, ddof=1) / tau2, rtol=1e-7)
    np.testing.assert_allclose(res.z, diff / np.sqrt(res.var_mc / 5 + res.var_chain_means / 3))


def test_results_do_not_depend_on_block_sizes():
    a = geweke(NormalToy(), SMALL)
    b = geweke(NormalToy(), replace(SMALL, chain_block=7, draw_block=333))
    # chains and draws are keyed by their index, so blocks change only floating-point reductions
    np.testing.assert_allclose(a.mean_sc, b.mean_sc, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(a.mean_mc, b.mean_mc, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(a.z, b.z, rtol=1e-9)
    c = geweke(NormalToy(), SMALL)
    np.testing.assert_array_equal(a.z, c.z)
    d = geweke(NormalToy(), replace(SMALL, seed=1))
    assert not np.allclose(a.z, d.z)
    e = geweke(NormalToy(name="other"), SMALL)  # composition id comes from the name
    assert not np.allclose(a.z, e.z)


def test_non_finite_test_functions_fail_with_counts():
    res = geweke(NormalToy(nan_above=2.2), SMALL)
    nan_g = [NAMES.index(n) for n in ("ybar", "ybar2", "theta_ybar")]
    assert not res.passed
    assert set(res.failing) >= {"ybar", "ybar2", "theta_ybar"}
    assert np.all(res.n_nonfinite[nan_g] > 0) and res.n_nonfinite[0] == 0
    assert np.all(res.p[nan_g] == 0.0)


@dataclass(frozen=True)
class SamplerFailure(NormalToy):
    """The sampler (not the model) breaks: NaN once θ > 2.8 after sweep 20. Only the SC chains run
    the sampler, so the MC simulator stays finite and only the SC count can see it."""

    name: str = "sampler_failure"

    def step(self, key, it, th, y):
        out = super().step(key, it, th, y)
        return jnp.where((it >= 20) & (out > 2.8), jnp.nan, out)


def test_sampler_failures_in_the_sc_chains_fail_with_counts():
    res = geweke(SamplerFailure(), SMALL)
    ok = geweke(NormalToy(name="sampler_failure"), SMALL)  # same keys, no failure
    np.testing.assert_array_equal(res.mean_mc, ok.mean_mc)  # MC is untouched
    assert np.all(res.n_nonfinite > 0)  # a NaN state breaks every test function
    assert np.all(res.p == 0.0) and set(res.failing) == set(NAMES) and not res.passed
    # the NaN sweeps are excluded from the chain sums, so z alone would not have caught it
    assert np.all(np.isfinite(res.z)) and np.max(np.abs(res.z)) < res.t_crit
    # n_nonfinite(θ) is exactly the number of NaN states over the C chains (rerun directly)
    toy = SamplerFailure()
    keys = rep_keys(SMALL.seed, Tier.G, composition_id(toy), np.arange(SMALL.C))
    th = jax.vmap(lambda k: toy.prior_sample(random.fold_in(k, int(Part.PRIOR))))(keys)
    sim = jax.vmap(lambda k: random.fold_in(k, int(Part.SIM)))(keys)
    stp = jax.vmap(lambda k: random.fold_in(k, int(Part.STEP)))(keys)
    nan_states, chains = 0, np.zeros(SMALL.C, dtype=bool)
    for it in range(SMALL.M_c):
        u = jnp.full(SMALL.C, it, dtype=jnp.uint32)
        y = jax.vmap(lambda k, i, t: toy.simulate(random.fold_in(k, i), t))(sim, u, th)
        th = jax.vmap(toy.step)(stp, u, th, y)
        nan_states += int(np.sum(~np.isfinite(th)))
        chains |= ~np.isfinite(np.asarray(th))
    assert res.n_nonfinite[NAMES.index("theta")] == nan_states
    assert 0 < chains.sum() < 0.2 * SMALL.C  # rare chains


@dataclass(frozen=True)
class ClippedSampler(NormalToy):
    """The sampler never returns θ > 2.4, and ``ybar`` is NaN for θ > 2.5 (``nan_above``). The SC
    chains evaluate the test functions only after a step, so only the MC simulator, which draws
    θ from the prior, can produce the NaN."""

    nan_above: float | None = 2.5
    name: str = "clipped_sampler"

    def step(self, key, it, th, y):
        return jnp.minimum(super().step(key, it, th, y), 2.4)


def test_non_finite_mc_draws_fail_with_counts():
    toy = ClippedSampler()
    res = geweke(toy, SMALL)
    nan_g = [NAMES.index(n) for n in ("ybar", "ybar2", "theta_ybar")]
    # n_nonfinite is exactly the number of MC prior draws above 2.5 (rerun from the MC keys)
    keys = rep_keys(SMALL.seed, Tier.G, composition_id(toy), np.arange(SMALL.M1))
    th = jax.vmap(lambda k: toy.prior_sample(random.fold_in(k, int(Part.MC_PRIOR))))(keys)
    n_mc = int(np.sum(np.asarray(th) > 2.5))
    assert 0 < n_mc < 0.05 * SMALL.M1  # P(θ > 2.5) = P(Z > 2) ≈ 2.3%
    np.testing.assert_array_equal(res.n_nonfinite[nan_g], n_mc)
    other = [j for j in range(len(NAMES)) if j not in nan_g]
    assert np.all(res.n_nonfinite[other] == 0)
    assert np.all(res.p[nan_g] == 0.0) and set(res.failing) >= {"ybar", "ybar2", "theta_ybar"}
    assert not res.passed and res.p_bonferroni == 0.0
    # the NaN draws are excluded from the MC moments, so the means alone stay finite
    assert np.all(np.isfinite(res.mean_mc)) and np.all(np.isfinite(res.z[nan_g]))


def test_constant_test_function_carries_no_evidence():
    @dataclass(frozen=True)
    class WithConstant(NormalToy):
        names: tuple[str, ...] = (*NAMES, "one")

        def test_functions(self, th, y):
            return jnp.concatenate([super().test_functions(th, y), jnp.ones(1)])

    res = geweke(WithConstant(), SMALL)
    assert res.z[-1] == 0.0 and res.p[-1] == 1.0 and "one" not in res.failing


def test_table_and_result_type():
    res = geweke(NormalToy(var_mult=1.5), SMALL)
    assert isinstance(res, GewekeResult)
    table = res.table()
    assert list(table.index[:1]) == [NAMES[int(np.argmax(np.abs(res.z)))]]
    assert table["failing"].sum() == len(res.failing)


def test_gandy_scott_around_geweke():
    def runner(toy, base):
        def run(stage):
            m = stage_multiplier(stage)
            cfg = replace(base, seed=stage_seed(base.seed, stage), C=base.C * m, M1=base.M1 * m)
            return geweke(toy, cfg).p_bonferroni

        return run

    base = GewekeConfig(C=300, M_c=30, M1=10_000, alpha=1e-3, seed=5)
    assert gandy_scott(runner(NormalToy(), base), 1e-3).passed
    res = gandy_scott(runner(NormalToy(var_mult=1.5), base), 1e-3)
    assert not res.passed and res.stage == 1


def test_geweke_sequential_reruns_fresh_and_larger():
    base = GewekeConfig(C=300, M_c=30, M1=10_000, alpha=1e-3, seed=5)
    seq, runs = geweke_sequential(NormalToy(), base)
    assert seq.passed and len(runs) == seq.stage
    seq, runs = geweke_sequential(NormalToy(var_mult=1.5), base)
    assert not seq.passed and seq.stage == 1
    # α = 0.9, k = 2: β₁ = 0.45 and γ + β₁ > 1, so any q₁ > 0.45 is marginal and reruns with
    # fresh keys at Δ× the size (M_c unchanged)
    seq, runs = geweke_sequential(NormalToy(), replace(base, alpha=0.9), k=2, delta=3)
    assert seq.stage == 2 and seq.multipliers == (1, 3)
    assert (runs[1].C, runs[1].M1, runs[1].M_c) == (900, 30_000, 30)
    assert not np.allclose(runs[0].z, runs[1].z)


def test_validation_errors():
    with pytest.raises(ValueError, match="names"):
        geweke(NormalToy(names=("a", "b")), SMALL)

    @dataclass(frozen=True)
    class Float32Toy(NormalToy):
        def test_functions(self, th, y):
            return super().test_functions(th, y).astype(jnp.float32)

    with pytest.raises(TypeError, match="float64"):
        geweke(Float32Toy(), SMALL)
    with pytest.raises(ValueError, match="C must"):
        GewekeConfig(C=1, M_c=1, M1=10, alpha=0.1, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        GewekeConfig(C=10, M_c=1, M1=10, alpha=0.0, seed=0)
    with pytest.raises(TypeError, match="seed"):
        GewekeConfig(C=10, M_c=1, M1=10, alpha=0.1, seed=1.5)


def test_rep_keys_are_validation_keys():
    keys = rep_keys(4, Tier.S, 17, np.array([0, 5, 2**32 - 1]))
    for k, r in zip(keys, (0, 5, 2**32 - 1), strict=True):
        np.testing.assert_array_equal(
            random.key_data(k), random.key_data(validation_key(4, int(Tier.S), 17, r))
        )
    with pytest.raises(ValueError, match="2\\*\\*32"):
        rep_keys(4, Tier.S, 17, [2**32])
    assert "threefry2x32" in str(jax.random.key_impl(keys))
