"""SBC and block-conditional SBC on conjugate normal toys (N06 §4, §5.2; DESIGN §9.4, §9.6).

Toy: θ ~ N(μ₀, τ²), y_t | θ ~ N(θ, s²), t = 1..T. The sampler is the exact AR(1) kernel
θ′ = m + ρ(θ − m) + √(v(1 − ρ²)) z, which leaves the posterior N(m, v) invariant and has
IAT = (1 + ρ)/(1 − ρ), so the per-replication thinning is exercised (ρ = 0.8: IAT = 9).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy import stats as jstats
from scipy import stats

from rai_bvar.engines.mcmc.keys import Tier, validation_key
from rai_bvar.validation.geweke import Part, composition_id
from rai_bvar.validation.sbc import (
    KEPT_LEVELS,
    SBCConfig,
    SBCResult,
    _acov,
    _geyer_tau,
    _indicators,
    _pilot_iat,
    _pooled_ess,
    _thinned_ess,
    block_sbc,
    iat,
    sbc,
    sbc_sequential,
)
from rai_bvar.validation.uniformity import rank_with_ties

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


@dataclass(frozen=True)
class ARToy:
    mu0: float = 0.5
    tau: float = 1.0
    s: float = 1.0
    T: int = 4
    rho: float = 0.8
    var_mult: float = 1.0
    sampler_mu0: float | None = None
    name: str = "ar_toy"
    names: tuple[str, ...] = ("theta",)

    def prior_sample(self, key):
        return self.mu0 + self.tau * random.normal(key, (), dtype=jnp.float64)

    def simulate(self, key, th):
        return th + self.s * random.normal(key, (self.T,), dtype=jnp.float64)

    def step(self, key, it, th, y):
        mu0 = self.mu0 if self.sampler_mu0 is None else self.sampler_mu0
        v = 1.0 / (1.0 / self.tau**2 + self.T / self.s**2)
        m = v * (mu0 / self.tau**2 + jnp.sum(y) / self.s**2)
        z = random.normal(random.fold_in(key, it), (), dtype=jnp.float64)
        return m + self.rho * (th - m) + jnp.sqrt(v * self.var_mult * (1 - self.rho**2)) * z

    def test_functions(self, th, y):
        return jnp.stack([th])


NAMES = ("theta", "loglik", "theta_positive")


def quantities(th, y):
    """Parameter, log-likelihood (data-dependent; Modrák et al.) and a discrete quantity (ties)."""
    loglik = jnp.sum(jstats.norm.logpdf(y, th, 1.0))
    return jnp.stack([th, loglik, (th > 0.5).astype(jnp.float64)])


def nan_quantities(th, y):
    return jnp.where(th > 2.0, jnp.nan, quantities(th, y))


CFG = SBCConfig(N=100, L=99, burn=30, pilot=200, t_max=50, alpha=0.05, seed=0)


def test_correct_toy_holds_the_nominal_level_over_20_seeds():
    results = [sbc(ARToy(), quantities, NAMES, replace(CFG, seed=seed)) for seed in range(20)]
    assert all(isinstance(r, SBCResult) and r.ranks.shape == (100, 3) for r in results)
    # Bonferroni over 3 quantities at α = 0.05 per run: at most 3 of 20 runs may fail
    assert sum(not r.passed for r in results) <= 3
    p = np.array([r.p for r in results])  # (20, 3); exact γ p-values are super-uniform
    assert np.all(np.sum(p <= 0.05, axis=0) <= 4)  # P(Bin(20, 0.05) ≥ 5) = 0.003
    assert all(r.ess_ok == 1.0 and not r.failed.any() for r in results)
    # the pooled kept-draw ESS is precise: well clear of 0.8·L for well-thinned draws
    assert all(np.all(r.ess_pooled[:2] > 0.85 * CFG.L) for r in results)
    assert not any(reason.startswith("kept draws") for r in results for reason in r.reasons)
    ranks = np.concatenate([r.ranks for r in results])  # 2000 ranks per quantity
    for q in range(3):
        counts = np.bincount(ranks[:, q] // 10, minlength=10)
        assert stats.chisquare(counts).pvalue > 1e-3


@pytest.mark.parametrize(
    ("toy", "quantity"),
    [
        (ARToy(var_mult=0.3), "theta"),  # under-dispersed sampler (∪-shaped ranks)
        (ARToy(sampler_mu0=2.0), "theta"),  # biased sampler: wrong prior mean inside
    ],
)
def test_deliberately_wrong_toys_fail(toy, quantity):
    res = sbc(toy, quantities, NAMES, replace(CFG, N=200, alpha=1e-2, seed=4))
    assert not res.passed
    assert quantity in res.failing
    assert res.p_bonferroni <= 1e-2
    assert any("γ test" in reason for reason in res.reasons)


def test_sbc_sequential():
    seq, runs = sbc_sequential(ARToy(), quantities, NAMES, replace(CFG, N=60, alpha=1e-2))
    assert seq.passed and len(runs) == seq.stage
    seq, runs = sbc_sequential(
        ARToy(var_mult=0.3), quantities, NAMES, replace(CFG, N=200, alpha=1e-2)
    )
    assert not seq.passed and seq.stage == 1
    seq, _ = sbc_sequential(ARToy(), nan_quantities, NAMES, replace(CFG, N=60, alpha=1e-2))
    assert not seq.passed and seq.q == (0.0,)  # failed replications fail at once


def test_thinning_follows_the_pilot_iat():
    iid = sbc(ARToy(rho=0.0), quantities, NAMES, replace(CFG, N=50))
    assert np.median(iid.thin) <= 4  # IAT ≈ 1 (tail indicators from 200 pilot draws are noisy)
    ar = sbc(ARToy(rho=0.8), quantities, NAMES, replace(CFG, N=50, pilot=1000))
    assert 7.0 < np.median(ar.iat) < 13.0  # IAT = 9, inflated slightly by the tail indicators
    np.testing.assert_array_equal(ar.thin, np.minimum(np.ceil(2 * ar.iat), 50).astype(int))
    assert np.all(ar.ess_thin > 0.9 * 99)


def test_capped_thinning_fails_the_ess_rule():
    res = sbc(ARToy(rho=0.99), quantities, NAMES, replace(CFG, N=40, pilot=400, t_max=8))
    assert np.all(res.thin <= 8)
    assert res.ess_ok < 0.99 and not res.passed
    assert any("thinned ESS" in reason for reason in res.reasons)


def test_failed_replications_fail_the_test():
    res = sbc(ARToy(), nan_quantities, NAMES, replace(CFG, N=60))
    assert res.failed.any() and not res.passed
    assert res.ranks.shape[0] == 60  # failed replications are kept and reported, never dropped
    assert any("failed" in reason for reason in res.reasons)


def truth_and_data(toy, cfg, i):
    k = validation_key(cfg.seed, int(Tier.S), composition_id(toy), i)
    th = toy.prior_sample(random.fold_in(k, int(Part.PRIOR)))
    return k, th, toy.simulate(random.fold_in(k, int(Part.SIM)), th)


@dataclass(frozen=True)
class LateFailure(ARToy):
    """The sampler breaks (NaN) only after the pilot's sweep range, and only for datasets with
    ``mean(y) > 2``: the pilot cannot see it, only the production chain."""

    fail_at: int = 0
    name: str = "late_failure"

    def step(self, key, it, th, y):
        out = super().step(key, it, th, y)
        return jnp.where((it >= self.fail_at) & (jnp.mean(y) > 2.0), jnp.nan, out)


def test_late_sampler_failures_fail_the_test():
    cfg = replace(CFG, N=100)
    toy = LateFailure(fail_at=cfg.burn + cfg.pilot + 10)
    res = sbc(toy, quantities, NAMES, cfg)
    expected = [float(jnp.mean(truth_and_data(toy, cfg, int(i))[2])) > 2.0 for i in res.kept]
    assert 0 < sum(expected) < res.kept.size
    np.testing.assert_array_equal(res.failed, expected)  # exactly the broken datasets
    assert np.all(res.thin[res.failed] < cfg.t_max)  # their pilots were finite
    assert not res.passed and any("failed" in reason for reason in res.reasons)
    assert res.ranks.shape[0] == res.kept.size  # reported, never dropped


@dataclass(frozen=True)
class ClippedToy(ARToy):
    """The chain never exceeds 2.4, so only the truth θ̃ can reach the NaN region of
    :func:`truth_nan_quantities`."""

    name: str = "clipped"

    def step(self, key, it, th, y):
        return jnp.minimum(super().step(key, it, th, y), 2.4)


def truth_nan_quantities(th, y):
    return jnp.where(th >= 2.5, jnp.nan, quantities(th, y))


def test_non_finite_truth_alone_fails_the_test():
    cfg = replace(CFG, N=200)
    res = sbc(ClippedToy(), truth_nan_quantities, NAMES, cfg)
    expected = [float(truth_and_data(ClippedToy(), cfg, int(i))[1]) >= 2.5 for i in res.kept]
    assert sum(expected) > 0
    np.testing.assert_array_equal(res.failed, expected)
    assert np.all(res.thin[res.failed] < cfg.t_max)  # the pilots never saw a NaN
    assert not res.passed and any("failed" in reason for reason in res.reasons)


def test_pooled_ess_catches_a_pilot_that_underestimates_the_iat():
    # ρ = 0.995 (IAT = 399): 200 pilot draws see a fraction of the IAT and Geyer truncates early,
    # so t_i = ⌈2·IAT_i⌉ ≈ 95 is uncapped and the per-replication rule is blind (ess_ok = 1)
    res = sbc(ARToy(rho=0.995), quantities, NAMES, replace(CFG, N=60, pilot=200, t_max=2000))
    assert res.ess_ok == 1.0 and np.all(res.thin < 2000)
    phi = 0.995**res.thin  # the true lag-1 autocorrelation of the kept draws
    assert np.median((1 - phi) / (1 + phi)) < 0.5  # true ESS < L/2
    assert np.all(res.ess_pooled[:2] < 0.8 * CFG.L)
    assert not res.passed
    assert any(r.startswith("kept draws: pooled ESS") and "theta" in r for r in res.reasons)
    seq, _ = sbc_sequential(
        ARToy(rho=0.995), quantities, NAMES, replace(CFG, N=60, pilot=200, t_max=2000)
    )
    assert not seq.passed and seq.q == (0.0,)  # an ESS failure is not a marginal γ result


def test_data_only_filter():
    cfg = replace(CFG, N=80, max_abs_data=1.6)
    res = sbc(ARToy(), quantities, NAMES, cfg)
    expected = [i for i in range(80) if np.max(np.abs(truth_and_data(ARToy(), cfg, i)[2])) < 1.6]
    np.testing.assert_array_equal(res.kept, expected)
    assert 0 < res.kept.size < 80 and res.n_total == 80
    assert res.ranks.shape == (res.kept.size, 3)
    assert res.passed  # selecting on ỹ alone keeps ranks uniform


def test_ranks_follow_the_documented_keys():
    toy, cfg = ARToy(), replace(CFG, N=6, burn=3, L=5, pilot=20)
    res = sbc(toy, quantities, NAMES, cfg)
    for j, i in enumerate(res.kept):
        k, th_true, y = truth_and_data(toy, cfg, int(i))
        # pilot: independent prior start, burn + pilot sweeps, sweep counters from 0
        pk = random.fold_in(k, int(Part.PILOT_STEP))
        th = toy.prior_sample(random.fold_in(k, int(Part.PILOT_INIT)))
        draws = []
        for it in range(cfg.burn + cfg.pilot):
            th = toy.step(pk, jnp.uint32(it), th, y)
            if it >= cfg.burn:
                draws.append(quantities(th, y))
        draws = np.asarray(draws)
        q05, q95 = np.quantile(draws, [0.05, 0.95], axis=0)
        seqs = np.concatenate([draws.T, (draws <= q05).T, (draws <= q95).T]).astype(float)
        tau = iat(seqs).max()
        assert res.iat[j] == pytest.approx(tau, rel=1e-10)
        t = int(min(np.ceil(2 * tau), cfg.t_max))
        assert res.thin[j] == t
        # production: fresh start and key, keep every t-th sweep after burn-in
        ck = random.fold_in(k, int(Part.CHAIN_STEP))
        th = toy.prior_sample(random.fold_in(k, int(Part.CHAIN_INIT)))
        for it in range(cfg.burn):
            th = toy.step(ck, jnp.uint32(it), th, y)
        kept = []
        for d in range(cfg.L):
            for r in range(t):
                th = toy.step(ck, jnp.uint32(cfg.burn + d * t + r), th, y)
            kept.append(quantities(th, y))
        u = random.uniform(random.fold_in(k, int(Part.TIES)), (3,), dtype=jnp.float64)
        expected = rank_with_ties(quantities(th_true, y), jnp.stack(kept), u)
        np.testing.assert_array_equal(res.ranks[j], np.asarray(expected))


def test_reproducible_and_block_invariant():
    a = sbc(ARToy(), quantities, NAMES, replace(CFG, N=50))
    b = sbc(ARToy(), quantities, NAMES, replace(CFG, N=50, rep_block=17))
    c = sbc(ARToy(), quantities, NAMES, replace(CFG, N=50))
    np.testing.assert_array_equal(a.ranks, c.ranks)
    np.testing.assert_array_equal(a.ranks, b.ranks)
    np.testing.assert_array_equal(a.thin, b.thin)


def test_iat_estimator():
    rng = np.random.default_rng(0)
    n, rho = 200_000, 0.8
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * e[t]
    assert iat(x) == pytest.approx((1 + rho) / (1 - rho), rel=0.08)
    assert iat(rng.standard_normal((3, 5000))) == pytest.approx(np.ones(3), abs=0.15)
    assert iat(np.full(10, 2.5)) == 1.0
    assert iat(np.stack([x[:1000], np.zeros(1000)])).shape == (2,)
    with pytest.raises(ValueError):
        iat(np.zeros(3))


def test_geyer_truncation_and_monotone_step():
    # pairs P_k = ρ_2k + ρ_2k+1 = 1.5, 0.2, 0.6, 0.1, −0.1, 0.3: truncated before P_4; the
    # initial-positive sum gives −1 + 2(1.5 + 0.2 + 0.6 + 0.1) = 3.8, the monotone one
    # (P_2 → min(0.6, 0.2)) −1 + 2(1.5 + 0.2 + 0.2 + 0.1) = 3.0
    rho = np.array([1.0, 0.5, 0.1, 0.1, 0.3, 0.3, 0.05, 0.05, -0.05, -0.05, 0.15, 0.15])
    assert _geyer_tau(2.5 * rho) == pytest.approx(3.0, abs=1e-12)  # scale-free (γ_k/γ_0)
    np.testing.assert_allclose(_geyer_tau(np.stack([rho[:10], 4.0 * rho[:10]])), 3.0, rtol=1e-12)
    # floored at 1, and a constant sequence (γ_0 = 0) gives 1
    assert _geyer_tau(np.array([1.0, -0.4, -0.2, 0.0])) == 1.0
    assert _geyer_tau(np.zeros(6)) == 1.0


def test_pilot_iat_uses_the_tail_indicators():
    # zero-mean regime switching: x = ±10 + ε in a persistent regime B, ε otherwise. x itself is
    # serially uncorrelated (IAT ≈ 1); the tail indicators follow the regime (IAT ≈ 20–50).
    rng = np.random.default_rng(0)
    n = 20_000
    u = rng.random(n)
    b = np.zeros(n, dtype=bool)
    for t in range(1, n):
        b[t] = u[t] >= 0.01 if b[t - 1] else u[t] < 0.0025
    x = np.where(b, 10.0 * rng.choice([-1.0, 1.0], n), 0.0) + rng.standard_normal(n)
    q05, q95 = np.quantile(x, [0.05, 0.95])
    tails = iat(np.stack([x <= q05, x <= q95]).astype(float))
    assert iat(x) < 1.1 and np.all(tails > 15.0)
    assert _pilot_iat(x[:, None]) == pytest.approx(tails.max(), rel=1e-12)
    two = np.stack([rng.standard_normal(n), x], axis=1)  # the max runs over quantities too
    assert _pilot_iat(two) == pytest.approx(tails.max(), rel=1e-12)


def test_thinned_ess_exact_values():
    tau, t = np.array([9.0, 9.0, 1.0, 3.0]), np.array([1, 2, 1, 4])
    # ρ = (τ − 1)/(τ + 1) = 0.8, 0.8, 0, 0.5;  ESS = L(1 − ρ^t)/(1 + ρ^t)
    expected = 99 * np.array([0.2 / 1.8, 0.36 / 1.64, 1.0, 0.9375 / 1.0625])
    np.testing.assert_allclose(_thinned_ess(tau, t, 99), expected, rtol=1e-13)
    assert _thinned_ess(np.array([9.0]), np.array([18]), 99)[0] > 0.96 * 99  # uncapped t = 2τ


def ar1(rng, shape, phi):
    """Stationary Gaussian AR(1) along axis 1 of ``shape`` (unit variance)."""
    e = rng.standard_normal(shape)
    x = np.empty(shape)
    x[:, 0] = e[:, 0]
    for t in range(1, shape[1]):
        x[:, t] = phi * x[:, t - 1] + np.sqrt(1 - phi**2) * e[:, t]
    return x


def pooled(draws, min_reps=40):
    acov = _acov(_indicators(draws, KEPT_LEVELS))
    return _pooled_ess(acov.sum(axis=0), (acov[..., 0] ** 2).sum(axis=0), draws.shape[1], min_reps)


def test_pooled_ess_is_scale_free_and_needs_effective_replications():
    rng = np.random.default_rng(3)
    L = 99
    iid = rng.standard_normal((100, L, 1))
    assert pooled(iid)[0] > 0.9 * L
    assert pooled(ar1(rng, (100, L, 1), 0.6))[0] < 0.6 * L  # autocorrelated kept draws
    # one slow (φ = 0.95) replication in 100 (the 1% that ess_coverage tolerates) with 1000× the
    # scale: the pooled lag-1 autocorrelation of f is dominated by it (N06 §4.2), while the
    # indicator rule weighs every replication equally
    mixed = iid.copy()
    mixed[:1] = 1000.0 * ar1(rng, (1, L, 1), 0.95)
    xc = mixed[..., 0] - mixed[..., 0].mean(axis=1, keepdims=True)
    assert np.sum(xc[:, 1:] * xc[:, :-1]) / np.sum(xc**2) > 0.7
    assert pooled(mixed)[0] > 0.9 * L
    np.testing.assert_array_equal(
        pooled(mixed * np.linspace(1, 50, 100)[:, None, None]), pooled(mixed)
    )
    # a binary quantity that varies in only a few replications has too few effective replications
    rare = np.zeros((100, L, 2))
    rare[:, :, 0] = rng.standard_normal((100, L))
    rare[:3, 10:20, 1] = 1.0  # strongly autocorrelated, but 3 replications carry no evidence
    ess = pooled(rare)
    assert ess[0] > 0.9 * L and np.isnan(ess[1])
    assert pooled(rare, min_reps=3)[1] < 0.8 * L


# --- block-conditional SBC ----------------------------------------------------------------------
# Two blocks b1, b2 with independent normal priors and y_t ~ N(b1 + 2 b2, s²): the exact
# conditional of b1 | b2, y is Gaussian; the "coupling" error ignores b2 (the CCM-2019 class).

M0, S0, S_OBS, T_OBS = jnp.array([0.3, -0.2]), jnp.array([0.8, 0.5]), 0.7, 6


def joint_sample(key):
    k1, k2 = random.split(key)
    b = M0 + S0 * random.normal(k1, (2,), dtype=jnp.float64)
    y = b[0] + 2.0 * b[1] + S_OBS * random.normal(k2, (T_OBS,), dtype=jnp.float64)
    return {"b1": b[0], "b2": b[1]}, y


def make_conditional(var_mult=1.0, couple=True):
    def draw(key, state, y):
        b2 = state["b2"] if couple else 0.0
        prec = 1.0 / S0[0] ** 2 + T_OBS / S_OBS**2
        m = (M0[0] / S0[0] ** 2 + jnp.sum(y - 2.0 * b2) / S_OBS**2) / prec
        b1 = m + jnp.sqrt(var_mult / prec) * random.normal(key, (), dtype=jnp.float64)
        return {"b1": b1, "b2": state["b2"]}

    return draw


def block_stat(state, y):
    loglik = jnp.sum(jstats.norm.logpdf(y, state["b1"] + 2.0 * state["b2"], S_OBS))
    return jnp.stack([state["b1"], loglik])


def test_block_sbc_correct_conditional_holds_the_level():
    conditional = make_conditional()  # one function object: compiled once
    results = [
        block_sbc(
            joint_sample, conditional, block_stat, N=200, L=99, seed=s, names=("b1", "loglik")
        )
        for s in range(10)
    ]
    assert sum(not r.passed for r in results) <= 1  # α = 1e-3 per run
    assert all(r.ranks.shape == (200, 2) and r.thin is None for r in results)
    ranks = np.concatenate([r.ranks for r in results])
    for q in range(2):
        assert stats.chisquare(np.bincount(ranks[:, q] // 10, minlength=10)).pvalue > 1e-3


@pytest.mark.parametrize("conditional", [make_conditional(couple=False), make_conditional(2.0)])
def test_block_sbc_detects_wrong_conditionals(conditional):
    res = block_sbc(
        joint_sample, conditional, block_stat, N=200, L=99, seed=1, names=("b1", "loglik")
    )
    assert not res.passed and res.failing


def test_block_sbc_validation():
    with pytest.raises(ValueError, match="names"):
        block_sbc(joint_sample, make_conditional(), block_stat, N=10, names=("only_one",))
    with pytest.raises(TypeError, match="f64"):
        block_sbc(joint_sample, make_conditional(), lambda s, y: s["b1"], N=10)
    with pytest.raises(ValueError, match="alpha"):
        block_sbc(joint_sample, make_conditional(), block_stat, N=10, alpha=2.0)


def test_config_validation():
    with pytest.raises(ValueError, match="pilot"):
        SBCConfig(pilot=2)
    with pytest.raises(ValueError, match="alpha"):
        SBCConfig(alpha=1.0)
    with pytest.raises(ValueError, match="uint32"):
        SBCConfig(burn=2**31, L=99, t_max=2**26)
    with pytest.raises(ValueError, match="names"):
        sbc(ARToy(), quantities, ("a", "b"), CFG)
