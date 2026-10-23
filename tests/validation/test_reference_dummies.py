"""The SOC/SUR dummy-row reference (``validation.dummies``, WP25) against closed forms written here in
a third algebra (precision form, NumPy): the prior *after* the rows is the Gaussian (Mode I,
reduced form) or normal–inverse-gamma (Mode II) posterior of the base prior given the rows.

* ``log_prior`` (Bayes' rule in covariance form) equals ``log N(θ; μ_d, K_d⁻¹)`` (Mode I, reduced)
  and ``log IG(λ; c + m/2, C + ½ρ²) + log N(θ; θ̂, λK̃⁻¹)`` (Mode II) to 1e-10;
* the prior sampler (Matheron's rule) has those moments (Monte Carlo, ``|z| ≤ 5``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from rai_bvar.engines.mcmc.state import ConstantEq, LinearEq, RecursiveEq
from rai_bvar.validation.dummies import (
    reduced_dummy_log_factor,
    reduced_dummy_prior_sample,
    structural_dummy_log_factor_eq,
    structural_dummy_prior_sample_eq,
)
from rai_bvar.validation.joint import joint_spec
from rai_bvar.validation.reference import references_for

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261026, impl="threefry2x32")
N_MC = 40_000


def _slice(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def closed_form(spec, i: int):
    """Precision-form posterior of equation ``i``'s ``θ_i = [δ_i | a_{i,<i}]`` given the rows: λ-free
    ``(K, μ_d)`` and, for Mode II, the IG update ``ρ² = y'y + μ₀'Pμ₀ − μ_d'Kμ_d`` (``w`` folded)."""
    prior = spec.prior
    Yd, Xd = np.asarray(prior.shared.dummies.Y), np.asarray(prior.shared.dummies.X)
    Mc = np.asarray(prior.shared.Mc)
    Fd = np.hstack([Xd, Xd @ Mc[:, :i] - Yd[:, :i]])
    prec = np.concatenate([np.asarray(prior.eq.coef.prec0)[i],
                           np.asarray(prior.eq.contemp.prec0)[i, :i]])  # fmt: skip
    mu0 = np.concatenate([np.asarray(prior.eq.coef.mean0)[i], np.zeros(i)])
    w = float(prior.eq.coef.dummy_w[i])
    y = Yd[:, i]
    K = np.diag(prec) + w * Fd.T @ Fd
    mu = np.linalg.solve(K, prec * mu0 + w * Fd.T @ y)
    rho2 = w * y @ y + mu0 @ (prec * mu0) - mu @ K @ mu
    return K, mu, rho2, Fd.shape[0]


def log_mvn(x, mu, K):
    d = x - mu
    _, logdet = np.linalg.slogdet(K)
    return 0.5 * (logdet - len(x) * np.log(2.0 * np.pi) - d @ K @ d)


def test_mode1_log_density_and_sampler():
    spec = joint_spec("structural_constant_dummies", n=3)
    static, prior = spec.static, spec.prior
    k = static.shapes.k
    refs = references_for(static)
    rng = np.random.default_rng(1)
    for i in range(3):
        K, mu, _, _ = closed_form(spec, i)
        prior_i = _slice(prior.eq, i)
        for _ in range(3):
            th = mu + rng.standard_normal(mu.size) * 0.3
            a = np.zeros(2)
            a[:i] = th[k:]
            ref = (
                refs.structure.log_prior_eq(RecursiveEq(a=jnp.asarray(a)), prior_i, static, None)
                + refs.mean.log_prior_eq(LinearEq(delta=jnp.asarray(th[:k])), prior_i, static, None)
                + structural_dummy_log_factor_eq(jnp.int32(i), jnp.asarray(th[:k]), jnp.asarray(a),
                                                 0.0, prior_i, prior.shared, static)
            )  # fmt: skip
            np.testing.assert_allclose(float(ref), log_mvn(th, mu, K), rtol=1e-10)
        core = ConstantEq(log_lam=jnp.zeros(()))
        draw = jax.jit(jax.vmap(lambda key, i=i, prior_i=prior_i, core=core:
                                structural_dummy_prior_sample_eq(
                                    key, jnp.int32(i), prior_i, prior.shared, static, core)))  # fmt: skip
        _, a_s, d_s = draw(random.split(random.fold_in(KEY, i), N_MC))
        th_s = np.hstack([np.asarray(d_s.delta), np.asarray(a_s.a)[:, :i]])
        assert np.all(np.asarray(a_s.a)[:, i:] == 0.0)
        cov = np.linalg.inv(K)
        z_mean = (th_s.mean(axis=0) - mu) / np.sqrt(np.diag(cov) / N_MC)
        z_var = (th_s.var(axis=0) / np.diag(cov) - 1.0) / np.sqrt(2.0 / N_MC)
        assert np.max(np.abs(z_mean)) < 5.0 and np.max(np.abs(z_var)) < 5.0


def test_mode2_log_density_and_sampler():
    spec = joint_spec("structural_mode2_constant_dummies", n=3)
    static, prior = spec.static, spec.prior
    k = static.shapes.k
    refs = references_for(static)
    rng = np.random.default_rng(2)
    for i in range(3):
        K, mu, rho2, m = closed_form(spec, i)
        prior_i = _slice(prior.eq, i)
        c, C = float(prior_i.vol_core.shape), float(prior_i.vol_core.scale)
        shape, scale = c + m / 2.0, C + rho2 / 2.0
        for _ in range(3):
            lam = float(stats.invgamma.rvs(shape, scale=scale, random_state=rng))
            th = mu + np.sqrt(lam) * rng.standard_normal(mu.size) * 0.3
            a = np.zeros(2)
            a[:i] = th[k:]
            ll = jnp.asarray(np.log(lam))
            ref = (
                refs.core.log_prior_eq(ConstantEq(log_lam=ll), prior_i, static, None)
                + refs.structure.log_prior_eq(RecursiveEq(a=jnp.asarray(a)), prior_i, static,
                                              None, log_scale=ll)
                + refs.mean.log_prior_eq(LinearEq(delta=jnp.asarray(th[:k])), prior_i, static,
                                         None, log_scale=ll)
                + structural_dummy_log_factor_eq(jnp.int32(i), jnp.asarray(th[:k]), jnp.asarray(a),
                                                 ll, prior_i, prior.shared, static)
            )  # fmt: skip
            want = stats.invgamma.logpdf(lam, shape, scale=scale) + log_mvn(th, mu, K / lam)
            np.testing.assert_allclose(float(ref), want, rtol=1e-10)
        core = ConstantEq(log_lam=jnp.zeros(()))
        draw = jax.jit(jax.vmap(lambda key, i=i, prior_i=prior_i, core=core:
                                structural_dummy_prior_sample_eq(
                                    key, jnp.int32(i), prior_i, prior.shared, static, core)))  # fmt: skip
        core_s, a_s, d_s = draw(random.split(random.fold_in(KEY, 10 + i), N_MC))
        lam_s = np.exp(np.asarray(core_s.log_lam))
        assert stats.kstest(lam_s, stats.invgamma(shape, scale=scale).cdf).pvalue > 1e-4
        th_s = np.hstack([np.asarray(d_s.delta), np.asarray(a_s.a)[:, :i]])
        std = (th_s - mu) / np.sqrt(lam_s)[:, None]  # ~ N(0, K⁻¹) whatever λ
        cov = np.linalg.inv(K)
        z_mean = std.mean(axis=0) / np.sqrt(np.diag(cov) / N_MC)
        z_var = (std.var(axis=0) / np.diag(cov) - 1.0) / np.sqrt(2.0 / N_MC)
        assert np.max(np.abs(z_mean)) < 5.0 and np.max(np.abs(z_var)) < 5.0


def test_reduced_log_density_and_sampler():
    spec = joint_spec("reduced_constant_dummies")
    prior = spec.prior
    Yd, Xd = np.asarray(prior.shared.dummies.Y), np.asarray(prior.shared.dummies.X)
    mean0, prec0 = np.asarray(prior.eq.coef.mean0), np.asarray(prior.eq.coef.prec0)
    w = np.asarray(prior.eq.coef.dummy_w)
    n, k = mean0.shape
    Ks = [np.diag(prec0[j]) + w[j] * Xd.T @ Xd for j in range(n)]
    mus = [np.linalg.solve(Ks[j], prec0[j] * mean0[j] + w[j] * Xd.T @ Yd[:, j]) for j in range(n)]
    rng = np.random.default_rng(3)
    refs = references_for(spec.static)
    for _ in range(3):
        Pi = np.stack([mus[j] + 0.2 * rng.standard_normal(k) for j in range(n)], axis=1)
        from rai_bvar.engines.mcmc.state import CCCMGlobal

        ref = refs.mean.log_prior_global(CCCMGlobal(Pi=jnp.asarray(Pi)), prior, spec.static)
        want = sum(log_mvn(Pi[:, j], mus[j], Ks[j]) for j in range(n))
        np.testing.assert_allclose(float(ref), want, rtol=1e-10)
        base = float(reduced_dummy_log_factor(jnp.asarray(Pi), prior))
        assert np.isfinite(base)
    draws = jax.jit(jax.vmap(lambda key: reduced_dummy_prior_sample(key, prior).Pi))(
        random.split(KEY, N_MC)
    )
    draws = np.asarray(draws)
    for j in range(n):
        cov = np.linalg.inv(Ks[j])
        z_mean = (draws[:, :, j].mean(axis=0) - mus[j]) / np.sqrt(np.diag(cov) / N_MC)
        z_var = (draws[:, :, j].var(axis=0) / np.diag(cov) - 1.0) / np.sqrt(2.0 / N_MC)
        assert np.max(np.abs(z_mean)) < 5.0 and np.max(np.abs(z_var)) < 5.0


def test_iw_equivalent_rejection_sampler():
    """``iw_equivalent`` with rows: the rejection draws of ``λ_i | d`` follow the exact density
    ``∝ IG(λ; c, C) N(r; 0, S₀ + λUUᵀ)`` (CDF by numerical quadrature on a fine log-λ grid, KS
    test), and ``θ_i | λ_i, d`` is the Gaussian posterior of the base prior given the rows with the
    λ-scaled ``a`` block (standardised moments, ``|z| ≤ 5``)."""
    spec = joint_spec("structural_iw_equivalent_dummies")
    static, prior = spec.static, spec.prior
    k = static.shapes.k
    Yd, Xd = np.asarray(prior.shared.dummies.Y), np.asarray(prior.shared.dummies.X)
    Mc = np.asarray(prior.shared.Mc)
    for i in range(3):
        prior_i = _slice(prior.eq, i)
        c, C = float(prior_i.vol_core.shape), float(prior_i.vol_core.scale)
        w = float(prior.eq.coef.dummy_w[i])
        Fd = np.hstack([Xd, Xd @ Mc[:, :i] - Yd[:, :i]])
        vd = 1.0 / np.asarray(prior.eq.coef.prec0)[i]
        va = 1.0 / np.asarray(prior.eq.contemp.prec0)[i, :i]
        mu0 = np.concatenate([np.asarray(prior.eq.coef.mean0)[i], np.zeros(i)])
        r = Yd[:, i] - Fd @ mu0
        m = Fd.shape[0]

        def log_post(lam, Fd=Fd, vd=vd, va=va, r=r, w=w, c=c, C=C, m=m):
            S = (Fd[:, :k] * vd) @ Fd[:, :k].T + lam * (Fd[:, k:] * va) @ Fd[:, k:].T
            S = S + np.eye(m) / w
            return stats.invgamma.logpdf(lam, c, scale=C) + stats.multivariate_normal.logpdf(
                r, np.zeros(m), S
            )

        grid = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 20_001))
        lp = np.array([log_post(x) for x in grid])
        dens = np.exp(lp - lp.max()) * grid  # density of log λ on the uniform log grid
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]))])
        cdf /= cdf[-1]
        core = ConstantEq(log_lam=jnp.zeros(()))
        draw = jax.jit(
            jax.vmap(
                lambda key, i=i, prior_i=prior_i, core=core: structural_dummy_prior_sample_eq(
                    key, jnp.int32(i), prior_i, prior.shared, static, core
                )
            )
        )
        core_s, a_s, d_s = draw(random.split(random.fold_in(KEY, 20 + i), N_MC))
        lam_s = np.exp(np.asarray(core_s.log_lam))
        pval = stats.kstest(
            lam_s, lambda x, grid=grid, cdf=cdf: np.interp(np.log(x), np.log(grid), cdf)
        ).pvalue
        assert pval > 1e-4, (i, pval)
        # θ | λ: Gaussian posterior of N(μ₀, diag(vd, λ va)) given the rows (noise 1/w)
        th_s = np.hstack([np.asarray(d_s.delta), np.asarray(a_s.a)[:, :i]])
        zs = []
        for lam, th in zip(lam_s[:4000], th_s[:4000], strict=True):
            P = np.diag(1.0 / np.concatenate([vd, lam * va])) + w * Fd.T @ Fd
            mu = np.linalg.solve(P, P @ mu0 - w * Fd.T @ (Fd @ mu0) + w * Fd.T @ Yd[:, i])
            Lp = np.linalg.cholesky(P)
            zs.append(Lp.T @ (th - mu))  # ~ N(0, I)
        zs = np.asarray(zs)
        n_z = zs.shape[0]
        assert np.max(np.abs(zs.mean(axis=0)) * np.sqrt(n_z)) < 5.0
        assert np.max(np.abs(zs.var(axis=0) - 1.0) / np.sqrt(2.0 / n_z)) < 5.0
