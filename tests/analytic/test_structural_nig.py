"""WP16: the Chan (2022) structural NIG closed form (DESIGN §9.8 A1, §13 WP16; N05 §9, §13.2).

Oracles, none of which calls ``structural_nig``'s own linear algebra:

* the Phase 1 engine (``AnalyticFit``): log ML with and without SOC/SUR dummies, the NIW
  posterior ``(B̄, Ω̄, S̄, ν̄)`` and the EB mode at ``cross = 1`` (A1);
* textbook dense formulas with explicit inverses (fine in tests): the conjugate regression, the
  D3 prior in γ-coordinates (N05 §4.1) and the multivariate-t marginal
  ``y ~ t_{2ν₀}(Zμ₀, (S₀/ν₀)(I + ZV₀Z'))``;
* an exact cancellation case for the residual-form ``Ŝ`` (N05 §7.6), and exact rational
  arithmetic (``fractions``) where the level dwarfs the noise;
* the WP6 JAX-side prior builders, for the NumPy-built Mode II arrays;
* Monte Carlo moments of the iid draws against the closed-form moments.
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, stats
from scipy.special import gammaln

import rai_bvar as rb
from rai_bvar.data.design import lagged_design
from rai_bvar.engines.structural_nig import (
    EBResult,
    NIGPosterior,
    draw_structural_nig,
    eb_structural,
    nig_posterior,
    structural_log_ml,
    structural_posteriors,
    structural_to_reduced,
)

_LOG_2PI = float(np.log(2.0 * np.pi))
_Z_MAX = 4.5  # Monte Carlo z-score bound (fixed seeds; ~50 quantities per test)


# --------------------------------------------------------------------------- helpers


def _var_data(n: int, T: int, seed: int) -> rb.VARData:
    """A stationary VAR(1) sample with unequal scales and correlated shocks (tcode 1)."""
    rng = np.random.default_rng(seed)
    A = np.diag(np.linspace(0.9, 0.3, n)) + 0.05 * rng.standard_normal((n, n))
    sd = np.linspace(0.5, 3.0, n)
    chol = np.linalg.cholesky(0.3 * np.ones((n, n)) + 0.7 * np.eye(n)) * sd[:, None]
    y, out = np.zeros(n), np.empty((T, n))
    for t in range(T + 50):
        y = 0.2 + A @ y + chol @ rng.standard_normal(n)
        if t >= 50:
            out[t - 50] = y
    index = pd.period_range("1980Q1", periods=T, freq="Q")
    return rb.VARData.from_frame(
        pd.DataFrame(out, index=index, columns=[f"v{j}" for j in range(n)])
    )


def _phase1(prior: rb.Minnesota, data: rb.VARData, lags: int, deterministic: str):
    model = rb.BVAR(mean=rb.Linear(prior=prior), lags=lags, deterministic=deterministic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # tcode clashes of the chosen own-lag means
        return model.fit(data)


def _regression(seed: int, T: int, q: int):
    rng = np.random.default_rng(seed)
    scale = np.logspace(-0.5, 0.5, q)  # moderate conditioning: the T×T dense oracle stays exact
    Z = rng.standard_normal((T, q)) * scale
    y = Z @ (rng.standard_normal(q) / scale) + rng.standard_normal(T)
    mu0 = rng.standard_normal(q)
    var0 = np.logspace(-1, 1, q)
    return Z, y, mu0, var0


def _mvt_log_ml(Z, y, mu0, var0, nu0, S0) -> float:
    """Marginal of the conjugate regression: ``y ~ t_{2ν₀}(Zμ₀, (S₀/ν₀)(I + Z diag(var0) Z'))``."""
    shape = (S0 / nu0) * (np.eye(len(y)) + (Z * var0) @ Z.T)
    return float(stats.multivariate_t(loc=Z @ mu0, shape=shape, df=2.0 * nu0).logpdf(y))


# --------------------------------------------------------------------------- one regression


def test_nig_posterior_matches_dense_formulas():
    Z, y, mu0, var0 = _regression(0, 40, 6)
    nu0, S0 = 2.5, 0.7
    post = nig_posterior(Z, y, mu0, var0, nu0, S0)
    assert isinstance(post, NIGPosterior)
    K = np.diag(1.0 / var0) + Z.T @ Z
    theta = np.linalg.solve(K, mu0 / var0 + Z.T @ y)
    assert np.array_equal(post.R, np.triu(post.R)) and np.all(np.diag(post.R) > 0)
    assert_allclose(post.R.T @ post.R, K, rtol=1e-12, atol=1e-12 * np.abs(K).max())
    assert_allclose(post.theta_hat, theta, rtol=1e-10)
    naive = S0 + 0.5 * (y @ y + mu0 @ (mu0 / var0) - theta @ K @ theta)  # Chan eq. 10
    assert post.S_hat == pytest.approx(naive, rel=1e-10)
    assert post.nu_hat == nu0 + 20.0
    assert post.log_ml == pytest.approx(_mvt_log_ml(Z, y, mu0, var0, nu0, S0), rel=1e-11)
    assert post.lam_mean == pytest.approx(post.S_hat / (post.nu_hat - 1.0))
    assert_allclose(post.cov(), post.lam_mean * np.linalg.inv(K), rtol=1e-9)


def test_nig_posterior_without_rows_is_the_prior():
    _, _, mu0, var0 = _regression(1, 1, 4)
    post = nig_posterior(np.zeros((0, 4)), np.zeros(0), mu0, var0, 1.5, 0.4)
    assert_allclose(post.theta_hat, mu0, rtol=1e-15)
    assert_allclose(post.R.T @ post.R, np.diag(1.0 / var0), rtol=1e-14)
    assert (post.nu_hat, post.S_hat) == (1.5, 0.4)
    assert post.log_ml == pytest.approx(0.0, abs=1e-13)


def test_residual_form_survives_cancellation():
    """Level 1e8, residuals ±0.5 orthogonal to both regressors and a prior mean at the truth:
    ``θ̂ = μ₀`` and ``Ŝ = S₀ + T/8`` exactly, while ``y'y + μ₀'V₀⁻¹μ₀ − θ̂'Kθ̂`` cancels ~1e18."""
    T, level, nu0, S0 = 64, 1e8, 2.0, 0.3
    e = 0.5 * np.tile([1.0, 1.0, -1.0, -1.0], T // 4)
    Z = np.column_stack([np.ones(T), np.tile([1.0, -1.0], T // 2)])  # Z'e = 0, Z'Z = T·I
    y = level + e
    assert np.array_equal(y - level, e)  # exact in float64
    mu0, var0 = np.array([level, 0.0]), np.array([1e7, 1e7])
    post = nig_posterior(Z, y, mu0, var0, nu0, S0)
    exact = S0 + T / 8.0
    assert post.S_hat == pytest.approx(exact, rel=1e-13)
    K = post.R.T @ post.R
    naive = S0 + 0.5 * (y @ y + mu0 @ (mu0 / var0) - post.theta_hat @ K @ post.theta_hat)
    assert abs(naive - exact) > 0.1 * exact  # the textbook form is useless here
    nu_hat = nu0 + T / 2.0
    log_ml = (
        -0.5 * T * _LOG_2PI
        - 0.5 * np.log(var0).sum()
        - 0.5 * np.log(1.0 / var0 + T).sum()  # K = diag(1/var0 + T)
        + gammaln(nu_hat)
        - gammaln(nu0)
        + nu0 * np.log(S0)
        - nu_hat * np.log(exact)
    )
    assert post.log_ml == pytest.approx(log_ml, rel=1e-13)


def _exact_nig(Z, y, mu0, var0, nu0, S0) -> tuple[float, float, np.ndarray]:
    """``(Ŝ, log p(y), θ̂)`` in exact rational arithmetic (every float64 is a rational): Gaussian
    elimination on ``K = V₀⁻¹ + Z'Z``; only the final logs are taken in float64."""
    T, q = Z.shape
    Zf = [[Fraction(float(v)) for v in row] for row in Z]
    yf = [Fraction(float(v)) for v in y]
    p0 = [1 / Fraction(float(v)) for v in var0]
    m0 = [Fraction(float(v)) for v in mu0]
    aug = [
        [sum(Zf[t][a] * Zf[t][b] for t in range(T)) + (p0[a] if a == b else 0) for b in range(q)]
        + [sum(Zf[t][a] * yf[t] for t in range(T)) + p0[a] * m0[a]]
        for a in range(q)
    ]
    det = Fraction(1)
    for c in range(q):
        det *= aug[c][c]
        for r in range(c + 1, q):
            f = aug[r][c] / aug[c][c]
            aug[r] = [aug[r][j] - f * aug[c][j] for j in range(q + 1)]
    theta = [Fraction(0)] * q
    for r in reversed(range(q)):
        theta[r] = (aug[r][q] - sum(aug[r][j] * theta[j] for j in range(r + 1, q))) / aug[r][r]
    rho2 = sum((yf[t] - sum(Zf[t][j] * theta[j] for j in range(q))) ** 2 for t in range(T))
    rho2 += sum(p0[j] * (theta[j] - m0[j]) ** 2 for j in range(q))
    S_hat = Fraction(S0) + rho2 / 2
    nu_hat = nu0 + T / 2.0
    log_ml = (
        -0.5 * T * _LOG_2PI
        - 0.5 * np.log(var0).sum()
        - 0.5 * math.log(det)
        + gammaln(nu_hat)
        - gammaln(nu0)
        + nu0 * math.log(S0)
        - nu_hat * math.log(S_hat)
    )
    return float(S_hat), float(log_ml), np.array([float(v) for v in theta])


@pytest.mark.parametrize("seed", range(3))
def test_s_hat_is_exact_when_the_level_dwarfs_the_noise(seed):
    """Random walks at level 1e6 with steps of 0.01 (noise/level 1e-8): ``ρ²`` read off the
    prior-first QR keeps ``Ŝ`` to ~1e-12, where an explicit residual ``y − Zθ̂`` loses ~1e-9
    (reviewer finding). Powers-of-4 prior variances make ``V₀^{−½}`` exact, so the rational
    oracle sees the same problem. θ̂ is ill-conditioned along the intercept (posterior s.d. ~40);
    its error is judged in posterior standard deviations."""
    rng = np.random.default_rng(seed)
    T = 60
    walk = 1e6 + np.cumsum(0.01 * rng.standard_normal((T + 1, 2)), axis=0)
    Z, y = np.column_stack([np.ones(T), walk[:-1]]), walk[1:, 0]
    mu0, var0 = np.array([0.0, 1.0, 0.0]), np.array([2.0**24, 2.0**-4, 2.0**-6])
    nu0, S0 = 1.5, 2.0**-15
    S_hat, log_ml, theta = _exact_nig(Z, y, mu0, var0, nu0, S0)
    post = nig_posterior(Z, y, mu0, var0, nu0, S0)
    assert post.S_hat == pytest.approx(S_hat, rel=1e-10)
    assert post.log_ml == pytest.approx(log_ml, abs=1e-9)
    sd = np.sqrt(np.diag(post.cov()))
    assert np.max(np.abs(post.theta_hat - theta) / sd) < 1e-6


@pytest.mark.parametrize(
    "change, message",
    [
        ({"Z": np.zeros(5)}, "matrix"),
        ({"y": np.zeros(4)}, "shapes"),
        ({"mu0": np.zeros(2)}, "shapes"),
        ({"var0": np.array([1.0, 0.0, 1.0])}, "var0"),
        ({"var0": np.array([1.0, np.inf, 1.0])}, "var0"),
        ({"y": np.array([0.0, np.nan, 0.0, 0.0, 0.0])}, "finite"),
        ({"nu0": 0.0}, "nu0"),
        ({"S0": -1.0}, "S0"),
    ],
)
def test_nig_posterior_rejects_bad_input(change, message):
    args = {
        "Z": np.ones((5, 3)) + np.eye(5, 3),
        "y": np.zeros(5),
        "mu0": np.zeros(3),
        "var0": np.ones(3),
        "nu0": 2.0,
        "S0": 1.0,
    }
    with pytest.raises(ValueError, match=message):
        nig_posterior(**{**args, **change})


def test_nig_posterior_raises_on_rank_deficiency():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((30, 3))
    Z = np.column_stack([Z, Z[:, 0]])  # duplicated column, prior too flat to separate them
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        nig_posterior(Z, rng.standard_normal(30), np.zeros(4), np.full(4, 1e30), 2.0, 1.0)
    nig_posterior(Z, rng.standard_normal(30), np.zeros(4), np.full(4, 1e6), 2.0, 1.0)  # proper


# --------------------------------------------------------------------------- A1: Phase 1 twin

A1_CASES = {
    "c": dict(n=3, lags=2, deterministic="c"),
    "soc+sur": dict(  # N05 §13.2 check A1: n = 4, own means (1, .5, 0, .9), SOC 1.5, SUR 2
        n=4,
        lags=2,
        deterministic="c",
        soc=1.5,
        sur=2.0,
        mean={"v0": 1.0, "v1": 0.5, "v2": 0.0, "v3": 0.9},
    ),
    "trend+sur": dict(n=3, lags=1, deterministic="ct", sur=0.7, psi=(0.8, 1.3, 0.6)),
    "no-det+soc-all": dict(n=2, lags=3, deterministic="n", soc=0.8, soc_scope="all"),
    "n=1": dict(n=1, lags=2, deterministic="c", sur=1.2),
}


def _a1_setup(case: dict):
    case = dict(case)
    n, lags, det = case.pop("n"), case.pop("lags"), case.pop("deterministic")
    data = _var_data(n, 90, 7)
    prior = rb.Minnesota(lambda_=0.3, alpha=1.7, var_const=10.0, hierarchy="fixed", **case)
    return data, prior, lags, det


@pytest.mark.parametrize("name", list(A1_CASES))
def test_a1_log_ml_equals_phase1(name):
    data, prior, lags, det = _a1_setup(A1_CASES[name])
    fit = _phase1(prior, data, lags, det)
    hypers = [
        fit.hyper,
        {**fit.hyper, "lambda": 0.6, "alpha": 1.2},
        {**fit.hyper, "psi": fit.hyper["psi"] * np.linspace(0.7, 1.4, data.n_vars)},
    ]
    if prior.soc is not None:
        hypers.append({**fit.hyper, "soc": 4.0})
    if prior.sur is not None:
        hypers.append({**fit.hyper, "sur": 0.2})
    for hyper in hypers:
        value = structural_log_ml(data, prior, hyper, lags, det)
        assert value == pytest.approx(fit.log_ml(hyper), abs=1e-10)
    posts = structural_posteriors(data, prior, fit.hyper, lags, det)
    assert sum(p.log_ml for p in posts) == pytest.approx(fit.log_ml(), abs=1e-10)
    # missing entries come from the prior and psi="ar" from the Phase 1 AR variances
    assert structural_log_ml(data, prior, {}, lags, det) == pytest.approx(fit.log_ml(), abs=1e-10)


@pytest.mark.parametrize("level", [1e4, 1e6])
def test_a1_holds_when_the_level_dwarfs_the_noise(level):
    """Random walks at ``level`` with step s.d. 0.01 … 10 (reviewer finding): A1 still holds to
    1e-10. With data rows first and an explicit residual the gap was 8e-10 (1e4) and 1e-7 (1e6)."""
    rng = np.random.default_rng(1)
    T, n = 120, 4
    y = level + np.cumsum(rng.standard_normal((T, n)) * np.array([0.01, 0.1, 1.0, 10.0]), axis=0)
    index = pd.period_range("1960Q1", periods=T, freq="Q")
    data = rb.VARData.from_frame(pd.DataFrame(y, index=index, columns=[f"v{j}" for j in range(n)]))
    prior = rb.Minnesota(var_const=1e7, hierarchy="fixed")
    fit = _phase1(prior, data, 2, "c")
    value = structural_log_ml(data, prior, fit.hyper, 2, "c")
    assert value == pytest.approx(fit.log_ml(), abs=1e-10)


@pytest.mark.parametrize(
    "hyper, dummy", [("auto", None), (("lambda", "alpha"), None), ("auto", 1.0)]
)
def test_a1_eb_equals_phase1_eb_at_cross_one(hyper, dummy):
    """Started at the prior defaults (not at the Phase 1 mode), the structural EB mode is the
    Phase 1 EB mode, including SOC/SUR weights when they are free."""
    data = _var_data(3, 120, 11)
    prior = rb.Minnesota(hyper=hyper, soc=dummy, sur=dummy, var_const=10.0)
    fit = _phase1(prior, data, 2, "c")
    assert fit.converged
    result = eb_structural(data, prior, 2, "c", {})
    assert isinstance(result, EBResult)
    assert result.converged and result.warnings == ()
    assert result.free == prior.free
    for name in prior.free:
        assert result.hyper[name] == pytest.approx(fit.hyper[name], abs=1e-6)
    assert_allclose(result.hyper["psi"], fit.hyper["psi"], rtol=1e-15)
    assert result.log_ml == pytest.approx(fit.log_ml(), abs=1e-8)
    assert result.log_posterior == pytest.approx(fit.log_posterior(), abs=1e-8)


# --------------------------------------------------------------------------- posterior identities


def test_posterior_identities_against_phase1_niw():
    """N05 §13.2 item 2 with SOC/SUR dummies: ``ν̂_i = (ν̄ − n + ι)/2``, ``2Ŝ_i`` is the Schur
    complement of ``S̄``, ``E[a_i] = −S̄_{<i}⁻¹S̄_{<i,i}``, ``Cov(a_i | λ)/λ = S̄_{<i}⁻¹``, and
    ``γ_i | a_i`` has precision ``X*'X*`` and mean ``B̄ã_i`` (explicit inverses)."""
    data, prior, lags, det = _a1_setup(A1_CASES["soc+sur"])
    fit = _phase1(prior, data, lags, det)
    niw = fit.posterior()
    B, S, nu_bar, P = niw.mean, niw.scale, niw.df, niw.row_upper.T @ niw.row_upper
    n, k = data.n_vars, B.shape[0]
    posts = structural_posteriors(data, prior, fit.hyper, lags, det)
    rng = np.random.default_rng(3)
    for i, post in enumerate(posts):
        assert post.theta_hat.shape == (k + i,)
        assert post.nu_hat == pytest.approx((nu_bar - n + i + 1) / 2.0, rel=1e-15)
        schur = S[i, i] - (S[i, :i] @ np.linalg.solve(S[:i, :i], S[:i, i]) if i else 0.0)
        assert 2.0 * post.S_hat == pytest.approx(schur, rel=1e-10)
        K = post.R.T @ post.R
        assert_allclose(K[:k, :k], P, rtol=1e-10, atol=1e-12 * np.abs(P).max())
        a_hat = post.theta_hat[k:]
        if i:
            assert_allclose(a_hat, -np.linalg.solve(S[:i, :i], S[:i, i]), rtol=1e-9, atol=1e-12)
            cov_a = np.linalg.inv(K)[k:, k:]
            assert_allclose(cov_a, np.linalg.inv(S[:i, :i]), rtol=1e-9)
        a = a_hat + rng.standard_normal(i)
        cond_mean = post.theta_hat[:k] - np.linalg.solve(K[:k, :k], K[:k, k:] @ (a - a_hat))
        a_tilde = np.concatenate([a, [1.0], np.zeros(n - i - 1)])
        assert_allclose(cond_mean, B @ a_tilde, rtol=1e-9, atol=1e-10 * np.abs(B).max())


def test_gamma_coordinates_match_the_dense_d3_prior():
    """``κ_c = 0.5``, ``κ_a = 0.6``: the γ-coordinate posteriors equal the dense γ-coordinate
    regression on ``[X, −Y_{<i}]`` under the joint D3 prior of N05 §4.1, with Mode II scales
    written out from DESIGN §6.1, and the log ML equals the multivariate-t marginal."""
    psi = np.array([0.8, 1.9, 0.6])
    means = {"v0": 1.0, "v1": 0.4, "v2": 0.9}
    lam, alpha, cross, contemp, var_const = 0.3, 1.5, 0.5, 0.6, 10.0
    prior = rb.Minnesota(
        lambda_=lam,
        alpha=alpha,
        psi=tuple(psi),
        mean=means,
        cross=cross,
        contemp=contemp,
        var_const=var_const,
        form="structural",
        hierarchy="fixed",
    )
    data = _var_data(3, 70, 5)
    n, lags = 3, 2
    posts = structural_posteriors(data, prior, {}, lags, "c")
    Y, X = lagged_design(data.values, lags, "c")
    k = X.shape[1]
    M = np.zeros((k, n))
    M[1 + np.arange(n), np.arange(n)] = list(means.values())
    for i, post in enumerate(posts):
        omega = [var_const]
        for lag in range(1, lags + 1):
            for m in range(n):
                c = 1.0 if m == i else cross**2
                omega.append(lam**2 * c / (lag**alpha * psi[m]))  # (d − n − 1) = 1
        V_g, V_a = np.diag(omega), np.diag(contemp / psi[:i])
        Mi = M[:, :i]
        V0 = np.block([[V_g + Mi @ V_a @ Mi.T, Mi @ V_a], [V_a @ Mi.T, V_a]])  # [γ_i | a_i]
        mu0 = np.concatenate([M[:, i], np.zeros(i)])
        P0 = np.linalg.inv(V0)
        Z = np.hstack([X, -Y[:, :i]])
        y = Y[:, i]
        K = P0 + Z.T @ Z
        theta = np.linalg.solve(K, P0 @ mu0 + Z.T @ y)
        nu0, S0 = 1.0 + (i + 1) / 2.0, psi[i] / 2.0
        resid, gap = y - Z @ theta, theta - mu0
        assert np.array_equal(post.R, np.triu(post.R))
        assert_allclose(post.R.T @ post.R, K, rtol=1e-9, atol=1e-11 * np.abs(K).max())
        assert_allclose(post.theta_hat, theta, rtol=1e-9, atol=1e-11)
        assert post.S_hat == pytest.approx(S0 + 0.5 * (resid @ resid + gap @ P0 @ gap), rel=1e-10)
        shape = (S0 / nu0) * (np.eye(len(y)) + Z @ V0 @ Z.T)
        ref = stats.multivariate_t(loc=Z @ mu0, shape=shape, df=2.0 * nu0).logpdf(y)
        assert post.log_ml == pytest.approx(float(ref), abs=1e-8)
    total = sum(p.log_ml for p in posts)
    assert structural_log_ml(data, prior, {}, lags, "c") == pytest.approx(total, abs=1e-12)


def test_log_ml_is_permutation_invariant_only_at_cross_one():
    """N05 §13.2 (c): NIW (``cross = contemp = 1``) is permutation-equivariant; Chan's
    asymmetric prior (Cor. 2) is not."""
    data = _var_data(3, 90, 13)
    order = ["v2", "v0", "v1"]
    means = {"v0": 1.0, "v1": 0.0, "v2": 0.5}
    for cross, invariant in ((1.0, True), (0.4, False)):
        prior = rb.Minnesota(
            lambda_=0.3, mean=means, cross=cross, form="structural", hierarchy="fixed"
        )
        a = structural_log_ml(data, prior, {}, 2, "c")
        b = structural_log_ml(data.select(order), prior, {}, 2, "c")
        if invariant:
            assert a == pytest.approx(b, abs=1e-9)
        else:
            assert abs(a - b) > 1e-3


# --------------------------------------------------------------------------- iid draws


def test_draws_reproduce_the_phase1_niw_moments():
    data, prior, lags, det = _a1_setup(A1_CASES["trend+sur"])
    fit = _phase1(prior, data, lags, det)
    posts = structural_posteriors(data, prior, fit.hyper, lags, det)
    size = 40_000
    a, gamma, lam = draw_structural_nig(posts, size, np.random.default_rng(4))
    coefs, chol = structural_to_reduced(a, gamma, lam)
    sigma = chol @ chol.transpose(0, 2, 1)
    niw = fit.posterior()
    for draws, target in ((coefs, niw.mean), (sigma, niw.sigma_mean)):
        z = (draws.mean(axis=0) - target) / (draws.std(axis=0) / np.sqrt(size))
        assert np.max(np.abs(z)) < _Z_MAX


def test_draws_match_the_chan_closed_form_moments():
    data = _var_data(3, 80, 17)
    prior = rb.Minnesota(lambda_=0.25, cross=0.5, contemp=0.5, form="structural", var_const=10.0)
    posts = structural_posteriors(data, prior, {}, 2, "c")
    size = 40_000
    a, gamma, lam = draw_structural_nig(posts, size, np.random.default_rng(5))
    k = gamma.shape[2]
    for i, post in enumerate(posts):
        theta = np.hstack([gamma[:, i], a[:, i, :i]])
        assert np.all(a[:, i, i:] == 0.0)
        cov = post.cov()
        sd = np.sqrt(np.diag(cov))
        z = (theta.mean(axis=0) - post.theta_hat) / (sd / np.sqrt(size))
        assert np.max(np.abs(z)) < _Z_MAX
        corr_gap = (np.cov(theta, rowvar=False) - cov) / np.outer(sd, sd)
        assert np.max(np.abs(corr_gap)) < 0.04
        nu, S = post.nu_hat, post.S_hat
        lam_sd = S / ((nu - 1.0) * np.sqrt(nu - 2.0))
        assert abs(lam[:, i].mean() - post.lam_mean) / (lam_sd / np.sqrt(size)) < _Z_MAX
        assert theta.shape == (size, k + i)


def test_draw_structural_nig_contract():
    data = _var_data(2, 60, 19)
    prior = rb.Minnesota(cross=0.5, form="structural", hierarchy="fixed")
    posts = structural_posteriors(data, prior, {}, 1, "c")
    a, gamma, lam = draw_structural_nig(posts, 7, np.random.default_rng(6))
    assert (a.shape, gamma.shape, lam.shape) == ((7, 2, 1), (7, 2, 3), (7, 2))
    assert a.dtype == gamma.dtype == lam.dtype == np.float64
    assert np.all(a[:, 0] == 0.0) and np.all(lam > 0)
    again = draw_structural_nig(posts, 7, np.random.default_rng(6))
    assert all(np.array_equal(x, y) for x, y in zip((a, gamma, lam), again, strict=True))
    other = draw_structural_nig(posts, 7, np.random.default_rng(7))
    assert not np.array_equal(gamma, other[1])
    with pytest.raises(ValueError, match="k \\+ i"):
        draw_structural_nig(posts[::-1], 7, np.random.default_rng(6))
    with pytest.raises(ValueError, match="size"):
        draw_structural_nig(posts, 0, np.random.default_rng(6))
    with pytest.raises(TypeError, match="NIGPosterior"):
        draw_structural_nig([], 3, np.random.default_rng(6))
    one = structural_posteriors(data.select(["v1"]), prior, {}, 1, "c")
    a1, g1, l1 = draw_structural_nig(one, 5, np.random.default_rng(8))
    assert (a1.shape, g1.shape, l1.shape) == ((5, 1, 0), (5, 1, 2), (5, 1))


def test_structural_to_reduced_matches_explicit_inverse():
    rng = np.random.default_rng(9)
    S, n, k = 6, 4, 5
    a = np.tril(rng.standard_normal((S, n, n)), -1)[:, :, : n - 1]
    gamma = rng.standard_normal((S, n, k))
    lam = rng.uniform(0.2, 3.0, (S, n))
    coefs, chol = structural_to_reduced(a, gamma, lam)
    for s in range(S):
        A = np.eye(n)
        A[:, : n - 1] += a[s]
        A_inv = np.linalg.inv(A)
        assert_allclose(coefs[s], (A_inv @ gamma[s]).T, rtol=1e-12, atol=1e-12)
        sigma = A_inv @ np.diag(lam[s]) @ A_inv.T
        assert_allclose(chol[s], np.linalg.cholesky(sigma), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="lam"):
        structural_to_reduced(a, gamma, -lam)


# --------------------------------------------------------------------------- EB at cross ≠ 1


def test_eb_at_cross_below_one_maximises_the_chan_objective():
    data = _var_data(3, 120, 11)
    twin = _phase1(rb.Minnesota(var_const=10.0), data, 2, "c")
    prior = rb.Minnesota(cross=0.5, contemp=0.8, form="structural", var_const=10.0)
    result = eb_structural(data, prior, 2, "c", twin.hyper)
    assert result.converged and result.free == ("lambda",) and result.warnings == ()
    assert_allclose(result.hyper["psi"], twin.hyper["psi"], rtol=0.0)
    hp = prior.hyperpriors.lambda_

    def negative(z: float) -> float:
        lam = float(np.exp(z))
        return -(structural_log_ml(data, prior, {"lambda": lam}, 2, "c") + hp.logpdf(lam))

    ref = optimize.minimize_scalar(
        negative, bounds=(np.log(hp.lower), np.log(hp.upper)), options={"xatol": 1e-10}
    )
    assert result.hyper["lambda"] == pytest.approx(np.exp(ref.x), rel=1e-5)
    assert result.log_posterior == pytest.approx(-ref.fun, abs=1e-8)
    assert abs(result.hyper["lambda"] - twin.hyper["lambda"]) > 1e-3  # cross matters


def test_eb_rejects_free_psi_and_passes_fixed_hyperparameters_through():
    data = _var_data(3, 80, 23)
    psi_free = rb.Minnesota(cross=0.5, form="structural", hyper=("lambda", "psi"))
    with pytest.raises(ValueError, match="psi"):
        eb_structural(data, psi_free, 2, "c", {})
    fixed = rb.Minnesota(lambda_=0.3, cross=0.5, form="structural", hierarchy="fixed")
    result = eb_structural(data, fixed, 2, "c", {"alpha": 1.5})
    assert result.free == () and result.converged
    assert (result.hyper["lambda"], result.hyper["alpha"]) == (0.3, 1.5)
    expected = structural_log_ml(data, fixed, {"alpha": 1.5}, 2, "c")
    assert result.log_ml == result.log_posterior == pytest.approx(expected, abs=1e-12)


def test_structural_log_ml_validates_its_inputs():
    data = _var_data(2, 60, 29)
    prior = rb.Minnesota(cross=0.5, form="structural", hierarchy="fixed")
    with pytest.raises(KeyError, match="unknown"):
        structural_log_ml(data, prior, {"tightness": 0.2}, 1, "c")
    with pytest.raises(ValueError, match="covid"):
        structural_log_ml(data, prior, {"covid": np.ones(4)}, 1, "c")
    with pytest.raises(ValueError, match="lambda"):
        structural_log_ml(data, prior, {"lambda": -0.1}, 1, "c")
    with pytest.raises(TypeError, match="alpha"):
        structural_log_ml(data, prior, {"alpha": "2"}, 1, "c")
    no_persistence = rb.Minnesota(mean=0.0, soc=1.0, form="structural", hierarchy="fixed")
    with pytest.raises(ValueError, match="SOC"):
        structural_log_ml(data, no_persistence, {}, 1, "c")
    with pytest.raises(TypeError, match="VARData"):
        structural_log_ml(data.values, prior, {}, 1, "c")
    with pytest.raises(TypeError, match="Minnesota"):
        structural_log_ml(data, None, {}, 1, "c")


def test_prior_arrays_equal_the_wp6_builders():
    """The NumPy-built Mode II arrays are the WP6 ``structural_arrays(mode="II")``,
    ``contemp_arrays(scaling="lambda")`` and ``constant_prior`` (which carry JAX pytree classes)."""
    pytest.importorskip("jax")
    from rai_bvar.engines.structural_nig import _Window
    from rai_bvar.priors.structural import contemp_arrays, structural_arrays
    from rai_bvar.priors.sv_arrays import constant_prior

    data = _var_data(4, 80, 31)
    prior = rb.Minnesota(
        cross=0.4,
        contemp=0.7,
        var_const=50.0,
        mean={"v0": 1.0, "v1": 0.0, "v2": 0.6, "v3": 0.9},
        form="structural",
        hierarchy="fixed",
    )
    lags, det = 3, "ct"
    window = _Window(data, prior, lags, det)
    h = window.complete({"lambda": 0.35, "alpha": 1.6})
    coef, shared = structural_arrays(
        window.scales, prior, h, n=4, lags=lags, n_det=window.n_det, mode="II"
    )
    contemp = contemp_arrays(window.scales, prior, 4, scaling="lambda")
    vol = constant_prior(window.scales, 4)
    assert_allclose(window.Mc, np.asarray(shared.Mc), rtol=0.0)
    for i, (_, _, _, _, mu0, var0, nu0, S0) in enumerate(window.equations(h)):
        k = coef.mean0.shape[1]
        assert_allclose(mu0[:k], np.asarray(coef.mean0[i]), rtol=0.0)
        assert_allclose(mu0[k:], np.asarray(contemp.mean0[i, :i]), rtol=0.0)
        assert_allclose(1.0 / var0[:k], np.asarray(coef.prec0[i]), rtol=1e-15)
        assert_allclose(1.0 / var0[k:], np.asarray(contemp.prec0[i, :i]), rtol=1e-15)
        assert (nu0, S0) == (float(vol.shape[i]), float(vol.scale[i]))


def test_module_never_loads_jax():
    """NumPy only: importing the module and calling every public function leaves JAX unloaded
    (and its global ``jax_enable_x64`` flag untouched)."""
    code = (
        "import sys\n"
        "import numpy as np, pandas as pd\n"
        "import rai_bvar as rb\n"
        "from rai_bvar.engines import structural_nig as s\n"
        "assert 'jax' not in sys.modules, 'importing structural_nig loaded JAX'\n"
        "rng = np.random.default_rng(0)\n"
        "idx = pd.period_range('1980Q1', periods=60, freq='Q')\n"
        "d = rb.VARData.from_frame(pd.DataFrame(rng.standard_normal((60, 3)), index=idx,\n"
        "                                       columns=list('abc')))\n"
        "p = rb.Minnesota(cross=0.5, form='structural', soc=1.0, sur=1.0)\n"
        "s.structural_log_ml(d, p, {}, 2, 'c')\n"
        "posts = s.structural_posteriors(d, p, {}, 2, 'c')\n"
        "s.structural_to_reduced(*s.draw_structural_nig(posts, 3, rng))\n"
        "s.eb_structural(d, p, 2, 'c', {})\n"
        "assert 'jax' not in sys.modules, 'a structural_nig call loaded JAX'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
