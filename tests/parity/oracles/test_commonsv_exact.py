"""The exact ``CommonSV`` oracle's own validation (docs/phase3/DESIGN.md §9.8 "Its own
validation" (a)–(d); WP38a) and the reads of its fixtures.

**(a)** Geweke joint-distribution test of the whole sweep (weekly; P2 §9.5's machinery at the
G-small sizes ``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``): ``n = 2, p = 1, T_eff = 20`` with one
SUR row (``δ = 1``, presample row ``(0.5, −0.5)``), ``λ = 0.2``, ``α = 2``, ``ψ = (1, 1)``, own-lag
means 0.5, ``var_const = 1``, CommonSV's defaults; bounded test functions (E148).

**(b)** Block 1: its posterior parameters against the closed form computed here (an augmented
least-squares problem, 1e-10); 10⁵ draws against the matrix-normal–IW moments (``|z| ≤ max(4,
z_{1−0.01/(2m)})``, Bonferroni over the ``m`` moments, D-60); and the non-circular cross-check:
at ``h ≡ 0`` and at one fixed non-zero ``h`` path (data rows rescaled by ``e^{−h_t/2}``, dummy rows
unscaled) block 1's ``(B̄, Ω̄, Ψ̄, d̄)`` equal Phase 1's conjugate NIW posterior
(``rai_bvar.engines.analytic.ConjugateProblem``) to 1e-10 on C1's and C2's prior settings. A
1e-10 comparison is ``max |a − b| ≤ 1e-10 · max |b|`` over the array.

**(c)** Block 2 for ``T ∈ {2, 3}``, ``n = 2``, ``ρ = 0.9``, ``σ_h = 0.3``, ``q = (1.5, 0.4, 3.0)``
(first ``T`` entries): the ``h`` posterior by tensor quadrature against its draws, KS per
marginal, family-wise ``α = 1e-3`` over the ``m = 5`` tests (D-60).

**(d)** Block 3 on one ``T = 20`` path drawn from the AR(1) at ``ρ = 0.9, σ_h = 0.3`` (seed
:data:`D_SEED`), ``n = 2``, ``q_t = 2``: ``p(ρ | h, σ_h = 0.3)`` and the non-centred
``p(log σ_h | x = h/0.3, q, ρ = 0.9)`` by quadrature against their draws (KS, ``m = 2``).

**Predictive.** :func:`commonsv_exact.predictive` against the model's 1- and 2-step moments at
one fixed draw.

The quadrature densities are written here from the model (SciPy's normal and inverse-gamma
densities, the AR(1) covariance in closed form), not taken from the oracle. The draws of (c) and
(d) are the final states of :data:`KS_DRAWS` independent chains, each started from a prior draw
and run :data:`BLOCK2_CALLS` (c) or :data:`SLICE_UPDATES` (d) updates of the block under test, so
the KS tests see independent draws. Tiers: (b)–(d), the predictive check and the fixture reads in
the PR tier; (a) weekly (``slow``).

**Data.** C1's data ``qd3_growth.csv`` is WP29's (PR #21); the tests that need it (the C1 cases
of (b) and the C1 fixture reads) fail with that explanation while it is missing.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import cache
from pathlib import Path
from statistics import NormalDist
from types import ModuleType

import commonsv_exact as cx  # tests/parity/oracles is on sys.path (pytest's rootdir import)
import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "parity" / "fixtures"
TOOL = ROOT / "tools" / "make_commonsv_oracle.py"
PROBS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

TOL = 1e-10  # (b): posterior parameters
KS_ALPHA = 1e-3  # (c), (d): family-wise (D-60)
MOMENT_DRAWS = 100_000  # (b): draws for the moment comparison
KS_DRAWS = 50_000  # (c), (d): independent chains, one draw each
BLOCK2_CALLS = 10  # (c): calls of block 2 (5 elliptical slice updates each) per chain
SLICE_UPDATES = 30  # (d): slice updates per chain (from a prior draw: the bias decays by ≈ 0.5
#                     per update, so it is below 1e-8 of a posterior s.d. after 30)
D_SEED = 20261031  # (d): the T = 20 path


@cache
def tool() -> ModuleType:
    """``tools/make_commonsv_oracle.py`` (the case registry: data, lags, prior numbers)."""
    name = "_rai_bvar_make_commonsv_oracle"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case(name: str):
    c = tool().CASES[name]
    path = tool().DATA_DIR / c.data_file
    if not path.exists():
        owner = (
            "WP29's (PR #21, merged before WP38a; DESIGN §9.8, D-57)"
            if name == "C1"
            else "written by tools/make_commonsv_oracle.py data"
        )
        pytest.fail(f"{path.relative_to(ROOT)} is missing: {name}'s data is {owner}")
    return c


def assert_close(actual, expected, what: str, tol: float = TOL) -> float:
    """``max |a − b| ≤ tol · max |b|`` over the array; returns the relative error."""
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    err = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
    assert err <= tol, f"{what}: relative error {err:.3e} > {tol:g}"
    return err


def fixed_path(T: int) -> np.ndarray:
    """(b)'s fixed non-zero ``h`` path: ``0.8 sin(0.15t) + 0.4 cos(0.37t)``, ``t = 0..T−1``."""
    t = np.arange(T, dtype=np.float64)
    return 0.8 * np.sin(0.15 * t) + 0.4 * np.cos(0.37 * t)


def z_bound(m: int) -> float:
    """``max(4, z_{1 − 0.01/(2m)})`` (the Phase 2 rule's bound, D-60's moment rule)."""
    return max(4.0, NormalDist().inv_cdf(1.0 - 0.01 / (2.0 * m)))


# ------------------------------------------------------------------------------------------------
# (b) block 1
# ------------------------------------------------------------------------------------------------

B_SETTINGS = [(c, h) for c in ("C1", "C2") for h in ("zero", "path")]


def weights(problem, kind: str) -> np.ndarray:
    return np.ones(problem.T) if kind == "zero" else np.exp(-fixed_path(problem.T))


def closed_form(problem, w: np.ndarray) -> dict[str, np.ndarray | float]:
    """Block 1's posterior written as a least-squares problem: the Minnesota rows
    ``Ω^{−1/2}(B − b)``, the data rows scaled by ``√w_t`` and the dummy rows, solved by QR."""
    sw = np.sqrt(w)[:, None]
    root = 1.0 / np.sqrt(problem.omega)
    A = np.vstack([np.diag(root), sw * problem.X, problem.Xd])
    Z = np.vstack([root[:, None] * problem.b, sw * problem.Y, problem.Yd])
    Q, R = np.linalg.qr(A)
    Bbar = linalg.solve_triangular(R, Q.T @ Z)
    resid = Z - A @ Bbar
    Rinv = linalg.solve_triangular(R, np.eye(R.shape[0]))
    return {
        "mean": Bbar,
        "omega": Rinv @ Rinv.T,
        "scale": np.diag(problem.prior.psi) + resid.T @ resid,
        "df": problem.prior.d + problem.T + problem.Yd.shape[0],
    }


@pytest.mark.parametrize(("name", "h"), B_SETTINGS)
def test_block1_posterior_parameters_equal_the_closed_form(name, h):
    """(b): ``B̄, Ω̄, Ψ̄, d̄`` of the oracle against the augmented least-squares solution (1e-10)."""
    problem = case(name).problem()
    w = weights(problem, h)
    post = cx.niw_posterior(problem, w)
    ref = closed_form(problem, w)
    assert_close(post.mean, ref["mean"], "B̄")
    assert_close(post.omega, ref["omega"], "Ω̄")
    assert_close(post.scale, ref["scale"], "Ψ̄")
    assert post.df == ref["df"] == problem.prior.d + problem.T + problem.m


def phase1_posterior(c, problem, h: str):
    """Phase 1's conjugate NIW posterior on the case's data and prior settings; for the fixed
    path the data rows are divided by ``s_t = e^{h_t/2}`` through ``row_scales`` (Phase 1's GLS
    rescaling of the data rows, SPEC §4.6), the dummy rows unscaled."""
    from rai_bvar.data import VARData
    from rai_bvar.engines.analytic import ConjugateProblem
    from rai_bvar.priors.minnesota import Minnesota

    raw = c.frame()
    frame = raw.drop(columns="date").set_index(pd.PeriodIndex(raw["date"], freq="Q"))
    prior = problem.prior
    minnesota = Minnesota(
        lambda_=prior.lam,
        alpha=prior.alpha,
        psi=tuple(prior.psi),
        soc=prior.soc,
        sur=prior.sur,
        soc_scope="persistent",
        mean=c.mean,
        var_const=prior.var_const,
        form="conjugate",
        hierarchy="fixed",
    )
    conj = ConjugateProblem(VARData.from_frame(frame), minnesota, c.lags, "c", None)
    hyper = conj.complete(None)
    np.testing.assert_array_equal(conj.row_scales(hyper), np.ones(problem.T))
    if h == "path":
        scales = np.exp(0.5 * fixed_path(problem.T))
        conj.row_scales = lambda _hyper: scales  # the instance's rows scales, s_t = e^{h_t/2}
    return conj, hyper, conj.posterior(hyper)


@pytest.mark.parametrize(("name", "h"), B_SETTINGS)
def test_block1_equals_phase1_conjugate_posterior(name, h):
    """(b), the non-circular cross-check: block 1's ``(B̄, Ω̄, Ψ̄, d̄)`` equal Phase 1's analytic
    engine's conjugate posterior to 1e-10 (and its ``Ω``, ``ψ`` and dummy rows equal ours)."""
    c = case(name)
    problem = c.problem()
    conj, hyper, ref = phase1_posterior(c, problem, h)
    np.testing.assert_allclose(np.asarray(hyper["psi"]), problem.prior.psi, rtol=1e-15)
    omega = np.asarray(conj.omega(hyper))
    assert_close(np.diag(omega) if omega.ndim == 2 else omega, problem.omega, "prior Ω")
    Yd, Xd = conj.dummies(hyper)
    assert Yd.shape == problem.Yd.shape and Xd.shape == problem.Xd.shape
    if problem.m:
        assert_close(Yd, problem.Yd, "dummy Y")
        assert_close(Xd, problem.Xd, "dummy X")
    post = cx.niw_posterior(problem, weights(problem, h))
    assert_close(post.mean, ref.mean, "B̄")
    assert_close(post.omega, ref.omega(), "Ω̄")
    assert_close(post.scale, ref.scale, "Ψ̄")
    assert post.df == float(ref.df)


@pytest.mark.parametrize("name", ["C1", "C2"])
def test_block1_draws_have_the_matrix_normal_inverse_wishart_moments(name):
    """(b): 10⁵ draws of block 1 at the fixed path against ``E[B] = B̄``, ``Var(B_ij) = Ω̄_ii
    Ψ̄_jj/(d̄ − n − 1)``, ``E[Σ] = Ψ̄/(d̄ − n − 1)`` and the inverse-Wishart ``Var(Σ_ij)``; the
    sample variances' standard errors from the draws' fourth moments."""
    problem = case(name).problem()
    post = cx.niw_posterior(problem, weights(problem, "path"))
    B, Sigma = cx.draw_niw(post, np.random.default_rng(20261032), size=MOMENT_DRAWS)
    n, nu, Psi = problem.n, post.df, post.scale
    rows, cols = np.tril_indices(n)
    S = Sigma[:, rows, cols]
    ES = (Psi / (nu - n - 1))[rows, cols]
    VS = ((nu - n + 1) * Psi**2 + (nu - n - 1) * np.outer(np.diag(Psi), np.diag(Psi))) / (
        (nu - n) * (nu - n - 1) ** 2 * (nu - n - 3)
    )
    VS = VS[rows, cols]
    EB = post.mean
    VB = np.diag(post.omega)[:, None] * np.diag(Psi)[None, :] / (nu - n - 1)
    N = MOMENT_DRAWS

    def var_z(x: np.ndarray, var: np.ndarray) -> np.ndarray:
        dev = x - x.mean(axis=0)
        s2 = (dev**2).sum(axis=0) / (N - 1)
        m4 = (dev**4).mean(axis=0)
        return (s2 - var) / np.sqrt((m4 - s2**2) / N)

    z = np.concatenate(
        [
            ((B.mean(axis=0) - EB) / np.sqrt(VB / N)).ravel(),
            (S.mean(axis=0) - ES) / np.sqrt(VS / N),
            var_z(B, VB).ravel(),
            var_z(S, VS),
        ]
    )
    bound = z_bound(z.size)
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) <= bound, f"max |z| = {np.max(np.abs(z)):.2f} > {bound:.2f}"


# ------------------------------------------------------------------------------------------------
# (c) block 2
# ------------------------------------------------------------------------------------------------

C_RHO, C_SIGMA, C_N = 0.9, 0.3, 2
C_Q = np.array([1.5, 0.4, 3.0])
C_GRID = {2: (1201, -6.0, 5.0), 3: (241, -6.0, 5.0)}  # points per dimension, range


def ar1_covariance(T: int, rho: float, sigma: float) -> np.ndarray:
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    return sigma**2 * rho**lag / (1.0 - rho**2)


def trapezoid_cdf(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    steps = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(steps)])
    return cdf / cdf[-1]


def h_posterior_marginal_cdfs(T: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Marginal CDFs of ``p(h | q) ∝ N(h; 0, Σ_AR) Π_t exp(−(n/2)h_t − ½q_t e^{−h_t})`` on a
    tensor grid (trapezoid weights)."""
    G, lo, hi = C_GRID[T]
    grid = np.linspace(lo, hi, G)
    prec = np.linalg.inv(ar1_covariance(T, C_RHO, C_SIGMA))
    axes = np.meshgrid(*([grid] * T), indexing="ij", sparse=True)
    logp = np.zeros((G,) * T)
    for i in range(T):
        logp = logp - 0.5 * C_N * axes[i] - 0.5 * C_Q[i] * np.exp(-axes[i])
        for j in range(T):
            logp = logp - 0.5 * prec[i, j] * axes[i] * axes[j]
    dens = np.exp(logp - logp.max())
    edge = max(np.take(dens, idx, axis=ax).max() for ax in range(T) for idx in (0, -1))
    assert edge < 1e-12, f"T = {T}: the grid truncates the posterior ({edge:.1e})"
    w = np.full(G, grid[1] - grid[0])
    w[[0, -1]] *= 0.5
    cdfs = []
    for t in range(T):
        marg = dens
        for ax in reversed(range(T)):
            if ax != t:
                marg = np.tensordot(marg, w, axes=([ax], [0]))
        cdfs.append(trapezoid_cdf(grid, marg))
    return grid, cdfs


def test_block2_h_posterior_matches_tensor_quadrature():
    """(c): the final states of ``KS_DRAWS`` chains of block 2 alone (5 elliptical slice updates
    per call, ``BLOCK2_CALLS`` calls from a stationary-prior draw) against the quadrature
    marginals; ``m = 5`` KS tests (``T = 2``: 2, ``T = 3``: 3), each must have ``p > 1e-3/5``."""
    rng = np.random.default_rng(20261033)
    p_values = {}
    for T in (2, 3):
        K = KS_DRAWS
        rho, sigma_h = np.full(K, C_RHO), np.full(K, C_SIGMA)
        h = cx.ar1_stationary(rho, sigma_h, rng.standard_normal((K, T)))
        logq = np.broadcast_to(np.log(C_Q[:T]), (K, T)).copy()
        for _ in range(BLOCK2_CALLS):
            h = cx.block2(h, logq, rho, sigma_h, C_N, rng)
        grid, cdfs = h_posterior_marginal_cdfs(T)
        for t in range(T):
            cdf = cdfs[t]
            p_values[f"T={T}, h_{t}"] = stats.kstest(
                h[:, t], lambda v, c=cdf, g=grid: np.interp(v, g, c)
            ).pvalue
    m = len(p_values)
    assert m == 5
    assert min(p_values.values()) > KS_ALPHA / m, p_values


# ------------------------------------------------------------------------------------------------
# (d) block 3
# ------------------------------------------------------------------------------------------------

D_RHO, D_SIGMA, D_N, D_T, D_Q = 0.9, 0.3, 2, 20, 2.0


def d_path() -> np.ndarray:
    """(d)'s path: ``h_0 = 0.3 z_0/√(1 − 0.81)``, ``h_t = 0.9h_{t−1} + 0.3z_t`` with
    ``z ~ default_rng(D_SEED).standard_normal(20)``."""
    z = np.random.default_rng(D_SEED).standard_normal(D_T)
    h = np.empty(D_T)
    h[0] = D_SIGMA / math.sqrt(1.0 - D_RHO**2) * z[0]
    for t in range(1, D_T):
        h[t] = D_RHO * h[t - 1] + D_SIGMA * z[t]
    return h


def rho_log_density(rho: np.ndarray, h: np.ndarray, sigma: float) -> np.ndarray:
    """``log N(ρ; 0.9, 0.2²) + log N(h_0; 0, σ²/(1 − ρ²)) + Σ_t log N(h_t; ρh_{t−1}, σ²)``."""
    out = stats.norm.logpdf(rho, 0.9, 0.2)
    out = out + stats.norm.logpdf(h[0], 0.0, sigma / np.sqrt(1.0 - rho**2))
    for t in range(1, h.size):
        out = out + stats.norm.logpdf(h[t], rho * h[t - 1], sigma)
    return out


def log_sigma_log_density(s: np.ndarray, x: np.ndarray, q: float, n: int) -> np.ndarray:
    """``log IG(e^{2s}; 5, 0.04) + log(2e^{2s}) + Σ_t[−(n/2)h_t − ½q e^{−h_t}]``, ``h = e^{s}x``."""
    out = stats.invgamma.logpdf(np.exp(2.0 * s), 5.0, scale=0.04) + np.log(2.0) + 2.0 * s
    h = np.exp(s)[:, None] * x[None, :]
    return out + np.sum(-0.5 * n * h - 0.5 * q * np.exp(-h), axis=1)


def grid_cdf(
    grid: np.ndarray, logd: np.ndarray, what: str, open_support: bool = True
) -> np.ndarray:
    """CDF on ``grid``; on an open support the grid must not cut off mass (the density at both
    ends below 1e-12 of its maximum). ``ρ``'s grid spans ``(−1, 1)`` up to 1e-9 from each end,
    where the density vanishes like ``√(1 − ρ²)``, so the mass outside is below 1e-13."""
    dens = np.exp(logd - logd.max())
    if open_support:
        assert max(dens[0], dens[-1]) < 1e-12, f"{what}: the grid truncates the density"
    return trapezoid_cdf(grid, dens)


def test_block3_rho_and_noncentred_log_sigma_match_quadrature():
    """(d): ``ρ`` draws of the slice sampler (``σ_h = 0.3`` and ``h`` fixed) and ``log σ_h``
    draws of the non-centred move (``x = h/0.3``, ``q_t = 2``, ``ρ = 0.9``) against their
    densities by quadrature; ``m = 2`` KS tests, each ``p > 1e-3/2``."""
    prior = cx.Prior(psi=(1.0, 1.0), means=(0.0, 0.0))  # CommonSV's defaults; ψ, means unused
    h = d_path()
    K = KS_DRAWS
    rng = np.random.default_rng(20261034)
    hk = np.broadcast_to(h, (K, D_T))
    sigma = np.full(K, D_SIGMA)
    rho = cx.draw_rho_prior(prior, rng, K)
    for _ in range(SLICE_UPDATES):
        rho = cx.update_rho(hk, rho, sigma, prior, rng)
    x = np.broadcast_to(h / D_SIGMA, (K, D_T))
    logq = np.full((K, D_T), math.log(D_Q))
    s = 0.5 * np.log(prior.sigma2_scale / rng.gamma(prior.sigma2_shape, size=K))
    for _ in range(SLICE_UPDATES):
        s = cx.update_log_sigma_noncentred(x, s, logq, D_N, prior, rng)

    rho_grid = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, 400_001)
    rho_cdf = grid_cdf(rho_grid, rho_log_density(rho_grid, h, D_SIGMA), "rho", open_support=False)
    s_grid = np.linspace(-6.0, 2.0, 400_001)
    s_cdf = grid_cdf(s_grid, log_sigma_log_density(s_grid, h / D_SIGMA, D_Q, D_N), "log sigma")
    p_values = {
        "rho": stats.kstest(rho, lambda v: np.interp(v, rho_grid, rho_cdf)).pvalue,
        "log_sigma_h": stats.kstest(s, lambda v: np.interp(v, s_grid, s_cdf)).pvalue,
    }
    m = len(p_values)
    assert min(p_values.values()) > KS_ALPHA / m, p_values


def test_block3_centred_sigma2_draw_is_the_inverse_gamma_conditional():
    """Block 3's centred step on (d)'s path at ``ρ = 0.9`` against SciPy's
    ``IG(5 + T/2, 0.04 + S/2)`` (``S`` computed here; KS, ``m = 1``)."""
    prior = cx.Prior(psi=(1.0, 1.0), means=(0.0, 0.0))
    h = d_path()
    S = (1.0 - D_RHO**2) * h[0] ** 2 + np.sum((h[1:] - D_RHO * h[:-1]) ** 2)
    hk = np.broadcast_to(h, (KS_DRAWS, D_T))
    draws = cx.draw_sigma2(hk, np.full(KS_DRAWS, D_RHO), prior, np.random.default_rng(20261035))
    ref = stats.invgamma(5.0 + D_T / 2.0, scale=0.04 + S / 2.0)
    assert stats.kstest(draws, ref.cdf).pvalue > KS_ALPHA


# ------------------------------------------------------------------------------------------------
# The predictive simulation (§9.8 Runs)
# ------------------------------------------------------------------------------------------------


def test_predictive_paths_have_the_model_moments():
    """``predictive`` at one fixed parameter draw on C2's problem (``p = 2``) against the model:
    ``E[y_T] = Bᵀx_T`` and ``Cov(y_T) = E[e^{h_T}]Σ`` with ``h_T ~ N(ρh_{T−1}, σ_h²)`` (the
    regressors from the last ``p`` data rows), and ``E[y_{T+1}] = c + A₁E[y_T] + A₂y_{T−1}``
    (``|z| ≤ max(4, z_{1−0.01/(2m)})``)."""
    c = tool().CASES["C2"]
    problem = c.problem()
    n, S = problem.n, 400_000
    post = cx.niw_posterior(problem, np.ones(problem.T))
    B1, Sigma1 = cx.draw_niw(post, np.random.default_rng(20261037), size=1)
    rho, sigma_h, h_last = 0.8, 0.3, 0.4
    B = np.broadcast_to(B1[0], (S, *B1.shape[1:]))
    Sigma = np.broadcast_to(Sigma1[0], (S, n, n))
    paths = cx.predictive(problem, B, Sigma, np.full(S, h_last), np.full(S, rho),
                          np.full(S, sigma_h), np.random.default_rng(20261038))  # fmt: skip
    assert paths.shape == (S, 4, n)
    y_last, y_prev = problem.Y[-1], problem.Y[-2]
    x_T = np.concatenate([[1.0], y_last, y_prev])
    m1 = x_T @ B1[0]
    scale1 = math.exp(rho * h_last + 0.5 * sigma_h**2)
    cov1 = scale1 * Sigma1[0]
    const, A1, A2 = B1[0][0], B1[0][1 : n + 1].T, B1[0][n + 1 :].T
    m2 = const + A1 @ m1 + A2 @ y_last
    y1, y2 = paths[:, 0], paths[:, 1]
    z = [(y1.mean(axis=0) - m1) / np.sqrt(np.diag(cov1) / S)]
    dev = y1 - m1
    emp = dev.T @ dev / S
    i, j = np.triu_indices(n)
    var_prod = (dev[:, i] * dev[:, j]).var(axis=0)
    z.append((emp[i, j] - cov1[i, j]) / np.sqrt(var_prod / S))
    z.append((y2.mean(axis=0) - m2) / (y2.std(axis=0) / math.sqrt(S)))
    z = np.concatenate(z)
    bound = z_bound(z.size)
    assert np.max(np.abs(z)) <= bound, (np.round(z, 2), bound)


# ------------------------------------------------------------------------------------------------
# (a) Geweke joint-distribution test (weekly)
# ------------------------------------------------------------------------------------------------

G_SMALL = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}
GS_STAGES, GS_DELTA = 3, 4  # Gandy–Scott sequential wrapper of P2 §9.5 (k = 3, Δ = 4)
G_T, G_DATES = 20, (0, 10, 19)  # T_eff and the three dates of h (0, ⌊T/2⌋, T − 1)
G_SEED = 20261036


def geweke_problem():
    prior = cx.Prior(psi=(1.0, 1.0), means=(0.5, 0.5), lam=0.2, alpha=2.0, var_const=1.0, sur=1.0)
    data = np.vstack([[0.5, -0.5], np.zeros((G_T, 2))])  # the presample row; data are simulated
    return cx.build_problem(data, 1, prior)


def geweke_functions(state) -> tuple[list[str], np.ndarray]:
    """The bounded test functions (E148): ``atan`` of every ``B`` entry, of ``log Σ_ii``, of
    ``h`` at three dates and of mean ``h``; ``ρ`` and ``σ_h``."""
    K, k, n = state.B.shape
    names = [f"atan B[{i},{j}]" for i in range(k) for j in range(n)]
    names += [f"atan log Sigma[{i},{i}]" for i in range(n)]
    names += [f"atan h_{t}" for t in G_DATES] + ["atan mean h", "rho", "sigma_h"]
    g = np.column_stack(
        [
            np.arctan(state.B.reshape(K, k * n)),
            np.arctan(np.log(np.diagonal(state.Sigma, axis1=1, axis2=2))),
            np.arctan(state.h[:, list(G_DATES)]),
            np.arctan(state.h.mean(axis=1)),
            state.rho,
            state.sigma_h,
        ]
    )
    return names, g


def geweke_stage(problem, C: int, M_c: int, M1: int, seed: int) -> dict:
    """Marginal-conditional draws (prior; the test functions involve the parameters only) and
    ``C`` successive-conditional chains of ``M_c`` steps (data redrawn, then one sweep), each
    started at a prior draw; ``z_g = (ḡ_MC − ḡ_SC)/√(σ̂²_g/M₁ + s²_C/C)``, ``t_{C−1}`` p-values."""
    rng_mc = np.random.default_rng([seed, 0])
    rng_sc = np.random.default_rng([seed, 1])
    names, g_mc = geweke_functions(cx.draw_prior(problem, rng_mc, M1))
    state = cx.draw_prior(problem, rng_sc, C)
    sums = np.zeros((C, len(names)))
    for _ in range(M_c):
        Y, X = cx.simulate_data(problem.presample, state, rng_sc)
        state = cx.sweep(state, problem.with_data(Y, X), rng_sc)
        sums += geweke_functions(state)[1]
    chain_means = sums / M_c
    se = np.sqrt(g_mc.var(axis=0) / M1 + chain_means.var(axis=0, ddof=1) / C)
    z = (g_mc.mean(axis=0) - chain_means.mean(axis=0)) / se
    p = 2.0 * stats.t.sf(np.abs(z), C - 1)
    return {"names": names, "z": z, "p": p, "C": C, "M_c": M_c, "M1": M1, "seed": seed}


def record(kind: str, payload: dict) -> None:
    """One JSON line in ``$RAI_BVAR_VALIDATION_RECORD/<kind>.jsonl`` when the variable is set."""
    folder = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with (Path(folder) / f"{kind}.jsonl").open("a") as fh:
            fh.write(json.dumps(payload) + "\n")


@pytest.mark.slow
@pytest.mark.weekly
def test_geweke_joint_distribution_of_the_oracle_sweep():
    """(a): G-small with Bonferroni over the ``G = 14`` test functions inside the Gandy–Scott
    sequential wrapper (``α = 1e-3``, ``k = 3``, ``Δ = 4``: ``β₁ = α/k``, ``γ = β₁^{1/k}``; stage
    ``i`` fails if ``q_i = G·min p ≤ β_i``, passes if ``q_i > γ + β_i`` (the last stage: if
    ``q_k > β_k``), else ``β_{i+1} = β_i/γ``; stages 2–3 have ``Δ``× the chains and prior draws
    and fresh seeds)."""
    problem = geweke_problem()
    alpha = G_SMALL["alpha"]
    beta = alpha / GS_STAGES
    gamma = beta ** (1.0 / GS_STAGES)
    stages = []
    passed = False
    for i in range(GS_STAGES):
        scale = 1 if i == 0 else GS_DELTA
        res = geweke_stage(
            problem, scale * G_SMALL["C"], G_SMALL["M_c"], scale * G_SMALL["M1"], G_SEED + i
        )
        G = len(res["names"])
        qi = G * float(res["p"].min())
        stages.append(
            {**res, "q": qi, "beta": beta, "z": res["z"].tolist(), "p": res["p"].tolist()}
        )
        last = i == GS_STAGES - 1
        if qi <= beta:
            break
        if qi > (beta if last else gamma + beta):
            passed = True
            break
        beta = beta / gamma
    record(
        "commonsv_oracle_geweke",
        {
            "test": "WP38a (a)",
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
            "passed": passed,
            "stages": stages,
        },
    )
    summary = "; ".join(
        f"stage {j + 1}: max |z| {np.max(np.abs(s['z'])):.2f}, q {s['q']:.3g}, beta {s['beta']:.3g}"
        for j, s in enumerate(stages)
    )
    assert passed, summary


# ------------------------------------------------------------------------------------------------
# Fixtures (the runs of §9.8, written by tools/make_commonsv_oracle.py)
# ------------------------------------------------------------------------------------------------

FIXTURE_CASES = ("C1", "C2")
SUFFIXES = ("_q", "_mcse", "_mcse_vehtari", "_mcse_batch", "_mean", "_sd", "_ess", "_ess_tail",
            "_mcse_mean", "_rhat", "_ess_run", "_mixing_ok")  # fmt: skip
PREFIXES = ("B", "Sigma", "h", "sv", "forecast")


@cache
def fixture(name: str):
    import xarray as xr

    c = case(name)
    path = FIXTURES / f"{c.fixture}.nc"
    if not path.exists():
        pytest.fail(f"{path.relative_to(ROOT)} is missing (tools/make_commonsv_oracle.py all)")
    return xr.load_dataset(path, engine="scipy")


def test_c2_data_regenerate_from_the_stated_process():
    """The committed C2 CSV equals the tool's simulation from ``default_rng(20261026)``."""
    t = tool()
    committed = pd.read_csv(t.DATA_DIR / "commonsv_n3.csv", float_precision="round_trip")
    fresh, h = t.simulate_c2()
    assert list(committed.columns) == ["date", "y1", "y2", "y3"]
    assert len(committed) == 202 and h.shape == (200,)
    np.testing.assert_array_equal(committed["date"], fresh["date"])
    np.testing.assert_array_equal(committed.drop(columns="date"), fresh.drop(columns="date"))
    assert t.C2_DGP["seed"] == 20261026


@pytest.mark.parametrize("name", FIXTURE_CASES)
def test_fixture_provenance_and_prior_numbers(name):
    c = case(name)
    ds = fixture(name)
    a = ds.attrs
    assert a["kind"] == "commonsv_oracle" and a["case"] == name and int(a["quick"]) == 0
    assert a["data_md5"] == hashlib.md5((FIXTURES / a["data_file"]).read_bytes()).hexdigest()
    assert (int(a["chains"]), int(a["draws"]), int(a["burnin"])) == (4, 200_000, 10_000)
    np.testing.assert_array_equal(a["seeds"], c.seeds)
    np.testing.assert_array_equal(a["predictive_seeds"], c.predictive_seeds)
    assert len(set(c.seeds) | set(c.predictive_seeds)) == 8
    for key in ("git_commit", "numpy_version", "scipy_version", "cpu"):
        assert str(a[key]).strip(), key
    # the runs used this oracle (a change to it needs new fixtures) on a clean tracked tree
    assert a["oracle_md5"] == hashlib.md5((ROOT / a["oracle_file"]).read_bytes()).hexdigest()
    assert int(a["git_dirty"]) == 0
    problem = c.problem()
    prior = problem.prior
    assert (float(a["prior_lambda"]), float(a["prior_alpha"])) == (0.2, 2.0)
    assert float(a["prior_var_const"]) == 1e7 and int(a["prior_d"]) == problem.n + 2
    np.testing.assert_array_equal(a["prior_psi"], prior.psi)
    np.testing.assert_array_equal(ds["prior_psi"].values, prior.psi)
    np.testing.assert_array_equal(a["prior_own_lag_means"], prior.means)
    np.testing.assert_array_equal(ds["prior_omega"].values, problem.omega)
    np.testing.assert_array_equal(ds["prior_b"].values, problem.b)
    assert (float(a["prior_rho_mean"]), float(a["prior_rho_sd"])) == (0.9, 0.2)
    assert (float(a["prior_sigma2_shape"]), float(a["prior_sigma2_scale"])) == (5.0, 0.04)
    assert (int(a["n"]), int(a["lags"]), int(a["T"]), int(a["m"])) == (
        problem.n, problem.lags, problem.T, problem.m)  # fmt: skip
    if name == "C1":
        assert a["data_file"] == "data/qd3_growth.csv" and int(a["m"]) == 0
        assert a["data_md5"] == "c313a9667f34a88df1ca4930c99424d2"
        assert int(a["prior_soc"]) == 0 == int(a["prior_sur"])
        np.testing.assert_array_equal(prior.means, 0.0)
    else:
        assert a["data_file"] == "data/commonsv_n3.csv" and int(a["m"]) == 4
        assert float(a["prior_soc_mu"]) == 1.0 == float(a["prior_sur_delta"])
        np.testing.assert_array_equal(prior.psi, 1.0)
        np.testing.assert_array_equal(prior.means, 0.5)
        np.testing.assert_array_equal(ds["dummy_y"].values, problem.Yd)
        np.testing.assert_array_equal(ds["dummy_x"].values, problem.Xd)
    assert int(a["ess_updates"]) == 5
    assert (float(a["rho_slice_width"]), int(a["rho_slice_max_steps"])) == (0.1, 50)
    assert (float(a["s_slice_width"]), int(a["s_slice_max_steps"])) == (0.5, 50)
    assert np.all(np.asarray(a["wall_time_seconds"]) > 0.0)


@pytest.mark.parametrize("name", FIXTURE_CASES)
def test_fixture_summaries_are_complete_and_the_gate_is_recorded(name):
    """Every compared quantity of §9.8 has its summary block; ``*_mixing_ok`` and the fixture's
    ``not_met`` list follow from ``*_rhat`` and ``*_ess`` (R-hat ≤ 1.01, ESS ≥ 400)."""
    ds = fixture(name)
    a = ds.attrs
    T, n, k = int(a["T"]), int(a["n"]), int(a["k"])
    np.testing.assert_array_equal(ds["quantile"].values, PROBS)
    np.testing.assert_array_equal(ds["t_sel"].values, [0, T // 4, T // 2, T - 1])
    np.testing.assert_array_equal(ds["horizon"].values, [1, 4])
    assert list(ds["sv_param"].values) == ["rho", "sigma_h"]
    assert ds["B_q"].shape == (k, n, 5) and ds["Sigma_q"].shape == (n * (n + 1) // 2, 5)
    assert ds["forecast_q"].shape == (2, n, 5)
    not_met = []
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            var = ds[f"{prefix}{suffix}"]
            if suffix != "_mixing_ok":
                assert np.all(np.isfinite(var.values)), f"{prefix}{suffix}"
        q = ds[f"{prefix}_q"].values
        assert np.all(np.diff(q, axis=-1) >= 0.0)
        np.testing.assert_array_equal(
            ds[f"{prefix}_mcse"].values,
            np.maximum(ds[f"{prefix}_mcse_vehtari"].values, ds[f"{prefix}_mcse_batch"].values),
        )
        assert np.all(ds[f"{prefix}_mcse"].values > 0.0)
        ok = (ds[f"{prefix}_rhat"].values <= 1.01) & (ds[f"{prefix}_ess"].values >= 400.0)
        np.testing.assert_array_equal(ds[f"{prefix}_mixing_ok"].values, ok.astype(np.int32))
        assert ds[f"{prefix}_ess_run"].dims[0] == "run" and ds.sizes["run"] == 4
        not_met += [1 for flag in ok.ravel() if not flag]
    assert int(a["n_not_met"]) == len(not_met)
    assert int(a["mixing_ok"]) == int(not not_met)
    assert len(str(a["gate_table"]).splitlines()) == sum(ds[f"{p}_rhat"].size for p in PREFIXES)


def test_c2_posterior_is_near_the_simulated_process():
    """Data and fixture belong together: C2's posterior medians lie near the DGP."""
    ds = fixture("C2")
    rho_med, sigma_med = ds["sv_q"].values[:, 2]
    assert 0.5 < rho_med < 1.0 and 0.03 < sigma_med < 0.5
    B_med = ds["B_q"].values[..., 2]
    np.testing.assert_allclose(np.diag(B_med[1:4]), 0.5, atol=0.2)  # A₁ = 0.5 I
    np.testing.assert_allclose(B_med[0], [0.2, 0.1, 0.3], atol=0.4)  # c
    r = np.corrcoef(ds["h_path_mean"].values, ds["dgp_h"].values)[0, 1]
    assert r > 0.3, r
