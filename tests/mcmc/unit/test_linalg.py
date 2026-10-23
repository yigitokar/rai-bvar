"""Linear-algebra kernels (DESIGN §13 WP4, §9.3 item 1; N05 §7; N07 §8, §13.1; N02 §5.5, §6.1).

Dense oracles with injected noise, exact padding, packed/tiled/batched Gram agreement, the two
Constant fast paths against the Cholesky path (moments to 1e-12 and KS on 10⁴ draws), the eigen
and Cholesky paths on Minnesota level data against a 45-digit solve (mean ``1e-12 + 5εκ(1 + t)``,
covariance ``1e-12 + 5εκ``, ``κ = cond(sKs)``, ``t`` the largest posterior t-ratio) with regression
pins on the graded decomposition of ``eigen_prepare``, the
tridiagonal sampler against dense ``Q``/``slogdet``, the stable 2×2 algebra against exact rationals
(N02 T7), Jacobi scaling at 1e8 precision disparities, failures as values, ``vmap`` = loop, and a
non-gating timing table at the reference shapes.
"""

from __future__ import annotations

import time
from decimal import Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from jax import random
from scipy import stats

from rai_bvar.engines.mcmc import linalg as LA
from rai_bvar.engines.mcmc.state import EigenCache, FixedFactorCache, StepInfo

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def spd(r: np.random.Generator, m: int, cond: float = 10.0) -> np.ndarray:
    """SPD matrix with eigenvalues log-spaced in [1, cond] and a random basis."""
    Qm, _ = np.linalg.qr(r.normal(size=(m, m)))
    return (Qm * np.geomspace(1.0, cond, m)) @ Qm.T


def dense_draw(K: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """NumPy oracle: ``K⁻¹b + chol(K)⁻ᵀz`` (equal to the Jacobi-scaled form in exact arithmetic)."""
    L = np.linalg.cholesky(K)
    return np.linalg.solve(K, b) + sla.solve_triangular(L.T, z, lower=False)


def j(*arrays):
    return tuple(jnp.asarray(a) for a in arrays)


# --------------------------------------------------------------------------- packed Gram


def test_packed_index_is_column_major():
    iu, ju, pos = LA._packed_index(4)
    pairs = list(zip(iu.tolist(), ju.tolist(), strict=True))
    assert pairs == [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
    for c, (r, s) in enumerate(pairs):
        assert pos[r, s] == pos[s, r] == c == s * (s + 1) // 2 + r
    assert iu.dtype == ju.dtype == pos.dtype == np.int32
    assert not pos.flags.writeable
    # prefix property (pilot §10): the first q'(q'+1)/2 columns are the leading q' × q' block
    iu7, ju7, _ = LA._packed_index(7)
    assert np.array_equal(iu7[:10], iu) and np.array_equal(ju7[:10], ju)


def test_packed_order_matches_data_module():
    data = pytest.importorskip("rai_bvar.engines.mcmc.data")
    if not hasattr(data, "packed_index"):
        pytest.skip("data.packed_index (WP5) not available yet")
    for q in (1, 2, 5, 13):
        for mine, theirs in zip(LA._packed_index(q), data.packed_index(q), strict=True):
            assert np.array_equal(mine, theirs)


def gram_problem(seed: int, T_pad: int, T_eff: int, q: int):
    r = rng(seed)
    F = r.normal(size=(T_pad, q)) * np.exp(r.normal(size=q))
    F[T_eff:] = 0.0
    w = np.exp(3.0 * r.normal(size=T_pad))  # weights over ~6 orders of magnitude
    w[T_eff:] = 0.0
    iu, ju, _ = LA._packed_index(q)
    return F, w, F[:, iu] * F[:, ju]


def assert_gram_close(G: np.ndarray, G_ref: np.ndarray, tol: float) -> None:
    """``|ΔG_rs| ≤ tol·√(G_rr G_ss)`` (Cauchy–Schwarz scale of a weighted Gram entry)."""
    d = np.sqrt(np.diag(G_ref))
    assert np.max(np.abs(G - G_ref) / np.outer(d, d)) <= tol


@pytest.mark.parametrize("q,T_pad,T_eff", [(1, 32, 20), (13, 64, 50), (100, 320, 300)])
def test_weighted_gram_modes_match_dense(q, T_pad, T_eff):
    F, w, FF = gram_problem(q, T_pad, T_eff, q)
    G_ref = F.T @ (w[:, None] * F)
    P = q * (q + 1) // 2
    out = {"packed": LA.weighted_gram(*j(w), FF=jnp.asarray(FF), mode="packed", q=q)}
    out["batched"] = LA.weighted_gram(*j(w), F=jnp.asarray(F), mode="batched", q=q)
    for tile in sorted({1, 7, 512, 2048, P, P + 5} - ({1} if P > 1000 else set())):
        out[f"tiled{tile}"] = LA.weighted_gram(
            *j(w), F=jnp.asarray(F), mode="packed_tiled", q=q, tile=tile
        )
    for G in out.values():
        G = np.asarray(G)
        assert G.dtype == np.float64 and G.shape == (q, q)
        assert np.array_equal(G, G.T)  # exactly symmetric
        assert_gram_close(G, G_ref, 1e-13)
    # tiling only changes the GEMM shape: the packed vectors agree to rounding
    assert_gram_close(np.asarray(out["tiled7"]), np.asarray(out["packed"]), 1e-14)


def test_padded_rows_do_not_enter_the_gram():
    _, w, FF = gram_problem(1, 64, 40, 9)
    junk_w = w.copy()
    junk_w[40:] = 1e300  # padded rows have F = 0 exactly, so any weight leaves G unchanged
    G0 = LA.weighted_gram(*j(w), FF=jnp.asarray(FF), mode="packed", q=9)
    G1 = LA.weighted_gram(*j(junk_w), FF=jnp.asarray(FF), mode="packed", q=9)
    assert np.array_equal(np.asarray(G0), np.asarray(G1))


def test_gram_from_packed_and_argument_errors():
    q = 5
    iu, ju, _ = LA._packed_index(q)
    M = rng(2).normal(size=(q, q))
    M = M + M.T
    v = M[iu, ju]
    assert np.array_equal(np.asarray(LA.gram_from_packed(jnp.asarray(v), q)), M)
    F, w, FF = gram_problem(3, 32, 30, q)
    with pytest.raises(ValueError, match="length"):
        LA.gram_from_packed(jnp.asarray(v[:-1]), q)
    with pytest.raises(ValueError, match="needs the packed products"):
        LA.weighted_gram(jnp.asarray(w), F=jnp.asarray(F), mode="packed", q=q)
    with pytest.raises(ValueError, match="packed columns"):
        LA.weighted_gram(jnp.asarray(w), FF=jnp.asarray(FF[:, :-1]), mode="packed", q=q)
    with pytest.raises(ValueError, match="tile > 0"):
        LA.weighted_gram(jnp.asarray(w), F=jnp.asarray(F), mode="packed_tiled", q=q)
    with pytest.raises(ValueError, match="needs the design"):
        LA.weighted_gram(jnp.asarray(w), FF=jnp.asarray(FF), mode="batched", q=q)
    with pytest.raises(ValueError, match="columns"):
        LA.weighted_gram(jnp.asarray(w), F=jnp.asarray(F), mode="batched", q=q + 1)
    with pytest.raises(ValueError, match="unknown Gram mode"):
        bad_mode: Any = "syrk"
        LA.weighted_gram(jnp.asarray(w), F=jnp.asarray(F), mode=bad_mode, q=q)


# --------------------------------------------------------------------------- precision draw (D-14)


@pytest.mark.parametrize("m", [1, 2, 5, 30])
def test_precision_draw_matches_dense_oracle(m):
    r = rng(10 + m)
    S = np.exp(r.normal(size=m))
    K = S[:, None] * spd(r, m, cond=1e4) * S[None, :]
    b, z = r.normal(size=m), r.normal(size=m)
    th, info = LA.precision_draw(*j(K, b, z))
    ref = dense_draw(K, b, z)
    sd = np.sqrt(np.diag(np.linalg.inv(K)))
    assert np.max(np.abs(np.asarray(th) - ref) / (np.abs(ref) + sd)) <= 1e-10
    assert bool(info.ok)
    assert 0.0 < float(info.min_pivot) <= 1.0  # pivots of the unit-diagonal scaled matrix


def test_precision_draw_reads_only_the_lower_triangle():
    r = rng(4)
    K = spd(r, 6)
    b, z = r.normal(size=6), r.normal(size=6)
    K_junk = np.tril(K) + np.triu(np.full((6, 6), 1e30), 1)
    a, _ = LA.precision_draw(*j(K, b, z))
    c, info = LA.precision_draw(*j(K_junk, b, z))
    assert np.array_equal(np.asarray(a), np.asarray(c)) and bool(info.ok)


def test_padding_is_exact():
    """Trailing inactive slots (prior N(0,1), zero data) give a block-diagonal L: the active draw
    equals the unpadded draw with z[:q_i] and the padded coordinates are z (D-15, N07 §2.2)."""
    r = rng(5)
    q, qi = 14, 9
    F, w, FF = gram_problem(6, 64, 50, q)
    cmask = np.r_[np.ones(qi), np.zeros(q - qi)]
    prec = np.where(cmask > 0, np.exp(r.normal(size=q)), 1.0)
    mean = r.normal(size=q) * cmask
    y = r.normal(size=64)
    y[50:] = 0.0
    G = np.asarray(LA.weighted_gram(*j(w), FF=jnp.asarray(FF), mode="packed", q=q))
    K = np.outer(cmask, cmask) * G + np.diag(prec)
    b = prec * mean + cmask * (F.T @ (w * y))
    z = r.normal(size=q)
    th, info = LA.precision_draw(*j(K, b, z))
    th_u, info_u = LA.precision_draw(*j(K[:qi, :qi], b[:qi], z[:qi]))
    th = np.asarray(th)
    assert np.max(np.abs(th[:qi] - np.asarray(th_u)) / (1.0 + np.abs(th[:qi]))) <= 1e-12
    assert np.array_equal(th[qi:], z[qi:])
    assert bool(info.ok) and bool(info_u.ok)


def _dec_cholesky_draw(K: np.ndarray, b: np.ndarray, z: np.ndarray, digits: int = 50):
    """``s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z)``, ``L = chol(sKs)``, in ``digits``-digit decimal arithmetic."""
    m = len(b)
    with localcontext() as ctx:
        ctx.prec = digits
        Kd = [[Decimal(float(K[i, k])) for k in range(m)] for i in range(m)]
        s = [1 / Kd[i][i].sqrt() for i in range(m)]
        A = [[s[i] * Kd[i][k] * s[k] for k in range(m)] for i in range(m)]
        L = [[Decimal(0)] * m for _ in range(m)]
        for c in range(m):
            L[c][c] = (A[c][c] - sum(L[c][t] ** 2 for t in range(c))).sqrt()
            for i in range(c + 1, m):
                L[i][c] = (A[i][c] - sum(L[i][t] * L[c][t] for t in range(c))) / L[c][c]
        v = [Decimal(0)] * m
        for i in range(m):
            v[i] = (s[i] * Decimal(float(b[i])) - sum(L[i][t] * v[t] for t in range(i))) / L[i][i]
        wv = [v[i] + Decimal(float(z[i])) for i in range(m)]
        x = [Decimal(0)] * m
        for i in reversed(range(m)):
            x[i] = (wv[i] - sum(L[t][i] * x[t] for t in range(i + 1, m))) / L[i][i]
        return np.array([float(s[i] * x[i]) for i in range(m)])


@pytest.mark.parametrize("span", [1e8, 1e16])
def test_jacobi_scaling_with_large_precision_disparities(span):
    """``K = S A S`` with a well-conditioned ``A`` and ``diag(K)`` spanning ``span`` (intercept
    prior precision ~1e-7 against data precision ~1e8 under SV, N05 §7.2): the scaled Cholesky
    draw matches a 50-digit evaluation of the same formula componentwise."""
    r = rng(7)
    m = 8
    A = spd(r, m, cond=20.0)
    S = np.geomspace(1.0, np.sqrt(span), m)[r.permutation(m)]
    K = S[:, None] * A * S[None, :]
    b = S * r.normal(size=m)  # moderate scaled mean
    z = r.normal(size=m)
    ref = _dec_cholesky_draw(K, b, z)
    th, info = LA.precision_draw(*j(K, b, z))
    assert bool(info.ok)
    assert np.max(np.abs(np.asarray(th) - ref) / np.abs(ref)) <= 1e-12


def test_precision_moments_match_dense():
    r = rng(8)
    K = spd(r, 7, cond=1e3)
    b = r.normal(size=7)
    mean, cov = LA.precision_moments(*j(K, b))
    cov_ref = np.linalg.inv(K)
    np.testing.assert_allclose(np.asarray(mean), cov_ref @ b, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(cov), cov_ref, rtol=1e-11, atol=1e-14)
    assert np.array_equal(np.asarray(cov), np.asarray(cov).T)


def test_factor_and_factored_draw_compose_to_precision_draw():
    r = rng(9)
    K = spd(r, 5)
    b, z = r.normal(size=5), r.normal(size=5)
    L, s, info = LA.precision_factor(jnp.asarray(K))
    th = LA.precision_draw_factored(L, s, *j(b, z))
    th2, info2 = LA.precision_draw(*j(K, b, z))
    assert np.array_equal(np.asarray(th), np.asarray(th2))
    assert bool(info.ok) and float(info.min_pivot) == float(info2.min_pivot)


def test_failures_are_values_not_exceptions():
    draw = jax.jit(LA.precision_draw)
    K_bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    _, info = draw(*j(K_bad, np.ones(2), np.ones(2)))
    assert not bool(info.ok)
    assert float(info.min_pivot) <= 0.0 and np.isfinite(float(info.min_pivot))
    _, info = draw(*j(np.eye(2), np.array([np.nan, 1.0]), np.ones(2)))  # poisoned rhs
    assert not bool(info.ok) and float(info.min_pivot) == 1.0
    _, info = draw(*j(-np.eye(2), np.ones(2), np.ones(2)))  # negative diagonal: s is NaN
    assert not bool(info.ok) and float(info.min_pivot) <= 0.0
    st = StepInfo.success().merge(info)
    assert not bool(st.ok) and np.isfinite(float(st.min_pivot))


def test_solve_unit_lower():
    r = rng(11)
    n = 6
    A = np.tril(r.normal(size=(n, n)), -1) + np.diag(r.normal(size=n) * 100)  # diagonal ignored
    B = r.normal(size=(n, 3))
    A1 = np.tril(A, -1) + np.eye(n)
    X = LA.solve_unit_lower(*j(A, B))
    np.testing.assert_allclose(np.asarray(X), np.linalg.solve(A1, B), rtol=1e-12, atol=1e-13)
    x = LA.solve_unit_lower(*j(A, B[:, 0]))
    assert x.shape == (n,)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A1, B[:, 0]), rtol=1e-12, atol=1e-13)


# ------------------------------------------------------------------- Constant fast paths (D-26)


def mode1_problem(
    seed: int,
    i: int = 3,
    n: int = 6,
    p: int = 2,
    T: int = 120,
    var_const=1e7,
    *,
    rho: float = 0.5,
    drift: float = 0.3,
    y0: float = 0.0,
):
    """A structural Mode I problem at the Minnesota scales (DESIGN §6: intercept variance
    ``var_const·ψ_i``, own lags ``(0.2/l)²``, cross lags ``(0.1/l)²ψ_i/ψ_j``,
    ``a_ij ~ N(0, ψ_i/ψ_j)``) for AR(1) data ``y_t = ρy_{t−1} + drift + ψ^{½}ε_t``, ``y_0 = y0``
    (level data: ``ρ`` near 1, ``drift = (1 − ρ)·level``, ``y0 = level``)."""
    r = rng(seed)
    k = 1 + n * p
    q = k + n - 1
    psi = np.exp(r.normal(size=n))
    Y = np.zeros((T + p, n))
    Y[0] = y0
    for t in range(1, T + p):
        Y[t] = rho * Y[t - 1] + r.normal(size=n) * np.sqrt(psi) + drift
    X = np.column_stack([np.ones(T)] + [Y[p - lag : T + p - lag] for lag in range(1, p + 1)])
    F = np.column_stack([X, X @ (0.1 * r.normal(size=(k, n - 1))) - Y[p:, : n - 1]])
    cmask = np.r_[np.ones(k), (np.arange(n - 1) < i).astype(float)]
    d = np.empty(q)
    d[0] = 1.0 / (var_const * psi[i])
    for lag in range(1, p + 1):
        for jj in range(n):
            sd = (0.2 if jj == i else 0.1) / lag * np.sqrt(psi[i] / psi[jj])
            d[1 + (lag - 1) * n + jj] = 1.0 / sd**2
    d[k:] = np.where(cmask[k:] > 0, psi[i] / psi[: n - 1], 1.0)
    mean0 = 0.1 * r.normal(size=q) * cmask
    return F.T @ F, d, mean0, F.T @ Y[p:, i], cmask, psi[i]


def linear_moments(draw, q: int):
    """Mean (``z = 0``) and covariance ``JJᵀ`` of a draw that is affine in ``z``."""
    m = draw(jnp.zeros(q))
    J = jax.vmap(lambda e: draw(e) - m)(jnp.eye(q)).T
    return np.asarray(m), np.asarray(J @ J.T)


def assert_moments_close(m, C, m_ref, C_ref, cmask, tol):
    act = cmask > 0
    sd = np.sqrt(np.diag(C_ref))[act]
    assert np.max(np.abs((m - m_ref)[act]) / sd) <= tol
    assert np.max(np.abs((C - C_ref)[np.ix_(act, act)]) / np.outer(sd, sd)) <= tol


@pytest.mark.parametrize("var_const", [1e2, 1e7])
@pytest.mark.parametrize("lam_mult", [0.01, 1.0, 100.0])
def test_eigen_path_matches_cholesky_moments(var_const, lam_mult):
    G, d, mean0, fty, cmask, lam0 = mode1_problem(21, var_const=var_const)
    q = len(d)
    lam = lam0 * lam_mult
    c = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    assert isinstance(c, EigenCache) and c.Q.shape == (q, q) and c.g.shape == (q,)
    np.testing.assert_allclose(np.asarray(c.d_half), np.sqrt(d), rtol=1e-15)
    K = np.diag(d) + np.outer(cmask, cmask) * G / lam  # cholesky_const (DESIGN §3.7.2)
    rhs = d * mean0 + cmask * fty / lam
    m_ref, C_ref = (np.asarray(a) for a in LA.precision_moments(*j(K, rhs)))
    m, C = linear_moments(lambda z: LA.eigen_draw(c, lam, z)[0] * cmask, q)
    assert_moments_close(m, C, m_ref * cmask, C_ref, cmask, 1e-12)
    _, info = LA.eigen_draw(c, lam, jnp.zeros(q))
    assert bool(info.ok) and float(info.min_pivot) == np.inf


def _dec_moments(K: np.ndarray, b: np.ndarray, digits: int = 45):
    """Mean ``K⁻¹b`` and covariance ``K⁻¹`` (``L⁻ᵀL⁻¹``, ``K = LLᵀ``) in ``digits``-digit decimal
    arithmetic from the float64 entries of ``K`` and ``b``."""
    m = len(b)
    with localcontext() as ctx:
        ctx.prec = digits
        Kd = [[Decimal(float(K[i, k])) for k in range(m)] for i in range(m)]
        L = [[Decimal(0)] * m for _ in range(m)]
        for c in range(m):
            L[c][c] = (Kd[c][c] - sum(L[c][t] ** 2 for t in range(c))).sqrt()
            for i in range(c + 1, m):
                L[i][c] = (Kd[i][c] - sum(L[i][t] * L[c][t] for t in range(c))) / L[c][c]

        def fwd(v):
            x = [Decimal(0)] * m
            for i in range(m):
                x[i] = (v[i] - sum(L[i][t] * x[t] for t in range(i))) / L[i][i]
            return x

        v = fwd([Decimal(float(x)) for x in b])
        mean = [Decimal(0)] * m
        for i in reversed(range(m)):
            mean[i] = (v[i] - sum(L[t][i] * mean[t] for t in range(i + 1, m))) / L[i][i]
        Li = [fwd([Decimal(int(k == j)) for k in range(m)]) for j in range(m)]  # Li[j] = L⁻¹e_j
        cov = [[sum(Li[r][t] * Li[s][t] for t in range(m)) for s in range(m)] for r in range(m)]
        return np.array([float(x) for x in mean]), np.array([[float(x) for x in c] for c in cov])


def level_problem(seed: int, n: int, p: int, T: int, i: int, rho: float, level: float):
    """:func:`mode1_problem` on level data: ``drift = (1 − ρ)·level``, ``y_0 = level``."""
    return mode1_problem(seed, i=i, n=n, p=p, T=T, rho=rho, drift=(1.0 - rho) * level, y0=level)


def level_reference(G, d, mean0, fty, cmask, lam) -> SimpleNamespace:
    """45-digit mean and covariance of the active block of ``K = diag(d) + cmask⊗cmask ⊙ G/λ``
    (cholesky_const, DESIGN §3.7.2) from the float64 ``K``, with ``κ = cond(sKs)`` (Jacobi-scaled
    precision) and ``t = max_k |E θ_k|/sd_k``; also the float64 ``K`` and right-hand side."""
    act = cmask > 0
    K = np.diag(d) + np.outer(cmask, cmask) * G / lam
    rhs = d * mean0 + cmask * fty / lam
    Ka = K[np.ix_(act, act)]
    m_ex, C_ex = _dec_moments(Ka, rhs[act])
    s = 1.0 / np.sqrt(np.diag(Ka))
    kappa = float(np.linalg.cond(s[:, None] * Ka * s[None, :]))
    t = float(np.max(np.abs(m_ex) / np.sqrt(np.diag(C_ex))))
    return SimpleNamespace(act=act, K=K, rhs=rhs, m=m_ex, C=C_ex, kappa=kappa, t=t)


def sd_errors(m, C, ref) -> tuple[float, float]:
    """Largest mean and covariance errors on the active block, in posterior sd."""
    a = ref.act
    sd = np.sqrt(np.diag(ref.C))
    em = np.max(np.abs(m[a] - ref.m) / sd)
    ec = np.max(np.abs(C[np.ix_(a, a)] - ref.C) / np.outer(sd, sd))
    return float(em), float(ec)


def assert_in_envelope(m, C, ref, c: float) -> None:
    """Mean error ``≤ 1e-12 + cεκ(1 + t)`` and covariance error ``≤ 1e-12 + cεκ`` (the accuracy
    envelope of :func:`LA.eigen_prepare`; both paths are observed within ``c = 1.5``)."""
    em, ec = sd_errors(m, C, ref)
    eps = np.finfo(np.float64).eps
    assert em <= 1e-12 + c * eps * ref.kappa * (1.0 + ref.t), (em, ref.kappa, ref.t)
    assert ec <= 1e-12 + c * eps * ref.kappa, (ec, ref.kappa)


# (seed, n, p, T, i, ρ, level): Minnesota level data with var_const = 1e7. Well-conditioned but
# graded (level 0), persistent levels with cond(sKs) up to ~1e10, the reference shape (q = 100),
# and T < q, where the correlation of the Gram is singular and eigen_prepare falls back to eigh.
# 2021 (T < q, cond ≤ 940) fails by ~50x with the default divide-and-conquer SVD; 1010, 5320 and
# 21064 have posterior t-ratios of 280–700 (eigen mean error 35–300εκ); 20113 has the largest
# observed eigen mean error relative to the envelope (c = 1.5 at κ = 25, t = 308).
LEVEL_PROBLEMS = [
    (102, 4, 2, 60, 2, 0.5, 0.0),
    (109, 4, 2, 60, 1, 0.99, 100.0),
    (110, 4, 2, 60, 2, 0.999, 1e3),
    (100, 4, 2, 60, 0, 0.9995, 1e4),
    (2, 20, 4, 300, 19, 0.999, 900.0),
    (202, 6, 4, 15, 3, 0.5, 0.0),
    (201, 6, 4, 15, 2, 0.99, 100.0),
    (2021, 6, 4, 25, 1, 0.95, 0.0),
    (1010, 3, 1, 200, 0, 0.99, 1e3),
    (5320, 2, 1, 200, 1, 0.99, 1e4),
    (21064, 2, 1, 200, 0, 0.999, 1e4),
    (20113, 2, 1, 80, 0, 0.99, 1.0),
]


@pytest.mark.parametrize("seed,n,p,T,i,rho,level", LEVEL_PROBLEMS)
def test_constant_paths_on_level_data_vs_45_digits(seed, n, p, T, i, rho, level):
    """Eigen and Cholesky paths against a 45-digit solve on Minnesota level data (DESIGN §9.3
    item 1), in the accuracy envelope of :func:`LA.eigen_prepare`: mean error
    ``≤ 1e-12 + cεκ(1 + t)``, covariance error ``≤ 1e-12 + cεκ`` (``κ = cond(sKs)``, ``t`` the
    largest posterior t-ratio). Over 4,400 random problem–λ pairs the eigen path stays within
    ``c = 1.5`` (tested at 5) and the Cholesky path within 0.5 (tested at 1). Both mean errors
    grow with ``t`` (eigen up to ``1e-12 + 470εκ``, Cholesky ``39εκ`` on seed 5320), so the two
    paths agree to 1e-12 only when ``εκ(1 + t)`` is small."""
    G, d, mean0, fty, cmask, lam0 = level_problem(seed, n, p, T, i, rho, level)
    q = len(d)
    c = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    for lam_mult in (0.01, 1.0, 100.0):
        lam = lam0 * lam_mult
        ref = level_reference(G, d, mean0, fty, cmask, lam)
        m_ch, C_ch = (np.asarray(a) for a in LA.precision_moments(*j(ref.K, ref.rhs)))
        m_e, C_e = linear_moments(lambda z, lam=lam: LA.eigen_draw(c, lam, z)[0] * cmask, q)
        assert_in_envelope(m_ch, C_ch, ref, 1.0)
        assert_in_envelope(m_e, C_e, ref, 5.0)
        assert bool(LA.eigen_draw(c, lam, jnp.zeros(q))[1].ok)


# Problems on which an alternative to a choice in eigen_prepare is far outside a tenth of the
# envelope while the shipped decomposition is well inside it (units of c, all three λ):
# 42033: shipped ≤ 0.04, eigh factor of A in place of its Cholesky factor 1.2–1.3;
# 8049: shipped ≤ 0.17, Householder QR of Bᵀ without the column sort 1.1–2.9.
PIN_PROBLEMS = [
    (42033, 5, 3, 25, 4, 0.9995, 100.0),
    (8049, 2, 1, 200, 1, 0.9995, 1e4),
]


@pytest.mark.parametrize("seed,n,p,T,i,rho,level", PIN_PROBLEMS)
def test_eigen_prepare_graded_choices_are_pinned(seed, n, p, T, i, rho, level):
    """Regression pins on the choices that make :func:`LA.eigen_prepare` accurate on graded
    problems: the Cholesky factor of the correlation ``A`` (eigh only as the fallback), the sort
    of the columns of ``Bᵀ`` by decreasing ``δ`` before the QR, and the QR-iteration SVD. On these
    problems the eigen path is within ``c = 0.4`` of the envelope, which the alternatives listed
    at ``PIN_PROBLEMS`` miss by 2.7x or more; the divide-and-conquer SVD fails the envelope test
    itself on seed 2021. A pin, not a bound: elsewhere the shipped path reaches ``c = 1.5``."""
    G, d, mean0, fty, cmask, lam0 = level_problem(seed, n, p, T, i, rho, level)
    q = len(d)
    c = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    for lam_mult in (0.01, 1.0, 100.0):
        lam = lam0 * lam_mult
        ref = level_reference(G, d, mean0, fty, cmask, lam)
        m_e, C_e = linear_moments(lambda z, lam=lam: LA.eigen_draw(c, lam, z)[0] * cmask, q)
        assert_in_envelope(m_e, C_e, ref, 0.4)


def test_eigen_path_gives_an_all_zero_regressor_its_prior():
    """An active regressor that is 0 in every row has ``δ = 0``: its column of ``Bᵀ`` must be 0
    (not the unit pivot that stands in for it in the Cholesky of ``A``), so its draw is the prior
    ``N(mean0_k, 1/d_k)``, as on the Cholesky path."""
    G, d, mean0, fty, cmask, lam = mode1_problem(21)
    k = 2
    G[k, :] = G[:, k] = 0.0
    fty[k] = 0.0
    q = len(d)
    cache = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    K = np.diag(d) + np.outer(cmask, cmask) * G / lam
    m_ref, C_ref = (
        np.asarray(a) for a in LA.precision_moments(*j(K, d * mean0 + cmask * fty / lam))
    )
    m, C = linear_moments(lambda z: LA.eigen_draw(cache, lam, z)[0] * cmask, q)
    assert_moments_close(m, C, m_ref * cmask, C_ref, cmask, 1e-12)
    assert abs(m[k] - mean0[k]) * np.sqrt(d[k]) <= 1e-12 and abs(C[k, k] * d[k] - 1.0) <= 1e-12


def ks_standardised(draws: np.ndarray, mean: np.ndarray, cov: np.ndarray, r) -> list[float]:
    """KS p-values of ``aᵀθ`` standardised with the exact moments, for unit and random ``a``."""
    q = len(mean)
    dirs = [np.eye(q)[k] for k in (0, q // 2, q - 1)] + [r.normal(size=q) for _ in range(3)]
    out = []
    for a in dirs:
        x = (draws @ a - a @ mean) / np.sqrt(a @ cov @ a)
        out.append(stats.kstest(x, "norm").pvalue)
    return out


def test_eigen_path_ks_on_10k_draws():
    G, d, mean0, fty, cmask, lam = mode1_problem(22)
    act = cmask > 0
    q = len(d)
    c = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    K = np.diag(d) + np.outer(cmask, cmask) * G / lam
    m_ref, C_ref = (
        np.asarray(a) for a in LA.precision_moments(*j(K, d * mean0 + cmask * fty / lam))
    )
    Z = random.normal(random.fold_in(KEY, 1), (10_000, q))
    draws = np.asarray(jax.vmap(lambda z: LA.eigen_draw(c, lam, z)[0])(Z))[:, act]
    pvals = ks_standardised(draws, m_ref[act], C_ref[np.ix_(act, act)], rng(23))
    assert min(pvals) > 1e-3, pvals


def test_eigen_prepare_flags_a_non_psd_gram():
    G, d, mean0, fty, cmask, lam = mode1_problem(24)
    G_bad = G.copy()
    G_bad[1, 2] = G_bad[2, 1] = 2.0 * np.sqrt(G[1, 1] * G[2, 2])  # |correlation| = 2
    c = LA.eigen_prepare(*j(G_bad, d, mean0, fty, cmask))
    _, info = LA.eigen_draw(c, lam, jnp.zeros(len(d)))
    assert not bool(info.ok)


@pytest.mark.parametrize("lam", [0.003, 1.0, 250.0])
def test_fixed_factor_path_matches_cholesky_moments(lam):
    """Mode II × Constant: ``K = K̃/λ`` and ``rhs/λ``; ``θ̂`` is λ-free, the covariance
    ``λK̃⁻¹``."""
    G, d, mean0, fty, cmask, _ = mode1_problem(25)
    q = len(d)
    K_t = np.outer(cmask, cmask) * G + np.diag(d)
    rhs = d * mean0 + cmask * fty
    c = LA.fixed_factor_prepare(*j(K_t, rhs))
    assert isinstance(c, FixedFactorCache)
    m_ref, C_ref = (np.asarray(a) for a in LA.precision_moments(*j(K_t / lam, rhs / lam)))
    m, C = linear_moments(lambda z: LA.fixed_factor_draw(c, lam, z) * cmask, q)
    assert_moments_close(m, C, m_ref * cmask, C_ref, cmask, 1e-12)
    np.testing.assert_allclose(np.asarray(c.theta_hat) * cmask, m_ref * cmask, rtol=1e-11, atol=0)


def test_fixed_factor_ks_on_10k_draws():
    G, d, mean0, fty, cmask, lam = mode1_problem(26)
    act = cmask > 0
    q = len(d)
    K_t = np.outer(cmask, cmask) * G + np.diag(d)
    rhs = d * mean0 + cmask * fty
    c = LA.fixed_factor_prepare(*j(K_t, rhs))
    m_ref, C_ref = (np.asarray(a) for a in LA.precision_moments(*j(K_t / lam, rhs / lam)))
    Z = random.normal(random.fold_in(KEY, 2), (10_000, q))
    draws = np.asarray(jax.vmap(lambda z: LA.fixed_factor_draw(c, lam, z))(Z))[:, act]
    pvals = ks_standardised(draws, m_ref[act], C_ref[np.ix_(act, act)], rng(27))
    assert min(pvals) > 1e-3, pvals


def test_fixed_factor_failure_poisons_every_draw():
    c = LA.fixed_factor_prepare(*j(np.array([[1.0, 3.0], [3.0, 1.0]]), np.ones(2)))
    assert not np.all(np.isfinite(np.asarray(LA.fixed_factor_draw(c, 1.0, jnp.ones(2)))))


# ------------------------------------------------------------------- tridiagonal sampler (S4)


def sv_like_tridiag(seed: int, m: int, m_eff: int, phi=0.95, sigma=0.2):
    """NC AWOL entries (DESIGN §3.7.4) on ``m`` states, the first ``m_eff`` real; later states are
    decoupled N(0,1) dummies (diag 1, off 0, b 0)."""
    r = rng(seed)
    D = 1.0 / r.choice([0.11265, 0.40611, 0.98583, 2.54498, 7.33342], size=m)
    diag = np.ones(m)
    diag[0] = 1.0 + phi**2
    diag[1 : m_eff - 1] = sigma**2 * D[1 : m_eff - 1] + 1.0 + phi**2
    diag[m_eff - 1] = sigma**2 * D[m_eff - 1] + 1.0
    off = np.zeros(m - 1)
    off[: m_eff - 1] = -phi
    b = np.zeros(m)
    b[1:m_eff] = sigma * D[1:m_eff] * r.normal(size=m_eff - 1)
    return diag, off, b, r.normal(size=m)


def dense_tridiag(diag, off):
    return np.diag(diag) + np.diag(off, -1) + np.diag(off, 1)


@pytest.mark.parametrize("m,m_eff", [(1, 1), (2, 2), (41, 41), (65, 51)])
def test_tridiag_sampler_matches_dense(m, m_eff):
    diag, off, b, z = sv_like_tridiag(30 + m, m, m_eff)
    x, logdet, minpiv = LA.tridiag_precision_sample(*j(diag, off, b, z))
    Q = dense_tridiag(diag, off)
    ref = dense_draw(Q, b, z)
    sd = np.sqrt(np.diag(np.linalg.inv(Q)))
    assert np.max(np.abs(np.asarray(x) - ref) / (np.abs(ref) + sd)) <= 1e-12
    # Oracle log|Q| from the dense Cholesky factor (Q is SPD: the factorisation succeeding is the
    # positive-definiteness check). Not np.linalg.slogdet/det: their LU (getrf) in Apple
    # Accelerate raises a spurious divide-by-zero flag on these matrices (NumPy 2.5, arm64), which
    # the suite's error::RuntimeWarning filter turns into a failure of the oracle itself.
    L = np.linalg.cholesky(Q)
    ld = 2.0 * float(np.sum(np.log(np.diag(L))))
    assert abs(float(logdet) - ld) <= 1e-12 * max(1.0, abs(ld))
    piv = np.diag(L) ** 2
    np.testing.assert_allclose(float(minpiv), piv.min(), rtol=1e-13)
    if m_eff < m:  # decoupled padded states return their noise and add nothing to log|Q|
        x = np.asarray(x)
        assert np.array_equal(x[m_eff:], z[m_eff:])
        x_u, ld_u, _ = LA.tridiag_precision_sample(
            *j(diag[:m_eff], off[: m_eff - 1], b[:m_eff], z[:m_eff])
        )
        np.testing.assert_allclose(x[:m_eff], np.asarray(x_u), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(float(logdet), float(ld_u), rtol=1e-14)


def test_tridiag_sampler_reports_a_non_pd_matrix():
    diag = np.array([1.0, 1.0, 1.0, 1.0])
    off = np.array([0.3, 1.2, 0.1])  # indefinite at the third pivot
    x, _, minpiv = LA.tridiag_precision_sample(*j(diag, off, np.zeros(4), np.ones(4)))
    assert float(minpiv) < 0.0  # the actual negative pivot, not NaN
    assert not np.all(np.isfinite(np.asarray(x)))


# --------------------------------------------------------------------------- stable 2×2 (N02 T7)


def simulate_h(mu, phi, sigma, T, seed):
    r = rng(seed)
    h = np.empty(T + 1)
    h[0] = mu + sigma / np.sqrt(1.0 - phi**2) * r.normal()
    for t in range(1, T + 1):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma * r.normal()
    return h


def exact_bT(x, z, eps1, eps2):
    """``b_T = (X'X + B₀⁻¹)⁻¹X'z`` in exact rational arithmetic, plus ``A``'s entries."""
    X, Z = [Fraction(v) for v in x], [Fraction(v) for v in z]
    T = len(X)
    A11, A12 = T + Fraction(eps1), sum(X)
    A22 = sum(v * v for v in X) + Fraction(eps2)
    sz, sxz = sum(Z), sum(a * b for a, b in zip(X, Z, strict=True))
    det = A11 * A22 - A12 * A12
    return ((A22 * sz - A12 * sxz) / det, (A11 * sxz - A12 * sz) / det), (A11, A12, A22, det)


def frac_sqrt(f: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return (Decimal(f.numerator) / Decimal(f.denominator)).sqrt()


EPS1, EPS2 = 1e-12, 1e-8  # B₀ = diag(1e12, 1e8) (D-6)


@pytest.mark.parametrize("mu,phi,sigma", [(-25.0, 0.5, 1e-4), (-9.0, 0.99, 1e-6)])
def test_stable_ar1_regression_vs_exact_rationals(mu, phi, sigma):
    """N02 T7: large, nearly constant ``h``. Centred ``b_T`` exact to 1e-15 relative; the
    uncentred form (negative control) is off by more than 1e-8."""
    for seed in range(3):
        h = simulate_h(mu, phi, sigma, 300, seed)
        x, z = h[:-1], h[1:]
        b, L = LA.stable_ar1_regression(*j(x, z, np.ones(300)), EPS1, EPS2)
        (g_ex, p_ex), (_, A12, A22, det) = exact_bT(x, z, EPS1, EPS2)
        for got, ex in zip(np.asarray(b), (g_ex, p_ex), strict=True):
            assert abs(float((Fraction(float(got)) - ex) / ex)) <= 1e-15
        L_ex = [
            frac_sqrt(A22 / det),
            -Decimal(A12.numerator) / Decimal(A12.denominator) / frac_sqrt(det * A22),
            1 / frac_sqrt(A22),
        ]
        L = np.asarray(L)
        for got, ex in zip((L[0, 0], L[1, 0], L[1, 1]), L_ex, strict=True):
            assert abs(float((Decimal(float(got)) - ex) / ex)) <= 1e-14
        assert L[0, 1] == 0.0
        # uncentred b_T (negative control: the form the cross-check rejected)
        A = np.array([[300 + EPS1, x.sum()], [x.sum(), x @ x + EPS2]])
        b_unc = np.linalg.solve(A, np.array([z.sum(), x @ z]))
        assert abs(float((Fraction(float(b_unc[1])) - p_ex) / p_ex)) > 1e-8


def test_stable_ar1_regression_vs_inverse_on_typical_data():
    for seed in range(5):
        h = simulate_h(-1.0, 0.95, 0.2, 200, 40 + seed)
        x, z = h[:-1], h[1:]
        b, L = LA.stable_ar1_regression(*j(x, z, np.ones(200)), EPS1, EPS2)
        X = np.column_stack([np.ones(200), x])
        A = X.T @ X + np.diag([EPS1, EPS2])
        B_T = np.linalg.inv(A)
        np.testing.assert_allclose(np.asarray(b), B_T @ X.T @ z, rtol=5e-13)
        L = np.asarray(L)
        np.testing.assert_allclose(L @ L.T, B_T, rtol=5e-13)
        np.testing.assert_allclose(L, np.linalg.cholesky(B_T), rtol=5e-13, atol=1e-30)


def test_stable_ar1_regression_masks_padded_rows():
    h = simulate_h(-2.0, 0.9, 0.3, 64, 3)
    x, z = h[:-1], h[1:]
    m = (np.arange(64) < 50).astype(float)
    xj, zj = x.copy(), z.copy()
    xj[50:], zj[50:] = np.nan, np.inf  # junk on padded rows is never read
    b, L = LA.stable_ar1_regression(*j(xj, zj, m), EPS1, EPS2)
    b_u, L_u = LA.stable_ar1_regression(*j(x[:50], z[:50], np.ones(50)), EPS1, EPS2)
    np.testing.assert_allclose(np.asarray(b), np.asarray(b_u), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(L), np.asarray(L_u), rtol=1e-14)


def test_stable_ar1_regression_with_every_row_masked_is_the_auxiliary_prior():
    b, L = LA.stable_ar1_regression(*j(np.ones(8), np.ones(8), np.zeros(8)), EPS1, EPS2)
    assert np.array_equal(np.asarray(b), np.zeros(2))
    np.testing.assert_allclose(np.asarray(L), np.diag([EPS1**-0.5, EPS2**-0.5]), rtol=1e-15)


# --------------------------------------------------------------------------- batching, dtypes


def test_vmap_lanes_equal_unbatched_calls():
    """Every kernel is written for one problem; ``vmap`` lanes equal separate calls (1e-14)."""
    r = rng(50)
    B, q, T = 4, 9, 40

    def check(f, *batched):
        out_b = jax.vmap(f)(*batched)
        for lane in range(B):
            out_l = f(*(jax.tree.map(lambda a, lane=lane: a[lane], x) for x in batched))
            for a, c in zip(jax.tree.leaves(out_b), jax.tree.leaves(out_l), strict=True):
                a, c = np.asarray(a)[lane], np.asarray(c)
                if a.dtype == bool:
                    assert a == c
                else:
                    np.testing.assert_allclose(a, c, rtol=1e-14, atol=1e-300)

    F, _, FF = gram_problem(51, T, 32, q)
    W = jnp.asarray(np.exp(r.normal(size=(B, T))))
    Fj, FFj = jnp.asarray(F), jnp.asarray(FF)
    check(lambda w: LA.weighted_gram(w, FF=FFj, mode="packed", q=q), W)
    check(lambda w: LA.weighted_gram(w, F=Fj, mode="packed_tiled", q=q, tile=7), W)
    check(lambda w: LA.weighted_gram(w, F=Fj, mode="batched", q=q), W)
    Ks = jnp.asarray(np.stack([spd(r, q, 1e3) for _ in range(B)]))
    bs, zs = jnp.asarray(r.normal(size=(B, q))), jnp.asarray(r.normal(size=(B, q)))
    check(LA.precision_draw, Ks, bs, zs)
    check(LA.precision_moments, Ks, bs)
    check(LA.solve_unit_lower, Ks, bs)
    probs = [mode1_problem(60 + lane, i=lane % 3, n=3, p=1, T=30) for lane in range(B)]
    Gs, ds, ms, fs, cs, lams = (jnp.asarray(np.stack(a)) for a in zip(*probs, strict=True))
    caches = jax.vmap(LA.eigen_prepare)(Gs, ds, ms, fs, cs)
    check(LA.eigen_prepare, Gs, ds, ms, fs, cs)
    zq = jnp.asarray(r.normal(size=(B, Gs.shape[-1])))
    check(LA.eigen_draw, caches, lams, zq)
    Kt = cs[:, :, None] * Gs * cs[:, None, :] + jax.vmap(jnp.diag)(ds)
    check(LA.fixed_factor_prepare, Kt, ds * ms + cs * fs)
    ffc = jax.vmap(LA.fixed_factor_prepare)(Kt, ds * ms + cs * fs)
    check(LA.fixed_factor_draw, ffc, lams, zq)
    tri = [sv_like_tridiag(70 + lane, 33, 33 - 2 * lane) for lane in range(B)]
    check(LA.tridiag_precision_sample, *(jnp.asarray(np.stack(a)) for a in zip(*tri, strict=True)))
    hs = np.stack([simulate_h(-1.0, 0.9, 0.2, T, 80 + lane) for lane in range(B)])
    masks = jnp.asarray(
        np.stack([(np.arange(T) < T - 3 * lane).astype(float) for lane in range(B)])
    )
    check(
        lambda x, z, m: LA.stable_ar1_regression(x, z, m, EPS1, EPS2),
        *j(hs[:, :-1], hs[:, 1:]),
        masks,
    )


def test_outputs_are_float64_under_jit():
    r = rng(90)
    K, b, z = spd(r, 4), r.normal(size=4), r.normal(size=4)
    th, info = jax.jit(LA.precision_draw)(*j(K, b, z))
    assert th.dtype == jnp.float64 and info.min_pivot.dtype == jnp.float64
    assert info.ok.dtype == jnp.bool_
    x, ld, mp = jax.jit(LA.tridiag_precision_sample)(*j(np.full(4, 2.0), np.full(3, -0.5), b, z))
    assert x.dtype == ld.dtype == mp.dtype == jnp.float64


# --------------------------------------------------------------------------- timing (non-gating)


def test_timing_table_at_reference_shapes(capsys):
    """Prints ms per call for all lanes at the reference shape (``n = 20``, ``p = 4``: ``k = 81``,
    ``q = 100``, ``T_pad = 320``, 4 chains → 80 lanes). Informational only (WP4 acceptance)."""
    C, n, q, T = 4, 20, 100, 320
    B = C * n
    r = rng(99)
    F, _, FF = gram_problem(99, T, 300, q)
    W = jnp.asarray(np.exp(r.normal(size=(B, T))))
    Fj, FFj = jnp.asarray(F), jnp.asarray(FF)
    Ks = jnp.asarray(np.stack([spd(r, q, 1e3) for _ in range(4)] * (B // 4)))
    bs, zs = jnp.asarray(r.normal(size=(B, q))), jnp.asarray(r.normal(size=(B, q)))
    G, d, mean0, fty, cmask, lam = mode1_problem(98, n=n, p=4, T=300, i=10)
    cache = LA.eigen_prepare(*j(G, d, mean0, fty, cmask))
    caches = jax.tree.map(lambda a: jnp.broadcast_to(a, (B, *a.shape)), cache)
    ffc = LA.fixed_factor_prepare(*j(np.outer(cmask, cmask) * G + np.diag(d), d * mean0))
    ffcs = jax.tree.map(lambda a: jnp.broadcast_to(a, (B, *a.shape)), ffc)
    lams = jnp.full((B,), lam)
    tri = sv_like_tridiag(97, T + 1, T + 1)
    tris = [jnp.asarray(np.broadcast_to(a, (B, *a.shape))) for a in tri]
    cases = {
        "weighted_gram packed (80 × 320 × P=5050)": (
            jax.vmap(lambda w: LA.weighted_gram(w, FF=FFj, mode="packed", q=q)), (W,)),
        "weighted_gram packed_tiled, tile 2048": (
            jax.vmap(lambda w: LA.weighted_gram(w, F=Fj, mode="packed_tiled", q=q, tile=2048)),
            (W,)),
        "weighted_gram batched": (
            jax.vmap(lambda w: LA.weighted_gram(w, F=Fj, mode="batched", q=q)), (W,)),
        "precision_draw (80 × q=100)": (jax.vmap(LA.precision_draw), (Ks, bs, zs)),
        "eigen_draw (80 × q=100)": (jax.vmap(LA.eigen_draw), (caches, lams, zs)),
        "fixed_factor_draw (80 × q=100)": (jax.vmap(LA.fixed_factor_draw), (ffcs, lams, zs)),
        "tridiag_precision_sample (80 × 321)": (jax.vmap(LA.tridiag_precision_sample), tris),
    }  # fmt: skip
    rows = []
    for name, (f, args) in cases.items():
        fj = jax.jit(f)
        jax.block_until_ready(fj(*args))
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(fj(*args))
            best = min(best, time.perf_counter() - t0)
        rows.append((name, 1e3 * best))
    with capsys.disabled():
        print("\n  WP4 linalg timing (min of 5, ms per call, all lanes; non-gating)")
        for name, ms in rows:
            print(f"    {name:44s} {ms:8.3f}")
    assert all(np.isfinite(ms) for _, ms in rows)
