"""Dense linear algebra helpers (SPEC §8: no explicit inverses, triangular factors, QR)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rai_bvar.linalg import (
    gram_cholesky,
    marginal_chol,
    mvn_logpdf_chol,
    positive_diagonal,
    qr_least_squares,
    sample_inverse_wishart_chol,
    sample_matrix_normal,
    solve_lower,
)


def _random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _assert_lower_positive(L: np.ndarray) -> None:
    np.testing.assert_array_equal(np.triu(L, k=1), 0.0)
    assert np.all(np.diagonal(L, axis1=-2, axis2=-1) > 0)


# --------------------------------------------------------------------------- least squares


@pytest.mark.parametrize(("T", "k", "n"), [(50, 4, 3), (200, 13, 5), (8, 8, 1)])
def test_qr_least_squares_matches_lstsq(T, k, n):
    rng = np.random.default_rng(T + k)
    X = rng.standard_normal((T, k))
    Y = rng.standard_normal((T, n))
    B, R, resid = qr_least_squares(X, Y)
    expected, *_ = np.linalg.lstsq(X, Y, rcond=None)
    np.testing.assert_allclose(B, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(resid, Y - X @ B, atol=1e-12)
    np.testing.assert_array_equal(np.tril(R, k=-1), 0.0)
    assert np.all(np.diag(R) > 0)
    np.testing.assert_allclose(R.T @ R, X.T @ X, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(X.T @ resid, 0.0, atol=1e-10)  # normal equations


def test_qr_least_squares_accepts_vectors_and_lists():
    X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    B, _, resid = qr_least_squares(X, np.array([1.0, 3.0, 5.0, 7.0]))
    np.testing.assert_allclose(B, [1.0, 2.0], atol=1e-14)
    np.testing.assert_allclose(resid, 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda X: np.column_stack([X, X[:, 1]]),  # duplicated column
        lambda X: np.column_stack([X, X[:, 0] - 2.0 * X[:, 2]]),  # linear combination
        lambda X: np.column_stack([X, np.zeros(len(X))]),  # zero column
        lambda X: X[:2],  # fewer observations than regressors
    ],
)
def test_qr_least_squares_rank_deficient_raises(make_bad):
    X = np.random.default_rng(0).standard_normal((30, 3))
    Xb = make_bad(X)
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        qr_least_squares(Xb, np.ones((Xb.shape[0], 2)))


def test_qr_least_squares_is_invariant_to_column_scaling():
    """A well-posed design in mixed units (GDP in dollars, a rate in fractions) is not singular."""
    rng = np.random.default_rng(1)
    T = 120
    X = np.column_stack([np.ones(T), rng.standard_normal(T), rng.standard_normal(T)])
    beta = np.array([[0.5], [2.0], [-1.0]])
    Y = X @ beta + 0.1 * rng.standard_normal((T, 1))
    scale = np.array([1.0, 2e13, 1e-3])
    B, _, resid = qr_least_squares(X * scale, Y)
    B0, _, resid0 = qr_least_squares(X, Y)
    np.testing.assert_allclose(B * scale[:, None], B0, rtol=1e-9)
    np.testing.assert_allclose(resid, resid0, atol=1e-10)


def test_qr_least_squares_on_ill_conditioned_design():
    """Condition number ~1e12: QR stays accurate where the normal equations break down (SPEC §10.2)."""
    rng = np.random.default_rng(0)
    T = 100
    z, w, v = rng.standard_normal((3, T))
    # nearly collinear columns (angle ~1e-6) in very different units
    X = np.column_stack([np.ones(T), z, z + 1e-6 * w, v]) * np.array([1.0, 1e3, 1e-3, 1.0])
    assert np.linalg.cond(X) > 5e11
    beta = np.array([[1.0], [2.0], [-3.0], [0.5]])
    Y = X @ beta  # exactly in the column space: zero residual
    B, _, resid = qr_least_squares(X, Y)
    np.testing.assert_allclose(B, beta, rtol=1e-3)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9 * np.abs(Y).max())
    naive = np.linalg.solve(X.T @ X, X.T @ Y)  # what the QR path avoids
    assert np.max(np.abs(naive - beta) / np.abs(beta)) > 1.0


# --------------------------------------------------------------------------- Cholesky helpers


def test_gram_cholesky():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((40, 4))
    L = gram_cholesky(A)
    _assert_lower_positive(L)
    np.testing.assert_allclose(L @ L.T, A.T @ A, rtol=1e-12)
    np.testing.assert_allclose(L, np.linalg.cholesky(A.T @ A), rtol=1e-10)


@pytest.mark.parametrize(
    "A",
    [np.column_stack([np.ones(5), np.zeros(5)]), np.ones((2, 3))],
    ids=["zero column", "fewer rows than columns"],
)
def test_gram_cholesky_singular_raises(A):
    with pytest.raises(np.linalg.LinAlgError):
        gram_cholesky(A)


@pytest.mark.parametrize("upper", [True, False])
def test_positive_diagonal_preserves_the_product(upper):
    rng = np.random.default_rng(4)
    T = np.triu(rng.standard_normal((3, 3))) if upper else np.tril(rng.standard_normal((3, 3)))
    P = positive_diagonal(T, upper=upper)
    assert np.all(np.diag(P) > 0)
    if upper:
        np.testing.assert_allclose(P.T @ P, T.T @ T)
    else:
        np.testing.assert_allclose(P @ P.T, T @ T.T)


def test_marginal_chol():
    rng = np.random.default_rng(5)
    S = 6
    covs = np.stack([_random_spd(4, rng) for _ in range(S)])
    chols = np.linalg.cholesky(covs)
    for idx in ([2], [0, 3], [3, 1, 0], [0, 1, 2, 3]):
        idx = np.array(idx)
        M = marginal_chol(chols, idx)
        assert M.shape == (S, idx.size, idx.size)
        for s in range(S):
            _assert_lower_positive(M[s])
            np.testing.assert_allclose(M[s] @ M[s].T, covs[s][np.ix_(idx, idx)], rtol=1e-12)


# --------------------------------------------------------------------------- densities


def test_mvn_logpdf_chol_matches_scipy():
    rng = np.random.default_rng(6)
    n, S = 3, 7
    covs = np.stack([_random_spd(n, rng) for _ in range(S)])
    means = rng.standard_normal((S, n))
    y = rng.standard_normal(n)
    out = mvn_logpdf_chol(y, means, np.linalg.cholesky(covs))
    expected = [stats.multivariate_normal(means[s], covs[s]).logpdf(y) for s in range(S)]
    np.testing.assert_allclose(out, expected, rtol=1e-12)

    # a batch of observations against one shared covariance
    ys = rng.standard_normal((5, 2, n))
    out = mvn_logpdf_chol(ys, means[0], np.linalg.cholesky(covs[0]))
    assert out.shape == (5, 2)
    np.testing.assert_allclose(
        out, stats.multivariate_normal(means[0], covs[0]).logpdf(ys), rtol=1e-12
    )


def test_solve_lower_matches_scipy():
    from scipy.linalg import solve_triangular

    rng = np.random.default_rng(14)
    chols = np.linalg.cholesky(np.stack([_random_spd(4, rng) for _ in range(6)])).reshape(
        2, 3, 4, 4
    )
    b = rng.standard_normal((2, 3, 4))
    expected = solve_triangular(chols, b[..., None], lower=True)[..., 0]
    np.testing.assert_allclose(solve_lower(chols, b), expected, rtol=1e-12)
    # one factor against many right-hand sides, and many factors against one
    np.testing.assert_allclose(
        solve_lower(chols[0, 0], b),
        solve_triangular(chols[0, 0], b.reshape(-1, 4).T, lower=True).T.reshape(b.shape),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        solve_lower(chols, b[0, 0]),
        solve_triangular(chols, np.broadcast_to(b[0, 0], b.shape)[..., None], lower=True)[..., 0],
        rtol=1e-12,
    )


def test_mvn_logpdf_chol_univariate():
    out = mvn_logpdf_chol(np.array([1.3]), np.array([0.2]), np.array([[2.0]]))
    np.testing.assert_allclose(out, stats.norm(0.2, 2.0).logpdf(1.3), rtol=1e-14)


# --------------------------------------------------------------------------- samplers


def _psi() -> np.ndarray:
    return np.array([[2.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 0.5]])


def test_inverse_wishart_factors_are_lower_with_positive_diagonal():
    L = sample_inverse_wishart_chol(np.linalg.cholesky(_psi()), 8.0, 50, np.random.default_rng(7))
    assert L.shape == (50, 3, 3)
    for s in range(50):
        _assert_lower_positive(L[s])


def test_inverse_wishart_moments():
    psi, df, size = _psi(), 12.0, 40_000
    n = psi.shape[0]
    L = sample_inverse_wishart_chol(np.linalg.cholesky(psi), df, size, np.random.default_rng(8))
    sigma = L @ np.swapaxes(L, -1, -2)
    # E[Σ] = Ψ / (df − n − 1)
    se = sigma.std(axis=0) / np.sqrt(size)
    assert np.all(np.abs(sigma.mean(axis=0) - psi / (df - n - 1)) < 5 * se)
    # E[Σ⁻¹] = df · Ψ⁻¹ (Wishart mean of the precision)
    precision = np.linalg.inv(sigma)
    se = precision.std(axis=0) / np.sqrt(size)
    assert np.all(np.abs(precision.mean(axis=0) - df * np.linalg.inv(psi)) < 5 * se)


def test_inverse_wishart_diagonal_marginals():
    """Σ_ii ~ IW₁(ψ_ii, df − n + 1) = InvGamma((df − n + 1)/2, ψ_ii/2); checks every Bartlett χ² df."""
    psi, df, size = _psi(), 7.5, 20_000
    n = psi.shape[0]
    L = sample_inverse_wishart_chol(np.linalg.cholesky(psi), df, size, np.random.default_rng(9))
    sigma = L @ np.swapaxes(L, -1, -2)
    for i in range(n):
        dist = stats.invgamma(a=(df - n + 1) / 2, scale=psi[i, i] / 2)
        assert stats.kstest(sigma[:, i, i], dist.cdf).pvalue > 1e-3


def test_inverse_wishart_matches_scipy_in_distribution():
    psi, df, size = _psi(), 6.0, 20_000
    ours = sample_inverse_wishart_chol(np.linalg.cholesky(psi), df, size, np.random.default_rng(10))
    ours = ours @ np.swapaxes(ours, -1, -2)
    ref = stats.invwishart(df=df, scale=psi).rvs(size, random_state=np.random.default_rng(11))
    for stat in (
        lambda s: np.linalg.slogdet(s)[1],
        lambda s: s[:, 0, 1] / np.sqrt(s[:, 0, 0] * s[:, 1, 1]),
    ):
        assert stats.ks_2samp(stat(ours), stat(ref)).pvalue > 1e-3


def test_inverse_wishart_rejects_small_df():
    with pytest.raises(ValueError, match="df"):
        sample_inverse_wishart_chol(np.eye(3), 2.0, 5, np.random.default_rng(0))


def test_inverse_wishart_is_reproducible():
    chol = np.linalg.cholesky(_psi())
    a = sample_inverse_wishart_chol(chol, 9.0, 10, np.random.default_rng(42))
    b = sample_inverse_wishart_chol(chol, 9.0, 10, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_matrix_normal_moments():
    """vec(B) ~ N(vec(M), Σ ⊗ (RᵀR)⁻¹) with column-major vec."""
    rng = np.random.default_rng(12)
    k, n, size = 3, 2, 60_000
    mean = rng.standard_normal((k, n))
    XtX = _random_spd(k, rng)
    R = np.linalg.cholesky(XtX).T  # upper, XtX = RᵀR
    sigma = np.array([[1.0, 0.4], [0.4, 0.5]])
    col_chol = np.broadcast_to(np.linalg.cholesky(sigma), (size, n, n))
    draws = sample_matrix_normal(mean, R, col_chol, rng)
    assert draws.shape == (size, k, n)

    se_mean = draws.std(axis=0) / np.sqrt(size)
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 5 * se_mean)

    vec = np.swapaxes(draws, 1, 2).reshape(size, -1)  # stack columns
    cov = np.cov(vec, rowvar=False)
    expected = np.kron(sigma, np.linalg.inv(XtX))
    d = np.sqrt(np.diag(expected))
    se_cov = np.sqrt((np.outer(d, d) ** 2 + expected**2) / size)
    assert np.all(np.abs(cov - expected) < 5 * se_cov)


def test_matrix_normal_uses_each_draws_column_factor():
    rng = np.random.default_rng(13)
    k, n = 2, 2
    mean = np.zeros((k, n))
    col_chol = np.stack([np.eye(n), 10.0 * np.eye(n)])
    draws = sample_matrix_normal(mean, np.eye(k), np.repeat(col_chol, 5000, axis=0), rng)
    small, large = draws[:5000], draws[5000:]
    np.testing.assert_allclose(large.std(), 10.0 * small.std(), rtol=0.05)
