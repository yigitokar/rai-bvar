"""Dense linear algebra for the NumPy engines. No explicit inverses (SPEC §8).

Batched routines broadcast over leading axes; matrices live in the last two axes.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

_LOG_2PI = float(np.log(2.0 * np.pi))


def positive_diagonal(r: np.ndarray, *, upper: bool) -> np.ndarray:
    """Flip rows (upper) or columns (lower) of a triangular factor so its diagonal is positive."""
    signs = np.sign(np.diagonal(r, axis1=-2, axis2=-1))
    signs = np.where(signs == 0, 1.0, signs)
    return r * signs[..., :, None] if upper else r * signs[..., None, :]


def qr_least_squares(X: np.ndarray, Y: np.ndarray, *, rtol: float = 1e-10):
    """Least squares via thin QR.

    Returns ``(B, R, resid)`` with ``B`` of shape ``(k, n)``, ``R`` the upper-triangular factor of
    ``X`` (positive diagonal, so ``X'X = R'R``) and the residuals ``Y − XB``.

    Rank deficiency is judged column by column and is invariant to column scaling: column ``j``
    is rejected when ``|R_jj| <= rtol·‖X_j‖``, i.e. when it lies (numerically) in the span of
    the previous columns.
    """
    X = np.asarray(X, dtype=np.float64)
    Q, R = np.linalg.qr(X, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q, R = Q * signs, R * signs[:, None]
    if X.shape[0] < X.shape[1] or np.any(np.abs(np.diag(R)) <= rtol * np.linalg.norm(X, axis=0)):
        raise np.linalg.LinAlgError("design matrix is (numerically) rank deficient")
    B = solve_triangular(R, Q.T @ Y, lower=False)
    return B, R, Y - X @ B


def gram_cholesky(
    A: np.ndarray, *, rtol: float = 1e-12, ref_norms: np.ndarray | None = None
) -> np.ndarray:
    """Lower Cholesky factor of ``A'A`` computed from the QR of ``A`` (never forms ``A'A``).

    Raises when a column is (numerically) a combination of the previous ones: ``|R_jj|`` is
    compared with ``rtol · ref_j``. ``ref_norms`` defaults to the column norms of ``A`` (unit
    free); callers whose ``A`` is the result of a cancellation (residuals, demeaned data) pass
    the norms before cancellation so that exactly fitted columns are caught.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < A.shape[1]:
        raise np.linalg.LinAlgError(f"A'A is singular: A has fewer rows than columns {A.shape}")
    R = positive_diagonal(np.linalg.qr(A, mode="r"), upper=True)
    ref = (
        np.linalg.norm(A, axis=0) if ref_norms is None else np.asarray(ref_norms, dtype=np.float64)
    )
    ref = np.where(ref > 0, ref, 1.0)
    bad = np.flatnonzero(np.diag(R) <= rtol * ref)
    if bad.size:
        j = int(bad[0])
        raise np.linalg.LinAlgError(
            f"A'A is (numerically) singular at column {j} (|R_jj| / ref = {np.diag(R)[j] / ref[j]:.2g}); "
            "no silent jitter (SPEC §8)"
        )
    return R.T


def sample_inverse_wishart_chol(
    scale_chol: np.ndarray, df: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Lower Cholesky factors of ``Σ ~ IW(Ψ, df)`` with ``Ψ = LL'``; shape ``(size, n, n)``.

    Bartlett decomposition: ``Σ⁻¹ = L⁻ᵀ A Aᵀ L⁻¹`` with ``A`` lower triangular,
    ``A_ii² ~ χ²(df − i)`` and standard normal entries below the diagonal. Then ``Σ = CCᵀ``
    with ``C = LA⁻ᵀ``, obtained by solving ``ACᵀ = Lᵀ``; the QR of ``Cᵀ`` gives the lower
    Cholesky factor.
    """
    L = np.asarray(scale_chol, dtype=np.float64)
    n = L.shape[-1]
    if df <= n - 1:
        raise ValueError(f"inverse-Wishart needs df > n − 1 = {n - 1}, got {df}")
    A = np.zeros((size, n, n))
    rows, cols = np.tril_indices(n, k=-1)
    A[:, rows, cols] = rng.standard_normal((size, rows.size))
    A[:, np.arange(n), np.arange(n)] = np.sqrt(rng.chisquare(df - np.arange(n), size=(size, n)))
    Ct = solve_triangular(A, np.broadcast_to(L.T, (size, n, n)), lower=True)
    R = positive_diagonal(np.linalg.qr(Ct, mode="r"), upper=True)
    return np.swapaxes(R, -1, -2)


def sample_matrix_normal(
    mean: np.ndarray, row_upper: np.ndarray, col_chol: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draws of ``B ~ MN(mean, (RᵀR)⁻¹, Σ)`` given ``R`` upper triangular and Cholesky factors of Σ.

    ``col_chol`` has shape ``(size, n, n)``; the result has shape ``(size, k, n)``. At most two
    arrays of the output's size are alive at once.
    """
    size, k, n = col_chol.shape[0], mean.shape[0], mean.shape[1]
    W = solve_triangular(
        row_upper, rng.standard_normal((size, k, n)), lower=False, overwrite_b=True
    )
    out = np.matmul(W, np.swapaxes(col_chol, -1, -2))
    del W
    out += mean
    return out


def mvn_logpdf_chol(y: np.ndarray, mean: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """Gaussian log density with covariance ``chol @ cholᵀ``; batched over leading axes."""
    diff = np.asarray(y, dtype=np.float64) - mean
    chol = np.broadcast_to(chol, (*diff.shape, diff.shape[-1]))
    z = solve_lower(chol, diff)
    logdet = np.log(np.abs(np.diagonal(chol, axis1=-2, axis2=-1))).sum(axis=-1)
    return -0.5 * np.einsum("...i,...i->...", z, z) - logdet - 0.5 * diff.shape[-1] * _LOG_2PI


def solve_lower(chol: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``chol @ z = b`` by forward substitution, vectorised over the leading axes.

    ``chol`` is ``(..., n, n)`` lower triangular and ``b`` is ``(..., n)``. SciPy's batched
    ``solve_triangular`` loops over the batch in Python, which dominates scoring with many draws.
    """
    chol = np.asarray(chol, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.empty(np.broadcast_shapes(chol.shape[:-1], b.shape))
    for i in range(b.shape[-1]):
        partial = np.einsum("...j,...j->...", chol[..., i, :i], z[..., :i])
        z[..., i] = (b[..., i] - partial) / chol[..., i, i]
    return z


def marginal_chol(chol: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Cholesky factor of the covariance of a subset ``idx`` of the variables."""
    rows = chol[..., idx, :]
    return gram_cholesky_batched(np.swapaxes(rows, -1, -2))


def gram_cholesky_batched(A: np.ndarray) -> np.ndarray:
    """Batched :func:`gram_cholesky`: lower factor of ``AᵀA`` for ``A`` of shape ``(..., m, q)``."""
    R = positive_diagonal(np.linalg.qr(A, mode="r"), upper=True)
    return np.swapaxes(R, -1, -2)
