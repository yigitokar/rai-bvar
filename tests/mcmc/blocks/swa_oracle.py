"""NumPy oracles for the reduced-form Π block (CCCM 2022), independent of the JAX kernel.

Test code only (DESIGN §9.3 item 1, §9.8 A7; N04 §4, §5, §9). Nothing here imports JAX or the
kernel module, so every comparison with ``components/mean/linear_reduced.py`` is a comparison of
two implementations (DESIGN §9.1 two-implementation rule).

* **SUR oracle** (N04 §9.1). The joint Gaussian of ``vec Π = (π_1', …, π_n')'`` given ``A`` and
  ``Λ_{1:T}``, with ``Ω_t = Σ_t⁻¹ = A'Λ_t⁻¹A`` formed explicitly (``Λ_t`` is diagonal)::

      Q = blkdiag(diag p_j) + Σ_t Ω_t ⊗ x_t x_t',    b = vec_j(p_j ⊙ m_j) + Σ_t (Ω_t y_t) ⊗ x_t

  The full conditional of ``π_j`` has precision ``Q_jj`` and mean ``Q_jj⁻¹(b_j − Σ_{l≠j} Q_jl π_l)``.
* **Reference sweep**: the Gibbs scan ``j = 0..n−1`` over those conditionals, with the noise
  mapped exactly as the production draw (Jacobi-scaled Cholesky, D-14), so a kernel with the
  same injected ``Z`` must agree to rounding.
* **CCM 2019 step** (N04 §4): ``π_j`` from equation ``j`` alone, ``K = P_j + Σ_t x_t x_t'/λ_jt``,
  ``y*_jt = y_jt + Σ_{i<j} a_ji u_it`` with the already updated ``π_{<j}``. Exact only for
  ``j = n``; the regression guard and the A7 power check use it as the mutant.
* **System-wide algorithm (SWA)**: one joint draw of ``vec Π`` from ``(Q, b)`` (O((nk)³)), and a
  batched Gibbs sampler for reduced × Constant (``Π`` → ``a`` → ``λ``, DESIGN §4.3) whose ``Π``
  step is pluggable, so A7 compares SWA-Gibbs with CTA-Gibbs sharing the ``a`` and ``λ`` steps.

Layouts follow the engine: ``Π`` is ``(k, n)`` (column ``j`` = equation ``j``), ``a`` is the
``(n, n−1)`` state layout (``a[i, j] = a_ij`` for ``j < i``), ``λ`` and ``inv_lam`` are ``(n, T)``,
prior ``mean0``/``prec0`` are ``(n, k)`` (row ``j`` = ``π_j``), noise ``Z`` is ``(n, k)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

# --------------------------------------------------------------------------- toy (N04 §9.1)

TOY_X = np.array([[1, 0.5], [1, -1], [1, 2], [1, 0], [1, 1.5]], dtype=np.float64)
TOY_Y = np.array(
    [[1.0, 0.2, -0.5], [-0.3, 1.1, 0.4], [2.0, -0.7, 1.3], [0.1, 0.5, -0.2], [1.2, 0.9, 0.8]]
)
TOY_A = np.array([[1, 0, 0], [0.5, 1, 0], [-0.3, 0.8, 1]], dtype=np.float64)
TOY_LAM = np.array([[1, 0.5, 2], [2, 1, 0.5], [0.5, 1.5, 1], [1, 0.8, 1.2], [1.5, 0.6, 0.7]]).T
TOY_M = np.array([[0, 0.1, -0.1], [0.5, 0, 0.2]]).T  # (n, k): row j = m_j
TOY_V = np.array([[4, 1, 2], [0.25, 0.5, 1]]).T  # (n, k): prior variances
TOY_PI = np.array([[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]])  # (k, n): state of the conditionals

TOY_K = (
    np.array([[7.5810714286, 6.1736904762], [6.1736904762, 17.069702381]]),
    np.array([[11.270952381, 5.3647619048], [5.3647619048, 15.8938095238]]),
    np.array([[6.2619047619, 2.3928571429], [2.3928571429, 10.3392857143]]),
)
TOY_MEAN = (
    np.array([0.4584459034, 0.5225169483]),
    np.array([0.697919255, 0.1574959305]),
    np.array([0.9656974281, -0.4844619264]),
)
TOY_CCM_K = (
    np.array([[5.4166666667, 5.0], [5.0, 14.25]]),
    np.array([[7.5833333333, 3.8333333333], [3.8333333333, 9.9166666667]]),
)
TOY_CCM_MEAN = (np.array([0.4347305389, 0.6790419162]), np.array([0.5588890164, -0.0404108803]))
TOY_JOINT_MEAN = np.array(
    [0.4459332838, 0.6669945043, 0.5244110404, -0.1678802503, 0.5050315562, 0.1254023495]
)
TOY_JOINT_SD = np.array(
    [0.4950517687, 0.3086496251, 0.4314114021, 0.3527802469, 0.5687897169, 0.4303835395]
)


def toy_state_a() -> np.ndarray:
    """The toy ``A`` in the state layout ``(n, n−1)``."""
    return state_from_unit_lower(TOY_A)


# --------------------------------------------------------------------------- layouts


def unit_lower(a: np.ndarray) -> np.ndarray:
    """``A = I + strict-lower(a)`` from the state layout ``(n, n−1)`` (entries ``j ≥ i`` ignored)."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = a[i, :i]
    return A


def state_from_unit_lower(A: np.ndarray) -> np.ndarray:
    """The ``(n, n−1)`` state layout of a unit lower-triangular ``A``."""
    n = A.shape[0]
    a = np.zeros((n, max(n - 1, 0)))
    for i in range(n):
        a[i, :i] = A[i, :i]
    return a


def vec(Pi: np.ndarray) -> np.ndarray:
    """``vec Π`` stacking columns: ``(π_1', …, π_n')'``."""
    return np.asarray(Pi).T.reshape(-1)


def unvec(v: np.ndarray, k: int, n: int) -> np.ndarray:
    return np.asarray(v).reshape(n, k).T


# --------------------------------------------------------------------------- SUR oracle


def sur_system(
    a: np.ndarray,
    lam: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mean0: np.ndarray,
    prec0: np.ndarray,
    tmask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint precision ``Q`` (``nk × nk``) and linear term ``b`` of ``vec Π`` (module docstring).

    ``lam`` (``n × T``) are variances; rows with ``tmask_t = 0`` are skipped (padding)."""
    A = unit_lower(a)
    T, k = X.shape
    n = Y.shape[1]
    mask = np.ones(T) if tmask is None else np.asarray(tmask)
    Q = np.zeros((n * k, n * k))
    b = np.zeros(n * k)
    for j in range(n):
        Q[j * k : (j + 1) * k, j * k : (j + 1) * k] += np.diag(prec0[j])
        b[j * k : (j + 1) * k] += prec0[j] * mean0[j]
    for t in range(T):
        if mask[t] <= 0:
            continue
        Om = A.T @ np.diag(1.0 / lam[:, t]) @ A
        x = X[t]
        Q += np.kron(Om, np.outer(x, x))
        b += np.kron(Om @ Y[t], x)
    return 0.5 * (Q + Q.T), b


def conditional_j(
    Q: np.ndarray, b: np.ndarray, Pi: np.ndarray, j: int
) -> tuple[np.ndarray, np.ndarray]:
    """Precision ``Q_jj`` and mean ``Q_jj⁻¹(b_j − Σ_{l≠j} Q_jl π_l)`` of ``π_j | π_{−j}``."""
    k, n = Pi.shape
    sl = slice(j * k, (j + 1) * k)
    rest = b[sl].copy()
    for m in range(n):
        if m != j:
            rest -= Q[sl, m * k : (m + 1) * k] @ Pi[:, m]
    K = Q[sl, sl]
    return K, np.linalg.solve(K, rest)


def joint_moments(Q: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``Q⁻¹b`` and covariance ``Q⁻¹`` of the joint Gaussian (Cholesky solves)."""
    c = sla.cho_factor(Q, lower=True)
    cov = sla.cho_solve(c, np.eye(Q.shape[0]))
    return sla.cho_solve(c, b), 0.5 * (cov + cov.T)


def scaled_draw(K: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``s ⊙ L⁻ᵀ(L⁻¹(s ⊙ b) + z)``, ``s = diag(K)^{−½}``, ``L = chol(sKs)`` (D-14 algebra)."""
    s = 1.0 / np.sqrt(np.diag(K))
    L = np.linalg.cholesky(s[:, None] * K * s[None, :])
    v = sla.solve_triangular(L, s * b, lower=True)
    return s * sla.solve_triangular(L.T, v + z, lower=False)


def cta_sweep_reference(
    Pi: np.ndarray,
    a: np.ndarray,
    lam: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mean0: np.ndarray,
    prec0: np.ndarray,
    Z: np.ndarray,
    tmask: np.ndarray | None = None,
) -> np.ndarray:
    """One Gibbs scan ``j = 0..n−1`` over the SUR conditionals with injected noise ``Z``."""
    Q, b = sur_system(a, lam, X, Y, mean0, prec0, tmask)
    Pi = np.array(Pi, dtype=np.float64)
    for j in range(Pi.shape[1]):
        K, mu = conditional_j(Q, b, Pi, j)
        Pi[:, j] = scaled_draw(K, K @ mu, Z[j])
    return Pi


def block_gauss_seidel(Q: np.ndarray, k: int, n: int) -> np.ndarray:
    """``−(D + L)⁻¹U`` for the equation-block splitting ``Q = D + L + U`` (N04 §9.2)."""
    lower = np.zeros_like(Q)
    for j in range(n):
        for m in range(j + 1):
            lower[j * k : (j + 1) * k, m * k : (m + 1) * k] = Q[
                j * k : (j + 1) * k, m * k : (m + 1) * k
            ]
    return -np.linalg.solve(lower, Q - lower)


# --------------------------------------------------------------------------- CCM 2019 (the bug)


def ccm2019_conditional_j(
    Pi: np.ndarray,
    a: np.ndarray,
    lam: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mean0: np.ndarray,
    prec0: np.ndarray,
    j: int,
    tmask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CCM 2019 ``π_j`` step (N04 §4): equation ``j`` alone with ``y*_j = y_j + Σ_{i<j} a_ji u_i``,
    ``u_i = y_i − Xπ_i`` from the current ``Π`` (in a sweep: the already updated ``π_{<j}``)."""
    A = unit_lower(a)
    mask = np.ones(X.shape[0]) if tmask is None else np.asarray(tmask)
    w = mask / np.where(mask > 0, lam[j], 1.0)
    U = Y - X @ Pi
    ystar = Y[:, j] + U[:, :j] @ A[j, :j]
    K = np.diag(prec0[j]) + X.T @ (w[:, None] * X)
    bb = prec0[j] * mean0[j] + X.T @ (w * ystar)
    return K, np.linalg.solve(K, bb)


def ccm2019_sweep(
    Pi: np.ndarray,
    a: np.ndarray,
    lam: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mean0: np.ndarray,
    prec0: np.ndarray,
    Z: np.ndarray,
    tmask: np.ndarray | None = None,
) -> np.ndarray:
    """The CCM 2019 "triangular" draw for ``j = 0..n−1`` with injected noise (ignores ``Π_old``)."""
    Pi = np.array(Pi, dtype=np.float64)
    for j in range(Pi.shape[1]):
        K, mu = ccm2019_conditional_j(Pi, a, lam, X, Y, mean0, prec0, j, tmask)
        Pi[:, j] = scaled_draw(K, K @ mu, Z[j])
    return Pi


# --------------------------------------------------------------------------- SWA Gibbs (A7)


@dataclass(frozen=True)
class ReducedConstant:
    """Reduced form × Constant (N04 §2, §3; DESIGN §6.2): ``π_j ~ N(m_j, diag 1/p_j)``,
    ``a_i ~ N(m_a, diag 1/p_a)`` on active slots ``j < i``, ``λ_i ~ IG(shape_i, scale_i)``."""

    X: np.ndarray  # (T, k)
    Y: np.ndarray  # (T, n)
    mean0: np.ndarray  # (n, k)
    prec0: np.ndarray  # (n, k)
    a_mean0: np.ndarray  # (n, n−1)
    a_prec0: np.ndarray  # (n, n−1)
    shape: np.ndarray  # (n,)
    scale: np.ndarray  # (n,)

    @property
    def n(self) -> int:
        return self.Y.shape[1]

    @property
    def k(self) -> int:
        return self.X.shape[1]


def batched_unit_lower(a: np.ndarray) -> np.ndarray:
    """``A`` for a batch of state layouts ``(C, n, n−1)``."""
    C, n = a.shape[0], a.shape[1]
    full = np.concatenate([a, np.zeros((C, n, 1))], axis=2)
    return np.eye(n)[None] + np.tril(full, k=-1)


def swa_pi_step(
    prob: ReducedConstant, Pi: np.ndarray, a: np.ndarray, log_lam: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """SWA: one joint draw of ``vec Π`` per chain (Constant ``λ``: ``Ω`` constant over ``t``).

    ``Pi`` (C, k, n) is ignored (an exact draw); ``a`` (C, n, n−1), ``log_lam`` (C, n), ``Z``
    (C, n, k) standard normals (row ``j`` drives ``π_j`` through ``L⁻ᵀ``)."""
    C = a.shape[0]
    n, k = prob.n, prob.k
    A = batched_unit_lower(a)
    Om = np.einsum("cij,ci,cil->cjl", A, np.exp(-log_lam), A)  # A'Λ⁻¹A
    XtX = prob.X.T @ prob.X
    Q = np.einsum("cjl,rs->cjrls", Om, XtX).reshape(C, n * k, n * k)
    Q[:, np.arange(n * k), np.arange(n * k)] += prob.prec0.reshape(-1)[None]
    XtY = prob.X.T @ prob.Y  # (k, n)
    rhs = (
        np.einsum("rm,cmj->cjr", XtY, Om).reshape(C, n * k)
        + (prob.prec0 * prob.mean0).reshape(-1)[None]
    )
    L = np.linalg.cholesky(0.5 * (Q + np.swapaxes(Q, 1, 2)))
    v = np.linalg.solve(L, rhs[..., None])
    theta = np.linalg.solve(np.swapaxes(L, 1, 2), v + Z.reshape(C, n * k, 1))[..., 0]
    return np.swapaxes(theta.reshape(C, n, k), 1, 2)


def a_step(
    prob: ReducedConstant, rng: np.random.Generator, Pi: np.ndarray, log_lam: np.ndarray
) -> np.ndarray:
    """Rows ``a_j | Π, λ_j``: regression of ``u_j`` on ``−u_{<j}`` with weight ``1/λ_j``
    (N04 §6.2, minus sign); batched over chains. Returns (C, n, n−1)."""
    C = Pi.shape[0]
    n = prob.n
    U = prob.Y[None] - np.einsum("tk,ckn->ctn", prob.X, Pi)
    a = np.zeros((C, n, max(n - 1, 0)))
    for j in range(1, n):
        R = -U[:, :, :j]  # (C, T, j)
        w = np.exp(-log_lam[:, j])[:, None, None]
        K = np.einsum("ctr,cts->crs", R, R) * w + np.diag(prob.a_prec0[j, :j])[None]
        bb = np.einsum("ctr,ct->cr", R, U[:, :, j]) * w[:, :, 0] + (
            prob.a_prec0[j, :j] * prob.a_mean0[j, :j]
        )
        L = np.linalg.cholesky(K)
        mu = np.linalg.solve(K, bb[..., None])[..., 0]
        z = rng.standard_normal((C, j, 1))
        a[:, j, :j] = mu + np.linalg.solve(np.swapaxes(L, 1, 2), z)[..., 0]
    return a


def lam_step(
    prob: ReducedConstant, rng: np.random.Generator, Pi: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """``λ_i | Π, a ~ IG(shape_i + T/2, scale_i + ½Σ_t e_it²)``, ``e_t = A(y_t − Π'x_t)``;
    returns ``log λ`` (C, n)."""
    T = prob.X.shape[0]
    U = prob.Y[None] - np.einsum("tk,ckn->ctn", prob.X, Pi)
    E = np.einsum("ctl,cil->cti", U, batched_unit_lower(a))
    shape = prob.shape + 0.5 * T
    scale = prob.scale[None] + 0.5 * np.sum(E * E, axis=1)
    g = rng.gamma(np.broadcast_to(shape, scale.shape))
    return np.log(scale) - np.log(g)


PiStep = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def gibbs(
    prob: ReducedConstant,
    pi_step: PiStep,
    init: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    burn: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """``C`` independent chains of the Gibbs sampler ``Π → a → λ`` (DESIGN §4.3 order).

    ``pi_step(Pi, a, log_lam, Z) -> Pi`` is the ``Π`` block (SWA or the CTA kernel); ``Z``
    (C, n, k) comes from this sampler's own generator, so both samplers are driven by NumPy
    noise. ``init = (Pi (C, k, n), a (C, n, n−1), log_lam (C, n))``. Returns the kept draws
    ``Pi (C, D, k, n)``, ``a (C, D, n, n−1)``, ``log_lam (C, D, n)``."""
    rng = np.random.default_rng(seed)
    Pi, a, log_lam = (np.array(x, dtype=np.float64) for x in init)
    C, n, k = Pi.shape[0], prob.n, prob.k
    out = {
        "Pi": np.empty((C, draws, k, n)),
        "a": np.empty((C, draws, n, max(n - 1, 0))),
        "log_lam": np.empty((C, draws, n)),
    }
    for it in range(burn + draws):
        Pi = np.asarray(pi_step(Pi, a, log_lam, rng.standard_normal((C, n, k))))
        a = a_step(prob, rng, Pi, log_lam)
        log_lam = lam_step(prob, rng, Pi, a)
        if it >= burn:
            d = it - burn
            out["Pi"][:, d], out["a"][:, d], out["log_lam"][:, d] = Pi, a, log_lam
    return out


__all__ = [
    "TOY_A",
    "TOY_CCM_K",
    "TOY_CCM_MEAN",
    "TOY_JOINT_MEAN",
    "TOY_JOINT_SD",
    "TOY_K",
    "TOY_LAM",
    "TOY_M",
    "TOY_MEAN",
    "TOY_PI",
    "TOY_V",
    "TOY_X",
    "TOY_Y",
    "ReducedConstant",
    "a_step",
    "batched_unit_lower",
    "block_gauss_seidel",
    "ccm2019_conditional_j",
    "ccm2019_sweep",
    "conditional_j",
    "cta_sweep_reference",
    "gibbs",
    "joint_moments",
    "lam_step",
    "scaled_draw",
    "state_from_unit_lower",
    "sur_system",
    "swa_pi_step",
    "toy_state_a",
    "unit_lower",
    "unvec",
    "vec",
]
