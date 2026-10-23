"""Linear-algebra kernels of the MCMC engine (DESIGN §3.7, D-14, D-25, D-26; N05 §7; N07 §8).

Every function solves **one** problem and is batched by ``jax.vmap`` (chains × equations); none
reads a key (the noise ``z`` is injected, D-17), forms an explicit inverse, or adds jitter
(SPEC §8). Failures are values: a non-positive pivot or a non-finite result sets
``StepInfo.ok = False``; nothing raises inside ``jit`` (D-18).

* **Weighted Gram** (D-25). ``G = Fᵀ diag(w) F``. ``packed``: ``G = (w @ FF)[pos]`` with the
  packed Khatri–Rao products ``FF[t, c] = F[t, r_c]·F[t, s_c]`` (``r_c ≤ s_c``) built once per fit;
  under ``vmap`` one GEMM ``(C·n, T) × (T, P)``. ``packed_tiled``: the same product built tile by
  tile from ``F`` (``FF`` never stored; DESIGN §10.5). ``batched``: ``Fᵀ(w ⊙ F)``, lower triangle
  mirrored (the oracle). The packed order is column-major (pairs sorted by ``s`` then ``r``,
  ``c = s(s+1)/2 + r``), identical to ``data.packed_index`` (WP5; pilot §10).
* **Precision-form Gaussian draw** (D-14; N05 §7.2; N07 §8.2). With ``s = diag(K)^{−½}`` and
  ``L = chol(sKs)``: ``θ = s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z) ~ N(K⁻¹b, K⁻¹)`` for ``z ~ N(0, I)``, because
  ``s(sKs)⁻¹s = K⁻¹``. The Jacobi scaling is an exact change of variables, not jitter. Only the
  lower triangle of ``K`` is read. Pivots are ``diag(L)²`` of the scaled matrix (in ``(0, 1]``).
* **Constant-volatility fast paths** (D-26; N05 §7.4; DESIGN §3.7.2). Mode I (eigen):
  ``K(λ) = D + Gm/λ = D^{½}(I + G̃/λ)D^{½}`` with ``G̃ = D^{−½}GmD^{−½} = Q diag(g) Qᵀ`` computed
  once per fit, so each sweep costs ``O(q²)``. Mode II (fixed factor): ``K = K̃/λ`` with ``K̃``
  λ-free, so ``θ = θ̂ + √λ · s ⊙ Lc⁻ᵀz`` with one factorisation per fit.
* **Tridiagonal precision sampler** (N07 §8.3 S4; N02 §5.5): bidiagonal Cholesky fused with the
  forward solve in one ``lax.scan``, back substitution in a reverse scan; ``log|Q|`` for free.
  For a path-graph Laplacian plus a diagonal (the RW AWOL system) the pivots are carried as
  their excess over the next edge, which never cancels (:func:`laplacian_precision_sample`).
* **Stable AR(1) regression** (N02 §6.1): ``b_T`` and ``chol(B_T)`` of the 2×2 auxiliary posterior
  from **centred** cross-products (the uncentred form loses up to 5 digits; N02 cross-check 1).

**Pivots.** ``min_pivot`` is the smallest factorisation pivot (``l_t²``). A NaN pivot (failed or
poisoned factorisation; JAX fills a failed Cholesky with NaN) counts as 0, so ``min_pivot ≤ 0``
exactly when the factorisation failed and the value itself stays finite.
"""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.state import EigenCache, FixedFactorCache, StepInfo

Array = jax.Array

GramMode = Literal["packed", "packed_tiled", "batched"]

_EIG_NEG_TOL = 1e-10  # eigen path: eigenvalues ≥ −1e-10·max|eigenvalue| (DESIGN §3.7.2)

__all__ = [
    "eigen_draw",
    "eigen_prepare",
    "eigen_prepare_dense",
    "fixed_factor_draw",
    "fixed_factor_prepare",
    "gram_from_packed",
    "laplacian_precision_sample",
    "lstsq_draw",
    "precision_draw",
    "precision_draw_factored",
    "precision_factor",
    "precision_moments",
    "solve_unit_lower",
    "stable_ar1_regression",
    "tridiag_precision_sample",
    "weighted_gram",
]


# --------------------------------------------------------------------------- helpers


def _min_pivot(piv: Array) -> Array:
    """Smallest pivot; NaN pivots count as 0 (a failed factorisation), so the result is finite."""
    return jnp.min(jnp.where(jnp.isnan(piv), 0.0, piv))


def _solve_lower(L: Array, b: Array, *, transpose: bool = False) -> Array:
    """``L⁻¹b`` (or ``L⁻ᵀb``) for lower-triangular ``L`` and a vector ``b`` (explicit ``(m, 1)``
    right-hand side, so a batched call is never an ambiguous 1-D solve; N07 §8.1)."""
    x = lax.linalg.triangular_solve(
        L, b[:, None], left_side=True, lower=True, transpose_a=transpose
    )
    return x[:, 0]


@functools.cache
def _packed_index(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-major packed maps ``(iu, ju, pos)`` (NumPy int32, read-only; built at trace time).

    Packed column ``c = s(s+1)/2 + r`` holds ``(r, s) = (iu_c, ju_c)``, ``r ≤ s``;
    ``pos[r, s] = pos[s, r] = c``. Same order as ``data.packed_index`` (WP5).
    """
    if q < 1:
        raise ValueError(f"q must be a positive integer, got {q}")
    iu = np.concatenate([np.arange(s + 1) for s in range(q)]).astype(np.int32)
    ju = np.concatenate([np.full(s + 1, s) for s in range(q)]).astype(np.int32)
    pos = np.zeros((q, q), dtype=np.int32)
    pos[iu, ju] = np.arange(iu.size, dtype=np.int32)
    pos[ju, iu] = pos[iu, ju]
    for a in (iu, ju, pos):
        a.setflags(write=False)
    return iu, ju, pos


# --------------------------------------------------------------------------- weighted Gram (D-25)


def gram_from_packed(v: Array, q: int) -> Array:
    """The symmetric ``q × q`` matrix ``v[pos]`` of a packed vector ``v`` (length ``q(q+1)/2``)."""
    q = int(q)
    P = q * (q + 1) // 2
    if v.shape[-1] != P:
        raise ValueError(f"packed vector has length {v.shape[-1]}, expected q(q+1)/2 = {P}")
    _, _, pos = _packed_index(q)
    return v[..., pos]


def weighted_gram(
    w: Array,
    *,
    F: Array | None = None,
    FF: Array | None = None,
    mode: GramMode,
    q: int,
    tile: int = 0,
) -> Array:
    """``G = Fᵀ diag(w) F`` (``q × q``, exactly symmetric) for weights ``w`` (``T``).

    ``mode="packed"`` needs ``FF`` (``T × P``, column-major packed products): ``(w @ FF)[pos]``.
    ``mode="packed_tiled"`` needs ``F`` and ``tile > 0`` packed columns per tile: each tile of
    ``FF`` is formed from ``F`` and contracted at once (memory ``T·tile``). ``mode="batched"``
    needs ``F``: ``Fᵀ(w ⊙ F)`` with its lower triangle mirrored. All three are deterministic; they
    agree to rounding (DESIGN §9.3 item 1).
    """
    q = int(q)
    P = q * (q + 1) // 2
    if mode == "packed":
        if FF is None:
            raise ValueError('weighted_gram(mode="packed") needs the packed products FF')
        if FF.shape[-1] != P:
            raise ValueError(f"FF has {FF.shape[-1]} packed columns, expected q(q+1)/2 = {P}")
        return gram_from_packed(w @ FF, q)
    if mode == "packed_tiled":
        if F is None:
            raise ValueError('weighted_gram(mode="packed_tiled") needs the design F')
        if F.shape[-1] != q:
            raise ValueError(f"F has {F.shape[-1]} columns, expected q = {q}")
        tile = int(tile)
        if tile <= 0:
            raise ValueError(f'mode="packed_tiled" needs tile > 0, got {tile}')
        iu, ju, _ = _packed_index(q)
        parts = [w @ (F[:, iu[c : c + tile]] * F[:, ju[c : c + tile]]) for c in range(0, P, tile)]
        return gram_from_packed(jnp.concatenate(parts), q)
    if mode == "batched":
        if F is None:
            raise ValueError('weighted_gram(mode="batched") needs the design F')
        if F.shape[-1] != q:
            raise ValueError(f"F has {F.shape[-1]} columns, expected q = {q}")
        G = F.T @ (w[:, None] * F)
        return jnp.tril(G) + jnp.tril(G, -1).T
    raise ValueError(f"unknown Gram mode {mode!r}; expected packed, packed_tiled or batched")


# --------------------------------------------------------------------------- precision draw (D-14)


def precision_factor(K: Array) -> tuple[Array, Array, StepInfo]:
    """Jacobi-scaled Cholesky ``(L, s, info)``: ``s = diag(K)^{−½}``, ``LLᵀ = sKs`` (lower triangle
    of ``K`` read). ``info.ok`` iff every ``L_jj`` is finite and positive; pivots ``L_jj²``."""
    s = 1.0 / jnp.sqrt(jnp.diagonal(K))
    L = lax.linalg.cholesky(s[:, None] * K * s[None, :], symmetrize_input=False)
    dL = jnp.diagonal(L)
    ok = jnp.all(jnp.isfinite(dL) & (dL > 0.0))
    return L, s, StepInfo(ok, _min_pivot(dL * dL))


def precision_draw_factored(L: Array, s: Array, b: Array, z: Array) -> Array:
    """``θ = s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z)`` from a :func:`precision_factor` of ``K`` (two solves)."""
    v = _solve_lower(L, s * b)
    return s * _solve_lower(L, v + z, transpose=True)


def precision_draw(K: Array, b: Array, z: Array) -> tuple[Array, StepInfo]:
    """``θ ~ N(K⁻¹b, K⁻¹)`` for ``z ~ N(0, I)`` (D-14), with ``ok`` also requiring a finite ``θ``.

    Padding is exact (D-15): trailing inactive slots with unit precision, zero cross-precision and
    ``b = 0`` give a block-diagonal ``L``, so the active draw equals the unpadded draw with
    ``z[:q_i]`` and the padded coordinates equal ``z`` (the caller multiplies by the mask).
    """
    L, s, info = precision_factor(K)
    theta = precision_draw_factored(L, s, b, z)
    return theta, StepInfo(info.ok & jnp.all(jnp.isfinite(theta)), info.min_pivot)


def lstsq_draw(A: Array, rhs: Array, z: Array) -> tuple[Array, StepInfo]:
    """``θ ~ N(K⁻¹b, K⁻¹)`` with ``K = AᵀA`` and ``b = Aᵀrhs`` **without forming K** (whitened
    regression rows ``A``, e.g. ``[diag(√prec); √w ⊙ Z]``), for injected ``z ~ N(0, I)``.

    Householder QR ``A = QR`` with ``diag(R) > 0`` makes ``R`` the upper Cholesky factor of ``K``,
    so ``θ = R⁻¹(Qᵀrhs + z)`` is the same function of ``z`` as :func:`precision_draw` ``(K, b, z)``
    (``= K⁻¹b + R⁻¹z``) up to rounding, but the condition number of ``A`` (``√cond(K)``) is never
    squared: :func:`precision_draw` loses ``K``'s small eigenvalues once ``cond(K) ≳ 1/ε``
    (the contemporaneous Gram of a ~1e8-scale regressor next to an O(1) one), this does not.
    ``min_pivot = min_j (R_jj/‖A_·j‖)²``: the pivots of the Jacobi-scaled ``K`` (as
    :func:`precision_factor` reports them). ``ok`` iff ``diag(R) > 0`` and ``θ`` is finite.
    Padding is exact as in :func:`precision_draw` (unit prior rows, zero data columns)."""
    Q, R = jnp.linalg.qr(A, mode="reduced")
    sign = jnp.where(jnp.diagonal(R) < 0.0, -1.0, 1.0)
    R = sign[:, None] * R
    c = sign * jnp.matmul(Q.T, rhs, precision=lax.Precision.HIGHEST)
    theta = solve_triangular(R, c + z, lower=False)
    dR = jnp.diagonal(R)
    norms = jnp.sqrt(jnp.sum(A * A, axis=0))
    ok = jnp.all(jnp.isfinite(dR) & (dR > 0.0)) & jnp.all(jnp.isfinite(theta))
    return theta, StepInfo(ok, _min_pivot((dR / norms) ** 2))


def precision_moments(K: Array, b: Array) -> tuple[Array, Array]:
    """Mean ``K⁻¹b`` and covariance ``K⁻¹ = s L⁻ᵀL⁻¹ s`` (exactly symmetric) through the same
    factor (tests, failure replay)."""
    L, s, _ = precision_factor(K)
    mean = precision_draw_factored(L, s, b, jnp.zeros_like(b))
    Linv = lax.linalg.triangular_solve(
        L, jnp.eye(L.shape[0], dtype=L.dtype), left_side=True, lower=True
    )
    C = s[:, None] * (Linv.T @ Linv) * s[None, :]
    return mean, jnp.tril(C) + jnp.tril(C, -1).T


def solve_unit_lower(A: Array, B: Array) -> Array:
    """``A⁻¹B`` for unit lower-triangular ``A`` (diagonal not read; N05 §8: no inverses).

    ``B`` is a vector ``(n,)`` or a matrix ``(n, m)``.
    """
    B2 = B[:, None] if B.ndim == 1 else B
    X = lax.linalg.triangular_solve(A, B2, left_side=True, lower=True, unit_diagonal=True)
    return X[:, 0] if B.ndim == 1 else X


# ------------------------------------------------------------------- Constant fast paths (D-26)


def eigen_prepare(G: Array, d: Array, mean0: Array, fty: Array, cmask: Array) -> EigenCache:
    """Per-fit cache of the Mode I × Constant path (one equation; DESIGN §3.7.2, N05 §7.4).

    Inputs: ``G = Fᵀ D_m F`` (unmasked), ``d = precθ`` (λ-free prior precision, 1 on inactive
    slots), ``mean0 = meanθ``, ``fty = Fᵀ D_m y_i``, ``cmask = cmaskθ``. With
    ``G̃ = d^{−½}(cmask⊗cmask ⊙ G)d^{−½} = Q diag(g) Qᵀ``:
    ``u1 = Qᵀ(d^{½} ⊙ mean0)``, ``u2 = Qᵀ(d^{−½} ⊙ cmask ⊙ fty)``, ``d_half = d^{½}``.

    ``G̃`` is graded: the default ``var_const = 1e7`` intercept prior makes its diagonal entry
    ``~1e9`` times the others, and a plain ``eigh`` (absolute eigenvalue error ``ε‖G̃‖``) then
    loses up to 7 digits in the other directions. The decomposition is therefore computed from a
    graded factor ``G̃ = BBᵀ``: with ``δ = diag(G̃)^{½}`` and the unit-diagonal correlation
    ``A = δ⁻¹G̃δ⁻¹``, ``Bᵀ = Lᵀ ⊙ δ`` for ``A = LLᵀ`` (Cholesky: componentwise backward error, so
    ``BBᵀ = G̃ + E``, ``|E_rs| ≤ cε·δ_rδ_s``). When the Cholesky breaks down (``A`` numerically
    singular: ``T < q``, collinear regressors) ``Lᵀ`` is replaced by ``diag(√a⁺)Vᵀ`` from
    ``A = V diag(a) Vᵀ``. ``Bᵀ`` is column-graded, so its columns are sorted by decreasing ``δ``,
    Householder QR gives ``Bᵀ = Q_r R``, and the SVD of the row-graded ``Rᵀ = U S Wᵀ`` by QR
    iteration (not divide and conquer, which loses the small singular values of graded
    matrices) gives ``Q = U`` with the rows unsorted and ``g = S²``.

    Accuracy (DESIGN §9.3 item 1). Observed, not proven: errors in posterior standard deviations
    against a 40–45-digit solve of the float64 ``K``, over about 1,500 random Minnesota level-data
    problems (4,400 problem–λ pairs: levels 0–1e4, AR roots 0.3–0.9995, ``var_const = 1e7``,
    ``q = 4–30`` plus the ``q = 100`` reference shape, ``T = 12–300`` with ``T < q`` included,
    ``λ`` from ``0.01ψ`` to ``100ψ``). With ``κ = cond(sKs)`` the condition number of the
    Jacobi-scaled posterior precision ``K = diag(d) + cmask⊗cmask ⊙ G/λ`` on the active block and
    ``t = max_k |E θ_k|/sd_k`` its largest posterior t-ratio, the covariance error is
    ``≤ 1e-12 + 0.8εκ`` and the mean error ``≤ 1e-12 + 1.5εκ(1 + t)``. The mean carries a relative
    error of order ``εκ``, so in sd units it grows with ``t``, which reaches 1e3 on level data
    (intercepts and own lags hundreds of sd from 0) while ``κ`` reaches 1e9–1e11. A plain
    ``1e-12 + cεκ`` bound on the mean would need ``c ≈ 470``; the largest mean error seen is
    ``9e-5`` sd (statistically negligible). The cholesky_const path (:func:`precision_draw`) stays within half
    of the same envelope; its mean error also grows with ``t`` (``39εκ`` at ``t = 696``). So the
    two paths agree to ``1e-12`` only when ``εκ(1 + t)`` is small: they are ``6.9e-12`` apart at
    ``κ = 374``, ``t = 259``. Path-vs-path tests should compare each path with a high-precision
    reference, or allow ``2e-12 + 5εκ(1 + t)`` for the mean and ``2e-12 + 5εκ`` for the covariance.

    If ``A`` is not PSD (Cholesky fails and ``min a < −1e-10·max|a|``) or anything is non-finite,
    ``g`` is NaN, so :func:`eigen_draw` reports ``ok = False`` at the first sweep (D-18).
    """
    d_half = jnp.sqrt(d)
    Gt = (cmask[:, None] * G * cmask[None, :]) / (d_half[:, None] * d_half[None, :])
    Q, g = _graded_eigh(Gt)
    u1 = Q.T @ (d_half * mean0)
    u2 = Q.T @ (cmask * fty / d_half)
    return EigenCache(Q=Q, g=g, d_half=d_half, u1=u1, u2=u2)


def eigen_prepare_dense(G: Array, P: Array, b0: Array, fty: Array, cmask: Array) -> EigenCache:
    """:func:`eigen_prepare` for a **dense** λ-free prior precision (Mode I × Constant with SOC/SUR
    dummy rows; WP25, N05 §7.7: "the eigen trick then needs a dense ``P₀^{−½}``").

    Inputs: ``G = Fᵀ D_m F`` (unmasked), ``P`` the dense λ-free prior precision (``diag(precθ) +
    w_d(cmask⊗cmask ⊙ F_dᵀF_d)``; 1 on the diagonal of inactive slots), ``b0 = P μ`` its linear
    term (``precθ ⊙ meanθ + w_d cmask ⊙ F_dᵀy_d``), ``fty = Fᵀ D_m y_i``, ``cmask = cmaskθ``. With
    ``P = L_P L_Pᵀ`` and ``G̃ = L_P⁻¹(cmask⊗cmask ⊙ G)L_P⁻ᵀ = Q diag(g) Qᵀ`` (the graded
    decomposition of :func:`eigen_prepare`): ``K = L_P(I + G̃/λ)L_Pᵀ``, so
    ``θ = L_P⁻ᵀQ(v ⊙ u1 + u2/(λ + g) + √v ⊙ z)`` with ``u1 = QᵀL_P⁻¹b0``, ``u2 = QᵀL_P⁻¹(cmask ⊙ fty)``.
    The cache stores ``L_P⁻ᵀQ`` as ``Q`` and ``d_half = 1``, so :func:`eigen_draw` is unchanged
    (``O(q²)`` per sweep). A failed Cholesky of ``P`` leaves NaN in ``g`` (``ok = False``).
    """
    q = P.shape[0]
    L = lax.linalg.cholesky(0.5 * (P + P.T), symmetrize_input=False)
    Gm = cmask[:, None] * G * cmask[None, :]
    W = solve_triangular(L, Gm, lower=True)  # L⁻¹ Gm
    Gt = solve_triangular(L, W.T, lower=True)  # L⁻¹ Gm L⁻ᵀ (Gm symmetric)
    Gt = 0.5 * (Gt + Gt.T)
    Q, g = _graded_eigh(Gt)
    g = jnp.where(jnp.all(jnp.isfinite(L)), g, jnp.nan)
    u1 = Q.T @ solve_triangular(L, b0, lower=True)
    u2 = Q.T @ solve_triangular(L, cmask * fty, lower=True)
    Qc = solve_triangular(L.T, Q, lower=False)  # L⁻ᵀ Q
    return EigenCache(Q=Qc, g=g, d_half=jnp.ones((q,), Qc.dtype), u1=u1, u2=u2)


def _graded_eigh(Gt: Array) -> tuple[Array, Array]:
    """``Gt = Q diag(g) Qᵀ`` by the graded factorisation of :func:`eigen_prepare` (``g`` NaN when
    ``Gt`` is not PSD)."""
    delta = jnp.sqrt(jnp.diagonal(Gt))
    live = delta > 0.0
    dpos = jnp.where(live, delta, 1.0)
    A = Gt / (dpos[:, None] * dpos[None, :])
    # zero slots (inactive or an all-zero regressor) get a unit pivot; their column of Bᵀ is 0
    L = lax.linalg.cholesky(A + jnp.diag(jnp.where(live, 0.0, 1.0)), symmetrize_input=False)
    chol_ok = jnp.all(jnp.isfinite(L))
    V, a = lax.linalg.eigh(A, symmetrize_input=True)
    psd = chol_ok | (jnp.all(a >= -_EIG_NEG_TOL * jnp.max(jnp.abs(a))) & jnp.all(jnp.isfinite(a)))
    Ft = jnp.where(chol_ok, L.T, jnp.sqrt(jnp.maximum(a, 0.0))[:, None] * V.T)
    order = jnp.argsort(-delta, stable=True)
    _, R = jnp.linalg.qr((Ft * delta[None, :])[:, order])  # Bᵀ with columns sorted
    U, sv, _ = lax.linalg.svd(R.T, full_matrices=False, algorithm=lax.linalg.SvdAlgorithm.QR)
    Q = jnp.zeros_like(U).at[order].set(U)
    g = jnp.where(psd, sv * sv, jnp.nan)
    return Q, g


def eigen_draw(c: EigenCache, lam: Array, z: Array) -> tuple[Array, StepInfo]:
    """Mode I × Constant draw in ``O(q²)`` (no factorisation; N05 §7.4):
    ``θ = d^{−½} ⊙ Q(v ⊙ u1 + u2/(λ + g) + √v ⊙ z)``, ``v = λ/(λ + g)``, ``g`` clipped at 0.

    Exact: ``K = d^{½}(I + G̃/λ)d^{½}``, so ``K⁻¹ = d^{−½}Q diag(v) Qᵀd^{−½}`` and
    ``K⁻¹r = d^{−½}Q(v ⊙ (u1 + u2/λ))`` for ``r = d ⊙ mean0 + cmask ⊙ fty/λ``. ``λ`` is the variance
    (not its log). Unmasked: the caller multiplies by ``cmaskθ``. ``min_pivot = +inf``.
    """
    g = jnp.maximum(c.g, 0.0)
    denom = lam + g
    v = lam / denom
    theta = (c.Q @ (v * c.u1 + c.u2 / denom + jnp.sqrt(v) * z)) / c.d_half
    ok = (
        jnp.all(jnp.isfinite(c.g))
        & jnp.all(c.g >= -_EIG_NEG_TOL * jnp.max(jnp.abs(c.g)))
        & jnp.all(jnp.isfinite(theta))
    )
    return theta, StepInfo(ok, jnp.full((), jnp.inf, dtype=theta.dtype))


def fixed_factor_prepare(K_tilde: Array, rhs: Array) -> FixedFactorCache:
    """Per-fit cache of the Mode II × Constant path (one equation; N05 §5.1, §7.4).

    ``K̃ = cmask⊗cmask ⊙ G + diag(prec̃θ)`` and ``rhs = prec̃θ ⊙ meanθ + cmask ⊙ fty`` are λ-free;
    ``s = diag(K̃)^{−½}``, ``Lc = chol(sK̃s)``, ``θ̂ = s ⊙ Lc⁻ᵀLc⁻¹(s ⊙ rhs)`` (the posterior mean,
    independent of λ). A failed factorisation leaves NaN in ``Lc`` and ``θ̂``, so every draw is NaN.
    """
    Lc, s, _ = precision_factor(K_tilde)
    theta_hat = precision_draw_factored(Lc, s, rhs, jnp.zeros_like(rhs))
    return FixedFactorCache(Lc=Lc, s=s, theta_hat=theta_hat)


def fixed_factor_draw(c: FixedFactorCache, lam: Array, z: Array) -> Array:
    """``θ = θ̂ + √λ · s ⊙ Lc⁻ᵀz ~ N(θ̂, λK̃⁻¹)`` (``λ`` the variance; caller masks)."""
    return c.theta_hat + jnp.sqrt(lam) * c.s * _solve_lower(c.Lc, z, transpose=True)


# --------------------------------------------------------------------------- SV kernels


def tridiag_precision_sample(
    diag: Array, off: Array, b: Array, z: Array
) -> tuple[Array, Array, Array]:
    """``x = Q⁻¹b + L⁻ᵀz ~ N(Q⁻¹b, Q⁻¹)`` for SPD tridiagonal ``Q`` (N07 §8.3 S4; N02 §5.5).

    ``diag`` (m,), ``off`` (m−1,) with ``off[t] = Q[t+1, t]``. Forward scan (``Lv = b``):
    ``c_t = off[t−1]/l_{t−1}``, pivot ``p_t = diag[t] − c_t²``, ``l_t = √p_t``,
    ``v_t = (b_t − c_t v_{t−1})/l_t``; reverse scan (``Lᵀx = w``, ``w = v + z``):
    ``x_t = (w_t − c_{t+1} x_{t+1})/l_t``.
    Returns ``(x, log|Q| = Σ log p_t, min pivot)``; ``min pivot ≤ 0`` iff ``Q`` is not PD
    (NaN pivots count as 0). Decoupled padded states (``diag = 1``, ``off = 0``, ``b = 0``) return
    their ``z`` and contribute 0 to ``log|Q|`` (D-15).
    """
    l0 = jnp.sqrt(diag[0])
    v0 = b[0] / l0

    def fwd(carry, inp):
        l_prev, v_prev = carry
        d_t, e_t, b_t = inp
        c_t = e_t / l_prev
        p_t = d_t - c_t * c_t
        l_t = jnp.sqrt(p_t)
        v_t = (b_t - c_t * v_prev) / l_t
        return (l_t, v_t), (p_t, l_t, c_t, v_t)

    _, (ps, ls, cs, vs) = lax.scan(fwd, (l0, v0), (diag[1:], off, b[1:]))
    piv = jnp.concatenate([diag[:1], ps])
    ell = jnp.concatenate([l0[None], ls])
    w = jnp.concatenate([v0[None], vs]) + z
    x_last = w[-1] / ell[-1]

    def bwd(x_next, inp):
        w_t, l_t, c_next = inp
        x_t = (w_t - c_next * x_next) / l_t
        return x_t, x_t

    _, xs = lax.scan(bwd, x_last, (w[:-1], ell[:-1], cs), reverse=True)
    x = jnp.concatenate([xs, x_last[None]])
    return x, jnp.sum(jnp.log(piv)), _min_pivot(piv)


def laplacian_precision_sample(
    delta: Array, w: Array, b: Array, z: Array
) -> tuple[Array, Array, Array]:
    """``x = Q⁻¹b + L⁻ᵀz ~ N(Q⁻¹b, Q⁻¹)`` for ``Q = Lap_w + diag(δ)``: the path-graph Laplacian
    with edge weights ``w[t−1] ≥ 0`` on ``(t−1, t)`` (``Q[t, t−1] = −w[t−1]``,
    ``Q[t, t] = δ_t + w[t−1] + w[t]``) plus ``δ ≥ 0`` (the RW AWOL system, DESIGN §3.7.4).

    The pivots are those of :func:`tridiag_precision_sample` on the assembled ``Q``, but computed
    without cancellation: with ``q_t = p_t − w[t]`` (the pivot less the next edge),
    ``q_0 = δ_0`` and ``q_t = δ_t + w[t−1]q_{t−1}/(w[t−1] + q_{t−1})`` (a series conductance), so
    every term is non-negative. Assembling ``Q`` first rounds ``δ`` away when it is below the ulp
    of the degrees (``σ²D ≪ ε``), and the pivot of the near-null level direction, ``≈ Σδ``, is
    then rounding noise; here it keeps its relative accuracy. ``log|Q| = Σ log p_t`` (``log1p``
    on unit edges). Same returns as :func:`tridiag_precision_sample`; ``δ_t = 1``, ``w = 0`` is a
    decoupled padded state.
    """
    w_next = jnp.concatenate([w, jnp.zeros((1,), w.dtype)])

    def piv_scan(q_prev, inp):
        d_t, w_t = inp
        series = jnp.where(w_t > 0.0, w_t * q_prev / (w_t + q_prev), 0.0)
        q_t = d_t + series
        return q_t, q_t

    q0 = delta[0]
    _, qs = lax.scan(piv_scan, q0, (delta[1:], w))
    q = jnp.concatenate([q0[None], qs])
    piv = q + w_next
    log_piv = jnp.where(w_next > 0.0, jnp.log(w_next) + jnp.log1p(q / w_next), jnp.log(q))
    ell = jnp.sqrt(piv)
    c = -w / ell[:-1]  # L[t, t−1]
    v0 = b[0] / ell[0]

    def fwd(v_prev, inp):
        b_t, c_t, l_t = inp
        v_t = (b_t - c_t * v_prev) / l_t
        return v_t, v_t

    _, vs = lax.scan(fwd, v0, (b[1:], c, ell[1:]))
    wv = jnp.concatenate([v0[None], vs]) + z
    x_last = wv[-1] / ell[-1]

    def bwd(x_next, inp):
        w_t, l_t, c_next = inp
        x_t = (w_t - c_next * x_next) / l_t
        return x_t, x_t

    _, xs = lax.scan(bwd, x_last, (wv[:-1], ell[:-1], c), reverse=True)
    x = jnp.concatenate([xs, x_last[None]])
    return x, jnp.sum(log_piv), _min_pivot(piv)


def stable_ar1_regression(
    x: Array, z: Array, m: Array, eps1: float, eps2: float
) -> tuple[Array, Array]:
    """Auxiliary posterior of ``(γ, φ)`` in ``z_t = γ + φx_t + σu_t`` (N02 §6.1, D-6).

    Rows with ``m_t = 1`` enter (``m`` is a 0/1 mask, ``T = Σm``; masked rows are never read, so
    they may hold anything, NaN included); ``B₀ = diag(1/ε₁, 1/ε₂)``.
    With centred sums ``S_c = Σm(x − x̄)²`` and ``S_xz = Σm(x − x̄)(z − z̄)``:
    ``A = X'X + B₀⁻¹ = [[T+ε₁, Tx̄], [Tx̄, A₂₂]]``, ``A₂₂ = S_c + Tx̄² + ε₂``,
    ``det A = T·S_c + T·ε₂ + ε₁A₂₂`` (no cancellation),
    ``b_T = (γ̂, φ̂) = ([Tz̄(S_c + ε₂) − Tx̄·S_xz], [T·S_xz + ε₁(S_xz + Tx̄z̄)]) / det``,
    ``chol(B_T) = [[√(A₂₂/det), 0], [−Tx̄/√(det·A₂₂), 1/√A₂₂]]`` with ``B_T = A⁻¹``.
    Returns ``(b_T, chol(B_T))``; the proposal is ``N(b_T, σ²B_T)``. Means are refined once
    (``x̄ = x̄₀ + Σm(x − x̄₀)/T``), which keeps the centring exact to rounding. With every row
    masked (``T = 0``) the result is the auxiliary prior: ``b_T = 0``, ``chol(B_T) =
    diag(ε₁^{−½}, ε₂^{−½})``.
    """
    on = m > 0
    T = jnp.sum(jnp.where(on, 1.0, 0.0))
    Tn = jnp.maximum(T, 1.0)  # x̄ = z̄ = 0 when no row enters
    x = jnp.where(on, x, 0.0)
    z = jnp.where(on, z, 0.0)
    xb0 = jnp.sum(x) / Tn
    zb0 = jnp.sum(z) / Tn
    xb = xb0 + jnp.sum(jnp.where(on, x - xb0, 0.0)) / Tn
    zb = zb0 + jnp.sum(jnp.where(on, z - zb0, 0.0)) / Tn
    dx = jnp.where(on, x - xb, 0.0)
    dz = jnp.where(on, z - zb, 0.0)
    S_c = jnp.sum(dx * dx)
    S_xz = jnp.sum(dx * dz)
    A12 = T * xb
    A22 = S_c + A12 * xb + eps2
    det = T * S_c + T * eps2 + eps1 * A22
    gam = (T * zb * (S_c + eps2) - A12 * S_xz) / det
    phi = (T * S_xz + eps1 * (S_xz + A12 * zb)) / det
    r22 = jnp.sqrt(A22)
    chol = jnp.stack(
        [
            jnp.stack([jnp.sqrt(A22 / det), jnp.zeros_like(det)]),
            jnp.stack([-A12 / (jnp.sqrt(det) * r22), 1.0 / r22]),
        ]
    )
    return jnp.stack([gam, phi]), chol
