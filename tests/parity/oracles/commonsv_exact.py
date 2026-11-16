"""The exact independent ``CommonSV`` oracle (docs/phase3/DESIGN.md §9.8; WP38a, Q40, E168).

A NumPy reference sampler for the composition
``BVAR(Linear(Minnesota(form="conjugate", hierarchy="fixed", …)), CommonSV(), lags=p)``, written in
reduced form from DESIGN §9.8 and SPEC §4.6 alone. It imports nothing from ``rai_bvar``: it lags
the data and builds ``Ω``, ``b``, ``Ψ`` and the SOC/SUR dummy rows in its own code, and uses NumPy
only (no SciPy is needed: every distribution below is drawn from NumPy's generators).

**Model** (§9.8; ``T`` data rows ``t = 0..T−1`` after the ``p`` presample rows, 0-based):

* ``y_t = Bᵀx_t + u_t``, ``u_t | h_t, Σ ~ N(0, e^{h_t}Σ)``, ``x_t = (1, y_{t−1}ᵀ, …, y_{t−p}ᵀ)ᵀ``;
* ``Σ ~ IW(Ψ, d)``, ``Ψ = diag(ψ)``, ``d = n + 2``; ``vec B | Σ ~ N(vec b, Σ ⊗ Ω)``, ``Ω``
  diagonal: lag ``l`` of variable ``j`` has ``λ²(d − n − 1)/(l^α ψ_j)``, the constant
  ``var_const``; ``b`` is zero except the own first lags (``means``);
* the SOC/SUR dummy rows of SPEC §4.6 (GLP 2015) with weight 1, never scaled by ``e^{h}``;
* ``h_t = ρh_{t−1} + σ_h η_t``, ``h_0 ~ N(0, σ_h²/(1 − ρ²))`` at the first data row, ``E h = 0``;
* ``ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)``, ``σ_h² ~ IG(5, 0.04)`` (shape, scale; CommonSV's defaults).

**Sampler** (one sweep = blocks 1, 2, 3 in this order; §9.8):

1. ``(B, Σ) | h, Y``: one joint matrix-normal–inverse-Wishart draw with row weights
   ``w_t = e^{−h_t}`` on the data rows and 1 on the dummy rows: ``Ω̄ = (Ω⁻¹ + X̃ᵀX̃)⁻¹``,
   ``B̄ = Ω̄(Ω⁻¹b + X̃ᵀỸ)``, ``Ψ̄ = Ψ + ỸᵀỸ + bᵀΩ⁻¹b − B̄ᵀΩ̄⁻¹B̄``, ``d̄ = d + T + m``;
   ``Σ ~ IW(Ψ̄, d̄)`` by the Bartlett decomposition, then ``B | Σ ~ MN(B̄, Ω̄, Σ)``.
2. ``h | B, Σ, ρ, σ_h, Y``: five elliptical slice sampling updates of the whole path (Murray,
   Adams & MacKay 2010) under the stationary AR(1) prior, log-likelihood
   ``Σ_t [−(n/2)h_t − ½e^{−h_t}q_t]``, ``q_t = u_tᵀΣ⁻¹u_t``.
3. ``ρ | h, σ_h`` by univariate slice sampling (Neal 2003: stepping out, width 0.1, at most 50
   steps, then shrinkage); ``σ_h² | h, ρ ~ IG(5 + T/2, 0.04 + S/2)``; then the non-centred move
   (Papaspiliopoulos, Roberts & Sköld 2007): with ``x = h/σ_h`` fixed, slice sampling (width 0.5,
   at most 50 steps) of ``s = log σ_h`` under ``−2c₀s − C₀e^{−2s} + Σ_t[−(n/2)e^{s}x_t −
   ½e^{−e^{s}x_t}q_t]``, then ``h = e^{s}x``.

Every function works on a leading batch axis of ``K`` independent chains (``K = 1`` for the
fixture runs of ``tools/make_commonsv_oracle.py``; the validation in ``test_commonsv_exact.py``
runs thousands at once). The chains share nothing but the random generator.

**Numerical guards** (pytest turns ``RuntimeWarning`` into errors): every exponential whose
argument can be large is computed as ``exp(min(a, EXP_CAP))``. A capped term is at least
``e^{600}`` in a log density that the slice thresholds compare with a finite value near the
current state's log density, so a capped state is rejected exactly as the uncapped one would be
(both are numerically −∞); no accepted state is ever affected. ``log q_t`` is ``−∞`` for
``q_t = 0`` (probability zero), which gives the correct term 0.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------- settings (§9.8)

ESS_UPDATES = 5  # elliptical slice sampling updates of the h path per sweep
RHO_WIDTH = 0.1  # slice sampler for ρ: stepping-out width ...
RHO_MAX_STEPS = 50  # ... and the maximum number of steps (Neal 2003, Fig. 3: m)
S_WIDTH = 0.5  # non-centred slice sampler for s = log σ_h: width ...
S_MAX_STEPS = 50  # ... and maximum number of steps
EXP_CAP = 600.0  # exponent cap of the numerical guards (module docstring)


# --------------------------------------------------------------------------- prior and rows


@dataclass(frozen=True)
class Prior:
    """Every prior number of one run (§9.8; SPEC §4.6; the CommonSV defaults of D-46).

    ``soc`` is ``μ`` and ``sur`` is ``δ`` of the SOC/SUR dummy rows (``None``: no such rows).
    SOC rows exist only for variables with a non-zero own-lag mean (``soc_scope="persistent"``).
    """

    psi: tuple[float, ...]
    means: tuple[float, ...]
    lam: float = 0.2
    alpha: float = 2.0
    var_const: float = 1e7
    soc: float | None = None
    sur: float | None = None
    rho_mean: float = 0.9
    rho_sd: float = 0.2
    sigma2_shape: float = 5.0  # c₀ of σ_h² ~ IG(c₀, C₀)
    sigma2_scale: float = 0.04  # C₀

    def __post_init__(self) -> None:
        if len(self.psi) != len(self.means):
            raise ValueError("psi and means must have one entry per variable")
        if min(self.psi) <= 0.0:
            raise ValueError("psi must be positive")

    @property
    def n(self) -> int:
        return len(self.psi)

    @property
    def d(self) -> int:
        """IW degrees of freedom ``d = n + 2``."""
        return self.n + 2


@dataclass(frozen=True)
class Problem:
    """The data rows, the dummy rows and the prior arrays of one run.

    ``Y`` is ``(…, T, n)`` and ``X`` is ``(…, T, k)`` with ``k = 1 + n·p`` (a leading batch axis
    only in the Geweke test, where every chain has its own simulated data); ``Yd, Xd`` are the
    ``m`` dummy rows; ``omega`` is the diagonal of ``Ω`` and ``b`` the prior mean of ``B``.
    """

    prior: Prior
    lags: int
    presample: np.ndarray  # (p, n): the p rows before the first data row
    Y: np.ndarray
    X: np.ndarray
    Yd: np.ndarray  # (m, n)
    Xd: np.ndarray  # (m, k)
    omega: np.ndarray  # (k,)
    b: np.ndarray  # (k, n)

    @property
    def n(self) -> int:
        return self.prior.n

    @property
    def k(self) -> int:
        return 1 + self.n * self.lags

    @property
    def T(self) -> int:
        return self.Y.shape[-2]

    @property
    def m(self) -> int:
        return self.Yd.shape[0]

    def with_data(self, Y: np.ndarray, X: np.ndarray) -> Problem:
        """The same prior and dummy rows with other data rows (the Geweke test's simulated data)."""
        return replace(self, Y=Y, X=X)


def lagged(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """``(Y, X)`` from ``p + T`` consecutive rows: ``Y`` the last ``T`` rows and ``X`` the rows
    ``x_t = (1, y_{t−1}ᵀ, …, y_{t−p}ᵀ)`` (the engine's regressor order, §9.8 Rows)."""
    data = np.asarray(data, dtype=np.float64)
    T, n = data.shape[0] - lags, data.shape[1]
    X = np.empty((T, 1 + n * lags))
    X[:, 0] = 1.0
    for lag in range(1, lags + 1):
        X[:, 1 + (lag - 1) * n : 1 + lag * n] = data[lags - lag : lags - lag + T]
    return data[lags:].copy(), X


def minnesota_omega(prior: Prior, lags: int) -> np.ndarray:
    """Diagonal of ``Ω``: ``var_const`` for the constant, then lag ``l`` of variable ``j``:
    ``λ²(d − n − 1)/(l^α ψ_j)`` (SPEC §4.6, §9.8 Model)."""
    n, d = prior.n, prior.d
    psi = np.asarray(prior.psi, dtype=np.float64)
    blocks = [np.array([prior.var_const])]
    for lag in range(1, lags + 1):
        blocks.append(prior.lam**2 * (d - n - 1) / (lag**prior.alpha * psi))
    return np.concatenate(blocks)


def minnesota_mean(prior: Prior, lags: int) -> np.ndarray:
    """``b``: zero except the own first lag of variable ``j`` (row ``1 + j``, column ``j``)."""
    n = prior.n
    b = np.zeros((1 + n * lags, n))
    b[1 + np.arange(n), np.arange(n)] = prior.means
    return b


def dummy_rows(prior: Prior, presample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The SOC and SUR rows of SPEC §4.6 (GLP 2015), in that order, as ``(Yd, Xd)``.

    ``ȳ₀`` is the mean of the ``p`` presample rows. One SOC row per variable ``j`` with a
    non-zero own-lag mean: ``y_d = e_jᵀȳ₀_j/μ``, ``x_d = (0, y_d, …, y_d)``; then one SUR row
    ``y_d = ȳ₀ᵀ/δ``, ``x_d = (1/δ, y_d, …, y_d)`` (``y_d`` repeated once per lag).
    """
    presample = np.asarray(presample, dtype=np.float64)
    lags, n = presample.shape
    ybar0 = presample.mean(axis=0)
    ys, xs = [], []
    if prior.soc is not None:
        for j in range(n):
            if prior.means[j] != 0.0:
                yd = np.zeros(n)
                yd[j] = ybar0[j] / prior.soc
                ys.append(yd)
                xs.append(np.concatenate([[0.0], np.tile(yd, lags)]))
    if prior.sur is not None:
        yd = ybar0 / prior.sur
        ys.append(yd)
        xs.append(np.concatenate([[1.0 / prior.sur], np.tile(yd, lags)]))
    k = 1 + n * lags
    return np.reshape(ys, (len(ys), n)), np.reshape(xs, (len(xs), k))


def build_problem(data: np.ndarray, lags: int, prior: Prior) -> Problem:
    """The problem of ``p + T`` rows of data: the first ``p`` rows are the presample."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != prior.n:
        raise ValueError(f"data must be (p + T, {prior.n}), got {data.shape}")
    Y, X = lagged(data, lags)
    presample = data[:lags].copy()
    Yd, Xd = dummy_rows(prior, presample)
    return Problem(
        prior=prior,
        lags=lags,
        presample=presample,
        Y=Y,
        X=X,
        Yd=Yd,
        Xd=Xd,
        omega=minnesota_omega(prior, lags),
        b=minnesota_mean(prior, lags),
    )


# --------------------------------------------------------------------------- block 1: (B, Σ) | h


@dataclass(frozen=True)
class NIW:
    """``Σ ~ IW(scale, df)`` and ``B | Σ ~ MN(mean, precision⁻¹, Σ)``, batched over ``(…)``."""

    mean: np.ndarray  # B̄, (…, k, n)
    precision: np.ndarray  # Ω̄⁻¹ = Ω⁻¹ + X̃ᵀX̃, (…, k, k)
    scale: np.ndarray  # Ψ̄, (…, n, n)
    df: float  # d̄ = d + T + m

    @property
    def omega(self) -> np.ndarray:
        """``Ω̄`` (dense)."""
        return np.linalg.inv(self.precision)


def niw_posterior(problem: Problem, w: np.ndarray) -> NIW:
    """Block 1's posterior parameters with row weights ``w_t`` on the data rows (``e^{−h_t}`` in
    the sweep) and 1 on the dummy rows.

    ``X̃, Ỹ`` are the data rows scaled by ``√w_t`` stacked on the dummy rows, so
    ``X̃ᵀX̃ = Σ_t w_t x_t x_tᵀ + X_dᵀX_d``. ``Ψ̄`` is computed in the equivalent residual form
    ``Ψ + (Ỹ − X̃B̄)ᵀ(Ỹ − X̃B̄) + (B̄ − b)ᵀΩ⁻¹(B̄ − b)`` (equal to ``Ψ + ỸᵀỸ + bᵀΩ⁻¹b −
    B̄ᵀΩ̄⁻¹B̄`` since ``Ω̄⁻¹B̄ = Ω⁻¹b + X̃ᵀỸ``), which avoids the cancellation of the textbook form.
    With ``T = 0`` data rows it is the prior updated by the dummy rows alone (the effective prior).
    """
    X, Y = problem.X, problem.Y
    w = np.asarray(w, dtype=np.float64)
    prec_prior = 1.0 / problem.omega  # Ω⁻¹ (diagonal)
    Xw = X * w[..., :, None]
    P = np.swapaxes(Xw, -1, -2) @ X + problem.Xd.T @ problem.Xd
    P = P + np.diag(prec_prior)
    rhs = np.swapaxes(Xw, -1, -2) @ Y + problem.Xd.T @ problem.Yd + prec_prior[:, None] * problem.b
    Bbar = np.linalg.solve(P, rhs)
    R = Y - X @ Bbar
    Rd = problem.Yd - problem.Xd @ Bbar
    dev = Bbar - problem.b
    S = (
        np.swapaxes(R * w[..., :, None], -1, -2) @ R
        + np.swapaxes(Rd, -1, -2) @ Rd
        + np.swapaxes(dev * prec_prior[:, None], -1, -2) @ dev
    )
    S = S + np.diag(np.asarray(problem.prior.psi, dtype=np.float64))
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    return NIW(mean=Bbar, precision=P, scale=S, df=float(problem.prior.d + problem.T + problem.m))


def draw_niw(
    post: NIW, rng: np.random.Generator, size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One ``(B, Σ)`` per batch element of ``post`` (or ``size`` draws of an unbatched ``post``).

    ``Σ ~ IW(Ψ̄, d̄)`` by the Bartlett decomposition: ``A`` lower triangular with
    ``A_ii² ~ χ²(d̄ − i)`` (``i = 0..n−1``) and ``A_ij ~ N(0, 1)`` below the diagonal, so
    ``AAᵀ ~ W(I, d̄)``; with ``Ψ̄ = CCᵀ`` (Cholesky), ``Σ⁻¹ = C⁻ᵀAAᵀC⁻¹ ~ W(Ψ̄⁻¹, d̄)``, i.e.
    ``Σ = FFᵀ`` with ``F = CA⁻ᵀ``. Then ``B = B̄ + L⁻ᵀZFᵀ`` with ``Ω̄⁻¹ = LLᵀ`` and ``Z`` a
    ``k × n`` standard normal matrix, so ``B | Σ ~ MN(B̄, Ω̄, Σ)``. Random numbers in the order:
    the ``n`` chi-squares, the ``n(n−1)/2`` normals of ``A`` (row-major lower triangle), ``Z``.
    """
    k, n = post.mean.shape[-2:]
    batch = post.mean.shape[:-2] if size is None else (size, *post.mean.shape[:-2])
    A = np.zeros((*batch, n, n))
    for i in range(n):
        A[..., i, i] = np.sqrt(rng.chisquare(post.df - i, size=batch))
    rows, cols = np.tril_indices(n, -1)
    A[..., rows, cols] = rng.standard_normal((*batch, rows.size))
    Z = rng.standard_normal((*batch, k, n))
    C = np.linalg.cholesky(post.scale)
    F = C @ np.swapaxes(np.linalg.inv(A), -1, -2)
    Sigma = F @ np.swapaxes(F, -1, -2)
    Sigma = 0.5 * (Sigma + np.swapaxes(Sigma, -1, -2))
    L = np.linalg.cholesky(post.precision)
    row_factor = np.swapaxes(np.linalg.inv(L), -1, -2)  # L⁻ᵀ: (L⁻ᵀ)(L⁻ᵀ)ᵀ = Ω̄
    B = post.mean + row_factor @ Z @ np.swapaxes(F, -1, -2)
    return B, Sigma


def quad_forms(problem: Problem, B: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """``q_t = u_tᵀΣ⁻¹u_t`` with ``u_t = y_t − Bᵀx_t``, shape ``(K, T)``."""
    U = problem.Y - problem.X @ B
    Linv = np.linalg.inv(np.linalg.cholesky(Sigma))
    V = U @ np.swapaxes(Linv, -1, -2)
    return np.sum(V * V, axis=-1)


def log_q(q: np.ndarray) -> np.ndarray:
    """``log q_t`` (``−∞`` for ``q_t = 0``)."""
    with np.errstate(divide="ignore"):
        return np.log(q)


# --------------------------------------------------------------------------- block 2: h | rest


def ar1_stationary(rho: np.ndarray, sigma_h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Paths of the stationary AR(1) prior from standard normals ``z`` (``(…, K, T)``).

    ``h_0 = σ_h z_0/√(1 − ρ²)`` and ``h_t = ρh_{t−1} + σ_h z_t``. The recursion is evaluated
    exactly by recursive doubling: after the step with shift ``s`` every ``h_t`` holds
    ``Σ_{j<2s} ρ^j e_{t−j}`` (``e_t`` the innovations), so ``⌈log₂ T⌉`` vector steps replace the
    ``T`` scalar ones; only powers ``ρ^{2^i}`` with ``|ρ| < 1`` are formed.
    """
    rho = np.asarray(rho, dtype=np.float64)
    sigma_h = np.asarray(sigma_h, dtype=np.float64)
    h = sigma_h[..., None] * z
    h[..., 0] = h[..., 0] / np.sqrt(1.0 - rho * rho)
    a = rho[..., None]
    shift, T = 1, h.shape[-1]
    while shift < T:
        h[..., shift:] += a * h[..., :-shift]
        a = a * a
        shift *= 2
    return h


def h_loglik(h: np.ndarray, logq: np.ndarray, n: int) -> np.ndarray:
    """``Σ_t [−(n/2)h_t − ½e^{−h_t}q_t]`` along the last axis (guarded, module docstring)."""
    return np.sum(-0.5 * n * h - 0.5 * np.exp(np.minimum(logq - h, EXP_CAP)), axis=-1)


def ess_update(
    h: np.ndarray, nu: np.ndarray, logq: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """One elliptical slice sampling update of every row of ``h`` (``(K, T)``) with the prior
    draws ``nu`` (Murray, Adams & MacKay 2010, Fig. 2; the prior has mean zero).

    ``log y = L(h) − Exp(1)``; ``θ ~ U(0, 2π)`` with the bracket ``[θ − 2π, θ]``; propose
    ``h cos θ + ν sin θ``, accept if its log-likelihood is at least ``log y``, otherwise shrink the
    bracket towards 0 and redraw ``θ`` uniformly in it.

    The linear part of the log-likelihood of a proposal is ``−(n/2)(cos θ Σh + sin θ Σν)``, so
    only the exponential term is summed per proposal; the chains still waiting are compressed
    only when some of them accept.
    """
    K = h.shape[0]
    half_n = 0.5 * n
    sum_h, sum_nu = h.sum(axis=-1), nu.sum(axis=-1)
    logy = h_loglik(h, logq, n) - rng.standard_exponential(K)
    theta = 2.0 * np.pi * rng.random(K)
    lo, hi = theta - 2.0 * np.pi, theta.copy()
    out = h.copy()
    idx = np.arange(K)
    hh, vv, lq, sh, sv, ly = h, nu, logq, sum_h, sum_nu, logy
    while True:
        th = theta[idx]
        c, s = np.cos(th), np.sin(th)
        prop = hh * c[:, None] + vv * s[:, None]
        expo = np.exp(np.minimum(lq - prop, EXP_CAP)).sum(axis=-1)
        ok = -half_n * (c * sh + s * sv) - 0.5 * expo >= ly
        if ok.all():
            out[idx] = prop
            return out
        if ok.any():
            out[idx[ok]] = prop[ok]
            wait = ~ok
            idx, th = idx[wait], th[wait]
            hh, vv, lq, sh, sv, ly = hh[wait], vv[wait], lq[wait], sh[wait], sv[wait], ly[wait]
        below = th < 0.0
        lo[idx[below]] = th[below]
        hi[idx[~below]] = th[~below]
        theta[idx] = lo[idx] + (hi[idx] - lo[idx]) * rng.random(idx.size)


def block2(
    h: np.ndarray,
    logq: np.ndarray,
    rho: np.ndarray,
    sigma_h: np.ndarray,
    n: int,
    rng: np.random.Generator,
    updates: int = ESS_UPDATES,
) -> np.ndarray:
    """``updates`` elliptical slice updates of the ``h`` paths (``(K, T)``) under the stationary
    AR(1) prior at the current ``ρ, σ_h`` (all prior draws ``ν`` first, then the updates)."""
    K, T = h.shape
    nus = ar1_stationary(rho, sigma_h, rng.standard_normal((updates, K, T)))
    for u in range(updates):
        h = ess_update(h, nus[u], logq, n, rng)
    return h


# --------------------------------------------------------------------------- block 3: ρ, σ_h


def slice_sample(
    x0: np.ndarray,
    logf: Callable[[np.ndarray, np.ndarray], np.ndarray],
    width: float,
    max_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One univariate slice sampling update of every element of ``x0`` (``(K,)``; Neal 2003,
    Fig. 3 stepping out with at most ``m = max_steps`` steps, Fig. 5 shrinkage).

    ``logf(x, idx)`` is the log density of the chains ``idx`` at ``x`` (``−∞`` outside the
    support). ``y = f(x₀) − Exp(1)``; ``L = x₀ − wU``, ``R = L + w``; ``J = ⌊mV⌋`` steps at most to
    the left and ``m − 1 − J`` to the right while the end is inside the slice; then shrinkage:
    ``x₁ ~ U(L, R)``, accepted if ``f(x₁) ≥ y``, otherwise the end on ``x₁``'s side of ``x₀``
    moves to ``x₁``.
    """
    K = x0.shape[0]
    every = np.arange(K)
    y = logf(x0, every) - rng.standard_exponential(K)
    left = x0 - width * rng.random(K)
    right = left + width
    j_left = np.floor(max_steps * rng.random(K)).astype(np.int64)
    j_right = (max_steps - 1) - j_left
    for end, steps, sign in ((left, j_left, -1.0), (right, j_right, 1.0)):
        idx = every[steps > 0]
        while idx.size:
            idx = idx[logf(end[idx], idx) > y[idx]]
            end[idx] += sign * width
            steps[idx] -= 1
            idx = idx[steps[idx] > 0]
    out = x0.copy()
    idx = every
    while True:
        x1 = left[idx] + (right[idx] - left[idx]) * rng.random(idx.size)
        ok = logf(x1, idx) >= y[idx]
        if ok.all():
            out[idx] = x1
            return out
        if ok.any():
            out[idx[ok]] = x1[ok]
            idx, x1 = idx[~ok], x1[~ok]
        below = x1 < x0[idx]
        left[idx[below]] = x1[below]
        right[idx[~below]] = x1[~below]


def ar1_sum_of_squares(h: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``S = (1 − ρ²)h_0² + Σ_{t≥1}(h_t − ρh_{t−1})²`` per chain."""
    rho = np.asarray(rho, dtype=np.float64)
    resid = h[..., 1:] - rho[..., None] * h[..., :-1]
    return (1.0 - rho * rho) * h[..., 0] ** 2 + np.sum(resid * resid, axis=-1)


def log_target_rho(rho: np.ndarray, h: np.ndarray, sigma_h: np.ndarray, prior: Prior) -> np.ndarray:
    """``log π(ρ) + ½ log(1 − ρ²) − (1 − ρ²)h_0²/(2σ_h²) − Σ_{t≥1}(h_t − ρh_{t−1})²/(2σ_h²)`` on
    ``(−1, 1)``, ``−∞`` outside; ``π`` the ``N(0.9, 0.2²)`` density (its truncation is the
    support)."""
    inside = np.abs(rho) < 1.0
    r = np.where(inside, rho, 0.0)
    val = (
        -0.5 * ((r - prior.rho_mean) / prior.rho_sd) ** 2
        + 0.5 * np.log(1.0 - r * r)
        - ar1_sum_of_squares(h, r) / (2.0 * sigma_h**2)
    )
    return np.where(inside, val, -np.inf)


def draw_sigma2(
    h: np.ndarray, rho: np.ndarray, prior: Prior, rng: np.random.Generator
) -> np.ndarray:
    """The centred step ``σ_h² | h, ρ ~ IG(c₀ + T/2, C₀ + S/2)`` (a scale over a unit gamma)."""
    T = h.shape[-1]
    scale = prior.sigma2_scale + 0.5 * ar1_sum_of_squares(h, rho)
    return scale / rng.gamma(prior.sigma2_shape + 0.5 * T, size=scale.shape)


def log_target_s(
    s: np.ndarray, x: np.ndarray, logq: np.ndarray, n: int, prior: Prior
) -> np.ndarray:
    """The non-centred target of ``s = log σ_h`` given ``x = h/σ_h``: ``−2c₀s − C₀e^{−2s} +
    Σ_t[−(n/2)e^{s}x_t − ½e^{−e^{s}x_t}q_t]`` (the ``IG(c₀, C₀)`` prior of ``σ_h²`` in ``s``
    with its Jacobian, times the likelihood at ``h = e^{s}x``; guarded)."""
    c0, C0 = prior.sigma2_shape, prior.sigma2_scale
    scale = np.exp(np.minimum(s, EXP_CAP))
    h = scale[..., None] * x
    return -2.0 * c0 * s - C0 * np.exp(np.minimum(-2.0 * s, EXP_CAP)) + h_loglik(h, logq, n)


def update_rho(
    h: np.ndarray, rho: np.ndarray, sigma_h: np.ndarray, prior: Prior, rng: np.random.Generator
) -> np.ndarray:
    """Block 3, first move: the slice sampler of ``ρ | h, σ_h`` (width 0.1, at most 50 steps)."""
    return slice_sample(
        rho,
        lambda r, i: log_target_rho(r, h[i], sigma_h[i], prior),
        RHO_WIDTH,
        RHO_MAX_STEPS,
        rng,
    )


def update_log_sigma_noncentred(
    x: np.ndarray,
    log_sigma: np.ndarray,
    logq: np.ndarray,
    n: int,
    prior: Prior,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 3, last move: the slice sampler of ``s = log σ_h | x, q, ρ`` (width 0.5, at most 50
    steps); ``ρ`` enters only through the prior of ``x``, which does not involve ``σ_h``."""
    return slice_sample(
        log_sigma,
        lambda s, i: log_target_s(s, x[i], logq[i], n, prior),
        S_WIDTH,
        S_MAX_STEPS,
        rng,
    )


def block3(
    h: np.ndarray,
    logq: np.ndarray,
    rho: np.ndarray,
    sigma_h: np.ndarray,
    n: int,
    prior: Prior,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``ρ`` by slice sampling, the centred ``σ_h²`` draw, then the non-centred ``σ_h`` move
    with ``x = h/σ_h`` held fixed; returns the new ``(h, ρ, σ_h)``."""
    rho = update_rho(h, rho, sigma_h, prior, rng)
    sigma_h = np.sqrt(draw_sigma2(h, rho, prior, rng))
    x = h / sigma_h[:, None]
    log_sigma = update_log_sigma_noncentred(x, np.log(sigma_h), logq, n, prior, rng)
    sigma_h = np.exp(log_sigma)
    return sigma_h[:, None] * x, rho, sigma_h


# --------------------------------------------------------------------------- the sweep


@dataclass
class State:
    """One state per chain: ``B (K, k, n)``, ``Sigma (K, n, n)``, ``h (K, T)``, ``rho (K,)``,
    ``sigma_h (K,)``."""

    B: np.ndarray
    Sigma: np.ndarray
    h: np.ndarray
    rho: np.ndarray
    sigma_h: np.ndarray


def sweep(state: State, problem: Problem, rng: np.random.Generator) -> State:
    """One Gibbs sweep: block 1, block 2, block 3 (§9.8 Sampler)."""
    n, prior = problem.n, problem.prior
    w = np.exp(np.minimum(-state.h, EXP_CAP))
    B, Sigma = draw_niw(niw_posterior(problem, w), rng)
    logq = log_q(quad_forms(problem, B, Sigma))
    h = block2(state.h, logq, state.rho, state.sigma_h, n, rng)
    h, rho, sigma_h = block3(h, logq, state.rho, state.sigma_h, n, prior, rng)
    return State(B=B, Sigma=Sigma, h=h, rho=rho, sigma_h=sigma_h)


# --------------------------------------------------------------------------- prior and data draws


def draw_rho_prior(prior: Prior, rng: np.random.Generator, size: int) -> np.ndarray:
    """``ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)`` by rejection."""
    out = np.empty(size)
    todo = np.arange(size)
    while todo.size:
        r = prior.rho_mean + prior.rho_sd * rng.standard_normal(todo.size)
        ok = np.abs(r) < 1.0
        out[todo[ok]] = r[ok]
        todo = todo[~ok]
    return out


def draw_sv_prior(
    prior: Prior, T: int, rng: np.random.Generator, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, ρ, σ_h)`` from the prior: ``ρ``, then ``σ_h² ~ IG(c₀, C₀)``, then the stationary
    AR(1) path of length ``T``."""
    rho = draw_rho_prior(prior, rng, size)
    sigma_h = np.sqrt(prior.sigma2_scale / rng.gamma(prior.sigma2_shape, size=size))
    h = ar1_stationary(rho, sigma_h, rng.standard_normal((size, T)))
    return h, rho, sigma_h


def draw_prior(problem: Problem, rng: np.random.Generator, size: int) -> State:
    """A joint draw from the prior: ``(B, Σ)`` from the NIW prior updated by the dummy rows (the
    effective prior of a dummy-observation prior), then ``(h, ρ, σ_h)``."""
    empty = problem.with_data(np.zeros((0, problem.n)), np.zeros((0, problem.k)))
    B, Sigma = draw_niw(niw_posterior(empty, np.zeros(0)), rng, size=size)
    h, rho, sigma_h = draw_sv_prior(problem.prior, problem.T, rng, size)
    return State(B=B, Sigma=Sigma, h=h, rho=rho, sigma_h=sigma_h)


def simulate_data(
    presample: np.ndarray, state: State, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``(Y, X)`` of ``T`` rows per chain from the model given ``state`` (``T = h.shape[1]``):
    ``y_t = Bᵀx_t + e^{h_t/2}Lε_t`` with ``Σ = LLᵀ``, recursively from the fixed presample."""
    K, T = state.h.shape
    lags, n = presample.shape
    chol = np.linalg.cholesky(state.Sigma)
    eps = rng.standard_normal((K, T, n))
    hist = np.broadcast_to(presample[::-1], (K, lags, n)).copy()  # y_{t−1}, …, y_{t−p}
    Y = np.empty((K, T, n))
    X = np.empty((K, T, 1 + n * lags))
    for t in range(T):
        X[:, t, 0] = 1.0
        X[:, t, 1:] = hist.reshape(K, lags * n)
        shock = np.einsum("kij,kj->ki", chol, eps[:, t]) * np.exp(0.5 * state.h[:, t])[:, None]
        Y[:, t] = np.einsum("ki,kin->kn", X[:, t], state.B) + shock
        hist = np.concatenate([Y[:, t][:, None], hist[:, :-1]], axis=1)
    return Y, X


def initial_state(problem: Problem, rng: np.random.Generator, size: int = 1) -> State:
    """A start for ``size`` chains: ``(h, ρ, σ_h)`` from the prior and ``(B, Σ)`` from block 1
    given that ``h`` (the first sweep's block 1 redraws them)."""
    h, rho, sigma_h = draw_sv_prior(problem.prior, problem.T, rng, size)
    B, Sigma = draw_niw(niw_posterior(problem, np.exp(np.minimum(-h, EXP_CAP))), rng)
    return State(B=B, Sigma=Sigma, h=h, rho=rho, sigma_h=sigma_h)


# --------------------------------------------------------------------------- runs and forecasts


def compared_dates(T: int) -> np.ndarray:
    """The compared ``h_t``: ``t ∈ {0, ⌊T/4⌋, ⌊T/2⌋, T − 1}`` (§9.8 Comparison)."""
    return np.array([0, T // 4, T // 2, T - 1])


def run(
    problem: Problem,
    burn: int,
    keep: int,
    seed: int,
    progress: Callable[[int, float], None] | None = None,
) -> dict[str, np.ndarray | float]:
    """One run (a single chain, ``K = 1``) of ``burn`` discarded and ``keep`` kept sweeps from
    ``initial_state`` with ``numpy.random.default_rng(seed)``.

    Returns every kept draw of the compared quantities (``B``, ``Sigma``, ``h`` at
    :func:`compared_dates`, ``rho``, ``sigma_h``), the last data row's ``h_{T−1}`` (the
    predictive's start), the posterior mean of the whole ``h`` path and the wall time.
    """
    rng = np.random.default_rng(seed)
    t_sel = compared_dates(problem.T)
    k, n, T = problem.k, problem.n, problem.T
    out = {
        "B": np.empty((keep, k, n)),
        "Sigma": np.empty((keep, n, n)),
        "h": np.empty((keep, t_sel.size)),
        "h_last": np.empty(keep),
        "rho": np.empty(keep),
        "sigma_h": np.empty(keep),
    }
    h_sum = np.zeros(T)
    start = time.perf_counter()
    state = initial_state(problem, rng)
    for i in range(burn + keep):
        state = sweep(state, problem, rng)
        j = i - burn
        if j >= 0:
            out["B"][j] = state.B[0]
            out["Sigma"][j] = state.Sigma[0]
            out["h"][j] = state.h[0, t_sel]
            out["h_last"][j] = state.h[0, T - 1]
            out["rho"][j] = state.rho[0]
            out["sigma_h"][j] = state.sigma_h[0]
            h_sum += state.h[0]
        if progress is not None and (i + 1) % 10_000 == 0:
            progress(i + 1, time.perf_counter() - start)
    out["h_path_mean"] = h_sum / keep
    out["wall_time"] = time.perf_counter() - start
    return out


def predictive(
    problem: Problem,
    B: np.ndarray,
    Sigma: np.ndarray,
    h_last: np.ndarray,
    rho: np.ndarray,
    sigma_h: np.ndarray,
    rng: np.random.Generator,
    horizon: int = 4,
) -> np.ndarray:
    """One predictive path per draw (``S`` draws), shape ``(S, horizon, n)``.

    ``h_{T+j} = ρh_{T+j−1} + σ_h η_j`` from ``h_{T−1}``; ``y_{T+j} = Bᵀx_{T+j} + e^{h_{T+j}/2}Lε_j``
    with ``Σ = LLᵀ``; the first regressors come from the last ``p`` data rows. The random numbers
    are one ``(S, horizon, 1 + n)`` standard normal array: ``η_j`` in column 0, ``ε_j`` after it.
    """
    S = B.shape[0]
    n, lags = problem.n, problem.lags
    z = rng.standard_normal((S, horizon, 1 + n))
    chol = np.linalg.cholesky(Sigma)
    data_tail = problem.Y[-lags:] if problem.T >= lags else None
    if data_tail is None:
        raise ValueError("the predictive needs at least p data rows")
    hist = np.broadcast_to(data_tail[::-1], (S, lags, n)).copy()  # y_{T−1}, …, y_{T−p}
    h = np.asarray(h_last, dtype=np.float64).copy()
    out = np.empty((S, horizon, n))
    for j in range(horizon):
        h = rho * h + sigma_h * z[:, j, 0]
        x = np.concatenate([np.ones((S, 1)), hist.reshape(S, lags * n)], axis=1)
        shock = np.einsum("sij,sj->si", chol, z[:, j, 1:]) * np.exp(0.5 * h)[:, None]
        out[:, j] = np.einsum("si,sin->sn", x, B) + shock
        hist = np.concatenate([out[:, j][:, None], hist[:, :-1]], axis=1)
    return out
