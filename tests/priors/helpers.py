"""Shared data and independent prior samplers for the host prior-builder tests (WP6).

``make_data(n, T, freq)`` builds raw levels whose transformed series have distinct scales
(variable ``j`` is about ``10^{j/2}`` times larger than variable 0), so a ψ or label mix-up shows
up as an order-of-magnitude error (N08 §B5 item 6). Transformation codes cycle through
``(1, 5, 2, 4)``: two persistent codes (own-lag prior mean 1) and two differenced ones (0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rai_bvar.data import VARData

TCODE_CYCLE = (1, 5, 2, 4)


def make_frame(
    n: int, T: int, freq: str = "Q", seed: int = 0, start: str = "1980-01"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.period_range(pd.Period(start, freq=freq), periods=T, freq=freq)
    cols = {}
    for j in range(n):
        tcode = TCODE_CYCLE[j % len(TCODE_CYCLE)]
        scale = 10.0 ** (j / 2.0)
        shocks = rng.standard_normal(T)
        if tcode == 1:  # stationary AR(1) level
            x = np.zeros(T)
            for t in range(1, T):
                x[t] = 0.8 * x[t - 1] + shocks[t]
            cols[f"v{j}"] = scale * x
        elif tcode == 5:  # Δ log: positive level with growth noise
            cols[f"v{j}"] = 100.0 * np.exp(np.cumsum(0.005 + 0.01 * scale * shocks / 10.0))
        elif tcode == 2:  # first difference of a random walk with drift
            cols[f"v{j}"] = scale * np.cumsum(0.1 + shocks)
        else:  # tcode 4: log level, persistent
            x = np.zeros(T)
            for t in range(1, T):
                x[t] = 0.9 * x[t - 1] + 0.05 * shocks[t]
            cols[f"v{j}"] = np.exp(2.0 + x)
    return pd.DataFrame(cols, index=index)


def make_data(n: int, T: int = 120, freq: str = "Q", seed: int = 0) -> VARData:
    frame = make_frame(n, T, freq, seed)
    tcodes = {c: TCODE_CYCLE[j % len(TCODE_CYCLE)] for j, c in enumerate(frame.columns)}
    return VARData.from_frame(frame, tcodes)


# --------------------------------------------------------------------------- independent samplers
# Written from the prior statements of N05 §4.1–4.4 (explicit inverses are fine in tests); they
# read the arrays under test but share no code with the builders.


def ig_draws(
    rng: np.random.Generator, shape: np.ndarray, scale: np.ndarray, size: int
) -> np.ndarray:
    """``λ ~ IG(shape, scale)`` (density ∝ λ^{−shape−1} e^{−scale/λ}), shape ``(size, n)``."""
    return scale / rng.gamma(shape, 1.0, size=(size, len(shape)))


def sample_structural(
    rng: np.random.Generator,
    coef,
    contemp,
    shared,
    lam: np.ndarray,
    *,
    scale_coef: bool,
    scale_contemp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws ``(B, Σ)`` of the D3 structural prior: ``a_i ~ N(m_a, s_i/P_a)``,
    ``δ_i ~ N(m_δ, s_i/P_δ)`` with ``s_i = λ_i`` for a λ-scaled block (else 1), ``γ_i = δ_i +
    M_c a_i``, ``B' = A⁻¹Γ``, ``Σ = A⁻¹ΛA⁻¹'``. ``lam`` (size, n) are the λ draws."""
    size, n = lam.shape
    k = coef.mean0.shape[1]
    s_a = lam if scale_contemp else np.ones_like(lam)
    s_d = lam if scale_coef else np.ones_like(lam)
    a = contemp.mean0 + np.sqrt(s_a[:, :, None] / contemp.prec0) * rng.standard_normal(
        (size, n, n - 1)
    )
    a = a * contemp.cmask
    delta = coef.mean0 + np.sqrt(s_d[:, :, None] / coef.prec0) * rng.standard_normal((size, n, k))
    gamma = delta + np.einsum("rj,sij->sir", shared.Mc, a)
    A = np.broadcast_to(np.eye(n), (size, n, n)).copy()
    rows, cols = np.tril_indices(n, -1)
    A[:, rows, cols] = a[:, rows, cols]
    A_inv = np.linalg.inv(A)
    B = np.swapaxes(A_inv @ gamma, 1, 2)  # (size, k, n)
    Sigma = A_inv @ (lam[:, :, None] * np.swapaxes(A_inv, 1, 2))
    return B, Sigma


def sample_niw(
    rng: np.random.Generator, M: np.ndarray, omega: np.ndarray, psi: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 1 prior: ``Σ ~ IW(diag ψ, n + 2)`` (SciPy) and ``vec B | Σ ~ N(vec M, Σ ⊗ diag Ω)``."""
    from scipy import stats

    n = psi.size
    Sigma = stats.invwishart(df=n + 2, scale=np.diag(psi)).rvs(size=size, random_state=rng)
    Sigma = Sigma.reshape(size, n, n)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((size, M.shape[0], n))
    B = M + np.sqrt(omega)[None, :, None] * (Z @ np.swapaxes(L, 1, 2))
    return B, Sigma
