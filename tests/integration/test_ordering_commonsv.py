"""Ordering invariance of the order-invariant ``CommonSV`` composition within MCSE (docs/phase3/
DESIGN.md §0.2 row 27, §9.11; N08 §B5 item 2; E140, E159; WP35).

``Minnesota(form="conjugate") × CommonSV()`` is structural Mode II at ``cross = contemp = 1``:
the NIW prior with one common scalar volatility, ``u_t ~ N(0, e^{h_t}Σ)`` (P3 §5.3), exactly
order-invariant, which ``monitor.is_order_invariant`` relies on (E99). So the posterior of the
user ordering and of its reverse agree after un-permutation: the reduced-form coefficients ``B``,
the base covariance ``Σ = A⁻¹ΛA⁻ᵀ`` (lower triangle), ``h_T``, ``ρ``, ``σ_h`` and the forecast
paths at horizons 1–4. The comparison is the Phase 2 parity rule (P3 Conventions): every compared
quantile (5%, 50%, 95% of every quantity) passes when ``|z| ≤ max(4, z_{1−0.01/(2m)})``, ``z`` the
difference over the combined MCSE, in the CDF form of A8 (``tests/analytic/
test_common_sv_twins.py``): ``F̂_reverse(x̂_q) − F̂_user(x̂_q)`` over the indicator MCSEs
(``diagnostics.mcse_mean_many``), ``x̂_q`` the user ordering's sample quantile.

Data: a VAR(2) of three variables of unequal scales (×1, ×3, ×0.5, so a label mix-up shows) with
one common log volatility (``ρ = 0.9``, ``σ_h² = 0.05``), ``T = 200``; both fits at the
composition's default budget (D-51: 4 × 10,000 kept draws after 1,000), different seeds.
Weekly (``slow``, ``weekly``).
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import rai_bvar as rb
from rai_bvar import diagnostics
from rai_bvar.data.design import regressor_names
from rai_bvar.structural.ordering import permute_model

Q = np.array([0.05, 0.5, 0.95])
LAGS, H = 2, 4


def common_sv_data(T: int = 200, seed: int = 27) -> rb.VARData:
    """``y_t = c + A₁y_{t−1} + A₂y_{t−2} + D C e^{h_t/2} z_t``, ``h_t = 0.9h_{t−1} + √0.05 η_t``
    (one common ``h``), ``D = diag(1, 3, 0.5)`` (module docstring)."""
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 3.0, 0.5])
    A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    A2 = np.array([[0.1, 0.0, 0.0], [0.05, 0.1, 0.0], [0.0, 0.0, 0.1]])
    A1, A2 = ((s[:, None] * A) / s[None, :] for A in (A1, A2))
    C = np.linalg.cholesky(np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.25], [0.2, 0.25, 1.0]]))
    burn = 100
    y = np.zeros((T + burn, 3))
    h = 0.0
    for t in range(2, T + burn):
        h = 0.9 * h + math.sqrt(0.05) * rng.standard_normal()
        u = s * (C @ rng.standard_normal(3)) * math.exp(0.5 * h)
        y[t] = 0.1 * s + A1 @ y[t - 1] + A2 @ y[t - 2] + u
    frame = pd.DataFrame(
        y[burn:], index=pd.period_range("1970Q1", periods=T, freq="Q"), columns=["x", "y", "z"]
    )
    return rb.VARData.from_frame(frame, tcodes=1)


def base_sigma(a: np.ndarray, log_lam: np.ndarray) -> np.ndarray:
    """``Σ = A⁻¹ diag(λ) A⁻ᵀ`` for ``a`` (…, n, n−1) and ``log λ`` (…, n)."""
    n = a.shape[-2]
    A = np.broadcast_to(np.eye(n), (*a.shape[:-2], n, n)).copy()
    rows, cols = np.tril_indices(n, -1)
    A[..., rows, cols] = a[..., rows, cols]
    Ainv = np.linalg.inv(A)
    return Ainv @ (np.exp(log_lam)[..., :, None] * np.swapaxes(Ainv, -1, -2))


def parity_z(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """``z`` of the Phase 2 parity rule in its CDF form for every quantity (last axis) and every
    quantile of :data:`Q`; ``x`` are ``(chains, draws, m)`` samples, ``x_b`` the reference side."""
    z = []
    for q in Q:
        point = np.quantile(x_b.reshape(-1, x_b.shape[-1]), q, axis=0)
        ind_a = (x_a <= point).astype(np.float64)
        ind_b = (x_b <= point).astype(np.float64)
        var = diagnostics.mcse_mean_many(ind_a) ** 2 + diagnostics.mcse_mean_many(ind_b) ** 2
        z.append((ind_a.mean(axis=(0, 1)) - ind_b.mean(axis=(0, 1))) / np.sqrt(var))
    return np.concatenate(z)


def summaries(fit, order: tuple[str, ...], variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """``(chains, draws, m)`` draws of the compared quantities in the user ordering's labels."""
    B = fit.reduced_form_draws()
    regs = regressor_names(variables, LAGS, "c")
    B = B.sel(variable=variables, regressor=regs).values  # (C, D, k, n)
    C, D = B.shape[:2]
    p = fit.param_draws
    S = base_sigma(np.asarray(p.struct.a), np.asarray(p.vol.core.log_lam))  # fit's ordering
    idx = [order.index(v) for v in variables]
    S = S[..., idx, :][..., :, idx]
    rows, cols = np.tril_indices(len(variables))
    post = fit.posterior["posterior"]
    scal = np.stack([post[k].values for k in ("common_h_T", "rho", "sigma")], axis=-1)
    fc = fit.forecast(H, draws=C * D, seed=5)
    paths = fc.paths.sel(variable=variables).values  # (C, D', H, n)
    assert paths.shape[:2] == (C, D)
    x = np.concatenate([B.reshape(C, D, -1), S[..., rows, cols].reshape(C, D, -1), scal,
                        paths.reshape(C, D, -1)], axis=-1)  # fmt: skip
    labels = ([f"B[{r},{v}]" for r in regs for v in variables]
              + [f"Sigma[{variables[i]},{variables[j]}]" for i, j in zip(rows, cols, strict=True)]
              + ["h_T", "rho", "sigma_h"]
              + [f"y[h={h + 1},{v}]" for h in range(H) for v in variables])  # fmt: skip
    return x, labels


@pytest.mark.slow
@pytest.mark.weekly
def test_conjugate_common_sv_is_order_invariant_within_mcse(record_property):
    """The user ordering and its reverse agree under the Phase 2 parity rule (module
    docstring)."""
    data = common_sv_data()
    variables = list(data.variables)
    model = rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="conjugate", hierarchy="fixed")),
        vol=rb.CommonSV(), lags=LAGS,
    )  # fmt: skip
    assert model.mcmc.chains == 4 and model.mcmc.draws == 10_000
    draws = {}
    for order, seed in ((tuple(variables), 1), (tuple(reversed(variables)), 2)):
        m = permute_model(model, order, variables=variables)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = m.fit(data.select(order), seed=seed)
        assert fit.static.mean.const_path == "shared_factor"
        draws[order], labels = summaries(fit, order, variables)
    user, reverse = draws[tuple(variables)], draws[tuple(reversed(variables))]
    z = parity_z(reverse, user)
    crit = max(4.0, float(stats.norm.isf(0.01 / (2.0 * z.size))))
    names = [f"{lab}@q{q:g}" for q in Q for lab in labels]
    j = int(np.argmax(np.abs(z)))
    out = {"m": int(z.size), "max_abs_z": float(np.abs(z[j])), "argmax": names[j],
           "critical": crit, "passed": bool(np.all(np.abs(z) <= crit))}  # fmt: skip
    record_property("p3_record", json.dumps({"test": "ordering-invariance row 27", **out}))
    assert out["passed"], (out, sorted(zip(np.abs(z), names, strict=True))[-5:])
