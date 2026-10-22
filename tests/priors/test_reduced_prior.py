"""Reduced-form coefficient prior for CCCM (DESIGN §6.2; N04 §3.1–3.2; D-11) and the explicit
parity hook ``explicit_reduced`` (DESIGN §9.9)."""

from __future__ import annotations

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.priors.reduced import explicit_reduced, reduced_arrays
from rai_bvar.priors.scales import WindowScales, window_scales
from rai_bvar.priors.structural import contemp_arrays


def _scales(n: int) -> WindowScales:
    psi = np.array([0.5, 2.0, 0.8, 3.0][:n])
    return WindowScales(psi=psi, own_means=np.array([1.0, 0.0, 1.0, 0.0][:n]), freq="M")


@pytest.mark.parametrize("n_det", [0, 1, 2])
def test_reduced_variances_follow_the_design_table(n_det):
    """``V_j``: own lag ``κ²/l^α``; lag ``l`` of ``m ≠ j``: ``κ²κ_c²ψ_j/(l^αψ_m)``; deterministic
    ``κ_d ψ_j`` (the NIW marginal, D-11). Means: own first lag ``m_j``."""
    n, lags = 4, 3
    lam, alpha, cross, kd = 0.15, 1.0, 0.4, 3e6
    scales = _scales(n)
    psi = scales.psi
    prior = rb.Minnesota(form="reduced", cross=cross, var_const=kd, lambda_=0.9, alpha=3.0)
    coef, shared = reduced_arrays(
        scales, prior, {"lambda": lam, "alpha": alpha}, n=n, lags=lags, n_det=n_det
    )
    k = n_det + n * lags
    assert coef.mean0.shape == coef.prec0.shape == coef.group.shape == (n, k)
    np.testing.assert_array_equal(shared.Mc, np.zeros((k, n - 1)))  # no D3 centring
    var = 1.0 / coef.prec0
    for j in range(n):
        for r in range(n_det):
            assert var[j, r] == pytest.approx(kd * psi[j], rel=1e-14)
            assert coef.group[j, r] == 0
        for lag in range(1, lags + 1):
            for m in range(n):
                r = n_det + (lag - 1) * n + m
                if m == j:
                    expected = lam**2 / lag**alpha
                else:
                    expected = lam**2 * cross**2 * psi[j] / (lag**alpha * psi[m])
                assert var[j, r] == pytest.approx(expected, rel=1e-14)
                assert coef.group[j, r] == (1 if m == j else 2)
                mean = scales.own_means[j] if (m == j and lag == 1) else 0.0
                assert coef.mean0[j, r] == mean


def test_reduced_prior_is_the_niw_marginal_at_cross_one(data4):
    """``Var(B_rj) = E[Σ_jj] Ω_r = ψ_j Ω_r`` under the Phase 1 NIW prior (``d = n + 2``)."""
    lags = 2
    problem = ConjugateProblem(data4, rb.Minnesota(), lags, "c", None)
    hyper = {"lambda": 0.3, "alpha": 1.7}
    h = problem.complete(hyper)
    prior = rb.Minnesota(form="reduced")
    scales = window_scales(data4, prior, lags, "c")
    coef, _ = reduced_arrays(scales, prior, hyper, n=4, lags=lags, n_det=1)
    np.testing.assert_allclose(
        1 / coef.prec0, scales.psi[:, None] * problem.omega(h)[None, :], rtol=1e-14
    )
    np.testing.assert_array_equal(coef.mean0, problem.b.T)


def test_explicit_reduced_reproduces_the_built_arrays():
    n, lags, n_det = 3, 2, 1
    scales = _scales(n)
    prior = rb.Minnesota(form="reduced", contemp=3.0)
    coef, _ = reduced_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det)
    con = contemp_arrays(scales, prior, n, scaling="none")
    coef2, con2 = explicit_reduced(
        coef.mean0, 1 / coef.prec0, con.mean0, np.where(con.cmask > 0, 1 / con.prec0, np.nan),
        n_det=n_det,
    )  # fmt: skip
    np.testing.assert_allclose(coef2.prec0, coef.prec0, rtol=1e-15)
    np.testing.assert_array_equal(coef2.mean0, coef.mean0)
    np.testing.assert_array_equal(coef2.group, coef.group)
    np.testing.assert_allclose(con2.prec0, con.prec0, rtol=1e-15)
    np.testing.assert_array_equal(con2.mean0, con.mean0)
    np.testing.assert_array_equal(con2.cmask, con.cmask)


def test_explicit_reduced_parity_parameter_set():
    """DESIGN §9.9: ``π_j ~ N(m_j, diag v_π)`` and ``a_ij ~ N(0, v_a)`` from scalars."""
    n, lags = 3, 2
    mean0 = np.zeros((n, 1 + n * lags))
    mean0[np.arange(n), 1 + np.arange(n)] = 0.9
    coef, con = explicit_reduced(mean0, 10.0, 0.0, 4.0)
    np.testing.assert_array_equal(coef.prec0, 0.1)
    np.testing.assert_array_equal(coef.mean0, mean0)
    np.testing.assert_array_equal(coef.group[:, 0], 0)
    np.testing.assert_array_equal(np.diag(coef.group[:, 1 : 1 + n]), 1)
    np.testing.assert_array_equal(con.cmask, [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(con.prec0, [[1, 1], [0.25, 1], [0.25, 0.25]])
    np.testing.assert_array_equal(con.mean0, 0.0)
    # non-zero contemporaneous means on active slots only; inactive input is ignored
    _, con = explicit_reduced(mean0, 10.0, np.full((n, n - 1), 0.5), 4.0)
    np.testing.assert_array_equal(con.mean0, [[0, 0], [0.5, 0], [0.5, 0.5]])
    # one equation: no contemporaneous slots
    coef1, con1 = explicit_reduced(np.zeros((1, 3)), 1.0, 0.0, 1.0, n_det=1)
    assert con1.prec0.shape == (1, 0) and coef1.group.tolist() == [[0, 1, 1]]


def test_explicit_reduced_validation():
    mean0 = np.zeros((2, 5))
    with pytest.raises(ValueError, match="n_det"):
        explicit_reduced(mean0, 1.0, 0.0, 1.0, n_det=2)  # 5 − 2 is not a multiple of n = 2
    with pytest.raises(ValueError, match="var0"):
        explicit_reduced(mean0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="broadcast"):
        explicit_reduced(mean0, np.ones(3), 0.0, 1.0)
    with pytest.raises(ValueError, match="active"):
        explicit_reduced(mean0, 1.0, 0.0, np.array([[1.0], [-1.0]]))
    with pytest.raises(ValueError, match="active"):
        explicit_reduced(mean0, 1.0, np.array([[0.0], [np.nan]]), 1.0)
    with pytest.raises(ValueError, match="finite"):
        explicit_reduced(np.full((2, 5), np.nan), 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"\(n, k\)"):
        explicit_reduced(np.zeros(5), 1.0, 0.0, 1.0)
    # inactive slots may hold anything (they are overwritten)
    _, con = explicit_reduced(mean0, 1.0, np.array([[np.nan], [0.0]]), np.array([[0.0], [2.0]]))
    np.testing.assert_array_equal(con.prec0, [[1.0], [0.5]])
