"""Parameter draws and impulse responses of an MCMC fit (DESIGN §13 WP17; §7.4; N03 §5.4; N05 §8).

``draw_parameters`` returns ``B = (A⁻¹Γ)′`` and ``chol = A⁻¹ diag(exp(½ log λ_date))`` (checked
against ``fit.reduced_form_draws`` and NumPy); dates other than the last in-sample period, and
the effective variance under a transitory scale, come from the stored paths and raise without
them; dates outside the sample raise. ``irf`` through ``recursive_irf`` equals the direct
``Φ_h A⁻¹ Λ^{½}`` (companion-matrix powers, 1e-12); unit shocks give the date-free ``Φ_h A⁻¹``.
Under ``t_scale="unit"`` the persistent variance is ``e^h ν/(ν−2)`` (N03 §5.4).
"""

from __future__ import annotations

import copy
import dataclasses

import mcmc_fits as mf
import numpy as np
import pandas as pd
import pytest

from rai_bvar import SV, Constant
from rai_bvar.engines.mcmc import forecast as F
from rai_bvar.engines.mcmc.state import OutlierTDraw
from rai_bvar.settings import FitPlan

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


@pytest.fixture(scope="module")
def fit(sim_data):
    return mf.shared_fit(sim_data)


def _A(a: np.ndarray) -> np.ndarray:
    n = a.shape[-2]
    A = np.broadcast_to(np.eye(n), (*a.shape[:-2], n, n)).copy()
    r, c = np.tril_indices(n, -1)
    A[..., r, c] = a[..., r, c]
    return A


def _B(fit, c, p):
    a = np.asarray(fit.param_draws.struct.a)[c, p]
    gamma = np.asarray(fit.param_draws.mean.gamma)[c, p]
    return np.swapaxes(np.linalg.solve(_A(a), gamma), -1, -2)


def _direct_irf(B: np.ndarray, A: np.ndarray, lam: np.ndarray, lags: int, n_det: int, H: int):
    """``Θ_h = J F^h J′ A⁻¹ Λ^{½}`` with the companion matrix ``F`` of ``B`` (written out)."""
    n = A.shape[0]
    F_ = np.zeros((n * lags, n * lags))
    for lag in range(lags):
        F_[:n, lag * n : (lag + 1) * n] = B[n_det + lag * n : n_det + (lag + 1) * n].T
    F_[n:, :-n] = np.eye(n * (lags - 1))
    impact = np.linalg.solve(A, np.diag(np.sqrt(lam)))
    out, Fh = [], np.eye(n * lags)
    for _ in range(H + 1):
        out.append(Fh[:n, :n] @ impact)
        Fh = F_ @ Fh
    return np.stack(out)


def test_draw_parameters_at_the_last_period(fit):
    """``B`` equals ``reduced_form_draws`` at the evenly spaced pooled draws; ``chol =
    A⁻¹ diag(exp(½ h_T))`` (lower, positive diagonal, ``LLᵀ = A⁻¹ΛA⁻ᵀ``)."""
    coefs, chol = fit.draw_parameters(25)
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    flat = np.round(np.linspace(0, C * D - 1, 25)).astype(int)
    c, p = np.divmod(flat, D)
    B = fit.reduced_form_draws().values[c, p]
    np.testing.assert_allclose(coefs, B, rtol=1e-11, atol=1e-13)
    A = _A(np.asarray(fit.param_draws.struct.a)[c, p])
    lam = np.exp(np.asarray(fit.param_draws.vol.core.h_T)[c, p])
    Ainv = np.linalg.solve(A, np.eye(3))
    sigma = Ainv @ (lam[..., None] * np.swapaxes(Ainv, -1, -2))
    np.testing.assert_allclose(chol @ np.swapaxes(chol, -1, -2), sigma, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(chol, np.linalg.cholesky(sigma), rtol=1e-11, atol=1e-16)
    assert np.all(np.triu(chol, 1) == 0) and np.all(np.diagonal(chol, axis1=1, axis2=2) > 0)
    # the last in-sample period by name is the same as date=None (no paths needed)
    c2, L2 = fit.draw_parameters(25, date=fit.periods[-1])
    np.testing.assert_array_equal(c2, coefs)
    np.testing.assert_array_equal(L2, chol)


def test_draw_parameters_at_a_date_uses_the_stored_paths(fit):
    """``chol = A⁻¹ diag(exp(½ h_t))`` (persistent) or with ``2 log o_t + log τ_t`` added
    (effective), from the path draws (every ``path_thin``-th stored draw)."""
    date = "2001Q3"
    t = int(fit.periods.get_loc(pd.Period(date, "Q")))
    coefs, chol = fit.draw_parameters(10, date=date)
    coefs_e, chol_e = fit.draw_parameters(10, date=date, which="effective")
    paths = fit.posterior["paths"]
    h = paths["h"].values  # (C, Dp, n, T_eff+1)
    s2 = paths["log_scale2"].values
    C, Dp = h.shape[:2]
    flat = np.round(np.linspace(0, C * Dp - 1, 10)).astype(int)
    c, q = np.divmod(flat, Dp)
    where = np.searchsorted(fit.draw_index, paths["draw"].values[q])
    A = _A(np.asarray(fit.param_draws.struct.a)[c, where])
    for L, lam in ((chol, h[c, q, :, t + 1]), (chol_e, h[c, q, :, t + 1] + s2[c, q, :, t])):
        ref = np.linalg.solve(A, np.exp(0.5 * lam)[:, None, :] * np.eye(3))
        np.testing.assert_allclose(L, ref, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(coefs, _B(fit, c, where), rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(coefs_e, coefs)
    # the effective variance at the last period also needs the paths (o_T, τ_T)
    _, last_e = fit.draw_parameters(10, which="effective")
    assert last_e.shape == (10, 3, 3)


def test_draw_parameters_date_errors(fit, sim_data):
    with pytest.raises(ValueError, match="outside the estimation sample"):
        fit.draw_parameters(5, date="1900Q1")
    with pytest.raises(ValueError, match="outside the estimation sample"):
        fit.draw_parameters(5, date=sim_data.index[0])  # presample (lags = 1)
    with pytest.raises(ValueError, match="not a period"):
        fit.draw_parameters(5, date="not-a-date")
    with pytest.raises(ValueError, match="which"):
        fit.draw_parameters(5, which="total")
    with pytest.raises(ValueError, match="positive"):
        fit.draw_parameters(0)
    no_paths = copy.copy(fit)
    no_paths.path_draws = None
    with pytest.raises(ValueError, match="store='paths'"):
        no_paths.draw_parameters(5, date="2001Q3")
    with pytest.raises(ValueError, match="store='paths'"):
        no_paths.draw_parameters(5, which="effective")
    with pytest.raises(ValueError, match="store='paths'"):
        no_paths.irf(horizon=2, date="2001Q3")
    no_paths.draw_parameters(5)  # persistent at the last period: the stored h_T
    no_paths.irf(horizon=2, date="2001Q3", shock="unit")  # unit shocks are date-free


def test_constant_volatility_is_date_free(sim_data):
    fit = mf.cached_fit(
        mf.model("structural", Constant("ig")), sim_data, plan=FitPlan(horizon=4, draws=30, seed=5)
    )
    assert fit.path_draws is None
    c1, L1 = fit.draw_parameters(8)
    c2, L2 = fit.draw_parameters(8, date="1990Q2", which="effective")
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(L1, L2)
    lam = np.exp(np.asarray(fit.param_draws.vol.core.log_lam))
    C, D = lam.shape[:2]
    c, p = np.divmod(np.round(np.linspace(0, C * D - 1, 8)).astype(int), D)
    A = _A(np.asarray(fit.param_draws.struct.a)[c, p])
    ref = np.linalg.solve(A, np.sqrt(lam[c, p])[:, None, :] * np.eye(3))
    np.testing.assert_allclose(L1, ref, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("date", [None, "2001Q3"])
def test_irf_equals_direct_phi_ainv_lambda(fit, date):
    """``Θ_h = Φ_h A⁻¹ Λ^{½}_date`` (persistent), per selected draw, with ``(chain, draw)`` kept."""
    H = 8
    res = fit.irf(horizon=H, draws=30, date=date)
    vals = res.data.values
    C = vals.shape[0]
    assert res.data.dims == ("chain", "draw", "horizon", "response", "shock")
    assert res.data.attrs["which"] == "persistent"
    if date is None:
        D = np.shape(fit.param_draws.struct.a)[1]
        pos, _ = F.select_draws(D, C, 30)
        where = pos
        lam = np.asarray(fit.param_draws.vol.core.h_T)
        lam = np.stack([lam[c, pos] for c in range(C)])
        assert res.data.attrs["date"] == str(fit.periods[-1])
    else:
        paths = fit.posterior["paths"]
        t = int(fit.periods.get_loc(pd.Period(date, "Q")))
        Dp = paths.sizes["draw"]
        pos, _ = F.select_draws(Dp, C, 30)
        where = np.searchsorted(fit.draw_index, paths["draw"].values[pos])
        lam = paths["h"].values[:, pos][..., t + 1]
    assert vals.shape == (C, pos.size, H + 1, 3, 3)
    s = fit.static.shapes
    for c in range(C):
        for j in (0, pos.size // 2, pos.size - 1):
            B = _B(fit, c, where[j])
            A = _A(np.asarray(fit.param_draws.struct.a)[c, where[j]])
            ref = _direct_irf(B, A, np.exp(lam[c, j]), s.lags, s.n_det, H)
            np.testing.assert_allclose(vals[c, j], ref, rtol=1e-12, atol=1e-15)
    fevd = res.fevd().values
    np.testing.assert_allclose(fevd.sum(axis=-1), 1.0, rtol=1e-12)


def test_unit_shock_irf_is_phi_ainv_and_date_free(fit):
    H = 5
    a = fit.irf(horizon=H, draws=20, shock="unit")
    b = fit.irf(horizon=H, draws=20, shock="unit", date="1995Q1", which="effective")
    np.testing.assert_array_equal(a.data.values, b.data.values)
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    pos, _ = F.select_draws(D, C, 20)
    s = fit.static.shapes
    A = _A(np.asarray(fit.param_draws.struct.a)[1, pos[2]])
    ref = _direct_irf(_B(fit, 1, pos[2]), A, np.ones(3), s.lags, s.n_det, H)
    np.testing.assert_allclose(a.data.values[1, 2], ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.diagonal(a.data.values[:, :, 0], axis1=-2, axis2=-1), 1.0)


def test_irf_effective_differs_only_by_the_transitory_scale(fit):
    p = fit.irf(horizon=3, draws=10, date="2001Q3")
    e = fit.irf(horizon=3, draws=10, date="2001Q3", which="effective")
    assert e.data.attrs["which"] == "effective"
    paths = fit.posterior["paths"]
    C = e.data.sizes["chain"]
    pos, _ = F.select_draws(paths.sizes["draw"], C, 10)
    t = int(fit.periods.get_loc(pd.Period("2001Q3", "Q")))
    scale = np.exp(0.5 * paths["log_scale2"].values[:, pos][..., t])  # o_t √τ_t per shock
    np.testing.assert_allclose(
        e.data.values, p.data.values * scale[:, :, None, None, :], rtol=1e-12, atol=1e-16
    )


def test_irf_argument_errors(fit):
    with pytest.raises(ValueError, match="shock"):
        fit.irf(shock="big")
    with pytest.raises(ValueError, match="horizon"):
        fit.irf(horizon=-1)
    with pytest.raises(ValueError, match="draws"):
        fit.irf(draws=0)


def test_reduced_form_parameters_and_irf(sim_data):
    """Reduced form (CCCM): ``coefs = Π``; ``chol = A⁻¹ diag(exp(½ log λ))``; the IRF equals
    the direct ``Φ_h A⁻¹ Λ^{½}`` (the contract module's reduced fit, shared per process)."""
    fit = mf.cached_fit(mf.model("reduced", Constant("ig")), sim_data)
    coefs, chol = fit.draw_parameters(6)
    Pi = np.asarray(fit.param_draws.glob.mean.Pi)
    a = np.asarray(fit.param_draws.struct.a)
    lam = np.exp(np.asarray(fit.param_draws.vol.core.log_lam))
    C, D = lam.shape[:2]
    c, p = np.divmod(np.round(np.linspace(0, C * D - 1, 6)).astype(int), D)
    np.testing.assert_array_equal(coefs, Pi[c, p])
    ref = np.linalg.solve(_A(a[c, p]), np.sqrt(lam[c, p])[:, None, :] * np.eye(3))
    np.testing.assert_allclose(chol, ref, rtol=1e-12, atol=1e-16)
    res = fit.irf(horizon=4, draws=4)
    pos, _ = F.select_draws(D, C, 4)
    s = fit.static.shapes
    for cc in range(C):
        for j, w in enumerate(pos):
            want = _direct_irf(Pi[cc, w], _A(a[cc, w]), lam[cc, w], s.lags, s.n_det, 4)
            np.testing.assert_allclose(res.data.values[cc, j], want, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- unit-scale t (N03 §5.4)


def _relabel_t(fit, nu, t_scale: str):
    """``fit`` (SV with outliers) relabelled as an outlier-t scale with ``t_scale`` and every
    stored ``ν_i = nu[i]``; the other draws stay those of ``fit``."""
    out = copy.copy(fit)
    vol = dataclasses.replace(fit.static.vol, transitory="outlier_t", t_scale=t_scale)
    out.static = dataclasses.replace(fit.static, vol=vol)
    trans = fit.param_draws.vol.trans
    nu = np.broadcast_to(np.asarray(nu, dtype=np.float64), np.shape(trans.log_p)).copy()
    t = OutlierTDraw(log_p=trans.log_p, log1m_p=trans.log1m_p, nu=nu)
    out.param_draws = dataclasses.replace(
        fit.param_draws, vol=dataclasses.replace(fit.param_draws.vol, trans=t)
    )
    return out


def test_unit_t_persistent_variance_includes_e_tau(fit):
    """N03 §5.4: under ``t_scale="unit"`` the persistent (one-s.d.) shock variance is
    ``e^h ν/(ν−2)`` (``E τ``), at the last period and at a date from the paths; standardized t,
    the effective variance ``e^h o²τ`` and unit shocks carry no ``ν`` factor; ``ν ≤ 2`` (infinite
    variance) raises."""
    nu = np.array([3.0, 5.0, 12.0])
    factor = np.sqrt(nu / (nu - 2.0))  # per shock (column of chol)
    unit, std = _relabel_t(fit, nu, "unit"), _relabel_t(fit, nu, "standardized")
    date = "2001Q3"
    for kw in ({}, {"date": date}):
        _, L0 = fit.draw_parameters(10, **kw)
        _, Lu = unit.draw_parameters(10, **kw)
        np.testing.assert_allclose(Lu, L0 * factor, rtol=1e-13, atol=1e-16)
        np.testing.assert_array_equal(std.draw_parameters(10, **kw)[1], L0)
        np.testing.assert_array_equal(
            unit.draw_parameters(10, which="effective", **kw)[1],
            fit.draw_parameters(10, which="effective", **kw)[1],
        )
    base = fit.irf(horizon=3, draws=10, date=date).data.values
    np.testing.assert_allclose(
        unit.irf(horizon=3, draws=10, date=date).data.values, base * factor, rtol=1e-12,
        atol=1e-16,
    )  # fmt: skip
    bad = _relabel_t(fit, [2.0, 5.0, 12.0], "unit")
    with pytest.raises(ValueError, match="ν ≤ 2"):
        bad.draw_parameters(3)
    with pytest.raises(ValueError, match="ν ≤ 2"):
        bad.irf(horizon=1, draws=2, date=date)
    bad.draw_parameters(3, date=date, which="effective")
    np.testing.assert_array_equal(
        bad.irf(horizon=2, draws=6, shock="unit").data.values,
        fit.irf(horizon=2, draws=6, shock="unit").data.values,
    )


@pytest.mark.slow
def test_persistent_variance_under_unit_t_on_a_fit(sim_data):
    """A fitted ``SV(tails="t", t_scale="unit")``: ``chol = A⁻¹ diag(√(e^{h_T} ν/(ν−2)))`` with
    the stored ``ν`` draws (every stored draw), and the IRF impact equals it."""
    m = mf.model("structural", SV(tails="t", t_scale="unit"))
    fit = mf.cached_fit(m, sim_data)
    assert fit.static.vol.t_scale == "unit" and fit.static.vol.transitory == "t"
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    nu = np.asarray(fit.param_draws.vol.trans.nu)
    lam = np.exp(np.asarray(fit.param_draws.vol.core.h_T)) * nu / (nu - 2.0)
    ref = np.linalg.solve(_A(np.asarray(fit.param_draws.struct.a)), np.sqrt(lam)[..., None, :]
                          * np.eye(3))  # fmt: skip
    _, chol = fit.draw_parameters(C * D)  # every stored draw, chain-major
    np.testing.assert_allclose(chol, ref.reshape(C * D, 3, 3), rtol=1e-12, atol=1e-16)
    pos, _ = F.select_draws(D, C, 10)
    impact = fit.irf(horizon=2, draws=10).data.values[:, :, 0]
    np.testing.assert_allclose(impact, ref[:, pos], rtol=1e-12, atol=1e-16)
