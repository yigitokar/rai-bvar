"""Structural prior arrays (DESIGN §6.1, §4.7; N05 §4.1–4.4) and their implied reduced-form
moments (N05 §4.3; N08 §0, §B5).

Oracles: the Phase 1 ``ConjugateProblem.omega`` (Mode II at ``cross = 1``); the DESIGN §6.1 table
written out entry by entry; independent NumPy samplers of the priors (``helpers.py``) against
the NIW prior drawn with SciPy (KS tests) and against the closed forms of N05 §4.3 (moment match,
leak ratio, shares) and N05 §4.4 (Mode III growth)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rai_bvar as rb
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.engines.mcmc.state import ContempPriorArrays, SharedPrior
from rai_bvar.priors.reduced import reduced_arrays
from rai_bvar.priors.scales import WindowScales, window_scales
from rai_bvar.priors.structural import (
    ImpliedMoments,
    contemp_arrays,
    implied_reduced_form_moments,
    structural_arrays,
)
from rai_bvar.priors.sv_arrays import constant_prior

from .helpers import ig_draws, make_data, sample_niw, sample_structural

N_DET = {"n": 0, "c": 1, "ct": 2}


def _scales(n: int, psi=None, means=None) -> WindowScales:
    psi = np.array([0.5, 2.0, 0.8, 3.0, 1.5, 0.3][:n] if psi is None else psi, dtype=float)
    means = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0][:n] if means is None else means, dtype=float)
    return WindowScales(psi=psi, own_means=means, freq="Q-DEC")


# --------------------------------------------------------------------------- arrays


@pytest.mark.parametrize("deterministic", ["n", "c", "ct"])
def test_mode_ii_omega_equals_the_phase1_conjugate_omega(data4, deterministic):
    lags, hyper = 2, {"lambda": 0.3, "alpha": 1.5}
    problem = ConjugateProblem(data4, rb.Minnesota(form="conjugate"), lags, deterministic, None)
    h = problem.complete(hyper)
    prior = rb.Minnesota(form="structural")
    scales = window_scales(data4, prior, lags, deterministic)
    n_det = N_DET[deterministic]
    coef, shared = structural_arrays(scales, prior, hyper, n=4, lags=lags, n_det=n_det, mode="II")
    for i in range(4):
        np.testing.assert_allclose(1.0 / coef.prec0[i], problem.omega(h), rtol=1e-15, atol=0)
    np.testing.assert_array_equal(coef.mean0, problem.b.T)  # m_i: own first lag from tcodes
    np.testing.assert_array_equal(shared.Mc, problem.b[:, :3])  # M_c = M[:, :n−1]
    assert coef.prec0.dtype == shared.Mc.dtype == np.float64 and coef.group.dtype == np.int32


@pytest.mark.parametrize("mode", ["I", "II"])
def test_structural_arrays_follow_the_design_table(mode):
    n, lags, n_det = 3, 2, 2  # constant and trend
    lam, alpha, cross, kd = 0.25, 1.5, 0.5, 1e5
    prior = rb.Minnesota(form="structural", cross=cross, var_const=kd)
    scales = _scales(n)
    psi = scales.psi
    coef, shared = structural_arrays(
        scales, prior, {"lambda": lam, "alpha": alpha}, n=n, lags=lags, n_det=n_det, mode=mode
    )
    var = 1.0 / coef.prec0
    for i in range(n):
        iota = i + 1.0
        sbar = psi[i] / iota if mode == "I" else 1.0  # Mode II holds the λ-free Ω^{(i)}
        for r in range(n_det):
            assert var[i, r] == pytest.approx(kd * sbar, rel=1e-14)  # κ_d ψ_i/ι | κ_d
            assert coef.group[i, r] == 0 and coef.mean0[i, r] == 0.0
        for lag in range(1, lags + 1):
            for m in range(n):
                r = n_det + (lag - 1) * n + m
                c = 1.0 if m == i else cross**2
                expected = sbar * lam**2 * c / (lag**alpha * psi[m])
                assert var[i, r] == pytest.approx(expected, rel=1e-14)
                assert coef.group[i, r] == (1 if m == i else 2)
                own_first = m == i and lag == 1
                assert coef.mean0[i, r] == (scales.own_means[i] if own_first else 0.0)
    if mode == "I":  # e.g. own lag l: κ²/(ι l^α); cross: κ²κ_c²ψ_i/(ι l^α ψ_m)
        assert var[2, n_det + n + 2] == pytest.approx(lam**2 / (3 * 2**alpha), rel=1e-14)
        assert var[2, n_det + 0] == pytest.approx(
            lam**2 * cross**2 * psi[2] / (3 * psi[0]), rel=1e-14
        )
    np.testing.assert_array_equal(shared.Mc, coef.mean0[: n - 1].T)
    if mode == "I":  # iw_equivalent keeps the Mode I δ prior
        alias, _ = structural_arrays(
            scales, prior, {"lambda": lam, "alpha": alpha}, n=n, lags=lags, n_det=n_det,
            mode="iw_equivalent",
        )  # fmt: skip
        np.testing.assert_array_equal(alias.prec0, coef.prec0)


def test_mode_i_is_mode_ii_at_the_fixed_scale():
    scales, prior = _scales(4), rb.Minnesota(form="structural", cross=0.7)
    one, _ = structural_arrays(scales, prior, {}, n=4, lags=3, n_det=1, mode="I")
    two, _ = structural_arrays(scales, prior, {}, n=4, lags=3, n_det=1, mode="II")
    sbar = scales.psi / np.arange(1, 5)
    np.testing.assert_allclose(1 / one.prec0, sbar[:, None] / two.prec0, rtol=1e-14)


@pytest.mark.parametrize("contemp", [1.0, 2.5])
def test_contemp_arrays(contemp):
    n = 4
    scales = _scales(n)
    psi = scales.psi
    prior = rb.Minnesota(form="structural", contemp=contemp)
    mode1 = contemp_arrays(scales, prior, n, scaling="none")
    scaled = contemp_arrays(scales, prior, n, scaling="lambda")
    for arr in (mode1, scaled):
        assert arr.mean0.shape == arr.prec0.shape == arr.cmask.shape == (n, n - 1)
        np.testing.assert_array_equal(arr.mean0, 0.0)
    for i in range(n):
        for j in range(n - 1):
            if j < i:
                assert mode1.cmask[i, j] == scaled.cmask[i, j] == 1.0
                # Mode I: Var(a_ij) = κ_a ψ_i/(ι ψ_j); scaled: a_ij | λ_i ~ N(0, λ_i κ_a/ψ_j)
                assert 1 / mode1.prec0[i, j] == pytest.approx(
                    contemp * psi[i] / ((i + 1) * psi[j]), rel=1e-14
                )
                assert 1 / scaled.prec0[i, j] == pytest.approx(contemp / psi[j], rel=1e-14)
            else:  # inactive slots: mean 0, precision 1, mask 0 (DESIGN §2.3)
                assert mode1.cmask[i, j] == scaled.cmask[i, j] == 0.0
                assert mode1.prec0[i, j] == scaled.prec0[i, j] == 1.0
    with pytest.raises(ValueError, match="scaling"):
        contemp_arrays(scales, prior, n, scaling="conjugate")
    # no default: the DESIGN §13 three-argument call cannot silently give Mode I precisions to a
    # λ-scaled composition (Constant("iw"), Constant("iw_equivalent"))
    with pytest.raises(TypeError, match="scaling"):
        contemp_arrays(scales, prior, n)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        contemp_arrays(scales, prior, n, "lambda")  # type: ignore[misc]  # keyword-only


def test_single_equation_shapes():
    scales = _scales(1)
    prior = rb.Minnesota(form="structural")
    coef, shared = structural_arrays(scales, prior, {}, n=1, lags=2, n_det=1, mode="I")
    con = contemp_arrays(scales, prior, 1, scaling="none")
    assert coef.prec0.shape == (1, 3) and shared.Mc.shape == (3, 0)
    assert con.mean0.shape == con.prec0.shape == con.cmask.shape == (1, 0)
    im = implied_reduced_form_moments(coef, con, shared, np.array([0.5]), mode="I")
    np.testing.assert_allclose(im.var, 1 / coef.prec0.T, rtol=1e-15)
    np.testing.assert_array_equal(im.sigma_mean, [[0.5]])


def test_structural_arrays_validation():
    scales, prior = _scales(3), rb.Minnesota(form="structural")
    with pytest.raises(ValueError, match="mode"):
        structural_arrays(scales, prior, {}, n=3, lags=1, n_det=1, mode="III")
    with pytest.raises(ValueError, match="3 variables"):
        structural_arrays(scales, prior, {}, n=4, lags=1, n_det=1, mode="I")
    with pytest.raises(ValueError, match="lags"):
        structural_arrays(scales, prior, {}, n=3, lags=0, n_det=1, mode="I")
    with pytest.raises(KeyError):
        structural_arrays(scales, prior, {"kappa": 1.0}, n=3, lags=1, n_det=1, mode="I")
    with pytest.raises(TypeError, match="WindowScales"):
        contemp_arrays(scales.psi, prior, 3, scaling="none")


# --------------------------------------------------------------------------- implied moments


def _niw_reduced_var(scales: WindowScales, lam: float, alpha: float, n: int, lags: int, n_det):
    """``Var(B_{(l,m),j}) = κ²ψ_j/(l^αψ_m)``, det ``κ_d ψ_j``: the NIW marginal at cross = 1."""
    ref, _ = reduced_arrays(
        scales, rb.Minnesota(form="reduced", lambda_=lam, alpha=alpha), {}, n=n, lags=lags,
        n_det=n_det,
    )  # fmt: skip
    return 1.0 / ref.prec0.T


def test_mode_i_matches_the_niw_moments_exactly_and_by_monte_carlo(data4):
    """DESIGN §13 WP6: at ``κ_c = κ_a = 1`` Mode I has the NIW ``Var(B)`` and ``E[Σ] = diag ψ``
    (closed form exact; Monte Carlo ``Var(B)/NIW ∈ [0.98, 1.02]``, ``E[Σ_ii]/ψ_i ∈ [0.95, 1.05]``)."""
    n, lags, n_det = 4, 2, 1
    prior = rb.Minnesota(form="structural")
    scales = window_scales(data4, prior, lags, "c")
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det, mode="I")
    con = contemp_arrays(scales, prior, n, scaling="none")
    lam_prior = constant_prior(scales, n)
    vol_mean = lam_prior.scale / (lam_prior.shape - 1.0)
    np.testing.assert_allclose(vol_mean, scales.psi / np.arange(1, 5), rtol=1e-15)
    target = _niw_reduced_var(scales, prior.lambda_, prior.alpha, n, lags, n_det)

    closed = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="I")
    assert isinstance(closed, ImpliedMoments) and closed.method == "closed_form"
    np.testing.assert_allclose(closed.ratio_to(target), 1.0, rtol=1e-12)
    np.testing.assert_allclose(closed.sigma_mean, np.diag(scales.psi), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(closed.mean, coef.mean0.T)

    rng = np.random.default_rng(20261022)
    sum_sq, N = np.zeros_like(target), 0
    for _ in range(4):
        lam = ig_draws(rng, lam_prior.shape, lam_prior.scale, 50_000)
        B, _ = sample_structural(rng, coef, con, shared, lam, scale_coef=False, scale_contemp=False)
        sum_sq += ((B - closed.mean) ** 2).sum(axis=0)
        N += len(B)
    ratio = sum_sq / N / target
    assert 0.98 <= ratio.min() and ratio.max() <= 1.02, ratio

    sigma_sum, N = np.zeros(n), 0
    for _ in range(4):
        lam = ig_draws(rng, lam_prior.shape, lam_prior.scale, 250_000)
        _, Sigma = sample_structural(
            rng, coef, con, shared, lam, scale_coef=False, scale_contemp=False
        )
        sigma_sum += np.einsum("sii->i", Sigma)
        N += len(Sigma)
    sigma_ratio = sigma_sum / N / scales.psi
    assert 0.95 <= sigma_ratio.min() and sigma_ratio.max() <= 1.05, sigma_ratio


@pytest.mark.parametrize("mode", ["I", "II"])
def test_leak_ratio_closed_form_formula_and_monte_carlo(mode):
    """N05 §4.3: ``Var_D3(B_{(l,m),j})/[κ²ψ_j/(l^αψ_m)] = κ_c² + (1 − κ_c²)s_jm`` with
    ``s_jj = 1/j``, ``s_jm = 1/(m(m+1))`` (``m < j``), 0 (``m > j``), 1-based."""
    n, lags, n_det, cross = 5, 2, 1, 0.5
    scales = _scales(n)
    prior = rb.Minnesota(form="structural", cross=cross)
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det, mode=mode)
    con = contemp_arrays(scales, prior, n, scaling="none" if mode == "I" else "lambda")
    vol_mean = scales.psi / np.arange(1, n + 1)
    closed = implied_reduced_form_moments(coef, con, shared, vol_mean, mode=mode)
    ratio = closed.ratio_to(_niw_reduced_var(scales, prior.lambda_, prior.alpha, n, lags, n_det))

    jj, mm = np.meshgrid(np.arange(1.0, n + 1), np.arange(1.0, n + 1), indexing="ij")
    s = np.where(mm == jj, 1 / jj, np.where(mm < jj, 1 / (mm * (mm + 1)), 0.0))
    np.testing.assert_allclose(closed.shares, s, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(closed.shares.sum(axis=1), 1.0, rtol=1e-14)
    var_of = np.tile(np.arange(n), lags)
    expected = cross**2 + (1 - cross**2) * s[:, var_of].T  # (n·p, n)
    np.testing.assert_allclose(ratio[n_det:], expected, rtol=1e-12)
    np.testing.assert_allclose(ratio[:n_det], 1.0, rtol=1e-12)  # deterministic terms: exact
    # N05 §4.3 numbers: own first lags 1, .625, .5, .4375; equation 2 on variable 1: .625
    np.testing.assert_allclose(np.diag(ratio[n_det : n_det + n])[:4], [1, 0.625, 0.5, 0.4375])
    assert ratio[n_det + 0, 1] == pytest.approx(0.625, rel=1e-12)

    mc = implied_reduced_form_moments(coef, con, shared, vol_mean, mode=mode, draws=300_000)
    assert mc.method == "monte_carlo" and mc.draws == 300_000
    np.testing.assert_allclose(mc.var / closed.var, 1.0, atol=0.03)
    np.testing.assert_allclose(mc.shares, closed.shares, atol=0.01)


@pytest.mark.parametrize(("cross", "contemp"), [(1.0, 1.0), (0.5, 2.0)])
def test_iw_equivalent_implied_moments(cross, contemp):
    """R8 summary of ``Constant("iw_equivalent")`` (N08 §0, §B5): Mode I ``δ``, λ-scaled ``a``
    with ``P̃_a = ψ_j/κ_a``, so ``E[Var a_ij] = E[λ_i]κ_a/ψ_j = κ_a ψ_i/(ι ψ_j)``, the Mode I
    a-variance: the moments equal Mode I's. Oracle for ``E[Σ_ii] = c_i ψ_i``: ``u_i = ε_i −
    a_i'u_{<i}`` with ``a_i ⊥ u_{<i}`` gives ``ι c_i = 1 + κ_a Σ_{j<i} c_j`` (``c ≡ 1`` at
    ``κ_a = 1``: the ``IW(diag ψ, n + 2)`` mean). At ``κ_c = κ_a = 1``, ``Var(B)`` is the NIW
    marginal."""
    n, lags, n_det = 5, 2, 1
    scales = _scales(n)
    prior = rb.Minnesota(form="structural", cross=cross, contemp=contemp)
    coef, shared = structural_arrays(
        scales, prior, {}, n=n, lags=lags, n_det=n_det, mode="iw_equivalent"
    )
    con = contemp_arrays(scales, prior, n, scaling="lambda")
    lam_prior = constant_prior(scales, n)
    vol_mean = lam_prior.scale / (lam_prior.shape - 1.0)
    closed = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="iw_equivalent")
    assert closed.method == "closed_form"

    c = np.zeros(n)
    for i in range(n):
        c[i] = (1.0 + contemp * c[:i].sum()) / (i + 1.0)
    np.testing.assert_allclose(np.diag(closed.sigma_mean), c * scales.psi, rtol=1e-12)
    off = closed.sigma_mean - np.diag(np.diag(closed.sigma_mean))
    np.testing.assert_allclose(off, 0.0, atol=1e-14)  # zero contemporaneous means
    np.testing.assert_allclose(closed.shares.sum(axis=1), 1.0, rtol=1e-14)
    mode_i = implied_reduced_form_moments(
        coef, contemp_arrays(scales, prior, n, scaling="none"), shared, vol_mean, mode="I"
    )
    for name in ("var", "sigma_mean", "shares"):
        np.testing.assert_allclose(getattr(closed, name), getattr(mode_i, name), rtol=1e-12)
    target = _niw_reduced_var(scales, prior.lambda_, prior.alpha, n, lags, n_det)
    if cross == contemp == 1.0:
        np.testing.assert_allclose(closed.sigma_mean, np.diag(scales.psi), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(closed.ratio_to(target), 1.0, rtol=1e-12)
    else:  # κ_a = 2 inflates E[Σ_ii] for i ≥ 2: c = 1, 1.5, 2, 2.5, 3
        np.testing.assert_allclose(c, [1.0, 1.5, 2.0, 2.5, 3.0], rtol=1e-14)

    mc = implied_reduced_form_moments(
        coef, con, shared, vol_mean, mode="iw_equivalent", draws=200_000
    )
    assert mc.method == "monte_carlo" and mc.draws == 200_000
    np.testing.assert_allclose(mc.var / closed.var, 1.0, atol=0.03)
    np.testing.assert_allclose(np.diag(mc.sigma_mean) / np.diag(closed.sigma_mean), 1, atol=0.03)
    np.testing.assert_allclose(mc.shares, closed.shares, atol=0.01)

    # the mode decides which blocks are λ-scaled: reading the same arrays as Mode I (a unscaled,
    # Var a_ij = κ_a/ψ_j) changes E[Σ] and Var(B); as Mode II (δ also scaled) changes Var(B)
    as_i = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="I")
    assert np.max(np.abs(np.diag(as_i.sigma_mean) / np.diag(closed.sigma_mean) - 1)) > 0.1
    assert np.max(np.abs(as_i.var / closed.var - 1)) > 0.1
    as_ii = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="II")
    np.testing.assert_allclose(as_ii.sigma_mean, closed.sigma_mean, rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(as_ii.var / closed.var - 1)) > 0.1


def test_leak_ratio_independent_monte_carlo():
    """The Mode I prior sampled from its full definition (λ ~ IG) leaks as the closed form says."""
    n, lags, n_det = 4, 1, 1
    scales = _scales(n)
    prior = rb.Minnesota(form="structural", cross=0.5)
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det, mode="I")
    con = contemp_arrays(scales, prior, n, scaling="none")
    lam_prior = constant_prior(scales, n)
    closed = implied_reduced_form_moments(
        coef, con, shared, lam_prior.scale / (lam_prior.shape - 1), mode="I"
    )
    rng = np.random.default_rng(7)
    lam = ig_draws(rng, lam_prior.shape, lam_prior.scale, 300_000)
    B, _ = sample_structural(rng, coef, con, shared, lam, scale_coef=False, scale_contemp=False)
    mc_var = ((B - closed.mean) ** 2).mean(axis=0)
    np.testing.assert_allclose(mc_var / closed.var, 1.0, atol=0.03)


def test_mode_iii_growth_is_reported():
    """N05 §4.4: ``Var(a_ij) = κ_a ψ_i/ψ_j`` with ``E[λ_i] = ψ_i`` gives ``E[Σ_ii] =
    (1 + κ_a)^{i−1}ψ_i`` (CKY 2024 §2: ``2^{i−1}`` at ``κ_a = 1``)."""
    n = 6
    scales = _scales(n)
    psi = scales.psi
    prior = rb.Minnesota(form="structural")
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=1, n_det=1, mode="I")
    for ka in (1.0, 0.5):
        base = contemp_arrays(scales, prior, n, scaling="none")
        prec = np.where(base.cmask > 0, psi[None, : n - 1] / (ka * psi[:, None]), 1.0)
        con = ContempPriorArrays(mean0=base.mean0, prec0=prec, cmask=base.cmask)
        im = implied_reduced_form_moments(coef, con, shared, psi, mode="I")
        np.testing.assert_allclose(
            np.diag(im.sigma_mean) / psi, (1 + ka) ** np.arange(n), rtol=1e-12
        )


def test_general_contemporaneous_means_closed_form_vs_monte_carlo():
    """Non-zero ``a`` means (e.g. ``explicit_reduced`` or Chan's Cor. 3): the exact second-moment
    recursion against an independent sampler with ``λ`` at its mean."""
    n, lags, n_det = 4, 1, 1
    scales = _scales(n)
    prior = rb.Minnesota(form="structural", contemp=0.5)
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det, mode="I")
    base = contemp_arrays(scales, prior, n, scaling="none")
    means = np.array([[0, 0, 0], [0.4, 0, 0], [-0.3, 0.8, 0], [0.2, -0.5, 0.6]])
    con = ContempPriorArrays(mean0=means * base.cmask, prec0=base.prec0, cmask=base.cmask)
    vol_mean = np.array([0.7, 1.1, 0.4, 2.0])
    closed = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="I")
    assert abs(closed.sigma_mean[1, 0]) > 0.05  # off-diagonal E[Σ] no longer vanishes
    np.testing.assert_allclose(closed.sigma_mean, closed.sigma_mean.T, rtol=1e-14)

    rng = np.random.default_rng(11)
    lam = np.broadcast_to(vol_mean, (300_000, n))
    B, Sigma = sample_structural(rng, coef, con, shared, lam, scale_coef=False, scale_contemp=False)
    scale = np.sqrt(np.outer(np.diag(closed.sigma_mean), np.diag(closed.sigma_mean)))
    np.testing.assert_allclose(Sigma.mean(axis=0) / scale, closed.sigma_mean / scale, atol=0.02)
    np.testing.assert_allclose(((B - closed.mean) ** 2).mean(axis=0) / closed.var, 1, atol=0.03)
    mc = implied_reduced_form_moments(coef, con, shared, vol_mean, mode="I", draws=200_000)
    np.testing.assert_allclose(mc.sigma_mean / scale, closed.sigma_mean / scale, atol=0.02)


def test_implied_moments_validation():
    scales, prior = _scales(3), rb.Minnesota(form="structural")
    coef, shared = structural_arrays(scales, prior, {}, n=3, lags=1, n_det=1, mode="I")
    con = contemp_arrays(scales, prior, 3, scaling="none")
    vm = np.ones(3)
    with pytest.raises(ValueError, match="mode"):
        implied_reduced_form_moments(coef, con, shared, vm, mode="III")
    with pytest.raises(ValueError, match="vol_mean"):
        implied_reduced_form_moments(coef, con, shared, np.ones(2), mode="I")
    with pytest.raises(ValueError, match="vol_mean"):
        implied_reduced_form_moments(coef, con, shared, -vm, mode="I")
    with pytest.raises(ValueError, match="D3"):
        implied_reduced_form_moments(coef, con, SharedPrior(Mc=shared.Mc + 1), vm, mode="I")
    with pytest.raises(ValueError, match="draws"):
        implied_reduced_form_moments(coef, con, shared, vm, mode="I", draws=-1)
    im = implied_reduced_form_moments(coef, con, shared, vm, mode="I")
    with pytest.raises(ValueError, match="shape"):
        im.ratio_to(np.ones((2, 2)))


# --------------------------------------------------------------------------- distributions (KS)


def _ks_ok(x: np.ndarray, y: np.ndarray) -> float:
    return stats.ks_2samp(x, y).pvalue


def _sigma_functionals(Sigma: np.ndarray) -> dict[str, np.ndarray]:
    sd = np.sqrt(np.einsum("sii->si", Sigma))
    corr = Sigma / (sd[:, :, None] * sd[:, None, :])
    n = Sigma.shape[1]
    out = {"logdet": np.linalg.slogdet(Sigma)[1]}
    out |= {f"S{i}{i}": Sigma[:, i, i] for i in range(n)}
    out |= {f"corr{i}{j}": corr[:, i, j] for i in range(n) for j in range(i)}
    return out


def test_mode_ii_prior_is_the_niw_prior():
    """N05 §4.2: Mode II at ``κ_c = κ_a = 1`` is the Phase 1 NIW prior. KS on ``Σ`` functionals
    and on coefficient marginals (own first lag, cross lag, intercept)."""
    data = make_data(3, 80, seed=6)
    lags, n = 1, 3
    prior = rb.Minnesota(form="structural", lambda_=0.4)
    scales = window_scales(data, prior, lags, "c")
    problem = ConjugateProblem(data, rb.Minnesota(form="conjugate", lambda_=0.4), lags, "c", None)
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=1, mode="II")
    con = contemp_arrays(scales, prior, n, scaling="lambda")
    lam_prior = constant_prior(scales, n)
    rng = np.random.default_rng(5)
    size = 20_000
    lam = ig_draws(rng, lam_prior.shape, lam_prior.scale, size)
    B2, S2 = sample_structural(rng, coef, con, shared, lam, scale_coef=True, scale_contemp=True)
    omega = problem.omega(problem.complete({}))
    B_niw, S_niw = sample_niw(rng, problem.b, omega, scales.psi, size)
    pvals = {k: _ks_ok(v, _sigma_functionals(S_niw)[k]) for k, v in _sigma_functionals(S2).items()}
    for r, j in [(1, 0), (2, 1), (3, 2), (2, 0), (1, 2), (0, 1)]:  # own lags, cross lags, constant
        pvals[f"B{r}{j}"] = _ks_ok(B2[:, r, j], B_niw[:, r, j])
    assert min(pvals.values()) > 1e-3, pvals


def test_iw_equivalent_sigma_is_inverse_wishart():
    """Chan (2022) Prop. 1; N08 §B5 items 3–4: ``a_ij | λ_i ~ N(0, λ_i/ψ_j)``, ``λ_i ~ IG(1 + ι/2,
    ψ_i/2)`` gives ``Σ ~ IW(diag ψ, n + 2)``; ``Σ_ii ~ IG(3/2, ψ_i/2)`` at every position."""
    n = 5
    scales = _scales(n)
    prior = rb.Minnesota(form="structural")
    coef, shared = structural_arrays(scales, prior, {}, n=n, lags=1, n_det=1, mode="I")
    con = contemp_arrays(scales, prior, n, scaling="lambda")
    lam_prior = constant_prior(scales, n)
    rng = np.random.default_rng(9)
    size = 40_000
    lam = ig_draws(rng, lam_prior.shape, lam_prior.scale, size)
    _, S = sample_structural(rng, coef, con, shared, lam, scale_coef=False, scale_contemp=True)
    S_iw = stats.invwishart(df=n + 2, scale=np.diag(scales.psi)).rvs(size=size, random_state=rng)
    ours, ref = _sigma_functionals(S), _sigma_functionals(S_iw)
    pvals = {k: _ks_ok(ours[k], ref[k]) for k in ("logdet", "S00", "S44", "corr40", "corr21")}
    for i in (0, 4):
        ig = stats.invgamma(a=1.5, scale=scales.psi[i] / 2)
        pvals[f"IG{i}"] = stats.kstest(S[:, i, i], ig.cdf).pvalue
    assert min(pvals.values()) > 1e-3, pvals
    # Mode I shares the λ law but its a_i do not scale with λ_i: Σ_55 is then not IG(3/2, ψ_5/2)
    con1 = contemp_arrays(scales, prior, n, scaling="none")
    _, S1 = sample_structural(rng, coef, con1, shared, lam, scale_coef=False, scale_contemp=False)
    assert _ks_ok(S1[:, 4, 4], ref["S44"]) < 1e-20
