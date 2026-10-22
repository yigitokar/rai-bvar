"""Volatility prior arrays (DESIGN §2.6, §6.1, §6.3–6.4; D-2, D-8, D-9, D-13) and the assembly of
every builder into the WP1 ``PriorArrays`` template of each Phase 2 composition."""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.api import static_parts
from rai_bvar.components import SV, Constant, Linear
from rai_bvar.engines.mcmc.config import MeanStatic, Shapes, StaticConfig, scaling_mode
from rai_bvar.engines.mcmc.state import (
    ConstantPrior,
    Empty,
    EqPrior,
    PriorArrays,
    SVPrior,
    TransitoryPrior,
    assert_float64,
    prior_template,
)
from rai_bvar.priors.reduced import explicit_reduced, reduced_arrays
from rai_bvar.priors.scales import WindowScales, window_scales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.priors.sv import Beta, DiscreteUniform, InvGamma, Normal, OutlierPrior, ScaledChi2
from rai_bvar.priors.sv_arrays import (
    C_REL,
    constant_prior,
    outlier_beta,
    sv_prior_arrays,
    transitory_arrays,
)

GRID = tuple(range(3, 41))


def _scales(freq: str = "Q-DEC") -> WindowScales:
    return WindowScales(psi=np.array([0.5, 2.0, 0.8]), own_means=np.zeros(3), freq=freq)


def test_constant_prior_is_ig_one_plus_iota_half():
    scales = _scales()
    cp = constant_prior(scales, 3)
    assert isinstance(cp, ConstantPrior)
    np.testing.assert_array_equal(cp.shape, [1.5, 2.0, 2.5])  # (d − n + ι)/2, d = n + 2
    np.testing.assert_array_equal(cp.scale, scales.psi / 2)
    np.testing.assert_allclose(cp.scale / (cp.shape - 1), scales.psi / [1, 2, 3], rtol=1e-15)
    with pytest.raises(ValueError, match="3 variables"):
        constant_prior(scales, 4)


def test_sv_default_arrays_are_data_centred():
    scales = _scales()
    p = sv_prior_arrays(SV(), scales)
    assert isinstance(p, SVPrior)
    level = np.log(scales.psi / np.arange(1, 4))
    np.testing.assert_allclose(p.b_mu, level, rtol=1e-15)  # μ ~ N(log(ψ_ι/ι), 10)
    np.testing.assert_array_equal(p.B_mu, 10.0)
    np.testing.assert_array_equal(p.a_phi, 20.0)  # (φ+1)/2 ~ Beta(20, 1.5)
    np.testing.assert_array_equal(p.b_phi, 1.5)
    np.testing.assert_array_equal(p.B_sigma, 1.0)  # σ² ~ 1·χ²₁
    np.testing.assert_allclose(p.log_c, np.log(1e-12 * scales.psi), rtol=1e-14)  # D-2
    assert C_REL == 1e-12
    # placeholders of unused fields: finite and inert
    np.testing.assert_array_equal(p.c0, 1.0)
    np.testing.assert_array_equal(p.C0, 1.0)
    np.testing.assert_array_equal(p.B0, 1.0)
    np.testing.assert_array_equal(p.m0, 0.0)
    np.testing.assert_array_equal(p.V0, 1.0)
    for leaf in jax.tree.leaves(p):
        assert leaf.dtype == np.float64 and leaf.shape == (3,) and np.isfinite(leaf).all()


def test_sv_explicit_priors_and_offsets():
    scales = _scales()
    sv = SV(
        mu_prior=Normal(-1.0, 100.0),
        phi_prior=Beta(5.0, 1.5),
        sigma2_prior=InvGamma(2.5, 0.1),
        h0=3.0,
        offset=1e-4,
    )
    p = sv_prior_arrays(sv, scales)
    np.testing.assert_array_equal(p.b_mu, -1.0)
    np.testing.assert_array_equal(p.B_mu, 100.0)
    np.testing.assert_array_equal(p.a_phi, 5.0)
    np.testing.assert_array_equal(p.c0, 2.5)
    np.testing.assert_array_equal(p.C0, 0.1)
    np.testing.assert_array_equal(p.B_sigma, 1.0)  # placeholder (InvGamma in use)
    np.testing.assert_array_equal(p.B0, 3.0)  # h_0 ~ N(μ, B₀σ²)
    np.testing.assert_allclose(p.log_c, np.log(1e-4), rtol=1e-15)
    # stochvol "main" parity run (DESIGN §9.9): absolute zero offset → log c = −inf
    main = SV(
        mu_prior=Normal(0.0, 100.0),
        phi_prior=Beta(20, 1.5),
        sigma2_prior=ScaledChi2(0.1),
        offset=0.0,
        exact=False,
    )
    p = sv_prior_arrays(main, scales)
    np.testing.assert_array_equal(p.b_mu, 0.0)
    np.testing.assert_array_equal(p.B_mu, 100.0)
    np.testing.assert_array_equal(p.B_sigma, 0.1)
    assert np.all(p.log_c == -np.inf)


def test_sv_random_walk_arrays():
    scales = _scales()
    p = sv_prior_arrays(SV(rw=True), scales)
    np.testing.assert_array_equal(p.B_sigma, 0.1)  # RW default σ² ~ 0.1·χ²₁
    np.testing.assert_allclose(p.m0, np.log(scales.psi / np.arange(1, 4)), rtol=1e-15)
    np.testing.assert_array_equal(p.V0, 4.0)  # h_0 ~ N(log(ψ_ι/ι), 4)
    np.testing.assert_array_equal(p.b_mu, 0.0)  # no μ, φ in the RW model: placeholders
    np.testing.assert_array_equal(p.B_mu, 1.0)
    np.testing.assert_array_equal(p.a_phi, 1.0)
    p = sv_prior_arrays(SV(rw=True, rw_h0_var=2.0, sigma2_prior=ScaledChi2(0.3)), scales)
    np.testing.assert_array_equal(p.V0, 2.0)
    np.testing.assert_array_equal(p.B_sigma, 0.3)
    with pytest.raises(TypeError, match="SV"):
        sv_prior_arrays(Constant(), scales)


# --------------------------------------------------------------------------- transitory


def test_no_transitory_component_is_empty():
    assert isinstance(transitory_arrays(SV(), _scales(), GRID), Empty)
    with pytest.raises(ValueError, match="no outlier"):
        outlier_beta(SV(), "Q")


@pytest.mark.parametrize(
    ("freq", "sv", "beta"),
    [
        ("Q-DEC", SV(outliers=True), (2.5, 37.5)),  # SVO: one outlier every 4 years, 10 prior years
        ("M", SV(outliers=True), (2.5, 117.5)),
        ("Q-DEC", SV(outliers=True, tails="t"), (1.0, 39.0)),  # SVO-t: every 10 years
        ("M", SV(outliers=True, tails="t"), (1.0, 119.0)),
        ("QS-DEC", SV(outliers=True, outlier_prior=OutlierPrior(2.0, 5.0)), (2.5, 17.5)),
        ("Y-DEC", SV(outliers=True, outlier_prior=Beta(3.0, 50.0)), (3.0, 50.0)),
    ],
)
def test_outlier_beta_per_frequency(freq, sv, beta):
    tp = transitory_arrays(sv, _scales(freq), GRID)
    assert isinstance(tp, TransitoryPrior)
    np.testing.assert_array_equal(tp.a_p, beta[0])
    np.testing.assert_array_equal(tp.b_p, beta[1])
    assert tp.log_nu_w.shape == (3, len(GRID))
    assert outlier_beta(sv, freq) == Beta(*beta)


def test_unknown_frequency_needs_an_explicit_beta():
    with pytest.raises(ValueError, match="quarterly or monthly"):
        transitory_arrays(SV(outliers=True), _scales("Y-DEC"), GRID)
    tp = transitory_arrays(SV(outliers=True, outlier_prior=Beta(1.0, 9.0)), _scales("Y-DEC"), GRID)
    np.testing.assert_array_equal(tp.b_p, 9.0)


def test_outlier_prior_follows_the_window_frequency(data4_monthly):
    scales = window_scales(data4_monthly, rb.Minnesota(), 2, "c")
    tp = transitory_arrays(SV(outliers=True), scales, GRID)
    np.testing.assert_array_equal(tp.b_p, 117.5)


def test_nu_prior_weights_on_the_static_grid():
    scales = _scales()
    tp = transitory_arrays(SV(tails="t"), scales, GRID)
    np.testing.assert_allclose(tp.log_nu_w, -np.log(38.0), rtol=1e-15)  # DiscreteUniform(3, 40)
    np.testing.assert_array_equal(tp.a_p, 1.0)  # no outliers: placeholders
    fixed = SV(tails="t", nu_prior=DiscreteUniform(5, 5))  # SV-t parity run with ν = 5
    tp = transitory_arrays(fixed, scales, GRID)
    row = tp.log_nu_w[0]
    assert row[GRID.index(5)] == 0.0 and np.all(np.delete(row, GRID.index(5)) == -np.inf)
    tp = transitory_arrays(fixed, scales, (5,))
    np.testing.assert_array_equal(tp.log_nu_w, np.zeros((3, 1)))
    unit = SV(tails="t", t_scale="unit", nu_prior=DiscreteUniform(1, 10))
    tp = transitory_arrays(unit, scales, tuple(range(1, 11)))
    np.testing.assert_allclose(np.exp(tp.log_nu_w).sum(axis=1), 1.0, rtol=1e-14)
    with pytest.raises(ValueError, match="static grid"):
        transitory_arrays(SV(tails="t", nu_prior=DiscreteUniform(3, 50)), scales, GRID)
    with pytest.raises(ValueError, match="nu_grid"):
        transitory_arrays(SV(tails="t"), scales, ())
    tp = transitory_arrays(SV(outliers=True), scales, GRID)  # no t: finite placeholder weights
    np.testing.assert_allclose(tp.log_nu_w, -np.log(38.0), rtol=1e-15)


# --------------------------------------------------------------------------- buffer ownership


def test_builder_leaves_never_share_a_buffer():
    """Every leaf of every builder output is its own buffer, and none aliases the read-only
    ``WindowScales`` arrays or a caller's input: perturbing one prior leaf in place (a Geweke
    perturbation, a validation reference, a test) must not move another."""
    scales = WindowScales(
        psi=np.array([0.5, 2.0, 0.8]), own_means=np.array([1.0, 0.0, 1.0]), freq="Q-DEC"
    )
    prior = rb.Minnesota(form="structural", cross=0.5)
    dims = {"n": 3, "lags": 2, "n_det": 1}
    explicit_mean = np.zeros((3, 7))
    svs = [
        SV(),
        SV(rw=True),
        SV(h0=2.0, sigma2_prior=InvGamma(3.0, 0.5)),
        SV(mu_prior=Normal(0.0, 100.0), offset=0.0),
        SV(offset=1e-6),
        SV(rw=True, outliers=True, tails="t"),
    ]
    trees: list = [sv_prior_arrays(sv, scales) for sv in svs]
    trees += [
        transitory_arrays(SV(outliers=True, tails="t"), scales, GRID),
        transitory_arrays(SV(outliers=True), scales, GRID),
        transitory_arrays(SV(tails="t"), scales, GRID),
        constant_prior(scales, 3),
        structural_arrays(scales, prior, {}, mode="I", **dims),
        structural_arrays(scales, prior, {}, mode="II", **dims),
        contemp_arrays(scales, prior, 3, scaling="none"),
        contemp_arrays(scales, prior, 3, scaling="lambda"),
        reduced_arrays(scales, prior, {}, **dims),
        explicit_reduced(explicit_mean, 1.0, 0.0, 1.0),
    ]
    owned = [scales.psi, scales.own_means, explicit_mean]
    for tree in trees:
        leaves = [np.asarray(x) for x in jax.tree.leaves(tree)]
        for a, b in itertools.combinations(leaves, 2):
            assert not np.shares_memory(a, b), type(tree)
        for a, b in itertools.product(leaves, owned):
            assert not np.shares_memory(a, b), type(tree)
    # the reported probe: SV() placeholders c0, C0, B0, V0 (all 1) and RW's B_mu, a_phi, b_phi
    p = sv_prior_arrays(SV(), scales)
    p.V0[0] = 7.0
    for leaf in (p.c0, p.C0, p.B0):
        np.testing.assert_array_equal(leaf, 1.0)
    rw = sv_prior_arrays(SV(rw=True), scales)
    rw.B_mu[:] = 5.0
    np.testing.assert_array_equal(rw.a_phi, 1.0)
    np.testing.assert_array_equal(rw.b_phi, 1.0)


# --------------------------------------------------------------------------- assembly


def _model(form: str, vol) -> rb.BVAR:
    prior = rb.Minnesota(form=form, cross=0.5, hierarchy="fixed")
    return rb.BVAR(mean=Linear(prior=prior), vol=vol, lags=2)


COMPOSITIONS = {
    "structural x SV": ("structural", SV()),
    "structural x SV(rw, outliers, t)": ("structural", SV(rw=True, outliers=True, tails="t")),
    "structural x SV(t, B0, IG)": (
        "structural",
        SV(tails="t", h0=2.0, sigma2_prior=InvGamma(3.0, 0.5)),
    ),
    "structural x Constant(ig)": ("structural", Constant("ig")),
    "structural x Constant(iw) [Mode II]": ("structural", Constant("iw")),
    "structural x Constant(iw_equivalent)": ("structural", Constant("iw_equivalent")),
    "reduced x SV(outliers)": ("reduced", SV(outliers=True)),
    "reduced x Constant(iw_equivalent)": ("reduced", Constant("iw_equivalent")),
}


@pytest.mark.parametrize("name", list(COMPOSITIONS))
def test_builders_fill_the_prior_template_of_every_composition(data4, name):
    """What ``resolve.py`` (WP18) does: the builders' arrays have exactly the structure, shapes
    and dtypes of ``state.prior_template(static)`` for the composition's ``StaticConfig``."""
    form, vol = COMPOSITIONS[name]
    model = _model(form, vol)
    prior = model.mean.prior
    parts = static_parts(model)
    n, lags, n_det = data4.n_vars, model.lags, 1
    static = StaticConfig(
        Shapes.of(n, lags, n_det, 64),
        parts.struct,
        MeanStatic(kind=parts.mean_kind, coef_scaling=parts.coef_scaling),
        parts.coef,
        parts.vol,
    )
    scales = window_scales(data4, prior, lags, model.deterministic)
    mode = scaling_mode(static.struct, static.mean)
    dims = {"n": n, "lags": lags, "n_det": n_det}
    if static.mean.kind == "linear_structural":
        coef, shared = structural_arrays(
            scales, prior, {}, mode="II" if mode == "II" else "I", **dims
        )
    else:
        coef, shared = reduced_arrays(scales, prior, {}, **dims)
    contemp = contemp_arrays(scales, prior, n, scaling=static.struct.contemp_scaling)
    if isinstance(vol, Constant):
        core, trans = constant_prior(scales, n), Empty()
    else:
        core = sv_prior_arrays(vol, scales)
        trans = transitory_arrays(vol, scales, static.vol.nu_grid)
    arrays = PriorArrays(eq=EqPrior(coef, contemp, core, trans), shared=shared)
    template = prior_template(static)
    assert jax.tree.structure(arrays) == jax.tree.structure(template)
    for got, ref in zip(jax.tree.leaves(arrays), jax.tree.leaves(template), strict=True):
        assert np.shape(got) == ref.shape and np.asarray(got).dtype == ref.dtype
    assert_float64(arrays)
    # Mode II and iw_equivalent hold the λ-free a precisions ψ_j/κ_a; Mode I the scaled ones
    active = contemp.cmask > 0
    expected = scales.psi[None, : n - 1] / prior.contemp
    if static.struct.contemp_scaling == "none":
        expected = expected * (np.arange(1.0, n + 1) / scales.psi)[:, None]
    np.testing.assert_allclose(
        contemp.prec0[active], np.broadcast_to(expected, active.shape)[active]
    )
