"""Phase 3 user-facing specifications (P3 DESIGN §3.2, §3.8, §4.1, §5.1, §7.1, §7.3, §7.4; WP27).

* ``DiscreteGrid``: normalisation, the Gruber–Kastner grid and its named spec.
* The five global-local families: defaults, keyword-only shared fields, validation (E-4, E-9,
  E-10, E-11), group layouts and counts (D-38, E-13), labels and spec round trips.
* ``TruncatedNormal``, ``CommonSV`` (E-14) and ``AsymmetricConjugate`` (E-15): defaults, labels,
  validation and specs.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.priors.asymmetric import AC_HYPER_NAMES
from rai_bvar.priors.global_local import FAMILIES, GlobalLocalPrior

P = rb.sv_priors
GK = rb.DiscreteGrid.gruber_kastner()
GL_CLASSES = [rb.Horseshoe, rb.NormalGamma, rb.R2D2, rb.DirichletLaplace, rb.SSVS]


def roundtrip(spec):
    return json.loads(json.dumps(spec))  # what a stored spec looks like


# ---------------------------------------------------------------------------------- DiscreteGrid


def test_discrete_grid_normalises_and_validates():
    g = rb.DiscreteGrid((0.5, 1), (1, 3))
    assert g.support == (0.5, 1.0) and g.weights == (0.25, 0.75) and g.size == 2
    assert rb.DiscreteGrid(g.support, g.weights) == g  # normalised weights are kept as given
    assert hash(g) == hash(rb.DiscreteGrid((0.5, 1.0), (0.25, 0.75)))
    for support, weights in [((), ()), ((0.5,), (1, 2)), ((0.5, 1), (1, 0)), ((0.5, 1), (1, -1))]:
        with pytest.raises(ValueError):
            rb.DiscreteGrid(support, weights)
    with pytest.raises(ValueError, match="finite"):
        rb.DiscreteGrid((0.5, np.inf), (1, 1))
    with pytest.raises(TypeError):
        rb.DiscreteGrid("ab", (1, 1))


def test_the_gruber_kastner_grid():
    support = np.array(GK.support)
    assert GK.size == 1000 and support[0] == 0.001 and support[-1] == 1.0
    np.testing.assert_allclose(np.diff(support), 0.999 / 999, rtol=1e-9)
    w = np.array(GK.weights)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(w / w[0], np.exp(-(support - support[0])), rtol=1e-12)
    assert rb.DiscreteGrid.gruber_kastner() is GK


def test_grid_specs():
    assert GK.to_spec() == {"class": "DiscreteGrid", "name": "gruber_kastner"}
    assert rb.DiscreteGrid.from_spec(GK.to_spec()) is GK
    g = rb.DiscreteGrid((0.1, 0.25, 0.5), (1, 1, 2))
    assert rb.DiscreteGrid.from_spec(roundtrip(g.to_spec())) == g
    # an equal grid is the named grid: one prior, one spec
    same = rb.DiscreteGrid(GK.support, GK.weights)
    assert same.to_spec() == GK.to_spec()
    with pytest.raises(ValueError, match="named"):
        rb.DiscreteGrid.from_spec({"class": "DiscreteGrid", "name": "uniform"})
    with pytest.raises(ValueError, match="not a DiscreteGrid"):
        rb.DiscreteGrid.from_spec({"class": "Beta", "a": 1, "b": 1})


# ---------------------------------------------------------------------------------- families


def test_defaults_are_gk_best_configurations():  # §3.2 "Defaults" (Q23)
    assert rb.Horseshoe().global_scale == 1.0
    ng = rb.NormalGamma()
    assert (ng.a, ng.b, ng.c) == (GK, 0.5, "a/2")
    assert (rb.R2D2().a, rb.R2D2().b) == (GK, 0.5)
    assert rb.DirichletLaplace().a == GK
    ssvs = rb.SSVS()
    assert (ssvs.c0, ssvs.c1, ssvs.semiautomatic, ssvs.p) == (0.01, 100.0, True, 0.5)
    for cls in GL_CLASSES:
        prior = cls()
        assert (prior.groups, prior.scale, prior.form) == ("olcl-lagwise", "psi", "auto")
        assert (prior.mean, prior.psi, prior.var_const, prior.contemp) == (None, "ar", 1e7, 1.0)
        assert prior.resolved_form == "structural"
        assert FAMILIES[cls.__name__] is cls


@pytest.mark.parametrize("cls", GL_CLASSES, ids=lambda c: c.__name__)
def test_shared_fields_are_keyword_only(cls):
    names = [f.name for f in dataclasses.fields(cls) if not f.kw_only]
    assert names == list(cls._OWN)
    with pytest.raises(TypeError):
        cls(*([None] * (len(names) + 1)))


def test_kinds_and_static_properties():
    kinds = {cls: cls.kind for cls in GL_CLASSES}
    assert kinds == {
        rb.Horseshoe: "horseshoe",
        rb.NormalGamma: "normal_gamma",
        rb.R2D2: "r2d2",
        rb.DirichletLaplace: "dirichlet_laplace",
        rb.SSVS: "ssvs",
    }
    assert rb.NormalGamma().a_grid == 1000 and rb.NormalGamma(a=0.3).a_grid == 0
    assert rb.Horseshoe().a_grid == 0 and rb.SSVS().a_grid == 0
    assert rb.SSVS(p=P.Beta(2, 3)).p_beta and not rb.SSVS().p_beta and not rb.R2D2().p_beta


def test_form_and_scale_validation():
    with pytest.raises(ValueError) as err:
        rb.Horseshoe(form="conjugate")
    assert str(err.value) == (  # E-4
        "global-local priors are not conjugate; use form='structural' (default) or 'reduced'"
    )
    with pytest.raises(ValueError, match="form must be one of"):
        rb.R2D2(form="kronecker")
    with pytest.raises(ValueError, match=r"scale must be one of \('psi', 'none'\)"):  # E-9
        rb.SSVS(scale="lambda")


@pytest.mark.parametrize(
    "groups",
    ["lagwise", [[0, 1], [1]], [[0, -1]], [[0.0, 1.0]], [0, 1], [[]], [[True, False]], 3],
    ids=repr,
)
def test_groups_validation(groups):
    with pytest.raises(ValueError) as err:  # E-9
        rb.Horseshoe(groups=groups)
    message = str(err.value)
    assert "Horseshoe.groups must be one of" in message
    for name in ("olcl-lagwise", "global", "equation-wise", "covariate-wise"):
        assert name in message
    assert "non-negative integer" in message


def test_shape_parameter_validation():  # E-10
    for cls in (rb.NormalGamma, rb.R2D2, rb.DirichletLaplace):
        for a in (0.0, -1.0, np.nan, "a", rb.DiscreteGrid((0.5, 1.5), (1, 1))):
            with pytest.raises(ValueError, match=rf"{cls.__name__}\.a must be .* \(0, 1\]"):
                cls(a=a)
        assert cls(a=2.5).a == 2.5  # a fixed a has no upper bound
        assert cls(a=rb.DiscreteGrid((0.25, 1.0), (1, 1))).a_grid == 2
    for cls in (rb.NormalGamma, rb.R2D2):
        with pytest.raises(ValueError, match=rf"{cls.__name__}\.b must be a positive number"):
            cls(b=0.0)
    with pytest.raises(ValueError, match=r"NormalGamma\.c must be a positive number or 'a/2'"):
        rb.NormalGamma(c=-1.0)
    with pytest.raises(ValueError, match=r"NormalGamma\.c"):
        rb.NormalGamma(c="a/3")
    assert rb.NormalGamma(c=1).c == 1.0
    with pytest.raises(ValueError, match=r"Horseshoe\.global_scale must be a positive number"):
        rb.Horseshoe(global_scale=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"c0": 0.0}, {"c0": 1.0, "c1": 1.0}, {"c0": 2.0, "c1": 1.0}, {"p": 0.0}, {"p": 1.0},
     {"p": "half"}, {"c1": np.inf}],
    ids=repr,
)  # fmt: skip
def test_ssvs_validation(kwargs):
    with pytest.raises(ValueError) as err:  # E-11
        rb.SSVS(**kwargs)
    assert str(err.value) == "SSVS needs 0 < c0 < c1 and p in (0, 1) or a Beta prior"


def test_ssvs_accepts_a_beta_prior_and_rejects_non_bool_switch():
    assert rb.SSVS(p={"class": "Beta", "a": 2, "b": 3}).p == P.Beta(2.0, 3.0)
    with pytest.raises(TypeError, match="semiautomatic must be a bool"):
        rb.SSVS(semiautomatic=1)


def test_minnesota_fields_are_validated_and_normalised_through_the_form_prior():
    hs = rb.Horseshoe(psi=[1, 2], mean={"b": 1, "a": 0}, var_const=10, form="reduced")
    assert hs.psi == (1.0, 2.0) and hs.mean == {"a": 0.0, "b": 1.0} and hs.var_const == 10.0
    internal = hs.minnesota()
    assert isinstance(internal, rb.Minnesota)
    assert (internal.form, internal.hierarchy, internal.psi) == ("reduced", "fixed", (1.0, 2.0))
    assert rb.Horseshoe().minnesota().form == "structural"
    with pytest.raises(ValueError, match="psi values must be positive"):
        rb.R2D2(psi=(1.0, -1.0))
    with pytest.raises(ValueError, match="must be positive"):
        rb.SSVS(contemp=0.0)


# ---------------------------------------------------------------------------------- groups


def test_group_layouts_and_counts():  # D-38; columns: the n variables at lag 1, then lag 2
    n, p = 3, 2
    olcl = rb.Horseshoe().group_ids(n, p)
    expected = np.array(
        [[0, 1, 1, 2, 3, 3],
         [1, 0, 1, 3, 2, 3],
         [1, 1, 0, 3, 3, 2]]
    )  # fmt: skip
    np.testing.assert_array_equal(olcl, expected)
    assert olcl.dtype == np.int32 and olcl.shape == (n, n * p)
    assert rb.Horseshoe().n_groups(n, p) == 2 * p
    assert (rb.Horseshoe(groups="global").group_ids(n, p) == 0).all()
    np.testing.assert_array_equal(
        rb.NormalGamma(groups="equation-wise").group_ids(n, p),
        np.repeat(np.arange(n), n * p).reshape(n, -1),
    )
    cov = rb.SSVS(groups="covariate-wise").group_ids(n, p)
    np.testing.assert_array_equal(cov, np.tile(np.arange(n * p), (n, 1)))
    counts = {
        g: rb.R2D2(groups=g).n_groups(n, p) for g in ("global", "equation-wise", "covariate-wise")
    }
    assert counts == {"global": 1, "equation-wise": n, "covariate-wise": n * p}
    # n = 1: no cross lags, so only the p own-lag groups exist
    assert rb.Horseshoe().n_groups(1, 3) == 3


def test_custom_groups_are_relabelled_and_shape_checked():
    hs = rb.Horseshoe(groups=np.array([[7, 2], [2, 9]]))
    assert hs.groups == ((7, 2), (2, 9))  # hashable
    np.testing.assert_array_equal(hs.group_ids(2, 1), [[1, 0], [0, 2]])
    assert hs.n_groups(2, 1) == 3
    with pytest.raises(ValueError) as err:  # E-13 (raised at fit by the WP31 builder)
        hs.group_ids(2, 2)
    assert str(err.value) == "groups array has shape (2, 2); expected (n, n·lags) = (2, 4)"


# ---------------------------------------------------------------------------------- labels, specs


def test_labels():
    assert rb.Horseshoe().label == "Horseshoe(structural, olcl-lagwise)"
    assert rb.NormalGamma(form="reduced", groups="global", c=1.0).label == (
        "NormalGamma(reduced, global, c=1.0)"
    )
    assert rb.SSVS(semiautomatic=False, c0=0.1).label == (
        "SSVS(structural, olcl-lagwise, c0=0.1, semiautomatic=False)"
    )
    grid = rb.R2D2(a=rb.DiscreteGrid((0.5, 1.0), (1, 1))).label
    assert grid.startswith("R2D2(structural, olcl-lagwise, a=") and len(grid.split("a=")[1]) == 7
    custom = rb.Horseshoe(groups=[[0, 1], [1, 0]]).label
    assert (
        custom.startswith("Horseshoe(structural, custom-")
        and custom != rb.Horseshoe(groups=[[1, 0], [0, 1]]).label
    )
    assert rb.DirichletLaplace(scale="none", psi=(1.0, 2.0)).label == (
        "DirichletLaplace(structural, olcl-lagwise, scale=none, psi=(1.0,2.0))"
    )


@pytest.mark.parametrize(
    "prior",
    [rb.Horseshoe(), rb.Horseshoe(global_scale=0.1, groups=[[0, 1], [1, 1]], form="reduced"),
     rb.NormalGamma(), rb.NormalGamma(a=0.5, b=2.0, c=1.0, psi=(0.5, 2.0), mean={"y": 0.4}),
     rb.R2D2(a=rb.DiscreteGrid((0.1, 0.5, 1.0), (1, 1, 1)), groups="equation-wise"),
     rb.DirichletLaplace(a=0.5, scale="none"), rb.SSVS(p=P.Beta(2.0, 3.0), semiautomatic=False),
     rb.SSVS(c0=0.05, c1=10.0, contemp=0.5, var_const=100.0)],
    ids=lambda p: p.label,
)  # fmt: skip
def test_spec_roundtrip(prior):
    spec = prior.to_spec()
    assert spec["class"] == type(prior).__name__
    assert json.loads(json.dumps(spec)) == spec  # plain JSON
    assert GlobalLocalPrior.from_spec(roundtrip(spec)) == prior


def test_spec_errors():
    with pytest.raises(ValueError, match="not a global-local prior spec"):
        GlobalLocalPrior.from_spec({"class": "Minnesota"})
    with pytest.raises(ValueError, match=r"unknown Horseshoe fields \['a'\]"):
        GlobalLocalPrior.from_spec({"class": "Horseshoe", "a": 1.0})


# ---------------------------------------------------------------------------------- SV priors


def test_truncated_normal():
    tn = P.TruncatedNormal(0.9, 0.04)
    assert (tn.mean, tn.var, tn.low, tn.high) == (0.9, 0.04, -1.0, 1.0)
    assert P.prior_from_spec(roundtrip(P.prior_to_spec(tn))) == tn
    with pytest.raises(ValueError, match="low < high"):
        P.TruncatedNormal(0.0, 1.0, low=1.0, high=1.0)
    with pytest.raises(ValueError, match="var must be positive"):
        P.TruncatedNormal(0.0, 0.0)


# ---------------------------------------------------------------------------------- CommonSV


def test_common_sv_defaults_label_and_spec():
    csv = rb.CommonSV()
    assert csv.rho_prior == P.TruncatedNormal(0.9, 0.04)
    assert csv.sigma2_prior == P.InvGamma(5.0, 0.04) and csv.block == 100
    assert csv.name == "CommonSV" and csv.label == "CommonSV"
    custom = rb.CommonSV(rho_prior=P.Beta(20, 1.5), sigma2_prior=P.ScaledChi2(0.1), block=7)
    assert custom.name == "CommonSV"
    assert custom.label.startswith("CommonSV(rho_prior=") and "block=7" in custom.label
    for model in (csv, custom):
        spec = model.to_spec()
        assert spec["class"] == "CommonSV" and json.loads(json.dumps(spec)) == spec
        assert rb.CommonSV.from_spec(roundtrip(spec)) == model
    assert rb.CommonSV(sigma2_prior={"class": "ScaledChi2", "B": 0.1}).sigma2_prior == P.ScaledChi2(
        0.1
    )


def test_common_sv_validation():  # E-14
    with pytest.raises(ValueError, match=r"CommonSV\.block must be at least 2"):
        rb.CommonSV(block=1)
    with pytest.raises(TypeError, match=r"CommonSV\.block must be an int"):
        rb.CommonSV(block=2.0)
    with pytest.raises(TypeError, match=r"CommonSV\.rho_prior must be a TruncatedNormal or Beta"):
        rb.CommonSV(rho_prior=P.Normal(0.9, 0.04))
    with pytest.raises(TypeError, match=r"CommonSV\.sigma2_prior must be a InvGamma or ScaledChi2"):
        rb.CommonSV(sigma2_prior=P.Beta(1, 1))
    with pytest.raises(ValueError, match=r"CommonSV.rho_prior must be truncated to .* \(-1, 1\)"):
        rb.CommonSV(rho_prior=P.TruncatedNormal(0.9, 0.04, low=0.0))
    with pytest.raises(ValueError, match="unknown CommonSV fields"):
        rb.CommonSV.from_spec({"class": "CommonSV", "rho": 0.9})


# ---------------------------------------------------------------------------------- AsymmetricConjugate


def test_asymmetric_conjugate_defaults_and_free_set():
    ac = rb.AsymmetricConjugate()
    assert (ac.lambda_, ac.cross, ac.contemp, ac.alpha) == (0.2, 0.5, 1.0, 2.0)
    assert (ac.hierarchy, ac.hyper, ac.form) == ("eb", "auto", "structural")
    assert ac.free == ("lambda", "cross")
    assert rb.AsymmetricConjugate(soc=1.0).free == ("lambda", "cross", "soc")
    assert rb.AsymmetricConjugate(hierarchy="fixed").free == ()
    assert rb.AsymmetricConjugate(hyper=("contemp", "lambda")).free == ("lambda", "contemp")
    assert AC_HYPER_NAMES == ("lambda", "alpha", "soc", "sur", "psi", "cross", "contemp")
    internal = ac.minnesota()
    assert (internal.form, internal.cross, internal.contemp, internal.hierarchy) == (
        "structural", 0.5, 1.0, "fixed"
    )  # fmt: skip


def test_asymmetric_conjugate_labels():
    assert rb.AsymmetricConjugate().label == (
        "AsymmetricConjugate(eb, log cross flat on [0.001, 10])"
    )
    assert rb.AsymmetricConjugate(hierarchy="fixed", cross=0.3).label == (
        "AsymmetricConjugate(fixed, cross=0.3)"
    )
    assert rb.AsymmetricConjugate(soc=1.0, hyper=("lambda", "contemp")).label == (
        "AsymmetricConjugate(eb, SOC, hyper=(lambda,contemp), log contemp flat on [0.001, 10])"
    )


@pytest.mark.parametrize("name", ["lambda_", "cross", "contemp"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf])
def test_asymmetric_conjugate_positive_parameters(name, value):  # E-15
    with pytest.raises(ValueError, match=rf"AsymmetricConjugate\.{name} must be a positive"):
        rb.AsymmetricConjugate(**{name: value})


@pytest.mark.parametrize(
    "kwargs",
    [{"hyper": ("lambda", "psi")}, {"hyper": ("psi", "cross")}, {"cross": 1.0, "hyper": ("cross", "psi")},
     {"hierarchy": "mh", "hyper": ("lambda", "psi")}],
    ids=repr,
)  # fmt: skip
def test_asymmetric_conjugate_rejects_psi_with_cross_shrinkage(kwargs):
    with pytest.raises(ValueError) as err:  # E-15
        rb.AsymmetricConjugate(**kwargs)
    assert str(err.value) == (
        "empirical Bayes over psi is unavailable with cross-variable shrinkage (Chan 2022 "
        "marginal likelihood over lambda, cross, contemp and alpha only); remove 'psi' from "
        "hyper or fix psi"
    )


def test_asymmetric_conjugate_allowed_psi_and_other_errors():
    rb.AsymmetricConjugate(cross=1.0, hyper=("lambda", "psi"))  # cross fixed at 1
    rb.AsymmetricConjugate(hierarchy="fixed", hyper=("lambda", "psi"))  # nothing is free
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        rb.AsymmetricConjugate(hyper=("kappa",))
    with pytest.raises(ValueError, match="'soc' is a hyperparameter but the soc dummy is off"):
        rb.AsymmetricConjugate(hyper=("lambda", "soc"))
    with pytest.raises(ValueError, match="hierarchy must be"):
        rb.AsymmetricConjugate(hierarchy="bayes")


@pytest.mark.parametrize(
    "prior",
    [rb.AsymmetricConjugate(), rb.AsymmetricConjugate(cross=0.3, psi=(1.0, 2.0), soc=1.0,
     hyper=("lambda", "cross", "soc"), hierarchy="mh", mh=rb.MHSettings(draws=50))],
)  # fmt: skip
def test_asymmetric_conjugate_spec_roundtrip(prior):
    spec = prior.to_spec()
    assert spec["class"] == "AsymmetricConjugate" and json.loads(json.dumps(spec)) == spec
    assert rb.AsymmetricConjugate.from_spec(roundtrip(spec)) == prior
