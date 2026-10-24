"""User-facing Phase 2 specifications (DESIGN §6.3–6.4, §11.2, §11.4–11.5).

* ``rb.sv_priors``: ``Normal``, ``Beta``, ``ScaledChi2``, ``InvGamma``, ``DiscreteUniform`` and
  ``OutlierPrior`` validate at construction, store canonical Python numbers and round-trip
  through ``prior_to_spec``/``prior_from_spec``; ``OutlierPrior.resolve`` gives the CCMM Beta
  priors for quarterly and monthly data (N03 §0, §3.1).
* ``SV``: defaults (D-13), names and labels of every variant, validation, settings a variant
  does not use are ignored (reset to their defaults), canonical ``None`` for variant defaults,
  ``to_spec``/``from_spec``.
* ``Constant`` priors, names and grid labels; ``Minnesota(form="auto", contemp)``.
* ``MCMCSettings``, ``Runtime`` and ``FitPlan`` defaults and validation (DESIGN §11.2).
* ``import rai_bvar`` imports no JAX; lazy package attributes (``from rai_bvar import *`` and
  ``hasattr`` work while a lazy module is not part of the build).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.priors.sv import periods_per_year, prior_from_spec, prior_to_spec

P = rb.sv_priors

# =========================================================================== SV priors

PRIOR_EXAMPLES = [
    P.Normal(0.0, 10.0),
    P.Normal(-1.5, 0.25),
    P.Beta(20.0, 1.5),
    P.ScaledChi2(0.1),
    P.InvGamma(2.5, 0.025),
    P.DiscreteUniform(3, 40),
    P.DiscreteUniform(5, 5),
    P.OutlierPrior(),
    P.OutlierPrior(every_years=10.0, prior_years=20.0),
]


@pytest.mark.parametrize("prior", PRIOR_EXAMPLES, ids=repr)
def test_sv_priors_round_trip_through_json(prior):
    spec = prior_to_spec(prior)
    assert spec["class"] == type(prior).__name__
    rebuilt = prior_from_spec(json.loads(json.dumps(spec)))
    assert rebuilt == prior
    assert type(rebuilt) is type(prior)
    assert repr(rebuilt) == repr(prior)


@pytest.mark.parametrize(
    ("make", "forms"),
    [
        (P.Normal, [(0, 10), (0.0, 10.0), (np.int64(0), np.float32(10.0)), (np.float64(0), 10)]),
        (P.Beta, [(20, 1.5), (np.int32(20), np.float64(1.5)), (20.0, np.float32(1.5))]),
        (P.ScaledChi2, [(1,), (1.0,), (np.int64(1),), (np.float32(1.0),)]),
        (P.InvGamma, [(2, 1), (2.0, 1.0), (np.float32(2.0), np.int8(1))]),
        (P.DiscreteUniform, [(3, 40), (np.int64(3), np.int32(40))]),
        (P.OutlierPrior, [(4, 10), (4.0, 10.0), (np.float32(4.0), np.int64(10))]),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_sv_priors_store_canonical_python_numbers(make, forms):
    priors = [make(*f) for f in forms]
    assert all(p == priors[0] for p in priors)
    assert len({repr(p) for p in priors}) == 1
    for p in priors:
        for f in dataclasses.fields(p):
            assert type(getattr(p, f.name)) in (int, float)


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda: P.Normal(0.0, 0.0), ValueError),
        (lambda: P.Normal(0.0, -1.0), ValueError),
        (lambda: P.Normal(np.nan, 1.0), ValueError),
        (lambda: P.Normal(0.0, np.inf), ValueError),
        (lambda: P.Normal("0", 1.0), TypeError),
        (lambda: P.Normal(True, 1.0), TypeError),
        (lambda: P.Beta(0.0, 1.0), ValueError),
        (lambda: P.Beta(1.0, -2.0), ValueError),
        (lambda: P.ScaledChi2(0.0), ValueError),
        (lambda: P.ScaledChi2(None), TypeError),
        (lambda: P.InvGamma(-1.0, 1.0), ValueError),
        (lambda: P.InvGamma(1.0, 0.0), ValueError),
        (lambda: P.DiscreteUniform(3.0, 40), TypeError),
        (lambda: P.DiscreteUniform(3, True), TypeError),
        (lambda: P.DiscreteUniform(0, 40), ValueError),
        (lambda: P.DiscreteUniform(10, 5), ValueError),
        (lambda: P.OutlierPrior(every_years=0.0), ValueError),
        (lambda: P.OutlierPrior(prior_years=-1.0), ValueError),
    ],
)
def test_sv_priors_reject_invalid_parameters(make, error):
    with pytest.raises(error):
        make()


def test_prior_spec_helpers_reject_unknown_input():
    with pytest.raises(TypeError, match="not an SV prior"):
        prior_to_spec(rb.BetaPrior(mode=0.5, sd=0.1))
    with pytest.raises(ValueError, match="'class'"):
        prior_from_spec({"class": "Gamma", "a": 1.0})
    with pytest.raises(ValueError, match="'class'"):
        prior_from_spec({"mean": 0.0, "var": 1.0})
    with pytest.raises(ValueError, match="unknown fields"):
        prior_from_spec({"class": "Normal", "mean": 0.0, "var": 1.0, "sd": 1.0})


def test_discrete_uniform_support_is_the_nu_grid():
    assert P.DiscreteUniform(3, 40).support == tuple(range(3, 41))
    assert len(P.DiscreteUniform(3, 40).support) == 38  # G (DESIGN §2.2)
    assert P.DiscreteUniform(5, 5).support == (5,)


def test_beta_mean():
    assert P.Beta(2.5, 37.5).mean == pytest.approx(0.0625)


# N03 §3.1 table: CCMM SVO / SVO-t priors for monthly (f = 12) and quarterly (f = 4) data.
@pytest.mark.parametrize(
    ("prior", "freq", "a", "b"),
    [
        (P.OutlierPrior(), "Q-DEC", 2.5, 37.5),
        (P.OutlierPrior(), "M", 2.5, 117.5),
        (P.OutlierPrior(every_years=10.0), "Q", 1.0, 39.0),
        (P.OutlierPrior(every_years=10.0), "M", 1.0, 119.0),
        (P.OutlierPrior(every_years=2.0, prior_years=20.0), "Q-MAR", 10.0, 70.0),
    ],
)
def test_outlier_prior_resolves_with_the_data_frequency(prior, freq, a, b):
    beta = prior.resolve(freq)
    assert isinstance(beta, P.Beta)
    assert (beta.a, beta.b) == pytest.approx((a, b), rel=1e-15)
    f = periods_per_year(freq)
    assert beta.mean == pytest.approx(1.0 / (f * prior.every_years))  # one outlier per every_years
    assert beta.a + beta.b == pytest.approx(f * prior.prior_years)  # prior sample of prior_years


@pytest.mark.parametrize(
    ("freq", "f"),
    [("Q", 4), ("Q-DEC", 4), ("QS-DEC", 4), ("q", 4), ("M", 12), ("MS", 12), ("ME", 12)],
)
def test_periods_per_year(freq, f):
    assert periods_per_year(freq) == f


def test_periods_per_year_follows_vardata(sim_data_stationary):
    assert periods_per_year(sim_data_stationary.freq) == 4


@pytest.mark.parametrize("freq", ["Y-DEC", "W-SUN", "D", "B", ""])
def test_outlier_prior_needs_an_explicit_beta_for_other_frequencies(freq):
    with pytest.raises(ValueError, match=r"quarterly or monthly .*Beta\(a, b\)"):
        P.OutlierPrior().resolve(freq)


def test_outlier_prior_rejects_a_prior_mean_of_one():
    with pytest.raises(ValueError, match="f·every_years > 1"):
        P.OutlierPrior(every_years=0.25).resolve("Q")  # mean 1/(4·0.25) = 1
    assert P.OutlierPrior(every_years=0.25).resolve("M").mean == pytest.approx(1 / 3)


def test_sv_priors_are_exported_without_clashing_with_glp_priors():
    assert rb.sv_priors is importlib.import_module("rai_bvar.priors.sv")
    assert rb.sv_priors.Beta is not rb.BetaPrior
    assert "Beta" not in rb.__all__ and "sv_priors" in rb.__all__


# =========================================================================== Constant


@pytest.mark.parametrize(
    ("prior", "name"),
    [
        ("auto", "Constant"),
        ("iw", "Constant"),
        ("ig", "Constant"),
        ("iw_equivalent", "Constant(iw_equivalent)"),
    ],
)
def test_constant_priors_and_names(prior, name):
    assert rb.Constant(prior).name == name
    assert rb.Constant(prior=prior).prior == prior


@pytest.mark.parametrize(
    ("prior", "label"),
    [
        ("auto", "Constant"),
        ("iw", "Constant"),  # Phase 1 grid names unchanged
        ("ig", "Constant(ig)"),  # Mode I is named apart from Mode II in grids
        ("iw_equivalent", "Constant(iw_equivalent)"),
    ],
)
def test_constant_grid_labels(prior, label):
    assert rb.Constant(prior).label == label


@pytest.mark.parametrize("prior", ["IW", "iw-equivalent", "ivg", None, 1])
def test_constant_rejects_unknown_priors_listing_the_choices(prior):
    with pytest.raises(ValueError, match="'auto', 'iw_equivalent', 'iw' or 'ig'"):
        rb.Constant(prior=prior)


def test_components_package_imports_specs_only():
    import rai_bvar.components as components
    from rai_bvar.components import specs

    assert components.SV is specs.SV is rb.SV
    assert components.Linear is specs.Linear and components.Constant is specs.Constant


# =========================================================================== Minnesota


def test_minnesota_defaults_to_auto_form_and_unit_contemp():
    prior = rb.Minnesota()
    assert prior.form == "auto"
    assert prior.contemp == 1.0 and type(prior.contemp) is float


@pytest.mark.parametrize("form", ["auto", "conjugate"])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cross": 0.5}, "cross != 1"),
        ({"contemp": 2.0}, "contemp != 1"),
        ({"contemp": 0.5}, "contemp != 1"),
    ],
)
def test_conjugate_and_auto_forms_admit_unit_cross_and_contemp_only(form, kwargs, match):
    with pytest.raises(ValueError, match=f"{match}.*form='structural' or 'reduced'"):
        rb.Minnesota(form=form, **kwargs)


@pytest.mark.parametrize("form", ["structural", "reduced"])
def test_contemp_shrinkage_is_allowed_outside_conjugate_form(form):
    prior = rb.Minnesota(form=form, contemp=np.float32(0.5), cross=0.2)
    assert prior.contemp == 0.5 and type(prior.contemp) is float
    assert prior.label == f"Minnesota({form}, eb, cross=0.2, contemp=0.5)"


@pytest.mark.parametrize("contemp", [0.0, -1.0, np.inf, np.nan, True])
def test_contemp_must_be_positive_and_finite(contemp):
    with pytest.raises(ValueError):
        rb.Minnesota(form="structural", contemp=contemp)


def test_unresolved_auto_form_is_labelled_by_its_analytic_resolution():
    assert rb.Minnesota().label == "Minnesota(conjugate, eb)"
    assert rb.Minnesota(lambda_=0.3).label == rb.Minnesota(form="conjugate", lambda_=0.3).label
    assert rb.Minnesota(form="structural").label == "Minnesota(structural, eb)"
    assert rb.Minnesota(form="reduced", hierarchy="fixed").label == "Minnesota(reduced, fixed)"
    with pytest.raises(ValueError, match="form must be 'auto', 'conjugate'"):
        rb.Minnesota(form="Auto")


# =========================================================================== SV


def test_sv_defaults_follow_the_design():
    sv = rb.SV()
    assert (sv.rw, sv.outliers, sv.tails) == (False, False, "normal")
    assert (sv.mu_prior, sv.mu_var) == ("data", 10.0)
    assert sv.phi_prior == P.Beta(20.0, 1.5)
    assert sv.sigma2_prior is None and sv.resolved_sigma2_prior() == P.ScaledChi2(1.0)
    assert sv.h0 == "stationary" and sv.rw_h0_var == 4.0
    assert sv.outlier_max == 20 and sv.outlier_prior is None and sv.resolved_outlier_prior() is None
    assert sv.nu_prior == P.DiscreteUniform(3, 40)
    assert (sv.t_scale, sv.mixture, sv.offset) == ("standardized", "ocsn10", "relative")
    assert sv.exact is True and sv.asis is True


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({}, "SV"),
        ({"rw": True}, "SV(rw)"),
        ({"outliers": True}, "SV(outliers)"),
        ({"tails": "t"}, "SV(t)"),
        ({"outliers": True, "tails": "t"}, "SV(outliers,t)"),
        ({"rw": True, "outliers": True}, "SV(rw,outliers)"),
        ({"rw": True, "tails": "t"}, "SV(rw,t)"),
        ({"rw": True, "outliers": True, "tails": "t"}, "SV(rw,outliers,t)"),
    ],
)
def test_sv_names_of_every_variant(kwargs, name):
    sv = rb.SV(**kwargs)
    assert sv.name == name
    assert sv.label == name  # defaults: nothing to add


def test_variant_defaults_of_sigma2_and_outlier_priors():
    assert rb.SV(rw=True).resolved_sigma2_prior() == P.ScaledChi2(0.1)
    assert rb.SV(outliers=True).resolved_outlier_prior() == P.OutlierPrior(4.0, 10.0)
    assert rb.SV(outliers=True, tails="t").resolved_outlier_prior() == P.OutlierPrior(10.0, 10.0)
    assert rb.SV(outliers=True).resolved_outlier_prior().resolve("Q") == P.Beta(2.5, 37.5)
    assert rb.SV(outliers=True, tails="t").resolved_outlier_prior().resolve("M") == P.Beta(1, 119)


@pytest.mark.parametrize(
    ("explicit", "default"),
    [
        ({"sigma2_prior": P.ScaledChi2(1.0)}, {}),
        ({"rw": True, "sigma2_prior": P.ScaledChi2(0.1)}, {"rw": True}),
        ({"outliers": True, "outlier_prior": P.OutlierPrior()}, {"outliers": True}),
        (
            {"outliers": True, "tails": "t", "outlier_prior": P.OutlierPrior(every_years=10)},
            {"outliers": True, "tails": "t"},
        ),
    ],
)
def test_explicit_variant_defaults_are_stored_as_none(explicit, default):
    """One model, one spec: an explicit default equals the implicit one."""
    a, b = rb.SV(**explicit), rb.SV(**default)
    assert a == b and a.to_spec() == b.to_spec() and a.label == b.label


def test_non_default_sigma2_and_outlier_priors_are_kept():
    assert rb.SV(rw=True, sigma2_prior=P.ScaledChi2(1.0)).sigma2_prior == P.ScaledChi2(1.0)
    assert rb.SV(sigma2_prior=P.InvGamma(2.5, 0.025)).resolved_sigma2_prior() == P.InvGamma(
        2.5, 0.025
    )
    beta = P.Beta(2.0, 40.0)
    assert rb.SV(outliers=True, outlier_prior=beta).resolved_outlier_prior() == beta
    # the "every 4 years" default of SVO is not the default of SVO-t
    kept = rb.SV(outliers=True, tails="t", outlier_prior=P.OutlierPrior())
    assert kept.outlier_prior == P.OutlierPrior()


def test_sv_labels_show_non_default_settings():
    assert rb.SV(exact=False).label == "SV(exact=False)"
    assert rb.SV(outliers=True, exact=False, offset=0.0).label == (
        "SV(outliers, offset=0.0, exact=False)"
    )
    assert rb.SV(outliers=True, tails="t", mixture="ksc7").label == "SV(outliers,t, mixture=ksc7)"
    label = rb.SV(mu_prior=P.Normal(0.0, 100.0)).label
    assert label.startswith("SV(mu_prior=") and len(label) == len("SV(mu_prior=xxxxxx)")
    other = rb.SV(mu_prior=P.Normal(0.0, 100.0000001)).label
    assert other != label  # short hashes keep full precision
    assert rb.SV(h0=2.0).label == "SV(h0=2.0)"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rw": 1}, TypeError, "SV.rw must be a bool"),
        ({"exact": "yes"}, TypeError, "SV.exact must be a bool"),
        ({"tails": "student"}, ValueError, "SV.tails"),
        ({"t_scale": "std"}, ValueError, "SV.t_scale"),
        ({"mixture": "ocsn"}, ValueError, "SV.mixture"),
        ({"mu_prior": 0.0}, TypeError, "SV.mu_prior must be a Normal"),
        ({"mu_prior": "stochvol"}, TypeError, "SV.mu_prior"),
        ({"phi_prior": rb.BetaPrior(mode=0.9, sd=0.05)}, TypeError, "SV.phi_prior must be a Beta"),
        ({"sigma2_prior": P.Normal(0, 1)}, TypeError, "ScaledChi2 or InvGamma"),
        ({"nu_prior": (3, 40)}, TypeError, "DiscreteUniform"),
        ({"mu_var": 0.0}, ValueError, "SV.mu_var"),
        ({"rw": True, "rw_h0_var": -1.0}, ValueError, "SV.rw_h0_var"),
        ({"h0": 0.0}, ValueError, "SV.h0 must be 'stationary' or"),
        ({"h0": "fixed"}, TypeError, "SV.h0 must be 'stationary' or"),
        ({"offset": -1e-4}, ValueError, "SV.offset"),
        ({"offset": "absolute"}, TypeError, "SV.offset"),
        ({"outliers": True, "outlier_max": 1}, ValueError, "at least 2"),
        ({"outliers": True, "outlier_max": 20.0}, TypeError, "SV.outlier_max must be an int"),
        (
            {"tails": "t", "nu_prior": P.DiscreteUniform(2, 40)},
            ValueError,
            "standardized Student-t needs ν > 2",
        ),
    ],
)
def test_sv_rejects_invalid_settings(kwargs, error, match):
    with pytest.raises(error, match=match):
        rb.SV(**kwargs)


IGNORED = [
    ({"rw": True, "phi_prior": P.Beta(10.0, 2.0)}, {"rw": True}),
    ({"rw": True, "mu_prior": P.Normal(0.0, 1.0)}, {"rw": True}),
    ({"rw": True, "h0": 2.0}, {"rw": True}),
    ({"rw": True, "mu_var": 4.0}, {"rw": True}),
    ({"rw_h0_var": 1.0}, {}),
    ({"mu_prior": P.Normal(0.0, 1.0), "mu_var": 4.0}, {"mu_prior": P.Normal(0.0, 1.0)}),
    ({"outlier_max": 10}, {}),
    ({"outlier_prior": P.OutlierPrior(5.0, 12.0)}, {}),
    ({"outlier_prior": P.Beta(2.0, 40.0)}, {}),
    ({"tails": "normal", "nu_prior": P.DiscreteUniform(5, 5)}, {}),
    ({"t_scale": "unit"}, {}),
    ({"nu_prior": P.DiscreteUniform(2, 40)}, {}),  # standardized ν > 2 applies to tails="t" only
    (
        {
            "rw": True,
            "outliers": True,
            "phi_prior": P.Beta(5.0, 1.5),
            "nu_prior": P.DiscreteUniform(4, 9),
        },
        {"rw": True, "outliers": True},
    ),
]


@pytest.mark.parametrize(("kwargs", "used"), IGNORED, ids=repr)
def test_sv_ignores_settings_its_variant_does_not_use(kwargs, used):
    """DESIGN §6.3 ("ignored if rw") and §3.8 ("rw ignores h0"): unused settings are reset to
    their defaults, so one model has one spec, label and hash."""
    sv, ref = rb.SV(**kwargs), rb.SV(**used)
    assert sv == ref and hash(sv) == hash(ref)
    assert sv.to_spec() == ref.to_spec() and sv.label == ref.label
    defaults = {f.name: f.default for f in dataclasses.fields(rb.SV)}
    for name in set(kwargs) - set(used) - {"rw", "outliers", "tails"}:
        assert getattr(sv, name) == defaults[name]


def test_sv_keeps_the_settings_its_variant_uses():
    kept = rb.SV(
        outliers=True,
        tails="t",
        mu_prior=P.Normal(0.0, 1.0),
        phi_prior=P.Beta(10.0, 2.0),
        h0=2.0,
        outlier_max=10,
        outlier_prior=P.Beta(2.0, 40.0),
        nu_prior=P.DiscreteUniform(5, 9),
        t_scale="unit",
    )
    assert (kept.mu_prior, kept.phi_prior, kept.h0) == (P.Normal(0.0, 1.0), P.Beta(10.0, 2.0), 2.0)
    assert (kept.outlier_max, kept.outlier_prior) == (10, P.Beta(2.0, 40.0))
    assert (kept.nu_prior, kept.t_scale) == (P.DiscreteUniform(5, 9), "unit")
    assert rb.SV(mu_var=4.0).mu_var == 4.0 and rb.SV(rw=True, rw_h0_var=1.0).rw_h0_var == 1.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rw": True, "phi_prior": rb.BetaPrior(mode=0.9, sd=0.05)}, TypeError),
        ({"rw": True, "h0": 0.0}, ValueError),
        ({"rw_h0_var": -1.0}, ValueError),
        ({"outlier_max": 1}, ValueError),
        ({"outlier_prior": P.Normal(0.0, 1.0)}, TypeError),
        ({"t_scale": "std"}, ValueError),
    ],
    ids=repr,
)
def test_ignored_settings_are_still_validated(kwargs, error):
    """Ignoring is not a licence for invalid values (a typo is caught whatever the variant)."""
    with pytest.raises(error):
        rb.SV(**kwargs)


def test_unit_t_scale_admits_small_nu():
    sv = rb.SV(tails="t", t_scale="unit", nu_prior=P.DiscreteUniform(1, 30))
    assert sv.nu_prior.support[0] == 1


def test_sv_accepts_nested_priors_as_spec_mappings():
    sv = rb.SV(
        outliers=True,
        mu_prior={"class": "Normal", "mean": 0, "var": 100},
        outlier_prior={"class": "Beta", "a": 2, "b": 40},
    )
    assert sv.mu_prior == P.Normal(0.0, 100.0)
    assert sv.outlier_prior == P.Beta(2.0, 40.0)
    with pytest.raises(ValueError, match="'class'"):
        rb.SV(mu_prior={"mean": 0.0, "var": 1.0})


def test_sv_numeric_forms_are_canonical():
    forms = [
        rb.SV(h0=2, mu_var=np.float32(5.0), offset=np.int64(0)),
        rb.SV(h0=np.float64(2.0), mu_var=5, offset=0.0),
        rb.SV(h0=2.0, mu_var=5.0, offset=np.float32(0.0), exact=np.bool_(True)),
    ]
    assert all(f == forms[0] for f in forms)
    assert len({json.dumps(f.to_spec()) for f in forms}) == 1
    assert type(forms[0].h0) is float and type(forms[0].offset) is float


SV_EXAMPLES = {
    "default": rb.SV(),
    "rw": rb.SV(rw=True, sigma2_prior=P.InvGamma(3.0, 0.2)),
    "svo": rb.SV(outliers=True, outlier_max=10, outlier_prior=P.OutlierPrior(5.0, 12.0)),
    "svo-beta": rb.SV(outliers=True, outlier_prior=P.Beta(2.0, 50.0)),
    "svt": rb.SV(tails="t", nu_prior=P.DiscreteUniform(5, 5), t_scale="unit"),
    "svot": rb.SV(outliers=True, tails="t"),
    "stochvol-parity": rb.SV(
        mu_prior=P.Normal(0.0, 100.0),
        phi_prior=P.Beta(5.0, 1.5),
        sigma2_prior=P.ScaledChi2(0.1),
        h0=1.0,
        offset=0.0,
        exact=False,
        mixture="ksc7",
        asis=False,
    ),
}


@pytest.mark.parametrize("sv", SV_EXAMPLES.values(), ids=SV_EXAMPLES.keys())
def test_sv_spec_round_trips_through_json(sv):
    spec = sv.to_spec()
    assert spec["class"] == "SV"
    payload = json.dumps(spec)  # plain JSON
    rebuilt = rb.SV.from_spec(json.loads(payload))
    assert rebuilt == sv
    assert json.dumps(rebuilt.to_spec()) == payload
    assert rebuilt.label == sv.label and rebuilt.name == sv.name


def test_sv_spec_serialises_nested_priors_with_their_class():
    spec = SV_EXAMPLES["stochvol-parity"].to_spec()
    assert spec["mu_prior"] == {"class": "Normal", "mean": 0.0, "var": 100.0}
    assert spec["phi_prior"] == {"class": "Beta", "a": 5.0, "b": 1.5}
    assert spec["sigma2_prior"] == {"class": "ScaledChi2", "B": 0.1}
    assert spec["nu_prior"] == {"class": "DiscreteUniform", "low": 3, "high": 40}
    assert spec["outlier_prior"] is None and spec["offset"] == 0.0
    assert rb.SV().to_spec()["mu_prior"] == "data"


def test_sv_from_spec_defaults_missing_fields_and_rejects_unknown_ones():
    assert rb.SV.from_spec({"class": "SV"}) == rb.SV()
    assert rb.SV.from_spec({"outliers": True}) == rb.SV(outliers=True)
    with pytest.raises(ValueError, match="unknown SV fields"):
        rb.SV.from_spec({"class": "SV", "nu": 5})
    with pytest.raises(ValueError, match="not an SV spec"):
        rb.SV.from_spec({"class": "Constant"})


def test_sv_is_frozen_hashable_and_picklable():
    import pickle

    sv = SV_EXAMPLES["stochvol-parity"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        sv.rw = True
    assert hash(sv) == hash(pickle.loads(pickle.dumps(sv)))
    assert len({rb.SV(), rb.SV(), rb.SV(rw=True)}) == 2


# =========================================================================== settings


def test_mcmc_settings_defaults_follow_the_design():
    s = rb.MCMCSettings()
    assert (s.chains, s.draws, s.burn, s.thin, s.warm_burn, s.seed) == (4, None, 1000, 1, None, 0)
    assert (s.store, s.store_max, s.path_thin, s.t_bucket) == ("params", 2500, 10, 32)
    assert (s.gram, s.const_path, s.rhat_max, s.ess_min) == ("auto", "auto", 1.01, 400.0)
    assert s.resolved_warm_burn == 250
    # draws=None is the composition default (D-35), resolved by BVAR or with_default_draws
    with pytest.raises(ValueError, match="draws=None is the composition's default"):
        _ = s.store_thin
    sv, constant = s.with_default_draws(sv=True), s.with_default_draws(sv=False)
    assert sv == rb.MCMCSettings(draws=10_000) and sv.store_thin == 4
    assert constant == rb.MCMCSettings(draws=2000) and constant.store_thin == 1


def test_mcmc_settings_derived_values():
    assert rb.MCMCSettings(burn=4000).resolved_warm_burn == 1000
    assert rb.MCMCSettings(burn=4000, warm_burn=0).resolved_warm_burn == 0
    assert rb.MCMCSettings(draws=10_000).store_thin == 4  # ⌈10000/2500⌉
    assert rb.MCMCSettings(draws=2501).store_thin == 2


def test_mcmc_settings_store_canonical_types():
    s = rb.MCMCSettings(
        chains=np.int64(2),
        draws=np.int32(100),
        burn=np.uint16(50),
        warm_burn=np.int64(10),
        rhat_max=np.float32(1.5),
        ess_min=np.int64(100),
    )
    assert s == rb.MCMCSettings(
        chains=2, draws=100, burn=50, warm_burn=10, rhat_max=1.5, ess_min=100
    )
    for f in dataclasses.fields(s):
        assert type(getattr(s, f.name)) in (int, float, str, type(None))
    json.dumps(dataclasses.asdict(s))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"chains": 0}, ValueError),
        ({"draws": 0}, ValueError),
        ({"burn": -1}, ValueError),
        ({"thin": 0}, ValueError),
        ({"warm_burn": -1}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 2**63}, ValueError),
        ({"store": "draws"}, ValueError),
        ({"store_max": 0}, ValueError),
        ({"path_thin": 0}, ValueError),
        ({"t_bucket": 0}, ValueError),
        ({"gram": "dense"}, ValueError),
        ({"const_path": "eigen"}, ValueError),
        ({"rhat_max": 1.0}, ValueError),
        ({"ess_min": -1.0}, ValueError),
        ({"rhat_max": np.nan}, ValueError),
        ({"chains": 2.0}, TypeError),
        ({"draws": True}, TypeError),
        ({"ess_min": "400"}, TypeError),
        ({"draws": 2**31, "thin": 2}, ValueError),  # uint32 sweep counter (DESIGN §5.6)
    ],
)
def test_mcmc_settings_reject_invalid_values(kwargs, error):
    with pytest.raises(error):
        rb.MCMCSettings(**kwargs)


@pytest.mark.parametrize(
    ("last", "one_more"),
    [
        ({"burn": 1000, "draws": 2**32 - 1001}, {"burn": 1000, "draws": 2**32 - 1000}),
        (
            {"burn": 1, "warm_burn": 0, "draws": 2**31 - 1, "thin": 2},
            {"burn": 2, "warm_burn": 0, "draws": 2**31 - 1, "thin": 2},
        ),
        (
            {"burn": 2**32 - 2001, "draws": 1000, "thin": 2},
            {"burn": 2**32 - 2000, "draws": 1000, "thin": 2},
        ),
        (  # a warm burn-in longer than burn counts too
            {"burn": 0, "warm_burn": 2**32 - 2, "draws": 1},
            {"burn": 0, "warm_burn": 2**32 - 1, "draws": 1},
        ),
    ],
    ids=["draws", "thin", "burn", "warm_burn"],
)
def test_mcmc_settings_sweep_counter_boundary(last, one_more):
    """The host bound ``max(burn, warm_burn) + draws·thin < 2³²`` (DESIGN §5.6): a run of
    ``2³² − 1`` sweeps is accepted, one more sweep is rejected."""
    s = rb.MCMCSettings(**last)
    assert max(s.burn, s.resolved_warm_burn) + s.draws * s.thin == 2**32 - 1
    with pytest.raises(ValueError, match="uint32 sweep counter"):
        rb.MCMCSettings(**one_more)


def test_runtime_defaults_and_validation(tmp_path):
    r = rb.Runtime()
    assert (r.devices, r.chunk, r.memory_budget_gb, r.storage_dir, r.replay, r.progress) == (
        1,  # DESIGN §5.2 Stage 2 amendment: sharding is opt-in
        250,
        2.0,
        None,
        True,
        False,
    )
    path = rb.Runtime(storage_dir=pathlib.Path(tmp_path), devices=np.int64(2))
    assert path.storage_dir == str(tmp_path) and path.devices == 2
    for kwargs in (
        {"devices": 0},
        {"devices": "all"},
        {"chunk": 0},
        {"memory_budget_gb": 0.0},
        {"storage_dir": 3},
        {"replay": "yes"},
        {"progress": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            rb.Runtime(**kwargs)


def test_fit_plan_validation():
    assert rb.FitPlan(8) == rb.FitPlan(horizon=8, draws=1000, seed=0)
    for kwargs in ({"horizon": 0}, {"horizon": 4, "draws": 0}, {"horizon": 4, "seed": -1}):
        with pytest.raises(ValueError):
            rb.FitPlan(**kwargs)
    with pytest.raises(TypeError):
        rb.FitPlan(horizon=4.0)


@pytest.mark.parametrize("obj", [rb.MCMCSettings(), rb.Runtime(), rb.FitPlan(4)], ids=type)
def test_settings_are_frozen(obj):
    name = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, name, getattr(obj, name))


# =========================================================================== package surface


def test_import_rai_bvar_imports_no_jax():
    code = (
        "import sys\n"
        "import rai_bvar as rb\n"
        "import rai_bvar.settings, rai_bvar.components, rai_bvar.priors.sv\n"
        "rb.SV(outliers=True, tails='t'); rb.MCMCSettings(); rb.Runtime(); rb.FitPlan(4)\n"
        "assert not [m for m in sys.modules if m == 'jax' or m.startswith('jax.')], 'jax imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


LAZY = {
    "set_cpu_devices": "rai_bvar._devices",
    "enable_compilation_cache": "rai_bvar._devices",
    "ordering_sensitivity": "rai_bvar.structural.ordering",  # WP22
}


def _in_build(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_new_names_are_exported():
    for name in ("SV", "MCMCSettings", "Runtime", "FitPlan", "sv_priors"):
        assert name in rb.__all__ and hasattr(rb, name)
    # lazy names (their modules may import JAX) are exported when their module is in the build
    for name, module in LAZY.items():
        assert (name in rb.__all__) is _in_build(module)
        assert (name in dir(rb)) is _in_build(module)
    assert {"set_cpu_devices", "enable_compilation_cache"} <= set(rb.__all__)  # WP1's _devices


@pytest.mark.parametrize(("name", "module"), LAZY.items())
def test_lazy_attributes_resolve_to_their_modules(name, module):
    mod = pytest.importorskip(module)  # provided by WP1 / WP22
    assert getattr(rb, name) is getattr(mod, name)


@pytest.mark.parametrize(("name", "module"), LAZY.items())
def test_lazy_attributes_outside_the_build_are_missing_attributes(name, module):
    if _in_build(module):
        pytest.skip(f"{module} is part of this build")
    assert not hasattr(rb, name)
    with pytest.raises(AttributeError, match=rf"no attribute '{name}' \({module} is not part"):
        getattr(rb, name)


def test_a_lazy_module_outside_the_build_is_a_missing_attribute(monkeypatch):
    monkeypatch.setitem(rb._LAZY, "wp2_probe", "rai_bvar._wp2_no_such_module")
    monkeypatch.delitem(rb.__dict__, "wp2_probe", raising=False)
    assert not hasattr(rb, "wp2_probe")
    assert "wp2_probe" not in dir(rb)


def test_a_broken_lazy_module_raises_its_import_error(monkeypatch, tmp_path):
    """Only the lazy module itself may be missing: a missing dependency of an existing module is
    a real error and propagates."""
    package = tmp_path / "wp2_lazy_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "broken.py").write_text("import wp2_no_such_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rb._LAZY, "wp2_probe", "wp2_lazy_probe.broken")
    try:
        with pytest.raises(ModuleNotFoundError, match="wp2_no_such_dependency"):
            rb.wp2_probe  # noqa: B018
    finally:
        for name in ("wp2_lazy_probe.broken", "wp2_lazy_probe"):
            sys.modules.pop(name, None)


def test_star_import_works_and_imports_no_jax():
    code = (
        "import sys\n"
        "from rai_bvar import *\n"
        "import rai_bvar as rb\n"
        "missing = [n for n in rb.__all__ if n not in globals()]\n"
        "assert not missing, missing\n"
        "assert not [m for m in sys.modules if m == 'jax' or m.startswith('jax.')], 'jax imported'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_unknown_package_attributes_raise_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        rb.not_a_name  # noqa: B018
