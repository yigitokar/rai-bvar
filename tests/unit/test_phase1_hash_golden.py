"""Golden Phase 1 identities: Phase 2 must not change any Phase 1 spec, hash, name or axes.

The spec hash is the backtest cache key, so a Phase 1 model must hash exactly as before Phase 2
(DESIGN §11.5, E13; §13 rule 5). The values below were recorded with the Phase 1 code (commit
``cbfcc5b``) for every model of the Phase 1 contract registry plus one model per Phase 1
setting. Phase 2 adds ``Minnesota(form="auto", contemp=1.0)``, ``Constant(prior="auto")`` and
``BVAR(mcmc=None)``: the defaults resolve to the Phase 1 values and the new fields stay out of the
spec at their defaults.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.models import spec_hash

FAST_MH = rb.MHSettings(draws=100, burn=100, chains=2, seed=7)


def _bvar(lags=1, deterministic="c", outliers=None, **prior):
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(**prior)),
        lags=lags,
        deterministic=deterministic,
        outliers=outliers,
    )


GOLDEN_MODELS = {
    # the Phase 1 contract registry (tests/contract/test_forecast_contract.py::BVAR_MODELS)
    "contract-fixed-p1": lambda: _bvar(1, hierarchy="fixed"),
    "contract-fixed-p2-n": lambda: _bvar(2, "n", hierarchy="fixed"),
    "contract-fixed-p2-soc-sur": lambda: _bvar(2, hierarchy="fixed", soc=1.0, sur=1.0),
    "contract-fixed-p1-n-soc": lambda: _bvar(1, "n", hierarchy="fixed", soc=0.5),
    "contract-eb-p2": lambda: _bvar(2, hierarchy="eb"),
    "contract-eb-p1-soc-sur": lambda: _bvar(1, hierarchy="eb", soc=1.0, sur=1.0),
    "contract-eb-p2-n-sur": lambda: _bvar(2, "n", hierarchy="eb", sur=1.0),
    "contract-mh-p1": lambda: _bvar(1, hierarchy="mh", mh=FAST_MH),
    "contract-eb-lp": lambda: _bvar(1, outliers=rb.LenzaPrimiceri(), hierarchy="eb"),
    # defaults and single-setting variants
    "default": lambda: rb.BVAR(),
    "default-explicit": lambda: rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="conjugate")), vol=rb.Constant(prior="iw")
    ),
    "lags-4": lambda: rb.BVAR(lags=4),
    "det-ct": lambda: rb.BVAR(deterministic="ct"),
    "lambda": lambda: _bvar(lambda_=0.1),
    "alpha": lambda: _bvar(alpha=1.5),
    "psi": lambda: _bvar(psi=(1.0, 2.0, 3.0)),
    "mean-mapping": lambda: _bvar(mean={"rate": 0.9, "gdp": np.float32(0.5), "prices": 0}),
    "var-const": lambda: _bvar(var_const=1e5),
    "soc-scope-all": lambda: _bvar(soc=2, soc_scope="all"),
    "hyper-psi": lambda: _bvar(hyper=("lambda", "psi")),
    "hyperpriors": lambda: _bvar(
        soc=1,
        sur=1,
        hyper=("lambda", "alpha", "soc", "sur", "psi"),
        hyperpriors=rb.GLPHyperpriors(
            lambda_=rb.GammaPrior(mode=0.3, sd=0.5, lower=1e-3, upper=4),
            alpha=rb.GammaPrior(mode=1.5, sd=0.3, lower=1, upper=3),
            soc=rb.GammaPrior(mode=2, sd=1, lower=1e-3, upper=40),
            sur=rb.GammaPrior(mode=0.5, sd=1, lower=1e-3, upper=30),
            psi=rb.InvGammaPrior(scale=0.01, shape=0.01, lower=1e-3, upper=100),
        ),
    ),
    "mh-settings": lambda: _bvar(hierarchy="mh", mh=rb.MHSettings(draws=1000, seed=1)),
    "lp-start-fixed": lambda: rb.BVAR(
        outliers=rb.LenzaPrimiceri(start="2020Q2", fixed=(2.0, 2.0, 2.0, 0.8))
    ),
    "lp-priors": lambda: rb.BVAR(
        outliers=rb.LenzaPrimiceri(
            start="2020Q2",
            scale_prior=rb.ParetoPrior(scale=1, shape=2, upper=500),
            rho_prior=rb.BetaPrior(mode=0.5, sd=0.1, lower=0.01, upper=0.99),
        )
    ),
    "everything": lambda: _bvar(
        lags=2,
        deterministic="ct",
        hierarchy="mh",
        mh={"draws": 200, "chains": 2, "seed": 11},
        psi=(1.0, 2.0, 3.0),
        mean={"gdp": 1.0},
        soc=1.5,
        sur=0.5,
        soc_scope="all",
        hyper=("lambda", "soc"),
        outliers=rb.LenzaPrimiceri(start="2020Q1", fixed=(4, 2, 1.5, 0.6)),
    ),
}


# (spec_hash, name, axes) recorded with the Phase 1 code
GOLDEN = {
    "contract-fixed-p1": (
        "a2d7cbda13a980d5dbbedc648f9a3111768f319871e71b6aa99926356671a73f",
        "BVAR(Minnesota(conjugate, fixed), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, fixed)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "contract-fixed-p2-n": (
        "cee0b97cce6b41562b8b67a1400b058322ebf47142430133e3fa17df259fbc4e",
        "BVAR(Minnesota(conjugate, fixed), p=2, det=n)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, fixed)",
            "vol": "Constant",
            "lags": "2",
            "deterministic": "n",
            "outliers": "none",
        },
    ),
    "contract-fixed-p2-soc-sur": (
        "153493cd0ea4651e0df666f785e602da94865eb41250d4fc9c5ccb3da6240370",
        "BVAR(Minnesota(conjugate, fixed, SOC+SUR), p=2)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, fixed, SOC+SUR)",
            "vol": "Constant",
            "lags": "2",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "contract-fixed-p1-n-soc": (
        "103575289d3aa37d8cc0e8b451f2ded77e4955590e106aadfd6b934b81dddafc",
        "BVAR(Minnesota(conjugate, fixed, SOC, soc=0.5), p=1, det=n)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, fixed, SOC, soc=0.5)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "n",
            "outliers": "none",
        },
    ),
    "contract-eb-p2": (
        "6bde7d822990bba9449cfe5dbd7d0918cd1d785daeaae95710e17e1c87e6aaa1",
        "BVAR(Minnesota(conjugate, eb), p=2)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "2",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "contract-eb-p1-soc-sur": (
        "0177f4101a9426795bc39d3495daf23cfa5473b031d7d4094084aaea2f508900",
        "BVAR(Minnesota(conjugate, eb, SOC+SUR), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, SOC+SUR)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "contract-eb-p2-n-sur": (
        "df568be523529c3d0a24512e4e0109b441d236b5745b7633e8da782ede570565",
        "BVAR(Minnesota(conjugate, eb, SUR), p=2, det=n)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, SUR)",
            "vol": "Constant",
            "lags": "2",
            "deterministic": "n",
            "outliers": "none",
        },
    ),
    "contract-mh-p1": (
        "6bcbbebb40a78d6133d83560efa621d712759d56b1afd4dd02ecc6802e508a00",
        "BVAR(Minnesota(conjugate, mh, mh=4cca22), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, mh, mh=4cca22)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "contract-eb-lp": (
        "534b99a1238defb30a7829b7d324f71523137205d2ad6f5750bcea5d57c78938",
        "BVAR(Minnesota(conjugate, eb), p=1, COVID)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "lenza-primiceri",
        },
    ),
    "default": (
        "15499749c5e8ed942ede8b54b080f363eb0851fbbd16952b37a8e460e20c12bc",
        "BVAR(Minnesota(conjugate, eb), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "default-explicit": (
        "15499749c5e8ed942ede8b54b080f363eb0851fbbd16952b37a8e460e20c12bc",
        "BVAR(Minnesota(conjugate, eb), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "lags-4": (
        "bdf1658760444b234bebfc73b7ff7569c468e3621aa8102671cb9f4767259346",
        "BVAR(Minnesota(conjugate, eb), p=4)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "4",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "det-ct": (
        "acade1f9907985d8c23269316b16ee9802f25dd2ade896e76636905d7f272e92",
        "BVAR(Minnesota(conjugate, eb), p=1, det=ct)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "ct",
            "outliers": "none",
        },
    ),
    "lambda": (
        "2020a3351fb705f3fae79d2ab85869a5b368df001e7cbf71a8726556845563c5",
        "BVAR(Minnesota(conjugate, eb, lambda=0.1), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, lambda=0.1)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "alpha": (
        "9444a5ed095074152abc6c7da0c2ce74a7484734a4d730b7590d7a163fed988b",
        "BVAR(Minnesota(conjugate, eb, alpha=1.5), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, alpha=1.5)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "psi": (
        "0dacbfa148b4cd541ed631d2d7c626b3d1d21ced9daac029f2c1f5062a233662",
        "BVAR(Minnesota(conjugate, eb, psi=(1.0,2.0,3.0)), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, psi=(1.0,2.0,3.0))",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "mean-mapping": (
        "0db1f2738a5914d533ebc3a3019e225d9da26ef75a3dc92d7235af66ad4cf9da",
        "BVAR(Minnesota(conjugate, eb, mean={gdp:0.5,prices:0.0,rate:0.9}), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, mean={gdp:0.5,prices:0.0,rate:0.9})",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "var-const": (
        "b8edd885e4a4ec89da027a5d792af26b66b98dbaa0c27f42f58be973863d27ea",
        "BVAR(Minnesota(conjugate, eb, var_const=100000.0), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, var_const=100000.0)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "soc-scope-all": (
        "33d62c248369df34e78c741dd5343ff19cc6ba41eaee65af31f73bc6007998d8",
        "BVAR(Minnesota(conjugate, eb, SOC, soc_scope=all, soc=2.0), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, SOC, soc_scope=all, soc=2.0)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "hyper-psi": (
        "70c48a844ba7baa9317918fbc0b72d9fad5fad3825cbe2935bac3f4e38be234e",
        "BVAR(Minnesota(conjugate, eb, hyper=(lambda,psi)), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, hyper=(lambda,psi))",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "hyperpriors": (
        "250eeb9f1bc2d4ca30368a459b503112246f4dfc38ad54aa5c0f9cde0dca842d",
        "BVAR(Minnesota(conjugate, eb, SOC+SUR, hyper=(lambda,alpha,soc,sur,psi), hyperpriors=03ec7d), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb, SOC+SUR, hyper=(lambda,alpha,soc,sur,psi), hyperpriors=03ec7d)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "mh-settings": (
        "c1a5d7b016ee7f3e2860974dcc4e7cf70935acfe1feb1cfdc4537d2e4c1bd8eb",
        "BVAR(Minnesota(conjugate, mh, mh=6c8770), p=1)",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, mh, mh=6c8770)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "none",
        },
    ),
    "lp-start-fixed": (
        "5b3e1ffaf59fdd552708c76cf67271f605e97d0f3f62e2849189e8690cbd6ab9",
        "BVAR(Minnesota(conjugate, eb), p=1, COVID (start=2020Q2,fixed=(2.0, 2.0, 2.0, 0.8)))",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "lenza-primiceri(start=2020Q2,fixed=(2.0, 2.0, 2.0, 0.8))",
        },
    ),
    "lp-priors": (
        "8c4546295fcb49e0392dc302d12581e15507485a7c0cc714d173368db5d8592c",
        "BVAR(Minnesota(conjugate, eb), p=1, COVID (start=2020Q2,scale_prior=ParetoPrior(scale=1.0, shape=2.0, upper=500.0),rho_prior=BetaPrior(mode=0.5, sd=0.1, lower=0.01, upper=0.99)))",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, eb)",
            "vol": "Constant",
            "lags": "1",
            "deterministic": "c",
            "outliers": "lenza-primiceri(start=2020Q2,scale_prior=ParetoPrior(scale=1.0, shape=2.0, upper=500.0),rho_prior=BetaPrior(mode=0.5, sd=0.1, lower=0.01, upper=0.99))",
        },
    ),
    "everything": (
        "b5d5d478ef66b23c19e2ef2b7e89940b8c7fb4157097e8509fe916916bbba000",
        "BVAR(Minnesota(conjugate, mh, SOC+SUR, psi=(1.0,2.0,3.0), mean={gdp:1.0}, hyper=(lambda,soc), soc_scope=all, soc=1.5, sur=0.5, mh=859127), p=2, det=ct, COVID (start=2020Q1,fixed=(4.0, 2.0, 1.5, 0.6)))",
        {
            "mean": "Linear",
            "prior": "Minnesota(conjugate, mh, SOC+SUR, psi=(1.0,2.0,3.0), mean={gdp:1.0}, hyper=(lambda,soc), soc_scope=all, soc=1.5, sur=0.5, mh=859127)",
            "vol": "Constant",
            "lags": "2",
            "deterministic": "ct",
            "outliers": "lenza-primiceri(start=2020Q1,fixed=(4.0, 2.0, 1.5, 0.6))",
        },
    ),
}


# ``BVAR().spec()`` recorded with the Phase 1 code (JSON with sorted keys)
DEFAULT_SPEC = '{"class": "BVAR", "deterministic": "c", "lags": 1, "mean": {"prior": {"alpha": 2.0, "cross": 1.0, "form": "conjugate", "hierarchy": "eb", "hyper": "auto", "hyperpriors": {"alpha": {"lower": 1.0, "mode": 2.0, "sd": 0.25, "upper": 3.0}, "lambda_": {"lower": 0.0001, "mode": 0.2, "sd": 0.4, "upper": 5.0}, "psi": {"lower": null, "scale": 0.0004, "shape": 0.0004, "upper": null}, "soc": {"lower": 0.0001, "mode": 1.0, "sd": 1.0, "upper": 50.0}, "sur": {"lower": 0.0001, "mode": 1.0, "sd": 1.0, "upper": 50.0}}, "lambda_": 0.2, "mean": null, "mh": {"burn": 1000, "chains": 4, "draws": 2500, "ess_min": 400.0, "rhat_max": 1.01, "seed": 0, "target_acceptance": 0.3, "thin": 1}, "psi": "ar", "soc": null, "soc_scope": "persistent", "sur": null, "var_const": 10000000.0}}, "outliers": null, "vol": {"prior": "iw"}}'


@pytest.mark.parametrize("key", GOLDEN_MODELS)
def test_phase1_models_keep_their_spec_hash_name_and_axes(key):
    model = GOLDEN_MODELS[key]()
    expected_hash, expected_name, expected_axes = GOLDEN[key]
    assert spec_hash(model) == expected_hash
    assert model.name == expected_name
    assert model.axes == expected_axes
    assert model.engine == "analytic"
    assert model.mcmc is None


@pytest.mark.parametrize("key", GOLDEN_MODELS)
def test_phase1_specs_carry_no_phase2_fields(key):
    spec = GOLDEN_MODELS[key]().spec()
    assert "mcmc" not in spec
    assert "contemp" not in spec["mean"]["prior"]
    assert spec["mean"]["prior"]["form"] == "conjugate"
    assert spec["vol"] == {"prior": "iw"}  # no type tag on Phase 1 components
    json.dumps(spec)  # plain JSON


@pytest.mark.parametrize("key", GOLDEN_MODELS)
def test_phase1_specs_round_trip_to_the_same_hash(key):
    model = GOLDEN_MODELS[key]()
    rebuilt = rb.BVAR.from_spec(json.loads(json.dumps(model.spec())))
    assert rebuilt == model
    assert spec_hash(rebuilt) == GOLDEN[key][0]


def test_default_spec_is_byte_identical_to_phase1():
    assert json.dumps(rb.BVAR().spec(), sort_keys=True) == DEFAULT_SPEC


def test_auto_defaults_resolve_to_the_phase1_composition():
    explicit = rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="conjugate")), vol=rb.Constant(prior="iw")
    )
    for model in (rb.BVAR(), rb.BVAR(mean=rb.Linear(), vol=rb.Constant()), explicit):
        assert model == explicit
        assert model.mean.prior.form == "conjugate"
        assert model.vol == rb.Constant(prior="iw")
    assert rb.Minnesota().form == "auto"
    assert rb.Constant().prior == "auto"


def test_phase1_prior_labels_are_unchanged():
    assert rb.Minnesota().label == "Minnesota(conjugate, eb)"
    assert rb.Minnesota(form="conjugate").label == "Minnesota(conjugate, eb)"
    assert rb.Linear().label == "Linear[Minnesota(conjugate, eb)]"
    assert rb.Constant().name == rb.Constant("iw").name == rb.Constant("ig").name == "Constant"


@pytest.mark.parametrize(
    "settings",
    [rb.MCMCSettings(), rb.MCMCSettings(chains=2, draws=10), {"chains": 3}],
    ids=["default", "custom", "mapping"],
)
def test_mcmc_settings_on_an_analytic_model_leave_its_spec_unchanged(settings):
    model = rb.BVAR(mcmc=settings)
    assert model.mcmc is None
    assert model == rb.BVAR()
    assert spec_hash(model) == GOLDEN["default"][0]


def test_contemp_at_its_default_is_omitted_and_numeric_forms_are_canonical():
    forms = [1, 1.0, np.float64(1.0), np.int64(1), np.float32(1.0)]
    hashes = {spec_hash(_bvar(contemp=c)) for c in forms}
    assert hashes == {GOLDEN["default"][0]}
