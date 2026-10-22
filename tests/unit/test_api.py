"""The ``BVAR`` façade and its components (SPEC §3 R1, §4.4–4.6, §5.6).

Construction and validation of ``BVAR`` / ``Linear`` / ``Constant`` / ``Minnesota``; engine
resolution; ``name`` / ``axes`` / ``spec`` and the spec hash; the ``outliers`` shortcut; and the
hyperparameter inspection API of the analytic fit (``summary``, ``log_ml``, ``log_posterior``,
``posterior`` with partial overrides, seeding).
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import rai_bvar as rb
from rai_bvar.engines import AnalyticFit
from rai_bvar.models import FittedModel, Model, spec_hash

QD_SMALL = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "qd_small_levels.csv"
QD_VARIABLES = ("GDPC1", "GDPCTPI", "FEDFUNDS")
COVID_NAMES = ("covid_s0", "covid_s1", "covid_s2", "covid_rho")
SMALL_MH = rb.MHSettings(draws=150, burn=100, chains=2)


def bvar(lags: int = 1, outliers=None, vol=None, **prior) -> rb.BVAR:
    """``BVAR(Linear(Minnesota(**prior)), lags)``; the default ``Constant`` unless ``vol``."""
    vol = rb.Constant() if vol is None else vol
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(**prior)), vol=vol, lags=lags, outliers=outliers
    )


@pytest.fixture(scope="module")
def qd() -> rb.VARData:
    """Quarterly 100·log GDP, 100·log prices and the policy rate, 1960Q1–2019Q4, tcode 1."""
    df = pd.read_csv(QD_SMALL, index_col="date")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return rb.VARData.from_frame(df, tcodes=1)


@pytest.fixture(scope="module")
def covid_data(simulate_var) -> rb.VARData:
    """Simulated stationary data 1990Q1–2021Q4: the sample reaches the 2020Q1 ``t*``."""
    index = pd.period_range("1990Q1", periods=128, freq="Q")
    frame = pd.DataFrame(simulate_var(128, 7), index=index, columns=["gdp", "prices", "rate"])
    return rb.VARData.from_frame(frame, 1)


@pytest.fixture(scope="module")
def eb_dummies_fit(qd) -> AnalyticFit:
    """Empirical Bayes with SOC and SUR dummies: λ, μ, δ free; ψ fixed at the AR variances."""
    return bvar(lags=4, mean=1.0, soc=1.0, sur=1.0).fit(qd)


# =========================================================================== construction


def test_default_composition_is_the_conjugate_glp_bvar():
    model = rb.BVAR()
    assert model == rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota()),
        vol=rb.Constant(prior="iw"),
        lags=1,
        deterministic="c",
        outliers=None,
    )
    assert model.engine == "analytic"
    prior = model.mean.prior
    assert (prior.lambda_, prior.alpha, prior.psi, prior.cross) == (0.2, 2.0, "ar", 1.0)
    assert (prior.soc, prior.sur, prior.mean) == (None, None, None)
    assert (prior.form, prior.hierarchy, prior.hyper) == ("conjugate", "eb", "auto")
    assert prior.hyperpriors == rb.GLPHyperpriors()
    assert prior.mh == rb.MHSettings()


@pytest.mark.parametrize(
    ("obj", "attr", "value"),
    [
        (rb.BVAR(), "lags", 2),
        (rb.Linear(), "prior", rb.Minnesota(lambda_=0.1)),
        (rb.Constant(), "prior", "ig"),
        (rb.Minnesota(), "lambda_", 0.1),
        (rb.MHSettings(), "seed", 1),
    ],
    ids=["BVAR", "Linear", "Constant", "Minnesota", "MHSettings"],
)
def test_specifications_are_immutable(obj, attr, value):
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, value)


def test_minnesota_normalises_sequences_and_mappings():
    means = {"GDPC1": 0.9}
    prior = rb.Minnesota(psi=[1, 2, 3], hyper=["psi", "lambda"], mean=means)
    assert prior.psi == (1.0, 2.0, 3.0)
    assert all(type(v) is float for v in prior.psi)
    assert prior.hyper == ("psi", "lambda")
    assert prior.mean == means
    means["GDPC1"] = 0.0  # the prior keeps its own copy
    assert prior.mean == {"GDPC1": 0.9}


def test_own_lag_means_scalar_mapping_and_transform_defaults():
    defaults = {"a": 1.0, "b": 0.0, "c": 1.0}
    variables = ("a", "b", "c")
    assert rb.Minnesota().own_lag_means(variables, defaults) == [1.0, 0.0, 1.0]
    assert rb.Minnesota(mean=0.5).own_lag_means(variables, defaults) == [0.5, 0.5, 0.5]
    partial = rb.Minnesota(mean={"b": 0.3})
    assert partial.own_lag_means(variables, defaults) == [1.0, 0.3, 1.0]


# =========================================================================== validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cross": 0.5}, "form='structural' or 'reduced'"),
        ({"cross": 0.0}, "form='structural' or 'reduced'"),
        ({"form": "Conjugate"}, "form must be"),
        ({"form": "bayesian"}, "form must be"),
        ({"hierarchy": "EB"}, "hierarchy must be"),
        ({"hierarchy": "hb"}, "hierarchy must be"),
        ({"psi": "ols"}, "psi must be"),
        ({"psi": (1.0, -1.0, 2.0)}, "psi values must be positive"),
        ({"psi": (1.0, 0.0)}, "psi values must be positive"),
        ({"hyper": ("lambda", "beta")}, "unknown hyperparameters"),
        ({"hyper": ("lambda_",)}, "unknown hyperparameters"),
        ({"hyper": "lambda"}, "hyper must be"),
        ({"hyper": ("soc",)}, "'soc' is a hyperparameter but the soc dummy is off"),
        ({"hyper": ("lambda", "sur"), "soc": 1.0}, "'sur' is a hyperparameter"),
        ({"lambda_": 0.0}, "must be positive"),
        ({"lambda_": -0.2}, "must be positive"),
        ({"alpha": 0.0}, "must be positive"),
        ({"var_const": 0.0}, "must be positive"),
        ({"soc": 0.0}, "soc must be positive"),
        ({"sur": -1.0}, "sur must be positive"),
    ],
)
def test_minnesota_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rb.Minnesota(**kwargs)


@pytest.mark.parametrize("form", ["structural", "reduced"])
def test_cross_variable_shrinkage_is_allowed_outside_conjugate_form(form):
    assert rb.Minnesota(form=form, cross=0.5).cross == 0.5


@pytest.mark.parametrize("prior", ["IW", "t", None])
def test_constant_rejects_unknown_priors(prior):
    with pytest.raises(ValueError, match="'iw' or 'ig'"):
        rb.Constant(prior=prior)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lags": 0}, "lags"),
        ({"lags": -1}, "lags"),
        ({"deterministic": "x"}, "deterministic"),
        ({"deterministic": "const"}, "deterministic"),
        ({"outliers": "covid"}, "lenza-primiceri"),
        ({"outliers": "Lenza-Primiceri"}, "lenza-primiceri"),
    ],
)
def test_bvar_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rb.BVAR(**kwargs)


@pytest.mark.parametrize("outliers", [True, 1.0, ("lenza-primiceri",)], ids=repr)
def test_bvar_rejects_non_string_outlier_settings_at_construction(outliers):
    """SPEC §4.5: invalid compositions fail at construction, not deep inside ``fit``."""
    with pytest.raises((TypeError, ValueError)):
        rb.BVAR(outliers=outliers)


def test_fit_rejects_psi_of_the_wrong_length(qd):
    with pytest.raises(ValueError, match="psi needs 3 values, got 2"):
        bvar(psi=(1.0, 2.0)).fit(qd)


# =========================================================================== engine resolution


def test_conjugate_minnesota_with_inverse_wishart_runs_on_the_analytic_engine(qd):
    model = bvar(lags=2, hierarchy="fixed")
    assert model.engine == "analytic"
    fit = model.fit(qd)
    assert isinstance(fit, AnalyticFit)
    assert fit.engine == "analytic"


@pytest.mark.parametrize(
    "prior",
    [
        rb.Minnesota(form="structural"),
        rb.Minnesota(form="reduced"),
        rb.Minnesota(form="structural", cross=0.5),
        rb.Minnesota(form="reduced", cross=0.2, hierarchy="mh"),
    ],
    ids=["structural", "reduced", "structural-cross", "reduced-cross-mh"],
)
@pytest.mark.parametrize("vol", ["iw", "ig"])
def test_non_conjugate_forms_need_the_mcmc_engine(prior, vol):
    with pytest.raises(NotImplementedError, match="MCMC"):
        rb.BVAR(mean=rb.Linear(prior=prior), vol=rb.Constant(prior=vol))


def test_conjugate_form_requires_the_inverse_wishart_covariance():
    with pytest.raises(ValueError, match=r"Constant\(prior='iw'\)"):
        rb.BVAR(vol=rb.Constant(prior="ig"))


@dataclass(frozen=True)
class _StochasticVolatility:
    """A stand-in for a volatility block that does not exist yet."""

    prior: str = "sv"


@dataclass(frozen=True)
class _MeanWithPrior:
    """A stand-in non-linear mean block that carries a Minnesota prior."""

    prior: rb.Minnesota = field(default_factory=rb.Minnesota)


@dataclass(frozen=True)
class _MeanWithoutPrior:
    """A stand-in non-linear mean block (e.g. a tree ensemble) with no ``prior`` attribute."""

    trees: int = 200


def test_volatility_blocks_other_than_constant_are_not_implemented():
    with pytest.raises(NotImplementedError, match="Linear × Constant"):
        rb.BVAR(vol=_StochasticVolatility())


@pytest.mark.parametrize(
    "mean", [_MeanWithPrior(), _MeanWithoutPrior()], ids=lambda m: type(m).__name__
)
def test_mean_blocks_other_than_linear_are_not_implemented(mean):
    with pytest.raises(NotImplementedError, match="Linear × Constant"):
        rb.BVAR(mean=mean)


def test_bvar_and_its_fit_satisfy_the_model_protocols(qd):
    model = bvar(hierarchy="fixed")
    assert isinstance(model, Model)
    assert isinstance(model.fit(qd), FittedModel)


# =========================================================================== name, axes, spec


def test_name_and_axes_of_a_full_composition():
    model = bvar(lags=4, outliers="lenza-primiceri", soc=1.0, sur=1.0, hierarchy="mh")
    assert model.name == "BVAR(Minnesota(conjugate, mh, SOC+SUR), p=4, COVID)"
    assert model.axes == {
        "mean": "Linear",
        "prior": "Minnesota(conjugate, mh, SOC+SUR)",
        "vol": "Constant",
        "lags": "4",
        "deterministic": "c",
        "outliers": "lenza-primiceri",
    }


NAME_VARIANTS = [
    bvar(),
    bvar(hierarchy="fixed"),
    bvar(hierarchy="mh"),
    bvar(soc=1.0),
    bvar(sur=1.0),
    bvar(soc=1.0, sur=1.0),
    bvar(soc=1.0, sur=1.0, hierarchy="fixed"),
    bvar(lags=4),
    bvar(outliers="lenza-primiceri"),
    bvar(lags=4, soc=1.0, outliers="lenza-primiceri"),
]


def test_names_distinguish_hierarchy_dummies_lags_and_outliers():
    names = [m.name for m in NAME_VARIANTS]
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("model", NAME_VARIANTS, ids=lambda m: m.name)
def test_axes_are_string_labels_along_the_ablation_axes(model):
    axes = model.axes
    assert set(axes) == {"mean", "prior", "vol", "lags", "deterministic", "outliers"}
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in axes.items())
    assert axes["prior"] == model.mean.prior.label
    assert axes["lags"] == str(model.lags)
    assert axes["outliers"] == ("none" if model.outliers is None else "lenza-primiceri")


def test_prior_axis_separates_hierarchies_and_dummies():
    labels = {m.axes["prior"] for m in NAME_VARIANTS}
    # fixed / eb / mh without dummies, eb with SOC, SUR, SOC+SUR, fixed with SOC+SUR
    assert len(labels) == 7


SPEC_VARIANTS = {
    "default": rb.BVAR(),
    "lambda": bvar(lambda_=0.1),
    "alpha": bvar(alpha=1.5),
    "psi": bvar(psi=(1.0, 2.0, 3.0)),
    "psi-other": bvar(psi=(1.0, 2.0, 4.0)),
    "soc": bvar(soc=1.0),
    "soc-other": bvar(soc=2.0),
    "sur": bvar(sur=1.0),
    "mean-scalar": bvar(mean=1.0),
    "mean-zero": bvar(mean=0.0),
    "mean-mapping": bvar(mean={"GDPC1": 1.0, "FEDFUNDS": 0.9}),
    "var-const": bvar(var_const=1e6),
    "fixed": bvar(hierarchy="fixed"),
    "mh": bvar(hierarchy="mh"),
    "hyper-psi": bvar(hyper=("lambda", "psi")),
    "hyper-alpha": bvar(hyper=("lambda", "alpha")),
    "hyperpriors": bvar(
        hyperpriors=rb.GLPHyperpriors(
            lambda_=rb.GammaPrior(mode=0.3, sd=0.4, lower=1e-4, upper=5.0)
        )
    ),
    "mh-seed": bvar(hierarchy="mh", mh=rb.MHSettings(seed=1)),
    "mh-draws": bvar(hierarchy="mh", mh=rb.MHSettings(draws=1000)),
    "lags": rb.BVAR(lags=4),
    "deterministic-ct": rb.BVAR(deterministic="ct"),
    "deterministic-n": rb.BVAR(deterministic="n"),
    "outliers": rb.BVAR(outliers="lenza-primiceri"),
    "outliers-start": rb.BVAR(outliers=rb.LenzaPrimiceri(start="2020Q2")),
    "outliers-fixed": rb.BVAR(outliers=rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8))),
    "outliers-rho-prior": rb.BVAR(
        outliers=rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.5, sd=0.2))
    ),
}


@pytest.mark.parametrize("model", SPEC_VARIANTS.values(), ids=SPEC_VARIANTS.keys())
def test_spec_is_json_serialisable(model):
    spec = model.spec()
    assert spec["class"] == "BVAR"
    assert {"mean", "vol", "lags", "deterministic", "outliers"} <= set(spec)
    payload = json.dumps(spec)  # no ``default=``: every value must be plain JSON
    assert json.loads(payload)["mean"]["prior"]["lambda_"] == model.mean.prior.lambda_


def test_spec_hash_distinguishes_every_prior_and_model_setting():
    hashes = {name: spec_hash(model) for name, model in SPEC_VARIANTS.items()}
    assert all(isinstance(h, str) and len(h) == 64 for h in hashes.values())
    assert len(set(hashes.values())) == len(hashes)
    specs = [json.dumps(m.spec(), sort_keys=True) for m in SPEC_VARIANTS.values()]
    assert len(set(specs)) == len(specs)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (rb.BVAR(outliers="lenza-primiceri"), rb.BVAR(outliers=rb.LenzaPrimiceri())),
        (bvar(psi=[1, 2, 3]), bvar(psi=(1.0, 2.0, 3.0))),
        (bvar(hyper=["lambda", "psi"]), bvar(hyper=("lambda", "psi"))),
        (bvar(mean={"a": 1.0, "b": 0.0}), bvar(mean={"b": 0.0, "a": 1.0})),
        (rb.BVAR(), rb.BVAR(mean=rb.Linear(), vol=rb.Constant())),
    ],
    ids=["outliers-string", "psi-list", "hyper-list", "mean-order", "defaults"],
)
def test_equal_models_have_equal_spec_hashes(a, b):
    assert a == b
    assert a.spec() == b.spec()
    assert spec_hash(a) == spec_hash(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (bvar(lambda_=1), bvar(lambda_=1.0)),
        (bvar(soc=2, sur=1), bvar(soc=2.0, sur=1.0)),
        (rb.BVAR(lags=np.int64(4)), rb.BVAR(lags=4)),
    ],
    ids=["lambda-int", "dummies-int", "lags-numpy-int"],
)
def test_spec_hash_is_consistent_with_model_equality_across_numeric_types(a, b):
    """``a == b`` (Python compares ``1 == 1.0``), so the cache key must agree as well."""
    assert a == b
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_survives_pickling():
    model = bvar(
        lags=4,
        outliers="lenza-primiceri",
        mean={"GDPC1": 1.0},
        soc=1.0,
        hierarchy="mh",
        mh=SMALL_MH,
    )
    clone = pickle.loads(pickle.dumps(model))
    assert clone == model
    assert spec_hash(clone) == spec_hash(model)


def test_spec_hash_is_stable_across_processes():
    """The hash is a cache key: it must not depend on ``PYTHONHASHSEED`` or object identity."""
    code = (
        "import rai_bvar as rb\n"
        "from rai_bvar.models import spec_hash\n"
        "m = rb.Minnesota(mean={'GDPC1': 1.0, 'FEDFUNDS': 0.9}, psi=(1.0, 2.0, 3.0), soc=1.0)\n"
        "print(spec_hash(rb.BVAR(mean=rb.Linear(prior=m), lags=4, outliers='lenza-primiceri')))\n"
    )
    expected = spec_hash(
        bvar(
            lags=4,
            outliers="lenza-primiceri",
            mean={"FEDFUNDS": 0.9, "GDPC1": 1.0},
            psi=(1.0, 2.0, 3.0),
            soc=1.0,
        )
    )
    env = {**os.environ, "PYTHONHASHSEED": "12345"}  # differs from this process's random seed
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == expected


def test_forecast_attributes_carry_name_engine_and_spec_hash(qd):
    model = bvar(lags=2, hierarchy="fixed", soc=1.0)
    fc = model.fit(qd).forecast(4, draws=20, seed=0)
    attrs = fc.dataset.attrs
    assert attrs["model"] == model.name
    assert attrs["engine"] == "analytic"
    assert attrs["spec_hash"] == spec_hash(model)
    assert json.loads(attrs["spec"]) == json.loads(json.dumps(model.spec()))


# =========================================================================== outliers


def test_outliers_string_is_the_default_lenza_primiceri_scaling():
    model = rb.BVAR(outliers="lenza-primiceri")
    assert isinstance(model.outliers, rb.LenzaPrimiceri)
    assert model.outliers == rb.LenzaPrimiceri()
    assert model == rb.BVAR(outliers=rb.LenzaPrimiceri())
    assert model.name.endswith(", COVID)")
    assert model.axes["outliers"] == "lenza-primiceri"


def test_lenza_primiceri_parameters_join_the_free_hyperparameters(covid_data):
    fit = rb.BVAR(outliers="lenza-primiceri").fit(covid_data)
    summary = fit.summary()
    assert summary.index.tolist() == ["lambda", *COVID_NAMES]
    assert fit.hyper["covid"].shape == (4,)
    np.testing.assert_array_equal(summary.loc[list(COVID_NAMES), "mode"], fit.hyper["covid"])


def test_lenza_primiceri_with_fixed_hierarchy_needs_fixed_values(covid_data):
    # rejected at construction, not only once a backtest sample reaches t*
    with pytest.raises(ValueError, match=r"fixed=\(s0, s1, s2, rho\)"):
        bvar(hierarchy="fixed", outliers="lenza-primiceri")
    fixed = rb.LenzaPrimiceri(fixed=(3.0, 2.0, 1.5, 0.7))
    fit = bvar(hierarchy="fixed", outliers=fixed).fit(covid_data)
    np.testing.assert_array_equal(fit.hyper["covid"], [3.0, 2.0, 1.5, 0.7])
    assert fit.summary().empty


def test_lenza_primiceri_is_inert_before_the_covid_period(covid_data):
    fit = rb.BVAR(outliers="lenza-primiceri").fit(covid_data.upto("2019Q4"))
    assert fit.hyper["covid"] is None
    assert fit.summary().index.tolist() == ["lambda"]


# =========================================================================== free hyperparameters


@pytest.mark.parametrize(
    ("prior", "free"),
    [
        (rb.Minnesota(), ("lambda",)),
        (rb.Minnesota(soc=1.0), ("lambda", "soc")),
        (rb.Minnesota(sur=1.0), ("lambda", "sur")),
        (rb.Minnesota(soc=1.0, sur=1.0), ("lambda", "soc", "sur")),
        (rb.Minnesota(soc=1.0, sur=1.0, hierarchy="mh"), ("lambda", "soc", "sur")),
        (rb.Minnesota(alpha=1.5, psi=(1.0, 2.0, 3.0)), ("lambda",)),
        (rb.Minnesota(hyper=("psi", "lambda")), ("lambda", "psi")),
        (rb.Minnesota(hyper=("alpha",)), ("alpha",)),
        (rb.Minnesota(hyper=(), soc=1.0), ()),
        (rb.Minnesota(hyper=("lambda",), soc=1.0, sur=1.0), ("lambda",)),
        (
            rb.Minnesota(hyper=("psi", "sur", "alpha", "soc", "lambda"), soc=1.0, sur=1.0),
            ("lambda", "alpha", "soc", "sur", "psi"),
        ),
        (rb.Minnesota(hierarchy="fixed"), ()),
        (rb.Minnesota(hierarchy="fixed", soc=1.0, sur=1.0), ()),
        (rb.Minnesota(hierarchy="fixed", hyper=("lambda", "psi")), ()),
    ],
    ids=[
        "auto",
        "auto-soc",
        "auto-sur",
        "auto-soc-sur",
        "auto-mh",
        "auto-ignores-alpha-psi",
        "explicit-order",
        "explicit-alpha",
        "explicit-empty",
        "explicit-subset",
        "explicit-all",
        "fixed",
        "fixed-dummies",
        "fixed-explicit",
    ],
)
def test_minnesota_free_hyperparameters(prior, free):
    assert prior.free == free


# =========================================================================== summary


def test_summary_of_empirical_bayes_lists_the_mode(eb_dummies_fit):
    fit = eb_dummies_fit
    summary = fit.summary()
    assert summary.columns.tolist() == ["mode"]
    assert summary.index.tolist() == ["lambda", "soc", "sur"]
    for name in ("lambda", "soc", "sur"):
        assert summary.loc[name, "mode"] == fit.hyper[name]
    assert fit.mh is None
    assert fit.converged
    assert fit.optimization["success"]


def test_summary_names_one_psi_row_per_variable(qd):
    fit = bvar(lags=2, mean=1.0, hyper=("lambda", "psi")).fit(qd)
    summary = fit.summary()
    assert summary.index.tolist() == ["lambda", *(f"psi_{v}" for v in QD_VARIABLES)]
    np.testing.assert_array_equal(summary["mode"].to_numpy()[1:], fit.psi)


def test_summary_of_fixed_hierarchy_is_empty(qd):
    fit = bvar(lags=2, hierarchy="fixed", lambda_=0.3, soc=2.0).fit(qd)
    summary = fit.summary()
    assert summary.columns.tolist() == ["mode"]
    assert summary.empty
    assert fit.optimization is None
    assert fit.mh is None
    assert (fit.hyper["lambda"], fit.hyper["soc"], fit.hyper["sur"]) == (0.3, 2.0, None)


def test_summary_of_metropolis_hastings_adds_quantiles_and_diagnostics(qd):
    fit = bvar(lags=2, mean=1.0, soc=1.0, hierarchy="mh", mh=SMALL_MH).fit(qd, seed=1)
    summary = fit.summary()
    assert summary.columns.tolist() == ["mode", "q05", "q50", "q95", "rhat", "ess"]
    assert summary.index.tolist() == ["lambda", "soc"]
    assert np.all(summary["q05"] <= summary["q50"])
    assert np.all(summary["q50"] <= summary["q95"])
    assert np.all(np.isfinite(summary["rhat"])) and np.all(summary["ess"] > 0)
    assert fit.mh.chains.shape == (SMALL_MH.chains, SMALL_MH.draws, 2)
    # the MH chain starts from the same posterior mode that empirical Bayes reports
    eb = bvar(lags=2, mean=1.0, soc=1.0).fit(qd)
    np.testing.assert_array_equal(summary["mode"], eb.summary()["mode"])


# =========================================================================== seeding


def _mh_model(**mh) -> rb.BVAR:
    return bvar(lags=2, mean=1.0, hierarchy="mh", mh=replace(SMALL_MH, **mh))


def test_fit_seed_makes_metropolis_hastings_reproducible(qd):
    model = _mh_model()
    a, b, c = model.fit(qd, seed=3), model.fit(qd, seed=3), model.fit(qd, seed=4)
    np.testing.assert_array_equal(a.mh.chains, b.mh.chains)
    pd.testing.assert_frame_equal(a.summary(), b.summary())
    assert not np.array_equal(a.mh.chains, c.mh.chains)
    fa, fb = a.forecast(4, draws=50, seed=11), b.forecast(4, draws=50, seed=11)
    np.testing.assert_array_equal(fa.paths.values, fb.paths.values)
    np.testing.assert_array_equal(fa.cond_chol.values, fb.cond_chol.values)


def test_fit_seed_overrides_the_settings_seed(qd):
    default = _mh_model()  # MHSettings.seed = 0
    np.testing.assert_array_equal(default.fit(qd).mh.chains, default.fit(qd, seed=0).mh.chains)
    np.testing.assert_array_equal(
        _mh_model(seed=3).fit(qd).mh.chains, default.fit(qd, seed=3).mh.chains
    )


def test_equal_specs_give_identical_forecasts(qd):
    """Model protocol: equal specs must give identical forecasts (fit afresh, same seed)."""
    a = bvar(lags=2, soc=1.0).fit(qd).forecast(6, draws=40, seed=5)
    b = bvar(lags=2, soc=1.0).fit(qd).forecast(6, draws=40, seed=5)
    np.testing.assert_array_equal(a.paths.values, b.paths.values)


# =========================================================================== hyperparameter API


def _gamma_logpdf(x: float, mode: float, sd: float) -> float:
    """Gamma density with the given mode and s.d.: solve ``(k − 1)²/k = (mode/sd)²`` for ``k``."""
    k = optimize.brentq(lambda k: (k - 1.0) ** 2 / k - (mode / sd) ** 2, 1.0 + 1e-12, 1e8)
    return float(stats.gamma(a=k, scale=mode / (k - 1.0)).logpdf(x))


def _assert_same_posterior(a, b) -> None:
    np.testing.assert_allclose(a.mean, b.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.row_upper, b.row_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.scale_chol, b.scale_chol, rtol=1e-12, atol=1e-12)
    assert a.df == b.df


def test_no_override_means_the_fitted_hyperparameters(eb_dummies_fit):
    fit = eb_dummies_fit
    assert fit.log_ml() == fit.log_ml({}) == fit.log_ml(fit.hyper)
    assert fit.log_posterior() == fit.log_posterior({}) == fit.log_posterior(fit.hyper)
    _assert_same_posterior(fit.posterior(), fit.posterior({}))
    _assert_same_posterior(fit.posterior(), fit.posterior({"lambda": fit.hyper["lambda"]}))
    # at the EB mode the log posterior is the optimiser's objective value
    np.testing.assert_allclose(fit.log_posterior(), fit.optimization["log_posterior"], rtol=1e-12)


@pytest.mark.parametrize(
    ("override", "fixed_kwargs"),
    [
        ({"soc": 2.0}, {"soc": 2.0}),
        ({"lambda": 0.3}, {"lambda_": 0.3}),
        ({"lambda": 0.3, "sur": 0.5}, {"lambda_": 0.3, "sur": 0.5}),
        ({"alpha": 1.5}, {"alpha": 1.5}),
        ({"psi": [2.0, 0.05, 3.0]}, {"psi": (2.0, 0.05, 3.0)}),
    ],
    ids=["soc", "lambda", "lambda-sur", "alpha-not-free", "psi-not-free"],
)
def test_partial_overrides_keep_the_other_fitted_hyperparameters(
    qd, eb_dummies_fit, override, fixed_kwargs
):
    fit = eb_dummies_fit
    before = {k: fit.hyper[k] for k in ("lambda", "alpha", "soc", "sur")}
    psi_before = fit.psi.copy()
    mode = {"lambda_": fit.hyper["lambda"], "soc": fit.hyper["soc"], "sur": fit.hyper["sur"]}
    reference = bvar(lags=4, mean=1.0, hierarchy="fixed", **{**mode, **fixed_kwargs}).fit(qd)
    np.testing.assert_allclose(fit.log_ml(override), reference.log_ml(), rtol=1e-12)
    _assert_same_posterior(fit.posterior(override), reference.posterior())
    # the fit itself is untouched
    assert {k: fit.hyper[k] for k in before} == before
    np.testing.assert_array_equal(fit.psi, psi_before)


def test_log_posterior_adds_the_hyperpriors_of_the_free_hyperparameters_only(eb_dummies_fit):
    fit = eb_dummies_fit
    override = {"lambda": 0.3, "alpha": 1.5, "psi": 2.0 * fit.psi}  # α, ψ are not free here
    prior = (
        _gamma_logpdf(0.3, 0.2, 0.4)
        + _gamma_logpdf(fit.hyper["soc"], 1.0, 1.0)
        + _gamma_logpdf(fit.hyper["sur"], 1.0, 1.0)
    )
    np.testing.assert_allclose(
        fit.log_posterior(override) - fit.log_ml(override), prior, rtol=1e-10
    )


def test_log_posterior_equals_log_ml_with_fixed_hierarchy(qd):
    fit = bvar(lags=2, hierarchy="fixed", soc=1.0).fit(qd)
    for hyper in (None, {"lambda": 0.5}, {"soc": 3.0, "alpha": 1.2}):
        assert fit.log_posterior(hyper) == fit.log_ml(hyper)


@pytest.mark.parametrize("method", ["log_ml", "log_posterior", "posterior"])
@pytest.mark.parametrize("key", ["lam", "lambda_", "mu", "delta", "cross", "psi_GDPC1"])
def test_unknown_hyperparameter_raises_key_error(eb_dummies_fit, method, key):
    with pytest.raises(KeyError, match=f"unknown hyperparameter '{key}'"):
        getattr(eb_dummies_fit, method)({key: 0.3})
    with pytest.raises(KeyError, match=f"'{key}'"):  # also when mixed with valid ones
        getattr(eb_dummies_fit, method)({"lambda": 0.3, key: 0.3})
