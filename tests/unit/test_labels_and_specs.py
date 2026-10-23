"""Model labels, specs and seeds: what makes two models (or two forecasts) the same or different.

* ``Minnesota.label`` is canonical: prior-mean mappings in any key order, ``int`` / ``float`` /
  NumPy scalars and hyperpriors given with integers all give one label (and one spec hash),
  while values that differ in their last represented digit (``0.3`` vs ``0.3000001``) do not.
* ``BVAR.name`` / ``BVAR.axes`` separate priors, deterministic terms and Lenza-Primiceri
  settings, so ``compare()`` accepts lists of such models (names must be unique) and a change
  of one setting moves exactly one ablation axis.
* ``grid`` labels entries with the model's own axes when they cover the grid keys (so grid
  entries pair with hand-built models in ``verdict(axis=...)``) and falls back to the grid
  labels for models with coarse axes (``OLSVAR``, ``RandomWalk``).
* The verdict without an axis allows identical specs and refuses different specs that share
  every declared axis (SPEC R2).
* ``BVAR.from_spec`` inverts ``spec()`` through JSON for every composition.
* Validation at construction: MH settings, hyperprior types, non-finite values, the
  Lenza-Primiceri start, Beta / inverse-gamma / Pareto priors, ``fit(seed=)`` and ``None``
  overrides of free dummy hyperparameters.
* Seed labels in forecast and IRF attrs (SPEC §7): integers, ``SeedSequence``, fresh and
  advanced ``Generator`` objects.
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import rai_bvar as rb
from rai_bvar.evaluation import Entry
from rai_bvar.models import spec_hash
from rai_bvar.models.base import seed_label

VARIABLES = ("gdp", "prices", "rate")
TOL_MH = rb.MHSettings(draws=100, burn=100, chains=2, rhat_max=1.5, ess_min=0.0)
LP_FIXED = rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8))


def bvar(lags: int = 1, outliers=None, deterministic: str = "c", **prior) -> rb.BVAR:
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(**prior)),
        lags=lags,
        deterministic=deterministic,
        outliers=outliers,
    )


def fixed(**prior) -> rb.Linear:
    """A ``Linear`` mean block with a fixed-hierarchy Minnesota prior (cheap to fit)."""
    return rb.Linear(prior=rb.Minnesota(hierarchy="fixed", **prior))


@pytest.fixture(scope="module")
def data(sim_data_stationary) -> rb.VARData:
    """Simulated stationary quarterly data, 1960Q1–2019Q4, tcode 1 (stops before COVID)."""
    return sim_data_stationary


@pytest.fixture(scope="module")
def fixed_fit(data):
    return bvar(hierarchy="fixed").fit(data)


@pytest.fixture(scope="module")
def mh_fit(data):
    return bvar(hierarchy="mh", mh=TOL_MH).fit(data, seed=4)


@pytest.fixture(scope="module")
def covid_data(simulate_var) -> rb.VARData:
    """Simulated stationary data 1990Q1–2021Q4: the sample reaches the 2020Q1 ``t*``."""
    index = pd.period_range("1990Q1", periods=128, freq="Q")
    frame = pd.DataFrame(simulate_var(128, 7), index=index, columns=list(VARIABLES))
    return rb.VARData.from_frame(frame, 1)


def _same_identity(models) -> None:
    """Every model has one label, name, axes, spec hash and plain-JSON spec."""
    models = list(models)
    assert len({m.mean.prior.label for m in models}) == 1, [m.mean.prior.label for m in models]
    assert len({m.name for m in models}) == 1
    assert len({json.dumps(m.axes, sort_keys=True) for m in models}) == 1
    assert len({spec_hash(m) for m in models}) == 1
    for m in models:
        json.dumps(m.spec())  # no ``default=``: canonical values are plain JSON
        assert m == models[0]


# =========================================================================== canonical labels


@pytest.mark.parametrize(
    "order", list(itertools.permutations(VARIABLES)), ids=lambda o: "-".join(o)
)
def test_label_ignores_the_key_order_of_prior_mean_mappings(order):
    values = {"gdp": 0.9, "prices": 0.75, "rate": 1.0}
    prior = rb.Minnesota(mean={k: values[k] for k in order})
    assert list(prior.mean) == sorted(VARIABLES)
    reference = bvar(mean=values)
    _same_identity([reference, bvar(mean={k: values[k] for k in order})])
    assert prior.label == "Minnesota(conjugate, eb, mean={gdp:0.9,prices:0.75,rate:1.0})"


NUMERIC_FORMS = {
    "lambda_": [1, 1.0, np.float64(1.0), np.int64(1), np.float32(1.0), np.int32(1)],
    "alpha": [3, 3.0, np.float64(3.0), np.int64(3), np.float32(3.0)],
    "soc": [2, 2.0, np.float64(2.0), np.int64(2), np.float32(2.0)],
    "sur": [1, 1.0, np.float64(1.0), np.int64(1), np.float32(1.0)],
    "var_const": [100000, 1e5, np.float64(1e5), np.int64(100000)],
    "mean": [1, 1.0, np.float64(1.0), np.int64(1), np.float32(1.0)],
    "mean-mapping": [
        {"gdp": 1, "rate": 0.5},
        {"rate": np.float32(0.5), "gdp": np.int64(1)},
        {np.str_("gdp"): np.float64(1.0), "rate": 0.5},
    ],
    "psi": [
        (1, 2, 4),
        (1.0, 2.0, 4.0),
        [np.int64(1), np.float32(2.0), np.float64(4.0)],
        np.array([1, 2, 4]),
        np.array([1.0, 2.0, 4.0], dtype=np.float32),
    ],
}


@pytest.mark.parametrize("setting", NUMERIC_FORMS)
def test_label_and_spec_are_canonical_across_numeric_types(setting):
    key = "mean" if setting == "mean-mapping" else setting
    _same_identity(bvar(**{key: value}) for value in NUMERIC_FORMS[setting])


def test_hyperpriors_given_with_ints_or_numpy_scalars_have_the_float_label():
    as_ints = rb.GLPHyperpriors(
        lambda_=rb.GammaPrior(mode=1, sd=1, lower=np.float32(0.5), upper=5),
        alpha=rb.GammaPrior(mode=np.int64(2), sd=np.float64(0.5), lower=1, upper=3),
        soc=rb.GammaPrior(mode=2, sd=1, lower=1, upper=50),
        sur=rb.GammaPrior(mode=np.float64(2.0), sd=1, lower=1, upper=np.int32(40)),
        psi=rb.InvGammaPrior(scale=1, shape=np.int64(2), lower=1, upper=100),
    )
    as_floats = rb.GLPHyperpriors(
        lambda_=rb.GammaPrior(mode=1.0, sd=1.0, lower=0.5, upper=5.0),
        alpha=rb.GammaPrior(mode=2.0, sd=0.5, lower=1.0, upper=3.0),
        soc=rb.GammaPrior(mode=2.0, sd=1.0, lower=1.0, upper=50.0),
        sur=rb.GammaPrior(mode=2.0, sd=1.0, lower=1.0, upper=40.0),
        psi=rb.InvGammaPrior(scale=1.0, shape=2.0, lower=1.0, upper=100.0),
    )
    assert as_ints == as_floats
    assert repr(as_ints) == repr(as_floats)
    for prior in (as_ints.lambda_, as_ints.alpha, as_ints.soc, as_ints.sur, as_ints.psi):
        assert all(type(getattr(prior, f)) is float for f in prior.__dataclass_fields__)
    _same_identity([bvar(hyperpriors=as_ints), bvar(hyperpriors=as_floats)])
    assert "hyperpriors=" in bvar(hyperpriors=as_ints).name


def test_mh_settings_from_a_mapping_or_numpy_scalars_have_one_label():
    explicit = rb.MHSettings(
        draws=200, burn=50, chains=3, seed=7, target_acceptance=0.25, rhat_max=1.05, ess_min=100.0
    )
    forms = [
        explicit,
        {
            "draws": 200,
            "burn": 50,
            "chains": 3,
            "seed": 7,
            "target_acceptance": 0.25,
            "rhat_max": 1.05,
            "ess_min": 100,
        },
        rb.MHSettings(
            draws=np.int64(200),
            burn=np.int32(50),
            chains=3,
            seed=np.uint8(7),
            target_acceptance=np.float64(0.25),
            rhat_max=np.float64(1.05),
            ess_min=np.int64(100),
        ),
    ]
    models = [bvar(hierarchy="mh", mh=mh) for mh in forms]
    assert all(m.mean.prior.mh == explicit for m in models)
    _same_identity(models)
    assert "mh=" in models[0].mean.prior.label
    # under EB the MH settings are unused and stay out of the label
    assert "mh=" not in bvar(mh=explicit).mean.prior.label


REPR_PAIRS = {
    "lambda": ({"lambda_": 0.3}, {"lambda_": 0.3000001}),
    "lambda-sum": ({"lambda_": 0.3}, {"lambda_": 0.1 + 0.2}),
    "alpha": ({"alpha": 2.5}, {"alpha": 2.5000000001}),
    "psi": ({"psi": (1.0, 2.0, 0.3)}, {"psi": (1.0, 2.0, 0.3000001)}),
    "mean": ({"mean": 0.9}, {"mean": 0.9000001}),
    "mean-mapping": ({"mean": {"gdp": 0.9}}, {"mean": {"gdp": 0.9000001}}),
    "soc": ({"soc": 2.0}, {"soc": 2.0000001}),
    "soc-one": ({"soc": 1.0}, {"soc": 1.0000001}),
    "sur": ({"sur": 0.5}, {"sur": 0.5000001}),
    "var-const": ({"var_const": 1e7 + 1.0}, {"var_const": 1e7 + 2.0}),
    "hyperpriors": (
        {"hyperpriors": rb.GLPHyperpriors(lambda_=rb.GammaPrior(0.3, 0.4, 1e-4, 5.0))},
        {"hyperpriors": rb.GLPHyperpriors(lambda_=rb.GammaPrior(0.3000001, 0.4, 1e-4, 5.0))},
    ),
    "mh-target": (
        {"hierarchy": "mh", "mh": rb.MHSettings(target_acceptance=0.3)},
        {"hierarchy": "mh", "mh": rb.MHSettings(target_acceptance=0.3000001)},
    ),
    "mh-seed": (
        {"hierarchy": "mh", "mh": rb.MHSettings(seed=1)},
        {"hierarchy": "mh", "mh": rb.MHSettings(seed=2)},
    ),
}


@pytest.mark.parametrize(("a", "b"), REPR_PAIRS.values(), ids=REPR_PAIRS.keys())
def test_labels_keep_full_repr_precision(a, b):
    ma, mb = bvar(**a), bvar(**b)
    assert ma != mb
    assert spec_hash(ma) != spec_hash(mb)
    assert ma.mean.prior.label != mb.mean.prior.label
    assert ma.mean.label != mb.mean.label
    assert ma.name != mb.name
    assert ma.axes["prior"] != mb.axes["prior"]


def test_repr_precision_label_shows_the_exact_value():
    assert rb.Minnesota(lambda_=0.3).label == "Minnesota(conjugate, eb, lambda=0.3)"
    assert rb.Minnesota(lambda_=0.3000001).label == "Minnesota(conjugate, eb, lambda=0.3000001)"
    assert rb.Minnesota(lambda_=0.1 + 0.2).label.endswith("lambda=0.30000000000000004)")


LP_REPR_PAIRS = {
    "fixed-rho": (
        rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8)),
        rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8000001)),
    ),
    "scale-upper": (
        rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(upper=500.0)),
        rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(upper=500.0000001)),
    ),
    "rho-sd": (
        rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.8, sd=0.2)),
        rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.8, sd=0.2000001)),
    ),
}


@pytest.mark.parametrize(("a", "b"), LP_REPR_PAIRS.values(), ids=LP_REPR_PAIRS.keys())
def test_outlier_labels_keep_full_repr_precision(a, b):
    ma, mb = rb.BVAR(outliers=a), rb.BVAR(outliers=b)
    assert ma.name != mb.name
    assert ma.axes["outliers"] != mb.axes["outliers"]
    assert spec_hash(ma) != spec_hash(mb)


@pytest.mark.parametrize(
    "forms",
    [
        [
            rb.LenzaPrimiceri(fixed=(2, 2, 2, 0.8)),
            rb.LenzaPrimiceri(fixed=[2.0, 2.0, 2.0, 0.8]),
            rb.LenzaPrimiceri(fixed=np.array([2, 2, 2, 0.8])),
            rb.LenzaPrimiceri(fixed=(np.int64(2), np.float32(2.0), 2.0, np.float64(0.8))),
        ],
        [
            rb.LenzaPrimiceri(
                scale_prior=rb.ParetoPrior(scale=1, shape=2, upper=500), fixed=(2, 2, 2, 0.5)
            ),
            rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(1.0, 2.0, 500.0), fixed=(2, 2, 2, 0.5)),
            rb.LenzaPrimiceri(
                scale_prior=rb.ParetoPrior(np.int64(1), np.float64(2.0), np.float32(500.0)),
                fixed=(2, 2, 2, 0.5),
            ),
        ],
        [
            rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.5, sd=0.25), fixed=(2, 2, 2, 0.5)),
            rb.LenzaPrimiceri(
                rho_prior=rb.BetaPrior(mode=np.float64(0.5), sd=np.float32(0.25)),
                fixed=(2, 2, 2, 0.5),
            ),
        ],
    ],
    ids=["fixed", "scale-prior", "rho-prior"],
)
def test_outlier_names_are_canonical_across_numeric_types(forms):
    _same_identity(rb.BVAR(mean=fixed(), outliers=lp) for lp in forms)
    assert "np." not in rb.BVAR(mean=fixed(), outliers=forms[-1]).name


# =========================================================================== names and axes

BASE = rb.BVAR()
PRIOR_VARIANTS = {
    "lambda": bvar(lambda_=0.1),
    "lambda-close": bvar(lambda_=0.1000001),
    "alpha": bvar(alpha=1.5),
    "psi": bvar(psi=(1.0, 2.0, 3.0)),
    "psi-other": bvar(psi=(1.0, 2.0, 3.5)),
    "mean": bvar(mean=0.9),
    "mean-mapping": bvar(mean={"gdp": 0.9}),
    "mean-mapping-two": bvar(mean={"gdp": 0.9, "rate": 0.8}),
    "soc": bvar(soc=1.0),
    "soc-2": bvar(soc=2.0),
    "soc-all": bvar(soc=1.0, soc_scope="all"),
    "sur": bvar(sur=1.0),
    "var-const": bvar(var_const=1e5),
    "fixed": bvar(hierarchy="fixed"),
    "mh": bvar(hierarchy="mh"),
    "mh-seed": bvar(hierarchy="mh", mh=rb.MHSettings(seed=1)),
    "mh-draws": bvar(hierarchy="mh", mh=rb.MHSettings(draws=500)),
    "hyper": bvar(hyper=("lambda", "psi")),
    "hyper-alpha": bvar(hyper=("lambda", "alpha")),
    "hyperpriors": bvar(
        hyperpriors=rb.GLPHyperpriors(lambda_=rb.GammaPrior(mode=0.3, sd=0.4, lower=1e-4, upper=5))
    ),
    "hyperpriors-psi": bvar(
        hyperpriors=rb.GLPHyperpriors(psi=rb.InvGammaPrior(scale=0.01, shape=0.01))
    ),
}
DETERMINISTIC_VARIANTS = {
    "det-ct": rb.BVAR(deterministic="ct"),
    "det-n": rb.BVAR(deterministic="n"),
}
OUTLIER_VARIANTS = {
    "lp": rb.BVAR(outliers="lenza-primiceri"),
    "lp-start": rb.BVAR(outliers=rb.LenzaPrimiceri(start="2020Q2")),
    "lp-start-monthly": rb.BVAR(outliers=rb.LenzaPrimiceri(start="2020-04")),
    "lp-fixed": rb.BVAR(outliers=rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8))),
    "lp-fixed-other": rb.BVAR(outliers=rb.LenzaPrimiceri(fixed=(3.0, 2.0, 2.0, 0.8))),
    "lp-scale-prior": rb.BVAR(outliers=rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(upper=100.0))),
    "lp-scale-shape": rb.BVAR(outliers=rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(shape=2.0))),
    "lp-rho-prior": rb.BVAR(outliers=rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.5, sd=0.2))),
    "lp-rho-bounds": rb.BVAR(
        outliers=rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(mode=0.8, sd=0.2, lower=0.1, upper=0.95))
    ),
    "lp-start-fixed": rb.BVAR(
        outliers=rb.LenzaPrimiceri(start="2020Q2", fixed=(2.0, 2.0, 2.0, 0.8))
    ),
}
LAG_VARIANTS = {"lags-2": rb.BVAR(lags=2), "lags-4": rb.BVAR(lags=4)}
AXIS_OF = {
    **dict.fromkeys(PRIOR_VARIANTS, "prior"),
    **dict.fromkeys(DETERMINISTIC_VARIANTS, "deterministic"),
    **dict.fromkeys(OUTLIER_VARIANTS, "outliers"),
    **dict.fromkeys(LAG_VARIANTS, "lags"),
}
ALL_VARIANTS = {
    "base": BASE,
    **PRIOR_VARIANTS,
    **DETERMINISTIC_VARIANTS,
    **OUTLIER_VARIANTS,
    **LAG_VARIANTS,
}


def test_names_axes_and_spec_hashes_are_distinct_across_variants():
    names = [m.name for m in ALL_VARIANTS.values()]
    assert len(set(names)) == len(names), names
    axes = [json.dumps(m.axes, sort_keys=True) for m in ALL_VARIANTS.values()]
    assert len(set(axes)) == len(axes)
    hashes = [spec_hash(m) for m in ALL_VARIANTS.values()]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("key", AXIS_OF)
def test_one_setting_moves_exactly_one_axis(key):
    model, axis = ALL_VARIANTS[key], AXIS_OF[key]
    differing = sorted(k for k in BASE.axes if BASE.axes[k] != model.axes[k])
    assert differing == [axis]
    assert set(model.axes) == set(BASE.axes)


def test_name_spells_out_deterministic_terms_and_outlier_settings():
    assert rb.BVAR(deterministic="ct").name == "BVAR(Minnesota(conjugate, eb), p=1, det=ct)"
    assert rb.BVAR(deterministic="n").name == "BVAR(Minnesota(conjugate, eb), p=1, det=n)"
    assert rb.BVAR(deterministic="c").name == "BVAR(Minnesota(conjugate, eb), p=1)"
    assert rb.BVAR(outliers="lenza-primiceri").name == "BVAR(Minnesota(conjugate, eb), p=1, COVID)"
    start = OUTLIER_VARIANTS["lp-start"]
    assert start.name == "BVAR(Minnesota(conjugate, eb), p=1, COVID (start=2020Q2))"
    assert start.axes["outliers"] == "lenza-primiceri(start=2020Q2)"
    lp_fixed = OUTLIER_VARIANTS["lp-fixed"]
    assert "fixed=(2.0, 2.0, 2.0, 0.8)" in lp_fixed.name
    assert "fixed=(2.0, 2.0, 2.0, 0.8)" in lp_fixed.axes["outliers"]
    assert "upper=100.0" in OUTLIER_VARIANTS["lp-scale-prior"].axes["outliers"]
    assert "mode=0.5" in OUTLIER_VARIANTS["lp-rho-prior"].axes["outliers"]


def test_compare_accepts_lists_of_models_that_differ_in_label_details(data):
    models = [
        rb.BVAR(mean=fixed(lambda_=0.3)),
        rb.BVAR(mean=fixed(lambda_=0.3000001)),
        rb.BVAR(mean=fixed(), deterministic="ct"),
        rb.BVAR(mean=fixed(), deterministic="n"),
        rb.BVAR(mean=fixed(mean={"gdp": 0.9, "prices": 0.8})),
        rb.BVAR(mean=fixed(mean={"prices": 0.8, "gdp": 0.95})),
        rb.BVAR(mean=fixed(), outliers=LP_FIXED),
        rb.BVAR(mean=fixed(), outliers=rb.LenzaPrimiceri(start="2020Q2", fixed=LP_FIXED.fixed)),
        rb.BVAR(
            mean=fixed(),
            outliers=rb.LenzaPrimiceri(fixed=LP_FIXED.fixed, rho_prior=rb.BetaPrior(0.5, 0.2)),
        ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing about these settings should warn
        res = rb.compare(
            models,
            data,
            origins=rb.Origins("2018Q1", "2018Q4"),
            window=rb.Rolling(80),
            draws=50,
        )
    assert list(res.results) == [m.name for m in models]
    assert res.baseline == models[0].name
    for m in models:
        result = res.results[m.name]
        assert result.axes == m.axes
        assert result.spec == m.spec()
        assert np.isfinite(result.scores["lpd"].values).all()


def test_compare_still_rejects_duplicate_names(data):
    with pytest.raises(ValueError, match="unique"):
        rb.compare(
            [rb.BVAR(mean=fixed(lambda_=1)), rb.BVAR(mean=fixed(lambda_=1.0))],
            data,
            origins=rb.Origins("2018Q4"),
        )


# =========================================================================== grid and verdicts


def test_grid_over_bvar_components_uses_the_model_axes():
    means = [fixed(lambda_=0.1), fixed(lambda_=0.1000001)]
    entries = rb.grid(rb.BVAR, mean=means, lags=[1, 2])
    assert len(entries) == 4
    for entry in entries:
        assert entry.axes == entry.model.axes
        assert set(entry.axes) == {"mean", "prior", "vol", "lags", "deterministic", "outliers"}
    assert entries[0].name == f"{means[0].label} × 1"
    assert entries[3].name == f"{means[1].label} × 2"
    assert len({e.name for e in entries}) == 4
    assert entries[0].axes["prior"] != entries[2].axes["prior"]


@pytest.mark.parametrize(
    "axes",
    [
        {"deterministic": ["c", "ct", "n"]},
        {"outliers": [None, "lenza-primiceri", LP_FIXED]},
        {"lags": [1, 4], "deterministic": ["c", "n"]},
    ],
    ids=["deterministic", "outliers", "lags-deterministic"],
)
def test_grid_over_other_bvar_axes_uses_the_model_axes(axes):
    entries = rb.grid(rb.BVAR, **axes)
    assert len(entries) == int(np.prod([len(v) for v in axes.values()]))
    for entry in entries:
        assert entry.axes == entry.model.axes
    assert len({json.dumps(e.axes, sort_keys=True) for e in entries}) == len(entries)


@pytest.mark.parametrize(
    ("factory", "axes", "expected"),
    [
        (
            rb.OLSVAR,
            {"lags": [1, 2], "uncertainty": ["flat", "none"]},
            [
                {"lags": "1", "uncertainty": "flat"},
                {"lags": "1", "uncertainty": "none"},
                {"lags": "2", "uncertainty": "flat"},
                {"lags": "2", "uncertainty": "none"},
            ],
        ),
        (rb.OLSVAR, {"lags": [3]}, [{"lags": "3"}]),
        (rb.RandomWalk, {"drift": [False, True]}, [{"drift": "False"}, {"drift": "True"}]),
    ],
    ids=["olsvar", "olsvar-one", "random-walk"],
)
def test_grid_falls_back_to_labels_when_model_axes_are_coarse(factory, axes, expected):
    entries = rb.grid(factory, **axes)
    assert [dict(e.axes) for e in entries] == expected
    assert all(set(e.model.axes) == {"model"} for e in entries)


@pytest.fixture(scope="module")
def grid_compare(data):
    """Plain BVAR(p=1) baseline, a grid over (λ = 0.1) × (p = 1, 2) and a plain BVAR(p=2)."""
    grid = rb.grid(rb.BVAR, mean=[fixed(lambda_=0.1)], lags=[1, 2])
    plain = [rb.BVAR(mean=fixed(), lags=1), rb.BVAR(mean=fixed(), lags=2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = rb.compare(
            [plain[0], *grid, plain[1]],
            data,
            origins=rb.Origins("2014Q1", "2018Q4"),
            window=rb.Rolling(80),
            draws=200,
        )
    return res, grid, plain


def test_grid_entries_pair_with_plain_bvars_along_the_prior_axis(grid_compare):
    res, grid, plain = grid_compare
    verdict = res.verdict(axis="prior").set_index("model")
    assert set(verdict.index) == {grid[0].name, grid[1].name}
    assert verdict.loc[grid[0].name, "reference"] == plain[0].name
    assert verdict.loc[grid[1].name, "reference"] == plain[1].name
    assert (verdict["axis"] == "prior").all()
    assert verdict["verdict"].isin(["better", "worse", "inconclusive"]).all()
    assert np.isfinite(verdict["dlpd"]).all()


def test_grid_entries_pair_with_each_other_along_the_lags_axis(grid_compare):
    res, grid, plain = grid_compare
    verdict = res.verdict(axis="lags").set_index("model")
    assert set(verdict.index) == {grid[1].name, plain[1].name}
    assert verdict.loc[grid[1].name, "reference"] == grid[0].name
    assert verdict.loc[plain[1].name, "reference"] == plain[0].name
    assert verdict["verdict"].isin(["better", "worse", "inconclusive"]).all()


def test_verdict_against_the_baseline_counts_the_differing_axes(grid_compare):
    res, grid, plain = grid_compare
    verdict = res.verdict().set_index("model")
    assert verdict.loc[grid[0].name, "axis"] == "prior"
    assert verdict.loc[grid[0].name, "verdict"] != "not comparable"
    assert verdict.loc[plain[1].name, "axis"] == "lags"
    assert verdict.loc[plain[1].name, "verdict"] != "not comparable"
    assert verdict.loc[grid[1].name, "axis"] == "lags,prior"
    assert verdict.loc[grid[1].name, "verdict"] == "not comparable"
    table = res.table()
    assert {"axis:prior", "axis:lags", "axis:outliers"} <= set(table.columns)


def test_verdict_without_axis_allows_identical_specs(data):
    model = rb.BVAR(mean=fixed())
    res = rb.compare(
        {"A": model, "B": rb.BVAR(mean=fixed())},
        data,
        origins=rb.Origins("2014Q1", "2018Q4"),
        window=rb.Rolling(80),
        draws=100,
    )
    row = res.verdict().iloc[0]
    assert (row["model"], row["reference"], row["axis"]) == ("B", "A", "(none)")
    assert row["verdict"] == "inconclusive"
    assert row["dlpd"] == 0.0


@pytest.mark.parametrize(
    ("a", "b", "axes"),
    [
        (rb.OLSVAR(1), rb.OLSVAR(2), {}),
        (rb.BVAR(mean=fixed()), rb.BVAR(mean=fixed(lambda_=0.1)), {"model": "BVAR"}),
        (rb.BVAR(mean=fixed()), rb.BVAR(mean=fixed(), deterministic="ct"), {"model": "BVAR"}),
    ],
    ids=["olsvar-no-axes", "bvar-prior", "bvar-deterministic"],
)
def test_verdict_without_axis_refuses_different_specs_with_identical_axes(data, a, b, axes):
    res = rb.compare(
        [Entry("A", a, axes), Entry("B", b, dict(axes))],
        data,
        origins=rb.Origins("2014Q1", "2018Q4"),
        window=rb.Rolling(80),
        draws=100,
    )
    row = res.verdict().iloc[0]
    assert row["axis"] == "(none)"
    assert row["verdict"] == "not comparable"
    assert "no declared axis" in row["reasons"]


# =========================================================================== from_spec round trip

ROUND_TRIP = {
    "default": rb.BVAR(),
    "lp-default": rb.BVAR(outliers="lenza-primiceri"),
    "lp-start": rb.BVAR(outliers=rb.LenzaPrimiceri(start="2020-04")),
    "lp-fixed-ints": bvar(hierarchy="fixed", outliers=rb.LenzaPrimiceri(fixed=(3, 2, 1.5, 0.7))),
    "lp-priors": rb.BVAR(
        outliers=rb.LenzaPrimiceri(
            start="2020Q2",
            scale_prior=rb.ParetoPrior(scale=1, shape=2, upper=500),
            rho_prior=rb.BetaPrior(mode=0.5, sd=0.1, lower=0.01, upper=0.99),
        )
    ),
    "lp-tiny-rho-sd": rb.BVAR(outliers=rb.LenzaPrimiceri(rho_prior=rb.BetaPrior(0.8, 1e-5))),
    "mh-settings": bvar(
        hierarchy="mh",
        mh=rb.MHSettings(
            draws=100,
            burn=50,
            chains=3,
            thin=2,
            target_acceptance=0.25,
            seed=7,
            rhat_max=1.05,
            ess_min=100,
        ),
    ),
    "mh-mapping": bvar(hierarchy="mh", mh={"draws": 100, "seed": np.int64(3)}),
    "hyperpriors": bvar(
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
    "hyperpriors-psi-unbounded": bvar(
        hyper=("lambda", "psi"),
        hyperpriors=rb.GLPHyperpriors(psi=rb.InvGammaPrior(scale=0.02, shape=0.03)),
    ),
    "psi-tuple": bvar(psi=(1, 2.5, np.float64(3.25))),
    "psi-array": bvar(psi=np.array([0.1, 0.2, 0.3])),
    "mean-mapping": bvar(mean={"rate": 0.9, "gdp": np.float32(0.5), "prices": 0}),
    "mean-scalar": bvar(mean=np.int64(1)),
    "soc-scope-all": bvar(soc=2, soc_scope="all"),
    "soc-sur-fixed": bvar(hierarchy="fixed", soc=0.5, sur=3.0, lambda_=0.15, alpha=1.5),
    "hyper-explicit": bvar(hyper=["psi", "lambda"]),
    "deterministic-ct": bvar(lags=3, deterministic="ct"),
    "deterministic-n": bvar(deterministic="n", hierarchy="fixed", var_const=1e5),
    "everything": bvar(
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


@pytest.mark.parametrize("model", ROUND_TRIP.values(), ids=ROUND_TRIP.keys())
def test_from_spec_inverts_spec_through_json(model):
    payload = json.dumps(model.spec())  # plain JSON, no ``default=``
    rebuilt = rb.BVAR.from_spec(json.loads(payload))
    assert rebuilt == model
    assert spec_hash(rebuilt) == spec_hash(model)
    assert rebuilt.name == model.name
    assert rebuilt.axes == model.axes
    assert json.dumps(rebuilt.spec()) == payload
    # and once more: the rebuilt model's spec rebuilds the same model
    assert rb.BVAR.from_spec(json.loads(json.dumps(rebuilt.spec()))) == model


def test_forecast_attrs_rebuild_the_model_that_made_them(data):
    model = bvar(
        lags=2,
        hierarchy="fixed",
        psi=(1e-4, 2e-5, 0.05),
        mean={"rate": 0.9, "gdp": 0.6},
        soc=1.0,
        outliers=rb.LenzaPrimiceri(fixed=(3.0, 2.0, 1.5, 0.7)),
    )
    fit = model.fit(data)
    attrs = fit.forecast(2, draws=20, seed=1).dataset.attrs
    rebuilt = rb.BVAR.from_spec(json.loads(attrs["spec"]))
    assert rebuilt == model
    assert spec_hash(rebuilt) == attrs["spec_hash"]
    assert attrs["model"] == rebuilt.name
    irf_attrs = fit.irf(2, draws=10, seed=1).data.attrs
    assert irf_attrs["spec_hash"] == attrs["spec_hash"]


WRONG_LP_PRIORS = {
    "rho-pareto": {"rho_prior": rb.ParetoPrior()},
    "rho-gamma": {"rho_prior": rb.GammaPrior(mode=0.8, sd=0.2, lower=0.01, upper=0.99)},
    "rho-none": {"rho_prior": None},
    "scale-beta": {"scale_prior": rb.BetaPrior(mode=0.5, sd=0.2)},
    "scale-none": {"scale_prior": None},
    "scale-float": {"scale_prior": 1.0},
}


def test_lenza_primiceri_rejects_hyperpriors_of_the_wrong_type():
    """Like ``GLPHyperpriors`` and ``Minnesota(hyperpriors=)``: invalid compositions fail at
    construction (SPEC §4.5), not with an AttributeError inside ``fit`` or a broken round trip.
    All cases are checked in one test (one root cause); the message lists every accepted one."""
    accepted = []
    for name, kwargs in WRONG_LP_PRIORS.items():
        try:
            rb.LenzaPrimiceri(**kwargs)
        except (TypeError, ValueError):
            continue
        accepted.append(name)
    assert not accepted, f"LenzaPrimiceri accepted wrongly typed priors: {accepted}"


def test_lenza_primiceri_prior_mappings_are_converted_or_rejected():
    """A mapping either becomes the right prior (so the spec round-trips) or is rejected."""
    problems = []
    for kwargs in (
        {"scale_prior": {"scale": 1.0, "shape": 2.0, "upper": 500.0}},
        {"rho_prior": {"mode": 0.5, "sd": 0.2}},
    ):
        try:
            lp = rb.LenzaPrimiceri(**kwargs)
        except (TypeError, ValueError):
            continue
        if not (
            isinstance(lp.scale_prior, rb.ParetoPrior) and isinstance(lp.rho_prior, rb.BetaPrior)
        ):
            problems.append(
                f"{next(iter(kwargs))} kept as {type(next(iter(kwargs.values()))).__name__}"
            )
            continue
        model = rb.BVAR(outliers=lp)
        if rb.BVAR.from_spec(json.loads(json.dumps(model.spec()))) != model:
            problems.append(f"{next(iter(kwargs))}: spec does not round-trip")
    assert not problems, problems


# =========================================================================== validation


def test_minnesota_converts_an_mh_mapping_to_mh_settings():
    prior = rb.Minnesota(hierarchy="mh", mh={"draws": 300, "chains": 2, "seed": 5})
    assert isinstance(prior.mh, rb.MHSettings)
    assert prior.mh == rb.MHSettings(draws=300, chains=2, seed=5)
    assert prior == rb.Minnesota(hierarchy="mh", mh=rb.MHSettings(draws=300, chains=2, seed=5))
    assert rb.Minnesota(mh={}).mh == rb.MHSettings()


@pytest.mark.parametrize(
    "mh",
    ["fast", None, 5, ("draws", 100), rb.GLPHyperpriors(), {"bogus": 1}, {"draws": 1.5}],
    ids=["str", "none", "int", "tuple", "hyperpriors", "unknown-key", "float-draws"],
)
def test_minnesota_rejects_mh_settings_of_the_wrong_type(mh):
    with pytest.raises(TypeError):
        rb.Minnesota(hierarchy="mh", mh=mh)


@pytest.mark.parametrize(
    "hyperpriors",
    [
        None,
        {},
        "glp",
        rb.GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=5.0),
        {"lambda_": {"mode": 0.2, "sd": 0.4, "lower": 1e-4, "upper": 5.0}},
    ],
    ids=["none", "empty-mapping", "str", "gamma", "mapping"],
)
def test_minnesota_rejects_hyperpriors_of_the_wrong_type(hyperpriors):
    with pytest.raises(TypeError, match="GLPHyperpriors"):
        rb.Minnesota(hyperpriors=hyperpriors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_": rb.InvGammaPrior(scale=1.0, shape=1.0)},
        {"alpha": {"mode": 2.0, "sd": 0.25, "lower": 1.0, "upper": 3.0}},
        {"soc": None},
        {"sur": 1.0},
        {"psi": rb.GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=5.0)},
    ],
    ids=["lambda-invgamma", "alpha-mapping", "soc-none", "sur-float", "psi-gamma"],
)
def test_glp_hyperpriors_reject_priors_of_the_wrong_type(kwargs):
    with pytest.raises(TypeError, match="GLPHyperpriors"):
        rb.GLPHyperpriors(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"psi": (1.0, np.nan, 2.0)},
        {"psi": (1.0, np.inf, 2.0)},
        {"psi": np.array([1.0, 2.0, -np.inf])},
        {"mean": np.nan},
        {"mean": np.inf},
        {"mean": float("-inf")},
        {"mean": {"gdp": np.inf}},
        {"mean": {"gdp": 1.0, "rate": np.nan}},
    ],
    ids=[
        "psi-nan",
        "psi-inf",
        "psi-minus-inf",
        "mean-nan",
        "mean-inf",
        "mean-minus-inf",
        "mean-mapping-inf",
        "mean-mapping-nan",
    ],
)
def test_minnesota_rejects_non_finite_psi_and_means(kwargs):
    with pytest.raises(ValueError, match="finite"):
        rb.Minnesota(**kwargs)


@pytest.mark.parametrize(
    "start",
    [pd.Period("2020Q1", freq="Q"), pd.Timestamp("2020-03-01"), 2020, 2020.25, ("2020Q1",)],
    ids=["period", "timestamp", "int", "float", "tuple"],
)
def test_lenza_primiceri_start_must_be_a_string(start):
    with pytest.raises(TypeError, match="start must be a str"):
        rb.LenzaPrimiceri(start=start)


@pytest.mark.parametrize("start", ["", "nan", "NaT", " ", "not a date"])
def test_lenza_primiceri_start_must_parse_to_a_period(start):
    with pytest.raises(ValueError, match="cannot parse start"):
        rb.LenzaPrimiceri(start=start)


@pytest.mark.parametrize("start", ["2020Q2", "2020-04", "2021Q1"])
def test_lenza_primiceri_accepts_period_strings(start):
    assert rb.LenzaPrimiceri(start=start).start == start


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": 0.8, "sd": 0.2, "lower": 0.1, "upper": 0.5}, "within its bounds"),
        ({"mode": 0.2, "sd": 0.1, "lower": 0.3, "upper": 0.9}, "within its bounds"),
        ({"mode": 0.5, "sd": 0.2, "lower": 0.5, "upper": 0.5 + 1e-9}, "of its mass"),
        ({"mode": 0.5, "sd": 0.2, "lower": 0.6, "upper": 0.6}, "0 ≤ lower < upper ≤ 1"),
        ({"mode": 0.5, "sd": 0.2, "lower": 0.7, "upper": 0.6}, "0 ≤ lower < upper ≤ 1"),
        ({"mode": 0.5, "sd": 0.2, "lower": -0.1}, "0 ≤ lower < upper ≤ 1"),
        ({"mode": 0.5, "sd": 0.2, "upper": 1.5}, "0 ≤ lower < upper ≤ 1"),
        ({"mode": 0.5, "sd": 0.2, "lower": np.nan}, "0 ≤ lower < upper ≤ 1"),
        ({"mode": 0.0, "sd": 0.2}, r"mode must lie in \(0, 1\)"),
        ({"mode": 1.0, "sd": 0.2}, r"mode must lie in \(0, 1\)"),
        ({"mode": np.nan, "sd": 0.2}, r"mode must lie in \(0, 1\)"),
        ({"mode": 0.5, "sd": 0.0}, "sd must be positive"),
        ({"mode": 0.5, "sd": -0.1}, "sd must be positive"),
        ({"mode": 0.5, "sd": np.nan}, "sd must be positive"),
        ({"mode": 0.5, "sd": np.inf}, "sd must be positive"),
        ({"mode": 0.5, "sd": 0.5}, "no unimodal Beta"),
    ],
    ids=[
        "mode-above-upper",
        "mode-below-lower",
        "tiny-mass",
        "empty-bounds",
        "inverted-bounds",
        "negative-lower",
        "upper-above-one",
        "nan-lower",
        "mode-zero",
        "mode-one",
        "mode-nan",
        "sd-zero",
        "sd-negative",
        "sd-nan",
        "sd-inf",
        "sd-too-large",
    ],
)
def test_beta_prior_checks_mode_sd_bounds_and_mass(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rb.BetaPrior(**kwargs)


def test_beta_prior_rejects_booleans():
    with pytest.raises(TypeError):
        rb.BetaPrior(mode=True, sd=0.2)


def test_beta_prior_with_little_but_enough_mass_samples_inside_its_bounds():
    prior = rb.BetaPrior(mode=0.5, sd=0.01, lower=0.1, upper=0.5)  # half the mass
    draws = prior.sample(2000, np.random.default_rng(0))
    assert np.all((draws >= 0.1) & (draws <= 0.5))
    assert prior.logpdf(0.5) > prior.logpdf(0.45)


def test_beta_prior_with_tiny_sd_is_constructed_exactly():
    sd = 1e-5
    prior = rb.BetaPrior(mode=0.8, sd=sd)
    a, b = prior.a, prior.b
    np.testing.assert_allclose((a - 1.0) / (a + b - 2.0), 0.8, rtol=1e-12)
    np.testing.assert_allclose(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))), sd, rtol=1e-8)
    # essentially normal: the log density at the mode is −log(sd·√(2π))
    np.testing.assert_allclose(prior.logpdf(0.8), -np.log(sd * np.sqrt(2 * np.pi)), atol=1e-3)
    draws = prior.sample(4000, np.random.default_rng(1))
    assert np.all(np.isfinite(draws))
    assert np.all((draws >= prior.lower) & (draws <= prior.upper))
    assert abs(draws.mean() - 0.8) < 5 * sd / np.sqrt(draws.size)
    np.testing.assert_allclose(draws.std(), sd, rtol=0.1)
    lp = rb.LenzaPrimiceri(rho_prior=prior)
    rho = lp.sample_prior(500, np.random.default_rng(2))[:, 3]
    assert np.all(np.isfinite(rho)) and abs(rho.mean() - 0.8) < 1e-5
    assert spec_hash(rb.BVAR(outliers=lp)) != spec_hash(rb.BVAR(outliers="lenza-primiceri"))


@pytest.mark.parametrize("sd", [1e-5, 1e-7, 1e-9, 1e-12])
def test_beta_prior_with_tiny_sd_is_rejected_or_numerically_sound(sd):
    """A Beta prior the constructor accepts must have usable draws and log density (both feed
    the ρ prior of the Lenza-Primiceri scaling); otherwise construction should refuse it."""
    try:
        prior = rb.BetaPrior(mode=0.8, sd=sd)
    except ValueError:
        return
    problems = []
    normal = -np.log(sd * np.sqrt(2 * np.pi))  # essentially normal at these precisions
    if not abs(prior.logpdf(0.8) - normal) < 0.5:
        problems.append(f"logpdf at the mode {prior.logpdf(0.8):.6g}, expected {normal:.6g}")
    draws = prior.sample(300, np.random.default_rng(0))  # SciPy's ppf is slow when it fails
    if not np.all(np.isfinite(draws)):
        problems.append(f"{np.isnan(draws).sum()} of {draws.size} draws are NaN")
    elif not np.all((draws >= prior.lower) & (draws <= prior.upper)):
        problems.append("draws outside the bounds")
    elif not abs(draws.std() / sd - 1.0) < 0.25:
        problems.append(f"draw s.d. {draws.std():.3g} instead of {sd:g}")
    assert not problems, problems


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"upper": 0.0}, "bounds must be positive"),
        ({"upper": -1.0}, "bounds must be positive"),
        ({"upper": 0}, "bounds must be positive"),
        ({"lower": 0.0}, "bounds must be positive"),
        ({"lower": -2.0, "upper": 1.0}, "bounds must be positive"),
        ({"upper": np.inf}, "bounds must be finite"),
        ({"lower": 2.0, "upper": 1.0}, "lower bound must be below"),
    ],
    ids=[
        "upper-zero",
        "upper-negative",
        "upper-int-zero",
        "lower-zero",
        "lower-negative",
        "upper-inf",
        "inverted",
    ],
)
def test_inverse_gamma_prior_rejects_non_positive_or_invalid_bounds(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rb.InvGammaPrior(scale=1.0, shape=1.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": np.inf},
        {"shape": -np.inf},
        {"shape": np.nan},
        {"shape": 0.0},
        {"scale": np.inf},
        {"scale": np.nan},
        {"upper": np.inf},
        {"upper": np.nan},
        {"scale": 2.0, "upper": 2.0},
    ],
    ids=[
        "shape-inf",
        "shape-minus-inf",
        "shape-nan",
        "shape-zero",
        "scale-inf",
        "scale-nan",
        "upper-inf",
        "upper-nan",
        "upper-at-scale",
    ],
)
def test_pareto_prior_rejects_non_finite_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="Pareto prior needs"):
        rb.ParetoPrior(**kwargs)


@pytest.mark.parametrize(
    "seed",
    [1.5, True, False, -1, "3", np.float64(2.0), np.random.default_rng(0), (1, 2)],
    ids=["float", "true", "false", "negative", "str", "numpy-float", "generator", "tuple"],
)
def test_fit_rejects_seeds_that_are_not_non_negative_ints(data, seed):
    with pytest.raises(ValueError, match="seed must be a non-negative int"):
        bvar(hierarchy="fixed").fit(data, seed=seed)


def test_fit_accepts_numpy_integer_seeds(data):
    fit = bvar(hierarchy="fixed").fit(data, seed=np.int64(3))
    assert fit.mh_seed == 3 and type(fit.mh_seed) is int


@pytest.fixture(scope="module")
def eb_dummies_fit(data):
    """EB with SOC and SUR: λ, μ, δ are free."""
    return bvar(soc=1.0, sur=1.0).fit(data)


@pytest.mark.parametrize("method", ["log_ml", "log_posterior", "posterior"])
@pytest.mark.parametrize("key", ["soc", "sur"])
def test_overriding_a_free_dummy_hyperparameter_with_none_is_an_error(eb_dummies_fit, method, key):
    with pytest.raises(ValueError, match=f"{key} is a free hyperparameter here"):
        getattr(eb_dummies_fit, method)({key: None})


def test_none_switches_off_a_dummy_that_is_not_free(data):
    """With ``hyper=("lambda",)`` the SOC weight is fixed, so ``{"soc": None}`` drops the dummy."""
    fit = bvar(soc=1.0, hyper=("lambda",)).fit(data)
    assert fit.problem.free.names == ("lambda",)
    lam = 0.25
    reference = bvar(hierarchy="fixed", lambda_=lam).fit(data)
    np.testing.assert_allclose(
        fit.log_ml({"soc": None, "lambda": lam}), reference.log_ml(), rtol=1e-12
    )
    fixed_soc = bvar(hierarchy="fixed", soc=1.0).fit(data)
    np.testing.assert_allclose(
        fixed_soc.log_ml({"soc": None}), bvar(hierarchy="fixed").fit(data).log_ml(), rtol=1e-12
    )


# =========================================================================== seed labels


@pytest.mark.parametrize("seed", [5, np.int64(5), np.uint32(5)], ids=["int", "int64", "uint32"])
def test_integer_seeds_are_labelled_by_their_value(fixed_fit, seed):
    assert fixed_fit.forecast(2, draws=10, seed=seed).dataset.attrs["seed"] == "5"
    assert fixed_fit.irf(2, draws=10, seed=seed).data.attrs["seed"] == "5"
    assert seed_label(seed) == "5"


def test_seed_sequences_are_labelled_by_entropy_and_spawn_key(fixed_fit):
    root = np.random.SeedSequence(7)
    child = np.random.SeedSequence(7).spawn(3)[2]
    a = fixed_fit.forecast(2, draws=10, seed=root)
    b = fixed_fit.forecast(2, draws=10, seed=child)
    assert a.dataset.attrs["seed"] == "SeedSequence(entropy=7, spawn_key=())"
    assert b.dataset.attrs["seed"] == "SeedSequence(entropy=7, spawn_key=(2,))"
    again = fixed_fit.forecast(2, draws=10, seed=np.random.SeedSequence(7))
    np.testing.assert_array_equal(a.paths.values, again.paths.values)
    assert not np.array_equal(a.paths.values, b.paths.values)


def test_a_fresh_generator_is_labelled_by_its_seed_sequence(fixed_fit):
    a = fixed_fit.forecast(2, draws=10, seed=np.random.default_rng(5))
    b = fixed_fit.forecast(2, draws=10, seed=np.random.default_rng(5))
    label = a.dataset.attrs["seed"]
    assert label == "SeedSequence(entropy=5, spawn_key=())"
    assert b.dataset.attrs["seed"] == label
    np.testing.assert_array_equal(a.paths.values, b.paths.values)
    # the Generator draws the same stream as the SeedSequence it is labelled with
    c = fixed_fit.forecast(2, draws=10, seed=np.random.SeedSequence(5))
    assert c.dataset.attrs["seed"] == label
    np.testing.assert_array_equal(a.paths.values, c.paths.values)
    spawned = np.random.default_rng(5).spawn(2)[1]
    assert seed_label(spawned) == "SeedSequence(entropy=5, spawn_key=(1,))"


@pytest.mark.parametrize("kind", ["forecast", "irf"])
def test_an_advanced_generator_gets_a_new_label_for_every_draw(fixed_fit, kind):
    def run(gen):
        out = getattr(fixed_fit, kind)(2, draws=10, seed=gen)
        return out.dataset.attrs["seed"] if kind == "forecast" else out.data.attrs["seed"]

    gen = np.random.default_rng(5)
    expected, labels = [], []
    for _ in range(3):
        expected.append(seed_label(gen))  # the label describes the state before drawing
        labels.append(run(gen))
    assert labels == expected
    assert len(set(labels)) == 3
    assert labels[0] == "SeedSequence(entropy=5, spawn_key=())"
    assert all(label.startswith("PCG64(state=") for label in labels[1:])
    # replaying the generator replays the labels
    replay = np.random.default_rng(5)
    assert [run(replay) for _ in range(3)] == labels


def test_advanced_generators_are_labelled_by_state_for_the_benchmarks(data):
    for model in (rb.OLSVAR(1), rb.RandomWalk(), rb.RandomWalk(drift=True)):
        fit = model.fit(data)
        gen = np.random.default_rng(3)
        labels = [fit.forecast(1, draws=5, seed=gen).dataset.attrs["seed"] for _ in range(3)]
        assert len(set(labels)) == 3, (model, labels)
        assert labels[0] == "SeedSequence(entropy=3, spawn_key=())"


def test_advanced_generators_with_other_bit_generators_get_distinct_labels(fixed_fit):
    gen = np.random.Generator(np.random.Philox(11))
    first = fixed_fit.forecast(1, draws=5, seed=gen).dataset.attrs["seed"]
    second = fixed_fit.forecast(1, draws=5, seed=gen).dataset.attrs["seed"]
    assert first == "SeedSequence(entropy=11, spawn_key=())"
    assert second.startswith("Philox(state=") and second != first


def test_no_seed_still_records_a_reproducible_label(fixed_fit):
    a = fixed_fit.forecast(1, draws=5).dataset.attrs["seed"]
    b = fixed_fit.forecast(1, draws=5).dataset.attrs["seed"]
    assert a.startswith("SeedSequence(entropy=") and a != b
    entropy = int(a.removeprefix("SeedSequence(entropy=").split(",")[0])
    replay = fixed_fit.forecast(1, draws=5, seed=np.random.SeedSequence(entropy))
    assert replay.dataset.attrs["seed"] == a


def test_irf_attrs_carry_seed_mh_seed_and_spec_hash(fixed_fit):
    attrs = fixed_fit.irf(3, draws=10, seed=3).data.attrs
    assert attrs["seed"] == "3"
    assert attrs["mh_seed"] == -1  # no MH: the hyperparameters are fixed
    assert attrs["spec_hash"] == spec_hash(fixed_fit.model)
    assert attrs["model"] == fixed_fit.model.name


def test_mh_seed_is_recorded_in_forecast_and_irf_attrs(mh_fit):
    assert mh_fit.mh is not None and mh_fit.mh_seed == 4
    irf = mh_fit.irf(3, draws=10, seed=np.random.SeedSequence(9)).data.attrs
    fc = mh_fit.forecast(2, draws=10, seed=1).dataset.attrs
    assert irf["seed"] == "SeedSequence(entropy=9, spawn_key=())"
    assert fc["seed"] == "1"
    assert irf["mh_seed"] == fc["mh_seed"] == 4
    assert irf["spec_hash"] == fc["spec_hash"] == spec_hash(mh_fit.model)
    # the settings seed is used (and recorded) when fit() gets none
    default = mh_fit.model.fit(mh_fit.data)
    assert default.mh_seed == TOL_MH.seed
    assert default.forecast(1, draws=5, seed=0).dataset.attrs["mh_seed"] == TOL_MH.seed


def test_eb_forecasts_record_no_mh_seed(data):
    fit = bvar(lambda_=0.2).fit(data, seed=9)
    assert fit.mh is None
    assert fit.forecast(1, draws=5, seed=0).dataset.attrs["mh_seed"] == -1
    assert fit.irf(1, draws=5, seed=0).data.attrs["mh_seed"] == -1


# =========================================================================== slow


@pytest.mark.slow
def test_rebuilt_specs_reproduce_mh_and_covid_forecasts_bit_for_bit(covid_data):
    """Equal specs must give identical forecasts (Model protocol), also through JSON with MH and
    Lenza-Primiceri scaling; the attrs (seed, mh_seed, spec_hash) agree as well."""
    model = bvar(
        lags=2,
        soc=1.0,
        mean={"gdp": 0.9, "prices": 0.8, "rate": 0.95},
        hierarchy="mh",
        mh={"draws": 600, "burn": 300, "chains": 2, "seed": 2},
        outliers=rb.LenzaPrimiceri(scale_prior=rb.ParetoPrior(upper=200)),
    )
    rebuilt = rb.BVAR.from_spec(json.loads(json.dumps(model.spec())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # MH convergence notes are not the point
        fits = [m.fit(covid_data, seed=5) for m in (model, rebuilt)]
    np.testing.assert_array_equal(fits[0].mh.chains, fits[1].mh.chains)
    fcs = [f.forecast(6, draws=300, seed=np.random.SeedSequence(8)) for f in fits]
    np.testing.assert_array_equal(fcs[0].paths.values, fcs[1].paths.values)
    np.testing.assert_array_equal(fcs[0].cond_chol.values, fcs[1].cond_chol.values)
    for key in ("seed", "mh_seed", "spec_hash", "model", "spec"):
        assert fcs[0].dataset.attrs[key] == fcs[1].dataset.attrs[key]
    assert fcs[0].dataset.attrs["mh_seed"] == 5
    irfs = [f.irf(4, draws=200, seed=3) for f in fits]
    np.testing.assert_array_equal(irfs[0].data.values, irfs[1].data.values)
    assert irfs[0].data.attrs == irfs[1].data.attrs


@pytest.mark.slow
def test_grid_over_outlier_treatments_pairs_along_the_outliers_axis(covid_data):
    """EB BVARs with and without COVID scaling (two priors each) over origins spanning 2020."""
    entries = rb.grid(
        rb.BVAR,
        mean=[rb.Linear(), rb.Linear(prior=rb.Minnesota(sur=1.0))],
        outliers=[None, "lenza-primiceri"],
    )
    assert all(e.axes == e.model.axes for e in entries)
    plain = rb.BVAR(outliers="lenza-primiceri")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no-outlier-treatment notes over 2020
        res = rb.compare(
            [*entries, Entry("plain LP", plain, plain.axes)],
            covid_data,
            origins=rb.Origins("2019Q3", "2021Q2"),
            window=rb.Expanding(),
            draws=200,
        )
    by_axis = res.verdict(axis="outliers").set_index("model")
    with_lp = [e for e in entries if e.model.outliers is not None]
    assert set(by_axis.index) == {e.name for e in with_lp} | {"plain LP"}
    for e in with_lp:
        target = {**e.axes, "outliers": "none"}
        ref = next(f.name for f in entries if f.axes == target)
        assert by_axis.loc[e.name, "reference"] == ref
    assert by_axis.loc["plain LP", "reference"] == entries[0].name
    assert by_axis["verdict"].isin(["better", "worse", "inconclusive"]).all()
    # against the baseline (default prior, no LP) the hand-built LP model differs on one axis,
    # and it is the same model as its grid twin, so both get the same scores
    assert entries[1].model == plain
    against_base = res.verdict().set_index("model")
    assert against_base.loc["plain LP", "axis"] == "outliers"
    assert against_base.loc["plain LP", "verdict"] != "not comparable"
    np.testing.assert_array_equal(
        res.scores("plain LP")["lpd"].values, res.scores(entries[1].name)["lpd"].values
    )
