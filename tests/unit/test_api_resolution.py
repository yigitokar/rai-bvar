"""Composition and engine resolution of ``BVAR`` (DESIGN §11.1, §11.3; D-10, D-11, D-19, D-21).

* The ``auto`` defaults resolve as specified and the resolved components are stored.
* The engine table of §11.1, row by row, with the static parts each MCMC row maps to (Mode I,
  Mode II, IW-equivalent, CCCM; ``static_parts`` and ``config.validate_composition``).
* Every construction error of §11.1 with its message.
* Names, axes (``scaling=conjugate`` for Mode II), capability flags and ``mcmc`` settings.
* ``fit``: keyword rules, argument forwarding to ``engines.mcmc.resolve.fit_bvar`` and the
  "engine not yet available" error while that entry point is missing.
* Constructing and describing MCMC compositions imports no JAX.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar import api
from rai_bvar.api import StaticParts, static_parts
from rai_bvar.engines.mcmc import config
from rai_bvar.models import spec_hash

P = rb.sv_priors


def bvar(form="auto", vol=None, lags=1, outliers=None, mcmc=None, **prior) -> rb.BVAR:
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form=form, **prior)),
        vol=rb.Constant() if vol is None else vol,
        lags=lags,
        outliers=outliers,
        mcmc=mcmc,
    )


# =========================================================================== resolution table

# (form, vol) → (resolved form, resolved vol, engine, mode)
#   mode: struct.contemp_scaling, mean_kind, coef_scaling, vol.persistent, vol.coupled
MODE_I = ("none", "linear_structural", "none")
MODE_II = ("lambda", "linear_structural", "lambda")
IW_EQ = ("lambda", "linear_structural", "none")
CCCM = ("none", "linear_reduced", "none")
CCCM_IW_EQ = ("lambda", "linear_reduced", "none")

TABLE = {
    # rows 1 and the auto resolution
    "auto×auto": ("auto", rb.Constant(), "conjugate", rb.Constant("iw"), "analytic", None),
    "auto×iw": ("auto", rb.Constant("iw"), "conjugate", rb.Constant("iw"), "analytic", None),
    "conj×auto": ("conjugate", rb.Constant(), "conjugate", rb.Constant("iw"), "analytic", None),
    "conj×iw": ("conjugate", rb.Constant("iw"), "conjugate", rb.Constant("iw"), "analytic", None),
    # row 3: structural × Constant("ig"), Mode I
    "auto×ig": ("auto", rb.Constant("ig"), "structural", rb.Constant("ig"), "mcmc", MODE_I),
    "struct×auto": ("structural", rb.Constant(), "structural", rb.Constant("ig"), "mcmc", MODE_I),
    "struct×ig": ("structural", rb.Constant("ig"), "structural", rb.Constant("ig"), "mcmc", MODE_I),
    # row 4: structural × Constant("iw"), Mode II
    "struct×iw": (
        "structural",
        rb.Constant("iw"),
        "structural",
        rb.Constant("iw"),
        "mcmc",
        MODE_II,
    ),
    # row 5: structural × Constant("iw_equivalent")
    "auto×iweq": (
        "auto",
        rb.Constant("iw_equivalent"),
        "structural",
        rb.Constant("iw_equivalent"),
        "mcmc",
        IW_EQ,
    ),
    "struct×iweq": (
        "structural",
        rb.Constant("iw_equivalent"),
        "structural",
        rb.Constant("iw_equivalent"),
        "mcmc",
        IW_EQ,
    ),
    # row 6: structural × SV, Mode I
    "auto×SV": ("auto", rb.SV(), "structural", rb.SV(), "mcmc", MODE_I),
    "struct×SVO-t": (
        "structural",
        rb.SV(outliers=True, tails="t"),
        "structural",
        rb.SV(outliers=True, tails="t"),
        "mcmc",
        MODE_I,
    ),
    # row 7: reduced × Constant("ig") or SV, CCCM
    "red×auto": ("reduced", rb.Constant(), "reduced", rb.Constant("ig"), "mcmc", CCCM),
    "red×ig": ("reduced", rb.Constant("ig"), "reduced", rb.Constant("ig"), "mcmc", CCCM),
    "red×SV(rw)": ("reduced", rb.SV(rw=True), "reduced", rb.SV(rw=True), "mcmc", CCCM),
    # row 8: reduced × Constant("iw_equivalent")
    "red×iweq": (
        "reduced",
        rb.Constant("iw_equivalent"),
        "reduced",
        rb.Constant("iw_equivalent"),
        "mcmc",
        CCCM_IW_EQ,
    ),
}


@pytest.mark.parametrize("row", TABLE.values(), ids=TABLE.keys())
def test_resolution_table(row):
    form, vol, resolved_form, resolved_vol, engine, mode = row
    model = bvar(form, vol)
    assert model.mean.prior.form == resolved_form  # resolved values are stored
    assert model.vol == resolved_vol
    assert model.engine == engine
    if engine == "analytic":
        assert model.mcmc is None
        with pytest.raises(ValueError, match="analytic engine"):
            static_parts(model)
        return
    assert model.mcmc == rb.MCMCSettings()
    parts = static_parts(model)
    assert isinstance(parts, StaticParts)
    assert (parts.struct.contemp_scaling, parts.mean_kind, parts.coef_scaling) == mode
    assert parts.struct == config.StructStatic(kind="recursive", contemp_scaling=mode[0])
    assert parts.coef == config.CoefPriorStatic()
    persistent = "constant" if isinstance(resolved_vol, rb.Constant) else "sv"
    assert parts.vol.persistent == persistent
    assert parts.vol.coupled == (mode[0] == "lambda")  # coupling iff a λ-scaled prior block
    # the parts form a legal composition (and resolve.py builds the StaticConfig from them)
    config.validate_composition(
        parts.struct,
        config.MeanStatic(kind=parts.mean_kind, coef_scaling=parts.coef_scaling),
        parts.coef,
        parts.vol,
    )


def test_bvar_with_sv_resolves_to_structural_mode_one():
    model = rb.BVAR(vol=rb.SV())
    assert model.engine == "mcmc"
    assert model.mean.prior.form == "structural"
    parts = static_parts(model)
    assert parts.struct.contemp_scaling == "none" and parts.coef_scaling == "none"  # Mode I
    assert parts.vol == config.VolStatic(persistent="sv")
    assert "scaling" not in model.axes["prior"]


SV_STATIC = {
    "rw": (rb.SV(rw=True), {"rw": True}),
    "fixed-b0": (rb.SV(h0=2.0), {"h0": "fixed_b0"}),
    "inv-gamma": (rb.SV(sigma2_prior=P.InvGamma(2.5, 0.1)), {"sigma2_prior": "inv_gamma"}),
    "rw-inv-gamma": (
        rb.SV(rw=True, sigma2_prior=P.InvGamma(2.5, 0.1)),
        {"rw": True, "sigma2_prior": "inv_gamma"},
    ),
    "outliers": (rb.SV(outliers=True, outlier_max=10), {"transitory": "outlier", "outlier_K": 10}),
    "t": (
        rb.SV(tails="t", nu_prior=P.DiscreteUniform(5, 12), t_scale="unit"),
        {"transitory": "t", "nu_grid": tuple(range(5, 13)), "t_scale": "unit"},
    ),
    "outliers-t": (rb.SV(outliers=True, tails="t"), {"transitory": "outlier_t"}),
    "parity": (
        rb.SV(offset=0.0, exact=False, asis=False, mixture="ksc7"),
        {"zero_offset": True, "exact": False, "asis": False, "mixture": "ksc7"},
    ),
    "absolute-offset": (rb.SV(offset=1e-4), {}),  # only c = 0 switches floor-hit masking on
}


@pytest.mark.parametrize(("sv", "changes"), SV_STATIC.values(), ids=SV_STATIC.keys())
def test_sv_settings_map_to_the_static_vol_config(sv, changes):
    parts = static_parts(rb.BVAR(vol=sv))
    assert parts.vol == config.VolStatic(persistent="sv", **changes)


# =========================================================================== errors

CONJUGATE_MSG = (
    "the conjugate Minnesota prior requires Constant(prior='iw'); use form='structural' "
    "(the MCMC default) or 'reduced'"
)
REDUCED_IW_MSG = (
    "Constant('iw') is the conjugate NIW prior; with form='reduced' use 'iw_equivalent' "
    "(Chan 2022) or 'ig'"
)
MH_MSG = (
    "hierarchical Minnesota inside MCMC arrives with the Phase 3 global-scale step; "
    "use 'eb' or 'fixed'"
)


@pytest.mark.parametrize(
    "vol",
    [rb.Constant("ig"), rb.Constant("iw_equivalent"), rb.SV(), rb.SV(outliers=True)],
    ids=lambda v: v.name if isinstance(v, rb.SV) else v.prior,
)
def test_conjugate_form_with_any_other_volatility_raises(vol):
    with pytest.raises(ValueError) as err:
        bvar("conjugate", vol)
    assert str(err.value) == CONJUGATE_MSG


@pytest.mark.parametrize("hierarchy", ["fixed", "eb", "mh"])
def test_reduced_form_with_the_conjugate_iw_prior_raises(hierarchy):
    with pytest.raises(ValueError) as err:
        bvar("reduced", rb.Constant("iw"), hierarchy=hierarchy)
    assert str(err.value) == REDUCED_IW_MSG


@pytest.mark.parametrize(
    "vol", [rb.Constant("ig"), rb.SV()], ids=lambda v: v.name if isinstance(v, rb.SV) else v.prior
)
@pytest.mark.parametrize("form", ["structural", "reduced"])
def test_metropolis_hastings_hierarchy_is_rejected_on_mcmc(form, vol):
    with pytest.raises(NotImplementedError) as err:
        bvar(form, vol, hierarchy="mh")
    assert str(err.value) == MH_MSG


@pytest.mark.parametrize(
    "outliers", ["lenza-primiceri", rb.LenzaPrimiceri(fixed=(2.0, 2.0, 2.0, 0.8))], ids=repr
)
@pytest.mark.parametrize("vol", [rb.Constant("ig"), rb.SV()], ids=["ig", "SV"])
def test_lenza_primiceri_scaling_is_rejected_on_mcmc(vol, outliers):
    with pytest.raises(NotImplementedError, match="arrives in Phase 3") as err:
        bvar(vol=vol, outliers=outliers, hierarchy="fixed")
    assert "double treatment" not in str(err.value)
    assert "SV(outliers=True)" in str(err.value)


def test_lenza_primiceri_with_sv_outliers_is_a_double_treatment():
    with pytest.raises(NotImplementedError, match="double treatment"):
        bvar(vol=rb.SV(outliers=True), outliers="lenza-primiceri")


@pytest.mark.parametrize("dummies", [{"soc": 1.0}, {"sur": 1.0}, {"soc": 1.0, "sur": 2.0}])
@pytest.mark.parametrize(
    ("form", "vol"),
    [("structural", rb.Constant("ig")), ("structural", rb.Constant("iw")),
     ("structural", rb.SV()), ("reduced", rb.Constant("ig")),
     ("reduced", rb.Constant("iw_equivalent"))],
)  # fmt: skip
def test_soc_sur_dummies_run_on_mcmc(form, vol, dummies):
    """WP25: SOC/SUR dummies are accepted on the MCMC engine (every form and volatility), with
    their label; the analytic engine is unchanged; ``hierarchy="mh"`` is still rejected."""
    model = bvar(form, vol, **dummies)
    assert model.engine == "mcmc"
    label = "+".join(d.upper() for d in ("soc", "sur") if d in dummies)
    assert label in model.axes["prior"]
    assert rb.BVAR.from_spec(model.spec()) == model
    assert bvar("conjugate", **dummies).engine == "analytic"
    with pytest.raises(NotImplementedError, match="hierarchical Minnesota inside MCMC"):
        bvar(form, vol, hierarchy="mh", **dummies)


def test_structural_eb_over_psi_with_cross_shrinkage_is_rejected():
    with pytest.raises(ValueError, match=r"empirical Bayes over psi .* cross != 1"):
        bvar("structural", rb.SV(), cross=0.5, hyper=("lambda", "psi"))
    # allowed: psi fixed by the hierarchy, cross = 1, alpha free, or the reduced form
    bvar("structural", rb.SV(), cross=0.5, hyper=("lambda", "psi"), hierarchy="fixed")
    bvar("structural", rb.SV(), cross=1.0, hyper=("lambda", "psi"))
    bvar("structural", rb.SV(), cross=0.5, hyper=("lambda", "alpha"))
    bvar("reduced", rb.SV(), cross=0.5, hyper=("lambda", "psi"))


@pytest.mark.parametrize("form", ["auto", "conjugate"])
def test_contemp_shrinkage_needs_a_non_conjugate_form(form):
    with pytest.raises(ValueError, match="contemp != 1"):
        bvar(form, rb.SV(), contemp=0.5)
    assert bvar("structural", rb.SV(), contemp=0.5).mean.prior.contemp == 0.5


def test_unknown_components_are_not_implemented():
    class Bart:
        prior = rb.Minnesota()

    with pytest.raises(NotImplementedError, match="Linear × Constant and Linear × SV"):
        rb.BVAR(mean=Bart())
    with pytest.raises(NotImplementedError, match="Linear × Constant and Linear × SV"):
        rb.BVAR(vol="sv")
    with pytest.raises(TypeError, match="must be a Minnesota prior"):
        rb.BVAR(mean=rb.Linear(prior="minnesota"))


def test_construction_runs_the_jax_free_composition_check(monkeypatch):
    seen = []

    def spy(struct, mean, coef, vol):
        seen.append((struct, mean, coef, vol))
        raise ValueError("rejected by validate_composition")

    monkeypatch.setattr(config, "validate_composition", spy)
    with pytest.raises(ValueError, match="rejected by validate_composition"):
        bvar("structural", rb.Constant("iw"))
    struct, mean, coef, vol = seen[0]
    assert mean == config.MeanStatic(kind="linear_structural", coef_scaling="lambda")
    assert struct.contemp_scaling == "lambda" and vol.coupled and coef == config.CoefPriorStatic()
    rb.BVAR()  # analytic compositions never reach the MCMC check
    assert len(seen) == 1


# =========================================================================== identity


def test_mode_two_prior_axis_carries_the_conjugate_scaling():
    mode1, mode2 = bvar("structural", rb.Constant("ig")), bvar("structural", rb.Constant("iw"))
    assert mode2.axes["prior"] == "Minnesota(structural, eb), scaling=conjugate"
    assert mode1.axes["prior"] == "Minnesota(structural, eb)"
    differing = sorted(k for k in mode1.axes if mode1.axes[k] != mode2.axes[k])
    assert differing == ["prior"]  # R2: Mode I vs Mode II is a prior comparison, not a vol one
    assert mode1.axes["vol"] == mode2.axes["vol"] == "Constant"


def test_iw_equivalent_is_a_volatility_choice():
    iweq, mode1 = bvar("structural", rb.Constant("iw_equivalent")), bvar("structural")
    assert iweq.axes["vol"] == "Constant(iw_equivalent)"
    assert iweq.axes["prior"] == mode1.axes["prior"] == "Minnesota(structural, eb)"
    assert sorted(k for k in iweq.axes if iweq.axes[k] != mode1.axes[k]) == ["vol"]


def test_sv_arms_differ_from_constant_along_the_vol_axis_only():
    base = bvar("structural", rb.Constant("ig"))
    for vol in (rb.SV(), rb.SV(outliers=True), rb.SV(tails="t"), rb.SV(rw=True)):
        model = bvar("structural", vol)
        assert sorted(k for k in base.axes if base.axes[k] != model.axes[k]) == ["vol"]
        assert model.axes["vol"] == vol.name


MCMC_MODELS = {
    "mode-i": bvar("structural", rb.Constant("ig")),
    "mode-ii": bvar("structural", rb.Constant("iw")),
    "iw-eq": bvar("structural", rb.Constant("iw_equivalent")),
    "sv": bvar(vol=rb.SV()),
    "sv-exact-off": bvar(vol=rb.SV(exact=False)),
    "svo": bvar(vol=rb.SV(outliers=True)),
    "svo-t": bvar(vol=rb.SV(outliers=True, tails="t")),
    "sv-rw": bvar(vol=rb.SV(rw=True)),
    "reduced-sv": bvar("reduced", rb.SV()),
    "reduced-iweq": bvar("reduced", rb.Constant("iw_equivalent")),
    "sv-p4": bvar(vol=rb.SV(), lags=4),
    "sv-cross": bvar("structural", rb.SV(), cross=0.5),
    "sv-contemp": bvar("structural", rb.SV(), contemp=2.0),
    "sv-draws": bvar(vol=rb.SV(), mcmc=rb.MCMCSettings(draws=500)),
    "sv-seed": bvar(vol=rb.SV(), mcmc=rb.MCMCSettings(seed=1)),
}


def test_mcmc_names_and_spec_hashes_are_distinct():
    names = [m.name for m in MCMC_MODELS.values()]
    assert len(set(names)) == len(names), names
    hashes = [spec_hash(m) for m in MCMC_MODELS.values()]
    assert len(set(hashes)) == len(hashes)
    assert MCMC_MODELS["sv"].name == "BVAR(Minnesota(structural, eb), SV, p=1)"
    assert MCMC_MODELS["mode-ii"].name == "BVAR(Minnesota(structural, eb), Constant(iw), p=1)"
    assert MCMC_MODELS["svo-t"].name == "BVAR(Minnesota(structural, eb), SV(outliers,t), p=1)"
    assert MCMC_MODELS["sv-draws"].name.startswith("BVAR(Minnesota(structural, eb), SV, p=1, mcmc=")


def test_phase1_names_do_not_mention_the_volatility():
    assert rb.BVAR().name == "BVAR(Minnesota(conjugate, eb), p=1)"


def test_capability_flags_follow_the_engine():
    for model in MCMC_MODELS.values():
        assert model.uses_fit_seed and model.supports_warm_start
        assert model.requires_in_loop_forecast is False
    analytic = rb.BVAR()
    assert not analytic.uses_fit_seed and not analytic.supports_warm_start
    assert rb.BVAR.requires_in_loop_forecast is False


def test_mcmc_settings_are_resolved_and_validated():
    assert bvar(vol=rb.SV()).mcmc == rb.MCMCSettings()
    from_mapping = bvar(vol=rb.SV(), mcmc={"chains": 2, "draws": np.int64(100)})
    assert from_mapping.mcmc == rb.MCMCSettings(chains=2, draws=100)
    with pytest.raises(TypeError, match="mcmc must be an MCMCSettings"):
        bvar(vol=rb.SV(), mcmc=4)
    with pytest.raises(TypeError, match="mcmc must be an MCMCSettings"):
        rb.BVAR(mcmc="fast")  # validated even where it is unused
    with pytest.raises(ValueError):
        bvar(vol=rb.SV(), mcmc={"chains": 0})


def test_equal_mcmc_models_are_equal_and_survive_pickling():
    a = bvar(vol=rb.SV(outliers=True), mcmc=rb.MCMCSettings(chains=2))
    b = rb.BVAR(
        mean=rb.Linear(rb.Minnesota(form="structural")),
        vol=rb.SV(outliers=True, outlier_prior=P.OutlierPrior()),
        mcmc={"chains": 2},
    )
    assert a == b and spec_hash(a) == spec_hash(b) and a.name == b.name
    clone = pickle.loads(pickle.dumps(a))
    assert clone == a and spec_hash(clone) == spec_hash(a)


def test_mcmc_spec_records_the_settings():
    spec = bvar(vol=rb.SV(), mcmc=rb.MCMCSettings(chains=2, seed=5)).spec()
    assert spec["mcmc"]["chains"] == 2 and spec["mcmc"]["seed"] == 5
    assert spec["vol"]["class"] == "SV"
    assert spec["mean"]["prior"]["form"] == "structural"
    json.dumps(spec)


# =========================================================================== fit


@pytest.fixture
def fit_spy(monkeypatch):
    calls = []

    def fit_bvar(model, data, *, seed, init, runtime, plan):
        calls.append(
            {
                "model": model,
                "data": data,
                "seed": seed,
                "init": init,
                "runtime": runtime,
                "plan": plan,
            }
        )
        return "fitted"

    monkeypatch.setattr(api, "_mcmc_entry_point", lambda: fit_bvar)
    return calls


def test_mcmc_fit_forwards_its_arguments(fit_spy, sim_data_stationary):
    model = bvar(vol=rb.SV())
    assert model.fit(sim_data_stationary) == "fitted"
    call = fit_spy[-1]
    assert call["model"] is model and call["data"] is sim_data_stationary
    assert (call["seed"], call["init"], call["plan"]) == (None, None, None)
    assert call["runtime"] == rb.Runtime()  # the default runtime
    runtime, plan, init = rb.Runtime(chunk=50), rb.FitPlan(4), object()
    model.fit(sim_data_stationary, seed=np.int64(7), init=init, runtime=runtime, plan=plan)
    call = fit_spy[-1]
    assert call["seed"] == 7 and type(call["seed"]) is int
    assert call["init"] is init and call["runtime"] is runtime and call["plan"] is plan


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**63},
        {"seed": 1.5},
        {"seed": True},
        {"runtime": {"chunk": 50}},
        {"plan": 4},
    ],
    ids=repr,
)
def test_mcmc_fit_validates_its_keywords(fit_spy, sim_data_stationary, kwargs):
    with pytest.raises(TypeError):
        bvar(vol=rb.SV()).fit(sim_data_stationary, **kwargs)
    assert not fit_spy


@pytest.mark.parametrize(
    "kwargs", [{"init": object()}, {"runtime": rb.Runtime()}, {"plan": rb.FitPlan(4)}], ids=repr
)
def test_analytic_fit_rejects_mcmc_keywords(sim_data_stationary, kwargs):
    with pytest.raises(ValueError, match="apply to MCMC compositions only"):
        rb.BVAR(mean=rb.Linear(rb.Minnesota(hierarchy="fixed"))).fit(sim_data_stationary, **kwargs)


@pytest.mark.parametrize(
    "module", ["rai_bvar.engines.mcmc._no_such_module", "rai_bvar._no_such_package.resolve"]
)
def test_mcmc_fit_raises_until_the_engine_is_available(monkeypatch, sim_data_stationary, module):
    monkeypatch.setattr(api, "_RESOLVE_MODULE", module)
    with pytest.raises(NotImplementedError, match="MCMC engine is not yet available"):
        bvar(vol=rb.SV()).fit(sim_data_stationary)


@pytest.mark.parametrize(
    ("body", "error", "match"),
    [
        ("import wp2_no_such_dependency\n", ModuleNotFoundError, "wp2_no_such_dependency"),
        ("from math import wp2_no_such_name\n", ImportError, "wp2_no_such_name"),
        ("raise RuntimeError('engine failed to load')\n", RuntimeError, "engine failed to load"),
    ],
    ids=["missing-dependency", "import-error", "runtime-error"],
)
def test_a_broken_engine_module_is_not_masked(
    monkeypatch, tmp_path, sim_data_stationary, body, error, match
):
    """Only a missing resolver module means "not yet available"; an existing resolver that fails
    to import (e.g. a missing dependency) raises its own error."""
    package = tmp_path / "wp2_engine_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "resolve.py").write_text(body + "def fit_bvar(*args, **kwargs):\n    return 'x'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(api, "_RESOLVE_MODULE", "wp2_engine_probe.resolve")
    try:
        with pytest.raises(error, match=match) as err:
            bvar(vol=rb.SV()).fit(sim_data_stationary)
        assert not isinstance(err.value, NotImplementedError)
    finally:
        for name in ("wp2_engine_probe.resolve", "wp2_engine_probe"):
            sys.modules.pop(name, None)


def test_a_working_engine_module_is_called(monkeypatch, tmp_path, sim_data_stationary):
    package = tmp_path / "wp2_engine_ok"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "resolve.py").write_text(
        "def fit_bvar(model, data, *, seed, init, runtime, plan):\n    return (model, seed)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(api, "_RESOLVE_MODULE", "wp2_engine_ok.resolve")
    model = bvar(vol=rb.SV())
    try:
        assert model.fit(sim_data_stationary, seed=3) == (model, 3)
    finally:
        for name in ("wp2_engine_ok.resolve", "wp2_engine_ok"):
            sys.modules.pop(name, None)


def test_mcmc_fit_raises_while_the_entry_point_is_a_stub(monkeypatch, sim_data_stationary):
    monkeypatch.setattr(api, "_RESOLVE_MODULE", "rai_bvar.settings")  # a module without fit_bvar
    with pytest.raises(NotImplementedError, match="fit_bvar is missing"):
        bvar(vol=rb.SV()).fit(sim_data_stationary)


def test_entry_point_is_the_wp18_resolver():
    assert api._RESOLVE_MODULE == "rai_bvar.engines.mcmc.resolve"


# =========================================================================== no JAX


def test_mcmc_compositions_are_built_and_described_without_jax():
    code = (
        "import json, sys\n"
        "import rai_bvar as rb\n"
        "from rai_bvar.api import static_parts\n"
        "for vol in (rb.SV(), rb.SV(outliers=True, tails='t'), rb.Constant('iw_equivalent')):\n"
        "    m = rb.BVAR(vol=vol, lags=2)\n"
        "    static_parts(m); json.dumps(m.spec()); m.name; m.axes\n"
        "    assert rb.BVAR.from_spec(json.loads(json.dumps(m.spec()))) == m\n"
        "rb.grid(rb.BVAR, vol=[rb.Constant(), rb.SV()])\n"
        "assert not [m for m in sys.modules if m == 'jax' or m.startswith('jax.')], 'jax imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
