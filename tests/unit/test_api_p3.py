"""Phase 3 resolution, engine table and construction errors of ``BVAR`` (P3 DESIGN §7.2–§7.5).

* Every row of the §7.2 table, with the ``auto`` resolution and the static parts it maps to.
* Every §7.3 message raised at construction, verbatim. E-13 (custom ``groups`` shape) is tested
  through ``group_ids`` in ``test_specs_p3``; E-12 (SSVS at fit) and E-16 (conditional forecasts)
  are raised by WP31 and WP34 and tested there.
* Specs, hashes, names and axes (§7.4); budgets (D-51); grid harmonisation (D-53).
* Constructing and describing Phase 3 compositions imports no JAX.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import rai_bvar as rb
from rai_bvar.api import static_parts
from rai_bvar.engines.mcmc import config
from rai_bvar.models import spec_hash

P = rb.sv_priors
GL = [rb.Horseshoe, rb.NormalGamma, rb.R2D2, rb.DirichletLaplace, rb.SSVS]


def mn(form="auto", **kw):
    return rb.Linear(prior=rb.Minnesota(form=form, **kw))


def gl(cls=rb.Horseshoe, **kw):
    return rb.Linear(prior=cls(**kw))


AC = rb.Linear(prior=rb.AsymmetricConjugate())


# =========================================================================== the §7.2 table


def test_row_1_conjugate_constant_iw_is_analytic_and_unchanged():
    for model in (rb.BVAR(), rb.BVAR(mean=mn("conjugate"), vol=rb.Constant("iw"))):
        assert model.engine == "analytic" and model.vol == rb.Constant("iw")
        assert model.mean.prior.form == "conjugate" and model.mcmc is None


def test_row_2_conjugate_common_sv_is_mode_ii_on_the_shared_factor_layout():
    model = rb.BVAR(mean=mn("conjugate"), vol=rb.CommonSV())
    assert model.engine == "mcmc" and model.mean.prior.form == "conjugate"
    parts = static_parts(model)
    assert parts.mean_kind == "linear_structural"  # not a lookup on the form (E96)
    assert (parts.coef_scaling, parts.struct.contemp_scaling) == ("lambda", "lambda")
    assert parts.vol == config.VolStatic(persistent="constant", coupled=True, common=True)
    assert parts.coef == config.CoefPriorStatic() and not parts.coef_log
    # the label predicate is split from the static Mode II predicate: no scaling=conjugate
    assert model.axes["prior"] == "Minnesota(conjugate, eb)" and model.axes["vol"] == "CommonSV"
    # the static parts pass the composition rules with the shared-factor path (rule 14)
    config.validate_composition(
        parts.struct,
        config.MeanStatic(parts.mean_kind, parts.coef_scaling, "shared_factor"),
        parts.coef,
        parts.vol,
    )


E1 = (
    "the conjugate Minnesota prior requires Constant(prior='iw') or CommonSV(); use "
    "form='structural' (the MCMC default) or 'reduced'"
)


@pytest.mark.parametrize(
    "vol",
    [rb.Constant("ig"), rb.Constant("iw_equivalent"), rb.SV(), rb.SV(rw=True, tails="t")],
    ids=lambda v: v.label,
)
def test_row_3_conjugate_with_anything_else_raises_e1(vol):
    with pytest.raises(ValueError) as err:
        rb.BVAR(mean=mn("conjugate"), vol=vol)
    assert str(err.value) == E1


def test_rows_4_to_9_are_unchanged():
    assert static_parts(rb.BVAR(vol=rb.SV())).mean_kind == "linear_structural"
    assert static_parts(rb.BVAR(mean=mn("reduced"), vol=rb.Constant("ig"))).mean_kind == (
        "linear_reduced"
    )
    assert (
        rb.BVAR(mean=mn("structural"), vol=rb.Constant("iw"))
        .axes["prior"]
        .endswith("scaling=conjugate")
    )


@pytest.mark.parametrize("form", ["auto", "structural", "reduced"])
def test_row_10_minnesota_common_sv_runs_the_per_equation_program(form):
    model = rb.BVAR(mean=mn(form), vol=rb.CommonSV(block=7))
    resolved = "structural" if form == "auto" else form  # auto with CommonSV is structural (D3)
    assert model.engine == "mcmc" and model.mean.prior.form == resolved
    parts = static_parts(model)
    assert parts.mean_kind == f"linear_{resolved}"
    assert (parts.coef_scaling, parts.struct.contemp_scaling) == ("none", "none")
    assert parts.vol == config.VolStatic(persistent="constant", common=True, csv_block=7)
    assert not parts.vol.constant_weights


@pytest.mark.parametrize("cls", GL, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("vol", "coupled"),
    [(rb.Constant(), False), (rb.Constant("ig"), False), (rb.Constant("iw_equivalent"), True),
     (rb.SV(), False), (rb.SV(outliers=True, tails="t"), False), (rb.CommonSV(), False)],
    ids=lambda v: getattr(v, "label", str(v)),
)  # fmt: skip
@pytest.mark.parametrize("form", ["auto", "structural", "reduced"])
def test_row_11_global_local_priors_run_on_mcmc(cls, vol, coupled, form):
    model = rb.BVAR(mean=gl(cls, form=form), vol=vol, lags=2)
    resolved = "reduced" if form == "reduced" else "structural"  # auto → structural (D-43)
    assert model.engine == "mcmc" and model.mean.prior.form == resolved
    if isinstance(vol, rb.Constant):
        assert model.vol.prior == ("ig" if vol.prior == "auto" else vol.prior)
    parts = static_parts(model)
    assert parts.mean_kind == f"linear_{resolved}" and parts.coef_log
    assert parts.coef_scaling == "none"
    assert parts.struct.contemp_scaling == ("lambda" if coupled else "none")
    assert parts.coef.kind == cls.kind and parts.coef.varies
    assert parts.coef.n_groups == 1  # G needs n: resolved at fit
    assert static_parts(model, n=3).coef.n_groups == 4  # olcl-lagwise: 2p groups
    assert parts.coef.a_grid == model.mean.prior.a_grid
    assert parts.vol.common == isinstance(vol, rb.CommonSV)


E2 = (
    "Constant('iw') makes the coefficient prior λ-scaled (Mode II, the conjugate twin); "
    "global-local priors are independent priors: use Constant('ig'), Constant('iw_equivalent'), "
    "SV() or CommonSV()"
)


@pytest.mark.parametrize("cls", GL, ids=lambda c: c.__name__)
@pytest.mark.parametrize("form", ["auto", "structural", "reduced"])
def test_row_12_global_local_with_constant_iw_raises_e2(cls, form):
    with pytest.raises(ValueError) as err:
        rb.BVAR(mean=gl(cls, form=form), vol=rb.Constant("iw"))
    assert str(err.value) == E2


@pytest.mark.parametrize("vol", [rb.Constant(), rb.Constant("iw")], ids=["auto", "iw"])
def test_row_13_asymmetric_conjugate_is_analytic(vol):
    model = rb.BVAR(mean=AC, vol=vol)
    assert model.engine == "analytic" and model.vol == rb.Constant("iw") and model.mcmc is None
    assert not model.uses_fit_seed and not model.supports_warm_start
    with pytest.raises(ValueError, match="analytic engine"):
        static_parts(model)


E3 = (  # amended by P3 E141
    "AsymmetricConjugate is conjugate only with Constant('iw') (Constant() resolves to it); for "
    "Constant('ig') or stochastic volatility use Linear(prior=Minnesota(form='structural', "
    "cross=...)) with that volatility, or Minnesota(form='conjugate') with CommonSV()"
)


@pytest.mark.parametrize(
    "vol",
    [rb.Constant("ig"), rb.Constant("iw_equivalent"), rb.SV(), rb.CommonSV()],
    ids=lambda v: v.label,
)
def test_row_14_asymmetric_conjugate_with_anything_else_raises_e3(vol):
    with pytest.raises(ValueError) as err:
        rb.BVAR(mean=AC, vol=vol)
    assert str(err.value) == E3


def test_fitting_an_asymmetric_conjugate_model_needs_structural_nig_fit(monkeypatch):
    # the analytic dispatch reaches StructuralNIGFit by name (E134): with WP33 it exists and
    # validates its data; a build without it says so (no data read)
    from rai_bvar.engines import structural_nig

    with pytest.raises(TypeError, match="data must be a VARData"):
        rb.BVAR(mean=AC).fit(None)
    monkeypatch.delattr(structural_nig, "StructuralNIGFit")
    with pytest.raises(NotImplementedError, match="StructuralNIGFit"):
        rb.BVAR(mean=AC).fit(None)


# =========================================================================== §7.3 messages


E5 = (
    "hierarchy='mh' is not available on the MCMC engine in this version (DESIGN Q15); use 'eb' "
    "or 'fixed', or the analytic engine (Minnesota(form='conjugate') with Constant())"
)


@pytest.mark.parametrize(
    ("form", "vol"),
    [("structural", rb.SV()), ("reduced", rb.Constant("ig")), ("conjugate", rb.CommonSV()),
     ("structural", rb.CommonSV())],
)  # fmt: skip
def test_e5_metropolis_hastings_on_mcmc(form, vol):
    with pytest.raises(NotImplementedError) as err:
        rb.BVAR(mean=mn(form, hierarchy="mh"), vol=vol)
    assert str(err.value) == E5


E6 = (
    "Lenza-Primiceri scaling runs on the analytic engine only (DESIGN Q16); use SV(outliers=True) "
    "for outlier-robust MCMC models"
)


@pytest.mark.parametrize(
    "mean", [mn("structural"), gl(), gl(rb.SSVS, form="reduced")], ids=["mn", "hs", "ssvs"]
)
@pytest.mark.parametrize("vol", [rb.Constant("ig"), rb.SV()], ids=["ig", "SV"])
def test_e6_lenza_primiceri_on_mcmc(mean, vol):
    with pytest.raises(NotImplementedError) as err:
        rb.BVAR(mean=mean, vol=vol, outliers="lenza-primiceri")
    assert str(err.value) == E6


def test_e6_keeps_the_double_treatment_clause_for_sv_outliers():
    with pytest.raises(NotImplementedError) as err:
        rb.BVAR(vol=rb.SV(outliers=True), outliers="lenza-primiceri")
    assert str(err.value) == (
        "Lenza-Primiceri scaling runs on the analytic engine only (DESIGN Q16) (and "
        "SV(outliers=True) already models outliers: a double treatment); use SV(outliers=True) "
        "for outlier-robust MCMC models"
    )


@pytest.mark.parametrize("mean", [mn(), mn("conjugate"), mn("reduced"), gl()], ids=repr)
def test_e7_lenza_primiceri_with_common_sv(mean):
    with pytest.raises(ValueError) as err:
        rb.BVAR(mean=mean, vol=rb.CommonSV(), outliers=rb.LenzaPrimiceri())
    assert str(err.value) == (
        "CommonSV already scales every variance by e^{h_t}; adding Lenza-Primiceri scaling is a "
        "double treatment"
    )


@pytest.mark.parametrize("hierarchy", ["fixed", "eb", "mh"])
def test_e8_lenza_primiceri_with_asymmetric_conjugate(hierarchy):
    prior = rb.AsymmetricConjugate(hierarchy=hierarchy)
    with pytest.raises(ValueError) as err:
        rb.BVAR(mean=rb.Linear(prior), outliers="lenza-primiceri")
    assert str(err.value) == (
        "Lenza-Primiceri scaling is not available with AsymmetricConjugate in this version; use "
        "Minnesota(form='conjugate')"
    )


def test_e4_e9_e10_e11_e14_e15_are_raised_when_the_component_is_built():
    # the component classes raise them (test_specs_p3 checks each text), so a BVAR never sees them
    for build in (
        lambda: rb.Horseshoe(form="conjugate"),  # E-4
        lambda: rb.NormalGamma(scale="lambda"),  # E-9
        lambda: rb.R2D2(b=0.0),  # E-10
        lambda: rb.SSVS(c0=2.0, c1=1.0),  # E-11
        lambda: rb.AsymmetricConjugate(cross=0.0),  # E-15
    ):
        with pytest.raises(ValueError):
            build()
    with pytest.raises(ValueError):
        rb.CommonSV(block=1)  # E-14


def test_unknown_priors_and_components():
    with pytest.raises(TypeError, match="Horseshoe, NormalGamma, R2D2, DirichletLaplace, SSVS"):
        rb.BVAR(mean=rb.Linear(prior=rb.sv_priors.Beta(1, 1)))
    with pytest.raises(NotImplementedError, match="Linear × CommonSV"):
        rb.BVAR(vol=rb.sv_priors.Beta(1, 1))


def test_construction_passes_coef_log_to_the_composition_check(monkeypatch):
    seen = []
    original = config.validate_composition

    def spy(struct, mean, coef, vol):
        seen.append((mean, coef))
        return original(struct, mean, coef, vol)

    monkeypatch.setattr(config, "validate_composition", spy)
    rb.BVAR(mean=gl(rb.NormalGamma), vol=rb.SV())
    rb.BVAR(vol=rb.CommonSV())
    (gl_mean, gl_coef), (csv_mean, csv_coef) = seen
    assert gl_mean.coef_log and gl_coef.kind == "normal_gamma" and gl_coef.a_grid == 1000
    assert not csv_mean.coef_log and csv_coef.kind == "fixed_diagonal"


# =========================================================================== specs, names, budgets


MODELS = {
    "hs": rb.BVAR(mean=gl(), vol=rb.SV(), lags=2),
    "ng-reduced": rb.BVAR(mean=gl(rb.NormalGamma, form="reduced", a=0.3, c=1.0)),
    "r2d2-csv": rb.BVAR(mean=gl(rb.R2D2, groups="global"), vol=rb.CommonSV(block=7)),
    "dl-iweq": rb.BVAR(mean=gl(rb.DirichletLaplace), vol=rb.Constant("iw_equivalent")),
    "ssvs-beta": rb.BVAR(mean=gl(rb.SSVS, p=P.Beta(2.0, 3.0), groups=[[0, 1], [1, 0]])),
    "csv": rb.BVAR(vol=rb.CommonSV(rho_prior=P.Beta(20.0, 1.5), sigma2_prior=P.ScaledChi2(0.1))),
    "conj-csv": rb.BVAR(mean=mn("conjugate", soc=1.0), vol=rb.CommonSV()),
    "ac": rb.BVAR(mean=AC, lags=2),
    "ac-mh": rb.BVAR(mean=rb.Linear(rb.AsymmetricConjugate(hierarchy="mh", cross=0.3))),
}


@pytest.mark.parametrize("model", MODELS.values(), ids=MODELS.keys())
def test_specs_roundtrip_and_hash(model):
    spec = model.spec()
    stored = json.loads(json.dumps(spec, default=str))
    back = rb.BVAR.from_spec(stored)
    assert back == model and spec_hash(back) == spec_hash(model)
    prior = spec["mean"]["prior"]
    assert prior.get("class") == (
        None if isinstance(model.mean.prior, rb.Minnesota) else type(model.mean.prior).__name__
    )
    assert ("mcmc" in spec) == (model.engine == "mcmc")
    if isinstance(model.vol, rb.CommonSV):
        assert spec["vol"]["class"] == "CommonSV"


def test_hashes_differ_across_phase_3_models():
    hashes = {spec_hash(m) for m in MODELS.values()}
    assert len(hashes) == len(MODELS)
    assert spec_hash(rb.BVAR(mean=gl(), vol=rb.SV(), lags=2)) == spec_hash(MODELS["hs"])


def test_unknown_classes_in_specs():
    spec = rb.BVAR(vol=rb.SV()).spec()
    spec["mean"]["prior"]["class"] = "Lasso"
    with pytest.raises(ValueError, match="unknown prior class 'Lasso'"):
        rb.BVAR.from_spec(spec)
    spec = MODELS["csv"].spec()
    spec["vol"]["class"] = "FactorSV"
    with pytest.raises(ValueError, match="unknown volatility class"):
        rb.BVAR.from_spec(spec)


def test_names_and_axes():
    assert MODELS["hs"].name == "BVAR(Horseshoe(structural, olcl-lagwise), SV, p=2)"
    assert MODELS["hs"].axes["prior"] == "Horseshoe(structural, olcl-lagwise)"
    assert MODELS["r2d2-csv"].name == ("BVAR(R2D2(structural, global), CommonSV(block=7), p=1)")
    assert MODELS["r2d2-csv"].axes["vol"] == "CommonSV"
    assert MODELS["dl-iweq"].axes["vol"] == "Constant(iw_equivalent)"
    assert MODELS["conj-csv"].axes["prior"] == "Minnesota(conjugate, eb, SOC)"
    assert MODELS["ac"].name == "BVAR(AsymmetricConjugate(eb, log cross flat on [0.001, 10]), p=2)"
    # CommonSV vs Constant is a one-axis comparison against the analytic engine
    analytic = rb.BVAR(mean=mn("conjugate", soc=1.0))
    differ = {k for k in analytic.axes if analytic.axes[k] != MODELS["conj-csv"].axes[k]}
    assert differ == {"vol"}
    sv = rb.BVAR(vol=rb.SV())
    differ = {k for k in sv.axes if sv.axes[k] != rb.BVAR(vol=rb.CommonSV()).axes[k]}
    assert differ == {"vol"}


def test_budgets_count_common_sv_as_stochastic_volatility():  # D-51
    assert rb.BVAR(vol=rb.CommonSV()).mcmc.draws == 10_000
    assert MODELS["conj-csv"].mcmc.draws == 10_000
    assert MODELS["ng-reduced"].mcmc.draws == 2_000  # GL: the D-35 default of its volatility
    assert MODELS["hs"].mcmc.draws == 10_000
    assert "mcmc=" not in MODELS["r2d2-csv"].name
    explicit = rb.BVAR(vol=rb.CommonSV(), mcmc=rb.MCMCSettings(draws=2_000))
    assert "mcmc=" in explicit.name


def test_capability_flags():
    for key in ("hs", "csv", "conj-csv"):
        assert MODELS[key].uses_fit_seed and MODELS[key].supports_warm_start
    assert not MODELS["ac"].uses_fit_seed


# =========================================================================== harmonisation (D-53)


def arms(grid):
    return [(c.get("mean", rb.Linear()).prior, c.get("vol", rb.Constant())) for c in grid]


def test_a_global_local_arm_moves_auto_arms_to_structural():
    grid = rb.BVAR.harmonise_grid([{}, {"mean": gl()}, {"mean": AC}])
    (p0, v0), (p1, v1), (p2, v2) = arms(grid)
    assert (p0.form, v0) == ("structural", rb.Constant("ig"))
    assert p1 == rb.Horseshoe() and v1 == rb.Constant()  # GL arms are never rewritten
    assert p2 == rb.AsymmetricConjugate() and v2 == rb.Constant()  # nor AC arms


@pytest.mark.parametrize("form", ["auto", "structural", "reduced"])
def test_common_sv_triggers_unless_its_form_is_explicitly_conjugate(form):  # E119
    grid = rb.BVAR.harmonise_grid([{}, {"mean": mn(form), "vol": rb.CommonSV()}])
    assert arms(grid)[0][0].form == "structural" and arms(grid)[0][1] == rb.Constant("ig")
    assert arms(grid)[1][0].form == ("structural" if form == "auto" else form)
    screen = [{}, {"mean": mn("conjugate"), "vol": rb.CommonSV()}]
    assert rb.BVAR.harmonise_grid(screen) == screen  # the explicit conjugate screen
    assert rb.BVAR(**rb.BVAR.harmonise_grid(screen)[0]).engine == "analytic"


def test_asymmetric_conjugate_arms_do_not_trigger():
    grid = [{}, {"mean": AC}, {"mean": AC, "vol": rb.Constant("iw")}]
    assert rb.BVAR.harmonise_grid(grid) == grid


def test_phase_2_harmonisation_is_unchanged():  # P2 §11.4's trigger is kept (E119)
    grid = rb.BVAR.harmonise_grid([{}, {"vol": rb.SV()}])
    assert [p.form for p, _ in arms(grid)] == ["structural", "structural"]
    every_constant = [{"vol": rb.Constant(v)} for v in ("ig", "iw", "iw_equivalent")]
    models = [rb.BVAR(**c) for c in rb.BVAR.harmonise_grid(every_constant)]
    assert [m.engine for m in models] == ["mcmc"] * 3
    assert [m.mean.prior.form for m in models] == ["structural"] * 3
    assert rb.BVAR.harmonise_grid([{}, {"vol": rb.Constant("iw")}]) == [
        {},
        {"vol": rb.Constant("iw")},
    ]


# =========================================================================== import rules


def test_phase_3_specs_import_no_jax():
    code = """
import sys
import rai_bvar as rb
P = rb.sv_priors
models = [
    rb.BVAR(mean=rb.Linear(rb.Horseshoe()), vol=rb.SV()),
    rb.BVAR(mean=rb.Linear(rb.NormalGamma(form="reduced")), vol=rb.CommonSV()),
    rb.BVAR(mean=rb.Linear(rb.Minnesota(form="conjugate")), vol=rb.CommonSV()),
    rb.BVAR(mean=rb.Linear(rb.AsymmetricConjugate())),
]
for m in models:
    m.name, m.axes, m.spec(), rb.BVAR.from_spec(m.spec())
rb.BVAR.harmonise_grid([{"mean": rb.Linear(rb.SSVS())}, {}])
from rai_bvar.api import static_parts
static_parts(models[0], n=3)
assert "jax" not in sys.modules, sorted(m for m in sys.modules if m.startswith("jax"))
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stdout + out.stderr
