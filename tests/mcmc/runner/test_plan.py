"""Composition and the plan as data (DESIGN §3.8, §4.1–§4.3, §4.8, §4.9)."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.compose import Program, compose, lax_map_eq
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.plan import build_plan, check_invariants, covers
from rai_bvar.engines.mcmc.protocols import Batching
from rai_bvar.engines.mcmc.runner import make_runner
from rai_bvar.engines.mcmc.sweep import make_sweep, trace_steps
from rai_bvar.validation.joint import BLOCK_CONFIGS, COMPOSITIONS, joint_spec

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _program(static, **kw):
    return compose(static, overrides=toys.toy_overrides(static, **kw))


def _probe(static, program_overrides, prior=None, T_eff=20, seed=0):
    """A one-chain state after a cold start and three sweeps (a 'random state')."""
    prior = toys.toy_prior(static) if prior is None else prior
    data = toys.toy_data(static, T_eff, seed)
    fns = make_runner(static, overrides=program_overrides)
    cache = fns.prepare(data, prior)
    ikeys = keys.init_keys(seed, 11, 1)
    st = fns.init(ikeys, data, prior, cache)
    ckeys = keys.chain_keys(keys.fit_root(seed, 11), 1)
    st = fns.burn(ckeys, st, data, prior, cache, np.uint32(0), np.int32(3))
    one = jax.tree.map(lambda x: x[0], st)
    return (ckeys[0], np.uint32(3), one, data, prior, cache)


# --------------------------------------------------------------------------- plans


def test_structural_plan_order_and_description():
    static = toys.toy_static("linear_structural", "sv")
    plan = build_plan(_program(static))
    assert plan.names == ("equation", "residual", "vol")
    eq = plan.equation
    assert eq[0].block == Block.EQ_MEAN and eq[0].writes == {"eq.mean", "eq.struct.a"}
    assert (
        eq[1].writes == frozenset() and eq[2].writes == {"eq.vol"} and eq[2].block == Block.EQ_VOL
    )
    text = plan.describe()
    for word in (
        "equation",
        "residual",
        "vol",
        "EQ_MEAN(0)",
        "EQ_VOL(2)",
        "s → h",
        "post      (none)",
    ):
        assert word in text


def test_reduced_plan_has_pre_cccm_then_contemp():
    static = toys.toy_static("linear_reduced", "sv", n=3)
    plan = build_plan(_program(static))
    assert plan.names == ("cccm", "contemp", "residual", "vol")
    assert plan.pre[0].block == Block.G_CCCM and plan.pre[0].writes == {"glob.mean.Pi"}
    assert plan.equation[0].block == Block.EQ_CONTEMP


def test_zero_mean_plan_omits_stateless_mean_and_coef_steps():
    static = toys.toy_static("zero", "sv", n=1)
    assert build_plan(_program(static)).names == ("contemp", "residual", "vol")


def test_stateful_non_fusing_mean_gets_a_mean_step():
    static = toys.toy_static("linear_structural", "sv")
    program = _program(static, mean=toys.ToyZeroMean(static.mean, static.shapes))  # template has δ
    assert build_plan(program).names == ("mean", "contemp", "residual", "vol")


@pytest.mark.parametrize("name", COMPOSITIONS + BLOCK_CONFIGS)
def test_plan_invariants_every_composition_with_tree_diff(name):
    """§4.8 for every composition of §9.5 and every block configuration of §9.2 (toy kernels with
    the production class variables), including the write-set tree diff at a random state."""
    spec = joint_spec(name)
    static = dataclasses.replace(spec.static, chunk=5)
    ov = toys.toy_overrides(static)
    program = compose(static, overrides=ov)
    plan = build_plan(program)
    reduced = static.mean.kind == "linear_reduced"
    fused = static.mean.kind == "linear_structural"
    assert ("cccm" in plan.names) == reduced
    assert ("equation" in plan.names) == fused and ("contemp" in plan.names) == (not fused)
    check_invariants(plan, program, probe=_probe(static, ov, T_eff=spec.fixed.T_eff))


_PR_PRODUCTION = ("structural_sv_outliers_t", "reduced_sv", "structural_mode2_constant")


@pytest.mark.parametrize(
    "name",
    [n if n in _PR_PRODUCTION else pytest.param(n, marks=pytest.mark.slow) for n in COMPOSITIONS],
)
def test_plan_invariants_production_kernels(name):
    """The same with the production kernels (WP10–WP13), once they exist: three representative
    compositions per PR (transitory scales, the pre-phase CCCM step, the coupling), every
    composition nightly."""
    static = dataclasses.replace(joint_spec(name).static, chunk=5)
    try:
        program = compose(static)
    except NotImplementedError as err:
        pytest.skip(f"production kernels not available yet: {err}")
    spec = joint_spec(name)
    check_invariants(
        build_plan(program), program, probe=_probe(static, None, spec.prior, spec.fixed.T_eff)
    )


# --------------------------------------------------------------------------- violations


def test_tree_diff_catches_undeclared_write():
    static = toys.toy_static("linear_structural", "constant")
    ov = toys.toy_overrides(static)
    ov["global_pre"] = (toys.LeakyPreStep(),)
    program = compose(static, overrides=ov)
    plan = build_plan(program)
    check_invariants(plan, program)  # declared writes are legal
    with pytest.raises(ValueError, match=r"modified \['eq.struct.a'\] outside"):
        check_invariants(plan, program, probe=_probe(static, ov))


def test_post_step_writing_residual_inputs_is_rejected():
    static = toys.toy_static("linear_structural", "constant")
    ov = toys.toy_overrides(static)
    ov["global_pre"] = (toys.PostMeanStep(),)
    program = compose(static, overrides=ov)
    assert [type(g) for g in program.global_post] == [toys.PostMeanStep]
    plan = build_plan(program)
    with pytest.raises(ValueError, match="item 1") as err:
        check_invariants(plan, program)
    assert "item 3" in str(err.value)
    with pytest.raises(ValueError, match="plan invariants"):
        make_sweep(program)


def test_coupling_without_coupled_core_is_rejected():
    @dataclasses.dataclass(frozen=True)
    class LeakyCoupling(toys.ToyLinear):
        def _coupling(self, theta, ctx, pm, cm):
            from rai_bvar.engines.mcmc.state import Coupling

            return Coupling(jax.numpy.ones(()), jax.numpy.ones(()))

    static = toys.toy_static("linear_structural", "constant")
    ov = toys.toy_overrides(static, mean=LeakyCoupling(static.mean, static.shapes))
    program = compose(static, overrides=ov)
    with pytest.raises(ValueError, match="non-zero Coupling"):
        check_invariants(build_plan(program), program, probe=_probe(static, ov))


def test_global_step_rules_in_compose():
    static = toys.toy_static("linear_structural", "constant")
    ov = toys.toy_overrides(static)
    with pytest.raises(ValueError, match="marginal_of_indicators"):
        compose(static, overrides={**ov, "global_pre": (toys.NotMarginalStep(),)})
    with pytest.raises(ValueError, match="duplicate global step block"):
        compose(static, overrides={**ov, "global_pre": (toys.LeakyPreStep(), toys.LeakyPreStep())})


def test_covers():
    assert covers("eq.vol", "eq.vol.core.h") and covers("eq.vol.core.h", "eq.vol")
    assert covers("glob", "glob.mean") and not covers("eq.mean", "eq.mean_x")


# --------------------------------------------------------------------------- compose


def test_compose_rejects_unknown_overrides_and_attaches_internals():
    static = toys.toy_static("linear_structural", "constant")
    ov = toys.toy_overrides(static)
    with pytest.raises(ValueError, match="not overridable"):
        compose(static, overrides={**ov, "sweep_order": object()})

    def fake(*args):
        return None

    program = compose(static, overrides={**ov, "contemp_conditional": fake})
    assert program.structure.impl("contemp_conditional") is fake
    assert program.mean.impl("linear_conditional") == program.mean.linear_conditional
    assert program.overridden and ("contemp_conditional", fake) in program.overrides


def test_compose_builds_kernels_by_field_name(toy_registry):
    """The production path: registry classes built from the StaticConfig by parameter name,
    lru-cached on the StaticConfig (value equality)."""
    static = toys.toy_static("linear_reduced", "constant", n=3, mode="iw_equivalent")
    program = compose(static)
    assert isinstance(program.structure, toys.ToyRecursive)
    assert program.structure.static is static.struct and program.structure.shapes == static.shapes
    assert isinstance(program.mean, toys.ToyReducedMean) and isinstance(program.vol, toys.ToyVol)
    (cccm,) = program.global_pre
    assert isinstance(cccm, toys.ToyCCCM) and cccm.vol is program.vol and cccm.static == static.mean
    assert compose(dataclasses.replace(static)) is program  # cached by value
    assert program.eq_map is jax.vmap and program.batching == Batching.VMAP
    assert make_runner(static) is make_runner(dataclasses.replace(static))


def test_compose_missing_parameter_is_a_clear_error(toy_registry, monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class NeedsThing(toys.ToyRecursive):
        thing: int = dataclasses.field(default=None)  # type: ignore[assignment]

    @dataclasses.dataclass(frozen=True)
    class Required:
        mystery: int

    from rai_bvar.engines.mcmc import registry

    kinds = {**toys.TOY_KINDS, "recursive": Required}
    monkeypatch.setattr(registry, "load", lambda kind: kinds[kind])
    with pytest.raises(TypeError, match=r"no value for \['mystery'\]"):
        compose(toys.toy_static())


def test_map_batching_matches_vmap():
    static = toys.toy_static("linear_structural", "sv", transitory="outlier", n=3)
    static_map = dataclasses.replace(static, batching="map")
    p_map = _program(static_map)
    assert p_map.batching == Batching.MAP and p_map.eq_map is lax_map_eq
    probe = _probe(static, toys.toy_overrides(static))
    ckey, it, st, data, prior, cache = probe
    fns_v = make_runner(static, overrides=toys.toy_overrides(static))
    fns_m = make_runner(static_map, overrides=toys.toy_overrides(static_map))
    cache_m = fns_m.prepare(data, prior)
    a = make_sweep(fns_v.program)(ckey, it, st, data, prior, cache)
    b = make_sweep(fns_m.program)(ckey, it, st, data, prior, cache_m)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-12)
    assert isinstance(fns_v.program, Program)


def test_trace_steps_reports_each_step():
    static = toys.toy_static("linear_reduced", "sv", n=3)
    ov = toys.toy_overrides(static)
    program = compose(static, overrides=ov)
    traces = trace_steps(program, build_plan(program), *_probe(static, ov))
    mods = {t.step.name: t.modified for t in traces}
    assert mods["cccm"] == {"glob.mean.Pi"}
    assert mods["contemp"] == {"eq.struct.a"}
    assert mods["residual"] == frozenset()
    assert mods["vol"] and all(m.startswith("eq.vol.") for m in mods["vol"])
