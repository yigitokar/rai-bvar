"""Phase 3 contract additions (docs/phase3/DESIGN.md §2; WP26).

* Phase 2 goldens: every Phase 2 composition keeps its state, prior and draw templates (leaf
  paths, shapes, dtypes), and the Phase 2 MCMC models and the DGP study's three arms keep their
  spec hashes and names, as recorded before WP26 (``golden_phase2_contracts.json``).
* The static-configuration rules (11)–(15) of P3 §2.1, the new fields and their validation.
* The new state, prior-array and draw templates, accumulators and invariants (P3 §2.2–§2.3).
* Registry kinds, optional protocols and ``compose`` collecting a volatility block's global steps.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.engines.mcmc import compose as compose_mod
from rai_bvar.engines.mcmc import config as cfg
from rai_bvar.engines.mcmc import protocols as P
from rai_bvar.engines.mcmc import registry
from rai_bvar.engines.mcmc import state as S
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
    validate_composition,
)
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.models import spec_hash
from rai_bvar.validation.joint import BLOCK_CONFIGS, COMPOSITIONS, joint_spec

GOLDEN = json.loads((Path(__file__).with_name("golden_phase2_contracts.json")).read_text())

LIN, RED = "linear_structural", "linear_reduced"
S0, SL = StructStatic(), StructStatic(contemp_scaling="lambda")
HS4 = CoefPriorStatic("horseshoe", n_groups=4)
SV = VolStatic("sv")
COMMON = VolStatic("constant", common=True)
COMMON_C = VolStatic("constant", coupled=True, common=True)


def _leaves(tree):
    return [
        [S.leaf_path(p), list(x.shape), str(x.dtype)]
        for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]
    ]


def make(mean=None, coef=None, vol=None, struct=S0, n=3, lags=2, n_det=1, T_pad=32):
    return StaticConfig(
        shapes=Shapes.of(n, lags, n_det, T_pad),
        struct=struct,
        mean=mean or MeanStatic(LIN),
        coef=coef or CoefPriorStatic(),
        vol=vol or SV,
    )


def valid_state(st: StaticConfig, T_eff: int, seed: int = 0) -> S.ChainState:
    """A random state that satisfies every invariant (Phase 2 rules and P3 §2.2)."""
    rng = np.random.default_rng(seed)
    n = st.shapes.n

    def fill(path, leaf):
        name = S.leaf_path(path)
        if name.startswith("status"):
            return leaf
        if leaf.dtype == np.int32:
            hi = {"glob.coef.a_idx": max(st.coef.a_grid, 1), "eq.coef.gamma": 2}.get(name, 1)
            out = rng.integers(0, hi, leaf.shape).astype(np.int32)
            if name.startswith("eq.coef."):
                out[..., : st.shapes.n_det] = 0
            return out
        out = rng.uniform(0.1, 0.9, leaf.shape)
        if name in ("eq.vol.trans.log_tau", "glob.vol.h"):
            out[..., T_eff:] = 0.0
        if name.startswith("eq.coef.") and st.coef.varies:
            out[..., : st.shapes.n_det] = 0.0
        if name == "eq.struct.a":
            out *= np.arange(n - 1)[None, :] < np.arange(n)[:, None]
        return out

    return jax.tree_util.tree_map_with_path(fill, S.state_template(st))


def _mutate(state: S.ChainState, path: str, fn) -> S.ChainState:
    def go(p, leaf):
        return fn(np.array(leaf)) if S.leaf_path(p) == path else leaf

    return jax.tree_util.tree_map_with_path(go, state)


def _set(value, where=(Ellipsis,)):
    def fn(x):
        x[where] = value
        return x

    return fn


# ---------------------------------------------------------------------------------- goldens


@pytest.mark.parametrize("key", sorted(GOLDEN["templates"]))
def test_phase2_templates_are_unchanged(key):
    name, size = key.split("/")
    st = joint_spec(name, size).static
    got = {
        "state": _leaves(S.state_template(st)),
        "prior": _leaves(S.prior_template(st)),
        "draw": _leaves(S.draw_template(st)),
    }
    assert got == GOLDEN["templates"][key]


def test_the_golden_covers_every_phase2_composition():
    names = {k.split("/")[0] for k in GOLDEN["templates"]}
    assert names == set(COMPOSITIONS) | set(BLOCK_CONFIGS)


def _golden_models():
    M = rb.Minnesota

    def bv(vol, form="auto", lags=1, **kw):
        return rb.BVAR(mean=rb.Linear(prior=M(form=form)), vol=vol, lags=lags, **kw)

    return {
        "dgp-constant-ig": rb.BVAR(vol=rb.Constant("ig"), lags=2),
        "dgp-sv": rb.BVAR(vol=rb.SV(), lags=2),
        "dgp-sv-outliers": rb.BVAR(vol=rb.SV(outliers=True), lags=2),
        "structural-constant-ig": bv(rb.Constant("ig")),
        "structural-mode2": bv(rb.Constant("iw"), "structural"),
        "structural-iw-equivalent": bv(rb.Constant("iw_equivalent"), "structural"),
        "structural-sv": bv(rb.SV(), "structural", lags=4),
        "structural-sv-rw": bv(rb.SV(rw=True)),
        "structural-sv-t": bv(rb.SV(tails="t")),
        "structural-sv-outliers-t": bv(rb.SV(outliers=True, tails="t")),
        "reduced-constant-ig": bv(rb.Constant("ig"), "reduced"),
        "reduced-iw-equivalent": bv(rb.Constant("iw_equivalent"), "reduced"),
        "reduced-sv": bv(rb.SV(), "reduced", lags=2),
        "reduced-sv-rw": bv(rb.SV(rw=True), "reduced", lags=4),
        "structural-sv-dummies": rb.BVAR(
            mean=rb.Linear(prior=M(form="structural", soc=1.0, sur=1.0)), vol=rb.SV()
        ),
        "structural-cross": rb.BVAR(
            mean=rb.Linear(prior=M(form="structural", cross=0.5)), vol=rb.SV()
        ),
        "explicit-budget": rb.BVAR(vol=rb.SV(), mcmc=rb.MCMCSettings(draws=2000, seed=3)),
        "analytic-default": rb.BVAR(),
    }


@pytest.mark.parametrize("key", sorted(GOLDEN["spec_hashes"]))
def test_phase2_spec_hashes_and_names_are_unchanged(key):
    model = _golden_models()[key]
    assert spec_hash(model) == GOLDEN["spec_hashes"][key]["hash"]
    assert model.name == GOLDEN["spec_hashes"][key]["name"]


def test_the_dgp_study_arms_keep_their_registered_hashes():
    # benchmarks/reports/dgp_study_preregistration.md (the parallel DGP session) pins these
    models = _golden_models()
    assert spec_hash(models["dgp-constant-ig"])[:16] == "8aebe5adec44e966"
    assert spec_hash(models["dgp-sv"])[:16] == "b3daa828453542ed"
    assert spec_hash(models["dgp-sv-outliers"])[:16] == "7f0aeb733eb2d07e"


# ---------------------------------------------------------------------------------- static config


def test_new_fields_validate():
    with pytest.raises(TypeError, match="coef_log"):
        MeanStatic(LIN, coef_log=1)
    with pytest.raises(ValueError, match="n_groups"):
        CoefPriorStatic("ssvs")
    with pytest.raises(ValueError, match="n_groups"):
        CoefPriorStatic("fixed_diagonal", n_groups=2)
    with pytest.raises(ValueError, match="a_grid"):
        CoefPriorStatic("horseshoe", n_groups=2, a_grid=5)
    with pytest.raises(ValueError, match="p_beta"):
        CoefPriorStatic("normal_gamma", n_groups=2, p_beta=True)
    CoefPriorStatic("normal_gamma", n_groups=2, a_grid=1000)
    CoefPriorStatic("ssvs", n_groups=2, p_beta=True)
    with pytest.raises(ValueError, match="csv_block"):
        VolStatic("constant", common=True, csv_block=1)
    with pytest.raises(ValueError, match="csv_rho_prior"):
        VolStatic("constant", common=True, csv_rho_prior="uniform")
    with pytest.raises(ValueError, match="csv_sigma2_prior"):
        VolStatic("constant", common=True, csv_sigma2_prior="gamma")


def test_common_volatility_weights_are_not_constant():
    assert VolStatic("constant").constant_weights
    assert not COMMON.constant_weights
    assert CoefPriorStatic().varies is False and HS4.varies is True


@pytest.mark.parametrize(
    ("mean", "coef", "vol", "struct", "message"),
    [
        (MeanStatic(LIN), HS4, SV, S0, "needs MeanStatic\\(coef_log=True\\)"),
        (MeanStatic(LIN, const_path="cholesky_const", coef_log=True), HS4, VolStatic("constant"),
         S0, "needs a fixed prior"),
        (MeanStatic(LIN, "lambda", coef_log=True), HS4, VolStatic("constant", coupled=True), SL,
         "independent prior"),
        (MeanStatic("zero", coef_log=True), HS4, SV, S0, "Linear mean path"),
        (MeanStatic(LIN, coef_log=True), CoefPriorStatic(), SV, S0, "is fixed"),
        (MeanStatic(LIN), CoefPriorStatic(), VolStatic("sv", common=True), S0,
         "needs the Constant persistent core"),
        (MeanStatic(LIN), CoefPriorStatic(), VolStatic("constant", transitory="t", common=True),
         S0, "no transitory scale"),
        (MeanStatic(LIN, const_path="shared_factor"), CoefPriorStatic(), COMMON, S0,
         "shared_factor"),
        (MeanStatic(RED, const_path="shared_factor"), CoefPriorStatic(), COMMON_C, SL,
         "structural Linear path"),
        (MeanStatic(LIN, "lambda", "shared_factor"), CoefPriorStatic(),
         VolStatic("constant", coupled=True), SL, "time-constant weights|shared_factor"),
    ],
)  # fmt: skip
def test_phase3_composition_rules(mean, coef, vol, struct, message):
    with pytest.raises(ValueError, match=message):
        validate_composition(struct, mean, coef, vol)


def test_legal_phase3_compositions():
    for mean, coef, vol, struct in [
        (MeanStatic(LIN, coef_log=True), HS4, SV, S0),
        (MeanStatic(RED, coef_log=True), CoefPriorStatic("ssvs", n_groups=2), SV, S0),
        (MeanStatic(LIN, coef_log=True), CoefPriorStatic("r2d2", n_groups=1, a_grid=5), COMMON, S0),
        (MeanStatic(LIN), CoefPriorStatic(), COMMON, S0),  # Mode I × CommonSV
        (MeanStatic(RED), CoefPriorStatic(), COMMON, S0),  # reduced × CommonSV
        (MeanStatic(LIN, "lambda", "shared_factor"), CoefPriorStatic(), COMMON_C, SL),
        (MeanStatic(LIN, "lambda"), CoefPriorStatic(), COMMON_C, SL),  # weighted path (A8 twin)
        (MeanStatic(LIN, coef_log=True), HS4, VolStatic("constant", coupled=True), SL),  # iw_eq
    ]:
        validate_composition(struct, mean, coef, vol)
        make(mean=mean, coef=coef, vol=vol, struct=struct)


# ---------------------------------------------------------------------------------- templates


GL_KINDS = {
    "horseshoe": ({"log_theta", "log_nu"}, {"log_zeta", "log_tau"}),
    "normal_gamma": ({"log_w"}, {"log_xi", "a_idx"}),
    "r2d2": ({"log_varrho", "log_w"}, {"log_xi", "a_idx"}),
    "dirichlet_laplace": ({"log_theta", "log_varrho"}, {"a_idx"}),
    "ssvs": ({"gamma", "rb_incl"}, {"log_p", "log1m_p"}),
}


@pytest.mark.parametrize("kind", sorted(GL_KINDS))
@pytest.mark.parametrize("mean_kind", [LIN, RED])
def test_global_local_templates(kind, mean_kind):
    st = make(mean=MeanStatic(mean_kind, coef_log=True), coef=CoefPriorStatic(kind, n_groups=4))
    n, k = st.shapes.n, st.shapes.k
    state = dict((S.leaf_path(p), x) for p, x in jax.tree_util.tree_flatten_with_path(
        S.state_template(st))[0])  # fmt: skip
    eq_names, glob_names = GL_KINDS[kind]
    for name in eq_names:
        assert state[f"eq.coef.{name}"].shape == (n, k)
    for name in glob_names:
        assert state[f"glob.coef.{name}"].shape == (4,)
    dev = "eq.mean.dev_log" if mean_kind == LIN else "glob.mean.dev_log"
    assert state[dev].shape == (n, k)
    prior = S.prior_template(st)
    assert prior.eq.coef.gl.mask.shape == (n, k)
    assert isinstance(prior.eq.coef.gl.ssvs, S.SSVSArrays) == (kind == "ssvs")
    assert prior.shared.gl.a_support.shape == (1,)
    assert S.check_state_invariants(valid_state(st, 20), st, 20) == []


def test_common_sv_templates_and_accumulators():
    st = make(vol=COMMON)
    tmpl = S.state_template(st)
    assert isinstance(tmpl.glob.vol, S.CommonSVGlobal)
    assert tmpl.glob.vol.h.shape == (st.shapes.T_pad,)
    assert isinstance(S.prior_template(st).shared.csv, S.CommonSVPrior)
    assert isinstance(S.draw_template(st).glob.vol, S.CommonSVDraw)
    filled = jax.tree.map(lambda x: np.ones_like(x), tmpl)
    zeroed = S.zero_accumulators(filled)
    for name in ("acc_csv_h", "acc_csv_rho", "acc_csv_newton", "rb_ch_sum", "rb_ch_sq"):
        assert np.all(getattr(zeroed.glob.vol, name) == 0)
    for name in ("h", "rho", "log_sigma"):
        assert np.all(getattr(zeroed.glob.vol, name) == 1)
    shared = make(mean=MeanStatic(LIN, "lambda", "shared_factor"), vol=COMMON_C, struct=SL)
    assert S.prior_template(shared).shared.shared_prec.shape == (shared.shapes.q,)


def test_ssvs_accumulator_is_reset():
    st = make(mean=MeanStatic(LIN, coef_log=True), coef=CoefPriorStatic("ssvs", n_groups=2))
    filled = jax.tree.map(lambda x: np.ones_like(x), S.state_template(st))
    zeroed = S.zero_accumulators(filled)
    assert np.all(zeroed.eq.coef.rb_incl == 0) and np.all(zeroed.eq.coef.gamma == 1)


SSVS2 = CoefPriorStatic("ssvs", n_groups=2)
NG5 = CoefPriorStatic("normal_gamma", n_groups=2, a_grid=5)


@pytest.mark.parametrize(
    ("coef", "vol", "path", "fn", "message"),
    [
        (NG5, COMMON, "glob.coef.a_idx", _set(5, (1,)), "a_idx: index outside \\[0, 5\\)"),
        (NG5, COMMON, "glob.coef.a_idx", _set(-1, (0,)), "a_idx"),
        (SSVS2, SV, "eq.coef.gamma", _set(2, (0, 0)), "must be 0 or 1"),
        (NG5, COMMON, "glob.vol.rho", _set(1.0), "\\|ρ\\| must be < 1"),
        (NG5, COMMON, "glob.vol.rho", _set(-1.2), "\\|ρ\\|"),
        (NG5, COMMON, "glob.vol.h", _set(0.5, (25,)), "padded states"),
        (NG5, COMMON, "glob.vol.acc_csv_h", _set(-1.0), "negative accumulator"),
        (NG5, COMMON, "glob.vol.h", _set(np.nan, (3,)), "non-finite"),
        (NG5, COMMON, "eq.mean.dev_log", _set(np.inf, (0, 1)), "non-finite"),
        (NG5, COMMON, "eq.coef.log_w", _set(0.3, (1, 0)), "deterministic slots"),
        (SSVS2, SV, "eq.coef.gamma", _set(1, (0, 0)), "deterministic slots"),
        (SSVS2, SV, "eq.coef.rb_incl", _set(2.0, (2, 0)), "deterministic slots"),
    ],
)  # fmt: skip
def test_phase3_invariants(coef, vol, path, fn, message):
    st = make(mean=MeanStatic(LIN, coef_log=True), coef=coef, vol=vol)
    good = valid_state(st, 20)
    assert S.check_state_invariants(good, st, 20) == []
    problems = S.check_state_invariants(_mutate(good, path, fn), st, 20)
    assert any(path in m and re.search(message, m) for m in problems), problems


def test_new_view_leaves_default_to_empty():
    assert isinstance(S.LinearEq(delta=np.zeros((2, 3))).dev_log, S.Empty)
    assert isinstance(S.CCCMGlobal(Pi=np.zeros((3, 2))).dev_log, S.Empty)
    assert jax.tree.leaves(S.LinearEq(delta=np.zeros((2, 3)))) != []
    assert len(jax.tree.leaves(S.LinearEq(delta=np.zeros((2, 3))))) == 1


# ---------------------------------------------------------------------------------- registry, protocols


def test_registry_kinds_and_owners():
    for kind in (*cfg.GLKinds, "gl_scales"):
        assert registry.KERNELS[kind].startswith("rai_bvar.components.coef.global_local:")
        assert registry.OWNERS[kind] == "WP31"
    assert registry.KERNELS["common_sv"] == "rai_bvar.components.vol.common:CommonSVStep"
    assert registry.OWNERS["common_sv"] == "WP32"


def test_optional_protocols_and_internals():
    assert {"gig_sampler", "cccm_core_log", "gl_a_conditional"} <= P.INTERNALS
    assert hasattr(P, "SupportsGlobalSteps") and hasattr(P, "SupportsGlobalVol")
    assert hasattr(P, "SupportsInitGlobal") and P.GaussianLogDiag is S.GaussianLogDiag


@dataclasses.dataclass(frozen=True)
class _ToyCommonStep(P.KernelBase):
    block: ClassVar[Block] = Block.G_COMMON_SV
    phase: ClassVar[str] = "post"
    reads: ClassVar[frozenset[str]] = frozenset({"eq.vol", "glob.vol"})
    writes: ClassVar[frozenset[str]] = frozenset({"glob.vol"})
    marginal_of_indicators: ClassVar[bool] = True

    def init(self, key, st, data, prior, cache):
        return st

    def sample(self, key, st, ctx, eq_map):
        return st, S.StepInfo.success()

    def log_prior(self, glob, prior):
        return jnp.zeros(())

    def future(self, key, gd, H):
        return S.Empty()


def test_compose_collects_the_volatility_blocks_global_steps():
    st = make(vol=VolStatic("constant"))
    base = compose_mod.compose(st)
    step = _ToyCommonStep()

    class WithGlobal(type(base.vol)):
        def global_steps(self):
            return (step,)

    vol = WithGlobal(**{f.name: getattr(base.vol, f.name) for f in dataclasses.fields(base.vol)})
    prog = compose_mod.compose(st, overrides={"vol": vol})
    assert step in prog.global_post and prog.global_pre == base.global_pre
