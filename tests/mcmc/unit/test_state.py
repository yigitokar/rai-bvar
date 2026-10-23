"""Engine pytrees (DESIGN §2.5–§2.9): documented leaves, templates, helpers, invariants; and the
kernel contract of ``protocols.py`` (DESIGN §3: KernelBase, overridable names, errors)."""

from __future__ import annotations

import dataclasses
import pickle
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines import mcmc
from rai_bvar.engines.mcmc import protocols as P
from rai_bvar.engines.mcmc import state as S
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.keys import Block, Sub

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

# The documented leaves of every class, in order (DESIGN §2.5–§2.9, plus the draw containers).
DOCUMENTED = {
    S.EngineData: "Y X tmask T_eff hist t_last",
    S.FitCache: "F FF XX G_const FtY eq dummy",
    S.EigenCache: "Q g d_half u1 u2",
    S.FixedFactorCache: "Lc s theta_hat",
    S.CoefPriorArrays: "mean0 prec0 group dummy_w",
    S.ContempPriorArrays: "mean0 prec0 cmask",
    S.ConstantPrior: "shape scale",
    S.SVPrior: "b_mu B_mu a_phi b_phi B_sigma c0 C0 B0 m0 V0 log_c",
    S.TransitoryPrior: "a_p b_p log_nu_w",
    S.EqPrior: "coef contemp vol_core transitory",
    S.SharedPrior: "Mc dummies",
    S.DummyRows: "Y X",
    S.DummyCache: "F G FtY w",
    S.PriorArrays: "eq shared",
    S.RecursiveEq: "a",
    S.LinearEq: "delta",
    S.ConstantEq: "log_lam",
    S.SVEq: "h mu phi sigma acc_h acc_cgp acc_cs2 acc_ncphi acc_ncms acc_fold acc_floor "
    "acc_rej_run acc_rej_run_max rb_h_sum rb_h_sq",
    S.SVRWEq: "h sigma acc_h acc_cs2 acc_ncms acc_fold acc_floor acc_rej_run acc_rej_run_max "
    "rb_h_sum rb_h_sq",
    S.OutlierEq: "o_idx log_p log1m_p rb_out",
    S.TEq: "log_tau nu_idx acc_nu_move",
    S.OutlierTEq: "o_idx log_p log1m_p log_tau nu_idx rb_out acc_nu_move",
    S.VolEq: "core trans",
    S.EqState: "struct mean coef vol",
    S.CCCMGlobal: "Pi",
    S.GlobalState: "mean coef vol struct",
    S.Status: "code it min_pivot",
    S.ChainState: "eq glob status",
    S.EqContext: "i target cols cmask weights log_scale tmask",
    S.GaussianDiag: "mean prec cmask",
    S.GaussianPrec: "K b",
    S.Coupling: "n_active quad",
    S.StepInfo: "ok min_pivot",
    S.EqOut: "e log_lam",
    S.SweepCtx: "data prior cache eq_out",
    S.StructDraw: "a",
    S.LinearDraw: "gamma",
    S.CCCMDraw: "Pi",
    S.GlobDraw: "mean coef vol struct",
    S.ConstantDraw: "log_lam",
    S.SVDraw: "h_T mu phi sigma",
    S.SVRWDraw: "h_T sigma",
    S.OutlierDraw: "log_p log1m_p",
    S.TDraw: "nu",
    S.OutlierTDraw: "log_p log1m_p nu",
    S.VolDraw: "core trans",
    S.ParamDraw: "glob mean struct vol",
    S.PathDraw: "h log_scale2",
}


def static(
    persistent="sv",
    transitory="none",
    mean="linear_structural",
    *,
    rw=False,
    n=3,
    lags=2,
    T_pad=64,
    contemp_scaling="none",
    coef_scaling="none",
    coupled=False,
    nu_grid=tuple(range(3, 41)),
    float32_leaves=frozenset(),
) -> StaticConfig:
    return StaticConfig(
        shapes=Shapes.of(n, lags, 1, T_pad),
        struct=StructStatic(contemp_scaling=contemp_scaling),
        mean=MeanStatic(mean, coef_scaling=coef_scaling),
        coef=CoefPriorStatic(),
        vol=VolStatic(persistent, rw=rw, transitory=transitory, coupled=coupled, nu_grid=nu_grid),
        float32_leaves=float32_leaves,
    )


CONFIGS = {
    "structural_constant": static("constant"),
    "structural_sv": static(),
    "structural_sv_rw": static(rw=True),
    "structural_sv_outlier": static(transitory="outlier"),
    "structural_sv_t": static(transitory="t", nu_grid=(5, 6, 7)),
    "structural_sv_outlier_t": static(transitory="outlier_t"),
    "mode2_constant": static(
        "constant", contemp_scaling="lambda", coef_scaling="lambda", coupled=True
    ),
    "reduced_sv_outlier_t": static(transitory="outlier_t", mean="linear_reduced"),
    "reduced_iw_equivalent": static(
        "constant", mean="linear_reduced", contemp_scaling="lambda", coupled=True
    ),
    "zero_sv_n1": static(mean="zero", n=1),
}


def valid_state(st: StaticConfig, rng: np.random.Generator, T_eff: int) -> S.ChainState:
    """A random state that satisfies every §2.8 invariant."""
    tmpl = S.state_template(st)
    n = st.shapes.n

    def fill(path, leaf):
        name = S.leaf_path(path)
        if name.startswith("status"):
            return leaf
        if leaf.dtype == np.int32:
            if name.endswith("o_idx"):
                out = rng.integers(0, st.vol.outlier_K, leaf.shape).astype(np.int32)
                out[..., T_eff:] = 0
                return out
            if name.endswith("nu_idx"):
                return rng.integers(0, len(st.vol.nu_grid), leaf.shape).astype(np.int32)
            return rng.integers(0, 5, leaf.shape).astype(np.int32)
        out = rng.uniform(0.1, 0.9, leaf.shape)
        if name.endswith("log_tau"):
            out[..., T_eff:] = 0.0
        if name == "eq.struct.a":
            out *= np.arange(n - 1)[None, :] < np.arange(n)[:, None]
        return out

    return jax.tree_util.tree_map_with_path(fill, tmpl)


@pytest.mark.parametrize("cls", list(DOCUMENTED), ids=lambda c: c.__name__)
def test_documented_leaves_flatten_and_unflatten(cls):
    names = DOCUMENTED[cls].split()
    assert [f.name for f in dataclasses.fields(cls)] == names
    obj = cls(*(np.full((2,), float(j)) for j in range(len(names))))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(obj)
    assert [S.leaf_path(p) for p, _ in leaves] == names
    back = jax.tree.unflatten(treedef, [leaf for _, leaf in leaves])
    assert type(back) is cls
    for name in names:
        assert getattr(back, name) is getattr(obj, name)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.__setattr__(names[0], None)


def test_empty_has_no_leaves():
    assert jax.tree.leaves(S.Empty()) == []
    assert S.Empty() == S.Empty() and hash(S.Empty()) == hash(S.Empty())
    assert jax.tree.structure(S.Empty()) == jax.tree.structure(S.Empty())
    cache = S.FitCache.empty()
    assert jax.tree.leaves(cache) == []
    assert all(isinstance(getattr(cache, f.name), S.Empty) for f in dataclasses.fields(cache))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_templates_match_the_static_config(name):
    st = CONFIGS[name]
    n, k, T = st.shapes.n, st.shapes.k, st.shapes.T_pad
    state = S.state_template(st)
    assert state.eq.struct.a.shape == (n, n - 1)
    reduced = st.mean.kind == "linear_reduced"
    structural = st.mean.kind == "linear_structural"
    assert isinstance(state.glob.mean, S.CCCMGlobal) == reduced
    assert isinstance(state.eq.mean, S.LinearEq) == structural
    core = state.eq.vol.core
    expected_core = {"constant": S.ConstantEq, "sv": S.SVRWEq if st.vol.rw else S.SVEq}
    assert type(core) is expected_core[st.vol.persistent]
    if st.vol.persistent == "sv":
        assert core.h.shape == (n, T + 1) and core.rb_h_sum.shape == (n, T + 1)
        assert core.acc_rej_run.dtype == np.int32
    expected_trans = {
        "none": S.Empty,
        "outlier": S.OutlierEq,
        "t": S.TEq,
        "outlier_t": S.OutlierTEq,
    }
    assert type(state.eq.vol.trans) is expected_trans[st.vol.transitory]
    assert state.status.code.dtype == np.int32 and state.status.it.dtype == np.uint32
    S.assert_float64(state)
    prior = S.prior_template(st)
    assert prior.eq.coef.mean0.shape == (n, k) and prior.eq.coef.group.dtype == np.int32
    assert prior.shared.Mc.shape == (k, n - 1)
    assert isinstance(
        prior.eq.vol_core, S.ConstantPrior if st.vol.persistent == "constant" else S.SVPrior
    )
    if st.vol.transitory == "none":
        assert isinstance(prior.eq.transitory, S.Empty)
    else:
        assert prior.eq.transitory.log_nu_w.shape == (n, len(st.vol.nu_grid))
    draw = S.draw_template(st)
    assert isinstance(draw.glob.mean, S.CCCMDraw) == reduced
    assert isinstance(draw.mean, S.LinearDraw) == structural
    assert draw.struct.a.shape == (n, n - 1)
    assert jax.tree.leaves(draw.vol.core)[0].shape == (n,)
    # a valid random state passes every invariant, with or without a chain axis
    T_eff = T - 13
    good = valid_state(st, np.random.default_rng(0), T_eff)
    assert S.check_state_invariants(good, st, T_eff) == []
    chains = jax.tree.map(lambda x: np.stack([x, x]), good)
    assert S.check_state_invariants(chains, st, T_eff) == []
    on_device = jax.tree.map(jnp.asarray, chains)
    assert S.check_state_invariants(on_device, st, T_eff) == []


def test_pytrees_cross_jit_vmap_and_scan():
    st = CONFIGS["reduced_sv_outlier_t"]
    state = valid_state(st, np.random.default_rng(1), 50)
    chains = jax.tree.map(lambda x: jnp.stack([x, x + 0]), state)

    @jax.jit
    def step(s: S.ChainState) -> S.ChainState:
        h = s.eq.vol.core.h + 1.0
        core = dataclasses.replace(s.eq.vol.core, h=h)
        return dataclasses.replace(
            s, eq=dataclasses.replace(s.eq, vol=S.VolEq(core, s.eq.vol.trans))
        )

    out = jax.vmap(step)(chains)
    assert jax.tree.structure(out) == jax.tree.structure(chains)
    np.testing.assert_array_equal(out.eq.vol.core.h, np.asarray(chains.eq.vol.core.h) + 1.0)

    def body(carry, _):
        return step(carry), carry.status.code

    final, codes = jax.lax.scan(body, jax.tree.map(jnp.asarray, state), length=3)
    assert codes.shape == (3,) and codes.dtype == jnp.int32
    assert jax.tree.structure(final) == jax.tree.structure(state)


@pytest.mark.parametrize(
    "name", ["structural_sv_outlier_t", "structural_sv_rw", "structural_constant"]
)
def test_zero_accumulators_touches_exactly_acc_and_rb_leaves(name):
    st = CONFIGS[name]
    state = valid_state(st, np.random.default_rng(2), 40)
    for tree in (state, jax.tree.map(jnp.asarray, state)):
        zeroed = S.zero_accumulators(tree)
        before = jax.tree_util.tree_flatten_with_path(tree)[0]
        after = jax.tree.leaves(zeroed)
        n_acc = 0
        for (path, old), new in zip(before, after, strict=True):
            leaf = S.leaf_path(path).split(".")[-1]
            if leaf.startswith(("acc_", "rb_")):
                n_acc += 1
                assert new.dtype == old.dtype and new.shape == old.shape
                assert not np.any(np.asarray(new)) and np.any(np.asarray(old))
                assert isinstance(new, type(old)) or isinstance(old, jax.Array)
            else:
                assert new is old
        # SVEq: 9 acc_* + 2 rb_*; SVRWEq: 7 + 2; OutlierTEq: rb_out, acc_nu_move; Constant: none
        expected = {"structural_sv_outlier_t": 11 + 2, "structural_sv_rw": 9}.get(name, 0)
        assert n_acc == expected
    jitted = jax.jit(S.zero_accumulators)(jax.tree.map(jnp.asarray, state))
    assert jax.tree.structure(jitted) == jax.tree.structure(state)


def test_assert_float64():
    st = CONFIGS["structural_sv"]
    state = S.state_template(st)
    S.assert_float64(state)
    bad = dataclasses.replace(
        state, status=dataclasses.replace(state.status, min_pivot=np.float32(1.0))
    )
    with pytest.raises(TypeError, match=r"status\.min_pivot: float32"):
        S.assert_float64(bad)
    S.assert_float64(bad, allow=frozenset({"min_pivot"}))  # suffix match
    S.assert_float64(bad, allow=frozenset({"status.min_pivot"}))  # full path
    with pytest.raises(TypeError):
        S.assert_float64(bad, allow=frozenset({"pivot"}))  # not a dotted suffix
    with pytest.raises(TypeError, match="complex64"):
        S.assert_float64({"z": np.zeros(2, np.complex64)})
    S.assert_float64({"i": np.zeros(2, np.int32), "k": jax.random.key(0), "x": 1.0})


def test_tree_where_keeps_structure_and_dtypes():
    a = {"f": jnp.ones(3), "i": jnp.ones(2, jnp.int32), "u": jnp.ones((), jnp.uint32)}
    b = jax.tree.map(jnp.zeros_like, a)
    for pred, want in ((True, a), (False, b)):
        out = jax.jit(S.tree_where)(jnp.asarray(pred), a, b)
        for key in a:
            assert out[key].dtype == a[key].dtype
            np.testing.assert_array_equal(out[key], want[key])
    picked = jax.vmap(S.tree_where)(
        jnp.array([True, False]), *(jax.tree.map(lambda x: jnp.stack([x, x]), t) for t in (a, b))
    )
    np.testing.assert_array_equal(picked["f"], np.array([[1.0] * 3, [0.0] * 3]))


def test_error_codes_round_trip():
    for block in Block:
        for eq in (0, 1, 998, 999):
            code = S.encode_error(block, eq)
            assert code.dtype == jnp.int32 and int(code) == 1 + 1000 * int(block) + eq
            assert S.decode_error(int(code)) == (block, eq)
    traced = jax.jit(lambda i: S.encode_error(Block.EQ_VOL, i))(jnp.int32(7))
    assert traced.dtype == jnp.int32 and S.decode_error(int(traced)) == (Block.EQ_VOL, 7)
    assert S.encode_error(16, 0) == 16001
    with pytest.raises(ValueError):
        S.decode_error(0)
    with pytest.raises(ValueError, match=r"names no keys\.Block"):
        S.decode_error(1 + 1000 * 5)
    with pytest.raises(TypeError):
        S.encode_error(Sub.H, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        S.encode_error(Block.EQ_MEAN, 1000)


def test_views_helpers():
    total = S.Coupling(jnp.float64(3.0), jnp.float64(0.5)) + S.Coupling.zeros()
    assert float(total.n_active) == 3.0 and float(total.quad) == 0.5
    assert total.quad.dtype == jnp.float64
    ok = S.StepInfo.success()
    assert bool(ok.ok) and np.isinf(float(ok.min_pivot))
    merged = ok.merge(S.StepInfo(jnp.asarray(False), jnp.float64(1e-3)))
    assert not bool(merged.ok) and float(merged.min_pivot) == 1e-3
    status = S.Status.initial()
    assert status.code.dtype == jnp.int32 and status.it.dtype == jnp.uint32
    assert status.min_pivot.dtype == jnp.float64 and np.isinf(float(status.min_pivot))
    vmapped = jax.vmap(lambda c: S.Coupling(c, c) + S.Coupling.zeros())(jnp.arange(3.0))
    assert vmapped.quad.shape == (3,)


def _mutate(state: S.ChainState, path: str, fn) -> S.ChainState:
    def go(p, leaf):
        return fn(np.array(leaf)) if S.leaf_path(p) == path else leaf

    return jax.tree_util.tree_map_with_path(go, state)


def _set(value, where=(Ellipsis,)):
    def fn(x):
        x[where] = value
        return x

    return fn


@pytest.mark.parametrize(
    ("path", "fn", "message"),
    [
        ("eq.vol.core.h", lambda x: x.astype(np.float32), "dtype float32"),
        ("eq.vol.core.mu", lambda x: x[:2], "shape"),
        ("eq.vol.core.h", _set(np.nan, (0, 3)), "non-finite"),
        ("eq.vol.core.h", _set(np.inf, (0, 3)), "non-finite"),
        ("eq.vol.core.mu", _set(-np.inf, (1,)), "non-finite"),
        ("status.it", lambda x: x.astype(np.int64), "dtype int64"),  # E7: counters are uint32
        ("status.it", lambda x: x.astype(np.int32), "dtype int32"),
        ("eq.vol.trans.o_idx", lambda x: x.astype(np.int64), "dtype int64"),
        ("eq.vol.trans.nu_idx", lambda x: x.astype(np.uint32), "dtype uint32"),
        ("status.code", lambda x: x.astype(np.int64), "dtype int64"),
        ("status.code", lambda x: np.int32(1 + 1000 * 2 + 1), "EQ_VOL, equation 1"),
        ("eq.struct.a", _set(0.5, (0, 0)), "unit lower triangular"),
        ("eq.vol.trans.o_idx", _set(1, (1, 45)), "padded rows"),
        ("eq.vol.trans.log_tau", _set(0.1, (2, 63)), "padded rows"),
        ("eq.vol.trans.o_idx", _set(20, (1, 3)), "index outside"),
        ("eq.vol.trans.nu_idx", _set(-1, (0,)), "index outside"),
        ("eq.vol.core.sigma", _set(0.0, (1,)), "σ must be > 0"),
        ("eq.vol.core.phi", _set(-1.0, (2,)), "|φ| must be < 1"),
        ("eq.vol.core.acc_h", _set(-1.0, (0,)), "negative accumulator"),
        ("status.min_pivot", lambda x: np.float64(np.nan), "non-finite"),
        ("status.min_pivot", lambda x: np.float64(-np.inf), "non-finite"),
    ],
)
def test_check_state_invariants_detects_violations(path, fn, message):
    st = CONFIGS["structural_sv_outlier_t"]
    T_eff = 40
    good = valid_state(st, np.random.default_rng(3), T_eff)
    problems = S.check_state_invariants(_mutate(good, path, fn), st, T_eff)
    assert any(message in p and (path in p or path == "status.code") for p in problems), problems


def test_check_state_invariants_structure_keys_and_T_eff():
    st = CONFIGS["structural_sv"]
    good = valid_state(st, np.random.default_rng(4), 30)
    assert np.isposinf(good.status.min_pivot)  # the +inf sentinel is accepted
    assert S.check_state_invariants(good, st, 30) == []
    assert "structure" in S.check_state_invariants(good, CONFIGS["structural_constant"], 30)[0]
    assert "outside" in S.check_state_invariants(good, st, 0)[0]
    assert "outside" in S.check_state_invariants(good, st, st.shapes.T_pad + 1)[0]
    keyed = _mutate(good, "status.it", lambda x: x)
    keyed = dataclasses.replace(
        keyed, status=dataclasses.replace(keyed.status, it=jax.random.key(0))
    )
    assert any("keys are never state" in p for p in S.check_state_invariants(keyed, st, 30))
    ragged = jax.tree.map(lambda x: np.stack([x, x]), good)
    ragged = _mutate(ragged, "eq.vol.core.mu", lambda x: np.concatenate([x, x[:1]]))
    assert any("batch axes" in p for p in S.check_state_invariants(ragged, st, 30))
    # declared float32 leaves are exempt from the float64 rule (reserved for Phase 4 bartz)
    st32 = static(float32_leaves=frozenset({"h"}))
    f32 = _mutate(
        valid_state(st32, np.random.default_rng(5), 30),
        "eq.vol.core.h",
        lambda x: x.astype(np.float32),
    )
    assert S.check_state_invariants(f32, st32, 30) == []


# ------------------------------------------------------------------------------ protocols


@dataclasses.dataclass(frozen=True)
class ToyCoef(P.KernelBase):
    """A FixedDiagonal-like kernel: the minimal CoefPrior."""

    static: CoefPriorStatic
    scale: float = 1.0
    varies: ClassVar[bool] = False

    def init_eq(self, key, i, prior_i):
        return S.Empty()

    def moments_eq(self, cs_i, i, prior_i, glob_coef):
        mean0 = self.impl("constant_conditional")(prior_i.coef.mean0)
        return S.GaussianDiag(mean0, prior_i.coef.prec0, jnp.ones_like(mean0))

    def sample_local_eq(self, key, cs_i, coef_i, prior_i, glob_coef):
        return cs_i

    def log_prior_eq(self, cs_i, coef_i, prior_i, glob_coef):
        return jnp.zeros(())

    def global_steps(self):
        return ()

    def constant_conditional(self, x):  # an overridable internal (name from OVERRIDABLE)
        return self.scale * x


def test_overridable_names_are_frozen():
    assert P.KERNEL_SLOTS == {"structure", "mean", "coef", "vol", "global_pre"}
    assert P.OVERRIDABLE == P.KERNEL_SLOTS | P.INTERNALS
    assert P.OVERRIDABLE == {
        "structure", "mean", "coef", "vol", "global_pre", "structural_gamma",
        "linear_conditional", "cccm_core", "contemp_conditional", "indicator_logits",
        "mixture_table", "h_accept", "c_step", "nc_step", "nc_accept", "nc_fold",
        "constant_conditional", "outlier_logits", "p_conditional", "nu_logits", "tau_conditional",
    }  # fmt: skip


def test_kernel_base_impl_and_overrides():
    base = ToyCoef(CoefPriorStatic(), 2.0)  # positional fields stay positional
    assert base == ToyCoef(CoefPriorStatic(), 2.0) and hash(base) == hash(
        ToyCoef(CoefPriorStatic(), 2.0)
    )
    assert base.impl("constant_conditional")(3.0) == 6.0

    def mutant(x):
        return -x

    over = base.with_overrides({"constant_conditional": mutant})
    assert over.impl("constant_conditional") is mutant and over != base
    assert hash(over) == hash(base.with_overrides({"constant_conditional": mutant}))
    assert base.impl("constant_conditional")(3.0) == 6.0  # the original is untouched
    with pytest.raises(ValueError, match="not overridable"):
        base.with_overrides({"vol": mutant})  # a kernel slot belongs to compose, not a kernel
    with pytest.raises(ValueError, match="not overridable"):
        ToyCoef(CoefPriorStatic(), overrides=(("nope", mutant),))
    with pytest.raises(KeyError, match="OVERRIDABLE"):
        base.impl("nope")
    with pytest.raises(NotImplementedError, match="nc_fold"):
        base.impl("nc_fold")
    # kernels are static Python objects: closed over in jitted code, data passed as arguments
    prior = jax.tree.map(jnp.asarray, S.prior_template(CONFIGS["structural_sv"]))
    prior = dataclasses.replace(
        prior,
        eq=dataclasses.replace(
            prior.eq, coef=dataclasses.replace(prior.eq.coef, mean0=prior.eq.coef.mean0 + 1.0)
        ),
    )
    for kernel, want in ((base, 2.0), (over, -1.0)):
        pm = jax.jit(jax.vmap(lambda p, k=kernel: k.moments_eq(S.Empty(), 0, p, S.Empty())))(
            prior.eq
        )
        assert isinstance(pm, S.GaussianDiag) and pm.mean.shape == prior.eq.coef.mean0.shape
        np.testing.assert_array_equal(pm.mean, want)


def test_protocols_are_runtime_checkable():
    kernel = ToyCoef(CoefPriorStatic())
    assert isinstance(kernel, P.CoefPrior)
    assert not isinstance(kernel, P.VolBlock) and not isinstance(kernel, P.MeanBlock)
    for proto in (
        P.CovarianceStructure, P.MeanBlock, P.CoefPrior, P.VolBlock, P.PersistentCore,
        P.TransitoryScale, P.GlobalStep, P.SupportsCollapsedQuadratic,
    ):  # fmt: skip
        assert not isinstance(object(), proto)


def test_batching_order_and_labels():
    assert max(P.Batching.VMAP, P.Batching.MAP) is P.Batching.MAP
    assert [b.label for b in P.Batching] == ["vmap", "map"]


def test_numerical_error_carries_context_and_pickles():
    err = P.MCMCNumericalError(
        "EQ_MEAN", 3, 7, {"diag_ratio": 1e17, "weight_range": (1e-9, 2.0)}, chain=1
    )
    assert isinstance(err, RuntimeError)
    assert (err.block, err.equation, err.iteration, err.chain) == ("EQ_MEAN", 3, 7, 1)
    text = str(err)
    assert "EQ_MEAN" in text and "equation 3" in text and "sweep 7" in text and "diag_ratio" in text
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is P.MCMCNumericalError and str(back) == text
    assert back.detail == err.detail and back.chain == 1
    assert mcmc.MCMCNumericalError is P.MCMCNumericalError  # lazy re-export
    with pytest.raises(AttributeError):
        mcmc.not_a_name  # noqa: B018
