"""``ZeroMean`` (DESIGN §3.7.7, §13 WP10): ``g ≡ 0``, no coefficients, non-fusing.

Every protocol method; the DESIGN §4.5 equation body for a non-fusing mean run with the real
``ZeroMean``, ``FixedDiagonal`` and ``Recursive`` kernels (``a_i`` regressed on ``−y_{<i}`` with
``g = 0``, residual ``e_i = y_i + Σ_{j<i} a_ij y_j``); the ``n = 1`` shape of the block-level
stochvol parity (``e = y``, no contemporaneous slots); and a block Geweke test of that body
(``zero_sv`` with ``n = 3``, ``a`` live). WP13's volatility block is replaced by a test double.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from rai_bvar.components.coef.fixed import FixedDiagonal
from rai_bvar.components.mean.zero import ZeroMean
from rai_bvar.components.structure.recursive import Recursive
from rai_bvar.engines.mcmc.config import MeanStatic
from rai_bvar.engines.mcmc.data import base_cache
from rai_bvar.engines.mcmc.keys import Block, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import Batching, MeanBlock
from rai_bvar.engines.mcmc.state import (
    ChainState,
    Coupling,
    Empty,
    EngineData,
    EqState,
    FitCache,
    RecursiveEq,
    StepInfo,
    VolEq,
)
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.joint import JointSpec, hold_frozen, joint_spec
from rai_bvar.validation.log_joint import residuals
from rai_bvar.validation.reference import coupling_log_scale, effective_log_var

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261025, impl="threefry2x32")


class StubVol:
    """VolBlock double (WP13 stands in): the reference latent log variance and coupling scale."""

    def __init__(self, static: Any) -> None:
        self.static = static

    def log_var_eq(self, vs: VolEq, gvol: Any, which: str = "effective") -> jax.Array:
        one = jax.tree.map(lambda x: x[None], vs)
        return effective_log_var(one, self.static, which)[0]

    def log_scale_eq(self, vs: VolEq) -> jax.Array:
        return jnp.asarray(coupling_log_scale(vs.core, self.static), jnp.float64)


def eq_slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def problem(spec: JointSpec, seed: int) -> tuple[ChainState, EngineData]:
    k = random.fold_in(KEY, seed)
    st = spec.prior_sample(random.fold_in(k, 0))
    return st, spec.simulate(random.fold_in(k, 1), st)


def nonfusing_phase(
    spec: JointSpec, key: jax.Array, it: jax.Array, st: ChainState, data: EngineData
):
    """The DESIGN §4.5 equation body for a non-fusing mean (``ZeroMean``): mean step on
    ``target − cols·a``, ``g = fitted_eq``, ``Recursive.contemp_step``, residual, ``coef`` local
    step. Returns the new state (volatility untouched) and the residuals ``e`` (n × T_pad)."""
    static, prior = spec.static, spec.prior
    S, M, C = (
        Recursive.from_static(static),
        ZeroMean.from_static(static),
        FixedDiagonal.from_static(static),
    )
    vol = StubVol(static)
    cache = M.prepare(data, prior, base_cache(data, None, static))
    ytil = S.pre_residual(st.glob, data, M.fitted_reduced(st.glob, data))
    kit = random.fold_in(key, it)

    def body(i, eq_i: EqState, prior_i):
        def k(b: Block) -> jax.Array:
            return random.fold_in(random.fold_in(kit, int(b)), i.astype(jnp.uint32))

        ctx = S.context(i, ytil, eq_i, vol, data, st.glob.vol)
        cm = S.contemp_moments(ctx, prior_i)
        pm = C.moments_eq(eq_i.coef, i, prior_i, st.glob.coef)
        a0 = eq_i.struct.a
        ctx_m = replace(ctx, target=ctx.target - (ctx.cols * ctx.cmask) @ a0)
        ms, a_back, cpl_m, i1 = M.sample_eq(
            k(Block.EQ_MEAN), eq_i.mean, a0, ctx_m, pm, None, Empty(), cache, prior.shared
        )
        g = M.fitted_eq(ms, a0, ctx_m, cache, prior.shared)
        ss, cpl_c, i2 = S.contemp_step(k(Block.EQ_CONTEMP), eq_i.struct, ctx, g, cm)
        e = S.residual(ss.a, ctx, g)
        cs = C.sample_local_eq(
            k(Block.EQ_COEF_LOCAL),
            eq_i.coef,
            M.coefficients_eq(ms, ss.a, prior.shared),
            prior_i,
            st.glob.coef,
        )
        cpl = cpl_m + cpl_c
        ok = i1.merge(i2).ok & jnp.all(a_back == a0)
        return replace(eq_i, struct=ss, mean=ms, coef=cs), e, cpl.n_active + cpl.quad, ok

    eq, e, cpl, ok = jax.vmap(body)(jnp.arange(static.shapes.n, dtype=jnp.int32), st.eq, prior.eq)
    return replace(st, eq=eq), e, cpl, ok


# --------------------------------------------------------------------------- protocol


def test_protocol_and_methods():
    spec = joint_spec("zero_sv", n=3)
    static = spec.static
    M = ZeroMean.from_static(static)
    assert isinstance(M, MeanBlock) and M.n == 3 and M.static == static.mean
    assert not ZeroMean.fuses_contemp and ZeroMean.batching is Batching.VMAP
    assert ZeroMean.float32_leaves == frozenset()
    with pytest.raises(ValueError, match="kind='zero'"):
        ZeroMean(MeanStatic(kind="linear_structural"), n=2)
    with pytest.raises(ValueError, match="positive integer"):
        ZeroMean(MeanStatic(kind="zero"), n=0)
    st, data = problem(spec, 0)
    base = base_cache(data, None, static)
    assert M.prepare(data, spec.prior, base) is base and M.fitted_reduced(st.glob, data) is None
    a = jnp.array([0.3, -0.2])
    T = static.shapes.T_pad
    ctx = Recursive.from_static(static).context(
        jnp.asarray(2, jnp.int32), data.Y, eq_slice(st.eq, 2), StubVol(static), data, None
    )
    ms, a_out = M.init_eq(KEY, a, ctx, None, None, Empty(), base, spec.prior.shared)
    assert isinstance(ms, Empty) and a_out is a
    ms, a_out, cpl, info = M.sample_eq(
        KEY, Empty(), a, ctx, None, None, Empty(), base, spec.prior.shared
    )
    assert isinstance(ms, Empty) and a_out is a and bool(info.ok)
    assert float(cpl.n_active) == 0.0 and float(cpl.quad) == 0.0
    cond = M.conditional_eq(Empty(), a, ctx, None, None, Empty(), base, spec.prior.shared)
    assert cond.K.shape == (0, 0) and cond.b.shape == (0,)
    g = np.asarray(M.fitted_eq(Empty(), a, ctx, base, spec.prior.shared))
    assert g.shape == (T,) and np.all(g == 0.0)
    assert M.coefficients_eq(Empty(), a, spec.prior.shared).shape == (0,)
    assert float(M.log_prior_eq(Empty(), a, None, None, spec.prior.shared)) == 0.0
    assert isinstance(M.extract_eq(Empty(), a, spec.prior.shared), Empty)
    gd = M.extract_global(st.glob)
    assert all(isinstance(getattr(gd, f.name), Empty) for f in dataclasses.fields(gd))
    assert isinstance(M.future_eq(KEY, Empty(), 3), Empty)
    x = jnp.ones(static.shapes.k)
    assert float(M.predict_eq(Empty(), Empty(), x, jnp.asarray(1), Empty())) == 0.0
    f = np.asarray(M.predict_reduced(gd, x, jnp.asarray(1), Empty()))
    assert f.shape == (3,) and np.all(f == 0.0)
    assert hash(M) == hash(ZeroMean.from_static(static))
    # FixedDiagonal passes the (empty) coefficient vector through unchanged
    C = FixedDiagonal.from_static(static)
    assert (
        C.sample_local_eq(
            KEY, Empty(), M.coefficients_eq(Empty(), a, spec.prior.shared), None, Empty()
        )
        == Empty()
    )


# --------------------------------------------------------------------------- the non-fusing body


def test_nonfusing_body_regresses_on_minus_lagged_y():
    """``a_i | y`` with ``g = 0`` is the weighted regression of ``y_i`` on ``−y_{<i}`` (dense oracle
    with the same noise), the residual is ``e_i = y_i + Σ_{j<i} a_ij y_j`` (= the reference
    residuals), and the mean step returns ``a`` unchanged with zero coupling."""
    spec = joint_spec("zero_sv", n=3, T_eff=41)
    st, data = problem(spec, 1)
    it = jnp.uint32(4)
    new, e, cpl, ok = jax.jit(lambda k, s, d: nonfusing_phase(spec, k, it, s, d))(KEY, st, data)
    assert bool(jnp.all(ok)) and np.all(np.asarray(cpl) == 0.0)
    T = 41
    Y = np.asarray(data.Y)[:T]
    lv = np.asarray(effective_log_var(st.eq.vol, spec.static))[:, :T]
    a = np.asarray(new.eq.struct.a)
    kit = random.fold_in(KEY, it)
    for i in range(3):
        assert np.all(a[i, i:] == 0.0)
        if i == 0:
            continue
        Z, w = -Y[:, :i], np.exp(-lv[i])
        P = np.asarray(spec.prior.eq.contemp.prec0)[i, :i]
        m = np.asarray(spec.prior.eq.contemp.mean0)[i, :i]
        K = np.diag(P) + Z.T @ (w[:, None] * Z)
        b = P * m + Z.T @ (w * Y[:, i])
        key = random.fold_in(random.fold_in(kit, int(Block.EQ_CONTEMP)), jnp.uint32(i))
        z = np.asarray(random.normal(sub_key(key, Sub.Z), (2,), dtype=jnp.float64))[:i]
        expect = np.linalg.solve(K, b) + np.linalg.solve(np.linalg.cholesky(K).T, z)
        np.testing.assert_allclose(a[i, :i], expect, rtol=1e-10, atol=1e-12)
    ref = np.asarray(residuals(new, data, spec.prior, spec.static))
    np.testing.assert_allclose(np.asarray(e), ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(e)[:, :T], (Y + Y[:, :2] @ a.T).T, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(new.eq.vol.core.h), np.asarray(st.eq.vol.core.h))


def test_single_equation_parity_shape():
    """``n = 1`` (block-level stochvol parity, DESIGN §9.9): no contemporaneous slots, ``e = y`` on
    real rows and 0 on padded rows, a successful step with zero coupling."""
    spec = joint_spec("zero_sv")
    assert spec.static.shapes.n == 1
    st, data = problem(spec, 2)
    new, e, cpl, ok = nonfusing_phase(spec, KEY, jnp.uint32(0), st, data)
    assert new.eq.struct.a.shape == (1, 0) and bool(ok[0]) and float(cpl[0]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(e)[0], np.asarray(data.Y)[:, 0] * np.asarray(data.tmask)
    )
    assert np.all(np.asarray(e)[0, spec.fixed.T_eff :] == 0.0)
    assert isinstance(StepInfo.success(), StepInfo) and isinstance(Coupling.zeros(), Coupling)
    assert isinstance(FitCache.empty().eq, Empty) and isinstance(new.eq.struct, RecursiveEq)


# --------------------------------------------------------------------------- block Geweke


@dataclass(frozen=True, eq=False)
class ZeroContemp:
    """Zero mean, ``a`` live (volatility held at its prior draw): JointSpec's prior, simulator and
    test functions with the non-fusing equation body as the step."""

    spec: JointSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def names(self) -> tuple[str, ...]:
        return self.spec.names

    def prior_sample(self, key):
        return self.spec.prior_sample(key)

    def simulate(self, key, st):
        return self.spec.simulate(key, st)

    def step(self, key, it, st, data):
        return hold_frozen(nonfusing_phase(self.spec, key, it, st, data)[0], st, self.spec.live)

    def test_functions(self, st, data):
        return self.spec.test_functions(st, data)


@pytest.mark.geweke
def test_block_geweke_zero_mean_contemp():
    """Block Geweke of the non-fusing body with ``ZeroMean`` (``zero_sv``, ``n = 3``, ``a`` live) at
    G-small (C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3)."""
    spec = joint_spec("zero_sv", n=3, live=frozenset({"eq.struct"}))
    res = geweke(ZeroContemp(spec), GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=41))
    assert res.passed, res.table().head(8)
