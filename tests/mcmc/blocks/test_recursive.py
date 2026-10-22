"""Recursive covariance structure (DESIGN §3.2, §3.7.1, §13 WP10; N04 §6.2, §9.4; N05 §5.4, §8).

Tier U: context/prior/residual algebra against NumPy; the contemporaneous conditional against a dense
regression with injected noise, padded = unpadded (1e-12); the N04 §9.4 sign check (zero gradient at
the conditional mean, ``−∇² = K_a``; the ``+u`` mutant fails); coupling against the hand formula;
the reduced-form maps of N05 §13.1 item 5 (``B′ = A⁻¹Γ``, ``A⁻¹Λ^{½} = chol(Σ)``, ``B − M =
(Δ − M′)′A⁻ᵀ``, forecast recursion); density-ratio identities of the contemporaneous step against
the independent ``validation.log_joint`` (reduced form, ``iw_equivalent``, SV weights, zero mean).
Tier B: block-conditional SBC of the step and block Geweke ``contemp_only`` (DESIGN §9.2, §9.4).

The volatility block (WP13) and the reduced-form mean (WP11) are replaced by test doubles: the
latent log variance comes from the validation references, and ``ỹ = Y − XΠ`` is formed here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from rai_bvar.components.structure.recursive import Recursive, contemp_prior, slot_mask, unit_lower
from rai_bvar.engines.mcmc.config import StructStatic
from rai_bvar.engines.mcmc.keys import Block, Sub, sub_key
from rai_bvar.engines.mcmc.protocols import CovarianceStructure
from rai_bvar.engines.mcmc.state import (
    ChainState,
    ConstantEq,
    ContempPriorArrays,
    Coupling,
    EngineData,
    EqContext,
    EqPrior,
    EqState,
    GaussianDiag,
    RecursiveEq,
    StructDraw,
    VolEq,
)
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.identities import density_ratio_error
from rai_bvar.validation.joint import JointSpec, hold_frozen, joint_spec, production_sweep
from rai_bvar.validation.log_joint import log_joint, loglik
from rai_bvar.validation.reference import coupling_log_scale, effective_log_var
from rai_bvar.validation.sbc import block_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")
RNG_SEED = 1010


# --------------------------------------------------------------------------- test doubles


class StubVol:
    """VolBlock double (WP13 stands in): the reference latent log variance and coupling scale."""

    def __init__(self, static: Any) -> None:
        self.static = static

    def log_var_eq(self, vs: VolEq, gvol: Any, which: str = "effective") -> jax.Array:
        one = jax.tree.map(lambda x: x[None], vs)
        return effective_log_var(one, self.static, which)[0]

    def log_scale_eq(self, vs: VolEq) -> jax.Array:
        return jnp.asarray(coupling_log_scale(vs.core, self.static), jnp.float64)


def ytil_of(spec: JointSpec, S: Recursive, st: ChainState, data: EngineData) -> jax.Array:
    """``ỹ = Y`` (zero mean) or ``Y − XΠ`` (reduced form; WP11's ``fitted_reduced`` stands in)."""
    f = data.X @ st.glob.mean.Pi if spec.static.mean.kind == "linear_reduced" else None
    return S.pre_residual(st.glob, data, f)


def eq_slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def contemp_phase(
    spec: JointSpec,
    key: jax.Array,
    it: jax.Array,
    st: ChainState,
    data: EngineData,
    overrides: tuple[tuple[str, Any], ...] = (),
):
    """Equation phase of a non-fusing mean with ``g_i = 0`` (DESIGN §4.5 body, ``contemp`` step):
    every ``a_i`` redrawn by ``Recursive.contemp_step``; the other leaves unchanged."""
    S = Recursive.from_static(spec.static)
    if overrides:
        S = S.with_overrides(dict(overrides))
    vol = StubVol(spec.static)
    ytil = ytil_of(spec, S, st, data)
    kit = random.fold_in(key, it)

    def body(i: jax.Array, eq_i: EqState, prior_i: EqPrior) -> RecursiveEq:
        k = random.fold_in(random.fold_in(kit, int(Block.EQ_CONTEMP)), i.astype(jnp.uint32))
        ctx = S.context(i, ytil, eq_i, vol, data, st.glob.vol)
        cm = S.contemp_moments(ctx, prior_i)
        ss, _, _ = S.contemp_step(k, eq_i.struct, ctx, jnp.zeros_like(ctx.target), cm)
        return ss

    n = spec.static.shapes.n
    struct = jax.vmap(body)(jnp.arange(n, dtype=jnp.int32), st.eq, spec.prior.eq)
    return replace(st, eq=replace(st.eq, struct=struct))


def problem(spec: JointSpec, seed: int) -> tuple[ChainState, EngineData]:
    k = random.fold_in(KEY, seed)
    st = spec.prior_sample(random.fold_in(k, 0))
    return st, spec.simulate(random.fold_in(k, 1), st)


# --------------------------------------------------------------------------- synthetic contexts


@dataclass
class Synthetic:
    ytil: np.ndarray  # (T_pad, n), padded rows 0
    w: np.ndarray  # (T_pad,), padded rows 0
    tmask: np.ndarray
    T_eff: int
    prec0: np.ndarray  # (n, n−1)
    mean0: np.ndarray
    cmask: np.ndarray


def synthetic(seed: int, n: int = 5, T_pad: int = 40, T_eff: int = 33) -> Synthetic:
    r = np.random.default_rng(seed)
    tmask = (np.arange(T_pad) < T_eff).astype(float)
    ytil = r.normal(size=(T_pad, n)) * np.array([1.0, 3.0, 0.2, 8.0, 0.5][:n]) * tmask[:, None]
    w = np.exp(r.normal(scale=1.5, size=T_pad)) * tmask
    cmask = (np.arange(n - 1)[None, :] < np.arange(n)[:, None]).astype(float)
    prec0 = np.where(cmask > 0, np.exp(r.normal(size=(n, n - 1))), 1.0)
    mean0 = np.where(cmask > 0, r.normal(scale=0.3, size=(n, n - 1)), 0.0)
    return Synthetic(ytil, w, tmask, T_eff, prec0, mean0, cmask)


def synth_ctx(sy: Synthetic, i: int, log_scale: float = 0.0) -> EqContext:
    n = sy.ytil.shape[1]
    return EqContext(
        i=jnp.asarray(i, jnp.int32),
        target=jnp.asarray(sy.ytil[:, i]),
        cols=jnp.asarray(-sy.ytil[:, : n - 1]),
        cmask=slot_mask(jnp.asarray(i), n),
        weights=jnp.asarray(sy.w),
        log_scale=jnp.asarray(log_scale),
        tmask=jnp.asarray(sy.tmask),
    )


def synth_prior(sy: Synthetic, i: int) -> EqPrior:
    c = ContempPriorArrays(
        mean0=jnp.asarray(sy.mean0[i]),
        prec0=jnp.asarray(sy.prec0[i]),
        cmask=jnp.asarray(sy.cmask[i]),
    )
    return EqPrior(coef=None, contemp=c, vol_core=None, transitory=None)  # type: ignore[arg-type]


def dense_contemp(sy: Synthetic, i: int, g: np.ndarray, scale: float = 1.0):
    """NumPy oracle on the active block: regress ``ỹ_i − g`` on ``−ỹ_{<i}`` (N04 §6.2)."""
    T = sy.T_eff
    Z = -sy.ytil[:T, :i]
    w = sy.w[:T]
    prec = sy.prec0[i, :i] * scale
    K = np.diag(prec) + Z.T @ (w[:, None] * Z)
    b = prec * sy.mean0[i, :i] + Z.T @ (w * (sy.ytil[:T, i] - g[:T]))
    return K, b


# --------------------------------------------------------------------------- construction


def test_protocol_and_construction():
    spec = joint_spec("reduced_iw_equivalent")
    S = Recursive.from_static(spec.static)
    assert isinstance(S, CovarianceStructure)
    assert S.static == spec.static.struct and S.scaled
    assert Recursive().static == StructStatic() and not Recursive().scaled
    assert Recursive.has_contemp and Recursive.supports_reduced_mean
    assert hash(S) == hash(Recursive(StructStatic(contemp_scaling="lambda")))
    with pytest.raises(ValueError, match="not overridable"):
        Recursive(overrides=(("cccm", lambda: None),))


def test_unit_lower_and_masks():
    a = np.arange(1.0, 13.0).reshape(4, 3)
    A = np.asarray(unit_lower(jnp.asarray(a)))
    expect = np.eye(4)
    for i in range(4):
        for j in range(i):
            expect[i, j] = a[i, j]
    np.testing.assert_array_equal(A, expect)
    np.testing.assert_array_equal(np.asarray(slot_mask(jnp.asarray(2), 4)), [1.0, 1.0, 0.0])
    assert np.asarray(slot_mask(jnp.asarray(0), 1)).shape == (0,)


def test_init_pre_residual_and_context():
    spec = joint_spec("reduced_sv_outliers")
    S = Recursive.from_static(spec.static)
    st, data = problem(spec, 0)
    s = spec.static.shapes
    np.testing.assert_array_equal(
        np.asarray(S.init_eq(KEY, jnp.asarray(1), eq_slice(spec.prior.eq, 1)).a), np.zeros(s.n - 1)
    )
    ytil = np.asarray(ytil_of(spec, S, st, data))
    Y, X, m = (np.asarray(v) for v in (data.Y, data.X, data.tmask))
    np.testing.assert_allclose(ytil, (Y - X @ np.asarray(st.glob.mean.Pi)) * m[:, None], atol=1e-13)
    assert np.all(ytil[spec.fixed.T_eff :] == 0.0)
    np.testing.assert_array_equal(np.asarray(S.pre_residual(st.glob, data, None)), Y)
    vol = StubVol(spec.static)
    lv = np.asarray(effective_log_var(st.eq.vol, spec.static))
    for i in range(s.n):
        ctx = S.context(
            jnp.asarray(i, jnp.int32), jnp.asarray(ytil), eq_slice(st.eq, i), vol, data, None
        )
        np.testing.assert_array_equal(np.asarray(ctx.target), ytil[:, i])
        np.testing.assert_array_equal(np.asarray(ctx.cols), -ytil[:, : s.n - 1])
        np.testing.assert_array_equal(np.asarray(ctx.cmask), (np.arange(s.n - 1) < i).astype(float))
        np.testing.assert_allclose(np.asarray(ctx.weights), m * np.exp(-lv[i] * m), rtol=1e-15)
        assert float(ctx.log_scale) == 0.0 and ctx.i.dtype == jnp.int32
    # λ-scaled a prior: log_scale = log λ_i; never −∞/NaN weights from padded rows
    spec2 = joint_spec("reduced_iw_equivalent")
    S2 = Recursive.from_static(spec2.static)
    st2, data2 = problem(spec2, 1)
    eq0 = eq_slice(st2.eq, 2)
    ctx2 = S2.context(
        jnp.asarray(2, jnp.int32),
        ytil_of(spec2, S2, st2, data2),
        eq0,
        StubVol(spec2.static),
        data2,
        None,
    )
    assert float(ctx2.log_scale) == float(eq0.vol.core.log_lam)

    class InfVol(StubVol):
        def log_var_eq(self, vs, gvol, which="effective"):
            lv = super().log_var_eq(vs, gvol, which)
            return jnp.where(data.tmask > 0, lv, -jnp.inf)

    ctx3 = S.context(
        jnp.asarray(1, jnp.int32),
        jnp.asarray(ytil),
        eq_slice(st.eq, 1),
        InfVol(spec.static),
        data,
        None,
    )
    w3 = np.asarray(ctx3.weights)
    assert np.all(np.isfinite(w3)) and np.all(w3[spec.fixed.T_eff :] == 0.0)


def test_contemp_moments_scaling_and_exclusions():
    sy = synthetic(3)
    i = 3
    pr = synth_prior(sy, i)
    Smode1 = Recursive()
    Sscaled = Recursive(StructStatic(contemp_scaling="lambda"))
    ctx = synth_ctx(sy, i, log_scale=0.7)
    cm1 = Smode1.contemp_moments(ctx, pr)
    cm2 = Sscaled.contemp_moments(ctx, pr)
    free = contemp_prior(pr, ctx.cmask)  # λ-free prior (the fast-path caches use it)
    for a_, b_ in zip(jax.tree.leaves(free), jax.tree.leaves(cm1), strict=True):
        np.testing.assert_array_equal(np.asarray(a_), np.asarray(b_))
    act = np.arange(4) < i
    np.testing.assert_array_equal(np.asarray(cm1.prec)[act], sy.prec0[i, act])
    np.testing.assert_allclose(
        np.asarray(cm2.prec)[act], sy.prec0[i, act] * np.exp(-0.7), rtol=1e-15
    )
    for cm in (cm1, cm2):
        np.testing.assert_array_equal(np.asarray(cm.prec)[~act], 1.0)
        np.testing.assert_array_equal(np.asarray(cm.mean)[~act], 0.0)
        np.testing.assert_array_equal(np.asarray(cm.mean)[act], sy.mean0[i, act])
        np.testing.assert_array_equal(np.asarray(cm.cmask), act.astype(float))
    # an exclusion restriction in the prior mask (N05 §7.1) deactivates an otherwise active slot
    excl = replace(pr, contemp=replace(pr.contemp, cmask=pr.contemp.cmask.at[1].set(0.0)))
    cm3 = Smode1.contemp_moments(ctx, excl)
    np.testing.assert_array_equal(np.asarray(cm3.cmask), [1.0, 0.0, 1.0, 0.0])
    assert float(cm3.prec[1]) == 1.0 and float(cm3.mean[1]) == 0.0


# --------------------------------------------------------------------------- conditional and draw


@pytest.mark.parametrize("i", [0, 1, 2, 4])
def test_contemp_conditional_and_draw_vs_dense(i):
    sy = synthetic(7)
    S = Recursive()
    r = np.random.default_rng(11 + i)
    g = r.normal(size=sy.ytil.shape[0]) * sy.tmask
    ctx = synth_ctx(sy, i)
    cm = S.contemp_moments(ctx, synth_prior(sy, i))
    cond = S.contemp_conditional(ctx, jnp.asarray(g), cm)
    K, b = (np.asarray(v) for v in (cond.K, cond.b))
    np.testing.assert_array_equal(K, K.T)
    Kd, bd = dense_contemp(sy, i, g)
    np.testing.assert_allclose(K[:i, :i], Kd, rtol=1e-12, atol=1e-12 * np.abs(Kd).max(initial=1.0))
    np.testing.assert_allclose(b[:i], bd, rtol=1e-12, atol=1e-12 * np.abs(bd).max(initial=1.0))
    np.testing.assert_array_equal(K[i:, i:], np.eye(4 - i))
    np.testing.assert_array_equal(K[:i, i:], 0.0)
    np.testing.assert_array_equal(b[i:], 0.0)
    # injected-noise draw (Sub.Z) = dense K⁻¹b + chol(K)⁻ᵀz on the active block, exact zeros elsewhere
    key = random.fold_in(KEY, 100 + i)
    s_i = RecursiveEq(a=jnp.full((4,), 9.0))
    new, cpl, info = S.contemp_step(key, s_i, ctx, jnp.asarray(g), cm)
    z = np.asarray(random.normal(sub_key(key, Sub.Z), (4,), dtype=jnp.float64))
    a = np.asarray(new.a)
    assert bool(info.ok)
    np.testing.assert_array_equal(a[i:], 0.0)
    core = S.contemp_core(jnp.asarray(z), ctx, jnp.asarray(g), cm)[0].a  # the noise-injected core
    np.testing.assert_array_equal(np.asarray(core), a)
    if i:
        L = np.linalg.cholesky(Kd)
        expect = np.linalg.solve(Kd, bd) + np.linalg.solve(L.T, z[:i])
        np.testing.assert_allclose(a[:i], expect, rtol=1e-10, atol=1e-12)
    assert float(cpl.n_active) == 0.0 and float(cpl.quad) == 0.0


def test_contemp_step_n1_and_failure_as_value():
    S = Recursive()
    ctx = EqContext(
        i=jnp.asarray(0, jnp.int32), target=jnp.ones(8), cols=jnp.zeros((8, 0)), cmask=jnp.zeros(0),
        weights=jnp.ones(8), log_scale=jnp.zeros(()), tmask=jnp.ones(8),
    )  # fmt: skip
    cm = GaussianDiag(mean=jnp.zeros(0), prec=jnp.zeros(0), cmask=jnp.zeros(0))
    s_i = RecursiveEq(a=jnp.zeros(0))
    new, cpl, info = S.contemp_step(KEY, s_i, ctx, jnp.zeros(8), cm)
    assert new.a.shape == (0,) and bool(info.ok) and float(cpl.quad) == 0.0
    np.testing.assert_array_equal(np.asarray(S.residual(new.a, ctx, jnp.zeros(8))), np.ones(8))
    # a non-PD conditional is reported, never raised or jittered (D-18)
    sy = synthetic(2)
    ctx2 = synth_ctx(sy, 3)
    bad = GaussianDiag(mean=jnp.zeros(4), prec=jnp.array([-1e6, 1.0, 1.0, 1.0]), cmask=ctx2.cmask)
    _, _, info2 = S.contemp_step(KEY, RecursiveEq(a=jnp.zeros(4)), ctx2, jnp.zeros(40), bad)
    assert not bool(info2.ok)


def test_padded_equals_unpadded():
    """The draw of equation ``i`` in the ``n − 1`` padded slots equals the draw of the unpadded
    ``i``-slot problem with ``z[:i]`` (1e-12; D-15)."""
    sy = synthetic(13)
    S = Recursive()
    for i in (1, 2, 3, 4):
        ctx = synth_ctx(sy, i)
        cm = S.contemp_moments(ctx, synth_prior(sy, i))
        key = random.fold_in(KEY, 200 + i)
        a = np.asarray(
            S.contemp_step(key, RecursiveEq(a=jnp.zeros(4)), ctx, jnp.zeros(40), cm)[0].a
        )
        small = EqContext(
            i=ctx.i, target=ctx.target, cols=ctx.cols[:, :i], cmask=jnp.ones(i), weights=ctx.weights,
            log_scale=ctx.log_scale, tmask=ctx.tmask,
        )  # fmt: skip
        cm_small = GaussianDiag(mean=cm.mean[:i], prec=cm.prec[:i], cmask=jnp.ones(i))
        z = random.normal(sub_key(key, Sub.Z), (4,), dtype=jnp.float64)
        cond = S.contemp_conditional(small, jnp.zeros(40), cm_small)
        from rai_bvar.engines.mcmc.linalg import precision_draw

        a_small = np.asarray(precision_draw(cond.K, cond.b, z[:i])[0])
        np.testing.assert_allclose(a[:i], a_small, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(a[i:], 0.0)


def test_sign_check_gradient_and_hessian():
    """N04 §9.4: the quadratic ``−½Σ_t w_t e_t² − ½(a − m)′P(a − m)``, ``e = ỹ_i − g + ỹ_{<i}a``, has
    zero gradient at ``K_a⁻¹b_a`` and negative Hessian ``K_a``; the ``+u_{<j}`` variant (GK App. B,
    mutant M14) does not."""
    sy = synthetic(17)
    i = 4
    g = np.random.default_rng(5).normal(size=40) * sy.tmask
    S = Recursive()
    ctx = synth_ctx(sy, i)
    cm = S.contemp_moments(ctx, synth_prior(sy, i))
    yt = jnp.asarray(sy.ytil)

    def objective(a: jax.Array) -> jax.Array:
        e = yt[:, i] - g + yt[:, :i] @ a  # e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i (N05 §3)
        return -0.5 * jnp.sum(ctx.weights * e**2) - 0.5 * jnp.sum(
            cm.prec[:i] * (a - cm.mean[:i]) ** 2
        )

    for conditional in (S.contemp_conditional,):
        cond = conditional(ctx, jnp.asarray(g), cm)
        mu = jnp.linalg.solve(cond.K[:i, :i], cond.b[:i])
        grad = np.asarray(jax.grad(objective)(mu))
        scale = np.abs(np.asarray(cond.b[:i])).max()
        assert np.abs(grad).max() <= 1e-10 * scale
        np.testing.assert_allclose(
            -np.asarray(jax.hessian(objective)(mu)), np.asarray(cond.K[:i, :i]), rtol=1e-12
        )

    def plus_sign(ctx_, g_, cm_):  # mutant M14: regress on +ỹ_{<i}
        return S.contemp_conditional(replace(ctx_, cols=-ctx_.cols), g_, cm_)

    bad = plus_sign(ctx, jnp.asarray(g), cm)
    mu_bad = jnp.linalg.solve(bad.K[:i, :i], bad.b[:i])
    assert np.abs(np.asarray(jax.grad(objective)(mu_bad))).max() > 1e-3
    # the override seam reaches the step
    mutant = S.with_overrides({"contemp_conditional": plus_sign})
    a_ok = mutant.contemp_step(KEY, RecursiveEq(a=jnp.zeros(4)), ctx, jnp.asarray(g), cm)[0].a
    a_ref = S.contemp_step(KEY, RecursiveEq(a=jnp.zeros(4)), ctx, jnp.asarray(g), cm)[0].a
    assert np.abs(np.asarray(a_ok) - np.asarray(a_ref)).max() > 1e-3


def test_coupling_hand_formula():
    """λ-scaled ``a`` prior (``iw_equivalent``, Mode II): ``n_active = i``, ``quad = ½Σ P̃(a − m)²``
    with the λ-free ``P̃`` (DESIGN §4.7); zeros otherwise."""
    sy = synthetic(19)
    i = 3
    log_lam = 0.4
    S = Recursive(StructStatic(contemp_scaling="lambda"))
    ctx = synth_ctx(sy, i, log_scale=log_lam)
    cm = S.contemp_moments(ctx, synth_prior(sy, i))
    new, cpl, info = S.contemp_step(
        random.fold_in(KEY, 7), RecursiveEq(a=jnp.zeros(4)), ctx, jnp.zeros(40), cm
    )
    a = np.asarray(new.a)
    assert bool(info.ok)
    assert float(cpl.n_active) == float(i)
    expect = 0.5 * np.sum(sy.prec0[i, :i] * (a[:i] - sy.mean0[i, :i]) ** 2)
    np.testing.assert_allclose(float(cpl.quad), expect, rtol=1e-13)
    # the scaled conditional uses P̃/λ (dense oracle)
    Kd, bd = dense_contemp(sy, i, np.zeros(40), scale=np.exp(-log_lam))
    cond = S.contemp_conditional(ctx, jnp.zeros(40), cm)
    np.testing.assert_allclose(np.asarray(cond.K)[:i, :i], Kd, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(cond.b)[:i], bd, rtol=1e-11, atol=1e-13)
    assert Coupling.zeros().quad == 0.0


def test_residual_and_jacobian():
    sy = synthetic(23)
    S = Recursive()
    r = np.random.default_rng(3)
    for i in range(5):
        ctx = synth_ctx(sy, i)
        a = np.where(np.arange(4) < i, r.normal(size=4), 0.0)
        g = r.normal(size=40)
        e = np.asarray(S.residual(jnp.asarray(a), ctx, jnp.asarray(g)))
        expect = (sy.ytil[:, i] + sy.ytil[:, :4] @ a - g) * sy.tmask
        np.testing.assert_allclose(e, expect, rtol=1e-13, atol=1e-13)
        assert np.all(e[sy.T_eff :] == 0.0)
    assert float(S.log_jacobian(None, jnp.asarray(30))) == 0.0  # type: ignore[arg-type]
    host = {"a": np.ones(3)}
    assert S.realign(host, 2, 3) is host


# --------------------------------------------------------------------------- maps (N05 §13.1 item 5)


def random_structure(seed: int, n: int = 4, k: int = 9):
    r = np.random.default_rng(seed)
    a = np.where(
        np.arange(n - 1)[None, :] < np.arange(n)[:, None], r.normal(scale=0.8, size=(n, n - 1)), 0.0
    )
    return a, r.normal(size=(n, k)), r.normal(scale=0.7, size=n), r


def test_reduced_form_chol_and_mean_map():
    a, Gam, log_lam, r = random_structure(29)
    n, k = Gam.shape
    S = Recursive()
    sd = StructDraw(a=jnp.asarray(a))
    A = np.asarray(unit_lower(jnp.asarray(a)))
    Ainv = np.linalg.inv(A)  # test-only inverse
    B = np.asarray(S.reduced_form(sd, jnp.asarray(Gam)))
    np.testing.assert_allclose(B, (Ainv @ Gam).T, rtol=1e-12, atol=1e-12)
    C = np.asarray(S.chol(sd, jnp.asarray(log_lam)))
    Sigma = Ainv @ np.diag(np.exp(log_lam)) @ Ainv.T
    np.testing.assert_allclose(C, np.linalg.cholesky(Sigma), rtol=1e-12, atol=1e-12)
    assert np.all(np.triu(C, 1) == 0.0)
    np.testing.assert_array_equal(np.diag(C), np.exp(0.5 * log_lam))
    g, f = r.normal(size=n), r.normal(size=n)
    np.testing.assert_allclose(
        np.asarray(S.mean_map(sd, jnp.asarray(g), jnp.asarray(f))), Ainv @ g + f, rtol=1e-12
    )
    # D3: Γ = Δ + (A − I)M′ ⇒ B − M = (Δ − M′)′A⁻ᵀ (N05 §8)
    Mt = r.normal(size=(n, k))  # M′
    Delta = r.normal(size=(n, k))
    Gam2 = Delta + (A - np.eye(n)) @ Mt
    B2 = np.asarray(S.reduced_form(sd, jnp.asarray(Gam2)))
    np.testing.assert_allclose(B2 - Mt.T, (Delta - Mt).T @ Ainv.T, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(S.extract(RecursiveEq(a=jnp.asarray(a)), None).a), a)  # type: ignore[arg-type]


def test_forecast_recursion_equals_reduced_form_simulation():
    """``y_i = −Σ_{j<i} a_ij y_j + γ_i′x + √λ_i ε_i`` recursively = ``B′x + A⁻¹Λ^{½}ε`` (1e-12)."""
    a, Gam, log_lam, r = random_structure(31)
    n, k = Gam.shape
    S = Recursive()
    sd = StructDraw(a=jnp.asarray(a))
    B = np.asarray(S.reduced_form(sd, jnp.asarray(Gam)))
    C = np.asarray(S.chol(sd, jnp.asarray(log_lam)))
    for _ in range(5):
        x, eps = r.normal(size=k), r.normal(size=n)
        y = np.zeros(n)
        for i in range(n):
            y[i] = -a[i, :i] @ y[:i] + Gam[i] @ x + np.exp(0.5 * log_lam[i]) * eps[i]
        via_map = np.asarray(
            S.mean_map(sd, jnp.asarray(Gam @ x + np.exp(0.5 * log_lam) * eps), jnp.zeros(n))
        )
        np.testing.assert_allclose(via_map, y, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B.T @ x + C @ eps, y, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- density-ratio identities


DENSITY_SPECS = {
    "reduced_constant": {},
    "reduced_sv": {},
    "reduced_iw_equivalent": {},
    "reduced_sv_outliers_t": {},
    "zero_sv": {"n": 3},
}


@pytest.mark.parametrize("name", list(DENSITY_SPECS))
def test_density_ratio_contemp_step(name):
    """``log q(a′) − log q(a) = log p(a′, ·, y) − log p(a, ·, y)`` with the independent
    ``validation.log_joint`` at 20 random states (relative 1e-8; DESIGN §9.3 item 2)."""
    spec = joint_spec(name, **DENSITY_SPECS[name])
    n = spec.static.shapes.n
    S = Recursive.from_static(spec.static)
    vol = StubVol(spec.static)

    @jax.jit
    def run(key: jax.Array):
        st = spec.prior_sample(random.fold_in(key, 0))
        data = spec.simulate(random.fold_in(key, 1), st)
        ytil = ytil_of(spec, S, st, data)
        out = []
        for i in range(1, n):
            eq_i = eq_slice(st.eq, i)
            ctx = S.context(jnp.asarray(i, jnp.int32), ytil, eq_i, vol, data, None)
            cm = S.contemp_moments(ctx, eq_slice(spec.prior.eq, i))
            cond = S.contemp_conditional(ctx, jnp.zeros_like(ctx.target), cm)
            a0 = eq_i.struct.a
            a1 = jnp.where(
                cm.cmask > 0, a0 + 0.4 * random.normal(random.fold_in(key, 2 + i), a0.shape), 0.0
            )

            def log_q(a, cond=cond):
                return -0.5 * a @ cond.K @ a + cond.b @ a

            def with_a(a, i=i):
                return replace(
                    st, eq=replace(st.eq, struct=RecursiveEq(a=st.eq.struct.a.at[i].set(a)))
                )

            out.append(
                jnp.stack([log_q(a0), log_q(a1), log_joint(with_a(a0), data, spec.prior, spec.static),
                           log_joint(with_a(a1), data, spec.prior, spec.static)])
            )  # fmt: skip
        return jnp.stack(out)

    worst = 0.0
    for rep in range(20):
        vals = np.asarray(run(random.fold_in(KEY, 1000 + rep)))
        for lq0, lq1, lj0, lj1 in vals:
            err = density_ratio_error(lambda x: x[0], lambda x: x[1], (lq0, lj0), (lq1, lj1))
            worst = max(worst, err)
    assert worst <= 1e-8, worst


def test_contemp_phase_keeps_other_leaves():
    spec = joint_spec("reduced_sv")
    st, data = problem(spec, 4)
    new = jax.jit(lambda k, s, d: contemp_phase(spec, k, jnp.uint32(3), s, d))(KEY, st, data)
    np.testing.assert_array_equal(np.asarray(new.glob.mean.Pi), np.asarray(st.glob.mean.Pi))
    np.testing.assert_array_equal(np.asarray(new.eq.vol.core.h), np.asarray(st.eq.vol.core.h))
    a = np.asarray(new.eq.struct.a)
    assert np.all(a[np.triu_indices(3, 0, 2)] == 0.0) and np.all(np.isfinite(a))
    assert np.abs(a - np.asarray(st.eq.struct.a)).max() > 0
    # vmap over equations equals a Python loop over equations (1e-14)
    S = Recursive.from_static(spec.static)
    vol = StubVol(spec.static)
    ytil = ytil_of(spec, S, st, data)
    kit = random.fold_in(KEY, jnp.uint32(3))
    for i in range(3):
        k = random.fold_in(random.fold_in(kit, int(Block.EQ_CONTEMP)), jnp.uint32(i))
        eq_i = eq_slice(st.eq, i)
        ctx = S.context(jnp.asarray(i, jnp.int32), ytil, eq_i, vol, data, None)
        cm = S.contemp_moments(ctx, eq_slice(spec.prior.eq, i))
        ai = np.asarray(S.contemp_step(k, eq_i.struct, ctx, jnp.zeros_like(ctx.target), cm)[0].a)
        np.testing.assert_allclose(a[i], ai, rtol=1e-14, atol=1e-15)


# --------------------------------------------------------------------------- tier B and block Geweke


def test_block_sbc_contemp_step():
    """Block-conditional SBC of ``a | Π, λ, y`` (N06 §5.2; DESIGN §9.4): N = 200, L = 99."""
    spec = joint_spec("reduced_sv")
    n = spec.static.shapes.n

    def joint_sample(key):
        st = spec.prior_sample(random.fold_in(key, 0))
        return st, spec.simulate(random.fold_in(key, 1), st)

    def conditional_draws(key, st, data):
        return contemp_phase(spec, key, jnp.uint32(0), st, data)

    def stat(st, data):
        a = st.eq.struct.a
        vals = [a[i, j] for i in range(n) for j in range(i)]
        return jnp.stack([*vals, loglik(st, data, spec.prior, spec.static)])

    res = block_sbc(joint_sample, conditional_draws, stat, N=200, L=99, seed=11, alpha=1e-3)
    assert res.passed, (res.reasons, res.p)


@dataclass(frozen=True, eq=False)
class ContempOnly:
    """``contemp_only`` (DESIGN §9.2): JointSpec's prior, simulator and test functions; the step is
    the equation phase of the reduced form with Π held (WP11's identity ``cccm_core``) and λ held."""

    spec: JointSpec
    overrides: tuple[tuple[str, Any], ...] = ()

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
        new = contemp_phase(self.spec, key, it, st, data, self.overrides)
        return hold_frozen(new, st, self.spec.live)

    def test_functions(self, st, data):
        return self.spec.test_functions(st, data)


@pytest.mark.geweke
def test_block_geweke_contemp_only():
    """Block Geweke ``contemp_only`` at G-small (C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3)."""
    spec = joint_spec("contemp_only")
    assert spec.live == frozenset({"eq.struct"})
    res = geweke(ContempOnly(spec), GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=7))
    assert res.passed, res.table().head(8)


@pytest.mark.geweke
def test_block_geweke_contemp_only_scaled():
    """``contemp_only`` with the λ-scaled ``a`` prior (``reduced_iw_equivalent``, λ held)."""
    spec = joint_spec("reduced_iw_equivalent", live=frozenset({"eq.struct"}))
    res = geweke(ContempOnly(spec), GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=8))
    assert res.passed, res.table().head(8)


def test_contemp_only_spec_shape():
    spec = joint_spec("contemp_only")
    assert spec.static.mean.kind == "linear_reduced" and spec.static.vol.persistent == "constant"
    st, _ = problem(spec, 9)
    assert isinstance(st.eq.vol.core, ConstantEq)


def plus_sign_conditional(ctx, g, cm):
    """Mutant M14: the A-row regression on ``+ỹ_{<i}`` (GK App. B as printed; N04 §6.2)."""
    return Recursive().contemp_conditional(replace(ctx, cols=-ctx.cols), g, cm)


@pytest.mark.geweke
def test_block_geweke_detects_sign_mutant():
    """Power check: the M14 sign mutant through the ``contemp_conditional`` seam fails the same
    block Geweke test (log-likelihood and standardized residual sums move)."""
    spec = joint_spec("contemp_only")
    model = ContempOnly(spec, (("contemp_conditional", plus_sign_conditional),))
    res = geweke(model, GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=9))
    assert not res.passed


@pytest.mark.geweke
def test_block_geweke_contemp_only_production_sweep():
    """``contemp_only`` through ``JointSpec.step`` (production sweep with WP11's identity
    ``cccm_core``; DESIGN §9.2); runs once WP14 lands."""
    spec = joint_spec("contemp_only")
    try:
        production_sweep(spec.static, spec.overrides)
    except NotImplementedError as err:
        pytest.skip(f"production sweep not available yet: {err}")
    res = geweke(spec, GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=10))
    assert res.passed, res.table().head(8)
