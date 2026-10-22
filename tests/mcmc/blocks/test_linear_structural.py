"""Structural Linear mean and FixedDiagonal coefficient prior (DESIGN §3.4, §3.7.2, §13 WP10; N05
§4.1, §5.1, §7.1–7.2, §13.1).

Tier U: N05 §13.1 items 1–4, 6, 7 (precision draw vs dense, γ/δ coordinate invariance, masking with
padded = unpadded at 1e-12, ``n = 1`` dense GLS, the structural log-likelihood identity, scale
equivariance in Modes I and II), the Mode II and ``iw_equivalent`` coupling against hand formulas,
the D3 algebra (``fitted``, residual, ``γ = δ + M_c a``), ``log_prior_eq`` against the references,
the override seams (``structural_gamma``: the M8 "centring dropped" semantics; ``linear_conditional``),
and density-ratio identities of the equation step against the independent ``validation.log_joint``.
Tier B: block-conditional SBC of the equation step and block Geweke ``equation_only`` on every
path (DESIGN §9.2, §9.4), with a power check (the M8 mutant fails).

The volatility block (WP13) is replaced by a test double that reads the reference latent log
variance; the equation phase is the DESIGN §4.5 body restricted to the fused mean step.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from jax import random
from scipy import stats

from rai_bvar.components.coef.fixed import FixedDiagonal
from rai_bvar.components.mean.linear import LinearStructural
from rai_bvar.components.structure.recursive import Recursive
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.data import base_cache, build_engine_data
from rai_bvar.engines.mcmc.keys import Block, Sub, sub_key
from rai_bvar.engines.mcmc.linalg import precision_draw
from rai_bvar.engines.mcmc.protocols import CoefPrior, MeanBlock
from rai_bvar.engines.mcmc.state import (
    ChainState,
    ConstantEq,
    ConstantPrior,
    Empty,
    EngineData,
    EqPrior,
    LinearEq,
    PriorArrays,
    RecursiveEq,
    StructDraw,
    VolEq,
    state_template,
)
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import WindowScales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.identities import density_ratio_error
from rai_bvar.validation.joint import JointSpec, hold_frozen, joint_spec, production_sweep
from rai_bvar.validation.log_joint import log_joint, log_prior_eq, loglik
from rai_bvar.validation.reference import coupling_log_scale, effective_log_var, references_for
from rai_bvar.validation.sbc import block_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261023, impl="threefry2x32")


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


def eq_slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


@dataclass
class Views:
    """Batched (leading ``n``) inputs of the equation step, built as the sweep body builds them."""

    S: Recursive
    M: LinearStructural
    ctx: Any
    pm: Any
    cm: Any
    cache: Any

    def eq(self, i: int):
        cache_i = Empty() if isinstance(self.cache.eq, Empty) else eq_slice(self.cache.eq, i)
        return eq_slice(self.ctx, i), eq_slice(self.pm, i), eq_slice(self.cm, i), cache_i


def views(
    static: StaticConfig, prior: PriorArrays, st: ChainState, data: EngineData, M=None
) -> Views:
    S = Recursive.from_static(static)
    M = LinearStructural.from_static(static) if M is None else M
    C = FixedDiagonal.from_static(static)
    vol = StubVol(static)
    cache = M.prepare(data, prior, base_cache(data, prior.shared.Mc, static))
    ytil = S.pre_residual(st.glob, data, None)

    def one(i, eq_i, prior_i):
        ctx = S.context(i, ytil, eq_i, vol, data, None)
        return ctx, C.moments_eq(eq_i.coef, i, prior_i, None), S.contemp_moments(ctx, prior_i)

    ctx, pm, cm = jax.vmap(one)(jnp.arange(static.shapes.n, dtype=jnp.int32), st.eq, prior.eq)
    return Views(S, M, ctx, pm, cm, cache)


def equation_phase(
    spec: JointSpec,
    key: jax.Array,
    it: jax.Array,
    st: ChainState,
    data: EngineData,
    overrides: tuple[tuple[str, Any], ...] = (),
) -> ChainState:
    """The DESIGN §4.5 equation body for a fusing mean (S1 only): ``prepare`` then every
    ``(δ_i, a_i)`` redrawn by ``LinearStructural.sample_eq``; volatility untouched."""
    static, prior = spec.static, spec.prior
    M = LinearStructural.from_static(static)
    if overrides:
        M = M.with_overrides(dict(overrides))
    v = views(static, prior, st, data, M)
    kit = random.fold_in(key, it)

    def body(i, eq_i, ctx, pm, cm, cache_i):
        k = random.fold_in(random.fold_in(kit, int(Block.EQ_MEAN)), i.astype(jnp.uint32))
        ms, a, _, _ = M.sample_eq(
            k, eq_i.mean, eq_i.struct.a, ctx, pm, cm, cache_i, v.cache, prior.shared
        )
        return ms, RecursiveEq(a=a)

    n = static.shapes.n
    mean, struct = jax.vmap(body)(
        jnp.arange(n, dtype=jnp.int32), st.eq, v.ctx, v.pm, v.cm, v.cache.eq
    )
    return replace(st, eq=replace(st.eq, mean=mean, struct=struct))


def problem(spec: JointSpec, seed: int) -> tuple[ChainState, EngineData]:
    k = random.fold_in(KEY, seed)
    st = spec.prior_sample(random.fold_in(k, 0))
    return st, spec.simulate(random.fold_in(k, 1), st)


def dense_problem(static, prior, st, data, i: int):
    """NumPy oracle of equation ``i`` on its active coordinates (N05 §5.1, δ-coordinates)."""
    T = int(data.T_eff)
    X = np.asarray(data.X)[:T]
    Y = np.asarray(data.Y)[:T]
    Mc = np.asarray(prior.shared.Mc)
    F = np.concatenate([X, X @ Mc[:, :i] - Y[:, :i]], axis=1)
    lv = np.asarray(effective_log_var(st.eq.vol, static))[i, :T]
    w = np.exp(-lv)
    log_lam = float(coupling_log_scale(st.eq.vol.core, static)[i]) if static.vol.coupled else 0.0
    pd_ = np.asarray(prior.eq.coef.prec0)[i] * (
        np.exp(-log_lam) if static.mean.coef_scaling == "lambda" else 1.0
    )
    pa = np.asarray(prior.eq.contemp.prec0)[i, :i] * (
        np.exp(-log_lam) if static.struct.contemp_scaling == "lambda" else 1.0
    )
    prec = np.concatenate([pd_, pa])
    mean = np.concatenate(
        [np.asarray(prior.eq.coef.mean0)[i], np.asarray(prior.eq.contemp.mean0)[i, :i]]
    )
    K = np.diag(prec) + F.T @ (w[:, None] * F)
    b = prec * mean + F.T @ (w * Y[:, i])
    return K, b, F, w, prec, mean, X, Y


def active(n_k: int, i: int, q: int) -> np.ndarray:
    return np.arange(q) < n_k + i


# --------------------------------------------------------------------------- construction


def test_protocols_and_construction():
    spec = joint_spec("structural_mode2_constant")
    M = LinearStructural.from_static(spec.static)
    assert isinstance(M, MeanBlock) and isinstance(FixedDiagonal(), CoefPrior)
    assert M.coef_scaled and M.contemp_scaled and M.n == 2 and M.draw_path == "fixed_factor"
    assert LinearStructural.fuses_contemp and LinearStructural.float32_leaves == frozenset()
    with pytest.raises(ValueError, match="contemp_scaling must be 'lambda'"):
        LinearStructural(spec.static.mean, contemp_scaling="none", n=2)
    with pytest.raises(ValueError, match="linear_structural"):
        LinearStructural(MeanStatic(kind="zero"), contemp_scaling="none", n=2)
    with pytest.raises(ValueError, match="positive integer"):
        LinearStructural(MeanStatic(kind="linear_structural"), contemp_scaling="none", n=0)
    with pytest.raises(TypeError):
        LinearStructural(MeanStatic(kind="linear_structural"))  # type: ignore[call-arg]
    a = LinearStructural(MeanStatic(kind="linear_structural"), contemp_scaling="none", n=3)
    assert a == LinearStructural(MeanStatic(kind="linear_structural"), contemp_scaling="none", n=3)
    assert a != a.with_overrides({"structural_gamma": lambda d, a_, Mc: d})


def test_fixed_diagonal():
    spec = joint_spec("structural_sv")
    C = FixedDiagonal.from_static(spec.static)
    assert C.static == CoefPriorStatic() and not FixedDiagonal.varies and C.global_steps() == ()
    p0 = eq_slice(spec.prior.eq, 1)
    assert isinstance(C.init_eq(KEY, jnp.asarray(1), p0), Empty)
    pm = C.moments_eq(Empty(), jnp.asarray(1), p0, Empty())
    np.testing.assert_array_equal(np.asarray(pm.mean), np.asarray(p0.coef.mean0))
    np.testing.assert_array_equal(np.asarray(pm.prec), np.asarray(p0.coef.prec0))
    np.testing.assert_array_equal(np.asarray(pm.cmask), 1.0)
    assert C.sample_local_eq(KEY, Empty(), jnp.ones(3), p0, Empty()) == Empty()
    assert float(C.log_prior_eq(Empty(), jnp.ones(3), p0, Empty())) == 0.0
    batched = jax.vmap(lambda i, p: C.moments_eq(Empty(), i, p, Empty()))(
        jnp.arange(2), spec.prior.eq
    )
    np.testing.assert_array_equal(np.asarray(batched.prec), np.asarray(spec.prior.eq.coef.prec0))


# --------------------------------------------------------------------------- N05 §13.1 items 1, 3


@pytest.mark.parametrize(
    "name", ["structural_sv", "structural_mode2_constant", "structural_iw_equivalent"]
)
def test_conditional_and_draw_vs_dense(name):
    """Item 1: ``K``, ``b`` and the conditional mean equal the dense NumPy build (1e-10); the draw
    with injected noise equals ``K⁻¹b + chol(K)⁻ᵀz`` (Jacobi scaling is exact, 1e-10)."""
    spec = joint_spec(name, n=3, T_eff=27, const_path="weighted")
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 1)
    v = views(static, prior, st, data)
    k, q = static.shapes.k, static.shapes.q
    for i in range(3):
        ctx, pm, cm, cache_i = v.eq(i)
        cond = v.M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, prior.shared)
        Kd, bd, *_ = dense_problem(static, prior, st, data, i)
        on = active(k, i, q)
        K, b = np.asarray(cond.K), np.asarray(cond.b)
        np.testing.assert_array_equal(K, K.T)
        np.testing.assert_allclose(K[np.ix_(on, on)], Kd, rtol=1e-12, atol=1e-12 * np.abs(Kd).max())
        np.testing.assert_allclose(b[on], bd, rtol=1e-11, atol=1e-12 * np.abs(bd).max())
        np.testing.assert_array_equal(K[np.ix_(~on, ~on)], np.eye(q - k - i))
        np.testing.assert_array_equal(K[np.ix_(on, ~on)], 0.0)
        key = random.fold_in(KEY, 50 + i)
        ms, a, _, info = v.M.sample_eq(key, None, None, ctx, pm, cm, cache_i, v.cache, prior.shared)
        theta = np.concatenate([np.asarray(ms.delta), np.asarray(a)])
        z = np.asarray(random.normal(sub_key(key, Sub.Z), (q,), dtype=jnp.float64))
        L = np.linalg.cholesky(Kd)
        mu = np.linalg.solve(Kd, bd)
        expect = mu + sla.solve_triangular(L.T, z[on], lower=False)
        assert bool(info.ok)
        np.testing.assert_allclose(theta[on], expect, rtol=1e-10, atol=1e-10 * np.abs(expect).max())
        np.testing.assert_array_equal(theta[~on], 0.0)


def test_draw_moments_monte_carlo():
    """Item 1: the sample mean and covariance of 10⁵ draws match ``K⁻¹b`` and ``K⁻¹`` within MC
    error (z-scores ≤ 5 over every entry)."""
    spec = joint_spec("structural_sv", n=2, T_eff=20)
    st, data = problem(spec, 2)
    v = views(spec.static, spec.prior, st, data)
    ctx, pm, cm, cache_i = v.eq(1)
    q = spec.static.shapes.q
    draw = jax.jit(
        jax.vmap(lambda z: v.M.draw_core(z, ctx, pm, cm, cache_i, v.cache, spec.prior.shared)[0])
    )
    N = 100_000
    th = np.asarray(draw(random.normal(KEY, (N, q), dtype=jnp.float64)))
    cond = v.M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, spec.prior.shared)
    Sig = np.linalg.inv(np.asarray(cond.K))  # test-only inverse
    mu = Sig @ np.asarray(cond.b)
    sd = np.sqrt(np.diag(Sig))
    z_mean = (th.mean(0) - mu) / (sd / np.sqrt(N))
    C = np.cov(th, rowvar=False)
    se = np.sqrt((np.outer(np.diag(Sig), np.diag(Sig)) + Sig**2) / N)
    assert np.abs(z_mean).max() < 5.0 and np.abs((C - Sig) / se).max() < 5.0


def test_padded_equals_unpadded():
    """Item 3: the draw of equation ``i`` equals the unpadded ``(k+i)``-slot draw with ``z[:k+i]``
    (1e-12), its mean and covariance too, and masked ``a`` entries are exactly 0 (D-15)."""
    spec = joint_spec("structural_sv", n=4, T_eff=45, const_path="weighted")
    st, data = problem(spec, 3)
    v = views(spec.static, spec.prior, st, data)
    k, q = spec.static.shapes.k, spec.static.shapes.q
    for i in range(4):
        ctx, pm, cm, cache_i = v.eq(i)
        on = active(k, i, q)
        key = random.fold_in(KEY, 70 + i)
        ms, a, _, _ = v.M.sample_eq(
            key, None, None, ctx, pm, cm, cache_i, v.cache, spec.prior.shared
        )
        theta = np.concatenate([np.asarray(ms.delta), np.asarray(a)])
        cond = v.M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, spec.prior.shared)
        Ku, bu = cond.K[np.ix_(on, on)], cond.b[on]
        z = random.normal(sub_key(key, Sub.Z), (q,), dtype=jnp.float64)
        small = np.asarray(precision_draw(Ku, bu, z[: k + i])[0])
        np.testing.assert_allclose(theta[on], small, rtol=1e-12, atol=1e-13)
        np.testing.assert_array_equal(np.asarray(a)[i:], 0.0)
        Sig = np.linalg.inv(np.asarray(cond.K))
        np.testing.assert_allclose(
            Sig[np.ix_(on, on)], np.linalg.inv(np.asarray(Ku)), rtol=1e-12, atol=1e-16
        )


# --------------------------------------------------------------------------- items 2 and 4


def test_coordinate_invariance():
    """Item 2: γ-coordinates ``[X | −Y_{<i}]`` with the joint D3 prior of N05 §4.1 give the same
    posterior mean and covariance of ``(γ_i, a_i)`` as our δ-coordinates mapped by ``γ = δ + M_c a``
    (1e-10)."""
    spec = joint_spec("structural_sv", n=3, T_eff=30)
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 4)
    v = views(static, prior, st, data)
    k = static.shapes.k
    Mc = np.asarray(prior.shared.Mc)
    for i in (1, 2):
        ctx, pm, cm, cache_i = v.eq(i)
        cond = v.M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, prior.shared)
        on = active(k, i, static.shapes.q)
        Kd = np.asarray(cond.K)[np.ix_(on, on)]
        Sig = np.linalg.inv(Kd)
        mu = Sig @ np.asarray(cond.b)[on]
        Tm = np.eye(k + i)
        Tm[:k, k:] = Mc[:, :i]
        mu_g, Sig_g = Tm @ mu, Tm @ Sig @ Tm.T
        # dense γ-coordinates
        _, _, _, w, prec, mean, X, Y = dense_problem(static, prior, st, data, i)
        Vg_inv = np.diag(prec[:k])
        Va_inv = np.diag(prec[k:])
        Mi = Mc[:, :i]
        P = np.block([[Vg_inv, -Vg_inv @ Mi], [-Mi.T @ Vg_inv, Va_inv + Mi.T @ Vg_inv @ Mi]])
        Pmu = np.concatenate([Vg_inv @ mean[:k], Va_inv @ mean[k:] - Mi.T @ Vg_inv @ mean[:k]])
        Xt = np.concatenate([X, -Y[:, :i]], axis=1)
        Kg = P + Xt.T @ (w[:, None] * Xt)
        bg = Pmu + Xt.T @ (w * Y[:, i])
        Sg = np.linalg.inv(Kg)
        np.testing.assert_allclose(mu_g, Sg @ bg, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Sig_g, Sg, rtol=1e-10, atol=1e-12 * np.abs(Sg).max())


def test_single_equation_dense_gls():
    """Item 4: ``n = 1`` (no ``a`` slots): ``(P₀ + X′WX)⁻¹(P₀μ₀ + X′Wy)`` (1e-10), weighted and
    Constant paths."""
    for name, path in (("structural_sv", "weighted"), ("structural_constant", "eigen")):
        spec = joint_spec(name, n=1, T_eff=23, const_path=path)
        static, prior = spec.static, spec.prior
        assert static.shapes.q == static.shapes.k
        st, data = problem(spec, 5)
        v = views(static, prior, st, data)
        ctx, pm, cm, cache_i = v.eq(0)
        T = 23
        X, y = np.asarray(data.X)[:T], np.asarray(data.Y)[:T, 0]
        w = np.exp(-np.asarray(effective_log_var(st.eq.vol, static))[0, :T])
        P0, mu0 = np.asarray(prior.eq.coef.prec0)[0], np.asarray(prior.eq.coef.mean0)[0]
        gls = np.linalg.solve(np.diag(P0) + X.T @ (w[:, None] * X), P0 * mu0 + X.T @ (w * y))
        mean = np.asarray(
            v.M.draw_core(jnp.zeros(static.shapes.q), ctx, pm, cm, cache_i, v.cache, prior.shared)[
                0
            ]
        )
        np.testing.assert_allclose(mean, gls, rtol=1e-10, atol=1e-12)
        ms, a, _cpl, info = v.M.sample_eq(
            KEY, None, None, ctx, pm, cm, cache_i, v.cache, prior.shared
        )
        assert a.shape == (0,) and bool(info.ok) and ms.delta.shape == (static.shapes.k,)


# --------------------------------------------------------------------------- item 6: log-likelihood


def test_structural_loglik_equals_reduced_form_density():
    """Item 6: ``Σ_iΣ_t log N(e_it; 0, λ_it)`` with the kernel residuals equals
    ``Σ_t log N(y_t; B′x_t, Σ_t)``, ``Σ_t = A⁻¹Λ_tA⁻ᵀ`` (1e-10)."""
    spec = joint_spec("structural_sv_outliers", n=3, T_eff=25)
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 6)
    v = views(static, prior, st, data)
    lv = np.asarray(effective_log_var(st.eq.vol, static))
    T = 25
    total = 0.0
    gam = []
    for i in range(3):
        ctx, *_ = v.eq(i)
        eq_i = eq_slice(st.eq, i)
        g = v.M.fitted_eq(eq_i.mean, eq_i.struct.a, ctx, v.cache, prior.shared)
        e = np.asarray(v.S.residual(eq_i.struct.a, ctx, g))
        assert np.all(e[T:] == 0.0)
        total += np.sum(stats.norm.logpdf(e[:T], scale=np.exp(0.5 * lv[i, :T])))
        gam.append(np.asarray(v.M.extract_eq(eq_i.mean, eq_i.struct.a, prior.shared).gamma))
    sd = StructDraw(a=st.eq.struct.a)
    B = np.asarray(v.S.reduced_form(sd, jnp.asarray(np.stack(gam))))
    X, Y = np.asarray(data.X)[:T], np.asarray(data.Y)[:T]
    ref = 0.0
    for t in range(T):
        C = np.asarray(v.S.chol(sd, jnp.asarray(lv[:, t])))
        ref += stats.multivariate_normal.logpdf(Y[t], mean=B.T @ X[t], cov=C @ C.T)
    np.testing.assert_allclose(total, ref, rtol=1e-10)
    np.testing.assert_allclose(total, float(loglik(st, data, prior, static)), rtol=1e-10)


# --------------------------------------------------------------------------- item 7: equivariance


def equivariance_setup(mode: str, path: str, c: np.ndarray):
    n, p, T_eff = 3, 2, 37
    T_pad = 64
    r = np.random.default_rng(77)
    psi = np.array([0.8, 1.6, 0.4])
    own = np.array([0.5, 0.0, 0.9])
    mn = Minnesota(lambda_=0.3, cross=0.5, var_const=10.0, contemp=1.5, form="structural")
    scaled = mode in ("II", "iw_equivalent")
    static = StaticConfig(
        shapes=Shapes.of(n, p, 1, T_pad),
        struct=StructStatic(contemp_scaling="lambda" if scaled else "none"),
        mean=MeanStatic(
            kind="linear_structural",
            coef_scaling="lambda" if mode == "II" else "none",
            const_path=path,
        ),
        coef=CoefPriorStatic(),
        vol=VolStatic(persistent="constant", coupled=scaled),
    )
    y = np.zeros((T_pad, n))
    pre = r.normal(size=(p, n))
    for t in range(T_eff):
        prev = y[t - 1] if t else pre[-1]
        y[t] = 0.3 + own * prev + r.normal(size=n) * np.sqrt(psi)
    log_lam = np.log(psi) + r.normal(scale=0.2, size=n)

    def build(cs: np.ndarray):
        sc = WindowScales(psi=psi * cs**2, own_means=own, freq="Q")
        coef, shared = structural_arrays(
            sc, mn, {}, n=n, lags=p, n_det=1, mode="II" if mode == "II" else "I"
        )
        contemp = contemp_arrays(sc, mn, n, scaling="lambda" if scaled else "none")
        prior = PriorArrays(
            eq=EqPrior(coef=coef, contemp=contemp, vol_core=ConstantPrior(shape=np.full(n, 3.0), scale=psi * cs**2),
                       transitory=Empty()),
            shared=shared,
        )  # fmt: skip
        prior = jax.tree.map(jnp.asarray, prior)
        data = build_engine_data(
            jnp.asarray(y * cs), jnp.asarray(pre * cs), jnp.int32(T_eff), n_det=1, t0=p + 1.0
        )
        tmpl = jax.tree.map(jnp.asarray, state_template(static))
        st = replace(
            tmpl,
            eq=replace(
                tmpl.eq,
                vol=VolEq(
                    core=ConstantEq(log_lam=jnp.asarray(log_lam + 2 * np.log(cs))), trans=Empty()
                ),
            ),
        )
        return static, prior, st, data

    return build


@pytest.mark.parametrize(
    "mode, path",
    [
        ("I", "eigen"),
        ("I", "weighted"),
        ("II", "fixed_factor"),
        ("II", "cholesky_const"),
        ("iw_equivalent", "cholesky_const"),
    ],
)
def test_scale_equivariance(mode, path):
    """Item 7: ``y_j → c_j y_j`` (ψ_j → c_j²ψ_j, λ_i → c_i²λ_i, priors rebuilt by the WP6 builders):
    with the same key ``a_ij`` scales by ``c_i/c_j``, the intercept by ``c_i``, lag ``l`` of ``m`` by
    ``c_i/c_m`` and ``γ`` and ``B`` consistently (1e-9)."""
    c = np.array([3.0, 0.1, 7.0])
    build = equivariance_setup(mode, path, c)
    n, p, k = 3, 2, 7
    out = []
    for cs in (np.ones(3), c):
        static, prior, st, data = build(cs)
        v = views(static, prior, st, data)
        thetas, gammas = [], []
        for i in range(n):
            ctx, pm, cm, cache_i = v.eq(i)
            ms, a, _, info = v.M.sample_eq(
                random.fold_in(KEY, i), None, None, ctx, pm, cm, cache_i, v.cache, prior.shared
            )
            assert bool(info.ok)
            thetas.append(np.concatenate([np.asarray(ms.delta), np.asarray(a)]))
            gammas.append(np.asarray(v.M.extract_eq(ms, a, prior.shared).gamma))
        a_all = np.stack([t[k:] for t in thetas])
        B = np.asarray(
            v.S.reduced_form(StructDraw(a=jnp.asarray(a_all)), jnp.asarray(np.stack(gammas)))
        )
        out.append((np.stack(thetas), np.stack(gammas), B))
    var_of = np.concatenate([[-1], np.tile(np.arange(n), p)])  # regressor layout (det, lag blocks)
    xscale = np.where(var_of < 0, 1.0, c[np.maximum(var_of, 0)])
    for i in range(n):
        D = np.concatenate([c[i] / xscale, c[i] / c[: n - 1]])
        np.testing.assert_allclose(out[1][0][i], D * out[0][0][i], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            out[1][1][i], c[i] / xscale * out[0][1][i], rtol=1e-9, atol=1e-12
        )
    np.testing.assert_allclose(
        out[1][2], (xscale[:, None] ** -1 * c[None, :]) * out[0][2], rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------- coupling, D3 algebra


@pytest.mark.parametrize(
    "name", ["structural_mode2_constant", "structural_iw_equivalent", "structural_constant"]
)
def test_coupling_hand_formulas(name):
    """Mode II: ``Coupling(k + i, ½Σ_active P̃θ(θ − meanθ)²)``; ``iw_equivalent``: the same over the
    active ``a``-slots (``n_active = i``); Mode I: zeros (DESIGN §4.7)."""
    spec = joint_spec(name, n=3)
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 8)
    v = views(static, prior, st, data)
    k = static.shapes.k
    for i in range(3):
        ctx, pm, cm, cache_i = v.eq(i)
        ms, a, cpl, _info = v.M.sample_eq(
            random.fold_in(KEY, 90 + i), None, None, ctx, pm, cm, cache_i, v.cache, prior.shared
        )
        d, a = np.asarray(ms.delta), np.asarray(a)[:i]
        P_d, m_d = np.asarray(prior.eq.coef.prec0)[i], np.asarray(prior.eq.coef.mean0)[i]
        P_a, m_a = (
            np.asarray(prior.eq.contemp.prec0)[i, :i],
            np.asarray(prior.eq.contemp.mean0)[i, :i],
        )
        quad_a = 0.5 * np.sum(P_a * (a - m_a) ** 2)
        quad_d = 0.5 * np.sum(P_d * (d - m_d) ** 2)
        if name == "structural_mode2_constant":
            assert float(cpl.n_active) == k + i
            np.testing.assert_allclose(float(cpl.quad), quad_a + quad_d, rtol=1e-12)
        elif name == "structural_iw_equivalent":
            assert float(cpl.n_active) == i
            np.testing.assert_allclose(float(cpl.quad), quad_a, rtol=1e-12, atol=1e-300)
        else:
            assert float(cpl.n_active) == 0.0 and float(cpl.quad) == 0.0


def test_fitted_residual_extract_predict():
    spec = joint_spec("structural_sv", n=3, T_eff=29)
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 9)
    v = views(static, prior, st, data)
    k = static.shapes.k
    F = np.asarray(v.cache.F)
    Mc = np.asarray(prior.shared.Mc)
    for i in range(3):
        ctx, _pm, _cm, _cache_i = v.eq(i)
        eq_i = eq_slice(st.eq, i)
        theta = np.concatenate([np.asarray(eq_i.mean.delta), np.asarray(eq_i.struct.a)])
        g = v.M.fitted_eq(eq_i.mean, eq_i.struct.a, ctx, v.cache, prior.shared)
        e = np.asarray(v.S.residual(eq_i.struct.a, ctx, g))
        np.testing.assert_allclose(
            e, (np.asarray(data.Y)[:, i] - F @ theta) * np.asarray(data.tmask), atol=1e-12
        )
        gam = np.asarray(v.M.extract_eq(eq_i.mean, eq_i.struct.a, prior.shared).gamma)
        np.testing.assert_allclose(gam, theta[:k] + Mc @ theta[k:], rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(np.asarray(g), np.asarray(data.X) @ gam, rtol=1e-13, atol=1e-14)
        x = np.random.default_rng(i).normal(size=k)
        np.testing.assert_allclose(float(v.M.predict_eq(v.M.extract_eq(eq_i.mean, eq_i.struct.a, prior.shared), Empty(), jnp.asarray(x), jnp.asarray(1), Empty())), gam @ x, rtol=1e-13)  # fmt: skip
        np.testing.assert_array_equal(
            np.asarray(v.M.coefficients_eq(eq_i.mean, eq_i.struct.a, prior.shared)), theta[:k]
        )
    gd = v.M.extract_global(st.glob)
    assert all(isinstance(getattr(gd, f.name), Empty) for f in dataclasses.fields(gd))
    np.testing.assert_array_equal(
        np.asarray(v.M.predict_reduced(gd, jnp.ones(k), jnp.asarray(1), Empty())), np.zeros(3)
    )
    assert (
        isinstance(v.M.future_eq(KEY, None, 4), Empty) and v.M.fitted_reduced(st.glob, data) is None
    )
    assert v.M.prepare(data, prior, v.cache) is v.cache  # weighted path: no per-equation cache


@pytest.mark.parametrize(
    "name", ["structural_constant", "structural_mode2_constant", "structural_iw_equivalent"]
)
def test_log_prior_eq_matches_references(name):
    """``log_prior_eq`` (``pm`` λ-free, ``log_scale`` given; ``cm`` scaled by the structure) equals the
    independent references' ``log p(δ_i) + log p(a_i)`` (1e-12)."""
    spec = joint_spec(name, n=3)
    static, prior = spec.static, spec.prior
    st, data = problem(spec, 10)
    v = views(static, prior, st, data)
    tm = data.tmask
    for i in range(3):
        ctx, pm, cm, _ = v.eq(i)
        eq_i = eq_slice(st.eq, i)
        mine = float(
            v.M.log_prior_eq(
                eq_i.mean, eq_i.struct.a, pm, cm, prior.shared, log_scale=ctx.log_scale
            )
        )
        refs = references_for(static)
        scale = coupling_log_scale(eq_i.vol.core, static)
        p_i = eq_slice(prior.eq, i)
        ref = float(
            refs.mean.log_prior_eq(eq_i.mean, p_i, static, tm, log_scale=scale)
            + refs.structure.log_prior_eq(eq_i.struct, p_i, static, tm, log_scale=scale)
        )
        np.testing.assert_allclose(mine, ref, rtol=1e-12)
        only_d = float(
            v.M.log_prior_eq(
                eq_i.mean, eq_i.struct.a, pm, None, prior.shared, log_scale=ctx.log_scale
            )
        )
        np.testing.assert_allclose(
            only_d,
            float(refs.mean.log_prior_eq(eq_i.mean, p_i, static, tm, log_scale=scale)),
            rtol=1e-12,
        )
    assert np.isfinite(float(log_prior_eq(eq_slice(st.eq, 0), eq_slice(prior.eq, 0), static, tm)))


def test_init_eq_draw():
    """``init_eq`` draws one exact ``(δ_i, a_i)`` with the context's weights on any path (batched Gram
    from ``cache.F``): equal to the weighted-path draw with the same key (1e-12)."""
    spec = joint_spec("structural_constant", n=3, T_eff=33)
    st, data = problem(spec, 11)
    v_e = views(spec.static, spec.prior, st, data)
    spec_w = joint_spec("structural_constant", n=3, T_eff=33, const_path="weighted")
    v_w = views(spec_w.static, spec_w.prior, st, data)
    for i in range(3):
        ctx, pm, cm, cache_i = v_e.eq(i)
        key = random.fold_in(KEY, 300 + i)
        ms, a = v_e.M.init_eq(key, jnp.zeros(2), ctx, pm, cm, cache_i, v_e.cache, spec.prior.shared)
        ctx_w, pm_w, cm_w, cache_w = v_w.eq(i)
        ms_w, a_w, _, _ = v_w.M.sample_eq(
            key, None, None, ctx_w, pm_w, cm_w, cache_w, v_w.cache, spec.prior.shared
        )
        np.testing.assert_allclose(
            np.asarray(ms.delta), np.asarray(ms_w.delta), rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(np.asarray(a), np.asarray(a_w), rtol=1e-12, atol=1e-13)
        np.testing.assert_array_equal(np.asarray(a)[i:], 0.0)


# --------------------------------------------------------------------------- override seams


def no_centring(delta, a, Mc):
    """Mutant M8: ``γ_i = δ_i`` (D3 centring dropped)."""
    return delta + 0.0 * (Mc @ a)


def test_structural_gamma_override_rebuilds_design():
    """An overridden ``structural_gamma`` is honoured everywhere: the conditional is the dense
    regression on ``[X | X·∂γ/∂a − Y_{<i}]`` (here ``[X | −Y_{<i}]``), ``fitted = Xδ``, the stored
    ``γ = δ``, and the fast paths switch to the Cholesky path."""
    for path in ("eigen", "weighted"):
        spec = joint_spec("structural_constant", n=3, const_path=path)
        static, prior = spec.static, spec.prior
        st, data = problem(spec, 12)
        M = LinearStructural.from_static(static).with_overrides({"structural_gamma": no_centring})
        assert M.draw_path == ("cholesky_const" if path == "eigen" else "weighted")
        v = views(static, prior, st, data, M)
        k, q = static.shapes.k, static.shapes.q
        for i in (1, 2):
            ctx, pm, cm, cache_i = v.eq(i)
            cond = M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, prior.shared)
            _, _, _, w, prec, mean, X, Y = dense_problem(static, prior, st, data, i)
            Xt = np.concatenate([X, -Y[:, :i]], axis=1)
            Kd = np.diag(prec) + Xt.T @ (w[:, None] * Xt)
            bd = prec * mean + Xt.T @ (w * Y[:, i])
            on = active(k, i, q)
            np.testing.assert_allclose(
                np.asarray(cond.K)[np.ix_(on, on)], Kd, rtol=1e-11, atol=1e-12 * np.abs(Kd).max()
            )
            np.testing.assert_allclose(
                np.asarray(cond.b)[on], bd, rtol=1e-11, atol=1e-12 * np.abs(bd).max()
            )
            ms, a, _, info = M.sample_eq(
                KEY, None, None, ctx, pm, cm, cache_i, v.cache, prior.shared
            )
            assert bool(info.ok)
            np.testing.assert_array_equal(
                np.asarray(M.extract_eq(ms, a, prior.shared).gamma), np.asarray(ms.delta)
            )
            g = np.asarray(M.fitted_eq(ms, a, ctx, v.cache, prior.shared))
            np.testing.assert_allclose(
                g, np.asarray(data.X) @ np.asarray(ms.delta), rtol=1e-13, atol=1e-14
            )


def test_linear_conditional_override_reaches_every_path():
    calls = []

    def spy(ctx, pm, cm, cache, shared):
        calls.append(1)
        return LinearStructural.from_static(spec.static).linear_conditional(
            ctx, pm, cm, cache, shared
        )

    spec = joint_spec("structural_mode2_constant", n=3)
    st, data = problem(spec, 13)
    M = LinearStructural.from_static(spec.static).with_overrides({"linear_conditional": spy})
    assert M.draw_path == "cholesky_const"
    v = views(spec.static, spec.prior, st, data, M)
    ctx, pm, cm, cache_i = v.eq(2)
    _ms, _a, _, info = M.sample_eq(
        KEY, None, None, ctx, pm, cm, cache_i, v.cache, spec.prior.shared
    )
    assert calls and bool(info.ok)
    # same law as the production fast path: the conditional mean (z = 0) agrees
    v0 = views(spec.static, spec.prior, st, data)
    ctx0, pm0, cm0, c0 = v0.eq(2)
    q = spec.static.shapes.q  # equation 2 of 3: every slot active
    m_fast = np.asarray(
        v0.M.draw_core(jnp.zeros(q), ctx0, pm0, cm0, c0, v0.cache, spec.prior.shared)[0]
    )
    m_spy = np.asarray(
        M.draw_core(jnp.zeros(q), ctx, pm, cm, cache_i, v.cache, spec.prior.shared)[0]
    )
    np.testing.assert_allclose(m_spy, m_fast, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- density-ratio identities


DENSITY_SPECS = [
    "structural_constant",
    "structural_sv",
    "structural_mode2_constant",
    "structural_iw_equivalent",
    "structural_sv_outliers_t",
]


@pytest.mark.parametrize("name", DENSITY_SPECS)
def test_density_ratio_equation_step(name):
    """``log q(θ′_i) − log q(θ_i) = log p(θ′_i, ·, y) − log p(θ_i, ·, y)`` with ``q`` the exposed
    conditional and the independent ``validation.log_joint`` at 20 random states (1e-8)."""
    spec = joint_spec(name, n=3)
    static, prior = spec.static, spec.prior
    n, k = static.shapes.n, static.shapes.k

    @jax.jit
    def run(key):
        st = spec.prior_sample(random.fold_in(key, 0))
        data = spec.simulate(random.fold_in(key, 1), st)
        v = views(static, prior, st, data)
        out = []
        for i in range(n):
            ctx, pm, cm, cache_i = v.eq(i)
            cond = v.M.conditional_eq(None, None, ctx, pm, cm, cache_i, v.cache, prior.shared)
            mask = jnp.concatenate([pm.cmask, cm.cmask])
            th0 = jnp.concatenate([st.eq.mean.delta[i], st.eq.struct.a[i]])
            th1 = jnp.where(
                mask > 0, th0 + 0.05 * random.normal(random.fold_in(key, 5 + i), th0.shape), 0.0
            )

            def log_q(th, cond=cond):
                return -0.5 * th @ cond.K @ th + cond.b @ th

            def with_theta(th, i=i):
                eq = replace(
                    st.eq,
                    mean=LinearEq(delta=st.eq.mean.delta.at[i].set(th[:k])),
                    struct=RecursiveEq(a=st.eq.struct.a.at[i].set(th[k:])),
                )
                return replace(st, eq=eq)

            out.append(
                jnp.stack([log_q(th0), log_q(th1), log_joint(with_theta(th0), data, prior, static),
                           log_joint(with_theta(th1), data, prior, static)])
            )  # fmt: skip
        return jnp.stack(out)

    worst = 0.0
    for rep in range(20):
        for lq0, lq1, lj0, lj1 in np.asarray(run(random.fold_in(KEY, 2000 + rep))):
            worst = max(
                worst, density_ratio_error(lambda x: x[0], lambda x: x[1], (lq0, lj0), (lq1, lj1))
            )
    assert worst <= 1e-8, worst


def test_equation_phase_vmap_equals_loop_and_keeps_vol():
    spec = joint_spec("structural_sv_t", n=3)
    st, data = problem(spec, 14)
    it = jnp.uint32(5)
    new = jax.jit(lambda k, s, d: equation_phase(spec, k, it, s, d))(KEY, st, data)
    np.testing.assert_array_equal(np.asarray(new.eq.vol.core.h), np.asarray(st.eq.vol.core.h))
    v = views(spec.static, spec.prior, st, data)
    kit = random.fold_in(KEY, it)
    for i in range(3):
        ctx, pm, cm, cache_i = v.eq(i)
        k = random.fold_in(random.fold_in(kit, int(Block.EQ_MEAN)), jnp.uint32(i))
        ms, a, _, _ = v.M.sample_eq(k, None, None, ctx, pm, cm, cache_i, v.cache, spec.prior.shared)
        np.testing.assert_allclose(
            np.asarray(new.eq.mean.delta[i]), np.asarray(ms.delta), rtol=1e-14, atol=1e-15
        )
        np.testing.assert_allclose(
            np.asarray(new.eq.struct.a[i]), np.asarray(a), rtol=1e-14, atol=1e-15
        )


# --------------------------------------------------------------------------- tier B, block Geweke


@pytest.mark.parametrize(
    "name", ["structural_sv", "structural_constant", "structural_mode2_constant"]
)
def test_block_sbc_equation_step(name):
    """Block-conditional SBC of ``(δ, a) | λ, y`` (N06 §5.2; DESIGN §9.4): N = 200, L = 99."""
    spec = joint_spec(name)
    n, k = spec.static.shapes.n, spec.static.shapes.k

    def joint_sample(key):
        st = spec.prior_sample(random.fold_in(key, 0))
        return st, spec.simulate(random.fold_in(key, 1), st)

    def conditional_draws(key, st, data):
        return equation_phase(spec, key, jnp.uint32(0), st, data)

    def stat(st, data):
        vals = [st.eq.mean.delta[i, r] for i in range(n) for r in range(k)]
        vals += [st.eq.struct.a[i, j] for i in range(n) for j in range(i)]
        return jnp.stack([*vals, loglik(st, data, spec.prior, spec.static)])

    res = block_sbc(joint_sample, conditional_draws, stat, N=200, L=99, seed=21, alpha=1e-3)
    assert res.passed, (res.reasons, res.p)


@dataclass(frozen=True, eq=False)
class EquationOnly:
    """``equation_only`` (DESIGN §9.2): JointSpec's prior, simulator and test functions; the step is
    the equation phase (fused ``(δ_i, a_i)`` draw) with every other block held at its prior draw."""

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
        new = equation_phase(self.spec, key, it, st, data, self.overrides)
        return hold_frozen(new, st, self.spec.live)

    def test_functions(self, st, data):
        return self.spec.test_functions(st, data)


LIVE = frozenset({"eq.mean", "eq.struct"})
G_SMALL = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}

GEWEKE_CASES = {
    "eigen": ("equation_only", {}),  # the configuration of DESIGN §9.2 (Mode I × Constant)
    "weighted_sv": ("structural_sv", {"live": LIVE}),
    "fixed_factor": ("structural_mode2_constant", {"live": LIVE}),
    "cholesky_const": ("structural_iw_equivalent", {"live": LIVE}),
    "weighted_tiled_n3": (
        "structural_sv_outliers",
        {"live": LIVE, "n": 3, "gram": "packed_tiled", "gram_tile": 7},
    ),
}


@pytest.mark.geweke
@pytest.mark.parametrize("case", list(GEWEKE_CASES))
def test_block_geweke_equation_only(case):
    """Block Geweke ``equation_only`` at G-small on every path (C = 1000, M_c = 100, M₁ = 2·10⁵,
    α = 1e-3; DESIGN §9.5 "blocks")."""
    name, kw = GEWEKE_CASES[case]
    spec = joint_spec(name, **kw)
    expected = {"eigen": "eigen", "weighted_sv": "weighted", "fixed_factor": "fixed_factor",
                "cholesky_const": "cholesky_const", "weighted_tiled_n3": "weighted"}[case]  # fmt: skip
    assert LinearStructural.from_static(spec.static).draw_path == expected
    res = geweke(EquationOnly(spec), GewekeConfig(seed=31, **G_SMALL))
    assert res.passed, res.table().head(8)


@pytest.mark.geweke
def test_block_geweke_detects_dropped_centring():
    """Power check: the M8 mutant (``γ = δ`` through the ``structural_gamma`` seam) fails the same
    block Geweke test (the D3 cross moment ``a·γ`` and ``‖Γ − AM′‖²`` move)."""
    spec = joint_spec("equation_only")
    res = geweke(
        EquationOnly(spec, (("structural_gamma", no_centring),)), GewekeConfig(seed=32, **G_SMALL)
    )
    assert not res.passed


def require_production_sweep(spec: JointSpec) -> None:
    """Skip until the production sweep (WP14, with the WP11/WP13 kernels) can be composed."""
    try:
        production_sweep(spec.static, spec.overrides)
    except NotImplementedError as err:
        pytest.skip(f"production sweep not available yet: {err}")


@pytest.mark.geweke
def test_block_geweke_equation_only_production_sweep():
    """``equation_only`` through ``JointSpec.step``, i.e. the production ``compose``/``make_sweep``
    (DESIGN §9.2); runs once WP14 lands (the local equation phase above covers it until then)."""
    spec = joint_spec("equation_only")
    require_production_sweep(spec)
    res = geweke(spec, GewekeConfig(seed=33, **G_SMALL))
    assert res.passed, res.table().head(8)
