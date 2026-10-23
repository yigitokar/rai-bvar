"""The four ``LinearStructural`` paths draw the same law (DESIGN §3.7.2, §9.3 item 1, §13 WP10; D-25,
D-26; N05 §7.4).

``weighted`` (packed, packed-tiled and batched Grams), ``cholesky_const``, ``eigen`` (Mode I ×
Constant) and ``fixed_factor`` (Mode II × Constant) are compared through the draw itself: every
path is linear in the injected noise, ``θ(z) = μ + Rz``, so ``μ = θ(0)`` and ``RRᵀ = Cov`` are read
off the production draw. Well-conditioned posteriors (``cond(sKs) < 1e3``) agree to 1e-12 in
posterior standard deviations; Minnesota level data (``var_const = 1e7``, persistent levels,
``cond(sKs) ~ 1e6``) agree within the WP4 envelope ``2e-12 + 5εκ(1 + t)`` (mean) and
``2e-12 + 5εκ`` (covariance). KS tests on 10⁴ draws per path, the per-fit caches, failures as
values, and (``bench``, opt-in) the weighted-path timing at the reference shape against the WP3
pilot (``docs/phase2/09_pilot.md`` §10: 3.30 ms per mean draw for 80 lanes).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, random
from scipy import stats

from rai_bvar.components.coef.fixed import FixedDiagonal
from rai_bvar.components.mean.linear import LinearStructural, theta_prior_arrays
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
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.state import (
    ChainState,
    ConstantEq,
    ConstantPrior,
    EigenCache,
    Empty,
    EngineData,
    EqPrior,
    FixedFactorCache,
    PriorArrays,
    VolEq,
    state_template,
)
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import WindowScales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.validation.joint import JointSpec, joint_spec
from rai_bvar.validation.reference import coupling_log_scale, effective_log_var

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261024, impl="threefry2x32")
EPS = float(np.finfo(np.float64).eps)
PILOT_MS = 3.30  # docs/phase2/09_pilot.md §10: one-bucket mean draw, 80 lanes, M1 Pro


# --------------------------------------------------------------------------- helpers


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


def eq_inputs(static: StaticConfig, prior: PriorArrays, st: ChainState, data: EngineData, i: int):
    """``(M, ctx, pm, cm, cache_i, cache)`` of equation ``i`` as the sweep body builds them."""
    S = Recursive.from_static(static)
    M = LinearStructural.from_static(static)
    C = FixedDiagonal.from_static(static)
    cache = M.prepare(data, prior, base_cache(data, prior.shared.Mc, static))
    ytil = S.pre_residual(st.glob, data, None)
    eq_i, prior_i = eq_slice(st.eq, i), eq_slice(prior.eq, i)
    ctx = S.context(jnp.asarray(i, jnp.int32), ytil, eq_i, StubVol(static), data, None)
    pm = C.moments_eq(eq_i.coef, ctx.i, prior_i, None)
    cm = S.contemp_moments(ctx, prior_i)
    cache_i = Empty() if isinstance(cache.eq, Empty) else eq_slice(cache.eq, i)
    return M, ctx, pm, cm, cache_i, cache


def path_moments(static, prior, st, data, i: int):
    """``(μ, Cov, K)`` of equation ``i``'s draw on its path, read off ``θ(z) = μ + Rz``."""
    M, ctx, pm, cm, cache_i, cache = eq_inputs(static, prior, st, data, i)
    q = static.shapes.q

    def f(z):
        return M.draw_core(z, ctx, pm, cm, cache_i, cache, prior.shared)[0]

    mu = f(jnp.zeros(q))
    R_T = jax.vmap(f)(jnp.eye(q)) - mu
    cond = M.conditional_eq(None, None, ctx, pm, cm, cache_i, cache, prior.shared)
    return np.asarray(mu), np.asarray(R_T.T @ R_T), np.asarray(cond.K)


def compare(static_of, prior, st, data, paths, *, envelope: bool):
    """Every path against the first; returns the worst (mean, cov) errors and κ."""
    n, k, q = (
        static_of(paths[0]).shapes.n,
        static_of(paths[0]).shapes.k,
        static_of(paths[0]).shapes.q,
    )
    res = {p: [path_moments(static_of(p), prior, st, data, i) for i in range(n)] for p in paths}
    worst = []
    for i in range(n):
        on = np.arange(q) < k + i
        mu0, C0, K = res[paths[0]][i]
        Ka = K[np.ix_(on, on)]
        s = 1.0 / np.sqrt(np.diag(Ka))
        kappa = float(np.linalg.cond(s[:, None] * Ka * s[None, :]))
        sd = np.sqrt(np.diag(C0)[on])
        t = float(np.max(np.abs(mu0[on]) / sd))
        tol_m = 2e-12 + 5 * EPS * kappa * (1 + t) if envelope else 1e-12
        tol_c = 2e-12 + 5 * EPS * kappa if envelope else 1e-12
        for p in paths[1:]:
            mu, C, _ = res[p][i]
            dm = float(np.max(np.abs(mu[on] - mu0[on]) / sd))
            dc = float(np.max(np.abs(C - C0)[np.ix_(on, on)] / np.outer(sd, sd)))
            assert dm <= tol_m, (p, i, dm, tol_m, kappa, t)
            assert dc <= tol_c, (p, i, dc, tol_c, kappa)
            worst.append((dm, dc, kappa))
    return worst


def problem(spec: JointSpec, seed: int):
    k = random.fold_in(KEY, seed)
    st = spec.prior_sample(random.fold_in(k, 0))
    return st, spec.simulate(random.fold_in(k, 1), st)


PATHS = {
    "structural_constant": ["cholesky_const", "eigen", "weighted"],
    "structural_mode2_constant": ["cholesky_const", "fixed_factor", "weighted"],
    "structural_iw_equivalent": ["cholesky_const", "weighted"],
}


# --------------------------------------------------------------------------- agreement


@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("name", list(PATHS))
def test_paths_agree_well_conditioned(name, size):
    """Mean and covariance of every path equal the ``cholesky_const`` oracle to 1e-12 (sd units) on
    well-conditioned posteriors (DESIGN §9.3 item 1)."""
    specs = {p: joint_spec(name, size, const_path=p) for p in PATHS[name]}
    st, data = problem(specs[PATHS[name][0]], 1 if size == "small" else 2)
    prior = specs[PATHS[name][0]].prior
    worst = compare(lambda p: specs[p].static, prior, st, data, PATHS[name], envelope=False)
    assert max(k for *_, k in worst) < 1e3


def level_problem(mode: str):
    """Minnesota level data: random-walk levels 5–3000, ``var_const = 1e7``, own-lag means 1."""
    n, p, T_eff, T_pad = 4, 2, 90, 96
    r = np.random.default_rng(5)
    lev = np.cumsum(r.normal(size=(T_eff + p, n)) * np.array([0.5, 2.0, 0.1, 5.0]), axis=0)
    lev = lev + np.array([100.0, 1000.0, 5.0, 3000.0])
    y = np.zeros((T_pad, n))
    y[:T_eff] = lev[p:]
    psi = np.array([0.25, 4.0, 0.01, 25.0])
    mn = Minnesota(lambda_=0.2, form="structural")
    scaled = mode != "I"
    sc = WindowScales(psi=psi, own_means=np.ones(n), freq="Q")
    coef, shared = structural_arrays(
        sc, mn, {}, n=n, lags=p, n_det=1, mode="II" if mode == "II" else "I"
    )
    contemp = contemp_arrays(sc, mn, n, scaling="lambda" if scaled else "none")
    core = ConstantPrior(shape=np.full(n, 3.0), scale=psi)
    prior = PriorArrays(
        eq=EqPrior(coef=coef, contemp=contemp, vol_core=core, transitory=Empty()), shared=shared
    )
    prior = jax.tree.map(jnp.asarray, prior)
    data = build_engine_data(
        jnp.asarray(y), jnp.asarray(lev[:p]), jnp.int32(T_eff), n_det=1, t0=p + 1.0
    )

    def static_of(path: str) -> StaticConfig:
        return StaticConfig(
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

    tmpl = jax.tree.map(jnp.asarray, state_template(static_of("weighted")))
    log_lam = jnp.asarray(np.log(psi * np.array([0.5, 1.0, 2.0, 3.0])))
    st = replace(
        tmpl, eq=replace(tmpl.eq, vol=VolEq(core=ConstantEq(log_lam=log_lam), trans=Empty()))
    )
    return static_of, prior, st, data


@pytest.mark.parametrize("mode", ["I", "II"])
def test_paths_agree_minnesota_level_data(mode):
    """On Minnesota level data (``cond(sKs) ~ 1e6``) the fast paths and the weighted path agree with
    ``cholesky_const`` within the WP4 envelope (``linalg.eigen_prepare``)."""
    static_of, prior, st, data = level_problem(mode)
    paths = ["cholesky_const", "eigen" if mode == "I" else "fixed_factor", "weighted"]
    worst = compare(static_of, prior, st, data, paths, envelope=True)
    assert max(k for *_, k in worst) > 1e5  # the case is genuinely ill-conditioned


def test_gram_modes_agree():
    """Packed, packed-tiled and batched weighted Grams give the same conditional (1e-13 relative) and
    the same draws (1e-12)."""
    variants = {
        "packed": {},
        "packed_tiled": {"gram": "packed_tiled", "gram_tile": 5},
        "batched": {"gram": "batched"},
    }
    specs = {g: joint_spec("structural_sv", "full", **kw) for g, kw in variants.items()}
    st, data = problem(specs["packed"], 3)
    prior = specs["packed"].prior
    out = {}
    for g, spec in specs.items():
        M, ctx, pm, cm, cache_i, cache = eq_inputs(spec.static, prior, st, data, 2)
        assert isinstance(cache.FF, Empty) == (g != "packed")
        cond = M.conditional_eq(None, None, ctx, pm, cm, cache_i, cache, prior.shared)
        ms, a, _, info = M.sample_eq(KEY, None, None, ctx, pm, cm, cache_i, cache, prior.shared)
        assert bool(info.ok)
        out[g] = (np.asarray(cond.K), np.concatenate([np.asarray(ms.delta), np.asarray(a)]))
    K0, th0 = out["packed"]
    for g in ("packed_tiled", "batched"):
        K, th = out[g]
        np.testing.assert_allclose(K, K0, rtol=1e-13, atol=1e-13 * np.abs(K0).max())
        np.testing.assert_allclose(th, th0, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", list(PATHS))
def test_paths_ks(name):
    """10⁴ draws per path: every active coordinate, standardized by the exact moments, and the
    Mahalanobis distance (``χ²_q``) pass KS tests (Bonferroni α = 1e-3)."""
    N = 10_000
    specs = {p: joint_spec(name, "full", const_path=p) for p in PATHS[name]}
    st, data = problem(specs[PATHS[name][0]], 4)
    prior = specs[PATHS[name][0]].prior
    i = 2  # every slot active
    mu, C, _ = path_moments(specs["cholesky_const"].static, prior, st, data, i)
    sd = np.sqrt(np.diag(C))
    L = np.linalg.cholesky(C)
    pvals = []
    for j, (path, spec) in enumerate(specs.items()):
        M, ctx, pm, cm, cache_i, cache = eq_inputs(spec.static, prior, st, data, i)

        def one(key, M=M, ctx=ctx, pm=pm, cm=cm, cache_i=cache_i, cache=cache):
            ms, a, _, _ = M.sample_eq(key, None, None, ctx, pm, cm, cache_i, cache, prior.shared)
            return jnp.concatenate([ms.delta, a])

        base = random.fold_in(KEY, 500 + j)
        keys = jax.vmap(lambda r, base=base: random.fold_in(base, r))(
            jnp.arange(N, dtype=jnp.uint32)
        )
        th = np.asarray(jax.jit(jax.vmap(one))(keys))
        zs = (th - mu) / sd
        pvals += [stats.kstest(zs[:, c], "norm").pvalue for c in range(zs.shape[1])]
        maha = np.sum(np.linalg.solve(L, (th - mu).T) ** 2, axis=0)
        pvals.append(stats.kstest(maha, stats.chi2(df=th.shape[1]).cdf).pvalue)
        assert path in ("cholesky_const", "eigen", "fixed_factor", "weighted")
    assert min(pvals) > 1e-3 / len(pvals), sorted(pvals)[:3]


# --------------------------------------------------------------------------- caches and paths


def test_prepare_caches_and_theta_prior():
    for name, path, cls in (
        ("structural_constant", "eigen", EigenCache),
        ("structural_mode2_constant", "fixed_factor", FixedFactorCache),
    ):
        spec = joint_spec(name, n=3)
        st, data = problem(spec, 5)
        static, prior = spec.static, spec.prior
        M = LinearStructural.from_static(static)
        assert M.draw_path == path
        base = base_cache(data, prior.shared.Mc, static)
        cache = M.prepare(data, prior, base)
        assert isinstance(cache.eq, cls) and isinstance(cache.FF, Empty)
        n, q = static.shapes.n, static.shapes.q
        for leaf in jax.tree.leaves(cache.eq):
            assert leaf.shape[:2] == (n, q) and leaf.dtype == jnp.float64
        assert cache.G_const is base.G_const and cache.FtY is base.FtY
        # the cached prior equals the sweep-time prior at λ = 1 (λ-free)
        for i in range(n):
            _, ctx, pm, _cm, _, _ = eq_inputs(static, prior, st, data, i)
            ctx1 = replace(ctx, log_scale=jnp.zeros(()))
            cm1 = Recursive.from_static(static).contemp_moments(ctx1, eq_slice(prior.eq, i))
            ref = M.theta_prior(ctx1, pm, cm1)
            mine = theta_prior_arrays(eq_slice(prior.eq, i), jnp.asarray(i), n)
            for x, y in zip(jax.tree.leaves(mine), jax.tree.leaves(ref), strict=True):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for path in ("weighted", "cholesky_const"):
        spec = joint_spec("structural_constant", n=3, const_path=path)
        st, data = problem(spec, 6)
        base = base_cache(data, spec.prior.shared.Mc, spec.static)
        assert LinearStructural.from_static(spec.static).prepare(data, spec.prior, base) is base


def test_eigen_lambda_tracks_weights():
    """The eigen path reads ``λ_i = 1/w_0``: changing ``λ`` in the state changes the draw exactly as
    the Cholesky oracle's (1e-12, sd units)."""
    for shift in (-2.0, 0.0, 3.0):
        specs = {
            p: joint_spec("structural_constant", n=3, const_path=p)
            for p in ("eigen", "cholesky_const")
        }
        st, data = problem(specs["eigen"], 7)
        core = st.eq.vol.core
        st = replace(
            st,
            eq=replace(
                st.eq, vol=replace(st.eq.vol, core=ConstantEq(log_lam=core.log_lam + shift))
            ),
        )
        prior = specs["eigen"].prior
        compare(
            lambda p, specs=specs: specs[p].static,
            prior,
            st,
            data,
            ["cholesky_const", "eigen"],
            envelope=False,
        )


def test_failures_are_values():
    """Non-PD conditionals and poisoned caches give ``ok = False`` on every path, never an exception
    or jitter (D-18)."""
    for name, path in (
        ("structural_constant", "eigen"),
        ("structural_mode2_constant", "fixed_factor"),
        ("structural_constant", "cholesky_const"),
        ("structural_sv", "weighted"),
    ):
        spec = joint_spec(name, n=3, const_path=path)
        st, data = problem(spec, 8)
        M, ctx, pm, cm, cache_i, cache = eq_inputs(spec.static, spec.prior, st, data, 2)
        _ms, _a, _, info = M.sample_eq(
            KEY, None, None, ctx, pm, cm, cache_i, cache, spec.prior.shared
        )
        assert bool(info.ok)
        if path in ("eigen", "fixed_factor"):
            bad_i = jax.tree.map(lambda x: x * jnp.nan, cache_i)
            *_, info2 = M.sample_eq(KEY, None, None, ctx, pm, cm, bad_i, cache, spec.prior.shared)
        else:
            bad_pm = replace(pm, prec=pm.prec.at[0].set(-1e12))
            *_, info2 = M.sample_eq(
                KEY, None, None, ctx, bad_pm, cm, cache_i, cache, spec.prior.shared
            )
        assert not bool(info2.ok)


# --------------------------------------------------------------------------- timing (opt-in)


def _pilot_module():
    """``benchmarks/pilot/pilot_common.py`` (WP3), imported read-only for the same-load baseline."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "benchmarks" / "pilot" / "pilot_common.py"
    if not path.exists():
        pytest.skip("WP3 pilot prototypes not present")
    import sys

    name = "wp3_pilot_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


@pytest.mark.bench
@pytest.mark.slow
def test_weighted_path_timing_reference_shape():
    """Weighted path at the BENCH-P2 shape (``n = 20``, ``p = 4``, ``T = 300 → T_pad = 320``,
    ``q = 100``, 4 chains = 80 lanes; packed Gram + Jacobi Cholesky + two solves + residual) within
    25% of the WP3 pilot. The pilot's 3.30 ms (09_pilot.md §10) is machine- and load-specific, so
    the gate compares with the pilot's own one-bucket prototype (``pilot_common.mean_draw``) timed
    in the same process, interleaved, on the same design (minimum over 10 rounds of 50 draws)."""
    pc = _pilot_module()
    C, n, p, T_eff = 4, 20, 4, 300
    spec = joint_spec("structural_sv", n=n, lags=p, T_eff=T_eff)
    static, prior = spec.static, spec.prior
    assert static.shapes.T_pad == 320 and static.shapes.q == 100 and static.mean.gram == "packed"
    r = np.random.default_rng(0)
    data = build_engine_data(
        jnp.asarray(r.normal(size=(320, n))), jnp.asarray(r.normal(size=(p, n))), jnp.int32(T_eff),
        n_det=1, t0=p + 1.0,
    )  # fmt: skip
    st0 = jax.tree.map(jnp.asarray, state_template(static))
    h = jnp.asarray(r.normal(scale=0.3, size=(C, n, 321)) * np.r_[1.0, np.asarray(data.tmask)])
    S = Recursive.from_static(static)
    M = LinearStructural.from_static(static)
    Cf = FixedDiagonal.from_static(static)
    vol = StubVol(static)
    cache = M.prepare(data, prior, base_cache(data, prior.shared.Mc, static))

    def chain(ckey, h_c):
        core = replace(st0.eq.vol.core, h=h_c)
        st = replace(st0, eq=replace(st0.eq, vol=replace(st0.eq.vol, core=core)))
        ytil = S.pre_residual(st.glob, data, None)

        def body(i, eq_i, prior_i):
            k = random.fold_in(random.fold_in(ckey, int(Block.EQ_MEAN)), i.astype(jnp.uint32))
            ctx = S.context(i, ytil, eq_i, vol, data, None)
            cm = S.contemp_moments(ctx, prior_i)
            pm = Cf.moments_eq(eq_i.coef, i, prior_i, None)
            ms, a, _, _ = M.sample_eq(
                k, eq_i.mean, eq_i.struct.a, ctx, pm, cm, Empty(), cache, prior.shared
            )
            g = M.fitted_eq(ms, a, ctx, cache, prior.shared)
            return jnp.sum(S.residual(a, ctx, g))

        return jax.vmap(body)(jnp.arange(n, dtype=jnp.int32), st.eq, prior.eq)

    # the pilot prototype on the same design and prior (buckets.py, one bucket)
    _, _, pos = pc.packed_index(static.shapes.q)
    th_prior = jax.vmap(lambda i, p_: theta_prior_arrays(p_, i, n))(jnp.arange(n), prior.eq)
    Yeq, tmask = data.Y.T, data.tmask

    def pilot_chain(ckey, h_c):
        w = jnp.exp(-h_c[:, 1:]) * tmask
        ks = jax.vmap(lambda i: pc.fold(ckey, i))(jnp.arange(n, dtype=jnp.uint32))
        one = lambda k_, w_, y_, c_, p_, m_: jnp.sum(  # noqa: E731
            pc.mean_draw(k_, w_, y_, cache.F, cache.FF, pos, c_, p_, m_)[1]
        )
        return jax.vmap(one)(ks, w, Yeq, th_prior.cmask, th_prior.prec, th_prior.mean)

    keys = jax.vmap(lambda c: random.fold_in(KEY, c))(jnp.arange(C, dtype=jnp.uint32))
    iters, rounds = 50, 10

    def runner(fn):
        @jax.jit
        def run(keys, h):
            def step(r_, acc):
                ks = jax.vmap(lambda kk: random.fold_in(kk, r_))(keys)
                return acc + jnp.sum(jax.vmap(fn)(ks, h + 1e-12 * r_))

            return lax.fori_loop(0, iters, step, jnp.zeros(()))

        return run

    ours, pilot = runner(chain), runner(pilot_chain)
    ours(keys, h).block_until_ready()
    pilot(keys, h).block_until_ready()
    t_ours, t_pilot = [], []
    for _ in range(rounds):
        t_ours.append(_timed(ours, keys, h))
        t_pilot.append(_timed(pilot, keys, h))
    ms_ours, ms_pilot = (min(t) / iters * 1e3 for t in (t_ours, t_pilot))
    print(
        f"\nweighted path {ms_ours:.3f} ms vs pilot prototype {ms_pilot:.3f} ms per mean draw "
        f"(80 lanes; ratio {ms_ours / ms_pilot:.3f}); 09_pilot.md reference {PILOT_MS} ms"
    )
    assert ms_ours <= 1.25 * ms_pilot, (ms_ours, ms_pilot)


def _timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args).block_until_ready()
    return time.perf_counter() - t0
