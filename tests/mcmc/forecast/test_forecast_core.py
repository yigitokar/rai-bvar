"""The forecast core (DESIGN §13 WP17; §7.1, §7.2, §7.5; N03 §5.2–5.3; N05 §8).

On synthetic parameter draws with the production kernels (constant and trend): the recursion
equals ``B′x + A⁻¹(Λ^{½}ε)`` with the same ``ε`` (1e-12) for structural, reduced and zero means;
``cond_chol`` equals ``np.linalg.cholesky(A⁻¹ΛA⁻ᵀ)`` (1e-12); the volatility futures are the
kernels' ``draw_ahead_eq`` with the injected keys; ``draw_ahead`` moments (AR and RW SV, outlier
frequency and grid, standardized ``E τ = 1``); the chunked executable equals the core; the §7.1
draw selection. On an MCMC fit: forecast-key reproducibility (same ``fseed`` → identical paths,
``o``, ``τ``; replicate keys), post-hoc forecasts recomputed from the stored draws and the
documented keys, and in-loop (``FitPlan``) forecasts equal to post-hoc ones (1e-12), also with
thinned storage (kept-draw index ≠ store position); a trend fit equals Phase 1's simulator.
"""

from __future__ import annotations

import copy
import functools

import jax
import jax.numpy as jnp
import mcmc_fits as mf
import numpy as np
import pandas as pd
import pytest
from jax import random

from rai_bvar import SV, Constant
from rai_bvar.data.design import regressors_from_history
from rai_bvar.engines.mcmc import forecast as F
from rai_bvar.engines.mcmc import keys as KY
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.keys import Ahead
from rai_bvar.engines.mcmc.runner import forecast_chain_keys, forecast_shocks
from rai_bvar.engines.mcmc.state import (
    CCCMDraw,
    ConstantDraw,
    Empty,
    GlobDraw,
    LinearDraw,
    OutlierDraw,
    OutlierTDraw,
    ParamDraw,
    StructDraw,
    SVDraw,
    SVRWDraw,
    TDraw,
    VolDraw,
)
from rai_bvar.engines.mcmc.storage import forecast_selection
from rai_bvar.forecast.simulate import simulate_linear_var
from rai_bvar.settings import FitPlan

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

N, LAGS, NDET, H = 3, 2, 2, 5  # deterministic "ct": the trend continues at t_last + h
K = NDET + N * LAGS
DET = {0: "n", 1: "c", 2: "ct"}[NDET]


def _static(mean="linear_structural", persistent="sv", transitory="none", rw=False, n=N):
    return StaticConfig(
        shapes=Shapes.of(n, LAGS, NDET, 32),
        struct=StructStatic(),
        mean=MeanStatic(kind=mean),
        coef=CoefPriorStatic(),
        vol=VolStatic(persistent=persistent, transitory=transitory, rw=rw, nu_grid=(5, 8, 12)),
    )


def _draw(static: StaticConfig, seed: int = 0, *, p=0.15, nu=8.0) -> ParamDraw:
    """A random ``ParamDraw`` of ``static`` (one draw, per-equation leaves)."""
    rng = np.random.default_rng(seed)
    s, vol = static.shapes, static.vol
    n, k = s.n, s.k
    a = rng.normal(0.0, 0.5, (n, n - 1)) * (np.arange(n - 1)[None, :] < np.arange(n)[:, None])
    gamma = rng.normal(0.0, 0.3, (n, k))
    Pi = rng.normal(0.0, 0.3, (k, n))
    if vol.persistent == "constant":
        core = ConstantDraw(log_lam=rng.normal(-1.0, 0.3, n))
    elif vol.rw:
        core = SVRWDraw(h_T=rng.normal(-1.0, 0.3, n), sigma=rng.uniform(0.1, 0.3, n))
    else:
        core = SVDraw(
            h_T=rng.normal(-1.0, 0.3, n), mu=rng.normal(-1.2, 0.2, n),
            phi=rng.uniform(0.6, 0.95, n), sigma=rng.uniform(0.1, 0.3, n),
        )  # fmt: skip
    lp, l1p = np.full(n, np.log(p)), np.full(n, np.log1p(-p))
    trans = {
        "none": Empty(),
        "outlier": OutlierDraw(log_p=lp, log1m_p=l1p),
        "t": TDraw(nu=np.full(n, nu)),
        "outlier_t": OutlierTDraw(log_p=lp, log1m_p=l1p, nu=np.full(n, nu)),
    }[vol.transitory]
    kind = static.mean.kind
    return ParamDraw(
        glob=GlobDraw(
            mean=CCCMDraw(Pi=Pi) if kind == "linear_reduced" else Empty(),
            coef=Empty(),
            vol=Empty(),
            struct=Empty(),
        ),
        mean=LinearDraw(gamma=gamma) if kind == "linear_structural" else Empty(),
        struct=StructDraw(a=a),
        vol=VolDraw(core=core, trans=trans),
    )


def _unit_lower(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = a[np.tril_indices(n, -1)]
    return A


def _B(static: StaticConfig, draw: ParamDraw) -> np.ndarray:
    """Reduced-form coefficients ``B`` (k, n) written out with NumPy."""
    kind = static.mean.kind
    if kind == "linear_structural":
        return np.linalg.solve(_unit_lower(np.asarray(draw.struct.a)), draw.mean.gamma).T
    if kind == "linear_reduced":
        return np.asarray(draw.glob.mean.Pi)
    return np.zeros((static.shapes.k, static.shapes.n))


@functools.cache
def _core_fn(static: StaticConfig):
    program = compose(static)
    return jax.jit(lambda d, h, t, s: F.forecast_core(d, h, t, s, program))


def _core(static, draw, hist, t_last, shocks):
    return jax.tree.map(np.asarray, _core_fn(static)(draw, hist, t_last, shocks))


def _shocks(seed: int, d: int, n: int = N, horizon: int = H):
    return forecast_shocks(forecast_chain_keys(seed, 2)[1], jnp.uint32(d), horizon, n)


COMPOSITIONS = [
    ("linear_structural", "sv", "outlier_t", False),
    ("linear_structural", "constant", "none", False),
    ("linear_reduced", "sv", "outlier", False),
    ("linear_reduced", "sv", "t", True),
    ("zero", "sv", "none", False),
]


@pytest.mark.parametrize("comp", COMPOSITIONS, ids=lambda c: "-".join(map(str, c)))
def test_recursion_equals_reduced_form_with_the_same_shocks(comp):
    """``y_{T+h} = B′x_{T+h} + A⁻¹(Λ^{½}_{T+h} ε_h)`` and ``cond_mean = B′x`` (N05 §8: 1e-12),
    ``x`` built by ``data.design.regressors_from_history`` from the simulated path; ``cond_chol =
    cholesky(A⁻¹ΛA⁻ᵀ)`` (1e-12); ``cond_scale = exp(½ log λ^eff)``."""
    mean, persistent, transitory, rw = comp
    static = _static(mean, persistent, transitory, rw)
    draw = _draw(static, seed=7)
    rng = np.random.default_rng(1)
    hist, t_last = rng.normal(size=(LAGS, N)), 40.0
    shocks = _shocks(3, 11)
    out = _core(static, draw, hist, t_last, shocks)
    B = _B(static, draw)
    A = _unit_lower(np.asarray(draw.struct.a))
    eps = np.asarray(shocks.eps)
    log_lam = 2.0 * np.log(out.cond_scale)
    window = hist.copy()
    for h in range(H):
        x = regressors_from_history(window, t_last + h + 1, DET)
        lam = np.exp(log_lam[h])
        y = B.T @ x + np.linalg.solve(A, np.sqrt(lam) * eps[h])
        np.testing.assert_allclose(out.cond_mean[h], B.T @ x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out.paths[h], y, rtol=1e-12, atol=1e-12)
        Ainv = np.linalg.solve(A, np.eye(N))
        chol = np.linalg.cholesky(Ainv @ np.diag(lam) @ Ainv.T)
        np.testing.assert_allclose(out.cond_chol[h], chol, rtol=1e-12, atol=1e-14)
        window = np.vstack([window[1:], y])
    assert np.all(np.triu(out.cond_chol, 1) == 0.0)
    if transitory == "none":
        np.testing.assert_allclose(log_lam, out.log_lam_pers, rtol=0, atol=1e-14)


@pytest.mark.parametrize("comp", COMPOSITIONS[:4], ids=lambda c: "-".join(map(str, c)))
def test_volatility_futures_are_the_kernels_draw_ahead(comp):
    """``(log λ^pers, log λ^eff)`` are ``vol.draw_ahead_eq(vol_keys[i], vd_i, ·, H, gfut)``."""
    static = _static(*comp)
    draw = _draw(static, seed=2)
    shocks = _shocks(5, 4)
    out = _core(static, draw, np.zeros((LAGS, N)), 10.0, shocks)
    vol = compose(static).vol
    for i in range(N):
        vd = jax.tree.map(lambda x, i=i: jnp.asarray(x)[i], draw.vol)
        pers, eff = vol.draw_ahead_eq(shocks.vol_keys[i], vd, None, H, ())
        np.testing.assert_allclose(out.log_lam_pers[:, i], pers, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(2.0 * np.log(out.cond_scale[:, i]), eff, rtol=1e-14, atol=1e-14)


def test_simulate_paths_is_the_gaussian_part():
    """``simulate_paths`` with the core's shocks and variances reproduces the core (DESIGN §7.5)."""
    static = _static("linear_structural", "sv", "outlier")
    draw = _draw(static, seed=4)
    shocks = _shocks(9, 2)
    hist = np.ones((LAGS, N))
    out = _core(static, draw, hist, 20.0, shocks)
    program = compose(static)
    log_lam = 2.0 * np.log(out.cond_scale)
    paths, mean, chol = F.simulate_paths(draw, hist, 20.0, shocks.eps, log_lam, program)
    np.testing.assert_allclose(paths, out.paths, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(mean, out.cond_mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(chol, out.cond_chol, rtol=1e-13, atol=1e-13)
    # common random numbers: a shocked first-period ε moves only the paths, never the moments at h=1
    eps2 = np.asarray(shocks.eps).copy()
    eps2[0, 0] += 1.0
    p2, m2, c2 = F.simulate_paths(draw, hist, 20.0, eps2, log_lam, program)
    np.testing.assert_array_equal(np.asarray(m2)[0], np.asarray(mean)[0])
    np.testing.assert_array_equal(np.asarray(c2), np.asarray(chol))
    assert not np.allclose(np.asarray(p2)[1:], np.asarray(paths)[1:])


def _many(static, draw, M: int, horizon: int, seed: int = 0):
    """``M`` forecasts of one draw with independent keys (chunked executable)."""
    run = F.make_forecast(static, horizon, M)
    params = jax.tree.map(lambda x: np.broadcast_to(np.asarray(x), (M, *np.shape(x))), draw)
    ck = forecast_chain_keys(seed, 1)
    out = run(params, np.zeros((LAGS, static.shapes.n)), 50.0, ck[np.zeros(M, int)], np.arange(M))
    return jax.tree.map(np.asarray, out)


def test_draw_ahead_moments_ar_sv():
    """AR SV: ``h_{T+j} ~ N(μ + φ^j(h_T − μ), σ²(1 − φ^{2j})/(1 − φ²))`` (N03 §5.2 step 1)."""
    static = _static("zero", "sv", "none", n=2)
    draw = _draw(static, seed=5)
    M, Hm = 4000, 4
    out = _many(static, draw, M, Hm)
    core = draw.vol.core
    j = np.arange(1, Hm + 1)[:, None]
    mu, phi, sig, hT = (np.asarray(x) for x in (core.mu, core.phi, core.sigma, core.h_T))
    mean = mu + phi**j * (hT - mu)
    var = sig**2 * (1 - phi ** (2 * j)) / (1 - phi**2)
    h = out.log_lam_pers  # (M, H, n)
    z = (h.mean(axis=0) - mean) / np.sqrt(var / M)
    assert np.all(np.abs(z) < 4.5), z
    ratio = h.var(axis=0, ddof=1) / var
    assert np.all(np.abs(ratio - 1.0) < 6 * np.sqrt(2.0 / M)), ratio
    np.testing.assert_allclose(2.0 * np.log(out.cond_scale), h, rtol=1e-14, atol=1e-14)


def test_draw_ahead_moments_rw_sv_and_constant():
    """RW SV: ``h_{T+j} − h_T ~ N(0, jσ²)``; Constant: ``log λ`` unchanged."""
    static = _static("zero", "sv", "none", rw=True, n=2)
    draw = _draw(static, seed=6)
    M = 4000
    out = _many(static, draw, M, 3)
    core = draw.vol.core
    d = out.log_lam_pers - np.asarray(core.h_T)
    j = np.arange(1, 4)[:, None]
    assert np.all(np.abs(d.mean(axis=0)) < 4.5 * np.sqrt(j * core.sigma**2 / M))
    ratio = d.var(axis=0, ddof=1) / (j * np.asarray(core.sigma) ** 2)
    assert np.all(np.abs(ratio - 1.0) < 6 * np.sqrt(2.0 / M))
    cstatic = _static("zero", "constant", "none", n=2)
    cdraw = _draw(cstatic, seed=6)
    cout = _many(cstatic, cdraw, 8, 3)
    np.testing.assert_array_equal(
        cout.log_lam_pers, np.broadcast_to(np.asarray(cdraw.vol.core.log_lam), (8, 3, 2))
    )


def test_draw_ahead_outlier_frequency_and_t_scale():
    """Outliers: ``P(o > 1) = p``, ``o | o > 1`` uniform on ``{2..K}``; standardized t:
    ``E τ = 1``, ``log τ = log(c/2) − loggamma(ν/2)`` (N03 §5.2 steps 2–3)."""
    p, M = 0.2, 6000
    static = _static("zero", "constant", "outlier", n=2)
    out = _many(static, _draw(static, seed=1, p=p), M, 3)
    log_o2 = 2.0 * np.log(out.cond_scale) - out.log_lam_pers  # 2 log o
    o = np.rint(np.exp(0.5 * log_o2)).astype(int)
    np.testing.assert_allclose(np.exp(0.5 * log_o2), o, rtol=1e-12)
    frac = (o > 1).mean()
    assert abs(frac - p) < 4.5 * np.sqrt(p * (1 - p) / o.size)
    big = o[o > 1]
    assert big.min() >= 2 and big.max() <= 20
    counts = np.bincount(big, minlength=21)[2:]
    expected = big.size / 19
    assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected))
    tstatic = _static("zero", "constant", "t", n=2)
    tout = _many(tstatic, _draw(tstatic, seed=1, nu=12.0), M, 3)
    tau = np.exp(2.0 * np.log(tout.cond_scale) - tout.log_lam_pers)
    var_tau = 1.0 / (12.0 / 2 - 2)  # IG(ν/2, (ν−2)/2): Var = 1/(ν/2 − 2)
    assert abs(tau.mean() - 1.0) < 4.5 * np.sqrt(var_tau / tau.size)


def test_chunked_executable_equals_the_core_and_checks_its_size():
    static = _static("linear_reduced", "sv", "outlier_t")
    draws = [_draw(static, seed=s) for s in range(4)]
    params = jax.tree.map(lambda *x: np.stack([np.asarray(v) for v in x]), *draws)
    ck = forecast_chain_keys(17, 2)
    c, d = np.array([0, 1, 1, 0]), np.array([3, 3, 8, 120])
    hist = np.arange(LAGS * N, dtype=float).reshape(LAGS, N) / 10
    out = F.make_forecast(static, H, 4)(params, hist, 9.0, ck[c], d)
    for s in range(4):
        ref = _core(static, draws[s], hist, 9.0, forecast_shocks(ck[c[s]], jnp.uint32(d[s]), H, N))
        for name in F.ForecastOut._fields:
            np.testing.assert_allclose(getattr(out, name)[s], getattr(ref, name), rtol=1e-13)
    with pytest.raises(ValueError, match="chunks of 4"):
        F.make_forecast(static, H, 4)(params, hist, 9.0, ck[c[:3]], d[:3])
    with pytest.raises(ValueError, match="positive"):
        F.make_forecast(static, 0, 4)


def test_forecast_shocks_follow_the_key_layout():
    """``key(fseed) → FORECAST → c → d → Ahead.X → i`` (DESIGN §5.6), as ``keys.forecast_key``."""
    fseed, c, d = 123, 1, 7
    sh = forecast_shocks(forecast_chain_keys(fseed, 2)[c], jnp.uint32(d), H, N)
    kd = jax.random.key_data
    for i in range(N):
        np.testing.assert_array_equal(
            sh.eps[:, i], random.normal(KY.forecast_key(fseed, c, d, Ahead.SHOCK, i), (H,))
        )
        np.testing.assert_array_equal(
            kd(sh.vol_keys[i]), kd(KY.forecast_key(fseed, c, d, Ahead.VOL, i))
        )
    assert isinstance(sh, F.ForecastShocks)


def test_select_draws_rule():
    """DESIGN §7.1: ``⌈S/C⌉`` evenly spaced draws per chain, else every draw ``R`` times; the
    rule of ``storage.forecast_selection`` (shared with the in-loop hook)."""
    pos, R = F.select_draws(10, 2, 5)
    np.testing.assert_array_equal(pos, [0, 3, 6])
    assert R == 1
    pos, R = F.select_draws(10, 4, 1000)
    np.testing.assert_array_equal(pos, np.arange(10))
    assert R == 25
    for stored, chains, draws in [(7, 3, 20), (100, 4, 1000), (2500, 4, 1000), (5, 1, 5)]:
        pos, R = F.select_draws(stored, chains, draws)
        total = chains * pos.size * R
        assert draws <= total < draws + (chains if R == 1 else chains * stored)
        assert np.all(np.diff(pos) > 0) and pos[0] == 0 and pos[-1] < stored
        idx = np.arange(stored) * 3
        ref, ref_R = forecast_selection(chains, idx, draws)
        np.testing.assert_array_equal(idx[pos], ref)
        assert R == ref_R
    with pytest.raises(ValueError, match="positive"):
        F.select_draws(10, 0, 5)


# --------------------------------------------------------------------------- on an MCMC fit


@pytest.fixture(scope="module")
def fit(sim_data):
    return mf.shared_fit(sim_data)


def test_same_seed_gives_identical_paths_and_transitory_scales(fit):
    """Same ``fseed`` → identical paths, ``o`` and ``τ`` (``cond_scale``); a ``Generator``
    advances; another seed differs (N03 §5.3 requirement 1)."""
    a = fit.forecast(6, draws=40, seed=11)
    b = fit.forecast(6, draws=40, seed=11)
    for name in ("paths", "cond_mean", "cond_chol", "cond_scale", "cond_scale_persistent"):
        np.testing.assert_array_equal(a.dataset[name].values, b.dataset[name].values)
    assert a.dataset.attrs["fseed"] == int(np.random.default_rng(11).integers(0, 2**32))
    gen = np.random.default_rng(11)
    g1 = fit.forecast(6, draws=40, seed=gen)
    np.testing.assert_array_equal(g1.stacked(), a.stacked())
    g2 = fit.forecast(6, draws=40, seed=gen)
    assert not np.array_equal(g2.stacked(), a.stacked())
    assert g1.dataset.attrs["seed"] != g2.dataset.attrs["seed"]
    assert not np.array_equal(fit.forecast(6, draws=40, seed=12).stacked(), a.stacked())


def test_post_hoc_forecast_recomputed_from_stored_draws_and_keys(fit):
    """Slice ``(c, j)`` of ``fit.forecast`` is the core on stored draw ``pos_j`` of chain ``c``
    with the keys of its kept-draw index (DESIGN §7.1); a larger request replicates every stored
    draw ``R`` times with keys ``d + r·D``."""
    fr = fit.forecast(4, draws=10, seed=3)
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    pos, R = F.select_draws(D, C, 10)
    assert R == 1 and fr.dataset.sizes["draw"] == pos.size == 5
    fseed = int(fr.dataset.attrs["fseed"])
    ck = forecast_chain_keys(fseed, C)
    program = fit.program
    core = jax.jit(lambda p, s: F.forecast_core(p, fit.engine_data.hist, fit.engine_data.t_last, s,
                                                program))  # fmt: skip
    ds = fr.dataset
    for c, j in [(0, 0), (1, 3), (0, 4)]:
        d = int(fit.draw_index[pos[j]])
        params = jax.tree.map(lambda x, c=c, w=pos[j]: np.asarray(x)[c, w], fit.param_draws)
        ref = core(params, forecast_shocks(ck[c], jnp.uint32(d), 4, 3))
        np.testing.assert_allclose(fr.paths.values[c, j], ref.paths, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fr.cond_chol.values[c, j], ref.cond_chol, rtol=1e-12)
        # accelerator variables (DESIGN §7.3): exp(½ log λ^eff) and exp(½ log λ^pers)
        np.testing.assert_allclose(ds["cond_scale"].values[c, j], ref.cond_scale, rtol=1e-12)
        np.testing.assert_allclose(
            ds["cond_scale_persistent"].values[c, j], np.exp(0.5 * ref.log_lam_pers), rtol=1e-12
        )
    big = fit.forecast(2, draws=C * D * 2 + 1, seed=3)
    assert big.dataset.sizes["draw"] == 3 * D and big.dataset.attrs["draws_per_parameter"] == 3
    fseed = int(big.dataset.attrs["fseed"])
    ck = forecast_chain_keys(fseed, C)
    D_kept = fit.settings.draws
    for slot in (0, 1, 2, 3 * D - 1):
        j, r = divmod(slot, 3)
        d = int(fit.draw_index[j]) + r * D_kept
        params = jax.tree.map(lambda x, j=j: np.asarray(x)[1, j], fit.param_draws)
        ref = core(params, forecast_shocks(ck[1], jnp.uint32(d), 2, 3))
        np.testing.assert_allclose(big.paths.values[1, slot], ref.paths, rtol=1e-12, atol=1e-12)
    # replicates of one draw share the conditional mean at h = 1 but not the shocks
    np.testing.assert_allclose(big.cond_mean.values[0, 0, 0], big.cond_mean.values[0, 1, 0])
    assert not np.allclose(big.paths.values[0, 0, 0], big.paths.values[0, 1, 0])


def _chunk_size(total: int) -> int:
    """The chunk size ``min(S_CHUNK, 2^⌈log₂ S⌉)`` of ``forecast`` for ``S`` items (§7.2)."""
    return min(F.S_CHUNK, 1 << (total - 1).bit_length())


def test_post_hoc_forecast_across_chunk_boundaries(fit, monkeypatch):
    """More paths than ``S_CHUNK`` (DESIGN §7.2: chunks of ``S_c``, the last one padded), with
    ``S_CHUNK`` patched to 16 (the production 1024 would need ≥ 1025 paths): the items on both
    sides of every chunk boundary, the first and the last are the core on their stored draw with
    the keys ``d + r·D_kept``, and the whole forecast equals the one-chunk forecast (1e-12: the
    vmap batch sizes differ, so not bitwise)."""
    Hc, chunk = 2, 16
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    draws = 2 * C * D + 1
    pos, R = F.select_draws(D, C, draws)
    total = C * pos.size * R
    assert R == 3 and _chunk_size(total) >= total  # one chunk unpatched
    one = fit.forecast(Hc, draws=draws, seed=3)
    monkeypatch.setattr(F, "S_CHUNK", chunk)
    assert _chunk_size(total) == chunk and total % chunk != 0  # several chunks, the last padded
    fr = fit.forecast(Hc, draws=draws, seed=3)
    assert int(fr.dataset.attrs["fseed"]) == int(one.dataset.attrs["fseed"])
    for name in ("paths", "cond_mean", "cond_chol", "cond_scale", "cond_scale_persistent"):
        np.testing.assert_allclose(
            fr.dataset[name].values, one.dataset[name].values, rtol=1e-12, atol=1e-12
        )
    ck = forecast_chain_keys(int(fr.dataset.attrs["fseed"]), C)
    D_kept = fit.settings.draws
    core = jax.jit(lambda p, s: F.forecast_core(p, fit.engine_data.hist, fit.engine_data.t_last, s,
                                                fit.program))  # fmt: skip
    items = {0, total - 1}
    for lo in range(chunk, total, chunk):
        items |= {lo - 1, lo}
    for item in sorted(items):  # items in (chain, draw, replicate) order
        c, slot = divmod(item, pos.size * R)
        j, r = divmod(slot, R)
        d = int(fit.draw_index[pos[j]]) + r * D_kept
        params = jax.tree.map(lambda x, c=c, w=pos[j]: np.asarray(x)[c, w], fit.param_draws)
        ref = core(params, forecast_shocks(ck[c], jnp.uint32(d), Hc, N))
        np.testing.assert_allclose(fr.paths.values[c, slot], ref.paths, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fr.cond_chol.values[c, slot], ref.cond_chol, rtol=1e-12)


def test_forecast_argument_errors(fit):
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)
    with pytest.raises(ValueError, match="draws"):
        fit.forecast(3, draws=0)


# --------------------------------------------------------------------------- in-loop (§5.8)


def _check_in_loop(fit, H_plan: int, draws: int, seed: int) -> None:
    """In-loop forecasts equal the post-hoc core on the stored draws they came from (1e-12),
    and ``fit.forecast(H, draws, seed)`` without the in-loop container reproduces them."""
    ilf = fit.in_loop_forecast
    assert ilf is not None and ilf.matches(H_plan, draws, seed)
    C = np.shape(fit.param_draws.struct.a)[0]
    ck = forecast_chain_keys(ilf.fseed, C)
    program = fit.program
    core = jax.jit(lambda p, s: F.forecast_core(p, fit.engine_data.hist, fit.engine_data.t_last, s,
                                                program))  # fmt: skip
    kept = np.asarray(ilf.draw_index).reshape(-1)
    key_index = np.asarray(getattr(ilf, "key_index", kept)).reshape(-1)
    out = ilf.out
    get = (lambda o, k: o[k]) if isinstance(out, dict) else getattr
    for c in range(C):
        for m in (0, kept.size // 2, kept.size - 1):
            where = int(np.searchsorted(fit.draw_index, kept[m]))
            params = jax.tree.map(lambda x, c=c, w=where: np.asarray(x)[c, w], fit.param_draws)
            ref = core(params, forecast_shocks(ck[c], jnp.uint32(key_index[m]), H_plan, 3))
            for name in ("paths", "cond_mean", "cond_chol"):
                np.testing.assert_allclose(
                    np.asarray(get(out, name))[c, m], getattr(ref, name), rtol=1e-12, atol=1e-12
                )
    # fit.forecast returns the in-loop result; a fresh post-hoc forecast reproduces it
    via_plan = fit.forecast(H_plan, draws=draws, seed=seed)
    assert via_plan.dataset.attrs["in_loop"] == 1
    fresh = copy.copy(fit)
    fresh.in_loop_forecast = None
    post = fresh.forecast(H_plan, draws=draws, seed=seed)
    assert post.dataset.attrs["in_loop"] == 0
    for name in ("paths", "cond_mean", "cond_chol", "cond_scale", "cond_A"):
        np.testing.assert_allclose(
            via_plan.dataset[name].values, post.dataset[name].values, rtol=1e-12, atol=1e-12
        )


def test_in_loop_forecast_equals_post_hoc_constant(sim_data):
    plan = FitPlan(horizon=4, draws=30, seed=5)
    m = mf.model("structural", Constant("ig"))
    _check_in_loop(mf.cached_fit(m, sim_data, plan=plan), 4, 30, 5)


@pytest.mark.slow
def test_in_loop_forecast_equals_post_hoc_sv(sim_data):
    """DESIGN §5.8: tested on Linear × SV (the outlier scale exercises ``o`` keys too)."""
    plan = FitPlan(horizon=6, draws=40, seed=5)
    m = mf.model("structural", SV(outliers=True))
    _check_in_loop(mf.cached_fit(m, sim_data, plan=plan), 6, 40, 5)


@pytest.fixture(scope="module")
def thinned_trend_fit(sim_data):
    """Constant("ig"), ``lags = 2``, ``deterministic = "ct"``, ``store_max = 20``: every
    ``store_thin = ⌈60/20⌉ = 3``-rd kept draw is stored, so the kept-draw index (the ``draw``
    coordinate) differs from the store position; with a ``FitPlan`` (in-loop forecasts)."""
    m = mf.model("structural", Constant("ig"), lags=2, deterministic="ct", store_max=20)
    return mf.cached_fit(m, sim_data, plan=FitPlan(horizon=4, draws=30, seed=5))


def test_in_loop_forecast_equals_post_hoc_with_thinned_storage(thinned_trend_fit):
    """Stage 2 amendment 2: in-loop and post-hoc forecasts are both keyed by the kept-draw index,
    not the store position (``store_thin = 3``)."""
    fit = thinned_trend_fit
    assert fit.settings.store_thin == 3
    np.testing.assert_array_equal(fit.draw_index, 3 * np.arange(20))
    _check_in_loop(fit, 4, 30, 5)


def test_trend_and_thinned_keys_match_the_phase1_simulator(thinned_trend_fit, sim_data):
    """Slice ``(c, j)`` of a post-hoc forecast is Phase 1's ``simulate_linear_var(B, L,
    values[-p:], H, "ct", len(values))`` (trend ``len(values) + h`` at ``T + h``) with
    ``L = A⁻¹ diag(exp(½ log λ))`` and the shocks of kept-draw index ``draw_index[pos_j]``
    (``≠ pos_j`` here), 1e-12."""
    fit, Hf = thinned_trend_fit, 3
    fr = fit.forecast(Hf, draws=8, seed=21)
    assert fr.dataset.attrs["in_loop"] == 0
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    pos, R = F.select_draws(D, C, 8)
    assert R == 1 and np.all(fit.draw_index[pos[1:]] != pos[1:])
    ck = forecast_chain_keys(int(fr.dataset.attrs["fseed"]), C)
    values = sim_data.values
    for c in range(C):
        for j, w in enumerate(pos):
            params = jax.tree.map(lambda x, c=c, w=w: np.asarray(x)[c, w], fit.param_draws)
            B = _B(fit.static, params)
            A = _unit_lower(np.asarray(params.struct.a))
            L = np.linalg.solve(A, np.diag(np.exp(0.5 * np.asarray(params.vol.core.log_lam))))
            eps = np.asarray(forecast_shocks(ck[c], jnp.uint32(fit.draw_index[w]), Hf, N).eps)
            paths, mean, chol = simulate_linear_var(
                B[None], L[None], values[-2:], Hf, "ct", len(values), np.random.default_rng(0),
                shocks=eps[None],
            )  # fmt: skip
            np.testing.assert_allclose(fr.cond_mean.values[c, j], mean[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fr.paths.values[c, j], paths[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(fr.cond_chol.values[c, j], chol[0], rtol=1e-12, atol=1e-15)


def test_engine_level_fit_without_data_or_model(fit):
    """A fit from ``fit_mcmc`` without ``VARData`` or model (validation, parity): no level
    paths, a synthetic annual origin at ordinal ``T_eff − 1``, ``model = "mcmc"``, integer dates."""
    bare = copy.copy(fit)
    bare.data, bare.model = None, None
    bare.periods = pd.RangeIndex(int(np.asarray(fit.engine_data.T_eff)))
    fr = bare.forecast(3, draws=4, seed=1)
    assert fr.paths_level is None and fr.dataset.attrs["model"] == "mcmc"
    assert "spec" not in fr.dataset.attrs and fr.dataset.attrs["tcodes"] == "{}"
    assert fr.origin == pd.Period(ordinal=len(bare.periods) - 1, freq="Y")
    np.testing.assert_array_equal(fr.stacked(), fit.forecast(3, draws=4, seed=1).stacked())
    c1, L1 = bare.draw_parameters(3, date=len(bare.periods) - 1)
    c2, L2 = fit.draw_parameters(3)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(L1, L2)
    _, L3 = bare.draw_parameters(3, date=5)
    _, L4 = fit.draw_parameters(3, date=fit.periods[5])
    np.testing.assert_array_equal(L3, L4)
    with pytest.raises(ValueError, match="data row"):
        bare.draw_parameters(3, date="2001Q1")
