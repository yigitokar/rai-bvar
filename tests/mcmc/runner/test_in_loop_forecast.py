"""In-loop forecasts (DESIGN §5.8, Stage 2 amendment): the ``sample`` executable calls the forecast
core on exactly the kept states that DESIGN §7.1 selects post hoc (``⌈S/C⌉`` stored draws per
chain, evenly spaced; every stored draw with ``⌈S/(C·D_s)⌉`` paths when that exceeds them), with
the post-hoc keys ``key(fseed) → FORECAST → c → d → Ahead → i``, ``d`` the kept-draw index.

The forecast core belongs to WP17; these tests install the double ``toys.toy_forecast_core``."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines.mcmc import forecast as forecast_mod
from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.config import StoreStatic
from rai_bvar.engines.mcmc.keys import Ahead
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.runner import (
    forecast_chain_keys,
    forecast_seed,
    forecast_shocks,
    make_runner,
    run_chains,
)
from rai_bvar.engines.mcmc.storage import forecast_selection
from rai_bvar.settings import FitPlan, MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

H = 3
BASE = toys.toy_static("linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=3)
STATIC = dataclasses.replace(BASE, store=StoreStatic(in_loop_forecast=H))
PLAN = FitPlan(horizon=H, draws=5, seed=11)
FSEED_11 = 574671950
"""``fseed`` of ``FitPlan(seed=11)``: ``int(np.random.default_rng(11).integers(0, 2**32))``
(DESIGN §7.1), pinned as a literal so a different derivation cannot pass unnoticed."""
SETTINGS = MCMCSettings(chains=2, draws=10, burn=3)  # ⌈5/2⌉ = 3 per chain: kept draws 0, 3, 6


@pytest.fixture(autouse=True)
def forecast_double(monkeypatch):
    monkeypatch.setattr(forecast_mod, "forecast_core", toys.toy_forecast_core, raising=False)


def _fit(static=STATIC, plan=PLAN, settings=SETTINGS, data=None):
    data = toys.toy_data(static, 20, 5) if data is None else data
    return fit_mcmc(
        static, data, toys.toy_prior(static), settings, Runtime(chunk=static.chunk), seed=0, salt=1,
        meta=FitMeta(variables=("a", "b")), plan=plan, _overrides=toys.toy_overrides(static),
    )  # fmt: skip


def section_7_1(C, stored, S):
    """DESIGN §7.1 written out: ``m = ⌈S/C⌉`` stored draws per chain, evenly spaced from the first
    (stored positions ``⌊j·D_s/m⌋``); when ``m > D_s``, every stored draw with ``⌈S/(C·D_s)⌉``
    paths. Returns (kept-draw indices, paths per draw)."""
    D_s = len(stored)
    m = math.ceil(S / C)
    if m <= D_s:
        return [int(stored[(j * D_s) // m]) for j in range(m)], 1
    return [int(d) for d in stored], math.ceil(S / (C * D_s))


_CORE = {}


def _post_hoc(fit, c, p, d, fseed=FSEED_11):
    """The double applied after the run to stored draw ``p`` of chain ``c``, with the shocks of
    forecast-key index ``d``."""
    data = fit.engine_data
    fkey = forecast_chain_keys(fseed, fit.settings.chains)[c]
    params = jax.tree.map(lambda x: x[c, p], fit.param_draws)
    shocks = forecast_shocks(fkey, jnp.uint32(d), H, 2)
    if fit.program not in _CORE:
        _CORE[fit.program] = jax.jit(
            lambda p_, h, t, s, prog=fit.program: toys.toy_forecast_core(p_, h, t, s, prog)
        )
    return _CORE[fit.program](params, data.hist, data.t_last, shocks)


def _assert_equals_post_hoc(fit):
    """Every in-loop path equals the post-hoc double on the stored parameters of its kept draw
    (looked up by the posterior ``draw`` coordinate) with the key index ``d + q·D``."""
    ilf = fit.in_loop_forecast
    coord = list(fit.posterior["posterior"]["draw"].values)
    for c in range(fit.settings.chains):
        for m, (d, kd) in enumerate(zip(ilf.draw_index, ilf.key_index, strict=True)):
            ref = _post_hoc(fit, c, coord.index(int(d)), int(kd))
            for name, value in ref.items():
                np.testing.assert_allclose(ilf.out[name][c, m], value, rtol=1e-12, atol=1e-12)


def test_forecast_seed_follows_design_7_1():
    assert forecast_seed(11) == FSEED_11 == int(np.random.default_rng(11).integers(0, 2**32))


@pytest.mark.parametrize("thin", [1, 2])
def test_in_loop_forecasts_equal_post_hoc_forecasts(thin):
    """Selected kept draws ``0, 3, 6`` (chunk 3 does not divide the 10 draws; they sit in
    chunks 0, 1, 2) and the post-hoc double on the stored parameters with the same keys agree to
    1e-12. With ``thin = 2`` the kept-draw index ``c·chunk + j`` differs from the sweep offset
    ``c·chunk·thin + j``, so a key built from the sweep offset fails."""
    fit = _fit(settings=dataclasses.replace(SETTINGS, thin=thin))
    ilf = fit.in_loop_forecast
    assert ilf is not None and ilf.matches(H, 5, 11) and not ilf.matches(H, 5, 12)
    assert ilf.fseed == FSEED_11 and ilf.per_chain == 3 and ilf.paths_per_draw == 1
    np.testing.assert_array_equal(ilf.draw_index, [0, 3, 6])
    np.testing.assert_array_equal(ilf.key_index, [0, 3, 6])
    assert ilf.out["paths"].shape == (2, 3, H, 2) and ilf.out["cond_chol"].shape == (2, 3, H, 2, 2)
    assert fit.metadata["in_loop_forecast_per_chain"] == 3
    assert fit.metadata["in_loop_forecast_paths_per_draw"] == 1
    assert fit.metadata["in_loop_forecast_fseed"] == FSEED_11
    _assert_equals_post_hoc(fit)
    # distinct draws and chains give distinct shocks
    assert not np.allclose(ilf.out["paths"][0, 0], ilf.out["paths"][0, 1])
    assert not np.allclose(ilf.out["paths"][0, 0], ilf.out["paths"][1, 0])


@pytest.mark.parametrize(("C", "D", "S"), [(2, 3, 5), (2, 10, 15), (3, 10, 7)])
def test_in_loop_selection_is_the_design_7_1_selection(C, D, S):
    """``C·D/S`` not an integer: the in-loop forecasts use exactly the §7.1 draws, ``⌈S/C⌉`` per
    chain (``≥ S`` paths in all, fewer than ``C`` extra), never fewer than ``S`` paths."""
    fit = _fit(
        plan=FitPlan(horizon=H, draws=S, seed=11), settings=MCMCSettings(chains=C, draws=D, burn=2)
    )
    ilf = fit.in_loop_forecast
    selected, R = section_7_1(C, fit.draw_index, S)
    assert R == 1 and ilf.paths_per_draw == 1
    np.testing.assert_array_equal(ilf.draw_index, selected)
    paths = ilf.out["paths"]
    assert paths.shape == (C, len(selected), H, 2)
    total = paths.shape[0] * paths.shape[1]
    assert S <= total < S + C
    assert np.all(np.any(paths != 0.0, axis=(-2, -1)))  # every slice is a computed path
    _assert_equals_post_hoc(fit)


def test_more_paths_than_stored_draws_use_independent_shocks():
    """``C = 2``, ``D = 3``, ``S = 10``: ``⌈10/2⌉ = 5`` exceeds the 3 stored draws, so every
    stored draw gets ``⌈10/6⌉ = 2`` paths (§7.1), path ``q`` keyed by ``d + q·D``; each
    ``(chain, draw)`` slice is one path from one parameter draw."""
    fit = _fit(
        plan=FitPlan(horizon=H, draws=10, seed=11), settings=MCMCSettings(chains=2, draws=3, burn=2)
    )
    ilf = fit.in_loop_forecast
    stored, R = section_7_1(2, fit.draw_index, 10)
    assert (R, ilf.paths_per_draw, ilf.per_chain) == (2, 2, 3) and stored == [0, 1, 2]
    np.testing.assert_array_equal(ilf.draw_index, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(ilf.replicate, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(ilf.key_index, [0, 3, 1, 4, 2, 5])
    assert ilf.out["paths"].shape == (2, 6, H, 2)
    _assert_equals_post_hoc(fit)
    assert not np.allclose(ilf.out["paths"][0, 0], ilf.out["paths"][0, 1])
    assert fit.metadata["in_loop_forecast_paths_per_draw"] == 2


def test_forecast_key_is_the_kept_draw_index_with_store_thin():
    """``store_thin = ⌈10/4⌉ = 3``: stored kept draws 0, 3, 6, 9 (the posterior ``draw``
    coordinate). ``S = 4``, ``C = 2``: stored positions 0, 2, i.e. kept draws 0, 6, and the keys
    use ``d = 0, 6`` (the kept-draw index, DESIGN §5.6), not the store positions 0, 2."""
    settings = MCMCSettings(chains=2, draws=10, burn=3, thin=2, store_max=4)
    fit = _fit(plan=FitPlan(horizon=H, draws=4, seed=11), settings=settings)
    coord = fit.posterior["posterior"]["draw"].values
    np.testing.assert_array_equal(coord, [0, 3, 6, 9])
    ilf = fit.in_loop_forecast
    np.testing.assert_array_equal(ilf.draw_index, [0, 6])
    np.testing.assert_array_equal(ilf.key_index, coord[[0, 2]])
    _assert_equals_post_hoc(fit)
    by_position = _post_hoc(fit, 0, 2, 2)  # the stored draw of kept draw 6, keyed by its slot
    assert not np.allclose(ilf.out["paths"][0, 1], by_position["paths"])


def test_forecast_shocks_follow_the_documented_key_layout():
    """``forecast_shocks`` derives exactly ``keys.forecast_key(fseed, c, d, Ahead.X, i)``."""
    fseed, c, d, n = FSEED_11, 1, 4, 2
    fkey = forecast_chain_keys(fseed, 2)[c]
    sh = forecast_shocks(fkey, jnp.uint32(d), H, n)
    data = jax.random.key_data

    def ref(kind, i):
        return keys.forecast_key(fseed, c, d, kind, i)

    for i in range(n):
        np.testing.assert_array_equal(data(sh.vol_keys[i]), data(ref(Ahead.VOL, i)))
        np.testing.assert_array_equal(data(sh.mean_keys[i]), data(ref(Ahead.MEAN, i)))
        eps_i = jax.random.normal(ref(Ahead.SHOCK, i), (H,), dtype=jnp.float64)
        np.testing.assert_array_equal(sh.eps[:, i], eps_i)
    np.testing.assert_array_equal(data(sh.glob_key), data(ref(Ahead.GLOBAL, 0)))
    assert sh.eps.shape == (H, n)


def test_unselected_draws_skip_the_forecast_core():
    """The selection mask is shared by the chains, so under ``vmap`` the hook stays a ``cond``
    (one more than without the hook) instead of a ``select`` over both branches."""
    data, prior = toys.toy_data(STATIC, 20, 1), toys.toy_prior(STATIC)
    u = np.uint32

    def conds(static, *extra):
        fns = make_runner(static, overrides=toys.toy_overrides(static))
        cache = fns.prepare(data, prior)
        st = fns.init(keys.init_keys(0, 1, 2), data, prior, cache)
        ck = keys.chain_keys(keys.fit_root(0, 1), 2)
        args = (ck, st, data, prior, cache, u(0), u(1), np.int32(2), *extra)
        return str(jax.make_jaxpr(fns.sample)(*args)).count("cond[")

    sel = np.array([True, False, True])
    assert conds(STATIC, forecast_chain_keys(5, 2), u(0), sel, u(10)) == conds(BASE) + 1


def test_in_loop_forecasts_do_not_depend_on_the_chunk():
    a = _fit().in_loop_forecast.out
    b = _fit(dataclasses.replace(STATIC, chunk=4)).in_loop_forecast.out
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def test_in_loop_forecasts_with_a_fit_axis_equal_looped_fits():
    datas = [toys.toy_data(STATIC, T, sd) for T, sd in ((20, 1), (17, 2))]
    prior = toys.toy_prior(STATIC)
    ov = toys.toy_overrides(STATIC)
    stack = lambda *xs: np.stack(xs)  # noqa: E731
    fns_f = make_runner(STATIC, fit_axis=True, overrides=ov)
    out_f = run_chains(
        fns_f, STATIC, jax.tree.map(stack, *datas), jax.tree.map(stack, prior, prior), SETTINGS,
        Runtime(chunk=STATIC.chunk), seed=[3, 4], salt=[5, 6], plan=PLAN,
    )  # fmt: skip
    fns = make_runner(STATIC, overrides=ov)
    for f, (data, seed, salt) in enumerate(zip(datas, (3, 4), (5, 6), strict=True)):
        out = run_chains(
            fns, STATIC, data, prior, SETTINGS, Runtime(chunk=STATIC.chunk), seed=seed, salt=salt,
            plan=PLAN,
        )  # fmt: skip
        for x, y in zip(
            jax.tree.leaves(out_f.store.stream("forecast")),
            jax.tree.leaves(out.store.stream("forecast")),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(x)[f], y, rtol=1e-10, atol=1e-10)


def test_plan_consistency_checks():
    with pytest.raises(ValueError, match="needs plan=FitPlan"):
        _fit(plan=None)
    with pytest.raises(ValueError, match="horizon"):
        _fit(plan=FitPlan(horizon=H + 1, draws=5))
    with pytest.raises(ValueError, match="in_loop_forecast"):
        _fit(BASE)  # a plan without the static hook


def test_forecast_selection_rules():
    idx, R = forecast_selection(4, np.arange(1000), 1000)
    assert R == 1 and idx.size == 250
    np.testing.assert_array_equal(idx, np.arange(0, 1000, 4))
    idx, R = forecast_selection(3, np.arange(10) * 2, 7)  # ⌈7/3⌉ = 3 of 10 stored draws
    np.testing.assert_array_equal(idx, [0, 6, 12])
    assert R == 1
    idx, R = forecast_selection(2, [0, 3, 6], 13)  # ⌈13/2⌉ = 7 > 3: ⌈13/6⌉ = 3 paths each
    np.testing.assert_array_equal(idx, [0, 3, 6])
    assert R == 3
    for C, D_s, S in ((2, 10, 15), (3, 7, 20), (4, 5, 21), (5, 3, 4)):
        idx, R = forecast_selection(C, np.arange(D_s), S)
        assert (list(idx), R) == section_7_1(C, list(range(D_s)), S)
        assert C * idx.size * R >= S
    with pytest.raises(ValueError, match="positive"):
        forecast_selection(2, np.arange(10), 0)


def test_no_plan_no_forecast_stream():
    fit = _fit(BASE, plan=None)
    assert fit.in_loop_forecast is None and "forecast" not in fit.store.plan.streams
