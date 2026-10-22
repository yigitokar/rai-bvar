"""In-loop forecasts (DESIGN §5.8): the ``sample`` executable calls the forecast core on every
``⌈C·D/S⌉``-th kept state with the post-hoc keys ``key(fseed) → FORECAST → c → d → Ahead → i``.

The forecast core belongs to WP17; these tests install the double ``toys.toy_forecast_core``."""

from __future__ import annotations

import dataclasses

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
from rai_bvar.engines.mcmc.storage import in_loop_stride
from rai_bvar.settings import FitPlan, MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

H = 3
BASE = toys.toy_static("linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=3)
STATIC = dataclasses.replace(BASE, store=StoreStatic(in_loop_forecast=H))
PLAN = FitPlan(horizon=H, draws=5, seed=11)
SETTINGS = MCMCSettings(chains=2, draws=10, burn=3)  # stride ⌈2·10/5⌉ = 4: kept draws 0, 4, 8


@pytest.fixture(autouse=True)
def forecast_double(monkeypatch):
    monkeypatch.setattr(forecast_mod, "forecast_core", toys.toy_forecast_core, raising=False)


def _fit(static=STATIC, plan=PLAN, settings=SETTINGS, data=None):
    data = toys.toy_data(static, 20, 5) if data is None else data
    return fit_mcmc(
        static, data, toys.toy_prior(static), settings, Runtime(chunk=static.chunk), seed=0, salt=1,
        meta=FitMeta(variables=("a", "b")), plan=plan, _overrides=toys.toy_overrides(static),
    )  # fmt: skip


def _post_hoc(fit, c, d):
    """The double applied after the run to stored draw ``d`` of chain ``c`` (store_thin = 1)."""
    data = fit.engine_data
    fkey = forecast_chain_keys(forecast_seed(PLAN.seed), fit.settings.chains)[c]
    params = jax.tree.map(lambda x: x[c, d], fit.param_draws)
    shocks = forecast_shocks(fkey, jnp.uint32(d), H, 2)
    core = jax.jit(lambda p, h, t, s: toys.toy_forecast_core(p, h, t, s, fit.program))
    return core(params, data.hist, data.t_last, shocks)


def test_in_loop_forecasts_equal_post_hoc_forecasts():
    """Selected kept draws ``0, 4, 8`` (across chunks of 3: positions 0, 1, 2 of chunks 0, 1, 2)
    and the post-hoc double on the stored parameters with the same keys agree to 1e-12."""
    fit = _fit()
    ilf = fit.in_loop_forecast
    assert ilf is not None and ilf.matches(H, 5, 11) and not ilf.matches(H, 5, 12)
    assert ilf.stride == in_loop_stride(2, 10, 5) == 4 and ilf.fseed == forecast_seed(11)
    np.testing.assert_array_equal(ilf.draw_index, [0, 4, 8])
    assert ilf.out["paths"].shape == (2, 3, H, 2) and ilf.out["cond_chol"].shape == (2, 3, H, 2, 2)
    assert fit.metadata["in_loop_forecast_stride"] == 4
    for c in range(2):
        for m, d in enumerate(ilf.draw_index):
            ref = _post_hoc(fit, c, int(d))
            for name, value in ref.items():
                np.testing.assert_allclose(ilf.out[name][c, m], value, rtol=1e-12, atol=1e-12)
    # distinct draws give distinct shocks
    assert not np.allclose(ilf.out["paths"][0, 0], ilf.out["paths"][0, 1])


def test_forecast_shocks_follow_the_documented_key_layout():
    """``forecast_shocks`` derives exactly ``keys.forecast_key(fseed, c, d, Ahead.X, i)``."""
    fseed, c, d, n = forecast_seed(11), 1, 4, 2
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
    """The selection predicate is shared by the chains, so under ``vmap`` the hook stays a
    ``cond`` (one more than without the hook) instead of a ``select`` over both branches."""
    data, prior = toys.toy_data(STATIC, 20, 1), toys.toy_prior(STATIC)
    u = np.uint32

    def conds(static, *extra):
        fns = make_runner(static, overrides=toys.toy_overrides(static))
        cache = fns.prepare(data, prior)
        st = fns.init(keys.init_keys(0, 1, 2), data, prior, cache)
        ck = keys.chain_keys(keys.fit_root(0, 1), 2)
        args = (ck, st, data, prior, cache, u(0), u(1), np.int32(2), *extra)
        return str(jax.make_jaxpr(fns.sample)(*args)).count("cond[")

    assert conds(STATIC, forecast_chain_keys(5, 2), u(0), u(4)) == conds(BASE) + 1


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
    with pytest.raises(ValueError, match="exceeds"):
        _fit(plan=FitPlan(horizon=H, draws=21))
    with pytest.raises(ValueError, match="in_loop_forecast"):
        _fit(BASE)  # a plan without the static hook
    assert in_loop_stride(4, 1000, 1000) == 4 and in_loop_stride(3, 10, 30) == 1
    with pytest.raises(ValueError, match="positive"):
        in_loop_stride(2, 10, 0)


def test_no_plan_no_forecast_stream():
    fit = _fit(BASE, plan=None)
    assert fit.in_loop_forecast is None and "forecast" not in fit.store.plan.streams
