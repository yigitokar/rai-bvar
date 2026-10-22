"""Chains on several CPU devices (DESIGN §5.2, Stage 2 amendment): ``shard_map`` equals the
single-device run bit for bit with ``chains="map"`` (one identical one-chain program per chain)
and to rounding with ``chains="vmap"``, whose per-device batch size changes how XLA vectorises
reductions (on x86 Linux even the toy kernels differ in the last bit; ``rtol 1e-12`` for toys,
``1e-9`` for the production SV kernels on short runs). ``Runtime.devices`` therefore defaults to
one device, and sharding is opt-in."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import jax
import pytest

from rai_bvar.engines.mcmc.runner import resolve_devices
from rai_bvar.settings import Runtime

TESTS_MCMC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(
    """
    import dataclasses, sys
    sys.path.insert(0, {path!r})
    import rai_bvar
    rai_bvar.set_cpu_devices(4)
    import jax
    import numpy as np
    from runner import toys
    from rai_bvar.engines.mcmc import keys
    from rai_bvar.engines.mcmc.runner import make_runner, run_chains
    from rai_bvar.settings import MCMCSettings, Runtime

    assert jax.local_device_count() == 4

    def same(a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def close(a, b):  # another vmap batch size: equal to rounding (DESIGN §5.6)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            x, y = np.asarray(x), np.asarray(y)
            if np.issubdtype(x.dtype, np.floating):
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
            else:
                np.testing.assert_array_equal(x, y)

    toys.install_monitor_doubles()
    static = toys.toy_static("linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5)
    data, prior = toys.toy_data(static, 20, 1), toys.toy_prior(static)
    # vmap over chains: 2 chains per device; map over chains: 1 chain per device
    for chains, C in (("vmap", 8), ("map", 4)):
        fns = toys.cached_runner(static, toys.toy_overrides(static), chains=chains)
        settings = MCMCSettings(chains=C, draws=7, burn=6, thin=2)
        one = run_chains(fns, static, data, prior, settings, Runtime(chunk=5, devices=1), seed=3, salt=4)
        four = run_chains(fns, static, data, prior, settings, Runtime(chunk=5, devices=4), seed=3, salt=4)
        auto = run_chains(fns, static, data, prior, settings, Runtime(chunk=5, devices="auto"), seed=3,
                          salt=4)
        default = run_chains(fns, static, data, prior, settings, Runtime(chunk=5), seed=3, salt=4)
        sharded = "shard_map(4)" + ("" if chains == "vmap" else "+map")
        assert one.chain_method == chains and four.chain_method == sharded == auto.chain_method
        assert default.devices == 1 and default.chain_method == chains  # opt-in sharding
        for other in (four, auto, default):
            check = same if (chains == "map" or other is default) else close
            check(one.store.stream("monitor"), other.store.stream("monitor"))
            check(one.store.stream("params"), other.store.stream("params"))
            check(one.final_state, other.final_state)
    print("TOYS-OK")

    # in-loop forecasts (DESIGN §5.8) under shard_map: the hook's extra arguments are sharded too
    from rai_bvar.engines.mcmc import forecast
    from rai_bvar.engines.mcmc.config import StoreStatic
    from rai_bvar.settings import FitPlan

    forecast.forecast_core = toys.toy_forecast_core
    fstatic = dataclasses.replace(static, store=StoreStatic(in_loop_forecast=3))
    fns = make_runner(fstatic, overrides=toys.toy_overrides(fstatic))
    settings = MCMCSettings(chains=8, draws=7, burn=6, thin=2)
    plan = FitPlan(horizon=3, draws=10, seed=2)
    one = run_chains(fns, fstatic, data, prior, settings, Runtime(chunk=5, devices=1), seed=3, salt=4,
                     plan=plan)
    four = run_chains(fns, fstatic, data, prior, settings, Runtime(chunk=5, devices=4), seed=3, salt=4,
                      plan=plan)
    assert four.chain_method == "shard_map(4)"
    close(one.store.stream("forecast"), four.store.stream("forecast"))
    close(one.store.stream("params"), four.store.stream("params"))
    # several paths per stored draw (DESIGN §7.1: ⌈200/8⌉ = 25 > 7 stored draws → 4 paths each)
    many = FitPlan(horizon=3, draws=200, seed=2)
    one = run_chains(fns, fstatic, data, prior, settings, Runtime(chunk=5, devices=1), seed=3, salt=4,
                     plan=many)
    four = run_chains(fns, fstatic, data, prior, settings, Runtime(chunk=5, devices=4), seed=3, salt=4,
                      plan=many)
    assert one.forecast["paths_per_draw"] == four.forecast["paths_per_draw"] == 4
    close(one.store.stream("forecast"), four.store.stream("forecast"))
    print("FORECAST-OK")

    from rai_bvar.engines.mcmc.compose import compose
    from rai_bvar.engines.mcmc.runner import _sharded
    from rai_bvar.validation.joint import joint_spec

    spec = joint_spec("structural_sv")
    static = dataclasses.replace(spec.static, chunk=5)
    try:
        compose(static)
    except NotImplementedError:
        print("PRODUCTION-SKIPPED")
        raise SystemExit(0)
    key = jax.random.key(7)
    data = jax.tree.map(np.asarray, spec.simulate(jax.random.fold_in(key, 1), spec.prior_sample(key)))
    # production kernels: the sharded run agrees to rounding, rtol 1e-9 (DESIGN §5.2, §5.6 Stage 2
    # amendment: XLA vectorises the SV reductions differently for another per-device batch size)
    for chains in ("vmap", "map"):
        fns = make_runner(static, chains=chains)
        cache = fns.prepare(data, spec.prior)
        st = fns.init(keys.init_keys(3, 4, 4), data, spec.prior, cache)
        ck = keys.chain_keys(keys.fit_root(3, 4), 4)
        host = jax.device_get(st)
        ref = fns.burn(ck, jax.device_put(host), data, spec.prior, cache, np.uint32(0), np.int32(2))
        burn4, _, csh, rep = _sharded(fns.chain, 4, chains)
        args = [jax.device_put(t, rep) for t in (data, spec.prior, cache)]
        sharded = burn4(jax.device_put(ck, csh), jax.device_put(host, csh), *args,
                        np.uint32(0), np.int32(2))
        bitwise = True
        for x, y in zip(jax.tree.leaves(jax.device_get(ref)), jax.tree.leaves(jax.device_get(sharded))):
            x, y = np.asarray(x), np.asarray(y)
            bitwise &= bool(np.array_equal(x, y))
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12)
        print("PRODUCTION-" + chains + "-bitwise=" + str(bitwise))
    print("PRODUCTION-OK")
    """
)


def test_shard_map_matches_vmap_in_subprocess():
    """Toys: bitwise with ``map``, ``rtol 1e-12`` with ``vmap`` (in-loop forecasts included);
    production SV kernels: ``rtol 1e-9`` after a short burn; ``Runtime()`` runs on one device even
    with 4 available."""
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(path=TESTS_MCMC)],
        capture_output=True, text=True, env=env, timeout=1800,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    assert "TOYS-OK" in proc.stdout and "FORECAST-OK" in proc.stdout
    assert "PRODUCTION-OK" in proc.stdout or "PRODUCTION-SKIPPED" in proc.stdout


def test_resolve_devices_rules():
    assert Runtime().devices == 1  # DESIGN §5.2 Stage 2 amendment: sharding is opt-in
    assert resolve_devices(Runtime().devices, 4) == 1
    avail = jax.local_device_count()
    if avail == 1:
        assert resolve_devices("auto", 4) == 1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert resolve_devices(avail + 1, 4 * (avail + 1)) == 1
    assert any("set_cpu_devices" in str(w.message) for w in rec)
    assert resolve_devices(1, 3) == 1
    if avail >= 2:
        with pytest.raises(ValueError, match="multiple"):
            resolve_devices(2, 3)
