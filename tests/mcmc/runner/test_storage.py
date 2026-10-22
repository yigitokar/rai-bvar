"""Storage plan and host store (DESIGN §5.3, §10.5)."""

from __future__ import annotations

import os

import jax
import numpy as np
import pytest
import xarray as xr

from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.runner import run_chains
from rai_bvar.engines.mcmc.state import Empty
from rai_bvar.engines.mcmc.storage import HostStore, StorageBudgetError, plan_storage
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

STATIC = toys.toy_static(
    "linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5, paths=True
)


def _record():
    S = jax.ShapeDtypeStruct
    mon = {"x": S((), np.float64), "v": S((2,), np.float64)}
    params = {"p": S((3,), np.float64)}
    paths = {"h": S((2, 4), np.float64)}
    return mon, params, paths


def test_plan_bytes_and_budget():
    settings = MCMCSettings(chains=4, draws=10, store_max=4, path_thin=2)
    plan = plan_storage(STATIC, settings, Runtime(), record=_record())
    # monitor: every draw (3 floats); params: store_thin 3 → 4 records; paths: every 6th → 2
    assert plan.streams["params"].every == 3 and plan.streams["params"].count == 4
    assert plan.streams["paths"].every == 6 and plan.streams["paths"].count == 2
    assert plan.bytes_by_stream["monitor"] == 4 * 10 * 3 * 8
    assert plan.bytes_by_stream["params"] == 4 * 4 * 3 * 8
    assert plan.bytes_by_stream["paths"] == 4 * 2 * 8 * 8
    with pytest.raises(StorageBudgetError, match=r"store_max.*storage_dir"):
        plan_storage(STATIC, settings, Runtime(memory_budget_gb=1e-9), record=_record())
    ok = plan_storage(
        STATIC, settings, Runtime(memory_budget_gb=1e-9, storage_dir="unused"), record=_record()
    )
    assert ok.total_bytes > ok.budget_bytes


def test_run_raises_over_budget_before_sampling():
    fns = toys.cached_runner(STATIC, toys.toy_overrides(STATIC))
    with pytest.raises(StorageBudgetError):
        run_chains(
            fns, STATIC, toys.toy_data(STATIC, 20, 0), toys.toy_prior(STATIC),
            MCMCSettings(chains=2, draws=4, burn=1, store="paths"), Runtime(chunk=5, memory_budget_gb=1e-12),
            seed=0, salt=0,
        )  # fmt: skip


def test_default_record_comes_from_the_production_runner(toy_registry):
    static = toys.toy_static("linear_structural", "constant", n=2, T_eff=20, chunk=5)
    plan = plan_storage(static, MCMCSettings(chains=2, draws=3), Runtime())
    params = plan.streams["params"].record
    assert params.mean.gamma.shape == (2, 3) and params.vol.core.log_lam.shape == (2,)
    assert isinstance(params.vol.trans, Empty) and "paths" not in plan.streams


def test_append_selects_thinned_records():
    plan = plan_storage(STATIC, MCMCSettings(chains=2, draws=10, store_max=4, path_thin=2), Runtime(),
                        record=_record())  # fmt: skip
    store = HostStore.allocate(plan)
    for c, stop in ((0, 4), (1, 4)):  # two chunks of 5 kept draws
        d = np.arange(5) + 5 * c
        mon = {
            "x": np.broadcast_to(d, (2, 5)).astype(float),
            "v": np.zeros((2, 5, 2)) + d[None, :, None],
        }
        out = (
            mon,
            {"p": np.zeros((2, 5, 3)) + d[None, :, None]},
            {"h": np.zeros((2, 5, 2, 4)) + d[None, :, None, None]},
        )
        store.append(out, stop)
    np.testing.assert_array_equal(
        store.arrays["monitor"][1][0], np.arange(10)
    )  # "x" (sorted keys: v, x)
    np.testing.assert_array_equal(store.stream("params")["p"][0, :, 0], [0, 3, 6, 9])
    np.testing.assert_array_equal(store.stream("paths")["h"][1, :, 0, 0], [0, 6])
    assert store.kept == 10


def _fit(storage_dir=None):
    return fit_mcmc(
        STATIC, toys.toy_data(STATIC, 20, 2), toys.toy_prior(STATIC),
        MCMCSettings(chains=2, draws=7, burn=3, store_max=3, path_thin=2, store="paths"),
        Runtime(chunk=5, storage_dir=storage_dir), seed=4, salt=4,
        meta=FitMeta(variables=("a", "b"), periods=toys.periods(20)),
        _overrides=toys.toy_overrides(STATIC),
    )  # fmt: skip


@pytest.mark.parametrize("mean", ["linear_structural", "linear_reduced"])
def test_k_equals_n_labels_follow_the_leaf_names(mean):
    """``deterministic="n"`` with one lag gives ``k = n``: ``gamma (n, k)``, ``Pi (k, n)`` and the
    ``(n, k)`` prior arrays must be labelled by name, not by (ambiguous) shape."""
    static = toys.toy_static(mean, "constant", n=2, lags=1, n_det=0, T_eff=20, chunk=5)
    assert static.shapes.k == static.shapes.n == 2
    fit = fit_mcmc(
        static, toys.toy_data(static, 20, 4), toys.toy_prior(static), MCMCSettings(chains=2, draws=4, burn=2),
        Runtime(chunk=5), seed=0, salt=1, meta=FitMeta(variables=("gdp", "cpi")),
        _overrides=toys.toy_overrides(static),
    )  # fmt: skip
    post = fit.posterior["posterior"].to_dataset()
    const = fit.posterior["constant_data"].to_dataset()
    assert list(post["regressor"].values) == ["gdp.L1", "cpi.L1"]
    if mean == "linear_structural":
        raw = np.asarray(fit.param_draws.mean.gamma)  # (C, D, equation, regressor)
        g = post["gamma"]
        assert g.dims == ("chain", "draw", "variable", "regressor")
        assert (
            float(g.isel(chain=1, draw=2).sel(variable="cpi", regressor="gdp.L1"))
            == raw[1, 2, 1, 0]
        )
        assert const["prior_shared_Mc"].dims == ("regressor", "variable_")
    else:
        raw = np.asarray(fit.param_draws.glob.mean.Pi)  # (C, D, regressor, equation)
        p = post["Pi"]
        assert p.dims == ("chain", "draw", "regressor", "variable")
        assert (
            float(p.isel(chain=1, draw=2).sel(variable="cpi", regressor="gdp.L1"))
            == raw[1, 2, 0, 1]
        )
    prior = toys.toy_prior(static)
    for name in ("mean0", "prec0"):
        da = const[f"prior_eq_coef_{name}"]
        assert da.dims == ("variable", "regressor")
        expected = getattr(prior.eq.coef, name)[1, 0]
        assert float(da.sel(variable="cpi", regressor="gdp.L1")) == expected
    assert const["prior_eq_contemp_prec0"].dims == ("variable", "variable_")


def test_ambiguous_unnamed_leaves_get_generic_dims():
    from rai_bvar.engines.mcmc.storage import _Names

    names = _Names(("chain",), 2, 2, ("a", "b"), ("a.L1", "b.L1"), ("b~a",), None, None)
    assert names.dims("gamma", (2, 2)) == ("variable", "regressor")
    assert names.dims("glob.mean.Pi", (2, 2)) == ("regressor", "variable")
    assert names.dims("mystery", (2, 2)) == ("mystery_dim0", "mystery_dim1")
    assert names.dims("mystery", (2, 2), lead_variable=True) == ("variable", "mystery_dim1")
    assert names.dims("mystery", (2, 1)) == ("variable", "variable_")
    assert names.dims("mystery", (2,)) == ("variable",)
    wide = _Names(("chain",), 2, 3, ("a", "b"), ("c", "a.L1", "b.L1"), ("b~a",), None, None)
    assert wide.dims("mystery", (2, 3)) == ("variable", "regressor")
    assert wide.dims("mystery", (3, 2)) == ("regressor", "variable")


def test_memmap_store_equals_memory_store(tmp_path):
    mem = _fit()
    disk = _fit(str(tmp_path))
    assert not mem.store.memmap and disk.store.memmap
    assert disk.store.directory.startswith(str(tmp_path)) and os.listdir(disk.store.directory)
    for name in ("monitor", "params", "paths"):
        for x, y in zip(mem.store.arrays[name], disk.store.arrays[name], strict=True):
            assert isinstance(y, np.memmap) or y.size == 0
            np.testing.assert_array_equal(x, np.asarray(y))
    for group in ("posterior", "monitor", "paths", "sample_stats", "latent_moments"):
        xr.testing.assert_identical(
            mem.posterior[group].to_dataset(), disk.posterior[group].to_dataset()
        )
