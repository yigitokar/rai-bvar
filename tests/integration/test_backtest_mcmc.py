"""Backtest integration of the MCMC engine (DESIGN §11.6, §13 WP23).

DESIGN WP23 test list and where each item lives:

* analytic backtests unchanged — :func:`test_analytic_backtests_match_the_golden` (scores and
  cache keys against ``tests/fixtures/backtest_analytic_golden.npz``, written with the pre-WP23
  ``backtest.py``; ``rtol = 1e-8`` because the golden crosses machines and BLAS kernels: the EB
  optimiser and the MH sampler turn last-bit BLAS differences into relative differences of up
  to 1.1e-10 (measured with ``OPENBLAS_CORETYPE`` = Haswell, Zen, SandyBridge against the
  SkylakeX golden), far below any change to the analytic path) and
  :func:`test_analytic_models_are_called_exactly_as_in_phase_1` (bitwise, same program);
* fit seeds reach MCMC models only — :func:`test_fit_seeds_reach_models_that_declare_them_only`,
  :func:`test_mcmc_backtest_fits_with_the_origin_fit_seed`;
* resumed warm backtest equals an uninterrupted one (bitwise: same program) —
  :func:`test_resumed_warm_backtest_equals_an_uninterrupted_one`;
* warm vs cold scores statistically equal — :func:`test_warm_and_cold_scores_agree` (PR tier,
  lenient) and :func:`test_warm_and_cold_scores_are_statistically_equal` (``slow``);
* cache semantics with init provenance — :func:`test_warm_cache_keys_record_init_provenance`,
  :func:`test_warm_backtest_after_a_cold_one_refits_the_missing_state`,
  :func:`test_cached_state_round_trips_bitwise`;
* spawn context — :func:`test_mcmc_pool_uses_the_spawn_context_and_warns` (a fake executor) and
  :func:`test_spawn_pool_matches_the_serial_backtest` (``slow``, real worker processes);
* one compile per ``T`` bucket — :func:`test_one_compile_per_t_bucket`.

PR-tier MCMC runs share one ``StaticConfig`` per composition (``n = 2``, one lag,
``T_pad = 64``, 2 chains), so the engine compiles once per process (the compile test uses its
own ``t_bucket``).

Run ``python tests/integration/test_backtest_mcmc.py --write-golden`` only to re-pin the
golden after an *intended* change of analytic backtest results.
"""

from __future__ import annotations

import importlib
import math
import sys
import warnings
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rai_bvar import BVAR, SV, Linear, MCMCSettings, Minnesota, VARData
from rai_bvar.evaluation.backtest import Cache, Expanding, Rolling, backtest, fit_seed
from rai_bvar.evaluation.scores import score_forecast
from rai_bvar.models import OLSVAR, RandomWalk
from rai_bvar.priors.minnesota import MHSettings

bt_mod = importlib.import_module("rai_bvar.evaluation.backtest")

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "backtest_analytic_golden.npz"
GOLDEN_VERSION_KEY = "__version__"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# =========================================================================== golden (analytic)


def golden_data(T: int = 100, seed: int = 11) -> VARData:
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.05, 0.4]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        # elementwise IEEE operations only (no BLAS): the same bits on every machine
        y[t] = 0.1 + A[:, 0] * y[t - 1, 0] + A[:, 1] * y[t - 1, 1] + rng.normal(scale=0.5, size=2)
    frame = pd.DataFrame(
        y, index=pd.period_range("1985Q1", periods=T, freq="Q"), columns=["x", "z"]
    )
    return VARData.from_frame(frame)


def golden_cases():
    """The analytic backtests pinned by the golden (Phase 1 behaviour, before WP23)."""
    fixed = BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed")), lags=2)
    eb = BVAR(lags=2)
    mh = BVAR(
        mean=Linear(prior=Minnesota(hierarchy="mh", mh=MHSettings(draws=150, burn=100, chains=2))),
        lags=1,
    )
    origins = ["2005Q1", "2005Q3", "2006Q4"]
    return {
        "fixed_expanding": (fixed, dict(origins=origins, window=Expanding(), seed=3)),
        "eb_rolling": (eb, dict(origins=origins, window=Rolling(60), seed=0)),
        "mh_expanding": (mh, dict(origins=origins[:2], window=Expanding(), seed=5)),
        "olsvar": (OLSVAR(1), dict(origins=origins, window=Expanding(), seed=1)),
        "randomwalk": (RandomWalk(), dict(origins=origins, window=Rolling(40), seed=2)),
    }


def run_golden_case(name: str):
    model, kwargs = golden_cases()[name]
    return backtest(model, golden_data(), horizon=3, draws=200, **kwargs)


def golden_cache_keys(name: str) -> list[str]:
    """Cache keys of every origin of a golden case (unchanged by WP23 for analytic models)."""
    model, kwargs = golden_cases()[name]
    data = golden_data()
    keys = []
    for o in kwargs["origins"]:
        origin = pd.Period(o, freq="Q")
        sample = kwargs["window"].select(data.upto(origin))
        truth = data.realized(origin, 3)
        keys.append(
            bt_mod._cache_key(
                model, sample, truth, origin, kwargs["window"], 3, 200, kwargs["seed"]
            )
        )
    return keys


def _ordered(da) -> np.ndarray:
    return da.transpose(*[d for d in ("origin", "horizon", "variable") if d in da.dims]).values


def write_golden() -> None:  # pragma: no cover - run by hand, with the pre-WP23 backtest
    from rai_bvar._version import __version__

    out: dict[str, np.ndarray] = {GOLDEN_VERSION_KEY: np.array(__version__)}
    for name in golden_cases():
        scores = run_golden_case(name).scores
        for var in sorted(scores.data_vars):
            out[f"{name}/{var}"] = _ordered(scores[var])
        out[f"{name}/keys"] = np.array(golden_cache_keys(name))
    np.savez(GOLDEN, **out)
    print(f"wrote {GOLDEN} ({len(out)} arrays)")


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.mark.parametrize("name", list(golden_cases()))
def test_analytic_backtests_match_the_golden(name, golden, monkeypatch):
    scores = run_golden_case(name).scores
    stored = sorted(k.split("/", 1)[1] for k in golden if k.startswith(f"{name}/"))
    assert sorted([*scores.data_vars, "keys"]) == stored
    for var in scores.data_vars:
        np.testing.assert_allclose(
            _ordered(scores[var]), golden[f"{name}/{var}"], rtol=1e-8, atol=0, equal_nan=True,
            err_msg=f"{name}/{var}",
        )  # fmt: skip
    assert set(scores.attrs) <= {"sample_start", "sample_end", "n_obs"}  # no MCMC attrs
    # the cache keys are the Phase 1 keys (pinned at the golden's package version)
    monkeypatch.setattr(bt_mod, "__version__", str(golden[GOLDEN_VERSION_KEY]))
    assert golden_cache_keys(name) == list(golden[f"{name}/keys"])


CALLS: list[tuple[str, tuple, dict]] = []


@dataclass(frozen=True)
class CallSpy:
    """A model that records the arguments of every ``fit`` call (wrapping another model);
    ``mcmc=True`` declares ``uses_fit_seed`` like an MCMC model."""

    inner: object
    mcmc: bool = False

    @property
    def name(self):
        return f"spy({'mcmc' if self.mcmc else 'plain'})"

    @property
    def axes(self):
        return {"model": self.name}

    @property
    def uses_fit_seed(self):
        return self.mcmc

    def spec(self):
        return {"class": "CallSpy", "mcmc": self.mcmc, "inner": self.inner.spec()}

    def fit(self, data, *args, **kwargs):
        CALLS.append((str(data.last), args, kwargs))
        return self.inner.fit(data)


def test_analytic_models_are_called_exactly_as_in_phase_1():
    """Bitwise against the Phase 1 per-origin code path, re-run here (same program)."""
    model, kwargs = golden_cases()["fixed_expanding"]
    data, seed = golden_data(), kwargs["seed"]
    got = backtest(model, data, horizon=3, draws=200, **kwargs).scores
    for o in kwargs["origins"]:
        origin = pd.Period(o, freq="Q")
        rng = np.random.default_rng([seed, zlib.crc32(str(origin).encode())])
        fc = model.fit(data.upto(origin)).forecast(3, 200, seed=rng)
        want = score_forecast(fc, data.realized(origin, 3), rng=rng)
        for var in want.data_vars:
            np.testing.assert_array_equal(got[var].sel(origin=o).values, want[var].values)
    CALLS.clear()
    backtest(CallSpy(model), data, horizon=2, draws=50, origins=kwargs["origins"][:2])
    assert [c[1:] for c in CALLS] == [((), {})] * 2  # fit(sample), nothing else


# =========================================================================== fit seeds


def test_fit_seed_formula():
    for seed, origin in [(0, "2005Q1"), (7, "1999Q4"), (2**40, "2011-03")]:
        crc = zlib.crc32(str(origin).encode())
        want = int(np.random.default_rng([seed, crc, 0x4D434D43]).integers(2**31))
        assert fit_seed(seed, origin) == want
    assert fit_seed(3, pd.Period("2005Q1", freq="Q")) == fit_seed(3, "2005Q1")
    assert len({fit_seed(0, "2005Q1"), fit_seed(1, "2005Q1"), fit_seed(1, "2005Q2")}) == 3


def test_fit_seeds_reach_models_that_declare_them_only():
    data = golden_data()
    origins = ["2005Q1", "2006Q2", "2007Q3"]
    CALLS.clear()
    backtest(CallSpy(OLSVAR(1), mcmc=True), data, origins=origins, horizon=2, draws=50, seed=9)
    assert [c[0] for c in CALLS] == origins
    assert [c[1] for c in CALLS] == [()] * 3
    assert [c[2] for c in CALLS] == [{"seed": fit_seed(9, o)} for o in origins]
    CALLS.clear()
    backtest(CallSpy(OLSVAR(1)), data, origins=origins, horizon=2, draws=50, seed=9)
    assert [c[1:] for c in CALLS] == [((), {})] * 3
    assert not BVAR().uses_fit_seed and not BVAR().supports_warm_start  # analytic
    assert mcmc_model().uses_fit_seed and mcmc_model().supports_warm_start


# =========================================================================== MCMC fixtures

MCMC_SMALL = MCMCSettings(chains=2, draws=60, burn=60, warm_burn=30)


def mcmc_model(settings: MCMCSettings = MCMC_SMALL, **sv) -> BVAR:
    return BVAR(
        mean=Linear(prior=Minnesota(form="reduced", hierarchy="fixed")),
        vol=SV(**sv),
        lags=1,
        mcmc=settings,
    )


def sv_data(T: int = 64, seed: int = 4) -> VARData:
    """A bivariate VAR(1) whose shock variances shift half-way (so SV has work to do)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        scale = 0.4 if t < T // 2 else 1.2
        y[t] = y[t - 1] @ A.T + scale * rng.standard_normal(2)
    frame = pd.DataFrame(
        y, index=pd.period_range("1990Q1", periods=T, freq="Q"), columns=["a", "b"]
    )
    return VARData.from_frame(frame)


@pytest.fixture(scope="module")
def data2() -> VARData:
    return sv_data()


def origins_of(data: VARData, first: int, last: int) -> list[str]:
    """Origins at sample positions ``first..last`` (position 0 is the first period); with
    ``T = 64`` and one lag, positions 33..62 keep every window in the ``T_pad = 64`` bucket."""
    return [str(p) for p in data.index[first : last + 1]]


def test_mcmc_backtest_fits_with_the_origin_fit_seed(data2):
    model = mcmc_model()
    origin = origins_of(data2, 40, 40)[0]
    got = backtest(model, data2, origins=[origin], horizon=2, draws=80, seed=3).scores
    assert got.attrs["fit_seed"] == fit_seed(3, origin)
    rng = np.random.default_rng([3, zlib.crc32(origin.encode())])
    fit = model.fit(data2.upto(origin), seed=fit_seed(3, origin))
    want = score_forecast(fit.forecast(2, 80, seed=rng), data2.realized(origin, 2), rng=rng)
    np.testing.assert_array_equal(got["lpd"].sel(origin=origin).values, want["lpd"].values)
    np.testing.assert_array_equal(got["crps"].sel(origin=origin).values, want["crps"].values)
    # the model's own MCMCSettings.seed is overridden by the fit seed
    other = mcmc_model(replace(MCMC_SMALL, seed=123))
    got2 = backtest(other, data2, origins=[origin], horizon=2, draws=80, seed=3).scores
    np.testing.assert_array_equal(got2["lpd"].values, got["lpd"].values)


# =========================================================================== warm starts


def count_fits(monkeypatch) -> list[tuple[str, bool]]:
    """Record ``(origin, warm)`` for every fit the backtest runs."""
    seen: list[tuple[str, bool]] = []
    inner = bt_mod._fit_and_score

    def spy(model, sample, truth, origin, horizon, draws, seed, init=None):
        seen.append((str(origin), init is not None))
        return inner(model, sample, truth, origin, horizon, draws, seed, init=init)

    monkeypatch.setattr(bt_mod, "_fit_and_score", spy)
    return seen


def test_resumed_warm_backtest_equals_an_uninterrupted_one(data2, tmp_path, monkeypatch):
    model = mcmc_model()
    origins = origins_of(data2, 40, 44)
    kw = dict(horizon=2, draws=80, seed=1, warm_start=True)
    full = backtest(model, data2, origins=origins, cache=tmp_path / "a", **kw)
    assert full.cached == 0
    assert list(full.scores["origin"].values) == origins
    # no cache at all: the same states are passed in memory
    nocache = backtest(model, data2, origins=origins, **kw)
    # interrupted after two origins, then resumed: origins 0-1 come from the cache and origin 2
    # starts from the state of origin 1 read back from <key>.state.npz
    backtest(model, data2, origins=origins[:2], cache=tmp_path / "b", **kw)
    seen = count_fits(monkeypatch)
    resumed = backtest(model, data2, origins=origins, cache=tmp_path / "b", **kw)
    assert resumed.cached == 2
    assert seen == [(o, True) for o in origins[2:]]
    for other in (nocache, resumed):
        for var in full.scores.data_vars:
            np.testing.assert_array_equal(other.scores[var].values, full.scores[var].values)


def test_warm_cache_keys_record_init_provenance(data2, tmp_path, monkeypatch):
    model = mcmc_model()
    origins = origins_of(data2, 40, 42)
    kw = dict(horizon=2, draws=80, seed=1)
    window = Expanding()
    periods = [pd.Period(o, freq="Q") for o in origins]
    inputs = [(window.select(data2.upto(p)), data2.realized(p, 2)) for p in periods]
    cold = [
        bt_mod._cache_key(model, s, t, p, window, 2, 80, 1)
        for p, (s, t) in zip(periods, inputs, strict=True)
    ]
    warm = [cold[0]]
    for p, (s, t) in zip(periods[1:], inputs[1:], strict=True):
        warm.append(bt_mod._cache_key(model, s, t, p, window, 2, 80, 1, init=f"warm:{warm[-1]}"))
    assert len(set(cold + warm)) == 2 * len(origins) - 1  # only the first origin shares its key
    cache = Cache(tmp_path)
    backtest(model, data2, origins=origins, cache=cache, warm_start=True, **kw)
    for key in warm:
        assert cache.get(key) is not None and cache.state_path(key).exists()
    assert [int(cache.get(k).attrs["warm_start"]) for k in warm] == [0, 1, 1]
    assert [int(cache.get(k).attrs["fit_seed"]) for k in warm] == [fit_seed(1, o) for o in origins]
    assert all(cache.get(key) is None for key in cold[1:])
    # a cold run reuses the first origin only; the warm run is still fully cached afterwards
    seen = count_fits(monkeypatch)
    cold_res = backtest(model, data2, origins=origins, cache=cache, **kw)
    assert cold_res.cached == 1 and seen == [(o, False) for o in origins[1:]]
    seen.clear()
    again = backtest(model, data2, origins=origins, cache=cache, warm_start=True, **kw)
    assert again.cached == len(origins) and seen == []
    # another predecessor is another provenance: starting one origin later is a new chain whose
    # first origin is the cold fit of origins[1] (cached by the cold run above)
    seen.clear()
    later = backtest(model, data2, origins=origins[1:], cache=cache, warm_start=True, **kw)
    assert later.cached == 1 and seen == [(origins[1], False), (origins[2], True)]
    # Cache.clear() removes the states too
    cache.clear()
    assert not list(tmp_path.glob("*/*.state.npz")) and not list(tmp_path.glob("*/*.nc"))


def test_warm_backtest_after_a_cold_one_refits_the_missing_state(data2, tmp_path, monkeypatch):
    model = mcmc_model()
    origins = origins_of(data2, 46, 48)
    kw = dict(horizon=2, draws=80, seed=2)
    reference = backtest(model, data2, origins=origins, warm_start=True, **kw)
    backtest(model, data2, origins=origins[:1], cache=tmp_path, **kw)  # cold: no state file
    seen = count_fits(monkeypatch)
    res = backtest(model, data2, origins=origins, cache=tmp_path, warm_start=True, **kw)
    assert res.cached == 1
    assert seen == [(origins[0], False), (origins[1], True), (origins[2], True)]
    for var in reference.scores.data_vars:
        np.testing.assert_array_equal(res.scores[var].values, reference.scores[var].values)
    # lose the state of the last origin: extending the backtest refits that origin from the
    # state before it (still on disk) and continues; the results are unchanged
    cache = Cache(tmp_path)
    states = {}
    for path in tmp_path.glob("*/*.state.npz"):
        key = path.name.removesuffix(".state.npz")
        states[str(cache.get_state(key).periods[-1])] = path
    assert sorted(states) == origins
    states[origins[2]].unlink()
    seen.clear()
    extended = origins_of(data2, 46, 49)
    res2 = backtest(model, data2, origins=extended, cache=tmp_path, warm_start=True, **kw)
    assert res2.cached == 3 and seen == [(origins[2], True), (extended[3], True)]
    ref2 = backtest(model, data2, origins=extended, warm_start=True, **kw)
    for var in ref2.scores.data_vars:
        np.testing.assert_array_equal(res2.scores[var].values, ref2.scores[var].values)


def test_cached_state_round_trips_bitwise(data2, tmp_path):
    import jax

    model = mcmc_model(rw=True, tails="t")
    fit = model.fit(data2.upto("2002Q1"), seed=5)
    cache = Cache(tmp_path)
    key = "ab" + "0" * 62
    cache.put_state(key, fit.final_state)
    back = cache.get_state(key)
    assert back.static == fit.final_state.static
    assert back.periods.equals(fit.final_state.periods)
    assert back.seed_meta == fit.final_state.seed_meta
    a = jax.tree_util.tree_flatten_with_path(fit.final_state.state)
    b = jax.tree_util.tree_flatten_with_path(back.state)
    assert a[1] == b[1]
    for (pa, x), (pb, y) in zip(a[0], b[0], strict=True):
        assert pa == pb and np.asarray(x).dtype == np.asarray(y).dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert cache.get_state("cd" + "0" * 62) is None


def _paired_z(warm, cold) -> float:
    d = np.asarray(warm) - np.asarray(cold)
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))


def test_warm_and_cold_scores_agree(data2):
    """PR tier: few origins and short chains, so only a gross bias would fail (|z| < 4)."""
    model = mcmc_model()
    origins = origins_of(data2, 40, 51)
    kw = dict(horizon=1, draws=150, seed=4)
    warm = backtest(model, data2, origins=origins, warm_start=True, **kw).scores
    cold = backtest(model, data2, origins=origins, **kw).scores
    assert (warm["lpd"].values != cold["lpd"].values).any()  # really different runs
    for var in ("lpd", "crps"):
        w = warm[var].sel(horizon=1).values.reshape(len(origins), -1).mean(axis=1)
        c = cold[var].sel(horizon=1).values.reshape(len(origins), -1).mean(axis=1)
        assert abs(_paired_z(w, c)) < 4.0, var


def test_warm_start_validation(data2):
    with pytest.raises(ValueError, match="supports warm starts"):
        backtest(OLSVAR(1), data2, origins=origins_of(data2, 40, 41), warm_start=True)
    with pytest.raises(TypeError, match="warm_start"):
        backtest(mcmc_model(), data2, origins=origins_of(data2, 40, 41), warm_start="yes")
    # out of time order, a warm start would carry later data into an earlier origin's fit (R9)
    for origins in (origins_of(data2, 40, 41)[::-1], origins_of(data2, 40, 40) * 2):
        with pytest.raises(ValueError, match="strictly increasing"):
            backtest(mcmc_model(), data2, origins=origins, warm_start=True)


def test_mcmc_cache_keys_record_the_fit_seeding(data2, monkeypatch):
    """Keys of ``uses_fit_seed`` models carry ``"seeding"`` (scores cached before per-origin fit
    seeds are never served); analytic keys stay the Phase 1 keys (the golden pins those)."""
    window, p = Expanding(), pd.Period(origins_of(data2, 40, 40)[0], freq="Q")
    args = (window.select(data2.upto(p)), data2.realized(p, 2), p, window, 2, 80, 1)
    payloads = []
    real_dumps = bt_mod.json.dumps

    def spy(obj, **kw):
        payloads.append(obj)
        return real_dumps(obj, **kw)

    monkeypatch.setattr(bt_mod.json, "dumps", spy)
    bt_mod._cache_key(CallSpy(OLSVAR(1), mcmc=True), *args)
    bt_mod._cache_key(CallSpy(OLSVAR(1), mcmc=False), *args)
    bt_mod._cache_key(mcmc_model(), *args)
    assert [q.get("seeding") for q in payloads] == ["fit_seed-v1", None, "fit_seed-v1"]


def test_warm_start_ignores_n_jobs_with_a_warning(data2, monkeypatch):
    seen = count_fits(monkeypatch)
    origins = origins_of(data2, 40, 41)
    with pytest.warns(UserWarning, match="n_jobs is ignored"):
        backtest(mcmc_model(), data2, origins=origins, horizon=1, draws=50, n_jobs=2,
                 warm_start=True)  # fmt: skip
    assert seen == [(origins[0], False), (origins[1], True)]


# =========================================================================== processes


class FakePool:
    """Stands in for ``ProcessPoolExecutor``: records its arguments, maps serially."""

    made: ClassVar[list[dict]] = []

    def __init__(self, max_workers=None, mp_context=None):
        FakePool.made.append({"max_workers": max_workers, "mp_context": mp_context})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *iterables):
        return map(fn, *iterables)


def test_mcmc_pool_uses_the_spawn_context_and_warns(monkeypatch):
    monkeypatch.setattr(bt_mod, "ProcessPoolExecutor", FakePool)
    data = golden_data()
    origins = ["2005Q1", "2005Q2"]
    FakePool.made.clear()
    with pytest.warns(UserWarning, match="spawn"):
        backtest(CallSpy(OLSVAR(1), mcmc=True), data, origins=origins, horizon=1, draws=20,
                 n_jobs=2)  # fmt: skip
    assert FakePool.made[-1]["max_workers"] == 2
    assert FakePool.made[-1]["mp_context"].get_start_method() == "spawn"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # analytic: no warning, the Phase 1 pool
        backtest(CallSpy(OLSVAR(1)), data, origins=origins, horizon=1, draws=20, n_jobs=-1)
    assert FakePool.made[-1] == {"max_workers": None, "mp_context": None}


@pytest.mark.slow
def test_spawn_pool_matches_the_serial_backtest(data2):
    """Real spawned workers (each imports JAX and compiles); same platform, so equal to the
    serial run to rounding (``1e-12``; one process against another, no bitwise claim)."""
    model = mcmc_model()
    origins = origins_of(data2, 40, 43)
    kw = dict(horizon=2, draws=80, seed=6)
    serial = backtest(model, data2, origins=origins, **kw).scores
    with pytest.warns(UserWarning, match="spawn"):
        pooled = backtest(model, data2, origins=origins, n_jobs=2, **kw).scores
    for var in ("lpd", "lpd_marginal", "crps", "sq_err"):
        np.testing.assert_allclose(pooled[var].values, serial[var].values, rtol=1e-12)


# =========================================================================== compiles


def test_one_compile_per_t_bucket(data2):
    """Each origin pads to its own bucket; origins in one bucket reuse its executables."""
    from rai_bvar.engines.mcmc import runner
    from rai_bvar.engines.mcmc.config import bucket

    settings = replace(MCMC_SMALL, t_bucket=4, draws=20, burn=20)
    model = mcmc_model(settings)
    origins = origins_of(data2, 33, 40)  # T_eff = 33..40: buckets 36 and 40
    expected = {bucket(len(data2.upto(o)) - 1, 4) for o in origins}
    assert expected == {36, 40}
    runner.clear_caches()
    backtest(model, data2, origins=origins, horizon=1, draws=20)
    info = runner._make_runner_cached.cache_info()
    assert info.misses == len(expected)
    assert info.hits >= len(origins) - len(expected)


# =========================================================================== slow


@pytest.mark.slow
def test_warm_and_cold_scores_are_statistically_equal():
    """Nightly: 40 origins, longer chains; paired t-tests on the per-origin joint LPD and mean
    CRPS at h = 1 and 2, Holm-corrected at 1% (warm and cold differ only by Monte Carlo noise)."""
    data = sv_data(T=110, seed=8)
    settings = MCMCSettings(chains=2, draws=400, burn=300, warm_burn=100)
    model = mcmc_model(settings)
    origins = [str(p) for p in data.index[64:104]]
    kw = dict(horizon=2, draws=400, seed=10)
    warm = backtest(model, data, origins=origins, warm_start=True, **kw).scores
    cold = backtest(model, data, origins=origins, **kw).scores
    pvalues = []
    for h in (1, 2):
        for var in ("lpd", "crps"):
            w = warm[var].sel(horizon=h).values.reshape(len(origins), -1).mean(axis=1)
            c = cold[var].sel(horizon=h).values.reshape(len(origins), -1).mean(axis=1)
            pvalues.append(stats.ttest_rel(w, c).pvalue)
    holm = sorted(pvalues)
    assert all(p * (len(holm) - i) > 0.01 for i, p in enumerate(holm)), pvalues


if __name__ == "__main__":  # pragma: no cover
    if sys.argv[1:] == ["--write-golden"]:
        write_golden()
