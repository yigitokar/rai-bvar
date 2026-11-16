"""The LPD estimator option through backtests, comparisons and the ordering check (D-34).

``lpd_method`` (DESIGN §7.3, D-34) is a plain string threaded from ``backtest`` / ``compare`` /
``ordering_sensitivity`` to ``score_forecast``:

* cache keys: unchanged at the default (the analytic golden in ``test_backtest_mcmc.py`` pins
  them), tagged ``"lpd": "integrated-v1"`` otherwise —
  :func:`test_cache_keys_are_tagged_only_for_another_estimator`;
* an MCMC backtest with ``"integrated"`` runs end to end; its ``h = 1`` scores and every
  non-LPD score equal the default run's bitwise (same program, same seeds), its conditioned
  columns are the default run's LPD, its ``h = 2`` LPD is the direct integrated call —
  :func:`test_integrated_mcmc_backtest_against_the_default_run`;
* the worker-pool path (a serial stand-in for the pool, PR tier) and real ``spawn`` workers
  (``slow``; ``1e-12`` across processes, never bitwise) —
  :func:`test_the_pool_path_forwards_the_estimator`, :func:`test_spawn_pool_integrated_backtest`;
* warm starts, ``estimate_cost``, ``compare`` and validation;
* ``ordering_sensitivity`` in backtest mode (its cache keys differ from the default's and a
  direct integrated backtest of an ordering is served from its cache) and in forecast mode.

Models without ``lag_coefs`` (benchmarks) are legal at ``h = 1`` only, which keeps the
forwarding tests of ``compare`` and ``ordering_sensitivity`` cheap. Analytic forecasts carry
``lag_coefs`` on request only (P3 DESIGN D-49, E120; amended in WP33 as P3 §1.3 states): a default
analytic forecast still raises for the integrated estimator at ``h > 1``, and the backtest asks
for ``forecast(..., lag_coefs=True)`` —
:func:`test_integrated_analytic_backtest_requests_lag_coefs`.
"""

from __future__ import annotations

import importlib
import json
import re
import zlib
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from rai_bvar import BVAR, SV, Linear, MCMCSettings, Minnesota, VARData
from rai_bvar.evaluation.backtest import Cache, Expanding, backtest, estimate_cost, fit_seed
from rai_bvar.evaluation.compare import compare
from rai_bvar.models import OLSVAR
from rai_bvar.structural import ordering as so
from rai_bvar.structural.ordering import ordering_sensitivity, permute_model

bt_mod = importlib.import_module("rai_bvar.evaluation.backtest")
cmp_mod = importlib.import_module("rai_bvar.evaluation.compare")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

LPD = ("lpd", "lpd_mcse", "lpd_marginal")
CONDITIONED = {"lpd_conditioned": "lpd", "lpd_mcse_conditioned": "lpd_mcse",
               "lpd_marginal_conditioned": "lpd_marginal"}  # fmt: skip
EXTRA = {*CONDITIONED, "lpd_marginal_mcse"}

# the composition of test_backtest_mcmc.py (n = 2, one lag, T_pad = 64, 2 chains): one compile
MCMC_SMALL = MCMCSettings(chains=2, draws=60, burn=60, warm_burn=30)


def mcmc_model(settings: MCMCSettings = MCMC_SMALL) -> BVAR:
    return BVAR(
        mean=Linear(prior=Minnesota(form="reduced", hierarchy="fixed")),
        vol=SV(),
        lags=1,
        mcmc=settings,
    )


def analytic() -> BVAR:
    return BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed")), lags=1)


def sv_data(T: int = 64, seed: int = 4) -> VARData:
    """A bivariate VAR(1) whose shock variances shift half-way."""
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
    return [str(p) for p in data.index[first : last + 1]]


def assert_integrated_matches_default(integ, default) -> None:
    """``integ`` (scores of ``lpd_method="integrated"``) against ``default`` (the same backtest
    with the default estimator; same program and seeds, so bitwise)."""
    assert set(integ.data_vars) == set(default.data_vars) | EXTRA
    assert integ.attrs["lpd_method"] == "integrated" and "lpd_method" not in default.attrs
    for name in set(default.data_vars) - set(LPD):
        np.testing.assert_array_equal(integ[name].values, default[name].values, err_msg=name)
    for new, old in CONDITIONED.items():
        np.testing.assert_array_equal(integ[new].values, default[old].values, err_msg=new)
    for name in LPD:
        np.testing.assert_array_equal(
            integ[name].sel(horizon=1).values, default[name].sel(horizon=1).values, err_msg=name
        )
        later = integ[name].sel(horizon=slice(2, None)).values
        assert (
            np.isfinite(later).all()
            and (later != default[name].sel(horizon=slice(2, None)).values).all()
        )


# =========================================================================== cache keys


def test_cache_keys_are_tagged_only_for_another_estimator(data2, monkeypatch):
    window, p = Expanding(), pd.Period(origins_of(data2, 40, 40)[0], freq="Q")
    args = (window.select(data2.upto(p)), data2.realized(p, 2), p, window, 2, 80, 1)
    payloads, keys = [], []
    real_dumps = bt_mod.json.dumps

    def spy(obj, **kw):
        payloads.append(obj)
        return real_dumps(obj, **kw)

    monkeypatch.setattr(bt_mod.json, "dumps", spy)
    for model in (analytic(), mcmc_model()):
        for kw in ({}, {"lpd_method": "conditioned"}, {"lpd_method": "integrated"},
                   {"init": "warm:abc", "lpd_method": "integrated"}):  # fmt: skip
            keys.append(bt_mod._cache_key(model, *args, **kw))
    for m in (0, 4):  # per model: default, explicit default, integrated, warm + integrated
        default, explicit, integ, warm = payloads[m : m + 4]
        assert "lpd" not in default and explicit == default and keys[m] == keys[m + 1]
        assert integ == {**default, "lpd": "integrated-v1"} and keys[m + 2] != keys[m]
        assert warm == {**integ, "init": "warm:abc"}
    assert payloads[0].get("seeding") is None and payloads[4]["seeding"] == "fit_seed-v1"


# =========================================================================== MCMC backtests


def test_integrated_mcmc_backtest_against_the_default_run(data2):
    model = mcmc_model()
    origins = origins_of(data2, 40, 42)
    kw = dict(origins=origins, horizon=2, draws=80, seed=3)
    default = backtest(model, data2, **kw).scores
    integ = backtest(model, data2, lpd_method="integrated", **kw).scores
    assert_integrated_matches_default(integ, default)
    # the h = 2 scores of one origin are the direct integrated calls on its forecast
    origin = origins[1]
    rng = np.random.default_rng([3, zlib.crc32(origin.encode())])
    fit = model.fit(data2.upto(origin), seed=fit_seed(3, origin))
    fc = fit.forecast(2, 80, seed=rng)
    truth = data2.realized(origin, 2)
    lpd, mcse = fc.log_pred_density(truth[1], 2, return_mcse=True, lpd_method="integrated")
    cell = integ.sel(origin=origin, horizon=2)
    assert cell["lpd"].item() == lpd and cell["lpd_mcse"].item() == mcse
    for j, var in enumerate(fc.variables):
        want = fc.log_pred_density(
            truth[1, j : j + 1], 2, [var], return_mcse=True, lpd_method="integrated"
        )
        assert (cell["lpd_marginal"].sel(variable=var).item(),
                cell["lpd_marginal_mcse"].sel(variable=var).item()) == want  # fmt: skip


class SerialPool:
    """Stands in for ``ProcessPoolExecutor`` (maps serially in this process)."""

    made: ClassVar[list[dict]] = []

    def __init__(self, max_workers=None, mp_context=None):
        SerialPool.made.append({"max_workers": max_workers, "mp_context": mp_context})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *iterables):
        return map(fn, *iterables)


def test_the_pool_path_forwards_the_estimator(data2, monkeypatch):
    model = mcmc_model()
    kw = dict(origins=origins_of(data2, 40, 41), horizon=2, draws=80, seed=5)
    serial = backtest(model, data2, lpd_method="integrated", **kw).scores
    monkeypatch.setattr(bt_mod, "ProcessPoolExecutor", SerialPool)
    SerialPool.made.clear()
    pooled = backtest(model, data2, lpd_method="integrated", n_jobs=2, **kw).scores
    assert SerialPool.made[-1]["mp_context"].get_start_method() == "spawn"
    assert set(pooled.data_vars) == set(serial.data_vars) and pooled.attrs["lpd_method"]
    for name in serial.data_vars:
        np.testing.assert_array_equal(pooled[name].values, serial[name].values, err_msg=name)


@pytest.mark.slow
def test_spawn_pool_integrated_backtest(data2):
    """Real spawned workers (each imports JAX and compiles); ``1e-12`` against the serial run,
    and the ``h = 1`` LPD equals the default serial run's to the same tolerance."""
    model = mcmc_model()
    kw = dict(origins=origins_of(data2, 40, 42), horizon=2, draws=80, seed=6)
    serial = backtest(model, data2, lpd_method="integrated", **kw).scores
    default = backtest(model, data2, **kw).scores
    pooled = backtest(model, data2, lpd_method="integrated", n_jobs=2, **kw).scores
    assert set(pooled.data_vars) == set(serial.data_vars)
    assert pooled.attrs["lpd_method"] == "integrated"
    for name in serial.data_vars:
        np.testing.assert_allclose(pooled[name].values, serial[name].values, rtol=1e-12)
    for name in LPD:
        np.testing.assert_allclose(
            pooled[name].sel(horizon=1).values, default[name].sel(horizon=1).values, rtol=1e-12
        )


def test_warm_backtest_forwards_the_estimator(data2, tmp_path, monkeypatch):
    model = mcmc_model()
    origins = origins_of(data2, 44, 46)
    kw = dict(origins=origins, horizon=2, draws=80, seed=2, warm_start=True)
    default = backtest(model, data2, cache=tmp_path, **kw)
    seen: list[str] = []
    inner = bt_mod._fit_and_score

    def spy(*args, lpd_method="conditioned", **kwargs):
        seen.append(lpd_method)
        return inner(*args, lpd_method=lpd_method, **kwargs)

    monkeypatch.setattr(bt_mod, "_fit_and_score", spy)
    integ = backtest(model, data2, cache=tmp_path, lpd_method="integrated", **kw)
    assert integ.cached == 0 and seen == ["integrated"] * len(origins)  # other keys: refit
    assert_integrated_matches_default(integ.scores, default.scores)
    again = backtest(model, data2, cache=tmp_path, lpd_method="integrated", **kw)
    assert again.cached == len(origins) and len(seen) == len(origins)
    for name in integ.scores.data_vars:
        np.testing.assert_array_equal(again.scores[name].values, integ.scores[name].values)
    assert again.scores.attrs["lpd_method"] == "integrated"  # read back from the cache
    # warm keys carry both tags; the cache holds both runs' scores and states
    assert len(list(tmp_path.glob("*/*.nc"))) == len(list(tmp_path.glob("*/*.state.npz"))) == 6


# =========================================================================== entry points


def test_estimate_cost_forwards_the_estimator(data2, monkeypatch):
    seen = []
    monkeypatch.setattr(bt_mod, "_run_origin", lambda *args: seen.append(args))
    origins = origins_of(data2, 40, 41)
    estimate_cost(mcmc_model(), data2, origins=origins, horizon=2, draws=50)
    estimate_cost(
        mcmc_model(), data2, origins=origins, horizon=2, draws=50, lpd_method="integrated"
    )
    assert [a[-1] for a in seen] == ["conditioned", "integrated"]


def test_compare_forwards_the_estimator(data2, monkeypatch):
    seen = []
    inner = cmp_mod.backtest

    def spy(*args, **kwargs):
        seen.append(kwargs.get("lpd_method"))
        return inner(*args, **kwargs)

    monkeypatch.setattr(cmp_mod, "backtest", spy)
    models = {"fixed": analytic(), "ols": OLSVAR(1)}
    kw = dict(origins=origins_of(data2, 40, 45), horizons=(1,), draws=100)
    default = compare(models, data2, **kw)
    integ = compare(models, data2, lpd_method="integrated", **kw)  # h = 1: no lag_coefs needed
    assert seen == ["conditioned"] * 2 + ["integrated"] * 2
    for name in models:
        got, want = integ.scores(name), default.scores(name)
        assert got.attrs["lpd_method"] == "integrated"
        for var in want.data_vars:  # h = 1 only: the estimators coincide bitwise
            np.testing.assert_array_equal(got[var].values, want[var].values, err_msg=var)
    pd.testing.assert_frame_equal(integ.table(), default.table())


def test_unknown_or_unsupported_estimators_raise(data2):
    origins = origins_of(data2, 40, 41)
    for call in (
        lambda: backtest(mcmc_model(), data2, origins=origins, lpd_method="exact"),
        lambda: estimate_cost(mcmc_model(), data2, origins=origins, lpd_method="exact"),
        lambda: compare([analytic()], data2, origins=origins, lpd_method="exact"),
        lambda: ordering_sensitivity(analytic(), data2, n_perm=2, lpd_method="exact"),
    ):
        with pytest.raises(ValueError, match="lpd_method must be one of"):
            call()
    with pytest.raises(ValueError, match="lpd_method applies to"):
        ordering_sensitivity(analytic(), data2, mode="prior", lpd_method="integrated")
    # models without lag_coefs: integrated at h > 1 raises at the first origin (no fallback)
    with pytest.raises(ValueError, match="lag_coefs"):
        backtest(OLSVAR(1), data2, origins=origins, horizon=2, draws=50, lpd_method="integrated")
    # a default analytic forecast still carries none (E120)
    fc = analytic().fit(data2).forecast(2, 50, seed=1)
    with pytest.raises(ValueError, match="lag_coefs"):
        fc.log_pred_density(data2.values[-1], 2, lpd_method="integrated")


def test_integrated_analytic_backtest_requests_lag_coefs(data2):
    """P3 D-49/E120: with ``"integrated"`` at ``h > 1`` the backtest makes every analytic forecast
    with ``lag_coefs=True``. That adds no random numbers, so every non-LPD score and the ``h = 1``
    LPD equal the default run's bitwise, and the ``h = 2`` LPD is the direct integrated call on
    ``forecast(..., lag_coefs=True)``."""
    model, origins = analytic(), origins_of(data2, 40, 42)
    kw = dict(origins=origins, horizon=2, draws=80, seed=3)
    default = backtest(model, data2, **kw).scores
    integ = backtest(model, data2, lpd_method="integrated", **kw).scores
    assert_integrated_matches_default(integ, default)
    origin = origins[1]
    rng = np.random.default_rng([3, zlib.crc32(origin.encode())])
    fc = model.fit(data2.upto(origin)).forecast(2, 80, seed=rng, lag_coefs=True)
    truth = data2.realized(origin, 2)
    lpd, mcse = fc.log_pred_density(truth[1], 2, return_mcse=True, lpd_method="integrated")
    cell = integ.sel(origin=origin, horizon=2)
    assert cell["lpd"].item() == lpd and cell["lpd_mcse"].item() == mcse


@pytest.mark.parametrize("make", [lambda: OLSVAR(1)], ids=["ols"])
def test_scoring_errors_name_the_model_and_the_origin(data2, make):
    """Review finding contract-2: in ``compare`` the failing model is not the last one fitted,
    so the error says which model and origin it came from (the cause is kept)."""
    model, origins = make(), origins_of(data2, 40, 41)
    prefix = re.escape(f"{model.name} at origin {origins[0]}: ")
    with pytest.raises(ValueError, match=f"^{prefix}.*'lag_coefs'") as info:
        backtest(model, data2, origins=origins, horizon=2, draws=50, lpd_method="integrated")
    assert isinstance(info.value.__cause__, ValueError)
    assert str(info.value) == f"{model.name} at origin {origins[0]}: {info.value.__cause__}"
    with pytest.raises(ValueError, match=f"^{prefix}"):
        compare({"m": model}, data2, origins=origins, horizons=(1, 2), draws=50,
                lpd_method="integrated")  # fmt: skip


# =========================================================================== ordering check


def test_ordering_backtest_mode_forwards_the_estimator(data2, tmp_path, monkeypatch):
    model, origins = mcmc_model(), origins_of(data2, 40, 41)
    kw = dict(n_perm=2, mode="backtest", horizons=(1, 2), origins=origins, noise_reps=1, draws=80)
    seen = []
    inner = bt_mod.backtest

    def spy(*args, **kwargs):
        seen.append(kwargs.get("lpd_method"))
        return inner(*args, **kwargs)

    monkeypatch.setattr(bt_mod, "backtest", spy)
    cache = Cache(tmp_path)
    default = ordering_sensitivity(model, data2, cache=cache, **kw).dataset
    n_files = len(list(tmp_path.glob("*/*.nc")))
    integ = ordering_sensitivity(model, data2, cache=cache, lpd_method="integrated", **kw).dataset
    assert seen == ["conditioned"] * 3 + ["integrated"] * 3  # 2 orderings + 1 noise replicate
    assert len(list(tmp_path.glob("*/*.nc"))) == 2 * n_files == 2 * 3 * len(origins)
    assert (
        integ.attrs["lpd_method"] == "integrated" and default.attrs["lpd_method"] == "conditioned"
    )
    for name in ("origin_scores", "origin_joint_lpd"):
        np.testing.assert_array_equal(
            integ[name].sel(horizon=1).values, default[name].sel(horizon=1).values
        )
    a, b = integ["origin_scores"].sel(horizon=2), default["origin_scores"].sel(horizon=2)
    assert (a.sel(score="lpd").values != b.sel(score="lpd").values).all()
    np.testing.assert_array_equal(  # CRPS and squared errors do not depend on the estimator
        a.sel(score=["crps", "sq_err"]).values, b.sel(score=["crps", "sq_err"]).values
    )
    a, b = integ["origin_joint_lpd"].sel(horizon=2), default["origin_joint_lpd"].sel(horizon=2)
    assert (a.values != b.values).all()
    # the reversed ordering's integrated backtest, called directly, is served from the cache
    j, order = 1, [str(v) for v in json.loads(integ.attrs["orderings"])[1]]
    m = so._with_seed(permute_model(model, order, variables=data2.variables), integ.attrs["seed"])
    seeds = json.loads(integ.attrs["backtest_seeds"])
    direct = inner(m, data2.select(order), origins=origins, horizon=2, draws=80, seed=seeds[j],
                   cache=cache, lpd_method="integrated")  # fmt: skip
    assert direct.cached == len(origins)
    np.testing.assert_array_equal(
        integ["origin_joint_lpd"].sel(ordering=j).values,
        direct.scores["lpd"].transpose("origin", "horizon").values,
    )


def test_ordering_forecast_mode_forwards_the_estimator(data2):
    model, origin = mcmc_model(), data2.index[-6]
    kw = dict(n_perm=2, noise_reps=0, horizons=(1, 2), origins=[origin], draws=80)
    default = ordering_sensitivity(model, data2, **kw).dataset
    integ = ordering_sensitivity(model, data2, lpd_method="integrated", **kw).dataset
    assert integ.attrs["lpd_method"] == "integrated"
    lpd = {"value": ("stat", "lpd"), "joint": ("jstat", "lpd"), "mcse": ("stat", "lpd"),
           "mcse_joint": ("jstat", "lpd")}  # fmt: skip
    for name, (dim, label) in lpd.items():
        a, b = integ[name].sel({dim: label}), default[name].sel({dim: label})
        np.testing.assert_array_equal(a.sel(horizon=1).values, b.sel(horizon=1).values)
        assert (a.sel(horizon=2).values != b.sel(horizon=2).values).all(), name
    other = [s for s in integ["stat"].values if s != "lpd"]
    np.testing.assert_array_equal(
        integ["value"].sel(stat=other).values, default["value"].sel(stat=other).values
    )
    # ordering 1's joint h = 2 LPD is the direct integrated call on its own forecast
    j, order = 1, [str(v) for v in json.loads(integ.attrs["orderings"])[1]]
    seeds = json.loads(integ.attrs["forecast_seeds"])
    window = data2.upto(origin).select(order)
    fit = permute_model(model, order, variables=data2.variables).fit(window, seed=0)
    fc = fit.forecast(2, 80, seed=seeds[j])
    y = data2.realized(origin, 2)[1][[data2.variables.index(v) for v in order]]
    want = fc.log_pred_density(y, 2, lpd_method="integrated")
    assert integ["joint"].sel(ordering=j, horizon=2, jstat="lpd").item() == pytest.approx(
        want, rel=1e-12
    )
