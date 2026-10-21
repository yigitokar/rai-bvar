import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation.backtest import (
    Cache,
    Expanding,
    Origins,
    Rolling,
    backtest,
    estimate_cost,
)
from rai_bvar.models import OLSVAR, RandomWalk

SEEN: list[tuple[str, str, int]] = []


@dataclass(frozen=True)
class Spy:
    """OLS-VAR that records the sample it was fit on."""

    lags: int = 1

    @property
    def name(self):
        return "spy"

    @property
    def axes(self):
        return {"model": "spy"}

    def spec(self):
        return {"class": "Spy", "lags": self.lags}

    def fit(self, data: VARData):
        SEEN.append((str(data.first), str(data.last), len(data)))
        return OLSVAR(self.lags).fit(data)


def make_data(T=160, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.1 + y[t - 1] @ A.T + rng.normal(scale=0.5, size=2)
    frame = pd.DataFrame(
        y, index=pd.period_range("1980Q1", periods=T, freq="Q"), columns=["x", "z"]
    )
    return VARData.from_frame(frame)


@pytest.fixture
def data():
    return make_data()


def test_origins_resolve_defaults_and_step(data):
    origins = Origins("2010Q1", step=4).resolve(data.index)
    assert str(origins[0]) == "2010Q1"
    assert origins[-1] <= data.index[-2]
    assert all((b - a).n == 4 for a, b in itertools.pairwise(origins))
    with pytest.raises(ValueError):
        Origins("1900Q1").resolve(data.index)


def test_no_look_ahead_and_window_sizes(data):
    SEEN.clear()
    backtest(
        Spy(), data, origins=Origins("2015Q1", "2016Q4"), window=Rolling(60), horizon=2, draws=50
    )
    assert [last for _, last, _ in SEEN] == [
        str(p) for p in pd.period_range("2015Q1", "2016Q4", freq="Q")
    ]
    assert {size for _, _, size in SEEN} == {60}
    SEEN.clear()
    backtest(Spy(), data, origins=["2015Q1"], window=Expanding(), horizon=2, draws=50)
    assert SEEN == [(str(data.first), "2015Q1", len(data.upto("2015Q1")))]


def test_window_too_small_raises(data):
    with pytest.raises(ValueError):
        backtest(RandomWalk(), data, origins=["1981Q1"], window=Rolling(60), horizon=1, draws=10)
    with pytest.raises(ValueError):
        backtest(
            RandomWalk(),
            data,
            origins=["1981Q1"],
            window=Expanding(min_size=40),
            horizon=1,
            draws=10,
        )


def test_scores_have_origin_dimension_and_nan_beyond_sample(data):
    last = str(data.index[-2])
    result = backtest(OLSVAR(1), data, origins=["2010Q1", last], horizon=3, draws=100)
    assert result.scores.sizes["origin"] == 2
    lpd = result.scores["lpd"].sel(origin=last).values
    assert np.isfinite(lpd[0]) and np.isnan(lpd[1:]).all()


def test_results_do_not_depend_on_other_origins(data):
    both = backtest(OLSVAR(1), data, origins=["2010Q1", "2012Q3"], horizon=2, draws=200, seed=7)
    single = backtest(OLSVAR(1), data, origins=["2012Q3"], horizon=2, draws=200, seed=7)
    np.testing.assert_array_equal(
        both.scores["lpd"].sel(origin="2012Q3"), single.scores["lpd"].sel(origin="2012Q3")
    )


def test_cache_roundtrip_and_invalidation(data, tmp_path):
    kwargs = dict(origins=Origins("2012Q1", "2013Q4"), horizon=2, draws=100, cache=tmp_path)
    first = backtest(OLSVAR(1), data, **kwargs)
    second = backtest(OLSVAR(1), data, **kwargs)
    assert first.cached == 0 and second.cached == 8
    np.testing.assert_allclose(first.scores["lpd"], second.scores["lpd"])
    np.testing.assert_allclose(first.scores["crps"], second.scores["crps"])
    assert backtest(OLSVAR(1), data, **{**kwargs, "draws": 101}).cached == 0
    assert backtest(OLSVAR(2), data, **kwargs).cached == 0
    Cache(tmp_path).clear()
    assert backtest(OLSVAR(1), data, **kwargs).cached == 0


def test_callable_data_source_needs_realized(data):
    with pytest.raises(ValueError):
        backtest(OLSVAR(1), lambda origin: data, origins=["2012Q1"], horizon=1, draws=10)
    result = backtest(
        OLSVAR(1), lambda origin: data, origins=["2012Q1"], horizon=1, draws=10, realized=data
    )
    assert result.scores.sizes["origin"] == 1


def test_parallel_matches_serial(data):
    kwargs = dict(origins=Origins("2012Q1", "2013Q2"), horizon=2, draws=100, seed=3)
    serial = backtest(OLSVAR(1), data, **kwargs)
    parallel = backtest(OLSVAR(1), data, n_jobs=2, **kwargs)
    np.testing.assert_array_equal(serial.scores["lpd"], parallel.scores["lpd"])


def test_estimate_cost(data):
    cost = estimate_cost(OLSVAR(1), data, origins=Origins("2010Q1"), horizon=2, draws=50)
    assert cost["seconds"] > 0 and cost["origins"] == len(Origins("2010Q1").resolve(data.index))
