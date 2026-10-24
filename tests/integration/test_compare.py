import dataclasses

import matplotlib
import numpy as np
import pandas as pd
import pytest

from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation import CompareResult, Entry, Expanding, Origins, Rolling, compare, grid
from rai_bvar.models import OLSVAR, RandomWalk

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    T = 200
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = 0.2 + y[t - 1] @ A.T + rng.normal(scale=0.5, size=3)
    frame = pd.DataFrame(
        y, index=pd.period_range("1970Q1", periods=T, freq="Q"), columns=["a", "b", "c"]
    )
    return VARData.from_frame(frame)


@pytest.fixture(scope="module")
def rw_vs_var(data):
    return compare(
        {"RW": RandomWalk(), "VAR1": OLSVAR(1)},
        data,
        origins=Origins("2005Q1"),
        window=Rolling(100),
        horizons=[1, 4],
        draws=400,
    )


def test_table_structure(rw_vs_var):
    table = rw_vs_var.table()
    assert set(table.index) == {("RW", 1), ("RW", 4), ("VAR1", 1), ("VAR1", 4)}
    assert table.loc[("RW", 1), "dlpd"] == 0.0
    assert table.loc[("RW", 1), "crps_rel"] == pytest.approx(1.0)
    assert table["tests_valid"].all()
    assert {"axis:model", "mcs_pvalue", "in_mcs_25", "energy", "variogram"} <= set(table.columns)


def test_var_beats_random_walk_on_stationary_data(rw_vs_var):
    verdict = rw_vs_var.verdict()
    assert (verdict["verdict"] == "better").all()
    assert (verdict["dlpd"] > 0).all()


def _with_converged(res: CompareResult, name: str, converged: np.ndarray) -> CompareResult:
    """``res`` with the per-origin ``converged`` flags of ``name`` replaced (a copy: the module
    fixture is not touched)."""
    r = res.results[name]
    scores = r.scores.assign(converged=("origin", converged))
    results = {**res.results, name: dataclasses.replace(r, scores=scores)}
    return CompareResult(res.entries, results, res.horizons, res.baseline, res.window, res.freq)


@pytest.mark.parametrize("baseline, expected", [("RW", "better"), ("VAR1", "worse")])
@pytest.mark.parametrize("side", ["model", "reference"])
def test_min_converged_gates_the_verdict(rw_vs_var, baseline, expected, side):
    """SPEC §9.5 item 4: the converged share of a pair (the smaller of the two shares over the
    common origins) below ``min_converged`` turns an otherwise ``better`` verdict, and with the
    baseline swapped an otherwise ``worse`` one, into ``inconclusive`` and says so; at or above
    the threshold the verdict and every other field stand, and ``min_converged=0.0`` disables
    the gate even when no origin converged."""
    f = rw_vs_var
    res = CompareResult(f.entries, f.results, f.horizons, baseline, f.window, f.freq)
    name = baseline if side == "reference" else next(n for n in res.results if n != baseline)
    k = 6
    origins = res.results[name].scores["origin"].values
    converged = np.ones(origins.size)
    converged[:k] = 0.0
    gated = _with_converged(res, name, converged)
    never = _with_converged(res, name, np.zeros(origins.size))
    for h in res.horizons:
        want = res.verdict(horizons=[h]).iloc[0]
        assert want["verdict"] == expected and want["reference"] == baseline and want[side] == name
        common = gated.common(h)[name]["origin"].values
        assert set(origins[:k]) <= set(common)
        share = float(np.mean(converged[np.isin(origins, common)]))
        assert share == (common.size - k) / common.size < 0.95
        note = f"; converged at only {share:.0%} of origins"
        default, just_above = (
            gated.verdict(horizons=[h]).iloc[0],  # min_converged=0.95
            gated.verdict(horizons=[h], min_converged=np.nextafter(share, 1.0)).iloc[0],
        )
        for row in (default, just_above):
            assert row["verdict"] == "inconclusive" and row["reasons"] == want["reasons"] + note
            pd.testing.assert_series_equal(
                row.drop(["verdict", "reasons"]), want.drop(["verdict", "reasons"])
            )
        for threshold in (share, share - 0.05, 0.0):  # at and below the share
            row = gated.verdict(horizons=[h], min_converged=threshold).iloc[0]
            pd.testing.assert_series_equal(row, want)
        pd.testing.assert_series_equal(never.verdict(horizons=[h], min_converged=0.0).iloc[0], want)
        assert never.verdict(horizons=[h]).iloc[0]["verdict"] == "inconclusive"


def test_expanding_window_tests_are_descriptive(data):
    res = compare(
        {"RW": RandomWalk(), "VAR1": OLSVAR(1)},
        data,
        origins=Origins("2015Q1"),
        window=Expanding(),
        draws=200,
    )
    assert not res.table()["tests_valid"].any()
    assert "descriptive only" in res.verdict()["reasons"].iloc[0]


def test_identical_models_are_inconclusive(data):
    res = compare(
        {"A": OLSVAR(1), "B": OLSVAR(1)},
        data,
        origins=Origins("2015Q1"),
        window=Rolling(80),
        draws=200,
    )
    row = res.verdict().iloc[0]
    assert row["dlpd"] == 0.0 and row["verdict"] == "inconclusive"


def test_two_axis_differences_are_not_comparable(data):
    entries = grid(OLSVAR, lags=[1, 2], uncertainty=["flat", "none"])
    res = compare(
        entries, data, origins=Origins("2016Q1"), window=Rolling(80), draws=200, baseline="1 × flat"
    )
    verdict = res.verdict().set_index("model")
    assert verdict.loc["2 × none", "verdict"] == "not comparable"
    by_axis = res.verdict(axis="lags").set_index("model")
    assert by_axis.loc["2 × none", "reference"] == "1 × none"
    assert by_axis.loc["2 × flat", "reference"] == "1 × flat"


def test_grid_labels_and_entries():
    entries = grid(OLSVAR, lags=[1, 2], deterministic=["c"])
    assert [e.name for e in entries] == ["1 × c", "2 × c"]
    assert entries[1].axes == {"lags": "2", "deterministic": "c"}
    assert isinstance(entries[0], Entry) and entries[0].model == OLSVAR(1, "c")


def test_duplicate_names_and_unknown_baseline_raise(data):
    with pytest.raises(ValueError):
        compare([OLSVAR(1), OLSVAR(1)], data, origins=Origins("2015Q1"))
    with pytest.raises(ValueError):
        compare({"A": OLSVAR(1)}, data, origins=Origins("2015Q1"), baseline="B")


def test_plot_cumulative_lpd(rw_vs_var):
    ax = rw_vs_var.plot_cumulative_lpd(horizon=1)
    assert len(ax.lines) == 2  # one model line plus the zero line
