"""Backtest integration of the MCMC engine (DESIGN §11.6, §13 WP23).

GOLDEN_CASES (placeholder; the tests follow once the golden is written).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rai_bvar import BVAR, Linear, Minnesota, VARData
from rai_bvar.evaluation.backtest import Expanding, Rolling, backtest
from rai_bvar.models import OLSVAR, RandomWalk
from rai_bvar.priors.minnesota import MHSettings

bt_mod = importlib.import_module("rai_bvar.evaluation.backtest")

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "backtest_analytic_golden.npz"
GOLDEN_VERSION_KEY = "__version__"


def golden_data(T: int = 100, seed: int = 11) -> VARData:
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.05, 0.4]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.1 + y[t - 1] @ A.T + rng.normal(scale=0.5, size=2)
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
    data = golden_data()
    return backtest(model, data, horizon=3, draws=200, **kwargs)


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
            bt_mod._cache_key(model, sample, truth, origin, kwargs["window"], 3, 200, kwargs["seed"])
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


if __name__ == "__main__":  # pragma: no cover
    if sys.argv[1:] == ["--write-golden"]:
        write_golden()
