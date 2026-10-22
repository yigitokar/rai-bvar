"""Pseudo out-of-sample backtests (SPEC §9.1).

At every forecast origin the model sees only data available at that origin (``upto``), cut to
the estimation window, and is fit afresh (SPEC §3 R9). Scores are cached on disk per
``(model spec, estimation sample, realised outcomes, origin, settings)``, so backtests resume
and adding a model to a comparison only fits the new model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar._version import __version__
from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation.scores import score_forecast
from rai_bvar.models.base import Model

DataSource = VARData | Callable[[pd.Period], VARData]


@dataclass(frozen=True)
class Expanding:
    """Estimate on all data up to the origin."""

    min_size: int | None = None
    valid_tests: bool = field(default=False, init=False, repr=False)

    def select(self, available: VARData) -> VARData:
        if self.min_size is not None and len(available) < self.min_size:
            raise ValueError(
                f"only {len(available)} observations, expanding window needs {self.min_size}"
            )
        return available

    def describe(self) -> dict[str, Any]:
        return {"window": "expanding", "min_size": self.min_size}


@dataclass(frozen=True)
class Rolling:
    """Estimate on the last ``size`` observations up to the origin (valid for GW/AG tests)."""

    size: int
    valid_tests: bool = field(default=True, init=False, repr=False)

    def select(self, available: VARData) -> VARData:
        if len(available) < self.size:
            raise ValueError(
                f"only {len(available)} observations, rolling window needs {self.size}"
            )
        return available.tail(self.size)

    def describe(self) -> dict[str, Any]:
        return {"window": "rolling", "size": self.size}


Window = Expanding | Rolling


@dataclass(frozen=True)
class Origins:
    """Forecast origins from ``start`` to ``end`` (default: second-to-last period) every ``step``."""

    start: str | pd.Period
    end: str | pd.Period | None = None
    step: int = 1

    def resolve(self, index: pd.PeriodIndex) -> list[pd.Period]:
        start = pd.Period(self.start, freq=index.freq)
        end = pd.Period(self.end, freq=index.freq) if self.end is not None else index[-2]
        if start < index[0] or end > index[-1] or start > end:
            raise ValueError(f"origins {start}..{end} must lie within {index[0]}..{index[-1]}")
        return list(pd.period_range(start, end, freq=index.freq)[:: self.step])


class Cache:
    """On-disk store of per-origin score datasets (netCDF)."""

    def __init__(self, directory: str | os.PathLike = ".rai-bvar-cache"):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.nc"

    def get(self, key: str) -> xr.Dataset | None:
        path = self._path(key)
        return xr.load_dataset(path, engine="scipy") if path.exists() else None

    def put(self, key: str, scores: xr.Dataset) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        scores.to_netcdf(tmp, engine="scipy")
        os.replace(tmp, path)

    def clear(self) -> None:
        for path in self.directory.glob("*/*.nc"):
            path.unlink()


@dataclass
class BacktestResult:
    name: str
    axes: dict[str, str]
    spec: dict[str, Any]
    window: Window
    scores: xr.Dataset  # dims (origin, horizon[, variable])
    seconds: float
    cached: int
    warnings: dict[str, list[str]] = field(default_factory=dict)  # message → origins (SPEC §10.6)

    @property
    def origins(self) -> list[str]:
        return [str(o) for o in self.scores["origin"].values]

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.name!r}, origins={len(self.origins)}, "
            f"horizons={self.scores.sizes['horizon']}, cached={self.cached}, seconds={self.seconds:.1f})"
        )


def backtest(
    model: Model,
    data: DataSource,
    *,
    origins: Origins | Sequence[str | pd.Period],
    window: Window = Expanding(),
    horizon: int = 8,
    draws: int = 1000,
    seed: int = 0,
    cache: Cache | str | os.PathLike | None = None,
    n_jobs: int = 1,
    realized: VARData | None = None,
    name: str | None = None,
    axes: dict[str, str] | None = None,
) -> BacktestResult:
    """Fit and score ``model`` at every origin.

    ``data`` is either one ``VARData`` (pseudo real time: each origin sees ``data.upto(origin)``)
    or a callable returning the vintage available at an origin. Outcomes are scored against
    ``realized`` (default: ``data`` itself, which is required when ``data`` is a callable).
    Each origin gets its own random stream derived from ``(seed, origin)``, so results do not
    depend on which other origins run or on parallel scheduling.
    """
    if realized is None:
        if not isinstance(data, VARData):
            raise ValueError("pass realized= when data is a callable")
        realized = data
    origin_list = (
        origins.resolve(realized.index)
        if isinstance(origins, Origins)
        else [pd.Period(o, freq=realized.index.freq) for o in origins]
    )
    cache = Cache(cache) if cache is not None and not isinstance(cache, Cache) else cache
    tasks = [(model, data, realized, o, window, horizon, draws, seed, cache) for o in origin_list]

    start = time.perf_counter()
    if n_jobs == 1:
        outputs = [_run_origin(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
            outputs = list(pool.map(_run_origin, *zip(*tasks, strict=True)))
    by_message: dict[str, list[str]] = {}
    for ds, _, passthrough in outputs:
        for message in filter(None, str(ds.attrs.get("warnings", "")).split("\n")):
            by_message.setdefault(message, []).append(str(ds["origin"].values[0]))
        for category, message, filename, lineno in passthrough:  # e.g. RuntimeWarnings from workers
            warnings.warn_explicit(message, category, filename, lineno)
    _emit_aggregated(by_message, len(outputs), name or model.name)
    scores = xr.concat([ds for ds, _, _ in outputs], dim="origin")
    scores.attrs.pop("warnings", None)
    return BacktestResult(
        warnings=by_message,
        name=name or model.name,
        axes=dict(axes if axes is not None else model.axes),
        spec=model.spec(),
        window=window,
        scores=scores,
        seconds=time.perf_counter() - start,
        cached=sum(hit for _, hit, _ in outputs),
    )


def estimate_cost(
    model: Model,
    data: DataSource,
    *,
    origins: Origins | Sequence[str | pd.Period],
    window: Window = Expanding(),
    horizon: int = 8,
    draws: int = 1000,
    realized: VARData | None = None,
) -> dict[str, float]:
    """Time one origin (the last, usually the largest sample) and extrapolate to all origins."""
    realized = realized if realized is not None else data
    if not isinstance(realized, VARData):
        raise ValueError("pass realized= when data is a callable")
    origin_list = origins.resolve(realized.index) if isinstance(origins, Origins) else list(origins)
    start = time.perf_counter()
    _run_origin(
        model,
        data,
        realized,
        pd.Period(origin_list[-1], freq=realized.index.freq),
        window,
        horizon,
        draws,
        0,
        None,
    )
    per_origin = time.perf_counter() - start
    return {
        "seconds_per_origin": per_origin,
        "origins": len(origin_list),
        "seconds": per_origin * len(origin_list),
    }


def _run_origin(
    model: Model,
    data: DataSource,
    realized: VARData,
    origin: pd.Period,
    window: Window,
    horizon: int,
    draws: int,
    seed: int,
    cache: Cache | None,
) -> tuple[xr.Dataset, bool, list]:
    available = (data if isinstance(data, VARData) else data(origin)).upto(origin)
    sample = window.select(available)
    truth = realized.realized(origin, horizon)
    key = _cache_key(model, sample, truth, origin, window, horizon, draws, seed)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit, True, []
    rng = np.random.default_rng([seed, zlib.crc32(str(origin).encode())])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        forecast = model.fit(sample).forecast(horizon, draws, seed=rng)
    messages, passthrough = [], []
    for w in caught:  # user warnings are aggregated across origins; others go back to the caller
        if issubclass(w.category, UserWarning):
            if str(w.message) not in messages:
                messages.append(str(w.message))
        else:
            passthrough.append((w.category, str(w.message), w.filename, w.lineno))
    scores = score_forecast(forecast, truth, rng=rng)
    scores["converged"] = float(forecast.dataset.attrs.get("converged", 1))
    scores = scores.expand_dims(origin=[str(origin)])
    scores.attrs = {
        "sample_start": str(sample.first),
        "sample_end": str(sample.last),
        "n_obs": len(sample),
        "warnings": "\n".join(messages),
    }
    if cache is not None:
        cache.put(key, scores)
    return scores, False, passthrough


def _cache_key(model, sample, truth, origin, window, horizon, draws, seed) -> str:
    payload = {
        "spec": model.spec(),
        "sample": sample.fingerprint(),
        "truth": hashlib.sha256(np.nan_to_num(truth, nan=-1.2345e300).tobytes()).hexdigest(),
        "origin": str(origin),
        "window": window.describe(),
        "horizon": horizon,
        "draws": draws,
        "seed": seed,
        "version": __version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


__all__ = [
    "BacktestResult",
    "Cache",
    "Expanding",
    "Origins",
    "Rolling",
    "backtest",
    "estimate_cost",
]


def _emit_aggregated(by_message: dict[str, list[str]], n_origins: int, model: str) -> None:
    """One warning per distinct message; messages that differ only in numbers are grouped."""
    groups: dict[str, list[tuple[str, list[str]]]] = {}
    for message, origins in by_message.items():
        key = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", message)
        groups.setdefault(key, []).append((message, origins))
    for items in groups.values():
        origins = sorted({o for _, o_list in items for o in o_list})
        example = items[0][0]
        more = f" (and {len(items) - 1} variants with other numbers)" if len(items) > 1 else ""
        warnings.warn(
            f"{model}: {example}{more} [at {len(origins)} of {n_origins} origins: {origins[0]}..{origins[-1]}]",
            UserWarning,
            stacklevel=3,
        )
