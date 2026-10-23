"""Pseudo out-of-sample backtests (SPEC §9.1; DESIGN §11.6).

At every forecast origin the model sees only data available at that origin (``upto``), cut to
the estimation window, and is fit afresh (SPEC §3 R9). Scores are cached on disk per
``(model spec, estimation sample, realised outcomes, origin, settings)``, so backtests resume
and adding a model to a comparison only fits the new model.

MCMC models (DESIGN §11.6, WP23):

* **Fit seeds.** A model that declares ``uses_fit_seed`` is fit with
  ``fit(sample, seed=fit_seed)``, ``fit_seed = default_rng([seed, crc32(origin),
  0x4D434D43]).integers(2**31)`` (:func:`fit_seed`); every other model is called exactly as in
  Phase 1, ``fit(sample)``. The forecast generator is unchanged. The backtest seed is in the cache
  key and the MCMC settings are in the spec, so the key still determines the result.
* **Padding.** Each origin's fit pads its own window to ``T_pad = t_bucket·⌈T_eff/t_bucket⌉``
  (``engines.mcmc.resolve``), never to a backtest maximum, so results depend only on ``(spec,
  data, seed)``; an expanding backtest compiles once per ``T`` bucket.
* **Processes.** With an MCMC model and ``n_jobs > 1`` a ``UserWarning`` is issued (XLA is
  already multi-threaded) and the pool starts its workers with
  ``multiprocessing.get_context("spawn")`` (forking after JAX has initialised is unsafe). A
  script that runs such a backtest must guard its entry point with ``if __name__ ==
  "__main__":`` and pass a picklable model and data source. Other models keep the Phase 1 pool.
* **Warm starts** (``warm_start=True``, opt-in; DESIGN §5.5, D-32): the origins of a model that
  declares ``supports_warm_start`` run sequentially in one process; each fit gets
  ``init=<previous origin's final_state>`` (the first origin is a cold start). The final states
  are cached next to the scores as ``<key>.state.npz`` (plain arrays and JSON, no pickles) so a
  resumed backtest continues warm, and the cache key of every origin after the first gains
  ``"init": "warm:<previous origin key>"``. The first origin's key equals the cold key (it *is*
  the cold fit; a dict entry ``"init": "cold"`` would only duplicate work). An origin whose
  scores are cached but whose state is needed and missing (e.g. scores cached by a cold run) is
  refit, which reproduces the same state. ``n_jobs`` is ignored, with a warning, in this mode.
  The origins must be strictly increasing: a warm start from a later origin would carry data
  after the earlier origin into its fit (R9).
* **Cache keys.** Analytic models keep the Phase 1 keys. Models with ``uses_fit_seed`` add
  ``"seeding": "fit_seed-v1"``, so scores cached before the per-origin fit seeds (fit with
  ``MCMCSettings.seed`` at every origin) are never served as results of the current seeding.
* **LPD estimator** (``lpd_method``, DESIGN §7.3, D-34): passed to
  :func:`~rai_bvar.evaluation.scores.score_forecast` at every origin (a plain string, so it
  pickles for the worker pool). The default ``"conditioned"`` is the SPEC §7 estimator and leaves
  the scores, the score layout and the cache keys exactly as before. Any other value adds
  ``"lpd": "<method>-v1"`` to the cache key of every model (the estimator changes the scores,
  not the fit) and ``attrs["lpd_method"]`` to the scores; ``"integrated"`` scores also carry the
  default estimator next to it (``score_forecast``). At ``h > 1`` the integrated estimator needs
  the ``lag_coefs`` of MCMC forecasts: an analytic or benchmark model raises ``ValueError`` at
  its first origin, after that origin's fit, with the model's name and the origin in the message
  (``compare`` runs its backtests one after another, so list such a model first or pass a
  cache). Changing ``lpd_method`` on an existing cache refits every origin (and every warm
  chain), although fits and forecasts do not depend on it; ``"integrated"`` stores the default
  estimator too, so one run gives both.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import re
import time
import warnings
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar._version import __version__
from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation.scores import score_forecast
from rai_bvar.forecast.result import LPD_METHODS
from rai_bvar.models.base import Model

DataSource = VARData | Callable[[pd.Period], VARData]
DEFAULT_LPD = "conditioned"  # the SPEC §7 estimator: Phase 1 / WP23 scores and cache keys


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
    """On-disk store of per-origin score datasets (netCDF) and warm-start states (``.npz``)."""

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

    def state_path(self, key: str) -> Path:
        """Where the final MCMC state of the origin with cache key ``key`` is stored."""
        return self.directory / key[:2] / f"{key}.state.npz"

    def get_state(self, key: str) -> Any:
        """The cached ``FinalState`` of ``key`` (warm starts, DESIGN §11.6), or ``None``."""
        path = self.state_path(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as npz:
            return _final_state_from_arrays({k: npz[k] for k in npz.files})

    def put_state(self, key: str, final_state: Any) -> None:
        """Store a ``FinalState`` next to the scores of ``key`` (written atomically)."""
        buffer = io.BytesIO()
        np.savez(buffer, **_final_state_to_arrays(final_state))
        path = self.state_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(buffer.getvalue())
        os.replace(tmp, path)

    def clear(self) -> None:
        for pattern in ("*/*.nc", "*/*.state.npz"):
            for path in self.directory.glob(pattern):
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
    warm_start: bool = False,
    lpd_method: Literal["conditioned", "integrated"] = DEFAULT_LPD,
) -> BacktestResult:
    """Fit and score ``model`` at every origin.

    ``data`` is either one ``VARData`` (pseudo real time: each origin sees ``data.upto(origin)``)
    or a callable returning the vintage available at an origin. Outcomes are scored against
    ``realized`` (default: ``data`` itself, which is required when ``data`` is a callable).
    Each origin gets its own random stream derived from ``(seed, origin)``, so results do not
    depend on which other origins run or on parallel scheduling; models that declare
    ``uses_fit_seed`` (MCMC) also get a per-origin fit seed (:func:`fit_seed`).

    ``warm_start=True`` (models that declare ``supports_warm_start`` only) runs the origins in
    order and starts each fit from the previous origin's final state (module docstring); results
    then depend on the previous origins, which the cache key records.

    ``lpd_method`` selects the LPD estimator of the scores (``"conditioned"``, the default, or
    ``"integrated"``; module docstring, DESIGN §7.3, D-34).
    """
    _check_lpd_method(lpd_method)
    if not isinstance(warm_start, bool | np.bool_):
        raise TypeError(f"warm_start must be a bool, got {warm_start!r}")
    if warm_start and not getattr(model, "supports_warm_start", False):
        raise ValueError(
            "warm_start=True needs a model that supports warm starts (an MCMC model); "
            f"{getattr(model, 'name', type(model).__name__)} does not"
        )
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
    tasks = [
        (model, data, realized, o, window, horizon, draws, seed, cache, lpd_method)
        for o in origin_list
    ]

    start = time.perf_counter()
    if warm_start:
        # origin i is fit from the final state of origin i − 1: out of time order, that state was
        # sampled on data after origin i (an R9 look-ahead through init)
        if any(b <= a for a, b in itertools.pairwise(origin_list)):
            raise ValueError("warm_start=True needs strictly increasing origins (time order)")
        if n_jobs != 1:
            warnings.warn(
                "warm_start=True runs the origins sequentially in one process; n_jobs is ignored",
                UserWarning,
                stacklevel=2,
            )
        outputs = _run_warm(
            model, data, realized, origin_list, window, horizon, draws, seed, cache, lpd_method
        )
    elif n_jobs == 1:
        outputs = [_run_origin(*task) for task in tasks]
    else:
        workers = None if n_jobs < 0 else n_jobs
        if getattr(model, "uses_fit_seed", False):  # MCMC (DESIGN §11.6)
            warnings.warn(
                f"n_jobs={n_jobs} with an MCMC model: XLA already runs each fit on several "
                "threads, so worker processes compete for the same cores (no verified "
                "single-thread recipe, N07 §6); workers are started with 'spawn'",
                UserWarning,
                stacklevel=2,
            )
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
        with pool:
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
    lpd_method: Literal["conditioned", "integrated"] = DEFAULT_LPD,
) -> dict[str, float]:
    """Time one origin (the last, usually the largest sample) and extrapolate to all origins."""
    _check_lpd_method(lpd_method)
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
        lpd_method,
    )
    per_origin = time.perf_counter() - start
    return {
        "seconds_per_origin": per_origin,
        "origins": len(origin_list),
        "seconds": per_origin * len(origin_list),
    }


FIT_SEED_SALT = 0x4D434D43
"""Third entropy word of the per-origin fit seed (ASCII ``"MCMC"``; DESIGN §11.6)."""


def fit_seed(seed: int, origin: pd.Period | str) -> int:
    """The fit seed at ``origin`` for models with ``uses_fit_seed`` (DESIGN §11.6)::

    default_rng([seed, crc32(str(origin)), 0x4D434D43]).integers(2**31)
    """
    crc = zlib.crc32(str(origin).encode())
    return int(np.random.default_rng([seed, crc, FIT_SEED_SALT]).integers(2**31))


def _check_lpd_method(lpd_method: Any) -> None:
    if lpd_method not in LPD_METHODS:  # up front: a worker would only fail at its first h > 1
        raise ValueError(f"lpd_method must be one of {LPD_METHODS}, got {lpd_method!r}")


def _lpd_kwargs(lpd_method: str) -> dict[str, str]:
    """``lpd_method=`` for :func:`_fit_and_score` only when it is not the default, so the default
    call stays the WP23 one."""
    return {} if lpd_method == DEFAULT_LPD else {"lpd_method": lpd_method}


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
    lpd_method: str = DEFAULT_LPD,
) -> tuple[xr.Dataset, bool, list]:
    sample, truth = _origin_inputs(data, realized, origin, window, horizon)
    key = _cache_key(
        model, sample, truth, origin, window, horizon, draws, seed, lpd_method=lpd_method
    )
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit, True, []
    scores, passthrough, _ = _fit_and_score(
        model, sample, truth, origin, horizon, draws, seed, **_lpd_kwargs(lpd_method)
    )
    if cache is not None:
        cache.put(key, scores)
    return scores, False, passthrough


def _origin_inputs(
    data: DataSource, realized: VARData, origin: pd.Period, window: Window, horizon: int
) -> tuple[VARData, np.ndarray]:
    available = (data if isinstance(data, VARData) else data(origin)).upto(origin)
    return window.select(available), realized.realized(origin, horizon)


def _fit_and_score(
    model: Model,
    sample: VARData,
    truth: np.ndarray,
    origin: pd.Period,
    horizon: int,
    draws: int,
    seed: int,
    init: Any = None,
    lpd_method: str = DEFAULT_LPD,
) -> tuple[xr.Dataset, list, Any]:
    """Fit on ``sample``, forecast and score one origin; returns ``(scores, passthrough, fit)``.

    Models without ``uses_fit_seed`` are called exactly as in Phase 1, ``fit(sample)``. The
    forecast generator ``default_rng([seed, crc32(origin)])`` is the Phase 1 one for every model.
    ``lpd_method`` goes to ``score_forecast`` (it uses no random numbers, so every other score is
    the same whatever the estimator). A ``ValueError`` from scoring is re-raised with the model's
    name and the origin in front of its message (a plain string: it pickles from a worker).
    """
    rng = np.random.default_rng([seed, zlib.crc32(str(origin).encode())])
    uses_seed = bool(getattr(model, "uses_fit_seed", False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        if uses_seed:
            kwargs: dict[str, Any] = {"seed": fit_seed(seed, origin)}
            if init is not None:
                kwargs["init"] = init
            fit = model.fit(sample, **kwargs)
        else:
            fit = model.fit(sample)
        forecast = fit.forecast(horizon, draws, seed=rng)
    messages, passthrough = [], []
    for w in caught:  # user warnings are aggregated across origins; others go back to the caller
        if issubclass(w.category, UserWarning):
            if str(w.message) not in messages:
                messages.append(str(w.message))
        else:
            passthrough.append((w.category, str(w.message), w.filename, w.lineno))
    try:
        scores = score_forecast(forecast, truth, rng=rng, lpd_method=lpd_method)
    except ValueError as e:  # e.g. an integrated LPD without lag_coefs: say which model and origin
        name = getattr(model, "name", type(model).__name__)
        raise ValueError(f"{name} at origin {origin}: {e}") from e
    scores["converged"] = float(forecast.dataset.attrs.get("converged", 1))
    scores = scores.expand_dims(origin=[str(origin)])
    scores.attrs = {
        "sample_start": str(sample.first),
        "sample_end": str(sample.last),
        "n_obs": len(sample),
        "warnings": "\n".join(messages),
    }
    if lpd_method != DEFAULT_LPD:  # the cached scores say which estimator they hold
        scores.attrs["lpd_method"] = lpd_method
    if uses_seed:
        scores.attrs["fit_seed"] = fit_seed(seed, origin)
        scores.attrs["warm_start"] = int(init is not None)
    return scores, passthrough, fit


def _run_warm(
    model: Model,
    data: DataSource,
    realized: VARData,
    origins: list[pd.Period],
    window: Window,
    horizon: int,
    draws: int,
    seed: int,
    cache: Cache | None,
    lpd_method: str = DEFAULT_LPD,
) -> list[tuple[xr.Dataset, bool, list]]:
    """Warm-started origins, in order (DESIGN §11.6; module docstring).

    Origin ``i > 0`` is fit with ``init`` = the final state of origin ``i − 1`` and keyed with
    ``"init": "warm:<key of origin i − 1>"``. The scores of a cached origin are reused; its final
    state is loaded from ``<key>.state.npz`` only when the next origin must be fit. When that
    file is missing, the origins back to the last available state (or to the cold first origin)
    are refit, which reproduces their states exactly.
    """
    inputs: list[tuple[VARData, np.ndarray]] = []
    keys: list[str] = []
    states: dict[int, Any] = {}  # in-memory final states by origin index

    def refit(j: int, init: Any) -> tuple[xr.Dataset, list, Any]:
        sample, truth = inputs[j]
        scores, passthrough, fit = _fit_and_score(
            model, sample, truth, origins[j], horizon, draws, seed, init=init,
            **_lpd_kwargs(lpd_method),
        )  # fmt: skip
        if cache is not None:  # the state first: cached scores then imply a cached state
            cache.put_state(keys[j], fit.final_state)
            cache.put(keys[j], scores)
        states[j] = fit.final_state
        return scores, passthrough, fit

    def state_of(j: int) -> Any:
        if j in states:
            return states[j]
        if cache is not None and (state := cache.get_state(keys[j])) is not None:
            states[j] = state
            return state
        # missing: walk back to the last origin with a known state (or the cold first origin)
        start = j
        while start > 0 and start - 1 not in states:
            if cache is not None and (state := cache.get_state(keys[start - 1])) is not None:
                states[start - 1] = state
                break
            start -= 1
        for m in range(start, j + 1):
            refit(m, None if m == 0 else states[m - 1])
        return states[j]

    outputs = []
    for i, origin in enumerate(origins):
        sample, truth = _origin_inputs(data, realized, origin, window, horizon)
        inputs.append((sample, truth))
        init_tag = None if i == 0 else f"warm:{keys[i - 1]}"
        key = _cache_key(
            model, sample, truth, origin, window, horizon, draws, seed, init=init_tag,
            lpd_method=lpd_method,
        )  # fmt: skip
        keys.append(key)
        if cache is not None and (hit := cache.get(keys[i])) is not None:
            outputs.append((hit, True, []))
            continue
        scores, passthrough, _ = refit(i, None if i == 0 else state_of(i - 1))
        for j in [j for j in states if j < i]:  # only the latest state is needed from here on
            del states[j]
        outputs.append((scores, False, passthrough))
    return outputs


def _cache_key(
    model, sample, truth, origin, window, horizon, draws, seed, init=None, lpd_method=DEFAULT_LPD
) -> str:
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
    if getattr(model, "uses_fit_seed", False):  # per-origin fit seeds (§11.6) since this tag;
        payload["seeding"] = "fit_seed-v1"  # scores cached before it used MCMCSettings.seed
    if init is not None:  # warm starts only: cold keys stay the Phase 1 keys (DESIGN §11.6)
        payload["init"] = init
    if lpd_method != DEFAULT_LPD:  # another estimator, other scores (D-34); default keys unchanged
        payload["lpd"] = f"{lpd_method}-v1"
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- warm-state files


def _final_state_to_arrays(final_state: Any) -> dict[str, np.ndarray]:
    """``FinalState`` → plain arrays (no pickles): the state leaves by dotted path, the period
    labels, and the ``StaticConfig`` and seed metadata as JSON strings."""
    import jax

    from rai_bvar.engines.mcmc.state import leaf_path

    out: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(final_state.state)[0]:
        out[f"leaf:{leaf_path(path)}"] = np.asarray(leaf)
    periods = final_state.periods
    if isinstance(periods, pd.PeriodIndex):
        out["periods"] = np.array([str(p) for p in periods])
        out["periods_freq"] = np.array(periods.freqstr)
    else:
        out["periods"] = np.asarray(periods)
        out["periods_freq"] = np.array("")
    out["static"] = np.array(json.dumps(_static_to_dict(final_state.static), sort_keys=True))
    out["seed_meta"] = np.array(json.dumps(final_state.seed_meta, sort_keys=True, default=int))
    return out


def _final_state_from_arrays(arrays: dict[str, np.ndarray]) -> Any:
    import jax

    from rai_bvar.engines.mcmc.state import leaf_path, state_template
    from rai_bvar.engines.mcmc.warmstart import FinalState

    static = _static_from_dict(json.loads(str(arrays["static"])))
    flat, treedef = jax.tree_util.tree_flatten_with_path(state_template(static))
    names = [leaf_path(p) for p, _ in flat]
    stored = {k.removeprefix("leaf:") for k in arrays if k.startswith("leaf:")}
    if stored != set(names):
        raise ValueError("the cached warm-start state does not match its StaticConfig")
    state = jax.tree.unflatten(treedef, [np.asarray(arrays[f"leaf:{n}"]) for n in names])
    freq = str(arrays["periods_freq"])
    labels = arrays["periods"]
    periods = pd.PeriodIndex([str(x) for x in labels], freq=freq) if freq else pd.Index(labels)
    return FinalState(
        state=state,
        periods=periods,
        static=static,
        seed_meta=json.loads(str(arrays["seed_meta"])),
    )


def _static_to_dict(static: Any) -> dict[str, Any]:
    out = dataclasses.asdict(static)
    out["float32_leaves"] = sorted(static.float32_leaves)
    out["vol"]["nu_grid"] = list(static.vol.nu_grid)
    if _static_from_dict(json.loads(json.dumps(out))) != static:  # pragma: no cover
        raise ValueError("this StaticConfig does not round-trip through JSON")
    return out


def _static_from_dict(d: dict[str, Any]) -> Any:
    from rai_bvar.engines.mcmc import config as c

    return c.StaticConfig(
        shapes=c.Shapes(**d["shapes"]),
        struct=c.StructStatic(**d["struct"]),
        mean=c.MeanStatic(**d["mean"]),
        coef=c.CoefPriorStatic(**d["coef"]),
        vol=c.VolStatic(**{**d["vol"], "nu_grid": tuple(d["vol"]["nu_grid"])}),
        store=c.StoreStatic(**d["store"]),
        batching=d["batching"],
        chunk=d["chunk"],
        float32_leaves=frozenset(d["float32_leaves"]),
    )


__all__ = [
    "BacktestResult",
    "Cache",
    "Expanding",
    "Origins",
    "Rolling",
    "backtest",
    "estimate_cost",
    "fit_seed",
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
