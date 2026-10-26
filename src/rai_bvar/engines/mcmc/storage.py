"""Storage plan, host store and the posterior ``DataTree`` (DESIGN §5.3, §5.6).

Streams emitted by the ``sample`` executable for every kept draw of every chain, and what the
host keeps of them:

=========  ==========================================================  ============================
stream     content (one chain, one record)                              kept draws stored
=========  ==========================================================  ============================
monitor    ``monitor.monitor_values`` (WP15): primary and secondary     every kept draw
params     ``ParamDraw``: γ or Π, ``a``, ``VolDraw``                    ``d ≡ 0 (mod store_thin)``,
                                                                        ``store_thin = ⌈draws/store_max⌉``
paths      ``PathDraw``: ``h`` and ``2 log o + log τ`` (``store="paths"``)  ``d ≡ 0 (mod store_thin·path_thin)``
forecast   in-loop ``ForecastOut`` (``FitPlan``; DESIGN §5.8), ``R`` paths  the §7.1 selection
           per selected draw                                            (:func:`forecast_selection`)
=========  ==========================================================  ============================

Latent moments (``rb_*``) and acceptance rates (``acc_*``) come once per run from the final
state's accumulators. :func:`plan_storage` sizes every stream before sampling and raises
:class:`StorageBudgetError` when the total exceeds ``Runtime.memory_budget_gb`` and no
``Runtime.storage_dir`` is set (the budget only decides whether to raise, never what is
computed; E14). :class:`HostStore` holds preallocated NumPy arrays ``(*lead, D, …)`` (``lead =
(C,)``, or ``(F, C)`` with a fit axis), or ``np.memmap`` files in a fresh directory under
``storage_dir``; float32 storage is not offered (R4).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar.engines.mcmc.state import leaf_path

if TYPE_CHECKING:
    from rai_bvar.engines.mcmc.config import StaticConfig
    from rai_bvar.engines.mcmc.results import FitMeta
    from rai_bvar.settings import MCMCSettings, Runtime

__all__ = [
    "STREAMS",
    "HostStore",
    "StorageBudgetError",
    "StoragePlan",
    "StreamSpec",
    "forecast_selection",
    "plan_storage",
    "regressor_names",
]

STREAMS: tuple[str, ...] = ("monitor", "params", "paths", "forecast")
"""Position of each stream in the ``sample`` output tuple ``(monitor, params, paths[, forecast])``
(``forecast`` only with an in-loop forecast plan, DESIGN §5.8)."""


def forecast_selection(
    chains: int, stored_index: Sequence[int] | np.ndarray, forecast_draws: int
) -> tuple[np.ndarray, int]:
    """The draws a forecast of ``S = forecast_draws`` paths uses (DESIGN §7.1; in-loop §5.8).

    ``stored_index`` holds the kept-draw indices of the ``D_s`` stored parameter draws of a chain
    (``0, st, 2st, …``, ``st = store_thin``). Every chain uses the same ``m = ⌈S/C⌉`` of them,
    evenly spaced from the first: stored positions ``⌊j·D_s/m⌋``, ``j = 0..m−1``, one path each
    (``C·m ≥ S`` paths in all, the excess below ``C``). When ``m > D_s`` every stored draw is
    used with ``R = ⌈S/(C·D_s)⌉`` paths with independent shocks. Returns ``(kept-draw indices of
    the selected draws (ascending), R)``. Each path is keyed by ``(fseed, c, d)`` with ``d`` the
    kept-draw index (DESIGN §5.6); path ``q`` of a draw with ``R > 1`` uses ``d + q·D`` (``D``
    kept draws per chain), so path ``0`` is the single-path key and every path has its own key.
    """
    C, S = int(chains), int(forecast_draws)
    idx = np.asarray(stored_index, dtype=np.int64).reshape(-1)
    if C < 1:
        raise ValueError(f"chains must be positive, got {C}")
    if S < 1:
        raise ValueError(f"FitPlan.draws must be positive, got {S}")
    D_s = idx.size
    if D_s == 0:
        raise ValueError("no stored draws to forecast from")
    m = -(-S // C)
    if m <= D_s:
        return idx[(np.arange(m, dtype=np.int64) * D_s) // m], 1
    return idx.copy(), -(-S // (C * D_s))


_GIB = 1024**3


class StorageBudgetError(ValueError):
    """The storage plan exceeds ``Runtime.memory_budget_gb`` and no ``storage_dir`` is set."""


@dataclass(frozen=True)
class StreamSpec:
    """One stored stream: record shapes (one chain, one draw), thinning and stored count."""

    name: str
    record: Any  # pytree of jax.ShapeDtypeStruct
    every: int  # kept draws per stored record (0 when ``index`` selects them)
    count: int  # stored records per chain (and fit)
    index: np.ndarray | None = field(default=None, compare=False)  # explicit kept-draw indices

    @property
    def bytes_per_record(self) -> int:
        return int(
            sum(
                math.prod(x.shape) * np.dtype(x.dtype).itemsize
                for x in jax.tree.leaves(self.record)
            )
        )

    def draw_index(self) -> np.ndarray:
        """Kept-draw indices of the stored records."""
        if self.index is not None:
            return np.asarray(self.index, dtype=np.int64).copy()
        return np.arange(self.count, dtype=np.int64) * self.every

    def select(self, d0: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Of kept draws ``d0 .. d0+stop``: the record positions ``j`` (``d = d0 + j``) this
        stream keeps and their slots in the store."""
        j = np.arange(int(stop) + 1, dtype=np.int64)
        d = d0 + j
        if self.index is None:
            hit = d % self.every == 0
            return j[hit], d[hit] // self.every
        idx = np.asarray(self.index, dtype=np.int64)
        pos = np.searchsorted(idx, d)
        hit = pos < idx.size
        hit[hit] = idx[pos[hit]] == d[hit]
        return j[hit], pos[hit]


@dataclass(frozen=True)
class StoragePlan:
    """Bytes per stream for the whole run (all chains) and the budget they were checked against."""

    streams: Mapping[str, StreamSpec]
    lead: tuple[int, ...]
    bytes_by_stream: Mapping[str, int]
    budget_bytes: int
    storage_dir: str | None = None

    @property
    def total_bytes(self) -> int:
        return int(sum(self.bytes_by_stream.values()))


def _record_from_runner(static: StaticConfig) -> tuple[Any, ...]:
    """Record shapes of the production extract (and in-loop forecast) via ``eval_shape``."""
    from rai_bvar.engines.mcmc.runner import forecast_chain_keys, make_runner
    from rai_bvar.engines.mcmc.state import prior_template, state_template

    data = _abstract_data(static)
    prior = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), prior_template(static))
    st = jax.tree.map(lambda x: jax.ShapeDtypeStruct(np.shape(x), np.asarray(x).dtype),
                      state_template(static))  # fmt: skip
    fns = make_runner(static)
    cache = jax.eval_shape(fns.prepare, data, prior)
    record = jax.eval_shape(fns.extract, st, data, prior, cache)
    if static.store.in_loop_forecast and fns.chain is not None and fns.chain.forecast is not None:
        fkey = forecast_chain_keys(0, 1)[0]
        fc = jax.eval_shape(fns.chain.forecast, record[1], data, fkey, np.uint32(0))
        record = (*record, fc)
    return record


def _abstract_data(static: StaticConfig) -> Any:
    from rai_bvar.engines.mcmc.state import EngineData

    s = static.shapes
    f64, i32 = np.float64, np.int32
    return EngineData(
        Y=jax.ShapeDtypeStruct((s.T_pad, s.n), f64),
        X=jax.ShapeDtypeStruct((s.T_pad, s.k), f64),
        tmask=jax.ShapeDtypeStruct((s.T_pad,), f64),
        T_eff=jax.ShapeDtypeStruct((), i32),
        hist=jax.ShapeDtypeStruct((s.lags, s.n), f64),
        t_last=jax.ShapeDtypeStruct((), f64),
    )


def plan_storage(
    static: StaticConfig,
    settings: MCMCSettings,
    runtime: Runtime,
    *,
    record: Sequence[Any] | None = None,
    lead: tuple[int, ...] | None = None,
    plan: Any = None,
) -> StoragePlan:
    """Size every stream (DESIGN §5.3); raise :class:`StorageBudgetError` over budget.

    ``record`` is the ``(monitor, params, paths[, forecast])`` record of one chain and one draw as
    pytrees of ``jax.ShapeDtypeStruct`` (the runner passes the shapes of its own ``extract`` and
    forecast); by default it is derived from the production ``make_runner(static)`` with
    ``jax.eval_shape``. ``lead`` defaults to ``(settings.chains,)``. ``plan`` (a ``FitPlan``,
    needed when ``static.store.in_loop_forecast > 0``) sizes the ``forecast`` stream:
    the draws of :func:`forecast_selection` over the stored parameter draws, each record with a
    leading axis of ``R`` paths.
    """
    if record is None:
        record = _record_from_runner(static)
    lead = (settings.chains,) if lead is None else tuple(int(x) for x in lead)
    draws, st = settings.draws, settings.store_thin
    every = {"monitor": 1, "params": st, "paths": st * settings.path_thin}
    names = ["monitor", "params"] + (["paths"] if static.store.paths else [])
    streams = {
        name: StreamSpec(name, record[k], every[name], -(-draws // every[name]))
        for k, name in enumerate(STREAMS)
        if name in names
    }
    if static.store.in_loop_forecast:
        if plan is None:
            raise ValueError("an in-loop forecast (StoreStatic.in_loop_forecast) needs its FitPlan")
        index, reps = forecast_selection(
            settings.chains, streams["params"].draw_index(), plan.draws
        )
        fc_record = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct((reps, *x.shape), x.dtype),
            record[STREAMS.index("forecast")],
        )
        streams["forecast"] = StreamSpec("forecast", fc_record, 0, int(index.size), index)
    n, T1 = static.shapes.n, static.shapes.T_pad + 1
    per_lead = math.prod(lead)
    sizes = {name: per_lead * s.count * s.bytes_per_record for name, s in streams.items()}
    sizes["latent"] = per_lead * 3 * n * T1 * 8  # h mean, h sd, outlier probabilities
    budget = int(runtime.memory_budget_gb * _GIB)
    plan = StoragePlan(streams, lead, sizes, budget, runtime.storage_dir)
    if plan.total_bytes > budget and runtime.storage_dir is None:
        detail = ", ".join(f"{k} {v / _GIB:.3f} GiB" for k, v in sizes.items())
        raise StorageBudgetError(
            f"stored draws need {plan.total_bytes / _GIB:.3f} GiB ({detail}), above "
            f"Runtime.memory_budget_gb={runtime.memory_budget_gb}; lower MCMCSettings.store_max "
            f"(now {settings.store_max}) or draws, raise thin, set Runtime.storage_dir to store "
            "on disk (memmap), or raise memory_budget_gb"
        )
    return plan


# --------------------------------------------------------------------------- host store


def _empty_arrays(
    spec: StreamSpec, lead: tuple[int, ...], directory: str | None
) -> list[np.ndarray]:
    out = []
    flat = jax.tree_util.tree_flatten_with_path(spec.record)[0]
    for k, (path, leaf) in enumerate(flat):
        shape = (*lead, spec.count, *leaf.shape)
        dtype = np.dtype(leaf.dtype)
        if directory is None:
            out.append(np.zeros(shape, dtype))
        else:
            name = (leaf_path(path) or "leaf").replace("/", "_")
            fname = os.path.join(directory, f"{spec.name}.{k:03d}.{name}.dat")
            if math.prod(shape) == 0:
                out.append(np.zeros(shape, dtype))
            else:
                out.append(np.memmap(fname, dtype=dtype, mode="w+", shape=shape))
    return out


@dataclass
class HostStore:
    """Stored draws on the host: one array ``(*lead, D, …)`` per record leaf and stream.

    ``append(out, stop)`` takes one ``sample`` output (``out = (monitor, params, paths[,
    forecast])`` with host leaves ``(*lead, chunk, …)``) and keeps the records ``0..stop`` that
    the stream's thinning (or, for ``forecast``, its explicit kept-draw index) selects
    (:meth:`StreamSpec.select`); the kept-draw counter advances by ``stop + 1``.
    :meth:`finalize` attaches the once-per-run quantities, :meth:`to_datatree` builds
    ``fit.posterior``.
    """

    plan: StoragePlan
    directory: str | None = None
    arrays: dict[str, list[np.ndarray]] = field(default_factory=dict)
    kept: int = 0
    acceptance: dict[str, np.ndarray] = field(default_factory=dict)
    latent: dict[str, np.ndarray] = field(default_factory=dict)
    min_pivot: np.ndarray | None = None
    constant: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    T_eff: Any = None
    static: StaticConfig | None = None

    @classmethod
    def allocate(cls, plan: StoragePlan, static: StaticConfig | None = None) -> HostStore:
        """Preallocate every stream (memmap files in a fresh directory under ``storage_dir``)."""
        directory = None
        if plan.storage_dir is not None:
            os.makedirs(plan.storage_dir, exist_ok=True)
            directory = tempfile.mkdtemp(prefix="rai_bvar_draws_", dir=plan.storage_dir)
        store = cls(plan=plan, directory=directory, static=static)
        for name, spec in plan.streams.items():
            store.arrays[name] = _empty_arrays(spec, plan.lead, directory)
        return store

    @property
    def memmap(self) -> bool:
        return self.directory is not None

    def append(self, out: Sequence[Any], stop: int) -> None:
        """Store the kept draws ``0..stop`` of one chunk output (see the class docstring)."""
        stop = int(stop)
        axis = len(self.plan.lead)
        d0 = self.kept
        for name, spec in self.plan.streams.items():
            leaves = jax.tree.leaves(out[STREAMS.index(name)])
            sel, pos = spec.select(d0, stop)
            if sel.size == 0:
                continue
            for dst, src in zip(self.arrays[name], leaves, strict=True):
                src = np.asarray(src)
                idx_dst = (slice(None),) * axis + (pos,)
                idx_src = (slice(None),) * axis + (sel,)
                dst[idx_dst] = src[idx_src]
        self.kept = d0 + stop + 1
        if self.memmap:
            for arrs in self.arrays.values():
                for a in arrs:
                    if isinstance(a, np.memmap):
                        a.flush()

    def stream(self, name: str) -> Any:
        """The stored stream as its record pytree with leaves ``(*lead, D, …)``."""
        spec = self.plan.streams[name]
        treedef = jax.tree.structure(spec.record)
        return jax.tree.unflatten(treedef, self.arrays[name])

    def monitor(self) -> dict[str, np.ndarray]:
        """The monitor stream as ``{name: (*lead, D, …)}``."""
        mon = self.stream("monitor")
        if not isinstance(mon, Mapping):
            return {leaf_path(p): a for p, a in jax.tree_util.tree_flatten_with_path(mon)[0]}
        return dict(mon)

    def finalize(
        self,
        *,
        acceptance: Mapping[str, np.ndarray],
        latent: Mapping[str, np.ndarray],
        min_pivot: np.ndarray,
        T_eff: Any,
        constant: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach the once-per-run quantities (accumulator rates, latent moments, metadata)."""
        self.acceptance = dict(acceptance)
        self.latent = dict(latent)
        self.min_pivot = np.asarray(min_pivot)
        self.T_eff = T_eff
        self.constant = dict(constant or {})
        self.attrs.update(dict(attrs or {}))

    # ------------------------------------------------------------------ DataTree
    def to_datatree(self, meta: FitMeta) -> xr.DataTree:
        """``fit.posterior`` (DESIGN §5.3): groups ``posterior``, ``monitor``, ``paths`` (if
        stored), ``sample_stats``, ``latent_moments``, ``constant_data``; dims ``(chain, draw, …)``
        (``(fit, chain, draw, …)`` with a fit axis); run metadata in the root attrs."""
        names = _Names.of(self, meta)
        groups: dict[str, xr.Dataset] = {}
        groups["posterior"] = self._posterior(names)
        groups["monitor"] = self._monitor(names)
        if "paths" in self.plan.streams:
            groups["paths"] = self._paths(names)
        groups["sample_stats"] = self._sample_stats(names)
        groups["latent_moments"] = self._latent(names)
        groups["constant_data"] = self._constant(names, meta)
        root = xr.Dataset(attrs=_clean_attrs({**self.attrs, **_meta_attrs(meta)}))
        return xr.DataTree.from_dict({"/": root, **groups})

    def _lead_coords(self, names: _Names) -> dict[str, Any]:
        coords: dict[str, Any] = {
            d: np.arange(s) for d, s in zip(names.lead_dims, self.plan.lead, strict=True)
        }
        return coords

    def _posterior(self, names: _Names) -> xr.Dataset:
        spec = self.plan.streams["params"]
        draw = spec.draw_index()
        lead = names.lead_dims
        data_vars: dict[str, xr.DataArray] = {}
        for path, arr in zip(_paths(spec.record), self.arrays["params"], strict=True):
            key = path.split(".")[-1]
            base = (*lead, "draw")
            if path == "struct.a":
                rows, cols = np.tril_indices(names.n, -1)
                if rows.size:
                    data_vars["a"] = xr.DataArray(
                        arr[..., rows, cols], dims=(*base, "pair"),
                        coords={"pair": list(names.pairs), "pair_i": ("pair", rows), "pair_j": ("pair", cols)},
                    )  # fmt: skip
                continue
            if path == "glob.coef.incl":  # SSVS indicators γ, int8 (P3 E123 as amended)
                data_vars["ssvs_gamma"] = xr.DataArray(arr, dims=(*base, "variable", "regressor"))
                continue
            if key == "log1m_p":
                continue
            if key == "log_p":
                data_vars["p_outlier"] = xr.DataArray(np.exp(arr), dims=(*base, "variable"))
                continue
            name = key if key in _POSTERIOR_NAMES else path.replace(".", "_")
            equation = path.split(".")[0] in _EQUATION_PARTS  # glob.* leaves are not per equation
            data_vars[name] = xr.DataArray(
                arr, dims=(*base, *names.dims(name, arr.shape[len(base) :], equation=equation))
            )
        coords = {**self._lead_coords(names), "draw": draw, **names.coords}
        return _dataset(data_vars, coords)

    def _monitor(self, names: _Names) -> xr.Dataset:
        spec = self.plan.streams["monitor"]
        lead = names.lead_dims
        data_vars = {}
        for path, arr in zip(_paths(spec.record), self.arrays["monitor"], strict=True):
            base = (*lead, "draw")
            data_vars[path] = xr.DataArray(
                arr, dims=(*base, *names.dims(path, arr.shape[len(base) :]))
            )
        coords = {**self._lead_coords(names), "draw": spec.draw_index(), **names.coords}
        return _dataset(data_vars, coords)

    def _paths(self, names: _Names) -> xr.Dataset:
        spec = self.plan.streams["paths"]
        lead = names.lead_dims
        data_vars = {}
        for path, arr in zip(_paths(spec.record), self.arrays["paths"], strict=True):
            base = (*lead, "draw")
            if path == "h":
                arr = arr[..., : names.T1] if names.T1 else arr
                data_vars["h"] = xr.DataArray(arr, dims=(*base, "variable", "h_index"))
            elif path == "log_scale2":
                arr = arr[..., : names.T] if names.T else arr
                data_vars["log_scale2"] = xr.DataArray(arr, dims=(*base, "variable", "time"))
            else:
                data_vars[path] = xr.DataArray(
                    arr, dims=(*base, *names.dims(path, arr.shape[len(base) :]))
                )
        coords = {
            **self._lead_coords(names),
            "draw": spec.draw_index(),
            **names.coords,
            **names.time_coords,
        }
        return _dataset(data_vars, coords)

    def _sample_stats(self, names: _Names) -> xr.Dataset:
        lead = names.lead_dims
        data_vars: dict[str, xr.DataArray] = {}
        for key, arr in self.acceptance.items():
            arr = np.asarray(arr)
            data_vars[key] = xr.DataArray(
                arr, dims=(*lead, *names.dims(key, arr.shape[len(lead) :]))
            )
        if self.min_pivot is not None:
            data_vars["min_pivot"] = xr.DataArray(self.min_pivot, dims=lead)
        mon = self.plan.streams["monitor"]
        for path, arr in zip(_paths(mon.record), self.arrays["monitor"], strict=True):
            if path in ("loglik", "log_lik", "log_likelihood"):
                data_vars["loglik"] = xr.DataArray(arr, dims=(*lead, "draw"))
        coords = {**self._lead_coords(names), "draw": mon.draw_index(), **names.coords}
        return _dataset(data_vars, coords)

    def _latent(self, names: _Names) -> xr.Dataset:
        lead = names.lead_dims
        data_vars = {}
        for key, arr in self.latent.items():
            arr = np.asarray(arr)
            if key in ("h_mean", "h_sd"):
                arr = arr[..., : names.T1] if names.T1 else arr
                data_vars[key] = xr.DataArray(arr, dims=(*lead, "variable", "h_index"))
            elif key == "p_outlier":
                arr = arr[..., : names.T] if names.T else arr
                data_vars[key] = xr.DataArray(arr, dims=(*lead, "variable", "time"))
            else:
                data_vars[key] = xr.DataArray(
                    arr, dims=(*lead, *names.dims(key, arr.shape[len(lead) :]))
                )
        coords = {**self._lead_coords(names), **names.coords, **names.time_coords}
        return _dataset(data_vars, coords)

    def _constant(self, names: _Names, meta: FitMeta) -> xr.Dataset:
        data_vars = {}
        attrs = {}
        for key, value in self.constant.items():
            arr = np.asarray(value)
            if arr.dtype == object:
                continue
            equation = (
                True if key.startswith("prior_eq_") else False if key.startswith("prior_") else None
            )  # prior.eq leaves lead with the equation; prior.shared leaves do not
            dims = names.dims(key, arr.shape, equation=equation) if arr.ndim else ()
            data_vars[key] = xr.DataArray(arr, dims=dims)
        for key, value in (meta.hyper or {}).items():
            if (
                isinstance(value, str | bool | int | float | np.integer | np.floating)
                or value is None
            ):
                attrs[f"hyper_{key}"] = value
                continue
            arr = np.asarray(value)
            if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
                attrs[f"hyper_{key}"] = str(value)
                continue
            dims = names.dims(f"hyper_{key}", arr.shape) if arr.ndim else ()
            data_vars[f"hyper_{key}"] = xr.DataArray(arr, dims=dims)
        ds = _dataset(data_vars, names.coords)
        ds.attrs.update(_clean_attrs(attrs))
        return ds


_POSTERIOR_NAMES = frozenset({"gamma", "Pi", "mu", "phi", "sigma", "h_T", "log_lam", "nu"})


def _paths(record: Any) -> list[str]:
    return [leaf_path(p) for p, _ in jax.tree_util.tree_flatten_with_path(record)[0]]


def _dataset(data_vars: Mapping[str, xr.DataArray], coords: Mapping[str, Any]) -> xr.Dataset:
    used = set()
    for da in data_vars.values():
        used |= set(da.dims)
    keep = {}
    for k, v in coords.items():
        dim = v[0] if isinstance(v, tuple) else k
        if dim in used:
            keep[k] = v
    return xr.Dataset(dict(data_vars), coords=keep)


def regressor_names(variables: Sequence[str], lags: int, n_det: int) -> list[str]:
    """Names of ``x_t = (d_t', y_{t−1}', …, y_{t−p}')'`` (``data/design.py`` layout)."""
    det = ["const", "trend"][:n_det]
    return det + [f"{v}.L{lag}" for lag in range(1, lags + 1) for v in variables]


@dataclass(frozen=True)
class _Names:
    """Dimension names and coordinates of one fit (variables, regressors, periods)."""

    lead_dims: tuple[str, ...]
    n: int
    k: int
    variables: tuple[str, ...]
    regressors: tuple[str, ...]
    pairs: tuple[str, ...]
    T: int | None  # real rows (None with a fit axis: ragged windows are not trimmed)
    periods: tuple[str, ...] | None
    nu: tuple[int, ...] = ()  # the static ν grid (coordinate of the ν prior weights)
    G: int = 0  # coefficient groups of a global-local prior (P3 D-38; 0 for a fixed prior)

    @classmethod
    def of(cls, store: HostStore, meta: FitMeta) -> _Names:
        n = int(store.plan.streams["params"].record.struct.a.shape[0])
        variables = tuple(meta.variables) if meta.variables else tuple(f"y{i}" for i in range(n))
        if len(variables) != n:
            raise ValueError(f"FitMeta.variables has {len(variables)} names for n = {n}")
        static = store.static
        k = static.shapes.k if static is not None else 0
        regs = (
            tuple(regressor_names(variables, static.shapes.lags, static.shapes.n_det))
            if static is not None
            else tuple(f"x{j}" for j in range(k))
        )
        rows, cols = np.tril_indices(n, -1)
        pairs = tuple(f"{variables[i]}~{variables[j]}" for i, j in zip(rows, cols, strict=True))
        lead_dims = ("chain",) if len(store.plan.lead) == 1 else ("fit", "chain")
        T = None
        periods = None
        if len(store.plan.lead) == 1 and store.T_eff is not None:
            T = int(np.asarray(store.T_eff))
            if meta.periods is not None and len(meta.periods) == T:
                periods = tuple(str(p) for p in meta.periods)
            else:
                periods = tuple(str(t) for t in range(T))
        nu = tuple(static.vol.nu_grid) if static is not None else ()
        G = static.coef.n_groups if static is not None else 0
        return cls(lead_dims, n, k, variables, regs, pairs, T, periods, nu, G)

    @property
    def T1(self) -> int | None:
        return None if self.T is None else self.T + 1

    @property
    def coords(self) -> dict[str, Any]:
        out: dict[str, Any] = {"variable": list(self.variables)}
        if self.n > 1:
            out["variable_"] = list(self.variables[: self.n - 1])
            out["pair"] = list(self.pairs)
        if self.k:
            out["regressor"] = list(self.regressors)
        if self.nu:
            out["nu"] = list(self.nu)
        if self.G:
            out["group"] = list(range(self.G))
        return out

    @property
    def time_coords(self) -> dict[str, Any]:
        if self.periods is None:
            return {}
        return {"time": list(self.periods), "h_index": ["h_0", *self.periods]}

    def _sizes(self) -> dict[str, int]:
        return {
            "variable": self.n,
            "variable_": self.n - 1,
            "regressor": self.k,
            "nu": len(self.nu),
            "group": self.G,
        }

    def dims(
        self, name: str, shape: tuple[int, ...], *, equation: bool | None = None
    ) -> tuple[str, ...]:
        """Dimension names of a record leaf's trailing ``shape``.

        Leaves with a known layout are named by their name (:data:`_NAMED_DIMS`: ``gamma`` and the
        δ prior arrays ``(variable, regressor)``, ``Pi`` ``(regressor, variable)``, the ν prior
        weights ``(variable, nu)``, …), never by their shape alone, so ``k = n``
        (``deterministic="n"``, one lag) or ``k = |ν grid|`` cannot mislabel an axis.

        ``equation`` says whether the leaf leads with the equation axis: ``True`` (per-equation
        leaves: ``prior.eq``, the ``mean``/``vol``/``struct`` parts of a draw), ``False``
        (``prior.shared``, ``glob``), ``None`` (decided by the name: the per-equation names of
        :data:`_EQUATION_NAMES`, else unknown). Only per-equation leaves have their trailing axes
        named by shape, when exactly one per-equation layout fits (:data:`_EQUATION_DIMS`).
        Every other axis gets a generic ``<name>_dim<j>`` name, except that an unknown leaf whose
        leading axis has ``n`` entries keeps the ``variable`` label for it."""
        if shape == ():
            return ()
        sizes = self._sizes()
        last = name.split(".")[-1]
        if last == "a" and shape == (len(self.pairs),) and self.pairs:
            return ("pair",)  # the monitor's strict lower triangle of a, row-major
        named = _NAMED_DIMS.get(name.replace(".", "_"), _NAMED_DIMS.get(last))
        if named is not None and self._fits(named, shape, sizes):
            return named
        if equation is None and last in _EQUATION_NAMES:
            equation = True
        if equation:
            fits = [d for d in _EQUATION_DIMS if self._fits(d, shape, sizes)]
            if len(fits) == 1:
                return fits[0]
        base = name.replace(".", "_")
        dims = [f"{base}_dim{j}" for j in range(len(shape))]
        if equation is not False and shape[0] == self.n and self.n > 0:
            dims[0] = "variable"
        return tuple(dims)

    def _fits(
        self, dims: tuple[str, ...], shape: tuple[int, ...], sizes: Mapping[str, int]
    ) -> bool:
        if len(dims) != len(shape) or any(sizes[d] != s for d, s in zip(dims, shape, strict=True)):
            return False
        # a zero-length axis has no coordinate to label (n = 1: no variable_; k = 0: no regressor)
        return all(sizes[d] > 0 for d in dims)


_NAMED_DIMS: dict[str, tuple[str, ...]] = {
    "gamma": ("variable", "regressor"),
    "Pi": ("regressor", "variable"),
    "prior_eq_coef_mean0": ("variable", "regressor"),
    "prior_eq_coef_prec0": ("variable", "regressor"),
    "prior_eq_coef_group": ("variable", "regressor"),
    "prior_eq_contemp_mean0": ("variable", "variable_"),
    "prior_eq_contemp_prec0": ("variable", "variable_"),
    "prior_eq_contemp_cmask": ("variable", "variable_"),
    "prior_shared_Mc": ("regressor", "variable_"),
    "prior_eq_transitory_log_nu_w": ("variable", "nu"),
    # Phase 3 global-local and SSVS priors (P3 §2.3, §3.7, D-52)
    "prior_eq_coef_gl_mask": ("variable", "regressor"),
    "prior_eq_coef_gl_group": ("variable", "regressor"),
    "prior_eq_coef_gl_log_r": ("variable", "regressor"),
    "prior_eq_coef_gl_ssvs_log_tau0": ("variable", "regressor"),
    "prior_eq_coef_gl_ssvs_log_tau1": ("variable", "regressor"),
    "gl_log_zeta": ("group",),
    "gl_log_tau": ("group",),
    "gl_log_xi": ("group",),
    "gl_a_idx": ("group",),
    "gl_p": ("group",),
    "incl_prob": ("variable", "regressor"),
}
"""Layouts of the leaves whose shape alone can be ambiguous (``k = n``, ``k = |ν grid|``); keys
are a leaf's full underscored path or its last path component."""

_EQUATION_DIMS: tuple[tuple[str, ...], ...] = (
    ("variable",),
    ("variable", "variable_"),
    ("variable", "regressor"),
    ("variable", "nu"),
)
"""Layouts tried by shape for per-equation leaves without a named layout (used only when exactly
one fits, so a trailing axis of ``k = n − 1`` or ``k = |ν grid|`` entries stays generic)."""

_EQUATION_PARTS = frozenset({"mean", "struct", "vol"})
"""Top-level parts of a ``ParamDraw`` / ``PathDraw`` whose leaves lead with the equation."""

_EQUATION_NAMES = frozenset({
    "mu", "phi", "sigma", "h_T", "log_lam", "nu", "log_p", "log1m_p", "p_outlier", "B_own1",
    "h_1", "h_mid", "intercept", "outlier_share", "outlier_at_max",
})  # fmt: skip
"""Per-equation stored, monitored and latent quantities (DESIGN §5.3, §8.1)."""


def _meta_attrs(meta: FitMeta) -> dict[str, Any]:
    out: dict[str, Any] = {"variables": list(meta.variables)}
    if meta.notes:
        out["notes"] = list(meta.notes)
    if meta.eb_converged is not None:
        out["eb_converged"] = bool(meta.eb_converged)
    if meta.periods is not None and len(meta.periods):
        out["first_period"] = str(meta.periods[0])
        out["last_period"] = str(meta.periods[-1])
    return out


def _clean_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Attrs that serialise to NetCDF: scalars, strings, lists of those; others as strings."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool | np.bool_):
            out[key] = int(value)
        elif isinstance(value, str | int | float | np.integer | np.floating):
            out[key] = value
        elif value is None:
            out[key] = "None"
        elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
            out[key] = list(value)
        elif isinstance(value, list | tuple) and all(
            isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool)
            for v in value
        ):
            out[key] = list(value)
        elif isinstance(value, pd.Period):
            out[key] = str(value)
        else:
            out[key] = str(value)
    return out
