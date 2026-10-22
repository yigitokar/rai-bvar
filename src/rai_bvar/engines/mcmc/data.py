"""Estimation-window data and the per-fit design cache of the MCMC engine (DESIGN §2.3, §2.5).

Two builders of :class:`EngineData` produce the same arrays (bitwise):

* :func:`engine_data_from_vardata` (host, NumPy leaves) cuts one window from a ``VARData`` with the
  Phase 1 ``data/design.py`` layout and right-pads it to ``T_pad = bucket(T_eff, t_bucket)``;
* :func:`build_engine_data` (``jnp``, jit-safe with a traced ``T_eff``) rebuilds the same window
  from targets and a presample; the validation simulators call it inside JIT (DESIGN §9.2).

Layout (``data/design.py``, SPEC §4.1): ``x_t = (d_t', y_{t−1}', …, y_{t−p}')'`` with ``d_t = (1,
τ_t)[:n_det]`` and the trend ``τ_t = t0 + t`` (data row ``t = 0..T_eff−1``; ``t0 = p + 1`` is the
1-based sample position of the first target). Rows ``t ≥ T_eff`` of ``Y`` and ``X`` are exactly 0 and
``m_t = tmask_t = 1{t < T_eff}`` (D-15). ``hist`` holds the last ``p`` observations and ``t_last =
t0 + T_eff − 1`` continues the trend into the forecast (``x_{T+h}`` uses ``τ = t_last + h``).

Per-fit design cache (:func:`base_cache`; DESIGN §2.3, §2.5, D-25, D-26; N05 §4.1, §7.1):

    F       = [X | X M_c − Y_{:, :n−1}]          (T_pad × q) δ-coordinate design, one for all equations
    FF[t,c] = F[t, iu_c]·F[t, ju_c]              (T_pad × P) packed Khatri–Rao products, G = (w @ FF)[pos]
                                                 (weighted path only: its one consumer, §3.7.2)
    XX      = the same for X                     (T_pad × P_x) reduced form (CCCM)
    G_const = Fᵀ D_m F,   FtY = Fᵀ D_m Y          Constant-volatility paths (N07 §2.2)

The packed order is column-major (pairs ``(r, s)``, ``r ≤ s``, sorted by ``s`` then ``r``; pilot
09 §10): the leading ``q'(q'+1)/2`` columns cover the leading ``q' × q'`` block, so ``FF[:, :P_x]``
equals ``XX`` (``F[:, :k] = X``) and a second equation bucket could read a prefix of ``FF``.
Packed-index maps are NumPy constants built at trace time from a static ``q``, never module-level
``jnp`` arrays (DESIGN §1.1 rule 3).
"""

from __future__ import annotations

import dataclasses
import functools
import numbers
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rai_bvar.data.design import lagged_design, n_deterministic
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig, bucket
from rai_bvar.engines.mcmc.state import EngineData, FitCache

if TYPE_CHECKING:
    import pandas as pd

    from rai_bvar.data.dataset import VARData

Array = jax.Array

_HIGHEST = lax.Precision.HIGHEST  # full float64 products on every backend (CPU ignores it)


# --------------------------------------------------------------------------- packed products


@functools.lru_cache(maxsize=64)
def _packed_index(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = q * (q + 1) // 2
    ju = np.repeat(np.arange(q, dtype=np.int32), np.arange(1, q + 1))
    iu = (np.arange(P, dtype=np.int64) - ju.astype(np.int64) * (ju + 1) // 2).astype(np.int32)
    pos = np.empty((q, q), dtype=np.int32)
    pos[iu, ju] = np.arange(P, dtype=np.int32)
    pos[ju, iu] = np.arange(P, dtype=np.int32)
    for a in (iu, ju, pos):
        a.setflags(write=False)  # cached: shared by every caller
    return iu, ju, pos


def packed_index(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-major upper-triangle maps ``(iu, ju, pos)`` (int32, read-only, cached).

    Packed column ``c = s(s+1)/2 + r`` holds the pair ``(r, s) = (iu_c, ju_c)``, ``r ≤ s``; ``pos[r, s]
    = pos[s, r] = c``, so a packed vector ``v`` unpacks to the symmetric ``v[pos]`` (q × q).
    """
    if isinstance(q, bool) or not isinstance(q, numbers.Integral) or q < 1:
        raise ValueError(f"q must be a positive integer, got {q!r}")
    return _packed_index(int(q))


def packed_products(F: Array) -> Array:
    """Packed Khatri–Rao products ``FF[..., t, c] = F[..., t, iu_c]·F[..., t, ju_c]``, shape ``(..., T, P)``.

    ``Fᵀ diag(w) F = (w @ FF)[pos]`` for any weights ``w`` (D-25); zero rows of ``F`` stay zero.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    if F.ndim < 2:
        raise ValueError(f"F must have shape (..., T, q), got {F.shape}")
    iu, ju, _ = packed_index(F.shape[-1])
    return F[..., iu] * F[..., ju]


# --------------------------------------------------------------------------- regressors


def _check_n_det(n_det: int) -> int:
    if isinstance(n_det, bool) or not isinstance(n_det, numbers.Integral) or n_det not in (0, 1, 2):
        raise ValueError(
            f"n_det must be 0 (none), 1 (constant) or 2 (constant, trend), got {n_det!r}"
        )
    return int(n_det)


def _deterministic(t: Array, n_det: int) -> Array:
    """``d = (1, τ)[:n_det]`` for trend values ``t``; shape ``(*t.shape, n_det)``."""
    cols = [jnp.ones_like(t), t][:n_det]
    return jnp.stack(cols, axis=-1) if cols else jnp.zeros((*t.shape, 0), jnp.float64)


def regressors_from_window(window: Array, t_next: Array, n_det: int) -> Array:
    """Regressors ``x = (d', y_{−1}', …, y_{−p}')'`` for the period after a lag window.

    ``jnp`` port of ``data.design.regressors_from_history``: ``window`` has shape ``(..., p, n)``
    (most recent observation last), ``t_next`` is the trend value of the next period (broadcast
    over the leading axes); the result has shape ``(..., n_det + n·p)``. Forecasting uses
    ``t_next = t_last + h`` (DESIGN §7.2).
    """
    n_det = _check_n_det(n_det)
    window = jnp.asarray(window, dtype=jnp.float64)
    if window.ndim < 2:
        raise ValueError(f"window must have shape (..., p, n), got {window.shape}")
    lead = window.shape[:-2]
    det = jnp.broadcast_to(_deterministic(jnp.asarray(t_next, jnp.float64), n_det), (*lead, n_det))
    lagged = window[..., ::-1, :].reshape(*lead, -1)  # lag 1 block first
    return jnp.concatenate([det, lagged], axis=-1)


# --------------------------------------------------------------------------- EngineData builders


def _concrete_int(x: object) -> int | None:
    """``int(x)`` when ``x`` is known on the host, ``None`` for a tracer."""
    if isinstance(x, jax.core.Tracer):
        return None
    return int(np.asarray(x))


def build_engine_data(
    y: Array, presample: Array, T_eff: Array, *, n_det: int, t0: float | Array
) -> EngineData:
    """One estimation window from targets ``y`` (``T_pad × n``) and the presample ``y_{1−p:0}`` (``p × n``).

    jit-safe: ``T_eff`` (integer scalar, ``1 ≤ T_eff ≤ T_pad``) and ``t0`` may be traced, so every
    ``T_eff`` of a bucket shares one compiled program; ``n_det`` is static. Rows ``t ≥ T_eff`` of
    ``y`` are ignored (they may hold anything, NaN included) and come out exactly 0::

        Y_t = 1{t<T_eff} y_t;   x_t = 1{t<T_eff} (d(t0 + t)', y_{t−1}', …, y_{t−p}')'
        hist = (y_{T_eff−p}, …, y_{T_eff−1});   t_last = t0 + T_eff − 1

    where ``y_{t}`` for ``t < 0`` is read from ``presample`` (most recent last). With ``t0 = p + 1``
    this reproduces ``data.design.lagged_design`` on the stacked series.
    """
    n_det = _check_n_det(n_det)
    y = jnp.asarray(y, dtype=jnp.float64)
    presample = jnp.asarray(presample, dtype=jnp.float64)
    if y.ndim != 2 or presample.ndim != 2 or presample.shape[1] != y.shape[1]:
        raise ValueError(
            f"need y (T_pad, n) and presample (p, n), got {y.shape} and {presample.shape}"
        )
    T_pad, _ = y.shape
    p = presample.shape[0]
    if p < 1:
        raise ValueError("presample must hold p >= 1 observations")
    if not jnp.issubdtype(jnp.result_type(T_eff), jnp.integer) or jnp.ndim(T_eff) != 0:
        raise TypeError(f"T_eff must be an integer scalar, got {T_eff!r}")
    if jnp.ndim(t0) != 0:
        raise ValueError("t0 must be a scalar")
    concrete = _concrete_int(T_eff)
    if concrete is not None and not 1 <= concrete <= T_pad:
        raise ValueError(f"T_eff must lie in [1, T_pad={T_pad}], got {concrete}")
    T_eff = jnp.asarray(T_eff).astype(jnp.int32)
    t0 = jnp.asarray(t0, dtype=jnp.float64)

    live = jnp.arange(T_pad, dtype=jnp.int32) < T_eff
    Y = jnp.where(live[:, None], y, 0.0)
    full = jnp.concatenate([presample, Y], axis=0)  # row p + t holds y_t
    lags = [full[p - lag : p - lag + T_pad] for lag in range(1, p + 1)]
    det = _deterministic(t0 + jnp.arange(T_pad, dtype=jnp.float64), n_det)
    X = jnp.where(live[:, None], jnp.concatenate([det, *lags], axis=1), 0.0)
    return EngineData(
        Y=Y,
        X=X,
        tmask=live.astype(jnp.float64),
        T_eff=T_eff,
        hist=lax.dynamic_slice_in_dim(full, T_eff, p, axis=0),
        t_last=t0 + (T_eff - 1).astype(jnp.float64),
    )


def engine_data_from_vardata(
    data: VARData, lags: int, deterministic: str, t_bucket: int
) -> tuple[EngineData, pd.PeriodIndex]:
    """Host builder: the window of ``data`` as :class:`EngineData` with NumPy leaves.

    ``Y, X = lagged_design(data.values, lags, deterministic)`` (Phase 1 layout and trend: ``t0 =
    lags + 1``, ``t_last = len(data.values)``), right-padded with zero rows to ``T_pad =
    bucket(T_eff, t_bucket)``. Returns the data and the periods of the ``T_eff`` target rows
    (row ``t`` ↔ ``index[t]``; SV state ``t + 1`` belongs to the same period).
    """
    if isinstance(lags, bool) or not isinstance(lags, numbers.Integral):
        raise TypeError(f"lags must be an integer, got {lags!r}")
    lags = int(lags)
    values = data.values
    Y, X = lagged_design(values, lags, deterministic)  # validates lags, length and deterministic
    T_eff, n = Y.shape
    T_pad = bucket(T_eff, t_bucket)
    Y_pad = np.zeros((T_pad, n))
    X_pad = np.zeros((T_pad, n_deterministic(deterministic) + n * lags))
    Y_pad[:T_eff], X_pad[:T_eff] = Y, X
    tmask = np.zeros(T_pad)
    tmask[:T_eff] = 1.0
    engine_data = EngineData(
        Y=Y_pad,
        X=X_pad,
        tmask=tmask,
        T_eff=np.asarray(T_eff, dtype=np.int32),
        hist=np.ascontiguousarray(values[-lags:]),
        t_last=np.asarray(len(values), dtype=np.float64),
    )
    return engine_data, data.index[lags:]


# --------------------------------------------------------------------------- per-fit design cache


def structural_design(data: EngineData, Mc: Array) -> Array:
    """δ-coordinate design ``F = [X | X M_c − Y_{:, :n−1}]`` (``T_pad × q``; DESIGN §2.3, N05 §4.1).

    Slot ``k + j`` is ``x_t'm_j − y_{j,t}`` (minus the prior-mean residual of variable ``j``), which
    does not depend on the equation, so one ``F`` serves all equations: ``Fθ_i = Xγ_i − Y_{:, :n−1}a_i``
    with ``γ_i = δ_i + M_c a_i`` (D3). Padded rows are exactly 0.
    """
    X = jnp.asarray(data.X, dtype=jnp.float64)
    Y = jnp.asarray(data.Y, dtype=jnp.float64)
    Mc = jnp.asarray(Mc, dtype=jnp.float64)
    k, n = X.shape[1], Y.shape[1]
    if Mc.shape != (k, n - 1):
        raise ValueError(f"Mc must have shape (k, n−1) = {(k, n - 1)}, got {Mc.shape}")
    live = jnp.asarray(data.tmask, dtype=jnp.float64)[:, None] > 0
    cols = jnp.where(live, jnp.matmul(X, Mc, precision=_HIGHEST) - Y[:, : n - 1], 0.0)
    return jnp.concatenate([X, cols], axis=1)


def _check_shapes(data: EngineData, static: StaticConfig) -> None:
    s = static.shapes
    expected = {
        "Y": (s.T_pad, s.n),
        "X": (s.T_pad, s.k),
        "tmask": (s.T_pad,),
        "T_eff": (),
        "hist": (s.lags, s.n),
        "t_last": (),
    }
    bad = [
        f"{name} {tuple(jnp.shape(getattr(data, name)))} (expected {shape})"
        for name, shape in expected.items()
        if tuple(jnp.shape(getattr(data, name))) != shape
    ]
    if bad:
        raise ValueError("EngineData does not match StaticConfig.shapes: " + ", ".join(bad))


def _gram_masked(F: Array, tmask: Array) -> Array:
    """``Fᵀ D_m F``, made exactly symmetric by mirroring its upper triangle."""
    G = jnp.matmul(F.T, F * tmask[:, None], precision=_HIGHEST)
    upper = np.triu(np.ones(G.shape, dtype=bool))  # NumPy constant (static shape)
    return jnp.where(upper, G, G.T)


def base_cache(data: EngineData, Mc: Array | None, static: StaticConfig) -> FitCache:
    """The mean-independent part of :class:`FitCache` (DESIGN §2.5); ``eq`` stays ``Empty()``.

    Which leaves exist is a function of ``static.mean`` alone (the tree structure is static):

    =========================  =================================================================
    ``F``                      ``linear_structural`` (needs ``Mc``; :func:`structural_design`)
    ``FF``                     ``linear_structural``, ``gram="packed"``, ``const_path="weighted"``
    ``XX``                     ``linear_reduced`` and ``gram="packed"``: packed products of X
    ``G_const``, ``FtY``       ``const_path ≠ "weighted"``: ``Fᵀ D_m F`` (exactly symmetric), ``Fᵀ D_m Y``
    =========================  =================================================================

    ``FF`` has one consumer, the ``weighted`` path's ``G = weighted_gram(w, FF)`` (§3.7.2); the
    Constant fast paths (``eigen``, ``fixed_factor``, ``cholesky_const``) read only ``G_const`` and
    ``FtY``, so ``FF`` (``T_pad·P·8`` bytes, up to 256 MiB under the §10.5 rule) is not built
    there. A test comparing a fast path with the weighted oracle builds the oracle's cache with
    ``const_path="weighted"``. ``packed_tiled`` and ``batched`` Grams form their products from
    ``F`` (or ``X``) on the fly (WP4 ``weighted_gram``), so ``FF``/``XX`` are absent there too.
    jit-safe (``static`` is closed over); called by the runner's ``prepare`` before
    ``MeanBlock.prepare`` fills ``eq``.
    """
    _check_shapes(data, static)
    mean = static.mean
    cache = FitCache.empty()
    if mean.kind == "linear_structural":
        if Mc is None:
            raise ValueError(
                "the structural Linear mean needs the D3 centring columns Mc (k × n−1)"
            )
        F = structural_design(data, Mc)
        cache = dataclasses.replace(cache, F=F)
        if mean.gram == "packed" and mean.const_path == "weighted":
            cache = dataclasses.replace(cache, FF=packed_products(F))
        if mean.const_path != "weighted":
            tmask = jnp.asarray(data.tmask, dtype=jnp.float64)
            Y = jnp.asarray(data.Y, dtype=jnp.float64)
            cache = dataclasses.replace(
                cache,
                G_const=_gram_masked(F, tmask),
                FtY=jnp.matmul(F.T, Y * tmask[:, None], precision=_HIGHEST),
            )
    elif mean.kind == "linear_reduced" and mean.gram == "packed":
        cache = dataclasses.replace(cache, XX=packed_products(data.X))
    return cache
