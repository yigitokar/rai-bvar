"""Forecasting, impulse responses and parameter draws from an MCMC fit (DESIGN §7; N03 §5; N05 §8).

**Forecast core** (DESIGN §7.2), pure in its injected randomness and vmapped over draws in
chunks of ``S_c = min(1024, 2^⌈log₂ S⌉)`` for ``S`` paths (:func:`make_forecast`, compiled once
per ``(program, H, S_c)``; the last chunk is padded)::

    (log λ^pers_i, log λ^eff_i)_{T+1..T+H} = vol.draw_ahead_eq(vol_keys[i], vd_i, ·, H, gfut)
    fut_i = mean.future_eq(mean_keys[i], md_i, H);   gfut = global steps' futures (Empty in Phase 2)
    scan h = 1..H (carry: the lag window (p, n), most recent last):
        x = regressors_from_window(window, t_last + h, n_det)
        g = [mean.predict_eq(md_i, fut_i, x, h, gfut)]_i      structural: Γx;  reduced: 0
        f = mean.predict_reduced(gd, x, h, gfut)              reduced: Π'x;   structural: 0
        m = structure.mean_map(sd, g, f)                      A⁻¹g + f = B'x
        L = structure.chol(sd, log λ^eff_{T+h})               A⁻¹ diag(exp(½ log λ^eff))
        y = m + L ε_h;  window ← [window[1:], y]

so ``y_{T+h} | draw, path_{<T+h}, λ_{T+h} ~ N(m, LLᵀ)`` exactly and ``L`` is the Cholesky factor of
the conditional covariance (SPEC §7 invariant 1, conditional on the drawn transitory scales as
SPEC §6.1 allows; N03 §5.3, N05 §8). :func:`simulate_paths` is the Gaussian part alone, given
``ε`` and the future variances (DESIGN §7.5: GIRF with common random numbers, conditional
forecasts, the in-loop hook).

**Draws and keys** (DESIGN §7.1, §5.6). ``fit.forecast(H, draws, seed)`` takes ``m = ⌈draws/C⌉``
evenly spaced stored draws per chain (:func:`select_draws`, positions ``⌊j·D/m⌋``), keeping the
``(chain, draw)`` structure. When ``m`` exceeds the ``D`` stored draws, every stored draw gives
``R = ⌈draws/(C·D)⌉`` paths with independent shocks (each ``(chain, draw)`` slice is still one
path from one parameter draw). ``rng = default_rng(seed)``, ``label = seed_label(seed, rng)``,
``fseed = rng.integers(0, 2**32)``; the shocks of stored draw ``d`` (its kept-draw index, the
``draw`` coordinate of ``fit.posterior``) of chain ``c`` come from ``key(fseed) → FORECAST → c →
d → Ahead.{VOL, SHOCK, MEAN, GLOBAL} → i`` (``runner.forecast_shocks``), so in-loop and post-hoc
forecasts agree and recomputing the moments reproduces ``o_{T+h}``, ``τ_{T+h}`` and ``ε``
(N03 §5.3 requirement 1). Replicate ``r ≥ 1`` of draw ``d`` uses the virtual kept-draw index
``d + r·D_kept`` (``D_kept`` kept draws per chain), which no kept draw has. A fit with a matching
``FitPlan`` returns its in-loop forecasts (DESIGN §5.8).

**Accelerator variables** (DESIGN §7.3; never contract fields), from each slot's parameter draw
(replicate slots and in-loop forecasts included): ``cond_A``, ``cond_scale``, with a transitory
scale ``cond_scale_persistent`` and ``log_p``/``log1m_p``/``nu``, and (D-34) ``lag_coefs[c, d,
l−1, i, j] = A_l[i, j] = B[n_det + (l−1)·n + j, i]`` with ``B`` as below (linear structural,
CCCM and zero means; other mean kinds attach none) for ``log_pred_density(lpd_method=
"integrated")``. An in-loop forecast whose draws are not all stored carries none of them.
``lag_coefs`` is attached to every such forecast (the default estimator never reads it) and
costs ``S·p·n²·8`` bytes, ``(p/H)·nbytes(cond_chol)``, with each draw's ``A_l`` repeated for its
replicate slots; for ``p > H`` it is the largest variable of the dataset.

**Parameters and IRFs** (DESIGN §7.4; N03 §5.4). ``coefs = B = (A⁻¹Γ)'`` (``structure.
reduced_form``) or ``Π``; ``chol = structure.chol(a, log λ_date) = A⁻¹ diag(exp(½ log λ))``.
``date=None`` is the last in-sample period (``h_T`` of the stored ``VolDraw``); other dates, and
the effective variance under a transitory scale, need ``store="paths"``; a Constant core is
date-free. One-s.d. shocks use the **persistent** variance by default (a "one-s.d. shock in March
2020" excludes the outlier scale): ``e^{h}``, or ``e^{h} ν/(ν−2)`` under ``t_scale="unit"`` (the
shock variance net of outliers); unit shocks give ``Φ_h A⁻¹`` whatever the date.
"""

from __future__ import annotations

import functools
import numbers
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax, random

from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc.compose import Program, compose
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.data import regressors_from_window
from rai_bvar.engines.mcmc.state import CCCMDraw, Empty, LinearDraw, ParamDraw

Array = jax.Array

__all__ = [
    "S_CHUNK",
    "ForecastOut",
    "ForecastShocks",
    "draw_parameters",
    "forecast",
    "forecast_core",
    "irf",
    "make_forecast",
    "select_draws",
    "simulate_paths",
]

S_CHUNK = 1024  # forecast draws per compiled chunk (DESIGN §7.2)
_HIGHEST = lax.Precision.HIGHEST
_U32 = 2**32
_WHICH = ("persistent", "effective")
_LAG_MEANS = frozenset(("linear_structural", "linear_reduced", "zero"))  # kinds with lag_coefs
# fit metadata keys renamed fit_* in forecast attrs (``seed`` and ``draws`` describe the forecast)
_FIT_KEYS = frozenset(
    ("seed", "salt", "chains", "draws", "burn", "thin", "chunk", "warm_start", "it_first",
     "it_last")
)  # fmt: skip


class ForecastShocks(NamedTuple):
    """Injected randomness of one draw (DESIGN §7.2): ``eps`` f64[H, n] standard normals, one
    volatility and one mean key per equation, one key for global futures."""

    eps: Array
    vol_keys: Array
    mean_keys: Array
    glob_key: Array


class ForecastOut(NamedTuple):
    """One draw's forecast (DESIGN §7.2): ``paths``, ``cond_mean`` f64[H, n], ``cond_chol``
    f64[H, n, n] (lower, positive diagonal), ``cond_scale = exp(½ log λ^eff)`` f64[H, n] and
    ``log_lam_pers`` f64[H, n] (the persistent log variances, for the transitory RB)."""

    paths: Array
    cond_mean: Array
    cond_chol: Array
    cond_scale: Array
    log_lam_pers: Array


# --------------------------------------------------------------------------- pure core (§7.2)


def _with_index(eq_map: Callable, f: Callable, n: int) -> Callable:
    """``eq_map`` of ``f(i, *args)`` with the equation index as the first mapped argument, so a
    batch of argument trees without leaves (``Empty`` draws) still has a mapped axis."""

    def mapped(*args: Any) -> Any:
        return eq_map(f)(jnp.arange(n, dtype=jnp.int32), *args)

    return mapped


def _global_futures(program: Program, key: Array, gd: Any, H: int) -> tuple[Any, ...]:
    """Futures of the program's global steps (``GlobalStep.future``; ``Empty`` in Phase 2), each
    from ``fold_in(glob_key, block)``."""
    return tuple(g.future(random.fold_in(key, int(g.block)), gd, H) for g in program.global_steps)


def simulate_paths(
    params: ParamDraw,
    hist: Array,
    t_last: Array,
    eps: Array,
    log_lam_eff: Array,
    program: Program,
    *,
    fut: Any = None,
    gfut: Any = None,
) -> tuple[Array, Array, Array]:
    """``(paths, cond_mean, cond_chol)`` (f64[H, n], f64[H, n], f64[H, n, n]) of one draw given
    the shocks ``eps`` f64[H, n] and the future effective log variances ``log_lam_eff`` f64[H, n]
    (DESIGN §7.5): the scan of the module docstring. ``fut`` (per-equation mean futures, leading
    axis ``n``) and ``gfut`` default to the Phase 2 ``Empty`` futures."""
    s = program.static.shapes
    n = s.n
    structure, mean = program.structure, program.mean
    hist = jnp.asarray(hist, jnp.float64)
    t_last = jnp.asarray(t_last, jnp.float64)
    fut = Empty() if fut is None else fut
    gfut = () if gfut is None else gfut

    def step(window: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, tuple]:
        h, e, log_lam = inputs
        x = regressors_from_window(window, t_last + h.astype(jnp.float64), s.n_det)
        g = _with_index(program.eq_map, lambda i, md, ft: mean.predict_eq(md, ft, x, h, gfut), n)(
            params.mean, fut
        )
        f = mean.predict_reduced(params.glob, x, h, gfut)
        m = structure.mean_map(params.struct, g, f)
        L = structure.chol(params.struct, log_lam)
        y = m + jnp.matmul(L, e, precision=_HIGHEST)
        return jnp.concatenate([window[1:], y[None, :]], axis=0), (y, m, L)

    H = eps.shape[0]
    hs = jnp.arange(1, H + 1, dtype=jnp.int32)
    _, (paths, cond_mean, cond_chol) = lax.scan(
        step, hist, (hs, jnp.asarray(eps, jnp.float64), jnp.asarray(log_lam_eff, jnp.float64))
    )
    return paths, cond_mean, cond_chol


def forecast_core(
    draw: ParamDraw, hist: Array, t_last: Array, shocks: ForecastShocks, program: Program
) -> ForecastOut:
    """One draw's joint forecast with injected randomness (DESIGN §7.2, module docstring).

    ``draw`` is one stored ``ParamDraw`` (per-equation leaves with leading axis ``n``); ``hist``
    f64[p, n] the last ``p`` observations (most recent last); ``t_last`` the 1-based sample
    position of the last observation (the trend continues at ``t_last + h``). The volatility
    futures come from ``vol.draw_ahead_eq`` with the per-equation ``shocks.vol_keys`` (its prior
    argument is unused by the Phase 2 blocks and passed as ``None``)."""
    n = program.static.shapes.n
    H = int(shocks.eps.shape[0])
    vol, mean = program.vol, program.mean
    gfut = _global_futures(program, shocks.glob_key, draw.glob, H)
    pers, eff = _with_index(
        program.eq_map, lambda i, k, vd: vol.draw_ahead_eq(k, vd, None, H, gfut), n
    )(shocks.vol_keys, draw.vol)
    fut = _with_index(program.eq_map, lambda i, k, md: mean.future_eq(k, md, H), n)(
        shocks.mean_keys, draw.mean
    )
    log_lam_eff, log_lam_pers = eff.T, pers.T  # (H, n)
    paths, cond_mean, cond_chol = simulate_paths(
        draw, hist, t_last, shocks.eps, log_lam_eff, program, fut=fut, gfut=gfut
    )
    return ForecastOut(
        paths=paths,
        cond_mean=cond_mean,
        cond_chol=cond_chol,
        cond_scale=jnp.exp(0.5 * log_lam_eff),
        log_lam_pers=log_lam_pers,
    )


@functools.lru_cache(maxsize=32)
def _forecast_chunk(program: Program, H: int, S_c: int) -> Callable[..., ForecastOut]:
    from rai_bvar.engines.mcmc.runner import forecast_shocks

    n = program.static.shapes.n

    def one(params: ParamDraw, hist: Array, t_last: Array, ckey: Array, d: Array) -> ForecastOut:
        return forecast_core(params, hist, t_last, forecast_shocks(ckey, d, H, n), program)

    batched = jax.jit(jax.vmap(one, in_axes=(0, None, None, 0, 0)))

    def run(params: ParamDraw, hist: Any, t_last: Any, ckeys: Array, d: Any) -> ForecastOut:
        if np.shape(d) != (S_c,):
            raise ValueError(f"this forecast executable takes chunks of {S_c} draws")
        with _jax.x64():
            return batched(params, hist, t_last, ckeys, jnp.asarray(d, jnp.uint32))

    return run


def make_forecast(static: StaticConfig, H: int, S_c: int) -> Callable[..., ForecastOut]:
    """The jitted forecast of ``S_c`` draws for ``static`` (production kernels), cached per
    ``(StaticConfig, H, S_c)``: ``run(params[S_c], hist, t_last, chain_keys[S_c], d[S_c]) ->
    ForecastOut[S_c]`` with ``chain_keys`` the forecast keys ``fold_in(fold_in(key(fseed),
    FORECAST), c)`` of each draw's chain and ``d`` its (uint32) kept-draw index."""
    for name, value in (("H", H), ("S_c", S_c)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    with _jax.x64():
        return _forecast_chunk(compose(static), int(H), int(S_c))


# --------------------------------------------------------------------------- draw selection (§7.1)


def select_draws(stored: int, chains: int, draws: int) -> tuple[np.ndarray, int]:
    """DESIGN §7.1: positions (into each chain's ``stored`` draws) and paths per draw ``R``, the
    rule of ``storage.forecast_selection`` (shared with the in-loop hook, §5.8):
    ``m = ⌈draws/chains⌉`` per chain at positions ``⌊j·stored/m⌋`` (``j = 0..m−1``: distinct,
    evenly spaced, starting at the first stored draw) when ``m ≤ stored``; otherwise every
    stored draw with ``R = ⌈draws/(chains·stored)⌉`` paths each."""
    from rai_bvar.engines.mcmc.storage import forecast_selection

    _check_positive(stored=stored, chains=chains, draws=draws)
    index, R = forecast_selection(int(chains), np.arange(int(stored)), int(draws))
    return np.asarray(index, dtype=np.int64), int(R)


# --------------------------------------------------------------------------- forecasts


def _check_positive(**values: Any) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _get(out: Any, name: str) -> np.ndarray:
    return np.asarray(out[name] if isinstance(out, dict) else getattr(out, name))


def _chains_stored(fit: Any) -> tuple[int, int]:
    leaf = jax.tree.leaves(fit.param_draws)[0]
    shape = np.shape(leaf)
    if len(shape) < 2:
        raise ValueError("fit.param_draws must have leading (chain, draw) axes")
    if len(np.shape(fit.engine_data.hist)) != 2:
        raise NotImplementedError("forecasting a fit with a leading fit axis is not supported")
    return int(shape[0]), int(shape[1])


def _origin(fit: Any) -> pd.Period:
    """The last in-sample period: ``fit.data.last``, ``fit.periods[-1]``, or (engine-level fits
    without periods) a synthetic annual period with ordinal ``T_eff − 1``."""
    data = getattr(fit, "data", None)
    if data is not None and hasattr(data, "last"):
        return data.last
    periods = getattr(fit, "periods", None)
    if isinstance(periods, pd.PeriodIndex) and len(periods):
        return periods[-1]
    T_eff = int(np.asarray(fit.engine_data.T_eff))
    return pd.Period(ordinal=T_eff - 1, freq="Y")


def _attrs(fit: Any, label: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Forecast attrs as the analytic engine writes them plus the MCMC metadata (DESIGN §7.2,
    §5.6); the fit's own MCMC seed and salt are recorded as ``fit_seed`` / ``fit_salt``."""
    from rai_bvar._version import __version__
    from rai_bvar.models.base import spec_hash_of

    meta = dict(getattr(fit, "metadata", None) or {})
    out: dict[str, Any] = {(f"fit_{k}" if k in _FIT_KEYS else k): v for k, v in meta.items()}
    model, data, vol = getattr(fit, "model", None), getattr(fit, "data", None), fit.static.vol
    spec = model.spec() if model is not None and hasattr(model, "spec") else None
    out.update(
        model=getattr(model, "name", "mcmc") if model is not None else "mcmc",
        engine="mcmc",
        tcodes=dict(getattr(data, "tcodes", {}) or {}),
        seed=label,
        converged=int(bool(getattr(fit, "converged", True))),
        warnings="\n".join(getattr(fit, "warnings", []) or []),
        rai_bvar_version=__version__,
        ordering=list(fit.ordering),
        n_chains=int(_chains_stored(fit)[0]),
        mixture=vol.mixture,
        exact=int(vol.exact),
        transitory=vol.transitory,
        outlier_K=int(vol.outlier_K),
        t_scale=vol.t_scale,
    )
    if spec is not None:
        out["spec"] = spec
        out["spec_hash"] = spec_hash_of(spec)
    out.update(extra)
    return out


def _unit_lower(a: np.ndarray) -> np.ndarray:
    """``A = I + strict-lower(a)`` from the stored layout ``a (…, n, n−1)``."""
    n = a.shape[-2]
    A = np.broadcast_to(np.eye(n), (*a.shape[:-2], n, n)).copy()
    rows, cols = np.tril_indices(n, -1)
    A[..., rows, cols] = a[..., rows, cols]
    return A


def _lag_coefs(fit: Any, params: Any) -> np.ndarray | None:
    """``lag_coefs[c, d, l−1, i, j] = A_l[i, j] = B[n_det + (l−1)·n + j, i]`` (DESIGN §7.3, D-34)
    of every ``(chain, draw)`` slice, with ``B`` (k, n) from :func:`_param_maps` (``structure.
    reduced_form(a, Γ)``, the CCCM ``Π`` or zeros); ``None`` for a mean kind outside
    :data:`_LAG_MEANS`, which has no lag matrices to attach."""
    if fit.static.mean.kind not in _LAG_MEANS:
        return None
    s = fit.static.shapes
    lead = np.shape(jax.tree.leaves(params)[0])[:2]
    S = int(np.prod(lead))  # explicit: reshape(-1, …) fails on empty leaves (n = 1: a (C, D, 1, 0))
    flat = jax.tree.map(lambda x: np.asarray(x).reshape(S, *np.shape(x)[2:]), params)
    B, _ = _coefs_chol(fit, flat, np.zeros((S, s.n)))
    blocks = B[:, s.n_det :, :].reshape(*lead, s.lags, s.n, s.n)  # [c, d, l−1, j, i]
    return np.swapaxes(blocks, -1, -2)


def _extras(fit: Any, params: Any, out: Any) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Accelerator variables (DESIGN §7.3): ``cond_A``, ``cond_scale``, ``lag_coefs`` (linear or
    zero means, :func:`_lag_coefs`) and, with a transitory scale, ``cond_scale_persistent`` and
    ``log_p``/``log1m_p``/``nu``. ``params`` holds the parameter draw of every ``(chain, draw)``
    slice (leaves ``(C, D', …)``, replicate slots included)."""
    extra: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "cond_scale": (("chain", "draw", "horizon", "variable"), _get(out, "cond_scale"))
    }
    struct = getattr(params, "struct", None)
    if struct is not None and hasattr(struct, "a"):
        extra["cond_A"] = (("chain", "draw", "variable", "variable_"), _unit_lower(struct.a))
    lag = _lag_coefs(fit, params)
    if lag is not None:
        extra["lag_coefs"] = (("chain", "draw", "lag", "variable", "variable_"), lag)
    trans = params.vol.trans
    if not isinstance(trans, Empty):
        pers = np.exp(0.5 * _get(out, "log_lam_pers"))
        extra["cond_scale_persistent"] = (("chain", "draw", "horizon", "variable"), pers)
        for name in ("log_p", "log1m_p", "nu"):
            if hasattr(trans, name):
                extra[name] = (("chain", "draw", "variable"), np.asarray(getattr(trans, name)))
    return extra


def _result(fit: Any, out: Any, params: Any, label: str, extra_attrs: dict) -> Any:
    from rai_bvar.forecast.result import ForecastResult

    paths = _get(out, "paths")
    data = getattr(fit, "data", None)
    level = (
        data.inverse_paths(paths) if data is not None and hasattr(data, "inverse_paths") else None
    )
    return ForecastResult.from_arrays(
        paths=paths,
        cond_mean=_get(out, "cond_mean"),
        cond_chol=_get(out, "cond_chol"),
        paths_level=level,
        variables=list(fit.ordering),
        origin=_origin(fit),
        attrs=_attrs(fit, label, extra_attrs),
        extra=None if params is None else _extras(fit, params, out),
    )


def _take(tree: Any, c: np.ndarray, p: np.ndarray) -> Any:
    return jax.tree.map(lambda x: np.asarray(x)[c, p], tree)


def _in_loop_result(fit: Any, ilf: Any, label: str) -> Any:
    """The in-loop forecasts of a matching ``FitPlan`` (DESIGN §5.8) as a ``ForecastResult``.
    ``ilf.out`` has leaves ``(chain, M, …)``; slice ``m`` is a path of kept draw
    ``ilf.draw_index[m]``. The accelerator variables need those draws' parameters, so they are
    attached when every such draw is stored (always, for the §7.1 selection)."""
    kept = np.asarray(ilf.draw_index, dtype=np.int64).reshape(-1)
    C = _chains_stored(fit)[0]
    params = None
    stored = np.asarray(fit.draw_index, dtype=np.int64)
    where = np.minimum(np.searchsorted(stored, kept), stored.size - 1)
    if np.array_equal(stored[where], kept):
        params = _take(fit.param_draws, np.arange(C)[:, None], where[None, :])
    extra = {
        "fseed": int(ilf.fseed),
        "draws_per_parameter": int(getattr(ilf, "paths_per_draw", 1)),
        "in_loop": 1,
    }
    return _result(fit, ilf.out, params, label, extra)


def forecast(fit: Any, horizon: int, draws: int = 1000, seed: Any = None) -> Any:
    """Joint predictive paths and their exact conditional moments (DESIGN §7.1–§7.3).

    Returns a ``ForecastResult`` with dims ``(chain, draw, horizon, variable)``: ``C`` chains and
    ``m·R`` draws per chain (module docstring), the contract fields, ``paths_level`` (when the fit
    has its ``VARData``), the accelerator variables of DESIGN §7.3 and attrs as the analytic
    engine (``model``, ``spec``, ``spec_hash``, ``tcodes``, ``seed``, ``converged``,
    ``warnings``, version) plus ``engine="mcmc"``, ``ordering``, ``mixture``, ``exact``,
    ``n_chains``, ``fseed`` and the fit's DESIGN §5.6 metadata. ``seed`` is ``SeedLike``; a
    ``Generator`` advances.
    """
    from rai_bvar.models.base import seed_label

    _check_positive(horizon=horizon, draws=draws)
    ilf = getattr(fit, "in_loop_forecast", None)
    if ilf is not None and ilf.matches(horizon, draws, seed):
        return _in_loop_result(fit, ilf, seed_label(seed))
    rng = np.random.default_rng(seed)
    label = seed_label(seed, rng)
    fseed = int(rng.integers(0, _U32))
    C, D = _chains_stored(fit)
    kept = np.asarray(fit.draw_index, dtype=np.int64)
    pos, R = select_draws(D, C, draws)
    m = pos.size
    D_kept = int(getattr(fit.settings, "draws", int(kept[-1]) + 1))
    if int(kept[pos].max()) + (R - 1) * D_kept >= _U32:
        raise ValueError("forecast keys need kept-draw indices below 2**32")
    # items in (chain, draw, replicate) order: draw slot j·R + r of chain c
    c_idx = np.repeat(np.arange(C), m * R)
    j_idx = np.tile(np.repeat(np.arange(m), R), C)
    r_idx = np.tile(np.arange(R), C * m)
    p_idx = pos[j_idx]
    d_key = (kept[p_idx] + r_idx * D_kept).astype(np.uint32)
    total = c_idx.size
    S_c = min(S_CHUNK, 1 << (total - 1).bit_length())
    from rai_bvar.engines.mcmc.runner import forecast_chain_keys

    with _jax.x64():
        run = _forecast_chunk(fit.program, int(horizon), S_c)
        ckeys = forecast_chain_keys(fseed, C)
        hist, t_last = fit.engine_data.hist, fit.engine_data.t_last
        chunks = []
        for lo in range(0, total, S_c):
            sl = np.minimum(np.arange(lo, lo + S_c), total - 1)  # pad with the last item
            params = _take(fit.param_draws, c_idx[sl], p_idx[sl])
            out = run(params, hist, t_last, ckeys[c_idx[sl]], d_key[sl])
            keep = min(S_c, total - lo)
            chunks.append(jax.tree.map(lambda x, k=keep: np.asarray(x)[:k], out))
    out = jax.tree.map(lambda *xs: np.concatenate(xs).reshape(C, m * R, *xs[0].shape[1:]), *chunks)
    params = _take(fit.param_draws, c_idx.reshape(C, m * R), p_idx.reshape(C, m * R))
    extra = {"fseed": fseed, "draws_per_parameter": int(R), "in_loop": 0}
    return _result(fit, out, params, label, extra)


# --------------------------------------------------------------------------- parameters (§7.4)


def _date_row(fit: Any, date: Any) -> int | None:
    """Data row ``t`` (0-based, of the ``T_eff`` rows) of ``date``; ``None`` for ``None``."""
    if date is None:
        return None
    periods = getattr(fit, "periods", None)
    T_eff = int(np.asarray(fit.engine_data.T_eff))
    if isinstance(periods, pd.PeriodIndex):
        try:
            period = pd.Period(date, freq=periods.freq)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"date {date!r} is not a period of frequency {periods.freqstr}"
            ) from err
        if period not in periods:
            raise ValueError(
                f"date {period} is outside the estimation sample {periods[0]}..{periods[-1]} "
                "(the first lags observations are presample)"
            )
        return int(periods.get_loc(period))
    if isinstance(date, numbers.Integral) and not isinstance(date, bool) and 0 <= date < T_eff:
        return int(date)
    raise ValueError(f"date must be a data row in [0, {T_eff}) for a fit without periods")


class _Pool(NamedTuple):
    """Where parameter draws come from: stored params (``paths is None``) or the path draws."""

    params: Any  # ParamDraw, leaves (C, D, …)
    paths: Any  # PathDraw, leaves (C, Dp, …), or None


def _pool(fit: Any, row: int | None, which: str, shock: str = "sd") -> _Pool:
    """The draw pool for ``(date row, which)``: the stored params, or the subset of them with
    stored paths when the volatility at ``row`` needs ``h`` or ``2 log o + log τ`` paths."""
    if which not in _WHICH:
        raise ValueError(f"which must be one of {_WHICH}, got {which!r}")
    vol = fit.static.vol
    T_eff = int(np.asarray(fit.engine_data.T_eff))
    transitory = vol.transitory != "none"
    needs = shock == "sd" and (
        (row is not None and row != T_eff - 1 and vol.persistent != "constant")
        or (which == "effective" and transitory)
    )
    if not needs:
        return _Pool(fit.param_draws, None)
    if getattr(fit, "path_draws", None) is None:
        if which == "effective" and transitory:
            what = "the effective variance (persistent × transitory scale)"
        else:
            what = "a date other than the last in-sample period"
        raise ValueError(
            f"{what} needs the latent paths: refit with MCMCSettings(store='paths') "
            "(or use date=None, which='persistent', or shock='unit')"
        )
    kept_paths = np.asarray(fit.store.plan.streams["paths"].draw_index(), dtype=np.int64)
    where = np.searchsorted(fit.draw_index, kept_paths)
    if not np.array_equal(np.asarray(fit.draw_index)[where], kept_paths):
        raise RuntimeError("path draws are not a subset of the stored parameter draws")
    C = _chains_stored(fit)[0]
    params = _take(fit.param_draws, np.arange(C)[:, None], where[None, :])
    return _Pool(params, fit.path_draws)


def _log_lam(fit: Any, pool: _Pool, row: int | None, which: str) -> np.ndarray:
    """Log shock variances (C, D, n) at data row ``row`` (``None``: the last in-sample period).

    ``"effective"``: ``h_t + 2 log o_t + log τ_t`` (the stored paths). ``"persistent"``: the
    variance net of outliers (N03 §5.4), ``e^{h_t}``, times ``E τ = ν/(ν−2)`` under
    ``t_scale="unit"`` (standardized t has ``E τ = 1``); ``ν ≤ 2`` (infinite variance) raises.
    """
    T_eff = int(np.asarray(fit.engine_data.T_eff))
    t = T_eff - 1 if row is None else row
    if pool.paths is not None:
        lam = np.asarray(pool.paths.h)[..., t + 1]  # h index t + 1 ↔ data row t
        if fit.static.vol.persistent == "constant":
            lam = np.asarray(pool.params.vol.core.log_lam)
        if which == "effective":
            return lam + np.asarray(pool.paths.log_scale2)[..., t]
    else:
        core = pool.params.vol.core
        if hasattr(core, "h_T"):
            lam = np.asarray(core.h_T)
        elif hasattr(core, "log_lam"):
            lam = np.asarray(core.log_lam)
        else:
            raise TypeError(f"no log variance in a {type(core).__name__} draw")
    nu = getattr(pool.params.vol.trans, "nu", None)
    if which == "persistent" and fit.static.vol.t_scale == "unit" and nu is not None:
        nu = np.asarray(nu, dtype=np.float64)
        if np.any(nu <= 2.0):
            raise ValueError(
                "the persistent shock variance e^h ν/(ν−2) of a unit-scale Student-t is infinite "
                "for ν ≤ 2, which some posterior draws have (N03 §5.4): use shock='unit', "
                "which='effective' (with store='paths') or nu_prior=DiscreteUniform(low ≥ 3, …)"
            )
        lam = lam + np.log(nu / (nu - 2.0))
    return lam


@functools.lru_cache(maxsize=32)
def _param_maps(program: Program) -> Callable[..., tuple[Array, Array]]:
    """``(coefs, chol)`` of a batch of draws: ``B = structure.reduced_form(a, Γ)`` (structural),
    ``Π`` (reduced) or zeros, and ``structure.chol(a, log λ)``."""
    s = program.static.shapes
    structure = program.structure

    def one(params: ParamDraw, log_lam: Array) -> tuple[Array, Array]:
        if isinstance(params.mean, LinearDraw):
            B = structure.reduced_form(params.struct, params.mean.gamma)
        elif isinstance(params.glob.mean, CCCMDraw):
            B = params.glob.mean.Pi
        else:
            B = jnp.zeros((s.k, s.n), jnp.float64)
        return B, structure.chol(params.struct, log_lam)

    return jax.jit(jax.vmap(one))


def _coefs_chol(fit: Any, params: Any, log_lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(coefs, chol)`` for parameter draws with leaves ``(S, …)`` and ``log_lam (S, n)``."""
    with _jax.x64():
        B, L = _param_maps(fit.program)(params, jnp.asarray(log_lam, jnp.float64))
    return np.asarray(B), np.asarray(L)


def draw_parameters(
    fit: Any, size: int, rng: Any = None, date: Any = None, which: str = "persistent"
) -> tuple[np.ndarray, np.ndarray]:
    """``(coefs (S, k, n), chol (S, n, n))`` from the stored posterior draws (DESIGN §7.4).

    ``S = size`` draws evenly spaced over the pooled (chain-major) stored draws, as
    ``AnalyticFit`` takes MH hyperparameter draws; ``rng`` is accepted for the
    ``AnalyticFit.draw_parameters`` signature and unused (the selection is deterministic).
    ``coefs = B = (A⁻¹Γ)'`` or ``Π``; ``chol = A⁻¹ diag(exp(½ log λ_date))`` with the persistent
    (default; ``e^h``, times ``ν/(ν−2)`` under ``t_scale="unit"``, N03 §5.4) or effective
    (``e^h o² τ``) variance at ``date`` (``None``: the last in-sample period). Other
    dates and the effective variance under a transitory scale need ``store="paths"`` (then the
    pool is the stored path draws) and raise with that advice otherwise; a Constant core is
    date-free. Dates outside the estimation sample raise ``ValueError``.
    """
    _check_positive(size=size)
    row = _date_row(fit, date)
    pool = _pool(fit, row, which)
    lam = _log_lam(fit, pool, row, which)  # (C, D, n)
    C, D = lam.shape[:2]
    flat = np.round(np.linspace(0, C * D - 1, int(size))).astype(np.int64)
    c, p = np.divmod(flat, D)
    return _coefs_chol(fit, _take(pool.params, c, p), lam[c, p])


def irf(
    fit: Any,
    horizon: int = 20,
    draws: int = 1000,
    seed: Any = None,
    shock: str = "sd",
    date: Any = None,
    which: str = "persistent",
) -> Any:
    """Recursive impulse responses (DESIGN §7.4) through ``structural.irf.recursive_irf``.

    Per chain ``min(⌈draws/C⌉, D)`` evenly spaced stored draws (:func:`select_draws`; IRFs are
    deterministic given a draw, so draws are never replicated), keeping ``(chain, draw)``.
    ``shock="sd"``: ``Θ_h = Φ_h A⁻¹ Λ_date^{½}`` with the persistent variance by default (``e^h``,
    the shock variance net of outliers; ``e^h ν/(ν−2)`` under ``t_scale="unit"``), and
    ``which="effective"`` uses ``e^h o²τ`` (N03 §5.4); ``shock="unit"``: ``Φ_h A⁻¹``, date-free.
    ``seed`` is recorded (the responses use no randomness). FEVD (``IRFResult.fevd``) is
    date-specific under SV.
    """
    from rai_bvar.models.base import seed_label
    from rai_bvar.structural import IRFResult, recursive_irf

    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 0:
        raise ValueError(f"horizon must be an integer >= 0, got {horizon!r}")
    _check_positive(draws=draws)
    if shock not in ("sd", "unit"):
        raise ValueError(f"shock must be 'sd' or 'unit', got {shock!r}")
    rng = np.random.default_rng(seed)
    label = seed_label(seed, rng)
    row = _date_row(fit, date)
    pool = _pool(fit, row, which, shock)
    if shock == "unit":  # Φ_h A⁻¹ needs no variance: chol = A⁻¹ (unit diagonal)
        C, D = np.shape(pool.params.struct.a)[:2]
        lam = np.zeros((C, D, fit.static.shapes.n))
    else:
        lam = _log_lam(fit, pool, row, which)
    C, D = lam.shape[:2]
    pos, _ = select_draws(D, C, min(int(draws), C * D))
    c = np.repeat(np.arange(C), pos.size)
    p = np.tile(pos, C)
    coefs, chol = _coefs_chol(fit, _take(pool.params, c, p), lam[c, p])
    s = fit.static.shapes
    responses = recursive_irf(coefs, chol, s.lags, s.n_det, int(horizon), shock=shock)
    responses = responses.reshape(C, pos.size, *responses.shape[1:])
    periods = getattr(fit, "periods", None)
    T_eff = int(np.asarray(fit.engine_data.T_eff))
    t = T_eff - 1 if row is None else row
    when = str(periods[t]) if isinstance(periods, pd.PeriodIndex) else str(t)
    full = _attrs(fit, label, {"date": when, "which": which})
    keep = ("model", "spec_hash", "seed", "engine", "converged", "fit_seed", "fit_salt", "date",
            "which")  # fmt: skip
    return IRFResult.from_arrays(
        responses, list(fit.ordering), shock=shock, attrs={k: full[k] for k in keep if k in full}
    )
