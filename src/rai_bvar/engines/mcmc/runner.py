"""Chain runner: executables, host driver, devices, deterministic replay (DESIGN §5).

Executables (``make_runner``; lru-cached on ``StaticConfig`` when there are no overrides)::

    prepare (data, prior) -> FitCache                                          jit
    init    (init_keys[C], data, prior, cache) -> ChainState[C]                jit
    burn    (ckeys[C], state, data, prior, cache, it0: u32[], n: i32[]) -> state          jit, donated
    sample  (ckeys[C], state, data, prior, cache, it0: u32[], thin: u32[], stop: i32[])
            -> (state, snap, out[C, chunk])                                    jit, donated
    extract (state, data, prior, cache) -> (monitor, ParamDraw, PathDraw | Empty)   one chain

``burn`` runs exactly ``n`` sweeps with absolute indices ``it0 .. it0+n−1`` (a traced trip count:
no recompile across ``burn``, ``warm_burn`` or piece sizes; D-27). ``sample`` scans ``chunk``
kept draws, each after ``thin`` sweeps (``it0 + j·thin + r``, all uint32; E7), and keeps ``snap``,
the exact state at kept draw ``stop`` (D-27). Keys depend only on ``(seed, salt, chain, it,
block, equation, sub-draw)``, so kept draws and the final state are bitwise independent of the
chunk size, the burn piece size, thinning alignment and fused vs unfused plan execution (DESIGN
§5.1, §5.6). Across chain-count prefixes and ``vmap`` vs ``shard_map`` (§5.2) they agree **to
rounding** (Stage 2 amendment of §5.2, §5.6): chain ``c`` consumes the same random numbers
whatever ``C`` is, but XLA vectorises the reductions of the production SV and CCCM kernels
differently for different per-device batch sizes, so last bits differ (``rtol 1e-9`` on short
runs, tested; over long runs a last-bit difference can flip an accept/reject decision, after
which chains are equal only in distribution). Batch-size-invariant kernels (the test toys) and
``chains="map"`` (:func:`chain_map`, one identical one-chain program per chain) keep chain-count
prefixes bitwise. ``Runtime.devices`` therefore defaults to one device; the device count and the
chain method are recorded in ``fit.metadata`` (``device_count``, ``chain_method``).

Host driver ``run_chains`` (DESIGN §5.1): prepare → init (or a warm state) → burn in pieces of
``min(remaining, chunk·thin)`` sweeps → zero every ``acc_*``/``rb_*`` leaf at the boundary →
``⌈draws/chunk⌉`` sample calls whose records ``0..stop`` go to the :class:`HostStore`; the final
state is ``snap`` of the last call (exactly ``burn + draws·thin`` sweeps). After every call the
status codes and the state invariants (DESIGN §2.8) are checked on the host; a failure is replayed
deterministically from the saved host copy of the call's start state (D-18, §5.7) and raised as
``MCMCNumericalError`` with diagnostics recomputed from the failing block's exposed conditional
(an equation step's ``conditional_eq`` / ``contemp_conditional``, a global step's
``conditional_j`` for every ``j``, which also locates the failing equation of a global step).

In-loop forecasts (DESIGN §5.8; ``StoreStatic.in_loop_forecast = H > 0`` with a ``FitPlan``):
``sample`` takes four more arguments ``(fkeys[C], d0: u32[], sel: bool[chunk], D: u32[])`` and a
static keyword ``reps`` (default 1), and calls the forecast core
``engines.mcmc.forecast.forecast_core`` (WP17, §7.2) on the parameters of kept draw ``d = d0 + j``
when ``sel[j]``, under a ``lax.cond`` whose predicate is shared by all chains (so it stays a branch
under ``vmap``). The host precomputes ``sel`` from :func:`storage.forecast_selection`: exactly the
draws §7.1 selects post hoc (``⌈S/C⌉`` stored draws per chain, evenly spaced; when that exceeds the
stored draws, every stored draw with ``reps = ⌈S/(C·D_s)⌉`` paths), so the run yields at least
``S = FitPlan.draws`` paths (fewer than ``C`` extra in the first case). Shocks come from
:func:`forecast_shocks`: ``key(fseed) → Stream.FORECAST → c → d → Ahead.{VOL, SHOCK, MEAN, GLOBAL}
→ i`` (§5.6), ``d`` the kept-draw index (the ``draw`` coordinate of ``fit.posterior``, whatever
``store_thin`` is; path ``q`` of a draw uses ``d + q·D``), ``fseed`` from ``FitPlan.seed`` as in
§7.1. The records go to the host store's ``forecast`` stream (``MCMCFit.in_loop_forecast``).
"""

from __future__ import annotations

import dataclasses
import functools
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from rai_bvar.engines.mcmc import _blas, _jax, keys
from rai_bvar.engines.mcmc import data as data_mod
from rai_bvar.engines.mcmc import monitor as monitor_mod
from rai_bvar.engines.mcmc.compose import Program, compose
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.keys import Ahead, Block, Stream
from rai_bvar.engines.mcmc.plan import Plan, Step, build_plan
from rai_bvar.engines.mcmc.protocols import MCMCNumericalError
from rai_bvar.engines.mcmc.state import (
    ChainState,
    ConstantPrior,
    Empty,
    EngineData,
    EqState,
    FitCache,
    ParamDraw,
    PathDraw,
    PriorArrays,
    Status,
    SVPrior,
    SweepCtx,
    check_state_invariants,
    decode_error,
    leaf_path,
    state_template,
    tree_where,
    zero_accumulators,
)
from rai_bvar.engines.mcmc.storage import HostStore, StoragePlan, forecast_selection, plan_storage
from rai_bvar.engines.mcmc.sweep import (
    EQUATION_STEPS,
    empty_env,
    equation_phase,
    make_sweep,
    mean_context,
    sweep_values,
)

Array = jax.Array

__all__ = [
    "ChainBodies",
    "ForecastShocks",
    "RunOutput",
    "RunnerFns",
    "chain_map",
    "clear_caches",
    "failure_detail",
    "forecast_chain_keys",
    "forecast_seed",
    "forecast_shocks",
    "initial_log_variance",
    "make_runner",
    "run_chains",
]

_U32 = 2**32


class RunnerFns(NamedTuple):
    """The jitted executables of one ``StaticConfig`` (DESIGN §5.1), plus their program, whether
    they carry a leading fit axis, the unjitted one-chain bodies (``shard_map``, replay) and the
    chain batching (:func:`chain_map`). The first five fields are the DESIGN contract."""

    prepare: Callable[..., Any]
    init: Callable[..., Any]
    burn: Callable[..., Any]
    sample: Callable[..., Any]
    extract: Callable[..., Any]
    program: Program | None = None
    fit_axis: bool = False
    chain: ChainBodies | None = None
    chains: str = "vmap"


@dataclass(frozen=True)
class ChainBodies:
    """One-fit, one-chain functions behind :class:`RunnerFns` (not jitted, not batched).

    ``forecast(params, data, fkey, d) -> ForecastOut`` (in-loop forecasts only, else ``None``):
    the forecast core on one draw's parameters with the shocks of kept draw ``d``."""

    prepare: Callable[..., Any]
    init: Callable[..., Any]
    burn: Callable[..., Any]
    sample: Callable[..., Any]
    extract: Callable[..., Any]
    plan: Plan
    forecast: Callable[..., Any] | None = None


# --------------------------------------------------------------------------- in-loop forecasts (§5.8)


class ForecastShocks(NamedTuple):
    """Injected forecast randomness of one draw (DESIGN §7.2): ``eps`` (H, n) standard normals,
    per-equation keys for the volatility and mean futures, one key for global futures. Used when
    ``engines.mcmc.forecast`` (WP17) does not define its own ``ForecastShocks`` with these fields."""

    eps: Array
    vol_keys: Array
    mean_keys: Array
    glob_key: Array


def forecast_seed(seed: int) -> int:
    """``fseed`` of a ``FitPlan`` seed: ``int(default_rng(seed).integers(0, 2**32))``, the
    derivation of DESIGN §7.1 for an integer seed (so in-loop and post-hoc forecasts share keys)."""
    return int(np.random.default_rng(int(seed)).integers(0, 2**32))


def forecast_chain_keys(fseed: int, C: int) -> Array:
    """``fold_in(fold_in(key(fseed), Stream.FORECAST), c)`` for ``c = 0..C−1`` (DESIGN §5.6)."""
    root = random.fold_in(random.key(int(fseed), impl=keys.PRNG_IMPL), int(Stream.FORECAST))
    return keys.chain_keys(root, C)


def forecast_shocks(chain_fkey: Array, d: Array, H: int, n: int) -> Any:
    """The shocks of kept draw ``d`` of the chain whose forecast key is ``chain_fkey``
    (DESIGN §5.6, §7.2): with ``k(X, i) = fold_in(fold_in(fold_in(chain_fkey, d), Ahead.X), i)``,
    ``eps[:, i] = N(0, I_H)`` from ``k(SHOCK, i)``, ``vol_keys[i] = k(VOL, i)``, ``mean_keys[i] =
    k(MEAN, i)``, ``glob_key = k(GLOBAL, 0)``; equal to ``keys.forecast_key(fseed, c, d, X, i)``.
    Post-hoc forecasts that must reproduce in-loop ones (§5.8) draw their shocks this way."""
    kd = random.fold_in(chain_fkey, jnp.asarray(d, jnp.uint32))
    idx = jnp.arange(n, dtype=jnp.uint32)

    def kind_keys(kind: Ahead) -> Array:
        k = random.fold_in(kd, int(kind))
        return jax.vmap(lambda i: random.fold_in(k, i))(idx)

    eps = jax.vmap(lambda k: random.normal(k, (H,), dtype=jnp.float64))(kind_keys(Ahead.SHOCK)).T
    glob_key = random.fold_in(random.fold_in(kd, int(Ahead.GLOBAL)), jnp.uint32(0))
    from rai_bvar.engines.mcmc import forecast as forecast_mod

    cls = getattr(forecast_mod, "ForecastShocks", ForecastShocks)
    return cls(
        eps=eps, vol_keys=kind_keys(Ahead.VOL), mean_keys=kind_keys(Ahead.MEAN), glob_key=glob_key
    )


def _forecast_core() -> Callable[..., Any]:
    from rai_bvar.engines.mcmc import forecast as forecast_mod

    fn = getattr(forecast_mod, "forecast_core", None)
    if fn is None:
        raise NotImplementedError(
            "in-loop forecasting (StoreStatic.in_loop_forecast, DESIGN §5.8) calls "
            "engines.mcmc.forecast.forecast_core (WP17, DESIGN §7.2), which is not available yet"
        )
    return fn


def _forecast_fn(program: Program) -> Callable[..., Any] | None:
    """``forecast(params, data, fkey, d)`` of one chain and one draw, or ``None`` when off."""
    H = program.static.store.in_loop_forecast
    if not H:
        return None
    core = _forecast_core()
    n = program.static.shapes.n

    def forecast_one(params: ParamDraw, data: EngineData, fkey: Array, d: Array) -> Any:
        shocks = forecast_shocks(fkey, d, H, n)
        return core(params, data.hist, data.t_last, shocks, program)

    return forecast_one


# --------------------------------------------------------------------------- initialisation (§5.4)


def initial_log_variance(prior_i: Any, rw: bool) -> Array:
    """``log E λ_i`` under the volatility prior: the weights of the cold-start draw (DESIGN §5.4).

    Constant ``IG(a, b)``: ``log(b/(a − 1))`` (``a > 1``; Mode I: ``ψ_ι/ι``), else ``log(b/a)``;
    SV: the level ``b_μ`` (AR) or ``m₀`` (RW), i.e. ``log(ψ_ι/ι)`` with the data-centred priors.
    """
    core = prior_i.vol_core
    if isinstance(core, ConstantPrior):
        shape = jnp.asarray(core.shape, jnp.float64)
        return jnp.log(core.scale) - jnp.log(jnp.where(shape > 1.0, shape - 1.0, shape))
    if isinstance(core, SVPrior):
        return jnp.asarray(core.m0 if rw else core.b_mu, jnp.float64)
    return jnp.zeros((), jnp.float64)


@dataclass(frozen=True)
class _ProvisionalVol:
    """Volatility stand-in for the cold-start context: constant ``log λ`` from the prior."""

    log_lam: Array
    T_pad: int

    def log_var_eq(self, vs: Any, gvol: Any, which: str = "effective") -> Array:
        return jnp.broadcast_to(self.log_lam, (self.T_pad,))

    def log_scale_eq(self, vs: Any) -> Array:
        return self.log_lam


def _jnp_template(static: StaticConfig) -> ChainState:
    return jax.tree.map(jnp.asarray, state_template(static))


def _init_fn(program: Program) -> Callable[..., ChainState]:
    static = program.static
    n, T_pad = static.shapes.n, static.shapes.T_pad
    structure, mean, coef, vol = program.structure, program.mean, program.coef, program.vol

    def init_one(ikey: Array, data: EngineData, prior: PriorArrays, cache: FitCache) -> ChainState:
        idx = jnp.arange(n, dtype=jnp.int32)
        tmpl = _jnp_template(static)

        def key(block: Block, i: Array) -> Array:
            return keys.eq_key(keys.block_key(ikey, block), i)

        def first(i, prior_i):
            s_i = structure.init_eq(key(Block.EQ_CONTEMP, i), i, prior_i)
            c_i = coef.init_eq(key(Block.EQ_COEF_LOCAL, i), i, prior_i)
            return s_i, c_i

        struct, coefs = program.eq_map(first)(idx, prior.eq)
        eq0 = EqState(struct=struct, mean=tmpl.eq.mean, coef=coefs, vol=tmpl.eq.vol)
        st = ChainState(eq=eq0, glob=tmpl.glob, status=Status.initial())
        for g in program.global_pre:
            st = g.init(keys.block_key(ikey, Block(int(g.block))), st, data, prior, cache)
        ytil = structure.pre_residual(st.glob, data, mean.fitted_reduced(st.glob, data))
        glob = st.glob

        def second(i, eq_i, prior_i, cache_i):
            vol0 = _ProvisionalVol(initial_log_variance(prior_i, static.vol.rw), T_pad)
            ctx = structure.context(i, ytil, eq_i, vol0, data, glob.vol)
            cm = structure.contemp_moments(ctx, prior_i)
            pm = coef.moments_eq(eq_i.coef, i, prior_i, glob.coef)
            a0 = eq_i.struct.a
            ctx_m = ctx if mean.fuses_contemp else mean_context(ctx, a0)
            ms, a = mean.init_eq(
                key(Block.EQ_MEAN, i), a0, ctx_m, pm, cm, cache_i, cache, prior.shared
            )
            g = mean.fitted_eq(ms, a, ctx_m, cache, prior.shared)
            e = structure.residual(a, ctx, g)
            vs = vol.init_eq(key(Block.EQ_VOL, i), e, data.tmask, data.T_eff, prior_i)
            return EqState(
                struct=dataclasses.replace(eq_i.struct, a=a), mean=ms, coef=eq_i.coef, vol=vs
            )

        eq = program.eq_map(second)(idx, st.eq, prior.eq, cache.eq)
        st = dataclasses.replace(st, eq=eq)
        for g in program.global_post:
            st = g.init(keys.block_key(ikey, Block(int(g.block))), st, data, prior, cache)
        return st

    return init_one


# --------------------------------------------------------------------------- extract


def _monitor_values(st, data, prior, cache, program: Program) -> Any:
    fn = getattr(monitor_mod, "monitor_values", None)
    if fn is None:
        raise NotImplementedError(
            "engines.mcmc.monitor.monitor_values (WP15) is not available yet; the sample "
            "executable records the monitored quantities through it (DESIGN §8.1)"
        )
    return fn(st, data, prior, cache, program)


def extract_params(
    program: Program, st: ChainState, data: EngineData, prior: PriorArrays
) -> ParamDraw:
    """The stored parameters of one chain's state (DESIGN §2.10, §5.3)."""
    mean, vol = program.mean, program.vol
    glob = mean.extract_global(st.glob)
    mean_d = program.eq_map(lambda ms, a: mean.extract_eq(ms, a, prior.shared))(
        st.eq.mean, st.eq.struct.a
    )
    struct_d = program.structure.extract(st.eq.struct, st.glob)
    vol_d = program.eq_map(lambda vs: vol.extract_eq(vs, data.T_eff))(st.eq.vol)
    return ParamDraw(glob=glob, mean=mean_d, struct=struct_d, vol=vol_d)


def extract_paths(program: Program, st: ChainState) -> PathDraw:
    """``h`` (``T_pad+1`` states) and ``2 log o + log τ`` per equation (``store="paths"``)."""
    vol, T = program.vol, program.static.shapes.T_pad
    gvol = st.glob.vol

    def lv(vs, which):
        return jnp.broadcast_to(vol.log_var_eq(vs, gvol, which=which), (T,))

    pers = program.eq_map(lambda vs: lv(vs, "persistent"))(st.eq.vol)
    eff = program.eq_map(lambda vs: lv(vs, "effective"))(st.eq.vol)
    core = st.eq.vol.core
    h = core.h if hasattr(core, "h") else jnp.concatenate([pers[:, :1], pers], axis=1)
    return PathDraw(h=h, log_scale2=eff - pers)


def _extract_fn(program: Program) -> Callable[..., Any]:
    paths = program.static.store.paths

    def extract_one(st, data, prior, cache):
        mon = _monitor_values(st, data, prior, cache, program)
        params = extract_params(program, st, data, prior)
        return mon, params, (extract_paths(program, st) if paths else Empty())

    return extract_one


# --------------------------------------------------------------------------- executables


def _prepare_fn(program: Program) -> Callable[..., FitCache]:
    static = program.static

    def prepare_one(data: EngineData, prior: PriorArrays) -> FitCache:
        Mc = prior.shared.Mc if static.mean.kind == "linear_structural" else None
        if static.mean.n_dummy:  # SOC/SUR dummy rows (WP25); closed path otherwise
            base = data_mod.base_cache(
                data, Mc, static, dummies=prior.shared.dummies, dummy_w=prior.eq.coef.dummy_w
            )
        else:
            base = data_mod.base_cache(data, Mc, static)
        return program.mean.prepare(data, prior, base)

    return prepare_one


def _bodies(program: Program) -> ChainBodies:
    static = program.static
    sweep = make_sweep(program)
    extract_one = _extract_fn(program)
    forecast_one = _forecast_fn(program)
    chunk = static.chunk

    def burn_one(ckey, st, data, prior, cache, it0, n):
        def body(r, s):
            return sweep(ckey, it0 + r.astype(jnp.uint32), s, data, prior, cache)

        return lax.fori_loop(0, n, body, st)

    def in_loop(params, data, fkey, d, pred, dstride, reps):
        """The ``reps`` forecast paths (leading axis) of kept draw ``d`` when ``pred`` (the draw is
        selected), zeros otherwise; path ``q`` is keyed by ``d + q·dstride`` (``dstride = D``).
        ``pred`` does not depend on the chain, so ``vmap`` keeps the branch (no wasted work)."""

        def paths(params, data, fkey, d, dstride):
            if reps == 1:
                return jax.tree.map(lambda x: x[None], forecast_one(params, data, fkey, d))
            q = jnp.arange(reps, dtype=jnp.uint32)
            return jax.vmap(lambda qq: forecast_one(params, data, fkey, d + qq * dstride))(q)

        shape = jax.eval_shape(paths, params, data, fkey, d, dstride)

        def zeros(*_: Any) -> Any:
            return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shape)

        return lax.cond(pred, paths, zeros, params, data, fkey, d, dstride)

    def sample_one(
        ckey, st, data, prior, cache, it0, thin, stop, fkey=None, d0=None, sel=None, dstride=None,
        *, reps: int = 1,
    ):  # fmt: skip
        def keep(carry, j):
            st, snap = carry
            base = it0 + j * thin  # uint32 throughout (E7)

            def body(r, s):
                return sweep(ckey, base + r.astype(jnp.uint32), s, data, prior, cache)

            st = lax.fori_loop(0, thin.astype(jnp.int32), body, st)
            snap = tree_where(j == stop.astype(jnp.uint32), st, snap)
            rec = extract_one(st, data, prior, cache)
            if forecast_one is not None:  # DESIGN §5.8: (monitor, params, paths, forecast)
                rec = (*rec, in_loop(rec[1], data, fkey, d0 + j, sel[j], dstride, reps))
            return (st, snap), rec

        (st, snap), out = lax.scan(keep, (st, st), jnp.arange(chunk, dtype=jnp.uint32))
        return st, snap, out

    return ChainBodies(
        prepare=_prepare_fn(program),
        init=_init_fn(program),
        burn=burn_one,
        sample=sample_one,
        extract=extract_one,
        plan=sweep.plan,
        forecast=forecast_one,
    )


def _named(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


ChainBatching = Literal["vmap", "map"]


def chain_map(
    f: Callable[..., Any], in_axes: tuple[int | None, ...], chains: ChainBatching = "vmap"
) -> Callable[..., Any]:
    """Batch ``f`` over the chain axis (``in_axes``: 0 for batched arguments, ``None`` shared).

    ``"vmap"`` (default; DESIGN §5.2): one program for all chains, the equation-chain GEMMs of
    §10.3. Chain-count prefixes agree to rounding with the production SV and CCCM kernels (XLA
    vectorises their reductions differently for different batch sizes; §5.6 Stage 2 amendment,
    ``rtol 1e-9`` on short runs, tested) and bitwise with batch-size-invariant kernels.
    ``"map"``: ``lax.map`` over chains, each chain running the identical one-chain program, so
    every chain's draws are bitwise independent of the chain count by construction. Across device
    layouts (``shard_map``) both modes agree with one device to rounding only.
    """
    if chains == "vmap":
        return jax.vmap(f, in_axes=in_axes)
    if chains != "map":
        raise ValueError(f"chains must be 'vmap' or 'map', got {chains!r}")
    where = [k for k, ax in enumerate(in_axes) if ax is not None]

    def mapped(*args: Any) -> Any:
        def one(batched: tuple[Any, ...]) -> Any:
            full = list(args)
            for k, value in zip(where, batched, strict=True):
                full[k] = value
            return f(*full)

        return lax.map(one, tuple(args[k] for k in where))

    return mapped


_CHAIN_AXES = {
    "init": (0, None, None, None),
    "burn": (0, 0, None, None, None, None, None),
    "sample": (0, 0, None, None, None, None, None, None),
    "forecast": (0, None, None, None),  # (fkeys[C], d0, sel, D) of the in-loop forecast (§5.8)
}


def _sample_axes(static: StaticConfig) -> tuple[int | None, ...]:
    extra = _CHAIN_AXES["forecast"] if static.store.in_loop_forecast else ()
    return (*_CHAIN_AXES["sample"], *extra)


def _make_runner(
    static: StaticConfig,
    fit_axis: bool,
    overrides: Mapping[str, Any] | None,
    chains: ChainBatching = "vmap",
) -> RunnerFns:
    program = compose(static, overrides=overrides)
    b = _bodies(program)
    prepare = b.prepare
    init = chain_map(b.init, _CHAIN_AXES["init"], chains)
    burn = chain_map(b.burn, _CHAIN_AXES["burn"], chains)
    fc_fit = (0, None, None, None) if static.store.in_loop_forecast else ()

    @functools.cache
    def sample_paths(reps: int) -> Callable[..., Any]:
        """The batched ``sample`` body with ``reps`` in-loop forecast paths per selected draw."""
        f = chain_map(functools.partial(b.sample, reps=reps), _sample_axes(static), chains)
        if fit_axis:
            f = jax.vmap(f, in_axes=(0, 0, 0, 0, 0, None, None, None, *fc_fit))
        return f

    if fit_axis:  # one more vmap over a leading fit axis F (DESIGN §5.2)
        prepare = jax.vmap(prepare)
        init = jax.vmap(init, in_axes=(0, 0, 0, 0))
        burn = jax.vmap(burn, in_axes=(0, 0, 0, 0, 0, None, None))
    p, i, bu = prepare, init, burn

    def mcmc_prepare(data, prior):
        return p(data, prior)

    def mcmc_init(ikeys, data, prior, cache):
        return i(ikeys, data, prior, cache)

    def mcmc_burn(ckeys, state, data, prior, cache, it0, n):
        return bu(ckeys, state, data, prior, cache, it0, n)

    def mcmc_sample_chunk(ckeys, state, data, prior, cache, it0, thin, stop, *forecast, reps=1):
        s = sample_paths(int(reps))
        return s(ckeys, state, data, prior, cache, it0, thin, stop, *forecast)

    def mcmc_extract(state, data, prior, cache):
        return b.extract(state, data, prior, cache)

    return RunnerFns(
        prepare=jax.jit(mcmc_prepare),
        init=jax.jit(mcmc_init),
        burn=jax.jit(mcmc_burn, donate_argnums=(1,)),
        sample=jax.jit(mcmc_sample_chunk, donate_argnums=(1,), static_argnames=("reps",)),
        extract=jax.jit(mcmc_extract),
        program=program,
        fit_axis=bool(fit_axis),
        chain=b,
        chains=chains,
    )


@functools.lru_cache(maxsize=32)
def _make_runner_cached(static: StaticConfig, fit_axis: bool, chains: ChainBatching) -> RunnerFns:
    return _make_runner(static, fit_axis, None, chains)


def make_runner(
    static: StaticConfig,
    *,
    fit_axis: bool = False,
    overrides: Mapping[str, Any] | None = None,
    chains: ChainBatching = "vmap",
) -> RunnerFns:
    """The executables of ``static`` (DESIGN §5.1); lru-cached unless ``overrides`` are given.

    ``chains`` (an addition to the DESIGN signature) selects the chain batching of
    :func:`chain_map`; ``fit_axis`` adds a leading fit axis (``vmap``; DESIGN §5.2). With
    ``static.store.in_loop_forecast > 0`` the ``sample`` executable carries the in-loop forecast
    hook (DESIGN §5.8; four more arguments and the static ``reps``, see the module docstring;
    each ``reps`` compiles once) and building it raises
    ``NotImplementedError`` while ``engines.mcmc.forecast.forecast_core`` (WP17) is missing.
    """
    if overrides:
        return _make_runner(static, bool(fit_axis), dict(overrides), chains)
    return _make_runner_cached(static, bool(fit_axis), chains)


def clear_caches() -> None:
    """Drop the cached programs, sweeps, runners and sharded executables (tests)."""
    from rai_bvar.engines.mcmc import compose as compose_mod
    from rai_bvar.engines.mcmc import sweep as sweep_mod

    compose_mod.clear_cache()
    sweep_mod.clear_cache()
    _make_runner_cached.cache_clear()
    _sharded.cache_clear()


# --------------------------------------------------------------------------- devices (§5.2)


@functools.lru_cache(maxsize=16)
def _sharded(
    bodies: ChainBodies, D: int, chains: ChainBatching = "vmap", reps: int = 1
) -> tuple[Callable[..., Any], Callable[..., Any], Any, Any]:
    """``burn`` and ``sample`` under ``shard_map`` over ``D`` devices (DESIGN §5.2; equal to the
    one-device executables to rounding with the production kernels), and the chain-sharded and
    replicated ``NamedSharding``s. ``reps``: in-loop forecast paths per selected draw (§5.8)."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((D,), ("chain",))
    in_loop = bodies.forecast is not None
    fc_axes = _CHAIN_AXES["forecast"] if in_loop else ()
    burn_v = chain_map(bodies.burn, _CHAIN_AXES["burn"], chains)
    sample_v = chain_map(
        functools.partial(bodies.sample, reps=reps), (*_CHAIN_AXES["sample"], *fc_axes), chains
    )
    c, r = P("chain"), P()
    fc_specs = (c, r, r, r) if in_loop else ()
    burn_s = jax.shard_map(
        burn_v, mesh=mesh, in_specs=(c, c, r, r, r, r, r), out_specs=c, check_vma=False
    )
    sample_s = jax.shard_map(
        sample_v, mesh=mesh, in_specs=(c, c, r, r, r, r, r, r, *fc_specs), out_specs=(c, c, c),
        check_vma=False,
    )  # fmt: skip

    def mcmc_burn_sharded(ckeys, state, data, prior, cache, it0, n):
        return burn_s(ckeys, state, data, prior, cache, it0, n)

    def mcmc_sample_chunk_sharded(ckeys, state, data, prior, cache, it0, thin, stop, *forecast):
        return sample_s(ckeys, state, data, prior, cache, it0, thin, stop, *forecast)

    return (
        jax.jit(mcmc_burn_sharded, donate_argnums=(1,)),
        jax.jit(mcmc_sample_chunk_sharded, donate_argnums=(1,)),
        NamedSharding(mesh, c),
        NamedSharding(mesh, r),
    )


def resolve_devices(devices: int | str, chains: int, fit_axis: bool = False) -> int:
    """Number of devices the chains run on (DESIGN §5.2). ``Runtime.devices`` defaults to ``1``
    (Stage 2 amendment: sharded draws equal one-device draws only to rounding, so sharding is
    opt-in). ``"auto"`` uses every local device when there are at least two and they divide
    ``chains``; an integer ``D`` must divide ``chains`` and falls back to one device (with a
    warning) when fewer than ``D`` devices exist; a fit axis always runs on one device."""
    avail = jax.local_device_count()
    if devices == "auto":
        D = avail if avail >= 2 and chains % avail == 0 else 1
    else:
        D = int(devices)
        if D > avail:
            warnings.warn(
                f"Runtime(devices={D}) but only {avail} local device(s) exist; running on one "
                "device (call rai_bvar.set_cpu_devices(k) before JAX initialises)",
                UserWarning,
                stacklevel=3,
            )
            D = 1
        elif chains % D:
            raise ValueError(f"chains ({chains}) must be a multiple of Runtime.devices ({D})")
    if fit_axis and D > 1:
        warnings.warn(
            "a fit axis runs on one device; ignoring Runtime.devices", UserWarning, stacklevel=3
        )
        D = 1
    return D


# --------------------------------------------------------------------------- host driver


@dataclass
class RunOutput:
    """Everything one run produces on the host (DESIGN §5.1 item 5)."""

    store: HostStore
    final_state: ChainState  # host NumPy, leading (C,) or (F, C): snap of the last sample call
    acceptance: dict[str, np.ndarray]  # acc_* / (draws·thin); acc_rej_run_max raw
    latent: dict[str, np.ndarray]  # h_mean, h_sd, p_outlier, other rb_* means
    min_pivot: np.ndarray
    seed: Any
    salt: Any
    chains: int
    burn: int
    draws: int
    thin: int
    chunk: int
    warm: bool
    devices: int
    chain_method: str
    plan: StoragePlan
    timing: dict[str, float] = field(default_factory=dict)
    forecast: dict[str, Any] | None = None
    """In-loop forecast plan (DESIGN §5.8): ``horizon``, ``draws``, ``seed``, ``fseed``,
    ``per_chain`` (selected draws per chain), ``paths_per_draw`` and ``index`` (their kept-draw
    indices, :func:`storage.forecast_selection`)."""

    @property
    def sweeps(self) -> int:
        """Sweeps run per chain: ``burn + draws·thin``."""
        return self.burn + self.draws * self.thin


class _CompileTimer:
    """Seconds spent tracing, lowering and compiling while active (``jax.monitoring``)."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.functions: list[str] = []

    def _listener(self, event: str, duration: float, **kwargs: Any) -> None:
        if event.startswith("/jax/core/compile/"):
            self.seconds += float(duration)
            if event.endswith("backend_compile_duration"):
                self.functions.append(str(kwargs.get("fun_name", "")))

    def __enter__(self) -> _CompileTimer:
        jax.monitoring.register_event_duration_secs_listener(self._listener)
        return self

    def __exit__(self, *exc: Any) -> None:
        jax.monitoring.unregister_event_duration_listener(self._listener)


def _as_seq(value: Any, F: int, name: str) -> list[int]:
    if isinstance(value, int | np.integer):
        return [int(value)] * F
    out = [int(v) for v in value]
    if len(out) != F:
        raise ValueError(f"{name} needs {F} values (one per fit), got {len(out)}")
    return out


def _host(tree: Any) -> Any:
    return jax.tree.map(np.asarray, jax.device_get(tree))


def _index(tree: Any, idx: tuple[int, ...]) -> Any:
    return jax.tree.map(lambda x: x[idx], tree)


def _u32(x: int) -> np.ndarray:
    return np.asarray(x, dtype=np.uint32)


def _i32(x: int) -> np.ndarray:
    return np.asarray(x, dtype=np.int32)


@dataclass
class _Run:
    """What the host checks of one ``run_chains`` call need (keeps the driver readable)."""

    fns: RunnerFns
    static: StaticConfig
    lead: tuple[int, ...]
    T_eff: list[int]
    single: tuple[Any, Any, Any, Any]  # (ckeys, data, prior, cache) on one device, for replay

    def check(self, host_state: ChainState, start: ChainState | None, it0: int, n: int) -> None:
        """Status codes and state invariants after a call that ran sweeps ``it0..it0+n−1``."""
        codes = np.asarray(host_state.status.code)
        if (codes != 0).any():
            its = np.asarray(host_state.status.it).astype(np.int64)
            cand = np.argwhere(codes != 0)
            order = sorted(map(tuple, cand), key=lambda ix: (int(its[ix]), ix))
            idx = order[0]
            raise self.failure(int(codes[idx]), int(its[idx]), idx, start, it0, host_state)
        bad = self.violations(host_state)
        if bad:
            raise self.nonfinite(bad, start, it0, n, host_state)

    def violations(self, host_state: ChainState) -> list[str]:
        if not self.fns.fit_axis:
            return check_state_invariants(host_state, self.static, self.T_eff[0])
        out = []
        for f, T in enumerate(self.T_eff):
            out += [
                f"fit {f}: {v}"
                for v in check_state_invariants(_index(host_state, (f,)), self.static, T)
            ]
        return out

    def _fit(self, idx: tuple[int, ...]) -> tuple[Any, Any, Any, Any]:
        """Data, prior, cache and chain key of the failing lane (host trees)."""
        ckeys, data, prior, cache = self.single
        if self.fns.fit_axis:
            f = idx[0]
            return (
                _index(_host(data), (f,)),
                _index(_host(prior), (f,)),
                _index(_host(cache), (f,)),
                ckeys[idx],
            )
        return _host(data), _host(prior), _host(cache), ckeys[idx]

    def replay_to(self, start: ChainState, it0: int, n: int) -> ChainState:
        """Deterministic replay: the state after sweeps ``it0..it0+n−1`` from ``start`` (host)."""
        ckeys, data, prior, cache = self.single
        st = self.fns.burn(ckeys, jax.device_put(start), data, prior, cache, _u32(it0), _i32(n))
        return _host(st)

    def failure(
        self, code: int, it_fail: int, idx: tuple[int, ...], start: ChainState | None, it0: int,
        state: ChainState,
    ) -> MCMCNumericalError:  # fmt: skip
        block, eq = decode_error(code)
        chain = int(idx[-1])
        detail: dict[str, Any] = {
            "code": code,
            "min_pivot": float(np.asarray(state.status.min_pivot)[idx]),
        }
        if self.fns.fit_axis:
            detail["fit"] = int(idx[0])
        is_global = block >= Block.G_CCCM  # global steps record equation 0 (one StepInfo)
        if start is None:
            detail["replay"] = "disabled (Runtime.replay=False)"
            if is_global:
                detail["equation_located"] = False
        else:
            before = _index(self.replay_to(start, it0, it_fail - it0), idx)
            data, prior, cache, ckey = self._fit(idx)
            try:
                detail.update(
                    failure_detail(
                        self.fns.program, before, data, prior, cache, ckey, it_fail, block, eq
                    )
                )
            except Exception as err:  # diagnostics must never hide the failure itself
                detail["replay_error"] = f"{type(err).__name__}: {err}"
            if is_global and detail.get("equation_located"):
                eq = int(detail["equation"])
        return MCMCNumericalError(block.name, eq, it_fail, detail, chain=chain)

    def nonfinite(
        self, bad: list[str], start: ChainState | None, it0: int, n: int, state: ChainState
    ) -> MCMCNumericalError:
        """Invariant violation without a status code: bisect the replay for the first offending
        sweep (with ``Runtime.replay=False`` the call's last sweep is reported)."""
        nlead = len(self.lead)
        if n == 0:  # the initial (cold or warm) state itself
            return MCMCNumericalError(
                "init", _first_bad_equation(state, nlead), 0, {"violations": bad[:10]}
            )
        it_fail = it0 + n - 1
        if start is not None and n > 1:
            lo, hi = 1, n  # smallest m such that m sweeps from start violate the invariants
            while lo < hi:
                mid = (lo + hi) // 2
                if self.violations(self.replay_to(start, it0, mid)):
                    hi = mid
                else:
                    lo = mid + 1
            it_fail = it0 + lo - 1
            state = self.replay_to(start, it0, lo)
            bad = self.violations(state)
        return MCMCNumericalError(
            "state", _first_bad_equation(state, nlead), it_fail, {"violations": bad[:10]}
        )


def _first_bad_equation(st: ChainState, nlead: int) -> int:
    best = -1
    for path, leaf in jax.tree_util.tree_flatten_with_path(st.eq)[0]:
        arr = np.asarray(leaf)
        if not np.issubdtype(arr.dtype, np.floating) or arr.ndim <= nlead:
            continue
        bad = ~np.isfinite(arr)
        if bad.any():
            per_eq = bad.reshape(*arr.shape[: nlead + 1], -1).any(axis=-1)
            eqs = np.nonzero(per_eq.reshape(-1, arr.shape[nlead]).any(axis=0))[0]
            if eqs.size and (best < 0 or eqs[0] < best):
                best = int(eqs[0])
        del path
    return best


def _accumulators(final: ChainState, sweeps: int) -> tuple[dict, dict]:
    """Acceptance rates (``acc_* / sweeps``) and latent moments (``rb_* / sweeps``)."""
    acc: dict[str, np.ndarray] = {}
    rb: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(final)[0]:
        name = getattr(path[-1], "name", None)
        if not isinstance(name, str):
            continue
        arr = np.asarray(leaf)
        if name.startswith("acc_"):
            key = name if name not in acc else leaf_path(path)
            if name == "acc_rej_run":
                continue
            acc[key] = arr if name == "acc_rej_run_max" else arr / sweeps
        elif name.startswith("rb_"):
            rb[name if name not in rb else leaf_path(path)] = arr / sweeps
    latent: dict[str, np.ndarray] = {}
    if "rb_h_sum" in rb:
        mean = rb.pop("rb_h_sum")
        latent["h_mean"] = mean
        if "rb_h_sq" in rb:
            latent["h_sd"] = np.sqrt(np.maximum(rb.pop("rb_h_sq") - mean**2, 0.0))
    if "rb_out" in rb:
        latent["p_outlier"] = rb.pop("rb_out")
    latent.update({f"{k}_mean": v for k, v in rb.items()})
    return acc, latent


def run_chains(
    fns: RunnerFns,
    static: StaticConfig,
    data: EngineData,
    prior: PriorArrays,
    settings: Any,
    runtime: Any,
    *,
    seed: int | Sequence[int],
    salt: int | Sequence[int],
    init_state: ChainState | None = None,
    plan: Any = None,
) -> RunOutput:
    """Run ``settings.chains`` chains of the program behind ``fns`` (DESIGN §5.1 host driver).

    ``init_state`` (host or device, leading ``(C,)``) is a warm start: it replaces the cold
    initialisation and ``settings.resolved_warm_burn`` replaces ``settings.burn`` (DESIGN §5.5).
    With ``fns.fit_axis`` the data, prior, ``seed`` and ``salt`` carry a leading fit axis (sequences
    of length ``F`` for the last two). ``plan`` (a ``FitPlan``; keyword addition to the DESIGN
    signature) is required exactly when ``static.store.in_loop_forecast > 0`` and must have that
    horizon (DESIGN §5.8; every fit of a fit axis uses the same plan). Raises
    ``MCMCNumericalError`` on a numerical failure and ``StorageBudgetError`` before sampling when
    the stored draws exceed the memory budget.
    """
    t_start = time.perf_counter()
    # one OpenBLAS thread for jaxlib's LAPACK calls while the chains run: oversubscription
    # removed; results unchanged below a 128 × 128 factor (engines.mcmc._blas; WP24)
    with _jax.x64(), _blas.single_thread(), _CompileTimer() as ct:
        if fns.program is None or fns.chain is None:
            raise TypeError("fns must come from make_runner")
        if static != fns.program.static:
            raise ValueError(
                "static does not match the StaticConfig the executables were built for"
            )
        if (settings.store == "paths") != static.store.paths:
            raise ValueError(
                f"MCMCSettings.store={settings.store!r} but StaticConfig.store.paths="
                f"{static.store.paths}: the stored streams are compiled into the sample executable"
            )
        if runtime.chunk != static.chunk:
            raise ValueError(
                f"Runtime.chunk ({runtime.chunk}) must equal StaticConfig.chunk ({static.chunk}): "
                "the chunk length is compiled into the sample executable"
            )
        C, draws, thin, chunk = settings.chains, settings.draws, settings.thin, static.chunk
        warm = init_state is not None
        burn = settings.resolved_warm_burn if warm else settings.burn
        n_chunks = -(-draws // chunk)
        if burn + n_chunks * chunk * thin >= _U32:
            raise ValueError(
                f"burn + ⌈draws/chunk⌉·chunk·thin = {burn + n_chunks * chunk * thin} exceeds the "
                "uint32 sweep counter (DESIGN §5.6)"
            )
        H = static.store.in_loop_forecast
        fc_meta: dict[str, Any] | None = None
        if H:
            if plan is None:
                raise ValueError(
                    f"StaticConfig.store.in_loop_forecast={H} needs plan=FitPlan(horizon={H}, …) "
                    "(DESIGN §5.8)"
                )
            if int(plan.horizon) != H:
                raise ValueError(
                    f"FitPlan.horizon ({plan.horizon}) must equal StaticConfig.store."
                    f"in_loop_forecast ({H}): the horizon is compiled into the sample executable"
                )
            fseed = forecast_seed(plan.seed)
            st = settings.store_thin
            stored = np.arange(-(-draws // st), dtype=np.int64) * st  # the params stream (§5.3)
            index, reps = forecast_selection(C, stored, int(plan.draws))
            if reps * draws > _U32:
                raise ValueError(
                    f"FitPlan.draws ({plan.draws}) needs {reps} paths per kept draw: the forecast "
                    f"key index d + q·draws exceeds the uint32 counter (DESIGN §5.6)"
                )
            fc_meta = {
                "horizon": H, "draws": int(plan.draws), "seed": int(plan.seed), "fseed": fseed,
                "per_chain": int(index.size), "paths_per_draw": int(reps), "index": index,
            }  # fmt: skip
        elif plan is not None:
            raise ValueError(
                "a FitPlan needs StaticConfig.store.in_loop_forecast = FitPlan.horizon (DESIGN §5.8)"
            )
        if fns.fit_axis:
            F = int(np.shape(data.T_eff)[0])
            seeds, salts = _as_seq(seed, F, "seed"), _as_seq(salt, F, "salt")
            ckeys = jnp.stack(
                [keys.chain_keys(keys.fit_root(s, z), C) for s, z in zip(seeds, salts, strict=True)]
            )
            ikeys = jnp.stack([keys.init_keys(s, z, C) for s, z in zip(seeds, salts, strict=True)])
            lead: tuple[int, ...] = (F, C)
            T_eff = [int(t) for t in np.asarray(data.T_eff)]
        else:
            ckeys = keys.chain_keys(keys.fit_root(int(seed), int(salt)), C)
            ikeys = keys.init_keys(int(seed), int(salt), C)
            lead = (C,)
            T_eff = [int(np.asarray(data.T_eff))]
        fkeys = None
        if fc_meta is not None:
            fkeys = forecast_chain_keys(fc_meta["fseed"], C)
            if fns.fit_axis:
                fkeys = jnp.stack([fkeys] * lead[0])
        D = resolve_devices(runtime.devices, C, fns.fit_axis)

        data = jax.tree.map(jnp.asarray, data)
        prior = jax.tree.map(jnp.asarray, prior)
        t0 = time.perf_counter()
        cache = fns.prepare(data, prior)
        if warm:
            state = jax.device_put(init_state)
        else:
            state = fns.init(ikeys, data, prior, cache)
        host_state = _host(state)
        t_init = time.perf_counter() - t0

        # storage plan before any sampling (DESIGN §5.3)
        one = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[len(lead) :], x.dtype), host_state
        )
        if fns.fit_axis:
            d1, p1, c1 = (jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), t)
                          for t in (data, prior, cache))  # fmt: skip
        else:
            d1, p1, c1 = data, prior, cache
        record = jax.eval_shape(fns.chain.extract, one, d1, p1, c1)
        if fkeys is not None:  # the in-loop forecast record of one chain and one draw (§5.8)
            fkey1 = fkeys[(0,) * len(lead)]
            fc = jax.eval_shape(fns.chain.forecast, record[1], d1, fkey1, np.uint32(0))
            record = (*record, fc)
        splan = plan_storage(static, settings, runtime, record=record, lead=lead, plan=plan)
        store = HostStore.allocate(splan, static)

        single = (ckeys, data, prior, cache)
        reps = 1 if fc_meta is None else fc_meta["paths_per_draw"]
        if fc_meta is not None:  # the plan and the run select the same draws (§7.1)
            assert np.array_equal(splan.streams["forecast"].index, fc_meta["index"])
        burn_fn, place = fns.burn, (lambda x: x)
        sample_fn = fns.sample if reps == 1 else functools.partial(fns.sample, reps=reps)
        chain_method = "vmap" if fns.chains == "vmap" else "map"
        if D > 1:
            burn_fn, sample_fn, chain_sh, repl = _sharded(fns.chain, D, fns.chains, reps)
            chain_method = f"shard_map({D})" + ("" if fns.chains == "vmap" else "+map")
            ckeys = jax.device_put(ckeys, chain_sh)
            if fkeys is not None:
                fkeys = jax.device_put(fkeys, chain_sh)
            data, prior, cache = (jax.device_put(t, repl) for t in (data, prior, cache))

            def place(x: Any, _s: Any = chain_sh) -> Any:
                return jax.device_put(x, _s)

        run = _Run(fns, static, lead, T_eff, single)
        run.check(host_state, host_state, 0, 0)
        state = place(state)

        # burn-in: exact count, pieces of min(remaining, chunk·thin) sweeps (D-27)
        t0 = time.perf_counter()
        it = 0
        while it < burn:
            m = min(burn - it, chunk * thin)
            start = host_state if runtime.replay else None
            state = burn_fn(ckeys, state, data, prior, cache, _u32(it), _i32(m))
            host_state = _host(state)
            run.check(host_state, start, it, m)
            it += m
        t_burn = time.perf_counter() - t0

        # boundary: every acc_* / rb_* leaf to zero (D-27, E8)
        state = place(zero_accumulators(state))
        host_state = zero_accumulators(host_state)

        t0 = time.perf_counter()
        t_host = 0.0
        compile_before = ct.seconds
        snap_host = host_state
        if fc_meta is not None:  # the §7.1 selection as a mask over kept draws (§5.8)
            selected = np.zeros(n_chunks * chunk, dtype=bool)
            selected[fc_meta["index"]] = True
        for c in range(n_chunks):
            it0 = burn + c * chunk * thin
            stop = min(chunk, draws - c * chunk) - 1
            start = host_state if runtime.replay else None
            fc_args = () if fc_meta is None else (
                fkeys, _u32(c * chunk), selected[c * chunk : (c + 1) * chunk], _u32(draws)
            )  # fmt: skip
            state, snap, out = jax.block_until_ready(
                sample_fn(
                    ckeys, state, data, prior, cache, _u32(it0), _u32(thin), _i32(stop), *fc_args
                )
            )
            th = time.perf_counter()
            out_host = jax.device_get(out)
            last = c == n_chunks - 1
            host_state = _host(state)
            snap_host = _host(snap) if last else host_state
            run.check(snap_host, start, it0, (stop + 1) * thin)
            store.append(out_host, stop)
            t_host += time.perf_counter() - th
        t_sample = time.perf_counter() - t0
        sample_compile = ct.seconds - compile_before

        th = time.perf_counter()
        acc, latent = _accumulators(snap_host, draws * thin)
        store.finalize(
            acceptance=acc,
            latent=latent,
            min_pivot=np.asarray(snap_host.status.min_pivot),
            T_eff=T_eff[0] if not fns.fit_axis else None,
            attrs={
                "chains": C, "draws": draws, "burn": burn, "thin": thin, "warm_start": int(warm),
                "devices": D, "chain_method": chain_method, "chunk": chunk,
                "it_first": 0, "it_last": burn + draws * thin - 1,
                **({} if fc_meta is None else {
                    f"in_loop_forecast_{k}": v for k, v in fc_meta.items() if k != "index"
                }),
            },
        )  # fmt: skip
        t_host += time.perf_counter() - th
    total = time.perf_counter() - t_start
    sweeps_sampled = max(draws * thin, 1)
    timing = {
        "compile": ct.seconds,
        "init": t_init,
        "burn": t_burn,
        "sample": t_sample,
        "host": t_host,
        "total": total,
        "ms_per_sweep": 1000.0 * max(t_sample - t_host - sample_compile, 0.0) / sweeps_sampled,
    }
    return RunOutput(
        store=store, final_state=snap_host, acceptance=acc, latent=latent,
        min_pivot=np.asarray(snap_host.status.min_pivot), seed=seed, salt=salt, chains=C,
        burn=burn, draws=draws, thin=thin, chunk=chunk, warm=warm, devices=D,
        chain_method=chain_method, plan=splan, timing=timing, forecast=fc_meta,
    )  # fmt: skip


# --------------------------------------------------------------------------- replay diagnostics


def _gaussian_detail(K: Any) -> dict[str, Any]:
    """Conditioning of a precision matrix: lower bound ``max/min diag(L)²`` on ``cond₂`` of the
    Jacobi-scaled ``sKs`` (N05 §7.3), smallest eigenvalue, non-positive diagonal entries."""
    K = np.asarray(K, dtype=np.float64)
    out: dict[str, Any] = {"dimension": int(K.shape[0])}
    finite = np.isfinite(K)
    out["nonfinite_entries"] = int((~finite).sum())
    d = np.diag(K)
    out["nonpositive_diagonal"] = int((d <= 0).sum())
    if not finite.all() or out["nonpositive_diagonal"]:
        out["cholesky_ok"] = False
        out["diag_ratio"] = float("inf")
        if finite.all():
            out["min_eigenvalue"] = float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())
        return out
    s = 1.0 / np.sqrt(d)
    Ks = s[:, None] * K * s[None, :]
    Ks = 0.5 * (Ks + Ks.T)
    out["min_eigenvalue_scaled"] = float(np.linalg.eigvalsh(Ks).min())
    try:
        L = np.linalg.cholesky(Ks)
        piv = np.diag(L) ** 2
        out["cholesky_ok"] = True
        out["diag_ratio"] = float(piv.max() / piv.min())
    except np.linalg.LinAlgError:
        out["cholesky_ok"] = False
        out["diag_ratio"] = float("inf")
    return out


def _range(x: Any, mask: Any | None = None) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if mask is not None:
        x = x[np.asarray(mask) > 0]
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(x)), float(np.max(x)))


def failure_detail(
    program: Program,
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    cache: FitCache,
    ckey: Array,
    it: int,
    block: Block,
    eq: int,
) -> dict[str, Any]:
    """Diagnostics of a failure at sweep ``it`` (DESIGN §5.7), evaluated eagerly from ``st``, the
    one-chain state just before that sweep. The steps of sweep ``it`` before the failing one are
    re-run (with the sweep's own keys), then the failing block's exposed conditional is evaluated:
    for an equation step ``conditional_eq`` / ``contemp_conditional`` of equation ``eq``; for a
    global step its ``conditional_j`` for every ``j`` (:func:`_global_detail`, which also locates
    the failing equation: global steps record equation 0). Reported: the diagonal-ratio bound on
    ``cond₂``, Cholesky status, the weight range, the prior precision range."""
    with _jax.x64():
        it_u = _u32(it)
        plan = build_plan(program)
        ctx_g = SweepCtx(data, prior, cache, Empty())
        for g, step in zip(program.global_pre, plan.pre, strict=True):
            if step.block == block:
                return _global_detail(program, g, step, st, ctx_g)
            st, _ = g.sample(keys.step_key(ckey, it_u, step.block), st, ctx_g, program.eq_map)
        sh = sweep_values(program, ckey, it_u, st, data, prior, cache)
        if any(step.block == block for step in plan.equation):
            return _equation_detail(program, plan, sh, st, data, prior, cache, block, eq)
        new_eq, eq_out, _ = equation_phase(program, plan.equation, sh, st, prior, cache, True)
        st = dataclasses.replace(st, eq=new_eq)
        ctx_g = dataclasses.replace(ctx_g, eq_out=eq_out)
        for g, step in zip(program.global_post, plan.post, strict=True):
            if step.block == block:
                return _global_detail(program, g, step, st, ctx_g)
            st, _ = g.sample(keys.step_key(ckey, it_u, step.block), st, ctx_g, program.eq_map)
        return {"step": f"unknown block {block.name}"}


def _equation_detail(
    program: Program, plan: Plan, sh: Any, st: ChainState, data: EngineData,
    prior: PriorArrays, cache: FitCache, block: Block, eq: int,
) -> dict[str, Any]:  # fmt: skip
    """Equation ``eq``: its steps up to the failing one, then that step's exposed conditional."""
    out: dict[str, Any] = {}
    i = jnp.int32(eq)
    eq_i = _index(st.eq, (eq,))
    prior_i = _index(prior.eq, (eq,))
    cache_i = _index(cache.eq, (eq,))
    env = empty_env(None)
    for step in plan.equation:
        if step.block == block and step.name != "residual":
            ctx = program.structure.context(i, sh.ytil, eq_i, program.vol, data, sh.glob.vol)
            out["step"] = step.name
            out["weight_range"] = _range(ctx.weights, ctx.tmask)
            cm = program.structure.contemp_moments(ctx, prior_i)
            pm = program.coef.moments_eq(eq_i.coef, i, prior_i, sh.glob.coef)
            out["prior_precision_range"] = _range(
                np.concatenate([np.ravel(pm.prec), np.ravel(cm.prec)])
            )
            if step.name == "equation":
                gp = program.mean.conditional_eq(
                    eq_i.mean, eq_i.struct.a, ctx, pm, cm, cache_i, cache, prior.shared
                )
                out.update(_gaussian_detail(gp.K))
            elif step.name == "mean":
                a0 = eq_i.struct.a
                gp = program.mean.conditional_eq(
                    eq_i.mean, a0, mean_context(ctx, a0), pm, None, cache_i, cache, prior.shared
                )
                out.update(_gaussian_detail(gp.K))
            elif step.name == "contemp":
                g = env["g"]
                if g is None:
                    a0 = eq_i.struct.a
                    g = program.mean.fitted_eq(
                        eq_i.mean, a0, mean_context(ctx, a0), cache, prior.shared
                    )
                gp = program.structure.contemp_conditional(ctx, g, cm)
                out.update(_gaussian_detail(gp.K))
            elif step.name == "vol":
                out["residual_range"] = _range(env["e"], ctx.tmask)
            return out
        eq_i, env = EQUATION_STEPS[step.name](program, sh, i, eq_i, prior_i, cache_i, env)
    out["step"] = "unknown"
    return out


def _global_detail(
    program: Program, g: Any, step: Step, st: ChainState, ctx: SweepCtx
) -> dict[str, Any]:
    """A global step at the state just before it (DESIGN §5.7). With an exposed
    ``conditional_j(st, ctx, j, eq_map) -> GaussianPrec`` (the CCCM step: the exact conditional of
    ``π_j``; its precision ``K_j`` does not depend on ``Π``, so the in-step column updates do not
    change it), every ``j`` is evaluated and the first ``j`` whose ``K_j`` fails the Cholesky
    test (or whose ``b_j`` is not finite) is reported as ``equation`` with its conditioning
    (``equation_located=True``); if none fails, the worst-conditioned ``j`` is reported
    (``equation_located=False``). ``weight_range``: ``m_t exp(−log λ_eff,i,t)`` over real rows
    (``inv_lam`` of the step); ``prior_precision_range``: the step's prior moments."""
    out: dict[str, Any] = {"step": step.name}
    data, prior, eq_map = ctx.data, ctx.prior, program.eq_map
    moments = getattr(g, "moments", None)
    prec = moments(st, prior, eq_map).prec if callable(moments) else prior.eq.coef.prec0
    out["prior_precision_range"] = _range(prec)
    inv_lam = getattr(g, "inv_lam", None)
    if callable(inv_lam):
        w = np.asarray(inv_lam(st, data.tmask, eq_map))
        out["weight_range"] = _range(w, np.broadcast_to(np.asarray(data.tmask), w.shape))
    cond = getattr(g, "conditional_j", None)
    if not callable(cond):
        out["conditional"] = "not exposed by the step"
        out["equation_located"] = False
        return out
    per = []
    for j in range(program.static.shapes.n):
        gp = cond(st, ctx, j, eq_map)
        d = _gaussian_detail(gp.K)
        d["b_finite"] = bool(np.isfinite(np.asarray(gp.b)).all())
        per.append(d)
    bad = [j for j, d in enumerate(per) if not d["cholesky_ok"] or not d["b_finite"]]
    j = bad[0] if bad else int(np.argmax([d["diag_ratio"] for d in per]))
    out["equation"] = j
    out["equation_located"] = bool(bad)
    out.update(per[j])
    out["diag_ratio_by_equation"] = [d["diag_ratio"] for d in per]
    return out
