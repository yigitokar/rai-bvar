"""The fixed Gibbs sweep skeleton (DESIGN §4.5): ``make_sweep(program)`` executes the plan.

One sweep of one chain, at absolute sweep index ``it`` (uint32, burn-in included; D-16)::

    pre phase     for g in program.global_pre:  st, info = g.sample(step_key(ckey, it, g.block), …)
    residuals     ỹ = structure.pre_residual(glob, data, mean.fitted_reduced(glob, data))
    equations     eq_map over i of the plan's equation steps (keys eq_key(step_key(ckey, it, B), i)):
                    equation | mean, contemp → residual → [coef_local] → vol
    post phase    for g in program.global_post: … with SweepCtx.eq_out = (e, log λ_eff)

Every equation step rebuilds its ``EqContext`` from the state at the time it runs
(``structure.context``: weights ``m_t·exp(−log λ_eff)`` from the *current* volatility state), so
the steps are pure functions of ``(state, env)`` and the plan can run **fused** (all equation steps
in one ``eq_map`` body; the production default) or **unfused** (one ``eq_map`` per step, the
intermediate ``env`` — fitted mean ``g``, residual ``e``, ``Coupling``, step infos — batched over
equations). Both give bitwise-identical draws (DESIGN §9.3 item 4); the unfused executor is also
what :func:`trace_steps` uses for the write-set tree diff of §4.8 item 4.

Failures are values (D-18): every block returns ``StepInfo``; the sweep keeps the **first**
``(code, it)`` per chain with ``code = 1 + 1000·block + eq`` (plan order, then the lowest failing
equation) and the smallest pivot seen. Nothing raises inside JIT and nothing is jittered.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rai_bvar.engines.mcmc import (
    _jax,  # noqa: F401  (enables float64 before any computation)
    keys,
)
from rai_bvar.engines.mcmc.compose import Program, compose
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.plan import Plan, Step, build_plan, check_invariants
from rai_bvar.engines.mcmc.state import (
    ChainState,
    Coupling,
    Empty,
    EngineData,
    EqOut,
    FitCache,
    PriorArrays,
    Status,
    StepInfo,
    SweepCtx,
    encode_error,
    leaf_path,
)

Array = jax.Array

__all__ = [
    "EQUATION_STEPS",
    "StepTrace",
    "clear_cache",
    "empty_env",
    "equation_phase",
    "make_sweep",
    "mean_context",
    "record_eq_status",
    "record_status",
    "sweep_values",
    "trace_steps",
]

_HIGHEST = lax.Precision.HIGHEST

Env = dict[str, Any]  # g, e, lv: f64[T_pad] | None; cpl: Coupling; infos: {step name: StepInfo}


@dataclass(frozen=True)
class _Sweep:
    """Values shared by every equation of one sweep (traced; closed over by the body)."""

    ckey: Array
    it: Array
    ytil: Array
    glob: Any
    data: EngineData
    shared: Any
    cache: FitCache
    T_pad: int


# --------------------------------------------------------------------------- status


def _set_status(st: ChainState, code: Array, it: Array, min_pivot: Array) -> ChainState:
    s = st.status
    first = (s.code == 0) & (code != 0)
    status = Status(
        code=jnp.where(first, code, s.code).astype(jnp.int32),
        it=jnp.where(first, it, s.it).astype(jnp.uint32),
        min_pivot=jnp.asarray(min_pivot, jnp.float64),
    )
    return dataclasses.replace(st, status=status)


def record_status(st: ChainState, info: StepInfo, block: Block, eq: Any, it: Array) -> ChainState:
    """Keep the first failure ``(1 + 1000·block + eq, it)`` and the smallest pivot (one step)."""
    ok = jnp.all(jnp.asarray(info.ok, dtype=bool))
    code = jnp.where(ok, jnp.int32(0), encode_error(block, eq))
    piv = jnp.minimum(st.status.min_pivot, jnp.min(jnp.asarray(info.min_pivot, jnp.float64)))
    return _set_status(st, code, it, piv)


def record_eq_status(
    st: ChainState, infos: Mapping[str, StepInfo], steps: tuple[Step, ...], it: Array
) -> ChainState:
    """Equation-phase status: the first failing step in plan order, then its lowest equation."""
    code = jnp.int32(0)
    piv = st.status.min_pivot
    for step in steps:
        info = infos.get(step.name)
        if info is None:
            continue
        bad = ~jnp.asarray(info.ok, dtype=bool)
        eq = jnp.argmax(bad).astype(jnp.int32)
        c = jnp.where(jnp.any(bad), encode_error(step.block, eq), jnp.int32(0))
        code = jnp.where(code == 0, c, code)
        piv = jnp.minimum(piv, jnp.min(jnp.asarray(info.min_pivot, jnp.float64)))
    return _set_status(st, code, it, piv)


# --------------------------------------------------------------------------- equation steps


def _key(sh: _Sweep, block: Block, i: Array) -> Array:
    return keys.eq_key(keys.step_key(sh.ckey, sh.it, block), i)


def _context(program: Program, sh: _Sweep, i: Array, eq_i: Any) -> Any:
    return program.structure.context(i, sh.ytil, eq_i, program.vol, sh.data, sh.glob.vol)


def mean_context(ctx: Any, a: Array) -> Any:
    """Target handed to a non-fusing mean: ``ỹ_i − (cols ⊙ cmask)·a_i`` (DESIGN §4.5)."""
    shift = jnp.matmul(ctx.cols * ctx.cmask, a, precision=_HIGHEST)
    return dataclasses.replace(ctx, target=ctx.target - shift)


def _env_update(env: Env, info: tuple[str, StepInfo] | None = None, **values: Any) -> Env:
    out = dict(env)
    out.update(values)
    if info is not None:
        out["infos"] = {**env["infos"], info[0]: info[1]}
    return out


def _fitted(program: Program, sh: _Sweep, eq_i: Any, ctx: Any, env: Env) -> Array:
    """``g_i``: from an earlier step of this body, else the (non-fusing) mean's current fit."""
    if env["g"] is not None:
        return env["g"]
    a0 = eq_i.struct.a
    return program.mean.fitted_eq(eq_i.mean, a0, mean_context(ctx, a0), sh.cache, sh.shared)


def _step_equation(program, sh, i, eq_i, prior_i, cache_i, env):
    """S1 (structural): the fused ``(δ_i, a_i) | λ_eff,i`` draw (DESIGN §4.2)."""
    ctx = _context(program, sh, i, eq_i)
    cm = program.structure.contemp_moments(ctx, prior_i)
    pm = program.coef.moments_eq(eq_i.coef, i, prior_i, sh.glob.coef)
    ms, a, cpl, info = program.mean.sample_eq(
        _key(sh, Block.EQ_MEAN, i), eq_i.mean, eq_i.struct.a, ctx, pm, cm, cache_i, sh.cache,
        sh.shared,
    )  # fmt: skip
    g = program.mean.fitted_eq(ms, a, ctx, sh.cache, sh.shared)
    eq_i = dataclasses.replace(eq_i, mean=ms, struct=dataclasses.replace(eq_i.struct, a=a))
    return eq_i, _env_update(env, ("equation", info), g=g, cpl=env["cpl"] + cpl)


def _step_mean(program, sh, i, eq_i, prior_i, cache_i, env):
    """A non-fusing mean with state, given ``a_i`` (Phase 4 BART)."""
    ctx = _context(program, sh, i, eq_i)
    pm = program.coef.moments_eq(eq_i.coef, i, prior_i, sh.glob.coef)
    a0 = eq_i.struct.a
    ctx_m = mean_context(ctx, a0)
    ms, _, cpl, info = program.mean.sample_eq(
        _key(sh, Block.EQ_MEAN, i), eq_i.mean, a0, ctx_m, pm, None, cache_i, sh.cache, sh.shared
    )
    g = program.mean.fitted_eq(ms, a0, ctx_m, sh.cache, sh.shared)
    return dataclasses.replace(eq_i, mean=ms), _env_update(
        env, ("mean", info), g=g, cpl=env["cpl"] + cpl
    )


def _step_contemp(program, sh, i, eq_i, prior_i, cache_i, env):
    """``a_i | g_i, λ_eff,i`` by the structure (non-fusing means; DESIGN §4.3)."""
    ctx = _context(program, sh, i, eq_i)
    cm = program.structure.contemp_moments(ctx, prior_i)
    g = _fitted(program, sh, eq_i, ctx, env)
    ss, cpl, info = program.structure.contemp_step(
        _key(sh, Block.EQ_CONTEMP, i), eq_i.struct, ctx, g, cm
    )
    return dataclasses.replace(eq_i, struct=ss), _env_update(
        env, ("contemp", info), g=g, cpl=env["cpl"] + cpl
    )


def _step_residual(program, sh, i, eq_i, prior_i, cache_i, env):
    """``e_i = target − (cols ⊙ cmask)·a_i − g_i`` from the new draws (0 on padded rows)."""
    ctx = _context(program, sh, i, eq_i)
    g = _fitted(program, sh, eq_i, ctx, env)
    e = program.structure.residual(eq_i.struct.a, ctx, g)
    return eq_i, _env_update(env, g=g, e=e)


def _step_coef_local(program, sh, i, eq_i, prior_i, cache_i, env):
    """Local scales of the coefficient prior (identity in Phase 2, so omitted from the plan)."""
    coef_i = program.mean.coefficients_eq(eq_i.mean, eq_i.struct.a, sh.shared)
    cs = program.coef.sample_local_eq(
        _key(sh, Block.EQ_COEF_LOCAL, i), eq_i.coef, coef_i, prior_i, sh.glob.coef
    )
    return dataclasses.replace(eq_i, coef=cs), env


def _step_vol(program, sh, i, eq_i, prior_i, cache_i, env):
    """``VolBlock.sample_eq`` (atomic, DESIGN §4.4) on the residual, with the coupling."""
    vs, info = program.vol.sample_eq(
        _key(sh, Block.EQ_VOL, i), eq_i.vol, env["e"], sh.data.tmask, sh.data.T_eff, prior_i,
        env["cpl"], sh.glob.vol,
    )  # fmt: skip
    lv = jnp.broadcast_to(program.vol.log_var_eq(vs, sh.glob.vol), (sh.T_pad,))
    return dataclasses.replace(eq_i, vol=vs), _env_update(env, ("vol", info), lv=lv)


EQUATION_STEPS: dict[str, Callable[..., tuple[Any, Env]]] = {
    "equation": _step_equation,
    "mean": _step_mean,
    "contemp": _step_contemp,
    "residual": _step_residual,
    "coef_local": _step_coef_local,
    "vol": _step_vol,
}


def empty_env(n: int | None) -> Env:
    """Empty environment: one equation (``n=None``) or batched over ``n`` equations."""
    shape = () if n is None else (n,)
    zeros = Coupling(jnp.zeros(shape), jnp.zeros(shape))
    return {"g": None, "e": None, "lv": None, "cpl": zeros, "infos": {}}


def sweep_values(
    program: Program, ckey: Array, it: Array, st: ChainState, data, prior, cache
) -> _Sweep:
    ytil = program.structure.pre_residual(st.glob, data, program.mean.fitted_reduced(st.glob, data))
    return _Sweep(ckey, it, ytil, st.glob, data, prior.shared, cache, program.static.shapes.T_pad)


def equation_phase(
    program: Program, steps: tuple[Step, ...], sh: _Sweep, st: ChainState, prior, cache, fused: bool
) -> tuple[Any, EqOut, dict[str, StepInfo]]:
    """The equation phase of one sweep: new ``st.eq``, the post phase's ``EqOut`` and the step
    infos (batched over equations); fused or one ``eq_map`` per step (module docstring)."""
    n = program.static.shapes.n
    idx = jnp.arange(n, dtype=jnp.int32)
    fns = [functools.partial(EQUATION_STEPS[s.name], program, sh) for s in steps]
    if fused:

        def body(i, eq_i, prior_i, cache_i):
            env = empty_env(None)
            for fn in fns:
                eq_i, env = fn(i, eq_i, prior_i, cache_i, env)
            return eq_i, env

        eq, env = program.eq_map(body)(idx, st.eq, prior.eq, cache.eq)
    else:
        eq, env = st.eq, empty_env(n)
        for fn in fns:
            eq, env = program.eq_map(fn)(idx, eq, prior.eq, cache.eq, env)
    return eq, EqOut(env["e"], env["lv"]), env["infos"]


def _sweep_fn(program: Program, plan: Plan, fused: bool) -> Callable[..., ChainState]:
    pre = tuple(zip(program.global_pre, plan.pre, strict=True))
    post = tuple(zip(program.global_post, plan.post, strict=True))
    eq_steps = plan.equation

    def sweep(
        ckey: Array,
        it: Array,
        st: ChainState,
        data: EngineData,
        prior: PriorArrays,
        cache: FitCache,
    ) -> ChainState:
        ctx = SweepCtx(data, prior, cache, Empty())
        for g, step in pre:  # static tuple, unrolled at trace time
            st, info = g.sample(keys.step_key(ckey, it, step.block), st, ctx, program.eq_map)
            st = record_status(st, info, step.block, 0, it)
        sh = sweep_values(program, ckey, it, st, data, prior, cache)
        eq, eq_out, infos = equation_phase(program, eq_steps, sh, st, prior, cache, fused)
        st = record_eq_status(dataclasses.replace(st, eq=eq), infos, eq_steps, it)
        ctx = dataclasses.replace(ctx, eq_out=eq_out)
        for g, step in post:
            st, info = g.sample(keys.step_key(ckey, it, step.block), st, ctx, program.eq_map)
            st = record_status(st, info, step.block, 0, it)
        return st

    sweep.plan = plan  # type: ignore[attr-defined]
    sweep.program = program  # type: ignore[attr-defined]
    return sweep


@functools.lru_cache(maxsize=64)
def _make_sweep_cached(program: Program, fused: bool) -> Callable[..., ChainState]:
    plan = build_plan(program)
    check_invariants(plan, program)
    return _sweep_fn(program, plan, fused)


def make_sweep(
    program: Program, *, fused: bool = True, _overrides: Mapping[str, Any] | None = None
) -> Callable[..., ChainState]:
    """``sweep(ckey, it: u32[], st, data, prior, cache) -> ChainState`` for one chain (DESIGN §4.5).

    ``fused=False`` runs every equation step in its own ``eq_map`` (bitwise-identical; tests).
    ``_overrides`` re-composes the program with extra test-seam overrides (DESIGN §4.9). The
    function carries ``.plan`` and ``.program``. Cached on the program unless it has overrides.
    """
    if _overrides:
        merged = {**dict(program.overrides), **dict(_overrides)}
        program = compose(program.static, overrides=merged)
    if program.overridden:
        plan = build_plan(program)
        check_invariants(plan, program)
        return _sweep_fn(program, plan, fused)
    return _make_sweep_cached(program, bool(fused))


def clear_cache() -> None:
    """Drop every cached sweep (tests that swap registry classes)."""
    _make_sweep_cached.cache_clear()


# --------------------------------------------------------------------------- tree diff (§4.8 item 4)


@dataclass(frozen=True)
class StepTrace:
    """What one step did when executed on its own: the leaves it changed and whether it returned
    a non-zero ``Coupling``."""

    step: Step
    modified: frozenset[str]
    coupling_nonzero: bool


def _diff(a: Any, b: Any, prefix: str = "") -> frozenset[str]:
    la = jax.tree_util.tree_flatten_with_path(a)[0]
    lb = jax.tree_util.tree_flatten_with_path(b)[0]
    if [p for p, _ in la] != [p for p, _ in lb]:
        return frozenset({prefix.rstrip(".") or "<structure>"})
    out = set()
    for (path, x), (_, y) in zip(la, lb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=True):
            out.add(prefix + leaf_path(path))
    return frozenset(out)


def trace_steps(
    program: Program,
    plan: Plan,
    ckey: Array,
    it: Array,
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    cache: FitCache,
) -> list[StepTrace]:
    """Execute ``plan`` one step at a time (unfused; each step jitted on its own) from the one-chain
    state ``st`` and report each step's modified leaves and coupling (DESIGN §4.8 items 4–5)."""
    out: list[StepTrace] = []

    def global_step(g: Any, step: Step, st: ChainState, ctx: SweepCtx) -> ChainState:
        key = keys.step_key(ckey, it, step.block)
        return jax.jit(lambda s, c: g.sample(key, s, c, program.eq_map)[0])(st, ctx)

    ctx = SweepCtx(data, prior, cache, Empty())
    for g, step in zip(program.global_pre, plan.pre, strict=True):
        new = global_step(g, step, st, ctx)
        out.append(StepTrace(step, _diff(st, new), False))
        st = new
    n = program.static.shapes.n
    idx = jnp.arange(n, dtype=jnp.int32)
    eq, env = st.eq, empty_env(n)
    for step in plan.equation:

        def run(st_, eq_, env_, name=step.name):
            sh = sweep_values(program, ckey, it, st_, data, prior, cache)
            fn = functools.partial(EQUATION_STEPS[name], program, sh)
            return program.eq_map(fn)(idx, eq_, prior.eq, cache.eq, env_)

        eq_new, env_new = jax.jit(run)(st, eq, env)
        dc = jax.tree.map(
            lambda x, y: np.any(np.asarray(x) != np.asarray(y)), env_new["cpl"], env["cpl"]
        )
        out.append(StepTrace(step, _diff(eq, eq_new, "eq."), any(jax.tree.leaves(dc))))
        eq, env = eq_new, env_new
    st = dataclasses.replace(st, eq=eq)
    ctx = dataclasses.replace(ctx, eq_out=EqOut(env["e"], env["lv"]))
    for g, step in zip(program.global_post, plan.post, strict=True):
        new = global_step(g, step, st, ctx)
        out.append(StepTrace(step, _diff(st, new), False))
        st = new
    return out
