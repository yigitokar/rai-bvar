"""Composition: ``StaticConfig`` → :class:`Program` (DESIGN §3.8, §4.9).

``compose(static)`` looks up the kernel classes of every slot through ``registry.load`` (lazy
import strings; DESIGN §3.8 item 1), instantiates them, checks the capability rules that need the
kernels' class variables (item 2; the JAX-free part already ran in ``config.validate_composition``),
derives the batching and the float32 exemptions (item 3), collects and orders the global steps
(item 4) and returns a frozen, hashable :class:`Program` that carries ``eq_map`` (item 5).

**Construction convention.** Kernels are frozen dataclasses (``protocols.KernelBase``) whose fields
are static choices. A class with a ``from_static(static: StaticConfig, **kw)`` classmethod is built
by it, receiving the program's kernels its signature names (``vol``, ``coef``, ``structure``,
``mean``, ``core``, ``transitory``). Any other class gets every constructor parameter *by name*
from the slot's pool:

=====================  ==========================================================================
every slot             ``static`` (the slot's sub-config: ``StructStatic``, ``MeanStatic``,
                       ``CoefPriorStatic`` or ``VolStatic``), every field of that sub-config by
                       its own name (``contemp_scaling``, ``gram``, ``rw``, ``exact``, …), every
                       field of ``Shapes`` (``n``, ``lags``, ``n_det``, ``k``, ``q``, ``T_pad``),
                       ``shapes`` and ``config`` / ``static_config`` (``StaticConfig``)
structure              ``coef_scaling``
mean                   ``contemp_scaling``, ``constant_weights``
volatility block       ``core`` / ``persistent_core`` and ``transitory`` / ``trans`` (the kernels,
                       built first when the block's constructor names them)
global steps           ``structure``, ``mean``, ``coef``, ``vol`` (the kernels); ``static`` is the
                       ``MeanStatic`` (the CCCM step's gram and precompute choices)
=====================  ==========================================================================

A parameter with a default that the pool does not name keeps its default; a required parameter
the pool cannot fill raises ``TypeError``.

**Test seam** (DESIGN §4.9). ``overrides`` maps names of ``protocols.OVERRIDABLE`` to
replacements: a kernel slot (``structure``, ``mean``, ``coef``, ``vol``) takes a kernel instance or
a class (built by the convention above); ``global_pre`` takes a tuple of global steps (instances or
classes); a name of ``protocols.INTERNALS`` is attached to every kernel of the program through
``KernelBase.with_overrides`` (kernels resolve their internals with ``impl(name)``, so only the
owner of the name ever calls it). Overrides bypass every cache; production code never passes them.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
from jax import lax

from rai_bvar.engines.mcmc import (
    _jax,  # noqa: F401  (enables float64 before any computation)
    registry,
)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.protocols import (
    INTERNALS,
    KERNEL_SLOTS,
    OVERRIDABLE,
    Batching,
    EqMap,
    KernelBase,
)

__all__ = ["Program", "clear_cache", "compose", "eq_map_for", "lax_map_eq"]

_PHASE_ORDER = {"pre": 0, "post": 1}


def lax_map_eq(f: Callable[..., Any]) -> Callable[..., Any]:
    """``Batching.MAP``: equations one after another with ``lax.map`` (non-vmappable kernels)."""

    def mapped(*args: Any) -> Any:
        return lax.map(lambda t: f(*t), args)

    return mapped


def eq_map_for(batching: Batching) -> EqMap:
    """The equation batching of DESIGN §4.5: ``jax.vmap`` (``in_axes=0``) or :func:`lax_map_eq`."""
    return jax.vmap if batching == Batching.VMAP else lax_map_eq


@dataclass(frozen=True)
class Program:
    """A composed sampler: frozen kernels plus the derived static choices (DESIGN §3.8 item 5).

    Hashable by value (kernels are frozen dataclasses; ``StaticConfig`` has value equality), so it
    keys the ``make_sweep`` cache. ``overrides`` records the test-seam replacements it was built
    with (empty in production).
    """

    static: StaticConfig
    structure: Any
    mean: Any
    coef: Any
    vol: Any
    global_pre: tuple[Any, ...]
    global_post: tuple[Any, ...]
    batching: Batching
    float32_leaves: frozenset[str]
    overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def eq_map(self) -> EqMap:
        """Batching of a per-equation function over the leading equation axis (DESIGN §4.5)."""
        return eq_map_for(self.batching)

    @property
    def overridden(self) -> bool:
        return bool(self.overrides)

    @property
    def global_steps(self) -> tuple[Any, ...]:
        return self.global_pre + self.global_post


# --------------------------------------------------------------------------- construction


def _sub_fields(sub: Any) -> dict[str, Any]:
    return {f.name: getattr(sub, f.name) for f in dataclasses.fields(sub)}


def _pool(static: StaticConfig, slot: str) -> dict[str, Any]:
    """Constructor arguments available to the kernels of ``slot`` (module docstring table)."""
    sub = {
        "structure": static.struct,
        "mean": static.mean,
        "coef": static.coef,
        "core": static.vol,
        "transitory": static.vol,
        "vol": static.vol,
        "global": static.mean,
    }[slot]
    pool: dict[str, Any] = dict(_sub_fields(static.shapes))
    pool.update(_sub_fields(sub))
    pool.update(static=sub, shapes=static.shapes, config=static, static_config=static)
    if slot == "structure":
        pool["coef_scaling"] = static.mean.coef_scaling
    elif slot == "mean":
        pool["contemp_scaling"] = static.struct.contemp_scaling
        pool["constant_weights"] = static.vol.constant_weights
    elif slot == "global":
        pool["mean_static"] = static.mean
    return pool


def _parameters(cls: type) -> Mapping[str, inspect.Parameter]:
    try:
        return inspect.signature(cls).parameters
    except (TypeError, ValueError):  # no introspectable signature: a no-argument kernel
        return {}


_KERNEL_ARGS = ("structure", "mean", "coef", "vol", "core", "transitory")


def _construct(cls: type, pool: Mapping[str, Any]) -> Any:
    """Instantiate ``cls``: ``cls.from_static(config, **kernels it names)`` when it has that
    classmethod, else every constructor parameter by name from ``pool``."""
    factory = getattr(cls, "from_static", None)
    if callable(factory):
        names = _parameters(factory)
        extra = {
            k: pool[k] for k in _KERNEL_ARGS if k in names and k in pool and not _is_static(pool[k])
        }
        return factory(pool["config"], **extra)
    params = _parameters(cls)
    kwargs: dict[str, Any] = {}
    missing = []
    for name, par in params.items():
        if name == "overrides" or par.kind in (par.VAR_POSITIONAL, par.VAR_KEYWORD):
            continue
        if name in pool:
            kwargs[name] = pool[name]
        elif par.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            f"cannot construct kernel {cls.__module__}.{cls.__qualname__}: no value for "
            f"{missing}; compose fills parameters by name from {sorted(pool)} "
            "(engines/mcmc/compose.py docstring)"
        )
    return cls(**kwargs)


def _is_static(value: Any) -> bool:
    """A configuration object rather than a kernel (the pools also hold sub-configs)."""
    return dataclasses.is_dataclass(value) and type(value).__module__.endswith("mcmc.config")


def _with_internals(kernel: Any, internals: Mapping[str, Callable[..., Any]]) -> Any:
    if internals and isinstance(kernel, KernelBase):
        return kernel.with_overrides(internals)
    return kernel


def _build(kind: str, override: Any, pool: Mapping[str, Any], internals: Mapping[str, Any]) -> Any:
    """The kernel of one slot: the override (instance or class) or the registry class."""
    if override is None:
        kernel = _construct(registry.load(kind), pool)
    elif isinstance(override, type):
        kernel = _construct(override, pool)
    else:
        kernel = override
    return _with_internals(kernel, internals)


def _check_overrides(overrides: Mapping[str, Any]) -> None:
    unknown = sorted(set(overrides) - OVERRIDABLE)
    if unknown:
        raise ValueError(
            f"not overridable: {unknown}; allowed names are protocols.OVERRIDABLE "
            f"(slots {sorted(KERNEL_SLOTS)} and kernel internals)"
        )
    for name, value in overrides.items():
        if name in INTERNALS and not callable(value):
            raise TypeError(f"override of internal {name!r} must be callable, got {value!r}")


def _batching(*kernels: Any) -> Batching:
    return max(
        (Batching(int(getattr(type(k), "batching", Batching.VMAP))) for k in kernels),
        default=Batching.VMAP,
    )


def _check_capabilities(
    static: StaticConfig, structure: Any, mean: Any, coef: Any, vol: Any
) -> None:
    """Rules of DESIGN §3.8 item 2 that read the kernels' class variables."""
    if static.mean.kind == "linear_reduced" and not getattr(
        structure, "supports_reduced_mean", False
    ):
        raise ValueError(
            f"{type(structure).__name__} does not support a reduced-form mean (supports_reduced_mean)"
        )
    if getattr(mean, "fuses_contemp", False) and not getattr(structure, "has_contemp", False):
        raise ValueError(
            f"{type(mean).__name__} fuses the contemporaneous step but "
            f"{type(structure).__name__} has no contemporaneous coefficients"
        )
    path = static.mean.const_path
    if path in ("eigen", "fixed_factor") and getattr(coef, "varies", False):
        raise ValueError(f"const_path={path!r} needs a coefficient prior that does not vary")
    if path not in ("weighted", "shared_factor"):  # shared_factor: common time-varying weights
        constant = getattr(vol, "constant_weights", None)
        if callable(constant) and not constant():
            raise ValueError(f"const_path={path!r} needs a volatility block with constant weights")


def _order_global_steps(steps: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Validate and sort global steps by ``(phase, block)``; duplicate block ids raise."""
    seen: dict[int, str] = {}
    for g in steps:
        name = type(g).__name__
        phase = getattr(g, "phase", None)
        if phase not in _PHASE_ORDER:
            raise ValueError(f"global step {name} has phase {phase!r}; expected 'pre' or 'post'")
        if getattr(g, "marginal_of_indicators", False) is not True:
            raise ValueError(
                f"global step {name} must declare marginal_of_indicators = True (DESIGN §4.8 item 6)"
            )
        block = int(g.block)
        if block in seen:
            raise ValueError(f"duplicate global step block id {block}: {seen[block]} and {name}")
        seen[block] = name
    ordered = sorted(steps, key=lambda g: (_PHASE_ORDER[g.phase], int(g.block)))
    pre = tuple(g for g in ordered if g.phase == "pre")
    post = tuple(g for g in ordered if g.phase == "post")
    return pre, post


def _compose(static: StaticConfig, overrides: Mapping[str, Any]) -> Program:
    if not isinstance(static, StaticConfig):
        raise TypeError(f"compose needs a StaticConfig, got {type(static).__name__}")
    _check_overrides(overrides)
    internals = {k: v for k, v in overrides.items() if k in INTERNALS}
    kinds = registry.slot_kinds(static)

    structure = _build(
        kinds["structure"], overrides.get("structure"), _pool(static, "structure"), internals
    )
    coef = _build(kinds["coef"], overrides.get("coef"), _pool(static, "coef"), internals)
    mean = _build(kinds["mean"], overrides.get("mean"), _pool(static, "mean"), internals)
    vol_override = overrides.get("vol")
    if vol_override is None or isinstance(vol_override, type):
        vol_cls = registry.load(kinds["vol"]) if vol_override is None else vol_override
        vol_pool = _pool(static, "vol")
        wanted = set(_parameters(vol_cls))
        if wanted & {"core", "persistent_core"}:
            core = _build(kinds["core"], None, _pool(static, "core"), internals)
            vol_pool.update(core=core, persistent_core=core)
        if wanted & {"transitory", "trans"}:
            trans = _build(kinds["transitory"], None, _pool(static, "transitory"), internals)
            vol_pool.update(transitory=trans, trans=trans)
        vol = _with_internals(_construct(vol_cls, vol_pool), internals)
    else:
        vol = _with_internals(vol_override, internals)

    gpool = _pool(static, "global")
    gpool.update(structure=structure, mean=mean, coef=coef, vol=vol)
    if "global_pre" in overrides:
        given = overrides["global_pre"]
        given = (given,) if not isinstance(given, tuple | list) else tuple(given)
        pre_steps = tuple(_build("", g, gpool, internals) for g in given)
    else:
        pre_steps = tuple(_build(kind, None, gpool, internals) for kind in kinds["global_pre"])
    coef_steps = tuple(_with_internals(g, internals) for g in coef.global_steps())
    # Phase 3: a volatility block with global state brings its own steps (CommonSV, P3 §2.4)
    vol_global = getattr(vol, "global_steps", None)
    vol_steps = tuple(_with_internals(g, internals) for g in vol_global()) if vol_global else ()
    global_pre, global_post = _order_global_steps(pre_steps + coef_steps + vol_steps)
    if static.vol.common and not any(g.block == Block.G_COMMON_SV for g in global_post):
        # a Phase 2 volatility block would silently ignore the common h (P3 E149)
        raise NotImplementedError(
            "VolStatic(common=True) needs the volatility block's common-volatility step "
            "(G_COMMON_SV), which this build does not provide (CommonSV arrives with P3 WP32)"
        )

    _check_capabilities(static, structure, mean, coef, vol)
    batching = max(
        Batching.MAP if static.batching == "map" else Batching.VMAP,
        _batching(structure, mean, coef, vol, *global_pre, *global_post),
    )
    f32 = set(static.float32_leaves)
    for k in (structure, mean, coef, vol, *global_pre, *global_post):
        f32 |= set(getattr(type(k), "float32_leaves", frozenset()))
    return Program(
        static=static,
        structure=structure,
        mean=mean,
        coef=coef,
        vol=vol,
        global_pre=global_pre,
        global_post=global_post,
        batching=batching,
        float32_leaves=frozenset(f32),
        overrides=tuple(
            sorted(
                ((k, tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()),
                key=lambda kv: kv[0],
            )
        ),
    )


@functools.lru_cache(maxsize=64)
def _compose_cached(static: StaticConfig) -> Program:
    return _compose(static, {})


def compose(static: StaticConfig, *, overrides: Mapping[str, Any] | None = None) -> Program:
    """The :class:`Program` of ``static`` (lru-cached when ``overrides`` is empty; DESIGN §3.8).

    Raises ``NotImplementedError`` when a registered kernel module does not exist yet,
    ``ValueError`` for an illegal composition and ``TypeError`` for a kernel that cannot be built.
    """
    if overrides:
        return _compose(static, dict(overrides))
    return _compose_cached(static)


def clear_cache() -> None:
    """Drop every cached :class:`Program` (tests that swap registry classes)."""
    _compose_cached.cache_clear()
