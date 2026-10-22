"""Kernel registry: kind string → ``"module:Class"`` (DESIGN §3.8 item 1). Pure Python.

Kinds are the values of the ``StaticConfig`` fields that pick a kernel (``StructStatic.kind``,
``MeanStatic.kind``, ``CoefPriorStatic.kind``, ``VolStatic.persistent``,
``VolStatic.transitory``), plus the volatility composition (``"vol_block"``) and the reduced-form
global step (``"cccm"``). The classes live at the listed paths and are imported only by
:func:`load`, so ``import rai_bvar`` and ``api.py`` never load JAX (DESIGN §1.1). Later work
packages implement the classes at these paths and never edit this table; Phase 3+ components
append entries.
"""

from __future__ import annotations

import importlib

from rai_bvar.engines.mcmc.config import StaticConfig

KERNELS: dict[str, str] = {
    # covariance structures (StructStatic.kind)
    "recursive": "rai_bvar.components.structure.recursive:Recursive",
    # conditional means (MeanStatic.kind) and the reduced-form pre-phase step
    "linear_structural": "rai_bvar.components.mean.linear:LinearStructural",
    "linear_reduced": "rai_bvar.components.mean.linear_reduced:ReducedEq",
    "cccm": "rai_bvar.components.mean.linear_reduced:CCCMStep",
    "zero": "rai_bvar.components.mean.zero:ZeroMean",
    # coefficient priors (CoefPriorStatic.kind)
    "fixed_diagonal": "rai_bvar.components.coef.fixed:FixedDiagonal",
    # persistent volatility cores (VolStatic.persistent; SVCore covers AR and RW)
    "constant": "rai_bvar.components.vol.constant:ConstantCore",
    "sv": "rai_bvar.components.vol.sv:SVCore",
    # transitory scales (VolStatic.transitory)
    "none": "rai_bvar.components.vol.transitory:NoTransitory",
    "outlier": "rai_bvar.components.vol.transitory:OutlierScale",
    "t": "rai_bvar.components.vol.transitory:TScale",
    "outlier_t": "rai_bvar.components.vol.transitory:OutlierTScale",
    # the volatility block = transitory ∘ persistent (the only VolBlock in Phase 2)
    "vol_block": "rai_bvar.components.vol.robust:VolBlockImpl",
}

OWNERS: dict[str, str] = {
    "recursive": "WP10",
    "linear_structural": "WP10",
    "linear_reduced": "WP11",
    "cccm": "WP11",
    "zero": "WP10",
    "fixed_diagonal": "WP10",
    "constant": "WP13",
    "sv": "WP12",
    "none": "WP13",
    "outlier": "WP13",
    "t": "WP13",
    "outlier_t": "WP13",
    "vol_block": "WP13",
}


def load(kind: str) -> type:
    """Import and return the kernel class registered for ``kind``.

    Raises ``KeyError`` for an unknown kind and ``NotImplementedError`` when the module or class
    does not exist yet; errors raised while importing an existing module propagate unchanged.
    """
    try:
        target = KERNELS[kind]
    except KeyError:
        raise KeyError(f"unknown kernel kind {kind!r}; registered: {sorted(KERNELS)}") from None
    module_name, _, attr = target.partition(":")
    owner = OWNERS.get(kind, "its work package")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        parts = module_name.split(".")
        prefixes = {".".join(parts[:j]) for j in range(1, len(parts) + 1)}
        if err.name not in prefixes:
            raise
        raise NotImplementedError(
            f"kernel {kind!r} is not available: module {module_name} cannot be imported "
            f"({err}); it is implemented by {owner}"
        ) from err
    cls = getattr(module, attr, None)
    if cls is None:
        raise NotImplementedError(
            f"kernel {kind!r}: {module_name} defines no {attr} yet (implemented by {owner})"
        )
    if not isinstance(cls, type):
        raise TypeError(f"{target} is not a class")
    return cls


def slot_kinds(static: StaticConfig) -> dict[str, str | tuple[str, ...]]:
    """The registry kinds a ``StaticConfig`` composes, by slot (DESIGN §3.8)."""
    reduced = static.mean.kind == "linear_reduced"
    return {
        "structure": static.struct.kind,
        "mean": static.mean.kind,
        "coef": static.coef.kind,
        "core": static.vol.persistent,
        "transitory": static.vol.transitory,
        "vol": "vol_block",
        "global_pre": ("cccm",) if reduced else (),
    }
