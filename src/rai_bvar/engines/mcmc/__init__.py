"""The JAX MCMC engine (Phase 2; docs/phase2/DESIGN.md).

Importing this package imports no JAX (DESIGN §1.1): ``config`` and ``registry`` are pure
Python, so ``api.py`` can validate compositions without loading JAX. The public names below are
re-exported lazily; the first access imports the engine, which enables float64 process-wide
(``_jax``, SPEC §8).
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "fit_mcmc": "rai_bvar.engines.mcmc.results",
    "MCMCFit": "rai_bvar.engines.mcmc.results",
    "MCMCNumericalError": "rai_bvar.engines.mcmc.protocols",
}

__all__ = ["MCMCFit", "MCMCNumericalError", "fit_mcmc"]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_LAZY[name])
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{name} is not available yet ({_LAZY[name]} does not define it)"
        ) from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
