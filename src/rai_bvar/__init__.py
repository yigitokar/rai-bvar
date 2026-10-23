"""rai-bvar: ablation-ready Bayesian VARs (see SPEC.md).

``import rai_bvar`` imports no JAX (DESIGN §1.1, D-31): specs, settings and the analytic engine
are NumPy-only, and the MCMC engine is imported when an MCMC composition is fitted. The device
helpers ``set_cpu_devices`` / ``enable_compilation_cache`` and ``ordering_sensitivity`` are
resolved lazily on first access, so ``rb.set_cpu_devices(k)`` can still run before JAX starts. A
lazy name whose module is not part of the build is neither in ``__all__`` nor an attribute, so
``from rai_bvar import *`` and ``hasattr`` keep working.
"""

from typing import Any

from rai_bvar import datasets
from rai_bvar._version import __version__
from rai_bvar.api import BVAR
from rai_bvar.components import SV, Constant, Linear
from rai_bvar.data import VARData
from rai_bvar.data.outliers import LenzaPrimiceri
from rai_bvar.evaluation import (
    Cache,
    Entry,
    Expanding,
    Origins,
    Rolling,
    backtest,
    compare,
    diebold_mariano,
    estimate_cost,
    grid,
    model_confidence_set,
    score_forecast,
)
from rai_bvar.forecast import ForecastResult
from rai_bvar.models import OLSVAR, RandomWalk
from rai_bvar.priors import (
    BetaPrior,
    GammaPrior,
    GLPHyperpriors,
    InvGammaPrior,
    MHSettings,
    Minnesota,
    ParetoPrior,
)
from rai_bvar.priors import sv as sv_priors
from rai_bvar.settings import FitPlan, MCMCSettings, Runtime
from rai_bvar.structural import IRFResult

_LAZY = {
    "set_cpu_devices": "rai_bvar._devices",
    "enable_compilation_cache": "rai_bvar._devices",
    "ordering_sensitivity": "rai_bvar.structural.ordering",  # WP22
}


def _in_build(module: str) -> bool:
    """Whether ``module`` can be imported (located without being executed)."""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a missing parent package
        return False


def __getattr__(name: str) -> Any:
    """Lazy attributes (PEP 562): their modules may import JAX, which ``import rai_bvar`` must not."""
    if name in _LAZY:
        import importlib

        module = _LAZY[name]
        try:
            value = getattr(importlib.import_module(module), name)
        except ModuleNotFoundError as err:
            missing = err.name or ""
            if not (module == missing or module.startswith(missing + ".")):
                raise  # a dependency of an existing module is missing: a real error
            raise AttributeError(
                f"module 'rai_bvar' has no attribute {name!r} ({module} is not part of this build)"
            ) from err
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rai_bvar' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {n for n, m in _LAZY.items() if _in_build(m)})


__all__ = [
    "BVAR",
    "OLSVAR",
    "SV",
    "BetaPrior",
    "Cache",
    "Constant",
    "Entry",
    "Expanding",
    "FitPlan",
    "ForecastResult",
    "GLPHyperpriors",
    "GammaPrior",
    "IRFResult",
    "InvGammaPrior",
    "LenzaPrimiceri",
    "Linear",
    "MCMCSettings",
    "MHSettings",
    "Minnesota",
    "Origins",
    "ParetoPrior",
    "RandomWalk",
    "Rolling",
    "Runtime",
    "VARData",
    "__version__",
    "backtest",
    "compare",
    "datasets",
    "diebold_mariano",
    "estimate_cost",
    "grid",
    "model_confidence_set",
    "score_forecast",
    "sv_priors",
] + [n for n, m in _LAZY.items() if _in_build(m)]
