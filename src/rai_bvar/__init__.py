"""rai-bvar: ablation-ready Bayesian VARs (see SPEC.md)."""

from rai_bvar import datasets
from rai_bvar._version import __version__
from rai_bvar.data import VARData
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

__all__ = [
    "OLSVAR",
    "Cache",
    "Entry",
    "Expanding",
    "ForecastResult",
    "Origins",
    "RandomWalk",
    "Rolling",
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
]
