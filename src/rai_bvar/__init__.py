"""rai-bvar: ablation-ready Bayesian VARs (see SPEC.md)."""

from rai_bvar import datasets
from rai_bvar._version import __version__
from rai_bvar.api import BVAR
from rai_bvar.components import Constant, Linear
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
from rai_bvar.structural import IRFResult

__all__ = [
    "BVAR",
    "OLSVAR",
    "BetaPrior",
    "Cache",
    "Constant",
    "Entry",
    "Expanding",
    "ForecastResult",
    "GLPHyperpriors",
    "GammaPrior",
    "IRFResult",
    "InvGammaPrior",
    "LenzaPrimiceri",
    "Linear",
    "MHSettings",
    "Minnesota",
    "Origins",
    "ParetoPrior",
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
