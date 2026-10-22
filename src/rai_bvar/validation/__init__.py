"""Validation harness of the MCMC engine (DESIGN §9; N06).

Model-agnostic statistics (WP7): Geweke joint-distribution tests (:mod:`.geweke`), simulation-based
calibration (:mod:`.sbc`), rank uniformity (:mod:`.uniformity`), multiplicity and the Gandy–Scott
sequential wrapper (:mod:`.sequential`), and density-ratio / MH identities (:mod:`.identities`).
Independent references (WP9): prior samplers, simulators and log joints (:mod:`.reference`,
:mod:`.simulate`, :mod:`.log_joint`, :mod:`.testfns`, :mod:`.joint`).

Importing this package imports no JAX; the names below are resolved lazily (the JAX drivers enable
float64 when first imported, SPEC §8).
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "JointModel": "rai_bvar.validation.geweke",
    "GewekeConfig": "rai_bvar.validation.geweke",
    "GewekeResult": "rai_bvar.validation.geweke",
    "geweke": "rai_bvar.validation.geweke",
    "geweke_sequential": "rai_bvar.validation.geweke",
    "SBCConfig": "rai_bvar.validation.sbc",
    "SBCResult": "rai_bvar.validation.sbc",
    "sbc": "rai_bvar.validation.sbc",
    "block_sbc": "rai_bvar.validation.sbc",
    "sbc_sequential": "rai_bvar.validation.sbc",
    "rank_with_ties": "rai_bvar.validation.uniformity",
    "gamma_statistic": "rai_bvar.validation.uniformity",
    "gamma_pvalue": "rai_bvar.validation.uniformity",
    "gamma_test": "rai_bvar.validation.uniformity",
    "SequentialResult": "rai_bvar.validation.sequential",
    "gandy_scott": "rai_bvar.validation.sequential",
    "density_ratio_error": "rai_bvar.validation.identities",
    "mh_ratio_error": "rai_bvar.validation.identities",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
