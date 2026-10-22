"""Empirical Bayes for the MCMC engine: plug-in Minnesota hyperparameters (DESIGN §6.5; D-12, E5,
E25; N05 §9). NumPy only: this module imports no JAX.

Minnesota hyperparameters are fixed during MCMC. With ``hierarchy="eb"`` (the default) the mode of
the free set (``lambda``; optionally ``alpha`` and ``psi``) is computed per window before sampling:

1. **Conjugate twin.** ``BVAR(Linear(Minnesota(form="conjugate", cross=1, contemp=1, <same
   lambda_, alpha, psi, mean, var_const, hyper, hyperpriors, soc_scope>, hierarchy="eb")),
   Constant("iw"), lags, deterministic)`` is fitted by the Phase 1 ``AnalyticFit`` on the same
   window (ψ default, multistart, stationarity test: all unchanged). At ``κ_c = κ_a = 1`` its NIW
   marginal likelihood is exactly the marginal likelihood of the Mode II Constant model (N05 §4.2),
   so the mode is true type-II ML for the twin and a documented plug-in for Mode I and SV
   (N05 §9 "Phase 2 shortcut").
2. ``form="structural"``, ``cross ≠ 1``: Chan's (2022) per-equation NIG log ML (eq. 11) plus the
   GLP log hyperprior, maximised over the free set with ``cross`` (and ``contemp``) fixed, starting
   from the twin's mode (``engines.structural_nig.eb_structural``, WP16). ``psi`` cannot be free
   here (rejected when the ``BVAR`` is built).
3. ``form="reduced"``, ``cross ≠ 1``: no closed-form objective exists; the twin's mode is used and
   :data:`REDUCED_CROSS_NOTE` is recorded (a note, not a warning, so backtests stay quiet).

The twin's (or Chan EB's) ``converged`` flag and its optimisation and ψ messages are returned for
``MCMCFit.converged`` and ``fit.warnings``. The twin's *setting* warnings (tcode clash, 2020
coverage) are dropped: the MCMC fit recomputes them, because ``SV(outliers=True)`` counts as an
outlier treatment there (DESIGN §8.3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from rai_bvar.components import Constant, Linear
from rai_bvar.data.dataset import VARData
from rai_bvar.engines import structural_nig
from rai_bvar.engines.analytic import AnalyticFit, _setting_warnings

if TYPE_CHECKING:
    from rai_bvar.api import BVAR

__all__ = ["REDUCED_CROSS_NOTE", "EBOutcome", "conjugate_twin", "eb_hyper"]

REDUCED_CROSS_NOTE = "EB at cross=1 plugged into the reduced-form prior"


@dataclass(frozen=True)
class EBOutcome:
    """Plug-in EB values of one window (DESIGN §6.5).

    ``hyper``: ``lambda``, ``alpha`` (floats) and ``psi`` (n,) in use, plus ``eb_method``
    (``"conjugate_twin"`` or ``"structural_nig"``) and ``eb_free`` (the optimised names).
    ``converged``: the optimiser's stationarity test passed (Phase 1 rule). ``warnings``: ψ notes and
    optimisation messages for ``fit.warnings`` (never emitted here). ``note``: the reduced-form
    ``cross ≠ 1`` note, else ``None``.
    """

    hyper: dict[str, Any]
    converged: bool
    warnings: tuple[str, ...]
    note: str | None


def conjugate_twin(model: BVAR) -> BVAR:
    """The analytic twin of an MCMC ``model`` (DESIGN §6.5 step 1): the same Minnesota settings at
    ``form="conjugate"``, ``cross = contemp = 1``, ``hierarchy="eb"``, with ``Constant("iw")``,
    the same lags and deterministic terms. SOC/SUR dummies and Lenza–Primiceri scaling are
    rejected on the MCMC engine (D-21), so the twin has none."""
    from rai_bvar.api import BVAR

    prior = replace(
        model.mean.prior,
        form="conjugate",
        cross=1.0,
        contemp=1.0,
        hierarchy="eb",
        soc=None,
        sur=None,
    )
    return BVAR(
        mean=Linear(prior=prior),
        vol=Constant("iw"),
        lags=model.lags,
        deterministic=model.deterministic,
    )


def eb_hyper(model: BVAR, data: VARData) -> EBOutcome:
    """EB plug-in values of ``model`` (an MCMC ``BVAR`` with ``hierarchy="eb"``) on the window
    ``data`` (DESIGN §6.5). Only ``data`` is read, so the values depend on the window alone (R9)."""
    if getattr(model, "engine", None) != "mcmc":
        raise ValueError("eb_hyper serves MCMC compositions; analytic models run EB themselves")
    if not isinstance(data, VARData):
        raise TypeError(f"data must be a VARData, got {type(data).__name__}")
    prior = model.mean.prior
    if prior.hierarchy != "eb":
        raise ValueError(f"eb_hyper needs hierarchy='eb', got {prior.hierarchy!r}")
    twin_model = conjugate_twin(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # collected in twin.warnings
        twin = AnalyticFit(twin_model, data, None)
    setting = set(_setting_warnings(twin.problem, twin_model, data))
    notes = tuple(dict.fromkeys(twin.problem.notes))  # ψ messages (e.g. few AR degrees of freedom)
    messages = tuple(w for w in twin.warnings if w not in setting and w not in notes)
    free = tuple(prior.free)
    if prior.form == "structural" and prior.cross != 1.0:
        chan = structural_nig.eb_structural(
            data, prior, model.lags, model.deterministic, start=twin.hyper
        )
        return EBOutcome(
            hyper=_values(chan.hyper, "structural_nig", free),
            converged=bool(chan.converged),
            warnings=notes + tuple(chan.warnings),
            note=None,
        )
    note = REDUCED_CROSS_NOTE if prior.form == "reduced" and prior.cross != 1.0 else None
    return EBOutcome(
        hyper=_values(twin.hyper, "conjugate_twin", free),
        converged=bool(twin.converged),
        warnings=notes + messages,
        note=note,
    )


def _values(hyper: dict[str, Any], method: str, free: tuple[str, ...]) -> dict[str, Any]:
    """``lambda``, ``alpha`` and ``psi`` of a Phase 1 (or structural NIG) hyperparameter dict."""
    return {
        "lambda": float(hyper["lambda"]),
        "alpha": float(hyper["alpha"]),
        "psi": np.array(hyper["psi"], dtype=np.float64),
        "eb_method": method,
        "eb_free": free,
    }
