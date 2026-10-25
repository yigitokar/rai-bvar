"""What the backtest and compare engines need from a model."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rai_bvar.data.dataset import VARData
from rai_bvar.forecast.result import ForecastResult

SeedLike = int | np.random.Generator | np.random.SeedSequence | None


@runtime_checkable
class FittedModel(Protocol):
    """A fitted model: ``forecast(horizon, draws, seed, *, lag_coefs=False)``.

    ``lag_coefs=True`` asks for the accelerator variable ``lag_coefs`` (the lag matrices of each
    path's parameter draw, P3 DESIGN D-49, E120), which ``log_pred_density(lpd_method=
    "integrated")`` needs at ``h > 1``; a fit that cannot provide it accepts and ignores the
    keyword (the benchmarks). The default forecast is the Phase 1 one.
    """

    def forecast(
        self, horizon: int, draws: int = 1000, seed: SeedLike = None, *, lag_coefs: bool = False
    ) -> ForecastResult: ...


def forecast_options(fit: Any, *, lag_coefs: bool) -> dict[str, Any]:
    """Keywords for ``fit.forecast`` (P3 DESIGN D-49): ``{"lag_coefs": True}`` when
    ``lag_coefs`` is requested and ``fit.forecast`` takes the keyword, else ``{}`` (the call is
    then exactly the Phase 1 one). A forecast without the keyword is a fit that attaches
    ``lag_coefs`` whenever it can (``MCMCFit``, P2 D-34) or a model written to the Phase 1
    protocol; it is called as before."""
    if not lag_coefs:
        return {}
    try:
        params = inspect.signature(fit.forecast).parameters.values()
    except (TypeError, ValueError):  # no introspectable signature: call as before
        return {}
    takes = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == "lag_coefs" and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for p in params
    )
    return {"lag_coefs": True} if takes else {}


@runtime_checkable
class Model(Protocol):
    """A model specification: immutable, picklable, and fit afresh at every forecast origin.

    The protocol needs only ``fit(data)``. Optional capabilities (DESIGN §11.3, §11.6) are
    declared by boolean attributes; the backtest passes the matching keywords only to models
    that set them, so every other model is called exactly as ``fit(data)``:

    * ``uses_fit_seed``: ``fit(data, seed=<int>)`` takes a per-origin fit seed (MCMC ``BVAR``);
    * ``supports_warm_start``: ``fit(data, init=<previous final state>)`` warm-starts the
      sampler (opt-in in backtests);
    * ``requires_in_loop_forecast``: ``fit(data, plan=FitPlan(...))`` must compute forecasts
      inside the sampling loop (no Phase 2 model).

    ``BVAR.fit`` also takes ``runtime=Runtime(...)`` (execution knobs outside the spec).
    """

    @property
    def name(self) -> str: ...

    @property
    def axes(self) -> Mapping[str, str]:
        """Labels along the ablation axes (e.g. ``{"mean": "Linear", "vol": "SV"}``)."""
        ...

    def spec(self) -> dict[str, Any]:
        """JSON-serialisable description; equal specs must give identical forecasts."""
        ...

    def fit(self, data: VARData) -> FittedModel: ...


def spec_hash(model: Model) -> str:
    return spec_hash_of(model.spec())


def spec_hash_of(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_label(seed: SeedLike, rng: np.random.Generator | None = None) -> str:
    """A reproducible description of the randomness a forecast uses, for its attrs (SPEC §7).

    Call it right after building ``rng`` and before drawing. A Generator that has already been
    advanced is labelled by a digest of its current state, so different draws never share a
    label.
    """
    if isinstance(seed, int | np.integer) and not isinstance(seed, bool):
        return str(int(seed))
    if isinstance(seed, np.random.SeedSequence):
        return f"SeedSequence(entropy={seed.entropy}, spawn_key={tuple(seed.spawn_key)})"
    gen = rng if rng is not None else seed if isinstance(seed, np.random.Generator) else None
    if gen is None:
        return "unknown"
    bitgen = gen.bit_generator
    seq = getattr(bitgen, "seed_seq", None)
    digest = _state_digest(bitgen.state)
    if isinstance(seq, np.random.SeedSequence) and digest == _state_digest(type(bitgen)(seq).state):
        return f"SeedSequence(entropy={seq.entropy}, spawn_key={tuple(seq.spawn_key)})"
    return f"{type(bitgen).__name__}(state={digest[:16]})"


def _state_digest(state: dict) -> str:
    payload = json.dumps(state, sort_keys=True, default=lambda a: np.asarray(a).tolist())
    return hashlib.sha256(payload.encode()).hexdigest()
