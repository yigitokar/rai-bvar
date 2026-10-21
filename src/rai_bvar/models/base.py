"""What the backtest and compare engines need from a model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rai_bvar.data.dataset import VARData
from rai_bvar.forecast.result import ForecastResult

SeedLike = int | np.random.Generator | np.random.SeedSequence | None


@runtime_checkable
class FittedModel(Protocol):
    def forecast(
        self, horizon: int, draws: int = 1000, seed: SeedLike = None
    ) -> ForecastResult: ...


@runtime_checkable
class Model(Protocol):
    """A model specification: immutable, picklable, and fit afresh at every forecast origin."""

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
