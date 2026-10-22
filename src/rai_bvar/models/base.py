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
