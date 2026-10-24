"""MCMC settings, runtime knobs and in-loop forecast plans (DESIGN §11.2; fixes E14, E15).

Pure Python, so ``import rai_bvar`` stays free of JAX (D-31).

* ``MCMCSettings`` is **part of the model spec** (``BVAR.mcmc``): every field can change the
  draws, so it enters the spec hash and the backtest cache key. ``draws=None`` is resolved per
  composition by ``BVAR`` and stored (D-35), so a spec always holds a number; ``gram`` and
  ``const_path`` are resolved by deterministic shape rules (DESIGN §3.7.2, §10.5), never by
  machine properties.
* ``Runtime`` is **not** in the spec: it holds execution knobs whose invariance is tested
  (bitwise: chunk size, storage location, replay copies; to rounding: the device layout, so
  ``devices`` defaults to one device; DESIGN §5.1, §5.2, §5.6).
* ``FitPlan`` requests forecasts computed inside the sampling loop (DESIGN §5.8).
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

_U32 = 2**32


def _int(obj: object, name: str, minimum: int, *, optional: bool = False) -> None:
    value = getattr(obj, name)
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{type(obj).__name__}.{name} must be an int, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{type(obj).__name__}.{name} must be at least {minimum}, got {value}")
    object.__setattr__(obj, name, int(value))


def _float(obj: object, name: str) -> float:
    value = getattr(obj, name)
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{type(obj).__name__}.{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{type(obj).__name__}.{name} must be finite, got {value!r}")
    object.__setattr__(obj, name, value)
    return value


def _choice(obj: object, name: str, options: tuple[str, ...]) -> None:
    if getattr(obj, name) not in options:
        raise ValueError(
            f"{type(obj).__name__}.{name} must be one of {options}, got {getattr(obj, name)!r}"
        )


def _bool(obj: object, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{type(obj).__name__}.{name} must be a bool, got {value!r}")
    object.__setattr__(obj, name, bool(value))


@dataclass(frozen=True)
class MCMCSettings:
    """Sampler settings; part of the spec because results depend on every field.

    ``burn`` sweeps are run exactly (never rounded to a chunk), then ``draws`` kept draws per chain,
    one every ``thin`` sweeps. ``draws=None`` (the default) is the composition's default budget,
    resolved and stored by ``BVAR`` (D-35): :data:`SV_DEFAULT_DRAWS` (10,000) for a stochastic-
    volatility composition (``SV``, with or without outliers, Student-t tails or a random walk),
    else :data:`DEFAULT_DRAWS` (2,000); see :func:`default_draws`. An explicit ``draws`` is kept
    as given. ``warm_burn`` (default ``max(250, burn // 4)``) replaces ``burn`` for warm-started
    backtest fits (DESIGN §5.5). Parameter draws are stored every
    ``⌈draws/store_max⌉``-th kept draw; ``store="paths"`` also stores latent paths every
    ``path_thin``-th stored draw (DESIGN §5.3). ``t_bucket`` pads ``T_eff`` up to a multiple
    (``T_pad``) so one compiled program serves many windows (D-23). ``converged`` needs every
    primary monitor to have R-hat < ``rhat_max`` and bulk ESS > ``ess_min`` (DESIGN §8.2).
    """

    chains: int = 4
    draws: int | None = None  # None → default_draws(sv) at BVAR construction (D-35)
    burn: int = 1000
    thin: int = 1
    warm_burn: int | None = None
    seed: int = 0
    store: Literal["params", "paths"] = "params"
    store_max: int = 2500
    path_thin: int = 10
    t_bucket: int = 32
    gram: Literal["auto", "packed", "packed_tiled", "batched"] = "auto"
    const_path: Literal["auto", "cholesky"] = "auto"
    rhat_max: float = 1.01
    ess_min: float = 400.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("chains", 1),
            ("burn", 0),
            ("thin", 1),
            ("seed", 0),
            ("store_max", 1),
            ("path_thin", 1),
            ("t_bucket", 1),
        ):
            _int(self, name, minimum)
        _int(self, "draws", 1, optional=True)
        _int(self, "warm_burn", 0, optional=True)
        if self.seed >= 2**63:
            raise ValueError(f"MCMCSettings.seed must be below 2**63, got {self.seed}")
        _choice(self, "store", ("params", "paths"))
        _choice(self, "gram", ("auto", "packed", "packed_tiled", "batched"))
        _choice(self, "const_path", ("auto", "cholesky"))
        if not _float(self, "rhat_max") > 1.0:
            raise ValueError("MCMCSettings.rhat_max must exceed 1")
        if not _float(self, "ess_min") >= 0.0:
            raise ValueError("MCMCSettings.ess_min must be non-negative")
        if self.draws is None:  # checked again once BVAR resolves draws (``replace`` re-runs this)
            return
        sweeps = max(self.burn, self.resolved_warm_burn) + self.draws * self.thin
        if sweeps >= _U32:
            raise ValueError(
                f"burn + draws·thin = {sweeps} sweeps exceeds the uint32 sweep counter (2**32); "
                "reduce draws or thin"
            )

    @property
    def resolved_warm_burn(self) -> int:
        """Burn-in sweeps of a warm-started fit: ``warm_burn`` or ``max(250, burn // 4)``."""
        return max(250, self.burn // 4) if self.warm_burn is None else self.warm_burn

    @property
    def store_thin(self) -> int:
        """Kept draws per stored parameter draw: ``⌈draws/store_max⌉``."""
        return -(-self._resolved_draws("store_thin") // self.store_max)

    def with_default_draws(self, *, sv: bool) -> MCMCSettings:
        """These settings with ``draws=None`` replaced by :func:`default_draws` (``sv``: the
        composition has stochastic volatility); unchanged when ``draws`` is set (D-35)."""
        if self.draws is not None:
            return self
        return dataclasses.replace(self, draws=default_draws(sv=sv))

    def _resolved_draws(self, what: str) -> int:
        if self.draws is None:
            raise ValueError(
                f"MCMCSettings.{what} needs draws: draws=None is the composition's default "
                "budget, resolved by BVAR (or with_default_draws(sv=...)); D-35"
            )
        return self.draws


SV_DEFAULT_DRAWS = 10_000
"""Default kept draws per chain of a stochastic-volatility composition (D-35; DESIGN Q14): the
budget below which ``BVAR.fit`` of an ``SV`` composition warns (``fit.warnings``)."""

DEFAULT_DRAWS = 2_000
"""Default kept draws per chain of every other MCMC composition (``Constant`` volatility)."""


def default_draws(*, sv: bool) -> int:
    """The default ``MCMCSettings.draws`` of a composition with (``sv=True``) or without
    stochastic volatility (D-35)."""
    return SV_DEFAULT_DRAWS if sv else DEFAULT_DRAWS


@dataclass(frozen=True)
class Runtime:
    """Execution knobs outside the spec (DESIGN §5.2, §5.6; Stage 2 amendment).

    ``chunk`` (kept draws per compiled call), ``memory_budget_gb`` (only decides whether the
    storage plan raises), ``storage_dir`` (a memory-mapped host store) and ``replay`` (a host copy
    of each chunk's start state, so a numerical failure can be replayed deterministically; D-18)
    never change the draws: bitwise-invariance tested. ``devices`` (default ``1``) is opt-in
    sharding: an integer ``D ≥ 2`` dividing ``chains``, or ``"auto"`` (every local device when
    there are at least two and they divide ``chains``; see ``rb.set_cpu_devices``), runs the
    chains under ``shard_map``, whose draws equal the one-device draws only to rounding with the
    production SV and CCCM kernels (``rtol 1e-9`` on short runs, tested), not bitwise. The device
    count and chain method are recorded in ``fit.metadata``.
    """

    devices: int | Literal["auto"] = 1
    chunk: int = 250
    memory_budget_gb: float = 2.0
    storage_dir: str | None = None
    replay: bool = True
    progress: bool = False

    def __post_init__(self) -> None:
        if self.devices != "auto":
            _int(self, "devices", 1)
        _int(self, "chunk", 1)
        if not _float(self, "memory_budget_gb") > 0.0:
            raise ValueError("Runtime.memory_budget_gb must be positive")
        if self.storage_dir is not None:
            if not isinstance(self.storage_dir, str | os.PathLike):
                raise TypeError(
                    f"Runtime.storage_dir must be a path or None, got {self.storage_dir!r}"
                )
            object.__setattr__(self, "storage_dir", os.fspath(self.storage_dir))
        _bool(self, "replay")
        _bool(self, "progress")


@dataclass(frozen=True)
class FitPlan:
    """Forecasts computed inside the sampling loop (DESIGN §5.8): ``fit.forecast(horizon, draws,
    seed)`` returns them when the arguments match. Needed only by models that cannot store their
    draws (Phase 4 BART forests, Phase 5 TVP paths)."""

    horizon: int
    draws: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        _int(self, "horizon", 1)
        _int(self, "draws", 1)
        _int(self, "seed", 0)


__all__ = [
    "DEFAULT_DRAWS",
    "SV_DEFAULT_DRAWS",
    "FitPlan",
    "MCMCSettings",
    "Runtime",
    "default_draws",
]
