"""Model components along the ablation axes (SPEC §4.4). Phase 1 ships the conjugate ones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rai_bvar.priors.minnesota import Minnesota


@dataclass(frozen=True)
class Linear:
    """Linear conditional mean ``g(x_t) = B'x_t`` with a Minnesota-type prior."""

    prior: Minnesota = field(default_factory=Minnesota)

    @property
    def name(self) -> str:
        return "Linear"

    @property
    def label(self) -> str:
        """Name plus prior, so grids over priors get distinct labels."""
        return f"Linear[{self.prior.label}]"


@dataclass(frozen=True)
class Constant:
    """Constant error covariance. ``prior="iw"`` is the conjugate inverse-Wishart case."""

    prior: Literal["iw", "ig"] = "iw"

    def __post_init__(self) -> None:
        if self.prior not in ("iw", "ig"):
            raise ValueError("Constant prior must be 'iw' or 'ig'")

    @property
    def name(self) -> str:
        return "Constant"
