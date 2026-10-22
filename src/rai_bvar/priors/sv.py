"""Priors of the volatility block ``SV`` (DESIGN §6.3–6.4; N02 §1, §10.2; N03 §0, §3).

Each class is named by its natural parametrisation, which avoids the shape/rate/s.d. confusions
of N06 pitfall 3:

* ``Normal(mean, var)``: ``x ~ N(mean, var)``; ``var`` is a variance (stochvol's ``B_μ``).
* ``Beta(a, b)``: density ``∝ x^{a−1}(1 − x)^{b−1}`` on (0, 1). For ``SV`` it is the prior of
  ``(φ + 1)/2`` (stochvol ``priorphi = c(a, b)``) or of the outlier probability ``p``.
* ``ScaledChi2(B)``: ``σ² ~ B·χ²₁ ⇔ σ ~ |N(0, B)|`` (stochvol ``priorsigma = B``, not ``√B``).
* ``InvGamma(shape, scale)``: density ``∝ x^{−shape−1} exp(−scale/x)``.
* ``DiscreteUniform(low, high)``: uniform on the integers ``{low, …, high}``, the static ν grid
  of the Student-t scale mixture (D-8; N03 §0).
* ``OutlierPrior(every_years, prior_years)``: ``p ~ Beta(a, f·prior_years − a)`` with
  ``a = prior_years/every_years`` and ``f`` periods per year (12 monthly, 4 quarterly): prior mean
  one outlier every ``every_years`` years, prior sample size ``prior_years`` years (N03 §3.1;
  CCMM 2024 §III.B). Resolved per window from the data frequency (D-9).

None of these hyperparameters depends on the data, so there is no look-ahead (R9). The package
exports this module as ``rb.sv_priors`` so that ``Beta`` does not clash with the GLP
``BetaPrior`` (which is parametrised by its mode and s.d.). Numeric fields are stored as Python
``float``/``int``, so equal priors have equal specs and labels.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

_PERIODS_PER_YEAR = {"Q": 4, "M": 12}


def _number(obj: object, name: str, *, positive: bool = True) -> None:
    value = getattr(obj, name)
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{type(obj).__name__}.{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(
            f"{type(obj).__name__}.{name} must be {'positive and ' if positive else ''}finite, "
            f"got {value!r}"
        )
    object.__setattr__(obj, name, value)


def _integer(obj: object, name: str, minimum: int) -> None:
    value = getattr(obj, name)
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{type(obj).__name__}.{name} must be an int, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{type(obj).__name__}.{name} must be at least {minimum}, got {value}")
    object.__setattr__(obj, name, int(value))


@dataclass(frozen=True)
class Normal:
    """``x ~ N(mean, var)`` (``var`` is a variance)."""

    mean: float
    var: float

    def __post_init__(self) -> None:
        _number(self, "mean", positive=False)
        _number(self, "var")


@dataclass(frozen=True)
class Beta:
    """``x ~ Beta(a, b)`` on (0, 1): mean ``a/(a + b)``."""

    a: float
    b: float

    def __post_init__(self) -> None:
        _number(self, "a")
        _number(self, "b")

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)


@dataclass(frozen=True)
class ScaledChi2:
    """``σ² ~ B·χ²₁``, i.e. ``σ ~ |N(0, B)|`` (stochvol's ``priorsigma = B``)."""

    B: float

    def __post_init__(self) -> None:
        _number(self, "B")


@dataclass(frozen=True)
class InvGamma:
    """``x ~ IG(shape, scale)``: density ``∝ x^{−shape−1} exp(−scale/x)``."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        _number(self, "shape")
        _number(self, "scale")


@dataclass(frozen=True)
class DiscreteUniform:
    """Uniform prior on the integers ``{low, …, high}`` (the static ν grid, D-8)."""

    low: int
    high: int

    def __post_init__(self) -> None:
        _integer(self, "low", 1)
        _integer(self, "high", 1)
        if self.high < self.low:
            raise ValueError(f"DiscreteUniform needs low ≤ high, got ({self.low}, {self.high})")

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.low, self.high + 1))


@dataclass(frozen=True)
class OutlierPrior:
    """Outlier-probability prior by frequency-free settings (N03 §3.1; CCMM 2024 §III.B).

    With ``f`` periods per year, ``p ~ Beta(a, f·prior_years − a)``, ``a =
    prior_years/every_years``: prior mean ``1/(f·every_years)`` (one outlier every
    ``every_years`` years) and prior sample size ``f·prior_years``. Defaults: SVO
    ``every_years=4`` (quarterly ``Beta(2.5, 37.5)``), SVO-t ``every_years=10`` (``Beta(1, 39)``).
    """

    every_years: float = 4.0
    prior_years: float = 10.0

    def __post_init__(self) -> None:
        _number(self, "every_years")
        _number(self, "prior_years")

    def resolve(self, freq: str) -> Beta:
        """The ``Beta`` prior on ``p`` for data of frequency ``freq`` (a pandas frequency string
        such as ``"Q-DEC"`` or ``"M"``, i.e. ``VARData.freq``)."""
        f = periods_per_year(freq)
        a = self.prior_years / self.every_years
        b = f * self.prior_years - a
        if not b > 0:
            raise ValueError(
                f"OutlierPrior(every_years={self.every_years}, prior_years={self.prior_years}) "
                f"needs f·every_years > 1 (f = {f}): the prior mean probability would reach 1"
            )
        return Beta(a, b)


def periods_per_year(freq: str) -> int:
    """Periods per year of a pandas frequency string: 4 (quarterly) or 12 (monthly)."""
    base = str(freq).upper().split("-")[0].rstrip("SE")  # QS-DEC → Q, MS/ME → M
    if base not in _PERIODS_PER_YEAR:
        raise ValueError(
            f"the outlier prior needs quarterly or monthly data, got frequency {freq!r}; "
            "pass SV(outlier_prior=sv_priors.Beta(a, b)) explicitly for other frequencies"
        )
    return _PERIODS_PER_YEAR[base]


PRIOR_CLASSES: dict[str, type] = {
    cls.__name__: cls for cls in (Normal, Beta, ScaledChi2, InvGamma, DiscreteUniform, OutlierPrior)
}


def prior_to_spec(obj: Any) -> dict[str, Any]:
    """``{"class": <name>, **fields}``: the JSON form of an SV prior (DESIGN §11.5)."""
    if type(obj) not in PRIOR_CLASSES.values():
        raise TypeError(f"not an SV prior: {obj!r}")
    return {"class": type(obj).__name__, **asdict(obj)}


def prior_from_spec(d: dict[str, Any]) -> Any:
    """Inverse of :func:`prior_to_spec`."""
    cls = PRIOR_CLASSES.get(d.get("class")) if isinstance(d, dict) else None
    if cls is None:
        raise ValueError(
            f"not an SV prior spec: {d!r}; expected a 'class' in {list(PRIOR_CLASSES)}"
        )
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names - {"class"})
    if unknown:
        raise ValueError(f"unknown fields {unknown} for {cls.__name__}")
    return cls(**{k: v for k, v in d.items() if k != "class"})


__all__ = [
    "Beta",
    "DiscreteUniform",
    "InvGamma",
    "Normal",
    "OutlierPrior",
    "ScaledChi2",
    "periods_per_year",
    "prior_from_spec",
    "prior_to_spec",
]
