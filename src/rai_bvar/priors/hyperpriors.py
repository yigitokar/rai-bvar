"""Hyperpriors for the hierarchical Minnesota prior (GLP 2015 defaults) and COVID scaling.

All densities are on the natural scale of the hyperparameter and return ``−inf`` outside their
support. ``lower``/``upper`` are hard bounds for optimisation and Metropolis-Hastings
(proposals outside are rejected); the densities are not renormalised to the bounds.
Numeric fields are stored as Python floats so equal priors have equal specs and labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import optimize, special, stats


def _coerce(obj, *names: str) -> None:
    for name in names:
        value = getattr(obj, name)
        if value is not None:
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            object.__setattr__(obj, name, float(value))


def _check_bounds(lower: float | None, upper: float | None) -> None:
    for value in (lower, upper):
        if value is not None and not np.isfinite(value):
            raise ValueError("bounds must be finite")
    if lower is not None and upper is not None and not lower < upper:
        raise ValueError("lower bound must be below the upper bound")


@dataclass(frozen=True)
class GammaPrior:
    """Gamma prior parametrised by its mode and standard deviation."""

    mode: float
    sd: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _coerce(self, "mode", "sd", "lower", "upper")
        if not (np.isfinite(self.mode) and self.mode > 0 and np.isfinite(self.sd) and self.sd > 0):
            raise ValueError("Gamma prior needs a positive finite mode and sd")
        if self.lower <= 0:
            raise ValueError("Gamma prior bounds must be positive")
        _check_bounds(self.lower, self.upper)

    @property
    def shape(self) -> float:
        r = self.mode**2 / self.sd**2
        return (2.0 + r + np.sqrt((4.0 + r) * r)) / 2.0

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.sd**2 / self.shape))

    def logpdf(self, x: float) -> float:
        if x is None or not x > 0:
            return -np.inf
        k, theta = self.shape, self.scale
        return float((k - 1.0) * np.log(x) - x / theta - special.gammaln(k) - k * np.log(theta))


@dataclass(frozen=True)
class InvGammaPrior:
    """Inverse-gamma prior with density ``∝ x^{−shape−1} exp(−scale/x)``.

    For ψ, ``lower``/``upper`` default to the ψ mode (the AR residual variance) divided and
    multiplied by 100.
    """

    scale: float
    shape: float
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        _coerce(self, "scale", "shape", "lower", "upper")
        if not (
            np.isfinite(self.scale)
            and self.scale > 0
            and np.isfinite(self.shape)
            and self.shape > 0
        ):
            raise ValueError("inverse-gamma prior needs positive finite scale and shape")
        if any(b is not None and b <= 0 for b in (self.lower, self.upper)):
            raise ValueError("inverse-gamma bounds must be positive")
        _check_bounds(self.lower, self.upper)

    def logpdf(self, x: float) -> float:
        if x is None or not x > 0:
            return -np.inf
        a, b = self.shape, self.scale
        return float(a * np.log(b) - special.gammaln(a) - (a + 1.0) * np.log(x) - b / x)


@dataclass(frozen=True)
class BetaPrior:
    """Beta prior parametrised by its mode and standard deviation, truncated to ``[lower, upper]``."""

    mode: float
    sd: float
    lower: float = 1e-6
    upper: float = 1.0 - 1e-6

    def __post_init__(self) -> None:
        _coerce(self, "mode", "sd", "lower", "upper")
        if not 0.0 < self.mode < 1.0:
            raise ValueError("Beta mode must lie in (0, 1)")
        if not np.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("Beta sd must be positive")
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Beta bounds must satisfy 0 ≤ lower < upper ≤ 1")
        if not self.lower <= self.mode <= self.upper:
            raise ValueError("Beta mode must lie within its bounds")
        mass = float(np.diff(stats.beta(self.a, self.b).cdf([self.lower, self.upper]))[0])
        if mass < 1e-6:
            raise ValueError(
                f"the Beta prior puts only {mass:.2g} of its mass inside [lower, upper]"
            )

    @cached_property
    def _ab(self) -> tuple[float, float]:
        def excess(c: float) -> float:  # c = a + b > 2; the variance falls as c grows
            a = self.mode * (c - 2.0) + 1.0
            b = c - a
            return a * b / (c**2 * (c + 1.0)) - self.sd**2

        lo = 2.0 + 1e-9
        if excess(lo) < 0:
            raise ValueError(f"no unimodal Beta has mode {self.mode} and sd {self.sd}")
        hi = 1e3
        while excess(hi) > 0:
            hi *= 1e3
            if hi > 1e300:
                raise ValueError(f"Beta sd {self.sd} is too small to represent")
        c = optimize.brentq(excess, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
        if c > 1e12:  # logpdf terms of size ~c lose all precision beyond this
            raise ValueError(
                f"Beta sd {self.sd} is too small to represent; fix the parameter instead"
            )
        a = self.mode * (c - 2.0) + 1.0
        return a, c - a

    @property
    def a(self) -> float:
        return self._ab[0]

    @property
    def b(self) -> float:
        return self._ab[1]

    def logpdf(self, x: float) -> float:
        if x is None or not 0.0 < x < 1.0:
            return -np.inf
        a, b = self._ab
        return float((a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - special.betaln(a, b))

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws from the prior truncated to ``[lower, upper]`` (inverse CDF)."""
        dist = stats.beta(self.a, self.b)
        lo, hi = dist.cdf([self.lower, self.upper])
        return np.clip(dist.ppf(lo + rng.uniform(size=size) * (hi - lo)), self.lower, self.upper)


@dataclass(frozen=True)
class ParetoPrior:
    """Pareto prior, density ``α x_m^α / x^{α+1}`` for ``x ≥ x_m``, truncated above at ``upper``.

    The truncation matters for Lenza-Primiceri scales that no observation informs (forecast
    origins at ``t*`` or ``t*+1``): Pareto(1, 1) has an infinite mean, so the truncation point
    sets the prior predictive variance there (``E[s²] = upper`` for Pareto(1, 1)).
    """

    scale: float = 1.0
    shape: float = 1.0
    upper: float = 1e3

    def __post_init__(self) -> None:
        _coerce(self, "scale", "shape", "upper")
        finite = bool(np.isfinite([self.scale, self.shape, self.upper]).all())
        if not (finite and self.scale > 0 and self.shape > 0 and self.upper > self.scale):
            raise ValueError("Pareto prior needs finite scale > 0, shape > 0 and upper > scale")

    @property
    def lower(self) -> float:
        return self.scale

    def logpdf(self, x: float) -> float:
        if x is None or not x >= self.scale:
            return -np.inf
        a, m = self.shape, self.scale
        return float(np.log(a) + a * np.log(m) - (a + 1.0) * np.log(x))

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Exact draws from the prior truncated to ``[scale, upper]`` (inverse CDF)."""
        a, m = self.shape, self.scale
        tail = 1.0 - (m / self.upper) ** a
        return m * (1.0 - rng.uniform(size=size) * tail) ** (-1.0 / a)


@dataclass(frozen=True)
class GLPHyperpriors:
    """Defaults of Giannone, Lenza & Primiceri (2015): Gamma priors with modes 0.2 / 1 / 1 and
    standard deviations 0.4 / 1 / 1 on λ, μ (SOC), δ (SUR); Inverse-Gamma(0.02², 0.02²) on
    ``ψ_j / (d − n − 1)``. The lag-decay α is fixed at 2 in GLP; its prior is used only when α
    is estimated."""

    lambda_: GammaPrior = GammaPrior(mode=0.2, sd=0.4, lower=1e-4, upper=5.0)
    alpha: GammaPrior = GammaPrior(mode=2.0, sd=0.25, lower=1.0, upper=3.0)
    soc: GammaPrior = GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=50.0)
    sur: GammaPrior = GammaPrior(mode=1.0, sd=1.0, lower=1e-4, upper=50.0)
    psi: InvGammaPrior = InvGammaPrior(scale=0.02**2, shape=0.02**2)

    def __post_init__(self) -> None:
        for name in ("lambda_", "alpha", "soc", "sur"):
            if not isinstance(getattr(self, name), GammaPrior):
                raise TypeError(f"GLPHyperpriors.{name} must be a GammaPrior")
        if not isinstance(self.psi, InvGammaPrior):
            raise TypeError("GLPHyperpriors.psi must be an InvGammaPrior")
