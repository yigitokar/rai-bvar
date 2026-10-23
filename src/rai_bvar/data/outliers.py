"""COVID-period residual scaling of Lenza & Primiceri (2022) (SPEC §4.6, §5.6).

``y_t = B'x_t + s_t ε_t`` with one scalar ``s_t`` common to all variables: ``s_t = 1`` before
``t*``; ``s_{t*}, s_{t*+1}, s_{t*+2}`` are free; afterwards the excess over one decays
geometrically, ``s_{t*+j} = 1 + (s_{t*+2} − 1) ρ^{j−2}``. The parameters are hyperparameters
with Pareto(1, 1) priors on the scales (truncated at ``scale_prior.upper``, default 1000) and a
Beta prior (mode 0.8, s.d. 0.2) on ρ.

A parameter that no estimation-sample period depends on (e.g. ``s_{t*+1}`` at a March-2020
forecast origin) is not estimated: under EB it stays at its prior mode, under MH it is drawn
from its prior when forecasting (its posterior equals its prior).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rai_bvar.priors.hyperpriors import BetaPrior, ParetoPrior

COVID_NAMES = ("covid_s0", "covid_s1", "covid_s2", "covid_rho")


@dataclass(frozen=True)
class LenzaPrimiceri:
    """``start`` is ``t*`` in the data's own frequency; by default the period containing March
    2020 (monthly and quarterly data only). ``fixed=(s0, s1, s2, ρ)`` uses given values instead
    of estimating them."""

    start: str | None = None
    fixed: tuple[float, float, float, float] | None = None
    scale_prior: ParetoPrior = field(default_factory=ParetoPrior)
    rho_prior: BetaPrior = field(default_factory=lambda: BetaPrior(mode=0.8, sd=0.2))

    def __post_init__(self) -> None:
        for name, cls in (("scale_prior", ParetoPrior), ("rho_prior", BetaPrior)):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, cls(**value))
            elif not isinstance(value, cls):
                raise TypeError(f"{name} must be a {cls.__name__}")
        if self.start is not None:
            if not isinstance(self.start, str):
                raise TypeError(
                    f"start must be a str such as '2020Q2' or '2020-03', not {type(self.start).__name__}"
                )
            try:
                parsed = pd.Period(self.start)
            except (ValueError, TypeError) as err:
                raise ValueError(f"cannot parse start={self.start!r} as a period") from err
            if pd.isna(parsed):
                raise ValueError(f"cannot parse start={self.start!r} as a period")
        if self.fixed is not None:
            fixed = tuple(float(v) for v in self.fixed)
            if len(fixed) != 4 or not all(np.isfinite(fixed)):
                raise ValueError("fixed must be four finite values (s0, s1, s2, rho)")
            if min(fixed[:3]) < 1.0 or not 0.0 <= fixed[3] < 1.0:
                raise ValueError("fixed must be (s0, s1, s2, rho) with s ≥ 1 and 0 ≤ rho < 1")
            object.__setattr__(self, "fixed", fixed)

    def start_period(self, freq) -> pd.Period:
        code = pd.Period("2020-01", freq=freq).freqstr.split("-")[0].rstrip("SE")
        if code not in ("M", "Q"):
            raise ValueError(
                f"Lenza-Primiceri scaling supports monthly and quarterly data, not {freq}"
            )
        if self.start is not None:
            return pd.Period(self.start, freq=freq)
        return pd.Period("2020-03", freq="M").asfreq(freq)

    def offsets(self, periods: pd.PeriodIndex) -> np.ndarray:
        """``j = t − t*`` for each period."""
        t_star = self.start_period(periods.freq)
        return np.array([(p - t_star).n for p in periods], dtype=int)

    def scales(self, periods: pd.PeriodIndex, params: np.ndarray) -> np.ndarray:
        """``s_t`` for each period; ``params`` has shape ``(..., 4)``, result ``(..., len(periods))``."""
        params = np.asarray(params, dtype=np.float64)
        j = self.offsets(periods)
        s0, s1, s2, rho = (params[..., i, None] for i in range(4))
        decay = 1.0 + (s2 - 1.0) * rho ** np.maximum(j - 2, 0)
        out = np.where(
            j < 0, 1.0, np.where(j == 0, s0, np.where(j == 1, s1, np.where(j == 2, s2, decay)))
        )
        return np.broadcast_to(out, (*params.shape[:-1], len(periods))).copy()

    def active(self, periods: pd.PeriodIndex) -> bool:
        """Whether the estimation sample reaches ``t*`` (otherwise the scaling is inert)."""
        return len(periods) > 0 and periods[-1] >= self.start_period(periods.freq)

    def informed(self, periods: pd.PeriodIndex) -> np.ndarray:
        """Which of ``(s0, s1, s2, ρ)`` some period of the sample depends on."""
        j = self.offsets(periods)
        return np.array([np.any(j == 0), np.any(j == 1), np.any(j >= 2), np.any(j >= 3)])

    def mode(self) -> np.ndarray:
        return np.array([self.scale_prior.scale] * 3 + [self.rho_prior.mode])

    def log_prior_components(self, params: np.ndarray) -> np.ndarray:
        s, rho = params[:3], params[3]
        return np.array([self.scale_prior.logpdf(v) for v in s] + [self.rho_prior.logpdf(rho)])

    def log_prior(self, params: np.ndarray) -> float:
        return float(self.log_prior_components(params).sum())

    def sample_prior(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draws of ``(s0, s1, s2, ρ)`` from their (truncated) priors, shape ``(size, 4)``."""
        scales = [self.scale_prior.sample(size, rng) for _ in range(3)]
        return np.column_stack([*scales, self.rho_prior.sample(size, rng)])

    def bounds(self) -> list[tuple[float, float]]:
        scale = (self.scale_prior.lower, self.scale_prior.upper)
        return [scale, scale, scale, (self.rho_prior.lower, self.rho_prior.upper)]
