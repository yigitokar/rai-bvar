"""``BVAR``: the model façade. A model is a composition of components (SPEC §3 R1, §4).

The engine is chosen by the composition, not by the user (SPEC §4.5): the conjugate Minnesota
prior with a constant inverse-Wishart covariance runs on the analytic engine; every other
composition needs the MCMC engine of Phase 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from rai_bvar.components import Constant, Linear
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import n_deterministic
from rai_bvar.data.outliers import LenzaPrimiceri
from rai_bvar.engines.analytic import AnalyticFit
from rai_bvar.priors.hyperpriors import (
    BetaPrior,
    GammaPrior,
    GLPHyperpriors,
    InvGammaPrior,
    ParetoPrior,
)
from rai_bvar.priors.minnesota import MHSettings, Minnesota


@dataclass(frozen=True)
class BVAR:
    mean: Linear = field(default_factory=Linear)
    vol: Constant = field(default_factory=Constant)
    lags: int = 1
    deterministic: str = "c"
    outliers: LenzaPrimiceri | str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.lags, bool)
            or not isinstance(self.lags, int | np.integer)
            or self.lags < 1
        ):
            raise ValueError("lags must be a positive integer")
        object.__setattr__(self, "lags", int(self.lags))
        n_deterministic(self.deterministic)
        if isinstance(self.outliers, str):
            if self.outliers != "lenza-primiceri":
                raise ValueError("outliers must be None, 'lenza-primiceri' or a LenzaPrimiceri()")
            object.__setattr__(self, "outliers", LenzaPrimiceri())
        elif self.outliers is not None and not isinstance(self.outliers, LenzaPrimiceri):
            raise TypeError("outliers must be None, 'lenza-primiceri' or a LenzaPrimiceri()")
        self.engine  # noqa: B018 — validate the composition at construction time
        prior = self.mean.prior
        if (
            isinstance(self.outliers, LenzaPrimiceri)
            and self.outliers.fixed is None
            and prior.hierarchy == "fixed"
        ):
            raise ValueError(
                "Lenza-Primiceri scaling needs fixed=(s0, s1, s2, rho) with hierarchy='fixed'"
            )

    @property
    def engine(self) -> str:
        if not isinstance(self.mean, Linear) or not isinstance(self.vol, Constant):
            raise NotImplementedError("only Linear × Constant compositions exist so far (Phase 1)")
        prior = self.mean.prior
        if not isinstance(prior, Minnesota):
            raise TypeError("Linear(prior=...) must be a Minnesota prior")
        if prior.form == "conjugate":
            if self.vol.prior != "iw":
                raise ValueError("the conjugate Minnesota prior requires Constant(prior='iw')")
            return "analytic"
        raise NotImplementedError(
            f"Minnesota(form={prior.form!r}) runs on the MCMC engine, which arrives in Phase 2"
        )

    @property
    def name(self) -> str:
        extra = "" if self.deterministic == "c" else f", det={self.deterministic}"
        if self.outliers is not None:
            label = _outlier_label(self.outliers).removeprefix("lenza-primiceri")
            extra += ", COVID" + (f" {label}" if label else "")
        return f"BVAR({self.mean.prior.label}, p={self.lags}{extra})"

    @property
    def axes(self) -> dict[str, str]:
        return {
            "mean": self.mean.name,
            "prior": self.mean.prior.label,
            "vol": self.vol.name,
            "lags": str(self.lags),
            "deterministic": self.deterministic,
            "outliers": "none" if self.outliers is None else _outlier_label(self.outliers),
        }

    def spec(self) -> dict[str, Any]:
        return {"class": "BVAR", **asdict(self)}

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> BVAR:
        """Rebuild a model from :meth:`spec` (the inverse of ``spec()``)."""
        spec = {k: v for k, v in spec.items() if k != "class"}
        prior = dict(spec["mean"]["prior"])
        hp = prior.get("hyperpriors", {})
        prior["hyperpriors"] = GLPHyperpriors(
            **{k: (InvGammaPrior if k == "psi" else GammaPrior)(**v) for k, v in hp.items()}
        )
        prior["mh"] = MHSettings(**prior.get("mh", {}))
        for key in ("psi", "hyper"):
            if isinstance(prior.get(key), list):
                prior[key] = tuple(prior[key])
        outliers = spec.get("outliers")
        if isinstance(outliers, dict):
            outliers = LenzaPrimiceri(
                start=outliers["start"],
                fixed=None if outliers["fixed"] is None else tuple(outliers["fixed"]),
                scale_prior=ParetoPrior(**outliers["scale_prior"]),
                rho_prior=BetaPrior(**outliers["rho_prior"]),
            )
        return cls(
            mean=Linear(prior=Minnesota(**prior)),
            vol=Constant(**spec["vol"]),
            lags=spec["lags"],
            deterministic=spec["deterministic"],
            outliers=outliers,
        )

    def fit(self, data: VARData, *, seed: int | None = None) -> AnalyticFit:
        """Estimate on ``data``. ``seed`` overrides the MH seed of ``hierarchy="mh"``; the seed in
        use is recorded in forecast attrs (``mh_seed``)."""
        if self.engine == "analytic":
            return AnalyticFit(self, data, seed)
        raise AssertionError("unreachable")


def _outlier_label(outliers: LenzaPrimiceri) -> str:
    default = LenzaPrimiceri()
    changed = [
        f.name
        for f in fields(LenzaPrimiceri)
        if getattr(outliers, f.name) != getattr(default, f.name)
    ]
    return "lenza-primiceri" + (
        f"({','.join(f'{c}={getattr(outliers, c)}' for c in changed)})" if changed else ""
    )
