"""The asymmetric conjugate prior of Chan (2022) (P3 DESIGN §4, D-45).

Per equation ``i`` of the recursive structural form, in δ-coordinates with the D3 centring of
P2 §6.1 (Mode II): ``θ_i = [δ_i | a_i] | λ_i ~ N(μ₀, λ_i V₀)``, ``λ_i ~ IG(1 + ι/2, ψ_i/2)``, with
``V₀`` diagonal: own lag ``l`` ``λ²(d−n−1)/(l^α ψ_i)``, lag ``l`` of variable ``m ≠ i``
``λ² cross² (d−n−1)/(l^α ψ_m)``, deterministic ``var_const``, ``a_ij`` ``contemp/ψ_j``. At
``cross = contemp = 1`` it is the Phase 1 NIW prior exactly. Equations are independent a priori,
so the posterior is closed form: the composition runs on the analytic engine
(``engines/structural_nig.py``, WP33) with iid draws and an exact log marginal likelihood.

``hierarchy``: ``"fixed"``; ``"eb"`` (default; free ``lambda`` and ``cross`` plus ``soc``/``sur``
when on, Chan's EB); ``"mh"`` (random-walk MH over the log free hyperparameters, as Phase 1).
``cross`` and ``contemp`` have a flat prior on their logarithm over ``[1e-3, 10]`` when free.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from rai_bvar.priors.hyperpriors import GammaPrior, GLPHyperpriors, InvGammaPrior
from rai_bvar.priors.minnesota import HYPER_NAMES, MHSettings, Minnesota, _short

AC_HYPER_NAMES = (*HYPER_NAMES, "cross", "contemp")
LOG_FLAT_BOX = (1e-3, 10.0)  # flat prior on log cross, log contemp (Chan 2022 §4; P3 §4.3)


@dataclass(frozen=True)
class AsymmetricConjugate:
    lambda_: float = 0.2
    cross: float = 0.5
    contemp: float = 1.0
    alpha: float = 2.0
    psi: Literal["ar"] | tuple[float, ...] = "ar"
    mean: float | Mapping[str, float] | None = None
    var_const: float = 1e7
    soc: float | None = None
    sur: float | None = None
    soc_scope: Literal["persistent", "all"] = "persistent"
    hierarchy: Literal["fixed", "eb", "mh"] = "eb"
    hyper: Literal["auto"] | tuple[str, ...] = "auto"
    hyperpriors: GLPHyperpriors = field(default_factory=GLPHyperpriors)
    mh: MHSettings = field(default_factory=MHSettings)

    def __post_init__(self) -> None:
        for name in ("lambda_", "cross", "contemp"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float | np.integer | np.floating)
                or not (np.isfinite(float(value)) and float(value) > 0)
            ):
                raise ValueError(
                    f"AsymmetricConjugate.{name} must be a positive finite number, got {value!r}"
                )
        if self.hierarchy not in ("fixed", "eb", "mh"):
            raise ValueError("hierarchy must be 'fixed', 'eb' or 'mh'")
        if isinstance(self.mh, Mapping):
            object.__setattr__(self, "mh", MHSettings(**self.mh))
        if not isinstance(self.hyper, str):
            hyper = tuple(self.hyper)
            unknown = [h for h in hyper if h not in AC_HYPER_NAMES]
            if unknown:
                raise ValueError(
                    f"unknown hyperparameters {unknown}; expected some of {AC_HYPER_NAMES}"
                )
            object.__setattr__(self, "hyper", hyper)
        elif self.hyper != "auto":
            raise ValueError("hyper must be 'auto' or a tuple of names")
        # every Minnesota field means what it means in structural form (P3 §4.2): validate and
        # normalise through the internal prior the fit uses
        internal = self.minnesota()
        for name in ("lambda_", "cross", "contemp", "alpha", "psi", "mean", "var_const", "soc",
                     "sur"):  # fmt: skip
            object.__setattr__(self, name, getattr(internal, name))
        for name in ("soc", "sur"):
            if name in self.free_names and getattr(self, name) is None:
                raise ValueError(f"'{name}' is a hyperparameter but the {name} dummy is off")
        free = self.free
        if "psi" in free and ("cross" in free or self.cross != 1.0):
            raise ValueError(
                "empirical Bayes over psi is unavailable with cross-variable shrinkage (Chan 2022 "
                "marginal likelihood over lambda, cross, contemp and alpha only); remove 'psi' "
                "from hyper or fix psi"
            )

    @property
    def form(self) -> Literal["structural"]:
        """Always the structural form (Mode II arrays with ``cross`` and ``contemp``)."""
        return "structural"

    @property
    def free_names(self) -> tuple[str, ...]:
        """The hyperparameters ``hyper`` names (``"auto"``: ``lambda``, ``cross`` and the dummies
        that are on), whatever the hierarchy."""
        if self.hyper != "auto":
            return tuple(h for h in AC_HYPER_NAMES if h in self.hyper)
        dummies = tuple(h for h in ("soc", "sur") if getattr(self, h) is not None)
        return ("lambda", "cross", *dummies)

    @property
    def free(self) -> tuple[str, ...]:
        """Hyperparameters estimated by the hierarchy (empty for ``hierarchy="fixed"``)."""
        return () if self.hierarchy == "fixed" else self.free_names

    def minnesota(self) -> Minnesota:
        """The internal structural Minnesota prior the NIG closed form reads (the adapter of
        P3 §4.2); it never reaches the user or a spec."""
        return Minnesota(
            lambda_=self.lambda_,
            alpha=self.alpha,
            psi=self.psi,
            soc=self.soc,
            sur=self.sur,
            soc_scope=self.soc_scope,
            mean=self.mean,
            cross=self.cross,
            contemp=self.contemp,
            var_const=self.var_const,
            form="structural",
            hierarchy="fixed",
            hyperpriors=self.hyperpriors,
            mh=self.mh,
        )

    @property
    def label(self) -> str:
        """Hierarchy, dummies, every setting that differs from the defaults and, when ``cross`` or
        ``contemp`` is free, their flat log prior (as ``Minnesota.label``)."""
        parts = [self.hierarchy]
        dummies = "+".join(d.upper() for d in ("soc", "sur") if getattr(self, d) is not None)
        parts += [dummies] if dummies else []
        default = AsymmetricConjugate()
        for name in ("lambda_", "cross", "contemp", "alpha", "psi", "mean", "var_const", "hyper",
                     "hyperpriors", "soc_scope"):  # fmt: skip
            if getattr(self, name) != getattr(default, name):
                parts.append(f"{name.rstrip('_')}={_short(getattr(self, name))}")
        for name in ("soc", "sur"):
            value = getattr(self, name)
            if value is not None and value != 1.0:
                parts.append(f"{name}={_short(value)}")
        log_flat = [h for h in ("cross", "contemp") if h in self.free]
        if log_flat:
            low, high = LOG_FLAT_BOX
            parts.append(f"log {'+'.join(log_flat)} flat on [{low:g}, {high:g}]")
        if self.hierarchy == "mh" and self.mh != default.mh:
            parts.append(f"mh={_short(self.mh)}")
        return f"AsymmetricConjugate({', '.join(parts)})"

    def to_spec(self) -> dict[str, Any]:
        """``{"class": "AsymmetricConjugate", <fields>}`` in plain JSON (nested settings as dicts,
        tuples as lists; P3 §7.4)."""
        return {"class": "AsymmetricConjugate", **json.loads(json.dumps(asdict(self)))}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> AsymmetricConjugate:
        if spec.get("class") != "AsymmetricConjugate":
            raise ValueError(f"not an AsymmetricConjugate spec: {dict(spec)!r}")
        kwargs = {k: v for k, v in spec.items() if k != "class"}
        hp = kwargs.get("hyperpriors", {})
        kwargs["hyperpriors"] = GLPHyperpriors(
            **{k: (InvGammaPrior if k == "psi" else GammaPrior)(**v) for k, v in hp.items()}
        )
        kwargs["mh"] = MHSettings(**kwargs.get("mh", {}))
        for key in ("psi", "hyper"):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


__all__ = ["AC_HYPER_NAMES", "LOG_FLAT_BOX", "AsymmetricConjugate"]
