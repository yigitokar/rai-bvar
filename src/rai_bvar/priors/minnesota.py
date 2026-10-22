"""The Minnesota prior specification (SPEC §4.4, §4.6).

In conjugate form (Kadiyala & Karlsson 1997; GLP 2015):

* ``Σ ~ IW(Ψ, d)`` with ``Ψ = diag(ψ)`` and ``d = n + 2``;
* ``vec(B) | Σ ~ N(vec(b), Σ ⊗ Ω)`` with Ω diagonal: the coefficient on lag ``l`` of variable
  ``j`` has ``Ω = λ² / (l^α · ψ_j / (d − n − 1))``; deterministic terms have ``var_const``;
* ``b`` is zero except the own first lag of each variable, which has prior mean ``mean``
  (default from the transformation codes: 1 for persistent levels, 0 for differences);
* optional sum-of-coefficients (``soc = μ``) and single-unit-root (``sur = δ``) dummy
  observations built from the mean of the first ``p`` observations. With
  ``soc_scope="persistent"`` (default) SOC rows exist only for variables whose own-lag prior
  mean is non-zero, so differenced variables are not pulled towards a unit root;
  ``soc_scope="all"`` builds them for every variable (GLP 2015, R ``BVAR``, BoE ``bvar``).

The Kronecker structure forces equal tightness on own and cross lags, so ``cross`` must be 1
in conjugate form.

Phase 2 (DESIGN §6.1–6.2, §11.1; D-10, D-11, D-19):

* ``form="auto"`` (default) is resolved by ``BVAR``: ``"conjugate"`` when the volatility is a
  ``Constant`` with prior ``"auto"``/``"iw"`` (the analytic engine), ``"structural"`` otherwise.
  The resolved form is stored, so Phase 1 models keep their specs. An unresolved ``auto`` form is
  labelled by its analytic resolution (``conjugate``), so Phase 1 labels are unchanged; it admits
  ``cross = contemp = 1`` only (choose ``"structural"`` or ``"reduced"`` explicitly otherwise).
* ``contemp`` (``κ_a``) scales the prior variance of the contemporaneous coefficients,
  ``a_ij ~ N(0, κ_a ψ_i/(ι ψ_j))`` in Mode I (N05 §4.1–4.4); it must be 1 in conjugate form.
* ``cross`` (``κ_c``) scales the prior **standard deviation** of cross lags in the structural and
  reduced forms (N05 §2; N04 §3.1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rai_bvar.priors.hyperpriors import GLPHyperpriors

HYPER_NAMES = ("lambda", "alpha", "soc", "sur", "psi")


@dataclass(frozen=True)
class MHSettings:
    """Random-walk Metropolis-Hastings over hyperparameters (``hierarchy="mh"``).

    A run counts as converged when every free hyperparameter has rank-normalised split
    R-hat below ``rhat_max`` and bulk ESS above ``ess_min`` (SPEC §10.6).
    """

    draws: int = 2500
    burn: int = 1000
    chains: int = 4
    thin: int = 1
    target_acceptance: float = 0.3
    seed: int = 0
    rhat_max: float = 1.01
    ess_min: float = 400.0

    def __post_init__(self) -> None:
        for name in ("draws", "burn", "chains", "thin", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"MHSettings.{name} must be an int")
            object.__setattr__(self, name, int(value))
        if self.draws < 1 or self.chains < 1 or self.thin < 1 or self.burn < 0 or self.seed < 0:
            raise ValueError("MHSettings needs draws ≥ 1, chains ≥ 1, thin ≥ 1, burn ≥ 0, seed ≥ 0")
        if not 0.0 < self.target_acceptance < 1.0:
            raise ValueError("target_acceptance must lie in (0, 1)")
        for name in ("target_acceptance", "rhat_max", "ess_min"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not (self.rhat_max > 1.0 and self.ess_min >= 0):
            raise ValueError("rhat_max must exceed 1 and ess_min must be non-negative")


@dataclass(frozen=True)
class Minnesota:
    lambda_: float = 0.2
    alpha: float = 2.0
    psi: Literal["ar"] | tuple[float, ...] = "ar"
    soc: float | None = None
    sur: float | None = None
    soc_scope: Literal["persistent", "all"] = "persistent"
    mean: float | Mapping[str, float] | None = None
    cross: float = 1.0
    contemp: float = 1.0
    var_const: float = 1e7
    form: Literal["auto", "conjugate", "structural", "reduced"] = "auto"
    hierarchy: Literal["fixed", "eb", "mh"] = "eb"
    hyper: Literal["auto"] | tuple[str, ...] = "auto"
    hyperpriors: GLPHyperpriors = field(default_factory=GLPHyperpriors)
    mh: MHSettings = field(default_factory=MHSettings)

    def __post_init__(self) -> None:
        for name in ("lambda_", "alpha", "cross", "contemp", "var_const", "soc", "sur"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not np.isfinite(float(value)):
                    raise ValueError(f"{name} must be a finite number")
                object.__setattr__(self, name, float(value))
        if self.soc_scope not in ("persistent", "all"):
            raise ValueError("soc_scope must be 'persistent' or 'all'")
        if self.form not in ("auto", "conjugate", "structural", "reduced"):
            raise ValueError("form must be 'auto', 'conjugate', 'structural' or 'reduced'")
        if self.hierarchy not in ("fixed", "eb", "mh"):
            raise ValueError("hierarchy must be 'fixed', 'eb' or 'mh'")
        if self.form in ("auto", "conjugate"):
            auto = " (form='auto' resolves to it with Constant() volatility)" * (
                self.form == "auto"
            )
            if self.cross != 1.0:
                raise ValueError(
                    f"cross-variable shrinkage (cross != 1) is impossible in conjugate form{auto}; "
                    "use form='structural' or 'reduced'"
                )
            if self.contemp != 1.0:
                raise ValueError(
                    f"contemporaneous shrinkage (contemp != 1) is impossible in conjugate form{auto}; "
                    "use form='structural' or 'reduced'"
                )
        if self.lambda_ <= 0 or self.alpha <= 0 or self.var_const <= 0 or self.contemp <= 0:
            raise ValueError("lambda_, alpha, contemp and var_const must be positive")
        for name in ("soc", "sur"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if not isinstance(self.psi, str):
            psi = tuple(float(v) for v in self.psi)
            if not all(np.isfinite(v) and v > 0 for v in psi):
                raise ValueError("psi values must be positive and finite")
            object.__setattr__(self, "psi", psi)
        elif self.psi != "ar":
            raise ValueError("psi must be 'ar' or a sequence of variances")
        if isinstance(self.mean, Mapping):
            mean = {
                str(k): float(v) for k, v in sorted(self.mean.items(), key=lambda kv: str(kv[0]))
            }
            object.__setattr__(self, "mean", mean)
        elif self.mean is not None:
            object.__setattr__(self, "mean", float(self.mean))
        means = self.mean.values() if isinstance(self.mean, dict) else [self.mean]
        if any(m is not None and not np.isfinite(m) for m in means):
            raise ValueError("prior means must be finite")
        if isinstance(self.mh, Mapping):
            object.__setattr__(self, "mh", MHSettings(**self.mh))
        elif not isinstance(self.mh, MHSettings):
            raise TypeError("mh must be an MHSettings")
        if not isinstance(self.hyperpriors, GLPHyperpriors):
            raise TypeError("hyperpriors must be a GLPHyperpriors")
        if not isinstance(self.hyper, str):
            hyper = tuple(self.hyper)
            unknown = [h for h in hyper if h not in HYPER_NAMES]
            if unknown:
                raise ValueError(
                    f"unknown hyperparameters {unknown}; expected some of {HYPER_NAMES}"
                )
            for name in ("soc", "sur"):
                if name in hyper and getattr(self, name) is None:
                    raise ValueError(f"'{name}' is a hyperparameter but the {name} dummy is off")
            object.__setattr__(self, "hyper", hyper)
        elif self.hyper != "auto":
            raise ValueError("hyper must be 'auto' or a tuple of names")

    @property
    def free(self) -> tuple[str, ...]:
        """Hyperparameters estimated by the hierarchy (empty for ``hierarchy="fixed"``)."""
        if self.hierarchy == "fixed":
            return ()
        if self.hyper != "auto":
            return tuple(h for h in HYPER_NAMES if h in self.hyper)
        return ("lambda", *tuple(h for h in ("soc", "sur") if getattr(self, h) is not None))

    @property
    def label(self) -> str:
        """Form, hierarchy, dummies and every setting that differs from the defaults.

        An unresolved ``form="auto"`` shows as ``conjugate`` (its analytic resolution)."""
        parts = ["conjugate" if self.form == "auto" else self.form, self.hierarchy]
        dummies = "+".join(d.upper() for d in ("soc", "sur") if getattr(self, d) is not None)
        parts += [dummies] if dummies else []
        default = Minnesota()
        for name in (
            "lambda_",
            "alpha",
            "psi",
            "mean",
            "cross",
            "contemp",
            "var_const",
            "hyper",
            "hyperpriors",
            "soc_scope",
        ):
            if getattr(self, name) != getattr(default, name):
                parts.append(f"{name.rstrip('_')}={_short(getattr(self, name))}")
        for name in ("soc", "sur"):
            value = getattr(self, name)
            if value is not None and value != 1.0:
                parts.append(f"{name}={_short(value)}")
        if self.hierarchy == "mh" and self.mh != default.mh:
            parts.append(f"mh={_short(self.mh)}")
        return f"Minnesota({', '.join(parts)})"

    def own_lag_means(self, variables: Sequence[str], defaults: Mapping[str, float]) -> list[float]:
        if self.mean is None:
            return [float(defaults[v]) for v in variables]
        if isinstance(self.mean, Mapping):
            unknown = sorted(set(self.mean) - set(variables))
            if unknown:
                raise KeyError(f"prior means given for unknown variables {unknown}")
            return [float(self.mean.get(v, defaults[v])) for v in variables]
        return [float(self.mean)] * len(variables)


def _short(value) -> str:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return "(" + ",".join(_short(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_short(v)}" for k, v in value.items()) + "}"
    if hasattr(value, "__dataclass_fields__"):
        return hashlib.sha256(repr(value).encode()).hexdigest()[:6]
    return str(value)
