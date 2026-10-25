"""User-facing component specifications along the ablation axes (SPEC §4.4; DESIGN §6.3, §11).

Pure Python (no JAX): these frozen dataclasses only describe a model. The MCMC kernels that
implement them live under ``components/{structure,mean,coef,vol}/`` and are loaded lazily by
``engines/mcmc/registry.py`` (DESIGN §1.1 rule 5).

* ``Linear``: linear conditional mean ``g_i(x_t) = x_t'γ_i`` with a Minnesota-type prior.
* ``Constant``: constant error variances. ``prior="auto"`` (default) resolves in ``BVAR`` to
  ``"iw"`` (conjugate inverse-Wishart, analytic engine) or ``"ig"`` (independent inverse-gamma
  ``λ_i``, Mode I); ``"iw"`` with a structural mean is Mode II (λ-scaled NIG, the NIW twin);
  ``"iw_equivalent"`` is Chan's (2022, Prop. 1) ``(a, λ)`` prior, whose implied ``Σ`` is
  ``IW(diag ψ, n + 2)`` at ``contemp = 1`` (DESIGN §6.1, §11.1; N08 §0).
* ``SV``: stochastic volatility ``h_t = log λ_t``, AR(1) (``φ ∈ (−1, 1)``) or random walk, with
  optional outlier and Student-t transitory scales (DESIGN §6.3–6.4; N02, N03).
* ``CommonSV``: one stochastic volatility factor shared by every variance,
  ``λ_{i,t} = λ_i e^{h_t}``, ``h_t`` a stationary AR(1) with mean 0 (P3 DESIGN §5; Carriero,
  Clark & Marcellino 2016; Chan 2020).

``Linear(prior=...)`` takes a ``Minnesota`` prior, a global-local prior (``Horseshoe``,
``NormalGamma``, ``R2D2``, ``DirichletLaplace``, ``SSVS``) or ``AsymmetricConjugate`` (P3 §7.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import numpy as np

from rai_bvar.priors.asymmetric import AsymmetricConjugate
from rai_bvar.priors.global_local import GlobalLocalPrior
from rai_bvar.priors.minnesota import Minnesota, _short
from rai_bvar.priors.sv import (
    Beta,
    DiscreteUniform,
    InvGamma,
    Normal,
    OutlierPrior,
    ScaledChi2,
    TruncatedNormal,
    prior_from_spec,
    prior_to_spec,
)

CONSTANT_PRIORS = ("auto", "iw", "ig", "iw_equivalent")


@dataclass(frozen=True)
class Linear:
    """Linear conditional mean ``g(x_t) = B'x_t`` with a Minnesota, global-local or asymmetric
    conjugate prior."""

    prior: Minnesota | GlobalLocalPrior | AsymmetricConjugate = field(default_factory=Minnesota)

    @property
    def name(self) -> str:
        return "Linear"

    @property
    def label(self) -> str:
        """Name plus prior, so grids over priors get distinct labels."""
        return f"Linear[{self.prior.label}]"


@dataclass(frozen=True)
class Constant:
    """Constant error variances ``λ_i`` (DESIGN §6.1, §11.1).

    ``"auto"`` resolves in ``BVAR``: ``"iw"`` with the conjugate Minnesota form, ``"ig"``
    otherwise. ``"iw"``: conjugate inverse-Wishart (analytic) or, with a structural mean, Mode II
    (``λ_i ~ IG(1 + ι/2, ψ_i/2)`` and every coefficient λ-scaled). ``"ig"``: Mode I, ``λ_i`` an
    independent ``IG(1 + ι/2, ψ_i/2)``. ``"iw_equivalent"``: only ``a_i`` is λ-scaled (Chan 2022,
    Prop. 1), either form.
    """

    prior: Literal["auto", "iw", "ig", "iw_equivalent"] = "auto"

    def __post_init__(self) -> None:
        if self.prior not in CONSTANT_PRIORS:
            raise ValueError(
                f"Constant prior must be 'auto', 'iw_equivalent', 'iw' or 'ig', got {self.prior!r}"
            )

    @property
    def name(self) -> str:
        """Axis value (DESIGN §11.4): ``"Constant"`` for ``"iw"`` and ``"ig"`` (Phase 1 unchanged)."""
        return "Constant(iw_equivalent)" if self.prior == "iw_equivalent" else "Constant"

    @property
    def label(self) -> str:
        """Grid-entry label: ``name``, but ``"Constant(ig)"`` for Mode I, so a grid over
        ``[Constant("ig"), Constant("iw")]`` names its Mode I and Mode II arms apart (their
        ``axes`` differ on ``prior``, R2); ``"auto"``/``"iw"`` keep the Phase 1 label."""
        return "Constant(ig)" if self.prior == "ig" else self.name


# ---------------------------------------------------------------------------------------- SV

_NESTED = {
    "mu_prior": (Normal,),
    "phi_prior": (Beta,),
    "sigma2_prior": (ScaledChi2, InvGamma),
    "outlier_prior": (OutlierPrior, Beta),
    "nu_prior": (DiscreteUniform,),
}
_PHI_PRIOR = Beta(20.0, 1.5)  # (φ + 1)/2: prior mean φ ≈ 0.86 (N02 §10.2; D-13)
_NU_PRIOR = DiscreteUniform(3, 40)  # ν grid {3, …, 40} (JPR 2004; CCMM 2024; D-8)
_BOOLS = ("rw", "outliers", "exact", "asis")
_CHOICES = {
    "tails": ("normal", "t"),
    "t_scale": ("standardized", "unit"),
    "mixture": ("ocsn10", "ksc7"),
}


@dataclass(frozen=True)
class SV:
    """Stochastic volatility for each structural equation (DESIGN §6.3–6.4, D-1…D-9, D-13).

    ``h_t = μ + φ(h_{t−1} − μ) + σ η_t`` (``rw=True``: ``h_t = h_{t−1} + σ η_t``), with
    stochvol's parametrisation: ``μ ~ N(b_μ, B_μ)``, ``(φ+1)/2 ~ Beta(a_φ, b_φ)``,
    ``σ² ~ B_σ·χ²₁`` (N02 §1, §10.2).

    * ``mu_prior="data"``: ``μ_i ~ N(log(ψ_i/ι), mu_var)``, centred per window on the Mode I
      variance scale (R9; open question Q3); a ``Normal`` gives fixed values (stochvol parity).
    * ``sigma2_prior=None``: ``ScaledChi2(1.0)``, or ``ScaledChi2(0.1)`` with ``rw=True``.
    * ``h0="stationary"``: ``h_0 ~ N(μ, σ²/(1 − φ²))``; a float ``B₀`` gives ``N(μ, B₀σ²)``.
      With ``rw=True``: ``h_0 ~ N(log(ψ_i/ι), rw_h0_var)``.
    * ``outliers=True``: ``e_t = o_t ẽ_t``, ``o_t ∈ {1, …, outlier_max}``, ``P(o_t > 1) = p``;
      ``outlier_prior=None`` means ``OutlierPrior(every_years=4, prior_years=10)``
      (``every_years=10`` with ``tails="t"``), resolved with the data frequency (N03 §3.1).
    * ``tails="t"``: ``ẽ_t = √τ_t ε_t``, ``τ_t ~ IG(ν/2, (ν − 2)/2)`` (``t_scale="standardized"``,
      ``E τ = 1``) or ``IG(ν/2, ν/2)`` (``"unit"``), ν on the grid of ``nu_prior`` (D-8).
    * ``offset="relative"``: ``log(ẽ² + c_i)`` with ``c_i = 1e-12·ψ_i`` (D-2); a float is an
      absolute offset (``0.0``: stochvol/bayesianVARs parity). ``exact=True`` adds the
      mixture-error MH correction (D-1; ``False`` for parity); ``asis`` toggles the ASIS step.

    Settings that a variant does not use are ignored (DESIGN §6.3, §3.8: ``rw`` ignores
    ``mu_prior``, ``mu_var``, ``phi_prior`` and ``h0``; ``rw_h0_var`` needs ``rw``; ``mu_var``
    applies to ``mu_prior="data"``; ``outlier_*`` need ``outliers``; ``nu_prior`` and ``t_scale``
    need ``tails="t"``): they are validated, then reset to their defaults. ``sigma2_prior`` and
    ``outlier_prior`` equal to the variant's default are stored as ``None``. So one model has one
    spec, label and hash.
    """

    rw: bool = False
    outliers: bool = False
    tails: Literal["normal", "t"] = "normal"
    mu_prior: Normal | Literal["data"] = "data"
    mu_var: float = 10.0
    phi_prior: Beta = _PHI_PRIOR
    sigma2_prior: ScaledChi2 | InvGamma | None = None
    h0: Literal["stationary"] | float = "stationary"
    rw_h0_var: float = 4.0
    outlier_max: int = 20
    outlier_prior: OutlierPrior | Beta | None = None
    nu_prior: DiscreteUniform = _NU_PRIOR
    t_scale: Literal["standardized", "unit"] = "standardized"
    mixture: Literal["ocsn10", "ksc7"] = "ocsn10"
    offset: Literal["relative"] | float = "relative"
    exact: bool = True
    asis: bool = True

    def __post_init__(self) -> None:
        for name in _BOOLS:
            value = getattr(self, name)
            if not isinstance(value, bool | np.bool_):
                raise TypeError(f"SV.{name} must be a bool, got {value!r}")
            object.__setattr__(self, name, bool(value))
        for name, options in _CHOICES.items():
            if getattr(self, name) not in options:
                raise ValueError(f"SV.{name} must be one of {options}, got {getattr(self, name)!r}")
        for name, classes in _NESTED.items():
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = prior_from_spec(dict(value))
                object.__setattr__(self, name, value)
            allowed = name in ("sigma2_prior", "outlier_prior") and value is None
            allowed |= name == "mu_prior" and value == "data"
            if not (allowed or isinstance(value, classes)):
                kinds = " or ".join(c.__name__ for c in classes)
                raise TypeError(f"SV.{name} must be a {kinds}, got {value!r}")
        for name in ("mu_var", "rw_h0_var"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if isinstance(self.outlier_max, bool) or not isinstance(self.outlier_max, int | np.integer):
            raise TypeError(f"SV.outlier_max must be an int, got {self.outlier_max!r}")
        if self.outlier_max < 2:
            raise ValueError("SV.outlier_max must be at least 2 (outlier scales {1, …, K})")
        object.__setattr__(self, "outlier_max", int(self.outlier_max))
        if self.h0 != "stationary":
            object.__setattr__(self, "h0", _positive("h0", self.h0, "'stationary' or "))
        if self.offset != "relative":
            object.__setattr__(self, "offset", _offset(self.offset))
        for name in _unused(self):
            object.__setattr__(self, name, _FIELD_DEFAULTS[name])  # ignored (DESIGN §6.3, §3.8)
        if self.sigma2_prior == _default_sigma2(self.rw):
            object.__setattr__(self, "sigma2_prior", None)
        if self.outlier_prior == _default_outlier(self.outliers, self.tails):
            object.__setattr__(self, "outlier_prior", None)
        if self.tails == "t" and self.t_scale == "standardized" and self.nu_prior.low <= 2:
            raise ValueError(
                "a standardized Student-t needs ν > 2 (unit variance): use "
                "nu_prior=DiscreteUniform(low ≥ 3, …) or t_scale='unit'"
            )

    # ------------------------------------------------------------------ labels

    @property
    def name(self) -> str:
        """``"SV"``, ``"SV(rw)"``, ``"SV(outliers)"``, ``"SV(t)"``, ``"SV(rw,outliers,t)"``, …"""
        flags = ",".join(_flags(self))
        return f"SV({flags})" if flags else "SV"

    @property
    def label(self) -> str:
        """Name plus every setting that differs from the variant's defaults (nested priors as
        short hashes, as in ``Minnesota.label``)."""
        changed = [
            f"{f.name}={_short(getattr(self, f.name))}"
            for f in fields(self)
            if f.name not in ("rw", "outliers", "tails")
            and getattr(self, f.name) != _FIELD_DEFAULTS[f.name]
        ]
        if not changed:
            return self.name
        flags = ",".join(_flags(self))
        return f"SV({', '.join([flags, *changed] if flags else changed)})"

    # ------------------------------------------------------------------ resolved defaults

    def resolved_sigma2_prior(self) -> ScaledChi2 | InvGamma:
        """The ``σ²`` prior in use: ``ScaledChi2(1.0)`` (AR) or ``ScaledChi2(0.1)`` (RW) if unset."""
        return self.sigma2_prior or _default_sigma2(self.rw)

    def resolved_outlier_prior(self) -> OutlierPrior | Beta | None:
        """The prior on the outlier probability (``None`` without outliers)."""
        if not self.outliers:
            return None
        return self.outlier_prior or _default_outlier(True, self.tails)

    # ------------------------------------------------------------------ spec

    def to_spec(self) -> dict[str, Any]:
        """``{"class": "SV", <fields>}`` with nested priors as ``{"class": …, …}`` (DESIGN §11.5)."""
        spec: dict[str, Any] = {"class": "SV"}
        for f in fields(self):
            value = getattr(self, f.name)
            spec[f.name] = prior_to_spec(value) if f.name in _NESTED and _is_prior(value) else value
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> SV:
        """Inverse of :meth:`to_spec`; missing fields take their defaults."""
        if spec.get("class", "SV") != "SV":
            raise ValueError(f"not an SV spec: {dict(spec)!r}")
        kwargs = {k: v for k, v in spec.items() if k != "class"}
        unknown = sorted(set(kwargs) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown SV fields {unknown}")
        return cls(**kwargs)


_FIELD_DEFAULTS = {f.name: f.default for f in fields(SV)}


def _unused(sv: SV) -> tuple[str, ...]:
    """Fields the variant ignores (reset to their defaults at construction; DESIGN §6.3, §3.8)."""
    return tuple(
        name
        for name, used in (
            ("mu_prior", not sv.rw),
            ("mu_var", not sv.rw and sv.mu_prior == "data"),
            ("phi_prior", not sv.rw),
            ("h0", not sv.rw),
            ("rw_h0_var", sv.rw),
            ("outlier_max", sv.outliers),
            ("outlier_prior", sv.outliers),
            ("nu_prior", sv.tails == "t"),
            ("t_scale", sv.tails == "t"),
        )
        if not used
    )


def _default_sigma2(rw: bool) -> ScaledChi2:
    """``σ² ~ 1·χ²₁`` (AR) or ``0.1·χ²₁`` (RW) (D-13; N02 §8.2, §10.2)."""
    return ScaledChi2(0.1 if rw else 1.0)


def _default_outlier(outliers: bool, tails: str) -> OutlierPrior | None:
    """One outlier every 4 years (SVO) or 10 years (SVO-t), 10 prior years (N03 §0, §3.1)."""
    return OutlierPrior(every_years=10.0 if tails == "t" else 4.0) if outliers else None


def _flags(sv: SV) -> list[str]:
    return [f for f, on in (("rw", sv.rw), ("outliers", sv.outliers), ("t", sv.tails == "t")) if on]


def _is_prior(value: object) -> bool:
    return any(isinstance(value, c) for classes in _NESTED.values() for c in classes)


def _positive(name: str, value: object, prefix: str = "") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"SV.{name} must be {prefix}a positive number, got {value!r}")
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"SV.{name} must be {prefix}a positive finite number, got {value!r}")
    return value


def _offset(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"SV.offset must be 'relative' or a number ≥ 0, got {value!r}")
    value = float(value)
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"SV.offset must be 'relative' or a finite number ≥ 0, got {value!r}")
    return value


# ---------------------------------------------------------------------------------- CommonSV

_CSV_NESTED = {"rho_prior": (TruncatedNormal, Beta), "sigma2_prior": (InvGamma, ScaledChi2)}
_CSV_RHO = TruncatedNormal(0.9, 0.04)  # ρ ~ N(0.9, 0.2²)·1(|ρ| < 1) (D-46; Chan 2020)
_CSV_SIGMA2 = InvGamma(5.0, 0.04)  # σ_h² ~ IG(5, 0.04): prior mean 0.01 (D-46; Chan 2020)


@dataclass(frozen=True)
class CommonSV:
    """Common stochastic volatility (P3 DESIGN §5.1, D-46): ``u_t ~ N(0, e^{h_t} Σ)``,
    ``h_t = ρ h_{t−1} + σ_h η_t``, ``h_0 ~ N(0, σ_h²/(1 − ρ²))`` at the first data row.

    * ``rho_prior``: a ``TruncatedNormal`` on ``(−1, 1)`` or a ``Beta`` on ``(ρ + 1)/2``.
    * ``sigma2_prior``: ``InvGamma(shape, scale)`` or ``ScaledChi2(B)`` (``σ_h² ~ B·χ²₁``).
    * ``block``: length of the blocks of the ``h`` independence MH step (D-48), at least 2.

    With ``Minnesota(form="conjugate")`` the mean takes the shared-factor path at the conjugate
    speed (§5.3); with every other prior each equation keeps its own variance ``λ_i`` (§5.2).
    """

    rho_prior: TruncatedNormal | Beta = _CSV_RHO
    sigma2_prior: InvGamma | ScaledChi2 = _CSV_SIGMA2
    block: int = 100

    def __post_init__(self) -> None:
        for name, classes in _CSV_NESTED.items():
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = prior_from_spec(dict(value))
                object.__setattr__(self, name, value)
            if not isinstance(value, classes):
                kinds = " or ".join(c.__name__ for c in classes)
                raise TypeError(f"CommonSV.{name} must be a {kinds}, got {value!r}")
        rho = self.rho_prior
        if isinstance(rho, TruncatedNormal) and (rho.low, rho.high) != (-1.0, 1.0):
            raise ValueError(
                "CommonSV.rho_prior must be truncated to the stationary region (-1, 1), got "
                f"({rho.low}, {rho.high})"
            )
        if isinstance(self.block, bool) or not isinstance(self.block, int | np.integer):
            raise TypeError(f"CommonSV.block must be an int, got {self.block!r}")
        if self.block < 2:
            raise ValueError(f"CommonSV.block must be at least 2, got {self.block}")
        object.__setattr__(self, "block", int(self.block))

    @property
    def name(self) -> str:
        return "CommonSV"

    @property
    def label(self) -> str:
        """``"CommonSV"`` plus every setting that differs from the defaults (nested priors as
        short hashes, as in ``SV.label``)."""
        changed = [
            f"{f.name}={_short(getattr(self, f.name))}"
            for f in fields(self)
            if getattr(self, f.name) != _CSV_DEFAULTS[f.name]
        ]
        return f"CommonSV({', '.join(changed)})" if changed else "CommonSV"

    def to_spec(self) -> dict[str, Any]:
        """``{"class": "CommonSV", <fields>}`` with nested priors as ``{"class": …, …}``."""
        return {
            "class": "CommonSV",
            "rho_prior": prior_to_spec(self.rho_prior),
            "sigma2_prior": prior_to_spec(self.sigma2_prior),
            "block": self.block,
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> CommonSV:
        """Inverse of :meth:`to_spec`; missing fields take their defaults."""
        if spec.get("class", "CommonSV") != "CommonSV":
            raise ValueError(f"not a CommonSV spec: {dict(spec)!r}")
        kwargs = {k: v for k, v in spec.items() if k != "class"}
        unknown = sorted(set(kwargs) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown CommonSV fields {unknown}")
        return cls(**kwargs)


_CSV_DEFAULTS = {f.name: f.default for f in fields(CommonSV)}
