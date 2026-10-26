"""Global-local and selection priors on the lag coefficients (P3 DESIGN §3, §7.1; D-36 … D-44).

Pure Python (no JAX): these frozen dataclasses only describe a prior. Each governs the
**deviation** ``d_j = coefficient − m_j`` of every lag coefficient from its transform-aware prior
mean (D-36), with kernel variance ``r_j v_j``: ``r_j`` puts the prior in the units of the Mode I
Minnesota prior at ``λ = 1`` (``scale="psi"``, D-37) or is 1 (``scale="none"``), and ``v_j`` is
the family's mixing variance (§3.2). Deterministic terms and the contemporaneous ``a_i`` keep the
Phase 2 Normal priors of the form; GL priors have no SOC/SUR dummy rows and no EB.

* ``Horseshoe(global_scale)``: ``v_j = ϑ_j ζ_g``, ``√ϑ_j ~ C⁺(0, 1)``, ``√ζ_g ~ C⁺(0, s)``.
* ``NormalGamma(a, b, c)``: ``ϑ_j ~ G(a_g, a_g/2)``, ``ζ_g⁻¹ ~ G(b, c)`` (``c="a/2"`` ties ``c``).
* ``R2D2(a, b)``: ``v_j = ϱ_j ϖ_j/2``, ``ϱ_j ~ Exp(½)``, ``ϖ_j | ξ_g ~ G(a_g, ξ_g)``, ``ξ_g ~ G(b, 1)``.
* ``DirichletLaplace(a)``: Laplace kernel with scale ``θ̃_j ~ G(a_g, ½)``.
* ``SSVS(c0, c1, semiautomatic, p)``: spike-and-slab, ``γ_j ~ Bernoulli(p_g)``.

``a`` is a fixed float or a :class:`DiscreteGrid` hyperprior (default: Gruber & Kastner's grid);
every group has its own ``a_g``. The fields after the family's own (``groups``, ``scale``,
``form``, ``mean``, ``psi``, ``var_const``, ``contemp``) are keyword-only and shared by the five
families. ``groups`` (D-38): ``"olcl-lagwise"`` (default; ``2p`` groups: own lags of lag ``l``
across equations, cross lags of lag ``l`` across equations), ``"global"``, ``"equation-wise"``,
``"covariate-wise"`` (regressor ``(l, m)`` across equations) or an explicit ``(n, n·p)`` array of
non-negative group ids (columns in the design order: the ``n`` variables at lag 1, then lag 2, …).
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

import numpy as np

from rai_bvar.priors.minnesota import Minnesota, _short
from rai_bvar.priors.sv import Beta, prior_from_spec, prior_to_spec

GROUPS = ("olcl-lagwise", "global", "equation-wise", "covariate-wise")
SCALES = ("psi", "none")
FORMS = ("auto", "structural", "reduced")
_GK_POINTS = 1000  # Gruber & Kastner (2025): 1,000 points on [0.001, 1], weights ∝ Exp(1) density


@dataclass(frozen=True)
class DiscreteGrid:
    """A discrete prior on ``support`` with probabilities ``weights`` (normalised at construction).

    Both are stored as tuples of floats, so a grid is hashable and equal grids have equal specs.
    Weights that already sum to 1 (within 1e-12) are kept as given, so a spec round trip is exact.
    """

    support: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        support = _floats(self.support, "DiscreteGrid.support")
        weights = _floats(self.weights, "DiscreteGrid.weights")
        if not support or len(support) != len(weights):
            raise ValueError(
                "DiscreteGrid needs non-empty support and weights of equal length, got "
                f"{len(support)} and {len(weights)}"
            )
        if not all(w > 0 for w in weights):
            raise ValueError("DiscreteGrid.weights must be positive")
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-12:
            weights = tuple(w / total for w in weights)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def gruber_kastner(cls) -> DiscreteGrid:
        """Gruber & Kastner's (2025) grid: 1,000 equally spaced points on ``[0.001, 1]`` with
        weights proportional to the ``Exp(1)`` density (P3 DESIGN §3.2)."""
        return _gruber_kastner()

    @property
    def size(self) -> int:
        return len(self.support)

    def to_spec(self) -> dict[str, Any]:
        """``{"class": "DiscreteGrid", "support": [...], "weights": [...]}``, or ``{"class":
        "DiscreteGrid", "name": "gruber_kastner"}`` for the default grid (P3 §7.4, E68)."""
        if self == _gruber_kastner():
            return {"class": "DiscreteGrid", "name": "gruber_kastner"}
        return {
            "class": "DiscreteGrid",
            "support": list(self.support),
            "weights": list(self.weights),
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> DiscreteGrid:
        if spec.get("class") != "DiscreteGrid":
            raise ValueError(f"not a DiscreteGrid spec: {dict(spec)!r}")
        if "name" in spec:
            if spec["name"] != "gruber_kastner" or set(spec) != {"class", "name"}:
                raise ValueError(f"unknown named DiscreteGrid spec: {dict(spec)!r}")
            return _gruber_kastner()
        return cls(tuple(spec["support"]), tuple(spec["weights"]))


@functools.cache
def _gruber_kastner() -> DiscreteGrid:
    support = np.linspace(0.001, 1.0, _GK_POINTS)
    weights = np.exp(-support)
    return DiscreteGrid(tuple(map(float, support)), tuple(map(float, weights / weights.sum())))


# ---------------------------------------------------------------------------------- families


@dataclass(frozen=True)
class GlobalLocalPrior:
    """Fields shared by the five families (keyword-only); see the module docstring."""

    groups: str | tuple[tuple[int, ...], ...] = field(default="olcl-lagwise", kw_only=True)
    scale: Literal["psi", "none"] = field(default="psi", kw_only=True)
    form: Literal["auto", "structural", "reduced"] = field(default="auto", kw_only=True)
    mean: float | Mapping[str, float] | None = field(default=None, kw_only=True)
    psi: Literal["ar"] | tuple[float, ...] = field(default="ar", kw_only=True)
    var_const: float = field(default=1e7, kw_only=True)
    contemp: float = field(default=1.0, kw_only=True)

    kind: ClassVar[str]  # the engine's CoefPriorStatic.kind
    _OWN: ClassVar[tuple[str, ...]]  # the family's own fields, in label order

    def __post_init__(self) -> None:
        if self.form == "conjugate":
            raise ValueError(
                "global-local priors are not conjugate; use form='structural' (default) or "
                "'reduced'"
            )
        if self.form not in FORMS:
            raise ValueError(f"{self._name}.form must be one of {FORMS}, got {self.form!r}")
        if self.scale not in SCALES:
            raise ValueError(f"{self._name}.scale must be one of {SCALES}, got {self.scale!r}")
        object.__setattr__(self, "groups", _groups(self.groups, self._name))
        # ψ, the own-lag means, var_const and contemp mean what they mean for the Minnesota prior
        # of the form (P3 §3.6): validate and normalise them through it
        internal = self.minnesota()
        for name in ("psi", "mean", "var_const", "contemp"):
            object.__setattr__(self, name, getattr(internal, name))

    @property
    def _name(self) -> str:
        return type(self).__name__

    @property
    def resolved_form(self) -> Literal["structural", "reduced"]:
        """``form="auto"`` is ``"structural"`` (D-43)."""
        return "reduced" if self.form == "reduced" else "structural"

    @property
    def a_grid(self) -> int:
        """Size of the discrete hyperprior on ``a`` (0: a fixed ``a`` or no ``a``)."""
        a = getattr(self, "a", None)
        return a.size if isinstance(a, DiscreteGrid) else 0

    @property
    def p_beta(self) -> bool:
        """Whether SSVS ``p`` has a Beta hyperprior."""
        return isinstance(getattr(self, "p", None), Beta)

    def minnesota(self) -> Minnesota:
        """The internal Minnesota prior of the form that supplies ψ, the own-lag means, the
        deterministic and the ``a_i`` priors (P3 §3.6); it never reaches the user or a spec."""
        return Minnesota(
            form=self.resolved_form,
            psi=self.psi,
            mean=self.mean,
            var_const=self.var_const,
            contemp=self.contemp,
            hierarchy="fixed",
        )

    def group_ids(self, n: int, lags: int) -> np.ndarray:
        """The ``(n, n·lags)`` int32 group id of every lag coefficient (row: equation; column:
        the lag slot in design order), relabelled to ``0 … G − 1`` in sorted order, so no group
        is empty (with ``n = 1``, ``"olcl-lagwise"`` has no cross-lag groups; D-38)."""
        n, lags = int(n), int(lags)
        own = np.arange(n)[:, None] == np.tile(np.arange(n), lags)[None, :]
        lag = np.repeat(np.arange(lags), n)[None, :]
        if isinstance(self.groups, tuple):
            ids = np.asarray(self.groups, dtype=np.int64)
            if ids.shape != (n, n * lags):
                raise ValueError(
                    f"groups array has shape {ids.shape}; expected (n, n·lags) = ({n}, {n * lags})"
                )
        elif self.groups == "olcl-lagwise":
            ids = 2 * lag + (~own).astype(np.int64)
        elif self.groups == "global":
            ids = np.zeros((n, n * lags), np.int64)
        elif self.groups == "equation-wise":
            ids = np.broadcast_to(np.arange(n)[:, None], (n, n * lags))
        else:  # covariate-wise
            ids = np.broadcast_to(np.arange(n * lags)[None, :], (n, n * lags))
        return np.unique(ids, return_inverse=True)[1].reshape(n, n * lags).astype(np.int32)

    def n_groups(self, n: int, lags: int) -> int:
        """``G``, the number of (non-empty) groups for ``n`` variables and ``lags`` lags."""
        return int(self.group_ids(n, lags).max()) + 1

    @property
    def groups_label(self) -> str:
        if isinstance(self.groups, str):
            return self.groups
        return "custom-" + hashlib.sha256(repr(self.groups).encode()).hexdigest()[:6]

    @property
    def label(self) -> str:
        """``"Horseshoe(structural, olcl-lagwise)"`` plus every setting that differs from the
        family's defaults (nested values as short hashes, as in ``Minnesota.label``)."""
        default = type(self)()
        parts = [self.resolved_form, self.groups_label]
        for name in (*self._OWN, "scale", "psi", "mean", "var_const", "contemp"):
            if getattr(self, name) != getattr(default, name):
                parts.append(f"{name}={_short(getattr(self, name))}")
        return f"{self._name}({', '.join(parts)})"

    def to_spec(self) -> dict[str, Any]:
        """``{"class": <family>, <fields>}`` in plain JSON (P3 §7.4)."""
        spec: dict[str, Any] = {"class": self._name}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, DiscreteGrid):
                value = value.to_spec()
            elif isinstance(value, Beta):
                value = prior_to_spec(value)
            elif f.name == "groups" and isinstance(value, tuple):
                value = [list(row) for row in value]
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, dict):
                value = dict(value)
            spec[f.name] = value
        return spec

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> GlobalLocalPrior:
        """Inverse of :meth:`to_spec` for any family (dispatch on ``"class"``)."""
        family = FAMILIES.get(spec.get("class")) if isinstance(spec, Mapping) else None
        if family is None:
            raise ValueError(f"not a global-local prior spec: {spec!r}")
        kwargs = {k: v for k, v in spec.items() if k != "class"}
        unknown = sorted(set(kwargs) - {f.name for f in fields(family)})
        if unknown:
            raise ValueError(f"unknown {family.__name__} fields {unknown}")
        for name, value in kwargs.items():
            if isinstance(value, Mapping) and value.get("class") == "DiscreteGrid":
                kwargs[name] = DiscreteGrid.from_spec(value)
            elif isinstance(value, Mapping) and name == "p":
                kwargs[name] = prior_from_spec(dict(value))
            elif name == "groups" and isinstance(value, list):
                kwargs[name] = tuple(tuple(row) for row in value)
            elif name == "psi" and isinstance(value, list):
                kwargs[name] = tuple(value)
        return family(**kwargs)


@dataclass(frozen=True)
class Horseshoe(GlobalLocalPrior):
    """Horseshoe (Carvalho, Polson & Scott 2010): ``√ζ_g ~ C⁺(0, global_scale)``."""

    global_scale: float = 1.0

    kind: ClassVar[str] = "horseshoe"
    _OWN: ClassVar[tuple[str, ...]] = ("global_scale",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_scale", _positive(self, "global_scale"))
        super().__post_init__()


@dataclass(frozen=True)
class NormalGamma(GlobalLocalPrior):
    """Normal-Gamma (Griffin & Brown 2010; GK's parametrisation): ``ϑ_j ~ G(a_g, a_g/2)``,
    ``ζ_g⁻¹ ~ G(b, c)``; ``c="a/2"`` ties ``c`` to ``a`` (GK's default)."""

    a: float | DiscreteGrid = field(default_factory=DiscreteGrid.gruber_kastner)
    b: float = 0.5
    c: float | Literal["a/2"] = "a/2"

    kind: ClassVar[str] = "normal_gamma"
    _OWN: ClassVar[tuple[str, ...]] = ("a", "b", "c")

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _shape_a(self))
        object.__setattr__(self, "b", _positive(self, "b"))
        if self.c != "a/2":
            object.__setattr__(self, "c", _positive(self, "c", "or 'a/2'"))
        super().__post_init__()


@dataclass(frozen=True)
class R2D2(GlobalLocalPrior):
    """R2D2 (Zhang, Naughton, Bondell & Reich 2022; GK's gamma form): ``R² ~ Beta(n_g a, b)``
    in each group."""

    a: float | DiscreteGrid = field(default_factory=DiscreteGrid.gruber_kastner)
    b: float = 0.5

    kind: ClassVar[str] = "r2d2"
    _OWN: ClassVar[tuple[str, ...]] = ("a", "b")

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _shape_a(self))
        object.__setattr__(self, "b", _positive(self, "b"))
        super().__post_init__()


@dataclass(frozen=True)
class DirichletLaplace(GlobalLocalPrior):
    """Dirichlet–Laplace (Bhattacharya, Pati, Pillai & Dunson 2015; gamma form): Laplace kernel
    with scale ``θ̃_j ~ G(a_g, ½)``."""

    a: float | DiscreteGrid = field(default_factory=DiscreteGrid.gruber_kastner)

    kind: ClassVar[str] = "dirichlet_laplace"
    _OWN: ClassVar[tuple[str, ...]] = ("a",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _shape_a(self))
        super().__post_init__()


@dataclass(frozen=True)
class SSVS(GlobalLocalPrior):
    """Stochastic search variable selection (George & McCulloch 1993; George, Sun & Ni 2008).

    Spike and slab s.d. ``τ_0 = c0·ŝ_j``, ``τ_1 = c1·ŝ_j`` with ``ŝ_j`` the flat-prior posterior
    s.d. of the coefficient on the window (``semiautomatic=True``, D-44), else ``c·√r_j``;
    ``p`` a fixed inclusion probability in (0, 1) or a ``sv_priors.Beta(s1, s2)`` hyperprior.
    """

    c0: float = 0.01
    c1: float = 100.0
    semiautomatic: bool = True
    p: float | Beta = 0.5

    kind: ClassVar[str] = "ssvs"
    _OWN: ClassVar[tuple[str, ...]] = ("c0", "c1", "semiautomatic", "p")

    def __post_init__(self) -> None:
        if not isinstance(self.semiautomatic, bool | np.bool_):
            raise TypeError(f"SSVS.semiautomatic must be a bool, got {self.semiautomatic!r}")
        object.__setattr__(self, "semiautomatic", bool(self.semiautomatic))
        if isinstance(self.p, Mapping):
            object.__setattr__(self, "p", prior_from_spec(dict(self.p)))
        c0, c1, p = _finite(self.c0), _finite(self.c1), self.p
        if not isinstance(p, Beta):
            p = _finite(p)
        if (
            c0 is None
            or c1 is None
            or p is None
            or not (0 < c0 < c1)
            or (not isinstance(p, Beta) and not 0 < p < 1)
        ):
            raise ValueError("SSVS needs 0 < c0 < c1 and p in (0, 1) or a Beta prior")
        object.__setattr__(self, "c0", c0)
        object.__setattr__(self, "c1", c1)
        object.__setattr__(self, "p", p)
        super().__post_init__()


FAMILIES: dict[str, type[GlobalLocalPrior]] = {
    cls.__name__: cls for cls in (Horseshoe, NormalGamma, R2D2, DirichletLaplace, SSVS)
}


# ---------------------------------------------------------------------------------- helpers


def _floats(values: Any, what: str) -> tuple[float, ...]:
    try:
        values = list(values)
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{what} must be a sequence of numbers, got {values!r}") from err
    if any(isinstance(v, bool) for v in values) or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must hold finite numbers")
    return out


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _positive(prior: GlobalLocalPrior, name: str, alternative: str = "") -> float:
    value = _finite(getattr(prior, name))
    if value is None or value <= 0:
        extra = f" {alternative}" if alternative else ""
        raise ValueError(
            f"{type(prior).__name__}.{name} must be a positive number{extra}, got "
            f"{getattr(prior, name)!r}"
        )
    return value


def _shape_a(prior: GlobalLocalPrior) -> float | DiscreteGrid:
    a = prior.a  # type: ignore[attr-defined]
    if isinstance(a, Mapping):
        a = DiscreteGrid.from_spec(a)
    if isinstance(a, DiscreteGrid):
        if all(0 < x <= 1 for x in a.support):
            return a
    elif (value := _finite(a)) is not None and value > 0:
        return value
    raise ValueError(
        f"{type(prior).__name__}.a must be a positive number (a fixed a) or a DiscreteGrid with "
        f"every point in (0, 1], got {_short(a) if isinstance(a, DiscreteGrid) else repr(a)}"
    )


def _groups(groups: Any, owner: str) -> str | tuple[tuple[int, ...], ...]:
    if isinstance(groups, str):
        if groups in GROUPS:
            return groups
    else:
        try:
            arr = np.asarray(groups)
        except (TypeError, ValueError):
            arr = None
        if (
            arr is not None
            and arr.ndim == 2
            and arr.size > 0
            and arr.dtype.kind in "iu"
            and (arr >= 0).all()
        ):
            return tuple(tuple(int(v) for v in row) for row in arr)
    raise ValueError(
        f"{owner}.groups must be one of {GROUPS} or a rectangular (n, n·lags) array of "
        f"non-negative integer group ids, got {groups!r}"
    )


__all__ = [
    "FAMILIES",
    "GROUPS",
    "R2D2",
    "SSVS",
    "DirichletLaplace",
    "DiscreteGrid",
    "GlobalLocalPrior",
    "Horseshoe",
    "NormalGamma",
]
