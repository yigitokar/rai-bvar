"""Static configuration of the MCMC engine (DESIGN §2.4, §3.8). Pure Python: imports no JAX.

Every shape- or branch-determining choice is a field of a frozen dataclass with value equality,
so ``StaticConfig`` is the key of every ``lru_cache`` factory (``compose``, ``make_sweep``,
``make_runner``, ``make_forecast``) and, together with the argument shapes, of every jit cache
(DESIGN §4.9). Data, priors, ``T_eff``, ``thin`` and hyperparameter values are never static
(D-23). ``api.py`` calls :func:`validate_composition` when a ``BVAR`` is constructed, so this
module must stay free of JAX (DESIGN §1.1 rule 2).

Dimensions (DESIGN §2.2): ``k = n_det + n·lags`` lagged regressors, ``q = k + n − 1`` padded
structural coefficients ``[δ_i | a_i]``, packed Gram lengths ``P = q(q+1)/2`` and
``P_x = k(k+1)/2``, compiled rows ``T_pad = t_bucket·⌈T_eff/t_bucket⌉``.
"""

from __future__ import annotations

import itertools
import numbers
from dataclasses import dataclass
from typing import Literal, get_args

StructKind = Literal["recursive"]
Scaling = Literal["none", "lambda"]
MeanKind = Literal["linear_structural", "linear_reduced", "zero"]
ConstPath = Literal["weighted", "cholesky_const", "eigen", "fixed_factor", "shared_factor"]
GramKind = Literal["packed", "packed_tiled", "batched"]
CoefKind = Literal[
    "fixed_diagonal", "horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"
]  # fmt: skip
GLKinds = ("horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs")  # Phase 3 (P3 D-39)
RhoPrior = Literal["truncated_normal", "beta"]
CSVSigma2Prior = Literal["inv_gamma", "scaled_chi2"]
Persistent = Literal["constant", "sv"]
H0 = Literal["stationary", "fixed_b0"]
Sigma2Prior = Literal["scaled_chi2", "inv_gamma"]
Transitory = Literal["none", "outlier", "t", "outlier_t"]
TScaleKind = Literal["standardized", "unit"]
Mixture = Literal["ocsn10", "ksc7"]
BatchingKind = Literal["vmap", "map"]

MAX_EQUATIONS = 999  # error codes encode the equation in three digits (DESIGN §5.7)

# JAX-free capability tables of the Phase 2 kernels (mirrors of their ClassVars; DESIGN §3.8).
SUPPORTS_REDUCED_MEAN: dict[str, bool] = {"recursive": True}  # CovarianceStructure
COEF_VARIES: dict[str, bool] = {  # CoefPrior.varies
    "fixed_diagonal": False,
    **dict.fromkeys(GLKinds, True),  # global-local and SSVS priors (P3 D-43)
}


def _choice(owner: str, name: str, value: object, alias: object) -> None:
    options = get_args(alias)
    if value not in options:
        raise ValueError(f"{owner}.{name} must be one of {options}, got {value!r}")


def _int(obj: object, name: str, minimum: int) -> None:
    value = getattr(obj, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{type(obj).__name__}.{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{type(obj).__name__}.{name} must be >= {minimum}, got {value}")
    object.__setattr__(obj, name, int(value))  # canonical Python int (stable hash and repr)


def _bool(obj: object, name: str) -> None:
    value = getattr(obj, name)
    if not isinstance(value, bool):
        raise TypeError(f"{type(obj).__name__}.{name} must be a bool, got {value!r}")


@dataclass(frozen=True)
class Shapes:
    """Static dimensions; ``k`` and ``q`` are stored (and checked) so the cache key shows them."""

    n: int
    lags: int
    n_det: int
    k: int
    q: int
    T_pad: int

    def __post_init__(self) -> None:
        _int(self, "n", 1)
        _int(self, "lags", 1)
        _int(self, "n_det", 0)
        _int(self, "k", 1)
        _int(self, "q", 1)
        _int(self, "T_pad", 1)
        if self.n > MAX_EQUATIONS:
            raise ValueError(f"at most {MAX_EQUATIONS} equations (error codes), got n={self.n}")
        if self.k != self.n_det + self.n * self.lags:
            raise ValueError(f"k must equal n_det + n·lags = {self.n_det + self.n * self.lags}")
        if self.q != self.k + self.n - 1:
            raise ValueError(f"q must equal k + n − 1 = {self.k + self.n - 1}")

    @classmethod
    def of(cls, n: int, lags: int, n_det: int, T_pad: int) -> Shapes:
        k = n_det + n * lags
        return cls(n=n, lags=lags, n_det=n_det, k=k, q=k + n - 1, T_pad=T_pad)

    @property
    def P(self) -> int:
        """Packed structural Gram length ``q(q+1)/2``."""
        return self.q * (self.q + 1) // 2

    @property
    def P_x(self) -> int:
        """Packed regressor Gram length ``k(k+1)/2`` (reduced form)."""
        return self.k * (self.k + 1) // 2


@dataclass(frozen=True)
class StructStatic:
    """Covariance structure. ``contemp_scaling="lambda"``: ``a_i | λ_i ~ N(m, λ_i P̃⁻¹)`` (Mode II
    and ``Constant("iw_equivalent")``; DESIGN §4.7)."""

    kind: StructKind = "recursive"
    contemp_scaling: Scaling = "none"

    def __post_init__(self) -> None:
        _choice("StructStatic", "kind", self.kind, StructKind)
        _choice("StructStatic", "contemp_scaling", self.contemp_scaling, Scaling)


@dataclass(frozen=True)
class MeanStatic:
    """Conditional mean. ``coef_scaling="lambda"`` scales the δ_i prior by λ_i (Mode II only);
    ``const_path`` and ``gram`` are resolved values (DESIGN §3.7.2, §10.5), never ``"auto"``.

    ``n_dummy`` (WP25) is the number ``m`` of SOC/SUR dummy rows of the coefficient prior (shape
    defining: ``SharedPrior.dummies``, ``CoefPriorArrays.dummy_w`` and ``FitCache.dummy`` exist
    iff ``m > 0``); 0, the default, keeps every pre-WP25 program unchanged."""

    kind: MeanKind
    coef_scaling: Scaling = "none"
    const_path: ConstPath = "weighted"
    gram: GramKind = "packed"
    gram_tile: int = 0  # packed_tiled: columns of P per tile
    cccm_precompute: bool = True  # reduced form: every L_j before the j-loop (shape rule)
    n_dummy: int = 0  # SOC/SUR dummy rows of the coefficient prior (0 = none; WP25)
    coef_log: bool = False  # deviation path with log-space Jacobi scaling (P3 §3.4; varying priors)

    def __post_init__(self) -> None:
        _choice("MeanStatic", "kind", self.kind, MeanKind)
        _choice("MeanStatic", "coef_scaling", self.coef_scaling, Scaling)
        _choice("MeanStatic", "const_path", self.const_path, ConstPath)
        _choice("MeanStatic", "gram", self.gram, GramKind)
        _int(self, "gram_tile", 0)
        _bool(self, "cccm_precompute")
        _int(self, "n_dummy", 0)
        _bool(self, "coef_log")
        if (self.gram == "packed_tiled") != (self.gram_tile > 0):
            raise ValueError(
                "gram_tile > 0 is required by, and only allowed with, gram='packed_tiled'"
            )
        if self.n_dummy and self.kind == "zero":
            raise ValueError("dummy rows belong to a coefficient prior; the zero mean has none")


@dataclass(frozen=True)
class CoefPriorStatic:
    """Coefficient prior kernel: ``fixed_diagonal`` (Phase 2) or a global-local / SSVS sibling
    (P3 §3). ``n_groups`` is the number ``G`` of coefficient groups (shape-defining; resolved at
    fit because it can depend on ``n``), ``a_grid`` the size of the discrete hyperprior grid on
    ``a`` (0: ``a`` fixed; NG, R2D2, DL only) and ``p_beta`` the SSVS Beta hyperprior on ``p``."""

    kind: CoefKind = "fixed_diagonal"
    n_groups: int = 0
    a_grid: int = 0
    p_beta: bool = False

    def __post_init__(self) -> None:
        _choice("CoefPriorStatic", "kind", self.kind, CoefKind)
        _int(self, "n_groups", 0)
        _int(self, "a_grid", 0)
        _bool(self, "p_beta")
        gl = self.kind in GLKinds
        if gl != (self.n_groups >= 1):
            raise ValueError(
                "n_groups >= 1 is required by, and only allowed with, a global-local or SSVS "
                f"coefficient prior (kind={self.kind!r}, n_groups={self.n_groups})"
            )
        if self.a_grid and self.kind not in ("normal_gamma", "r2d2", "dirichlet_laplace"):
            raise ValueError(
                f"a_grid > 0 (a hyperprior on a) exists for normal_gamma, r2d2 and "
                f"dirichlet_laplace only, not {self.kind!r}"
            )
        if self.p_beta and self.kind != "ssvs":
            raise ValueError(
                f"p_beta (a Beta hyperprior on p) is an SSVS option, not {self.kind!r}"
            )

    @property
    def varies(self) -> bool:
        """Whether the coefficient moments change during sampling (``COEF_VARIES``)."""
        return COEF_VARIES[self.kind]


@dataclass(frozen=True)
class VolStatic:
    """Volatility block: persistent core ∘ transitory scale (DESIGN §3.5, §3.7.4–§3.7.6).

    ``nu_grid`` is shape-defining (its prior weights are data); ``coupled`` means the Constant
    core consumes the ``Coupling`` of a λ-scaled prior block (D-30).
    """

    persistent: Persistent
    rw: bool = False
    h0: H0 = "stationary"
    sigma2_prior: Sigma2Prior = "scaled_chi2"
    transitory: Transitory = "none"
    outlier_K: int = 20
    nu_grid: tuple[int, ...] = tuple(range(3, 41))
    t_scale: TScaleKind = "standardized"
    mixture: Mixture = "ocsn10"
    exact: bool = True
    asis: bool = True
    zero_offset: bool = False
    coupled: bool = False
    common: bool = False  # CommonSV: λ_{i,t} = λ_i e^{h_t}, h in glob.vol (P3 §5)
    csv_block: int = 100  # CommonSV h-step MH block length (P3 D-48)
    csv_rho_prior: RhoPrior = "truncated_normal"
    csv_sigma2_prior: CSVSigma2Prior = "inv_gamma"

    def __post_init__(self) -> None:
        _choice("VolStatic", "persistent", self.persistent, Persistent)
        _choice("VolStatic", "h0", self.h0, H0)
        _choice("VolStatic", "sigma2_prior", self.sigma2_prior, Sigma2Prior)
        _choice("VolStatic", "transitory", self.transitory, Transitory)
        _choice("VolStatic", "t_scale", self.t_scale, TScaleKind)
        _choice("VolStatic", "mixture", self.mixture, Mixture)
        for name in ("rw", "exact", "asis", "zero_offset", "coupled", "common"):
            _bool(self, name)
        _int(self, "outlier_K", 1)
        _int(self, "csv_block", 2)
        _choice("VolStatic", "csv_rho_prior", self.csv_rho_prior, RhoPrior)
        _choice("VolStatic", "csv_sigma2_prior", self.csv_sigma2_prior, CSVSigma2Prior)
        grid = tuple(self.nu_grid)
        if not grid or any(
            isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in grid
        ):
            raise TypeError("VolStatic.nu_grid must be a non-empty tuple of integers")
        grid = tuple(int(v) for v in grid)
        if grid[0] < 1 or any(b <= a for a, b in itertools.pairwise(grid)):
            raise ValueError("VolStatic.nu_grid must be strictly increasing and positive")
        object.__setattr__(self, "nu_grid", grid)

    @property
    def has_outliers(self) -> bool:
        return self.transitory in ("outlier", "outlier_t")

    @property
    def has_t(self) -> bool:
        return self.transitory in ("t", "outlier_t")

    @property
    def constant_weights(self) -> bool:
        """Equation weights ``m_t/λ_i`` are time-constant: Constant core, no transitory scale and no
        common volatility (P3 E50: with ``common`` the weights are ``m_t/(λ_i e^{h_t})``)."""
        return self.persistent == "constant" and self.transitory == "none" and not self.common


@dataclass(frozen=True)
class StoreStatic:
    """What the sample executable emits besides monitors and parameter draws (DESIGN §5.3, §5.8)."""

    paths: bool = False  # h and log(o²τ) paths every path_thin stored draws
    in_loop_forecast: int = 0  # FitPlan horizon H (0 = off)

    def __post_init__(self) -> None:
        _bool(self, "paths")
        _int(self, "in_loop_forecast", 0)


@dataclass(frozen=True)
class StaticConfig:
    """The complete static description of one compiled program (DESIGN §2.4).

    Construction validates every field and the composition (:func:`validate_composition`).
    """

    shapes: Shapes
    struct: StructStatic
    mean: MeanStatic
    coef: CoefPriorStatic
    vol: VolStatic
    store: StoreStatic = StoreStatic()
    batching: BatchingKind = "vmap"
    chunk: int = 250  # kept draws per sample executable (Runtime)
    float32_leaves: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name, cls in (
            ("shapes", Shapes),
            ("struct", StructStatic),
            ("mean", MeanStatic),
            ("coef", CoefPriorStatic),
            ("vol", VolStatic),
            ("store", StoreStatic),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"StaticConfig.{name} must be a {cls.__name__}")
        _choice("StaticConfig", "batching", self.batching, BatchingKind)
        _int(self, "chunk", 1)
        if isinstance(self.float32_leaves, str):
            raise TypeError("StaticConfig.float32_leaves must be a collection of leaf names")
        leaves = frozenset(self.float32_leaves)
        if not all(isinstance(x, str) and x for x in leaves):
            raise TypeError("StaticConfig.float32_leaves must contain non-empty strings")
        object.__setattr__(self, "float32_leaves", leaves)
        validate_composition(self.struct, self.mean, self.coef, self.vol)


def bucket(T_eff: int, t_bucket: int) -> int:
    """``T_pad = t_bucket·⌈T_eff/t_bucket⌉``: a function of ``(T_eff, t_bucket)`` only (D-23)."""
    for name, value in (("T_eff", T_eff), ("t_bucket", t_bucket)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(t_bucket) * (-(-int(T_eff) // int(t_bucket)))


def scaling_mode(struct: StructStatic, mean: MeanStatic) -> Literal["I", "II", "iw_equivalent"]:
    """Which prior blocks are λ-scaled (DESIGN §6.1): none (Mode I), all (Mode II), ``a`` only."""
    if mean.coef_scaling == "lambda":
        return "II"
    return "iw_equivalent" if struct.contemp_scaling == "lambda" else "I"


def validate_composition(
    struct: StructStatic, mean: MeanStatic, coef: CoefPriorStatic, vol: VolStatic
) -> None:
    """Legality of a composition (DESIGN §3.8 item 2, §4.7, §4.8 item 5). Raises ``ValueError``.

    Rules, in order:

    1. a reduced-form mean needs a structure with ``supports_reduced_mean``;
    2. ``coef_scaling="lambda"`` needs ``contemp_scaling="lambda"`` (Mode II scales both);
    3. Mode II (λ-scaled δ) is a structural ``Linear`` prior only;
    4. ``coupled`` needs the Constant core, and holds exactly when some prior block is λ-scaled;
    5. a constant-path other than ``weighted`` belongs to ``linear_structural`` and needs
       time-constant weights (Constant core, no transitory scale);
    6. ``eigen`` needs Mode I and a non-varying coefficient prior (with SOC/SUR dummy rows it uses
       the dense prior precision, ``linalg.eigen_prepare_dense``; N05 §7.7, WP25);
    7. ``fixed_factor`` needs Mode II and a non-varying coefficient prior;
    8. ``rw`` and ``zero_offset`` are SV options (``rw`` ignores ``h0``);
    9. outliers need ``outlier_K >= 2``;
    10. Student-t with ``t_scale="standardized"`` needs every grid ν > 2;

    Phase 3 (P3 DESIGN §2.1; rule 5 exempts ``shared_factor``, which has time-varying common
    weights by construction):

    11. a varying coefficient prior needs ``coef_log``, the ``weighted`` path, no λ-scaled
        coefficient block (global-local priors are independent priors; P3 D-43) and no SOC/SUR
        dummy rows (``n_dummy = 0``; GL priors have none, P3 D-36, E153);
    12. ``coef_log`` needs a ``linear_structural`` or ``linear_reduced`` mean and a varying prior;
    13. ``common`` (CommonSV) needs the Constant core and no transitory scale;
    14. ``shared_factor`` needs ``linear_structural``, Mode II, ``common`` and a non-varying
        coefficient prior (the equation invariance of its prior precision is data, checked by
        ``resolve.py``).
    """
    scaled = struct.contemp_scaling == "lambda" or mean.coef_scaling == "lambda"
    mode = scaling_mode(struct, mean)
    if mean.kind == "linear_reduced" and not SUPPORTS_REDUCED_MEAN[struct.kind]:
        raise ValueError(f"the {struct.kind!r} structure does not support a reduced-form mean")
    if mean.coef_scaling == "lambda" and struct.contemp_scaling != "lambda":
        raise ValueError(
            "coef_scaling='lambda' (Mode II) requires contemp_scaling='lambda': "
            "Mode II scales every coefficient by λ_i"
        )
    if mean.coef_scaling == "lambda" and mean.kind != "linear_structural":
        raise ValueError(
            "a λ-scaled coefficient prior (Mode II, Constant('iw')) exists only for the structural "
            "Linear mean; use contemp scaling alone ('iw_equivalent') otherwise"
        )
    if vol.coupled and vol.persistent != "constant":
        raise ValueError("VolStatic.coupled requires the Constant persistent core (DESIGN §4.7)")
    if scaled and not vol.coupled:
        raise ValueError(
            "λ-scaled prior blocks need VolStatic(persistent='constant', coupled=True): "
            "the coupling channel is the only prior-volatility interaction (D-30)"
        )
    if vol.coupled and not scaled:
        raise ValueError("VolStatic.coupled=True without a λ-scaled prior block")
    varies = COEF_VARIES[coef.kind]
    if mean.const_path != "weighted":
        if mean.kind != "linear_structural":
            raise ValueError(
                f"const_path={mean.const_path!r} is a structural Linear path; "
                f"use 'weighted' for mean kind {mean.kind!r}"
            )
        if not vol.constant_weights and mean.const_path != "shared_factor":
            raise ValueError(
                f"const_path={mean.const_path!r} needs time-constant weights "
                "(Constant core, no transitory scale); use 'weighted'"
            )
    if mean.const_path == "eigen" and (mode != "I" or COEF_VARIES[coef.kind]):
        raise ValueError("const_path='eigen' needs Mode I and a non-varying coefficient prior")
    if mean.const_path == "fixed_factor" and (mode != "II" or COEF_VARIES[coef.kind]):
        raise ValueError(
            "const_path='fixed_factor' needs Mode II and a non-varying coefficient prior"
        )
    if vol.rw and vol.persistent != "sv":
        raise ValueError("rw=True is a stochastic-volatility option (persistent='sv')")
    if vol.zero_offset and vol.persistent != "sv":
        raise ValueError("zero_offset=True is a stochastic-volatility parity option")
    if vol.has_outliers and vol.outlier_K < 2:
        raise ValueError(f"outlier_K must be >= 2, got {vol.outlier_K}")
    if vol.has_t and vol.t_scale == "standardized" and min(vol.nu_grid) <= 2:
        raise ValueError(
            "a standardized Student-t needs ν > 2 on the whole grid (Var = 1 requires ν > 2); "
            "drop ν <= 2 or use t_scale='unit'"
        )
    if varies:
        if not mean.coef_log:
            raise ValueError(
                f"the {coef.kind!r} coefficient prior varies during sampling and needs "
                "MeanStatic(coef_log=True), the deviation path of P3 §3.4"
            )
        if mean.const_path != "weighted":
            raise ValueError(
                f"the {coef.kind!r} coefficient prior varies during sampling; "
                f"const_path={mean.const_path!r} needs a fixed prior: use 'weighted'"
            )
        if mean.coef_scaling == "lambda":
            raise ValueError(
                f"the {coef.kind!r} coefficient prior is an independent prior; a λ-scaled "
                "coefficient block (Mode II, Constant('iw')) is not available with it"
            )
        if mean.n_dummy:
            raise ValueError(
                f"the {coef.kind!r} coefficient prior has no SOC/SUR dummy rows (P3 D-36); "
                f"MeanStatic.n_dummy must be 0, got {mean.n_dummy}"
            )
    if mean.coef_log:
        if mean.kind not in ("linear_structural", "linear_reduced"):
            raise ValueError(
                f"coef_log=True is a Linear mean path; mean kind {mean.kind!r} has none"
            )
        if not varies:
            raise ValueError(
                "coef_log=True (the deviation path) belongs to a varying coefficient prior; "
                f"{coef.kind!r} is fixed"
            )
    if vol.common and (vol.persistent != "constant" or vol.transitory != "none"):
        raise ValueError(
            "CommonSV (common=True) needs the Constant persistent core and no transitory scale"
        )
    if mean.const_path == "shared_factor":
        if mean.kind != "linear_structural" or mode != "II" or not vol.common or varies:
            raise ValueError(
                "const_path='shared_factor' needs the structural Linear mean, Mode II (the "
                "conjugate prior), CommonSV (common=True) and a fixed coefficient prior"
            )
