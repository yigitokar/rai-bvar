"""Joint models for Geweke and SBC: :class:`JointSpec`, the composition table and test priors
(DESIGN §9.2, §9.5; N06 §2, §3, §9.1).

A :class:`JointSpec` implements the model-agnostic ``validation.geweke.JointModel`` protocol (WP7)
for one composition of the production engine:

* ``prior_sample(key)`` — θ ~ p(θ) from the independent references (``simulate.prior_sample``);
* ``simulate(key, st)`` — y ~ p(y | θ), recursive in t with the **current** latents;
* ``step(key, it, st, data)`` — ``prepare(data, prior)`` then **one production sweep**
  (``compose`` → ``make_sweep``, WP14), never a copy of the sampler;
* ``test_functions(st, data)`` — the finite-variance test functions of ``testfns.py``.

Test priors (N06 §9.1, DESIGN §9.2; proper and tight, ``r = Var_prior/Var_post ≲ 5``):
``a_ij ~ N(0, 0.7²)``; ``M`` own first lag 0.4; δ (or π) intercept s.d. 1, own lag s.d. ``0.10/ℓ``,
cross lag s.d. ``0.05/ℓ`` (explosive mass ≤ 2e-4); Constant ``λ ~ IG(5, 4)``; SV
``μ ~ N(0, 0.5²)``, ``(φ+1)/2 ~ Beta(10, 2)``, ``B_σ = 0.1`` (``InvGamma``: ``IG(5, 0.4)``; fixed
``B₀ = 2``); RW ``h_0 ~ N(0, 1)``; outliers ``p ~ Beta(2, 18)``, ``K = 20``; t on the static grid
``ν ∈ {5..12}`` uniform, standardized ``τ``; offset ``c = 1e-12`` (ψ ≡ 1, R9 recomputation off).
SOC/SUR dummy rows (WP25; compositions ``*_dummies``, ids 201–205): the Phase 1 rows
(``priors.dummies.dummy_rows``) of a fixed ``ȳ₀ = (1.5, −1.0, 0.8)`` (the PR presample is zero),
SOC rows for every variable and the SUR row, ``μ = δ = 0.5`` (rows of size ~2–3, enough to move the
intercept and contemporaneous priors), weight 1 (``= 1/E[λ]`` under ``IG(5, 4)``; Mode II: the
λ-free 1). Sizes: ``"small"`` (PR) ``n = 2`` structural / ``3`` reduced, ``p = 1``, intercept, ``T_eff = 20``,
``y_{1−p:0} = 0``; ``"full"`` (nightly) ``n = 3``, ``p = 2``, ``T_eff = 50``, a fixed stored
presample. The §9.5 table's ``n`` column overrides both sizes where it gives one value (rows 6–13
and 18: 3; rows 14–16: 2; row 17: 1). ``T_eff`` is never a multiple of 32, so the padding masks are
always exercised.

Block-level configurations (one block live; DESIGN §9.2 table) hold the other blocks at their
prior draws. Mechanism (a documented deviation from "identity kernels through ``_overrides``",
which would need kernel constructors here): the production sweep runs, then every leaf outside
``live`` is restored to its value before the sweep (:func:`hold_frozen`). This is an exact kernel
for the live blocks given the frozen ones whenever every frozen block runs **after** every live
block within the sweep (vol after the equation step; core after the transitory steps inside
``VolBlock.sample_eq``; ``a`` and vol after CCCM). A frozen block that runs *before* a live one is
held by an identity override of its named internal instead (``contemp_only``: ``cccm_core``).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
    bucket,
)
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.state import (
    ChainState,
    CoefPriorArrays,
    CommonSVPrior,
    ConstantPrior,
    ContempPriorArrays,
    DummyRows,
    Empty,
    EngineData,
    EqOut,
    EqPrior,
    GLArrays,
    GLShared,
    PriorArrays,
    SharedPrior,
    SSVSArrays,
    StepInfo,
    SVPrior,
    SweepCtx,
    TransitoryPrior,
    leaf_path,
    prior_template,
    state_template,
)
from rai_bvar.priors.dummies import dummy_rows
from rai_bvar.validation import testfns
from rai_bvar.validation.log_joint import log_joint, residuals
from rai_bvar.validation.reference import (
    PriorTails,
    gl_log_r_reference,
    prior_tails,
    state_log_var,
)
from rai_bvar.validation.simulate import (
    gl_coefficient_draw,
    prior_sample,
    simulate_aux_obs,
    simulate_engine_data,
    simulate_obs,
    structural_gamma,
    tmask_of,
    unit_lower,
)

Array = jax.Array
_F64 = jnp.float64
ALL: frozenset[str] = frozenset({"all"})
T_BUCKET = 32
TEST_NU_GRID: tuple[int, ...] = tuple(range(5, 13))
TEST_YBAR0: tuple[float, ...] = (1.5, -1.0, 0.8)  # ȳ₀ of the test SOC/SUR rows (WP25)
# Fixed presample of the "full" size (p = 2 rows, up to 3 variables; stored, not drawn).
FULL_PRESAMPLE = np.array([[0.31, -0.42, 0.18], [-0.12, 0.27, -0.35]], dtype=np.float64)


# --------------------------------------------------------------------------- fixed design


@dataclass(frozen=True, eq=False)
class FixedDesign:
    """Fixed part of the data: presample ``y_{1−p:0}`` (``p × n``), ``T_eff`` and the trend value
    ``t0`` of the first target row (``p + 1`` reproduces ``data.design.lagged_design``).

    P3 (D-56): ``exogenous=True`` holds a fixed regressor matrix ``X`` (``T_eff × k``; ones for the
    deterministic terms, ``N(0, 1)`` elsewhere, :meth:`exogenous_design`), and ``y`` is simulated
    given θ without lag feedback; the presample then only fills ``EngineData.hist``."""

    presample: np.ndarray
    T_eff: int
    t0: float = 2.0
    exogenous: bool = False
    X: np.ndarray | None = None

    def __post_init__(self) -> None:
        pre = np.array(self.presample, dtype=np.float64)
        if pre.ndim != 2 or pre.shape[0] < 1:
            raise ValueError(f"presample must have shape (p, n) with p >= 1, got {pre.shape}")
        if not np.all(np.isfinite(pre)):
            raise ValueError("presample must be finite")
        pre.setflags(write=False)
        object.__setattr__(self, "presample", pre)
        if isinstance(self.T_eff, bool) or int(self.T_eff) != self.T_eff or self.T_eff < 1:
            raise ValueError(f"T_eff must be a positive integer, got {self.T_eff!r}")
        object.__setattr__(self, "T_eff", int(self.T_eff))
        object.__setattr__(self, "t0", float(self.t0))
        if not isinstance(self.exogenous, bool):
            raise TypeError(f"exogenous must be a bool, got {self.exogenous!r}")
        if self.exogenous:
            if self.X is None:
                raise ValueError("an exogenous design needs its fixed X (T_eff × k)")
            X = np.array(self.X, dtype=np.float64)
            if X.ndim != 2 or X.shape[0] != self.T_eff or not np.all(np.isfinite(X)):
                raise ValueError(f"X must be a finite ({self.T_eff}, k) matrix, got {X.shape}")
            X.setflags(write=False)
            object.__setattr__(self, "X", X)
        elif self.X is not None:
            raise ValueError("X belongs to an exogenous design (exogenous=True)")

    @classmethod
    def exogenous_design(
        cls,
        presample: np.ndarray,
        T_eff: int,
        *,
        n_det: int,
        k: int,
        seed: int | tuple[int, ...],
        t0: float = 2.0,
    ) -> FixedDesign:
        """The D-56 design: ``X`` (``T_eff × k``) with a column of ones for each deterministic
        term and ``N(0, 1)`` entries elsewhere, drawn once from ``numpy.random.default_rng(seed)``
        (the configuration seed) and held fixed."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((int(T_eff), int(k)))
        X[:, :n_det] = 1.0
        return cls(presample, T_eff, t0, exogenous=True, X=X)

    def _key(self) -> tuple[Any, ...]:
        x = () if self.X is None else (self.X.shape, self.X.tobytes())
        return (
            self.presample.shape,
            self.presample.tobytes(),
            self.T_eff,
            self.t0,
            self.exogenous,
            x,
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FixedDesign) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


# --------------------------------------------------------------------------- test priors


def reference_prior(
    static: StaticConfig,
    *,
    contemp_sd: float = 0.7,
    own_lag_mean: float = 0.4,
    intercept_sd: float = 1.0,
    trend_sd: float = 0.01,
    own_sd: float = 0.10,
    cross_sd: float = 0.05,
    ig_shape: float = 5.0,
    ig_scale: float = 4.0,
    b_mu: float = 0.0,
    B_mu: float = 0.25,
    a_phi: float = 10.0,
    b_phi: float = 2.0,
    B_sigma: float = 0.1,
    c0: float = 5.0,
    C0: float = 0.4,
    B0: float = 2.0,
    m0: float = 0.0,
    V0: float = 1.0,
    offset_rel: float = 1e-12,
    a_p: float = 2.0,
    b_p: float = 18.0,
    dummy_ybar0: tuple[float, ...] = TEST_YBAR0,
    dummy_soc: float = 0.5,
    dummy_sur: float = 0.5,
    dummy_weight: float = 1.0,
) -> PriorArrays:
    """The validation test prior (N06 §9.1 corrected values; module docstring) as
    :class:`PriorArrays` for ``static`` (ψ ≡ 1, so the relative offset is absolute). With
    ``static.mean.n_dummy > 0`` (WP25): the Phase 1 SOC rows of every variable and the SUR row of
    ``dummy_ybar0`` (``n + 1`` rows), each equation's weight ``dummy_weight``."""
    s, vol = static.shapes, static.vol
    n, p, nd, k = s.n, s.lags, s.n_det, s.k
    has_mean = static.mean.kind != "zero"
    M = np.zeros((k, n))
    sd = np.ones((n, k))
    group = np.zeros((n, k), np.int32)
    sd[:, :nd] = [intercept_sd, trend_sd][:nd]
    for i in range(n):
        for lag in range(1, p + 1):
            for var in range(n):
                col = nd + (lag - 1) * n + var
                sd[i, col] = (own_sd if var == i else cross_sd) / lag
                group[i, col] = 1 if var == i else 2
        if has_mean:
            M[nd + i, i] = own_lag_mean
    if has_mean:
        coef = CoefPriorArrays(mean0=M.T.copy(), prec0=1.0 / sd**2, group=group)
    else:
        coef = CoefPriorArrays(mean0=np.zeros((n, k)), prec0=np.ones((n, k)), group=group)
    cmask = (np.arange(n - 1)[None, :] < np.arange(n)[:, None]).astype(np.float64)
    contemp = ContempPriorArrays(
        mean0=np.zeros((n, n - 1)),
        prec0=np.where(cmask > 0, 1.0 / contemp_sd**2, 1.0),
        cmask=cmask,
    )
    one = np.ones(n)
    if vol.persistent == "constant":
        core: ConstantPrior | SVPrior = ConstantPrior(shape=ig_shape * one, scale=ig_scale * one)
    else:
        log_c = -np.inf if vol.zero_offset else math.log(offset_rel)
        core = SVPrior(
            b_mu=b_mu * one, B_mu=B_mu * one, a_phi=a_phi * one, b_phi=b_phi * one,
            B_sigma=B_sigma * one, c0=c0 * one, C0=C0 * one, B0=B0 * one, m0=m0 * one,
            V0=V0 * one, log_c=log_c * one,
        )  # fmt: skip
    G = len(vol.nu_grid)
    trans: TransitoryPrior | Empty = (
        Empty()
        if vol.transitory == "none"
        else TransitoryPrior(a_p=a_p * one, b_p=b_p * one, log_nu_w=np.full((n, G), -math.log(G)))
    )
    Mc = M[:, : n - 1] if static.mean.kind == "linear_structural" else np.zeros((k, n - 1))
    shared = SharedPrior(Mc=Mc)
    if static.mean.n_dummy:  # SOC/SUR rows of the test ȳ₀ (WP25)
        ybar0 = np.asarray(dummy_ybar0, dtype=np.float64)[:n]
        Yd, Xd = dummy_rows(ybar0, np.arange(n), dummy_soc, dummy_sur, lags=p, n_det=nd)
        if Yd.shape[0] != static.mean.n_dummy:
            raise ValueError(
                f"the test dummy rows are n + 1 = {Yd.shape[0]}, but MeanStatic.n_dummy = "
                f"{static.mean.n_dummy}"
            )
        coef = dataclasses.replace(coef, dummy_w=np.full(n, float(dummy_weight)))
        shared = SharedPrior(Mc=Mc, dummies=DummyRows(Y=Yd, X=Xd))
    tree = PriorArrays(
        eq=EqPrior(coef=coef, contemp=contemp, vol_core=core, transitory=trans),
        shared=shared,
    )
    return jax.tree.map(jnp.asarray, tree)


def update_prior(prior: PriorArrays, updates: Mapping[str, Any]) -> PriorArrays:
    """Replace prior leaves by dotted path (``"eq.vol_core.B_sigma"``), broadcasting each value
    to the leaf's shape (dtype kept). Unknown paths raise ``KeyError``."""
    seen: set[str] = set()

    def upd(path: tuple[Any, ...], leaf: Array) -> Array:
        name = leaf_path(path)
        if name not in updates:
            return leaf
        seen.add(name)
        return jnp.broadcast_to(jnp.asarray(updates[name], leaf.dtype), leaf.shape)

    out = jax.tree_util.tree_map_with_path(upd, prior)
    missing = sorted(set(updates) - seen)
    if missing:
        raise KeyError(f"no prior leaves {missing}")
    return out


# --------------------------------------------------------------------------- production sweep


def identity_cccm_core(*args: Any, **kwargs: Any) -> tuple[Array, StepInfo]:
    """Override of the ``cccm_core`` internal (DESIGN §3.7.3) that returns ``Π`` unchanged: holds
    ``Π`` fixed in the ``contemp_only`` block configuration."""
    Pi = kwargs["Pi"] if "Pi" in kwargs else args[0]
    return Pi, StepInfo.success()


IDENTITY_INTERNALS: dict[str, Callable[..., Any]] = {"cccm_core": identity_cccm_core}


@functools.cache
def production_sweep(
    static: StaticConfig, overrides: tuple[tuple[str, Callable[..., Any]], ...] = ()
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """``(prepare, sweep)`` of the production engine (WP14) for ``static``:
    ``prepare(data, prior) -> FitCache`` and ``sweep(ckey, it, st, data, prior, cache)``."""
    try:
        from rai_bvar.engines.mcmc.compose import compose
        from rai_bvar.engines.mcmc.runner import make_runner
        from rai_bvar.engines.mcmc.sweep import make_sweep
    except ImportError as err:
        raise NotImplementedError(
            "JointSpec.step drives the production sweep (engines.mcmc.compose/sweep/runner, "
            f"WP14), which is not available yet: {err}"
        ) from err
    ov = dict(overrides) or None
    program = compose(static, overrides=ov)
    return make_runner(static, overrides=ov).prepare, make_sweep(program)


def _is_live(name: str, live: frozenset[str]) -> bool:
    return any(name == path or name.startswith(path + ".") for path in live)


def hold_frozen(new: ChainState, old: ChainState, live: frozenset[str]) -> ChainState:
    """Every leaf outside ``live`` (and outside ``status``) back to its value in ``old``."""
    if "all" in live:
        return new

    def pick(path: tuple[Any, ...], a: Array, b: Array) -> Array:
        name = leaf_path(path)
        return a if name.startswith("status.") or _is_live(name, live) else b

    return jax.tree_util.tree_map_with_path(pick, new, old)


_LATENT_PATHS = ("eq.vol.core.h", "eq.vol.trans.o_idx", "eq.vol.trans.log_tau")


def with_fresh_latents(st: ChainState, fresh: ChainState) -> ChainState:
    """``st`` with the latent paths (``h``, ``o``, ``τ``) of ``fresh`` (mutant M3)."""

    def pick(path: tuple[Any, ...], a: Array, b: Array) -> Array:
        return b if leaf_path(path) in _LATENT_PATHS else a

    return jax.tree_util.tree_map_with_path(pick, st, fresh)


# --------------------------------------------------------------------------- JointSpec


@dataclass(frozen=True, eq=False)
class JointSpec:
    """One joint model ``p(θ, y)`` with the production sweep as its posterior kernel (implements
    ``validation.geweke.JointModel``; DESIGN §9.2).

    Extra fields beyond the DESIGN sketch: ``obs`` (``"aux"``: auxiliary mixture observations),
    ``sampler_prior`` (a different prior for the sampler only: mutant M5), ``overrides`` (kernel
    internals or slots passed to ``compose``: mutants and block configurations), ``fresh_latents``
    (mutant M3: the simulator ignores the state's latents) and ``raw_moments`` (raw data moments
    among the test functions; flagged finite only when ``E y⁴ < ∞``). The finite-variance flags
    of every test function follow the prior's tails (:attr:`tails`, N06 §3.4)."""

    name: str
    static: StaticConfig
    prior: PriorArrays
    fixed: FixedDesign
    live: frozenset[str] = ALL
    obs: Literal["gaussian", "aux"] = "gaussian"
    sampler_prior: PriorArrays | None = None
    overrides: tuple[tuple[str, Callable[..., Any]], ...] = field(default=(), repr=False)
    fresh_latents: bool = False
    raw_moments: bool = False

    def __post_init__(self) -> None:
        tmpl = jax.tree.structure(prior_template(self.static))
        for label, pr in (("prior", self.prior), ("sampler_prior", self.sampler_prior)):
            if pr is not None and jax.tree.structure(pr) != tmpl:
                raise ValueError(f"{label} does not match the StaticConfig's prior structure")
        if self.obs not in ("gaussian", "aux"):
            raise ValueError(f"obs must be 'gaussian' or 'aux', got {self.obs!r}")
        if self.obs == "aux" and self.static.vol.persistent != "sv":
            raise ValueError("obs='aux' needs the SV persistent core")
        if self.fixed.presample.shape != (self.static.shapes.lags, self.static.shapes.n):
            raise ValueError("presample shape must be (lags, n)")
        if not 1 <= self.fixed.T_eff <= self.static.shapes.T_pad:
            raise ValueError("T_eff must lie in [1, T_pad]")
        live = frozenset(self.live)
        paths = {leaf_path(p) for p, _ in jax.tree_util.tree_flatten_with_path(self._template)[0]}
        unknown = [x for x in live if x != "all" and not any(_is_live(p, {x}) for p in paths)]
        if unknown:
            raise ValueError(f"live paths {sorted(unknown)} match no state leaf")
        object.__setattr__(self, "live", live)
        object.__setattr__(self, "overrides", tuple(sorted(dict(self.overrides).items())))

    @property
    def _template(self) -> ChainState:
        return state_template(self.static)

    # ----------------------------------------------------------------- test functions
    @cached_property
    def tails(self) -> PriorTails:
        """Tail indices of :attr:`prior` that decide the finite-variance flags (N06 §3.4)."""
        return prior_tails(self.static, self.prior)

    @cached_property
    def function_flags(self) -> dict[str, bool]:
        """Ordered ``{test function name: finite variance under p(θ, y)}`` (prior-aware)."""
        return testfns.test_function_spec(
            self.static, self.prior, self.fixed.T_eff, raw_moments=self.raw_moments,
            tails=self.tails,
        )  # fmt: skip

    @property
    def names(self) -> tuple[str, ...]:
        """Geweke test functions: the finite-variance ones (N06 §3.4)."""
        return tuple(k for k, fin in self.function_flags.items() if fin)

    @property
    def all_names(self) -> tuple[str, ...]:
        """Every test quantity (SBC ranks need no moments)."""
        return tuple(self.function_flags)

    def _functions(self, st: ChainState, data: EngineData) -> testfns.TestFns:
        return testfns.test_functions(
            st, data, self.prior, self.static, raw_moments=self.raw_moments, tails=self.tails
        )

    def test_functions(self, st: ChainState, data: EngineData) -> Array:
        """``f64[G]`` in the order of :attr:`names`."""
        d = self._functions(st, data)
        return jnp.stack([d[k][0] for k in self.names])

    def quantities(self, st: ChainState, data: EngineData) -> Array:
        """``f64[Q]`` in the order of :attr:`all_names` (SBC)."""
        d = self._functions(st, data)
        return jnp.stack([d[k][0] for k in self.all_names])

    # ----------------------------------------------------------------- simulators and kernel
    @property
    def tmask(self) -> Array:
        return tmask_of(self.fixed.T_eff, self.static.shapes.T_pad)

    def prior_sample(self, key: Array) -> ChainState:
        """θ ~ p(θ), including ``h``, ``o``, ``τ``, ``p``, ``ν`` (never ``s``)."""
        return prior_sample(key, self.prior, self.static, self.fixed.T_eff)

    def simulate(self, key: Array, st: ChainState) -> EngineData:
        """y ~ p(y | θ) with the state's current latents (Geweke error 2 otherwise)."""
        if self.fresh_latents:
            st = with_fresh_latents(st, self.prior_sample(random.fold_in(key, 1)))
            key = random.fold_in(key, 0)
        sim = simulate_aux_obs if self.obs == "aux" else simulate_obs
        return sim(key, st, self.static, self.prior, self.fixed)

    def step(self, key: Array, it: Array, st: ChainState, data: EngineData) -> ChainState:
        """``prepare(data, prior)`` then one production sweep; frozen blocks restored."""
        prepare, sweep = production_sweep(self.static, self.overrides)
        prior = self.prior if self.sampler_prior is None else self.sampler_prior
        cache = prepare(data, prior)
        new = sweep(key, it, st, data, prior, cache)
        return hold_frozen(new, st, self.live)

    def log_joint(self, st: ChainState, data: EngineData, s: Array | None = None) -> Array:
        """The reference ``log p(θ, y)`` (``validation.log_joint``) under this spec's prior."""
        return log_joint(st, data, self.prior, self.static, s)


# --------------------------------------------------------------------------- compositions


@dataclass(frozen=True)
class _Def:
    number: int
    mean: Literal["linear_structural", "linear_reduced", "zero"]
    persistent: Literal["constant", "sv"] = "sv"
    transitory: Literal["none", "outlier", "t", "outlier_t"] = "none"
    mode: Literal["I", "II", "iw_equivalent"] = "I"
    rw: bool = False
    sigma2_prior: Literal["scaled_chi2", "inv_gamma"] = "scaled_chi2"
    h0: Literal["stationary", "fixed_b0"] = "stationary"
    n: tuple[int, int] = (2, 3)  # (small, full)
    live: frozenset[str] = ALL
    obs: Literal["gaussian", "aux"] = "gaussian"
    exact: bool = True
    zero_offset: bool = False
    identity: tuple[str, ...] = ()  # internals held fixed by IDENTITY_INTERNALS
    dummies: bool = False  # SOC/SUR dummy rows of the coefficient prior (WP25)


_S, _R, _Z = "linear_structural", "linear_reduced", "zero"

# DESIGN §9.5 rows 1–18 (their inexact variants are the SV rows with exact=False:
# joint_spec(name, exact=False); they keep the row's id).
_COMPOSITIONS: dict[str, _Def] = {
    "structural_constant": _Def(1, _S, "constant"),
    "structural_sv": _Def(2, _S),
    "structural_sv_outliers": _Def(3, _S, transitory="outlier"),
    "structural_sv_t": _Def(4, _S, transitory="t"),
    "structural_mode2_constant": _Def(5, _S, "constant", mode="II"),
    "structural_iw_equivalent": _Def(6, _S, "constant", mode="iw_equivalent", n=(3, 3)),
    "reduced_constant": _Def(7, _R, "constant", n=(3, 3)),
    "reduced_sv": _Def(8, _R, n=(3, 3)),
    "reduced_sv_outliers": _Def(9, _R, transitory="outlier", n=(3, 3)),
    "reduced_sv_t": _Def(10, _R, transitory="t", n=(3, 3)),
    "reduced_iw_equivalent": _Def(11, _R, "constant", mode="iw_equivalent", n=(3, 3)),
    "structural_sv_outliers_t": _Def(12, _S, transitory="outlier_t", n=(3, 3)),
    "reduced_sv_outliers_t": _Def(13, _R, transitory="outlier_t", n=(3, 3)),
    "structural_sv_rw": _Def(14, _S, rw=True, n=(2, 2)),
    "structural_sv_invgamma": _Def(15, _S, sigma2_prior="inv_gamma", n=(2, 2)),
    "structural_sv_fixed_b0": _Def(16, _S, h0="fixed_b0", n=(2, 2)),
    "zero_sv": _Def(17, _Z, n=(1, 1)),
    # the Clark–Ravazzolo primary SV arm's composition (reduced-form CCCM × random-walk SV)
    "reduced_sv_rw": _Def(18, _R, rw=True, n=(3, 3)),
    # WP25 (DESIGN §13): rows with SOC/SUR dummy rows in the coefficient prior (ids 201+)
    "structural_constant_dummies": _Def(201, _S, "constant", dummies=True),
    "structural_mode2_constant_dummies": _Def(202, _S, "constant", mode="II", dummies=True),
    "reduced_constant_dummies": _Def(203, _R, "constant", n=(3, 3), dummies=True),
    "structural_sv_dummies": _Def(204, _S, dummies=True),
    "structural_iw_equivalent_dummies": _Def(
        205, _S, "constant", mode="iw_equivalent", n=(3, 3), dummies=True
    ),
}

# DESIGN §9.2 block-level configurations.
_BLOCKS: dict[str, _Def] = {
    "aux_sv": _Def(101, _Z, n=(1, 1), obs="aux", exact=False, zero_offset=True),
    "gauss_sv_exact": _Def(102, _Z, n=(1, 1), exact=True),
    "transitory_known_lambda": _Def(
        103, _Z, "constant", transitory="outlier_t", n=(1, 1), live=frozenset({"eq.vol.trans"})
    ),
    "constant_transitory": _Def(104, _Z, "constant", transitory="outlier_t", n=(1, 1)),
    "equation_only": _Def(105, _S, "constant", live=frozenset({"eq.mean", "eq.struct"})),
    "cccm_only": _Def(106, _R, "constant", n=(3, 3), live=frozenset({"glob.mean"})),
    "contemp_only": _Def(
        107, _R, "constant", n=(3, 3), live=frozenset({"eq.struct"}), identity=("cccm_core",)
    ),
}

COMPOSITIONS: tuple[str, ...] = tuple(_COMPOSITIONS)
BLOCK_CONFIGS: tuple[str, ...] = tuple(_BLOCKS)


def composition_id(name: str) -> int:
    """Stable id for ``keys.validation_key``: the DESIGN §9.5 row (1–18), 101–107 (blocks) or
    201–205 (SOC/SUR dummy rows, WP25); Phase 3 rows 301–330 and blocks 401–408 (P3 §9.3–§9.4)."""
    if name in _P3_DEFS:
        return _P3_DEFS[name].number
    return _definition(name).number


def _definition(name: str) -> _Def:
    d = _COMPOSITIONS.get(name) or _BLOCKS.get(name)
    if d is None:
        raise KeyError(
            f"unknown joint spec {name!r}; compositions: {COMPOSITIONS}; blocks: {BLOCK_CONFIGS}; "
            f"Phase 3: {COMPOSITIONS_P3}; {BLOCK_CONFIGS_P3}"
        )
    return d


_VOL_FIELDS = frozenset(
    {"exact", "asis", "mixture", "t_scale", "outlier_K", "nu_grid", "zero_offset", "rw", "h0",
     "sigma2_prior", "transitory"}
)  # fmt: skip
_MEAN_FIELDS = frozenset({"const_path", "gram", "gram_tile", "cccm_precompute"})  # not n_dummy
_DESIGN_FIELDS = frozenset({"n", "lags", "T_eff", "presample", "t0"})
_SPEC_FIELDS = frozenset({"live", "obs", "fresh_latents", "raw_moments"})
_PRIOR_FIELDS = frozenset(
    {"prior", "prior_kwargs", "prior_updates", "sampler_prior_updates", "kernel_overrides"}
)
OVERRIDE_KEYS: frozenset[str] = (
    _VOL_FIELDS | _MEAN_FIELDS | _DESIGN_FIELDS | _SPEC_FIELDS | _PRIOR_FIELDS
)


def _const_path(mean: str, vol: VolStatic, mode: str) -> str:
    """The resolve rule of DESIGN §3.7.2 (``const_path="auto"``; with dummy rows too, WP25)."""
    if mean != "linear_structural" or not vol.constant_weights:
        return "weighted"
    return {"I": "eigen", "II": "fixed_factor", "iw_equivalent": "cholesky_const"}[mode]


def joint_spec(name: str, size: Literal["small", "full"] = "small", **overrides: Any) -> JointSpec:
    """Build the :class:`JointSpec` of a composition (``COMPOSITIONS``, DESIGN §9.5) or a block
    configuration (``BLOCK_CONFIGS``, §9.2).

    ``overrides`` (keys in :data:`OVERRIDE_KEYS`): ``VolStatic`` fields (``exact``, ``mixture``,
    ``nu_grid``, …), ``MeanStatic`` fields (``const_path``, ``gram``, …), the design (``n``,
    ``lags``, ``T_eff``, ``presample``, ``t0``), spec fields (``live``, ``obs``,
    ``fresh_latents``, ``raw_moments``), ``prior`` (explicit :class:`PriorArrays`),
    ``prior_kwargs`` (keywords of :func:`reference_prior`), ``prior_updates`` /
    ``sampler_prior_updates`` (``{leaf path: value}``; the latter changes the sampler's prior only)
    and ``kernel_overrides`` (``{name: callable}`` for ``compose``; names in
    ``protocols.OVERRIDABLE``).

    A Phase 3 name (``COMPOSITIONS_P3``, ``BLOCK_CONFIGS_P3``; D-61) is built by
    :func:`joint_spec_p3`.
    """
    if name in _P3_DEFS:
        return joint_spec_p3(name, size, **overrides)
    d = _definition(name)
    unknown = sorted(set(overrides) - OVERRIDE_KEYS)
    if unknown:
        raise TypeError(f"unknown joint_spec overrides {unknown}; allowed: {sorted(OVERRIDE_KEYS)}")
    if size not in ("small", "full"):
        raise ValueError(f"size must be 'small' or 'full', got {size!r}")
    full = size == "full"
    n = int(overrides.get("n", d.n[1] if full else d.n[0]))
    lags = int(overrides.get("lags", 2 if full else 1))
    T_eff = int(overrides.get("T_eff", 50 if full else 20))
    if "presample" in overrides:
        presample = np.asarray(overrides["presample"], dtype=np.float64)
    elif full and lags <= FULL_PRESAMPLE.shape[0] and n <= FULL_PRESAMPLE.shape[1]:
        presample = FULL_PRESAMPLE[:lags, :n]
    else:
        presample = np.zeros((lags, n))
    fixed = FixedDesign(presample, T_eff, float(overrides.get("t0", lags + 1)))

    vol_kw: dict[str, Any] = {
        "persistent": d.persistent, "rw": d.rw, "h0": d.h0, "sigma2_prior": d.sigma2_prior,
        "transitory": d.transitory, "nu_grid": TEST_NU_GRID, "exact": d.exact,
        "zero_offset": d.zero_offset, "coupled": d.mode != "I",
    }  # fmt: skip
    vol_kw.update({k: v for k, v in overrides.items() if k in _VOL_FIELDS})
    vol = VolStatic(**vol_kw)
    mean_kw: dict[str, Any] = {
        "kind": d.mean,
        "coef_scaling": "lambda" if d.mode == "II" else "none",
        "const_path": _const_path(d.mean, vol, d.mode),
        "n_dummy": n + 1 if d.dummies else 0,  # SOC rows of every variable + the SUR row
    }
    mean_kw.update({k: v for k, v in overrides.items() if k in _MEAN_FIELDS})
    static = StaticConfig(
        shapes=Shapes.of(n, lags, 1, bucket(T_eff, T_BUCKET)),
        struct=StructStatic(contemp_scaling="none" if d.mode == "I" else "lambda"),
        mean=MeanStatic(**mean_kw),
        coef=CoefPriorStatic(),
        vol=vol,
    )

    prior = overrides.get("prior")
    if prior is None:
        prior = reference_prior(static, **overrides.get("prior_kwargs", {}))
    if overrides.get("prior_updates"):
        prior = update_prior(prior, overrides["prior_updates"])
    sampler_prior = None
    if overrides.get("sampler_prior_updates"):
        sampler_prior = update_prior(prior, overrides["sampler_prior_updates"])

    kernel_ov: dict[str, Callable[..., Any]] = {k: IDENTITY_INTERNALS[k] for k in d.identity}
    kernel_ov.update(overrides.get("kernel_overrides", {}))
    raw = overrides.get("raw_moments")
    if raw is None:  # raw data moments only where they have finite variance (N06 §3.4)
        raw = testfns.raw_moments_finite(static, prior_tails(static, prior))
    return JointSpec(
        name=name,
        static=static,
        prior=prior,
        fixed=fixed,
        live=frozenset(overrides.get("live", d.live)),
        obs=overrides.get("obs", d.obs),
        sampler_prior=sampler_prior,
        overrides=tuple(kernel_ov.items()),
        fresh_latents=bool(overrides.get("fresh_latents", False)),
        raw_moments=bool(raw),
    )


# --------------------------------------------------------------------------- Phase 3 (D-61)
# docs/phase3/DESIGN.md §9.3–§9.4: rows 301–330 and block configurations 401–408, in a registry
# separate from Phase 2's (COMPOSITIONS / BLOCK_CONFIGS stay unchanged). Written by an author who
# wrote no Phase 3 kernel, from the design document only (§9, §13 WP30).

# Test priors of P3 §9.4 ("proper and tight"; everything not listed is the Phase 2 §9.2 prior).
TEST_PSI: dict[int, tuple[float, ...]] = {
    2: (0.5, 2.0),
    3: (0.5, 1.0, 2.0),
}  # fixed, unequal: r ≠ 1
TEST_A_GRID: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)  # equal weights
# CommonSV: ρ ~ TN(0.8, 0.15²), σ_h² ~ IG(5, 0.2). The parity options of row 325 (Beta ρ,
# ScaledChi2 σ_h²) have no values in §9.4; they take the Phase 2 §9.2 SV test values
# ((φ+1)/2 ~ Beta(10, 2), B_σ = 0.1), the "everything else P2 §9.2" rule.
TEST_CSV: dict[str, float] = {
    "rho_mean": 0.8, "rho_var": 0.15**2, "rho_a": 10.0, "rho_b": 2.0, "s2_shape": 5.0,
    "s2_scale": 0.2, "s2_B": 0.1,
}  # fmt: skip
# Conjugate rows: NIW Ψ = 0.5 I, d = n + 4, Ω_rr = 0.01 on every lag coefficient (own and cross),
# intercept Ω = 1 (Ω entries, not implied variances); own-lag prior mean 0.4.
TEST_CONJUGATE: dict[str, float] = {"psi": 0.5, "d_extra": 4.0, "omega_lag": 0.01, "omega_det": 1.0}

GLGroups = Literal["olcl-lagwise", "global", "equation-wise", "covariate-wise"]
Kernel = Literal["sweep", "global_step"]


def test_psi(n: int) -> np.ndarray:
    """The fixed unequal ψ of the GL test priors (``(0.5, 2)`` at ``n = 2``, ``(0.5, 1, 2)`` at
    ``n = 3``; geometric on ``[0.5, 2]`` otherwise)."""
    return np.asarray(TEST_PSI.get(n, tuple(np.geomspace(0.5, 2.0, n))), dtype=np.float64)


def gl_groups(groups: GLGroups, n: int, lags: int, n_det: int) -> tuple[np.ndarray, int]:
    """Group ids ``(n, k)`` (0 off the lag slots) and the group count ``G`` (D-38):
    ``olcl-lagwise`` ``2(l−1)`` (own lag ``l``) or ``2(l−1) + 1`` (cross lags of lag ``l``), across
    equations; ``global`` 0; ``equation-wise`` ``i``; ``covariate-wise`` ``(l−1)n + m``."""
    k = n_det + n * lags
    out = np.zeros((n, k), np.int32)
    for i in range(n):
        for lag in range(1, lags + 1):
            for m in range(n):
                r = n_det + (lag - 1) * n + m
                out[i, r] = {
                    "olcl-lagwise": 2 * (lag - 1) + (0 if m == i else 1),
                    "global": 0,
                    "equation-wise": i,
                    "covariate-wise": (lag - 1) * n + m,
                }[groups]
    G = {"olcl-lagwise": 2 * lags, "global": 1, "equation-wise": n, "covariate-wise": n * lags}
    return out, G[groups]


def gl_log_r(psi: np.ndarray, n: int, lags: int, n_det: int, form: str) -> np.ndarray:
    """``log r_j`` of the test priors (``scale="psi"``): :func:`reference.gl_log_r_reference`."""
    return gl_log_r_reference(psi, n, lags, n_det, form, "psi")


def gl_test_arrays(
    static: StaticConfig, groups: GLGroups, params: Mapping[str, Any], psi: np.ndarray
) -> tuple[GLArrays, GLShared]:
    """``GLArrays`` and ``GLShared`` of a GL test prior (``params``: ``global_scale``; ``a``
    (float or ``"grid"``), ``b``, ``c`` (float or ``"a/2"``, encoded ``c < 0``); ``tau0``,
    ``tau1`` (SSVS spike and slab s.d. as multiples of ``√r_j``), ``p`` (float or ``"beta"``),
    ``s1``, ``s2``). Unused ``GLShared`` fields are 0."""
    s, coef = static.shapes, static.coef
    n, lags, nd, k = s.n, s.lags, s.n_det, s.k
    group, G = gl_groups(groups, n, lags, nd)
    if G != coef.n_groups:
        raise ValueError(
            f"groups={groups!r} gives G = {G}, but CoefPriorStatic.n_groups = {coef.n_groups}"
        )
    mask = np.zeros((n, k))
    mask[:, nd:] = 1.0
    log_r = gl_log_r(psi, n, lags, nd, static.mean.kind)
    ssvs: SSVSArrays | Empty = Empty()
    if coef.kind == "ssvs":
        half = 0.5 * log_r
        tau0 = np.where(mask > 0, math.log(params["tau0"]) + half, 0.0)
        tau1 = np.where(mask > 0, math.log(params["tau1"]) + half, 0.0)
        ssvs = SSVSArrays(log_tau0=tau0, log_tau1=tau1)
    a = params.get("a", 1.0)
    if coef.a_grid:
        support = np.asarray(TEST_A_GRID, dtype=np.float64)
        logw = np.full(support.shape, -math.log(support.size))
        log_a_fixed = 0.0
    else:
        support, logw = np.zeros(1), np.zeros(1)
        log_a_fixed = (
            math.log(float(a))
            if coef.kind in ("normal_gamma", "r2d2", "dirichlet_laplace")
            else 0.0
        )
    c = params.get("c", 0.0)
    p = params.get("p", 0.0)
    shared = GLShared(
        b=np.asarray(float(params.get("b", 0.0))),
        c=np.asarray(-1.0 if c == "a/2" else float(c)),
        global_scale=np.asarray(float(params.get("global_scale", 0.0))),
        log_a_fixed=np.asarray(log_a_fixed),
        a_support=support,
        a_logw=logw,
        p_fixed=np.asarray(0.0 if p == "beta" else float(p)),
        s1=np.asarray(float(params.get("s1", 0.0))),
        s2=np.asarray(float(params.get("s2", 0.0))),
    )
    return GLArrays(mask=mask, group=group, log_r=log_r, ssvs=ssvs), shared


def common_sv_test_prior(n: int, **updates: float) -> CommonSVPrior:
    """``CommonSVPrior`` of the CommonSV test prior (``TEST_CSV``; every option's fields filled,
    the static selects which apply)."""
    v = {**TEST_CSV, **updates}
    return CommonSVPrior(
        rho_mean=np.asarray(v["rho_mean"]), rho_var=np.asarray(v["rho_var"]),
        rho_a=np.asarray(v["rho_a"]), rho_b=np.asarray(v["rho_b"]),
        s2_shape=np.asarray(v["s2_shape"]), s2_scale=np.asarray(v["s2_scale"]),
        s2_B=np.asarray(v["s2_B"]), n_eq=np.asarray(float(n)),
    )  # fmt: skip


def conjugate_test_prior(
    static: StaticConfig,
    *,
    psi: float = TEST_CONJUGATE["psi"],
    d_extra: float = TEST_CONJUGATE["d_extra"],
    omega_lag: float = TEST_CONJUGATE["omega_lag"],
    omega_det: float = TEST_CONJUGATE["omega_det"],
    own_lag_mean: float = 0.4,
) -> PriorArrays:
    """The conjugate (Mode II at ``cross = contemp = 1``) test prior of P3 §9.4: ``Σ ~ IW(ψI, d)``
    with ``d = n + d_extra`` and ``B | Σ ~ MN(M, Σ ⊗ Ω)``, written structurally (P2 §6.1; N08
    §A11 item 2): ``λ_i ~ IG((d − n + ι)/2, ψ/2)``, ``a_ij | λ_i ~ N(0, λ_i/ψ)``,
    ``δ_i | λ_i ~ N(m_i, λ_i Ω)``; with SOC/SUR rows (``MeanStatic.n_dummy``) those of the Phase 2
    test prior at the Mode II weight 1."""
    if static.mean.coef_scaling != "lambda":
        raise ValueError("the conjugate test prior is a Mode II (λ-scaled) prior")
    s = static.shapes
    n, nd, k = s.n, s.n_det, s.k
    base = jax.tree.map(np.asarray, reference_prior(static, own_lag_mean=own_lag_mean))
    prec0 = np.full((n, k), 1.0 / omega_lag)
    prec0[:, :nd] = 1.0 / omega_det
    cmask = np.asarray(base.eq.contemp.cmask)
    contemp = dataclasses.replace(base.eq.contemp, prec0=np.where(cmask > 0, psi, 1.0))
    iota = np.arange(1, n + 1, dtype=np.float64)
    core = ConstantPrior(shape=(d_extra + iota) / 2.0, scale=np.full(n, psi / 2.0))
    eq = dataclasses.replace(
        base.eq,
        coef=dataclasses.replace(base.eq.coef, prec0=prec0),
        contemp=contemp,
        vol_core=core,
    )
    return PriorArrays(eq=eq, shared=base.shared)


def reference_prior_p3(
    static: StaticConfig,
    *,
    groups: GLGroups = "olcl-lagwise",
    gl: Mapping[str, Any] | None = None,
    csv: Mapping[str, float] | None = None,
    **prior_kwargs: Any,
) -> PriorArrays:
    """The Phase 3 test prior of ``static`` (P3 §9.4): Mode II → :func:`conjugate_test_prior`,
    else the Phase 2 :func:`reference_prior` (``prior_kwargs``); plus the GL arrays
    (:func:`gl_test_arrays`, ψ of :func:`test_psi`), the CommonSV prior
    (:func:`common_sv_test_prior`) and the shared-factor precision where ``static`` has them."""
    if static.mean.coef_scaling == "lambda":
        base = conjugate_test_prior(static, **prior_kwargs)
    else:
        base = jax.tree.map(np.asarray, reference_prior(static, **prior_kwargs))
    n = static.shapes.n
    eq, shared = base.eq, base.shared
    if static.coef.varies:
        arrays, gl_shared = gl_test_arrays(static, groups, dict(gl or {}), test_psi(n))
        eq = dataclasses.replace(eq, coef=dataclasses.replace(eq.coef, gl=arrays))
        shared = dataclasses.replace(shared, gl=gl_shared)
    if static.vol.common:
        shared = dataclasses.replace(shared, csv=common_sv_test_prior(n, **dict(csv or {})))
    if static.mean.const_path == "shared_factor":  # the λ-free precision of θ = [δ | a]
        prec = np.asarray(eq.coef.prec0)
        if not np.all(prec == prec[0]):
            raise ValueError("the shared-factor path needs an equation-invariant prior precision")
        a_prec = np.asarray(eq.contemp.prec0)[-1]  # the last equation's row: every slot active
        shared = dataclasses.replace(shared, shared_prec=np.concatenate([prec[0], a_prec]))
    tree = PriorArrays(eq=eq, shared=shared)
    if jax.tree.structure(tree) != jax.tree.structure(prior_template(static)):
        raise ValueError("internal: the P3 test prior does not match the StaticConfig's template")
    return jax.tree.map(jnp.asarray, tree)


# ---- the registry


@dataclass(frozen=True)
class _DefP3:
    number: int
    mean: Literal["linear_structural", "linear_reduced"]
    persistent: Literal["constant", "sv"] = "constant"
    transitory: Literal["none", "outlier", "t", "outlier_t"] = "none"
    mode: Literal["I", "II", "iw_equivalent"] = "I"
    common: bool = False
    coef: str = "fixed_diagonal"
    groups: GLGroups = "olcl-lagwise"
    gl: tuple[tuple[str, Any], ...] = ()
    design: Literal["exogenous", "lagged"] = "exogenous"
    n: tuple[int, int] = (3, 3)  # (small, full)
    dummies: bool = False
    const_path: str | None = None  # None: the resolve rule (shared_factor for conjugate × CommonSV)
    csv_rho_prior: Literal["truncated_normal", "beta"] = "truncated_normal"
    csv_sigma2_prior: Literal["inv_gamma", "scaled_chi2"] = "inv_gamma"
    csv_block: int = 100
    live: frozenset[str] = ALL
    kernel: Kernel = "sweep"
    global_block: int = 0
    refresh: Literal["none", "gl_coefficients"] = "none"
    data_irrelevant: bool = False


_HS = (("global_scale", 0.1),)
_NG = (("a", 0.5), ("b", 2.0), ("c", 1.0))
_NG_GRID_TIED = (("a", "grid"), ("b", 2.0), ("c", "a/2"))
_NG_GRID_C1 = (("a", "grid"), ("b", 2.0), ("c", 1.0))
_NG_LAGGED_RED = (("a", 0.5), ("b", 10.0), ("c", 0.01))  # row 316 (E93)
_NG_LAGGED_STR = (("a", 0.5), ("b", 10.0), ("c", 0.005))  # row 319 (E93)
_R2D2 = (("a", 0.5), ("b", 2.0))
_R2D2_GRID = (("a", "grid"), ("b", 2.0))
_DL = (("a", 0.5),)
_DL_GRID = (("a", "grid"),)
_SSVS = (("tau0", 0.01), ("tau1", 0.5), ("p", 0.3))
_SSVS_BETA = (("tau0", 0.01), ("tau1", 0.5), ("p", "beta"), ("s1", 2.0), ("s2", 3.0))
_SSVS_LAGGED = (("tau0", 0.01), ("tau1", 0.05), ("p", 0.3))  # row 330: slab 0.05·√r (E93)
_HSK, _NGK, _R2K, _DLK, _SSK = "horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"

# P3 §9.4 rows 301–330.
_COMPOSITIONS_P3: dict[str, _DefP3] = {
    "structural_constant_hs": _DefP3(301, _S, coef=_HSK, gl=_HS, n=(2, 3)),
    "structural_sv_hs": _DefP3(302, _S, "sv", coef=_HSK, gl=_HS, n=(2, 3)),
    "structural_constant_ng": _DefP3(303, _S, coef=_NGK, gl=_NG),
    "structural_constant_ng_grid": _DefP3(304, _S, coef=_NGK, gl=_NG_GRID_TIED),
    "structural_constant_r2d2": _DefP3(305, _S, coef=_R2K, gl=_R2D2),
    "structural_constant_r2d2_grid": _DefP3(306, _S, coef=_R2K, gl=_R2D2_GRID),
    "structural_constant_dl": _DefP3(307, _S, coef=_DLK, gl=_DL),
    "structural_constant_dl_grid": _DefP3(308, _S, coef=_DLK, gl=_DL_GRID),
    "structural_constant_ssvs": _DefP3(309, _S, coef=_SSK, gl=_SSVS),
    "structural_constant_ssvs_beta": _DefP3(310, _S, coef=_SSK, gl=_SSVS_BETA),
    "reduced_sv_hs_global": _DefP3(311, _R, "sv", coef=_HSK, gl=_HS, groups="global"),
    "reduced_sv_ng_global": _DefP3(312, _R, "sv", coef=_NGK, gl=_NG, groups="global"),
    "reduced_sv_r2d2_global": _DefP3(313, _R, "sv", coef=_R2K, gl=_R2D2, groups="global"),
    "reduced_sv_dl_global": _DefP3(314, _R, "sv", coef=_DLK, gl=_DL, groups="global"),
    "reduced_sv_ssvs_global": _DefP3(315, _R, "sv", coef=_SSK, gl=_SSVS, groups="global"),
    "reduced_sv_ng_lagged": _DefP3(316, _R, "sv", coef=_NGK, gl=_NG_LAGGED_RED, design="lagged"),
    "structural_sv_outliers_hs_eqwise": _DefP3(
        317, _S, "sv", transitory="outlier", coef=_HSK, gl=_HS, groups="equation-wise"
    ),
    "structural_iw_equivalent_hs": _DefP3(318, _S, mode="iw_equivalent", coef=_HSK, gl=_HS),
    "structural_sv_ng_lagged": _DefP3(319, _S, "sv", coef=_NGK, gl=_NG_LAGGED_STR, design="lagged"),
    "conjugate_commonsv_dummies": _DefP3(
        320, _S, mode="II", common=True, design="lagged", dummies=True
    ),
    "structural_commonsv": _DefP3(321, _S, common=True, design="lagged", n=(2, 3)),
    "reduced_commonsv": _DefP3(322, _R, common=True, design="lagged"),
    "conjugate_commonsv_dummies_weighted": _DefP3(
        323, _S, mode="II", common=True, design="lagged", dummies=True, const_path="weighted"
    ),
    "structural_commonsv_hs": _DefP3(324, _S, common=True, coef=_HSK, gl=_HS),
    "conjugate_commonsv_parity": _DefP3(
        325,
        _S,
        mode="II",
        common=True,
        design="lagged",
        csv_rho_prior="beta",
        csv_sigma2_prior="scaled_chi2",
    ),
    "structural_constant_ng_grid_c1": _DefP3(326, _S, coef=_NGK, gl=_NG_GRID_C1),
    "reduced_constant_hs_global": _DefP3(327, _R, coef=_HSK, gl=_HS, groups="global"),
    "reduced_commonsv_ng": _DefP3(328, _R, common=True, coef=_NGK, gl=_NG),
    "conjugate_commonsv_block7": _DefP3(
        329, _S, mode="II", common=True, design="lagged", csv_block=7
    ),
    "structural_sv_ssvs_lagged": _DefP3(330, _S, "sv", coef=_SSK, gl=_SSVS_LAGGED, design="lagged"),
}

_GL_LIVE = frozenset({"eq.coef", "glob.coef"})


def _gl_scales_only(number: int, kind: str, gl: tuple[tuple[str, Any], ...]) -> _DefP3:
    """P3 §9.3 rows 401–405: the GL scale step alone (``G_COEF_GLOBAL``), then the coefficients
    redrawn from ``p(d | current scales)``; the data are irrelevant (zero)."""
    return _DefP3(
        number, _S, coef=kind, gl=gl, live=_GL_LIVE, kernel="global_step",
        global_block=int(Block.G_COEF_GLOBAL), refresh="gl_coefficients", data_irrelevant=True,
    )  # fmt: skip


# P3 §9.3 block configurations 401–408 (families and hyperpriors chosen to exercise every line of
# §3.2: grid a with fixed c for NG, grids for R2D2 and DL, the Beta p for SSVS).
_BLOCKS_P3: dict[str, _DefP3] = {
    "gl_scales_only_horseshoe": _gl_scales_only(401, _HSK, _HS),
    "gl_scales_only_normal_gamma": _gl_scales_only(402, _NGK, _NG_GRID_C1),
    "gl_scales_only_r2d2": _gl_scales_only(403, _R2K, _R2D2_GRID),
    "gl_scales_only_dirichlet_laplace": _gl_scales_only(404, _DLK, _DL_GRID),
    "gl_scales_only_ssvs": _gl_scales_only(405, _SSK, _SSVS_BETA),
    "gl_mean_only": _DefP3(406, _S, coef=_HSK, gl=_HS, live=frozenset({"eq.mean", "eq.struct"})),
    "common_sv_only": _DefP3(
        407,
        _S,
        common=True,
        design="lagged",
        live=frozenset({"glob.vol"}),
        kernel="global_step",
        global_block=int(Block.G_COMMON_SV),
    ),
    "conjugate_niw_only": _DefP3(
        408,
        _S,
        mode="II",
        common=True,
        design="lagged",
        dummies=True,
        live=frozenset({"eq.mean", "eq.struct", "eq.vol"}),
    ),
}

COMPOSITIONS_P3: tuple[str, ...] = tuple(_COMPOSITIONS_P3)
BLOCK_CONFIGS_P3: tuple[str, ...] = tuple(_BLOCKS_P3)
_P3_DEFS: dict[str, _DefP3] = {**_COMPOSITIONS_P3, **_BLOCKS_P3}
LAGGED_P3: tuple[str, ...] = tuple(nm for nm, d in _P3_DEFS.items() if d.design == "lagged")

_P3_FIELDS = frozenset(
    {"design_seed", "csv_block", "csv_rho_prior", "csv_sigma2_prior", "gl", "csv"}
)
OVERRIDE_KEYS_P3: frozenset[str] = OVERRIDE_KEYS | _P3_FIELDS


def _p3_static(
    d: _DefP3, n: int, lags: int, T_eff: int, overrides: Mapping[str, Any]
) -> StaticConfig:
    vol_kw: dict[str, Any] = {
        "persistent": d.persistent, "transitory": d.transitory, "nu_grid": TEST_NU_GRID,
        "coupled": d.mode != "I", "common": d.common, "csv_block": d.csv_block,
        "csv_rho_prior": d.csv_rho_prior, "csv_sigma2_prior": d.csv_sigma2_prior,
    }  # fmt: skip
    vol_kw.update({k: v for k, v in overrides.items() if k in _VOL_FIELDS})
    vol_kw.update(
        {
            k: overrides[k]
            for k in ("csv_block", "csv_rho_prior", "csv_sigma2_prior")
            if k in overrides
        }
    )
    vol = VolStatic(**vol_kw)
    varies = d.coef != "fixed_diagonal"
    G = gl_groups(d.groups, n, lags, 1)[1] if varies else 0
    params = dict(d.gl)
    coef = CoefPriorStatic(
        kind=d.coef,  # type: ignore[arg-type]
        n_groups=G,
        a_grid=len(TEST_A_GRID) if params.get("a") == "grid" else 0,
        p_beta=params.get("p") == "beta",
    )
    if d.const_path is not None:
        path = d.const_path
    elif d.mode == "II" and d.common:
        path = "shared_factor"  # the conjugate × CommonSV resolve rule (P3 §5.3)
    elif varies:
        path = "weighted"  # rule (11)
    else:
        path = _const_path(d.mean, vol, d.mode)
    mean_kw: dict[str, Any] = {
        "kind": d.mean,
        "coef_scaling": "lambda" if d.mode == "II" else "none",
        "const_path": path,
        "n_dummy": n + 1 if d.dummies else 0,
        "coef_log": varies,
    }
    mean_kw.update({k: v for k, v in overrides.items() if k in _MEAN_FIELDS})
    return StaticConfig(
        shapes=Shapes.of(n, lags, 1, bucket(T_eff, T_BUCKET)),
        struct=StructStatic(contemp_scaling="none" if d.mode == "I" else "lambda"),
        mean=MeanStatic(**mean_kw),
        coef=coef,
        vol=vol,
    )


def joint_spec_p3(
    name: str, size: Literal["small", "full"] = "small", **overrides: Any
) -> JointSpecP3:
    """Build the :class:`JointSpecP3` of a Phase 3 row (``COMPOSITIONS_P3``, P3 §9.4) or block
    configuration (``BLOCK_CONFIGS_P3``, §9.3) **without composing a production program** (the
    kernels are WP31/WP32's; :meth:`JointSpecP3.step` composes lazily).

    Sizes as Phase 2: ``"small"`` ``p = 1``, ``T_eff = 20``; ``"full"`` ``p = 2``, ``T_eff = 50``;
    ``n`` from the row (§9.4 "n" column). Exogenous rows (D-56) draw their fixed ``X`` from the
    configuration seed ``(row, 0 | 1)`` (override ``design_seed``); lagged rows use the Phase 2
    presample. ``overrides``: the Phase 2 keys (:data:`OVERRIDE_KEYS`; ``prior_kwargs`` go to
    :func:`reference_prior_p3`) plus ``design_seed``, ``csv_block``, ``csv_rho_prior``,
    ``csv_sigma2_prior``, ``gl`` (test-prior parameters merged into the row's) and ``csv``
    (``CommonSVPrior`` values)."""
    d = _P3_DEFS.get(name)
    if d is None:
        raise KeyError(
            f"unknown Phase 3 joint spec {name!r}; {COMPOSITIONS_P3}; {BLOCK_CONFIGS_P3}"
        )
    unknown = sorted(set(overrides) - OVERRIDE_KEYS_P3)
    if unknown:
        raise TypeError(
            f"unknown joint_spec overrides {unknown}; allowed: {sorted(OVERRIDE_KEYS_P3)}"
        )
    if size not in ("small", "full"):
        raise ValueError(f"size must be 'small' or 'full', got {size!r}")
    full = size == "full"
    n = int(overrides.get("n", d.n[1] if full else d.n[0]))
    lags = int(overrides.get("lags", 2 if full else 1))
    T_eff = int(overrides.get("T_eff", 50 if full else 20))
    if d.gl or "gl" in overrides:
        d = dataclasses.replace(
            d, gl=tuple({**dict(d.gl), **dict(overrides.get("gl", {}))}.items())
        )
    static = _p3_static(d, n, lags, T_eff, overrides)
    if "presample" in overrides:
        presample = np.asarray(overrides["presample"], dtype=np.float64)
    elif (
        full
        and d.design == "lagged"
        and lags <= FULL_PRESAMPLE.shape[0]
        and n <= FULL_PRESAMPLE.shape[1]
    ):
        presample = FULL_PRESAMPLE[:lags, :n]
    else:
        presample = np.zeros((lags, n))
    t0 = float(overrides.get("t0", lags + 1))
    if d.design == "exogenous":
        seed = overrides.get("design_seed", (d.number, int(full)))
        fixed = FixedDesign.exogenous_design(
            presample, T_eff, n_det=static.shapes.n_det, k=static.shapes.k, seed=seed, t0=t0
        )
    else:
        fixed = FixedDesign(presample, T_eff, t0)

    prior = overrides.get("prior")
    if prior is None:
        prior = reference_prior_p3(
            static, groups=d.groups, gl=dict(d.gl), csv=overrides.get("csv"),
            **overrides.get("prior_kwargs", {}),
        )  # fmt: skip
    if overrides.get("prior_updates"):
        prior = update_prior(prior, overrides["prior_updates"])
    sampler_prior = None
    if overrides.get("sampler_prior_updates"):
        sampler_prior = update_prior(prior, overrides["sampler_prior_updates"])
    raw = overrides.get("raw_moments")
    if raw is None:
        raw = testfns.raw_moments_finite(static, prior_tails(static, prior))
    return JointSpecP3(
        name=name,
        static=static,
        prior=prior,
        fixed=fixed,
        live=frozenset(overrides.get("live", d.live)),
        obs=overrides.get("obs", "gaussian"),
        sampler_prior=sampler_prior,
        overrides=tuple(dict(overrides.get("kernel_overrides", {})).items()),
        fresh_latents=bool(overrides.get("fresh_latents", False)),
        raw_moments=bool(raw) and not d.data_irrelevant,
        kernel=d.kernel,
        global_block=d.global_block,
        refresh=d.refresh,
        data_irrelevant=d.data_irrelevant,
    )


def _data_dependent(name: str) -> bool:
    """Test functions that read the data (dropped when the data are irrelevant)."""
    return (
        name.startswith("data.")
        or name in ("loglik", "joint.W")
        or name.endswith((".joint.loglik", ".joint.rss_std", ".joint.loglik_marginal"))
    )


@functools.cache
def production_program(
    static: StaticConfig, overrides: tuple[tuple[str, Callable[..., Any]], ...] = ()
) -> tuple[Any, Callable[..., Any]]:
    """``(program, prepare)`` of the production engine for ``static`` (``compose`` and the
    runner's ``prepare``), composed on first use; raises ``NotImplementedError`` while a kernel
    is missing (``registry.load``)."""
    try:
        from rai_bvar.engines.mcmc.compose import compose
        from rai_bvar.engines.mcmc.runner import make_runner
    except ImportError as err:  # pragma: no cover - Phase 2 provides both
        raise NotImplementedError(f"the production engine is not available: {err}") from err
    ov = dict(overrides) or None
    program = compose(static, overrides=ov)
    have = {int(g.block) for g in program.global_steps}
    for needed, block, owner in (
        (static.coef.varies, Block.G_COEF_GLOBAL, "WP31 (GLScaleStep)"),
        (static.vol.common, Block.G_COMMON_SV, "WP32 (CommonSVStep)"),
    ):
        if needed and int(block) not in have:  # never validate a program without its P3 step
            raise NotImplementedError(
                f"the composed program has no {block.name} global step for this Phase 3 "
                f"composition; it is implemented by {owner}"
            )
    return program, make_runner(static, overrides=ov).prepare


def run_global_step(
    program: Any,
    block: int,
    key: Array,
    it: Array,
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    cache: Any,
) -> ChainState:
    """One production global step alone (the step of ``program`` whose ``block`` matches), with
    the sweep's key ``step_key(key, it, block)``, its status recording and a ``SweepCtx`` whose
    ``eq_out`` holds the reference residuals and effective log variances of ``st`` (what the
    equation phase would hand a post step with those coefficients and volatilities)."""
    from rai_bvar.engines.mcmc import keys
    from rai_bvar.engines.mcmc.sweep import record_status

    found = [g for g in program.global_steps if int(g.block) == int(block)]
    if len(found) != 1:
        raise ValueError(f"the program has {len(found)} global steps with block {block}")
    g = found[0]
    static = program.static
    eq_out = EqOut(e=residuals(st, data, prior, static), log_lam=state_log_var(st, static))
    ctx = SweepCtx(data=data, prior=prior, cache=cache, eq_out=eq_out)
    new, info = g.sample(keys.step_key(key, it, g.block), st, ctx, program.eq_map)
    return record_status(new, info, g.block, 0, it)


@dataclass(frozen=True, eq=False)
class JointSpecP3(JointSpec):
    """A Phase 3 joint model (P3 §9.3–§9.4); a :class:`JointSpec` with:

    * ``kernel="sweep"``: :meth:`JointSpec.step` (prepare, one production sweep, frozen blocks
      restored); exact for blocks held after the live ones (406 and 408);
    * ``kernel="global_step"``: only the production global step ``global_block`` runs
      (:func:`run_global_step`), then the frozen leaves are restored: the block configurations
      whose live step is a post step that the held blocks precede (401–405 ``G_COEF_GLOBAL``, 407
      ``G_COMMON_SV``);
    * ``refresh="gl_coefficients"``: after the step the coefficients are redrawn from
      ``p(d | current scales)`` (``simulate.gl_coefficient_draw``, key
      ``step_key(key, it, Block.EQ_COEF_LOCAL)``, a block id no Phase 3 step uses; P3 D-40), so
      the chain is a two-block Gibbs sampler of the prior ``p(scales, d)``;
    * ``data_irrelevant``: :meth:`simulate` returns zero targets (the live step reads no data) and
      the data-dependent test functions are dropped.
    """

    kernel: Kernel = "sweep"
    global_block: int = 0
    refresh: Literal["none", "gl_coefficients"] = "none"
    data_irrelevant: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kernel not in ("sweep", "global_step"):
            raise ValueError(f"kernel must be 'sweep' or 'global_step', got {self.kernel!r}")
        if self.kernel == "global_step" and int(self.global_block) not in (
            int(Block.G_COEF_GLOBAL),
            int(Block.G_COMMON_SV),
        ):
            raise ValueError(f"global_block {self.global_block} is not a Phase 3 global step")
        if self.refresh not in ("none", "gl_coefficients"):
            raise ValueError(f"refresh must be 'none' or 'gl_coefficients', got {self.refresh!r}")
        if self.refresh == "gl_coefficients" and not self.static.coef.varies:
            raise ValueError("refresh='gl_coefficients' needs a varying coefficient prior")
        if self.data_irrelevant and self.obs != "gaussian":
            raise ValueError("data_irrelevant specs simulate no observations (obs='gaussian')")
        if self.fixed.exogenous and self.fixed.X.shape != (self.fixed.T_eff, self.static.shapes.k):
            raise ValueError("the exogenous X must have shape (T_eff, k)")

    @cached_property
    def function_flags(self) -> dict[str, bool]:
        flags = testfns.test_function_spec(
            self.static, self.prior, self.fixed.T_eff, raw_moments=self.raw_moments,
            tails=self.tails,
        )  # fmt: skip
        if self.data_irrelevant:
            flags = {k: f for k, f in flags.items() if not _data_dependent(k)}
        return flags

    def simulate(self, key: Array, st: ChainState) -> EngineData:
        """``y ~ p(y | θ)`` (exogenous: given the fixed ``X``, no lag feedback); zero targets when
        the data are irrelevant."""
        if not self.data_irrelevant:
            return super().simulate(key, st)
        s = self.static.shapes
        Y = jnp.zeros((s.T_pad, s.n), _F64)
        return simulate_engine_data(Y, self.static, self.fixed)

    def step(self, key: Array, it: Array, st: ChainState, data: EngineData) -> ChainState:
        from rai_bvar.engines.mcmc import keys

        prior = self.prior if self.sampler_prior is None else self.sampler_prior
        program, prepare = production_program(self.static, self.overrides)  # checks the P3 steps
        if self.kernel == "sweep":
            new = super().step(key, it, st, data)
        else:
            new = run_global_step(
                program, self.global_block, key, it, st, data, prior, prepare(data, prior)
            )
            new = hold_frozen(new, st, self.live)
        if self.refresh == "gl_coefficients":
            rkey = keys.step_key(key, it, Block.EQ_COEF_LOCAL)
            new = gl_coefficient_draw(rkey, new, prior, self.static)
        return new


# ---- explosive prior mass of the lagged rows (P3 D-56 target 2e-4, E93: the level Phase 2
# measured for its corrected prior at n = 3, p = 2; Phase 2's bound is P(ρ > 1) < 1e-3, N06 §9.1)

EXPLOSIVE_BOUND = 2e-4
EXPLOSIVE_DRAWS = 1_000_000
# Measured by prior simulation (WP30, 2026-10-25): explosive_mass(joint_spec(name, size), 10**6,
# seed=0); counts of draws whose companion matrix has spectral radius >= 1. P3 §9.4 expected 316:
# 1.7e-5, 319: 4.6e-5, 330: 1.3e-5 (superseded: 0 measured, E150) and 1.2e-4 for the conjugate rows
# at p = 2. Above the target by the Monte Carlo rule of E150 (findings, not retuned): 320/full
# 2.38e-4 and 408/full 2.39e-4 (the conjugate prior with the SOC/SUR rows; without them 325/full
# 1.36e-4, 329/full 1.13e-4). Within it: 323/full 2.04e-4 and 321/full 2.07e-4 (the Phase 2 §9.2
# Mode I test prior, reproducing Phase 2's measured level; its twin 407/full 1.89e-4).
EXPLOSIVE_MASS_P3: dict[str, dict[str, int]] = {
    "reduced_sv_ng_lagged": {"small": 4, "full": 18},
    "structural_sv_ng_lagged": {"small": 0, "full": 0},
    "conjugate_commonsv_dummies": {"small": 18, "full": 238},
    "structural_commonsv": {"small": 2, "full": 207},
    "reduced_commonsv": {"small": 0, "full": 0},
    "conjugate_commonsv_dummies_weighted": {"small": 12, "full": 204},
    "conjugate_commonsv_parity": {"small": 27, "full": 136},
    "conjugate_commonsv_block7": {"small": 19, "full": 113},
    "structural_sv_ssvs_lagged": {"small": 0, "full": 0},
    "common_sv_only": {"small": 66, "full": 189},
    "conjugate_niw_only": {"small": 16, "full": 239},
}


def reduced_form_coefficients(st: ChainState, prior: PriorArrays, static: StaticConfig) -> Array:
    """``B`` (``k × n``) with ``y_t' = x_t'B + u_t'``: structural ``(A⁻¹Γ)'`` (D3 ``Γ``), reduced
    ``Π``."""
    if static.mean.kind == "linear_reduced":
        return st.glob.mean.Pi
    A = unit_lower(st.eq.struct.a)
    return jax.scipy.linalg.solve_triangular(
        A, structural_gamma(st, prior), lower=True, unit_diagonal=True
    ).T


def spectral_radius(B: np.ndarray, n: int, lags: int, n_det: int) -> np.ndarray:
    """Largest modulus of the companion-matrix eigenvalues of ``B`` (``(..., k, n)``), with lag
    block ``Φ_l = B[n_det + (l−1)n : n_det + ln]ᵀ`` (``y_t = Σ_l Φ_l y_{t−l} + …``)."""
    B = np.asarray(B, dtype=np.float64)
    batch = B.shape[:-2]
    C = np.zeros((*batch, n * lags, n * lags))
    for lag in range(lags):
        C[..., :n, lag * n : (lag + 1) * n] = np.swapaxes(
            B[..., n_det + lag * n : n_det + (lag + 1) * n, :], -1, -2
        )
    if lags > 1:
        C[..., n:, : n * (lags - 1)] = np.eye(n * (lags - 1))
    return np.max(np.abs(np.linalg.eigvals(C)), axis=-1)


def explosive_mass(
    spec: JointSpec, draws: int, *, seed: int = 0, chunk: int = 100_000
) -> tuple[int, int]:
    """``(count, draws)``: prior draws whose companion matrix has spectral radius ``≥ 1``, by prior
    simulation (``spec.prior_sample``); draw ``r`` uses ``validation_key(seed, Tier.U, row, r)``."""
    from rai_bvar.engines.mcmc.keys import Tier, validation_key

    s = spec.static.shapes
    row = composition_id(spec.name)

    @jax.jit
    def coefs(reps: Array) -> Array:
        keys_ = jax.vmap(lambda r: validation_key(seed, int(Tier.U), row, r))(reps)
        sts = jax.vmap(spec.prior_sample)(keys_)
        return jax.vmap(lambda st: reduced_form_coefficients(st, spec.prior, spec.static))(sts)

    count = 0
    for start in range(0, draws, chunk):
        m = min(chunk, draws - start)
        reps = np.arange(start, start + chunk, dtype=np.uint32)  # fixed shape: one compile
        rad = spectral_radius(np.asarray(coefs(reps))[:m], s.n, s.lags, s.n_det)
        count += int(np.sum(rad >= 1.0))
    return count, draws


__all__ = [
    "ALL",
    "BLOCK_CONFIGS",
    "BLOCK_CONFIGS_P3",
    "COMPOSITIONS",
    "COMPOSITIONS_P3",
    "EXPLOSIVE_BOUND",
    "EXPLOSIVE_DRAWS",
    "EXPLOSIVE_MASS_P3",
    "FULL_PRESAMPLE",
    "IDENTITY_INTERNALS",
    "LAGGED_P3",
    "OVERRIDE_KEYS",
    "OVERRIDE_KEYS_P3",
    "TEST_A_GRID",
    "TEST_CONJUGATE",
    "TEST_CSV",
    "TEST_NU_GRID",
    "TEST_PSI",
    "TEST_YBAR0",
    "FixedDesign",
    "JointSpec",
    "JointSpecP3",
    "common_sv_test_prior",
    "composition_id",
    "conjugate_test_prior",
    "explosive_mass",
    "gl_groups",
    "gl_log_r",
    "gl_test_arrays",
    "hold_frozen",
    "identity_cccm_core",
    "joint_spec",
    "joint_spec_p3",
    "production_program",
    "production_sweep",
    "reduced_form_coefficients",
    "reference_prior",
    "reference_prior_p3",
    "run_global_step",
    "spectral_radius",
    "test_psi",
    "update_prior",
    "with_fresh_latents",
]
