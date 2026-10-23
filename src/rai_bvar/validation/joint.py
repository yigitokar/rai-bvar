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
SOC/SUR dummy rows (WP25; compositions ``*_dummies``, ids 201–204): the Phase 1 rows
(``priors.dummies.dummy_rows``) of a fixed ``ȳ₀ = (1.5, −1.0, 0.8)`` (the PR presample is zero),
SOC rows for every variable and the SUR row, ``μ = δ = 0.5`` (rows of size ~2–3, enough to move the
intercept and contemporaneous priors), weight 1 (``= 1/E[λ]`` under ``IG(5, 4)``; Mode II: the
λ-free 1). Sizes: ``"small"`` (PR) ``n = 2`` structural / ``3`` reduced, ``p = 1``, intercept, ``T_eff = 20``,
``y_{1−p:0} = 0``; ``"full"`` (nightly) ``n = 3``, ``p = 2``, ``T_eff = 50``, a fixed stored
presample. The §9.5 table's ``n`` column overrides both sizes where it gives one value (rows 6–13:
3; rows 14–16: 2; row 17: 1). ``T_eff`` is never a multiple of 32, so the padding masks are always
exercised.

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
from rai_bvar.engines.mcmc.state import (
    ChainState,
    CoefPriorArrays,
    ConstantPrior,
    ContempPriorArrays,
    DummyRows,
    Empty,
    EngineData,
    EqPrior,
    PriorArrays,
    SharedPrior,
    StepInfo,
    SVPrior,
    TransitoryPrior,
    leaf_path,
    prior_template,
    state_template,
)
from rai_bvar.priors.dummies import dummy_rows
from rai_bvar.validation import testfns
from rai_bvar.validation.log_joint import log_joint
from rai_bvar.validation.reference import PriorTails, prior_tails
from rai_bvar.validation.simulate import prior_sample, simulate_aux_obs, simulate_obs, tmask_of

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
    ``t0`` of the first target row (``p + 1`` reproduces ``data.design.lagged_design``)."""

    presample: np.ndarray
    T_eff: int
    t0: float = 2.0

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

    def _key(self) -> tuple[Any, ...]:
        return (self.presample.shape, self.presample.tobytes(), self.T_eff, self.t0)

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

# DESIGN §9.5 rows 1–17 (rows 18+ are these with exact=False: joint_spec(name, exact=False)).
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
    # WP25 (DESIGN §13): rows with SOC/SUR dummy rows in the coefficient prior (ids 201+)
    "structural_constant_dummies": _Def(201, _S, "constant", dummies=True),
    "structural_mode2_constant_dummies": _Def(202, _S, "constant", mode="II", dummies=True),
    "reduced_constant_dummies": _Def(203, _R, "constant", n=(3, 3), dummies=True),
    "structural_sv_dummies": _Def(204, _S, dummies=True),
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
    """Stable id for ``keys.validation_key``: the DESIGN §9.5 row (1–17), 101–107 (blocks) or
    201–204 (SOC/SUR dummy rows, WP25)."""
    return _definition(name).number


def _definition(name: str) -> _Def:
    d = _COMPOSITIONS.get(name) or _BLOCKS.get(name)
    if d is None:
        raise KeyError(
            f"unknown joint spec {name!r}; compositions: {COMPOSITIONS}; blocks: {BLOCK_CONFIGS}"
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


def _const_path(mean: str, vol: VolStatic, mode: str, dummies: bool = False) -> str:
    """The resolve rule of DESIGN §3.7.2 (``const_path="auto"``); Mode I with dummy rows takes
    ``cholesky_const`` (WP25, ``resolve.resolve_const_path``)."""
    if mean != "linear_structural" or not vol.constant_weights:
        return "weighted"
    if mode == "I" and dummies:
        return "cholesky_const"
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
    """
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
        "const_path": _const_path(d.mean, vol, d.mode, d.dummies),
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


__all__ = [
    "ALL",
    "BLOCK_CONFIGS",
    "COMPOSITIONS",
    "FULL_PRESAMPLE",
    "IDENTITY_INTERNALS",
    "OVERRIDE_KEYS",
    "TEST_NU_GRID",
    "TEST_YBAR0",
    "FixedDesign",
    "JointSpec",
    "composition_id",
    "hold_frozen",
    "identity_cccm_core",
    "joint_spec",
    "production_sweep",
    "reference_prior",
    "update_prior",
    "with_fresh_latents",
]
