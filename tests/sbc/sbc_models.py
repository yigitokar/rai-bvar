"""Joint models for the composition-level SBC and Tier B tests (DESIGN §9.4, §9.6, §13 WP19).

**Predictive SBC** (:class:`PredictiveJoint`, DESIGN §9.6 "the predictive unknowns ``h_{T+1}``,
``y_{T+1}`` through the production forecast core"). The unknowns are ``(θ, y_{T+1})`` given
``y_{1:T}``. The truth is drawn by the independent references only (WP9):

    θ̃ ~ p(θ) with latents up to T + 1     (``joint_spec(name, T_eff=T + 1).prior_sample``)
    ε̃ ~ N(0, I_n)                          (the held-out structural shock)
    ỹ_{1:T} ~ p(y | θ̃)                     (``joint_spec(name, T_eff=T).simulate``)
    ỹ_{T+1} = A⁻¹(Γx_{T+1} + e) | Π'x_{T+1} + A⁻¹e | A⁻¹e,   e = exp(½ log λ̃_eff,T+1) ⊙ ε̃

with ``x_{T+1}`` from ``engines.mcmc.data.regressors_from_window(hist, t_last + 1)`` (the jnp
design builder the two-implementation rule allows) and ``log λ̃_eff,T+1`` from the reference
``effective_log_var`` at row ``T`` (``h̃_{T+1}``, ``õ_{T+1}``, ``τ̃_{T+1}`` from the prior draw).
The restriction of θ̃ to ``h_{0:T}`` and rows ``< T`` has the ``T``-model prior, so ``(θ̃, ỹ_{1:T})``
is a draw of the joint the sampler targets. Every sweep of the sampler side is the production
sweep (``JointSpec.step``) followed by one draw of ``(h_{T+1}, y_{T+1})`` from the production
forecast core (``forecast_core`` with ``H = 1`` on ``runner.extract_params``), so a kept draw is
a draw from ``p(θ, y_{T+1} | y_{1:T})`` and the rank of ``(θ̃, ỹ_{T+1})`` is uniform (Talts et al.
2018 Theorem 1). The state's ``drawn`` flag (0 for a prior draw, 1 after a step) tells
:meth:`PredictiveJoint.quantities` whether to read the predictive slot (sampler draws) or to
compute the truth from ``ε̃``; the data-only filter never sees ``ỹ_{T+1}``.

The forecast key of sweep ``it`` is ``fold_in(fold_in(chain_key, it), PRED_TAG)`` with
``PRED_TAG = 0xFFFF`` outside every ``keys.Block`` value, so it never collides with a sweep key
``fold_in(fold_in(chain_key, it), block)`` (DESIGN §5.6).

**Quantities** (DESIGN §9.6): every reference test quantity except data functions (constant given
``y``), squares and cross-products (Geweke moment devices; a rank needs no moments): parameters,
``h`` at four dates, mean ``h``, the total and per-equation log-likelihoods, ``log p(θ)``,
``‖Γ − AM′‖²``, the transitory-marginal log-likelihood, ``N_1``, mean ``log τ``, plus
``y_{T+1}`` and (SV) ``h_{T+1}`` per equation.

**Tier B** helpers (:func:`block_joint_sample`, :func:`block_conditional_draws`,
:func:`cccm_column_draws`, :func:`invariance_pvalue`): one block live through the production sweep (``JointSpec.live``,
frozen blocks restored by ``hold_frozen`` or held by identity internals); see
``test_block_tier.py``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax.scipy.linalg import solve_triangular
from scipy import stats

from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc.data import regressors_from_window
from rai_bvar.engines.mcmc.keys import Tier
from rai_bvar.engines.mcmc.state import CCCMGlobal, ChainState, Empty, EngineData
from rai_bvar.validation.geweke import Part, part, rep_keys
from rai_bvar.validation.joint import JointSpec, joint_spec
from rai_bvar.validation.reference import effective_log_var
from rai_bvar.validation.sequential import bonferroni
from rai_bvar.validation.simulate import structural_gamma, unit_lower

Array = jax.Array
_F64 = jnp.float64
PRED_TAG = 0xFFFF  # forecast key tag: outside every keys.Block value (0..63)


def keep_quantity(name: str) -> bool:
    """SBC quantities (module docstring): no data functions, squares or cross-products."""
    return not name.startswith("data.") and "^2" not in name and "*" not in name


@functools.cache
def program_for(static: Any, overrides: tuple[tuple[str, Any], ...]) -> Any:
    """The production ``Program`` of ``static`` with a ``JointSpec``'s kernel overrides (cached)."""
    from rai_bvar.engines.mcmc.compose import compose

    return compose(static, overrides=dict(overrides) or None)


class PredState(NamedTuple):
    """A sampler state plus the predictive slot (module docstring)."""

    st: ChainState
    eps_next: Array  # f64[n]: held-out structural shock (read for the truth only)
    y_next: Array  # f64[n]: y_{T+1} drawn by the forecast core (after a step)
    h_next: Array  # f64[n]: persistent log variance at T + 1 (after a step)
    drawn: Array  # f64[]: 0 for a prior draw (the truth), 1 after a step


def reference_next(st: ChainState, data: EngineData, eps: Array, spec: JointSpec) -> Array:
    """``ỹ_{T+1}`` from the references (module docstring) for a state whose latents reach row ``T``."""
    static = spec.static
    x = regressors_from_window(data.hist, data.t_last + 1.0, static.shapes.n_det)
    A = unit_lower(st.eq.struct.a)
    lv = effective_log_var(st.eq.vol, static)  # (n, T_pad); row T is the period T + 1
    lv_next = lax.dynamic_index_in_dim(lv, data.T_eff, axis=1, keepdims=False)
    e = jnp.exp(0.5 * lv_next) * eps
    kind = static.mean.kind
    if kind == "linear_structural":
        g = structural_gamma(st, spec.prior) @ x
        return solve_triangular(A, g + e, lower=True, unit_diagonal=True)
    shock = solve_triangular(A, e, lower=True, unit_diagonal=True)
    if kind == "linear_reduced":
        return st.glob.mean.Pi.T @ x + shock
    return shock


def reference_h_next(st: ChainState, data: EngineData) -> Array:
    """``h̃_{T+1}`` (state index ``T + 1``) of a prior draw with latents up to ``T + 1``."""
    return lax.dynamic_index_in_dim(st.eq.vol.core.h, data.T_eff + 1, axis=1, keepdims=False)


@dataclass(frozen=True, eq=False)
class PredictiveJoint:
    """``JointModel`` over ``(θ, y_{T+1})`` given ``y_{1:T}`` (module docstring)."""

    spec: JointSpec
    truth: JointSpec
    base_names: tuple[str, ...] = field(init=False)
    _index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.truth.static != self.spec.static
            or self.truth.fixed.T_eff != self.spec.fixed.T_eff + 1
        ):
            raise ValueError("truth must be the same composition with T_eff + 1 (same T_pad)")
        base = tuple(k for k in self.spec.all_names if keep_quantity(k))
        object.__setattr__(self, "base_names", base)
        object.__setattr__(self, "_index", np.array([self.spec.all_names.index(k) for k in base]))

    @classmethod
    def of(cls, name: str, size: str = "full", **overrides: Any) -> PredictiveJoint:
        """The predictive joint of composition ``name`` (``joint_spec`` sizes and overrides)."""
        spec = joint_spec(name, size, **overrides)
        T = spec.fixed.T_eff
        truth = joint_spec(name, size, **{**overrides, "T_eff": T + 1})
        return cls(spec, truth)

    @property
    def name(self) -> str:
        return f"{self.spec.name}.predictive"

    @property
    def sv(self) -> bool:
        return self.spec.static.vol.persistent == "sv"

    @property
    def names(self) -> tuple[str, ...]:
        n = self.spec.static.shapes.n
        pred = [f"pred.y_next[{i}]" for i in range(n)]
        if self.sv:
            pred += [f"pred.h_next[{i}]" for i in range(n)]
        return (*self.base_names, *pred)

    # ----------------------------------------------------------------- JointModel
    def prior_sample(self, key: Array) -> PredState:
        st = self.truth.prior_sample(random.fold_in(key, 0))
        n = self.spec.static.shapes.n
        eps = random.normal(random.fold_in(key, 1), (n,), dtype=_F64)
        z = jnp.zeros((n,), _F64)
        return PredState(st, eps, z, z, jnp.zeros((), _F64))

    def simulate(self, key: Array, ps: PredState) -> EngineData:
        return self.spec.simulate(key, ps.st)

    def forecast(
        self, key: Array, it: Array, st: ChainState, data: EngineData
    ) -> tuple[Array, Array]:
        """One draw of ``(y_{T+1}, log λ^pers_{T+1})`` from the production forecast core."""
        from rai_bvar.engines.mcmc.forecast import forecast_core
        from rai_bvar.engines.mcmc.runner import extract_params, forecast_shocks

        program = program_for(self.spec.static, self.spec.overrides)
        prior = self.spec.prior if self.spec.sampler_prior is None else self.spec.sampler_prior
        params = extract_params(program, st, data, prior)
        fkey = random.fold_in(random.fold_in(key, it), PRED_TAG)
        shocks = forecast_shocks(fkey, jnp.uint32(0), 1, self.spec.static.shapes.n)
        out = forecast_core(params, data.hist, data.t_last, shocks, program)
        return out.paths[0], out.log_lam_pers[0]

    def step(self, key: Array, it: Array, ps: PredState, data: EngineData) -> PredState:
        st = self.spec.step(key, it, ps.st, data)
        y, h = self.forecast(key, it, st, data)
        return PredState(st, ps.eps_next, y, h, jnp.ones((), _F64))

    def quantities(self, ps: PredState, data: EngineData) -> Array:
        base = self.spec.quantities(ps.st, data)[self._index]
        drawn = ps.drawn > 0
        y = jnp.where(drawn, ps.y_next, reference_next(ps.st, data, ps.eps_next, self.spec))
        parts = [base, y]
        if self.sv:
            parts.append(jnp.where(drawn, ps.h_next, reference_h_next(ps.st, data)))
        return jnp.concatenate(parts)

    test_functions = quantities


# --------------------------------------------------------------------------- Tier B helpers


def block_joint_sample(spec: JointSpec) -> Callable[[Array], tuple[ChainState, EngineData]]:
    """``key -> (θ̃, ỹ)`` from the joint (references only)."""

    def sample(key: Array) -> tuple[ChainState, EngineData]:
        st = spec.prior_sample(random.fold_in(key, 0))
        return st, spec.simulate(random.fold_in(key, 1), st)

    return sample


def block_conditional_draws(spec: JointSpec) -> Callable[..., ChainState]:
    """``(key, θ̃, ỹ) -> θ`` with the live blocks redrawn by one production sweep and every other
    block restored (``JointSpec.step``): an iid draw from the live blocks' exact conditional when
    they are exact-draw blocks that run before every frozen block (``joint.py`` docstring)."""

    def draw(key: Array, st: ChainState, data: EngineData) -> ChainState:
        return spec.step(key, jnp.uint32(0), st, data)

    return draw


def cccm_column_draws(spec: JointSpec, j: int) -> Callable[..., ChainState]:
    """``(key, θ̃, ỹ) -> θ`` with column ``j`` of ``Π`` redrawn from ``π_j | π_{−j}, A, λ_eff, y``
    and every other column restored. Column 0 comes from the program's ``cccm_core`` (one call with
    fresh noise; the scan draws column 0 first, given the current ``Π``); a later column would be
    drawn given the scan's new earlier columns, so columns ``j ≥ 1`` come from the exposed exact
    conditional ``cccm_conditional_j`` and ``precision_draw``. The weights ``m_t exp(−log λ_eff)``
    come from the reference ``effective_log_var``. SOC/SUR dummy rows (WP25) are handed over as the
    production step does (``dummy=`` from the production ``prepare``'s ``FitCache.dummy``)."""
    from rai_bvar.components.mean.linear_reduced import cccm_conditional_j
    from rai_bvar.engines.mcmc.linalg import precision_draw
    from rai_bvar.validation.joint import production_sweep

    static = spec.static
    core = program_for(static, spec.overrides).global_pre[0].impl("cccm_core")
    n, k = static.shapes.n, static.shapes.k
    coef = spec.prior.eq.coef
    prepare = production_sweep(static, spec.overrides)[0]

    def draw(key: Array, st: ChainState, data: EngineData) -> ChainState:
        lv = effective_log_var(st.eq.vol, static)
        live = data.tmask[None, :] > 0
        inv_lam = jnp.where(live, jnp.exp(-jnp.where(live, lv, 0.0)), 0.0)
        Pi, a = st.glob.mean.Pi, st.eq.struct.a
        kw: dict[str, Any] = {}
        if static.mean.n_dummy:
            d = prepare(data, spec.prior).dummy
            kw["dummy"] = (d.G, d.FtY, d.w)
        if j == 0:
            Z = random.normal(key, (n, k), dtype=_F64)
            new, _ = core(Pi, a, inv_lam, data.X, data.Y, Empty(), coef.mean0, coef.prec0, Z,
                          gram="batched", precompute=True, **kw)  # fmt: skip
            col = new[:, 0]
        else:
            cond = cccm_conditional_j(Pi, a, inv_lam, data.X, data.Y, coef.mean0, coef.prec0, j,
                                      **kw)  # fmt: skip
            col, _ = precision_draw(cond.K, cond.b, random.normal(key, (k,), dtype=_F64))
        P = Pi.at[:, j].set(col)
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=P)))

    return draw


def _invariance_fn(spec: JointSpec, L: int, stat: Callable) -> Callable:
    sample = block_joint_sample(spec)

    def one(key: Array) -> tuple[Array, Array]:
        st, y = sample(part(key, Part.PRIOR))
        step_key = part(key, Part.STEP)

        def body(j: Array, s: ChainState) -> ChainState:
            return spec.step(step_key, j.astype(jnp.uint32), s, y)

        out = lax.fori_loop(0, L, body, st)
        ref, y_ref = sample(part(key, Part.MC_PRIOR))  # independent reference draw
        return stat(out, y), stat(ref, y_ref)

    return jax.jit(jax.vmap(one))


@functools.cache
def _cached_invariance_fn(spec: JointSpec, L: int, stat: Callable) -> Callable:
    return _invariance_fn(spec, L, stat)


def invariance_pvalue(
    spec: JointSpec, stat: Callable, N: int, L: int, seed: int, *, composition: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One-block invariance (N06 §5.3; DESIGN §9.4): ``N`` joint draws ``(θ̃, ỹ)``, ``L`` kernel
    applications of the live blocks with ``ỹ`` fixed, against ``N`` independent joint draws. Under
    invariance both samples have the joint law, so each statistic's two-sample KS p-value is valid.
    Returns ``(Bonferroni p, per-statistic p, after, reference)``; a non-finite statistic gives
    ``p = 0``."""
    fn = _cached_invariance_fn(spec, int(L), stat)
    with _jax.x64():
        keys = rep_keys(seed, Tier.B, composition, np.arange(N))
        after, ref = (np.asarray(a) for a in fn(keys))
    ps = np.array([
        stats.ks_2samp(after[:, g], ref[:, g]).pvalue
        if np.all(np.isfinite(after[:, g])) and np.all(np.isfinite(ref[:, g]))
        else 0.0
        for g in range(after.shape[1])
    ])  # fmt: skip
    return bonferroni(ps), ps, after, ref


__all__ = [
    "PRED_TAG",
    "PredState",
    "PredictiveJoint",
    "block_conditional_draws",
    "block_joint_sample",
    "cccm_column_draws",
    "invariance_pvalue",
    "keep_quantity",
    "program_for",
    "reference_h_next",
    "reference_next",
]
