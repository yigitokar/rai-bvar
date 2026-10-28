"""Joint models of the Phase 3 SBC suites (docs/phase3/DESIGN.md §9.5; WP35).

§9.5: every row of §9.4 at the P2 §9.6 configuration; quantities the §9.4 test functions, the
total and per-equation log-likelihood, ``log p(θ)``, ``W`` (N08 §A11) and, **on lagged rows**,
the predictive ``y_{T+1}`` through the production forecast core (exogenous rows have no
``x_{T+1}``, E49). Three kinds of joint model, by row (:func:`sbc_model`):

* **exogenous rows** (D-56): the row's ``JointSpecP3`` itself, its quantities the SBC selection of
  Phase 2 (``sbc_models.keep_quantity``: every test function except data functions, squares and
  cross-products, which are moment devices or constant given ``y``);
* **lagged rows without CommonSV** (316, 319, 330): Phase 2's ``sbc_models.PredictiveJoint``
  unchanged (``y_{T+1}`` and, with SV, ``h_{T+1}``);
* **lagged CommonSV rows** (320–323, 325, 329): :class:`CommonSVPredictiveJoint`. Phase 2's
  wrapper cannot run them (an open choice of WP35, leaving ``sbc_models.py`` untouched): its truth
  ``ỹ_{T+1}`` reads the effective log variance from ``eq.vol`` alone, while under CommonSV it is
  ``log λ_i + h_T`` with the common ``h`` in ``glob.vol``. The truth here uses WP30's
  ``state_log_var`` of the ``T + 1`` prior draw at row ``T`` (``h̃_{T}``, the period ``T + 1``), the
  sampler side the production forecast core (``h_{T+1} = ρh_T + σ_hη`` from ``CommonSVStep.future``,
  then ``y``), so ``(θ̃, ỹ_{T+1})`` and the kept draws have the same law under the ``T``-row model
  (Talts et al. 2018 Theorem 1). Before each production step the common ``h`` is masked to the
  ``T`` data rows: the chain's start is a prior draw of the ``T + 1`` truth model, whose row-``T``
  state is no state of the ``T``-row model (padded states are exactly 0, P3 §2.2).

Sizes ``"full"`` (``n = 3, p = 2, T_eff = 50``) as Phase 2's SBC; the configuration is
:data:`SBC_CONFIG` (P2 §9.6 with Phase 2's live cap ``t_max = 200``, P2 §9.6 erratum of
2026-10-23; ``t_max = 400`` for rows 304, 306, 308, 311, 315, 325, 326 and 327 by Q44, E179:
:data:`SBC_OVERRIDES`); ``RAI_BVAR_SBC_N`` overrides ``N`` for local smoke runs.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular
from sbc_models import PredictiveJoint, PredState, keep_quantity

from rai_bvar.engines.mcmc.data import regressors_from_window
from rai_bvar.engines.mcmc.state import ChainState, EngineData
from rai_bvar.validation.joint import (
    COMPOSITIONS_P3,
    LAGGED_P3,
    JointSpec,
    composition_id,
    joint_spec,
)
from rai_bvar.validation.reference import state_log_var
from rai_bvar.validation.simulate import structural_gamma, unit_lower

Array = jax.Array
_F64 = jnp.float64

SEED = 20261026  # the Phase 2 SBC seed (tests/sbc); the keys fold the CRC-32 of the model's name
SBC_N = int(os.environ.get("RAI_BVAR_SBC_N", "1000"))
SBC_CONFIG: dict[str, Any] = {"L": 99, "burn": 300, "pilot": 1000, "t_max": 200, "alpha": 1e-2}
#: Per-row deviations from :data:`SBC_CONFIG`, keyed by row, as Phase 2's ``SBC_OVERRIDES`` gives
#: row 18 its cap (``tests/sbc/test_sbc_compositions.py``). P3 Q44, E179 (owner, 2026-10-26: "Bu 8
#: satırda t_max = 400"): in run 40414682400 these eight rows failed only the per-replication ESS
#: rule at ``t_max = 200`` (the γ test clean on all 30 rows), so they run at ``t_max = 400``, fixed
#: before the rerun; ``N``, ``L``, burn, pilot, the seed, the γ test and both ESS gates are
#: unchanged, every other row keeps 200, and the rerun's result is binding (E179).
T_MAX_400_ROWS = (304, 306, 308, 311, 315, 325, 326, 327)
SBC_OVERRIDES: dict[int, dict[str, int]] = {r: {"t_max": 400} for r in T_MAX_400_ROWS}
#: P3 §9.11: ``sbc-p3a`` the structural rows, ``sbc-p3b`` the reduced and CommonSV rows (the
#: Geweke split, by file), split further ("split further if not"): one file per group,
#: ``test_sbc_p3_<group>.py``. First balanced on the pre-dispatch estimates (``SBC_ESTIMATE_MIN``,
#: below); re-balanced after Q44 (E179) on the times of run 40414682400 on 4-vCPU runners, with the
#: eight ``t_max = 400`` rows at twice their measured time (``tools/weekly_shards.py``): the three
#: heaviest of them (306, 308, 325) and 315 (whose γ stage-1 value 0.072 could trigger a Gandy–Scott
#: stage 2 at four times the replications) run alone.
SBC_ROWS: dict[str, tuple[int, ...]] = {
    "a1": (317, 303),
    "a2": (308,),
    "a3": (306,),
    "a4": (304, 326, 302),
    "a5": (301, 305, 307, 309, 310, 318, 319, 330),
    "b1": (315,),
    "b2": (313, 316),
    "b3": (311, 327, 329),
    "b4": (325,),
    "b5": (324, 328, 314),
    "b6": (320, 321, 322, 323, 312),
}
#: The pre-dispatch estimate of each row's SBC time in minutes on an Apple M5 Max under a load
#: average of 140–170 (about half as fast as the 4-vCPU EPYC 7763 runner; see
#: ``tools/weekly_shards.py``): a pilot of 200 replications (burn 300, pilot 1000) gives the pilot IATs, hence the thinning
#: ``⌈2·IAT⌉ ≤ 200`` and the ten sorted production blocks of 100 replications, at the measured
#: time of a sweep of 100 replications, plus the compilation; Gandy–Scott reruns are not included.
SBC_ESTIMATE_MIN: dict[int, float] = {
    301: 14.7, 302: 27.7, 303: 3.9, 304: 25.3, 305: 8.6, 306: 48.5, 307: 13.2, 308: 56.9,
    309: 8.8, 310: 9.6, 311: 29.4, 312: 21.4, 313: 31.9, 314: 37.1, 315: 40.9, 316: 32.5,
    317: 76.1, 318: 23.3, 319: 26.2, 320: 35.6, 321: 33.3, 322: 35.2, 323: 31.5, 324: 67.3,
    325: 74.4, 326: 27.5, 327: 13.8, 328: 36.2, 329: 25.2, 330: 13.2,
}  # fmt: skip
BY_ID: dict[int, str] = {composition_id(n): n for n in COMPOSITIONS_P3}


def reference_next_common(st: ChainState, data: EngineData, eps: Array, spec: JointSpec) -> Array:
    """``ỹ_{T+1}`` of a CommonSV state whose latents reach row ``T`` (module docstring): the
    references' reduced form at ``x_{T+1}`` plus ``A⁻¹ e``, ``e = exp(½ (log λ_i + h_T)) ⊙ ε``."""
    static = spec.static
    x = regressors_from_window(data.hist, data.t_last + 1.0, static.shapes.n_det)
    A = unit_lower(st.eq.struct.a)
    lv = state_log_var(st, static)  # (n, T_pad): log λ_i + h_t; row T is the period T + 1
    lv_next = lax.dynamic_index_in_dim(lv, data.T_eff, axis=1, keepdims=False)
    e = jnp.exp(0.5 * lv_next) * eps
    if static.mean.kind == "linear_structural":
        g = structural_gamma(st, spec.prior) @ x
        return solve_triangular(A, g + e, lower=True, unit_diagonal=True)
    shock = solve_triangular(A, e, lower=True, unit_diagonal=True)
    return st.glob.mean.Pi.T @ x + shock


def mask_common_h(st: ChainState, data: EngineData) -> ChainState:
    """``st`` with the common ``h`` exactly 0 beyond the ``T`` data rows."""
    gv = st.glob.vol
    h = jnp.where(data.tmask > 0, gv.h, 0.0)
    return dataclasses.replace(
        st, glob=dataclasses.replace(st.glob, vol=dataclasses.replace(gv, h=h))
    )


@dataclass(frozen=True, eq=False)
class CommonSVPredictiveJoint(PredictiveJoint):
    """``JointModel`` over ``(θ, y_{T+1})`` given ``y_{1:T}`` for a lagged CommonSV row (module
    docstring); everything else is Phase 2's :class:`sbc_models.PredictiveJoint`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.spec.static.vol.common:
            raise ValueError("CommonSVPredictiveJoint needs a CommonSV composition")

    def step(self, key: Array, it: Array, ps: PredState, data: EngineData) -> PredState:
        st = self.spec.step(key, it, mask_common_h(ps.st, data), data)
        y, h = self.forecast(key, it, st, data)
        return PredState(st, ps.eps_next, y, h, jnp.ones((), _F64))

    def quantities(self, ps: PredState, data: EngineData) -> Array:
        base = self.spec.quantities(ps.st, data)[self._index]
        y = jnp.where(
            ps.drawn > 0, ps.y_next, reference_next_common(ps.st, data, ps.eps_next, self.spec)
        )
        return jnp.concatenate([base, y])

    test_functions = quantities


def plain_quantities(spec: JointSpec) -> tuple[Callable[..., Array], tuple[str, ...]]:
    """The Phase 2 SBC selection of ``spec``'s test functions (module docstring)."""
    names = tuple(k for k in spec.all_names if keep_quantity(k))
    idx = jnp.asarray([spec.all_names.index(k) for k in names])
    return (lambda st, data: spec.quantities(st, data)[idx]), names


def sbc_model(name: str, size: str = "full") -> tuple[Any, Callable[..., Array], tuple[str, ...]]:
    """``(model, quantities, names)`` of row ``name`` for ``sbc_sequential`` (module docstring)."""
    if name not in LAGGED_P3:
        spec = joint_spec(name, size)
        fn, names = plain_quantities(spec)
        return spec, fn, names
    cls = CommonSVPredictiveJoint if joint_spec(name, size).static.vol.common else PredictiveJoint
    model = cls.of(name, size)
    return model, model.quantities, model.names


def nanmin(x: Any) -> float | None:
    x = None if x is None else np.asarray(x, dtype=np.float64)
    return None if x is None or not np.any(np.isfinite(x)) else float(np.nanmin(x))


def iat_summary(res: Any, min_ess: float, L: int) -> dict[str, float | int | None]:
    """The pilot-IAT tail behind the ESS rule (recorded, not gating), as ``tests/sbc`` records it."""
    if res.iat is None or not np.size(res.iat):
        return {"iat_median": None, "iat_q99": None, "iat_max": None, "ess_fail": 0}
    tau = np.asarray(res.iat, dtype=np.float64)
    q50, q99 = np.quantile(tau, [0.5, 0.99])
    fail = int(np.sum(np.asarray(res.ess_thin) < min_ess * L))
    return {"iat_median": float(q50), "iat_q99": float(q99), "iat_max": float(tau.max()),
            "ess_fail": fail}  # fmt: skip


__all__ = [
    "BY_ID",
    "SBC_CONFIG",
    "SBC_ESTIMATE_MIN",
    "SBC_N",
    "SBC_OVERRIDES",
    "SBC_ROWS",
    "SEED",
    "T_MAX_400_ROWS",
    "CommonSVPredictiveJoint",
    "iat_summary",
    "mask_common_h",
    "nanmin",
    "plain_quantities",
    "reference_next_common",
    "sbc_model",
]
