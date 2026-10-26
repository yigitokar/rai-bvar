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
2026-10-23); ``RAI_BVAR_SBC_N`` overrides ``N`` for local smoke runs.
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
#: P3 §9.11: ``sbc-p3a`` the structural rows, ``sbc-p3b`` the reduced and CommonSV rows (the
#: Geweke split, by file), each split once more ("split further if not": ≤ 200 min per shard on
#: a 2-vCPU EPYC 7763; ``tools/weekly_shards.py`` gives the estimate): one file per shard
SBC_ROWS: dict[str, tuple[int, ...]] = {
    "a1": (301, 302, 303, 304, 305, 306, 307, 308),  # structural: HS, HS × SV, NG, R2D2, DL
    "a2": (309, 310, 317, 318, 319, 326, 330),  # structural: SSVS, the HS variants, lagged rows
    "b1": (311, 312, 313, 314, 315, 316, 327),  # reduced form
    "b2": (320, 321, 322, 323, 324, 325, 328, 329),  # CommonSV
}
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
    "SBC_N",
    "SBC_ROWS",
    "SEED",
    "CommonSVPredictiveJoint",
    "iat_summary",
    "mask_common_h",
    "nanmin",
    "plain_quantities",
    "reference_next_common",
    "sbc_model",
]
