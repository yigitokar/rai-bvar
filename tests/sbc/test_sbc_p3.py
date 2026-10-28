"""Tier S of Phase 3: the joint models' checks (docs/phase3/DESIGN.md §9.5; WP35).

The SBC runs themselves are weekly, one shard file per row group (P3 §9.11: ``sbc-p3a`` the
structural rows, ``sbc-p3b`` the reduced and CommonSV rows, each split further:
``test_sbc_p3_a1.py`` … ``…_a5.py``, ``…_b1.py`` … ``…_b6.py``; ``sbc_p3.SBC_ROWS``); the joint
models are ``sbc_p3.py``'s. Here:

* ``test_quantity_selection_p3`` (PR): every row has a joint model of the right kind (exogenous:
  the row itself, no predictive; lagged: ``y_{T+1}``, and ``h_{T+1}`` with SV); the quantities are
  the §9.5 ones (the GL and CommonSV test functions, ``W``, the log-likelihoods, ``log p(θ)``)
  without data functions, squares or cross-products; the shard files cover every row once;
* ``test_truth_and_forecast_core_are_the_same_map_p3`` (weekly, beside the SBC it guards): the
  truth ``ỹ_{T+1}`` of the references equals the production ``simulate_paths`` at the same
  ``(θ, log λ_eff,T+1, ε)`` (1e-12), for a CommonSV row of each form and a lagged GL row, and the
  wrapper's flags and shapes are consistent after a step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from sbc_models import PredictiveJoint, program_for, reference_next
from sbc_p3 import (
    BY_ID,
    NOT_MET_ROWS,
    SBC_CONFIG,
    SBC_OVERRIDES,
    SBC_ROWS,
    CommonSVPredictiveJoint,
    reference_next_common,
    sbc_model,
)

from rai_bvar.validation.joint import COMPOSITIONS_P3, LAGGED_P3, JointSpecP3, composition_id
from rai_bvar.validation.reference import effective_log_var, state_log_var

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def test_quantity_selection_p3():
    """Joint-model kinds, quantities and shard coverage (module docstring)."""
    rows = [r for shard in SBC_ROWS.values() for r in shard]
    assert sorted(rows) == list(range(301, 331))  # every row in exactly one shard file
    # Q44, E179: t_max = 400 for exactly these eight rows, nothing else changed; 200 elsewhere
    assert SBC_CONFIG["t_max"] == 200
    assert SBC_OVERRIDES == {r: {"t_max": 400} for r in (304, 306, 308, 311, 315, 325, 326, 327)}
    lagged = {r for r, nm in BY_ID.items() if nm in LAGGED_P3}
    assert lagged == {316, 319, 320, 321, 322, 323, 325, 329, 330}
    kinds = {}
    for r in (301, 316, 320):
        model, _, names = sbc_model(BY_ID[r])
        kinds[r] = type(model).__name__
        assert not any(k.startswith("data.") or "^2" in k or "*" in k for k in names), r
        assert "loglik" in names and "log_prior" in names, r
        pred = [k for k in names if k.startswith("pred.")]
        n = 3
        if r in lagged:
            assert pred[:n] == [f"pred.y_next[{i}]" for i in range(n)], r
            sv = model.spec.static.vol.persistent == "sv"
            assert len(pred) == (2 * n if sv else n), r
        else:
            assert not pred, r
        if r == 320:  # CommonSV: the bounded and unbounded functions of h, W, rss_std
            for want in ("glob.vol.h0", "glob.vol.mean_h", "glob.vol.atan_hT", "glob.vol.rho",
                         "glob.vol.log_sigma", "joint.W", "eq2.joint.loglik",
                         "eq0.joint.rss_std"):  # fmt: skip
                assert want in names, want
        else:  # global-local: the bounded deviation functions and the group functions
            assert any(k.startswith("glob.coef.") for k in names), r
            assert any(".coef.atan_dev[" in k for k in names), r
            assert any(".coef.small_dev[" in k for k in names), r
    assert kinds == {301: "JointSpecP3", 316: "PredictiveJoint", 320: "CommonSVPredictiveJoint"}
    assert len(COMPOSITIONS_P3) == 30


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["conjugate_commonsv_dummies", "reduced_commonsv", "reduced_sv_ng_lagged"]
)
def test_truth_and_forecast_core_are_the_same_map_p3(name):
    """``ỹ_{T+1}`` of the references equals the production ``simulate_paths`` at the same
    ``(θ, log λ_eff,T+1, ε)`` (1e-12); after a step the predictive slot holds a forecast-core
    draw and the quantities are finite."""
    from rai_bvar.engines.mcmc.forecast import simulate_paths
    from rai_bvar.engines.mcmc.runner import extract_params

    model, quantities, names = sbc_model(name, "small")
    spec = model.spec
    assert isinstance(spec, JointSpecP3) and isinstance(model, PredictiveJoint)
    common = spec.static.vol.common
    assert isinstance(model, CommonSVPredictiveJoint) == common
    n, T = spec.static.shapes.n, spec.fixed.T_eff
    program = program_for(spec.static, ())

    @jax.jit
    def run(key):
        ps = model.prior_sample(random.fold_in(key, 0))
        data = model.simulate(random.fold_in(key, 1), ps)
        if common:
            lv = state_log_var(ps.st, spec.static)[:, T]
            y_ref = reference_next_common(ps.st, data, ps.eps_next, spec)
        else:
            lv = effective_log_var(ps.st.eq.vol, spec.static)[:, T]
            y_ref = reference_next(ps.st, data, ps.eps_next, spec)
        params = extract_params(program, ps.st, data, spec.prior)
        paths, _, _ = simulate_paths(
            params, data.hist, data.t_last, ps.eps_next[None], lv[None], program
        )
        truth_q = quantities(ps, data)
        new = model.step(random.fold_in(key, 2), jnp.uint32(0), ps, data)
        return ps, data, paths[0], y_ref, truth_q, new, quantities(new, data)

    for r in range(3):
        ps, _data, y_core, y_ref, q_truth, new, q_new = run(random.key(r, impl="threefry2x32"))
        np.testing.assert_allclose(np.asarray(y_ref), np.asarray(y_core), rtol=1e-12, atol=1e-12)
        assert float(ps.drawn) == 0.0 and float(new.drawn) == 1.0
        assert q_truth.shape == (len(names),) and np.all(np.isfinite(np.asarray(q_new)))
        base = len(model.base_names)
        np.testing.assert_array_equal(np.asarray(q_truth[base : base + n]), np.asarray(y_ref))
        np.testing.assert_array_equal(np.asarray(q_new[base : base + n]), np.asarray(new.y_next))
        if common:  # the common h of the sampler's state is 0 beyond the T data rows
            assert np.all(np.asarray(new.st.glob.vol.h)[T:] == 0.0)


# Q45, E180: the narrow strict xfail of the rows E179 records as not met, checked on the outcomes
# recorded in run 40430948633 (the reasons verbatim) and on the outcomes that must stay red.
_POOLED_308 = (
    "pooled kept-draw ESS below 0.8·L for ['log_prior']: the pilot underestimates the IAT (raise "
    "pilot, burn or iat_factor)"
)
_POOLED_325 = (
    "pooled kept-draw ESS below 0.8·L for ['log_prior', 'glob.vol.mean_abs_h', 'glob.vol.sigma', "
    "'glob.vol.log_sigma']: the pilot underestimates the IAT (raise pilot, burn or iat_factor)"
)
_GAMMA = "γ test (Bonferroni over 58 at α = 0.01): ('glob.vol.sigma',)"
_ESS_RULE = "thinned ESS ≥ 0.8·L in only 98.0% of replications (need 99%; raise t_max or pilot)"
_FAILED = "3 replication(s) failed (non-finite quantities)"


def _stage(reasons=(), failing=(), n_failed=0, ess_ok=1.0):
    from types import SimpleNamespace

    import pandas as pd

    failed = np.zeros(1000, bool)
    failed[:n_failed] = True
    return SimpleNamespace(reasons=tuple(reasons), failing=tuple(failing), failed=failed,
                           ess_ok=ess_ok, table=lambda: pd.DataFrame())  # fmt: skip


@pytest.mark.parametrize(
    "row, last, outcome",
    [
        (308, _stage([_POOLED_308]), "xfail"),  # the recorded outcome of run 40430948633
        (325, _stage([_POOLED_325]), "xfail"),  # the recorded outcome of run 40430948633
        (325, _stage([_GAMMA, _POOLED_325], failing=["glob.vol.sigma"]), "fail"),  # γ breaks
        (308, _stage([_ESS_RULE, _POOLED_308], ess_ok=0.98), "fail"),  # the per-replication rule
        (325, _stage([_FAILED, _POOLED_325], n_failed=3), "fail"),  # failed replications
        (308, _stage([_POOLED_308], ess_ok=0.98), "fail"),  # rule not met although unreported
        (325, _stage(), "unexpected-pass"),  # a pass needs a re-decision
        (304, _stage(), "pass"),  # every other row is unchanged
        (304, _stage([_POOLED_308]), "assert"),
    ],
)  # fmt: skip
def test_not_met_rows_are_a_narrow_strict_xfail(row, last, outcome):
    """P3 Q45, E180: rows 308 and 325 are XFAIL only when they fail on the pooled kept-draw ESS gate
    alone; any other failure and a pass FAIL; every other row must pass as before."""
    from sbc_p3_run import check_sbc

    assert NOT_MET_ROWS == (308, 325)
    name = BY_ID[row]
    assert composition_id(name) == row
    passed = not last.reasons
    out = {"ess_coverage": 0.99}
    if outcome == "xfail":
        with pytest.raises(pytest.xfail.Exception, match="E180"):
            check_sbc(name, passed, out, last)
    elif outcome in ("fail", "unexpected-pass"):
        with pytest.raises(pytest.fail.Exception, match="E180"):
            check_sbc(name, passed, out, last)
    elif outcome == "pass":
        check_sbc(name, passed, out, last)
    else:
        with pytest.raises(AssertionError):
            check_sbc(name, passed, out, last)
