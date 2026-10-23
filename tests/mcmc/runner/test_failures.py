"""Failures are values, replayed deterministically on the host (D-18; DESIGN §5.7, §9.3 item 4)."""

from __future__ import annotations

import dataclasses
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines.mcmc.protocols import MCMCNumericalError
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.state import StepInfo
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

STATIC = toys.toy_static("linear_structural", "constant", n=3, T_eff=20, chunk=5)


@dataclasses.dataclass(frozen=True)
class SilentDrift(toys.DriftMean):
    """The DriftMean whose failure is not flagged: the NaN must be caught by the invariants."""

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        ms, a, cpl, _ = super().sample_eq(key, ms, a_i, ctx, pm, cm, cache_i, cache, shared)
        return ms, a, cpl, StepInfo(jnp.ones((), bool), jnp.ones(()))


def _fit(settings, *, mean=None, replay=True):
    mean = toys.DriftMean(STATIC.mean, STATIC.shapes) if mean is None else mean
    return fit_mcmc(
        STATIC, toys.toy_data(STATIC, 20, 3), toys.toy_prior(STATIC), settings,
        Runtime(chunk=5, replay=replay), seed=0, salt=0,
        meta=FitMeta(variables=("x", "y", "z")), _overrides=toys.toy_overrides(STATIC, mean=mean),
    )  # fmt: skip


def _check(err: MCMCNumericalError) -> None:
    assert (err.block, err.equation, err.iteration, err.chain) == ("EQ_MEAN", 1, 7, 0)
    d = err.detail
    assert d["step"] == "equation" and d["code"] == 1 + 1000 * 0 + 1
    assert d["cholesky_ok"] is False and d["nonpositive_diagonal"] >= 1
    assert d["prior_precision_range"] == (0.25, 2.0)  # the coefficient and a_i prior blocks
    assert d["min_eigenvalue"] < 0 and d["diag_ratio"] == float("inf")
    lo, hi = d["weight_range"]
    assert 0 < lo <= hi and np.isfinite(hi)
    assert d["min_pivot"] <= 0
    assert "sweep 7" in str(err) and "equation 1" in str(err)
    again = pickle.loads(pickle.dumps(err))
    assert (again.block, again.equation, again.iteration, again.chain) == ("EQ_MEAN", 1, 7, 0)


def test_failure_in_burn_in_is_replayed():
    """A non-PD prior injected at sweep 7 (burn pieces of 5 sweeps): block, equation, iteration
    and the replayed conditional's diagnostics."""
    with pytest.raises(MCMCNumericalError) as info:
        _fit(MCMCSettings(chains=2, draws=5, burn=20))
    _check(info.value)


def test_failure_in_sampling_is_replayed():
    with pytest.raises(MCMCNumericalError) as info:
        _fit(MCMCSettings(chains=2, draws=10, burn=3))  # sweep 7 = kept draw 4 of the first chunk
    _check(info.value)


def test_failure_without_replay_copy():
    with pytest.raises(MCMCNumericalError) as info:
        _fit(MCMCSettings(chains=2, draws=5, burn=20), replay=False)
    err = info.value
    assert (err.block, err.equation, err.iteration) == ("EQ_MEAN", 1, 7)
    assert err.detail["replay"].startswith("disabled")


def test_failure_after_the_last_kept_draw_is_ignored():
    """Surplus sweeps of the last chunk never raise: draws end at sweep 6 (burn 3 + 4 draws)."""
    fit = _fit(MCMCSettings(chains=2, draws=4, burn=3))
    assert fit.final_state.state.status.code.tolist() == [0, 0]
    assert np.isfinite(fit.monitor["delta0"]).all()


REDUCED = toys.toy_static("linear_reduced", "constant", n=3, T_eff=20, chunk=5)


def _fit_reduced(settings, *, replay=True):
    ov = toys.toy_overrides(REDUCED)
    ov["global_pre"] = (toys.DriftCCCM(REDUCED.mean, REDUCED.shapes, ov["vol"]),)
    return fit_mcmc(
        REDUCED, toys.toy_data(REDUCED, 20, 3), toys.toy_prior(REDUCED), settings,
        Runtime(chunk=5, replay=replay), seed=0, salt=0, meta=FitMeta(variables=("x", "y", "z")),
        _overrides=ov,
    )  # fmt: skip


@pytest.mark.parametrize("burn", [20, 3])  # sweep 7 in burn-in / in the first sampling chunk
def test_global_step_failure_is_located_by_replay(burn):
    """The CCCM-like pre step fails at sweep 7 in column 1 (its prior precision turns negative).
    Global steps record equation 0; the replay evaluates the step's exposed conditional
    ``conditional_j`` for every column and reports the failing one with its conditioning."""
    with pytest.raises(MCMCNumericalError) as info:
        _fit_reduced(MCMCSettings(chains=2, draws=10, burn=burn))
    err = info.value
    assert (err.block, err.equation, err.iteration, err.chain) == ("G_CCCM", 1, 7, 0)
    d = err.detail
    assert d["code"] == 1 + 1000 * 16 + 0 and d["step"] == "cccm"
    assert d["equation"] == 1 and d["equation_located"] is True
    assert d["cholesky_ok"] is False and d["b_finite"] is True
    assert d["nonpositive_diagonal"] >= 1 and d["min_eigenvalue"] < 0
    ratios = d["diag_ratio_by_equation"]
    assert (
        len(ratios) == 3 and ratios[1] == float("inf") and np.isfinite([ratios[0], ratios[2]]).all()
    )
    lo, hi = d["weight_range"]
    assert 0 < lo <= hi and np.isfinite(hi)
    assert d["prior_precision_range"][0] < 0 < d["prior_precision_range"][1]
    assert "equation 1" in str(err) and "sweep 7" in str(err)
    again = pickle.loads(pickle.dumps(err))
    assert (again.block, again.equation, again.iteration) == ("G_CCCM", 1, 7)


def test_global_step_failure_without_replay_copy():
    with pytest.raises(MCMCNumericalError) as info:
        _fit_reduced(MCMCSettings(chains=2, draws=5, burn=20), replay=False)
    err = info.value
    assert (err.block, err.equation, err.iteration) == ("G_CCCM", 0, 7)
    assert err.detail["equation_located"] is False and err.detail["replay"].startswith("disabled")


def test_unflagged_non_finite_state_is_located():
    """A NaN without a status code: the invariant check fails and a bisection replay finds the
    first offending sweep and equation."""
    mean = SilentDrift(STATIC.mean, STATIC.shapes)
    with pytest.raises(MCMCNumericalError) as info:
        _fit(MCMCSettings(chains=2, draws=5, burn=20), mean=mean)
    err = info.value
    assert (err.block, err.equation, err.iteration) == ("state", 1, 7)
    assert any("non-finite" in v for v in err.detail["violations"])


def test_replay_of_an_equation_failure_after_the_production_cccm_step():
    """``failure_detail`` re-runs the pre-phase steps eagerly from the host (NumPy) copy of the
    state. With the production CCCM step its column scan indexes ``Π`` with the traced column, so
    host arrays must be converted first: before the fix the replay of a reduced-form EQ_VOL
    failure raised ``TracerArrayConversionError`` (found in the full Clark–Ravazzolo run) and the
    diagnostics were lost; now it reports the vol step of the failing equation."""
    from jax import random

    from rai_bvar.engines.mcmc.compose import compose
    from rai_bvar.engines.mcmc.keys import Block
    from rai_bvar.engines.mcmc.runner import failure_detail
    from rai_bvar.validation.joint import joint_spec, production_sweep

    spec = joint_spec("reduced_sv")
    key = random.key(5, impl="threefry2x32")
    st = spec.prior_sample(random.fold_in(key, 0))
    data = spec.simulate(random.fold_in(key, 1), st)
    cache = production_sweep(spec.static)[0](data, spec.prior)
    host = [jax_to_numpy(x) for x in (st, data, spec.prior, cache)]
    detail = failure_detail(compose(spec.static), *host, random.fold_in(key, 2), 7, Block.EQ_VOL, 0)
    assert detail["step"] == "vol" and "residual_range" in detail


def jax_to_numpy(tree):
    import jax

    return jax.tree.map(np.asarray, tree)
