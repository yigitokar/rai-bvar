"""Warm starts: final state remapped onto a new window (DESIGN §5.5)."""

from __future__ import annotations

import dataclasses
import pickle

import jax
import numpy as np
import pytest
from jax import random

from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.config import Shapes
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.state import check_state_invariants, state_template
from rai_bvar.engines.mcmc.warmstart import FinalState, incompatibility, remap, warm_state
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

STATIC = toys.toy_static("linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5)
Y = toys.simulate_y(2, 80, 11)  # one long history; windows are cut from it
P_ALL = toys.periods(80)


def _window(static, start, T_eff):
    """Rows start .. start+T_eff−1 of the history (lags taken from just before)."""
    lags = static.shapes.lags
    y = Y[start - lags : start + T_eff]
    return toys.toy_data(static, T_eff, y=y), P_ALL[start : start + T_eff]


def _fit(static, start, T_eff, settings, init=None):
    data, periods = _window(static, start, T_eff)
    return fit_mcmc(
        static, data, toys.toy_prior(static), settings, Runtime(chunk=static.chunk), seed=1, salt=start,
        meta=FitMeta(variables=("a", "b"), periods=periods), init=init,
        _overrides=toys.toy_overrides(static),
    )  # fmt: skip


@pytest.fixture(scope="module")
def prev():
    fit = _fit(STATIC, 5, 20, MCMCSettings(chains=2, draws=4, burn=6))
    return fit.final_state


def _flat(st):
    return {
        jax.tree_util.keystr(p, simple=True, separator="."): np.asarray(x)
        for p, x in jax.tree_util.tree_flatten_with_path(st)[0]
    }


def _check_common(new, prev_state, static, T_new):
    assert check_state_invariants(new, static, T_new) == []
    f, g = _flat(new), _flat(prev_state)
    for name in ("eq.mean.delta", "eq.struct.a", "eq.vol.core.mu", "eq.vol.core.phi", "eq.vol.core.sigma",
                 "eq.vol.trans.log_p", "eq.vol.trans.log1m_p"):  # fmt: skip
        np.testing.assert_array_equal(f[name], g[name])
    for name, x in f.items():
        if name.split(".")[-1].startswith(("acc_", "rb_")):
            assert not x.any(), name
    assert (f["status.code"] == 0).all() and np.isinf(f["status.min_pivot"]).all()
    o = f["eq.vol.trans.o_idx"]
    assert not o[..., T_new:].any()


def _simulated(new_h, prev_state, key, t0):
    """h for rows t ≥ t0 follows h' = μ + φ(h − μ) + ση with the remap's noise."""
    mu, phi, sigma = (np.asarray(x) for x in (prev_state.eq.vol.core.mu, prev_state.eq.vol.core.phi,
                                               prev_state.eq.vol.core.sigma))  # fmt: skip
    C, n = mu.shape
    for c in range(C):
        for i in range(n):
            eta = np.asarray(
                random.normal(keys.eq_key(keys.eq_key(key, c), i), (new_h.shape[-1] - 1,))
            )
            for t in range(t0, eta.size):
                if t + 1 >= new_h.shape[-1]:
                    break
                expect = mu[c, i] + phi[c, i] * (new_h[c, i, t] - mu[c, i]) + sigma[c, i] * eta[t]
                np.testing.assert_allclose(new_h[c, i, t + 1], expect, rtol=1e-13)


def test_expanding_window(prev):
    data, periods = _window(STATIC, 5, 23)
    key = random.key(3)
    new = remap(prev, data, periods, STATIC, key)
    _check_common(new, prev.state, STATIC, 23)
    h_new, h_old = np.asarray(new.eq.vol.core.h), np.asarray(prev.state.eq.vol.core.h)
    np.testing.assert_array_equal(h_new[..., :21], h_old[..., :21])  # h_0 and the 20 shared rows
    np.testing.assert_array_equal(
        np.asarray(new.eq.vol.trans.o_idx)[..., :20],
        np.asarray(prev.state.eq.vol.trans.o_idx)[..., :20],
    )
    _simulated(h_new[..., :24], prev.state, key, 20)  # rows 20..22 forward-simulated
    assert np.isfinite(h_new[..., 21:24]).all() and not h_new[..., 24:].any()


def test_rolling_window(prev):
    data, periods = _window(STATIC, 7, 20)  # drop 2 periods in front, add 2 at the back
    key = random.key(4)
    new = remap(prev, data, periods, STATIC, key)
    _check_common(new, prev.state, STATIC, 20)
    h_new, h_old = np.asarray(new.eq.vol.core.h), np.asarray(prev.state.eq.vol.core.h)
    np.testing.assert_array_equal(
        h_new[..., 0], h_old[..., 2]
    )  # state just before the new first period
    np.testing.assert_array_equal(h_new[..., 1:19], h_old[..., 3:21])
    np.testing.assert_array_equal(
        np.asarray(new.eq.vol.trans.o_idx)[..., :18],
        np.asarray(prev.state.eq.vol.trans.o_idx)[..., 2:20],
    )
    _simulated(h_new[..., :21], prev.state, key, 18)


def test_bucket_change(prev):
    static64 = dataclasses.replace(STATIC, shapes=Shapes.of(2, 1, 1, 64))
    data, periods = _window(static64, 5, 40)
    new = remap(prev, data, periods, static64, random.key(5))
    assert np.shape(new.eq.vol.core.h) == (2, 2, 65)
    _check_common(new, prev.state, static64, 40)
    np.testing.assert_array_equal(
        np.asarray(new.eq.vol.core.h)[..., :21], np.asarray(prev.state.eq.vol.core.h)[..., :21]
    )


def test_random_walk_h_and_no_overlap():
    static = dataclasses.replace(
        STATIC, vol=dataclasses.replace(STATIC.vol, rw=True, transitory="none")
    )
    st = jax.tree.map(
        lambda x: np.broadcast_to(x, (2, *np.shape(x))).copy(), state_template(static)
    )
    st.eq.vol.core.h[:] = np.arange(33.0)[None, None, :]
    st.eq.vol.core.sigma[:] = 0.5
    prev_rw = FinalState(st, P_ALL[5:25], static)
    data, periods = _window(static, 40, 10)  # no shared period: everything simulated from h_0
    key = random.key(6)
    new = remap(prev_rw, data, periods, static, key)
    h = np.asarray(new.eq.vol.core.h)
    assert (h[..., 0] == 0.0).all()  # RW: the old h_0
    for c in range(2):
        for i in range(2):
            eta = np.asarray(random.normal(keys.eq_key(keys.eq_key(key, c), i), (10,)))
            np.testing.assert_allclose(h[c, i, 1:11], np.cumsum(0.5 * eta), rtol=1e-12)


def test_incompatible_composition(prev):
    other = toys.toy_static("linear_structural", "sv", transitory="outlier", n=3, T_eff=20, chunk=5)
    assert "n differs" in incompatibility(prev, other)
    assert incompatibility(prev, dataclasses.replace(STATIC, chunk=7)) is None
    with pytest.raises(ValueError, match="cannot warm-start"):
        remap(prev, toys.toy_data(other, 20, 0), P_ALL[5:25], other, random.key(0))
    with pytest.warns(UserWarning, match="n differs"):
        assert (
            warm_state(prev, toys.toy_data(other, 20, 0), P_ALL[5:25], other, random.key(0)) is None
        )
    assert warm_state(None, toys.toy_data(other, 20, 0), P_ALL[5:25], other, random.key(0)) is None


def test_warm_fit_uses_warm_burn_and_pickles(prev):
    settings = MCMCSettings(chains=2, draws=4, burn=100, warm_burn=3)
    data, periods = _window(STATIC, 5, 23)
    init = remap(prev, data, periods, STATIC, random.key(1))
    fit = _fit(STATIC, 5, 23, settings, init=init)
    assert fit.posterior.attrs["burn"] == 3 and fit.posterior.attrs["warm_start"] == 1
    assert fit.final_state.seed_meta["warm_start"] is True
    again = pickle.loads(pickle.dumps(fit.final_state))
    np.testing.assert_array_equal(again.state.eq.vol.core.h, fit.final_state.state.eq.vol.core.h)
    assert again.static == STATIC and list(again.periods) == list(periods)
