"""Density-ratio and MH identities on Gaussian toys (Grosse & Duvenaud 2014 eq. 1; N06 §5.1)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy import stats as jstats

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64)
from rai_bvar.validation.identities import density_ratio_error, mh_ratio_error, scaled_error

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

TOL = 1e-8  # DESIGN §9.3 item 2

# Two-block Gaussian toy: θ = (b1, b2) with independent priors, y_t ~ N(b1 + 2 b2, s²).
M0 = jnp.array([0.3, -0.2])
S0 = jnp.array([0.8, 0.5])
S = 0.7


def make_data(seed: int, T: int = 12) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(0.5, 1.0, T))


def log_joint(state, y):
    """Independent implementation: prior log-densities plus the likelihood."""
    b = jnp.stack([state["b1"], state["b2"]])
    prior = jnp.sum(jstats.norm.logpdf(b, M0, S0))
    return prior + jnp.sum(jstats.norm.logpdf(y, state["b1"] + 2.0 * state["b2"], S))


def b1_conditional(state, y, var_mult=1.0):
    """Exact N(m, v) of b1 | b2, y (precision form), optionally with a wrong variance."""
    prec = 1.0 / S0[0] ** 2 + y.size / S**2
    lin = M0[0] / S0[0] ** 2 + jnp.sum(y - 2.0 * state["b2"]) / S**2
    return lin / prec, var_mult / prec


def log_q_b1(state, y, var_mult=1.0):
    m, v = b1_conditional(state, y, var_mult)
    return jstats.norm.logpdf(state["b1"], m, jnp.sqrt(v))


def random_pairs(seed: int, n: int = 20):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        b2 = rng.normal()
        yield {"b1": rng.normal(), "b2": b2}, {"b1": rng.normal() * 2, "b2": b2}


def test_exact_conditional_satisfies_the_identity():
    y = make_data(0)
    errs = [
        density_ratio_error(lambda s: log_q_b1(s, y), lambda s: log_joint(s, y), x0, x1)
        for x0, x1 in random_pairs(1)
    ]
    assert max(errs) < 1e-12


def test_wrong_conditional_is_detected_by_one_pair():
    y = make_data(0)
    errs = [
        density_ratio_error(lambda s: log_q_b1(s, y, 1.3), lambda s: log_joint(s, y), x0, x1)
        for x0, x1 in random_pairs(2)
    ]
    assert min(errs) > 1e-4

    def ignore_b2(s):  # CCM-style coupling error: conditional ignores the other block
        return log_q_b1({"b1": s["b1"], "b2": jnp.float64(0.0)}, y)

    errs = [
        density_ratio_error(ignore_b2, lambda s: log_joint(s, y), x0, x1)
        for x0, x1 in random_pairs(3)
    ]
    assert min(errs) > 1e-4


def test_identity_under_jit_and_unnormalised_conditionals():
    y = make_data(4)
    log_q = jax.jit(lambda s: log_q_b1(s, y) + 123.456)  # constants cancel
    lj = jax.jit(lambda s: log_joint(s, y))
    for x0, x1 in random_pairs(5, 5):
        assert density_ratio_error(log_q, lj, x0, x1) < TOL


# --- MH acceptance ratios -------------------------------------------------------------------------

A_IG, B_IG = 4.0, 3.0  # target: σ² ~ IG(a, b) (log-density up to a constant)


def log_target(x):
    return -(A_IG + 1.0) * jnp.log(x) - B_IG / x


def log_prop_lognormal(to, frm, step=0.4):
    """Multiplicative random walk: log x′ ~ N(log x, step²), density in x′ includes 1/x′."""
    return jstats.norm.logpdf(jnp.log(to), jnp.log(frm), step) - jnp.log(to)


def kernel_log_alpha(x, x_prop, jacobian=True):
    """What an implementation computes: target ratio plus the log-scale Jacobian x′/x."""
    out = log_target(x_prop) - log_target(x)
    return out + (jnp.log(x_prop) - jnp.log(x)) if jacobian else out


def test_mh_ratio_with_jacobian_matches_definition():
    rng = np.random.default_rng(6)
    for _ in range(20):
        x, x_prop = rng.gamma(2.0, 1.0, size=2)
        err = mh_ratio_error(kernel_log_alpha, log_target, log_prop_lognormal, x, x_prop)
        assert err < 1e-12


def test_mh_ratio_without_jacobian_is_detected():
    rng = np.random.default_rng(7)
    errs = [
        mh_ratio_error(
            lambda a, b: kernel_log_alpha(a, b, jacobian=False),
            log_target,
            log_prop_lognormal,
            *rng.gamma(2.0, 1.0, size=2),
        )
        for _ in range(20)
    ]
    assert min(errs) > 1e-6


def test_independence_sampler_hastings_term():
    # proposal N(1, 1) for a N(0, 1) target: the ratio needs log q(x) − log q(x′)
    def lt(x):
        return jstats.norm.logpdf(x)

    def lp(to, frm):
        return jstats.norm.logpdf(to, 1.0, 1.0)

    def right(x, xp):
        return lt(xp) - lt(x) + lp(x, xp) - lp(xp, x)

    def wrong(x, xp):
        return lt(xp) - lt(x)

    assert mh_ratio_error(right, lt, lp, 0.3, -1.2) < 1e-14
    assert mh_ratio_error(wrong, lt, lp, 0.3, -1.2) > 0.1


def test_non_finite_handling():
    assert scaled_error(-math.inf, -math.inf) == 0.0
    assert scaled_error(-math.inf, 0.0) == math.inf
    assert scaled_error(math.nan, 0.0) == math.inf
    assert scaled_error(1.0, 1.0 + 1e-9) < 1e-9
    # both states outside the support on both sides: identical infinite terms cancel
    assert density_ratio_error(lambda x: -jnp.inf, lambda x: -jnp.inf, 0.0, 1.0) == 0.0
    # support mismatch: q says impossible, the joint does not
    assert (
        density_ratio_error(
            lambda x: jnp.where(x > 0.5, -jnp.inf, 0.0), lambda x: -0.5 * x**2, 0.0, 1.0
        )
        == math.inf
    )
    with pytest.raises(ValueError, match="scalar"):
        density_ratio_error(lambda x: jnp.zeros(2), lambda x: 0.0, 0.0, 1.0)
