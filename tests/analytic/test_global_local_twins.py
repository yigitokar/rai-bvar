"""Analytic twins of the global-local priors, A12 and A13 (P3 DESIGN §9.7; WP31).

**A12.** The deviation path (P3 §3.4: ``linalg.precision_draw_log`` with the Jacobi scaling in
logs) with frozen finite scales draws the same function of the noise as the Phase 2 path (D-14's
``precision_draw`` on ``K = diag(exp(log P)) + G``): the structural fused ``(δ_i, a_i)`` draw of
``LinearStructural.sample_eq`` (``GaussianLogDiag`` vs the equivalent ``GaussianDiag`` moments)
and the CCCM scan (``cccm_core_log`` vs ``cccm_core`` on the same ``Z``), within P2 §9.3 item 1's
conditioning-aware rule in posterior s.d.: ``2e-12 + 5εκ(1 + t)``, ``κ = cond(sKs)`` of the
active block, ``t`` the largest posterior t-ratio (the CCCM scan: the largest over its ``n``
conditionals, since a column's conditional reads the columns drawn before it).

**A13.** SSVS with ``τ₀ = τ₁`` (engine-level arrays) is the fixed Normal prior ``N(m_j, τ_j²)``
(the indicators then never touch the coefficients): its MCMC posterior against the
``FixedDiagonal`` composition with ``prec0 = τ_j⁻²`` on the lag slots, same data, other arrays
equal. Quantiles 5/25/50/75/95% of every ``δ``, ``a`` and ``log λ``, the Phase 2 parity rule:
``|z| ≤ max(4, z_{1−0.01/(2m)})`` with ``z`` in the CDF form of the two-sample quantile comparison
(``F̂_SSVS(x̂_q) − q`` over the combined indicator MCSE of both sides, ``x̂_q`` the FixedDiagonal
sample quantile; ``diagnostics.mcse_mean_many``).
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from rai_bvar import diagnostics
from rai_bvar.components.mean.linear_reduced import (
    cccm_conditional_j,
    cccm_core,
    cccm_core_log,
    cccm_noise,
)
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.config import CoefPriorStatic
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.runner import _prepare_fn
from rai_bvar.engines.mcmc.state import (
    CoefPriorArrays,
    Empty,
    GaussianDiag,
    PriorArrays,
)
from rai_bvar.engines.mcmc.sweep import sweep_values
from rai_bvar.settings import MCMCSettings, Runtime
from rai_bvar.validation.joint import joint_spec

EPS = float(np.finfo(np.float64).eps)
KEY = random.key(1212)
Q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def envelope(K: np.ndarray, b: np.ndarray, act: np.ndarray):
    """Posterior s.d., ``κ = cond(sKs)`` and ``t = max |E θ|/sd`` on the active block."""
    Ka = K[np.ix_(act, act)]
    C = np.linalg.inv(Ka)
    sd = np.sqrt(np.diag(C))
    s = 1.0 / np.sqrt(np.diag(Ka))
    kappa = float(np.linalg.cond(s[:, None] * Ka * s[None, :]))
    t = float(np.max(np.abs(C @ b[act]) / sd))
    return sd, kappa, t


def tolerance(kappa: float, t: float) -> float:
    return 2e-12 + 5.0 * EPS * kappa * (1.0 + t)


# --------------------------------------------------------------------------- A12


A12_STRUCTURAL = ["structural_constant_hs", "structural_constant_ng_grid", "structural_sv_hs",
                  "structural_iw_equivalent_hs", "structural_constant_ssvs_beta"]  # fmt: skip


@pytest.mark.parametrize("name", A12_STRUCTURAL)
def test_a12_structural_deviation_path_equals_phase2_path(name):
    """A12 (structural): ``LinearStructural.sample_eq`` with ``GaussianLogDiag`` moments equals the
    Phase 2 weighted path with ``GaussianDiag(mean, exp(log_prec))`` on the same key (module
    docstring), every equation of 10 prior states; ``dev_log = log|δ − m|`` where representable."""
    spec = joint_spec(name, "full")
    program = compose(spec.static)
    mean = program.mean
    n, k = spec.static.shapes.n, spec.static.shapes.k
    worst = 0.0
    for rep in range(10):
        kr = random.fold_in(KEY, rep)
        st = spec.prior_sample(random.fold_in(kr, 0))
        data = spec.simulate(random.fold_in(kr, 1), st)
        cache = _prepare_fn(program)(data, spec.prior)
        sh = sweep_values(program, KEY, jnp.uint32(0), st, data, spec.prior, cache)
        for i in range(n):
            eq_i = jax.tree.map(lambda x, i=i: x[i], st.eq)
            prior_i = jax.tree.map(lambda x, i=i: x[i], spec.prior.eq)
            cache_i = jax.tree.map(lambda x, i=i: x[i], cache.eq)
            ctx = program.structure.context(
                jnp.int32(i), sh.ytil, eq_i, program.vol, data, st.glob.vol
            )
            cm = program.structure.contemp_moments(ctx, prior_i)
            pm = program.coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
            pd = GaussianDiag(mean=pm.mean, prec=jnp.exp(pm.log_prec), cmask=pm.cmask)
            key = random.fold_in(kr, 10 + i)
            args = (eq_i.mean, eq_i.struct.a, ctx)
            ms_l, a_l, _, info_l = mean.sample_eq(key, *args, pm, cm, cache_i, cache,
                                                 spec.prior.shared)  # fmt: skip
            ms_p, a_p, _, info_p = mean.sample_eq(key, *args, pd, cm, cache_i, cache,
                                                 spec.prior.shared)  # fmt: skip
            assert bool(info_l.ok) and bool(info_p.ok)
            cond = mean.conditional_eq(*args, pd, cm, cache_i, cache, spec.prior.shared)
            act = np.concatenate([np.ones(k), np.asarray(cm.cmask)]) > 0
            sd, kappa, t = envelope(np.asarray(cond.K), np.asarray(cond.b), act)
            th_l = np.concatenate([np.asarray(ms_l.delta), np.asarray(a_l)])[act]
            th_p = np.concatenate([np.asarray(ms_p.delta), np.asarray(a_p)])[act]
            err = float(np.max(np.abs(th_l - th_p) / sd))
            worst = max(worst, err / tolerance(kappa, t))
            d = np.asarray(ms_l.delta) - np.asarray(pm.mean)
            dl = np.asarray(ms_l.dev_log)
            rep_ok = EPS * (np.abs(np.asarray(ms_l.delta)) + np.abs(d)) <= 1e-13 * np.abs(d)
            np.testing.assert_allclose(dl[rep_ok], np.log(np.abs(d[rep_ok])), rtol=0, atol=1e-12)
    print(f"A12 {name}: worst error / tolerance = {worst:.3g}")
    assert worst <= 1.0, worst


@pytest.mark.parametrize("name", ["reduced_sv_hs_global", "reduced_sv_ng_global",
                                  "reduced_constant_hs_global", "reduced_sv_ssvs_global"])  # fmt: skip
def test_a12_cccm_deviation_path_equals_phase2_scan(name):
    """A12 (CCCM): ``cccm_core_log`` equals ``cccm_core`` with ``prec0 = exp(log_prec)`` on the same
    ``Z`` at 10 prior states (module docstring); ``dev_log`` row ``j`` is ``log|π_j − m_j|``."""
    spec = joint_spec(name, "full")
    program = compose(spec.static)
    step = program.global_pre[0]
    worst = 0.0
    for rep in range(10):
        kr = random.fold_in(KEY, 100 + rep)
        st = spec.prior_sample(random.fold_in(kr, 0))
        data = spec.simulate(random.fold_in(kr, 1), st)
        cache = _prepare_fn(program)(data, spec.prior)
        pm = step.moments(st, spec.prior)
        inv_lam = step.inv_lam(st, data.tmask)
        n, k = data.Y.shape[1], data.X.shape[1]
        Z = cccm_noise(random.fold_in(kr, 2), n, k)
        a = st.eq.struct.a
        kw = {"gram": spec.static.mean.gram, "precompute": spec.static.mean.cccm_precompute}
        Pi0 = st.glob.mean.Pi
        P_log, dev, info_l = cccm_core_log(Pi0, a, inv_lam, data.X, data.Y, cache.XX, pm.mean,
                                           pm.log_prec, Z, **kw)  # fmt: skip
        P_d14, info_p = cccm_core(Pi0, a, inv_lam, data.X, data.Y, cache.XX, pm.mean,
                                  jnp.exp(pm.log_prec), Z, **kw)  # fmt: skip
        assert bool(info_l.ok) and bool(info_p.ok)
        prec = jnp.exp(pm.log_prec)
        sds, kappa, t = [], 0.0, 0.0
        for j in range(n):  # the conditional of column j at the state the scan reached there
            Pj = P_d14.at[:, j:].set(Pi0[:, j:])
            g = cccm_conditional_j(Pj, a, inv_lam, data.X, data.Y, pm.mean, prec, j)
            sd, kj, tj = envelope(np.asarray(g.K), np.asarray(g.b), np.ones(k, bool))
            sds.append(sd)
            kappa, t = max(kappa, kj), max(t, tj)
        err = np.max(np.abs(np.asarray(P_log) - np.asarray(P_d14)) / np.stack(sds, axis=1))
        worst = max(worst, float(err) / tolerance(kappa, t))
        d = np.asarray(P_log).T - np.asarray(pm.mean)
        ok = np.abs(d) > 1e-6
        np.testing.assert_allclose(np.asarray(dev)[ok], np.log(np.abs(d[ok])), rtol=0, atol=1e-9)
    print(f"A12 {name}: worst error / tolerance = {worst:.3g}")
    assert worst <= 1.0, worst


def test_a12_deviation_path_reaches_extreme_scales():
    """At log precisions of ±1600 the deviation path is finite (``ok``) with ``dev_log`` near
    ``−½ log P`` where the prior dominates, where D-14's draw overflows (P3 §3.4, D-41)."""
    spec = joint_spec("structural_constant_hs", "full")
    program = compose(spec.static)
    st = spec.prior_sample(random.fold_in(KEY, 7))
    data = spec.simulate(random.fold_in(KEY, 8), st)
    cache = _prepare_fn(program)(data, spec.prior)
    sh = sweep_values(program, KEY, jnp.uint32(0), st, data, spec.prior, cache)
    i = 1
    eq_i = jax.tree.map(lambda x: x[i], st.eq)
    prior_i = jax.tree.map(lambda x: x[i], spec.prior.eq)
    ctx = program.structure.context(jnp.int32(i), sh.ytil, eq_i, program.vol, data, st.glob.vol)
    cm = program.structure.contemp_moments(ctx, prior_i)
    pm = program.coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
    gl = np.asarray(spec.prior.eq.coef.gl.mask[i]) > 0
    lp = np.where(gl, np.where(np.arange(gl.size) % 2 == 0, 1600.0, -12.0), pm.log_prec)
    pm = dataclasses.replace(pm, log_prec=jnp.asarray(lp))
    ms, _, _, info = program.mean.sample_eq(
        random.fold_in(KEY, 9), eq_i.mean, eq_i.struct.a, ctx, pm, cm, None, cache,
        spec.prior.shared,
    )  # fmt: skip
    assert bool(info.ok)
    tight = gl & (lp == 1600.0)
    dl = np.asarray(ms.dev_log)
    assert np.all(np.isfinite(dl)) and np.all(np.abs(dl[tight] + 800.0) < 10.0)


# --------------------------------------------------------------------------- A13


def _fixed_twin(spec):
    """The ``FixedDiagonal`` composition and arrays of an SSVS spec whose ``τ₀ = τ₁``."""
    static = spec.static
    mean = dataclasses.replace(static.mean, coef_log=False)
    fstatic = dataclasses.replace(static, mean=mean, coef=CoefPriorStatic())
    c = spec.prior.eq.coef
    on = np.asarray(c.gl.mask) > 0
    prec0 = np.where(on, np.exp(-2.0 * np.asarray(c.gl.ssvs.log_tau1)), np.asarray(c.prec0))
    coef = CoefPriorArrays(mean0=c.mean0, prec0=jnp.asarray(prec0), group=c.group)
    eq = dataclasses.replace(spec.prior.eq, coef=coef)
    shared = dataclasses.replace(spec.prior.shared, gl=Empty())
    return fstatic, PriorArrays(eq=eq, shared=shared)


def _draws(fit):
    p = fit.param_draws
    gamma = np.asarray(p.mean.gamma)  # (C, D, n, k): γ_i = δ_i + M_c a_i
    a = np.asarray(p.struct.a)
    lam = np.asarray(p.vol.core.log_lam)
    n = a.shape[-2]
    rows, cols = np.tril_indices(n, -1)
    C, D = gamma.shape[:2]
    return np.concatenate(
        [gamma.reshape(C, D, -1), a[..., rows, cols].reshape(C, D, -1), lam.reshape(C, D, -1)],
        axis=-1,
    )


def test_a13_ssvs_with_equal_scales_is_the_normal_prior():
    """A13 (module docstring): SSVS ``τ₀ = τ₁`` against ``FixedDiagonal`` on the same data,
    4 × 4,000 draws each, Phase 2 parity rule on the 5/25/50/75/95% quantiles of every ``γ``,
    ``a`` and ``log λ``."""
    spec = joint_spec("structural_constant_ssvs", "full")
    arr = spec.prior.eq.coef.gl
    ssvs = dataclasses.replace(arr.ssvs, log_tau0=arr.ssvs.log_tau1)
    prior_s = dataclasses.replace(
        spec.prior,
        eq=dataclasses.replace(
            spec.prior.eq,
            coef=dataclasses.replace(spec.prior.eq.coef, gl=dataclasses.replace(arr, ssvs=ssvs)),
        ),
    )
    st = spec.prior_sample(random.fold_in(KEY, 13))
    data = spec.simulate(random.fold_in(KEY, 14), st)
    fstatic, prior_f = _fixed_twin(dataclasses.replace(spec, prior=prior_s))
    settings = MCMCSettings(chains=4, draws=4000, burn=500, seed=0)
    runtime = Runtime()
    n = spec.static.shapes.n
    meta = FitMeta(model=None, vardata=None, periods=None,
                   variables=tuple(f"y{i}" for i in range(n)), hyper={})  # fmt: skip
    with warnings.catch_warnings():  # convergence notes of the short runs are not the subject
        warnings.simplefilter("ignore", UserWarning)
        fit_s = fit_mcmc(spec.static, data, prior_s, settings, runtime, seed=3, salt=1, meta=meta)
        fit_f = fit_mcmc(fstatic, data, prior_f, settings, runtime, seed=4, salt=1, meta=meta)
    xs, xf = _draws(fit_s), _draws(fit_f)
    z = []
    for q in Q:
        point = np.quantile(xf.reshape(-1, xf.shape[-1]), q, axis=0)
        ind_s = (xs <= point).astype(np.float64)
        ind_f = (xf <= point).astype(np.float64)
        var = diagnostics.mcse_mean_many(ind_s) ** 2 + diagnostics.mcse_mean_many(ind_f) ** 2
        z.append((ind_s.mean(axis=(0, 1)) - ind_f.mean(axis=(0, 1))) / np.sqrt(var))
    z = np.concatenate(z)
    crit = max(4.0, float(stats.norm.isf(0.01 / (2.0 * z.size))))
    print(f"A13: m = {z.size}, max |z| = {np.max(np.abs(z)):.2f}, critical {crit:.2f}")
    assert np.all(np.abs(z) <= crit), (np.sort(np.abs(z))[-5:], crit)
