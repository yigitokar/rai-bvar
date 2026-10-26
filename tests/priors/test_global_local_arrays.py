"""Host builders of the global-local and SSVS priors (P3 DESIGN §3.6; §9.2 item 5, E126; WP31).

(a) ``log r`` against WP30's ``gl_log_r_reference`` at 1e-14 on every slot (``n = 3, p = 2``,
unequal ψ, deterministic terms, both forms, ``scale="none"`` exactly 0); (b) the semi-automatic
``ŝ`` against WP30's dense ``ssvs_shat_reference`` (1e-10), both forms, and E-12 at the ``T_eff``
boundaries of D-44 (in the builder and at fit, verbatim); (c) scale equivariance: with
``y_j → c_j y_j``, ``c = (3, 0.1, 7)``, ψ rebuilt through ``window_scales`` and ``gl_arrays`` and the
same key, ``log r`` shifts by ``2(log c_i − log c_m)``, the semi-automatic ``log τ₀, log τ₁`` by
``log c_i − log c_m``, every δ (or π) and GL deviation after one production sweep (equation
phase, volatility, scale step) scales by ``c_i/c_m`` and the scale-step draws agree to 1e-9;
every family, structural Mode I and reduced form. Also: the group layouts of D-38 against WP30's
``joint.gl_groups``, E-13 at fit, and the ``GLShared`` encodings.
"""

from __future__ import annotations

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import random

import rai_bvar as rb
from rai_bvar.data import VARData
from rai_bvar.data.design import lagged_design
from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.resolve import resolve
from rai_bvar.engines.mcmc.runner import _prepare_fn, make_runner
from rai_bvar.engines.mcmc.state import CCCMGlobal, ConstantEq, LinearEq
from rai_bvar.engines.mcmc.sweep import make_sweep
from rai_bvar.priors.global_local_arrays import gl_arrays, gl_log_r, ssvs_shat
from rai_bvar.priors.scales import WindowScales, prior_mean_matrix, window_scales
from rai_bvar.priors.sv import Beta
from rai_bvar.settings import Runtime
from rai_bvar.validation import joint as J
from rai_bvar.validation.reference import gl_log_r_reference, ssvs_shat_reference

PSI = np.array([0.5, 1.7, 4.2])
C = np.array([3.0, 0.1, 7.0])
FAMILIES = {
    "horseshoe": lambda **kw: rb.Horseshoe(global_scale=0.5, **kw),
    "normal_gamma": lambda **kw: rb.NormalGamma(**kw),
    "normal_gamma_c": lambda **kw: rb.NormalGamma(a=0.3, b=1.5, c=0.7, **kw),
    "r2d2": lambda **kw: rb.R2D2(**kw),
    "dirichlet_laplace": lambda **kw: rb.DirichletLaplace(**kw),
    "ssvs": lambda **kw: rb.SSVS(**kw),
    "ssvs_beta_fixed_tau": lambda **kw: rb.SSVS(p=Beta(2.0, 3.0), semiautomatic=False, **kw),
}


def frame(T: int = 90, n: int = 3, seed: int = 0, scale: np.ndarray | None = None) -> VARData:
    """A stationary VAR(1)-like sample in levels (tcode 1: the transformed series is the data, so
    scaling the frame scales the series)."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T, n))
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, -0.1], [0.0, 0.3, 0.6]])[:n, :n]
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(n) * np.array([1.0, 0.5, 2.0])[:n]
    if scale is not None:
        y = y * scale[None, :]
    index = pd.period_range("1990Q1", periods=T, freq="Q")
    df = pd.DataFrame(y, index=index, columns=[f"y{j}" for j in range(n)])
    return VARData.from_frame(df, dict.fromkeys(df.columns, 1))


def scales_of(psi: np.ndarray) -> WindowScales:
    return WindowScales(psi=psi, own_means=np.full(psi.size, 0.4), freq="Q")


# --------------------------------------------------------------------------- (a) log r


@pytest.mark.parametrize("form", ["structural", "reduced"])
@pytest.mark.parametrize("n_det", [0, 1, 2])
@pytest.mark.parametrize("scale", ["psi", "none"])
def test_log_r_against_reference(form, n_det, scale):
    """``log r`` of ``gl_arrays`` equals WP30's ``gl_log_r_reference`` on every slot (1e-14;
    ``n = 3, p = 2``, unequal ψ); ``scale="none"`` and the deterministic slots exactly 0."""
    n, lags = 3, 2
    prior = rb.Horseshoe(scale=scale, form=form)
    arr, _, G = gl_arrays(prior, scales_of(PSI), n=n, lags=lags, n_det=n_det, form=form)
    ref = gl_log_r_reference(PSI, n, lags, n_det, form, scale)
    np.testing.assert_allclose(arr.log_r, ref, rtol=0, atol=1e-14)
    assert np.all(arr.log_r[:, :n_det] == 0.0)
    if scale == "none":
        assert np.all(arr.log_r == 0.0)
    np.testing.assert_allclose(gl_log_r(PSI, n, lags, n_det, form, scale), arr.log_r, atol=0)
    assert np.all(arr.mask[:, :n_det] == 0.0) and np.all(arr.mask[:, n_det:] == 1.0)
    assert G == 2 * lags


# --------------------------------------------------------------------------- D-38 groups


@pytest.mark.parametrize("groups", ["olcl-lagwise", "global", "equation-wise", "covariate-wise"])
@pytest.mark.parametrize("n,lags", [(3, 2), (1, 3), (4, 1)])
def test_group_layouts_against_reference(groups, n, lags):
    """D-38's group layouts against WP30's ``joint.gl_groups`` (up to a relabelling that keeps
    every group non-empty: ``olcl-lagwise`` at ``n = 1`` has no cross-lag groups)."""
    n_det = 1
    arr, _, G = gl_arrays(rb.NormalGamma(groups=groups), scales_of(np.ones(n)), n=n, lags=lags,
                          n_det=n_det, form="structural")  # fmt: skip
    ref, G_ref = J.gl_groups(groups, n, lags, n_det)
    ref = np.asarray(ref)
    lag = arr.mask > 0
    _, inverse = np.unique(ref[lag], return_inverse=True)
    np.testing.assert_array_equal(arr.group[lag], inverse)
    assert G == inverse.max() + 1 <= G_ref
    assert np.all(arr.group[~lag] == 0)


def test_custom_groups_and_E13_at_fit():
    """A custom ``(n, n·p)`` array is placed on the lag slots; another shape raises E-13 at fit."""
    n, lags = 3, 2
    ids = np.arange(n * n * lags).reshape(n, n * lags) % 4
    arr, _, G = gl_arrays(rb.Horseshoe(groups=ids), scales_of(PSI), n=n, lags=lags, n_det=1,
                          form="structural")  # fmt: skip
    np.testing.assert_array_equal(arr.group[:, 1:], ids)
    assert G == 4
    bad = rb.BVAR(mean=rb.Linear(rb.Horseshoe(groups=np.zeros((2, 4), dtype=int))),
                  vol=rb.Constant(), lags=2, mcmc=rb.MCMCSettings(chains=2, draws=10, burn=10))  # fmt: skip
    msg = "groups array has shape (2, 4); expected (n, n·lags) = (3, 6)"
    with pytest.raises(ValueError, match=re.escape(msg)):
        bad.fit(frame())


def test_gl_shared_encodings():
    """``GLShared``: NG's tied ``c`` as ``−1``, a fixed ``a`` as ``log_a_fixed`` with a one-point
    zero grid, the grid's log weights, the HS scale, SSVS ``p`` or its Beta (unused fields 0)."""
    s = scales_of(PSI)
    kw = dict(n=3, lags=2, n_det=1, form="structural")
    _, ng, _ = gl_arrays(rb.NormalGamma(), s, **kw)
    gk = rb.DiscreteGrid.gruber_kastner()
    assert float(ng.c) == -1.0 and float(ng.b) == 0.5
    np.testing.assert_array_equal(ng.a_support, gk.support)
    np.testing.assert_allclose(np.exp(ng.a_logw), gk.weights, rtol=1e-15)
    _, ngc, _ = gl_arrays(rb.NormalGamma(a=0.3, c=0.7), s, **kw)
    assert float(ngc.c) == 0.7 and float(ngc.log_a_fixed) == np.log(0.3)
    assert ngc.a_support.shape == (1,) and not ngc.a_support.any()
    _, hs, _ = gl_arrays(rb.Horseshoe(global_scale=0.3), s, **kw)
    assert float(hs.global_scale) == 0.3 and float(hs.b) == 0.0
    _, sv, _ = gl_arrays(rb.SSVS(p=Beta(2.0, 3.0), semiautomatic=False), s, **kw)
    assert (float(sv.s1), float(sv.s2), float(sv.p_fixed)) == (2.0, 3.0, 0.0)
    arr, sf, _ = gl_arrays(rb.SSVS(c0=0.02, c1=4.0, semiautomatic=False, p=0.3), s, **kw)
    assert float(sf.p_fixed) == 0.3
    lag = arr.mask > 0
    np.testing.assert_allclose(arr.ssvs.log_tau0[lag], (np.log(0.02) + 0.5 * arr.log_r)[lag])
    np.testing.assert_allclose(arr.ssvs.log_tau1[lag], (np.log(4.0) + 0.5 * arr.log_r)[lag])


# --------------------------------------------------------------------------- (b) ŝ and E-12


@pytest.mark.parametrize("form", ["structural", "reduced"])
@pytest.mark.parametrize("det", ["c", "ct"])
def test_shat_against_reference(form, det):
    """The semi-automatic ``ŝ`` (one QR per equation) against WP30's dense oracle (1e-10), and
    the SSVS arrays ``log τ = log c + log ŝ`` on the lag slots."""
    data = frame(T=60)
    lags, n = 2, 3
    Y, X = lagged_design(data.values, lags, det)
    n_det = X.shape[1] - n * lags
    prior = rb.SSVS(c0=0.05, c1=3.0, form=form)
    scales = window_scales(data, prior.minnesota(), lags, det)
    Mc = prior_mean_matrix(scales, n, lags, n_det)[:, : n - 1]
    ref = ssvs_shat_reference(Y, X, form, Mc=Mc if form == "structural" else None)
    got = ssvs_shat(Y, X, form, Mc=Mc)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=0)
    arr, _, _ = gl_arrays(prior, scales, n=n, lags=lags, n_det=n_det, form=form, data=(Y, X))
    lag = arr.mask > 0
    np.testing.assert_allclose(arr.ssvs.log_tau0[lag], np.log(0.05 * ref)[lag], rtol=0, atol=1e-10)
    np.testing.assert_allclose(arr.ssvs.log_tau1[lag], np.log(3.0 * ref)[lag], rtol=0, atol=1e-10)


@pytest.mark.parametrize("form", ["structural", "reduced"])
def test_E12_at_the_T_eff_boundaries(form):
    """D-44: ``T_eff > k + n + 1`` (structural ``T_eff > q_i + 2`` for every ``i`` binds at
    ``i = n − 1``): at ``T_eff = k + n + 1`` the builder and the fit raise E-12 verbatim (and WP30's
    reference rejects the divisor); one more row passes."""
    n, lags = 3, 2
    k = 1 + n * lags
    T_bad = k + n + 1
    msg = (
        "SSVS(semiautomatic=True) scales by the flat-prior posterior s.d., which needs more "
        f"observations than coefficients (T = {T_bad}, needed > {T_bad}); use "
        "semiautomatic=False or fewer lags"
    )
    for T_eff, ok in ((T_bad, False), (T_bad + 1, True)):
        data = frame(T=T_eff + lags)
        Y, X = lagged_design(data.values, lags, "c")
        assert Y.shape[0] == T_eff
        prior = rb.SSVS(form=form)
        scales = window_scales(data, prior.minnesota(), lags, "c")
        Mc = prior_mean_matrix(scales, n, lags, 1)[:, : n - 1]
        if ok:
            ssvs_shat(Y, X, form, Mc=Mc)
            ssvs_shat_reference(Y, X, form, Mc=Mc if form == "structural" else None)
            continue
        with pytest.raises(ValueError, match=re.escape(msg)):
            ssvs_shat(Y, X, form, Mc=Mc)
        with pytest.raises(ValueError):
            ssvs_shat_reference(Y, X, form, Mc=Mc if form == "structural" else None)
        model = rb.BVAR(mean=rb.Linear(prior), vol=rb.Constant(), lags=lags,
                        mcmc=rb.MCMCSettings(chains=2, draws=10, burn=10))  # fmt: skip
        with pytest.raises(ValueError, match=re.escape(msg)):
            model.fit(data)


# --------------------------------------------------------------------------- (c) equivariance


def _model(family: str, form: str) -> rb.BVAR:
    prior = FAMILIES[family](form=form)
    return rb.BVAR(mean=rb.Linear(prior), vol=rb.Constant("ig"), lags=2,
                   mcmc=rb.MCMCSettings(chains=1, draws=10, burn=10))  # fmt: skip


def _factors(n: int, lags: int, n_det: int) -> tuple[np.ndarray, np.ndarray]:
    """``(f, g)`` (``n × k``): coefficient of equation ``i`` on regressor ``r`` scales by
    ``f = c_i/c_m`` (lag of variable ``m``) or ``c_i`` (deterministic), and its log by
    ``g = log f``."""
    k = n_det + n * lags
    m = np.concatenate([np.full(n_det, -1), np.tile(np.arange(n), lags)])
    f = np.where(m[None, :] < 0, C[:, None], C[:, None] / C[np.maximum(m, 0)][None, :])
    assert f.shape == (n, k)
    return f, np.log(f)


def _image(st, static, f, g):
    """The equivariant image of a one-chain state under ``y_j → c_j y_j``."""
    n = static.shapes.n
    lam = st.eq.vol.core.log_lam + 2.0 * np.log(C)
    a = st.eq.struct.a * (C[:, None] / C[None, : n - 1])
    eq = dataclasses.replace(
        st.eq,
        struct=dataclasses.replace(st.eq.struct, a=a),
        vol=dataclasses.replace(st.eq.vol, core=ConstantEq(log_lam=lam)),
    )
    if static.mean.kind == "linear_structural":
        eq = dataclasses.replace(
            eq, mean=LinearEq(delta=st.eq.mean.delta * f, dev_log=st.eq.mean.dev_log + g)
        )
        return dataclasses.replace(st, eq=eq)
    mean = CCCMGlobal(Pi=st.glob.mean.Pi * f.T, dev_log=st.glob.mean.dev_log + g)
    return dataclasses.replace(st, eq=eq, glob=dataclasses.replace(st.glob, mean=mean))


@pytest.mark.parametrize("form", ["structural", "reduced"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_scale_equivariance(family, form):
    """P3 §9.2 item 5(c) for every family in structural Mode I and reduced form (module docstring);
    the coefficients and deviations at 1e-9 relative (``log`` scale: 1e-9 absolute), the scale-step
    draws at 1e-9 (logs; indices and indicators exactly)."""
    data, data_c = frame(), frame(scale=C)
    model = _model(family, form)
    runtime = Runtime()
    r0, r1 = resolve(model, data, runtime), resolve(model, data_c, runtime)
    static = r0.static
    assert r1.static == static
    n, lags, n_det = static.shapes.n, static.shapes.lags, static.shapes.n_det
    f, g = _factors(n, lags, n_det)
    a0, a1 = r0.prior.eq.coef.gl, r1.prior.eq.coef.gl
    lag = np.asarray(a0.mask) > 0
    np.testing.assert_allclose((a1.log_r - a0.log_r)[lag], 2.0 * g[lag], rtol=0, atol=1e-12)
    if family == "ssvs":
        for leaf in ("log_tau0", "log_tau1"):
            d = getattr(a1.ssvs, leaf) - getattr(a0.ssvs, leaf)
            np.testing.assert_allclose(d[lag], g[lag], rtol=0, atol=1e-10)
    fns = make_runner(static)
    program = compose(static)
    sweep = jax.jit(make_sweep(program))
    prep = jax.jit(_prepare_fn(program))
    d0, p0 = jax.tree.map(jnp.asarray, (r0.data, r0.prior))
    d1, p1 = jax.tree.map(jnp.asarray, (r1.data, r1.prior))
    c0, c1 = prep(d0, p0), prep(d1, p1)
    ikeys = keys.init_keys(5, 17, 1)
    st0 = jax.tree.map(lambda x: x[0], fns.init(ikeys, d0, p0, c0))
    st1 = _image(st0, static, f, g)
    ckey = random.fold_in(random.key(5), 3)
    new0 = sweep(ckey, jnp.uint32(11), st0, d0, p0, c0)
    new1 = sweep(ckey, jnp.uint32(11), st1, d1, p1, c1)
    assert int(new0.status.code) == 0 and int(new1.status.code) == 0
    if static.mean.kind == "linear_structural":
        x0, x1 = np.asarray(new0.eq.mean.delta), np.asarray(new1.eq.mean.delta)
        l0, l1 = np.asarray(new0.eq.mean.dev_log), np.asarray(new1.eq.mean.dev_log)
    else:
        x0, x1 = np.asarray(new0.glob.mean.Pi).T, np.asarray(new1.glob.mean.Pi).T
        l0, l1 = np.asarray(new0.glob.mean.dev_log), np.asarray(new1.glob.mean.dev_log)
    np.testing.assert_allclose(x1, x0 * f, rtol=1e-9, atol=0)
    np.testing.assert_allclose(l1, l0 + g, rtol=0, atol=1e-9)
    for path, u in jax.tree_util.tree_flatten_with_path((new0.eq.coef, new0.glob.coef))[0]:
        v = jax.tree_util.tree_flatten_with_path((new1.eq.coef, new1.glob.coef))[0]
        v = dict((jax.tree_util.keystr(p), x) for p, x in v)[jax.tree_util.keystr(path)]
        u, v = np.asarray(u), np.asarray(v)
        if np.issubdtype(u.dtype, np.integer):
            np.testing.assert_array_equal(u, v)
        else:
            np.testing.assert_allclose(v, u, rtol=0, atol=1e-9, err_msg=str(path))
