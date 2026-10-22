"""Reference log densities against ``scipy.stats`` and the closed-form conditionals of the notes
(DESIGN §9.2–§9.3, §13 WP9; N06 §5.1, §5.4; N02 §5.2; N03 §4.1–§4.4; N05 §5.1–§5.2).

1. ``log_prior``, ``loglik`` and ``log_joint`` equal independent ``scipy.stats`` sums (1e-10);
   the augmented joint with mixture indicators equals its NumPy evaluation (exact and auxiliary).
2. Padded rows and padded SV states never enter any density.
3. The reference joint reproduces the exact conditionals the kernels must implement: the IG
   λ conditionals with and without coupling, the Gaussian ``δ_i``, ``a_i`` (D3 design) and
   reduced-form ``Π`` (SUR) precisions, the AWOL precision of ``h | s``, and the ``o``, ``p`` and
   ``τ`` conditionals — the density-ratio identities of tier U, with the kernels replaced by the
   formulas of the notes.
"""

from __future__ import annotations

import math
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats
from scipy.special import logsumexp

from rai_bvar.engines.mcmc.state import leaf_path
from rai_bvar.validation import joint as J
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation import reference as R

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _np(tree):
    return jax.tree.map(np.asarray, tree)


def set_leaf(st, path: str, fn):
    """``st`` with the leaf at dotted ``path`` replaced by ``fn(leaf)``."""
    return jax.tree_util.tree_map_with_path(lambda p, x: fn(x) if leaf_path(p) == path else x, st)


def close(a, b, tol=1e-10):
    a, b = float(a), float(b)
    assert abs(a - b) <= tol * (1.0 + abs(b)), (a, b, a - b)


def setup(name: str, size: str = "small", seed: int = 0, **kw):
    spec = J.joint_spec(name, size, **kw)
    st = spec.prior_sample(random.key(seed))
    data = spec.simulate(random.key(seed + 100), st)
    return spec, st, data


# --------------------------------------------------------------------------- NumPy / scipy oracles


def numpy_effective_log_var(spec, st):
    st = _np(st)
    static, vol = spec.static, st.eq.vol
    T_pad = static.shapes.T_pad
    if static.vol.persistent == "constant":
        lv = np.repeat(vol.core.log_lam[:, None], T_pad, axis=1)
    else:
        lv = vol.core.h[:, 1:].copy()
    if static.vol.transitory in ("outlier", "outlier_t"):
        lv = lv + 2.0 * np.log(vol.trans.o_idx + 1.0)
    if static.vol.transitory in ("t", "outlier_t"):
        lv = lv + vol.trans.log_tau
    return lv


def numpy_residuals(spec, st, data):
    st, data = _np(st), _np(data)
    n = spec.static.shapes.n
    T = spec.fixed.T_eff
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = st.eq.struct.a[i, :i]
    Y, X = data.Y[:T], data.X[:T]
    kind = spec.static.mean.kind
    if kind == "linear_structural":
        Gam = st.eq.mean.delta + st.eq.struct.a @ np.asarray(spec.prior.shared.Mc).T
        E = Y @ A.T - X @ Gam.T
    elif kind == "linear_reduced":
        E = (Y - X @ st.glob.mean.Pi) @ A.T
    else:
        E = Y @ A.T
    return E.T  # (n, T)


def scipy_loglik(spec, st, data):
    T = spec.fixed.T_eff
    e = numpy_residuals(spec, st, data)
    lv = numpy_effective_log_var(spec, st)[:, :T]
    return float(np.sum(stats.norm.logpdf(e, 0.0, np.exp(0.5 * lv))))


def scipy_log_prior(spec, st):
    st, pr = _np(st), _np(spec.prior)
    static, vol = spec.static, spec.static.vol
    n, T = static.shapes.n, spec.fixed.T_eff
    total = 0.0
    for i in range(n):
        core = jax.tree.map(lambda x, i=i: x[i], st.eq.vol.core)
        cp = jax.tree.map(lambda x, i=i: x[i], pr.eq.vol_core)
        lam = 1.0
        if vol.persistent == "constant":
            lam = math.exp(core.log_lam)
            total += stats.invgamma.logpdf(lam, cp.shape, scale=cp.scale)
        else:
            s = core.sigma
            if vol.sigma2_prior == "scaled_chi2":
                total += stats.halfnorm.logpdf(s, scale=math.sqrt(cp.B_sigma))
            else:
                total += stats.invgamma.logpdf(s**2, cp.c0, scale=cp.C0) + math.log(2 * s)
            h = core.h
            if vol.rw:
                total += stats.norm.logpdf(h[0], cp.m0, math.sqrt(cp.V0))
                total += np.sum(stats.norm.logpdf(h[1 : T + 1], h[:T], s))
            else:
                mu, phi = core.mu, core.phi
                total += stats.norm.logpdf(mu, cp.b_mu, math.sqrt(cp.B_mu))
                total += stats.beta.logpdf((phi + 1) / 2, cp.a_phi, cp.b_phi) - math.log(2)
                var0 = s**2 / (1 - phi**2) if vol.h0 == "stationary" else cp.B0 * s**2
                total += stats.norm.logpdf(h[0], mu, math.sqrt(var0))
                total += np.sum(stats.norm.logpdf(h[1 : T + 1], mu + phi * (h[:T] - mu), s))
        if vol.transitory != "none":
            tr = jax.tree.map(lambda x, i=i: x[i], st.eq.vol.trans)
            tp = jax.tree.map(lambda x, i=i: x[i], pr.eq.transitory)
        if vol.transitory in ("outlier", "outlier_t"):
            p, K = math.exp(tr.log_p), vol.outlier_K
            total += stats.beta.logpdf(p, tp.a_p, tp.b_p)
            o = tr.o_idx[:T] + 1
            total += np.sum(np.where(o == 1, math.log1p(-p), math.log(p / (K - 1))))
        if vol.transitory in ("t", "outlier_t"):
            nu = float(vol.nu_grid[tr.nu_idx])
            c = nu - 2 if vol.t_scale == "standardized" else nu
            total += -math.log(len(vol.nu_grid))
            total += np.sum(stats.invgamma.logpdf(np.exp(tr.log_tau[:T]), nu / 2, scale=c / 2))
        cs = lam if static.struct.contemp_scaling == "lambda" else 1.0
        for j in range(i):
            total += stats.norm.logpdf(st.eq.struct.a[i, j], 0.0, 0.7 * math.sqrt(cs))
        if static.mean.kind == "linear_structural":
            ms = lam if static.mean.coef_scaling == "lambda" else 1.0
            sd = 1 / np.sqrt(pr.eq.coef.prec0[i])
            total += np.sum(
                stats.norm.logpdf(st.eq.mean.delta[i], pr.eq.coef.mean0[i], sd * math.sqrt(ms))
            )
    if static.mean.kind == "linear_reduced":
        sd = 1 / np.sqrt(pr.eq.coef.prec0.T)
        total += np.sum(stats.norm.logpdf(st.glob.mean.Pi, pr.eq.coef.mean0.T, sd))
    return float(total)


ALL_NAMES = J.COMPOSITIONS + J.BLOCK_CONFIGS


@pytest.mark.parametrize("name", ALL_NAMES)
def test_log_prior_loglik_joint_vs_scipy(name):
    for size, seed in (("small", 1), ("full", 2)):
        spec, st, data = setup(name, size, seed)
        lp = LJ.log_prior(st, spec.prior, spec.static, data.tmask)
        ll = LJ.loglik(st, data, spec.prior, spec.static)
        close(lp, scipy_log_prior(spec, st))
        close(ll, scipy_loglik(spec, st, data))
        close(LJ.log_joint(st, data, spec.prior, spec.static), float(lp) + float(ll), 1e-13)
        close(spec.log_joint(st, data), float(lp) + float(ll), 1e-13)


@pytest.mark.parametrize("name", ["structural_sv_outliers_t", "reduced_sv", "zero_sv", "aux_sv"])
@pytest.mark.parametrize("exact", [True, False])
def test_augmented_joint_vs_numpy(name, exact):
    spec, st, data = setup(name, seed=3)
    sh = spec.static.shapes
    T = spec.fixed.T_eff
    s = np.asarray(random.randint(random.key(9), (sh.n, sh.T_pad), 0, 10), dtype=np.int32)
    tab = R.mixture_table(spec.static.vol.mixture)
    e = numpy_residuals(spec, st, data)
    lv = numpy_effective_log_var(spec, st)[:, :T]
    h = np.asarray(st.eq.vol.core.h)[:, 1 : T + 1]
    log_c = np.asarray(spec.prior.eq.vol_core.log_c)[:, None]
    ystar = np.logaddexp(np.log(e**2) - (lv - h), log_c)  # log(ẽ² + c), ẽ² = e²/(o²τ)
    xi = ystar - h
    si = s[:, :T]
    aug = np.log(tab.p[si]) + stats.norm.logpdf(xi, tab.m[si], np.sqrt(tab.v[si]))
    if exact:
        g = logsumexp(
            np.log(tab.p) + stats.norm.logpdf(xi[..., None], tab.m, np.sqrt(tab.v)), axis=-1
        )
        aug = aug - g
    ref = scipy_log_prior(spec, st) + np.sum(aug) + (scipy_loglik(spec, st, data) if exact else 0.0)
    got = LJ.log_joint(st, data, spec.prior, spec.static, jnp.asarray(s), exact=exact)
    close(got, ref)
    np.testing.assert_allclose(
        np.asarray(LJ.y_star(st, data, spec.prior, spec.static))[:, :T], ystar, rtol=1e-12
    )
    if exact == spec.static.vol.exact:  # default follows the static flag
        close(LJ.log_joint(st, data, spec.prior, spec.static, jnp.asarray(s)), ref)


def test_augmented_joint_needs_sv_core():
    spec, st, data = setup("structural_constant")
    s = jnp.zeros((2, spec.static.shapes.T_pad), jnp.int32)
    with pytest.raises(ValueError, match="SV core"):
        LJ.log_joint(st, data, spec.prior, spec.static, s)


@pytest.mark.parametrize("name", ["structural_sv_outliers_t", "reduced_sv_t", "structural_sv_rw"])
def test_padding_never_enters_a_density(name):
    spec, st, data = setup(name, seed=4)
    T = spec.fixed.T_eff
    base = float(spec.log_joint(st, data))
    st2 = set_leaf(st, "eq.vol.core.h", lambda h: h.at[:, T + 1 :].set(50.0))
    st2 = set_leaf(st2, "eq.vol.trans.log_tau", lambda x: x.at[:, T:].set(7.0))
    st2 = set_leaf(st2, "eq.vol.trans.o_idx", lambda x: x.at[:, T:].set(5))
    data2 = data.__class__(
        Y=data.Y.at[T:].set(1e3), X=data.X.at[T:].set(-1e3), tmask=data.tmask, T_eff=data.T_eff,
        hist=data.hist, t_last=data.t_last,
    )  # fmt: skip
    assert float(spec.log_joint(st2, data2)) == base
    s = jnp.zeros((spec.static.shapes.n, spec.static.shapes.T_pad), jnp.int32)
    s2 = s.at[:, T:].set(9)
    assert float(spec.log_joint(st2, data2, s2)) == float(spec.log_joint(st, data, s))


# --------------------------------------------------------------------------- exact conditionals


def _eq_rss(spec, st, data, i):
    e = numpy_residuals(spec, st, data)[i]
    T = spec.fixed.T_eff
    lv = numpy_effective_log_var(spec, st)[i, :T] - (
        np.asarray(st.eq.vol.core.log_lam)[i] if spec.static.vol.persistent == "constant" else 0.0
    )
    return float(np.sum(e**2 * np.exp(-lv)))  # Σ e²/(o²τ): the λ-free weighted RSS


@pytest.mark.parametrize(
    ("name", "n_active"),
    [("structural_constant", None), ("constant_transitory", None),
     ("structural_mode2_constant", "k+i"), ("structural_iw_equivalent", "i"),
     ("reduced_iw_equivalent", "i")],
)  # fmt: skip
def test_constant_lambda_conditional_including_coupling(name, n_active):
    """``λ_i | · ~ IG(c + (T + n_active)/2, C + ½ Σ e²/(o²τ) + quad)`` (N05 §5.2; DESIGN §3.7.5,
    §4.7; N03 §4.5): the reference joint's λ-ratio equals the IG density ratio."""
    spec, st, data = setup(name, seed=5)
    sh, T = spec.static.shapes, spec.fixed.T_eff
    pr = _np(spec.prior)
    for i in range(sh.n):
        shape, scale = float(pr.eq.vol_core.shape[i]), float(pr.eq.vol_core.scale[i])
        a_dev = np.asarray(st.eq.struct.a)[i, :i] ** 2 * pr.eq.contemp.prec0[i, :i]
        if n_active is None:
            q, quad = 0, 0.0
        elif n_active == "i":
            q, quad = i, 0.5 * float(np.sum(a_dev))
        else:
            d = np.asarray(st.eq.mean.delta)[i] - pr.eq.coef.mean0[i]
            q, quad = sh.k + i, 0.5 * float(np.sum(a_dev) + np.sum(d**2 * pr.eq.coef.prec0[i]))
        post_a = shape + 0.5 * (T + q)
        post_b = scale + 0.5 * _eq_rss(spec, st, data, i) + quad
        lj = []
        for lam in (0.37, 2.9):
            st2 = set_leaf(
                st, "eq.vol.core.log_lam", lambda x, lam=lam, i=i: x.at[i].set(math.log(lam))
            )
            lj.append(float(spec.log_joint(st2, data)))
        ref = stats.invgamma.logpdf(2.9, post_a, scale=post_b) - stats.invgamma.logpdf(
            0.37, post_a, scale=post_b
        )
        close(lj[1] - lj[0], ref, 1e-9)


def _weights(spec, st, data, i):
    T = spec.fixed.T_eff
    lv = numpy_effective_log_var(spec, st)[i, :T]
    return np.exp(-lv)


@pytest.mark.parametrize(
    "name", ["structural_sv_outliers_t", "structural_mode2_constant", "structural_sv_rw"]
)
def test_structural_equation_precisions(name):
    """``(δ_i, a_i) | ·`` is Gaussian with precision ``diag(P̃) + F_iᵀ W F_i`` in δ-coordinates,
    ``F = [X | X M_c − Y_{<i}]`` (N05 §4.1, §5.1; DESIGN §3.7.2): Hessians and gradients of the
    reference joint."""
    spec, st, data = setup(name, "full", seed=6)
    sh, T = spec.static.shapes, spec.fixed.T_eff
    pr = _np(spec.prior)
    X, Y = np.asarray(data.X)[:T], np.asarray(data.Y)[:T]
    Mc = pr.shared.Mc
    for i in range(sh.n):
        w = _weights(spec, st, data, i)
        lam = math.exp(float(st.eq.vol.core.log_lam[i])) if spec.static.vol.coupled else 1.0
        pd = pr.eq.coef.prec0[i] / (lam if spec.static.mean.coef_scaling == "lambda" else 1.0)
        pa = pr.eq.contemp.prec0[i, :i] / (
            lam if spec.static.struct.contemp_scaling == "lambda" else 1.0
        )
        F = np.hstack([X, X @ Mc[:, :i] - Y[:, :i]])
        K = np.diag(np.concatenate([pd, pa])) + F.T @ (w[:, None] * F)
        theta0 = np.concatenate(
            [np.asarray(st.eq.mean.delta)[i], np.asarray(st.eq.struct.a)[i, :i]]
        )

        def f(theta, i=i):
            s2 = set_leaf(st, "eq.mean.delta", lambda d: d.at[i].set(theta[: sh.k]))
            s2 = set_leaf(s2, "eq.struct.a", lambda a: a.at[i, :i].set(theta[sh.k :]))
            return spec.log_joint(s2, data)

        H = np.asarray(jax.jit(jax.hessian(f))(jnp.asarray(theta0)))
        np.testing.assert_allclose(-H, K, rtol=1e-9, atol=1e-9 * np.abs(K).max())
        # gradient = b − Kθ with b = P̃⊙m + Fᵀ W y_i (target y_i, regressors F)
        mean0 = np.concatenate([pr.eq.coef.mean0[i], pr.eq.contemp.mean0[i, :i]])
        b = np.concatenate([pd, pa]) * mean0 + F.T @ (w * Y[:, i])
        g = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(theta0)))
        np.testing.assert_allclose(g, b - K @ theta0, rtol=1e-9, atol=1e-9 * np.abs(b).max())


def test_reduced_form_sur_precision():
    """``vec Π | A, Λ`` has precision ``V⁻¹ + Σ_t Σ_t⁻¹ ⊗ x_t x_tᵀ``, ``Σ_t⁻¹ = AᵀΛ_t⁻¹A`` (N06 §5.4;
    CCCM 2022): the full cross-equation coupling the corrected algorithm must respect."""
    spec, st, data = setup("reduced_sv_outliers_t", "full", seed=7)
    sh, T = spec.static.shapes, spec.fixed.T_eff
    pr = _np(spec.prior)
    X = np.asarray(data.X)[:T]
    n, k = sh.n, sh.k
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = np.asarray(st.eq.struct.a)[i, :i]
    lam_inv = np.exp(-numpy_effective_log_var(spec, st)[:, :T])  # (n, T)
    Sinv = np.einsum("ji,jt,jl->til", A, lam_inv, A)  # Σ_t⁻¹ = Aᵀ Λ_t⁻¹ A
    # H[r, i, s, l] = ∂²/∂Π_ri ∂Π_sl: Σ_t (Σ_t⁻¹)_il x_tr x_ts + δ_rs δ_il / V_{ri}
    K = np.einsum("til,tr,ts->risl", Sinv, X, X)
    K = K + np.einsum("rs,il,ri->risl", np.eye(k), np.eye(n), pr.eq.coef.prec0.T)

    def f(Pi):
        return spec.log_joint(set_leaf(st, "glob.mean.Pi", lambda _: Pi), data)

    H = np.asarray(jax.jit(jax.hessian(f))(st.glob.mean.Pi))  # (k, n, k, n)
    np.testing.assert_allclose(-H, K, rtol=1e-9, atol=1e-9 * np.abs(K).max())


@pytest.mark.parametrize(
    ("name", "fixed_b0"), [("zero_sv", False), ("structural_sv_fixed_b0", True)]
)
def test_awol_precision_of_h_given_s(name, fixed_b0):
    """``h_{0:T} | y*, s, θ`` has the tridiagonal precision of N02 §5.2 (auxiliary model, ``s``
    given): ``Q_00 = (ω₀ + φ²)/σ²``, ``Q_tt = 1/v_s + (1+φ²)/σ²``, ``Q_TT = 1/v_s + 1/σ²``,
    ``Q_{t,t−1} = −φ/σ²``; padded states carry no density."""
    spec, st, data = setup(name, seed=8, exact=False)
    sh, T = spec.static.shapes, spec.fixed.T_eff
    s = random.randint(random.key(3), (sh.n, sh.T_pad), 0, 10).astype(jnp.int32)
    tab = R.mixture_table("ocsn10")
    core = _np(st.eq.vol.core)
    mu, phi, sig = float(core.mu[0]), float(core.phi[0]), float(core.sigma[0])
    omega0 = 1.0 / float(spec.prior.eq.vol_core.B0[0]) if fixed_b0 else 1.0 - phi**2
    v = tab.v[np.asarray(s)[0, :T]]
    Q = np.zeros((T + 1, T + 1))
    Q[0, 0] = (omega0 + phi**2) / sig**2
    for t in range(1, T + 1):
        Q[t, t] = 1 / v[t - 1] + ((1 + phi**2) if t < T else 1.0) / sig**2
        Q[t, t - 1] = Q[t - 1, t] = -phi / sig**2

    def f(h0T):
        st2 = set_leaf(st, "eq.vol.core.h", lambda h: h.at[0, : T + 1].set(h0T))
        return spec.log_joint(st2, data, s)

    H = np.asarray(jax.jit(jax.hessian(f))(st.eq.vol.core.h[0, : T + 1]))
    np.testing.assert_allclose(-H, Q, rtol=1e-9, atol=1e-9 * np.abs(Q).max())
    del mu

    def g(pad):
        st2 = set_leaf(st, "eq.vol.core.h", lambda h: h.at[0, T + 1 :].set(pad))
        return spec.log_joint(st2, data, s)

    grad = np.asarray(jax.grad(g)(st.eq.vol.core.h[0, T + 1 :]))
    assert np.all(grad == 0.0)


def test_transitory_conditionals():
    """``o_t``, ``p`` and ``τ_t`` conditionals of N03 §4.1, §4.2, §4.4 as density ratios of the
    reference joint (Constant core: no mixture approximation anywhere; N03 §4.5)."""
    spec, st, data = setup("constant_transitory", "full", seed=9)
    T, K = spec.fixed.T_eff, spec.static.vol.outlier_K
    tr = _np(st.eq.vol.trans)
    lam = math.exp(float(st.eq.vol.core.log_lam[0]))
    e = numpy_residuals(spec, st, data)[0]
    z2 = e**2 / lam
    nu = float(J.TEST_NU_GRID[int(tr.nu_idx[0])])
    c = nu - 2.0
    log_p, log1m_p = float(tr.log_p[0]), float(tr.log1m_p[0])
    t = 3
    tau = math.exp(float(tr.log_tau[0, t]))
    # o_t | e, h, τ, p: logits log w_k − log k − z²/(2k²τ) (the 1/k factor included)
    ks = np.arange(1, K + 1)
    logw = np.where(ks == 1, log1m_p, log_p - math.log(K - 1))
    logits = logw - np.log(ks) - z2[t] / (2 * ks**2 * tau)
    lj = [float(spec.log_joint(set_leaf(st, "eq.vol.trans.o_idx", lambda o, k=k: o.at[0, t].set(k)), data))
          for k in range(K)]  # fmt: skip
    np.testing.assert_allclose(np.array(lj) - lj[0], logits - logits[0], rtol=1e-9, atol=1e-9)
    # p | o ~ Beta(a + N_1, b + T − N_1) (density of p)
    n1 = int(np.sum(tr.o_idx[0, :T] > 0))
    a, b = 2.0 + n1, 18.0 + T - n1

    def at_p(p):
        s2 = set_leaf(st, "eq.vol.trans.log_p", lambda x: x.at[0].set(math.log(p)))
        return float(
            spec.log_joint(
                set_leaf(s2, "eq.vol.trans.log1m_p", lambda x: x.at[0].set(math.log1p(-p))), data
            )
        )

    close(
        at_p(0.31) - at_p(0.04), stats.beta.logpdf(0.31, a, b) - stats.beta.logpdf(0.04, a, b), 1e-9
    )
    # τ_t | e, h, o, ν ~ IG((ν+1)/2, (c + z²/o²)/2) (density of τ)
    o = tr.o_idx[0, t] + 1.0
    sh_, sc_ = (nu + 1) / 2, (c + z2[t] / o**2) / 2

    def at_tau(x):
        return float(
            spec.log_joint(
                set_leaf(st, "eq.vol.trans.log_tau", lambda lt: lt.at[0, t].set(math.log(x))), data
            )
        )

    close(
        at_tau(2.2) - at_tau(0.4),
        stats.invgamma.logpdf(2.2, sh_, scale=sc_) - stats.invgamma.logpdf(0.4, sh_, scale=sc_),
        1e-9,
    )


@pytest.mark.parametrize(
    ("name", "kw"),
    [("structural_sv", {}), ("structural_sv_outliers", {}), ("structural_sv_t", {}),
     ("structural_sv_t", {"t_scale": "unit"}), ("structural_sv_outliers_t", {})],
)  # fmt: skip
def test_transitory_marginal_density_vs_quadrature(name, kw):
    """``log f(z)`` of N03 §2.2 for every transitory kind against direct mixtures over ``o`` of
    ``scipy.stats.norm`` / ``scipy.stats.t`` (scale ``√(c/ν)``), and its normalisation."""
    spec = J.joint_spec(name, **kw)
    vol = spec.static.vol
    st = spec.prior_sample(random.key(0))
    tr = jax.tree.map(lambda x: x[0], st.eq.vol.trans)
    z = np.array([-4.0, -0.3, 1e-3, 0.7, 6.0])
    if vol.transitory in ("t", "outlier_t"):
        nu = float(J.TEST_NU_GRID[int(tr.nu_idx)])
        c = nu - 2.0 if vol.t_scale == "standardized" else nu
        base = lambda x: stats.t.pdf(x, nu, scale=math.sqrt(c / nu))  # noqa: E731
    else:
        base = stats.norm.pdf
    if vol.transitory in ("outlier", "outlier_t"):
        p, K = math.exp(float(tr.log_p)), vol.outlier_K
        ks = np.arange(1, K + 1)
        w = np.where(ks == 1, 1 - p, p / (K - 1))
        dens = np.sum(w / ks * base(z[:, None] / ks), axis=1)
    else:
        dens = base(z)
    got = np.asarray(R.transitory_log_density(jnp.log(jnp.asarray(z) ** 2), tr, spec.static))
    np.testing.assert_allclose(got, np.log(dens), rtol=1e-11)
    grid = np.linspace(-400, 400, 400001)
    f = np.exp(
        np.asarray(R.transitory_log_density(jnp.log(jnp.asarray(grid) ** 2), tr, spec.static))
    )
    assert np.trapezoid(f, grid) == pytest.approx(1.0, abs=2e-3)


def test_references_never_import_kernels():
    """Two-implementation rule (DESIGN §9.1, D-22): the reference modules import no kernel module
    (``components/*``, ``engines/mcmc/{linalg,random}``) and no engine assembly (WP14), which
    ``JointSpec.step`` loads lazily."""
    code = (
        "import sys\n"
        "import rai_bvar.validation.joint, rai_bvar.validation.log_joint, rai_bvar.validation.testfns\n"
        "bad = [m for m in sys.modules if m.startswith(('rai_bvar.components.structure', "
        "'rai_bvar.components.mean', 'rai_bvar.components.coef', 'rai_bvar.components.vol')) "
        "or m in ('rai_bvar.engines.mcmc.linalg', 'rai_bvar.engines.mcmc.random', "
        "'rai_bvar.engines.mcmc.compose', 'rai_bvar.engines.mcmc.sweep', 'rai_bvar.engines.mcmc.runner')]\n"
        "print(bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
