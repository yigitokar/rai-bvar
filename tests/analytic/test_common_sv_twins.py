"""Analytic twins and exact oracles of ``CommonSV``, A8 (``n = 3``) and A11 (P3 DESIGN §9.7; WP32).

**A8.** The conjugate prior × ``CommonSV`` on its shared-factor path (§5.3) against the same
composition with the weighted path forced (row 323's configuration, ``const_path="weighted"``;
engine-level ``StaticConfig``s and WP30's arrays, ``n = 3, p = 2, T = 60``, SOC/SUR rows): the
quantiles 5/50/95% of every ``B`` entry, the base ``Σ = A⁻¹ΛA⁻ᵀ`` (lower triangle), ``h_t`` at the
first, middle and last data rows, ``ρ`` and ``σ_h``, under the Phase 2 parity rule
``|z| ≤ max(4, z_{1−0.01/(2m)})`` (the CDF form: ``F̂_shared(x̂_q) − F̂_weighted(x̂_q)`` over the
combined indicator MCSE, ``x̂_q`` the weighted sample quantile).

**A11** (N08 §A11 items 1 and 3):

* the stationary distribution of block configuration 408 (``conjugate_niw_only``: the
  shared-factor ``(θ_i, λ_i)`` draw and the coupled common core with ``h`` held) against the
  Phase 1 NIW posterior on rows rescaled by ``e^{−h_t/2}`` (Phase 1's ``ConjugateProblem`` closed
  form, ``row_scales = e^{h/2}``, the SOC/SUR rows unscaled): quantiles of ``B`` and ``Σ``, the
  Phase 2 parity rule;
* ``log p(Y | h)`` = Phase 1 log ML on the rescaled rows ``− (n/2)Σh`` (1e-10): our side is the
  basic marginal-likelihood identity ``log p(Y | θ, h) + log p(θ) − log p(θ | Y, h)`` at a prior
  state, with WP30's likelihood and prior and the exact posterior of the structural parameters
  given ``h`` from the production conditional (``linear_conditional`` × ``λ_i``, the IG marginal of
  ``λ_i`` with ``θ_i`` integrated: shape ``c_i + (T + m)/2``);
* the ``h`` posterior for ``T ∈ {2, 3}``, ``n = 5`` at fixed ``(q, ρ, σ_h)`` by tensor quadrature of
  ``exp ℓ(h)`` against 10⁵ draws of the production blocked MH (thinned to their ESS), a KS test
  per marginal; D-60's family-wise rule over the ``m = 5`` marginals: pass iff ``min p > 1e-3/m``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, random
from scipy import stats
from scipy.special import gammaln

from rai_bvar import diagnostics
from rai_bvar.components.vol import common as CS
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.runner import _prepare_fn
from rai_bvar.settings import MCMCSettings, Runtime
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation.joint import joint_spec, reduced_form_coefficients
from rai_bvar.validation.reference import reference

KEY = random.key(808)
Q = np.array([0.05, 0.5, 0.95])


def parity_z(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """``z`` of the Phase 2 parity rule in its CDF form for every quantity (last axis) and every
    quantile of :data:`Q`: ``x`` are ``(chains, draws, m)`` samples of the two sides."""
    z = []
    for q in Q:
        point = np.quantile(x_b.reshape(-1, x_b.shape[-1]), q, axis=0)
        ind_a = (x_a <= point).astype(np.float64)
        ind_b = (x_b <= point).astype(np.float64)
        var = diagnostics.mcse_mean_many(ind_a) ** 2 + diagnostics.mcse_mean_many(ind_b) ** 2
        z.append((ind_a.mean(axis=(0, 1)) - ind_b.mean(axis=(0, 1))) / np.sqrt(var))
    return np.concatenate(z)


def critical(m: int) -> float:
    return max(4.0, float(stats.norm.isf(0.01 / (2.0 * m))))


def base_sigma(a: np.ndarray, log_lam: np.ndarray) -> np.ndarray:
    """``Σ = A⁻¹ diag(λ) A⁻ᵀ`` for ``a`` (…, n, n−1) and ``log λ`` (…, n)."""
    n = a.shape[-2]
    A = np.broadcast_to(np.eye(n), (*a.shape[:-2], n, n)).copy()
    rows, cols = np.tril_indices(n, -1)
    A[..., rows, cols] = a[..., rows, cols]
    Ainv = np.linalg.inv(A)
    return Ainv @ (np.exp(log_lam)[..., :, None] * np.swapaxes(Ainv, -1, -2))


def lower(S: np.ndarray) -> np.ndarray:
    r, c = np.tril_indices(S.shape[-1])
    return S[..., r, c]


# --------------------------------------------------------------------------- A8


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a8_shared_factor_path_equals_the_weighted_path():
    """A8 (module docstring): 4 × 4,000 draws each after 1,000, the same data and arrays."""
    spec_s = joint_spec("conjugate_commonsv_dummies", "full", T_eff=60)
    spec_w = joint_spec("conjugate_commonsv_dummies_weighted", "full", T_eff=60)
    assert spec_s.static.mean.const_path == "shared_factor"
    assert spec_w.static.mean.const_path == "weighted"
    st = spec_s.prior_sample(random.fold_in(KEY, 8))
    data = spec_s.simulate(random.fold_in(KEY, 9), st)
    settings = MCMCSettings(chains=4, draws=4000, burn=1000, seed=0, store_max=4000)
    n = spec_s.static.shapes.n
    meta = FitMeta(model=None, vardata=None, periods=None,
                   variables=tuple(f"y{i}" for i in range(n)), hyper={})  # fmt: skip
    fits = []
    for spec, seed in ((spec_s, 3), (spec_w, 4)):
        fits.append(fit_mcmc(spec.static, data, spec.prior, settings, Runtime(), seed=seed,
                             salt=1, meta=meta))  # fmt: skip

    def draws(fit):
        p = fit.param_draws
        B = np.asarray(fit.reduced_form_draws().values)  # (C, D, k, n)
        C, D = B.shape[:2]
        S = base_sigma(np.asarray(p.struct.a), np.asarray(p.vol.core.log_lam))
        h = np.stack([np.asarray(fit.monitor[f]) for f in ("common_h_0", "common_h_mid",
                                                             "common_h_T")], axis=-1)  # fmt: skip
        post = fit.posterior["posterior"]
        scal = np.stack([post["rho"].values, post["sigma"].values], axis=-1)
        return np.concatenate([B.reshape(C, D, -1), lower(S).reshape(C, D, -1),
                               h[:, fit.draw_index], scal], axis=-1)  # fmt: skip

    xs, xw = draws(fits[0]), draws(fits[1])
    z = parity_z(xs, xw)
    crit = critical(z.size)
    print(f"A8: m = {z.size}, max |z| = {np.max(np.abs(z)):.2f}, critical {crit:.2f}")
    assert np.all(np.abs(z) <= crit), (np.sort(np.abs(z))[-5:], crit)


# --------------------------------------------------------------------------- A11: the NIW twin


def _phase1(spec, data, h):
    """The Phase 1 closed form of the conjugate test prior on rows rescaled by ``s_t = e^{h_t/2}``:
    ``(log ML − n Σ log s, NIWPosterior)`` from ``ConjugateProblem._core`` (``_log_ml_small`` takes
    out the SOC/SUR rows' marginal, as ``ConjugateProblem.evaluate`` does)."""
    s_, prior = spec.static.shapes, spec.prior
    n, k = s_.n, s_.k
    T = int(np.asarray(data.T_eff))
    shape = np.asarray(prior.eq.vol_core.shape)
    d = 2.0 * shape + n - np.arange(1, n + 1)  # λ_i ~ IG((d − n + ι)/2, ψ_i/2)
    assert np.allclose(d, d[0])
    psi = 2.0 * np.asarray(prior.eq.vol_core.scale)
    omega = 1.0 / np.asarray(prior.eq.coef.prec0)[0]
    ns = SimpleNamespace(k=k, n=n, d=float(d[0]), b=np.asarray(prior.eq.coef.mean0).T)
    Y = np.asarray(data.Y)[:T]
    X = np.asarray(data.X)[:T]
    s = np.exp(0.5 * np.asarray(h)[:T])
    Yd, Xd = np.asarray(prior.shared.dummies.Y), np.asarray(prior.shared.dummies.X)
    Ys = np.vstack([Yd, Y / s[:, None]])
    Xs = np.vstack([Xd, X / s[:, None]])
    log_ml, post = ConjugateProblem._core(ns, Ys, Xs, omega, psi)
    if Yd.shape[0]:
        log_ml -= ConjugateProblem._log_ml_small(ns, Yd, Xd, omega, psi)
    return log_ml - n * np.log(s).sum(), post


def _chain(spec, st0, data, iters: int, chains: int):
    """``chains`` runs of block 408's kernel (``spec.step``: the production sweep with ``h`` held)
    from ``st0``: ``B`` and ``Σ`` every iteration."""

    def one(ck):
        def body(st, it):
            st = spec.step(ck, it.astype(jnp.uint32), st, data)
            B = reduced_form_coefficients(st, spec.prior, spec.static)
            return st, (B, st.eq.struct.a, st.eq.vol.core.log_lam)

        return lax.scan(body, st0, jnp.arange(iters))[1]

    keys = jax.vmap(lambda c: random.fold_in(KEY, 4080 + c))(jnp.arange(chains))
    B, a, ll = jax.jit(jax.vmap(one))(keys)
    return np.asarray(B), base_sigma(np.asarray(a), np.asarray(ll))


def test_a11_block_408_is_the_niw_posterior_given_h():
    """A11 item 1: block 408's stationary distribution (4 chains × 5,000 iterations after 500)
    against 20,000 Phase 1 NIW draws with ``row_scales = e^{h/2}``: quantiles of every ``B`` entry
    and the lower triangle of ``Σ``, the Phase 2 parity rule."""
    spec = joint_spec("conjugate_niw_only", "full")
    st0 = spec.prior_sample(random.fold_in(KEY, 11))
    data = spec.simulate(random.fold_in(KEY, 12), st0)
    B, S = _chain(spec, st0, data, 5500, 4)
    B, S = B[:, 500:], S[:, 500:]
    _, post = _phase1(spec, data, st0.glob.vol.h)
    Bn, chol = post.draw(20000, np.random.default_rng(4))
    Sn = chol @ np.swapaxes(chol, -1, -2)
    x_mc = np.concatenate([B.reshape(*B.shape[:2], -1), lower(S).reshape(*S.shape[:2], -1)], -1)
    x_nw = np.concatenate([Bn.reshape(1, 20000, -1), lower(Sn).reshape(1, 20000, -1)], -1)
    z = parity_z(x_mc, x_nw)
    crit = critical(z.size)
    print(f"A11 (408 vs NIW): m = {z.size}, max |z| = {np.max(np.abs(z)):.2f}, critical {crit:.2f}")
    assert np.all(np.abs(z) <= crit), (np.sort(np.abs(z))[-5:], crit)


def test_a11_marginal_likelihood_given_h():
    """A11 item 1: ``log p(Y | h)`` from the basic marginal-likelihood identity at three prior
    states equals the Phase 1 log ML on the rescaled rows minus ``(n/2)Σh`` (1e-10)."""
    spec = joint_spec("conjugate_niw_only", "full")
    program = compose(spec.static)
    prior, static = spec.prior, spec.static
    s_ = static.shapes
    n, k = s_.n, s_.k
    csv_ref = reference("common_sv", static)
    worst = 0.0
    for rep in range(3):
        st = spec.prior_sample(random.fold_in(KEY, 30 + rep))
        data = spec.simulate(random.fold_in(KEY, 40 + rep), st)
        T = int(np.asarray(data.T_eff))
        tm = jnp.asarray(data.tmask)
        cache = _prepare_fn(program)(data, prior)
        loglik = float(LJ.loglik(st, data, prior, static))
        lp = float(LJ.log_prior(st, prior, static, tm))
        lp -= float(csv_ref.log_prior_global(st.glob.vol, prior, static, tmask=tm))
        m = int(static.mean.n_dummy)
        ytil = program.structure.pre_residual(st.glob, data, None)
        log_post = 0.0
        for i in range(n):
            take = lambda t, i=i: jax.tree.map(lambda x: x[i], t)  # noqa: E731
            eq_i, prior_i = take(st.eq), take(prior.eq)
            ctx = program.structure.context(jnp.int32(i), ytil, eq_i, program.vol, data,
                                            st.glob.vol)  # fmt: skip
            cm = program.structure.contemp_moments(ctx, prior_i)
            pm = program.coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
            cond = program.mean.linear_conditional(ctx, pm, cm, cache, prior.shared)
            lam = float(np.exp(np.asarray(eq_i.vol.core.log_lam)))
            act = np.concatenate([np.ones(k, bool), np.arange(n - 1) < i])
            Kt = lam * np.asarray(cond.K)[np.ix_(act, act)]  # K̃_i: λ-free
            bt = lam * np.asarray(cond.b)[act]
            theta = np.concatenate([np.asarray(eq_i.mean.delta), np.asarray(eq_i.struct.a)])[act]
            mu = np.concatenate([np.asarray(pm.mean), np.asarray(cm.mean)])[act]
            Pt = np.concatenate([np.asarray(pm.prec), lam * np.asarray(cm.prec)])[act]
            w = np.asarray(ctx.common_w)[:T]
            y = np.asarray(ctx.target)[:T]
            yd = np.asarray(prior.shared.dummies.Y)[:, i]
            with np.errstate(all="ignore"):  # Accelerate sets spurious FP flags in LAPACK
                L = np.linalg.cholesky(Kt)
                mean = np.linalg.solve(Kt, bt)
            assert np.all(np.isfinite(L)) and np.all(np.isfinite(mean))
            resid_ss = y @ (w * y) + yd @ yd + mu @ (Pt * mu) - bt @ mean
            a_post = float(prior_i.vol_core.shape) + 0.5 * (T + m)
            b_post = float(prior_i.vol_core.scale) + 0.5 * resid_ss
            log_ig = (a_post * math.log(b_post) - gammaln(a_post) - (a_post + 1.0) * math.log(lam)
                      - b_post / lam)  # fmt: skip
            dlt = theta - mean
            q_i = act.sum()
            log_n = (-0.5 * q_i * math.log(2.0 * math.pi * lam) + np.log(np.diag(L)).sum()
                     - 0.5 * dlt @ Kt @ dlt / lam)  # fmt: skip
            log_post += log_ig + log_n
        ours = loglik + lp - log_post
        theirs, _ = _phase1(spec, data, st.glob.vol.h)
        err = abs(ours - theirs) / (1.0 + abs(theirs))
        worst = max(worst, err)
        print(f"A11 log p(Y | h): ours {ours:.10f}, Phase 1 {theirs:.10f}, scaled error {err:.1e}")
    assert worst <= 1e-10


# --------------------------------------------------------------------------- A11: h by quadrature


def _quadrature_marginals(lq, n, rho, s2, T, points):
    """Marginal CDFs of ``p(h | q, ρ, σ_h) ∝ exp ℓ(h)`` on a tensor grid (``points`` per axis over
    the Laplace mode ± 10 s.d.): ``[(grid_t, cdf_t)]`` per date."""
    H = np.eye(T) - rho * np.eye(T, k=-1)
    Sinv = np.full(T, 1.0 / s2)
    Sinv[0] = (1.0 - rho * rho) / s2
    Kh = H.T @ (Sinv[:, None] * H)
    h = lq - math.log(n)
    for _ in range(100):  # Newton on the concave ℓ (dense; T ≤ 3)
        e = 0.5 * np.exp(lq - h)
        g = -0.5 * n + e - Kh @ h
        h = h + np.linalg.solve(Kh + np.diag(e), g)
    sd = np.sqrt(np.diag(np.linalg.inv(Kh + np.diag(0.5 * np.exp(lq - h)))))
    axes = [np.linspace(h[t] - 10 * sd[t], h[t] + 10 * sd[t], points) for t in range(T)]
    grids = np.meshgrid(*axes, indexing="ij")
    X = np.stack([g.ravel() for g in grids], axis=-1)
    quad = np.einsum("ij,jk,ik->i", X, Kh, X)
    ell = -0.5 * n * X.sum(1) - 0.5 * np.exp(lq[None, :] - X).sum(1) - 0.5 * quad
    dens = np.exp(ell - ell.max()).reshape(grids[0].shape)
    out = []
    for t in range(T):
        marg = dens.sum(axis=tuple(j for j in range(T) if j != t))
        x = axes[t]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (marg[1:] + marg[:-1]) * np.diff(x))])
        out.append((x, cdf / cdf[-1]))
    return out


def _mh_draws(lq, n, rho, s2, T, draws: int, seed: int) -> np.ndarray:
    """``draws`` iterations of the production blocked MH (``CommonSVStep.h_step``, block 100, a
    random offset each sweep) at fixed ``(q, ρ, σ_h)``, ``T_pad = 8``."""
    T_pad = 8
    step = CS.CommonSVStep(VolStatic(persistent="constant", common=True), T_pad)
    lqp = jnp.asarray(np.concatenate([lq, np.zeros(T_pad - T)]))
    log_sigma = jnp.float64(0.5 * math.log(s2))

    def body(h, it):
        k = random.fold_in(random.key(seed), it)
        h, _, _, info = step.h_step(k, h, lqp, jnp.float64(rho), log_sigma, jnp.float64(n), T)
        return h, (h[:T], info.ok)

    _, (hs, ok) = jax.jit(lambda: lax.scan(body, jnp.zeros(T_pad), jnp.arange(draws)))()
    assert bool(jnp.all(ok))
    return np.asarray(hs)


@pytest.mark.parametrize("T", [2, 3])
def test_a11_h_posterior_against_tensor_quadrature(T):
    """A11 item 3 (module docstring): ``n = 5``, ``ρ = 0.9``, ``σ_h² = 0.1``, ``q_t`` drawn from
    the model at a stationary ``h``; 10⁵ MH draws thinned to their ESS against the quadrature
    marginals, KS per date. Each ``T`` is half of D-60's family (``m = 5`` tests in all): every
    ``p > 1e-3/5``."""
    n, rho, s2, m = 5, 0.9, 0.1, 5
    rng = np.random.default_rng(100 + T)
    h_true = np.zeros(T)
    h_true[0] = math.sqrt(s2 / (1 - rho**2)) * rng.standard_normal()
    for t in range(1, T):
        h_true[t] = rho * h_true[t - 1] + math.sqrt(s2) * rng.standard_normal()
    lq = h_true + np.log(rng.chisquare(n, T))
    marg = _quadrature_marginals(lq, n, rho, s2, T, 801 if T == 2 else 241)
    hs = _mh_draws(lq, n, rho, s2, T, 100_000, seed=T)
    ess = diagnostics.ess_bulk_many(hs[None])
    thin = math.ceil(hs.shape[0] / float(np.min(ess)))
    ps = []
    for t in range(T):
        x, cdf = marg[t]
        u = np.interp(hs[::thin, t], x, cdf)
        ps.append(float(stats.kstest(u, "uniform").pvalue))
    print(
        f"A11 h quadrature T = {T}: ESS {np.round(ess).astype(int)}, thin {thin}, "
        f"KS p {[f'{p:.3f}' for p in ps]} (family-wise bound {1e-3 / m:.0e}, m = {m})"
    )
    assert min(ps) > 1e-3 / m, ps


def test_a11_proposal_state_independence():
    """A11 (proposal state independence, as Tier U): the block proposal of the quadrature
    problem does not move with the current ``h``."""
    T_pad, T = 8, 3
    step = CS.CommonSVStep(VolStatic(persistent="constant", common=True), T_pad)
    lq = jnp.asarray([0.3, -0.2, 0.9, 0, 0, 0, 0, 0], dtype=jnp.float64)
    args = (lq, jnp.float64(0.9), jnp.float64(0.5 * math.log(0.1)), T, 0, 3)
    a = step.proposal(step.block_of(jnp.zeros(T_pad), *args), jnp.float64(5.0))
    b = step.proposal(step.block_of(jnp.asarray([2.0, -1, 3, 0, 0, 0, 0, 0.0]), *args),
                      jnp.float64(5.0))  # fmt: skip
    for f in ("mode", "diag", "off"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))
