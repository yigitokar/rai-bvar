"""The whitened least-squares contemporaneous draw (``Recursive.contemp_core`` →
``linalg.lstsq_draw``; queue item 6 of the WP25 follow-ups).

``test_g_full[reduced_sv]`` had one chain of 4000 reach ``h ≈ 31–37`` in equation 0 (``|y| ≈ 2e8``),
where equation 2's ``a``-row precision ``K_a = diag(prec) + ZᵀWZ`` has exact eigenvalues 14.8 and
1.9e17: forming the Gram rounds its small eigenvalue away and the Cholesky pivot is 0 (status 1003).
The QR draw of ``[diag(√prec); √w ⊙ Z]`` never squares the condition number. Tests:

* it is the same function of the injected noise as ``precision_draw(K_a, b_a, z)``: draws agree
  within the DESIGN §9.3 path-vs-path envelope (``2e-12 + 5εκ(1 + t)`` in sd units) at 20 random
  states of every reduced composition, with the same ``ok`` and pivots;
* at an extreme-scale state the Cholesky path fails while the QR path matches the exact
  conditional (60-digit decimal arithmetic) to ``ε·√cond(K)``;
* an overridden ``contemp_conditional`` (mutant M14's seam) is still what the step draws from.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.linalg import lstsq_draw, precision_draw
from rai_bvar.engines.mcmc.state import GaussianPrec
from rai_bvar.validation.joint import joint_spec

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261027, impl="threefry2x32")
REDUCED = (
    "reduced_constant",
    "reduced_sv",
    "reduced_sv_outliers_t",
    "reduced_iw_equivalent",
    "reduced_constant_dummies",
)
EPS = np.finfo(np.float64).eps


def _eq(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def _contexts(spec, st, data):
    program = compose(spec.static)
    S = program.structure
    ytil = S.pre_residual(st.glob, data, program.mean.fitted_reduced(st.glob, data))
    out = []
    for i in range(1, spec.static.shapes.n):
        ctx = S.context(jnp.int32(i), ytil, _eq(st.eq, i), program.vol, data, st.glob.vol)
        out.append((i, ctx, S.contemp_moments(ctx, _eq(spec.prior.eq, i))))
    return S, out


@pytest.mark.parametrize("name", REDUCED)
def test_lstsq_draw_equals_the_cholesky_draw(name):
    spec = joint_spec(name)
    worst = 0.0
    for rep in range(20):
        k = random.fold_in(KEY, rep)
        st = spec.prior_sample(random.fold_in(k, 0))
        data = spec.simulate(random.fold_in(k, 1), st)
        S, ctxs = _contexts(spec, st, data)
        for i, ctx, cm in ctxs:
            g = jnp.zeros_like(ctx.target)
            z = random.normal(random.fold_in(k, 10 + i), cm.mean.shape, dtype=jnp.float64)
            a_qr, info_qr = S.contemp_lstsq_draw(z, ctx, g, cm)
            cond = S.contemp_conditional(ctx, g, cm)
            a_ch, info_ch = precision_draw(cond.K, cond.b, z)
            assert bool(info_qr.ok) and bool(info_ch.ok)
            K = np.asarray(cond.K)[:i, :i]
            s = 1.0 / np.sqrt(np.diag(K))
            kappa = np.linalg.cond(s[:, None] * K * s[None, :])
            cov = np.linalg.inv(K)
            sd = np.sqrt(np.diag(cov))
            t = np.max(np.abs(cov @ np.asarray(cond.b)[:i]) / sd)
            err = np.max(np.abs(np.asarray(a_qr - a_ch)[:i]) / sd)
            assert err <= 2e-12 + 5 * EPS * kappa * (1 + t), (rep, i, err, kappa, t)
            np.testing.assert_allclose(info_qr.min_pivot, info_ch.min_pivot, rtol=1e-8)
            worst = max(worst, err)
    assert worst < 1e-10


def test_extreme_scale_state():
    """A ~1e8-scale regressor and a near-copy of it (``u_1 ≈ −a u_0``) next to O(1) noise
    (``cond(K) > 1e15``, where forming the Gram leaves the small eigenvalue to rounding: the
    Cholesky pivot is then 0 or a few ε, depending on the rounding): the QR draw is exact to
    ``ε·√cond``."""
    rng = np.random.default_rng(7)
    T, m = 50, 2
    u0 = 1.5e8 * rng.standard_normal(T)
    u1 = -0.8 * u0 + rng.standard_normal(T)
    Z = -np.stack([u0, u1], axis=1)  # cols = −u_{<i}
    w = np.exp(-rng.uniform(-0.5, 0.5, T))
    target = Z @ np.array([0.3, -0.6]) + np.sqrt(1 / w) * rng.standard_normal(T)
    prec, mean = np.full(m, 2.0), np.zeros(m)
    K = np.diag(prec) + Z.T @ (w[:, None] * Z)
    b = prec * mean + Z.T @ (w * target)
    z = jnp.asarray(rng.standard_normal(m))
    del K, b
    A = np.vstack([np.diag(np.sqrt(prec)), np.sqrt(w)[:, None] * Z])
    rhs = np.concatenate([np.sqrt(prec) * mean, np.sqrt(w) * target])
    a0, info = lstsq_draw(jnp.asarray(A), jnp.asarray(rhs), jnp.zeros(m))
    a1, _ = lstsq_draw(jnp.asarray(A), jnp.asarray(rhs), z)
    assert bool(info.ok) and float(info.min_pivot) > 0.0
    getcontext().prec = 60
    D = Decimal
    Kx = [[(D(float(prec[r])) if r == c else D(0))
           + sum(D(float(w[t])) * D(float(Z[t, r])) * D(float(Z[t, c])) for t in range(T))
           for c in range(m)] for r in range(m)]  # fmt: skip
    bx = [D(float(prec[r] * mean[r])) + sum(D(float(w[t])) * D(float(Z[t, r])) * D(float(target[t]))
                                             for t in range(T)) for r in range(m)]  # fmt: skip
    det = Kx[0][0] * Kx[1][1] - Kx[0][1] * Kx[1][0]
    inv = [[Kx[1][1] / det, -Kx[0][1] / det], [-Kx[1][0] / det, Kx[0][0] / det]]
    mu = np.array([float(inv[r][0] * bx[0] + inv[r][1] * bx[1]) for r in range(m)])
    cov = np.array([[float(inv[r][c]) for c in range(m)] for r in range(m)])
    sd = np.sqrt(np.diag(cov))
    tr = Kx[0][0] + Kx[1][1]
    disc = (tr * tr - 4 * det).sqrt()
    cond = float((tr + disc) / (tr - disc))
    assert cond > 1e15
    tol = 50 * EPS * np.sqrt(cond)
    assert np.max(np.abs(np.asarray(a0) - mu) / sd) <= tol
    # the noise map: a1 − a0 = R⁻¹z has covariance K⁻¹, i.e. (a1 − a0)ᵀ K (a1 − a0) = zᵀz
    d = [D(float(x)) for x in np.asarray(a1 - a0)]  # dᵀKd in exact arithmetic (K is ~1e17)
    quad = sum(d[r] * Kx[r][c] * d[c] for r in range(m) for c in range(m))
    np.testing.assert_allclose(float(quad), float(z @ z), rtol=1e-6)


def test_overridden_conditional_is_still_drawn_from():
    """With ``contemp_conditional`` overridden (the M14 seam), ``contemp_core`` draws from the
    override's ``(K, b)`` through ``precision_draw``."""
    spec = joint_spec("reduced_constant")
    st = spec.prior_sample(random.fold_in(KEY, 99))
    data = spec.simulate(random.fold_in(KEY, 98), st)
    S, ctxs = _contexts(spec, st, data)
    _, ctx, cm = ctxs[-1]
    m = cm.mean.shape[0]

    def fixed(ctx, g, cm):
        return GaussianPrec(K=jnp.eye(m) * 4.0, b=jnp.full((m,), 2.0))

    So = S.with_overrides({"contemp_conditional": fixed})
    z = jnp.ones(m)
    new, _, _ = So.contemp_core(z, ctx, jnp.zeros_like(ctx.target), cm)
    want = np.where(np.asarray(cm.cmask) > 0, 0.5 + 0.5 * 1.0, 0.0)
    np.testing.assert_allclose(np.asarray(new.a), want, rtol=1e-14)


@pytest.mark.slow
def test_g_full_reduced_sv_failing_chain_is_fixed():
    """The ``test_g_full[reduced_sv]`` finding, replayed: chain 3214 of stage 0 (500-chain block
    3000–3499, the harness's keys), run with the previous Cholesky ``a``-row draw (forced through
    the ``contemp_conditional`` seam: the same operations, so the same trajectory), fails at sweep
    194 with status 1003 (pivot 0); the production sweep from the same state at sweep 194 succeeds
    and every draw is finite."""
    import zlib

    from jax import lax

    from rai_bvar.engines.mcmc.keys import Tier
    from rai_bvar.validation.geweke import Part, part, rep_keys

    name, seed, chain_ix, fail_it, block0 = "reduced_sv", 20261023, 3214, 194, 3000
    spec = joint_spec(name, "full")
    old = joint_spec(
        name, "full", kernel_overrides={"contemp_conditional": compose(spec.static).structure
                                        .contemp_conditional}
    )  # fmt: skip
    keys_b = rep_keys(seed, Tier.G, zlib.crc32(name.encode()) & 0xFFFFFFFF,
                      np.arange(block0, block0 + 500))  # fmt: skip

    def run_to(key):
        th0 = old.prior_sample(part(key, Part.PRIOR))
        sim_key, ckey = part(key, Part.SIM), part(key, Part.STEP)

        def body(th, it):
            return old.step(ckey, it, th, old.simulate(random.fold_in(sim_key, it), th)), None

        th, _ = lax.scan(body, th0, jnp.arange(fail_it, dtype=jnp.uint32))
        y = old.simulate(random.fold_in(sim_key, fail_it), th)
        it = jnp.uint32(fail_it)
        return th, old.step(ckey, it, th, y), spec.step(ckey, it, th, y)

    pre, with_old, with_new = jax.jit(jax.vmap(run_to))(keys_b)
    j = chain_ix - block0
    assert int(pre.status.code[j]) == 0
    assert float(np.max(np.asarray(pre.eq.vol.core.h[j]))) > 30.0  # the extreme-tail state
    assert int(with_old.status.code[j]) == 1003  # EQ_CONTEMP, equation 2: Cholesky pivot 0
    assert int(with_new.status.code[j]) == 0
    assert all(bool(jnp.all(jnp.isfinite(x[j]))) for x in jax.tree.leaves(with_new.eq))
