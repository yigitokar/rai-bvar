"""The shared-factor path of the conjugate ``CommonSV`` (``linalg.shared_factor_*``; P3 D-47, §5.3,
§9.2 item 1).

The shared-factor draw of every equation against the weighted path (P2's Mode II
``precision_draw`` with the λ-scaled prior, weights ``m_t/(λ_i e^{h_t})`` and dummy rows of weight
``1/λ_i``) on the same noise, with the conditioning-aware rule of P2 §9.3 item 1 (in posterior sd:
``2e-12 + 5εκ(1 + t)``); with and without SOC/SUR-type dummy rows, padded time rows and ``n`` up
to 7; the dummy rows weighted by ``e^{−h_0}`` (mutant M32) fail that comparison; the leading block
of the shared factor is the per-equation factor; ``vmap`` over equations with the factor
unbatched equals per-equation calls; the Gram against a dense oracle; inactive slots are 0;
failures as values.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines.mcmc import linalg as LA

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

EPS = float(np.finfo(np.float64).eps)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def problem(
    seed: int, n: int = 3, p: int = 2, T_eff: int = 40, T_pad: int = 48, n_dummy: int = 0
) -> SimpleNamespace:
    """A structural Mode II problem at ``cross = contemp = 1`` (P3 §5.3): the shared δ-design
    ``F = [X | X M_c − Y_{:, :n−1}]`` (``q = k + n − 1``, ``k = 1 + np``) with padded rows past
    ``T_eff``, a stationary AR(1) common log-volatility ``h`` (0 on padded rows), common weights
    ``w = m_t e^{−h_t}``, the λ-free shared prior precision ``P̃`` and per-equation means, dummy
    rows ``(F_d, Y_d)``, ``λ_i`` and noise. Equation ``i`` has the leading ``q_i = k + i`` slots."""
    r = rng(seed)
    k = 1 + n * p
    q = k + n - 1
    Y = r.normal(size=(T_pad, n))
    X = np.concatenate([np.ones((T_pad, 1)), r.normal(size=(T_pad, n * p))], axis=1)
    Mc = 0.3 * r.normal(size=(k, n - 1))
    F = np.concatenate([X, X @ Mc - Y[:, : n - 1]], axis=1)
    mt = (np.arange(T_pad) < T_eff).astype(np.float64)
    h = np.zeros(T_pad)
    h[0] = 0.4 * r.normal() / np.sqrt(1.0 - 0.9**2)
    for t in range(1, T_eff):
        h[t] = 0.9 * h[t - 1] + 0.4 * r.normal()
    w = mt * np.exp(-h)
    prec = np.exp(r.normal(0.0, 1.0, q))
    means = 0.5 * r.normal(size=(n, q))
    Fd = r.normal(size=(n_dummy, q)) if n_dummy else None
    yd = r.normal(size=(n_dummy, n)) if n_dummy else None
    lam = np.exp(r.normal(0.0, 0.5, n))
    z = r.normal(size=(n, q))
    cmask = (np.arange(q)[None, :] < (k + np.arange(n))[:, None]).astype(np.float64)
    return SimpleNamespace(
        n=n, k=k, q=q, Y=Y, F=F, h=h, w=w, mt=mt, prec=prec, means=means, Fd=Fd, yd=yd,
        lam=lam, z=z, cmask=cmask,
    )  # fmt: skip


@jax.jit
def _weighted_all(F, w, Fd, yd, Y, prec, means, cmask, lam, z):
    """The weighted path of every equation (``vmap``; see :func:`weighted_path`)."""
    q = F.shape[1]

    def one(y, ydi, c, lam_i, mean_i, z_i):
        wl = w / lam_i
        G = LA.weighted_gram(wl, F=F, mode="batched", q=q) + Fd.T @ Fd / lam_i
        fty = F.T @ (wl * y) + Fd.T @ ydi / lam_i
        pr_i = jnp.where(c > 0, prec / lam_i, 1.0)
        K = c[:, None] * G * c[None, :] + jnp.diag(pr_i)
        bb = pr_i * jnp.where(c > 0, mean_i, 0.0) + c * fty
        theta, info = LA.precision_draw(K, bb, z_i)
        return theta, K, bb, info

    return jax.vmap(one)(Y.T, yd.T, cmask, lam, means, z)


def weighted_path(pr: SimpleNamespace):
    """Every equation on the weighted path, as ``LinearStructural._conditional`` builds it in
    Mode II: Gram of the weights ``w/λ_i`` (``weighted_gram``, batched mode), dummy rows of weight
    ``1/λ_i``, prior precision ``P̃/λ_i`` on active slots (1 elsewhere); ``precision_draw`` on
    ``z_i``. Returns the draws, ``K``, ``b`` and the infos (leading axis: equation)."""
    theta, K, b, info = _weighted_all(*_arrays(pr))
    return np.asarray(theta), np.asarray(K), np.asarray(b), info


def _arrays(pr: SimpleNamespace, dummy_weight: float = 1.0):
    Fd = np.zeros((0, pr.q)) if pr.Fd is None else np.sqrt(dummy_weight) * pr.Fd
    yd = np.zeros((0, pr.n)) if pr.yd is None else np.sqrt(dummy_weight) * pr.yd
    return tuple(
        jnp.asarray(a)
        for a in (pr.F, pr.w, Fd, yd, pr.Y, pr.prec, pr.means, pr.cmask, pr.lam, pr.z)
    )


@jax.jit
def _shared_all(F, w, Fd, yd, Y, prec, means, cmask, lam, z):
    """Every equation through one shared factor (``vmap`` over equations; ``F``, ``w``, ``F_d``
    and the factor unbatched)."""
    G = LA.shared_factor_gram(w, F, Fd)
    L, s, info = LA.shared_factor_prepare(prec, G)

    def one(mean, y, ydi, c, lam_i, z_i):
        b = LA.shared_factor_rhs(prec, mean, F, w, y, c, Fd, ydi)
        return LA.shared_factor_draw(L, s, b, c, lam_i, z_i)

    theta, infos = jax.vmap(one)(means, Y.T, yd.T, cmask, lam, z)
    return theta, infos, (L, s, info)


def shared_path(pr: SimpleNamespace, *, dummy_weight: float = 1.0):
    """:func:`_shared_all` on ``pr``; ``dummy_weight`` scales the dummy rows (1 in production;
    mutant M32 uses ``e^{−h_0}``)."""
    theta, infos, factor = _shared_all(*_arrays(pr, dummy_weight))
    return np.asarray(theta), infos, factor


def sd_error(theta_a, theta_b, K, b, act):
    """Largest difference in posterior sd on the active block, and the rule's tolerance
    ``2e-12 + 5εκ(1 + t)`` (``κ = cond(sKs)``, ``t = max |E θ|/sd``; P2 §9.3 item 1)."""
    Ka = K[np.ix_(act, act)]
    C = np.linalg.inv(Ka)
    sd = np.sqrt(np.diag(C))
    s = 1.0 / np.sqrt(np.diag(Ka))
    kappa = float(np.linalg.cond(s[:, None] * Ka * s[None, :]))
    t = float(np.max(np.abs(C @ b[act]) / sd))
    err = float(np.max(np.abs(theta_a[act] - theta_b[act]) / sd))
    return err, 2e-12 + 5.0 * EPS * kappa * (1.0 + t)


# (seed, n, p, T_eff, T_pad, dummy rows)
PROBLEMS = [
    (1, 3, 2, 40, 48, 0),
    (2, 3, 2, 40, 48, 4),
    (3, 4, 1, 25, 32, 0),
    (4, 5, 2, 60, 64, 6),
    (5, 7, 2, 50, 50, 8),
    (6, 3, 4, 12, 16, 3),  # T_eff < q: the prior and the dummy rows carry the rank
]


@pytest.mark.parametrize("seed,n,p,T_eff,T_pad,n_dummy", PROBLEMS)
def test_shared_factor_equals_the_weighted_path_on_the_same_noise(
    seed, n, p, T_eff, T_pad, n_dummy
):
    """Every equation's shared-factor draw equals the weighted path's draw on the same noise
    within the conditioning-aware rule (P3 §5.3, §9.2 item 1); 0 on inactive slots."""
    pr = problem(seed, n, p, T_eff, T_pad, n_dummy)
    theta, infos, (_, _, info) = shared_path(pr)
    th_w, K, b, info_w = weighted_path(pr)
    assert bool(info.ok) and np.asarray(infos.ok).all() and np.asarray(info_w.ok).all()
    for i in range(n):
        act = pr.cmask[i] > 0
        err, tol = sd_error(theta[i], th_w[i], K[i], b[i], act)
        assert err <= tol, (i, err, tol)
        assert np.all(theta[i][~act] == 0.0)
    # each equation's pivots are D-14's pivots of its own K (the leading block)
    np.testing.assert_allclose(
        np.asarray(infos.min_pivot), np.asarray(info_w.min_pivot), rtol=1e-10, atol=1e-13
    )


def test_dummy_rows_weighted_by_the_first_volatility_fail_the_comparison():
    """Mutant M32 (dummy rows weighted by ``e^{−h_0}``, the first real row's weight) is caught by
    the comparison against the weighted path (P3 §9.6): the shared factor with production weights
    passes the rule, the mutant's does not."""
    pr = problem(2, 3, 2, 40, 48, 4)
    assert abs(pr.h[0]) > 0.1  # e^{−h_0} visibly different from 1
    good, _, _ = shared_path(pr)
    bad, _, _ = shared_path(pr, dummy_weight=float(np.exp(-pr.h[0])))
    th_w, K, b, _ = weighted_path(pr)
    for i in range(pr.n):
        act = pr.cmask[i] > 0
        err_good, tol = sd_error(good[i], th_w[i], K[i], b[i], act)
        err_bad, _ = sd_error(bad[i], th_w[i], K[i], b[i], act)
        assert err_good <= tol and err_bad > 1e3 * tol, (i, err_good, err_bad, tol)


def test_leading_block_of_the_shared_factor_is_the_equation_factor():
    """``K̃_i`` is the leading ``q_i × q_i`` block of ``K̃``, so its Jacobi scaling and Cholesky
    factor are the leading parts of the shared ones (to rounding, 1e-13)."""
    pr = problem(7, 4, 2, 40, 48, 3)
    G = LA.shared_factor_gram(*(jnp.asarray(a) for a in (pr.w, pr.F, pr.Fd)))
    L, s, _ = LA.shared_factor_prepare(jnp.asarray(pr.prec), G)
    Kt = np.diag(pr.prec) + np.asarray(G)
    for i in range(pr.n):
        qi = pr.k + i
        si = 1.0 / np.sqrt(np.diag(Kt[:qi, :qi]))
        Li = np.linalg.cholesky(si[:, None] * Kt[:qi, :qi] * si[None, :])
        np.testing.assert_allclose(np.asarray(L)[:qi, :qi], Li, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(np.asarray(s)[:qi], si, rtol=1e-15)


def test_vmap_over_equations_equals_per_equation_calls():
    """The per-equation right-hand side and draw, batched by ``vmap`` with the shared factor and
    ``F`` unbatched, equal separate calls (1e-14 of the largest entry: a batched triangular solve
    rounds differently from a single one)."""
    pr = problem(8, 4, 2, 30, 32, 2)
    theta, infos, (L, s, _) = shared_path(pr)
    F, w, prec = (jnp.asarray(a) for a in (pr.F, pr.w, pr.prec))
    Fd = jnp.asarray(pr.Fd)
    for i in range(pr.n):
        b = LA.shared_factor_rhs(
            prec, jnp.asarray(pr.means[i]), F, w, jnp.asarray(pr.Y[:, i]),
            jnp.asarray(pr.cmask[i]), Fd, jnp.asarray(pr.yd[:, i]),
        )  # fmt: skip
        th, info = LA.shared_factor_draw(
            L, s, b, jnp.asarray(pr.cmask[i]), jnp.asarray(pr.lam[i]), jnp.asarray(pr.z[i])
        )
        scale = np.max(np.abs(theta[i]))
        np.testing.assert_allclose(theta[i], np.asarray(th), rtol=0.0, atol=1e-14 * scale)
        assert bool(info.ok) == bool(infos.ok[i])


def test_shared_factor_gram_matches_dense():
    """``(√w ⊙ F)ᵀ(√w ⊙ F) + F_dᵀF_d`` against the dense ``Fᵀdiag(w)F + F_dᵀF_d`` (1e-13
    relative to the largest entry), exactly symmetric; padded rows (``w = 0``) never enter."""
    pr = problem(9, 5, 2, 44, 52, 5)
    G = np.asarray(LA.shared_factor_gram(*(jnp.asarray(a) for a in (pr.w, pr.F, pr.Fd))))
    ref = pr.F.T @ (pr.w[:, None] * pr.F) + pr.Fd.T @ pr.Fd
    assert np.max(np.abs(G - ref)) <= 1e-13 * np.max(np.abs(ref))
    assert np.array_equal(G, G.T)
    F2 = pr.F.copy()
    F2[pr.mt == 0] = 1e6  # padded rows hold anything finite
    G2 = np.asarray(LA.shared_factor_gram(*(jnp.asarray(a) for a in (pr.w, F2, pr.Fd))))
    np.testing.assert_array_equal(G2, G)
    G0 = np.asarray(LA.shared_factor_gram(jnp.asarray(pr.w), jnp.asarray(pr.F)))
    np.testing.assert_allclose(G0, G - pr.Fd.T @ pr.Fd, rtol=0, atol=1e-12 * np.max(np.abs(G)))


def test_failures_are_values_and_poison_every_equation():
    """A non-PD shared matrix (a negative prior precision) fails the factor and every
    equation's draw, without an exception (P2 D-18)."""
    pr = problem(10, 3, 2, 40, 48, 0)
    pr.prec[0] = -1e3
    theta, infos, (_, _, info) = shared_path(pr)
    assert not np.isfinite(theta).all()
    assert not bool(info.ok) and not np.asarray(infos.ok).any()
    with pytest.raises(ValueError, match="targets"):
        LA.shared_factor_rhs(
            *(jnp.asarray(a) for a in (pr.prec, pr.means[0], pr.F, pr.w, pr.Y[:, 0])),
            jnp.asarray(pr.cmask[0]), jnp.ones((2, pr.q)),
        )  # fmt: skip
