"""The deviation draw in log space, ``linalg.precision_draw_log`` (P3 D-41, §3.4, §9.2 item 1).

Against D-14's ``precision_draw`` on the same noise with the conditioning-aware path-vs-path rule
of P2 §9.3 item 1 (in posterior sd: ``2e-12 + 5εκ(1 + t)``, ``κ = cond(sKs)`` on the active
block, ``t`` the largest posterior t-ratio); against a dense mpmath oracle that takes log
precisions up to ``±1600`` (``d = K⁻¹b̃ + chol(K)⁻ᵀz`` at 50 digits, 1e-10 relative on
``dev_log``); ``dev_log`` equal to ``log|θ − mean|`` wherever that is representable (1e-12);
exact padding; failures as values; ``vmap`` lanes equal unbatched calls; float64 under ``jit``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

from rai_bvar.engines.mcmc import linalg as LA

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

EPS = float(np.finfo(np.float64).eps)
_DRAW = jax.jit(LA.precision_draw)
_DRAW_LOG = jax.jit(LA.precision_draw_log)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def problem(
    seed: int,
    m: int,
    T: int,
    *,
    q: int | None = None,
    log_prec_sd: float = 3.0,
    log_prec_mean: float = 0.0,
    collinear: float = 0.0,
    grade: float = 0.0,
    mean_sd: float = 2.0,
) -> SimpleNamespace:
    """A deviation-path problem with ``q ≤ m`` active slots (the rest padded): ``G`` the masked
    data Gram ``Xᵀdiag(w)X`` (columns scaled by ``10^{grade·U}``, column 1 a near-copy of column
    0 when ``collinear > 0``), log prior precisions ``N(0, log_prec_sd²)``, a prior mean, and the
    data right-hand side ``c ⊙ Xᵀ(w ⊙ y)`` in both forms: ``b = P ⊙ mean + data`` (D-14) and
    ``b̃ = data − G·mean`` (deviation coordinates)."""
    r = rng(seed)
    q = m if q is None else q
    X = r.normal(size=(T, m)) * 10.0 ** (grade * r.uniform(-1.0, 1.0, m))
    if collinear > 0.0 and m > 1:
        X[:, 1] = X[:, 0] + collinear * r.normal(size=T)
    w = np.exp(r.normal(0.0, 0.5, T))
    c = (np.arange(m) < q).astype(np.float64)
    G = np.outer(c, c) * (X.T @ (w[:, None] * X))
    G = np.tril(G) + np.tril(G, -1).T
    log_prec = np.where(c > 0, r.normal(log_prec_mean, log_prec_sd, m), 0.0)
    mean = np.where(c > 0, r.normal(0.0, mean_sd, m), 0.0)
    data = c * (X.T @ (w * (X @ r.normal(size=m) + r.normal(size=T))))
    z = r.normal(size=m)
    P = np.exp(log_prec)
    K = np.diag(P) + G
    return SimpleNamespace(
        G=G, log_prec=log_prec, mean=mean, data=data, z=z, c=c, q=q, K=K,
        b=P * mean + data, b_dev=data - G @ mean,
    )  # fmt: skip


def envelope(K: np.ndarray, b: np.ndarray, act: np.ndarray) -> SimpleNamespace:
    """Posterior sd, ``κ = cond(sKs)`` and ``t = max |E θ|/sd`` on the active block (float64:
    these only scale the tolerance)."""
    Ka = K[np.ix_(act, act)]
    C = np.linalg.inv(Ka)
    sd = np.sqrt(np.diag(C))
    s = 1.0 / np.sqrt(np.diag(Ka))
    kappa = float(np.linalg.cond(s[:, None] * Ka * s[None, :]))
    t = float(np.max(np.abs(C @ b[act]) / sd))
    return SimpleNamespace(sd=sd, kappa=kappa, t=t)


def path_tolerance(env: SimpleNamespace) -> float:
    """P2 §9.3 item 1's path-vs-path rule for the mean, in posterior sd."""
    return 2e-12 + 5.0 * EPS * env.kappa * (1.0 + env.t)


def j(*arrays):
    return tuple(jnp.asarray(a) for a in arrays)


# --------------------------------------------------------------------------- against D-14

# (seed, m, T, q, log-precision mean and sd, collinearity, grading): well conditioned, graded
# columns, near-collinear regressors, a near-flat prior on a collinear pair (κ ≈ 1e9), T < m (the
# prior carries the rank), padded slots, the reference q = 100.
D14_PROBLEMS = [
    (1, 3, 20, 3, 0.0, 1.0, 0.0, 0.0),
    (2, 8, 40, 8, 0.0, 3.0, 0.0, 0.0),
    (3, 8, 40, 6, 0.0, 3.0, 0.0, 0.0),
    (4, 12, 60, 12, 0.0, 4.0, 0.0, 1.5),
    (5, 12, 60, 12, 0.0, 2.0, 1e-3, 0.0),
    (6, 12, 60, 10, 0.0, 2.0, 1e-5, 1.0),
    (11, 6, 50, 6, -14.0, 0.5, 1e-4, 0.0),
    (7, 20, 10, 20, 0.0, 2.0, 0.0, 0.0),
    (8, 30, 200, 27, 0.0, 5.0, 0.0, 0.5),
    (9, 30, 200, 30, 0.0, 6.0, 1e-4, 1.0),
    (10, 100, 300, 100, 0.0, 3.0, 0.0, 0.5),
]


@pytest.mark.parametrize("seed,m,T,q,lpm,lps,col,grade", D14_PROBLEMS)
def test_matches_precision_draw_on_the_same_noise(seed, m, T, q, lpm, lps, col, grade):
    """``mean + d`` equals D-14's ``precision_draw(K, b, z)`` on the active slots within the
    conditioning-aware rule (P3 §3.4, §9.2 item 1; P2 §9.3 item 1): the same function of ``z``
    to rounding. Both report the same pivots (``sKs`` is ``A``)."""
    pr = problem(seed, m, T, q=q, log_prec_mean=lpm, log_prec_sd=lps, collinear=col, grade=grade)
    th, info = _DRAW(*j(pr.K, pr.b, pr.z))
    d, _, info_l = _DRAW_LOG(*j(pr.log_prec, pr.G, pr.b_dev, pr.z))
    assert bool(info.ok) and bool(info_l.ok)
    act = pr.c > 0
    env = envelope(pr.K, pr.b, act)
    err = np.max(np.abs((pr.mean + np.asarray(d))[act] - np.asarray(th)[act]) / env.sd)
    assert err <= path_tolerance(env), (err, env.kappa, env.t)
    np.testing.assert_allclose(float(info_l.min_pivot), float(info.min_pivot), rtol=1e-10)


# --------------------------------------------------------------------------- against mpmath


def mp_deviation_log(log_prec, G, b_dev, z, dps: int = 50) -> np.ndarray:
    """``log|d|``, ``d = K⁻¹b̃ + chol(K)⁻ᵀz``, ``K = diag(exp(log_prec)) + G`` built from the log
    precisions at ``dps`` digits (mpmath's exponent range is unbounded), by a plain Cholesky
    ``K = RRᵀ`` and triangular solves (mpmath's ``lu_solve`` rejects the dynamic range). In exact
    arithmetic ``s ⊙ L⁻ᵀ = chol(K)⁻ᵀ`` for ``LLᵀ = sKs``, so the oracle does not depend on the
    scaling."""
    m = len(log_prec)
    with mpmath.workdps(dps):
        K = [[mpmath.mpf(float(G[r, c])) for c in range(m)] for r in range(m)]
        for r in range(m):
            K[r][r] += mpmath.exp(mpmath.mpf(float(log_prec[r])))
        R = [[mpmath.mpf(0)] * m for _ in range(m)]
        for c in range(m):
            R[c][c] = mpmath.sqrt(K[c][c] - mpmath.fsum(R[c][t] ** 2 for t in range(c)))
            for r in range(c + 1, m):
                R[r][c] = (K[r][c] - mpmath.fsum(R[r][t] * R[c][t] for t in range(c))) / R[c][c]

        def fwd(v):  # R⁻¹v
            x = [mpmath.mpf(0)] * m
            for r in range(m):
                x[r] = (v[r] - mpmath.fsum(R[r][t] * x[t] for t in range(r))) / R[r][r]
            return x

        def bwd(v):  # R⁻ᵀv
            x = [mpmath.mpf(0)] * m
            for r in reversed(range(m)):
                x[r] = (v[r] - mpmath.fsum(R[t][r] * x[t] for t in range(r + 1, m))) / R[r][r]
            return x

        bv = [mpmath.mpf(float(v)) for v in b_dev]
        zv = [mpmath.mpf(float(v)) for v in z]
        mean = bwd(fwd(bv))
        noise = bwd(zv)
        return np.array([float(mpmath.log(abs(mean[r] + noise[r]))) for r in range(m)])


# log precisions up to ±1600 (a precision of e^{1600} overflows; a deviation of e^{−800} is below
# the smallest float), mixed with ordinary ones in the same problem
LOG_PREC_SETS = [
    [1600.0, -1600.0, 0.0, 3.0, -2.0, 800.0],
    [-1600.0, -800.0, -50.0, 50.0, 800.0, 1600.0],
    [1600.0, 1600.0, 1600.0, -3.0, 0.5, 2.0],
    [-1600.0, -1600.0, -1600.0, -1600.0, 0.0, 0.0],
    [745.0, 709.0, -745.0, -709.0, 20.0, -20.0],
]


@pytest.mark.parametrize("which", range(len(LOG_PREC_SETS)))
def test_dev_log_against_a_dense_mpmath_oracle(which):
    """``dev_log`` against the 50-digit dense oracle to 1e-10 relative, with log precisions up to
    ``±1600`` (P3 §3.4, §9.2 item 1): exact although ``d`` underflows to 0 and ``e^{1600}``
    overflows; ``info.ok``."""
    m, T = 6, 15
    pr = problem(100 + which, m, T)
    log_prec = np.asarray(LOG_PREC_SETS[which])
    d, dev_log, info = _DRAW_LOG(*j(log_prec, pr.G, pr.b_dev, pr.z))
    ref = mp_deviation_log(log_prec, pr.G, pr.b_dev, pr.z)
    assert bool(info.ok) and np.all(np.isfinite(np.asarray(dev_log)))
    np.testing.assert_allclose(np.asarray(dev_log), ref, rtol=1e-10, atol=0.0)
    if np.any(log_prec >= 1600.0):
        assert np.any(np.asarray(d) == 0.0)  # underflowed deviations, exact logs


# --------------------------------------------------------------------------- dev_log = log|θ − m|


def test_dev_log_equals_log_abs_deviation_where_representable():
    """``dev_log = log|θ − mean|`` (1e-12) wherever ``θ − mean`` is representable: exactly with
    ``mean = 0``; with a nonzero mean wherever ``θ = mean + d`` keeps ``d`` to better than 1e-13
    (``ε(|θ| + |d|)/|d| ≤ 1e-13``). Where it does not (``|d|`` below the resolution of ``m + d``)
    ``dev_log`` is still finite (P3 §3.4)."""
    checked = unrepresentable = 0
    for seed in range(20):
        pr = problem(200 + seed, 10, 50, log_prec_sd=6.0)
        d, dev_log, info = _DRAW_LOG(*j(pr.log_prec, pr.G, pr.b_dev, pr.z))
        d, dev_log = np.asarray(d), np.asarray(dev_log)
        assert bool(info.ok)
        # mean = 0: θ = d exactly
        normal = np.abs(d) >= np.finfo(np.float64).tiny
        assert np.all(np.abs(dev_log[normal] - np.log(np.abs(d[normal]))) <= 1e-12)
        # a nonzero mean (an own-lag mean of 1 and others): θ as the kernel forms it
        mean = np.where(np.arange(10) % 3 == 0, 1.0, 0.37 * (seed + 1))
        theta = mean + d
        dev = theta - mean
        rep = (dev != 0.0) & (EPS * (np.abs(theta) + np.abs(d)) <= 1e-13 * np.abs(d))
        assert np.all(np.abs(dev_log[rep] - np.log(np.abs(dev[rep]))) <= 1e-12)
        checked += int(rep.sum())
        unrepresentable += int((~rep).sum())
        assert np.all(np.isfinite(dev_log))
    assert checked > 100  # the comparison is not vacuous
    # tiny deviations next to a mean of 1 occur and keep a finite exact log
    pr = problem(300, 10, 50)
    lp = pr.log_prec.copy()
    lp[0] = 200.0  # prior s.d. e^{−100}: d ≈ e^{−100}·z, invisible in 1 + d
    d, dev_log, _ = _DRAW_LOG(*j(lp, pr.G, pr.b_dev, pr.z))
    assert 1.0 + float(d[0]) == 1.0 and np.isfinite(float(dev_log[0]))
    assert abs(float(dev_log[0]) - (-100.0 + np.log(abs(pr.z[0])))) < 1e-6


# --------------------------------------------------------------------------- padding, failures


def test_padding_is_exact():
    """Trailing inactive slots (log precision 0, zero Gram, ``b̃ = 0``) leave the active draw
    equal to the unpadded draw with ``z[:q]`` and return ``d = z`` there (P2 D-15; 1e-12)."""
    pr = problem(400, 12, 60, q=8)
    d, dev_log, info = _DRAW_LOG(*j(pr.log_prec, pr.G, pr.b_dev, pr.z))
    q = pr.q
    du, dlu, info_u = _DRAW_LOG(*j(pr.log_prec[:q], pr.G[:q, :q], pr.b_dev[:q], pr.z[:q]))
    assert bool(info.ok) and bool(info_u.ok)
    np.testing.assert_allclose(np.asarray(d)[:q], np.asarray(du), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(dev_log)[:q], np.asarray(dlu), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(d)[q:], pr.z[q:], rtol=1e-15, atol=0.0)


def test_failures_are_values_not_exceptions():
    pr = problem(500, 6, 30)
    bad_diag = pr.G.copy()
    bad_diag[2, 2] = -1.0  # negative Gram diagonal: log of a negative number
    indefinite = pr.G.copy()
    indefinite[0, 1] = indefinite[1, 0] = 10.0 * np.sqrt(pr.G[0, 0] * pr.G[1, 1])
    lp_small = pr.log_prec.copy()
    lp_small[:2] = -30.0  # the data dominate both slots, so A[0, 1] ≈ 10: not PD
    z_nan = pr.z.copy()
    z_nan[3] = np.nan
    for args in (
        (pr.log_prec, bad_diag, pr.b_dev, pr.z),
        (lp_small, indefinite, pr.b_dev, pr.z),
        (pr.log_prec, pr.G, pr.b_dev, z_nan),
        (np.full(6, -2000.0), np.zeros((6, 6)), np.zeros(6), pr.z),  # s = e^{1000} overflows
    ):
        _, _, info = _DRAW_LOG(*j(*args))
        assert not bool(info.ok)


# --------------------------------------------------------------------------- batching, dtypes


def test_vmap_lanes_equal_unbatched_calls_and_float64():
    probs = [problem(600 + i, 9, 40, q=9 - i, log_prec_sd=5.0) for i in range(4)]
    args = [jnp.asarray(np.stack([getattr(p, f) for p in probs])) for f in ("log_prec", "G")]
    args += [
        jnp.asarray(np.stack([p.b_dev for p in probs])),
        jnp.asarray(np.stack([p.z for p in probs])),
    ]
    d, dl, info = jax.jit(jax.vmap(LA.precision_draw_log))(*args)
    for i in range(4):
        di, dli, infoi = _DRAW_LOG(*(a[i] for a in args))
        np.testing.assert_allclose(np.asarray(d[i]), np.asarray(di), rtol=1e-14, atol=1e-300)
        np.testing.assert_allclose(np.asarray(dl[i]), np.asarray(dli), rtol=1e-14, atol=1e-14)
        assert bool(info.ok[i]) == bool(infoi.ok)
    assert d.dtype == dl.dtype == info.min_pivot.dtype == jnp.float64
    assert info.ok.dtype == jnp.bool_
