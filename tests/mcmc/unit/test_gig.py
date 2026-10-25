"""The GIG sampler on the log scale and the Bernoulli-logit draw (P3 D-42, §3.5, §9.2 item 1).

``random.log_gig`` (Devroye 2014 §6, every term in logs) against the Tier U list of P3 §3.5,
with the multiplicity rule of the review erratum (M9, E117+; it replaces the per-cell KS rule of
§3.5/D-60): on the pinned grid (``λ ∈ {0, ±1e-8, ±1e-3, ±0.5, ±1, ±10, ±50}``, every ``a − ½``
and ``a − 1`` at the GK grid's first, middle and last points, ``½ − T_max/2`` for
``T_max ∈ {200, 600}``, crossed with ``log ω ∈ {−2000, …, 12}``; ``m`` cells, each with its own
key ``fold_in(key, cell)``, the ``−λ`` mirrors included) one family of Kolmogorov–Smirnov tests
(10⁴ draws per cell) of ``Y = log X − log η`` against the CDF of ``exp(λy − ω cosh y)`` from
numerical integration in log space around the mode, passing iff ``min p > 1e-3/m``; and
``E log X`` against ``log η + ∂_λ log K_λ(ω)`` (mpmath, 50 digits) within 4 Monte Carlo s.e.,
Bonferroni over the ``m`` compared moments (D-60: ``|z| ≤ max(4, z_{1−0.01/(2m)})``, the Phase 2
rule's form); no ``ok = False`` on the grid, mean and worst-case attempts reported; the gamma and
inverse-gamma limits; improper input as a status; ``vmap`` lanes equal unbatched calls (1e-15);
determinism by key. Beyond the list: the hat dominates ``f`` on every grid point.
``random.bernoulli_logit``: frequencies (``|z| ≤ 4``), extreme logits, one uniform per element.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from jax import random as jr
from scipy import special as sc
from scipy import stats

from rai_bvar.engines.mcmc import random as R

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = jr.key(20261025, impl="threefry2x32")
N_DRAWS = 10_000
KS_ALPHA = 1e-3  # family-wise (P3 D-60, §3.5; review M9: Bonferroni over a family of m tests)
MC_SE = 4.0  # E log X within 4 Monte Carlo standard errors (P3 §3.5)
MOMENT_FAMILY = 1e-2  # D-60 Bonferroni over compared moments, in the Phase 2 rule's form

# P3 §3.5's pinned grid. The GK grid has 1,000 points (no single middle), so both middle points
# (indices 499 and 500) are used.
_GK = np.linspace(0.001, 1.0, 1000)[[0, 499, 500, 999]]
GRID_LAMBDA = np.unique(
    np.concatenate(
        [
            [0.0, 1e-8, -1e-8, 1e-3, -1e-3, 0.5, -0.5, 1.0, -1.0, 10.0, -10.0, 50.0, -50.0],
            _GK - 0.5,
            _GK - 1.0,
            [0.5 - 200 / 2, 0.5 - 600 / 2],
        ]
    )
)
GRID_LOG_OMEGA = np.array([-2000.0, -1000.0, -745.0, -300.0, -50.0, -12.0, -5.0, 0.0, 5.0, 12.0])
GRID = [(float(lam), float(lw)) for lam in GRID_LAMBDA for lw in GRID_LOG_OMEGA]
LOG_ETA = (0.0, 3.7, -11.2)  # log η cycles over the grid points, so the reduction is exercised


def key(i: int):
    return jr.fold_in(KEY, i)


_ONE = jax.jit(R.log_gig_attempts)  # compiled once per module (eager calls retrace the loop)
_MANY = jax.jit(jax.vmap(R.log_gig_attempts))


# --------------------------------------------------------------------------- oracles


def _log_cosh(u: np.ndarray) -> np.ndarray:
    a = np.abs(u)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def _log_density(u, lam: float, log_omega: float):
    """``λu − ω cosh u`` (the log density of ``log Y``, ``Y ~ GIG(λ, ω)``, up to a constant)."""
    with np.errstate(over="ignore"):
        return lam * u - np.exp(log_omega + _log_cosh(np.asarray(u, dtype=np.float64)))


def _mode(lam: float, log_omega: float) -> float:
    """``asinh(λ/ω)``, formed from ``log|λ| − log ω``."""
    if lam == 0.0:
        return 0.0
    r = np.log(abs(lam)) - log_omega
    return float(np.sign(lam) * (np.log(2.0) + r if r > 20.0 else np.arcsinh(np.exp(r))))


def log_y_cdf_table(lam: float, log_omega: float, cut: float = 60.0, npts: int = 200_001):
    """CDF of ``log Y`` by trapezoidal integration of ``exp(g(u) − g(u*))`` on a uniform grid
    over the region where ``g(u) − g(u*) > −cut`` (edges by doubling, then bisection)."""
    us = _mode(lam, log_omega)
    gs = _log_density(us, lam, log_omega)

    def edge(direction: float) -> float:
        step = 1e-6
        while _log_density(us + direction * step, lam, log_omega) - gs > -cut:
            step *= 2.0
        lo, hi = step / 2.0, step
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _log_density(us + direction * mid, lam, log_omega) - gs > -cut:
                lo = mid
            else:
                hi = mid
        return us + direction * hi

    u = np.linspace(edge(-1.0), edge(1.0), npts)
    dens = np.exp(_log_density(u, lam, log_omega) - gs)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u))])
    return u, c / c[-1]


def expected_log_y(lam: float, log_omega: float) -> float:
    """``E log Y = ∂_λ log K_λ(ω)`` (mpmath, 50 digits)."""
    with mpmath.workdps(50):
        omega = mpmath.exp(log_omega)
        return float(mpmath.diff(lambda v: mpmath.log(mpmath.besselk(v, omega)), mpmath.mpf(lam)))


# --------------------------------------------------------------------------- the pinned grid


@pytest.fixture(scope="module")
def grid_draws():
    """``N_DRAWS`` draws at every grid cell: ``log ψ = log ω − log η``, ``log χ = log ω + log η``
    with ``log η`` cycling over ``LOG_ETA``. Cell ``i`` has its own key ``fold_in(key(1), i)``
    (the ``−λ`` mirror cells too: step 2's sign flip would make same-key mirrors identical) and
    lane ``j`` uses ``fold_in(cell key, j)``. Returns ``Y = log X − log η``, ``ok``, attempts and
    the sampling time."""
    lanes = jnp.arange(N_DRAWS, dtype=jnp.uint32)
    Y, OK, ATT = [], [], []
    seconds = 0.0
    for i, (lam, lw) in enumerate(GRID):
        le = LOG_ETA[i % len(LOG_ETA)]
        cell_key = jr.fold_in(key(1), i)
        keys = jax.vmap(lambda c, ck=cell_key: jr.fold_in(ck, c))(lanes)
        args = (jnp.full((N_DRAWS,), v) for v in (lam, lw - le, lw + le))
        t0 = time.perf_counter()
        log_x, ok, att = jax.block_until_ready(_MANY(keys, *args))
        seconds += time.perf_counter() - t0
        Y.append(np.asarray(log_x) - le)
        OK.append(np.asarray(ok))
        ATT.append(np.asarray(att))
    return np.stack(Y), np.stack(OK), np.stack(ATT), seconds


def test_grid_is_the_pinned_one():
    lam = set(GRID_LAMBDA.tolist())
    for v in (0.0, 1e-8, -1e-8, 1e-3, -1e-3, 0.5, -0.5, 1.0, -1.0, 10.0, -10.0, 50.0, -50.0):
        assert v in lam
    assert {-99.5, -299.5} <= lam
    assert np.isclose(_GK, [0.001, 0.5, 0.501, 1.0]).all()
    assert min(GRID_LOG_OMEGA) == -2000.0 and max(GRID_LOG_OMEGA) == 12.0
    assert len(GRID) == GRID_LAMBDA.size * GRID_LOG_OMEGA.size


def test_no_failures_on_the_grid_and_attempts_reported(grid_draws, capsys):
    """No ``ok = False`` anywhere on the grid; mean and worst-case attempts are reported (P3
    §3.5; Theorem 2 bounds the expected attempts by 3.4597, R2a measured a worst rejection
    constant of 1.40)."""
    Y, ok, att, seconds = grid_draws
    assert ok.all(), [GRID[i] for i in np.flatnonzero(~ok.all(axis=1))]
    assert np.isfinite(Y).all()
    assert (att >= 1).all() and (att <= R.GIG_MAX_ATTEMPTS).all()
    means = att.mean(axis=1)
    worst = int(np.argmax(means))
    with capsys.disabled():
        print(
            f"\n  log_gig on the P3 §3.5 grid: {len(GRID)} points × {N_DRAWS} draws in "
            f"{seconds:.1f} s ({1e3 * seconds / len(GRID):.1f} ms per {N_DRAWS}-lane call); "
            f"attempts mean {att.mean():.4f}, worst-point mean "
            f"{means[worst]:.4f} at (λ, log ω) = {GRID[worst]}, worst single draw {att.max()}"
        )
    assert means.max() <= 3.4597  # Theorem 2's uniform bound (sampled: a sanity check)


def ks_family_pvalues(Y: np.ndarray) -> np.ndarray:
    """The KS p-value of every grid cell (``Y = log X − log η`` against the numerical CDF)."""
    out = []
    for i, (lam, lw) in enumerate(GRID):
        u, c = log_y_cdf_table(lam, lw)
        out.append(stats.kstest(Y[i], lambda y, u=u, c=c: np.interp(y, u, c)).pvalue)
    return np.asarray(out)


def moment_z(Y: np.ndarray) -> np.ndarray:
    """``(mean log Y − ∂_λ log K_λ(ω))/s.e.`` per grid cell."""
    out = []
    for i, (lam, lw) in enumerate(GRID):
        se = Y[i].std(ddof=1) / np.sqrt(Y.shape[1])
        out.append((Y[i].mean() - expected_log_y(lam, lw)) / se)
    return np.asarray(out)


def test_ks_of_log_y_against_the_numerical_cdf(grid_draws):
    """One family of ``m = len(GRID)`` (180) KS tests, 10⁴ draws per cell, of
    ``Y = log X − log η`` against the CDF of ``exp(λy − ω cosh y)`` integrated numerically in log
    space: passes iff ``min p > 1e-3/m`` (P3 §3.5 with the review's multiplicity rule M9)."""
    m = len(GRID)
    assert m == 180
    p = ks_family_pvalues(grid_draws[0])
    worst = np.argsort(p)[:3]
    assert p.min() > KS_ALPHA / m, [(GRID[i], float(p[i])) for i in worst]


def test_mean_log_x_against_the_bessel_derivative(grid_draws):
    """``E log X = log η + ∂_λ log K_λ(ω)`` (mpmath, 50 digits) within 4 Monte Carlo s.e. at
    every one of the ``m = 180`` cells, Bonferroni over the compared moments (D-60):
    ``|z| ≤ max(4, z_{1−0.01/(2m)})`` (P3 §3.5)."""
    m = len(GRID)
    crit = max(MC_SE, float(stats.norm.isf(MOMENT_FAMILY / (2.0 * m))))
    z = moment_z(grid_draws[0])
    bad = np.flatnonzero(~(np.abs(z) <= crit))
    assert bad.size == 0, (crit, [(GRID[i], float(z[i])) for i in bad])


def test_hat_dominates_f_on_the_grid():
    """``f(x) ≤ log hat(x)`` on a fine grid of ``x`` for every ``(|λ|, log ω)`` of the grid
    (the envelope is the tangent construction, never a secant; P3 §3.5, mutant M34), and the
    hat's pieces are ordered ``−s' ≤ 0 ≤ t'`` with positive masses."""
    pairs = [(lam, lw) for lam in np.unique(np.abs(GRID_LAMBDA)) for lw in GRID_LOG_OMEGA]
    lam = jnp.asarray([p[0] for p in pairs])
    lw = jnp.asarray([p[1] for p in pairs])
    h = jax.tree.map(np.asarray, jax.jit(jax.vmap(R.gig_hat))(lam, lw))
    assert (h.tp >= 0.0).all() and (h.sp >= 0.0).all() and (h.p > 0).all() and (h.r > 0).all()
    grid = np.linspace(0.0, 1.0, 2001)
    x = np.concatenate(
        [
            (-h.sp - 40.0 * h.p)[:, None] + (40.0 * h.p)[:, None] * grid,
            -h.sp[:, None] + (h.sp + h.tp)[:, None] * grid,
            h.tp[:, None] + (40.0 * h.r)[:, None] * grid,
        ],
        axis=1,
    )
    log_neg_f = jax.jit(jax.vmap(R._gig_log_neg_f))(
        jnp.asarray(x), h.log_alpha, h.log_lam, jnp.asarray(h.lam > 0.0)
    )
    with np.errstate(over="ignore"):
        f = -np.exp(np.asarray(log_neg_f))
    log_hat = np.where(
        x > h.tp[:, None],
        -h.eta[:, None] - h.zeta[:, None] * (x - h.t[:, None]),
        np.where(x < -h.sp[:, None], -h.theta[:, None] + h.xi[:, None] * (x + h.s[:, None]), 0.0),
    )
    excess = f - log_hat - 1e-12 * (1.0 + np.abs(f))
    bad = np.flatnonzero((excess > 0.0).any(axis=1))
    assert bad.size == 0, [(pairs[i], float(excess[i].max())) for i in bad]


# --------------------------------------------------------------------------- limits, status


def log_gamma_cdf(x: np.ndarray, a: float) -> np.ndarray:
    """CDF of ``log G``, ``G ~ Gamma(a, 1)``: ``P(a, eˣ)``, and ``exp(a x − lnΓ(a + 1))`` (the
    series' leading term, exact to rounding) where ``eˣ < 1e-300`` would underflow."""
    x = np.asarray(x, dtype=np.float64)
    tiny = x < np.log(1e-300)
    with np.errstate(under="ignore"):
        direct = sc.gammainc(a, np.exp(np.where(tiny, 0.0, x)))
    return np.where(tiny, np.exp(a * np.where(tiny, x, 0.0) - sc.gammaln(a + 1.0)), direct)


@pytest.mark.parametrize("lam", [1e-3, 0.5, 3.0, 50.0])
def test_gamma_limit(lam):
    """``log χ = −∞``, ``λ > 0``: ``log X = LG(λ) − log(ψ/2)`` exactly (KS ``α = 1e-3`` of
    ``log X + log(ψ/2)`` against ``log Gamma(λ, 1)``; no rejection attempts)."""
    log_psi = 1.3
    keys = jr.split(key(10), N_DRAWS)
    full = (jnp.full((N_DRAWS,), v) for v in (lam, log_psi, -jnp.inf))
    lx, ok, att = _MANY(keys, *full)
    assert np.asarray(ok).all() and (np.asarray(att) == 0).all()
    g = np.asarray(lx) + (log_psi - np.log(2.0))
    p = stats.kstest(g, lambda y: log_gamma_cdf(y, lam)).pvalue
    assert p > KS_ALPHA, p  # m = 1 KS test per test function


@pytest.mark.parametrize("lam", [-1e-3, -0.5, -3.0, -50.0])
def test_inverse_gamma_limit(lam):
    """``log ψ = −∞``, ``λ < 0``: ``log X = log(χ/2) − LG(−λ)`` exactly (KS ``α = 1e-3``)."""
    log_chi = -0.7
    keys = jr.split(key(11), N_DRAWS)
    full = (jnp.full((N_DRAWS,), v) for v in (lam, -jnp.inf, log_chi))
    lx, ok, att = _MANY(keys, *full)
    assert np.asarray(ok).all() and (np.asarray(att) == 0).all()
    g = (log_chi - np.log(2.0)) - np.asarray(lx)
    p = stats.kstest(g, lambda y: log_gamma_cdf(y, -lam)).pvalue
    assert p > KS_ALPHA, p  # m = 1 KS test per test function


IMPROPER = [
    (0.0, 0.0, -np.inf),  # χ = 0 needs λ > 0
    (-1.0, 0.0, -np.inf),
    (0.0, -np.inf, 0.0),  # ψ = 0 needs λ < 0
    (2.0, -np.inf, 0.0),
    (1.0, -np.inf, -np.inf),
    (1.0, np.inf, 0.0),
    (-1.0, 0.0, np.inf),
    (np.inf, 0.0, 0.0),
    (np.nan, 0.0, 0.0),
    (1.0, np.nan, 0.0),
    (1.0, 0.0, np.nan),
]
REGULAR = [
    (0.0, -2000.0, -2000.0),
    (-299.5, 12.0, 12.0),
    (50.0, -2000.0, 0.0),
    (1e-8, -745.0, 3.0),
    (-0.499, -5.0, 2.0),
    (10.0, 5.0, -5.0),
    (0.5, 2.0, -np.inf),  # gamma limit
    (-3.0, -np.inf, 1.0),  # inverse-gamma limit
    (-1.0, 0.0, 0.0),
    (1e-3, -300.0, -301.0),
    (0.3, 0.0, 0.0),
]


@pytest.fixture(scope="module")
def mixed_batch():
    """One ``vmap`` over improper and regular lanes (one compilation, shared by the tests)."""
    rows = IMPROPER + REGULAR
    lam, lp, lc = (jnp.asarray([row[c] for row in rows]) for c in range(3))
    keys = jr.split(key(12), len(rows))
    lx, ok, att = _MANY(keys, lam, lp, lc)
    return keys, (lam, lp, lc), np.asarray(lx), np.asarray(ok), np.asarray(att)


def test_improper_input_is_a_status_not_an_endless_loop(mixed_batch):
    """Every improper input gives ``ok = False`` and ``log x = NaN`` without a rejection attempt
    (P3 §3.5 step 1; P2 D-18); the regular lanes batched with them succeed."""
    _, _, lx, ok, att = mixed_batch
    m = len(IMPROPER)
    assert not ok[:m].any() and np.isnan(lx[:m]).all() and (att[:m] == 0).all()
    assert ok[m:].all() and np.isfinite(lx[m:]).all()


def test_vmap_lanes_equal_unbatched_calls(mixed_batch):
    """``vmap`` lanes equal separate unbatched calls with the same keys (1e-15 relative), with
    identical ``ok`` and attempts, whatever the other lanes hold (P3 §3.5)."""
    keys, (lam, lp, lc), lx, ok, att = mixed_batch
    for i in range(len(IMPROPER), lam.shape[0]):
        a, b, c = _ONE(keys[i], lam[i], lp[i], lc[i])
        assert bool(b) == bool(ok[i]) and int(c) == int(att[i])
        np.testing.assert_allclose(float(lx[i]), float(a), rtol=1e-15, atol=0.0)


def test_deterministic_by_key():
    lam, lp, lc = jnp.asarray(-0.5), jnp.asarray(-3.0), jnp.asarray(1.0)
    a = _ONE(key(14), lam, lp, lc)
    b = _ONE(key(14), lam, lp, lc)
    d = _ONE(key(15), lam, lp, lc)
    assert float(a[0]) == float(b[0]) and bool(a[1]) and int(a[2]) == int(b[2])
    assert float(d[0]) != float(a[0])
    assert a[0].dtype == jnp.float64 and a[1].dtype == jnp.bool_ and a[2].dtype == jnp.int32


# --------------------------------------------------------------------------- Bernoulli (SSVS)


@pytest.mark.parametrize("logit", [-30.0, -5.0, -1.0, 0.0, 0.7, 4.0, 12.0])
def test_bernoulli_logit_frequencies(logit):
    """``P(γ = 1) = σ(logit)``: binomial ``|z| ≤ 4`` on 2·10⁵ draws (P3 D-60)."""
    n = 200_000
    g = np.asarray(R.bernoulli_logit(key(20), jnp.full((n,), logit)))
    assert g.dtype == np.int32 and set(np.unique(g)) <= {0, 1}
    p = 1.0 / (1.0 + np.exp(-logit))
    se = np.sqrt(max(p * (1 - p), 1e-300) / n)
    if p * n < 1e-3:  # expected count of ones below 1e-3: none at all
        assert g.sum() == 0
    elif (1 - p) * n < 1e-3:
        assert g.sum() == n
    else:
        assert abs(g.mean() - p) <= 4.0 * se, (g.mean(), p)


def test_bernoulli_logit_extremes_and_one_uniform_per_element():
    """``±700`` and ``±inf`` are exact (``σ`` never exponentiated); the draw is
    ``1{log u < −softplus(−logit)}`` with ``u = uniform_tiny(key, shape)``; ``vmap`` lanes equal
    unbatched calls."""
    logit = jnp.asarray(np.random.default_rng(5).normal(0.0, 3.0, (7, 11)))
    ext = jnp.asarray([700.0, np.inf, -700.0, -np.inf] * 250)
    keys = jr.split(key(24), 7)

    @jax.jit
    def run():
        return (
            R.bernoulli_logit(key(21), ext),
            R.bernoulli_logit(key(23), logit),
            R.uniform_tiny(key(23), (7, 11)),
            jax.vmap(R.bernoulli_logit)(keys, logit),
            jnp.stack([R.bernoulli_logit(keys[i], logit[i]) for i in range(7)]),
        )

    e, g, u, lanes, single = (np.asarray(v) for v in run())
    assert e.reshape(250, 4)[:, :2].all() and not e.reshape(250, 4)[:, 2:].any()
    ref = np.log(u) < -np.logaddexp(0.0, -np.asarray(logit))
    assert g.dtype == np.int32 and g.shape == (7, 11) and np.array_equal(g, ref.astype(np.int32))
    assert np.array_equal(np.asarray(R.bernoulli_logit_core(u, logit)), g)
    assert np.array_equal(lanes, single)
