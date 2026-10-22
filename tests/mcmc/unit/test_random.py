"""Random-variate kernels (DESIGN §13 WP4; D-4; N07 §9, §13.1; pilot §5).

Inverse-CDF categorical against brute force and a χ² test with log weights shifted by ±700;
log-gamma, inverse-gamma and beta draws at shapes down to 1e-3 (KS and moments); the truncated
normal against ``scipy.stats.truncnorm`` and a log-space round trip in the far tails; ``vmap``
lanes equal unbatched calls; key discipline.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random as jr
from scipy import special as sc
from scipy import stats

from rai_bvar.engines.mcmc import random as R

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = jr.key(20261022, impl="threefry2x32")
N = 200_000


def key(i: int):
    return jr.fold_in(KEY, i)


# --------------------------------------------------------------------------- uniforms


def test_uniform_tiny_range_shape_and_stream():
    u = R.uniform_tiny(key(0), (100_000,))
    assert u.dtype == jnp.float64 and u.shape == (100_000,)
    assert float(u.min()) >= R.TINY and float(u.max()) < 1.0
    assert R.uniform_tiny(key(0), 5).shape == (5,) and R.uniform_tiny(key(0)).shape == ()
    ref = jr.uniform(key(0), (100_000,), dtype=jnp.float64, minval=np.finfo(np.float64).tiny)
    assert np.array_equal(np.asarray(u), np.asarray(ref))
    assert np.all(np.isfinite(np.log(np.asarray(u))))


def test_uniform_tiny_floors_the_zero_draw(monkeypatch):
    """The draw whose random bits give ``u = 0`` (probability 2⁻⁵² per draw, so no sample shows
    it) comes out as ``TINY``, which keeps ``log u`` finite in every log-uniform (N07 §9.1),
    including the ``a < 1`` boost of :func:`R.loggamma`. A stand-in ``jax.random`` whose
    ``uniform`` returns the bottom of the requested interval (``maxval = minval``, through JAX's
    own affine map and floor: what the zero-bits draw gives) makes every draw that one."""
    real = jr.uniform

    def bottom(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
        return real(key, shape, dtype=dtype, minval=minval, maxval=minval)

    stub = types.ModuleType("jax_random_zero_bits")
    stub.__dict__.update(vars(jr))
    stub.uniform = bottom
    monkeypatch.setattr(R, "jr", stub)
    assert np.array_equal(np.asarray(R.uniform_tiny(key(0), (3,))), np.full(3, R.TINY))
    lg = np.asarray(R.loggamma(key(1), jnp.full(64, 1e-3)))
    assert np.all(np.isfinite(lg)) and np.all(lg < -7e5)  # log G_{a+1} + log(TINY)/a


# --------------------------------------------------------------------------- categorical (D-4)


def brute_categorical(u: np.ndarray, logw: np.ndarray) -> np.ndarray:
    """Smallest ``k`` with ``cw_k > u·cw_J`` (inverse CDF, N07 §9.2), clamped to ``J − 1``."""
    w = np.exp(logw - logw.max(axis=-1, keepdims=True))
    cw = np.cumsum(w, axis=-1)
    out = np.array(
        [np.searchsorted(c, ui * c[-1], side="right") for c, ui in zip(cw, u, strict=True)]
    )
    return np.minimum(out, logw.shape[-1] - 1)


def test_categorical_inverse_cdf_matches_brute_force():
    r = np.random.default_rng(1)
    J, M = 10, 5000
    logw = r.normal(size=(M, J)) * 5.0
    logw[r.random((M, J)) < 0.2] = -np.inf
    logw[:, 3] = r.normal(size=M)  # at least one finite weight per row
    u = r.random(M)
    u[:5] = [0.0, 1.0 - 2.0**-53, 0.5, 1e-300, 0.999999]
    k = R.categorical_inverse_cdf(jnp.asarray(u), jnp.asarray(logw))
    assert k.dtype == jnp.int32 and k.shape == (M,)
    assert np.array_equal(np.asarray(k), brute_categorical(u, logw))
    assert np.all(np.isfinite(logw[np.arange(M), np.asarray(k)]))  # never a zero-weight category


def test_zero_weight_categories_are_never_chosen():
    logw = jnp.asarray([-np.inf, 0.0, -np.inf, 1.0, -np.inf, -np.inf])
    for u in (0.0, 1e-300, 0.2, 0.5, 0.9, 1.0 - 2.0**-53):
        assert int(R.categorical_inverse_cdf(jnp.asarray(u), logw)) in (1, 3)
    assert int(R.categorical_inverse_cdf(jnp.asarray(0.0), logw)) == 1
    assert int(R.categorical_inverse_cdf(jnp.asarray(1.0 - 2.0**-53), logw)) == 3
    assert int(R.categorical_inverse_cdf(jnp.asarray(1.0), logw)) == 5  # the J − 1 clamp


def test_categorical_logsumexp_output():
    r = np.random.default_rng(2)
    logw = r.normal(size=(50, 7)) * 30 + 650
    logw[:, 2] = -np.inf
    u = jnp.asarray(r.random(50))
    k, lse = R.categorical_inverse_cdf_lse(u, jnp.asarray(logw))
    np.testing.assert_allclose(np.asarray(lse), sc.logsumexp(logw, axis=-1), rtol=1e-15, atol=0)
    assert np.array_equal(np.asarray(k), np.asarray(R.categorical_inverse_cdf(u, logw)))


@pytest.mark.parametrize("shift", [700.0, -700.0, 0.0])
def test_categorical_chi2_with_shifted_log_weights(shift):
    """N07 §9.2: target (0.001, 0.2, 0.3, 0.499) with log weights shifted by ±700."""
    p = np.array([0.001, 0.2, 0.3, 0.499])
    logw = jnp.broadcast_to(jnp.asarray(np.log(p) + shift), (N * 2, 4))
    k = np.asarray(R.categorical(key(3), logw))
    counts = np.bincount(k, minlength=4)
    assert stats.chisquare(counts, 2 * N * p).pvalue > 1e-3


def test_categorical_uses_one_uniform_per_row():
    logw = jnp.asarray(np.log(np.full((3, 11, 10), 0.1)))
    k = R.categorical(key(4), logw)
    u = jr.uniform(key(4), (3, 11), dtype=jnp.float64)
    assert k.shape == (3, 11)
    assert np.array_equal(np.asarray(k), np.asarray(R.categorical_inverse_cdf(u, logw)))
    assert R.categorical(key(4), logw[0, 0]).shape == ()


# --------------------------------------------------------------------------- gamma family


def loggamma_cdf(x: np.ndarray, a: float) -> np.ndarray:
    """CDF of ``log G``, ``G ~ Gamma(a, 1)``: ``P(a, eˣ)``; for ``x < −30`` the leading term
    ``e^{ax}/Γ(a+1)`` (relative error ``O(eˣ)``), since ``eˣ`` underflows for small ``a``."""
    x = np.asarray(x, dtype=np.float64)
    out = sc.gammainc(a, np.exp(np.minimum(x, 700.0)))
    small = x < -30.0
    out[small] = np.exp(a * x[small] - sc.gammaln(a + 1.0))
    return out


def check_moments(x: np.ndarray, mean: float, var: float, kappa4: float, z: float = 5.0) -> None:
    """Sample mean and variance within ``z`` standard errors (fourth cumulant ``kappa4``)."""
    n = x.size
    assert abs(x.mean() - mean) <= z * np.sqrt(var / n)
    assert abs(x.var() - var) <= z * np.sqrt((kappa4 + 2 * var**2) / n)


@pytest.mark.parametrize("a", [1e-3, 0.3, 1.0, 2.5, 20.5, 1e4])
def test_loggamma_distribution(a):
    x = np.asarray(R.loggamma(key(10), a, (N,)))
    assert x.dtype == np.float64 and np.all(np.isfinite(x))
    assert stats.kstest(x, lambda v: loggamma_cdf(v, a)).pvalue > 1e-3
    # log G has cumulants ψ(a), ψ'(a), ψ''(a), ψ'''(a)
    check_moments(x, sc.digamma(a), sc.polygamma(1, a), sc.polygamma(3, a))


def test_loggamma_shapes_broadcasting_and_invalid_input():
    a = jnp.asarray([[0.5], [3.0]])
    assert R.loggamma(key(11), a, (2, 7)).shape == (2, 7)
    assert R.loggamma(key(11), a).shape == (2, 1)
    bad = np.asarray(
        jax.jit(lambda k: R.loggamma(k, jnp.asarray([0.0, -1.0, np.nan, np.inf, 2.0])))(key(12))
    )
    assert np.all(np.isnan(bad[:4])) and np.isfinite(bad[4])  # returns, never loops forever


@pytest.mark.parametrize("shape,scale", [(1e-3, 2.0), (2.0, 3.0), (50.0, 0.5)])
def test_log_inv_gamma(shape, scale):
    """``log X = log(scale) − log G``: KS of ``log(scale) − log X`` against log-gamma."""
    x = np.asarray(R.log_inv_gamma(key(13), jnp.full((N,), shape), scale))
    assert np.all(np.isfinite(x))
    g = np.log(scale) - x
    assert stats.kstest(g, lambda v: loggamma_cdf(v, shape)).pvalue > 1e-3
    if shape > 2:  # E[X] = scale/(shape − 1)
        X = np.exp(x)
        mean = scale / (shape - 1)
        var = scale**2 / ((shape - 1) ** 2 * (shape - 2))
        assert abs(X.mean() - mean) <= 5 * np.sqrt(var / N)


def test_log_inv_gamma_broadcasts_shape_and_scale():
    out = R.log_inv_gamma(key(14), jnp.asarray([[2.0], [3.0], [4.0]]), jnp.asarray([1.0, 2.0]))
    assert out.shape == (3, 2) and out.dtype == jnp.float64


@pytest.mark.parametrize("a,b", [(1e-3, 1e-3), (2.0, 5.0), (0.5, 30.0), (1e-3, 4.0)])
def test_log_beta(a, b):
    lp, l1p = (np.asarray(v) for v in R.log_beta(key(15), jnp.full((N,), a), b))
    assert np.all(np.isfinite(lp)) and np.all(np.isfinite(l1p))
    assert np.max(np.abs(np.logaddexp(lp, l1p))) <= 1e-15  # p + (1 − p) = 1 in log space
    # log p has mean ψ(a) − ψ(a+b) and variance ψ'(a) − ψ'(a+b)
    check_moments(
        lp,
        sc.digamma(a) - sc.digamma(a + b),
        sc.polygamma(1, a) - sc.polygamma(1, a + b),
        sc.polygamma(3, a) - sc.polygamma(3, a + b),
    )
    if min(a, b) >= 0.5:
        assert stats.kstest(np.exp(lp), stats.beta(a, b).cdf).pvalue > 1e-3
    if a == b:  # symmetric: P(p < ½) = ½
        assert abs(np.mean(lp < np.log(0.5)) - 0.5) <= 5 * np.sqrt(0.25 / N)


# --------------------------------------------------------------------------- truncated normal (S6)


INTERVALS = [
    (-1.0, 1.0), (2.0, 3.0), (9.0, np.inf), (-np.inf, -40.0), (-410.0, -10.0), (30.0, 30.5),
    (-0.5, np.inf), (-np.inf, np.inf), (-3.0, 50.0), (37.0, 38.0), (0.0, np.inf),
    (-1e-3, 1e-3), (5.0, 1e3), (-np.inf, 0.5),
    # |x| in [20, 38]: where log_ndtr's default order-3 asymptotic series is off by up to 4e-9
    (-np.inf, -20.0), (20.0, 25.0), (-25.0, -20.0), (-35.0, -19.99), (-np.inf, -30.0),
    # Φ(lo)/Φ(hi) ≈ 1e-309 is subnormal: the old ratio form flushed it to 0 at u = 1e-300
    (-38.0, -5.0),
]  # fmt: skip


def test_log_ndtr_is_accurate_across_the_series_switch():
    """``log Φ`` in the lower tail to rounding, including the ``x ≤ −20`` asymptotic branch
    (SciPy's ``log_ndtr`` as the reference)."""
    x = np.r_[np.linspace(-60.0, 0.0, 6001), -19.999, -20.0, -20.001, -1e3, -1e5]
    got = np.asarray(R._log_ndtr(jnp.asarray(x)))
    assert np.max(np.abs(got - sc.log_ndtr(x)) / np.abs(sc.log_ndtr(x))) <= 4e-15


def log_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(Φ(b) − Φ(a))`` for ``a < b`` in log space from SciPy's ``log_ndtr`` (independent of
    the JAX implementation): lower tail, upper tail or straddling zero."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        low = sc.log_ndtr(b) + np.log1p(-np.exp(sc.log_ndtr(a) - sc.log_ndtr(b)))
        up = sc.log_ndtr(-a) + np.log1p(-np.exp(sc.log_ndtr(-b) - sc.log_ndtr(-a)))
        mid = np.log1p(-np.exp(sc.log_ndtr(a)) - np.exp(sc.log_ndtr(-b)))
    return np.where(b <= 0, low, np.where(a >= 0, up, mid))


@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_truncated_normal_quantile(lo, hi):
    u = np.r_[
        1e-300, 1e-250, 1e-100, 1e-20, 1e-8, np.linspace(1e-3, 0.999, 999), 1 - 1e-12, 1 - 2.0**-53
    ]
    x = np.asarray(R.truncated_std_normal(jnp.asarray(u), lo, hi))
    assert x.dtype == np.float64 and np.all(np.isfinite(x))
    assert np.all((x >= lo) & (x <= hi))
    assert np.all(np.diff(x) >= -1e-15)  # increasing in u (to rounding at the tail switch)
    # same-u comparison with SciPy on the bulk of u
    bulk = (u >= 1e-3) & (u <= 0.999)
    ref = stats.truncnorm.ppf(u[bulk], lo, hi)
    assert np.all(np.abs(x[bulk] - ref) <= 1e-12 * np.maximum(1.0, np.abs(ref)))
    # round trip in log space on the smaller tail mass m (u ≤ ½: lower, else upper), including
    # u = 1e-300 and 1 − 2⁻⁵³. Inverting Φ in its own tail resolves x to Δx ≈ ε·min(Φ, 1−Φ)/φ
    # (and never below one ulp of x); Δx moves log m by f(x)·Δx/m, f = φ/Z. Tolerance: 1e-10
    # plus eight such steps.
    total = log_mass(lo, hi)
    inner = (x > lo) & (x < hi)
    lower = u <= 0.5
    with np.errstate(divide="ignore"):
        lm = np.where(lower, log_mass(lo, x), log_mass(x, hi)) - total
    target = np.where(lower, np.log(u), np.log1p(-u))
    log_phi = -0.5 * x * x - 0.5 * np.log(2 * np.pi)
    log_tail = np.minimum(sc.log_ndtr(x), sc.log_ndtr(-x))
    dx = np.maximum(np.spacing(np.abs(x)), np.finfo(float).eps * np.exp(log_tail - log_phi))
    tol = 1e-10 + 8.0 * dx * np.exp(log_phi - total - target)
    sel = inner & np.isfinite(lm)
    assert np.all(np.abs(lm[sel] - target[sel]) <= tol[sel])
    assert sel.sum() >= 990


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (9.0, np.inf), (-410.0, -10.0), (-0.5, np.inf)])
def test_truncated_normal_ks(lo, hi):
    u = R.uniform_tiny(key(20), (N,))
    x = np.asarray(R.truncated_std_normal(u, lo, hi))
    lmz = log_mass(lo, hi)

    def cdf(v):
        with np.errstate(divide="ignore"):
            return np.exp(log_mass(lo, np.clip(v, lo, hi)) - lmz)

    assert stats.kstest(x, cdf).pvalue > 1e-3


def test_truncated_normal_broadcasts_bounds():
    out = R.truncated_std_normal(jnp.full((3, 4), 0.3), jnp.asarray([[-1.0], [2.0], [9.0]]), np.inf)
    assert out.shape == (3, 4) and np.all(np.asarray(out)[1] >= 2.0)


# --------------------------------------------------------------------------- batching, keys


def test_vmap_lanes_equal_unbatched_calls():
    keys = jax.vmap(key)(jnp.arange(4, dtype=jnp.uint32))
    a = jnp.asarray([1e-3, 0.7, 3.0, 40.0])
    logw = jnp.log(jnp.asarray(np.random.default_rng(5).dirichlet(np.ones(10), size=(4, 50))))
    cases = {
        "uniform_tiny": (lambda k, ai, lw: R.uniform_tiny(k, (50,))),
        "categorical": (lambda k, ai, lw: R.categorical(k, lw)),
        "loggamma": (lambda k, ai, lw: R.loggamma(k, ai, (50,))),
        "log_inv_gamma": (lambda k, ai, lw: R.log_inv_gamma(k, jnp.full((50,), ai), 2.0)),
        "log_beta": (lambda k, ai, lw: R.log_beta(k, jnp.full((50,), ai), 2.0)),
        "truncated_std_normal": (
            lambda k, ai, lw: R.truncated_std_normal(R.uniform_tiny(k, (50,)), ai, ai + 1.0)
        ),
    }
    for name, f in cases.items():
        batched = jax.jit(jax.vmap(f))(keys, a, logw)
        f_single = jax.jit(f)
        for lane in range(4):
            single = f_single(keys[lane], a[lane], logw[lane])
            for x, y in zip(jax.tree.leaves(batched), jax.tree.leaves(single), strict=True):
                np.testing.assert_allclose(
                    np.asarray(x)[lane], np.asarray(y), rtol=1e-14, atol=0, err_msg=name
                )


def test_same_key_same_draws_and_jit_equals_eager():
    a = jnp.asarray([0.2, 5.0, 1e-3])
    eager = np.asarray(R.loggamma(key(30), a, (3,)))
    assert np.array_equal(eager, np.asarray(R.loggamma(key(30), a, (3,))))
    assert np.array_equal(
        eager, np.asarray(jax.jit(R.loggamma, static_argnums=2)(key(30), a, (3,)))
    )
    assert not np.array_equal(eager, np.asarray(R.loggamma(key(31), a, (3,))))


def test_no_key_reuse():
    """Each function consumes its key once (sub-streams via ``fold_in``; DESIGN §3.1 rule 5)."""
    logw = jnp.zeros((5, 3))
    with jax.debug_key_reuse(True):
        R.uniform_tiny(key(40), (3,))
        R.categorical(key(41), logw)
        R.loggamma(key(42), jnp.asarray([0.3, 2.0]))
        R.log_inv_gamma(key(43), 2.0, 1.0)
        R.log_beta(key(44), 0.5, 0.5)

        @jax.jit
        def f(k):
            k1, k2 = jr.fold_in(k, 0), jr.fold_in(k, 1)
            return R.log_beta(k1, 1e-3, 2.0)[0] + R.loggamma(k2, 0.5)

        f(key(45))
        k = key(46)
        R.categorical(k, logw)
        with pytest.raises(Exception, match=r"(?i)reuse"):
            R.categorical(k, logw)  # the checker is live: a second consumer is caught


# --------------------------------------------------------------------------- timing (non-gating)


def test_timing_table_at_reference_shapes(capsys):
    """Prints ms per call at the reference shape (80 lanes = 4 chains × 20 equations, ``T_pad =
    320``): indicators (J = 10), τ-like log-gamma draws, IG and Beta scalars, truncated normals.
    ``jax.random.loggamma`` is shown for comparison (pilot §5). Informational only."""
    import time

    B, T = 80, 320
    keys = jax.vmap(key)(jnp.arange(B, dtype=jnp.uint32))
    logw = jnp.asarray(np.random.default_rng(6).normal(size=(B, T, 10)))
    shp = jnp.full((B, T), 3.0)  # (ν + 1)/2 at ν = 5
    cases = {
        "categorical (80 × 320 × J=10)": (jax.vmap(R.categorical), (keys, logw)),
        "loggamma, a = 3 (80 × 320)": (jax.vmap(lambda k, a: R.loggamma(k, a)), (keys, shp)),
        "jax.random.loggamma, a = 3 (80 × 320)": (
            jax.vmap(lambda k, a: jr.loggamma(k, a)),
            (keys, shp),
        ),
        "log_inv_gamma (80 scalars)": (
            jax.vmap(lambda k, a: R.log_inv_gamma(k, a, 2.0)),
            (keys, shp[:, 0] * 50),
        ),
        "log_beta (80 scalars)": (jax.vmap(lambda k, a: R.log_beta(k, a, 18.0)), (keys, shp[:, 0])),
        "truncated_std_normal (80 × 320)": (
            jax.vmap(lambda k: R.truncated_std_normal(R.uniform_tiny(k, (T,)), -1.0, 9.0)),
            (keys,),
        ),
    }
    rows = []
    for name, (f, args) in cases.items():
        fj = jax.jit(f)
        jax.block_until_ready(fj(*args))
        best = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(fj(*args))
            best = min(best, time.perf_counter() - t0)
        rows.append((name, 1e3 * best))
    with capsys.disabled():
        print("\n  WP4 random timing (min of 5, ms per call, all lanes; non-gating)")
        for name, ms in rows:
            print(f"    {name:44s} {ms:8.3f}")
    assert all(np.isfinite(ms) for _, ms in rows)
