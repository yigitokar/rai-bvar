"""The tridiagonal Newton of the CommonSV ``h`` block (``linalg.tridiag_newton``; P3 D-48, §5.4,
§9.2 item 1).

The block conditional of ``ℓ(h) = −(n/2)Σh − ½Σ q e^{−h} − ½hᵀK_h h`` (``K_h`` the stationary
AR(1) precision of §5.4, 0-based, last diagonal ``1/σ_h²`` at ``T_eff − 1``) given the current
``h`` outside a block of the §5.4 layout (``B_0 = [0, o)``, ``B_b = [o + (b−1)·block,
o + b·block) ∩ [0, T_eff)``), assembled here as the WP32 kernel will: the gradient of the full
``ℓ`` is below 1e-8 at the returned mode for every block of several layouts; the proposal
``N(ĥ, G⁻¹)`` is unchanged (bitwise) when the current values inside the block change, and
changes with a neighbour (the start depends on ``q``, the neighbours and ``(ρ, σ_h)`` only; a
start at the current block with one iteration, mutant M28, fails this); the returned Hessian is
``K_BB + diag(½ q e^{−ĥ})``; damping from far starts, the iteration cap, masked and empty
windows, and ``vmap`` lanes.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines.mcmc import linalg as LA

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

_NEWTON = jax.jit(LA.tridiag_newton, static_argnames=("max_iter", "max_halvings", "tol"))


def newton(system, n, x0=None, **kw):
    """:func:`LA.tridiag_newton` on a :func:`block_system` tuple ``(log_q, diag, off, lin, mask)``."""
    log_q, diag, off, lin, mask = system
    return _NEWTON(log_q, n, diag, off, lin, mask, x0, **kw)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def ar1_precision(rho: float, sigma2: float, T: int) -> tuple[np.ndarray, np.ndarray]:
    """``K_h`` on ``t < T``: ``σ⁻²·tridiag(diag = [1, 1+ρ², …, 1+ρ², 1], off = −ρ)`` (P3 §5.4;
    ``T = 1``: ``(1 − ρ²)/σ²``), as the stationary ``h_0 ~ N(0, σ²/(1 − ρ²))`` and
    ``h_t | h_{t−1} ~ N(ρh_{t−1}, σ²)`` imply."""
    t = np.arange(T)
    d = ((1.0 - rho**2) * (t == 0) + (t >= 1) + rho**2 * (t <= T - 2)) / sigma2
    return d, np.full(T - 1, -rho / sigma2)


def csv_problem(seed: int, T_eff: int, T_pad: int, n: int, rho: float, sigma2: float):
    """A data-generating ``h`` (stationary AR(1)), ``q_t = e^{h_t}χ²_n`` and a current state
    ``h`` (the truth plus noise; 0 for ``t ≥ T_eff``)."""
    r = rng(seed)
    h_true = np.zeros(T_eff)
    h_true[0] = np.sqrt(sigma2 / (1.0 - rho**2)) * r.normal()
    for t in range(1, T_eff):
        h_true[t] = rho * h_true[t - 1] + np.sqrt(sigma2) * r.normal()
    log_q = np.zeros(T_pad)
    log_q[:T_eff] = h_true + np.log(r.chisquare(n, T_eff))
    h = np.zeros(T_pad)
    h[:T_eff] = h_true + 0.3 * r.normal(size=T_eff)
    return SimpleNamespace(log_q=log_q, h=h, n=n, rho=rho, sigma2=sigma2, T=T_eff, T_pad=T_pad)


def blocks(T_eff: int, T_pad: int, block: int, o: int) -> list[tuple[int, int]]:
    """The §5.4 layout: ``B_0 = [0, o)``, ``B_b = [o + (b−1)·block, o + b·block)`` for
    ``b = 1 … ⌈T_pad/block⌉``, each intersected with ``[0, T_eff)`` (empty blocks kept)."""
    nb = -(-T_pad // block)
    out = [(0, min(o, T_eff))]
    for b in range(1, nb + 1):
        lo, hi = o + (b - 1) * block, o + b * block
        out.append((min(lo, T_eff), min(hi, T_eff)))
    return out


def block_system(pr: SimpleNamespace, h: np.ndarray, lo: int, hi: int, width: int):
    """``(log_q, diag, off, lin, mask)`` of block ``[lo, hi)`` in a window of static length
    ``width`` starting at ``lo``: ``K_BB`` from ``K_h``, ``lin = −K_{B,B^c} h_{B^c}`` (the left and
    right neighbours ``ρ/σ²·h_{lo−1}``, ``ρ/σ²·h_hi``); dates past the block masked."""
    kd, ko = ar1_precision(pr.rho, pr.sigma2, pr.T)
    idx = lo + np.arange(width)
    mask = idx < hi
    safe = np.minimum(idx, pr.T - 1)
    diag = np.where(mask, kd[safe], 1.0)
    pair = mask[1:] & mask[:-1]
    off = np.where(pair, ko[np.minimum(idx[:-1], max(pr.T - 2, 0))], 0.0)
    lin = np.zeros(width)
    c = pr.rho / pr.sigma2
    if hi > lo and lo >= 1:
        lin[0] += c * h[lo - 1]
    if hi > lo and hi < pr.T:
        lin[hi - 1 - lo] += c * h[hi]
    log_q = np.where(mask, pr.log_q[np.minimum(idx, pr.T_pad - 1)], 0.0)
    return tuple(jnp.asarray(a) for a in (log_q, diag, off, lin, mask.astype(np.float64)))


def full_gradient(pr: SimpleNamespace, h: np.ndarray) -> np.ndarray:
    """``∂ℓ/∂h_t`` of the full ``ℓ(h)`` on ``t < T_eff`` (dense ``K_h``)."""
    kd, ko = ar1_precision(pr.rho, pr.sigma2, pr.T)
    K = np.diag(kd) + np.diag(ko, 1) + np.diag(ko, -1)
    x = h[: pr.T]
    return -0.5 * pr.n + 0.5 * np.exp(pr.log_q[: pr.T] - x) - K @ x


# (seed, T_eff, T_pad, n, ρ, σ_h², block, offset)
LAYOUTS = [
    (1, 50, 64, 3, 0.9, 0.04, 20, 7),
    (2, 200, 224, 7, 0.98, 0.01, 100, 37),
    (3, 37, 40, 1, 0.5, 1.0, 10, 0),
    (4, 120, 128, 20, 0.999, 1e-4, 50, 49),
    (5, 20, 32, 5, -0.7, 0.2, 7, 3),
    (6, 9, 16, 2, 0.95, 0.04, 7, 6),
]


@pytest.mark.parametrize("seed,T_eff,T_pad,n,rho,s2,block,o", LAYOUTS)
def test_gradient_below_1e_8_at_the_mode(seed, T_eff, T_pad, n, rho, s2, block, o):
    """For every block of the layout, given the current ``h`` outside it, the full gradient of
    ``ℓ`` on the block is below 1e-8 at the returned mode (P3 §9.2 item 1); the Newton converged
    within the cap, and the returned Hessian is ``K_BB + diag(½ q e^{−ĥ})`` (1e-14)."""
    pr = csv_problem(seed, T_eff, T_pad, n, rho, s2)
    h = pr.h.copy()
    for lo, hi in blocks(T_eff, T_pad, block, o):
        res = newton(block_system(pr, h, lo, hi, block), pr.n)
        m = hi - lo
        mode = np.asarray(res.mode)
        assert bool(res.info.ok) and bool(res.converged), (lo, hi, int(res.iterations))
        assert int(res.iterations) <= LA.NEWTON_MAX_ITER
        assert np.all(mode[m:] == 0.0)
        if m == 0:
            continue
        h_mode = h.copy()
        h_mode[lo:hi] = mode[:m]
        g = full_gradient(pr, h_mode)[lo:hi]
        assert np.max(np.abs(g)) < 1e-8, (lo, hi, float(np.max(np.abs(g))))
        kd, _ = ar1_precision(rho, s2, T_eff)
        hd = kd[lo:hi] + 0.5 * np.exp(pr.log_q[lo:hi] - mode[:m])
        np.testing.assert_allclose(np.asarray(res.diag)[:m], hd, rtol=1e-14)
        h = h_mode  # the sequential scan: the next block sees the updated neighbour


def proposal(res) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a) for a in (res.mode, res.diag, res.off))


def test_proposal_is_unchanged_when_the_current_block_values_change():
    """The block proposal ``N(ĥ, G⁻¹)`` is a function of ``q``, the neighbours and ``(ρ, σ_h)``
    only (P3 §5.4; the M28 catcher of §9.6): two current states that differ only inside the block
    give bitwise-identical ``(ĥ, G)``; a neighbour that changes changes it; a Newton started at
    the current block for one iteration (mutant M28) gives a proposal that moves with the
    current values."""
    pr = csv_problem(11, 80, 96, 4, 0.95, 0.02)
    lo, hi, width = 30, 50, 25
    h_a = pr.h.copy()
    h_b = pr.h.copy()
    h_b[lo:hi] += np.random.default_rng(12).normal(0.0, 2.0, hi - lo)  # inside the block only
    sys_a = block_system(pr, h_a, lo, hi, width)
    sys_b = block_system(pr, h_b, lo, hi, width)
    for a, b in zip(sys_a, sys_b, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    pa = proposal(newton(sys_a, pr.n))
    pb = proposal(newton(sys_b, pr.n))
    for a, b in zip(pa, pb, strict=True):
        np.testing.assert_array_equal(a, b)
    # a neighbour changes the proposal
    h_c = h_a.copy()
    h_c[lo - 1] += 1.0
    pc = proposal(newton(block_system(pr, h_c, lo, hi, width), pr.n))
    assert np.max(np.abs(pc[0] - pa[0])) > 1e-3
    # mutant M28: start at the current block, one iteration
    m28 = []
    for h in (h_a, h_b):
        x0 = np.zeros(width)
        x0[: hi - lo] = h[lo:hi]
        m28.append(
            proposal(newton(block_system(pr, h, lo, hi, width), pr.n, jnp.asarray(x0), max_iter=1))
        )
    assert np.max(np.abs(m28[0][0] - m28[1][0])) > 1e-3


def test_damped_newton_from_far_starts_reaches_the_same_mode():
    """Starts 40 below or above the mode, or with extreme ``q`` (``log q`` shifted by ±30 on half
    the block), converge (damping keeps every step finite and ascending) to the mode from the
    default start (1e-10); one iteration from a far start reports ``converged = False``."""
    pr = csv_problem(21, 60, 64, 3, 0.9, 0.05)
    pr.log_q[10:40:2] += 30.0
    pr.log_q[11:40:2] -= 30.0
    sysm = block_system(pr, pr.h, 5, 45, 40)
    ref = newton(sysm, pr.n)
    assert bool(ref.converged)
    for shift in (-40.0, 40.0):
        far = newton(sysm, pr.n, ref.mode + shift)
        assert bool(far.converged) and bool(far.info.ok)
        np.testing.assert_allclose(np.asarray(far.mode), np.asarray(ref.mode), rtol=0, atol=1e-10)
        capped = newton(sysm, pr.n, ref.mode + shift, max_iter=1)
        assert not bool(capped.converged) and int(capped.iterations) == 1
    g = full_gradient(pr, np.concatenate([pr.h[:5], np.asarray(ref.mode), pr.h[45:]]))[5:45]
    assert np.max(np.abs(g)) < 1e-8


def test_masked_and_empty_windows():
    """Masked slots are inert (mode 0, unit diagonal, zero off-diagonal next to them), so a block
    in a longer window equals the same block in a window of its own length (1e-12); an empty
    block returns zeros at once."""
    pr = csv_problem(31, 40, 48, 6, 0.8, 0.1)
    short = newton(block_system(pr, pr.h, 12, 22, 10), pr.n)
    long_ = newton(block_system(pr, pr.h, 12, 22, 17), pr.n)
    np.testing.assert_allclose(np.asarray(long_.mode)[:10], np.asarray(short.mode), rtol=1e-12)
    assert np.all(np.asarray(long_.mode)[10:] == 0.0) and np.all(np.asarray(long_.diag)[10:] == 1.0)
    assert np.all(np.asarray(long_.off)[9:] == 0.0)
    empty = newton(block_system(pr, pr.h, 40, 40, 10), pr.n)
    assert np.all(np.asarray(empty.mode) == 0.0) and bool(empty.converged)
    assert int(empty.iterations) == 1 and bool(empty.info.ok)


def test_vmap_lanes_equal_unbatched_calls():
    """Chains batched by ``vmap`` (lanes converge after different numbers of iterations) equal
    separate calls (1e-13)."""
    systems, ns = [], []
    for c in range(4):
        pr = csv_problem(40 + c, 50, 64, 2 + c, 0.9, 0.03 * (c + 1))
        systems.append(block_system(pr, pr.h, 10 + c, 30 + 2 * c, 30))
        ns.append(float(pr.n))
    lq, dg, of, ln, mk = (jnp.stack([s[f] for s in systems]) for f in range(5))
    out = jax.jit(jax.vmap(LA.tridiag_newton))(lq, jnp.asarray(ns), dg, of, ln, mk)
    for c in range(4):
        one = newton(systems[c], ns[c])
        np.testing.assert_allclose(
            np.asarray(out.mode[c]), np.asarray(one.mode), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(np.asarray(out.diag[c]), np.asarray(one.diag), rtol=1e-13)
        assert int(out.iterations[c]) == int(one.iterations)
    assert out.mode.dtype == out.diag.dtype == jnp.float64 and out.iterations.dtype == jnp.int32
