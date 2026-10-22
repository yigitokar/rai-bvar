"""Toy posteriors by quadrature and SIR (DESIGN §9.8 A6, owned by WP12; N01 §12.6; N02 §3.3 T13).

Cheap end-to-end checks of what Geweke tests on OCSN-10 cannot see (pilot §3.4): chains start at
draws from the **exact** posterior of a tiny model (2-D quadrature, or SIR from the prior) and run
the production kernel; their moments must stay at the exact values (``|z| < 4.5``), while the
deliberately wrong kernels must leave them (``max |z| > 8``):

* **exact-mode invariance** of the ``h`` step (θ held fixed through the ``c_step`` seam, no ASIS):
  passes for any table and offset (N01 §10.2); ``exact=False`` with the one-component table fails;
* **ASIS correction** of the non-centred ``(μ, σ)`` move (``h`` step forced to reject, θ_C held):
  exact mode passes for AR and RW, ``B_σχ²₁`` and InvGamma (the single accept with ``α_aux``,
  E6); the uncorrected move (mutant M7) fails, and so does dropping ``α_aux`` under InvGamma;
* **step order** on ``y_t = β + e^{h_t/2}ε_t`` with an exact Gaussian ``β | h`` step: the production
  order (``s`` drawn inside the vol step, immediately before ``h``) passes in exact mode and in
  approximate OCSN-10 mode at ``c = 1e-3``; Primiceri's Algorithm 1 (``s`` carried across sweeps)
  fails (N01 §9; DNP 2015);
* **T13** (N02 §3.3): under Geweke data regeneration the stale order collapses ``h`` to the offset
  floor in every chain, while the production order stays in range.

The one-component table ``COARSE1`` is injected through the ``mixture_table`` override (the
pilot's coarse configuration): exact mode is exact for any table, so a poor table only removes
the approximation's cover and gives the tests power.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from rai_bvar.components.vol import mixture as MX
from rai_bvar.components.vol import sv as SV
from rai_bvar.engines.mcmc.config import VolStatic
from rai_bvar.engines.mcmc.linalg import tridiag_precision_sample
from rai_bvar.engines.mcmc.state import SVEq, SVPrior, SVRWEq

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")
N_CHAINS = 20_000
Z_PASS, Z_FAIL = 4.5, 8.0
TABLES = {"ocsn10": MX.OCSN10, "coarse1": MX.COARSE1}


def sv_prior(**kw) -> SVPrior:
    d = dict(b_mu=0.0, B_mu=0.5, a_phi=10.0, b_phi=2.0, B_sigma=0.3, c0=3.0, C0=0.6, B0=2.0,
             m0=0.2, V0=0.7, log_c=math.log(1e-12))  # fmt: skip
    d.update(kw)
    return SVPrior(**{k: jnp.asarray(v, jnp.float64) for k, v in d.items()})


def lnorm(x, m, v):
    return -0.5 * (np.log(2 * np.pi * v) + (x - m) ** 2 / v)


def grid_draws(r, logp, axes, N):
    """Draws from a density on a regular grid: cell sampling plus uniform jitter."""
    p = np.exp(logp - logp.max())
    p /= p.sum()
    idx = np.unravel_index(r.choice(p.size, N, p=p.ravel()), p.shape)
    return [ax[i] + (r.random(N) - 0.5) * (ax[1] - ax[0]) for ax, i in zip(axes, idx, strict=True)]


def zstats(out: np.ndarray, ref: np.ndarray, ref_var: np.ndarray | None = None) -> np.ndarray:
    """``(mean − exact)/se`` per column; ``ref_var/ESS`` added for SIR references."""
    var = out.var(0) / out.shape[0] + (0.0 if ref_var is None else ref_var)
    return (out.mean(0) - ref) / np.sqrt(var)


def ident_c_step(h, mu, phi, sigma, tmask, T_eff, prior, noise, *, vol):
    """``c_step`` override holding the centred θ fixed (block tests; the DESIGN §4.9 seam)."""
    f = jnp.zeros((), bool)
    return SV.CStep(mu, phi, sigma, f, f, jnp.ones((), jnp.float64), jnp.zeros(2))


def reject_h(w_prop, w_cur):
    return jnp.asarray(-jnp.inf)


def m7_nc_accept(log_alpha_aux, w_prop, w_cur, exact):
    """Mutant M7: the NC move without the exact-mode ``w`` correction."""
    return log_alpha_aux


def w_only_nc_accept(log_alpha_aux, w_prop, w_cur, exact):
    """Wrong combination under InvGamma: the ``w`` ratio without ``α_aux``."""
    return w_prop - w_cur


def sv_state(h, mu, phi, sigma, rw=False):
    z, zi = jnp.zeros(()), jnp.zeros((), jnp.int32)
    zh = jnp.zeros(h.shape[-1])
    common = dict(h=h, sigma=sigma, acc_h=z, acc_cs2=z, acc_ncms=z, acc_fold=z, acc_floor=z,
                  acc_rej_run=zi, acc_rej_run_max=zi, rb_h_sum=zh, rb_h_sq=zh)  # fmt: skip
    if rw:
        return SVRWEq(**common)
    return SVEq(mu=mu, phi=phi, acc_cgp=z, acc_ncphi=z, **common)


def log_e2(e, tmask):
    return jnp.where(tmask > 0, jnp.log(jnp.where(tmask > 0, e * e, 1.0)), 0.0)


def run_chains(kernel, starts, le2, tmask, T_eff, sp, sweeps, rw=False):
    """``sweeps`` production updates from every start (``vmap`` over chains); returns the final
    states and whether every step was ``ok``."""

    def chain(key, h, mu, phi, sigma):
        st = sv_state(h, mu, phi, sigma, rw)

        def body(st, it):
            st, info = kernel.sample(random.fold_in(key, it), st, le2, tmask, T_eff, sp)
            return st, info.ok

        st, ok = jax.lax.scan(body, st, jnp.arange(sweeps, dtype=jnp.uint32))
        return st, jnp.all(ok)

    N = starts[0].shape[0]
    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(N, dtype=jnp.uint32))
    return jax.jit(jax.vmap(chain))(keys, *(jnp.asarray(a) for a in starts))


# --------------------------------------------------------------------------- exact-mode invariance


def h_step_toy(table, exact, c):
    """θ = (−0.3, 0.6, 0.8) fixed, ``e = (0.05, 1.7)`` (a small residual), ``T_eff = 2`` padded to 4.
    Exact ``p(h_1, h_2 | e, θ)`` on a 2-D grid; ``h_0 | h_1 ~ N(μ + φ(h_1 − μ), σ²)``."""
    mu, phi, sigma, T, Tp = -0.3, 0.6, 0.8, 2, 4
    e = np.array([0.05, 1.7])
    g = np.linspace(-9.0, 5.0, 1401)
    H1, H2 = np.meshgrid(g, g, indexing="ij")
    lp = (lnorm(H1, mu, sigma**2 / (1 - phi**2)) + lnorm(H2, mu + phi * (H1 - mu), sigma**2)
          + lnorm(e[0], 0.0, np.exp(H1)) + lnorm(e[1], 0.0, np.exp(H2)))  # fmt: skip
    p = np.exp(lp - lp.max())
    p /= p.sum()
    E1, E2, E11, E22 = (p * H1).sum(), (p * H2).sum(), (p * H1**2).sum(), (p * H2**2).sum()
    E0 = mu + phi * (E1 - mu)
    E00 = sigma**2 + mu**2 * (1 - phi) ** 2 + 2 * mu * (1 - phi) * phi * E1 + phi**2 * E11
    ref = np.array([E0, E1, E2, E00, E11, E22])
    r = np.random.default_rng(0)
    h1, h2 = grid_draws(r, lp, (g, g), N_CHAINS)
    h0 = mu + phi * (h1 - mu) + sigma * r.normal(size=N_CHAINS)
    H = np.zeros((N_CHAINS, Tp + 1))
    H[:, 0], H[:, 1], H[:, 2] = h0, h1, h2
    full = np.full(N_CHAINS, 1.0)
    starts = (H, mu * full, phi * full, sigma * full)
    tmask = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    le2 = log_e2(jnp.asarray(np.r_[e, 0.0, 0.0]), tmask)
    k = SV.SVCore(VolStatic(persistent="sv", asis=False, exact=exact), Tp).with_overrides(
        {"c_step": ident_c_step, "mixture_table": lambda name: table}
    )
    st, ok = run_chains(k, starts, le2, tmask, jnp.int32(T), sv_prior(log_c=math.log(c)), 10)
    h = np.asarray(st.h)
    out = np.column_stack([h[:, 0], h[:, 1], h[:, 2], h[:, 0] ** 2, h[:, 1] ** 2, h[:, 2] ** 2])
    return zstats(out, ref), np.asarray(ok), np.asarray(st.acc_h).mean() / 10


@pytest.mark.parametrize(("table", "c"), [("ocsn10", 1e-12), ("coarse1", 1e-12), ("coarse1", 0.5)])
def test_exact_h_step_leaves_the_exact_posterior_invariant(table, c):
    z, ok, acc = h_step_toy(TABLES[table], True, c)
    assert ok.all()
    assert np.max(np.abs(z)) < Z_PASS, z
    assert 0.3 < acc < 1.0


def test_approximate_h_step_is_detected_with_a_coarse_table():
    z, ok, acc = h_step_toy(MX.COARSE1, False, 1e-12)
    assert ok.all() and acc == 1.0
    assert np.max(np.abs(z)) > Z_FAIL, z


# --------------------------------------------------------------------------- ASIS correction


def nc_toy(rw, sigma2_prior, nc_accept=None):
    """``h̃`` fixed (the ``h`` step is forced to reject and θ_C held), ``T = 3``, ``e = (0.04, −1.3,
    0.6)``. The exact target of ``(μ, σ̄)`` (RW: ``(h_0, σ̄)``) on the sign-extended plane is
    ``p(μ)p_ext(σ̄)Π N(e_t; 0, e^{μ + σ̄h̃_t})`` (the indicators sum out); compare ``μ``, ``|σ|``."""
    sp = sv_prior()
    d = {k: float(v) for k, v in sp.__dict__.items()}
    T, Tp = 3, 4
    e = np.array([0.04, -1.3, 0.6])
    ht = np.array([0.0 if rw else 0.3, 0.9, -1.4, 0.5, 0.0])
    mus, sgs = np.linspace(-6.0, 5.0, 1101), np.linspace(-6.0, 6.0, 1201)
    M, S = np.meshgrid(mus, sgs, indexing="ij")
    lp = lnorm(M, d["m0"], d["V0"]) if rw else lnorm(M, d["b_mu"], d["B_mu"])
    if sigma2_prior == "scaled_chi2":
        lp = lp + lnorm(S, 0.0, d["B_sigma"])
    else:
        S2 = np.where(S == 0.0, 1e-300, S * S)
        lp = lp - (d["c0"] + 0.5) * np.log(S2) - d["C0"] / S2
    for t in range(T):
        lp = lp + lnorm(e[t], 0.0, np.exp(M + S * ht[t + 1]))
    p = np.exp(lp - lp.max())
    p /= p.sum()
    ref = np.array([(p * M).sum(), (p * np.abs(S)).sum(), (p * M**2).sum(), (p * S**2).sum()])
    r = np.random.default_rng(1)
    mu0, sg0 = grid_draws(r, lp, (mus, sgs), N_CHAINS)
    hts = np.where(sg0[:, None] < 0, -ht[None], ht[None])  # the folded representation (D-7)
    sig0 = np.abs(sg0)
    starts = (mu0[:, None] + sig0[:, None] * hts, mu0, np.full(N_CHAINS, 0.7), sig0)
    tmask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    le2 = log_e2(jnp.asarray(np.r_[e, 0.0]), tmask)
    ov = {"c_step": ident_c_step, "h_accept": reject_h, "mixture_table": lambda name: MX.COARSE1}
    if nc_accept is not None:
        ov["nc_accept"] = nc_accept
    v = VolStatic(persistent="sv", rw=rw, sigma2_prior=sigma2_prior, exact=True)
    k = SV.SVCore(v, Tp).with_overrides(ov)
    st, ok = run_chains(k, starts, le2, tmask, jnp.int32(T), sp, 10, rw=rw)
    lev = np.asarray(st.h)[:, 0] if rw else np.asarray(st.mu)
    sig = np.asarray(st.sigma)
    out = np.column_stack([lev, sig, lev**2, sig**2])
    return zstats(out, ref), np.asarray(ok), np.asarray(st.acc_ncms).mean() / 10


@pytest.mark.parametrize("rw", [False, True], ids=["ar", "rw"])
@pytest.mark.parametrize("sigma2_prior", ["scaled_chi2", "inv_gamma"])
def test_asis_move_with_w_correction_is_exact(rw, sigma2_prior):
    z, ok, acc = nc_toy(rw, sigma2_prior)
    assert ok.all() and 0.0 < acc < 1.0
    assert np.max(np.abs(z)) < Z_PASS, z


@pytest.mark.parametrize("rw", [False, True], ids=["ar", "rw"])
def test_uncorrected_asis_move_m7_is_detected(rw):
    """M7 (DESIGN §9.7): |z| ≈ 20 here; under InvGamma it is ≈ 7, below the margin we require, so
    that prior's combination is covered by the dropped-``α_aux`` test below (|z| ≈ 150)."""
    z, _, _ = nc_toy(rw, "scaled_chi2", nc_accept=m7_nc_accept)
    assert np.max(np.abs(z)) > Z_FAIL, z


def test_inverse_gamma_needs_the_single_combined_accept():
    """``log α = log α_aux + Δw`` (E6): dropping ``α_aux`` (the p_ext ratio) is detected."""
    z, _, _ = nc_toy(False, "inv_gamma", nc_accept=w_only_nc_accept)
    assert np.max(np.abs(z)) > Z_FAIL, z


# --------------------------------------------------------------------------- step order (SIR)

Y_TOY = np.array([0.05, -1.5, 0.3])


def sir_posterior(M=3_000_000, seed=0):
    """Exact posterior of ``y_t = β + e^{h_t/2}ε_t``, ``β ~ N(0, 1)``, AR SV with the test prior
    (``μ ~ N(0, 0.5²)``, ``Beta(10, 2)``, ``B_σ = 0.1``), by importance sampling from the prior."""
    r = np.random.default_rng(seed)
    T = Y_TOY.size
    beta = r.normal(size=M)
    mu = 0.5 * r.normal(size=M)
    phi = 2 * r.beta(10, 2, size=M) - 1
    sig = math.sqrt(0.1) * np.abs(r.normal(size=M))
    h = np.empty((M, T + 1))
    h[:, 0] = mu + sig / np.sqrt(1 - phi**2) * r.normal(size=M)
    for t in range(1, T + 1):
        h[:, t] = mu + phi * (h[:, t - 1] - mu) + sig * r.normal(size=M)
    lw = np.sum(-0.5 * (h[:, 1:] + (Y_TOY[None] - beta[:, None]) ** 2 * np.exp(-h[:, 1:])), 1)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return (beta, mu, phi, sig, h), w, r


def order_functions(beta, mu, phi, sig, h):
    return np.column_stack([beta, beta**2, mu, phi, sig, h[:, 0], h[:, 1], h[:, 2], h[:, 3],
                            h[:, 1] ** 2])  # fmt: skip


@pytest.fixture(scope="module")
def sir():
    draws, w, r = sir_posterior()
    F = order_functions(*draws)
    ref = w @ F
    ref_var = (w @ (F - ref) ** 2) * np.sum(w**2)  # var/ESS
    idx = r.choice(w.size, N_CHAINS, p=w)
    return [a[idx] for a in draws], ref, ref_var


def order_chains(starts, *, order, table, exact, c, sweeps=20):
    """Chains of ``β | h`` (exact Gaussian) composed with the SV block in the given order."""
    T, Tp = 3, 4
    tmask = jnp.asarray([1.0, 1.0, 1.0, 0.0])
    yj = jnp.asarray(np.r_[Y_TOY, 0.0])
    sp = SVPrior(**{k: jnp.asarray(v, jnp.float64) for k, v in dict(
        b_mu=0.0, B_mu=0.25, a_phi=10.0, b_phi=2.0, B_sigma=0.1, c0=3.0, C0=0.4, B0=2.0, m0=0.0,
        V0=1.0, log_c=math.log(c)).items()})  # fmt: skip
    k = SV.SVCore(VolStatic(persistent="sv", exact=exact), Tp).with_overrides(
        {"mixture_table": lambda name: table}
    )
    Te = jnp.int32(T)

    def beta_step(key, h):
        w = tmask * jnp.exp(-h[1:])
        prec = 1.0 + jnp.sum(w)
        return jnp.sum(w * yj) / prec + random.normal(key, dtype=jnp.float64) / jnp.sqrt(prec)

    def le2_of(b):
        return log_e2((yj - b) * tmask, tmask)

    def chain(key, b, mu, phi, sig, h):
        st = sv_state(h, mu, phi, sig)
        if order == "production":

            def body(c_, it):
                b, st = c_
                kk = random.fold_in(key, it)
                b = beta_step(random.fold_in(kk, 0), st.h)
                st, _ = k.sample(random.fold_in(kk, 1), st, le2_of(b), tmask, Te, sp)
                return (b, st), None

            (b, st), _ = jax.lax.scan(body, (b, st), jnp.arange(sweeps, dtype=jnp.uint32))
            return b, st.mu, st.phi, st.sigma, st.h

        # Primiceri (2005) Algorithm 1: h | θ, s → s | h, θ → θ | h, with s carried across sweeps
        u0 = random.uniform(random.fold_in(key, 10**6), (Tp,), dtype=jnp.float64)
        s0, _ = SV.draw_indicators(u0, SV.y_star(le2_of(b), sp.log_c, tmask) - h[1:], table)

        def body(c_, it):
            b, mu, phi, sig, h, s = c_
            kk = random.fold_in(key, it)
            ys = SV.y_star(le2_of(b), sp.log_c, tmask)
            dg, off, rhs = SV.awol_system(ys, s, tmask, Te, mu, phi, sig, sp, table)
            z = random.normal(random.fold_in(kk, 0), (Tp + 1,), dtype=jnp.float64)
            h = mu + sig * tridiag_precision_sample(dg, off, rhs, z)[0]
            u = random.uniform(random.fold_in(kk, 1), (Tp,), dtype=jnp.float64)
            s, _ = SV.draw_indicators(u, ys - h[1:], table)
            b = beta_step(random.fold_in(kk, 2), h)
            nz = k.draw_noise(random.fold_in(kk, 3), Tp, Te, sp)
            cs = SV.c_step(h, mu, phi, sig, tmask, Te, sp, nz, vol=k.static)
            return (b, cs.mu, cs.phi, cs.sigma, h, s), None

        init = (b, mu, phi, sig, h, s0)
        (b, mu, phi, sig, h, _), _ = jax.lax.scan(body, init, jnp.arange(sweeps, dtype=jnp.uint32))
        return b, mu, phi, sig, h

    beta, mu, phi, sig, h = starts
    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(beta.size, dtype=jnp.uint32))
    hpad = np.column_stack([h, np.zeros(beta.size)])
    out = jax.jit(jax.vmap(chain))(keys, *(jnp.asarray(a) for a in (beta, mu, phi, sig, hpad)))
    return order_functions(*(np.asarray(a) for a in out))


@pytest.mark.parametrize(
    ("table", "exact", "c"),
    [("ocsn10", True, 1e-12), ("coarse1", True, 1e-12), ("ocsn10", False, 1e-3)],
)
def test_production_step_order_targets_the_posterior(sir, table, exact, c):
    """DNP Algorithm 3 (exact) for any table, and Algorithm 2 with OCSN-10 at ``c = 1e-3``."""
    starts, ref, ref_var = sir
    out = order_chains(starts, order="production", table=TABLES[table], exact=exact, c=c)
    z = zstats(out, ref, ref_var)
    assert np.max(np.abs(z)) < Z_PASS, z


def test_stale_indicator_order_algorithm_1_fails(sir):
    starts, ref, ref_var = sir
    out = order_chains(starts, order="alg1", table=MX.OCSN10, exact=False, c=1e-3)
    z = zstats(out, ref, ref_var)
    assert np.max(np.abs(z)) > Z_FAIL, z


# --------------------------------------------------------------------------- T13 (N02 §3.3)


def regression_sv_hmin(order: str, chains: int = 16, sweeps: int = 1000) -> np.ndarray:
    """``y_t = βx_t + e^{h_t/2}ε_t`` (``T = 20``, padded to 32) with Geweke regeneration of ``y``
    every sweep; the minimum real ``h`` reached along each chain."""
    T, Tp = 20, 32
    sp = SVPrior(**{k: jnp.asarray(v, jnp.float64) for k, v in dict(
        b_mu=0.0, B_mu=1.0, a_phi=20.0, b_phi=1.5, B_sigma=0.1, c0=3.0, C0=0.4, B0=2.0, m0=0.0,
        V0=1.0, log_c=math.log(1e-12)).items()})  # fmt: skip
    x = jnp.asarray(np.r_[np.random.default_rng(1).normal(size=T), np.zeros(Tp - T)])
    tmask = jnp.asarray((np.arange(Tp) < T).astype(float))
    Te = jnp.int32(T)
    k = SV.SVCore(VolStatic(persistent="sv", exact=False), Tp)
    live = jnp.arange(Tp + 1) <= T

    def beta_step(key, y, h):
        w = tmask * jnp.exp(-h[1:])
        prec = 1.0 + jnp.sum(w * x * x)
        return jnp.sum(w * x * y) / prec + random.normal(key, dtype=jnp.float64) / jnp.sqrt(prec)

    def chain(key):
        k0 = random.fold_in(key, 10**6)
        mu = random.normal(random.fold_in(k0, 1), dtype=jnp.float64)
        h = mu + 0.3 * random.normal(random.fold_in(k0, 2), (Tp + 1,), dtype=jnp.float64)
        st = sv_state(h, mu, jnp.asarray(0.9), jnp.asarray(0.3))
        b = random.normal(random.fold_in(k0, 0), dtype=jnp.float64)
        s = jnp.full((Tp,), 4, jnp.int32)

        def body(c_, it):
            b, st, s, hmin = c_
            kk = random.fold_in(key, it)
            eps = random.normal(random.fold_in(kk, 9), (Tp,), dtype=jnp.float64)
            y = (b * x + jnp.exp(st.h[1:] / 2) * eps) * tmask  # y | β, h (current latents)
            b = beta_step(random.fold_in(kk, 0), y, st.h)
            le2 = log_e2(y - b * x, tmask)
            if order == "production":
                st, _ = k.sample(random.fold_in(kk, 1), st, le2, tmask, Te, sp)
            else:  # stale: h | s_old, y*(β_new) → θ | h → s | h, β
                ys = SV.y_star(le2, sp.log_c, tmask)
                dg, off, rhs = SV.awol_system(ys, s, tmask, Te, st.mu, st.phi, st.sigma, sp,
                                              MX.OCSN10)  # fmt: skip
                z = random.normal(random.fold_in(kk, 2), (Tp + 1,), dtype=jnp.float64)
                h = st.mu + st.sigma * tridiag_precision_sample(dg, off, rhs, z)[0]
                nz = k.draw_noise(random.fold_in(kk, 3), Tp, Te, sp)
                cs = SV.c_step(h, st.mu, st.phi, st.sigma, tmask, Te, sp, nz, vol=k.static)
                u = random.uniform(random.fold_in(kk, 4), (Tp,), dtype=jnp.float64)
                s, _ = SV.draw_indicators(u, ys - h[1:], MX.OCSN10)
                st = sv_state(h, cs.mu, cs.phi, cs.sigma)
            hm = jnp.min(jnp.where(live, st.h, jnp.inf))
            return (b, st, s, jnp.minimum(hmin, jnp.where(jnp.isfinite(hm), hm, -jnp.inf))), None

        init = (b, st, s, jnp.asarray(jnp.inf))
        (_, _, _, hmin), _ = jax.lax.scan(body, init, jnp.arange(sweeps, dtype=jnp.uint32))
        return hmin

    keys = jax.vmap(lambda i: random.fold_in(random.fold_in(KEY, 13), i))(
        jnp.arange(chains, dtype=jnp.uint32)
    )
    return np.asarray(jax.jit(jax.vmap(chain))(keys))


def test_t13_stale_order_collapses_h_production_order_does_not():
    """N02 §3.3: the stale order pins ``h`` at the offset floor (``log c ≈ −27.6``: here −30) via
    positive feedback; the production order stays in range (min ``h`` ≈ −4 to −8)."""
    good = regression_sv_hmin("production")
    bad = regression_sv_hmin("stale")
    assert np.all(good > -15.0), good
    assert np.mean(bad < -20.0) >= 0.75, bad
