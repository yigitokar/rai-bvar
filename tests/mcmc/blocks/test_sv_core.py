"""SV persistent core (DESIGN §13 WP12, §3.7.4, §9.3–§9.5; N02 T1–T9, §13.2–§13.4; N01 §10).

Tier U: AWOL systems against dense ``HᵀS⁻¹H + GᵀV⁻¹G`` (AR stationary, fixed B₀, RW; masks, padding
and ``T_eff``), ``Q̃ = σ²Q``, the KFS marginal system, the integrated likelihood against a dense
MVN, the stable 2×2 centre against exact rationals, MH-ratio identities for every move (exact
``h``, centred ``(γ, φ)`` and σ², NC φ, NC ``(μ, σ)`` with the ``w`` terms, the InvGamma single
accept and the ASIS Jacobian), the data flow between moves (σ² given the accepted ``(μ′, φ′)``;
the NC accept against ``w`` of the post-``h``-step state, pilot §3.4), kernel-level scale
equivariance (N01 §12.1), padding exactness, ``vmap`` = loop, reproducibility, failures as
values, the zero-offset floor rule, the cold start (pilot §3.1), extract/ahead/transition and the
log prior against the independent WP9 reference.

Tier B: one-block invariance of the centred step (exact conditional by quadrature plus GIG/IG
draws) and of the NC φ move; prior recovery with every observation masked (N02 §13.3); exact-``h``
acceptance at ``T = 300``.

Tier G: block Geweke ``gauss_sv_exact`` and ``aux_sv`` (DESIGN §9.2) through a local test double of
the production sweep (``ZeroMean × SV``, ``n = 1``: the vol step on ``e = y``), and the mutants the
pilot found catchable: M6 (``|σ|`` without the flip, N02 T16), ``exact=False`` and M7 under the
deliberately coarse one-component mixture (pilot §3.4). RW / InvGamma / fixed-B₀ / KSC-7 /
no-ASIS variants run nightly.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from jax import random
from scipy import special, stats

from rai_bvar.components.vol import mixture as MX
from rai_bvar.components.vol import sv as SV
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.keys import Block, Sub
from rai_bvar.engines.mcmc.linalg import stable_ar1_regression, tridiag_precision_sample
from rai_bvar.engines.mcmc.protocols import KernelBase, PersistentCore
from rai_bvar.engines.mcmc.random import TINY
from rai_bvar.engines.mcmc.state import (
    EqPrior,
    StepInfo,
    SVDraw,
    SVEq,
    SVPrior,
    SVRWDraw,
    SVRWEq,
    state_template,
)
from rai_bvar.validation.geweke import GewekeConfig, geweke

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")
LOG2PI = math.log(2.0 * math.pi)
F64 = jnp.float64


# --------------------------------------------------------------------------- fixtures and oracles


def prior(**kw) -> SVPrior:
    d = dict(b_mu=-0.5, B_mu=2.0, a_phi=10.0, b_phi=2.0, B_sigma=0.3, c0=3.0, C0=0.4, B0=2.0,
             m0=-0.3, V0=1.5, log_c=math.log(1e-12))  # fmt: skip
    d.update(kw)
    return SVPrior(**{k: jnp.asarray(v, F64) for k, v in d.items()})


def vol(**kw) -> VolStatic:
    return VolStatic(persistent="sv", **kw)


def core(T_pad: int | None = None, **kw) -> SV.SVCore:
    return SV.SVCore(vol(**kw), T_pad)


def static_for(v: VolStatic, T_pad: int) -> StaticConfig:
    """``ZeroMean × SV`` at ``n = 1`` (the block-level composition)."""
    return StaticConfig(shapes=Shapes.of(1, 1, 1, T_pad), struct=StructStatic(),
                        mean=MeanStatic(kind="zero"), coef=CoefPriorStatic(), vol=v)  # fmt: skip


def eq_prior(sp: SVPrior) -> EqPrior:
    """An ``EqPrior`` slice carrying only the SV block (the references read ``vol_core``)."""
    return EqPrior(coef=None, contemp=None, vol_core=sp, transitory=None)  # type: ignore[arg-type]


def zero_noise(T_pad: int, **kw) -> SV.SVNoise:
    """A deterministic noise record: uniforms at ``tiny`` (every MH move with a finite ratio is
    accepted), zero normals; ``kw`` replaces fields."""
    tiny = jnp.asarray(TINY)
    base = dict(u_s=jnp.full(T_pad, 0.5), z_h=jnp.zeros(T_pad + 1), u_h=tiny, z_gp=jnp.zeros(2),
                u_gp=tiny, lg_s2=jnp.asarray(0.0), u_s2=tiny, z_phi=jnp.asarray(0.0), u_phi=tiny,
                z_ms=jnp.zeros(2), u_ms=tiny)  # fmt: skip
    base.update({k_: jnp.asarray(v_, F64) for k_, v_ in kw.items()})
    return SV.SVNoise(**base)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sim_h(r, T, mu, phi, sigma, rw=False, h0=None):
    h = np.empty(T + 1)
    h[0] = (mu if h0 is None else h0) + (0.0 if rw else sigma / math.sqrt(1 - phi**2) * r.normal())
    for t in range(1, T + 1):
        h[t] = (
            h[t - 1] + sigma * r.normal() if rw else mu + phi * (h[t - 1] - mu) + sigma * r.normal()
        )
    return h


def problem(
    seed, T_eff=15, T_pad=20, rw=False, mu=-0.7, phi=0.8, sigma=0.4, n_mask=2, table=MX.OCSN10
):
    """A random one-equation problem: h, ẽ, y*, indicators, masks (padded and masked rows)."""
    r = rng(seed)
    h_real = sim_h(r, T_eff, mu, phi, sigma, rw=rw, h0=-0.3 if rw else None)
    h = np.concatenate([h_real, r.normal(size=T_pad - T_eff)])
    e = np.zeros(T_pad)
    e[:T_eff] = np.exp(h_real[1:] / 2) * r.normal(size=T_eff)
    tmask = (np.arange(T_pad) < T_eff).astype(float)
    obs = tmask.copy()
    obs[r.choice(T_eff, n_mask, replace=False)] = 0.0
    le2c = np.where(tmask > 0, np.log(np.where(tmask > 0, e * e, 1.0)), 0.0)
    ys = np.where(obs > 0, np.logaddexp(le2c, math.log(1e-12)), 0.0)
    s = r.integers(0, table.J, size=T_pad).astype(np.int32)
    return dict(h=h, e=e, le2c=le2c, ys=ys, s=s, tmask=tmask, obs=obs, T_eff=T_eff, T_pad=T_pad,
                mu=mu, phi=phi, sigma=sigma)  # fmt: skip


def dense_qc(P, sp, kind, table=MX.OCSN10):
    """Dense centred posterior precision ``Q`` and canonical mean ``c`` of ``h_{0:T_eff}``:
    ``HᵀS⁻¹H + GᵀV⁻¹G`` and ``HᵀS⁻¹δ + GᵀV⁻¹(y* − m_s)`` (N02 §5.1), built independently."""
    T, mu, phi, sigma = P["T_eff"], P["mu"], P["phi"], P["sigma"]
    sp = {k: float(np.asarray(v)) for k, v in dataclasses.asdict(sp).items()}
    rw = kind == "rw"
    ph = 1.0 if rw else phi
    H = np.eye(T + 1) - ph * np.eye(T + 1, k=-1)
    if rw:
        var0, delta = sp["V0"], np.r_[sp["m0"], np.zeros(T)]
    else:
        var0 = sigma**2 / (1 - phi**2) if kind == "stationary" else sp["B0"] * sigma**2
        delta = np.r_[mu, np.full(T, mu * (1 - phi))]
    Sinv = np.diag(np.r_[1.0 / var0, np.full(T, 1.0 / sigma**2)])
    Kh = H.T @ Sinv @ H
    D = np.r_[0.0, P["obs"][:T] / table.v[P["s"][:T]]]
    yprime = np.r_[0.0, P["ys"][:T] - table.m[P["s"][:T]]]
    return Kh + np.diag(D), H.T @ Sinv @ delta + D * yprime, Kh


def dense_from_system(diag, off):
    n = diag.size
    return np.diag(diag) + np.diag(off, -1) + np.diag(off, 1) if n > 1 else np.diag(diag)


def awol(P, sp, kind, table=MX.OCSN10):
    return SV.awol_system(
        jnp.asarray(P["ys"]), jnp.asarray(P["s"]), jnp.asarray(P["obs"]), jnp.int32(P["T_eff"]),
        jnp.asarray(P["mu"]), jnp.asarray(P["phi"]), jnp.asarray(P["sigma"]), sp, table,
        h0="fixed_b0" if kind == "fixed_b0" else "stationary", rw=kind == "rw",
    )  # fmt: skip


def lnorm(x, m, v):
    return -0.5 * (LOG2PI + np.log(v) + (x - m) ** 2 / v)


def logg_np(r, table):
    comps = np.log(table.p) + stats.norm.logpdf(np.asarray(r)[..., None], table.m, np.sqrt(table.v))
    return special.logsumexp(comps, axis=-1)


KINDS = ["stationary", "fixed_b0", "rw"]


# --------------------------------------------------------------------------- AWOL (N02 T2–T5)


@pytest.mark.parametrize("kind", KINDS)
def test_awol_system_matches_dense(kind):
    sp = prior()
    for seed in range(3):
        P = problem(seed, rw=kind == "rw")
        T, Tp, mu, sigma = P["T_eff"], P["T_pad"], P["mu"], P["sigma"]
        diag, off, b = (np.asarray(a) for a in awol(P, sp, kind))
        assert diag.shape == (Tp + 1,) and off.shape == (Tp,) and b.shape == (Tp + 1,)
        Qt = dense_from_system(diag, off)
        Q, c, _ = dense_qc(P, sp, kind)
        m = T + 1
        np.testing.assert_allclose(Qt[:m, :m], sigma**2 * Q, rtol=1e-13, atol=1e-13)  # Q̃ = σ²Q
        # decoupled padded states: identity block, no coupling, zero right-hand side (D-15)
        np.testing.assert_array_equal(Qt[m:, m:], np.eye(Tp - T))
        np.testing.assert_array_equal(Qt[:m, m:], 0.0)
        np.testing.assert_array_equal(b[m:], 0.0)
        # right-hand side: h = μ + σx̃ (AR) or h = σx̃ (RW) has mean Q⁻¹c and covariance Q⁻¹
        mean_h = np.linalg.solve(Q, c)
        cov_h = np.linalg.inv(Q)
        xm = np.linalg.solve(Qt[:m, :m], b[:m])
        mean_k = sigma * xm + (0.0 if kind == "rw" else mu)
        np.testing.assert_allclose(mean_k, mean_h, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(sigma**2 * np.linalg.inv(Qt[:m, :m]), cov_h, rtol=1e-10,
                                   atol=1e-13)  # fmt: skip
        # the production draw with injected noise equals the dense formula Q⁻¹c + chol(Q)⁻ᵀz
        z = rng(100 + seed).normal(size=Tp + 1)
        x, logdet, piv = tridiag_precision_sample(*(jnp.asarray(a) for a in (diag, off, b, z)))
        h = sigma * np.asarray(x) + (0.0 if kind == "rw" else mu)
        L = np.linalg.cholesky(Q)
        np.testing.assert_allclose(h[:m], mean_h + sla.solve_triangular(L.T, z[:m]), rtol=1e-10,
                                   atol=1e-10)  # fmt: skip
        np.testing.assert_allclose(np.asarray(x)[m:], z[m:], rtol=0, atol=0)
        assert float(logdet) == pytest.approx(np.linalg.slogdet(Qt)[1], rel=1e-12)
        assert float(piv) > 0


def test_awol_nc_identity_stationary():
    """``Q̃`` is ``σ²Q``, ``Q̃_00 = 1`` exactly (stationary) and ``b̃_0 = 0`` (N02 §5.3, T3)."""
    P = problem(7)
    diag, off, b = (np.asarray(a) for a in awol(P, prior(), "stationary"))
    assert diag[0] == 1.0 and b[0] == 0.0
    np.testing.assert_array_equal(off[: P["T_eff"]], -P["phi"])


def test_kfs_marginal_system_is_the_schur_complement():
    """Our joint ``(T+1)`` system marginalises to KFS's ``T``-system and ``h_0 | h_1`` (N02 §5.2, T5)."""
    sp = prior()
    for kind in ("stationary", "fixed_b0"):
        P = problem(3, n_mask=0)
        Q, c, _ = dense_qc(P, sp, kind)
        mu, phi, s2 = P["mu"], P["phi"], P["sigma"] ** 2
        B0 = 1 / (1 - phi**2) if kind == "stationary" else float(sp.B0)
        Om = Q[1:, 1:] - np.outer(Q[1:, 0], Q[0, 1:]) / Q[0, 0]
        v1 = MX.OCSN10.v[P["s"][0]]
        assert Om[0, 0] == pytest.approx(1 / v1 + 1 / (s2 * (1 + phi**2 * B0)) + phi**2 / s2,
                                         rel=1e-12)  # fmt: skip
        np.testing.assert_allclose(Om[1:, 1:], Q[2:, 2:], rtol=1e-14)
        # h_0 | h_1 ~ N(μ + φB₀(h_1 − μ)/(1 + φ²B₀), σ²B₀/(1 + φ²B₀))
        assert 1 / Q[0, 0] == pytest.approx(s2 * B0 / (1 + phi**2 * B0), rel=1e-12)
        h1 = 0.37
        cond_mean = (c[0] - Q[0, 1] * h1) / Q[0, 0]
        assert cond_mean == pytest.approx(mu + phi * B0 * (h1 - mu) / (1 + phi**2 * B0), rel=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_integrated_likelihood_matches_dense_mvn(kind):
    """``log p̃(y* | s, θ)`` (N02 §5.6, T6) against a dense MVN over the observed rows."""
    sp = prior()
    for seed in range(3):
        P = problem(seed, rw=kind == "rw")
        T = P["T_eff"]
        _, _, Kh = dense_qc(P, sp, kind)
        cov_h = np.linalg.inv(Kh)
        on = P["obs"][:T] > 0
        rows = np.flatnonzero(on) + 1
        v = MX.OCSN10.v[P["s"][:T]][on]
        level = float(sp.m0) if kind == "rw" else P["mu"]
        yobs = (P["ys"][:T] - MX.OCSN10.m[P["s"][:T]])[on]
        ref = stats.multivariate_normal.logpdf(yobs, np.full(on.sum(), level),
                                               np.diag(v) + cov_h[np.ix_(rows, rows)])  # fmt: skip
        got = SV.aux_log_likelihood(
            jnp.asarray(P["ys"]), jnp.asarray(P["s"]), jnp.asarray(P["obs"]), jnp.int32(T),
            jnp.asarray(P["mu"]), jnp.asarray(P["phi"]), jnp.asarray(P["sigma"]), sp, MX.OCSN10,
            h0="fixed_b0" if kind == "fixed_b0" else "stationary", rw=kind == "rw",
        )  # fmt: skip
        assert float(got) == pytest.approx(ref, abs=1e-10)


def test_c_step_centre_is_exact_on_large_nearly_constant_h():
    """With ``z = 0`` and a forced accept the centred proposal is ``b_T`` of the centred 2×2 algebra;
    it equals exact rational arithmetic (N02 §6.1, T7)."""
    T = 300
    sp = prior(b_mu=-25.0, B_mu=100.0)
    for mu, phi, sigma, seed in ((-25.0, 0.5, 1e-4, 0), (-9.0, 0.99, 1e-6, 1)):
        h = sim_h(rng(seed), T, mu, phi, sigma)
        tmask = np.ones(T)
        X = [(Fraction(1), Fraction(float(x))) for x in h[:-1]]
        Z = [Fraction(float(z)) for z in h[1:]]
        e1, e2 = Fraction(1) / Fraction(SV.B0_GAMMA), Fraction(1) / Fraction(SV.B0_PHI)
        a11 = sum(x[0] * x[0] for x in X) + e1
        a12 = sum(x[0] * x[1] for x in X)
        a22 = sum(x[1] * x[1] for x in X) + e2
        r1 = sum(x[0] * z for x, z in zip(X, Z, strict=True))
        r2 = sum(x[1] * z for x, z in zip(X, Z, strict=True))
        det = a11 * a22 - a12 * a12
        gam_x, phi_x = (a22 * r1 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det
        noise = zero_noise(T)
        out = SV.c_step(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi), jnp.asarray(sigma),
                        jnp.asarray(tmask), jnp.int32(T), sp, noise, vol=vol())  # fmt: skip
        gam_p, phi_p = (float(v) for v in out.proposal)
        assert phi_p == pytest.approx(float(phi_x), rel=1e-14)
        assert gam_p == pytest.approx(float(gam_x), rel=1e-13, abs=1e-300)
        if bool(out.acc_gp):  # accepted: the state is (γ′/(1 − φ′), φ′)
            assert float(out.phi) == phi_p and float(out.mu) == gam_p / (1 - phi_p)


# --------------------------------------------------------------------------- MH identities (§9.3 item 2)


def _log_p_h(h, T, mu, phi, sigma, sp, kind, tmask):
    """Dense ``log p(h_{0:T} | θ)`` (real states, masked transitions)."""
    if kind == "rw":
        lp = lnorm(h[0], float(sp.m0), float(sp.V0))
        tr = lnorm(h[1 : T + 1], h[:T], sigma**2)
    else:
        v0 = sigma**2 / (1 - phi**2) if kind == "stationary" else float(sp.B0) * sigma**2
        lp = lnorm(h[0], mu, v0)
        tr = lnorm(h[1 : T + 1], mu + phi * (h[:T] - mu), sigma**2)
    return lp + np.sum(tr * tmask[:T])


def _log_aug_obs(h, P, table):
    """``Σ_obs [log N(ẽ; 0, e^h) + log p_s + log N(y* − h; m_s, v_s) − log g(y* − h)]``."""
    T, on = P["T_eff"], P["obs"][: P["T_eff"]] > 0
    hd, s, ys, le = h[1 : T + 1], P["s"][:T], P["ys"][:T], P["le2c"][:T]
    val = (-0.5 * (LOG2PI + hd + np.exp(le - hd)) + np.log(table.p[s])
           + lnorm(ys - hd, table.m[s], table.v[s]) - logg_np(ys - hd, table))  # fmt: skip
    return np.sum(val[on])


def _w(h, P, table):
    return SV.w_weight(jnp.asarray(h), jnp.asarray(P["le2c"]), jnp.asarray(P["ys"]),
                       jnp.asarray(P["obs"]), table)  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("table", [MX.OCSN10, MX.KSC7], ids=["ocsn10", "ksc7"])
def test_exact_h_log_accept_identity(kind, table, seed):
    """``w(h′) − w(h) = log π(h′) + log q(h) − log π(h) − log q(h′)`` with the augmented target
    ``π ∝ p(h|θ)Πp(ẽ|h)p̃(s|h, y*)`` and the AWOL proposal ``q`` (N01 §10.2)."""
    sp = prior()
    P = problem(seed, rw=kind == "rw", table=table)
    P["s"] = rng(50 + seed).integers(0, table.J, size=P["T_pad"]).astype(np.int32)
    T, m = P["T_eff"], P["T_eff"] + 1
    Q, c, _ = dense_qc(P, sp, kind, table)
    mean, cov = np.linalg.solve(Q, c), np.linalg.inv(Q)
    h_cur = P["h"].copy()
    h_new = h_cur.copy()
    h_new[:m] = rng(9 + seed).multivariate_normal(mean, cov)

    def log_pi_over_q(h):
        lp = _log_p_h(h, T, P["mu"], P["phi"], P["sigma"], sp, kind, P["tmask"])
        lp += _log_aug_obs(h, P, table)
        return lp - stats.multivariate_normal.logpdf(h[:m], mean, cov)

    ref = log_pi_over_q(h_new) - log_pi_over_q(h_cur)
    got = SV.h_log_accept(_w(h_new, P, table), _w(h_cur, P, table))
    assert float(got) == pytest.approx(ref, abs=1e-8 * max(1.0, abs(ref)))
    # the indicator step's logsumexp reused as log g gives the same w (DESIGN §10.4 lever 6)
    _, lg = SV.draw_indicators(jnp.zeros(P["T_pad"]), jnp.asarray(P["ys"] - h_cur[1:]), table)
    w_reused = SV.w_weight(jnp.asarray(h_cur), jnp.asarray(P["le2c"]), jnp.asarray(P["ys"]),
                           jnp.asarray(P["obs"]), table, lg)  # fmt: skip
    assert float(w_reused) == pytest.approx(float(_w(h_cur, P, table)), rel=1e-13)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", ["stationary", "fixed_b0"])
def test_centred_gamma_phi_log_ratio_identity(kind, seed):
    """``log R_γφ`` equals ``log π + log q`` differences with the target in ``(γ, φ)`` coordinates
    (``|dμ/dγ| = 1/(1 − φ)``) and the auxiliary proposal ``N(b_T, σ²B_T)`` (N02 §6.1, D-6)."""
    sp = prior()
    P = problem(seed, n_mask=0)
    h, T, tm = P["h"], P["T_eff"], P["tmask"]
    mu, phi, sigma = P["mu"], P["phi"], P["sigma"]
    X = np.column_stack([np.ones(T), h[:T]])
    A = X.T @ X + np.diag([1 / SV.B0_GAMMA, 1 / SV.B0_PHI])
    bT = np.linalg.solve(A, X.T @ h[1 : T + 1])
    cov = sigma**2 * np.linalg.inv(A)
    gam_p, phi_p = bT + np.linalg.cholesky(cov) @ rng(seed).normal(size=2)
    phi_p = float(np.clip(phi_p, -0.99, 0.99))

    def log_pi_over_q(g, f):
        m_ = g / (1 - f)
        lp = _log_p_h(h, T, m_, f, sigma, sp, kind, tm)
        lp += lnorm(m_, float(sp.b_mu), float(sp.B_mu)) - np.log(1 - f)
        lp += stats.beta.logpdf((f + 1) / 2, float(sp.a_phi), float(sp.b_phi))
        return lp - stats.multivariate_normal.logpdf([g, f], bT, cov)

    ref = log_pi_over_q(gam_p, phi_p) - log_pi_over_q(mu * (1 - phi), phi)
    lr, _ = SV.c_step_log_ratios(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi),
                                 jnp.asarray(sigma), jnp.asarray(gam_p), jnp.asarray(phi_p),
                                 jnp.asarray(0.1), sp, h0_kind=kind)  # fmt: skip
    assert float(lr) == pytest.approx(ref, abs=1e-8 * max(1.0, abs(ref)))
    lr, _ = SV.c_step_log_ratios(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi),
                                 jnp.asarray(sigma), jnp.asarray(0.1), jnp.asarray(1.2),
                                 jnp.asarray(0.1), sp)  # fmt: skip
    assert float(lr) == -np.inf  # |φ′| ≥ 1: immediate rejection


@pytest.mark.parametrize("kind", KINDS)
def test_sigma2_log_ratio_identity_and_ig_gibbs(kind):
    """σ² under ``B_σχ²₁``: ``(σ² − σ²′)/(2B_σ)`` equals the independence-MH ratio with proposal
    ``IG(shape, C_T)``; under ``IG(c₀, C₀)`` the full conditional is ``IG(shape, C₀ + C_T)`` (a
    density-ratio identity); shapes per :func:`sv.sigma2_proposal_shape` (N02 §6.1, §8.1)."""
    sp = prior()
    rw = kind == "rw"
    P = problem(4, rw=rw, n_mask=0)
    h, T, tm, mu, phi = P["h"], P["T_eff"], P["tmask"], P["mu"], P["phi"]
    if rw:
        C_T = 0.5 * np.sum(np.diff(h[: T + 1]) ** 2)
    else:
        om = 1 - phi**2 if kind == "stationary" else 1 / float(sp.B0)
        C_T = 0.5 * (np.sum((h[1 : T + 1] - mu - phi * (h[:T] - mu)) ** 2) + om * (h[0] - mu) ** 2)
    v_chi = vol(rw=rw, h0="fixed_b0" if kind == "fixed_b0" else "stationary")
    v_ig = dataclasses.replace(v_chi, sigma2_prior="inv_gamma")
    shape = float(SV.sigma2_proposal_shape(jnp.int32(T), sp, v_chi))
    assert shape == (T - 1) / 2 if rw else shape == T / 2
    shape_ig = float(SV.sigma2_proposal_shape(jnp.int32(T), sp, v_ig))
    assert shape_ig == float(sp.c0) + (T / 2 if rw else (T + 1) / 2)

    def lp_h(s2):
        return _log_p_h(h, T, mu, phi, math.sqrt(s2), sp, kind, tm)

    s2, s2p = 0.17, 0.31
    B = float(sp.B_sigma)
    lt = lambda x: lp_h(x) + stats.gamma.logpdf(x, 0.5, scale=2 * B)  # noqa: E731
    lq = lambda x: stats.invgamma.logpdf(x, shape, scale=C_T)  # noqa: E731
    ref = (lt(s2p) - lq(s2p)) - (lt(s2) - lq(s2))
    _, got = SV.c_step_log_ratios(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi),
                                  jnp.asarray(math.sqrt(s2)), jnp.asarray(0.0), jnp.asarray(0.0),
                                  jnp.asarray(s2p), sp)  # fmt: skip
    assert float(got) == pytest.approx(ref, abs=1e-10)
    c0, C0 = float(sp.c0), float(sp.C0)
    lt_ig = lambda x: lp_h(x) + stats.invgamma.logpdf(x, c0, scale=C0)  # noqa: E731
    lpost = lambda x: stats.invgamma.logpdf(x, shape_ig, scale=C0 + C_T)  # noqa: E731
    assert lt_ig(s2p) - lt_ig(s2) == pytest.approx(lpost(s2p) - lpost(s2), abs=1e-10)
    # the kernel maps its log-gamma noise to that IG draw (Gibbs: always accepted)
    noise = zero_noise(P["T_pad"], lg_s2=1.3, z_gp=[0.0, 1e9])  # |φ′| ≥ 1: (γ, φ) move rejected
    out = SV.c_step(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi), jnp.asarray(0.4),
                    jnp.asarray(tm), jnp.int32(T), sp, noise, vol=v_ig)  # fmt: skip
    assert bool(out.acc_s2)
    assert float(out.sigma) ** 2 == pytest.approx((C0 + C_T) / math.exp(1.3), rel=1e-13)
    assert float(out.C_T) == pytest.approx(C_T, rel=1e-13)


@pytest.mark.parametrize("sigma2_prior", ["scaled_chi2", "inv_gamma"])
@pytest.mark.parametrize("kind", ["stationary", "fixed_b0"])
def test_sigma2_step_conditions_on_the_accepted_mu_phi(kind, sigma2_prior):
    """After an **accepted** centred ``(γ, φ)`` move, σ² is drawn given the new ``(μ′, φ′)``:
    ``C_T = ½[Σ(h_j − μ′ − φ′(h_{j−1} − μ′))² + ω₀(φ′)(h_0 − μ′)²]`` (N02 §6.1 step 3). A stale
    ``(μ, φ)`` here keeps the marginals the one-block invariance test compares, so it is pinned
    deterministically: ``zero_noise`` forces both accepts and the proposal is ``b_T``."""
    sp = prior()
    P = problem(4, n_mask=0)
    h, T, tm = P["h"], P["T_eff"], P["tmask"]
    mu, phi, sigma, lg = 0.6, 0.2, 0.4, 1.3  # far from b_T, so (μ′, φ′) moves visibly
    v = vol(h0=kind, sigma2_prior=sigma2_prior)
    out = SV.c_step(jnp.asarray(h), jnp.asarray(mu), jnp.asarray(phi), jnp.asarray(sigma),
                    jnp.asarray(tm), jnp.int32(T), sp, zero_noise(P["T_pad"], lg_s2=lg),
                    vol=v)  # fmt: skip
    assert bool(out.acc_gp) and bool(out.acc_s2)
    gam_p, phi_p = (float(x) for x in out.proposal)
    mu1, phi1 = float(out.mu), float(out.phi)
    assert phi1 == phi_p and mu1 == pytest.approx(gam_p / (1.0 - phi_p), rel=1e-15)

    def c_t(m_, f):
        om = 1.0 - f * f if kind == "stationary" else 1.0 / float(sp.B0)
        res = h[1 : T + 1] - m_ - f * (h[:T] - m_)
        return 0.5 * (np.sum(res * res) + om * (h[0] - m_) ** 2)

    C_new, C_old = c_t(mu1, phi1), c_t(mu, phi)
    assert abs(C_new - C_old) > 0.05 * C_new  # a stale (μ, φ) would be visible
    assert float(out.C_T) == pytest.approx(C_new, rel=1e-12)
    scale = C_new if sigma2_prior == "scaled_chi2" else float(sp.C0) + C_new
    assert float(out.sigma) ** 2 == pytest.approx(scale / math.exp(lg), rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["stationary", "fixed_b0"])
def test_nc_phi_log_ratio_identity(kind, seed):
    sp = prior()
    P = problem(seed, n_mask=0)
    T, tm = P["T_eff"], P["tmask"]
    ht = (P["h"] - P["mu"]) / P["sigma"]
    x, z = ht[:T], ht[1 : T + 1]
    sxx, sxy = x @ x, x @ z
    phi, phi_p = 0.7, float(np.clip(sxy / sxx + rng(seed).normal() / math.sqrt(sxx), -0.99, 0.99))

    def log_pi_over_q(f):
        om = 1 - f**2 if kind == "stationary" else 1 / float(sp.B0)
        lp = lnorm(ht[0], 0.0, 1 / om) + np.sum(lnorm(z, f * x, 1.0) * tm[:T])
        lp += stats.beta.logpdf((f + 1) / 2, float(sp.a_phi), float(sp.b_phi))
        return lp - lnorm(f, sxy / sxx, 1 / sxx)

    ref = log_pi_over_q(phi_p) - log_pi_over_q(phi)
    got = SV.nc_phi_log_ratio(jnp.asarray(ht), jnp.asarray(phi), jnp.asarray(phi_p), sp,
                              h0_kind=kind)  # fmt: skip
    assert float(got) == pytest.approx(ref, abs=1e-9 * max(1.0, abs(ref)))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["stationary", "rw"])
@pytest.mark.parametrize("sigma2_prior", ["scaled_chi2", "inv_gamma"])
@pytest.mark.parametrize("table", [MX.OCSN10, MX.COARSE1], ids=["ocsn10", "coarse1"])
def test_nc_mu_sigma_log_accept_identity(kind, sigma2_prior, table, seed):
    """Exact-mode NC accept ``log α_aux + w(μ′ + σ′h̃) − w(μ + σh̃)`` equals the MH ratio of the
    regression proposal for the exact target on the sign-extended space in NC coordinates
    (the Jacobian ``σ^{T+1}`` cancels against ``p(h | θ)``; N01 §10.3, E6). InvGamma: flat σ in the
    proposal and ``p_ext`` in the target (one accept, N02 §7)."""
    sp = prior()
    rw = kind == "rw"
    v = vol(rw=rw, sigma2_prior=sigma2_prior, exact=True)
    P = problem(seed, rw=rw, table=table)
    P["s"] = rng(70 + seed).integers(0, table.J, size=P["T_pad"]).astype(np.int32)
    T = P["T_eff"]
    lev, sigma = (P["h"][0] if rw else P["mu"]), P["sigma"]
    ht = (P["h"] - (P["h"][0] if rw else P["mu"])) / sigma
    m_s, v_s = table.m[P["s"]], table.v[P["s"]]
    gp = SV.nc_regression(jnp.asarray(ht), jnp.asarray(P["ys"]), jnp.asarray(m_s),
                          jnp.asarray(v_s), jnp.asarray(P["obs"]), sp, vol=v)  # fmt: skip
    # independent dense regression
    on = P["obs"][:T] > 0
    Xd = np.column_stack([np.ones(on.sum()), ht[1 : T + 1][on]])
    Wd = np.diag(1 / v_s[:T][on])
    yd = (P["ys"][:T] - m_s[:T])[on]
    p1, m1 = (1 / float(sp.V0), float(sp.m0)) if rw else (1 / float(sp.B_mu), float(sp.b_mu))
    p2 = 1 / float(sp.B_sigma) if sigma2_prior == "scaled_chi2" else 0.0
    Kd = np.diag([p1, p2]) + Xd.T @ Wd @ Xd
    bd = np.array([p1 * m1, 0.0]) + Xd.T @ Wd @ yd
    np.testing.assert_allclose(np.asarray(gp.K), Kd, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(gp.b), bd, rtol=1e-13, atol=1e-13)
    mean, cov = np.linalg.solve(Kd, bd), np.linalg.inv(Kd)
    lev_p, sig_p = rng(seed).multivariate_normal(mean, cov)

    def log_pi_over_q(lv, sg):
        h = lv + sg * ht
        if sigma2_prior == "scaled_chi2":
            lp = lnorm(sg, 0.0, float(sp.B_sigma))
        else:
            lp = -(2 * float(sp.c0) + 1) * np.log(abs(sg)) - float(sp.C0) / sg**2
        lp += lnorm(lv, m1, 1 / p1) + _log_aug_obs(h, P, table)
        return lp - stats.multivariate_normal.logpdf([lv, sg], mean, cov)

    ref = log_pi_over_q(lev_p, sig_p) - log_pi_over_q(lev, sigma)
    if sigma2_prior == "scaled_chi2":
        laux = 0.0
    else:
        laux = float(SV.log_p_ext_sigma(jnp.asarray(sig_p), sp.c0, sp.C0)
                     - SV.log_p_ext_sigma(jnp.asarray(sigma), sp.c0, sp.C0))  # fmt: skip
    got = SV.nc_log_accept(jnp.asarray(laux), _w(lev_p + sig_p * ht, P, table),
                           _w(lev + sigma * ht, P, table), True)  # fmt: skip
    assert float(got) == pytest.approx(ref, abs=1e-8 * max(1.0, abs(ref)))
    # non-exact mode: the auxiliary ratio alone is the MH ratio for the approximate target
    assert float(SV.nc_log_accept(jnp.asarray(laux), 1.0, 2.0, False)) == laux


def test_fold_is_measure_preserving_map():
    ht = jnp.asarray([0.3, -1.2, 2.5])
    for s in (0.7, -0.7):
        sig, h2 = SV.nc_fold(jnp.asarray(s), ht)
        assert float(sig) == abs(s)
        np.testing.assert_array_equal(np.asarray(-0.4 + sig * h2), np.asarray(-0.4 + s * ht))


# --------------------------------------------------------------------------- kernel mechanics


def setup(seed=0, T_eff=40, T_pad=48, rw=False, **vkw):
    r = rng(seed)
    h = sim_h(r, T_eff, -0.5, 0.9, 0.3, rw=rw, h0=-0.5 if rw else None)
    e = np.zeros(T_pad)
    e[:T_eff] = np.exp(h[1:] / 2) * r.normal(size=T_eff)
    tmask = jnp.asarray((np.arange(T_pad) < T_eff).astype(float))
    le2 = jnp.where(tmask > 0, jnp.log(jnp.where(tmask > 0, jnp.asarray(e) ** 2, 1.0)), 0.0)
    k = core(T_pad, rw=rw, **vkw)
    sp = prior()
    cs = jax.jit(k.init)(random.fold_in(KEY, seed), le2, tmask, jnp.int32(T_eff), sp)
    return k, cs, le2, tmask, jnp.int32(T_eff), sp


def test_protocol_construction_and_overrides():
    k = core(16)
    assert isinstance(k, PersistentCore) and isinstance(k, KernelBase)
    assert k == SV.SVCore(vol(), 16) and hash(k) == hash(SV.SVCore(vol(), 16))
    assert SV.SVCore.from_static(static_for(vol(), 16)) == k
    with pytest.raises(ValueError, match="persistent='sv'"):
        SV.SVCore(VolStatic(persistent="constant"))
    with pytest.raises(ValueError, match="not overridable"):
        k.with_overrides({"no_such_internal": lambda: 0})
    ko = k.with_overrides({"h_accept": lambda wp, wc: -jnp.inf})
    assert ko != k and ko.impl("h_accept") is not SV.h_log_accept
    assert k.impl("nc_fold") is SV.nc_fold and k.impl("mixture_table")("ksc7") is MX.KSC7
    # the registry resolves the kind to this class (vol-kernel convention cls(vol_static, T_pad))
    from rai_bvar.engines.mcmc import registry

    assert registry.load("sv") is SV.SVCore
    with pytest.raises(ValueError, match="T_pad=16"):
        k.sample(KEY, None, jnp.zeros(20), jnp.ones(20), jnp.int32(20), prior())


@pytest.mark.parametrize("rw", [False, True])
def test_state_structure_dtypes_and_init(rw):
    k, cs, le2, tmask, T_eff, sp = setup(rw=rw)
    tmpl = jax.tree.map(lambda x: x[0], state_template(static_for(k.static, 48)).eq.vol.core)
    assert jax.tree.structure(cs) == jax.tree.structure(tmpl)
    for a, b in zip(jax.tree.leaves(cs), jax.tree.leaves(tmpl), strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
    assert isinstance(cs, SVRWEq if rw else SVEq)
    h = np.asarray(cs.h)
    assert np.all(np.isfinite(h)) and float(cs.sigma) > 0
    assert np.ptp(h[: int(T_eff) + 1]) > 0  # never h ≡ μ (pilot §3.1: σ = 0 would be absorbing)
    for name in ("acc_h", "acc_cs2", "acc_ncms", "acc_fold", "acc_floor", "rb_h_sum"):
        assert np.all(np.asarray(getattr(cs, name)) == 0)
    # pilot §3.1 regression: every exact-h proposal rejected, yet σ stays > 0 and the step is ok
    reject = {"h_accept": lambda wp, wc: jnp.asarray(-jnp.inf)}
    kr = k.with_overrides(reject)

    def five(st):
        def body(st, it):
            st, info = kr.sample(random.fold_in(KEY, it), st, le2, tmask, T_eff, sp)
            return st, info.ok

        return jax.lax.scan(body, st, jnp.arange(5, dtype=jnp.uint32))

    st, ok = jax.jit(five)(cs)
    assert np.all(np.asarray(ok)) and float(st.sigma) > 0
    assert float(st.acc_h) == 0 and int(st.acc_rej_run) == 5 and int(st.acc_rej_run_max) == 5
    # without the NC move, a rejected h step leaves h bit for bit
    kr0 = SV.SVCore(dataclasses.replace(k.static, asis=False), k.T_pad).with_overrides(reject)
    st, _ = jax.jit(lambda c: kr0.sample(KEY, c, le2, tmask, T_eff, sp))(cs)
    m = int(T_eff) + 1
    np.testing.assert_array_equal(np.asarray(st.h)[:m], np.asarray(cs.h)[:m])


@pytest.mark.parametrize("table", [MX.OCSN10, MX.COARSE1], ids=["ocsn10", "coarse1"])
@pytest.mark.parametrize("accept", [True, False], ids=["accept", "reject"])
@pytest.mark.parametrize("rw", [False, True], ids=["ar", "rw"])
def test_nc_accept_uses_w_of_the_post_h_step_state(rw, accept, table):
    """Pilot §3.4 stale-``w`` regression, deterministic (the PR-size Geweke lacks the power: the
    coarse-table mutant fails only at ~10× chain length). With the exact-``h`` accept forced
    (``+∞``) or refused (``−∞``), :meth:`SVCore.h_move` returns ``w`` of the ``h`` it returns, and
    inside :meth:`SVCore.sample_core` the NC step receives that ``h`` (the centred step does not
    move it) and its accept compares against that ``w``, recomputed here without the reused
    ``log g`` (N01 §10.3)."""
    k, cs, le2, tmask, Te, sp = setup(rw=rw)
    seen: dict[str, jax.Array] = {}

    def nc_step_spy(h, *a, **kw):
        seen["h"] = h
        return SV.nc_step(h, *a, **kw)

    def nc_accept_spy(la, wp, wc, ex):
        seen["w_cur"] = wc
        return SV.nc_log_accept(la, wp, wc, ex)

    forced = jnp.asarray(jnp.inf if accept else -jnp.inf)
    ko = k.with_overrides({"h_accept": lambda wp, wc: forced, "nc_step": nc_step_spy,
                           "nc_accept": nc_accept_spy, "mixture_table": lambda nm: table})  # fmt: skip
    noise = k.draw_noise(random.fold_in(KEY, 17), int(k.T_pad), Te, sp)
    obs, _ = SV.observation_mask(le2, tmask, False)
    ys = SV.y_star(le2, sp.log_c, obs)
    mu, phi = (jnp.zeros((), F64), jnp.ones((), F64)) if rw else (cs.mu, cs.phi)
    h1, acc, w_h, *_ = ko.h_move(noise, cs.h, mu, phi, cs.sigma, le2, ys, obs, Te, sp)
    assert bool(acc) is accept
    w_new = float(SV.w_weight(h1, le2, ys, obs, table))
    w_old = float(SV.w_weight(cs.h, le2, ys, obs, table))
    assert float(w_h) == pytest.approx(w_new, rel=1e-12, abs=1e-12)
    if accept:  # h moved, and a stale w(h_old) would be far outside the tolerance
        assert abs(w_new - w_old) > 1e-6 * max(1.0, abs(w_new))
    else:
        np.testing.assert_array_equal(np.asarray(h1), np.asarray(cs.h))
    ko.sample_core(noise, cs, le2, tmask, Te, sp)  # eager, so the spies record concrete values
    np.testing.assert_array_equal(np.asarray(seen["h"]), np.asarray(h1))
    assert float(seen["w_cur"]) == pytest.approx(w_new, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("rw", [False, True])
def test_sample_bookkeeping_and_reproducibility(rw):
    k, cs, le2, tmask, T_eff, sp = setup(rw=rw)
    run = jax.jit(lambda key, s: k.sample(key, s, le2, tmask, T_eff, sp))

    def many(st):
        def body(st, it):
            st, info = k.sample(random.fold_in(KEY, it), st, le2, tmask, T_eff, sp)
            return st, info.ok

        return jax.lax.scan(body, st, jnp.arange(30, dtype=jnp.uint32))

    st, ok = jax.jit(many)(cs)
    assert np.all(np.asarray(ok))
    total = 30
    a1, _ = run(random.fold_in(KEY, 99), st)
    a2, _ = run(random.fold_in(KEY, 99), st)
    for x, y in zip(jax.tree.leaves(a1), jax.tree.leaves(a2), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert 0 < float(st.acc_h) <= total and float(st.acc_cs2) <= total
    assert np.asarray(st.acc_rej_run).dtype == np.int32
    assert int(st.acc_rej_run_max) >= int(st.acc_rej_run)
    assert np.all(np.asarray(st.rb_h_sq) >= np.asarray(st.rb_h_sum) ** 2 / total - 1e-9)
    # padded states are the step's decoupled dummies μ + σz (RW: σz): finite, moderate
    hp = np.asarray(st.h)[int(T_eff) + 1 :]
    assert np.all(np.isfinite(hp)) and np.max(np.abs(hp)) < 50


def test_vmap_equals_loop():
    k, cs, le2, tmask, T_eff, sp = setup()
    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(4, dtype=jnp.uint32))
    css = jax.tree.map(lambda x: jnp.stack([x] * 4), cs)
    out, info = jax.jit(jax.vmap(lambda kk, s: k.sample(kk, s, le2, tmask, T_eff, sp)))(keys, css)
    single = jax.jit(lambda kk, s: k.sample(kk, s, le2, tmask, T_eff, sp))
    for i in range(4):
        one, _ = single(keys[i], cs)
        for x, y in zip(jax.tree.leaves(out), jax.tree.leaves(one), strict=True):
            np.testing.assert_allclose(np.asarray(x)[i], np.asarray(y), rtol=1e-14, atol=1e-14)
    assert np.all(np.asarray(info.ok))


def test_key_derivation_uses_one_sub_key_per_consumer():
    k = core(8)
    n = k.draw_noise(KEY, 8, jnp.int32(6), prior())
    np.testing.assert_array_equal(
        np.asarray(n.z_h), np.asarray(random.normal(random.fold_in(KEY, int(Sub.H)), (9,)))
    )
    np.testing.assert_array_equal(
        np.asarray(n.u_s), np.asarray(random.uniform(random.fold_in(KEY, int(Sub.S)), (8,)))
    )
    assert float(n.u_h) > 0 and n.z_ms.shape == (2,) and n.z_gp.shape == (2,)


@pytest.mark.parametrize("rw", [False, True])
def test_padding_is_exact(rw):
    """Real outputs do not depend on the padded inputs, and a padded problem equals the unpadded
    one given the same real noise (D-15; 1e-12)."""
    T_eff, T_big = 23, 32
    k_big, cs_big, le2_big, tm_big, Te, sp = setup(T_eff=T_eff, T_pad=T_big, rw=rw)
    noise = k_big.draw_noise(random.fold_in(KEY, 3), T_big, Te, sp)
    step_big = jax.jit(lambda nz, c, le: k_big.sample_core(nz, c, le, tm_big, Te, sp))
    out_big, i_big = step_big(noise, cs_big, le2_big)
    # garbage in padded rows of the inputs and of the state
    cs_g = dataclasses.replace(cs_big, h=cs_big.h.at[T_eff + 1 :].set(37.0))
    le2_g = le2_big.at[T_eff:].set(-5.0)
    out_g, _ = step_big(noise, cs_g, le2_g)
    m = T_eff + 1
    for x, y in zip(jax.tree.leaves(out_big), jax.tree.leaves(out_g), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim == 1 and x.shape[0] == T_big + 1 and x.dtype == np.float64:
            np.testing.assert_array_equal(x[:m], y[:m])
        elif x.ndim == 0:
            np.testing.assert_array_equal(x, y)
    # unpadded problem, same real noise
    k_s = core(T_eff, rw=rw)
    trim = lambda a, n_: a[:n_]  # noqa: E731
    noise_s = dataclasses.replace(noise, u_s=trim(noise.u_s, T_eff), z_h=trim(noise.z_h, m))
    cs_s = jax.tree.map(lambda a: a[:m] if a.ndim == 1 else a, cs_big)
    out_s, i_s = jax.jit(k_s.sample_core)(noise_s, cs_s, trim(le2_big, T_eff), trim(tm_big, T_eff),
                                          Te, sp)  # fmt: skip
    assert bool(i_big.ok) and bool(i_s.ok)
    for x, y in zip(jax.tree.leaves(out_big), jax.tree.leaves(out_s), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        x = x[:m] if x.ndim == 1 else x
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def _shift_levels(sp: SVPrior, cs, d: float):
    """Prior and state with every level shifted by ``d = 2 log k``: the relative offset
    ``log c``, ``b_μ``, ``m₀``, ``h`` and (AR) ``μ``."""
    sp_k = dataclasses.replace(sp, log_c=sp.log_c + d, b_mu=sp.b_mu + d, m0=sp.m0 + d)
    kw = {"h": cs.h + d} | ({"mu": cs.mu + d} if isinstance(cs, SVEq) else {})
    return sp_k, dataclasses.replace(cs, **kw)


@pytest.mark.parametrize(
    "vkw",
    [
        {},
        {"h0": "fixed_b0"},
        {"sigma2_prior": "inv_gamma"},
        {"rw": True},
        {"rw": True, "sigma2_prior": "inv_gamma"},
        {"mixture": "ksc7"},
        {"exact": False},
    ],
)
def test_scale_equivariance_at_kernel_level(vkw):
    """N01 §12.1 "scale equivariance", at kernel level as its cross-check correction requires: with
    the relative offset and every level prior shifted (``log c``, ``b_μ``, ``m₀ ↦ + 2 log k``),
    scaling ``ẽ`` by ``k`` makes the s-kernel draw the same indicators and the ``h | s`` kernel
    (with its exact accept) shift ``h`` by exactly ``2 log k`` on the same injected noise; the σ²
    and NC moves follow, so whole sweeps are equivariant (``φ``, ``σ``, accept counts unchanged).
    The centred ``(γ, φ)`` draw is equivariant in law only (its Cholesky is in uncentred ``(γ, φ)``
    coordinates): its proposal is refused in the sweeps (``z_gp = (0, 10⁹)``, ``|φ′| ≥ 1``) and its
    proposal law and MH ratio are checked under the map ``(γ, φ) ↦ (γ + 2 log k (1 − φ), φ)``. A
    fixed level prior must break the sweep equivariance (N01 §12.1)."""
    T_eff, T_pad, S = 30, 36, 8
    k, cs, le2, tmask, Te, sp = setup(seed=3, T_eff=T_eff, T_pad=T_pad, **vkw)
    rw, m = k.static.rw, T_eff + 1
    refuse_gp = jnp.asarray([0.0, 1e9])
    noises = [
        dataclasses.replace(
            k.draw_noise(random.fold_in(KEY, 300 + i), T_pad, Te, sp), z_gp=refuse_gp
        )
        for i in range(S)
    ]
    step = jax.jit(lambda nz, c, le, p: k.sample_core(nz, c, le, tmask, Te, p))

    def h_move(nz, c, le, p):
        mu, phi = (jnp.zeros((), F64), jnp.ones((), F64)) if rw else (c.mu, c.phi)
        ys = SV.y_star(le, p.log_c, tmask)
        return k.h_move(nz, c.h, mu, phi, c.sigma, le, ys, tmask, Te, p)

    move = jax.jit(h_move)

    def sweeps(c, le, p):
        for nz in noises:
            c, info = step(nz, c, le, p)
            assert bool(info.ok)
        return c

    base = sweeps(cs, le2, sp)
    h_base, acc_base, w_base, ms_base, vs_base, piv_base = move(noises[0], cs, le2, sp)
    for d in (2.0 * math.log(1e3), 2.0 * math.log(1e-2)):
        le_k = jnp.where(tmask > 0, le2 + d, 0.0)
        sp_k, cs_k = _shift_levels(sp, cs, d)
        # one h move: same indicators on real rows (padded ones are never read), same pivots and
        # decision, h + d; w moves by the constant −½ d T_obs
        h_k, acc_k, w_k, ms_k, vs_k, piv_k = move(noises[0], cs_k, le_k, sp_k)
        np.testing.assert_array_equal(np.asarray(ms_k)[:T_eff], np.asarray(ms_base)[:T_eff])
        np.testing.assert_array_equal(np.asarray(vs_k)[:T_eff], np.asarray(vs_base)[:T_eff])
        assert bool(acc_k) == bool(acc_base)
        assert float(piv_k) == pytest.approx(float(piv_base), rel=1e-12)
        np.testing.assert_allclose(np.asarray(h_k)[:m] - d, np.asarray(h_base)[:m], rtol=0,
                                   atol=1e-12)  # fmt: skip
        w_shift = -0.5 * d * T_eff if k.static.exact else 0.0
        assert float(w_k) - w_shift == pytest.approx(float(w_base), abs=1e-10)
        # whole sweeps (observed errors ≤ 2e-14)
        out = sweeps(cs_k, le_k, sp_k)
        np.testing.assert_allclose(np.asarray(out.h)[:m] - d, np.asarray(base.h)[:m], rtol=0,
                                   atol=1e-11)  # fmt: skip
        assert float(out.sigma) == pytest.approx(float(base.sigma), rel=1e-12)
        if not rw:
            assert float(out.mu) - d == pytest.approx(float(base.mu), abs=1e-11)
            assert float(out.phi) == pytest.approx(float(base.phi), rel=1e-12)
        for name in ("acc_h", "acc_cs2", "acc_ncms", "acc_fold"):
            assert float(getattr(out, name)) == float(getattr(base, name)), name
        assert float(base.acc_ncms) > 0  # the NC (μ, σ) move was exercised
        # a fixed level prior (b_μ, m₀ not shifted) is not scale-equivariant: the test has power
        sp_fixed = dataclasses.replace(sp_k, b_mu=sp.b_mu, m0=sp.m0)
        bad = sweeps(cs_k, le_k, sp_fixed)
        assert np.max(np.abs(np.asarray(bad.h)[:m] - d - np.asarray(base.h)[:m])) > 0.1
    if rw:
        return
    # centred (γ, φ) move: proposal law N(b_T, σ²B_T) and MH ratio equivariant under the map
    d = 2.0 * math.log(1e3)
    sp_k, cs_k = _shift_levels(sp, base, d)
    h, h_k = base.h, cs_k.h
    bT, cB = stable_ar1_regression(h[:-1], h[1:], tmask, 1.0 / SV.B0_GAMMA, 1.0 / SV.B0_PHI)
    bT_k, cB_k = stable_ar1_regression(h_k[:-1], h_k[1:], tmask, 1.0 / SV.B0_GAMMA,
                                       1.0 / SV.B0_PHI)  # fmt: skip
    bT, cB, bT_k, cB_k = (np.asarray(a) for a in (bT, cB, bT_k, cB_k))
    J = np.array([[1.0, -d], [0.0, 1.0]])
    np.testing.assert_allclose(bT_k, [bT[0] + d * (1.0 - bT[1]), bT[1]], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cB_k @ cB_k.T, J @ cB @ cB.T @ J.T, rtol=1e-8, atol=1e-12)
    sig = float(base.sigma)
    for z in rng(8).normal(size=(4, 2)):
        gam_p, phi_p = bT + sig * cB @ z
        lr, _ = SV.c_step_log_ratios(h, base.mu, base.phi, base.sigma, jnp.asarray(gam_p),
                                     jnp.asarray(phi_p), jnp.asarray(0.1), sp,
                                     h0_kind=k.static.h0)  # fmt: skip
        lr_k, _ = SV.c_step_log_ratios(h_k, cs_k.mu, cs_k.phi, cs_k.sigma,
                                       jnp.asarray(gam_p + d * (1.0 - phi_p)), jnp.asarray(phi_p),
                                       jnp.asarray(0.1), sp_k, h0_kind=k.static.h0)  # fmt: skip
        assert float(lr_k) == pytest.approx(float(lr), abs=1e-8 * max(1.0, abs(float(lr))))


def test_zero_offset_floor_rows_are_masked_and_counted():
    T_eff, T_pad = 30, 32
    k, cs, le2, tmask, Te, sp = setup(T_eff=T_eff, T_pad=T_pad, zero_offset=True)
    sp0 = dataclasses.replace(sp, log_c=jnp.asarray(-jnp.inf))
    le2z = le2.at[jnp.asarray([3, 17])].set(-jnp.inf)  # two exact zeros in real rows
    st, info = jax.jit(k.sample)(KEY, cs, le2z, tmask, Te, sp0)
    assert bool(info.ok) and float(st.acc_floor) == 2.0
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(st))
    obs, nf = SV.observation_mask(le2z, tmask, True)
    assert float(nf) == 2 and float(obs[3]) == 0 and float(obs[4]) == 1 and float(obs[T_eff]) == 0
    # with a relative offset an exact zero is absorbed: y* = log c, nothing masked (D-2)
    k2 = core(T_pad)
    st2, info2 = jax.jit(k2.sample)(KEY, cs, le2z, tmask, Te, sp)
    assert bool(info2.ok) and float(st2.acc_floor) == 0.0
    ys = SV.y_star(le2z, sp.log_c, tmask)
    assert float(ys[3]) == pytest.approx(math.log(1e-12), rel=1e-15)


def test_failures_are_values():
    k, cs, le2, tmask, Te, sp = setup()
    run = jax.jit(lambda c, le: k.sample(KEY, c, le, tmask, Te, sp))
    bad = dataclasses.replace(cs, sigma=jnp.asarray(0.0))
    _, info = run(bad, le2)
    assert not bool(info.ok)
    _, info = run(cs, le2.at[5].set(jnp.nan))
    assert not bool(info.ok)
    _, info = run(cs, le2)
    assert bool(info.ok) and isinstance(info, StepInfo) and float(info.min_pivot) > 0


@pytest.mark.parametrize("rw", [False, True])
def test_extract_ahead_transition(rw):
    k, cs, _, _, Te, _ = setup(rw=rw)
    d = k.extract(cs, Te)
    assert isinstance(d, SVRWDraw if rw else SVDraw)
    assert float(d.h_T) == float(cs.h[int(Te)])
    np.testing.assert_array_equal(np.asarray(k.log_var(cs)), np.asarray(cs.h[1:]))
    assert float(k.log_scale(cs)) == 0.0
    H, N = 6, 40_000
    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(N, dtype=jnp.uint32))
    paths = np.asarray(jax.jit(jax.vmap(lambda kk: k.ahead(kk, d, H)))(keys))
    steps = np.asarray(jax.jit(jax.vmap(lambda kk: k.transition(kk, d.h_T, d)))(keys))
    hT, s2 = float(d.h_T), float(d.sigma) ** 2
    j = np.arange(1, H + 1)
    if rw:
        mean, var = np.full(H, hT), j * s2
    else:
        mu, phi = float(d.mu), float(d.phi)
        mean, var = mu + phi**j * (hT - mu), s2 * (1 - phi ** (2 * j)) / (1 - phi**2)
    se = np.sqrt(var / N)
    assert np.max(np.abs(paths.mean(0) - mean) / se) < 5
    assert np.max(np.abs(paths.var(0) / var - 1)) < 0.05
    assert abs(steps.mean() - mean[0]) / se[0] < 5 and abs(steps.var() / var[0] - 1) < 0.05


@pytest.mark.parametrize(
    "vkw",
    [
        {},
        {"h0": "fixed_b0"},
        {"sigma2_prior": "inv_gamma"},
        {"rw": True},
        {"rw": True, "sigma2_prior": "inv_gamma"},
    ],
)
def test_log_prior_matches_independent_reference(vkw):
    """``SVCore.log_prior`` equals the WP9 reference density (two implementations, DESIGN §9.1)."""
    from rai_bvar.validation import reference as R

    k, cs, le2, tmask, Te, sp = setup(T_eff=21, T_pad=32, **vkw)
    static = static_for(k.static, 32)
    ref = R.SVRWRef() if k.static.rw else R.SVRef()
    eqp = eq_prior(sp)
    run = jax.jit(lambda kk, c: k.sample(kk, c, le2, tmask, Te, sp))
    for it in range(3):
        cs, _ = run(random.fold_in(KEY, it), cs)
        got = float(k.log_prior(cs, sp, tmask))
        want = float(ref.log_prior_eq(cs, eqp, static, tmask))
        assert got == pytest.approx(want, abs=1e-10)


# --------------------------------------------------------------------------- tier B: invariance


def _grid_sample(r, logp, axes, N):
    """Draws from a density tabulated on a regular grid (cell sampling plus uniform jitter)."""
    p = np.exp(logp - logp.max()).ravel()
    p /= p.sum()
    idx = np.unravel_index(r.choice(p.size, size=N, p=p), logp.shape)
    out = []
    for ax, i in zip(axes, idx, strict=True):
        step = ax[1] - ax[0]
        out.append(ax[i] + (r.random(N) - 0.5) * step)
    return out


def _centred_target_sample(r, h, T, sp, kind, sigma2_prior, N):
    """Exact draws of ``(μ, φ, σ²) | h`` (AR): ``(μ, φ)`` from a 2-D grid of the marginal (σ²
    integrated in closed form: GIG for ``B_σχ²₁``, IG for InvGamma), then σ² exactly."""
    mus = np.linspace(-4.0, 3.0, 701)
    phis = np.linspace(-0.995, 0.995, 600)
    M, Ph = np.meshgrid(mus, phis, indexing="ij")
    om = 1 - Ph**2 if kind == "stationary" else np.full_like(Ph, 1 / float(sp.B0))
    res = (
        h[None, None, 1 : T + 1] - M[..., None] - Ph[..., None] * (h[None, None, :T] - M[..., None])
    )
    C = 0.5 * (np.sum(res**2, -1) + om * (h[0] - M) ** 2)
    lp = lnorm(M, float(sp.b_mu), float(sp.B_mu)) + stats.beta.logpdf((Ph + 1) / 2, float(sp.a_phi),
                                                                     float(sp.b_phi))  # fmt: skip
    lp += 0.5 * np.log(om)
    if sigma2_prior == "inv_gamma":
        a = float(sp.c0) + (T + 1) / 2
        lp += -a * np.log(float(sp.C0) + C)
    else:  # ∫ x^{λ−1} e^{−(χ/x + ψx)/2} dx = 2(χ/ψ)^{λ/2} K_λ(√(χψ)), λ = −T/2, χ = 2C, ψ = 1/B
        lam, psi = -T / 2, 1 / float(sp.B_sigma)
        chi = 2 * C
        arg = np.sqrt(chi * psi)
        lp += 0.5 * lam * np.log(chi / psi) + np.log(special.kve(lam, arg)) - arg
    mu_d, phi_d = _grid_sample(r, lp, (mus, phis), N)
    om_d = 1 - phi_d**2 if kind == "stationary" else np.full(N, 1 / float(sp.B0))
    res = h[None, 1 : T + 1] - mu_d[:, None] - phi_d[:, None] * (h[None, :T] - mu_d[:, None])
    C_d = 0.5 * (np.sum(res**2, -1) + om_d * (h[0] - mu_d) ** 2)
    if sigma2_prior == "inv_gamma":
        s2 = stats.invgamma.rvs(
            float(sp.c0) + (T + 1) / 2, scale=float(sp.C0) + C_d, random_state=r
        )
    else:
        lam, psi = -T / 2, 1 / float(sp.B_sigma)
        chi = 2 * C_d
        s2 = stats.geninvgauss.rvs(
            lam, np.sqrt(chi * psi), scale=np.sqrt(chi / psi), random_state=r
        )
    return mu_d, phi_d, s2


@pytest.mark.parametrize("kind", ["stationary", "fixed_b0"])
@pytest.mark.parametrize("sigma2_prior", ["scaled_chi2", "inv_gamma"])
def test_centred_step_one_block_invariance(kind, sigma2_prior):
    """Draw ``(μ, φ, σ²) | h`` exactly, apply the centred step 20 times (fresh noise), compare with
    fresh exact draws (KS; N06 §5.3, N02 §13.2)."""
    r = rng(11)
    T, T_pad, N, L = 10, 12, 3000, 20
    sp = prior(b_mu=-0.5, B_mu=1.0, B_sigma=0.1, c0=3.0, C0=0.2)
    h = np.r_[sim_h(r, T, -0.5, 0.8, 0.35), np.zeros(T_pad - T)]
    mu0, phi0, s20 = _centred_target_sample(r, h, T, sp, kind, sigma2_prior, N)
    v = vol(h0=kind, sigma2_prior=sigma2_prior)
    tmask = jnp.asarray((np.arange(T_pad) < T).astype(float))
    hj = jnp.asarray(h)

    def apply(key, mu, phi, sigma):
        def body(c, l_):
            mu, phi, sigma = c
            nz = SV.SVCore(v).draw_noise(random.fold_in(key, l_), T_pad, jnp.int32(T), sp)
            o = SV.c_step(hj, mu, phi, sigma, tmask, jnp.int32(T), sp, nz, vol=v)
            return (o.mu, o.phi, o.sigma), None

        return jax.lax.scan(body, (mu, phi, sigma), jnp.arange(L, dtype=jnp.uint32))[0]

    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(N, dtype=jnp.uint32))
    mu1, phi1, sig1 = jax.jit(jax.vmap(apply))(keys, jnp.asarray(mu0), jnp.asarray(phi0),
                                               jnp.sqrt(jnp.asarray(s20)))  # fmt: skip
    mu2, phi2, s22 = _centred_target_sample(r, h, T, sp, kind, sigma2_prior, N)
    for got, want in ((mu1, mu2), (phi1, phi2), (np.asarray(sig1) ** 2, s22)):
        assert stats.ks_2samp(np.asarray(got), want).pvalue > 1e-3


def test_nc_phi_move_one_block_invariance():
    r = rng(12)
    T, T_pad, N, L = 12, 16, 4000, 20
    sp = prior(a_phi=10.0, b_phi=2.0)
    v = vol()
    ht = np.r_[sim_h(r, T, 0.0, 0.7, 1.0), np.zeros(T_pad - T)]
    phis = np.linspace(-0.9995, 0.9995, 20001)
    x, z = ht[:T], ht[1 : T + 1]
    lp = (0.5 * np.log(1 - phis**2) - 0.5 * (1 - phis**2) * ht[0] ** 2
          - 0.5 * np.sum((z[None] - phis[:, None] * x[None]) ** 2, -1)
          + stats.beta.logpdf((phis + 1) / 2, 10, 2))  # fmt: skip
    (phi0,) = _grid_sample(r, lp, (phis,), N)
    (phi_ref,) = _grid_sample(r, lp, (phis,), N)
    tmask = jnp.asarray((np.arange(T_pad) < T).astype(float))
    hj = jnp.asarray(ht)
    zeros = jnp.zeros(T_pad)

    def apply(key, phi):
        def body(phi, l_):
            nz = SV.SVCore(v).draw_noise(random.fold_in(key, l_), T_pad, jnp.int32(T), sp)
            mv = SV.nc_step(hj, 0.0, phi, 1.0, zeros, zeros, jnp.ones(T_pad), zeros, tmask, sp,
                            nz, vol=v)  # fmt: skip
            return mv.phi, None

        return jax.lax.scan(body, phi, jnp.arange(L, dtype=jnp.uint32))[0]

    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(N, dtype=jnp.uint32))
    out = np.asarray(jax.jit(jax.vmap(apply))(keys, jnp.asarray(phi0)))
    assert stats.ks_2samp(out, phi_ref).pvalue > 1e-3


@pytest.mark.parametrize(
    "vkw",
    [
        {},
        {"h0": "fixed_b0"},
        {"rw": True},
        {"sigma2_prior": "inv_gamma", "asis": False},
        {"rw": True, "sigma2_prior": "inv_gamma", "asis": False},
    ],
)
def test_prior_recovery_with_every_observation_masked(vkw):
    """With no observations the kernel samples the prior (N02 §13.3): chains from a fixed start,
    150 sweeps, last states against fresh prior draws (KS)."""
    from rai_bvar.validation import reference as R

    T_eff, T_pad, C, S = 12, 16, 1200, 150
    sp = prior(b_mu=0.0, B_mu=0.5, a_phi=10.0, b_phi=2.0, B_sigma=0.1, c0=4.0, C0=0.3, m0=0.0,
               V0=1.0)  # fmt: skip
    k = core(T_pad, **vkw)
    tmask = jnp.asarray((np.arange(T_pad) < T_eff).astype(float))
    zeros, Te = jnp.zeros(T_pad), jnp.int32(T_eff)
    start = k.init(KEY, zeros, tmask, Te, sp)

    def chain(key):
        def body(st, it):
            nz = k.draw_noise(random.fold_in(key, it), T_pad, Te, sp)
            st, info = k.sample_core(nz, st, zeros, tmask, Te, sp, obs=zeros)
            return st, info.ok

        return jax.lax.scan(body, start, jnp.arange(S, dtype=jnp.uint32))

    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(C, dtype=jnp.uint32))
    st, ok = jax.jit(jax.vmap(chain))(keys)
    assert np.all(np.asarray(ok))
    static = static_for(k.static, T_pad)
    ref = R.SVRWRef() if k.static.rw else R.SVRef()
    eqp = eq_prior(sp)
    draws = jax.jit(jax.vmap(lambda kk: ref.prior_sample_eq(kk, 0, eqp, static, tmask, Te)))(
        jax.vmap(lambda i: random.fold_in(random.fold_in(KEY, 777), i))(
            jnp.arange(4000, dtype=jnp.uint32)
        )
    )
    names = ["sigma"] + ([] if k.static.rw else ["mu", "phi"])
    for name in names:
        assert (
            stats.ks_2samp(np.asarray(getattr(st, name)), np.asarray(getattr(draws, name))).pvalue
            > 1e-3
        ), name
    for j in (0, 1, T_eff // 2, T_eff):
        assert stats.ks_2samp(np.asarray(st.h[:, j]), np.asarray(draws.h[:, j])).pvalue > 1e-3, j


@pytest.mark.parametrize(("mixture", "lo", "hi"), [("ocsn10", 0.95, 0.995), ("ksc7", 0.6, 0.9)])
def test_exact_h_acceptance_at_T300(mixture, lo, hi):
    """Joint-path exact-``h`` acceptance at ``T = 300`` (N02 §9: 0.96–0.98; pilot §3: OCSN-10 mean
    over chains in [0.95, 0.995], KSC-7 ≈ 0.75–0.82); short rejection runs."""
    T, T_pad, C, burn, S = 300, 320, 4, 200, 1000
    r = rng(5)
    h = sim_h(r, T, -1.0, 0.95, 0.2)
    e = np.zeros(T_pad)
    e[:T] = np.exp(h[1:] / 2) * r.normal(size=T)
    tmask = jnp.asarray((np.arange(T_pad) < T).astype(float))
    le2 = jnp.where(tmask > 0, jnp.log(jnp.where(tmask > 0, jnp.asarray(e) ** 2, 1.0)), 0.0)
    sp = prior(
        b_mu=float(np.log(np.mean(e[:T] ** 2))), B_mu=10.0, a_phi=20.0, b_phi=1.5, B_sigma=1.0
    )
    k, Te = core(T_pad, mixture=mixture), jnp.int32(T)

    def chain(key):
        st = k.init(key, le2, tmask, Te, sp)

        def body(st, it):
            return k.sample(random.fold_in(key, it), st, le2, tmask, Te, sp)[0], None

        st = jax.lax.scan(body, st, jnp.arange(burn, dtype=jnp.uint32))[0]
        # every rate below is over the S post-burn-in sweeps
        st = dataclasses.replace(st, acc_h=jnp.zeros(()), acc_ncms=jnp.zeros(()),
                                 acc_rej_run_max=jnp.zeros((), jnp.int32))  # fmt: skip
        return jax.lax.scan(body, st, jnp.arange(burn, burn + S, dtype=jnp.uint32))[0]

    keys = jax.vmap(lambda i: random.fold_in(KEY, i))(jnp.arange(C, dtype=jnp.uint32))
    st = jax.jit(jax.vmap(chain))(keys)
    rate = np.asarray(st.acc_h) / S
    assert lo <= rate.mean() <= hi, rate
    if mixture == "ocsn10":
        assert int(np.max(np.asarray(st.acc_rej_run_max))) <= 10
        assert np.asarray(st.acc_ncms).mean() / S > 0.97  # NC accept (pilot: 0.994)


# --------------------------------------------------------------------------- tier G: block Geweke


@dataclasses.dataclass(frozen=True, eq=False)
class CoreJoint:
    """Test double of ``JointSpec.step`` for ``ZeroMean × SV`` at ``n = 1`` (DESIGN §9.2 block
    configurations): the vol step on ``e = y`` with the production :class:`SVCore` (no transitory
    scale, so ``log ẽ² = log e²``). Prior, simulator and test functions are WP9's. A failed step
    turns the state into NaN, which fails the Geweke test (no silent dropping)."""

    spec: object
    core: SV.SVCore
    name: str

    @property
    def names(self):
        return self.spec.names

    def prior_sample(self, key):
        return self.spec.prior_sample(key)

    def simulate(self, key, st):
        return self.spec.simulate(key, st)

    def test_functions(self, st, data):
        return self.spec.test_functions(st, data)

    def step(self, key, it, st, data):
        k = random.fold_in(random.fold_in(random.fold_in(key, it), int(Block.EQ_VOL)), 0)
        tm, e = data.tmask, data.Y[:, 0]
        le2 = jnp.where(tm > 0, jnp.log(jnp.where(tm > 0, e * e, 1.0)), 0.0)
        cs = jax.tree.map(lambda x: x[0], st.eq.vol.core)
        pr = jax.tree.map(lambda x: x[0], self.spec.prior.eq)
        new, info = self.core.sample(k, cs, le2, tm, data.T_eff, pr)
        new = dataclasses.replace(new, sigma=jnp.where(info.ok, new.sigma, jnp.nan))
        vol_ = dataclasses.replace(st.eq.vol, core=jax.tree.map(lambda x: x[None], new))
        return dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vol_))


def _block_geweke(name, *, seed=7, coarse=False, overrides=None, cls=SV.SVCore, M_c=100, **spec_kw):
    from rai_bvar.validation.joint import joint_spec

    spec = joint_spec(name, **spec_kw)
    k = cls.from_static(spec.static)
    ov = dict(overrides or {})
    if coarse:
        ov["mixture_table"] = lambda nm: MX.COARSE1
    if ov:
        k = k.with_overrides(ov)
    tag = f"wp12-{name}-{sorted(spec_kw.items())}-{coarse}-{sorted(ov)}"
    if cls is not SV.SVCore or M_c != 100:  # the PR-tier names (and so their streams) are unchanged
        tag += f"-{cls.__name__}-{M_c}"
    model = CoreJoint(spec, k, tag)
    return geweke(model, GewekeConfig(C=1000, M_c=M_c, M1=200_000, alpha=1e-3, seed=seed))


@dataclasses.dataclass(frozen=True)
class StaleW(SV.SVCore):
    """Pilot §3.4 stale-``w`` mutant: the NC accept compares against ``w`` of the state *before*
    the ``h`` step, even after an accepted ``h`` move (``w_h = w_cur``)."""

    def h_move(self, noise, h, mu, phi, sigma, log_e2c, ys, obs, T_eff, prior_i):
        out = super().h_move(noise, h, mu, phi, sigma, log_e2c, ys, obs, T_eff, prior_i)
        return (out[0], out[1], SV.w_weight(h, log_e2c, ys, obs, self.table()), *out[3:])


def test_data_of_zero_mean_spec_is_the_residual():
    """The test double reads ``e = y``: true for ZeroMean at ``n = 1`` (WP9 simulator)."""
    from rai_bvar.validation.joint import joint_spec
    from rai_bvar.validation.simulate import gaussian_shocks  # noqa: F401  (WP9 API present)

    spec = joint_spec("gauss_sv_exact")
    st = spec.prior_sample(random.fold_in(KEY, 1))
    d = spec.simulate(random.fold_in(KEY, 2), st)
    assert spec.static.mean.kind == "zero" and spec.static.shapes.n == 1
    assert np.all(np.asarray(d.Y)[int(d.T_eff) :] == 0.0)
    assert spec.static.vol == SV.SVCore.from_static(spec.static).static


@pytest.mark.geweke
@pytest.mark.parametrize("name", ["gauss_sv_exact", "aux_sv"])
def test_block_geweke_passes(name):
    res = _block_geweke(name)
    assert res.passed, res.table()


@pytest.mark.geweke
def test_block_geweke_exact_mode_with_coarse_mixture():
    """Exact mode is exact for any table (N01 §10.2): with the one-component table it passes,
    while ``exact=False`` and M7 (uncorrected NC move) fail (pilot §3.4)."""
    res = _block_geweke("gauss_sv_exact", coarse=True)
    assert res.passed, res.table()
    assert not _block_geweke("gauss_sv_exact", coarse=True, exact=False).passed
    m7 = {"nc_accept": lambda la, wp, wc, ex: la}
    assert not _block_geweke("gauss_sv_exact", coarse=True, overrides=m7).passed


@pytest.mark.geweke
@pytest.mark.slow
def test_block_geweke_power_against_stale_w_nightly():
    """The stale-``w`` mutant (pilot §3.4) at 80× the PR chain length (``M_c = 8000``) with the
    coarse table: the exact kernel passes and the mutant fails. Scratch power study (2–3 seeds per
    size): mutant max |z| ≈ 2 at ``M_c = 100``, 4.0–4.3 at 1000 (borderline against t_crit 4.1),
    5.4–5.8 at 4000, 6.9–8.7 at 8000; the exact kernel stays below 2.5. At the PR size the mutant
    passes, so the PR-tier guard is the deterministic
    ``test_nc_accept_uses_w_of_the_post_h_step_state``."""
    res = _block_geweke("gauss_sv_exact", coarse=True, M_c=8000)
    assert res.passed, res.table()
    assert not _block_geweke("gauss_sv_exact", coarse=True, M_c=8000, cls=StaleW).passed


@pytest.mark.geweke
def test_block_geweke_catches_abs_sigma_without_flip():
    """N02 T16 / mutant M6: ``|σ|`` without flipping ``h̃`` fails Geweke at ``T = 20``."""
    m6 = {"nc_fold": lambda s, ht: (jnp.abs(s), ht)}
    assert not _block_geweke("gauss_sv_exact", overrides=m6).passed


@pytest.mark.geweke
@pytest.mark.slow
@pytest.mark.parametrize(
    "spec_kw",
    [
        {"rw": True},
        {"sigma2_prior": "inv_gamma"},
        {"h0": "fixed_b0"},
        {"rw": True, "sigma2_prior": "inv_gamma"},
        {"mixture": "ksc7"},
        {"asis": False},
        {"rw": True, "asis": False},
    ],
)
@pytest.mark.parametrize("coarse", [False, True])
def test_block_geweke_variants_nightly(spec_kw, coarse):
    res = _block_geweke("gauss_sv_exact", seed=11, coarse=coarse, **spec_kw)
    assert res.passed, res.table()


@pytest.mark.geweke
@pytest.mark.parametrize("name", ["gauss_sv_exact", "aux_sv"])
def test_block_geweke_through_production_sweep(name):
    """Once WP14's sweep exists, the same block configurations through ``JointSpec.step``."""
    from rai_bvar.validation.joint import joint_spec, production_sweep

    spec = joint_spec(name)
    try:
        production_sweep(spec.static, spec.overrides)
    except (NotImplementedError, ImportError) as err:
        pytest.skip(f"production sweep not available yet: {err}")
    res = geweke(spec, GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=7))
    assert res.passed, res.table()
