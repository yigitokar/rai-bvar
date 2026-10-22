"""Reference prior samplers against closed forms (DESIGN §9.1, §13 WP9; N06 §2.1, §9.1; N02 §2.1,
§8.1; N03 §2.1–§2.3).

Monte Carlo checks of ``validation.simulate.prior_sample`` (the references composed by
``joint_spec``): IG, Beta and Normal marginals; the D3 centring ``E[γ_i | a_i] = M ã_i``; the
λ scaling of Mode II and ``iw_equivalent``; stationary, fixed-``B₀`` and random-walk ``h`` moments;
outlier frequencies; ``E τ = 1`` for the standardized t; inert padding; and the finite-variance
flags of the test functions. Means are tested within 5 standard errors and variances within 5
standard errors of the sample variance (fixed seeds, so the outcome is deterministic).
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest
from jax import random
from scipy import integrate, special, stats

from rai_bvar.engines.mcmc.state import Empty, check_state_invariants
from rai_bvar.validation import joint as J
from rai_bvar.validation import reference as R

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

N = 20_000
K_SE = 5.0


def draws(spec: J.JointSpec, seed: int = 0, size: int = N):
    keys = jax.vmap(lambda i: random.fold_in(random.key(seed), i))(np.arange(size, dtype=np.uint32))
    st = jax.jit(jax.vmap(spec.prior_sample))(keys)
    return jax.tree.map(np.asarray, st)


def assert_mean(x, mean, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean() - mean) < K_SE * se + 1e-12, f"{label}: mean {x.mean()} vs {mean} (se {se})"


def assert_var(x, var, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    s2 = x.var(ddof=1)
    m4 = np.mean((x - x.mean()) ** 4)
    se = math.sqrt(max(m4 - s2**2, 0.0) / x.size)
    assert abs(s2 - var) < K_SE * se, f"{label}: var {s2} vs {var} (se {se})"


def test_constant_mode1_ig_normal_and_d3_centring():
    spec = J.joint_spec("structural_constant")  # n = 2, p = 1, intercept
    st = draws(spec)
    nd = spec.static.shapes.n_det
    # λ ~ IG(5, 4): log λ = log 4 − log G, G ~ Gamma(5)
    log_lam = st.eq.vol.core.log_lam
    for i in range(2):
        assert_mean(log_lam[:, i], math.log(4.0) - special.digamma(5.0), "log λ")
        assert_var(log_lam[:, i], special.polygamma(1, 5.0), "log λ")
    # a_10 ~ N(0, 0.7²); a_0· and a_11 inactive (exactly 0)
    a = st.eq.struct.a
    assert_mean(a[:, 1, 0], 0.0, "a")
    assert_var(a[:, 1, 0], 0.49, "a")
    assert np.all(a[:, 0, :] == 0.0)
    # δ_i ~ N(m_i, sd²): intercept s.d. 1, own lag 0.10, cross 0.05; own-lag mean 0.4
    delta = st.eq.mean.delta
    m = np.asarray(spec.prior.eq.coef.mean0)
    sd = 1.0 / np.sqrt(np.asarray(spec.prior.eq.coef.prec0))
    np.testing.assert_allclose(sd[0], [1.0, 0.10, 0.05])
    np.testing.assert_allclose(m[1], [0.0, 0.0, 0.4])
    for i in range(2):
        for r in range(3):
            assert_mean(delta[:, i, r], m[i, r], f"δ[{i},{r}]")
            assert_var(delta[:, i, r], sd[i, r] ** 2, f"δ[{i},{r}]")
    # D3 (N05 §4.1): γ_1 = δ_1 + M_c a_1, so E[γ_{1,(lag1,0)} | a_10] = M[lag1 of var 0, 0]·a_10
    Mc = np.asarray(spec.prior.shared.Mc)
    gamma = delta + np.einsum("cij,rj->cir", a, Mc)
    g = gamma[:, 1, nd + 0]
    slope = np.cov(g, a[:, 1, 0])[0, 1] / a[:, 1, 0].var(ddof=1)
    se = sd[1, nd] / (0.7 * math.sqrt(N))
    assert Mc[nd + 0, 0] == 0.4
    assert abs(slope - 0.4) < K_SE * se
    for r in range(3):  # E[γ_1 − M_c a_1] = m_1, i.e. E[γ_1 | a_1] = M ã_1
        assert_mean(gamma[:, 1, r] - a[:, 1, 0] * Mc[r, 0], m[1, r], f"γ − M_c a [{r}]")
    resid = g - 0.4 * a[:, 1, 0]
    assert abs(np.corrcoef(resid, a[:, 1, 0])[0, 1]) < K_SE / math.sqrt(N)


@pytest.mark.parametrize(("name", "scale_delta"), [("structural_mode2_constant", True),
                                                   ("structural_iw_equivalent", False)])  # fmt: skip
def test_lambda_scaled_blocks(name, scale_delta):
    spec = J.joint_spec(name)
    st = draws(spec, seed=1)
    lam = np.exp(st.eq.vol.core.log_lam)
    a = st.eq.struct.a
    # a_ij | λ_i ~ N(0, 0.49 λ_i): the standardised draw is N(0, 0.49) and independent of λ_i
    za2 = a[:, 1, 0] ** 2 / lam[:, 1]
    assert_mean(za2, 0.49, "a²/λ")
    assert abs(np.corrcoef(za2, np.log(lam[:, 1]))[0, 1]) < K_SE / math.sqrt(N)
    m = np.asarray(spec.prior.eq.coef.mean0)
    var0 = 1.0 / np.asarray(spec.prior.eq.coef.prec0)
    dev2 = (st.eq.mean.delta[:, 1, 1] - m[1, 1]) ** 2
    if scale_delta:  # Mode II: δ_i | λ_i ~ N(m_i, λ_i V)
        assert_mean(dev2 / lam[:, 1], var0[1, 1], "(δ−m)²/λ")
    else:  # iw_equivalent: only a_i is λ-scaled
        assert_mean(dev2, var0[1, 1], "(δ−m)²")
        assert abs(np.corrcoef(dev2, np.log(lam[:, 1]))[0, 1]) < K_SE / math.sqrt(N)


def test_reduced_form_pi_and_contemporaneous():
    spec = J.joint_spec("reduced_constant")  # n = 3
    st = draws(spec, seed=2)
    Pi = st.glob.mean.Pi  # (N, k, n)
    m = np.asarray(spec.prior.eq.coef.mean0).T
    sd = 1.0 / np.sqrt(np.asarray(spec.prior.eq.coef.prec0)).T
    k, n = m.shape
    for r in range(k):
        for j in range(n):
            assert_mean(Pi[:, r, j], m[r, j], f"Π[{r},{j}]")
            assert_var(Pi[:, r, j], sd[r, j] ** 2, f"Π[{r},{j}]")
    assert isinstance(st.eq.mean, Empty)  # reduced form: no per-equation mean state
    a = st.eq.struct.a
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        assert_var(a[:, i, j], 0.49, "a")
    assert np.all(a[:, 0, :] == 0) and np.all(a[:, 1, 1] == 0)


def _beta_inv_1m_phi2(a: float, b: float) -> float:
    """E[1/(1−φ²)] for (φ+1)/2 = u ~ Beta(a, b): E[1/(4u(1−u))] = B(a−1, b−1)/(4 B(a, b))."""
    return math.exp(special.betaln(a - 1, b - 1) - special.betaln(a, b)) / 4.0


def test_stationary_sv_parameters_and_h_moments():
    # b_φ = 4 so that h has a finite fourth moment (the variance test needs it; N06 §3.4)
    spec = J.joint_spec("zero_sv", "full", prior_kwargs={"b_phi": 4.0})
    st = draws(spec, seed=3)
    core = st.eq.vol.core
    mu, phi, sigma, h = core.mu[:, 0], core.phi[:, 0], core.sigma[:, 0], core.h[:, 0]
    assert_mean(mu, 0.0, "μ")
    assert_var(mu, 0.25, "μ")
    a, b = 10.0, 4.0
    u = (phi + 1) / 2
    assert_mean(u, a / (a + b), "(φ+1)/2")
    assert_var(u, a * b / ((a + b) ** 2 * (a + b + 1)), "(φ+1)/2")
    assert np.all(np.abs(phi) < 1) and np.all(sigma > 0)
    assert_mean(sigma**2, 0.1, "σ²")  # σ² ~ 0.1·χ²₁
    assert_mean(sigma, math.sqrt(2 * 0.1 / math.pi), "σ")
    T = spec.fixed.T_eff
    var_h = 0.25 + 0.1 * _beta_inv_1m_phi2(a, b)
    e_phi = integrate.quad(
        lambda x: (2 * x - 1) / (4 * x * (1 - x)) * stats.beta.pdf(x, a, b), 0, 1
    )[0]
    cov1 = 0.25 + 0.1 * e_phi
    for t in (0, 1, T // 2, T):  # stationarity: the same law at every date, h_0 included
        assert_mean(h[:, t], 0.0, f"h_{t}")
        assert_var(h[:, t], var_h, f"h_{t}")
    c = np.mean(h[:, T] * h[:, T - 1]) - h[:, T].mean() * h[:, T - 1].mean()
    x = h[:, T] * h[:, T - 1]
    assert abs(c - cov1) < K_SE * x.std(ddof=1) / math.sqrt(N)
    pad = h[:, T + 1 :]  # decoupled N(0, 1) dummies beyond T_eff
    assert pad.shape[1] == spec.static.shapes.T_pad - T
    assert_mean(pad, 0.0, "padding")
    assert_var(pad[:, 0], 1.0, "padding")


def test_random_walk_and_fixed_b0_initial_states():
    rw = draws(J.joint_spec("structural_sv_rw", "full"), seed=4)
    h = rw.eq.vol.core.h[:, 0]
    for t in (0, 1, 10, 50):  # h_0 ~ N(0, 1), increments σ η with E σ² = 0.1
        assert_mean(h[:, t], 0.0, f"rw h_{t}")
        assert_var(h[:, t], 1.0 + 0.1 * t, f"rw h_{t}")
    fb = draws(J.joint_spec("structural_sv_fixed_b0"), seed=5)
    core = fb.eq.vol.core
    h0, h1, phi = core.h[:, 0, 0], core.h[:, 0, 1], core.phi[:, 0]
    assert_var(h0, 0.25 + 2.0 * 0.1, "h_0 = μ + N(0, B₀σ²)")
    a, b = 10.0, 2.0
    e_u, e_u2 = a / (a + b), a * (a + 1) / ((a + b) * (a + b + 1))
    e_phi2 = 4 * e_u2 - 4 * e_u + 1
    assert_var(h1, 0.25 + 0.1 * (2.0 * e_phi2 + 1.0), "h_1")
    assert np.all(np.abs(phi) < 1)


def test_inverse_gamma_sigma2_prior():
    st = draws(J.joint_spec("structural_sv_invgamma"), seed=6)
    s2 = st.eq.vol.core.sigma**2
    log_s2 = np.log(s2.ravel())
    assert_mean(log_s2, math.log(0.4) - special.digamma(5.0), "log σ²")  # σ² ~ IG(5, 0.4)
    assert_var(log_s2, special.polygamma(1, 5.0), "log σ²")
    assert_mean(s2, 0.1, "σ²")


def test_outlier_frequencies_and_moments():
    spec = J.joint_spec("structural_sv_outliers", "full")
    st = draws(spec, seed=7)
    tr = st.eq.vol.trans
    T, K = spec.fixed.T_eff, spec.static.vol.outlier_K
    p = np.exp(tr.log_p)
    np.testing.assert_allclose(np.exp(tr.log_p) + np.exp(tr.log1m_p), 1.0, rtol=1e-12)
    assert_mean(p, 0.1, "p")  # p ~ Beta(2, 18)
    assert_var(p, 2 * 18 / (20**2 * 21), "p")
    o_idx = tr.o_idx[..., :T]
    assert np.all(tr.o_idx[..., T:] == 0)  # padding inert
    assert tr.o_idx.dtype == np.int32 and o_idx.min() >= 0 and o_idx.max() <= K - 1
    share = (o_idx > 0).mean(axis=-1)
    assert_mean(share, 0.1, "share o > 1")  # E[1{o > 1}] = E p
    o = o_idx + 1.0
    assert_mean((o**2).mean(axis=-1), 1 + (151 - 1) * 0.1, "E o²")  # N03 §2.3: 1 + 150 p
    counts = np.bincount(o_idx[o_idx > 0].ravel(), minlength=K)[1:]  # o | o > 1 uniform on 2..K
    assert stats.chisquare(counts).pvalue > 1e-4


@pytest.mark.parametrize("t_scale", ["standardized", "unit"])
def test_student_t_scale(t_scale):
    spec = J.joint_spec("structural_sv_t", "full", t_scale=t_scale)
    st = draws(spec, seed=8, size=10_000)
    tr = st.eq.vol.trans
    T = spec.fixed.T_eff
    grid = np.asarray(J.TEST_NU_GRID, dtype=np.float64)
    counts = np.bincount(tr.nu_idx.ravel(), minlength=grid.size)
    assert counts.size == grid.size and stats.chisquare(counts).pvalue > 1e-4  # uniform ν prior
    nu = grid[tr.nu_idx]
    assert np.all(tr.log_tau[..., T:] == 0.0)
    tau = np.exp(tr.log_tau[..., :T])
    c = nu - 2.0 if t_scale == "standardized" else nu
    e_tau = 1.0 if t_scale == "standardized" else float(np.mean(grid / (grid - 2.0)))
    assert_mean(tau.mean(axis=-1), e_tau, "E τ")  # D-8: standardized E τ = 1
    cg = grid - 2.0 if t_scale == "standardized" else grid
    e_log = float(np.mean(np.log(cg / 2.0) - special.digamma(grid / 2.0)))
    assert_mean(np.log(tau).mean(axis=-1), e_log, "E log τ")
    # given ν, 1/τ ~ Gamma(ν/2, rate c/2): E[1/τ | ν] = ν/c
    assert_mean((1.0 / tau).mean(axis=-1) - (nu / c)[..., None].mean(axis=-1), 0.0, "E[1/τ | ν]")


def test_outlier_t_combined_and_status():
    spec = J.joint_spec("reduced_sv_outliers_t")
    st = draws(spec, seed=9, size=4000)
    T = spec.fixed.T_eff
    tr = st.eq.vol.trans
    assert np.all(tr.o_idx[..., T:] == 0) and np.all(tr.log_tau[..., T:] == 0)
    assert np.all(st.status.code == 0) and np.all(st.status.it == 0)
    assert np.all(np.isinf(st.status.min_pivot))
    for leaf in (tr.rb_out, tr.acc_nu_move, st.eq.vol.core.acc_h, st.eq.vol.core.rb_h_sum):
        assert np.all(leaf == 0)
    assert_mean(np.exp(tr.log_p), 0.1, "p")


@pytest.mark.parametrize(
    "name",
    ["structural_sv_outliers_t", "reduced_sv_outliers_t", "structural_sv_rw",
     "structural_mode2_constant", "zero_sv", "contemp_only"],
)  # fmt: skip
def test_prior_draws_satisfy_state_invariants(name):
    # every name at the small size: test_simulate.py::test_joint_spec_builds_for_every_name
    for size in ("full",):
        spec = J.joint_spec(name, size)
        st = draws(spec, seed=10, size=8)
        bad = []
        for c in range(8):
            one = jax.tree.map(lambda x, c=c: x[c], st)
            bad += check_state_invariants(one, spec.static, spec.fixed.T_eff)
        assert bad == [], (name, size, bad)
        assert st.eq.struct.a.dtype == np.float64


def _infinite(spec: J.JointSpec) -> set[str]:
    return {k for k, fin in spec.function_flags.items() if not fin}


def test_finite_variance_flags():
    sv = J.joint_spec("structural_sv")
    assert sv.function_flags["eq0.core.hT^2"] is False  # E[h⁴] = ∞ under Beta(·, 2) (N06 §3.4)
    assert "eq0.core.hT^2" not in sv.names and "eq0.core.hT^2" in sv.all_names
    assert all(sv.function_flags[k] for k in sv.names)
    assert not any(k.startswith("data.mean_y2") for k in sv.all_names)  # no raw moments under SV
    const = J.joint_spec("structural_constant")  # IG(5, 4): E λ² < ∞, raw moments valid
    assert const.raw_moments and "data.mean_y2[0]" in const.names
    # raw data moments need E y⁴ < ∞, i.e. E λ² < ∞: IG shape > 2 (N06 §3.4)
    mid = J.joint_spec("structural_constant", prior_kwargs={"ig_shape": 3.0})
    assert mid.raw_moments and "data.mean_y2[0]" in mid.names
    weak = J.joint_spec("structural_constant", prior_kwargs={"ig_shape": 2.0})
    assert not weak.raw_moments and "data.mean_y2[0]" not in weak.all_names
    forced = J.joint_spec("structural_constant", prior_kwargs={"ig_shape": 2.0}, raw_moments=True)
    assert _infinite(forced) == {k for k in forced.all_names if k.startswith(
        ("data.mean_y2", "data.acov1_y", "data.mean_yy"))} != set()  # fmt: skip
    tr = J.joint_spec("constant_transitory")  # o, τ make y² heavy-tailed: no raw moments
    assert not tr.raw_moments
    for name in ("structural_mode2_constant", "structural_iw_equivalent"):  # √λ in the lags
        assert not J.joint_spec(name).raw_moments
    assert J.joint_spec("reduced_constant").raw_moments
    for name in ("structural_sv_outliers_t", "reduced_sv_t"):
        spec = J.joint_spec(name)
        assert "loglik" in spec.names and "log_prior" in spec.names
        assert any(k.endswith("loglik_marginal") for k in spec.names)
        assert len(set(spec.names)) == len(spec.names)


@pytest.mark.parametrize("size", ["small", "full"])
def test_default_test_priors_flag_only_h_squared(size):
    """At the DESIGN §9.2 test priors every function has finite variance except ``h_T²`` under the
    stationary SV prior with ``b_φ = 2``; fixed ``B₀`` keeps ``Var(h_t | θ) ≤ σ²(B₀ + t)``."""
    for name in J.COMPOSITIONS + J.BLOCK_CONFIGS:
        spec = J.joint_spec(name, size)
        vol = spec.static.vol
        stationary = vol.persistent == "sv" and not vol.rw and vol.h0 == "stationary"
        expect = {k for k in spec.all_names if k.endswith(".core.hT^2")} if stationary else set()
        assert _infinite(spec) == expect, name
        assert (
            "joint.gamma_minus_AM_sq" in spec.names or spec.static.mean.kind != "linear_structural"
        )


def test_flags_follow_the_prior_tails():
    """Prior-aware flags (N06 §3.4): ``λ``-scaled coefficients need ``E λ`` (values) and ``E λ²``
    (squares, products within an equation); ``σ`` needs ``E σ²``, ``σ²`` needs ``E σ⁴``; ``h`` needs
    ``E[σ²/(1 − φ²)]`` (stationary) or ``E σ²`` (fixed ``B₀``, random walk)."""
    m2 = J.joint_spec("structural_mode2_constant", prior_kwargs={"ig_shape": 2.0})
    assert m2.tails == R.PriorTails(lam=2.0, sigma2=math.inf, h2=math.inf)
    sq = {k for k in m2.all_names if k.endswith("^2") and (".mean.delta" in k or ".struct.a" in k)}
    assert _infinite(m2) == sq | {"eq1.joint.a0*gamma_lag1_0", "joint.gamma_minus_AM_sq"}
    m1 = J.joint_spec("structural_mode2_constant", prior_kwargs={"ig_shape": 1.0})
    lin = {k for k in m1.all_names if ".mean.delta" in k or ".struct.a" in k or "gamma" in k}
    assert _infinite(m1) == lin | {"eq0.joint.B_own1", "eq1.joint.B_own1"}
    assert {"eq0.core.log_lam^2", "loglik", "log_prior", "data.mean_log_y2[0]"} <= set(m1.names)
    iw = J.joint_spec("structural_iw_equivalent", prior_kwargs={"ig_shape": 2.0})  # a scaled, δ not
    assert "eq1.struct.a0^2" not in iw.names and "eq1.joint.a0*gamma_lag1_0" not in iw.names
    assert "eq1.mean.delta1^2" in iw.names and "joint.gamma_minus_AM_sq" in iw.names
    assert "eq1.joint.B_own1" in iw.names and "eq2.joint.B_own1" in iw.names  # E λ suffices
    m1_mode1 = J.joint_spec("structural_constant", prior_kwargs={"ig_shape": 1.0})  # unscaled
    assert _infinite(m1_mode1) == set()

    ig2 = J.joint_spec("structural_sv_invgamma", prior_kwargs={"c0": 2.0})
    assert ig2.tails.sigma2 == 2.0 and ig2.tails.h2 == 2.0
    assert _infinite(ig2) == {f"eq{i}.core.{f}" for i in range(2) for f in ("sigma^2", "hT^2")}
    ig1 = J.joint_spec("structural_sv_invgamma", prior_kwargs={"c0": 1.0})
    core = ("sigma", "sigma^2", "phi*sigma", "h0", "h1", "hmid", "hT", "mean_h", "mean_abs_h",
            "mu*hT", "hT^2")  # fmt: skip
    expect = {f"eq{i}.core.{f}" for i in range(2) for f in core} | {
        "loglik", "eq0.joint.loglik", "eq1.joint.loglik", "data.mean_log_y2[0]",
        "data.mean_log_y2[1]"}  # fmt: skip
    assert _infinite(ig1) == expect
    assert {"log_prior", "eq0.joint.rss_std", "data.corr_y[0,1]", "eq0.core.mu"} <= set(ig1.names)
    phi1 = J.joint_spec("structural_sv_outliers", prior_kwargs={"b_phi": 1.0})  # E (1−φ)⁻¹ = ∞
    assert phi1.tails.h2 == 1.0 and "eq0.core.hT" not in phi1.names
    assert "eq0.joint.loglik_marginal" not in phi1.names and "eq0.core.sigma^2" in phi1.names
    assert "eq0.trans.p" in phi1.names
    assert J.joint_spec("structural_sv", prior_kwargs={"a_phi": 1.0}).tails.h2 == 1.0
    assert "eq0.core.hT^2" in J.joint_spec("structural_sv", prior_kwargs={"b_phi": 4.0}).names
    b0 = J.joint_spec("structural_sv_fixed_b0", prior_kwargs={"b_phi": 1.0})  # φ-free bound
    assert b0.tails.h2 == math.inf and _infinite(b0) == set()
    rw = J.joint_spec("structural_sv_rw", sigma2_prior="inv_gamma", prior_kwargs={"c0": 2.0})
    assert _infinite(rw) == {f"eq{i}.core.{f}" for i in range(2) for f in ("sigma^2", "sigma*hT")}
    # a reference called without tails assumes nothing about the prior
    st = J.joint_spec("structural_sv")
    part = jax.tree.map(lambda x: x[0], st.prior_sample(random.key(0)).eq.vol.core)
    fns = R.reference("sv", st.static).test_functions_eq(part, st.tmask)
    assert {k for k, (_, fin) in fns.items() if not fin} == set(core)
    with pytest.raises(ValueError, match="concrete prior"):
        jax.jit(lambda pr: R.prior_tails(st.static, pr).h2)(st.prior)
