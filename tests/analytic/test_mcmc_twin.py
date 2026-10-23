"""WP20: analytic twins of the MCMC engine at scale, A2–A4 (DESIGN §9.8, §6.1, D-10, D-12; N05 §4.2,
§9, §13.2(b), §13.5). A5 (Mode I quadrature) lives in ``test_mode1_quadrature.py``.

**A2/A3: Mode II × ``Constant("iw")`` vs the Phase 1 NIW posterior.** With ``κ_c = κ_a = 1`` the
structural Mode II prior *is* the Phase 1 NIW prior (N05 §4.2; A1 pins the marginal likelihood),
so the MCMC posterior of ``(B, Σ, A, Λ)`` must equal the Phase 1 posterior ``Σ | Y ~ IW(S̄, ν̄)``,
``B | Σ, Y ~ MN(B̄, Ω̄, Σ)`` (``AnalyticFit.posterior``) at the same hyperparameters. Exact
marginals (``n`` variables, ``ν' = ν̄ − n + 1``):

* ``B_rc ~ t_{ν'}(B̄_rc, Ω̄_rr S̄_cc/ν')`` and ``Σ_cc ~ IG(ν'/2, S̄_cc/2)``;
* ``A`` and ``Λ`` from ``AΣAᵀ = Λ`` (``A`` unit lower triangular, so ``−a_i`` regresses ``u_i`` on
  ``u_{<i}``) by the inverse-Wishart partition theorem applied to the leading ``(i+1)``-block
  ``Σ_{≤i} ~ IW(S̄_{≤i}, ν̄ − n + i + 1)``: ``λ_i = Σ_{ii·<i} ~ IG(ν_i/2, S̄_{ii·<i}/2)`` and
  ``−a_i | λ_i ~ N(S̄_{<i,<i}⁻¹S̄_{<i,i}, λ_i S̄_{<i,<i}⁻¹)``, hence ``a_ij ~ t_{ν_i}`` with
  ``ν_i = ν̄ − n + i + 1`` (a test pins these to Chan's per-equation closed form, WP16);
* ``Σ_cd`` off the diagonal has no closed-form quantiles: iid draws from the Phase 1 sampler
  (``linalg.sample_inverse_wishart_chol``), whose sampling error enters the denominator.

**A4: Mode II with ``κ_c = 0.5``** (Chan's 2022 asymmetric conjugate prior, not an NIW) vs the
closed form of ``engines.structural_nig``: exact ``t_{2ν̂}`` marginals of ``γ`` and ``a``,
``IG(ν̂, Ŝ)`` of ``λ``, and iid draws (``draw_structural_nig``) for ``B = (A⁻¹Γ)ᵀ`` and ``Σ``.

**Comparison** (DESIGN §9.8, "4 quantile-MCSE, Holm"): the 5/25/50/75/95% quantiles ``x_q`` (as
in the parity runs, §9.9), each tested in its CDF form

    z = (F̂_MCMC(x_q) − q)/√(MCSE²(F̂_MCMC(x_q)) + q(1 − q)/N_ref),

where ``F̂_MCMC(x_q)`` is the mean of the indicator ``1{θ ≤ x_q}`` and its MCSE comes from that
indicator's ESS (Vehtari et al. 2021 §4.3, ``diagnostics.mcse_mean_many``; never the bulk ESS of
``θ``, N06 pitfall), ``x_q`` the exact quantile (``N_ref = ∞``) or the sample quantile of
``N_ref`` iid reference draws. To first order this is ``(q̂_MCMC − x_q)/MCSE(q̂)`` (delta method)
with the same indicator ESS, i.e. the quantile-MCSE comparison of DESIGN §9.8. The CDF form avoids
the density estimate of the order-statistic MCSE (``diagnostics.mcse_quantile_many``: the spacing
of ~40 draws at a 5% tail with 10⁴ draws), whose noise makes ``(q̂ − x_q)/MCSE_q`` heavy-tailed:
with ~8,000 quantiles at ``n = 20`` it gave 2–3 values beyond 4 (0.5 expected) and one at −4.99
(seed 221), whereas the CDF form matched ``N(0, 1)`` (variance 0.91–1.05, no excess kurtosis) on
the same draws. Decision: every ``|z| ≤ max(4, z_Bonf)``, ``z_Bonf`` the two-sided Bonferroni
critical value at family level 1e-2 (N06 §3.5: Holm equals Bonferroni for the global decision), so
the ~165 quantiles of A2 are held to 4 MCSE and the ~10⁴ of A3 keep a family-wise false-alarm rate
≤ 1%; Holm lists the failures. Negative controls show the PR-tier comparison has power (one extra
degree of freedom of the NIW posterior, ~1.5% on ``Σ``, fails it).

Cadence (DESIGN §9.10): A2 at ``n = 3, p = 2, T = 60`` on the production ``fixed_factor`` path per
PR; the other constant-weight paths, A3 (``n = 20, p = 4, T = 300``, the SPEC §10.3 realistic-size
oracle) and A4 nightly (``slow``). Every fit uses the default EB plug-in (DESIGN §6.5): the
conjugate twin at ``κ_c = 1`` (equal to the Phase 1 EB mode) and Chan's structural EB at
``κ_c = 0.5``; the oracles are evaluated at the fit's own ``hyper``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.linalg import solve_triangular

import rai_bvar as rb
from rai_bvar import diagnostics
from rai_bvar.data.design import regressor_names
from rai_bvar.engines.analytic import NIWPosterior
from rai_bvar.engines.structural_nig import (
    NIGPosterior,
    draw_structural_nig,
    structural_posteriors,
    structural_to_reduced,
)
from rai_bvar.linalg import sample_inverse_wishart_chol
from rai_bvar.validation.sequential import holm_reject

pytestmark = [
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
    # structural compositions with many variables say so (R8); expected here
    pytest.mark.filterwarnings("ignore:this composition is not order-invariant:UserWarning"),
]

LAGS, DET = 2, "c"
Q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
BOUND = 4.0  # quantile-MCSE units (DESIGN §9.8)
ALPHA = 1e-2  # family level once 4 MCSE stops controlling it (N06 §3.5)
PR_SETTINGS = rb.MCMCSettings(chains=4, draws=10_000, burn=500, store_max=10_000, seed=220)
NIGHTLY_SETTINGS = rb.MCMCSettings(chains=4, draws=2_500, burn=500, store_max=2_500, seed=221)
_IID_CHUNK = 10_000


# --------------------------------------------------------------------------- data and fits


@pytest.fixture(scope="module")
def data3(sim_data_stationary) -> rb.VARData:
    """``n = 3``, ``p = 2``, ``T_eff = 60`` (not a multiple of ``t_bucket = 32``: masks exercised)."""
    return sim_data_stationary.tail(60 + LAGS)


def var_data(n: int, T: int, seed: int) -> rb.VARData:
    """A stationary VAR(1) sample (tcode 1) with levels −2…5, scales 0.1…10 and correlated shocks."""
    rng = np.random.default_rng(seed)
    A = np.diag(np.linspace(0.9, 0.2, n)) + 0.02 * rng.standard_normal((n, n))
    assert np.abs(np.linalg.eigvals(A)).max() < 0.95
    sd = np.geomspace(0.1, 10.0, n)
    chol = np.linalg.cholesky(0.3 * np.ones((n, n)) + 0.7 * np.eye(n)) * sd[:, None]
    mean = np.linspace(-2.0, 5.0, n)
    const = (np.eye(n) - A) @ mean
    y, out = mean.copy(), np.empty((T, n))
    for t in range(T + 100):
        y = const + A @ y + chol @ rng.standard_normal(n)
        if t >= 100:
            out[t - 100] = y
    index = pd.period_range("1930Q1", periods=T, freq="Q")  # ends before 2020
    columns = [f"x{j:02d}" for j in range(n)]
    return rb.VARData.from_frame(pd.DataFrame(out, index=index, columns=columns))


def twin_model(settings: rb.MCMCSettings, lags: int = LAGS, cross: float = 1.0) -> rb.BVAR:
    """Structural ``Linear × Constant("iw")``: Mode II, the analytic twin (D-10)."""
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="structural", cross=cross)),
        vol=rb.Constant("iw"),
        lags=lags,
        deterministic=DET,
        mcmc=settings,
    )


def run_fit(model: rb.BVAR, data: rb.VARData, const_path: str = "auto"):
    """``model.fit(data)`` on the resolved path (``fixed_factor`` for Mode II × Constant), or on
    ``"cholesky_const"`` (``MCMCSettings(const_path="cholesky")``) or ``"weighted"`` (the general
    weighted-Gram path, reference oracle of the fast paths, DESIGN §3.7.2; engine-level
    ``fit_mcmc`` on the resolved inputs)."""
    if const_path == "auto":
        return model.fit(data)
    if const_path == "cholesky_const":
        cholesky = dataclasses.replace(model.mcmc, const_path="cholesky")
        return dataclasses.replace(model, mcmc=cholesky).fit(data)
    assert const_path == "weighted"
    from rai_bvar.engines.mcmc import resolve
    from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc

    runtime = rb.Runtime()
    r = resolve.resolve(model, data, runtime)
    mean = dataclasses.replace(r.static.mean, const_path="weighted")
    static = dataclasses.replace(r.static, mean=mean)
    meta = FitMeta(
        model=model,
        vardata=data,
        periods=r.periods,
        variables=tuple(data.variables),
        hyper=r.hyper,
        notes=r.notes,
        eb_converged=r.eb_converged,
    )
    return fit_mcmc(
        static, r.data, r.prior, model.mcmc, runtime, seed=model.mcmc.seed, salt=r.salt, meta=meta
    )


def phase1_fit(data: rb.VARData, lags: int):
    """The Phase 1 conjugate fit (``AnalyticFit``) with the default EB hierarchy."""
    model = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota()), lags=lags, deterministic=DET)
    assert model.engine == "analytic"
    return model.fit(data)


def plug_in(fit) -> dict:
    """The hyperparameters a fit used, as a Phase 1 / WP16 hyperparameter dict."""
    return {key: fit.hyper[key] for key in ("lambda", "alpha", "psi")}


# --------------------------------------------------------------------------- draws


def structural_draws(fit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``a`` (C, D, n, n−1), ``gamma`` (C, D, n, k), ``lam`` (C, D, n) of the stored draws."""
    p = fit.param_draws
    a = np.asarray(p.struct.a)
    gamma = np.asarray(p.mean.gamma)
    lam = np.exp(np.asarray(p.vol.core.log_lam))
    return a, gamma, lam


def sigma_draws(a: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """``Σ = A⁻¹ΛA⁻ᵀ`` per draw (``A = I + a`` below the diagonal), by a dense solve."""
    n = lam.shape[-1]
    rows, cols = np.tril_indices(n, -1)
    A = np.broadcast_to(np.eye(n), (*lam.shape[:-1], n, n)).copy()
    A[..., rows, cols] = a[..., rows, cols]
    C = np.linalg.solve(A, np.sqrt(lam)[..., None, :] * np.eye(n))  # A⁻¹Λ^{½}
    return C @ np.swapaxes(C, -1, -2)


def pair_names(variables: tuple[str, ...]) -> list[str]:
    rows, cols = np.tril_indices(len(variables), -1)
    return [f"{variables[i]}~{variables[j]}" for i, j in zip(rows, cols, strict=True)]


# --------------------------------------------------------------------------- comparison


class Report:
    """z-scores of quantile comparisons in their CDF form (module docstring) and their names."""

    def __init__(self) -> None:
        self.z: list[np.ndarray] = []
        self.names: list[str] = []

    def at(
        self, label: str, names: list[str], x: np.ndarray, points: np.ndarray, n_ref: float
    ) -> None:
        """``z = (F̂(x_q) − q)/√(MCSE² + q(1 − q)/n_ref)`` for MCMC draws ``x`` (C, D, m) at the
        reference quantiles ``points`` (Q, m) (``n_ref = ∞``: exact quantiles)."""
        m = len(names)
        assert x.shape[-1] == m and points.shape == (Q.size, m)
        z = np.empty((Q.size, m))
        for qi, q in enumerate(Q):
            ind = (x <= points[qi]).astype(np.float64)
            var = diagnostics.mcse_mean_many(ind) ** 2 + q * (1.0 - q) / n_ref
            z[qi] = (ind.mean(axis=(0, 1)) - q) / np.sqrt(var)
        self.z.append(z.ravel())
        self.names += [f"{label}[{nm}]@q{q:g}" for q in Q for nm in names]

    def exact(self, label: str, names: list[str], x: np.ndarray, exact: np.ndarray) -> None:
        """MCMC draws ``x`` (C, D, m) against exact quantiles ``exact`` (Q, m)."""
        self.at(label, names, x, exact, np.inf)

    def iid(self, label: str, names: list[str], x: np.ndarray, ref: np.ndarray) -> None:
        """MCMC draws ``x`` (C, D, m) against the type-7 quantiles of iid draws ``ref`` (N, m),
        whose CDF at its own sample quantile has variance ``q(1 − q)/N``."""
        self.at(label, names, x, np.quantile(ref, Q, axis=0), ref.shape[0])

    def check(self) -> None:
        assert_within_mcse(np.concatenate(self.z), self.names)


def assert_within_mcse(z: np.ndarray, names: list[str]) -> None:
    """Every ``|z| ≤ max(4, z_Bonf(α = 1e-2))`` (module docstring); Holm lists the failures."""
    z = np.asarray(z, dtype=np.float64).ravel()
    assert z.size == len(names)
    G = z.size
    crit = max(BOUND, float(stats.norm.isf(ALPHA / (2.0 * G))))
    p = 2.0 * stats.norm.sf(np.abs(z))
    bad = ~(np.abs(z) <= crit)  # NaN fails
    if bad.any():
        holm = holm_reject(p, min(ALPHA, 2.0 * G * float(stats.norm.sf(crit))))
        rows = [
            f"{names[g]}: z = {z[g]:+.2f}{' (Holm)' if holm[g] else ''}"
            for g in np.argsort(-np.abs(np.nan_to_num(z, nan=np.inf)))
            if bad[g]
        ]
        raise AssertionError(
            f"{bad.sum()} of {G} quantiles outside {crit:.2f} MCSE:\n" + "\n".join(rows[:40])
        )


# --------------------------------------------------------------------------- oracles


@dataclass(frozen=True)
class StructuralMarginals:
    """Exact marginals of ``A`` and ``Λ``: ``a_ij ~ t_dof(loc, scale)`` over the pairs ``i > j``
    (``np.tril_indices(n, −1)`` order) and ``λ_i ~ IG(lam_shape, lam_scale)``."""

    a_loc: np.ndarray
    a_scale: np.ndarray
    a_dof: np.ndarray
    lam_shape: np.ndarray
    lam_scale: np.ndarray

    def a_quantiles(self) -> np.ndarray:
        return stats.t.ppf(Q[:, None], self.a_dof, loc=self.a_loc, scale=self.a_scale)

    def lam_quantiles(self) -> np.ndarray:
        return stats.invgamma.ppf(Q[:, None], self.lam_shape, scale=self.lam_scale)


def niw_structural(post: NIWPosterior) -> StructuralMarginals:
    """``A``, ``Λ`` marginals of ``Σ ~ IW(S̄, ν̄)`` by the partition theorem (module docstring)."""
    S, nu = post.scale, post.df
    n = S.shape[0]
    rows, _ = np.tril_indices(n, -1)
    loc, scale, dof = (np.empty(rows.size) for _ in range(3))
    lam_shape, lam_scale = np.empty(n), np.empty(n)
    for i in range(n):
        S11, s12 = S[:i, :i], S[:i, i]
        beta = np.linalg.solve(S11, s12) if i else np.zeros(0)
        s22 = S[i, i] - s12 @ beta
        nu_i = nu - n + i + 1.0
        lam_shape[i], lam_scale[i] = nu_i / 2.0, s22 / 2.0
        if i:
            sel = rows == i
            diag_inv = np.diag(np.linalg.solve(S11, np.eye(i)))
            loc[sel], scale[sel], dof[sel] = -beta, np.sqrt(s22 / nu_i * diag_inv), nu_i
    return StructuralMarginals(loc, scale, dof, lam_shape, lam_scale)


@dataclass(frozen=True)
class NIGMarginals:
    """Exact marginals of Chan's per-equation posterior ``λ_i ~ IG(ν̂, Ŝ)``, ``θ_i | λ_i ~ N(θ̂,
    λ_i K⁻¹)`` in γ-coordinates: ``θ_ir ~ t_{2ν̂}(θ̂_r, (Ŝ/ν̂)(K⁻¹)_rr)``."""

    structural: StructuralMarginals
    gamma_loc: np.ndarray  # (n, k)
    gamma_scale: np.ndarray  # (n, k)
    gamma_dof: np.ndarray  # (n,)

    def gamma_quantiles(self) -> np.ndarray:
        """(Q, n, k)."""
        return stats.t.ppf(
            Q[:, None, None],
            self.gamma_dof[:, None],
            loc=self.gamma_loc,
            scale=self.gamma_scale,
        )


def nig_marginals(posts: tuple[NIGPosterior, ...]) -> NIGMarginals:
    n = len(posts)
    k = posts[0].theta_hat.size
    rows, _ = np.tril_indices(n, -1)
    loc, scale, dof = (np.empty(rows.size) for _ in range(3))
    g_loc, g_scale = np.empty((n, k)), np.empty((n, k))
    g_dof, lam_shape, lam_scale = np.empty(n), np.empty(n), np.empty(n)
    for i, post in enumerate(posts):
        R_inv = solve_triangular(post.R, np.eye(k + i), lower=False)
        sd = np.sqrt(post.S_hat / post.nu_hat * np.sum(R_inv**2, axis=1))
        g_loc[i], g_scale[i], g_dof[i] = post.theta_hat[:k], sd[:k], 2.0 * post.nu_hat
        lam_shape[i], lam_scale[i] = post.nu_hat, post.S_hat
        if i:
            sel = rows == i
            loc[sel], scale[sel], dof[sel] = post.theta_hat[k:], sd[k:], 2.0 * post.nu_hat
    return NIGMarginals(
        StructuralMarginals(loc, scale, dof, lam_shape, lam_scale), g_loc, g_scale, g_dof
    )


def niw_iid_offdiag(post: NIWPosterior, size: int, seed: int) -> np.ndarray:
    """iid ``Σ_ij`` (i > j) from the Phase 1 inverse-Wishart sampler: (size, n(n−1)/2)."""
    n = post.scale.shape[0]
    rows, cols = np.tril_indices(n, -1)
    rng = np.random.default_rng(seed)
    out = np.empty((size, rows.size))
    for start in range(0, size, _IID_CHUNK):
        m = min(_IID_CHUNK, size - start)
        L = sample_inverse_wishart_chol(post.scale_chol, post.df, m, rng)
        out[start : start + m] = (L @ np.swapaxes(L, -1, -2))[:, rows, cols]
    return out


def compare_niw(fit, post: NIWPosterior, *, n_iid: int, seed: int) -> Report:
    """A2/A3: quantiles of ``B``, ``Σ``, ``A`` and ``Λ`` of ``fit`` against the NIW ``post``."""
    variables = tuple(fit.ordering)
    n = len(variables)
    s = fit.static.shapes
    regs = regressor_names(variables, s.lags, DET)
    report = Report()
    S, nu = post.scale, post.df
    dof = nu - n + 1.0
    B = fit.reduced_form_draws().values  # (C, D, k, n)
    omega = np.diag(post.omega())
    for c, v in enumerate(variables):  # one column at a time bounds the memory at n = 20
        exact = stats.t.ppf(
            Q[:, None], dof, loc=post.mean[:, c], scale=np.sqrt(omega * S[c, c] / dof)
        )
        report.exact("B", [f"{r}, {v}" for r in regs], B[..., c], exact)
    del B
    a, _, lam = structural_draws(fit)
    sigma = sigma_draws(a, lam)
    diag = np.arange(n)
    report.exact(
        "Sigma",
        [f"{v}, {v}" for v in variables],
        sigma[..., diag, diag],
        stats.invgamma.ppf(Q[:, None], dof / 2.0, scale=np.diag(S) / 2.0),
    )
    rows, cols = np.tril_indices(n, -1)
    pairs = pair_names(variables)
    report.iid("Sigma", pairs, sigma[..., rows, cols], niw_iid_offdiag(post, n_iid, seed))
    marg = niw_structural(post)
    report.exact("a", pairs, a[..., rows, cols], marg.a_quantiles())
    report.exact("lambda", list(variables), lam, marg.lam_quantiles())
    return report


def compare_nig(fit, posts: tuple[NIGPosterior, ...], *, n_iid: int, seed: int) -> Report:
    """A4: quantiles of ``γ``, ``A``, ``Λ`` (exact) and ``B``, ``Σ`` (iid) against Chan's closed
    form ``posts`` (γ-coordinates)."""
    variables = tuple(fit.ordering)
    n = len(variables)
    s = fit.static.shapes
    regs = regressor_names(variables, s.lags, DET)
    rows, cols = np.tril_indices(n, -1)
    pairs = pair_names(variables)
    marg = nig_marginals(posts)
    report = Report()
    a, gamma, lam = structural_draws(fit)
    g_exact = marg.gamma_quantiles()
    for i, v in enumerate(variables):
        report.exact("gamma", [f"{v}, {r}" for r in regs], gamma[..., i, :], g_exact[:, i])
    report.exact("a", pairs, a[..., rows, cols], marg.structural.a_quantiles())
    report.exact("lambda", list(variables), lam, marg.structural.lam_quantiles())
    sigma = sigma_draws(a, lam)
    del gamma
    rng = np.random.default_rng(seed)
    a_ref, g_ref, lam_ref = draw_structural_nig(posts, n_iid, rng)
    coefs, chol = structural_to_reduced(a_ref, g_ref, lam_ref)  # (N, k, n), (N, n, n)
    del g_ref
    B = fit.reduced_form_draws().values
    for c, v in enumerate(variables):
        report.iid("B", [f"{r}, {v}" for r in regs], B[..., c], coefs[..., c])
    del B, coefs
    sigma_ref = chol @ np.swapaxes(chol, -1, -2)
    low, high = np.tril_indices(n)
    names = [f"{variables[i]}, {variables[j]}" for i, j in zip(low, high, strict=True)]
    report.iid("Sigma", names, sigma[..., low, high], sigma_ref[:, low, high])
    return report


# --------------------------------------------------------------------------- the oracles themselves


def test_niw_structural_marginals_equal_the_nig_closed_form(data3):
    """The partition-theorem marginals of ``A`` and ``Λ`` under the Phase 1 posterior equal the
    per-equation marginals of Chan's closed form at ``κ_c = κ_a = 1`` (N05 §4.2: the same
    posterior; two independent derivations)."""
    hyper = {"lambda": 0.3, "alpha": 2.0}
    post = phase1_fit(data3, LAGS).posterior(hyper)
    prior = rb.Minnesota(form="structural", hierarchy="fixed")
    posts = structural_posteriors(data3, prior, hyper, LAGS, DET)
    niw, nig = niw_structural(post), nig_marginals(posts).structural
    np.testing.assert_allclose(niw.a_loc, nig.a_loc, rtol=1e-9)
    np.testing.assert_allclose(niw.a_scale, nig.a_scale, rtol=1e-9)
    np.testing.assert_array_equal(niw.a_dof, nig.a_dof)
    np.testing.assert_array_equal(niw.lam_shape, nig.lam_shape)
    np.testing.assert_allclose(niw.lam_scale, nig.lam_scale, rtol=1e-10)
    # ν̄ = d + T with d = n + 2, so ν_i = T + i + 3 = 2ν̂_i (ν̂_i = 1 + ι/2 + T/2)
    np.testing.assert_array_equal(niw.lam_shape * 2.0, 60.0 + np.arange(3) + 3.0)


def test_niw_exact_marginals_match_the_phase1_sampler(data3):
    """The exact marginals used by A2 against iid draws of the Phase 1 sampler
    (``NIWPosterior.draw``): ``B`` and ``Σ_cc`` (t and IG degrees of freedom), and ``A``, ``Λ``
    of the LDL factors of the drawn ``Σ``."""
    post = phase1_fit(data3, LAGS).posterior({"lambda": 0.3})
    n_draws, n = 100_000, 3
    coefs, chols = post.draw(n_draws, np.random.default_rng(7))

    def as_chains(x: np.ndarray) -> np.ndarray:  # iid draws viewed as 4 "chains"
        return x.reshape(4, n_draws // 4, -1)

    S, dof = post.scale, post.df - n + 1.0
    report = Report()
    exact_B = stats.t.ppf(
        Q[:, None, None],
        dof,
        loc=post.mean,
        scale=np.sqrt(np.outer(np.diag(post.omega()), np.diag(S)) / dof),
    )
    report.exact(
        "B", [str(g) for g in range(coefs[0].size)], as_chains(coefs), exact_B.reshape(Q.size, -1)
    )
    sigma = chols @ np.swapaxes(chols, -1, -2)
    exact_S = stats.invgamma.ppf(Q[:, None], dof / 2.0, scale=np.diag(S) / 2.0)
    report.exact("Sigma", ["0", "1", "2"], as_chains(sigma[:, np.arange(n), np.arange(n)]), exact_S)
    lam = np.diagonal(chols, axis1=-2, axis2=-1) ** 2  # Σ = (A⁻¹Λ^{½})(·)ᵀ = LLᵀ: Λ^{½} = diag L
    A = np.linalg.solve(chols / np.sqrt(lam)[:, None, :], np.eye(n))  # L = A⁻¹Λ^{½}
    rows, cols = np.tril_indices(n, -1)
    marg = niw_structural(post)
    report.exact("a", ["10", "20", "21"], as_chains(A[:, rows, cols]), marg.a_quantiles())
    report.exact("lambda", ["0", "1", "2"], as_chains(lam), marg.lam_quantiles())
    report.check()


def test_assert_within_mcse_rule():
    """4 MCSE for small families, the Bonferroni critical value at 1e-2 for large ones; NaN
    fails."""
    assert_within_mcse(np.array([3.9, -3.99]), ["a", "b"])
    with pytest.raises(AssertionError, match=r"b: z = -4\.10"):
        assert_within_mcse(np.array([0.0, -4.1]), ["a", "b"])
    big = np.zeros(10_000)
    big[0] = 4.5  # below the Bonferroni critical value 4.89 at G = 10⁴
    assert_within_mcse(big, [str(g) for g in range(big.size)])
    with pytest.raises(AssertionError, match="nan"):
        assert_within_mcse(np.array([np.nan]), ["nan"])


# --------------------------------------------------------------------------- A2–A4


def check_a2(fit, data: rb.VARData, lags: int, *, n_iid: int, seed: int) -> None:
    """Mode II × ``Constant("iw")`` fit against the Phase 1 NIW posterior at the fit's hyper;
    the EB plug-in is the Phase 1 EB mode (DESIGN §6.5 step 1)."""
    assert fit.metadata["prior_mode"] == "II"
    assert fit.converged, fit.warnings
    phase1 = phase1_fit(data, lags)
    np.testing.assert_allclose(fit.hyper["lambda"], phase1.hyper["lambda"], rtol=1e-8)
    np.testing.assert_allclose(fit.hyper["psi"], phase1.hyper["psi"], rtol=1e-12)
    post = phase1.posterior(plug_in(fit))
    compare_niw(fit, post, n_iid=n_iid, seed=seed).check()


@pytest.fixture(scope="module")
def a2_fit(data3):
    """The PR-tier A2 fit: Mode II × ``Constant("iw")``, EB plug-in, ``n = 3, p = 2, T = 60``."""
    return run_fit(twin_model(PR_SETTINGS), data3)


def test_a2_mode2_iw_matches_the_phase1_niw_posterior(data3, a2_fit):
    """A2 (PR): MCMC structural Mode II × ``Constant("iw")`` on its production ``fixed_factor``
    path (D-26), ``n = 3, p = 2, T = 60``: quantiles of ``B``, ``Σ``, ``A``, ``Λ`` within 4
    quantile-MCSE of the Phase 1 NIW posterior."""
    assert a2_fit.metadata["const_path"] == "fixed_factor"
    check_a2(a2_fit, data3, LAGS, n_iid=200_000, seed=2)


def test_a2_rejects_a_slightly_wrong_posterior(data3, a2_fit):
    """Negative control (power): the same draws against the NIW posterior with one extra degree
    of freedom (``ν̄ + 1``: ~1.5% on the scale of ``Σ``; mutant M13's padded-row count adds
    ``T_pad − T_eff = 4``) fail the comparison."""
    post = phase1_fit(data3, LAGS).posterior(plug_in(a2_fit))
    wrong = dataclasses.replace(post, df=post.df + 1.0)
    with pytest.raises(AssertionError, match="quantiles outside"):
        compare_niw(a2_fit, wrong, n_iid=200_000, seed=2).check()


@pytest.mark.slow
@pytest.mark.parametrize("const_path", ["weighted", "cholesky_const"])
def test_a2_other_constant_weight_paths(data3, const_path):
    """A2 (nightly) on the general weighted-Gram path and the constant-weight Cholesky path."""
    fit = run_fit(twin_model(PR_SETTINGS), data3, const_path)
    assert fit.metadata["const_path"] == const_path
    check_a2(fit, data3, LAGS, n_iid=200_000, seed=3)


@pytest.mark.slow
def test_a3_realistic_size():
    """A3 (nightly; SPEC §10.3 realistic-size oracle): A2 at ``n = 20, p = 4, T = 300``."""
    lags = 4
    data = var_data(20, 300 + lags, seed=2030)
    fit = run_fit(twin_model(NIGHTLY_SETTINGS, lags=lags), data)
    assert fit.metadata["const_path"] == "fixed_factor"
    check_a2(fit, data, lags, n_iid=40_000, seed=4)


@pytest.mark.slow
@pytest.mark.parametrize(("n", "lags", "T"), [(3, 2, 60), (20, 4, 300)], ids=["n3", "n20"])
def test_a4_chan_asymmetric_prior(data3, n, lags, T):
    """A4 (nightly): Mode II with ``κ_c = 0.5`` (Chan's 2022 asymmetric conjugate prior, D3
    centring) against the closed form of ``engines.structural_nig`` at the Chan EB plug-in
    (DESIGN §6.5 step 2): exact ``γ``, ``A``, ``Λ`` marginals and iid ``B``, ``Σ``."""
    data = data3 if n == 3 else var_data(n, T + lags, seed=2040)
    settings = PR_SETTINGS if n == 3 else NIGHTLY_SETTINGS
    model = twin_model(settings, lags=lags, cross=0.5)
    fit = run_fit(model, data)
    assert fit.metadata["prior_mode"] == "II" and fit.metadata["const_path"] == "fixed_factor"
    assert fit.hyper["eb_method"] == "structural_nig"
    assert fit.converged, fit.warnings
    posts = structural_posteriors(data, model.mean.prior, plug_in(fit), lags, DET)
    n_iid = 200_000 if n == 3 else 8_000
    compare_nig(fit, posts, n_iid=n_iid, seed=5).check()


# --------------------------------------------------------------------------- WP25: SOC/SUR dummies


DUMMIES = {"soc": 1.0, "sur": 1.0}  # starting values; λ, soc and sur are EB-free (hyper "auto")


def dummy_twin_model(settings: rb.MCMCSettings, lags: int = LAGS) -> rb.BVAR:
    """Mode II × ``Constant("iw")`` with the SOC/SUR rows (N05 §7.7: weight ``1/λ_i``)."""
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="structural", **DUMMIES)),
        vol=rb.Constant("iw"),
        lags=lags,
        deterministic=DET,
        mcmc=settings,
    )


def phase1_dummy_fit(data: rb.VARData, lags: int):
    """The Phase 1 conjugate fit with the same SOC/SUR dummies and the default EB hierarchy."""
    model = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(**DUMMIES)), lags=lags, deterministic=DET)
    assert model.engine == "analytic"
    return model.fit(data)


def check_a1_dummies(fit, data: rb.VARData, lags: int, *, n_iid: int, seed: int) -> None:
    """The MCMC posterior with the rows against the Phase 1 NIW posterior with the same rows, at
    the fit's hyperparameters; the EB plug-in (λ, soc, sur) is the Phase 1 EB mode."""
    assert fit.metadata["prior_mode"] == "II"
    assert fit.hyper["n_dummy"] == data.n_vars + 1  # SOC rows (every level) and the SUR row
    assert fit.converged, fit.warnings
    phase1 = phase1_dummy_fit(data, lags)
    for key in ("lambda", "soc", "sur"):
        np.testing.assert_allclose(fit.hyper[key], phase1.hyper[key], rtol=1e-8)
    np.testing.assert_allclose(fit.hyper["psi"], phase1.hyper["psi"], rtol=1e-12)
    hyper = {key: fit.hyper[key] for key in ("lambda", "alpha", "psi", "soc", "sur")}
    compare_niw(fit, phase1.posterior(hyper), n_iid=n_iid, seed=seed).check()


@pytest.fixture(scope="module")
def a1_dummy_fit(data3):
    """The PR-tier fit: Mode II × ``Constant("iw")`` with SOC/SUR rows, EB, ``n = 3, p = 2, T = 60``."""
    return run_fit(dummy_twin_model(PR_SETTINGS), data3)


def test_a1_dummies_mode2_iw_matches_the_phase1_posterior_with_dummies(data3, a1_dummy_fit):
    """A1 extension (WP25, PR): with SOC/SUR dummy rows weighted ``1/λ_i`` and counted by the
    coupled ``λ_i`` step, MCMC structural Mode II × ``Constant("iw")`` on its ``fixed_factor``
    path matches the Phase 1 NIW posterior with the same rows: quantiles of ``B``, ``Σ``, ``A``,
    ``Λ`` within 4 quantile-MCSE (the A2 comparison), and the EB modes of λ, μ, δ agree."""
    assert a1_dummy_fit.metadata["const_path"] == "fixed_factor"
    check_a1_dummies(a1_dummy_fit, data3, LAGS, n_iid=200_000, seed=6)


def test_a1_dummies_comparison_sees_the_rows(data3, a1_dummy_fit):
    """Negative control (power): the same draws against the Phase 1 posterior *without* the rows
    (same λ, α, ψ) fail the comparison."""
    phase1 = phase1_fit(data3, LAGS)
    post = phase1.posterior(plug_in(a1_dummy_fit))
    with pytest.raises(AssertionError, match="quantiles outside"):
        compare_niw(a1_dummy_fit, post, n_iid=200_000, seed=6).check()


@pytest.mark.slow
@pytest.mark.parametrize("const_path", ["weighted", "cholesky_const"])
def test_a1_dummies_other_constant_weight_paths(data3, const_path):
    """A1 extension (nightly) on the general weighted-Gram path and the Cholesky path."""
    fit = run_fit(dummy_twin_model(PR_SETTINGS), data3, const_path)
    assert fit.metadata["const_path"] == const_path
    check_a1_dummies(fit, data3, LAGS, n_iid=200_000, seed=7)


@pytest.mark.slow
def test_a1_dummies_realistic_size():
    """A1 extension (nightly) at ``n = 20, p = 4, T = 300`` (the A3 size)."""
    lags = 4
    data = var_data(20, 300 + lags, seed=2031)
    fit = run_fit(dummy_twin_model(NIGHTLY_SETTINGS, lags=lags), data)
    assert fit.metadata["const_path"] == "fixed_factor"
    check_a1_dummies(fit, data, lags, n_iid=40_000, seed=8)
