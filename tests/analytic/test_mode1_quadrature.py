"""WP20: A5, the Mode I × Constant analytic twin by per-equation 1-D quadrature (DESIGN §9.8 A5,
§6.1, D-10; N05 §4.1, §4.4, §9, §13.2).

**Model** (DESIGN §6.1). Under ``Linear(Minnesota(form="structural")) × Constant("ig")`` every
structural equation ``i`` (0-based, ``ι = i + 1``) is a regression in δ-coordinates ``θ_i = [δ_i |
a_i]`` (``γ_i = δ_i + M_c a_i``, unit Jacobian; N05 §4.1)::

    y_i = F_i θ_i + e_i,   e_i ~ N(0, λ_i I_T),   F_i = [X | X M_c,<i − Y_<i],
    θ_i ~ N(μ₀, V₀)  (diagonal, λ-free),           λ_i ~ IG(c_i, C_i) = IG(1 + ι/2, ψ_i/2).

The priors are independent across equations and ``|A| = 1``, so the posterior factorises over
``i`` and the oracle is exact at any ``n``. Integrating ``θ_i`` out leaves one dimension::

    p(λ_i | y) ∝ IG(λ_i; c_i, C_i)·N(y_i; F_iμ₀, λ_i I + F_iV₀F_iᵀ),

evaluated in ``q × q`` (determinant lemma ``|λI + FV₀Fᵀ| = λ^T |V₀| |V₀⁻¹ + FᵀF/λ|``; Woodbury
in the residual form ``rᵀ(λI + FV₀Fᵀ)⁻¹r = min_θ ‖y − Fθ‖²/λ + ‖V₀^{−½}(θ − μ₀)‖²``) from one QR
of the stacked rows ``[V₀^{−½} | V₀^{−½}μ₀ ; R_{Fy}/√λ]`` per node, where ``R_{Fy}`` is the
triangular factor of ``[F | y]`` (orthogonal invariance: only ``[F | y]ᵀ[F | y]`` enters). The same
QR gives ``θ | λ, y ~ N(θ̂(λ), (RᵀR)⁻¹)``. Gauss–Legendre in ``u = log λ`` around the mode then
yields ``E[λ_i]``, ``E[θ_i]`` and ``Var(θ_i) = E[Var(θ | λ)] + Var(E[θ | λ])``; doubling the rule
and widening the interval changes nothing above 1e-10 (asserted on every call).

**Generalisation used as the oracle's own check.** With a λ-scaled prior ``V₀(λ) = V₀ ⊙ λ^s``
(``s ∈ {0, 1}`` per coordinate) the integral stays one-dimensional: ``s ≡ 1`` is Mode II, where
the quadrature must reproduce Chan's (2022) closed form (``engines.structural_nig``) to rounding;
``s = 1`` on ``a_i`` only is ``Constant("iw_equivalent")`` (Chan 2022 Prop. 1; DESIGN §6.1), run
nightly as an extension of A5.

**Comparison** (DESIGN §9.8): for ``g ∈ {λ_i, θ_ir, (θ_ir − E*θ_ir)²}`` over γ-coordinates
``[γ_i | a_i]``, ``z = (ḡ_MCMC − E*g)/MCSE(ḡ)`` with the mean MCSE of ``diagnostics.mcse_mean_many``
(Vehtari et al. 2021). Decision ("within 4 MCSE, Holm"): every ``|z| ≤ max(4, z_Bonf)`` with
``z_Bonf`` the two-sided Bonferroni critical value at family level 1e-2 (N06 §3.5: Holm equals
Bonferroni for the global decision), so small families are held to 4 MCSE and large ones (``n =
20``: thousands of functions) keep a family-wise false-alarm rate ≤ 1%; Holm lists the failures.
A negative control (the IG shape off by ½, a ~1.6% shift of ``E[λ_i]``) shows the PR comparison
has power.

The prior is written here from the DESIGN §6.1 formulas (two-implementation rule, DESIGN §9.1) and
the regression design from the Phase 0 NumPy ``lagged_design``; a separate test pins both to the
engine's resolved arrays. Cadence: ``n = 3`` per PR (production ``eigen`` path), ``n = 20`` and the
other constant-weight paths nightly (``slow``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.linalg import solve_triangular
from scipy.special import gammaln, roots_legendre

import rai_bvar as rb
from rai_bvar import diagnostics
from rai_bvar.data.design import lagged_design, n_deterministic, regressor_names
from rai_bvar.engines.structural_nig import structural_posteriors
from rai_bvar.priors.scales import window_scales
from rai_bvar.validation.sequential import holm_reject

pytestmark = [
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
    # structural compositions with many variables say so (R8); expected here
    pytest.mark.filterwarnings("ignore:this composition is not order-invariant:UserWarning"),
]

Scaling = Literal["I", "II", "iw_equivalent"]

LAGS, DET = 2, "c"
BOUND = 4.0  # MCSE units (DESIGN §9.8)
ALPHA = 1e-2  # family level once 4 MCSE stops controlling it (N06 §3.5)
PR_SETTINGS = rb.MCMCSettings(chains=4, draws=10_000, burn=500, store_max=10_000, seed=520)
NIGHTLY_SETTINGS = rb.MCMCSettings(chains=4, draws=2_500, burn=500, store_max=2_500, seed=521)
_RULES = ((128, 16.0), (256, 24.0))  # (Gauss–Legendre nodes, half-width in posterior s.d. of u)
_QUAD_RTOL = 1e-10  # agreement of the two rules


# --------------------------------------------------------------------------- data


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


# --------------------------------------------------------------------------- prior (DESIGN §6.1)


@dataclass(frozen=True)
class EquationPrior:
    """Prior of equation ``i`` in δ-coordinates ``[δ_i | a_i]`` (length ``k + i``): ``θ ~ N(mean,
    var ⊙ λ^scaled)`` and ``λ ~ IG(shape, scale)`` (density ``∝ λ^{−shape−1} e^{−scale/λ}``)."""

    mean: np.ndarray
    var: np.ndarray
    scaled: np.ndarray  # bool: the variance is multiplied by λ_i
    shape: float
    scale: float


def equation_prior(
    i: int, hyper: dict[str, Any], n: int, lags: int, n_det: int, scaling: Scaling
) -> EquationPrior:
    """DESIGN §6.1 with ``κ = lambda``, ``α = alpha``, ``κ_c = cross``, ``κ_d = var_const``,
    ``κ_a = contemp``, ``d = n + 2``, ``Ω^{(i)}_{(l,m)} = κ² c_im (d−n−1)/(l^α ψ_m)``:

    * δ_i: Mode I ``N(m_i, σ̄_i² Ω^{(i)})`` with ``σ̄_i² = ψ_i/ι`` (``"I"``, ``"iw_equivalent"``);
      Mode II ``N(m_i, λ_i Ω^{(i)})`` (``"II"``); deterministic terms ``κ_d σ̄_i²`` / ``λ_i κ_d``;
    * a_ij: ``N(0, κ_a ψ_i/(ι ψ_j))`` (``"I"``) or ``N(0, λ_i κ_a/ψ_j)`` (``"II"``,
      ``"iw_equivalent"``);
    * λ_i ~ IG(1 + ι/2, ψ_i/2).
    """
    psi = np.asarray(hyper["psi"], dtype=np.float64)
    own = np.asarray(hyper["own_means"], dtype=np.float64)
    kappa, alpha, cross = hyper["lambda"], hyper["alpha"], hyper["cross"]
    var_const, contemp = hyper["var_const"], hyper["contemp"]
    iota, d = i + 1.0, n + 2.0
    k = n_det + n * lags
    sigma2 = 1.0 if scaling == "II" else psi[i] / iota
    mean_d, var_d = np.zeros(k), np.empty(k)
    var_d[:n_det] = var_const * sigma2
    for lag in range(1, lags + 1):
        for v in range(n):
            col = n_det + (lag - 1) * n + v
            c_iv = 1.0 if v == i else cross**2
            var_d[col] = kappa**2 * c_iv * (d - n - 1.0) / (lag**alpha * psi[v]) * sigma2
            if v == i and lag == 1:
                mean_d[col] = own[i]
    if scaling == "I":
        var_a = contemp * psi[i] / (iota * psi[:i])
    else:
        var_a = contemp / psi[:i]
    scaled = np.concatenate(
        [np.full(k, scaling == "II"), np.full(i, scaling in ("II", "iw_equivalent"))]
    )
    return EquationPrior(
        mean=np.concatenate([mean_d, np.zeros(i)]),
        var=np.concatenate([var_d, var_a]),
        scaled=scaled,
        shape=1.0 + iota / 2.0,
        scale=psi[i] / 2.0,
    )


def prior_mean_matrix(own: np.ndarray, n: int, lags: int, n_det: int) -> np.ndarray:
    """``M`` (k, n): own first lag ``m_i``, zero elsewhere (DESIGN §6.1)."""
    M = np.zeros((n_det + n * lags, n))
    M[n_det + np.arange(n), np.arange(n)] = own
    return M


# --------------------------------------------------------------------------- quadrature oracle


@dataclass(frozen=True)
class Moments:
    """Exact posterior moments of one equation: ``E[λ]``, ``E[θ]`` (q,), ``Cov(θ)`` (q, q)."""

    lam_mean: float
    mean: np.ndarray
    cov: np.ndarray

    def to_gamma(self, Mc_i: np.ndarray) -> Moments:
        """``[δ | a] → [γ | a]`` with ``γ = δ + M_c,<i a`` (``T`` unit upper triangular)."""
        k, i = Mc_i.shape
        T = np.eye(k + i)
        T[:k, k:] = Mc_i
        return Moments(self.lam_mean, T @ self.mean, T @ self.cov @ T.T)


class EquationPosterior:
    """``p(λ, θ | y)`` of one regression under :class:`EquationPrior` (module docstring)."""

    def __init__(self, y: np.ndarray, F: np.ndarray, prior: EquationPrior):
        self.T, self.q = F.shape
        self.prior = prior
        # [F | y] = Q R_Fy: the data enter the stacked least squares through R_Fy only
        self.R_Fy = np.linalg.qr(np.column_stack([F, y]), mode="r")

    def node(self, lam: float) -> tuple[float, np.ndarray, np.ndarray]:
        """``(log p(y | λ), θ̂(λ), (RᵀR)⁻¹)`` with ``θ | λ, y ~ N(θ̂, (RᵀR)⁻¹)``."""
        q, p = self.q, self.prior
        v = p.var * np.where(p.scaled, lam, 1.0)
        s = 1.0 / np.sqrt(v)
        stacked = np.vstack([np.column_stack([np.diag(s), s * p.mean]), self.R_Fy / np.sqrt(lam)])
        R_aug = np.linalg.qr(stacked, mode="r")
        R_aug *= np.where(np.diag(R_aug) < 0, -1.0, 1.0)[:, None]
        R, r, rho = R_aug[:q, :q], R_aug[:q, q], R_aug[q, q]
        log_lik = (
            -0.5 * self.T * np.log(2.0 * np.pi * lam)
            - 0.5 * np.log(v).sum()
            - np.log(np.diag(R)).sum()
            - 0.5 * rho**2
        )
        R_inv = solve_triangular(R, np.eye(q), lower=False)
        return float(log_lik), R_inv @ r, R_inv @ R_inv.T

    def log_prior_u(self, u: np.ndarray | float) -> np.ndarray | float:
        """``log IG(e^u; c, C) + u``: the λ prior on ``u = log λ`` with its Jacobian ``e^u``."""
        c, C = self.prior.shape, self.prior.scale
        return c * np.log(C) - gammaln(c) - c * u - C * np.exp(-u)

    def log_density_u(self, u: float) -> float:
        """``log p(u | y)`` up to a constant."""
        return float(self.log_prior_u(u)) + self.node(float(np.exp(u)))[0]

    def mode_u(self) -> tuple[float, float]:
        """Mode of ``u`` (grid, then bounded Brent) and the s.d. implied by the curvature there."""
        centre = np.log(self.prior.scale / self.prior.shape)
        grid = centre + np.arange(-24.0, 24.5, 0.5)
        values = np.array([self.log_density_u(u) for u in grid])
        j = int(np.argmax(values))
        assert 0 < j < grid.size - 1, "posterior mode of log λ outside the search grid"
        res = optimize.minimize_scalar(
            lambda u: -self.log_density_u(u),
            bounds=(grid[j - 1], grid[j + 1]),
            method="bounded",
            options={"xatol": 1e-10},
        )
        u0, h = float(res.x), 1e-3
        curv = (
            -(
                self.log_density_u(u0 + h)
                - 2.0 * self.log_density_u(u0)
                + self.log_density_u(u0 - h)
            )
            / h**2
        )
        assert curv > 0.0
        return u0, 1.0 / np.sqrt(curv)

    def moments(self, nodes: int, width: float, mode: tuple[float, float]) -> Moments:
        u0, sd = mode
        x, w = roots_legendre(nodes)
        u = u0 + width * sd * x
        out = [self.node(float(np.exp(v))) for v in u]
        log_w = np.log(w) + self.log_prior_u(u) + np.array([o[0] for o in out])
        wt = np.exp(log_w - log_w.max())
        wt /= wt.sum()
        theta = np.array([o[1] for o in out])
        mean = wt @ theta
        dev = theta - mean
        cov = np.einsum("n,nij->ij", wt, np.array([o[2] for o in out]))
        cov += np.einsum("n,ni,nj->ij", wt, dev, dev)
        return Moments(float(wt @ np.exp(u)), mean, cov)

    def exact(self) -> Moments:
        """Moments by the finer rule, after checking that both rules agree to ``_QUAD_RTOL``."""
        mode = self.mode_u()
        coarse, fine = (self.moments(nodes, width, mode) for nodes, width in _RULES)
        sd = np.sqrt(np.diag(fine.cov))
        assert abs(coarse.lam_mean - fine.lam_mean) <= _QUAD_RTOL * fine.lam_mean
        assert np.all(np.abs(coarse.mean - fine.mean) <= _QUAD_RTOL * (sd + np.abs(fine.mean)))
        assert np.all(np.abs(np.diag(coarse.cov) - sd**2) <= _QUAD_RTOL * sd**2)
        return fine


def exact_moments(
    data: rb.VARData,
    hyper: dict[str, Any],
    lags: int,
    det: str,
    scaling: Scaling,
    alter: Callable[[EquationPrior], EquationPrior] | None = None,
) -> list[Moments]:
    """Exact posterior moments of every equation in γ-coordinates ``[γ_i | a_i]`` at ``hyper``
    (``alter`` modifies each equation's prior; negative controls only)."""
    n, n_det = data.n_vars, n_deterministic(det)
    Y, X = lagged_design(data.values, lags, det)
    M = prior_mean_matrix(np.asarray(hyper["own_means"]), n, lags, n_det)
    out = []
    for i in range(n):
        prior = equation_prior(i, hyper, n, lags, n_det, scaling)
        prior = prior if alter is None else alter(prior)
        F = np.hstack([X, X @ M[:, :i] - Y[:, :i]])
        out.append(EquationPosterior(Y[:, i], F, prior).exact().to_gamma(M[:, :i]))
    return out


# --------------------------------------------------------------------------- comparison


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
            f"{bad.sum()} of {G} functions outside {crit:.2f} MCSE:\n" + "\n".join(rows[:40])
        )


def mcmc_z(
    fit, moments: list[Moments], variables: tuple[str, ...], lags: int, det: str
) -> tuple[np.ndarray, list[str]]:
    """z-scores of ``E[λ_i]``, ``E[θ_ir]`` and ``Var(θ_ir)`` (γ-coordinates) for every equation."""
    p = fit.param_draws
    gamma = np.asarray(p.mean.gamma)  # (chain, draw, n, k)
    a = np.asarray(p.struct.a)  # (chain, draw, n, n−1)
    lam = np.exp(np.asarray(p.vol.core.log_lam))  # (chain, draw, n)
    regs = regressor_names(variables, lags, det)
    zs, names = [], []
    for i, mom in enumerate(moments):
        v = variables[i]
        labels = [f"gamma[{v}, {r}]" for r in regs] + [f"a[{v}~{w}]" for w in variables[:i]]
        theta = np.concatenate([gamma[..., i, :], a[..., i, :i]], axis=-1)
        sq = (theta - mom.mean) ** 2
        zs += [
            [(lam[..., i].mean() - mom.lam_mean) / diagnostics.mcse_mean(lam[..., i])],
            (theta.mean(axis=(0, 1)) - mom.mean) / diagnostics.mcse_mean_many(theta),
            (sq.mean(axis=(0, 1)) - np.diag(mom.cov)) / diagnostics.mcse_mean_many(sq),
        ]
        names += [f"E[lambda[{v}]]"] + [f"E[{s}]" for s in labels] + [f"Var[{s}]" for s in labels]
    return np.concatenate([np.ravel(z) for z in zs]), names


def structural_model(vol: rb.Constant, settings: rb.MCMCSettings, lags: int = LAGS) -> rb.BVAR:
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="structural")),
        vol=vol,
        lags=lags,
        deterministic=DET,
        mcmc=settings,
    )


def run_fit(model: rb.BVAR, data: rb.VARData, const_path: str = "auto"):
    """``model.fit(data)`` on the resolved constant-weight path, or on an explicit path:
    ``"cholesky_const"`` through ``MCMCSettings(const_path="cholesky")``; ``"weighted"`` (the
    general weighted-Gram path, the reference oracle of the fast paths, DESIGN §3.7.2) through the
    engine-level ``fit_mcmc`` on the resolved inputs."""
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


def check_a5(
    data: rb.VARData,
    vol: rb.Constant,
    scaling: Scaling,
    settings: rb.MCMCSettings,
    lags: int,
    const_path: str = "auto",
    expected_path: str | None = None,
    fit=None,
) -> None:
    """Fit (unless ``fit`` is given) and compare with the quadrature posterior at ``fit.hyper``."""
    if fit is None:
        fit = run_fit(structural_model(vol, settings, lags), data, const_path)
    assert fit.metadata["prior_mode"] == scaling
    assert fit.metadata["const_path"] == (expected_path or const_path)
    assert fit.converged, fit.warnings
    moments = exact_moments(data, fit.hyper, lags, DET, scaling)
    z, names = mcmc_z(fit, moments, tuple(data.variables), lags, DET)
    assert_within_mcse(z, names)


# --------------------------------------------------------------------------- the oracle itself


def _fixed_hyper(data: rb.VARData, lags: int, **prior: Any) -> dict[str, Any]:
    p = rb.Minnesota(form="structural", hierarchy="fixed", **prior)
    scales = window_scales(data, p, lags, DET)
    return {
        "lambda": p.lambda_,
        "alpha": p.alpha,
        "cross": p.cross,
        "contemp": p.contemp,
        "var_const": p.var_const,
        "psi": scales.psi,
        "own_means": scales.own_means,
        "prior": p,
    }


@pytest.mark.parametrize("cross", [1.0, 0.5])
def test_quadrature_reproduces_the_structural_nig_closed_form(data3, cross):
    """With every coordinate λ-scaled (Mode II) the 1-D quadrature must equal Chan's (2022)
    closed form (WP16): ``E[λ] = Ŝ/(ν̂ − 1)``, ``E[θ] = θ̂``, ``Cov(θ) = E[λ](RᵀR)⁻¹``."""
    hyper = _fixed_hyper(data3, LAGS, cross=cross, lambda_=0.3)
    moments = exact_moments(data3, hyper, LAGS, DET, "II")
    posts = structural_posteriors(
        data3, hyper["prior"], {"lambda": 0.3, "psi": hyper["psi"]}, LAGS, DET
    )
    for mom, post in zip(moments, posts, strict=True):
        R_inv = solve_triangular(post.R, np.eye(post.R.shape[0]), lower=False)
        cov = post.lam_mean * (R_inv @ R_inv.T)
        sd = np.sqrt(np.diag(cov))
        assert mom.lam_mean == pytest.approx(post.lam_mean, rel=1e-11)
        np.testing.assert_allclose(mom.mean, post.theta_hat, rtol=0, atol=1e-10 * sd.min())
        np.testing.assert_allclose(mom.cov, cov, rtol=1e-9, atol=1e-11 * np.outer(sd, sd).max())


@pytest.mark.parametrize("scaling", ["I", "iw_equivalent"])
def test_quadrature_node_matches_the_dense_gaussian(data3, scaling):
    """``log p(y | λ)``, ``E[θ | λ, y]`` and ``Var(θ | λ, y)`` of one node against the textbook
    ``T × T`` density and the dense normal equations (a proper, well-conditioned prior)."""
    n, n_det = data3.n_vars, n_deterministic(DET)
    hyper = _fixed_hyper(data3, LAGS, var_const=10.0)
    Y, X = lagged_design(data3.values, LAGS, DET)
    M = prior_mean_matrix(hyper["own_means"], n, LAGS, n_det)
    i = n - 1
    prior = equation_prior(i, hyper, n, LAGS, n_det, scaling)
    F, y = np.hstack([X, X @ M[:, :i] - Y[:, :i]]), Y[:, i]
    post = EquationPosterior(y, F, prior)
    for lam in (0.3 * hyper["psi"][i], hyper["psi"][i], 4.0 * hyper["psi"][i]):
        v = prior.var * np.where(prior.scaled, lam, 1.0)
        dense = stats.multivariate_normal(F @ prior.mean, lam * np.eye(len(y)) + (F * v) @ F.T)
        P = np.diag(1.0 / v) + F.T @ F / lam
        cov = np.linalg.inv(P)
        log_lik, mean, cond = post.node(lam)
        assert log_lik == pytest.approx(dense.logpdf(y), rel=1e-10)
        np.testing.assert_allclose(mean, cov @ (prior.mean / v + F.T @ y / lam), rtol=1e-8)
        np.testing.assert_allclose(cond, cov, rtol=1e-8, atol=1e-12 * np.abs(cov).max())


@pytest.mark.parametrize(
    ("vol", "scaling"),
    [
        (rb.Constant("ig"), "I"),
        (rb.Constant("iw"), "II"),
        (rb.Constant("iw_equivalent"), "iw_equivalent"),
    ],
    ids=["mode-I", "mode-II", "iw-equivalent"],
)
def test_oracle_prior_is_the_engine_prior(data3, vol, scaling):
    """The DESIGN §6.1 formulas written above equal the arrays ``resolve`` hands the engine
    (λ-free δ and ``a`` variances, means, ``M_c``, the IG ``λ`` prior), so A5 compares the sampler
    with the posterior of the prior it was given."""
    from rai_bvar.engines.mcmc import resolve

    n, n_det = data3.n_vars, n_deterministic(DET)
    r = resolve.resolve(structural_model(vol, PR_SETTINGS), data3, rb.Runtime())
    k = n_det + n * LAGS
    eq = r.prior.eq
    M = prior_mean_matrix(np.asarray(r.hyper["own_means"]), n, LAGS, n_det)
    np.testing.assert_array_equal(np.asarray(r.prior.shared.Mc), M[:, : n - 1])
    for i in range(n):
        prior = equation_prior(i, r.hyper, n, LAGS, n_det, scaling)
        np.testing.assert_allclose(1.0 / np.asarray(eq.coef.prec0)[i], prior.var[:k], rtol=1e-13)
        np.testing.assert_array_equal(np.asarray(eq.coef.mean0)[i], prior.mean[:k])
        contemp_var = 1.0 / np.asarray(eq.contemp.prec0)[i, :i]
        np.testing.assert_allclose(contemp_var, prior.var[k:], rtol=1e-13)
        np.testing.assert_array_equal(np.asarray(eq.contemp.mean0)[i, :i], 0.0)
        assert float(np.asarray(eq.vol_core.shape)[i]) == pytest.approx(prior.shape, rel=1e-15)
        assert float(np.asarray(eq.vol_core.scale)[i]) == pytest.approx(prior.scale, rel=1e-15)


# --------------------------------------------------------------------------- A5


@pytest.fixture(scope="module")
def a5_fit(data3):
    """The PR-tier A5 fit: Mode I × ``Constant("ig")``, EB plug-in, ``n = 3, p = 2, T = 60``."""
    return run_fit(structural_model(rb.Constant("ig"), PR_SETTINGS), data3)


def test_a5_mode1_constant_matches_quadrature(data3, a5_fit):
    """A5 (PR): ``Linear(structural) × Constant("ig")`` on its production ``eigen`` fast path
    (DESIGN §3.7.2, D-26), EB plug-in λ, ``n = 3, p = 2, T = 60``: ``E[λ_i]``, ``E[θ_i]`` and
    ``Var(θ_i)`` within 4 MCSE of the quadrature posterior."""
    check_a5(data3, rb.Constant("ig"), "I", PR_SETTINGS, LAGS, expected_path="eigen", fit=a5_fit)


def test_a5_rejects_a_slightly_wrong_posterior(data3, a5_fit):
    """Negative control (power): the same draws against the posterior whose IG shape is off by
    ½ (a ~1.6% shift of ``E[λ_i]``; mutant M13 shifts it by ``(T_pad − T_eff)/2 = 2``) fail."""
    moments = exact_moments(
        data3,
        a5_fit.hyper,
        LAGS,
        DET,
        "I",
        alter=lambda p: dataclasses.replace(p, shape=p.shape + 0.5),
    )
    z, names = mcmc_z(a5_fit, moments, tuple(data3.variables), LAGS, DET)
    with pytest.raises(AssertionError, match=r"E\[lambda"):
        assert_within_mcse(z, names)


@pytest.mark.slow
@pytest.mark.parametrize("const_path", ["weighted", "cholesky_const"])
def test_a5_mode1_constant_other_paths(data3, const_path):
    """A5 (nightly) on the general weighted-Gram path and the constant-weight Cholesky path."""
    check_a5(data3, rb.Constant("ig"), "I", PR_SETTINGS, LAGS, const_path=const_path)


@pytest.mark.slow
@pytest.mark.parametrize("const_path", ["auto", "weighted"])
def test_a5_mode1_constant_n20(const_path):
    """A5 (nightly): ``n = 20, p = 4, T = 300`` (the equations stay independent, so the oracle is
    exact at any ``n``); production ``eigen`` path and the weighted-Gram reference path."""
    data = var_data(20, 304, seed=2020)
    expected = "eigen" if const_path == "auto" else None
    check_a5(data, rb.Constant("ig"), "I", NIGHTLY_SETTINGS, 4, const_path, expected_path=expected)


@pytest.mark.slow
@pytest.mark.parametrize(("n", "lags", "T"), [(3, 2, 60), (20, 4, 300)], ids=["n3", "n20"])
def test_a5_extension_iw_equivalent(data3, n, lags, T):
    """A5 extension (nightly): structural × ``Constant("iw_equivalent")`` (Mode I δ, λ-scaled
    ``a_i``; Chan 2022 Prop. 1) is still one-dimensional per equation, so the same quadrature is
    exact; production ``cholesky_const`` path."""
    data = data3 if n == 3 else var_data(n, T + lags, seed=2021)
    settings = PR_SETTINGS if n == 3 else NIGHTLY_SETTINGS
    check_a5(
        data,
        rb.Constant("iw_equivalent"),
        "iw_equivalent",
        settings,
        lags,
        expected_path="cholesky_const",
    )


def test_assert_within_mcse_rule():
    """The decision rule: 4 MCSE for small families, the Bonferroni critical value at 1e-2 for
    large ones, NaN fails."""
    assert_within_mcse(np.array([3.9, -3.99]), ["a", "b"])
    with pytest.raises(AssertionError, match=r"b: z = -4\.10"):
        assert_within_mcse(np.array([0.0, -4.1]), ["a", "b"])
    big = np.zeros(10_000)
    big[0] = 4.5  # below the Bonferroni critical value 4.89 at G = 10⁴
    assert_within_mcse(big, [str(g) for g in range(big.size)])
    with pytest.raises(AssertionError, match="nan"):
        assert_within_mcse(np.array([np.nan]), ["nan"])
