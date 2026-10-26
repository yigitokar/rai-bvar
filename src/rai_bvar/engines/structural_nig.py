"""Chan (2022) structural normal–inverse-gamma closed form, equation by equation (DESIGN §6.1,
§6.5, §9.8 A1; N05 §4.2, §4.4, §5.2, §7.6, §7.7, §9). NumPy only.

**Model** (Mode II, DESIGN §6.1; N05 §4.4). Equation ``i`` (0-based, ``ι = i + 1``) of
``A y_t = Γ x_t + Λ^{½} ε_t`` in δ-coordinates ``δ_i = γ_i − M_c a_i`` (unit Jacobian, N05 §4.1):

    y_i = F_i θ_i + e_i,   e_i ~ N(0, λ_i I),   θ_i = [δ_i | a_i]   (q_i = k + i coordinates),
    F_i = [X | X M_c − Y_{:, :i}],   M_c = M[:, :n−1]                    (DESIGN §2.3),
    θ_i | λ_i ~ N(μ₀, λ_i V₀),   λ_i ~ IG(ν_i, S_i),   ν_i = 1 + ι/2,   S_i = ψ_i/2,

with diagonal ``V₀ = [Ω^{(i)} | κ_a/ψ_j (j < i)]``, ``Ω^{(i)}_{(l,m)} = κ² c_im (d−n−1)/(l^α ψ_m)``
(``c_ii = 1``, ``c_im = κ_c²``), ``Ω^{(i)}_det = κ_d``, ``μ₀ = [m_i | 0]``: the WP6 Mode II arrays
(``priors.structural.structural_arrays(mode="II")``, ``contemp_arrays(scaling="lambda")``,
``priors.sv_arrays.constant_prior``). Those builders return JAX pytree classes, so the arrays are
formed here from WP6's NumPy helpers (``priors.scales.conjugate_omega``, ``prior_mean_matrix``)
and no call loads JAX; a test pins them to the builders. Equations are independent a priori
and ``|A| = 1``, so the posterior and the marginal likelihood factorise over ``i`` (N05 §0). At
``κ_c = κ_a = 1`` the prior is the Phase 1 NIW prior (N05 §4.2); otherwise it is Chan's (2022)
asymmetric conjugate prior with D3 centring.

**Posterior** (Chan 2022 §2.3, eq. 10), from one QR of the stacked rows, no inverse (N05 §9):

    [V₀^{−½}  V₀^{−½}μ₀ ; F_i  y_i] = Q [R  r ; 0  ρ],   K = V₀⁻¹ + F_i'F_i = R'R,   θ̂ = R⁻¹r,
    Ŝ = S_i + ½ρ²,   ρ² = ‖y_i − F_iθ̂‖² + ‖V₀^{−½}(θ̂ − μ₀)‖²,   ν̂ = ν_i + T/2,
    λ_i | y ~ IG(ν̂, Ŝ),   θ_i | λ_i, y ~ N(θ̂, λ_i K⁻¹).

``Ŝ`` is the residual form read off the factor: Chan's ``S + ½(y'y + μ₀'V₀⁻¹μ₀ − θ̂'Kθ̂)`` is
equal in exact arithmetic (``Kθ̂ = V₀⁻¹μ₀ + F'y``) but cancels catastrophically on well-fitted
data (N05 §7.6), and an explicit residual ``y_i − F_iθ̂`` still cancels when the level dwarfs
the noise. The prior rows go first, as in Phase 1's ``ConjugateProblem._core``.

**Log marginal likelihood** (Chan 2022 §2.4, eq. 11):

    log p(y_i) = −(T/2) log 2π − ½ log|V₀| − log|R| + lnΓ(ν̂) − lnΓ(ν_i) + ν_i ln S_i − ν̂ ln Ŝ.

SOC/SUR dummy rows (Phase 1 convention, built from ``ȳ₀``, the mean of the first ``p``
observations) enter each equation as extra rows of ``(y_i, F_i)``, transformed like the data
(N05 §7.7); the data marginal likelihood is ``Σ_i [log p_i(dummies + data) − log p_i(dummies)]``.
At ``κ_c = κ_a = 1`` it equals the Phase 1 ``AnalyticFit.log_ml()`` (DESIGN §9.8 A1; N05 §13.2).

**Coordinates.** :func:`nig_posterior` keeps the coordinates it is given. The posteriors of
:func:`structural_posteriors` are returned in γ-coordinates ``[γ_i | a_i]`` through the unit upper
triangular map ``γ_i = δ_i + M_c a_i``: ``θ̂^γ = Tθ̂`` and ``R^γ = RT⁻¹ = [[R₁₁, R₁₂ − R₁₁M_c],
[0, R₂₂]]``, still upper triangular with the same log ML; :func:`draw_structural_nig` consumes them.

**Empirical Bayes** (DESIGN §6.5 step 2; D-12): :func:`eb_structural` maximises
``Σ_i log p(y_i | h) + log p(h)`` (GLP hyperpriors, natural-scale densities as in Phase 1) over the
free set by L-BFGS-B on log values with Phase 1's stationarity test. The Phase 2 plug-in frees
``lambda`` (optionally ``alpha``, ``soc``, ``sur``) with ``cross`` and ``contemp`` fixed by the
prior; P3 §4.3 adds ``cross`` and ``contemp`` to the free names (flat prior on their log over
``[1e-3, 10]``, so EB over them maximises the marginal likelihood, Chan 2022 §4). ``_Window``
takes ``cross`` and ``contemp`` per evaluation, with the arithmetic of the fixed-``cross`` code,
so at a fixed ``cross`` every evaluation is bit for bit the Phase 2 one (E118). ``psi`` is never
free here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import gammaln

from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import lagged_design, n_deterministic
from rai_bvar.engines.analytic import _STATIONARITY_TOL, _LinearDraws, _stationarity
from rai_bvar.linalg import positive_diagonal
from rai_bvar.priors.asymmetric import LOG_FLAT_BOX
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.scales import conjugate_omega, prior_mean_matrix, window_scales

_LOG_2PI = float(np.log(2.0 * np.pi))
_RANK_TOL = 1e-10  # |R_jj| relative to the stacked column norm (as ConjugateProblem._core)
_EB_NAMES = ("lambda", "alpha", "soc", "sur")  # the GLP hyperparameters (Phase 1, P2 §6.5)
_SCALE_NAMES = ("cross", "contemp")  # κ_c, κ_a: flat on their log over LOG_FLAT_BOX (P3 §4.3)
_HYPER_NAMES = (*_EB_NAMES, *_SCALE_NAMES)
# L-BFGS-B tolerances when cross or contemp is free (P3 Q38, owner 2026-10-25: tighten the free-scale
# path so A9's projected gradient at the mode is below 1e-6). The Phase 2 plug-in (cross and contemp
# fixed) keeps ftol = 1e-12 and SciPy's default gtol, so its results stay bitwise (E118).
_FREE_SCALE_OPTIONS = {"ftol": 1e-15, "gtol": 1e-9}


@dataclass(frozen=True)
class NIGPosterior:
    """``λ | y ~ IG(ν̂, Ŝ)`` and ``θ | λ, y ~ N(θ̂, λ(RᵀR)⁻¹)`` (Chan 2022 eq. 10), with ``log p(y)``.

    ``theta_hat`` (q,), ``R`` (q, q) upper triangular with positive diagonal (``K = RᵀR``),
    ``nu_hat``/``S_hat`` the IG shape and scale, ``log_ml`` the log marginal likelihood (eq. 11).
    """

    theta_hat: np.ndarray
    R: np.ndarray
    nu_hat: float
    S_hat: float
    log_ml: float

    @property
    def lam_mean(self) -> float:
        """``E[λ | y] = Ŝ/(ν̂ − 1)`` (``inf`` when ``ν̂ ≤ 1``)."""
        return self.S_hat / (self.nu_hat - 1.0) if self.nu_hat > 1.0 else np.inf

    def cov(self) -> np.ndarray:
        """Marginal posterior covariance ``E[λ]·K⁻¹`` of θ (dense; inspection and tests only)."""
        q = self.R.shape[0]
        return self.lam_mean * cho_solve((self.R, False), np.eye(q))

    def draw(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``size`` iid draws ``(θ (S, q), λ (S,))``: ``λ = Ŝ/G`` with ``G ~ Gamma(ν̂, 1)``, then
        ``θ = θ̂ + √λ·R⁻¹z`` with ``z ~ N(0, I)`` (one triangular solve; N05 §9)."""
        size = _positive_int(size, "size")
        lam = self.S_hat / rng.standard_gamma(self.nu_hat, size)
        z = rng.standard_normal((self.R.shape[0], size))
        dev = solve_triangular(self.R, z, lower=False).T
        return self.theta_hat + np.sqrt(lam)[:, None] * dev, lam


@dataclass(frozen=True)
class EBResult:
    """Structural (Chan 2022) EB mode of the free Minnesota hyperparameters (DESIGN §6.5 step 2).

    ``hyper`` holds complete values (``lambda``, ``alpha``, ``psi``, ``soc``, ``sur``) for the WP6
    builders; ``free`` the optimised names. ``converged``: the best L-BFGS-B end point passes Phase
    1's stationarity test (predicted Newton gain ``½Σg²/H < 1e-6`` over coordinates not pinned at
    a bound). ``log_ml`` is the data log ML at the mode and ``log_posterior`` adds the GLP log
    hyperprior of the free names. ``warnings`` carries the non-convergence message for
    ``fit.warnings``; nothing is emitted through ``warnings.warn`` here.
    """

    hyper: dict[str, Any]
    free: tuple[str, ...]
    converged: bool
    log_ml: float
    log_posterior: float
    message: str
    evaluations: int
    predicted_gain: float
    projected_gradient: float
    warnings: tuple[str, ...] = ()


class _LogFlat:
    """The flat prior on ``log κ`` over ``[log 1e-3, log 10]`` of ``cross`` and ``contemp`` (P3
    DESIGN §4.3), as the EB objective sees it: a constant (the density of ``log κ``), so EB over
    ``κ`` maximises the marginal likelihood (Chan 2022 §4). The natural-scale density
    ``1/(κ·log(10/1e-3))``, which the Metropolis–Hastings target needs, is :meth:`logpdf_natural`."""

    lower, upper = LOG_FLAT_BOX

    @classmethod
    def logpdf(cls, value: float) -> float:
        return -float(np.log(np.log(cls.upper) - np.log(cls.lower)))

    @classmethod
    def logpdf_natural(cls, value: float) -> float:
        return cls.logpdf(value) - float(np.log(value))


_LOG_FLAT = _LogFlat()


# --------------------------------------------------------------------------- one regression


def nig_posterior(
    Z: np.ndarray, y: np.ndarray, mu0: np.ndarray, var0: np.ndarray, nu0: float, S0: float
) -> NIGPosterior:
    """Conjugate NIG regression ``y = Zθ + e``, ``e ~ N(0, λI)``, ``θ | λ ~ N(μ₀, λ diag(var0))``,
    ``λ ~ IG(ν₀, S₀)`` (shape, scale): posterior and log marginal likelihood (module docstring;
    Chan 2022 eqs. 10–11). ``Z`` is (T, q) with ``T ≥ 0`` rows (``T = 0`` returns the prior with
    ``log_ml = 0``). Raises ``np.linalg.LinAlgError`` when the stacked regressors are numerically
    rank deficient (``|R_jj| ≤ 1e-10·‖column j‖``)."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    var0 = np.asarray(var0, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be a (T, q) matrix, got shape {Z.shape}")
    T, q = Z.shape
    if y.shape != (T,) or mu0.shape != (q,) or var0.shape != (q,):
        raise ValueError(
            f"shapes do not match Z {Z.shape}: y {y.shape} (needs ({T},)), mu0 {mu0.shape} and "
            f"var0 {var0.shape} (need ({q},))"
        )
    if not (np.all(np.isfinite(Z)) and np.all(np.isfinite(y)) and np.all(np.isfinite(mu0))):
        raise ValueError("Z, y and mu0 must be finite")
    if not np.all(np.isfinite(var0) & (var0 > 0)):
        raise ValueError("var0 must hold positive finite prior variances")
    nu0, S0 = float(nu0), float(S0)
    if not (np.isfinite(nu0) and nu0 > 0 and np.isfinite(S0) and S0 > 0):
        raise ValueError(f"nu0 and S0 must be positive and finite, got {nu0!r}, {S0!r}")
    s = 1.0 / np.sqrt(var0)  # V₀^{−½}
    stacked = np.zeros((q + T, q + 1))  # prior rows first, as ConjugateProblem._core
    stacked[np.arange(q), np.arange(q)] = s
    stacked[:q, q] = s * mu0
    stacked[q:, :q], stacked[q:, q] = Z, y
    R_aug = positive_diagonal(np.triu(np.linalg.qr(stacked, mode="r")), upper=True)
    R, r = R_aug[:q, :q], R_aug[:q, q]
    diag = np.diag(R)
    if not np.all(np.isfinite(R_aug)) or np.any(
        diag <= _RANK_TOL * np.linalg.norm(stacked[:, :q], axis=0)
    ):
        raise np.linalg.LinAlgError("stacked regressors are (numerically) rank deficient")
    theta = solve_triangular(R, r, lower=False)
    # ρ² = ‖ỹ − Z̃θ̂‖² read off the QR (N05 §9); with T = 0 the factor has q rows and ρ = 0
    S_hat = S0 + 0.5 * R_aug[q, q] ** 2 if T else S0
    nu_hat = nu0 + 0.5 * T
    log_ml = (
        -0.5 * T * _LOG_2PI
        - 0.5 * np.log(var0).sum()
        - np.log(diag).sum()
        + gammaln(nu_hat)
        - gammaln(nu0)
        + nu0 * np.log(S0)
        - nu_hat * np.log(S_hat)
    )
    return NIGPosterior(theta, np.ascontiguousarray(R), nu_hat, float(S_hat), float(log_ml))


# --------------------------------------------------------------------------- one window


class _Window:
    """Hyperparameter-free pieces of the per-equation structural problem of one data window.

    ``Y``, ``X`` from ``lagged_design``; ``W = X M_c − Y_{:, :n−1}`` the shared quasi-differenced
    contemporaneous columns (N05 §4.1); ψ and the own-lag means from the WP6 ``window_scales``
    (``psi`` overrides ψ); ``ȳ₀`` and the SOC rows as in Phase 1 (``soc_scope``); the
    hyperparameter-free Mode II prior piece (DESIGN §6.1) ``λ_i ~ IG(1 + ι/2, ψ_i/2)``.
    ``cross`` (``κ_c``) and ``contemp`` (``κ_a``, in ``a_ij | λ_i ~ N(0, λ_i κ_a/ψ_j)``) are
    hyperparameters like ``lambda``: :meth:`complete` takes them from ``hyper`` (else the prior)
    and :meth:`equations` reads them per evaluation (P3 DESIGN §4.3, E118), with the arithmetic of
    the fixed-``cross`` code, so at the prior's values every array is bit for bit the same.
    """

    def __init__(
        self,
        data: VARData,
        prior: Minnesota,
        lags: int,
        deterministic: str,
        psi: np.ndarray | None = None,
    ):
        if not isinstance(data, VARData):
            raise TypeError(f"data must be a VARData, got {type(data).__name__}")
        if not isinstance(prior, Minnesota):
            raise TypeError(f"prior must be a Minnesota prior, got {type(prior).__name__}")
        self.scales = window_scales(data, prior, lags, deterministic, psi=psi)
        self.prior, self.lags = prior, int(lags)
        self.n_det = n_deterministic(deterministic)
        values = data.values
        self.Y, self.X = lagged_design(values, self.lags, deterministic)
        self.T, self.n = self.Y.shape
        self.k = self.X.shape[1]
        self.ybar0 = values[: self.lags].mean(axis=0)
        means = self.scales.own_means
        self.soc_rows = np.arange(self.n) if prior.soc_scope == "all" else np.flatnonzero(means)
        self.M = prior_mean_matrix(self.scales, self.n, self.lags, self.n_det)  # (k, n)
        self.Mc = self.M[:, : self.n - 1]
        self.W = self.X @ self.Mc - self.Y[:, : self.n - 1]
        psi = self.scales.psi
        self.nu0 = 1.0 + self.scales.iota / 2.0  # constant_prior shape
        self.S0 = psi / 2.0  # constant_prior scale

    def complete(self, hyper: Mapping[str, Any]) -> dict[str, Any]:
        """``lambda``, ``alpha``, ``soc``, ``sur``, ``cross``, ``contemp`` from ``hyper`` (else the
        prior), validated, and the window's ``psi``. A ``covid`` entry must be ``None`` (no
        Lenza–Primiceri, D-21). ``cross`` and ``contemp`` given in ``hyper`` must be positive and
        finite; the prior's values are taken as they are (Phase 2)."""
        unknown = sorted(set(hyper) - {*_HYPER_NAMES, "psi", "covid"})
        if unknown:
            raise KeyError(
                f"unknown hyperparameters {unknown}; expected some of "
                f"{sorted({*_HYPER_NAMES, 'psi', 'covid'})}"
            )
        if hyper.get("covid") is not None:
            raise ValueError(
                "the structural NIG closed form has no Lenza-Primiceri scaling: covid must be None"
            )
        if hyper.get("psi") is not None and not np.array_equal(
            np.asarray(hyper["psi"], dtype=np.float64), self.scales.psi
        ):
            raise ValueError("hyper['psi'] differs from the psi this window was built with")
        p = self.prior
        out: dict[str, Any] = {"lambda": p.lambda_, "alpha": p.alpha, "soc": p.soc, "sur": p.sur}
        out.update({key: hyper[key] for key in _EB_NAMES if key in hyper})
        for key in _EB_NAMES:
            value = out[key]
            if value is None and key in ("soc", "sur"):
                continue
            _check_positive(key, value)
            out[key] = float(value)
        if out["soc"] is not None and self.soc_rows.size == 0:
            raise ValueError(
                "the SOC dummy needs at least one variable with a non-zero own-lag prior mean"
            )
        for key in _SCALE_NAMES:  # κ_c, κ_a: per evaluation (P3 §4.3)
            if key in hyper:
                _check_positive(key, hyper[key])
                out[key] = float(hyper[key])
            else:
                out[key] = getattr(p, key)
        out["psi"] = self.scales.psi.copy()
        return out

    def dummies(self, h: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """SOC rows ``diag(ȳ₀)/μ`` (one per SOC variable, zero on deterministic terms) and the SUR
        row ``ȳ₀'/δ`` (``1/δ`` on the constant), repeated over the lags (Phase 1 convention)."""
        n, n_det, lags = self.n, self.n_det, self.lags
        rows_y, rows_x = [np.zeros((0, n))], [np.zeros((0, self.k))]
        if h["soc"] is not None:
            yd = (np.diag(self.ybar0) / h["soc"])[self.soc_rows]
            rows_y.append(yd)
            rows_x.append(np.hstack([np.zeros((len(yd), n_det)), np.tile(yd, lags)]))
        if h["sur"] is not None:
            yd = self.ybar0[None, :] / h["sur"]
            det = np.zeros((1, n_det))
            if n_det:
                det[0, 0] = 1.0 / h["sur"]  # constant only; a trend gets no SUR weight
            rows_y.append(yd)
            rows_x.append(np.hstack([det, np.tile(yd, lags)]))
        return np.vstack(rows_y), np.vstack(rows_x)

    def equations(self, h: Mapping[str, Any]):
        """Per equation: ``(Z, y, Zd, yd, μ₀, var0, ν_i, S_i)`` in δ-coordinates ``[δ_i | a_i]``
        (data rows, dummy rows, Mode II prior ``μ₀ = [m_i | 0]``, ``var0 = [Ω^{(i)} | κ_a/ψ_j]``;
        ``Ω`` from WP6 ``conjugate_omega``, bit for bit Phase 1's ``Ω`` at ``cross = 1``)."""
        omega = conjugate_omega(
            self.scales,
            self.prior,
            h["lambda"],
            h["alpha"],
            self.n,
            self.lags,
            self.n_det,
            cross=h["cross"],
        )  # (n, k)
        var_a = h["contemp"] / self.scales.psi[:-1]  # κ_a/ψ_j (contemp_arrays, scaling="lambda")
        Yd, Xd = self.dummies(h)
        Wd = Xd @ self.Mc - Yd[:, : self.n - 1]
        for i in range(self.n):
            yield (
                np.hstack([self.X, self.W[:, :i]]),
                self.Y[:, i],
                np.hstack([Xd, Wd[:, :i]]),
                Yd[:, i],
                np.concatenate([self.M[:, i], np.zeros(i)]),
                np.concatenate([omega[i], var_a[:i]]),
                float(self.nu0[i]),
                float(self.S0[i]),
            )

    def posteriors(self, h: Mapping[str, Any]) -> list[NIGPosterior]:
        """δ-coordinate posteriors of every equation (dummies + data); ``log_ml`` holds the data
        term ``log p_i(dummies + data) − log p_i(dummies)``."""
        out = []
        for Z, y, Zd, yd, mu0, var0, nu0, S0 in self.equations(h):
            post = nig_posterior(np.vstack([Zd, Z]), np.concatenate([yd, y]), mu0, var0, nu0, S0)
            if yd.size:
                prior_rows = nig_posterior(Zd, yd, mu0, var0, nu0, S0).log_ml
                post = NIGPosterior(
                    post.theta_hat, post.R, post.nu_hat, post.S_hat, post.log_ml - prior_rows
                )
            out.append(post)
        return out

    def log_ml(self, h: Mapping[str, Any]) -> float:
        return float(sum(post.log_ml for post in self.posteriors(h)))

    def to_gamma(self, post: NIGPosterior, i: int) -> NIGPosterior:
        """``[δ_i | a_i] → [γ_i | a_i]`` with ``γ_i = δ_i + M_c a_i``: ``θ̂^γ = Tθ̂``,
        ``R^γ = RT⁻¹`` (``T⁻¹`` unit upper triangular, so ``R^γ`` stays upper triangular)."""
        k, Mc_i = self.k, self.Mc[:, :i]
        theta = post.theta_hat.copy()
        theta[:k] += Mc_i @ theta[k:]
        R = post.R.copy()
        R[:k, k:] -= R[:k, :k] @ Mc_i
        return NIGPosterior(theta, R, post.nu_hat, post.S_hat, post.log_ml)


# --------------------------------------------------------------------------- public functions


def structural_log_ml(
    data: VARData,
    prior: Minnesota,
    hyper: Mapping[str, float],
    lags: int,
    deterministic: str,
) -> float:
    """``Σ_i log p(y_i | h)``: the Mode II structural marginal likelihood (Chan 2022 eq. 11).

    ``hyper`` may be a Phase 1 hyperparameter dict (``lambda``, ``alpha``, ``psi``, ``soc``,
    ``sur``, ``covid=None``); missing entries come from ``prior`` and ψ from ``window_scales``
    (``psi="ar"``: the Phase 1 AR(p) residual variances). With SOC/SUR dummies it is
    ``Σ_i [log p_i(dummies + data) − log p_i(dummies)]``. ``prior.form`` is not consulted: the
    closed form uses ``lambda_``, ``alpha``, ``cross``, ``contemp``, ``var_const`` and the means.
    At ``cross = contemp = 1`` it equals the Phase 1 ``AnalyticFit.log_ml()`` (A1).
    """
    window = _Window(data, prior, lags, deterministic, psi=hyper.get("psi"))
    return window.log_ml(window.complete(hyper))


def structural_posteriors(
    data: VARData,
    prior: Minnesota,
    hyper: Mapping[str, float],
    lags: int,
    deterministic: str,
) -> tuple[NIGPosterior, ...]:
    """Per-equation Mode II posteriors (dummies + data) in γ-coordinates ``[γ_i | a_i]`` (length
    ``k + i``), the input of :func:`draw_structural_nig`. ``hyper`` as in
    :func:`structural_log_ml`; each ``log_ml`` is the equation's data term, so their sum is
    ``structural_log_ml``. Rows ``γ_i`` of ``Γ`` follow the ``lagged_design`` layout."""
    window = _Window(data, prior, lags, deterministic, psi=hyper.get("psi"))
    h = window.complete(hyper)
    return tuple(window.to_gamma(post, i) for i, post in enumerate(window.posteriors(h)))


def draw_structural_nig(
    posts: Sequence[NIGPosterior], size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """iid joint posterior draws ``a`` (S, n, n−1), ``gamma`` (S, n, k), ``lam`` (S, n).

    ``posts[i]`` is equation ``i`` in γ-coordinates ``[γ_i | a_i]`` (length ``k + i``), as
    :func:`structural_posteriors` returns them. Given the hyperparameters the equations are
    independent (N05 §0 item 1), so each is drawn by :meth:`NIGPosterior.draw` in turn from
    ``rng``. ``a[:, i, j]`` is ``a_ij`` for ``j < i`` and exactly 0 for ``j ≥ i`` (DESIGN §2.7).
    """
    posts = tuple(posts)
    if not posts or not all(isinstance(p, NIGPosterior) for p in posts):
        raise TypeError("posts must be a non-empty sequence of NIGPosterior")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    size = _positive_int(size, "size")
    n, k = len(posts), posts[0].theta_hat.size
    sizes = [p.theta_hat.size for p in posts]
    if sizes != [k + i for i in range(n)]:
        raise ValueError(
            f"posts[i] must have k + i coordinates [γ_i | a_i]; got sizes {sizes} (k = {k})"
        )
    a = np.zeros((size, n, n - 1))
    gamma = np.empty((size, n, k))
    lam = np.empty((size, n))
    for i, post in enumerate(posts):
        theta, lam[:, i] = post.draw(size, rng)
        gamma[:, i] = theta[:, :k]
        a[:, i, :i] = theta[:, k:]
    return a, gamma, lam


def structural_to_reduced(
    a: np.ndarray, gamma: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced form of structural draws (N05 §8): ``coefs`` (S, k, n) with ``B' = A⁻¹Γ`` and
    ``chol`` (S, n, n) ``= A⁻¹Λ^{½}``, the lower Cholesky factor of ``Σ = A⁻¹ΛA⁻ᵀ``.

    ``A = I + a`` is unit lower triangular, so ``A⁻¹C`` is forward substitution
    ``row_j = C_j − Σ_{l<j} a_jl row_l``, batched over draws (no inverse, no factorisation).
    """
    a = np.asarray(a, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    if gamma.ndim != 3 or a.shape != (*gamma.shape[:2], gamma.shape[1] - 1):
        raise ValueError(f"need gamma (S, n, k) and a (S, n, n−1), got {gamma.shape} and {a.shape}")
    if lam.shape != gamma.shape[:2] or not np.all(lam > 0):
        raise ValueError(f"lam must be positive with shape {gamma.shape[:2]}, got {lam.shape}")
    n = gamma.shape[1]
    bt = gamma.copy()  # B' (S, n, k)
    chol = np.zeros((gamma.shape[0], n, n))
    chol[:, np.arange(n), np.arange(n)] = np.sqrt(lam)
    for j in range(1, n):
        bt[:, j] -= np.einsum("sl,slk->sk", a[:, j, :j], bt[:, :j])
        chol[:, j] -= np.einsum("sl,slk->sk", a[:, j, :j], chol[:, :j])
    return bt.transpose(0, 2, 1), chol


def eb_structural(
    data: VARData,
    prior: Minnesota,
    lags: int,
    deterministic: str,
    start: Mapping[str, float],
    *,
    free: Sequence[str] | None = None,
) -> EBResult:
    """EB mode of the free hyperparameters under the structural NIG marginal likelihood (DESIGN
    §6.5 step 2; P3 DESIGN §4.3).

    Maximises ``structural_log_ml + Σ_free log p(h)`` by L-BFGS-B on ``log h`` within the
    hyperprior bounds, starting from ``start`` (a Phase 1 hyperparameter dict, e.g. the conjugate
    twin's mode: ``psi`` may be a vector, ``covid`` must be ``None``), with Phase 1's multistart
    along the dummy axes, one perturbed restart and stationarity test. ``lambda``, ``alpha``,
    ``soc``, ``sur`` have the GLP hyperpriors of ``prior.hyperpriors`` (natural-scale densities);
    ``cross`` and ``contemp`` a flat prior on their logarithm over ``LOG_FLAT_BOX = [1e-3, 10]``,
    so over them the objective is the marginal likelihood (Chan 2022 §4). ``free`` (default
    ``prior.free``, the Phase 2 plug-in: ``cross`` and ``contemp`` fixed) names the optimised
    hyperparameters; entries of ``start`` outside it are held fixed, and ``cross``/``contemp``
    missing from ``start`` come from ``prior``. ``psi`` in the free set raises: with ``cross ≠ 1``
    it is rejected at construction, and at ``cross = 1`` the twin estimates it.
    """
    if not isinstance(prior, Minnesota):
        raise TypeError(f"prior must be a Minnesota prior, got {type(prior).__name__}")
    free = tuple(prior.free) if free is None else tuple(free)
    if "psi" in free:
        raise ValueError(
            "eb_structural estimates lambda, alpha, soc and sur only; psi is free only for the "
            "conjugate twin at cross = 1 (DESIGN §6.5)"
        )
    unknown = [name for name in free if name not in _HYPER_NAMES]
    if unknown or len(set(free)) != len(free):
        raise ValueError(f"free must name distinct hyperparameters of {_HYPER_NAMES}, got {free}")
    window = _Window(data, prior, lags, deterministic, psi=start.get("psi"))
    base = window.complete(start)
    for name in free:
        if base[name] is None:
            raise ValueError(f"{name!r} is a free hyperparameter here and cannot be None")
    hp = prior.hyperpriors
    hyperpriors = {"lambda": hp.lambda_, "alpha": hp.alpha, "soc": hp.soc, "sur": hp.sur}
    hyperpriors.update({name: _LOG_FLAT for name in _SCALE_NAMES})
    lower = np.array([hyperpriors[name].lower for name in free], dtype=np.float64)
    upper = np.array([hyperpriors[name].upper for name in free], dtype=np.float64)
    evaluations = 0

    def merge(x: np.ndarray) -> dict[str, Any]:
        return {**base, **{name: float(v) for name, v in zip(free, x, strict=True)}}

    def evaluate(x: np.ndarray) -> tuple[float, float]:
        """``(log ML, log posterior)`` at natural values ``x`` (clipped into the box)."""
        nonlocal evaluations
        evaluations += 1
        x = np.clip(x, lower, upper)
        try:
            log_ml = window.log_ml(merge(x))
        except np.linalg.LinAlgError:
            return -np.inf, -np.inf
        value = log_ml + sum(hyperpriors[nm].logpdf(v) for nm, v in zip(free, x, strict=True))
        return (log_ml, value) if np.isfinite(value) else (-np.inf, -np.inf)

    if not free:
        log_ml, value = evaluate(np.zeros(0))
        return EBResult(base, (), True, log_ml, value, "no free hyperparameters", 1, 0.0, 0.0)

    lo, hi = np.log(lower), np.log(upper)
    z0 = np.clip(np.log([base[name] for name in free]), lo, hi)
    state: dict[str, Any] = {"z": z0, "f": None}

    def objective(z: np.ndarray) -> float:
        value = evaluate(np.exp(z))[1]
        if np.isfinite(value):
            state["z"], state["f"] = z.copy(), -value
            return -value
        # finite penalty growing away from the last finite point (as Phase 1)
        anchor = 0.0 if state["f"] is None else state["f"]
        return anchor + 1e4 * (1.0 + float(np.sum((z - state["z"]) ** 2)))

    options: dict[str, float] = {"ftol": 1e-12, "maxiter": 2000}
    if any(name in _SCALE_NAMES for name in free):
        options.update(_FREE_SCALE_OPTIONS)

    def run(z_start: np.ndarray):
        res = optimize.minimize(
            objective,
            z_start,
            method="L-BFGS-B",
            jac="3-point",
            bounds=list(zip(lo, hi, strict=True)),
            options=options,
        )
        gain, grad = _stationarity(objective, res.x, lo, hi)
        return res, evaluate(np.exp(res.x))[1], gain, grad

    def passes(candidate) -> bool:
        return bool(np.isfinite(candidate[1]) and candidate[2] < _STATIONARITY_TOL)

    def better(candidate, incumbent) -> bool:
        if not np.isfinite(candidate[1]):
            return False
        if not np.isfinite(incumbent[1]) or candidate[1] > incumbent[1] + 1e-7:
            return True
        return candidate[1] > incumbent[1] - 1e-7 and passes(candidate) and not passes(incumbent)

    starts = [z0]
    for name in ("soc", "sur"):  # known secondary modes along the dummy weights (Phase 1)
        if name in free:
            for v in (0.3, 5.0):
                z = z0.copy()
                z[free.index(name)] = np.log(v)
                starts.append(np.clip(z, lo, hi))
    best = run(starts[0])
    for z in starts[1:]:
        candidate = run(z)
        if better(candidate, best):
            best = candidate
    if not passes(best):
        candidate = run(np.clip(z0 + 0.25, lo, hi))
        if better(candidate, best):
            best = candidate
    result, _, gain, grad = best
    ok = passes(best)
    hyper = merge(np.clip(np.exp(result.x), lower, upper))
    log_ml, value = evaluate(np.exp(result.x))
    messages: tuple[str, ...] = ()
    if not ok:
        messages = (
            f"structural EB (Chan 2022) optimisation did not converge: {result.message}; "
            f"predicted gain {gain:.2g}, projected gradient {grad:.2g}; fit.converged is False",
        )
    return EBResult(
        hyper=hyper,
        free=free,
        converged=ok,
        log_ml=float(log_ml),
        log_posterior=float(value),
        message=str(result.message),
        evaluations=evaluations,
        predicted_gain=float(gain),
        projected_gradient=float(grad),
        warnings=messages,
    )


# --------------------------------------------------------------------------- the fit (P3 §4)

_GLP_NAMES = ("lambda", "alpha", "soc", "sur", "psi")  # Minnesota.free order (HYPER_NAMES)
_ORDERING_N = 4  # the R8 note from this many variables on (as the MCMC engine)
_ORDERING_NOTE = (
    "this composition is not order-invariant: with {n} variables its results depend on their "
    "ordering (R8; N08 §B2); gauge how much with "
    "rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)"
)


class _NIGProblem:
    """The structural NIG log posterior of the free hyperparameters in the form Phase 1's
    ``_metropolis`` reads (``free``, a ``_Free``, and ``log_posterior_vector``; P3 §4.3).

    The GLP names (``lambda``, ``alpha``, ``soc``, ``sur``; ``psi`` only at ``cross = contemp =
    1``) keep Phase 1's bounds and natural-scale log hyperprior (``ConjugateProblem.free`` and
    ``log_prior`` of the conjugate twin, which has the same ψ default); ``cross`` and ``contemp``
    have the flat prior on their log over ``[1e-3, 10]``, whose natural-scale density
    ``1/(κ log(10/1e-3))`` with ``_metropolis``'s log-Jacobian is flat in ``log κ``.
    """

    def __init__(self, fit: StructuralNIGFit, twin_prior: Minnesota, order: tuple[str, ...]):
        from rai_bvar.engines.analytic import ConjugateProblem, _Free

        self.fit = fit
        self.twin = ConjugateProblem(fit.data, twin_prior, fit.lags, fit.deterministic, None)
        tf = self.twin.free
        scale = [name for name in order if name in _SCALE_NAMES]
        low, high = LOG_FLAT_BOX
        self.scale_names = tuple(scale)
        self.free = _Free(
            [*tf.blocks, *[(name, 1, -1) for name in scale]],
            (*tf.names, *scale),
            np.concatenate([tf.lower, np.full(len(scale), low)]),
            np.concatenate([tf.upper, np.full(len(scale), high)]),
            np.concatenate([tf.logit, np.zeros(len(scale), dtype=bool)]),
        )

    def hyper(self, x: np.ndarray) -> dict[str, Any]:
        """The fit's hyperparameters with the free values ``x`` (natural scale, clipped)."""
        return self.free.merge(self.fit.hyper, np.clip(x, self.free.lower, self.free.upper))

    def log_prior(self, h: Mapping[str, Any]) -> float:
        """Natural-scale log hyperprior density of the free names."""
        total = self.twin.log_prior(dict(h))
        return float(total + sum(_LogFlat.logpdf_natural(h[name]) for name in self.scale_names))

    def log_posterior_vector(self, x: np.ndarray) -> float:
        """``Σ_i log p(y_i | h) + log p(h)`` at free values ``x`` (natural scale; clipped)."""
        if not np.all(np.isfinite(x)):
            return -np.inf
        h = self.hyper(x)
        try:
            value = self.fit._log_ml(h) + self.log_prior(h)
        except np.linalg.LinAlgError:
            return -np.inf
        return value if np.isfinite(value) else -np.inf


class StructuralNIGFit(_LinearDraws):
    """A fitted ``AsymmetricConjugate`` BVAR on the analytic engine (P3 DESIGN §4; D-45).

    The prior is read through its internal structural ``Minnesota`` (``priors.scales.
    structural_minnesota``, the adapter of §4.2); the posterior is the per-equation closed form of
    this module: iid draws ``(a, γ, λ) → B = (A⁻¹Γ)ᵀ``, ``chol = A⁻¹Λ^{½}`` and the exact log
    marginal likelihood. The interface is ``AnalyticFit``'s: ``log_ml(hyper)``,
    ``posterior(hyper)`` (the per-equation ``NIGPosterior`` in γ-coordinates ``[γ_i | a_i]``),
    ``draw_parameters``, ``forecast(..., lag_coefs=False)`` (D-49), ``irf``, ``summary()``,
    ``hyper``, ``converged``, ``warnings``, ``mh``, ``mh_seed``, ``optimization``; and
    :meth:`posterior_moments` (``E[B]``, ``E[Σ]``).

    Hyperparameters (§4.3): ``"fixed"`` as given; ``"eb"`` the mode of ``Σ_i log p(y_i | h) +
    log p(h)`` over the free names by :func:`eb_structural` (``cross``, ``contemp`` flat on their
    log over ``[1e-3, 10]``), started from the Phase 1 EB mode of the conjugate twin
    (``Minnesota(form="conjugate", cross = contemp = 1)`` with the same settings and the GLP
    names of the free set) with ``cross`` and ``contemp`` at the prior's values; ``"mh"`` that mode,
    then Phase 1's random-walk Metropolis (``_metropolis``, the prior's ``MHSettings``, the
    seed ``fit(seed=…)`` or ``MHSettings.seed``) over :class:`_NIGProblem`, and iid parameter
    draws per kept MH draw. ``psi`` can be free only at ``cross = contemp = 1`` fixed (E-15),
    where the model is the Phase 1 NIW model and the twin's EB mode is the structural one (A1).

    ``warnings``: ψ notes, Phase 1's setting checks (tcode clash, 2020 coverage), optimisation
    and MH messages, and the R8 note for ``n ≥ 4`` when ``cross ≠ 1`` or ``contemp ≠ 1`` in use
    (then the prior is not the NIW prior and the fit depends on the ordering; Chan 2022).
    """

    def __init__(self, model, data: VARData, seed: int | None):
        from rai_bvar.engines.analytic import _metropolis, _mh_warning, _setting_warnings
        from rai_bvar.priors.asymmetric import AsymmetricConjugate
        from rai_bvar.priors.scales import structural_minnesota

        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0
        ):
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        if not isinstance(data, VARData):
            raise TypeError(f"data must be a VARData, got {type(data).__name__}")
        spec = model.mean.prior
        if not isinstance(spec, AsymmetricConjugate):
            raise TypeError(f"StructuralNIGFit fits AsymmetricConjugate, got {type(spec).__name__}")
        if model.outliers is not None:  # rejected when the BVAR is built (E-8)
            raise ValueError(
                "Lenza-Primiceri scaling is not available with AsymmetricConjugate in this "
                "version; use Minnesota(form='conjugate')"
            )
        self.model, self.data = model, data
        self.lags, self.deterministic = int(model.lags), model.deterministic
        self.prior = structural_minnesota(spec)  # the internal Minnesota (never a spec)
        self.warnings: list[str] = []
        self.optimization: dict[str, Any] | None = None
        self.mh = None
        self.mh_seed = spec.mh.seed if seed is None else int(seed)
        self._windows: dict[Any, _Window] = {}
        base = self._window(None)
        self.k, self.n, self.n_det = base.k, base.n, base.n_det
        periods = data.index[self.lags :]
        shim = _SettingInputs(base.scales.own_means, periods)
        for note in (*base.scales.notes, *_setting_warnings(shim, model, data)):
            self._warn(note)
        free = spec.free
        self.free_names = (
            *(name for name in _GLP_NAMES if name in free),
            *(name for name in _SCALE_NAMES if name in free),
        )
        self._problem: _NIGProblem | None = None
        self._eb_converged = True
        if not self.free_names:
            self.hyper = base.complete({})
        else:
            twin_prior = self._twin_prior(spec)
            self.hyper = self._optimise(spec, twin_prior)
            self._problem = _NIGProblem(self, twin_prior, self.free_names)
            if spec.hierarchy == "mh":
                mode = self._problem.free.vector(self.hyper)
                self.mh = _metropolis(self._problem, mode, spec, self.mh_seed)
                if not self.mh.converged:
                    self._warn(_mh_warning(self.mh, spec))
        if self.n >= _ORDERING_N and (self.hyper["cross"] != 1.0 or self.hyper["contemp"] != 1.0):
            self._warn(_ORDERING_NOTE.format(n=self.n))
        self._posts = self._posteriors(self.hyper)

    def _warn(self, message: str, stacklevel: int = 4) -> None:
        import warnings

        self.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=stacklevel)

    # ------------------------------------------------------------------ hyperparameters

    def _twin_prior(self, spec) -> Minnesota:
        """The conjugate twin's prior (P2 §6.5 step 1): the same settings at ``form="conjugate"``,
        ``cross = contemp = 1``, ``hierarchy="eb"``, with the GLP names of the free set."""
        p = self.prior
        return Minnesota(
            lambda_=p.lambda_,
            alpha=p.alpha,
            psi=p.psi,
            soc=p.soc,
            sur=p.sur,
            soc_scope=p.soc_scope,
            mean=p.mean,
            var_const=p.var_const,
            form="conjugate",
            hierarchy="eb",
            hyper=tuple(name for name in self.free_names if name in _GLP_NAMES),
            hyperpriors=p.hyperpriors,
            mh=spec.mh,
        )

    def _optimise(self, spec, twin_prior: Minnesota) -> dict[str, Any]:
        """The EB mode (P3 §4.3): the conjugate twin's Phase 1 mode as the start, then
        :func:`eb_structural` over the free names (or, with ``psi`` free, the twin's mode)."""
        import warnings

        from rai_bvar.api import BVAR
        from rai_bvar.components import Constant, Linear
        from rai_bvar.engines.analytic import AnalyticFit

        twin_model = BVAR(
            mean=Linear(prior=twin_prior),
            vol=Constant("iw"),
            lags=self.lags,
            deterministic=self.deterministic,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # collected in twin.warnings
            twin = AnalyticFit(twin_model, self.data, None)
        if "psi" in self.free_names:
            if self.prior.cross != 1.0 or self.prior.contemp != 1.0 or "contemp" in spec.free:
                raise ValueError(
                    "empirical Bayes over psi needs cross = contemp = 1 (then the prior is the "
                    "Phase 1 NIW prior); with contemporaneous shrinkage the Chan (2022) marginal "
                    "likelihood is over lambda, cross, contemp and alpha only: remove 'psi' from "
                    "hyper or fix psi"
                )
            # at cross = contemp = 1 the structural marginal likelihood is the NIW one (A1), so
            # the twin's mode is the EB mode; its optimisation messages are this fit's
            dropped = set(twin.problem.notes) | set(self.warnings)
            for message in twin.warnings:
                if message not in dropped:
                    self._warn(message, stacklevel=5)
            self.optimization = twin.optimization
            self._eb_converged = bool(twin.converged)
            window = self._window(twin.hyper["psi"])
            return window.complete({k: v for k, v in twin.hyper.items() if k != "covid"})
        start = {k: v for k, v in twin.hyper.items() if k != "covid"}
        start |= {"cross": self.prior.cross, "contemp": self.prior.contemp}
        names = tuple(name for name in self.free_names if name != "psi")
        result = eb_structural(
            self.data, self.prior, self.lags, self.deterministic, start, free=names
        )
        for message in result.warnings:
            self._warn(message, stacklevel=5)
        self.optimization = {
            "success": bool(result.converged),
            "message": result.message,
            "evaluations": int(result.evaluations),
            "log_posterior": float(result.log_posterior),
            "projected_gradient": float(result.projected_gradient),
            "predicted_gain": float(result.predicted_gain),
        }
        self._eb_converged = bool(result.converged)
        return dict(result.hyper)

    # ------------------------------------------------------------------ inspection

    @property
    def psi(self) -> np.ndarray:
        return np.asarray(self.hyper["psi"])

    @property
    def converged(self) -> bool:
        """EB optimisation succeeded and, with MH, the chains passed R-hat and ESS checks."""
        return self._eb_converged and (self.mh is None or self.mh.converged)

    def _window(self, psi: np.ndarray | None) -> _Window:
        """The window with ψ ``psi`` (``None``: the prior's ψ); the base window and the last other
        one are kept (MH over ψ asks for a new one at every proposal)."""
        key = None if psi is None else np.asarray(psi, dtype=np.float64).tobytes()
        window = self._windows.get(key)
        if window is None:
            window = _Window(self.data, self.prior, self.lags, self.deterministic, psi=psi)
            if len(self._windows) >= 2:
                self._windows.pop(next(k for k in self._windows if k is not None))
            self._windows[key] = window
        return window

    def _merge(self, hyper: Mapping[str, Any] | None) -> dict[str, Any]:
        return {**self.hyper, **(hyper or {})}

    def _log_ml(self, h: Mapping[str, Any]) -> float:
        window = self._window(h["psi"])
        return window.log_ml(window.complete(h))

    def _posteriors(self, h: Mapping[str, Any]) -> tuple[NIGPosterior, ...]:
        window = self._window(h["psi"])
        posts = window.posteriors(window.complete(h))
        return tuple(window.to_gamma(post, i) for i, post in enumerate(posts))

    def log_ml(self, hyper: Mapping[str, Any] | None = None) -> float:
        """``Σ_i log p(y_i | h)``: the data marginal likelihood (Chan 2022 eq. 11), without
        hyperprior terms, at the fit's hyperparameters overridden by ``hyper``."""
        return self._log_ml(self._merge(hyper))

    def posterior(self, hyper: Mapping[str, Any] | None = None) -> tuple[NIGPosterior, ...]:
        """Per-equation posteriors in γ-coordinates ``[γ_i | a_i]`` (dummies + data), at the fit's
        hyperparameters overridden by ``hyper``."""
        return self._posts if hyper is None else self._posteriors(self._merge(hyper))

    def posterior_moments(
        self, hyper: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact posterior means ``(E[B] (k, n), E[Σ] (n, n))`` (P3 §9.7, E127).

        Given the hyperparameters the equations are independent, and row ``i`` of ``A⁻¹C`` is
        ``C_i − a_iᵀ(A⁻¹C)_{<i}`` with ``(A⁻¹C)_{<i}`` a function of the earlier equations only, so

            E[B]ᵀ_i = E[γ_i] − E[a_i]ᵀ E[B]ᵀ_{<i}               (E[B] = (E[A]⁻¹E[Γ])ᵀ),
            E[Σ]_{i,<i} = −E[a_i]ᵀ E[Σ]_{<i,<i},
            E[Σ]_{ii} = E[λ_i] + tr(E[Σ]_{<i,<i} E[a_i a_iᵀ]),

        from ``Σ = A⁻¹ΛA⁻ᵀ`` and each equation's second moments ``E[a_i a_iᵀ] = E[λ_i](K⁻¹)_{aa} +
        â_i â_iᵀ``, ``E[λ_i] = Ŝ/(ν̂ − 1)``.
        """
        posts = self.posterior(hyper)
        k, n = self.k, self.n
        bt = np.zeros((n, k))
        sigma = np.zeros((n, n))
        for i, post in enumerate(posts):
            gamma, a = post.theta_hat[:k], post.theta_hat[k:]
            bt[i] = gamma - a @ bt[:i]
            prev = sigma[:i, :i]
            second = post.cov()[k:, k:] + np.outer(a, a)
            sigma[i, :i] = sigma[:i, i] = -a @ prev
            sigma[i, i] = post.lam_mean + float(np.sum(prev * second))
        return bt.T, sigma

    def summary(self):
        """Hyperparameter values in use and, with MH, posterior quantiles and diagnostics."""
        import pandas as pd

        if self._problem is None:
            return pd.DataFrame({"mode": np.zeros(0)}, index=[])
        free = self._problem.free
        frame = pd.DataFrame({"mode": free.vector(self.hyper)}, index=list(free.names))
        if self.mh is not None:
            q = np.quantile(
                self.mh.chains.reshape(-1, len(self.mh.names)), [0.05, 0.5, 0.95], axis=0
            )
            frame["q05"], frame["q50"], frame["q95"] = q
            frame["rhat"] = [self.mh.rhat[nm] for nm in frame.index]
            frame["ess"] = [self.mh.ess[nm] for nm in frame.index]
        return frame

    # ------------------------------------------------------------------ draws

    def _layout(self) -> tuple[int, int, int, None]:
        return self.k, self.n, self.n_det, None

    def _iter_draws(self, size: int, rng: np.random.Generator):
        """Parameter draws in chunks of bounded memory: ``(slice, coefs, chol, None)``, as
        ``AnalyticFit._iter_draws``: with MH, ``size`` hyperparameter draws taken evenly from the
        pooled chains and one ``(a, γ, λ)`` draw from the conditional posterior of each; fixed
        blocks of ``_DRAW_BLOCK`` draws with their own streams (chunk-size invariance)."""
        from rai_bvar.engines.analytic import _CHUNK_ELEMENTS, _DRAW_BLOCK

        if size < 1:
            raise ValueError("draws must be positive")
        chunk = max(1, _CHUNK_ELEMENTS // (self.k * self.n))
        if self.mh is None:
            groups = [(self._posts, self.hyper, size)]
        else:
            pooled = self.mh.chains.reshape(-1, self.mh.chains.shape[-1])
            picks = np.round(np.linspace(0, len(pooled) - 1, size)).astype(int)
            unique, counts = np.unique(pooled[picks], axis=0, return_counts=True)
            groups = [
                (None, self._problem.hyper(x), int(c)) for x, c in zip(unique, counts, strict=True)
            ]
        blocks = []
        for gid, (posts, h, count) in enumerate(groups):
            while count > 0:
                take = min(count, _DRAW_BLOCK)
                blocks.append((gid, posts, h, take))
                count -= take
        streams = np.random.default_rng(rng.integers(0, 2**63)).spawn(len(blocks))
        batches, batch, filled = [], [], 0
        for b, block in enumerate(blocks):
            if batch and filled + block[3] > chunk:
                batches.append(batch)
                batch, filled = [], 0
            batch.append((b, *block))
            filled += block[3]
        batches.append(batch)
        cached: tuple[int, tuple[NIGPosterior, ...]] | None = None
        start = 0
        for batch in batches:
            total = sum(block[4] for block in batch)
            coefs = np.empty((total, self.k, self.n))
            chol = np.empty((total, self.n, self.n))
            pos = 0
            for b, gid, posts, h, take in batch:
                if posts is None:
                    if cached is None or cached[0] != gid:
                        cached = (gid, self._posteriors(h))
                    posts = cached[1]
                a, gamma, lam = draw_structural_nig(posts, take, streams[b])
                coefs[pos : pos + take], chol[pos : pos + take] = structural_to_reduced(
                    a, gamma, lam
                )
                pos += take
            yield slice(start, start + total), coefs, chol, None
            start += total


@dataclass(frozen=True)
class _SettingInputs:
    """What Phase 1's ``_setting_warnings`` reads of a problem: own-lag means and the periods."""

    means: np.ndarray
    periods: Any


def _check_positive(key: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise TypeError(f"hyperparameter {key!r} must be a number, got {value!r}")
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"hyperparameter {key!r} must be positive and finite, got {value!r}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


__all__ = [
    "EBResult",
    "NIGPosterior",
    "StructuralNIGFit",
    "draw_structural_nig",
    "eb_structural",
    "nig_posterior",
    "structural_log_ml",
    "structural_posteriors",
    "structural_to_reduced",
]
