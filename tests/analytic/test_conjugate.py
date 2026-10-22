"""Exact analytic checks of the conjugate Normal-inverse-Wishart engine (SPEC §4.6, §10.2).

The oracles below are written out from the textbook formulas (Kadiyala & Karlsson 1997; GLP
2015): explicit inverses, Kronecker products and SciPy densities are fine in tests. None of
them calls rai_bvar's linear algebra; only the fitted model under test does.

Model: ``y_t = B'x_t + s_t u_t`` with ``u_t ~ N(0, Σ)``, ``Σ ~ IW(diag(ψ), d = n + 2)`` and
``vec(B) | Σ ~ N(vec(b), Σ ⊗ Ω)``. ``s_t = 1`` except under the Lenza-Primiceri scaling.

* Chib (1995) identity: for any ``(B, Σ)``,
  ``log p(Y) = log p(Y | B, Σ) + log p(B, Σ) − log p(B, Σ | Y)``. With SOC/SUR dummies the prior
  is the NIW obtained by updating the Minnesota prior with the dummy rows, so the identity
  checks ``ML(dummies + data) − ML(dummies)``; with heteroskedastic rows it checks the
  Lenza-Primiceri Jacobian ``−n Σ log s_t``.
* ``n = p = 1``, ``T = 20``: log ML and posterior moments by tensor Gauss-Legendre quadrature of
  ``p(Y | c, a, σ²) p(c, a | σ²) p(σ²)`` (SPEC §10.2: relative tolerance 1e-6).
* Posterior ``B̄``, ``Ω̄``, ``S̄``, ``ν̄`` against the explicit-inverse formulas.
* Monte Carlo moments of ``draw_parameters`` against the analytic NIW moments.
* ``h = 1`` predictive density: ``y_{T+1} | Y ~ t_{ν̄−n+1}(B̄'x, S̄(s² + x'Ω̄x)/(ν̄ − n + 1))``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.special import logsumexp

import rai_bvar as rb

QD_SMALL = Path(__file__).resolve().parents[1] / "parity" / "fixtures" / "qd_small_levels.csv"
_LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- textbook oracles


def matrix_normal_logpdf(B: np.ndarray, M: np.ndarray, P: np.ndarray, V: np.ndarray) -> float:
    """``log MN(B; M, P⁻¹, V)``: ``vec(B) ~ N(vec(M), V ⊗ P⁻¹)`` with row precision ``P`` (k × k)."""
    k, n = M.shape
    D = B - M
    quad = np.trace(np.linalg.solve(V, D.T @ P @ D))
    return -0.5 * (
        k * n * _LOG_2PI - n * np.linalg.slogdet(P)[1] + k * np.linalg.slogdet(V)[1] + quad
    )


@dataclass(frozen=True)
class NIW:
    """``Σ ~ IW(scale, df)`` and ``B | Σ ~ MN(mean, precision⁻¹, Σ)``."""

    mean: np.ndarray  # (k, n)
    precision: np.ndarray  # (k, k) inverse row covariance
    scale: np.ndarray  # (n, n)
    df: float

    @property
    def omega(self) -> np.ndarray:
        """Row covariance (explicit inverse)."""
        return np.linalg.inv(self.precision)

    def update(self, Y: np.ndarray, X: np.ndarray) -> NIW:
        """Conjugate update with the rows ``(Y, X)``: the normal equations, solved directly."""
        precision = X.T @ X + self.precision
        mean = np.linalg.solve(precision, X.T @ Y + self.precision @ self.mean)
        resid, gap = Y - X @ mean, mean - self.mean
        scale = self.scale + resid.T @ resid + gap.T @ self.precision @ gap
        return NIW(mean, precision, 0.5 * (scale + scale.T), self.df + Y.shape[0])

    def logpdf(self, B: np.ndarray, sigma: np.ndarray) -> float:
        iw = stats.invwishart(df=self.df, scale=self.scale).logpdf(sigma)
        return matrix_normal_logpdf(B, self.mean, self.precision, sigma) + float(iw)

    @property
    def sigma_mean(self) -> np.ndarray:
        return self.scale / (self.df - self.scale.shape[0] - 1.0)


def gaussian_loglik(
    Y: np.ndarray, X: np.ndarray, B: np.ndarray, sigma: np.ndarray, s: np.ndarray
) -> float:
    """``Σ_t log N(y_t; B'x_t, s_t² Σ)``."""
    return float(
        sum(
            stats.multivariate_normal(X[t] @ B, s[t] ** 2 * sigma).logpdf(Y[t])
            for t in range(Y.shape[0])
        )
    )


@dataclass(frozen=True)
class Case:
    """One conjugate model: Minnesota hyperparameters, dummies, design and COVID scaling."""

    n: int
    p: int
    lam: float = 0.35
    alpha: float = 1.5
    soc: float | None = None
    sur: float | None = None
    deterministic: str = "c"
    var_const: float = 1e7
    own_mean: float = 1.0
    psi: tuple[float, ...] | None = None  # defaults to PSI[:n]
    covid: tuple[float, float, float, float] | None = None
    covid_start: str | None = None

    @property
    def psi_values(self) -> np.ndarray:
        return np.array(self.psi if self.psi is not None else PSI[: self.n])

    @property
    def n_det(self) -> int:
        return {"n": 0, "c": 1, "ct": 2}[self.deterministic]

    def model(self, hierarchy: str = "fixed", **prior) -> rb.BVAR:
        minnesota = rb.Minnesota(
            lambda_=self.lam,
            alpha=self.alpha,
            psi=tuple(self.psi_values),
            soc=self.soc,
            sur=self.sur,
            mean=self.own_mean,
            var_const=self.var_const,
            hierarchy=hierarchy,
            **prior,
        )
        outliers = None
        if self.covid is not None:
            outliers = rb.LenzaPrimiceri(start=self.covid_start, fixed=self.covid)
        return rb.BVAR(
            mean=rb.Linear(prior=minnesota),
            lags=self.p,
            deterministic=self.deterministic,
            outliers=outliers,
        )

    # -- oracle pieces, written out from the model description

    def design(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``x_t = (1, t, y_{t−1}', …, y_{t−p}')'`` with ``t`` the 1-based position in ``values``."""
        rows = []
        for t in range(self.p, values.shape[0]):
            det = [1.0, t + 1.0][: self.n_det]
            rows.append(np.concatenate([det, *(values[t - lag] for lag in range(1, self.p + 1))]))
        return values[self.p :], np.array(rows)

    def minnesota(self) -> NIW:
        n, d = self.n, self.n + 2.0
        omega = [self.var_const] * self.n_det
        for lag in range(1, self.p + 1):
            for j in range(n):
                omega.append(self.lam**2 / (lag**self.alpha * self.psi_values[j] / (d - n - 1.0)))
        b = np.zeros((self.n_det + n * self.p, n))
        for j in range(n):
            b[self.n_det + j, j] = self.own_mean
        return NIW(b, np.diag(1.0 / np.array(omega)), np.diag(self.psi_values), d)

    def dummies(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ybar0 = values[: self.p].mean(axis=0)
        rows_y, rows_x = [], []
        if self.soc is not None:  # n rows: diag(ȳ0)/μ on y and on every lag, 0 on deterministics
            rows_y.append(np.diag(ybar0) / self.soc)
            lagged = np.tile(np.diag(ybar0) / self.soc, (1, self.p))
            rows_x.append(np.hstack([np.zeros((self.n, self.n_det)), lagged]))
        if self.sur is not None:  # one row: ȳ0'/δ on y and every lag, 1/δ on the constant
            rows_y.append(ybar0[None, :] / self.sur)
            det = np.zeros((1, self.n_det))
            if self.n_det:
                det[0, 0] = 1.0 / self.sur
            rows_x.append(np.hstack([det, np.tile(ybar0 / self.sur, self.p)[None, :]]))
        k = self.n_det + self.n * self.p
        if not rows_y:
            return np.zeros((0, self.n)), np.zeros((0, k))
        return np.vstack(rows_y), np.vstack(rows_x)

    def scales(self, periods: pd.PeriodIndex) -> np.ndarray:
        """Lenza-Primiceri ``s_t``: 1 before t*, then s0, s1, s2 and ``1 + (s2 − 1)ρ^{j−2}``."""
        if self.covid is None:
            return np.ones(len(periods))
        s0, s1, s2, rho = self.covid
        t_star = pd.Period(self.covid_start or "2020Q1", freq="Q").ordinal
        out = []
        for period in periods:
            j = period.ordinal - t_star
            if j < 0:
                out.append(1.0)
            elif j <= 2:
                out.append((s0, s1, s2)[j])
            else:
                out.append(1.0 + (s2 - 1.0) * rho ** (j - 2))
        return np.array(out)

    def oracle(self, frame: pd.DataFrame) -> Oracle:
        values = frame.to_numpy(dtype=np.float64)
        Y, X = self.design(values)
        prior = self.minnesota()
        Yd, Xd = self.dummies(values)
        if Yd.shape[0]:
            prior = prior.update(Yd, Xd)
        s = self.scales(frame.index[self.p :])
        posterior = prior.update(Y / s[:, None], X / s[:, None])
        return Oracle(Y, X, s, prior, posterior)


@dataclass(frozen=True)
class Oracle:
    Y: np.ndarray
    X: np.ndarray
    s: np.ndarray
    prior: NIW  # Minnesota prior updated with the SOC/SUR dummy rows
    posterior: NIW

    def chib_log_ml(self, B: np.ndarray, sigma: np.ndarray) -> float:
        return (
            gaussian_loglik(self.Y, self.X, B, sigma, self.s)
            + self.prior.logpdf(B, sigma)
            - self.posterior.logpdf(B, sigma)
        )

    def points(self, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        """Two evaluation points: the posterior mode of Σ with B̄, and an inflated draw-like point."""
        post = self.posterior
        k, n = post.mean.shape
        centre = (post.mean, post.scale / (post.df + n + 1.0))
        A = rng.standard_normal((n, n))
        sigma = 1.5 * post.sigma_mean + 0.1 * (A @ A.T) * np.sqrt(np.diag(post.sigma_mean)).mean()
        Z = rng.standard_normal((k, n))
        row = np.linalg.cholesky(post.omega)
        other = (post.mean + 1.5 * row @ Z @ np.linalg.cholesky(sigma).T, sigma)
        return [centre, other]


PSI = (0.8, 0.3, 3.0)
COLUMNS = ("y1", "y2", "y3")


def var_frame(
    n: int, T: int, seed: int, start: str = "1985Q1", scales: np.ndarray | None = None
) -> pd.DataFrame:
    """A persistent VAR(1) in levels around non-zero means (so ȳ0 and the dummies matter)."""
    rng = np.random.default_rng(seed)
    mu = np.array([4.0, -2.0, 10.0])[:n]
    A = 0.8 * np.eye(n) + 0.05 * (np.ones((n, n)) - np.eye(n))
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    chol = np.diag([1.0, 0.5, 2.0][:n]) @ np.linalg.cholesky(corr)
    s = np.ones(T) if scales is None else scales
    y, prev = np.empty((T, n)), mu.copy()
    for t in range(T):
        prev = mu + A @ (prev - mu) + s[t] * chol @ rng.standard_normal(n)
        y[t] = prev
    index = pd.period_range(start, periods=T, freq="Q")
    return pd.DataFrame(y, index=index, columns=list(COLUMNS[:n]))


def to_data(frame: pd.DataFrame) -> rb.VARData:
    return rb.VARData.from_frame(frame, tcodes=1)


def qd_small_frame() -> pd.DataFrame:
    df = pd.read_csv(QD_SMALL, index_col="date")
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


# --------------------------------------------------------------------------- oracle self-checks


def test_matrix_normal_oracle_matches_kronecker_gaussian():
    rng = np.random.default_rng(0)
    k, n = 4, 3
    U = np.cov(rng.standard_normal((k, 3 * k))) + np.eye(k)
    V = np.cov(rng.standard_normal((n, 3 * n))) + np.eye(n)
    M, B = rng.standard_normal((2, k, n))
    vec = stats.multivariate_normal(M.T.reshape(-1), np.kron(V, U)).logpdf(B.T.reshape(-1))
    assert matrix_normal_logpdf(B, M, np.linalg.inv(U), V) == pytest.approx(vec, rel=1e-12)


def test_niw_update_oracle_matches_completing_the_square():
    """``S̄ = Ψ + Y'Y + b'Ω⁻¹b − B̄'Ω̄⁻¹B̄`` equals the residual form used by ``NIW.update``."""
    case = Case(n=2, p=1)
    oracle = case.oracle(var_frame(2, 40, seed=1))
    prior, post, Y, X = case.minnesota(), oracle.posterior, oracle.Y, oracle.X
    P0 = np.diag(1.0 / np.diag(prior.omega))  # the Minnesota Ω is diagonal
    S = (
        prior.scale
        + Y.T @ Y
        + prior.mean.T @ P0 @ prior.mean
        - post.mean.T @ (X.T @ X + P0) @ post.mean
    )
    np.testing.assert_allclose(post.scale, S, rtol=1e-8)
    np.testing.assert_allclose(post.mean, np.linalg.solve(X.T @ X + P0, X.T @ Y + P0 @ prior.mean))


# --------------------------------------------------------------------------- Chib identity

DUMMIES = {
    "none": {},
    "soc": {"soc": 0.7},
    "sur": {"sur": 1.3},
    "soc+sur": {"soc": 0.7, "sur": 1.3},
}

CHIB_CASES = [
    pytest.param(Case(n=n, p=p, **dummies), id=f"n{n}-p{p}-{label}")
    for n in (1, 2, 3)
    for p in (1, 2)
    for label, dummies in DUMMIES.items()
]
EXTRA_CASES = [
    pytest.param(Case(n=2, p=2, soc=0.5, sur=2.0, var_const=4.0), id="tight-constant"),
    pytest.param(Case(n=3, p=1, soc=0.7, sur=1.3, deterministic="n"), id="no-constant"),
    pytest.param(Case(n=2, p=2, soc=0.7, deterministic="ct"), id="trend-soc"),
    pytest.param(Case(n=3, p=3, lam=0.1, alpha=2.0, own_mean=0.0), id="white-noise-mean"),
    pytest.param(Case(n=1, p=2, lam=2.0, alpha=1.0, soc=5.0, sur=0.2), id="loose-lambda"),
]


@pytest.mark.parametrize("case", CHIB_CASES + EXTRA_CASES)
def test_log_ml_satisfies_the_chib_identity(case: Case):
    frame = var_frame(case.n, 60, seed=10 * case.n + case.p)
    fit = case.model().fit(to_data(frame))
    oracle = case.oracle(frame)
    log_ml = fit.log_ml()
    assert np.isfinite(log_ml)
    for B, sigma in oracle.points(np.random.default_rng(case.n)):
        assert log_ml == pytest.approx(oracle.chib_log_ml(B, sigma), rel=1e-12, abs=1e-9)


def test_log_ml_chib_identity_on_quarterly_levels():
    """GLP setting on real data: three 100·log levels, p = 4, SOC + SUR, AR(4) ψ."""
    frame = qd_small_frame()
    data = to_data(frame)
    base = Case(n=3, p=4, lam=0.2, alpha=2.0, soc=1.0, sur=1.0)
    psi = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(hierarchy="fixed")), lags=4).fit(data).psi
    case = replace(base, psi=tuple(psi))
    fit = case.model().fit(data)
    oracle = case.oracle(frame)
    for B, sigma in oracle.points(np.random.default_rng(4)):
        assert fit.log_ml() == pytest.approx(oracle.chib_log_ml(B, sigma), rel=1e-9, abs=1e-6)


def test_log_ml_with_dummies_is_the_difference_of_two_marginal_likelihoods():
    """``ML(data | dummies) = ML(dummies + data) / ML(dummies)``, each by the Chib identity."""
    case = Case(n=2, p=2, soc=0.7, sur=1.3)
    frame = var_frame(2, 60, seed=3)
    values = frame.to_numpy()
    Y, X = case.design(values)
    Yd, Xd = case.dummies(values)
    minnesota = case.minnesota()

    def chib(rows_y, rows_x):
        post = minnesota.update(rows_y, rows_x)
        B, sigma = post.mean, post.sigma_mean
        ones = np.ones(rows_y.shape[0])
        return (
            gaussian_loglik(rows_y, rows_x, B, sigma, ones)
            + minnesota.logpdf(B, sigma)
            - post.logpdf(B, sigma)
        )

    expected = chib(np.vstack([Yd, Y]), np.vstack([Xd, X])) - chib(Yd, Xd)
    assert case.model().fit(to_data(frame)).log_ml() == pytest.approx(expected, rel=1e-12)


def test_log_ml_at_other_hyperparameters():
    """``log_ml(hyper)`` and ``posterior(hyper)`` evaluate at the given values, not the fitted ones."""
    case = Case(n=3, p=2, soc=0.7, sur=1.3)
    frame = var_frame(3, 60, seed=5)
    fit = case.model().fit(to_data(frame))
    other = replace(case, lam=0.9, alpha=1.2, soc=3.0, sur=0.4, psi=(2.0, 0.1, 5.0))
    hyper = {"lambda": 0.9, "alpha": 1.2, "soc": 3.0, "sur": 0.4, "psi": [2.0, 0.1, 5.0]}
    oracle = other.oracle(frame)
    B, sigma = oracle.posterior.mean, oracle.posterior.sigma_mean
    assert fit.log_ml(hyper) == pytest.approx(oracle.chib_log_ml(B, sigma), rel=1e-12)
    np.testing.assert_allclose(fit.posterior(hyper).mean, oracle.posterior.mean, rtol=1e-10)
    assert fit.log_ml(hyper) != pytest.approx(fit.log_ml())


# --------------------------------------------------------------------------- Lenza-Primiceri

COVID_CASES = [
    pytest.param(Case(n=2, p=2, covid=(5.0, 3.0, 2.0, 0.6)), id="default-start"),
    pytest.param(
        Case(n=3, p=1, soc=0.7, sur=1.3, covid=(8.0, 1.0, 1.5, 0.3)), id="dummies-default-start"
    ),
    pytest.param(
        Case(n=2, p=2, soc=1.0, covid=(2.5, 1.5, 1.2, 0.9), covid_start="2019Q3"), id="custom-start"
    ),
    pytest.param(Case(n=1, p=1, covid=(4.0, 2.0, 3.0, 0.0)), id="n1-no-decay"),
]


def _covid_frame(case: Case, end: str = "2021Q4") -> pd.DataFrame:
    index = pd.period_range("2006Q1", end, freq="Q")
    return var_frame(case.n, len(index), seed=7, start="2006Q1", scales=case.scales(index))


@pytest.mark.parametrize("case", COVID_CASES)
def test_lenza_primiceri_jacobian_by_the_heteroskedastic_chib_identity(case: Case):
    """Rows divided by ``s_t`` plus the Jacobian ``−n Σ log s_t`` is the exact log ML under
    ``y_t = B'x_t + s_t u_t`` (SPEC §4.6)."""
    frame = _covid_frame(case)
    fit = case.model().fit(to_data(frame))
    oracle = case.oracle(frame)
    assert np.any(oracle.s > 1.0)
    assert oracle.s[frame.index[case.p :] < pd.Period(case.covid_start or "2020Q1", "Q")].max() == 1
    for B, sigma in oracle.points(np.random.default_rng(1)):
        assert fit.log_ml() == pytest.approx(oracle.chib_log_ml(B, sigma), rel=1e-12, abs=1e-9)
    np.testing.assert_allclose(fit.posterior().mean, oracle.posterior.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.posterior().scale, oracle.posterior.scale, rtol=1e-10)


def test_lenza_primiceri_with_unit_scales_is_homoskedastic():
    case = Case(n=2, p=2, soc=0.7, covid=(1.0, 1.0, 1.0, 0.0))
    frame = _covid_frame(case)
    plain = replace(case, covid=None).model().fit(to_data(frame)).log_ml()
    assert case.model().fit(to_data(frame)).log_ml() == pytest.approx(plain, rel=1e-12)


# --------------------------------------------------------------------------- posterior moments


@pytest.mark.parametrize("case", CHIB_CASES + EXTRA_CASES)
def test_posterior_matches_textbook_formulas(case: Case):
    """``Ω̄ = (X*'X* + Ω⁻¹)⁻¹``, ``B̄ = Ω̄(X*'Y* + Ω⁻¹b)``, ``S̄ = Ψ + Ê'Ê + (B̄ − b)'Ω⁻¹(B̄ − b)``,
    ``ν̄ = d + T + (dummy rows)``, with ``X*``, ``Y*`` the dummy rows stacked on the data."""
    frame = var_frame(case.n, 60, seed=10 * case.n + case.p)
    post = case.model().fit(to_data(frame)).posterior()
    expected = case.oracle(frame).posterior
    rows = (60 - case.p) + (case.n if case.soc else 0) + (1 if case.sur else 0)
    assert post.df == expected.df == case.n + 2 + rows
    np.testing.assert_allclose(post.mean, expected.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(post.omega(), expected.omega, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(post.scale, expected.scale, rtol=1e-10)
    np.testing.assert_allclose(post.sigma_mean, expected.sigma_mean, rtol=1e-10)
    L = post.scale_chol
    np.testing.assert_array_equal(np.triu(L, k=1), 0.0)
    assert np.all(np.diag(L) > 0)
    R = post.row_upper
    np.testing.assert_array_equal(np.tril(R, k=-1), 0.0)


def test_default_psi_is_the_ar_residual_variance():
    """ψ = ``"ar"``: residual variance of a univariate AR(p) with constant, SSR / (T − p − (p + 1))."""
    frame = var_frame(3, 60, seed=2)
    p = 3
    fit = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(hierarchy="fixed")), lags=p).fit(to_data(frame))
    for j, column in enumerate(frame.columns):
        Y, X = Case(n=1, p=p).design(frame[[column]].to_numpy())
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        assert fit.psi[j] == pytest.approx(float(resid[:, 0] @ resid[:, 0]) / (len(Y) - p - 1))


# --------------------------------------------------------------------------- quadrature (n = p = 1)


def _gauss_legendre(lo: float, hi: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(m)
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), half * w


def quadrature(Y: np.ndarray, X: np.ndarray, case: Case, n_u: int = 400, n_z: int = 96):
    """``log ∫ p(Y | c, a, σ²) p(c, a | σ²) p(σ²)`` and the posterior means of ``(c, a, σ²)``.

    ``p(c, a | σ²) = N(c; 0, σ² v_c) N(a; m, σ² ω_a)`` and ``p(σ²) = IG(d/2, ψ/2)`` (IW for
    ``n = 1``). Tensor Gauss-Legendre in ``u = log σ²`` and ``z`` with ``B = m̃ + σ L z``; the
    centre ``m̃`` and the factor ``L`` only place the grid (any affine map is a valid change of
    variables), the integrand is evaluated from SciPy densities.
    """
    prior = case.minnesota()
    guide = prior.update(Y, X)
    centre, L = guide.mean[:, 0], np.linalg.cholesky(guide.omega)
    nu, s2 = guide.df, guide.scale[0, 0]
    u0 = np.log(s2 / nu)  # mode of the marginal posterior of log σ²
    u, wu = _gauss_legendre(u0 - 7.0, u0 + 8.0 + 80.0 / nu, n_u)
    z, wz = _gauss_legendre(-12.0, 12.0, n_z)
    Z = np.stack(np.meshgrid(z, z, indexing="ij"), axis=-1).reshape(-1, 2) @ L.T
    log_wz = (np.log(wz)[:, None] + np.log(wz)[None, :]).reshape(-1)
    b, v = prior.mean[:, 0], np.diag(prior.omega)
    sigma_prior = stats.invgamma(a=prior.df / 2.0, scale=prior.scale[0, 0] / 2.0)
    log_det_l = float(np.log(np.diag(L)).sum())

    terms, values = [], []
    for chunk in np.array_split(np.arange(n_u), 10):
        sigma2 = np.exp(u[chunk])[:, None]
        sigma = np.sqrt(sigma2)
        c = centre[0] + sigma * Z[None, :, 0]
        a = centre[1] + sigma * Z[None, :, 1]
        log_f = stats.norm.logpdf(c, b[0], np.sqrt(sigma2 * v[0]))
        log_f += stats.norm.logpdf(a, b[1], np.sqrt(sigma2 * v[1]))
        log_f += sigma_prior.logpdf(sigma2)
        for t in range(Y.shape[0]):
            log_f += stats.norm.logpdf(Y[t, 0], X[t, 0] * c + X[t, 1] * a, sigma)
        # dB = σ² |L| dz (k = 2) and dσ² = σ² du
        log_f += 2.0 * np.log(sigma2) + log_det_l + np.log(wu[chunk])[:, None] + log_wz[None, :]
        terms.append(log_f.reshape(-1))
        values.append(np.stack([c, a, np.broadcast_to(sigma2, c.shape)], axis=-1).reshape(-1, 3))
    terms, values = np.concatenate(terms), np.concatenate(values)
    log_ml = float(logsumexp(terms))
    weights = np.exp(terms - log_ml)
    return log_ml, weights @ values


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case(n=1, p=1, lam=0.3, psi=(0.8,)), id="minnesota"),
        pytest.param(Case(n=1, p=1, lam=0.3, psi=(0.8,), soc=0.7, sur=1.3), id="soc+sur"),
        pytest.param(Case(n=1, p=1, lam=0.5, psi=(2.0,), var_const=3.0, own_mean=0.0), id="tight"),
    ],
)
def test_n1_p1_log_ml_and_moments_by_quadrature(case: Case):
    """SPEC §10.2: ``n = 1, p = 1, T = 20``, relative tolerance 1e-6 on the ML and the moments."""
    frame = var_frame(1, 21, seed=11)
    fit = case.model().fit(to_data(frame))
    values = frame.to_numpy()
    Y, X = case.design(values)
    assert Y.shape == (20, 1)
    Yd, Xd = case.dummies(values)
    log_ml, moments = quadrature(np.vstack([Yd, Y]), np.vstack([Xd, X]), case)
    if Yd.shape[0]:
        log_ml -= quadrature(Yd, Xd, case)[0]
    assert fit.log_ml() == pytest.approx(log_ml, abs=1e-6)
    post = fit.posterior()
    np.testing.assert_allclose(post.mean[:, 0], moments[:2], rtol=1e-6)
    assert post.sigma_mean[0, 0] == pytest.approx(moments[2], rel=1e-6)


# --------------------------------------------------------------------------- Monte Carlo moments


def _check_draw_moments(coefs, chol, mean, cov_vec, sigma_mean, precision_mean, z=5.0):
    S, _, n = coefs.shape
    assert chol.shape == (S, n, n)
    assert np.all(np.triu(chol, k=1) == 0.0)
    assert np.all(np.diagonal(chol, axis1=1, axis2=2) > 0)
    vec = np.swapaxes(coefs, 1, 2).reshape(S, -1)  # column-major vec(B)
    se = np.sqrt(np.diag(cov_vec) / S)
    assert np.all(np.abs(vec.mean(axis=0) - mean.T.reshape(-1)) < z * se)
    d = np.sqrt(np.diag(cov_vec))
    se_cov = np.sqrt((np.outer(d, d) ** 2 + cov_vec**2) / S)
    assert np.all(np.abs(np.cov(vec, rowvar=False) - cov_vec) < z * se_cov)
    sigma = chol @ np.swapaxes(chol, 1, 2)
    se = sigma.std(axis=0) / np.sqrt(S)
    assert np.all(np.abs(sigma.mean(axis=0) - sigma_mean) < z * se)
    precision = np.linalg.inv(sigma)
    se = precision.std(axis=0) / np.sqrt(S)
    assert np.all(np.abs(precision.mean(axis=0) - precision_mean) < z * se)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(Case(n=3, p=2, soc=0.7, sur=1.3), id="n3-p2-soc+sur"),
        pytest.param(Case(n=1, p=1), id="n1-p1"),
        pytest.param(Case(n=2, p=1, covid=(5.0, 3.0, 2.0, 0.6)), id="n2-covid"),
    ],
)
def test_draw_parameters_moments_match_the_niw_posterior(case: Case):
    """``E[B] = B̄``, ``Cov(vec B) = E[Σ] ⊗ Ω̄``, ``E[Σ] = S̄/(ν̄ − n − 1)``, ``E[Σ⁻¹] = ν̄ S̄⁻¹``."""
    frame = _covid_frame(case) if case.covid else var_frame(case.n, 60, seed=10 * case.n + case.p)
    fit = case.model().fit(to_data(frame))
    post = case.oracle(frame).posterior
    coefs, chol = fit.draw_parameters(40_000, rng=123)
    _check_draw_moments(
        coefs,
        chol,
        post.mean,
        np.kron(post.sigma_mean, post.omega),
        post.sigma_mean,
        post.df * np.linalg.inv(post.scale),
    )
    a, b = fit.draw_parameters(50, rng=7), fit.draw_parameters(50, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_eb_draws_use_the_posterior_at_the_mode():
    case = Case(n=2, p=2, soc=1.0, sur=1.0)
    frame = var_frame(2, 80, seed=21)
    fit = case.model(hierarchy="eb").fit(to_data(frame))
    h = fit.hyper
    at_mode = replace(case, lam=h["lambda"], soc=h["soc"], sur=h["sur"])
    post = at_mode.oracle(frame).posterior
    coefs, chol = fit.draw_parameters(40_000, rng=9)
    _check_draw_moments(
        coefs,
        chol,
        post.mean,
        np.kron(post.sigma_mean, post.omega),
        post.sigma_mean,
        post.df * np.linalg.inv(post.scale),
    )


@pytest.mark.slow
def test_mh_draws_integrate_over_the_hyperparameter_draws():
    """With MH, ``(B, Σ)`` draws follow the equal-weight NIW mixture over the pooled ``h`` draws:
    ``E[vec B] = mean_i vec B̄_i`` and ``Cov(vec B) = mean_i E[Σ]_i ⊗ Ω̄_i + Cov_i(vec B̄_i)``."""
    case = Case(n=2, p=1, soc=1.0, sur=1.0)
    frame = var_frame(2, 80, seed=22)
    settings = rb.MHSettings(draws=5000, burn=500, chains=4)
    fit = case.model(hierarchy="mh", mh=settings).fit(to_data(frame), seed=4)
    assert fit.mh.names == ("lambda", "soc", "sur")
    pooled = fit.mh.chains.reshape(-1, len(fit.mh.names))
    S = pooled.shape[0]
    components = [
        replace(case, lam=lam, soc=soc, sur=sur).oracle(frame).posterior for lam, soc, sur in pooled
    ]
    vec_means = np.stack([c.mean.T.reshape(-1) for c in components])
    mean = vec_means.mean(axis=0)
    second = np.mean([np.kron(c.sigma_mean, c.omega) for c in components], axis=0)
    cov_vec = second + np.cov(vec_means, rowvar=False, bias=True)
    sigma_mean = np.mean([c.sigma_mean for c in components], axis=0)
    precision_mean = np.mean([c.df * np.linalg.inv(c.scale) for c in components], axis=0)
    coefs, chol = fit.draw_parameters(S, rng=5)
    _check_draw_moments(
        coefs, chol, mean.reshape(case.n, -1).T, cov_vec, sigma_mean, precision_mean
    )
    # power: draws from the posterior at the EB mode alone would fail the mean check
    h = fit.hyper
    at_mode = replace(case, lam=h["lambda"], soc=h["soc"], sur=h["sur"]).oracle(frame).posterior
    se = np.sqrt(np.diag(cov_vec) / S)
    assert np.max(np.abs(at_mode.mean.T.reshape(-1) - mean) / se) > 8


# --------------------------------------------------------------------------- one-step predictive


def _matrix_t_one_step(case: Case, frame: pd.DataFrame):
    """``y_{T+1} | Y ~ t_ν(B̄'x, S̄(s² + x'Ω̄x)/ν)`` with ``ν = ν̄ − n + 1``; returns the pieces."""
    post = case.oracle(frame).posterior
    values = frame.to_numpy()
    T = values.shape[0]
    det = [1.0, T + 1.0][: case.n_det]
    x = np.concatenate([det, *(values[T - lag] for lag in range(1, case.p + 1))])
    s = case.scales(pd.period_range(frame.index[-1] + 1, periods=1, freq="Q"))[0]
    return post.mean.T @ x, post.scale, float(x @ post.omega @ x), s, post.df - case.n + 1.0


@pytest.mark.parametrize(
    ("case", "end"),
    [
        pytest.param(Case(n=3, p=2, soc=0.7, sur=1.3), None, id="n3-p2-soc+sur"),
        pytest.param(Case(n=1, p=1), None, id="n1-p1"),
        pytest.param(Case(n=2, p=2, deterministic="ct"), None, id="trend"),
        pytest.param(Case(n=2, p=2, soc=1.0, covid=(5.0, 3.0, 2.0, 0.6)), "2021Q2", id="covid"),
        pytest.param(Case(n=2, p=1, covid=(5.0, 3.0, 2.0, 0.6)), "2020Q1", id="covid-s1"),
    ],
)
def test_one_step_predictive_density_is_matrix_t(case: Case, end: str | None):
    """Rao-Blackwellised ``log_pred_density`` at ``h = 1`` vs the closed-form matrix-t, within 4
    Monte Carlo standard errors (``hierarchy="fixed"``; under the COVID scaling the shock at
    ``T + 1`` has scale ``s_{T+1}``)."""
    frame = _covid_frame(case, end) if case.covid else var_frame(case.n, 60, seed=30 + case.n)
    fr = case.model().fit(to_data(frame)).forecast(1, draws=20_000, seed=17)
    loc, scale, q, s, nu = _matrix_t_one_step(case, frame)
    shape = scale * (s**2 + q) / nu
    sd = np.sqrt(np.diag(shape))
    exact = stats.multivariate_t(loc=loc, shape=shape, df=nu)
    offsets = ([0.0, 0.0, 0.0], [1.0, -0.5, 1.5], [2.5, 2.5, -2.0])
    for offset in offsets:
        y = loc + sd * np.array(offset[: case.n])
        lpd, mcse = fr.log_pred_density(y, 1, return_mcse=True)
        assert 0 < mcse < 0.05
        assert abs(lpd - exact.logpdf(y)) < 4 * mcse, (lpd, exact.logpdf(y), mcse)
    if case.n > 1:  # marginal of one variable: Student-t with the same ν
        y = loc + 2.0 * sd
        lpd, mcse = fr.log_pred_density(y[-1:], 1, [fr.variables[-1]], return_mcse=True)
        marginal = stats.t(df=nu, loc=loc[-1], scale=sd[-1]).logpdf(y[-1])
        assert abs(lpd - marginal) < 4 * mcse
    # power: the tolerance is far tighter than the gap to predictives that ignore the coefficient
    # uncertainty (x'Ω̄x) or, under the COVID scaling, the shock scale s_{T+1}
    lpd, mcse = fr.log_pred_density(loc, 1, return_mcse=True)
    wrong = [scale * s**2 / nu] + ([scale * (1.0 + q) / nu] if s != 1.0 else [])
    for alt in wrong:
        gap = abs(lpd - stats.multivariate_t(loc=loc, shape=alt, df=nu).logpdf(loc))
        assert gap > 10 * mcse, (gap, mcse)


# --------------------------------------------------------------------------- hierarchy target


def _gamma_by_mode_sd(mode: float, sd: float):
    """Gamma with the given mode ``(k − 1)θ`` and standard deviation ``√k θ``."""
    k = optimize.brentq(lambda k: (k - 1.0) ** 2 / k - (mode / sd) ** 2, 1.0 + 1e-12, 1e6)
    return stats.gamma(a=k, scale=sd / np.sqrt(k))


def _beta_by_mode_sd(mode: float, sd: float):
    """Beta(a, b) with the given mode ``(a − 1)/(a + b − 2)`` and standard deviation.

    Independent of the library's parametrisation: ``b`` follows from the mode as a function of
    ``a``, and the variance equation is solved for ``a`` by bracketing (no fsolve, whose
    convergence warnings differ across platforms).
    """

    def b_of(a: float) -> float:
        return (a - 1.0) * (1.0 - mode) / mode + 1.0

    def excess(a: float) -> float:
        b = b_of(a)
        return a * b / ((a + b) ** 2 * (a + b + 1.0)) - sd**2

    a = optimize.brentq(excess, 1.0 + 1e-12, 1e12, xtol=1e-14, rtol=1e-15)
    return stats.beta(a, b_of(a))


def test_log_posterior_adds_the_glp_hyperpriors():
    """EB/MH target: ``log p(Y | h) + log p(h)`` with Gamma(mode .2, sd .4) on λ, Gamma(mode 1,
    sd 1) on μ and δ, IG(0.02², 0.02²) on ψ/(d − n − 1), Pareto(1, 1) on the COVID scales and
    Beta(mode .8, sd .2) on ρ (all densities on the natural scale)."""
    case = Case(n=2, p=2, soc=1.0, sur=1.0)
    index = pd.period_range("2006Q1", "2021Q4", freq="Q")
    frame = var_frame(2, len(index), seed=8, start="2006Q1")
    prior = rb.Minnesota(soc=1.0, sur=1.0, hyper=("lambda", "soc", "sur", "psi"), mean=1.0)
    model = rb.BVAR(mean=rb.Linear(prior=prior), lags=case.p, outliers="lenza-primiceri")
    fit = model.fit(to_data(frame))
    h = {"lambda": 0.4, "soc": 2.0, "sur": 0.5, "psi": [0.7, 0.2], "covid": [6.0, 3.0, 1.5, 0.7]}
    c = 1.0  # d − n − 1
    expected = (
        _gamma_by_mode_sd(0.2, 0.4).logpdf(0.4)
        + _gamma_by_mode_sd(1.0, 1.0).logpdf(2.0)
        + _gamma_by_mode_sd(1.0, 1.0).logpdf(0.5)
        + sum(
            stats.invgamma(a=0.02**2, scale=0.02**2).logpdf(v / c) - np.log(c) for v in (0.7, 0.2)
        )
        + sum(stats.pareto(b=1.0).logpdf(s) for s in (6.0, 3.0, 1.5))
        + _beta_by_mode_sd(0.8, 0.2).logpdf(0.7)
    )
    assert fit.log_posterior(h) - fit.log_ml(h) == pytest.approx(expected, rel=1e-10)
    # and log_ml(h) is the heteroskedastic conjugate ML at those values
    oracle = replace(case, lam=0.4, soc=2.0, sur=0.5, alpha=2.0, psi=(0.7, 0.2))
    oracle = replace(oracle, covid=(6.0, 3.0, 1.5, 0.7)).oracle(frame)
    B, sigma = oracle.posterior.mean, oracle.posterior.sigma_mean
    assert fit.log_ml(h) == pytest.approx(oracle.chib_log_ml(B, sigma), rel=1e-12)


def test_eb_mode_is_a_stationary_point_of_the_log_posterior():
    """At the reported EB mode the Newton step on log hyperparameters is tiny in posterior-s.d.
    units: ``|g_i| / √H_ii < 0.01`` for every interior coordinate."""
    case = Case(n=3, p=2, soc=1.0, sur=1.0)
    frame = var_frame(3, 80, seed=12)
    fit = case.model(hierarchy="eb").fit(to_data(frame))
    assert fit.optimization["success"]
    z0 = np.log([fit.hyper[name] for name in ("lambda", "soc", "sur")])

    def target(z):
        return fit.log_posterior(dict(zip(("lambda", "soc", "sur"), np.exp(z), strict=True)))

    step = 1e-4
    for i in range(3):
        e = np.zeros(3)
        e[i] = step
        up, mid, down = target(z0 + e), target(z0), target(z0 - e)
        grad = (up - down) / (2 * step)
        curvature = -(up - 2 * mid + down) / step**2
        assert curvature > 0
        assert abs(grad) / np.sqrt(curvature) < 0.01, (i, grad, curvature)
