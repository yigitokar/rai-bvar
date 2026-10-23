"""Numerical regression tests for the Phase 1 engines (review rounds 1–2, SPEC §8, §7, §10.6).

* ``gram_cholesky(ref_norms=...)``: rank is judged against the column norms *before* a
  cancellation, so the benchmarks catch singular Σ from deterministic equations (a lagged
  identity or an exact AR function in ``OLSVAR``, a linear trend in ``RandomWalk``), while
  well-conditioned data in wildly different units still fit.
* Unit equivariance of the conjugate BVAR: under ``y_j → c_j y_j`` (with ``ψ_j → c_j² ψ_j``)
  ``log p(Y) + T Σ_j log c_j`` is unchanged, for ``c`` as extreme as ``1e±10``.
* ``ConjugateProblem._core`` (one QR of ``[X* | Y*]``) against the naive normal-equations
  posterior, including fewer data rows than ``k + n`` (``R_yy`` then has fewer rows than ``n``);
  ``_log_ml_small`` (determinant lemma) against the dense GLP formula.
* ``cond_chol`` is a zero-stride view over ``(S, n, n)`` without the COVID scaling and a
  materialised ``(S, H, n, n)`` array with it.
* Forecast memory is bounded by chunking the parameter draws (``_CHUNK_ELEMENTS``).
* ``ess_bulk`` is NaN when the split halves have ≤ 4 draws (4–9 draws per chain) and finite,
  data-dependent from 10 draws on.

The oracles use explicit inverses and determinants (fine in tests); only the code under test
uses rai_bvar's linear algebra.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import multigammaln

import rai_bvar as rb
import rai_bvar.engines.analytic as analytic
from rai_bvar.diagnostics import ess_bulk
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.linalg import gram_cholesky

PSI3 = (4e-5, 1e-5, 0.06)  # roughly the innovation variances of the conftest simulation
COLUMNS3 = ["gdp", "prices", "rate"]
TINY_MH = rb.MHSettings(draws=200, burn=200, chains=2)
MH_NOT_CONVERGED = "ignore:hyperparameter MH did not converge:UserWarning"


# --------------------------------------------------------------------------- data helpers


def _frame(values: np.ndarray, start: str = "1980Q1", columns=None) -> pd.DataFrame:
    index = pd.period_range(start, periods=len(values), freq="Q")
    columns = columns or [f"y{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=list(columns))


def _data(values: np.ndarray, start: str = "1980Q1", tcodes=1, columns=None) -> rb.VARData:
    return rb.VARData.from_frame(_frame(values, start, columns), tcodes)


def _var_values(n: int, T: int, seed: int) -> np.ndarray:
    """A stationary VAR(1) around non-zero means with unequal innovation scales (largest
    eigenvalue of ``A`` is ``0.6 + 0.3 (n − 1)/n < 0.9`` for every ``n``)."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(-2.0, 4.0, n)
    A = 0.6 * np.eye(n) + (0.3 / n) * (np.ones((n, n)) - np.eye(n))
    sd = np.linspace(0.5, 2.0, n)
    out, prev = np.empty((T, n)), mu.copy()
    for t in range(T):
        prev = mu + A @ (prev - mu) + sd * rng.standard_normal(n)
        out[t] = prev
    return out


def _bvar(lags: int = 2, outliers=None, deterministic: str = "c", **prior) -> rb.BVAR:
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(**prior)),
        lags=lags,
        deterministic=deterministic,
        outliers=outliers,
    )


def _fit_quietly(model: rb.BVAR, data: rb.VARData):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "hyperparameter MH did not converge", UserWarning)
        return model.fit(data)


# --------------------------------------------------------------------------- gram_cholesky


def _cancelled_residuals(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """OLS residuals of three columns on a design; column 1 lies exactly in the design's span,
    so its residual is pure rounding noise that looks well scaled relative to itself."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(80), rng.standard_normal((80, 2))])
    Y = rng.standard_normal((80, 3))
    Y[:, 1] = X @ np.array([1.0, 0.3, -2.0])
    resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    return resid, Y


def test_gram_cholesky_ref_norms_catch_a_cancelled_column():
    resid, Y = _cancelled_residuals()
    assert np.linalg.norm(resid[:, 1]) < 1e-12 * np.linalg.norm(Y[:, 1])  # rounding noise only
    gram_cholesky(resid)  # relative to its own (tiny) norm the noise column looks fine
    with pytest.raises(np.linalg.LinAlgError, match=r"column 1\b"):
        gram_cholesky(resid, ref_norms=np.linalg.norm(Y, axis=0))


def test_gram_cholesky_default_ref_norms_are_the_column_norms():
    A = np.random.default_rng(1).standard_normal((30, 4))
    np.testing.assert_array_equal(
        gram_cholesky(A), gram_cholesky(A, ref_norms=np.linalg.norm(A, axis=0))
    )


@pytest.mark.parametrize(
    "scale",
    [(1e10, 1e-10, 1.0, 3.0), (1e-10, 1e10, 1e6, 1e-6), (1e-100, 1.0, 1e100, 1.0)],
    ids=["1e+-10", "mixed", "1e+-100"],
)
def test_gram_cholesky_is_unit_free(scale):
    """Well-conditioned columns in wildly different units: no error, and ``L`` scales by row."""
    c = np.array(scale)
    A = np.random.default_rng(2).standard_normal((40, 4))
    L = gram_cholesky(A)
    Lc = gram_cholesky(A * c)
    np.testing.assert_allclose(Lc / c[:, None], L, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- benchmarks: singular Σ


def _base_values(T: int = 120, seed: int = 0) -> np.ndarray:
    return _var_values(3, T, seed)


def _lagged_identity(values: np.ndarray, lag: int) -> np.ndarray:
    z = values.copy()
    z[lag:, 1] = z[:-lag, 0]  # y2_t = y1_{t−lag}
    return z


def _exact_ar_function(values: np.ndarray, lag: int) -> np.ndarray:
    z = values.copy()
    for t in range(lag, len(z)):
        z[t, 1] = 0.3 * z[t - lag, 0] + 0.2 * z[t - 1, 2] + 1.0
    return z


@pytest.mark.parametrize("make", [_lagged_identity, _exact_ar_function], ids=["identity", "ar"])
@pytest.mark.parametrize("lags", [1, 2, 3])
@pytest.mark.parametrize("uncertainty", ["flat", "none"])
def test_olsvar_deterministic_equation_raises_naming_the_column(make, lags, uncertainty):
    """Variable 1's equation is exact in the regressors (its last lag of variable 0), so its
    residual column is rounding noise and Σ is singular; the old column-relative test let this
    through with forecast s.d. ~1e-14. The lag is chosen so the design itself stays full rank."""
    data = _data(make(_base_values(), lags))
    with pytest.raises(np.linalg.LinAlgError, match=r"singular at column 1\b"):
        rb.OLSVAR(lags=lags, uncertainty=uncertainty).fit(data)


def test_olsvar_collinear_design_is_reported_as_such():
    """A first-lag identity with p = 2 duplicates a regressor (y2_{t−1} = y1_{t−2}): the design,
    not Σ, is singular, and the error says so."""
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        rb.OLSVAR(lags=2).fit(_data(_lagged_identity(_base_values(), 1)))


@pytest.mark.parametrize("slope", [0.1, 3.7, 1e-10, 1.0])
@pytest.mark.parametrize("drift", [False, True])
def test_random_walk_on_a_linear_trend_raises_naming_the_column(slope, drift):
    """``Δy = slope`` up to rounding: demeaning cancels everything (0.1·t used to slip through
    because its rounding noise is not exactly zero)."""
    values = _base_values()
    values[:, 2] = slope * np.arange(len(values))
    with pytest.raises(np.linalg.LinAlgError, match=r"singular at column 2\b"):
        rb.RandomWalk(drift=drift).fit(_data(values))


@pytest.mark.parametrize(
    "model",
    [
        rb.OLSVAR(lags=2),
        rb.OLSVAR(lags=2, uncertainty="none"),
        rb.RandomWalk(),
        rb.RandomWalk(drift=True),
    ],
    ids=["ols-flat", "ols-none", "rw", "rw-drift"],
)
@pytest.mark.parametrize(
    "scale", [(1e10, 1e-10, 1.0), (1e-10, 1.0, 1e10), (1e6, 1e-6, 1e-10)], ids=str
)
def test_benchmarks_accept_badly_scaled_data(model, scale):
    """Distinct series in mixed units are not singular, and forecasts are unit-equivariant:
    with the same seed, paths and conditional factors scale by ``c_j`` row by row."""
    c = np.array(scale)
    values = _base_values()
    base = model.fit(_data(values)).forecast(3, draws=20, seed=4)
    scaled = model.fit(_data(values * c)).forecast(3, draws=20, seed=4)
    np.testing.assert_allclose(scaled.paths.values / c, base.paths.values, rtol=1e-10, atol=1e-12)
    chol = scaled.cond_chol.values / c[:, None]
    np.testing.assert_allclose(chol, base.cond_chol.values, rtol=1e-10, atol=1e-14)
    assert np.all(np.diagonal(base.cond_chol.values, axis1=-2, axis2=-1) > 1e-3)


# --------------------------------------------------------------------------- BVAR unit equivariance

UNIT_SCALES = {
    "1e6": np.full(3, 1e6),
    "1e-6": np.full(3, 1e-6),
    "1e10": np.full(3, 1e10),
    "1e-10": np.full(3, 1e-10),
    "mixed-a": np.array([1e6, 1e-10, 1e10]),
    "mixed-b": np.array([1e-6, 1e10, 1e-10]),
}
PRIOR_VARIANTS = {
    "minnesota": {},
    "soc+sur": {"soc": 0.7, "sur": 1.3},
    "soc+sur-all": {"soc": 0.7, "sur": 1.3, "soc_scope": "all"},
}
TCODES = {"levels": 1, "one-differenced": {"gdp": 2, "prices": 1, "rate": 1}}


@pytest.fixture(scope="module")
def unit_values(simulate_var) -> np.ndarray:
    return simulate_var(120, 5)


@pytest.mark.parametrize("tcodes", list(TCODES))
@pytest.mark.parametrize("psi", ["explicit", "ar"])
@pytest.mark.parametrize("variant", list(PRIOR_VARIANTS))
@pytest.mark.parametrize("scale", list(UNIT_SCALES))
def test_bvar_log_ml_is_unit_equivariant(unit_values, scale, variant, psi, tcodes):
    """``y_j → c_j y_j`` with ``ψ_j → c_j² ψ_j``: ``log p(cY) + T Σ log c_j = log p(Y)``."""
    c = UNIT_SCALES[scale]
    codes = TCODES[tcodes]

    def fit(factor: np.ndarray):
        given = tuple(np.array(PSI3) * factor**2) if psi == "explicit" else "ar"
        model = _bvar(hierarchy="fixed", lambda_=0.3, psi=given, **PRIOR_VARIANTS[variant])
        return model.fit(_data(unit_values * factor, tcodes=codes, columns=COLUMNS3))

    base, scaled = fit(np.ones(3)), fit(c)
    T = base.problem.T
    assert scaled.problem.T == T
    np.testing.assert_allclose(scaled.psi / c**2, base.psi, rtol=1e-12)
    expected = base.log_ml()
    assert np.isfinite(expected)
    assert scaled.log_ml() + T * np.log(c).sum() == pytest.approx(expected, rel=1e-12, abs=1e-8)
    # the same holds away from the fitted hyperparameters
    other = {"lambda": 0.9, "alpha": 1.3, "psi": np.array([1e-4, 3e-5, 0.2])}
    moved = scaled.log_ml({**other, "psi": other["psi"] * c**2}) + T * np.log(c).sum()
    assert moved == pytest.approx(base.log_ml(other), rel=1e-12, abs=1e-8)


@pytest.mark.parametrize("scale", ["mixed-a", "1e-10"])
def test_bvar_posterior_and_forecasts_are_unit_equivariant(unit_values, scale):
    """``B̄ → D⁻¹ B̄ C`` (``D`` scales the regressors), ``S̄ → C S̄ C``; with a common seed the
    forecast paths and conditional factors scale by ``c_j``."""
    c = UNIT_SCALES[scale]
    model = _bvar(hierarchy="fixed", soc=1.0, sur=1.0)
    base = model.fit(_data(unit_values, columns=COLUMNS3))
    scaled = model.fit(_data(unit_values * c, columns=COLUMNS3))
    d = np.concatenate([[1.0], c, c])  # constant, lag 1, lag 2
    pb, ps = base.posterior(), scaled.posterior()
    np.testing.assert_allclose(d[:, None] * ps.mean / c, pb.mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ps.scale / np.outer(c, c), pb.scale, rtol=1e-10)
    fb = base.forecast(4, draws=200, seed=9)
    fs = scaled.forecast(4, draws=200, seed=9)
    np.testing.assert_allclose(fs.paths.values / c, fb.paths.values, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fs.cond_chol.values / c[:, None], fb.cond_chol.values, rtol=1e-10)
    y = unit_values[-1]
    lpd = fs.log_pred_density(y * c, 1) + np.log(c).sum()
    assert lpd == pytest.approx(fb.log_pred_density(y, 1), abs=1e-9)


@pytest.mark.parametrize("scale", ["mixed-a", "1e-6", "1e10"])
def test_eb_mode_is_unit_invariant(unit_values, scale):
    """The log ML moves by a constant, so the EB mode of (λ, μ, δ) does not move."""
    c = UNIT_SCALES[scale]
    model = _bvar(soc=1.0, sur=1.0)
    base = model.fit(_data(unit_values, columns=COLUMNS3))
    scaled = model.fit(_data(unit_values * c, columns=COLUMNS3))
    assert base.optimization["success"] and scaled.optimization["success"]
    for name in ("lambda", "soc", "sur"):
        assert scaled.hyper[name] == pytest.approx(base.hyper[name], rel=1e-6), name
    shift = base.problem.T * np.log(c).sum()
    assert scaled.optimization["log_posterior"] + shift == pytest.approx(
        base.optimization["log_posterior"], abs=1e-7
    )


# --------------------------------------------------------------------------- _core and _log_ml_small

CORE_CONFIGS = {
    "n3-p2-c": dict(n=3, lags=2, deterministic="c", prior={"lambda_": 0.4}),
    "n2-p3-ct": dict(n=2, lags=3, deterministic="ct", prior={"var_const": 100.0, "mean": 0.5}),
    "n1-p1-n": dict(n=1, lags=1, deterministic="n", prior={"lambda_": 0.8, "alpha": 1.0}),
    "n4-p1-c": dict(n=4, lags=1, deterministic="c", prior={"lambda_": 0.25, "var_const": 1e3}),
}


def _problem(config: str) -> ConjugateProblem:
    cfg = CORE_CONFIGS[config]
    n = cfg["n"]
    data = _data(_var_values(n, 60, seed=n + cfg["lags"]))
    psi = tuple(np.linspace(0.3, 2.5, n))
    prior = rb.Minnesota(hierarchy="fixed", psi=psi, **cfg["prior"])
    return ConjugateProblem(data, prior, cfg["lags"], cfg["deterministic"], None)


def _dense_niw(prob: ConjugateProblem, Y: np.ndarray, X: np.ndarray, omega, psi):
    """GLP log ML and posterior from the normal equations: ``P = X'X + Ω⁻¹``,
    ``B̄ = P⁻¹(X'Y + Ω⁻¹b)``, ``S̄ = Ψ + Ê'Ê + (B̄ − b)'Ω⁻¹(B̄ − b)`` and

    ``log p = −(nT/2) log π + log Γ_n((T+d)/2) − log Γ_n(d/2) − (n/2) log|Ω| − (n/2) log|P|
    + (d/2) log|Ψ| − ((T+d)/2) log|S̄|``."""
    rows, n, d, b = Y.shape[0], prob.n, prob.d, prob.b
    P0 = np.diag(1.0 / omega)
    P = X.T @ X + P0
    B = np.linalg.solve(P, X.T @ Y + P0 @ b)
    E = Y - X @ B
    S = np.diag(psi) + E.T @ E + (B - b).T @ P0 @ (B - b)
    log_ml = (
        -0.5 * n * rows * np.log(np.pi)
        + multigammaln(0.5 * (rows + d), n)
        - multigammaln(0.5 * d, n)
        - 0.5 * n * np.log(omega).sum()
        - 0.5 * n * np.linalg.slogdet(P)[1]
        + 0.5 * d * np.log(psi).sum()
        - 0.5 * (rows + d) * np.linalg.slogdet(S)[1]
    )
    return float(log_ml), B, np.linalg.inv(P), S


def _core_cases():
    for config, cfg in CORE_CONFIGS.items():
        n = cfg["n"]
        k = {"n": 0, "c": 1, "ct": 2}[cfg["deterministic"]] + n * cfg["lags"]
        for rows in sorted({*range(k + n), k + n, 3 * (k + n)}):
            yield pytest.param(config, rows, id=f"{config}-rows{rows}")


@pytest.mark.parametrize(("config", "rows"), list(_core_cases()))
def test_core_matches_the_normal_equations(config, rows):
    """Fewer rows than ``k + n`` included: the QR of ``[X* | Y*]`` then has fewer than ``k + n``
    rows, ``R_yy`` fewer than ``n``, and ``S̄`` must still be ``Ψ + R_yy'R_yy``."""
    prob = _problem(config)
    h = prob.default
    omega, psi = prob.omega(h), h["psi"]
    Y, X = prob.Y[:rows], prob.X[:rows]
    assert Y.shape == (rows, prob.n)
    log_ml, post = prob._core(Y, X, omega, psi)
    expected, B, omega_bar, S = _dense_niw(prob, Y, X, omega, psi)
    assert log_ml == pytest.approx(expected, rel=1e-10, abs=1e-8)
    if rows == 0:  # no data: the marginal likelihood of nothing is 1 and the posterior is the prior
        assert log_ml == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(post.mean, prob.b, atol=1e-12)
    assert post.df == prob.d + rows
    np.testing.assert_allclose(post.mean, B, rtol=1e-8, atol=1e-9 * np.abs(B).max())
    np.testing.assert_allclose(post.scale, S, rtol=1e-10, atol=1e-12 * np.abs(S).max())
    np.testing.assert_allclose(post.omega(), omega_bar, rtol=1e-8, atol=1e-10 * omega.max())
    np.testing.assert_array_equal(np.tril(post.row_upper, k=-1), 0.0)
    assert np.all(np.diag(post.row_upper) > 0)
    np.testing.assert_array_equal(np.triu(post.scale_chol, k=1), 0.0)
    assert np.all(np.diag(post.scale_chol) > 0)


def _small_cases():
    for config, cfg in CORE_CONFIGS.items():
        for rows in range(1, cfg["n"] + 3):
            yield pytest.param(config, "design", rows, id=f"{config}-design{rows}")
        yield pytest.param(config, "soc+sur", None, id=f"{config}-soc+sur")
        yield pytest.param(config, "sur", None, id=f"{config}-sur")


@pytest.mark.parametrize(("config", "kind", "rows"), list(_small_cases()))
def test_log_ml_small_matches_the_dense_formula(config, kind, rows):
    """``|Ω||Ω⁻¹ + X'X| = |I + XΩX'|``: the few-row ML (SOC/SUR dummies) equals the dense
    GLP formula and the QR-based ``_core``."""
    prob = _problem(config)
    if kind == "design":
        h = prob.default
        Y, X = prob.Y[:rows], prob.X[:rows]
    else:
        h = prob.complete({"soc": 0.7 if kind == "soc+sur" else None, "sur": 1.3})
        Y, X = prob.dummies(h)
        assert Y.shape[0] == (prob.soc_rows.size if kind == "soc+sur" else 0) + 1
    omega, psi = prob.omega(h), h["psi"]
    value = prob._log_ml_small(Y, X, omega, psi)
    expected = _dense_niw(prob, Y, X, omega, psi)[0]
    assert value == pytest.approx(expected, rel=1e-9, abs=1e-8)
    assert value == pytest.approx(prob._core(Y, X, omega, psi)[0], rel=1e-9, abs=1e-8)


# --------------------------------------------------------------------------- cond_chol layout


@pytest.fixture(scope="module")
def plain_data(simulate_var) -> rb.VARData:
    """1985Q1–2014Q4: clear of 2020, so no outlier treatment is needed."""
    return _data(simulate_var(120, 7), start="1985Q1", columns=COLUMNS3)


@pytest.fixture(scope="module")
def covid_data(simulate_var) -> rb.VARData:
    """1990Q2–2020Q1: the sample ends at t* = 2020Q1."""
    return _data(simulate_var(120, 8), start="1990Q2", columns=COLUMNS3)


@pytest.fixture(scope="module")
def plain_fits(plain_data) -> dict:
    return {
        "fixed": _fit_quietly(_bvar(hierarchy="fixed", psi=PSI3), plain_data),
        "eb": _fit_quietly(_bvar(soc=1.0, sur=1.0), plain_data),
        "mh": _fit_quietly(_bvar(hierarchy="mh", soc=1.0, sur=1.0, mh=TINY_MH), plain_data),
    }


COVID_FIXED = (5.0, 3.0, 2.0, 0.6)


@pytest.fixture(scope="module")
def covid_fits(covid_data) -> dict:
    lp = rb.LenzaPrimiceri(fixed=COVID_FIXED)
    return {
        "fixed": _fit_quietly(_bvar(hierarchy="fixed", psi=PSI3, outliers=lp), covid_data),
        "eb": _fit_quietly(_bvar(psi=PSI3, outliers=lp), covid_data),
        "mh": _fit_quietly(_bvar(hierarchy="mh", psi=PSI3, mh=TINY_MH, outliers=lp), covid_data),
    }


def _chunk(monkeypatch, fit, draws_per_chunk: int | None) -> int:
    """Patch ``_CHUNK_ELEMENTS`` so that each chunk holds ``draws_per_chunk`` coefficient draws."""
    if draws_per_chunk is None:
        return 10**9
    monkeypatch.setattr(
        analytic, "_CHUNK_ELEMENTS", draws_per_chunk * fit.problem.k * fit.problem.n
    )
    return draws_per_chunk


@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("chunk", [None, 7], ids=["one-chunk", "chunks-of-7"])
@pytest.mark.parametrize("hierarchy", ["fixed", "eb", "mh"])
def test_cond_chol_is_a_zero_stride_view_without_covid_scaling(
    plain_fits, monkeypatch, hierarchy, chunk
):
    fit = plain_fits[hierarchy]
    _chunk(monkeypatch, fit, chunk)
    draws, horizon, n = 50, 4, fit.problem.n
    fr = fit.forecast(horizon, draws=draws, seed=3)
    for array in (fr.cond_chol.values, fr.dataset["cond_chol"].data):
        assert array.shape == (1, draws, horizon, n, n)
        assert array.strides[2] == 0  # (chain, draw, horizon, variable, variable_)
    values = fr.cond_chol.values
    assert not values.flags.writeable  # a broadcast view: writes would hit every horizon
    for h in range(1, horizon):
        np.testing.assert_array_equal(values[:, :, h], values[:, :, 0])
    np.testing.assert_array_equal(np.triu(values[0, :, 0], k=1), 0.0)
    assert np.all(np.diagonal(values[0, :, 0], axis1=-2, axis2=-1) > 0)
    if chunk is None:  # one chunk: the factors are exactly the parameter draws of the same seed
        _, chol = fit.draw_parameters(draws, rng=3)
        np.testing.assert_array_equal(values[0, :, 0], chol)


@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("chunk", [None, 7], ids=["one-chunk", "chunks-of-7"])
@pytest.mark.parametrize("hierarchy", ["fixed", "eb", "mh"])
def test_cond_chol_is_materialised_with_covid_scaling(covid_fits, monkeypatch, hierarchy, chunk):
    """``cond_chol[s, h] = s_{T+h} L_s`` with ``s_{T+1..4} = s1, s2, 1 + (s2 − 1)ρ, …``."""
    fit = covid_fits[hierarchy]
    _chunk(monkeypatch, fit, chunk)
    draws, horizon = 50, 4
    fr = fit.forecast(horizon, draws=draws, seed=3)
    values = fr.cond_chol.values
    assert values.shape == (1, draws, horizon, 3, 3)
    assert values.strides[2] != 0
    _, s1, s2, rho = COVID_FIXED  # the sample ends at t*, so s0 scales no forecast period
    scales = np.array([s1, s2, 1 + (s2 - 1) * rho, 1 + (s2 - 1) * rho**2])
    base = values[0, :, 0] / scales[0]
    for h in range(horizon):
        np.testing.assert_allclose(values[0, :, h], scales[h] * base, rtol=1e-14)


# --------------------------------------------------------------------------- chunked forecasts


@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_forecast_chunks_bound_the_coefficient_block(plain_fits, monkeypatch, hierarchy):
    """Every simulation call and every matrix-normal draw handles at most one chunk of draws,
    and the result is reassembled from exactly those chunks."""
    fit = plain_fits[hierarchy]
    chunk = _chunk(monkeypatch, fit, 4)
    calls, blocks = [], []
    simulate, sample_mn = analytic.simulate_linear_var, analytic.sample_matrix_normal

    def spy_simulate(coefs, chol, *args, **kwargs):
        out = simulate(coefs, chol, *args, **kwargs)
        calls.append((coefs.shape[0], chol.copy(), out[0].copy(), out[1].copy()))
        return out

    def spy_sample(mean, row_upper, col_chol, rng):
        out = sample_mn(mean, row_upper, col_chol, rng)
        blocks.append(out.shape[0])
        return out

    monkeypatch.setattr(analytic, "simulate_linear_var", spy_simulate)
    monkeypatch.setattr(analytic, "sample_matrix_normal", spy_sample)
    draws = 400 if hierarchy == "mh" else 50  # MH: every pooled draw, so groups repeat
    fr = fit.forecast(3, draws=draws, seed=5)
    # draws come in fixed random-stream blocks, so one call holds at most max(chunk, block)
    bound = max(chunk, analytic._DRAW_BLOCK)
    sizes = [size for size, *_ in calls]
    assert sum(sizes) == draws and max(sizes) <= bound and len(sizes) >= draws // bound
    assert sum(blocks) == draws and max(blocks) <= bound
    np.testing.assert_array_equal(fr.paths.values[0], np.concatenate([c[2] for c in calls]))
    np.testing.assert_array_equal(fr.cond_mean.values[0], np.concatenate([c[3] for c in calls]))
    np.testing.assert_array_equal(
        fr.cond_chol.values[0, :, 0], np.concatenate([c[1] for c in calls])
    )


@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_chunked_forecast_is_reproducible(plain_fits, monkeypatch, hierarchy):
    fit = plain_fits[hierarchy]
    _chunk(monkeypatch, fit, 3)
    a, b = fit.forecast(3, draws=60, seed=11), fit.forecast(3, draws=60, seed=11)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(a.dataset[name].values, b.dataset[name].values)


@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_chunked_forecast_is_bit_identical_to_a_single_chunk(plain_fits, monkeypatch, hierarchy):
    """Chunking bounds memory only: the draws must not depend on ``_CHUNK_ELEMENTS``."""
    fit = plain_fits[hierarchy]
    single = fit.forecast(3, draws=60, seed=11)
    _chunk(monkeypatch, fit, 7)
    chunked = fit.forecast(3, draws=60, seed=11)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(
            chunked.dataset[name].values, single.dataset[name].values, err_msg=name
        )


def test_forecast_memory_is_bounded_by_the_chunk(monkeypatch):
    """Transient memory of ``forecast`` (traced peak minus the retained result) stays a small
    multiple of one chunk of coefficient draws, far below the full ``(S, k, n)`` block; the
    unchunked control shows that the measurement sees that block."""
    n, lags, draws = 8, 6, 3000
    fit = _bvar(lags=lags, hierarchy="fixed", psi=tuple(np.linspace(0.3, 2.5, n))).fit(
        _data(_var_values(n, 160, seed=3), start="1970Q1")
    )
    full = draws * fit.problem.k * n * 8  # bytes of all coefficient draws at once

    def transient() -> int:
        tracemalloc.start()
        try:
            fr = fit.forecast(1, draws=draws, seed=1)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        del fr
        return peak - current

    unchunked = transient()
    chunk = _chunk(monkeypatch, fit, 50)
    chunked = transient()
    block = chunk * fit.problem.k * n * 8
    assert unchunked > full, (unchunked, full)
    assert chunked < 10 * block, (chunked, block)
    assert chunked < full / 4, (chunked, full)


@pytest.mark.slow
@pytest.mark.filterwarnings(MH_NOT_CONVERGED)
@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_chunked_forecast_has_the_single_chunk_distribution(plain_fits, monkeypatch, hierarchy):
    """Whatever the stream order, chunks of 13 draws give the same predictive distribution:
    means and standard deviations of every (horizon, variable) agree within 5 Monte Carlo s.e."""
    fit = plain_fits[hierarchy]
    draws = 20_000
    single = fit.forecast(4, draws=draws, seed=21).paths.values[0]
    _chunk(monkeypatch, fit, 13)
    chunked = fit.forecast(4, draws=draws, seed=22).paths.values[0]
    sd = single.std(axis=0)
    assert np.all(np.abs(chunked.mean(axis=0) - single.mean(axis=0)) < 5 * sd * np.sqrt(2 / draws))
    kurt = ((single - single.mean(axis=0)) ** 4).mean(axis=0) / sd**4
    se_sd = sd * np.sqrt((kurt - 1) / (4 * draws)) * np.sqrt(2)
    assert np.all(np.abs(chunked.std(axis=0) - sd) < 5 * se_sd)


@pytest.mark.slow
def test_forecast_memory_is_bounded_at_a_larger_size(monkeypatch):
    """n = 20, p = 4, S = 4000: with the shipped ``_CHUNK_ELEMENTS`` scaled down to 100 draws per
    chunk the transient stays below a quarter of the full coefficient block."""
    n, lags, draws = 20, 4, 4000
    fit = _bvar(lags=lags, hierarchy="fixed", psi=tuple(np.linspace(0.3, 2.5, n))).fit(
        _data(_var_values(n, 200, seed=4), start="1960Q1")
    )
    full = draws * fit.problem.k * n * 8
    _chunk(monkeypatch, fit, 100)
    tracemalloc.start()
    try:
        fr = fit.forecast(2, draws=draws, seed=1)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fr.cond_chol.values.strides[2] == 0
    assert peak - current < full / 4, (peak - current, full)


# --------------------------------------------------------------------------- ess_bulk on short chains


def _ar1_chains(rng: np.random.Generator, rho: float, chains: int, draws: int) -> np.ndarray:
    x = np.empty((chains, draws))
    x[:, 0] = rng.standard_normal(chains) / np.sqrt(1.0 - rho**2)
    for t in range(1, draws):
        x[:, t] = rho * x[:, t - 1] + rng.standard_normal(chains)
    return x


@pytest.mark.parametrize("rho", [0.0, 0.99, -0.9])
@pytest.mark.parametrize("chains", [1, 4, 64])
@pytest.mark.parametrize("draws", [4, 5, 6, 7, 8, 9])
def test_ess_bulk_is_nan_for_split_halves_of_at_most_four_draws(draws, chains, rho):
    """At 8–9 draws (halves of 4) ESS used to return the data-independent cap ``N log10 N``,
    which with 64 chains × 8 draws (1387) cleared the ``ess_min = 400`` gate."""
    x = _ar1_chains(np.random.default_rng(draws + 10 * chains), rho, chains, draws)
    value = ess_bulk(x)
    assert np.isnan(value)
    assert not value > 400.0


@pytest.mark.parametrize("chains", [1, 4, 64])
@pytest.mark.parametrize("draws", [*range(10, 21), 31, 50])
def test_ess_bulk_is_finite_from_ten_draws(draws, chains):
    size = draws * chains
    for rho in (0.0, 0.9, -0.5):
        value = ess_bulk(_ar1_chains(np.random.default_rng(draws + chains), rho, chains, draws))
        assert np.isfinite(value) and 0.0 < value <= size * np.log10(size) * (1 + 1e-12)


@pytest.mark.parametrize("chains", [4, 64])
def test_ess_bulk_at_ten_draws_depends_on_the_data(chains):
    """No placeholder at 10 draws: near-unit-root chains get far fewer effective draws than iid
    chains, and fewer than the number of draws."""
    rng = np.random.default_rng(1)
    iid = ess_bulk(_ar1_chains(rng, 0.0, chains, 10))
    sticky = ess_bulk(_ar1_chains(rng, 0.99, chains, 10))
    size = 10 * chains
    assert sticky < size < size * np.log10(size)
    assert iid > 2 * sticky


@pytest.mark.slow
def test_ess_bulk_short_chain_threshold_over_many_seeds():
    """NaN exactly when ``draws < 10`` (split halves ≤ 4), across chain counts, dependence and
    seeds; never a RuntimeWarning (an error in this suite)."""
    for chains in (1, 2, 3, 4, 64):
        for draws in range(4, 31):
            for rho in (0.0, 0.9, 0.99, -0.5):
                for seed in range(20):
                    x = _ar1_chains(np.random.default_rng(seed), rho, chains, draws)
                    value = ess_bulk(x)
                    assert np.isnan(value) == (draws < 10), (chains, draws, rho, seed, value)
