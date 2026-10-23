"""The path-integrated log predictive density (``lpd_method="integrated"``; DESIGN §7.3, D-34).

On synthetic ``ForecastResult``\\ s whose paths come from a VAR(2) with a constant and a trend
(no JAX): with one parameter draw and a constant ``Σ`` the estimator is the exact predictive
density of ``y_{T+h}`` at ``h = 2..8`` (1e-10), for the joint vector, leading blocks (in any
order), non-leading subsets and a 6-s.d. tail outcome, with a zero MCSE; with heterogeneous draws
and time-varying shock factors it is the mixture of the draws' exact densities. The oracle is
written out independently, in companion form: ``z_k = F z_{k−1} + E c_k`` and ``P_k = F P_{k−1}Fᵀ +
E Σ_k Eᵀ`` from the history, with ``scipy.stats.multivariate_normal``. Also: ``h = 1`` is the
conditioned estimator bitwise, the errors (no ``lag_coefs``, malformed dims, ``rb_transitory``, an
unknown method) and unit equivariance with variables at 1e±6 scales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logsumexp

from rai_bvar.data.design import deterministic_terms, regressors_from_history
from rai_bvar.forecast import ForecastResult, check_structure
from rai_bvar.forecast.result import LAG_DIMS
from rai_bvar.forecast.simulate import simulate_linear_var

N, P, H = 3, 2, 8
DET, NDET = "ct", 2  # constant and trend: the deterministic terms continue at t_last + h
T_LAST = 40.0
VARIABLES = ["a", "b", "c"]
SUBSETS = [None, ["a", "b"], ["b", "a"], ["b"], ["c"], ["c", "b"], ["c", "a"]]


def _coefs(rng: np.random.Generator, spread: float = 0.0) -> np.ndarray:
    """A stable VAR(2) ``B`` (k, n), rows ``(const, trend, y_{−1}, y_{−2})``; ``spread`` perturbs
    a fixed base system (heterogeneous draws)."""
    base = np.random.default_rng(99)
    A1 = 0.5 * np.eye(N) + base.normal(0.0, 0.15, (N, N))
    A2 = -0.2 * np.eye(N) + base.normal(0.0, 0.1, (N, N))
    det = np.array([[0.3, -0.2, 0.5], [0.01, 0.02, -0.01]])
    B = np.vstack([det, A1.T, A2.T])
    B = B + spread * rng.normal(0.0, 1.0, B.shape)
    F = _companion(B)
    assert np.max(np.abs(np.linalg.eigvals(F))) < 0.95
    return B


def _lag_matrices(B: np.ndarray) -> np.ndarray:
    """``A_l[i, j] = B[n_det + (l−1)·n + j, i]``, written out element by element."""
    A = np.empty((P, N, N))
    for lag in range(P):
        for i in range(N):
            for j in range(N):
                A[lag, i, j] = B[NDET + lag * N + j, i]
    return A


def _companion(B: np.ndarray) -> np.ndarray:
    F = np.zeros((N * P, N * P))
    F[:N] = np.hstack([B[NDET + lag * N : NDET + (lag + 1) * N].T for lag in range(P)])
    F[N:, :-N] = np.eye(N * (P - 1))
    return F


def _exact(B: np.ndarray, chol: np.ndarray, hist: np.ndarray, h: int):
    """Mean and covariance of ``y_{T+h}`` given ``B``, the shock factors ``chol (H, n, n)`` and
    the history ``(p, n)`` (most recent last): companion form, deterministic terms included."""
    F = _companion(B)
    E = np.zeros((N * P, N))
    E[:N] = np.eye(N)
    z = hist[::-1].reshape(-1)  # (y_T, y_{T−1})
    cov = np.zeros((N * P, N * P))
    for k in range(1, h + 1):
        d = deterministic_terms(np.asarray(T_LAST + k), DET)
        z = F @ z + E @ (B[:NDET].T @ d)
        cov = F @ cov @ F.T + E @ chol[k - 1] @ chol[k - 1].T @ E.T
    return z[:N], cov[:N, :N]


def _simulate(B: np.ndarray, chol: np.ndarray, hist: np.ndarray, eps: np.ndarray):
    """Paths and conditional moments of draws ``B (S, k, n)`` with time-varying shock factors
    ``chol (S, H, n, n)``: ``y = B′x + L_h ε_h``, ``x`` from the simulated path."""
    S = B.shape[0]
    window = np.broadcast_to(hist, (S, P, N)).copy()
    paths, means = np.empty((S, H, N)), np.empty((S, H, N))
    for h in range(H):
        x = regressors_from_history(window, T_LAST + h + 1, DET)
        means[:, h] = np.einsum("sk,skn->sn", x, B)
        paths[:, h] = means[:, h] + np.einsum("sij,sj->si", chol[:, h], eps[:, h])
        window = np.concatenate([window[:, 1:], paths[:, h, None]], axis=1)
    return paths, means


def _result(paths, means, chol, A, *, lag_dims=LAG_DIMS, attrs=None) -> ForecastResult:
    extra = None if A is None else {"lag_coefs": (lag_dims, A)}
    return ForecastResult.from_arrays(
        paths=paths, cond_mean=means, cond_chol=chol, variables=VARIABLES,
        origin=pd.Period("2000Q1", "Q"), attrs=attrs, extra=extra,
    )  # fmt: skip


def _constant_problem(S: int = 200, seed: int = 3, B: np.ndarray | None = None):
    """One VAR(2) (``B``, default :func:`_coefs`) and one ``Σ = LLᵀ`` for every draw (paths from
    Phase 1's simulator)."""
    rng = np.random.default_rng(seed)
    B = _coefs(rng) if B is None else B
    L = np.linalg.cholesky(np.array([[1.0, 0.3, -0.2], [0.3, 0.8, 0.1], [-0.2, 0.1, 0.5]]))
    hist = rng.normal(0.0, 1.0, (P, N))
    paths, means, chol = simulate_linear_var(
        np.broadcast_to(B, (S, *B.shape)), np.broadcast_to(L, (S, N, N)), hist, H, DET,
        int(T_LAST), rng,
    )  # fmt: skip
    A = np.broadcast_to(_lag_matrices(B), (S, P, N, N))
    fr = _result(paths, means, np.ascontiguousarray(chol), A)
    return fr, B, np.broadcast_to(L, (H, N, N)), hist


def _mixture_problem(S: int = 30, seed: int = 11):
    """Heterogeneous draws ``B_s`` and SV-like factors ``L_{s,h} = A_s⁻¹ diag(exp(½ h_{s,h}))``."""
    rng = np.random.default_rng(seed)
    B = np.stack([_coefs(rng, spread=0.03) for _ in range(S)])
    a = rng.normal(0.0, 0.4, (S, 3))
    Ainv = np.zeros((S, N, N))
    for s in range(S):
        A = np.eye(N)
        A[np.tril_indices(N, -1)] = a[s]
        Ainv[s] = np.linalg.solve(A, np.eye(N))
    logv = rng.normal(-0.5, 0.3, (S, 1, N)) + np.cumsum(rng.normal(0.0, 0.25, (S, H, N)), axis=1)
    chol = Ainv[:, None] * np.exp(0.5 * logv)[:, :, None, :]  # A⁻¹ diag(ℓ_h)
    hist = rng.normal(0.0, 1.0, (P, N))
    paths, means = _simulate(B, chol, hist, rng.standard_normal((S, H, N)))
    A = np.stack([_lag_matrices(b) for b in B])
    return _result(paths, means, chol, A), B, chol, hist


def _indices(subset) -> np.ndarray:
    return np.arange(N) if subset is None else np.array([VARIABLES.index(v) for v in subset])


def _tail(mean: np.ndarray, cov: np.ndarray, sds: float) -> np.ndarray:
    """A point at Mahalanobis distance ``sds`` from ``mean``."""
    v = np.array([1.0, -1.0, 0.5])[: mean.size]
    return mean + sds * np.linalg.cholesky(cov) @ (v / np.linalg.norm(v))


def _logpdf(y, mean, cov, idx) -> float:
    return float(stats.multivariate_normal(mean[idx], cov[np.ix_(idx, idx)]).logpdf(y))


@pytest.mark.parametrize("h", range(2, H + 1))
@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "joint" if s is None else "-".join(s))
def test_integrated_lpd_is_exact_for_a_single_parameter_draw(subset, h):
    """All draws share ``B`` and ``Σ``: the estimator is the exact ``N(μ_h, P_h)`` density (the
    drawn paths integrate out), at the mean, 1.5 s.d. and a 6-s.d. tail outcome; MCSE ≈ 0."""
    fr, B, L, hist = _constant_problem()
    mean, cov = _exact(B, L, hist, h)
    idx = _indices(subset)
    sub_cov = cov[np.ix_(idx, idx)]
    for sds in (0.0, 1.5, 6.0):
        y = _tail(mean[idx], sub_cov, sds)
        lpd, mcse = fr.log_pred_density(y, h, subset, return_mcse=True, lpd_method="integrated")
        np.testing.assert_allclose(lpd, _logpdf(y, mean, cov, idx), rtol=1e-12, atol=1e-10)
        assert 0.0 <= mcse < 1e-9


def test_conditioned_estimator_misses_the_tail_that_the_integrated_one_gets():
    """The motivation (NOTES 8b): with a persistent VAR at ``h = 8`` and 6 s.d. the
    path-conditioned average is carried by the few draws whose paths wandered towards the
    outcome (−35.2 against the exact −22.3 here, with an MCSE of 1.0, as in NOTES 8b); the
    integrated one is exact."""
    persistent = np.vstack(
        [[[0.3, -0.2, 0.5], [0.01, 0.02, -0.01]], 0.9 * np.eye(N), np.zeros((N, N))]
    )
    fr, B, L, hist = _constant_problem(B=persistent)
    mean, cov = _exact(B, L, hist, H)
    y = _tail(mean, cov, 6.0)
    exact = _logpdf(y, mean, cov, np.arange(N))
    conditioned, mcse = fr.log_pred_density(y, H, return_mcse=True)
    integrated = fr.log_pred_density(y, H, lpd_method="integrated")
    np.testing.assert_allclose(integrated, exact, rtol=1e-12, atol=1e-10)
    assert conditioned < exact - 5.0 and mcse > 0.5


@pytest.mark.parametrize("h", [2, 3, 5, 8])
@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "joint" if s is None else "-".join(s))
def test_integrated_lpd_is_the_mixture_of_exact_densities(subset, h):
    """Heterogeneous ``B_s`` and time-varying ``L_{s,k}``: ``logsumexp_s log N(y; μ_{s,h}, P_{s,h})
    − log S`` with each draw's exact moments given its own shock-factor path (1e-10); the MCSE is
    the delta-method formula of those densities."""
    fr, B, chol, hist = _mixture_problem()
    idx = _indices(subset)
    S = B.shape[0]
    moments = [_exact(B[s], chol[s], hist, h) for s in range(S)]
    centre = np.mean([m[idx] for m, _ in moments], axis=0)
    spread = np.mean([c[np.ix_(idx, idx)] for _, c in moments], axis=0)
    for sds in (0.0, 2.0, 6.0):
        y = _tail(centre, spread, sds)
        logp = np.array([_logpdf(y, m, c, idx) for m, c in moments])
        lpd, mcse = fr.log_pred_density(y, h, subset, return_mcse=True, lpd_method="integrated")
        np.testing.assert_allclose(lpd, logsumexp(logp) - np.log(S), rtol=1e-12, atol=1e-10)
        w = np.exp(logp - logp.max())
        np.testing.assert_allclose(mcse, w.std(ddof=1) / (np.sqrt(S) * w.mean()), rtol=1e-8)


@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "joint" if s is None else "-".join(s))
def test_horizon_one_is_the_conditioned_estimator_bitwise(subset):
    """At ``h = 1`` there is no path to integrate: the conditioned code runs (bitwise), with or
    without ``lag_coefs``."""
    fr, *_ = _mixture_problem()
    plain = ForecastResult(fr.dataset.drop_vars("lag_coefs"))
    idx = _indices(subset)
    for y in (fr.stacked("cond_mean")[:, 0].mean(axis=0), fr.stacked("paths")[3, 0] + 2.0):
        want = fr.log_pred_density(y[idx], 1, subset, return_mcse=True)
        assert fr.log_pred_density(y[idx], 1, subset, return_mcse=True,
                                   lpd_method="integrated") == want  # fmt: skip
        assert plain.log_pred_density(y[idx], 1, subset, return_mcse=True,
                                      lpd_method="integrated") == want  # fmt: skip


def test_default_estimator_ignores_lag_coefs():
    fr, *_ = _mixture_problem()
    plain = ForecastResult(fr.dataset.drop_vars("lag_coefs"))
    y = fr.stacked("paths")[0, 4] + 1.0
    for h in (1, 3, 8):
        for subset in SUBSETS:
            idx = _indices(subset)
            assert fr.log_pred_density(y[idx], h, subset, return_mcse=True) == (
                plain.log_pred_density(y[idx], h, subset, return_mcse=True)
            )


def test_integrated_errors():
    """No ``lag_coefs`` (analytic, benchmark, pooled forecasts), malformed dims,
    ``rb_transitory`` at ``h > 1`` and an unknown method raise ``ValueError``."""
    fr, *_ = _mixture_problem(S=12)
    ds = fr.dataset
    y = fr.stacked("cond_mean")[:, 1].mean(axis=0)
    plain = ForecastResult(ds.drop_vars("lag_coefs"))
    with pytest.raises(ValueError, match="needs the 'lag_coefs' variable"):
        plain.log_pred_density(y, 2, lpd_method="integrated")
    with pytest.raises(ValueError, match="rb_transitory"):
        fr.log_pred_density(y, 2, rb_transitory=True, lpd_method="integrated")
    with pytest.raises(ValueError, match="lpd_method"):
        fr.log_pred_density(y, 2, lpd_method="exact")
    with pytest.raises(ValueError, match="lpd_method"):
        fr.log_pred_density(y, 1, lpd_method="Integrated")
    A = ds["lag_coefs"].values
    bad = {
        "variables swapped": (("chain", "draw", "lag", "variable_", "variable"), A),
        "no lag axis": (("chain", "draw", "variable", "variable_"), A[:, :, 0]),
        "no chain axis": (("draw", "lag", "variable", "variable_"), A[0]),
        "lag first": (("lag", "chain", "draw", "variable", "variable_"), np.moveaxis(A, 2, 0)),
    }
    for dims, values in bad.values():
        broken = ForecastResult(ds.drop_vars("lag_coefs").assign(lag_coefs=(dims, values)))
        check_structure(broken)  # extras never fail the contract checks
        with pytest.raises(ValueError, match="must have dims"):
            broken.log_pred_density(y, 2, lpd_method="integrated")
        assert np.isfinite(broken.log_pred_density(y, 2))  # the default never reads it


@pytest.mark.parametrize("lpd_method", ["conditioned", "integrated"])
def test_repeated_or_empty_variable_subsets_raise(lpd_method):
    """Review finding math-4: a repeated variable makes the subset covariance singular (the
    estimators returned NaN or finite nonsense, +30 nats), and an empty subset failed with an
    ``IndexError``; both are ``ValueError`` now, for either estimator and every horizon."""
    fr, *_ = _mixture_problem(S=12)
    y = fr.stacked("cond_mean")[:, 2].mean(axis=0)
    for subset in (["a", "a"], ["b", "b"], ["c", "a", "c"], ["a", "b", "a"]):
        for h in (1, 3):
            with pytest.raises(ValueError, match="non-empty and distinct"):
                fr.log_pred_density(y[_indices(subset)], h, subset, lpd_method=lpd_method)
    with pytest.raises(ValueError, match="non-empty and distinct"):
        fr.log_pred_density(np.array([]), 3, [], lpd_method=lpd_method)
    assert np.isfinite(fr.log_pred_density(y[[2, 0]], 3, ["c", "a"], lpd_method=lpd_method))


@pytest.mark.parametrize("scale", [(1e6, 1e-6, 1.0), (1e-6, 1.0, 1e6), (1e6, 1e6, 1e-6)], ids=str)
@pytest.mark.parametrize("problem", ["constant", "mixture"])
def test_integrated_lpd_is_unit_equivariant(problem, scale):
    """Variables in units ``c = 1e±6``: ``y → c ⊙ y``, ``L → diag(c) L``, ``A_l → diag(c) A_l
    diag(c)⁻¹`` gives ``log p̂ − Σ_{idx} log c_i`` (1e-10) against the unit-scale oracle, finite,
    with the MCSE unchanged; ``M_h M_hᵀ`` (condition number ~1e24 here) is never formed."""
    fr, B, chol, hist = _constant_problem(S=60) if problem == "constant" else _mixture_problem()
    c = np.array(scale)
    ds = fr.dataset
    scaled = _result(
        ds["paths"].values * c, ds["cond_mean"].values * c, ds["cond_chol"].values * c[:, None],
        ds["lag_coefs"].values * c[:, None] / c[None, :],
    )  # fmt: skip
    S = fr.n_samples
    chols = np.broadcast_to(chol, (S, H, N, N)) if problem == "constant" else chol
    Bs = np.broadcast_to(B, (S, *B.shape[-2:]))
    for h in (2, 5, 8):
        moments = [_exact(Bs[s], chols[s], hist, h) for s in range(S)]
        for subset in (None, ["a", "b"], ["b"], ["c", "a"]):
            idx = _indices(subset)
            centre = np.mean([m[idx] for m, _ in moments], axis=0)
            spread = np.mean([v[np.ix_(idx, idx)] for _, v in moments], axis=0)
            y = _tail(centre, spread, 4.0)
            logp = np.array([_logpdf(y, m, v, idx) for m, v in moments])
            oracle = logsumexp(logp) - np.log(S)
            base, se = fr.log_pred_density(y, h, subset, return_mcse=True, lpd_method="integrated")
            got, se_scaled = scaled.log_pred_density(
                y * c[idx], h, subset, return_mcse=True, lpd_method="integrated"
            )
            assert np.isfinite(got)
            np.testing.assert_allclose(base, oracle, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(got + np.log(c[idx]).sum(), oracle, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(se_scaled, se, rtol=1e-6, atol=1e-12)
