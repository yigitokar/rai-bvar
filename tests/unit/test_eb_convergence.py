"""Convergence of the empirical-Bayes hyperparameter optimiser (SPEC §4.6, §10.6).

``hierarchy="eb"`` maximises ``log p(Y | h) + log p(h)`` with L-BFGS-B (central-difference
gradients) on unconstrained scales and accepts the result when the predicted Newton gain
``½ Σ g_i² / H_ii`` over the coordinates not pinned at a bound is below a small tolerance in
log-posterior units (``_stationarity``). One restart runs only when that test fails. These tests
guard against both kinds of error:

* false alarms: realistic sizes (n = 20 here, n = 50 / 100 and real FRED-QD in the slow and
  network suites), monthly Lenza-Primiceri samples ending in 2024, warm starts at the mode and
  modes on a bound (α at its default cap 3, the data-dependent ψ box, λ at a lower bound) must
  all be reported converged, without a warning and without a restart;
* missed failures: an optimisation stopped after one iteration, or one that never sees a finite
  log posterior, must warn and set ``fit.converged`` and the forecast attr ``converged`` to 0;

plus the bookkeeping in ``fit.optimization`` and the accuracy of the reported mode against a
dense grid and a Nelder-Mead polish.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

import rai_bvar as rb
from rai_bvar.engines import analytic
from rai_bvar.engines.analytic import ConjugateProblem, _stationarity

FIXTURES = Path(__file__).resolve().parents[1] / "parity" / "fixtures"
TOL = analytic._STATIONARITY_TOL  # accepted predicted gain, log-posterior units
NOT_CONVERGED = "hyperparameter optimisation did not converge"
OPTIMIZATION_KEYS = {
    "success",
    "message",
    "evaluations",
    "log_posterior",
    "projected_gradient",
    "predicted_gain",
}


# --------------------------------------------------------------------------- data


def _load(name: str) -> rb.VARData:
    frame = pd.read_csv(FIXTURES / name, index_col="date")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return rb.VARData.from_frame(frame, tcodes=1)


@pytest.fixture(scope="module")
def qd() -> rb.VARData:
    """GDP, prices (100·log levels) and the Fed funds rate, 1960Q1–2019Q4."""
    return _load("qd_small_levels.csv")


@pytest.fixture(scope="module")
def sim3() -> rb.VARData:
    """Simulated persistent VAR (3 variables), 1970Q1–2019Q4."""
    return _load("sim_var3.csv")


def stationary_panel(n: int, T: int, seed: int, rho: float = 0.6) -> rb.VARData:
    """``T`` observations of ``x_t = ρ x_{t-1} + e_t`` (``e_t ~ N(0, I_n)``), monthly from 1960-02.

    The levels are stored as the cumulated ``x`` with tcode 2, so the transformed sample is ``x``
    and the transform-aware prior mean is 0 (no tcode warning); it stops well before 2020.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 50, n))
    for t in range(1, T + 50):
        y[t] = rho * y[t - 1] + rng.standard_normal(n)
    index = pd.period_range("1960-02", periods=T, freq="M")
    return as_differences(pd.DataFrame(y[50:], index=index, columns=[f"v{i}" for i in range(n)]))


def as_differences(frame: pd.DataFrame) -> rb.VARData:
    """VARData whose transformed sample is ``frame``: cumulated levels (one leading zero row)
    with tcode 2, so every own-lag prior mean is 0."""
    levels = np.vstack([np.zeros((1, frame.shape[1])), np.cumsum(frame.to_numpy(), axis=0)])
    index = pd.period_range(frame.index[0] - 1, periods=len(frame) + 1, freq=frame.index.freq)
    data = rb.VARData.from_frame(pd.DataFrame(levels, index=index, columns=frame.columns), 2)
    np.testing.assert_allclose(data.values, frame.to_numpy(), atol=1e-8)
    return data


def covid_panel(seed: int, n: int = 3, T: int = 312) -> rb.VARData:
    """Monthly persistent VAR(1) in levels, 2000-01 on, whose shocks are scaled by a COVID burst
    ``s = (8, 12, 7)`` from 2020-03 that decays at rate 0.85."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=T, freq="M")
    j = np.array([(p - pd.Period("2020-03", freq="M")).n for p in index])
    decay = 1.0 + 6.0 * 0.85 ** np.maximum(j - 2, 0)
    s = np.where(j < 0, 1.0, np.where(j == 0, 8.0, np.where(j == 1, 12.0, decay)))
    y = np.zeros((T, n))
    for t in range(1, T):
        y[t] = 0.95 * y[t - 1] + s[t] * rng.standard_normal(n)
    frame = pd.DataFrame(y + 50.0, index=index, columns=[f"y{i}" for i in range(n)])
    return rb.VARData.from_frame(frame, tcodes=1)


def bvar(lags: int = 4, outliers=None, **prior) -> rb.BVAR:
    return rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(**prior)), lags=lags, outliers=outliers)


def glp(lags: int) -> rb.BVAR:
    """λ, μ (SOC) and δ (SUR) free; SOC rows for every variable (GLP 2015)."""
    return bvar(lags, soc=1.0, sur=1.0, soc_scope="all")


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def lbfgs(monkeypatch) -> list:
    """Every L-BFGS-B run of the EB optimiser, as ``(start, result)`` (the library only)."""
    runs: list = []
    real = optimize.minimize

    def recording(fun, x0, *args, **kwargs):
        result = real(fun, x0, *args, **kwargs)
        runs.append((np.array(x0, dtype=float), result))
        return result

    monkeypatch.setattr(analytic, "optimize", SimpleNamespace(minimize=recording))
    return runs


def fit_recording(model: rb.BVAR, data: rb.VARData, **kwargs):
    """``(fit, warning messages)`` with every warning recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = model.fit(data, **kwargs)
    return fit, [str(w.message) for w in caught]


def _multistart_count(fit) -> int:
    """L-BFGS-B runs of the deterministic multistart: the default start plus two per axis."""
    names = fit.problem.free.names
    return 1 + 2 * sum(name in names for name in ("soc", "sur", "covid_rho"))


def assert_converged(fit, messages, runs: list | None = None) -> None:
    opt = fit.optimization
    assert set(opt) == OPTIMIZATION_KEYS
    assert opt["success"] is True, opt
    assert fit.converged is True
    assert np.isfinite(opt["log_posterior"])
    assert 0.0 <= opt["predicted_gain"] < TOL, opt
    assert not [m for m in messages if NOT_CONVERGED in m], messages
    assert not [m for m in fit.warnings if NOT_CONVERGED in m]
    if runs is not None:  # the deterministic multistart only; no perturbed restart
        assert len(runs) == _multistart_count(fit), [r.message for _, r in runs]


def neg_log_posterior(problem: ConjugateProblem):
    """``−log p(Y | h) − log p(h)`` as a function of the unconstrained free vector ``z``."""
    free = problem.free
    return lambda z: -problem.log_posterior_vector(free.from_z(np.asarray(z, dtype=float)))


def nelder_mead_polish(fit) -> tuple[float, np.ndarray]:
    """Nelder-Mead from the reported mode on the unconstrained scale (inside the box):
    ``(log-posterior gain over the reported mode, polished z − reported z)``."""
    free = fit.problem.free
    lo, hi = free.z_bounds()
    z = free.to_z(free.vector(fit.hyper))
    objective = neg_log_posterior(fit.problem)
    polished = optimize.minimize(
        lambda u: objective(np.clip(u, lo, hi)),
        z,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxfev": 4000},
    )
    best = np.clip(polished.x, lo, hi)
    return float(-objective(best) - fit.log_posterior()), best - z


# --------------------------------------------------------------------------- _stationarity


def quadratic(H, c):
    H, c = np.asarray(H, dtype=float), np.asarray(c, dtype=float)
    return lambda z: 0.5 * float(np.sum(H * (np.asarray(z) - c) ** 2))


def boxed(f, lo, hi):
    """``f`` that fails the test if it is ever evaluated outside ``[lo, hi]``."""

    def g(z):
        z = np.asarray(z, dtype=float)
        assert np.all(z >= lo) and np.all(z <= hi), f"evaluated outside the box at {z}"
        return f(z)

    return g


WIDE = (np.full(3, -10.0), np.full(3, 10.0))


def test_stationarity_at_an_interior_minimum_is_zero():
    H, c = [4.0, 0.25, 1e4], [0.3, -1.0, 2.0]
    gain, grad = _stationarity(quadratic(H, c), np.array(c), *WIDE)
    assert 0.0 <= gain < 1e-12
    assert grad < 1e-6


def test_stationarity_away_from_the_minimum_is_the_newton_decrement():
    """For a separable quadratic ``½ Σ g_i²/H_ii`` is exactly ``f(z) − min f``; the finite
    differences are exact for quadratics, so the statistic must reproduce it."""
    H, c = np.array([4.0, 0.25, 1e4]), np.array([0.3, -1.0, 2.0])
    f = quadratic(H, c)
    z = c + np.array([0.01, -0.2, 1e-3])
    gain, grad = _stationarity(f, z, *WIDE)
    np.testing.assert_allclose(gain, f(z) - f(c), rtol=1e-6)
    np.testing.assert_allclose(grad, np.max(np.abs(H * (z - c))), rtol=1e-6)
    assert gain > TOL  # an unfinished point is not accepted


def test_stationarity_gain_is_invariant_to_rescaling_the_coordinates():
    """The predicted gain is in objective units and does not depend on how each coordinate is
    scaled (the gradient does): this is what makes the test usable at any n·T."""
    H, c = np.array([4.0, 0.25, 30.0]), np.array([0.3, -1.0, 2.0])
    f = quadratic(H, c)
    z = c + np.array([0.05, -0.3, 0.02])
    a = np.array([10.0, 0.1, 3.0])  # u = z / a

    def g(u):
        return f(np.asarray(u) * a)

    gain_z, grad_z = _stationarity(f, z, *WIDE)
    gain_u, grad_u = _stationarity(g, z / a, *WIDE)
    np.testing.assert_allclose(gain_u, gain_z, rtol=1e-6)
    assert not np.isclose(grad_u, grad_z, rtol=1e-2)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_stationarity_ignores_a_bound_coordinate_whose_descent_leaves_the_box(side):
    """Coordinate 0 sits on its bound and its unconstrained minimum lies outside the box: the
    bound is the constrained minimum, so it contributes nothing (not even to the gradient); the
    objective is never evaluated outside the box."""
    lo, hi = np.array([0.0, -5.0]), np.array([1.0, 5.0])
    outside = -0.7 if side == "lower" else 1.7
    edge = lo[0] if side == "lower" else hi[0]
    H, c = np.array([50.0, 2.0]), np.array([outside, 0.4])
    f = boxed(quadratic(H, c), lo, hi)
    gain, grad = _stationarity(f, np.array([edge, 0.4]), lo, hi)
    assert gain < 1e-12
    assert grad < 1e-6
    # within one finite-difference step of the bound the one-sided scheme is used as well
    near = edge + (5e-5 if side == "lower" else -5e-5)
    gain, grad = _stationarity(f, np.array([near, 0.4]), lo, hi)
    assert gain < 1e-12
    # the other coordinate is still checked
    gain, grad = _stationarity(f, np.array([edge, 0.9]), lo, hi)
    np.testing.assert_allclose(gain, 0.5 * 2.0 * 0.5**2, rtol=1e-6)
    np.testing.assert_allclose(grad, 2.0 * 0.5, rtol=1e-6)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_stationarity_counts_a_bound_coordinate_whose_descent_enters_the_box(side):
    lo, hi = np.array([0.0, -5.0]), np.array([1.0, 5.0])
    inside = 0.3 if side == "lower" else 0.6
    edge = lo[0] if side == "lower" else hi[0]
    H, c = np.array([50.0, 2.0]), np.array([inside, 0.4])
    f = boxed(quadratic(H, c), lo, hi)
    z = np.array([edge, 0.4])
    gain, grad = _stationarity(f, z, lo, hi)
    np.testing.assert_allclose(gain, f(z) - f(c), rtol=1e-6)  # one-sided differences, exact
    np.testing.assert_allclose(grad, 50.0 * abs(edge - inside), rtol=1e-6)


@pytest.mark.parametrize(
    "f",
    [
        lambda z: float(z[0] ** 2 - z[1] ** 2 + z[2] ** 2),  # saddle
        lambda z: float(-np.sum(np.asarray(z) ** 2)),  # maximum
        lambda z: float(z[0] ** 2 + z[2] ** 2),  # flat in z[1]
    ],
    ids=["saddle", "maximum", "flat"],
)
def test_stationarity_without_positive_curvature_is_infinite(f):
    gain, _ = _stationarity(f, np.zeros(3), *WIDE)
    assert gain == np.inf


def test_stationarity_curvature_at_a_bound():
    """Negative curvature on a bound coordinate whose descent enters the box is not a minimum;
    on a pinned coordinate (descent leaves the box) it does not matter."""
    lo, hi = np.array([0.0, -5.0]), np.array([1.0, 5.0])

    def entering(z):
        return float(-((z[0] + 1.0) ** 2) + z[1] ** 2)  # df/dz0 = −2 at 0: descent enters

    def pinned(z):
        return float(-((z[0] - 1.5) ** 2) + z[1] ** 2)  # df/dz0 = +3 at 0: descent leaves

    assert _stationarity(boxed(entering, lo, hi), np.zeros(2), lo, hi)[0] == np.inf
    assert _stationarity(boxed(pinned, lo, hi), np.zeros(2), lo, hi)[0] < 1e-12


# --------------------------------------------------------------------------- realistic sizes


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_eb_converges_on_a_simulated_n20_p4_panel(seed, lbfgs):
    data = stationary_panel(20, 300, seed)
    fit, messages = fit_recording(glp(4), data)
    assert fit.problem.free.names == ("lambda", "soc", "sur")
    assert_converged(fit, messages, lbfgs)
    assert messages == []
    assert fit.forecast(2, draws=20, seed=1).dataset.attrs["converged"] == 1


def test_eb_mode_at_n20_survives_a_nelder_mead_polish():
    fit, messages = fit_recording(glp(4), stationary_panel(20, 300, seed=0))
    assert_converged(fit, messages)
    gain, dz = nelder_mead_polish(fit)
    assert gain < TOL, gain
    assert np.max(np.abs(dz)) < 1e-3, dz


def test_eb_lambda_at_n20_is_the_maximiser_on_a_dense_grid():
    data = stationary_panel(20, 300, seed=1)
    fit, messages = fit_recording(bvar(4), data)
    assert fit.problem.free.names == ("lambda",)
    assert_converged(fit, messages)
    lam = fit.hyper["lambda"]
    grid = np.geomspace(1e-4, 5.0, 1201)  # the default bounds of λ
    values = np.array([fit.log_posterior({"lambda": float(x)}) for x in grid])
    best = int(values.argmax())
    assert 0 < best < grid.size - 1
    assert fit.log_posterior() >= values.max() - 1e-9
    assert grid[best - 1] <= lam <= grid[best + 1]
    refined = optimize.minimize_scalar(
        lambda x: -fit.log_posterior({"lambda": float(x)}),
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    np.testing.assert_allclose(lam, refined.x, rtol=1e-4)
    assert fit.log_posterior() >= -refined.fun - TOL


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
def test_eb_converges_on_a_simulated_n50_p4_panel(seed, lbfgs):
    """Before the scale-free test every one of these seeds was flagged non-converged."""
    fit, messages = fit_recording(glp(4), stationary_panel(50, 300, seed))
    assert_converged(fit, messages, lbfgs)
    gain, dz = nelder_mead_polish(fit)
    assert gain < TOL and np.max(np.abs(dz)) < 1e-3, (gain, dz)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("lags", "seed"), [(4, 0), (4, 1), (13, 0), (13, 1)], ids=["p4-0", "p4-1", "p13-0", "p13-1"]
)
def test_eb_converges_on_a_simulated_n100_panel(lags, seed, lbfgs):
    """The Phase 1 target size (n = 100, p = 13, k = 1301)."""
    fit, messages = fit_recording(glp(lags), stationary_panel(100, 300, seed))
    assert fit.problem.k == 1 + 100 * lags
    assert_converged(fit, messages, lbfgs)


@pytest.fixture(scope="module")
def fred_qd_path(tmp_path_factory) -> Path:
    from rai_bvar.datasets import fred

    cache = os.environ.get("RAI_BVAR_DATA") or tmp_path_factory.mktemp("fred")
    return fred.fetch("fred-qd", "2026-09", cache_dir=cache)


@pytest.mark.network
@pytest.mark.parametrize(("n", "lags"), [(20, 13), (100, 4), (100, 13), (160, 4)])
def test_eb_converges_on_real_fred_qd(fred_qd_path, n, lags, lbfgs):
    """FRED-QD 2026-09, the first ``n`` series complete over 1959–2019 with their official
    tcodes, to 2019Q4: with an absolute gradient tolerance n ≥ 100 used to be flagged."""
    from rai_bvar.datasets import fred

    raw, _ = fred.parse_fred_csv(fred_qd_path, "qd")
    window = raw.loc[pd.Period("1959Q1", freq="Q") : pd.Period("2019Q4", freq="Q")]
    complete = [c for c in window.columns if np.isfinite(window[c].to_numpy()).all()]
    assert len(complete) >= n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # loader notes (leading periods etc.)
        data = fred.load(
            "fred-qd",
            "2026-09",
            variables=complete[:n],
            tcodes="official",
            cache_dir=fred_qd_path.parents[1],
        ).upto("2019Q4")
    fit, messages = fit_recording(bvar(lags, soc=1.0, sur=1.0), data)
    assert fit.problem.n == n
    assert_converged(fit, messages, lbfgs)


# --------------------------------------------------------------------------- Lenza-Primiceri


LP_ENDS = ["2024-01", "2024-06", "2024-12"]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dummies", [False, True], ids=["lambda", "soc-sur"])
def test_monthly_lenza_primiceri_samples_ending_2024_converge(seed, dummies, lbfgs):
    """Samples reaching years past 2020 inform ρ sharply (curvature ~ 4e3 on its scale); the
    optimiser used to stop where only its forward-difference gradient vanished, and most such
    fits were flagged by the absolute gradient test."""
    data = covid_panel(seed)
    prior = {"soc": 1.0, "sur": 1.0} if dummies else {}
    model = bvar(4, outliers="lenza-primiceri", **prior)
    for end in LP_ENDS:
        lbfgs.clear()
        fit, messages = fit_recording(model, data.upto(end))
        assert fit.problem.free.names[-4:] == (
            "covid_s0",
            "covid_s1",
            "covid_s2",
            "covid_rho",
        )
        assert_converged(fit, messages, lbfgs)
        assert messages == [], end


def test_monthly_lenza_primiceri_n10_converges(lbfgs):
    data = covid_panel(0, n=10)
    for end in ("2024-01", "2024-12"):
        lbfgs.clear()
        fit, messages = fit_recording(
            bvar(4, outliers="lenza-primiceri", soc=1.0, sur=1.0), data.upto(end)
        )
        assert_converged(fit, messages, lbfgs)
        assert fit.forecast(3, draws=20, seed=1).dataset.attrs["converged"] == 1


def test_monthly_lenza_primiceri_mode_survives_a_nelder_mead_polish():
    fit, messages = fit_recording(
        bvar(4, outliers="lenza-primiceri"), covid_panel(1).upto("2024-06")
    )
    assert_converged(fit, messages)
    rho = fit.hyper["covid"][3]
    assert 0.5 < rho < 0.99  # an interior ρ (on the logit scale)
    gain, dz = nelder_mead_polish(fit)
    assert gain < TOL, gain
    assert np.max(np.abs(dz)) < 1e-3, dz


@pytest.mark.slow
@pytest.mark.parametrize("lags", [4, 13])
@pytest.mark.parametrize("seed", range(3))
def test_monthly_lenza_primiceri_scan_2022_to_2025(seed, lags):
    """Round-2 scan of samples ending 2022-12 … 2025-11: 32 of 48 fits (n = 3, p = 4) were
    flagged, and every p = 13 run."""
    data = covid_panel(seed, n=10 if lags == 13 else 3)
    failures = []
    for end in ("2022-12", "2023-06", "2024-03", "2024-09", "2025-06", "2025-11"):
        for prior in ({}, {"soc": 1.0, "sur": 1.0}):
            fit, messages = fit_recording(
                bvar(lags, outliers="lenza-primiceri", **prior), data.upto(end)
            )
            opt = fit.optimization
            if not (opt["success"] and fit.converged) or messages:
                failures.append((end, bool(prior), opt, messages))
    assert not failures, failures


# --------------------------------------------------------------------------- warm starts


WARM_CONFIGS = {
    "lambda": {},
    "soc-sur": {"soc": 1.0, "sur": 1.0},
    "soc": {"soc": 1.0},
    "sur": {"sur": 1.0},
}


@pytest.mark.parametrize("dataset", ["qd", "sim3"])
@pytest.mark.parametrize("config", list(WARM_CONFIGS))
def test_warm_start_at_the_mode_is_converged(request, dataset, config, lbfgs):
    """Re-fitting from the reported mode (exactly, or rounded to 6 decimals) must be accepted
    without a restart: L-BFGS-B stops at (or next to) the start and the stationarity test must
    agree with it."""
    data = request.getfixturevalue(dataset)
    prior = rb.Minnesota(**WARM_CONFIGS[config])
    for lags in (1, 2, 4, 8):
        lbfgs.clear()
        cold, messages = fit_recording(rb.BVAR(mean=rb.Linear(prior=prior), lags=lags), data)
        assert_converged(cold, messages, lbfgs)
        h = cold.hyper
        for digits in (None, 6):
            r = (lambda v: v) if digits is None else (lambda v, d=digits: round(v, d))
            warm_prior = replace(
                prior,
                lambda_=r(h["lambda"]),
                soc=None if h["soc"] is None else r(h["soc"]),
                sur=None if h["sur"] is None else r(h["sur"]),
            )
            lbfgs.clear()
            warm, messages = fit_recording(
                rb.BVAR(mean=rb.Linear(prior=warm_prior), lags=lags), data
            )
            assert_converged(warm, messages, lbfgs)
            np.testing.assert_allclose(
                warm.optimization["log_posterior"],
                cold.optimization["log_posterior"],
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                warm.problem.free.vector(warm.hyper), cold.problem.free.vector(h), rtol=1e-3
            )


def test_warm_start_at_the_mode_is_converged_under_mh(qd):
    """Under ``hierarchy="mh"`` the EB mode seeds the chains; a warm start must not mark a run
    whose chains pass R-hat and ESS as non-converged."""
    cold, _ = fit_recording(bvar(4, mean=1.0), qd)
    settings = rb.MHSettings(draws=1500, burn=500, chains=4)
    warm, messages = fit_recording(
        bvar(4, mean=1.0, hierarchy="mh", lambda_=cold.hyper["lambda"], mh=settings), qd
    )
    assert warm.optimization["success"] is True, warm.optimization
    assert not [m for m in messages if NOT_CONVERGED in m]
    assert warm.mh.converged, (warm.mh.rhat, warm.mh.ess)
    assert warm.converged is True
    assert warm.forecast(2, draws=20, seed=1).dataset.attrs["converged"] == 1


# --------------------------------------------------------------------------- modes on bounds


def assert_on_bound(fit, name: str, side: str) -> None:
    free = fit.problem.free
    i = free.names.index(name)
    bound = (free.lower if side == "lower" else free.upper)[i]
    np.testing.assert_allclose(free.vector(fit.hyper)[i], bound, rtol=1e-9)


def assert_constrained_maximum(fit, starts: int = 8) -> None:
    """No point of the box is better than the reported mode: a Nelder-Mead polish (clipped to
    the box) and box-constrained L-BFGS-B runs from a perturbed start and from random starts in
    the box all gain less than TOL."""
    gain, _ = nelder_mead_polish(fit)
    assert gain < TOL, gain
    free = fit.problem.free
    lo, hi = free.z_bounds()
    z = free.to_z(free.vector(fit.hyper))
    objective = neg_log_posterior(fit.problem)
    rng = np.random.default_rng(0)
    for start in [np.clip(z + 0.3, lo, hi), *(rng.uniform(lo, hi) for _ in range(starts))]:
        other = optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac="3-point",
            bounds=list(zip(lo, hi, strict=True)),
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000},
        )
        assert -other.fun - fit.log_posterior() < TOL, (start, other.x, -other.fun)


def test_eb_alpha_on_its_default_upper_bound_3(lbfgs):
    """With 13 lags of a weakly persistent process and a loose prior on α the lag decay wants to
    exceed its default cap α ≤ 3."""
    hp = rb.GLPHyperpriors(alpha=rb.GammaPrior(mode=2.0, sd=1.0, lower=1.0, upper=3.0))
    assert hp.alpha.upper == rb.GLPHyperpriors().alpha.upper == 3.0
    data = stationary_panel(4, 400, seed=3, rho=0.5)
    wide = replace(hp, alpha=replace(hp.alpha, upper=8.0))  # positive control: the cap binds
    uncapped, _ = fit_recording(bvar(13, hyper=("lambda", "alpha"), hyperpriors=wide), data)
    assert uncapped.hyper["alpha"] > 3.5
    lbfgs.clear()
    model = bvar(13, hyper=("lambda", "alpha"), hyperpriors=hp)
    fit, messages = fit_recording(model, data)
    assert fit.problem.free.names == ("lambda", "alpha")
    assert_on_bound(fit, "alpha", "upper")
    assert fit.hyper["alpha"] == 3.0
    assert_converged(fit, messages, lbfgs)
    assert_constrained_maximum(fit)


def _lagged_copy_frame(seed: int = 5, T: int = 300) -> pd.DataFrame:
    """``y_t = 10 x_{t-1} + 0.001 e_t``: univariately ``y`` is white noise with variance ~100,
    inside the VAR it is almost exactly predicted, so ψ_y wants far below its AR default."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    y = np.r_[0.0, 10.0 * x[:-1]] + 1e-3 * rng.standard_normal(T)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.7 * z[t - 1] + rng.standard_normal()
    index = pd.period_range("1960-01", periods=T, freq="M")
    return pd.DataFrame({"x": x, "y": y, "z": z}, index=index)


def test_eb_psi_on_its_data_dependent_lower_bound(lbfgs):
    """With ψ free and no ψ bounds given, the box is the AR residual variance / 100 … × 100."""
    data = as_differences(_lagged_copy_frame())
    fit, messages = fit_recording(bvar(2, hyper=("lambda", "psi")), data)
    free, default = fit.problem.free, fit.problem.default["psi"]
    np.testing.assert_allclose(free.lower[1:], default / 100.0, rtol=1e-12)
    np.testing.assert_allclose(free.upper[1:], default * 100.0, rtol=1e-12)
    assert_on_bound(fit, "psi_x", "lower")
    assert_on_bound(fit, "psi_y", "lower")
    z_index = free.names.index("psi_z")
    assert free.lower[z_index] * 1.01 < fit.hyper["psi"][2] < free.upper[z_index] / 1.01
    assert_converged(fit, messages, lbfgs)
    assert_constrained_maximum(fit)
    # positive control: with explicit wide ψ bounds the modes of ψ_x and ψ_y lie far below
    ig = rb.GLPHyperpriors().psi
    wide = rb.GLPHyperpriors(psi=replace(ig, lower=1e-10, upper=1e6))
    unboxed, _ = fit_recording(bvar(2, hyper=("lambda", "psi"), hyperpriors=wide), data)
    assert np.all(unboxed.hyper["psi"][:2] < 0.1 * default[:2] / 100.0)


def test_eb_psi_on_the_bounds_of_a_user_psi(lbfgs):
    """A user ψ far too small (large) for a variable puts its mode on the upper (lower) edge of
    the user-ψ ×/÷ 100 box; the middle variable stays interior."""
    data = stationary_panel(3, 300, seed=1, rho=0.9)
    model = bvar(4, psi=(1e-6, 0.01, 1e6), hyper=("lambda", "psi"))
    fit, messages = fit_recording(model, data)
    assert fit.problem.free.names == ("lambda", "psi_v0", "psi_v1", "psi_v2")
    assert_on_bound(fit, "psi_v0", "upper")
    assert_on_bound(fit, "psi_v2", "lower")
    x, free = fit.problem.free.vector(fit.hyper), fit.problem.free
    assert free.lower[2] * 1.01 < x[2] < free.upper[2] / 1.01
    assert_converged(fit, messages, lbfgs)
    assert_constrained_maximum(fit)


@pytest.mark.parametrize("bound", [0.05, 0.2], ids=["inside", "start-on-bound"])
def test_eb_lambda_on_a_lower_bound(bound, lbfgs):
    """White noise with prior mean 0: the λ mode (≈ 0.02) lies below the bound. With bound 0.2
    the optimiser starts exactly on it (the default λ = 0.2) and must stay there."""
    hp = rb.GLPHyperpriors(lambda_=rb.GammaPrior(mode=0.2, sd=0.4, lower=bound, upper=5.0))
    data = stationary_panel(3, 300, seed=4, rho=0.0)
    unconstrained, _ = fit_recording(bvar(4), data)
    assert unconstrained.hyper["lambda"] < bound
    lbfgs.clear()
    fit, messages = fit_recording(bvar(4, hyperpriors=hp), data)
    assert_on_bound(fit, "lambda", "lower")
    assert_converged(fit, messages, lbfgs)
    assert_constrained_maximum(fit)


def test_eb_lambda_on_a_lower_bound_with_free_dummies(qd, lbfgs):
    """λ capped below at 0.6 (its unconstrained mode with SOC/SUR is ≈ 0.44): λ sits on the
    bound and (μ, δ) equal their EB mode with λ fixed at 0.6."""
    hp = rb.GLPHyperpriors(lambda_=rb.GammaPrior(mode=0.2, sd=0.4, lower=0.6, upper=5.0))
    fit, messages = fit_recording(bvar(4, mean=1.0, soc=1.0, sur=1.0, hyperpriors=hp), qd)
    assert_on_bound(fit, "lambda", "lower")
    assert_converged(fit, messages, lbfgs)
    pinned, _ = fit_recording(
        bvar(4, mean=1.0, lambda_=0.6, soc=1.0, sur=1.0, hyper=("soc", "sur")), qd
    )
    np.testing.assert_allclose(
        [fit.hyper["soc"], fit.hyper["sur"]], [pinned.hyper["soc"], pinned.hyper["sur"]], rtol=1e-3
    )
    assert_constrained_maximum(fit)


# --------------------------------------------------------------------------- failures flagged


def _truncate_lbfgs(monkeypatch, maxiter: int = 1) -> list:
    runs: list = []
    real = optimize.minimize

    def truncated(fun, x0, *args, options=None, **kwargs):
        result = real(fun, x0, *args, options={**(options or {}), "maxiter": maxiter}, **kwargs)
        runs.append((np.array(x0, dtype=float), result))
        return result

    monkeypatch.setattr(analytic, "optimize", SimpleNamespace(minimize=truncated))
    return runs


@pytest.mark.parametrize("prior", [{}, {"soc": 1.0, "sur": 1.0}], ids=["lambda", "soc-sur"])
def test_an_unfinished_optimisation_is_flagged(qd, monkeypatch, prior):
    full = bvar(4, mean=1.0, **prior).fit(qd)
    assert full.optimization["success"]
    runs = _truncate_lbfgs(monkeypatch, maxiter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = bvar(4, mean=1.0, **prior).fit(qd)
    flagged = [w for w in caught if NOT_CONVERGED in str(w.message)]
    assert len(flagged) == 1, [str(w.message) for w in caught]
    w = flagged[0]
    text = str(w.message)
    assert w.category is UserWarning
    assert Path(w.filename) == Path(__file__)  # points at the caller's fit()
    opt = fit.optimization
    assert set(opt) == OPTIMIZATION_KEYS
    assert opt["success"] is False
    assert "ITERATIONS" in opt["message"] and opt["message"] in text
    assert "predicted gain" in text and "projected gradient" in text
    assert "fit.converged is False" in text
    assert opt["predicted_gain"] > TOL
    assert full.optimization["log_posterior"] - opt["log_posterior"] > TOL  # really unfinished
    starts = _multistart_count(fit)
    assert len(runs) == starts + 1  # the deterministic multistart, then one perturbed restart
    assert not np.array_equal(runs[0][0], runs[-1][0])
    assert fit.converged is False
    assert text in fit.warnings
    attrs = fit.forecast(2, draws=20, seed=1).dataset.attrs
    assert attrs["converged"] == 0
    assert text in attrs["warnings"]


def test_an_optimisation_without_a_finite_log_posterior_is_flagged(qd, monkeypatch):
    monkeypatch.setattr(ConjugateProblem, "log_posterior_vector", lambda self, x: -np.inf)
    fit, messages = fit_recording(bvar(4, mean=1.0, soc=1.0, sur=1.0), qd)
    assert fit.optimization["success"] is False
    assert fit.optimization["log_posterior"] == -np.inf
    assert fit.converged is False
    assert [m for m in messages if NOT_CONVERGED in m]
    assert fit.forecast(2, draws=10, seed=1).dataset.attrs["converged"] == 0


@pytest.mark.parametrize("hierarchy", ["eb", "mh"])
def test_an_unfinished_optimisation_overrides_converged_chains(qd, monkeypatch, hierarchy):
    """``fit.converged`` is the EB status AND the MH diagnostics."""
    _truncate_lbfgs(monkeypatch, maxiter=1)
    settings = rb.MHSettings(draws=1500, burn=500, chains=4)
    fit, messages = fit_recording(bvar(4, mean=1.0, hierarchy=hierarchy, mh=settings), qd)
    assert fit.optimization["success"] is False
    assert len([m for m in messages if NOT_CONVERGED in m]) == 1
    if hierarchy == "mh":
        assert fit.mh.converged  # the chains themselves are fine (they adapt from the start)
        assert not [m for m in messages if "MH did not converge" in m]
    assert fit.converged is False
    assert fit.forecast(2, draws=10, seed=1).dataset.attrs["converged"] == 0


# --------------------------------------------------------------------------- fit.optimization


def _spy_counts(monkeypatch) -> dict:
    """Counts objective calls made by L-BFGS-B and by ``_stationarity``, and every evaluation of
    ``ConjugateProblem.log_posterior_vector``."""
    counts = {"minimize": 0, "stationarity": 0, "log_posterior": 0, "runs": 0}
    real_minimize, real_stationarity = analytic.optimize.minimize, analytic._stationarity
    real_lpv = ConjugateProblem.log_posterior_vector

    def minimize(fun, x0, *args, **kwargs):
        counts["runs"] += 1

        def counted(z):
            counts["minimize"] += 1
            return fun(z)

        return real_minimize(counted, x0, *args, **kwargs)

    def stationarity(objective, *args, **kwargs):
        def counted(z):
            counts["stationarity"] += 1
            return objective(z)

        return real_stationarity(counted, *args, **kwargs)

    def lpv(self, x):
        counts["log_posterior"] += 1
        return real_lpv(self, x)

    monkeypatch.setattr(analytic, "optimize", SimpleNamespace(minimize=minimize))
    monkeypatch.setattr(analytic, "_stationarity", stationarity)
    monkeypatch.setattr(ConjugateProblem, "log_posterior_vector", lpv)
    return counts


def test_optimization_record(qd):
    fit, _ = fit_recording(bvar(4, mean=1.0, soc=1.0, sur=1.0), qd)
    opt = fit.optimization
    assert set(opt) == OPTIMIZATION_KEYS
    assert isinstance(opt["success"], bool) and isinstance(opt["message"], str)
    assert isinstance(opt["evaluations"], int) and opt["evaluations"] > 0
    for key in ("log_posterior", "projected_gradient", "predicted_gain"):
        assert isinstance(opt[key], float)
    np.testing.assert_allclose(opt["log_posterior"], fit.log_posterior(), rtol=1e-12)
    # the reported diagnostics are those of the reported mode
    free = fit.problem.free
    z = free.to_z(free.vector(fit.hyper))
    gain, grad = _stationarity(neg_log_posterior(fit.problem), z, *free.z_bounds())
    assert gain < TOL and opt["predicted_gain"] < TOL
    np.testing.assert_allclose(opt["projected_gradient"], grad, rtol=0, atol=1e-4)
    assert opt["success"] == bool(np.isfinite(opt["log_posterior"]) and gain < TOL)
    fixed = bvar(4, mean=1.0, hierarchy="fixed").fit(qd)
    assert fixed.optimization is None and fixed.converged is True


@pytest.mark.parametrize("truncate", [False, True], ids=["converged", "with-restart"])
def test_evaluations_count_every_objective_call(qd, monkeypatch, truncate):
    """Both L-BFGS-B runs (when the restart runs) and every stationarity check count."""
    if truncate:
        _truncate_lbfgs(monkeypatch, maxiter=1)
    counts = _spy_counts(monkeypatch)
    fit, _ = fit_recording(bvar(4, mean=1.0, soc=1.0, sur=1.0), qd)
    starts = _multistart_count(fit)
    assert counts["runs"] == (starts + 1 if truncate else starts)
    assert counts["stationarity"] > 0
    # plus one re-evaluation of each run's end point
    expected = counts["minimize"] + counts["stationarity"] + counts["runs"]
    assert fit.optimization["evaluations"] == expected


@pytest.mark.parametrize("truncate", [False, True], ids=["converged", "with-restart"])
def test_evaluations_count_every_log_posterior_evaluation(qd, monkeypatch, truncate):
    """``evaluations`` is the optimiser's cost: every log-posterior evaluation it made, including
    the re-evaluation at each run's end point (round-2 finding: "the re-evaluations at
    result.x and retry.x are never counted")."""
    if truncate:
        _truncate_lbfgs(monkeypatch, maxiter=1)
    counts = _spy_counts(monkeypatch)
    fit, _ = fit_recording(bvar(4, mean=1.0, soc=1.0, sur=1.0), qd)
    assert fit.mh is None  # every log-posterior evaluation of an EB fit is the optimiser's
    assert fit.optimization["evaluations"] == counts["log_posterior"], counts
