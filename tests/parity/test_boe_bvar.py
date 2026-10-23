"""Parity with the Bank of England ``bvar`` package (PyPI ``bvar``, MIT; SPEC §10.4, decision D5).

BoE ``bvar`` is run directly in the test (no fixtures): it is pip-installable and fast. The
parametrisation map, every known difference from GLP (2015) and from ``rai-bvar``, and the
reasons behind each tolerance are in ``BOE_BVAR_NOTES.md`` next to this file. In short:

* Only exactly mappable quantities are compared: the log marginal likelihood at fixed
  hyperparameters, the NIW posterior moments, hyperparameter modes and forecast quantiles.
* ψ is always passed explicitly: BoE fixes it at its own AR(1) residual MSE (never estimated).
* BoE's SOC/SUR dummies average observations ``p+1..2p``; GLP (and ``rai-bvar``) average the
  first ``p`` observations. Every SOC/SUR comparison therefore runs BoE with its two dummy
  builders wrapped so that they receive GLP's ``ȳ_0`` (the rest of BoE's code is untouched).
  ``test_boe_dummy_initial_mean_convention`` pins BoE's own convention.
* The ``test_boe_*`` tests check the oracle itself (layout, prior variances, ψ, dummies, log-ML
  formula) and run without the ``rai-bvar`` engine; the ``test_parity_*`` tests compare the
  engine and skip until ``rai_bvar.BVAR`` exists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import multigammaln

bvar = pytest.importorskip(
    "bvar", reason="Bank of England `bvar` is not installed (pip install bvar)"
)

import bvar.dummy_observations as boe_dummies  # noqa: E402
from bvar.models.common import ar1_mse  # noqa: E402
from bvar.models.conjugate.marginal_likelihood import objective_function  # noqa: E402
from bvar.models.conjugate.minnesota import prior_minnesota  # noqa: E402
from bvar.utils import construct_Y_Z  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
BOE_VERSION_MAPPED = "0.3.5"  # version the mapping in BOE_BVAR_NOTES.md was derived from
NO_COVID = np.empty(0, dtype=int)
BOE_FAILURE = -1e5  # BoE's log_marginal_likelihood returns this on a failed factorisation
BOE_NAME = {"lambda": "c1", "alpha": "c3", "soc": "mu", "sur": "theta"}

# Tolerances (reasons and measured margins: BOE_BVAR_NOTES.md §6–§8, §14).
RTOL_LOGML = 1e-8
RTOL_POSTERIOR = 1e-8
# BoE's S_post = S0 + A0'V⁻¹A0 + Y'Y − A'KA cancels on level data (elementwise error ≤ 4e-6,
# 4e-9 on the correlation scale); 1e-7 still catches a one-off error in ν̄ (≈ 4e-3).
CORR_TOL_BOE_SIGMA_POINT = 1e-7
RTOL_HYPER = 1e-3
ATOL_OBJECTIVE = 1e-4  # our mode may not be worse than BoE's by more than this (log points)
RTOL_BOE_OPTIMISER = 0.1  # BoE's own finite-difference BFGS on level data (measured ≤ 2.6%)
Z_MC = 5.0  # Monte Carlo z-score bound (≈120 correlated comparisons per test)


# --------------------------------------------------------------------------------------- cases


@dataclass(frozen=True)
class Case:
    """One configuration, expressed in ``rai-bvar`` terms (BoE names in comments)."""

    id: str
    dataset: str  # "qd" (levels, prior mean 1) or "sim" (stationary, prior mean 0)
    lags: int
    lam: float  # λ        ↔ BoE c1
    alpha: float  # α      ↔ BoE c3
    soc: float | None = None  # μ  ↔ BoE mu    (model.soc flag)
    sur: float | None = None  # δ  ↔ BoE theta (model.sur flag)
    var_const: float = 1e7  # ↔ BoE lambda_constant

    @property
    def levels(self) -> bool:  # BoE stationary = not levels
        return self.dataset == "qd"

    @property
    def prior_mean(self) -> float:
        return 1.0 if self.levels else 0.0

    @property
    def dummies(self) -> bool:
        return self.soc is not None or self.sur is not None


FIXED_CASES = [
    Case("qd-p4-glp", "qd", 4, lam=0.2, alpha=2.0),
    Case("qd-p4-soc", "qd", 4, lam=0.35, alpha=1.7, soc=0.6),
    Case("qd-p4-sur", "qd", 4, lam=0.15, alpha=2.5, sur=1.8, var_const=10.0),
    Case("qd-p4-soc-sur", "qd", 4, lam=0.2, alpha=2.0, soc=1.0, sur=1.0),
    Case("qd-p2-soc-sur", "qd", 2, lam=0.4, alpha=1.0, soc=0.5, sur=2.0, var_const=1e5),
    Case("sim-p4", "sim", 4, lam=0.25, alpha=2.0),
    Case("sim-p2", "sim", 2, lam=0.5, alpha=3.0, var_const=100.0),
]
EB_CASES = [
    Case("qd-p4", "qd", 4, lam=0.2, alpha=2.0),
    Case("qd-p4-soc-sur", "qd", 4, lam=0.2, alpha=2.0, soc=1.0, sur=1.0),
    Case("sim-p4", "sim", 4, lam=0.2, alpha=2.0),
    Case("sim-p2", "sim", 2, lam=0.2, alpha=2.0),
]
FORECAST_CASES = [
    Case("qd-p4-soc-sur", "qd", 4, lam=0.2, alpha=2.0, soc=1.0, sur=1.0),
    Case("sim-p2", "sim", 2, lam=0.3, alpha=2.0),
]


def _ids(cases: list[Case]) -> list[str]:
    return [c.id for c in cases]


# ---------------------------------------------------------------------------------- data, BoE


def _load(dataset: str) -> pd.DataFrame:
    name = {"qd": "qd_small_levels", "sim": "sim_var3"}[dataset]
    frame = pd.read_csv(FIXTURES / f"{name}.csv")
    frame.index = pd.PeriodIndex(frame.pop("date"), freq="Q")
    return frame.astype(np.float64)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {"qd": _load("qd"), "sim": _load("sim")}


_BOE_SOC_BUILDER = boe_dummies.sum_of_coefficients_prior
_BOE_SUR_BUILDER = boe_dummies.single_unit_root_prior


def _use_glp_initial_mean(monkeypatch: pytest.MonkeyPatch, data: np.ndarray, lags: int) -> None:
    """Make BoE build SOC/SUR dummies from GLP's ȳ_0 = mean of the first ``p`` observations.

    BoE's builders average the first ``p`` rows of the post-lag ``Y`` (observations p+1..2p).
    The wrappers hand BoE's own builders the first ``p`` observations instead; ``stack_dummies``
    resolves the builders through the module namespace, so the sampler, the ML objective and
    the optimiser all pick them up.
    """
    initial = np.array(data[:lags], dtype=np.float64)

    def soc(y, n_lags, mu, levels, covid_indices=None):
        covid = NO_COVID if covid_indices is None else covid_indices
        return _BOE_SOC_BUILDER(initial, n_lags, mu, levels, covid_indices=covid)

    def sur(y, n_lags, theta, levels, covid_indices=None):
        covid = NO_COVID if covid_indices is None else covid_indices
        return _BOE_SUR_BUILDER(initial, n_lags, theta, levels, covid_indices=covid)

    monkeypatch.setattr(boe_dummies, "sum_of_coefficients_prior", soc)
    monkeypatch.setattr(boe_dummies, "single_unit_root_prior", sur)


def _boe_model(case: Case, *, c3_sd: float = 0.5):
    model = bvar.NaturalConjugate(soc=case.soc is not None, sur=case.sur is not None)
    model.set_priors(
        c1=case.lam,
        c3=case.alpha,
        lambda_constant=case.var_const,
        mu=1.0 if case.soc is None else case.soc,
        theta=1.0 if case.sur is None else case.sur,
        c3_sd=c3_sd,
    )
    return model


def _boe_bvar(case: Case, *, method: str = "none", seed: int | None = None, c3_sd: float = 0.5):
    return bvar.BVAR(
        n_lags=case.lags,
        model=_boe_model(case, c3_sd=c3_sd),
        stationary=not case.levels,
        optimisation_method=method,
        random_state=seed,
    )


def _inv_softplus(x) -> np.ndarray:
    return np.log(np.expm1(np.asarray(x, dtype=np.float64)))


class BoEObjective:
    """BoE's own hyperparameter objective ``−[log p(Y | h) + log p(h)]`` (numpy path).

    ``add_priors=False`` gives ``−log p(Y | h)``. Values are passed in ``rai-bvar`` names;
    hyperparameters not given keep the case values.
    """

    def __init__(self, case: Case, frame: pd.DataFrame, *, add_priors: bool, c3_sd: float = 0.5):
        self.case = case
        self.data = frame.to_numpy(dtype=np.float64)
        n = self.data.shape[1]
        self.model = _boe_model(case, c3_sd=c3_sd)
        self.model.pars.nu_0 = n + 2
        self.model.pars.S_0 = np.diag(ar1_mse(self.data, NO_COVID)) * (self.model.pars.nu_0 - n - 1)
        self.levels = np.ones(n) if case.levels else np.zeros(n)
        self.Y, self.Z = construct_Y_Z(self.data, case.lags, NO_COVID)
        self.add_priors = add_priors
        self.order = ["lambda", "alpha"] + [
            h for h in ("soc", "sur") if getattr(case, h) is not None
        ]
        self.defaults = {"lambda": case.lam, "alpha": case.alpha, "soc": case.soc, "sur": case.sur}

    @property
    def psi(self) -> np.ndarray:
        """ψ in ``rai-bvar`` units: the IW scale diagonal, ``diag(S_0)``."""
        return np.diag(self.model.pars.S_0).copy()

    def __call__(self, values: dict[str, float] | None = None) -> float:
        h = {**self.defaults, **(values or {})}
        x = _inv_softplus([h[name] for name in self.order])
        value = objective_function(
            x,
            self.data,
            self.case.lags,
            NO_COVID,
            self.levels,
            0,
            self.model,
            self.Y,
            self.Z,
            self.add_priors,
            self.case.soc is not None,
            self.case.sur is not None,
        )
        assert -value > BOE_FAILURE + 1.0, f"BoE log-ML hit its failure value at {h}"
        return float(value)

    def mode(self, free: list[str], start: dict[str, float]) -> tuple[dict[str, float], float]:
        """Mode of this objective over ``free`` (others fixed), polished to ~1e-5 relative.

        BoE's own optimiser uses forward-difference gradients, which are swamped by the
        ~1e-9 rounding noise of the level-data log-ML; restarted Nelder-Mead on log h is not.
        """

        def f(z: np.ndarray) -> float:
            return self(dict(zip(free, np.exp(z), strict=True)))

        z = np.log([start[name] for name in free])
        for _ in range(3):
            res = minimize(
                f, z, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxfev": 20000}
            )
            z = res.x
        return dict(zip(free, np.exp(z), strict=True)), float(res.fun)


@dataclass
class BoEPosterior:
    mean: np.ndarray  # B̄, (k, n)
    sigma_point: np.ndarray  # BoE's posterior mean of Σ (cancellation formula)
    scale: np.ndarray  # S̄, residual form from BoE's B̄, prior and design
    df: float  # ν̄ = ν0 + T + #dummy rows
    omega: np.ndarray  # Ω̄ = (V⁻¹ + Z'Z)⁻¹ from BoE's prior precision and design
    psi: np.ndarray  # diag(S_0)


def _boe_posterior(case: Case, frame: pd.DataFrame) -> BoEPosterior:
    model = _boe_bvar(case)
    model.sample(frame, N_draws=1, point_only=True, progressbar=False)
    n, k = model.n, model.k
    beta_0 = model.model.beta_0.reshape(n, k).T
    precision = model.model.V_A_inv
    S0, nu0 = model.model.pars.S_0, model.model.pars.nu_0
    Y, Z = construct_Y_Z(model.data, case.lags, NO_COVID)
    if model.soc_ or model.sur_:
        Y, Z, _ = boe_dummies.stack_dummies(
            Y,
            Z,
            case.lags,
            model.vars_in_levels,
            model.model,
            NO_COVID,
            soc=model.soc_,
            sur=model.sur_,
        )
    B = model.beta_point.reshape(n, k).T
    E, D = Y - Z @ B, B - beta_0
    return BoEPosterior(
        mean=B,
        sigma_point=model.sigma_point.reshape(n, n),
        scale=S0 + E.T @ E + D.T @ precision @ D,
        df=float(nu0 + Y.shape[0]),
        omega=np.linalg.inv(precision + Z.T @ Z),
        psi=np.diag(S0).copy(),
    )


# ------------------------------------------------------------------------------ rai-bvar side


@pytest.fixture(scope="module")
def rb():
    try:
        import rai_bvar
    except ImportError as exc:  # pragma: no cover - the package is installed in dev envs
        pytest.skip(f"rai_bvar not importable: {exc}")
    missing = [
        n
        for n in ("BVAR", "Linear", "Constant", "Minnesota", "GLPHyperpriors")
        if not hasattr(rai_bvar, n)
    ]
    if missing:
        pytest.skip(
            f"rai_bvar analytic engine not available yet (missing {missing}); BoE parity not run"
        )
    return rai_bvar


def _gamma_prior(rb, **kwargs):
    gamma = getattr(rb, "GammaPrior", None)
    if gamma is None:
        from rai_bvar.priors import GammaPrior as gamma
    return gamma(**kwargs)


def _boe_hyperpriors(rb):
    """GLP defaults with BoE's lag-decay hyperprior Gamma(mode 2, sd 0.5) and wide bounds."""
    return rb.GLPHyperpriors(alpha=_gamma_prior(rb, mode=2.0, sd=0.5, lower=0.1, upper=10.0))


def _fit(rb, case: Case, frame: pd.DataFrame, psi: np.ndarray, **prior_kwargs):
    kwargs = {
        "lambda_": case.lam,
        "alpha": case.alpha,
        "psi": tuple(float(v) for v in psi),
        "soc": case.soc,
        "sur": case.sur,
        "soc_scope": "all",  # BoE bvar builds SOC rows for every variable (GLP)
        "mean": case.prior_mean,
        "var_const": case.var_const,
        "hierarchy": "fixed",
        **prior_kwargs,
    }
    model = rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(**kwargs)),
        vol=rb.Constant(prior="iw"),
        lags=case.lags,
        deterministic="c",
    )
    with warnings.catch_warnings():
        # prior mean 0 on tcode-1 data (the stationary simulated set) is deliberate here
        warnings.filterwarnings("ignore", message="(?i).*mean", category=UserWarning)
        return model.fit(rb.VARData.from_frame(frame, tcodes=1))


def _hyper(fit, name: str) -> float:
    for key in (name, f"{name}_"):
        if key in fit.hyper:
            return float(np.asarray(fit.hyper[key]))
    raise KeyError(f"fit.hyper has no {name!r}: {sorted(fit.hyper)}")


def _at(fit, case: Case) -> dict:
    """``fit.hyper`` with the hyperparameters set to the case values."""
    values = {"lambda": case.lam, "alpha": case.alpha, "soc": case.soc, "sur": case.sur}
    return {**fit.hyper, **{k: v for k, v in values.items() if v is not None}}


# ======================================================================= oracle self-checks


def test_boe_version_is_mapped():
    """Mapping in BOE_BVAR_NOTES.md was derived from this version; re-audit when it changes."""
    if bvar.__version__ != BOE_VERSION_MAPPED:
        pytest.xfail(f"bvar {bvar.__version__} != {BOE_VERSION_MAPPED}: re-check BOE_BVAR_NOTES.md")


@pytest.mark.parametrize("lags", [2, 4])
def test_boe_regressor_layout(frames, lags):
    """BoE rows are ``[1, y_{t-1}', …, y_{t-p}']``: constant first, lag blocks in variable order."""
    data = frames["sim"].to_numpy()
    Y, Z = construct_Y_Z(data, lags, NO_COVID)
    T = data.shape[0] - lags
    expected = np.hstack(
        [np.ones((T, 1))] + [data[lags - lag : lags - lag + T] for lag in range(1, lags + 1)]
    )
    np.testing.assert_array_equal(Y, data[lags:])
    np.testing.assert_array_equal(Z, expected)


def test_boe_psi_is_ar1_mse(frames):
    """BoE's ψ: mean squared residual of a per-variable AR(1) with constant over all T−1 pairs."""
    data = frames["qd"].to_numpy()
    expected = []
    for j in range(data.shape[1]):
        X = np.column_stack([np.ones(len(data) - 1), data[:-1, j]])
        resid = data[1:, j] - X @ np.linalg.lstsq(X, data[1:, j], rcond=None)[0]
        expected.append(np.mean(resid**2))
    np.testing.assert_allclose(ar1_mse(data, NO_COVID), expected, rtol=1e-10)


@pytest.mark.parametrize("case", [FIXED_CASES[1], FIXED_CASES[6]], ids=lambda c: c.id)
def test_boe_minnesota_prior(frames, case):
    """Ω_ii = var_const for the constant and λ² / (l^α · ψ_j / (d − n − 1)) for lag l of
    variable j; prior mean 1 (levels) or 0 on the own first lag. Same as GLP and rai-bvar."""
    data = frames[case.dataset].to_numpy()
    n, p = data.shape[1], case.lags
    model = _boe_model(case)
    model.pars.nu_0 = n + 4  # d ≠ n + 2 to expose the ψ / (d − n − 1) scaling
    psi = np.array([0.7, 1.3, 2.1])
    model.pars.S_0 = np.diag(psi)
    levels = np.full(n, case.levels)
    beta_0, precision = prior_minnesota(data, p, NO_COVID, levels, model.pars)
    lag = np.repeat(np.arange(1, p + 1), n)
    expected = np.concatenate(
        [[case.var_const], case.lam**2 / (lag**case.alpha * np.tile(psi / 3.0, p))]
    )
    np.testing.assert_allclose(1.0 / np.diag(precision), expected, rtol=1e-12)
    assert np.count_nonzero(precision - np.diag(np.diag(precision))) == 0
    mean = beta_0.reshape(n, -1).T
    np.testing.assert_array_equal(mean[1 : 1 + n], case.prior_mean * np.eye(n))
    assert not mean[0].any() and not mean[1 + n :].any()


def test_boe_dummy_initial_mean_convention(frames):
    """Pins BoE's SOC/SUR convention: ȳ_0 = mean of observations p+1..2p (GLP: 1..p)."""
    data, p, mu, delta = frames["qd"].to_numpy(), 4, 0.7, 1.9
    n = data.shape[1]
    model = bvar.NaturalConjugate(soc=True, sur=True)
    model.set_priors(mu=mu, theta=delta)
    Y, Z = construct_Y_Z(data, p, NO_COVID)
    Ys, Zs, nd = boe_dummies.stack_dummies(Y, Z, p, np.ones(n), model, NO_COVID, soc=True, sur=True)
    assert nd == n + 1
    ybar_boe, ybar_glp = data[p : 2 * p].mean(0), data[:p].mean(0)
    # row 0: SUR; rows 1..n: SOC; then the data
    np.testing.assert_allclose(Ys[0], ybar_boe / delta, rtol=1e-14)
    np.testing.assert_allclose(
        Zs[0], np.concatenate([[1 / delta], np.tile(ybar_boe / delta, p)]), rtol=1e-14
    )
    np.testing.assert_allclose(Ys[1 : n + 1], np.diag(ybar_boe) / mu, rtol=1e-14)
    np.testing.assert_allclose(
        Zs[1 : n + 1], np.hstack([np.zeros((n, 1)), np.tile(np.diag(ybar_boe) / mu, p)]), rtol=1e-14
    )
    np.testing.assert_array_equal(Ys[n + 1 :], Y)
    assert not np.allclose(ybar_boe, ybar_glp, rtol=1e-3)  # the difference is material


def _glp_log_ml(Y, Z, A0, precision, psi, d):
    """Independent GLP (2015, App. A) closed form, via QR of the prior-augmented design."""
    T, n = Y.shape
    s = np.sqrt(precision)
    Za, Ya = np.vstack([np.diag(s), Z]), np.vstack([s[:, None] * A0, Y])
    Q, R = np.linalg.qr(Za)
    E = Ya - Za @ np.linalg.solve(R, Q.T @ Ya)
    return (
        -n * T / 2 * np.log(np.pi)
        + multigammaln((T + d) / 2, n)
        - multigammaln(d / 2, n)
        - n / 2 * (2 * np.sum(np.log(np.abs(np.diag(R)))) - np.sum(np.log(precision)))
        + d / 2 * np.sum(np.log(psi))
        - (T + d) / 2 * np.linalg.slogdet(np.diag(psi) + E.T @ E)[1]
    )


@pytest.mark.parametrize(
    "case", [FIXED_CASES[3], FIXED_CASES[4], FIXED_CASES[6]], ids=lambda c: c.id
)
def test_boe_log_ml_is_glp_formula(frames, monkeypatch, case):
    """BoE's log-ML is log p(Y | h) exactly (incl. −nT/2·log π; dummy-only ML subtracted):
    the 'documented constant' between BoE and rai-bvar is zero."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    objective = BoEObjective(case, frame, add_priors=False)
    n = objective.data.shape[1]
    beta_0, precision = prior_minnesota(
        objective.data, case.lags, NO_COVID, objective.levels, objective.model.pars
    )
    A0, prec = beta_0.reshape(n, -1).T, np.diag(precision)
    Y, Z, nd = boe_dummies.stack_dummies(
        objective.Y, objective.Z, case.lags, objective.levels, objective.model, NO_COVID,
        soc=case.soc is not None, sur=case.sur is not None,
    )  # fmt: skip
    expected = _glp_log_ml(Y, Z, A0, prec, objective.psi, n + 2)
    if nd:
        expected -= _glp_log_ml(Y[:nd], Z[:nd], A0, prec, objective.psi, n + 2)
    np.testing.assert_allclose(-objective(), expected, rtol=1e-9)


# ====================================================================== rai-bvar vs BoE


@pytest.mark.parametrize("case", FIXED_CASES, ids=_ids(FIXED_CASES))
def test_parity_log_ml_fixed_hyperparameters(rb, frames, monkeypatch, case):
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = BoEObjective(case, frame, add_priors=False)
    fit = _fit(rb, case, frame, boe.psi)
    np.testing.assert_allclose(np.asarray(fit.psi), boe.psi, rtol=1e-15)
    np.testing.assert_allclose(fit.log_ml(), -boe(), rtol=RTOL_LOGML)


@pytest.mark.parametrize("case", FIXED_CASES, ids=_ids(FIXED_CASES))
def test_parity_posterior_moments(rb, frames, monkeypatch, case):
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = _boe_posterior(case, frame)
    post = _fit(rb, case, frame, boe.psi).posterior()
    n = boe.scale.shape[0]

    # B̄: BoE solves the normal equations (cond ≈ 1e10 on level data, elementwise error up to
    # ~3e-6 on near-zero entries, ~2e-9 per column), so near-zero entries get a column-scaled
    # absolute tolerance.
    col_scale = np.max(np.abs(boe.mean), axis=0, keepdims=True)
    mean = np.asarray(post.mean)
    assert mean.shape == boe.mean.shape
    excess = np.abs(mean - boe.mean) / (RTOL_POSTERIOR * (np.abs(boe.mean) + col_scale))
    assert np.max(excess) <= 1.0, (
        f"B̄ off by {np.max(excess):.2f}× tolerance at {np.unravel_index(np.argmax(excess), excess.shape)}"
    )
    # S̄ (residual form from BoE's B̄: first-order insensitive to BoE's B̄ rounding) and ν̄
    np.testing.assert_allclose(post.scale, boe.scale, rtol=RTOL_POSTERIOR)
    assert float(post.df) == boe.df
    # E[Σ | Y] = S̄ / (ν̄ − n − 1) against BoE's own point estimate, on the correlation scale
    sd_sigma = np.sqrt(np.diag(boe.sigma_point))
    gap = np.abs(post.scale / (post.df - n - 1) - boe.sigma_point) / np.outer(sd_sigma, sd_sigma)
    assert np.max(gap) <= CORR_TOL_BOE_SIGMA_POINT, np.max(gap)
    # Ω̄ on the correlation scale |ΔΩ_ij| ≤ tol·sqrt(Ω_ii Ω_jj)
    omega = np.asarray(post.omega())
    sd = np.sqrt(np.diag(boe.omega))
    assert np.all(np.abs(omega - boe.omega) <= RTOL_POSTERIOR * np.outer(sd, sd))


@pytest.mark.parametrize("case", [FIXED_CASES[3], FIXED_CASES[6]], ids=lambda c: c.id)
def test_parity_parameter_draws(rb, frames, monkeypatch, case):
    """Our iid NIW draws reproduce BoE's posterior means of B and Σ within Monte Carlo error."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = _boe_posterior(case, frame)
    fit = _fit(rb, case, frame, boe.psi)
    coefs, chol = fit.draw_parameters(4000, np.random.default_rng(20261021))
    sigma = chol @ np.swapaxes(chol, -1, -2)
    for draws, target in ((coefs, boe.mean), (sigma, boe.sigma_point)):
        mcse = draws.std(axis=0, ddof=1) / np.sqrt(draws.shape[0])
        z = (draws.mean(axis=0) - target) / mcse
        assert np.max(np.abs(z)) < Z_MC, np.max(np.abs(z))


@pytest.mark.parametrize("case", [FIXED_CASES[3], FIXED_CASES[6]], ids=lambda c: c.id)
def test_parity_log_posterior(rb, frames, monkeypatch, case):
    """log p(Y | h) + log p(h) with BoE's hyperpriors: Gamma on λ (0.2, 0.4), α (2, 0.5),
    μ (1, 1), δ (1, 1), normalised, on the natural scale, no ψ prior."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = BoEObjective(case, frame, add_priors=True)
    free = tuple(boe.order)
    fit = _fit(
        rb, case, frame, boe.psi, hierarchy="eb", hyper=free, hyperpriors=_boe_hyperpriors(rb)
    )
    at = _at(fit, case)
    np.testing.assert_allclose(fit.log_posterior(hyper=at), -boe(), rtol=RTOL_LOGML)
    np.testing.assert_allclose(
        fit.log_ml(hyper=at), -BoEObjective(case, frame, add_priors=False)(), rtol=RTOL_LOGML
    )


@pytest.mark.parametrize("case", EB_CASES, ids=_ids(EB_CASES))
def test_parity_eb_glp_default(rb, frames, monkeypatch, case):
    """GLP baseline (α fixed at 2): mode of λ (and μ, δ) vs the mode of BoE's objective with c3
    held at 2. BoE's tune_priors cannot hold c3 fixed, so its objective is maximised here."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = BoEObjective(case, frame, add_priors=True)
    free = [h for h in boe.order if h != "alpha"]
    mode, best = boe.mode(free, start={"lambda": 0.2, "soc": 1.0, "sur": 1.0})
    fit = _fit(rb, case, frame, boe.psi, hierarchy="eb", hyper="auto")
    ours = {h: _hyper(fit, h) for h in free}
    assert _hyper(fit, "alpha") == case.alpha
    for h in free:
        np.testing.assert_allclose(ours[h], mode[h], rtol=RTOL_HYPER, err_msg=h)
    assert boe(ours) <= best + ATOL_OBJECTIVE


@pytest.mark.parametrize("case", EB_CASES, ids=_ids(EB_CASES))
def test_parity_eb_boe_default(rb, frames, monkeypatch, case):
    """BoE's default ML optimisation: (λ, α[, μ, δ]) with α ~ Gamma(2, 0.5). BoE's end-to-end
    optimiser (numpy backend) is run, then its result is polished on BoE's own objective."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    boe = BoEObjective(case, frame, add_priors=True)
    free = boe.order
    model = _boe_bvar(case, method="ml")
    start = np.array([case.lam, case.alpha] + [1.0] * (len(free) - 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.optimise_hyperparameters(frame, initial_values=start, optimisation_backend="numpy")
    raw = {h: float(getattr(model.model.pars, BOE_NAME[h])) for h in free}
    mode, best = boe.mode(free, start=raw)
    for h in free:  # BoE's own optimiser is only accurate to a few per cent on level data
        np.testing.assert_allclose(raw[h], mode[h], rtol=RTOL_BOE_OPTIMISER, err_msg=f"BoE {h}")
    fit = _fit(
        rb,
        case,
        frame,
        boe.psi,
        hierarchy="eb",
        hyper=tuple(free),
        hyperpriors=_boe_hyperpriors(rb),
    )
    ours = {h: _hyper(fit, h) for h in free}
    for h in free:
        np.testing.assert_allclose(ours[h], mode[h], rtol=RTOL_HYPER, err_msg=h)
    assert boe(ours) <= best + ATOL_OBJECTIVE


QUANTILES = np.array([0.05, 0.16, 0.5, 0.84, 0.95])
HORIZON, N_PATHS = 8, 5000


@pytest.mark.parametrize("case", FORECAST_CASES, ids=_ids(FORECAST_CASES))
def test_parity_forecast_quantiles(rb, frames, monkeypatch, case):
    """Unconditional predictive quantiles (one Gaussian path per iid posterior draw in both)."""
    frame = frames[case.dataset]
    if case.dummies:
        _use_glp_initial_mean(monkeypatch, frame.to_numpy(), case.lags)
    model = _boe_bvar(case, seed=11)
    model.sample(frame, N_draws=N_PATHS, progressbar=False)
    model.forecast(H=HORIZON, N_draws=N_PATHS)
    boe_paths = model.forecast_unconditional[:, -HORIZON:, :]
    psi = np.diag(model.model.pars.S_0)

    result = _fit(rb, case, frame, psi).forecast(HORIZON, draws=N_PATHS, seed=12)
    ours = (
        result.quantiles(list(QUANTILES))
        .sel(variable=list(frame.columns))
        .transpose("quantile", "horizon", "variable")
        .to_numpy()
    )
    boe_q = np.quantile(boe_paths, QUANTILES, axis=0)
    # MC standard error of a difference of sample quantiles; 1/density by Siddiqui's estimator
    d = 0.025
    inv_density = (
        np.quantile(boe_paths, QUANTILES + d, axis=0)
        - np.quantile(boe_paths, QUANTILES - d, axis=0)
    ) / (2 * d)
    n_ours = int(getattr(result, "n_samples", N_PATHS))
    se = (
        np.sqrt(QUANTILES * (1 - QUANTILES))[:, None, None]
        * inv_density
        * np.sqrt(1 / N_PATHS + 1 / n_ours)
    )
    z = (ours - boe_q) / se
    worst = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    assert np.max(np.abs(z)) < Z_MC, (
        f"|z| = {np.abs(z)[worst]:.2f} at (quantile, horizon, variable) {worst}"
    )
