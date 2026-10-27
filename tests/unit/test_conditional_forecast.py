"""Conditional forecasts with hard conditions (P3 DESIGN §6, D-50; WP34): A10, A10b and the
contract checks (§6.1, §6.3, §9.7, §9.11; PR tier).

**A10** (§9.7, E144): every ``(chain, draw)`` slice of a conditional forecast against WP30's
``validation.reference.conditional_forecast_reference``, an independent stacked companion-form
oracle with a dense Schur complement, on four legs: the analytic engine (``AnalyticFit``, constant
``Σ``), the structural NIG fit (``StructuralNIGFit``, the analytic engine's other fit), and MCMC
draws under ``SV`` (with outliers, so the scale path varies with ``h``) and ``CommonSV``. The
oracle gets inputs computed without the update's code: the lag matrices and the zero-shock path
``μ`` from the slice's coefficient draw ``B`` (``fit.draw_parameters`` replayed with the forecast
seed for the analytic engine; ``fit.reduced_form_draws`` with the forecast's draw selection for
MCMC) through ``zero_shock_path``, the shock factors ``L_k = cond_chol_k`` and ``ε_k =
L_k⁻¹(paths_k − cond_mean_k)`` of the unconditional forecast. Every leg conditions a variable
ordered before a free one (E32). Exact rule (1e-10): ``Y = μ + Mε``, ``M``, ``Y_c``, ``E_c``, the
per-horizon ``L^c L^cᵀ``, the joint ``Cov_c`` and ``log w``. Statistical rule: the sample moments
of the conditional paths against the oracle's ``E_c`` and ``Cov_c``, pooled over draws (each
slice is one draw from its own Gaussian given its parameters), ``|z| ≤ max(4, z_{1−0.01/(2m)})``
over the ``m`` compared moments (D-60 as read by E152).

**A10b** (§9.7, E39): the Bank of England ``bvar`` 0.3.5 (MIT) ``draw_constrained_forecasts`` at
one fixed ``(beta, sigma)`` against our update on the same parameter draw: its point output (the
conditional mean) at 1e-10, the covariance of 10⁴ of its draws against our ``Cov_c`` by the same
``|z|`` rule, and (beyond the row) its draw from given shocks against ours from the same shocks.

**Contract** (§6.1, §6.3): the result's fields, dropped and added variables and attrs; the
structure and per-horizon KS checks; the refusals of ``score_forecast`` and the backtest; the
``log_pred_density`` rules; the E-16 errors; determinism; the ``_update`` seam (mutant M33).

Tolerance reading of "1e-10" (an open choice of WP34): ``|ours − oracle| ≤ 1e-10·(|oracle| +
s)`` per entry, with ``s`` the cell's unconditional standard deviation for paths and means,
``s_i s_j`` for covariances and 1 for log weights, so a cell is held to its own scale.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

import rai_bvar as rb
from rai_bvar.data import VARData
from rai_bvar.forecast import conditional as cf
from rai_bvar.forecast.invariants import (
    ContractError,
    check_conditional_structure,
    check_structure,
    conditional_simulation_consistency,
)
from rai_bvar.forecast.result import LAG_DIMS, ForecastResult
from rai_bvar.forecast.simulate import simulate_linear_var
from rai_bvar.linalg import mvn_logpdf_chol
from rai_bvar.validation.reference import conditional_forecast_reference, zero_shock_path

TOL = 1e-10  # A10 / A10b exact rule (§9.7)
DRAWS = 1000
ANALYTIC_DRAWS = 10_000  # the analytic legs, as A10b (owner Q43, E174: more draws, same seed)
SEED = 20261026
KS_ALPHA = 1e-3  # §6.3: family-wise over the horizons (D-60, E124)


def z_bound(m: int) -> float:
    """``|z| ≤ 4`` with Bonferroni over ``m`` compared moments (D-60; E152's reading)."""
    return max(4.0, float(stats.norm.isf(0.01 / (2 * m))))


def _fits_module():
    """``tests/mcmc/forecast/mcmc_fits.py`` as ``mcmc_fits`` (shared with the fast-tier forecast
    tests, so the SV fit is compiled once per process)."""
    if "mcmc_fits" in sys.modules:
        return sys.modules["mcmc_fits"]
    path = Path(__file__).resolve().parents[1] / "mcmc" / "forecast" / "mcmc_fits.py"
    spec = importlib.util.spec_from_file_location("mcmc_fits", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcmc_fits"] = module
    spec.loader.exec_module(module)
    return module


def _commonsv_data(T: int = 90, n: int = 3, seed: int = 5) -> VARData:
    """The dataset of ``tests/mcmc/blocks/test_common_sv.py::fits`` (same static configuration, so
    the CommonSV runner can be shared in-process)."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T, n))
    h = 0.0
    for t in range(1, T):
        h = 0.9 * h + 0.3 * rng.standard_normal()
        y[t] = 0.5 * y[t - 1] + np.exp(0.5 * h) * rng.standard_normal(n) * np.linspace(0.5, 2.0, n)
    df = pd.DataFrame(y, index=pd.period_range("1990Q1", periods=T, freq="Q"),
                      columns=[f"y{j}" for j in range(n)])  # fmt: skip
    return VARData.from_frame(df, dict.fromkeys(df.columns, 1))


# --------------------------------------------------------------------------- legs


class Leg(NamedTuple):
    name: str
    fit: Any
    base: ForecastResult  # the unconditional forecast (same horizon, draws, seed)
    result: cf.ConditionalForecastResult
    conditions: dict[str, list[float]]
    cells: list[tuple[int, int, float]]  # (h, j, r)
    coefs: np.ndarray  # (S, k, n): B of every stacked slice
    hist: np.ndarray  # (p, n)
    n_det: int
    draws: int  # the forecast's draws (S)


def _cells(conditions: dict, variables: list[str]) -> list[tuple[int, int, float]]:
    out = []
    for var, seq in conditions.items():
        for h, value in enumerate(seq, start=1):
            if np.isfinite(value):
                out.append((h, variables.index(var), float(value)))
    return out


def _mcmc_slot_coefs(fit, draws: int) -> np.ndarray:
    """``B`` of every stacked slice of ``fit.forecast(·, draws, ·)`` from ``fit.reduced_form_draws``
    and the forecast's selection (DESIGN §7.1: slot ``j·R + r`` of chain ``c`` is stored draw
    ``pos[j]``)."""
    from rai_bvar.engines.mcmc.forecast import select_draws

    B = np.asarray(fit.reduced_form_draws().values)  # (C, D, k, n)
    C, D = B.shape[:2]
    pos, R = select_draws(D, C, draws)
    slots = np.repeat(pos, R)
    return B[:, slots].reshape(-1, *B.shape[2:])


LEGS = ("analytic", "structural-nig", "mcmc-sv", "mcmc-commonsv")


@pytest.fixture(scope="module", params=LEGS)
def leg(request, sim_data, sim_data_stationary) -> Leg:
    name = request.param
    draws = ANALYTIC_DRAWS if name in ("analytic", "structural-nig") else DRAWS
    if name in ("analytic", "structural-nig"):
        data = sim_data_stationary
        prior = (
            rb.Minnesota(hierarchy="fixed")
            if name == "analytic"
            else rb.AsymmetricConjugate(hierarchy="fixed", cross=0.5)
        )
        fit = rb.BVAR(mean=rb.Linear(prior), lags=2).fit(data)
        H, conditions = 4, {"gdp": [0.004, 0.006], "rate": [np.nan, np.nan, 4.3]}
        coefs = fit.draw_parameters(draws, rng=SEED)[0]  # the forecast's own parameter stream
    elif name == "mcmc-sv":
        data = sim_data
        fit = _fits_module().shared_fit(data)  # structural Mode I × SV(outliers), lags 1
        H, conditions = 6, {"gdp": [0.004, 0.006], "rate": [np.nan, np.nan, 4.3]}
        coefs = _mcmc_slot_coefs(fit, draws)
    else:
        data = _commonsv_data()
        settings = rb.MCMCSettings(chains=2, draws=40, burn=40, store="paths")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = rb.BVAR(vol=rb.CommonSV(), lags=2, mcmc=settings).fit(data)
        H, conditions = 4, {"y0": [0.5, 0.2], "y2": [np.nan, np.nan, -0.3]}
        coefs = _mcmc_slot_coefs(fit, draws)
    base = fit.forecast(H, draws, SEED, lag_coefs=True)
    result = fit.conditional_forecast(H, conditions, draws=draws, seed=SEED)
    variables = list(data.variables)
    lags = fit.model.lags
    return Leg(name, fit, base, result, conditions, _cells(conditions, variables), coefs,
               data.values[-lags:], 1, draws)  # fmt: skip


class Oracle(NamedTuple):
    lag: np.ndarray  # (S, p, n, n) A_l from B
    mu: np.ndarray  # (S, H, n) zero-shock path from B
    eps: np.ndarray  # (S, H, n)
    M: np.ndarray  # (S, nH, nH)
    mean: np.ndarray  # (S, H, n) E_c
    cov: np.ndarray  # (S, nH, nH) Cov_c
    path: np.ndarray  # (S, H, n) Y_c
    log_weight: np.ndarray  # (S,)
    sd: np.ndarray  # (S, H, n) unconditional standard deviations


@pytest.fixture(scope="module")
def oracle(leg) -> Oracle:
    base = leg.base
    H, n = base.horizon, len(base.variables)
    paths, mean, chol = base.stacked("paths"), base.stacked("cond_mean"), base.stacked("cond_chol")
    S = paths.shape[0]
    p = leg.hist.shape[0]
    out = {k: [] for k in Oracle._fields}
    for s in range(S):
        B = leg.coefs[s]
        lag = B[leg.n_det :].reshape(p, n, n).swapaxes(-1, -2)
        mu = zero_shock_path(B, leg.hist, H, n_det=leg.n_det)
        eps = solve_triangular(chol[s], (paths[s] - mean[s])[..., None], lower=True)[..., 0]
        ref = conditional_forecast_reference(lag, chol[s], mu, leg.cells, eps=eps)
        for key, value in (
            ("lag", lag), ("mu", mu), ("eps", eps), ("M", ref.M), ("mean", ref.mean),
            ("cov", ref.cov), ("path", ref.path), ("log_weight", ref.log_weight),
            ("sd", np.sqrt(np.diag(ref.uncond_cov)).reshape(H, n)),
        ):  # fmt: skip
            out[key].append(value)
    return Oracle(**{k: np.asarray(v) for k, v in out.items()})


def assert_close(actual, expected, scale, what: str, tol: float = TOL) -> float:
    """``|actual − expected| ≤ tol·(|expected| + scale)`` entrywise (module docstring)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    err = np.abs(actual - expected) / (np.abs(expected) + scale)
    worst = float(err.max())
    assert worst <= tol, f"{what}: relative error {worst:.3g} > {tol:g}"
    return worst


def _ours(leg: Leg, *, full_cov: bool = True) -> cf.ConditionalDraws:
    """Our update on the base forecast's stacked draws, with the joint ``Cov_c``."""
    base = leg.base
    paths, mean = base.stacked("paths"), base.stacked("cond_mean")
    phi = cf.ma_matrices(base.stacked("lag_coefs"), base.horizon)
    mu = cf.zero_shock_mean(paths, mean, phi)
    cond = cf.parse_conditions(leg.conditions, base.variables, base.horizon)
    M = cf.impact_matrix(phi, base.stacked("cond_chol"))
    return cf.exact_update(mu, M, paths, cond.index, cond.values, full_cov=full_cov)


# --------------------------------------------------------------------------- A10


def test_a10_inputs_and_the_stacked_representation(leg, oracle):
    """The oracle's inputs line up with the forecast (``lag_coefs`` from ``B``), and ``Y = μ +
    Mε`` exactly (§6.2), with our ``M`` and ``μ`` equal to the companion-form ``M`` and the
    zero-shock path of ``B``."""
    base = leg.base
    np.testing.assert_allclose(base.stacked("lag_coefs"), oracle.lag, rtol=1e-12, atol=1e-13)
    sd = oracle.sd.reshape(len(oracle.sd), -1)
    stacked = oracle.mu.reshape(len(sd), -1) + np.einsum(
        "sij,sj->si", oracle.M, oracle.eps.reshape(len(sd), -1)
    )
    assert_close(stacked, base.stacked("paths").reshape(len(sd), -1), sd, "Y = μ + Mε")
    phi = cf.ma_matrices(base.stacked("lag_coefs"), base.horizon)
    mu = cf.zero_shock_mean(base.stacked("paths"), base.stacked("cond_mean"), phi)
    assert_close(mu, oracle.mu, oracle.sd, "μ")
    M = cf.impact_matrix(phi, base.stacked("cond_chol"))
    assert_close(M, oracle.M, sd[:, :, None], "M")


def test_a10_update_matches_the_oracle(leg, oracle):
    """A10 exact part: ``Y_c``, ``E_c``, ``L^c_h L^c_hᵀ`` (the full ``n × n`` block of horizon h),
    the joint ``Cov_c`` and ``log w`` of every slice at 1e-10."""
    res = leg.result
    H, n = res.horizon, len(res.variables)
    sd = oracle.sd.reshape(len(oracle.sd), -1)
    sd2 = sd[:, :, None] * sd[:, None, :]
    assert_close(res.stacked("paths"), oracle.path, oracle.sd, "Y_c")
    assert_close(res.stacked("cond_mean"), oracle.mean, oracle.sd, "E_c")
    L = res.stacked("cond_chol")
    for h in range(H):
        block = oracle.cov[:, h * n : (h + 1) * n, h * n : (h + 1) * n]
        scale = sd2[:, h * n : (h + 1) * n, h * n : (h + 1) * n]
        assert_close(L[:, h] @ np.swapaxes(L[:, h], 1, 2), block, scale, f"Cov_c,{h + 1}")
    assert_close(_ours(leg).cov, oracle.cov, sd2, "joint Cov_c")
    assert_close(res.stacked("log_weight"), oracle.log_weight, 1.0, "log w")
    # the conditioned cells are r exactly, in paths and in cond_mean
    for h, j, r in leg.cells:
        assert np.all(res.paths.values[:, :, h - 1, j] == r)
        assert np.all(res.cond_mean.values[:, :, h - 1, j] == r)


# History (DESIGN P3 E174, Q43): at the seed fixed before the first run and DRAWS = 1000, the two
# analytic-engine legs failed the statistical part on one moment (max |z| = 4.11 and 4.35 > 4.00;
# the legs share the forecast's shocks). Over 400 other seeds the rule's family-wise failure rate
# at 1000 draws was 1.0% against ≈ 0.34% nominal. The owner decided that the analytic legs run at
# ANALYTIC_DRAWS = 10⁴ with the same seed, statistic and bound, the outcome binding: a leg that
# fails at 10⁴ is not met for this check, with no further change.
MOMENTS_XFAIL: dict[str, float] = {}


def test_a10_sample_moments_of_the_conditional_paths(request, leg, oracle):
    """A10 statistical part: given its parameters, slice ``s`` is one draw of ``N(E_c,s,
    Cov_c,s)``, so ``z_s = (Y_c,s − E_c,s)/σ_s`` (free cells, all horizons) has mean 0 and
    correlation ``ρ_s``; the pooled first moments (known variance 1) and the products
    ``z_i z_j − ρ_ij`` (sample s.e.) are compared with ``|z| ≤ max(4, z_{1−0.01/(2m)})``."""
    if leg.name in MOMENTS_XFAIL:
        request.applymarker(pytest.mark.xfail(strict=True, reason=(
            f"not met (Q43, E174): max |z| = {MOMENTS_XFAIL[leg.name]} > 4 at 10⁴ draws and the "
            "pre-registered seed (module comment above)"
        )))  # fmt: skip
    res = leg.result
    S = res.n_samples
    free = ~res.conditioned.values.reshape(-1)
    d = (res.stacked("paths") - oracle.mean).reshape(S, -1)[:, free]
    cov = oracle.cov[:, free][:, :, free]
    sig = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    z = d / sig
    rho = cov / (sig[:, :, None] * sig[:, None, :])
    first = z.mean(axis=0) * np.sqrt(S)
    iu = np.triu_indices(z.shape[1])
    w = z[:, iu[0]] * z[:, iu[1]] - rho[:, iu[0], iu[1]]
    second = w.mean(axis=0) / (w.std(axis=0, ddof=1) / np.sqrt(S))
    stat = np.abs(np.concatenate([first, second]))
    bound = z_bound(stat.size)
    assert stat.max() <= bound, f"{leg.name}: max |z| = {stat.max():.2f} > {bound:.2f}"


@pytest.mark.parametrize("leg", ["analytic"], indirect=True)
def test_a10_mutant_m33_through_the_update_seam(leg, oracle):
    """The keyword-only ``_update`` seam (E63) replaces the update; with ``Mᵀ`` where ``M``
    belongs (mutant M33) the oracle comparison of A10 fails."""

    def m33(mu, M, Y, index, values, **kw):
        return cf.exact_update(mu, np.swapaxes(M, 1, 2), Y, index, values, **kw)

    bad = rb.conditional_forecast(leg.fit, leg.base.horizon, leg.conditions, leg.draws, SEED,
                                  _update=m33)  # fmt: skip
    with pytest.raises(AssertionError, match="E_c"):
        assert_close(bad.stacked("cond_mean"), oracle.mean, oracle.sd, "E_c")


# --------------------------------------------------------------------------- contract


def test_conditional_structure_and_simulation_consistency(leg):
    """§6.3: ``check_conditional_structure`` passes and the per-horizon KS of the standardised
    free residuals passes as one family over the horizons (α = 1e-3, D-60); the Phase 1
    ``check_structure`` does not apply (the conditioned diagonal is 0)."""
    res = leg.result
    check_conditional_structure(res)
    p = conditional_simulation_consistency(res)
    finite = p[np.isfinite(p)]
    assert finite.size == res.horizon  # every horizon here has a free variable
    assert finite.min() >= KS_ALPHA / finite.size, p
    with pytest.raises(ContractError, match="positive diagonal"):
        check_structure(res)


def test_result_fields_attrs_and_dropped_variables(leg):
    """§6.3: the contract fields with the base's coordinates; ``condition``, ``conditioned``,
    ``log_weight`` added; the unconditional accelerator variables dropped (E33); the base's
    attrs (``seed`` is the base forecast's) plus ``conditional`` and the Kish ESS of the weights;
    ``paths_level`` the pathwise inverse of ``Y_c``."""
    res, base = leg.result, leg.base
    ds = res.dataset
    assert isinstance(res, cf.ConditionalForecastResult) and isinstance(res, ForecastResult)
    for name in cf.DROPPED:
        assert name not in ds, name
    if leg.name == "mcmc-sv":  # every accelerator variable an MCMC forecast can carry
        assert {"cond_A", "cond_scale", "cond_scale_persistent", "log_p", "log1m_p",
                "lag_coefs"} <= set(base.dataset.data_vars)  # fmt: skip
    for name in ("chain", "draw", "horizon", "variable", "variable_", "date"):
        np.testing.assert_array_equal(ds[name].values, base.dataset[name].values)
    assert ds["condition"].dims == ("horizon", "variable")
    assert ds["conditioned"].dtype == bool and ds["log_weight"].dims == ("chain", "draw")
    table = np.full((res.horizon, len(res.variables)), np.nan)
    for h, j, r in leg.cells:
        table[h - 1, j] = r
    np.testing.assert_array_equal(ds["condition"].values, table)
    np.testing.assert_array_equal(ds["conditioned"].values, np.isfinite(table))
    attrs = ds.attrs
    assert attrs["conditional"] == "hard-per-draw-v1"
    for key, value in base.dataset.attrs.items():
        assert attrs[key] == value, key
    w = np.exp(res.stacked("log_weight") - res.stacked("log_weight").max())
    assert attrs["weight_ess"] == pytest.approx(w.sum() ** 2 / (w * w).sum(), rel=1e-12)
    np.testing.assert_allclose(
        ds["paths_level"].values, leg.fit.data.inverse_paths(res.paths.values), rtol=0, atol=0
    )
    assert res.quantiles([0.5]).dims == ("quantile", "horizon", "variable")
    assert "ConditionalForecastResult" in repr(res)


@pytest.mark.parametrize("leg", ["analytic", "mcmc-sv"], indirect=True)
def test_same_seed_reproduces_bitwise_and_the_entry_points_agree(leg):
    """No random number is drawn: ``rb.conditional_forecast`` (lazy, P3 §1.1) and the fit method
    give the same result, and conditioning a given ``base`` gives it too."""
    res = leg.result
    again = rb.conditional_forecast(leg.fit, res.horizon, leg.conditions, leg.draws, SEED)
    given = rb.conditional_forecast(leg.fit, res.horizon, leg.conditions, base=leg.base)
    for other in (again, given):
        for name in ("paths", "cond_mean", "cond_chol", "log_weight", "paths_level"):
            np.testing.assert_array_equal(other.dataset[name].values, res.dataset[name].values)


@pytest.mark.parametrize("leg", ["analytic"], indirect=True)
def test_base_from_another_sample_is_refused(leg):
    """A ``base`` whose origin is not the fit's last period is refused (its levels would be
    inverted against the wrong history; the R6 review of PR #28)."""
    shifted = ForecastResult(leg.base.dataset.assign_attrs(origin=str(leg.base.origin - 1)))
    with pytest.raises(ValueError, match="base forecasts from"):
        rb.conditional_forecast(leg.fit, leg.base.horizon, leg.conditions, base=shifted)


# --------------------------------------------------------------------------- A10b


@pytest.fixture(scope="module")
def boe_case(sim_data_stationary):
    """One fixed parameter draw ``(beta, sigma)`` of an analytic fit (lags 2, constant only:
    BoE's ``beta = [c, B_1 … B_p]`` layout is ours with ``deterministic="c"``), the last ``p``
    observations and the conditions of the analytic leg."""
    data = sim_data_stationary
    fit = rb.BVAR(mean=rb.Linear(rb.Minnesota(hierarchy="fixed")), lags=2).fit(data)
    coefs, chol = fit.draw_parameters(1, rng=SEED + 1)
    beta = coefs[0]
    sigma = chol[0] @ chol[0].T
    L = np.linalg.cholesky(sigma)  # BoE's A_0 = chol(Σ)ᵀ: the same factor on both sides
    H, n, p = 4, data.n_vars, 2
    table = np.full((H, n), np.nan)
    table[0, 0], table[1, 0], table[2, 2] = 0.004, 0.006, 4.3
    return SimpleNamespace(
        beta=beta,
        sigma=sigma,
        L=L,
        hist=data.values[-p:],
        H=H,
        n=n,
        p=p,
        table=table,
        t_last=len(data.values),
        variables=list(data.variables),
    )


def _our_update(case, shocks: np.ndarray) -> tuple[cf.ConditionalDraws, np.ndarray]:
    """Our update of paths of the fixed draw driven by ``shocks`` ``(S, H, n)``, and the
    unconditional standard deviation of every stacked cell (the tolerance scale)."""
    S = shocks.shape[0]
    coefs = np.broadcast_to(case.beta, (S, *case.beta.shape))
    chol = np.broadcast_to(case.L, (S, case.n, case.n))
    paths, mean, cond_chol = simulate_linear_var(coefs, chol, case.hist, case.H, "c", case.t_last,
                                                 np.random.default_rng(0), shocks=shocks)  # fmt: skip
    lag = np.broadcast_to(case.beta[1:].reshape(case.p, case.n, case.n).swapaxes(-1, -2),
                          (S, case.p, case.n, case.n))  # fmt: skip
    phi = cf.ma_matrices(lag, case.H)
    mu = cf.zero_shock_mean(paths, mean, phi)
    cond = cf.parse_conditions(
        {v: case.table[:, j] for j, v in enumerate(case.variables)}, case.variables, case.H
    )
    M = cf.impact_matrix(phi, np.asarray(cond_chol))
    sd = np.sqrt(np.einsum("ij,ij->i", M[0], M[0]))
    return cf.exact_update(mu, M, paths, cond.index, cond.values, full_cov=True), sd


def _boe(case, point_only: bool, rng: np.random.Generator) -> np.ndarray:
    from bvar.forecast.conditional import draw_constrained_forecasts, get_constraint

    C, f, Sigma_f, shape_f = get_constraint(case.table, None, None)
    return draw_constrained_forecasts(
        case.sigma, case.beta, C, f, Sigma_f, shape_f, case.hist, case.p, case.n, 0, case.H,
        point_only, None, "andersson_et_al", rng,
    )  # fmt: skip


def test_a10b_boe_conditional_mean(boe_case):
    """A10b: BoE's point output (``b + M'D⁺(f − Cb)``, the minimum-norm shocks) equals our
    ``E_c`` on the same parameter draw at 1e-10."""
    import bvar

    assert bvar.__version__ == "0.3.5"  # the version whose source the mapping was read from
    ours, sd = _our_update(boe_case, np.zeros((1, boe_case.H, boe_case.n)))
    boe = _boe(boe_case, True, np.random.default_rng(0))
    assert_close(ours.mean[0].reshape(-1), boe, sd, "BoE point vs E_c")


def test_a10b_boe_covariance_of_draws(boe_case):
    """A10b: the sample covariance of 10⁴ BoE draws (Andersson et al.'s shock adjustment with
    zero constraint variance, i.e. hard conditions) against our ``Cov_c`` on the free cells,
    ``|z| ≤ max(4, z_{1−0.01/(2m)})`` with the sample s.e. of each covariance entry."""
    rng = np.random.default_rng(SEED + 2)
    draws = np.array([_boe(boe_case, False, rng) for _ in range(10_000)])
    ours, _ = _our_update(boe_case, np.zeros((1, boe_case.H, boe_case.n)))
    free = np.isnan(boe_case.table).reshape(-1)
    x = draws[:, free] - draws[:, free].mean(axis=0)
    N = x.shape[0]
    iu = np.triu_indices(x.shape[1])
    prod = x[:, iu[0]] * x[:, iu[1]]
    sample = prod.sum(axis=0) / (N - 1)
    se = prod.std(axis=0, ddof=1) / np.sqrt(N)
    target = ours.cov[0][np.ix_(free, free)][iu]
    zs = np.abs(sample - target) / se
    bound = z_bound(zs.size)
    assert zs.max() <= bound, f"max |z| = {zs.max():.2f} > {bound:.2f}"
    # conditioned cells of every BoE draw sit on the conditions (to rounding)
    held = ~free
    target = np.broadcast_to(boe_case.table.reshape(-1)[held], draws[:, held].shape)
    np.testing.assert_allclose(draws[:, held], target, rtol=1e-12, atol=1e-14)


def test_a10b_boe_draws_from_the_same_shocks(boe_case):
    """Beyond the A10b row: BoE's draw ``b + M'(ξ + D⁺(z − Dξ))`` from shocks ``ξ`` (the first
    ``nH`` normals of the generator it is given) equals our ``Y_c`` from the same shocks."""
    nH = boe_case.H * boe_case.n
    for i in range(5):
        xi = np.random.default_rng(100 + i).standard_normal(nH)
        boe = _boe(boe_case, False, np.random.default_rng(100 + i))
        ours, sd = _our_update(boe_case, xi.reshape(1, boe_case.H, boe_case.n))
        assert_close(ours.paths[0].reshape(-1), boe, sd, f"draw {i}")


# --------------------------------------------------------------------------- contract (analytic)


@pytest.fixture(scope="module")
def small(sim_data_stationary):
    """A cheap analytic fit and its conditional forecast for the contract checks."""
    data = sim_data_stationary
    fit = rb.BVAR(mean=rb.Linear(rb.Minnesota(hierarchy="fixed")), lags=2).fit(data)
    conditions = {"gdp": [0.004, 0.006], "rate": [np.nan, np.nan, 4.3]}
    res = fit.conditional_forecast(4, conditions, draws=200, seed=3)
    return SimpleNamespace(fit=fit, data=data, conditions=conditions, res=res)


def test_score_forecast_and_the_backtest_refuse_a_conditional_forecast(small):
    """§6.3: a scenario is no forecast to score (``TypeError``), also when it is re-wrapped as a
    plain ``ForecastResult``; the backtest scores through ``score_forecast`` and refuses too."""
    realized = np.zeros((4, 3))
    with pytest.raises(TypeError, match="scenario"):
        rb.score_forecast(small.res, realized)
    with pytest.raises(TypeError, match="scenario"):
        rb.score_forecast(ForecastResult(small.res.dataset), realized)

    class Scenario:
        name, axes = "scenario", {}

        def spec(self):
            return {"class": "Scenario"}

        def fit(self, data):
            inner = rb.BVAR(mean=rb.Linear(rb.Minnesota(hierarchy="fixed")), lags=2).fit(data)

            def forecast(horizon, draws=1000, seed=None, **_):
                return inner.conditional_forecast(horizon, {"gdp": [0.0]}, draws, seed)

            return SimpleNamespace(forecast=forecast)

    origin = str(small.data.index[-6])
    with pytest.raises(TypeError, match="scenario"):
        rb.backtest(Scenario(), small.data, origins=[origin], horizon=2, draws=50)


def test_log_pred_density_rules(small):
    """§6.3: ``log_pred_density`` raises for a conditioned ``(horizon, variable)`` (also with
    ``variables=None``) and for ``lpd_method="integrated"``; for free variables it is the
    equal-weight mixture ``log (1/S) Σ_s N(y; E_c,s, Cov_c,s)`` (leading and other subsets)."""
    res = small.res
    with pytest.raises(ValueError, match="conditioned at horizon 1"):
        res.log_pred_density(np.zeros(1), 1, ["gdp"])
    with pytest.raises(ValueError, match="conditioned at horizon 3"):
        res.log_pred_density(np.zeros(3), 3)
    with pytest.raises(ValueError, match="integrated"):
        res.log_pred_density(np.zeros(1), 2, ["prices"], lpd_method="integrated")
    mean, chol = res.stacked("cond_mean"), res.stacked("cond_chol")
    y = np.array([0.005, 4.1])
    got = res.log_pred_density(y, 1, ["prices", "rate"])
    want = logsumexp(mvn_logpdf_chol(y, mean[:, 0, 1:], chol[:, 0, 1:, 1:])) - np.log(len(mean))
    assert got == pytest.approx(want, rel=1e-12)
    y4 = np.array([0.004, 0.005, 4.2])  # horizon 4 is free: every variable
    got4 = res.log_pred_density(y4, 4)
    want4 = logsumexp(mvn_logpdf_chol(y4, mean[:, 3], chol[:, 3])) - np.log(len(mean))
    assert got4 == pytest.approx(want4, rel=1e-12)
    # the leading free pair of horizon 3 (rate is conditioned there), given in another order
    y3 = np.array([0.004, 0.005])  # (prices, gdp)
    got3 = res.log_pred_density(y3, 3, ["prices", "gdp"])
    want3 = logsumexp(mvn_logpdf_chol(y3[::-1], mean[:, 2, :2], chol[:, 2, :2, :2]))
    assert got3 == pytest.approx(want3 - np.log(len(mean)), rel=1e-12)


def test_condition_forms_agree(small):
    """§6.1: a DataFrame by horizon, a DataFrame by forecast period and the mapping form give
    the same result bitwise; ``NaN`` padding of the mapping."""
    fit, res = small.fit, small.res
    by_h = pd.DataFrame({"gdp": [0.004, 0.006, np.nan], "rate": [np.nan, np.nan, 4.3]},
                        index=[1, 2, 3])  # fmt: skip
    origin = small.data.last
    by_p = by_h.set_axis(pd.PeriodIndex([origin + 1, origin + 2, origin + 3]))
    for conditions in (by_h, by_p):
        other = fit.conditional_forecast(4, conditions, draws=200, seed=3)
        np.testing.assert_array_equal(other.paths.values, res.paths.values)
        np.testing.assert_array_equal(other.cond_chol.values, res.cond_chol.values)


def test_e16_condition_errors(small):
    """E-16 (§6.1, at call): unknown variable, horizon or period outside the forecast, no
    condition, a non-finite condition, every cell conditioned, a sequence longer than the
    horizon, a duplicated horizon, a wrong type; the message names the culprit."""
    fit = small.fit

    def call(conditions, horizon=3):
        return fit.conditional_forecast(horizon, conditions, draws=10, seed=1)

    with pytest.raises(KeyError, match=r"unknown variables \['gnp'\].*'gdp', 'prices', 'rate'"):
        call({"gnp": [1.0]})
    with pytest.raises(ValueError, match=r"horizon 4 is outside 1\.\.3"):
        call(pd.DataFrame({"gdp": [1.0]}, index=[4]))
    with pytest.raises(ValueError, match=r"horizon 0 is outside 1\.\.3"):
        call(pd.DataFrame({"gdp": [1.0]}, index=[0]))
    with pytest.raises(ValueError, match="outside the forecast periods"):
        call(pd.DataFrame({"gdp": [1.0]}, index=pd.PeriodIndex([small.data.last])))
    with pytest.raises(ValueError, match="hold no value"):
        call({"gdp": [np.nan, np.nan]})
    with pytest.raises(ValueError, match=r"'rate' at horizon 2 is not finite"):
        call({"rate": [1.0, np.inf]})
    full = {v: [0.0, 0.0] for v in small.data.variables}
    with pytest.raises(ValueError, match="every cell of the 2 × 3 forecast is conditioned"):
        call(full, horizon=2)
    with pytest.raises(ValueError, match=r"4 values; the forecast has horizons 1\.\.3"):
        call({"gdp": [0.0] * 4})
    with pytest.raises(ValueError, match="horizon twice"):
        call(pd.DataFrame({"gdp": [0.0, 1.0]}, index=[1, 1]))
    with pytest.raises(TypeError, match="DataFrame"):
        call([0.0, 1.0])
    with pytest.raises(ValueError, match="positive integer"):
        fit.conditional_forecast(0, {"gdp": [0.0]})


def test_e16_lag_coefs_and_base_errors(small):
    """E-16: a base forecast without ``lag_coefs`` names why (an analytic forecast made without
    ``lag_coefs=True``; a benchmark; an MCMC in-loop forecast whose draws are not all stored; a
    non-linear MCMC mean); a mismatched or conditional ``base`` is refused."""
    fit, data = small.fit, small.data
    plain = fit.forecast(3, 20, seed=1)
    with pytest.raises(ValueError, match=r"call forecast\(\.\.\., lag_coefs=True\)"):
        rb.conditional_forecast(fit, 3, {"gdp": [0.0]}, base=plain)
    ols = rb.OLSVAR(lags=2).fit(data)
    with pytest.raises(ValueError, match="benchmark"):
        rb.conditional_forecast(ols, 3, {"gdp": [0.0]}, draws=20, seed=1)
    for in_loop, match in ((1, "in-loop forecast"), (0, "not linear")):
        ds = plain.dataset.copy()
        ds.attrs.update(engine="mcmc", in_loop=in_loop)
        with pytest.raises(ValueError, match=match):
            cf.condition_forecast(ForecastResult(ds), {"gdp": [0.0]})
    with_lags = fit.forecast(3, 20, seed=1, lag_coefs=True)
    with pytest.raises(ValueError, match="horizon 3; conditional_forecast was asked for horizon 2"):
        rb.conditional_forecast(fit, 2, {"gdp": [0.0]}, base=with_lags)
    with pytest.raises(TypeError, match="unconditional"):
        rb.conditional_forecast(fit, 4, {"gdp": [0.0]}, base=small.res)
    renamed = with_lags.dataset.assign_coords(variable=["a", "b", "c"], variable_=["a", "b", "c"])
    with pytest.raises(ValueError, match="ordering"):
        rb.conditional_forecast(fit, 3, {"a": [0.0]}, base=ForecastResult(renamed))


def test_e16_near_dependent_conditions_raise_without_jitter():
    """E-16: ``R₁`` with a diagonal below ``1e-12·max|R₁|`` (two conditions on nearly collinear
    cells) raises and names the draw; the update never jitters."""
    S, H, n = 2, 2, 2
    chol = np.broadcast_to(np.eye(n), (S, H, n, n)).copy()
    chol[1, 0] = [[1.0, 0.0], [1.0, 1e-14]]  # y_1 ≈ y_0 at horizon 1 in draw 1
    base = ForecastResult.from_arrays(
        paths=np.zeros((S, H, n)), cond_mean=np.zeros((S, H, n)), cond_chol=chol,
        variables=["a", "b"], origin=pd.Period("2000Q1"),
        extra={"lag_coefs": (LAG_DIMS, np.zeros((S, 1, n, n)))},
    )  # fmt: skip
    with pytest.raises(ValueError, match=r"nearly\) linearly dependent.*\[chain 0, draw 1\]"):
        cf.condition_forecast(base, {"a": [1.0], "b": [1.0]})
    ok = cf.condition_forecast(base, {"a": [1.0, 1.0]})  # one condition per draw is fine
    check_conditional_structure(ok)


def test_rb_conditional_forecast_resolves_through_the_lazy_table():
    """P3 §1.1 (E57): ``rb.conditional_forecast`` is the module function, exported in ``__all__``.
    That the module is NumPy-only at import (E137) is checked by
    ``tests/unit/test_specs_phase2.py::test_star_import_works_and_imports_no_jax``, whose star
    import now resolves this name."""
    assert rb.conditional_forecast is cf.conditional_forecast
    assert "conditional_forecast" in rb.__all__ and "conditional_forecast" in dir(rb)
