"""Reduced-form parity: bayesianVARs (blocked) and the exact system-wide Gibbs oracle that replaces
it (DESIGN §9.9 bayesianVARs row, §6.2–§6.3, E11; N04 §3.4, §9.5, §9.8; work package WP21).

**Status.** bayesianVARs 0.1.8 could not be installed here (no CRAN macOS binary for R 4.1 and no
gfortran for a source build; ``fixtures/probe_bayesianvars.nc`` records the evidence), so the
probes P1–P5 of DESIGN §9.9 are unresolved and the bayesianVARs parity runs are blocked
(``SV_PARAMETRIZATION.md`` §4). :func:`test_bayesianvars_parity` is written against the frozen
fixture format (that of ``bvars_swa_*``, with ``kind = "bvars_bayesianvars"`` and the P1–P5
answers applied by the generator) and skips cleanly while no such fixture exists or a probe is
unresolved.

**Replacement oracle** (``fixtures/bvars_swa_*.nc``, WP8). An exact system-wide Gibbs sampler
written in R (N04 §9.5): ``vec Π | A, Λ`` drawn jointly from its Gaussian full conditional, the
rows of ``A`` by Bayesian regression, and each equation's SV state by one sweep of stochvol's fast
sampler at offset 0 (as bayesianVARs does through stochvol), or ``λ_i`` from its IG conditional. It
targets the posterior of ``Linear(Minnesota(form="reduced"))`` with ``SV(exact=False, offset=0)``
(or ``Constant("ig")``) and shares no code with the JAX engine (CCCM 2022 corrected triangular
algorithm, D-11).

**One parameter set, both sides derived** (E11). ``π_j ~ N(m_j, diag v_j)`` with ``m = 0.5`` on the
own first lag, ``v = (0.4/l)²`` (own lag ``l``), ``(0.2/l)²`` (cross lag ``l``), ``s_c² = 10²``
(intercept); ``a_ij ~ N(0, 1)``; ``SV(mu_prior=Normal(0, 1e4), phi_prior=Beta(20, 1.5),
sigma2_prior=ScaledChi2(1.0), h0="stationary", offset=0.0, exact=False)``; Constant:
``λ_i ~ IG(3, 2)``. From it: the fixture's prior arrays (checked), our ``CoefPriorArrays`` /
``ContempPriorArrays`` (``priors.reduced.explicit_reduced``), our ``SVPrior``
(``priors.sv_arrays``), and the bayesianVARs call of DESIGN §9.9 (checked against the call the
generator recorded)::

    prior_intercept = s_c   priormean = m_own   cholesky_normal_sds = v_a (= √v_a: P1-proof)
    cholesky_priormu = c(b_μ, √B_μ)   cholesky_priorphi = c(a_φ, b_φ)
    cholesky_priorsigma2 = c(0.5, 0.5/B_σ)   cholesky_priorh0 = "stationary"
    cholesky_priorhomoscedastic = cbind(rep(shape, M), rep(scale, M))

**Ours.** ``fit_mcmc`` on the reduced-form composition (static parts from ``api.static_parts``),
data from ``engine_data_from_vardata`` (rai-bvar regressor layout: constant, lag-1 block, …).

**Comparison** (as ``test_stochvol.py``): quantiles of ``Π`` (regressor × equation), the free
``a_ij``, ``(μ, φ, σ)`` and ``h_t`` at ``t_sel`` with ``t ≥ 1`` (or ``λ_i``), and the 1–4 step
forecasts (one path per stored draw), ``|z| ≤ max(4, z_{1 − α/(2m)})`` with ``α = 1%`` over the
fixture's ``m`` comparisons (4.05–4.15 for ``n = 3``, 4.51 for ``n = 7``). D-7: the SV quantities of a
series whose fold fraction exceeds 0.5% are excluded when they fail (the oracle's SV step is
stochvol's, which takes ``|σ|`` without the flip).

**Tiers.** PR: the parameter-set and call derivations, the status record, and ``n3p2_const`` at
4 × 3,000 kept draws. Nightly (``slow``, ``parity_long``): every SWA fixture at 4 × 20,000 kept
draws, and the bayesianVARs fixtures when they exist.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import test_stochvol as svp  # shared comparison helpers (tests/parity is on sys.path)

from rai_bvar.api import BVAR, static_parts
from rai_bvar.components.specs import SV, Constant, Linear
from rai_bvar.data.dataset import VARData
from rai_bvar.diagnostics import rhat_many
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.sv import Beta, Normal, ScaledChi2

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

FIXTURES = Path(__file__).parent / "fixtures"
DOC = Path(__file__).parent / "SV_PARAMETRIZATION.md"
PROBE = "probe_bayesianvars"
PROBES = ("P1", "P2", "P3", "P4", "P5")
SWA_FIXTURES = ("bvars_swa_n3p2_sv", "bvars_swa_n3p2_const", "bvars_swa_n7p4_sv")
BV_KIND = "bvars_bayesianvars"
BV_FIXTURES = sorted(p.stem for p in FIXTURES.glob(f"{BV_KIND}_*.nc")) or [f"{BV_KIND}_*"]
HORIZON = 4
FOLD_MAX = 0.005  # D-7 materiality (as the stochvol rows)
RHAT_MAX = 1.05

PR_DRAWS, PR_BURN = 3_000, 500
LONG_DRAWS, LONG_BURN = 20_000, 2_000
SEED = 20261023


# --------------------------------------------------------------------------- the parameter set


@dataclass(frozen=True)
class ReducedParameterSet:
    """DESIGN §9.9: Minnesota-shaped independent normal ``Π`` prior, ``a_ij ~ N(0, a_var)``, the
    bayesianVARs-default SV prior or ``λ_i ~ IG(ig_shape, ig_scale)``."""

    m_own: float = 0.5
    own_sd: float = 0.4
    cross_sd: float = 0.2
    intercept_sd: float = 10.0
    a_var: float = 1.0
    b_mu: float = 0.0
    B_mu: float = 1e4
    a_phi: float = 20.0
    b_phi: float = 1.5
    B_sigma: float = 1.0
    ig_shape: float = 3.0
    ig_scale: float = 2.0

    def arrays(self, n: int, lags: int) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, var)`` of ``Π`` as ``(regressor, equation)`` arrays (the fixture layout):
        rows ``const, y1.L1, …, yn.L1, …, yn.Lp``."""
        k = 1 + n * lags
        mean, var = np.zeros((k, n)), np.empty((k, n))
        mean[1 : n + 1] = self.m_own * np.eye(n)
        var[0] = self.intercept_sd**2
        for lag in range(1, lags + 1):
            block = np.full((n, n), (self.cross_sd / lag) ** 2)
            np.fill_diagonal(block, (self.own_sd / lag) ** 2)
            var[1 + (lag - 1) * n : 1 + lag * n] = block  # row: variable, column: equation
        return mean, var

    def sv(self) -> SV:
        return SV(
            mu_prior=Normal(self.b_mu, self.B_mu),
            phi_prior=Beta(self.a_phi, self.b_phi),
            sigma2_prior=ScaledChi2(self.B_sigma),
            h0="stationary",
            offset=0.0,
            exact=False,
        )

    def bayesianvars_args(self, vol: str) -> list[str]:
        """The bayesianVARs arguments derived through the DESIGN §9.9 mapping (``cholesky_*``
        SV priors as stochvol's ``priormu``/``priorphi``; ``priorsigma2 = c(0.5, 0.5/B_σ)``)."""
        r = svp._r
        args = [
            f"prior_intercept = {r(self.intercept_sd)}",  # documented as an s.d.
            f"priormean = {r(self.m_own)}",
            f"cholesky_normal_sds = {r(self.a_var)}",  # 1 = √1: the same under both P1 readings
            'cholesky_U_prior = "normal"',
            'sv_keep = "all"',
        ]
        if vol == "sv":
            args += [
                "cholesky_heteroscedastic = TRUE",
                f"cholesky_priormu = c({r(self.b_mu)}, {r(math.sqrt(self.B_mu))})",
                f"cholesky_priorphi = c({r(self.a_phi)}, {r(self.b_phi)})",
                f"cholesky_priorsigma2 = c(0.5, {r(0.5 / self.B_sigma)})",
                'cholesky_priorh0 = "stationary"',
            ]
        else:
            args += [
                "cholesky_heteroscedastic = FALSE",
                "cholesky_priorhomoscedastic = "
                f"cbind(rep({r(self.ig_shape)}, M), rep({r(self.ig_scale)}, M))",
            ]
        return args


PARAMS = ReducedParameterSet()


# --------------------------------------------------------------------------- fixture access


@cache
def fixture(name: str):
    import xarray as xr

    path = FIXTURES / f"{name}.nc"
    if not path.exists():
        return None
    return xr.load_dataset(path, engine="scipy")


def require(name: str):
    ds = fixture(name)
    if ds is None:
        pytest.skip(f"fixture {name}.nc is missing (tools/make_r_fixtures.R, section bvars)")
    return ds


def free_entries(ds) -> tuple[np.ndarray, np.ndarray]:
    """0-based ``(i, j)`` of the free ``a_ij`` (``i > j``), row-major as the fixtures list them."""
    return ds["a_row"].values - 1, ds["a_col"].values - 1


# --------------------------------------------------------------------------- the reduced-form run


@dataclass
class ReducedRun:
    fit: Any
    forecast: Any
    fold: np.ndarray | None  # (n,) fold fraction per equation (SV only)


def run_reduced(name: str, draws: int, burn: int, path_thin: int = 1) -> ReducedRun:
    """``fit_mcmc`` on ``Linear(Minnesota(form="reduced"))`` × the fixture's volatility with the
    fixture's explicit prior arrays (DESIGN §9.9), then ``fit.forecast`` with one path per stored
    draw."""
    from rai_bvar.engines.mcmc.config import MeanStatic, Shapes, StaticConfig, StoreStatic, bucket
    from rai_bvar.engines.mcmc.data import engine_data_from_vardata
    from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
    from rai_bvar.engines.mcmc.state import (
        ConstantPrior,
        Empty,
        EqPrior,
        PriorArrays,
        SharedPrior,
    )
    from rai_bvar.priors.reduced import explicit_reduced
    from rai_bvar.priors.scales import WindowScales
    from rai_bvar.priors.sv_arrays import sv_prior_arrays
    from rai_bvar.settings import MCMCSettings, Runtime

    ds = require(name)
    a = ds.attrs
    sv_vol = a["vol"] == "sv"
    frame = pd.read_csv(FIXTURES / a["data_file"], float_precision="round_trip")
    frame.index = pd.PeriodIndex(frame.pop("date"), freq="Q")
    data = VARData.from_frame(frame)  # tcode 1: the CSV values are the sample
    n, lags = data.n_vars, int(a["lags"])
    assert list(ds["equation"].values) == list(data.variables)
    vol = PARAMS.sv() if sv_vol else Constant("ig")
    parts = static_parts(BVAR(mean=Linear(prior=Minnesota(form="reduced")), vol=vol, lags=lags))
    assert parts.mean_kind == "linear_reduced" and parts.struct.contemp_scaling == "none"
    engine_data, periods = engine_data_from_vardata(data, lags, "c", 32)
    T = int(np.asarray(engine_data.T_eff))
    assert T == int(a["T"])
    chunk = min(draws, 2_000)
    static = StaticConfig(
        shapes=Shapes.of(n, lags, 1, bucket(T, 32)),
        struct=parts.struct,
        mean=MeanStatic(kind="linear_reduced"),
        coef=parts.coef,
        vol=parts.vol,
        store=StoreStatic(paths=sv_vol),
        chunk=chunk,
    )
    rows, cols = free_entries(ds)
    a_mean, a_var = np.zeros((n, n - 1)), np.ones((n, n - 1))
    a_mean[rows, cols] = ds["a_prior_mean"].values
    a_var[rows, cols] = ds["a_prior_var"].values
    coef, contemp = explicit_reduced(
        ds["prior_mean"].values.T, ds["prior_var"].values.T, a_mean, a_var, n_det=1
    )
    if sv_vol:  # ψ is not read: explicit μ prior, absolute offset 0
        scales = WindowScales(psi=np.ones(n), own_means=np.zeros(n), freq=data.freq)
        core = sv_prior_arrays(vol, scales)
        np.testing.assert_array_equal(core.log_c, -np.inf)
    else:
        core = ConstantPrior(shape=ds["ig_shape"].values.copy(), scale=ds["ig_scale"].values.copy())
    prior = PriorArrays(
        eq=EqPrior(coef=coef, contemp=contemp, vol_core=core, transitory=Empty()),
        shared=SharedPrior(Mc=np.zeros((static.shapes.k, n - 1))),
    )
    settings = MCMCSettings(
        chains=4, draws=draws, burn=burn, seed=SEED, store="paths" if sv_vol else "params",
        path_thin=path_thin, store_max=draws,
    )  # fmt: skip
    fit = fit_mcmc(
        static, engine_data, prior, settings, Runtime(chunk=chunk), seed=SEED,
        salt=zlib.crc32(name.encode()),
        meta=FitMeta(vardata=data, periods=periods, variables=data.variables),
    )  # fmt: skip
    forecast = fit.forecast(HORIZON, draws=4 * draws, seed=SEED)
    fold = np.mean(fit.acceptance["acc_fold"], axis=0) if sv_vol else None
    return ReducedRun(fit=fit, forecast=forecast, fold=fold)


def comparisons(name: str, run: ReducedRun) -> list[svp.Comparison]:
    """The DESIGN §9.9 comparisons of a reduced-form fixture."""
    ds = require(name)
    fit = run.fit
    variables = list(ds["equation"].values)
    regs = list(ds["regressor"].values)
    p = fit.param_draws
    Pi = np.asarray(p.glob.mean.Pi)  # (C, D, k, n): Π[regressor, equation]
    rows, cols = free_entries(ds)
    A = np.asarray(p.struct.a)[..., rows, cols]  # a[i, j] = a_ij (j < i): A u_t = e_t
    out = [
        svp.Comparison.of(
            "Pi", [f"{r}->{v}" for r in regs for v in variables], Pi, ds["Pi_q"], ds["Pi_mcse"]
        ),
        svp.Comparison.of("A", list(ds["a_free"].values), A, ds["A_q"], ds["A_mcse"]),
    ]
    paths = np.asarray(run.forecast.paths.transpose("chain", "draw", "horizon", "variable"))
    assert list(run.forecast.variables) == variables and paths.shape[2] == HORIZON
    out.append(
        svp.Comparison.of(
            "forecast", [f"h{h}:{v}" for h in range(1, HORIZON + 1) for v in variables], paths,
            ds["forecast_q"], ds["forecast_mcse"],
        )
    )  # fmt: skip
    core = p.vol.core
    if "sv_q" in ds:
        sv = np.stack([np.asarray(getattr(core, s)) for s in ds["sv_param"].values], axis=2)
        out.append(
            svp.Comparison.of(
                "sv", [f"{s}:{v}" for s in ds["sv_param"].values for v in variables], sv,
                ds["sv_q"], ds["sv_mcse"],
            )
        )  # fmt: skip
        keep = ds["t_sel"].values >= 1  # h_0's documented prior mean differs (N04 §3.3)
        t_sel = ds["t_sel"].values[keep]
        h = np.asarray(fit.path_draws.h)[..., t_sel]  # (C, D, n, t_sel); index 0 is h_0
        out.append(
            svp.Comparison.of(
                "h", [f"t={t}:{v}" for t in t_sel for v in variables], np.swapaxes(h, -1, -2),
                ds["h_q"].values[keep], ds["h_mcse"].values[keep],
            )
        )  # fmt: skip
    else:
        lam = np.exp(np.asarray(core.log_lam))
        out.append(svp.Comparison.of("lambda", variables, lam, ds["lambda_q"], ds["lambda_mcse"]))
    return out


def excluded_by_fold(run: ReducedRun, variables: list[str]) -> set[str]:
    """Series whose fold fraction exceeds 0.5% (D-7)."""
    if run.fold is None:
        return set()
    return {v for v, f in zip(variables, run.fold, strict=True) if f > FOLD_MAX}


def check_reduced(name: str, draws: int, burn: int, path_thin: int = 1) -> float:
    run = run_reduced(name, draws, burn, path_thin)
    ds = require(name)
    variables = list(ds["equation"].values)
    Pi = np.asarray(run.fit.param_draws.glob.mean.Pi)
    rhat = rhat_many(Pi)
    assert np.all(rhat < RHAT_MAX), f"{name}: our chains do not mix (max R-hat of Π {rhat.max()})"
    comps = comparisons(name, run)
    worst, message = svp.parity_failures(comps, name)
    if message is None:
        return worst
    # D-7: drop the failing SV quantities of series with a material fold fraction, keep the bound
    drop = excluded_by_fold(run, variables)
    m = sum(c.z.size for c in comps)
    bound = svp.z_bound(m)
    left = [
        f"{c.group}[{label}]"
        for c in comps
        for label, zs in zip(c.labels, c.z, strict=True)
        if np.any(np.abs(zs) > bound)
        and not (c.group in ("sv", "h") and label.rsplit(":", 1)[-1] in drop)
    ]
    assert not left, f"{message} (fold fractions {run.fold})"
    pytest.skip(f"excluded (DESIGN §9.9, D-7): failing SV quantities of {sorted(drop)}; {message}")


# --------------------------------------------------------------------------- tests: derivations


@pytest.mark.parametrize("name", SWA_FIXTURES)
def test_swa_fixture_is_the_parameter_set(name):
    """E11: the oracle ran with the arrays derived from the one parameter set; our arrays from
    ``explicit_reduced`` are their equation-major transpose, with the active ``a`` slots set."""
    from rai_bvar.priors.reduced import explicit_reduced

    ds = require(name)
    a = ds.attrs
    n, lags = ds.sizes["equation"], int(a["lags"])
    mean, var = PARAMS.arrays(n, lags)
    np.testing.assert_array_equal(ds["prior_mean"].values, mean)
    np.testing.assert_allclose(ds["prior_var"].values, var, rtol=1e-15)
    np.testing.assert_array_equal(ds["a_prior_mean"].values, 0.0)
    np.testing.assert_array_equal(ds["a_prior_var"].values, PARAMS.a_var)
    coef, contemp = explicit_reduced(mean.T, var.T, 0.0, PARAMS.a_var, n_det=1)
    np.testing.assert_array_equal(coef.mean0, mean.T)
    np.testing.assert_allclose(coef.prec0, 1.0 / var.T, rtol=1e-15)
    np.testing.assert_array_equal(contemp.cmask, np.tril(np.ones((n, n)), -1)[:, : n - 1])
    if a["vol"] == "sv":
        assert (
            "mu_i ~ N(0, 1e4), (phi_i + 1)/2 ~ Beta(20, 1.5), sigma_i^2 ~ 1 * chi2_1"
            in (a["prior_vol"])
        )
        spec = re.search(r"SV\(.*\)\)", a["rai_bvar_spec"].replace("'", '"')).group(0)[:-1]
        assert svp.parse_spec(spec) == PARAMS.sv()
    else:
        np.testing.assert_array_equal(ds["ig_shape"].values, PARAMS.ig_shape)
        np.testing.assert_array_equal(ds["ig_scale"].values, PARAMS.ig_scale)


def test_bayesianvars_calls_derive_from_the_parameter_set():
    """The bayesianVARs calls recorded by the generator carry the arguments derived from the same
    parameter set through the DESIGN §9.9 mapping."""
    probe = require(PROBE)
    calls = dict(line.split(": ", 1) for line in probe.attrs["parity_calls"].splitlines())
    assert set(calls) == set(SWA_FIXTURES)
    for name, call in calls.items():
        ds = require(name)
        assert f"lags = {int(ds.attrs['lags'])}" in call
        for arg in PARAMS.bayesianvars_args(ds.attrs["vol"]):
            assert arg in call, (name, arg)


def test_bayesianvars_status_is_recorded():
    """SV_PARAMETRIZATION.md states the probe outcomes the probe fixture records, and says the
    bayesianVARs parity is blocked while the package is unavailable."""
    probe = require(PROBE)
    text = DOC.read_text(encoding="utf-8")
    available = bool(int(probe.attrs["bayesianvars_available"]))
    for pid in PROBES:
        status = probe.attrs[f"{pid}_status"]
        assert re.search(rf"\| {pid} \|[^\n]*\| {status} \|", text), (pid, status)
    if not available:
        assert "bayesianVARs parity is blocked" in text
        assert "gfortran" in text and probe.attrs["bayesianvars_target_version"] in text


# --------------------------------------------------------------------------- tests: parity


def test_swa_parity_constant():
    """PR tier: reduced form × ``Constant("ig")`` (``n = 3, p = 2``) against the exact
    system-wide Gibbs oracle within 4 quantile-MCSE (Bonferroni-widened for 100+ quantities)."""
    check_reduced("bvars_swa_n3p2_const", PR_DRAWS, PR_BURN)


@pytest.mark.slow
@pytest.mark.parity_long
@pytest.mark.parametrize("name", SWA_FIXTURES)
def test_swa_parity_long(name):
    """Nightly: every system-wide Gibbs fixture at 4 × 20,000 kept draws."""
    check_reduced(name, LONG_DRAWS, LONG_BURN, path_thin=4)


@pytest.mark.slow
@pytest.mark.parity_long
@pytest.mark.parametrize("name", BV_FIXTURES)
def test_bayesianvars_parity(name):
    """Nightly, once bayesianVARs fixtures exist and P1–P5 are settled (DESIGN §9.9): the same
    comparison against a bayesianVARs run in the frozen ``bvars_swa`` format."""
    probe = fixture(PROBE)
    if fixture(name) is None:
        evidence = "" if probe is None else f"; {probe.attrs['consequence']}"
        pytest.skip(f"no bayesianVARs parity fixture {name}.nc{evidence}")
    if probe is None or not bool(int(probe.attrs["bayesianvars_available"])):
        pytest.skip("probe_bayesianvars.nc does not record an installed bayesianVARs")
    unresolved = [p for p in PROBES if probe.attrs[f"{p}_status"] != "answered"]
    if unresolved:
        pytest.skip(f"bayesianVARs conventions {unresolved} unresolved (DESIGN §9.9 P1–P5)")
    ds = fixture(name)
    assert ds.attrs["kind"] == BV_KIND
    for arg in PARAMS.bayesianvars_args(ds.attrs["vol"]):
        assert arg in ds.attrs["call"], arg
    check_reduced(name, LONG_DRAWS, LONG_BURN, path_thin=4)
