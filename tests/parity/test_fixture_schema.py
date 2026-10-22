"""Schema of the Phase 2 parity fixtures (DESIGN §9.9, work package WP8).

The fixtures are generated offline and committed; nothing here runs R:

* ``fixtures/data/*.csv``: NumPy datasets written by ``tools/make_sv_parity_data.py data`` (the
  stochvol datasets (i)–(v), the VAR-SV datasets of the system-wide Gibbs oracle and the
  bayesianVARs probe datasets);
* ``fixtures/stochvol_<dataset>_<run>.nc``: stochvol ``svsample`` runs (section (e) of
  ``tools/make_r_fixtures.R``); the ``exact`` runs call stochvol's sampler directly and summarise
  the raw, time-ordered chains with stochvol's mixture-correction weights (self-normalised
  importance sampling) instead of svsample's resampling;
* ``fixtures/bvars_swa_*.nc``: the exact system-wide Gibbs oracle that replaces the bayesianVARs
  runs (section (f)); ``fixtures/probe_bayesianvars.nc``: the probes P1–P5.

Variables of a stochvol fixture (``quantile`` = 0.05, 0.25, 0.5, 0.75, 0.95; ``t`` = 0..T with
``t = 0`` the initial state ``h_0``; ``obs`` = 1..T; ``t_sel`` = {0, 1, T//4, T//2, T}): ``y``;
``param_*`` over ``param`` = (mu, phi, sigma); ``h_*`` over ``t_sel``; ``tau_*`` over ``tau_t`` (the
three largest ``|y_t|``, SV-t run only), each as ``*_q``, ``*_mcse`` (= max of ``*_mcse_vehtari``,
Vehtari et al. 2021 §4.3, and ``*_mcse_batch``), ``*_mean``, ``*_sd``, ``*_ess``, ``*_mcse_mean``,
``*_rhat``; ``h_path_q`` / ``h_path_mean`` / ``h_path_sd`` over ``t``; ``param_draws`` (chain,
draw, param), plus ``param_draws_weight`` (chain, draw) in the exact runs. A system-wide Gibbs fixture has the same summaries for ``Pi`` (regressor,
equation), ``A`` (a_free), ``forecast`` (horizon, equation), ``sv`` (sv_param, equation) and ``h``
(t_sel, equation) or ``lambda`` (equation), plus the prior arrays it was run with.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DATA = FIXTURES / "data"
GENERATOR = Path(__file__).parents[2] / "tools" / "make_sv_parity_data.py"
PROBS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

SV_FIXTURES = {
    "stochvol_i_main": ("i", "main"),
    "stochvol_i_defaults": ("i", "defaults"),
    "stochvol_i_exact": ("i", "exact"),
    "stochvol_ii_main": ("ii", "main"),
    "stochvol_ii_defaults": ("ii", "defaults"),
    "stochvol_iii_main": ("iii", "main"),
    "stochvol_iii_defaults": ("iii", "defaults"),
    "stochvol_iv_main": ("iv", "main"),
    "stochvol_iv_defaults": ("iv", "defaults"),
    "stochvol_iv_exact": ("iv", "exact"),
    "stochvol_v_t5": ("v", "t5"),
}
SWA_FIXTURES = {  # name: (data file, lags, vol)
    "bvars_swa_n3p2_sv": ("var_sv_n3.csv", 2, "sv"),
    "bvars_swa_n3p2_const": ("var_sv_n3.csv", 2, "constant"),
    "bvars_swa_n7p4_sv": ("var_sv_n7.csv", 4, "sv"),
}
PROBE_FIXTURE = "probe_bayesianvars"
PROBES = ("P1", "P2", "P3", "P4", "P5")
SUMMARY_SUFFIXES = ("_q", "_mcse", "_mcse_vehtari", "_mcse_batch", "_mean", "_sd", "_ess",
                    "_mcse_mean", "_rhat")  # fmt: skip
COMMON_ATTRS = ("generator", "r_version", "platform", "stochvol_version", "created", "kind",
                "mcse_method", "quick", "rng_kind")  # fmt: skip

# Mixing requirements for committed (full-length) runs: split R-hat and multi-chain ESS.
MAX_RHAT = 1.01
MIN_ESS = 400.0


# ------------------------------------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------------------------------------


@cache
def generator() -> ModuleType:
    """``tools/make_sv_parity_data.py`` imported from its path (``tools`` is not a package)."""
    name = "_rai_bvar_make_sv_parity_data"
    spec = importlib.util.spec_from_file_location(name, GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
def fixture(name: str):
    import xarray as xr

    return xr.load_dataset(FIXTURES / f"{name}.nc", engine="scipy")


@cache
def frame(file: str) -> pd.DataFrame:
    return pd.read_csv(DATA / file, float_precision="round_trip")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def summary_prefixes(ds) -> set[str]:
    return {
        name[: -len("_mcse_vehtari")] for name in ds.data_vars if name.endswith("_mcse_vehtari")
    }


# ------------------------------------------------------------------------------------------------
# Data (CSV)
# ------------------------------------------------------------------------------------------------


def test_data_files_match_the_generator_registry():
    expected = {ds.file for ds in generator().DATASETS.values()}
    assert {p.name for p in DATA.glob("*.csv")} == expected


@pytest.mark.parametrize("name", ["sv_i", "sv_ii", "sv_iii", "sv_iv", "sv_v"])
def test_sv_data_have_no_zeros(name):
    ds = generator().DATASETS[name]
    data = frame(ds.file)
    y = data["y"].to_numpy()
    assert y.dtype == np.float64
    assert np.all(np.isfinite(y))
    assert np.all(np.abs(y) > 0.0), "stochvol adds an offset to data with exact zeros"
    assert len(y) == ds.dgp.get("T", 235)
    dates = pd.PeriodIndex(data["date"], freq="Q")
    assert dates.is_monotonic_increasing and dates[-1] < pd.Period("2020Q1", freq="Q")
    assert np.all(np.diff(dates.asi8) == 1)


@pytest.mark.parametrize("name", ["sv_i", "sv_ii", "sv_iii", "sv_v"])
def test_sv_simulated_paths_follow_the_stated_dgp(name):
    g = generator().DATASETS[name].dgp
    data = frame(f"{name}.csv")
    h = data["h"].to_numpy()
    eta = (h[1:] - g["mu"] - g["phi"] * (h[:-1] - g["mu"])) / g["sigma"]
    assert abs(eta.mean()) < 0.2 and 0.85 < eta.std() < 1.15
    scale = np.exp(0.5 * h) * (np.sqrt(data["tau"].to_numpy()) if g["tails"] == "t" else 1.0)
    eps = data["y"].to_numpy() / scale
    assert abs(eps.mean()) < 0.2 and 0.85 < eps.std() < 1.15
    if g["tails"] == "t":
        base = frame("sv_i.csv")
        np.testing.assert_array_equal(h, base["h"].to_numpy())  # (v) reuses the h path of (i)
        np.testing.assert_allclose(eps, base["y"].to_numpy() / np.exp(0.5 * h), rtol=1e-14)
        assert np.all(data["tau"].to_numpy() > 0.0)


def test_sv_iv_is_the_standardised_gdp_ar4_residual():
    data = frame("sv_iv.csv")
    fresh = generator().generate("sv_iv")  # deterministic: no random numbers involved
    assert list(fresh["date"]) == list(data["date"])
    np.testing.assert_allclose(fresh["y"].to_numpy(), data["y"].to_numpy(), rtol=0, atol=1e-12)
    y = data["y"].to_numpy()
    assert abs(y.mean()) < 1e-12 and abs(y.std(ddof=1) - 1.0) < 1e-12
    assert data["date"].iloc[0] == "1961Q2"  # growth drops one quarter, the AR(4) four more


@pytest.mark.parametrize("name", ["var_sv_n3", "var_sv_n7", "probe_tiny", "probe_corr", "probe_sv"])
def test_var_and_probe_data_shapes(name):
    ds = generator().DATASETS[name]
    data = frame(ds.file)
    n = ds.dgp["n"]
    assert list(data.columns) == ["date"] + [f"y{i + 1}" for i in range(n)]
    assert len(data) == ds.dgp["T_raw"]
    values = data.drop(columns="date").to_numpy()
    assert values.dtype == np.float64 and np.all(np.isfinite(values))
    assert pd.PeriodIndex(data["date"], freq="Q")[-1] < pd.Period("2020Q1", freq="Q")


def test_probe_corr_has_the_correlation_the_probe_needs():
    y = frame("probe_corr.csv").drop(columns="date").to_numpy()
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    coef, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
    u = y[1:] - X @ coef
    assert np.corrcoef(u.T)[0, 1] > 0.7  # P2 decides by the sign of a strong correlation
    # P3 looks for the PHI row nearest (5, −3): the intercepts stand out from the lag rows.
    np.testing.assert_allclose(coef[0], [5.0, -3.0], atol=1.5)
    assert np.min(np.abs(coef[0])) > 2.0 > np.max(np.abs(coef[1:]))


# ------------------------------------------------------------------------------------------------
# Fixtures (NetCDF): files, provenance, common structure
# ------------------------------------------------------------------------------------------------

ALL_FIXTURES = [*SV_FIXTURES, *SWA_FIXTURES, PROBE_FIXTURE]


def test_every_expected_fixture_exists():
    missing = [name for name in ALL_FIXTURES if not (FIXTURES / f"{name}.nc").exists()]
    assert not missing


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_provenance_attributes(name):
    attrs = fixture(name).attrs
    for key in COMMON_ATTRS:
        assert key in attrs, key
    assert attrs["generator"] == "tools/make_r_fixtures.R"
    assert attrs["r_version"].startswith("R version ")
    assert int(attrs["quick"]) == 0, "--quick fixtures must not be committed"
    assert attrs["packed_by"] == "tools/make_sv_parity_data.py pack"
    if name != PROBE_FIXTURE:
        assert isinstance(attrs["seed"], np.integer | int)
        path = FIXTURES / attrs["data_file"]
        assert path.exists()
        assert attrs["data_md5"] == md5(path), "fixture was not generated from the committed data"


@pytest.mark.parametrize("name", [*SV_FIXTURES, *SWA_FIXTURES])
def test_summary_blocks_are_complete_and_well_formed(name):
    ds = fixture(name)
    np.testing.assert_array_equal(ds["quantile"].values, PROBS)
    prefixes = summary_prefixes(ds)
    assert prefixes
    for prefix in prefixes:
        q = ds[f"{prefix}_q"]
        assert q.dims[-1] == "quantile"
        for suffix in SUMMARY_SUFFIXES:
            var = ds[f"{prefix}{suffix}"]
            assert var.dims == (q.dims if suffix in SUMMARY_SUFFIXES[:4] else q.dims[:-1])
            assert var.dtype == np.float64
            assert np.all(np.isfinite(var.values)), f"{prefix}{suffix}"
        assert np.all(np.diff(q.values, axis=-1) >= 0.0), f"{prefix}: quantiles not sorted"
        for suffix in ("_mcse", "_mcse_vehtari", "_mcse_batch", "_sd", "_mcse_mean"):
            assert np.all(ds[f"{prefix}{suffix}"].values > 0.0), f"{prefix}{suffix}"
        np.testing.assert_array_equal(
            ds[f"{prefix}_mcse"].values,
            np.maximum(ds[f"{prefix}_mcse_vehtari"].values, ds[f"{prefix}_mcse_batch"].values),
        )
        mean = ds[f"{prefix}_mean"].values
        assert np.all((q.values[..., 0] <= mean) & (mean <= q.values[..., -1]))


@pytest.mark.parametrize("name", [*SV_FIXTURES, *SWA_FIXTURES])
def test_runs_are_converged(name):
    ds = fixture(name)
    for prefix in summary_prefixes(ds):
        assert float(ds[f"{prefix}_rhat"].max()) < MAX_RHAT, prefix
        assert float(ds[f"{prefix}_ess"].min()) > MIN_ESS, prefix


# ------------------------------------------------------------------------------------------------
# stochvol fixtures
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SV_FIXTURES)
def test_stochvol_fixture_layout(name):
    dataset, run = SV_FIXTURES[name]
    ds = fixture(name)
    a = ds.attrs
    assert a["kind"] == "stochvol" and a["dataset"] == dataset and a["run"] == run
    y = frame(f"sv_{dataset}.csv")["y"].to_numpy()
    T = len(y)
    assert int(a["T"]) == T
    np.testing.assert_array_equal(ds["y"].values, y)  # the CSV doubles, unchanged
    assert np.all(ds["y"].values != 0.0)
    np.testing.assert_array_equal(ds["obs"].values, np.arange(1, T + 1))
    np.testing.assert_array_equal(ds["t"].values, np.arange(T + 1))
    np.testing.assert_array_equal(ds["t_sel"].values, [0, 1, T // 4, T // 2, T])
    assert list(ds["param"].values) == ["mu", "phi", "sigma"]
    assert ds["param_q"].shape == (3, 5) and ds["h_q"].shape == (5, 5)
    assert ds["h_path_q"].shape == (T + 1, 5)
    assert ds["param_draws"].dims == ("chain", "draw", "param")
    assert ds.sizes["chain"] == int(a["chains"]) == 4
    assert int(a["draws"]) == 50_000 and int(a["burnin"]) == 5_000
    # h_q at t_sel and the path summary come from the same draws.
    np.testing.assert_allclose(ds["h_path_q"].values[ds["t_sel"].values], ds["h_q"].values)
    # Parameter quantiles lie in the support.
    q = ds["param_q"].values
    assert q[1, 0] > -1.0 and q[1, -1] < 1.0 and q[2, 0] > 0.0
    svsample_call = a["equivalent_svsample_call"] if run == "exact" else a["call"]
    assert svsample_call.startswith("svsample(y, draws = 50000L, burnin = 5000L, n_chains = 4L")
    if run == "exact":
        assert "stochvol::svsample_fast_cpp(y, draws = 50000L, burnin = 5000L" in a["call"]
        assert f"set.seed({int(a['seed'])}L)" in a["call"]
        w = ds["param_draws_weight"]
        assert w.dims == ("chain", "draw") and np.all(w.values > 0.0)
        assert np.all(np.abs(w.values.mean(axis=1) - 1.0) < 0.2)  # mean 1 over each whole chain
    else:
        assert "param_draws_weight" not in ds
    assert a["rai_bvar_spec"].startswith("SV(") and "offset=0.0" in a["rai_bvar_spec"]


def test_stochvol_runs_map_the_priors_of_design_9_9():
    main = fixture("stochvol_i_main").attrs
    assert "priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1" in main["call"]
    assert (
        "Normal(0.0, 100.0)" in main["rai_bvar_spec"] and "ScaledChi2(0.1)" in main["rai_bvar_spec"]
    )
    assert "exact=False" in main["rai_bvar_spec"]
    defaults = fixture("stochvol_i_defaults").attrs
    assert "prior" not in defaults["call"]  # stochvol's own defaults
    assert (
        "Normal(0.0, 1e4)" in defaults["rai_bvar_spec"]
        and "Beta(5, 1.5)" in defaults["rai_bvar_spec"]
    )
    t5 = fixture("stochvol_v_t5").attrs
    assert "nu = sv_constant(5)" in t5["call"] and "sv_gamma(0.5, 0.5/0.1)" in t5["call"]
    assert 'tails="t"' in t5["rai_bvar_spec"] and "DiscreteUniform(5, 5)" in t5["rai_bvar_spec"]
    exact = fixture("stochvol_i_exact").attrs
    assert "correct_model_misspecification = TRUE" in exact["call"]
    assert "exact=True" in exact["rai_bvar_spec"]


@pytest.mark.parametrize("name", SV_FIXTURES)
def test_stochvol_importance_weighting_attributes(name):
    a = fixture(name).attrs
    assert int(a["resampled"]) == 0  # no fixture summarises a resampled (shuffled) chain
    if SV_FIXTURES[name][1] == "exact":
        assert int(a["importance_weighted"]) == 1
        for part in ("para", "latent"):
            wess = np.asarray(a[f"weight_ess_{part}"])
            assert wess.shape == (4,) and np.all(wess > 0.5 * wess.max())
            assert 1.0 <= float(a[f"mcse_inflation_{part}"]) < 1.5
        assert "latent0" in a["correction"] and "No resampling" in a["correction"]
        assert float(a["check_direct_call_max_abs_diff"]) == 0.0
        assert "importance-weighted" in a["mcse_method"]
    else:
        assert int(a["importance_weighted"]) == 0
        assert float(a["mcse_inflation_para"]) == 1.0 == float(a["mcse_inflation_latent"])


@pytest.mark.parametrize("dataset", ["i", "iv"])
def test_exact_runs_are_summarised_from_time_ordered_chains(dataset):
    """Same data, priors and sampler as ``main``, weights near 1: the ESS and MCSE must be alike.

    A shuffled (resampled) chain has an ESS near the number of draws and a far too small MCSE.
    """
    main, exact = fixture(f"stochvol_{dataset}_main"), fixture(f"stochvol_{dataset}_exact")
    for prefix in ("param", "h"):
        ratio = exact[f"{prefix}_ess"].values / main[f"{prefix}_ess"].values
        assert np.all(ratio < 1.5), (prefix, ratio)
        mcse = np.median(exact[f"{prefix}_mcse"].values / main[f"{prefix}_mcse"].values, axis=-1)
        assert np.all(mcse > 1.0 / 1.5), (prefix, mcse)
    assert float(exact["param_ess"].max()) < 0.5 * 4 * int(exact.attrs["draws"])


def test_stochvol_t_run_stores_tau_at_the_largest_observations():
    ds = fixture("stochvol_v_t5")
    y = ds["y"].values
    largest = np.sort(np.argsort(-np.abs(y))[:3] + 1)  # 1-based t
    np.testing.assert_array_equal(ds["tau_t"].values, largest)
    assert ds["tau_q"].shape == (3, 5) and ds["tau_path_q"].shape == (len(y), 5)
    assert np.all(ds["tau_q"].values > 0.0)
    assert "tau_q" not in fixture("stochvol_i_main")


@pytest.mark.parametrize("name", [*SV_FIXTURES, *SWA_FIXTURES])
def test_quantile_mcse_is_on_the_scale_of_the_ess(name):
    """MCSE(q_α) ≈ sqrt(α(1 − α)) / (f(q_α) sqrt(ESS)), with f from the stored quantile spacing.

    Per quantity, the median over α = 0.25, 0.5, 0.75 of the ratio stored / predicted lies within
    a factor 3 of 1 (observed 0.38–1.08: tail and bulk ESS differ for skewed parameters).
    """
    ds = fixture(name)
    alpha = PROBS[1:4]
    for prefix in summary_prefixes(ds):
        q = ds[f"{prefix}_q"].values
        density = np.stack(
            [
                (PROBS[2] - PROBS[0]) / (q[..., 2] - q[..., 0]),
                (PROBS[3] - PROBS[1]) / (q[..., 3] - q[..., 1]),
                (PROBS[4] - PROBS[2]) / (q[..., 4] - q[..., 2]),
            ],
            axis=-1,
        )
        ess = ds[f"{prefix}_ess"].values[..., None]
        ratio = ds[f"{prefix}_mcse"].values[..., 1:4] * density * np.sqrt(ess)
        ratio = np.median(ratio / np.sqrt(alpha * (1.0 - alpha)), axis=-1)
        assert np.all((ratio > 1.0 / 3.0) & (ratio < 3.0)), (prefix, ratio.min(), ratio.max())


def _lagged_correlations(path: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """corr(path_t, target_t), corr(path_{t+1}, target_t) and corr(path_t, target_{t+1})."""
    return (
        np.corrcoef(path, target)[0, 1],
        np.corrcoef(path[1:], target[:-1])[0, 1],
        np.corrcoef(path[:-1], target[1:])[0, 1],
    )


@pytest.mark.parametrize("name", SV_FIXTURES)
def test_stochvol_h_path_is_aligned_with_the_data(name):
    """h_t belongs to y_t: the smoothed h correlates best with log y_t² at lag 0.

    This catches a shift of the time index by one period, even where h is very persistent.
    """
    ds = fixture(name)
    log_y2 = np.log(ds["y"].values ** 2)
    for path in (ds["h_path_mean"].values[1:], ds["h_path_q"].values[1:, 2]):
        aligned, ahead, behind = _lagged_correlations(path, log_y2)
        assert aligned > max(ahead, behind), (aligned, ahead, behind)


@pytest.mark.parametrize("name", ["stochvol_ii_main", "stochvol_ii_defaults"])
def test_stochvol_ii_h_path_is_aligned_with_the_truth(name):
    """In (ii) h is weakly persistent (φ = 0.5), so a one-period shift visibly lowers the fit."""
    truth = frame("sv_ii.csv")["h"].to_numpy()
    aligned, ahead, behind = _lagged_correlations(fixture(name)["h_path_mean"].values[1:], truth)
    assert aligned > max(ahead, behind) + 0.03, (aligned, ahead, behind)


# Lower bounds on corr(posterior mean of h_t, true h_t). In (ii) (φ = 0.5, σ = 0.5) h is close to
# white noise and only weakly identified, so the bound is lower there.
H_TRACKING = {"i": 0.7, "ii": 0.3, "iii": 0.6, "v": 0.7}


@pytest.mark.parametrize("name", [n for n, (d, _) in SV_FIXTURES.items() if d in H_TRACKING])
def test_stochvol_posterior_tracks_the_simulated_path(name):
    """Data and fixture belong together: the posterior mean of h follows the true path."""
    dataset = SV_FIXTURES[name][0]
    truth = frame(f"sv_{dataset}.csv")["h"].to_numpy()
    r = np.corrcoef(fixture(name)["h_path_mean"].values[1:], truth)[0, 1]
    assert r > H_TRACKING[dataset]


# ------------------------------------------------------------------------------------------------
# System-wide Gibbs oracle (replaces the bayesianVARs runs)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SWA_FIXTURES)
def test_swa_fixture_layout(name):
    file, lags, vol = SWA_FIXTURES[name]
    ds = fixture(name)
    a = ds.attrs
    data = frame(file)
    variables = [c for c in data.columns if c != "date"]
    n, T = len(variables), len(data) - lags
    k = 1 + n * lags
    assert a["kind"] == "bvars_swa" and a["vol"] == vol
    assert (int(a["n"]), int(a["lags"]), int(a["T"]), int(a["k"])) == (n, lags, T, k)
    assert a["data_file"] == f"data/{file}"
    assert list(ds["equation"].values) == variables
    regressors = ["const"] + [f"{v}.L{lag}" for lag in range(1, lags + 1) for v in variables]
    assert list(ds["regressor"].values) == regressors
    free = [(i, j) for i in range(2, n + 1) for j in range(1, i)]
    assert list(zip(ds["a_row"].values, ds["a_col"].values, strict=True)) == free
    assert list(ds["a_free"].values) == [f"a{i}{j}" for i, j in free]
    assert ds["Pi_q"].shape == (k, n, 5) and ds["A_q"].shape == (len(free), 5)
    assert ds["forecast_q"].dims == ("horizon", "equation", "quantile")
    if vol == "sv":
        assert list(ds["sv_param"].values) == ["mu", "phi", "sigma"]
        np.testing.assert_array_equal(ds["t_sel"].values, [0, 1, T // 4, T // 2, (3 * T) // 4, T])
        assert ds["h_q"].shape == (6, n, 5) and ds["h_path_q"].shape == (T + 1, n, 5)
        assert "lambda_q" not in ds
    else:
        assert ds["lambda_q"].shape == (n, 5) and "sv_q" not in ds
        np.testing.assert_array_equal(ds["ig_shape"].values, 3.0)
        np.testing.assert_array_equal(ds["ig_scale"].values, 2.0)
    assert "bayesianVARs" in a["replaces"] and "system-wide" in a["oracle"]
    assert int(a["draws"]) == 20_000 and int(a["chains"]) == 4


@pytest.mark.parametrize("name", SWA_FIXTURES)
def test_swa_prior_arrays_are_the_documented_parameter_set(name):
    ds = fixture(name)
    lags = int(ds.attrs["lags"])
    n = ds.sizes["equation"]
    mean, var = ds["prior_mean"].values, ds["prior_var"].values
    expected_mean = np.zeros_like(mean)
    expected_mean[1 : n + 1] = 0.5 * np.eye(n)
    expected_var = np.empty_like(var)
    expected_var[0] = 10.0**2
    for lag in range(1, lags + 1):
        block = np.full((n, n), (0.2 / lag) ** 2)
        np.fill_diagonal(block, (0.4 / lag) ** 2)
        expected_var[1 + (lag - 1) * n : 1 + lag * n] = block  # row: variable, column: equation
    np.testing.assert_array_equal(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var, rtol=1e-15)
    np.testing.assert_array_equal(ds["a_prior_mean"].values, 0.0)
    np.testing.assert_array_equal(ds["a_prior_var"].values, 1.0)


@pytest.mark.parametrize("name", SWA_FIXTURES)
def test_swa_oracle_self_checks_passed(name):
    a = fixture(name).attrs
    assert float(a["check_pi_system_rel_err"]) < 1e-13
    assert float(a["check_a_gradient"]) < 1e-6
    assert float(a["check_stochvol_single_step_max_abs_z"]) < 4.5


@pytest.mark.parametrize("name", ["bvars_swa_n3p2_sv", "bvars_swa_n3p2_const"])
def test_swa_posterior_is_near_the_simulated_parameters(name):
    """Data and fixture belong together: n = 3, T = 200 identifies the DGP reasonably well."""
    ds = fixture(name)
    a = ds.attrs
    n, lags = 3, 2
    B = np.asarray(a["dgp_B"]).reshape(lags, n, n)  # B[l, i, j]: equation i, variable j
    Pi_mean = ds["Pi_mean"].values  # (regressor, equation)
    for lag in range(lags):
        block = Pi_mean[1 + lag * n : 1 + (lag + 1) * n].T  # (equation, variable)
        assert np.max(np.abs(block - B[lag])) < 0.3
    A = np.asarray(a["dgp_A"]).reshape(n, n)
    truth = np.array(
        [A[i - 1, j - 1] for i, j in zip(ds["a_row"].values, ds["a_col"].values, strict=True)]
    )
    assert np.max(np.abs(ds["A_mean"].values - truth)) < 0.3


# ------------------------------------------------------------------------------------------------
# bayesianVARs probes P1–P5
# ------------------------------------------------------------------------------------------------


def test_probes_are_answered_or_recorded_with_evidence():
    ds = fixture(PROBE_FIXTURE)
    a = ds.attrs
    assert a["kind"] == "probe"
    assert list(ds["probe"].values) == list(PROBES)
    answered = ds["answered"].values
    assert answered.dtype == np.int32
    for probe, flag in zip(PROBES, answered, strict=True):
        for field in (
            "question",
            "documented",
            "dataset",
            "design",
            "status",
            "answer",
            "evidence",
        ):
            assert str(a[f"{probe}_{field}"]).strip(), f"{probe}_{field}"
        assert a[f"{probe}_status"] == ("answered" if flag else "unresolved")
        data_file = str(a[f"{probe}_dataset"]).split()[0]
        assert (FIXTURES / data_file).exists(), data_file
    if not int(a["bayesianvars_available"]):
        assert not answered.any()
        for probe in PROBES:
            assert "not installed" in a[f"{probe}_evidence"]
            assert "FLIBS" in a[f"{probe}_evidence"]
        assert "bvars_swa" in a["consequence"]


def test_probe_fixture_records_the_derived_bayesianvars_calls():
    calls = fixture(PROBE_FIXTURE).attrs["parity_calls"].splitlines()
    assert [c.split(":")[0] for c in calls] == list(SWA_FIXTURES)
    for call in calls:
        assert 'type = "cholesky"' in call and "prior_intercept = 10" in call
        assert 'sv_keep = "all"' in call
    assert "cholesky_priorsigma2 = c(0.5, 0.5)" in calls[0]
    assert "cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))" in calls[1]
