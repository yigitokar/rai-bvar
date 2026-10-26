"""Schema of the parity fixtures of Phase 2 (DESIGN §9.9, work package WP8) and Phase 3
(docs/phase3/DESIGN.md §9.8, work package WP29).

The fixtures are generated outside every test job and committed; nothing here runs R:

* ``fixtures/data/*.csv``: NumPy datasets written by ``tools/make_sv_parity_data.py data`` (the
  stochvol datasets (i)–(v), the VAR-SV datasets of the system-wide Gibbs oracle and the
  bayesianVARs probe datasets);
* ``fixtures/stochvol_<dataset>_<run>.nc``: stochvol ``svsample`` runs (section (e) of
  ``tools/make_r_fixtures.R``); the ``exact`` runs call stochvol's sampler directly and summarise
  the raw, time-ordered chains with stochvol's mixture-correction weights (self-normalised
  importance sampling) instead of svsample's resampling;
* ``fixtures/bvars_swa_*.nc``: the exact system-wide Gibbs oracle that replaces the bayesianVARs
  runs (section (f)); ``fixtures/probe_bayesianvars.nc``: the probes P1–P5.
* Phase 3 (the dispatch-only ``r-fixtures`` job of ``.github/workflows/ci.yml``, D-54; R 4.5.3 and
  one Posit Package Manager snapshot): ``bvars_bayesianvars_*.nc`` (bayesianVARs 0.1.8 runs R1,
  R2 in the frozen ``bvars_swa`` format), ``bvars_gl_*.nc`` (R3–R9, global-local priors, with
  the hyperparameter draws), ``probe_bayesianvars_p3.nc`` (P6–P10), ``probe_bvars_p8_*.nc`` (the
  bayesianVARs side of P8), ``probe_wbvars.nc`` and ``probe_wbvars_p16.nc`` (bvars 1.1, P11–P16),
  ``wbvars_csv.nc`` and ``wbvars_ncsv.nc`` (the bvars CommonSV runs); ``stochvol_v_t5.nc`` is
  regenerated with stochvol 3.2.9 (P2 §2.10 Stage 4 amendment 1).

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
import re
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


# Datasets written by other generators: the exact CommonSV oracle's C2 data (Phase 3 DESIGN §9.8,
# tools/make_commonsv_oracle.py; checked in tests/parity/oracles/test_commonsv_exact.py).
OTHER_DATA = {"commonsv_n3.csv"}


def test_data_files_match_the_generator_registry():
    expected = {ds.file for ds in generator().DATASETS.values()} | OTHER_DATA
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


@pytest.mark.parametrize(
    "name",
    ["var_sv_n3", "var_sv_n7", "probe_tiny", "probe_corr", "probe_sv", "probe_gl", "probe_t2n5"],
)
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


# ------------------------------------------------------------------------------------------------
# Phase 3 (docs/phase3/DESIGN.md §9.8, work package WP29): bayesianVARs 0.1.8 and bvars 1.1
# ------------------------------------------------------------------------------------------------

P3_R_VERSION = "R version 4.5.3"
P3_SNAPSHOT = "2026-10-20"
BV_RUNS = {  # name: (data file, lags, vol, run) — the frozen bvars_swa format (R1, R2)
    "bvars_bayesianvars_n3p2_sv": ("var_sv_n3.csv", 2, "sv", "R1"),
    "bvars_bayesianvars_n7p4_sv": ("var_sv_n7.csv", 4, "sv", "R1"),
    "bvars_bayesianvars_n3p2_const": ("var_sv_n3.csv", 2, "constant", "R2"),
}
GL_RUNS = {  # name: (data file, lags, family, groups, run)
    **{
        f"bvars_gl_{run.lower()}_{fam.lower()}_global_n{n}p2": (
            f"var_sv_n{n}.csv",
            2,
            fam,
            "global",
            run,
        )
        for run, fam in (("R3", "HS"), ("R4", "NG"), ("R5", "R2D2"), ("R6", "DL"), ("R7", "SSVS"))
        for n in (3, 7)
    },
    "bvars_gl_r8_hs_olcl_n7p4": ("var_sv_n7.csv", 4, "HS", "olcl-lagwise", "R8"),
    "bvars_gl_r9_nggrid_global_n7p2": ("var_sv_n7.csv", 2, "NG", "global", "R9"),
}
P8_FIXTURES = {  # name: family
    "probe_bvars_p8_hs": "HS",
    "probe_bvars_p8_ng_a0.1_b1_c4": "NG",
    "probe_bvars_p8_ng_a0.1_b1_c1": "NG",
    "probe_bvars_p8_r2d2_a0.1_b4": "R2D2",
    "probe_bvars_p8_r2d2_a0.1_b0.5": "R2D2",
    "probe_bvars_p8_dl_a1overK": "DL",
    "probe_bvars_p8_dl_a0.5": "DL",
}
WB_RUNS = {"wbvars_csv": "cSV", "wbvars_ncsv": "ncSV"}
PROBE_P3 = "probe_bayesianvars_p3"
PROBE_WB = "probe_wbvars"
PROBE_P16 = "probe_wbvars_p16"
PROBE_IDS = {PROBE_P3: ("P6", "P7", "P8", "P9", "P10"),
             PROBE_WB: ("P11", "P12", "P13", "P14", "P15", "P16")}  # fmt: skip
PENDING = {"P8", "P16"}  # data committed, decision pending WP30 / WP38
BV_SUMMARY_FIXTURES = [*BV_RUNS, *GL_RUNS, *P8_FIXTURES]
P3_FIXTURES = [*BV_SUMMARY_FIXTURES, *WB_RUNS, PROBE_P3, PROBE_WB, PROBE_P16]
COMPARED = ("Pi", "A", "sv", "h", "forecast", "lambda", "B", "Sigma")  # the gated summaries
DOC = Path(__file__).parent / "SV_PARAMETRIZATION.md"
# the SV runs: R1 and every GL run
BV_SV_FIXTURES = [n for n, v in BV_RUNS.items() if v[2] == "sv"] + list(GL_RUNS)


def documented_runs() -> dict[str, str]:
    """The r-fixtures runs §7 of SV_PARAMETRIZATION.md names as the source of committed
    fixtures, with the commit each ran at: "run <id> (commit `<sha>`)"."""
    text = DOC.read_text(encoding="utf-8")
    runs = dict(re.findall(r"run (\d{9,}) \(commit `([0-9a-f]{7,40})`\)", text))
    assert runs, "SV_PARAMETRIZATION.md §7 names no fixture run"
    return runs


def check_run_provenance(a) -> None:
    """The fixture comes from a CI run §7 names; a fixture that records its commit was made by
    that run's commit."""
    runs = documented_runs()
    ci_run = str(a["ci_run"])
    assert ci_run != "not set" and re.fullmatch(
        r"https://github.com/[^/]+/[^/]+/actions/runs/\d+", ci_run
    )
    run_id = ci_run.rsplit("/", 1)[1]
    assert run_id in runs, f"run {run_id} is not named in SV_PARAMETRIZATION.md §7"
    if "git_commit" in a:
        assert str(a["git_commit"]).startswith(runs[run_id]), (run_id, a["git_commit"])


def check_summary_block(ds, prefix: str) -> None:
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


def test_every_phase3_fixture_exists():
    missing = [name for name in P3_FIXTURES if not (FIXTURES / f"{name}.nc").exists()]
    assert not missing


@pytest.mark.parametrize("name", P3_FIXTURES)
def test_phase3_provenance(name):
    """Every Phase 3 fixture names the R version, the package versions and the snapshot."""
    a = fixture(name).attrs
    for key in COMMON_ATTRS:
        assert key in a, key
    assert a["generator"] == "tools/make_r_fixtures.R"
    assert a["packed_by"] == "tools/make_sv_parity_data.py pack"
    assert int(a["quick"]) == 0, "--quick fixtures must not be committed"
    assert a["design"].startswith("docs/phase3/DESIGN.md sec. 9.8")
    assert a["r_version"].startswith(P3_R_VERSION)
    assert a["ppm_snapshot"] == P3_SNAPSHOT
    assert f"/{P3_SNAPSHOT}" in a["repos"] and "packagemanager.posit.co" in a["repos"]
    packages = a["packages"]
    for pkg in ("bayesianVARs 0.1.8", "bvars 1.1", "bsvars 4.0", "stochvol 3.2.9"):
        assert pkg in packages, pkg
    assert a["stochvol_version"] == "3.2.9"
    check_run_provenance(a)
    if "data_file" in a:
        path = FIXTURES / a["data_file"]
        assert path.exists()
        assert a["data_md5"] == md5(path), "fixture was not generated from the committed data"


@pytest.mark.parametrize("name", [*BV_SUMMARY_FIXTURES, *WB_RUNS])
def test_phase3_summary_blocks_are_well_formed(name):
    ds = fixture(name)
    np.testing.assert_array_equal(ds["quantile"].values, PROBS)
    prefixes = summary_prefixes(ds)
    assert prefixes
    for prefix in prefixes:
        check_summary_block(ds, prefix)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _mixes(rhat, ess) -> bool:
    rhat, ess = _finite(rhat), _finite(ess)
    return bool(rhat.size and ess.size and rhat.max() < MAX_RHAT and ess.min() > MIN_ESS)


@pytest.mark.parametrize("name", [*BV_RUNS, *GL_RUNS, *WB_RUNS])
def test_phase3_mixing_record_matches_the_summaries(name):
    """The fixture's mixing record (split R-hat < 1.01 and multi-chain ESS > 400 over what it
    covers) agrees with its stored R-hat and ESS. A fixture with `mixing_covers` names the
    summaries the record covers, the GL hyperparameters included (review C3): the indicator
    summaries (incl_prob, grid_prob) are checked on their non-constant indicators, and a grid
    parameter that never left one point is not mixing. Whether an unmixed fixture can serve a
    comparison is the parity test's decision (D-59, Q39(b)), not the schema's."""
    ds = fixture(name)
    a = ds.attrs
    if "mixing_covers" in a:
        covers = [c.strip() for c in str(a["mixing_covers"]).split(",")]
        continuous = [c for c in covers if c in summary_prefixes(ds)]
        indicators = [c for c in covers if c not in continuous]
        assert set(summary_prefixes(ds) & set(COMPARED)) <= set(continuous)
    else:  # the fixtures staged before the record named what it covers
        continuous, indicators = sorted(summary_prefixes(ds) & set(COMPARED)), []
    assert continuous
    rhat = max(float(_finite(ds[f"{p}_rhat"].values).max()) for p in continuous)
    ess = min(float(_finite(ds[f"{p}_ess"].values).min()) for p in continuous)
    np.testing.assert_allclose(float(a["mixing_max_rhat"]), rhat, rtol=1e-12)
    np.testing.assert_allclose(float(a["mixing_min_ess"]), ess, rtol=1e-12)
    failed = [p for p in continuous if not _mixes(ds[f"{p}_rhat"].values, ds[f"{p}_ess"].values)]
    for item in indicators:
        if item == "incl_prob":
            prob = ds["incl_prob"].values
            moving = (prob > 0.0) & (prob < 1.0)
            if moving.any() and not _mixes(
                ds["incl_prob_rhat"].values[moving], ds["incl_prob_ess"].values[moving]
            ):
                failed.append(item)
        else:
            kind, param = item.split(" ")
            assert kind == "grid_prob", item
            row = list(ds["grid_param"].values).index(param)
            prob = ds["grid_prob"].values[row]
            visited = prob > 0.0
            if visited.sum() == 1 or not _mixes(
                ds["grid_prob_rhat"].values[row][visited], ds["grid_prob_ess"].values[row][visited]
            ):
                failed.append(item)
    assert int(a["mixing_ok"]) == int(not failed), failed
    if "mixing_failed" in a:
        entries = [] if a["mixing_failed"] == "none" else str(a["mixing_failed"]).split("; ")
        assert [e.split(" (")[0] for e in entries] == failed


@pytest.mark.parametrize("name", [*BV_RUNS, *GL_RUNS, *P8_FIXTURES])
def test_bayesianvars_runs_are_four_independent_runs(name):
    a = fixture(name).attrs
    assert (int(a["chains"]), int(a["draws"]), int(a["burnin"])) == (4, 50_000, 10_000)
    seeds = np.asarray(a["chain_seeds"])
    assert seeds.shape == (4,) and len(set(seeds.tolist())) == 4
    call = a["call"]
    assert call.startswith("bvar(data, lags = ")
    assert "draws = 50000L, burnin = 10000L, thin = 1L" in call
    assert 'sv_keep = "all"' in call and "prior_intercept = 10" in call
    assert "PHI_tol = 0" in call and "cholesky_U_tol = 0" in call  # P10: every floor is 0
    assert 'cholesky_U_prior = "normal", cholesky_normal_sds = 1' in call
    assert "expert_sv_offset" not in call  # bayesianVARs' default offset 0 (DESIGN §9.8)
    for tol in ("NG_tol", "R2D2_tol", "DL_tol"):
        if tol in call:
            assert f"{tol} = 0" in call
    # recorded, not judged: P10's "a draw at a floor => that run not met" is applied at
    # comparison (D-59, WP38), not here
    for key in ("floor_fraction_phi", "floor_fraction_u"):
        assert np.isfinite(float(a[key])) and 0.0 <= float(a[key]) <= 1.0, key
    assert int(a["nonfinite_phi"]) == 0


def _regressors(file: str, lags: int) -> tuple[list[str], list[str]]:
    data = frame(file)
    variables = [c for c in data.columns if c != "date"]
    return variables, ["const"] + [f"{v}.L{lag}" for lag in range(1, lags + 1) for v in variables]


@pytest.mark.parametrize("name", BV_RUNS)
def test_bayesianvars_fixture_layout(name):
    """R1/R2: the frozen bvars_swa layout, the P2 §9.9 parameter set and its derived call."""
    file, lags, vol, run = BV_RUNS[name]
    ds = fixture(name)
    a = ds.attrs
    variables, regressors = _regressors(file, lags)
    n, T = len(variables), len(frame(file)) - lags
    assert a["kind"] == "bvars_bayesianvars" and a["vol"] == vol and a["run"] == run
    assert (int(a["n"]), int(a["lags"]), int(a["T"]), int(a["k"])) == (n, lags, T, 1 + n * lags)
    assert list(ds["equation"].values) == variables and list(ds["regressor"].values) == regressors
    free = [(i, j) for i in range(2, n + 1) for j in range(1, i)]
    assert list(zip(ds["a_row"].values, ds["a_col"].values, strict=True)) == free
    assert ds["Pi_q"].shape == (1 + n * lags, n, 5) and ds["A_q"].shape == (len(free), 5)
    assert ds["forecast_q"].dims == ("horizon", "equation", "quantile")
    if vol == "sv":
        # bayesianVARs returns no h_0 (P4): t_sel starts at 1
        np.testing.assert_array_equal(ds["t_sel"].values, [1, T // 4, T // 2, (3 * T) // 4, T])
        assert ds["h_path_q"].shape == (T, n, 5)
    else:
        np.testing.assert_array_equal(ds["ig_shape"].values, 3.0)
        np.testing.assert_array_equal(ds["ig_scale"].values, 2.0)
        assert "cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))" in a["call"]  # P5
    # the parameter set of P2 §9.9 (as test_swa_prior_arrays_are_the_documented_parameter_set)
    mean, var = ds["prior_mean"].values, ds["prior_var"].values
    expected_mean = np.zeros_like(mean)
    expected_mean[1 : n + 1] = 0.5 * np.eye(n)
    np.testing.assert_array_equal(mean, expected_mean)
    assert np.all(var[0] == 100.0) and "prior_intercept = 10" in a["call"]
    # P1 (s.d., column-major over the PHI lag rows) and P3 (lag rows in rai-bvar order)
    np.testing.assert_allclose(ds["bv_normal_sds"].values, np.sqrt(var[1:]).ravel(order="F"),
                               rtol=1e-15)  # fmt: skip
    assert "normal_sds = bv_normal_sds, priormean = 0.5" in a["call"]


@pytest.mark.parametrize("name", BV_SV_FIXTURES)
def test_sv_parameters_are_per_equation(name):
    """Review C1: the SV parameters (mu, phi, sigma) of each equation are that equation's own
    draws. The n = 7 fixtures of run 40243289623 held the first three equations' draws recycled
    over seven columns (P4's n = 3 permutation applied at n = 7): mu's posterior mean was the same
    in every equation to within 0.005, against a spread above 1 in the data's equations."""
    ds = fixture(name)
    params = list(ds["sv_param"].values)
    assert params == ["mu", "phi", "sigma"]
    mean = ds["sv_mean"].values  # (sv_param, equation)
    assert mean.shape == (3, int(ds.attrs["n"]))
    mu = mean[params.index("mu")]
    assert np.ptp(mu) > 1e-2, f"mu's posterior mean is the same in every equation: {mu}"
    for k in range(3):  # no two equations share a posterior
        assert len(np.unique(np.round(mean[k], 12))) == mean.shape[1], params[k]


@pytest.mark.parametrize("name", BV_SV_FIXTURES)
def test_sv_predictive_sources(name):
    """E173 (Q42): the compared SV predictive (`forecast_*`) is predict()'s at h = 1 and the path
    recursion from bayesianVARs' own draws at h >= 2; predict()'s native predictive is kept as
    evidence (`forecast_predict_*`), outside the mixing record and never compared."""
    ds = fixture(name)
    a = ds.attrs
    assert a["forecast_source"] == "h1: predict(); h>=2: path recursion from own draws (E173, Q42)"
    assert ds["forecast_predict_q"].dims == ds["forecast_q"].dims
    assert ds["forecast_predict_q"].shape == ds["forecast_q"].shape
    for suffix in SUMMARY_SUFFIXES:  # h = 1: the same draws in both blocks
        np.testing.assert_array_equal(
            ds[f"forecast{suffix}"].values[0], ds[f"forecast_predict{suffix}"].values[0]
        )
    assert "forecast_predict" not in [c.strip() for c in a["mixing_covers"].split(",")]
    assert int(a["forecast_h1_check_pass"]) in (0, 1)
    assert np.isfinite(float(a["forecast_h1_check_max_abs_z"]))
    assert np.asarray(a["forecast_path_seeds"]).shape == (4,)


@pytest.mark.parametrize("name", GL_RUNS)
def test_gl_fixture_layout(name):
    """R3–R9: the global-local runs, their hyperparameter draws and the compared extras."""
    file, lags, family, groups, run = GL_RUNS[name]
    ds = fixture(name)
    a = ds.attrs
    variables, regressors = _regressors(file, lags)
    n, T = len(variables), len(frame(file)) - lags
    assert a["kind"] == "bvars_gl" and a["run"] == run and a["family"] == family
    assert a["groups"] == groups and a["vol"] == "sv"
    assert f'prior = "{family}"' in a["call"] and f'global_grouping = "{groups}"' in a["call"]
    assert "priormean = 0" in a["call"]
    assert list(ds["regressor"].values) == regressors
    np.testing.assert_array_equal(ds["prior_mean"].values, 0.0)
    np.testing.assert_array_equal(ds["t_sel"].values, [1, T // 4, T // 2, (3 * T) // 4, T])
    draws = ds["hyper_draws"]
    assert draws.dims == ("chain", "hyper_draw", "hyper") and draws.shape[0] == 4
    assert np.all(np.isfinite(draws.values))
    names = list(ds["hyper"].values)
    if family == "SSVS":
        assert "SSVS_semiautomatic = FALSE" in a["call"]
        p = ds["incl_prob"].values
        assert len(p) == n * n * lags and np.all((p >= 0.0) & (p <= 1.0))
        assert all(name.startswith("gamma: ") for name in ds["incl"].values)
    if family == "HS":
        groups_n = {"global": 1, "olcl-lagwise": 2 * lags}[groups]
        assert [h for h in names if h.startswith("zeta")] == [
            f"zeta{g}" for g in range(1, groups_n + 1)
        ]
        assert set(ds["hyper_sel"].values) == {f"zeta{g}" for g in range(1, groups_n + 1)}
    if "groups_matrix" in ds:
        G = ds["groups_matrix"].values
        assert G.shape == (n * lags, n) and len(np.unique(G)) == (
            1 if groups == "global" else 2 * lags
        )
    if run == "R9":
        support, weight = ds["ng_grid_support"].values, ds["ng_grid_weight"].values
        assert "NG_a = ng_grid" in a["call"] and np.isclose(weight.sum(), 1.0)
        k = names.index("a1")
        if "grid_prob" in ds:  # a's posterior on its grid (staged whether or not a moved)
            np.testing.assert_array_equal(ds["grid_support"].values, support)
            prob = ds["grid_prob"].values
            assert prob.shape == (1, len(support)) and np.isclose(prob.sum(), 1.0)
            assert list(ds["grid_param"].values) == ["a1"]
            assert "a1: grid" in a["hyper_columns"]
            if (prob[0] > 0).sum() == 1:  # a never moved: recorded as not mixing (Q39(b), E167)
                assert int(a["mixing_ok"]) == 0 and "grid_prob a1 (never left" in a["mixing_failed"]
        else:  # run 40243289623's layout: a never left one grid point, a constant column
            values = np.unique(draws.values[:, :, k])
            assert len(values) == 1 and np.isclose(support, values[0]).any()
            assert "a1: constant" in a["hyper_columns"]


@pytest.mark.parametrize("name", P8_FIXTURES)
def test_p8_fixture_layout(name):
    """The bayesianVARs side of P8: constant variances, all hyperparameter and V_prior draws."""
    ds = fixture(name)
    a = ds.attrs
    assert a["kind"] == "probe_p8" and a["family"] == P8_FIXTURES[name] and a["vol"] == "constant"
    assert a["data_file"] == "data/probe_gl.csv" and int(a["T"]) == 8 and int(a["lags"]) == 2
    assert "cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))" in a["call"]
    assert ds["lambda_q"].shape == (3, 5) and "forecast_q" not in ds
    for var in ("hyper_draws", "vprior_draws"):
        assert ds[var].dims[:2] == ("chain", "hyper_draw") and np.all(np.isfinite(ds[var].values))


def _probe_records(name):
    ds = fixture(name)
    a = ds.attrs
    ids = PROBE_IDS[name]
    assert list(ds["probe"].values) == list(ids)
    flags = ds["answered"].values
    out = {}
    for probe, flag in zip(ids, flags, strict=True):
        for field in ("question", "documented", "dataset", "design", "status", "answer"):
            assert str(a[f"{probe}_{field}"]).strip(), f"{probe}_{field}"
        status = a[f"{probe}_status"]
        assert status in ("answered", "unresolved", "pending")
        assert bool(flag) == (status == "answered")
        assert (status == "pending") == (probe in PENDING), (probe, status)
        if status == "unresolved":
            assert str(a[f"{probe}_evidence"]).strip(), f"{probe}: undecided without evidence"
        out[probe] = status
    return out


def _answers(text: str) -> dict[str, str]:
    out = {}
    for line in str(text).split("\n"):
        probe, rest = line.split(": ", 1)
        out[probe] = rest
    return out


@pytest.mark.parametrize("name", [*BV_RUNS, *GL_RUNS, *P8_FIXTURES, *WB_RUNS])
def test_run_fixtures_carry_the_committed_probe_decisions(name):
    """Every run fixture records the probe decisions its run staged it under (`probe_answers`):
    they equal the committed probe fixtures' status and answer, so fixtures regenerated by a later
    run (review C1) rest on the same decisions as the probe fixtures."""
    a = fixture(name).attrs
    sources = (
        [(PROBE_WB, PROBE_IDS[PROBE_WB])]
        if name in WB_RUNS
        else [
            ("probe_bayesianvars", ("P1", "P2", "P3", "P4", "P5")),
            (PROBE_P3, PROBE_IDS[PROBE_P3]),
        ]
    )
    expected = {}
    for source, ids in sources:
        b = fixture(source).attrs
        expected.update({k: f"{b[f'{k}_status']} - {b[f'{k}_answer']}" for k in ids})
    assert _answers(a["probe_answers"]) == expected
    if name in WB_RUNS:
        assert a["gating"] == fixture(PROBE_WB).attrs["gating_comparison"]


@pytest.mark.parametrize("name", [PROBE_P3, PROBE_WB])
def test_phase3_probes_are_decided_or_recorded(name):
    """D-59: every probe is answered, recorded as undecided with its evidence, or (P8, P16)
    pending with its data committed; SV_PARAMETRIZATION.md records the same status."""
    statuses = _probe_records(name)
    text = DOC.read_text(encoding="utf-8")
    for probe, status in statuses.items():
        assert re.search(rf"\| {probe} \|[^\n]*\| {status} \|", text), (probe, status)


def test_p7_grouping_matrices_are_recorded():
    ds = fixture(PROBE_P3)
    assert list(ds["grouping"].values) == [
        "global",
        "equation-wise",
        "covariate-wise",
        "olcl-lagwise",
    ]
    G = ds["p7_groups"].values
    assert G.shape == (4, 6, 3) and np.all(G >= 0)
    np.testing.assert_array_equal(ds["p7_equals_d38"].values.shape, (4,))


def test_wbvars_probe_fields_are_data():
    """P11: the bvars defaults are stored as arrays (homoskedastic, cSV, ncSV)."""
    ds = fixture(PROBE_WB)
    for vol in ("homoskedastic", "cSV", "ncSV"):
        for field in ("A", "S", "nu", "Psi", "Gamma", "lambda"):
            assert f"p11_{vol}_prior_{field}" in ds, (vol, field)
        for field in ("A", "Sigma", "V", "h", "rho", "omega", "sigma2v", "sigma2_omega", "s_"):
            assert f"p11_{vol}_start_{field}" in ds, (vol, field)
    assert ds["p11_cSV_prior_A"].shape == (3, 13) and ds["p11_cSV_start_V"].shape == (13, 13)
    gating = ds.attrs["gating_comparison"]
    assert gating.startswith("cSV: branch ") and "; ncSV: branch " in gating


def test_wbvars_p16_draws():
    """P16 data: the chained draws of h (T = 2), rho, omega, ... for k = 4 runs per volatility."""
    ds = fixture(PROBE_P16)
    a = ds.attrs
    assert a["data_file"] == "data/probe_t2n5.csv" and int(a["lags"]) == 1
    calls = ds.sizes["iter"]
    for vol in ("cSV", "ncSV"):
        h = ds[f"p16_{vol}_h"]
        assert h.dims == ("run", "iter", "n2") and h.shape == (4, calls, 2)
        for field in ("h", "rho", "omega", "sigma2_omega", "s_"):
            assert np.all(np.isfinite(ds[f"p16_{vol}_{field}"].values)), (vol, field)
        assert int(a[f"{vol}_extraction_pass"]) in (0, 1)
        if fixture(PROBE_WB).attrs["P16_status"] == "pending":  # §9.8: a failure => undecided
            assert int(a[f"{vol}_extraction_pass"]) == 1, vol
        assert int(a[f"{vol}_draws_per_call"]) in (1, 2)
        assert np.asarray(a[f"{vol}_seeds"]).shape == (4,)


@pytest.mark.parametrize("name", WB_RUNS)
def test_wbvars_fixture_layout(name):
    ds = fixture(name)
    a = ds.attrs
    variables, regressors = _regressors("qd3_growth.csv", 4)
    assert a["kind"] == "wbvars" and a["vol"] == WB_RUNS[name]
    assert a["data_file"] == "data/qd3_growth.csv" and int(a["lags"]) == 4
    assert (int(a["chains"]), int(a["draws"]), int(a["burnin"])) == (4, 50_000, 10_000)
    assert a["call"].startswith("spec <- specify_bvar$new(data = data, p = 4L")
    assert "burn <- estimate(spec, S = 10000L" in a["call"]
    assert "post <- estimate(burn, S = 50000L" in a["call"]
    assert list(ds["regressor"].values) == regressors and list(ds["equation"].values) == variables
    assert ds["B_q"].shape == (13, 3, 5) and ds["Sigma_q"].shape == (6, 5)
    assert ds["forecast_q"].shape == (4, 3, 5)
    # the chained state runs' summaries carry dotted prefixes (h.h_*, state.state_*), as
    # documented in SV_PARAMETRIZATION.md §7.3
    assert {"h.h", "state.state"} <= summary_prefixes(ds)
    if "does not move V" in fixture(PROBE_WB).attrs["P12_answer"]:  # P12's recorded finding
        np.testing.assert_array_equal(ds["V_sd"].values, 0.0)
    assert int(a["extraction_pass"]) in (0, 1)  # chained state runs vs the main runs, recorded


def test_qd3_growth_is_the_transform_of_the_fred_qd_small_set():
    data = frame("qd3_growth.csv")
    fresh = generator().generate("qd3_growth")  # deterministic: no random numbers involved
    assert list(data.columns) == ["date", "dGDPC1", "dGDPCTPI", "FEDFUNDS"]
    assert list(fresh["date"]) == list(data["date"]) and data["date"].iloc[0] == "1960Q2"
    np.testing.assert_allclose(
        fresh.drop(columns="date").to_numpy(),
        data.drop(columns="date").to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_stochvol_t5_regenerated_with_a_fixed_stochvol():
    """P2 §2.10 Stage 4 amendment 1: the SV-t fixture comes from stochvol >= 3.2.5 and carries
    stochvol's answer on the four-observation SV-t problem of SV_PARAMETRIZATION.md §3.2."""
    a = fixture("stochvol_v_t5").attrs
    assert tuple(int(x) for x in a["stochvol_version"].split(".")) >= (3, 2, 5)
    for part in ("mu", "h2", "tau2"):
        q = np.asarray(a[f"check_4obs_{part}_q"])
        exact = np.asarray(a[f"check_4obs_exact_{part}_q"])
        assert q.shape == exact.shape == (5,) and np.all(np.diff(q) > 0)
    assert "check_4obs_error" not in a


def test_stochvol_t5_provenance():
    """The SV-t fixture keeps the Phase 2 attribute set (its design is Phase 2's) plus the
    Phase 3 record: R and stochvol versions, the snapshot, the packages and the CI run (D-54)."""
    a = fixture("stochvol_v_t5").attrs
    assert a["design"].startswith("docs/phase2/DESIGN.md") and "design_p3" in a
    assert a["r_version"].startswith(P3_R_VERSION) and a["stochvol_version"] == "3.2.9"
    assert a["ppm_snapshot"] == P3_SNAPSHOT
    assert f"/{P3_SNAPSHOT}" in a["repos"] and "packagemanager.posit.co" in a["repos"]
    assert "stochvol 3.2.9" in a["packages"]
    check_run_provenance(a)
