"""The predictive contract (SPEC §7) for the MCMC engine's compositions (DESIGN §13 WP17, §7).

The Phase 1 file ``test_forecast_contract.py`` is untouched; this one fits every Phase 2 MCMC
composition (structural Mode I / Mode II / IW-equivalent × Constant, structural × SV variants,
reduced × Constant / SV variants / SV(rw)) and checks the same invariants: structure, coordinates
and attrs, simulation consistency (``L⁻¹(paths − cond_mean)`` iid N(0, 1), conditional on the drawn
transitory scales), stacking, seed reproducibility, the pathwise level inverse, joint paths
(``cond_mean`` is the VAR mean of the simulated path) and the factorisation ``cond_chol =
A⁻¹ diag(cond_scale)``. Two compositions run in the fast tier; the others are ``slow``.

The fits come from ``tests/mcmc/forecast/mcmc_fits.py`` (a local double of WP18's ``fit_bvar``:
``hierarchy="fixed"``, the frozen prior and data builders, ``fit_mcmc``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rai_bvar import __version__
from rai_bvar.forecast import ForecastResult, check_structure, simulation_consistency
from rai_bvar.models import spec_hash

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _fits_module():
    """``tests/mcmc/forecast/mcmc_fits.py`` as the module ``mcmc_fits`` (shared, so fits are
    compiled once per process whichever test directory imports it first)."""
    if "mcmc_fits" in sys.modules:
        return sys.modules["mcmc_fits"]
    path = Path(__file__).resolve().parents[1] / "mcmc" / "forecast" / "mcmc_fits.py"
    spec = importlib.util.spec_from_file_location("mcmc_fits", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcmc_fits"] = module
    spec.loader.exec_module(module)
    return module


mf = _fits_module()

FAST = ("structural-SV(outliers)", "reduced-Constant(ig)")
H = 6  # > k = 4 regressors (lags = 1, constant): needed by the joint-path check
DRAWS = 100
SEED = 20261022


def _params():
    out = []
    for name in mf.COMPOSITIONS:
        marks = () if name in FAST else (pytest.mark.slow,)
        out.append(pytest.param(name, id=name, marks=marks))
    return out


@pytest.fixture(scope="module", params=_params())
def composition(request):
    return request.param


@pytest.fixture(scope="module")
def model(composition):
    if composition == "structural-SV(outliers)":
        return mf.shared_model()  # the fast-tier forecast tests' fit: compiled once
    form, vol = mf.COMPOSITIONS[composition]
    return mf.model(form, vol)


@pytest.fixture(scope="module")
def fitted(model, sim_data):
    return mf.cached_fit(model, sim_data)


@pytest.fixture(scope="module")
def forecast(fitted) -> ForecastResult:
    return fitted.forecast(H, draws=DRAWS, seed=SEED)


def test_engine_and_composition(model, fitted):
    assert model.engine == "mcmc" and fitted.engine == "mcmc"
    assert fitted.ordering == ("gdp", "prices", "rate")


def test_check_structure_passes(forecast):
    check_structure(forecast)


def test_coordinates_and_attributes(forecast, model, fitted, sim_data):
    ds = forecast.dataset
    C = fitted.settings.chains
    assert forecast.variables == list(sim_data.variables)
    assert forecast.horizon == H and forecast.origin == sim_data.last
    assert forecast.n_samples == C * -(-DRAWS // C)  # ⌈draws/C⌉ per chain (DESIGN §7.1)
    assert ds.sizes["chain"] == C
    assert list(ds["date"].values) == [str(sim_data.last + h) for h in range(1, H + 1)]
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        assert ds[name].dtype == np.float64
    attrs = ds.attrs
    assert attrs["model"] == model.name and attrs["engine"] == "mcmc"
    assert attrs["spec_hash"] == spec_hash(model)
    assert attrs["rai_bvar_version"] == __version__
    assert attrs["seed"] == str(SEED) and attrs["n_chains"] == C
    assert attrs["converged"] in (0, 1) and attrs["converged"] == int(fitted.converged)
    assert '"gdp": 5' in attrs["tcodes"] and attrs["ordering"] == '["gdp", "prices", "rate"]'
    assert attrs["transitory"] == fitted.static.vol.transitory
    for key in ("mixture", "exact", "prng_impl", "jax_version", "T_pad", "gram", "fit_seed"):
        assert key in attrs, key


def test_simulation_consistency(forecast):
    """Invariant 1: ``L⁻¹(paths − cond_mean)`` iid N(0, 1) (given the drawn scales)."""
    assert simulation_consistency(forecast) > 1e-3


def test_stacked_shapes(forecast):
    S = forecast.n_samples
    assert forecast.stacked().shape == (S, H, 3)
    assert forecast.stacked("cond_chol").shape == (S, H, 3, 3)
    assert forecast.stacked("paths_level").shape == (S, H, 3)


def test_same_seed_gives_identical_paths(fitted, forecast):
    again = fitted.forecast(H, draws=DRAWS, seed=SEED)
    for name in ("paths", "cond_mean", "cond_chol", "paths_level"):
        np.testing.assert_array_equal(again.dataset[name].values, forecast.dataset[name].values)
    gen = fitted.forecast(H, draws=DRAWS, seed=np.random.default_rng(SEED))
    np.testing.assert_array_equal(gen.stacked(), forecast.stacked())
    other = fitted.forecast(H, draws=DRAWS, seed=SEED + 1)
    assert not np.array_equal(other.stacked(), forecast.stacked())


def test_paths_level_is_the_pathwise_inverse(forecast, sim_data):
    np.testing.assert_allclose(
        forecast.stacked("paths_level"), sim_data.inverse_paths(forecast.stacked()), rtol=1e-14
    )


def test_joint_paths(forecast, sim_data):
    """Invariant 2: ``cond_mean[s, h] = B_s′x_{s,h}`` with ``x`` built from the data and the
    simulated path to ``T+h−1``; recovering ``B_s`` from all ``H > k`` horizons leaves no
    residual (paths assembled from marginals would fail)."""
    paths, mean = forecast.stacked(), forecast.stacked("cond_mean")
    S = paths.shape[0]
    values = sim_data.values
    full = np.concatenate([np.broadcast_to(values[-1:], (S, 1, 3)), paths], axis=1)
    X = np.stack(
        [np.concatenate([np.ones((S, 1)), full[:, h]], axis=1) for h in range(H)], axis=1
    )  # (S, H, k): constant, lag 1
    assert X.shape[2] < H
    coefs = np.linalg.pinv(X) @ mean
    scale = np.abs(mean).max(axis=(0, 1))
    np.testing.assert_allclose(np.abs(X @ coefs - mean).max(axis=(0, 1)) / scale, 0.0, atol=1e-10)


def test_cond_chol_is_a_inverse_times_scales(forecast):
    """``cond_chol = A⁻¹ diag(cond_scale)`` with the unit lower-triangular ``cond_A`` (DESIGN
    §7.3 accelerator variables, consistent with the contract field)."""
    A = forecast.stacked("cond_A")
    L = forecast.stacked("cond_chol")
    ell = forecast.stacked("cond_scale")
    np.testing.assert_allclose(
        np.einsum("sij,shjk->shik", A, L), ell[..., None] * np.eye(3), rtol=1e-12, atol=1e-14
    )


def test_log_pred_density(forecast):
    y = forecast.stacked("cond_mean")[:, 0].mean(axis=0)
    lpd, mcse = forecast.log_pred_density(y, 1, return_mcse=True)
    assert np.isfinite(lpd) and 0 <= mcse < 1.0
    for var in forecast.variables:
        j = forecast.variables.index(var)
        assert np.isfinite(forecast.log_pred_density(y[j : j + 1], 1, [var]))
