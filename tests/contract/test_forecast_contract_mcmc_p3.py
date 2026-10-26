"""The predictive contract (SPEC §7) for the new MCMC compositions of Phase 3 (docs/phase3/DESIGN.md
§9.11; §1.3 "tests/contract [WP35] the new MCMC compositions (fast: three; weekly: all)").

Phase 2's ``test_forecast_contract_mcmc.py`` is untouched; its invariants (structure,
coordinates and attrs, simulation consistency, stacking, seed reproducibility, the pathwise level
inverse, joint paths, ``cond_chol = A⁻¹ diag(cond_scale)``, the log predictive density) are
collected here unchanged, against Phase 3 fixtures: every new MCMC composition kind fitted
through the public ``BVAR(...).fit`` (``resolve.py`` builds the global-local and ``CommonSV``
arrays, which Phase 2's local fit double ``mcmc_fits.build_fit`` cannot), ``lags = 1`` with a
constant (so ``H = 6 > k = 4`` for the joint-path check), 2 chains × 60 draws after 60, on the
shared ``sim_data``.

**Compositions** (:data:`COMPOSITIONS_P3`): each global-local family and SSVS in both forms
(structural Mode I × ``Constant("ig")`` and reduced × ``SV``), the horseshoe with
``Constant("iw_equivalent")``, ``SV(outliers=True)``, ``SV`` (structural) and reduced ×
``Constant("ig")``, SSVS × ``SV``, and ``CommonSV`` with the conjugate prior (the shared-factor
path; also with its parity options and interior blocks), structural Mode I, the reduced form,
and × HS and × NG. The fast tier (§9.11) fits the three §9.11 names: structural × SV × HS,
conjugate × ``CommonSV`` and structural × ``CommonSV``; every other composition is ``slow``
(weekly).
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

import rai_bvar as rb
from rai_bvar import BVAR, SV, CommonSV, Constant, Linear, MCMCSettings, Minnesota
from rai_bvar.forecast import ForecastResult
from rai_bvar.priors.sv import Beta, ScaledChi2


def _phase2_contract():
    """Phase 2's ``test_forecast_contract_mcmc.py`` as a module (its invariant tests are
    collected here with this module's fixtures)."""
    name = "test_forecast_contract_mcmc"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_P2 = _phase2_contract()
H, DRAWS, SEED = _P2.H, _P2.DRAWS, _P2.SEED
SETTINGS = MCMCSettings(chains=2, draws=60, burn=60)

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _fixed(form: str) -> Minnesota:
    return Minnesota(form=form, hierarchy="fixed")


COMPOSITIONS_P3 = {
    # the fast tier (P3 §9.11)
    "structural-SV-HS": (rb.Horseshoe(form="structural"), SV()),
    "conjugate-CommonSV": (_fixed("conjugate"), CommonSV()),
    "structural-CommonSV": (_fixed("structural"), CommonSV()),
    # weekly: the global-local families and SSVS, both forms
    "structural-Constant(ig)-HS": (rb.Horseshoe(form="structural"), Constant("ig")),
    "structural-Constant(ig)-NG": (rb.NormalGamma(form="structural"), Constant("ig")),
    "structural-Constant(ig)-R2D2": (rb.R2D2(form="structural"), Constant("ig")),
    "structural-Constant(ig)-DL": (rb.DirichletLaplace(form="structural"), Constant("ig")),
    "structural-Constant(ig)-SSVS": (rb.SSVS(form="structural"), Constant("ig")),
    "structural-Constant(iw_equivalent)-HS": (
        rb.Horseshoe(form="structural"),
        Constant("iw_equivalent"),
    ),
    "structural-SV(outliers)-HS": (rb.Horseshoe(form="structural"), SV(outliers=True)),
    "structural-SV-SSVS": (rb.SSVS(form="structural"), SV()),
    "reduced-SV-HS": (rb.Horseshoe(form="reduced"), SV()),
    "reduced-SV-NG": (rb.NormalGamma(form="reduced"), SV()),
    "reduced-SV-R2D2": (rb.R2D2(form="reduced"), SV()),
    "reduced-SV-DL": (rb.DirichletLaplace(form="reduced"), SV()),
    "reduced-SV-SSVS": (rb.SSVS(form="reduced"), SV()),
    "reduced-Constant(ig)-HS": (rb.Horseshoe(form="reduced"), Constant("ig")),
    # weekly: CommonSV
    "conjugate-CommonSV(Beta,ScaledChi2,block=7)": (
        _fixed("conjugate"),
        CommonSV(rho_prior=Beta(10.0, 2.0), sigma2_prior=ScaledChi2(0.1), block=7),
    ),
    "reduced-CommonSV": (_fixed("reduced"), CommonSV()),
    "structural-CommonSV-HS": (rb.Horseshoe(form="structural"), CommonSV()),
    "reduced-CommonSV-NG": (rb.NormalGamma(form="reduced"), CommonSV()),
}
FAST_P3 = ("structural-SV-HS", "conjugate-CommonSV", "structural-CommonSV")


def _params():
    return [
        pytest.param(name, id=name, marks=() if name in FAST_P3 else (pytest.mark.slow,))
        for name in COMPOSITIONS_P3
    ]


@pytest.fixture(scope="module", params=_params())
def composition(request):
    return request.param


@pytest.fixture(scope="module")
def model(composition) -> BVAR:
    prior, vol = COMPOSITIONS_P3[composition]
    return BVAR(mean=Linear(prior=prior), vol=vol, lags=1, deterministic="c", mcmc=SETTINGS)


@pytest.fixture(scope="module")
def fitted(model, sim_data):
    with warnings.catch_warnings():  # short chains: convergence and budget warnings expected
        warnings.simplefilter("ignore", UserWarning)
        return model.fit(sim_data, seed=1)


@pytest.fixture(scope="module")
def forecast(fitted) -> ForecastResult:
    return fitted.forecast(H, draws=DRAWS, seed=SEED)


def test_composition_is_the_phase3_one(composition, model, fitted):
    """The fit runs the composition named: the MCMC engine, the global-local kind or the common
    volatility, and the conjugate screen's shared-factor path."""
    st = fitted.static
    prior, vol = COMPOSITIONS_P3[composition]
    assert model.engine == "mcmc" and fitted.engine == "mcmc"
    assert st.vol.common == isinstance(vol, CommonSV)
    assert st.coef.varies == (not isinstance(prior, Minnesota)), composition
    if isinstance(prior, Minnesota) and prior.form == "conjugate":
        assert st.mean.const_path == "shared_factor"
    else:
        assert st.mean.const_path != "shared_factor"
    form = "reduced" if composition.startswith("reduced") else "structural"
    assert st.mean.kind == f"linear_{form}"


# Phase 2's invariants, unchanged, with this module's fixtures
test_engine_and_composition = _P2.test_engine_and_composition
test_check_structure_passes = _P2.test_check_structure_passes
test_coordinates_and_attributes = _P2.test_coordinates_and_attributes
test_simulation_consistency = _P2.test_simulation_consistency
test_stacked_shapes = _P2.test_stacked_shapes
test_same_seed_gives_identical_paths = _P2.test_same_seed_gives_identical_paths
test_paths_level_is_the_pathwise_inverse = _P2.test_paths_level_is_the_pathwise_inverse
test_joint_paths = _P2.test_joint_paths
test_cond_chol_is_a_inverse_times_scales = _P2.test_cond_chol_is_a_inverse_times_scales
test_log_pred_density = _P2.test_log_pred_density
