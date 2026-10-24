"""The composition-aware default MCMC budget and the SV budget warning (DESIGN D-35; Q14).

* ``MCMCSettings.draws=None`` (the default) resolves in ``BVAR`` to 10,000 kept draws per chain for
  every stochastic-volatility composition (``SV``, robust SV with outliers, Student-t, random walk;
  structural or reduced form) and to 2,000 for ``Constant`` volatility; an explicit ``draws`` is
  kept as given, and an analytic composition keeps ``mcmc=None``.
* Hashes: every ``Constant`` MCMC composition keeps its pre-D-35 spec hash and name (values
  recorded at ``ae5f8d6``); a default-constructed SV composition changes hash (the one documented
  change), and an SV composition with an explicit 4 × 2,000 budget hashes exactly as the old
  default did, so its cached backtests are still served.
* Warning: an SV composition fitted with fewer than 10,000 kept draws per chain warns
  (``fit.warnings`` and a ``UserWarning``); never for ``Constant`` volatility, the analytic engine,
  the SV default or an explicit budget at or above the threshold.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import rai_bvar as rb
from rai_bvar.api import default_mcmc
from rai_bvar.engines.mcmc import monitor, resolve
from rai_bvar.models import spec_hash
from rai_bvar.settings import DEFAULT_DRAWS, SV_DEFAULT_DRAWS, default_draws

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _bvar(vol, form="auto", lags=1, mcmc=None, **prior):
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form=form, **prior)), vol=vol, lags=lags, mcmc=mcmc
    )


SV_COMPOSITIONS = {
    "SV": ("auto", rb.SV()),
    "SV(outliers)": ("structural", rb.SV(outliers=True)),
    "SV(t)": ("structural", rb.SV(tails="t")),
    "SV(outliers,t)": ("structural", rb.SV(outliers=True, tails="t")),
    "SV(rw)": ("structural", rb.SV(rw=True)),
    "reduced-SV": ("reduced", rb.SV()),
    "reduced-SV(rw)": ("reduced", rb.SV(rw=True)),
}
CONSTANT_COMPOSITIONS = {
    "Constant(ig)": ("auto", rb.Constant("ig")),
    "structural-auto": ("structural", rb.Constant()),
    "Constant(iw)": ("structural", rb.Constant("iw")),
    "Constant(iw_equivalent)": ("structural", rb.Constant("iw_equivalent")),
    "reduced-Constant(ig)": ("reduced", rb.Constant("ig")),
    "reduced-Constant(iw_equivalent)": ("reduced", rb.Constant("iw_equivalent")),
}

# spec hash and name of default-constructed MCMC models before D-35 (recorded at ``ae5f8d6``)
PRE_D35 = {
    "Constant(ig)": (
        lambda: _bvar(rb.Constant("ig")),
        "74be7bb0b99be6d71e26e488ed28d07c1a94c0518902da370f7500bf7939696d",
        "BVAR(Minnesota(structural, eb), Constant(ig), p=1)",
    ),
    "Constant(iw)": (
        lambda: _bvar(rb.Constant("iw"), "structural"),
        "1145858750a13688a60e915ba10fdc726a9cf1d19b86f22e9a5e45f19ab4350e",
        "BVAR(Minnesota(structural, eb), Constant(iw), p=1)",
    ),
    "Constant(iw_equivalent)": (
        lambda: _bvar(rb.Constant("iw_equivalent"), "structural"),
        "95424b77e205a09865bd4c400e837a60eeb84e4436277af7661401fbcdfc3c70",
        "BVAR(Minnesota(structural, eb), Constant(iw_equivalent), p=1)",
    ),
    "reduced-Constant(ig)": (
        lambda: _bvar(rb.Constant("ig"), "reduced"),
        "a74a50871ed34df55f21f17c1f885ce4e7af5f30b9f60c6f5ad8854a91570c93",
        "BVAR(Minnesota(reduced, eb), Constant(ig), p=1)",
    ),
    "reduced-Constant(iw_equivalent)-p4": (
        lambda: _bvar(rb.Constant("iw_equivalent"), "reduced", lags=4),
        "75816c9232345d659d34db3cf76f22c52a9b20d0e0f49bc5c8006ec99dc559ef",
        "BVAR(Minnesota(reduced, eb), Constant(iw_equivalent), p=4)",
    ),
}
PRE_D35_SV = {  # the old SV defaults (4 × 2,000): now reached only through an explicit budget
    "SV": (
        lambda **kw: _bvar(rb.SV(), **kw),
        "876e6ced508b11e9ebb0ff849f4a8053ee1d185fbbf7f06dacb9c033e496a226",
        "BVAR(Minnesota(structural, eb), SV, p=1)",
    ),
    "SV(outliers)-p4": (
        lambda **kw: _bvar(rb.SV(outliers=True), lags=4, **kw),
        "3f394dbe0a58ce3aa5b452edb84feccc96fd023dcd4b73f48d6f795206a722d3",
        "BVAR(Minnesota(structural, eb), SV(outliers), p=4)",
    ),
    "SV(outliers,t)": (
        lambda **kw: _bvar(rb.SV(outliers=True, tails="t"), **kw),
        "7c1d614fafc8267445e2af77cfe7751753c7095a00097da590eb588f6a627586",
        "BVAR(Minnesota(structural, eb), SV(outliers,t), p=1)",
    ),
    "reduced-SV(rw)": (
        lambda **kw: _bvar(rb.SV(rw=True), "reduced", **kw),
        "d3951b69ba3d13f37e4beb6df7182875f8a08780ac4243a6d24585812631a934",
        "BVAR(Minnesota(reduced, eb), SV(rw), p=1)",
    ),
}
OLD_DEFAULT = rb.MCMCSettings(draws=2000)  # 4 chains × 2,000, burn 1000: the pre-D-35 default


# =========================================================================== default resolution


def test_default_draws_constants():
    assert (SV_DEFAULT_DRAWS, DEFAULT_DRAWS) == (10_000, 2_000)
    assert default_draws(sv=True) == 10_000 and default_draws(sv=False) == 2_000
    assert rb.MCMCSettings().draws is None and rb.MCMCSettings().chains == 4
    assert default_mcmc(rb.SV()) == rb.MCMCSettings(draws=10_000)
    assert default_mcmc(rb.Constant("ig")) == rb.MCMCSettings(draws=2_000)


@pytest.mark.parametrize(("form", "vol"), SV_COMPOSITIONS.values(), ids=SV_COMPOSITIONS.keys())
def test_sv_compositions_default_to_four_by_ten_thousand(form, vol):
    model = _bvar(vol, form)
    assert model.engine == "mcmc"
    assert model.mcmc == rb.MCMCSettings(chains=4, draws=10_000)
    assert model.spec()["mcmc"]["draws"] == 10_000  # the spec records the resolved budget
    assert model.mcmc.store_thin == 4  # ⌈10000 / store_max 2500⌉


@pytest.mark.parametrize(
    ("form", "vol"), CONSTANT_COMPOSITIONS.values(), ids=CONSTANT_COMPOSITIONS.keys()
)
def test_constant_compositions_keep_four_by_two_thousand(form, vol):
    model = _bvar(vol, form)
    assert model.engine == "mcmc"
    assert model.mcmc == rb.MCMCSettings(chains=4, draws=2000) == OLD_DEFAULT
    assert model.spec()["mcmc"]["draws"] == 2000


def test_the_analytic_engine_is_unaffected():
    for settings in (None, rb.MCMCSettings(), rb.MCMCSettings(draws=50), {"seed": 3}):
        model = rb.BVAR(mcmc=settings)
        assert model.engine == "analytic" and model.mcmc is None
        assert "mcmc" not in model.spec()
        assert model == rb.BVAR()


def test_unset_draws_follow_the_composition_and_explicit_draws_are_kept():
    sv, constant = rb.SV(outliers=True), rb.Constant("ig")
    # other fields set, draws unset: still the composition's default
    assert _bvar(sv, mcmc=rb.MCMCSettings(seed=1)).mcmc == rb.MCMCSettings(draws=10_000, seed=1)
    assert _bvar(sv, mcmc={"chains": 2}).mcmc == rb.MCMCSettings(chains=2, draws=10_000)
    assert _bvar(constant, mcmc={"chains": 2}).mcmc == rb.MCMCSettings(chains=2, draws=2000)
    # an explicit budget is kept as given, below or above the default
    for draws in (500, 2000, 10_000, 20_000):
        assert _bvar(sv, mcmc=rb.MCMCSettings(draws=draws)).mcmc.draws == draws
        assert _bvar(constant, mcmc={"draws": draws}).mcmc.draws == draws


def test_the_resolved_budget_is_validated():
    """``draws=None`` skips the uint32 sweep bound until it is resolved; then the bound applies
    (``thin`` such that 10,000 kept draws exceed 2³² sweeps, 2,000 do not)."""
    settings = rb.MCMCSettings(thin=500_000)  # 1000 + 2000·thin < 2**32 < 10000·thin
    assert _bvar(rb.Constant("ig"), mcmc=settings).mcmc.draws == 2000
    with pytest.raises(ValueError, match="uint32 sweep counter"):
        _bvar(rb.SV(), mcmc=settings)


# =========================================================================== hashes and specs


@pytest.mark.parametrize("key", PRE_D35)
def test_constant_mcmc_models_keep_their_spec_hash_and_name(key):
    make, expected_hash, expected_name = PRE_D35[key]
    model = make()
    assert spec_hash(model) == expected_hash
    assert model.name == expected_name


@pytest.mark.parametrize("key", PRE_D35_SV)
def test_sv_defaults_change_hash_and_the_old_budget_reproduces_the_old_hash(key):
    """The documented D-35 change: a default SV model has a new spec (10,000 draws) and hence a new
    hash and backtest cache key; spelling out the old 4 × 2,000 budget gives back the old spec, so
    scores cached with it are still served (its name gains the settings suffix)."""
    make, old_hash, default_name = PRE_D35_SV[key]
    default, old = make(), make(mcmc=OLD_DEFAULT)
    assert spec_hash(default) != old_hash
    assert default.name == default_name  # the default budget is not named
    explicit = make(mcmc=rb.MCMCSettings(draws=10_000))  # BENCH-P2's budget: now the default
    assert explicit == default and explicit.name == default_name
    assert spec_hash(old) == old_hash
    assert old.name.startswith(default_name.removesuffix(")") + ", mcmc=")
    assert default.spec() == {**old.spec(), "mcmc": {**old.spec()["mcmc"], "draws": 10_000}}


@pytest.mark.parametrize(
    "model",
    [
        _bvar(rb.SV()),
        _bvar(rb.SV(rw=True), "reduced", mcmc=rb.MCMCSettings(seed=4)),
        _bvar(rb.SV(outliers=True), mcmc=OLD_DEFAULT),
        _bvar(rb.Constant("ig")),
        _bvar(rb.Constant("iw_equivalent"), "reduced", mcmc={"chains": 2}),
    ],
    ids=["sv-default", "sv-rw-seed", "sv-explicit-2000", "constant", "reduced-iweq"],
)
def test_resolved_budgets_round_trip_through_the_spec(model):
    rebuilt = rb.BVAR.from_spec(json.loads(json.dumps(model.spec())))
    assert rebuilt == model and spec_hash(rebuilt) == spec_hash(model)
    assert rebuilt.mcmc.draws == model.mcmc.draws and isinstance(rebuilt.mcmc.draws, int)


# =========================================================================== the warning


@pytest.fixture(scope="module")
def data2() -> rb.VARData:
    """``tests/integration/test_backtest_mcmc.sv_data()``: a bivariate VAR(1) with a variance
    shift, T = 64 (the fast tier's reduced × SV program, so the fit below reuses its compile)."""
    rng = np.random.default_rng(4)
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    y = np.zeros((64, 2))
    for t in range(1, 64):
        y[t] = y[t - 1] @ A.T + (0.4 if t < 32 else 1.2) * rng.standard_normal(2)
    index = pd.period_range("1990Q1", periods=64, freq="Q")
    return rb.VARData.from_frame(pd.DataFrame(y, index=index, columns=["a", "b"]))


SMALL = rb.MCMCSettings(chains=2, draws=60, burn=60, warm_burn=30)


def _reduced(vol, mcmc=SMALL):
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="reduced", hierarchy="fixed")),
        vol=vol,
        lags=1,
        mcmc=mcmc,
    )


def _budget_messages(messages):
    return [m for m in messages if "D-35" in m]


@pytest.mark.parametrize(
    ("vol", "mcmc", "expected"),
    [
        (rb.SV(), SMALL, (2, 60)),
        (rb.SV(outliers=True, tails="t"), OLD_DEFAULT, (4, 2000)),
        (rb.SV(rw=True), rb.MCMCSettings(draws=9_999), (4, 9_999)),
        (rb.SV(rw=True), rb.MCMCSettings(draws=2000, thin=5), (4, 2000)),  # kept draws count
        (rb.SV(), None, None),  # the SV default
        (rb.SV(outliers=True), rb.MCMCSettings(draws=10_000), None),  # explicit, at the default
        (rb.SV(tails="t"), rb.MCMCSettings(chains=2, draws=20_000), None),  # explicit, above it
        (rb.Constant("ig"), SMALL, None),
        (rb.Constant("iw_equivalent"), None, None),
    ],
    ids=[
        "sv-60",
        "svo-t-2000",
        "rw-9999",
        "rw-thinned",
        "sv-default",
        "svo-10000",
        "t-20000",
        "constant-60",
        "iweq-default",
    ],
)
def test_resolution_flags_short_sv_budgets_only(data2, vol, mcmc, expected):
    model = _reduced(vol, mcmc)
    r = resolve.resolve(model, data2, rb.Runtime())
    assert r.warning_inputs["sv_budget"] == expected
    messages = monitor.warning_catalogue(
        monitor.DiagnosticsInput(static=r.static, settings=model.mcmc, **r.warning_inputs)
    )
    budget = _budget_messages(messages)
    if expected is None:
        assert budget == []
    else:
        assert budget == [monitor.sv_budget_message(*expected)]


def test_the_message_names_the_evidence_and_the_remedy():
    message = monitor.sv_budget_message(4, 2000)
    assert message.startswith("stochastic-volatility fit with 4 × 2,000 kept draws")
    assert "fewer than the SV default of 10,000 per chain (D-35)" in message
    for evidence in ("0 of 11 origins", "68% of origins", "all 106", "4 of 101", "BENCH-P2"):
        assert evidence in message, evidence
    assert "mcmc=MCMCSettings(draws=10_000)" in message and "DESIGN Q14" in message
    # without the resolution step's input (engine-level fits) the catalogue adds nothing
    r = resolve.resolve(_reduced(rb.SV()), _toy_data(), rb.Runtime())
    inputs = {**r.warning_inputs, "sv_budget": None}
    unflagged = monitor.DiagnosticsInput(static=r.static, settings=rb.MCMCSettings(), **inputs)
    assert _budget_messages(monitor.warning_catalogue(unflagged)) == []


@pytest.mark.parametrize(
    ("vol", "expected"), [(rb.SV(), 10_000), (rb.Constant("ig"), 2000)], ids=["sv", "constant"]
)
def test_engine_level_fits_resolve_unset_draws(data2, vol, expected):
    """``fit_mcmc`` (and ``run_chains`` under it) resolve ``draws=None`` from ``static.vol`` as
    ``BVAR`` does; an engine-level fit has no resolution-step input, so no budget warning."""
    from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc

    r = resolve.resolve(_reduced(vol), data2, rb.Runtime())
    settings = rb.MCMCSettings(chains=2, burn=20)
    assert settings.draws is None
    meta = FitMeta(vardata=data2, periods=r.periods, variables=tuple(data2.variables))
    fit = fit_mcmc(
        r.static, r.data, r.prior, settings, rb.Runtime(), seed=0, salt=r.salt, meta=meta
    )
    assert fit.settings.draws == expected
    assert _budget_messages(fit.warnings) == []


def _toy_data() -> rb.VARData:
    rng = np.random.default_rng(0)
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    frame = pd.DataFrame(rng.standard_normal((40, 2)), index=index, columns=["a", "b"])
    return rb.VARData.from_frame(frame)


def test_a_short_sv_fit_warns_end_to_end(data2):
    """``BVAR.fit`` of an SV composition below the default budget: the message is in
    ``fit.warnings`` once and is raised as a ``UserWarning``; an analytic fit on the same data
    carries no budget message (Constant MCMC compositions: the resolution test above)."""
    with pytest.warns(UserWarning, match=r"stochastic-volatility fit with 2 × 60 kept draws"):
        fit = _reduced(rb.SV()).fit(data2)
    assert _budget_messages(fit.warnings) == [monitor.sv_budget_message(2, 60)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        analytic = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(hierarchy="fixed"))).fit(data2)
    assert analytic.engine == "analytic"
    assert _budget_messages(analytic.warnings) == []
