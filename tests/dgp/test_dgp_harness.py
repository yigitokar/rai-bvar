"""The DGP-test harness of SPEC §10.5 (``rai_bvar.validation.dgp``), without MCMC fits.

The simulators (validity checks, reproducibility, the sample is the simulated ``y``, the volatility
and outlier processes have their stated moments and reach the errors, constant-volatility errors
have the implied covariance), the row registry (arms, baselines and pairs consistent; pairs differ
along one axis; the arms use the library defaults; Phase 4–5 rows pending in this build, and not
pre-registered), the seeds (rows on one DGP share data), and the summaries (Monte Carlo errors,
Wilson intervals, the verdict classes and their consistency checks against the gate-off and
gate-on verdicts, the family-wise rate). One end-to-end replication runs the real ``compare()``
pipeline on two analytic arms, so the record layout and :func:`dgp.classify` see the engine's own
verdict frames.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from datetime import UTC, datetime

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.api import BVAR
from rai_bvar.models.base import spec_hash
from rai_bvar.settings import MCMCSettings
from rai_bvar.validation import dgp

DESIGN = dgp.Design()
RUNNABLE = [r for r in dgp.ROWS if dgp.row_status(r) is None]


# =========================================================================== simulators


def test_import_needs_no_jax():
    import subprocess

    code = "import sys, rai_bvar.validation.dgp; print('jax' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("name", [k for k, v in dgp.DGPS.items() if v is not None])
def test_registered_dgps_are_stationary_var2_in_three_variables(name):
    d = dgp.DGPS[name]
    assert d.name == name and (d.n, d.lags) == (3, 2)
    assert d.spectral_radius() == pytest.approx(0.8227, abs=1e-4)
    C = d.companion()
    assert C.shape == (6, 6) and np.array_equal(C[3:, :3], np.eye(3))


def test_phase2_dgps_differ_only_in_volatility():
    base = dgp.DGPS["linear-homoskedastic"].describe()
    sv = dgp.DGPS["linear-sv"].describe()
    svo = dgp.DGPS["linear-sv-outliers"].describe()
    same = ("intercept", "lag_coefs", "contemp", "log_var", "burn", "start", "variables")
    for key in same:
        assert base[key] == sv[key] == svo[key], key
    assert (base["vol"], sv["vol"], svo["vol"]) == ("constant", "sv", "sv")
    assert (sv["phi"], sv["sigma"]) == (svo["phi"], svo["sigma"]) == (0.95, 0.2)
    assert (base["outlier_prob"], sv["outlier_prob"]) == (0.0, 0.0)
    # the prior mean of SV(outliers=True) on quarterly data: one outlier every 4 years
    assert svo["outlier_prob"] == 1 / 16 and svo["outlier_max"] == SVO_K


SVO_K = rb.SV(outliers=True).outlier_max  # the model's outlier grid {1, …, K}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"contemp": ((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}, "unit lower"),
        ({"contemp": ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}, "unit lower"),
        ({"log_var": (0.0, 0.0)}, "log_var"),
        ({"vol": "garch"}, "vol must be"),
        ({"vol": "sv", "phi": 1.0}, "phi"),
        ({"outlier_prob": 1.0}, "outlier_prob"),
        ({"lag_coefs": (((1.0, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),)}, "stationary"),
    ],
)
def test_invalid_dgps_are_rejected(change, match):
    base = dataclasses.asdict(dgp.DGPS["linear-homoskedastic"])
    with pytest.raises(ValueError, match=match):
        dgp.LinearVARDGP(**{**base, **change})


@pytest.mark.parametrize("name", [k for k, v in dgp.DGPS.items() if v is not None])
def test_simulation_is_reproducible_and_the_sample_is_y(name):
    d = dgp.DGPS[name]
    a = d.simulate_arrays(200, dgp.data_rng(DESIGN, name, 3))
    b = d.simulate_arrays(200, dgp.data_rng(DESIGN, name, 3))
    c = d.simulate_arrays(200, dgp.data_rng(DESIGN, name, 4))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["y"], c["y"])
    data = d.simulate(200, dgp.data_rng(DESIGN, name, 3))
    assert len(data) == 200 and data.n_vars == 3 and data.freq == "Q-DEC"
    assert set(data.tcodes.values()) == {2}  # growth-rate semantics: own-lag prior mean 0
    assert str(data.first) == "1950Q2" and str(data.last) == "2000Q1"  # clear of 2020
    np.testing.assert_allclose(data.sample.to_numpy(), a["y"], rtol=0, atol=1e-10)
    assert data.fingerprint() == d.simulate(200, dgp.data_rng(DESIGN, name, 3)).fingerprint()


def test_rows_on_one_dgp_share_data_and_seeds():
    bart = dgp.get_row("linear-sv-bart")
    assert bart.dgp == dgp.get_row("linear-sv").dgp == "linear-sv"
    # the seeds depend on the DGP and the replication, never on the row
    a = dgp.data_rng(DESIGN, "linear-sv", 7).standard_normal(5)
    b = dgp.data_rng(DESIGN, bart.dgp, 7).standard_normal(5)
    np.testing.assert_array_equal(a, b)
    assert dgp.compare_seed(DESIGN, "linear-sv", 7) == dgp.compare_seed(DESIGN, bart.dgp, 7)
    seeds = {
        dgp.compare_seed(DESIGN, d, r)
        for d in ("linear-sv", "linear-sv-outliers")
        for r in range(50)
    }
    assert len(seeds) == 100


def test_constant_volatility_errors_have_the_implied_covariance():
    d = dgp.DGPS["linear-homoskedastic"]
    out = d.simulate_arrays(40_000, np.random.default_rng(0))
    assert np.all(out["h"] == np.asarray(d.log_var)) and np.all(out["o"] == 1.0)
    A = np.asarray(d.contemp)
    Ainv = np.linalg.inv(A)
    sigma = Ainv @ np.diag(np.exp(d.log_var)) @ Ainv.T
    np.testing.assert_allclose(np.cov(out["u"], rowvar=False), sigma, atol=0.02)
    # the recursion: y_t − c − B_1 y_{t−1} − B_2 y_{t−2} = u_t
    y, (B1, B2) = out["y"], np.asarray(d.lag_coefs)
    resid = y[2:] - np.asarray(d.intercept) - y[1:-1] @ B1.T - y[:-2] @ B2.T
    np.testing.assert_allclose(resid, out["u"][2:], atol=1e-10)
    np.testing.assert_allclose(y.mean(axis=0), d.mean(), atol=0.03)


def test_sv_paths_have_the_ar1_moments():
    d = dgp.DGPS["linear-sv"]
    h = d.simulate_arrays(60_000, np.random.default_rng(1))["h"]
    var = d.sigma**2 / (1 - d.phi**2)
    # the IAT of an AR(1) is (1 + φ)/(1 − φ) = 39: about 1,500 effective draws per equation
    np.testing.assert_allclose(h.mean(axis=0), d.log_var, atol=0.06)
    np.testing.assert_allclose(h.var(axis=0), var, rtol=0.12)
    lag1 = [np.corrcoef(h[1:, i], h[:-1, i])[0, 1] for i in range(3)]
    np.testing.assert_allclose(lag1, d.phi, atol=0.01)
    eta = (h[1:] - np.asarray(d.log_var)) - d.phi * (h[:-1] - np.asarray(d.log_var))
    np.testing.assert_allclose(eta.std(axis=0), d.sigma, rtol=0.02)


def _kurtosis(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return (x**4).mean(axis=0) / (x**2).mean(axis=0) ** 2


@pytest.mark.parametrize("name", ["linear-homoskedastic", "linear-sv", "linear-sv-outliers"])
def test_the_volatility_and_the_outliers_reach_the_errors(name):
    """The structural errors are ``e = o · exp(h/2) · ε`` with the returned ``h`` and ``o``.

    ``ε`` is the first draw of the documented draw order, so it is recovered from the seed, and
    ``A u`` must equal ``o exp(h/2) ε`` exactly. Unstandardised, the errors have the tails the
    mechanism implies: kurtosis 3 under constant volatility, 3 exp(σ²/(1 − φ²)) ≈ 4.5 under SV,
    far more with outliers.
    """
    d = dgp.DGPS[name]
    T = 50_000
    out = d.simulate_arrays(T, np.random.default_rng(3))
    eps = np.random.default_rng(3).standard_normal((d.burn + T, d.n))[d.burn :]
    e = out["u"] @ np.asarray(d.contemp).T
    np.testing.assert_allclose(e, out["o"] * np.exp(0.5 * out["h"]) * eps, rtol=0, atol=1e-12)
    z = e / (out["o"] * np.exp(0.5 * out["h"]))  # standardised: N(0, 1)
    np.testing.assert_allclose(z.var(axis=0), 1.0, atol=0.03)
    np.testing.assert_allclose(_kurtosis(z), 3.0, atol=0.15)
    kurt = _kurtosis(e)
    if d.outlier_prob:
        assert np.all(kurt > 20), kurt
    elif d.vol == "sv":
        assert np.all((kurt > 3.8) & (kurt < 6)), kurt
    else:
        np.testing.assert_allclose(kurt, 3.0, atol=0.15)


def test_outliers_have_the_stated_rate_and_scales():
    d = dgp.DGPS["linear-sv-outliers"]
    o = d.simulate_arrays(50_000, np.random.default_rng(2))["o"]
    hit = o > 1
    se = math.sqrt(d.outlier_prob * (1 - d.outlier_prob) / o.size)
    assert abs(hit.mean() - d.outlier_prob) < 4 * se
    scales = o[hit]
    assert set(np.unique(scales)) == set(range(2, d.outlier_max + 1))
    assert scales.mean() == pytest.approx((2 + d.outlier_max) / 2, abs=0.2)


# =========================================================================== registry


def test_phase2_rows_are_runnable_and_later_rows_pending():
    assert [r.name for r in RUNNABLE] == ["linear-homoskedastic", "linear-sv", "linear-sv-outliers"]
    assert {r.preregistration for r in RUNNABLE} == {dgp.PREREGISTRATION_P2}
    assert all(r.preregistration is None for r in dgp.ROWS if r not in RUNNABLE)
    pending = {r.name: dgp.row_status(r) for r in dgp.ROWS if dgp.row_status(r)}
    assert set(pending) == {
        "linear-sv-bart",
        "threshold-sv-bart",
        "threshold-sv-threshold",
        "tvp-sv",
    }
    assert "BART" in pending["linear-sv-bart"] and "Phase 4" in pending["linear-sv-bart"]
    assert "threshold-sv" in pending["threshold-sv-bart"]  # no simulator yet
    assert "Phase 5" in pending["tvp-sv"] and "TVP" in pending["tvp-sv"]
    assert all(s.endswith("a pre-registration") for s in pending.values())
    with pytest.raises(NotImplementedError, match="pending"):
        dgp.run_replication(dgp.get_row("linear-sv-bart"), 0, DESIGN)
    with pytest.raises(KeyError):
        dgp.get_row("no-such-row")


def test_a_row_this_build_can_run_stays_pending_until_it_is_preregistered(monkeypatch):
    row = dgp.get_row("linear-sv-bart")  # its DGP exists; it needs BART and a pre-registration
    assert dgp.row_missing(row) == ["BART"]
    monkeypatch.setattr(rb, "BART", object, raising=False)  # a stand-in for a Phase 4 build
    assert dgp.row_missing(row) == []
    assert dgp.row_status(row) == "pending (Phase 4): needs a pre-registration"
    registered = dataclasses.replace(row, preregistration="benchmarks/reports/some_row.md")
    assert dgp.row_status(registered) is None


@pytest.mark.parametrize("row", dgp.ROWS, ids=lambda r: r.name)
def test_rows_are_consistent(row):
    assert row.dgp in dgp.DGPS
    assert row.baseline in row.arms and len(set(row.arms)) == len(row.arms)
    assert all(a in dgp.ARMS for a in row.arms)
    for pair in row.pairs:
        assert {pair.model, pair.reference} <= set(row.arms)
        assert pair.expect in dgp.TARGET
        # verdict(axis=…) pairs each arm with its neighbour at the baseline's value
        assert pair.reference == row.baseline


@pytest.mark.parametrize("row", RUNNABLE, ids=lambda r: r.name)
def test_pairs_differ_along_their_axis_only_with_library_defaults(row):
    models = dgp.models(row, DESIGN)
    for pair in row.pairs:
        m, r = models[pair.model].axes, models[pair.reference].axes
        assert sorted(k for k in m if m[k] != r[k]) == [pair.axis]
    for name, model in models.items():
        assert isinstance(model, BVAR) and model.engine == "mcmc" and model.lags == DESIGN.lags
        assert model.mean.prior.form == "structural"  # Minnesota(form="auto") resolved
        assert model.mean.prior == dataclasses.replace(rb.Minnesota(), form="structural")
        # the per-composition default budget (D-35), nothing set by the harness
        sv = isinstance(model.vol, rb.SV)
        assert model.mcmc == MCMCSettings().with_default_draws(sv=sv), name
        assert model.mcmc.draws == (10_000 if sv else 2_000)


def test_registered_arms_have_the_registered_spec_hashes():
    """The hashes listed in the pre-registration (and checked by the study's label)."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "benchmarks" / "dgp_study.py"
    spec = importlib.util.spec_from_file_location("dgp_study_hashes", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dgp_study_hashes"] = module
    try:
        spec.loader.exec_module(module)
        expected = module.REGISTERED_SPEC_HASHES
    finally:
        sys.modules.pop("dgp_study_hashes", None)
    got = {}
    for row in RUNNABLE:
        for name, model in dgp.models(row, DESIGN).items():
            got[name] = spec_hash(model)[:16]
    assert got == expected


def test_pending_arms_name_the_missing_component():
    with pytest.raises(NotImplementedError, match=r"rai_bvar\.BART"):
        dgp.ARMS["BART × SV"](2, None)


def test_design_key_and_origins():
    assert dgp.Design().key() == DESIGN.key()
    for change in (
        {"T": 201},
        {"window": 100},
        {"seed": 1},
        {"horizons": (1,)},
        {"mcmc": MCMCSettings(draws=100)},
    ):
        assert dataclasses.replace(DESIGN, **change).key() != DESIGN.key(), change
    data = dgp.DGPS["linear-sv"].simulate(DESIGN.T, dgp.data_rng(DESIGN, "linear-sv", 0))
    origins = DESIGN.origins(data).resolve(data.index)
    assert len(origins) == DESIGN.T - DESIGN.window == 80
    assert origins[0] == data.index[DESIGN.window - 1] and origins[-1] == data.index[-2]


def test_verdict_defaults_are_read_from_the_engine():
    assert dgp.VERDICT_DEFAULTS == {
        "test_level": 0.10,
        "mcs_alpha": 0.25,
        "mc_multiple": 3.0,
        "min_converged": 0.95,
    }
    assert dgp.NOMINAL == 0.10


# =========================================================================== summaries


def test_mc_error_and_wilson_interval():
    assert dgp.mc_se(10, 100) == pytest.approx(0.03)
    assert dgp.mc_se(0, 50) == 0.0 and math.isnan(dgp.mc_se(0, 0))
    lo, hi = dgp.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.236593, 0.763407), abs=1e-6)
    lo, hi = dgp.wilson(0, 24)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.137976, abs=1e-6)
    lo, hi = dgp.wilson(20, 200)
    assert (lo, hi) == pytest.approx((0.065670, 0.149406), abs=1e-6)
    assert all(math.isnan(x) for x in dgp.wilson(0, 0))


def test_wilson_coverage():
    rng = np.random.default_rng(5)
    for p, n in ((0.1, 24), (0.1, 200), (0.8, 50)):
        k = rng.binomial(n, p, size=4000)
        cover = np.mean([lo <= p <= hi for lo, hi in (dgp.wilson(int(x), n) for x in k)])
        assert 0.92 < cover < 0.98, (p, n, cover)


def test_readings():
    assert dgp.reading("better", 3, 24) == "descriptive (power)"
    assert dgp.reading("not_better", 2, 24) == "not above nominal"
    assert dgp.reading("not_better", 9, 24) == "above nominal"  # Wilson lower bound 0.21
    assert dgp.wilson(30, 200)[0] == pytest.approx(0.1071, abs=1e-4)  # 0.15 is above 0.10
    assert dgp.reading("not_worse", 30, 200) == "above nominal"
    assert dgp.reading("not_worse", 25, 200) == "not above nominal"  # 0.125: [0.086, 0.178]
    assert dgp.reading("not_worse", 0, 0) == "no replications"


def _pair(**kw):
    base = {
        "model": "M",
        "reference": "R",
        "axis": "vol",
        "expect": "not_better",
        "horizon": 1,
        "verdict": "inconclusive",
        "verdict_gate_off": "inconclusive",
        "dlpd": 1.0,
        "dlpd_mcse": 1.0,
        "dm_pvalue_better": 0.5,
        "dm_pvalue_worse": 0.5,
        "tests_valid": True,
        "model_in_mcs": True,
        "reference_in_mcs": True,
        "origins": 80,
        "converged_model": 1.0,
        "converged_reference": 1.0,
        "reasons": "",
    }
    return base | kw


@pytest.mark.parametrize(
    "kw, cls",
    [
        ({}, "imprecise"),
        ({"dlpd": 5.0}, "no_route"),
        (
            {
                "dlpd": 5.0,
                "dm_pvalue_better": 0.05,
                "verdict": "better",
                "verdict_gate_off": "better",
            },
            "better:test",
        ),
        (
            {
                "dlpd": 5.0,
                "reference_in_mcs": False,
                "verdict": "better",
                "verdict_gate_off": "better",
            },
            "better:mcs",
        ),
        (
            {
                "dlpd": 5.0,
                "dm_pvalue_better": 0.01,
                "reference_in_mcs": False,
                "verdict": "better",
                "verdict_gate_off": "better",
            },
            "better:both",
        ),
        (
            {
                "dlpd": -5.0,
                "dm_pvalue_worse": 0.05,
                "verdict": "worse",
                "verdict_gate_off": "worse",
            },
            "worse:test",
        ),
        (
            {"dlpd": -5.0, "model_in_mcs": False, "verdict": "worse", "verdict_gate_off": "worse"},
            "worse:mcs",
        ),
        # the gate made it inconclusive
        (
            {
                "dlpd": 5.0,
                "dm_pvalue_better": 0.05,
                "verdict_gate_off": "better",
                "converged_model": 0.9,
            },
            "gate",
        ),
        # the gate binds on the reference arm as well
        (
            {
                "dlpd": 5.0,
                "dm_pvalue_better": 0.05,
                "verdict_gate_off": "better",
                "converged_reference": 0.9,
            },
            "gate",
        ),
        # a precise gain whose DM p-value is in the wrong direction is not a "better"
        ({"dlpd": 5.0, "dm_pvalue_worse": 0.01}, "no_route"),
        # an expanding window: the test route is closed
        ({"dlpd": 5.0, "dm_pvalue_better": 0.01, "tests_valid": False}, "no_route"),
    ],
)
def test_classify(kw, cls):
    assert dgp.classify(_pair(**kw)) == cls


def test_classify_rejects_a_misread_rule():
    with pytest.raises(RuntimeError, match="derived gate-off verdict"):
        dgp.classify(_pair(dlpd=5.0, verdict_gate_off="better"))  # no route held
    better = {"dlpd": 5.0, "dm_pvalue_better": 0.05, "verdict_gate_off": "better"}
    with pytest.raises(RuntimeError, match="derived gate-on verdict"):
        dgp.classify(_pair(**better, verdict="better", converged_model=0.5))  # the gate binds
    with pytest.raises(RuntimeError, match="derived gate-on verdict"):
        dgp.classify(_pair(**better, verdict="better", converged_reference=0.5))  # on both arms
    with pytest.raises(RuntimeError, match="derived gate-on verdict"):
        dgp.classify(_pair(**better, verdict="worse"))  # gate passed: the gate-off verdict
    with pytest.raises(RuntimeError, match="derived gate-on verdict"):
        dgp.classify(_pair(**better))  # gate passed, yet inconclusive
    # the gate is "at least": a share exactly at min_converged passes
    gate = dgp.VERDICT_DEFAULTS["min_converged"]
    assert dgp.classify(_pair(**better, verdict="better", converged_reference=gate)) == (
        "better:test"
    )


def _record(row, rep, pairs, conv=1.0, seconds=10.0):
    return {
        "row": row,
        "rep": rep,
        "design": DESIGN.key(),
        "seconds": seconds,
        "arms": {
            "M": {"seconds": 8.0, "cached": 0, "origins": 80, "converged": {"1": conv}},
            "R": {"seconds": 2.0, "cached": 0, "origins": 80, "converged": {"1": 1.0}},
        },
        "pairs": pairs,
    }


def test_summarise_family_and_convergence():
    better = _pair(dlpd=5.0, dm_pvalue_better=0.05, verdict="better", verdict_gate_off="better")
    gated = better | {"verdict": "inconclusive", "converged_model": 0.9}
    recs = [_record("linear-homoskedastic", i, [better]) for i in range(3)]
    recs += [_record("linear-homoskedastic", i, [gated], conv=0.9) for i in range(3, 5)]
    recs += [_record("linear-homoskedastic", i, [_pair()]) for i in range(5, 20)]
    (s,) = dgp.summarise(recs)
    assert (s["row"], s["model"], s["reference"], s["horizon"], s["n"]) == (
        "linear-homoskedastic",
        "M",
        "R",
        1,
        20,
    )
    assert (s["expect"], s["target"]) == ("not_better", "better")
    assert s["rate"] == pytest.approx(3 / 20) and s["se"] == pytest.approx(dgp.mc_se(3, 20))
    assert (s["lo"], s["hi"]) == pytest.approx(dgp.wilson(3, 20))
    assert s["reading"] == "not above nominal"
    assert [s["gate_on"][v]["k"] for v in dgp.VERDICTS] == [3, 0, 17]
    assert [s["gate_off"][v]["k"] for v in dgp.VERDICTS] == [5, 0, 15]
    assert s["classes"] == {"better:test": 3, "gate": 2, "imprecise": 15}
    assert s["gate_pass"] == pytest.approx(18 / 20) and s["converged_mean"] == pytest.approx(0.99)
    (f,) = dgp.family(recs, "linear-homoskedastic")
    assert (f["verdict"]["k"], f["verdict_gate_off"]["k"], f["verdict"]["n"]) == (3, 5, 20)
    conv = {(c["arm"], c["horizon"]): c for c in dgp.convergence(recs)}
    assert conv[("M", 1)]["gate_pass"] == pytest.approx(18 / 20)
    assert conv[("M", 1)]["converged_min"] == 0.9
    assert conv[("M", 1)]["seconds_per_origin"] == pytest.approx(0.1)


def test_warning_counts():
    by_message = {
        "MCMC not converged: 2 of 16 primary quantities fail R-hat < 1.01 or bulk ESS > 400 "
        "(worst: sigma[y3] R-hat 1.020, ESS 380; loglik R-hat 1.011, ESS 900)": [
            "1990Q1",
            "1991Q1",
        ],
        "MCMC not converged: 1 of 16 primary quantities fail (worst: loglik R-hat 1.012, ESS 500)": [
            "1992Q1"
        ],
        "secondary (informational; does not gate converged): 33% of a_ij": ["1990Q1"],
    }
    assert dgp.warning_counts(by_message) == {
        "warnings": {
            "MCMC not converged": 3,
            "secondary (informational; does not gate converged)": 1,
        },
        "worst": {"loglik": 1, "sigma[y3]": 2},
    }
    rec = _record("linear-sv", 0, [_pair()])
    rec["arms"]["M"] |= dgp.warning_counts(by_message)
    rec2 = _record("linear-sv", 1, [_pair()])
    rec2["arms"]["M"] |= {"worst": {"sigma[y3]": 1, "phi[y1]": 4}}
    (m,) = (c for c in dgp.convergence([rec, rec2]) if c["arm"] == "M")
    assert m["worst"] == {"phi[y1]": 4, "sigma[y3]": 3, "loglik": 1}


def test_family_counts_a_replication_once():
    b = _pair(dlpd=5.0, dm_pvalue_better=0.05, verdict="better", verdict_gate_off="better")
    rec = _record("linear-homoskedastic", 0, [b, b | {"model": "M2"}, _pair() | {"horizon": 4}])
    rec2 = _record("linear-homoskedastic", 1, [_pair(), _pair() | {"model": "M2"}])
    fam = {f["horizon"]: f for f in dgp.family([rec, rec2], "linear-homoskedastic")}
    assert fam[1]["verdict"]["k"] == 1 and fam[1]["verdict"]["n"] == 2
    assert fam[4]["verdict"]["k"] == 0


# =========================================================================== end to end


def test_one_replication_runs_the_real_pipeline(monkeypatch, tmp_path):
    """Two analytic arms that differ in their lags only (cheap): the record holds the engine's
    verdicts, classify() agrees with them, and a rerun is served from the cache unchanged."""
    arms = {
        "p1": lambda lags, mcmc: BVAR(lags=1),
        "p2": lambda lags, mcmc: BVAR(lags=2),
    }
    row = dgp.Row(
        "test-lags",
        "linear-homoskedastic",
        ("p1", "p2"),
        "p1",
        (dgp.Pair("p2", "p1", "lags", "not_worse"),),
        "test row",
    )
    monkeypatch.setattr(dgp, "ARMS", arms)
    # a seed of its own: no sample of the registered study is fitted by the tests
    design = dataclasses.replace(DESIGN, seed=7, origin_step=4, draws=200)
    # the start is taken before the sample is simulated, the end after compare()
    calls = []
    ticks = iter(f"2026-10-28T12:00:{i:02d}+00:00" for i in range(10))
    data_rng, compare = dgp.data_rng, dgp.compare

    def now():
        calls.append("now")
        return next(ticks)

    def simulating(*args):
        calls.append("simulate")
        return data_rng(*args)

    def comparing(*args, **kwargs):
        calls.append("compare")
        return compare(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(dgp, "utc_now", now)
        m.setattr(dgp, "data_rng", simulating)
        m.setattr(dgp, "compare", comparing)
        rec = dgp.run_replication(row, 0, design, cache=tmp_path)
    assert calls == ["now", "simulate", "compare", "now"]
    assert (rec["started_at"], rec["finished_at"]) == (
        "2026-10-28T12:00:00+00:00",
        "2026-10-28T12:00:01+00:00",
    )
    assert (rec["row"], rec["rep"], rec["design"]) == ("test-lags", 0, design.key())
    assert set(rec["arms"]) == {"p1", "p2"}
    assert all(a["origins"] == 20 and a["cached"] == 0 for a in rec["arms"].values())
    assert rec["arms"]["p2"]["spec_hash"] == spec_hash(BVAR(lags=2))
    assert [p["horizon"] for p in rec["pairs"]] == [1, 4]
    for p in rec["pairs"]:
        assert p["verdict"] in dgp.VERDICTS and p["verdict"] == p["verdict_gate_off"]
        # origins 1980Q1, 1981Q1, …, 1999Q1: the last one's h = 4 outcome is 2000Q1, the last
        assert p["tests_valid"] and p["origins"] == 20
        assert p["converged_model"] == p["converged_reference"] == 1.0
        dgp.classify(p)  # consistent with the engine's verdict
    before = datetime.now(UTC).replace(microsecond=0)
    again = dgp.run_replication(row, 0, design, cache=tmp_path)
    started, finished = (datetime.fromisoformat(again[k]) for k in ("started_at", "finished_at"))
    assert started.utcoffset().total_seconds() == 0  # UTC
    assert before <= started <= finished <= datetime.now(UTC)
    assert all(a["cached"] == 20 for a in again["arms"].values())
    assert again["pairs"] == rec["pairs"] and again["data_fingerprint"] == rec["data_fingerprint"]
