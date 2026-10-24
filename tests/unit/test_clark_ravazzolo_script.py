"""Wiring of ``benchmarks/clark_ravazzolo.py`` (DESIGN §7.3, D-34; no data download, no fits).

The ``acceptance`` config is the pre-registered one and the existing configs keep their meaning;
every ``backtest`` / ``ordering_sensitivity`` call of the script passes ``cfg.lpd_method`` (a
static check of the source, and ``main()`` run end to end on synthetic data with recording
stand-ins for ``backtest`` and ``ordering_sensitivity`` whose scores have the real
``score_forecast`` layout); ``--lpd`` overrides the config; the report records the estimator and
the MCMC budget and, with the integrated estimator, shows the path-conditioned ΔLPS for the
criterion-4 cells. The criteria rules are not exercised here (their numbers are synthetic).

Provenance and labels (D-34 review, findings cr-1 to cr-4): the report records the commit and
whether tracked files differ from it, the cache and how many origins it served; it is labelled
the acceptance run only for the registered config, estimator and seed with every stage run on a
clean tree, its main-stage report is labelled pending, ``--note`` text always reaches the report
and the acceptance report cites its pre-registration. ``git_state`` is a stand-in (a clean tree
at a fixed commit) unless a test says otherwise, so the labels do not depend on this checkout.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import itertools
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import rai_bvar as rb
from rai_bvar import MCMCSettings, VARData
from rai_bvar.evaluation.backtest import BacktestResult, Expanding
from rai_bvar.evaluation.scores import score_forecast
from rai_bvar.forecast import ForecastResult
from rai_bvar.forecast.result import LAG_DIMS

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "clark_ravazzolo.py"
VARS = ["GDP", "INFL", "UR", "TBILL"]


@pytest.fixture(scope="module")
def cr():
    name = "clark_ravazzolo_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module's annotations through it
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


# =========================================================================== configs


def test_acceptance_config_and_the_existing_ones(cr):
    acc = cr.CONFIGS["acceptance"]
    assert cr.ACCEPTANCE == "acceptance" and acc.name == "acceptance"
    assert acc.mcmc == MCMCSettings(chains=4, draws=20000, burn=1000)
    assert (acc.draws, acc.ordering_step, acc.origin_step) == (8000, 1, 1)
    assert acc.noise_reps == cr.CONFIGS["full"].noise_reps == 1
    assert acc.lpd_method == "integrated"
    # the existing configs keep their meaning: the default estimator
    full = cr.CONFIGS["full"]
    assert full.mcmc == MCMCSettings(chains=4, draws=5000, burn=1000)
    assert (full.draws, full.ordering_step, full.origin_step) == (1000, 1, 1)
    # explicit 4 × 2000: the SV default budget of D-35 (4 × 10,000) does not reach the quick run
    assert cr.CONFIGS["quick"].mcmc == MCMCSettings(chains=4, draws=2000, burn=1000)
    for name in ("full", "quick", "smoke"):
        assert cr.CONFIGS[name].lpd_method == "conditioned", name
    assert cr.reliability_note("conditioned") == cr.RELIABILITY_NOTE
    assert cr.reliability_note("integrated") == cr.RELIABILITY_NOTE_INTEGRATED
    for note in (cr.RELIABILITY_NOTE, cr.RELIABILITY_NOTE_INTEGRATED):  # _c4_secondary cites it
        assert note.startswith("### Reliability of the RB log score\n")
    assert cr.RB_CEILING == 0.9


def test_every_backtest_and_ordering_call_passes_the_config_estimator():
    """Static: each call of ``backtest`` or ``ordering_sensitivity`` in the script passes
    ``lpd_method=cfg.lpd_method`` (the ordering stage's rebuild must, or it silently refits)."""
    tree = ast.parse(SCRIPT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        in ("backtest", "ordering_sensitivity")
    ]
    assert len(calls) == 5  # main stage, structural, --only-ordering, ordering stage, rebuild
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("lpd_method") == "cfg.lpd_method", (call.lineno, kw)


# =========================================================================== main() end to end


def synthetic_data() -> VARData:
    rng = np.random.default_rng(0)
    index = pd.period_range("1965Q1", "2013Q4", freq="Q")
    y = np.cumsum(rng.normal(0.0, 0.3, (len(index), 4)), axis=0) * 0.1 + [0.8, 0.9, 6.0, 4.5]
    return VARData.from_frame(pd.DataFrame(y, index=index, columns=VARS), 1)


def template_scores(lpd_method: str) -> xr.Dataset:
    """``score_forecast`` of a synthetic VAR(1) forecast with ``lag_coefs`` (h = 1..8, n = 4):
    the real score layout of either estimator."""
    rng = np.random.default_rng(1)
    S, H, n = 200, 8, 4
    A = 0.6 * np.eye(n) + rng.normal(0.0, 0.05, (S, n, n))
    y = np.broadcast_to(rng.normal(size=n), (S, n))
    paths, means = np.empty((S, H, n)), np.empty((S, H, n))
    for h in range(H):
        means[:, h] = np.einsum("sij,sj->si", A, y)
        paths[:, h] = means[:, h] + 0.5 * rng.standard_normal((S, n))
        y = paths[:, h]
    fr = ForecastResult.from_arrays(
        paths=paths, cond_mean=means, cond_chol=np.broadcast_to(0.5 * np.eye(n), (S, H, n, n)),
        variables=VARS, origin=pd.Period("2000Q1", "Q"),
        extra={"lag_coefs": (LAG_DIMS, A[:, None])},
    )  # fmt: skip
    truth = paths.mean(axis=0) + 0.3
    return score_forecast(fr, truth, rng=np.random.default_rng(0), lpd_method=lpd_method)


class Recorder:
    """Recording stand-ins for ``backtest`` and ``ordering_sensitivity``."""

    def __init__(self):
        self.backtests: list[dict] = []
        self.orderings: list[dict] = []
        self.templates = {m: template_scores(m) for m in ("conditioned", "integrated")}

    def backtest(self, model, data, *, origins, horizon, draws, seed, cache, n_jobs,
                 name=None, lpd_method="conditioned", **kw):  # fmt: skip
        name = name or model.name
        self.backtests.append({"name": name, "lpd_method": lpd_method, "origins": len(origins)})
        assert horizon == 8 and not kw, kw
        parts = []
        for o in origins:
            rng = np.random.default_rng([seed, zlib.crc32(f"{name}|{o}".encode())])
            ds = self.templates[lpd_method].copy(deep=True)
            for var in ds.data_vars:  # positive scores stay positive
                ds[var] = ds[var] * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, ds[var].shape))
            ds["converged"] = 1.0
            parts.append(ds.expand_dims(origin=[str(o)]))
            if hasattr(model, "directory"):  # MomentsProbe: the Gaussian-approximation moments
                path = model.directory() / f"{o}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                cov = np.stack([np.eye(4) * (1.0 + 0.1 * h) for h in range(8)])
                np.savez(path, mean=np.zeros((8, 4)), cov=cov, variables=np.array(VARS))
        scores = xr.concat(parts, dim="origin")
        return BacktestResult(
            name=name, axes=dict(model.axes), spec={}, window=Expanding(), scores=scores,
            seconds=0.0, cached=len(origins),
        )  # fmt: skip

    def ordering_sensitivity(self, model, data, **kw):
        self.orderings.append({"model": model.name, "lpd_method": kw.get("lpd_method")})
        assert kw["mode"] == "backtest" and kw["design"] == "all"
        orders = list(itertools.permutations(VARS))
        ds = xr.Dataset(
            {
                "joint": (("ordering", "horizon", "jstat"), np.zeros((24, 4, 1))),
                "converged": (("ordering",), np.ones(24)),
            },
            coords={"horizon": [1, 2, 4, 8], "jstat": ["lpd"]},
            attrs={"backtest_seeds": json.dumps(list(range(25)))},
        )
        summary = pd.DataFrame({"lpd_range": [0.01]}, index=pd.Index(["GDP"], name="variable"))
        return type(
            "Result", (), {"material": False, "dataset": ds, "orderings": orders,
                           "summary": staticmethod(lambda: summary)},
        )()  # fmt: skip


CLEAN = {"git_commit": "0123456789abcdef0123456789abcdef01234567", "git_dirty": False,
         "git_diff_sha256": None}  # fmt: skip


class Interrupted(Exception):
    """Raised by the ordering stand-in to stop ``main`` after the main-stage report."""


@pytest.fixture
def run_main(cr, tmp_path, monkeypatch):
    """``main(argv)`` on synthetic data; returns ``(recorder, report json, report markdown)``
    (``git`` replaces the clean-tree stand-in of ``git_state``; ``interrupt=True`` stops ``main``
    in the ordering stage and returns the main-stage report it left behind)."""
    data = synthetic_data()
    agg = {k: {"quarters": 260, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "first": "1959Q1",
               "last": "2026Q2"} for k in ("UNRATE", "TB3MS")}  # fmt: skip
    monkeypatch.setattr(cr, "load_data", lambda qd_url, md_url: (data, agg))
    monkeypatch.setattr(
        cr, "assert_in_window", lambda m, d, o: {"origins": list(o), "identical": True}
    )
    monkeypatch.setenv(
        "JAX_COMPILATION_CACHE_DIR", os.environ.get("JAX_COMPILATION_CACHE_DIR", str(tmp_path))
    )

    def run(*argv: str, git: dict | None = None, interrupt: bool = False):
        rec = Recorder()
        monkeypatch.setattr(cr, "backtest", rec.backtest)
        monkeypatch.setattr(rb, "ordering_sensitivity", rec.ordering_sensitivity)
        monkeypatch.setattr(cr, "git_state", lambda: dict(CLEAN if git is None else git))
        out = tmp_path / f"report{len(list(tmp_path.glob('*.json')))}"
        argv = [str(out), "--cache", str(tmp_path / "cache"), *argv]
        if interrupt:

            def stop(*args, **kwargs):
                raise Interrupted

            monkeypatch.setattr(rb, "ordering_sensitivity", stop)
            with pytest.raises(Interrupted):
                cr.main(argv)
        else:
            assert cr.main(argv) == 0
        if "--only-ordering" in argv:
            return rec, None, None
        return (
            rec,
            json.loads(out.with_suffix(".json").read_text()),
            out.with_suffix(".md").read_text(),
        )

    return run


def _expected_calls(rec, method: str) -> None:
    main_names = ["Constant", "SV(rw)", "SV", "SV(rw,t10)", "AR-SV(0.9)"]
    structural = ["Constant [structural]", "SV(rw) [structural]"]
    rebuild = ["Constant", "SV(rw)"] * 24
    assert [b["name"] for b in rec.backtests] == main_names + structural + rebuild
    assert [o["lpd_method"] for o in rec.orderings] == [method] * 2
    assert {b["lpd_method"] for b in rec.backtests} == {method}


def test_main_passes_the_override_to_every_stage(cr, run_main):
    rec, rep, md = run_main("--config", "smoke", "--lpd", "integrated")
    _expected_calls(rec, "integrated")
    assert rep["config"]["lpd_method"] == "integrated" and rep["config"]["name"] == "smoke"
    assert rep["config"]["mcmc"]["draws"] == 100
    assert "Primary LPD estimator: `lpd_method='integrated'`" in md
    assert "NOT the acceptance run" in md  # smoke, whatever the estimator
    assert cr.RELIABILITY_NOTE_INTEGRATED in md and cr.RELIABILITY_NOTE not in md
    assert cr.LPD_INTEGRATED_DEVIATION in rep["deviations"]
    c4 = next(c for c in rep["criteria"] if c["id"] == 4)
    assert "path-conditioned (SPEC §7) estimator ΔLPS: full UR: h1" in c4["detail"]
    row = rep["relative"]["full"]["SV(rw)"]["variables"]["UR"]["8"]
    assert np.isfinite(row["dlps_conditioned"]) and row["dlps_conditioned"] != row["dlps"]
    assert np.isfinite(rep["relative"]["full"]["SV(rw)"]["joint"]["8"]["dlpd_conditioned"])


def test_main_keeps_the_default_estimator_of_existing_configs(cr, run_main):
    rec, rep, md = run_main("--config", "smoke")
    _expected_calls(rec, "conditioned")
    assert rep["config"]["lpd_method"] == "conditioned"
    assert cr.RELIABILITY_NOTE in md and cr.RELIABILITY_NOTE_INTEGRATED not in md
    assert cr.LPD_INTEGRATED_DEVIATION not in rep["deviations"]
    c4 = next(c for c in rep["criteria"] if c["id"] == 4)
    assert "path-conditioned" not in c4["detail"]
    assert "dlps_conditioned" not in rep["relative"]["full"]["SV(rw)"]["variables"]["UR"]["8"]


def test_main_runs_the_acceptance_config_with_its_estimator(cr, run_main, monkeypatch):
    # every 8th origin (as smoke) so the stand-ins stay cheap; everything else is the config's
    acc = dataclasses.replace(cr.CONFIGS["acceptance"], origin_step=8, ordering_step=8)
    monkeypatch.setitem(cr.CONFIGS, "acceptance", acc)
    rec, rep, md = run_main("--config", "acceptance")
    _expected_calls(rec, "integrated")
    assert rep["config"]["mcmc"] == dataclasses.asdict(
        MCMCSettings(chains=4, draws=20000, burn=1000)
    )
    assert "MCMC budget: 4 chains × 20000 kept draws, burn 1000, thin 1" in md
    assert "NOT the acceptance run" not in md.splitlines()[0]
    # the acceptance config with the other estimator is not the pre-registered run
    rec, rep, md = run_main("--config", "acceptance", "--lpd", "conditioned")
    _expected_calls(rec, "conditioned")
    assert "NOT the acceptance run" in md.splitlines()[0]
    # --only-ordering fills the cache of one model's ordering stage with the same estimator
    rec, _, _ = run_main("--config", "acceptance", "--only-ordering", "SV(rw)")
    assert rec.orderings == [{"model": rec.orderings[0]["model"], "lpd_method": "integrated"}]
    assert rec.backtests == []


# =========================================================================== provenance and labels


def _acceptance_cheap(cr, monkeypatch):
    """The acceptance config on every 8th origin (as smoke), everything else as registered."""
    acc = dataclasses.replace(cr.CONFIGS["acceptance"], origin_step=8, ordering_step=8)
    monkeypatch.setitem(cr.CONFIGS, "acceptance", acc)


def test_git_state_of_this_checkout(cr):
    """The real ``git_state``: a full commit hash and a boolean dirty flag (with the diff's
    SHA-256 exactly when dirty), or ``"unknown"`` without git."""
    git = cr.git_state()
    assert set(git) == {"git_commit", "git_dirty", "git_diff_sha256"}
    if git["git_commit"] == "unknown":
        assert git["git_dirty"] is None and git["git_diff_sha256"] is None
    else:
        assert len(git["git_commit"]) == 40 and int(git["git_commit"], 16) >= 0
        assert isinstance(git["git_dirty"], bool)
        assert (git["git_diff_sha256"] is not None) == git["git_dirty"]


def test_report_records_the_commit_and_the_cache(cr, run_main, monkeypatch):
    """Finding cr-1: the commit and the dirty flag are in the JSON and the header (the
    pre-registration says the run's commit "is recorded in the report header"); cr-3: so are
    the cache and the origins it served."""
    _acceptance_cheap(cr, monkeypatch)
    _, rep, md = run_main("--config", "acceptance")
    header = next(line for line in md.splitlines() if line.startswith("Generated"))
    assert {k: rep["versions"][k] for k in CLEAN} == CLEAN
    assert f"at commit `{CLEAN['git_commit']}` (clean working tree)" in header
    assert rep["config"]["cache"].endswith("cache")
    names = ["Constant", "SV(rw)", "SV", "SV(rw,t10)", "AR-SV(0.9)", "Constant [structural]",
             "SV(rw) [structural]"]  # fmt: skip
    n = rep["n_origins"]["full"]
    assert rep["cached"] == {name: {"cached": n, "origins": n} for name in names}
    assert f"{7 * n} of {7 * n} origin backtests" in header and rep["config"]["cache"] in header
    assert "NOT the acceptance run" not in md.splitlines()[0]
    dirty = {**CLEAN, "git_dirty": True, "git_diff_sha256": "ab" * 32}
    _, rep, md = run_main("--config", "acceptance", git=dirty)
    assert rep["versions"]["git_dirty"] is True
    assert f"`git diff HEAD` SHA-256 `{'ab' * 32}`" in md
    assert "NOT the acceptance run" in md.splitlines()[0]
    assert "tracked files had uncommitted changes" in md
    unknown = {"git_commit": "unknown", "git_dirty": None, "git_diff_sha256": None}
    _, rep, md = run_main("--config", "acceptance", git=unknown)
    assert "at an unknown commit" in md and "the commit is unknown" in md
    assert "NOT the acceptance run" in md.splitlines()[0]


def test_acceptance_label_needs_the_registered_run(cr, run_main, monkeypatch):
    """Finding cr-2: another seed or skipped stages are not the pre-registered run, whatever the
    config name; the banner says why and ``--note`` text reaches the report."""
    _acceptance_cheap(cr, monkeypatch)
    argv = ("--config", "acceptance", "--seed", "7", "--skip-orderings", "--skip-structural",
            "--note", "MYNOTE sentence.")  # fmt: skip
    _, rep, md = run_main(*argv)
    assert "NOT the acceptance run" in md.splitlines()[0]
    assert rep["config"]["skip_orderings"] is True and rep["config"]["skip_structural"] is True
    gaps = cr.acceptance_gaps(rep)
    assert gaps == [
        "seed = 7, not 0",
        "the 24-ordering stage was not run",
        "the structural-form pair was not run",
    ]
    assert f"This report differs from the pre-registered run: {'; '.join(gaps)}." in md
    assert "MYNOTE sentence." in md and rep["notes"] == ["MYNOTE sentence."]
    # the note reaches an acceptance report too (it used to sit in the not-acceptance banner)
    _, rep, md = run_main("--config", "acceptance", "--note", "MYNOTE sentence.")
    assert cr.is_acceptance(rep) and "> MYNOTE sentence." in md
    # other configs are never the acceptance run
    _, rep, md = run_main("--config", "smoke")
    assert cr.acceptance_gaps(rep) == ["config `smoke`"]


def test_main_stage_report_of_the_acceptance_run_is_pending(cr, run_main, monkeypatch):
    """Finding cr-2: the report written before the ordering stage (all that is left if that
    stage is interrupted) says it is pending, in the title, the banner and criterion 6."""
    _acceptance_cheap(cr, monkeypatch)
    _, rep, md = run_main("--config", "acceptance", interrupt=True)
    assert rep["pending"] == ["orderings"] and "orderings" not in rep
    assert md.splitlines()[0].endswith(
        "— acceptance config, main stage only (ordering stage pending)"
    )
    assert "Main-stage report of the acceptance run" in md and "NOT the acceptance run" not in md
    c6 = next(c for c in rep["criteria"] if c["id"] == 6)
    assert (
        c6["met"] is None
        and c6["detail"] == "ordering stage pending (this is the main-stage report)"
    )
    assert cr.acceptance_gaps(rep) == [cr.PENDING]
    # the same partial report of a non-acceptance config: NOT the acceptance run, pending
    _, rep, md = run_main("--config", "smoke", interrupt=True)
    assert "NOT the acceptance run" in md.splitlines()[0]
    assert "criterion 6 is pending" in md
    # --skip-orderings: criterion 6 keeps the "skipped" text, nothing is pending
    _, rep, md = run_main("--config", "smoke", "--skip-orderings")
    assert "pending" not in rep
    assert next(c for c in rep["criteria"] if c["id"] == 6)["detail"] == "ordering stage skipped"


def test_acceptance_report_cites_its_preregistration(cr, run_main, monkeypatch):
    """Finding cr-4: the acceptance report names its pre-registration and its two departures
    from the reference full run; cr-5 (c): the header says how the stored draws are thinned."""
    _acceptance_cheap(cr, monkeypatch)
    *_, md = run_main("--config", "acceptance")
    assert (
        f"The `acceptance` config is pre-registered in `{cr.PREREGISTRATION}`; it departs from "
        "the reference full run "
        f"(report at `{cr.REFERENCE_RUN}`, 4 × 5000 kept draws, burn 1000, the `conditioned` LPD) "
        "only in the MCMC budget (4 × 20000 kept draws, burn 1000, every MCMC stage) and the "
        "primary LPD estimator (`integrated`)." in md
    )
    assert (Path(SCRIPT).parents[1] / cr.PREREGISTRATION).exists()
    assert "parameters of one in 8 kept draws stored (`store_max` 2500" in md
    *_, md = run_main("--config", "smoke")
    assert "is pre-registered in" not in md and "parameters of every kept draw stored" in md


def test_startup_log_names_the_estimator_budget_and_commit(cr, run_main, monkeypatch, capsys):
    """Note plumbing-only-ordering-silent-refit: two invocations that differ in estimator (and
    so never share cached scores) say so in their logs, the ``--only-ordering`` one included."""
    _acceptance_cheap(cr, monkeypatch)
    run_main("--config", "acceptance", "--only-ordering", "SV(rw)", "--lpd", "conditioned")
    out = capsys.readouterr().out
    setup = "config acceptance: lpd_method=conditioned, MCMC 4 × 20000 (burn 1000)"
    assert f"{setup}" in out.splitlines()[0] and CLEAN["git_commit"] in out.splitlines()[0]
    assert any("ordering backtests of SV(rw) cached (" + setup in line for line in out.splitlines())


def test_moments_are_written_atomically(cr, tmp_path, monkeypatch):
    """Note plumbing-moments-store-shared: the moments go to a temporary file renamed into
    place, so a concurrent run sharing the store never reads a partial one; no file is left."""
    fr = ForecastResult.from_arrays(
        paths=np.random.default_rng(0).normal(size=(50, 8, 4)), cond_mean=np.zeros((50, 8, 4)),
        cond_chol=np.broadcast_to(np.eye(4), (50, 8, 4, 4)), variables=VARS,
        origin=pd.Period("2000Q1", "Q"),
    )  # fmt: skip
    inner = type("Fit", (), {"forecast": lambda self, h, d, seed=None: fr})()
    probe = cr.MomentsProbe(rb.models.OLSVAR(1), str(tmp_path), 0, 50, "tag")
    written = []
    savez = np.savez
    monkeypatch.setattr(
        cr.np, "savez", lambda path, **kw: (written.append(path), savez(path, **kw))
    )
    assert cr._ProbeFit(probe, inner, "2000Q1").forecast(8, 50) is fr
    final = probe.directory() / "2000Q1.npz"
    assert written and Path(written[0]) != final and Path(written[0]).parent == final.parent
    assert [f.name for f in final.parent.iterdir()] == ["2000Q1.npz"]
    m = probe.moments("2000Q1")
    np.testing.assert_allclose(m["mean"], fr.stacked().mean(axis=0))
    assert list(m["variables"]) == VARS
