"""Wiring of ``benchmarks/dgp_study.py`` (SPEC §10.5): configs, records, resume, labels, report.

``main()`` runs in-process (``--workers 1``) with a stand-in for ``dgp.run_replication`` that
returns synthetic records, so these tests fit nothing; their rates mean nothing. The pilot and
the full study share the registered design (the pilot's records are the full study's first
replications), the smoke config has a seed of its own, records are appended and a rerun resumes,
and the report is labelled the pre-registered run only for a registered config whose records all
come from one clean commit that contains the reported rows' pre-registration (looked up in the
records' commit, which the report names), with every row of that pre-registration present, every
replication present and none recorded twice. Without ``--rows`` the script runs one
pre-registration's rows (``--preregistration``, by default the Phase 2 file), so rows of another
pre-registration are another study, and a report of rows of two pre-registrations is never
labelled. A row without a pre-registration runs only when named and only under the smoke config;
under pilot and full the script also refuses an unreadable repository, a pre-registration that is
not a file of ``HEAD``, and replications to run unless the tree is known to be clean, before it
writes anything. ``git_state``, ``in_commit`` and ``modified_paths`` are stand-ins unless a test
says otherwise.
The ``slow`` test runs the real smoke config for one replication of one row (tiny chains).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import rai_bvar as rb
from rai_bvar.settings import MCMCSettings
from rai_bvar.validation import dgp

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "dgp_study.py"
CLEAN = {
    "git_commit": "c" * 40,
    "git_dirty": False,
    "git_diff_sha256": None,
    "preregistration_in_commit": True,
}


REGISTERED_HASHES = {
    "Linear × Constant": "8aebe5adec44e966" + "0" * 48,
    "Linear × SV": "b3daa828453542ed" + "0" * 48,
    "Linear × SV(outliers)": "7f0aeb733eb2d07e" + "0" * 48,
}


@pytest.fixture(scope="module")
def study():
    name = "dgp_study_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module's annotations through it
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def fake_replication(row: dgp.Row, rep: int, design: dgp.Design, *, cache=None) -> dict:
    """A record with the real layout; SV 'better' in every third replication."""
    pairs = []
    for pair in row.pairs:
        for h in design.horizons:
            win = rep % 3 == 0
            verdict = "better" if win else "inconclusive"
            pairs.append(
                {
                    "model": pair.model,
                    "reference": pair.reference,
                    "axis": pair.axis,
                    "expect": pair.expect,
                    "horizon": h,
                    "verdict": verdict,
                    "verdict_gate_off": verdict,
                    "dlpd": 10.0 if win else 0.5,
                    "dlpd_mcse": 1.0,
                    "dm_pvalue_better": 0.01 if win else 0.4,
                    "dm_pvalue_worse": 0.99 if win else 0.6,
                    "tests_valid": True,
                    "model_in_mcs": True,
                    "reference_in_mcs": True,
                    "origins": 80,
                    "converged_model": 1.0,
                    "converged_reference": 1.0,
                    "reasons": "",
                }
            )
    return {
        "row": row.name,
        "rep": rep,
        "design": design.key(),
        "dgp": row.dgp,
        "data_fingerprint": f"{row.dgp}-{rep}",
        "compare_seed": dgp.compare_seed(design, row.dgp, rep),
        "seconds": 100.0 * len(row.arms),
        "started_at": f"2026-10-28T10:{rep % 60:02d}:00+00:00",
        "finished_at": f"2026-10-28T11:{rep % 60:02d}:00+00:00",
        "arms": {
            a: {
                "spec_hash": REGISTERED_HASHES.get(a, "0" * 64),
                "seconds": 50.0,
                "cached": 0,
                "origins": 80,
                "converged": {str(h): 1.0 for h in design.horizons},
                "warnings": {},
            }
            for a in row.arms
        },
        "pairs": pairs,
    }


@pytest.fixture
def run_main(study, monkeypatch, tmp_path):
    monkeypatch.setattr(dgp, "run_replication", fake_replication)
    monkeypatch.setattr(study, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(study, "in_commit", lambda commit, path: True)
    monkeypatch.setattr(study, "modified_paths", lambda: ["src/rai_bvar/api.py"])
    monkeypatch.setattr(study, "_init_worker", lambda: None)  # no JAX config in this process
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "xla"))

    def run(*args: str) -> tuple[Path, dict]:
        out = tmp_path / "study"
        study.main([str(out), "--cache", str(tmp_path / "cache"), "--workers", "1", *args])
        return out, json.loads(out.with_suffix(".json").read_text())

    return run


def test_configs(study):
    assert study.REGISTERED == dgp.Design()
    assert study.CONFIGS["full"].design == study.CONFIGS["pilot"].design == study.REGISTERED
    assert (study.CONFIGS["full"].reps, study.CONFIGS["pilot"].reps) == (200, 24)
    smoke = study.CONFIGS["smoke"].design
    assert smoke.seed != study.REGISTERED.seed  # no registered replication is scored by it
    assert smoke.mcmc == MCMCSettings(chains=2, draws=100, burn=100)
    assert smoke.key() != study.REGISTERED.key()
    assert study.REGISTERED.mcmc is None  # the library default budgets (D-35)
    assert study.REGISTERED.key() == "81f067e0cefaaea5"  # the design key of the registered run
    assert study.PREREGISTRATION == "benchmarks/reports/dgp_study_preregistration.md"
    assert set(study.REGISTERED_CONFIGS) == {"pilot", "full"}


def test_pilot_run_records_resumes_and_is_labelled(run_main, study):
    out, rep = run_main("--config", "pilot")
    lines = out.with_suffix(".jsonl").read_text().splitlines()
    assert len(lines) == 3 * 24
    assert {json.loads(x)["git_commit"] for x in lines} == {CLEAN["git_commit"]}
    assert rep["registered"] and rep["not_registered_because"] == []
    assert rep["meta"]["ran"] == 72 and rep["meta"]["skipped"] == 0
    md = out.with_suffix(".md").read_text()
    assert "**The pre-registered pilot run**" in md and "This is the **pilot**" in md
    assert "dgp_study_preregistration.md" in md
    assert f"every record from commit `{CLEAN['git_commit'][:12]}`" in md
    assert f"- Code: records from commit `{CLEAN['git_commit']}`; rai-bvar" in md
    assert "ran no replication" not in md
    assert rep["records_commit"] == CLEAN["git_commit"] and rep["records_twice"] == []
    # the pending rows are listed, not run
    assert {p["name"] for p in rep["pending"]} == {
        "linear-sv-bart",
        "threshold-sv-bart",
        "threshold-sv-threshold",
        "tvp-sv",
    }
    # one summary per scored pair and horizon: 2 + 1 + 1 pairs × 2 horizons
    assert len(rep["summary"]) == 8
    s = next(x for x in rep["summary"] if x["row"] == "linear-sv" and x["horizon"] == 1)
    assert (s["n"], s["gate_on"]["better"]["k"]) == (24, 8)  # reps 0, 3, …, 21
    assert s["rate"] == pytest.approx(1 / 3) and s["reading"] == "descriptive (power)"
    fam1, fam4 = rep["family"]
    assert fam1["row"] == "linear-homoskedastic" and fam1["verdict"]["k"] == 8
    assert (fam1["horizon"], fam4["horizon"]) == (1, 4)
    # the estimate: per-row seconds × the full study's missing replications / workers
    e = rep["estimate"]
    assert e["rows"]["linear-homoskedastic"] == {
        "replications_timed": 24,
        "seconds_mean": 300.0,
        "missing": 176,
    }
    assert e["remaining_hours"] == pytest.approx((300 + 200 + 200) * 176 / 3600)
    assert e["full_hours"] == pytest.approx((300 + 200 + 200) * 200 / 3600)
    # a rerun resumes: nothing left to run, same records, still registered
    out, again = run_main("--config", "pilot")
    assert len(out.with_suffix(".jsonl").read_text().splitlines()) == 72
    assert again["registered"] and again["meta"]["skipped"] == 72 and "ran" not in again["meta"]
    assert again["summary"] == rep["summary"]
    # its machine, workers, notes and estimate are its own, as for --report-only
    assert (
        "- This invocation ran no replication (every selected replication was already "
        "recorded), so the machine"
    ) in out.with_suffix(".md").read_text()


def test_full_run_reuses_the_pilot_records(run_main):
    run_main("--config", "pilot")
    _, rep = run_main("--config", "full", "--reps", "24:30")
    assert rep["meta"]["ran"] == 18 and rep["meta"]["skipped"] == 0
    assert not rep["registered"]  # 30 of 200 replications
    assert any("30 of 200 replications" in r for r in rep["not_registered_because"])
    assert rep["summary"][0]["n"] == 30


def test_labels(run_main, study, monkeypatch, capsys):
    out, rep = run_main("--config", "pilot", "--reps", "0:2")
    assert not rep["registered"]
    assert any("2 of 24 replications" in r for r in rep["not_registered_because"])
    # a modified tree runs no replication of a registered config; it may still write a report
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_dirty": True})
    for config in ("pilot", "full"):
        with pytest.raises(SystemExit):
            run_main("--config", config)
        err = capsys.readouterr().err
        assert "tracked files differ from HEAD (src/rai_bvar/api.py)" in err
        assert "Commit the changes" not in err and "an OUT path that git does not track" in err
    path = out.with_suffix(".jsonl")
    assert len(path.read_text().splitlines()) == 6  # nothing ran
    _, rep = run_main("--config", "pilot", "--report-only")  # 66 missing, none run
    assert any("2 of 24 replications" in r for r in rep["not_registered_because"])
    _, rep = run_main("--config", "smoke", "--rows", "linear-sv")  # not registered: runs
    assert rep["meta"]["ran"] == 2
    monkeypatch.setattr(study, "git_state", lambda: dict(CLEAN))
    out, rep = run_main("--config", "pilot")  # a resume: 66 run, 6 recorded
    assert (
        rep["meta"]["ran"] == 66 and "ran no replication" not in out.with_suffix(".md").read_text()
    )
    # with every replication recorded, a run-mode invocation on a modified tree runs nothing and
    # writes its report: the label reads the records' trees, not this one
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_dirty": True})
    _, rep = run_main("--config", "pilot")
    assert rep["registered"] and "ran" not in rep["meta"]
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r | {"git_dirty": r["rep"] == 5}) + "\n" for r in recs))
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_dirty": True})
    out, rep = run_main("--config", "pilot", "--report-only")
    assert rep["not_registered_because"] == [
        "some records were produced with modified tracked files"
    ]
    md = out.with_suffix(".md").read_text()
    assert "**Not the pre-registered run:**" in md
    # the flag is this invocation's: it goes with the reporting commit, even the records' own one
    c = CLEAN["git_commit"]
    assert f"records from commit `{c}`; this report written at commit `{c}` (tracked files" in md
    # records whose tree git could not report are not known to be clean
    path.write_text(
        "".join(json.dumps(r | {"git_dirty": None}) + "\n" for r in recs[:3])
        + "".join(json.dumps(r) + "\n" for r in recs[3:])
    )
    _, rep = run_main("--config", "pilot", "--report-only")
    assert rep["not_registered_because"] == [
        "3 records do not say whether tracked files were modified"
    ]


def test_label_needs_the_registered_arm_specs(run_main, monkeypatch):
    def other_spec(row, rep, design, *, cache=None):
        rec = fake_replication(row, rep, design)
        for a in rec["arms"].values():
            a["spec_hash"] = "f" * 64
        return rec

    monkeypatch.setattr(dgp, "run_replication", other_spec)
    _, rep = run_main("--config", "pilot", "--rows", "linear-sv")
    assert not rep["registered"]
    assert any("unregistered specs" in r for r in rep["not_registered_because"])


def test_label_looks_for_the_preregistration_in_the_records_commit(run_main, study, monkeypatch):
    """Records made at a commit without the pre-registration are not the registered run when a
    later commit that has it writes the report; the report names the records' commit."""
    old, new = "a" * 40, "b" * 40
    looked_up = []

    def in_commit(commit, path):
        looked_up.append((commit, path))
        return commit != old

    monkeypatch.setattr(study, "in_commit", in_commit)
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_commit": old})
    run_main("--config", "pilot")
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_commit": new})
    looked_up.clear()
    out, rep = run_main("--config", "pilot", "--report-only")
    assert looked_up == [("HEAD", study.PREREGISTRATION), (old, study.PREREGISTRATION)]
    assert not rep["registered"] and rep["records_commit"] == old
    assert rep["not_registered_because"] == [f"{study.PREREGISTRATION} is not in commit {old[:12]}"]
    md = out.with_suffix(".md").read_text()
    assert f"records from commit `{old}`; this report written at commit `{new}`" in md
    assert "- This invocation ran no replication (regenerated with `--report-only`), so" in md
    # with the pre-registration in the records' commit, the label holds and names that commit
    monkeypatch.setattr(study, "in_commit", lambda commit, path: commit in ("HEAD", old))
    out, rep = run_main("--config", "pilot", "--report-only")
    assert rep["registered"]
    assert f"every record from commit `{old[:12]}`" in out.with_suffix(".md").read_text()


def test_in_commit_of_this_checkout(study):
    if study.git_state()["git_commit"] == "unknown":
        pytest.skip("not a git checkout")
    assert study.in_commit("HEAD", "benchmarks/dgp_study.py")
    assert not study.in_commit("HEAD", "benchmarks/no-such-file.md")
    assert not study.in_commit("0" * 40, study.PREREGISTRATION)
    assert not study.in_commit("", "benchmarks/dgp_study.py")  # not the index
    assert not study.in_commit("HEAD", "benchmarks/reports")  # a directory, not a file


def test_label_needs_every_preregistered_row(run_main):
    _, rep = run_main("--config", "pilot", "--rows", "linear-sv")
    assert not rep["registered"]
    assert rep["not_registered_because"] == [
        "row linear-homoskedastic is not in the report",
        "row linear-sv-outliers is not in the report",
    ]


def test_the_rows_of_another_preregistration_are_another_study(run_main, study, monkeypatch):
    """A later row with a pre-registration of its own (a Phase 4 row, say) is another study. The
    pre-registration's commands (no ``--rows``) still run and label the Phase 2 rows only, also
    with ``--report-only``; ``--preregistration`` selects the other study, whose label looks up its
    own file, asks for its own rows and names its file."""
    monkeypatch.setattr(rb, "BART", object, raising=False)  # a stand-in for a Phase 4 build
    path = "benchmarks/reports/another_study.md"
    row = dgp.get_row("linear-sv-bart")
    other = dataclasses.replace(row, preregistration=path)
    monkeypatch.setattr(dgp, "ROWS", tuple(other if r is row else r for r in dgp.ROWS))
    hashes = study.REGISTERED_SPEC_HASHES | {"BART × SV": "0" * 16}
    monkeypatch.setattr(study, "REGISTERED_SPEC_HASHES", hashes)
    assert dgp.row_status(other) is None
    looked_up = []

    def in_commit(commit, path):
        looked_up.append(path)
        return True

    monkeypatch.setattr(study, "in_commit", in_commit)
    phase2 = ["linear-homoskedastic", "linear-sv", "linear-sv-outliers"]
    out, rep = run_main("--config", "pilot")
    assert rep["registered"], rep["not_registered_because"]
    assert [r["name"] for r in rep["rows"]] == phase2
    assert "linear-sv-bart" not in {p["name"] for p in rep["pending"]}
    lines = out.with_suffix(".jsonl").read_text().splitlines()
    assert {json.loads(x)["row"] for x in lines} == set(phase2)
    assert set(looked_up) == {study.PREREGISTRATION}
    _, again = run_main("--config", "pilot", "--report-only")
    assert again["registered"] and again["summary"] == rep["summary"]
    _, rep = run_main("--config", "pilot", "--rows", "linear-sv", "--report-only")
    assert rep["not_registered_because"] == [
        "row linear-homoskedastic is not in the report",
        "row linear-sv-outliers is not in the report",
    ]
    # the other study
    looked_up.clear()
    out, rep = run_main("--config", "pilot", "--preregistration", path)
    assert [r["name"] for r in rep["rows"]] == ["linear-sv-bart"]
    assert rep["registered"], rep["not_registered_because"]
    assert set(looked_up) == {path} and rep["preregistration"] == path
    assert f"**The pre-registered pilot run** ({path})" in out.with_suffix(".md").read_text()
    # one report of both studies is reported, never labelled
    _, rep = run_main("--config", "pilot", "--rows", "linear-sv-bart", *phase2, "--report-only")
    assert rep["not_registered_because"] == [
        f"the reported rows come from 2 pre-registrations ({path}, {study.PREREGISTRATION}); "
        "a registered report is one pre-registration's study"
    ]


def test_a_preregistration_not_in_head_is_refused(run_main, study, monkeypatch, capsys):
    """Not committed yet, or a typo in the row's path: under pilot or full the row would score
    registered samples before its pre-registration exists in a commit."""
    monkeypatch.setattr(rb, "BART", object, raising=False)  # a stand-in for a Phase 4 build
    path = "benchmarks/reports/another_study.md"
    row = dgp.get_row("linear-sv-bart")
    other = dataclasses.replace(row, preregistration=path)
    monkeypatch.setattr(dgp, "ROWS", tuple(other if r is row else r for r in dgp.ROWS))
    monkeypatch.setattr(study, "in_commit", lambda commit, p: p != path)
    for args in (
        ["pilot", "--preregistration", path],
        ["full", "--preregistration", path],
        ["pilot", "--rows", "linear-sv-bart"],
        ["full", "--rows", "linear-sv-bart"],
        ["pilot", "--preregistration", path, "--report-only"],
    ):
        with pytest.raises(SystemExit):
            run_main("--config", *args)
        assert f"{path} is not a file of the checked-out commit (HEAD)" in capsys.readouterr().err
    out, rep = run_main("--config", "pilot")  # the Phase 2 rows' file is in HEAD
    assert rep["registered"]
    lines = out.with_suffix(".jsonl").read_text().splitlines()
    assert all(json.loads(x)["row"] != "linear-sv-bart" for x in lines)  # nothing scored
    _, rep = run_main("--config", "smoke", "--rows", "linear-sv-bart")  # a seed of its own
    assert {s["row"] for s in rep["summary"]} == {"linear-sv-bart"} and not rep["registered"]
    # complete records of the row, then its file is gone from HEAD: --report-only is refused too
    hashes = study.REGISTERED_SPEC_HASHES | {"BART × SV": "0" * 16}
    monkeypatch.setattr(study, "REGISTERED_SPEC_HASHES", hashes)
    monkeypatch.setattr(study, "in_commit", lambda commit, p: True)
    _, rep = run_main("--config", "pilot", "--preregistration", path)
    assert rep["registered"] and rep["meta"]["ran"] == 24
    monkeypatch.setattr(study, "in_commit", lambda commit, p: p != path)
    with pytest.raises(SystemExit):
        run_main("--config", "pilot", "--preregistration", path, "--report-only")
    assert f"{path} is not a file of the checked-out commit (HEAD)" in capsys.readouterr().err


def test_git_must_know_the_tree_under_a_registered_config(
    run_main, study, monkeypatch, capsys, tmp_path
):
    """A repository git cannot read, or a tree whose state git cannot report (``git status``
    failing while ``git cat-file`` works), runs no registered replication and writes nothing."""
    monkeypatch.setattr(study, "git_state", lambda: CLEAN | {"git_dirty": None})
    out = tmp_path / "new" / "study"
    args = [str(out), "--config", "pilot", "--workers", "1", "--cache", str(tmp_path / "c")]
    with pytest.raises(SystemExit):
        study.main(args)
    captured = capsys.readouterr()
    assert "git could not report whether tracked files differ from HEAD" in captured.err
    assert captured.out == "" and not (tmp_path / "new").exists()  # no log line, no directory
    _, rep = run_main("--config", "smoke", "--rows", "linear-sv")  # not registered: runs
    assert rep["meta"]["ran"] == 2
    unknown = {"git_commit": "unknown", "git_dirty": None, "git_diff_sha256": None}
    monkeypatch.setattr(study, "git_state", lambda: dict(unknown))
    for extra in ([], ["--report-only"]):
        with pytest.raises(SystemExit):
            run_main("--config", "pilot", *extra)
        assert "git could not read the repository" in capsys.readouterr().err


def test_the_row_selection_is_checked(run_main, capsys):
    for args, message in (
        (["--rows", "linear-sv", "linear-sv"], "--rows names linear-sv more than once"),
        (["--preregistration", "benchmarks/reports/no_such.md"], "no row is pre-registered in"),
        (["--rows", "linear-sv", "--preregistration", "x.md"], "not allowed with argument"),
    ):
        with pytest.raises(SystemExit):
            run_main("--config", "pilot", *args)
        assert message in capsys.readouterr().err


def test_a_named_row_this_build_cannot_run_is_pending(run_main):
    row = dgp.get_row("linear-sv-bart")
    assert dgp.row_missing(row) == ["BART"]
    out, rep = run_main("--config", "pilot", "--rows", "linear-sv-bart")
    assert [p["name"] for p in rep["pending"]] == ["linear-sv-bart"]
    assert rep["rows"] == [] and rep["summary"] == [] and not out.with_suffix(".jsonl").exists()
    assert rep["records_commit"] is None and not rep["registered"]
    assert rep["preregistration"] is None
    md = out.with_suffix(".md").read_text()
    assert f"- Code: no records; this report written at commit `{CLEAN['git_commit']}`" in md
    assert "- This invocation ran no replication (there was nothing to run), so" in md


def test_a_replication_recorded_twice_counts_once_and_is_not_registered(run_main):
    out, _ = run_main("--config", "pilot")
    path = out.with_suffix(".jsonl")
    lines = path.read_text().splitlines()
    again = [x for x in lines if json.loads(x)["row"] == "linear-sv"][:3]  # reps 0, 1, 2
    # the second copy of replication 0 has replication 1's (inconclusive) verdicts
    changed = json.loads(again[0]) | {"pairs": json.loads(again[1])["pairs"]}
    again[0] = json.dumps(changed)
    path.write_text("\n".join(lines + again) + "\n")
    _, rep = run_main("--config", "pilot", "--report-only")
    assert not rep["registered"]
    assert rep["not_registered_because"] == [
        "3 replications recorded more than once (linear-sv 0, linear-sv 1, linear-sv 2)"
    ]
    assert rep["records_twice"] == [["linear-sv", 0], ["linear-sv", 1], ["linear-sv", 2]]
    assert {s["n"] for s in rep["summary"]} == {24}  # each replication counted once
    s = next(x for x in rep["summary"] if x["row"] == "linear-sv" and x["horizon"] == 1)
    assert s["gate_on"]["better"]["k"] == 8  # the first copy of replication 0 counts


def test_records_of_two_commits_or_of_none(run_main, study, monkeypatch):
    looked_up = []

    def in_commit(commit, path):
        looked_up.append(commit)
        return True

    monkeypatch.setattr(study, "in_commit", in_commit)
    out, _ = run_main("--config", "pilot")
    path = out.with_suffix(".jsonl")
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    for r in recs[:10]:
        r["git_commit"] = "d" * 40
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out, rep = run_main("--config", "pilot", "--report-only")
    assert rep["records_commit"] is None and rep["records_commits"] == 2
    assert not rep["registered"]
    assert any("records come from 2 commits" in r for r in rep["not_registered_because"])
    assert "- Code: records from more than one commit (see above); this report written at" in (
        out.with_suffix(".md").read_text()
    )
    # records without a commit, or with an empty one (git would read the index for it)
    for commit in (None, ""):
        for r in recs:
            r.pop("git_commit", None)
            if commit is not None:
                r["git_commit"] = commit
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        looked_up.clear()
        out, rep = run_main("--config", "pilot", "--report-only")
        assert rep["records_commit"] is None and rep["records_commits"] == 1
        assert rep["not_registered_because"] == ["the records name no commit"]
        assert "- Code: records that name no commit; this report written at" in (
            out.with_suffix(".md").read_text()
        )
        assert looked_up == ["HEAD"]  # the script's check of this checkout, not the label's


def test_a_row_without_a_preregistration_runs_only_when_named_and_only_in_smoke(
    run_main, study, monkeypatch, capsys
):
    """Under pilot or full an unregistered row would score samples of the registered design and
    seed (``linear-sv-bart`` is on the ``linear-sv`` DGP) before its pre-registration: refused."""
    monkeypatch.setattr(rb, "BART", object, raising=False)  # a stand-in for a Phase 4 build
    row = dgp.get_row("linear-sv-bart")
    assert dgp.row_missing(row) == [] and row.preregistration is None
    out, rep = run_main("--config", "pilot", "--reps", "0:2")
    assert "linear-sv-bart" in {p["name"] for p in rep["pending"]}
    assert all(s["row"] != "linear-sv-bart" for s in rep["summary"])
    for args in (["pilot"], ["full"], ["pilot", "--report-only"], ["full", "--report-only"]):
        with pytest.raises(SystemExit):
            run_main("--config", *args, "--rows", "linear-sv-bart", "--reps", "0:2")
        err = capsys.readouterr().err
        assert "no pre-registration for row linear-sv-bart" in err and "seed (20261025)" in err
    lines = out.with_suffix(".jsonl").read_text().splitlines()
    assert all(json.loads(x)["row"] != "linear-sv-bart" for x in lines)  # nothing scored
    _, rep = run_main("--config", "smoke", "--rows", "linear-sv-bart")
    assert [r["name"] for r in rep["rows"]] == ["linear-sv-bart"]
    assert {s["row"] for s in rep["summary"]} == {"linear-sv-bart"}
    assert rep["config"]["design"]["seed"] != study.REGISTERED.seed
    assert not rep["registered"] and rep["preregistration"] is None
    assert "row linear-sv-bart is not pre-registered" in rep["not_registered_because"]


def test_the_report_gives_when_the_replications_ran(run_main, monkeypatch):
    calls = []
    ticks = iter(f"2026-10-28T12:00:{i:02d}+00:00" for i in range(60))

    def now():
        calls.append("now")
        return next(ticks)

    def replication(row, rep, design, *, cache=None):
        calls.append("replication")
        return fake_replication(row, rep, design)

    monkeypatch.setattr(dgp, "utc_now", now)
    monkeypatch.setattr(dgp, "run_replication", replication)
    out, rep = run_main("--config", "pilot")
    # the invocation's times: one before the first replication, one after the last
    assert calls == ["now"] + ["replication"] * 72 + ["now"]
    meta = rep["meta"]
    assert (meta["started_at"], meta["finished_at"]) == (
        "2026-10-28T12:00:00+00:00",
        "2026-10-28T12:00:01+00:00",
    )
    assert rep["timestamps"] == {
        "first_started_at": "2026-10-28T10:00:00+00:00",
        "last_finished_at": "2026-10-28T11:23:00+00:00",
        "records_without": 0,
    }
    md = out.with_suffix(".md").read_text()
    assert "- Replications ran from 2026-10-28T10:00:00+00:00 to 2026-10-28T11:23:00+00:00" in md
    assert (
        "- This invocation: started 2026-10-28T12:00:00+00:00, finished "
        "2026-10-28T12:00:01+00:00 (UTC)."
    ) in md
    # records written before the times were recorded (the Phase 2 study's) still report
    path = out.with_suffix(".jsonl")
    old = [json.loads(x) for x in path.read_text().splitlines()]
    for r in old:
        del r["started_at"], r["finished_at"]
    path.write_text("".join(json.dumps(r) + "\n" for r in old))
    out, rep = run_main("--config", "pilot", "--report-only")
    assert rep["registered"] and rep["timestamps"]["records_without"] == 72
    assert "- The records carry no wall-clock times." in out.with_suffix(".md").read_text()


def test_smoke_is_never_registered_and_says_so(run_main):
    out, rep = run_main("--config", "smoke", "--rows", "linear-sv")
    assert not rep["registered"]
    assert "Smoke run: the numbers mean nothing" in out.with_suffix(".md").read_text()
    assert rep["config"]["design"]["seed"] == 1


def test_records_of_another_design_are_ignored(run_main):
    run_main("--config", "smoke", "--rows", "linear-sv")
    _, rep = run_main("--config", "pilot", "--rows", "linear-sv", "--reps", "0:1")
    assert rep["meta"]["skipped"] == 0 and rep["summary"][0]["n"] == 1


def test_git_state_of_this_checkout(study):
    state = study.git_state()
    assert set(state) == {"git_commit", "git_dirty", "git_diff_sha256", "preregistration_in_commit"}
    if state["git_commit"] != "unknown":
        assert len(state["git_commit"]) == 40


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")  # tiny chains: every SV fit misses the gate
def test_smoke_config_runs_end_to_end(study, monkeypatch, tmp_path):
    """The real pipeline, one replication of one row at the smoke config's tiny chains."""
    monkeypatch.setattr(study, "_init_worker", lambda: None)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "xla"))
    out = tmp_path / "smoke"
    args = ["--config", "smoke", "--rows", "linear-sv", "--reps", "0:1", "--workers", "1"]
    study.main([str(out), "--cache", str(tmp_path / "cache"), *args])
    (line,) = out.with_suffix(".jsonl").read_text().splitlines()
    rec = json.loads(line)
    design = study.CONFIGS["smoke"].design
    assert (rec["row"], rec["rep"], rec["design"]) == ("linear-sv", 0, design.key())
    assert set(rec["arms"]) == {"Linear × Constant", "Linear × SV"}
    assert all(a["origins"] == 10 and a["cached"] == 0 for a in rec["arms"].values())
    for p in rec["pairs"]:
        assert p["verdict"] in dgp.VERDICTS and p["origins"] == 10
        dgp.classify(p)
    rep = json.loads(out.with_suffix(".json").read_text())
    assert not rep["registered"] and rep["summary"][0]["n"] == 1
    # the smoke budget is explicit, so the SV arm is not the D-35 default
    model = dgp.models(dgp.get_row("linear-sv"), design)["Linear × SV"]
    assert model.mcmc == dataclasses.replace(design.mcmc)
