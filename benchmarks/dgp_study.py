"""DGP study of the verdict engine (SPEC §10.5): size and power of ``compare().verdict()``.

The study is pre-registered in ``benchmarks/reports/dgp_study_preregistration.md``. The pilot,
from the repository root, on a clean working tree at the commit that adds the pre-registration::

    uv run python benchmarks/dgp_study.py benchmarks/reports/dgp_study_pilot --config pilot \\
        --workers 12 --cache ~/.cache/rai-bvar-dgp-$(git rev-parse --short HEAD)

and the full study the same way with ``--config full`` (``benchmarks/reports/dgp_study``). The
pilot's replications are the first ones of the full study (same design, seeds and cache keys), so
the full run reuses the pilot's records and runs the remaining replications only.

Usage::

    python benchmarks/dgp_study.py OUT [--config full|pilot|smoke] [--rows ROW ...]
        [--reps START:STOP] [--workers N] [--cache DIR] [--report-only] [--note TEXT]

writes ``OUT.jsonl`` (one record per replication, appended as each finishes), ``OUT.md`` (the
report) and ``OUT.json`` (every number in it). Rerunning the same command resumes: replications
whose record exists for the same design are skipped, and every backtest origin is cached
(``--cache``). ``--reps START:STOP`` runs a sub-range of replications (to split the study over
machines; concatenate the ``.jsonl`` files and pass ``--report-only``). Results do not depend on
``--workers``: a replication runs in one process, from its own seeds.

What it does (``rai_bvar.validation.dgp``): per replication, simulate ``T`` observations from the
row's DGP, run ``rb.compare()`` over the row's arms (library defaults; rolling window; the
SPEC §9.5 verdict with its defaults) and record every scored pair's verdict, the verdict with the
convergence gate off (secondary) and the converged fractions. The report gives, per row, pair and
horizon, the rate of the scored verdict (size or power) with its binomial Monte Carlo standard
error and Wilson 95% interval, the rates of all three verdicts with the gate on and off, why the
inconclusive verdicts were inconclusive, the convergence of every arm and the measured time per
replication, with the wall-time estimate of the full study. The report is labelled the
pre-registered run only when every record comes from one commit with a clean working tree, the
design is the registered one and every runnable row has all the config's replications.

``--config smoke`` only checks that the script runs (2 chains × 100 draws, every 8th origin, two
replications, a seed of its own so no registered replication is scored); its numbers mean nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rai_bvar as rb
from rai_bvar.settings import MCMCSettings
from rai_bvar.validation import dgp

PREREGISTRATION = "benchmarks/reports/dgp_study_preregistration.md"
REPO = Path(__file__).resolve().parents[1]
REGISTERED = dgp.Design()  # the pre-registered study cell (n = 3, T = 200, Rolling(120), p = 2)
# spec_hash(model)[:16] of the registered arms (library defaults at the pre-registration's commit)
REGISTERED_SPEC_HASHES = {
    "Linear × Constant": "8aebe5adec44e966",
    "Linear × SV": "b3daa828453542ed",
    "Linear × SV(outliers)": "7f0aeb733eb2d07e",
}


@dataclass(frozen=True)
class Config:
    name: str
    design: dgp.Design
    reps: int  # replications per row


CONFIGS = {
    "full": Config("full", REGISTERED, 200),
    "pilot": Config("pilot", REGISTERED, 24),  # replications 0–23 of the full study
    "smoke": Config(
        "smoke",
        dataclasses.replace(
            REGISTERED,
            seed=1,
            origin_step=8,
            draws=200,
            mcmc=MCMCSettings(chains=2, draws=100, burn=100),
        ),
        2,
    ),
}
REGISTERED_CONFIGS = ("pilot", "full")


def git_state() -> dict[str, Any]:
    """The commit (``git rev-parse HEAD``), whether tracked files differ from it, and the SHA-256
    of ``git diff HEAD`` when they do (as ``benchmarks/clark_ravazzolo.py``)."""
    import hashlib

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, check=True
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        diff = hashlib.sha256(git("diff", "HEAD")).hexdigest() if dirty else None
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None, "git_diff_sha256": None}
    try:
        git("cat-file", "-e", f"HEAD:{PREREGISTRATION}")
        prereg = True
    except subprocess.CalledProcessError:
        prereg = False
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_sha256": diff,
        "preregistration_in_commit": prereg,
    }


def machine() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
    }
    if sys.platform == "darwin":
        for key, name in (
            ("cpu", "machdep.cpu.brand_string"),
            ("performance_cores", "hw.perflevel0.physicalcpu"),
            ("efficiency_cores", "hw.perflevel1.physicalcpu"),
        ):
            try:
                out[key] = subprocess.run(
                    ["sysctl", "-n", name], capture_output=True, check=True, text=True
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                pass
    else:
        try:
            with open("/proc/cpuinfo") as f:
                out["cpu"] = next(
                    line.split(":", 1)[1].strip() for line in f if line.startswith("model name")
                )
        except (OSError, StopIteration):
            pass
    return out


# --------------------------------------------------------------------------- workers


def _init_worker() -> None:
    warnings.simplefilter("ignore", UserWarning)  # the backtests record theirs per origin
    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache:
        rb.enable_compilation_cache(cache, min_compile_secs=0)


def _run(row_name: str, rep: int, design: dgp.Design, cache: str, commit: str) -> dict[str, Any]:
    record = dgp.run_replication(dgp.get_row(row_name), rep, design, cache=cache)
    record["git_commit"] = commit
    record["pid"] = os.getpid()
    return record


def _execute(
    tasks: list[tuple[str, int]], cfg: Config, args: argparse.Namespace, git: dict[str, Any]
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Run the replications, yielding each record as it finishes: in this process with one
    worker, else in ``spawn``-started worker processes (forking after JAX starts is unsafe)."""
    run = (cfg.design, str(args.cache), git["git_commit"])
    if args.workers == 1:
        _init_worker()
        for name, rep in tasks:
            yield name, rep, _run(name, rep, *run)
        return
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init_worker) as pool:
        futures = {pool.submit(_run, name, rep, *run): (name, rep) for name, rep in tasks}
        for fut in as_completed(futures):
            yield (*futures[fut], fut.result())


def load_records(path: Path, design: dgp.Design) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["design"] == design.key():
                    out.append(rec)
    return out


# --------------------------------------------------------------------------- report


def _fmt_rate(r: dict[str, Any]) -> str:
    return f"{r['rate']:.3f} ± {r['se']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def estimate(
    records: list[dict[str, Any]], cfg_full: Config, workers: int, wall: float | None
) -> dict[str, Any]:
    """Wall time of the full study from the measured replications (same machine and workers).

    Per row, the mean in-worker seconds of a replication (measured under this run's concurrency)
    times the full study's missing replications, summed and divided by the worker count.
    """
    per_row: dict[str, list[float]] = {}
    for r in records:
        if all(a["cached"] == 0 for a in r["arms"].values()):
            per_row.setdefault(r["row"], []).append(r["seconds"])
    rows = {}
    task_seconds = 0.0
    for name, secs in per_row.items():
        done = len({r["rep"] for r in records if r["row"] == name})
        missing = max(cfg_full.reps - done, 0)
        mean = sum(secs) / len(secs)
        rows[name] = {"replications_timed": len(secs), "seconds_mean": mean, "missing": missing}
        task_seconds += mean * missing
    total_full = sum(v["seconds_mean"] for v in rows.values()) * cfg_full.reps
    return {
        "workers": workers,
        "rows": rows,
        "remaining_hours": task_seconds / max(workers, 1) / 3600.0,
        "full_hours": total_full / max(workers, 1) / 3600.0,
        "single_process_full_hours": total_full / 3600.0,
        "this_run_wall_hours": None if wall is None else wall / 3600.0,
    }


def label(
    records: list[dict[str, Any]], cfg: Config, rows: list[dgp.Row], meta_git: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Whether the records are the pre-registered run of ``cfg``, and why not."""
    reasons = []
    if cfg.name not in REGISTERED_CONFIGS or cfg.design != REGISTERED:
        reasons.append(f"config {cfg.name!r} is not a registered configuration")
    if not meta_git.get("preregistration_in_commit"):
        reasons.append(f"{PREREGISTRATION} is not in the commit")
    commits = {r.get("git_commit") for r in records}
    if len(commits) != 1:
        reasons.append(f"records come from {len(commits)} commits: {sorted(map(str, commits))}")
    if any(r.get("git_dirty") for r in records):
        reasons.append("some records were produced with modified tracked files")
    if cfg.name in REGISTERED_CONFIGS:
        off = sorted(
            {
                f"{arm} {a.get('spec_hash', '?')[:16]}"
                for r in records
                for arm, a in r["arms"].items()
                if a.get("spec_hash", "")[:16] != REGISTERED_SPEC_HASHES.get(arm)
            }
        )
        if off:
            reasons.append(f"arms with unregistered specs: {', '.join(off)}")
    for row in rows:
        have = {r["rep"] for r in records if r["row"] == row.name}
        if have != set(range(cfg.reps)):
            reasons.append(
                f"row {row.name}: {len(have & set(range(cfg.reps)))} of {cfg.reps} replications"
            )
    return (not reasons, reasons)


def build_report(
    records: list[dict[str, Any]],
    cfg: Config,
    rows: list[dgp.Row],
    pending: list[tuple[dgp.Row, str]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    registered, reasons = label(records, cfg, rows, meta["git"])
    return {
        "title": f"DGP study of the verdict engine: {cfg.name}",
        "registered": registered,
        "not_registered_because": reasons,
        "preregistration": PREREGISTRATION,
        "config": {"name": cfg.name, "reps": cfg.reps, "design": cfg.design.describe()},
        "design_key": cfg.design.key(),
        "dgps": {
            name: (None if d is None else d.describe())
            for name, d in dgp.DGPS.items()
            if any(r.dgp == name for r in rows)
        },
        "rows": [
            {
                "name": r.name,
                "dgp": r.dgp,
                "spec": r.spec,
                "arms": list(r.arms),
                "baseline": r.baseline,
                "pairs": [dataclasses.asdict(p) for p in r.pairs],
            }
            for r in rows
        ],
        "pending": [{"name": r.name, "spec": r.spec, "status": s} for r, s in pending],
        "nominal": dgp.NOMINAL,
        "verdict_defaults": dgp.VERDICT_DEFAULTS,
        "summary": dgp.summarise(records),
        "family": [
            f
            for r in rows
            if any(p.expect == "not_better" for p in r.pairs)
            for f in dgp.family(records, r.name)
        ],
        "convergence": dgp.convergence(records),
        "estimate": estimate(records, CONFIGS["full"], meta["workers"], meta.get("wall_seconds")),
        "meta": meta,
    }


def write_report(rep: dict[str, Any], out: Path) -> None:
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(markdown(rep))


def markdown(rep: dict[str, Any]) -> str:
    m, cfg, d = rep["meta"], rep["config"], rep["config"]["design"]
    lines = [f"# {rep['title']}", ""]
    if rep["registered"]:
        lines.append(
            f"**The pre-registered {cfg['name']} run** ({rep['preregistration']}): every record "
            f"from commit `{m['git'].get('git_commit', '?')[:12]}` with a clean working tree, the "
            "registered design, all replications of every runnable row."
        )
    else:
        lines.append(
            "**Not the pre-registered run:** " + "; ".join(rep["not_registered_because"]) + "."
        )
    if cfg["name"] == "pilot":
        lines += [
            "",
            f"This is the **pilot**: the first {cfg['reps']} replications per row of the full "
            "study, at its settings. Its rates are indicative only (see their Monte Carlo errors); "
            "the full study's reading is the one pre-registered.",
        ]
    if cfg["name"] == "smoke":
        lines += ["", "**Smoke run: the numbers mean nothing** (tiny chains, few origins)."]
    lines += [
        "",
        "## Setup",
        "",
        f"- Design `{rep['design_key']}`: n = 3, T = {d['T']}, VAR({d['lags']}) DGPs and models, "
        f"`Rolling({d['window']})`, origins every {d['origin_step']} period(s) from the first full "
        f"window to the second-to-last period, horizons {d['horizons']}, {d['draws']} forecast "
        f"draws per origin, MCMC budget "
        + ("the library default per composition (D-35)" if d["mcmc"] is None else f"`{d['mcmc']}`")
        + f", study seed {d['seed']}.",
        f"- {cfg['reps']} replications per row; `compare()` and `verdict()` with their defaults "
        f"(`{rep['verdict_defaults']}`); nominal level {rep['nominal']:.2f} (the rule's one-sided "
        "DM level).",
        f"- Code: commit `{m['git'].get('git_commit')}`"
        + (" (tracked files modified)" if m["git"].get("git_dirty") else "")
        + f"; rai-bvar {m['versions']['rai_bvar']}, JAX {m['versions']['jax']}.",
        f"- Machine: {m['machine'].get('cpu', '?')} ({m['machine'].get('cpus')} CPUs"
        + (
            f": {m['machine']['performance_cores']} performance, {m['machine']['efficiency_cores']} efficiency"
            if "performance_cores" in m["machine"]
            else ""
        )
        + f"), {m['machine']['platform']}; {m['workers']} worker processes; load averages "
        f"{m.get('load_start')} at the start and {m.get('load_end')} at the end.",
    ]
    if m.get("wall_seconds") is not None:
        lines.append(
            f"- Wall time of this invocation: {m['wall_seconds'] / 3600:.2f} h "
            f"({m['wall_seconds'] / 60:.0f} min) for "
            f"{m['ran']} replications ({m['skipped']} read from earlier records)."
        )
    for note in m.get("notes", []):
        lines.append(f"- Note: {note}")

    lines += ["", "## Size and power", ""]
    lines += [
        "Rate of the scored verdict over replications, ± its binomial Monte Carlo standard error, "
        "[Wilson 95% interval]. *Size* is the rate of a false `better` (expectation `not_better`), "
        "*power* the rate of `better` (expectation `better`); for `not_worse` the scored rate is "
        "that of `worse`. Reading (pre-registered): a size or `worse` rate is *above nominal* "
        f"only when its interval lies above {rep['nominal']:.2f}; power is descriptive. The gate-off "
        "columns are `verdict(min_converged=0)`: secondary, not the engine's verdict.",
        "",
        "| Row | Pair | h | Expect | Scored | Rate (gate on) | Reading | better / worse / inconcl. (gate on) | better / worse / inconcl. (gate off) |",
        "| :-- | :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- |",
    ]
    for s in rep["summary"]:
        on, off = s["gate_on"], s["gate_off"]
        lines.append(
            f"| {s['row']} | {s['model']} vs {s['reference']} | {s['horizon']} | {s['expect']} | "
            f"{s['target']} ({s['n']}) | {_fmt_rate(s)} | {s['reading']} | "
            f"{on['better']['k']} / {on['worse']['k']} / {on['inconclusive']['k']} | "
            f"{off['better']['k']} / {off['worse']['k']} / {off['inconclusive']['k']} |"
        )
    if rep["family"]:
        lines += [
            "",
            "Family-wise false-`better` rate (any `not_better` pair of the row `better`):",
            "",
            "| Row | h | Gate on | Gate off |",
            "| :-- | --: | :-- | :-- |",
        ]
        for f in rep["family"]:
            lines.append(
                f"| {f['row']} | {f['horizon']} | {_fmt_rate(f['verdict'])} | {_fmt_rate(f['verdict_gate_off'])} |"
            )

    lines += [
        "",
        "## Why the verdicts came out as they did",
        "",
        "Counts per class (`rai_bvar.validation.dgp.classify`): `better:test` / `better:mcs` / "
        "`better:both` name the route of condition 3 that held (valid DM test, MCS separation, "
        "both); `gate` is inconclusive only because of the convergence gate; `imprecise` is "
        "|ΔLPD| ≤ 3 MCSE; `no_route` is precise but neither route separates the pair. Mean "
        "ΔLPD (summed over origins) and its mean MCSE; `gate pass` is the share of replications "
        "whose two arms both converged at ≥ 95% of origins.",
        "",
        "| Row | Pair | h | Classes | Mean ΔLPD (median) | Mean MCSE | Mean origins | Gate pass |",
        "| :-- | :-- | --: | :-- | :-- | --: | --: | --: |",
    ]
    for s in rep["summary"]:
        classes = ", ".join(f"{k} {v}" for k, v in s["classes"].items())
        lines.append(
            f"| {s['row']} | {s['model']} vs {s['reference']} | {s['horizon']} | {classes} | "
            f"{s['dlpd_mean']:+.2f} ({s['dlpd_median']:+.2f}) | {s['dlpd_mcse_mean']:.2f} | "
            f"{s['origins_mean']:.1f} | {s['gate_pass']:.2f} |"
        )

    lines += [
        "",
        "## Convergence and cost per arm",
        "",
        "Converged share: the fraction of a replication's origins whose fit passed the "
        "convergence check (DESIGN §8.2); gate pass: the share of replications at ≥ 95%; worst: "
        "the primary quantities most often named worst by a failed check, with the number of "
        "failed fits (over all replications; both horizons share the fits).",
        "",
        "| Row | Arm | h | Replications | Mean converged share | Min | Gate pass | s / fitted origin | Worst |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |",
    ]
    for c in rep["convergence"]:
        worst = ", ".join(f"{q} {k}" for q, k in c["worst"].items()) or "–"
        lines.append(
            f"| {c['row']} | {c['arm']} | {c['horizon']} | {c['n']} | {c['converged_mean']:.3f} | "
            f"{c['converged_min']:.3f} | {c['gate_pass']:.2f} | {c['seconds_per_origin']:.2f} | "
            f"{worst} |"
        )

    e = rep["estimate"]
    lines += [
        "",
        "## Wall time of the full study",
        "",
        f"Measured per replication under this run's concurrency ({e['workers']} workers), "
        "replications with no cached origin only:",
        "",
        "| Row | Timed replications | Mean seconds per replication | Missing for the full study |",
        "| :-- | --: | --: | --: |",
    ]
    for name, r in sorted(e["rows"].items(), key=lambda kv: dgp._row_order(kv[0])):
        lines.append(
            f"| {name} | {r['replications_timed']} | {r['seconds_mean']:.0f} | {r['missing']} |"
        )
    lines += [
        "",
        f"- Full study ({CONFIGS['full'].reps} replications per row) from scratch: about "
        f"**{e['full_hours']:.1f} h** on {e['workers']} workers "
        f"({e['single_process_full_hours']:.0f} h of worker time).",
        f"- Remaining after the records above: about **{e['remaining_hours']:.1f} h**.",
        "- Both are extrapolations at the same machine, load and worker count, not measurements.",
    ]
    if rep["pending"]:
        lines += ["", "## Rows not run (pending later phases)", ""]
        for p in rep["pending"]:
            lines.append(f"- `{p['name']}` ({p['spec']}): {p['status']}.")
    lines += ["", "## DGPs", ""]
    for name, desc in rep["dgps"].items():
        lines.append(f"- `{name}`: `{json.dumps(desc)}`")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def _load() -> str:
    try:
        return " ".join(f"{x:.2f}" for x in os.getloadavg())
    except OSError:
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path, help="report path without suffix (.jsonl, .md, .json)")
    p.add_argument("--config", choices=sorted(CONFIGS), default="pilot")
    p.add_argument(
        "--rows", nargs="*", default=None, help="row names (default: every runnable row)"
    )
    p.add_argument("--reps", default=None, help="START:STOP, a sub-range of the replications")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "rai-bvar-dgp")
    p.add_argument("--report-only", action="store_true", help="aggregate the existing records")
    p.add_argument("--note", action="append", default=[], help="a sentence added to the report")
    args = p.parse_args(argv)
    cfg = CONFIGS[args.config]
    if args.workers < 1:
        p.error("--workers must be at least 1")
    names = args.rows or [r.name for r in dgp.ROWS]
    rows, pending = [], []
    for name in names:
        row = dgp.get_row(name)
        status = dgp.row_status(row)
        (pending.append((row, status)) if status else rows.append(row))
    start, stop = 0, cfg.reps
    if args.reps:
        a, b = args.reps.split(":")
        start, stop = int(a or 0), int(b or cfg.reps)
    git = git_state()
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.0f}s] {msg}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.out.with_suffix(".jsonl")
    records = load_records(records_path, cfg.design)
    done = {(r["row"], r["rep"]) for r in records}
    tasks = [
        (row.name, rep)
        for rep in range(start, stop)
        for row in sorted(rows, key=lambda r: -len(r.arms))  # the longest rows first
        if (row.name, rep) not in done
    ]
    meta: dict[str, Any] = {
        "git": git,
        "machine": machine(),
        "versions": {"rai_bvar": rb.__version__},
        "workers": args.workers,
        "load_start": _load(),
        "cache": str(args.cache),
        "notes": list(args.note),
        "skipped": sum(1 for row in rows for rep in range(start, stop) if (row.name, rep) in done),
    }
    log(
        f"config {cfg.name} (design {cfg.design.key()}), rows {[r.name for r in rows]}, "
        f"replications {start}:{stop}, {len(tasks)} to run, {meta['skipped']} recorded; "
        f"commit {git['git_commit'][:12]}{' (dirty)' if git['git_dirty'] else ''}; "
        f"{args.workers} workers; cache {args.cache}"
    )
    for row, status in pending:
        log(f"row {row.name}: {status}")

    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        # before any worker starts: spawned workers inherit the environment and share the cache
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.cache / "xla")
    import jax

    meta["versions"]["jax"] = jax.__version__
    if not args.report_only and tasks:
        args.cache.mkdir(parents=True, exist_ok=True)
        with open(records_path, "a") as sink:
            for i, (name, rep, rec) in enumerate(_execute(tasks, cfg, args, git), 1):
                rec["git_dirty"] = git["git_dirty"]
                sink.write(json.dumps(rec, default=str) + "\n")
                sink.flush()
                verdicts = ", ".join(
                    f"h{q['horizon']} {q['model']}: {q['verdict']}" for q in rec["pairs"]
                )
                log(f"{i}/{len(tasks)} {name} rep {rep}: {rec['seconds']:.0f}s; {verdicts}")
        meta["wall_seconds"] = time.time() - t0
        meta["ran"] = len(tasks)
    meta["load_end"] = _load()
    records = [
        r
        for r in load_records(records_path, cfg.design)
        if r["row"] in {row.name for row in rows} and r["rep"] < cfg.reps
    ]
    rep = build_report(records, cfg, rows, pending, meta)
    write_report(rep, args.out)
    log(
        f"report: {args.out.with_suffix('.md')} ({'registered' if rep['registered'] else 'not registered'})"
    )
    for s in rep["summary"]:
        log(
            f"{s['row']} {s['model']} vs {s['reference']} h={s['horizon']}: {s['target']} "
            f"{_fmt_rate(s)} ({s['reading']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
