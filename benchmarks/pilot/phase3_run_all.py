"""Run the Phase 3 pilot (WP36; ``docs/phase3/pilot.md``): the mixing campaign in parallel
worker processes, then the cost measurements one at a time on an idle machine.

* ``mixing``: one ``phase3_mixing.py`` process per (size, configuration, seed), ``--workers`` at
  a time (ESS and R-hat do not depend on the load, so the workers may share the machine). The
  jobs are ordered so that seed 0 of every BENCH-P2-shape configuration runs first.
* ``costs``: ``phase3_costs.py`` for the §10 items, sequentially, each block load-gated
  (``--gate``, ``--wait``); never run it while a mixing campaign (or anything else) runs.

Every process writes its JSON (machine fingerprint, load before and after) into ``--out``; logs
go to ``--out/logs``. Existing results are skipped, so an interrupted campaign can be resumed.

Usage::

    python -I benchmarks/pilot/phase3_run_all.py --out DIR --only mixing --workers 5
    python -I benchmarks/pilot/phase3_run_all.py --out DIR --only costs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

GL = ("hs", "ng", "r2d2", "dl", "ssvs", "ssvs_beta")
CONFIGS = (
    *(f"{f}_{v}" for v in ("const", "sv") for f in GL),
    "mn_const", "mn_sv", "csv_conj", "csv_struct", "csv_conj325", "csv_struct325",
)  # fmt: skip
SEEDS = {"qd7": (0, 1, 2, 3, 4), "bp2": (0, 1, 2)}
#: R29 (the shared-factor path's two-block Gibbs at ``q ≈ T``): ``n = 20, p = 4, T = 120``.
R29 = ("sim20x120", ("csv_conj", "csv_struct"), (0, 1, 2))
#: R22 (the CommonSV ``h`` acceptance at ``T = 600``): FRED-MD medium, 4 × 20,000.
R22 = ("md7", ("csv_conj", "csv_struct"), (0, 1, 2), 20_000)
#: The GK form (reduced, ``scale="none"``, mean 0) × SV at ``n = 21`` on real data
#: (``phase3_common.fredqd_21``, ``T_eff = 208``), seed 0, 4 × 20,000: does the structural-form
#: evidence transfer to the GK run (§9.10)?
GK21 = ("qd21", ("red_mn_sv", "red_hs_sv", "red_ng_sv"), (0,), 20_000)
#: Single-threaded XLA for the mixing workers (``--single-thread``): many workers then share
#: the machine without oversubscribing its cores (ESS and R-hat do not depend on threading).
XLA_SINGLE = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

#: §10 cost items (``phase3_costs.py`` argument lists), BENCH-P2 shape unless named.
COSTS: tuple[tuple[str, list[str]], ...] = (
    ("gl_const", ["--size", "bp2", "--configs", ",".join(f"{f}_const" for f in GL)]),
    ("gl_sv", ["--size", "bp2", "--configs", ",".join(f"{f}_sv" for f in GL)]),
    ("mn", ["--size", "bp2", "--configs", "mn_const,mn_sv"]),
    ("mn_cholesky", ["--size", "bp2", "--configs", "mn_const", "--const-path", "cholesky"]),
    ("csv", ["--size", "bp2", "--configs", "csv_conj,csv_struct"]),
    ("csv_weighted", ["--size", "bp2", "--configs", "csv_conj", "--const-path", "cholesky"]),
    ("qd7", ["--size", "qd7", "--configs", ",".join(CONFIGS)]),
)


def mixing_jobs(draws: int, bp2_seeds: tuple[int, ...] = SEEDS["bp2"],
                bp2_late_draws: int | None = None) -> list[list[str]]:  # fmt: skip
    """The campaign's jobs in order. ``bp2_late_draws`` caps the draws of the BENCH-P2-shape
    seeds after the first (the adaptive rule of ``pilot.md`` §2)."""
    jobs = []

    def job(size: str, config: str, seed: int, d: int) -> None:
        jobs.append(["--size", size, "--configs", config, "--seeds", str(seed), "--draws", str(d)])

    for c in CONFIGS:
        job("bp2", c, bp2_seeds[0], draws)
    for s in SEEDS["qd7"]:
        for c in CONFIGS:
            job("qd7", c, s, draws)
    for s in R29[2]:
        for c in R29[1]:
            job(R29[0], c, s, draws)
    for c in R22[1]:
        for s in R22[2]:
            job(R22[0], c, s, R22[3])
    for c in GK21[1]:
        for s in GK21[2]:
            job(GK21[0], c, s, GK21[3])
    for s in bp2_seeds[1:]:
        for c in CONFIGS:
            job("bp2", c, s, bp2_late_draws or draws)
    return jobs


def _run(script: str, args: list[str], out: Path, tag: str, *, skip_started: bool = False,
         env: dict[str, str] | None = None) -> int:  # fmt: skip
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{tag}.log"
    if skip_started and log.exists():  # started by an earlier driver (resumed campaign)
        return 0
    cmd = [sys.executable, "-I", str(HERE / script), "--out", str(out), *args]
    with log.open("a") as fh:
        fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, check=False, env=env
        ).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="mixing", help="mixing, costs or both (comma-separated)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draws", type=int, default=50_000)
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--wait", type=float, default=1800.0)
    ap.add_argument("--single-thread", action="store_true", help=f"XLA_FLAGS={XLA_SINGLE!r}")
    ap.add_argument("--skip-started", action="store_true", help="skip jobs that have a log")
    ap.add_argument("--sizes", default="", help="only these sizes (comma-separated)")
    ap.add_argument("--bp2-seeds", default="0,1,2")
    ap.add_argument("--bp2-late-draws", type=int, default=0)
    args = ap.parse_args(argv)
    only = set(args.only.split(","))
    status = 0
    if "mixing" in only:
        bp2 = tuple(int(x) for x in args.bp2_seeds.split(","))
        jobs = mixing_jobs(args.draws, bp2, args.bp2_late_draws or None)
        if args.sizes:
            jobs = [j for j in jobs if j[1] in set(args.sizes.split(","))]
        env = None
        if args.single_thread:
            env = {**os.environ, "XLA_FLAGS": XLA_SINGLE}

        def one(j: list[str]) -> int:
            tag = f"mix_{j[1]}_{j[3]}_s{j[5]}"
            return _run("phase3_mixing.py", j, args.out, tag, skip_started=args.skip_started,
                        env=env)  # fmt: skip

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            codes = list(ex.map(one, jobs))
        status |= int(any(codes))
    if "costs" in only:
        for tag, extra in COSTS:
            gate = ["--gate", str(args.gate), "--wait", str(args.wait)]
            status |= int(_run("phase3_costs.py", [*extra, *gate], args.out, f"cost_{tag}") != 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
