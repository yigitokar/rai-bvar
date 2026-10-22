"""Run every WP3 pilot measurement in sequence, each in its own ``python -I`` process.

Results (one JSON per run, with machine fingerprint, load average and versions) and logs go to
``--out``. ``docs/phase2/09_pilot.md`` records the numbers of the reference run and the defaults
they freeze. Run on an idle machine; the logs record the 1-minute load average before each run.

``--repeat N`` runs the selected list N times in interleaved order (``sweep.py`` runs get the tag
suffix ``_r<i>``), so A/B comparisons of sweep variants see the same load drift (09_pilot.md §4.1).

Usage: python -I run_all.py --out DIR [--only sv_block,sweep_ref,...] [--quick] [--repeat N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS: tuple[tuple[str, str, list[str]], ...] = (
    ("sv_block", "sv_block.py", []),
    ("sv_acceptance", "sv_acceptance.py", []),
    ("transitory", "transitory.py", []),
    ("prng", "prng.py", []),
    ("gram", "gram.py", []),
    ("buckets", "buckets.py", []),
    ("sweep_ref", "sweep.py", ["--tag", "ref", "--chunks", "50,125,250,500"]),
    ("sweep_nonexact", "sweep.py", ["--tag", "nonexact", "--exact", "0"]),
    ("sweep_philox", "sweep.py", ["--tag", "philox", "--prng", "philox4x32"]),
    ("sweep_buckets2", "sweep.py", ["--tag", "buckets2", "--buckets", "2"]),
    ("sweep_unroll4", "sweep.py", ["--tag", "unroll4", "--unroll", "4"]),
    ("sweep_dev4", "sweep.py", ["--tag", "dev4", "--devices", "4"]),
    (
        "sweep_svt_jax",
        "sweep.py",
        ["--tag", "svt_jax", "--transitory", "t", "--tau-sampler", "jax", "--nu-form", "logaddexp"],
    ),
    (
        "sweep_svt_mt2",
        "sweep.py",
        ["--tag", "svt_mt2", "--transitory", "t", "--tau-sampler", "mt2", "--nu-form", "log_sum"],
    ),
)
QUICK = {  # small settings for a smoke run of the whole pipeline
    "sv_block.py": ["--sweeps", "50", "--reps", "2"],
    "sv_acceptance.py": ["--R", "2", "--C", "2", "--burn", "1000", "--keep", "2000"],
    "transitory.py": ["--iters", "10", "--reps", "2"],
    "prng.py": ["--iters", "20", "--reps", "2"],
    "gram.py": ["--iters", "2", "--reps", "2"],
    "buckets.py": ["--iters", "10", "--reps", "2"],
    "sweep.py": ["--sweeps", "100", "--reps", "2", "--warm", "50", "--draws", "250"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in a.only.split(",") if s}
    for rep in range(a.repeat):
        for name, script, args in RUNS:
            if only and name not in only:
                continue
            run(name, script, args, rep, a, out)


def run(
    name: str, script: str, args: list[str], rep: int, a: argparse.Namespace, out: Path
) -> None:
    extra = QUICK.get(script, []) if a.quick else []
    if a.repeat > 1:
        name = f"{name}_r{rep}"
        if script == "sweep.py":
            extra = [*extra, "--tag", name.removeprefix("sweep_")]
    cmd = [sys.executable, "-I", str(HERE / script), *args, *extra, "--out", str(out)]
    load = os.getloadavg()[0]
    print(f"=== {name} (load {load:.2f}): {' '.join(cmd[2:])}", flush=True)
    t0 = time.perf_counter()
    with open(out / f"{name}.log", "w") as log:
        log.write(f"# load average before run: {os.getloadavg()}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(
        f"    {status} in {time.perf_counter() - t0:.1f} s; log {out / f'{name}.log'}", flush=True
    )


if __name__ == "__main__":
    main()
