"""The shards of the weekly tier (the ``nightly`` job of ``.github/workflows/ci.yml``, DESIGN §9.10).

The tier is ``pytest -m "(slow or network) and not bench"`` over the whole suite. One x86-64 runner
does not finish it within GitHub's 6 h job limit (run 40052069815: 330 min timeout at 53%, EPYC
7763, 2 vCPU), so the job runs as a matrix of shards. Every shard runs the same marker expression;
only the paths, ``-k`` and ``--ignore`` differ. A directory split in two uses ``-k E`` and
``-k "not E"`` on the same path, and ``rest`` is the suite minus the split directories, so a new
directory or test lands in some shard without an edit here.

``check`` collects the tier and every shard with ``pytest --collect-only`` and fails unless the
shards partition the tier: every test of the tier in exactly one shard, no test outside it. The
workflow's ``nightly-plan`` job runs it before any shard starts and passes the matrix on::

    uv run python tools/weekly_shards.py check --markers "(slow or network) and not bench"
    uv run python tools/weekly_shards.py check --markers "$TIER_MARKERS" --github-output "$GITHUB_OUTPUT"

The selection is by directory and composition only, from the per-test times of runs 40052069815
(EPYC 7763, 2 vCPU; up to the timeout) and 40106532730 (EPYC 9V45, 4 vCPU; complete); minutes on
the 9V45, cold XLA cache:

=====================  =====================================================  =========
shard                  contents                                               9V45 min
=====================  =====================================================  =========
``geweke-structural``  G-full, inexact and G-small rows of structural models  55
``geweke-other``       reduced and ``zero_sv`` rows, the block configurations 40
``sbc-structural``     SBC of structural models, the structural-step tier B   43
``sbc-other``          SBC of the other models, the rest of the block tier    41
``unit``               ``tests/unit`` (EB convergence: 47 min)                53
``rest``               analytic, contract, integration, mcmc, mutation,       41
                       parity, priors, validation
=====================  =====================================================  =========

The EPYC 7763 takes 2.0–2.9× as long, and a Gandy–Scott rerun (DESIGN §9.5) of an ``outliers_t``
row adds up to an hour there (the reruns depend on the CPU's rounding).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

SPLIT = ("tests/geweke", "tests/sbc", "tests/unit")  # the directories with shards of their own
SHARDS: dict[str, list[str]] = {
    "geweke-structural": ["tests/geweke", "-k", "structural"],
    "geweke-other": ["tests/geweke", "-k", "not structural"],
    "sbc-structural": ["tests/sbc", "-k", "structural"],
    "sbc-other": ["tests/sbc", "-k", "not structural"],
    "unit": ["tests/unit"],
    "rest": ["tests", *(f"--ignore={d}" for d in SPLIT)],
}

_COUNT = re.compile(r"^(\d+)(?:/\d+)? tests? collected")


def collect(markers: str, args: list[str]) -> list[str]:
    """Node ids that ``pytest -m markers *args`` would run, in collection order."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    out = subprocess.run([*cmd, "-m", markers, *args], capture_output=True, text=True)
    if out.returncode not in (0, 5):  # 5: nothing selected
        sys.exit(f"collection failed ({out.returncode}): {args}\n{out.stdout}{out.stderr}")
    lines = out.stdout.splitlines()
    ids = [line for line in lines[: lines.index("")] if "::" in line] if "" in lines else []
    counts = [int(m.group(1)) for line in lines if (m := _COUNT.match(line))]
    expected = counts[-1] if counts else 0  # "no tests collected" when a shard selects nothing
    if len(ids) != expected:
        sys.exit(f"read {len(ids)} node ids, pytest reports {expected}: {args}")
    return ids


def check(markers: str) -> list[tuple[str, int]]:
    """Fail unless the shards partition the tier; return each shard's test count."""
    names = list(SHARDS)
    with ThreadPoolExecutor() as pool:
        tier, *parts = pool.map(lambda a: collect(markers, a), [[], *SHARDS.values()])
    owners = Counter(nodeid for ids in parts for nodeid in ids)
    twice = sorted(n for n, k in owners.items() if k > 1)
    missing = sorted(set(tier) - set(owners))
    extra = sorted(set(owners) - set(tier))
    for label, bad in (("in more than one shard", twice), ("in no shard", missing),
                       ("outside the tier", extra)):  # fmt: skip
        if bad:
            sys.exit(f"{len(bad)} tests {label}, e.g. {bad[:5]}")
    sizes = [(name, len(ids)) for name, ids in zip(names, parts, strict=True)]
    if empty := [name for name, n in sizes if n == 0]:
        sys.exit(f"empty shards: {empty}")
    return [*sizes, ("tier", len(tier))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmd = sub.add_parser("check", help="verify the partition; write the matrix to --github-output")
    cmd.add_argument("--markers", required=True, help="the tier's marker expression (-m)")
    cmd.add_argument("--github-output", help="append matrix=<json> to this file ($GITHUB_OUTPUT)")
    opts = parser.parse_args()
    for name, n in check(opts.markers):
        print(f"{name:20s} {n:4d} {' '.join(SHARDS.get(name, ['-m', opts.markers]))}")
    matrix = {"include": [{"shard": name, "args": args} for name, args in SHARDS.items()]}
    if opts.github_output:
        with open(opts.github_output, "a") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")


if __name__ == "__main__":
    main()
