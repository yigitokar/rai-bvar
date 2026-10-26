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

=========================  ============================================================================
shard                  contents (estimate, min)
=====================  ============================================================================
``geweke-p3a1``        G-full of the structural rows 301–308 (≈ 75)
``geweke-p3a2``        G-full of 309, 310, 317–319, 326, 330; G-small of 401–408 (≈ 80)
``geweke-p3b1``        G-full of the reduced rows 311–316, 327 (≈ 75)
``geweke-p3b2``        G-full of the CommonSV rows 320–325, 328, 329 (≈ 85)
``sbc-p3a1``–``a5``    SBC of the structural rows, 2–4 rows each (51–86)
``sbc-p3b1``–``b8``    SBC of the reduced and CommonSV rows, 1–3 rows each (36–78);
                       ``b8`` also the predictive-map checks
``mutation-p3``        M19–M36: U checks and G-small mutant/control pairs (≈ 40)
=====================  ============================================================================

The weekly contract tests and the ordering-invariance test of conjugate × ``CommonSV`` (row 27)
stay in ``rest`` (local: 13 min for the 18 weekly contract compositions).
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
#: Phase 3 (P3 DESIGN §9.11, WP35): shards by file; the Phase 2 shards of the same directory
#: ``--ignore`` every Phase 3 file, so ``-k structural`` never reaches them
P3_SHARDS: dict[str, list[str]] = {
    "geweke-p3a1": ["tests/geweke/test_geweke_p3_full_a1.py"],
    "geweke-p3a2": [
        "tests/geweke/test_geweke_p3_full_a2.py",
        "tests/geweke/test_geweke_p3_blocks.py",
    ],
    "geweke-p3b1": ["tests/geweke/test_geweke_p3_full_b1.py", "tests/geweke/test_geweke_p3.py"],
    "geweke-p3b2": ["tests/geweke/test_geweke_p3_full_b2.py"],
    **{f"sbc-p3{g}": [f"tests/sbc/test_sbc_p3_{g}.py"] for g in ("a1", "a2", "a3", "a4", "a5")},
    **{
        f"sbc-p3{g}": [f"tests/sbc/test_sbc_p3_{g}.py"]
        for g in ("b1", "b2", "b3", "b4", "b5", "b6", "b7")
    },
    "sbc-p3b8": ["tests/sbc/test_sbc_p3_b8.py", "tests/sbc/test_sbc_p3.py"],
    "mutation-p3": ["tests/mutation/test_mutants_p3.py"],
}
P3_FILES = tuple(f for args in P3_SHARDS.values() for f in args)


def _ignore(prefix: str) -> list[str]:
    return [f"--ignore={f}" for f in P3_FILES if f.startswith(prefix)]


SHARDS: dict[str, list[str]] = {
    "geweke-structural": ["tests/geweke", "-k", "structural", *_ignore("tests/geweke/")],
    "geweke-other": ["tests/geweke", "-k", "not structural", *_ignore("tests/geweke/")],
    "sbc-structural": ["tests/sbc", "-k", "structural", *_ignore("tests/sbc/")],
    "sbc-other": ["tests/sbc", "-k", "not structural", *_ignore("tests/sbc/")],
    "unit": ["tests/unit"],
    **P3_SHARDS,
    "rest": ["tests", *(f"--ignore={d}" for d in SPLIT), *_ignore("tests/mutation/")],
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
