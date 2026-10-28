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

**Phase 3** (docs/phase3/DESIGN.md §9.11; WP35): shards by file (:data:`P3_SHARDS`), and the Phase
2 shards of ``tests/geweke`` and ``tests/sbc`` ``--ignore`` every Phase 3 file (their ``-k
structural`` would otherwise collect rows such as ``structural_constant_hs``); ``rest`` ignores the
Phase 3 mutation file, which runs as its own shard. §9.11 names ``geweke-p3a``/``geweke-p3b`` (the
structural rows; the reduced and CommonSV rows) and ``sbc-p3a``/``sbc-p3b`` (the same split, by
file), "each shard ≤ 200 min on a 2-vCPU EPYC 7763, split further if not". Each is split further.
The first split (19 Phase 3 shards) rested on estimates measured on an Apple M5 Max under a load
average of 140–170, about half as fast as the 4-vCPU EPYC 7763 runner (Phase 2's G-full of row 2:
2813 s for three Gandy–Scott stages, 9 stage-1 units, so 313 s per stage-1 unit against 156 s on
the runner of run 40140397848). Run 40414682400 (4-vCPU runners, ``t_max = 200``) then took 8–105
min per Phase 3 shard. After Q44 (E179: ``t_max = 400`` for SBC rows 304, 306, 308, 311, 315, 325,
326 and 327) the SBC shards are re-balanced on those measured times, with the eight rows at twice
theirs; the three heaviest of them (306, 308, 325) and 315 (whose stage-1 γ value 0.072 could
trigger a Gandy–Scott stage 2, four times the replications: about 150 min) run alone. Expected
minutes on a 4-vCPU EPYC 7763 (``sbc-p3a1`` and ``sbc-p3b2`` include the stage-2 reruns of 317 and
313 seen in run 40414682400):

=====================  ============================================================================
shard                  contents (expected, min)
=====================  ============================================================================
``geweke-p3a1``        G-full of the structural rows 301–308 (55)
``geweke-p3a2``        G-full of 309, 310, 317–319, 326, 330; G-small of 401–408 (22)
``geweke-p3b1``        G-full of the reduced rows 311–316, 327 (24)
``geweke-p3b2``        G-full of the CommonSV rows 320–325, 328, 329 (49)
``sbc-p3a1``           317, 303 (105)
``sbc-p3a2``           308 (≈ 50)
``sbc-p3a3``           306 (≈ 52)
``sbc-p3a4``           304, 326, 302 (≈ 70)
``sbc-p3a5``           301, 305, 307, 309, 310, 318, 319, 330 (≈ 55)
``sbc-p3b1``           315 (≈ 30; ≈ 150 with a stage-2 rerun)
``sbc-p3b2``           313, 316 (≈ 71)
``sbc-p3b3``           311, 327, 329 (≈ 39)
``sbc-p3b4``           325 (≈ 57)
``sbc-p3b5``           324, 328, 314 (≈ 58)
``sbc-p3b6``           320, 321, 322, 323, 312; the predictive-map checks (≈ 48)
``mutation-p3``        M19–M36: U checks and G-small mutant/control pairs (8)
=====================  ============================================================================

The 240-min step leaves headroom on every shard at these times. A Gandy–Scott stage 2 of 306, 308
or 325 at ``t_max = 400`` (about five times its stage 1, up to about 285 min for 325) would not
fit; their stage-1 γ values in run 40414682400 (0.47, 0.48, 1.0) were far from the band that
triggers one (0.0033–0.153). The weekly contract tests and the ordering-invariance test of
conjugate × ``CommonSV`` (row 27) stay in ``rest`` (local: 13 min for the 18 weekly contract
compositions).
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
    **{f"sbc-p3{g}": [f"tests/sbc/test_sbc_p3_{g}.py"] for g in ("b1", "b2", "b3", "b4", "b5")},
    "sbc-p3b6": ["tests/sbc/test_sbc_p3_b6.py", "tests/sbc/test_sbc_p3.py"],
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
