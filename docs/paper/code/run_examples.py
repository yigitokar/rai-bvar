"""Run the walk-through listings (ex_a.py, ex_b.py) against a rai-bvar checkout and capture output.

Usage (from anywhere; the main repository provides the environment and the package):

    python docs/paper/code/run_examples.py [--repo /Users/yigitokar/rai-bvar]

Each script runs as ``PYTHONDONTWRITEBYTECODE=1 uv run --frozen python <script>`` with the
repository as working directory. stdout goes to ``out/<name>.txt`` (pulled into the paper by
``\\lstinputlisting``); stderr goes to ``out/<name>.stderr.txt``, which the paper never prints
(it holds the library's warnings, one of which names other packages; TODO 16). ``out/env.json``
records the repository commit, whether its tracked files were clean, the platform and the
package versions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ("ex_a.py", "ex_b.py")

VERSIONS = r"""
import json, platform, sys
import numpy, scipy, pandas, xarray, jax, jaxlib, scoringrules, rai_bvar
print(json.dumps({
    "python": sys.version.split()[0], "platform": platform.platform(),
    "machine": platform.machine(), "processor": platform.processor(),
    "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
    "xarray": xarray.__version__, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
    "scoringrules": getattr(scoringrules, "__version__", "unknown"),
    "rai_bvar": rai_bvar.__version__,
}))
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.rstrip("\n")  # keep the leading status column of `git status --porcelain`


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="/Users/yigitokar/rai-bvar", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = ["uv", "run", "--frozen", "python"]

    record = {
        "repo": str(repo),
        "commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        # tracked files that differ from the commit (an empty list means a clean tree)
        "tracked_changes": git(repo, "status", "--porcelain", "--untracked-files=no").splitlines(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": json.loads(
            subprocess.run(
                [*cmd, "-c", VERSIONS], cwd=repo, env=env, check=True, capture_output=True,
                text=True,
            ).stdout
        ),
        "scripts": {},
    }
    status = 0
    for name in SCRIPTS:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [*cmd, str(HERE / name)], cwd=repo, env=env, capture_output=True, text=True
        )
        stem = Path(name).stem
        (out / f"{stem}.txt").write_text(proc.stdout)
        (out / f"{stem}.stderr.txt").write_text(proc.stderr)
        record["scripts"][name] = {
            "returncode": proc.returncode,
            "wall_seconds": round(time.perf_counter() - t0, 1),  # machine and load dependent
        }
        status = status or proc.returncode
        print(f"{name}: exit {proc.returncode}, {record['scripts'][name]['wall_seconds']} s")
    (out / "env.json").write_text(json.dumps(record, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
