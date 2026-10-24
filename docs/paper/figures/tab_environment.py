"""TabG2 (Appendix G): environment and regeneration record.

Upper panel, the environments, in two columns:

* the release validation record at 7f880e1, ``meta.json`` of ``tc.RELEASE_DIR``
  (``benchmarks/reports/validation/7f880e1_apple-m5-max/``) [REL]: commit, clean tree, machine,
  operating system and the versions of Python, NumPy, SciPy and jax it records;
* the environment record of the walk-through listings, ``docs/paper/code/out/env.json`` (written by
  ``docs/paper/code/run_examples.py``), taken at the commit and with the tracked changes it names,
  for the packages the release record does not list.

The rai-bvar version is the ``version`` of ``pyproject.toml`` at each commit (``git show``). The
BLAS/LAPACK backend is in neither record: it is read from ``numpy.show_config()`` of the Python that
runs this script (the main repository's environment) and labelled so. The release row gives release
0.2.0 (main 4ac7969, ``tc.RELEASE_TAG_COMMIT``) and checks with ``git diff`` that it differs from
7f880e1 only in the version string (``pyproject.toml``, ``uv.lock``) and in the committed reports
under ``benchmarks/reports/``. The BENCH-P2 row names both benchmark machines: the reference
baseline (``tests/bench/baseline.json``) and the x86-64 report of the CI bench job
(``benchmarks/reports/bench_p2_x86_github_779957f.json``).

Lower panel, the regeneration command of every float that OUTLINE.md section 14 plans: the float
id and the generator stem are parsed from the two tables of section 14; the generator file is
looked up under ``docs/paper/figures/`` (``make_<stem>.py`` for figures, ``<stem>.py`` for tables)
and ``docs/paper/code/``; floats whose generator does not exist are listed as such. The data
options (``--acc``, the per-origin export) follow the variables of ``docs/paper/Makefile``.

Usage: tab_environment.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tab_common as tc

NAME = "tab_environment"
LISTINGS_ENV = tc.PAPER / "code" / "out" / "env.json"
OUTLINE = tc.PAPER / "OUTLINE.md"
MAKEFILE = tc.PAPER / "Makefile"


def blas() -> str:
    import numpy as np

    try:
        cfg = np.show_config(mode="dicts")["Build Dependencies"]
        b, lp = cfg.get("blas", {}), cfg.get("lapack", {})
        return f"{b.get('name', 'unknown')} (BLAS), {lp.get('name', 'unknown')} (LAPACK)"
    except Exception:  # noqa: BLE001
        return "unknown"


def make_vars() -> dict[str, list[str]]:
    text = MAKEFILE.read_text().replace("\\\n", " ")
    out = {}
    for var in ("NOW", "ACC_REPORT", "ACC_EXPORT"):
        out[var] = tc.grab(text, rf"^{var} := (.*)$", flags=re.M).split()
    return out


def planned_floats() -> list[tuple[str, str, list[str]]]:
    """(float id, section or appendix, generator stems) from OUTLINE.md section 14."""
    text = OUTLINE.read_text()
    rows = []
    for heading in (r"^### 14\.1 ", r"^### 14\.2 "):
        for r in tc.md_table(tc.section(text, heading)):
            where = r.get("§") or r.get("App.") or ""
            stems = re.findall(r"`([a-z_]+)\.py`", r["Script"])
            rows.append((r["ID"], where, stems))
    return rows


def command(stem: str, mv: dict) -> tuple[str, bool]:
    fig = tc.HERE / f"make_{stem}.py"
    tab = tc.HERE / f"{stem}.py"
    code = tc.PAPER / "code" / f"{stem}.py"
    if fig.exists():
        extra = (r" \code{--acc $ACC}" if stem in mv["ACC_REPORT"] else
                 r" (after \code{make per-origin})" if stem in mv["ACC_EXPORT"] else "")
        return tc.code(f"figures/make_{stem}.py") + extra, True
    if tab.exists():
        return tc.code(f"figures/{stem}.py"), True
    if code.exists():
        return tc.code(f"code/{stem}.py"), True
    return "", False


X86_BENCH = "benchmarks/reports/bench_p2_x86_github_779957f.json"


def pyproject_version(commit: str) -> str:
    return tc.grab(tc.git_show(commit, "pyproject.toml"), r'^version = "([^"]+)"$', flags=re.M)


def release_equivalence() -> list[str]:
    """Files that differ between 7f880e1 and the release 4ac7969, outside benchmarks/reports/:
    must be the version string only (pyproject.toml; in uv.lock the project's version line, beside
    uv's lock-format rewrite, which moves no dependency version)."""
    files = tc.git("diff", "--name-only", tc.RELEASE_COMMIT, tc.RELEASE_TAG_COMMIT).split()
    other = [f for f in files if not f.startswith("benchmarks/reports/")]
    assert sorted(other) == ["pyproject.toml", "uv.lock"], other
    diff = tc.git("diff", "-U0", tc.RELEASE_COMMIT, tc.RELEASE_TAG_COMMIT, "--", "pyproject.toml")
    changed = [ln for ln in diff.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    assert changed == ['-version = "0.0.1"', '+version = "0.2.0"'], changed
    lock = tc.git("diff", "-U0", tc.RELEASE_COMMIT, tc.RELEASE_TAG_COMMIT, "--", "uv.lock")
    versions = [ln for ln in lock.splitlines() if re.match(r"^[+-]version = ", ln)]
    assert versions == ['-version = "0.0.1"', '+version = "0.2.0"'], versions  # no dependency moved
    return other


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    meta = tc.release_meta()
    env = json.loads(LISTINGS_ENV.read_text())
    v, rv, mach = env["versions"], meta["versions"], meta["machine"]
    for k in ("python", "numpy", "scipy", "jax"):  # the two records share these versions
        assert rv[k] == v[k], (k, rv[k], v[k])
    bench = tc.git_show_json(tc.git("log", "-1", "--format=%h", "--", "tests/bench/baseline.json"),
                             "tests/bench/baseline.json")["machine"]  # fmt: skip
    x86 = tc.git_show_json(tc.RELEASE_TAG_COMMIT, X86_BENCH)["machine"]
    release_equivalence()
    dirty = env.get("tracked_changes") or []
    rel_v, lst_v = pyproject_version(tc.RELEASE_COMMIT), pyproject_version(env["commit"])
    os_name = re.sub(r"-arm-64bit$", "", meta["machine"]["os"])
    up = [r"\begin{tabular}{@{}lll@{}}", r"\toprule",
          r"Item & Release validation record & Walk-through listings \\", r"\midrule",
          f"Commit & {tc.code(meta['commit'][:7])}, clean & {tc.code(env['commit'][:7])} (branch "
          f"{tc.code(env['branch'])})" + (f", {len(dirty)} tracked files modified" if dirty else ", clean") + r" \\",
          f"Started (UTC) & {tc.tex(meta['started_utc'][:16].replace('T', ' '))} & "
          f"{tc.tex(env['started_utc'][:16].replace('T', ' '))} \\\\",
          f"Machine, system & {tc.tex(mach['cpu'])}, {mach['ncpu']} cores, {tc.tex(os_name)} & "
          f"{tc.tex(v['platform'])} \\\\",
          f"Python; NumPy, SciPy & {tc.tex(rv['python'])}; {rv['numpy']}, {rv['scipy']} & "
          f"{tc.tex(v['python'])}; {v['numpy']}, {v['scipy']} \\\\",
          f"\\pkg{{jax}}, \\pkg{{jaxlib}} & {rv['jax']}, --- & {v['jax']}, {v['jaxlib']} \\\\",
          f"pandas, xarray, \\pkg{{scoringrules}} & --- & {v['pandas']}, {v['xarray']}, {v['scoringrules']} \\\\",
          f"\\pkg{{rai-bvar}} version (\\code{{pyproject.toml}}) & {rel_v} & {lst_v} \\\\",
          f"BLAS / LAPACK & \\multicolumn{{2}}{{l}}{{{tc.tex(blas())} (environment of this generator)}} \\\\",
          f"BENCH-P2 machines & \\multicolumn{{2}}{{l}}{{{tc.tex(bench['cpu'])}, {bench['physical_cores']} cores "
          f"(reference baseline); {tc.tex(x86['cpu'])}, {x86['logical_cpus']} vCPU (CI runner)}} \\\\",
          f"Release & \\multicolumn{{2}}{{l}}{{{pyproject_version(tc.RELEASE_TAG_COMMIT)}, "
          f"{tc.code(tc.RELEASE_TAG_COMMIT)} on \\code{{main}}: the code of {tc.code(tc.RELEASE_COMMIT)}, with the "
          r"version string set} \\",
          r"\bottomrule", r"\end{tabular}"]  # fmt: skip

    mv = make_vars()
    rows = planned_floats()
    sections = {p.name: p.read_text() for p in sorted((tc.PAPER / "sections").glob("*.tex"))
                if not p.name.startswith("staging")}
    cells, have = [], 0
    for fid, where, stems in rows:
        # the first planned generator that exists (OUTLINE may name a data step and a figure step)
        hit = [c for c in (command(st, mv) for st in stems) if c[1]]
        # a typed float carries its OUTLINE id as a label, or a label named after its planned stem
        # (Tab2: \label{tab:resolution} for tab_resolution.py)
        labels = [re.escape(fid)] + [re.escape(st.removeprefix("tab_")) for st in stems if st.startswith("tab_")]
        inline = [n for n, t in sections.items()
                  if re.search(r"\\label\{(tab:)?(" + "|".join(labels) + r")\}", t)]
        if hit:
            cmd = hit[0][0]
            have += 1
        elif inline:
            cmd = "typed in " + tc.code("sections/" + inline[0])
        elif stems:
            cmd = "not written: " + tc.code(stems[0] + ".py")
        else:
            cmd = "static"
        cells.append(f"{tc.tex(fid)} & {tc.tex(where)} & {cmd}")
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:] + [" & & "] * (2 * half - len(cells))
    lo = [r"\begin{tabular}{@{}lll@{\hspace{1.5em}}lll@{}}", r"\toprule",
          r"Float & \S & Generator & Float & \S & Generator \\", r"\midrule"]
    lo += [f"{x} & {y} \\\\" for x, y in zip(left, right, strict=True)]
    lo += [r"\bottomrule", r"\end{tabular}"]
    foot = (r"Upper panel: the release validation record, \code{meta.json} in "
            r"\code{benchmarks/reports/validation/7f880e1_apple-m5-max/}, and the environment record of the "
            r"walk-through listings, \code{docs/paper/code/out/env.json} (written by \code{code/run_examples.py}); "
            r"--- not recorded. "
            r"Lower panel: every float planned in OUTLINE \S14, with its generator (paths from \code{docs/paper}); run each with the "
            r"main repository's Python (\code{make figures} in \code{docs/paper} runs the figure "
            r"generators; \code{$ACC} is the acceptance report, \code{make per-origin} its per-origin "
            f"export). {have} of the {len(rows)} planned floats have a generator.")  # fmt: skip
    body = "\\footnotesize\n" + tc.fit("\n".join(up)) + "\n\n\\medskip\n" + tc.fit("\n".join(lo)) + "\n" + tc.note(foot)
    tc.write(NAME, body, generator="tab_environment.py",
             status=f"[REL] release meta.json at {tc.RELEASE_COMMIT}; listings env.json at {env['commit'][:7]}",
             sources=[str((tc.RELEASE_DIR / "meta.json").relative_to(tc.REPO_ROOT)),
                      str(LISTINGS_ENV.relative_to(tc.REPO_ROOT)),
                      "numpy.show_config() of the generating Python (BLAS/LAPACK)",
                      f"main repo tests/bench/baseline.json machine (CPU model); {X86_BENCH} at {tc.RELEASE_TAG_COMMIT}",
                      f"pyproject.toml at {tc.RELEASE_COMMIT}, {env['commit'][:7]} and {tc.RELEASE_TAG_COMMIT}; "
                      f"git diff {tc.RELEASE_COMMIT} {tc.RELEASE_TAG_COMMIT}",
                      "docs/paper/OUTLINE.md section 14 (float ids and generator stems); docs/paper/Makefile"])  # fmt: skip


if __name__ == "__main__":
    main()
