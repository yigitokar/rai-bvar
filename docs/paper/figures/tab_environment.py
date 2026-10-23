"""TabG2 (Appendix G): environment and regeneration record.

Upper panel, the environment: the release record ``docs/paper/data/release/env.json`` when it
exists ([REL]); until then the environment record of the walk-through listings,
``docs/paper/code/out/env.json`` (written by ``docs/paper/code/run_examples.py``), marked in red as
not the release record (it was taken at the commit and with the tracked changes it names). The
BLAS/LAPACK backend is not in that record: it is read from ``numpy.show_config()`` of the Python
that runs this script (the main repository's environment) and labelled so. The release commit is a
[TC] mark until OUTLINE TODO 20 fixes it.

Lower panel, the regeneration command of every float that OUTLINE.md section 14 plans: the float
id and the generator stem are parsed from the two tables of section 14; the generator file is
looked up under ``docs/paper/figures/`` (``make_<stem>.py`` for figures, ``<stem>.py`` for tables)
and ``docs/paper/code/``; floats whose generator does not exist are listed as such. The data
options (``--acc``, the per-origin export) follow the variables of ``docs/paper/Makefile``.

Usage: tab_environment.py [--outdir DIR] [--env PATH]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--env", type=Path, default=tc.RELEASE / "env.json")
    args = ap.parse_args()
    tc.apply_outdir(args)
    released = args.env.exists()
    env = json.loads((args.env if released else LISTINGS_ENV).read_text())
    v = env["versions"]
    bench = tc.git_show_json(tc.git("log", "-1", "--format=%h", "--", "tests/bench/baseline.json"),
                             "tests/bench/baseline.json")["machine"]  # fmt: skip
    dirty = env.get("tracked_changes") or []
    up = [r"\begin{tabular}{@{}ll@{}}", r"\toprule", r"Item & Value \\", r"\midrule",
          f"Python & {tc.tex(v['python'])} \\\\",
          f"NumPy, SciPy, pandas, xarray & {v['numpy']}, {v['scipy']}, {v['pandas']}, {v['xarray']} \\\\",
          f"\\pkg{{jax}}, \\pkg{{jaxlib}} & {v['jax']}, {v['jaxlib']} \\\\",
          f"\\pkg{{scoringrules}} & {v['scoringrules']} \\\\",
          f"\\pkg{{rai-bvar}} & {v['rai_bvar']} \\\\",
          f"BLAS / LAPACK & {tc.tex(blas())}"
          + ("" if released else r" (environment of this generator)") + r" \\",
          f"Platform & {tc.tex(v['platform'])} \\\\",
          f"Machine of the BENCH-P2 record & {tc.tex(bench['cpu'])}, {bench['physical_cores']} cores \\\\",
          f"Commit of this record & {tc.code(env['commit'][:7])} (branch {tc.code(env['branch'])})"
          + (f", {len(dirty)} tracked files modified" if dirty else ", clean") + r" \\",
          f"Release commit & {tc.mark('TC')} OUTLINE TODO 20 \\\\",
          r"\bottomrule", r"\end{tabular}"]  # fmt: skip

    mv = make_vars()
    rows = planned_floats()
    sections = {p.name: p.read_text() for p in sorted((tc.PAPER / "sections").glob("*.tex"))
                if not p.name.startswith("staging")}
    cells, have = [], 0
    for fid, where, stems in rows:
        # the first planned generator that exists (OUTLINE may name a data step and a figure step)
        hit = [c for c in (command(st, mv) for st in stems) if c[1]]
        inline = [n for n, t in sections.items()
                  if re.search(r"\\label\{(tab:)?" + re.escape(fid) + r"\}", t)]
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
    foot = ("Upper panel: " + (r"release record \code{docs/paper/data/release/env.json}" if released else
            r"the environment record of the walk-through listings, \code{docs/paper/code/out/env.json} "
            f"(written {tc.tex(env['started_utc'])} by \\code{{code/run_examples.py}})") + ". "
            r"Lower panel: every float planned in OUTLINE \S14, with its generator (paths from \code{docs/paper}); run each with the "
            r"main repository's Python (\code{make figures} in \code{docs/paper} runs the figure "
            r"generators; \code{$ACC} is the acceptance report, \code{make per-origin} its per-origin "
            f"export). {have} of the {len(rows)} planned floats have a generator.")  # fmt: skip
    body = "\\footnotesize\n" + tc.preamble(marks=True) + tc.fit("\n".join(up)) + "\n\n\\medskip\n" + tc.fit("\n".join(lo)) + "\n" + tc.note(foot)
    if not released:
        body += "\n" + tc.note(
            r"Not the release record: taken at " + tc.code(env["commit"][:7])
            + (f" with {len(dirty)} modified tracked files" if dirty else "")
            + r"; regenerate from \code{docs/paper/data/release/env.json} on the release commit "
            "(OUTLINE TODO 7, TODO 20).", red=True)  # fmt: skip
    tc.write(NAME, body, generator="tab_environment.py",
             status=("[REL] release env.json" if released else
                     "INTERIM: docs/paper/code/out/env.json (listings record), not the release record"),
             sources=[str((args.env if released else LISTINGS_ENV).relative_to(tc.PAPER.parent.parent))
                      if (args.env if released else LISTINGS_ENV).is_relative_to(tc.PAPER.parent.parent) else str(args.env),
                      "numpy.show_config() of the generating Python (BLAS/LAPACK)",
                      "main repo tests/bench/baseline.json machine (CPU model)",
                      "docs/paper/OUTLINE.md section 14 (float ids and generator stems); docs/paper/Makefile"])  # fmt: skip


if __name__ == "__main__":
    main()
