"""TabD1 (Appendix D): G-small Geweke results for the per-PR compositions.

Rows: the per-PR G-small set of ``tests/geweke/test_geweke_compositions.py`` at the main repository's
HEAD (``PR_ROWS``: rows 1--5, 7--10, 17, 18 and the dummy rows 201--205; 16 compositions with row
18, ``reduced_sv_rw``). Columns: number of test functions G, the t_{C-1} critical value, the
largest |z| and its test function, the Gandy--Scott stage(s) and Bonferroni q, the smallest
relative numerical efficiency, and the time.

Data, in order of preference:

1. a ``G-small`` tier in the release record ``record/geweke.jsonl`` of ``tc.RELEASE_DIR``
   (``benchmarks/reports/validation/7f880e1_apple-m5-max/``) ([REL]). The release run at 7f880e1
   ran the slow tiers only (``-m "(slow or network) and not bench"``), so that record has the block
   configurations, G-full, the inexact rows and row 14's ``h0`` variant, but no G-small row of the
   per-PR set; the script checks this and then uses
2. the G-small run of dossier 04 section 6.6 (commit 62d15f6, Apple M5 Max, reproduced twice;
   deterministic for fixed code and seed on that machine) for the 15 rows it covers, and for row 18,
   which joined the PR tier in 7f410dc, the G-small statistic recorded in DESIGN section 9.5 (row 18
   note) and in the commit message of 7f410dc ([R], Apple M5 Max under load; no per-stage record).
   The note under the table says so, and that the 16 per-PR G-small tests passed on the release
   commit in the CI run ``tc.CI_FAST_RUN`` (pass or fail only; the CI sets no record directory).

In both cases G and the stage-1 critical value ``t_{C-1}(1 - alpha/(2G))`` are recomputed from the
committed code (``joint_spec(name, "small").names``; ``G_SMALL`` of the test module) and must equal
the recorded ones. The computation imports the harness (no sampling; a few seconds).

Usage: tab_geweke.py [--outdir DIR] [--record PATH]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tab_common as tc  # noqa: E402

NAME = "tab_geweke"
TEST = "tests/geweke/test_geweke_compositions.py"
DOSSIER = tc.PAPER / "dossiers" / "04_validation.md"


def committed_sets() -> dict:
    src = tc.git_show("HEAD", TEST)
    dummy = {int(x) for x in re.findall(r"\d+", tc.grab(src, r"^DUMMY_ROWS = frozenset\(\{([^}]*)\}\)", flags=re.M))}
    pr = {int(x) for x in re.findall(r"\d+", tc.grab(src, r"^PR_ROWS = frozenset\(\{([^}]*)\}\) \| DUMMY_ROWS",
                                                         flags=re.M))}  # fmt: skip
    return {"pr_rows": pr | dummy, "sizes": tc.literal(src, "G_SMALL"),
            "seed": tc.literal(src, "SEED_SMALL")}  # fmt: skip


def recompute(rows: set[int], sizes: dict) -> dict:
    import sys

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tc.MAIN_REPO / "src"))
    from scipy import stats

    from rai_bvar.validation.joint import COMPOSITIONS, composition_id, joint_spec

    out = {}
    for name in COMPOSITIONS:
        row = composition_id(name)
        if row in rows:
            G = len(joint_spec(name, "small").names)
            out[row] = {"name": name, "G": G,
                        "t_crit": float(stats.t.isf(sizes["alpha"] / (2 * G), sizes["C"] - 1))}  # fmt: skip
    assert set(out) == rows, (set(out), rows)
    return out


def cell(s: str) -> str:
    """A dossier cell: ``x`` -> \\code{x}, arrows, seconds."""
    parts = re.split(r"`([^`]+)`", s)
    out = "".join(tc.code(p) if i % 2 else tc.tex(p) for i, p in enumerate(parts))
    out = out.replace("→", r"$\to$")
    return re.sub(r"(\d) s$", r"\1\\,s", out.strip())


def from_record(path: Path, comp: dict) -> tuple[list[str], str]:
    recs = [r for r in tc.read_jsonl(path) if r.get("tier") == "G-small"]
    by = {r["name"]: r for r in recs}
    lines = []
    for row in sorted(comp):
        c = comp[row]
        r = by.get(c["name"])
        if r is None:  # no record for this composition: a [REL] mark, not a crash
            lines.append(f"{row} & {tc.code(c['name'])} & {c['G']} & {c['t_crit']:.2f} & "
                         + " & ".join([tc.mark("REL")] * 4) + r" \\")
            continue
        st = r["stages"]
        assert st[0]["G"] == c["G"] and abs(st[0]["t_crit"] - c["t_crit"]) < 1e-9, (row, st[0], c)
        tcrit = r" $\to$ ".join(f"{s['t_crit']:.2f}" for s in st)
        z = r" $\to$ ".join(f"{s['max_abs_z']:.2f} ({tc.code(s['argmax'])})" for s in st)
        q = r" $\to$ ".join(f"{i + 1}: {qq:.3g}" for i, qq in enumerate(r["q"]))
        rne = min(s["min_rne"] for s in st)
        fail = "" if r["passed"] else " (fail)"
        lines.append(f"{row} & {tc.code(c['name'])} & {c['G']} & {tcrit} & {z}{fail} & {q} & {rne:.3f} & "
                     f"{r['seconds']:.1f}\\,s \\\\")  # fmt: skip
    return lines, f"release record {path}"


def interim(comp: dict) -> tuple[list[str], list[str]]:
    sec = tc.section(DOSSIER.read_text(), r"^### 6\.6 ")
    table = {int(r["Row"]): r for r in tc.md_table(sec)}
    design = tc.git_show("HEAD", "docs/phase2/DESIGN.md")
    row18 = tc.section(design, r"^### 9\.5 ")
    z18 = tc.grab(row18, r"Evidence\s+\(2026-10-23,\s+Apple\s+M5\s+Max\):\s+G-small\s+max\s+\|z\|\s+([\d.]+),")
    stage18 = "1" if re.search(r"G-small\s+max\s+\|z\|\s+[\d.]+,\s+G-full\s+[\d.]+,\s+SBC\s+min\s+p\s+[\d.]+"
                               r"\s+\(\d+\s+quantities\),\s+all\s+at\s+stage\s+1", row18) else None  # fmt: skip
    msg = tc.git("log", "-1", "--format=%B", "7f410dc")
    z18m, t18m = tc.grab(msg, r"G-small max\|z\| ([\d.]+) \(t_crit ([\d.]+)\)", 0).split(" (t_crit ")
    assert z18m.endswith(z18) and abs(float(t18m.rstrip(")")) - comp[18]["t_crit"]) < 0.005
    assert stage18 == "1"
    lines, notes = [], []
    for row in sorted(comp):
        c = comp[row]
        if row == 18:
            lines.append(f"18$^\\ast$ & {tc.code(c['name'])} & {c['G']} & {c['t_crit']:.2f} & {z18} (---) & "
                         f"{stage18}: --- & --- & --- \\\\")  # fmt: skip
            continue
        d = table[row]
        assert d["Composition"] == c["name"], (row, d["Composition"], c["name"])
        assert int(d["G"]) == c["G"], (row, d["G"], c["G"])
        assert d["t_crit"].split("→")[0].strip() == f"{c['t_crit']:.2f}", (row, d["t_crit"], c["t_crit"])
        lines.append(f"{row} & {tc.code(c['name'])} & {c['G']} & {cell(d['t_crit'])} & "
                     f"{cell(d['max |z| (argmax)'])} & {cell(d['stage(s), q'])} & {cell(d['min RNE'])} & "
                     f"{cell(d['time'])} \\\\")  # fmt: skip
    missing = set(table) - set(comp)
    assert not missing, missing
    notes.append(f"dossier 04 \\S6.6 (G-small on 62d15f6) for {len(comp) - 1} rows")
    return lines, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--record", type=Path, default=tc.RELEASE / "geweke.jsonl")
    ap.add_argument("--ci-log", type=Path, default=tc.CI_FAST_LOG,
                    help="excerpt of the log of CI run tc.CI_FAST_RUN (gh run view RUN --log): the "
                    "script checks that the per-PR G-small tests ran and passed in each Python job")
    args = ap.parse_args()
    tc.apply_outdir(args)
    sets = committed_sets()
    sz = sets["sizes"]
    comp = recompute(sets["pr_rows"], sz)
    released = args.record.exists() and any('"G-small"' in ln for ln in args.record.read_text().splitlines())
    head = [r"\begin{tabular}{@{}rlrllllr@{}}", r"\toprule",
            r"Row & Composition & $G$ & $t_{C-1}$ & max $|z|$ (argmax) & Stage: $q$ & min RNE & Time \\",
            r"\midrule"]  # fmt: skip
    if released:
        lines, src = from_record(args.record, comp)
        head[0:0] = [tc.preamble(marks=True).rstrip()]
        status, sources = "[REL] release record", [src]
        notes = ""
    else:
        tiers = {r.get("tier") for r in tc.read_jsonl(args.record)} if args.record.exists() else set()
        assert "G-small" not in tiers, tiers
        lines, _ = interim(comp)
        status = ("statistics: dossier 04 section 6.6 (62d15f6) + row 18 [R] (DESIGN 9.5, 7f410dc); the release "
                  f"record at {tc.RELEASE_COMMIT} has no G-small row; pass on the release commit: CI run {tc.CI_FAST_RUN}")
        sources = [str(DOSSIER.relative_to(tc.PAPER.parent.parent)) + " section 6.6 (G-small on 62d15f6, Apple M5 Max)",
                   "main repo docs/phase2/DESIGN.md section 9.5 row 18 note; commit message of 7f410dc",
                   f"main repo {TEST} at HEAD (PR_ROWS, DUMMY_ROWS, G_SMALL) and joint_spec(name, 'small').names",
                   f"{args.record.relative_to(tc.REPO_ROOT) if args.record.is_relative_to(tc.REPO_ROOT) else args.record} "
                   f"(tiers {sorted(t for t in tiers if t)}; no G-small)",
                   f"CI run {tc.CI_FAST_RUN} (gh run view {tc.CI_FAST_RUN}): fast suite at {tc.RELEASE_COMMIT}, "
                   "ubuntu-latest, Python 3.12, 3.13, 3.14"]  # fmt: skip
        # the Geweke module's fast tests: the 16 G-small rows, test_pr_rows_are_the_design_rows and
        # test_g_small_rw_h0_prior[reduced_sv_rw] (its row-14 variant is slow); every one passed
        src = tc.git_show("HEAD", TEST)
        fns = {f: marks for f, marks in tc.functions(src).items() if f.startswith("test_")}
        # test_g_small_rw_h0_prior carries "slow" only on its structural_sv_rw parameter
        assert sorted(f for f, m in fns.items() if "slow" not in m) == ["test_g_small",
                                                                          "test_pr_rows_are_the_design_rows"], fns
        assert 'pytest.param("structural_sv_rw", marks=pytest.mark.slow)' in src
        for job, mods in tc.ci_fast_modules(args.ci_log).items():
            prog = mods[TEST]
            assert prog == "." * (len(comp) + 2), (job, prog)
        print(f"CI run {tc.CI_FAST_RUN}: {len(comp) + 2} Geweke fast tests passed in each of 3 jobs")
        notes = tc.note(
            r"Statistics: rows other than 18 are the G-small run of dossier 04 \S6.6 on commit 62d15f6 "
            r"(Apple M5 Max; reproduced twice, deterministic for fixed code and seed on that machine); "
            r"$^\ast$row 18 (\code{reduced_sv_rw}, in the PR tier since 7f410dc) is recorded only in "
            r"DESIGN \S9.5 and the commit message of 7f410dc (Apple M5 Max under load; stage 1, no "
            r"per-stage record). The release validation record (" + tc.RELEASE_COMMIT + r", "
            r"\cref{app:validation}) covers the slow tiers and has no G-small row; on the release commit "
            f"the {len(comp)} per-PR G-small tests passed in the fast suite of CI run {tc.CI_FAST_RUN} "
            r"(ubuntu-latest, Python 3.12--3.14), which records pass or fail only.")  # fmt: skip
    seed = sets["seed"]
    foot = (f"{len(comp)} per-PR compositions of \\code{{{TEST}}} (\\code{{PR_ROWS}}). Exact mode, "
            f"$C = {sz['C']}$, $M_c = {sz['M_c']}$, $M_1 = {tc.thousands(sz['M1'])}$, "
            f"$\\alpha = 10^{{{int(round(__import__('math').log10(sz['alpha'])))}}}$, seed {seed}, "
            r"Gandy--Scott $k = 3$, $\Delta = 4$; a stage after the first is a fresh run at four times "
            r"$C$ and $M_1$. $G$ and the stage-1 critical value $t_{C-1}(1 - \alpha/2G)$ are "
            "recomputed from the committed code and equal the recorded ones. Time: pytest call "
            "duration.")  # fmt: skip
    body = tc.fit("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}"])) + "\n" + tc.note(foot)
    if notes:
        body += "\n" + notes
    tc.write(NAME, body, generator="tab_geweke.py", status=status, sources=sources)


if __name__ == "__main__":
    main()
