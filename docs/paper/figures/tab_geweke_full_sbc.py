"""TabD3 (Appendix D): G-full, the recorded exact=False rows, composition-level Tier B and full SBC.

Rows: every composition of ``src/rai_bvar/validation/joint.py`` at the main repository's HEAD (23
with row 18, ``reduced_sv_rw``). For each: G and the stage-1 critical value of G-full (recomputed
from the committed code: ``joint_spec(name, "full").names`` and ``G_FULL`` of
``tests/geweke/test_geweke_compositions.py``), the G-full result, the ``exact=False`` result where
the row has an inexact variant (``INEXACT_ROWS``), composition-level Tier B, and full SBC (minimum
p and its quantity, stage, thinning cap ``t_max`` with ``SBC_OVERRIDES``, largest thinning used,
replications failing the per-replication ESS rule, minimum pooled ESS).

Data:

* the release records ``docs/paper/data/release/geweke.jsonl`` (tiers ``G-full``,
  ``G-full-inexact``) and ``sbc.jsonl`` ([REL]; OUTLINE TODO 7) when they exist;
* until then the only results stated are those the appendix states (``sections/appendix.tex``,
  app:sbc: "no G-full or SBC result other than row 205"), plus row 18, which the orchestrator's
  coverage decision adds: row 205 from ``docs/paper/NOTES.md`` item 13 of the main repository
  (G-full 2.76, SBC min p 0.018; [R]) and row 18 from DESIGN section 9.5 (row 18 note), the module
  docstring of ``tests/sbc/test_sbc_compositions.py`` and the commit message of 7f410dc ([R], Apple
  M5 Max under load). Every other result cell is a red [REL] mark. The uncommitted local records in
  ``docs/paper/figures/data/validation_record/`` are not used (no commit or machine field; TODO 7).
* composition-level Tier B (``tests/sbc/test_block_tier.py``) has no record hook (TODO 7): [REL].

Usage: tab_geweke_full_sbc.py [--outdir DIR] [--records DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tab_common as tc  # noqa: E402

NAME = "tab_geweke_full_sbc"
GTEST = "tests/geweke/test_geweke_compositions.py"
STEST = "tests/sbc/test_sbc_compositions.py"
REL = tc.mark("REL")


def committed() -> dict:
    g = tc.git_show("HEAD", GTEST)
    s = tc.git_show("HEAD", STEST)
    inexact = {int(x) for x in re.findall(r"\d+", tc.grab(g, r"^INEXACT_ROWS = frozenset\(\{([^}]*)\}\)", flags=re.M))}
    return {"sizes": tc.literal(g, "G_FULL"), "inexact": inexact, "t_max": tc.literal(s, "T_MAX"),
            "pilot": tc.literal(s, "PILOT"), "overrides": tc.literal(s, "SBC_OVERRIDES"),
            "sbc_doc": s.split('"""')[1]}  # fmt: skip


def recompute(sizes: dict) -> list[dict]:
    import sys

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tc.MAIN_REPO / "src"))
    from scipy import stats

    from rai_bvar.validation.joint import COMPOSITIONS, composition_id, joint_spec

    out = []
    for name in COMPOSITIONS:
        G = len(joint_spec(name, "full").names)
        out.append({"row": composition_id(name), "name": name, "G": G,
                    "t_crit": float(stats.t.isf(sizes["alpha"] / (2 * G), sizes["C"] - 1))})  # fmt: skip
    return out


def recorded_interim(c: dict) -> dict:
    """Row 205 (NOTES item 13) and row 18 (DESIGN 9.5, SBC docstring, 7f410dc), [R]."""
    notes = tc.git_show("HEAD", "docs/paper/NOTES.md")
    m = re.search(r"Geweke row 205\s+\(G-small max \|z\| [\d.]+, G-full ([\d.]+)\), SBC row 205 \(min p ([\d.]+)\)", notes)
    assert m, "NOTES item 13 (row 205)"
    design = tc.section(tc.git_show("HEAD", "docs/phase2/DESIGN.md"), r"^### 9\.5 ")
    e = re.search(r"Evidence\s+\(2026-10-23,\s+Apple\s+M5\s+Max\):\s+G-small\s+max\s+\|z\|\s+[\d.]+,\s+G-full\s+([\d.]+),"
                  r"\s+SBC\s+min\s+p\s+([\d.]+)\s+\((\d+)\s+quantities\),\s+all\s+at\s+stage\s+1;\s+the\s+inexact\s+"
                  r"variant's\s+G-full\s+max\s+\|z\|\s+(\d+\.\d+)", design)  # fmt: skip
    assert e, "DESIGN 9.5 row 18 evidence"
    msg = tc.git("log", "-1", "--format=%B", "7f410dc")
    gm = re.search(r"G-full\s+([\d.]+)\s+\(([\d.]+)\);\s+inexact\s+([\d.]+)", msg)
    assert gm and gm.group(1) == e.group(1) and gm.group(3) == e.group(4)
    doc = re.sub(r"\s+", " ", c["sbc_doc"])
    p18 = re.search(r"the row then passes at stage 1 \(ESS rule 100%, min p ([\d.]+), pooled ESS ≥ (\d+)\)", doc)
    assert p18 and p18.group(1) == e.group(2)
    return {205: {"gfull": m.group(1), "sbc_p": m.group(2), "src": "NOTES 13"},
            18: {"gfull": e.group(1), "gfull_tcrit": gm.group(2), "inexact": e.group(4), "sbc_p": e.group(2),
                 "Q": e.group(3), "stage": "1", "ess_fail": "0", "pooled": p18.group(2), "src": "DESIGN 9.5"}}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--records", type=Path, default=tc.RELEASE)
    args = ap.parse_args()
    tc.apply_outdir(args)
    c = committed()
    rows = recompute(c["sizes"])
    gpath, spath = args.records / "geweke.jsonl", args.records / "sbc.jsonl"
    released = gpath.exists() and spath.exists()
    if released:
        g = {(r["tier"], r["name"]): r for r in tc.read_jsonl(gpath)}
        s = {r["name"]: r for r in tc.read_jsonl(spath)}
    else:
        rec = recorded_interim(c)
    t_max = {r["name"]: c["overrides"].get(r["name"], {}).get("t_max", c["t_max"]) for r in rows}
    assert t_max["reduced_sv_rw"] == 400 and c["t_max"] == 200

    lines = [r"\begin{tabular}{@{}rlrrllcllrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{G-full} & & \multicolumn{5}{c}{SBC} \\",
             r"\cmidrule(lr){3-6}\cmidrule(l){8-12}",
             r"Row & Composition & $G$ & $t_{C-1}$ & max $|z|$, stage & \code{exact=False} & Tier B & "
             r"min $p$, stage & $t_{\max}$ & max thin & ESS fail & pooled ESS \\", r"\midrule"]  # fmt: skip
    for r in sorted(rows, key=lambda x: x["row"]):
        row, name = r["row"], r["name"]
        has_inexact = row in c["inexact"]
        if released:
            # a composition without a record keeps its [REL] mark; ``stage`` is 1-based (= len(q))
            gf = g.get(("G-full", name))
            gi = g.get(("G-full-inexact", name))
            sb = s.get(name)
            if gf:
                assert gf["stages"][0]["G"] == r["G"], (name, gf["stages"][0]["G"], r["G"])
                assert gf["stage"] == len(gf["q"]), (name, gf["stage"], gf["q"])
                gcell = f"{gf['stages'][-1]['max_abs_z']:.2f}, {gf['stage']}" + ("" if gf["passed"] else " (fail)")
            else:
                gcell = REL
            icell = ("---" if not has_inexact else REL if not gi else
                     f"{gi['stages'][-1]['max_abs_z']:.2f}" + ("" if gi["passed"] else " (fail)"))
            if sb:
                scell = f"{sb['min_p']:.3g}, {sb['stage']}" + ("" if sb["passed"] else " (fail)")
                thin = str(sb.get("thin_max", "---"))
                fail = str(sb["ess_fail"]) if "ess_fail" in sb else "---"
                pooled = f"{sb['ess_pooled_min']:.0f}" if sb.get("ess_pooled_min") is not None else "---"
            else:
                scell, thin, fail, pooled = REL, REL, REL, REL
        elif row in rec:
            x = rec[row]
            assert "gfull_tcrit" not in x or abs(float(x["gfull_tcrit"]) - r["t_crit"]) < 0.005, (row, x, r)
            gcell = f"{x['gfull']}$^\\ast$" + (f", {x['stage']}" if "stage" in x else "")
            icell = (f"{x['inexact']}$^\\ast$" if "inexact" in x else REL) if has_inexact else "---"
            scell = f"{x['sbc_p']}$^\\ast$" + (f", {x['stage']}" if "stage" in x else "")
            thin = REL
            fail = f"{x['ess_fail']}$^\\ast$" if "ess_fail" in x else REL
            pooled = f"$\\geq {x['pooled']}^\\ast$" if "pooled" in x else REL
        else:
            gcell, icell = REL, (REL if has_inexact else "---")
            scell, thin, fail, pooled = REL, REL, REL, REL
        lines.append(f"{row} & {tc.code(name)} & {r['G']} & {r['t_crit']:.2f} & {gcell} & {icell} & {REL} & "
                     f"{scell} & {t_max[name]} & {thin} & {fail} & {pooled} \\\\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    sz = c["sizes"]
    foot = (f"{len(rows)} compositions of \\code{{src/rai_bvar/validation/joint.py}}. G-full: $C = {sz['C']}$, "
            f"$M_c = {sz['M_c']}$, $M_1 = 10^{{{int(round(math.log10(sz['M1'])))}}}$, $\\alpha = 10^{{{int(round(math.log10(sz['alpha'])))}}}$; "
            r"$G$ and the stage-1 critical value are recomputed from the committed code. "
            f"\\code{{exact=False}}: the inexact variant ({len(c['inexact'])} SV rows), recorded, not gating; "
            r"--- where the row has none. B: composition-level Tier~B of \code{tests/sbc/test_block_tier.py}. "
            f"SBC: $N = 1000$, $L = 99$, pilot {c['pilot']}, $\\alpha = 10^{{-2}}$; $t_{{\\max}}$ from "
            r"\code{T_MAX} and \code{SBC_OVERRIDES}; ESS fail: replications failing the per-replication "
            r"ESS rule; pooled ESS: minimum over quantities of the replication-pooled ESS.")  # fmt: skip
    body = tc.preamble(marks=True) + tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    if not released:
        body += "\n" + tc.note(
            r"Placeholder: [REL] cells wait for the release records \code{geweke.jsonl} and "
            r"\code{sbc.jsonl} in \code{docs/paper/data/release/} and a record hook for Tier~B (OUTLINE "
            r"TODO 7). $^\ast$Recorded only, not release records: row 205 from \code{docs/paper/NOTES.md} "
            r"item 13 of the main repository; row 18 from DESIGN \S9.5 (row 18 note), the docstring of "
            r"\code{tests/sbc/test_sbc_compositions.py} and the commit message of 7f410dc (Apple M5 Max "
            f"under load; SBC over {rec[18]['Q']} quantities, stage 1).", red=True)  # fmt: skip
    status = "[REL] release records" if released else "PLACEHOLDER [REL] except rows 205 and 18 ([R], recorded only)"
    sources = ([str(gpath), str(spath)] if released else
               ["main repo docs/paper/NOTES.md item 13 (row 205)",
                "main repo docs/phase2/DESIGN.md section 9.5 row 18 note; tests/sbc/test_sbc_compositions.py "
                "docstring; commit message of 7f410dc (row 18)"])  # fmt: skip
    sources.append(f"main repo {GTEST} (G_FULL, INEXACT_ROWS), {STEST} (T_MAX, PILOT, SBC_OVERRIDES), "
                   "joint_spec(name, 'full').names at HEAD")
    tc.write(NAME, body, generator="tab_geweke_full_sbc.py", status=status, sources=sources)


if __name__ == "__main__":
    main()
