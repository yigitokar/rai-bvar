"""TabD3 (Appendix D): G-full, the recorded exact=False rows, composition-level Tier B and full SBC.

Rows: every composition of ``src/rai_bvar/validation/joint.py`` at the main repository's HEAD (23
with row 18, ``reduced_sv_rw``). For each: G and the stage-1 critical value of G-full (recomputed
from the committed code: ``joint_spec(name, "full").names`` and ``G_FULL`` of
``tests/geweke/test_geweke_compositions.py``), the G-full result, the ``exact=False`` result where
the row has an inexact variant (``INEXACT_ROWS``), composition-level Tier B, and full SBC (minimum
p and its quantity, stage, thinning cap ``t_max`` with ``SBC_OVERRIDES``, largest thinning used,
replications failing the per-replication ESS rule, minimum pooled ESS).

Data: the release validation records at 7f880e1 (Apple M5 Max, clean tree; ``tc.RELEASE_DIR``,
``benchmarks/reports/validation/7f880e1_apple-m5-max/``, committed to main in 3d5a407) [REL]:

* ``record/geweke.jsonl``, tiers ``G-full`` and ``G-full-inexact``, and ``record/sbc.jsonl``, one
  JSON object per test as written by the test modules with ``RAI_BVAR_VALIDATION_RECORD`` set;
* composition-level Tier B (``tests/sbc/test_block_tier.py``, no record hook): the pass/fail status
  of each of its tests in ``junit_tests_sbc.xml``, the JUnit XML of the same run, keyed by the
  composition in the test's parameter.

A composition without a record keeps a red [REL] mark. The local records of unknown commit and machine
that the earlier drafts kept in ``docs/paper/figures/data/validation_record/`` (never used here) are
deleted from the paper branch.

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


def block_tier(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Composition-level Tier B from the JUnit XML: ``{composition: [status, ...]}`` for the tests
    parametrised by a composition, and ``{test id: status}`` for the rest (the volatility-block
    invariance tests, parametrised by a block)."""
    by_comp: dict[str, list[str]] = {}
    other: dict[str, str] = {}
    for key, st in tc.junit_cases(path).items():
        mod, name = key.split("::")
        if mod != "test_block_tier":
            continue
        fn, _, param = name.partition("[")
        param = param.rstrip("]")
        if fn == "test_invariance_vol_block":
            other[name] = st
        else:
            by_comp.setdefault(param, []).append(st)
    return by_comp, other


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--records", type=Path, default=tc.RELEASE)
    ap.add_argument("--junit", type=Path, default=tc.RELEASE_DIR / "junit_tests_sbc.xml")
    args = ap.parse_args()
    tc.apply_outdir(args)
    meta = tc.release_meta()
    c = committed()
    rows = recompute(c["sizes"])
    gpath, spath = args.records / "geweke.jsonl", args.records / "sbc.jsonl"
    g = {(r["tier"], r["name"]): r for r in tc.read_jsonl(gpath)}
    s = {r["name"]: r for r in tc.read_jsonl(spath)}
    bt, bt_other = block_tier(args.junit)
    assert set(bt) <= {r["name"] for r in rows}, set(bt) - {r["name"] for r in rows}
    t_max = {r["name"]: c["overrides"].get(r["name"], {}).get("t_max", c["t_max"]) for r in rows}
    assert t_max["reduced_sv_rw"] == 400 and c["t_max"] == 200

    lines = [r"\begin{tabular}{@{}rlrrllcllrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{G-full} & & \multicolumn{5}{c}{SBC} \\",
             r"\cmidrule(lr){3-6}\cmidrule(l){8-12}",
             r"Row & Composition & $G$ & $t_{C-1}$ & max $|z|$, stage & \code{exact=False} & Tier B & "
             r"min $p$, stage & $t_{\max}$ & max thin & ESS fail & pooled ESS \\", r"\midrule"]  # fmt: skip
    n_fail = 0
    for r in sorted(rows, key=lambda x: x["row"]):
        row, name = r["row"], r["name"]
        has_inexact = row in c["inexact"]
        # a composition without a record keeps its [REL] mark; ``stage`` is 1-based (= len(q))
        gf = g.get(("G-full", name))
        gi = g.get(("G-full-inexact", name))
        sb = s.get(name)
        if gf:
            assert gf["stages"][0]["G"] == r["G"], (name, gf["stages"][0]["G"], r["G"])
            assert abs(gf["stages"][0]["t_crit"] - r["t_crit"]) < 1e-9, (name, gf["stages"][0], r)
            assert gf["stage"] == len(gf["q"]), (name, gf["stage"], gf["q"])
            gcell = f"{gf['stages'][-1]['max_abs_z']:.2f}, {gf['stage']}" + ("" if gf["passed"] else " (fail)")
            n_fail += not gf["passed"]
        else:
            gcell = REL
        assert gi is None or has_inexact, (name, has_inexact)
        icell = ("---" if not has_inexact else REL if not gi else
                 f"{gi['stages'][-1]['max_abs_z']:.2f}" + ("" if gi["passed"] else " (fail)"))
        b = bt.get(name)
        bcell = "---" if not b else (f"{b.count('passed')}/{len(b)}")
        n_fail += bool(b) and b.count("passed") != len(b)
        if sb:
            assert sb["t_max"] == t_max[name], (name, sb["t_max"], t_max[name])
            assert sb["stage"] == len(sb["q"]), (name, sb["stage"], sb["q"])
            scell = f"{sb['min_p']:.3g}, {sb['stage']}" + ("" if sb["passed"] else " (fail)")
            n_fail += not sb["passed"]
            thin = str(sb["thin_max"])
            fail = str(sb["ess_fail"])
            pooled = f"{sb['ess_pooled_min']:.0f}"
        else:
            scell, thin, fail, pooled = REL, REL, REL, REL
        lines.append(f"{row} & {tc.code(name)} & {r['G']} & {r['t_crit']:.2f} & {gcell} & {icell} & {bcell} & "
                     f"{scell} & {t_max[name]} & {thin} & {fail} & {pooled} \\\\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    n_inexact = sum(k[0] == "G-full-inexact" for k in g)
    n_b = sum(len(v) for v in bt.values())
    assert all(v == "passed" for v in bt_other.values()), bt_other
    print(f"G-full {sum(k[0] == 'G-full' for k in g)}, inexact {n_inexact}, SBC {len(s)}, Tier B "
          f"{n_b} + {len(bt_other)} block-level; failures {n_fail}")  # fmt: skip
    sz = c["sizes"]
    foot = (f"{len(rows)} compositions of \\code{{src/rai_bvar/validation/joint.py}}. G-full: $C = {sz['C']}$, "
            f"$M_c = {sz['M_c']}$, $M_1 = 10^{{{int(round(math.log10(sz['M1'])))}}}$, $\\alpha = 10^{{{int(round(math.log10(sz['alpha'])))}}}$; "
            r"$G$ and the stage-1 critical value are recomputed from the committed code and equal the recorded ones; "
            r"max $|z|$ is that of the last stage. "
            f"\\code{{exact=False}}: the inexact variant ({len(c['inexact'])} SV rows), recorded, not gating; "
            r"--- where the row has none. Tier B: composition-level tests of \code{tests/sbc/test_block_tier.py} "
            r"for the row, passed of run (--- where the row has none); its "
            f"{len(bt_other)} invariance tests of the volatility blocks, parametrised by block, all passed. "
            f"SBC: $N = 1000$, $L = 99$, pilot {c['pilot']}, $\\alpha = 10^{{-2}}$; $t_{{\\max}}$ from "
            r"\code{T_MAX} and \code{SBC_OVERRIDES} (equal to the recorded one); max thin: the largest "
            r"thinning used; ESS fail: replications failing the per-replication "
            r"ESS rule; pooled ESS: minimum over quantities of the replication-pooled ESS. "
            f"Release validation record at commit {tc.RELEASE_COMMIT} (clean tree), {tc.tex(meta['machine']['cpu'])}, "
            f"{tc.tex(meta['started_utc'][:10])}: \\code{{record/geweke.jsonl}}, \\code{{record/sbc.jsonl}} and "
            r"\code{junit_tests_sbc.xml} in \code{benchmarks/reports/validation/7f880e1_apple-m5-max/}.")  # fmt: skip
    body = tc.preamble(marks=True) + tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    sources = [str(p.relative_to(tc.REPO_ROOT)) for p in (gpath, spath, args.junit)]
    sources.append(f"main repo {GTEST} (G_FULL, INEXACT_ROWS), {STEST} (T_MAX, PILOT, SBC_OVERRIDES), "
                   "joint_spec(name, 'full').names at HEAD")
    tc.write(NAME, body, generator="tab_geweke_full_sbc.py",
             status=f"[REL] release records at {tc.RELEASE_COMMIT} (Apple M5 Max)", sources=sources)


if __name__ == "__main__":
    main()
