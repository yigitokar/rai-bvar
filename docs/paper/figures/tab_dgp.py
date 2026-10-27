"""TabD-DGP (Appendix D, app:dgp): size and power of the verdict rule on simulated data, after the
release (SPEC section 10.5; the pre-registered DGP study of PR #15).

One row per scored pair and horizon: the DGP, the pair, the scored verdict, its rate over the 200
replications with the binomial Monte Carlo standard error and the Wilson 95% interval (gate on: the
engine's verdict), and the better / worse / inconclusive counts with the convergence gate on and
off. The note gives, row by row, the share of replications whose two arms both converged at >= 95%
of the origins.

Source and evidence tag [R]: ``benchmarks/reports/dgp_study.json`` of the main repository at
``--ref`` (the report of the pre-registered full run; every record from commit 58ab09f with a clean
tree). The script asserts the numbers that sections/*.tex quote and that the report is labelled the
registered run.

Usage: tab_dgp.py [--outdir DIR] [--ref COMMIT]
"""

from __future__ import annotations

import argparse

import tab_common as tc

NAME = "tab_dgp"
REPORT = "benchmarks/reports/dgp_study.json"
REF = "75e0c2a"  # PR #15 head; the report files are unchanged since 1959834
RUN_COMMIT = "58ab09f99012f825eccda604abd39d6bc942fbae"

DGP_LABEL = {"linear-homoskedastic": "homoskedastic", "linear-sv": "SV", "linear-sv-outliers": "SV with outliers"}
ARM = {"Linear × Constant": r"\code{Constant}", "Linear × SV": r"\code{SV}",
       "Linear × SV(outliers)": r"\code{SV(outliers)}"}  # fmt: skip
SCORED = {"better": "``better''", "worse": "``worse''"}


def interval(d: dict) -> str:
    lo = 0.0 if d["k"] == 0 else d["lo"]
    return f"[{lo:.3f}, {d['hi']:.3f}]"


def rate(d: dict) -> str:
    se = "0" if d["k"] == 0 else f"{d['se']:.3f}"
    return f"{d['rate']:.3f} $\\pm$ {se} {interval(d)}"


def counts(g: dict) -> str:
    return f"{g['better']['k']}/{g['worse']['k']}/{g['inconclusive']['k']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--ref", default=REF, help="read the report at this commit of the main repository")
    args = ap.parse_args()
    tc.apply_outdir(args)
    j = tc.git_show_json(args.ref, REPORT)
    assert j["registered"] is True and j["design_key"] == "81f067e0cefaaea5", (j["registered"], j["design_key"])
    g = j["meta"]["git"]
    assert g["git_commit"] == RUN_COMMIT and g["git_dirty"] is False and g["preregistration_in_commit"], g
    assert j["nominal"] == 0.1 and j["verdict_defaults"]["min_converged"] == 0.95, j["verdict_defaults"]
    S = j["summary"]
    assert len(S) == 8 and all(s["n"] == 200 for s in S)

    # the numbers that sections/*.tex quote (app:dgp, sec:limitations, sec:eval-verdict)
    size = [s for s in S if s["expect"] == "not_better"]
    assert max(s["rate"] for s in size) == 0.03 and all(s["reading"] == "not above nominal" for s in size)
    power = {s["horizon"]: s for s in S if s["expect"] == "better"}
    assert (power[1]["rate"], power[4]["rate"]) == (0.675, 0.51)
    assert (power[1]["gate_off"]["better"]["k"], power[4]["gate_off"]["better"]["k"]) == (142, 108)
    out = {s["horizon"]: s for s in S if s["expect"] == "not_worse"}
    assert all(out[h]["gate_on"]["worse"]["k"] == 0 and out[h]["gate_off"]["worse"]["k"] == 0 for h in (1, 4))
    assert (out[1]["gate_pass"], out[4]["gate_pass"]) == (0.405, 0.35)
    assert (out[1]["gate_off"]["better"]["k"], out[4]["gate_off"]["better"]["k"]) == (199, 102)
    assert (out[1]["classes"]["gate"], out[4]["classes"]["gate"]) == (118, 63)
    homo = {(s["model"], s["horizon"]): s for s in S if s["row"] == "linear-homoskedastic"}
    assert homo[("Linear × SV", 1)]["gate_on"]["worse"]["k"] == 95
    assert homo[("Linear × SV(outliers)", 1)]["gate_on"]["worse"]["k"] == 190
    fam = {f["horizon"]: f["verdict"]["rate"] for f in j["family"]}
    assert fam == {1: 0.0, 4: 0.03}, fam
    conv = {(c["row"], c["arm"], c["horizon"]): c for c in j["convergence"]}
    so = conv[("linear-sv-outliers", "Linear × SV(outliers)", 1)]
    assert round(so["converged_mean"], 2) == 0.92 and so["gate_pass"] == 0.47, so
    assert max(so["worst"], key=so["worst"].get).startswith("sigma"), so["worst"]

    rows = [r"\begin{tabular}{@{}llrllll@{}}", r"\toprule",
            r"DGP & Pair (model vs reference) & $h$ & Scored & Rate $\pm$ MC s.e.\ [Wilson 95\%] & "
            r"b/w/i, gate on & b/w/i, gate off \\",
            r"\midrule"]  # fmt: skip
    prev = None
    for s in S:
        dgp = DGP_LABEL[s["row"]] if s["row"] != prev else ""
        if prev is not None and s["row"] != prev:
            rows.append(r"\addlinespace")
        prev = s["row"]
        target = "worse" if s["expect"] == "not_worse" else "better"
        assert s["target"] == target, s
        pair = f"{ARM[s['model']]} vs {ARM[s['reference']]}"
        rows.append(f"{dgp} & {pair} & {s['horizon']} & {SCORED[target]} & {rate(s['gate_on'][target])} & "
                    f"{counts(s['gate_on'])} & {counts(s['gate_off'])} \\\\")  # fmt: skip
        assert abs(s["gate_on"][target]["rate"] - s["rate"]) < 1e-12, s
    rows += [r"\bottomrule", r"\end{tabular}"]
    gate = ", ".join(f"{s['gate_pass']:.3f}".rstrip("0").rstrip(".") for s in S)
    foot = ("Recorded, not re-run: " + tc.code(REPORT) + " of the main repository at " + tc.code(args.ref)
            + " (PR~\\#15), the pre-registered full run, every record from commit "
            + tc.code(RUN_COMMIT[:7]) + " with a clean tree. 200 replications per row; "
            "b/w/i counts ``better'', ``worse'' and inconclusive verdicts; gate off is "
            "\\code{verdict(min_converged=0)}, reported, not the engine's verdict. A rate of 0 has "
            "MC s.e.\\ 0 by the plug-in formula, so its Wilson upper bound is the informative figure. "
            "Share of replications with both arms converged at $\\geq 95\\%$ of the origins, row by "
            "row: " + gate + ".")  # fmt: skip
    body = tc.fit("\n".join(rows)) + "\n" + tc.note(foot)
    tc.write(NAME, body, generator="tab_dgp.py", status="[R] recorded study report (after the release)",
             sources=[f"main repo {REPORT} at {args.ref} (records at {RUN_COMMIT[:7]})"])


if __name__ == "__main__":
    main()
