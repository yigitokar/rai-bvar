"""TabF6 (Appendix F): convergence of the SV(rw) arm against the MCMC budget, fit level only.

Sources:

* the fit records of the convergence diagnostic, ``docs/paper/figures/data/convergence_diagnostic/
  {identity_ordering,worst_orderings}.jsonl`` (one JSON line per fit, written by ``conv_diag.py`` in
  the same folder: SV(rw), reduced Minnesota, p = 4, default SV priors, every main-stage origin,
  fit seed ``fit_seed(0, origin)``, no forecasts or scores). They are copies in the paper worktree;
  the script and its output are kept with the paper's sources (branch paper-draft), not in release 0.2.0, so the
  table carries the evidence tag [PREREG]: its numbers are asserted equal to those of item 1 of the
  committed pre-registration (``benchmarks/reports/clark_ravazzolo_preregistration.md``, c823d20);
* for the seed comparison, the first full run's report (``benchmarks/reports/clark_ravazzolo.json``
  at 551f181, [FULL], printed with ``\\fullrun``): main-stage convergence share and joint h = 1
  dLPD of SV(rw), and ``orderings.per_ordering[*].{converged_sv, dlpd_h1}`` of the ordering stage,
  which fits each ordering with its own backtest seed.

Usage: tab_cr_budget.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import tab_common as tc

NAME = "tab_cr_budget"
DIAG = tc.DATA / "convergence_diagnostic"
PREREG = ("c823d20", "benchmarks/reports/clark_ravazzolo_preregistration.md")
FIRST = ("551f181", "benchmarks/reports/clark_ravazzolo.json")
IDENTITY = ("GDP", "INFL", "UR", "TBILL")
SETTINGS = [  # conv_diag.py SETTINGS, in the table's order
    ("base", r"$4\times5000$, burn-in 1000 (first full run)"),
    ("burn", r"$4\times5000$, burn-in 5000"),
    ("chains8", r"$8\times5000$, burn-in 1000"),
    ("long", r"$4\times20{,}000$, burn-in 1000 (acceptance)"),
]


def aggregate() -> dict:
    recs = tc.read_jsonl(DIAG / "identity_ordering.jsonl") + tc.read_jsonl(DIAG / "worst_orderings.jsonl")
    errors = sum("error" in r for r in recs)
    g = defaultdict(list)
    for r in recs:
        if r["rep"] != 0:  # extra seeds at a few origins (base only); not part of item 1
            continue
        g[(r["setting"], tuple(r.get("order") or IDENTITY))].append(r)
    out = {}
    for key, rs in g.items():
        assert len({r["origin"] for r in rs}) == len(rs), key
        out[key] = {"n": len(rs), "conv": sum(bool(r.get("converged")) for r in rs),
                    "rhat": max(r["rhat_max"] for r in rs), "ess": min(r["ess_min"] for r in rs)}  # fmt: skip
    return {"cells": out, "n_identity": sum(1 for r in recs if not r.get("order")),
            "n_worst": sum(1 for r in recs if r.get("order")), "errors": errors}  # fmt: skip


def check_against_prereg(agg: dict, text: str) -> None:
    """Item 1 of the pre-registration states these numbers; the records must reproduce them."""
    c = agg["cells"]
    ident = {s: c[(s, IDENTITY)] for s, _ in SETTINGS}
    m = re.search(r"(\d+)/106 origins converge at 4 × 5000; (\d+)/106 with burn 5000;\s+(\d+)/106 "
                  r"with 8 chains; \*\*(\d+)/106 at 4 × 20,000\*\* \(max R-hat ([\d.]+), min bulk "
                  r"ESS (\d+)\)", text)  # fmt: skip
    assert m, "pre-registration item 1 (identity ordering) not found"
    assert [ident[s]["conv"] for s in ("base", "burn", "chains8", "long")] == [int(x) for x in m.groups()[:4]]
    assert f"{ident['long']['rhat']:.4f}" == m.group(5) and round(ident["long"]["ess"]) == int(m.group(6))
    m = re.search(r"(\d+)/106 and (\d+)/106 at 4 × 5000 with the main-stage fit seeds;\s+\*\*(\d+)/106 and "
                  r"(\d+)/106 at 4 × 20,000\*\* \(max R-hat ([\d.]+) and ([\d.]+), min ESS (\d+) and "
                  r"(\d+)\)", text)  # fmt: skip
    assert m, "pre-registration item 1 (worst orderings) not found"
    w1, w2 = ("INFL", "UR", "GDP", "TBILL"), ("INFL", "UR", "TBILL", "GDP")
    got = [c[("base", w1)]["conv"], c[("base", w2)]["conv"], c[("long", w1)]["conv"],
           c[("long", w2)]["conv"]]  # fmt: skip
    assert got == [int(x) for x in m.groups()[:4]], got
    assert [f"{c[('long', w)]['rhat']:.4f}" for w in (w1, w2)] == [m.group(5), m.group(6)]
    assert [round(c[("long", w)]["ess"]) for w in (w1, w2)] == [int(m.group(7)), int(m.group(8))]
    n = re.search(r"All (\d+) \+ (\d+) diagnostic fits finished without error", text)
    assert n and (agg["n_identity"], agg["n_worst"]) == (int(n.group(1)), int(n.group(2)))
    assert agg["errors"] == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    agg = aggregate()
    prereg = tc.git_show(*PREREG)
    check_against_prereg(agg, prereg)
    first = tc.git_show_json(*FIRST)
    n_full = first["n_origins"]["full"]
    po = {tuple(p["label"].split(" > ")): p for p in first["orderings"]["per_ordering"]}

    def lab(order) -> str:
        return " $>$ ".join(order)

    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Ordering & Budget & Converged origins & Max.\ R-hat & Min.\ bulk ESS \\", r"\midrule"]  # fmt: skip
    worst = sorted({o for (_, o) in agg["cells"] if o != IDENTITY})
    for order in (IDENTITY, *worst):
        first_row = True
        for s, label in SETTINGS:
            cell = agg["cells"].get((s, order))
            if cell is None:
                continue
            name = (lab(order) + (" (identity)" if order == IDENTITY else "")) if first_row else ""
            lines.append(f"{name} & {label} & {cell['conv']}/{cell['n']} & {cell['rhat']:.4f} & "
                         f"{cell['ess']:.0f} \\\\")  # fmt: skip
            first_row = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\midrule"
    lines += [r"\multicolumn{5}{@{}l}{\emph{Same ordering, $4\times5000$: main-stage fit seeds "
              r"against the first full run's ordering stage}} \\",
              r"Ordering & & Main-stage seeds & Ordering-stage seeds & \\", r"\cmidrule(r){1-1}\cmidrule(lr){3-4}"]  # fmt: skip
    for order in (IDENTITY, *worst):
        main_conv = agg["cells"][("base", order)]["conv"]
        p = po[order]
        ost = round(p["converged_sv"] * n_full)
        assert abs(p["converged_sv"] * n_full - ost) < 1e-9
        lines.append(f"{lab(order)} & & {main_conv}/{n_full} & {tc.fullrun(f'{ost}/{n_full}')} & \\\\")
    # the joint h = 1 dLPD of the same models at the two seed sets (identity ordering)
    d_main = first["relative"]["full"]["SV(rw)"]["joint"]["1"]["dlpd"]
    d_ord = po[IDENTITY]["dlpd_h1"]
    conv_main = round(first["converged"]["full"]["SV(rw)"] * n_full)
    assert conv_main == agg["cells"][("base", IDENTITY)]["conv"], "main stage vs diagnostic seeds"
    lines.append(f"\\quad joint $h = 1$ $\\Delta$LPD, identity & & {tc.fullrun(tc.signed(d_main))} & "
                 f"{tc.fullrun(tc.signed(d_ord))} & \\\\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    foot = ("Upper panel, the pre-registration's diagnostic: SV(rw) fits only (no forecasts or "
            f"scores) at the main-stage fit seeds over the {n_full} origins 1985Q1--2011Q2, from the "
            r"fit records \code{docs/paper/figures/data/convergence_diagnostic/*.jsonl}, which are "
            r"kept with the paper's sources, not with release 0.2.0; the "
            "generator checks that they reproduce item~1 of the pre-registration (c823d20) to the "
            f"digits stated there; all {agg['n_identity']} + {agg['n_worst']} fits finished without error. "
            "Converged: R-hat $< 1.01$ and bulk ESS $> 400$ on every primary quantity. Lower panel: "
            "first full run (report 551f181, history); the ordering stage fits each ordering with "
            "its own backtest seed.")  # fmt: skip
    body = tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    tc.write(NAME, body, generator="tab_cr_budget.py",
             status="[PREREG] (records uncommitted in the main repo, TODO 4); lower panel [FULL]",
             sources=["docs/paper/figures/data/convergence_diagnostic/identity_ordering.jsonl, "
                      "worst_orderings.jsonl (conv_diag.py)",
                      f"main repo {PREREG[1]} at {PREREG[0]} (item 1, asserted)",
                      f"main repo {FIRST[1]} at {FIRST[0]} (converged.full, relative.full['SV(rw)']"
                      ".joint['1'].dlpd, orderings.per_ordering)"])  # fmt: skip


if __name__ == "__main__":
    main()
