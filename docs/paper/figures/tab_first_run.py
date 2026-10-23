"""TabF8 (Appendix F): the first full run of the replication, for the record (history, not results).

Source: ``benchmarks/reports/clark_ravazzolo.json`` as committed in 551f181 (code 4bb978a; config
``full``: 4 x 5000 kept draws, burn-in 1000, 1000 forecast draws; path-conditioned RB log score as
the primary estimator, the Gaussian approximation beside it). Read with ``git show 551f181:...`` so
that the table is pinned to that commit whatever the working tree holds. Every number is wrapped in
``\\fullrun{}`` (evidence tag [FULL]).

Panels: (A) SV(rw) against Constant, full sample and pre-crisis, every variable and the joint
density, h = 1, 2, 4, 8; (B) every SV arm against Constant, joint dLPD and the share of converged
origins; (C) the variant comparisons of criterion 5; (D) the structural pair. The counts that the
appendix quotes (28 marginal and 8 joint RB/Gaussian sign-disagreement cells, all at h >= 2;
SV(rw) converged at 72 of 106 origins) are asserted.

Usage: tab_first_run.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import math

import tab_common as tc

NAME = "tab_first_run"
COMMIT, PATH = "551f181", "benchmarks/reports/clark_ravazzolo.json"
VARS = ("GDP", "INFL", "UR", "TBILL")
H = ("1", "2", "4", "8")
F = tc.fullrun


def s3(x) -> str:
    return F(tc.signed(x)) if isinstance(x, (int, float)) and math.isfinite(x) else "---"


def d3(x) -> str:
    return F(f"{x:.3f}") if isinstance(x, (int, float)) and math.isfinite(x) else "---"


def pval(x) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return "---"
    return F("$<0.001$" if x < 0.001 else f"{x:.3f}")


def pct(x) -> str:
    return F(f"{100 * x:.0f}\\%")


def yes(x) -> str:
    return F("yes") if x else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    r = tc.git_show_json(COMMIT, PATH)
    cfg, n = r["config"], r["n_origins"]
    assert cfg["name"] == "full" and (cfg["mcmc"]["chains"], cfg["mcmc"]["draws"]) == (4, 5000)
    assert cfg.get("lpd_method") in (None, "conditioned")  # the field did not exist yet (D-34)
    rel = r["relative"]
    arms = list(rel["full"])

    # quoted in the appendix and section 11
    dis_m = sum(bool(c.get("rb_gauss_disagree")) for s in rel for a in rel[s]
                for v in rel[s][a]["variables"].values() for c in v.values())  # fmt: skip
    dis_j = sum(bool(c.get("rb_gauss_disagree")) for s in rel for a in rel[s]
                for c in rel[s][a]["joint"].values())  # fmt: skip
    assert (dis_m, dis_j) == (28, 8), (dis_m, dis_j)
    assert not any(rel[s][a]["variables"][v]["1"].get("rb_gauss_disagree") or
                   rel[s][a]["joint"]["1"].get("rb_gauss_disagree")
                   for s in rel for a in rel[s] for v in VARS), "a disagreement at h = 1"  # fmt: skip
    assert round(r["converged"]["full"]["SV(rw)"] * n["full"]) == 72

    # ---- panel A: SV(rw) against Constant
    a = [r"\begin{tabular}{@{}llrrcrrrrrrcr@{}}", r"\toprule",
         r" & & \multicolumn{7}{c}{Full sample, 1985Q1--2011Q2 (" + str(n["full"]) + r" origins)} & "
         r"\multicolumn{4}{c}{Pre-crisis, 1985Q1--2007Q4 (" + str(n["pre_crisis"]) + r")} \\",
         r"\cmidrule(lr){3-9}\cmidrule(l){10-13}",
         r"Var. & $h$ & $\Delta$LPS & Gauss. & Sign & Degen. & DM $p$ & CRPS & RMSE & "
         r"$\Delta$LPS & Gauss. & Sign & Degen. \\", r"\midrule"]  # fmt: skip
    full, pre = rel["full"]["SV(rw)"], rel["pre_crisis"]["SV(rw)"]
    for v in (*VARS, "joint"):
        for i, h in enumerate(H):
            if v == "joint":
                c, p = full["joint"][h], pre["joint"][h]
                cells = [s3(c["dlpd"]), s3(c["dlpd_gauss"]), yes(c["rb_gauss_disagree"]),
                         F(str(c["rb_degenerate"])), pval(c["dm_p_better"]), "", "",
                         s3(p["dlpd"]), s3(p["dlpd_gauss"]), yes(p["rb_gauss_disagree"]),
                         F(str(p["rb_degenerate"]))]  # fmt: skip
            else:
                c, p = full["variables"][v][h], pre["variables"][v][h]
                cells = [s3(c["dlps"]), s3(c["dlps_gauss"]), yes(c["rb_gauss_disagree"]),
                         F(str(c["rb_degenerate"])), pval(c["dm_p_better"]), d3(c["crps_ratio"]),
                         d3(c["rmse_ratio"]), s3(p["dlps"]), s3(p["dlps_gauss"]),
                         yes(p["rb_gauss_disagree"]), F(str(p["rb_degenerate"]))]  # fmt: skip
            a.append(f"{v if i == 0 else ''} & {h} & " + " & ".join(cells) + r" \\")
        a.append(r"\addlinespace")
    a[-1] = r"\bottomrule"
    a.append(r"\end{tabular}")

    # ---- panel B: every SV arm against Constant (joint), converged shares, disagreement cells
    b = [r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Full, joint $\Delta$LPD} & \multicolumn{4}{c}{Pre-crisis, joint "
         r"$\Delta$LPD} & \multicolumn{2}{c}{Converged} & \multicolumn{2}{c}{Sign cells} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}\cmidrule(l){12-13}",
         r"Arm & $h=1$ & 2 & 4 & 8 & 1 & 2 & 4 & 8 & full & pre & marg. & joint \\", r"\midrule"]  # fmt: skip
    for arm in arms:
        jf, jp = rel["full"][arm]["joint"], rel["pre_crisis"][arm]["joint"]
        sm = sum(bool(c.get("rb_gauss_disagree")) for s in rel
                 for vv in rel[s][arm]["variables"].values() for c in vv.values())  # fmt: skip
        sj = sum(bool(c.get("rb_gauss_disagree")) for s in rel for c in rel[s][arm]["joint"].values())
        b.append(f"{tc.tex(arm)} & " + " & ".join(s3(jf[h]["dlpd"]) for h in H) + " & "
                 + " & ".join(s3(jp[h]["dlpd"]) for h in H)
                 + f" & {pct(r['converged']['full'][arm])} & {pct(r['converged']['pre_crisis'][arm])}"
                 + f" & {F(str(sm))} & {F(str(sj))} \\\\")  # fmt: skip
    b.append(f"Constant & \\multicolumn{{8}}{{c}}{{(reference)}} & {pct(r['converged']['full']['Constant'])}"
             f" & {pct(r['converged']['pre_crisis']['Constant'])} & & \\\\")  # fmt: skip
    b += [r"\bottomrule", r"\end{tabular}"]

    # ---- panel C: variants (criterion 5) and panel D: the structural pair
    c_ = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
          r"\multicolumn{7}{@{}l}{\emph{Variants (criterion 5), full sample, joint}} \\",
          r"Variant & against & $h=1$ & 2 & 4 & 8 & DM $p$, $h=1$ \\", r"\midrule"]  # fmt: skip
    for var, refs in r["variants"].items():
        for ref, cell in refs.items():
            j = cell["joint"]
            c_.append(f"{tc.tex(var)} & {tc.tex(ref)} & " + " & ".join(s3(j[h]["dlpd"]) for h in H)
                      + f" & {pval(j['1']['dm_p_better'])} \\\\")  # fmt: skip
    st = r["structural"]
    c_ += [r"\midrule",
           r"\multicolumn{7}{@{}l}{\emph{Structural pair, SV(rw) against Constant, full sample}} \\",
           r"Variable & & $h=1$ & 2 & 4 & 8 & DM $p$, $h=1$ \\", r"\midrule"]  # fmt: skip
    for v in (*VARS, "joint"):
        cells = st["joint"] if v == "joint" else st["variables"][v]
        key = "dlpd" if v == "joint" else "dlps"
        c_.append(f"{v} & & " + " & ".join(s3(cells[h][key]) for h in H)
                  + f" & {pval(cells['1']['dm_p_better'])} \\\\")  # fmt: skip
    c_ += [r"\bottomrule", r"\end{tabular}"]

    foot = (r"History: report \code{" + PATH + "} committed in " + COMMIT + ", code 4bb978a, "
            f"config \\code{{full}} ({cfg['mcmc']['chains']} chains $\\times$ {cfg['mcmc']['draws']} "
            f"kept draws, burn-in {cfg['mcmc']['burn']}, {cfg['draws']} forecast draws per origin). "
            r"$\Delta$LPS: path-conditioned RB estimator (the primary estimator of that run); "
            r"Gauss.: Gaussian approximation; Sign: RB and Gaussian $\Delta$LPS differ in sign "
            f"({dis_m} marginal and {dis_j} joint cells over all arms and both samples, none at "
            r"$h = 1$); Degen.: RB-degenerate origins (joint MCSE $\geq 0.9$); DM $p$: one-sided "
            "DM/HLN $p$-value for ``model better'', descriptive under the expanding window. "
            "Converged: share of origins at which the fit met the convergence rule.")  # fmt: skip
    body = ("\\footnotesize\n" + tc.fit("\n".join(a)) + "\n\n\\medskip\n" + tc.fit("\n".join(b)) + "\n\n\\medskip\n"
            + tc.fit("\n".join(c_)) + "\n" + tc.note(foot))  # fmt: skip
    tc.write(NAME, body, generator="tab_first_run.py", status="[FULL] history (first full run)",
             sources=[f"main repo {PATH} at {COMMIT} (git show)"])


if __name__ == "__main__":
    main()
