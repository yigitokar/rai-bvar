"""TabF4 (Appendix F): pre-crisis sample, SV(rw) against Constant at h = 1 and h = 8.

Every acceptance number is an ``\\acc{}`` placeholder (OUTLINE section 0.5 [ACC]); the script never
reads a value from the acceptance report. The keys are those of ``relative.pre_crisis['SV(rw)']``
in the report of ``benchmarks/clark_ravazzolo.py`` (``relative_table``): per variable
``variables.<v>['<h>'].{dlps, dlps_conditioned, dlps_gauss, rb_degenerate}``, joint
``joint['<h>'].{dlpd, dlpd_conditioned, dlpd_gauss, rb_degenerate}``. The CR column is
``CR_PRE_DLPS`` of ``benchmarks/clark_ravazzolo.py`` at c823d20 (N08 section C5; Clark and
Ravazzolo's working paper), read with ``ast`` from the committed file ([R]).

``--check-acc [REPORT]`` checks that every key resolves in a report (no value is printed).

Usage: tab_cr_precrisis.py [--outdir DIR] [--check-acc [REPORT]]
"""

from __future__ import annotations

import argparse

import tab_common as tc

NAME = "tab_cr_precrisis"
SCRIPT = ("c823d20", "benchmarks/clark_ravazzolo.py")
VARS = ("GDP", "INFL", "UR", "TBILL")
HORIZONS = ("1", "8")
BASE = "relative.pre_crisis['SV(rw)']"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap, acc_keys=True)
    args = ap.parse_args()
    tc.apply_outdir(args)
    src = tc.git_show(*SCRIPT)
    cr_pre = tc.literal(src, "CR_PRE_DLPS")  # {var: (h = 1, h = 8)}
    assert tuple(cr_pre) == VARS and all(len(v) == 2 for v in cr_pre.values())
    pre_end = tc.literal(src, "PRE_CRISIS_END")
    full = tc.literal(src, "FULL")

    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{$\Delta$LPS, SV(rw) $-$ Constant} & & CR \\",
             r"\cmidrule(lr){3-5}",
             r"Var. & $h$ & Integrated & Conditioned & Gaussian & Degen. & $\Delta$LPS \\",
             r"\midrule"]  # fmt: skip
    for v in (*VARS, "joint"):
        for i, h in enumerate(HORIZONS):
            if v == "joint":
                p = f"{BASE}.joint['{h}']"
                cells = [tc.acc(f"{p}.dlpd"), tc.acc(f"{p}.dlpd_conditioned"),
                         tc.acc(f"{p}.dlpd_gauss"), tc.acc(f"{p}.rb_degenerate"), "---"]  # fmt: skip
            else:
                p = f"{BASE}.variables.{v}['{h}']"
                cr = cr_pre[v][HORIZONS.index(h)]
                cells = [tc.acc(f"{p}.dlps"), tc.acc(f"{p}.dlps_conditioned"),
                         tc.acc(f"{p}.dlps_gauss"), tc.acc(f"{p}.rb_degenerate"), tc.signed(cr)]  # fmt: skip
            lines.append(f"{v if i == 0 else ''} & {h} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    foot = (f"Origins {full[0]}--{pre_end} of the expanding-window backtest. Integrated: "
            "path-integrated RB log score (primary); Conditioned: path-conditioned RB estimator on "
            "the same draws (secondary, in no rule); Gaussian: Gaussian approximation; Degen.: "
            r"RB-degenerate origins (joint MCSE $\geq 0.9$ for either arm). CR: Clark and "
            r"Ravazzolo's VAR-SV against VAR, working paper, Gaussian-approximation log score "
            r"(\code{CR_PRE_DLPS} of \code{benchmarks/clark_ravazzolo.py}).")  # fmt: skip
    body = (tc.preamble(acc_cells=True) + tc.fit("\n".join(lines)) + "\n" + tc.note(foot) + "\n"
            + tc.note(tc.acc_placeholder_note(BASE), red=True))
    path = tc.write(NAME, body, generator="tab_cr_precrisis.py",
                    status="[ACC] placeholders (no acceptance number written); CR column [R]",
                    sources=[f"acceptance report keys {BASE}.variables[v][h], .joint[h]",
                             f"main repo {SCRIPT[1]} at {SCRIPT[0]}: CR_PRE_DLPS, FULL, PRE_CRISIS_END"])  # fmt: skip
    tc.check_acc(path, args.check_acc)


if __name__ == "__main__":
    main()
