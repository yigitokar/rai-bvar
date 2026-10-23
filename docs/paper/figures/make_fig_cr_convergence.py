"""FigF2 (Appendix F): share of converged origins per ordering, first full run vs acceptance run.

Top: for each of the 24 recursive orderings, the share of the ordering stage's origins at which
the SV(rw) fit converged (``orderings.per_ordering[*].converged_sv``), in the first full run
(4 x 5000, report 551f181, history) and in the acceptance run (4 x 20,000), with the 95% gate of
``compare()``; the Constant arm's shares (``converged_constant``) are drawn in grey. Orderings are
in the order of the reports (ordering 0 is the identity ordering GDP > INFL > UR > TBILL).
Bottom: the joint h = 1 verdict of SV(rw) against Constant per ordering, with the convergence
gate (``verdict_h1``) and without it (``verdict_h1_ungated``), for both runs.

Data: ``--first`` (default: ``benchmarks/reports/clark_ravazzolo.json`` of the main repository,
551f181) and ``--acc`` (the acceptance report, to be committed, OUTLINE TODO 2). Without the
acceptance report the script writes a marked placeholder; see make_fig_cr_orderings.py for the
header check and the dry-run flags.

Usage: make_fig_cr_convergence.py [--acc PATH] [--first PATH] [--allow-non-acceptance] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

NAME = "fig_cr_convergence"
GATE = 0.95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mc.add_common_args(ap)
    ap.add_argument("--first", type=Path, default=mc.FIRST_RUN_JSON)
    args = ap.parse_args()
    mc.apply_outdir(args)
    acc = mc.load_acceptance(args)
    if acc is None or not args.first.exists():
        mc.placeholder(NAME, "FigF2", "Share of converged origins per ordering, SV(rw) and "
                       "Constant: first full run (4 x 5000) vs acceptance run (4 x 20,000), 95% gate; "
                       "gated and ungated h = 1 verdicts",
                       "benchmarks/reports/clark_ravazzolo_acceptance.json (acceptance run; TODO 2)",
                       f"python {Path(__file__).name} --acc <acceptance JSON>")  # fmt: skip
        return
    first = json.loads(args.first.read_text())
    runs = [("first full run, 4 x 5000", first, mc.SERIES[0], "o"),
            ("acceptance run, 4 x 20,000", acc, mc.SERIES[1], "D")]  # fmt: skip

    mc.style()
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(mc.TEXTWIDTH, 3.3), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1]})  # fmt: skip
    for k, (label, rep, col, mk) in enumerate(runs):
        per = sorted(rep["orderings"]["per_ordering"], key=lambda p: p["ordering"])
        x = np.array([p["ordering"] for p in per]) + (k - 0.5) * 0.18
        ax.scatter(x, [p["converged_sv"] for p in per], s=20, marker=mk, color=col,
                   edgecolors="white", linewidths=0.6, zorder=3, label=f"SV(rw), {label}")  # fmt: skip
        ax.scatter(x, [p["converged_constant"] for p in per], s=9, marker=mk, color=mc.MUTED,
                   edgecolors="none", zorder=2,
                   label="Constant" if k == 0 else None)  # fmt: skip
        for j, key in enumerate(("verdict_h1", "verdict_h1_ungated")):
            row = 2 * k + j
            for p in per:
                v = p[key]
                face = col if v == "better" else (mc.INK2 if v == "worse" else "#e6e5e1")
                axv.add_patch(plt.Rectangle((p["ordering"] - 0.42, row - 0.38), 0.84, 0.76,
                                            facecolor=face, edgecolor="none"))  # fmt: skip
    ax.axhline(GATE, color=mc.INK2, lw=0.9, zorder=1)
    mc.direct_label(ax, 23.6, GATE - 0.015, "95% gate", fontsize=6.3, va="top", ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of origins converged")
    ax.legend(loc="lower left", fontsize=6.3, ncol=3)
    axv.set_ylim(3.6, -0.6)
    axv.set_yticks(range(4))
    axv.set_yticklabels(["first, gated", "first, ungated", "acceptance, gated",
                         "acceptance, ungated"], fontsize=6.2)  # fmt: skip
    axv.grid(False)
    axv.tick_params(axis="y", length=0)
    axv.set_xticks(range(24))
    axv.set_xticklabels([str(i) for i in range(24)], fontsize=6)
    axv.set_xlabel("ordering (0 = GDP > INFL > UR > TBILL)")
    axv.set_xlim(-0.6, 23.6)
    handles = [Line2D([], [], ls="none", marker="s", color=c, markersize=6, label=lab)
               for c, lab in ((mc.SERIES[0], "better (first run)"), (mc.SERIES[1], "better (acceptance)"),
                              ("#e6e5e1", "inconclusive"), (mc.INK2, "worse"))]  # fmt: skip
    axv.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4,
               fontsize=6.2)  # fmt: skip
    axv.set_title("joint h = 1 verdict of SV(rw) against Constant", loc="left", fontsize=7)
    fig.tight_layout(h_pad=0.6)
    mc.save(fig, NAME)


if __name__ == "__main__":
    main()
