"""FigF3 (Appendix F): cumulative joint dLPD of SV(rw) against Constant over the forecast origins at
h = 1 and h = 4 (primary RB estimator: path-integrated in the acceptance run), with the end of the
pre-crisis sample (2007Q4) marked.

Data: the per-origin export ``docs/paper/data/cr_acceptance_per_origin.csv`` written by
make_data_cr_per_origin.py from the acceptance run's score cache (OUTLINE TODO 5). Origins with a
missing score at a horizon (outcome beyond the sample) are skipped. Without the export the
script writes a marked placeholder.

Usage: make_fig_cr_cumulative.py [--csv PATH] [--allow-non-acceptance --outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt

NAME = "fig_cr_cumulative"
PRE_CRISIS_END = "2007Q4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mc.add_per_origin_args(ap)
    args = ap.parse_args()
    mc.apply_outdir(args)
    df = mc.load_per_origin(args)
    if df is None:
        mc.placeholder(NAME, "FigF3", "Cumulative joint dLPD (SV(rw) - Constant) over origins, "
                       "h = 1 and h = 4, pre-crisis boundary marked",
                       "docs/paper/data/cr_acceptance_per_origin.csv (make_data_cr_per_origin.py; TODO 5)",
                       f"python {Path(__file__).name}")  # fmt: skip
        return
    mc.style()
    fig, ax = plt.subplots(figsize=(mc.TEXTWIDTH, 2.5))
    for k, h in enumerate((1, 4)):
        sel = df[(df.variable == "joint") & (df.horizon == h)]
        a = sel[sel.model == "SV(rw)"].set_index("period")["lpd"]
        b = sel[sel.model == "Constant"].set_index("period")["lpd"]
        d = (a - b).dropna().sort_index()
        x = np.array([mc.p_to_x(p) for p in d.index])
        cum = np.cumsum(d.values)
        ax.plot(x, cum, color=mc.SERIES[k], lw=1.6, label=f"h = {h}")
        mc.direct_label(ax, x[-1] + 0.3, cum[-1], f"h = {h}: {cum[-1]:+.1f}", fontsize=6.5)
    ax.axhline(0, color=mc.BASELINE, lw=0.8)
    ax.axvline(mc.p_to_x(PRE_CRISIS_END) + 0.125, color=mc.INK2, lw=0.8)
    mc.direct_label(ax, mc.p_to_x(PRE_CRISIS_END) - 0.3, ax.get_ylim()[1] * 0.92,
                    "end of pre-crisis sample", fontsize=6.0, ha="right")  # fmt: skip
    mc.quarter_axis(ax, list(df["period"].unique()))
    ax.set_xlim(right=ax.get_xlim()[1] + 2.5)
    ax.set_xlabel("forecast origin")
    ax.set_ylabel("cumulative joint dLPD (nats)")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    mc.save(fig, NAME)


if __name__ == "__main__":
    main()
