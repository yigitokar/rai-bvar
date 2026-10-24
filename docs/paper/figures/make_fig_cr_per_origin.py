"""FigF1 (Appendix F): per-origin dLPS of SV(rw) against Constant, unemployment, h = 8 (h = 1 for
contrast), origins 1985Q1-2011Q2: path-conditioned against path-integrated, from the same forecast
draws of the acceptance run.

Data: the per-origin export ``docs/paper/figures/data/cr_acceptance/cr_acceptance_per_origin.csv`` written by
make_data_cr_per_origin.py from the acceptance run's score cache (OUTLINE TODO 5). Columns used:
model, origin, horizon, variable, lpd (primary: path-integrated), lpd_conditioned. Without the
export the script writes a marked placeholder. At h = 1 the two estimators coincide by
construction (D-34), so the lower panel shows one series. The h = 8 panel has a symmetric-log
y axis (linear within +/-1 nat, logarithmic beyond), because the path-conditioned crisis origins
(up to about +52 nats) would otherwise flatten the path-integrated series (at most about +7.6);
the largest value of each series is labelled with its origin.

Usage: make_fig_cr_per_origin.py [--csv PATH] [--allow-non-acceptance --outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt

NAME = "fig_cr_per_origin"
PRE_CRISIS_END = "2007Q4"


def diff(df, var: str, h: int, col: str):
    a = df[(df.model == "SV(rw)") & (df.variable == var) & (df.horizon == h)].set_index("period")
    b = df[(df.model == "Constant") & (df.variable == var) & (df.horizon == h)].set_index("period")
    return (a[col] - b[col]).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mc.add_per_origin_args(ap)
    args = ap.parse_args()
    mc.apply_outdir(args)
    df = mc.load_per_origin(args)
    if df is None:
        mc.placeholder(NAME, "FigF1", "Per-origin dLPS (SV(rw) - Constant), UR, h = 8 and h = 1: "
                       "path-conditioned vs path-integrated, same draws",
                       "docs/paper/figures/data/cr_acceptance/cr_acceptance_per_origin.csv (make_data_cr_per_origin.py; TODO 5)",
                       f"python {Path(__file__).name}")  # fmt: skip
        return
    mc.style()
    fig, axes = plt.subplots(2, 1, figsize=(mc.TEXTWIDTH, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})  # fmt: skip
    for ax, h in zip(axes, (8, 1)):
        primary = ("path-integrated (primary)" if df.attrs.get("lpd_method") == "integrated"
                   else f"primary RB score ({df.attrs.get('lpd_method')})")  # fmt: skip
        series = [(primary, "lpd", mc.SERIES[0], "o")]
        if h > 1 and df["lpd_conditioned"].notna().any():
            series.append(("path-conditioned (secondary)", "lpd_conditioned", mc.SERIES[1], "s"))
        for z, (label, col, c, mk) in enumerate(series):
            d = diff(df, "UR", h, col)
            x = np.array([mc.p_to_x(p) for p in d.index])
            # the primary series is drawn on top; the secondary shows where the two differ
            ax.plot(x, d.values, color=c, lw=1.0 if z == 0 else 1.3, marker=mk,
                    markersize=2.6 if z == 0 else 2.2, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4 - z,
                    label=f"{label}: mean {np.nanmean(d.values):+.3f}")  # fmt: skip
            if h > 1:  # label the largest value of each series
                k = int(np.nanargmax(d.values))
                mc.direct_label(ax, x[k] + 0.6, d.values[k], f"{d.index[k]}: {d.values[k]:+.1f}",
                                fontsize=6.0, ha="left")  # fmt: skip
        ax.axhline(0, color=mc.BASELINE, lw=0.8)
        ax.axvline(mc.p_to_x(PRE_CRISIS_END) + 0.125, color=mc.INK2, lw=0.8)
        if h > 1:
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, max(hi, 80.0) if hi > 40 else hi)
            ticks = [t for t in (-5, -2, -1, 0, 1, 2, 5, 10, 20, 50)
                     if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]  # fmt: skip
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel(f"dLPS, h = {h} (nats)" + (", symlog" if h > 1 else ""))
        ax.set_title(f"Unemployment, h = {h}" + ("" if h > 1 else
                     "  (the estimators coincide at h = 1)"), loc="left")  # fmt: skip
        ax.legend(loc="upper left", fontsize=6.3)
    mc.direct_label(axes[0], mc.p_to_x(PRE_CRISIS_END) - 0.4, axes[0].get_ylim()[1] * 0.6,
                    "end of pre-crisis sample", fontsize=6.0, ha="right")  # fmt: skip
    mc.quarter_axis(axes[1], list(df["period"].unique()))
    axes[1].set_xlabel("forecast origin")
    fig.tight_layout(h_pad=0.6)
    mc.save(fig, NAME)


if __name__ == "__main__":
    main()
