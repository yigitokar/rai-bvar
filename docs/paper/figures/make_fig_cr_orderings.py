"""Fig6 (Section 11.9): ordering robustness of the Clark-Ravazzolo acceptance run at h = 1.

Left: dLPS of SV(rw) against Constant (mean over origins, nats) per variable and jointly, one dot
per recursive ordering (24), the identity ordering (GDP > INFL > UR > TBILL, ordering 0) marked.
Each column is labelled with the spread (max - min over orderings) of the plotted dLPS and with the
SV(rw) arm's signal-to-noise ratio at h = 1 from ``orderings.results['SV(rw)'].summary``
(``lpd_range_snr`` = sd over orderings / sigma_MC of the arm's own average log score, the ratio
the materiality flag compares with 3; ``structural/ordering.py``, ``signal_to_noise``). The noise
floor sigma_MC itself is not in the report (neither sigma_MC nor the arm's per-ordering scores are
stored), so it is not drawn; ``lpd_range / lpd_range_snr`` is range * sigma_MC / sd, not sigma_MC.
Right: the joint ordering range (``lpd_range``) by horizon for SV(rw) and for the order-invariant
Constant arm, the negative control; a ring marks a cell that ``ordering_sensitivity`` flags as
material.

Data: the acceptance report (``--acc``; ``benchmarks/reports/clark_ravazzolo_acceptance.json`` of
the main repository, to be committed, OUTLINE TODO 2):
``orderings.per_ordering[*].{ordering,label,dlps_h1,dlpd_h1}`` and
``orderings.results[*].summary[*].{variable,horizon,lpd_range,lpd_range_snr,material}``. Without
``--acc`` (or before the file exists) the script writes a marked placeholder. The report's header
is checked against the pre-registration (``make_common.acceptance_problems``); a dry run on
another report (e.g. the first full run, 551f181) needs ``--allow-non-acceptance --outdir DIR``.

Usage: make_fig_cr_orderings.py [--acc PATH] [--allow-non-acceptance] [--outdir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

VARS = ("GDP", "INFL", "UR", "TBILL")
HORIZONS = (1, 2, 4, 8)
NAME = "fig_cr_orderings"


def summary(rep: dict, model: str) -> dict[tuple[str, int], dict]:
    return {(r["variable"], int(r["horizon"])): r for r in rep["orderings"]["results"][model]["summary"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mc.add_common_args(ap)
    args = ap.parse_args()
    mc.apply_outdir(args)
    rep = mc.load_acceptance(args)
    if rep is None:
        mc.placeholder(NAME, "Fig6", "Ordering robustness at h = 1: dLPS (SV(rw) - Constant), "
                       "24 orderings; joint ordering range by horizon (SV(rw), Constant)",
                       "benchmarks/reports/clark_ravazzolo_acceptance.json (acceptance run; TODO 2)",
                       f"python {Path(__file__).name} --acc <acceptance JSON>")  # fmt: skip
        return

    per = rep["orderings"]["per_ordering"]
    sv, const = summary(rep, "SV(rw)"), summary(rep, "Constant")
    cols = [*VARS, "joint"]
    mc.style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(mc.TEXTWIDTH, 2.6),
                                  gridspec_kw={"width_ratios": [2.7, 1]})  # fmt: skip
    rng = np.random.default_rng(0)  # horizontal jitter only, for legibility
    ticklabels = []
    for i, v in enumerate(cols):
        vals = np.array([p["dlpd_h1"] if v == "joint" else p["dlps_h1"][v] for p in per])
        ident = next(p for p in per if p["ordering"] == 0)
        iv = ident["dlpd_h1"] if v == "joint" else ident["dlps_h1"][v]
        s = sv[(v, 1)]
        x = i + rng.uniform(-0.22, 0.22, size=vals.size)
        others = np.array([p["ordering"] != 0 for p in per])
        ax.scatter(x[others], vals[others], s=14, color=mc.SERIES[0], edgecolors="white",
                   linewidths=0.6, zorder=3)  # fmt: skip
        ax.scatter([i], [iv], s=40, marker="D", color=mc.SERIES[1], edgecolors="white",
                   linewidths=0.8, zorder=4)  # fmt: skip
        ticklabels.append(f"{v}\nspread {vals.max() - vals.min():.3f}\n"
                          f"SNR {s['lpd_range_snr']:.1f}" + (" *" if s["material"] else ""))
    ax.axhline(0, color=mc.BASELINE, lw=0.8, zorder=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(ticklabels, fontsize=5.6)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylabel("dLPS, SV(rw) - Constant (nats per origin)")
    ax.set_title("h = 1, all 24 recursive orderings", loc="left")
    handles = [
        Line2D([], [], ls="none", marker="o", color=mc.SERIES[0], markeredgecolor="white",
               markersize=4.5, label="ordering"),
        Line2D([], [], ls="none", marker="D", color=mc.SERIES[1], markeredgecolor="white",
               markersize=5, label="identity ordering"),
    ]  # fmt: skip
    ax.text(0.0, -0.30, "spread: max - min of dLPS over orderings; SNR: SV(rw)'s sd over orderings / "
            r"$\sigma_{MC}$ at h = 1 (* flagged material)", transform=ax.transAxes, fontsize=5.6,
            color=mc.INK2, va="top")  # fmt: skip
    ax.legend(handles=handles, loc="upper left", fontsize=6.3)

    for k, (model, s, mk) in enumerate((("SV(rw)", sv, "o"), ("Constant (control)", const, "s"))):
        y = [s[("joint", h)]["lpd_range"] for h in HORIZONS]
        flag = [bool(s[("joint", h)]["material"]) for h in HORIZONS]
        ax2.plot(range(len(HORIZONS)), y, color=mc.SERIES[k], marker=mk, markersize=4,
                 markeredgecolor="white", label=model)  # fmt: skip
        for j, f in enumerate(flag):
            if f:
                ax2.scatter([j], [y[j]], s=70, facecolors="none", edgecolors=mc.INK, lw=0.8,
                            zorder=4)  # fmt: skip
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(HORIZONS)))
    ax2.set_xticklabels([f"h = {h}" for h in HORIZONS])
    ax2.set_ylabel("joint lpd_range across orderings\n(nats per origin)")
    ax2.set_title("Joint ordering range", loc="left")
    h2 = ax2.get_legend_handles_labels()[0] + [
        Line2D([], [], ls="none", marker="o", markerfacecolor="none", markeredgecolor=mc.INK,
               markersize=7, label="flagged material")]  # fmt: skip
    ax2.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=6.0, ncol=1)
    fig.tight_layout(w_pad=1.2)
    mc.save(fig, NAME)


if __name__ == "__main__":
    main()
