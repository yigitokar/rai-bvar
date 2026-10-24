"""FigD1 (Appendix D): mutation power heatmap, mutants M1-M18 x tiers U, B, G, S.

Data: one JSON line per mutant x tier check, as written by ``tests/mutation/test_mutants.py`` with
``RAI_BVAR_VALIDATION_RECORD=<dir>`` (``<dir>/mutation.jsonl``). Default input: the release
validation record at 7f880e1 (Apple M5 Max, clean tree),
``benchmarks/reports/validation/7f880e1_apple-m5-max/record/mutation.jsonl`` [REL]; the footer names
it, with commit, machine and date from its ``meta.json``. Another record can be given with
``--record``; the footer then says that it is not the release record.

Cell colour = log10 of the mutant's evidence ratio against the tier's threshold:
  U  identity error / tolerance                      (caught when > 1)
  G  Geweke max|z| / t_{C-1} critical value at alpha/G (the Bonferroni level of the last stage)
  B  alpha / q, alpha = 1e-3 (block SBC or one-block invariance, Bonferroni q of the last stage)
  S  alpha / q, alpha = 1e-2 (SBC, Bonferroni q of the last stage)
so 0 is the nominal threshold, positive values are beyond it. (The sequential Gandy-Scott wrapper
rejects at stage 1 only when q <= alpha/3; the nominal alpha is the common reference here.) The
control's ratio is the small disc in the cell's corner, on the same colour scale. Grey cells: no
check is registered for that mutant and tier (``u_check=None``, no block statistic, or S not
listed). The script prints the 37-check summary it plots.

Usage: make_fig_mutation_heatmap.py [--record PATH] [--out NAME]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

TIERS = ("U", "B", "G", "S")
ALPHA = {"B": 1e-3, "S": 1e-2}
CAP = 4.0  # colour saturates at 10^4 times the threshold (and at 10^-2 below it)
FLOOR = -2.0


def ratios(rec: dict) -> tuple[float, float, str, str]:
    """(log10 mutant ratio, log10 control ratio, mutant label, control label)."""
    t = rec["tier"]
    if t == "U":
        tol = rec["tol"]
        m, c = rec["mutant_error"], rec["control_error"]
        lm = math.log10(m / tol) if m > 0 else -np.inf
        lc = math.log10(c / tol) if c > 0 else -np.inf
        return lm, lc, f"{m:.3g}", ("0" if c == 0 else f"{c:.1e}")
    if t == "G":
        m, c = rec["mutant"], rec["control"]
        return (math.log10(m["max_abs_z"] / m["t_crit"]), math.log10(c["max_abs_z"] / c["t_crit"]),
                f"|z| {m['max_abs_z']:.3g}", f"{c['max_abs_z']:.2f}")  # fmt: skip
    a = ALPHA[t]
    qm, qc = rec["mutant"]["q"][-1], rec["control"]["q"][-1]
    lm = math.log10(a / qm) if qm > 0 else np.inf
    lc = math.log10(a / qc) if qc > 0 else np.inf
    lab_m = "q < 1e-300" if qm == 0 else f"q {qm:.0e}"
    return lm, lc, lab_m, f"{qc:.2g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, default=mc.RELEASE_RECORD / "mutation.jsonl")
    ap.add_argument("--out", default="fig_mutation_heatmap")
    args = ap.parse_args()
    if not args.record.exists():
        mc.placeholder(args.out, "FigD1", "Mutation power heatmap (M1-M18 x U, B, G, S)",
                       f"{args.record} (pytest tests/mutation -m mutation with "
                       "RAI_BVAR_VALIDATION_RECORD)", f"python {Path(__file__).name}")  # fmt: skip
        return
    recs = mc.read_jsonl(args.record)
    ids = [f"M{i}" for i in range(1, 19)]
    cell = {(r["id"], r["tier"]): r for r in recs}
    n_caught = sum(bool(r["caught"]) for r in recs)
    print(f"{len(recs)} checks, {n_caught} caught; per tier:",
          {t: sum(r["tier"] == t for r in recs) for t in TIERS})  # fmt: skip

    mc.style()
    # diverging: blue arm below the threshold, red arm above, neutral grey at 0
    neg = mc.BLUE_ARM[::-1]  # dark blue .. neutral
    pos = mc.RED_ARM  # neutral .. dark red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "evidence", list(zip(np.r_[np.linspace(0, 1 / 3, len(neg))[:-1], np.linspace(1 / 3, 1, len(pos))],
                             neg[:-1] + pos)), N=256)  # fmt: skip
    norm = mcolors.Normalize(vmin=FLOOR, vmax=CAP)

    fig, ax = plt.subplots(figsize=(mc.TEXTWIDTH * 0.85, 5.6))
    ax.grid(False)
    ax.set_xlim(-0.5, len(TIERS) - 0.5)
    ax.set_ylim(len(ids) - 0.5, -0.5)
    for i, mid in enumerate(ids):
        for j, t in enumerate(TIERS):
            r = cell.get((mid, t))
            x0, y0 = j - 0.5, i - 0.5
            if r is None:
                ax.add_patch(plt.Rectangle((x0 + 0.02, y0 + 0.04), 0.96, 0.92, facecolor="#e6e5e1",
                                           edgecolor="none"))  # fmt: skip
                ax.text(j, i, "no check", ha="center", va="center", fontsize=6, color=mc.MUTED)
                continue
            lm, lc, lab_m, lab_c = ratios(r)
            face = cmap(norm(np.clip(lm, FLOOR, CAP)))
            ax.add_patch(plt.Rectangle((x0 + 0.02, y0 + 0.04), 0.96, 0.92, facecolor=face,
                                       edgecolor="none"))  # fmt: skip
            dark = norm(np.clip(lm, FLOOR, CAP)) > 0.72 or norm(np.clip(lm, FLOOR, CAP)) < 0.12
            ink = "white" if dark else mc.INK
            ax.text(j - 0.05, i - 0.12, lab_m + ("" if r["caught"] else "  (not caught)"),
                    ha="center", va="center", fontsize=6.3, color=ink, fontweight="bold")  # fmt: skip
            ax.text(j - 0.05, i + 0.22, f"control {lab_c}", ha="center", va="center",
                    fontsize=5.4, color=ink)  # fmt: skip
            # control marker on the same colour scale, ringed by the surface colour
            ax.scatter([j + 0.38], [i - 0.22], s=26, color=cmap(norm(np.clip(lc, FLOOR, CAP))),
                       edgecolors="white", linewidths=1.0, zorder=3)  # fmt: skip
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels(["U  identity", "B  block", "G  Geweke", "S  SBC"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{m}  ({cell[(m, 'G')]['composition']})" for m in ids], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40,
                      extend="both")  # fmt: skip
    cb.set_label("log10 evidence ratio of the mutant (cell) and of the control (disc): "
                 "U error/tol, G max|z|/t_crit, B and S alpha/q;  0 = threshold", fontsize=6.5)  # fmt: skip
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    mc.record_status_note(fig, [args.record])
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
