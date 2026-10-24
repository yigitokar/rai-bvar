"""FigD2 (Appendix D): Geweke max|z| per composition against the t_{C-1} critical values, with the
mutants' G statistics overlaid (log scale).

Data (JSON lines written with ``RAI_BVAR_VALIDATION_RECORD=<dir>``):
  * ``geweke.jsonl`` from ``tests/geweke`` (tiers ``G-small``, ``G-small-block``, ``G-full`` and
    ``G-full-inexact``, the recorded ``exact=False`` rows, which do not gate);
  * ``mutation.jsonl`` from ``tests/mutation``: the G tier's mutant and control statistics (the
    controls are G-small runs of the unmutated sampler at the mutation seeds).
Default input: the release validation record at 7f880e1 (Apple M5 Max, clean tree),
``benchmarks/reports/validation/7f880e1_apple-m5-max/record/{geweke,mutation}.jsonl`` [REL]; the
footer names it, with commit, machine and date from its ``meta.json``. The release run ran the slow
tiers only, so its ``geweke.jsonl`` has the tiers ``G-small-block``, ``G-full``, ``G-full-inexact``
and ``G-small-rw-h0`` (row 14 at the random-walk ``h0`` prior N(-1, 4)) but no ``G-small`` row of the
per-PR set (TabD1): the G-small controls drawn are then the mutation suite's controls only (legend
"mutation seeds"), not the registered G-small runs of TabD1. Other records can be given with
``--geweke`` (repeatable) and ``--mutation``.

Every record is plotted, failures included: a run that failed is drawn with a cross and labelled
with its reason (a statistic above t_crit, or non-finite test-function values, which count as
p = 0). The statistic of the last sequential stage is plotted; the critical value t_{C-1} at alpha/G of
the row's first record is a short vertical tick, and the shaded band spans all critical values.

Usage: make_fig_geweke_dots.py [--geweke PATH ...] [--mutation PATH] [--out NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

REC = mc.RELEASE_RECORD

# tier -> (label, colour slot, marker); categorical slots in their fixed order
SERIES = {
    "G-full": ("G-full control", mc.SERIES[0], "o"),
    "mutant": ("mutant (G-small)", mc.SERIES[1], "s"),
    "G-small": ("G-small control", mc.SERIES[2], "D"),
    "G-small-mut": ("G-small control (mutation seeds)", mc.SERIES[2], "D"),
    "G-full-inexact": ("G-full, exact=False (recorded, not gating)", mc.SERIES[3], "^"),
    "G-small-block": ("G-small, block configuration", mc.SERIES[2], "D"),
    "G-small-rw-h0": ("G-small at the random-walk h0 prior N(-1, 4)", mc.SERIES[2], "v"),
}


def last_stage(rec: dict) -> dict:
    return rec["stages"][-1] if "stages" in rec else rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--geweke", type=Path, action="append", default=None)
    ap.add_argument("--mutation", type=Path, default=REC / "mutation.jsonl")
    ap.add_argument("--out", default="fig_geweke_dots")
    args = ap.parse_args()
    gfiles = args.geweke or [REC / "geweke.jsonl"]
    missing = [p for p in [*gfiles, args.mutation] if not p.exists()]
    if missing:
        mc.placeholder(args.out, "FigD2", "Geweke max|z| per composition vs t_{C-1}",
                       ", ".join(map(str, missing)), f"python {Path(__file__).name}")  # fmt: skip
        return

    pts = []  # (row, tier, z, t_crit, failed, note)
    for f in gfiles:
        for r in mc.read_jsonl(f):
            s = last_stage(r)
            note = ""
            if not r["passed"]:
                nf = s.get("n_nonfinite", 0)
                note = (f"fails: {nf:,} non-finite test-function values" if nf
                        else "fails: max|z| above t_crit")  # fmt: skip
            pts.append((r["name"], r["tier"], s["max_abs_z"], s["t_crit"], not r["passed"], note))
    controls_seen = set()
    for r in mc.read_jsonl(args.mutation):
        if r["tier"] != "G":
            continue
        m, c = r["mutant"], r["control"]
        pts.append((r["composition"], "mutant", m["max_abs_z"], m["t_crit"], False, r["id"]))
        if r["composition"] not in controls_seen:  # one control per composition and seed
            controls_seen.add(r["composition"])
            pts.append((r["composition"], "G-small-mut", c["max_abs_z"], c["t_crit"],
                        not c["passed"], ""))  # fmt: skip

    # rows: compositions with a G-full record first (in record order), then the rest
    rows: list[str] = []
    for tier in ("G-full", "G-full-inexact", "G-small", "G-small-mut", "mutant", "G-small-rw-h0", "G-small-block"):
        for p in pts:
            if p[1] == tier and p[0] not in rows:
                rows.append(p[0])
    block_rows = {p[0] for p in pts if p[1] == "G-small-block"}
    y = {name: i for i, name in enumerate(rows)}
    print(f"{len(pts)} points on {len(rows)} rows; failures:",
          [(p[0], p[1], p[5]) for p in pts if p[4]])  # fmt: skip

    mc.style()
    fig, ax = plt.subplots(figsize=(mc.TEXTWIDTH, 0.17 * len(rows) + 1.1))
    ax.set_xscale("log")
    tc = [p[3] for p in pts]
    ax.axvspan(min(tc), max(tc), color=mc.NEUTRAL, zorder=0, lw=0)
    ax.text(np.sqrt(min(tc) * max(tc)), -1.0, f"$t_{{C-1}}$ {min(tc):.2f}-{max(tc):.2f}", ha="center",
            va="bottom", fontsize=6.5, color=mc.INK2)  # fmt: skip
    offsets = {"G-full": 0.0, "G-full-inexact": 0.22, "G-small": -0.22, "G-small-mut": -0.11,
               "G-small-rw-h0": -0.22, "G-small-block": 0.0, "mutant": 0.0}  # fmt: skip
    mut_count: dict[str, int] = {}
    ticked: set[str] = set()
    for name, tier, z, t, failed, note in pts:
        yy = y[name] + offsets[tier]
        lab, col, mk = SERIES[tier]
        if name not in ticked:  # one critical-value tick per row (its first record's t_crit)
            ticked.add(name)
            ax.plot([t, t], [y[name] - 0.38, y[name] + 0.38], color=mc.MUTED, lw=0.9, zorder=1)
        if failed:
            ax.scatter([z], [yy], marker="X", s=46, color=mc.STATUS_CRITICAL, edgecolors="white",
                       linewidths=0.8, zorder=4)  # fmt: skip
            mc.direct_label(ax, z * 1.15, yy, f"{tier}: {note}", color=mc.INK, fontsize=6.3)
            continue
        ax.scatter([z], [yy], marker=mk, s=22, color=col, edgecolors="white", linewidths=0.7,
                   zorder=3)  # fmt: skip
        if tier == "mutant":
            k = mut_count.get(name, 0)
            mut_count[name] = k + 1
            mc.direct_label(ax, z * 1.12, yy - 0.28 + 0.0 * k, note, fontsize=5.8, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r} (block)" if r in block_rows else r for r in rows], fontsize=6.3)
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.set_xlim(1.0, 3000)
    ax.set_xlabel("Geweke max |z| of the last sequential stage (log scale)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], ls="none", marker=SERIES[k][2], color=SERIES[k][1], markersize=5,
                      markeredgecolor="white", label=SERIES[k][0])
               for k in ("G-full", *(t for t in ("G-small", "G-small-mut", "G-small-rw-h0")
                                     if any(p[1] == t for p in pts)),
                         "G-full-inexact", "mutant")]  # fmt: skip
    handles.append(Line2D([], [], ls="none", marker="X", color=mc.STATUS_CRITICAL, markersize=6,
                          markeredgecolor="white", label="failed run"))  # fmt: skip
    handles.append(Line2D([], [], color=mc.MUTED, lw=0.8, label=r"row's $t_{C-1}$ critical value"))
    # between the mutant labels of rows 6 (M14) and 16 (M16), clear of M11 and M12 at the bottom
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.615), ncol=1, fontsize=6.3, frameon=True,
              facecolor="white", edgecolor=mc.GRID, framealpha=1.0)  # fmt: skip
    mc.record_status_note(fig, [*gfiles, args.mutation])
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
