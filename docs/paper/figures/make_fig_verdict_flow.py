"""Fig3 (Section 6.8): the verdict rule of ``CompareResult.verdict`` as a decision flow (static).

Transcribed from ``src/rai_bvar/evaluation/compare.py`` (main repository, ``CompareResult.verdict``,
defaults ``test_level=0.10``, ``mcs_alpha=0.25``, ``mc_multiple=3.0``, ``min_converged=0.95``):

  * a pair that differs along more than one declared axis, or along none while the specs differ,
    is "not comparable" (a pair with identical specs, which differs along no axis, passes);
  * dLPD = sum over common origins of (lpd_M - lpd_B); MCSE = sqrt(sum of the two models'
    per-origin MCSE^2); precise iff |dLPD| > 3 MCSE;
  * better iff dLPD > 0, precise, and (valid window and one-sided DM-HLN p_better < 0.10, or
    M's MCS p-value >= 0.25 and B's < 0.25, i.e. M is in the model confidence set at
    alpha = 0.25 and B is not); worse is the mirror image; otherwise inconclusive. The window is
    valid only when it is rolling (``Rolling.valid_tests``; ``Expanding`` is not);
  * the verdict is forced to "inconclusive" when either model converged at fewer than 95% of
    the origins.

Usage: make_fig_verdict_flow.py [--out NAME]
"""

from __future__ import annotations

import argparse

import make_common as mc
from matplotlib import pyplot as plt

W, H = 5.6, 3.9  # inches; data units are inches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="fig_verdict_flow")
    args = ap.parse_args()
    mc.style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    cx, lx, rx = 2.8, 1.12, 4.48  # centre, left lane (dLPD > 0), right lane (dLPD < 0)
    gw, gh = 2.04, 0.46  # gate boxes
    gate = dict(face="#f4f3ef", edge=mc.BASELINE)
    term = dict(face="white", edge=mc.INK, lw=1.0)

    mc.box(ax, cx, 3.66, 2.9, 0.3, "Model M against reference B, horizon h", **gate)
    mc.box(ax, cx, 3.13, 2.9, 0.38, "Axis check",
           "exactly one declared axis differs (or none: identical specs)", **gate)  # fmt: skip
    mc.box(ax, 5.0, 3.13, 1.05, 0.3, "not comparable", **term, fontsize=6.8)
    mc.arrow(ax, (cx + 1.45, 3.13), (5.0 - 0.525, 3.13), label="no", label_xy=(4.36, 3.2),
             label_ha="center")  # fmt: skip
    mc.arrow(ax, (cx, 3.51), (cx, 3.32))
    mc.arrow(ax, (cx, 2.94), (cx, 2.78), label="yes", label_xy=(cx + 0.06, 2.86))
    mc.box(ax, cx, 2.56, 2.9, 0.44, "Sign",
           r"$\Delta\mathrm{LPD}=\sum_t(\mathrm{lpd}_{M,t}-\mathrm{lpd}_{B,t})$ over common origins",
           **gate)  # fmt: skip

    lanes = [
        (lx, "> 0", [
            ("Monte Carlo error gate", r"$\Delta\mathrm{LPD} > 3\,\mathrm{MCSE}$,  "
             r"$\mathrm{MCSE}^2=\sum_t(\mathrm{mcse}^2_{M,t}+\mathrm{mcse}^2_{B,t})$"),
            ("Valid-test or MCS route", "rolling window and one-sided DM-HLN\n"
             r"$p_{\mathrm{better}} < 0.10$,  or  $p^{\mathrm{MCS}}_M \geq 0.25 > p^{\mathrm{MCS}}_B$"),
            ("Convergence gate", "both models converged at\n"
             r"$\geq 95\%$ of the forecast origins"),
        ], "better"),
        (rx, "< 0", [
            ("Monte Carlo error gate", r"$-\Delta\mathrm{LPD} > 3\,\mathrm{MCSE}$"),
            ("Valid-test or MCS route", "rolling window and one-sided DM-HLN\n"
             r"$p_{\mathrm{worse}} < 0.10$,  or  $p^{\mathrm{MCS}}_B \geq 0.25 > p^{\mathrm{MCS}}_M$"),
            ("Convergence gate", "both models converged at\n"
             r"$\geq 95\%$ of the forecast origins"),
        ], "worse"),
    ]  # fmt: skip
    ys = [1.92, 1.36, 0.80]
    bus_top = 2.34
    for x, sign, gates, outcome in lanes:
        side = 1 if x < cx else -1
        # from the sign box to the lane's first gate
        mc.arrow(ax, (cx - side * 1.45, 2.56), (x, 2.56), style="-")
        mc.arrow(ax, (x, 2.56), (x, ys[0] + gh / 2), label=r"$\Delta\mathrm{LPD}$ " + sign,
                 label_xy=(x + 0.06, 2.44))  # fmt: skip
        for k, (title, body) in enumerate(gates):
            mc.box(ax, x, ys[k], gw, gh, title, body, body_size=5.6, fontsize=6.4, **gate)
            nxt = ys[k + 1] + gh / 2 if k + 1 < len(gates) else 0.36
            mc.arrow(ax, (x, ys[k] - gh / 2), (x, nxt),
                     label="yes", label_xy=(x + 0.06, (ys[k] - gh / 2 + nxt) / 2))  # fmt: skip
            # "no" goes to the central bus
            edge = x + side * gw / 2
            mc.arrow(ax, (edge, ys[k]), (cx - side * 0.0, ys[k]), style="-")
            ax.text((edge + cx) / 2, ys[k] + 0.06, "no", ha="center", va="bottom", fontsize=6.0,
                    color=mc.INK2)  # fmt: skip
        mc.box(ax, x, 0.2, 1.1, 0.32, outcome, **term, fontsize=7.2)
    # the "no" bus (and dLPD = 0) down to "inconclusive"
    mc.arrow(ax, (cx, bus_top), (cx, 0.36), label=r"$=0$", label_xy=(cx + 0.06, 2.22))
    mc.box(ax, cx, 0.2, 1.1, 0.32, "inconclusive", **term, fontsize=7.2)
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
