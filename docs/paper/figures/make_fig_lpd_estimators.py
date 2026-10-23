"""Fig4 (Section 6.4): path-conditioned vs path-integrated Rao-Blackwellised log score at h > 1.

Left panel (computed here, no JAX): the synthetic case of
``tests/unit/test_lpd_integrated.py::test_conditioned_estimator_misses_the_tail_that_the_integrated_one_gets``
(main repository, b9c66c0): a persistent VAR(2) with a constant and a trend (own-lag 0.9), one
parameter draw and a constant Sigma, so the exact predictive density of y_{T+8} is known in
closed form (companion form, the test's own oracle ``_exact``). The outcome is moved out along
the test's direction to a Mahalanobis distance of 0 to 6.5; at each distance the figure shows
the error of the path-conditioned estimate (SPEC section 7; +/- its delta-method MCSE) and of
the path-integrated one (``lpd_method="integrated"``, D-34) against the exact value, for
S = 200 draws (the test's size; seed 3) and S = 2000 and 20,000 (seeds 4 and 5). The test's
own assertion at 6 s.d. and S = 200 is annotated.

Right panel: the crisis cell (Constant and SV(rw), unemployment, h = 8, origin 2008Q2) at the
acceptance MCMC budget. It needs ``docs/paper/data/rb_tail_cell.json`` (OUTLINE TODO 6, written by
``rb_tail_cell.py``, which does not exist yet); until then the panel is a marked placeholder.
Expected schema (one record per estimate)::

    {"origin": "2008Q2", "variable": "UR", "horizon": 8,
     "records": [{"model": "Constant" | "SV(rw)", "method": "conditioned" | "integrated",
                  "draws": 1000 | 8000 | 64000, "fit_seed": int, "forecast_seed": int,
                  "lpd": float, "mcse": float, "weight_ess": float}, ...]}

Usage: make_fig_lpd_estimators.py [--cell PATH] [--out NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import make_common as mc
from matplotlib import pyplot as plt

DISTANCES = np.round(np.arange(0.0, 6.51, 0.25), 2)
SIZES = ((200, 3), (2000, 4), (20000, 5))  # (draws S, seed); S = 200, seed 3 is the test's case


def synthetic() -> dict:
    mc.main_repo_on_path("tests/unit")
    import test_lpd_integrated as T  # the committed test's helpers and oracle

    persistent = np.vstack(
        [[[0.3, -0.2, 0.5], [0.01, 0.02, -0.01]], 0.9 * np.eye(T.N), np.zeros((T.N, T.N))]
    )  # exactly the test's persistent system
    out = {"distance": DISTANCES.tolist(), "runs": []}
    for S, seed in SIZES:
        fr, B, L, hist = T._constant_problem(S=S, seed=seed, B=persistent)
        mean, cov = T._exact(B, L, hist, T.H)
        run = {"S": S, "seed": seed, "exact": [], "cond": [], "mcse": [], "integ": []}
        for d in DISTANCES:
            y = T._tail(mean, cov, float(d))
            run["exact"].append(T._logpdf(y, mean, cov, np.arange(T.N)))
            c, m = fr.log_pred_density(y, T.H, return_mcse=True)
            run["cond"].append(float(c))
            run["mcse"].append(float(m))
            run["integ"].append(float(fr.log_pred_density(y, T.H, lpd_method="integrated")))
        out["runs"].append(run)
    return out


def right_panel(ax, cell: Path) -> None:
    if not cell.exists():
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                   facecolor="#f7f6f3", edgecolor=mc.STATUS_CRITICAL, lw=1.0))  # fmt: skip
        ax.text(0.5, 0.80, "PLACEHOLDER (right panel)", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, fontweight="bold", color=mc.STATUS_CRITICAL)  # fmt: skip
        ax.text(0.5, 0.52, "Crisis cell: Constant and SV(rw),\nUR, h = 8, origin 2008Q2, acceptance\n"
                "MCMC budget; conditioned with\nS = 1000, 8000, 64,000 vs integrated;\n"
                "seed spread and weight ESS", transform=ax.transAxes, ha="center", va="center",
                fontsize=6.3, color=mc.INK, linespacing=1.3)  # fmt: skip
        ax.text(0.5, 0.17, "needs docs/paper/data/\nrb_tail_cell.json (TODO 6)",
                transform=ax.transAxes, ha="center", va="center", fontsize=6,
                color=mc.INK2, family="monospace")  # fmt: skip
        return
    d = json.loads(cell.read_text())
    recs = d["records"]
    models = [m for m in ("Constant", "SV(rw)") if any(r["model"] == m for r in recs)]
    draws = sorted({r["draws"] for r in recs})
    xpos = {S: i for i, S in enumerate(draws)}
    for k, model in enumerate(models):
        col = mc.SERIES[k]
        for method, mk, dx in (("conditioned", "o", -0.12), ("integrated", "D", 0.12)):
            sel = [r for r in recs if r["model"] == model and r["method"] == method]
            if not sel:
                continue
            xs = [xpos[r["draws"]] + dx + 0.05 * (k - 0.5) for r in sel]
            ax.scatter(xs, [r["lpd"] for r in sel], marker=mk, s=18, color=col,
                       edgecolors="white", linewidths=0.6, zorder=3,
                       label=f"{model}, {method}")  # fmt: skip
            for S in draws:
                ess = [r["weight_ess"] for r in sel if r["draws"] == S]
                if ess and method == "conditioned":
                    mc.direct_label(ax, xpos[S] - 0.12, ax.get_ylim()[0], f"ESS {np.median(ess):.0f}",
                                    fontsize=5.8, ha="center", va="bottom")  # fmt: skip
    ax.set_xticks(range(len(draws)))
    ax.set_xticklabels([f"S = {S:,}" for S in draws])
    ax.set_ylabel("log score (nats)")
    ax.set_title(f"{d['variable']}, h = {d['horizon']}, origin {d['origin']}", loc="left")
    ax.legend(loc="lower right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=Path, default=mc.PAPER / "data" / "rb_tail_cell.json")
    ap.add_argument("--out", default="fig_lpd_estimators")
    args = ap.parse_args()

    syn = synthetic()
    mc.style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(mc.TEXTWIDTH, 2.7),
                                  gridspec_kw={"width_ratios": [1.25, 1]})  # fmt: skip
    x = np.asarray(syn["distance"])
    for k, run in enumerate(syn["runs"]):
        col = mc.SERIES[k]
        err = np.asarray(run["cond"]) - np.asarray(run["exact"])
        m = np.asarray(run["mcse"])
        ax.fill_between(x, err - m, err + m, color=col, alpha=0.18, lw=0)
        ax.plot(x, err, color=col, lw=1.5, marker="o", markersize=2.5,
                label=f"path-conditioned, S = {run['S']:,}")  # fmt: skip
    ierr = np.max(np.abs(np.asarray([np.asarray(r["integ"]) - np.asarray(r["exact"])
                                     for r in syn["runs"]])))  # fmt: skip
    ax.plot(x, np.zeros_like(x), color=mc.INK, lw=1.2,
            label=f"path-integrated, every S (max |error| {ierr:.0e})")  # fmt: skip
    r0 = syn["runs"][0]
    j = int(np.argmin(np.abs(x - 6.0)))
    e6 = r0["cond"][j] - r0["exact"][j]
    ax.annotate(f"S = 200, 6 s.d.: {r0['cond'][j]:.1f} (MCSE {r0['mcse'][j]:.1f})\n"
                f"vs exact {r0['exact'][j]:.1f}", xy=(x[j], e6), xytext=(0.1, -7.5),
                fontsize=6.5, color=mc.INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=mc.MUTED, lw=0.6))  # fmt: skip
    ax.set_xlabel("distance of the outcome from the predictive mean (Mahalanobis s.d.)")
    ax.set_ylabel("estimate minus exact log score (nats)")
    ax.set_title("Synthetic persistent VAR(2), joint score, h = 8", loc="left")
    ax.legend(loc="lower left", fontsize=6.3)
    right_panel(ax2, args.cell)
    fig.tight_layout(w_pad=1.5)
    print("6 s.d., S = 200: conditioned %.2f (MCSE %.2f), exact %.2f, integrated %.2f"
          % (r0["cond"][j], r0["mcse"][j], r0["exact"][j], r0["integ"][j]))  # fmt: skip
    for run in syn["runs"]:
        jj = int(np.argmin(np.abs(x - 6.0)))
        print(f"  S = {run['S']}: conditioned - exact at 6 s.d. = "
              f"{run['cond'][jj] - run['exact'][jj]:.2f} (MCSE {run['mcse'][jj]:.2f})")  # fmt: skip
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
