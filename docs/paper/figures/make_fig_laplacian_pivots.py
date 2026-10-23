"""FigE1 (Appendix E): the random-walk SV path system at small sigma.

For the scaled RW AWOL system Q~ = sigma^2 Q = Lap_w + diag(delta) of one equation, the figure
compares the two factorisations of the production code (main repository,
``src/rai_bvar/engines/mcmc/linalg.py``) against a 60-digit reference, as sigma goes to 0:

  * assembled: ``tridiag_precision_sample`` on the assembled tridiagonal (diag, off, b) of
    ``SV.awol_system`` (the draw before f60d736);
  * excess-pivot recursion: ``laplacian_precision_sample`` on (delta, w, b) of
    ``SV.awol_rw_laplacian`` (the draw since f60d736);
  * reference: ``_decimal_laplacian_sample`` of ``tests/mcmc/blocks/test_sv_core.py`` (plain
    recursion on the assembled matrix in 60-digit decimal arithmetic, from the exact float64
    inputs delta, w, b and the same noise z).

The problem is the test's own: ``problem(4, T_eff=88, T_pad=96, rw=True, sigma=sigma)`` with
``prior()`` and noise ``rng(41)`` (``test_rw_awol_at_a_tiny_sigma`` uses sigma = 5e-9). Left: the
relative error of the smallest pivot (the level direction, ~ sigma^2 (1/V0 + sum D)). Right: the
largest error of the drawn levels h_t = sigma x_t over the live states. A method that returns a
non-positive or non-finite pivot (the engine's failure status) is drawn at the top of the panel.
The sigma of the Clark-Ravazzolo crash state (sigma = 4.98e-9, NOTES 17; commit f60d736) is
marked; the problem itself is the test's synthetic one, not that state. Computed on the fly from
committed code; ``--json PATH`` also writes the plotted numbers (OUTLINE section 14 names
``docs/paper/data/laplacian_pivots.json``), otherwise nothing is written but the PDF. JAX runs
single-threaded on the CPU.

Usage: make_fig_laplacian_pivots.py [--out NAME] [--json PATH]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import make_common as mc  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

SIGMAS = np.logspace(-1, -10, 28)
CR_SIGMA = 4.98e-9
TOP = 1.0  # where failures are drawn (relative error 1 = total loss)


def compute() -> dict:
    mc.main_repo_on_path("tests/mcmc/blocks")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import test_sv_core as T
    from rai_bvar.engines.mcmc.linalg import laplacian_precision_sample, tridiag_precision_sample

    sp = T.prior()
    out = {k: [] for k in ("sigma", "piv_ref", "piv_old", "piv_new", "h_err_old", "h_err_new",
                           "ld_ref", "ld_old", "ld_new")}  # fmt: skip
    for sigma in SIGMAS:
        P = T.problem(4, T_eff=88, T_pad=96, rw=True, sigma=float(sigma))
        live = P["T_eff"] + 1
        diag, off, b = (np.asarray(a) for a in T.awol(P, sp, "rw"))
        delta, w, b2 = (np.asarray(a) for a in T.lap(P, sp))
        z = T.rng(41).normal(size=delta.size)
        x0, ld0, p0 = tridiag_precision_sample(*(jnp.asarray(a) for a in (diag, off, b, z)))
        x1, ld1, p1 = laplacian_precision_sample(*(jnp.asarray(a) for a in (delta, w, b2, z)))
        xr, ldr, pr = T._decimal_laplacian_sample(delta, w, b2, z)
        pmin = float(pr[:live].min())
        out["sigma"].append(float(sigma))
        out["piv_ref"].append(pmin)
        out["ld_ref"].append(float(ldr))
        for tag, x, ld, p in (("old", x0, ld0, p0), ("new", x1, ld1, p1)):
            p, x = float(p), np.asarray(x)
            ok = p > 0 and np.isfinite(x[:live]).all()
            out[f"piv_{tag}"].append(p)
            out[f"ld_{tag}"].append(float(ld))
            err = np.max(np.abs(sigma * x[:live] - sigma * xr[:live])) if ok else np.inf
            out[f"h_err_{tag}"].append(float(err))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="fig_laplacian_pivots")
    ap.add_argument("--json", default=None, help="also write the computed numbers here")
    args = ap.parse_args()
    d = compute()
    if args.json:
        import json
        from pathlib import Path

        Path(args.json).write_text(json.dumps(d, indent=1, default=float) + "\n")
    s = np.asarray(d["sigma"])
    ref = np.asarray(d["piv_ref"])
    mc.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(mc.TEXTWIDTH, 2.5))
    series = (("old", "assembled tridiagonal", mc.SERIES[1], "s"),
              ("new", "excess-pivot recursion", mc.SERIES[0], "o"))  # fmt: skip
    for tag, label, col, mk in series:
        p = np.asarray(d[f"piv_{tag}"])
        rel = np.where(p > 0, np.abs(p - ref) / ref, np.nan)
        rel = np.where(np.isfinite(rel), np.maximum(rel, 1e-17), np.nan)
        fail = ~(p > 0)
        rel_plot = np.where(rel > TOP, TOP, rel)
        ax1.plot(s[~fail], rel_plot[~fail], color=col, marker=mk, markersize=3, lw=1.2,
                 markeredgecolor="white", markeredgewidth=0.5, label=label)  # fmt: skip
        if fail.any():
            ax1.scatter(s[fail], np.full(fail.sum(), TOP), marker="X", s=30, color=col,
                        edgecolors="white", linewidths=0.6, zorder=4)  # fmt: skip
        e = np.asarray(d[f"h_err_{tag}"])
        bad = ~np.isfinite(e)
        ax2.plot(s[~bad], np.maximum(e[~bad], 1e-17), color=col, marker=mk, markersize=3, lw=1.2,
                 markeredgecolor="white", markeredgewidth=0.5, label=label)  # fmt: skip
        if bad.any():
            ax2.scatter(s[bad], np.full(bad.sum(), 1e3), marker="X", s=30, color=col,
                        edgecolors="white", linewidths=0.6, zorder=4)  # fmt: skip
        n_fail = int(fail.sum())
        print(f"{label}: pivot <= 0 or non-finite at {n_fail} of {len(s)} sigmas"
              + (f" (largest failing sigma {s[fail].max():.2g})" if n_fail else ""))  # fmt: skip
        good = ~fail & (s <= 1e-8)
        if good.any():
            print(f"  sigma <= 1e-8: max rel. pivot error {np.nanmax(rel[good]):.2g}, "
                  f"max |h error| {np.nanmax(e[good & np.isfinite(e)]) if np.isfinite(e[good]).any() else float('nan'):.2g}")  # fmt: skip
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.axvline(CR_SIGMA, color=mc.INK2, lw=0.8)
        ax.set_xlabel(r"$\sigma$ (random-walk volatility of log variance)")
    mc.direct_label(ax1, CR_SIGMA * 1.4, 1e-11, r"$\sigma$ of the CR" + "\ncrash state, 4.98e-9",
                    fontsize=6.0, ha="right")  # fmt: skip
    ax1.set_ylabel("relative error of the smallest pivot")
    ax1.set_title("Level-direction pivot", loc="left")
    ax1.set_ylim(1e-17, 3.0)
    ax2.set_ylabel(r"max$_t$ |error of $h_t$|")
    ax2.set_title(r"Drawn levels $h_t = \sigma x_t$", loc="left")
    ax2.set_ylim(1e-17, 3e3)
    ax1.legend(loc="center left", fontsize=6.3)
    from matplotlib.lines import Line2D

    ax2.legend(handles=[Line2D([], [], ls="none", marker="X", color=mc.MUTED, markersize=6,
                               label="pivot <= 0 or non-finite draw")],
               loc="upper left", fontsize=6.3)  # fmt: skip
    fig.tight_layout(w_pad=1.5)
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
