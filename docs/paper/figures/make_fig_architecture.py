"""Fig1 (Section 4.1): architecture of rai-bvar (static box diagram).

The layers and module names are those of the package (main repository, ``src/rai_bvar/``):
``api.py`` (``BVAR`` and engine resolution in ``__post_init__``), ``engines/analytic.py``,
``engines/mcmc/`` (``resolve``, ``compose``, ``plan``, ``sweep``, ``runner``, ``keys``),
``forecast/result.py`` (``ForecastResult``), ``evaluation/`` (``backtest``, ``scores``,
``compare``, ``CompareResult.verdict``), ``structural/`` (IRFs, ``ordering_sensitivity``, whose
default design is a balanced sample of orderings, ``design="random"`` a random one and
``design="all"`` every one) and
``validation/`` (independent references, Geweke, SBC, identities), which drives the production
sweep (``validation/joint.py``: ``production_sweep``). Dossier 02 section 1 and 7, dossier 03
section 1.

Usage: make_fig_architecture.py [--out NAME]
"""

from __future__ import annotations

import argparse

import make_common as mc
from matplotlib import pyplot as plt

W, H = 6.5, 4.15  # inches; data units are inches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="fig_architecture")
    args = ap.parse_args()
    mc.style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    layer = dict(face="#f4f3ef", edge=mc.BASELINE)
    accent = dict(face="#eaf2fc", edge=mc.BLUE[300])  # the shared contract
    side = dict(face="white", edge=mc.SERIES[0], lw=1.0)
    main_x, main_w = 2.45, 4.7

    mc.box(ax, main_x, 3.82, main_w, 0.56, "Specification (api.py, components/, priors/, settings.py)",
           "BVAR(mean=Linear(prior=Minnesota(...)), vol=Constant() | SV(...), lags=p, mcmc=...)\n"
           "spec() / from_spec()  ·  spec_hash  ·  axes {mean, prior, vol, lags, deterministic, outliers}",
           **layer)  # fmt: skip
    mc.box(ax, main_x, 3.1, main_w, 0.48, "Resolution (BVAR.__post_init__)",
           "'auto' defaults  →  9 engine rules  →  MCMC-only rules; no engine field;\n"
           "invalid compositions raise at construction", **layer)  # fmt: skip
    mc.arrow(ax, (main_x, 3.54), (main_x, 3.34))

    ax_x, mc_x, eng_w = 1.15, 3.6, 2.3
    mc.box(ax, ax_x, 2.2, eng_w, 0.9, "Analytic engine (NumPy/SciPy)",
           "engines/analytic.py\nconjugate NIW Minnesota, SOC/SUR rows,\n"
           "GLP hierarchy (fixed / EB / MH),\nCOVID scaling; posterior from one QR",
           **layer)  # fmt: skip
    mc.box(ax, mc_x, 2.2, eng_w, 0.9, "MCMC engine (JAX)",
           "engines/mcmc/: resolve → compose → Plan\nof Steps → sweep (batched over equations\n"
           "and chains) → runner: counter-based keys,\nchunks, failures as values, replay",
           **layer)  # fmt: skip
    mc.arrow(ax, (main_x - 0.6, 2.86), (ax_x, 2.65), connection="arc3,rad=0.0",
             label="conjugate NIW Minnesota", label_xy=(1.5, 2.8), label_ha="center")  # fmt: skip
    mc.arrow(ax, (main_x + 0.6, 2.86), (mc_x, 2.65), label="every other composition",
             label_xy=(3.4, 2.8), label_ha="center")  # fmt: skip

    mc.box(ax, main_x, 1.28, main_w, 0.48, "Predictive contract: ForecastResult (forecast/result.py)",
           "joint paths, paths_level, per-draw cond_mean and cond_chol; attrs: spec, spec_hash,\n"
           "converged, warnings, ...; one Rao–Blackwellised log predictive density for both engines",
           **accent)  # fmt: skip
    mc.arrow(ax, (ax_x, 1.75), (ax_x, 1.52))
    mc.arrow(ax, (mc_x, 1.75), (mc_x, 1.52))

    mc.box(ax, main_x, 0.42, main_w, 0.6, "Evaluation (evaluation/, structural/)",
           "backtest: look-ahead-free refits per origin, score cache  →  scores: joint and\n"
           "marginal LPD, CRPS, energy, ...  →  compare(): DM-HLN, MCS  →  verdict()\n"
           "ordering_sensitivity(): refits under recursive orderings (balanced or random, or all)",
           **layer)  # fmt: skip
    mc.arrow(ax, (main_x, 1.04), (main_x, 0.72))

    vx, vw = 5.75, 1.35
    mc.box(ax, vx, 2.2, vw, 0.9, "Validation package",
           "validation/: independent\nprior samplers, simulators\nand log_joint; identities,\n"
           "Geweke, SBC, block tier;\nruns the production sweep", **side, body_size=5.7)  # fmt: skip
    mc.arrow(ax, (vx - vw / 2, 2.2), (mc_x + eng_w / 2, 2.2), color=mc.SERIES[0])
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
