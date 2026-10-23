"""FigC1 (Appendix C): the Gibbs plan as data.

Fits two compositions of the public API to the simulated data of Listing 2
(``docs/paper/code/ex_b.py``; the same three-variable simulation, sample up to 2019Q3) and prints
``fit.plan.describe()`` verbatim for each, followed by each step's read set (``Step.reads`` of the
same plan; ``describe()`` itself lists the write sets):

  * structural SV(outliers, t): ``BVAR(vol=SV(outliers=True, tails="t"), lags=2)``;
  * reduced-form SV: ``BVAR(mean=Linear(prior=Minnesota(form="reduced")), vol=SV(), lags=2)``.

The plan depends on the composition only, not on the chain length, so the fits use a tiny MCMC
budget (2 chains x 20 draws, burn-in 20) whose draws are discarded. JAX runs single-threaded on
the CPU; the run takes about a minute.

Usage: make_fig_plan.py [--out NAME]
"""

from __future__ import annotations

import argparse
import os
import re
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import make_common as mc  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def ex_b_data():
    """The simulated data of ex_a.py / ex_b.py (copied verbatim)."""
    import pandas as pd

    import rai_bvar as rb

    rng = np.random.default_rng(0)
    A = np.array([[0.4, 0.0, -0.001], [0.1, 0.6, 0.0], [15.0, 25.0, 0.85]])
    mu = np.array([0.0075, 0.006, 4.0])
    sd = np.array([0.007, 0.003, 0.25])
    corr = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.2], [0.3, 0.2, 1.0]])
    L = np.linalg.cholesky(corr * np.outer(sd, sd))
    y, Y = mu.copy(), []
    for _ in range(200 + 240):
        y = (np.eye(3) - A) @ mu + A @ y + L @ rng.standard_normal(3)
        Y.append(y)
    Y = np.array(Y[200:])
    raw = pd.DataFrame(
        {"gdp": 100 * np.exp(np.cumsum(Y[:, 0])), "prices": 50 * np.exp(np.cumsum(Y[:, 1])),
         "rate": Y[:, 2]},
        index=pd.period_range("1960Q1", periods=240, freq="Q"),
    )  # fmt: skip
    return rb.VARData.from_frame(raw, tcodes={"gdp": 5, "prices": 5, "rate": 1})


def plans() -> list[tuple[str, str, str]]:
    mc.main_repo_on_path()
    import rai_bvar as rb

    data = ex_b_data().upto("2019Q3")
    tiny = rb.MCMCSettings(chains=2, draws=20, burn=20)
    models = [
        ("Structural SV(outliers, t)",
         rb.BVAR(vol=rb.SV(outliers=True, tails="t"), lags=2, mcmc=tiny)),
        ("Reduced-form SV",
         rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(form="reduced")), vol=rb.SV(), lags=2, mcmc=tiny)),
    ]  # fmt: skip
    out = []
    for title, model in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(data)
        reads = "\n".join(f"  {s.name:<11} reads {', '.join(sorted(s.reads))}" for s in fit.plan.steps)
        # the name without the MCMC-settings hash: the plan does not depend on the budget
        name = re.sub(r", mcmc=[0-9a-f]+", "", model.name)
        out.append((f"{title}:  {name}", fit.plan.describe(), reads))
        print(out[-1][0])
        print(out[-1][1])
        print(out[-1][2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="fig_plan")
    args = ap.parse_args()
    blocks = plans()
    mc.style()
    n_lines = sum(len(d.splitlines()) + len(r.splitlines()) + 3 for _, d, r in blocks)
    fig = plt.figure(figsize=(mc.TEXTWIDTH, 0.128 * n_lines + 0.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    y = 0.99
    step = 1.0 / (n_lines + 1)
    for title, desc, reads in blocks:
        ax.text(0.0, y, title, fontsize=7.2, fontweight="bold", color=mc.INK, va="top",
                transform=ax.transAxes)  # fmt: skip
        y -= 1.25 * step
        ax.text(0.0, y, ">>> print(fit.plan.describe())\n" + desc, fontsize=6.3,
                family="monospace", color=mc.INK, va="top", linespacing=1.18,
                transform=ax.transAxes)  # fmt: skip
        y -= (len(desc.splitlines()) + 1) * step
        ax.text(0.0, y, "read sets (Step.reads of the same plan):\n" + reads, fontsize=6.0,
                family="monospace", color=mc.INK2, va="top", linespacing=1.18,
                transform=ax.transAxes)  # fmt: skip
        y -= (len(reads.splitlines()) + 1.2) * step
    mc.save(fig, args.out)


if __name__ == "__main__":
    main()
