"""The crisis cell of Fig4's right panel (OUTLINE TODO 6): Constant and SV(rw), unemployment,
h = 8, origin 2008Q2, at the Clark--Ravazzolo acceptance MCMC budget (4 x 20,000 kept draws,
burn 1000; benchmarks/clark_ravazzolo.py CONFIGS["acceptance"]).

For each model and fit seed (the acceptance run's own fit seed, fit_seed(0, "2008Q2"), plus two
more), the fit is forecast with 1000, 8000 and 64,000 draws and both log-score estimators are
evaluated at the realised value: the path-conditioned SPEC section 7 estimator and the
path-integrated one (lpd_method="integrated", D-34). Each record carries the estimate, its
delta-method MCSE and the Kish ESS (sum w)^2 / sum w^2 of the per-draw weights w = exp(logp - max),
read off the logsumexp call inside ForecastResult.log_pred_density (no private API is changed).

Run from the main repository's environment (the code under test):
    cd /Users/yigitokar/rai-bvar && uv run python \
        /Users/yigitokar/rai-bvar-paper/docs/paper/figures/rb_tail_cell.py OUT.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/yigitokar/rai-bvar")
sys.path.insert(0, str(REPO / "benchmarks"))

import clark_ravazzolo as cr  # noqa: E402

from rai_bvar.evaluation.backtest import Expanding, fit_seed  # noqa: E402
from rai_bvar.forecast import result as result_mod  # noqa: E402

ORIGIN, VARIABLE, H = "2008Q2", "UR", 8
DRAWS = (1000, 8000, 64000)
FORECAST_SEEDS = {1000: (11, 12, 13), 8000: (11, 12), 64000: (11,)}

_captured: list[np.ndarray] = []
_lse = result_mod.logsumexp


def _capture(a, *args, **kwargs):
    _captured.append(np.asarray(a, dtype=np.float64).copy())
    return _lse(a, *args, **kwargs)


result_mod.logsumexp = _capture


def _kish(logp: np.ndarray) -> float:
    w = np.exp(logp - logp.max())
    return float(w.sum() ** 2 / (w * w).sum())


def main() -> None:
    out_path = Path(sys.argv[1])
    cfg = cr.CONFIGS["acceptance"]
    data, _ = cr.load_data(None, None)
    sample = Expanding().select(data.upto(pd.Period(ORIGIN, freq="Q")))
    truth = data.realized(pd.Period(ORIGIN, freq="Q"), H)[H - 1]
    j = list(data.variables).index(VARIABLE)
    y = float(truth[j])
    models = cr.models(cfg.mcmc)
    fit_seeds = [fit_seed(0, ORIGIN), fit_seed(1001, ORIGIN), fit_seed(1002, ORIGIN)]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    records = []
    t0 = time.time()
    for name in ("Constant", "SV(rw)"):
        for fs in fit_seeds:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fit = models[name].fit(sample, seed=fs)
            for S in DRAWS:
                for fcs in FORECAST_SEEDS[S]:
                    fc = fit.forecast(H, S, seed=np.random.default_rng([fcs, S]))
                    for method in ("conditioned", "integrated"):
                        _captured.clear()
                        lpd, mcse = fc.log_pred_density(
                            np.array([y]), H, [VARIABLE], return_mcse=True, lpd_method=method
                        )
                        records.append({
                            "model": name, "method": method, "draws": S, "fit_seed": int(fs),
                            "forecast_seed": int(fcs), "lpd": float(lpd), "mcse": float(mcse),
                            "weight_ess": _kish(_captured[-1]), "converged": bool(fit.converged),
                        })  # fmt: skip
            print(f"[{time.time() - t0:6.0f}s] {name} fit seed {fs} done", flush=True)
    out = {
        "origin": ORIGIN, "variable": VARIABLE, "horizon": H, "realised": y,
        "mcmc": {"chains": cfg.mcmc.chains, "draws": cfg.mcmc.draws, "burn": cfg.mcmc.burn},
        "code_commit": commit, "script": "docs/paper/figures/rb_tail_cell.py",
        "records": records,
    }  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
