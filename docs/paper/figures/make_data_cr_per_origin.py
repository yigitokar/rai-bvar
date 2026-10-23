"""Export the per-origin scores of the Clark-Ravazzolo acceptance run (OUTLINE TODO 5), for FigF1
(make_fig_cr_per_origin.py) and FigF3 (make_fig_cr_cumulative.py).

NOT RUN YET: run it once, after the acceptance run has finished, against that run's score cache::

    PYTHONDONTWRITEBYTECODE=1 /Users/yigitokar/rai-bvar/.venv/bin/python make_data_cr_per_origin.py \\
        --cache ~/.cache/rai-bvar-cr-c823d20 --repo <clean checkout of c823d20>

It reads, and never writes, the score cache: the main-stage backtests of the five arms (Constant,
SV(rw), SV, SV(rw,t10), AR-SV(0.9); identity ordering) are rebuilt exactly as
``benchmarks/clark_ravazzolo.py`` (c823d20) builds them -- same models, probe, origins, seed,
forecast draws and LPD estimator, so the same cache keys -- with ``Cache.put`` and every model fit
disabled: an origin that is not in the cache stops the export with an error instead of being
refitted. The Gaussian-approximation scores are recomputed from the cached forecast moments
(``<cache>/moments``), as the report does. The data are loaded with the script's own
``load_data`` (the pinned FRED-QD/MD 2026-09 vintage, checksummed; it is read from the library's
download cache if present).

Output (``--out``, default ``docs/paper/data/cr_acceptance_per_origin.csv``), one row per
(model, origin, horizon, variable), variable in GDP, INFL, UR, TBILL and ``joint``:
  lpd               the primary RB log score (path-integrated in the acceptance run)
  lpd_conditioned   the path-conditioned RB log score from the same draws (secondary)
  lps_gauss         the Gaussian-approximation log score (parity metric)
  lpd_mcse          delta-method MCSE of the joint primary score (joint rows only)
  crps, sq_err      marginal rows only
  converged         1 if the origin's fit passed the convergence rule
plus a JSON side file (``.meta.json``) with the cache path, the git state of ``--repo``, the
config and the number of origins.

Usage: make_data_cr_per_origin.py --cache DIR [--repo DIR] [--out CSV] [--config acceptance]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import make_common as mc

HORIZONS = (1, 2, 4, 8)


def git_state(repo: Path) -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=mc.MAIN_REPO)
    ap.add_argument("--config", default="acceptance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=mc.PAPER / "data" / "cr_acceptance_per_origin.csv")
    args = ap.parse_args()
    cache_dir = args.cache.expanduser().resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"no score cache at {cache_dir}")
    state = git_state(args.repo)
    if not state["commit"].startswith(("c823d20", "d58c9ef")) or state["dirty"]:
        print(f"warning: {args.repo} is at {state['commit'][:9]} (dirty={state['dirty']}); the "
              "cache keys are those of c823d20's script -- use a clean checkout of it")  # fmt: skip

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.repo / "src"), str(args.repo / "benchmarks")]
    import numpy as np
    import pandas as pd

    import clark_ravazzolo as cr
    from rai_bvar import Origins
    from rai_bvar.evaluation.backtest import Cache, backtest

    class ReadOnlyCache(Cache):
        def put(self, key, scores):  # noqa: ANN001
            raise RuntimeError(f"export is read-only: origin not cached (key {key[:12]}...)")

    def no_fit(self, data, **kw):  # noqa: ANN001
        raise RuntimeError(f"{self.name}: an origin is missing from the score cache; the export "
                           "never fits (wrong cache, config or code?)")  # fmt: skip

    cr.MomentsProbe.fit = no_fit  # an uncached origin fails instead of refitting

    cfg = cr.CONFIGS[args.config]
    data, _ = cr.load_data(None, None)
    origins = [str(o) for o in Origins(*cr.FULL).resolve(data.index)][:: cfg.origin_step]
    cache = ReadOnlyCache(cache_dir)
    store = str(cache_dir / "moments")
    rows = []
    for name, model in cr.models(cfg.mcmc).items():
        probe = cr.MomentsProbe(model, store, args.seed, cfg.draws, data.fingerprint())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = backtest(probe, data, origins=origins, horizon=max(HORIZONS), draws=cfg.draws,
                           seed=args.seed, cache=cache, n_jobs=1, name=name,
                           lpd_method=cfg.lpd_method)  # fmt: skip
        if res.cached != len(origins):
            raise SystemExit(f"{name}: {len(origins) - res.cached} origins were not read from the cache")
        sc = res.scores
        g = cr.gaussian_lps(probe, data, origins)
        has_cond = cr.CONDITIONED_JOINT in sc
        for i, o in enumerate(origins):
            conv = float(sc["converged"].sel(origin=o).values)
            for h in HORIZONS:
                base = {"model": name, "origin": o, "horizon": h, "converged": conv}
                rows.append({**base, "variable": "joint",
                             "lpd": float(sc["lpd"].sel(origin=o, horizon=h)),
                             "lpd_conditioned": float(sc[cr.CONDITIONED_JOINT].sel(origin=o, horizon=h))
                             if has_cond else np.nan,
                             "lps_gauss": float(g["joint"][i, h - 1]),
                             "lpd_mcse": float(sc["lpd_mcse"].sel(origin=o, horizon=h)),
                             "crps": np.nan, "sq_err": np.nan})  # fmt: skip
                for j, v in enumerate(cr.VARS):
                    sel = dict(origin=o, horizon=h, variable=v)
                    rows.append({**base, "variable": v,
                                 "lpd": float(sc["lpd_marginal"].sel(**sel)),
                                 "lpd_conditioned": float(sc[cr.CONDITIONED_MARGINAL].sel(**sel))
                                 if cr.CONDITIONED_MARGINAL in sc else np.nan,
                                 "lps_gauss": float(g["marginal"][i, h - 1, j]),
                                 "lpd_mcse": np.nan,
                                 "crps": float(sc["crps"].sel(**sel)),
                                 "sq_err": float(sc["sq_err"].sel(**sel))})  # fmt: skip
        print(f"{name}: {len(origins)} origins read from the cache")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.10g")
    meta = {"cache": str(cache_dir), "repo": str(args.repo), "git": state, "config": cfg.name,
            "lpd_method": cfg.lpd_method, "draws": cfg.draws, "seed": args.seed,
            "origins": [origins[0], origins[-1], len(origins)], "rows": len(df)}  # fmt: skip
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
