"""Convergence diagnostic for the CR SV(rw) arm (criterion 6 follow-up; no prior or criterion is
changed). For every main-stage origin, fit SV(rw) (reduced Minnesota, p = 4, default SV priors)
on the expanding window exactly as the CR main stage does (fit seed ``fit_seed(0, origin)``),
under several MCMC budgets, and record the primary diagnostics.

Usage: python conv_diag.py OUT.jsonl [--jobs 6] [--settings base,long,burn,chains8] [--reps K]
Each finished fit appends one JSON line, so a rerun skips what is done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BENCH = "/Users/yigitokar/rai-bvar/benchmarks"

SETTINGS = {
    "base": dict(chains=4, draws=5000, burn=1000),  # the CR full config
    "long": dict(chains=4, draws=20000, burn=1000),  # 4x the kept draws
    "burn": dict(chains=4, draws=5000, burn=5000),  # 5x the burn-in
    "chains8": dict(chains=8, draws=5000, burn=1000),  # 2x the chains
}


def _setup():
    sys.path.insert(0, BENCH)
    import clark_ravazzolo as cr
    from rai_bvar import Expanding, Origins

    data, _ = cr.load_data(None, None)
    origins = [str(o) for o in Origins(*cr.FULL).resolve(data.index)]
    return cr, data, origins, Expanding()


_STATE = {}


def _worker_init():
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache" / "jax-xla"))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    _STATE["setup"] = _setup()


def run_one(task):
    import numpy as np
    import pandas as pd

    from rai_bvar import MCMCSettings
    from rai_bvar.evaluation.backtest import fit_seed

    cr, data, _, window = _STATE["setup"]
    setting, origin, rep, order = (*task, None)[:4] if len(task) == 3 else task
    mcmc = MCMCSettings(**SETTINGS[setting])
    model = cr.models(mcmc)["SV(rw)"]
    src = data
    if order:
        from rai_bvar.structural.ordering import permute_model

        model = permute_model(model, list(order), variables=list(data.variables))
        src = data.select(list(order))
    sample = window.select(src.upto(pd.Period(origin, freq="Q")))
    seed = fit_seed(0, origin) if rep == 0 else fit_seed(1000 + rep, origin)
    t0 = time.time()
    rec = {"setting": setting, "origin": origin, "rep": rep, "seed": int(seed), "T": len(sample),
           "order": list(order) if order else list(data.variables)}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(sample, seed=seed)
        diag = fit.diagnostics
        prim = diag[diag["primary"].astype(bool)]
        rhat = prim["rhat"].astype(float)
        ess = prim["ess_bulk"].astype(float)
        bad = prim[(rhat >= mcmc.rhat_max) | (ess <= mcmc.ess_min)]
        rec.update(
            converged=bool(fit.converged),
            n_primary=int(len(prim)),
            n_bad=int(len(bad)),
            bad={str(q): [float(r), float(e)] for q, r, e in
                 zip(bad.index, bad["rhat"], bad["ess_bulk"], strict=True)},  # fmt: skip
            rhat_max=float(rhat.max()),
            ess_min=float(ess.min()),
            rhat={str(q): float(v) for q, v in rhat.items()},
            ess={str(q): float(v) for q, v in ess.items()},
        )
        sig = np.asarray(fit.monitor["sigma"])  # (chains, draws, n)
        flat = sig.reshape(-1, sig.shape[-1])
        rec["sigma_median"] = [float(v) for v in np.median(flat, axis=0)]
        rec["sigma_p_below_0.05"] = [float(v) for v in np.mean(flat < 0.05, axis=0)]
        rec["sigma_chain_medians"] = np.median(sig, axis=1).tolist()
        eqs = diag.attrs.get("equations")
        if isinstance(eqs, pd.DataFrame):
            rec["equations"] = json.loads(eqs.to_json(orient="index"))
        rec["warnings"] = sorted({str(w.message)[:160] for w in caught})
    except Exception as err:  # record and continue (a crash is a finding)
        rec.update(error=f"{type(err).__name__}: {err}"[:500])
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--settings", default="base,long,burn,chains8")
    p.add_argument("--reps", type=int, default=0, help="extra seeds per origin (base only)")
    p.add_argument("--rep-origins", default="", help="comma-separated origins for --reps")
    p.add_argument("--orders", default="", help="semicolon-separated orderings, e.g. INFL,UR,GDP,TBILL")
    a = p.parse_args()
    _, _, origins, _ = _setup()
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["origin"], r["rep"], tuple(r.get("order") or ())))
    orders = [tuple(x.split(",")) for x in a.orders.split(";") if x]
    if orders:
        tasks = [(s, o, 0, od) for od in orders for s in a.settings.split(",") if s for o in origins]
    else:
        tasks = [(s, o, 0) for s in a.settings.split(",") if s for o in origins]
    if a.reps:
        reps_at = [o for o in a.rep_origins.split(",") if o] or origins
        tasks += [("base", o, r) for o in reps_at for r in range(1, a.reps + 1)]
    default = ("GDP", "INFL", "UR", "TBILL")
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3] if len(t) > 3 else default) not in done]
    print(f"{len(origins)} origins, {len(tasks)} fits to run", flush=True)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=ctx, initializer=_worker_init) as pool, \
            a.out.open("a") as fh:  # fmt: skip
        for i, rec in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"[{time.time() - t0:6.0f}s] {i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
