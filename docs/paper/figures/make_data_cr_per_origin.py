"""Export the per-origin scores of the Clark-Ravazzolo acceptance run (OUTLINE TODO 5), for FigF1
(make_fig_cr_per_origin.py) and FigF3 (make_fig_cr_cumulative.py).

Run once, after the acceptance run has finished, against that run's score cache and a clean
checkout of the commit the run used (c823d20)::

    PYTHONDONTWRITEBYTECODE=1 /Users/yigitokar/rai-bvar/.venv/bin/python make_data_cr_per_origin.py \\
        --cache ~/.cache/rai-bvar-cr-c823d20 --repo <clean checkout of c823d20> \\
        --check <main repo>/benchmarks/reports/clark_ravazzolo_acceptance.json

It reads, and never writes, the score cache: the main-stage backtests of the five arms (Constant,
SV(rw), SV, SV(rw,t10), AR-SV(0.9); identity ordering) are rebuilt exactly as
``benchmarks/clark_ravazzolo.py`` (c823d20) builds them -- same models, probe, origins, seed,
forecast draws and LPD estimator, so the same cache keys -- with ``Cache.put`` and every model fit
disabled: an origin that is not in the cache stops the export with an error instead of being
refitted. The Gaussian-approximation scores are recomputed from the cached forecast moments
(``<cache>/moments``), as the report does. The data are loaded with the script's own
``load_data`` (the pinned FRED-QD/MD 2026-09 vintage, checksummed; it is read from the library's
download cache if present). The script and the library are imported from ``--repo`` (checked
after import), never from the environment's own source tree.

Output (``--out``, default ``docs/paper/figures/data/cr_acceptance/cr_acceptance_per_origin.csv``),
one row per (model, origin, horizon, variable), variable in GDP, INFL, UR, TBILL and ``joint``,
floats written with 17 significant digits (exact round trip):
  lpd                    the primary RB log score (path-integrated in the acceptance run):
                         ``lpd`` (joint rows) or ``lpd_marginal`` (marginal rows) of the cache
  lpd_conditioned        the path-conditioned RB log score from the same draws (secondary):
                         ``lpd_conditioned`` / ``lpd_marginal_conditioned``
  lps_gauss              the Gaussian-approximation log score (parity metric)
  lpd_mcse               delta-method MCSE of the primary score: ``lpd_mcse`` (joint) or
                         ``lpd_marginal_mcse`` (marginal)
  lpd_mcse_conditioned   ``lpd_mcse_conditioned`` (joint rows only)
  crps, sq_err           marginal rows only
  converged              1 if the origin's fit passed the convergence rule
plus a JSON side file (``.meta.json``) with the cache path, the git state of ``--repo``, the
config and the number of origins.

``--check REPORT`` then recomputes, from the CSV just written and with the report script's own
formulas (``relative_table``: mean, median, top three origins by |d|, n), every Delta-LPS /
Delta-LPD cell of ``relative.{full,pre_crisis}[arm]`` (primary, path-conditioned and Gaussian),
of ``variants[variant][ref]`` and ``converged``, and compares them with the report; any
difference larger than ``--tol`` stops the export with an error. The result is written into the
side file (``check``).

Usage: make_data_cr_per_origin.py --cache DIR [--repo DIR] [--out CSV] [--config acceptance]
                                  [--check REPORT] [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import make_common as mc

HORIZONS = (1, 2, 4, 8)
DEFAULT_OUT = mc.PER_ORIGIN_CSV


def git_state(repo: Path) -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


# ------------------------------------------------------------------------------------ check
def check_against_report(df, report: Path, *, pre_crisis_end: str, sv_names, base: str,
                         variants, tol: float) -> dict:  # fmt: skip
    """Recompute the report's Delta-LPS / Delta-LPD cells from the export (relative_table's
    formulas, benchmarks/clark_ravazzolo.py l.479) and compare. Returns a summary; raises
    SystemExit on any mismatch."""
    import numpy as np
    import pandas as pd

    rep = json.loads(report.read_text())
    origins = sorted(df["origin"].unique(), key=lambda o: pd.Period(o, freq="Q"))
    end = pd.Period(pre_crisis_end, freq="Q")
    samples = {"full": origins, "pre_crisis": [o for o in origins if pd.Period(o, freq="Q") <= end]}
    piv = {}  # (model, variable, horizon, column) -> Series over origin

    def series(model, var, h, col):
        k = (model, var, h, col)
        if k not in piv:
            s = df[(df.model == model) & (df.variable == var) & (df.horizon == h)]
            piv[k] = s.set_index("origin")[col]
        return piv[k]

    bad, n_cmp, worst = [], 0, 0.0
    top_checked = 0

    def cmp(where, got, want):
        nonlocal n_cmp, worst
        n_cmp += 1
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            if not (math.isfinite(got) and math.isfinite(want)):
                ok = (math.isnan(got) and math.isnan(want)) or got == want
                err = 0.0 if ok else math.inf
            else:
                err = abs(got - want)
                ok = err <= tol
            worst = max(worst, err if math.isfinite(err) else worst)
        else:
            ok = got == want
        if not ok:
            bad.append(f"{where}: export {got!r} vs report {want!r}")

    def diffs(model, ref, var, h, col, orgs):
        a = series(model, var, h, col).loc[orgs].to_numpy(float)
        b = series(ref, var, h, col).loc[orgs].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        return (a - b)[ok], np.asarray(orgs)[ok]

    def table(model, ref, orgs, row_of, where, gauss=True):
        nonlocal top_checked
        for v in mc_vars:
            for h in HORIZONS:
                row = row_of["variables"][v][str(h)]
                w = f"{where}.variables.{v}[{h}]"
                d, lab = diffs(model, ref, v, h, "lpd", orgs)
                cmp(f"{w}.n", int(d.size), row["n"])
                cmp(f"{w}.dlps", float(np.mean(d)), row["dlps"])
                cmp(f"{w}.dlps_median", float(np.median(d)), row["dlps_median"])
                top = np.argsort(-np.abs(d))[:3]
                for k, (o, val) in enumerate(row["dlps_top"]):
                    cmp(f"{w}.dlps_top[{k}].origin", str(lab[top[k]]), o)
                    cmp(f"{w}.dlps_top[{k}].d", float(d[top[k]]), val)
                    top_checked += 1
                if "dlps_conditioned" in row:
                    dq, _ = diffs(model, ref, v, h, "lpd_conditioned", orgs)
                    cmp(f"{w}.dlps_conditioned", float(np.mean(dq)), row["dlps_conditioned"])
                if gauss and "dlps_gauss" in row:
                    dg, _ = diffs(model, ref, v, h, "lps_gauss", orgs)
                    cmp(f"{w}.dlps_gauss", float(np.mean(dg)), row["dlps_gauss"])
        for h in HORIZONS:
            row = row_of["joint"][str(h)]
            w = f"{where}.joint[{h}]"
            d, _ = diffs(model, ref, "joint", h, "lpd", orgs)
            cmp(f"{w}.n", int(d.size), row["n"])
            cmp(f"{w}.dlpd", float(np.mean(d)), row["dlpd"])
            if "dlpd_conditioned" in row:
                dq, _ = diffs(model, ref, "joint", h, "lpd_conditioned", orgs)
                cmp(f"{w}.dlpd_conditioned", float(np.mean(dq)), row["dlpd_conditioned"])
            if gauss and "dlpd_gauss" in row:
                dg, _ = diffs(model, ref, "joint", h, "lps_gauss", orgs)
                cmp(f"{w}.dlpd_gauss", float(np.mean(dg)), row["dlpd_gauss"])

    mc_vars = [v for v in ("GDP", "INFL", "UR", "TBILL")]
    for sample, orgs in samples.items():
        cmp(f"n_origins.{sample}", len(orgs), rep["n_origins"][sample])
        for arm in sv_names:
            table(arm, base, orgs, rep["relative"][sample][arm], f"relative.{sample}['{arm}']")
        for model, share in rep["converged"][sample].items():
            c = df[(df.model == model) & (df.variable == "joint") & (df.horizon == 1)]
            got = float(c.set_index("origin")["converged"].loc[orgs].mean())
            cmp(f"converged.{sample}['{model}']", got, share)
    for variant in variants:
        for ref, row_of in rep["variants"][variant].items():
            table(variant, ref, samples["full"], row_of, f"variants['{variant}']['{ref}']",
                  gauss=False)  # fmt: skip
    # the cells the paper names (criterion 4 and its secondary numbers), printed for the record
    full_ur = rep["relative"]["full"]["SV(rw)"]["variables"]["UR"]["8"]
    named = {"relative.full['SV(rw)'].variables.UR['8'].dlps_top": full_ur["dlps_top"]}
    if bad:
        raise SystemExit(f"{len(bad)} of {n_cmp} report values not reproduced by the export:\n  "
                         + "\n  ".join(bad[:40]))  # fmt: skip
    return {"report": str(report), "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "values_compared": n_cmp, "top3_origins_compared": top_checked, "tolerance": tol,
            "max_abs_diff": worst, "mismatches": 0, "named": named}  # fmt: skip


# ------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=mc.MAIN_REPO)
    ap.add_argument("--config", default="acceptance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", type=Path, default=None,
                    help="the report JSON whose Delta-LPS cells the export must reproduce")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    cache_dir = args.cache.expanduser().resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"no score cache at {cache_dir}")
    repo = args.repo.expanduser().resolve()
    state = git_state(repo)
    if not state["commit"].startswith(("c823d20", "d58c9ef")) or state["dirty"]:
        print(f"warning: {repo} is at {state['commit'][:9]} (dirty={state['dirty']}); the "
              "cache keys are those of c823d20's script -- use a clean checkout of it")  # fmt: skip

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(repo / "src"), str(repo / "benchmarks")]
    import numpy as np
    import pandas as pd

    import clark_ravazzolo as cr
    import rai_bvar
    from rai_bvar import Origins
    from rai_bvar._version import __version__
    from rai_bvar.evaluation.backtest import Cache, backtest

    for mod in (cr, rai_bvar):  # an editable install's finder must not win over --repo
        if not Path(mod.__file__).resolve().is_relative_to(repo):
            raise SystemExit(f"{mod.__name__} was imported from {mod.__file__}, not from {repo}")
    print(f"code from {repo} ({state['commit'][:9]}), rai_bvar {__version__}")

    class ReadOnlyCache(Cache):
        def put(self, key, scores):  # noqa: ANN001
            raise RuntimeError(f"export is read-only: origin not cached (key {key[:12]}...)")

        def put_state(self, key, final_state):  # noqa: ANN001
            raise RuntimeError("export is read-only: no warm-start states are written")

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
    mods = cr.models(cfg.mcmc)
    for name, model in mods.items():
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
        has_cond_m = cr.CONDITIONED_MARGINAL in sc
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
                             "lpd_mcse_conditioned": float(sc["lpd_mcse_conditioned"].sel(origin=o, horizon=h))
                             if "lpd_mcse_conditioned" in sc else np.nan,
                             "crps": np.nan, "sq_err": np.nan})  # fmt: skip
                for j, v in enumerate(cr.VARS):
                    sel = dict(origin=o, horizon=h, variable=v)
                    rows.append({**base, "variable": v,
                                 "lpd": float(sc["lpd_marginal"].sel(**sel)),
                                 "lpd_conditioned": float(sc[cr.CONDITIONED_MARGINAL].sel(**sel))
                                 if has_cond_m else np.nan,
                                 "lps_gauss": float(g["marginal"][i, h - 1, j]),
                                 "lpd_mcse": float(sc["lpd_marginal_mcse"].sel(**sel))
                                 if "lpd_marginal_mcse" in sc else np.nan,
                                 "lpd_mcse_conditioned": np.nan,
                                 "crps": float(sc["crps"].sel(**sel)),
                                 "sq_err": float(sc["sq_err"].sel(**sel))})  # fmt: skip
        print(f"{name}: {res.cached} of {len(origins)} origins read from the cache")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.17g")
    meta = {"cache": str(cache_dir), "repo": str(repo), "git": state, "config": cfg.name,
            "lpd_method": cfg.lpd_method, "draws": cfg.draws, "seed": args.seed,
            "mcmc": {"chains": cfg.mcmc.chains, "draws": cfg.mcmc.draws, "burn": cfg.mcmc.burn},
            "rai_bvar": __version__, "models": list(mods),
            "origins": [origins[0], origins[-1], len(origins)], "rows": len(df),
            "csv_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}  # fmt: skip
    if args.check is not None:
        back = pd.read_csv(args.out, float_precision="round_trip")  # what was written, exactly
        meta["check"] = check_against_report(
            back, args.check, pre_crisis_end=cr.PRE_CRISIS_END,
            sv_names=[n for n in mods if n != cr.BASELINE], base=cr.BASELINE,
            variants=("SV(rw,t10)", "AR-SV(0.9)"), tol=args.tol)  # fmt: skip
        c = meta["check"]
        print(f"check: {c['values_compared']} report values reproduced (incl. "
              f"{c['top3_origins_compared']} top-3 origin cells), max |diff| {c['max_abs_diff']:.3g}"
              f" <= {c['tolerance']:g}")  # fmt: skip
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
