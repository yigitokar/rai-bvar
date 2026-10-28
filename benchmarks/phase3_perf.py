"""BENCH-P3: the Phase 3 performance benchmark (P3 DESIGN §10; WP36). Reported, not gated: SPEC
sets no Phase 3 target.

Cases (each fitted through the public API, ``BVAR.fit``, MCMC engine, default ``Runtime()``: one
device, 4 chains × 10,000 kept draws after 1,000 burn-in sweeps, ``thin = 1``):

* ``commonsv_n100``: the conjugate CommonSV screen (D2, P3 §5.3),
  ``Linear(Minnesota(form="conjugate", hierarchy="fixed")) × CommonSV()`` at ``n = 100, p = 4,
  T = 300`` on ``simulate_var_sv(100, 4, 300, seed = 0)`` (BENCH-P2's generator at ``n = 100``);
  the shared-factor path, ``q = 500`` (so also R29's ``q > T`` regime);
* ``hs_sv_bp2``: ``Linear(Horseshoe()) × SV()`` at the BENCH-P2 shape and data (``n = 20, p = 4,
  T = 300``);
* ``bench_p2``: BENCH-P2 itself (``benchmarks/phase2_perf.py``'s model and data), run in the same
  session so the HS × SV overhead is a same-machine, same-load ratio (it is also compared with
  the committed reference baseline ``tests/bench/baseline.json``).

As in BENCH-P2, each fit runs in a fresh process; one *repeat* is a cold process (empty
persistent compilation cache) and a warm one (the same cache). Per run: end-to-end ``fit()``
wall time, compile time, run time, post-runner time (fit assembly and diagnostics), the runner's
ms per sweep, the minimum of 3 × 1,000 sweeps of the compiled ``burn`` executable, peak RSS, the
load average and top CPU processes before and after (a run is ``contaminated`` when the 1-minute
load average exceeds 2 before or after it), the primary set's R-hat and bulk ESS by field and
the lag-1 autocorrelation of ``log λ_i`` and the CommonSV scalars (R29, R31).

Usage::

    python -I benchmarks/phase3_perf.py --out bench3.json                 # every case, 3 repeats
    python -I benchmarks/phase3_perf.py --cases hs_sv_bp2,bench_p2 --repeats 2 --out b.json
    python -I benchmarks/phase3_perf.py --quick                           # smoke run

This module imports neither JAX nor ``rai_bvar`` at import time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ("commonsv_n100", "hs_sv_bp2", "bench_p2")
CHAINS, BURN, DRAWS = 4, 1000, 10_000
LOAD_MAX = 2.0  # a run is contaminated above this 1-minute load average (the WP36 brief)


def _phase2_perf() -> Any:
    name = "wp36_phase2_perf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / "phase2_perf.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


def _top(k: int = 5) -> list[str]:
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pcpu=,comm="], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), parts[1][-60:]))
            except ValueError:
                pass
    rows.sort(reverse=True)
    return [f"{c:.0f}% {n}" for c, n in rows[:k]]


def load_snapshot() -> dict[str, Any]:
    return {"loadavg": list(os.getloadavg()), "top": _top(), "time": time.strftime("%H:%M:%S")}


# --------------------------------------------------------------------------- cases


def case_data(case: str, quick: bool = False) -> Any:
    perf = _phase2_perf()
    if case == "commonsv_n100":
        return perf.bench_data(8 if quick else 100, 2 if quick else 4, 100 if quick else 300, 0)
    if quick:
        return perf.bench_data(3, 2, 100, 0)
    return perf.bench_data()


def case_model(case: str, *, draws: int, burn: int, quick: bool = False) -> Any:
    import rai_bvar as rb

    lags = 2 if quick else 4
    settings = rb.MCMCSettings(chains=CHAINS, burn=burn, draws=draws, thin=1)
    if case == "commonsv_n100":
        prior = rb.Minnesota(form="conjugate", hierarchy="fixed")
        return rb.BVAR(mean=rb.Linear(prior=prior), vol=rb.CommonSV(), lags=lags, mcmc=settings)
    if case == "hs_sv_bp2":
        return rb.BVAR(mean=rb.Linear(prior=rb.Horseshoe()), vol=rb.SV(), lags=lags, mcmc=settings)
    if case == "bench_p2":
        return _phase2_perf().bench_model(draws=draws, burn=burn, lags=lags)
    raise ValueError(f"unknown case {case!r}")


# --------------------------------------------------------------------------- one run (child)


def _acf1(x: Any) -> float:
    import numpy as np

    z = np.asarray(x, dtype=np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    den = float(np.sum(z * z))
    return float(np.sum(z[:, 1:] * z[:, :-1]) / den) if den > 0 else float("nan")


def ess_summary(fit: Any, wall: float) -> dict[str, Any]:
    """Primary set by field: count, min and median bulk ESS, max R-hat, ESS per second; and the
    lag-1 autocorrelation of ``log λ_i`` (median over equations) and the CommonSV scalars."""
    import numpy as np

    diag = fit.diagnostics
    prim = diag[diag["primary"].astype(bool)]
    out: dict[str, Any] = {}
    for fld, rows in prim.groupby("field"):
        ess = rows["ess_bulk"].to_numpy(dtype=float)
        out[str(fld)] = {
            "count": int(ess.size),
            "ess_min": float(np.min(ess)),
            "ess_median": float(np.median(ess)),
            "rhat_max": float(np.nanmax(rows["rhat"].to_numpy(dtype=float))),
            "ess_per_s_min": float(np.min(ess) / wall),
        }
    ess = prim["ess_bulk"].to_numpy(dtype=float)
    out["primary"] = {
        "count": int(ess.size),
        "ess_min": float(np.min(ess)),
        "ess_median": float(np.median(ess)),
        "rhat_max": float(np.nanmax(prim["rhat"].to_numpy(dtype=float))),
        "n_fail": int(np.sum((prim["rhat"] >= 1.01) | (prim["ess_bulk"] <= 400))),
        "ess_per_s_min": float(np.min(ess) / wall),
    }
    mon = fit.posterior["monitor"]
    acf = {}
    if "log_lam" in mon:
        x = np.asarray(mon["log_lam"].values)  # (chain, draw, variable)
        vals = [_acf1(x[..., i]) for i in range(x.shape[-1])]
        acf["log_lam_median"] = float(np.median(vals))
        acf["log_lam_max"] = float(np.max(vals))
    for name in ("common_h_T", "common_rho", "common_sigma", "loglik"):
        if name in mon:
            acf[name] = _acf1(np.asarray(mon[name].values))
    out["acf1"] = acf
    return out


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    import rai_bvar as rb

    perf = _phase2_perf()
    if args.cache_dir:
        rb.enable_compilation_cache(args.cache_dir, min_compile_secs=0.0)
    data = case_data(args.case, args.quick)
    model = case_model(args.case, draws=args.draws, burn=args.burn, quick=args.quick)
    before = load_snapshot()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        fit = model.fit(data, runtime=rb.Runtime())
        wall = time.perf_counter() - t0
    after = load_snapshot()
    timing = dict(fit.timing)
    compile_s = float(timing.get("compile", 0.0))
    rec: dict[str, Any] = {
        "case": args.case,
        "kind": args.kind,
        "wall_s": wall,
        "compile_s": compile_s,
        "run_s": wall - compile_s,
        "post_runner_s": wall - float(timing.get("total", wall)),
        "runner_timing": timing,
        "runner_ms_per_sweep": float(timing.get("ms_per_sweep", float("nan"))),
        "converged": bool(fit.converged),
        "warnings": sorted({str(w.message).splitlines()[0][:160] for w in caught}),
        "static": {
            "mean": repr(fit.static.mean),
            "vol": repr(fit.static.vol),
            "shapes": repr(fit.static.shapes),
        },
        "load_before": before,
        "load_after": after,
        "contaminated": max(before["loadavg"][0], after["loadavg"][0]) > LOAD_MAX,
    }
    t = time.perf_counter()
    rec["ess"] = ess_summary(fit, wall)
    rec["ess_s"] = time.perf_counter() - t
    if args.sweep_reps > 0:
        b = load_snapshot()
        rec["sweep_ms"] = perf.sweep_ms(fit, sweeps=args.sweep_block, reps=args.sweep_reps)
        a = load_snapshot()
        rec["sweep_ms"]["load_before"], rec["sweep_ms"]["load_after"] = b, a
        rec["sweep_ms"]["contaminated"] = max(b["loadavg"][0], a["loadavg"][0]) > LOAD_MAX
    rec["peak_rss_bytes"] = perf._peak_rss_bytes()
    rec["machine"] = perf.machine_info()
    return rec


# --------------------------------------------------------------------------- orchestration


def _median(xs: list[Any]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over repeats, per case (cold and warm separately where the cache matters), the
    contamination count, and HS × SV over BENCH-P2 in the same session."""
    out: dict[str, Any] = {}
    for case in sorted({r["case"] for r in runs}):
        rs = [r for r in runs if r["case"] == case]
        cold = [r for r in rs if r["kind"] == "cold"]
        warm = [r for r in rs if r["kind"] == "warm"]
        out[case] = {
            "runs": len(rs),
            "contaminated": sum(bool(r["contaminated"]) for r in rs),
            "wall_cold_s": _median([r["wall_s"] for r in cold]),
            "wall_warm_s": _median([r["wall_s"] for r in warm]),
            "compile_cold_s": _median([r["compile_s"] for r in cold]),
            "compile_warm_s": _median([r["compile_s"] for r in warm]),
            "run_s": _median([r["run_s"] for r in rs]),
            "post_runner_s": _median([r["post_runner_s"] for r in rs]),
            "runner_ms_per_sweep": _median([r["runner_ms_per_sweep"] for r in rs]),
            "sweep_ms_min3x1000": _median([r["sweep_ms"]["min"] for r in rs if "sweep_ms" in r]),
            "peak_rss_mb": _median([r["peak_rss_bytes"] / 2**20 for r in rs]),
            "converged": [r["converged"] for r in rs],
            "primary_ess_min": _median([r["ess"]["primary"]["ess_min"] for r in rs]),
            "primary_rhat_max": _median([r["ess"]["primary"]["rhat_max"] for r in rs]),
            "primary_n_fail": _median([r["ess"]["primary"]["n_fail"] for r in rs]),
            "loadavg_1min": [r["load_before"]["loadavg"][0] for r in rs],
        }
    if "hs_sv_bp2" in out and "bench_p2" in out:
        h, b = out["hs_sv_bp2"], out["bench_p2"]
        out["hs_sv_over_bench_p2"] = {
            k: (h[k] / b[k] if h[k] and b[k] else None)
            for k in ("wall_warm_s", "wall_cold_s", "run_s", "sweep_ms_min3x1000")
        }
    return out


def _child_cmd(args: argparse.Namespace, case: str, kind: str, cache: str, result: Path) -> list:
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", "--case", case,
           "--kind", kind, "--cache-dir", cache, "--result", str(result), "--draws",
           str(args.draws), "--burn", str(args.burn), "--sweep-block", str(args.sweep_block),
           "--sweep-reps", str(args.sweep_reps)]  # fmt: skip
    return [*cmd, "--quick"] if args.quick else cmd


def wait_idle(threshold: float, timeout: float) -> float:
    t0 = time.time()
    while os.getloadavg()[0] >= threshold and time.time() - t0 < timeout:
        time.sleep(15)
    return time.time() - t0


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    perf = _phase2_perf()
    runs: list[dict[str, Any]] = []
    cases = [c for c in args.cases.split(",") if c]
    for r in range(args.repeats):
        for case in cases:  # interleaved: each round sees the same load drift for every case
            cache = tempfile.mkdtemp(prefix="bench-p3-xla-")
            try:
                for kind in ("cold", "warm"):
                    waited = wait_idle(args.gate, args.wait) if args.gate > 0 else 0.0
                    with tempfile.TemporaryDirectory(prefix="bench-p3-") as tmp:
                        res = Path(tmp) / "run.json"
                        env = {k: v for k, v in os.environ.items()
                               if not k.startswith("JAX_COMPILATION_CACHE")}  # fmt: skip
                        proc = subprocess.run(_child_cmd(args, case, kind, cache, res), env=env)
                        if proc.returncode != 0:
                            raise RuntimeError(f"BENCH-P3 child {case}/{kind} failed")
                        rec = json.loads(res.read_text())
                    rec["repeat"], rec["waited_s"] = r, waited
                    runs.append(rec)
                    print(f"[BENCH-P3] {r} {case} {kind}: wall {rec['wall_s']:.1f} s, compile "
                          f"{rec['compile_s']:.1f} s, sweep "
                          f"{rec.get('sweep_ms', {}).get('min', float('nan')):.2f} ms, load "
                          f"{rec['load_before']['loadavg'][0]:.2f}"
                          f"{' CONTAMINATED' if rec['contaminated'] else ''}", flush=True)  # fmt: skip
            finally:
                shutil.rmtree(cache, ignore_errors=True)
    baseline = {}
    bpath = ROOT / "tests" / "bench" / "baseline.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        baseline = {"label": b.get("label"), "machine": b.get("machine"),
                    "summary": {k: b["summary"].get(k) for k in
                                ("wall_cold_s", "wall_warm_s", "run_s", "sweep_ms_min3x1000",
                                 "runner_ms_per_sweep", "post_runner_s")}}  # fmt: skip
    return {
        "benchmark": "BENCH-P3",
        "design": "docs/phase3/DESIGN.md §10 (reported, not gated)",
        "settings": {
            "chains": CHAINS,
            "burn": args.burn,
            "draws": args.draws,
            "repeats": args.repeats,
            "cases": cases,
            "quick": args.quick,
        },
        "machine": perf.machine_info(with_jax=False),
        "summary": summarise(runs),
        "bench_p2_reference_baseline": baseline,
        "runs": runs,
    }


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--burn", type=int, default=BURN)
    ap.add_argument("--sweep-block", type=int, default=1000)
    ap.add_argument("--sweep-reps", type=int, default=3)
    ap.add_argument("--gate", type=float, default=LOAD_MAX, help="wait for load < this (0: off)")
    ap.add_argument("--wait", type=float, default=1800.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--case", default="", help=argparse.SUPPRESS)
    ap.add_argument("--kind", default="cold", help=argparse.SUPPRESS)
    ap.add_argument("--cache-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.quick and not args.child:
        args.draws, args.burn, args.sweep_block, args.repeats = 300, 100, 100, 1
    if args.child:
        rec = run_child(args)
        args.result.write_text(json.dumps(rec, indent=1, default=float))
        return 0
    result = run_benchmark(args)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
