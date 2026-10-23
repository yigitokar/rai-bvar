"""BENCH-P2: the Phase 2 performance benchmark (DESIGN §10.1, §10.6; WP24).

The target (§10.1): ``BVAR(Linear(Minnesota(form="structural", hierarchy="fixed")), SV(), lags=4,
mcmc=MCMCSettings(chains=4, burn=1000, draws=10_000, thin=1))`` on the committed simulated
stationary VAR(4)-SV DGP (``n = 20``, ``T = 300``, ``T_pad = 320``, seed 0; WP3's
``benchmarks/pilot/pilot_common.simulate_var_sv``), fitted through the public API
(``BVAR.fit``, MCMC engine, default ``Runtime()``: one device). Pass: end-to-end ``fit()`` wall
time < 120 s on one device on the idle reference laptop; stretch < 60 s with
``set_cpu_devices(4)`` (``--devices 4``).

Each fit runs in a fresh process, so compile time is never hidden by the in-process executable
cache. One *repeat* is two processes sharing a new, empty persistent compilation cache directory:

* **cold**: empty cache, every executable is traced and compiled;
* **warm**: the same program again, executables loaded from the persistent cache (§10.4 lever 7).

Per run the child records: end-to-end ``fit()`` wall time (data loading excluded; compilation,
sampling, storage and diagnostics included), the runner's compile time (``jax.monitoring``
compile events: tracing, lowering and backend compile or cache load), run time (wall − compile),
the runner's ms per sweep over the 10,000 sampling sweeps, **ms per sweep as the minimum of
3 × 1000 sweeps** of the compiled ``burn`` executable on the final state (§10.1), peak RSS, and
ESS per second (bulk ESS / end-to-end wall time) of the log-likelihood, ``h_{·,T}``, ``μ``, ``σ``
(and ``φ``, ``B_own1``) and one cross-lag reduced-form coefficient (``y1.L1`` in the ``y0``
equation; ESS of the stored draws, which ``store_max = 2500`` thins by 4). The load average is
recorded before and after every run. The summary reports medians over repeats (ESS/s: over the
warm-cache runs, so compile time does not enter it; compile time is reported on its own).

Usage::

    uv run python benchmarks/phase2_perf.py --out bench.json            # 3 repeats, 1 device
    uv run python benchmarks/phase2_perf.py --devices 4 --out bench4.json  # stretch target
    uv run python benchmarks/phase2_perf.py --kernels --out bench.json  # + tests/bench/bench_kernels.py
    uv run python benchmarks/phase2_perf.py --quick                     # smoke run (tiny settings)

**Reference baseline** (DESIGN §10.1: the idle reference laptop; close other applications, plug
in power; about 15–30 minutes). One command measures BENCH-P2 (3 × cold + warm) and the kernel
microbenchmarks and replaces ``tests/bench/baseline.json``::

    uv run python benchmarks/phase2_perf.py --kernels --write-baseline --reference \\
        --label "Apple M1 Pro, idle, reference" --out bench-ref.json

``--write-baseline`` writes ``tests/bench/baseline.json`` (the pre-release gate's reference,
compared by ``tests/bench/test_perf_phase2.py``: +20% time or −30% ESS/s fails, §10.6);
``--reference`` marks it as the reference machine, where the test also asserts the 120 s target;
``--label`` is a free-text description stored with it. ``--from-json bench.json`` writes the
baseline from an earlier ``--out`` result instead of running again. Without ``--reference`` the
baseline is labelled provisional (a regression reference for the same machine class only).

This module imports neither JAX nor ``rai_bvar`` at import time (the parent process only
orchestrates and summarises; the gate logic is pure Python and unit-tested without JAX).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import platform
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
BASELINE = ROOT / "tests" / "bench" / "baseline.json"
KERNELS = ROOT / "tests" / "bench" / "bench_kernels.py"

# BENCH-P2 (DESIGN §10.1)
N, LAGS, T = 20, 4, 300
CHAINS, BURN, DRAWS = 4, 1000, 10_000
DATA_SEED = 0
TARGET_S = 120.0  # pass, one device
STRETCH_S = 60.0  # stretch, set_cpu_devices(4)
SWEEP_BLOCK, SWEEP_REPS = 1000, 3  # ms per sweep: minimum of 3 × 1000 sweeps

# pre-release gate (DESIGN §10.6)
TIME_TOL = 0.20  # +20% time fails
ESS_TOL = 0.30  # −30% ESS/s fails

ESS_GROUPS = ("loglik", "h_T", "mu", "sigma", "phi", "B_own1")
CROSS_LAG = ("y1.L1", "y0")  # (regressor, equation) of the reported cross-lag coefficient


# --------------------------------------------------------------------------- data and model


def _pilot_common() -> Any:
    """WP3's ``benchmarks/pilot/pilot_common.py`` (the committed DGP), imported by path."""
    name = "wp3_pilot_common"
    if name not in sys.modules:
        path = ROOT / "benchmarks" / "pilot" / "pilot_common.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


def bench_levels(n: int = N, p: int = LAGS, T: int = T, seed: int = DATA_SEED) -> Any:
    """The ``(T + p, n)`` series of ``pilot_common.simulate_var_sv(n, p, T, seed)``: its
    ``p`` pre-sample rows (recovered from the first design row) followed by ``Y``."""
    import numpy as np

    d = _pilot_common().simulate_var_sv(n, p, T, seed=seed)
    X = np.asarray(d.X)
    pre = [X[0, 1 + (lag - 1) * n : 1 + lag * n] for lag in range(p, 0, -1)]  # y_0 .. y_{p−1}
    y = np.vstack([*pre, np.asarray(d.Y)])
    if not np.array_equal(X[:, 1 : 1 + n], y[p - 1 : -1]):  # lag-1 block of the design
        raise AssertionError("simulate_var_sv layout changed: cannot recover the levels")
    return y


def bench_data(n: int = N, p: int = LAGS, T: int = T, seed: int = DATA_SEED) -> Any:
    """BENCH-P2 data as ``VARData`` (tcode 1, quarterly labels; ``T`` effective rows at ``p``)."""
    import pandas as pd

    import rai_bvar as rb

    y = bench_levels(n, p, T, seed)
    index = pd.period_range("1900Q1", periods=y.shape[0], freq="Q")
    frame = pd.DataFrame(y, index=index, columns=[f"y{i}" for i in range(n)])
    return rb.VARData.from_frame(frame, 1)


def bench_model(
    *, chains: int = CHAINS, burn: int = BURN, draws: int = DRAWS, lags: int = LAGS
) -> Any:
    """The BENCH-P2 composition (DESIGN §10.1)."""
    import rai_bvar as rb

    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="structural", hierarchy="fixed")),
        vol=rb.SV(),
        lags=lags,
        mcmc=rb.MCMCSettings(chains=chains, burn=burn, draws=draws, thin=1),
    )


# --------------------------------------------------------------------------- machine


def _cpuinfo() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        text = ""
    if text:
        lines = [ln.split(":", 1) for ln in text.splitlines() if ":" in ln]
        kv = [(k.strip(), v.strip()) for k, v in lines]
        first = dict(reversed(kv))  # first occurrence wins
        out["cpu"] = first.get("model name", "")
        flags = set(first.get("flags", "").split())
        out["simd"] = sorted(f for f in ("avx", "avx2", "fma", "avx512f") if f in flags)
        phys = {v for k, v in kv if k == "physical id"}
        cores = first.get("cpu cores")
        out["physical_cores"] = int(cores) * max(len(phys), 1) if cores else None
    else:  # macOS
        for key, name in (
            ("cpu", "machdep.cpu.brand_string"),
            ("physical_cores", "hw.physicalcpu"),
            ("perf_cores", "hw.perflevel0.physicalcpu"),
        ):
            try:
                out[key] = subprocess.run(
                    ["sysctl", "-n", name], capture_output=True, text=True, check=True
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                out[key] = ""
    return out


def _meminfo_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine_info(with_jax: bool = True) -> dict[str, Any]:
    """CPU model, cores, memory, OS, Python, NumPy/JAX/jaxlib versions, devices, load average."""
    import numpy as np

    info: dict[str, Any] = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        **_cpuinfo(),
        "logical_cpus": os.cpu_count(),
        "mem_bytes": _meminfo_bytes(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "loadavg": list(os.getloadavg()),
        "commit": _git_commit(),
    }
    if with_jax:
        import jax
        import jaxlib

        info.update(
            jax=jax.__version__,
            jaxlib=jaxlib.__version__,
            backend=jax.default_backend(),
            devices=jax.local_device_count(),
        )
    else:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in ("jax", "jaxlib"):
            try:
                info[pkg] = version(pkg)
            except PackageNotFoundError:
                info[pkg] = ""
    return info


def same_machine(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two fingerprints describe the same machine class and software (CPU model,
    logical CPUs, ``jax``/``jaxlib``): the only case where a timing comparison means anything."""
    keys = ("cpu", "logical_cpus", "jax", "jaxlib", "machine")
    return all(a.get(k) == b.get(k) for k in keys)


# --------------------------------------------------------------------------- one run (child)


def _ess_summary(fit: Any, wall: float) -> dict[str, Any]:
    """Bulk ESS (min, median) and ESS per second of the monitored groups and one cross-lag
    reduced-form coefficient."""
    import numpy as np

    from rai_bvar.diagnostics import ess_bulk

    diag = fit.diagnostics
    out: dict[str, Any] = {}
    for group in ESS_GROUPS:
        ess = diag.loc[diag["field"] == group, "ess_bulk"].to_numpy(dtype=float)
        if ess.size == 0:
            continue
        out[group] = {
            "count": int(ess.size),
            "ess_min": float(np.min(ess)),
            "ess_median": float(np.median(ess)),
            "ess_per_s_min": float(np.min(ess) / wall),
            "ess_per_s_median": float(np.median(ess) / wall),
        }
    B = fit.reduced_form_draws()
    reg, var = CROSS_LAG
    x = np.asarray(B.sel(regressor=reg, variable=var))  # (chain, stored draw)
    e = float(ess_bulk(x))
    out["cross_lag"] = {
        "quantity": f"B[{reg}, {var}]",
        "stored_draws_per_chain": int(x.shape[1]),
        "ess": e,
        "ess_per_s": e / wall,
    }
    primary = diag.loc[diag["primary"].astype(bool), "ess_bulk"].to_numpy(dtype=float)
    out["primary"] = {
        "count": int(primary.size),
        "ess_min": float(np.min(primary)),
        "ess_median": float(np.median(primary)),
        "ess_per_s_min": float(np.min(primary) / wall),
        "ess_per_s_median": float(np.median(primary) / wall),
        "rhat_max": float(np.nanmax(diag.loc[diag["primary"].astype(bool), "rhat"])),
    }
    return out


def sweep_ms(fit: Any, *, sweeps: int = SWEEP_BLOCK, reps: int = SWEEP_REPS) -> dict[str, Any]:
    """Minimum (and all) ms per sweep of ``reps`` × ``sweeps`` sweeps of the fit's compiled
    ``burn`` executable, started from the final state each time (DESIGN §10.1)."""
    import jax
    import numpy as np

    from rai_bvar.engines.mcmc import _blas, _jax, keys
    from rai_bvar.engines.mcmc.runner import _sharded, make_runner

    devices = int(fit.metadata.get("devices", 1))
    with _jax.x64(), _blas.single_thread():  # as in run_chains
        fns = make_runner(fit.static)
        data = jax.tree.map(jax.numpy.asarray, fit.engine_data)
        prior = jax.tree.map(jax.numpy.asarray, fit.prior)
        cache = fns.prepare(data, prior)
        C = int(fit.settings.chains)
        ckeys = keys.chain_keys(keys.fit_root(int(fit.seed), int(fit.salt)), C)
        burn = fns.burn
        place: Any = jax.device_put
        if devices > 1:
            burn, _, chain_sh, repl = _sharded(fns.chain, devices, fns.chains, 1)
            ckeys = jax.device_put(ckeys, chain_sh)
            data, prior, cache = (jax.device_put(t, repl) for t in (data, prior, cache))

            def place(x: Any, _s: Any = chain_sh) -> Any:
                return jax.device_put(x, _s)

        host = fit.final_state.state
        it0 = np.asarray(fit.metadata.get("it_last", 0) + 1, dtype=np.uint32)
        n = np.asarray(sweeps, dtype=np.int32)
        jax.block_until_ready(burn(ckeys, place(host), data, prior, cache, it0, np.int32(1)))
        times = []
        for _ in range(reps):
            st = jax.block_until_ready(place(host))
            t0 = time.perf_counter()
            jax.block_until_ready(burn(ckeys, st, data, prior, cache, it0, n))
            times.append(time.perf_counter() - t0)
    ms = [1000.0 * t / sweeps for t in times]
    return {"min": min(ms), "all": ms, "sweeps": sweeps, "reps": reps}


def _peak_rss_bytes() -> int:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    """One BENCH-P2 fit in this (fresh) process; returns the run record."""
    if args.devices > 1:
        import rai_bvar as rb

        rb.set_cpu_devices(args.devices)  # before JAX initialises (DESIGN §5.2)
    import rai_bvar as rb

    if args.cache_dir:
        rb.enable_compilation_cache(args.cache_dir, min_compile_secs=0.0)
    data = bench_data(args.n, args.lags, args.T)  # data loading: not timed (§10.1)
    model = bench_model(chains=args.chains, burn=args.burn, draws=args.draws, lags=args.lags)
    runtime = rb.Runtime(devices=args.devices)
    load_before = list(os.getloadavg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        fit = model.fit(data, runtime=runtime)
        wall = time.perf_counter() - t0
    load_after = list(os.getloadavg())
    timing = dict(fit.timing)
    compile_s = float(timing.get("compile", 0.0))
    record: dict[str, Any] = {
        "kind": args.kind,
        "wall_s": wall,
        "compile_s": compile_s,
        "run_s": wall - compile_s,
        "post_runner_s": wall - float(timing.get("total", wall)),  # fit assembly, diagnostics
        "runner_timing": timing,
        "runner_ms_per_sweep": float(timing.get("ms_per_sweep", float("nan"))),
        "converged": bool(fit.converged),
        "warnings": sorted({str(w.message).splitlines()[0][:160] for w in caught}),
        "chain_method": fit.metadata.get("chain_method"),
        "devices": int(fit.metadata.get("devices", 1)),
        "gram": fit.static.mean.gram,
        "shapes": str(fit.static.shapes),
        "loadavg_before": load_before,
        "loadavg_after": load_after,
    }
    t_ess = time.perf_counter()
    record["ess"] = _ess_summary(fit, wall)
    record["ess_s"] = time.perf_counter() - t_ess
    if args.sweep_reps > 0:
        record["sweep_ms"] = sweep_ms(fit, sweeps=args.sweep_block, reps=args.sweep_reps)
    record["peak_rss_bytes"] = _peak_rss_bytes()
    record["machine"] = machine_info()
    return record


# --------------------------------------------------------------------------- orchestration


def _median(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over repeats (cold and warm runs separately where the cache matters)."""
    cold = [r for r in runs if r["kind"] == "cold"]
    warm = [r for r in runs if r["kind"] == "warm"]
    out: dict[str, Any] = {
        "repeats": len(cold),
        "compile_cold_s": _median([r["compile_s"] for r in cold]),
        "compile_warm_s": _median([r["compile_s"] for r in warm]),
        "wall_cold_s": _median([r["wall_s"] for r in cold]),
        "wall_warm_s": _median([r["wall_s"] for r in warm]),
        "run_s": _median([r["run_s"] for r in runs]),
        "post_runner_s": _median([r["post_runner_s"] for r in runs]),
        "runner_ms_per_sweep": _median([r["runner_ms_per_sweep"] for r in runs]),
        "sweep_ms_min3x1000": _median([r["sweep_ms"]["min"] for r in runs if "sweep_ms" in r]),
        "peak_rss_mb": _median([r["peak_rss_bytes"] / 2**20 for r in runs]),
        "loadavg_1min": [r["loadavg_before"][0] for r in runs],
        "converged": [r["converged"] for r in runs],
    }
    ess: dict[str, Any] = {}
    steady = warm or cold  # ESS/s of the warm-cache runs (compile time is reported separately)
    for group in (*ESS_GROUPS, "primary"):
        rows = [r["ess"][group] for r in steady if group in r["ess"]]
        if rows:
            ess[group] = {
                "ess_min": _median([x["ess_min"] for x in rows]),
                "ess_per_s_min": _median([x["ess_per_s_min"] for x in rows]),
                "ess_per_s_median": _median([x["ess_per_s_median"] for x in rows]),
            }
    rows = [r["ess"]["cross_lag"] for r in steady if "cross_lag" in r["ess"]]
    if rows:
        ess["cross_lag"] = {
            "quantity": rows[0]["quantity"],
            "ess": _median([x["ess"] for x in rows]),
            "ess_per_s": _median([x["ess_per_s"] for x in rows]),
        }
    out["ess"] = ess
    wall = out["wall_cold_s"]
    out["pass_120s_cold"] = None if wall is None else bool(wall < TARGET_S)
    out["pass_120s_warm"] = (
        None if out["wall_warm_s"] is None else bool(out["wall_warm_s"] < TARGET_S)
    )
    return out


def _child_cmd(args: argparse.Namespace, kind: str, cache_dir: str, result: Path) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--child", "--kind", kind,
        "--cache-dir", cache_dir, "--result", str(result),
        "--devices", str(args.devices), "--n", str(args.n), "--lags", str(args.lags),
        "--T", str(args.T), "--chains", str(args.chains), "--burn", str(args.burn),
        "--draws", str(args.draws), "--sweep-block", str(args.sweep_block),
        "--sweep-reps", str(args.sweep_reps),
    ]  # fmt: skip
    return cmd


def run_child_process(args: argparse.Namespace, kind: str, cache_dir: str) -> dict[str, Any]:
    """Run one fit in a fresh interpreter and return its record."""
    with tempfile.TemporaryDirectory(prefix="bench-p2-") as tmp:
        result = Path(tmp) / "run.json"
        env = dict(os.environ)
        for var in ("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"):
            env.pop(var, None)  # the child enables exactly the cache it is given
        proc = subprocess.run(_child_cmd(args, kind, cache_dir, result), env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"BENCH-P2 child ({kind}) failed with exit code {proc.returncode}")
        return json.loads(result.read_text())


def run_kernels(args: argparse.Namespace) -> dict[str, Any]:
    """``tests/bench/bench_kernels.py`` in a fresh process (its own JSON)."""
    with tempfile.TemporaryDirectory(prefix="bench-kernels-") as tmp:
        result = Path(tmp) / "kernels.json"
        cmd = [sys.executable, str(KERNELS), "--out", str(result), "--reps", str(args.kernel_reps)]
        subprocess.run(cmd, check=True)
        return json.loads(result.read_text())


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """``args.repeats`` × (cold, warm) fits, each in a fresh process; summary and records."""
    runs: list[dict[str, Any]] = []
    for r in range(args.repeats):
        cache_dir = tempfile.mkdtemp(prefix="bench-p2-xla-")
        try:
            for kind in ("cold", "warm"):
                rec = run_child_process(args, kind, cache_dir)
                rec["repeat"] = r
                runs.append(rec)
                print(
                    f"[BENCH-P2] repeat {r} {kind}: wall {rec['wall_s']:.1f} s, compile "
                    f"{rec['compile_s']:.1f} s, run {rec['run_s']:.1f} s, sweep "
                    f"{rec.get('sweep_ms', {}).get('min', float('nan')):.2f} ms, load "
                    f"{rec['loadavg_before'][0]:.2f}",
                    flush=True,
                )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    out: dict[str, Any] = {
        "benchmark": "BENCH-P2",
        "design": "docs/phase2/DESIGN.md §10.1, §10.6",
        "settings": {
            "n": args.n,
            "lags": args.lags,
            "T": args.T,
            "chains": args.chains,
            "burn": args.burn,
            "draws": args.draws,
            "thin": 1,
            "devices": args.devices,
            "data_seed": DATA_SEED,
            "sweep_block": args.sweep_block,
            "sweep_reps": args.sweep_reps,
        },
        "target_s": TARGET_S if args.devices == 1 else STRETCH_S,
        "machine": machine_info(with_jax=False),
        "summary": summarise(runs),
        "runs": runs,
    }
    if args.kernels:
        out["kernels"] = run_kernels(args)
    return out


# --------------------------------------------------------------------------- gate (§10.6)


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Regressions of ``current`` against ``baseline`` (both ``summary`` dicts): time more than
    +20% (warm wall time, run time, ms per sweep) or ESS/s more than −30% (the minimum over the
    primary set, and the cross-lag coefficient). Returns the failure messages (empty: pass)."""
    fails: list[str] = []
    for key in ("wall_warm_s", "run_s", "sweep_ms_min3x1000"):
        cur, ref = current.get(key), baseline.get(key)
        if cur is None or ref is None:
            continue
        if cur > (1.0 + TIME_TOL) * ref:
            fails.append(f"{key}: {cur:.3f} > {1 + TIME_TOL:.2f} × baseline {ref:.3f}")
    cur_ess, ref_ess = current.get("ess", {}), baseline.get("ess", {})
    pairs = [("primary", "ess_per_s_min"), ("cross_lag", "ess_per_s")]
    for group, field in pairs:
        cur = cur_ess.get(group, {}).get(field)
        ref = ref_ess.get(group, {}).get(field)
        if cur is None or ref is None:
            continue
        if cur < (1.0 - ESS_TOL) * ref:
            fails.append(f"ESS/s {group}.{field}: {cur:.3f} < {1 - ESS_TOL:.2f} × {ref:.3f}")
    return fails


def compare_kernels(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Kernels whose minimum time exceeds the baseline's by more than +20%."""
    fails = []
    ref = baseline.get("kernels", {})
    for name, row in current.get("kernels", {}).items():
        if name in ref and row["ms_min"] > (1.0 + TIME_TOL) * ref[name]["ms_min"]:
            fails.append(
                f"kernel {name}: {row['ms_min']:.3f} ms > {1 + TIME_TOL:.2f} × "
                f"baseline {ref[name]['ms_min']:.3f} ms"
            )
    return fails


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="write the full JSON result here")
    ap.add_argument("--repeats", type=int, default=3, help="cold + warm pairs (median reported)")
    ap.add_argument("--devices", type=int, default=1, help="CPU devices (4: stretch target)")
    ap.add_argument("--kernels", action="store_true", help="also run tests/bench/bench_kernels.py")
    ap.add_argument("--kernel-reps", type=int, default=50)
    ap.add_argument("--write-baseline", action="store_true", help=f"write {BASELINE}")
    ap.add_argument("--reference", action="store_true", help="mark the baseline as the idle "
                    "reference machine (the 120 s target is asserted there)")  # fmt: skip
    ap.add_argument("--label", default="", help="free-text description stored in the baseline")
    ap.add_argument("--from-json", type=Path, help="write the baseline from an earlier --out file")
    ap.add_argument("--quick", action="store_true", help="smoke run: n=3, p=2, T=100, 200 draws")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--lags", type=int, default=LAGS)
    ap.add_argument("--T", type=int, default=T)
    ap.add_argument("--chains", type=int, default=CHAINS)
    ap.add_argument("--burn", type=int, default=BURN)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--sweep-block", type=int, default=SWEEP_BLOCK)
    ap.add_argument("--sweep-reps", type=int, default=SWEEP_REPS)
    # child-process arguments
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--kind", default="cold", help=argparse.SUPPRESS)
    ap.add_argument("--cache-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    return ap


def make_baseline(result: dict[str, Any], *, reference: bool, label: str = "") -> dict[str, Any]:
    """The ``tests/bench/baseline.json`` document of a full ``run_benchmark`` result."""
    s = result["settings"]
    full = (s["n"], s["lags"], s["T"], s["chains"], s["burn"], s["draws"], s["devices"]) == (
        N,
        LAGS,
        T,
        CHAINS,
        BURN,
        DRAWS,
        1,
    )
    if not full:
        raise SystemExit("a baseline needs the full BENCH-P2 settings on one device (not --quick)")
    if reference:
        note = "Reference-machine baseline (DESIGN §10.1): the 120 s target is asserted here."
    else:
        note = (
            "PROVISIONAL, not the DESIGN §10.1 reference machine (an idle Apple M-series or x86 "
            "laptop): a regression reference for the same machine class only; the 120 s target "
            "is not asserted against it. Replace it with the reference command in "
            "benchmarks/phase2_perf.py's docstring."
        )
    return {
        "benchmark": "BENCH-P2",
        "reference": bool(reference),
        "label": label,
        "note": note,
        "tolerance": {"time": TIME_TOL, "ess_per_s": ESS_TOL},
        "settings": s,
        "machine": result["machine"],
        "summary": result["summary"],
        "kernels": result.get("kernels", {}).get("kernels", {}),
        "kernel_machine": result.get("kernels", {}).get("machine", {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.quick:
        args.n, args.lags, args.T = 3, 2, 100
        args.burn, args.draws, args.sweep_block = 100, 200, 100
        args.repeats = min(args.repeats, 1)
    if args.child:
        record = run_child(args)
        args.result.write_text(json.dumps(record, indent=1, default=float))
        return 0
    if args.from_json:
        result = json.loads(args.from_json.read_text())
    else:
        result = run_benchmark(args)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["summary"], indent=1))
    if args.write_baseline:
        baseline = make_baseline(result, reference=args.reference, label=args.label)
        BASELINE.write_text(json.dumps(baseline, indent=1, default=float) + "\n")
        print(f"wrote {BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
