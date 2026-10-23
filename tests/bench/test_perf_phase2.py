"""BENCH-P2 pre-release gate (DESIGN §9.10 pre-release tier, §10.1, §10.6; WP24).

The ``bench`` tests run the full benchmark (``benchmarks/phase2_perf.py``: 3 × cold + warm
fits of 4 × 11,000 sweeps, each in a fresh process; about 10–20 minutes) and the per-kernel
microbenchmarks (``tests/bench/bench_kernels.py``). They are also marked ``slow``, and CI
deselects ``bench`` in every scheduled tier, so they never run on a PR or in the weekly job;
run them before a release with::

    uv run pytest tests/bench -m bench -s

Gate (§10.6): against ``tests/bench/baseline.json``, more than +20% time (warm wall time, run
time, ms per sweep, any kernel) or −30% ESS/s (minimum over the primary set, the cross-lag
coefficient) fails, but only when the machine matches the baseline's fingerprint (CPU model,
logical CPUs, ``jax``/``jaxlib``); on another machine the comparison is meaningless and is
skipped with the numbers printed. The 120 s target (one device, end-to-end with a cold
compilation cache) is asserted only on the reference machine: when the baseline was recorded
there (``"reference": true``) and the fingerprint matches, or when ``RAI_BVAR_BENCH_REFERENCE=1``
says this machine is the idle reference laptop. ``RAI_BVAR_BENCH_REPEATS`` overrides the number
of repeats (default 3).

The unmarked tests check the committed baseline and the gate logic; they are pure Python (no
JAX, no fit) and take milliseconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "tests" / "bench" / "baseline.json"


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


def perf():
    return _load("rai_bvar_bench_phase2_perf", ROOT / "benchmarks" / "phase2_perf.py")


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


def _reference(baseline, machine) -> bool:
    if os.environ.get("RAI_BVAR_BENCH_REFERENCE") == "1":
        return True
    return bool(baseline.get("reference")) and perf().same_machine(machine, baseline["machine"])


# --------------------------------------------------------------------------- fast: baseline + gate


def test_baseline_is_complete(baseline):
    """The committed baseline records the BENCH-P2 settings, the machine and every gated number."""
    p = perf()
    assert baseline["benchmark"] == "BENCH-P2"
    s = baseline["settings"]
    assert (s["n"], s["lags"], s["T"], s["chains"], s["burn"], s["draws"], s["thin"]) == (
        p.N, p.LAGS, p.T, p.CHAINS, p.BURN, p.DRAWS, 1
    )  # fmt: skip
    assert s["devices"] == 1
    m = baseline["machine"]
    for key in ("cpu", "logical_cpus", "platform", "python", "jax", "jaxlib", "loadavg"):
        assert m.get(key) not in (None, ""), key
    summ = baseline["summary"]
    assert summ["repeats"] >= 3
    for key in ("compile_cold_s", "compile_warm_s", "wall_cold_s", "wall_warm_s", "run_s",
                "runner_ms_per_sweep", "sweep_ms_min3x1000", "peak_rss_mb"):  # fmt: skip
        assert summ[key] > 0, key
    for group in ("loglik", "h_T", "mu", "sigma", "primary"):
        assert summ["ess"][group]["ess_per_s_min"] > 0, group
    assert summ["ess"]["cross_lag"]["ess_per_s"] > 0
    assert baseline["tolerance"] == {"time": p.TIME_TOL, "ess_per_s": p.ESS_TOL}
    for name in ("gram_packed", "precision_draw", "sv_indicators", "sv_awol", "sv_exact",
                 "sv_core", "sweep"):  # fmt: skip
        assert baseline["kernels"][name]["ms_min"] > 0, name


def test_make_baseline_labels_and_settings(baseline):
    """``--write-baseline`` output: provisional unless ``--reference``; full settings only."""
    p = perf()
    result = {k: baseline[k] for k in ("settings", "machine", "summary")}
    doc = p.make_baseline(result, reference=False, label="x")
    assert doc["reference"] is False and "PROVISIONAL" in doc["note"] and doc["label"] == "x"
    assert p.make_baseline(result, reference=True)["reference"] is True
    quick = {**result, "settings": {**result["settings"], "draws": 200}}
    with pytest.raises(SystemExit):
        p.make_baseline(quick, reference=False)
    if not baseline["reference"]:
        assert "PROVISIONAL" in baseline["note"] and baseline["label"]


def test_gate_flags_time_and_ess_regressions(baseline):
    """+20% time or −30% ESS/s fails; anything inside the tolerance passes (§10.6)."""
    p = perf()
    ref = baseline["summary"]
    assert p.compare(ref, ref) == []
    slow = json.loads(json.dumps(ref))
    slow["run_s"] = ref["run_s"] * 1.25
    assert any("run_s" in f for f in p.compare(slow, ref))
    ok = json.loads(json.dumps(ref))
    ok["run_s"] = ref["run_s"] * 1.15
    ok["ess"]["primary"]["ess_per_s_min"] *= 0.75
    assert p.compare(ok, ref) == []
    worse = json.loads(json.dumps(ref))
    worse["ess"]["cross_lag"]["ess_per_s"] *= 0.6
    assert any("cross_lag" in f for f in p.compare(worse, ref))
    kern = {"kernels": {"sweep": {"ms_min": baseline["kernels"]["sweep"]["ms_min"] * 1.3}}}
    assert p.compare_kernels(kern, baseline) and not p.compare_kernels(baseline, baseline)


def test_same_machine_needs_cpu_cores_and_versions(baseline):
    p = perf()
    m = dict(baseline["machine"])
    assert p.same_machine(m, baseline["machine"])
    for key, value in (("cpu", "other"), ("logical_cpus", 999), ("jaxlib", "0.0")):
        assert not p.same_machine({**m, key: value}, baseline["machine"]), key


# --------------------------------------------------------------------------- pre-release (bench)


def _args(**kw) -> argparse.Namespace:
    p = perf()
    args = p._parser().parse_args([])
    args.repeats = int(os.environ.get("RAI_BVAR_BENCH_REPEATS", "3"))
    for k, v in kw.items():
        setattr(args, k, v)
    return args


@pytest.mark.bench
@pytest.mark.slow
def test_bench_p2(baseline):
    """BENCH-P2 on one device: report, compare with the baseline on the same machine, and assert
    the 120 s target on the reference machine (DESIGN §10.1, §13 WP24)."""
    p = perf()
    result = p.run_benchmark(_args())
    s = result["summary"]
    print(json.dumps({"machine": result["machine"], "summary": s}, indent=1))
    # ``converged`` is reported, not gated: at 4 × 10,000 a few SV σ_i / φ_i miss R-hat < 1.01
    # on the reference data (WP24 measurement; the WP3 pilot's decision 6 escalation)
    if _reference(baseline, result["machine"]):
        assert s["wall_cold_s"] < p.TARGET_S, f"BENCH-P2 {s['wall_cold_s']:.1f} s ≥ 120 s"
    if not p.same_machine(result["machine"], baseline["machine"]):
        pytest.skip("not the baseline machine: numbers printed, no regression comparison")
    fails = p.compare(s, baseline["summary"])
    assert not fails, "; ".join(fails)


@pytest.mark.bench
@pytest.mark.slow
def test_kernel_microbenchmarks(baseline):
    """Per-kernel timings (§10.3, §10.6) against the baseline on the same machine."""
    kernels = _load("rai_bvar_bench_kernels", ROOT / "tests" / "bench" / "bench_kernels.py")
    result = kernels.run_kernels()
    print(kernels.table(result))
    if not perf().same_machine(result["machine"], baseline.get("kernel_machine", {})):
        pytest.skip("not the baseline machine: kernel timings printed, no comparison")
    fails = perf().compare_kernels(result, baseline)
    assert not fails, "; ".join(fails)
