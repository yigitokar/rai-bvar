"""Phase 4 week-1 spike: per-sweep cost of the prototype BART × SV sweep (``docs/phase4/SPIKE.md``).

For each ``(n, T)`` it times the production sweep with the BART mean (``compose`` with the
``mean`` override, ``make_sweep``) and, for comparison, the production ``Linear × SV`` sweep
(``structural_sv``) at the same ``n``, ``p`` and ``T``; plus the bartz ``step`` alone (one
equation, data swap included) and the adapter alone. Times are the warm median and minimum of
``reps`` timings of a jitted ``lax.scan`` of ``K`` sweeps, divided by ``K`` (on a loaded machine the
minimum is the better estimate of the unloaded cost); compile time is reported separately.
``--chains 4`` times four chains under ``vmap`` (the time of one sweep of all four);
``--batching map`` runs the equations one after another (``Batching.MAP``, ``lax.map``) instead
of under ``vmap``.

    uv run --extra bart python benchmarks/phase4_bart_spike_cost.py --out bart_cost.json
    uv run --extra bart python benchmarks/phase4_bart_spike_cost.py --x64   # path (b), patched
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import platform
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
from jax import lax, random

from rai_bvar.components.mean import _bart_spike as bs
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.protocols import Batching
from rai_bvar.engines.mcmc.sweep import make_sweep
from rai_bvar.validation._bart_spike import bart_joint, bartz_x64_patch
from rai_bvar.validation.joint import joint_spec, production_sweep

CPU: list[float] = []  # process CPU seconds of the last timed call set (see _time)


def _time(fn, *args, reps: int) -> tuple[float, float, float]:
    """Compile time, then the median and minimum wall time of ``reps`` calls. The median process
    CPU time of the calls goes to :data:`CPU`: on a machine shared with other jobs it is the
    load-robust measure of the work (an upper bound on single-threaded wall time)."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    out, cpu = [], []
    for _ in range(reps):
        t0, c0 = time.perf_counter(), time.process_time()
        jax.block_until_ready(fn(*args))
        out.append(time.perf_counter() - t0)
        cpu.append(time.process_time() - c0)
    CPU[:] = [statistics.median(cpu)]
    return compile_s, statistics.median(out), min(out)


def _scan_sweeps(step, K: int, chains: int):
    def run(key, st, data):
        def body(s, it):
            return step(key, it, s, data), None

        return lax.scan(body, st, jnp.arange(K, dtype=jnp.uint32))[0]

    if chains == 1:
        return jax.jit(run)
    return jax.jit(jax.vmap(run, in_axes=(0, 0, None)))


@dataclasses.dataclass(frozen=True)
class _MapMean(bs.BartSpikeMean):
    """The prototype kernel with ``Batching.MAP``: equations under ``lax.map``."""

    batching = Batching.MAP


def _map_step(J):
    program = compose(J.static, overrides={"mean": _MapMean.for_static(J.static, J.bart)})
    sweep = make_sweep(program)

    def step(key, it, st, data):
        return sweep(key, it, st, data, J.base.prior, J.prepare(data))

    return step


def bart_case(
    n: int,
    T: int,
    lags: int,
    m: int,
    depth: int,
    K: int,
    reps: int,
    chains: int,
    x64: bool,
    batching: str = "vmap",
):
    J = bart_joint(
        f"bart_cost_{n}_{T}", bart=bs.BartPrior(num_trees=m, max_depth=depth, x64=x64), n=n,
        T_eff=T, lags=lags,
    )  # fmt: skip
    key = random.key(0)
    st0 = jax.jit(J.prior_sample)(key)
    data = jax.jit(J.simulate)(key, st0)
    burn = _scan_sweeps(J.step, 50, 1)
    st = burn(key, st0, data)  # grow the trees towards posterior sizes
    if chains > 1:
        st = jax.tree.map(lambda a: jnp.stack([a] * chains), st)
        keys = random.split(key, chains)
    else:
        keys = key
    step = J.step if batching == "vmap" else _map_step(J)
    c, t, t_min = _time(_scan_sweeps(step, K, chains), keys, st, data, reps=reps)
    cpu = CPU[0]
    # one equation's bartz step alone (data swap + step), and the swap alone
    ms0 = jax.tree.map(
        lambda a: a[0], st.eq.mean if chains == 1 else jax.tree.map(lambda b: b[0], st.eq.mean)
    )
    cache = J.prepare(data)
    X = cache.eq.Xbin[0]
    target = data.Y[:, 0]
    w = data.tmask

    def bartz_only(k, s):
        def body(s, i):
            return bs.bartz_step(random.fold_in(k, i), bs.set_data(s, X, target, w), x64), None

        return lax.scan(body, s, jnp.arange(K))[0]

    def swap_only(k, s):
        def body(s, i):
            s = bs.set_data(s, X, target + 1e-9 * i, w)
            return s, None

        return lax.scan(body, s, jnp.arange(K))[0]

    _, t_step, t_step_min = _time(jax.jit(bartz_only), key, ms0, reps=reps)
    cpu_step = CPU[0]
    _, t_swap, t_swap_min = _time(jax.jit(swap_only), key, ms0, reps=reps)
    cpu_swap = CPU[0]
    return {"compile_s": c, "sweep_ms": 1e3 * t / K, "sweep_ms_min": 1e3 * t_min / K,
            "sweep_cpu_ms": 1e3 * cpu / K,
            "bartz_eq_ms": 1e3 * t_step / K, "bartz_eq_ms_min": 1e3 * t_step_min / K,
            "bartz_eq_cpu_ms": 1e3 * cpu_step / K,
            "swap_eq_ms": 1e3 * t_swap / K, "swap_eq_ms_min": 1e3 * t_swap_min / K,
            "swap_eq_cpu_ms": 1e3 * cpu_swap / K}  # fmt: skip


def linear_case(n: int, T: int, lags: int, K: int, reps: int, chains: int):
    spec = joint_spec("structural_sv", "small", n=n, lags=lags, T_eff=T)
    prepare, sweep = production_sweep(spec.static, spec.overrides)
    key = random.key(0)
    st0 = jax.jit(spec.prior_sample)(key)
    data = jax.jit(spec.simulate)(key, st0)
    cache = jax.jit(prepare)(data, spec.prior)

    def step(k, it, s, d):
        return sweep(k, it, s, d, spec.prior, cache)

    st = st0
    if chains > 1:
        st = jax.tree.map(lambda a: jnp.stack([a] * chains), st)
        keys = random.split(key, chains)
    else:
        keys = key
    c, t, t_min = _time(_scan_sweeps(step, K, chains), keys, st, data, reps=reps)
    return {"compile_s": c, "sweep_ms": 1e3 * t / K, "sweep_ms_min": 1e3 * t_min / K,
            "sweep_cpu_ms": 1e3 * CPU[0] / K}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="3,5,7")
    ap.add_argument("--Ts", default="200,400")
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--x64", action="store_true", help="path (b): bartz under x64, patched")
    ap.add_argument("--batching", choices=("vmap", "map"), default="vmap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = {
        "machine": platform.machine(), "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(), "jax": jax.__version__, "bartz": __import__("bartz").__version__,
        "load_before": os.getloadavg(), "args": vars(a),
    }  # fmt: skip
    rows = []
    ctx = bartz_x64_patch() if a.x64 else contextlib.nullcontext()
    with ctx:
        for n in map(int, a.ns.split(",")):
            for T in map(int, a.Ts.split(",")):
                b = bart_case(
                    n, T, a.lags, a.trees, a.depth, a.K, a.reps, a.chains, a.x64, a.batching
                )
                lin = linear_case(n, T, a.lags, a.K, a.reps, a.chains)
                row = {"n": n, "T": T, "bart": b, "linear_sv": lin, "ratio": b["sweep_ms"] / lin["sweep_ms"],
                       "ratio_min": b["sweep_ms_min"] / lin["sweep_ms_min"],
                       "ratio_cpu": b["sweep_cpu_ms"] / lin["sweep_cpu_ms"],
                       "load": os.getloadavg()}  # fmt: skip
                rows.append(row)
                print(json.dumps(row), flush=True)
    meta["load_after"] = os.getloadavg()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
