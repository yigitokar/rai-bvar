"""Per-kernel microbenchmarks of the BENCH-P2 sweep (DESIGN §10.3, §10.6; WP24).

The dominant kernels of a Linear × SV sweep at the BENCH-P2 shape (``n = 20``, ``p = 4``,
``k = 81``, ``q = 100``, ``P = 5050``, ``T_pad = 320``, 4 chains: ``B = 80`` equation-chain lanes),
each the production function from ``src/rai_bvar`` vmapped over the 80 lanes, AOT-compiled
(``jax.jit(f).lower(*args).compile()``, compile time separated), timed with
``block_until_ready`` (minimum and median over ``--reps`` calls; each call includes JAX's
dispatch overhead of a few tens of µs), with XLA's ``cost_analysis()`` (flops, bytes accessed)
and ``memory_analysis()`` (argument, output and temporary bytes):

| kernel | production code | §10.3 budget line (ms) |
| :--- | :--- | ---: |
| ``gram_packed`` | ``linalg.weighted_gram(mode="packed")``: ``(w @ FF)[pos]`` (D-25) | 1.1 |
| ``gram_batched`` | ``weighted_gram(mode="batched")`` (the oracle; for the D-25 ratio) | — |
| ``precision_draw`` | ``linalg.precision_draw``: Jacobi-scaled Cholesky + two solves (D-14) | 1.9 (with GEMMs) |
| ``sv_indicators`` | ``vol.sv.draw_indicators`` (inverse-CDF categorical, D-4) | } 1.5 |
| ``sv_awol`` | ``awol_system`` + ``linalg.tridiag_precision_sample`` | } |
| ``sv_exact`` | the exact correction of the ``h`` move: one extra ``log g`` + two ``w`` | 0.4 (pilot 1.1) |
| ``sv_core`` | ``SVCore.sample``: the whole SV update (indicators, AWOL, exact ``h``, C and NC steps) | ≈ 3.6 |
| ``sweep`` | the runner's compiled ``burn`` executable, per sweep (all 80 lanes) | 5.6 |

Inputs come from the real BENCH-P2 resolution (``engines.mcmc.resolve``) of the committed data
and the runner's cold initial state, so shapes, the resolved ``gram`` and the static
configuration are exactly those of the benchmark fit. Run as a script (``--out`` writes JSON)
or call :func:`run_kernels`; ``benchmarks/phase2_perf.py --kernels`` runs it in a fresh process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

BUDGET_MS = {
    "gram_packed": 1.1,
    "precision_draw": 1.9,
    "sv_indicators+sv_awol": 1.5,
    "sv_exact": 0.4,
    "sweep": 5.6,
}


def _phase2_perf() -> Any:
    name = "rai_bvar_bench_phase2_perf"
    if name not in sys.modules:
        path = ROOT / "benchmarks" / "phase2_perf.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


def _cost(compiled: Any) -> dict[str, float]:
    try:
        ca = compiled.cost_analysis()
    except Exception:  # backends without cost analysis
        return {}
    if isinstance(ca, list | tuple):
        ca = ca[0] if ca else {}
    ca = dict(ca or {})
    return {
        k: float(ca[k]) for k in ("flops", "transcendentals", "bytes accessed") if k in ca
    }  # fmt: skip


def _memory(compiled: Any) -> dict[str, int]:
    try:
        ma = compiled.memory_analysis()
    except Exception:
        return {}
    if ma is None:
        return {}
    keys = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    return {k: int(getattr(ma, k)) for k in keys if hasattr(ma, k)}


def bench(
    fn: Callable[..., Any], *args: Any, reps: int, per: int = 1, donate: bool = False
) -> dict[str, Any]:
    """AOT-compile ``fn(*args)``, then time ``reps`` calls (``per`` units of work per call)."""
    import jax

    t0 = time.perf_counter()
    jitted = jax.jit(fn, donate_argnums=(1,) if donate else ())
    compiled = jitted.lower(*args).compile()
    compile_s = time.perf_counter() - t0
    times = []
    for i in range(reps + 1):  # the first call warms up and is not timed
        call_args = args
        if donate:  # a fresh copy of the donated state each call
            call_args = (args[0], jax.tree.map(lambda x: x.copy(), args[1]), *args[2:])
            jax.block_until_ready(call_args)
        if i == 0:
            jax.block_until_ready(compiled(*call_args))
            continue
        t = time.perf_counter()
        jax.block_until_ready(compiled(*call_args))
        times.append(time.perf_counter() - t)
    ms = sorted(1000.0 * t / per for t in times)
    return {
        "ms_min": ms[0],
        "ms_median": ms[len(ms) // 2],
        "reps": reps,
        "per_call_units": per,
        "compile_s": compile_s,
        "cost": _cost(compiled),
        "memory": _memory(compiled),
    }


def kernel_inputs() -> dict[str, Any]:
    """Static config, data, prior, cache, cold initial state and keys of BENCH-P2 (4 chains)."""
    import jax
    import jax.numpy as jnp

    import rai_bvar as rb
    from rai_bvar.engines.mcmc import _jax, keys
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.runner import make_runner

    perf = _phase2_perf()
    model = perf.bench_model()
    with _jax.x64():
        res = resolve(model, perf.bench_data(), rb.Runtime())
        fns = make_runner(res.static)
        data = jax.tree.map(jnp.asarray, res.data)
        prior = jax.tree.map(jnp.asarray, res.prior)
        cache = fns.prepare(data, prior)
        C = model.mcmc.chains
        seed = model.mcmc.seed
        state = fns.init(keys.init_keys(seed, res.salt, C), data, prior, cache)
        ckeys = keys.chain_keys(keys.fit_root(seed, res.salt), C)
    return {
        "static": res.static, "fns": fns, "data": data, "prior": prior, "cache": cache,
        "state": state, "ckeys": ckeys, "C": C,
    }  # fmt: skip


def run_kernels(reps: int = 50, sweeps: int = 200) -> dict[str, Any]:
    """Time every kernel of the module table; returns ``{"kernels": {...}, "machine": {...}}``."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from rai_bvar.components.vol.mixture import log_g
    from rai_bvar.components.vol.sv import (
        SVCore,
        awol_system,
        draw_indicators,
        w_weight,
        y_star,
    )
    from rai_bvar.engines.mcmc import _blas, _jax
    from rai_bvar.engines.mcmc.linalg import precision_draw, tridiag_precision_sample, weighted_gram

    out: dict[str, Any] = {}
    with _jax.x64(), _blas.single_thread():  # the runner's BLAS setting (engines.mcmc._blas)
        inp = kernel_inputs()
        static, data, prior, cache, state = (
            inp["static"], inp["data"], inp["prior"], inp["cache"], inp["state"]
        )  # fmt: skip
        C, n, q, T_pad = inp["C"], static.shapes.n, static.shapes.q, static.shapes.T_pad
        B = C * n
        rng = np.random.default_rng(0)
        core = state.eq.vol.core  # SVEq, leaves (C, n, …)
        lanes = lambda x: jnp.reshape(x, (B, *x.shape[2:]))  # noqa: E731
        h, mu, phi, sigma = (lanes(x) for x in (core.h, core.mu, core.phi, core.sigma))
        tmask, T_eff = data.tmask, data.T_eff
        w = jnp.exp(-h[:, 1:]) * tmask

        # mean block: Gram and precision draw (D-25, D-14)
        FF = cache.FF

        def gram_packed(w):
            return jax.vmap(lambda wi: weighted_gram(wi, FF=FF, mode="packed", q=q))(w)

        F = cache.F

        def gram_batched(w):
            return jax.vmap(lambda wi: weighted_gram(wi, F=F, mode="batched", q=q))(w)

        out["gram_packed"] = bench(gram_packed, w, reps=reps)
        out["gram_batched"] = bench(gram_batched, w, reps=reps)
        G = jax.jit(gram_packed)(w)
        K = G + jnp.eye(q)
        b = jnp.asarray(rng.normal(size=(B, q)))
        z = jnp.asarray(rng.normal(size=(B, q)))
        out["precision_draw"] = bench(
            lambda K, b, z: jax.vmap(precision_draw)(K, b, z), K, b, z, reps=reps
        )

        # SV kernels on realistic residuals: log ẽ² = h + log χ²₁ on real rows, 0 on padding
        sv = SVCore.from_static(static)
        table = sv.table()
        log_e2c = jnp.where(
            tmask > 0, h[:, 1:] + jnp.log(jnp.asarray(rng.chisquare(1, size=(B, T_pad)))), 0.0
        )
        sp = jax.tree.map(lambda x: jnp.tile(x, (C, *([1] * (x.ndim - 1)))), prior.eq.vol_core)
        obs = jnp.broadcast_to(tmask, (B, T_pad))
        ys = jax.vmap(y_star)(log_e2c, sp.log_c, obs)
        u = jnp.asarray(rng.uniform(size=(B, T_pad)))

        def indicators(u, ys, h):
            return jax.vmap(lambda u_, y_, h_: draw_indicators(u_, y_ - h_[1:], table))(u, ys, h)

        out["sv_indicators"] = bench(indicators, u, ys, h, reps=reps)
        s, lg_cur = jax.jit(indicators)(u, ys, h)
        zh = jnp.asarray(rng.normal(size=(B, T_pad + 1)))

        def awol(ys, s, mu, phi, sigma, zh):
            def one(y_, s_, o_, m_, p_, g_, sp_, z_):
                diag, off, bb = awol_system(y_, s_, o_, T_eff, m_, p_, g_, sp_, table)
                return tridiag_precision_sample(diag, off, bb, z_)[0]

            return jax.vmap(one)(ys, s, obs, mu, phi, sigma, sp, zh)

        out["sv_awol"] = bench(awol, ys, s, mu, phi, sigma, zh, reps=reps)
        x = jax.jit(awol)(ys, s, mu, phi, sigma, zh)
        h_prop = mu[:, None] + sigma[:, None] * x

        def exact(h, h_prop, lg_cur):
            def one(h_, hp_, lg_, le_, y_, o_):
                lg_prop = log_g(y_ - jnp.where(o_ > 0, hp_[1:], 0.0), table)
                return w_weight(hp_, le_, y_, o_, table, lg_prop) - w_weight(
                    h_, le_, y_, o_, table, lg_
                )

            return jax.vmap(one)(h, h_prop, lg_cur, log_e2c, ys, obs)

        out["sv_exact"] = bench(exact, h, h_prop, lg_cur, reps=reps)

        sv_keys = jax.vmap(lambda i: jax.random.fold_in(inp["ckeys"][0], i))(
            jnp.arange(B, dtype=jnp.uint32)
        )
        cs = jax.tree.map(lanes, core)

        def sv_core(keys_, cs, log_e2c):
            return jax.vmap(lambda k_, c_, l_, p_: sv.sample(k_, c_, l_, tmask, T_eff, p_)[0])(
                keys_, cs, log_e2c, sp
            )

        out["sv_core"] = bench(sv_core, sv_keys, cs, log_e2c, reps=reps)

        # the whole sweep: the runner's compiled burn executable, per sweep
        fns = inp["fns"]

        def burn(ckeys, st):
            return fns.burn(
                ckeys, st, data, prior, cache, jnp.uint32(0), jnp.int32(sweeps)
            )  # fmt: skip

        out["sweep"] = bench(
            burn, inp["ckeys"], state, reps=max(3, reps // 10), per=sweeps, donate=True
        )
    perf = _phase2_perf()
    budget = dict(BUDGET_MS)
    sv_pair = out["sv_indicators"]["ms_min"] + out["sv_awol"]["ms_min"]
    return {
        "shape": {"lanes": B, "n": n, "q": q, "P": q * (q + 1) // 2, "T_pad": T_pad},
        "kernels": out,
        "derived": {
            "gram_batched_over_packed": out["gram_batched"]["ms_min"]
            / out["gram_packed"]["ms_min"],
            "sv_indicators+sv_awol": sv_pair,
        },
        "budget_ms": budget,
        "machine": perf.machine_info(),
    }


def table(result: dict[str, Any]) -> str:
    """A Markdown table of the kernel timings."""
    rows = ["| kernel | ms min | ms median | GFLOP | MB accessed | temp MB | compile s |"]
    rows.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, r in result["kernels"].items():
        c, m = r.get("cost", {}), r.get("memory", {})
        per = r.get("per_call_units", 1)
        rows.append(
            f"| {name} | {r['ms_min']:.3f} | {r['ms_median']:.3f} | "
            f"{c.get('flops', float('nan')) / 1e9 / per:.3f} | "
            f"{c.get('bytes accessed', float('nan')) / 2**20 / per:.1f} | "
            f"{m.get('temp_size_in_bytes', 0) / 2**20:.1f} | {r['compile_s']:.2f} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BENCH-P2 per-kernel microbenchmarks")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--sweeps", type=int, default=200)
    args = ap.parse_args(argv)
    result = run_kernels(reps=args.reps, sweeps=args.sweeps)
    print(table(result))
    print(json.dumps(result["derived"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
