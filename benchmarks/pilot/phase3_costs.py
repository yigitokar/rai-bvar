"""Phase 3 cost measurements (P3 DESIGN §10; WP36): the production steps and sweeps, timed on a
burned-in state, 4 chains vmapped, one CPU device.

Per composition (``phase3_common.build_model``) and size the script resolves the model, builds
the runner, initialises and burns in ``--warm`` sweeps with the runner's ``burn`` executable (the
timed state is a burned-in one: GIG and Newton loop counts depend on the state), and then times,
each AOT-compiled and called with ``block_until_ready`` (minimum and median over ``--reps``
calls; every call gets a new sweep index, so the random draws, and with them the rejection-loop
counts, change from call to call):

* ``step:<name>``: every global step of the program alone (the GL scale step ``G_COEF_GLOBAL``,
  the CommonSV step ``G_COMMON_SV``, the reduced-form mean, …) on the state and the equation
  phase's output (``SweepCtx.eq_out``), 4 chains (``vmap``), one step per call;
* ``equation_phase``: the fused equation phase (``sweep.equation_phase``) alone;
* ``sweep_standalone``: one production sweep (``make_sweep(program)``) per call (``jit(vmap)``);
* ``sweep_scan``: ``--block`` sweeps per call in a ``lax.scan`` (ms per sweep);
* ``sweep_burn``: the runner's ``burn`` executable (``fori_loop`` over a traced count, donated
  state), ``--block`` sweeps per call (ms per sweep; BENCH-P2's ``sweep_ms`` definition).

Every block records the load average and the top CPU processes before and after it; with
``--gate L`` (default 2) the script first waits (up to ``--wait`` s) for the 1-minute load
average to fall below ``L`` and flags the block ``contaminated`` when the load before or after
exceeds ``L``. Results: ``DIR/cost_<size>_<config>.json``.

Usage::

    python -I benchmarks/pilot/phase3_costs.py --out DIR --size bp2 --configs hs_const,csv_conj
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import phase3_common as pc  # noqa: E402


def _time(compiled, make_args, reps: int, per: int = 1) -> dict:
    times = []
    for r in range(reps + 1):
        args = make_args(r)
        import jax

        jax.block_until_ready(args)
        t = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        if r:  # the first call warms up
            times.append(1000.0 * (time.perf_counter() - t) / per)
    times.sort()
    return {"ms_min": times[0], "ms_median": times[len(times) // 2], "reps": reps, "all": times}


def measure(config: str, size: str, *, warm: int, reps: int, block: int, gate: float,
            wait: float, const_path: str = "auto") -> dict:  # fmt: skip
    import jax
    import jax.numpy as jnp

    import rai_bvar as rb
    from rai_bvar.engines.mcmc import _blas, _jax, keys
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.runner import make_runner
    from rai_bvar.engines.mcmc.state import SweepCtx
    from rai_bvar.engines.mcmc.sweep import equation_phase, make_sweep, sweep_values

    data = pc.size_data(size)
    model = pc.build_model(config, draws=1000, burn=warm, seed=11, const_path=const_path)
    out: dict = {"config": config, "size": size, "const_path_setting": const_path, "blocks": {}}
    with _jax.x64(), _blas.single_thread():
        res = resolve(model, data, rb.Runtime())
        static = res.static
        out["static"] = {"mean": repr(static.mean), "coef": repr(static.coef),
                         "vol": repr(static.vol), "shapes": repr(static.shapes)}  # fmt: skip
        fns = make_runner(static)
        program = fns.program
        sweep = make_sweep(program)
        plan = sweep.plan
        d = jax.tree.map(jnp.asarray, res.data)
        p = jax.tree.map(jnp.asarray, res.prior)
        cache = fns.prepare(d, p)
        C = 4
        st = fns.init(keys.init_keys(11, res.salt, C), d, p, cache)
        ck = keys.chain_keys(keys.fit_root(11, res.salt), C)
        it = 0
        while it < warm:
            m = min(warm - it, 500)
            st = fns.burn(ck, st, d, p, cache, np.asarray(it, np.uint32), np.asarray(m, np.int32))
            it += m
        st = jax.block_until_ready(st)
        base_it = warm

        def its(r: int):
            return np.asarray(base_it + 1000 * r, np.uint32)

        # equation phase output of one sweep (for the post-phase steps' ctx)
        def eq_one(ckey, s, i, data, prior, cache):
            sh = sweep_values(program, ckey, i, s, data, prior, cache)
            eq, eq_out, _ = equation_phase(program, plan.equation, sh, s, prior, cache, True)
            return eq, eq_out

        eq_v = jax.jit(jax.vmap(eq_one, in_axes=(0, 0, None, None, None, None)))
        eq_c = eq_v.lower(ck, st, its(0), d, p, cache).compile()
        _, eq_out = jax.block_until_ready(eq_c(ck, st, its(0), d, p, cache))

        blocks: dict = {}

        def timed(name: str, fn, make_args, per: int = 1) -> None:
            gate_info = pc.wait_idle(gate, wait) if gate > 0 else {"gate_open": None}
            before = pc.load_snapshot()
            t0 = time.perf_counter()
            compiled = jax.jit(fn).lower(*make_args(0)).compile()
            compile_s = time.perf_counter() - t0
            res_t = _time(compiled, make_args, reps, per)
            after = pc.load_snapshot()
            res_t.update(compile_s=compile_s, load_before=before, load_after=after,
                         gate=gate_info, contaminated=pc.contaminated(before, after, gate or 2.0))  # fmt: skip
            blocks[name] = res_t
            print(f"  {config} {size} {name}: {res_t['ms_min']:.3f} ms (median "
                  f"{res_t['ms_median']:.3f}; load {before['loadavg'][0]:.2f} → "
                  f"{after['loadavg'][0]:.2f})", flush=True)  # fmt: skip

        timed("equation_phase", lambda ckey, s, i, dd, pp, cc: eq_v(ckey, s, i, dd, pp, cc)[1].e,
              lambda r: (ck, st, its(r), d, p, cache))  # fmt: skip

        for phase, steps, pstep in (("pre", program.global_pre, plan.pre),
                                    ("post", program.global_post, plan.post)):  # fmt: skip
            for g, step in zip(steps, pstep, strict=True):

                def one(ckey, s, eo, i, data, prior, cache, _g=g, _b=step.block):
                    ctx = SweepCtx(data, prior, cache, eo)
                    return _g.sample(keys.step_key(ckey, i, _b), s, ctx, program.eq_map)[0]

                fn = jax.vmap(one, in_axes=(0, 0, 0, None, None, None, None))
                timed(f"step:{phase}:{step.name}:{step.block.name}", fn,
                      lambda r: (ck, st, eq_out, its(r), d, p, cache))  # fmt: skip

        def sw_one(ckey, s, i, data, prior, cache):
            return sweep(ckey, i, s, data, prior, cache)

        sw = jax.vmap(sw_one, in_axes=(0, 0, None, None, None, None))
        timed("sweep_standalone", sw, lambda r: (ck, st, its(r), d, p, cache))

        def scan_one(ckey, s, i0, data, prior, cache):
            def body(c, j):
                return sweep(ckey, i0 + j, c, data, prior, cache), None

            return jax.lax.scan(body, s, jnp.arange(block, dtype=jnp.uint32))[0]

        sc = jax.vmap(scan_one, in_axes=(0, 0, None, None, None, None))
        timed("sweep_scan", sc, lambda r: (ck, st, its(r), d, p, cache), per=block)

        burn_fn = functools.partial(fns.burn)
        nblk = np.asarray(block, np.int32)
        gate_info = pc.wait_idle(gate, wait) if gate > 0 else {"gate_open": None}
        before = pc.load_snapshot()
        s_b = jax.block_until_ready(burn_fn(ck, jax.tree.map(jnp.copy, st), d, p, cache,
                                            its(0), np.asarray(1, np.int32)))  # fmt: skip
        times = []
        for r in range(1, reps + 1):
            s_b = jax.block_until_ready(s_b)
            t = time.perf_counter()
            s_b = jax.block_until_ready(burn_fn(ck, s_b, d, p, cache, its(r), nblk))
            times.append(1000.0 * (time.perf_counter() - t) / block)
        after = pc.load_snapshot()
        times.sort()
        blocks["sweep_burn"] = {
            "ms_min": times[0], "ms_median": times[len(times) // 2], "reps": reps, "all": times,
            "load_before": before, "load_after": after, "gate": gate_info,
            "contaminated": pc.contaminated(before, after, gate or 2.0),
        }  # fmt: skip
        print(f"  {config} {size} sweep_burn: {times[0]:.3f} ms (load "
              f"{before['loadavg'][0]:.2f} → {after['loadavg'][0]:.2f})", flush=True)  # fmt: skip
        out["blocks"] = blocks
        out["plan"] = {"pre": [s.name for s in plan.pre], "equation": [s.name for s in plan.equation],
                       "post": [s.name for s in plan.post]}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="bp2")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--const-path", default="auto")
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--block", type=int, default=200)
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--wait", type=float, default=900.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    status = 0
    for config in [c for c in args.configs.split(",") if c]:
        tag = f"{args.size}_{config}" + (f"_{args.const_path}" if args.const_path != "auto" else "")
        tag += f"_{args.tag}" if args.tag else ""
        try:
            res = measure(config, args.size, warm=args.warm, reps=args.reps, block=args.block,
                          gate=args.gate, wait=args.wait, const_path=args.const_path)  # fmt: skip
            res["machine"] = pc.fingerprint()
            pc.save_json(args.out, f"cost_{tag}", res)
        except Exception:
            status = 1
            traceback.print_exc()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
