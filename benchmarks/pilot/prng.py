"""Pilot: threefry2x32 vs philox4x32 (DESIGN §10.4 lever 9; N07 §4.1, §12).

Two measurements per implementation, each inside one jitted ``fori_loop`` (minimum over
``--reps``):

* bulk: 24,000 float64 uniforms and normals per call (the N07 §12 microbenchmark);
* sweep pattern: the draws one Linear × SV sweep makes for ``C·n = 80`` (chain, equation) lanes —
  ``fold_in`` chain → sweep → block → equation → sub-draw, then ``uniform(T_pad)`` (indicators),
  ``normal(T_pad + 1)`` (AWOL), ``normal(q)`` (mean) and eight scalar uniforms/normals.

The sweep-pattern number is the PRNG share of a full sweep; ``sweep.py --prng philox4x32``
measures the end-to-end effect. Statistical quality is not tested here (N07 §14 item 7).

Usage: python -I prng.py [--iters 200] [--reps 5] --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    jax = pc.setup_jax()
    import jax.numpy as jnp
    from jax import lax, random

    C, n, T_pad, q, N = 4, 20, 320, 100, a.iters
    res = {}
    for impl in ("threefry2x32", "philox4x32"):
        root = random.key(0, impl=impl)

        def bulk(kind, root=root):
            def run(r0):
                def body(r, acc):
                    k = pc.fold(root, r0 + r.astype(jnp.uint32))
                    x = random.uniform(k, (24_000,)) if kind == "u" else random.normal(k, (24_000,))
                    return acc + jnp.sum(x)

                return lax.fori_loop(0, N, body, jnp.zeros(()))

            return run

        def sweep_pattern(root=root):
            ck = jax.vmap(lambda c: pc.fold(root, c))(jnp.arange(C, dtype=jnp.uint32))

            def lane(kit, i):
                km = pc.fold(pc.fold(kit, pc.BLOCK_EQ_MEAN), i)
                kv = pc.fold(pc.fold(kit, pc.BLOCK_EQ_VOL), i)
                tot = jnp.sum(random.normal(pc.fold(km, pc.SUB_Z), (q,)))
                tot += jnp.sum(random.uniform(pc.fold(kv, pc.SUB_S), (T_pad,)))
                tot += jnp.sum(random.normal(pc.fold(kv, pc.SUB_H), (T_pad + 1,)))
                for sub in (2, 4, 6, 8, 10):
                    tot += pc.uniform_tiny(pc.fold(kv, sub))
                for sub in (3, 7, 9):
                    tot += jnp.sum(random.normal(pc.fold(kv, sub), (2,)))
                return tot

            def run(r0):
                def body(r, acc):
                    it = r0 + r.astype(jnp.uint32)
                    kits = jax.vmap(lambda k: pc.fold(k, it))(ck)
                    tot = jax.vmap(
                        lambda kit: jax.vmap(lambda i: lane(kit, i))(
                            jnp.arange(n, dtype=jnp.uint32)
                        )
                    )(kits)
                    return acc + jnp.sum(tot)

                return lax.fori_loop(0, N, body, jnp.zeros(()))

            return run

        out = {}
        for label, f in (
            ("bulk_uniform_24k", bulk("u")),
            ("bulk_normal_24k", bulk("n")),
            ("sweep_pattern_80_lanes", sweep_pattern()),
        ):
            compiled, tc = pc.aot(f, jnp.uint32(0))
            ctr = [0]

            def call(compiled=compiled, ctr=ctr):
                ctr[0] += N
                return compiled(jnp.uint32(ctr[0]))

            tm = pc.time_call(call, reps=a.reps)
            out[label] = {"ms": 1e3 * tm["min"] / N, "compile_s": tc}
            print(f"{impl:13s} {label:24s} {out[label]['ms']:.4f} ms per call", flush=True)
        res[impl] = out
    ratio = {
        k: res["threefry2x32"][k]["ms"] / res["philox4x32"][k]["ms"] for k in res["threefry2x32"]
    }
    print("threefry / philox time ratio:", {k: round(v, 2) for k, v in ratio.items()})
    path = pc.save_result(a.out, "prng", {"iters": N, "impls": res, "threefry_over_philox": ratio})
    print("wrote", path)


if __name__ == "__main__":
    main()
