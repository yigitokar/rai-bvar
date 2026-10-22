"""Pilot: one vs two static equation buckets for the structural mean block (DESIGN §2.3, §10.4).

Equation ``i`` has ``q_i = k + i`` active coordinates. One bucket pads every equation to
``q = k + n − 1``; two buckets pad equations ``i < s`` to ``q_A = k + s − 1`` (using the prefix
``FF[:, :q_A(q_A+1)/2]`` of the column-major packed layout) and the rest to ``q``. Measures the
weighted mean draw (packed Gram + Jacobi-scaled Cholesky + two solves + residual) for
``C·n`` lanes at the BENCH-P2 shape, minimum over ``--reps`` of ``--iters`` draws in one
``fori_loop``, and prints the exact padding ratios ``Σ(padded cost)/Σ(active cost)``.

Usage: python -I buckets.py [--iters 100] [--reps 5] --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    jax = pc.setup_jax()
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sweep import build_problem

    C = 4
    prob = build_problem(20, 4, 300)
    n, k, q = prob["n"], prob["k"], prob["q"]
    pos = prob["pos"]
    F, FF = jnp.asarray(prob["F"]), jnp.asarray(prob["FF"])
    Y = jnp.asarray(prob["Yeq"])
    cm, pr, m0 = (jnp.asarray(prob[x]) for x in ("cmask", "prec0", "mean0"))
    rng = np.random.default_rng(0)
    logw = jnp.asarray(rng.normal(scale=0.3, size=(C, n, prob["T_pad"])) * prob["tmask"])
    keys = pc.chain_keys(0, C)
    qi = k + np.arange(n)
    ratios = {}
    for s in (n // 2, 12, 14):
        qA = k + s - 1
        ratios[s] = {
            "chol_1bucket": n * q**3 / np.sum(qi**3),
            "chol_2bucket": (s * qA**3 + (n - s) * q**3) / np.sum(qi**3),
            "gram_1bucket": n * q * (q + 1) / np.sum(qi * (qi + 1)),
            "gram_2bucket": (s * qA * (qA + 1) + (n - s) * q * (q + 1)) / np.sum(qi * (qi + 1)),
        }
    print(
        "padding ratios by split s:",
        {s: {kk: round(v, 3) for kk, v in r.items()} for s, r in ratios.items()},
    )

    def one(key, w_i, y_i, c_i, p_i, m_i, F_, FF_, pos_):
        return pc.mean_draw(key, w_i, y_i, F_, FF_, pos_, c_i, p_i, m_i)[0]

    vone = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))

    def make(split):
        def chain(ckey, lw):
            w = jnp.exp(-lw) * jnp.asarray(prob["tmask"])
            ks = jax.vmap(lambda i: pc.fold(ckey, i))(jnp.arange(n, dtype=jnp.uint32))
            if split is None:
                return vone(ks, w, Y, cm, pr, m0, F, FF, pos)
            qA = k + split - 1
            PA = qA * (qA + 1) // 2
            sl, sr = slice(0, split), slice(split, n)
            thA = vone(
                ks[sl],
                w[sl],
                Y[sl],
                cm[sl, :qA],
                pr[sl, :qA],
                m0[sl, :qA],
                F[:, :qA],
                FF[:, :PA],
                pos[:qA, :qA],
            )
            thB = vone(ks[sr], w[sr], Y[sr], cm[sr], pr[sr], m0[sr], F, FF, pos)
            return jnp.concatenate([jnp.pad(thA, ((0, 0), (0, q - qA))), thB])

        def run(keys, lw):
            def body(r, acc):
                ks = jax.vmap(lambda kk: pc.fold(kk, r))(keys)
                return acc + jnp.sum(jax.vmap(chain)(ks, lw + 1e-12 * r))

            return lax.fori_loop(0, a.iters, body, jnp.zeros(()))

        return run, jax.jit(jax.vmap(chain))

    res = {}
    draws = {}
    for label, split in (
        ("one_bucket", None),
        ("two_buckets_s10", 10),
        ("two_buckets_s12", 12),
        ("two_buckets_s14", 14),
    ):
        run, single = make(split)
        compiled, tc = pc.aot(run, keys, logw)
        tm = pc.time_call(lambda c=compiled: c(keys, logw), reps=a.reps)
        draws[label] = np.asarray(single(keys, logw))
        res[label] = {"ms": 1e3 * tm["min"] / a.iters, "compile_s": tc}
        print(
            f"{label:18s} {res[label]['ms']:.3f} ms per mean draw (80 lanes), compile {tc:.2f} s",
            flush=True,
        )
    for label in res:
        res[label]["max_abs_diff_vs_one_bucket"] = float(
            np.max(np.abs(draws[label] - draws["one_bucket"]))
        )
    print(
        "max |θ(two) − θ(one)| (same z on active coordinates):",
        {kk: v["max_abs_diff_vs_one_bucket"] for kk, v in res.items()},
    )
    path = pc.save_result(
        a.out,
        "buckets",
        {
            "iters": a.iters,
            "variants": res,
            "padding_ratios": {str(s): r for s, r in ratios.items()},
        },
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
