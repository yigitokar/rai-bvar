"""Pilot: packed vs tiled vs batched weighted Gram; the ``gram="auto"`` threshold (DESIGN §10.5).

For ``B = C·n`` weight vectors ``w_b`` (chains × equations) and a shared design ``F`` (T_pad × q),
compute ``G_b = Fᵀ diag(w_b) F`` (q × q) in three ways:

* ``packed``: ``(W @ FF)[:, pos]`` with ``FF = F[:, iu]·F[:, ju]`` (T_pad × P) stored once (D-25);
* ``packed_tiled``: the same product built tile by tile from ``F`` each call (``FF`` never stored;
  tile = multiple of 512 packed columns), then one gather;
* ``batched``: ``einsum`` of ``√w ⊙ F`` with itself (the oracle path).

Shapes: the BENCH-P2 shape and the three §10.5 examples (plus ``n = 20, p = 13``). Reports ms per
call (minimum over ``--reps`` of ``--iters`` calls in one ``fori_loop``), the ``FF`` size,
the relative error against ``batched``, and the rule's choice.

Usage: python -I gram.py [--iters 20] [--reps 5] --out DIR
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc

SHAPES = (  # (label, n, p, T_eff, chains)
    ("n20_p4_T300", 20, 4, 300, 4),
    ("n50_p4_T300", 50, 4, 300, 4),
    ("n20_p13_T300", 20, 13, 300, 4),
    ("n30_p13_T600", 30, 13, 600, 4),
)
MiB = 2**20


def rule(T_pad: int, P: int) -> tuple[str, int]:
    """DESIGN §10.5: packed iff ``T_pad·P·8 ≤ 256 MiB``; else tiles with ``T_pad·tile·8 ≤ 64 MiB``."""
    if T_pad * P * 8 <= 256 * MiB:
        return "packed", 0
    tile = (64 * MiB // (T_pad * 8)) // 512 * 512
    return "packed_tiled", tile


def bench_shape(n: int, p: int, T: int, C: int, iters: int, reps: int, label: str) -> dict:
    """Time the three Gram modes for one shape; returns the result record."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    q = n * (p + 1)
    T_pad = 32 * math.ceil(T / 32)
    B = C * n
    iu, ju, pos = pc.packed_index(q)
    P = iu.size
    rng = np.random.default_rng(0)
    F = jnp.asarray(rng.normal(size=(T_pad, q)))
    W = jnp.asarray(np.exp(rng.normal(size=(B, T_pad))))
    FF = F[:, iu] * F[:, ju]
    choice, tile_rule = rule(T_pad, P)
    ff_mb = T_pad * P * 8 / 1e6
    out = {
        "q": q,
        "T_pad": T_pad,
        "B": B,
        "P": int(P),
        "FF_MB": ff_mb,
        "rule": choice,
        "rule_tile": tile_rule,
        "modes": {},
    }
    print(
        f"--- {label}: q={q} T_pad={T_pad} B={B} P={P} FF={ff_mb:.1f} MB rule={choice} {tile_rule}",
        flush=True,
    )

    def loop(f):
        def run(W_, arg):
            def body(r, acc):
                Wr = W_ * (1.0 + 1e-12 * r.astype(jnp.float64))
                return acc + jnp.sum(f(Wr, arg)[:, 0, :])

            return lax.fori_loop(0, iters, body, jnp.zeros(()))

        return run

    def packed(Wr, FF):
        return (Wr @ FF)[:, pos]

    def tiled(tile):
        starts = list(range(0, P, tile))

        def f(Wr, F):
            parts = [Wr @ (F[:, iu[s : s + tile]] * F[:, ju[s : s + tile]]) for s in starts]
            return jnp.concatenate(parts, axis=1)[:, pos]

        return f

    def batched(Wr, F):
        Fw = F[None] * jnp.sqrt(Wr)[:, :, None]
        return jnp.einsum("btk,btl->bkl", Fw, Fw)

    ref = np.asarray(jax.jit(batched)(W, F))
    cands = [("packed", packed, FF), ("batched", batched, F)]
    tiles = sorted({t for t in (512 * 4, 512 * 16, tile_rule or 512 * 64) if t < P})
    cands += [(f"tiled_{t}", tiled(t), F) for t in tiles]
    for name, f, arg in cands:
        try:
            compiled, tc = pc.aot(loop(f), W, arg)
            tm = pc.time_call(lambda c=compiled, arg=arg: c(W, arg), reps=reps)
            val = np.asarray(jax.jit(f)(W, arg))
            err = float(np.max(np.abs(val - ref)) / np.max(np.abs(ref)))
            out["modes"][name] = {"ms": 1e3 * tm["min"] / iters, "compile_s": tc, "rel_err": err}
            print(f"{name:14s} {out['modes'][name]['ms']:9.3f} ms  rel.err {err:.1e}", flush=True)
        except Exception as exc:  # memory exhaustion at the largest shapes is a result too
            out["modes"][name] = {"error": repr(exc)[:200]}
            print(f"{name:14s} failed: {exc!r}"[:200], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    pc.setup_jax()
    res = {label: bench_shape(n, p, T, C, a.iters, a.reps, label) for label, n, p, T, C in SHAPES}
    path = pc.save_result(a.out, "gram", {"iters": a.iters, "shapes": res})
    print("wrote", path)


if __name__ == "__main__":
    main()
