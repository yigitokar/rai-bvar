"""Pilot: cost of the real SV-parameter block and of the exact mode; AWOL scan ``unroll``.

DESIGN §10.3/§10.6 (WP3 items "exact-mode cost with the real SV-parameter block" and "scan
unroll"). ``B = C·n = 80`` lanes (4 chains × 20 equations), ``T_eff = 300``, ``T_pad = 320``,
OCSN-10. Each variant runs ``--sweeps`` SV updates inside one jitted ``fori_loop`` (dispatch
amortised, as in the runner) and reports the minimum over ``--reps`` of ms per sweep for all
lanes. Stage costs are differences of cumulative variants:

    V1 indicators + AWOL        V2 = V1 + exact h correction     V3 = V2 + centred (γ, φ), σ²
    V4 = V3 + NC φ, (μ, σ) with exact correction = DEFAULT        V5 = V4 without exact mode

Usage: python -I sv_block.py [--sweeps 300] [--reps 5] --out DIR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc

VARIANTS = {
    "V1_ind_awol": dict(exact=False, cstep=False, asis=False),
    "V2_plus_exact_h": dict(exact=True, cstep=False, asis=False),
    "V3_plus_cstep": dict(exact=True, cstep=True, asis=False),
    "V4_full_exact": dict(exact=True, cstep=True, asis=True),
    "V5_full_nonexact": dict(exact=False, cstep=True, asis=True),
}
SETTINGS = ((0.95, 0.2), (0.5, 0.05), (0.99, 0.1))


def build(B: int, T: int, T_pad: int, seed: int = 0):
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    log_e2 = np.zeros((B, T_pad))
    h0 = np.zeros((B, T_pad + 1))
    mu = np.zeros(B)
    for b in range(B):
        phi, sig = SETTINGS[b % 3]
        e, h = pc.simulate_sv_series(rng, T, -1.0, phi, sig)
        log_e2[b, :T] = np.log(e * e)
        h0[b, : T + 1] = h
        mu[b] = -1.0
    tmask = np.zeros(T_pad)
    tmask[:T] = 1.0
    psi = np.exp(mu)
    prior = pc.SVPrior(
        b_mu=jnp.asarray(np.log(psi)),
        B_mu=jnp.full(B, 10.0),
        a_phi=jnp.full(B, 20.0),
        b_phi=jnp.full(B, 1.5),
        B_sigma=jnp.ones(B),
        log_c=jnp.asarray(np.log(1e-12 * psi)),
    )
    phis = np.array([SETTINGS[b % 3][0] for b in range(B)])
    sigs = np.array([SETTINGS[b % 3][1] for b in range(B)])
    st = pc.sv_state_init(jnp.asarray(h0), jnp.asarray(mu), jnp.asarray(phis), jnp.asarray(sigs))
    return jnp.asarray(log_e2), jnp.asarray(tmask), jnp.int32(T), prior, st


def make_runner(variant: dict, table, unroll: int):
    import jax
    import jax.numpy as jnp
    from jax import lax

    def one(key, st, log_e2, tmask, T_eff, prior):
        return pc.sv_step(
            key, st, log_e2, tmask, T_eff, prior, table=table, unroll=unroll, **variant
        )[0]

    def run(keys, st, log_e2, tmask, T_eff, prior, it0, n):
        def body(r, s):
            it = it0 + r.astype(jnp.uint32)
            ks = jax.vmap(lambda k: pc.fold(pc.fold(k, it), pc.BLOCK_EQ_VOL))(keys)
            return jax.vmap(one, in_axes=(0, 0, 0, None, None, 0))(
                ks, s, log_e2, tmask, T_eff, prior
            )

        return lax.fori_loop(0, n, body, st)

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lanes", type=int, default=80)
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    pc.setup_jax()
    import jax.numpy as jnp
    import numpy as np

    B, T, T_pad = a.lanes, 300, 320
    log_e2, tmask, T_eff, prior, st0 = build(B, T, T_pad)
    keys = pc.chain_keys(0, B)
    results = {}

    def measure(label, variant, table=pc.OCSN10, unroll=1):
        run = make_runner(variant, table, unroll)
        args = (keys, st0, log_e2, tmask, T_eff, prior, jnp.uint32(0), jnp.int32(a.sweeps))
        compiled, t_compile = pc.aot(run, *args)
        st = compiled(*args)  # warm-up / burn
        pc.block(st)
        it = [a.sweeps]

        def call():
            out = compiled(
                keys, st, log_e2, tmask, T_eff, prior, jnp.uint32(it[0]), jnp.int32(a.sweeps)
            )
            it[0] += a.sweeps
            return out

        tm = pc.time_call(call, reps=a.reps, warmup=0)
        last = compiled(
            keys,
            pc.zero_acc(st),
            log_e2,
            tmask,
            T_eff,
            prior,
            jnp.uint32(10**6),
            jnp.int32(a.sweeps),
        )
        acc = {
            f: float(np.mean(np.asarray(getattr(last, f)) / a.sweeps))
            for f in ("acc_h", "acc_cgp", "acc_cs2", "acc_ncphi", "acc_ncms", "acc_fold")
        }
        res = {
            "ms_per_sweep_min": 1e3 * tm["min"] / a.sweeps,
            "ms_per_sweep_median": 1e3 * tm["median"] / a.sweeps,
            "compile_s": t_compile,
            "acceptance": acc,
            "finite": bool(np.all(np.isfinite(np.asarray(last.h)))),
            "loadavg": pc.fingerprint()["loadavg"],
        }
        results[label] = res
        print(
            f"{label:28s} {res['ms_per_sweep_min']:7.3f} ms/sweep (median {res['ms_per_sweep_median']:.3f}),"
            f" compile {t_compile:5.2f} s, acc_h {acc['acc_h']:.3f} acc_ncms {acc['acc_ncms']:.3f}",
            flush=True,
        )

    t0 = time.perf_counter()
    for label, v in VARIANTS.items():
        measure(label, v)
    measure("V4_full_exact_ksc7", VARIANTS["V4_full_exact"], table=pc.KSC7)
    for u in (2, 4, 8, 16):
        measure(f"V4_full_exact_unroll{u}", VARIANTS["V4_full_exact"], unroll=u)

    # Tridiagonal sampler alone (N07 §8.3 microbenchmark, B lanes × m = T_pad + 1).
    import jax

    rng = np.random.default_rng(1)
    m = T_pad + 1
    diag = jnp.asarray(2.0 + rng.uniform(size=(B, m)))
    off = jnp.asarray(-0.9 * np.ones((B, m - 1)))
    bb = jnp.asarray(rng.normal(size=(B, m)))
    zz = jnp.asarray(rng.normal(size=(B, m)))
    tri = {}
    for u in (1, 2, 4, 8, 16):
        f = jax.vmap(lambda d, o, b_, z_, u=u: pc.tridiag_precision_sample(d, o, b_, z_, u)[0])
        compiled, tc = pc.aot(f, diag, off, bb, zz)
        tm = pc.time_call(lambda c=compiled: c(diag, off, bb, zz), reps=30)
        tri[f"unroll{u}"] = {"ms": 1e3 * tm["min"], "compile_s": tc}
        print(
            f"tridiag alone unroll={u:2d}: {1e3 * tm['min']:.3f} ms, compile {tc:.2f} s", flush=True
        )
    results["tridiag_alone"] = tri

    base = results["V5_full_nonexact"]["ms_per_sweep_min"]
    full = results["V4_full_exact"]["ms_per_sweep_min"]
    stages = {
        "indicators_awol": results["V1_ind_awol"]["ms_per_sweep_min"],
        "exact_h": results["V2_plus_exact_h"]["ms_per_sweep_min"]
        - results["V1_ind_awol"]["ms_per_sweep_min"],
        "cstep": results["V3_plus_cstep"]["ms_per_sweep_min"]
        - results["V2_plus_exact_h"]["ms_per_sweep_min"],
        "ncstep_incl_exact": full - results["V3_plus_cstep"]["ms_per_sweep_min"],
        "exact_total": full - base,
    }
    print("stage costs (ms per sweep, 80 lanes):", {k: round(v, 3) for k, v in stages.items()})
    path = pc.save_result(
        a.out,
        "sv_block",
        {
            "lanes": B,
            "T_eff": T,
            "T_pad": T_pad,
            "sweeps": a.sweeps,
            "variants": results,
            "stages_ms": stages,
            "wall_s": time.perf_counter() - t0,
        },
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
