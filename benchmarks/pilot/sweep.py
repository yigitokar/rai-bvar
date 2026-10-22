"""Pilot: full Linear × SV structural sweep at the BENCH-P2 shape, run through a runner prototype.

DESIGN §10.1/§10.3/§10.6 (WP3). Structural Mode-I-style prior, weighted path with the packed
Khatri–Rao Gram (D-25), Jacobi-scaled two-solve precision draw (D-14), the real SV core of
``pilot_common.sv_step`` (exact mode by default), optional Student-t transitory step (collapsed
ν grid), PRNG keys by ``fold_in`` (absolute sweep → block → equation → sub-draw, §5.6).
Runner (§5.1): a burn executable (``fori_loop`` with a traced count, donated state) and a sample
executable (``scan`` over ``chunk`` kept draws, ``fori_loop`` over a traced ``thin``, snapshot
by ``tree_where``, per-draw extract), chains by ``vmap`` or ``shard_map`` over CPU devices.

Measured: compile times (AOT), ms per sweep (minimum over ``--reps`` × ``--sweeps`` burn sweeps),
ms per kept draw including the device → host copy for each ``--chunks`` value, acceptance rates
from the accumulators, and a BENCH-P2 projection
``compile + 1000·ms_burn + 10000·ms_kept(chunk=250) + host`` for 4 chains × (1000 + 10000).

Usage: python -I sweep.py [--n 20 --p 4 --T 300] [--exact 1] [--transitory none|t]
       [--nu-form logaddexp|log_sum] [--tau-sampler jax|mt2] [--prng threefry2x32|philox4x32]
       [--buckets 1|2] [--unroll 1] [--devices 1]
       [--chunks 250] [--sweeps 1000 --reps 3] [--tag NAME] --out DIR
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc


class Trans(NamedTuple):
    nu_idx: Any  # (n,) int32
    log_tau: Any  # (n, T_pad)


class ChainState(NamedTuple):
    theta: Any  # (n, q)
    sv: pc.SVState  # leaves with leading n
    trans: Any  # Trans or ()
    code: Any  # int32: first failing (block, equation), 0 = ok
    min_pivot: Any  # f64


def build_problem(n: int, p: int, T: int, t_bucket: int = 32, seed: int = 0) -> dict[str, Any]:
    """Host-side data, packed products and prior arrays (NumPy)."""
    import numpy as np

    d = pc.simulate_var_sv(n, p, T, seed=seed)
    k = 1 + n * p
    q = k + n - 1
    T_pad = t_bucket * math.ceil(T / t_bucket)
    F = np.zeros((T_pad, q))
    F[:T, :k] = d.X
    F[:T, k:] = -d.Y[:, : n - 1]  # δ-design with M_c = 0 (stationary data, zero own-lag means)
    iu, ju, pos = pc.packed_index(q)
    FF = F[:, iu] * F[:, ju]
    Yeq = np.zeros((n, T_pad))
    Yeq[:, :T] = d.Y.T
    tmask = (np.arange(T_pad) < T).astype(np.float64)
    psi = d.psi
    lam = 0.2
    cmask = np.ones((n, q))
    prec0 = np.ones((n, q))
    for i in range(n):
        cmask[i, k + i :] = 0.0  # a-slot j active iff j < i (inactive slots trailing, §2.3)
        prec0[i, 0] = 1.0 / (100.0 * psi[i])
        for lag in range(1, p + 1):
            for j in range(n):
                prec0[i, 1 + (lag - 1) * n + j] = (lag**2) * psi[j] / (lam**2 * psi[i])
        prec0[i, k : k + i] = psi[:i] / psi[i]
    iota = np.arange(1, n + 1)
    return {
        "n": n,
        "p": p,
        "k": k,
        "q": q,
        "T": T,
        "T_pad": T_pad,
        "psi": psi,
        "pos": pos,
        "F": F,
        "FF": FF,
        "Yeq": Yeq,
        "tmask": tmask,
        "cmask": cmask,
        "prec0": prec0,
        "mean0": np.zeros((n, q)),
        "b_mu": np.log(psi / iota),
        "log_c": np.log(1e-12 * psi),
        "P": int(iu.size),
    }


def make_fns(prob: dict[str, Any], a: argparse.Namespace):
    """Return (init, sweep, extract) closures over static configuration only."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    n, q, T_pad = prob["n"], prob["q"], prob["T_pad"]
    pos = prob["pos"]
    k = prob["k"]
    table = pc.TABLES[a.mixture]
    tdist = a.transitory == "t"
    if a.buckets == 2:
        split = n // 2
        qA = k + split - 1
        PA = qA * (qA + 1) // 2
        posA = pos[:qA, :qA]
        assert posA.max() < PA

    def mean_phase(kit, theta, log_lam, D, P):
        w = D["tmask"][None, :] * jnp.exp(-log_lam)  # (n, T_pad), never 1/exp
        kmean = pc.fold(kit, pc.BLOCK_EQ_MEAN)
        keys = jax.vmap(lambda i: pc.fold(kmean, i))(jnp.arange(n, dtype=jnp.uint32))

        def one(key, w_i, y_i, cm, pr, m0, F, FF, pos_):
            return pc.mean_draw(key, w_i, y_i, F, FF, pos_, cm, pr, m0)

        if a.buckets == 1:
            th, e, mind = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
                keys, w, D["Yeq"], P["cmask"], P["prec0"], P["mean0"], D["F"], D["FF"], pos
            )
        else:
            sl = slice(0, split)
            thA, eA, mA = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
                keys[sl],
                w[sl],
                D["Yeq"][sl],
                P["cmask"][sl, :qA],
                P["prec0"][sl, :qA],
                P["mean0"][sl, :qA],
                D["F"][:, :qA],
                D["FF"][:, :PA],
                posA,
            )
            sl = slice(split, n)
            thB, eB, mB = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
                keys[sl],
                w[sl],
                D["Yeq"][sl],
                P["cmask"][sl],
                P["prec0"][sl],
                P["mean0"][sl],
                D["F"],
                D["FF"],
                pos,
            )
            thA = jnp.pad(thA, ((0, 0), (0, q - qA)))
            th, e, mind = (jnp.concatenate(x) for x in ((thA, thB), (eA, eB), (mA, mB)))
        return th, e, mind

    def vol_phase(kit, sv, trans, e, D, P):
        kvol = pc.fold(kit, pc.BLOCK_EQ_VOL)
        keys = jax.vmap(lambda i: pc.fold(kvol, i))(jnp.arange(n, dtype=jnp.uint32))
        tm, T_eff = D["tmask"], D["T_eff"]

        def one(key, sv_i, tr_i, e_i, svp_i, lnw_i):
            log_e2 = pc.masked_log_e2(e_i, tm)
            if tdist:
                log_z2 = jnp.where(tm > 0, log_e2 - sv_i.h[1:], 0.0)
                g, log_tau = pc.t_step(
                    key, log_z2, tm, T_eff, lnw_i, form=a.nu_form, tau_sampler=a.tau_sampler
                )
                log_e2c = log_e2 - log_tau
                tr_new = Trans(g, log_tau)
            else:
                log_e2c, tr_new = log_e2, ()
            sv_new, piv = pc.sv_step(
                key,
                sv_i,
                log_e2c,
                tm,
                T_eff,
                svp_i,
                table=table,
                exact=bool(a.exact),
                asis=bool(a.asis),
                unroll=a.unroll,
            )
            return sv_new, tr_new, piv

        return jax.vmap(one)(keys, sv, trans, e, P["sv"], P["log_nu_w"])

    def log_lam_eff(st):
        ll = st.sv.h[:, 1:]
        return ll + st.trans.log_tau if tdist else ll

    def sweep(ckey, it, st, D, P):
        kit = pc.fold(ckey, it)
        th, e, mind = mean_phase(kit, st.theta, log_lam_eff(st), D, P)
        sv, trans, piv = vol_phase(kit, st.sv, st.trans, e, D, P)
        ok_m = jnp.isfinite(mind) & (mind > 0)
        ok_v = jnp.isfinite(piv) & (piv > 0) & jnp.all(jnp.isfinite(sv.h), axis=1)
        eqs = jnp.arange(n, dtype=jnp.int32)
        codes = jnp.where(ok_m, jnp.where(ok_v, 0, 1 + 2000 + eqs), 1 + eqs)
        first = jnp.max(codes)
        code = jnp.where(st.code == 0, first, st.code)
        mp = jnp.minimum(st.min_pivot, jnp.minimum(jnp.min(mind), jnp.min(piv)))
        return ChainState(th, sv, trans, code, mp)

    def extract(st, D):
        T_eff = D["T_eff"]
        out = {
            "theta": st.theta,
            "h_T": jax.vmap(lambda h: lax.dynamic_index_in_dim(h, T_eff, keepdims=False))(st.sv.h),
            "mu": st.sv.mu,
            "phi": st.sv.phi,
            "sigma": st.sv.sigma,
        }
        if tdist:
            out["nu"] = st.trans.nu_idx
        return out

    def init(ikey, D, P):
        """Cold start (DESIGN §5.4): exact draw with weights m·ι/ψ, residual-scale μ, h ≡ μ, then
        one AWOL draw of h accepted unconditionally (pilot fix: with h ≡ μ a rejected first exact
        h-step gives C_T = 0 and an absorbing σ = 0; see 09_pilot.md §3.1)."""
        lam0 = jnp.log(P["psi"] / jnp.arange(1, n + 1))
        log_lam = jnp.broadcast_to(lam0[:, None], (n, T_pad))
        th, e, _ = mean_phase(pc.fold(ikey, 0), None, log_lam, D, P)
        ms = jnp.sum(D["tmask"] * e * e, axis=1) / D["T_eff"]
        zs = jax.random.normal(pc.fold(ikey, 1), (2, n))
        mu = jnp.log(ms) + 0.5 * zs[0]
        h = jnp.broadcast_to(mu[:, None], (n, T_pad + 1))
        phi = jnp.full((n,), 2.0 * 20.0 / 21.5 - 1.0)
        sigma = jnp.sqrt(P["sv"].B_sigma) * jnp.exp(0.25 * zs[1])
        sv = pc.sv_state_init(h, mu, phi, sigma)
        kv = pc.fold(ikey, 2)
        sv = jax.vmap(
            lambda i, s, e_i, p: pc.sv_step(
                pc.fold(kv, i),
                s,
                pc.masked_log_e2(e_i, D["tmask"]),
                D["tmask"],
                D["T_eff"],
                p,
                table=table,
                exact=False,
                cstep=False,
                asis=False,
            )[0]
        )(jnp.arange(n, dtype=jnp.uint32), sv, e, P["sv"])
        sv = pc.zero_acc(sv)
        trans = Trans(jnp.full((n,), 19, jnp.int32), jnp.zeros((n, T_pad))) if tdist else ()
        return ChainState(th, sv, trans, jnp.int32(0), jnp.asarray(jnp.inf))

    return init, sweep, extract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--exact", type=int, default=1)
    ap.add_argument("--asis", type=int, default=1)
    ap.add_argument("--mixture", default="ocsn10")
    ap.add_argument("--transitory", default="none", choices=["none", "t"])
    ap.add_argument("--nu-form", default="logaddexp", choices=["logaddexp", "log1p_exp", "log_sum"])
    ap.add_argument("--tau-sampler", default="jax", help="jax (random.loggamma) or mt<M>")
    ap.add_argument("--prng", default="threefry2x32")
    ap.add_argument("--buckets", type=int, default=1, choices=[1, 2])
    ap.add_argument("--unroll", type=int, default=1)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--chunks", default="250")
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    jax = pc.setup_jax(a.devices)
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    C = a.chains
    t_start = time.perf_counter()
    prob = build_problem(a.n, a.p, a.T)
    n, q, T_pad = prob["n"], prob["q"], prob["T_pad"]
    G = pc.NU_GRID.size
    D = {
        "F": jnp.asarray(prob["F"]),
        "FF": jnp.asarray(prob["FF"]),
        "Yeq": jnp.asarray(prob["Yeq"]),
        "tmask": jnp.asarray(prob["tmask"]),
        "T_eff": jnp.int32(a.T),
    }
    P = {
        "cmask": jnp.asarray(prob["cmask"]),
        "prec0": jnp.asarray(prob["prec0"]),
        "mean0": jnp.asarray(prob["mean0"]),
        "psi": jnp.asarray(prob["psi"]),
        "sv": pc.SVPrior(
            jnp.asarray(prob["b_mu"]),
            jnp.full(n, 10.0),
            jnp.full(n, 20.0),
            jnp.full(n, 1.5),
            jnp.ones(n),
            jnp.asarray(prob["log_c"]),
        ),
        "log_nu_w": jnp.full((n, G), -np.log(G)),
    }
    init, sweep, extract = make_fns(prob, a)

    def burn_one(ckey, st, D, P, it0, nsw):
        return lax.fori_loop(
            0, nsw, lambda r, s: sweep(ckey, it0 + r.astype(jnp.uint32), s, D, P), st
        )

    def make_sample(chunk):
        def sample_one(ckey, st, D, P, it0, thin, stop):
            def keep(carry, j):
                st, snap = carry
                base = it0 + j * thin
                st = lax.fori_loop(
                    0,
                    thin.astype(jnp.int32),
                    lambda r, s: sweep(ckey, base + r.astype(jnp.uint32), s, D, P),
                    st,
                )
                snap = jax.tree.map(
                    lambda x, y: jnp.where(j == stop.astype(jnp.uint32), x, y), st, snap
                )
                return (st, snap), extract(st, D)

            (st, snap), out = lax.scan(keep, (st, st), jnp.arange(chunk, dtype=jnp.uint32))
            return st, snap, out

        return sample_one

    in_c = (0, 0, None, None, None, None)
    burn_v = jax.vmap(burn_one, in_axes=in_c)
    init_v = jax.vmap(init, in_axes=(0, None, None))
    ckeys = pc.chain_keys(0, C, impl=a.prng)
    ikeys = pc.chain_keys(0, C, impl=a.prng, stream=pc.STREAM_INIT)
    chunks = [int(c) for c in a.chunks.split(",")]
    if a.devices > 1:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as Ps

        mesh = jax.make_mesh((a.devices,), ("chain",))
        shc, shr = NamedSharding(mesh, Ps("chain")), NamedSharding(mesh, Ps())

        def shard(f, n_rep):
            return jax.shard_map(
                f,
                mesh=mesh,
                in_specs=(Ps("chain"), Ps("chain")) + (Ps(),) * n_rep,
                out_specs=Ps("chain"),
                check_vma=False,
            )

        burn_f = shard(burn_v, 4)
        sample_fs = {
            c: shard(jax.vmap(make_sample(c), in_axes=(0, 0, None, None, None, None, None)), 5)
            for c in chunks
        }
        ckeys = jax.device_put(ckeys, shc)
        ikeys = jax.device_put(ikeys, shc)
        D = jax.device_put(D, shr)
        P = jax.device_put(P, shr)
        init_f = jax.shard_map(
            init_v,
            mesh=mesh,
            in_specs=(Ps("chain"), Ps(), Ps()),
            out_specs=Ps("chain"),
            check_vma=False,
        )
    else:
        burn_f = burn_v
        sample_fs = {
            c: jax.vmap(make_sample(c), in_axes=(0, 0, None, None, None, None, None))
            for c in chunks
        }
        init_f = init_v

    st = jax.jit(init_f)(ikeys, D, P)
    pc.block(st)
    t_prep = time.perf_counter() - t_start
    u32 = jnp.uint32
    burn_c, t_cburn = pc.aot(burn_f, ckeys, st, D, P, u32(0), jnp.int32(1), donate_argnums=1)
    sample_cs, t_csample = {}, {}
    for c in chunks:
        sample_cs[c], t_csample[c] = pc.aot(
            sample_fs[c], ckeys, st, D, P, u32(0), u32(1), jnp.int32(c - 1), donate_argnums=1
        )
    print(f"[{a.tag}] compile burn {t_cburn:.2f} s, sample {t_csample}", flush=True)

    it = 0
    st = burn_c(ckeys, st, D, P, u32(it), jnp.int32(a.warm))
    it += a.warm
    pc.block(st)
    assert int(np.max(np.asarray(st.code))) == 0, "numerical failure during warm-up"
    st = st._replace(sv=pc.zero_acc(st.sv))
    times = []
    loads = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        st = burn_c(ckeys, st, D, P, u32(it), jnp.int32(a.sweeps))
        pc.block(st)
        times.append(time.perf_counter() - t0)
        loads.append(pc.fingerprint()["loadavg"][0])
        it += a.sweeps
    ms_sweep = 1e3 * min(times) / a.sweeps
    nsw = a.reps * a.sweeps
    sv = jax.tree.map(np.asarray, st.sv)
    acc = {
        f: getattr(sv, f) / nsw
        for f in ("acc_h", "acc_cgp", "acc_cs2", "acc_ncphi", "acc_ncms", "acc_fold")
    }
    acc_summary = {
        f: {
            "mean": float(v.mean()),
            "min": float(v.min()),
            "q05": float(np.quantile(v, 0.05)),
            "q95": float(np.quantile(v, 0.95)),
            "max": float(v.max()),
        }
        for f, v in acc.items()
    }
    acc_summary["acc_fold"]["share_gt_1pct"] = float(np.mean(acc["acc_fold"] > 0.01))
    acc_summary["rej_run_max"] = int(sv.rej_run_max.max())
    print(
        f"[{a.tag}] {ms_sweep:.3f} ms/sweep (reps {[round(1e3 * t / a.sweeps, 3) for t in times]}), "
        f"acc_h mean {acc_summary['acc_h']['mean']:.4f} min {acc_summary['acc_h']['min']:.4f}, "
        f"acc_ncms {acc_summary['acc_ncms']['mean']:.4f}, max rej run {acc_summary['rej_run_max']}, "
        f"fold mean {acc_summary['acc_fold']['mean']:.4f} max {acc_summary['acc_fold']['max']:.4f} "
        f"(>1% in {acc_summary['acc_fold']['share_gt_1pct']:.2f} of equation-chains)",
        flush=True,
    )

    chunk_res = {}
    for c in chunks:
        n_calls = math.ceil(a.draws / c)
        store = None
        t0 = time.perf_counter()
        for call in range(n_calls):
            stop = min(c, a.draws - call * c) - 1
            st, snap, out = sample_cs[c](ckeys, st, D, P, u32(it), u32(1), jnp.int32(stop))
            host = jax.device_get(out)
            if store is None:
                store = {
                    kk: np.empty((C, a.draws, *v.shape[2:]), v.dtype) for kk, v in host.items()
                }
            for kk, v in host.items():
                store[kk][:, call * c : call * c + stop + 1] = v[:, : stop + 1]
            last = call == n_calls - 1
            it += stop + 1 if last else c
            st = snap if last else st  # final state = snapshot at the last kept draw (§5.1)
        pc.block(st)
        dt = time.perf_counter() - t0
        draw_bytes = sum(v[0, 0].nbytes for v in store.values())
        chunk_res[c] = {
            "ms_per_kept": 1e3 * dt / (n_calls * c),
            "ms_per_kept_stored": 1e3 * dt / a.draws,
            "compile_s": t_csample[c],
            "bytes_per_chain_draw": int(draw_bytes),
            "loadavg": pc.fingerprint()["loadavg"][0],
        }
        print(
            f"[{a.tag}] chunk {c:4d}: {chunk_res[c]['ms_per_kept']:.3f} ms per kept draw incl. host copy "
            f"({draw_bytes / 1e3:.1f} KB per chain-draw), compile {t_csample[c]:.2f} s",
            flush=True,
        )

    # Host diagnostics estimate: ESS + split-R̂ of 330 monitors × C chains × 10,000 draws.
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(C, 10_000))
    t0 = time.perf_counter()
    for _ in range(330):
        [pc.ess(x) for x in xs]
        pc.split_rhat(xs)
    t_diag = time.perf_counter() - t0

    c_ref = 250 if 250 in chunk_res else chunks[0]
    proj = {
        "compile_s": t_cburn + t_csample[c_ref],
        "burn_s": 1000 * ms_sweep / 1e3,
        "sample_s": 10_000 * chunk_res[c_ref]["ms_per_kept"] / 1e3,
        "host_diag_s": t_diag,
        "prepare_init_s": t_prep,
    }
    proj["total_s"] = sum(proj.values())
    print(
        f"[{a.tag}] BENCH-P2 projection {proj['total_s']:.1f} s = "
        + ", ".join(f"{kk} {v:.1f}" for kk, v in proj.items() if kk != "total_s"),
        flush=True,
    )
    name = "sweep" + (f"_{a.tag}" if a.tag else "")
    path = pc.save_result(
        a.out,
        name,
        {
            "args": vars(a),
            "shape": {
                "n": n,
                "p": prob["p"],
                "k": prob["k"],
                "q": q,
                "T": a.T,
                "T_pad": T_pad,
                "P": prob["P"],
                "C": C,
            },
            "ms_per_sweep": ms_sweep,
            "ms_per_sweep_reps": [1e3 * t / a.sweeps for t in times],
            "loadavg_during": loads,
            "compile_burn_s": t_cburn,
            "chunks": chunk_res,
            "acceptance": acc_summary,
            "projection": proj,
            "final_code": int(np.max(np.asarray(st.code))),
        },
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
