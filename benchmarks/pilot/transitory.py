"""Pilot: ν-grid cost of SV(t), τ draws, and the outlier step (DESIGN §10.3, risk R5; N03 §4.3c).

``B = 80`` lanes (4 chains × 20 equations), ``T_eff = 300``, ``T_pad = 320``, ``G = 38``
(ν ∈ {3..40}), ``K = 20`` outlier grid. Each kernel runs ``--iters`` times inside one jitted
``fori_loop``; the minimum over ``--reps`` of ms per call (all 80 lanes) is reported.

Kernels: the O(T·G) sums of the collapsed ν logits in three algebraically equal forms
(``logaddexp``, ``log1p_exp``, ``log_sum``; agreement reported), the full collapsed ν step
plus τ draw (``t_step``), the τ draw alone with ``random.loggamma`` and with
``log(random.gamma)``, the uncollapsed ν | τ step (O(T + G)), and the outlier ``o | ·``, ``p | o``
step.

Usage: python -I transitory.py [--iters 100] [--reps 5] --out DIR
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
    from jax import lax, random

    B, T, T_pad, G = 80, 300, 320, pc.NU_GRID.size
    rng = np.random.default_rng(0)
    tm_np = (np.arange(T_pad) < T).astype(float)
    # log z² of standardized t(8) draws (realistic magnitudes); padded rows 0
    z = rng.standard_t(8, size=(B, T_pad)) * np.sqrt(6 / 8)
    log_z2 = jnp.asarray(np.where(tm_np > 0, np.log(z * z), 0.0))
    tm = jnp.asarray(tm_np)
    T_eff = jnp.int32(T)
    lnw = jnp.full((B, G), -np.log(G))
    _, log_c_np, _ = pc.nu_constants()
    log_c = jnp.asarray(log_c_np)
    keys = pc.chain_keys(0, B)
    N = a.iters
    res = {}

    def bench(label, per_lane, *args):
        """per_lane(key, *lane_args) -> pytree; summed into a scalar so nothing is dead code."""

        def run(keys, *args):
            def body(r, acc):
                ks = jax.vmap(lambda k: pc.fold(k, r))(keys)
                # inputs depend on r so XLA cannot hoist key-free kernels out of the loop
                eps = 1e-12 * r.astype(jnp.float64)
                args_r = [x + eps if jnp.issubdtype(x.dtype, jnp.floating) else x for x in args]
                out = jax.vmap(per_lane)(ks, *args_r)
                return acc + sum(jnp.sum(x.astype(jnp.float64)) for x in jax.tree.leaves(out))

            return lax.fori_loop(0, N, body, jnp.zeros(()))

        compiled, tc = pc.aot(run, keys, *args)
        tm_ = pc.time_call(lambda: compiled(keys, *args), reps=a.reps)
        res[label] = {
            "ms": 1e3 * tm_["min"] / N,
            "ms_median": 1e3 * tm_["median"] / N,
            "compile_s": tc,
        }
        print(
            f"{label:34s} {res[label]['ms']:7.3f} ms per call (80 lanes), compile {tc:.2f} s",
            flush=True,
        )

    obs = tm > 0
    for form in ("logaddexp", "log1p_exp", "log_sum"):
        bench(
            f"nu_sums_{form}",
            lambda k, lz, form=form: pc.nu_grid_sums(lz, obs, log_c, form),
            log_z2,
        )
    sums = {
        f: np.asarray(jax.jit(jax.vmap(lambda lz, f=f: pc.nu_grid_sums(lz, obs, log_c, f)))(log_z2))
        for f in ("logaddexp", "log1p_exp", "log_sum")
    }
    agree = {
        f: float(np.max(np.abs(sums[f] - sums["logaddexp"]) / np.abs(sums["logaddexp"])))
        for f in ("log1p_exp", "log_sum")
    }
    print("relative agreement with logaddexp:", agree)
    for form, ts in (("logaddexp", "jax"), ("logaddexp", "mt2"), ("log_sum", "mt2")):
        bench(
            f"t_step_full_{form}_{ts}",
            lambda k, lz, w, form=form, ts=ts: pc.t_step(
                k, lz, tm, T_eff, w, form=form, tau_sampler=ts
            ),
            log_z2,
            lnw,
        )
    shape_a = jnp.asarray(rng.choice(pc.NU_GRID, size=B) * 0.5 + 0.5)
    bench("tau_jax_loggamma", lambda k, s: random.loggamma(k, s, (T_pad,)), shape_a)
    bench("tau_jax_log_of_gamma", lambda k, s: jnp.log(random.gamma(k, s, (T_pad,))), shape_a)
    for M in (1, 2, 3, 4):
        bench(f"tau_mt{M}", lambda k, s, M=M: pc.loggamma_mt(k, s, (T_pad,), M=M), shape_a)
    # exactness check of loggamma_mt: KS against scipy's Gamma at several shapes
    from scipy import stats

    ks_p = {}
    for shp in (2.0, 2.5, 5.0, 20.5):
        draws = np.exp(
            np.asarray(
                jax.jit(lambda k, s=shp: pc.loggamma_mt(k, s, (200_000,), M=2))(jax.random.key(11))
            )
        )
        ks_p[str(shp)] = float(stats.kstest(draws, stats.gamma(shp).cdf).pvalue)
    print("loggamma_mt KS p-values vs scipy Gamma:", ks_p)
    log_tau = jnp.asarray(rng.normal(scale=0.3, size=(B, T_pad)) * tm_np)
    bench(
        "nu_uncollapsed_given_tau",
        lambda k, lt, w: pc.t_step_uncollapsed(k, lt, tm, T_eff, w),
        log_tau,
        lnw,
    )
    lp = jnp.full((B,), np.log(0.06))
    l1p = jnp.full((B,), np.log1p(-0.06))
    bench(
        "outlier_o_then_p",
        lambda k, lz, p_, q_: pc.outlier_step(k, lz, tm, p_, q_),
        log_z2,
        lp,
        l1p,
    )
    bench(
        "outlier_o_then_p_fast",
        lambda k, lz, p_, q_: pc.outlier_step(k, lz, tm, p_, q_, fast=True),
        log_z2,
        lp,
        l1p,
    )
    o1 = jax.jit(jax.vmap(lambda k, lz: pc.outlier_step(k, lz, tm, lp[0], l1p[0])[0]))(keys, log_z2)
    o2 = jax.jit(jax.vmap(lambda k, lz: pc.outlier_step(k, lz, tm, lp[0], l1p[0], fast=True)[0]))(
        keys, log_z2
    )
    same_o = float(np.mean(np.asarray(o1) == np.asarray(o2)))
    print("outlier fast vs per-element form: identical draws fraction", same_o)
    bench("uniform_T_pad_only", lambda k: random.uniform(k, (T_pad,)))
    path = pc.save_result(
        a.out,
        "transitory",
        {
            "lanes": B,
            "T_eff": T,
            "T_pad": T_pad,
            "G": G,
            "iters": N,
            "kernels": res,
            "nu_sum_agreement": agree,
            "loggamma_mt_ks_p": ks_p,
            "outlier_fast_identical_fraction": same_o,
        },
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
