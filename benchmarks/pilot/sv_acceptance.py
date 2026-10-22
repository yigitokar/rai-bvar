"""Pilot: joint-path exact-``h`` acceptance at ``T = 300`` and burn-in from crude starts.

DESIGN risk R2 / N01 §14 item 5 (UNVERIFIED before this pilot). For each
``(φ, σ) ∈ {(0.95, 0.2), (0.5, 0.05), (0.99, 0.1)}`` with ``μ = −1``: ``R`` simulated series of
length ``T = 300`` (padded to ``T_pad = 320``), ``C`` chains each, SV core alone (``ZeroMean``:
the residual is the simulated ``e_t``), default priors of DESIGN §6.3 with the data-centred
level ``μ ~ N(log ψ̂, 10)``, ``ψ̂ = mean(e²)``, relative offset ``c = 1e-12·ψ̂``.

Reported per setting and mixture (OCSN-10 default, KSC-7 informational) and for
``exact ∈ {True, False}``:

* acceptance of every MH step after burn-in (mean, median, minimum and maximum over chains),
  longest run of consecutive ``h`` rejections, fold fraction; per chain, the NC fold fraction
  and the posterior 5% quantile and median of ``σ`` (calibration data for the DESIGN §8.3
  fold-fraction warning);
* fold calibration: two extra settings, homoskedastic ``(0.95, 0)`` and weakly heteroskedastic
  ``(0.95, 0.05)``, run with OCSN-10, exact mode and the ``design_awol`` start only;
* ESS per 1,000 sweeps of ``μ, φ, σ, h_T`` and the log-likelihood (median over chains);
* starts: ``design_raw`` = DESIGN §5.4 as written (``μ = log ψ̂ + N(0, 0.5²)``, ``h ≡ μ``,
  ``φ`` prior mean, ``σ = exp(N(0, 0.25²))``); ``design_awol`` = the same followed by one AWOL draw
  of ``h`` given those values, accepted unconditionally (an initialisation, not an MCMC step);
  ``high5``/``low5`` = ``h ≡ log ψ̂ ± 5`` (deliberately bad). For ``design_raw`` the fraction of
  chains whose first exact ``h`` proposal is rejected and that end with ``σ = 0`` is reported
  (with ``h ≡ μ`` such a rejection leaves ``C_T = 0``, the σ² proposal ``IG(T/2, 0)`` is the
  point 0 and is accepted: an absorbing state);
* burn-in: acceptance in sweep windows, longest rejection run, sweeps until the log-likelihood
  first enters the central 98% band of the post-burn-in draws, sweeps until the chain coalesces
  exactly with the ``design_awol`` chain driven by the same keys, and split-R̂ over the C chains
  at the default ``burn = 1000``, ``draws = 2000``.

Usage: python -I sv_acceptance.py [--R 8] [--C 4] [--burn 2000] [--keep 10000] --out DIR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc

SETTINGS = ((0.95, 0.2), (0.5, 0.05), (0.99, 0.1))
FOLD_SETTINGS = ((0.95, 0.0), (0.95, 0.05))  # fold-warning calibration (OCSN-10 exact only)
STARTS = ("design_awol", "design_raw", "high5", "low5")
WINDOWS = ((0, 100), (100, 500), (500, 2000))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--keep", type=int, default=10000)
    ap.add_argument(
        "--out", required=True, help="result directory (required; keep it outside the repo)"
    )
    a = ap.parse_args()
    jax = pc.setup_jax()
    import jax.numpy as jnp
    import numpy as np
    from jax import lax, random

    R, C, T = a.R, a.C, a.T
    T_pad = 32 * -(-T // 32)
    L = R * C
    tm_np = (np.arange(T_pad) < T).astype(float)
    tm = jnp.asarray(tm_np)
    T_eff = jnp.int32(T)
    out: dict = {
        "R": R,
        "C": C,
        "T": T,
        "T_pad": T_pad,
        "burn": a.burn,
        "keep": a.keep,
        "settings": {},
    }
    t0 = time.perf_counter()

    def make_run(table, exact, n):
        def run(keys, st, log_e2, prior, it0):
            def body(st, r):
                it = it0 + r
                ks = jax.vmap(lambda k: pc.fold(pc.fold(k, it), pc.BLOCK_EQ_VOL))(keys)
                st = jax.vmap(
                    lambda k, s, le, p: pc.sv_step(
                        k, s, le, tm, T_eff, p, table=table, exact=exact
                    )[0]
                )(ks, st, log_e2, prior)
                ll = jax.vmap(lambda h, le: pc.sv_loglik(h, le, tm))(st.h, log_e2)
                trace = jnp.stack([st.mu, st.phi, st.sigma, st.h[:, T], ll])  # (5, L)
                return st, (trace, st.acc_h)

            return lax.scan(body, st, jnp.arange(n, dtype=jnp.uint32))

        return jax.jit(run)

    def init_awol(ikeys, st, log_e2, prior, table):
        """One AWOL draw of h given the initial (μ, φ, σ), accepted unconditionally."""
        f = jax.vmap(
            lambda k, s, le, p: pc.sv_step(
                pc.fold(k, pc.BLOCK_EQ_VOL),
                s,
                le,
                tm,
                T_eff,
                p,
                table=table,
                exact=False,
                cstep=False,
                asis=False,
            )[0]
        )
        return pc.zero_acc(jax.jit(f)(ikeys, st, log_e2, prior))

    for phi0, sig0 in SETTINGS + FOLD_SETTINGS:
        calib = (phi0, sig0) in FOLD_SETTINGS
        rng = np.random.default_rng(int(1000 * phi0 + 100 * sig0))
        log_e2_np = np.zeros((R, T_pad))
        for r in range(R):
            e, _ = pc.simulate_sv_series(rng, T, -1.0, phi0, sig0)
            log_e2_np[r, :T] = np.log(e * e)
        psi = np.mean(np.exp(log_e2_np[:, :T]), axis=1)
        rep = np.repeat(np.arange(R), C)  # lane → dataset
        log_e2 = jnp.asarray(log_e2_np[rep])
        prior = pc.SVPrior(
            jnp.asarray(np.log(psi[rep])),
            jnp.full(L, 10.0),
            jnp.full(L, 20.0),
            jnp.full(L, 1.5),
            jnp.ones(L),
            jnp.asarray(np.log(1e-12 * psi[rep])),
        )
        ikeys = pc.chain_keys(1, L, stream=pc.STREAM_INIT)
        z = np.asarray(jax.vmap(lambda k: random.normal(k, (2,)))(ikeys))
        res_setting = {}
        for table in (pc.OCSN10, pc.KSC7):
            for exact in (True, False):
                if table is pc.KSC7 and not exact:
                    continue
                if calib and not (table is pc.OCSN10 and exact):
                    continue
                ref_burn = None
                for start in STARTS:
                    if calib and start != "design_awol":
                        continue
                    if start in ("high5", "low5") and not (table is pc.OCSN10 and exact):
                        continue
                    if start == "design_raw" and not exact:
                        continue
                    mu0 = np.log(psi[rep]) + 0.5 * z[:, 0]
                    hval = {
                        "design_awol": mu0,
                        "design_raw": mu0,
                        "high5": np.log(psi[rep]) + 5.0,
                        "low5": np.log(psi[rep]) - 5.0,
                    }[start]
                    st = pc.sv_state_init(
                        jnp.asarray(np.tile(hval[:, None], (1, T_pad + 1))),
                        jnp.asarray(mu0),
                        jnp.full(L, 2 * 20 / 21.5 - 1),
                        jnp.asarray(np.exp(0.25 * z[:, 1])),
                    )
                    if start == "design_awol":
                        st = init_awol(ikeys, st, log_e2, prior, table)
                    keys = pc.chain_keys(0, L)
                    run_b = make_run(table, exact, a.burn)
                    run_k = make_run(table, exact, a.keep)
                    st, (tr_b, acc_b) = run_b(keys, st, log_e2, prior, jnp.uint32(0))
                    rej_burn = np.asarray(st.rej_run_max)
                    if start == "design_raw":
                        first_rej = float(np.mean(np.asarray(acc_b)[0] == 0.0))
                        stuck = float(np.mean(np.asarray(st.sigma) == 0.0))
                        key = f"{table.name}_{'exact' if exact else 'approx'}_{start}"
                        res_setting[key] = {
                            "first_h_rejected_fraction": first_rej,
                            "sigma_zero_fraction_after_burn": stuck,
                        }
                        print(
                            f"({phi0}, {sig0}) {key:22s} first exact-h step rejected in {first_rej:.3f} "
                            f"of chains; sigma == 0 (absorbed) in {stuck:.3f}",
                            flush=True,
                        )
                        continue
                    st = st._replace(**pc.zero_acc(st)._asdict())
                    st, (tr_k, _) = run_k(keys, st, log_e2, prior, jnp.uint32(a.burn))
                    tr_b, acc_b, tr_k = (np.asarray(x) for x in (tr_b, acc_b, tr_k))
                    stv = jax.tree.map(np.asarray, st)
                    acc = {
                        f: getattr(stv, f) / a.keep
                        for f in (
                            "acc_h",
                            "acc_cgp",
                            "acc_cs2",
                            "acc_ncphi",
                            "acc_ncms",
                            "acc_fold",
                        )
                    }
                    names = ("mu", "phi", "sigma", "h_T", "loglik")
                    ess = {
                        nm: float(np.median([pc.ess(tr_k[:, j, ll]) for ll in range(L)]))
                        * 1000
                        / a.keep
                        for j, nm in enumerate(names)
                    }
                    # burn-in diagnostics
                    dacc = np.diff(np.concatenate([np.zeros((1, L)), acc_b]), axis=0)
                    win = {
                        f"{w0}-{w1}": float(dacc[w0:w1].mean())
                        for w0, w1 in WINDOWS
                        if w1 <= a.burn
                    }
                    lo, hi = np.quantile(tr_k[:, 4], [0.01, 0.99], axis=0)
                    inband = (tr_b[:, 4] >= lo) & (tr_b[:, 4] <= hi)
                    first = np.where(inband.any(0), inband.argmax(0), a.burn)
                    if start == "design_awol":
                        ref_burn = tr_b
                        coal = None
                    elif ref_burn is not None:
                        same = np.all(tr_b[:, :4] == ref_burn[:, :4], axis=1)  # (burn, L)
                        # first sweep from which the chain equals the reference forever after
                        tail = np.flip(np.logical_and.accumulate(np.flip(same, 0), 0), 0)
                        cs = np.where(tail.any(0), tail.argmax(0), a.burn)
                        coal = {
                            "median": float(np.median(cs)),
                            "max": int(cs.max()),
                            "fraction_coalesced": float(np.mean(tail[-1])),
                        }
                    # split-R̂ with the default burn = 1000, draws = 2000 (per dataset over its C chains)
                    full = np.concatenate([tr_b, tr_k], axis=0)
                    rh = {
                        nm: float(
                            np.max(
                                [pc.split_rhat(full[1000:3000, j, rep == r].T) for r in range(R)]
                            )
                        )
                        for j, nm in enumerate(names)
                    }
                    key = f"{table.name}_{'exact' if exact else 'approx'}_{start}"
                    fold_c = acc["acc_fold"]
                    sig_q05, sig_med = np.quantile(tr_k[:, 2], [0.05, 0.5], axis=0)
                    res_setting[key] = {
                        "acceptance": {
                            f: {
                                "mean": float(v.mean()),
                                "median": float(np.median(v)),
                                "min": float(v.min()),
                                "max": float(v.max()),
                            }
                            for f, v in acc.items()
                        },
                        "fold_per_chain": {
                            "share_gt_1pct": float(np.mean(fold_c > 0.01)),
                            "share_gt_5pct": float(np.mean(fold_c > 0.05)),
                            "q95": float(np.quantile(fold_c, 0.95)),
                            "fold": fold_c,
                            "sigma_post_q05": sig_q05,
                            "sigma_post_median": sig_med,
                        },
                        "rej_run_max_sampling": int(stv.rej_run_max.max()),
                        "rej_run_max_burn": int(rej_burn.max()),
                        "ess_per_1000": ess,
                        "burn_acc_h_windows": win,
                        "sweeps_to_loglik_band": {
                            "median": float(np.median(first)),
                            "max": int(first.max()),
                        },
                        "sweeps_to_coalescence_with_design_awol": coal,
                        "max_split_rhat_burn1000_draws2000": rh,
                        "post_mean_vs_truth": {
                            "mu": float(tr_k[:, 0].mean()),
                            "phi": float(tr_k[:, 1].mean()),
                            "sigma": float(tr_k[:, 2].mean()),
                        },
                    }
                    s_ = res_setting[key]
                    print(
                        f"({phi0}, {sig0}) {key:22s} acc_h {s_['acceptance']['acc_h']['mean']:.4f} "
                        f"(min {s_['acceptance']['acc_h']['min']:.4f}) ncms {s_['acceptance']['acc_ncms']['mean']:.4f} "
                        f"cgp {s_['acceptance']['acc_cgp']['mean']:.3f} "
                        f"fold {s_['acceptance']['acc_fold']['mean']:.4f} "
                        f"(med {s_['acceptance']['acc_fold']['median']:.4f}, "
                        f"max {s_['acceptance']['acc_fold']['max']:.4f}, "
                        f">1% in {s_['fold_per_chain']['share_gt_1pct']:.2f}) "
                        f"rejrun {s_['rej_run_max_sampling']}/{s_['rej_run_max_burn']} "
                        f"ESS/1k mu {ess['mu']:.0f} phi {ess['phi']:.0f} sig {ess['sigma']:.0f} hT {ess['h_T']:.0f} "
                        f"band@ {s_['sweeps_to_loglik_band']['median']:.0f}/{s_['sweeps_to_loglik_band']['max']} "
                        f"Rhat_max {max(rh.values()):.3f}",
                        flush=True,
                    )
        out["settings"][f"phi{phi0}_sigma{sig0}"] = res_setting
    out["wall_s"] = time.perf_counter() - t0
    path = pc.save_result(a.out, "sv_acceptance", out)
    print("wrote", path)


if __name__ == "__main__":
    main()
