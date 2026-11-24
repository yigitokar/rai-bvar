"""Compute inputs of the Gruber–Kastner acceptance run (P3 DESIGN §9.10; Q26; WP36).

The GK run fits, at every expanding-window origin 1980Q1–2019Q4 (160 origins; one fit per origin
serves ``h = 1`` and ``h = 4``), the reduced-form VAR(4) with ``SV()`` at ``n = 21`` under eight
arms: Minnesota-EB, ``Horseshoe`` (olcl-lagwise, global), ``NormalGamma`` (olcl-lagwise, global),
``R2D2``, ``DirichletLaplace`` and ``SSVS(p = Beta(1, 1))``, with ``scale="none"`` and prior mean
0 (§9.10). With the sample start 1960Q1 the first origin has ``T_eff = 77`` and the last 236.

* ``--sweeps``: ms per sweep of the runner's ``burn`` executable (4 chains, ``vmap``, one device;
  minimum of ``--reps`` × ``--block`` sweeps on a state burned in for ``--warm`` sweeps) for every
  arm at ``T_eff ∈ {77, 156, 236}`` (``T_pad`` 96, 160, 256), and the compile time of the burn
  executable, on ``simulate_var_sv(21, 4, T_eff, seed = 0)`` (BENCH-P2's generator at ``n = 21``);
  each block load-gated, with the load and top processes before and after.
* ``--origins``: the end-to-end wall time of one backtest origin (fit, forecast of ``h = 1..4``
  with 1,000 draws and the integrated LPD; ``evaluation.backtest._run_origin``, as
  ``estimate_cost`` times it) at the first and the last window, cold (1,000 burn-in sweeps), for
  the listed arms and budgets, to measure the per-origin cost beyond the sweeps (resolution, EB,
  compilation, diagnostics, storage, forecast, scoring).

Results: ``DIR/gk_sweeps.json`` and ``DIR/gk_origins.json``; ``docs/phase3/pilot.md`` §6 turns them
into the Q26 estimate.

Usage::

    python -I benchmarks/pilot/phase3_gk.py --out DIR --sweeps
    python -I benchmarks/pilot/phase3_gk.py --out DIR --origins --arms red_hs_sv --draws 2000,10000
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import phase3_common as pc  # noqa: E402

ARMS = (
    "red_mn_eb_sv",
    "red_hs_sv",
    "red_hs@global_sv",
    "red_ng_sv",
    "red_ng@global_sv",
    "red_r2d2_sv",
    "red_dl_sv",
    "red_ssvs_beta_sv",
)
N_GK = 21
T_EFFS = (77, 156, 236)
START = "1960Q1"


def gk_data(T_eff: int) -> object:
    """``simulate_var_sv(21, 4, T_eff, 0)`` levels as quarterly ``VARData`` from 1960Q1."""
    import pandas as pd

    import rai_bvar as rb

    y = pc.phase2_perf().bench_levels(N_GK, pc.LAGS, T_eff, 0)
    idx = pd.period_range(START, periods=y.shape[0], freq="Q")
    frame = pd.DataFrame(y, index=idx, columns=[f"y{i}" for i in range(N_GK)])
    return rb.VARData.from_frame(frame, 1)


def sweeps(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp

    import rai_bvar as rb
    from rai_bvar.engines.mcmc import _blas, _jax, keys
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.runner import make_runner

    rows = []
    for arm in args.arms.split(","):
        # the idle gate once per arm, before this process loads the machine itself
        gate = pc.wait_idle(args.gate, args.wait) if args.gate > 0 else {"gate_open": None}
        gate_load = gate["load"]["loadavg"][0] if "load" in gate else None
        for T_eff in T_EFFS:
            data = gk_data(T_eff)
            model = pc.build_model(arm, draws=1000, burn=args.warm, seed=3)
            before = pc.load_snapshot()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = resolve(model, data, rb.Runtime())
            except ValueError as err:  # e.g. semiautomatic SSVS on a short window (D-44)
                rows.append({"arm": arm, "T_eff": T_eff, "error": str(err)})
                print(f"{arm} T={T_eff}: {err}", flush=True)
                continue
            with _jax.x64(), _blas.single_thread(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fns = make_runner(res.static)
                d = jax.tree.map(jnp.asarray, res.data)
                p = jax.tree.map(jnp.asarray, res.prior)
                cache = fns.prepare(d, p)
                st = fns.init(keys.init_keys(3, res.salt, 4), d, p, cache)
                ck = keys.chain_keys(keys.fit_root(3, res.salt), 4)
                t0 = time.perf_counter()
                st = jax.block_until_ready(
                    fns.burn(ck, st, d, p, cache, np.uint32(0), np.int32(args.warm))
                )
                first_s = time.perf_counter() - t0  # compile + warm sweeps
                times = []
                for r in range(args.reps):
                    t = time.perf_counter()
                    it0 = np.uint32(args.warm + r * args.block)
                    st = jax.block_until_ready(
                        fns.burn(ck, st, d, p, cache, it0, np.int32(args.block))
                    )
                    times.append(1000.0 * (time.perf_counter() - t) / args.block)
            after = pc.load_snapshot()
            row = {
                "arm": arm,
                "T_eff": T_eff,
                "T_pad": int(res.static.shapes.T_pad),
                "ms_per_sweep_min": min(times),
                "ms_per_sweep_all": times,
                "compile_plus_warm_s": first_s,
                "compile_s_est": first_s - args.warm * min(times) / 1000.0,
                "static": repr(res.static.coef),
                "load_before": before,
                "load_after": after,
                "gate": gate,
                "contaminated": pc.contaminated(
                    before, after, args.gate or 2.0, gate_load=gate_load
                ),
            }
            rows.append(row)
            print(f"{arm} T={T_eff}: {row['ms_per_sweep_min']:.2f} ms/sweep, compile ≈ "
                  f"{row['compile_s_est']:.1f} s, load {before['loadavg'][0]:.2f} → "
                  f"{after['loadavg'][0]:.2f}", flush=True)  # fmt: skip
    out = {"sweeps": rows, "machine": pc.fingerprint()}
    pc.save_json(args.out, "gk_sweeps" + (f"_{args.tag}" if args.tag else ""), out)
    return out


def origins(args: argparse.Namespace) -> dict:
    import pandas as pd

    from rai_bvar.evaluation.backtest import Expanding, _run_origin

    data = gk_data(241)  # rows 1960Q1–2021Q1: every h ≤ 4 outcome of the timed origins exists
    rows = []
    for arm in args.arms.split(","):
        for draws in [int(x) for x in args.draws.split(",")]:
            for origin in ("1980Q1", "2018Q4"):
                model = pc.build_model(arm, draws=draws, burn=1000, seed=0)
                gate = pc.wait_idle(args.gate, args.wait) if args.gate > 0 else {"gate_open": None}
                gate_load = gate["load"]["loadavg"][0] if "load" in gate else None
                before = pc.load_snapshot()
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _run_origin(model, data, data, pd.Period(origin, freq="Q"), Expanding(), 4,
                                1000, 0, None, "integrated")  # fmt: skip
                wall = time.perf_counter() - t0
                after = pc.load_snapshot()
                rows.append({
                    "arm": arm, "draws": draws, "origin": origin, "wall_s": wall,
                    "T_eff": len(data.upto(origin)) - pc.LAGS, "load_before": before,
                    "load_after": after, "gate": gate,
                    "contaminated": pc.contaminated(before, after, args.gate or 2.0,
                                                    gate_load=gate_load),
                })  # fmt: skip
                print(f"{arm} D={draws} {origin}: {wall:.1f} s (load {before['loadavg'][0]:.2f} "
                      f"→ {after['loadavg'][0]:.2f})", flush=True)  # fmt: skip
    out = {"origins": rows, "machine": pc.fingerprint()}
    pc.save_json(args.out, "gk_origins" + (f"_{args.tag}" if args.tag else ""), out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sweeps", action="store_true")
    ap.add_argument("--origins", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--draws", default="2000,10000")
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--block", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--wait", type=float, default=1800.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.sweeps:
        sweeps(args)
    if args.origins:
        origins(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
