"""Phase 3 mixing pilot (P3 DESIGN §8; WP36): bulk ESS per 1,000 sweeps of every primary
quantity and the production gate on prefixes of one long run.

For each configuration (``phase3_common.build_model``) and size (``qd7``: FRED-QD medium,
``n = 7, p = 4, T = 200``; ``bp2``: BENCH-P2's shape and data, ``n = 20, p = 4, T = 300``) and each
seed, one run of 4 chains × ``--draws`` kept draws after ``--burn`` sweeps is recorded with the
production executables (``phase3_common.run_chain``). The gate of P2 §8.2 (unchanged) is then
evaluated on the first ``D`` kept draws for every ``D`` of the ladder: those draws are exactly the
draws of a fit with ``draws = D`` and the same seed (the sweep keys depend only on the sweep
index; ``--check`` verifies this against ``BVAR.fit``). Results: one JSON per (config, size, seed)
in ``--out`` with the gate ladder, the ESS table (every monitored quantity and the extra
quantities: GL group functions, CommonSV ``mean h``), acceptance rates, timings, the machine
fingerprint and the load before and after (ESS does not depend on the load; the timings here are
reported only as the harness's own and are not the §10 cost measurements).

Usage::

    python -I benchmarks/pilot/phase3_mixing.py --out DIR --size qd7 --configs hs_sv,dl_const \\
        --seeds 0,1,2,3,4 --draws 50000
    python -I benchmarks/pilot/phase3_mixing.py --out DIR --check     # harness = BVAR.fit
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import phase3_common as pc  # noqa: E402


def run_one(config: str, size: str, seed: int, draws: int, burn: int, out: Path) -> dict:
    data = pc.size_data(size)
    model = pc.build_model(config, draws=draws, burn=burn, seed=seed)
    before = pc.load_snapshot()
    run = pc.run_chain(
        model, data, config=config, size=size, progress=lambda s: print(s, flush=True)
    )
    after = pc.load_snapshot()
    ladder = pc.gate_ladder(run, [D for D in pc.LADDER if D <= draws])
    table_full = pc.ess_table(run)
    tables = {str(D): pc.ess_table(run, D) for D in pc.LADDER if D < draws}
    payload = {
        "config": config,
        "size": size,
        "seed": seed,
        "draws": draws,
        "burn": burn,
        "chains": int(run.settings.chains),
        "model": repr(model),
        "static": {
            "mean": repr(run.static.mean),
            "coef": repr(run.static.coef),
            "vol": repr(run.static.vol),
            "shapes": repr(run.static.shapes),
        },
        "n_primary": len(run.spec.primary),
        "gate": ladder,
        "ess": table_full,
        "ess_at": tables,
        "chain_block_means": pc.chain_block_means(run),
        "acceptance": {k: v.tolist() for k, v in run.acceptance.items()},
        "timing": run.timing,
        "load_before": before,
        "load_after": after,
        "machine": pc.fingerprint(),
    }
    pc.save_json(out, f"mix_{size}_{config}_s{seed}", payload)
    return payload


def check(out: Path) -> dict:
    """The harness against ``BVAR.fit`` at ``qd7`` (4 × 2,000 and 4 × 3,000 after 1,000): the
    shorter fit's monitor stream is a prefix of the longer one's, the harness's stream equals
    the longer fit's, and the gate on the harness's prefixes equals each fit's ``converged``."""
    import warnings

    data = pc.size_data("qd7")
    results = {}
    for config in ("hs_sv", "dl_const", "csv_conj", "ssvs_beta_const"):
        fits = {}
        for draws in (2000, 3000):
            model = pc.build_model(config, draws=draws, burn=1000, seed=7)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fits[draws] = model.fit(data)
        run = pc.run_chain(
            pc.build_model(config, draws=3000, burn=1000, seed=7), data, config=config, extras=False
        )
        m2, m3 = fits[2000].posterior["monitor"], fits[3000].posterior["monitor"]
        prefix, harness = {}, {}
        for name in run.mon:
            a2 = np.asarray(m2[name].values)
            a3 = np.asarray(m3[name].values)
            prefix[name] = float(np.max(np.abs(a3[:, :2000] - a2))) if a2.size else 0.0
            harness[name] = float(np.max(np.abs(a3 - run.mon[name]))) if a3.size else 0.0
        g2 = pc.gate(run.mon, run.spec, run.settings, 2000)["pass"]
        g3 = pc.gate(run.mon, run.spec, run.settings, 3000)["pass"]
        results[config] = {
            "prefix_max_abs_diff": prefix,
            "harness_max_abs_diff": harness,
            "prefix_bitwise": all(v == 0.0 for v in prefix.values()),
            "harness_bitwise": all(v == 0.0 for v in harness.values()),
            "gate_2000": g2,
            "fit_2000_converged": bool(fits[2000].converged),
            "gate_3000": g3,
            "fit_3000_converged": bool(fits[3000].converged),
        }
        print(config, results[config]["prefix_bitwise"], results[config]["harness_bitwise"],
              g2, fits[2000].converged, g3, fits[3000].converged, flush=True)  # fmt: skip
    payload = {"check": results, "machine": pc.fingerprint()}
    pc.save_json(out, "mix_check", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="qd7")
    ap.add_argument("--configs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--draws", type=int, default=50_000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        check(args.out)
        return 0
    seeds = [int(s) for s in args.seeds.split(",") if s]
    status = 0
    for config in [c for c in args.configs.split(",") if c]:
        for seed in seeds:
            name = args.out / f"mix_{args.size}_{config}_s{seed}.json"
            if name.exists():
                print(f"skip {name.name} (exists)", flush=True)
                continue
            t0 = time.time()
            try:
                res = run_one(config, args.size, seed, args.draws, args.burn, args.out)
                gates = " ".join(
                    f"{g['D']}:{'P' if g['pass'] else 'F'}({g['rhat_max']:.3f},{g['ess_min']:.0f})"
                    for g in res["gate"]
                )
                print(f"[{config} {args.size} s{seed}] {time.time() - t0:.0f} s  {gates}",
                      flush=True)  # fmt: skip
            except Exception:
                status = 1
                traceback.print_exc()
                (args.out / f"mix_{args.size}_{config}_s{seed}.err").write_text(
                    traceback.format_exc()
                )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
