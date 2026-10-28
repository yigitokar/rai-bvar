"""Summarise the Phase 3 pilot's JSON results into the tables of ``docs/phase3/pilot.md`` (WP36).

* mixing (``mix_<size>_<config>_s<seed>.json``): the gate ladder (seeds passing at each ``D``) and
  the recommended budget by the rule of ``pilot.md`` §1; ESS per 1,000 sweeps of the primary
  fields (per field the slowest quantity, median over seeds) and of the extra quantities; the
  slowest primary quantity; lag-1 autocorrelations; CommonSV acceptance rates;
* costs (``cost_<size>_<config>[_cholesky].json``): ms per step / phase / sweep, flagged when a
  block was timed above the load threshold;
* GK (``gk_sweeps*.json``, ``gk_origins*.json``): the Q26 estimate (``q26``).

No JAX import; pure post-processing. Usage::

    python -I benchmarks/pilot/phase3_summarise.py DIR [--md out.md] [--json out.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

LADDER = (2_000, 5_000, 10_000, 20_000, 50_000)
#: The smallest window an arm can fit: semiautomatic SSVS in the reduced form needs
#: ``T_eff > k + n + 1 = 107`` at ``n = 21, p = 4`` (D-44), so GK's first 31 origins have no fit.
MIN_T = {"red_ssvs_beta_sv": 108}
FIELD_RE = re.compile(r"^([A-Za-z_.0-9]+?)(\[.*\])?$")


def _field(label: str) -> str:
    m = FIELD_RE.match(label)
    return m.group(1) if m else label


def _med(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def _flat_mean(v: Any) -> float:
    if isinstance(v, list):
        xs = [_flat_mean(x) for x in v]
        return statistics.mean(xs) if xs else float("nan")
    return float(v)


def load(dirs: list[Path], prefix: str) -> list[dict[str, Any]]:
    out = []
    for d in dirs:
        for p in sorted(d.glob(f"{prefix}*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------- mixing


def recommend(passes: dict[int, list[bool]]) -> int | None:
    """The smallest ladder ``D`` at which at least ⌈R/2⌉ seeds pass and every larger evaluated
    ``D`` also has a passing median (pilot.md §1); ``None`` if no evaluated ``D`` qualifies."""
    ok = {}
    for D, flags in passes.items():
        R = len(flags)
        ok[D] = R > 0 and sum(flags) >= math.ceil(R / 2)
    Ds = sorted(ok)
    for i, D in enumerate(Ds):
        if all(ok[E] for E in Ds[i:]):
            return D
    return None


def mixing_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by[(r["size"], r["config"])].append(r)
    out: dict[str, Any] = {}
    for (size, config), rs in sorted(by.items()):
        rs.sort(key=lambda r: r["seed"])
        passes: dict[int, list[bool]] = defaultdict(list)
        detail: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            for g in r["gate"]:
                passes[g["D"]].append(bool(g["pass"]))
                detail[g["D"]].append(
                    {k: g[k] for k in ("rhat_max", "ess_min", "worst", "n_fail", "n_primary")}
                )
        fields: dict[str, list[float]] = defaultdict(list)  # per seed: min ESS/1k over the field
        slow_name: list[str] = []
        slow_val: list[float] = []
        extras: dict[str, list[float]] = defaultdict(list)
        acf: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            per_field: dict[str, float] = {}
            per_field_acf: dict[str, float] = {}
            worst = (None, math.inf)
            for name, row in r["ess"].items():
                f = _field(name)
                key = f
                v = row["ess_per_1k"]
                if row["primary"]:
                    if v < worst[1]:
                        worst = (name, v)
                if row["primary"] or row["extra"] or f in ("gl_a_idx", "log_prior", "gl_log_tau",
                                                           "common_h_0", "common_h_mid"):  # fmt: skip
                    per_field[key] = min(per_field.get(key, math.inf), v)
                    per_field_acf[key] = max(per_field_acf.get(key, -math.inf), row["acf1"])
            for k, v in per_field.items():
                (extras if k.startswith(("gl.", "csv.")) else fields)[k].append(v)
                acf[k].append(per_field_acf[k])
            slow_name.append(worst[0])
            slow_val.append(worst[1])
        acc: dict[str, float] = {}
        for r in rs:
            for k, v in r.get("acceptance", {}).items():
                acc.setdefault(k, []).append(_flat_mean(v))  # type: ignore[arg-type]
        out[f"{size}/{config}"] = {
            "size": size,
            "config": config,
            "seeds": [r["seed"] for r in rs],
            "draws": min(r["draws"] for r in rs),
            "n_primary": rs[0]["n_primary"],
            "passes": {str(D): passes[D] for D in sorted(passes)},
            "gate_detail": {str(D): detail[D] for D in sorted(detail)},
            "recommended": recommend(dict(passes)),
            "ess_per_1k_fields": {k: _med(v) for k, v in fields.items()},
            "ess_per_1k_extras": {k: _med(v) for k, v in extras.items()},
            "acf1_max": {k: _med(v) for k, v in acf.items()},
            "slowest_primary": max(set(slow_name), key=slow_name.count),
            "slowest_primary_ess_per_1k": _med(slow_val),
            "acceptance": {k: _med(v) for k, v in acc.items()},  # type: ignore[arg-type]
            "harness_ms_per_sweep": _med([r["timing"]["ms_per_sweep_scan"] for r in rs]),
            "load_1min": [r["load_before"]["loadavg"][0] for r in rs],
        }
    return out


def mixing_markdown(summary: dict[str, Any], size: str) -> str:
    rows = [f"| configuration | primaries | {' | '.join(f'{D:,}' for D in LADDER)} | "
            "recommended | slowest primary (ESS/1k) |",
            "| :--- | ---: | " + " | ".join("---:" for _ in LADDER) + " | ---: | :--- |"]  # fmt: skip
    for s in summary.values():
        if s["size"] != size:
            continue
        cells = []
        for D in LADDER:
            flags = s["passes"].get(str(D))
            cells.append("—" if not flags else f"{sum(flags)}/{len(flags)}")
        rec = s["recommended"]
        rec_s = f"{rec:,}" if rec else "> " + f"{s['draws']:,}"
        rows.append(f"| `{s['config']}` | {s['n_primary']} | {' | '.join(cells)} | {rec_s} | "
                    f"`{s['slowest_primary']}` ({s['slowest_primary_ess_per_1k']:.1f}) |")  # fmt: skip
    return "\n".join(rows)


GL_COLS = ("loglik", "B_own1", "log_lam", "mu", "phi", "sigma", "h_T", "group scale",
           "gl_a_idx", "gl.mean_log_v", "log_prior")  # fmt: skip
CSV_COLS = ("loglik", "B_own1", "log_lam", "common_h_T", "common_rho", "common_sigma",
            "csv.mean_h", "log_prior")  # fmt: skip


def _cell(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")


def fields_markdown(summary: dict[str, Any], size: str, csv: bool = False) -> str:
    """ESS per 1,000 sweeps (median over seeds of the slowest quantity of each field) at the
    longest run; ``group scale`` is HS ``gl_log_zeta``, NG/R2D2 ``gl_log_xi`` or SSVS ``gl_p``."""
    cols = CSV_COLS if csv else GL_COLS
    head = "| configuration | " + " | ".join(f"`{c}`" if c != "group scale" else c for c in cols)
    rows = [head + " |", "| :--- |" + " ---: |" * len(cols)]
    for s in summary.values():
        if s["size"] != size or s["config"].startswith("csv_") != csv:
            continue
        f = {**s["ess_per_1k_fields"], **s["ess_per_1k_extras"]}
        grp = next((f[k] for k in ("gl_log_zeta", "gl_log_xi", "gl_p") if k in f), None)
        cells = [_cell(grp) if c == "group scale" else _cell(f.get(c)) for c in cols]
        rows.append(f"| `{s['config']}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- costs


def cost_summary(costs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for c in costs:
        key = f"{c['size']}/{c['config']}" + (
            f"/{c['const_path_setting']}" if c.get("const_path_setting", "auto") != "auto" else ""
        )
        out[key] = {
            name: {"ms_min": b["ms_min"], "ms_median": b["ms_median"],
                   "contaminated": bool(b.get("contaminated")),
                   "load": [b["load_before"]["loadavg"][0], b["load_after"]["loadavg"][0]]}
            for name, b in c["blocks"].items()
        }  # fmt: skip
        out[key]["_static"] = c.get("static", {})
    return out


# --------------------------------------------------------------------------- Q26


def q26(gk_sweeps: list[dict[str, Any]], gk_origins: list[dict[str, Any]],
        budgets: tuple[int, ...] = (5_000, 10_000, 20_000), warm_burn: int = 250,
        burn: int = 1000) -> dict[str, Any]:  # fmt: skip
    """The GK acceptance run's compute (P3 §9.10): 160 expanding-window origins 1980Q1–2019Q4
    (``T_eff = 77 + t``, ``t = 0..159``), one fit per origin and arm; the first origin of each arm
    cold (``burn``), the others warm-started (``warm_burn``). ms per sweep is interpolated
    linearly in ``T_eff`` between the measured windows; the per-origin cost beyond the sweeps
    (``overhead(D) = a + b·D``) is fitted to the end-to-end origin timings."""
    sweeps: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for doc in gk_sweeps:
        for r in doc["sweeps"]:
            if "error" not in r:
                sweeps[r["arm"]].append((r["T_eff"], r["ms_per_sweep_min"]))

    def ms(arm: str, T: int) -> float:
        pts = sorted(sweeps[arm])
        if T <= pts[0][0]:
            return pts[0][1]
        for (t0, m0), (t1, m1) in itertools.pairwise(pts):
            if t0 <= T <= t1:
                return m0 + (m1 - m0) * (T - t0) / (t1 - t0)
        return pts[-1][1]

    # overhead: wall − (burn + D)·ms(T) over the timed origins, linear in D
    ov_pts = []
    for doc in gk_origins:
        for r in doc["origins"]:
            if r["arm"] in sweeps:
                sweep_s = (burn + r["draws"]) * ms(r["arm"], r["T_eff"]) / 1000.0
                ov_pts.append((r["draws"], r["wall_s"] - sweep_s, r["arm"], r["T_eff"]))
    if len({p[0] for p in ov_pts}) >= 2:
        xs = [p[0] for p in ov_pts]
        ys = [p[1] for p in ov_pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
            (x - mx) ** 2 for x in xs
        )
        a = my - b * mx
    else:
        a, b = (ov_pts[0][1] if ov_pts else 0.0), 0.0
    origins = list(range(77, 77 + 160))
    table = {}
    for D in budgets:
        per_arm = {}
        for arm in sorted(sweeps):
            secs = 0.0
            feasible = [T for T in origins if T >= MIN_T.get(arm, 0)]
            for j, T in enumerate(feasible):
                nb = burn if j == 0 else warm_burn
                secs += (nb + D) * ms(arm, T) / 1000.0 + a + b * D
            per_arm[arm] = secs / 3600.0
        table[str(D)] = {"per_arm_h": per_arm, "total_h": sum(per_arm.values())}
    table["feasible_origins"] = {a: sum(T >= MIN_T.get(a, 0) for T in origins) for a in sweeps}
    return {
        "origins": len(origins),
        "overhead_s": {"a": a, "b_per_draw": b, "points": ov_pts},
        "ms_per_sweep": {arm: sorted(v) for arm, v in sweeps.items()},
        "budgets": table,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    mix = mixing_summary(load(args.dirs, "mix_qd7") + load(args.dirs, "mix_bp2")
                         + load(args.dirs, "mix_sim"))  # fmt: skip
    costs = cost_summary(load(args.dirs, "cost_"))
    gks, gko = load(args.dirs, "gk_sweeps"), load(args.dirs, "gk_origins")
    result = {"mixing": mix, "costs": costs}
    if gks:
        result["q26"] = q26(gks, gko)
    md = []
    for size in ("qd7", "bp2", "sim20x120", "md7", "qd21", "sim21x236"):
        if any(s["size"] == size for s in mix.values()):
            md.append(f"### {size}: gate ladder\n\n" + mixing_markdown(mix, size) + "\n")
            md.append(f"### {size}: ESS per 1,000 sweeps (GL)\n\n" + fields_markdown(mix, size))
            md.append(f"\n### {size}: ESS per 1,000 sweeps (CommonSV)\n\n"
                      + fields_markdown(mix, size, csv=True) + "\n")  # fmt: skip
    text = "\n".join(md)
    if args.md:
        args.md.write_text(text)
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
