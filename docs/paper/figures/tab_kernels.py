"""TabE3 (Appendix E): kernel microbenchmarks of the BENCH-P2 sweep at 80 equation-chain lanes.

Sources: ``kernels`` and ``kernel_machine`` of ``tests/bench/baseline.json`` (committed in d490730,
measured at deb3929 on a non-idle Apple M5 Max) and, for the production function behind each
kernel and its DESIGN section 10.3 budget line, the table in the module docstring of
``tests/bench/bench_kernels.py`` at the measured commit. Evidence tag [R]. XLA's flop and byte
counts are per call of the compiled executable, as XLA reports them; for ``sweep`` that is one call
of the 200-sweep executable, shown undivided and marked (``bench_kernels.table()`` divides it by
200, which gives 0.002 GFLOP per sweep, below the packed Gram alone). Units: GFLOP = 1e9 flops,
MiB = 2^20 bytes.

Asserted against the appendix text: the packed Gram has half the flops of the batched oracle and is
about twice as fast; the packed Gram, the precision draw and the SV core take 3.44 ms of the
3.87 ms sweep.

Usage: tab_kernels.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import re

import tab_common as tc

NAME = "tab_kernels"
BASELINE = "tests/bench/baseline.json"
MODULE = "tests/bench/bench_kernels.py"


def tex_cell(s: str) -> str:
    """A docstring cell: ``x`` -> \\code{x}, the rest as text."""
    parts = re.split(r"``(.+?)``", s)
    out = []
    for i, p in enumerate(parts):
        out.append(tc.code(p) if i % 2 else tc.tex(p))
    return "".join(out).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    commit = tc.git("log", "-1", "--format=%h", "--", BASELINE)
    b = tc.git_show_json(commit, BASELINE)
    k, mach = b["kernels"], b["kernel_machine"]
    src = tc.git_show(mach["commit"], MODULE)
    doc = src.split('"""')[1]
    rows = {tc.grab(r["kernel"], r"``(\w+)``"): r for r in tc.md_table(doc)}
    budget_ms = tc.literal(src, "BUDGET_MS")
    assert set(rows) == set(k), (set(rows), set(k))

    # the appendix's arithmetic (app:performance)
    r_flops = k["gram_batched"]["cost"]["flops"] / k["gram_packed"]["cost"]["flops"]
    r_time = k["gram_batched"]["ms_min"] / k["gram_packed"]["ms_min"]
    assert 1.9 < r_flops < 2.1 and 2.0 < r_time < 2.5, (r_flops, r_time)
    three = k["gram_packed"]["ms_min"] + k["precision_draw"]["ms_min"] + k["sv_core"]["ms_min"]
    assert f"{three:.2f}" == "3.44" and f"{k['sweep']['ms_min']:.2f}" == "3.87", three
    assert budget_ms["sv_indicators+sv_awol"] == 1.5

    lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.30\linewidth}rrrrrr@{}}",
             r"\toprule",
             r" & & \multicolumn{2}{c}{ms} & & & & Budget \\ \cmidrule(lr){3-4}",
             r"Kernel & Production code & min & median & GFLOP & MiB acc. & Temp MiB & ms \\",
             r"\midrule"]  # fmt: skip
    for name in k:  # the baseline's order
        r, d = k[name], rows[name]
        # XLA's cost analysis is per call of the compiled executable; the sweep executable runs
        # per_call_units sweeps per call, and its counts are shown per call (marked), not divided
        per = r.get("per_call_units", 1)
        dag = r"$^\dagger$" if per != 1 else ""
        c, mem = r.get("cost", {}), r.get("memory", {})
        budget = d["§10.3 budget line (ms)"].strip()
        if budget.startswith("}"):  # the indicators and AWOL share one budget line
            budget = (f"{budget_ms['sv_indicators+sv_awol']:g} (with AWOL)" if name == "sv_indicators"
                      else "(with indicators)")  # fmt: skip
        budget = tc.tex(budget.replace("≈ ", "about ").replace("—", "---"))
        lines.append(
            f"{tc.code(name)} & {tex_cell(d['production code'])} & {r['ms_min']:.3f} & "
            f"{r['ms_median']:.3f} & {c['flops'] / 1e9:.3f}{dag} & "
            f"{c['bytes accessed'] / 2**20:.1f}{dag} & {mem.get('temp_size_in_bytes', 0) / 2**20:.1f} & "
            f"{budget} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    reps = sorted({r["reps"] for n, r in k.items() if n != "sweep"})
    foot = (f"Recorded (not re-measured): {tc.code(BASELINE)} \\code{{kernels}}, committed in "
            f"{tc.code(commit)}, measured at commit {tc.code(mach['commit'])} on "
            f"{tc.tex(mach['date'][:10])}, {tc.tex(mach['cpu'])}, {mach['backend']} backend, "
            f"{mach['devices']} device, one-minute load average {mach['loadavg'][0]:.1f} (not idle). "
            f"Each kernel is the production function vmapped over the 80 lanes and AOT-compiled; "
            f"times are the minimum and median over {reps[0]} timed calls ({k['sweep']['reps']} calls "
            f"of {k['sweep']['per_call_units']} sweeps for \\code{{sweep}}, given per sweep). "
            f"GFLOP and MiB accessed are XLA's \\code{{cost_analysis()}}, temporary MiB its "
            f"\\code{{memory_analysis()}}, per call of the compiled executable ($^\\dagger$ the "
            f"{k['sweep']['per_call_units']}-sweep executable, as reported, not divided by the number "
            f"of sweeps). Budget: the DESIGN \\S10.3 line in "
            f"{tc.code(MODULE)}. The packed Gram, the precision draw and the SV core take "
            f"{three:.2f}\\,ms of the {k['sweep']['ms_min']:.2f}\\,ms sweep.")  # fmt: skip
    body = tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    tc.write(NAME, body, generator="tab_kernels.py", status="[R] recorded baseline",
             sources=[f"main repo {BASELINE} kernels, kernel_machine at {commit}",
                      f"main repo {MODULE} at {mach['commit']} (docstring table, BUDGET_MS)"])  # fmt: skip


if __name__ == "__main__":
    main()
