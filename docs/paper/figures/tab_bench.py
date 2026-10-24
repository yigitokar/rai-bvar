"""TabE2 (Appendix E): BENCH-P2 on the production engine.

Sources, evidence tag [R] (recorded by ``benchmarks/phase2_perf.py``, never re-measured here):

* ``tests/bench/baseline.json`` of the main repository as committed in d490730 (measured at commit
  deb3929 on a non-idle Apple M5 Max, ``reference: true``; OUTLINE TODO 9 asks for a re-measurement
  on an idle machine on the release commit);
* ``benchmarks/reports/bench_p2_x86_github_779957f.json`` (main 3d5a407, PR #9): the x86-64 report
  of the CI bench job on a GitHub ``ubuntu-latest`` runner (AMD EPYC 7763, 2 logical CPUs), measured
  at commit 779957f, before D-35 (e8cf9f1) and PR #8. DESIGN's WP24 amendment makes it reported,
  not gated: a shared 2-vCPU runner is not the protocol's x86 machine, so the 120 s target is
  neither required nor asserted there. Its settings must equal the baseline's.

Upper panel: medians over three cold and three warm fresh-process fits (``summary``), one column per
machine; lower panel: bulk ESS and ESS per second by quantity group (medians over the warm runs).

Checks against the numbers the appendix quotes (52.3 s / 49.6 s, 3.46 s / 1.00 s, 3.90 ms per
sweep, 3,043 MiB, sigma and phi minimum ESS/s 10.2 and 11.2, converged false in all six runs) are
asserted, so a changed baseline cannot silently disagree with the text.

Usage: tab_bench.py [--outdir DIR] [--baseline-commit COMMIT]
"""

from __future__ import annotations

import argparse

import tab_common as tc

NAME = "tab_bench"
BASELINE = "tests/bench/baseline.json"
X86 = "benchmarks/reports/bench_p2_x86_github_779957f.json"

GROUPS = [  # (key in summary.ess, label)
    ("loglik", r"log-likelihood"),
    ("h_T", r"$h_{i,T}$"),
    ("mu", r"$\mu_i$"),
    ("sigma", r"$\sigma_i$"),
    ("phi", r"$\phi_i$"),
    ("B_own1", r"own first-lag coefficients"),
    ("primary", r"primary monitor set (all)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--baseline-commit", default=None,
                    help="read the baseline at this commit (default: the last commit that changed it)")  # fmt: skip
    args = ap.parse_args()
    tc.apply_outdir(args)
    commit = args.baseline_commit or tc.git("log", "-1", "--format=%h", "--", BASELINE)
    b = tc.git_show_json(commit, BASELINE)
    s, m = b["summary"], b["machine"]
    st = b["settings"]

    # the numbers quoted in sections/appendix.tex (app:performance) and sections/10_performance.tex
    assert f"{s['wall_cold_s']:.1f}" == "52.3" and f"{s['wall_warm_s']:.1f}" == "49.6", s
    assert f"{s['compile_cold_s']:.2f}" == "3.46" and f"{s['compile_warm_s']:.2f}" == "1.00", s
    assert f"{s['sweep_ms_min3x1000']:.2f}" == "3.90", s
    assert round(s["peak_rss_mb"]) == 3043, s
    assert f"{s['ess']['sigma']['ess_per_s_min']:.1f}" == "10.2"
    assert f"{s['ess']['phi']['ess_per_s_min']:.1f}" == "11.2"
    assert s["converged"] == [False] * 6, s["converged"]

    x = tc.git_show_json("HEAD", X86)  # main repository HEAD (4ac7969)
    xs, xm = x["summary"], x["machine"]
    assert x["settings"] == st, (x["settings"], st)
    # the numbers quoted in sections/10_performance.tex and app:performance
    assert f"{xs['wall_cold_s']:.1f}" == "228.9" and f"{xs['wall_warm_s']:.1f}" == "198.4", xs
    assert f"{xs['runner_ms_per_sweep']:.1f}" == "17.2" and f"{xs['sweep_ms_min3x1000']:.1f}" == "16.8", xs
    assert xs["converged"] == [False] * 6 and xm["logical_cpus"] == 2 and xm["commit"] == "779957f", (xs, xm)

    load, xload = s["loadavg_1min"], xs["loadavg_1min"]

    def both(fmt):
        return f"{fmt(s)} & {fmt(xs)}"

    top = [
        ("Wall time, cold cache (median of 3)", both(lambda d: f"{d['wall_cold_s']:.1f}\\,s")),
        ("Wall time, warm cache (median of 3)", both(lambda d: f"{d['wall_warm_s']:.1f}\\,s")),
        ("Compilation, cold / from the persistent cache",
         both(lambda d: f"{d['compile_cold_s']:.2f}\\,s / {d['compile_warm_s']:.2f}\\,s")),
        ("Run time, wall minus compilation (median of 6)", both(lambda d: f"{d['run_s']:.1f}\\,s")),
        ("After the runner: assembly, diagnostics (median of 6)", both(lambda d: f"{d['post_runner_s']:.2f}\\,s")),
        ("Per sweep: min.\\ of $3\\times1000$ compiled burn-in sweeps (median of 6)",
         both(lambda d: f"{d['sweep_ms_min3x1000']:.2f}\\,ms")),
        (f"Per sweep: runner, over the {tc.thousands(st['draws'])} kept sweeps (median of 6)",
         both(lambda d: f"{d['runner_ms_per_sweep']:.2f}\\,ms")),
        ("Peak resident set (median of 6)", both(lambda d: f"{tc.thousands(d['peak_rss_mb'])}\\,MiB")),
        (r"\code{converged} (R-hat $<1.01$, bulk ESS $>400$, primary set)",
         both(lambda d: f"false in {d['converged'].count(False)} of {len(d['converged'])}")),
        ("120\\,s target (cold / warm)",
         f"{'met' if s['pass_120s_cold'] else 'missed'} / {'met' if s['pass_120s_warm'] else 'missed'} & "
         f"{'met' if xs['pass_120s_cold'] else 'missed'} / {'met' if xs['pass_120s_warm'] else 'missed'} (not gated)"),
    ]  # fmt: skip
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Quantity & Apple M5 Max & x86-64 runner, 2 vCPU \\", r"\midrule"]  # fmt: skip
    lines += [f"{a} & {v} \\\\" for a, v in top]
    lines += [r"\bottomrule", r"\end{tabular}"]
    upper = "\n".join(lines)

    ess, xess = s["ess"], xs["ess"]
    rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
            r" & \multicolumn{3}{c}{Apple M5 Max} & \multicolumn{2}{c}{x86-64 runner} \\",
            r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
            r"Quantity group & Min.\ bulk ESS & Min.\ ESS/s & Median ESS/s & Min.\ bulk ESS & Min.\ ESS/s \\",
            r"\midrule"]  # fmt: skip
    for key, label in GROUPS:
        g, gx = ess[key], xess[key]
        rows.append(f"{label} & {tc.thousands(g['ess_min'])} & {g['ess_per_s_min']:.1f} & "
                    f"{g['ess_per_s_median']:.1f} & {tc.thousands(gx['ess_min'])} & {gx['ess_per_s_min']:.1f} \\\\")  # fmt: skip
    cl, clx = ess["cross_lag"], xess["cross_lag"]
    assert cl["quantity"] == clx["quantity"], (cl, clx)
    rows.append(f"cross-lag coefficient {tc.code(cl['quantity'])} & {tc.thousands(cl['ess'])} & "
                f"{cl['ess_per_s']:.1f} & --- & {tc.thousands(clx['ess'])} & {clx['ess_per_s']:.1f} \\\\")  # fmt: skip
    rows += [r"\bottomrule", r"\end{tabular}"]
    lower = "\n".join(rows)

    setting = (f"$n = {st['n']}$, $p = {st['lags']}$, $T = {st['T']}$; {st['chains']} chains "
               f"$\\times$ {tc.thousands(st['draws'])} kept draws after {tc.thousands(st['burn'])} "
               f"burn-in, thin {st['thin']}, {st['devices']} CPU device")  # fmt: skip
    foot = (f"Recorded, not re-measured. Apple M5 Max: {tc.code(BASELINE)} as committed in "
            f"{tc.code(commit)}, measured at commit {tc.code(m['commit'])} on {tc.tex(m['date'][:10])}; "
            f"{m['physical_cores']} cores ({m['perf_cores']} performance), "
            f"{tc.tex(m['platform'])}, Python {m['python']}, NumPy {m['numpy']}, "
            f"\\pkg{{jax}}/\\pkg{{jaxlib}} {m['jax']}; one-minute load average "
            f"{min(load):.1f}--{max(load):.1f} before the six runs (not idle). "
            f"x86-64: {tc.code(X86)}, the report of the CI bench job on a GitHub runner, measured at commit "
            f"{tc.code(xm['commit'])} (before D-35) on {tc.tex(xm['date'][:10])}; {tc.tex(xm['cpu'])}, "
            f"{xm['logical_cpus']} logical CPUs, {tc.tex(xm['platform'])}, Python {xm['python']}, NumPy {xm['numpy']}, "
            f"\\pkg{{jax}}/\\pkg{{jaxlib}} {xm['jax']}; one-minute load average {min(xload):.1f}--{max(xload):.1f}. "
            "A shared 2-vCPU runner is not the protocol's x86 machine, so the 120\\,s target is neither "
            "required nor asserted there (reported, not gated). "
            f"{setting}. "
            "ESS per second is bulk ESS over the end-to-end wall time of the warm runs; the ESS of "
            "the cross-lag coefficient is of the stored draws.")  # fmt: skip
    body = (tc.fit(upper) + "\n\n\\medskip\n" + tc.fit(lower) + "\n" + tc.note(foot))
    tc.write(NAME, body, generator="tab_bench.py", status="[R] recorded baseline and x86 CI report",
             sources=[f"main repo {BASELINE} at {commit} (measured at {m['commit']})",
                      f"main repo {X86} at HEAD (measured at {xm['commit']})",
                      "benchmarks/phase2_perf.py (definitions of the summary fields)"])  # fmt: skip


if __name__ == "__main__":
    main()
