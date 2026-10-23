"""TabE2 (Appendix E): BENCH-P2 on the production engine.

Source: ``tests/bench/baseline.json`` of the main repository as committed in d490730 (measured at
commit deb3929 on a non-idle Apple M5 Max; OUTLINE TODO 9 asks for a re-measurement on the release
commit). Evidence tag [R]: recorded by ``benchmarks/phase2_perf.py --reference``, never re-measured
here. Upper panel: medians over three cold and three warm fresh-process fits (``summary``); lower
panel: bulk ESS and ESS per second by quantity group (medians over the warm runs).

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

    load = s["loadavg_1min"]
    top = [
        ("Wall time, cold cache (median of 3)", f"{s['wall_cold_s']:.1f}\\,s"),
        ("Wall time, warm cache (median of 3)", f"{s['wall_warm_s']:.1f}\\,s"),
        ("Compilation, cold / from the persistent cache",
         f"{s['compile_cold_s']:.2f}\\,s / {s['compile_warm_s']:.2f}\\,s"),
        ("Run time, wall minus compilation (median of 6)", f"{s['run_s']:.1f}\\,s"),
        ("After the runner: fit assembly, diagnostics (median of 6)", f"{s['post_runner_s']:.2f}\\,s"),
        ("Per sweep: minimum of $3\\times1000$ sweeps of the compiled burn-in (median of 6)",
         f"{s['sweep_ms_min3x1000']:.2f}\\,ms"),
        (f"Per sweep: runner, over the {tc.thousands(st['draws'])} kept sweeps (median of 6)",
         f"{s['runner_ms_per_sweep']:.2f}\\,ms"),
        ("Peak resident set (median of 6)", f"{tc.thousands(s['peak_rss_mb'])}\\,MiB"),
        (r"\code{converged} (R-hat $<1.01$, bulk ESS $>400$ on every primary quantity)",
         f"false in {s['converged'].count(False)} of {len(s['converged'])} runs"),
        ("120\\,s target (cold / warm)",
         f"{'met' if s['pass_120s_cold'] else 'missed'} / {'met' if s['pass_120s_warm'] else 'missed'}"),
    ]  # fmt: skip
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
    lines += [f"{a} & {v} \\\\" for a, v in top]
    lines += [r"\bottomrule", r"\end{tabular}"]
    upper = "\n".join(lines)

    ess = s["ess"]
    rows = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
            r"Quantity group & Min.\ bulk ESS & Min.\ ESS/s & Median ESS/s \\", r"\midrule"]  # fmt: skip
    for key, label in GROUPS:
        g = ess[key]
        rows.append(f"{label} & {tc.thousands(g['ess_min'])} & {g['ess_per_s_min']:.1f} & "
                    f"{g['ess_per_s_median']:.1f} \\\\")  # fmt: skip
    cl = ess["cross_lag"]
    rows.append(f"cross-lag coefficient {tc.code(cl['quantity'])} & {tc.thousands(cl['ess'])} & "
                f"{cl['ess_per_s']:.1f} & --- \\\\")  # fmt: skip
    rows += [r"\bottomrule", r"\end{tabular}"]
    lower = "\n".join(rows)

    setting = (f"$n = {st['n']}$, $p = {st['lags']}$, $T = {st['T']}$; {st['chains']} chains "
               f"$\\times$ {tc.thousands(st['draws'])} kept draws after {tc.thousands(st['burn'])} "
               f"burn-in, thin {st['thin']}, {st['devices']} CPU device")  # fmt: skip
    foot = (f"Recorded baseline (not re-measured): {tc.code(BASELINE)} as committed in "
            f"{tc.code(commit)}, measured at commit {tc.code(m['commit'])} on {tc.tex(m['date'][:10])}; "
            f"{tc.tex(m['cpu'])} ({m['physical_cores']} cores, {m['perf_cores']} performance), "
            f"{tc.tex(m['platform'])}, Python {m['python']}, NumPy {m['numpy']}, "
            f"\\pkg{{jax}}/\\pkg{{jaxlib}} {m['jax']}; one-minute load average "
            f"{min(load):.1f}--{max(load):.1f} before the six runs (not idle). {setting}. "
            "ESS per second is bulk ESS over the end-to-end wall time of the warm runs; the ESS of "
            "the cross-lag coefficient is of the stored draws.")  # fmt: skip
    body = (upper + "\n\n\\medskip\n" + lower + "\n" + tc.note(foot))
    tc.write(NAME, body, generator="tab_bench.py", status="[R] recorded baseline",
             sources=[f"main repo {BASELINE} at {commit} (measured at {m['commit']})",
                      "benchmarks/phase2_perf.py (definitions of the summary fields)"])  # fmt: skip


if __name__ == "__main__":
    main()
