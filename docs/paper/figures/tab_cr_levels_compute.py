"""TabF7 (Appendix F): the Constant arm's levels against CR's benchmark VAR, and the compute record.

Every acceptance number is an ``\\acc{}`` placeholder (OUTLINE section 0.5 [ACC]); the script never
reads a value from the acceptance report. Upper panel: the Constant arm's average log score and
RMSE at h = 1 over the full sample, ``relative.full['SV(rw)'].variables.<v>['1'].{lps_base,
rmse_base}`` (the reference arm's levels in the SV(rw) comparison), against the benchmark VAR's
levels that ``docs/phase2/08_common_sv_and_ordering.md`` section C5 records from Clark and
Ravazzolo's working paper ([R]; read from the committed note with a regular expression). Lower
panel: the run's compute record, ``config.{n_jobs, mcmc, draws, lpd_method}``,
``versions.{platform, rai_bvar, jax, git_commit, git_dirty}`` and ``seconds``. The report holds no
stage times; the stage-time row is read ([R]) from the run's own log, committed with the report in
65dd212 (``benchmarks/reports/clark_ravazzolo_acceptance.log``; elapsed seconds at "main stage
written", "ordering stage Constant", "ordering stage SV(rw)" and "done").

``--check-acc [REPORT]`` checks that every key resolves (no value is printed).

Usage: tab_cr_levels_compute.py [--outdir DIR] [--check-acc [REPORT]]
"""

from __future__ import annotations

import argparse
import re

import tab_common as tc

NAME = "tab_cr_levels_compute"
N08 = ("c823d20", "docs/phase2/08_common_sv_and_ordering.md")
VARS = (("GDP", "GDP"), ("INFL", "inflation"), ("UR", "UR"), ("TBILL", "bill"))  # (ours, N08's word)
LOG = ("65dd212", "benchmarks/reports/clark_ravazzolo_acceptance.log")


def stage_times() -> str:
    """Stage times from the committed run log ([R]): the elapsed-seconds stamps of its stage lines."""
    log = tc.git_show(*LOG)
    def at(pat: str) -> int:
        return int(tc.grab(log, r"\[\s*(\d+)s\] " + pat, flags=0))
    def dur(pat: str) -> int:
        return int(tc.grab(log, pat + r": (\d+)s", flags=0))
    main, done = at("main stage written"), at("done:")
    oc, osv = dur("ordering stage Constant"), dur(r"ordering stage SV\(rw\)")
    return (f"main stage {tc.thousands(main)}\\,s; ordering stage, Constant {tc.thousands(oc)}\\,s "
            f"and SV(rw) {tc.thousands(osv)}\\,s; done at {tc.thousands(done)}\\,s (run log, {LOG[0]})")


def cr_levels() -> dict:
    sec = tc.section(tc.git_show(*N08), r"^## C5\.")
    line = tc.grab(sec, r"Benchmark VAR levels \(full sample, h = 1\): (.*?)\n")
    lps_txt = tc.grab(line, r"average log score (.*?); RMSE")
    rmse_txt = tc.grab(line, r"RMSE (.*?)\.$")
    out = {}
    for ours, word in VARS:
        lp = tc.grab(lps_txt, re.escape(word) + r" ([−-]?[\d.]+)")
        rm = tc.grab(rmse_txt, re.escape(word) + r" ([−-]?[\d.]+)")
        out[ours] = (float(lp.replace("−", "-")), float(rm.replace("−", "-")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap, acc_keys=True)
    args = ap.parse_args()
    tc.apply_outdir(args)
    cr = cr_levels()
    acc = tc.acc
    a = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Average log score, $h = 1$} & \multicolumn{2}{c}{RMSE, $h = 1$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Variable & Constant arm & CR benchmark VAR & Constant arm & CR benchmark VAR \\", r"\midrule"]  # fmt: skip
    for v, _ in VARS:
        p = f"relative.full['SV(rw)'].variables.{v}['1']"
        a.append(f"{v} & {acc(p + '.lps_base')} & ${cr[v][0]:.3f}$ & {acc(p + '.rmse_base')} & "
                 f"${cr[v][1]:.3f}$ \\\\")  # fmt: skip
    a += [r"\bottomrule", r"\end{tabular}"]

    rows = [
        ("Commit, and whether tracked files differed from it",
         f"{acc('versions.git_commit')}; modified: {acc('versions.git_dirty')}"),
        ("Platform", acc("versions.platform")),
        (r"\pkg{rai-bvar}, \pkg{jax}", f"{acc('versions.rai_bvar')}, {acc('versions.jax')}"),
        ("Configuration and LPD estimator", f"{acc('config.name')}, {acc('config.lpd_method')}"),
        ("MCMC budget per fit", acc("config.mcmc")),
        ("Forecast draws per origin", acc("config.draws")),
        ("Worker processes", acc("config.n_jobs")),
        ("Stage times", stage_times()),
        ("Wall time", acc("seconds")),
    ]
    b = [r"\begin{tabular}{@{}ll@{}}", r"\toprule", r"Compute record of the acceptance run & \\", r"\midrule"]
    b += [f"{k} & {v} \\\\" for k, v in rows]
    b += [r"\bottomrule", r"\end{tabular}"]
    foot = ("Constant arm: the reference arm's levels in the SV(rw) comparison, full sample "
            "1985Q1--2011Q2, path-integrated RB log score (at $h = 1$ identical to the "
            "path-conditioned one). CR benchmark VAR: Clark and Ravazzolo's working paper as "
            r"recorded in \code{docs/phase2/08_common_sv_and_ordering.md} \S C5 (Gaussian-approximation "
            "log score, real-time data). The two log-score columns differ in estimator and data "
            "vintage: a sanity check of levels, not a parity test.")  # fmt: skip
    body = (tc.preamble(acc_cells=True, marks=True) + tc.fit("\n".join(a)) + "\n\n\\medskip\n"
            + tc.fit("\n".join(b)) + "\n" + tc.note(foot) + "\n"
            + tc.note(tc.acc_placeholder_note("relative.full, config, versions, seconds"), red=True))  # fmt: skip
    path = tc.write(NAME, body, generator="tab_cr_levels_compute.py",
                    status="[ACC] placeholders (no acceptance number written); CR levels [R]; stage times [R] (run log)",
                    sources=["acceptance report keys relative.full['SV(rw)'].variables[v]['1'].{lps_base,"
                             "rmse_base}, config, versions, seconds",
                             f"main repo {N08[1]} at {N08[0]}, section C5 (benchmark VAR levels)",
                             f"main repo {LOG[1]} at {LOG[0]} (stage times)"])  # fmt: skip
    tc.check_acc(path, args.check_acc)


if __name__ == "__main__":
    main()
