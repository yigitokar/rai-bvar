"""TabF1 (Appendix F): run history of the replication, from the quick run to the acceptance run.

Upper panel, one row per run: code commit, configuration, machine and load, wall time, outcome.
Lower panel, the commits between the runs: time (UTC), subject, and whether each is an ancestor
of the code of full run #1 (8a622c5), full run #2 (4bb978a) and the acceptance run (c823d20)
(``git merge-base --is-ancestor``; 4fc0162 is *not* an ancestor of 4bb978a and is listed after that
run's start).

Sources and evidence tags:

* quick run [R]: ``benchmarks/reports/clark_ravazzolo_quick.{json,md}`` as committed in 1ae5765
  (``config``, ``versions``, ``criteria``; the main stage's machine, load and run time from the
  report's banner);
* full run #2 [FULL]: ``benchmarks/reports/clark_ravazzolo.json`` at 551f181 (``config``,
  ``versions``, ``seconds``, ``criteria``), printed with ``\\fullrun``;
* full runs #1 and #2, start times, machine, load and stage times [R]: the comments of GitHub issue
  #3 of the main repository (``gh issue view 3``; each cell names the comment's UTC time). Without
  ``gh`` or network those cells become [TC] marks and the script says so;
* commit times and subjects: ``git log`` of the main repository;
* acceptance run [ACC]: ``\\acc{}`` placeholders for ``versions.git_commit``, ``config``,
  ``versions.platform``, ``seconds`` and ``criteria[*].met``; the script never reads a value from
  the acceptance report (``--check-acc`` only checks that the keys resolve).

Usage: tab_runs.py [--outdir DIR] [--check-acc [REPORT]] [--no-issue]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone

import tab_common as tc

NAME = "tab_runs"
QUICK = ("1ae5765", "benchmarks/reports/clark_ravazzolo_quick")
FULL2 = ("551f181", "benchmarks/reports/clark_ravazzolo.json")
RUN_CODE = {"full #1": "8a622c5", "full #2": "4bb978a", "acceptance": "c823d20"}
EVENTS = ["1ae5765", "3d3f51b", "93f2bcd", "f60d736", "4bb978a", "4fc0162", "551f181", "b9c66c0",
          "c823d20", "d58c9ef", "7f410dc", "65dd212"]  # fmt: skip
REPO_SLUG = "yigitokar/rai-bvar"


def utc(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def criteria_text(crit: list[dict]) -> str:
    met = [str(c["id"]) for c in crit if c.get("met") is True]
    miss = [str(c["id"]) for c in crit if c.get("met") is False]
    none = [str(c["id"]) for c in crit if c.get("met") is None]
    out = f"criteria {', '.join(met)} met"
    if miss:
        out += f"; {', '.join(miss)} not met"
    if none:
        out += f"; {', '.join(none)} not evaluated"
    return out


def mcmc_text(cfg: dict) -> str:
    m = cfg["mcmc"]
    return (f"\\code{{{cfg['name']}}}: {m['chains']} $\\times$ {tc.thousands(m['draws'])}, burn-in "
            f"{tc.thousands(m['burn'])}; {tc.thousands(cfg['draws'])} forecast draws")  # fmt: skip


def issue_facts() -> dict:
    """Facts from the comments of issue #3, each with the UTC time of its comment."""
    out = subprocess.run(["gh", "issue", "view", "3", "--repo", REPO_SLUG, "--json", "comments"],
                         capture_output=True, text=True, timeout=60, cwd=tc.MAIN_REPO)  # fmt: skip
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    comments = json.loads(out.stdout)["comments"]

    def find(needle: str) -> dict:
        hits = [c for c in comments if needle in c["body"]]
        if len(hits) != 1:
            raise ValueError(f"issue #3: {len(hits)} comments contain {needle!r}")
        return hits[0]

    f = {}
    c = find("**item 7 started**")
    f["full1_start"] = tc.grab(c["body"], r"\*\*item 7 started\*\* at (\S+Z) on `8a622c5`")
    f["full1_jobs"] = tc.grab(c["body"], r"--n-jobs (\d+)")
    f["full1_alongside"] = "weekly tier (item 2) is running alongside it" in c["body"]
    f["full1_start_at"] = utc(c["createdAt"])
    c = find("The first full run on `8a622c5` aborted")
    f["crash_at"] = utc(c["createdAt"])
    f["crash"] = tc.grab(c["body"], r"The error was `(MCMCNumericalError)` in `(EQ_VOL)`", 0)
    f["crash_code"] = tc.grab(c["body"], r"code (\d{4}), pivot 0")
    f["crash_stage"] = "aborted** in the ordering stage" in c["body"]
    f["full2_start"] = tc.grab(c["body"], r"\*\*Rerun:\*\* started at (~?\d\d:\d\dZ) on `4bb978a`")
    c = find("## Item 7: full Clark–Ravazzolo replication, results")
    f["full2_at"] = utc(c["createdAt"])
    f["full2_machine"] = tc.grab(c["body"], r"FRED-QD 2026-09, (Apple M5 Max)\.")
    f["full2_load"] = tc.grab(c["body"], r"at load about (\d+)")
    f["full2_stages"] = tc.grab(c["body"], r"Stage times: (main stage \d+ s; Constant orderings \d+ s; "
                                           r"SV\(rw\) orderings \d+ s)")  # fmt: skip
    f["full2_runtime"] = tc.grab(c["body"], r"Runtime \*\*([\d.]+) h\*\*")
    c = find("rai-bvari local: queue received")
    f["machine_note_at"] = utc(c["createdAt"])
    f["machine"] = tc.grab(find("## Item 0 — reference BENCH-P2: done")["body"], r"Machine: (Apple M5 Max)")
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap, acc_keys=True)
    ap.add_argument("--no-issue", action="store_true", help="do not query issue #3 (cells become [TC])")
    args = ap.parse_args()
    tc.apply_outdir(args)

    q = tc.git_show_json(QUICK[0], QUICK[1] + ".json")
    qmd = tc.git_show(QUICK[0], QUICK[1] + ".md")
    q_machine = tc.grab(qmd, r"Its main stage was computed on an? (.*?) in ([\d.]+) h;", 1)
    q_hours = tc.grab(qmd, r"Its main stage was computed on an? .*? in ([\d.]+) h;")
    f2 = tc.git_show_json(*FULL2)
    assert f2["config"]["name"] == "full" and q["config"]["name"] == "quick"
    assert "git_commit" not in q["versions"] and "git_commit" not in f2["versions"]  # not recorded then

    try:
        if args.no_issue:
            raise RuntimeError("--no-issue")
        iss = issue_facts()
    except Exception as err:  # noqa: BLE001
        print(f"warning: issue #3 not read ({err}); its cells are [TC] marks")
        iss = None

    def I(key: str, fmt=lambda v: v) -> str:  # noqa: E743
        return fmt(iss[key]) if iss else tc.mark("TC") + " issue \\#3"

    F = tc.fullrun
    # the full config at the code of run #1 (CONFIGS["full"] of the script at 8a622c5)
    m = tc.grab(tc.git_show(RUN_CODE["full #1"], "benchmarks/clark_ravazzolo.py"),
                r'"full": Config\("full", MCMCSettings\(chains=(\d+), draws=(\d+), burn=(\d+)\), (\d+), \d+\)', 0)
    ch, dr, bu, fd = re.findall(r"\d+", m.split("Config(", 1)[1])[:4]
    full1_cfg = (f"\\code{{full}}: {ch} $\\times$ {tc.thousands(int(dr))}, burn-in {tc.thousands(int(bu))}; "
                 f"{tc.thousands(int(fd))} forecast draws")
    f2h = f2["seconds"] / 3600
    if iss:
        assert f"{f2h:.2f}" == iss["full2_runtime"], (f2h, iss["full2_runtime"])
    runs = [
        ("Quick", "not recorded", mcmc_text(q["config"]),
         f"{tc.tex(q_machine)}", f"main stage {q_hours}\\,h; ordering stage stopped",
         criteria_text(q["criteria"]) + f"; report {tc.code(QUICK[0])}"),
        ("Full \\#1", tc.code(RUN_CODE["full #1"]),
         f"{full1_cfg}; {I('full1_jobs')} workers",
         I("full1_start", lambda v: f"{iss['machine']}{', weekly tier alongside' if iss['full1_alongside'] else ''}; "
           f"started {v.replace('2026-10-23T', '')}"),
         I("crash_at", lambda v: f"aborted; reported {v[11:]}"),
         I("crash_code", lambda v: f"crash F2 in the ordering stage (\\code{{MCMCNumericalError}}, "
           f"\\code{{EQ_VOL}}, code {v}); no report")),
        ("Full \\#2", tc.code(RUN_CODE["full #2"]),
         F(mcmc_text(f2["config"])) + f"; {F(str(f2['config']['n_jobs']))} workers",
         I("full2_load", lambda v: f"{iss['full2_machine']}, load about {v} (weekly tier alongside); "
           f"started {iss['full2_start'].replace('~', 'about ')}"),
         F(f"{tc.thousands(f2['seconds'])}\\,s ({f2h:.2f}\\,h)")
         + I("full2_stages", lambda v: f"; {tc.tex(v)}"),
         F(criteria_text(f2["criteria"])) + f"; report {tc.code(FULL2[0])}"),
        ("Acceptance", tc.acc("versions.git_commit"),
         f"{tc.acc('config.name')}: {tc.acc('config.mcmc')}; {tc.acc('config.draws')} forecast draws; "
         f"{tc.acc('config.lpd_method')}; {tc.acc('config.n_jobs')} workers",
         tc.acc("versions.platform"), tc.acc("seconds"),
         f"{tc.acc('criteria[*].met')}; report {tc.code('65dd212')}"),
    ]  # fmt: skip
    # \leavevmode: a cell that starts with the \color of \acc or \fullrun would open with an empty line
    P = r">{\raggedright\arraybackslash\leavevmode}p"
    a = [r"\begin{tabular}{@{}" + "".join(f"{P}{{{w}\\linewidth}}" for w in (0.11, 0.08, 0.18, 0.17, 0.15, 0.22))
         + "@{}}",
         r"\toprule", r"Run & Code & Configuration & Machine, load, start & Wall time & Outcome \\",
         r"\midrule"]  # fmt: skip
    for row in runs:
        a.append(" & ".join(row) + r" \\")
    a += [r"\bottomrule", r"\end{tabular}"]

    cols = list(RUN_CODE.values())
    b = [r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{0.52\linewidth}ccc@{}}", r"\toprule",
         r" & & & \multicolumn{3}{c}{In the code of} \\ \cmidrule(l){4-6}",
         r"Commit & Committed (UTC) & Subject & full \#1 & full \#2 & acceptance \\", r"\midrule"]  # fmt: skip
    for c in EVENTS:
        when = utc(tc.git("log", "-1", "--format=%cI", c))
        subj = tc.git("log", "-1", "--format=%s", c)
        flags = ["yes" if tc.is_ancestor(c, t) else "no" for t in cols]
        b.append(f"{tc.code(c)} & {when[5:]} & {tc.tex(subj)} & " + " & ".join(flags) + r" \\")
    assert not tc.is_ancestor("4fc0162", RUN_CODE["full #2"]), "4fc0162 must not be in full run #2"
    b += [r"\bottomrule", r"\end{tabular}"]

    foot = ("Quick run: report banner and header (1ae5765; its code commit was not recorded); its "
            "assembling invocation read cached origins. Full runs: config and outcome from the "
            "reports, start times, machine, load and stage times from the comments of issue \\#3 "
            + (f"(comments of {iss['full1_start_at'][11:]}, {iss['crash_at'][11:]} and "
               f"{iss['full2_at'][11:]} on 2026-10-23)" if iss else "(not read)")
            + ". Lower panel: commit times are committer times converted to UTC; ``In the code of'' "
            r"is \code{git merge-base --is-ancestor} against 8a622c5, 4bb978a and c823d20. 4fc0162 "
            "came after full run \\#2 had started and is not part of it.")  # fmt: skip
    body = (tc.preamble(acc_cells=True, marks=iss is None) + tc.fit("\n".join(a)) + "\n\n\\medskip\n"
            + tc.fit("\n".join(b)) + "\n" + tc.note(foot) + "\n"
            + tc.note(tc.acc_placeholder_note("config, versions, seconds, criteria"), red=True))  # fmt: skip
    path = tc.write(NAME, body, generator="tab_runs.py",
                    status="quick [R]; full #2 [FULL]; issue #3 facts [R]; acceptance row [ACC] placeholders",
                    sources=[f"main repo {QUICK[1]}.{{json,md}} at {QUICK[0]}",
                             f"main repo {FULL2[1]} at {FULL2[0]}",
                             f"GitHub issue #3 of {REPO_SLUG} (gh issue view 3)" + ("" if iss else ": NOT READ"),
                             "git log / git merge-base --is-ancestor of the main repo",
                             "acceptance report keys versions, config, seconds, criteria[*].met"])  # fmt: skip
    tc.check_acc(path, args.check_acc)


if __name__ == "__main__":
    main()
