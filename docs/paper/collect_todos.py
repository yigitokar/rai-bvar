#!/usr/bin/env python3
r"""Write AUTHOR_TODO.md: every \todo{...} of the sources (file:line, text), the author's TODO list of
OUTLINE.md section 16 (verbatim), and the assembler's notes.

    python3 collect_todos.py            # rewrites docs/paper/AUTHOR_TODO.md

The \todo list is regenerated from the sources each time; the assembler's notes and the status
lines of OUTLINE items are kept in ASSEMBLY_NOTES and STATUS below.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from texmarks import find_marks  # noqa: E402

OUT = HERE / "AUTHOR_TODO.md"

# Status of OUTLINE section 16 items, as checked at assembly (2026-10-23). Only what was verified.
STATUS = {
    1: "Partly done in the main repository: d58c9ef changes the '1000 forecast draws' text of "
       "`docs/paper/NOTES.md` and `docs/phase2/DESIGN.md` to 8000. NOTES 8b there still calls "
       "-19.1 \"exact given those draws\" (the paper never does). The paper-worktree copy "
       "`docs/paper/NOTES.md` is still the 62d15f6 file without the D-34 addendum.",
    2: "Done. The main repository commits `benchmarks/reports/clark_ravazzolo_acceptance."
       "{json,md,log}` in 65dd212 (2026-10-23 15:47 EDT; on `main` through the merge 08e1eda). "
       "The JSON passes the acceptance check of `fill_acceptance.py` (config `acceptance`, "
       "integrated LPD, 8000 forecast draws, 4 x 20,000 kept draws with burn-in 1000, seed 0, "
       "commit c823d20, clean tree, no `pending` key). The committed log opens with the start "
       "time 2026-10-23T16:10:56Z, 11 s after the commit time of c823d20 (16:10:45Z), and names "
       "c823d20 in its config line. The paper is filled (`acceptance_values.tex`, every key "
       "resolved, report sha256 d5c3ec8d...), the outcome prose of the abstract and Sections 1, "
       "11, 12 and 13 is written, and the TODO 2 mark of App. F is gone.",
    5: "Done, not committed: `docs/paper/figures/data/cr_acceptance/cr_acceptance_per_origin.csv` "
       "(10,600 rows) with its `.meta.json` and README, exported read-only from the run's score "
       "cache with the code at c823d20; its check recomputes 2236 report values with maximum "
       "difference 0. FigF1 and FigF3 are drawn from it (the export moved from `docs/paper/data/` "
       "to `figures/data/cr_acceptance/`). Commit it, with `figures/make_data_cr_per_origin.py`, "
       "together with the paper.",
    3: "Done for row 18 at integration (2026-10-23): 7f410dc registers row 18 `reduced_sv_rw` "
       "(G-small per PR, G-full and SBC weekly with a per-row t_max of 400, the N(-1, 4) h0 variant "
       "test), adds a `reduced-SV(rw)` contract fit and fits the composition through `BVAR.fit` "
       "(tests/unit/test_fit_mcmc_api.py). The paper states coverage at 7f410dc (23 composition rows, "
       "13 MCMC contract compositions; Sections 1, 3, 6, 8.10, 11, 12; App. C and D, app:row18). "
       "Open: the SV(rw, t) variant has no row or contract fit (the remaining TODO 3 marks), and the "
       "row-18 numbers are one run each on a loaded machine (re-record on the release commit, TODO 7).",
    20: "The paper's `Makefile` has `make tables` (the 14 appendix table generators `figures/tab_*.py` "
        "-> `figures/tables/`) and `make arxiv` (`make_arxiv.sh`). The 11 generators named only in "
        "comments of typed tables (`tab_components`, `tab_resolution`, ...) are not written (the "
        "TODO 20 mark in App. G). OUTLINE section 14 names `figures/out/` as the output directory; "
        "the generators write `figures/tables/`.",
    29: "Resolved in the text after the referee review (2026-10-23, macro referee minor 7): "
        "Section 9 (F2) and App. F (app:cr-history) now say that the committed quick report predates "
        "f60d736, so its random-walk SV numbers may have been affected and serve only as context, "
        "including as one run of reading rule B's yardstick; the TODO 29 mark is gone.",
    4: "Open. The script and the 520 + 424 fit records are tracked on branch `paper-draft` (since "
       "33dd9f8) under `docs/paper/figures/data/convergence_diagnostic/`, not on `main` (888f620); "
       "commit them to the main repository.",
    31: "Not needed under tectonic (XeTeX, TU encoding prints < and >); `main.tex` loads T1 only "
        "under pdfTeX, which is untested here. arXiv compiles with pdfLaTeX, so test that branch "
        "before submitting (`make arxiv` packages the sources; its --test build uses tectonic).",
}

ASSEMBLY_NOTES = r"""
These came up while assembling and integrating the draft and are not (or not only) `\todo` marks in
the text.

1. **Length after the referee revision (2026-10-23).** The main text (title page to the end of
   Section 13) ends on page 34 of the draft build, against page 33 before the revision; the
   appendices run from page 35 to page 93 and the PDF has 101 pages (REVIEW_RESPONSE.md). The page
   ends with almost no slack, so any addition to Sections 11-13 moves the end to page 35. Earlier
   record:
   **Length (after integration, 2026-10-23).** The main text (title page to the end of Section 13)
   is 31.8 pages in the draft build and 31.1 with the `\todo` boxes removed and `+0.000` in place
   of the `\acc` placeholders; Section 13 ends on page 32 either way. Per section (draft build,
   heading to heading): introduction 2.5 (budget 2.5), related 1.0 (1), framework 2.4 (2.5),
   architecture 3.3 (2.5; Table 1 of Section 3 and Fig. 1 fall in its span), samplers 2.5 (2.5),
   evaluation 3.1 (3), walk-through 2.0 (2), validation 4.9 (4.5), robustness 1.7 (1.5),
   performance 0.7 (0.75), replication 4.0 (4), limitations 2.0 (1.75), conclusion 1.0 (1). To
   get under 32 pages the integration moved four main-text floats to the appendices: Tab2 (engine
   resolution) to App. C (app:resolution, with a prose summary in Section 4.2), Fig3 (the verdict
   flow) to App. C (app:scores-tests; the boxed rule stays in Section 6.8), the compact Algorithm 2
   of Section 5.4 into Algorithm C.7 (alg:C-sv, which carries every step and the label alg:sv), and
   Tab12 (the six criteria) to App. F (app:cr-criteria; Section 11 states the outcome with
   `\acc{criteria[*].met}`). Tab2, Fig3 and Tab12 are main-text floats in OUTLINE section 14.1;
   revert a move if the venue allows more pages.
2. **Appendix length.** The appendices run to 56.5 pages (A 2.4, B 6.7, C 8.8, D 10.8, E 5.0,
   F 14.2, G 6.6, H 1.0, I 0.5, J 0.4) against the outline's 20-25, plus 8.5 pages of
   bibliography (97 pages in all). Candidates for a separate supplement: the walk-through scripts
   and outputs (App. G, app:walk-output, about 4 pages), TabF8 and the pre-registration quote
   (App. F), and the runner algorithms C.9-C.11.
3. **Table generators.** The 14 generators `figures/tab_*.py` write `figures/tables/*.tex`, which
   the appendix `\input`s (`make tables`). Cells that wait for data are `\acc{}` (acceptance run;
   `fill_acceptance.py` now scans `figures/tables/` too) or a red `[REL]` mark (release validation
   records, TODO 7), which stops a `\draftmarksfalse` build. TabD1, TabD3 and TabG2 carry red
   "interim" notes.
4. **FigE1 (TODO 23), decided.** The figure stays. Section 9 (F2) states that before the fix the
   old random-walk path could return a finite but wrong answer at a tiny positive pivot, with no
   failure reported, as a recomputation at the synthetic test state, with the figure script as
   evidence; App. E (app:float64-oldpath) gives the grid numbers. Still open: a committed test of
   the old-path values (TODO 23) and whether pre-fix runs were affected (TODO 29).
5. **Validation records of unknown provenance.** `docs/paper/figures/data/validation_record/`
   (tracked on branch `paper-draft` since 33dd9f8, not on `main`) feeds FigD1 and FigD2. It carries
   no commit or machine field and holds no G-small rows. Its G-full row of `reduced_sv` (passed
   false, 28,068 non-finite values, 92 of 137 functions failing, seed 20261023) matches the
   2026-10-22 failure documented in `tests/geweke/test_geweke_compositions.py` and fixed in 3d3f51b,
   an ancestor of 4bb978a and c823d20, so it is almost certainly a stale pre-fix record; the record
   cannot date itself. Section 8, Section 12 and the FigD2 caption now say so. A reviewer's re-run
   at 75db8a7 passed but is uncommitted and not cited. Record G-full `reduced_sv` on the release
   commit with commit and machine fields (TODO 7), then redraw both figures.
6. **Convergence-diagnostic records.** `docs/paper/figures/data/convergence_diagnostic/` (tracked on
   `paper-draft` since 33dd9f8, not on `main`: a script and 520 + 424 fit records) reproduces the
   pre-registration's budget numbers; commit it to the main repository (TODO 4) or keep TabF6 at
   [PREREG].
7. **Abstract length.** 1893 characters as plain text with the acceptance values filled, after the
   referee revision (about 1905 if pasted into the arXiv form with `--`, `$t$`, `$h > 1$` and
   `\%`); the arXiv limit is 1920, so any addition needs an equal cut. The Lenza--Primiceri COVID-19
   scaling and "a larger MCMC budget" were cut from the abstract to make room.
8. **Sentences built on filled values.** Done for the acceptance report (2026-10-23): the outcome
   prose of the abstract, Sections 1, 11, 12 and 13 and App. F is written from the filled values,
   with the reading rules of App. F stated as the paper's, not the pre-registration's (first
   committed in 33dd9f8, after the run had written its main-stage report). The first-run
   comparison of reading rule B (h = 1 moves 0.0031 marginal, 0.0116 joint at most) is recomputed
   from the two reports, not an `\acc` key. After any refill, re-read every sentence with an
   `\acc`; `fill_acceptance.py` prints values, not prose. With the outcome prose, Section 13
   starts on page 32 and ends on page 33 of the draft build.
9. **arXiv package.** `make arxiv` writes `arxiv.tar.gz` (main.tex, main.bbl, refs.bib, the
   sections, the figure PDFs, `figures/tables/`, and the four listing inputs in `code/`); its
   `--test` build compiles it with tectonic in a clean directory. arXiv uses pdfLaTeX, whose branch
   of `main.tex` is untested (STATUS of TODO 31). Set `\draftmarksfalse` first.
"""


REFEREE_ITEMS = r"""
From the triage of the three referee reports on the draft at 75db8a7 (statistics, macroeconometrics,
software; 2026-10-23). Only the author can do these: they change code in the main repository, need
new runs or new bibliography entries, or are decisions. What was applied to the paper itself is in
`REVIEW_RESPONSE.md`. "New run" means a departure from the pre-registration's "no further run with
different settings" (l.100-102) unless disclosed as such; "re-analysis" means existing outputs (the
run's cache `~/.cache/rai-bvar-cr-c823d20`, or the per-origin CSV once committed) and is compatible
with it.

**Verdict engine and tests (stat major 2, macro major 4 and minor 9, software majors 3-4)**
1. `significance.py`/`compare.py` (main repository): a pairwise MCS mode (or report the comparison
   set with every verdict), a lower default `mcs_alpha` (at most 0.10, or justify 0.25 against the
   10% DM route), a horizon-aware block length (at least h), and the MCS p-values in the reasons
   string. A Clark-McCracken-type nested bootstrap needs a bib key beyond `clark2001tests`.
2. Re-apply any changed rule to the existing per-origin scores (pairwise main-stage verdicts, a
   "which verdicts change" table): needs the CSV committed (TODO 5); no new run.
3. Decide the word "Fair" in the title (TODO 10), and "Validated Samplers" (stat major 5, macro
   major 1).
4. Optional: print both one-sided DM p-values in `compare()`'s reasons line (software minor 2).

**Convergence gate (stat major 3 and minor 14, macro major 5)**
5. Quote no R-hat false-alarm rate before a committed script computes it (the referee's simulation
   is not in the repository). Gate redesign and the default SV budget (raise it or warn): TODO 33.
6. Read the secondary-set R-hat and ESS (intercepts, a_ij) of the SV(rw) fits at the crisis origins
   from the run's cache, and the converged shares if intercepts were gated (re-analysis). A forecast
   functional in the primary set is a code change. Citing Vats and Knudson (2021) needs a bib entry.

**Log-score Monte Carlo error (stat major 4 and minors 9, 16; macro minor 4; software minor 10)**
7. Per-cell Kish ESS of the weights at the top origins, from the cache (re-analysis); inner
   volatility-path draws would be new forecasts at existing fits (disclose as a departure).
8. Redefine the RB-degenerate flag in `benchmarks/clark_ravazzolo.py` (it fires only near total
   collapse); cite Prop. B.5 in Sections 6.4 and 11.3 and at 06:78 only after its unit test (TODO 14).
   PSIS (k-hat) needs a bib key.
9. Paired MCSE of a total dLPD at a few origins (cache or replicate forecast seeds) and its ratio to
   the quadrature MCSE.

**Validation evidence (stat major 5 and minor 6, macro majors 1 and 7, software majors 8-9)**
10. Release records with commit, platform, version and seed fields on the release commit (TODO 7),
    including G-full `reduced_sv`; then redraw FigD1/FigD2 and fill TabD1/TabD3.
11. Committed positive-control tests for row 18 (the five one-off catches of 7f410dc); register
    SV(rw, t) with a contract fit or keep the caveat (TODO 3).
12. Optional: a mutant that perturbs only one side of the logits/table invariant of exact-mode SV;
    an automated operator-mutation kill rate (mutmut or cosmic-ray).

**Ordering diagnostic (stat major 7, macro major 6)**
13. `noise_reps` of 5-10 and `mode='prior'` mean new fits (TODO 26; disclose as a departure).

**Replication design (stat majors 1, 6 and minors 10-12, macro majors 2-3 and minor 13, software
majors 6-7 and minor 11)**
14. Check Clark and Ravazzolo's sample convention (by origin or by target) in the working paper's
    table notes and, if obtained, the JAE article; if it differs, add it to `DEVIATIONS` and report
    target-aligned h = 8 cells as labelled secondary analysis (needs the CSV, TODO 5). Obtain the
    JAE tables or keep "working paper" throughout (TODO 22).
15. Add the four differences (c)-(f) of app:cr-deviations (prior on A, sigma^2 prior shape, the
    1960Q2 start, the h > 1 windows) to the script's `DEVIATIONS` (main repository).
16. Decide on any exploratory run (CR's data from the JAE archive, CR's sigma^2 prior, a
    normal-diffuse baseline, a Linux re-run); each is a disclosed departure. Within the
    registration: the prior-predictive h = 8 tail-mass comparison of the two sigma^2 priors (no data).
17. A replicate-seed Monte Carlo yardstick for reading rule B (new runs).
18. The bootstrap CI of the h8 - h1 contrast and the crisis-window drop: only from a committed CSV
    (TODO 5) or committed script. Moving FigF1 into the main text is a page-budget decision.
19. Commit the convergence diagnostic and the validation records to `main` (TODO 4, TODO 7) and the
    per-origin CSV with its meta file (TODO 5); archive the cache on Zenodo if feasible.

**Release, process and text (stat majors 5-6 and minor 15, macro major 8 and minors 5, 10, software
majors 1-2, 8 and minors 6-9, 13)**
20. TODO 19 and TODO 20: tag, Zenodo DOIs (release and replication outputs), install instructions,
    README banner, re-anchor every code reference; generators for Tab7 and Tab13 and the single
    rebuild command; re-capture the walk-through outputs on the clean release commit and state the
    match.
21. TODO 30: write App. J role by role (design, kernels, references, reviews, dossiers, drafting, and
    what the author verified), and qualify "a work package whose owner wrote no kernel code"
    (08:30-33) and "independent review" (App. J, App. D) to match.
22. TODO 11 and TODO 15: prior-art searches and literature checks (including C3's estimator and the
    mutation release gate); drop any citation that cannot be checked; TODO 8; R licence checks.
23. arXiv: rebuild with `make arxiv` as the last step, check that the tarball holds only the listed
    files, add a pdfLaTeX test (TeX Live or Docker), confirm the engine arXiv uses, and build with
    `\draftmarksfalse` (every `\todo`, `\acc` and `[REL]` mark resolved).
24. Benchmark script: write a finish time into the report, and record the structural pair's
    convergence (software minors 8-9). Re-measure BENCH-P2 (TODO 9); decide whether the timing detail
    of Section 10 moves to App. E.
"""


def outline_section_16() -> str:
    text = (HERE / "OUTLINE.md").read_text(encoding="utf-8")
    m = re.search(r"^## 16\..*?(?=^## 17\.)", text, flags=re.S | re.M)
    if not m:
        return "(OUTLINE.md section 16 not found)\n"
    body = m.group(0).strip()
    body = re.sub(r"^## 16\.[^\n]*\n", "", body)
    out = []
    for line in body.split("\n"):
        out.append(line)
        mm = re.match(r"^(\d+)\.\s", line)
        if mm and int(mm.group(1)) in STATUS:
            out.append(f"   - *Status at assembly:* {STATUS[int(mm.group(1))]}")
    return "\n".join(out).rstrip("-\n ") + "\n"


def main() -> None:
    sources = [*sorted((HERE / "sections").glob("*.tex")), HERE / "main.tex"]
    marks = [m for m in find_marks("todo", sources) if not (m.rel == "main.tex" and "abstract not written yet" in m.flat)]
    by_file = defaultdict(list)
    for m in marks:
        by_file[m.rel].append(m)
    refs = Counter(int(n) for m in marks for n in re.findall(r"TODO\s*~?(\d+)", m.flat))

    lines = [
        "# Author TODO list for the rai-bvar paper",
        "",
        f"Generated by `docs/paper/collect_todos.py` on {date.today().isoformat()}; rerun it after "
        "editing the sources. Part 1 lists every `\\todo{}` in `sections/*.tex` and `main.tex` (each "
        "prints as an orange box in the draft and is a LaTeX error with `\\draftmarksfalse`). Part 2 is "
        "the author's TODO list of `OUTLINE.md` section 16, with status lines added at assembly. "
        "Part 3 lists issues found during assembly. Part 4 lists the referee items that only the "
        "author can resolve (REVIEW_RESPONSE.md has the full response).",
        "",
        f"## 1. `\\todo` marks in the sources ({len(marks)})",
        "",
        "Outline TODO numbers cited by the marks: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(refs.items())) + ".",
        "",
    ]
    for rel in sorted(by_file, key=lambda r: (r == "main.tex", r)):
        lines.append(f"### `{rel}` ({len(by_file[rel])})")
        lines.append("")
        for m in by_file[rel]:
            lines.append(f"- `{rel}:{m.line}`: {m.flat}")
        lines.append("")
    lines += ["## 2. OUTLINE.md section 16: TODO for the author", "", outline_section_16(), "",
              "## 3. Assembly notes", "", ASSEMBLY_NOTES.strip(), "",
              "## 4. Referee items for the author", "", REFEREE_ITEMS.strip(), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}: {len(marks)} \\todo marks in {len(by_file)} files")


if __name__ == "__main__":
    main()
