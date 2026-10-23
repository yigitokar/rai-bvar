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
    2: "Partly done since assembly: the main repository commits `benchmarks/reports/"
       "clark_ravazzolo_acceptance.{json,md,log}` in 65dd212 (2026-10-23 15:47 EDT; on `main` "
       "through the merge 08e1eda). The paper has not been filled: every `\\acc{}` is still a "
       "placeholder (`make fill`, then re-read each sentence; README.md). Whether the committed "
       "`.log` carries the UTC start time that the TODO 2 marks ask for was not checked.",
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
    31: "Not needed under tectonic (XeTeX, TU encoding prints < and >); `main.tex` loads T1 only "
        "under pdfTeX, which is untested here. arXiv compiles with pdfLaTeX, so test that branch "
        "before submitting (`make arxiv` packages the sources; its --test build uses tectonic).",
}

ASSEMBLY_NOTES = r"""
These came up while assembling and integrating the draft and are not (or not only) `\todo` marks in
the text.

1. **Length (after integration, 2026-10-23).** The main text (title page to the end of Section 13)
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
   (untracked) feeds FigD1 and FigD2. It carries no commit or machine field, holds no G-small rows,
   and its G-full row of `reduced_sv` failed with non-finite test functions (seed 20261023): either
   the float64 failure that 3d3f51b fixed (a stale record) or a regression, and the record cannot
   say which. Rerun G-full `reduced_sv` on the release commit before anything else. Both figures
   carry a draft-data footer; replace them with the release records (TODO 7).
6. **Convergence-diagnostic records.** `docs/paper/figures/data/convergence_diagnostic/` (untracked:
   a script and 520 + 424 fit records) reproduces the pre-registration's budget numbers; commit it
   to the main repository (TODO 4) or keep TabF6 at [PREREG].
7. **Abstract length.** About 1840 characters with a short placeholder (arXiv limit 1920). Fill
   `criteria[*].met` with a short phrase.
8. **Sentences built on filled values.** `fill_acceptance.py` prints values, not prose. Re-read every
   sentence with an `\acc` after filling (Sections 1, 11, 12, 13, App. F; the abstract), and write
   the interpretation sentences that the `\todo` marks in Section 11 ask for.
9. **arXiv package.** `make arxiv` writes `arxiv.tar.gz` (main.tex, main.bbl, refs.bib, the
   sections, the figure PDFs, `figures/tables/`, and the four listing inputs in `code/`); its
   `--test` build compiles it with tectonic in a clean directory. arXiv uses pdfLaTeX, whose branch
   of `main.tex` is untested (STATUS of TODO 31). Set `\draftmarksfalse` first.
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
        "Part 3 lists issues found during assembly.",
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
              "## 3. Assembly notes", "", ASSEMBLY_NOTES.strip(), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}: {len(marks)} \\todo marks in {len(by_file)} files")


if __name__ == "__main__":
    main()
