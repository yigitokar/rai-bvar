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
    2: "Open. No `clark_ravazzolo_acceptance.json` in the main repository's `benchmarks/reports/` "
       "at assembly time. When it is committed, run `python3 fill_acceptance.py` (README.md).",
    3: "Partly done after the acceptance commit: 7f410dc (2026-10-23 14:44 EDT) registers row 18 "
       "`reduced_sv_rw` (G-small per PR, G-full and SBC weekly with a per-row t_max of 400) and adds a "
       "`reduced-SV(rw)` contract fit; the SV(rw, t) variant is not covered. The paper's coverage "
       "statements are pinned to c823d20 (the acceptance run's commit) and stay true; the seven "
       "TODO-3 marks in the sources say what to change if the release commit includes 7f410dc.",
    20: "The paper's `Makefile` exists (`docs/paper/Makefile`: `make`, `make figures`, ...); the table "
        "generators `tab_*.py` named in the captions do not.",
    31: "Not needed under tectonic (XeTeX, TU encoding prints < and >); `main.tex` loads T1 only "
        "under pdfTeX, which is untested here.",
}

ASSEMBLY_NOTES = r"""
These came up while assembling the draft and are not (or not only) `\todo` marks in the text.

1. **Length.** The main text runs to about 47 pages in the draft build (about 45 with the `\todo`
   boxes removed and numbers in place of the `\acc` placeholders) against the outline's 29.5;
   the appendices run to about 36 pages against 20-25. Per section (draft build): introduction
   3.9 (budget 2.5), related 1.5 (1), framework 4.6 (2.5), architecture 3.3 (2.5), samplers 3.4
   (2.5), evaluation 5.5 (3), walk-through 4.7 (2), validation 7.5 (4.5), robustness 2.0 (1.5),
   performance 0.7 (0.75), replication 6.4 (4), limitations 1.9 (1.75), conclusion 1.3 (1).
   Cut candidates named by the section writers: Tab5 to App. D (risk 23); the two listing outputs of
   Section 7 to an appendix; Fig3 to App. C; the "Later" rows of Tab1 into its caption; the Mode I
   display and the Mode II / iw_equivalent detail of Section 3 to App. B; the eight-citation
   cluster in C5 and the paper map of Section 1; Section 9.7 into Section 12; App. C.9-C.12 and
   TabF8 to a supplement.
2. **Table generators.** None of the `tab_*.py` scripts named in captions and comments exists;
   the 14 `\InputIfFileExists{figures/tab_*.tex}` tables of the appendices print a `\todo` until
   they do. Each generated file must contain only a `tabular` (the float, caption and label are in
   `sections/appendix.tex`).
3. **FigE1 (TODO 23).** The figure, computed from committed code, shows the old assembled RW path
   returning badly wrong pivots and h levels at sigma <= 1e-8 without failing. Section 9 withholds
   that observation until a committed test exists. Decide: keep and state it, or crop the figure.
4. **Validation records of unknown provenance.** `docs/paper/figures/data/validation_record/`
   (untracked) feeds FigD1 and FigD2. It carries no commit or machine field, holds no G-small rows,
   and its G-full row of `reduced_sv` failed with non-finite test functions (seed 20261023): either
   the float64 failure that 3d3f51b fixed (a stale record) or a regression, and the record cannot
   say which. Rerun G-full `reduced_sv` on the release commit before anything else. Both figures
   carry a draft-data footer; replace them with the release records (TODO 7).
5. **Convergence-diagnostic records.** `docs/paper/figures/data/convergence_diagnostic/` (untracked:
   a script and 520 + 424 fit records) reproduces the pre-registration's budget numbers; commit it
   to the main repository (TODO 4) or keep TabF6 at [PREREG].
6. **Abstract length.** About 1840 characters with a short placeholder (arXiv limit 1920). Fill
   `criteria[*].met` with a short phrase.
7. **Sentences built on filled values.** `fill_acceptance.py` prints values, not prose. Re-read every
   sentence with an `\acc` after filling (Sections 1, 11, 12, 13, App. F; the abstract), and write
   the interpretation sentences that the `\todo` marks in Section 11 ask for.
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
