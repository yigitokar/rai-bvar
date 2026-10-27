#!/usr/bin/env python3
r"""Write AUTHOR_TODO.md for the submission: a census of the \todo{...} marks of the sources (which
must be empty, because a \todo is a LaTeX error under \draftmarksfalse) and the items that are left
to the author after submission.

    python3 collect_todos.py            # rewrites docs/paper/AUTHOR_TODO.md; exit 1 if a \todo is left

The census is regenerated from the sources each time; the post-submission items are kept in
AFTER_SUBMISSION below. The drafting-stage lists (the status of OUTLINE.md section 16, the assembly
notes and the referee backlog) were dropped from the output for the submission build; they are in
the earlier AUTHOR_TODO.md (``git show 83f2b56:docs/paper/AUTHOR_TODO.md``), OUTLINE.md section 16
and REVIEW_RESPONSE.md.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from texmarks import find_marks  # noqa: E402

OUT = HERE / "AUTHOR_TODO.md"

AFTER_SUBMISSION = r"""
These are the only open items for the author. The paper claims no reading statement, PyPI
release or DOI, so its text is correct without them; each item says what to change in the paper if
it is done. `ARXIV_SUBMISSION.md` has the arXiv metadata, the file to upload (tested with pdfLaTeX)
and the submission steps.

1. **Your own statement for App. J (optional).** The paper makes no claim about how far you read the
   code, the derivations, the pull requests or the paper: the earlier `author:` marks and the
   sentence "The record does not show how far he read ..." were removed, and no statement replaces
   them. Nor does it say whether you reviewed the acceptance run's pre-registered settings before
   the launch (the clause "The author did not review those settings before the launch" was removed
   from App. J, paragraph "Runs"; the paper says only that the session fixed them under your
   standing instruction). If you want a statement, add it to the paragraph "Authorship and
   responsibility" of App. J
   (`sections/appendix.tex`, label `app:process`) in your own words, together with your approval of
   App. J and the authorship statement against arXiv's policy and the target venue's, and any
   acknowledgements.

2. **Release: PyPI and Zenodo (optional).** Done on 2026-10-24: the repository is public, the tag
   `v0.2.0` at 4ac796997df80157d086c568714168fc2387f093 and its GitHub release exist, and the branch
   `paper-draft` is pushed (App. F, App. G and the caption of Fig. F.1 point to it). The install line
   of App. G reads `pip install git+https://github.com/yigitokar/rai-bvar@v0.2.0`; at 09:41Z it
   installed rai-bvar 0.2.0 from the network into a fresh Python 3.12 environment.
   - A PyPI release and Zenodo DOIs (the release, and the replication outputs) do not exist and the
     paper claims none ("there is no PyPI release and no archived DOI", App. G; "it is not on PyPI",
     Section 12, "Release state"). If you create them, update those two places and the install line.

3. **Third-party observations (OUTLINE TODO 13).** After the release the paper states one
   observation about another package, as a neutral black-box observation and, by your decision,
   without notifying its maintainers first ("Haber vermeden yaz"): on one fixture the SV predictive
   that `bayesianVARs` 0.1.8's `predict()` returns at h >= 2 differs from the exact oracle's, while a
   path recursion from its own posterior draws agrees; no cause is claimed (App. D, "Parity with
   independent implementations"). Contacting the `bayesianVARs` maintainers later stays your choice.
   The paper states no finding about the other packages' samplers (the notes for other
   implementations in Section 9 name no package). Before any such statement: re-test the `stochvol`
   sign-fold (D-7) and Student-$t$ observations on the current `stochvol` release
   (`tests/parity/SV_PARAMETRIZATION.md` section 3.2), and the R `BVAR` 1.0.5 and Bank of England
   `bvar` 0.3.5 items on current versions; then contact the maintainers of
   `stochvol`/`factorstochvol`, R `BVAR` and the Bank of England `bvar`, using the maintainer fields
   on CRAN and PyPI.

4. **PR #15 (the DGP study, App. D.11).** App. D.11, TabDdgp (`figures/tab_dgp.py`) and the
   evidence comments of Sections 1, 6 and 12 cite commits 58ab09f (where all 600 records were made;
   each record's `git_commit`) and 75e0c2a (the reports), which are on the open draft PR #15
   (branch `claude/compassionate-bose-ed1c0b`), not yet on `main`. Keep that branch, and merge PR #15
   with a merge commit (not squash or rebase), so that both commits stay reachable and `make tables`
   keeps working. `RUN_COMMIT` in `tab_dgp.py` must stay 58ab09f, because the records name it; after
   a merge, the paper may add "merged into main as <commit>" next to "commit 75e0c2a (PR #15)".
"""


def main() -> int:
    sources = [*sorted((HERE / "sections").glob("*.tex")), HERE / "main.tex"]
    marks = [m for m in find_marks("todo", sources) if not (m.rel == "main.tex" and "abstract not written yet" in m.flat)]
    by_file = defaultdict(list)
    for m in marks:
        by_file[m.rel].append(m)

    lines = [
        "# Author TODO list for the rai-bvar paper (after submission)",
        "",
        f"Generated by `docs/paper/collect_todos.py` on {date.today().isoformat()}; rerun it after "
        "editing the sources. Part 1 counts the `\\todo{}` marks left in `sections/*.tex` and "
        "`main.tex` (each is a LaTeX error in the submission build, `\\draftmarksfalse`, so the count "
        "must be 0). Part 2 lists what is left to the author after submission. The drafting-stage "
        "lists (the status of OUTLINE.md section 16, the assembly notes and the referee backlog) are "
        "in the earlier version of this file (`git show 83f2b56:docs/paper/AUTHOR_TODO.md`), in "
        "OUTLINE.md section 16 and in REVIEW_RESPONSE.md.",
        "",
        f"## 1. `\\todo` marks in the sources ({len(marks)})",
        "",
    ]
    if not marks:
        lines += ["None.", ""]
    for rel in sorted(by_file, key=lambda r: (r == "main.tex", r)):
        lines.append(f"### `{rel}` ({len(by_file[rel])})")
        lines.append("")
        for m in by_file[rel]:
            lines.append(f"- `{rel}:{m.line}`: {m.flat}")
        lines.append("")
    lines += ["## 2. For the author after submission", "", AFTER_SUBMISSION.strip(), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}: {len(marks)} \\todo marks in {len(by_file)} files")
    return 1 if marks else 0


if __name__ == "__main__":
    sys.exit(main())
