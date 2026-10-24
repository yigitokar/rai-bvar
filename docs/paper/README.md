# The rai-bvar paper (arXiv preprint)

LaTeX sources of *rai-bvar: Composable Bayesian VARs with Stochastic Volatility, Validated
Samplers and a Fair Out-of-Sample Verdict Engine in Python*. The plan is `OUTLINE.md`; the
fact-checked evidence is in `dossiers/`. `main.tex` is set to the submission build
(`\draftmarksfalse`; `\draftmarkstrue` shows the draft marks again). What is left to the author after
submission is in `AUTHOR_TODO.md`.

| Path | What it is |
| :- | :- |
| `main.tex` | master file: preamble, macros, the section inputs |
| `sections/00_abstract.tex` ... `13_conclusion.tex`, `appendix.tex` | one file per section; appendices A-J |
| `refs.bib` | bibliography (natbib, `plainnat`); cite only keys in this file |
| `figures/make_*.py`, `figures/*.pdf` | figure scripts and their output; `figures/data/` holds local inputs |
| `figures/tab_*.py`, `figures/tables/*.tex` | table generators (`tab_common.py` is their shared helper) and the table bodies they write, `\input` by the appendix |
| `figures/acceptance_check.py` | the test that a report is the registered acceptance run |
| `code/ex_a.py`, `code/ex_b.py`, `code/out/` | Listings 1 and 2 and their captured output (`code/run_examples.py`) |
| `fill_acceptance.py` | fills the `\acc{}` placeholders from the acceptance JSON (below) |
| `check_acceptance_numbers.py` | checks every acceptance-run number of the text (filled `\acc{}` values, typed numbers, table keys) against the report JSON; exit 1 on a mismatch |
| `collect_todos.py` | regenerates `AUTHOR_TODO.md` (census of `\todo` marks, which must be 0, and the post-submission items) |
| `texmarks.py` | shared parser for the draft marks |
| `build.sh`, `Makefile` | build and checks |
| `make_arxiv.sh` | writes `arxiv.tar.gz` for arXiv (`make arxiv`) and tests it with tectonic and pdfLaTeX |
| `ARXIV_SUBMISSION.md` | the arXiv metadata to paste, the file to upload, its pdfLaTeX test and the submission steps |

## Building

```sh
cd docs/paper
make pdf                 # tectonic + log checks (build.sh)
make                     # regenerate the figures first, then the PDF
make tables              # regenerate the appendix table bodies (figures/tables/*.tex)
make arxiv               # build, then write arxiv.tar.gz (ARXIV_TEST=1: also compile it in a temp dir)
make arxiv ARXIV_PDFLATEX=1   # ... and compile it with pdfLaTeX in the TeX Live docker image, as arXiv does
tectonic main.tex --keep-intermediates --keep-logs    # the bare build
STRICT=1 ./build.sh      # also fail on undefined references and duplicate labels
```

The build needs [tectonic](https://tectonic-typesetting.github.io/) (XeTeX). `main.tex` also has a
pdfTeX branch (T1 fonts), the one arXiv compiles; `make arxiv ARXIV_PDFLATEX=1` tests it with
pdfLaTeX in the `texlive/texlive:latest` docker image (`ARXIV_SUBMISSION.md`). `build.sh` lists, from `main.log`: the section file
each slot read, any orphan section file, missing glyphs, undefined citations and references,
duplicate labels, overfull boxes wider than 10pt, the number of draft marks left, and the page
count.

### Draft marks

| Macro | Meaning | Draft | `\draftmarksfalse` |
| :- | :- | :- | :- |
| `\acc{key}` | a number of the acceptance run, read from its JSON | red `[ACC: key]`, or the filled value in green | value, or a LaTeX error if unfilled |
| `\fullrun{...}` | a number of the first full run (report 551f181): history only | blue | plain text |
| `\todo{...}` | something only the author can decide or verify | orange box | LaTeX error |

Evidence tags are LaTeX comments at the end of the sentence they support (`% [V] ...`, `[R]`,
`[TC]`, `[REL]`, `[ACC]`, `[FULL]`, `[PREREG]`; `OUTLINE.md` section 0.5). `main.tex` is set to the
submission build, `\draftmarksfalse`: every unfilled `\acc` and every `\todo` stops the build (none is
left).

Other conventions. `\pkg{name}` for packages. `\code{...}` for identifiers, paths and test IDs,
with underscores plain or escaped; it works in text, math and captions. Cross-references use
`\cref` (`\eqref` for equations).

## Filling the acceptance placeholders

The acceptance run executes from the detached worktree `/Users/yigitokar/rai-bvar-accept` at
c823d20. No number of that run is written in the sources: each is a placeholder
`\acc{json.key.path}`, in the key grammar documented at the top of `sections/11_replication.tex`.
The sources keep their placeholders. The values go into a generated macro file, which `main.tex`
reads if it exists.

1. When the run has finished, commit `clark_ravazzolo_acceptance.{json,md}` to the main repository
   and archive the run log (OUTLINE TODO 2).
2. From `docs/paper`, run
   ```sh
   python3 fill_acceptance.py      # or: make fill
   # = python3 fill_acceptance.py /Users/yigitokar/rai-bvar/benchmarks/reports/clark_ravazzolo_acceptance.json
   ```
   The script first checks the report's header against the pre-registration
   (`figures/acceptance_check.py`, the same check the acceptance figures use): `config.name`
   `acceptance`, `lpd_method` `integrated`, 8000 forecast draws, 4 x 20,000 kept draws with burn-in
   1000, seed 0, commit c823d20, `git_dirty` false, no `pending` key, all stages complete. It
   refuses any other report. It then resolves every distinct `\acc` key of `sections/*.tex` (200
   at the time of writing). It writes `acceptance_values.tex`, one
   `\accdefine{key}{value}` line per key, each commented with the places that use it, under a
   header with the report path, its SHA-256, its `generated` field and its commit. If any key does
   not resolve, it lists the key with its file and line and writes nothing.
3. Rebuild (`make pdf`). Filled values print in green in the draft build.
4. Re-read every sentence that contains a filled value. The script prints values, not prose (the
   outcome sentences of the abstract and Sections 11-13 are written from the current report). Check
   that the abstract stays under arXiv's 1920 characters. Then `python3 collect_todos.py`.

Useful options:

```sh
python3 fill_acceptance.py REPORT --list       # print every key and the value it would get
python3 fill_acceptance.py REPORT --partial    # write the keys that resolve; the rest stay placeholders
# dry run on another report, e.g. the first full run: never into docs/paper
python3 fill_acceptance.py ../../../rai-bvar/benchmarks/reports/clark_ravazzolo.json \
    --allow-non-acceptance --partial --out /tmp/dry/acceptance_values.tex
```

`--allow-non-acceptance` refuses an output path inside `docs/paper`, so first-run numbers cannot
end up typeset as acceptance values. Delete `acceptance_values.tex` to go back to placeholders.

Formatting is fixed by the last key of each path (`FORMATS` in the script):

| Kind | Format |
| :- | :- |
| `dlps`, `dlpd` and their `_conditioned`/`_gauss` variants, `dlps_median`, `dlps_top[i][1]` | signed, three decimals (`dlps_top[i][0]`, the origin, as text) |
| p-values | three decimals, `<0.001` below that |
| ratios and ranges | three decimals |
| converged shares | percentages |
| counts | integers |
| booleans | `true`/`false` in code font; "yes"/"no" for `rb_gauss_disagree`; "met"/"not met" for criteria |
| verdicts | quoted |
| `criteria[*].met` | one phrase ("criteria 1, 2, 3 and 5 met; 4 and 6 not met") |
| report strings (criteria details) | escaped for LaTeX |

Change a format there, not in the sources.

## Figures

```sh
make figures        # every script; placeholders where the data do not exist yet
make acc-figures    # Fig6 and FigF2 from the acceptance report (ACC=path), FigF1/FigF3 from the export
make per-origin     # export the acceptance run's per-origin scores from its cache (read-only; once)
```

The scripts run with the main repository's environment
(`REPO=/Users/yigitokar/rai-bvar`, `PY=$(REPO)/.venv/bin/python`). They import its code read-only,
with `PYTHONDONTWRITEBYTECODE=1`, and write `figures/<name>.pdf`.

| Figure | Script | Status |
| :- | :- | :- |
| Fig1 architecture | `make_fig_architecture.py` | static |
| Fig3 verdict flow | `make_fig_verdict_flow.py` | static, transcribed from `CompareResult.verdict` |
| Fig4 LPD estimators | `make_fig_lpd_estimators.py` | left panel computed from committed code; right panel drawn from `data/rb_tail_cell.json` (written by `figures/rb_tail_cell.py` against main 99f658a, whose `src/` equals the release code) |
| FigC1 Gibbs plan | `make_fig_plan.py` | computed (fits two tiny models; about a minute) |
| FigD1 mutation heatmap | `make_fig_mutation_heatmap.py` | drawn from the release validation record at 7f880e1 (`benchmarks/reports/validation/7f880e1_apple-m5-max/record/mutation.jsonl`), named in a footer |
| FigD2 Geweke dots | `make_fig_geweke_dots.py` | same record (`geweke.jsonl`, `mutation.jsonl`); `--geweke`/`--mutation` select other records, and the footer then says they are not the release record |
| FigE1 RW pivots | `make_fig_laplacian_pivots.py` | computed from committed code; `--json PATH` writes the numbers (a script, not a test: no test asserts the old-path values, App. E) |
| Fig6, FigF2 | `make_fig_cr_orderings.py`, `make_fig_cr_convergence.py` | placeholders until the acceptance JSON exists |
| FigF1, FigF3 | `make_fig_cr_per_origin.py`, `make_fig_cr_cumulative.py` | drawn from the per-origin export `figures/data/cr_acceptance/` (tracked on `paper-draft`, TODO 5); their captions say so |

Like `fill_acceptance.py`, the acceptance figures refuse a report that is not the registered run.
They dry-run on another report only with `--allow-non-acceptance --outdir DIR`.

## Tables

`make tables` runs the 14 generators `figures/tab_*.py` (main repository environment, as for the
figures); each writes the body of one appendix float to `figures/tables/<stem>.tex`, which
`sections/appendix.tex` reads with `\input`. A generated file holds only the `tabular` and its
notes; the float, caption, label and evidence tags stay in the appendix. Cells that wait for data
are marks: `\acc{key}` (acceptance run; filled by `make fill`, which scans `figures/tables/` as well
as `sections/`) and a red `[REL]` for a cell whose release validation record is missing, which stops
a `\draftmarksfalse` build like `\acc` and `\todo`. The release records are read from
`benchmarks/reports/validation/7f880e1_apple-m5-max/` of this repository (`tab_common.RELEASE_DIR`):
the JSON records in `record/`, the JUnit XML per test directory and `meta.json`; the per-PR tier on
the release commit from the committed CI log excerpt
`figures/data/provenance/ci_run_40061724239_fast_suite.txt`. No `[REL]` cell is left. The other
tables (Tab1, Tab7, Tab13 in the main text; Tab2, Tab5 and Tab12, moved to Appendices C, D and F;
TabA1, TabC1, TabF2, TabF3, TabG1 and TabH1) are typed into the sources; the generators that OUTLINE
section 14 planned for them were not written.

## arXiv package

`make arxiv` rebuilds the PDF (so `main.bbl` is current) and runs `make_arxiv.sh`, which writes
`arxiv.tar.gz` with `main.tex`, `main.bbl`, `refs.bib`, the section files, the figure PDFs that the
sources include, the table bodies in `figures/tables/`, `acceptance_values.tex` if it exists, and
the four files that the listings print (`code/ex_a.py`, `code/ex_b.py`, `code/out/ex_a.txt`,
`code/out/ex_b.txt`). No scripts are shipped otherwise, and no data. `./make_arxiv.sh --test` (or
`make arxiv ARXIV_TEST=1`) also compiles the unpacked tarball in a clean temporary directory with
tectonic. arXiv itself compiles with pdfLaTeX, i.e. the pdfTeX branch of `main.tex` (`main.tex` is
already set to `\draftmarksfalse`): `./make_arxiv.sh --pdflatex` (or `make arxiv ARXIV_PDFLATEX=1`)
compiles the unpacked tarball with pdfLaTeX in the `texlive/texlive:latest` docker image twice, once
as arXiv does (three pdflatex runs on the shipped `main.bbl`, no BibTeX) and once with BibTeX, whose
regenerated `main.bbl` must equal the shipped one. `ARXIV_SUBMISSION.md` records the result, the
differences from the tectonic PDF, and the metadata and steps for the submission.

Listings 1 and 2: `python3 code/run_examples.py` runs `code/ex_a.py` and `code/ex_b.py` in the main
repository (with `uv run --frozen --no-sync`, so nothing is installed there) and rewrites `code/out/`
(stdout, stderr, `env.json`, which records the commit and the tracked changes before and after the
run). The listings were last run on 2026-10-24 at the release commit 4ac7969 with a clean tree; stdout
equals the earlier d58c9ef capture except the version line of `ex_b.txt`, and `ex_b.stderr.txt` adds
the D-35 budget warning (the listing sets 4 x 1000 kept draws).

## Where the data files live

State on branch `paper-draft` (fd8a3f7 merges `main` at 4ac7969, release 0.2.0); see
`REVIEW_RESPONSE.md` (software referee, minor 5).

- **On `main` (merged into `paper-draft`):** the release validation record
  `benchmarks/reports/validation/7f880e1_apple-m5-max/` and the x86-64 BENCH-P2 report
  `benchmarks/reports/bench_p2_x86_github_779957f.json` (PR #9). The earlier local records of
  unknown commit and machine (`figures/data/validation_record/`) are deleted.
- **Tracked on `paper-draft`, not on `main`:** `figures/data/convergence_diagnostic/` (the
  fit-level diagnostic of the pre-registration: a script and 520 + 424 fit records; not part of
  release 0.2.0, as App. F says) and `figures/data/provenance/` (GitHub activity record; excerpts of the
  logs of CI runs 40061724239, the fast suite at 7f880e1, and 40052069815, the x86-64 weekly tier at
  779957f; the test-collection counts at the release code).
- **Tracked on `paper-draft` since a3478d4:** `figures/data/cr_acceptance/` (the per-origin export
  behind FigF1 and FigF3, with its `.meta.json` and README; OUTLINE TODO 5) and
  `acceptance_values.tex` (generated by `fill_acceptance.py`; arXiv needs it, and it records which
  report filled the numbers).
- **Tracked build output:** `main.pdf`. The LaTeX intermediates and `arxiv.tar.gz` are ignored
  (`.gitignore`).
- **On `main` (888f620) and, through the merge 75db8a7, on `paper-draft`:** the acceptance report
  `benchmarks/reports/clark_ravazzolo_acceptance.{json,md,log}` (65dd212) and the pre-registration
  (c823d20).
