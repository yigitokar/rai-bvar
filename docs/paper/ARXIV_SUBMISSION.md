# arXiv submission of the rai-bvar paper

What to paste into the arXiv form, which file to upload, and how that file was tested with arXiv's
engine (pdfLaTeX). State of 2026-10-24, from the sources of the latest commit that changes this
file on branch `paper-draft` (first written in 8511b6c; Sections 0, 1 and 4 updated after the final
adversarial check of the same day; Sections 0, 1, 2, 4, 5 and 6 updated after the post-release F1
edits of cd8d605). Every number below comes from a build or a command run that
day; the commands are given so that they can be repeated.

## 0. Before you submit

Done on 2026-10-24:
- **Repository public:** https://github.com/yigitokar/rai-bvar (`gh repo view --json visibility` returns `PUBLIC`).
- **Tag and release:** tag `v0.2.0` at 4ac796997df80157d086c568714168fc2387f093, with a GitHub release (https://github.com/yigitokar/rai-bvar/releases/tag/v0.2.0). The install line in App. G uses `@v0.2.0`.
- **Branch pushed:** `paper-draft`, which holds the paper's sources, figure and table scripts and data that App. F, App. G and Fig. F.1 point to.
- **Date:** the title-page date is fixed to "October 2026" (`main.tex`, submission build), so arXiv's compile date does not appear.
- **After the release:** PR #11 (merge 3644c4b) replaced the non-portable F1 replay test; Sections 9 and 12 and Apps. D, E and J describe it (cd8d605), and the paper still describes release 0.2.0.

Still yours, both optional:
- Your own statement for App. J (AUTHOR_TODO.md, item 1). The paper makes no claim about how far you read the code or the paper, nor about whether you reviewed the acceptance run's pre-registered settings before its launch.
- A Zenodo DOI or a PyPI release, if wanted (neither exists; the paper says so).

## 1. The file to upload

`docs/paper/arxiv.tar.gz`: 48 files, 507,855 bytes, SHA-256
`8dd62127f7af2216ce6387f0c310ef5e55d2ddd88ac44c5f478197452023957a` (the tarball tested below,
built 2026-10-24 17:05Z by `STRICT=1 make arxiv ARXIV_TEST=1 ARXIV_PDFLATEX=1` from the sources of
the commit that records this checksum).

arXiv publishes the source files of a submission, so `make_arxiv.sh` ships its 31 `.tex` files
(`main.tex`, `sections/*.tex`, `figures/tables/*.tex`, `acceptance_values.tex`) stripped of
comment text by `strip_comments.py`; the annotated sources, with their evidence notes and TODO
tags, stay on the branch `paper-draft`. Checked on 2026-10-24:
- stripping the committed `.tex` files reproduces the 31 shipped copies byte for byte, and the
  other 17 files are byte-identical to the sources (`main.bbl` is generated from `refs.bib` and
  gitignored; the test's TeX Live 2026 BibTeX regenerates it byte for byte);
- the stripped tarball typesets to the same PDF as an unstripped one: the pdfLaTeX builds of both
  have 107 pages with byte-identical page content streams (compared with pypdf; re-checked for the
  tarball above, the unstripped files taken from cd8d605);
- the shipped `.tex` files hold no comment text; the only local path left in the tarball is in the
  header comments of `code/ex_a.py` and `code/ex_b.py`, lines the listings do not print.

`./make_arxiv.sh --keep-comments` ships the `.tex` files with their comments instead.
The tarball is not committed (`.gitignore`). Rebuild and test it with

```sh
cd docs/paper
STRICT=1 make arxiv ARXIV_TEST=1 ARXIV_PDFLATEX=1
```

A rebuilt tarball has a different checksum (tar stores file times), not different content. The
script prints the file list and ends with `pdfLaTeX test OK`.

Contents (`make_arxiv.sh` collects them from what the sources read):

- `main.tex`, `main.bbl` (arXiv does not run BibTeX; it reads this file), `refs.bib`,
  `acceptance_values.tex` (the filled acceptance numbers; the build needs it);
- `sections/*.tex` (15 files: the abstract, Sections 1 to 13, the appendices);
- `figures/tables/*.tex` (14 generated table bodies);
- `figures/*.pdf` (the 11 figure PDFs that the sources include);
- `code/ex_a.py`, `code/ex_b.py`, `code/out/ex_a.txt`, `code/out/ex_b.txt`: the inputs of
  `\lstinputlisting` for Listings 1, 2 and G.1 to G.4. The paper prints them, so the build needs
  them; they are not build scripts and nothing runs them.

Nothing else: no figure or table scripts, no `code/run_examples.py`, no data, no logs, no
dossiers, no `main.pdf`, no intermediates. There is a single top-level file with
`\documentclass` (`main.tex`), and `main.bbl` has the same base name, as arXiv requires.

## 2. Test with pdfLaTeX (arXiv's engine)

Engine: docker image `texlive/texlive:latest`, which reports pdfTeX 3.141592653-2.6-1.40.29
(TeX Live 2026), LaTeX2e <2026-06-01> and BibTeX 0.99e (TeX Live 2026). This note does not say
which TeX Live arXiv runs; check arXiv's help pages if it matters. `./make_arxiv.sh --pdflatex`
unpacks the tarball twice, each time into a fresh temporary directory, and compiles it there:

| Build | Runs | Exit codes | TeX errors | Undefined citations / references | Rerun requests | Missing characters | Overfull boxes | Pages |
|---|---|---|---|---|---|---|---|---|
| (a) as arXiv does | pdflatex ×3 on the shipped `main.bbl`, no BibTeX | 0 0 0 | 0 | 0 / 0 | 0 | 0 | 0 | 107 |
| (b) with BibTeX | pdflatex, bibtex, pdflatex, pdflatex | 0 0 0 0 | 0 | 0 / 0 | 0 | 0 | 0 | 107 |

- Missing characters are logged: this LaTeX sets `\tracinglostchars=2` by default (checked with a
  test file), and none were logged.
- The `.bbl`: build (b)'s BibTeX 0.99e with TeX Live 2026's `plainnat.bst` regenerated a
  `main.bbl` byte-identical to the shipped one, which tectonic's BibTeX 0.99d wrote (natbib,
  `plainnat`, 129 entries, 0 BibTeX warnings). The shipped `.bbl` therefore matches TeX Live 2026.
- The PDFs of (a) and (b) have identical text on all 107 pages.
- Converged: a fourth pdflatex pass on (a) leaves `main.aux` unchanged. tectonic's own rerun rule
  is not as reliable: from no `main.aux`, the sources of cd8d605 stopped one pass short, at 108
  pages with the main text spilling onto p. 36, and one more pass gave the converged 107. `build.sh`
  and `make_arxiv.sh --test` therefore rerun tectonic while `main.aux` still changes (at most three
  more times); a fresh and an incremental `build.sh` now give the same text on all 107 pages.
- Not counted as problems: 52 underfull-box messages (`build.sh` does not check these either).
  One of them quotes the code name `TracerArrayConversionError`, which is text from the paper,
  not a TeX error.

## 3. What was changed for pdfLaTeX (in commit 8511b6c)

Before these changes the package compiled under pdfLaTeX without errors, but the PDF was wrong in
three ways (items 2 to 4). With items 1 to 4 alone, its text still differed from the tectonic
build on 59 of 106 pages (item 5).

1. **`\pdfoutput=1` in the first line** of `main.tex`, for pdfTeX only
   (`\ifdefined\pdftexversion ... \fi`). tectonic also defines `\pdfoutput`: with an unguarded
   `\pdfoutput=1`, hyperref loaded its pdfTeX driver under XeTeX and the tectonic build stopped
   with an error.
2. **"Appendix", not "Section".** With TeX Live 2026, cleveref printed every reference to an
   appendix as "Section X". That includes the title-page footnote ("Section J states what the
   agents did") and the `\cref` references to App. A–J throughout. The cause is that cleveref no
   longer gives the appendix sections their own type after `\appendix`: the `.aux` has
   `[section]` where tectonic's has `[appendix]`. `\crefalias{section}{appendix}` (and the same
   for subsections and subsubsections) after `\appendix` in `main.tex` fixes it under both
   engines, as a minimal test file showed under each.
3. **Straight quotes in `\code`.** In T1 encoding, the typewriter font has curly quotes in the
   slots of `'` and `` ` ``, so `\code{Constant('iw_equivalent')}` printed `Constant(’iw_equivalent’)`.
   7 `\code` uses contain a quote. On the pdfTeX branch, `\code` now maps them to
   `\textquotesingle` and `\textasciigrave`. The listings were already correct (`upquote=true`).
4. **`--cache` in App. F.** The `--` of `\texttt{--cache ...}` became an en dash in T1
   typewriter; it is now written `-{}-cache`.
5. **No font expansion** (`\usepackage[expansion=false]{microtype}`). XeTeX has no font expansion
   and pdfTeX has it by default, so lines and pages broke differently. With expansion on, the text
   differed on 59 of 106 pages and floats in App. D to F moved by up to a page. With it off, 33
   pages differ, by lines that move across page breaks (Section 4, item 2).

Under tectonic, all five changes leave the output as it was. Compared with the build of the
previous commit (bd08c75), the PDF has identical text on all 106 pages and `main.aux` is
identical. `STRICT=1 ./build.sh` gives 0 missing characters, 0 undefined citations and
references, 0 multiply-defined labels and 0 overfull boxes, on 106 pages.

## 4. pdfLaTeX PDF against the tectonic PDF

Compared by text extraction (PyMuPDF) and by the `.aux` label tables of the two builds.

1. **The same document.** Both have 107 pages. In both, the main text runs from p. 1 to p. 35,
   App. A starts on p. 36, and the appendices end and the references start on p. 98. Both have 11
   figures, 29 tables, 6 listings, 13 algorithms and 129 references. All 245 labels have the same
   numbers in both builds. (Re-measured on 2026-10-24 for the tarball of Section 1, the tectonic
   side being `build.sh`'s converged build of cd8d605.)
2. **Page drift.** 232 of the 245 labels fall on the same page. Thirteen are one page off, all
   pdfLaTeX one page earlier except three: `sec:walk-mcmc` (p. 19 in tectonic, p. 18 in pdfLaTeX),
   `sec:validation-cost` and `sec:validation-tiers` (20, 19), `sec:robustness-f3` (26, 25),
   `sec:replication-prereg` and `sec:replication-first-run` (29, 28), `sec:replication-caveats`
   (31, 32), Fig. 6 (`Fig6`, `fig:Fig6`, `fig:cr-orderings`: 32, 31), `sec:conclusion` (35, 34),
   `app:licensing` (93, 94) and `app:errata` (94, 95). (An earlier version of this note listed
   five; measured the same way, the sources of 59573af already gave fifteen.)
   - Pages 1–18, 36–41 and 44–90 have the same text, up to the extraction differences of item 4
     (a word-sequence similarity of at least 0.97 per page: PyMuPDF text, whitespace-split words,
     Python's difflib). On pp. 41–42 one line of App. B.5 crosses the page break (pdfLaTeX sets it
     on p. 41), which, with the math of App. B.5 and B.6, puts pp. 42–43 at 0.956 and 0.960; no
     label moves there.
   - The drift starts after Listings 1 and 2 (pp. 18–19) and after Listings G.1–G.4
     (pp. 88–92). It ends at the page break before the appendices and at the end of the document.
   - The code listings use different typewriter fonts in the two engines: DejaVu Sans Mono under
     XeTeX and Latin Modern Typewriter under pdfTeX (`\lstfontfamily` in `main.tex`).
3. **Visible glyph differences, in the listings only.** tectonic prints `-` and `*` as a minus
   sign and an asterisk operator; pdfLaTeX prints the ASCII characters. Δ in the package output
   is a math Δ under pdfLaTeX.
4. **Not differences.** The two PDFs also extract differently in math (∑, √, large delimiters),
   in ligatures and in hyphenation. These come from the fonts' encodings, not from the typeset
   page.

## 5. Metadata to paste

**Title**

```
rai-bvar: Composable Bayesian VARs with Stochastic Volatility, Validated Samplers and a Fair Out-of-Sample Verdict Engine in Python
```

**Authors** (arXiv's format for affiliations; no e-mail, as on the title page)

```
Yigit Okar (1 and 2) ((1) Rational AI, Inc., New York, NY, USA, (2) Care Artificial Intelligence Research and Consulting LLC, Brooklyn, NY, USA)
```

**Abstract**: 1899 characters, within arXiv's limit of 1920. It is plain ASCII on one line,
converted from `sections/00_abstract.tex` by script:

- comments dropped;
- `\pkg{rai-bvar}` becomes rai-bvar;
- `--` becomes `-`;
- math loses its `$` signs (Student-t, h > 1, h = 8);
- `\%` becomes `%`;
- `\citet{clark2015macroeconomic}` becomes "Clark and Ravazzolo (2015)" (the PDF prints "2015a",
  because the bibliography has two 2015 entries of theirs);
- `\fullrun{2 of 6}` becomes "2 of 6";
- the acceptance value `count(criteria[*].met = true)` becomes 5, from `acceptance_values.tex`.

```
rai-bvar, an Apache-2.0 Python package, is built on the position that a density-forecast gain credited to a richer mean or prior in a Bayesian VAR (BVAR) is interpretable only with the volatility specification held fixed. A BVAR composes mean, volatility and prior components, which select the engine. Conjugate Minnesota models, with dummy observations and the Giannone-Lenza-Primiceri hierarchy, go to an analytic engine. Stochastic volatility, outliers and Student-t errors run on hand-written JAX Gibbs samplers, in recursive structural or corrected reduced form. Both engines return one contract, path draws with per-draw conditional Gaussian moments, so log scores are Rao-Blackwellised (RB). At h > 1 an opt-in MCMC estimator integrates intermediate outcomes out. A verdict engine calls a model better only if it differs from its reference along one axis, gains over three Monte Carlo standard errors in log score, is separated from it by a rolling-window test or the model confidence set (unproven under expanding windows), and both converged at 95% or more of the origins. Samplers are checked by identities, block-level tests, Geweke tests, simulation-based calibration (SBC), analytic twins and parity tests; full-size Geweke and SBC tests pass on the release commit. Eighteen curated bugs form a release gate for the identity, block-level, Geweke and SBC tiers: in all 37 mutant-by-tier checks the bug is caught and the unmodified sampler passes. An illustrative replication of Clark and Ravazzolo (2015) on FRED-QD missed 2 of 6 criteria in a first full run; after remedies registered before a second run (not blind to the first), the acceptance run met 5 of 6 as coded. The unmet one asks the unemployment gain at h = 8 to be below that at h = 1: on the path-integrated RB score it is larger, mostly through three crisis origins; on their Gaussian score it shrinks, as in the original.
```

**Comments**

```
107 pages (main text 35 pages, Appendices A-J, references), 11 figures, 29 tables. Code (Apache-2.0): https://github.com/yigitokar/rai-bvar ; release 0.2.0 (tag v0.2.0) is commit 4ac7969
```

**Primary category: stat.CO** (Statistics, Computation). The paper is a software paper about
statistical computing. Its contributions are the package's architecture, hand-written Gibbs
samplers and their validation (Geweke tests, SBC, mutation tests, analytic twins), numerical
robustness and performance.

**Cross-lists:**

- **econ.EM** (Econometrics). The models are macroeconometric: BVARs with Minnesota priors,
  stochastic volatility and structural identification. The illustration replicates a
  macroeconomic forecasting study, and econometricians are the main users of such a package.
- **stat.ME** (Statistics, Methodology). The predictive contract, the Rao-Blackwellised and
  path-integrated log scores, and the verdict rule (Monte Carlo standard errors, rolling-window
  tests, model confidence set, convergence gate) are methods for evaluating density forecasts.
  They are not specific to this package.

If you see the paper first as an econometrics contribution, swap stat.CO and econ.EM. arXiv may
ask for an endorsement the first time you submit to a category.

**Optional fields.**

- MSC class: `62M10, 62F15, 62-04, 65C05` (time series; Bayesian inference; statistical
  software; Monte Carlo methods).
- Leave ACM class, report number, journal reference and DOI empty: there is no journal version,
  and the release has no archived DOI.

**License: CC BY 4.0** (suggested; your choice). It lets anyone share and adapt the paper with
attribution, the counterpart for the text of the open Apache-2.0 code. The two licences are
independent. An arXiv licence cannot be withdrawn later. If a journal you target needs an
exclusive transfer of copyright, choose arXiv's non-exclusive distribution licence instead.

## 6. Steps

1. Section 0 is done. Check that the repository URL opens in a private
   browser window, and that `git ls-remote origin paper-draft` shows the commit that adds this
   file, or a later one.
2. `cd docs/paper && STRICT=1 make arxiv ARXIV_TEST=1 ARXIV_PDFLATEX=1`. It must end with
   `pdfLaTeX test OK`, and the tectonic test must report 107 pages.
3. On arxiv.org, choose "Start new submission". Select the license (Section 5) and the primary
   category stat.CO.
4. Upload `docs/paper/arxiv.tar.gz` as a single file; arXiv unpacks it. If arXiv asks for the
   top-level file, it is `main.tex`. Let arXiv process it.
5. Check arXiv's PDF:
   - 107 pages;
   - the title-page footnote reads "Appendix J states what the agents did ...";
   - Listing 1 on p. 18 and App. A starting on p. 36, as in the TeX Live 2026 test (if arXiv's
     TeX Live differs, a shift of a page is not an error; see Section 4, item 2);
   - no "??" or "[?]" anywhere (search the PDF for "??").
   If arXiv's log shows a missing file or package, compare it with Section 2 before changing
   anything.
6. Paste the title, authors, abstract and comments (Section 5). Add the cross-lists econ.EM and
   stat.ME, and optionally the MSC class.
7. Preview the abstract page, check the rendering of "Student-t", "h > 1" and "95%", and submit.
   The paper is announced in arXiv's next mailing after its daily deadline.
8. After announcement, the arXiv identifier can go into the repository README. The paper itself
   needs no change for it.
