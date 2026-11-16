# Contributing

This file is for someone who has not read the design documents. It says how to read the
reference codes that appear in the code and in pull requests, and what a pull request has to
show before it is merged.

## Reference codes

Docstrings, tests and pull requests cite the design documents by short codes. Each code is
defined in exactly one table. `P2` means `docs/phase2/DESIGN.md` and `P3` means
`docs/phase3/DESIGN.md`.

| Code | Meaning | Defined in |
| :--- | :--- | :--- |
| `D-1` … `D-35` | Design decision | P2 §0.1 |
| `D-36` … `D-63` | Design decision | P3 §0.3 |
| `E1` … `E28` | Erratum: a review finding and how it was resolved | P2 §0.2 |
| `E29` … `E182` | Erratum | P3 §0.4 |
| `Q1` … `Q14` | Question for the owner, with the answer | P2 §12.2 |
| `Q15` … `Q45` | Question for the owner | P3 §12 |
| `WP1` … `WP25` | Work package | P2 §13 |
| `WP26` … `WP41` | Work package | P3 §13 |
| `A1` … `A7` | Analytic twin or exact oracle | P2 §9.8 |
| `A8` … `A13` | Analytic twin or exact oracle | P3 §9.7 |
| `M1` … `M18` | Mutant: a deliberate bug that some test must catch | P2 §9.7 |
| `M19` … `M36` | Mutant | P3 §9.6 |
| Rows `301`, `302`, … | Geweke and SBC configuration | P3 §9.4, §9.5 |
| Rows `401` … `408` | Block configuration | P3 §9.3 |
| `P1`, `P2`, … (probe) | Probe of the R package's conventions before a parity run | P3 §9.8 |
| `N01` … `N09` | Research note | `docs/phase2/01_*.md` … `09_*.md` |
| `BENCH-P2`, `BENCH-P3` | Performance benchmark | P2 §10, P3 §10 |

`R` followed by a number has more than one meaning, so read it with its context:

- A requirement of `SPEC.md`: `R1` … `R7` (for example R6, no sampler without Geweke and SBC
  tests).
- A risk: P2 §12.1 (`R1` … `R17`) and P3 §11 (`R18` … `R31`).
- A parity run against R: P3 §9.8 (`R1` … `R9`).
- A review round, as in "the review R6" or "R3-M9" (finding M9 of review round 3). Reviews have
  no table of their own; their findings are in the errata tables. `BR` is the batch merge-gate
  review of the Phase 3 pull requests (P3, Amendment 4).

In new text, write which one is meant ("SPEC R6", "risk R22", "parity run R3").

To look a code up:

```bash
grep -n "^| D-60 " docs/phase*/DESIGN.md
```

## Writing docstrings and comments

Say what the code or the test does in plain words first. A code is a pointer for the reader
who wants the history, so it comes after the statement and never replaces it.

- Not enough: `"""E175 (the PR #25 review)."""`
- Enough: `"""A cold start at h = 0 fits when the first h step rejects every block (E175)."""`

A new code family needs a row in the table above in the same pull request.

## Pull requests

Use the template. Keep the description short: a reviewer should be able to read it in two
minutes and find the rest in the diff.

1. **Every claim in the description has a command behind it.** If the description says the
   script refuses a broken index, the description gives the command that shows it, and that
   command was run on the pull request's head. A claim that was not run is listed under
   "Not verified".
2. **A stand-in may only return what the real function can return.** When a test replaces a
   function (`monkeypatch.setattr`), the replacement's values must be ones the real function
   produces. A branch that only a stand-in can reach is dead code. Where the real function
   is cheap to run, add one test that runs it.
3. **A test is never loosened to make it pass.** Changing a tolerance, a seed, a criterion or
   an `xfail` needs a design decision or an erratum, cited in the description.
4. **Keep the branch close to `main`.** CI results on a branch far behind `main` describe old
   code. Merge `main` into the branch, or rebase, before asking for a merge.
5. **Keep the description current.** When the base, the scope or the numbers change, edit the
   description.
6. **Small pull requests are reviewed first.** One that touches `src/` says which tests cover
   the change. One that touches only tests, tools or documents says so in its first line.

## Running the checks

```bash
uv sync --group dev
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

`uv run pytest` runs the suite CI runs on every pull request. The weekly and benchmark tiers
are listed under "Development" in `README.md`.
