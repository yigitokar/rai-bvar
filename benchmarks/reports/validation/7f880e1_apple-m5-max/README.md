# Release validation records: commit 7f880e1, Apple M5 Max

Records of the slow validation tiers run at the release commit `7f880e1` (clean working tree):
- Machine: Apple M5 Max, 18 cores, macOS 26.6; machine, versions and load are in `meta.json`.
- Run: 2026-10-24 04:17–07:18 UTC.
- Markers: `-m "(slow or network) and not bench"`, with `RAI_BVAR_VALIDATION_RECORD` set and a JUnit XML per directory.

| Directory | Result |
| :- | :- |
| tests/geweke | 43 passed (every G-full row, the inexact rows, the block configurations; row 18 included) |
| tests/sbc | 72 passed (23 compositions; row 18 at its per-row thinning cap 400) |
| tests/mutation | 37 passed (every mutant caught by its tier, every control passes) |
| tests/parity | 13 passed, 1 skipped, 1 xfailed (the strict stochvol SV-t row) |
| tests/analytic | 43 passed (analytic twins) |
| tests/mcmc/blocks | 31 passed (block tier) |
| tests/contract | 130 passed (forecast-contract fits) |
| tests/validation | no slow tests (rc 5) |

`record/geweke.jsonl`, `record/sbc.jsonl` and `record/mutation.jsonl` hold one JSON object per test, with the statistics and decisions. In `mutation.jsonl`, `"passed": false` marks a *mutant* run that its tier rejected, which is the intended outcome; each mutant's control run passes.

`tests/unit` is excluded: its slow Phase 1 EB tests hit an intermittent macOS Accelerate heap corruption in a single process (GitHub issue #3). They pass in isolation and feed no table.

The x86-64 BENCH-P2 report from the CI bench job on a GitHub runner is `../../bench_p2_x86_github_779957f.json` (AMD EPYC 7763, 2 vCPU, commit 779957f):
- 228.9 s cold, 198.4 s warm, 17.2 ms/sweep; `converged=False`.
- It is reported, not required to meet the 120 s target on a 2-vCPU runner (DESIGN WP24).
