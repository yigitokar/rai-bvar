# Dossier 05: Numerical robustness lessons and findings about other software

Scope: `docs/paper/NOTES.md` sections A (other software), C (engineering lessons), D (SOC/SUR dummy
rows on the MCMC engine) and E (robustness found by the Clark–Ravazzolo replication). The dossier
also covers findings about other software that the ledger does not list yet (R `BVAR`, the Bank of
England `bvar`, Apple Accelerate) and the macOS platform finding from GitHub issue #3. For each
finding it gives what happened, how it was found, the fix, the evidence and whether it can be
published now.

## 0. Provenance and conventions

**Repository state.** All file citations refer to `/Users/yigitokar/rai-bvar-paper`, branch
`paper-draft`, `HEAD = 62d15f6` (merge of PR #4; `git rev-parse HEAD`). Paths are relative to that
root. "NOTES n" means item n of `docs/paper/NOTES.md`. "DESIGN §x" means
`docs/phase2/DESIGN.md`. "N02" means `docs/phase2/02_sv_sampler.md`.

**Status vocabulary** (from NOTES lines 7–9):

- **verified**: reproducible from a committed test or fixture;
- **recorded**: numbers from a run that is written down (commit message, issue comment, note)
  but not re-run independently;
- **to check**: must be re-run or re-derived before it goes into the paper.

This dossier keeps the NOTES status for every ledger item. For findings that are not in the ledger
it *proposes* a status with the same meaning, marked "(proposed)".

**Commands run for this dossier.** They were run on 2026-10-23, read-only. Settings:
`PYTHONDONTWRITEBYTECODE=1`, `pytest -p no:cacheprovider`, `uv run --frozen --no-sync`. Scratch
scripts were kept outside the repository. Machine: Apple M5 Max, macOS 26.6.2 arm64, Python 3.12.13,
numpy 2.5.4 and scipy 1.18.1 (both on Accelerate, per `numpy.show_config`), jax/jaxlib 0.12.0.1,
numba 0.68.0, scoringrules 0.11.0.

| Id | Command | Result |
| :- | :- | :- |
| R1 | `pytest tests/mcmc/blocks/test_contemp_lstsq.py`, the four RW tiny-σ / Laplacian tests in `tests/mcmc/blocks/test_sv_core.py`, `tests/mcmc/runner/test_failures.py::test_replay_of_an_equation_failure_after_the_production_cccm_step`, `tests/mcmc/unit/test_linalg.py` | 72 passed, 1 skipped (the slow replay), 28 s |
| R2 | `pytest -m slow tests/mcmc/blocks/test_contemp_lstsq.py::test_g_full_reduced_sv_failing_chain_is_fixed` | 1 passed, 5.6 s |
| R3 | Scratch script. At the `test_rw_awol_at_a_tiny_sigma` state (`problem(4, T_eff=88, T_pad=96, rw=True, sigma=5e-9)`), it draws with the pre-fix path (`tridiag_precision_sample` on the assembled `SV.awol_system`), the fixed path (`laplacian_precision_sample` on `SV.awol_rw_laplacian`) and the test's 60-digit reference | see §1.3 |
| R4 | Scratch script. It runs 20 seeds of the synthetic extreme-scale state of `test_extreme_scale_state`, comparing `precision_draw` (formed Gram, Cholesky) and `lstsq_draw` (QR) against a 60-digit exact conditional | see §1.2 |
| R5 | Five runs of `pytest -m slow "tests/unit/test_eb_convergence.py::test_eb_converges_on_a_simulated_n50_p4_panel[0]"` | 5 of 5 ended abnormally, see §4.7 |
| R6 | Read-only computations on committed fixtures: `tests/parity/fixtures/r_bvar_logml.json`, `r_bvar_eb.json`, `r_bvar_mh.json` | see §4.4 |

R1 and R2 re-run committed tests, so they confirm **verified** items. R3–R6 are this dossier's own
computations. They are not committed tests, so their results are **recorded**.

**Fact-check re-runs (2026-10-23, same machine and settings, `62d15f6`).** R1: 72 passed, 1 skipped,
29.1 s. R2: 1 passed, 6.5 s. R3: reproduced to the printed digits (old min pivot 2.22e-16, exact and
new 5.308e-15; logdet old −36.0437, new = reference −32.8695; `h₀` old −73.9032, reference −3.024114;
max |h_old − h_ref| 70.88, new 4.31e-14). R4: reproduced (Cholesky `ok = False` in 15 of 20, `ok =
True` with error > 1 sd in 4 of 20, seed 7 pivot 4.44e-16 and error 3.14 sd). R5: three further runs
of the same command all ended with rc 139 (SIGSEGV) during garbage collection inside pytest's
assertion rewriting after the test passed; the SIGBUS/SIGTRAP variety of the original R5 table was
not seen again. R6: every number in §5.4 recomputed from the fixtures. (added by fact-check)

**Issue #3** (`gh issue view 3 --repo yigitokar/rai-bvar --comments`) is the overnight work-queue
thread of 2026-10-23. Its comments are not committed files, and numbers taken only from them are
**recorded**.

---

## 1. At a glance

| § | Finding | NOTES | Ledger status | After this dossier | Publishable now? |
| :- | :- | :- | :- | :- | :- |
| 2.2 | Geweke chain at `h ≈ 31–37`: formed Gram `ZᵀWZ` loses the small eigenvalue (pivot 0). Fixed by a QR draw on the whitened rows | 10 | verified (fix) | fix verified (R1, R2); diagnostic numbers recorded | **Yes**, as an engineering/validation lesson, not as a new algorithm |
| 2.2 | *New:* the pre-fix Cholesky path can return `ok = True` with a draw > 1 sd off (4 of 20 synthetic states, including the committed test's own seed 7) | — | — | recorded (R4; re-run by fact-check) | Not yet. Needs a committed test and a decision on pivot floors (§7) |
| 2.3 | RW SV AWOL at `σ = 4.98e-9`: `σ²D` falls below the ulp of 2, so the level pivot cancels. Fixed by the excess-pivot recursion | 17 | verified (M5 Max) | verified (R1). Old-path behaviour reproduced (R3) | **Yes**, with prior-art positioning (subtraction-free elimination, [CITE?]) and a Linux re-run |
| 2.3 | RW `aux_log_likelihood` used the assembled system: 17534.8 instead of −1650.8 at `σ = 5e-9` | (17) | — | fix verified (committed test, R1); numbers recorded (commit `4fc0162`) | Yes, as part of 2.3 |
| 2.4 | Failure replay crashed (`TracerArrayConversionError`), so the first crash's diagnostics were lost | — | — | verified (committed test, R1) | Yes, as a short engineering note |
| 2.5 | Design-stage numeric errata E7, E14, E23 | 11 | verified | — | Yes (errata table as supplement) |
| 3 | Bitwise reproducibility limits | 9 | verified in CI (Linux) | — | Yes. Re-run on the release commit |
| 4.1 | Mode II λ step must count the dummy rows | 12 | recorded | — | After the injection is made a test (§7) |
| 4.2 | Closed-form λ marginal; exact rejection envelope under `iw_equivalent` | 13 | verified | — | **Yes** |
| 4.3 | Dense-prior eigen fast path survives dummy rows | 14 | verified (timing recorded) | — | Yes. Re-time on an idle machine |
| 4.4 | Wrong-weight dummy bugs are loud (M17, M18) | 15 | verified (two platforms) | — | **Yes** |
| 4.5 | Feature added bitwise-neutrally | 16 | recorded (one run) | — | Yes as a recorded statement. The structural test is verified |
| 5.1 | stochvol: NC `|σ|` without flipping `h̃` (D-7) | 1 | recorded (3.2.1) | mechanism verified on *our* side (M6); stochvol numbers recorded | **No.** Re-check current stochvol and contact the maintainers first |
| 5.2 | stochvol 3.2.1 SV-t misses the standardized-t posterior | 2 | recorded (3.2.1 only) | — | **No** (unblock path not run; report upstream first). Yes as a methods point about oracle deviations |
| 5.3 | bayesianVARs 0.1.8: blocked, probes P1–P5 unresolved | — | — | — | No finding to publish |
| 5.4 | R `BVAR` 1.0.5: dummy-only ML term, degenerate EB, `predict()` noise, ψ units, `ȳ₀` | not in ledger | — | fixture-verified numbers (R6); behaviour recorded for 1.0.5 | **No** until current-version check and maintainer contact. BVAR is a JSS package |
| 5.5 | BoE `bvar` 0.3.5: `ȳ₀` convention, forward-difference gradient noise, `S_post` cancellation, docs vs code | not in ledger | — | recorded (pinned by `test_boe_*`) | Not yet (same caveat) |
| 5.6 | Accelerate `slogdet`/`det`: spurious divide-by-zero on SPD tridiagonals | not in ledger | — | recorded (commit `dd8244e`) | Only as a platform caveat |
| 5.7 | Intermittent heap corruption in a Phase 1 test on macOS (issue #3) | not in ledger | — | reproduced 5/5 (R5) and 3/3 (fact-check); root cause unknown | Only as a platform caveat / reason for Linux CI |

---

## 2. Float64 failures found by the validation and replication runs

### 2.1 The common pattern

Three failures surfaced within about 24 hours (2026-10-22/23). All three occurred at states that
are legitimate under the model. None was a user report: two came from validation or replication
runs, and the third was a bug in the failure-reporting path.

1. A Geweke chain at the extreme tail of the test prior made a formed Gram singular (§2.2).
2. A Clark–Ravazzolo fit at `σ ≈ 5e-9` made a scaled tridiagonal system singular (§2.3).
3. The deterministic replay of the second failure itself crashed (§2.4).

Two design rules shaped how they were found and fixed:

- **Failures are values, never jittered** (D-18, DESIGN line 64; §5.7, lines 1367–1375). A failing
  block sets `(code, it)` in the chain carry with `code = 1 + 1000·block + eq`. The host replays
  the chunk deterministically and raises `MCMCNumericalError` with diagnostics. "No jitter is ever
  added and no chain is dropped" (DESIGN line 1375). The status codes below decode accordingly:
  - `1003` = block `EQ_CONTEMP = 1`, equation 2;
  - `2001` = block `EQ_VOL = 2`, equation 0.

  The enum values are in DESIGN lines 1339–1340.
- **Fix the kernel, not the test.** In both cases the alternative (narrow the test prior, or change
  priors or criteria so the run passes) was explicitly rejected:
  - commit `3d3f51b`;
  - issue #3, item 6 and the item 7 orchestration comment ("Don't change priors, settings or
    criteria to make anything pass").

### 2.2 F1: the `h₀ ≈ 30` Geweke chain and the QR whitened `a`-row draw (NOTES 10)

**What happened.**
- Run: the nightly `test_g_full[reduced_sv]` (G-full: `C = 4000` chains, `M_c = 500`, `M₁ = 10⁶`;
  `tests/geweke/test_geweke_compositions.py` line 86).
- Failure: one successive-conditional chain of 4000 failed at sweep 194 with status 1003,
  `EQ_CONTEMP`, equation 2, Cholesky pivot 0.
- State: the chain had reached the test prior's extreme tail, with equation 0 at `h ≈ 31–37`
  (`h₀ = 35.4`), so `|y| ≈ 2e8`. Sources: the `3d3f51b` commit message and
  `tests/mcmc/blocks/test_contemp_lstsq.py` lines 4–6 (`h ≈ 31–37`, `|y| ≈ 2e8`); `h₀ = 35.4`
  (with `h ∈ [30.9, 36.6]` and max h over the chain 37.4) is only in issue #3, item 6. The Geweke
  test-module docstring (lines 49–57) gives `h₀ ≈ 30–36` and `|y| ≈ 10⁸` instead (corrected by
  fact-check).
- The docstring describes the state as `h₀ ≈ 30–36, φ ≈ 0.999, σ ≈ 0.9` with
  `P(max h > 25) ≈ 1e-6` per prior draw.
- Issue #3, item 6 gives `P(max|h| > 30) ≈ 5e-7` per draw at `T = 50` from a 2e6-draw Monte Carlo;
  the commit message gives the 5e-7 and the 2e6 draws but not `T` (corrected by fact-check).

Both probabilities are recorded, and they use different thresholds. NOTES 10 and the Geweke test
docstring (line 51) give `|y| ≈ 1e8`; the `3d3f51b` commit, issue #3 item 6 and
`test_contemp_lstsq.py` (line 4) say `2e8` (corrected by fact-check). The paper should use `2e8`.

**How it was found.** The Geweke tier itself found it: one non-finite chain makes the test fail.
The weekly tier on the pinned commit `408a7f4` hit it again ("38 passed, 1 failed"; issue #3, item 2
partial and item 2 complete). Under D-18 the engine reported the failure and did not perturb it
(docstring, line 53).

**Diagnosis (recorded: commit `3d3f51b`, issue #3 item 6, NOTES 10).**
- The `a`-row precision of equation 2 is `K_a = diag(prec) + ZᵀWZ` (DESIGN §3.7.1, lines 824–826).
- At the pre-failure state its exact eigenvalues (60-digit decimal) are 14.8 and 1.9e17, so
  `cond = 1.3e16`.
- Forming `ZᵀWZ` in float64 rounds the small eigenvalue away, so the Jacobi-scaled Cholesky pivot is
  exactly 0.
- The conditional itself is benign: mean (−0.173, −0.663), sd (0.20, 0.16).

**Fix (verified).** `Recursive.contemp_core` now draws `a_i` with `linalg.lstsq_draw`
(`src/rai_bvar/engines/mcmc/linalg.py` lines 200–220; DESIGN lines 828–831):
- Householder QR of the whitened rows `A = [diag(√prec); √w ⊙ Z]` with `diag(R) > 0`, so `R` is the
  upper Cholesky factor of `K = AᵀA`;
- the draw is `θ = R⁻¹(Qᵀrhs + z)`.

This is the same function of the injected noise `z` as `precision_draw(K, b, z)` up to rounding.
The condition number of `A` is `√cond(K)` and is never squared. `min_pivot` keeps its meaning,
`(R_jj/‖A_·j‖)²` (linalg.py line 209). An overridden `contemp_conditional` (the seam of mutant M14)
still draws through `precision_draw` (DESIGN line 830; `test_overridden_conditional_is_still_drawn_from`).

**Alternative rejected (recorded, commit `3d3f51b`).** Narrowing the test prior's `h` tail with
`(φ+1)/2 ~ Beta(10, 3)` would take `P(max|h| > 30)` from 5e-7 to 0 in 2e6 draws. It "would have
hidden a real float64 weakness and dropped high-persistence coverage". Issue #3 (item 6 table), not
the commit message, also notes that it would change DESIGN §9.2's test prior (corrected by
fact-check).

**Evidence.**

| Claim | Source | Status |
| :- | :- | :- |
| QR draw equals the Cholesky draw within the DESIGN §9.3 path-vs-path envelope `2e-12 + 5εκ(1 + t)` (sd units) on 20 random states of each of 5 reduced compositions, with the same `ok` and pivots (rtol 1e-8); worst < 1e-10 | `test_contemp_lstsq.py::test_lstsq_draw_equals_the_cholesky_draw` (lines 60–86) | verified (R1) |
| At a synthetic extreme-scale state (regressor scale 1.5e8 next to a near-copy, `cond > 1e15`), the QR draw matches the 60-digit exact conditional to `50ε√cond`, and its noise map has covariance `K⁻¹` | `::test_extreme_scale_state` (lines 89–132) | verified (R1) |
| Replay of chain 3214 (stage 0, block 3000–3499, the harness keys): pre-state `max h > 30`; the old Cholesky draw gives status 1003; the production sweep from the same state succeeds and every draw is finite | `::test_g_full_reduced_sv_failing_chain_is_fixed` (slow; lines 155–196) | verified (R2: 1 passed, 5.6 s) |
| At the real failing state, QR matches the exact conditional to 1.6e-8 sd (mean) and 1.8e-10 (sd) | commit `3d3f51b`; issue #3 item 6; DESIGN lines 645–649 | recorded (no committed test asserts these numbers) |
| Agreement after 3 Geweke sweeps ≤ 3.7e-12 relative, after 60-sweep fits ≤ 4.1e-10 ("rounding compounding along the chain"); every structural program bitwise unchanged | commit `3d3f51b` | recorded |
| After the fix, `test_g_full[reduced_sv]`: q 1.0, max \|z\| 2.67, 0 non-finite; fast suite 4483 passed; M1 and M14 still caught | commit `3d3f51b`; issue #3 item 6 | recorded |
| Slow Geweke tier on `4bb978a`: 55 passed, including all 22 G-full compositions | issue #3, orchestrator summary | recorded |

**New observation from this dossier: the pre-fix path could also be *silently* wrong (recorded,
R4).** The `ok` rule of `precision_factor` is "every `L_jj` finite and positive" (linalg.py lines
172–179). `test_extreme_scale_state`'s own docstring notes that at such states the pivot "is then 0
or a few ε, depending on the rounding" (lines 90–93). R4 quantifies this on 20 seeds of that
synthetic construction:

- the Cholesky path reported failure (pivot 0) in 15 of 20;
- it returned `ok = True` with a draw more than 1 posterior sd from the exact mean in 4 of 20
  (seed 7: pivot 4.4e-16, error 3.14 sd);
- the QR path was within 2.3e-9 to 6.7e-8 sd on all 20 seeds (corrected by fact-check; the
  fact-check re-run measured the zero-noise draw, i.e. the conditional mean).

Seed 7 is the committed test's own state (`np.random.default_rng(7)`, line 94). There the
Cholesky path does not fail: it returns `ok = True` with pivot 4.4e-16 and a mean 3.14 sd off. The
module docstring's "at an extreme-scale state the Cholesky path fails" (line 12) is therefore not
what happens at the test's own seed, and the test never runs the Cholesky path (`del K, b`, line
105). (added by fact-check)

Caveats:
- This is synthetic. The production failure that was actually observed was loud (pivot exactly 0;
  R2 confirms status 1003).
- No committed test exercises the silent case, and no pivot-floor warning gates `ok`: `min_pivot`
  is only reported in `fit.diagnostics`. DESIGN §8.4 (line 1691) lists "smallest pivot" in the
  per-equation table, while the code reports it per chain, "not per equation"
  (`src/rai_bvar/engines/mcmc/monitor.py` line 78) (corrected by fact-check).
- Other steps still factor a formed Gram with the same `ok` rule: the coefficient step through
  `precision_draw` (`src/rai_bvar/components/mean/linear.py` lines 386, 418) and the CCCM `Π` step
  through `precision_factor` (`src/rai_bvar/components/mean/linear_reduced.py` line 208) (corrected
  by fact-check). Whether they are exposed at reachable states is not established (open question
  Q1).

**Publishable now?** Yes, as an engineering and validation lesson:
- a Geweke tier with a heavy-tailed test prior is a float64 stress test;
- the fix belongs in the kernel;
- the QR draw is the same noise map without squaring the condition number.

It is not a new algorithm. The normal-equations-vs-QR conditioning argument is textbook (Higham;
Björck; [CITE?]), and QR-based Gaussian conditional draws are standard in Bayesian regression
[CITE?]. Report the diagnostic numbers as "recorded" unless a test asserting them is committed.
Re-run R2 on Linux before submission (the evidence so far is from the M5 Max only).

### 2.3 F2: the RW SV AWOL Laplacian cancellation at `σ ≈ 5e-9` (NOTES 17)

**What happened.**
- Run: the first full Clark–Ravazzolo run on `8a622c5` aborted in the ordering stage, after the
  main stage and the Constant orderings.
- Error: `MCMCNumericalError` in `EQ_VOL`, equation 0, chain 0, burn-in sweep 318, code 2001,
  pivot 0.
- Fit: `SV(rw)`, reduced form, origin 1987Q2, ordering GDP, TBILL, INFL, UR, backtest seed
  1954198910. Sources: issue #3, "Item 7: the full run crashed"; commit `f60d736`.

**How it was found.** By the replication run, an acceptance exit criterion that requires all 24
orderings (issue #3, item 7). The first attempt to diagnose it failed because the failure replay
itself crashed (§2.4). The state was then dumped through a separate deterministic replay (issue #3,
item 7 crash comment).

**Diagnosis (recorded: commit `f60d736`, issue #3, NOTES 17, DESIGN lines 936–945).**
- GDP's volatility scale was `σ = 4.98e-9`.
- Under the non-centred prior `σ ~ ±N(0, B_σ)` (equivalently `σ² ~ B_σχ²₁`; N02 §7 line 304) the
  prior density is positive at 0, so the state is valid. The non-centred move of the same sweep
  would have taken `σ` back to 0.031.
- The RW AWOL system is solved in scaled form `Q̃ = σ²Q`, `h = σx̃`. There the observation terms
  `σ²D ≈ 1e-17` fall below the ulp of the Laplacian diagonal (2). The assembled matrix is therefore
  exactly the singular path Laplacian `[1, 2, …, 2, 1]`.
- The level direction's exact pivot `≈ σ²(1/V₀ + ΣD)` (order 1e-15) comes out 0.
- DESIGN gives the threshold as `σ ≲ 1e-8` (line 941).

**Fix (verified).** For `rw=True`, step (a) factors `Q̃ = Lap_w + diag(δ)`
(`SV.awol_rw_laplacian` → `linalg.laplacian_precision_sample`, linalg.py lines 425–480). Each pivot
is carried as its excess over the next edge:

    q_0 = δ_0,   q_t = δ_t + w_{t−1} q_{t−1} / (w_{t−1} + q_{t−1}),   p_t = q_t + w_t

On a unit edge this is `q_t = δ_t + q_{t−1}/(1 + q_{t−1})`, and after a padded edge `q_t = δ_t`.
Every term is non-negative, so nothing cancels. `log|Q|` uses `log w + log1p(q/w)` on edges
(linalg.py lines 434–442). The pivots are those of `tridiag_precision_sample` on the assembled
matrix in exact arithmetic, so the sampler is the same function of `z`. The AR path is unchanged
(DESIGN line 945); that the initialisation is untouched is stated in the `f60d736` commit message,
not in DESIGN (corrected by fact-check). The docstring calls the term `w q/(w + q)` "a series conductance" (linalg.py line 436). In this
dossier's reading (not a repository statement), the recursion is therefore a Schur complement (Kron
reduction) of a grounded path graph.

**Follow-ups from the independent review (commit `4fc0162`, recorded unless a test is named).** The
review found no defect. Its own checks:
- 150 random systems against exact rationals: `x` 7.6e-16, logdet 4.1e-15, pivot 2.9e-16;
- the CR state's whole SV step was ok 20/20 for `σ` from 5e-9 to 1e-16.

It also found that the RW `aux_log_likelihood` still used the assembled system: **17534.8 instead
of −1650.8 at `σ = 5e-9`**. It now uses the same `(δ, w)`, and a new test compares it with the dense
MVN under the closed-form RW covariance `V₀ + σ² min(s, t)`
(`test_rw_integrated_likelihood_at_a_tiny_sigma`). That it passes now is verified (R1); that it
failed before the fix is recorded only ("fails before", commit `4fc0162`) (corrected by
fact-check). Two further follow-ups:
- the tiny-σ test now checks that `δ` carries `σ²D` itself (a `δ` rebuilt as `(σ²D + 2) − 2` used to
  pass);
- the masked branches use safe denominators, so `jax.debug_nans` does not trip (asserted in
  `test_awol_rw_laplacian_is_the_rw_system`).

**Evidence.**

| Claim | Source | Status |
| :- | :- | :- |
| `Lap_w + diag(δ)` and `b̃` are the RW system; same `x` as the assembled path to 1e-13, logdet 1e-13, pivot 1e-12 on 3 random problems; padded states return `z`; no NaN under `debug_nans` | `test_sv_core.py::test_awol_rw_laplacian_is_the_rw_system` (lines 240–262) | verified (R1) |
| At `σ = 5e-9`, `T_eff = 88`: `δ_0 = σ²/V₀` and `δ_t = σ²D_t` to 1e-15; the level pivot (exact) < 1e-14; pivot rel 1e-12, logdet 1e-13, `h = σx` rel 1e-12 against the 60-digit reference; `ptp(h) < 1e-6`; increments `σΔx` to 1e-6 of their rms | `::test_rw_awol_at_a_tiny_sigma` (lines 292–319) | verified (R1) |
| `aux_log_likelihood` (RW) at `σ = 5e-9` equals the dense MVN to 1e-8 | `::test_rw_integrated_likelihood_at_a_tiny_sigma` (lines 322–338) | verified (R1) |
| The whole RW update at `σ = 5e-9` with flat `h`: `ok`, `min_pivot > 0`, all finite (5 noise draws) | `::test_rw_step_at_a_tiny_sigma_is_ok` (lines 341–352) | verified (R1) |
| Old path at the tiny-σ test state: pivot 2.2e-16 vs exact 5.3e-15; level of `h` off by 70 | commit `f60d736`; issue #3 | recorded. **Reproduced by R3:** old min pivot 2.220e-16, exact 5.308e-15, new 5.308e-15; logdet old −36.04, new −32.8695 (= reference); `h₀` old −73.90 vs reference −3.024; max \|h_old − h_ref\| = 70.88, new 4.3e-14 |
| Invariance: 639 of 656 arrays bitwise identical (every non-RW composition and fit); RW 3-sweep states ≤ 3.3e-11 relative, 60-sweep fits ≤ 6.7e-10; the 656-array set is bitwise unchanged by `4fc0162` | commits `f60d736`, `4fc0162` | recorded (one machine) |
| The crashing fit completes and converges; the rerun of the replication (1.10 h, all 24 orderings) had no crash | issue #3; `benchmarks/reports/clark_ravazzolo.md` line 3 | recorded (the report will be replaced by the acceptance run in progress) |

**A second silent mode in the old path (recorded, R3).**
- The SV step's `ok` rule is `isfinite(piv) & (piv > 0)` (`src/rai_bvar/components/vol/sv.py` line
  855).
- At the R3 state the pre-fix assembled path returns a *positive* rounding-noise pivot (2.2e-16).
  It would have passed that check, with `h₀ = −73.9` instead of `−3.02`.
- The production crash was the loud variant (pivot exactly 0).
- Pre-fix runs of SV(rw) near `σ → 0` could therefore have produced wrong `h` levels without any
  failure (an inference from the isolated AWOL step; see §9). This affects only reports produced before `f60d736`; the committed full report was
  produced on `4bb978a`, which contains the fix (issue #3, item 7 results). Whether the committed
  quick report (`benchmarks/reports/clark_ravazzolo_quick.md`, committed in `1ae5765` at 2026-10-23
  01:25 UTC, before `f60d736` at 04:09 UTC) was affected is not known (Q2).

**What the fix does not do.** "The fix removes the crash, not the slow mixing" (NOTES 17; issue #3).
The `σ → 0` region is where the centred `σ²` step mixes slowly and only the NC move escapes. The
committed CR report gives the following share of origins converged in the main stage
(`clark_ravazzolo.md` lines 34, 62, 90, 118):

| Arm | Converged |
| :- | -: |
| SV(rw) | 68% |
| SV | 37% |
| SV(rw,t10) | 57% |
| AR-SV(0.9) | 62% |

The misses are typically one SV `σ` with R-hat 1.012–1.020 (issue #3, item 7 results; the
report's lines 388–398 show one example message per group, with R-hat 1.012–1.020). The same
report warns of non-centred fold fractions above 1% in 12–30 origin-messages per SV arm (lines 385,
390, 393, 399). The report (and its JSON) keeps only one example message per group, so the
maximum over origins is not reported: 0.0676 (INFL, SV(rw,t10), line 393) is the largest of the
four shown examples, which range from 0.0137 to 0.0676 (corrected by fact-check). Real macro
equations therefore do enter the near-homoskedastic regime. These numbers are to be superseded by
the acceptance run in progress.

**Publishable now?** Yes, as a robustness case study. It has three parts:
- a valid state under a standard shrinkage prior;
- a scaled system whose assembly destroys exactly the information that matters;
- a subtraction-free factorisation that keeps relative accuracy.

**Positioning caution.** The recursion is an instance of known subtraction-free Gaussian elimination
for diagonally dominant M-matrices parametrised by off-diagonals and diagonal excess. Candidates are
the GTH algorithm of Grassmann, Taksar & Heyman (1985) and the accurate-LDLᵀ / smallest-eigenvalue
results of Alfa, Xue & Ye (2002) and Ye (2008) [CITE?]. The paper should cite these and claim the
application, not the algorithm. Before submission:
- re-run the tiny-σ tests on Linux (NOTES 17 is "verified (Apple M5 Max)");
- commit a test that asserts the old-path failure at the R3 state (today the commit message and R3
  are the only evidence that the old path fails there);
- keep the convergence numbers as context for the acceptance-run dossier.

### 2.4 F3: the failure replay itself failed (commit `93f2bcd`)

**What happened.** When the CR run hit the F2 failure, the D-18 replay (`runner.failure_detail`)
re-ran the sweep's earlier steps eagerly from the host (NumPy) copy of the state. The production
CCCM step indexes `Π` with its traced column inside a `fori_loop`. With host arrays this raised
`TracerArrayConversionError`, so the failure carried `replay_error` instead of `detail`, and the
diagnostics of the first crash were lost (commit `93f2bcd` message; issue #3, item 7 crash comment).

**Fix (verified).** `failure_detail` converts the replayed state and the fit inputs to device arrays
first (`src/rai_bvar/engines/mcmc/runner.py` docstring and the `jax.tree.map(jnp.asarray, …)` line
added in `93f2bcd`). New test:
`tests/mcmc/runner/test_failures.py::test_replay_of_an_equation_failure_after_the_production_cccm_step`
("it fails before the fix and passes after", commit message; passes in R1).

**Related robustness change in the same commit (recorded).** The `iw_equivalent` rejection sampler
of the dummy reference now gives up after 10,000 batches with NaN, so that "bad inputs never hang a
vmap batch" (commit `93f2bcd`; `src/rai_bvar/validation/dummies.py`). Under `vmap` an unbounded
rejection loop stalls every lane.

**Publishable now?** Yes, as a short engineering note. The error path is the least-exercised path,
and it needs a test that runs the production program, not toy kernels. The existing replay tests
(`test_failures.py` lines 1–40) use toy kernels.

### 2.5 Design-stage numeric errata (NOTES 11)

DESIGN §0.2 (lines 81–115) lists 28 judge findings (E1–E28) with resolutions. Those about numerics
or reproducibility:

- **E7:** `it0 + j·thin + r` promotes to int64 under x64. All counters are now uint32 with explicit
  casts and a host bound check (DESIGN §5.6, line 1345).
- **E14:** `Runtime` knobs (`gram`, `pad_T`) were not bitwise-neutral. `gram` is now resolved by a
  fixed shape rule and stored in the spec, and `Runtime` holds only knobs with a bitwise-invariance
  test.
- **E23:** `2 log|resid|` is `−inf` on padded rows, which gives NaN. The fix masks before the log
  (§3.7.4).

Status: verified (in repository). Publishable as a supplementary table showing that independent
design review caught defects before implementation.

### 2.6 Lessons for the paper (each with its evidence)

1. **Validation tiers are float64 stress tests.** F1 came from Geweke with a deliberately
   heavy-tailed test prior, and F2 from a replication with 24 orderings × 106 origins. Neither would
   be hit by unit tests at ordinary states.
2. **"Legitimate but extreme" states are part of the model.** The `B_σχ²₁` prior has positive
   density at `σ = 0` (DESIGN line 941). The test prior's `P(max|h| > 30) ≈ 5e-7` is small but
   reached among 4000 × 500 sweeps.
3. **Assembly can be the lossy step.** In both F1 and F2 the factorisation was correct for the
   matrix it received. The information was lost when the matrix was formed (`ZᵀWZ`; `σ²D + 2`).
   The fixes factor the data, never the assembled matrix.
4. **A positive pivot is not proof of a correct draw.** R3 and R4 show positive rounding-noise
   pivots with wrong draws (recorded).
5. **No jitter, no dropped chains.** This rule made F1 and F2 visible. (A counterfactual about
   jittering was moved to "Unsupported claims" by the fact-check.)
6. **Test the failure path with the production program** (F3).

---

## 3. Reproducibility limits (NOTES 9)

**Claim.** The guarantees are set out in DESIGN §5.6, lines 1353–1363:

- **Bitwise** across `runtime.chunk`, burn piece size, fused vs unfused plan execution. DESIGN does
  not list `chains="map"` among these guarantees; it says the prefix property "is still tested
  bitwise with `chains="map"` and with toy kernels" (line 1359), and NOTES 9 lists `chains='map'` as
  bitwise (corrected by fact-check).
- **To rounding only** across chain-count prefixes and `vmap` vs `shard_map`. XLA's
  batch-size-dependent vectorisation changes the last bits, and "over long runs a last-bit
  difference can flip an accept/reject decision, after which the chains are only equal in
  distribution".
- **Not across** JAX versions, platforms, PRNG implementations or `jax_threefry_partitionable`.

**Tests.**
- `tests/mcmc/runner/test_chunking.py`: `test_chain_prefix_invariance` is "bitwise on macOS, to
  rounding on x86 Linux (another vmap batch size)" (lines 83–91); the `chains="map"` prefix is
  bitwise (lines 93–101).
- `tests/mcmc/runner/test_devices_bitwise.py` (docstring lines 1–6): `shard_map` is bitwise with
  `chains="map"`, and to rounding with `vmap` (rtol 1e-12 for toys, 1e-9 for the production SV
  kernels on short runs).

**Supporting numbers.** F1 and F2 give magnitudes for "to rounding" at the kernel level: ≤ 3.7e-12
and ≤ 3.3e-11 relative after 3 sweeps, ≤ 4.1e-10 and ≤ 6.7e-10 after 60-sweep fits (§2.2, §2.3;
recorded). Caveat: these compare the old and the new (algebraically equivalent) kernel on one
machine, not one kernel across `vmap` batch sizes or platforms; the analogy "as with vmap batch
sizes" is issue #3's (item 6) (added by fact-check).

**Status.** Verified in CI (Linux) per NOTES. **Publishable:** yes. The macOS-vs-Linux asymmetry of
the prefix test is a finding worth one sentence. Re-run on the release commit, on both platforms.

---

## 4. SOC/SUR dummy rows on the MCMC engine (NOTES 12–16; DESIGN lines 618–640)

### 4.1 Mode II λ step must count the dummy rows (NOTES 12)

**What.** In conjugate (Mode II) structural form the rows are weighted `1/λ_i` like data rows, so
`λ_i | ·` gains `m/2` in its shape and the rows' λ-free squared residuals in its scale:
`IG(c + (T + q_i + m)/2, …)`. DESIGN line 629 says that Mode II's coupling adds `m` to `n_active`
and `½w_iΣ_d(y_{d,i} − F_dθ_i)²` to `quad`.

**How found / evidence.** Omitting the count is easy, and the coefficient step alone still looks
right. The dense oracles of the coefficient step do not catch it. The MCMC-vs-Phase-1 twin catches it
at once: 33 of 165 quantiles outside 4 MCSE at `n = 3, p = 2, T = 60`.
- The twin test is committed: `tests/analytic/test_mcmc_twin.py::test_a1_dummies_mode2_iw_matches_the_phase1_posterior_with_dummies`
  (lines 608–614), with a power control `test_a1_dummies_comparison_sees_the_rows` (lines 617–624)
  and nightly variants (lines 627–644).
- The omission itself was injected by hand and is not a registered mutant, "because the coupling is
  not an overridable internal" (NOTES 12).

**Status.** Recorded. **Publishable** after the injection is reproducible. Either expose the
coupling as an override seam and add an M19, or commit a scratch test (Q4). The *twin-test design*
(MCMC posterior vs the analytic NIW-with-dummies posterior) is publishable now.

### 4.2 Closed-form λ marginal and an exact rejection envelope (NOTES 13)

**What.** Under the dummy-augmented Mode II prior,
`λ_i | d ~ IG(c + m/2, C + ½ r'(F_d P̃⁻¹F_dᵀ + I/w)⁻¹r)` with `r = y_d − F_dμ₀`. The Geweke
reference can therefore sample the prior exactly: integrate θ out, draw λ, then condition θ on the
rows by Matheron's rule (`src/rai_bvar/validation/dummies.py` docstring, lines 1–43).

Under `iw_equivalent`, `p(d | λ) = N(r; 0, S₀ + λUUᵀ)` is not inverse-gamma. The base `IG(c, C)` is
nevertheless an exact rejection envelope. Because `S(λ) ⪰ S₀` grows with λ, `|S(λ)| ≥ |S₀|` and
`q(λ) ≥ q∞`, where `q∞` is the GLS residual. Hence

    g(λ) = (|S₀|/|S(λ)|)^{½} exp(−½(q(λ) − q∞)) ≤ 1

The acceptance rate is 0.6–1.0 on the test rows (NOTES 13; issue #3 item 3a). The log prior is
exact up to the intractable evidence. The reference is written in covariance form (`m × m`) and
the kernels in precision form (`q × q`), which keeps the two implementations algebraically distinct
(D-22; dummies.py lines 1–4).

**Evidence.**
- `tests/validation/test_reference_dummies.py`: log prior vs closed forms to 1e-10; prior-sampler
  moments `|z| ≤ 5`; KS of the λ draws against the quadrature CDF (40,000 draws, `p > 1e-4`; issue
  #3 item 3a).
- Geweke row 205: G-small max |z| 2.24 (t_crit 4.37); G-full 2.76.
- SBC row 205: min p 0.018.

**Status.** Verified. **Publishable now:** yes. It is a small methods contribution: an exact prior
sampler for a non-conjugate dummy-augmented prior, needed so that Geweke and SBC remain exact.
Matheron's rule and pathwise conditioning [CITE?].

### 4.3 The eigen fast path survives dense prior precision (NOTES 14)

**What.** Dummy rows break the diagonal prior precision. With `P = L_P L_Pᵀ` dense,
`K = L_P(I + L_P⁻¹GL_P⁻ᵀ/λ)L_Pᵀ`. One per-fit eigendecomposition again gives every λ in `O(q²)`, and
storing `L_P⁻ᵀQ` keeps the per-sweep draw code unchanged (DESIGN lines 632–637). Mode II keeps its
`fixed_factor` path.

**Evidence.**
- `tests/mcmc/blocks/test_dummies.py::test_dense_eigen_path_with_dummies_equals_cholesky_const_test_prior`
  (lines 798–808) and `…_minnesota[n]` (n = 3, 6; lines 811–827): within the §9.3 path-vs-path
  envelope.
- G-small over 7 seeds and G-full on the eigen path (issue #3 item 3b). At the PR seed, stage 1 of
  G-small gave q = 0.0012. Stage 2 at 4× size was clean (max |z| 1.94). Six further seeds gave
  p_bonf 0.20–1. This is "a chance excursion that the sequential test handled as designed" (issue
  #3).
- Timing: 0.46 vs 1.53 ms/sweep (Mode I × Constant, `n = 20, p = 4, T = 300`, min of 3 under a
  concurrent load of 3–5, Apple M5 Max).

**Status.** Verified (agreement). The timing is recorded under load. **Publishable:** yes. Re-time
on an idle machine for the paper.

### 4.4 Wrong-weight dummy bugs are loud (NOTES 15)

| Mutant | Bug | U (mutant / control) | G-small max \|z\| (mutant / control) |
| :- | :- | :- | :- |
| M17 | Mode I rows weighted by the current `1/λ_i` instead of the fixed `1/σ̄_i²` | 0.0602 / 2.25e-15 | 23.15 / 2.44 |
| M18 | rows dropped from the CCCM `Π` step | 0.405 / 2.35e-14 | 99.7 / 2.61 |

Sources: `tests/mutation/mutants.py` lines 827–833 and the `EVIDENCE` entries at lines 854–857.
The table values are from the comment block, which issue #3 (item 3c) identifies as the x86-64
reproduction's numbers. The `EVIDENCE` entries differ beyond rounding in the controls, so they come
from other runs: M17 U 0.060 / 2.4e-15 and G 23.1 / 1.94 (against the dense eigen-path control;
2.44 on `cholesky_const`), M18 U 0.405 / 2.9e-14 (comment block: 2.35e-14) (corrected by
fact-check). Each G-small run takes 3–6 s.

M17–M18 were recorded on the Apple M5 Max and reproduced on x86-64 Linux the same day ("4 passed").
The independent x86 review also caught four further hand-injected bugs (NOTES 15):
- Mode II rows weighted `1/σ̄²`;
- rows counted twice in the λ step;
- CCCM rows weighted by `(Σ⁻¹)_jj`;
- fixed-factor weight `2w`.

**Status.** Verified on two platforms. **Publishable now:** yes, and it belongs in the mutant ×
tier table (NOTES 8; another dossier).

### 4.5 Adding the feature left everything else bitwise unchanged (NOTES 16)

**What.** With `MeanStatic.n_dummy = 0`, no dummy code is traced. Before and after the change, 537
arrays were bitwise identical on one machine: prior draws, simulated data, caches, three production
sweeps of all 17 compositions and 7 block configurations, and 7 end-to-end EB fits with forecasts.

**Evidence.** The 537-array comparison is recorded (WP25 report; one run). The structural guarantee
is a committed test: `tests/mcmc/blocks/test_dummies.py::test_dummy_path_is_closed_without_dummies`
(lines 601–640) replaces every dummy helper with a raising stub and traces every pre-WP25
composition. It also checks `Empty` in `FitCache.dummy`, `SharedPrior.dummies` and
`CoefPriorArrays.dummy_w`, and has a positive control with rows. The same invariance practice was
reused for F1 (every structural program bitwise identical to the baseline; commit `3d3f51b`,
issue #3 item 6) and F2 (the 656-array set, §2.3); the 656-array set is documented for F2 only
(corrected by fact-check).

**Status.** Recorded (array comparison); verified (closed-path test). **Publishable:** yes.
Recommended wording: "verified structurally; bitwise array identity recorded on one machine".

---

## 5. Findings about other software

### 5.0 Ground rules before any of this goes into the paper

- **Clean room.** SPEC §2.1 (lines 77–86) forbids translating GPL source. It does not forbid reading
  it to confirm behaviour.
  - N02 states that GPL sources "were read only to confirm published defaults and behaviour …
    No code was transcribed" (N02 line 6, and line 608 for the method).
  - The D-7 mechanism was confirmed by reading **stochvol 3.2.9** source (N02 §7 line 307).
  - `tests/parity/SV_PARAMETRIZATION.md` lines 9–11 say the parity *note* did not read the sources.
    Its stochvol numbers come from running **stochvol 3.2.1**.
  - `tests/parity/BVAR_PARAMETRIZATION.md` lines 5–9 say BVAR 1.0.5 was established by running the
    package *and reading its source*.
  - The paper's clean-room statement must match this exactly ("read for behaviour, never
    translated"), and it must not imply that sources were never read (Q6).
- **Versions.** Every third-party claim below is about one version:
  - stochvol 3.2.1, R 4.1.2 x86_64 (SV_PARAMETRIZATION lines 13–14);
  - BVAR 1.0.5 (BVAR_PARAMETRIZATION line 15);
  - BoE `bvar` 0.3.5 (BOE_BVAR_NOTES line 4);
  - numpy 2.5.4 / Accelerate on macOS 26.6.

  NOTES' "Before submission" (lines 146–151) requires re-testing on current versions and
  contacting maintainers first.
- **Venue sensitivity.** `BVAR` (Kuschnig & Vashold 2021) and `stochvol` (Kastner 2016; Hosszejni &
  Kastner 2021) are themselves JSS software papers (SPEC line 763; N02 lines 583–584). Claims about
  them need the strongest evidence and neutral wording ("in version X we observe …").
- **R availability.** No R is installed on the machine where Phase 2 was finished. Queue items 4
  (regenerate the SV-t fixture under stochvol ≥ 3.2.9) and 5 (bayesianVARs) were skipped for that
  reason (issue #3, "Items 4 and 5: skipped").

### 5.1 stochvol: the non-centred step takes `|σ|` without flipping `h̃` (D-7; NOTES 1)

**What.** In the non-centred parametrisation `h = μ + σh̃`, the extended posterior is invariant under
`(σ, h̃) ↦ (−σ, −h̃)`. The measure-preserving fold flips both. Taking `|σ|` alone reflects `h` about
`μ` and does not preserve the posterior (N02 §7 lines 304–307). rai-bvar folds
(DESIGN line 53, D-7; line 928). Frühwirth-Schnatter & Wagner (2010) state the sign
non-identification and use a random sign switch, of which the deterministic fold is the
measure-preserving equivalent (N02 line 321) [CITE?].

**How found.**
1. Design cross-check (N02 §7):
   - stochvol 3.2.9 source read for behaviour: `draw_theta_2block` returns `fabs(sigma_new)`, and
     `update_fast_sv` then sets `h = μ + σh̃` with the unflipped `h̃`.
   - A clean-room prototype that differs only in this detail fails Geweke in 3/3 seeds (max |z| 16.0,
     18.8, 19.2). The flip version passes (2.16, 1.25, 1.18).
   - The prototype scripts (`phase2-research/xc02_check/c4_geweke.py`) are **not** in the
     repository. Status: recorded.
2. Our committed mutant: **M6** ("NC |σ| without flipping h̃") is caught by U (1.94 / 0) and G
   (max |z| 7.4 / control 2.34) (`tests/mutation/mutants.py` line 783, `EVIDENCE` line 841). There
   is also a block test, `tests/mcmc/blocks/test_sv_core.py::test_block_geweke_catches_abs_sigma_without_flip`
   (lines 1445–1449). Status: verified, and reproduced on x86-64 Linux with the rest of M1–M16
   (mutants.py lines 828–830).
3. Against stochvol itself: a four-observation problem `y = (0.4, −2.5, 0.3, 0.6)`, `μ ~ N(−1, 0.25)`,
   `Beta(20, 1.5)`, `B_σ = 0.1`, with the exact posterior from self-normalised importance sampling
   (4 × 10⁶ draws). 5/25/50/75/95% quantiles of `h_2` (SV_PARAMETRIZATION §3.1, lines 92–104):

   | Source | `h_2` quantiles |
   | :- | :- |
   | exact | −0.663, −0.259, 0.077, 0.486, 1.230 |
   | rai-bvar `exact=False` | −0.663, −0.264, 0.074, 0.480, 1.225 |
   | stochvol centred only | −0.659, −0.265, 0.071, 0.478, 1.215 |
   | stochvol NC only | −0.843, −0.303, 0.056, 0.469, 1.210 |
   | stochvol ASIS (default) | −0.837, −0.300, 0.059, 0.473, 1.218 |

   Our fold fraction there is 27%. The stochvol rows are **recorded**: no script for the
   four-observation stochvol runs is committed (`tools/make_r_fixtures.R` has none), and our side of
   this Gaussian four-observation comparison is not a committed test either (the committed exact-IS
   test is for SV-t, §5.2).

**Where it matters.**
- On the committed `T = 300` stochvol fixtures our fold fraction is 0–1.22%, and parity holds within
  4 quantile-MCSE without invoking the D-7 exclusion (SV_PARAMETRIZATION §6, lines 241–265).
- It also holds on `bvars_swa_n7p4_sv` with fold fractions up to 7.1% (series 7). That oracle's SV
  step is stochvol's sampler, so at `T = 200` (the `T` attribute of all three `bvars_swa_*.nc`
  fixtures; corrected by fact-check) the bias is not visible within the fixtures' MCSE.
- On real data, the committed CR report warns of fold fractions above 1% for 12–30 origin-messages
  per SV arm (`clark_ravazzolo.md` lines 385, 390, 393, 399; the largest of the shown examples is
  0.0676, the maximum over origins is not reported, see §2.3; corrected by fact-check), with
  samples of roughly 80–186 observations (1965Q1 to the 1985Q1–2011Q2 origins).
- So the regime exists in practice. The size of the bias there is unmeasured (Q7).

**Status.**
- Mechanism and its detectability on *our* sampler: verified (M6).
- stochvol behaviour: recorded for 3.2.1 (numbers) and 3.2.9 (source reading).

**Publishable now?** **No.** Re-run the four-observation comparison against the current stochvol
release, commit the R snippet and its output as a fixture, and contact the maintainers first (NOTES
line 151). After that, publish it with neutral wording and the caveat that it is visible only when
the fold fraction is large, that is in small samples or near-homoskedastic series.

### 5.2 stochvol 3.2.1: the Student-t sampler misses the standardized-t posterior (NOTES 2)

**What.** The SV-t fixture `stochvol_v_t5` disagrees with our sampler far beyond Monte Carlo error
(SV_PARAMETRIZATION §3.2, lines 109–129):
- our `μ` is about 0.10 higher (`z` = 29 at 4 × 6,000 kept draws, 62 at 4 × 40,000);
- `τ_t` at the largest observations has a heavier upper tail (95% quantile 7.5 vs 5.4);
- the median `τ_t` is 0.72 vs 0.81.

The four-observation problem with `t_5` errors and an exact IS posterior (`τ` integrated out)
decides between the two. Our `exact=False` and `exact=True` quantiles match the exact ones to
within 0.003 for `μ`, 0.008 for `h_2` and 0.06 (about 0.6%) for `τ_2`, the largest gap being
`τ_2`'s 95% quantile (SV_PARAMETRIZATION lines 120–122; corrected by fact-check: "about the third
decimal" overstated the `h_2` and `τ_2` agreement). stochvol's do not, with or without ASIS (for example
`τ_2` 95% quantile: exact 9.171, ours 9.115/9.124, stochvol ASIS 7.681, centred 5.923). The
unit-scale `τ ~ IG(ν/2, ν/2)` convention does not explain stochvol's numbers either. The deviation
persists without the NC step, so it is not D-7. The cause inside stochvol was not investigated,
because that would mean reading GPL code (line 130).

**Evidence.**
- Our side is verified:
  - `tests/parity/test_stochvol.py::test_sv_t_matches_the_exact_importance_sampling_posterior`
    (lines 612–636): 10⁶ IS draws, 45 comparisons, max |z| 1.83 (SV_PARAMETRIZATION line 254).
  - `stochvol_v_t5` is a strict `xfail(raises=OracleDeviation)` (test_stochvol.py lines 182–185). It
    is reached only after our `T = 300` run passes the R-hat guard, finite-`z` and a DGP-recovery
    guard (posterior ranks of the truth 0.58, 0.55, 0.75; coverage 0.85 for `h` and 0.92 for `τ`;
    SV_PARAMETRIZATION lines 141–151).
- The stochvol side is recorded for 3.2.1 only. The version caveat is at lines 153–156.

**Status.** Recorded (3.2.1 only). The unblock path (lines 158–170) has not been run, because there
is no R (issue #3, item 4 skipped).

**Publishable now?** **No**, as a claim about stochvol: NOTES 2 says "the unblock path in §3.2 must
be run before any public claim. Report upstream first." **Yes**, as a methods point, without naming
a defect. The point: an R-generated parity fixture can itself be wrong. The package handled this
with an exact low-dimensional oracle, and a strict `xfail` narrowed to the deviation fails on
crashes, NaNs, non-mixing or an unexpected pass. Those four outcomes were checked with mutated
scratch copies (line 151).

**Wording risk.** SV_PARAMETRIZATION line 263 says "the stochvol fixture is wrong". The paper should
say "in stochvol 3.2.1 the Student-t sampler did not reproduce the exact posterior on our test
problem".

### 5.3 bayesianVARs 0.1.8: blocked, no finding

No CRAN macOS binary exists for R 4.1, and a source build needs gfortran (SV_PARAMETRIZATION §4,
lines 172–186). Probes P1–P5 (units of `normal_sds`, storage and sign of `U`, the intercept row of
`PHI`, the `sv_para` layout, `cholesky_priorhomoscedastic`) are unresolved. The parity row was
replaced by an exact system-wide Gibbs oracle written in R with rai-bvar's conventions, which our
reduced form matches on all three fixtures (lines 198–206, 255–265). **Nothing to publish** about
bayesianVARs. The replacement-oracle design can be mentioned.

### 5.4 R `BVAR` 1.0.5 (not in the NOTES ledger; proposed status per item)

Source: `tests/parity/BVAR_PARAMETRIZATION.md`, with fixtures generated by `tools/make_r_fixtures.R`
(R 4.1.2 x86_64, BVAR 1.0.5, mvtnorm 1.1-3; `r_bvar_logml.json` `meta`). The tests compare against
the *exact* values and keep BVAR's own numbers as `*_bvar`
(`tests/parity/test_r_bvar.py::test_fixture_log_ml_matches_numpy_reference`, lines 321–330).

**(a) The dummy-only marginal-likelihood term plugs in the prior mean** (BVAR_PARAMETRIZATION §5,
lines 129–137). BVAR evaluates `L_d` with `B̂ = b`. That is exact only if `Y_d = X_d b`, which holds
for SOC/SUR with a random-walk prior mean. R6 recomputed `log_ml_bvar − log_ml` from the committed
fixture:
- sim, `b = 0`, GLP `ȳ₀`: +0.33, +1.23, +5.07 (SOC; λ = 0.1, 0.2, 0.5); +6.627 at every λ (SUR);
  +5.40, +6.41, +10.73 (SOC+SUR);
- sim, `b = 0`, BVAR's `ȳ₀`: 0.12–5.28 across the grid;
- every random-walk-mean case: `bvar_is_exact = True`, with differences ≤ 1e-8.

The note's "0.3–5 with SOC, a constant 6.6 with SUR" matches. **Status (proposed): verified** for
the numbers (committed fixture); the description of BVAR's behaviour is recorded for 1.0.5.

**(b) Consequence: the EB objective becomes degenerate** (lines 134–136). From R6 on
`r_bvar_eb.json`, run `sim_p4_zero_lambda_soc_sur_glp`:
- BVAR's `optim` stops at `soc = sur = 1e-4` (both at their lower bounds) with value −565.49;
- the exact objective's interior mode is λ 0.208, μ 0.331, δ 0.617, with log posterior −748.51.

**Status (proposed): verified** (fixture).

**(c) `predict.bvar`: two bugs** (§9, lines 228–240). The noise is `Σ z` (covariance `Σ²`) instead
of `chol(Σ) z`, and the shocks are not propagated through the dynamics. R6, ratio of the correct
90% band width to BVAR's native band (`r_bvar_mh.json`, `forecast` vs `forecast_bvar_native`):

| Run | h = 1 | h = 8 |
| :- | :- | :- |
| `qd_p4_rw_lambda_soc_sur_glp` | 1.24–3.19 | 2.59–3.62 |
| `qd_p4_rw_lambda` | 1.21–2.93 | 2.14–2.67 |
| `sim_p4_zero_lambda` | 0.90–1.37 | 1.30–1.70 |

The note's "about 3 times too narrow at `h = 8`" holds for the QD SOC+SUR run (2.6–3.6); the
λ-only QD run gives 2.1–2.7 (corrected by fact-check). The direction of the
`Σ`-vs-`chol(Σ)` error depends on units: where `Σ_ii > 1` the native band is too *wide* (sim,
`y1`, `h = 1`: ratio 0.90). The fixture labels the native output "BVAR bug … Reference only"
(`tools/make_r_fixtures.R` lines 1024–1033). **Status (proposed):** numbers verified (fixture);
behaviour recorded for 1.0.5.

**(d) Further deviations from GLP (2015)** (§10 table, lines 242–257). Status (proposed): recorded.
These are documentation and convention points rather than errors, except where noted:
- the auto ψ is a standard deviation used as a variance;
- the ψ hyperprior is IG(0.004, 0.004) vs GLP's IG(0.02², 0.02²);
- `ȳ₀` is the mean of raw rows `p+1…2p` instead of `1…p`, which moves the QD SOC+SUR log ML from
  −593.2554 to −593.6052 (line 97);
- the MH proposal is `diag(J² · scale_hess)` rather than an inverse Hessian, and in one dimension a
  variance formula is used as a standard deviation (lines 198–205);
- the normal-equation `B̂` loses about 1e-8 absolute on level data, and BVAR's NIW numbers deviate by
  up to 6e-9 (lines 221–224, 297).

**Publishable now?** **No.**
- BVAR is a JSS package, and the version used (1.0.5) must be checked against the current CRAN
  release.
- Items (a)–(c) should be reported to the maintainers first.
- NOTES should first gain a section-A entry for them (Q8).

With those done, (a)–(c) are strong because the evidence is committed fixtures with exact
references.

### 5.5 Bank of England `bvar` 0.3.5 (MIT; not in the NOTES ledger)

Source: `tests/parity/BOE_BVAR_NOTES.md`, "read from the source of `bvar` 0.3.5 … and checked
numerically" (lines 3–6). The behaviour is pinned by the `test_boe_*` tests in
`tests/parity/test_boe_bvar.py` (lines 362–458; the `test_parity_*` comparisons follow at lines
484–617+). `test_boe_version_is_mapped` (test file lines 362–365; BOE_BVAR_NOTES line 13) xfails
when the installed version differs from `BOE_VERSION_MAPPED = "0.3.5"` (line 42); the installed
version in the repository's environment is 0.3.5, so the pin is live (corrected by fact-check).

- **(a) `ȳ₀` = mean of observations `p+1…2p`** (§4, lines 82–97). This moves `log p(Y | h)` by
  0.1–1.1 nats on the QD levels data (qd-p4-sur −0.10, soc-sur +0.23, p2-soc-sur +0.70, soc +1.07).
  The convention is pinned by `test_boe_dummy_initial_mean_convention`.
- **(b) Forward-difference gradients on a noisy objective** (§6, lines 145–151).
  - The log ML has rounding noise of about 5e-9 nats, because `cond(V⁻¹ + Z'Z) ≈ 1e10`.
  - A forward difference with step 1.5e-8 turns that into gradient noise of order 0.3, and BFGS
    stops with "precision loss".
  - Relative error against the polished mode of BoE's own objective: QD p = 4, (λ, α) 2.8e-3; with
    SOC+SUR, λ 5.9e-3, α 6.1e-3, μ 1.1e-2, δ 2.6e-2.
  - With the optional JAX backend, autodiff gradients are used instead (lines 140–142). This applies
    to the NumPy backend only.
- **(c) `S_post` cancels** (§7, lines 173–178): up to 4e-6 relative error elementwise (4e-9 on the
  correlation scale) on level data with `cond(K_A) ≈ 3e10`.
- **(d) Docs vs code** (§11, lines 235–247), for example:
  - the claim that cross-variable coefficients are shrunk harder is false for `NaturalConjugate`;
  - the docs describe L-BFGS on bounded parameters, while the code uses unbounded BFGS in softplus
    space and always frees `c3`;
  - `stationary=True` silently disables SOC/SUR (line 32);
  - a failed factorisation returns the sentinel −1e5 (line 116).

**Status (proposed): recorded** (numbers from the note; the conventions are pinned by tests).
**Publishable now?** Not yet: same version check and maintainer contact. Items (b) and (c) are
generic numerical lessons, about forward differences on objectives with rounding noise and about the
`A'KA` subtraction form of `S_post`. They connect naturally to §2.6 lesson 3.

### 5.6 Apple Accelerate: spurious divide-by-zero in `slogdet`/`det` (commit `dd8244e`)

**What.**
- Platform: M5 Max, macOS 26.6, numpy 2.5.4 on Accelerate.
- `np.linalg.slogdet` and `det` raise `RuntimeWarning: divide by zero` from Accelerate's LU
  (`getrf`) on the well-conditioned SPD tridiagonal matrices of `test_tridiag_sampler_matches_dense[41-41]`
  and `[65-51]`. `cholesky` and `inv` are clean on the same matrices.
- The suite's `error::RuntimeWarning` filter turned the oracle's warning into a failure ("Before: 2
  failed"; issue #3 item 1).

**Fix.** The oracle takes `log|Q|` and the pivots from the dense Cholesky factor (exact for SPD `Q`).
The tolerances are unchanged and the code under test is untouched (`tests/mcmc/unit/test_linalg.py`
lines ~620–631, comment in the diff). After the fix, all 60 `test_linalg` tests pass on the M5 Max.
R1 confirms this on `62d15f6`.

**Status (proposed): recorded.** No committed test shows the warning itself, and the upstream
behaviour was not isolated in a minimal numpy reproducer in the repository.

**Fact-check reproduction (recorded, scratch script, not committed).** On `62d15f6`, building the
matrices of `[41-41]` and `[65-51]` with the test's own helpers (`sv_like_tridiag`,
`dense_tridiag`), `np.linalg.slogdet` and `np.linalg.det` each emit three RuntimeWarnings (divide
by zero, overflow, invalid value), not only divide by zero; `cholesky` and `inv` emit none; the
matrices have `cond ≈ 61` and `63`. `[1-1]` and `[2-2]` emit no warning. (added by fact-check)

**Publishable:** only as a platform caveat in the testing section. A Python test suite that turns warnings into errors depends on the
behaviour of the BLAS/LAPACK backend.

### 5.7 Platform: intermittent heap corruption in a Phase 1 test on macOS (issue #3)

**What (recorded, issue #3, "Item 2, last directories" and "Item 2, complete").**
- In the weekly tier on the pinned `408a7f4`, the single-process `tests/unit` run hung for 2.8 h at
  100% CPU on its 8th test, `tests/unit/test_eb_convergence.py::test_eb_converges_on_a_simulated_n50_p4_panel[0]`
  (a Phase 1 analytic EB test, `@pytest.mark.slow`, lines 353–355).
- Alone, the test passes in about 2 s, then the interpreter segfaults (rc 139) during garbage
  collection. This happened 3/3 times on `408a7f4` and also on the PR head `09a2dc1`.
- Toggling the hypothesis plugin, `--assert=plain`, `-p no:warnings`, `-p no:faulthandler`,
  `VECLIB_MAXIMUM_THREADS=1` or malloc guards did not remove it.
- `-s`, `--capture=sys`, `--capture=tee-sys` and `PYTHONMALLOC=debug` did remove it at first. A
  later `--capture=sys` sweep crashed with SIGTRAP after 20 tests, so fd capture is not the cause.
- Outside pytest it crashed 1 time in 13, in `scipy.optimize` (`issparse` → `abc.__subclasscheck__`),
  which points to corrupted type objects.
- One-process-per-test: 60/60 unit tests pass, and the four `n50[0–3]` tests crash on the first try
  and pass on their first clean retry.
- The issue's reading is heap corruption by a native extension:
  - numpy 2.5.4 and scipy 1.18.1 are on Accelerate;
  - numba is loaded through `scoringrules` (confirmed by the fact-check: `import rai_bvar` alone
    loads `scoringrules` and `numba`);
  - item 1's spurious `slogdet` warnings "point in the same direction", but the issue says this is
    "**not established**" (the dossier's earlier quotation "the Accelerate link is not
    established" was a paraphrase, not the issue's wording; corrected by fact-check).
- No native backtrace was obtained (lldb waited on a macOS authorization prompt).

**Re-run by this dossier (R5, recorded).** Five runs of the same single test on `62d15f6` with
`-p no:cacheprovider` all ended abnormally:

| Run | Exit | Where it crashed |
| :- | :- | :- |
| 1 | rc 139 (SIGSEGV) | after the test passed, during garbage collection inside pytest's assertion rewriting |
| 2 | rc 138 (SIGBUS) | inside `scipy.special._spfun_stats.multigammaln`, called from `src/rai_bvar/engines/analytic.py` line 442 (`_core`) during L-BFGS-B |
| 3 | rc 133 (SIGTRAP) | no traceback |
| 4 | rc 139 | while pytest formatted a failure report; the failure itself is not shown in the log |
| 5 | rc 133 | after the test passed |

The varying crash sites, including one inside a pure-numeric scipy routine called by the Phase 1
engine, fit heap corruption. The root cause is still unknown. The fact-check ran the same command
three more times: 3/3 rc 139 after the test passed, during garbage collection inside pytest's
assertion rewriting (the site of run 1); runs 2–5 of the table above are not backed by committed
logs (added by fact-check).

**What it is and is not.**
- It is a reproducible platform instability on this Mac with this dependency set, triggered by an
  `n = 50` EB fit.
- It is not a numerical result of rai-bvar, and it is not established to be an Accelerate bug. It
  is not tied to the MCMC work: it is Phase 1 code and reproduces on the pinned commit (issue #3).
- It matters for the paper's reproducibility claims. If the corruption originates in a dependency
  called by the analytic engine, results on that platform could in principle be affected without a
  crash. There is no evidence of that, but also no evidence against it (Q9).

**Status (proposed): recorded** (issue #3 + R5). **Publishable:** only as a platform caveat ("on
macOS 26.6 / Apple silicon with numpy 2.5.4 and scipy 1.18.1 on Accelerate, one test crashed the
interpreter intermittently; Linux CI is the reference platform"), after:
- the Linux check suggested in issue #3 (rerun the test in the weekly Linux job);
- ideally, a bisection of numpy/scipy/numba versions or a native backtrace.

### 5.8 Cross-reference

Errors in *published algorithms* rather than software belong to the validation-methodology dossier
(NOTES section B):
- the CCM-2019 triangular step (mutant M1; Carriero, Clark & Marcellino 2019 and the 2022
  corrigendum, SPEC lines 770, 772);
- the RB log-score tail breakdown (NOTES 8b).

---

## 6. Publishability summary and actions before submission

| Item | Can go in now | Must happen first |
| :- | :- | :- |
| F1 QR draw (§2.2) | yes, as a lesson | Linux re-run of R2; optionally commit a test asserting the real-state numbers; prior-art citations |
| Silent-draw observation (§2.2, §2.3) | no | committed tests at the R3/R4 states; a decision on a pivot-floor diagnostic (Q1) |
| F2 Laplacian recursion (§2.3) | yes, as a case study | Linux re-run of the tiny-σ tests; commit an old-path-fails test; position against GTH / Ye [CITE?] |
| F3 replay (§2.4) | yes | — |
| Reproducibility limits (§3) | yes | re-run on the release commit, both platforms |
| Dummy rows, NOTES 13–15 (§4.2–4.4) | yes | re-time NOTES 14 on an idle machine |
| NOTES 12 (§4.1) | twin design yes; the "33 of 165" number no | make the injection reproducible |
| NOTES 16 (§4.5) | yes, worded as structural + recorded | — |
| stochvol D-7 (§5.1) | no | current version; committed R snippet and fixture; maintainer contact; clean-room wording |
| stochvol SV-t (§5.2) | as a methods point only | unblock path (needs R); report upstream |
| BVAR 1.0.5 (§5.4) | no | ledger entry; current CRAN version; maintainer contact |
| BoE bvar 0.3.5 (§5.5) | no | ledger entry; current PyPI version; maintainer contact |
| Accelerate / heap corruption (§5.6–5.7) | as caveats only | Linux check; minimal reproducer or bisection |

General: re-run every "recorded" item on the release commit and cite the hash (NOTES line 148).
Disclose AI assistance as the venue requires (NOTES lines 149–150).

---

## 7. Open questions

- **Q1.** Should `ok` (or a warning) use a relative pivot floor rather than `pivot > 0`? R3 and R4
  show positive rounding-noise pivots with wrong draws. Which remaining formed-Gram draws are
  exposed: the structural coefficient path through `precision_draw`, CCCM `Π`, `cholesky_const`?
  Can a validation state reach them?
- **Q2.** Were any pre-fix SV(rw) results (the committed `clark_ravazzolo_quick.md`, or the first
  full run's main stage) affected by the silent variant of F2 (positive rounding-noise pivot with a
  wrong `h` level)?
- **Q3.** Should the real-state diagnostic numbers of F1 (eigenvalues 14.8 / 1.9e17; QR error 1.6e-8
  sd) and the old-path failure of F2 be asserted by committed tests, so that they become "verified"?
- **Q4.** NOTES 12: expose the Mode II coupling as an override seam (a registered M19), or commit
  the hand injection as a scratch test?
- **Q5.** Machine labels:
  - "M1 Pro" still appears in `SV_PARAMETRIZATION.md` §6 (line 235), in the Geweke test docstring
    (line 40) and in the M1–M16 evidence comment of `mutants.py` (lines 29, 827).
  - Issue #3 says the machine is an M5 Max and that the label was wrong in the WP25 strings, while
    "pre-existing M1–M16 strings are untouched".
  - Were those earlier runs also on the M5 Max?
- **Q6.** Clean-room wording: the paper must say that GPL sources (stochvol 3.2.9, BVAR 1.0.5) were
  read to confirm behaviour and never translated (N02 line 6; BVAR_PARAMETRIZATION lines 5–13). Is
  that acceptable to the author and the venue? Does the PR checkbox of SPEC §2.1 cover it?
- **Q7.** How large is the D-7 bias at the fold fractions seen in the CR data (above 1%; the
  report's shown examples are 1.4–6.8%, corrected by fact-check; at about 80–186 quarterly
  observations)? A simulation at those sizes with an exact or long-run reference would
  turn "visible only in tiny samples" into a quantified statement.
- **Q8.** Add the BVAR 1.0.5, BoE 0.3.5 and Accelerate findings to NOTES section A, with statuses,
  so that the ledger is complete?
- **Q9.** Root cause of the macOS heap corruption. Does the `n = 50` EB test also crash on Linux? Is
  numba (via `scoringrules`) needed for the crash? A bisection of numpy/scipy versions, or a native
  backtrace with Developer Mode, is the next step (issue #3).
- **Q10.** Does the paper present the excess-pivot recursion as new? Check it against GTH /
  Alfa–Xue–Ye / Ye 2008, and against the precision-sampler literature for Gaussian Markov random
  fields (Rue 2001; McCausland, Miller & Pelletier 2011), before claiming any novelty.

---

## 8. [CITE?] references (to be verified by the bibliography stage)

Entries also listed in SPEC §15 (lines 750–811; corrected by fact-check) or N02 §15 (lines
582–593) are marked "(in repo)".
The bibliography stage should still check them against the published versions.

- Alfa, A. S., Xue, J., & Ye, Q. (2002). Accurate computation of the smallest eigenvalue of a
  diagonally dominant M-matrix. *Mathematics of Computation*, 71(237), 217–236. [CITE?]
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM. [CITE?]
- Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian vector autoregressions with
  stochastic volatility and non-conjugate priors. *Journal of Econometrics*, 212(1), 137–154. (in
  repo) [CITE?]
- Carriero, A., Chan, J. C. C., Clark, T. E., & Marcellino, M. (2022). Corrigendum to "Large
  Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *Journal of
  Econometrics*, 227(2), 506–512. (in repo) [CITE?]
- Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation
  in state space models. *IJMMNO*, 1(1/2), 101–120. (in repo) [CITE?]
- Clark, T. E., & Ravazzolo, F. (2015). Macroeconomic forecasting performance under alternative
  specifications of time-varying volatility. *Journal of Applied Econometrics*, 30(4), 551–575. (in
  repo) [CITE?]
- Frühwirth-Schnatter, S., & Wagner, H. (2010). Stochastic model specification search for Gaussian
  and partial non-Gaussian state space models. *Journal of Econometrics*, 154(1), 85–100. (in N02)
  [CITE?]
- Geweke, J. (2004). Getting it right: Joint distribution tests of posterior simulators. *JASA*,
  99(467), 799–804. (in repo) [CITE?]
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions.
  *Review of Economics and Statistics*, 97(2), 436–451. (in repo, SPEC line 758; "GLP (2015)" in the
  parity notes) [CITE?]
- Grassmann, W. K., Taksar, M. I., & Heyman, D. P. (1985). Regenerative analysis and steady state
  distributions for Markov chains. *Operations Research*, 33(5), 1107–1116. [CITE?]
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. [CITE?]
- Hosszejni, D., & Kastner, G. (2021). Modeling univariate and multivariate stochastic volatility in
  R with stochvol and factorstochvol. *Journal of Statistical Software*, 100(12). (in N02) [CITE?]
- Kastner, G. (2016). Dealing with stochastic volatility in time series using the R package
  stochvol. *Journal of Statistical Software*, 69(5). (in N02) [CITE?]
- Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy
  (ASIS) for boosting MCMC estimation of stochastic volatility models. *CSDA*, 76, 408–423. (in
  repo) [CITE?]
- Kuschnig, N., & Vashold, L. (2021). BVAR: Bayesian vector autoregressions with hierarchical prior
  selection in R. *Journal of Statistical Software*, 100(14), 1–27. (in repo) [CITE?]
- McCausland, W. J., Miller, S., & Pelletier, D. (2011). Simulation smoothing for state-space
  models: A computational efficiency analysis. *CSDA*, 55(1), 199–212. (in repo) [CITE?]
- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage:
  Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2), 425–449. (in repo)
  [CITE?]
- Rue, H. (2001). Fast sampling of Gaussian Markov random fields. *JRSS-B*, 63(2), 325–338. (in N02)
  [CITE?]
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization,
  folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*,
  16(2), 667–718. (in repo) [CITE?]
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. P. (2020). Efficiently
  sampling functions from Gaussian process posteriors. *ICML 2020* (pathwise conditioning / Matheron's
  rule). [CITE?] For Matheron's rule itself, a geostatistics source (Journel & Huijbregts 1978,
  *Mining Geostatistics*, or Chilès & Delfiner) [CITE?]
- Ye, Q. (2008). Computing singular values of diagonally dominant matrices to high relative
  accuracy. *Mathematics of Computation*, 77(264), 2195–2230. [CITE?]
- Yu, Y., & Meng, X.-L. (2011). To center or not to center: That is not the question. *JCGS*,
  20(3), 531–570. (in N02) [CITE?]
- Bank of England `bvar` (PyPI, version 0.3.5; github.com/bank-of-england/bvar). Software citation;
  confirm the authors and the preferred citation. [CITE?]
- NumPy (Harris et al. 2020, *Nature* 585, 357–362), SciPy (Virtanen et al. 2020, *Nature Methods*
  17, 261–272), JAX (Bradbury et al. 2018, software). Software citations for the platform caveat.
  [CITE?]

---

## 9. Unsupported claims (moved or flagged by the fact-check)

Claims below could not be supported from committed files or from commands the fact-check ran.
Each is either moved here from the body or kept in place with this note.

- **Moved from §2.6 lesson 5:** "Jittering the diagonal would have turned both into silent bias."
  This is a counterfactual. No repository file, test or run examines a jittered variant of the F1
  or F2 kernels. It is plausible (a jitter large enough to restore the lost pivot changes the
  conditional), but the size of the bias is unmeasured. Use it in the paper only as an argument,
  not as a finding.
- **Kept in §5.7 as recorded, flagged here:** the per-run crash sites of R5 runs 2–5 (SIGBUS in
  `scipy.special._spfun_stats.multigammaln` from `analytic.py` line 442, SIGTRAP with no
  traceback, SIGSEGV while formatting a failure report). No log of these runs is committed, and the
  fact-check's three re-runs gave only the run-1 site (rc 139 during garbage collection in pytest's
  assertion rewriting). The cited source line itself is correct (`analytic.py` line 442 calls
  `multigammaln` inside `_core`, line 425).
- **Kept in §2.3 as recorded, flagged here:** "Pre-fix runs of SV(rw) near `σ → 0` could therefore
  have produced wrong `h` levels without any failure." R3 shows a silent wrong draw for the
  isolated AWOL step at a test state. Whether a whole pre-fix SV(rw) sweep at such a state would
  have kept that draw was not run (it is open question Q2).
