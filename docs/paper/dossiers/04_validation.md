# Dossier 04 — Validation methodology (the core methods contribution)

Scope: how `rai-bvar` establishes that its samplers target the stated posterior, for Phases 0–2
(analytic conjugate engine, JAX MCMC engine with stochastic volatility, evaluation engine). Later
phases appear only where the harness was built to absorb them (the reference registry).

## 0. Snapshot, conventions and status labels

- **Repository snapshot.** `/Users/yigitokar/rai-bvar-paper`, branch `paper-draft`, commit
  `62d15f68c7e7a3de4ce83ee0098293b9c62111b0` (2026-10-23 06:44 −0400; `git log -1`). Every file path
  below is relative to the repository root. `L` = line numbers in that commit.
- **Commands run for this dossier** (read-only with respect to the repository; outputs written to
  the session scratchpad, not to the repo). Machine: Apple M5 Max, macOS 26.6.2, Python 3.12.13,
  jax 0.12.0.1, numpy 2.5.4, scipy 1.18.1 (`sysctl`, `sw_vers`, `uv run python -c ...`), 18 cores,
  load average ≈ 3.5–4.2 from other jobs (`uptime`), no persistent XLA compilation cache (cold).
  1. `uv run --frozen pytest --collect-only -q -p no:cacheprovider` (plus `-m <marker>` variants) —
     suite counts (§3).
  2. `RAI_BVAR_VALIDATION_RECORD=<scratch> uv run --frozen pytest tests/geweke -m "geweke and not slow"`
     — the PR-tier G-small Geweke run of every PR composition (§6.6).
  3. `RAI_BVAR_VALIDATION_RECORD=<scratch> uv run --frozen pytest tests/mutation -m mutation` — the
     full mutation suite, 37 mutant × tier checks (§11).
  4. `uv run --frozen pytest -m "not slow and not network and not bench"` — the PR fast suite, for
     its wall-clock time (§3.3).
- **Fact-check re-run (added by fact-check, 2026-10-23 ≈ 11:00 −0400, same machine and commit, no
  persistent XLA cache, `PYTHONDONTWRITEBYTECODE=1 … -p no:cacheprovider`, records to the fact-check
  scratchpad):** command 1 reproduced every count of §3.1–3.2 (with the function-count correction
  in §3.2); command 2 gave 16 passed in 99.66 s with every statistic of the §6.6 table (G, t_crit,
  max |z| and argmax, stages and q, min RNE) identical to the digits shown; command 3 gave 37 passed
  in 177.40 s with every statistic of the §11.2 table identical to the digits shown (M2 705 / 2.48).
  Per-test times differ slightly from the author's run (load-dependent); see §3.3. Command 4 (run
  from a scratch working directory so that `.hypothesis/` and `.rai-bvar-cache/` were not written in
  the repository) gave the same outcome as the author's run: 4,507 passed, 12 skipped, 1 strict
  xfail, 0 failed, 1,006.9 s (§3.3).
- **Durability of my run records.** The JSON records of commands 2 and 3 (`geweke.jsonl`,
  `mutation.jsonl`) were written to the session scratchpad, not to the repository; the tables below
  carry their key fields. For the paper, the release run should point `RAI_BVAR_VALIDATION_RECORD` at
  a committed path so every table has a committed source (Q1, Q11).
- **Status labels** (following `docs/paper/NOTES.md`, header):
  - **verified** — reproducible from a committed test or fixture (NOTES' own label where NOTES
    gives one);
  - **recorded** — a number written in a committed file (docstring, note, report) from a run that
    was not re-run independently;
  - **to check** — must be re-run or re-derived before it goes into the paper;
  - **reproduced here** — I re-ran it for this dossier (command, machine and date above). This is
    stronger than "recorded" but is still one machine and one commit, not the release commit.
- Literature: names and years as the repository's notes give them; every citation is flagged
  **[CITE?]** for the bibliography stage (most have DOIs recorded in
  `docs/phase2/06_validation_geweke_sbc.md` §13, L805–824, verified there via Crossref, but I did not
  re-verify them).

---

## 1. The claim in one paragraph (for the abstract / introduction)

`rai-bvar` treats "the sampler targets the posterior" as a testable property with a measured power.
The rule is written into the specification: "No sampler is merged without Geweke and SBC tests
passing", motivated by the CCM (2019) episode, "a published and widely used sampler that targeted
the wrong posterior" (`SPEC.md` §3 R6, L121). The implementation is a layered harness — deterministic
identities (U), block-conditional tests (B), Geweke joint-distribution tests (G), simulation-based
calibration (S), exact analytic twins (A), parity against independent software (P) — whose *power*
is itself tested by a mutation suite (M) of 18 deliberately wrong samplers, each of which must be
caught by some tier while the unmutated control passes (`docs/phase2/DESIGN.md` §9, L1699–1970;
`tests/mutation/mutants.py`). Every comparison with "the truth" uses a truth written independently
of the kernels (the two-implementation rule, DESIGN §9.1, L1706–1711), and every statistical test
runs the **production** sweep, never a copy (DESIGN §9.1, L1729–1733; `src/rai_bvar/validation/joint.py`
docstring L1–11). On the snapshot commit, all 37 mutant × tier checks catch their mutant with a
passing control (reproduced here, 182 s), every PR-tier Geweke composition passes (reproduced
here, 95 s), and the whole PR fast suite passes (4,507 passed, 12 skipped, 1 strict xfail, 0 failed;
1,003 s with a cold compilation cache; reproduced here).

Candidate methods contributions, each developed below with evidence:

1. A **tiered test architecture with a mutation-measured power table** (§2, §11).
2. **Parallel, prior-started Geweke chains with no burn-in** and `t_{C−1}` critical values, with a
   closed-form design rule for the test prior (`r = Var_prior/Var_post ≤ 5`), a finite-variance rule
   for test functions, and the Gandy–Scott sequential wrapper so that marginal p-values trigger a
   fresh 4× rerun instead of a red build (§6).
3. **SBC with two-pass per-replication thinning, exact γ p-values and a pooled ESS gate**, including
   the erratum that per-replication ESS cannot be certified from `L = 99` draws (§7).
4. **Exactness-preserving test devices**: exact-mode SV is exact for any mixture table and offset,
   so a deliberately coarse table gives the tests power over the exact correction (M7), and
   table/indicator mutants (M11, M12) are bugs only of the approximate sampler (§11.3; NOTES B.3).
5. **The log-likelihood as an SBC/Geweke statistic** catches the CCM-2019 error where parameter
   ranks are blind (§7.5, §11.3; NOTES B.4).
6. **Analytic twins at realistic size** (`n = 20, p = 4, T = 300`), with a CDF-form quantile test
   that avoids the heavy-tailed order-statistic MCSE (§9).
7. **Parity findings about reference software** (stochvol 3.2.1's ASIS sign handling and its SV-t
   sampler; R `BVAR` 1.0.5's forecast noise) — handle with care, all "recorded/to check" (§10).

---

## 2. Tier map: what each tier is, where it lives, when it runs

### 2.1 Design (DESIGN §9.3–9.10)

| Tier | Content | Designed cadence | Source |
| :--- | :--- | :--- | :--- |
| **U** | deterministic and identity tests: dense oracles with injected noise, density-ratio identities, MH acceptance-ratio identities, closed forms, infrastructure (float64, no-JAX import, bitwise invariances, failure path) | every PR | DESIGN §9.3, L1789–1816 |
| **B** | block-conditional SBC (`N = 200`, `L = 99`) for exact-draw blocks; one-block invariance (`L = 20` kernel applications) for MH/ASIS blocks; Bonferroni α = 1e-3; Gandy–Scott `k = 3, Δ = 4` | every PR | DESIGN §9.4, L1818–1823 |
| **G** | Geweke joint-distribution tests: G-small (`C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3`), G-full (`C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`); approximate-sampler rows recorded, not gating | G-small PR, G-full nightly | DESIGN §9.5, L1825–1858 |
| **S** | full SBC, `N = 1000, L = 99`, burn 300, pilot 1000, γ test, α = 1e-2, ESS gates | nightly | DESIGN §9.6, L1860–1873 |
| **M** | mutation suite M1–M18; "a mutant caught by none blocks the release" | weekly | DESIGN §9.7, L1875–1898 |
| **A** | analytic twins and exact oracles A1–A7 | A1, A2, A5 (n = 3), A6 PR; A3, A4, A5 (n = 20), A7 nightly (as implemented A7 also has a PR-sized check, §9; added by fact-check) | DESIGN §9.8, L1900–1910 |
| **P** | parity with stochvol and bayesianVARs (and Phase 1 parity with R `BVAR` and the BoE `bvar`) | stochvol PR; bayesianVARs nightly | DESIGN §9.9, L1912–1958 |
| Pre-release | BENCH-P2, Clark–Ravazzolo replication | release | DESIGN §9.10, L1970 |

Markers (`pyproject.toml`, `[tool.pytest.ini_options]`): `slow`, `network`, `geweke`, `sbc`,
`mutation`, `bench`, `parity_long`, `weekly`; `--strict-markers`; `filterwarnings =
["error::RuntimeWarning"]`. MCMC test modules add `filterwarnings = "error:Explicitly requested dtype"`
so that any float32 truncation fails the test (DESIGN §9.10, L1962–1963; e.g.
`tests/geweke/test_geweke_compositions.py` `pytestmark`). Plain `pytest` skips `slow` and `network`
tests unless a marker expression is given (`tests/conftest.py`, `pytest_collection_modifyitems`).

### 2.2 As implemented (deviations the paper must state)

1. **The CI "nightly" job runs weekly.** `.github/workflows/ci.yml`: `schedule: cron "17 3 * * 0"`
   ("weekly (Sunday)") and the job comment `nightly:  # weekly despite the name (DESIGN §9.10 nightly
   + weekly tiers, run together)`. The PR job runs `pytest -m "not slow and not network and not
   bench"` on Ubuntu for Python 3.12, 3.13, 3.14 with a persistent XLA cache keyed by CPU model;
   macOS runs the fast suite weekly only; BENCH-P2 runs on manual dispatch. So G-full, S, A3/A4,
   bayesianVARs/SWA parity and the mutation suite all run **weekly** in CI, not nightly. Status:
   verified (committed workflow). The paper should say "weekly" or explain.
2. **Composition-level Tier B is weekly ("slow"), not per PR.** `tests/sbc/test_block_tier.py`
   docstring: "All of this tier is nightly (`slow`): the per-PR Tier B of every block runs in the
   kernel work packages' suites (WP10–WP13)". Verified (module `pytestmark = [pytest.mark.slow, …]`).
3. **Block-level Geweke configurations** run at G-small per PR inside the kernel suites
   (`test_linear_structural.py`, `test_recursive.py`, `test_cccm.py`, `test_sv_core.py`,
   `test_robust.py`); their `JointSpec.step` versions in `tests/geweke` are `slow`
   (`tests/geweke/test_geweke_compositions.py` docstring, "Deviations from DESIGN §9.10").
4. **SBC thinning cap `t_max = 200`**, not 50 (§7.4; NOTES B.6).
5. **Block-level configurations hold frozen blocks by restoring their leaves after the production
   sweep** (`hold_frozen`), not by identity kernels injected through `_overrides`; exact only when
   every frozen block runs after every live block, otherwise an identity override of a named internal
   is used (`contemp_only` → `cccm_core`) (`src/rai_bvar/validation/joint.py` docstring L30–38, `hold_frozen` L285).
6. **SPEC §10.6 says diagnostics "via ArviZ"; the implementation is in-repo NumPy** (D-24, DESIGN
   L70), pinned to ArviZ numbers in tests (§12.4).
7. **SPEC §10.3 says "a Holm correction"**; the decision rule is Bonferroni on `min p`, with Holm used
   only to list failures — equivalent for the global decision (N06 §3.5, L245–250;
   `src/rai_bvar/validation/sequential.py` docstring L3–5).
8. **Parity oracles differ from the design**: stochvol 3.2.1 installed vs 3.2.9 named; the SV-t row
   is a strict `xfail`; bayesianVARs is blocked and replaced by an exact system-wide Gibbs oracle
   (§10.2–10.3).

---

## 3. Quantifying the suite

### 3.1 Test counts by marker (parametrised test IDs; reproduced here, command 1)

| Selection | Test IDs |
| :--- | ---: |
| all collected | **4,964** |
| PR fast suite: `not slow and not network and not bench` | **4,520** |
| weekly CI job: `(slow or network) and not bench` | **441** |
| `bench` (pre-release) | 3 |
| `slow` | 438 |
| `network` | 6 |
| `geweke` | 95 (56 in `tests/geweke/` + 39 block-level Geweke tests in `tests/mcmc/blocks/`: `test_sv_core.py` 21, `test_linear_structural.py` 7, `test_recursive.py` 4, `test_cccm.py` 3, `test_robust.py` 2, `test_mixture.py` 1, `test_zero_mean.py` 1). Of the 39 block-level ones, 24 run per PR and 15 are `slow` (`test_sv_core.py::test_block_geweke_variants_nightly` 14, `::test_block_geweke_power_against_stale_w_nightly` 1); `-m "geweke and not slow"` collects 40 = 16 + 24 (corrected by fact-check: cadence split added, `--collect-only -m "geweke and slow"`) |
| `sbc` | 22 (one per composition in `tests/sbc/test_sbc_compositions.py::test_sbc`) |
| `mutation` | 37 (the mutant × tier checks; `test_registry` is unmarked and runs per PR) |
| `weekly` | 38 (37 mutation + 1 in `tests/integration/test_ordering.py`) |
| `parity_long` | 15 (11 `test_stochvol.py`, 4 `test_bayesianvars.py`) |

4,520 + 441 + 3 = 4,964. Counts are parametrised IDs, not functions.

### 3.2 Test IDs and test functions by directory (reproduced here)

| Directory | IDs | `def test` functions | Main tier |
| :--- | ---: | ---: | :--- |
| `tests/unit` | 1,912 | 591 | Phase 0/1 unit + Phase 2 API |
| `tests/mcmc/blocks` | 578 | 249 (corrected by fact-check: was 255) | U (oracles, identities), block B, block G, A6, A7 |
| `tests/mcmc/unit` | 398 | 133 | U infrastructure (keys, dtypes, devices, import) |
| `tests/parity` | 364 | 63 | P (R `BVAR` 147, fixture schema 135, BoE 38, stochvol 34, bayesianVARs/SWA 10) |
| `tests/contract` | 322 | 38 | forecast contract (MCMC and analytic) |
| `tests/validation` | 300 | 119 (corrected by fact-check: was 121) | self-tests of the harness statistics and references |
| `tests/analytic` | 299 | 113 | A1–A5, Phase 1 analytic |
| `tests/mcmc/diagnostics` | 238 | 63 | runtime diagnostics |
| `tests/mcmc/runner` | 161 | 88 | U infrastructure (chunking, plan, recompiles, failures) |
| `tests/integration` | 93 | 67 | backtest, ordering |
| `tests/sbc` | 69 | 8 | S (22 compositions), composition-level B (43) |
| `tests/priors` | 65 | 41 | prior builders |
| `tests/mcmc/forecast` | 65 | 41 | forecast/LPD/IRF |
| `tests/geweke` | 56 | 5 | G (15 G-small PR, 7 block, 22 G-full, 11 inexact recorded, 1 design-row check) |
| `tests/mutation` | 38 | 2 | M |
| `tests/bench` | 6 | 6 | BENCH-P2 |

The function counts are top-level `^def test` definitions per directory (corrected by fact-check:
the original pattern `'^def test\|^    def test'` also matched eight `def test_functions(self, …)`
JointModel protocol methods that pytest does not collect — six in `tests/mcmc/blocks`
(`test_robust.py`, `test_zero_mean.py`, `test_mixture.py`, `test_sv_core.py`,
`test_linear_structural.py`, `test_recursive.py`) and two in `tests/validation`
(`test_geweke_selftest.py`, `test_sbc_selftest.py`) — so it over-counted; the repository has no
`class Test*`, so the top-level count is exact). Of the `tests/validation` functions: `test_sbc_selftest.py` 25,
`test_geweke_selftest.py` 16, `test_reference_moments.py` 14, `test_simulate.py` 13,
`test_log_joint.py` 11, `test_uniformity.py` 11, `test_sequential.py` 10,
`test_mixture_transcription.py` 8, `test_identities_toy.py` 7, `test_reference_dummies.py` 4.

The harness code itself is 4,230 lines in `src/rai_bvar/validation/` (`wc -l`): `reference.py` 955,
`sbc.py` 661, `joint.py` 638, `geweke.py` 445, `dummies.py` 325, `simulate.py` 266, `testfns.py` 256,
`log_joint.py` 214, `uniformity.py` 176, `sequential.py` 150, `identities.py` 92, `__init__.py` 52.

### 3.3 Runtimes

| Item | Time | Machine / status | Source |
| :--- | :--- | :--- | :--- |
| PR fast suite (4,520 IDs), cold XLA cache, `nice -n 10` | **1,003 s (16 min 43 s; `real` 1,009 s)**: 4,507 passed, 12 skipped, 1 xfailed, 444 deselected, 0 failed. Skips: 6 ArviZ live comparisons (ArviZ not installed), 3 OpenBLAS thread-API tests (macOS Accelerate), 3 phase-gating checks in `test_specs_phase2.py`; the xfail is the strict `test_stochvol_parity[stochvol_v_t5]` oracle deviation (§10.1). Slowest call: `test_shard_map_matches_vmap_in_subprocess` 16.4 s. Fact-check re-run (same machine and commit, cold cache, run from a scratch working directory, load average ≈ 6–10): **4,507 passed, 12 skipped, 444 deselected, 1 xfailed, 0 failed in 1,006.90 s (`real` 1,012 s)**, with exactly the same skip reasons (6 × `arviz` not importable, 3 × no OpenBLAS thread API, 3 × "is part of this build" in `test_specs_phase2.py:726`) and the same strict xfail. The "slowest call" is run-dependent (corrected by fact-check): in the re-run the slowest was `test_g_small[reduced_sv_t]` 20.05 s, with `test_shard_map_matches_vmap_in_subprocess` third at 15.31 s | Apple M5 Max; load average rose to ≈ 18 during the run (other jobs), so an upper bound, not comparable to the warm-cache CI budget; reproduced here; outcome re-run by fact-check | command 4 |
| PR tier budget | ≈ 10–15 min with the persistent compilation cache | design target | DESIGN §9.10, L1967 |
| G-small, all 15 PR compositions (16 tests) | **94.6 s** (2.1–15.8 s per composition); fact-check re-run 99.66 s (2.1–16.7 s), identical statistics | reproduced here; re-run by fact-check | command 2, §6.6 |
| G-small per composition | 5–55 s incl. compilation; ten DESIGN rows 144 s cold / 85 s warm / 220 s under load; nine PR rows ≈ 2 min cold | recorded (M1 Pro, 10 cores, other jobs) | `tests/geweke/test_geweke_compositions.py` docstring |
| G-full per composition | 34 s (row 17) to 9 min (rows 12–13) at stage 1; ≈ 1 h for all 17 rows; `structural_sv` 13 min over three Gandy–Scott stages | recorded (M1 Pro) | same docstring |
| SBC per composition | 25–66 s (Constant rows), 92 s (`zero_sv`), 3.5–6 min (`structural_sv`, RW), 11–17 min (t and outlier-t rows); ≈ 2.5 h for 17 rows sequentially | recorded (M1 Pro, two SBC processes at once) | `tests/sbc/test_sbc_compositions.py` docstring |
| Mutation suite (37 checks) | **181.85 s** (0.0–21.5 s per check; slowest M4-G 21.5 s, M9-G 14.6 s, M10-G 13.4 s, M8-S 12.1 s); fact-check re-run 177.40 s wall (sum of JSON `seconds` 176.3 s; slowest M4-G 22.6 s, M9-G 13.5 s, M10-G 13.4 s, M8-S 10.9 s), identical statistics | reproduced here; re-run by fact-check | command 3 |
| Mutation suite | "~10 min with a warm compilation cache" | recorded (machine not stated) | NOTES B.8 |
| Parity PR tier (stochvol + bayesianVARs files) | ≈ 50–55 s; nightly tier ≈ 6 min | recorded (M1 Pro under load) | `tests/parity/SV_PARAMETRIZATION.md` §6 |
| Nightly tier budget | ≤ 3 h | design target | DESIGN §9.10, L1968 |
| Design-stage estimates (NumPy prototypes) | G-small ≈ 5 s, G-full ≈ 5 min, S ≤ 15 min per composition; 1.5–4 min per PR | recorded, superseded by the measured numbers above | N06 §9.3, L696–717 |

---

## 4. Principles of the harness

### 4.1 The two-implementation rule (D-22; DESIGN §9.1, L1706–1711)

- Prior samplers, observable simulators and `log_joint` are written by work packages that write no
  kernel, "from the parametrisations of §6 and the notes, never by kernel authors", and "they never
  import kernel modules (`components/*`, `engines/mcmc/linalg.py`)". Allowed engine imports: state
  containers, `keys.py`, `config.py`, the jnp design builder `engines/mcmc/data.py` (itself tested
  against the NumPy `data/design.py`). Source: DESIGN §9.1; `src/rai_bvar/validation/reference.py`
  docstring L1–9 (also forbids `engines/mcmc/random.py`).
- **Enforced by a test**: `tests/validation/test_log_joint.py::test_references_never_import_kernels`
  (L511–525) imports `validation.joint`, `log_joint` and `testfns` in a subprocess and asserts that no
  `rai_bvar.components.{structure,mean,coef,vol}` module and none of `engines.mcmc.{linalg, random,
  compose, sweep, runner}` is loaded. `JointSpec.step` imports `compose`, `runner`, `sweep` lazily
  (`validation/joint.py` L268–270). Status: verified (committed test, PR tier).
- **Different algebra, not just different authors.** The references are densities on the "natural"
  parameters (prior log-densities plus the likelihood), so a density-ratio identity compares a
  conditional with a joint assembled from a different formula (`reference.py` docstring,
  "Conventions"). For the SOC/SUR dummy rows the reference is written in **covariance form**
  (`m × m` systems, Matheron's rule, Bayes' rule) while the kernels use the **precision form**
  (`q × q` Gram updates) (`src/rai_bvar/validation/dummies.py` docstring L1–4; NOTES D.15).
  The phrase "references in covariance form" in the task brief is accurate for the dummy-row
  reference; for the other components the distinction is "joint density vs conditional formulas".
- **Second transcription of constants.** The OCSN-10/KSC-7 mixture tables are transcribed a second
  time in `validation/reference.py`; `tests/validation/test_mixture_transcription.py` requires
  agreement "on every printed digit" with (i) the tables of note 01 parsed from the Markdown at test
  time, (ii) its derived constants, (iii) the kernel's `components/vol/mixture.py` (docstring L1–7).
- **Exact references where no closed form exists**: for `iw_equivalent` with dummy rows,
  `p(d | λ)` is not inverse-gamma; the reference draws `λ | d` by rejection from the IG base prior
  with an exact envelope `g(λ) = (|S₀|/|S(λ)|)^{½} exp(−½(q(λ) − q∞)) ≤ 1` (acceptance 0.6–1.0 on the
  test rows) and the log prior is exact up to the evidence (`dummies.py` docstring; NOTES D.13,
  status verified: `tests/validation/test_reference_dummies.py`, KS against the quadrature CDF).
- **Registry for later phases**: `register_reference(kind)` mirrors the kernel registry, so a Phase
  3–5 component adds a reference class and immediately gets per-block and per-composition Geweke
  configurations (DESIGN §9.1, L1712–1727; `reference.py` docstring "Registry").
- Why: an earlier proposal composed the blocks' own `prior_sample`/`log_prior` for validation; a
  judge flagged it and the design was changed (errata E9, DESIGN §0.2, L93 (corrected by fact-check: was L92)). Oracles and joint tests
  are complementary: "an oracle shares any modelling convention we copied, while Geweke/SBC share
  only our prior and observables simulators" (N06 §10, L767).

### 4.2 Test the production code path

The harness "drives the **production** `make_sweep(static)` and `prepare`, never a copy" (DESIGN §9.1,
L1732–1733). `JointSpec.step` = `prepare(data, prior)` then one production sweep
(`validation/joint.py` docstring L4–11). Mutants are injected only through production seams:
named kernel internals (`compose(static, overrides={name: fn})`), a kernel slot (M2), the sampler's
prior arrays (M5), the simulator (M3) or a configuration (M16) (`tests/mutation/mutants.py`
docstring L4–7; DESIGN §4.9).

### 4.3 Test configurations in which every term matters (N06 §6, L437–464; DESIGN §9.1–9.2)

- **Proper, tight priors** (`r = Var_prior/Var_post ≤ 5`) so the Geweke SC chain mixes (§6.3).
- **No term vanishes**: `M ≠ 0` so the D3 centring is exercised; `a_ij ~ N(0, 0.7²)`; outlier
  probability `p ~ Beta(2, 18)` so outliers occur at `T = 20`; `ν ∈ {5..12}` so the t path is
  active (N06 pitfall 13, L461; test priors DESIGN §9.2, L1782–1787; `validation/joint.py` docstring
  L13–29).
- **Masks always exercised**: `T_eff` is never a multiple of 32 (20 → `T_pad` 32, 50 → 64), and the
  G-small test asserts `spec.fixed.T_eff % 32 != 0` (DESIGN §9.1, L1731–1732;
  `test_geweke_compositions.py::test_g_small`).
- **Explosive mass bounded**: own-lag mean 0.4, own s.d. `0.10/ℓ`, cross `0.05/ℓ` with
  `a_ij ~ N(0, 0.7²)` gives `P(ρ > 1)` ≤ 1e-6, 7e-5, 2e-4 for `(n, p) = (2,1), (3,1), (3,2)`
  (4·10⁵ draws). The first proposal (mean 0.5, s.d. 0.15/0.075) violated the bound (0.26–1.8 % in
  structural form, because `A⁻¹` amplifies) (N06 §6 item 5, L443–453; cross-check correction 3,
  L906). Status: recorded (N06 cross-check scripts, not committed).
- **R9 recomputation off; hyperparameters do not depend on `y`** (N06 §2.1, L106; DESIGN §9.1).

### 4.4 Failures fail; nothing is dropped

A non-finite test function fails that function (`p = 0`) and is counted in `n_nonfinite`
(`validation/geweke.py` docstring L31–32, L388–389); SBC "failed fits fail the test, never dropped"
and only data-only filtering (`finite(ỹ) ∧ max|ỹ| < 10³`) is allowed, justified by the
conditional-on-`y` uniformity of ranks (DESIGN §9.6, L1866; `validation/sbc.py` docstring L7–9, L24–25 (corrected by fact-check: was L20–22, which is the γ-test rule);
N06 §4.2, L297–299). This mirrors the engine's no-silent-jitter rule (D-18, DESIGN L64). Self-tests:
`tests/validation/test_geweke_selftest.py::test_non_finite_test_functions_fail_with_counts`,
`::test_sampler_failures_in_the_sc_chains_fail_with_counts`, `::test_non_finite_mc_draws_fail_with_counts`.

### 4.5 Deterministic seeds, independent streams

One fixed seed per tier (e.g. `SEED_SMALL = 20261022`, `SEED_FULL = 20261023` in
`tests/geweke/test_geweke_compositions.py`); keys `validation_key(seed, Tier, crc32(name), chain)` so
compositions use disjoint streams and results do not depend on vectorisation block sizes
(`validation/geweke.py` docstring L24–28; self-test `test_results_do_not_depend_on_block_sizes`).
Because seeds are fixed, a false alarm is deterministic for given code; the sequential wrapper (§6.5)
handles marginal results (N06 §3.5, L250).

---

## 5. Tier U: deterministic identities and oracles

### 5.1 Density-ratio identities (Grosse & Duvenaud 2014 eq. 1 [CITE?])

For a block `b` with implemented conditional `q_b(θ_b | θ_−b, y)` and two states differing only in
`θ_b`:

    log q_b(x1) − log q_b(x0) = log p(x1, y) − log p(x0, y),

because `p(θ_b | θ_−b, y) ∝ p(θ, y)`. The joint is the independent `validation.log_joint`. Partially
collapsed and augmented steps use the joint they target: the `s` and `h | s` steps the augmented
mixture joint, all others the non-augmented one. MH steps: the computed log acceptance ratio (before
`min(0, ·)`) must equal `log π(x′) + log q(x | x′) − log π(x) − log q(x′ | x)`, "including every
correction folded into it (exact-mode `w` terms, the single-accept IG-σ² combination, ASIS
Jacobians)". Error metric `|LHS − RHS|/(1 + |RHS|)`, asserted ≤ 1e-8 (`src/rai_bvar/validation/identities.py`
docstring L1–24; DESIGN §9.3 item 2, L1800–1805; N06 §5.1, L386–399).

Committed identity tests (node IDs; each parametrised over compositions where applicable):

- `tests/mcmc/blocks/test_linear_structural.py::test_density_ratio_equation_step`
- `tests/mcmc/blocks/test_recursive.py::test_density_ratio_contemp_step`
- `tests/mcmc/blocks/test_cccm.py::test_density_ratio_identities`
- `tests/mcmc/blocks/test_constant.py::test_density_ratio_identity_against_log_joint`
- `tests/mcmc/blocks/test_transitory.py::test_density_ratio_o_gaussian_and_p`,
  `::test_density_ratio_tau_given_o_nu`, `::test_density_ratio_collapsed_o_and_nu`
- `tests/mcmc/blocks/test_dummies.py::test_density_ratio_equation_step_with_dummies`,
  `::test_density_ratio_coupled_lambda_step_with_dummies`, `::test_density_ratio_cccm_step_with_dummies`
- MH identities in `tests/mcmc/blocks/test_sv_core.py`: `test_exact_h_log_accept_identity`,
  `test_centred_gamma_phi_log_ratio_identity`, `test_sigma2_log_ratio_identity_and_ig_gibbs`,
  `test_nc_phi_log_ratio_identity`, `test_nc_mu_sigma_log_accept_identity`,
  `test_fold_is_measure_preserving_map`, `test_awol_nc_identity_stationary`
- toy self-tests: `tests/validation/test_identities_toy.py` (7 functions, including
  `test_mh_ratio_with_jacobian_matches_definition` and `test_mh_ratio_without_jacobian_is_detected`).

Why it matters (power per unit cost): in the design-stage power study one random pair of states
detected the CCM-2019 step (median deviation 1.1–4.4 nats vs a correct-step maximum of 5e-10), i.e.
"1 evaluation", against `N = 20–50` for block SBC, `≈ 2.5·10⁴` Geweke sweeps and `N ≈ 200–1000`
for prior SBC (N06 §5.1 L399, §7.4 L612–622). Status: recorded (N06 NumPy prototypes; scripts under a
non-committed scratch path, N06 L470 — see open question Q9). In the production mutation suite the U
check of M1 gives 1.70 vs 1.6e-14 on the control (§11, reproduced here).

### 5.2 Dense oracles, closed forms and infrastructure (DESIGN §9.3, L1791–1816)

- Dense oracles with injected noise: `precision_draw` vs dense solve/Cholesky (1e-10); padding
  exactness (1e-12); packed/tiled Gram vs dense `Fᵀ diag(w) F` (1e-13 relative); tridiagonal SV
  sampler vs dense `Q` and `slogdet` (1e-12); CCCM vs the SUR oracle (rtol 1e-10) on 20 random
  problems; A-row gradient/Hessian.
- **A tolerance finding**: comparing the eigen and fixed-factor fast paths with the Cholesky path "to
  1e-12 is not achievable in general: both paths are accurate to O(ε·κ), κ = cond(sKs)", so they are
  compared with a high-precision reference or path-vs-path with `mean 2e-12 + 5εκ(1 + t)` and
  `covariance 2e-12 + 5εκ`, "verified on 2,250 fresh problems" (DESIGN §9.3 item 1, L1793–1796).
  Status: recorded (DESIGN text); the tests live in `tests/mcmc/blocks/test_linear_fastpaths.py`,
  `tests/mcmc/unit/test_linalg.py`.
- Closed forms: mixture mean −1.2704 ± 1e-4; indicator pmf χ² and Bayes consistency; transitory
  conditionals vs brute force; integrated likelihood vs dense MVN; Constant IG with coupling
  (DESIGN §9.3 item 3, L1806–1809).
- Infrastructure: float64 on every leaf; `import rai_bvar` imports no JAX (subprocess);
  `JAX_ENABLE_X64=0` subprocess; typed keys and a key-reuse checker; **bitwise** invariance across
  chunk sizes, burn piece size, thin with aligned sweeps, chain prefixes, `vmap` vs loop, fused vs
  unfused plans; `rtol ≈ 1e-12` across `vmap` batch sizes, platforms and `shard_map` (DESIGN §9.3
  item 4, L1810–1816; NOTES C.9, status verified in CI on Linux); a failure injected at sweep 7 must
  raise `MCMCNumericalError` naming block, equation and iteration 7.

---

## 6. Tier G: Geweke joint-distribution tests

### 6.1 The statistic (`src/rai_bvar/validation/geweke.py` docstring L1–32, `geweke()` L333–427)

Marginal-conditional (MC), iid: `θ^(m) ~ p(θ)`, `y^(m) ~ p(y | θ^(m))`, `m = 1..M₁`. Successive-
conditional (SC): `C` independent chains, **each started at a prior draw and run with no burn-in**:
`y^(m) ~ p(y | θ^(m−1))`, `θ^(m) ~ q(θ | θ^(m−1), y^(m))`. With chain means `ḡ_c`,

    z_g = (ḡ_MC − ḡ_SC) / √(σ̂²_g/M₁ + s²_C/C),   p_g = 2·P(t_{C−1} > |z_g|),

fail iff some `p_g ≤ α/G` (Bonferroni; Holm lists the failing functions). `RNE_g = σ̂²_g/(M_c s²_C)`.
Batch means within chains (`b = ⌊√M_c⌋`) are a non-gating cross-check.

Why this form (the methods argument, N06 §3.1–3.2, L118–179):

- If `θ̃^(0) ~ p(θ)`, `(θ̃^(0), ỹ^(1))` is already a joint draw, so under H0 every SC chain is
  stationary from its first sweep; discarding iterations only lowers power. This answers Talts et
  al.'s (2018) objection that Geweke z-scores need a converged chain [CITE?] (N06 L138).
- The chain means are iid with mean `E_p[g]` whenever the sweep leaves `p(θ | y)` invariant, so the
  **size** needs only invariance, not ergodicity; ergodicity affects only power (N06 pitfall 14,
  L462; cross-check correction 24, L927 (corrected by fact-check: was L928)).
- No spectral/bandwidth choice; it maps onto `vmap(scan(…))`.
- `t_{C−1}` critical values are needed at Bonferroni-sized tails: at `G = 100, α = 1e-3` the normal
  critical value has actual size 1.64e-3 at `C = 200` and 1.11e-3 at `C = 1000` (N06 §3.2 table,
  L162–167; cross-check correction 2, L905). Status: recorded (computation in N06 cross-check).

Self-tests on a conjugate scalar-normal toy (`θ ~ N(μ₀, τ²)`, `T = 4`, `r = 5`, `κ = 0.8`):
`tests/validation/test_geweke_selftest.py::test_correct_toy_holds_the_nominal_level_over_20_seeds`,
`::test_deliberately_wrong_toys_fail`, `::test_rne_matches_the_closed_form` (AR(1) long-run variance),
`::test_statistics_follow_the_documented_keys_and_formulas`, `::test_batch_means_follow_the_documented_formulas`,
`::test_geweke_sequential_reruns_fresh_and_larger` (file docstring L1–5). Status: verified (committed
tests, PR tier).

### 6.2 Test functions and the finite-variance rule (N06 §3.4, L218–243; DESIGN §9.5, L1838–1841)

Geweke's z is asymptotically normal only if `var g(θ, y) < ∞` (Geweke 2004 eq. 2 [CITE?]). Under the
SV test prior, `E[y²] = E[e^h] = ∞` (σ² has a χ² tail and φ has mass near 1): batch means of
`mean_t y²_t` ranged from 2.3 to 5.2·10³ while `mean_t log y²_t` was −1.27 in every batch (N06 L222).
Under `IG(3, 2)`, `var(λ²) = ∞` (needs shape > 4); `h²` under `Beta(·, 2)` is borderline; t errors need
`ν > 2k` for `y^{2k}`. Rules adopted: log-scale or bounded data functions under SV (`mean_t log y²`,
sample correlations, first autocorrelations, `mean_t tanh y`), `log λ` instead of `λ²`, Constant
prior `IG(5, 4)` (N06 §9.1, L670; DESIGN §9.1, L1729–1731 for the rules, §9.2 L1784 for `IG(5, 4)` (corrected by fact-check: the IG(5, 4) value is in §9.2, not §9.1)). The references carry per-function
finite-variance flags (`reference.py` docstring, "Finite-variance flags"; `test_functions_eq` returns
`(value, finite variance)`, DESIGN §9.1, L1726). Status of the heavy-tail evidence: recorded
(N06 cross-check script `heavy_g.py`, not committed). Note for the paper: this is a correction to a
practice the design-stage power study itself used (its E2 G included `λ²` under IG(3, 2); N06
correction 28, L931) and to Geweke's (2004) §5 choice of `Y'Y` eigenvalues, which were valid there
only because that model had Gaussian errors (N06 L220).

Function families: every scalar parameter, latent states at fixed dates, squares where finite,
within-block cross-products (`a_ij·γ_{i,(lag1,j)}`, `μφ`, `φσ`, `p·N_1`, `ν·mean log τ`),
cross-equation products `π_{1,l}π_{2,l}`, the exact log-likelihood, `Σ_t e²_it/λ_it`, and log-scale
data functions (DESIGN §9.5, L1838–1841). Reproduced here: `G` per PR composition ranges from 22
(`zero_sv`) to 134 (`reduced_sv_outliers`) (§6.6); the G-small test asserts `len(spec.names) ≥ 20`.

### 6.3 Design rule for the test prior: `r ≤ 5` (N06 §3.3, L181–216)

For the scalar conjugate model with `r = Var_prior/Var_post`, one SC sweep is an AR(1) with
coefficient `κ = 1 − 1/r`, so `IAT(θ) = 2r − 1` and `RNE ≈ 1/(2r − 1)`; if the inner sampler has
autocorrelation `ρ`, `κ_tot = 1 − (1 − κ)(1 − ρ)` (derived in N06; re-derived in its cross-check).
Empirical check (VAR with known Σ_t): predicted κ 0.71 vs observed 0.71 (`T = 50`, s.d. 0.2), 0.79
vs 0.74, 0.33 vs 0.42 (N06 L203–207). The same analysis explains Gandy & Scott's (2021) inflated
Geweke size [CITE?]: in their `y = θ₁ + θ₂ + ε` example `r ≈ 2000`, so their 3000 sweeps covered less
than one IAT (N06 L183, L378). Design rule: `r ≤ 5` per scalar parameter, achieved with small `T`
(20 per PR, 50 nightly) and tight proper priors; the stochvol default `μ ~ N(0, 100²)` gives
`r` up to ≈ 2000·T and is unusable (N06 L211–215). Status: derivation recorded; "r ≤ 5 holds for φ,
σ, p and ν" is **not** verified (N06 §12 item 4, L786–793) — see the RNE observation in §6.6.

### 6.4 Configurations (DESIGN §9.5 table, L1843–1855; `validation/joint.py`)

| Rows | Composition | n (small / full) | Tier |
| :--- | :--- | :--- | :--- |
| 1–4 | structural Mode I × {Constant("ig"), SV, SV(outliers), SV(t)} | 2 / 3 | PR + nightly |
| 5 | structural Mode II × Constant("iw") | 2 / 3 | PR + nightly |
| 6 | structural × Constant("iw_equivalent") | 3 | nightly |
| 7–10 | reduced × {Constant("ig"), SV, SV(outliers), SV(t)} | 3 | PR + nightly |
| 11 | reduced × Constant("iw_equivalent") | 3 | nightly |
| 12–13 | structural / reduced × SV(outliers, t) | 3 | nightly |
| 14–16 | structural × SV(rw); × SV(sigma2_prior=InvGamma); × SV(h0=B₀) | 2 | nightly |
| 17 | ZeroMean × SV, n = 1 | 1 | PR |
| 18+ | SV rows of 1–16 with `exact=False` | — | nightly, recorded, not gating |
| 201–205 | SOC/SUR dummy rows (WP25) | 2 / 3 | PR + nightly |
| blocks | 7 block-level configurations (`aux_sv`, `gauss_sv_exact`, `transitory_known_lambda`, `constant_transitory`, `equation_only`, `cccm_only`, `contemp_only`) | — | PR (kernel suites) |

`COMPOSITIONS` = rows 1–17 and 201–205 (22 names), `BLOCK_CONFIGS` has 7 entries, the inexact rows
are `{2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16}` (11), all asserted in
`tests/geweke/test_geweke_compositions.py::test_pr_rows_are_the_design_rows`. Sizes: small `n = 2`
structural / 3 reduced, `p = 1`, intercept, `T_eff = 20`, `y_{1−p:0} = 0`; full `n = 3`, `p = 2`,
`T_eff = 50`, fixed stored presample (DESIGN §9.2, L1786–1787).

The approximate (`exact=False`) rows are recorded, not gating: the mixture approximation is a model
change, and only a non-finite test function fails them (`test_g_full_inexact_recorded`). The
design-stage analysis (N06 §8, L628–656) computed per-observation KL(f‖g) = 7.9e-5 for OCSN-10 and
2.7e-3 for KSC-7, and found neither detectable by SBC at `N = 4000` on latent states with known SV
parameters (status recorded; "with unknown parameters it is UNVERIFIED", N06 L26, L653).

### 6.5 The Gandy–Scott sequential wrapper (Gandy & Scott 2021, Algorithm 3, Theorem 3.1 [CITE?])

`src/rai_bvar/validation/sequential.py` docstring L7–22: family level α, `k` stages, multiplier Δ;
`β₁ = α/k`, `γ = β₁^{1/k}`, `β_{i+1} = β_i/γ`; at stage `i` the Bonferroni-adjusted p-value `q_i` of
a **fresh, independent** experiment fails if `q_i ≤ β_i`, passes if `q_i > γ + β_i`, else continues;
`P(reject) ≤ Σ_i γ^{i−1}β_i = kβ₁ = α`. Δ multiplies the size once, after stage 1 (Algorithm 3, line
9); every stage uses new keys (`stage_seed`), never an extension of the previous chains, because
Theorem 3.1 needs independent stages. In `geweke_sequential` the later stages multiply `C` and `M₁`
by Δ, keeping `M_c` (`geweke.py` L430–445).

Settings used (N06 §4.6 table, L371–377; recomputed there):

| Setting | γ | β sequence | expected extra effort under H0 (eq. 1 / eq. 2 bound) |
| :--- | :--- | :--- | :--- |
| per PR: α = 1e-3, k = 3, Δ = 4 | 0.0693 | 3.33e-4, 4.81e-3, 0.0693 | 29.7 % / 29.9 % |
| nightly: α = 1e-2, k = 3, Δ = 4 | 0.149 | 3.33e-3, 0.0223, 0.149 | 68.7 % / 71.6 % |

So a correct sampler is rerun at 4× with probability ≤ γ ≈ 7 % per PR. The wrapper is used by every
statistical tier (G, B, S; `geweke_sequential`, `sbc_sequential`, `block_sbc_sequential`,
`mutants.invariance`). Self-tests: `tests/validation/test_sequential.py` (10 functions),
`test_geweke_selftest.py::test_gandy_scott_around_geweke`. Reproduced here: one PR composition went to
stage 2 (§6.6) — a live example of the wrapper doing its job.

### 6.6 Evidence: the PR-tier G-small run on the snapshot commit (reproduced here, command 2)

16 tests passed in 94.6 s (`test_pr_rows_are_the_design_rows` + 15 compositions). All exact mode,
`C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3`, seed 20261022, Gandy–Scott `k = 3, Δ = 4`.

| Row | Composition | G | t_crit | max \|z\| (argmax) | stage(s), q | min RNE | time |
| ---: | :--- | ---: | ---: | :--- | :--- | ---: | ---: |
| 1 | structural_constant | 43 | 4.25 | 2.77 (`data.acov1_y[1]`) | 1: 0.249 | 0.031 | 2.9 s |
| 2 | structural_sv | 64 | 4.34 | 2.59 (`eq1.core.sigma^2`) | 1: 0.623 | 0.034 | 3.6 s |
| 3 | structural_sv_outliers | 80 | 4.39 | 2.51 (`eq1.struct.a0^2`) | 1: 0.991 | 0.022 | 6.6 s |
| 4 | structural_sv_t | 78 | 4.39 | 2.12 (`eq1.trans.nu_le_5`) | 1: 1.0 | 0.025 | 10.8 s |
| 5 | structural_mode2_constant | 38 | 4.22 | 2.13 (`eq1.mean.delta0^2`) | 1: 1.0 | 0.033 | 2.1 s |
| 7 | reduced_constant | 80 | 4.39 | 3.03 (`glob.mean.Pi[3,0]`) | 1: 0.198 | 0.022 | 3.2 s |
| 8 | reduced_sv | 110 | 4.46 | 2.74 (`glob.mean.Pi[3,1]^2`) | 1: 0.697 | 0.021 | 5.1 s |
| 9 | reduced_sv_outliers | 134 | 4.50 | 2.98 (`eq0.core.phi^2`) | 1: 0.394 | 0.024 | 8.9 s |
| 10 | reduced_sv_t | 131 | 4.50 | 2.61 (`data.corr_y[0,1]`) | 1: 1.0 | 0.020 | 15.8 s |
| 17 | zero_sv | 22 | 4.10 | 2.34 (`data.mean_tanh_y[0]`) | 1: 0.427 | 0.008 | 2.5 s |
| 201 | structural_constant_dummies | 45 | 4.26 → 4.25 | 4.21 (`data.acf1_log_y2[1]`) → 1.94 (`eq0.mean.delta1`) | 1: 0.0012 → 2: 1.0 | 0.103 | 7.5 s |
| 202 | structural_mode2_constant_dummies | 40 | 4.23 | 2.26 (`eq0.core.log_lam`) | 1: 0.953 | 0.129 | 2.3 s |
| 203 | reduced_constant_dummies | 83 | 4.40 | 2.61 (`glob.mean.Pi[1,2]^2`) | 1: 0.753 | 0.055 | 2.9 s |
| 204 | structural_sv_dummies | 66 | 4.35 | 2.65 (`eq0.mean.delta1^2`) | 1: 0.543 | 0.020 | 3.9 s |
| 205 | structural_iw_equivalent_dummies | 72 | 4.37 | 2.24 (`eq2.mean.delta1^2`) | 1: 1.0 | 0.099 | 15.5 s |

(Times are the pytest call durations; the JSON records were written to the scratchpad by
`RAI_BVAR_VALIDATION_RECORD`.)

Observations:

- Largest `|z|` over all passing compositions at the final stage: 3.03, against `t_crit` 4.10–4.50.
- **Row 201 shows the sequential design at work.** Stage 1 gave a Bonferroni q of 0.0012 (`|z| = 4.21`
  on a log-scale data autocorrelation), between `β₁ = 3.33e-4` and `γ + β₁ ≈ 0.070`, so a fresh run at
  `C = 4000, M₁ = 8·10⁵` was made; it passed with q = 1.0. The committed mutation evidence records the
  same control behaviour ("stage 2 after a stage-1 q 0.0012", `mutants.py` EVIDENCE["M17"], L854–856).
  (Corrected by fact-check: this is the *same* computation, not an independent cross-check — the
  mutation suite's G controls are these PR runs, `SEED_G = 20261022  # the G-small seed of
  tests/geweke: the unmutated controls are those PR runs`, `mutants.py` L106 — and it is not a
  reproduction "across machines": the EVIDENCE string was recorded on an Apple M5 Max, as were the
  dossier's run and the fact-check re-run; its "x86-64 alike" gives no x86 numbers, and the explicit
  x86-64 numbers in the comment block, control 2.44 with t_crit 4.26, L831–833, predate the eigen path.
  What is supported: the stage-1 q 0.0012 → stage-2 pass recurs deterministically for fixed code and
  seed on Apple M5 Max.)
- Row 205 max |z| 2.24 equals NOTES D.13's "G-small max |z| 2.24" (status there: verified).
- **Minimum RNE is 0.008–0.129** across compositions. (Corrected by fact-check: not "far below" the
  `RNE ≈ 1/(2r − 1) ≥ 1/9 ≈ 0.11` that `r ≤ 5` would give for the θ chain in every case — in 12 of 15
  compositions the minimum is 0.008–0.055, well below it, but the three Constant dummy rows 201, 202,
  205 have 0.103, 0.129, 0.099, i.e. at or above it.) Under the parallel-chain design this does not
  affect the test's size (§6.1) but it reduces power for those functions. The JSON holds per-stage
  `min_rne` only; the fact-check extracted the lowest-RNE functions for four compositions (see Q2),
  and they are **not** φ or σ: intercepts (`eq0.mean.delta0`, `glob.mean.Pi[0,·]`) and
  `data.mean_tanh_y` in `structural_constant`, `reduced_constant` and `structural_sv`, and the
  `h`-level functions in `zero_sv`. So this is evidence on the RNE of those functions, not direct
  evidence on N06's open item "r ≤ 5 for φ, σ, p, ν is not verified" (N06 §12 item 4) (corrected by
  fact-check).

### 6.7 G-full, inexact rows and one numerical finding

- No committed record of a G-full or `exact=False` run exists (`find . -name '*.jsonl'` returns
  nothing; the tests write `geweke.jsonl` only when `RAI_BVAR_VALIDATION_RECORD` is set). The only
  committed G-full statistics are NOTES D.13 (row 205: G-full 2.76) and NOTES D.14 ("G-full on the
  eigen path"). Status of "G-full green for every composition": **to check** (open question Q1).
- **Known G-full finding (NOTES C.10).** `test_g_full[reduced_sv]` failed on one SC chain of 4000 at
  sweep 194, which had reached the prior's extreme tail (`h₀ ≈ 30–36`, `φ ≈ 0.999`, `σ ≈ 0.9`;
  `P(max h > 25) ≈ 1e-6` per prior draw) where `|y| ≈ 10⁸` made equation 2's contemporaneous precision
  singular in float64 (Cholesky pivot 0; status 1003). The engine reported it (no jitter, D-18). It
  was resolved in the kernel, not by narrowing the test prior: the `a`-row draw is now the Householder
  QR form of the whitened rows, which never forms the Gram (`cond(K_a) = 1.3e16`, exact eigenvalues 14.8
  and 1.9e17) (`tests/geweke/test_geweke_compositions.py` docstring; NOTES C.10). The replay is
  `tests/mcmc/blocks/test_contemp_lstsq.py::test_g_full_reduced_sv_failing_chain_is_fixed`. Status
  (corrected by fact-check: was "verified (NOTES)" for the whole bullet): the failure itself is
  **recorded** (NOTES C.10, first part; test docstring), the QR resolution is **verified** (NOTES C.10,
  second part); the replay test is `@pytest.mark.slow` (L155), so it runs in the weekly CI job, not
  per PR. Methods point: the Geweke harness, which explores the whole prior, found a
  numerical weakness that ordinary data never would, and the fix was in the sampler.

### 6.8 Design-stage power evidence for the Geweke tier (recorded; NumPy prototypes)

From N06 §7.3 (L552–571): against the CCM-2019 triangular step, `C = 1000` chains with `M_c = 25`
(2.5·10⁴ SC sweeps) gave max |z| 15.3–41.9 even for weak coupling (`A × 0.25`), against a Bonferroni
critical value of 3.87 (G = 92), while exact and CTA samplers never exceeded 3.76. With `Π, A, λ`
unknown (E2, `C = 2000`, `G = 190`, crit 4.04), TA-2019 gave 30.5 at `M_c = 50`. N06 concludes that
for this error class Geweke is about 2× cheaper than prior SBC in raw draws and about 25× in MCMC
sweeps, with a much larger margin (corrected by fact-check: not a verbatim quote; the wording is that
of cross-check correction 18, N06 L921, "2× in raw draws, about 25× in MCMC sweeps, plus a large z
margin"; §7.4 itself, L622, opens with "about 20× cheaper than prior SBC in MCMC sweeps" before
deriving 2× and about 25× — an internal inconsistency of N06 to resolve before quoting). Status: recorded; single seed per configuration;
scripts on a non-committed path (N06 §12 item 8, L797; N06 L470).

---

## 7. Tier S: simulation-based calibration

### 7.1 Algorithm (`src/rai_bvar/validation/sbc.py` docstring L1–64; DESIGN §9.6, L1860–1873)

Per replication: (1) `θ̃ ~ p(θ)`, `ỹ ~ p(y | θ̃)`, data-only filter; (2) **pilot**: a chain started at an
independent prior draw (never at `θ̃`, Gandy & Scott §2.2 [CITE?]), `burn` sweeps then `pilot`
recorded sweeps; `IAT_i = max_f` of the Geyer initial-monotone IAT [CITE?] of `f` and of the tail
indicators `1{f ≤ q̂_0.05}`, `1{f ≤ q̂_0.95}`; thinning `t_i = min(⌈2·IAT_i⌉, t_max)`; (3)
**production**: a fresh chain, `burn` sweeps, then `L` draws every `t_i` sweeps (`thin` traced;
replications sorted by `t_i` before vectorised blocking); (4) rank with ties
`n_less + ⌊u(n_tie + 1)⌋`, one `u` per quantity. The thinning decision depends only on `(ỹ, pilot
noise)`, so uniformity is preserved; a single global thinning factor is not enough (N06 §4.2, L284–296).

Configuration in the tests: `SBCConfig(N=1000, L=99, burn=300, pilot=1000, t_max=200, alpha=1e-2)`,
22 compositions at the "full" size (`n = 3`, `p = 2`, `T_eff = 50`; rows 14–16 `n = 2`; row 17 `n = 1`),
inside the Gandy–Scott wrapper (`tests/sbc/test_sbc_compositions.py` docstring L1–11).

### 7.2 Quantities, including the predictive unknowns

Parameters, `h` at three dates and `mean h`, **total and per-equation log-likelihood**, `log p(θ)`,
`‖Γ − AM′‖²`, the transitory-marginal log-likelihood, `N_1`, `Σ log τ`, and the predictive unknowns
`h_{T+1}`, `y_{T+1}` through the production forecast core (DESIGN §9.6, L1866–1868). The SBC test
excludes data functions, squares and cross-products
(`tests/sbc/test_sbc_compositions.py::test_quantity_selection`). The predictive truth (references)
and the predictive draws (production `simulate_paths`) are the same map at the same inputs to 1e-12
(`::test_truth_and_forecast_core_are_the_same_map`, slow). Rationale: Modrák et al.'s (2025)
Theorem 7 — a "posterior" equal to the prior passes SBC for every data-free quantity — makes
data-dependent quantities necessary [CITE?] (N06 §4.4, L336–351); ranking a held-out
`log p(y_{T+1} | θ)` would be invalid, ranking the predictive draw `y_{T+1}` is valid (N06 L349–351).

### 7.3 Uniformity: γ statistic with exact p-values (`src/rai_bvar/validation/uniformity.py`)

`γ = 2·min_i min{F_Bin(R_i; N, z_i), 1 − F_Bin(R_i − 1; N, z_i)}` with `z_i = i/(L+1)` (Säilynoja,
Bürkner & Vehtari 2022 eq. 13 = Modrák et al. 2025 eq. 7 [CITE?]). Discrete uniform ranks on `{0..L}`
make `(R_1..R_L)` a Markov chain with binomial increments, so the null law is computed **exactly** by
the forward recursion (Säilynoja et al. eqs 15–23). Implementation detail worth a sentence in the
paper: the p-value is accumulated as "the mass leaving the interior at each step, which keeps
far-tail p-values (Bonferroni at α/Q) accurate to relative precision instead of computing
1 − P(stay)", and `γ_obs` is computed from the same cached envelope table so it is exactly an atom of
the null (`uniformity.py` docstring L18–30, `gamma_pvalue` L115–157). Why exact rather than
simulated: Bonferroni at `α = 1e-3` over 20–40 quantities needs per-quantity tails near 3e-5–5e-5,
where a 10⁵-draw null table holds 3–5 draws; the first simulated table in N06 was off by up to 20 % at
q0.001 (N06 §4.3, L312–318; cross-check correction 4, L907). In the design-stage experiments γ was
more powerful than χ² with 20 bins wherever either had power (e.g. 0.77 vs 0.32 at `N = 500`; N06 D-6,
L24, Table L523–524; recorded). Self-tests: `tests/validation/test_uniformity.py` (11 functions,
including `test_exact_recursion_agrees_with_simulation[1000]` and
`test_null_pvalues_are_super_uniform`).

### 7.4 The ESS rule, its erratum, and the `t_max` deviation (NOTES B.5, B.6)

- **Rule** (DESIGN §9.6): per-replication thinned ESS ≥ 0.8 L for ≥ 99 % of replications.
- **Erratum (2026-10-22)** (DESIGN L1870–1873; `sbc.py` docstring L24–54). `L = 99` kept draws cannot
  certify 0.8 L for one replication: "with iid draws the Geyer ESS of the kept draws reaches 0.8L in
  only ~76% of replications for one quantity, so gating on it would fail every correct sampler". The
  per-replication ESS is therefore *predicted* from the pilot IAT with an AR(1) form,
  `ρ_i = (IAT_i − 1)/(IAT_i + 1)`, `ESS_i = L(1 − ρ_i^{t_i})/(1 + ρ_i^{t_i})`; uncapped,
  `t_i = ⌈2·IAT_i⌉` gives `ESS_i ≥ 0.96 L`, so only replications at the cap can fail
  (`ESS_i ≥ 0.8 L` iff `ρ_i^{t_i} ≤ 1/9`, i.e. at `t_max = 50` iff `IAT_i ≤ 45.5`). That rule cannot see
  a pilot that underestimates the IAT, so a **second gate** requires the replication-pooled kept-draw
  ESS (`ess_pooled`: Geyer IAT of the pooled autocovariances of ECDF indicators at each replication's
  own quantiles 0.05/0.25/0.5/0.75/0.95, Kish-weighted effective number of replications) ≥ 0.8 L for
  every quantity; "with iid draws, n = 40 and L = 99, the minimum over 2·10⁴ simulations was 0.836L".
  The pooled indicator form avoids the pitfall that the pooled lag-1 autocorrelation of `f` itself is
  dominated by a few large-variance replications (0.73 for a calibrated sampler in N06 E2; N06 L594,
  L610). Status: NOTES B.5 says verified (design change); the 76 % and 0.836 L figures are
  **recorded** in docstrings, with no committed test reproducing them (open question Q5).
- **`t_max = 200` deviation (NOTES B.6).** With `t_max = 50` the ESS rule failed while the γ test was
  clean: `reduced_sv_t` (min p 0.0032 over 80 quantities) had 12 of 1000 replications with pilot
  IATs of 45–135; `structural_sv_outliers_t` (min p 0.0084 over 102) had 69 above 45.5, 6 above 91,
  none above 182 ("outliers absorb variance, so the SV level of a few datasets mixes slowly").
  `reduced_sv_t` passes at 100 (99.9 %); 200 covers every replication observed
  (`tests/sbc/test_sbc_compositions.py` docstring L13–21). Status: recorded.
- **Why per-replication thinning** (design-stage, recorded): with a single global thinning of 10, a
  *correct* CTA sampler failed the log-likelihood γ test 11.5 % of the time at `N = 2000` and 26.5 % at
  `N = 4000`, driven by a heavy right tail of per-dataset IATs (median 1.9, max 433); thinning 40
  restored nominal size (N06 §7.3, L584–610; D-7, L25).

### 7.5 What SBC sees that parameter ranks miss (NOTES B.4; N06 §7.3)

Design-stage (recorded): against the CCM-2019 error in the prior-dominated regime (`τ = 0.25`),
parameter-rank power stayed at the nominal 1 % at `N = 2000` while the log-likelihood reached 1.00 at
`N = 500`; per-dataset mean shifts of up to 1.5 posterior s.d. cancel when averaged over datasets
(the situation of Modrák et al.'s Theorem 5) (N06 L533–541). The reason is derived in N06 §7.2
(L498–515): CCM-2019's factorisation is **exact under a flat prior** (the missing term is constant in
`π^(1)`) and wrong for every proper prior independent across reduced-form equations whenever some
`a_ij ≠ 0`, so the error is a prior effect, invisible when data dominate (flat prior: max mean shift
0.015 posterior s.d.; tight prior: 1.49; `check_flat.py`, re-run in the N06 cross-check with identical
output). Production evidence: the block SBC of the `π_0` draw catches M1 through the log-likelihood
rank (p 2.8e-11) with control min p 0.10, while one-block invariance of the whole CCCM scan does not
(q 0.29) (§11; reproduced here for the block SBC). NOTES B.4: "block SBC catch verified; the
whole-scan 'not caught' observation is recorded, not re-run".

### 7.6 Committed SBC evidence

No committed SBC result file exists (the test writes `sbc.jsonl` only when recording is enabled).
Committed statistics: NOTES D.13 (row 205: SBC min p 0.018); the failing `t_max = 50` runs above;
the mutation suite's M8 S check (§11). Status of "S green for every composition": **to check** (Q1).
Self-tests: `tests/validation/test_sbc_selftest.py` (25 functions).

---

## 8. Tier B: block-conditional SBC and one-block invariance

- **Block-conditional SBC** (N06 §5.2, L401–411; `sbc.block_sbc`): `(θ̃, ỹ)` from the joint, `L = 99`
  iid draws of the block from its exact conditional given `θ̃_−b`, ranks of the block's coordinates
  and of the exact log-likelihood; `N = 200`, Bonferroni α = 1e-3, Gandy–Scott. Exact because the live
  block runs before every frozen one. Composition-level coverage (`tests/sbc/test_block_tier.py`
  docstring L4–18): the fused structural `(δ_i, a_i)` step in every structural composition (all
  vol blocks, Mode II, `iw_equivalent`, fast paths), the reduced contemporaneous step, and each CCCM
  column `π_j` (column 0 from one call of `cccm_core`, columns `j ≥ 1` from the exposed exact
  conditional).
- **One-block invariance** (N06 §5.3, L413–418): `N = 1000` joint draws, `L = 20` applications of the
  live blocks with `ỹ` fixed, two-sample KS per statistic against `N` independent joint draws;
  Bonferroni α = 1e-3 inside Gandy–Scott. Used for MH, ASIS and Gibbs-scan blocks: the CCCM scan; the
  volatility block of `zero_sv` and its RW, InvGamma, fixed-`B₀`, outlier, t and outlier-t variants,
  and Constant with outlier-t (`test_block_tier.py` docstring L19–28).
- Design-stage power (recorded): block SBC detected the CCM-2019 `π^(1)` step at `N = 20` with power
  0.995 (`a ~ N(0,1)`, constant vol), 0.985 (SV-like), 0.45 (`a ~ N(0, 0.5²)`), while the correct step
  held 0.3–1.8 % (N06 §7.3 table L546–550).
- Cadence: 43 composition-level B tests, all `slow` (weekly in CI); the per-PR B tier lives in kernel
  suites (e.g. `tests/mcmc/blocks/test_sv_core.py::test_centred_step_one_block_invariance`,
  `::test_nc_phi_move_one_block_invariance`).

---

## 9. Tier A: analytic twins and exact oracles (A1–A7)

The point (SPEC §10.3, L510–516; the quoted sentence is L515 (corrected by fact-check: was L511–516)): Geweke and SBC run at `n ≤ 3`; an analytic twin is "an end-to-end
oracle for the MCMC engine at realistic sizes (`n = 20+`), where Geweke tests and SBC are too
expensive".

| Test | Oracle | Comparison and tolerance | Cadence | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Chan (2022) structural NIG log ML [CITE?] = Phase 1 `AnalyticFit.log_ml()` (Mode II, `κ_c = κ_a = 1`, `d = n + 2`), with and without SOC/SUR; EB λ equal at `cross = 1`; WP25 extension: MCMC Mode II × Constant("iw") with dummy rows vs the Phase 1 NIW posterior with the same rows, negative control without rows | 1e-10 (log ML); 1e-6 (EB λ); 4 quantile-MCSE (extension) | PR (`n = 3`) | DESIGN §9.8 L1904; `tests/analytic/test_structural_nig.py::test_a1_log_ml_equals_phase1`, `::test_a1_eb_equals_phase1_eb_at_cross_one`, `::test_a1_holds_when_the_level_dwarfs_the_noise`; `tests/analytic/test_mcmc_twin.py::test_a1_dummies_*` |
| A2 | MCMC structural Mode II × Constant("iw") vs Phase 1 NIW posterior (`B`, `Σ`, `A` quantiles), `n = 3, p = 2, T = 60` | 4 quantile-MCSE, Holm | PR (`fixed_factor` path); other paths nightly | `test_mcmc_twin.py::test_a2_mode2_iw_matches_the_phase1_niw_posterior`, `::test_a2_rejects_a_slightly_wrong_posterior`, `::test_a2_other_constant_weight_paths` |
| A3 | as A2 at `n = 20, p = 4, T = 300` | same; ~10⁴ quantiles at family level 1e-2 | nightly | `::test_a3_realistic_size` |
| A4 | Mode II with `κ_c = 0.5` (Chan's asymmetric conjugate prior, not NIW) vs closed form `t_{2ν̂}`, `IG(ν̂, Ŝ)` and iid draws | same | nightly | `::test_a4_chan_asymmetric_prior[n3, n20]` |
| A5 | Mode I × Constant: per-equation 1-D quadrature of `p(λ_i | y) ∝ IG·N(y_i; F_iμ₀, λI + F_iV₀F_iᵀ)`, exact `E[λ_i]`, `E[θ_i]`, `Var(θ_i)`; exact at any `n` because equations are independent and `|A| = 1` | `|z| ≤ max(4, z_Bonf)` with mean MCSE; quadrature self-checks at 1e-10 | PR (`n = 3`, `eigen` path), nightly (`n = 20`, other paths, `iw_equivalent` extension) | `tests/analytic/test_mode1_quadrature.py` docstring; `::test_a5_mode1_constant_matches_quadrature`, `::test_a5_rejects_a_slightly_wrong_posterior`, `::test_quadrature_reproduces_the_structural_nig_closed_form`, `::test_oracle_prior_is_the_engine_prior` |
| A6 | toy posteriors by 2-D quadrature / SIR: exact-mode invariance of the `h` step, ASIS correction, step order, T13 | moments stay at exact values `|z| < 4.5`; wrong kernels must reach `max |z| > 8` | PR | `tests/mcmc/blocks/test_sv_toy_quadrature.py` docstring |
| A7 | reduced × Constant: exact system-wide joint draw (O((nk)³), NumPy test code) vs CTA Gibbs, `n ∈ {2, 3, 4}` | MCSE | nightly (`test_a7_swa_vs_cta[n,p]`, 6 `slow` IDs), plus a PR-sized check `tests/mcmc/blocks/test_cccm.py::test_a7_swa_vs_cta_small` (n = 3, p = 1, T = 60, 64 chains; not `slow`) (corrected by fact-check: cadence was "nightly" only) | DESIGN §9.8 L1910; `tests/mcmc/blocks/swa_oracle.py` docstring |

Methods details worth reporting:

- **CDF-form quantile comparison** (`tests/analytic/test_mcmc_twin.py` docstring "Comparison"): each
  exact quantile `x_q` is tested as `z = (F̂_MCMC(x_q) − q)/√(MCSE²(F̂) + q(1 − q)/N_ref)`, with the
  indicator's own ESS (never the bulk ESS of `θ`). To first order it equals the quantile-MCSE test,
  but it avoids the density estimate in the order-statistic MCSE, whose noise made
  `(q̂ − x_q)/MCSE_q` heavy-tailed: with ~8,000 quantiles at `n = 20` it gave 2–3 values beyond 4
  (0.5 expected) and one at −4.99 (seed 221), whereas the CDF form matched `N(0, 1)` (variance
  0.91–1.05, no excess kurtosis) on the same draws. Decision: every `|z| ≤ max(4, z_Bonf)` at family
  level 1e-2. Status: recorded (docstring); the comparison itself is verified by the committed tests.
- **Every twin has a negative control with measured power**: A2 rejects one extra degree of freedom of
  the NIW posterior (~1.5 % on `Σ`); A5 rejects an IG shape off by ½ (~1.6 % shift of `E[λ_i]`)
  (docstrings of `test_mcmc_twin.py` and `test_mode1_quadrature.py`). Status: verified (committed tests
  `::test_a2_rejects_a_slightly_wrong_posterior`, `::test_a5_rejects_a_slightly_wrong_posterior`).
- **The quadrature oracle checks itself**: "doubling the rule and widening the interval changes
  nothing above 1e-10 (asserted on every call)"; with a λ-scaled prior on all coordinates it must
  reproduce Chan's closed form to rounding (`test_mode1_quadrature.py` docstring).
- **A twin caught a bug the block oracles did not** (NOTES D.12): in conjugate Mode II structural form
  the λ step must count the dummy rows (`IG(c + (T + q_i + m)/2, …)`); omitting it is not caught by the
  dense oracles of the coefficient step, but the MCMC-vs-Phase-1 twin catches it at once (33 of 165
  quantiles outside 4 MCSE, `n = 3, p = 2, T = 60`). The omission was injected by hand, not as a
  registered mutant. Status: recorded.
- **A6 and the coarse-table device**: chains start at draws from the exact posterior of a tiny model;
  "exact mode is exact for any table, so a poor table only removes the approximation's cover and gives
  the tests power" (`test_sv_toy_quadrature.py` docstring). The pilot's scratch Geweke test of the SV
  block (16 chains × 2·10⁵ sweeps, Holm at 1 %, reject iff max |z| > 3.34) shows why: with OCSN-10 all
  of exact, `exact=False`, stale `w` and M7 pass (2.07–2.88); with a one-component table exact mode
  passes (2.51) and `exact=False` (10.05), stale `w` (12.46) and M7 (5.31) fail — "an OCSN-10 Geweke
  test cannot validate the exact correction or catch errors in it" (`docs/phase2/09_pilot.md` §3.4,
  L187–211; status recorded, scratch prototype; reproduced independently by a review harness per the
  same section).

---

## 10. Tier P: parity against independent software

### 10.1 stochvol (Phase 2; `tests/parity/test_stochvol.py`, `tests/parity/SV_PARAMETRIZATION.md`)

- **Version**: R 4.1.2, **stochvol 3.2.1** (DESIGN §9.9 names 3.2.9; "3.2.1 is the installed version")
  (SV_PARAMETRIZATION.md L13–14).
- **Setup**: 4 chains × 50,000 draws after 5,000 burn-in on five committed datasets ((i) `T = 300,
  μ = −1, φ = 0.95, σ = 0.2`; (ii) `φ = 0.5, σ = 0.5`; (iii) `φ = 0.99, σ = 0.1`; (iv) standardised
  AR(4) residuals of real-GDP growth, `T = 235`; (v) (i) with standardized `t_5` errors); our side is
  the production `fit_mcmc` on `ZeroMean × SV`, `n = 1`. One parameter set per run, both sides derived
  from it (erratum E11); the test parses the fixture's spec attribute with a literal evaluator and
  requires it to equal the derived spec (§2, L47–67).
- **Statistic and tolerance**: `z = (q̂_ours − q̂_R)/√(mcse_ours² + mcse_R²)` over 5/25/50/75/95 %
  quantiles of `μ, φ, σ`, `h_t` at `t ∈ {0, 1, T/4, T/2, T}` (and `τ_t` at the three largest `|y_t|`
  for SV-t), with `mcse = max(Vehtari et al. quantile MCSE, batch-quantile MCSE)`; pass `|z| ≤ max(4,
  z_{1−0.01/(2m)})` — exactly 4 for every stochvol fixture (`m ≤ 157`); split R-hat < 1.05 guard
  (L73–81).
- **Results** (§6, L241–257; recorded, M1 Pro, jax 0.12):

| fixture | m | bound | PR max \|z\| (4 × 6,000) | nightly max \|z\| | fold (PR / nightly) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| stochvol_i_main | 40 | 4.00 | 1.86 | 3.57 | 0 / 0 |
| stochvol_i_defaults | 40 | 4.00 | 1.66 | 2.12 | 0.01 % / 0.03 % |
| stochvol_ii_main | 40 | 4.00 | 1.63 | 2.53 | 0.17 % / 0.12 % |
| stochvol_ii_defaults | 40 | 4.00 | 1.95 | 2.75 | 0.06 % / 0.04 % |
| stochvol_iii_main | 40 | 4.00 | 2.34 | 2.92 | 1.03 % / 0.72 % |
| stochvol_iii_defaults | 40 | 4.00 | 2.59 | 2.59 | 1.22 % / 1.07 % |
| stochvol_iv_main | 40 | 4.00 | 1.96 | 1.76 | 0 / 0 |
| stochvol_iv_defaults | 40 | 4.00 | 1.79 | 1.64 | 0 / 0 |
| stochvol_i_exact (`exact=True`) | 40 | 4.00 | nightly only | 2.14 | — / 0.01 % |
| stochvol_iv_exact (`exact=True`) | 40 | 4.00 | nightly only | 2.56 | — / 0 |
| stochvol_v_t5 (strict xfail) | 55 | 4.00 | **28.9** | **62.5** | 0.72 % / 0.65 % |
| SV-t vs exact importance sampling (4 obs.) | 45 | 4.00 | 1.83 | — | 26 % |

- **Finding 1 (NOTES A.1, D-7).** stochvol's non-centred (ASIS) step takes `|σ|` without flipping
  `h̃`; ours folds `(σ, h̃) ↦ (−σ, −h̃)` (DESIGN D-7, L53). On a four-observation problem
  (`y = (0.4, −2.5, 0.3, 0.6)`), with the exact posterior from self-normalised importance sampling
  (4·10⁶ prior draws), the 5 % quantile of `h_2` is −0.663 exact, −0.663/−0.665 ours, −0.659 stochvol
  centred-only, −0.843 stochvol non-centred-only, −0.837 stochvol default (SV_PARAMETRIZATION.md §3.1,
  L90–107). On the `T = 300` fixtures the effect is not visible within MCSE ("Every comparison passes
  without invoking the D-7 exclusion", L259–261). Status: recorded (stochvol 3.2.1); to check against
  the current release; contact maintainers first (NOTES "Before submission").
- **Finding 2 (NOTES A.2).** stochvol 3.2.1's Student-t SV sampler misses the standardized-t posterior
  that its own simulator uses (`svsim(…, nu = 5)` checked: mean τ 0.997): on the four-observation `t_5`
  problem the exact `μ` quantiles are −1.495, −1.042, −0.732, −0.425, 0.017; ours −1.495, −1.044,
  −0.734, −0.426, 0.018 (`exact=False`); stochvol default −1.549, −1.069, −0.743, −0.423, 0.029; `τ_2`
  95 % quantile 9.171 exact, 9.115 ours, 7.681 stochvol; the deviation persists without the
  non-centred step and is not explained by the unit-scale convention (§3.2, L109–130). The fixture row
  is a strict `xfail` narrowed to `OracleDeviation`, guarded by R-hat, finite-z and DGP-recovery checks
  (posterior ranks of the truth 0.58, 0.55, 0.75; coverage 0.85 for `h`, 0.92 for `τ`); the exit
  criterion is replaced by `test_sv_t_matches_the_exact_importance_sampling_posterior` (L132–151). The
  unblock path (regenerate under ≥ 3.2.9, rerun the four-observation check) is written down
  (L158–170). Status: recorded for 3.2.1 only; **to check** before any public claim; report upstream
  first (NOTES A.2).
- **Methodological point (independent of who is right)**: when two implementations disagree, a tiny
  problem with an exact importance-sampling posterior decides between them; this is the same "oracle
  written by someone else" logic as the twins.

### 10.2 bayesianVARs (Phase 2): blocked, replaced by an exact system-wide oracle

- bayesianVARs 0.1.8 could not be built (no CRAN macOS binary for R 4.1; source build needs gfortran);
  probes P1–P5 (variance vs s.d., `U` vs `U⁻¹` and sign, intercept row, `sv_para` layout, the
  homoscedastic prior's scale) are all **unresolved**; `fixtures/probe_bayesianvars.nc` records
  `bayesianvars_available = 0` (SV_PARAMETRIZATION.md §4, L172–186).
- **Replacement oracle**: an exact system-wide Gibbs sampler in R (`vec Π | A, Λ` jointly from its
  Gaussian conditional; `A` rows by regression; each equation's SV by one sweep of stochvol's fast
  sampler at offset 0, or `λ_i | e ~ IG`), in rai-bvar's own conventions so P1–P5 do not arise; it
  self-checks (dense-precision identity 1e-16, `A`-row gradient 2e-11, single-step stochvol chain vs
  `svsample` `|z| < 2.2`) (L198–206).
- Results (§6 table, recorded): `bvars_swa_n3p2_const` m = 195, bound 4.05, PR max |z| 2.76
  (4 × 3,000), nightly 2.75; `bvars_swa_n3p2_sv` m = 300, bound 4.15, nightly 3.45; `bvars_swa_n7p4_sv`
  m = 1,540, bound 4.51, nightly 3.56 (fold up to 7.1 % on series 7). The SPEC exit criterion
  "bayesianVARs parity for the reduced-form sampler" is therefore met against a substitute oracle, not
  bayesianVARs — the paper must say so.

### 10.3 Phase 1 parity (analytic engine)

- **R `BVAR` 1.0.5** (Kuschnig & Vashold [CITE?]; `tests/parity/test_r_bvar.py`, 147 IDs): tolerances
  log ML at fixed hyperparameters 1e-8 relative (BVAR and exact agree to < 1e-11), NIW posterior
  parameters 1e-8 relative to the largest entry (BVAR's own normal equations deviate by up to 6e-9),
  hyperparameter modes 1e-3 relative, Monte Carlo quantiles within `4·√(MCSE_R² + MCSE_ours²)`; at
  `h = 1` both sides are also checked against the exact multivariate t predictive
  (`tests/parity/BVAR_PARAMETRIZATION.md` §12, L279–301; `test_r_bvar.py` L33–40).
  Fixtures are generated by `tools/make_r_fixtures.R` (≈ 6 min, 319 internal consistency checks).
- **Finding (third-party; not in NOTES §A)**: BVAR 1.0.5's `predict.bvar` scales the forecast noise by
  `Σ` instead of a Cholesky factor (covariance `Σ²`) and does not propagate shocks through the
  dynamics; on `qd_small_levels` its 90 % band at `h = 8` is about 3 times too narrow; the fixtures
  therefore simulate forecasts from BVAR's posterior draws with the correct recursion
  (BVAR_PARAMETRIZATION.md §9, L228–240). Further documented deviations from GLP (2015) [CITE?]:
  ψ units, ψ defaults, ȳ₀ for SOC/SUR, the dummy-only marginal likelihood (exact only for a
  random-walk prior mean), optimiser, MH proposal (§10, L242–257). Status: **to check** against the
  current BVAR release before any public statement (same rule as NOTES "Before submission").
- **Bank of England `bvar` 0.3.5** (Python, MIT; `tests/parity/test_boe_bvar.py`, 38 IDs): log ML at
  fixed h rtol 1e-8 (measured ≤ 1.5e-11; "a 1e-6 relative change in ψ already moves the log ML by
  2.6e-8, so the check has power"); posterior moments; draw means within 5 MCSE; hyperparameter modes
  within 1e-3; forecast quantiles within MC tolerance. Margins to watch: B̄ on level data 2.9e-9 against
  1e-8; δ mode 6.5e-4 against 1e-3 (from the engine's L-BFGS-B stopping rule) (`tests/parity/BOE_BVAR_NOTES.md`
  §14, L286–315). Documented docs-vs-code discrepancies of the BoE package in §11 (L235–247).
  Status: recorded.

---

## 11. Tier M: the mutation suite (M1–M18)

### 11.1 Rules (`tests/mutation/mutants.py` docstring L1–45; `tests/mutation/test_mutants.py`)

- A mutant is **caught by a tier** when the tier's decision rule fails on the mutant **and passes on
  its control** (the same configuration without the mutation, common seeds); U checks: mutant error
  above and control error below the tolerance on the same joint draw (`run_tier`, L878–908).
- `test_registry` (per PR, no JAX work) requires M1–M18 to be registered with at least one confirmed
  catching tier that has a check; `test_mutant_caught` runs every confirmed (mutant, tier) pair
  (weekly). DESIGN §9.7: "a mutant caught by none blocks the release" (L1898).
- Tier checks: **U** density ratio or closed form at a joint draw; **B** block SBC of `π_0`
  (`N = 200, L = 99`) or one-block invariance of the vol block (`N = 1000, L = 20`, KS); **G**
  `geweke_sequential` at G-small; **S** `sbc_sequential` with a short configuration (`N = 200` (×4
  after stage 1), `L = 99`, burn 100, pilot 300, `t_max = 30`) (docstring L9–25; `sbc_check` L714–726).

### 11.2 The table (paper Table candidate)

Injection seams and compositions from `_ALL` (`mutants.py` L767–826); "recorded" statistics are the
committed `EVIDENCE` strings (L834–858: mutant / control; observed 2026-10-22 on an M1 Pro, re-run on
x86-64 Linux (4 CPUs) the same day; M17–M18 on Apple M5 Max and x86-64, L827–833). "Reproduced here"
is command 3 (M5 Max, commit 62d15f6, 2026-10-23): **37 of 37 checks caught, every control passed,
181.85 s**; every statistic below equals the recorded string to the digits shown except where noted.

| ID | What is injected | Seam | Composition | DESIGN expected | Confirmed catchers | Statistics (mutant / control) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | CCM-2019 triangular step instead of CTA (`π_j` from equation `j` alone) | `cccm_core` | reduced_constant | U, B, G | U, B, G | U 1.70 / 1.6e-14; B (π₀ block SBC) log-lik p 2.8e-11 / min p 0.10; G 20.8 / 3.03; whole-scan invariance q 0.29 (**not caught**, recorded) |
| M2 | stale indicators: Primiceri (2005) Alg. 1, `s` carried across sweeps and drawn after `h` [CITE?] | `vol` slot (`StaleIndicatorSV`), `exact=False` | structural_sv | G / divergence; to confirm | G | G 702 (log-lik) / 2.48 recorded (M1 Pro); **705 / 2.48 reproduced here**; x86-64 705 / 2.47 (recorded, L830) |
| M3 | simulator draws fresh latents (Geweke's "error 2" [CITE?]) | `JointSpec(fresh_latents=True)` | structural_sv | G | G | G 23.5 (mean \|h\|) / 2.59 |
| M4 | τ conditional with shape ν/2 instead of (ν+1)/2 | `tau_conditional` | structural_sv_t | U, G | U, G | U 0.163 / 5e-15; G 281 / 2.12 |
| M5 | `B_σ × 2` in the sampler only (prior mismatch) | sampler prior arrays | structural_sv | G | G | G 50.3 (log p(θ)) / 2.59 |
| M6 | NC `\|σ\|` without flipping `h̃` (stochvol's behaviour, D-7) | `nc_fold` | zero_sv | G | U, G | U 1.94 / 0; G 7.4 / 2.34 |
| M7 | uncorrected NC move in exact mode | `nc_accept` (+ `COARSE1` table in mutant and control) | zero_sv | G; to confirm | U, G | U 2.87 / 0; G with COARSE1 77.6 / 2.28 (OCSN-10: correction too small to see) |
| M8 | D3 centring dropped (`γ_i = δ_i`) | `structural_gamma` | structural_constant | G, S | U, G, S | U 0.585 / 0; G 101 / 2.77; S q 1.6e-13 / control passes at stage 3 (q 0.0069, 0.065, 0.95) |
| M9 | outlier pmf without `1/k` | `outlier_logits` | structural_sv_outliers | U, G | U, G | U 0.699 / 1e-15; G 1474 / 2.51 |
| M10 | `b_p + T` instead of `b_p + T − N₁` in the `p` update | `p_conditional` | structural_sv_outliers | U, G | U, G | U 0.120 / 1e-14; G 30.1 / 2.51 |
| M11 | indicator pmf without `σ_j⁻¹` | `indicator_logits` | aux_sv (`exact=False` sampler) | U, B | U, B, G | U 0.112 / 6e-16; B q 4.9e-19 / 0.93; G 86.9 / 2.18; exact mode (zero_sv): G passes 2.39 (**not a bug there**, recorded) |
| M12 | effective means shifted by −1.2704 twice | `mixture_table` | aux_sv | U, B | U, B, G | U 1.2704 / 8.3e-5 (tol 1e-4); B q 2.5e-162 / 0.93; G 444 / 2.18; exact mode: G passes 2.25 (not a bug there, recorded) |
| M13 | IG shape with `T_pad` instead of `T_eff` | `constant_conditional` | structural_constant | G | U, G | U 1.64 / 3e-15; G 429 (log-lik) / 2.77 |
| M14 | A-row regression on `+u_{<j}` instead of `−u_{<j}` | `contemp_conditional` | reduced_constant | U, G | U, G | U 14.9 / 9e-15; G 325 / 3.03 |
| M15 | ASIS map without the `h̃` rescale | `nc_step` | zero_sv | B, G | U, B, G | U 0.457 / 6e-17; B q 1.8e-14 / 1.0; G 50.5 / 2.34 |
| M16 | offset `c = 0.1` at `λ ≈ 0.01` (μ prior centred at log 0.01), `exact=False` | configuration | zero_sv | G (exact=False only) | G | G 264 / 1.99 (control offset 1e-12) |
| M17 | Mode I SOC/SUR rows weighted by the current `1/λ_i` instead of the fixed `1/σ̄_i²` (WP25) | `linear_conditional` | structural_constant_dummies | U, G | U, G | U 0.060 / 2.4e-15; G 23.1 / 1.94 (control at stage 2 after stage-1 q 0.0012) |
| M18 | SOC/SUR rows dropped from the CCCM `Π` step (WP25) | `cccm_core` | reduced_constant_dummies | U, G | U, G | U 0.405 / 2.9e-14; G 99.7 / 2.61 |

Tolerances of the U checks: 1e-8 except M8 and M11 (1e-12), M12 (1e-4), M15 (1e-10) (`_ALL`).
G critical values in the reproduced run: 4.10 (zero_sv, aux_sv) to 4.40.

Coverage summary (from `catchers` in `_ALL`, reproduced here): U catches 14 of 18 (all but M2, M3, M5,
M16), G catches 18 of 18 (M7 with the coarse table, M11/M12 under the auxiliary sampler), B 4 of 4
where a B check exists (M1, M11, M12, M15), S 1 (M8). 14 + 18 + 4 + 1 = 37 checks.

Per-check reproduced times: U 0.0–2.0 s, B 1.2–4.3 s, G 4.3–21.5 s, S 12.1 s (command 3; JSON
`seconds` field).

### 11.3 What the mutation suite taught (paper-worthy observations)

1. **The four mutants without a U check are the non-local errors** (corrected by fact-check: was
   "the four mutants U cannot see are exactly the non-local errors"): a wrong step *order* (M2), a
   wrong *simulator* (M3), a prior *mismatch* between sampler and model (M5), and a *model
   approximation* (M16). The registry gives them `u_check=None` (`mutants.py` `_ALL`: M2 L771–773, M3 L774–776, M5
   L780–782, M16 L815–818), so U is not *run* on them; that U *cannot* see them is not tested (M5 in particular is a
   sampler-side prior change that a density-ratio identity against the reference prior might detect;
   see "Unsupported claims"). Every conditional-formula error (M1, M4, M6–M15, M17–M18) is caught by a
   deterministic identity at one joint draw. This is the empirical case for running both identity
   tests and joint-distribution tests (status: derived from the reproduced table).
2. **Exactness changes what counts as a bug (NOTES B.3).** Under `exact=True` the correction uses
   `log g` of the same (mutated) logits and table, "so the augmented chain stays exact for any
   indicator law and any table" (N01 §10.2, L432–447); M11/M12 are bugs of the auxiliary-mixture
   sampler only, which the B and G checks run on (`aux_sv`, where that sampler is exact). Likewise
   M7 is invisible with OCSN-10 and caught with the one-component `COARSE1` table (mutants.py
   docstring L32–44). NOTES B.3 status: verified (re-run on x86-64 Linux, 2026-10-22); reproduced here
   on M5 Max for the catching checks (the exact-mode "not a bug" G passes 2.39/2.25 are recorded only).
3. **The offset mutant only exists in approximate mode**: under `exact=True` a large offset is not a
   bug (exact for any `c`), so M16 is defined with `exact=False` (DESIGN §9.7 L1894; errata E17, L101).
4. **Parameter ranks vs log-likelihood (NOTES B.4)**: M1 is caught by the block SBC through the
   log-likelihood rank; whole-scan one-block invariance with marginals of `Π` and the log-likelihood
   does not see it at `N = 1000` (mutants.py docstring L19–21).
5. **Geweke's "error 2" reproduced (NOTES B.7)**: M3 gives max |z| 23.5 vs control 2.59 (reproduced
   here).
6. **Wrong dummy weights are loud (NOTES D.15)**: M17 23.1 / 1.94, M18 99.7 / 2.61 (NOTES: "3–6 s per
   G-small run"; here 5.8–10.6 s per mutant-plus-control G check); the independent x86 review "also caught four further injected bugs (Mode II rows weighted
   1/σ̄², rows counted twice in the λ step, CCCM rows weighted by `(Σ⁻¹)_jj`, fixed-factor weight 2w)"
   (NOTES D.15; recorded, not registered mutants).
7. **Design expectations were conservative**: the DESIGN §9.7 column expected U for 9 mutants (M1,
   M4, M9–M12, M14, M17, M18); the suite confirmed U for 14 (M6, M7, M8, M13, M15 gained U catchers
   via closed-form or density-ratio checks) and G for all 18 (M11, M12 were expected only from U/B).

### 11.4 Cross-platform evidence

- Recorded: "Re-run independently on x86-64 Linux (4 CPUs) the same day: all 33 mutant × tier checks
  caught, every control passed, the statistics equal to the digits shown (M2: G 705 / 2.47)";
  M17–M18 recorded on Apple M5 Max and "reproduced independently on x86-64 Linux the same day
  (`-k "M17 or M18"`: 4 passed)" (`mutants.py` L827–833; NOTES B.3, B.8, D.15).
- Reproduced here: Apple M5 Max (arm64), 37/37 caught, statistics equal to EVIDENCE to the digits
  shown, M2 705 / 2.48 (vs M1 Pro 702 / 2.48 and x86-64 705 / 2.47).
- Interpretation: statistics agree across three machines to two–three significant figures (M2: 702
  vs 705 for the mutant, 2.47 vs 2.48 for the control) (corrected by fact-check: was "about three
  significant figures"); they are not expected to be bitwise equal across platforms (OpenBLAS vs
  Accelerate; NOTES C.9). Note that M17–M18 have explicit numbers only from Apple M5 Max (§6.6).

### 11.5 Inconsistencies to fix before the paper table

- NOTES B.8 says "33/33 mutant × tier checks"; with M17–M18 the suite has **37** checks
  (`-m mutation` collects 37; reproduced here 37/37). The 33 refers to M1–M16.
- The comment block above `EVIDENCE` quotes M17 "G max |z| 23.15 / 2.44 (t_crit 4.26)" and M18 "U
  0.405 / 2.35e-14", while `EVIDENCE` and this run give control 1.94 (dense eigen path; 2.44 on
  `cholesky_const`) and U control 2.9e-14 (`mutants.py` L831–833 vs L854–857). The EVIDENCE strings
  are current.
- The module docstring says the evidence was "observed on 2026-10-22, M1 Pro"; M17–M18 were recorded
  on an Apple M5 Max (L827–833).
- NOTES B.8 timing "~10 min with a warm compilation cache" vs 182 s cold here (machine difference;
  state the machine).

---

## 12. Runtime convergence diagnostics and the `converged` gate

### 12.1 Monitored sets (D-33; DESIGN §8.1, L1642–1650)

- **Primary** (decides `converged`): the exact log-likelihood given latents; per equation the own
  first-lag reduced-form coefficient; Constant `log λ_i`; AR SV `μ_i, φ_i, σ_i, h_{i,T}`; RW SV `σ_i,
  h_{i,T}`. **Secondary** (reported, warning only): every `a_ij`, `h_{i,1}`, `h_{i,⌈T/2⌉}`, intercepts,
  `log p_i`, `ν_i` with its escape probability, log prior.
- Sizes at `n = 20, p = 4` asserted by `tests/mcmc/diagnostics/test_monitor.py::test_monitored_set_sizes_at_n20`:
  primary 41 (Constant), 101 (SV), 61 (RW SV), 101 (SV outlier-t); secondary 211, 251, 251, 291.
- Rationale: an earlier proposal gated on ~330 quantities including every `a_ij`; the split keeps the
  gating multiplicity near 100 (errata E3, L87 (corrected by fact-check: was L86); D-33). ν and `p` are never gating (E16); acceptance
  rates never gate (E21).

### 12.2 The rule (DESIGN §8.2, L1652–1668; `src/rai_bvar/engines/mcmc/monitor.py` L24–25, `converged` L458–518; `src/rai_bvar/settings.py` L87–88)

    converged = (no chain raised an error) ∧ (chains ≥ 2)
              ∧ ∀ q ∈ primary: rank-normalised split R-hat (max of bulk and folded) < rhat_max (1.01)
              ∧ ∀ q ∈ primary: bulk ESS > ess_min (400)
              ∧ (hierarchy == "eb" ⇒ EB optimisation converged)

Tail ESS is reported, not gated. Single-chain runs are never converged ("split R-hat cannot see
between-chain disagreement"). The flag is copied into forecast attrs, so the backtest score and
`compare()`'s "≥ 95 % of origins" rule work unchanged (DESIGN L1662–1665). Defaults
`MCMCSettings(chains=4, draws=2000, burn=1000, rhat_max=1.01, ess_min=400.0)` (`settings.py` L75–88),
matching SPEC §10.6 ("4 chains, R-hat < 1.01, bulk ESS > 400", L544–547). Committed behaviour tests:
`test_monitor.py::test_one_chain_is_never_converged`, `::test_an_error_is_never_converged`,
`::test_eb_failure_is_never_converged`, `::test_low_primary_ess_is_not_converged`,
`::test_secondary_only_failure_converges_with_a_warning`, `::test_stuck_or_constant_primary_is_not_converged`,
`::test_thresholds_come_from_the_settings`.

### 12.3 Estimators (Vehtari et al. 2021 [CITE?]; `src/rai_bvar/diagnostics.py`)

Rank normalisation with `(r − 0.375)/(S + 0.25)` normal scores (`_rank_normalise`, L23–25), split
chains, R-hat as the max of the bulk and folded versions (`rhat`, L37), ESS with Geyer's initial
positive and monotone sequence truncation (`_geyer_ess`, L209); batched `rhat_many` L333. Phase 2 adds tail ESS, mean ESS,
MCSE of means and quantiles (indicator ESS, Vehtari et al. §4.3–4.4) and batched `*_many` versions
(D-24; DESIGN §8.4, L1689–1695). Cost (recorded): ≈ 1 s CPU for flag, table and warnings at `n = 20`
SV-O-t (392 quantities, 4 × 2000 draws), ≈ 7 s at 4 × 10,000 (DESIGN L1665–1667).

### 12.4 How the diagnostics are tested

- Reference values from ArviZ 0.23.4 and `arviz-stats` 1.3.3, which "agree to the last digit on every
  case", hard-coded (ArviZ is not a dependency), tolerance rtol 1e-10; known-answer tests on stationary
  Gaussian AR(1) chains with closed-form IATs, including the indicator IAT via Plackett's identity;
  statistical bounds calibrated over 30 seed sets, ≥ 3.5 sampling s.d. from the observed means
  (`tests/mcmc/diagnostics/test_diagnostics_ext.py` docstring L1–17; `::test_matches_arviz`,
  `::test_ar1_known_autocorrelation_times`, `::test_mcse_coverage_over_many_replications`,
  `::test_tail_ess_sees_what_bulk_ess_misses`). Status: verified (committed tests).

### 12.5 What the gate does in practice (an honest result for the paper)

- **Predicted by the pilot** (`docs/phase2/09_pilot.md` §3.3, L166–185; recorded): at the defaults
  (4 chains, burn 1000, draws 2000), max split-R̂ over 8 datasets was 1.004–1.007 for μ, 1.004–1.011
  for `h_T`, 1.018–1.029 for the log-likelihood, and **1.032–1.062 for φ, 1.033–1.045 for σ**; bulk ESS of
  φ and σ ≈ 180–220 (plain split-R̂, not rank-normalised). "So at the defaults … SV fits will
  routinely report `converged = False` through φ and σ"; options (more draws, move φ, σ to secondary,
  gate SV parameters on ESS only) were "escalated, not decided". As of the snapshot φ and σ are still
  primary (D-33; `test_monitored_set_sizes_at_n20`: 101 = 20 × 5 + 1 for SV).
- **Observed in the committed Clark–Ravazzolo report** (`benchmarks/reports/clark_ravazzolo.md`,
  generated 2026-10-23 04:19 UTC; 4 chains × 5000 kept draws, burn 1000; **to be replaced by the
  acceptance run in progress**, so: recorded, provisional). SV arms converged at 37–68 % of the 106
  full-sample origins (SV(rw) 68 %, SV 37 %, SV(rw,t10) 57 %, AR-SV(0.9) 62 %; Constant 100 %; L34–118),
  with typical failures "1 of 13 [or 21] primary quantities fail … (worst: sigma[UR] R-hat 1.015, ESS
  618)" (L388–398). Criterion 6 fails **only because of the gate**: `compare()` verdict SV(rw) vs
  Constant is 'better' in 0/24 orderings with the gate on and 24/24 with it off, with SV converged at
  57–82 % across orderings (criterion row L14; table L337–362). The gate is doing what it was designed to do (refuse a
  verdict when < 95 % of origins converged, SPEC §9.5), but at these run lengths it is binding for
  SV models on real data; whether to lengthen runs, change thresholds or move φ, σ is an open design
  question (Q6).

---

## 13. NOTES section B (validation methodology) with statuses

| NOTES item | Claim (short) | Evidence | NOTES status | This dossier |
| :--- | :--- | :--- | :--- | :--- |
| B.3 | Exact-mode SV is exact for any mixture table and indicator law, so M11/M12 are bugs only of the auxiliary sampler; M7 needs a coarse one-component table | `tests/mutation/mutants.py` docstring + EVIDENCE | verified (x86-64 re-run 2026-10-22) | catches reproduced here (M5 Max); exact-mode "not a bug" passes recorded only |
| B.4 | Parameter ranks are blind to the CCM-2019 error; the log-likelihood rank is not | `mutants.py` (M1), N06 §7.3 | block SBC catch verified (p 2.8e-11 vs control 0.10); whole-scan "not caught" recorded | block SBC reproduced here; whole-scan still recorded |
| B.5 | Per-replication SBC ESS cannot be certified from L = 99; predicted ESS + pooled ESS gate | DESIGN §9.6 erratum; `sbc.py` docstring | verified (design change) | the 76 % / 0.836 L figures are recorded only (Q5) |
| B.6 | SBC thinning cap: `t_max = 50` failed on slow datasets with γ clean; 200 covers all | `tests/sbc/test_sbc_compositions.py` docstring | recorded | unchanged |
| B.7 | Geweke "error 2" (fresh latents) caught: M3 max \|z\| 23.5 vs 2.59 | mutants.py EVIDENCE | verified | reproduced here |
| B.8 | Every mutant caught by some tier; mutant × tier table | `tests/mutation` | verified, "33/33 … ~10 min" | 37/37 reproduced here in 182 s; NOTES count stale |
| B.8b | Path-conditioned RB log score breaks down in the tails at h > 1 (UR h = 8 at 2008Q2: −93.6 with 1000 draws vs −19.1 exact path-integrated; quick CR UR h = 8 ΔLPS inflated ≈ 8×) | WP23 review; `benchmarks/clark_ravazzolo.py` RELIABILITY_NOTE; CR report L28 | recorded (re-run with the library fix before publishing) | the fix is not in commit 62d15f6; belongs to the forecasting dossier but is a validation lesson: the delta-method MCSE stayed ≈ 1 while the estimate was off by tens of nats |

Related items from other sections that the validation chapter should cite: C.9 (bitwise
reproducibility limits, verified in CI on Linux), C.10 (Geweke chain in the prior's tail exposed a
float64 Cholesky failure; fixed by QR; verified), D.12 (twin caught a missing dummy term; recorded),
D.13 (exact rejection reference for `λ | d`; verified), D.14 (eigen fast path validated by G-small over
7 seeds and G-full; verified), D.15 (covariance vs precision form; M17/M18; verified on two platforms),
A.1/A.2 (stochvol findings; recorded, to check).

---

## 14. Suggested structure for the paper's validation section

1. Principle (R6) and the CCM-2019 motivation; why "passes" must be quantified by a mutation suite.
2. Tier map (Table: §2.1 with the as-implemented cadence of §2.2).
3. Independence of the truth: the two-implementation rule, enforcement test, covariance/precision
   algebra, second transcriptions; production seams.
4. Deterministic layer: identities (one evaluation detects M1-class errors).
5. Joint layer: Geweke design (parallel prior-started chains, `t_{C−1}`, `r ≤ 5`, finite variance,
   Gandy–Scott) with the G-small table (§6.6).
6. SBC design (two-pass thinning, exact γ, ESS gates and erratum, predictive quantities).
7. Oracles: analytic twins at `n = 20` (CDF-form quantile test; negative controls) and parity (stochvol
   table; the bayesianVARs substitute; Phase 1 parity).
8. Power: the mutation table (§11.2) and its lessons (§11.3); cross-platform reproduction.
9. Runtime diagnostics and what the gate does on real data (§12.5).
10. Costs (§3.3) and limitations (open questions below).

---

## 15. Open questions

- **Q1. No committed record of G-full, `exact=False`, SBC or composition-level B runs.** The paper
  table for "Geweke and SBC green for every composition" (SPEC §13 exit criterion) needs a run on the
  release commit with `RAI_BVAR_VALIDATION_RECORD` and the JSON committed (NOTES "Before submission";
  NOTES B.8). Today only PR-tier G-small (reproduced here) and the mutation suite are backed by fresh
  evidence.
- **Q2. Low Geweke RNE.** Minimum RNE 0.008–0.129 across PR compositions (§6.6) vs the `r ≤ 5` design
  target (RNE ≳ 0.11). Which functions? (Partly answered by fact-check: a script calling
  `geweke_sequential(joint_spec(name, "small"), GewekeConfig(seed=20261022, C=1000, M_c=100,
  M1=200_000, alpha=1e-3))` and sorting stage-1 `rne`, run on commit 62d15f6, reproduced the §6.6
  max |z| and min RNE and gave the five lowest-RNE functions: `structural_constant` — `eq0.mean.delta0`
  0.031, `data.mean_tanh_y[0]` 0.031, `data.mean_tanh_y[1]` 0.033, `eq1.mean.delta0` 0.041,
  `eq1.struct.a0` 0.048; `reduced_constant` — `glob.mean.Pi[0,0]` 0.022, `data.mean_tanh_y[0]` 0.022,
  `glob.mean.Pi[0,1]` 0.024, `data.mean_tanh_y[1]` 0.026, `glob.mean.Pi[0,2]` 0.029; `structural_sv` —
  `data.mean_tanh_y[0]` 0.034, `eq0.mean.delta0` 0.035, `data.mean_tanh_y[1]` 0.035, `eq1.struct.a0`
  0.039, `eq1.joint.gamma_lag1_0` 0.040; `zero_sv` — `eq0.core.mean_abs_h` 0.008, `eq0.core.mean_h`
  0.014, `loglik` 0.017, `eq0.core.hmid` 0.019, `eq0.core.h1` 0.020. So the guess "likely SV φ, σ" is
  not borne out: the slowest functions are intercepts, a data function and, in `zero_sv`, the `h`
  levels; the other 11 compositions were not examined.) Does it cost power against realistic
  mutants? N06 §12 item 4 ("r ≤ 5 for φ, σ, p, ν not verified") is still open.
- **Q3. CI cadence.** The "nightly" job runs weekly (ci.yml). Either report "weekly" or change CI; the
  SPEC §10.1 table says Geweke full "nightly" and SBC "nightly".
- **Q4. Third-party claims.** stochvol 3.2.1 → 3.2.9+ recheck for D-7 and SV-t (unblock path in
  SV_PARAMETRIZATION.md §3.2); bayesianVARs parity unblock (gfortran build, probes P1–P5); BVAR 1.0.5
  forecast-noise claim vs the current release; contact maintainers before publication (NOTES).
- **Q5. Unreproduced numbers in docstrings.** The SBC "~76 %" and "0.836 L" (sbc.py), the "2,250 fresh
  problems" tolerance study (DESIGN §9.3), the A3 "−4.99 (seed 221)" order-statistic MCSE observation
  (test_mcmc_twin.py): add small committed scripts or tests if they go in the paper.
- **Q6. The `converged` gate vs SV mixing.** With φ and σ primary at R-hat 1.01, SV fits on real data
  fail the gate at 4 × 5000 draws at 32–63 % of full-sample origins per SV arm, and SV(rw) at 18–43 %
  of origins across the 24 orderings (CR report; provisional). Lengthen default SV
  runs, gate SV parameters on ESS only, or move φ, σ to secondary? The decision changes criterion 6 of
  the replication. Re-check against the improved acceptance run when it lands.
- **Q7. NOTES B.8 / mutants.py text drift** (33 vs 37 checks; M17 control 2.44 vs 1.94; machine labels)
  — fix in the source before quoting.
- **Q8. Approximation effect with unknown SV parameters** (N06 §8, §12 item 5) — the `exact=False`
  G-full rows are "recorded, not gating", but no record is committed, so the paper cannot yet say
  whether the OCSN-10 approximation is visible at G-full sizes.
- **Q9. Design-stage power study not committed.** N06's experiment scripts (E1, E1b, E2, E3,
  `check_flat.py`, cross-check `xc06_check/*`) live under scratch paths (N06 L470, L844), not in the
  repository. If the paper cites the CCM-2019 power tables (§5.1, §6.8, §7.5, §8), either commit the
  scripts (e.g. `benchmarks/validation_power/`) or present the production mutation evidence instead.
- **Q10. Gandy–Scott rank test** (exact, needs reversibility) on the PCG-ordered SV sweep or a
  palindromic scan: not used, applicability unverified (N06 §12 item 7).
- **Q11. Release-commit re-run.** All "reproduced here" numbers are on commit 62d15f6 and one machine;
  re-run on the release commit and cite its hash (NOTES "Before submission").
- **Q12. B.8b fix.** The path-integrated LPD estimator is not in this snapshot; whether its validation
  (e.g. SBC on `y_{T+h}` for h > 1, or an exact-mixture oracle) belongs in this chapter is undecided.

## 16. [CITE?] references mentioned in this dossier (to be verified by the bibliography stage)

Most of these have bibliographic details and DOIs recorded in `docs/phase2/06_validation_geweke_sbc.md`
§13 (L805–824) and `SPEC.md` §15 (L750–811); they are listed here as the repository gives them, not
re-verified.

- Geweke, J. (2004). Getting it right: Joint distribution tests of posterior simulators. *JASA*
  99(467), 799–804. [CITE?]
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., Gelman, A. (2018). Validating Bayesian
  inference algorithms with simulation-based calibration. arXiv:1804.06788. [CITE?]
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejsková, K., Gelman, A., Vehtari,
  A. (2025). Simulation-based calibration checking for Bayesian computation: The choice of test
  quantities shapes sensitivity. *Bayesian Analysis* 20(2), 461–488. [CITE?]
- Säilynoja, T., Bürkner, P.-C., Vehtari, A. (2022). Graphical test for discrete uniformity and its
  applications in goodness-of-fit evaluation and multiple sample comparison. *Statistics and
  Computing* 32, article 32. [CITE?]
- Säilynoja, T., Schmitt, M., Bürkner, P.-C., Vehtari, A. (2025). Posterior SBC: Simulation-based
  calibration checking conditional on data. arXiv:2502.03279. [CITE?]
- Gandy, A., Scott, J. A. (2021; arXiv v2). Unit testing for MCMC and other Monte Carlo methods.
  arXiv:2001.06465. [CITE?]
- Grosse, R. B., Duvenaud, D. K. (2014). Testing MCMC code. arXiv:1412.5218. [CITE?]
- Cook, S. R., Gelman, A., Rubin, D. B. (2006). Validation of software for Bayesian models using
  posterior quantiles. *JCGS* 15(3), 675–692; and Gelman (2017) correction. [CITE?]
- Besag, J., Clifford, P. (1989). Generalized Monte Carlo significance tests. *Biometrika* 76(4),
  633–642. [CITE?]
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021). Rank-normalization,
  folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*
  16(2), 667–718. [CITE?]
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4), 473–483. [CITE?]
- Flegal, J. M., Jones, G. L. (2010). Batch means and spectral variance estimators in Markov chain
  Monte Carlo. *Annals of Statistics* 38(2), 1034–1070. [CITE?]
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics*. [CITE?]
- Kish, L. (1965). *Survey Sampling* (effective sample size of weighted data). [CITE?]
- Carriero, A., Clark, T. E., Marcellino, M. (2019). Large Bayesian vector autoregressions with
  stochastic volatility and non-conjugate priors. *Journal of Econometrics*. [CITE?]
- Carriero, A., Chan, J., Clark, T. E., Marcellino, M. (2022). Corrigendum to "Large Bayesian VARs with
  stochastic volatility and non-conjugate priors". *Journal of Econometrics* 227(2), 506–512. [CITE?]
- Bognanni, M. (2022). Comment on "Large Bayesian VARs with stochastic volatility and non-conjugate
  priors". *Journal of Econometrics* 227(2), 498–505. [CITE?]
- Chan, J. C. C. (2022). Asymmetric conjugate priors for large Bayesian VARs. *Quantitative
  Economics* (as best I know). [CITE?]
- Kim, S., Shephard, N., Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison
  with ARCH models. *Review of Economic Studies*. [CITE?]
- Omori, Y., Chib, S., Shephard, N., Nakajima, J. (2007). Stochastic volatility with leverage: Fast and
  efficient likelihood inference. *Journal of Econometrics* 140(2), 425–449. [CITE?]
- Kastner, G., Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for
  boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data
  Analysis*. [CITE?]
- Del Negro, M., Primiceri, G. E. (2015). Time varying structural vector autoregressions and monetary
  policy: A corrigendum. *Review of Economic Studies*. [CITE?]
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Review
  of Economic Studies* 72(3), 821–852. [CITE?]
- Stroud, J. R., Müller, P., Polson, N. G. (2003). Nonlinear state-space models with state-dependent
  variances. *JASA* (cited in N01 §10.2 for the exact correction). [CITE?]
- Clark, T. E., Ravazzolo, F. (2015). Macroeconomic forecasting performance under alternative
  specifications of time-varying volatility. *Journal of Applied Econometrics*. [CITE?]
- Giannone, D., Lenza, M., Primiceri, G. E. (2015). Prior selection for vector autoregressions.
  *Review of Economics and Statistics*. [CITE?]
- Kuschnig, N., Vashold, L. (2021). BVAR: Bayesian vector autoregressions with hierarchical prior
  selection in R. *Journal of Statistical Software*. [CITE?]
- Hosszejni, D., Kastner, G. (2021). Modeling univariate and multivariate stochastic volatility in R
  with stochvol and factorstochvol. *Journal of Statistical Software* (the stochvol reference; the
  repository cites only the CRAN manual). [CITE?]
- bayesianVARs (R package, Gruber & Kastner) — software citation to be determined. [CITE?]
- Bank of England `bvar` (Python package, v0.3.5) — software citation to be determined. [CITE?]
- ArviZ (Kumar, Carroll, Hartikainen, Martin, 2019, *JOSS*) — used only as a source of reference
  values. [CITE?]
- Matheron's rule for Gaussian conditioning (used in `validation/dummies.py`); a standard reference
  (e.g. Journel & Huijbregts 1978, or Wilson et al. 2020) is to be chosen. [CITE?]

## 17. Unsupported claims (moved here by fact-check)

Fact-check of 2026-10-23 against commit 62d15f6. Literature citations were not checked (another
stage). Every other claim and number was compared with the cited file, test, report or command; the
corrections are marked inline "(corrected by fact-check)". The claims below could not be supported
as originally written and should not go into the paper in that form.

1. **"The four mutants U cannot see are exactly the non-local errors" (old §11.3 item 1).** The
   evidence shows only that M2, M3, M5 and M16 have no registered U check (`u_check=None` in
   `tests/mutation/mutants.py` `_ALL`), so U is never run on them. That U *cannot* detect them is
   not tested; for M5 (`B_σ × 2` in the sampler's prior arrays only) a density-ratio or MH-ratio
   identity of the σ² step against `validation.log_joint`, which uses the model prior, is a
   plausible detector. Supported wording: "the four mutants without a U check".
2. **"A deterministic reproduction across machines" of the row-201 / M17-control stage-2 behaviour
   (old §6.6).** The G controls of the mutation suite *are* the PR G-small runs (`SEED_G`,
   `mutants.py` L106), so EVIDENCE["M17"] is the same computation, and its explicit numbers come
   from an Apple M5 Max ("x86-64 alike" gives no x86 figures; the x86-64 numbers in the comment
   block, control 2.44 / t_crit 4.26, predate the eigen path). Supported: deterministic recurrence on
   Apple M5 Max for fixed code and seed.
3. **"Minimum RNE … far below the 0.11 that r ≤ 5 would give" and "direct evidence on N06's open
   item r ≤ 5 for φ, σ, p, ν" (old §6.6), and the guess "likely SV φ, σ" (old Q2).** Rows 201, 202
   and 205 have minimum RNE 0.099–0.129, near or above 0.11; and in the four compositions examined
   by the fact-check the lowest-RNE functions are intercepts, `data.mean_tanh_y` and (`zero_sv`)
   `h`-level functions, not φ or σ (Q2). The low RNE is real; its attribution to φ, σ is not
   supported.

Items that remain **reproduced-here only** (not unsupported, but with no committed source; state
the machine and commit if used, or commit the records — Q1, Q11): the dossier's own G-small and
mutation timings (94.6 s, 181.85 s; the fact-check re-run gave 99.66 s and 177.40 s with identical
statistics) and the PR fast-suite outcome and time of §1 / §3.3 (see the fact-check note in §3.3).
