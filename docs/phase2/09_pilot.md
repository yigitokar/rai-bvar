# 09 · Phase 2 pilot measurements and frozen defaults (WP3)

**Status.** Pilot of DESIGN §10.6 / §13 WP3, run 2026-10-22 on branch `phase-2-mcmc`. Clean-room prototypes in
`benchmarks/pilot/` (never imported by the package). They implement the DESIGN kernels faithfully (real SV-parameter
block with ASIS, exact mode, packed Gram, runner with burn/sample executables) but are timing and acceptance
vehicles, not the production code of WP4–WP14. The prototype SV kernel was checked before use: it reproduces N02's
reference acceptance rates and fold fractions, and a scratch Geweke test of the SV block alone (`T = 20`, 16 chains ×
10⁵ sweeps, 12 test functions) passed under Holm at 1% in exact mode (max |z| 2.6). That OCSN-10 test also passes
with `exact=False` (2.9), so it cannot see the mixture-approximation error and does **not** validate the exact
correction. The exact path was validated separately with a deliberately coarse one-component mixture, where
exact mode passes and `exact=False`, a stale-`w` mutant and DESIGN mutant M7 fail (§3.4).

**Reproduce.** `python -I benchmarks/pilot/run_all.py --out DIR` (all measurements, ≈ 15 min; `--quick` for a smoke
run; `--only name,…` for a subset). Every run writes `DIR/<name>.json` with the machine fingerprint, load average and
versions, plus `DIR/<name>.log`. Individual scripts take `--help`; each requires `--out DIR` (there is no default,
so results never land in the working tree by accident; use a directory outside the repository).

| Script | Measures (WP3 item) |
| :--- | :--- |
| `sv_block.py` | exact-mode cost with the real SV-parameter block; stage costs; scan `unroll` |
| `sv_acceptance.py` | joint `T = 300` exact-`h` acceptance for the three `(φ, σ)` settings; burn-in from crude starts; per-chain NC fold fractions, with two calibration DGPs (`σ = 0`, `σ = 0.05`) |
| `sweep.py` | full Linear × SV sweep at the BENCH-P2 shape through a burn/sample runner; chunk size; PRNG; buckets; devices; SV(t); BENCH-P2 projection |
| `transitory.py` | ν-grid cost, τ draws, outlier step |
| `prng.py` | threefry2x32 vs philox4x32 |
| `gram.py` | packed vs tiled vs batched Gram; the `gram="auto"` threshold |
| `buckets.py` | one vs two static equation buckets |
| `pilot_common.py` | shared kernels, data generators, ESS / split-R̂, fingerprint |

---

## 0. Machine, load and versions

| Item | Value |
| :--- | :--- |
| Machine | Apple M1 Pro, 8 performance + 2 efficiency cores (10 logical), 16 GB, macOS 26.3.1 (arm64) |
| Software | Python 3.12.7, `jax`/`jaxlib` 0.12.0.1 (CPU), NumPy 2.5.4, SciPy 1.18.1; `jax_enable_x64` on; `jax_threefry_partitionable` True; `XLA_FLAGS` unset |
| Devices | 1 CPU device (4 for the `dev4` run, `jax_num_cpu_devices = 4`) |
| Load average | **not idle.** Other workflow agents implemented Phase 2 work packages on the same machine. The 1-minute load average was 4.7–10.8 on 10 cores during the runs (recorded per run in the JSON files) |
| Data | `pilot_common.simulate_var_sv(n = 20, p = 4, T = 300, seed = 0)`: stationary VAR(4)-SV (own lag `l` coefficient `0.5/l`, small cross lags, spectral radius ≤ 0.9, unit lower-triangular `A`, AR(1) log variances with `φ = 0.95`, `σ = 0.2`); `k = 81`, `q = 100`, `P = 5050`, `T_pad = 320`, 4 chains |

All timings are the minimum over repeats (every executable call is AOT-compiled first and timed with
`block_until_ready`). Under this load they are **upper bounds**. Repeated interleaved runs (§4.1) put the
run-to-run noise at about ±5%. The x86 benchmark and the idle-machine re-run belong to WP24.

---

## 1. Decisions (one line per default)

| # | Default or question | Decision | Evidence |
| :-- | :--- | :--- | :--- |
| 1 | Q1: `SV(exact=True)` default | **Keep.** Exact mode costs +12% to +22% of the full sweep (≈ +0.7 to +1.2 ms; four paired runs) and ≤ 10% of ESS. BENCH-P2 projects to 75–80 s < 100 s on one loaded device, so Q1/Q2 are **not escalated** | §2, §3.3, §4 |
| 2 | §10.3 budget | Revise the exact-correction line from 0.4 to ≈ 1 ms (0.7–1.2 ms in the full sweep; 1.1 ms in isolation). Measured total 6.3–6.7 ms per sweep under load 6–10, against the 5.6 ms budget | §2, §4 |
| 3 | BENCH-P2 (§10.1) | Projected **75–80 s** on one device over four runs (pass < 120 s) and **47–51 s** with `set_cpu_devices(4)` (stretch < 60 s met) | §4 |
| 4 | SV cold start (DESIGN §5.4) | **Defect found; amendment required (escalated).** `h ≡ μ` plus a rejected first exact `h`-step gives `C_T = 0`, then σ = 0 is accepted and absorbing (3–6% of chains with OCSN-10, 34–59% with KSC-7). Fix: after the §5.4 values, draw `h` once from the AWOL proposal, accepted unconditionally | §3.1 |
| 5 | Exact-`h` acceptance at `T = 300` (R2, N01 §14 item 5) | **Verified:** OCSN-10 0.973 / 0.983 / 0.979 for `(φ, σ) = (0.95, 0.2) / (0.5, 0.05) / (0.99, 0.1)`; worst chain 0.925; longest rejection run ≤ 5; composed Linear × SV sweep 0.969 (worst of 80 equation-chains 0.955). WP12's test band "≈ 0.96–0.98" stands | §3 |
| 6 | Convergence defaults vs SV mixing | **Escalated (new open question).** φ and σ have ESS ≈ 22–28 per 1,000 sweeps per chain. With the default `draws = 2000`, 4 chains give bulk ESS ≈ 200 < `ess_min = 400`, and split-R̂ reaches 1.03–1.06 > `rhat_max = 1.01`. Since SV parameters are primary monitors (D-33), `converged = False` will be routine at default settings | §3.3 |
| 7 | PRNG (lever 9) | **Keep threefry2x32 (provisional, see below).** Philox4x32 makes the sweep's RNG work 2.4× faster (0.70 → 0.29 ms per sweep), which saves 6–8% of a full sweep; its statistical quality is untested and it would become a spec field | §6 |
| 8 | `Runtime.chunk` | **Keep 250.** 50–500 differ by less than the noise (6.58–6.86 ms per kept draw); compile 1.4–1.8 s regardless; 16.6 KB per chain-draw on the host | §7 |
| 9 | `gram="auto"` threshold (§10.5) | **Keep** packed iff `T_pad·P·8 ≤ 256 MiB`, else tiles of `T_pad·tile·8 ≤ 64 MiB`. Packed is fastest at every shape (3.9× batched at the reference shape); at 430 MB the rule's tiles cost 0–9% | §8 |
| 10 | Scan `unroll` (AWOL) | **Keep 1 (provisional).** Unroll 2–16 gives no gain in the SV block or the full sweep (−1% to +2%) and compiles slower | §9 |
| 11 | Two static equation buckets | **Not adopted (provisional).** Mean block −8% at the best split, but only −2% to −4% of the full sweep (about the noise), at the cost of a second compiled equation body and draws not bitwise equal to one bucket. Recommend a column-major packed order in WP5 so the lever stays available (prefix of `FF`) | §10 |
| 12 | ν grid (R5) | **Keep the partially collapsed grid step.** The O(T·G) sums cost 0.82 ms per sweep with `log(c + r²) − log c` (1.9 ms with `logaddexp`). The real SV(t) cost is the τ draw: `random.loggamma` costs **12.1 ms** per sweep (a batched rejection `while` loop); an exact vectorised Marsaglia–Tsang sampler with 2 pre-drawn candidates costs 2.1 ms. SV(t) full sweep 19.6 → 9.9 ms | §5 |
| 13 | Outlier step | Measured 1.56 ms per sweep (≈ one indicator step, as estimated); the `z²·k⁻²` form gives identical draws at 1.42 ms (optional) | §5 |

**Provisional decisions.** DESIGN §10.6 asks for an idle machine; this one was not idle (§0). Decisions 7, 10 and 11
rest on paired differences (philox −6 to −8%, two buckets −2 to −4%, unroll 4 −1% to +2%) that are about the size of
the noise (±2.5% within a round, ±5% across load levels). They stand as Phase 2 defaults but are **provisional until
WP24 re-runs `run_all.py --repeat 3 --only sweep_ref,sweep_philox,sweep_buckets2,sweep_unroll4` on an idle machine
and on x86**. A reversal there changes a default, not the design; philox and two buckets change the draws, so
adopting either would be a spec-level change, while `unroll` is internal. The other decisions are robust to the load:
1–3 and the no-escalation conclusion can only get safer on an idle machine, which is faster; 4–6 rest on acceptance
rates, ESS and failure counts, not timings; 8 rests on the absence of any trend across chunk sizes; 9 on gaps of
14–34% (tiles below the threshold) and 1.8–4.0× (batched) that are far above the noise; 12 on cost ratios of 2–6×;
13 is a cost estimate, not a choice between levers.

Escalations for the orchestrator: decision 4 (amend DESIGN §5.4; owners WP14 init and WP12 kernel), decision 6
(defaults of `MCMCSettings.draws`, `ess_min` or the primary monitor set; owners WP2/WP15/WP18). Recommendations to
other WPs, which their owners decide: §11. Two of them change DESIGN text and need the design owner as well: the
fold-fraction warning of §8.3 (§3.5) and the coarse-mixture Geweke configuration for exact mode and M7 (§3.4).

---

## 2. Exact-mode cost with the real SV-parameter block (`sv_block.py`)

`B = 80` lanes (4 chains × 20 equations), `T_eff = 300`, `T_pad = 320`, OCSN-10; 300 sweeps per jitted call, minimum
over 5 calls; load ≈ 6. Cumulative variants (stage cost = difference):

| Variant | ms per sweep (80 lanes) | Stage | Stage ms | §10.3 budget |
| :--- | ---: | :--- | ---: | ---: |
| V1 indicators + AWOL | 1.400 | indicators + AWOL | 1.40 | 1.5 |
| V2 + exact `h` correction | 2.085 | exact `h` (one extra `(T_pad, 10)` logsumexp + `w`) | 0.69 | } 0.4 |
| V3 + centred `(γ, φ)`, `σ²` | 2.297 | C-step | 0.21 | } 0.3 |
| V4 + NC `φ`, `(μ, σ)` with exact correction (**default**) | **2.865** | NC step incl. its exact correction | 0.57 | |
| V5 = V4 with `exact=False` | 1.756 | exact mode total (V4 − V5) | **1.11** | |
| V4 with KSC-7 | 2.526 | — | — | — |

- The real parameter block (C + NC without exact terms, V5 − V1) costs 0.36 ms, matching the 0.3 ms budget. It is
  ≈ 6% of a full sweep, and ≈ 18% together with exact mode, inside N07 §14 item 6's "+10–20% for the real block".
- The exact mode costs 1.1 ms, not 0.4 ms. Each extra logsumexp over `(80, 320, 10)` costs ≈ 0.39 ms: XLA-CPU
  float64 `exp` runs at about 1.5 ns per element, and a `(J, T)` layout is no faster (checked: identical draws, same
  time). In the full sweep the paired difference is 0.7–1.2 ms (§4.1).
- No cheaper exact alternative exists: tabulating `log g` would change the target, so it is rejected.

---

## 3. Joint-path exact-`h` acceptance and burn-in (`sv_acceptance.py`; risk R2)

SV core alone (ZeroMean), `μ = −1`, `T = 300` padded to 320, 8 simulated series × 4 chains per setting, burn 2,000,
keep 10,000; default priors (`μ ~ N(log ψ̂, 10)`, `Beta(20, 1.5)`, `B_σ = 1`, relative offset `c = 1e-12·ψ̂`).
Mean over 32 chains (worst chain in brackets; the per-chain fold fractions are in §3.5):

| `(φ, σ)` | mixture, mode | `h` | NC `(μ, σ)` | C `(γ, φ)` | C `σ²` | NC `φ` | fold (mean) | longest `h` rejection run |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.95, 0.2) | OCSN-10 exact | 0.973 (0.925) | 0.994 | 0.721 | 0.998 | 0.863 | 2.4% | 5 |
| (0.5, 0.05) | OCSN-10 exact | 0.983 (0.954) | 0.994 | 0.723 | 0.999 | 0.845 | 6.0% | 3 |
| (0.99, 0.1) | OCSN-10 exact | 0.979 (0.952) | 0.994 | 0.723 | 0.999 | 0.839 | 1.2% | 4 |
| (0.95, 0.2) | KSC-7 exact | 0.746 (0.610) | 0.935 | 0.720 | 0.998 | 0.862 | 2.3% | 34 |
| (0.5, 0.05) | KSC-7 exact | 0.817 (0.715) | 0.928 | 0.723 | 0.999 | 0.843 | 5.8% | 22 |
| (0.99, 0.1) | KSC-7 exact | 0.763 (0.675) | 0.936 | 0.723 | 0.999 | 0.840 | 1.2% | 30 |

In the composed Linear × SV sweep at the BENCH-P2 shape (§4, 80 equation-chains, 3,000 sweeps after warm-up): `h`
0.969 (worst 0.955), NC 0.993, C `(γ, φ)` 0.729, `σ²` 0.998, longest rejection run 4, NC fold fraction 0.53% (mean;
maximum 4.3%; above 1% in 15 of 80 equation-chains, §3.5).

The OCSN-10 numbers confirm N02 §9 (0.96–0.98) on joint `T = 300` paths. KSC-7 is usable as a replication option
(acceptance ≈ 0.75–0.82). Its longest rejection runs (34 / 22 / 30) stay below the current §8.3 threshold of 50
sweeps, so the longest-run warning would not fire. It would fire on KSC-7 only if WP15 lowers the threshold after
calibrating it on OCSN-10, where the pilot maximum is 5.

### 3.1 Initialisation defect (DESIGN §5.4) and its fix

DESIGN §5.4 initialises AR SV with `μ = log(Σ m ẽ²/T_eff) + N(0, 0.5²)` and `h ≡ μ`. If the first exact `h` proposal
is rejected, `h` stays exactly equal to `μ`, so `C_T = ½[Σ(h_{t+1} − μ − φ(h_t − μ))² + (1 − φ²)(h_0 − μ)²] = 0`. The
σ² independence proposal `IG(T/2, C_T)` is then the point 0, and its log ratio `(σ² − 0)/(2B_σ) > 0` accepts it.
σ = 0 is absorbing: `h̃ = (h − μ)/σ` is NaN, every later move is rejected and `h` is frozen. No status code fires,
because no Cholesky fails. Measured share of chains absorbed at sweep 1:

| `(φ, σ)` | OCSN-10 | KSC-7 |
| :--- | ---: | ---: |
| (0.95, 0.2) | 6.2% | 53% |
| (0.5, 0.05) | 6.2% | 59% |
| (0.99, 0.1) | 3.1% | 34% |

(200 single-chain starts on one dataset: 12/200 OCSN-10, 143/200 KSC-7.)

**Fix, used in every pilot run:** after the §5.4 values, draw `h` once from the AWOL proposal given those `(μ, φ, σ)`
and indicators drawn at `h ≡ μ`, and accept it unconditionally. This is an initialisation, not an MCMC step, so the
target is unchanged, and it makes `C_T > 0` almost surely. With it, no chain was absorbed in any run. The same
applies to the RW path (`S = Σ(h_t − h_{t−1})² = 0`). As defence in depth, WP12 should report `C_T > 0` and `σ > 0` in
the SV step's `StepInfo.ok`, so that a degenerate state raises (D-18) instead of freezing silently.

### 3.2 Burn-in from crude starts

From the fixed design start and from deliberately bad starts `h ≡ log ψ̂ ± 5`:

- the exact-`h` acceptance in sweeps 0–100 is already 0.966–0.982 (OCSN-10); there is no slow-start regime at
  `c_rel = 1e-12` (the N01 §7.2 pathology needs a large `c/λ`, ≈ 0.5 in its toy);
- the log-likelihood enters the central 98% band of its stationary draws after a median of 0 sweeps (design start)
  or 2–12 sweeps (±5 starts), and at most 18 (design) or 34 (±5) sweeps;
- under common random numbers, 28–81% of the ±5-start chains coalesced bitwise with the design-start chain by sweep
  2,000. The rest still differ in the last bits, and their post-burn-in summaries agree to the reported digits.

The default `burn = 1000` is therefore ample for the SV block. Mixing of φ and σ, not burn-in, is the binding
constraint (§3.3).

### 3.3 Mixing, and what it means for `converged`

ESS per 1,000 sweeps per chain (median over 32 chains; OCSN-10, exact; `exact=False` in brackets):

| `(φ, σ)` | μ | φ | σ | `h_T` | log-lik |
| :--- | ---: | ---: | ---: | ---: | ---: |
| (0.95, 0.2) | 221 (233) | 24 (27) | 22 (21) | 218 (227) | 42 (42) |
| (0.5, 0.05) | 161 (165) | 25 (26) | 26 (27) | 352 (353) | 55 (54) |
| (0.99, 0.1) | 271 (286) | 28 (30) | 25 (25) | 220 (232) | 65 (65) |

- Exact mode costs at most ≈ 10% of ESS (often nothing). With +12% to +22% time, its ESS per second is about
  0.75–0.89 of the approximate sampler's.
- Max split-R̂ over 8 datasets (4 chains, `burn = 1000`, `draws = 2000`, the `MCMCSettings` defaults; plain split-R̂,
  not rank-normalised): μ 1.004–1.007, `h_T` 1.004–1.011, log-likelihood 1.018–1.029, **φ 1.032–1.062, σ
  1.033–1.045**. Bulk ESS of φ and σ at 4 × 2000 draws ≈ 180–220.
- So at the defaults `rhat_max = 1.01`, `ess_min = 400`, SV fits will routinely report `converged = False` through
  φ and σ (D-33 lists SV parameters as primary monitors). BENCH-P2's 10,000 draws give ESS ≈ 1,000 and should pass.
  Options for the owners of WP2/WP15/WP18: raise the default `draws` for SV compositions, move φ and σ to the
  secondary set (keeping μ, `h_T` and the log-likelihood primary), or gate SV parameters on ESS only. Escalated, not
  decided here.

### 3.4 Geweke validation of the exact path (scratch; N01 §14 items 4–5)

Successive-conditional vs marginal-conditional Geweke test of the prototype SV block alone: `T = 20` (padded to 32),
prior `μ ~ N(0, 1)`, `Beta(20, 1.5)`, `B_σ = 0.1`, 16 chains × 2·10⁵ sweeps, 12 bounded or finite-variance test
functions (moments of μ, φ, σ; `h_0`, `h_T`, `μσ`, three `tanh` functions of `h`), batch-means standard errors, Holm
at 1% (reject iff max |z| > 3.34). Mutants change one line of `pilot_common.sv_step`:

| Kernel | OCSN-10 (production table) | one-component `N(−1.2704, π²/2)` (coarse) |
| :--- | ---: | ---: |
| exact (default) | 2.07 pass | 2.51 pass |
| `exact=False` | 2.13 pass | **10.05 fail** |
| stale `w`: NC `(μ, σ)` move compared against `w(h_cur)` instead of `w` after the `h` step | 2.88 pass | **12.46 fail** |
| M7 (DESIGN §9.7): NC `(μ, σ)` move uncorrected in exact mode | 2.11 pass | **5.31 fail** |

- With OCSN-10 the mixture-approximation error is far below what a CI-sized Geweke test resolves: `exact=False`
  passes. This settles the first part of N01 §14 item 5 at this size, and it means **an OCSN-10 Geweke test cannot
  validate the exact correction or catch errors in it** (stale `w`, M7).
- A deliberately coarse mixture makes the approximation error large, so the same test has power over the exact
  path: exact mode passes, while `exact=False`, stale `w` and M7 fail. Exact mode is exact for any mixture table
  (N01 §10.2), so the coarse table is a valid test configuration, not a different model.
- An independent review harness (`T = 15`, 16 × 2·10⁵ sweeps) reproduced this (stale `w`: 1.92 with OCSN-10, 12.98
  with one component). With OCSN-10 it caught the M6-type `|σ|` without the `h̃` flip (82.6) and `C_T` without the
  `h_0` term (25.9). Dropping the centred step's auxiliary-prior term (variances 10¹² / 10⁸, so its effect is
  ≈ 10⁻⁸) is an equivalent mutant (1.90).
- Earlier run (16 × 10⁵ sweeps, OCSN-10): exact 2.6, `exact=False` 2.9, consistent with the table.

### 3.5 NC fold fraction and the §8.3 warning

DESIGN §8.3 warns when the NC fold fraction exceeds 1% ("near-homoskedastic equation"). The §3 means exceed 1% in
every setting, including the clearly heteroskedastic `(0.95, 0.2)`. Per equation-chain (OCSN-10, exact, 10⁴ kept
sweeps, 32 chains = 8 datasets × 4; `sv_acceptance.py`, which adds two calibration settings: homoskedastic
`σ = 0` and weak `σ = 0.05` at `φ = 0.95`):

| DGP `(φ, σ)` | mean | median | max | chains > 1% | chains > 5% | σ posterior 5% quantile, chains ≤ 1% / > 1% |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| (0.95, 0.2) | 2.4% | 0.0% | 10.1% | 25% (2 datasets) | 25% | ≥ 0.082 / ≤ 0.007 |
| (0.99, 0.1) | 1.2% | 0.0% | 9.3% | 12% (1 dataset) | 12% | ≥ 0.051 / 0.006 |
| (0.5, 0.05) | 6.0% | 6.8% | 11.3% | 88% | 75% | ≥ 0.112 / ≤ 0.027 |
| (0.95, 0.05) | 6.3% | 6.8% | 12.2% | 88% | 75% | 0.043–0.046 / ≤ 0.028 |
| (0.95, 0) homoskedastic | 7.4% | 8.9% | 13.5% | **100%** | 75% | — / ≤ 0.043 |
| composed sweep, BENCH-P2 shape (80 equation-chains, 3,000 sweeps) | 0.53% | — | 4.3% | 19% | 0% | — |

- The fold fraction tracks the lower tail of the σ posterior, and the chains of one dataset agree, so it is a
  property of the series, not sampler noise. In the three reference settings the chains split into two groups,
  ≤ 0.4% (σ posterior bounded away from 0) and ≥ 1.7% (posterior mass near 0). Under `σ ≤ 0.05` one dataset each
  lies in between (0.6–1.0% and 1.1–1.5%). The mean over series (the §3 column) mixes the groups, so it must not
  be the criterion.
- Per equation, the 1% threshold detects "the posterior of σ reaches 0": it flags every chain under `σ = 0` (minimum
  1.1%) and 88% under `σ = 0.05`, while a 5% threshold would miss 25% of the `σ = 0` chains. But at `T = 300` that
  state is common for series from clearly heteroskedastic DGPs too: 12–25% of them here, and 19% of the
  equation-chains of the composed sweep. With `n = 20`, almost every SV fit would carry the warning.
- The chains near 1% (the `σ ≤ 0.05` settings) have a σ posterior 5% quantile of ≈ 0.04, and in the reference
  settings the two groups are separated by that quantile (≥ 0.051 vs ≤ 0.027). A criterion on the σ posterior is
  therefore equivalent at this `T`, and it does not depend on the NC parametrisation.

Recommendation to WP15 (and to the design owner for the §8.3 text; §11): evaluate the condition **per equation**,
pooled over that equation's chains and never averaged over equations. Report it as one informational line ("k of n
equations: the posterior of σ reaches 0; SV may be unnecessary there, and stochvol/bayesianVARs parity differs,
D-7") plus the per-equation fold fraction in `fit.diagnostics`, not as a warning of a sampler problem. Keep the 1%
threshold, or switch to a criterion on the σ posterior. The decision is WP15's.

---

## 4. Full Linear × SV sweep at the BENCH-P2 shape (`sweep.py`)

Structural Mode-I-style prior, weighted path with the packed Gram, Jacobi-scaled two-solve draw, real SV core
(exact, ASIS, OCSN-10), keys by `fold_in` (sweep → block → equation → sub-draw), burn executable (`fori_loop` with a
traced count, donated state) and sample executable (`scan` over the chunk, traced `thin`, `tree_where` snapshot,
per-draw extract, device → host copy). 4 chains, `vmap` unless stated; 300 warm-up sweeps, then 3 × 1,000 timed
sweeps (minimum). The table is the first full run (`run_all.py`). Single runs at different load levels differ by
up to ±5%, so the lever comparisons rest on the paired repeats of §4.1.

The cold start of every run uses the §3.1 fix; no status code fired in any run. The SV(t) runs use the collapsed
ν grid with standardized τ.

| Run | ms per sweep | ms per kept draw (chunk 250, incl. host copy) | BENCH-P2 projection | load |
| :--- | ---: | ---: | ---: | ---: |
| **reference** (exact, threefry, 1 bucket, unroll 1) | **6.27** | 6.63 | **77.0 s** | 8.4–10.5 |
| `exact=False` | 5.57 | 5.60 | 65.6 s | 7.9 |
| philox4x32 | 6.11 | 5.99 | 71.7 s | 6.6–7.0 |
| two equation buckets (`s = 10`) | 6.00 | 6.11 | 71.5 s | 6.4–6.7 |
| unroll 4 | 6.23 | 6.43 | 74.7 s | 6.1–6.6 |
| **4 CPU devices** (`shard_map`) | **3.53** | 3.95 | **47.7 s** | 5.6–6.6 |
| SV(t), `random.loggamma` τ, `logaddexp` ν sums | 19.62 | 20.94 | 233.9 s | 5.8–7.1 |
| SV(t), vectorised MT τ (M = 2), `log_sum` ν sums | 9.93 | 10.51 | 121.2 s | 6.9–7.6 |

Projection = compile (burn + sample executables, 3.1 s) + 1,000 burn sweeps + 10,000 kept draws at chunk 250 +
prepare/init (1.0 s) + ESS/split-R̂ for 330 monitors × 4 × 10,000 draws (0.4 s). Not included: the DataTree build
and storage plan (WP14) and EB (BENCH-P2 uses `hierarchy="fixed"`). These take seconds. Across the four reference
runs (75–80 s, §4.1) that leaves a margin of ≈ 1.25× under load to the 100 s escalation line and ≈ 1.5× to the
120 s target.

Consistency with §2 and §8: the mean block (packed Gram 1.12 ms + Cholesky, solves, GEMMs, normals ≈ 2.2 ms;
`buckets.py` 3.30 ms) plus the SV block (2.87 ms) is 6.2 ms, which matches the 6.27 ms sweep. The sample executable
adds ≈ 0.1–0.4 ms per kept draw (extract, snapshot `where`, host copy).

### 4.1 Interleaved repeats (noise check)

`run_all.py --repeat 3 --only sweep_ref,sweep_nonexact,sweep_philox,sweep_buckets2,sweep_unroll4,sweep_dev4`
runs the variants in interleaved rounds (load 6.2–10.6), so each round shares the load drift. Ratios are against the
reference run of the same round:

| Variant | ms per sweep (rounds 1–3) | ratio to reference | ms per kept draw, chunk 250 | BENCH-P2 projection |
| :--- | :--- | :--- | :--- | :--- |
| reference | 6.60, 6.49, 6.65 | 1 | 6.40, 6.84, 6.95 | 74.8, 79.0, 80.4 s |
| `exact=False` | 5.40, 5.60, 5.80 | 0.82, 0.86, 0.87 | 5.46, 5.67, 5.88 | 64.0, 66.5, 68.8 s |
| philox4x32 | 6.12, 6.11, 6.11 | 0.93, 0.94, 0.92 | 6.49, 6.27, 6.18 | 76.6, 74.6, 73.7 s |
| two buckets (`s = 10`) | 6.33, 6.29, 6.49 | 0.96, 0.97, 0.98 | 6.40, 6.34, 6.54 | 75.1, 74.4, 76.9 s |
| unroll 4 | 6.55, 6.47, 6.76 | 0.99, 1.00, 1.02 | 6.61, 6.59, 6.80 | 77.1, 76.7, 79.2 s |
| 4 CPU devices | 3.87, 3.75, 4.12 | 0.59, 0.58, 0.62 | 3.99, 3.87, 4.21 | 48.6, 47.4, 51.4 s |

Four further runs of the reference configuration alone gave 6.50–6.81 ms per sweep, so the run-to-run noise at this
load is about ±2.5% (±5% across runs at different load levels). Over the first full run (§4) and the three rounds
above, the reference BENCH-P2 projection ranges from **74.8 to 80.4 s**; a later reference run at load 4.3–4.9 (the
§3.5 re-run) gave 6.29 ms per sweep and 74.2 s. In paired terms (first full run and three rounds):

- exact mode costs +12%, +16%, +15% and +22% of the sweep (≈ +0.7 to +1.2 ms);
- philox saves 6–8%, two buckets 2–4%, and unroll 4 nothing (−1% to +2%);
- 4 devices give a 1.6–1.7× speed-up.

---

## 5. ν-grid, τ and outlier costs (`transitory.py`; risk R5)

`B = 80` lanes, `T_pad = 320`, `G = 38`, `K = 20`; ms per call for all lanes, minimum over 5 × 100 calls; load ≈ 6.

| Kernel | ms |
| :--- | ---: |
| ν sums `Σ_t m_t log1p(r²/c_g)`, `logaddexp(log r², log c) − log c` | 1.88 |
| same, `log1p(exp(log r² − log c))` | 2.37 |
| same, `log(c + r²) − log c` with `r²` once per row (rows with `log r² > 700` use `log r²`, exact in float64) | **0.82** |
| full collapsed ν step + τ, `random.loggamma` τ | 15.12 |
| full collapsed ν step + τ, MT (M = 2) τ, `logaddexp` sums | 4.86 |
| full collapsed ν step + τ, MT (M = 2) τ, `log_sum` sums | **3.64** |
| τ alone: `random.loggamma` / `log(random.gamma)` | 12.11 / 11.90 |
| τ alone: vectorised Marsaglia–Tsang, M = 1 / 2 / 3 / 4 candidates | 3.23 / **2.07** / 2.32 / 2.87 |
| uncollapsed ν \| τ (O(T + G); mixes ≈ 40× worse, N03 §4.3) | 0.10 |
| outlier `o \| ·` then `p \| o`; with the `z²·k⁻²` form | 1.56; 1.42 |
| 320 uniforms per lane (reference) | 0.22 |

- The three ν-sum forms agree to 1.1e-15 / 5.1e-15 relative.
- The MT sampler is exact: the first accepted candidate of an i.i.d. sequence is the rejection sampler; candidates
  are drawn in one vectorised batch and a `while_loop` handles elements that rejected all M. KS tests against
  `scipy.stats.gamma` at shapes 2, 2.5, 5 and 20.5 (2·10⁵ draws each) give p = 0.56, 0.52, 0.57, 0.67.
- `random.gamma`/`loggamma` under `vmap` run one batched rejection loop over all `C·n·T` elements. That costs
  ≈ 470 ns per draw, which makes the τ draw, not the ν grid, the SV(t) bottleneck.
- The outlier `z²·k⁻²` form gave identical draws on every element tested.
- SV(t) with both fixes costs 1.6× the SV sweep (9.9 vs 6.3 ms), projecting to ≈ 120 s at the BENCH-P2 shape.
  SV(t) is not part of the BENCH-P2 target. R5 is mitigated, not removed.

---

## 6. threefry2x32 vs philox4x32 (`prng.py`)

| Per call | threefry2x32 | philox4x32 | ratio |
| :--- | ---: | ---: | ---: |
| 24,000 float64 uniforms | 0.177 ms | 0.042 ms | 4.2× |
| 24,000 float64 normals | 0.294 ms | 0.153 ms | 1.9× |
| one sweep's draws for 80 lanes (`fold_in` chain + `uniform(320)`, `normal(321)`, `normal(100)`, 8 scalars) | 0.700 ms | 0.292 ms | 2.4× |

The sweep's PRNG work is ≈ 11% of a full sweep with threefry and ≈ 5% with philox. End to end, philox saves 6–8%
of the sweep in the interleaved repeats (§4.1). Decision 7 keeps threefry: the gain is small, philox's statistical
quality is untested (N07 §14 item 7), and switching the PRNG changes every stream, so it would be a spec field.

---

## 7. Chunk size (`sweep.py --chunks 50,125,250,500`)

| chunk | 50 | 125 | 250 | 500 |
| :--- | ---: | ---: | ---: | ---: |
| ms per kept draw incl. host copy | 6.58 | 6.60 | 6.63 | 6.86 |
| sample-executable compile (s) | 1.76 | 1.57 | 1.55 | 1.38 |

All four lie within the run-to-run noise. Each kept draw transfers 16.6 KB per chain (`θ` 100 × 20 plus four
`n`-vectors `h_T`, μ, φ, σ: 2,080 float64 = 16,640 B), so the output buffer is 16.6 MB per call at chunk 250 with
4 chains. Chunk size does not matter for speed; 250 stays (memory bound plus host checks every 250 draws).

---

## 8. Packed-tile threshold (`gram.py`)

`G_b = Fᵀ diag(w_b) F` for `B = 4n` weight vectors; ms per call (minimum over 5 × 20 calls); relative error
against `batched` ≤ 1.2e-15 everywhere.

| Shape | `q` | `T_pad` | `P` | `FF` | rule | packed | batched | tiled (tile = columns) |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :--- |
| `n = 20, p = 4, T = 300` | 100 | 320 | 5,050 | 12.9 MB | packed | **1.12** | 4.43 | 1.28 (2,048) |
| `n = 50, p = 4, T = 300` | 250 | 320 | 31,375 | 80.3 MB | packed | **13.8** | 40.8 | 16.5 (2,048), 15.8 (8,192) |
| `n = 20, p = 13, T = 300` | 280 | 320 | 39,340 | 100.7 MB | packed | **7.42** | 17.3 | 9.92 (2,048), 9.59 (32,768) |
| `n = 30, p = 13, T = 600` | 420 | 608 | 88,410 | 430 MB | tiled, 13,312 | 49.5 | 86.9 | 55.3 (2,048), **53.8 (13,312)** |

Packed is fastest at every shape. Tiles cost 14–34% below the threshold, where `FF` fits comfortably, and 0–9% above
it (an earlier run at the same shape measured 48.5 ms tiled vs 48.6 ms packed). The §10.5 shape rule (decision 9)
therefore keeps packed exactly where it pays and bounds memory where it does not.

---

## 9. Scan `unroll` (`sv_block.py`, `sweep.py --unroll 4`)

| unroll | 1 | 2 | 4 | 8 | 16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tridiagonal sampler alone, 80 × 321 (ms) | 0.189 | 0.134 | 0.137 | 0.142 | 0.154 |
| its compile (s) | 0.13 | 0.13 | 0.30 | 0.64 | 0.52 |
| full SV block, default variant (ms per sweep) | 2.865 | 2.840 | 2.850 | 2.841 | 2.864 |
| full sweep, ratio to the paired `unroll = 1` run (§4.1) | 1 | — | 0.99–1.02 | — | — |

The sampler alone gains 29% at unroll 2, but that gain does not survive in the SV block (−0.9%) or the full sweep
(within noise). Decision 10: `unroll = 1`, which matches N07 §5's loaded-machine finding.

---

## 10. Two static equation buckets (`buckets.py`, `sweep.py --buckets 2`)

Exact padding ratios `Σ(padded cost)/Σ(active cost)` at `n = 20, k = 81` (DESIGN §2.3 quoted ≈ 1.08×/1.05× for two
buckets; the exact values are higher):

| Split `s` (bucket A = equations `< s`, padded to `k + s − 1`) | Cholesky | Gram |
| :--- | ---: | ---: |
| one bucket | 1.333 | 1.215 |
| `s = 10` | 1.152 | 1.100 |
| `s = 12` | 1.156 | 1.103 |
| `s = 14` | 1.175 | 1.116 |

Mean draw for 80 lanes (packed Gram + Cholesky + two solves + residual): one bucket 3.30 ms; two buckets 3.32
(`s = 10`), **3.05 (`s = 12`)**, 3.19 (`s = 14`). The full sweep with `s = 10` saves 2–4% in the interleaved repeats
(§4.1), about the size of the noise. Active coordinates differ from one bucket by ≤ 3e-15 (GEMM blocking
changes with the shape), so a bucket choice changes draws and would be a spec-level shape rule. Decision 11: not
adopted in Phase 2. The prefix property needs the column-major packed order (pairs `(r, s)`, `r ≤ s`, sorted by `s`
then `r`; `pilot_common.packed_index`): with it, bucket A reads `FF[:, :q_A(q_A+1)/2]`. WP5 can adopt that order now
at no cost.

---

## 11. Recommendations to other work packages (owners decide)

| WP | Recommendation | Section |
| :--- | :--- | :--- |
| WP1 / design owner | Amend DESIGN §5.4: AR and RW SV cold start = §5.4 values + one unconditional AWOL draw of `h`. Revise the §10.3 exact-correction line to ≈ 1 ms and the §2.3 two-bucket ratios to 1.15×/1.10× | §3.1, §2, §10 |
| WP4 | Implement `log_inv_gamma` / `loggamma` for shape ≥ 1 as the vectorised Marsaglia–Tsang sampler with 2 pre-drawn candidates and a `while_loop` fallback (exact; 5.8× faster than `random.loggamma` under `vmap`); keep `random.loggamma` for shape < 1 or for scalar per-lane draws | §5 |
| WP5 | Use the column-major packed order (`pilot_common.packed_index`) for `FF` and `XX` | §10 |
| WP12 | Include `C_T > 0` and `σ > 0` in the SV step's `ok`; acceptance test band for OCSN-10 exact `h` at `T = 300`: mean over chains in [0.95, 0.995] | §3 |
| WP13 | ν sums via `log(c_g + r²) − log c_g` with `r²` computed once per row and the `log r² > 700` pass-through; τ via the WP4 sampler; optional `z²·k⁻²` outlier logits | §5 |
| WP14 | Initialise SV with the AWOL draw (§3.1); `chunk = 250` | §3.1, §7 |
| WP12 / WP19 | Validate exact mode with a Geweke configuration that uses a deliberately coarse mixture (one component, `N(−1.2704, π²/2)`, injected through the `mixture_table` slot that M12 already uses): there exact passes while `exact=False` (z ≈ 10), a stale-`w` NC move (≈ 12.5) and M7 (≈ 5.3) fail at 16 × 2·10⁵ sweeps. The OCSN-10 Geweke test cannot see any of these, so M7 must be run in the coarse configuration (its DESIGN §9.7 catcher is "G, tier to confirm"). Unit-test `h_log_accept` / `nc_log_accept` against the `w` of the post-`h`-step state | §3.4 |
| WP15 | Calibrate the longest-rejection-run warning on OCSN-10 (pilot maximum 5 in 10⁴ sweeps). KSC-7 reaches 34, below the current 50-sweep threshold | §3 |
| WP15 / design owner | Redefine the §8.3 fold-fraction warning: evaluate it per equation (pooled over chains), and report it as one informational line ("k of n equations: the posterior of σ reaches 0") plus the per-equation fold fraction in `fit.diagnostics`. As written it fires on 12–25% of equations from clearly heteroskedastic DGPs at `T = 300` (19% in the composed sweep). Keep the 1% threshold, or use the σ posterior 5% quantile (≈ 0.04 corresponds to 1%) | §3.5 |
| WP2 / WP15 / WP18 | Resolve the default `draws` / `ess_min` / primary-monitor conflict for SV (§3.3) | §3.3 |
| WP24 | Re-run `run_all.py` (and `--repeat 3` for the levers) idle and on x86, and confirm or reverse the provisional decisions 7, 10 and 11; BENCH-P2 baseline expected ≤ 6.3–6.7 ms per sweep on M1 Pro (one device) | §1, §4 |

---

## 12. Not measured here (and why)

- **x86 and an idle machine**: no x86 host was available, and the shared machine never went idle (§0). WP24 owns
  both; decisions 7, 10 and 11 stay provisional until then (§1).
- **ESS per second of the full composition**: the SV-block ESS (§3.3) is measured. ESS for the coefficients of the
  composed sampler needs the production monitors (WP15); BENCH-P2 reports it (WP24).
- **Reduced form (CCCM)**: not a WP3 item; N07 §12 has a loaded-machine proxy (9.1–13.3 ms per sweep for the
  coefficient block).
- **Philox statistical quality**: out of scope (decision 7 keeps threefry).
