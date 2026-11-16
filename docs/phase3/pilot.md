# Phase 3 pilot: GL and CommonSV mixing, costs, BENCH-P3 (WP36)

**Status.** Pilot of P3 DESIGN §8 and §10 (§13 WP36), run 2026-10-28 on branch `phase-3-wp36` from `main` at `8b5fce6`
(WP26–WP35 merged). It measures the **validated production kernels**: no kernel, prior, setting, tolerance or
criterion was changed, and nothing under `src/` was touched. It gives the budget evidence for D-58 (Q21, the owner's
decision), replaces the cost estimates of §10 (E163, E170) with idle-machine measurements, reports BENCH-P3, and
estimates the compute of the Gruber–Kastner acceptance run (§9.10; Q26). The decision rule of §1 was committed
(`ea50e09`) before the campaign ran, and its addendum (`8d677b4`) before the runs it governs. Before that commit two
checks of the scripts had printed gate verdicts: the harness check (seed 7, 4 × 2,000 and 4 × 3,000; described below) and a smoke test
(4 × 3,000 after 200 burn-in sweeps; DL × Constant, row 325's CommonSV, SSVS(Beta) × SV at `n = 7`), all failing at
2,000. They were not campaign settings, and the rule was written from the design, not from them.

**Reproduce.** The mixing campaign (ESS and R-hat do not depend on the load, so the workers may share the machine):
`python -I benchmarks/pilot/phase3_run_all.py --out DIR --only mixing --workers 6 --single-thread --bp2-seeds 0`
(every size's runs and BENCH-P2-shape seed 0), then the addendum's BENCH-P2-shape seeds 1–2:
`… --sizes bp2 --bp2-seeds 0,1,2 --bp2-late-draws 20000 --configs <the 12 GL configurations>` and
`… --sizes bp2 --bp2-seeds 0,1,2 --configs mn_const,mn_sv,csv_conj,csv_struct,csv_conj325,csv_struct325`
(each run takes `--skip-started`, so a campaign resumes). The timing campaign, on a quiet machine with nothing else of
this pilot running: `… --only costs` (twice), `python -I benchmarks/pilot/phase3_gk.py --out DIR --sweeps --origins`
and `python -I benchmarks/phase3_perf.py --out DIR/bench3.json`. `python -I benchmarks/pilot/phase3_summarise.py DIR…`
prints the tables of this document, and with `--csv-prefix benchmarks/reports/phase3_pilot` writes the two committed
CSV files: `phase3_pilot_ess.csv` (bulk ESS, ESS per 1,000 sweeps, split R-hat, lag-1 autocorrelation, mean and s.d. of
every primary, extra, hyperprior-index and log-prior quantity of the 172 mixing runs, at each run's full length) and
`phase3_pilot_gate.csv` (every run's gate ladder). Every result is a JSON file with the machine fingerprint, the load average, the
top CPU processes and the other jobs' CPU share before and after the measurement; every script requires `--out`
(results never land in the working tree).

| Script | Measures |
| :--- | :--- |
| `phase3_common.py` | data (FRED-QD/FRED-MD medium windows, BENCH-P2's generator), models, the chain harness, the gate on prefixes, ESS tables, load snapshots, the load gate, the fingerprint |
| `phase3_mixing.py` | §2–§3: one 4-chain run per (configuration, size, seed), the gate ladder, ESS per 1,000 sweeps of every monitored and extra quantity; `--check`: the harness against `BVAR.fit` |
| `phase3_costs.py` | §4: every global step, the equation phase, the standalone, scanned and in-runner (`burn`) sweep, load-gated |
| `phase3_gk.py` | §6: the GK arms' ms per sweep at `T_eff` 77, 156, 236 and end-to-end origin times |
| `phase3_summarise.py` | the tables of this document from the JSON files |
| `phase3_run_all.py` | the campaign driver |
| `../phase3_perf.py` | §5: BENCH-P3 |

**The harness measures the production kernels.** Unlike Phase 2's pilot (clean-room prototypes), nothing here
re-implements a kernel. `phase3_common.run_chain` resolves the model (`engines.mcmc.resolve`), builds the runner
(`make_runner`), initialises and burns in with the runner's own `init` and `burn` executables in `run_chains`'s
pieces, zeroes the accumulators at the boundary, and then scans the production sweep (`make_sweep(program)`, keyed
by the absolute sweep index) recording the production `monitor_values` after every sweep, plus extra quantities
production does not monitor: WP30's reference group functions (`mean_log_v[g]`, the group scales, SSVS `N1[g]`) and
CommonSV's `mean h`, `mean |h|`. `phase3_mixing.py --check` (FRED-QD medium, seed 7, HS × SV, DL × Constant,
conjugate × CommonSV, SSVS(Beta) × Constant): (i) a public fit at 4 × 2,000 is **bitwise** the first 2,000 draws of
a public fit at 4 × 3,000 for every monitor field and configuration (the prefix property the ladder rests on); (ii)
the harness's 4 × 3,000 stream equals the public fit's bitwise for HS × SV and to rounding for the others (max
absolute difference 4e-11 on `a`, 1e-11 on `log_prior`, 7e-12 on the log-likelihood, ≤ 2e-13 on every other field:
the scan and the runner's `sample` executable compile to different XLA programs; no accept/reject decision differs);
(iii) the production gate on the harness's prefixes equals each public fit's `converged`.

## 0. Machine, load and versions

| Item | Value |
| :--- | :--- |
| Machine | Apple M5 Max, 18 cores (6 performance + 12 efficiency), 64 GB, macOS 26.6.2 (arm64) |
| Software | Python 3.12.13, NumPy 2.5.4 (BLAS/LAPACK: Accelerate), SciPy 1.18.1, `jax`/`jaxlib` 0.12.0.1 (CPU backend, 1 device); `jax_enable_x64` on inside the engine; `XLA_FLAGS` unset for every timing unless a row says otherwise |
| Data | FRED-QD and FRED-MD vintage 2026-09 from the checksum-verified local cache (`rai_bvar.datasets`, SPEC §5.8); BENCH-P2's generator `pilot_common.simulate_var_sv` |
| Load, mixing campaign | **not idle, by design**: 1-minute load average 8–97 (median 28.5, 176 runs) while 5–11 pilot workers shared the machine with other sessions' test runs. ESS, R-hat and the gate do not depend on the load; no timing from the mixing campaign is used in §4–§6 |
| Load, timing campaign | three rounds on 2026-10-28, with nothing of this pilot running alongside except where a row says so: round 1 (19:43–21:32) the costs, the GK sweeps and origins; round 2 (21:42–23:33) BENCH-P3 and the costs again; round 3 (23:33–23:50) the GK sweeps of the arms round 1 had contaminated, a state check, and the GK arms side by side (loaded by design). Other sessions' tests ran intermittently throughout. The idle gate opened at a 1-minute load of 1.3–2.0; every figure in §4–§6 is from a clean block, except BENCH-P3's 2 contaminated processes of 18 (§5) |
| Threading | The mixing workers after the first five BENCH-P2-shape runs used single-threaded XLA (`--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`); every timing used the default (multi-threaded) XLA unless a row says otherwise |

Every timing is the minimum over repeats of an AOT-compiled call timed with `block_until_ready`, on a burned-in
state. Before and after every timed block the JSON records `uptime`, the 1-minute load average, the top CPU processes
and the CPU share of the other jobs (processes other than the measuring one using at least 50% of a core; the
desktop's daemons stay below that and together use 100–130% at rest). An idle gate runs once per configuration (per
GK arm, per BENCH-P3 process) **before** the measuring process loads the machine: it waits until the 1-minute load
average is below 2. A block is **contaminated** when that gate load exceeded 2 or other jobs used more than 1.5 cores
before or after it. The load average *after* a block is recorded but does not decide: it includes the measured work
itself (a JAX process keeps 2–3 threads busy), so with the first version of the rule, which also tested it, every
block of more than a few seconds was contaminated by its own measurement (`10c362a`). Contaminated numbers are
marked and are upper bounds.

---

## 1. Decision rule (fixed before the campaign ran; see Status)

- **Gate.** P2 §8.2 unchanged: every primary quantity (P2 D-33 with P3 D-52's additions) has rank-normalised split
  R-hat < 1.01 and bulk ESS > 400, 4 chains, no numerical failure. It is evaluated by the production function
  `engines.mcmc.monitor.converged`.
- **Fits at budget `D`.** One run per (configuration, size, seed): 4 chains, 1,000 burn-in sweeps, `D_max` kept draws
  (`thin = 1`). The first `D` kept draws of every chain are exactly the draws of a fit with `draws = D` and the same
  seed, because the sweep keys depend only on the absolute sweep index (checked against `BVAR.fit`:
  `phase3_mixing.py --check`). Each seed is one independent fit at every `D` of the ladder
  `D ∈ {2,000, 5,000, 10,000, 20,000, 50,000}` kept draws per chain.
- **"The median fit passes" at `D`** ⇔ at least ⌈R/2⌉ of the `R` seeds pass the gate at `D` (`R = 5` at `n = 7`,
  `R = 3` at the BENCH-P2 shape). The recommended budget for D-58 is the smallest ladder `D` at which the median fit
  passes and keeps passing at every larger ladder `D`, per family × volatility × size. No budget is inferred from ESS
  alone: R-hat does not extrapolate, so a recommendation is always a `D` that was evaluated. Where no ladder `D` passes,
  the recommendation is "above 50,000", with the ESS-based projection stated as a projection.
- **ESS per 1,000 sweeps** = bulk ESS of the 4 chains ÷ (4·D/1000), that is per 1,000 sweeps of one chain; the median
  over seeds is reported, with the slowest primary quantity named.

### 1.1 Sizes, configurations and seeds

- `qd7`: FRED-QD medium set (`rai_bvar.datasets`, vintage 2026-09, the verified local cache), 1969Q1–2019Q4, `n = 7`,
  `p = 4`, `T_eff = 200`; seeds 0–4.
- `bp2`: BENCH-P2's shape and data (`simulate_var_sv(20, 4, 300, seed = 0)`, P2 §10.1); seeds 0–2.
- Configurations: the five family defaults (`Horseshoe()`, `NormalGamma()`, `R2D2()`, `DirichletLaplace()`, `SSVS()`)
  and `SSVS(p = Beta(1, 1))` × {`Constant()`, `SV()`}; `Minnesota(form="structural", hierarchy="fixed")` × {`Constant()`,
  `SV()`} as the Phase 2 reference; `Minnesota(form="conjugate", hierarchy="fixed") × CommonSV()` (the shared-factor path),
  `Minnesota(form="structural", hierarchy="fixed") × CommonSV()` (the per-equation program), and both with row 325's
  CommonSV priors (`(ρ+1)/2 ~ Beta(10, 2)`, `σ_h² ~ 0.1·χ²₁`).

### 1.2 Addendum (2026-10-28, after seed 0 at the BENCH-P2 shape, before seeds 1–2 there)

Seed 0 of every BENCH-P2-shape configuration ran to 4 × 50,000 first. Every GL configuration with a result failed the
gate at every ladder `D` up to 50,000, most of them far from it (slowest primary bulk ESS 6–93 at 50,000 against 400;
split R-hat up to 1.8). With seed 0 failing at `D`, the median of three seeds passes at `D` only if seeds 1 and 2 both
pass there. So seeds 1 and 2 of a configuration whose seed 0 fails at 50,000 with its slowest primary bulk ESS below
200 run to 4 × 20,000 only: their prefixes give `R = 3` at every `D ≤ 20,000`, and their ESS per 1,000 sweeps of the
slowest quantities shows whether two passes at 50,000 are plausible. At 50,000 such a configuration then has `R = 1`
and is reported as "above 50,000 (seed 0; seeds 1–2 run to 20,000)". Every other configuration runs seeds 1–2 to
50,000 as planned. Also added: the GK form (reduced, `scale="none"`, mean 0) × SV at `n = 21` on real data (FRED-QD
large set + `UMCSENTx`, 1967Q1–2019Q4, `T_eff = 208`), seed 0 to 4 × 20,000, as a transfer check for §9.10; and
R22's `T = 600` (FRED-MD medium, 1969M9–2019M12) and R29's `q ≈ T` (`n = 20, T = 120`, BENCH-P2's generator)
CommonSV runs. The mixing workers started after the first five BENCH-P2 runs use single-threaded XLA
(`phase3_run_all.py --single-thread`); ESS and R-hat are properties of the chain's law, not of the threading.

### 1.3 Runs added after the addendum (before they ran)

Two sets of runs were added after the first results and committed before they ran (`0f8b89c`, `9193dc3`): R18's
isolation runs, `NormalGamma(a=0.1)` and `DirichletLaplace(a=0.5)` × `Constant()` at `qd7` (5 seeds, 4 × 50,000;
`bayesianVARs`' fixed `a` in place of GK's grid, everything else as the defaults), and four reruns of seed 0 at `qd7`
(NG, R2D2, DL, SSVS(Beta) × `Constant()`) that record per-chain block means (the same chains; their gate and ESS
results equal the first runs'). They change no rule of §1.

---

## 2. Global-local mixing and the D-58 recommendation

### 2.1 The gate on prefixes

Seeds passing the P2 §8.2 gate at each budget (kept draws per chain, 4 chains, 1,000 burn-in sweeps), and the
budget the rule of §1 recommends. Configuration names: `<family>_<volatility>` with `const` = `Constant()`, `sv` =
`SV()`; `mn` the structural Minnesota reference (`hierarchy="fixed"`); `ssvs_beta` = `SSVS(p=Beta(1, 1))`; `ngfix`,
`dlfix` = `NormalGamma(a=0.1)`, `DirichletLaplace(a=0.5)` (R18's isolation runs, §2.3, not defaults). "Slowest primary"
is the primary quantity with the lowest ESS per 1,000 sweeps (median over seeds, at the length every seed has: 50,000 at
`qd7`, 20,000 for the GL rows at `bp2`).

**`qd7`** (FRED-QD medium, `n = 7`, `p = 4`, `T_eff = 200`; 5 seeds, 4 × 50,000 each):

| configuration | primaries | 2,000 | 5,000 | 10,000 | 20,000 | 50,000 | recommended | slowest primary (ESS/1k) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `mn_const` | 15 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 2,000 | `log_lam[GDPCTPI]` (709.8) |
| `mn_sv` | 36 | 0/5 | 0/5 | 2/5 | 5/5 | 5/5 | 20,000 | `sigma[COMPRNFB]` (12.8) |
| `hs_const` | 23 | 0/5 | 0/5 | 0/5 | 3/5 | 5/5 | 20,000 | `gl_log_zeta[3]` (5.1) |
| `hs_sv` | 44 | 0/5 | 0/5 | 0/5 | 0/5 | 4/5 | 50,000 | `gl_log_zeta[3]` (3.3) |
| `ng_const` | 23 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `gl_log_xi[7]` (0.2) |
| `ng_sv` | 44 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `gl_log_xi[7]` (0.3) |
| `ngfix_const` | 23 | 0/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5,000 | `B_own1[GDPC1]` (46.7) |
| `r2d2_const` | 23 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `gl_log_xi[5]` (0.2) |
| `r2d2_sv` | 44 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `gl_log_xi[7]` (0.3) |
| `dl_const` | 15 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `loglik` (0.3) |
| `dl_sv` | 36 | 0/5 | 0/5 | 1/5 | 4/5 | 3/5 | 20,000 | `B_own1[HOANBS]` (5.4) |
| `dlfix_const` | 15 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 2,000 | `B_own1[GDPC1]` (229.9) |
| `ssvs_const` | 15 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `loglik` (0.2) |
| `ssvs_sv` | 36 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `B_own1[HOANBS]` (0.5) |
| `ssvs_beta_const` | 23 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `loglik` (0.1) |
| `ssvs_beta_sv` | 44 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | > 50,000 | `B_own1[HOANBS]` (0.1) |

**`bp2`** (BENCH-P2's shape and data, `n = 20`, `p = 4`, `T_eff = 300`; seed 0 to 4 × 50,000, seeds 1–2 to 4 × 20,000
for the GL configurations and 4 × 50,000 for the others, per the addendum of §1):

| configuration | primaries | 2,000 | 5,000 | 10,000 | 20,000 | 50,000 | recommended | slowest primary (ESS/1k) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `mn_const` | 41 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2,000 | `log_lam[y13]` (615.4) |
| `mn_sv` | 101 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 20,000 | `sigma[y9]` (10.7) |
| `hs_const` | 49 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_zeta[7]` (0.3) |
| `hs_sv` | 109 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_zeta[7]` (0.3) |
| `ng_const` | 49 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_xi[3]` (0.2) |
| `ng_sv` | 109 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_xi[7]` (0.2) |
| `r2d2_const` | 49 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_xi[7]` (0.2) |
| `r2d2_sv` | 109 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `gl_log_xi[7]` (0.3) |
| `dl_const` | 41 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `loglik` (0.4) |
| `dl_sv` | 101 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `loglik` (0.8) |
| `ssvs_const` | 41 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `B_own1[y17]` (0.1) |
| `ssvs_sv` | 101 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `B_own1[y17]` (0.1) |
| `ssvs_beta_const` | 49 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `loglik` (0.1) |
| `ssvs_beta_sv` | 109 | 0/3 | 0/3 | 0/3 | 0/3 | 0/1 | > 50,000 | `B_own1[y18]` (0.1) |

### 2.2 ESS per 1,000 sweeps

Per configuration, the median over seeds of the slowest quantity of each field, at the length every seed has (50,000
at `qd7`; 20,000 for the GL rows at `bp2`). "Group scale" is the D-52 primary group quantity (HS `gl_log_zeta`, NG/R2D2
`gl_log_xi`, SSVS(Beta) `gl_p`); `gl_a_idx` (secondary) is the grid index of `a`; `gl.mean_log_v` is the group mean of
the local log variances (WP30's reference group function; not monitored in production); `log_prior` is secondary.

**`qd7`:**

| configuration | `loglik` | `B_own1` | `log_lam` | `mu` | `phi` | `sigma` | `h_T` | group scale | `gl_a_idx` | `gl.mean_log_v` | `log_prior` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mn_const` | 815 | 942 | 710 | — | — | — | — | — | — | — | 981 |
| `mn_sv` | 40.7 | 257 | — | 169 | 20.1 | 12.8 | 88.6 | — | — | — | 18.8 |
| `hs_const` | 24.1 | 108 | 252 | — | — | — | — | 5.1 | — | 5.1 | 7.5 |
| `hs_sv` | 32.4 | 113 | — | 160 | 18.1 | 12.6 | 99.7 | 3.3 | — | 3.3 | 14.0 |
| `ng_const` | 0.19 | 6.0 | 10.5 | — | — | — | — | 0.17 | 0.04 | 0.04 | 0.03 |
| `ng_sv` | 10.6 | 17.4 | — | 159 | 18.3 | 12.5 | 97.1 | 0.26 | 0.03 | 0.03 | 0.04 |
| `ngfix_const` | 53.6 | 46.7 | 160 | — | — | — | — | 234 | — | 14.1 | 14.9 |
| `r2d2_const` | 0.38 | 2.9 | 23.5 | — | — | — | — | 0.24 | 0.05 | 0.04 | 0.04 |
| `r2d2_sv` | 12.5 | 9.6 | — | 160 | 18.4 | 12.4 | 97.6 | 0.25 | 0.04 | 0.04 | 0.04 |
| `dl_const` | 0.31 | 2.7 | 17.3 | — | — | — | — | — | 0.03 | 0.03 | 0.04 |
| `dl_sv` | 12.3 | 5.4 | — | 156 | 18.2 | 12.3 | 95.8 | — | 0.03 | 0.03 | 0.04 |
| `dlfix_const` | 301 | 230 | 520 | — | — | — | — | — | — | 116 | 128 |
| `ssvs_const` | 0.37 | 0.31 | 0.63 | — | — | — | — | — | — | — | 0.48 |
| `ssvs_sv` | 2.3 | 0.55 | — | 87.9 | 13.9 | 2.1 | 16.1 | — | — | — | 16.0 |
| `ssvs_beta_const` | 0.06 | 0.11 | 0.12 | — | — | — | — | 0.13 | — | — | 0.08 |
| `ssvs_beta_sv` | 0.34 | 0.10 | — | 2.4 | 3.7 | 1.4 | 9.8 | 0.18 | — | — | 14.3 |

**`bp2`:**

| configuration | `loglik` | `B_own1` | `log_lam` | `mu` | `phi` | `sigma` | `h_T` | group scale | `gl_a_idx` | `gl.mean_log_v` | `log_prior` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mn_const` | 718 | 881 | 615 | — | — | — | — | — | — | — | 961 |
| `mn_sv` | 41.0 | 485 | — | 102 | 14.4 | 10.7 | 50.3 | — | — | — | 18.3 |
| `hs_const` | 7.3 | 36.1 | 64.0 | — | — | — | — | 0.29 | — | 0.29 | 0.43 |
| `hs_sv` | 13.9 | 34.8 | — | 67.4 | 13.5 | 10.4 | 65.3 | 0.30 | — | 0.31 | 0.52 |
| `ng_const` | 0.80 | 33.6 | 47.4 | — | — | — | — | 0.22 | 0.08 | 0.07 | 0.07 |
| `ng_sv` | 4.8 | 34.0 | — | 66.3 | 12.0 | 9.9 | 69.3 | 0.25 | 0.08 | 0.08 | 0.07 |
| `r2d2_const` | 1.4 | 45.3 | 59.5 | — | — | — | — | 0.24 | 0.07 | 0.07 | 0.08 |
| `r2d2_sv` | 12.5 | 27.6 | — | 67.0 | 12.4 | 10.5 | 68.7 | 0.26 | 0.07 | 0.07 | 0.08 |
| `dl_const` | 0.37 | 2.0 | 3.5 | — | — | — | — | — | 0.07 | 0.07 | 0.07 |
| `dl_sv` | 0.79 | 1.4 | — | 28.3 | 12.4 | 9.2 | 50.8 | — | 0.07 | 0.07 | 0.07 |
| `ssvs_const` | 0.10 | 0.08 | 0.12 | — | — | — | — | — | — | — | 0.12 |
| `ssvs_sv` | 0.16 | 0.09 | — | 0.50 | 6.6 | 2.4 | 3.7 | — | — | — | 9.8 |
| `ssvs_beta_const` | 0.08 | 0.08 | 0.12 | — | — | — | — | 0.11 | — | — | 0.08 |
| `ssvs_beta_sv` | 0.11 | 0.07 | — | 0.67 | 9.0 | 0.93 | 1.1 | 0.11 | — | — | 2.1 |

**Where ESS per 1,000 sweeps is not a rate.** When a chain mixes, the bulk ESS of the slowest primary grows in
proportion to the run: from 10,000 to 50,000 draws its median ratio over seeds is 5.4–5.9 for HS, Minnesota × SV and
CommonSV at `n = 7` (3.3–6 at the BENCH-P2 shape). For NG, R2D2 and DL × Constant (GK's grid on `a`) it is 0.9–1.5,
and 1.2–2.5 for SSVS(Beta) at both sizes and SSVS at the BENCH-P2 shape: the four chains do not reconcile in 50,000 sweeps (split R-hat stays at 1.04–1.07 for NG and
R2D2 from 10,000 to 50,000). Their ESS per 1,000 sweeps (0.03–0.5) bounds a mixing time from below and does not
project to a budget: "400 ÷ (4 × 0.1/1000) = 10⁶ draws per chain" would understate the problem.

### 2.3 What mixes slowly (R18; E163(c); E179 rows 304, 306, 308, 311, 315, 326, 327)

- **GK's grid on `a` (NG, R2D2, DL) nearly absorbs the shrunk groups.** The cross-lag groups (olcl-lagwise ids 1, 3, 5,
  7) move to the small end of the 1,000-point grid (ids 3, 5, 7: median index 20–45, `a ≈ 0.02–0.05`; id 1 at `n = 7`:
  74–105), their local variances follow (`mean_log_v` medians between −35 and −82, single seeds down to −100, so local
  variances `e^{−35}` to `e^{−100}`), and the chain all but
  stops there: `gl_a_idx` and `mean_log_v` mix at 0.03–0.1 per 1,000 sweeps with split R-hat 1.2–1.8 at 4 × 50,000,
  the primary `log ξ_g` at 0.1–1.0. The per-chain means of 10,000-sweep blocks (`chain_block_means` in the results)
  show slow excursions, for example NG × Constant's `gl_a_idx[3]` wandering between 14 and 60 and `mean_log_v[5]`
  between −20 and −73 from block to block. **The grid is the cause:** with `bayesianVARs`' fixed `a` and everything else
  unchanged (olcl-lagwise groups, `scale="psi"`), NG (`a = 0.1`) passes at 5,000 (5 of 5 seeds) and DL (`a = 0.5`) at
  2,000 (5 of 5), the group scales mix at 230–670 per 1,000 sweeps and `mean_log_v` at 14 (NG) and 116–150 (DL). This
  is R18's "small `a` on the grid", measured, and the slow `mean_log_v[g]` and `log p(θ)` of E179's grid rows 304,
  306, 308 and 326 at pilot sizes. It bears on Q23 (GK's semi-global defaults with hyperpriors, or `bayesianVARs`'
  fixed `a`; the design recommends GK's).
- **The horseshoe's cross-lag group scales are slow but mix.** HS has no grid; its slowest quantities are `log ζ_g` of
  the cross lags at lags 2–4 (ids 3, 5, 7): 5.1, 5.7, 10.4 per 1,000 sweeps at `n = 7` (the cross lags of lag 1: 28; the own-lag groups 17–83), and at
  the BENCH-P2 shape, whose DGP has cross lags `N(0, 0.02²)/l`, `log ζ ≈ −12.5` with lag-1 autocorrelation
  0.998–0.999 and 0.3–1.1 per 1,000 sweeps. HS's ESS grows with the run (ratio 5.5–5.9 at `n = 7`, 3.3–3.6 at
  `n = 20`). E163(c)'s "ESS 6 at 2 × 200" was this quantity in a short run; E179's rows 311 and 327 (HS global, reduced
  form) slow on `log ζ` is the same mechanism.
- **The gate cannot see DL's group level (D-52).** D-52 gives DL no primary group quantity (`gl_a_idx` is secondary,
  `mean_log_v` is not monitored). DL × SV at `n = 7` passes the gate at 4 × 20,000 (4 of 5 seeds) while `gl_a_idx` and
  `mean_log_v` of groups 1, 3, 5, 7 have split R-hat 1.3–1.8 and 0.03–0.1 ESS per 1,000 sweeps at 4 × 50,000, and the
  log prior has split R-hat 1.3–1.4: a "converged" DL fit whose group-level scales have not converged. Under
  `Constant` (and at `n = 20`) the slow level reaches the log-likelihood and `B_own1`, and the gate fails. This is the
  blind spot E179 found for row 308 (pooled ESS of `log p(θ)`), at pilot sizes.
- **SSVS's indicators rarely switch** (D-44's semiautomatic scales, `c0 = 0.01`, `c1 = 100`): the inclusion counts
  `N1[g]` mix at 0.04–0.9 per 1,000 sweeps at `n = 7` and 0.04–2.5 at `n = 20`; the primaries they move inherit it
  (`B_own1` and the log-likelihood 0.04–0.6; split R-hat 1.3–1.6 at 4 × 50,000 at the BENCH-P2 shape). In the per-chain block means of
  SSVS(Beta) × Constant at `n = 7`, one chain keeps one cross-lag-2 coefficient in the model (`N1[3] ≈ 1.0`) for the
  last 30,000 sweeps while the other three keep none. Under `p ~ Beta(1, 1)` the primary `gl_p` of the cross-lag groups
  mixes at 0.1–0.6. E179 row 315's slow quantities (`small_dev`, `gamma`, `N1`) at pilot sizes.

### 2.4 Recommendation for D-58 (Q21; the owner decides)

The smallest ladder budget at which the median fit passes the unchanged gate, per family (kept draws per chain, 4
chains):

| Family (default prior) | `Constant`, `n = 7` | `SV`, `n = 7` | `Constant`, `n = 20` | `SV`, `n = 20` |
| :--- | ---: | ---: | ---: | ---: |
| Horseshoe | **20,000** (3/5; 5/5 at 50,000) | **50,000** (4/5) | > 50,000 (ESS projection 200,000–350,000) | > 50,000 |
| NormalGamma (GK grid on `a`) | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing |
| R2D2 (GK grid on `a`) | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing |
| DirichletLaplace (GK grid on `a`) | > 50,000, no mixing | 20,000 (4/5), **gate blind** (§2.3) | > 50,000, no mixing | > 50,000 |
| SSVS (`p = 0.5`) | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing |
| SSVS (`p ~ Beta(1, 1)`) | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing | > 50,000, no mixing |
| *Reference: Minnesota (fixed)* | *2,000* | *20,000* | *2,000* | *20,000* |
| *Isolation: NG `a = 0.1`, DL `a = 0.5`* | *5,000, 2,000* | — | — | — |

"No mixing" means the slowest primary's ESS does not grow with the run (§2.2), so no budget can be read off it.

**What this means for D-58.** A default budget can make only the horseshoe's fits pass, and only at `n = 7`. For the
families with GK's grid on `a` and for SSVS, no default budget reaches the gate at either size; with the grid replaced
by a fixed `a`, NG and DL pass at the Constant default's order (2,000–5,000). The options, for the owner:

1. **D-58 = 4 × 20,000 kept draws for every GL composition** (recommended as the budget): the smallest ladder budget at
   which the median HS × Constant fit passes at `n = 7`, and the budget at which DL × SV and the Minnesota × SV
   reference pass there; twice D-35's SV default and ten times its Constant default. It does not make NG, R2D2, DL ×
   Constant or SSVS converge, and HS × SV needs 50,000 at `n = 7`; the D-35-style warning below it would name the
   budget, and the convergence warning names the group scales.
2. **Keep D-35's defaults** (10,000 with SV, 2,000 with Constant) and let GL fits report `converged = False` routinely.
3. Treat the non-mixing families as a prior-default and sampler question rather than a budget one (outside WP36):
   the grid on `a` (Q23: `bayesianVARs`' fixed `a`, or a joint move of `(a_g, local scales)` for R18), an indicator
   move for SSVS, and D-52's DL blind spot (a primary DL group quantity, such as `mean_log_v[g]` or the grid index).

The budget does not decide the GK replication's convergence either (§2.5, §6).

### 2.5 Transfer to the GK run (reduced form, `n = 21`)

The GK form (reduced, `scale="none"`, mean 0) × SV at `n = 21`, seed 0, 4 × 20,000, on real data (`qd21`: the FRED-QD
large set and `UMCSENTx`, 1967Q1–2019Q4, `T_eff = 208`) and on BENCH-P2's generator at `n = 21, T = 236`
(`sim21x236`, the last GK window's size):

| size | configuration | primaries | 2,000 | 5,000 | 10,000 | 20,000 | slowest primary (ESS/1k) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `qd21` | `red_mn_sv` | 106 | 0/1 | 0/1 | 0/1 | 0/1 | `sigma[INDPRO]` (4.0) |
| `qd21` | `red_hs_sv` | 114 | 0/1 | 0/1 | 0/1 | 0/1 | `B_own1[PCECTPI]` (0.1) |
| `qd21` | `red_ng_sv` | 114 | 0/1 | 0/1 | 0/1 | 0/1 | `B_own1[GDPCTPI]` (0.1) |
| `sim21x236` | `red_mn_sv` | 106 | 0/1 | 0/1 | 0/1 | 0/1 | `sigma[y20]` (8.4) |
| `sim21x236` | `red_hs_sv` | 114 | 0/1 | 0/1 | 0/1 | 0/1 | `gl_log_zeta[5]` (0.7) |
| `sim21x236` | `red_ng_sv` | 114 | 0/1 | 0/1 | 0/1 | 0/1 | `gl_log_xi[7]` (0.2) |

On real data the GL arms do not converge at 4 × 20,000: HS × SV has `B_own1[PCECTPI]` at split R-hat 1.55 and bulk ESS
7, NG × SV `B_own1[GDPCTPI]` at 1.30 and 10, with 21–28 of 114 primaries failing; the Minnesota × SV reference fails
on one `σ_i` (R-hat 1.010, ESS 316). The structural-form evidence transfers: at the GK run's size a default budget
does not make the GL arms pass the gate, and on real data the slow direction reaches the coefficients themselves.

---

## 3. CommonSV mixing

### 3.1 The gate and ESS per 1,000 sweeps

`csv_conj`: `Minnesota(form="conjugate", hierarchy="fixed") × CommonSV()`, the shared-factor path (P3 §5.3);
`csv_struct`: the structural Mode I Minnesota (`hierarchy="fixed"`) × `CommonSV()`, the per-equation program (§5.2);
`…325`: row 325's priors (`(ρ+1)/2 ~ Beta(10, 2)`, `σ_h² ~ 0.1·χ²₁`). Primaries (D-52): the log-likelihood, `B_own1`,
every `log λ_i`, `common_h_T`, `common_rho`, `common_sigma`. `csv.mean_h`, the mean of `h` over the data rows (R31's
level), is not monitored in production. Sizes as in §2, plus `sim20x120` (R29: BENCH-P2's generator at `n = 20, p = 4,
T = 120`, so `q = 100 ≈ T`; 3 seeds × 4 × 50,000) and `md7` (R22: FRED-MD medium, 1969M9–2019M12, `T_eff = 600`;
3 seeds × 4 × 20,000).

| size | configuration | primaries | 2,000 | 5,000 | 10,000 | 20,000 | 50,000 | recommended | slowest primary (ESS/1k) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `qd7` | `csv_conj` | 18 | 0/5 | 0/5 | 3/5 | 5/5 | 5/5 | 10,000 | `log_lam[GDPC1]` (15.6) |
| `qd7` | `csv_conj325` | 18 | 0/5 | 0/5 | 2/5 | 5/5 | 5/5 | 20,000 | `log_lam[GDPC1]` (12.8) |
| `qd7` | `csv_struct` | 18 | 0/5 | 0/5 | 0/5 | 1/5 | 5/5 | 50,000 | `log_lam[HOANBS]` (5.3) |
| `qd7` | `csv_struct325` | 18 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 | 50,000 | `log_lam[PCECC96]` (4.6) |
| `bp2` | `csv_conj` | 44 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 5,000 | `common_sigma` (38.5) |
| `bp2` | `csv_conj325` | 44 | 0/3 | 0/3 | 0/3 | 2/3 | 3/3 | 20,000 | `common_sigma` (7.5) |
| `bp2` | `csv_struct` | 44 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 5,000 | `log_lam[y15]` (25.9) |
| `bp2` | `csv_struct325` | 44 | 0/3 | 0/3 | 0/3 | 1/3 | 3/3 | 50,000 | `common_sigma` (7.3) |
| `sim20x120` (R29) | `csv_conj` | 44 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5,000 | `common_sigma` (57.9) |
| `sim20x120` (R29) | `csv_struct` | 44 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5,000 | `common_sigma` (57.6) |
| `md7` (R22) | `csv_conj` | 18 | 0/3 | 0/3 | 0/3 | 0/3 | — | > 20,000 | `log_lam[GS10]` (4.4) |
| `md7` (R22) | `csv_struct` | 18 | 0/3 | 0/3 | 0/3 | 0/3 | — | > 20,000 | `log_lam[CPIAUCSL]` (0.6) |

ESS per 1,000 sweeps (median over seeds of each field's slowest quantity):

| size | configuration | `loglik` | `B_own1` | `log_lam` | `common_h_T` | `common_rho` | `common_sigma` | `csv.mean_h` | `log_prior` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qd7` | `csv_conj` | 33.9 | 280 | 15.6 | 76.1 | 26.7 | 19.8 | 10.8 | 17.1 |
| `qd7` | `csv_conj325` | 36.4 | 235 | 12.8 | 79.0 | 22.9 | 21.9 | 8.5 | 16.4 |
| `qd7` | `csv_struct` | 40.6 | 657 | 5.3 | 15.0 | 11.0 | 18.4 | 4.5 | 16.7 |
| `qd7` | `csv_struct325` | 51.6 | 573 | 4.6 | 16.8 | 9.2 | 21.0 | 3.8 | 18.4 |
| `bp2` | `csv_conj` | 60.1 | 417 | 46.3 | 91.7 | 250 | 38.5 | 21.2 | 28.9 |
| `bp2` | `csv_conj325` | 56.2 | 403 | 42.6 | 49.0 | 16.9 | 7.5 | 19.5 | 9.5 |
| `bp2` | `csv_struct` | 607 | 893 | 25.9 | 46.6 | 229 | 39.0 | 14.5 | 55.0 |
| `bp2` | `csv_struct325` | 550 | 896 | 23.0 | 26.8 | 16.4 | 7.3 | 13.0 | 8.0 |
| `sim20x120` | `csv_conj` | 120 | 567 | 106 | 144 | 204 | 57.9 | 37.8 | 42.7 |
| `sim20x120` | `csv_struct` | 459 | 822 | 94.7 | 110 | 199 | 57.6 | 35.1 | 134 |
| `md7` | `csv_conj` | 14.2 | 33.1 | 4.4 | 61.9 | 9.0 | 12.3 | 3.6 | 9.2 |
| `md7` | `csv_struct` | 7.8 | 603 | 0.59 | 4.5 | 1.2 | 4.9 | 0.56 | 3.9 |

### 3.2 What this shows

- **Budget.** D-51's CommonSV default (4 × 10,000, as SV) is the rule's answer for the shared-factor path at `n = 7`
  (3 of 5 seeds; `log λ_i` binding), and 5,000 suffices at the BENCH-P2 shape and at `q ≈ T`. The per-equation program
  needs 4 × 50,000 at `n = 7` (1 of 5 at 20,000; `log λ_i` binding). Row 325's priors move the shared-factor path to
  20,000 at both sizes (at `n = 7` through `log λ_i`, at the BENCH-P2 shape through `σ_h`). At `T = 600` neither path
  passes at 4 × 20,000.
- **R31: the level shift `(λ_i e^c, h − c)` is CommonSV's slowest direction, and it slows with `T` (Q34).** `mean h`
  mixes at 10.8 (shared-factor) and 4.5 (per-equation) per 1,000 sweeps at `T = 200` (lag-1 autocorrelation 0.974 and
  0.989), at 21 and 15 at `T = 300` (BENCH-P2 shape) and at 3.6 and 0.56 at `T = 600` (0.991 and 0.997); `log λ_i`
  follows it (15.6 and 5.3 at `T = 200`; 4.4 and 0.59 at `T = 600`). On the per-equation program it decides the budget,
  as E170(b) found at 4 × 2,000 (ESS 21 against 160): here the per-equation `log λ_i` mixes 2–3 times slower than the
  shared-factor path's at `T ≤ 300` and 7 times slower at `T = 600`. Q34's interweaving shift move targets exactly this
  direction; on this evidence it is what the per-equation program (every `Linear × CommonSV()` except the conjugate
  screen) needs, and at `T = 600` the shared-factor path too.
- **R29 (the shared-factor two-block `(θ_i, λ_i)` Gibbs at `q ≈ T`) does not show at the pilot sizes.** At `q = 100,
  T = 120` the shared-factor `log λ_i` mixes at 106 per 1,000 sweeps (lag-1 autocorrelation ≤ 0.56), against 94.7 on
  the per-equation program, and both pass at 5,000; `B_own1`'s lag-1 autocorrelation is ≤ 0.19 at every size.
  BENCH-P3 (§5) adds `q = 500 > T = 300`. These runs give Q32 (a collapsed `λ_i` draw) no support.
- **E169: the centred `σ_h² | h, ρ` step slows as `ρ → 1`.** `σ_h` mixes at 19.8 per 1,000 sweeps at `n = 7`
  (posterior `ρ ≈ 0.93`, lag-1 autocorrelation 0.93) and 38 at the BENCH-P2 shape under the defaults, but at 7.5 under
  row 325's priors there, where the posterior has `ρ ≈ 0.999` and `σ_h ≈ 0.035` (autocorrelation 0.98). That is E169's
  mechanism, and the `σ_h`, `log σ_h` part of row 325's pooled-ESS failure (E179); its `mean |h|` part is the level
  direction above (`mean |h|` and `mean h` have the same ESS here). For the C1/C2 runs (Q41: 4 × 200,000) the margin at
  these sizes is 10–20 over the budget at which the median fit passes.
- **R22: the `h` acceptance falls with `T` but stays far above §8's 0.1 warning.** Mean block acceptance of the `h`
  independence MH (blocks of 100 dates): 0.56–0.68 at `T = 200`, 0.92–0.96 at `T = 300` (BENCH-P2 shape, small
  `σ_h`), 0.54 at `T = 600`; `ρ` acceptance 0.66–0.95. No run hit the Newton cap (`acc_csv_newton = 0` everywhere).

---

## 4. Costs (§10): the estimates replaced

BENCH-P2's shape and data (`n = 20, p = 4, T = 300`, `T_pad = 320`, 4 chains `vmap`ped, one device, default XLA
threading), on a state burned in for 500 sweeps. ms per call of all 4 chains: the equation phase and each global step
the minimum of 10 calls (every call a new sweep index, so the rejection-loop counts vary), the sweeps the minimum of
10 calls of 200 sweeps (`scan`: the production sweep in a `lax.scan`; `burn`: the runner's `burn` executable, a
`fori_loop` over a traced count with the state donated, BENCH-P2's definition). Two rounds; per block the lowest clean
value of the two (a contaminated one would be marked `*`, as an upper bound; §0's rule). Round 1 (19:43–20:45) ran
while another session's tests kept about one core busy; its GL blocks came out 1.4–1.6 times slower than round 2
(23:06–23:33) and many were contaminated (finding 14 of §7). Every round-2 block was clean, so every entry below is.

| configuration | path | equation phase | global steps | sweep, standalone | sweep, scanned | sweep, in `burn` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `csv_conj` (bp2) | shared_factor | 0.53 | common_sv 0.96 | 1.49 | 1.43 | 1.44 |
| `csv_conj` (bp2) | weighted | 2.06 | common_sv 0.94 | 3.02 | 2.99 | 2.99 |
| `csv_struct` (bp2) | weighted | 2.05 | common_sv 0.94 | 3.04 | 3.03 | 3.03 |
| `dl_const` (bp2) | weighted | 2.38 | gl_scales 10.11 | 12.43 | 13.28 | 13.29 |
| `dl_sv` (bp2) | weighted | 2.45 | gl_scales 10.56 | 13.73 | 14.08 | 14.19 |
| `hs_const` (bp2) | weighted | 2.41 | gl_scales 3.42 | 5.85 | 6.30 | 6.34 |
| `hs_sv` (bp2) | weighted | 2.38 | gl_scales 3.41 | 7.02 | 7.43 | 7.43 |
| `mn_const` (bp2) | eigen | 0.28 | — | 0.39 | 0.31 | 0.31 |
| `mn_const` (bp2) | cholesky_const | 1.42 | — | 1.51 | 1.40 | 1.38 |
| `mn_sv` (bp2) | weighted | 2.05 | — | 3.50 | 3.87 | 3.87 |
| `ng_const` (bp2) | weighted | 2.41 | gl_scales 4.69 | 7.74 | 8.03 | 8.07 |
| `ng_sv` (bp2) | weighted | 2.37 | gl_scales 4.73 | 8.89 | 9.17 | 9.16 |
| `r2d2_const` (bp2) | weighted | 2.42 | gl_scales 9.82 | 12.44 | 13.39 | 13.49 |
| `r2d2_sv` (bp2) | weighted | 2.41 | gl_scales 10.06 | 13.71 | 14.44 | 14.55 |
| `ssvs_beta_const` (bp2) | weighted | 2.40 | gl_scales 0.29 | 2.64 | 2.62 | 2.62 |
| `ssvs_beta_sv` (bp2) | weighted | 2.38 | gl_scales 0.30 | 4.38 | 4.35 | 4.35 |
| `ssvs_const` (bp2) | weighted | 2.39 | gl_scales 0.23 | 2.64 | 2.54 | 2.53 |
| `ssvs_sv` (bp2) | weighted | 2.39 | gl_scales 0.25 | 4.24 | 4.36 | 4.35 |

### 4.1 The GL scale step per family (replaces §10's 0.2–0.6 ms and E163(a))

| Family | step, `Constant` | step, `SV` | E163(a) (load ≈ 3) | share of the GL × `Constant` sweep |
| :--- | ---: | ---: | ---: | ---: |
| Horseshoe | 3.42 | 3.41 | 6.0 | 54% |
| NormalGamma | 4.69 | 4.73 | 7.8 | 58% |
| R2D2 | 9.82 | 10.06 | 17.6 | 73% |
| DirichletLaplace | 10.11 | 10.56 | 15.5 | 76% |
| SSVS (`p = 0.5`) | 0.23 | 0.25 | 0.28 | 9% |
| SSVS (`p ~ Beta(1, 1)`) | 0.29 | 0.30 | — | 11% |

- §10 estimated 0.2–0.6 ms (about 7 elementwise passes for the slowest GIG lane). The shipped step costs 3.4 ms (HS)
  to 10–10.6 ms (R2D2, DL) per sweep of 4 chains: **6–50 times** the estimate. E163(a)'s loaded figures were 1.2–1.8
  times these, about what this pilot's own round 1 measured with one other job on the machine. Neither of E163's
  `random.py` options has landed (the file is unchanged since WP28), so these are the shipped sampler's costs.
- The scale step is 54–76% of the GL × `Constant` sweep for the families with gamma and GIG lines; the equation phase
  (the deviation path, D-41) is the same for every family (2.37–2.45 ms, against 2.05 ms for Minnesota × SV's
  weighted path). SSVS's step costs 0.23–0.30 ms.

### 4.2 The CommonSV step, the shared-factor phase, the standalone and in-runner sweep (replaces E170(a), §10)

At E170's size (`n = 20, p = 4, T = 300`, 4 chains):

| Quantity | measured | E170(a) (loaded) | §10 estimate |
| :--- | ---: | ---: | ---: |
| `h, ρ, σ_h` step (`G_COMMON_SV`, `block = 100`) | 0.95 ms | 0.87 ms | — |
| equation phase, shared-factor path (conjugate × CommonSV) | 0.53 ms | 0.61 ms | — |
| equation phase, weighted path (the same posterior, `const_path="cholesky"`) | 2.06 ms | 2.20 ms | — |
| conjugate × CommonSV sweep, standalone / scanned / in `burn` | 1.49 / 1.43 / 1.44 ms | 1.52 / — / 3.88 ms | 0.3–0.6 ms |
| per-equation program (Mode I Minnesota × CommonSV) sweep, in `burn` | 3.03 ms | — | "what structural SV costs" |
| *Minnesota × SV (BENCH-P2's composition) sweep, in `burn`* | *3.87 ms* | — | — |

- **The runner's loop adds nothing.** E170(a) measured the conjugate CommonSV sweep at 3.88 ms inside the runner's
  `burn` loop against 1.52 ms standalone and called it "a Phase 2 runner property" (Mode II × Constant 0.33 ms
  standalone, 1.22 ms in `burn`). On an idle machine the `burn` executable and a `scan` of the production sweep agree
  within 1.5% for every configuration of §4's table, and single standalone sweeps differ from them only by the call's
  dispatch (Minnesota × Constant: 0.39 standalone, 0.31 in `burn`) or by being the cheapest of 10 sweeps whose
  rejection loops vary (the GL rows). E170's in-runner figure was a loaded-machine measurement.
- The shared-factor path makes the equation phase 3.9 times cheaper than the weighted path; with the `h` step the
  conjugate CommonSV sweep is 1.44 ms, 2.1–2.7 times cheaper than the per-equation program and the Minnesota × SV
  sweep. §10's 0.3–0.6 ms for the whole sweep is 2.4–5 times low: the `h` step alone costs 0.95 ms.


### 4.3 GL × Constant loses the eigen path (§10: "0.35 → ≈ 3 ms per sweep")

| Minnesota × Constant | sweep (in `burn`) | equation phase |
| :--- | ---: | ---: |
| eigen path (Mode I, the default) | 0.31 ms | 0.28 ms |
| `const_path="cholesky"` (`cholesky_const`) | 1.38 ms | 1.42 ms |
| a GL prior (the weighted deviation path, any family) | — | 2.37–2.42 ms |

A GL coefficient prior varies, so the eigen path is off (D-43): the equation phase alone costs 7.6–7.9 times the
eigen path's whole sweep (§10's "0.35 → ≈ 3 ms" holds to its order), and with the scale step a GL × `Constant` sweep
costs 2.5 ms (SSVS) to 13.5 ms (R2D2), 8–44 times the Minnesota × `Constant` sweep and, except for SSVS, more than
the Minnesota × SV sweep (3.87 ms).

---

## 5. BENCH-P3 (reported, not gated)

`python -I benchmarks/phase3_perf.py --out DIR/bench3.json` (3 repeats, each a cold and a warm process, the three
cases interleaved in every repeat so they share the load drift; 4 chains × 10,000 kept draws after 1,000, one device;
an idle gate before every process). Medians over repeats:

| Case | wall, cold | wall, warm | compile, cold | sampling (runner) | after the runner | ms per sweep (min 3 × 1,000) | peak RSS | gate (primaries failing; worst) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `commonsv_n100`: conjugate × CommonSV, `n = 100, p = 4, T = 300` | 153.9 s | 147.3 s | 4.5 s | 81 s (7.8 ms per sweep) | 62 s | 6.80 | 25–31 GiB | 3 of 204; `common_sigma` R-hat 1.024, ESS 116 |
| `hs_sv_bp2`: HS × SV, BENCH-P2 shape | 95.3 s | 90.2 s | 6.6 s | 79 s (7.7 ms) | 4.1 s | 7.47 | 3.3 GiB | 7 of 109; `gl_log_zeta[5]` 1.114, 29 |
| `bench_p2`: BENCH-P2 (same session) | 52.0 s | 49.7 s | 3.3 s | 43 s (4.1 ms) | 3.9 s | 3.89 | 2.9 GiB | 4 of 101; `sigma[y7]` 1.016, 510 |
| *BENCH-P2 reference baseline* (`tests/bench/baseline.json`) | *52.3 s* | *49.6 s* | *3.5 s* | — | *4.0 s* | *3.90* | *3.0 GiB* | *4 of 101* |

Contamination (§0): 0 of 6 processes for `hs_sv_bp2` and `bench_p2`, 2 of 6 for `commonsv_n100` (repeat 1, launched
at a load of 2.8 and 2.2); without them its cold median is 152.2 s against 153.9 s.

- **The conjugate CommonSV screen at `n = 100, p = 4, T = 300`** (`q = 500`, the D2 screen's headline size, on
  BENCH-P2's generator at `n = 100`) fits in 154 s (cold) and 147 s (warm) end to end, 6.8 ms per sweep of 4 chains. §10
  estimated 3–6 min for 4 × 11,000 sweeps. **It does not run with the default runtime**: its stored draws need
  5.3 GiB at the default `store_max = 2500` (monitor 1.6 GiB, parameters 3.7 GiB), above `Runtime.memory_budget_gb =
  2`, so the fit stops before sampling with `StorageBudgetError` (its message names the remedies). BENCH-P3 raises the
  runtime's budget to 8 GiB and changes nothing in the model. Of its 154 s, 81 s are sampling (7.8 ms per sweep in the runner) and 62 s come after the runner:
diagnostics and storage of 5,257 monitored quantities, 4,950 of them the secondary `a_ij`; the process peaks at
25–31 GiB of resident memory. The fit does not pass the gate at 4 × 10,000 (3 of 204 primaries: `common_sigma` R-hat
1.024, ESS 116; `common_rho` 1.019, 143; the log-likelihood 1.015, 214), on E169's slow `σ_h` (lag-1 autocorrelation
0.985).
- **HS × SV at the BENCH-P2 shape** costs 1.82 times BENCH-P2 in the same session (wall time, warm cache), from
  7.47 ms against 3.89 ms per sweep; BENCH-P2 itself reproduced its reference baseline
  (`tests/bench/baseline.json`: 52.3 s cold, 49.6 s warm, 3.90 ms per sweep) at 52.0 s cold, 49.7 s warm and 3.89 ms per sweep: the same machine as the baseline, within 1%.
- **R29 at `q = 500 > T = 300`**: the shared-factor `log λ_i` have lag-1 autocorrelation 0.59 (median over the 100 equations; at most
0.63) and bulk ESS ≥ 687 at 4 × 10,000 (17 per 1,000 sweeps), against 0.56 and 106 at `q = 100 ≈ T = 120`: no sign of
R29's two-block stickiness when `q` exceeds `T`; the binding quantities are `σ_h` and `ρ` (E169).

---

## 6. Compute estimate for the GK acceptance run (Q26; §9.10)

**The run.** 160 expanding-window origins 1980Q1–2019Q4 from the sample start 1960Q1 (`T_eff = 77, …, 236`); one fit
per origin and arm serves `h = 1` and the matching `h = 4` origins (the forecast of `h = 1..4` is one call); eight arms
— Minnesota-EB, HS (olcl-lagwise and global), NG (olcl-lagwise and global), R2D2, DL, SSVS(`p ~ Beta(1, 1)`) — in the
reduced form with `SV()` at `n = 21, p = 4`, `scale="none"`, mean 0. Warm-started backtest: the first origin of an arm
burns 1,000 sweeps, every later one `warm_burn = 250`, then `D` kept draws per chain, 4 chains.

**Measured inputs** (§0's hygiene; `phase3_gk.py`; BENCH-P2's generator at `n = 21`, so the shapes and the samplers are
the run's, the posterior is not):

| Arm (reduced × SV, `n = 21`) | `T_eff = 77` | 156 | 236 | 8 arms side by side, single-threaded (ratio) |
| :--- | ---: | ---: | ---: | ---: |
| Minnesota-EB | 4.07 | 5.28 | 7.07 | 1.50 |
| HS, olcl-lagwise | 9.52 | 10.59 | 12.09 | 1.36 |
| HS, global | 9.64 | 10.41 | 12.67 | 1.38 |
| NG, olcl-lagwise | 10.89 | 11.91 | 13.61 | 1.33 |
| NG, global | 10.46 | 11.51 | 13.02 | 1.38 |
| R2D2 | 16.00 | 17.30 | 18.56 | 1.34 |
| DL | 15.68 | 17.22 | 18.90 | 1.36 |
| SSVS (`p ~ Beta(1, 1)`) | no fit (D-44) | 6.97 | 8.58 | 1.42 |

ms per sweep of 4 chains, the runner's `burn` executable, minimum over rounds (default threading; two rounds for
Minnesota-EB, HS, NG global, R2D2, DL, whose first round was contaminated: within 7% of these, except Minnesota-EB
at `T_eff = 77`, 20% higher); the last column is the mean
over the three windows of ms per sweep with all eight arms running at once as single-threaded processes over ms per
sweep alone. Compilation of the `burn` executable: 1.0–2.5 s per arm and `T_pad` bucket.

- **Per origin beyond the sweeps**: one cold backtest origin end to end (`_run_origin`: resolution, EB for
  Minnesota-EB, compilation, sampling, diagnostics, storage, the `h = 1..4` forecast with 1,000 draws and the integrated
  LPD), at `T_eff` 77 and 232 and `D` 2,000 and 6,000, for HS and Minnesota-EB: 37.3, 44.5, 71.7, 88.9 s (HS at 77/2,000, 232/2,000, 77/6,000, 232/6,000) and 17.4, 26.5,
  32.2, 54.5 s (Minnesota-EB). A fit of `a + b·D`
  to (wall − sweeps × ms per sweep) gives `a ≈ 8.1 s` per origin (an upper bound inside a backtest, where the
  executables of a `T_pad` bucket compile once) and no per-draw term (`b ≈ −0.0005 s` per draw, the fit's sampling loop being
  slightly cheaper than the `burn` executable).
- **SSVS cannot fit the first 31 origins**: D-44's semiautomatic scales need `T_eff > k + n + 1 = 107` in the reduced
  form at `n = 21, p = 4`, so 1980Q1–1987Q3 raise at fit ("SSVS(semiautomatic=True) scales by the flat-prior posterior
  s.d., which needs more observations than coefficients (T = 77, needed > 107)"); the estimate counts its 129 feasible
  origins. WP40's pre-registration has to decide this arm (`semiautomatic=False`, a later start for it, or not met).

**Estimate** (`D` kept draws per chain, 4 chains):

| `D` (kept draws per chain) | all arms one after another (device-hours) | slowest arm (DL, R2D2) | 8 arms side by side, single-threaded (wall) |
| ---: | ---: | ---: | ---: |
| 2,000 | 11.6 h | 2.0 h | 2.8 h |
| 5,000 | 23.2 h | 4.3 h | 5.8 h |
| **10,000** (D-35's SV default) | **42.5 h** | 8.0 h | **10.9 h** |
| **20,000** (§2.4's D-58) | **81.1 h** | 15.4 h | **21.0 h** |
| 50,000 | 197 h | 37.8 h | 51 h |

Per origin: (1,000 sweeps at an arm's first origin, else 250, plus `D`) × ms per sweep, interpolated linearly in
`T_eff` between the measured windows, plus `a`; "side by side" multiplies each arm by its measured contention ratio
and takes the slowest. Not included: the persistent compilation cache's savings across buckets, and any rerun.

- **At the budgets on the table**: 42.5 device-hours at the D-35 SV default (4 × 10,000), 81 at 4 × 20,000 (§2.4's
  D-58), 197 at 4 × 50,000. The arms are independent, so they can run side by side as eight single-threaded processes
  (`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`; with the default threading, side
  by side processes collapse the reduced form, §7 item 13): measured contention 1.33–1.50 times, so **about 11 hours of
  wall time at 4 × 10,000 and 21 hours at 4 × 20,000** on this machine.
- **A documented quick configuration first** (Q26's recommendation): every 8th origin (20 origins), 4 × 2,000, cold
  starts: 1.8 device-hours in all, 27 minutes of wall time with the arms side by side; every 4th
  origin: 3.7 device-hours, 53 minutes.
- **The budget does not buy convergence here.** At the GK size the GL arms do not pass the gate at 4 × 20,000 on real
  data (§2.5: HS × SV split R-hat 1.55 on `B_own1`, NG × SV 1.30), and the families with GK's grid on `a` do not mix at
  any ladder budget (§2.2). If §9.10's "beats" goes through `compare()`'s convergence rule (SPEC §9.5: "≥ 95% of
  origins converged"), the GL criteria C1–C6 would fail on the gate whatever the budget, as Clark–Ravazzolo's
  criterion 6 did before its budget was raised (`benchmarks/reports/clark_ravazzolo_preregistration.md`); WP40's
  pre-registration has to state how the gate enters the verdicts before any of this compute is spent.

---

## 7. Findings

Each is a measurement with its evidence; nothing was changed because of it (the brief: report, do not fix).

1. **GK's grid on `a` makes NG, R2D2 and DL fits non-mixing (R18; Q23).** §2.3: the shrunk cross-lag groups sit at
   the small end of the grid (`a ≈ 0.02–0.05`) with local variances `e^{−35}`–`e^{−100}`; `gl_a_idx` and
   `mean_log_v` mix at 0.03–0.1 per 1,000 sweeps; the gate's slowest primary does not gain ESS with the run. With
   `bayesianVARs`' fixed `a` (NG 0.1, DL 0.5) the same compositions pass at 5,000 and 2,000. Inputs for R18 (a move
   for `(a_g, local scales)` jointly, or the fixed-`a` default) and for Q23, whose recommendation (GK's hyperpriors as
   the defaults) these runs do not support as the samplers stand.
2. **The production gate is blind to DL's group level (D-52; E179 row 308).** DL × SV passes at 4 × 20,000 at `n = 7`
   with `gl_a_idx` and `mean_log_v` at split R-hat 1.3–1.8 (§2.3). D-52 gives DL no primary group quantity.
3. **The horseshoe's cross-lag group scales mix slowly but finitely (R18; E163(c); E179 rows 311, 327).** `log ζ_g` of
   the cross lags at lags 2–4: 5–10 per 1,000 sweeps at `n = 7`, 0.3–1.1 at `n = 20` where the DGP's cross lags are
   near zero; HS × Constant passes at 4 × 20,000 and HS × SV at 4 × 50,000 at `n = 7`, neither at `n = 20` on the
   ladder.
4. **SSVS's indicators barely switch (E179 row 315).** `N1[g]` 0.04–2.5 per 1,000 sweeps; one chain keeps a different
   inclusion pattern for 30,000 sweeps; no SSVS configuration passes on the ladder (§2.3).
5. **CommonSV's level shift is its slowest direction and slows with `T` (R31; Q34; E170(b)).** `mean h` 10.8 → 3.6 per
   1,000 sweeps on the shared-factor path from `T = 200` to `T = 600`, 4.5 → 0.56 on the per-equation program, where
   `log λ_i` decides the budget (4 × 50,000 at `n = 7`; no pass at 4 × 20,000 at `T = 600`). §3.2.
6. **The centred `σ_h² | h, ρ` step slows as `ρ → 1` (E169; E179 row 325).** 7.5 per 1,000 sweeps under row 325's
   priors at `ρ ≈ 0.999`, against 20–38 elsewhere; at `n = 100` (BENCH-P3) `σ_h` and `ρ` fail the gate at 4 × 10,000
   (§3.2, §5).
7. **R29 does not show; R22 is far from its threshold.** The shared-factor `log λ_i` mix at 106 per 1,000 sweeps at
   `q ≈ T` and 17 at `q = 500 > T = 300`; the `h` block acceptance is 0.54 at `T = 600`; no Newton cap was hit (§3.2,
   §5). Q32 has no support from these runs.
8. **The GL scale step is 6–50 times §10's estimate (E163(a) confirmed in kind).** 3.4 ms (HS) to 10–10.6 ms (R2D2,
   DL) per sweep of 4 chains at `n = 20, p = 4`, 54–76% of the GL × `Constant` sweep; E163's loaded figures were
   1.2–1.8 times these (§4.1).
9. **E170(a)'s in-runner cost is not reproduced.** The conjugate CommonSV sweep costs the same in the runner's `burn`
   loop as standalone (§4.2), against E170's 3.88 ms in `burn` and 1.52 ms standalone; the runner adds nothing on an
   idle machine (the same for Minnesota × Constant, against the P2 figures E170 cites).
10. **The D2 screen at `n = 100` does not run with the default runtime.** `StorageBudgetError` before sampling (5.3 GiB
    stored at the default `store_max` against `Runtime.memory_budget_gb = 2`); with the budget raised it fits in about
    2.5 minutes, two fifths of which come after the runner (diagnostics and storage, 4,950 of the 5,257 monitored
    quantities the secondary `a_ij`), at a peak of 25–31 GiB resident (§5).
11. **SSVS cannot fit GK's first 31 origins (§9.10; WP40).** D-44's semiautomatic scales need `T_eff > 107` in the
    reduced form at `n = 21, p = 4`; 1980Q1–1987Q3 have `T_eff` 77–107 (§6).
12. **The GL arms of the GK run do not pass the gate at a default budget (§9.10; WP40; Q31).** §2.5, §6: at `n = 21`
    on real data HS × SV and NG × SV fail at 4 × 20,000 with split R-hat 1.55 and 1.30 on `B_own1`; with SPEC §9.5's
    "≥ 95% of origins converged" in `compare()`, the GK criteria would be decided by the gate.
13. **XLA's default threading collapses the reduced-form sweep under contention.** At `n = 21` (reduced × SV), with the
    machine oversubscribed (1-minute load 23–37: other sessions' tests and this pilot's mixing workers), the `burn`
    executable ran at 400–485 ms per sweep (Minnesota-EB), 810–850 ms (HS) and 1,210–1,250 ms (SSVS) with XLA's
    default intra-op threading, against 29–38 ms for HS single-threaded within the same half hour; on a quiet machine the two
    agree (HS 9.5–12.3 ms, Minnesota-EB 4.1–7.3 ms either way). The structural HS × `Constant` sweep at `n = 20` was not
    affected (about 18–21 ms both ways at a load of about 30). The engine runs BLAS single-threaded
    (`engines.mcmc._blas.single_thread`) but leaves XLA's intra-op pool at the machine's core count, so several MCMC
    processes side by side (the GK arms, a backtest with `n_jobs`, a shared CI runner) can slow the reduced form by
    20–100 times. (Scratch measurements of `phase3_gk.py` during the mixing campaign, 15:54–17:22; not part of §6.)
    Reported, not changed.
14. **Clean timings of the GL kernels need a quiet machine, not only a low load average.** On this M5 Max (6
    performance, 12 efficiency cores), round 1 of the cost campaign, with another session's job holding about one core,
    measured the GL scale step and the deviation-path equation phase 1.4–1.6 times slower than round 2, while the
    Minnesota and CommonSV sweeps agreed within 3% between the rounds; the HS × SV sweep costs the same after 500 and
    after 11,000 sweeps (7.40 and 7.42 ms; R2D2 × `Constant` 13.40 and 13.60 ms), so the difference is the machine,
    not the chain's state. Presumably the GL kernels' elementwise work, spread over XLA's intra-op threads, lands
    partly on efficiency cores when a performance core is taken (not checked: core placement needs `powermetrics`). Repeated rounds and their minima are the clean figure (§4).

---

## 8. Decisions and open choices for the owner

| Item | What WP36 recommends | Evidence |
| :--- | :--- | :--- |
| D-58 (Q21): the GL default budget | **4 × 20,000** kept draws for every GL composition, stated as what it is: the smallest ladder budget at which the median HS × `Constant` fit (and DL × SV, and the Minnesota × SV reference) passes at `n = 7`; it does not make NG, R2D2, DL × `Constant` or SSVS converge at any ladder budget, nor HS at `n = 20`. The alternatives: keep D-35's defaults and accept `converged = False`, or 4 × 50,000 (HS × SV's answer at `n = 7`) | §2.4 |
| Q23 / R18: GK's grid on `a` as the default hyperprior | Not decided here; the runs show the grid, not the families, makes NG and DL non-mixing (fixed `a`: 5,000 and 2,000) | §2.3, §7 item 1 |
| D-52: no primary group quantity for DL | A design question: the gate passes DL × SV fits whose group level has split R-hat 1.3–1.8 | §2.3, §7 item 2 |
| Q34: CommonSV interweaving shift move | Supported: the level shift is CommonSV's slowest direction, sets the per-equation program's budget (4 × 50,000 at `n = 7`) and slows with `T` | §3.2 |
| Q32: a collapsed `λ_i` draw | Not supported: R29 does not show at `q ≈ T` or `q = 500 > T` | §3.2, §5 |
| Q26: compute for the GK run | 42.5 device-hours at 4 × 10,000, 81 at 4 × 20,000, 197 at 4 × 50,000 (one device, arms in sequence), about 11 h and 21 h of wall time with the eight arms side by side as single-threaded processes; a quick configuration first (every 8th origin, 4 × 2,000: 1.8 device-hours) | §6 |
| WP40's pre-registration | Three things to settle before any score: the SSVS arm on the first 31 origins (D-44's `T_eff > 107`); how the convergence gate enters "beats" (the GL arms do not pass it at `n = 21`); and single-threaded XLA when arms run side by side | §6, §7 items 11–13 |
| The D2 screen's default runtime at `n = 100` | The default `Runtime.memory_budget_gb = 2` stops it before sampling; a documentation or default question | §5 |

## 9. Not measured here (and why)

- **The raw results are not committed** (as in Phase 2's pilot): the JSON files of every run stay outside the repository;
  the per-quantity ESS and the gate ladders are in `benchmarks/reports/phase3_pilot_{ess,gate}.csv`, the timings in
  this document. The secondary `a_ij`, intercepts and `h` at fixed dates are in the JSON only.

- **x86 and a second Apple machine**: only this M5 Max was available (as in Phase 2's pilot). The P-core/E-core
  sensitivity (§7 item 14) is specific to heterogeneous CPUs.
- **More than one seed at `n = 21`**: the GK transfer check (§2.5) is one seed per arm at 4 × 20,000 (its failures
  are far from the gate: split R-hat 1.30–1.55).
- **R2D2 and SSVS with the grid or the spike/slab replaced** (only NG and DL were isolated with a fixed `a`): R2D2 shares
  the grid mechanism; SSVS's slow quantity is the indicator, which a different `c0`/`c1` would change, a prior choice.
- **The GK run's end-to-end backtest** (warm starts across 160 origins): §6 composes it from measured per-sweep and
  per-origin costs; WP40's quick configuration measures it.
- **Multi-device runs** (`set_cpu_devices`, `shard_map`): not part of WP36's items; P2 measured 1.6–1.7× at BENCH-P2.
