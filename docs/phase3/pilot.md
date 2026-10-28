# Phase 3 pilot: GL and CommonSV mixing, costs, BENCH-P3 (WP36)

**Status.** Pilot of P3 DESIGN §8 and §10 (§13 WP36), run 2026-10-28 on branch `phase-3-wp36` from `main` at `8b5fce6`
(WP26–WP35 merged). It measures the **validated production kernels**: no kernel, prior, setting, tolerance or
criterion was changed, and nothing under `src/` was touched. It gives the budget evidence for D-58 (Q21, the owner's
decision), replaces the cost estimates of §10 (E163, E170) with idle-machine measurements, reports BENCH-P3, and
estimates the compute of the Gruber–Kastner acceptance run (§9.10; Q26). The decision rule of §1 was committed
(`ea50e09`) before any ladder result existed, and its addendum (`8d677b4`) before the runs it governs.

**Reproduce.** `python -I benchmarks/pilot/phase3_run_all.py --out DIR --only mixing --workers 5` (the mixing
campaign; ESS and R-hat do not depend on the load, so the workers may share the machine), then, on an idle machine and
with nothing else running, `--only costs`, `python -I benchmarks/pilot/phase3_gk.py --out DIR --sweeps --origins` and
`python -I benchmarks/phase3_perf.py --out DIR/bench3.json`. `python -I benchmarks/pilot/phase3_summarise.py DIR`
prints the tables. Every result is a JSON file with the machine fingerprint and the load average and top CPU processes
before and after the measurement; every script requires `--out` (results never land in the working tree).

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
| Software | Python 3.12.13, NumPy 2.5.4 (BLAS/LAPACK: Accelerate), SciPy 1.18.1, `jax`/`jaxlib` 0.12.0.1 (CPU backend, 1 device); `jax_enable_x64` on inside the engine; `XLA_FLAGS` unset for every timing |
| Data | FRED-QD and FRED-MD vintage 2026-09 from the checksum-verified local cache (`rai_bvar.datasets`, SPEC §5.8); BENCH-P2's generator `pilot_common.simulate_var_sv` |
| Load, mixing campaign | **not idle, by design**: 1-minute load average 8–97 (median 29) while 5–11 pilot workers shared the machine with other sessions' test runs. ESS, R-hat and the gate do not depend on the load; no timing from the mixing campaign is used in §4–§6 |
| Load, timing campaign | LOAD-TIMING |
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

## 1. Decision rule (fixed before any ladder result)

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

---

## 2. Global-local mixing and the D-58 recommendation

### 2.1 The gate on prefixes

Seeds passing the P2 §8.2 gate at each budget (kept draws per chain, 4 chains, 1,000 burn-in sweeps), and the
budget the rule of §1 recommends. Configuration names: `<family>_<volatility>` with `const` = `Constant()`, `sv` =
`SV()`; `mn` the structural Minnesota reference (`hierarchy="fixed"`); `ssvs_beta` = `SSVS(p=Beta(1, 1))`; `ngfix`,
`dlfix` = `NormalGamma(a=0.1)`, `DirichletLaplace(a=0.5)` (R18's isolation runs, §2.3, not defaults). "Slowest primary"
is the primary quantity with the lowest ESS per 1,000 sweeps (median over seeds).

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
and 1.2–2.5 for SSVS: the four chains do not reconcile in 50,000 sweeps (split R-hat stays at 1.04–1.07 for NG and
R2D2 from 10,000 to 50,000). Their ESS per 1,000 sweeps (0.03–0.5) bounds a mixing time from below and does not
project to a budget: "400 ÷ (4 × 0.1/1000) = 10⁶ draws per chain" would understate the problem.

### 2.3 What mixes slowly (R18; E163(c); E179 rows 304, 306, 308, 311, 315, 326, 327)

- **GK's grid on `a` (NG, R2D2, DL) nearly absorbs the shrunk groups.** The cross-lag groups (olcl-lagwise ids 1, 3, 5,
  7) move to the small end of the 1,000-point grid (median index 20–45, `a ≈ 0.02–0.05`), their local variances follow
  (`mean_log_v` between −35 and −100 at `n = 7`, so local variances `e^{−35}` to `e^{−100}`), and the chain all but
  stops there: `gl_a_idx` and `mean_log_v` mix at 0.03–0.1 per 1,000 sweeps with split R-hat 1.2–1.8 at 4 × 50,000,
  the primary `log ξ_g` at 0.1–0.5. The per-chain means of 10,000-sweep blocks (`chain_block_means` in the results)
  show slow excursions, for example NG × Constant's `gl_a_idx[3]` wandering between 14 and 60 and `mean_log_v[5]`
  between −20 and −73 from block to block. **The grid is the cause:** with `bayesianVARs`' fixed `a` and everything else
  unchanged (olcl-lagwise groups, `scale="psi"`), NG (`a = 0.1`) passes at 5,000 (5 of 5 seeds) and DL (`a = 0.5`) at
  2,000 (5 of 5), the group scales mix at 230–670 per 1,000 sweeps and `mean_log_v` at 14 (NG) and 116–150 (DL). This
  is R18's "small `a` on the grid", measured, and the slow `mean_log_v[g]` and `log p(θ)` of E179's grid rows 304,
  306, 308 and 326 at pilot sizes. It bears on Q23 (GK's semi-global defaults with hyperpriors, or `bayesianVARs`'
  fixed `a`; the design recommends GK's).
- **The horseshoe's cross-lag group scales are slow but mix.** HS has no grid; its slowest quantities are `log ζ_g` of
  the cross lags at lags 2–4 (ids 3, 5, 7): 5.1, 5.7, 10.4 per 1,000 sweeps at `n = 7` (own-lag groups 30–83), and at
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
  (`B_own1` and the log-likelihood 0.04–0.6, split R-hat up to 1.8 at 4 × 50,000). In the per-chain block means of
  SSVS(Beta) × Constant at `n = 7`, one chain keeps one cross-lag-2 coefficient in the model (`N1[3] ≈ 1.0`) for the
  last 30,000 sweeps while the other three keep none. Under `p ~ Beta(1, 1)` the primary `gl_p` of the cross-lag groups
  mixes at 0.1–0.6. E179 row 315's slow quantities (`small_dev`, `gamma`, `N1`) at pilot sizes.

### 2.4 Recommendation for D-58 (Q21; the owner decides)

The smallest ladder budget at which the median fit passes the unchanged gate, per family (kept draws per chain, 4
chains):

| Family (default prior) | `Constant`, `n = 7` | `SV`, `n = 7` | `Constant`, `n = 20` | `SV`, `n = 20` |
| :--- | ---: | ---: | ---: | ---: |
| Horseshoe | **20,000** (3/5; 5/5 at 50,000) | **50,000** (4/5) | > 50,000 (projection ≈ 200,000) | > 50,000 |
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
  these sizes is about 10 over the budget at which the median fit passes.
- **R22: the `h` acceptance falls with `T` but stays far above §8's 0.1 warning.** Mean block acceptance of the `h`
  independence MH (blocks of 100 dates): 0.56–0.68 at `T = 200`, 0.92–0.96 at `T = 300` (BENCH-P2 shape, small
  `σ_h`), 0.54 at `T = 600`; `ρ` acceptance 0.66–0.95. No run hit the Newton cap (`acc_csv_newton = 0` everywhere).

---

## 4. Costs (§10)

Timing campaign in progress; this section is filled from it.
