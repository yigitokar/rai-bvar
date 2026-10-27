# DGP study of the verdict engine: pre-registration

Written 2026-10-25, before any replication of the study was scored. It fixes the settings of the
DGP tests of SPEC §10.5 (size and power of `compare().verdict()` on simulated data) for the
pilot and for the full study. Both run with

```
uv run python benchmarks/dgp_study.py benchmarks/reports/dgp_study_pilot --config pilot \
    --workers 12 --cache ~/.cache/rai-bvar-dgp-<short commit>
uv run python benchmarks/dgp_study.py benchmarks/reports/dgp_study --config full \
    --workers 12 --cache ~/.cache/rai-bvar-dgp-<short commit>
```

on a clean working tree at the commit that adds this file. The report records that commit and
whether tracked files differ from it. It is labelled the pre-registered run only when every record
comes from one clean commit that contains this file, the design is the registered one (design key
`81f067e0cefaaea5`), every arm has its registered spec hash (below) and every runnable row has all
of the config's replications.

What was run before this file existed:

- one timing probe of single fits on a sample drawn outside the study's seeds (no score);
- a throughput probe of concurrent fits on noise data;
- the smoke configuration of the script, which has a study seed of its own (1, not 20261025),
  tiny chains and every 8th origin;
- the harness tests. They draw samples with the registered seed to check reproducibility and
  fit none of them, except one. A draft of the end-to-end test fitted two analytic arms
  (`BVAR(lags=1)` and `BVAR(lags=2)`, not arms of the study) on replication 0 of
  `linear-homoskedastic`, every 4th origin. That test now uses a seed of its own.

No arm of the study was fitted on a sample of the registered study before this commit.

## Why this study

SPEC §10.5 asks for the verdict engine's size (false `better` rate on null DGPs) and power on
simulated data. The validation matrix (SPEC §10.1) marks it "not implemented; at the Phase 3 exit
(§13)", and the Phase 4 exit needs it too (power on a threshold DGP, false positives on a
linear-SV DGP). This study measures the engine **as it is**: the verdict rule, its thresholds, the
priors, the sampler settings and the convergence gate are the library's, and none is changed to
make size or power look better.

## What is fixed

### Data-generating processes (`rai_bvar.validation.dgp.DGPS`)

One linear VAR(2) in three variables, in the model's recursive structural form (SPEC §4.1):

    y_t = c + B_1 y_{t−1} + B_2 y_{t−2} + u_t,   u_t = A⁻¹ e_t,   e_{i,t} = o_{i,t} exp(h_{i,t}/2) ε_{i,t},

with ε ~ N(0, I) and

| | Value |
| :-- | :-- |
| `c` | (0.3, 0.2, 0.1) |
| `B_1` (rows = equations) | (0.5, 0.1, 0.0); (0.2, 0.4, 0.1); (0.0, 0.2, 0.3) |
| `B_2` | diag(0.2, 0.1, 0.1) |
| `A` (unit lower triangular) | rows (1, 0, 0); (−0.4, 1, 0); (0.2, −0.3, 1) |
| Log-variance levels `μ` | (log 1, log 0.5, log 0.25) |

The spectral radius of the companion matrix is 0.82. Under constant volatility the reduced-form
error correlations are 0.49 (y1, y2), −0.15 (y1, y3) and 0.26 (y2, y3). The three DGPs differ only
in the volatility:

| DGP | `h_{i,t}` | `o_{i,t}` |
| :-- | :-- | :-- |
| `linear-homoskedastic` | `μ_i` | 1 |
| `linear-sv` | `μ_i + 0.95 (h_{i,t−1} − μ_i) + 0.2 η_{i,t}`, η ~ N(0, 1), stationary start | 1 |
| `linear-sv-outliers` | as `linear-sv` | 1 with probability 15/16, else uniform on {2, …, 20} |

The SV values (φ = 0.95, σ = 0.2) are moderate, persistent log-variance dynamics inside the bulk of
the library's SV prior. They were set before any run and not tuned. The outlier mechanism is that
of `SV(outliers=True)` (DESIGN D-9: the scale of the structural shock is uniform on {2, …, K} with
K = 20). Its rate 1/16 is that model's prior mean on quarterly data (`OutlierPrior(every_years=4)`).

Each replication simulates 100 burn-in periods from the unconditional mean, then keeps T = 200.
The sample is quarterly, starting in 1950Q2 and ending in 2000Q1, so it stays clear of 2020 and
its warnings. It is stored as cumulated levels with tcode 2, so the estimation sample is the
simulated `y` (to rounding, < 1e-13) and the library's default own-lag prior mean is 0
(SPEC §5.3), as for growth rates. No prior setting is passed.

### Rows, arms and expected verdicts (`rai_bvar.validation.dgp.ROWS`)

Each row is one DGP and one `compare()` over its arms. The arms are
`BVAR(vol=…, lags=2)` with every other setting at the library default. `Minnesota()` resolves to
the structural form with `hierarchy="eb"` (the D-12 plug-in), and `Constant("ig")` keeps the
Constant arm on the MCMC engine (Mode I), so every pair differs along the volatility axis only
(R2). The MCMC budgets are the per-composition defaults of D-35: 4 chains × 2,000 kept draws for
`Constant`, 4 × 10,000 for `SV` and `SV(outliers=True)`, with burn-in 1,000 and thin 1. The
convergence check is the library's (DESIGN §8.2: R-hat < 1.01 and bulk ESS > 400 on every primary
quantity). The arms' spec hashes (`spec_hash(model)[:16]`) are recorded per replication. The
report is labelled registered only if every record has these:

| Arm | Spec hash |
| :-- | :-- |
| Linear × Constant | `8aebe5adec44e966` |
| Linear × SV | `b3daa828453542ed` |
| Linear × SV(outliers) | `7f0aeb733eb2d07e` |

| Row | Arms (baseline first) | Scored pair | Expectation | Scored rate |
| :-- | :-- | :-- | :-- | :-- |
| `linear-homoskedastic` | Constant, SV, SV(outliers) | SV vs Constant | `not_better` | size: rate of `better` |
| | | SV(outliers) vs Constant | `not_better` | size: rate of `better` |
| `linear-sv` | Constant, SV | SV vs Constant | `better` | power: rate of `better` |
| `linear-sv-outliers` | SV, SV(outliers) | SV(outliers) vs SV | `not_worse` | rate of `worse` |

For the homoskedastic row the family-wise rate, any of its two pairs `better` in a replication, is
also reported. "SV(outliers=True) ≥ SV" is read as "not worse". The rate of `worse` is the scored
rate, and the rate of `better` there is reported as power.

### The pipeline per replication

- `rb.compare(models, data, origins=…, window=rb.Rolling(120), horizons=[1, 4], draws=1000,
  seed=…, baseline=…, cache=…)`: every argument not listed is the default. That includes
  `lpd_method="conditioned"` (the SPEC §7 estimator) and `n_jobs=1`.
- Forecast origins run every period from the first full window (1980Q1) to the second-to-last
  period (1999Q4): 80 origins at h = 1 and 77 at h = 4, where the outcome must exist.
- `verdict(axis="vol")` with its defaults: one-sided DM level 0.10, 25% MCS, |ΔLPD| > 3 MCSE, and
  converged fits at ≥ 95% of origins for both arms. This is the verdict the study scores.
- Secondary: `verdict(axis="vol", min_converged=0.0)`, the same rule with the convergence gate
  off. It is reported next to the verdict and labelled. It is not the engine's verdict and enters
  no reading.

### Seeds

- Study seed 20261025.
- DGP noise of replication `r`: `np.random.default_rng([20261025, crc32(DGP name), r])`.
- `compare()` seed: `default_rng([20261025, crc32(DGP name), r, 1]).integers(2**31)`. The
  per-origin fit and forecast seeds derive from it (DESIGN §11.6).
- Both depend on the DGP and the replication, not on the row. Later rows on the same DGP (the
  Phase 4 BART row on `linear-sv`) therefore see the same samples, and their shared arm is served
  from the score cache.

### Replications

- **Full study: 200 replications per row.** That is the lower end of SPEC §10.3's SBC range, the
  project's convention for "many replications". At a rate of 0.10 the Monte Carlo standard error
  is 0.021 and the Wilson 95% interval is about [0.066, 0.149]. At a power of 0.8 the standard
  error is 0.028.
- **Pilot: replications 0–23 of every row** (24 per row, 72 replications). Same design, seeds and
  cache keys, so the full study includes the pilot's replications and runs the other 176 per row.
  The number of replications is fixed in advance and is not changed after the pilot, so including
  them involves no optional stopping. At 24 replications a rate of 0.10 has a standard error of
  0.061, so the pilot's rates are indicative only.

## Reported quantities

Per row, scored pair and horizon (h = 1 and h = 4, each on its own):

1. The rate of the scored verdict, its binomial Monte Carlo standard error `√(p̂(1 − p̂)/R)` and its
   Wilson 95% interval. Replications are independent, so this is the Monte Carlo error of the
   rate. It includes the MCMC noise inside each replication.
2. The counts of `better`, `worse` and `inconclusive`, with the gate on (the verdict) and off
   (secondary).
3. Why each verdict came out as it did (`dgp.classify`, derived from the numbers the rule used
   and checked against the engine's gate-off verdict). For `better` or `worse`, the route that held:
   the valid DM test, the MCS separation, or both. For `inconclusive`: the gate alone, imprecise
   (|ΔLPD| ≤ 3 MCSE), or no route.
4. Mean and median ΔLPD (summed over origins), mean MCSE, and the share of replications whose two
   arms both pass the 95% gate.
5. Per arm: the mean and minimum converged share of origins, the gate-pass share, the seconds per
   fitted origin, and the primary quantities most often named worst when a fit misses the check.
6. The measured seconds per replication per row, and the full study's wall time extrapolated from
   them.

## Readings (fixed now)

- **`not_better` (size) and `not_worse` (rate of `worse`)**: the rate is **above nominal** only when
  its Wilson 95% lower bound exceeds 0.10, the one-sided DM level of the rule. Otherwise it is
  **not above nominal**, and the point estimate and interval are reported next to it. SPEC §10.5's
  "false-positive rate ≤ nominal level" is read this way.
- **`better` (power)**: descriptive. SPEC §10.5 sets no power threshold and none is invented here.
  The power threshold of the Phase 4 threshold-DGP exit will be pre-registered with that row.
- The pilot's readings are printed by the same code but are indicative. The full study's are the
  reading.

## Stated in advance

- **The compound rule has no exact nominal size.** `better` needs ΔLPD > 0, |ΔLPD| > 3 MCSE and
  either a one-sided DM rejection at 10% or "reference out of, model in" the 25% MCS, plus the
  convergence gate. The MCS route is a second test at a looser level, so the false-`better` rate
  can exceed 0.10 even when the DM test has its nominal size. The 3-MCSE condition and the gate
  push the other way. The study measures the net rate and does not adjust the rule.
- **The convergence gate can dominate.** SV fits at 4 × 10,000 do not always pass the check
  (DESIGN Q14, D-35: BENCH-P2 misses it on 4 of 101 primary quantities; the README example
  converged at 10 of 11 origins). With 80 origins, five failed fits make a replication
  `inconclusive` at h = 1, and four of the 77 at h = 4. Power may be low for that reason alone, and size may look good for the wrong
  reason. The `gate` class, the gate-off counts and the per-arm convergence table make that
  visible. The budget is not raised for this study.
- **h = 4 uses the default path-conditioned LPD.** It can degenerate for outcomes in a predictive's
  tail (DESIGN D-34: biased low, while its MCSE stays small). That is most likely on the SV and
  outlier DGPs. h = 1 does not depend on the estimator. The h = 4 numbers are reported as the
  default engine produces them.
- **Small-sample DM.** With 80 (h = 1) or 77 (h = 4) origins the HLN-corrected statistic uses
  h − 1 autocovariances (rectangular kernel). Overlapping rolling windows make the per-origin
  losses dependent within a replication. That dependence is part of what the size measures.
  Replications are independent.
- **One design cell.** n = 3, T = 200 and p = 2 are the lower end of SPEC §10.5's n = 3–7,
  T = 200–400. Other cells (n = 7, T = 400) are not part of this registration.
- **Wall time.** Before the pilot, two probes ran on the Apple M5 Max (6 performance and
  12 efficiency cores; other sessions kept the load average near 4). Warm single-process fits on
  a 150-observation window took about 0.13 s (Constant), 3.9 s (SV) and 6.5 s (SV(outliers)).
  Warm SV fits on 121 observations took 4.8 s single-process and 6.5 s each with 12 concurrent
  processes, about 8.9× the single-process throughput. That suggests roughly 1–1.5 h for the pilot and roughly 10 h for the full
  study with 12 workers. **The full study is run locally in this work only if the pilot's
  extrapolation is at most 4 hours.** Otherwise the PR reports the estimate and the command, and
  the full run is left for a later session or other hardware. The results do not depend on the
  worker count or the split (`--reps START:STOP`), only on the platform (SPEC §8).
- **No further run with other settings** is made after the pilot's scores are seen, except to fix
  a bug in the harness. A bug fix is documented, and the affected rows are rerun in full with these
  settings and a fresh cache.

## Not in this registration

The rows that need Phase 4–5 models are declared in `rai_bvar.validation.dgp.ROWS` and skipped as
pending:

- BART × SV vs Linear × SV on `linear-sv` (`not_better`);
- BART × SV and Threshold × SV vs Linear × SV on a threshold-SV DGP (`better`; separate rows);
- TVP × SV vs Linear × SV on a TVP-SV DGP (`better`).

Their DGPs (threshold, TVP) and readings are to be pre-registered before their first run.

## Addendum (2026-10-27, after the full run; no setting changed)

- **The 4-hour limit.** The 4-hour limit for a local full run (under "Wall time" above) was waived by the owner on 2026-10-25, after the pre-pilot estimate and before the pilot finished. The pilot then extrapolated 28 h remaining.
- **How the run went.** It ran from a clean detached checkout of this file's commit, `58ab09f`, with the settings above. It was paused once on the owner's decision (2026-10-25 18:11 EDT, at 92 of 600 replications) and resumed on the owner's word (2026-10-26 09:25 EDT).
- **Where it is recorded.** These events are the run's notes in `benchmarks/reports/dgp_study.md`. That report is labelled the pre-registered full run.
