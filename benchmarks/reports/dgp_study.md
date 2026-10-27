# DGP study of the verdict engine: full

**The pre-registered full run** (benchmarks/reports/dgp_study_preregistration.md): every record from commit `58ab09f99012` with a clean working tree, the registered design, all replications of every runnable row.

## Setup

- Design `81f067e0cefaaea5`: n = 3, T = 200, VAR(2) DGPs and models, `Rolling(120)`, origins every 1 period(s) from the first full window to the second-to-last period, horizons [1, 4], 1000 forecast draws per origin, MCMC budget the library default per composition (D-35), study seed 20261025.
- 200 replications per row; `compare()` and `verdict()` with their defaults (`{'test_level': 0.1, 'mcs_alpha': 0.25, 'mc_multiple': 3.0, 'min_converged': 0.95}`); nominal level 0.10 (the rule's one-sided DM level).
- Code: commit `58ab09f99012f825eccda604abd39d6bc942fbae`; rai-bvar 0.2.0, JAX 0.12.0.1.
- Machine: Apple M5 Max (18 CPUs: 6 performance, 12 efficiency), Darwin arm64; 12 worker processes; load averages 4.40 4.75 4.04 at the start and 3.99 10.60 44.61 at the end.
- Wall time of this invocation: 24.53 h (1472 min) for 508 replications (92 read from earlier records).
- Note: Started on the owner's approval of 2026-10-25, given after the pre-pilot estimate of about 10 h and before the pilot finished. The pre-registration's 4-hour limit for a local run was waived by that approval (the pilot extrapolated 28 h remaining at its load); every study setting is the pre-registered one.
- Note: Replications 0-23 of every row are the pilot's records (same commit, design and seeds), copied into this run's records before it started.
- Note: Paused on 2026-10-25 at 18:11 EDT on the owner's decision (about 28 h remained at the machine's load), after 20 of the 528 replications beyond the pilot (92 of 600 recorded). The 12 replications in flight were stopped; on resume they rerun, with their finished origins served from the score cache. Resumed with the same command, settings and worker count, from the same checkout at 58ab09f (moved during the pause to ~/.cache/rai-bvar-dgp-58ab09f/checkout; clean, same commit).
- Note: Resumed on 2026-10-26 at 09:25 EDT on the owner's word, with load averages 4.32 4.76 4.03 (1, 5, 15 min) and 86% of memory free; the Mac is shared with another session's test runs. The wall time below is this invocation's (from 92 of 600 recorded); the paused run's 20 replications took about 1.6 h.

## Size and power

Rate of the scored verdict over replications, ± its binomial Monte Carlo standard error, [Wilson 95% interval]. *Size* is the rate of a false `better` (expectation `not_better`), *power* the rate of `better` (expectation `better`); for `not_worse` the scored rate is that of `worse`. Reading (pre-registered): a size or `worse` rate is *above nominal* only when its interval lies above 0.10; power is descriptive. The gate-off columns are `verdict(min_converged=0)`: secondary, not the engine's verdict.

| Row | Pair | h | Expect | Scored | Rate (gate on) | Reading | better / worse / inconcl. (gate on) | better / worse / inconcl. (gate off) |
| :-- | :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 1 | not_better | better (200) | 0.000 ± 0.000 [0.000, 0.019] | not above nominal | 0 / 95 / 105 | 0 / 96 / 104 |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 4 | not_better | better (200) | 0.030 ± 0.012 [0.014, 0.064] | not above nominal | 6 / 39 / 155 | 6 / 39 / 155 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 1 | not_better | better (200) | 0.000 ± 0.000 [0.000, 0.019] | not above nominal | 0 / 190 / 10 | 0 / 193 / 7 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 4 | not_better | better (200) | 0.005 ± 0.005 [0.001, 0.028] | not above nominal | 1 / 177 / 22 | 1 / 180 / 19 |
| linear-sv | Linear × SV vs Linear × Constant | 1 | better | better (200) | 0.675 ± 0.033 [0.607, 0.736] | descriptive (power) | 135 / 2 / 63 | 142 / 2 / 56 |
| linear-sv | Linear × SV vs Linear × Constant | 4 | better | better (200) | 0.510 ± 0.035 [0.441, 0.578] | descriptive (power) | 102 / 7 / 91 | 108 / 7 / 85 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 1 | not_worse | worse (200) | 0.000 ± 0.000 [0.000, 0.019] | not above nominal | 81 / 0 / 119 | 199 / 0 / 1 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 4 | not_worse | worse (200) | 0.000 ± 0.000 [0.000, 0.019] | not above nominal | 39 / 0 / 161 | 102 / 0 / 98 |

Family-wise false-`better` rate (any `not_better` pair of the row `better`):

| Row | h | Gate on | Gate off |
| :-- | --: | :-- | :-- |
| linear-homoskedastic | 1 | 0.000 ± 0.000 [0.000, 0.019] | 0.000 ± 0.000 [0.000, 0.019] |
| linear-homoskedastic | 4 | 0.030 ± 0.012 [0.014, 0.064] | 0.030 ± 0.012 [0.014, 0.064] |

## Why the verdicts came out as they did

Counts per class (`rai_bvar.validation.dgp.classify`): `better:test` / `better:mcs` / `better:both` name the route of condition 3 that held (valid DM test, MCS separation, both); `gate` is inconclusive only because of the convergence gate; `imprecise` is |ΔLPD| ≤ 3 MCSE; `no_route` is precise but neither route separates the pair. Mean ΔLPD (summed over origins) and its mean MCSE; `gate pass` is the share of replications whose two arms both converged at ≥ 95% of origins.

| Row | Pair | h | Classes | Mean ΔLPD (median) | Mean MCSE | Mean origins | Gate pass |
| :-- | :-- | --: | :-- | :-- | --: | --: | --: |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 1 | gate 1, imprecise 45, no_route 59, worse:both 78, worse:mcs 17 | -1.00 (-1.13) | 0.22 | 80.0 | 0.99 |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 4 | better:both 3, better:mcs 2, better:test 1, imprecise 144, no_route 11, worse:both 30, worse:mcs 9 | -0.49 (-0.67) | 0.59 | 77.0 | 0.98 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 1 | gate 3, imprecise 5, no_route 2, worse:both 179, worse:mcs 11 | -4.42 (-4.60) | 0.24 | 80.0 | 0.98 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 4 | better:mcs 1, gate 3, imprecise 9, no_route 10, worse:both 164, worse:mcs 10, worse:test 3 | -6.48 (-6.89) | 0.65 | 77.0 | 0.98 |
| linear-sv | Linear × SV vs Linear × Constant | 1 | better:both 111, better:mcs 22, better:test 2, gate 7, imprecise 10, no_route 46, worse:both 1, worse:mcs 1 | +8.27 (+6.89) | 0.31 | 80.0 | 0.96 |
| linear-sv | Linear × SV vs Linear × Constant | 4 | better:both 76, better:mcs 26, gate 6, imprecise 46, no_route 39, worse:both 5, worse:mcs 2 | +6.59 (+5.17) | 0.73 | 77.0 | 0.95 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 1 | better:both 81, gate 118, no_route 1 | +27.96 (+23.83) | 1.52 | 80.0 | 0.41 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 4 | better:both 30, better:mcs 9, gate 63, imprecise 86, no_route 12 | +13.48 (+9.16) | 2.73 | 77.0 | 0.35 |

## Convergence and cost per arm

Converged share: the fraction of a replication's origins whose fit passed the convergence check (DESIGN §8.2); gate pass: the share of replications at ≥ 95%; worst: the primary quantities most often named worst by a failed check, with the number of failed fits (over all replications; both horizons share the fits).

| Row | Arm | h | Replications | Mean converged share | Min | Gate pass | s / fitted origin | Worst |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| linear-homoskedastic | Linear × Constant | 1 | 200 | 1.000 | 1.000 | 1.00 | 0.98 | – |
| linear-homoskedastic | Linear × Constant | 4 | 200 | 1.000 | 1.000 | 1.00 | 0.98 | – |
| linear-homoskedastic | Linear × SV | 1 | 200 | 0.994 | 0.675 | 0.99 | 11.37 | sigma[y3] 38, sigma[y2] 36, sigma[y1] 13 |
| linear-homoskedastic | Linear × SV | 4 | 200 | 0.994 | 0.662 | 0.98 | 11.37 | sigma[y3] 38, sigma[y2] 36, sigma[y1] 13 |
| linear-homoskedastic | Linear × SV(outliers) | 1 | 200 | 0.994 | 0.700 | 0.98 | 22.23 | sigma[y2] 43, sigma[y3] 29, sigma[y1] 17 |
| linear-homoskedastic | Linear × SV(outliers) | 4 | 200 | 0.994 | 0.688 | 0.98 | 22.23 | sigma[y2] 43, sigma[y3] 29, sigma[y1] 17 |
| linear-sv | Linear × Constant | 1 | 200 | 1.000 | 1.000 | 1.00 | 0.99 | – |
| linear-sv | Linear × Constant | 4 | 200 | 1.000 | 1.000 | 1.00 | 0.99 | – |
| linear-sv | Linear × SV | 1 | 200 | 0.987 | 0.875 | 0.96 | 11.36 | sigma[y1] 75, sigma[y3] 72, sigma[y2] 66 |
| linear-sv | Linear × SV | 4 | 200 | 0.987 | 0.870 | 0.95 | 11.36 | sigma[y1] 75, sigma[y3] 72, sigma[y2] 66 |
| linear-sv-outliers | Linear × SV | 1 | 200 | 0.970 | 0.688 | 0.85 | 11.27 | phi[y1] 120, phi[y2] 105, phi[y3] 97 |
| linear-sv-outliers | Linear × SV | 4 | 200 | 0.971 | 0.675 | 0.81 | 11.27 | phi[y1] 120, phi[y2] 105, phi[y3] 97 |
| linear-sv-outliers | Linear × SV(outliers) | 1 | 200 | 0.916 | 0.463 | 0.47 | 22.21 | sigma[y2] 469, sigma[y1] 431, sigma[y3] 422 |
| linear-sv-outliers | Linear × SV(outliers) | 4 | 200 | 0.917 | 0.442 | 0.42 | 22.21 | sigma[y2] 469, sigma[y1] 431, sigma[y3] 422 |

## Wall time of the full study

Measured per replication under this run's concurrency (12 workers), replications with no cached origin only:

| Row | Timed replications | Mean seconds per replication | Missing for the full study |
| :-- | --: | --: | --: |
| linear-homoskedastic | 195 | 2754 | 0 |
| linear-sv | 198 | 984 | 0 |
| linear-sv-outliers | 195 | 2667 | 0 |

- Full study (200 replications per row) from scratch: about **29.7 h** on 12 workers (356 h of worker time).
- Remaining after the records above: about **0.0 h**.
- Both are extrapolations at the same machine, load and worker count, not measurements.

## Rows not run (pending later phases)

- `linear-sv-bart` (Linear, SV: BART × SV does not beat Linear × SV (false-positive rate ≤ nominal)): pending (Phase 4): needs BART.
- `threshold-sv-bart` (Threshold, SV: BART × SV beats Linear × SV (power)): pending (Phase 4): needs BART, the 'threshold-sv' DGP simulator.
- `threshold-sv-threshold` (Threshold, SV: Threshold × SV beats Linear × SV (power)): pending (Phase 5): needs Threshold, the 'threshold-sv' DGP simulator.
- `tvp-sv` (TVP with sizeable drift, SV: TVP × SV beats Linear × SV): pending (Phase 5): needs TVP, the 'tvp-sv' DGP simulator.

## DGPs

- `linear-homoskedastic`: `{"name": "linear-homoskedastic", "intercept": [0.3, 0.2, 0.1], "lag_coefs": [[[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]], [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]], "contemp": [[1.0, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.2, -0.3, 1.0]], "log_var": [0.0, -0.6931471805599453, -1.3862943611198906], "vol": "constant", "phi": 0.95, "sigma": 0.2, "outlier_prob": 0.0, "outlier_max": 20, "burn": 100, "start": "1950Q1", "variables": ["y1", "y2", "y3"]}`
- `linear-sv`: `{"name": "linear-sv", "intercept": [0.3, 0.2, 0.1], "lag_coefs": [[[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]], [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]], "contemp": [[1.0, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.2, -0.3, 1.0]], "log_var": [0.0, -0.6931471805599453, -1.3862943611198906], "vol": "sv", "phi": 0.95, "sigma": 0.2, "outlier_prob": 0.0, "outlier_max": 20, "burn": 100, "start": "1950Q1", "variables": ["y1", "y2", "y3"]}`
- `linear-sv-outliers`: `{"name": "linear-sv-outliers", "intercept": [0.3, 0.2, 0.1], "lag_coefs": [[[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]], [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]], "contemp": [[1.0, 0.0, 0.0], [-0.4, 1.0, 0.0], [0.2, -0.3, 1.0]], "log_var": [0.0, -0.6931471805599453, -1.3862943611198906], "vol": "sv", "phi": 0.95, "sigma": 0.2, "outlier_prob": 0.0625, "outlier_max": 20, "burn": 100, "start": "1950Q1", "variables": ["y1", "y2", "y3"]}`
