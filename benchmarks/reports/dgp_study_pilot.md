# DGP study of the verdict engine: pilot

**The pre-registered pilot run** (benchmarks/reports/dgp_study_preregistration.md): every record from commit `58ab09f99012` with a clean working tree, the registered design, all replications of every runnable row.

This is the **pilot**: the first 24 replications per row of the full study, at its settings. Its rates are indicative only (see their Monte Carlo errors); the full study's reading is the one pre-registered.

## Setup

- Design `81f067e0cefaaea5`: n = 3, T = 200, VAR(2) DGPs and models, `Rolling(120)`, origins every 1 period(s) from the first full window to the second-to-last period, horizons [1, 4], 1000 forecast draws per origin, MCMC budget the library default per composition (D-35), study seed 20261025.
- 24 replications per row; `compare()` and `verdict()` with their defaults (`{'test_level': 0.1, 'mcs_alpha': 0.25, 'mc_multiple': 3.0, 'min_converged': 0.95}`); nominal level 0.10 (the rule's one-sided DM level).
- Code: commit `58ab09f99012f825eccda604abd39d6bc942fbae`; rai-bvar 0.2.0, JAX 0.12.0.1.
- Machine: Apple M5 Max (18 CPUs: 6 performance, 12 efficiency), Darwin arm64; 12 worker processes; load averages 5.56 5.44 9.02 at the start and 31.90 39.41 66.44 at the end.
- Wall time of this invocation: 4.21 h (253 min) for 72 replications (0 read from earlier records).

## Size and power

Rate of the scored verdict over replications, ± its binomial Monte Carlo standard error, [Wilson 95% interval]. *Size* is the rate of a false `better` (expectation `not_better`), *power* the rate of `better` (expectation `better`); for `not_worse` the scored rate is that of `worse`. Reading (pre-registered): a size or `worse` rate is *above nominal* only when its interval lies above 0.10; power is descriptive. The gate-off columns are `verdict(min_converged=0)`: secondary, not the engine's verdict.

| Row | Pair | h | Expect | Scored | Rate (gate on) | Reading | better / worse / inconcl. (gate on) | better / worse / inconcl. (gate off) |
| :-- | :-- | --: | :-- | :-- | :-- | :-- | :-- | :-- |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 1 | not_better | better (24) | 0.000 ± 0.000 [0.000, 0.138] | not above nominal | 0 / 14 / 10 | 0 / 14 / 10 |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 4 | not_better | better (24) | 0.042 ± 0.041 [0.007, 0.202] | not above nominal | 1 / 6 / 17 | 1 / 6 / 17 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 1 | not_better | better (24) | 0.000 ± 0.000 [0.000, 0.138] | not above nominal | 0 / 23 / 1 | 0 / 23 / 1 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 4 | not_better | better (24) | 0.000 ± 0.000 [0.000, 0.138] | not above nominal | 0 / 21 / 3 | 0 / 21 / 3 |
| linear-sv | Linear × SV vs Linear × Constant | 1 | better | better (24) | 0.583 ± 0.101 [0.388, 0.755] | descriptive (power) | 14 / 1 / 9 | 15 / 1 / 8 |
| linear-sv | Linear × SV vs Linear × Constant | 4 | better | better (24) | 0.417 ± 0.101 [0.245, 0.612] | descriptive (power) | 10 / 3 / 11 | 10 / 3 / 11 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 1 | not_worse | worse (24) | 0.000 ± 0.000 [0.000, 0.138] | not above nominal | 12 / 0 / 12 | 24 / 0 / 0 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 4 | not_worse | worse (24) | 0.000 ± 0.000 [0.000, 0.138] | not above nominal | 8 / 0 / 16 | 14 / 0 / 10 |

Family-wise false-`better` rate (any `not_better` pair of the row `better`):

| Row | h | Gate on | Gate off |
| :-- | --: | :-- | :-- |
| linear-homoskedastic | 1 | 0.000 ± 0.000 [0.000, 0.138] | 0.000 ± 0.000 [0.000, 0.138] |
| linear-homoskedastic | 4 | 0.042 ± 0.041 [0.007, 0.202] | 0.042 ± 0.041 [0.007, 0.202] |

## Why the verdicts came out as they did

Counts per class (`rai_bvar.validation.dgp.classify`): `better:test` / `better:mcs` / `better:both` name the route of condition 3 that held (valid DM test, MCS separation, both); `gate` is inconclusive only because of the convergence gate; `imprecise` is |ΔLPD| ≤ 3 MCSE; `no_route` is precise but neither route separates the pair. Mean ΔLPD (summed over origins) and its mean MCSE; `gate pass` is the share of replications whose two arms both converged at ≥ 95% of origins.

| Row | Pair | h | Classes | Mean ΔLPD (median) | Mean MCSE | Mean origins | Gate pass |
| :-- | :-- | --: | :-- | :-- | --: | --: | --: |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 1 | imprecise 7, no_route 3, worse:both 12, worse:mcs 2 | -1.00 (-1.19) | 0.22 | 80.0 | 1.00 |
| linear-homoskedastic | Linear × SV vs Linear × Constant | 4 | better:mcs 1, imprecise 17, worse:both 6 | -0.60 (-0.68) | 0.61 | 77.0 | 1.00 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 1 | imprecise 1, worse:both 22, worse:mcs 1 | -4.47 (-4.59) | 0.24 | 80.0 | 1.00 |
| linear-homoskedastic | Linear × SV(outliers) vs Linear × Constant | 4 | imprecise 1, no_route 2, worse:both 19, worse:mcs 1, worse:test 1 | -6.58 (-6.58) | 0.69 | 77.0 | 1.00 |
| linear-sv | Linear × SV vs Linear × Constant | 1 | better:both 11, better:mcs 3, gate 1, imprecise 4, no_route 4, worse:both 1 | +6.44 (+5.87) | 0.29 | 80.0 | 0.96 |
| linear-sv | Linear × SV vs Linear × Constant | 4 | better:both 9, better:mcs 1, imprecise 5, no_route 6, worse:both 3 | +4.74 (+4.27) | 0.65 | 77.0 | 0.96 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 1 | better:both 12, gate 12 | +26.79 (+20.30) | 1.43 | 80.0 | 0.50 |
| linear-sv-outliers | Linear × SV(outliers) vs Linear × SV | 4 | better:both 6, better:mcs 2, gate 6, imprecise 9, no_route 1 | +15.93 (+11.97) | 2.70 | 77.0 | 0.46 |

## Convergence and cost per arm

Converged share: the fraction of a replication's origins whose fit passed the convergence check (DESIGN §8.2); gate pass: the share of replications at ≥ 95%; worst: the primary quantities most often named worst by a failed check, with the number of failed fits (over all replications; both horizons share the fits).

| Row | Arm | h | Replications | Mean converged share | Min | Gate pass | s / fitted origin | Worst |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| linear-homoskedastic | Linear × Constant | 1 | 24 | 1.000 | 1.000 | 1.00 | 1.01 | – |
| linear-homoskedastic | Linear × Constant | 4 | 24 | 1.000 | 1.000 | 1.00 | 1.01 | – |
| linear-homoskedastic | Linear × SV | 1 | 24 | 0.996 | 0.963 | 1.00 | 12.26 | sigma[y2] 7 |
| linear-homoskedastic | Linear × SV | 4 | 24 | 0.996 | 0.961 | 1.00 | 12.26 | sigma[y2] 7 |
| linear-homoskedastic | Linear × SV(outliers) | 1 | 24 | 0.997 | 0.975 | 1.00 | 23.94 | sigma[y2] 5 |
| linear-homoskedastic | Linear × SV(outliers) | 4 | 24 | 0.997 | 0.974 | 1.00 | 23.94 | sigma[y2] 5 |
| linear-sv | Linear × Constant | 1 | 24 | 1.000 | 1.000 | 1.00 | 1.02 | – |
| linear-sv | Linear × Constant | 4 | 24 | 1.000 | 1.000 | 1.00 | 1.02 | – |
| linear-sv | Linear × SV | 1 | 24 | 0.985 | 0.925 | 0.96 | 12.34 | sigma[y1] 14, sigma[y3] 12, sigma[y2] 3 |
| linear-sv | Linear × SV | 4 | 24 | 0.986 | 0.935 | 0.96 | 12.34 | sigma[y1] 14, sigma[y3] 12, sigma[y2] 3 |
| linear-sv-outliers | Linear × SV | 1 | 24 | 0.966 | 0.838 | 0.79 | 12.33 | phi[y2] 32, phi[y1] 13, sigma[y2] 9 |
| linear-sv-outliers | Linear × SV | 4 | 24 | 0.965 | 0.831 | 0.75 | 12.33 | phi[y2] 32, phi[y1] 13, sigma[y2] 9 |
| linear-sv-outliers | Linear × SV(outliers) | 1 | 24 | 0.929 | 0.463 | 0.62 | 23.52 | sigma[y1] 82, sigma[y3] 28, sigma[y2] 26 |
| linear-sv-outliers | Linear × SV(outliers) | 4 | 24 | 0.928 | 0.442 | 0.62 | 23.52 | sigma[y1] 82, sigma[y3] 28, sigma[y2] 26 |

## Wall time of the full study

Measured per replication under this run's concurrency (12 workers), replications with no cached origin only:

| Row | Timed replications | Mean seconds per replication | Missing for the full study |
| :-- | --: | --: | --: |
| linear-homoskedastic | 24 | 2978 | 176 |
| linear-sv | 24 | 1070 | 176 |
| linear-sv-outliers | 24 | 2868 | 176 |

- Full study (200 replications per row) from scratch: about **32.0 h** on 12 workers (384 h of worker time).
- Remaining after the records above: about **28.2 h**.
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
