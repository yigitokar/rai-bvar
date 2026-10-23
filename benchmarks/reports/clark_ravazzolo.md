# Clark–Ravazzolo (2015) replication on FRED-QD (WP23)

Generated 2026-10-23 04:19 UTC by `benchmarks/clark_ravazzolo.py` (config `full`, rai-bvar 0.0.1, jax 0.12.0.1, Darwin arm64, python 3.12.13). Run time of this invocation 1.10 h (cached origins are read, not refitted).

## Acceptance criteria (N08 §C5)

| # | Met | Operational rule | Result |
| :- | :- | :- | :- |
| 1 | **met** | ΔLPS(SV(rw) − Constant) > 0 at h = 1, 2 for all four variables in the full and pre-crisis samples, and the joint ΔLPD at h = 1 > 0 with one-sided DM/HLN p < 0.10 in both samples | 16 of 16 cells positive; joint h=1: full +0.809 nats/quarter (p=0.000), pre +0.790 nats/quarter (p=0.000) |
| 2 | **met** | CRPS ratio SV(rw)/Constant < 1 at h = 1, 2 for all four variables (full sample) | 8 of 8 cells < 1; pre-crisis: all 8 < 1 |
| 3 | **met** | RMSE ratio SV(rw)/Constant ≤ 1.02 ('about 1') at h = 1, 2 for all four variables (full sample); CR's gains for GDP and the bill are 5–10% | GDP/h1 0.909, GDP/h2 0.899, INFL/h1 0.898, INFL/h2 0.850, UR/h1 0.970, UR/h2 0.948, TBILL/h1 0.937, TBILL/h2 0.950 |
| 4 | **not met** | full sample: ΔLPS(h=8) < ΔLPS(h=1) for UR and TBILL; pre-crisis: ΔLPS(h=8) > 0 for GDP and INFL | full UR: h1 +0.159 → h8 +1.954; full TBILL: h1 +0.378 → h8 -0.207; pre-crisis GDP h8 +0.108; pre-crisis INFL h8 +0.295. Secondary (not part of the rule): full UR h8: median +0.099 (largest origins: 2008Q2 +41.6, 2007Q4 +41.4, 2008Q3 +40.4), Gaussian-approx h1 +0.125 → h8 -0.499; full TBILL h8: median -0.023 (largest origins: 2000Q2 -3.1, 1999Q4 -2.3, 2000Q1 -2.1), Gaussian-approx h1 +0.322 → h8 -0.234; pre-crisis GDP h8 Gaussian-approx -0.004; pre-crisis INFL h8 Gaussian-approx +0.345; RB estimate degenerate (joint lpd_mcse ≥ 0.9; see 'Reliability of the RB log score') at full UR h8: 12 origin(s) (RB and Gaussian-approx ΔLPS differ in sign), full TBILL h8: 12 origin(s), pre-crisis GDP h8: 8 origin(s) (RB and Gaussian-approx ΔLPS differ in sign), pre-crisis INFL h8: 8 origin(s) |
| 5 | **met** | neither SV(rw,t10) nor AR-SV(0.9) is better than both SV(rw) and SV (one-sided DM/HLN p < 0.10, positive ΔLPD) at ≥ 3 of 4 joint horizons or ≥ 8 of 16 marginal cells (full sample) | SV(rw,t10): significantly better than both SV(rw) and SV at 0/4 joint horizons and 0/16 marginal cells; AR-SV(0.9): significantly better than both SV(rw) and SV at 0/4 joint horizons and 0/16 marginal cells |
| 6 | **not met** | in every one of the 24 orderings: compare() verdict SV(rw) vs Constant at h = 1 is 'better', and the one-sided DM/HLN p-value of the marginal log score is < 0.10 for GDP and TBILL at h = 1 | verdict 'better' (convergence gate on, as compare() returns it) in 0/24 orderings; with the gate off in 24/24; DM GDP and TBILL < 0.10 in 24/24 |

## Design

* Data: FRED-QD 2026-09 (SHA-256 verified), GDP = 100·Δlog GDPC1, INFL = 100·Δlog GDPCTPI, UR = UNRATE, TBILL = TB3MS (levels); non-annualised quarterly rates; estimation sample starts 1965Q1; pseudo real time on the final vintage (D4).
* Asserted (§C6): FRED-QD UNRATE/TB3MS equal within-quarter means of FRED-MD monthly values of the same vintage — UNRATE: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2); TB3MS: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2).
* Asserted (R9, §C6): at origins 1985Q1, 2007Q4, 2011Q2 the resolved priors, hyperparameters and padded data are bitwise identical when the vintage is cut at the origin (no look-ahead; CR's last-vintage SV training sample is not replicated).
* Origins: 1985Q1–2011Q2 (106, full) and 1985Q1–2007Q4 (92, pre-crisis subset); expanding window; h = 1, 2, 4, 8.
* MCMC: 4 chains × 5000 kept draws, burn 1000, thin 1; 1000 forecast draws per origin; backtest seed 0; per-origin fit seeds (DESIGN §11.6); cold starts.
* Mean prior (all models): `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, alpha=2, cross=1, mean=0)`, p = 4. Volatility arms: `Constant` = `Constant(iw_equivalent)`, `SV(rw)` = `SV(rw)`, `SV` = `SV`, `SV(rw,t10)` = `SV(rw,t, nu_prior=7eca60, t_scale=unit)`, `AR-SV(0.9)` = `SV(phi_prior=7ea4a9)`.
* LPD is the library's Rao–Blackwellised predictive density (primary); `ΔLPS gauss` is CR's Gaussian approximation (Adolfson et al. 2007) from the same draws.

### Reliability of the RB log score

The primary LPD is the library's Rao–Blackwellised estimator (SPEC §7): for h > 1 it averages the density of y_{T+h} given each forecast draw's parameters **and its simulated path up to T+h−1**. When the outcome lies far in a predictive's tail (unemployment in 2008–10), one of the forecast draws carries almost all the weight, the estimate is biased low by up to tens of nats, and its delta-method MCSE saturates near 1 however far off it is. In the review of the quick run, Constant's UR h = 8 score at 2008Q2 was −93.6 with 1000 draws, −46 to −63 with 8000 and −36 with 64,000, against −19.1 for the exact path-integrated mixture; SV(rw)'s was −23.9 against about −12.7. The column **RB degen.** counts origins whose joint `lpd_mcse` is ≥ 0.9 for the arm or the baseline: cells with a non-zero count have RB means that are not reliable (they are kept, never dropped, because dropping exactly the tail origins biases the comparison). The criteria are evaluated as specified, on the primary estimator; the review found criterion 4's verdict unchanged when the degenerate origins are replaced by converged values (UR h = 8 about +0.23 against +0.157 at h = 1), but the magnitudes of h ≥ 4 cells with degenerate origins are inflated. A path-integrated RB estimator for h > 1 is a library change (SPEC §7) outside WP23.

RB and Gaussian-approx ΔLPS differ in sign (N08 §C6 expects the rankings to agree) in: full SV(rw) GDP h4, full SV(rw) UR h4, full SV(rw) UR h8, full SV UR h4, full SV UR h8, full SV TBILL h4, full SV(rw,t10) GDP h4, full SV(rw,t10) UR h2, full SV(rw,t10) UR h4, full SV(rw,t10) UR h8, full AR-SV(0.9) UR h4, full AR-SV(0.9) UR h8, full AR-SV(0.9) TBILL h4, pre-crisis SV(rw) GDP h4, pre-crisis SV(rw) GDP h8, pre-crisis SV(rw) UR h4, pre-crisis SV(rw) UR h8, pre-crisis SV GDP h4, pre-crisis SV UR h4, pre-crisis SV UR h8, pre-crisis SV TBILL h4, pre-crisis SV(rw,t10) GDP h4, pre-crisis SV(rw,t10) UR h4, pre-crisis SV(rw,t10) UR h8, pre-crisis AR-SV(0.9) GDP h4, pre-crisis AR-SV(0.9) UR h4, pre-crisis AR-SV(0.9) UR h8, pre-crisis AR-SV(0.9) TBILL h4.

## Full sample: each SV arm vs `Constant`

### SV(rw) (converged at 68% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.126 | +0.107 | 0 | 0.003 | 0.897 | 0.909 | +0.193 | 0.864 | 0.896 |
| GDP | 2 | +0.124 | +0.054 | **1** | 0.002 | 0.893 | 0.899 | +0.133 | 0.856 | 0.878 |
| GDP | 4 | +0.073 | -0.063 | **7** | 0.141 | 0.928 | 0.945 | +0.090 | 0.890 | 0.932 |
| GDP | 8 | +0.111 | +0.010 | **12** | 0.006 | 0.938 | 0.977 | +0.060 | 0.927 | 1.007 |
| INFL | 1 | +0.231 | +0.213 | 0 | 0.000 | 0.844 | 0.898 | +0.075 | 0.954 | 0.966 |
| INFL | 2 | +0.267 | +0.265 | **1** | 0.000 | 0.808 | 0.850 | +0.083 | 0.944 | 0.949 |
| INFL | 4 | +0.249 | +0.250 | **7** | 0.000 | 0.769 | 0.791 | +0.155 | 0.895 | 0.915 |
| INFL | 8 | +0.282 | +0.319 | **12** | 0.000 | 0.724 | 0.728 | +0.138 | 0.887 | 0.883 |
| UR | 1 | +0.159 | +0.125 | 0 | 0.008 | 0.936 | 0.970 | +0.162 | 0.922 | 0.967 |
| UR | 2 | +0.266 | +0.043 | **1** | 0.090 | 0.925 | 0.948 | +0.098 | 0.897 | 0.930 |
| UR | 4 | +0.969 | -0.256 | **7** | 0.153 | 0.929 | 0.937 | -0.182 | 0.906 | 0.914 |
| UR | 8 | +1.954 | -0.499 | **12** | 0.178 | 0.978 | 0.963 | -0.707 | 1.001 | 0.971 |
| TBILL | 1 | +0.378 | +0.322 | 0 | 0.000 | 0.799 | 0.937 | +0.284 | 0.822 | 0.933 |
| TBILL | 2 | +0.153 | +0.094 | **1** | 0.073 | 0.899 | 0.950 | +0.043 | 0.915 | 0.944 |
| TBILL | 4 | -0.081 | -0.143 | **7** | 0.713 | 0.988 | 0.971 | -0.304 | 1.010 | 0.974 |
| TBILL | 8 | -0.207 | -0.234 | **12** | 0.864 | 1.022 | 0.988 | -0.403 | 1.059 | 0.990 |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.809 | +0.665 | 0 | 0.000 |
| joint | 2 | +0.858 | +0.452 | **1** | 0.017 |
| joint | 4 | +2.254 | -0.032 | **7** | 0.146 |
| joint | 8 | +5.013 | -0.320 | **12** | 0.162 |

### SV (converged at 37% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.128 | +0.115 | 0 | 0.001 | 0.904 | 0.916 |  | |  |
| GDP | 2 | +0.136 | +0.084 | **1** | 0.002 | 0.901 | 0.910 |  | |  |
| GDP | 4 | +0.095 | +0.002 | **5** | 0.000 | 0.936 | 0.949 |  | |  |
| GDP | 8 | +0.101 | +0.026 | **8** | 0.000 | 0.943 | 0.971 |  | |  |
| INFL | 1 | +0.216 | +0.194 | 0 | 0.000 | 0.854 | 0.901 |  | |  |
| INFL | 2 | +0.247 | +0.234 | **1** | 0.000 | 0.819 | 0.856 |  | |  |
| INFL | 4 | +0.233 | +0.229 | **5** | 0.000 | 0.785 | 0.804 |  | |  |
| INFL | 8 | +0.278 | +0.286 | **8** | 0.000 | 0.741 | 0.753 |  | |  |
| UR | 1 | +0.176 | +0.148 | 0 | 0.002 | 0.951 | 0.998 |  | |  |
| UR | 2 | +0.304 | +0.140 | **1** | 0.059 | 0.941 | 0.973 |  | |  |
| UR | 4 | +1.003 | -0.021 | **5** | 0.142 | 0.943 | 0.956 |  | |  |
| UR | 8 | +2.265 | -0.163 | **8** | 0.167 | 0.994 | 0.978 |  | |  |
| TBILL | 1 | +0.380 | +0.305 | 0 | 0.000 | 0.800 | 0.951 |  | |  |
| TBILL | 2 | +0.188 | +0.151 | **1** | 0.009 | 0.894 | 0.959 |  | |  |
| TBILL | 4 | +0.006 | -0.008 | **5** | 0.475 | 0.976 | 0.979 |  | |  |
| TBILL | 8 | -0.096 | -0.086 | **8** | 0.778 | 1.014 | 0.997 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.822 | +0.678 | 0 | 0.000 |
| joint | 2 | +0.987 | +0.643 | **1** | 0.008 |
| joint | 4 | +2.607 | +0.467 | **5** | 0.118 |
| joint | 8 | +5.146 | +0.234 | **8** | 0.143 |

### SV(rw,t10) (converged at 57% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.138 | +0.104 | 0 | 0.001 | 0.890 | 0.903 |  | |  |
| GDP | 2 | +0.140 | +0.061 | **1** | 0.001 | 0.884 | 0.892 |  | |  |
| GDP | 4 | +0.111 | -0.021 | **6** | 0.010 | 0.912 | 0.932 |  | |  |
| GDP | 8 | +0.131 | +0.037 | **15** | 0.001 | 0.927 | 0.967 |  | |  |
| INFL | 1 | +0.223 | +0.204 | 0 | 0.000 | 0.851 | 0.903 |  | |  |
| INFL | 2 | +0.252 | +0.252 | **1** | 0.000 | 0.818 | 0.858 |  | |  |
| INFL | 4 | +0.237 | +0.243 | **6** | 0.000 | 0.785 | 0.807 |  | |  |
| INFL | 8 | +0.263 | +0.300 | **15** | 0.000 | 0.746 | 0.753 |  | |  |
| UR | 1 | +0.155 | +0.096 | 0 | 0.006 | 0.939 | 0.968 |  | |  |
| UR | 2 | +0.244 | -0.013 | **1** | 0.081 | 0.928 | 0.947 |  | |  |
| UR | 4 | +0.901 | -0.305 | **6** | 0.152 | 0.932 | 0.936 |  | |  |
| UR | 8 | +2.072 | -0.567 | **15** | 0.174 | 0.975 | 0.957 |  | |  |
| TBILL | 1 | +0.371 | +0.305 | 0 | 0.000 | 0.802 | 0.940 |  | |  |
| TBILL | 2 | +0.150 | +0.092 | **1** | 0.071 | 0.902 | 0.953 |  | |  |
| TBILL | 4 | -0.075 | -0.126 | **6** | 0.702 | 0.989 | 0.974 |  | |  |
| TBILL | 8 | -0.202 | -0.235 | **15** | 0.858 | 1.021 | 0.987 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.811 | +0.625 | 0 | 0.000 |
| joint | 2 | +0.866 | +0.384 | **1** | 0.014 |
| joint | 4 | +2.023 | -0.029 | **6** | 0.141 |
| joint | 8 | +4.636 | -0.378 | **15** | 0.158 |

### AR-SV(0.9) (converged at 62% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.130 | +0.118 | 0 | 0.001 | 0.903 | 0.916 |  | |  |
| GDP | 2 | +0.137 | +0.091 | **1** | 0.002 | 0.900 | 0.909 |  | |  |
| GDP | 4 | +0.097 | +0.009 | **5** | 0.000 | 0.936 | 0.950 |  | |  |
| GDP | 8 | +0.103 | +0.028 | **7** | 0.000 | 0.944 | 0.972 |  | |  |
| INFL | 1 | +0.216 | +0.198 | 0 | 0.000 | 0.853 | 0.900 |  | |  |
| INFL | 2 | +0.249 | +0.235 | **1** | 0.000 | 0.819 | 0.856 |  | |  |
| INFL | 4 | +0.234 | +0.227 | **5** | 0.000 | 0.786 | 0.805 |  | |  |
| INFL | 8 | +0.277 | +0.283 | **7** | 0.000 | 0.741 | 0.754 |  | |  |
| UR | 1 | +0.177 | +0.148 | 0 | 0.002 | 0.949 | 0.996 |  | |  |
| UR | 2 | +0.309 | +0.136 | **1** | 0.061 | 0.938 | 0.971 |  | |  |
| UR | 4 | +1.013 | -0.026 | **5** | 0.142 | 0.941 | 0.955 |  | |  |
| UR | 8 | +2.182 | -0.134 | **7** | 0.168 | 0.992 | 0.977 |  | |  |
| TBILL | 1 | +0.381 | +0.312 | 0 | 0.000 | 0.799 | 0.949 |  | |  |
| TBILL | 2 | +0.188 | +0.149 | **1** | 0.010 | 0.895 | 0.959 |  | |  |
| TBILL | 4 | +0.002 | -0.017 | **5** | 0.490 | 0.976 | 0.979 |  | |  |
| TBILL | 8 | -0.101 | -0.085 | **7** | 0.783 | 1.014 | 0.997 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.825 | +0.695 | 0 | 0.000 |
| joint | 2 | +0.957 | +0.646 | **1** | 0.006 |
| joint | 4 | +2.508 | +0.466 | **5** | 0.117 |
| joint | 8 | +5.326 | +0.265 | **7** | 0.148 |

### `compare()` verdicts vs `Constant` (full)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | RB degen. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | -: | :- | :- |
| SV(rw) | 1 | inconclusive | better | +85.7 | 1.02 | 0 | 0.000 | True | False |
| SV | 1 | inconclusive | better | +87.1 | 1.00 | 0 | 0.000 | True | False |
| SV(rw,t10) | 1 | inconclusive | better | +86.0 | 1.06 | 0 | 0.000 | True | False |
| AR-SV(0.9) | 1 | inconclusive | better | +87.5 | 1.00 | 0 | 0.000 | True | False |
| SV(rw) | 2 | inconclusive | inconclusive | +91.0 | 2.53 | **1** | 0.017 | False | False |
| SV | 2 | inconclusive | better | +104.6 | 2.26 | **1** | 0.008 | True | False |
| SV(rw,t10) | 2 | inconclusive | inconclusive | +91.8 | 2.67 | **1** | 0.014 | False | False |
| AR-SV(0.9) | 2 | inconclusive | better | +101.4 | 2.17 | **1** | 0.006 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +238.9 | 4.44 | **7** | 0.146 | False | False |
| SV | 4 | inconclusive | better | +276.4 | 3.98 | **5** | 0.118 | True | False |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +214.5 | 4.47 | **6** | 0.141 | False | False |
| AR-SV(0.9) | 4 | inconclusive | inconclusive | +265.9 | 3.90 | **5** | 0.117 | False | False |
| SV(rw) | 8 | inconclusive | inconclusive | +531.4 | 6.24 | **12** | 0.162 | False | False |
| SV | 8 | inconclusive | better | +545.5 | 5.70 | **8** | 0.143 | True | False |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +491.4 | 6.59 | **15** | 0.158 | False | False |
| AR-SV(0.9) | 8 | inconclusive | better | +564.5 | 5.71 | **7** | 0.148 | True | False |

## Pre-crisis: each SV arm vs `Constant`

### SV(rw) (converged at 67% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.115 | +0.124 | 0 | 0.006 | 0.901 | 0.915 | +0.222 | |  |
| GDP | 2 | +0.107 | +0.122 | 0 | 0.011 | 0.905 | 0.919 |  | |  |
| GDP | 4 | +0.073 | -0.076 | **2** | 0.143 | 0.950 | 0.980 |  | |  |
| GDP | 8 | +0.108 | -0.004 | **8** | 0.023 | 0.958 | 1.007 | +0.214 | |  |
| INFL | 1 | +0.259 | +0.240 | 0 | 0.000 | 0.828 | 0.892 | +0.077 | |  |
| INFL | 2 | +0.296 | +0.291 | 0 | 0.000 | 0.789 | 0.837 |  | |  |
| INFL | 4 | +0.272 | +0.269 | **2** | 0.000 | 0.757 | 0.785 |  | |  |
| INFL | 8 | +0.295 | +0.345 | **8** | 0.000 | 0.718 | 0.728 | +0.152 | |  |
| UR | 1 | +0.147 | +0.132 | 0 | 0.002 | 0.906 | 0.950 | +0.187 | |  |
| UR | 2 | +0.116 | +0.055 | 0 | 0.110 | 0.899 | 0.931 |  | |  |
| UR | 4 | +0.018 | -0.137 | **2** | 0.451 | 0.930 | 0.937 |  | |  |
| UR | 8 | +1.032 | -0.654 | **8** | 0.217 | 1.027 | 1.001 | -0.065 | |  |
| TBILL | 1 | +0.425 | +0.378 | 0 | 0.000 | 0.772 | 0.926 | +0.316 | |  |
| TBILL | 2 | +0.176 | +0.114 | 0 | 0.073 | 0.890 | 0.951 |  | |  |
| TBILL | 4 | -0.083 | -0.150 | **2** | 0.689 | 0.988 | 0.979 |  | |  |
| TBILL | 8 | -0.220 | -0.253 | **8** | 0.838 | 1.017 | 0.978 | -0.261 | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.790 | +0.683 | 0 | 0.000 |
| joint | 2 | +0.578 | +0.408 | 0 | 0.002 |
| joint | 4 | +0.061 | -0.157 | **2** | 0.430 |
| joint | 8 | +1.883 | -0.514 | **8** | 0.184 |

### SV (converged at 37% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.102 | +0.104 | 0 | 0.001 | 0.911 | 0.925 |  | |  |
| GDP | 2 | +0.094 | +0.091 | 0 | 0.001 | 0.917 | 0.933 |  | |  |
| GDP | 4 | +0.087 | -0.024 | 0 | 0.002 | 0.962 | 0.985 |  | |  |
| GDP | 8 | +0.097 | +0.008 | **4** | 0.000 | 0.964 | 1.001 |  | |  |
| INFL | 1 | +0.240 | +0.216 | 0 | 0.000 | 0.841 | 0.897 |  | |  |
| INFL | 2 | +0.272 | +0.253 | 0 | 0.000 | 0.804 | 0.845 |  | |  |
| INFL | 4 | +0.252 | +0.242 | 0 | 0.000 | 0.777 | 0.800 |  | |  |
| INFL | 8 | +0.288 | +0.303 | **4** | 0.000 | 0.738 | 0.756 |  | |  |
| UR | 1 | +0.154 | +0.133 | 0 | 0.000 | 0.915 | 0.968 |  | |  |
| UR | 2 | +0.141 | +0.106 | 0 | 0.023 | 0.908 | 0.947 |  | |  |
| UR | 4 | +0.082 | -0.000 | 0 | 0.208 | 0.935 | 0.953 |  | |  |
| UR | 8 | +1.179 | -0.342 | **4** | 0.195 | 1.030 | 1.017 |  | |  |
| TBILL | 1 | +0.417 | +0.348 | 0 | 0.000 | 0.778 | 0.941 |  | |  |
| TBILL | 2 | +0.207 | +0.169 | 0 | 0.012 | 0.890 | 0.963 |  | |  |
| TBILL | 4 | +0.009 | -0.007 | 0 | 0.471 | 0.982 | 0.993 |  | |  |
| TBILL | 8 | -0.098 | -0.094 | **4** | 0.748 | 1.017 | 1.000 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.776 | +0.642 | 0 | 0.000 |
| joint | 2 | +0.643 | +0.519 | 0 | 0.000 |
| joint | 4 | +0.384 | +0.253 | 0 | 0.035 |
| joint | 8 | +2.047 | +0.021 | **4** | 0.133 |

### SV(rw,t10) (converged at 55% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.120 | +0.128 | 0 | 0.003 | 0.893 | 0.908 |  | |  |
| GDP | 2 | +0.115 | +0.128 | 0 | 0.004 | 0.896 | 0.909 |  | |  |
| GDP | 4 | +0.113 | -0.027 | **2** | 0.009 | 0.932 | 0.964 |  | |  |
| GDP | 8 | +0.124 | +0.017 | **11** | 0.006 | 0.947 | 0.996 |  | |  |
| INFL | 1 | +0.253 | +0.231 | 0 | 0.000 | 0.835 | 0.897 |  | |  |
| INFL | 2 | +0.284 | +0.278 | 0 | 0.000 | 0.799 | 0.845 |  | |  |
| INFL | 4 | +0.262 | +0.264 | **2** | 0.000 | 0.773 | 0.802 |  | |  |
| INFL | 8 | +0.272 | +0.324 | **11** | 0.001 | 0.743 | 0.755 |  | |  |
| UR | 1 | +0.147 | +0.127 | 0 | 0.001 | 0.907 | 0.948 |  | |  |
| UR | 2 | +0.119 | +0.064 | 0 | 0.082 | 0.898 | 0.926 |  | |  |
| UR | 4 | +0.040 | -0.098 | **2** | 0.378 | 0.926 | 0.930 |  | |  |
| UR | 8 | +0.983 | -0.635 | **11** | 0.212 | 1.012 | 0.988 |  | |  |
| TBILL | 1 | +0.415 | +0.358 | 0 | 0.000 | 0.775 | 0.927 |  | |  |
| TBILL | 2 | +0.172 | +0.111 | 0 | 0.071 | 0.894 | 0.952 |  | |  |
| TBILL | 4 | -0.078 | -0.136 | **2** | 0.683 | 0.992 | 0.982 |  | |  |
| TBILL | 8 | -0.218 | -0.260 | **11** | 0.836 | 1.021 | 0.982 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.788 | +0.662 | 0 | 0.000 |
| joint | 2 | +0.584 | +0.415 | 0 | 0.001 |
| joint | 4 | +0.102 | -0.053 | **2** | 0.376 |
| joint | 8 | +1.659 | -0.490 | **11** | 0.171 |

### AR-SV(0.9) (converged at 60% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.104 | +0.107 | 0 | 0.001 | 0.911 | 0.925 |  | |  |
| GDP | 2 | +0.095 | +0.096 | 0 | 0.001 | 0.917 | 0.932 |  | |  |
| GDP | 4 | +0.089 | -0.016 | 0 | 0.001 | 0.962 | 0.986 |  | |  |
| GDP | 8 | +0.099 | +0.013 | **3** | 0.000 | 0.965 | 1.001 |  | |  |
| INFL | 1 | +0.241 | +0.220 | 0 | 0.000 | 0.840 | 0.895 |  | |  |
| INFL | 2 | +0.273 | +0.254 | 0 | 0.000 | 0.804 | 0.845 |  | |  |
| INFL | 4 | +0.254 | +0.240 | 0 | 0.000 | 0.776 | 0.800 |  | |  |
| INFL | 8 | +0.286 | +0.300 | **3** | 0.000 | 0.738 | 0.756 |  | |  |
| UR | 1 | +0.156 | +0.137 | 0 | 0.000 | 0.914 | 0.967 |  | |  |
| UR | 2 | +0.143 | +0.102 | 0 | 0.024 | 0.907 | 0.947 |  | |  |
| UR | 4 | +0.081 | -0.011 | 0 | 0.219 | 0.934 | 0.953 |  | |  |
| UR | 8 | +1.061 | -0.319 | **3** | 0.197 | 1.030 | 1.017 |  | |  |
| TBILL | 1 | +0.419 | +0.357 | 0 | 0.000 | 0.777 | 0.938 |  | |  |
| TBILL | 2 | +0.207 | +0.168 | 0 | 0.013 | 0.891 | 0.962 |  | |  |
| TBILL | 4 | +0.005 | -0.016 | 0 | 0.482 | 0.982 | 0.993 |  | |  |
| TBILL | 8 | -0.104 | -0.093 | **3** | 0.754 | 1.018 | 1.000 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.782 | +0.665 | 0 | 0.000 |
| joint | 2 | +0.642 | +0.523 | 0 | 0.000 |
| joint | 4 | +0.370 | +0.240 | 0 | 0.039 |
| joint | 8 | +2.191 | +0.048 | **3** | 0.149 |

### `compare()` verdicts vs `Constant` (pre-crisis)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | RB degen. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | -: | :- | :- |
| SV(rw) | 1 | inconclusive | better | +72.7 | 0.52 | 0 | 0.000 | True | False |
| SV | 1 | inconclusive | better | +71.4 | 0.45 | 0 | 0.000 | True | False |
| SV(rw,t10) | 1 | inconclusive | better | +72.5 | 0.55 | 0 | 0.000 | True | False |
| AR-SV(0.9) | 1 | inconclusive | better | +72.0 | 0.45 | 0 | 0.000 | True | False |
| SV(rw) | 2 | inconclusive | better | +53.2 | 1.72 | 0 | 0.002 | True | False |
| SV | 2 | inconclusive | better | +59.2 | 1.30 | 0 | 0.000 | True | False |
| SV(rw,t10) | 2 | inconclusive | better | +53.7 | 1.80 | 0 | 0.001 | True | False |
| AR-SV(0.9) | 2 | inconclusive | better | +59.1 | 1.25 | 0 | 0.000 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +5.6 | 3.42 | **2** | 0.430 | True | True |
| SV | 4 | inconclusive | inconclusive | +35.3 | 2.77 | 0 | 0.035 | True | True |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +9.4 | 3.47 | **2** | 0.376 | True | True |
| AR-SV(0.9) | 4 | inconclusive | inconclusive | +34.0 | 2.73 | 0 | 0.039 | True | True |
| SV(rw) | 8 | inconclusive | inconclusive | +173.2 | 5.39 | **8** | 0.184 | True | True |
| SV | 8 | inconclusive | inconclusive | +188.4 | 4.73 | **4** | 0.133 | True | True |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +152.6 | 5.74 | **11** | 0.171 | True | True |
| AR-SV(0.9) | 8 | inconclusive | inconclusive | +201.6 | 4.63 | **3** | 0.149 | True | True |

## Variants vs RW and AR SV (criterion 5, full sample)

| variant | reference | h | joint ΔLPD | DM p (variant better) |
| :- | :- | -: | -: | -: |
| SV(rw,t10) | SV(rw) | 1 | +0.002 | 0.446 |
| SV(rw,t10) | SV(rw) | 2 | +0.008 | 0.406 |
| SV(rw,t10) | SV(rw) | 4 | -0.231 | 0.810 |
| SV(rw,t10) | SV(rw) | 8 | -0.377 | 0.790 |
| SV(rw,t10) | SV | 1 | -0.011 | 0.625 |
| SV(rw,t10) | SV | 2 | -0.120 | 0.977 |
| SV(rw,t10) | SV | 4 | -0.584 | 0.953 |
| SV(rw,t10) | SV | 8 | -0.510 | 0.986 |
| AR-SV(0.9) | SV(rw) | 1 | +0.017 | 0.319 |
| AR-SV(0.9) | SV(rw) | 2 | +0.099 | 0.095 |
| AR-SV(0.9) | SV(rw) | 4 | +0.254 | 0.009 |
| AR-SV(0.9) | SV(rw) | 8 | +0.313 | 0.028 |
| AR-SV(0.9) | SV | 1 | +0.004 | 0.281 |
| AR-SV(0.9) | SV | 2 | -0.030 | 0.845 |
| AR-SV(0.9) | SV | 4 | -0.099 | 0.876 |
| AR-SV(0.9) | SV | 8 | +0.180 | 0.261 |

## Structural-form pair (robustness, full sample)

| variable | h=1 ΔLPS | h=2 ΔLPS | h=8 ΔLPS |
| :- | -: | -: | -: |
| GDP | +0.103 | +0.107 | +0.109 |
| INFL | +0.232 | +0.260 | +0.282 |
| UR | +0.128 | +0.197 | +1.754 |
| TBILL | +0.315 | +0.060 | -0.239 |
| joint | +0.751 | +0.589 | +2.913 |

## All 24 orderings (criterion 6)

Origins in this stage: 106 (1985Q1–2011Q2). Each ordering's SV(rw) and Constant backtests are compared with `compare()`; the Constant arm is order-invariant in theory (Chan 2022), so its spread is Monte Carlo noise.

| # | ordering | verdict h=1 (gate on) | gate off | joint ΔLPD h=1 | DM p joint | DM p GDP | DM p TBILL | SV converged | Const converged |
| -: | :- | :- | :- | -: | -: | -: | -: | -: | -: |
| 0 | GDP > INFL > UR > TBILL | inconclusive | better | +0.838 | 0.000 | 0.002 | 0.000 | 65% | 100% |
| 1 | GDP > INFL > TBILL > UR | inconclusive | better | +0.858 | 0.000 | 0.002 | 0.000 | 67% | 100% |
| 2 | GDP > UR > INFL > TBILL | inconclusive | better | +0.807 | 0.000 | 0.002 | 0.000 | 69% | 100% |
| 3 | GDP > UR > TBILL > INFL | inconclusive | better | +0.819 | 0.000 | 0.002 | 0.000 | 75% | 100% |
| 4 | GDP > TBILL > INFL > UR | inconclusive | better | +0.863 | 0.000 | 0.002 | 0.000 | 79% | 100% |
| 5 | GDP > TBILL > UR > INFL | inconclusive | better | +0.861 | 0.000 | 0.002 | 0.000 | 72% | 100% |
| 6 | INFL > GDP > UR > TBILL | inconclusive | better | +0.824 | 0.000 | 0.002 | 0.000 | 68% | 100% |
| 7 | INFL > GDP > TBILL > UR | inconclusive | better | +0.873 | 0.000 | 0.002 | 0.000 | 75% | 100% |
| 8 | INFL > UR > GDP > TBILL | inconclusive | better | +0.813 | 0.000 | 0.006 | 0.000 | 59% | 100% |
| 9 | INFL > UR > TBILL > GDP | inconclusive | better | +0.817 | 0.000 | 0.005 | 0.000 | 57% | 100% |
| 10 | INFL > TBILL > GDP > UR | inconclusive | better | +0.887 | 0.000 | 0.000 | 0.000 | 74% | 100% |
| 11 | INFL > TBILL > UR > GDP | inconclusive | better | +0.868 | 0.000 | 0.001 | 0.000 | 63% | 100% |
| 12 | UR > GDP > INFL > TBILL | inconclusive | better | +0.810 | 0.000 | 0.008 | 0.000 | 62% | 100% |
| 13 | UR > GDP > TBILL > INFL | inconclusive | better | +0.807 | 0.000 | 0.008 | 0.000 | 61% | 100% |
| 14 | UR > INFL > GDP > TBILL | inconclusive | better | +0.810 | 0.000 | 0.006 | 0.000 | 60% | 100% |
| 15 | UR > INFL > TBILL > GDP | inconclusive | better | +0.832 | 0.000 | 0.005 | 0.000 | 60% | 100% |
| 16 | UR > TBILL > GDP > INFL | inconclusive | better | +0.808 | 0.000 | 0.008 | 0.000 | 60% | 100% |
| 17 | UR > TBILL > INFL > GDP | inconclusive | better | +0.813 | 0.000 | 0.005 | 0.000 | 61% | 100% |
| 18 | TBILL > GDP > INFL > UR | inconclusive | better | +0.869 | 0.000 | 0.000 | 0.000 | 82% | 100% |
| 19 | TBILL > GDP > UR > INFL | inconclusive | better | +0.865 | 0.000 | 0.000 | 0.000 | 71% | 100% |
| 20 | TBILL > INFL > GDP > UR | inconclusive | better | +0.880 | 0.000 | 0.000 | 0.000 | 73% | 100% |
| 21 | TBILL > INFL > UR > GDP | inconclusive | better | +0.876 | 0.000 | 0.001 | 0.000 | 72% | 100% |
| 22 | TBILL > UR > GDP > INFL | inconclusive | better | +0.883 | 0.000 | 0.001 | 0.000 | 75% | 100% |
| 23 | TBILL > UR > INFL > GDP | inconclusive | better | +0.855 | 0.000 | 0.001 | 0.000 | 64% | 100% |

`ordering_sensitivity` for `Constant`: material = True; joint LPD range over orderings (nats per origin): h=1: 0.027, h=2: 0.189, h=4: 0.768, h=8: 2.280.

`ordering_sensitivity` for `SV(rw)`: material = True; joint LPD range over orderings (nats per origin): h=1: 0.083, h=2: 0.281, h=4: 1.414, h=8: 5.173.

## Deviations and caveats

* Not a deviation from CR, but from the library default: every own-lag prior mean is 0, as in CR (WP App. 6.1), including the UR and TBILL levels; the library warns that this contradicts their level transformation codes (it would centre them on 1).
* Pseudo real time on the final FRED-QD 2026-09 vintage instead of CR's RTDSM real-time vintages and second-release actuals (D4; §C2). Score levels differ; rankings are expected to be robust (§C6).
* Estimation starts in 1965Q1, not 1955Q1 (FRED-QD starts in 1959; §C2).
* Inflation uses the chain-type GDPCTPI (FRED-QD has no GDPDEF; §C6).
* Baseline: Constant('iw_equivalent') (Chan 2022) stands in for CR's normal-diffuse VAR (§C3); the SV arms use the library's default SV priors except the two CR variants (no prior was tuned). CR's RW-SV σ² prior IG(1/2, 0.0175) and its 36-observation training prior for h₀ are not used (the latter would violate R9, §C6).
* AR-SV variant: CR's a₁ ~ N(0.9, 0.05²) is mapped to (φ+1)/2 ~ Beta(71.25, 3.75) (same mean and sd on φ); the SV level μ keeps the library's in-window data-centred prior instead of CR's a₀ ~ N(0, 0.4²).
* LPD: the primary score is the Rao–Blackwellised predictive density; CR's Gaussian approximation is reported alongside (§C4, §C6).
* Significance: one-sided DM/HLN p-values are descriptive under an expanding window; compare() therefore decides 'better' through the Model Confidence Set (SPEC §9.5).
* compare() turns a verdict 'inconclusive' when fewer than 95% of origins converged (R-hat/ESS targets of DESIGN §8.2); verdicts are reported with that gate on (as returned) and off.

## Warnings raised during the run (aggregated)

Messages that differ only in numbers or variable lists are grouped; one example is shown.

* `Constant`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw)`, 15 origin-messages (15 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0261, 0.0459]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw)`, 34 origin-messages (34 variants), e.g.: MCMC not converged: 1 of 13 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[UR] R-hat 1.015, ESS 618); increase draws, burn or thin
* `SV`, 67 origin-messages (67 variants), e.g.: MCMC not converged: 1 of 21 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[TBILL] R-hat 1.013, ESS 550); increase draws, burn or thin
* `SV`, 13 origin-messages (13 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0137, 0.0307]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw,t10)`, 30 origin-messages (28 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL', 'UR'] (fractions [0.0454, 0.0676, 0.0159]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: Student-t ν has posterior mass ≥ 0.5 at the grid bound 10 for ['GDP', 'INFL', 'UR', 'TBILL'] (mass [1.0, 1.0, 1.0, 1.0]): near-Gaussian tails, SV may suffice (report only; mass at ν = 10 is normal, N03 §6)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw,t10)`, 46 origin-messages (46 variants), e.g.: MCMC not converged: 1 of 13 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[UR] R-hat 1.020, ESS 537); increase draws, burn or thin
* `AR-SV(0.9)`, 40 origin-messages (40 variants), e.g.: MCMC not converged: 1 of 21 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[TBILL] R-hat 1.012, ESS 743); increase draws, burn or thin
* `AR-SV(0.9)`, 12 origin-messages (12 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0155, 0.0324]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
