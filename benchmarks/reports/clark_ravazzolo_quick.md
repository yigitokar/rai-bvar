# Clark–Ravazzolo (2015) replication on FRED-QD (WP23) — quick, NOT the acceptance run

> **Config `quick`: this is not the acceptance run.** The acceptance run is `--config full` (4 chains × 5000 draws, all 24 orderings); see the module docstring of `benchmarks/clark_ravazzolo.py` for the command. This report has no 24-ordering stage, so criterion 6 is not evaluated. Its main stage was computed on a shared 4-core x86-64 VM (load average about 10) in 1.08 h; this file was assembled from that run's cache. Its 24-ordering stage was started and then stopped on request, to free the shared machine for other work; the acceptance run (config full, all 24 orderings) is to be run elsewhere. 

Generated 2026-10-23 01:25 UTC by `benchmarks/clark_ravazzolo.py` (config `quick`, rai-bvar 0.0.1, jax 0.12.0.1, Linux x86_64, python 3.13.16). Run time of this invocation 0.00 h (cached origins are read, not refitted).

## Acceptance criteria (N08 §C5)

| # | Met | Operational rule | Result |
| :- | :- | :- | :- |
| 1 | **met** | ΔLPS(SV(rw) − Constant) > 0 at h = 1, 2 for all four variables in the full and pre-crisis samples, and the joint ΔLPD at h = 1 > 0 with one-sided DM/HLN p < 0.10 in both samples | 16 of 16 cells positive; joint h=1: full +0.811 nats/quarter (p=0.000), pre +0.779 nats/quarter (p=0.000) |
| 2 | **met** | CRPS ratio SV(rw)/Constant < 1 at h = 1, 2 for all four variables (full sample) | 8 of 8 cells < 1; pre-crisis: all 8 < 1 |
| 3 | **met** | RMSE ratio SV(rw)/Constant ≤ 1.02 ('about 1') at h = 1, 2 for all four variables (full sample); CR's gains for GDP and the bill are 5–10% | GDP/h1 0.907, GDP/h2 0.902, INFL/h1 0.897, INFL/h2 0.847, UR/h1 0.970, UR/h2 0.946, TBILL/h1 0.932, TBILL/h2 0.945 |
| 4 | **not met** | full sample: ΔLPS(h=8) < ΔLPS(h=1) for UR and TBILL; pre-crisis: ΔLPS(h=8) > 0 for GDP and INFL | full UR: h1 +0.157 → h8 +1.922; full TBILL: h1 +0.375 → h8 -0.207; pre-crisis GDP h8 +0.108; pre-crisis INFL h8 +0.299. Secondary (not part of the rule): full UR h8: median +0.118 (largest origins: 2008Q2 +69.7, 2007Q3 +51.5, 2008Q1 +45.4), Gaussian-approx h1 +0.121 → h8 -0.557; full TBILL h8: median -0.032 (largest origins: 1999Q4 -2.8, 2000Q4 -2.6, 2000Q2 -2.5), Gaussian-approx h1 +0.307 → h8 -0.224; pre-crisis GDP h8 Gaussian-approx +0.002; pre-crisis INFL h8 Gaussian-approx +0.354 |
| 5 | **met** | neither SV(rw,t10) nor AR-SV(0.9) is better than both SV(rw) and SV (one-sided DM/HLN p < 0.10, positive ΔLPD) at ≥ 3 of 4 joint horizons or ≥ 8 of 16 marginal cells (full sample) | SV(rw,t10): significantly better than both SV(rw) and SV at 0/4 joint horizons and 0/16 marginal cells; AR-SV(0.9): significantly better than both SV(rw) and SV at 1/4 joint horizons and 0/16 marginal cells |
| 6 | not run | 24 orderings | ordering stage skipped |

## Design

* Data: FRED-QD 2026-09 (SHA-256 verified), GDP = 100·Δlog GDPC1, INFL = 100·Δlog GDPCTPI, UR = UNRATE, TBILL = TB3MS (levels); non-annualised quarterly rates; estimation sample starts 1965Q1; pseudo real time on the final vintage (D4).
* Asserted (§C6): FRED-QD UNRATE/TB3MS equal within-quarter means of FRED-MD monthly values of the same vintage — UNRATE: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2); TB3MS: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2).
* Asserted (R9, §C6): at origins 1985Q1, 2007Q4, 2011Q2 the resolved priors, hyperparameters and padded data are bitwise identical when the vintage is cut at the origin (no look-ahead; CR's last-vintage SV training sample is not replicated).
* Origins: 1985Q1–2011Q2 (106, full) and 1985Q1–2007Q4 (92, pre-crisis subset); expanding window; h = 1, 2, 4, 8.
* MCMC: 4 chains × 2000 kept draws, burn 1000, thin 1; 1000 forecast draws per origin; backtest seed 0; per-origin fit seeds (DESIGN §11.6); cold starts.
* Mean prior (all models): `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, alpha=2, cross=1, mean=0)`, p = 4. Volatility arms: `Constant` = `Constant(iw_equivalent)`, `SV(rw)` = `SV(rw)`, `SV` = `SV`, `SV(rw,t10)` = `SV(rw,t, nu_prior=7eca60, t_scale=unit)`, `AR-SV(0.9)` = `SV(phi_prior=7ea4a9)`.
* LPD is the library's Rao–Blackwellised predictive density (primary); `ΔLPS gauss` is CR's Gaussian approximation (Adolfson et al. 2007) from the same draws.

## Full sample: each SV arm vs `Constant`

### SV(rw) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.128 | +0.101 | 0.002 | 0.897 | 0.907 | +0.193 | 0.864 | 0.896 |
| GDP | 2 | +0.130 | +0.060 | 0.002 | 0.895 | 0.902 | +0.133 | 0.856 | 0.878 |
| GDP | 4 | +0.085 | -0.073 | 0.073 | 0.929 | 0.945 | +0.090 | 0.890 | 0.932 |
| GDP | 8 | +0.111 | +0.014 | 0.007 | 0.941 | 0.978 | +0.060 | 0.927 | 1.007 |
| INFL | 1 | +0.232 | +0.205 | 0.000 | 0.844 | 0.897 | +0.075 | 0.954 | 0.966 |
| INFL | 2 | +0.268 | +0.272 | 0.000 | 0.805 | 0.847 | +0.083 | 0.944 | 0.949 |
| INFL | 4 | +0.257 | +0.258 | 0.000 | 0.765 | 0.789 | +0.155 | 0.895 | 0.915 |
| INFL | 8 | +0.285 | +0.326 | 0.000 | 0.722 | 0.725 | +0.138 | 0.887 | 0.883 |
| UR | 1 | +0.157 | +0.121 | 0.009 | 0.937 | 0.970 | +0.162 | 0.922 | 0.967 |
| UR | 2 | +0.152 | +0.046 | 0.091 | 0.922 | 0.946 | +0.098 | 0.897 | 0.930 |
| UR | 4 | +0.807 | -0.236 | 0.156 | 0.927 | 0.937 | -0.182 | 0.906 | 0.914 |
| UR | 8 | +1.922 | -0.557 | 0.180 | 0.983 | 0.965 | -0.707 | 1.001 | 0.971 |
| TBILL | 1 | +0.375 | +0.307 | 0.000 | 0.796 | 0.932 | +0.284 | 0.822 | 0.933 |
| TBILL | 2 | +0.147 | +0.089 | 0.090 | 0.895 | 0.945 | +0.043 | 0.915 | 0.944 |
| TBILL | 4 | -0.117 | -0.138 | 0.749 | 0.982 | 0.967 | -0.304 | 1.010 | 0.974 |
| TBILL | 8 | -0.207 | -0.224 | 0.862 | 1.021 | 0.989 | -0.403 | 1.059 | 0.990 |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.811 | +0.606 | 0.000 |
| joint | 2 | +0.880 | +0.404 | 0.021 |
| joint | 4 | +1.903 | +0.005 | 0.148 |
| joint | 8 | +4.803 | -0.390 | 0.164 |

### SV (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.129 | +0.111 | 0.001 | 0.904 | 0.915 |  | |  |
| GDP | 2 | +0.137 | +0.094 | 0.002 | 0.903 | 0.912 |  | |  |
| GDP | 4 | +0.094 | -0.000 | 0.000 | 0.936 | 0.950 |  | |  |
| GDP | 8 | +0.099 | +0.023 | 0.000 | 0.946 | 0.970 |  | |  |
| INFL | 1 | +0.217 | +0.188 | 0.000 | 0.853 | 0.900 |  | |  |
| INFL | 2 | +0.250 | +0.242 | 0.000 | 0.815 | 0.853 |  | |  |
| INFL | 4 | +0.243 | +0.234 | 0.000 | 0.781 | 0.802 |  | |  |
| INFL | 8 | +0.277 | +0.293 | 0.000 | 0.737 | 0.750 |  | |  |
| UR | 1 | +0.173 | +0.148 | 0.002 | 0.952 | 0.999 |  | |  |
| UR | 2 | +0.221 | +0.138 | 0.035 | 0.938 | 0.971 |  | |  |
| UR | 4 | +0.876 | +0.024 | 0.142 | 0.942 | 0.956 |  | |  |
| UR | 8 | +2.037 | -0.185 | 0.168 | 0.998 | 0.980 |  | |  |
| TBILL | 1 | +0.378 | +0.303 | 0.000 | 0.799 | 0.944 |  | |  |
| TBILL | 2 | +0.182 | +0.145 | 0.013 | 0.892 | 0.955 |  | |  |
| TBILL | 4 | -0.019 | -0.015 | 0.567 | 0.973 | 0.976 |  | |  |
| TBILL | 8 | -0.099 | -0.077 | 0.780 | 1.012 | 0.998 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.829 | +0.652 | 0.000 |
| joint | 2 | +1.032 | +0.618 | 0.014 |
| joint | 4 | +2.151 | +0.514 | 0.116 |
| joint | 8 | +5.471 | +0.223 | 0.150 |

### SV(rw,t10) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.137 | +0.103 | 0.001 | 0.891 | 0.903 |  | |  |
| GDP | 2 | +0.141 | +0.073 | 0.001 | 0.887 | 0.896 |  | |  |
| GDP | 4 | +0.115 | -0.035 | 0.004 | 0.916 | 0.934 |  | |  |
| GDP | 8 | +0.131 | +0.038 | 0.001 | 0.928 | 0.967 |  | |  |
| INFL | 1 | +0.221 | +0.195 | 0.000 | 0.851 | 0.901 |  | |  |
| INFL | 2 | +0.253 | +0.257 | 0.000 | 0.815 | 0.856 |  | |  |
| INFL | 4 | +0.241 | +0.252 | 0.000 | 0.780 | 0.805 |  | |  |
| INFL | 8 | +0.264 | +0.306 | 0.000 | 0.742 | 0.749 |  | |  |
| UR | 1 | +0.153 | +0.101 | 0.006 | 0.940 | 0.970 |  | |  |
| UR | 2 | +0.161 | +0.003 | 0.069 | 0.927 | 0.947 |  | |  |
| UR | 4 | +0.700 | -0.280 | 0.155 | 0.932 | 0.936 |  | |  |
| UR | 8 | +1.737 | -0.616 | 0.179 | 0.979 | 0.959 |  | |  |
| TBILL | 1 | +0.370 | +0.296 | 0.000 | 0.799 | 0.934 |  | |  |
| TBILL | 2 | +0.153 | +0.095 | 0.066 | 0.897 | 0.945 |  | |  |
| TBILL | 4 | -0.087 | -0.125 | 0.714 | 0.982 | 0.968 |  | |  |
| TBILL | 8 | -0.216 | -0.221 | 0.863 | 1.018 | 0.988 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.816 | +0.592 | 0.000 |
| joint | 2 | +0.838 | +0.377 | 0.014 |
| joint | 4 | +1.779 | -0.002 | 0.138 |
| joint | 8 | +4.308 | -0.421 | 0.169 |

### AR-SV(0.9) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.132 | +0.114 | 0.001 | 0.904 | 0.915 |  | |  |
| GDP | 2 | +0.139 | +0.097 | 0.002 | 0.902 | 0.912 |  | |  |
| GDP | 4 | +0.096 | +0.007 | 0.000 | 0.936 | 0.951 |  | |  |
| GDP | 8 | +0.100 | +0.027 | 0.000 | 0.946 | 0.971 |  | |  |
| INFL | 1 | +0.218 | +0.193 | 0.000 | 0.853 | 0.899 |  | |  |
| INFL | 2 | +0.250 | +0.238 | 0.000 | 0.816 | 0.853 |  | |  |
| INFL | 4 | +0.242 | +0.237 | 0.000 | 0.782 | 0.802 |  | |  |
| INFL | 8 | +0.277 | +0.289 | 0.000 | 0.737 | 0.750 |  | |  |
| UR | 1 | +0.175 | +0.149 | 0.003 | 0.951 | 0.997 |  | |  |
| UR | 2 | +0.222 | +0.139 | 0.036 | 0.936 | 0.969 |  | |  |
| UR | 4 | +0.893 | +0.008 | 0.142 | 0.940 | 0.954 |  | |  |
| UR | 8 | +1.937 | -0.177 | 0.169 | 0.997 | 0.979 |  | |  |
| TBILL | 1 | +0.379 | +0.305 | 0.000 | 0.798 | 0.944 |  | |  |
| TBILL | 2 | +0.182 | +0.148 | 0.014 | 0.892 | 0.955 |  | |  |
| TBILL | 4 | -0.016 | -0.022 | 0.557 | 0.974 | 0.976 |  | |  |
| TBILL | 8 | -0.097 | -0.072 | 0.778 | 1.010 | 0.996 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.838 | +0.663 | 0.000 |
| joint | 2 | +1.036 | +0.628 | 0.013 |
| joint | 4 | +2.210 | +0.516 | 0.114 |
| joint | 8 | +5.514 | +0.218 | 0.149 |

### `compare()` verdicts vs `Constant` (full)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | :- | :- |
| SV(rw) | 1 | inconclusive | better | +85.9 | 0.70 | 0.000 | True | False |
| SV | 1 | inconclusive | better | +87.9 | 0.70 | 0.000 | True | False |
| SV(rw,t10) | 1 | inconclusive | better | +86.5 | 0.83 | 0.000 | True | False |
| AR-SV(0.9) | 1 | inconclusive | better | +88.8 | 0.67 | 0.000 | True | False |
| SV(rw) | 2 | inconclusive | inconclusive | +93.3 | 2.54 | 0.021 | False | False |
| SV | 2 | inconclusive | better | +109.4 | 2.45 | 0.014 | True | False |
| SV(rw,t10) | 2 | inconclusive | inconclusive | +88.8 | 2.57 | 0.014 | False | False |
| AR-SV(0.9) | 2 | inconclusive | better | +109.8 | 2.45 | 0.013 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +201.7 | 4.47 | 0.148 | False | False |
| SV | 4 | inconclusive | inconclusive | +228.0 | 3.87 | 0.116 | False | False |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +188.6 | 4.46 | 0.138 | False | False |
| AR-SV(0.9) | 4 | inconclusive | better | +234.2 | 3.87 | 0.114 | True | False |
| SV(rw) | 8 | inconclusive | inconclusive | +509.2 | 6.40 | 0.164 | True | True |
| SV | 8 | inconclusive | inconclusive | +579.9 | 5.91 | 0.150 | True | True |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +456.6 | 6.42 | 0.169 | True | True |
| AR-SV(0.9) | 8 | inconclusive | inconclusive | +584.5 | 5.90 | 0.149 | True | True |

## Pre-crisis: each SV arm vs `Constant`

### SV(rw) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.116 | +0.118 | 0.005 | 0.900 | 0.915 | +0.222 | |  |
| GDP | 2 | +0.108 | +0.114 | 0.010 | 0.908 | 0.923 |  | |  |
| GDP | 4 | +0.088 | -0.084 | 0.069 | 0.949 | 0.979 |  | |  |
| GDP | 8 | +0.108 | +0.002 | 0.023 | 0.961 | 1.011 | +0.214 | |  |
| INFL | 1 | +0.259 | +0.227 | 0.000 | 0.829 | 0.892 | +0.077 | |  |
| INFL | 2 | +0.296 | +0.299 | 0.000 | 0.787 | 0.832 |  | |  |
| INFL | 4 | +0.279 | +0.276 | 0.000 | 0.753 | 0.782 |  | |  |
| INFL | 8 | +0.299 | +0.354 | 0.000 | 0.715 | 0.725 | +0.152 | |  |
| UR | 1 | +0.144 | +0.124 | 0.003 | 0.906 | 0.953 | +0.187 | |  |
| UR | 2 | +0.117 | +0.056 | 0.112 | 0.895 | 0.930 |  | |  |
| UR | 4 | +0.021 | -0.132 | 0.446 | 0.926 | 0.936 |  | |  |
| UR | 8 | +0.620 | -0.718 | 0.245 | 1.031 | 1.003 | -0.065 | |  |
| TBILL | 1 | +0.420 | +0.357 | 0.000 | 0.770 | 0.919 | +0.316 | |  |
| TBILL | 2 | +0.169 | +0.108 | 0.090 | 0.885 | 0.944 |  | |  |
| TBILL | 4 | -0.127 | -0.144 | 0.734 | 0.983 | 0.974 |  | |  |
| TBILL | 8 | -0.216 | -0.241 | 0.834 | 1.014 | 0.977 | -0.261 | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.779 | +0.624 | 0.000 |
| joint | 2 | +0.567 | +0.370 | 0.003 |
| joint | 4 | +0.057 | -0.151 | 0.441 |
| joint | 8 | +2.034 | -0.581 | 0.188 |

### SV (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.102 | +0.098 | 0.001 | 0.912 | 0.926 |  | |  |
| GDP | 2 | +0.094 | +0.090 | 0.001 | 0.921 | 0.936 |  | |  |
| GDP | 4 | +0.089 | -0.022 | 0.001 | 0.960 | 0.985 |  | |  |
| GDP | 8 | +0.095 | +0.006 | 0.000 | 0.967 | 1.001 |  | |  |
| INFL | 1 | +0.241 | +0.207 | 0.000 | 0.841 | 0.896 |  | |  |
| INFL | 2 | +0.273 | +0.262 | 0.000 | 0.800 | 0.841 |  | |  |
| INFL | 4 | +0.261 | +0.247 | 0.000 | 0.773 | 0.798 |  | |  |
| INFL | 8 | +0.288 | +0.312 | 0.000 | 0.734 | 0.753 |  | |  |
| UR | 1 | +0.150 | +0.131 | 0.000 | 0.917 | 0.972 |  | |  |
| UR | 2 | +0.145 | +0.103 | 0.018 | 0.906 | 0.948 |  | |  |
| UR | 4 | +0.088 | +0.006 | 0.206 | 0.934 | 0.955 |  | |  |
| UR | 8 | +0.790 | -0.362 | 0.204 | 1.036 | 1.019 |  | |  |
| TBILL | 1 | +0.413 | +0.342 | 0.000 | 0.778 | 0.933 |  | |  |
| TBILL | 2 | +0.201 | +0.163 | 0.016 | 0.887 | 0.957 |  | |  |
| TBILL | 4 | -0.021 | -0.014 | 0.565 | 0.979 | 0.989 |  | |  |
| TBILL | 8 | -0.101 | -0.084 | 0.751 | 1.014 | 1.000 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.766 | +0.618 | 0.000 |
| joint | 2 | +0.636 | +0.500 | 0.000 |
| joint | 4 | +0.379 | +0.235 | 0.053 |
| joint | 8 | +2.678 | +0.012 | 0.156 |

### SV(rw,t10) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.117 | +0.122 | 0.003 | 0.894 | 0.909 |  | |  |
| GDP | 2 | +0.113 | +0.120 | 0.005 | 0.900 | 0.914 |  | |  |
| GDP | 4 | +0.119 | -0.044 | 0.004 | 0.935 | 0.967 |  | |  |
| GDP | 8 | +0.125 | +0.020 | 0.006 | 0.950 | 0.998 |  | |  |
| INFL | 1 | +0.250 | +0.219 | 0.000 | 0.836 | 0.897 |  | |  |
| INFL | 2 | +0.283 | +0.283 | 0.000 | 0.796 | 0.843 |  | |  |
| INFL | 4 | +0.266 | +0.270 | 0.000 | 0.768 | 0.799 |  | |  |
| INFL | 8 | +0.276 | +0.333 | 0.001 | 0.738 | 0.751 |  | |  |
| UR | 1 | +0.146 | +0.125 | 0.002 | 0.907 | 0.951 |  | |  |
| UR | 2 | +0.124 | +0.066 | 0.074 | 0.894 | 0.925 |  | |  |
| UR | 4 | -0.000 | -0.109 | 0.500 | 0.924 | 0.929 |  | |  |
| UR | 8 | +0.470 | -0.707 | 0.255 | 1.017 | 0.990 |  | |  |
| TBILL | 1 | +0.412 | +0.345 | 0.000 | 0.773 | 0.919 |  | |  |
| TBILL | 2 | +0.175 | +0.114 | 0.067 | 0.887 | 0.943 |  | |  |
| TBILL | 4 | -0.096 | -0.135 | 0.703 | 0.986 | 0.978 |  | |  |
| TBILL | 8 | -0.233 | -0.244 | 0.843 | 1.018 | 0.981 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.779 | +0.629 | 0.000 |
| joint | 2 | +0.581 | +0.398 | 0.001 |
| joint | 4 | +0.132 | -0.068 | 0.340 |
| joint | 8 | +1.729 | -0.538 | 0.201 |

### AR-SV(0.9) (converged at 0% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.103 | +0.101 | 0.001 | 0.912 | 0.926 |  | |  |
| GDP | 2 | +0.095 | +0.092 | 0.001 | 0.921 | 0.936 |  | |  |
| GDP | 4 | +0.090 | -0.015 | 0.001 | 0.960 | 0.986 |  | |  |
| GDP | 8 | +0.097 | +0.011 | 0.000 | 0.967 | 1.002 |  | |  |
| INFL | 1 | +0.242 | +0.212 | 0.000 | 0.841 | 0.895 |  | |  |
| INFL | 2 | +0.273 | +0.258 | 0.000 | 0.800 | 0.840 |  | |  |
| INFL | 4 | +0.261 | +0.251 | 0.000 | 0.773 | 0.797 |  | |  |
| INFL | 8 | +0.288 | +0.307 | 0.000 | 0.734 | 0.752 |  | |  |
| UR | 1 | +0.152 | +0.134 | 0.000 | 0.916 | 0.972 |  | |  |
| UR | 2 | +0.145 | +0.105 | 0.022 | 0.904 | 0.947 |  | |  |
| UR | 4 | +0.094 | -0.011 | 0.196 | 0.933 | 0.954 |  | |  |
| UR | 8 | +0.669 | -0.364 | 0.211 | 1.036 | 1.019 |  | |  |
| TBILL | 1 | +0.415 | +0.345 | 0.000 | 0.777 | 0.933 |  | |  |
| TBILL | 2 | +0.201 | +0.166 | 0.017 | 0.887 | 0.957 |  | |  |
| TBILL | 4 | -0.017 | -0.023 | 0.553 | 0.980 | 0.989 |  | |  |
| TBILL | 8 | -0.098 | -0.079 | 0.747 | 1.013 | 0.999 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |
| :- | -: | -: | -: | -: |
| joint | 1 | +0.773 | +0.636 | 0.000 |
| joint | 2 | +0.633 | +0.511 | 0.000 |
| joint | 4 | +0.392 | +0.238 | 0.039 |
| joint | 8 | +2.771 | +0.001 | 0.156 |

### `compare()` verdicts vs `Constant` (pre-crisis)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | :- | :- |
| SV(rw) | 1 | inconclusive | better | +71.6 | 0.54 | 0.000 | True | False |
| SV | 1 | inconclusive | better | +70.5 | 0.48 | 0.000 | True | False |
| SV(rw,t10) | 1 | inconclusive | better | +71.7 | 0.57 | 0.000 | True | False |
| AR-SV(0.9) | 1 | inconclusive | better | +71.1 | 0.47 | 0.000 | True | False |
| SV(rw) | 2 | inconclusive | better | +52.2 | 1.70 | 0.003 | True | False |
| SV | 2 | inconclusive | better | +58.5 | 1.36 | 0.000 | True | False |
| SV(rw,t10) | 2 | inconclusive | better | +53.4 | 1.78 | 0.001 | True | False |
| AR-SV(0.9) | 2 | inconclusive | better | +58.3 | 1.35 | 0.000 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +5.3 | 3.35 | 0.441 | True | True |
| SV | 4 | inconclusive | inconclusive | +34.9 | 2.61 | 0.053 | True | True |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +12.1 | 3.38 | 0.340 | True | True |
| AR-SV(0.9) | 4 | inconclusive | inconclusive | +36.1 | 2.56 | 0.039 | True | True |
| SV(rw) | 8 | inconclusive | inconclusive | +187.1 | 5.43 | 0.188 | True | True |
| SV | 8 | inconclusive | inconclusive | +246.4 | 4.77 | 0.156 | True | True |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +159.0 | 5.49 | 0.201 | True | True |
| AR-SV(0.9) | 8 | inconclusive | inconclusive | +254.9 | 4.75 | 0.156 | True | True |

## Variants vs RW and AR SV (criterion 5, full sample)

| variant | reference | h | joint ΔLPD | DM p (variant better) |
| :- | :- | -: | -: | -: |
| SV(rw,t10) | SV(rw) | 1 | +0.005 | 0.367 |
| SV(rw,t10) | SV(rw) | 2 | -0.042 | 0.774 |
| SV(rw,t10) | SV(rw) | 4 | -0.124 | 0.724 |
| SV(rw,t10) | SV(rw) | 8 | -0.496 | 0.875 |
| SV(rw,t10) | SV | 1 | -0.013 | 0.652 |
| SV(rw,t10) | SV | 2 | -0.193 | 0.946 |
| SV(rw,t10) | SV | 4 | -0.372 | 0.973 |
| SV(rw,t10) | SV | 8 | -1.163 | 0.922 |
| AR-SV(0.9) | SV(rw) | 1 | +0.027 | 0.237 |
| AR-SV(0.9) | SV(rw) | 2 | +0.156 | 0.052 |
| AR-SV(0.9) | SV(rw) | 4 | +0.307 | 0.031 |
| AR-SV(0.9) | SV(rw) | 8 | +0.711 | 0.056 |
| AR-SV(0.9) | SV | 1 | +0.009 | 0.069 |
| AR-SV(0.9) | SV | 2 | +0.005 | 0.398 |
| AR-SV(0.9) | SV | 4 | +0.058 | 0.066 |
| AR-SV(0.9) | SV | 8 | +0.043 | 0.202 |

## Structural-form pair (robustness, full sample)

| variable | h=1 ΔLPS | h=2 ΔLPS | h=8 ΔLPS |
| :- | -: | -: | -: |
| GDP | +0.105 | +0.111 | +0.107 |
| INFL | +0.232 | +0.258 | +0.293 |
| UR | +0.129 | +0.202 | +2.540 |
| TBILL | +0.313 | +0.063 | -0.222 |
| joint | +0.742 | +0.604 | +4.053 |

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
* `SV(rw)`, 106 origin-messages (106 variants), e.g.: MCMC not converged: 6 of 13 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: loglik R-hat 1.022, ESS 271; sigma[GDP] R-hat 1.013, ESS 223; sigma[UR] R-hat 1.017, ESS 236; h_T[GDP] R-hat 1.017, ESS 260; sigma[TBILL] R-hat 1.016, ESS 289); increase draws, burn or thin
* `SV(rw)`, 16 origin-messages (16 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0239, 0.0456]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw)`, 1 origin-messages (1 variants), e.g.: secondary (informational; does not gate converged): 17% of the 6 contemporaneous coefficients a_ij have R-hat ≥ 1.01 (worst R-hat: a[UR,GDP] 1.011)
* `SV`, 106 origin-messages (106 variants), e.g.: MCMC not converged: 7 of 21 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[TBILL] R-hat 1.026, ESS 191; loglik R-hat 1.021, ESS 206; phi[GDP] R-hat 1.018, ESS 465; sigma[UR] R-hat 1.016, ESS 256; sigma[GDP] R-hat 1.011, ESS 261); increase draws, burn or thin
* `SV`, 12 origin-messages (12 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0144, 0.0301]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV`, 3 origin-messages (3 variants), e.g.: secondary (informational; does not gate converged): 17% of the 6 contemporaneous coefficients a_ij have R-hat ≥ 1.01 (worst R-hat: a[UR,GDP] 1.017)
* `SV(rw,t10)`, 106 origin-messages (106 variants), e.g.: MCMC not converged: 5 of 13 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[UR] R-hat 1.035, ESS 205; h_T[GDP] R-hat 1.018, ESS 262; sigma[GDP] R-hat 1.015, ESS 229; sigma[TBILL] R-hat 1.014, ESS 309; loglik R-hat 1.013, ESS 438); increase draws, burn or thin
* `SV(rw,t10)`, 30 origin-messages (30 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL', 'UR'] (fractions [0.0451, 0.0661, 0.0185]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: Student-t ν has posterior mass ≥ 0.5 at the grid bound 10 for ['GDP', 'INFL', 'UR', 'TBILL'] (mass [1.0, 1.0, 1.0, 1.0]): near-Gaussian tails, SV may suffice (report only; mass at ν = 10 is normal, N03 §6)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `AR-SV(0.9)`, 106 origin-messages (106 variants), e.g.: MCMC not converged: 6 of 21 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: loglik R-hat 1.026, ESS 222; sigma[TBILL] R-hat 1.019, ESS 268; sigma[INFL] R-hat 1.019, ESS 443; sigma[GDP] R-hat 1.014, ESS 224; sigma[UR] R-hat 1.013, ESS 267); increase draws, burn or thin
* `AR-SV(0.9)`, 12 origin-messages (12 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0112, 0.03]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `AR-SV(0.9)`, 2 origin-messages (2 variants), e.g.: secondary (informational; does not gate converged): 17% of the 6 contemporaneous coefficients a_ij have R-hat ≥ 1.01 (worst R-hat: a[UR,GDP] 1.014)
