# Clark–Ravazzolo (2015) replication on FRED-QD (WP23)

The `acceptance` config is pre-registered in `benchmarks/reports/clark_ravazzolo_preregistration.md`; it departs from the reference full run (report at `551f181`, 4 × 5000 kept draws, burn 1000, the `conditioned` LPD) only in the MCMC budget (4 × 20000 kept draws, burn 1000, every MCMC stage) and the primary LPD estimator (`integrated`).

Generated 2026-10-23 16:31 UTC by `benchmarks/clark_ravazzolo.py` (config `acceptance`, rai-bvar 0.0.1, jax 0.12.0.1, Darwin arm64, python 3.12.13) at commit `c823d207e0525aed98f427f437d687f7f131c92c` (clean working tree). Run time of this invocation 3.58 h (0 of 742 origin backtests of the main and structural stages were read from the score cache `/Users/yigitokar/.cache/rai-bvar-cr-c823d20` instead of fitted; cache keys do not change with the code, so a run after a code change needs a fresh cache). Primary LPD estimator: `lpd_method='integrated'`; MCMC budget: 4 chains × 20000 kept draws, burn 1000, thin 1, in every stage; parameters of one in 8 kept draws stored (`store_max` 2500; R-hat and ESS use every kept draw); 8000 forecast draws per origin, from the stored draws.

## Acceptance criteria (N08 §C5)

| # | Met | Operational rule | Result |
| :- | :- | :- | :- |
| 1 | **met** | ΔLPS(SV(rw) − Constant) > 0 at h = 1, 2 for all four variables in the full and pre-crisis samples, and the joint ΔLPD at h = 1 > 0 with one-sided DM/HLN p < 0.10 in both samples | 16 of 16 cells positive; joint h=1: full +0.819 nats/quarter (p=0.000), pre +0.791 nats/quarter (p=0.000) |
| 2 | **met** | CRPS ratio SV(rw)/Constant < 1 at h = 1, 2 for all four variables (full sample) | 8 of 8 cells < 1; pre-crisis: all 8 < 1 |
| 3 | **met** | RMSE ratio SV(rw)/Constant ≤ 1.02 ('about 1') at h = 1, 2 for all four variables (full sample); CR's gains for GDP and the bill are 5–10% | GDP/h1 0.908, GDP/h2 0.901, INFL/h1 0.899, INFL/h2 0.848, UR/h1 0.969, UR/h2 0.946, TBILL/h1 0.935, TBILL/h2 0.945 |
| 4 | **not met** | full sample: ΔLPS(h=8) < ΔLPS(h=1) for UR and TBILL; pre-crisis: ΔLPS(h=8) > 0 for GDP and INFL | full UR: h1 +0.160 → h8 +0.273; full TBILL: h1 +0.377 → h8 -0.195; pre-crisis GDP h8 +0.105; pre-crisis INFL h8 +0.307. Secondary (not part of the rule): full UR h8: median +0.140 (largest origins: 2007Q4 +7.6, 2008Q1 +6.6, 2008Q2 +6.5), Gaussian-approx h1 +0.122 → h8 -0.526; full TBILL h8: median -0.059 (largest origins: 2000Q4 -2.3, 2000Q2 -2.2, 2000Q3 -2.1), Gaussian-approx h1 +0.315 → h8 -0.230; pre-crisis GDP h8 Gaussian-approx -0.001; pre-crisis INFL h8 Gaussian-approx +0.354; path-conditioned (SPEC §7) estimator ΔLPS: full UR: h1 +0.160 → h8 +1.309; full TBILL: h1 +0.377 → h8 -0.193; pre-crisis GDP h8 +0.105; pre-crisis INFL h8 +0.307 |
| 5 | **met** | neither SV(rw,t10) nor AR-SV(0.9) is better than both SV(rw) and SV (one-sided DM/HLN p < 0.10, positive ΔLPD) at ≥ 3 of 4 joint horizons or ≥ 8 of 16 marginal cells (full sample) | SV(rw,t10): significantly better than both SV(rw) and SV at 0/4 joint horizons and 0/16 marginal cells; AR-SV(0.9): significantly better than both SV(rw) and SV at 0/4 joint horizons and 0/16 marginal cells |
| 6 | **met** | in every one of the 24 orderings: compare() verdict SV(rw) vs Constant at h = 1 is 'better', and the one-sided DM/HLN p-value of the marginal log score is < 0.10 for GDP and TBILL at h = 1 | verdict 'better' (convergence gate on, as compare() returns it) in 24/24 orderings; with the gate off in 24/24; DM GDP and TBILL < 0.10 in 24/24 |

## Design

* Data: FRED-QD 2026-09 (SHA-256 verified), GDP = 100·Δlog GDPC1, INFL = 100·Δlog GDPCTPI, UR = UNRATE, TBILL = TB3MS (levels); non-annualised quarterly rates; estimation sample starts 1965Q1; pseudo real time on the final vintage (D4).
* Asserted (§C6): FRED-QD UNRATE/TB3MS equal within-quarter means of FRED-MD monthly values of the same vintage — UNRATE: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2); TB3MS: max |diff| 0.0000, mean 0.00002 over 270 quarters (1959Q1–2026Q2).
* Asserted (R9, §C6): at origins 1985Q1, 2007Q4, 2011Q2 the resolved priors, hyperparameters and padded data are bitwise identical when the vintage is cut at the origin (no look-ahead; CR's last-vintage SV training sample is not replicated).
* Origins: 1985Q1–2011Q2 (106, full) and 1985Q1–2007Q4 (92, pre-crisis subset); expanding window; h = 1, 2, 4, 8.
* MCMC: 4 chains × 20000 kept draws, burn 1000, thin 1; 8000 forecast draws per origin; backtest seed 0; per-origin fit seeds (DESIGN §11.6); cold starts.
* Mean prior (all models): `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, alpha=2, cross=1, mean=0)`, p = 4. Volatility arms: `Constant` = `Constant(iw_equivalent)`, `SV(rw)` = `SV(rw)`, `SV` = `SV`, `SV(rw,t10)` = `SV(rw,t, nu_prior=7eca60, t_scale=unit)`, `AR-SV(0.9)` = `SV(phi_prior=7ea4a9)`.
* LPD is the library's Rao–Blackwellised predictive density (primary; estimator `lpd_method='integrated'`, see below); `ΔLPS gauss` is CR's Gaussian approximation (Adolfson et al. 2007) from the same draws.

### Reliability of the RB log score

The primary LPD is the library's **path-integrated** Rao–Blackwellised estimator (`lpd_method='integrated'`, DESIGN §7.3, D-34): for h > 1 it averages, over the forecast draws, the exact Gaussian density of y_{T+h} given each draw's parameters and its drawn latent scale path (SV log-volatilities; the t arm's mixing scales), with y_{T+1..T+h−1} integrated out analytically. It uses no additional random numbers, but it is still a Monte Carlo average over the forecast's parameter and volatility-path draws. At h = 1 it is the SPEC §7 estimator, bitwise. It replaces the path-conditioned SPEC §7 estimator, which for h > 1 also conditions on each draw's simulated path up to T+h−1: when the outcome lies far in a predictive's tail (unemployment in 2008–10) one forecast draw carries almost all of its weight, the estimate is biased low by up to tens of nats and its delta-method MCSE saturates near 1 (in the review of the quick run, Constant's UR h = 8 score at 2008Q2 was −93.6 with 1000 draws against −19.1 path-integrated). The integrated estimator's weights still concentrate in the tails, less so: in the review at the acceptance config's budget (4 × 20,000; 2008Q2, UR h = 8, 1000 forecast draws) Constant's weight ESS was 2–6 draws and its estimate on average about 0.3–0.6 nats below the 80,000-draw value (−19.0 against −17.8 with 8000 draws at fit seed 1), and SV(rw)'s spread with sd ≈ 1.4 nats across forecast seeds against MCSEs of 0.45–0.94. Cells at h ≥ 4 with crisis origins therefore carry about a nat of Monte Carlo noise per arm and origin at 1000 draws, which the MCSE understates. The acceptance config therefore uses 8000 forecast draws: in the same review they removed the thinning error (sd 0.09–0.19 nats across thinning offsets, bias ≈ 0) but not the MCMC error (the 80,000-draw values still differ by about 1.3 nats across fit seeds at that origin). The path-conditioned ΔLPS is shown for the criterion-4 cells as a secondary number (information only, never part of a rule). The column **RB degen.** counts origins whose joint `lpd_mcse` of the primary estimator is ≥ 0.9 for the arm or the baseline (the weights then sit on about one parameter draw); such origins are kept, never dropped. The criteria are evaluated as specified, on the primary estimator.

RB and Gaussian-approx ΔLPS differ in sign (N08 §C6 expects the rankings to agree) in: full SV(rw) GDP h4, full SV(rw) UR h4, full SV(rw) UR h8, full SV UR h4, full SV UR h8, full SV TBILL h4, full SV(rw,t10) GDP h4, full SV(rw,t10) UR h2, full SV(rw,t10) UR h4, full SV(rw,t10) UR h8, full AR-SV(0.9) UR h4, full AR-SV(0.9) UR h8, full AR-SV(0.9) TBILL h4, pre-crisis SV(rw) GDP h4, pre-crisis SV(rw) GDP h8, pre-crisis SV(rw) UR h4, pre-crisis SV(rw) UR h8, pre-crisis SV GDP h4, pre-crisis SV UR h4, pre-crisis SV UR h8, pre-crisis SV TBILL h4, pre-crisis SV(rw,t10) GDP h4, pre-crisis SV(rw,t10) UR h4, pre-crisis SV(rw,t10) UR h8, pre-crisis AR-SV(0.9) GDP h4, pre-crisis AR-SV(0.9) UR h4, pre-crisis AR-SV(0.9) UR h8, pre-crisis AR-SV(0.9) TBILL h4.

## Full sample: each SV arm vs `Constant`

### SV(rw) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.126 | +0.103 | 0 | 0.002 | 0.897 | 0.908 | +0.193 | 0.864 | 0.896 |
| GDP | 2 | +0.124 | +0.065 | 0 | 0.002 | 0.894 | 0.901 | +0.133 | 0.856 | 0.878 |
| GDP | 4 | +0.089 | -0.061 | 0 | 0.063 | 0.927 | 0.944 | +0.090 | 0.890 | 0.932 |
| GDP | 8 | +0.108 | +0.010 | 0 | 0.008 | 0.940 | 0.977 | +0.060 | 0.927 | 1.007 |
| INFL | 1 | +0.233 | +0.210 | 0 | 0.000 | 0.845 | 0.899 | +0.075 | 0.954 | 0.966 |
| INFL | 2 | +0.269 | +0.270 | 0 | 0.000 | 0.807 | 0.848 | +0.083 | 0.944 | 0.949 |
| INFL | 4 | +0.255 | +0.257 | 0 | 0.000 | 0.767 | 0.788 | +0.155 | 0.895 | 0.915 |
| INFL | 8 | +0.293 | +0.326 | 0 | 0.000 | 0.720 | 0.725 | +0.138 | 0.887 | 0.883 |
| UR | 1 | +0.160 | +0.122 | 0 | 0.008 | 0.936 | 0.969 | +0.162 | 0.922 | 0.967 |
| UR | 2 | +0.206 | +0.048 | 0 | 0.082 | 0.924 | 0.946 | +0.098 | 0.897 | 0.930 |
| UR | 4 | +0.251 | -0.248 | 0 | 0.172 | 0.929 | 0.937 | -0.182 | 0.906 | 0.914 |
| UR | 8 | +0.273 | -0.526 | 0 | 0.282 | 0.981 | 0.965 | -0.707 | 1.001 | 0.971 |
| TBILL | 1 | +0.377 | +0.315 | 0 | 0.000 | 0.798 | 0.935 | +0.284 | 0.822 | 0.933 |
| TBILL | 2 | +0.153 | +0.095 | 0 | 0.077 | 0.895 | 0.945 | +0.043 | 0.915 | 0.944 |
| TBILL | 4 | -0.066 | -0.142 | 0 | 0.681 | 0.983 | 0.968 | -0.304 | 1.010 | 0.974 |
| TBILL | 8 | -0.195 | -0.230 | 0 | 0.857 | 1.019 | 0.986 | -0.403 | 1.059 | 0.990 |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.819 | +0.654 | 0 | 0.000 |
| joint | 2 | +0.743 | +0.463 | 0 | 0.003 |
| joint | 4 | +0.673 | +0.002 | 0 | 0.056 |
| joint | 8 | +0.604 | -0.350 | 0 | 0.119 |

### SV (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.129 | +0.113 | 0 | 0.001 | 0.904 | 0.917 |  | |  |
| GDP | 2 | +0.133 | +0.094 | 0 | 0.001 | 0.903 | 0.912 |  | |  |
| GDP | 4 | +0.097 | +0.002 | 0 | 0.000 | 0.935 | 0.949 |  | |  |
| GDP | 8 | +0.099 | +0.022 | 0 | 0.000 | 0.945 | 0.970 |  | |  |
| INFL | 1 | +0.218 | +0.192 | 0 | 0.000 | 0.853 | 0.902 |  | |  |
| INFL | 2 | +0.251 | +0.240 | 0 | 0.000 | 0.817 | 0.854 |  | |  |
| INFL | 4 | +0.236 | +0.233 | 0 | 0.000 | 0.784 | 0.801 |  | |  |
| INFL | 8 | +0.280 | +0.292 | 0 | 0.000 | 0.737 | 0.749 |  | |  |
| UR | 1 | +0.175 | +0.142 | 0 | 0.002 | 0.952 | 1.000 |  | |  |
| UR | 2 | +0.246 | +0.131 | 0 | 0.046 | 0.941 | 0.973 |  | |  |
| UR | 4 | +0.310 | -0.015 | 0 | 0.117 | 0.945 | 0.957 |  | |  |
| UR | 8 | +0.322 | -0.185 | 0 | 0.227 | 0.996 | 0.979 |  | |  |
| TBILL | 1 | +0.378 | +0.305 | 0 | 0.000 | 0.799 | 0.946 |  | |  |
| TBILL | 2 | +0.187 | +0.149 | 0 | 0.011 | 0.890 | 0.953 |  | |  |
| TBILL | 4 | +0.006 | -0.017 | 0 | 0.474 | 0.973 | 0.975 |  | |  |
| TBILL | 8 | -0.092 | -0.074 | 0 | 0.771 | 1.010 | 0.995 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.830 | +0.678 | 0 | 0.000 |
| joint | 2 | +0.835 | +0.648 | 0 | 0.001 |
| joint | 4 | +0.845 | +0.469 | 0 | 0.025 |
| joint | 8 | +0.760 | +0.232 | 0 | 0.053 |

### SV(rw,t10) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.135 | +0.104 | 0 | 0.001 | 0.891 | 0.902 |  | |  |
| GDP | 2 | +0.138 | +0.066 | 0 | 0.000 | 0.886 | 0.894 |  | |  |
| GDP | 4 | +0.113 | -0.023 | 0 | 0.008 | 0.913 | 0.932 |  | |  |
| GDP | 8 | +0.128 | +0.038 | 0 | 0.001 | 0.927 | 0.966 |  | |  |
| INFL | 1 | +0.222 | +0.200 | 0 | 0.000 | 0.852 | 0.903 |  | |  |
| INFL | 2 | +0.255 | +0.257 | 0 | 0.000 | 0.817 | 0.857 |  | |  |
| INFL | 4 | +0.241 | +0.247 | 0 | 0.000 | 0.782 | 0.804 |  | |  |
| INFL | 8 | +0.268 | +0.304 | 0 | 0.000 | 0.742 | 0.750 |  | |  |
| UR | 1 | +0.153 | +0.093 | 0 | 0.006 | 0.940 | 0.969 |  | |  |
| UR | 2 | +0.198 | -0.002 | 0 | 0.069 | 0.928 | 0.947 |  | |  |
| UR | 4 | +0.210 | -0.296 | 0 | 0.171 | 0.933 | 0.937 |  | |  |
| UR | 8 | +0.224 | -0.581 | 0 | 0.288 | 0.978 | 0.959 |  | |  |
| TBILL | 1 | +0.371 | +0.300 | 0 | 0.000 | 0.801 | 0.937 |  | |  |
| TBILL | 2 | +0.152 | +0.097 | 0 | 0.070 | 0.898 | 0.946 |  | |  |
| TBILL | 4 | -0.061 | -0.130 | 0 | 0.673 | 0.984 | 0.970 |  | |  |
| TBILL | 8 | -0.192 | -0.228 | 0 | 0.851 | 1.017 | 0.985 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.813 | +0.627 | 0 | 0.000 |
| joint | 2 | +0.735 | +0.421 | 0 | 0.002 |
| joint | 4 | +0.623 | -0.019 | 0 | 0.046 |
| joint | 8 | +0.552 | -0.387 | 0 | 0.110 |

### AR-SV(0.9) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.131 | +0.117 | 0 | 0.001 | 0.904 | 0.916 |  | |  |
| GDP | 2 | +0.135 | +0.097 | 0 | 0.002 | 0.902 | 0.911 |  | |  |
| GDP | 4 | +0.101 | +0.012 | 0 | 0.000 | 0.934 | 0.950 |  | |  |
| GDP | 8 | +0.101 | +0.025 | 0 | 0.000 | 0.945 | 0.971 |  | |  |
| INFL | 1 | +0.218 | +0.194 | 0 | 0.000 | 0.853 | 0.901 |  | |  |
| INFL | 2 | +0.250 | +0.239 | 0 | 0.000 | 0.817 | 0.853 |  | |  |
| INFL | 4 | +0.238 | +0.234 | 0 | 0.000 | 0.784 | 0.801 |  | |  |
| INFL | 8 | +0.279 | +0.287 | 0 | 0.000 | 0.737 | 0.749 |  | |  |
| UR | 1 | +0.177 | +0.145 | 0 | 0.002 | 0.950 | 0.998 |  | |  |
| UR | 2 | +0.248 | +0.132 | 0 | 0.046 | 0.938 | 0.970 |  | |  |
| UR | 4 | +0.314 | -0.015 | 0 | 0.117 | 0.942 | 0.955 |  | |  |
| UR | 8 | +0.336 | -0.156 | 0 | 0.225 | 0.994 | 0.978 |  | |  |
| TBILL | 1 | +0.380 | +0.311 | 0 | 0.000 | 0.799 | 0.946 |  | |  |
| TBILL | 2 | +0.187 | +0.151 | 0 | 0.011 | 0.890 | 0.953 |  | |  |
| TBILL | 4 | +0.005 | -0.021 | 0 | 0.480 | 0.973 | 0.975 |  | |  |
| TBILL | 8 | -0.092 | -0.071 | 0 | 0.772 | 1.010 | 0.994 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.837 | +0.695 | 0 | 0.000 |
| joint | 2 | +0.838 | +0.659 | 0 | 0.001 |
| joint | 4 | +0.846 | +0.482 | 0 | 0.023 |
| joint | 8 | +0.785 | +0.253 | 0 | 0.055 |

### `compare()` verdicts vs `Constant` (full)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | RB degen. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | -: | :- | :- |
| SV(rw) | 1 | better | better | +86.8 | 0.40 | 0 | 0.000 | True | False |
| SV | 1 | better | better | +88.0 | 0.38 | 0 | 0.000 | True | False |
| SV(rw,t10) | 1 | better | better | +86.2 | 0.43 | 0 | 0.000 | True | False |
| AR-SV(0.9) | 1 | better | better | +88.7 | 0.37 | 0 | 0.000 | True | False |
| SV(rw) | 2 | inconclusive | inconclusive | +78.7 | 0.64 | 0 | 0.003 | False | False |
| SV | 2 | better | better | +88.5 | 0.59 | 0 | 0.001 | True | False |
| SV(rw,t10) | 2 | inconclusive | inconclusive | +77.9 | 0.67 | 0 | 0.002 | False | False |
| AR-SV(0.9) | 2 | better | better | +88.9 | 0.55 | 0 | 0.001 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +71.4 | 1.56 | 0 | 0.056 | False | False |
| SV | 4 | better | better | +89.6 | 1.21 | 0 | 0.025 | True | False |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +66.1 | 1.32 | 0 | 0.046 | False | False |
| AR-SV(0.9) | 4 | better | better | +89.6 | 1.10 | 0 | 0.023 | True | False |
| SV(rw) | 8 | inconclusive | inconclusive | +64.0 | 1.56 | 0 | 0.119 | False | False |
| SV | 8 | inconclusive | inconclusive | +80.5 | 1.21 | 0 | 0.053 | False | False |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +58.5 | 1.82 | 0 | 0.110 | False | False |
| AR-SV(0.9) | 8 | better | better | +83.2 | 1.31 | 0 | 0.055 | True | False |

## Pre-crisis: each SV arm vs `Constant`

### SV(rw) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.115 | +0.121 | 0 | 0.006 | 0.901 | 0.915 | +0.222 | |  |
| GDP | 2 | +0.109 | +0.120 | 0 | 0.010 | 0.908 | 0.924 |  | |  |
| GDP | 4 | +0.091 | -0.073 | 0 | 0.060 | 0.948 | 0.977 |  | |  |
| GDP | 8 | +0.105 | -0.001 | 0 | 0.028 | 0.959 | 1.007 | +0.214 | |  |
| INFL | 1 | +0.260 | +0.233 | 0 | 0.000 | 0.830 | 0.895 | +0.077 | |  |
| INFL | 2 | +0.297 | +0.295 | 0 | 0.000 | 0.790 | 0.835 |  | |  |
| INFL | 4 | +0.277 | +0.277 | 0 | 0.000 | 0.755 | 0.782 |  | |  |
| INFL | 8 | +0.307 | +0.354 | 0 | 0.000 | 0.714 | 0.725 | +0.152 | |  |
| UR | 1 | +0.148 | +0.126 | 0 | 0.002 | 0.907 | 0.952 | +0.187 | |  |
| UR | 2 | +0.119 | +0.060 | 0 | 0.107 | 0.897 | 0.930 |  | |  |
| UR | 4 | +0.007 | -0.135 | 0 | 0.482 | 0.929 | 0.937 |  | |  |
| UR | 8 | +0.026 | -0.683 | 0 | 0.467 | 1.029 | 1.003 | -0.065 | |  |
| TBILL | 1 | +0.423 | +0.370 | 0 | 0.000 | 0.770 | 0.921 | +0.316 | |  |
| TBILL | 2 | +0.175 | +0.116 | 0 | 0.077 | 0.885 | 0.944 |  | |  |
| TBILL | 4 | -0.065 | -0.150 | 0 | 0.655 | 0.983 | 0.975 |  | |  |
| TBILL | 8 | -0.205 | -0.249 | 0 | 0.830 | 1.014 | 0.977 | -0.261 | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.791 | +0.671 | 0 | 0.000 |
| joint | 2 | +0.593 | +0.418 | 0 | 0.002 |
| joint | 4 | +0.319 | -0.128 | 0 | 0.113 |
| joint | 8 | +0.330 | -0.527 | 0 | 0.138 |

### SV (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.102 | +0.100 | 0 | 0.001 | 0.912 | 0.927 |  | |  |
| GDP | 2 | +0.094 | +0.091 | 0 | 0.001 | 0.921 | 0.937 |  | |  |
| GDP | 4 | +0.091 | -0.021 | 0 | 0.001 | 0.959 | 0.983 |  | |  |
| GDP | 8 | +0.094 | +0.006 | 0 | 0.000 | 0.966 | 1.000 |  | |  |
| INFL | 1 | +0.242 | +0.211 | 0 | 0.000 | 0.841 | 0.899 |  | |  |
| INFL | 2 | +0.274 | +0.259 | 0 | 0.000 | 0.803 | 0.843 |  | |  |
| INFL | 4 | +0.254 | +0.246 | 0 | 0.000 | 0.775 | 0.796 |  | |  |
| INFL | 8 | +0.289 | +0.311 | 0 | 0.000 | 0.734 | 0.751 |  | |  |
| UR | 1 | +0.154 | +0.129 | 0 | 0.000 | 0.917 | 0.973 |  | |  |
| UR | 2 | +0.142 | +0.103 | 0 | 0.022 | 0.907 | 0.948 |  | |  |
| UR | 4 | +0.069 | -0.005 | 0 | 0.258 | 0.936 | 0.954 |  | |  |
| UR | 8 | +0.107 | -0.350 | 0 | 0.344 | 1.034 | 1.018 |  | |  |
| TBILL | 1 | +0.415 | +0.348 | 0 | 0.000 | 0.777 | 0.934 |  | |  |
| TBILL | 2 | +0.205 | +0.167 | 0 | 0.013 | 0.886 | 0.957 |  | |  |
| TBILL | 4 | +0.009 | -0.017 | 0 | 0.468 | 0.980 | 0.988 |  | |  |
| TBILL | 8 | -0.095 | -0.080 | 0 | 0.743 | 1.013 | 0.998 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.776 | +0.641 | 0 | 0.000 |
| joint | 2 | +0.651 | +0.523 | 0 | 0.000 |
| joint | 4 | +0.471 | +0.234 | 0 | 0.007 |
| joint | 8 | +0.537 | +0.053 | 0 | 0.027 |

### SV(rw,t10) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.118 | +0.127 | 0 | 0.003 | 0.895 | 0.907 |  | |  |
| GDP | 2 | +0.115 | +0.126 | 0 | 0.004 | 0.899 | 0.915 |  | |  |
| GDP | 4 | +0.116 | -0.032 | 0 | 0.007 | 0.934 | 0.964 |  | |  |
| GDP | 8 | +0.121 | +0.021 | 0 | 0.008 | 0.948 | 0.997 |  | |  |
| INFL | 1 | +0.252 | +0.224 | 0 | 0.000 | 0.837 | 0.900 |  | |  |
| INFL | 2 | +0.284 | +0.282 | 0 | 0.000 | 0.800 | 0.845 |  | |  |
| INFL | 4 | +0.265 | +0.267 | 0 | 0.000 | 0.770 | 0.798 |  | |  |
| INFL | 8 | +0.279 | +0.329 | 0 | 0.001 | 0.738 | 0.751 |  | |  |
| UR | 1 | +0.148 | +0.123 | 0 | 0.001 | 0.907 | 0.950 |  | |  |
| UR | 2 | +0.121 | +0.066 | 0 | 0.079 | 0.897 | 0.926 |  | |  |
| UR | 4 | +0.012 | -0.107 | 0 | 0.466 | 0.926 | 0.930 |  | |  |
| UR | 8 | +0.016 | -0.648 | 0 | 0.477 | 1.014 | 0.990 |  | |  |
| TBILL | 1 | +0.415 | +0.351 | 0 | 0.000 | 0.774 | 0.923 |  | |  |
| TBILL | 2 | +0.173 | +0.116 | 0 | 0.071 | 0.889 | 0.946 |  | |  |
| TBILL | 4 | -0.064 | -0.141 | 0 | 0.655 | 0.987 | 0.979 |  | |  |
| TBILL | 8 | -0.207 | -0.253 | 0 | 0.830 | 1.017 | 0.981 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.788 | +0.661 | 0 | 0.000 |
| joint | 2 | +0.601 | +0.435 | 0 | 0.001 |
| joint | 4 | +0.333 | -0.048 | 0 | 0.094 |
| joint | 8 | +0.317 | -0.477 | 0 | 0.134 |

### AR-SV(0.9) (converged at 100% of origins; baseline 100%)

| variable | h | ΔLPS | ΔLPS gauss | RB degen. | DM p (1-sided) | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.104 | +0.103 | 0 | 0.001 | 0.912 | 0.926 |  | |  |
| GDP | 2 | +0.095 | +0.093 | 0 | 0.001 | 0.921 | 0.937 |  | |  |
| GDP | 4 | +0.094 | -0.011 | 0 | 0.001 | 0.959 | 0.984 |  | |  |
| GDP | 8 | +0.097 | +0.010 | 0 | 0.000 | 0.966 | 1.001 |  | |  |
| INFL | 1 | +0.243 | +0.213 | 0 | 0.000 | 0.841 | 0.899 |  | |  |
| INFL | 2 | +0.274 | +0.256 | 0 | 0.000 | 0.803 | 0.843 |  | |  |
| INFL | 4 | +0.257 | +0.248 | 0 | 0.000 | 0.775 | 0.796 |  | |  |
| INFL | 8 | +0.288 | +0.305 | 0 | 0.000 | 0.734 | 0.752 |  | |  |
| UR | 1 | +0.156 | +0.132 | 0 | 0.000 | 0.916 | 0.972 |  | |  |
| UR | 2 | +0.143 | +0.102 | 0 | 0.025 | 0.906 | 0.947 |  | |  |
| UR | 4 | +0.069 | -0.013 | 0 | 0.266 | 0.935 | 0.954 |  | |  |
| UR | 8 | +0.115 | -0.336 | 0 | 0.338 | 1.033 | 1.019 |  | |  |
| TBILL | 1 | +0.418 | +0.354 | 0 | 0.000 | 0.776 | 0.934 |  | |  |
| TBILL | 2 | +0.206 | +0.170 | 0 | 0.014 | 0.886 | 0.956 |  | |  |
| TBILL | 4 | +0.008 | -0.021 | 0 | 0.472 | 0.980 | 0.988 |  | |  |
| TBILL | 8 | -0.095 | -0.078 | 0 | 0.745 | 1.013 | 0.998 |  | |  |

| joint | h | ΔLPD | ΔLPD gauss | RB degen. | DM p (1-sided) |
| :- | -: | -: | -: | -: | -: |
| joint | 1 | +0.781 | +0.659 | 0 | 0.000 |
| joint | 2 | +0.654 | +0.530 | 0 | 0.000 |
| joint | 4 | +0.479 | +0.240 | 0 | 0.006 |
| joint | 8 | +0.555 | +0.060 | 0 | 0.030 |

### `compare()` verdicts vs `Constant` (pre-crisis)

| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | RB degen. | DM p | model in MCS25 | baseline in MCS25 |
| :- | -: | :- | :- | -: | -: | -: | -: | :- | :- |
| SV(rw) | 1 | better | better | +72.8 | 0.20 | 0 | 0.000 | True | False |
| SV | 1 | better | better | +71.4 | 0.17 | 0 | 0.000 | True | False |
| SV(rw,t10) | 1 | better | better | +72.5 | 0.21 | 0 | 0.000 | True | False |
| AR-SV(0.9) | 1 | better | better | +71.9 | 0.17 | 0 | 0.000 | True | False |
| SV(rw) | 2 | better | better | +54.6 | 0.24 | 0 | 0.002 | True | False |
| SV | 2 | better | better | +59.9 | 0.18 | 0 | 0.000 | True | False |
| SV(rw,t10) | 2 | better | better | +55.3 | 0.24 | 0 | 0.001 | True | False |
| AR-SV(0.9) | 2 | better | better | +60.1 | 0.18 | 0 | 0.000 | True | False |
| SV(rw) | 4 | inconclusive | inconclusive | +29.4 | 0.64 | 0 | 0.113 | False | False |
| SV | 4 | better | better | +43.4 | 0.32 | 0 | 0.007 | True | False |
| SV(rw,t10) | 4 | inconclusive | inconclusive | +30.7 | 0.44 | 0 | 0.094 | False | False |
| AR-SV(0.9) | 4 | better | better | +44.0 | 0.32 | 0 | 0.006 | True | False |
| SV(rw) | 8 | inconclusive | inconclusive | +30.4 | 1.21 | 0 | 0.138 | False | False |
| SV | 8 | better | better | +49.4 | 0.93 | 0 | 0.027 | True | False |
| SV(rw,t10) | 8 | inconclusive | inconclusive | +29.1 | 1.53 | 0 | 0.134 | False | False |
| AR-SV(0.9) | 8 | better | better | +51.0 | 1.02 | 0 | 0.030 | True | False |

## Variants vs RW and AR SV (criterion 5, full sample)

| variant | reference | h | joint ΔLPD | DM p (variant better) |
| :- | :- | -: | -: | -: |
| SV(rw,t10) | SV(rw) | 1 | -0.006 | 0.646 |
| SV(rw,t10) | SV(rw) | 2 | -0.007 | 0.630 |
| SV(rw,t10) | SV(rw) | 4 | -0.050 | 0.788 |
| SV(rw,t10) | SV(rw) | 8 | -0.052 | 0.775 |
| SV(rw,t10) | SV | 1 | -0.017 | 0.692 |
| SV(rw,t10) | SV | 2 | -0.100 | 0.959 |
| SV(rw,t10) | SV | 4 | -0.222 | 0.963 |
| SV(rw,t10) | SV | 8 | -0.208 | 0.987 |
| AR-SV(0.9) | SV(rw) | 1 | +0.018 | 0.309 |
| AR-SV(0.9) | SV(rw) | 2 | +0.096 | 0.048 |
| AR-SV(0.9) | SV(rw) | 4 | +0.172 | 0.023 |
| AR-SV(0.9) | SV(rw) | 8 | +0.181 | 0.020 |
| AR-SV(0.9) | SV | 1 | +0.007 | 0.089 |
| AR-SV(0.9) | SV | 2 | +0.003 | 0.276 |
| AR-SV(0.9) | SV | 4 | +0.000 | 0.487 |
| AR-SV(0.9) | SV | 8 | +0.025 | 0.140 |

## Structural-form pair (robustness, full sample)

| variable | h=1 ΔLPS | h=2 ΔLPS | h=8 ΔLPS |
| :- | -: | -: | -: |
| GDP | +0.102 | +0.105 | +0.103 |
| INFL | +0.233 | +0.263 | +0.300 |
| UR | +0.132 | +0.157 | +0.140 |
| TBILL | +0.315 | +0.066 | -0.228 |
| joint | +0.749 | +0.538 | +0.258 |

## All 24 orderings (criterion 6)

Origins in this stage: 106 (1985Q1–2011Q2). Each ordering's SV(rw) and Constant backtests are compared with `compare()`; the Constant arm is order-invariant in theory (Chan 2022), so its spread is Monte Carlo noise.

| # | ordering | verdict h=1 (gate on) | gate off | joint ΔLPD h=1 | DM p joint | DM p GDP | DM p TBILL | SV converged | Const converged |
| -: | :- | :- | :- | -: | -: | -: | -: | -: | -: |
| 0 | GDP > INFL > UR > TBILL | better | better | +0.819 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 1 | GDP > INFL > TBILL > UR | better | better | +0.866 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 2 | GDP > UR > INFL > TBILL | better | better | +0.812 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 3 | GDP > UR > TBILL > INFL | better | better | +0.800 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 4 | GDP > TBILL > INFL > UR | better | better | +0.859 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 5 | GDP > TBILL > UR > INFL | better | better | +0.853 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 6 | INFL > GDP > UR > TBILL | better | better | +0.825 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 7 | INFL > GDP > TBILL > UR | better | better | +0.863 | 0.000 | 0.002 | 0.000 | 100% | 100% |
| 8 | INFL > UR > GDP > TBILL | better | better | +0.824 | 0.000 | 0.006 | 0.000 | 100% | 100% |
| 9 | INFL > UR > TBILL > GDP | better | better | +0.821 | 0.000 | 0.005 | 0.000 | 100% | 100% |
| 10 | INFL > TBILL > GDP > UR | better | better | +0.876 | 0.000 | 0.000 | 0.000 | 100% | 100% |
| 11 | INFL > TBILL > UR > GDP | better | better | +0.873 | 0.000 | 0.001 | 0.000 | 100% | 100% |
| 12 | UR > GDP > INFL > TBILL | better | better | +0.826 | 0.000 | 0.008 | 0.000 | 100% | 100% |
| 13 | UR > GDP > TBILL > INFL | better | better | +0.810 | 0.000 | 0.007 | 0.000 | 100% | 100% |
| 14 | UR > INFL > GDP > TBILL | better | better | +0.822 | 0.000 | 0.006 | 0.000 | 100% | 100% |
| 15 | UR > INFL > TBILL > GDP | better | better | +0.829 | 0.000 | 0.005 | 0.000 | 100% | 100% |
| 16 | UR > TBILL > GDP > INFL | better | better | +0.812 | 0.000 | 0.006 | 0.000 | 100% | 100% |
| 17 | UR > TBILL > INFL > GDP | better | better | +0.815 | 0.000 | 0.005 | 0.000 | 100% | 100% |
| 18 | TBILL > GDP > INFL > UR | better | better | +0.868 | 0.000 | 0.000 | 0.000 | 100% | 100% |
| 19 | TBILL > GDP > UR > INFL | better | better | +0.869 | 0.000 | 0.000 | 0.000 | 100% | 100% |
| 20 | TBILL > INFL > GDP > UR | better | better | +0.876 | 0.000 | 0.000 | 0.000 | 100% | 100% |
| 21 | TBILL > INFL > UR > GDP | better | better | +0.877 | 0.000 | 0.001 | 0.000 | 100% | 100% |
| 22 | TBILL > UR > GDP > INFL | better | better | +0.871 | 0.000 | 0.001 | 0.000 | 100% | 100% |
| 23 | TBILL > UR > INFL > GDP | better | better | +0.871 | 0.000 | 0.001 | 0.000 | 100% | 100% |

`ordering_sensitivity` for `Constant`: material = False; joint LPD range over orderings (nats per origin): h=1: 0.008, h=2: 0.044, h=4: 0.068, h=8: 0.049.

`ordering_sensitivity` for `SV(rw)`: material = True; joint LPD range over orderings (nats per origin): h=1: 0.076, h=2: 0.184, h=4: 0.240, h=8: 0.195.

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
* LPD estimator: the primary score is the library's path-integrated Rao–Blackwellised estimator (`lpd_method='integrated'`, DESIGN §7.3, D-34), not the path-conditioned SPEC §7 default; at h = 1 the two are identical, so every h = 1 cell and criterion 6 (h = 1 only) are estimator-independent. The path-conditioned ΔLPS is reported for the criterion-4 cells only, as a secondary number (never in a rule).

## Warnings raised during the run (aggregated)

Messages that differ only in numbers or variable lists are grouped; one example is shown.

* `Constant`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw)`, 14 origin-messages (14 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0288, 0.0479]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV`, 13 origin-messages (12 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0157, 0.0337]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `SV(rw,t10)`, 32 origin-messages (29 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL', 'UR'] (fractions [0.0502, 0.0689, 0.0165]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: Student-t ν has posterior mass ≥ 0.5 at the grid bound 10 for ['GDP', 'INFL', 'UR', 'TBILL'] (mass [1.0, 1.0, 1.0, 1.0]): near-Gaussian tails, SV may suffice (report only; mass at ν = 10 is normal, N03 §6)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `SV(rw,t10)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
* `AR-SV(0.9)`, 13 origin-messages (13 variants), e.g.: non-centred sign fold in more than 1% of sweeps for ['GDP', 'INFL'] (fractions [0.0163, 0.033]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: this composition is not order-invariant: with 4 variables its results depend on their ordering (R8; N08 §B2); gauge how much with rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)
* `AR-SV(0.9)`, 106 origin-messages (1 variants), e.g.: own-lag prior means contradict the transformation codes for ['UR', 'TBILL'] (levels with mean < 0.5 or differences with mean ≥ 0.5)
