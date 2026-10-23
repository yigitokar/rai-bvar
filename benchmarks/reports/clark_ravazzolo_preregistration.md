# Clark–Ravazzolo replication: pre-registration of the acceptance run

Written 2026-10-23, before any score of the acceptance run was computed. The acceptance run is
`benchmarks/clark_ravazzolo.py benchmarks/reports/clark_ravazzolo_acceptance --config acceptance`
with a fresh cache (`--cache ~/.cache/rai-bvar-cr-<short commit>`; the command is in the script's
docstring), on a clean working tree at the commit that adds this file. The report header records
that commit and whether tracked files differ from it; the report is labelled the acceptance run
only when they do not, every stage ran, and the config, estimator and backtest seed are the ones
below. The previous full run (`--config full`, report `benchmarks/reports/clark_ravazzolo.md`
committed in `551f181`) is the reference this document departs from.

## Why there is a second run

The full run met criteria 1, 2, 3 and 5 and missed 4 and 6 (N08 §C5, as coded in
`benchmarks/clark_ravazzolo.py`). The two misses had diagnosable causes. Neither was a sign
that SV(rw) loses to Constant.

- **Criterion 6** failed only on `compare()`'s convergence gate. The SV(rw) verdict was
  "better" in 24/24 orderings with the gate off and 0/24 with it on, because SV(rw) converged at
  57–82% of origins against the 95% gate.
- **Criterion 4** failed on one cell: full-sample UR, ΔLPS(h=8) = +1.954 > ΔLPS(h=1) = +0.159.
  A handful of 2007–08 origins drive it. There the path-conditioned Rao–Blackwellised estimator
  is degenerate: one forecast draw carries all the weight, and the score is biased low by up to
  tens of nats, more for the thinner-tailed Constant arm.

These changes are motivated by that failure analysis, so the run is not blind to the previous
results. What is pre-registered is that the settings below were fixed before any score of the new
run existed, and that the criteria are evaluated on them as coded, whatever the outcome.

## What changes (and only this)

1. **MCMC budget for every MCMC arm, in every stage:** 4 chains × 20,000 kept draws, burn-in
   1000, thin 1 (previously 4 × 5000, burn 1000). With the default `store_max = 2500` the
   parameters of one in ⌈20,000 / 2500⌉ = 8 kept draws are stored (10,000 in all), and the
   forecast draws are taken from those; R-hat and ESS use every kept draw.
   - Evidence: fit-level convergence diagnostics only, with no forecasts or scores. The data are
     the same and the fit seeds are the CR main-stage ones.
   - SV(rw), identity ordering: 72/106 origins converge at 4 × 5000; 79/106 with burn 5000;
     92/106 with 8 chains; **106/106 at 4 × 20,000** (max R-hat 1.0061, min bulk ESS 1763).
   - The two worst orderings of the previous run (INFL > UR > GDP > TBILL and
     INFL > UR > TBILL > GDP): 71/106 and 63/106 at 4 × 5000 with the main-stage fit seeds;
     **106/106 and 106/106 at 4 × 20,000** (max R-hat 1.0049 and 1.0046, min ESS 1689 and 1717).
     The previous run's ordering stage, which fits each ordering with its own backtest seed, had
     63/106 and 60/106 there (and 69/106 for the identity ordering against 72/106 in the main
     stage), so this evidence is at the main-stage seeds, not at the ordering stage's.
   - All 520 + 424 diagnostic fits finished without error. Every base-budget failure is on a σ
     quantity and is marginal (R-hat < 1.03), consistent with R-hat sampling noise at bulk ESS
     ~600, not with a stuck sampler.
2. **Primary log predictive density: the path-integrated Rao–Blackwellised estimator**
   (`lpd_method="integrated"`; DESIGN, the decision row added with it).
   - For h > 1 it conditions each draw on its parameters and its drawn latent scale path only,
     and integrates the intermediate y's out analytically.
   - At h = 1 it is identical to the previous estimator.
   - The previous (path-conditioned) estimator is reported as a secondary number in criterion
     4's detail, and per cell in the JSON (`dlps_conditioned`, `dlpd_conditioned`); it never
     enters a rule.
3. **Forecast draws per origin: 8000** (previously 1000), in every stage.
   - Reason: the integrated LPD is exact given each draw but is still an average over draws, and
     at crisis origins the weights concentrate on a few of them. In the review of the estimator
     (2008Q2, UR h = 8, at the budget of item 1; precision only, no criterion evaluated),
     Constant's 1000-draw value sat on average about 0.3–0.6 nats below the 80,000-draw value,
     with an sd of 0.75–1.06 nats across thinning offsets. SV(rw)'s varied with sd ≈ 1.4 nats
     across forecast seeds, more than its MCSE, and its expected bias was as large as
     Constant's, so the error favours neither arm.
   - With 8000 draws the thinning error dropped to an sd of 0.09–0.19 nats with bias ≈ 0.
   - It does not remove the MCMC component: the 80,000-draw values still differ by about
     1.3 nats across fit seeds at that origin.
   - The cost is small next to the fits.

## What does not change

- **Data:** FRED-QD/FRED-MD vintage 2026-09 with the same SHA-256 checks, the four series and
  their transformations, and the estimation start 1965Q1.
- **Models:** Constant(iw_equivalent), SV(rw), SV, SV(rw,t10), AR-SV(0.9), and the structural
  pair. The mean prior is `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, alpha=2,
  cross=1, mean=0)` with p = 4.
- **SV priors:** the library defaults. CR's own RW-SV σ² prior IG(1/2, 0.0175) is **not**
  adopted. It stays a documented deviation, and adopting it now would be a modelling change
  decided after seeing results.
- **Origins and horizons:** origins 1985Q1–2011Q2 (106) with the 1985Q1–2007Q4 pre-crisis
  subset (92); expanding window; horizons 1, 2, 4, 8.
- **Seeds:** backtest seed 0, per-origin fit seeds (DESIGN §11.6), cold starts; the ordering
  stage with all 24 orderings and the same noise replications.
- **Rules:** the criteria 1–6 rules and thresholds exactly as coded. That includes the
  95% convergence gate of `compare()` (R-hat < 1.01 and bulk ESS > 400 on every primary
  quantity) and the RB-degenerate flag (joint `lpd_mcse` ≥ 0.9; descriptive only).

## Stated in advance

- **Criterion 4 may still fail on UR.** The WP23 review's estimate with corrected tail
  origins was UR h = 8 ≈ +0.23 against +0.157 at h = 1, which breaks the coded rule
  ΔLPS(h=8) < ΔLPS(h=1). If it fails, it is reported as a difference from CR, attributed with
  evidence where possible (final vintage instead of real-time data, Chan's iw_equivalent
  baseline, library-default SV priors). It is not tuned.
- **Criterion 6** is expected to pass if the convergence gate is met. Its h = 1 scores are
  estimator-independent.
- **The integrated LPD still carries Monte Carlo error at tail origins** (item 3): about
  1.3 nats per arm at a crisis origin from the MCMC itself. That amounts to a few hundredths of a
  nat in a 106-origin mean, but it is not zero. The RB-degenerate flag stays descriptive.
- **No further run with different settings** is made after this one's scores are seen, except
  to fix a bug. A bug fix is documented, and the affected stages are rerun in full with these
  settings and a fresh cache (the score-cache keys do not change with the code).
