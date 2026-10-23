# Dossier 06: The empirical illustration (Clark–Ravazzolo 2015 on FRED-QD)

Prepared 2026-10-23 for the methods/software paper on `rai-bvar` (target style: *Journal of
Statistical Software*). Paper checkout: `/Users/yigitokar/rai-bvar-paper`, branch `paper-draft`,
HEAD `62d15f6` (the merge of PR #4), working tree clean.

> **Every number in this dossier from the full run (`benchmarks/reports/clark_ravazzolo.{md,json}`,
> committed in `551f181`, code `4bb978a`) is PROVISIONAL.** The final numbers must come from the
> pre-registered acceptance run, which has not been run or committed yet (§9). In tables, the
> caption says so. In prose, full-run numbers carry the tag **(P)**.

Status vocabulary (from `docs/paper/NOTES.md`, preamble):

- **verified**: reproducible from a committed test or fixture;
- **recorded**: numbers from a run that is written down but was not re-run independently;
- **to check**: must be re-run or re-derived before it goes into the paper.

Every full-run number is "recorded" *and* PROVISIONAL. Anything that exists only in the uncommitted
working tree (tier T3 below) is "to check" whatever its content.

---

## 0. Provenance: where every statement comes from

### 0.1 Evidence tiers

| Tier | Source | What it holds | Status of its numbers |
| :- | :- | :- | :- |
| T1 | Committed files in the paper checkout at `62d15f6` | `benchmarks/clark_ravazzolo.py` (the script; cited by line), `benchmarks/reports/clark_ravazzolo.{md,json}` (full run), `benchmarks/reports/clark_ravazzolo_quick.{md,json}` (quick run), `docs/phase2/08_common_sv_and_ordering.md` (N08, the replication plan, Part C), `SPEC.md`, `docs/phase2/DESIGN.md`, `docs/paper/NOTES.md` (items 8b and 17), `src/rai_bvar/…`, tests | run numbers: recorded, PROVISIONAL; tests: verified |
| T2 | GitHub issue #3 ("orchestrator queue", `yigitokar/rai-bvar`), 17 comments, read with `gh issue view 3 --repo yigitokar/rai-bvar --comments` and `gh api …/issues/3/comments`; cited by UTC timestamp | run log: hold, crash, diagnosis, fix, rerun, results | recorded |
| T3 | **Uncommitted** files in the *other* checkout `/Users/yigitokar/rai-bvar`, branch `phase-2-cr-acceptance`, working tree as read on 2026-10-23 (`git status`: modified `NOTES.md`, `DESIGN.md`, `engines/mcmc/forecast.py`, `forecast/result.py`, `tests/mcmc/forecast/test_lpd_mcmc.py`; untracked `benchmarks/reports/clark_ravazzolo_preregistration.md`, `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`, `tests/unit/test_lpd_integrated.py`. (corrected by fact-check) This list was incomplete: `git status --short` in that checkout at fact-check time also shows modified `benchmarks/clark_ravazzolo.py`, `src/rai_bvar/evaluation/backtest.py`, `evaluation/compare.py`, `evaluation/scores.py`, `structural/ordering.py`, and untracked `tests/integration/test_backtest_lpd_integrated.py`, `tests/unit/test_scores_lpd_method.py`; the working tree is still changing) | the acceptance-run pre-registration, decision row D-34 (path-integrated LPD), the NOTES 8b addendum | **to check** (not committed; may change before commit) |
| C | Read-only commands I ran in the paper checkout (stated where used), e.g. `uv run --offline python -c '…json.load(open("benchmarks/reports/clark_ravazzolo.json"))…'` | derived quantities (ratios, ranges, counts) from T1 JSON | as the source they are derived from (PROVISIONAL) |

**Discrepancy with the task brief.** The brief says the issue #3 comments contain the convergence
diagnostic (4 × 20,000 draws, 106/106 origins converged) and the plan for the acceptance run with
the path-integrated RB estimator. They do not. The thread has 17 comments and ends at
2026-10-23T10:44:36Z with "PR #4 merged into `phase-1-conjugate` by the user (merge commit
`62d15f6`). The overnight queue is closed." PR #4 has one comment (weekly tier). The diagnostic
numbers and the pre-registration exist **only** in the uncommitted T3 file
`/Users/yigitokar/rai-bvar/benchmarks/reports/clark_ravazzolo_preregistration.md`. This dossier
cites them from there, tagged T3.

### 0.2 Inventory of runs

| Run | Code | Config (`CONFIGS`, script l. 166–173 (corrected by fact-check: was 166–175)) | Machine | Ordering stage | Report | Status |
| :- | :- | :- | :- | :- | :- | :- |
| quick | rai-bvar 0.0.1 (report header) | `quick`: library default `MCMCSettings()`, 4 chains × 2000 kept, burn 1000 | "shared 4-core x86-64 VM (load average about 10)", Linux x86_64, python 3.13.16; main stage 1.08 h (report banner) | started, then stopped on request (report banner) | `clark_ravazzolo_quick.{md,json}` (T1), "NOT the acceptance run" | recorded |
| full #1 | `8a622c5` | `full`: 4 × 5000 kept, burn 1000 | Apple M5 Max (T2, 01:34Z item 0 names the machine; corrected by fact-check, the 03:01Z comment does not), `--n-jobs 6` (T2, 03:01Z) | **aborted** in the SV(rw) ordering stage (T2, 04:10Z) | main-stage report kept only in `/tmp/claude-501/kept` (T2, 05:46Z); not committed | recorded (T2 only) |
| full #2 | `4bb978a` (integration head incl. the RW AWOL fix `f60d736`) | `full` | Apple M5 Max, Darwin arm64, python 3.12.13, jax 0.12.0.1 (report header); `--n-jobs 6`, fresh cache; weekly-tier jobs alongside, load about 25 (T2, 05:17Z) | all 24 orderings, both arms | `clark_ravazzolo.{md,json}`, committed `551f181` | recorded, **PROVISIONAL** |
| acceptance | not yet committed | `--config acceptance` (named in the T3 pre-registration; **absent** from the committed `CONFIGS`, which has only `full`, `quick`, `smoke`. (corrected by fact-check) The uncommitted working-tree script in the other checkout (T3, `git diff benchmarks/clark_ravazzolo.py`) now defines it: `ACCEPTANCE` config with `lpd_method="integrated"`, `noise_reps=1`, and a secondary column keyed `CONDITIONED_MARGINAL = "lpd_marginal_conditioned"`; to check until committed) | — | all 24 orderings | — | pre-registered (T3), not run |

Run time of full #2: 3950 s = 1.10 h (`seconds` in the JSON; report header), from a fresh cache.
Stage times (T2, 05:17Z): main stage 649 s, Constant orderings 982 s, SV(rw) orderings 2308 s.
The script docstring's *estimate* (l. 8–13) was "about 12 hours with `--n-jobs 1`" on the 4-core
x86 VM and "roughly 4–6 hours" with `--n-jobs 4` on 10 cores; the docstring itself labels that an
estimate, not a measurement. Fit counts (docstring l. 10–11): 742 main-stage fits and
2 × 25 × 106 = 5300 ordering-stage fits.

---

## 1. Role of the illustration in the paper

1. **It is a Phase 2 exit criterion.** SPEC §10.7 (l. 552): "Phase 2: a Clark & Ravazzolo
   (2015)-style ranking: SV beats homoskedastic models on density forecasts (FRED-QD)." SPEC §13,
   Phase 2 exit list; DESIGN exit-criteria map ("Clark–Ravazzolo-style replication | WP23") and
   WP23 acceptance: "the replication report meets N08 §C5 criteria 1–6 (including all 24
   orderings) or documents each failure". DESIGN §9.10 puts it in the pre-release CI tier.
   Note the wording: a *qualitative ranking* replication, not a number-for-number one. The paper
   should say so up front.
2. **It exercises the whole user-facing stack on real data**, which is what a JSS reader wants to
   see:
   - pinned, checksummed data download (SPEC §5.8, decision D4; `rai_bvar.datasets.load`);
   - the composition API `BVAR(mean=Linear(prior=Minnesota(...)), vol=SV(...))` (SPEC R1);
   - the backtest with per-origin fit seeds, disk cache and spawn-started worker pool (DESIGN §11.6;
     SPEC §9.1);
   - the score set (SPEC §9.2) and the `compare()` verdict rule (SPEC §9.5);
   - runtime convergence diagnostics (SPEC §10.6; DESIGN D-33);
   - `ordering_sensitivity` (SPEC §6.6; N08 §B4), the operational answer to R8.
3. **It produced two of the paper's methods findings**, which the illustration section can point
   forward to:
   - the tail degeneracy of the path-conditioned RB log score at h > 1 (NOTES 8b) and the
     path-integrated estimator that replaces it as primary in the acceptance run (D-34, T3);
   - the float64 failure of the scaled RW AWOL draw at a legitimate tiny-σ state (NOTES 17,
     verified).

   It also shows the interplay between a conservative convergence gate and a verdict engine
   (criterion 6, §7.1).

Suggested one-sentence pitch for the section: the package reproduces the qualitative
Clark–Ravazzolo ranking (SV beats a constant-volatility VAR in density forecasts at short
horizons), and it reports, rather than hides, where and why the replication departs from the
published numbers. The pitch must be checked against the acceptance run.

---

## 2. The target

- **Publication.** Clark & Ravazzolo (2015), *J. Applied Econometrics* 30(4), 551–575, DOI
  10.1002/jae.2379 [CITE?]. N08 §C1 records the bibliographic data as Crossref-confirmed and the
  abstract as read via OpenAlex. The JAE full text was **not** accessible: the JAE-specific
  numbers are UNVERIFIED (N08 §C1, §D item 1).
- **The numbers used are the working paper's.** Clark & Ravazzolo (2012), Cleveland Fed WP 12-18 /
  Norges Bank WP 2012/09, "The Macroeconomic Forecasting Performance of Autoregressive Models with
  Alternative Specifications of Time-Varying Volatility" [CITE?]. Its Tables 1–3 (US, VAR block)
  are hard-coded as `CR_FULL` / `CR_PRE_DLPS` (script l. 128–153) and listed in N08 §C5. N08's
  cross-check log says every Table 1–3 number quoted was re-read (all 40 cells). (corrected by
  fact-check: the "40" is N08's own count, but N08 §C5 quotes 56 Table 1–3 numbers, 48 full-sample
  plus 8 pre-crisis, as does the script's `CR_FULL`/`CR_PRE_DLPS`; the count mismatch should be
  resolved before the paper says "every number re-read".) **The paper must
  cite the WP for the numbers**, or obtain the JAE tables and re-check.
- **What CR estimated** (N08 §C3, from the WP):
  - a VAR(4) with a Minnesota prior without cross-variable shrinkage (θ = 0.2, 1/l² decay, zero
    means) and a normal-diffuse `Φ` (Kadiyala & Karlsson 1997 [CITE?]), 30,000 draws;
  - VAR-SV with RW log-volatilities, `φ_i ~ IG(1/2, 0.0175)` (Clark's "mean 0.035, 1 d.o.f.";
    no finite mean, N08 §C3) and a training-sample prior for `log λ_0`, sampled by the JPR (1994)
    single-move Metropolis step [CITE?];
  - VAR-ARSV with a nearly fixed `a₁ ∈ {0.9, 0.8, 0.5}`;
  - VAR-SVt with fixed d ∈ {5, 10, 15};
  - GARCH, mixture and TVP-SV variants, which are out of scope in Phase 2.
- **CR's model ordering is not stated** in the WP (N08 §C2, §D item 2). The replication uses the
  table order GDP, INFL, UR, TBILL and runs all 24 orderings.
- **CR's headline (WP Table 2, full sample, VAR-SV − VAR ΔLPS at h = 1):** GDP +0.193, INFL
  +0.075, UR +0.162, TBILL +0.284; the long-horizon UR/TBILL losses are UR −0.182 (h = 4),
  −0.707 (h = 8) and TBILL −0.304, −0.403 (N08 §C5). The marginal h = 1 gains sum to 0.714 (my
  arithmetic). N08 §B3 notes that CCM (2016) report +0.967 for a *joint* score on a different
  sample and variable set [CITE?] (corrected by fact-check: the +0.967 and the "different variable
  set" caveat are in N08 §B3; §C5 criterion 1 rounds it to "+0.97 joint for 1985–2013").

---

## 3. Data

| Model variable | CR definition (N08 §C2) | FRED-QD 2026-09 mnemonic | Transformation in the script | FRED-QD default tcode → override (N08 §C2) |
| :- | :- | :- | :- | :- |
| GDP | 100·Δlog real GDP/GNP, RTDSM real-time | `GDPC1` | `100·log`, then difference (tcode 2 on the log level; `load_data`, l. 179–209) | 5 → 5 |
| INFL | 100·Δlog GDP price index/deflator, RTDSM | `GDPCTPI` (chain-type; FRED-QD has no `GDPDEF`) | same as GDP | **6 → 5** |
| UR | level, quarterly average of monthly | `UNRATE` | level | **2 → 1** |
| TBILL | level, quarterly average of monthly | `TB3MS` | level | **2 → 1** |

- **Scale.** Non-annualised quarterly rates (script docstring l. 33–35). N08 §C2 gives three
  pieces of evidence that CR's numbers are non-annualised despite the WP's table notes. LPS
  differences and CRPS/RMSE ratios are invariant to rescaling a variable (N08 §C2).
- **Sample.** Estimation starts 1965Q1 (`ESTIMATION_START`, l. 121; asserted in `load_data`),
  because FRED-QD starts in 1959 (CR: 1955Q1). The usable data are T = 246, 1965Q1–2026Q2 (T2,
  03:01Z).
- **Asserted at run time (verified per run; numbers PROVISIONAL):**
  1. FRED-QD `UNRATE` and `TB3MS` are within-quarter means of FRED-MD monthly values of the same
     vintage (`assert_quarterly_averages`, l. 212–236; tolerance 0.01, > 200 quarters). Full
     run: max |diff| 3.3e-5 for both, over 270 quarters, 1959Q1–2026Q2 (`data_checks.aggregation`
     in the JSON; the MD rounds this to 0.0000). N08 §C6 adds a one-off cross-check against CR's
     own archive: `TB3MS` matches `ITB3M` to ≤ 0.0034 over 211 quarters; `UNRATE` matches with mean
     absolute difference 0.004, max 0.13.
  2. No look-ahead (SPEC R9): at origins 1985Q1, 2007Q4 and 2011Q2 the resolved priors,
     hyperparameters and padded data are bitwise identical when the vintage is cut at the origin
     (`assert_in_window`, l. 239–262; `data_checks.in_window.identical = true`). The check is run
     **for the primary SV(rw) model only** (`main`, l. 1230:
     `assert_in_window(mods[PRIMARY_SV], …)`), not for every arm.
  3. The SHA-256 checksum of the pinned vintage files is verified (SPEC §5.8; script docstring
     l. 27–29).
- **Deviations from CR** (`DEVIATIONS`, l. 1134–1156 (corrected by fact-check: was 1134–1159);
  printed in the report; the list below is a selection: `DEVIATIONS` has 9 entries, the others
  being the `iw_equivalent` baseline, the AR-SV prior mapping, the LPD estimator, descriptive
  significance and the convergence gate):
  1. pseudo real time on the final 2026-09 vintage, instead of RTDSM real-time vintages and
     second-release actuals (D4; N08 §C2);
  2. the 1965Q1 start;
  3. `GDPCTPI` instead of the deflator;
  4. no last-vintage training-sample prior for `h₀`, which would violate R9 (N08 §C6);
  5. own-lag prior means 0 for the UR and TBILL *levels*, as in CR. This is a departure from the
     library default; the library warns at every origin of every arm (warnings section of the
     report).

---

## 4. Models

Mean block, common to all arms: `Linear(prior=Minnesota(form="reduced", hierarchy="fixed",
lambda_=0.2, alpha=2.0, cross=1.0, mean=0.0))`, p = 4 (`minnesota`, l. 268–270). The reduced form
uses the corrected CCCM (2022) triangular sampler (DESIGN D-11; Carriero, Chan, Clark & Marcellino
2022 [CITE?]). `hierarchy="fixed"` keeps the Minnesota hyperparameters fixed (D-12). Every arm runs
on the MCMC engine: `BVAR.engine` is `"analytic"` only for `form="conjugate"` (`src/rai_bvar/api.py`
l. 189–191).

| Arm (report label) | Library spec (`models`, l. 273–285) | CR counterpart | Priors (DESIGN D-13 unless stated) | Role |
| :- | :- | :- | :- | :- |
| `Constant` | `Constant("iw_equivalent")` | VAR (normal-diffuse `Φ`) | Chan (2022) Prop. 1 `(a, λ)` prior, only `a_i` λ-scaled (D-30, E10) [CITE?] | baseline of every relative measure |
| `SV(rw)` | `SV(rw=True)` | VAR-SV | RW: `h₀ ~ N(log(ψ_i/ι), 4)`, `σ² ~ 0.1·χ²₁` (data-centred, per window); CR's `IG(1/2, 0.0175)` **not** used | **primary** SV arm of criteria 1–4 and 6 |
| `SV` | `SV()` | (VAR-ARSV-like, library default) | `μ_i ~ N(log(ψ_i/ι), 10)`, `(φ+1)/2 ~ Beta(20, 1.5)`, `σ² ~ 1·χ²₁`, stationary `h₀` | second reference of criterion 5 |
| `SV(rw,t10)` | `SV(rw=True, tails="t", nu_prior=DiscreteUniform(10, 10), t_scale="unit")` | VAR-SVt, d = 10 | ν fixed at 10; unit-scale t for parity (D-8) | criterion 5 variant |
| `AR-SV(0.9)` | `SV(phi_prior=Beta(71.25, 3.75))` | VAR-ARSV, a₁ = 0.9 | CR's `a₁ ~ N(0.9, 0.05²)` mapped to `(φ+1)/2 ~ Beta(71.25, 3.75)`, same mean and sd on φ (my arithmetic check: mean 71.25/75 = 0.95 → φ̄ = 0.9; sd √(71.25·3.75/(75²·76)) = 0.025 → 0.05 on φ); `μ` keeps the library's data-centred prior instead of CR's `a₀ ~ N(0, 0.4²)` | criterion 5 variant |
| `Constant [structural]` / `SV(rw) [structural]` | `Minnesota(form="structural")` × `Constant("ig")` / `SV(rw=True)` (`structural_models`, l. 288–292) | — | structural Mode I prior (D-10) | robustness pair (N08 §C3 R2) |

"No prior was tuned" (docstring l. 51–52 (corrected by fact-check: was l. 47); DEVIATIONS item 5). CR's GARCH, mixture and TVP models have
no Phase 2 counterpart (N08 §C3).

**Gap against DESIGN.** Risk R6 (DESIGN §12.1) promises that "Clark–Ravazzolo report includes Mode
I vs II and data-centred vs stochvol `μ` priors". Neither comparison is in the committed report or
in the pre-registration. See the open questions.

---

## 5. Evaluation design

- **Origins and horizons.** Recursive expanding window; origins 1985Q1–2011Q2 (106, full) and the
  1985Q1–2007Q4 subset (92, pre-crisis); h = 1, 2, 4, 8 quarterly rates, h = 1 being the first
  quarter after the last observation (docstring l. 41–44; `n_origins` in the JSON).
- **MCMC.** `full`: 4 chains × 5000 kept draws, burn 1000, thin 1. The JSON `config.mcmc` also
  records `store="params"`, `store_max=2500`, `path_thin=10`, `t_bucket=32`, `rhat_max=1.01` and
  `ess_min=400.0`. 1000 forecast draws per origin, backtest seed 0, cold starts (warm starts are
  opt-in, D-32).
- **Seeds and cache.** Per-origin fit seed `int(default_rng([seed, crc32(origin),
  0x4D434D43]).integers(2**31))` (DESIGN §11.6; `fit_seed_example` in the JSON: 1985Q1 →
  1958591862). Scores are cached per origin and keyed by spec, data, seed and the
  `"seeding": "fit_seed-v1"` tag, so interrupted runs resume (DESIGN §2.10, WP23-review amendment (3), l. 616;
  test `test_mcmc_cache_keys_record_the_fit_seeding` in `tests/integration/test_backtest_mcmc.py`).
  The `--n-jobs` worker count does not change results (docstring l. 26–27 (corrected by
  fact-check: was l. 23–25); test
  `test_spawn_pool_matches_the_serial_backtest`).
- **Scores per origin, horizon and variable** (SPEC §9.2):
  1. **LPD, primary:** the Rao–Blackwellised estimator of SPEC §7,
     `log p̂(y_{T+h}|Y) = logsumexp_s log N(y; m_{s,h}, L_{s,h}L_{s,h}') − log S`, joint and
     marginal. In the full run, `m_{s,h}` conditions on the simulated path to T+h−1
     ("path-conditioned"; SPEC §7 `cond_mean`). Its delta-method MCSE is ESS-adjusted for MCMC
     draws (`log_pred_density` docstring, `src/rai_bvar/forecast/result.py` l. 209–237).
  2. **Gaussian-approximation LPS** (CR's score; Adolfson, Lindé & Villani 2007 [CITE?]), from the
     same forecast draws' mean and covariance (`MomentsProbe` and `_ProbeFit`, l. 305–373;
     `gaussian_lps`, l. 376–398 (corrected by fact-check: were l. 305–368 and 376–401)). It equals −½ the Dawid–Sebastiani score, which is proper but not strictly proper
     (N08 §C4; Krüger et al. 2021 [CITE?]; Gneiting & Raftery 2007 [CITE?]). It is a parity
     metric only.
  3. **CRPS** from draws: `scoringrules.crps_ensemble` with the probability-weighted-moment
     estimator, which equals the fair (unbiased) estimator and is O(S log S) by sorting
     (`src/rai_bvar/evaluation/scores.py` l. 18 and l. 100–104; SPEC §9.2 "Sample-based fair
     estimator"). (corrected by fact-check: the dossier said "energy form"; the energy form
     `E|Y − y| − ½E|Y − Y'|` is CR's definition, N08 §C4, not the library's estimator.)
  4. **Squared error** of the predictive mean, for RMSE.
- **Relative measures** (`relative_table`, l. 415–490), always against `Constant` over common
  origins:
  - ΔLPS is the mean of the per-origin differences; the median and the 3 largest origins are
    stored too;
  - the CRPS ratio is a ratio of means;
  - the RMSE ratio is a ratio of root-mean-squares;
  - the one-sided DM/HLN p-value is for "the model is better" in log score (Diebold & Mariano
    1995; Harvey, Leybourne & Newbold 1997 [CITE?]).

  Under an expanding window these p-values are **descriptive** (SPEC §9.3; Clark & McCracken
  2001; Giacomini & White 2006 [CITE?]).
- **`compare()` verdict** (SPEC §9.5; `CompareResult.verdict`, `src/rai_bvar/evaluation/compare.py`
  l. 196–276). The verdict is "better" iff all of the following hold:
  - total ΔLPD > 0;
  - |total ΔLPD| > 3 × its MCSE;
  - either a *valid* test rejects at 10% (rolling windows only), or the model is in and the
    reference out of the 25% MCS (Hansen, Lunde & Nason 2011 [CITE?]).

  The verdict is then forced to "inconclusive" if fewer than 95% of origins converged
  (`min_converged=0.95`). The script reports the verdict as returned (gate on) and with
  `min_converged=0` (gate off), labelled (`verdicts`, l. 493–511).
- **Convergence** of a fit: R-hat < 1.01 and bulk ESS > 400 on every quantity of the *primary*
  monitor set: log-likelihood, own first lag of B, SV parameters, `h_T`, `log λ` (DESIGN D-33;
  SPEC §10.6; Vehtari et al. 2021 [CITE?]).
- **RB-degenerate flag** (`RB_CEILING = 0.9`, l. 298; `_degenerate`, l. 408–413). An origin is
  flagged at horizon h when the *joint* `lpd_mcse` of the arm or of the baseline is ≥ 0.9. It is
  descriptive only: flagged origins are kept, because dropping exactly the tail origins would bias
  the comparison (`RELIABILITY_NOTE`, l. 865–882). The count shown in a *marginal* cell is the
  joint-score count for that horizon, so it is the same for all four variables.
- **Ordering stage** (`ordering_stage`, l. 755–841 (corrected by fact-check: was 755–844)):
  - `rb.ordering_sensitivity(mode="backtest", design="all", n_perm=24, noise_reps=1)` for SV(rw)
    and Constant over all 106 origins;
  - then, per ordering, both backtests are rebuilt from the cache and compared with `compare()`.

  The noise floor is `σ_MC = max(sd over the user fit and its replicates, MCSE)`. With
  `noise_reps = 1` that sd comes from two values. (corrected by fact-check) In `mode="backtest"`
  there is no MCSE term: "Backtest scores carry no MCSE" (`ordering.py` module docstring l. 32), and
  the backtest dataset stores `mcse` as all-NaN (`nan_mcse`, `ordering.py` l. 1527 and 1534), so the
  floor here is the two-value across-seed sd alone. A metric is flagged when it exceeds its N08 §B4
  threshold (`lpd_range` > 0.05 nats per variable, > 0.10 joint) **and** its signal-to-noise ratio
  is > 3 (`src/rai_bvar/structural/ordering.py` module docstring, l. 25–50).

---

## 6. The six acceptance criteria

N08 §C5 states the criteria qualitatively ("proposed"). The script turns them into operational
rules (`criteria`, l. 581–749 (corrected by fact-check: was 581–752)). **The operationalisations are the project's own choices**, e.g.
"about 1" became ≤ 1.02 and "systematically" became ≥ 3 of 4 joint horizons or ≥ 8 of 16 cells. A
referee may question them, so the paper should print them verbatim.

| # | N08 §C5 (qualitative) | Operational rule as coded | Uses h ≥ 2 LPD? | Uses the convergence gate? |
| :- | :- | :- | :- | :- |
| 1 | SV beats Constant in average LPD at h = 1, 2, all four variables, both samples; joint h = 1 gain clearly positive | ΔLPS(SV(rw) − Constant) > 0 in all 16 cells (2 samples × 4 vars × h ∈ {1, 2}), and joint ΔLPD(h = 1) > 0 with one-sided DM/HLN p < 0.10 in both samples (l. 586–614) | yes (h = 2) | no |
| 2 | CRPS ratio < 1 at h ≤ 2, all four variables | CRPS ratio SV(rw)/Constant < 1 at h = 1, 2, all four variables, full sample; pre-crisis reported only (l. 616–634) | no | no |
| 3 | RMSE ratios ≤ about 1 at short horizons (CR: GDP and bill gains 5–10%) | RMSE ratio ≤ 1.02 at h = 1, 2, all four variables, full sample (l. 636–648) | no | no |
| 4 | Full sample: h = 8 LPD gain shrinks or turns negative for UR and the bill; pre-crisis it stays positive for GDP and inflation | full: ΔLPS(h = 8) < ΔLPS(h = 1) for UR and TBILL; pre-crisis: ΔLPS(h = 8) > 0 for GDP and INFL (l. 650–671) | **yes (h = 8)** | no |
| 5 | Neither SV-t nor AR-SV systematically beats RW/AR SV | neither SV(rw,t10) nor AR-SV(0.9) is better than *both* SV(rw) and SV (one-sided DM p < 0.10 and positive ΔLPD) at ≥ 3 of 4 joint horizons or ≥ 8 of 16 marginal cells, full sample (l. 673–708) | yes | no |
| 6 | `compare()` verdict "better" (DM one-sided 10% for at least GDP and the bill at h = 1) in all 24 orderings, or the report states which fail | in every ordering: `compare()` SV(rw) vs Constant at h = 1 is "better" with the gate on **and** off, and the DM p of the marginal log score < 0.10 for GDP and TBILL at h = 1 (l. 710–749 (corrected by fact-check: was 710–752); `met = not ungated and not dm_fail and not gated`, l. 722) | no (h = 1 only) | **yes** |

---

## 7. Results as they stand (full run #2, all PROVISIONAL)

### 7.1 Criteria

*Table 7.1. Source: `criteria` in `benchmarks/reports/clark_ravazzolo.json`; report §"Acceptance
criteria". All numbers PROVISIONAL.*

| # | Met | Numbers (P) |
| :- | :- | :- |
| 1 | met | 16/16 cells positive; joint h = 1: full +0.809 nats/quarter (DM p = 0.000), pre-crisis +0.790 (p = 0.000) |
| 2 | met | 8/8 full-sample cells < 1 (pre-crisis also 8/8); largest h ≤ 2 ratio UR h = 1: 0.936 |
| 3 | met | GDP 0.909/0.899, INFL 0.898/0.850, UR 0.970/0.948, TBILL 0.937/0.950 (h = 1/2) |
| 4 | **not met** | full UR h1 +0.159 → h8 **+1.954** (rule needs h8 < h1); full TBILL h1 +0.378 → h8 −0.207 (ok); pre-crisis GDP h8 +0.108, INFL h8 +0.295 (ok) |
| 5 | met | both variants: 0/4 joint horizons, 0/16 marginal cells significantly better than both SV(rw) and SV |
| 6 | **not met** | verdict "better" in 0/24 orderings with the gate on, 24/24 with it off; DM GDP and TBILL < 0.10 in 24/24 |

Diagnosis of the two misses (T1 report and T2 05:17Z; they are the motivation of the pre-registered
rerun, §9):

- **Criterion 6 fails on the convergence gate alone.** Every ordering is "better" once the gate is
  off. SV(rw) converged at 57–82% of origins across orderings (P), against the 95% gate.
- **Criterion 4 fails on one cell, full-sample UR h = 8**, which the RB-degeneracy section
  explains (§7.6): median per-origin difference +0.099 (P); the three largest origins are 2008Q2
  +41.6, 2007Q4 +41.4 and 2008Q3 +40.4 nats (P); 12 RB-degenerate origins (P); the Gaussian
  approximation reverses the sign (h1 +0.125 → h8 −0.499) (P).

### 7.2 Headline: SV(rw) vs Constant, full sample, with CR's numbers

*Table 7.2. Source: `relative.full["SV(rw)"].variables[v][h]` and `.joint[h]` in the full-run JSON;
CR columns from `CR_FULL` (script l. 128–147) = CR WP Tables 1–3. "Degen." = RB-degenerate origins
(joint MCSE ≥ 0.9, arm or baseline). Our numbers PROVISIONAL.*

| var | h | ΔLPS (RB, path-cond.) | ΔLPS (Gauss) | degen. | DM p | CRPS ratio | RMSE ratio | CR ΔLPS | CR CRPS | CR RMSE |
| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| GDP | 1 | +0.126 | +0.107 | 0 | 0.003 | 0.897 | 0.909 | +0.193 | 0.864 | 0.896 |
| GDP | 2 | +0.124 | +0.054 | 1 | 0.002 | 0.893 | 0.899 | +0.133 | 0.856 | 0.878 |
| GDP | 4 | +0.073 | −0.063 | 7 | 0.141 | 0.928 | 0.945 | +0.090 | 0.890 | 0.932 |
| GDP | 8 | +0.111 | +0.010 | 12 | 0.006 | 0.938 | 0.977 | +0.060 | 0.927 | 1.007 |
| INFL | 1 | +0.231 | +0.213 | 0 | 0.000 | 0.844 | 0.898 | +0.075 | 0.954 | 0.966 |
| INFL | 2 | +0.267 | +0.265 | 1 | 0.000 | 0.808 | 0.850 | +0.083 | 0.944 | 0.949 |
| INFL | 4 | +0.249 | +0.250 | 7 | 0.000 | 0.769 | 0.791 | +0.155 | 0.895 | 0.915 |
| INFL | 8 | +0.282 | +0.319 | 12 | 0.000 | 0.724 | 0.728 | +0.138 | 0.887 | 0.883 |
| UR | 1 | +0.159 | +0.125 | 0 | 0.008 | 0.936 | 0.970 | +0.162 | 0.922 | 0.967 |
| UR | 2 | +0.266 | +0.043 | 1 | 0.090 | 0.925 | 0.948 | +0.098 | 0.897 | 0.930 |
| UR | 4 | +0.969 | −0.256 | 7 | 0.153 | 0.929 | 0.937 | −0.182 | 0.906 | 0.914 |
| UR | 8 | +1.954 | −0.499 | 12 | 0.178 | 0.978 | 0.963 | −0.707 | 1.001 | 0.971 |
| TBILL | 1 | +0.378 | +0.322 | 0 | 0.000 | 0.799 | 0.937 | +0.284 | 0.822 | 0.933 |
| TBILL | 2 | +0.153 | +0.094 | 1 | 0.073 | 0.899 | 0.950 | +0.043 | 0.915 | 0.944 |
| TBILL | 4 | −0.081 | −0.143 | 7 | 0.713 | 0.988 | 0.971 | −0.304 | 1.010 | 0.974 |
| TBILL | 8 | −0.207 | −0.234 | 12 | 0.864 | 1.022 | 0.988 | −0.403 | 1.059 | 0.990 |
| joint | 1 | +0.809 | +0.665 | 0 | 0.000 | | | | | |
| joint | 2 | +0.858 | +0.452 | 1 | 0.017 | | | | | |
| joint | 4 | +2.254 | −0.032 | 7 | 0.146 | | | | | |
| joint | 8 | +5.013 | −0.320 | 12 | 0.162 | | | | | |

Reading (P; interpretation to check after the acceptance run):

- **Agreement with CR at h ≤ 2.** Every marginal ΔLPS is positive, every CRPS and RMSE ratio is
  below 1, and the signs match CR in all 8 short-horizon cells. The h = 1 sum of marginal gains is
  0.894 (P; my arithmetic, `C`) against CR's 0.714. The joint h = 1 gain of +0.809 (P) is of the
  order of CCM (2016)'s +0.967 joint (N08 §B3), which is a different sample and variable set.
- **Inflation gains are 1.6–3.2× CR's across horizons** (3.1× at h = 1: +0.231 vs +0.075; CRPS
  ratio 0.844 vs 0.954).
  Candidate reasons, none established:
  - final-vintage chain-type `GDPCTPI` vs CR's real-time deflator with second-release actuals;
  - the `iw_equivalent` baseline vs CR's normal-diffuse VAR.

  The baseline level comparison (§7.9) shows our constant-volatility inflation forecasts are
  much more accurate than CR's (RMSE 0.188 vs 0.269) and score higher (+0.138 vs −0.235).
  (corrected by fact-check: the dossier said "much sharper"; sharpness, i.e. predictive spread, is
  not reported in the JSON, only RMSE and the log-score level.)
- **The long-horizon UR/TBILL pattern.** TBILL turns negative at h ≥ 4 as in CR. UR does not under
  the RB score (+0.969, +1.954), but it does under the Gaussian approximation (−0.256, −0.499),
  CR's own score. The RB cells at h ≥ 4 carry 7–12 degenerate origins each, so their magnitudes
  are not reliable (`RELIABILITY_NOTE`). Whether the RB/Gaussian split survives the integrated
  estimator is **the** open empirical question of the acceptance run. A candidate explanation is
  N08 §C4's prediction that the exact density "rewards SV's scale-mixture tails, so expect larger
  SV gains than CR's Gaussian-approximated LPS, especially in 2008–09". It is a hypothesis, not a
  finding.
- The RB and Gaussian ΔLPS differ in sign in **28 cells**, across all arms and both samples, all at
  h ≥ 2 (report list; count recomputed, `C`). These are marginal cells; 8 joint ΔLPD cells
  (SV(rw) and SV(rw,t10), h = 4 and 8, both samples) also disagree (`rb_gauss_disagree` in the
  JSON; added by fact-check). N08 §C6 expected the rankings to agree.

### 7.3 Pre-crisis (1985Q1–2007Q4), SV(rw) vs Constant

*Table 7.3. Source: `relative.pre_crisis["SV(rw)"]`; CR from `CR_PRE_DLPS` (l. 148–153).
PROVISIONAL.*

| var | ΔLPS h1 | ΔLPS h8 | Gauss h8 | degen. h8 | CR h1 | CR h8 |
| :- | -: | -: | -: | -: | -: | -: |
| GDP | +0.115 | +0.108 | −0.004 | 8 | +0.222 | +0.214 |
| INFL | +0.259 | +0.295 | +0.345 | 8 | +0.077 | +0.152 |
| UR | +0.147 | +1.032 | −0.654 | 8 | +0.187 | −0.065 |
| TBILL | +0.425 | −0.220 | −0.253 | 8 | +0.316 | −0.261 |
| joint | +0.790 | +1.883 | −0.514 | 8 | | |

The pre-crisis GDP h = 8 cell, which criterion 4 needs to stay > 0, is positive under RB (+0.108)
but −0.004 under the Gaussian approximation, with 8 degenerate origins (P). It is the second cell
at risk in the acceptance run (§9.4).

### 7.4 All SV arms against Constant: joint ΔLPD, convergence and verdicts

*Table 7.4. Source: `relative.full[arm].joint[h].dlpd`, `converged.full[arm]`, `verdicts.full`.
PROVISIONAL.*

| arm | converged (share of 106 origins) | ΔLPD h1 | h2 | h4 | h8 | verdict h1, gate on / off |
| :- | -: | -: | -: | -: | -: | :- |
| SV(rw) | 68% | +0.809 | +0.858 | +2.254 | +5.013 | inconclusive / better |
| SV | 37% | +0.822 | +0.987 | +2.607 | +5.146 | inconclusive / better |
| SV(rw,t10) | 57% | +0.811 | +0.866 | +2.023 | +4.636 | inconclusive / better |
| AR-SV(0.9) | 62% | +0.825 | +0.957 | +2.508 | +5.326 | inconclusive / better |
| Constant | 100% | — | — | — | — | — |

- With the gate on, every verdict at every horizon and in both samples is "inconclusive", because
  no SV arm reaches 95% convergence.
- With the gate off, the full-sample h = 1 verdict is "better" for all four arms: total ΔLPD
  +85.7 to +87.5 nats, MC s.e. 1.00–1.06; each arm is in the 25% MCS and the baseline is not.
- Counted from the warnings section, the number of non-converged origins per arm (SV(rw) 34, SV 67,
  SV(rw,t10) 46, AR-SV(0.9) 40, out of 106) matches the shares above.
- The misses are marginal: typically 1 of 13–21 primary quantities, an SV σ with R-hat
  1.012–1.020 and ESS 540–740 (T2 05:17Z; report warnings, e.g. "sigma[UR] R-hat 1.015, ESS 618").
  (corrected by fact-check: "540–740" is T2's rounded wording; the four examples in the report's
  warnings section have ESS 537, 550, 618 and 743, and R-hat 1.012–1.020.)
- The quick run (4 × 2000) converged at 0% of origins for every SV arm
  (`converged` in `clark_ravazzolo_quick.json`).

### 7.5 Variants (criterion 5) and the structural pair

*Table 7.5a. Source: `variants[variant][reference].joint[h]`, full sample. PROVISIONAL.*

| variant | vs | ΔLPD h1 (p) | h2 (p) | h4 (p) | h8 (p) |
| :- | :- | :- | :- | :- | :- |
| SV(rw,t10) | SV(rw) | +0.002 (0.446) | +0.008 (0.406) | −0.231 (0.810) | −0.377 (0.790) |
| SV(rw,t10) | SV | −0.011 (0.625) | −0.120 (0.977) | −0.584 (0.953) | −0.510 (0.986) |
| AR-SV(0.9) | SV(rw) | +0.017 (0.319) | +0.099 (0.095) | +0.254 (0.009) | +0.313 (0.028) |
| AR-SV(0.9) | SV | +0.004 (0.281) | −0.030 (0.845) | −0.099 (0.876) | +0.180 (0.261) |

- AR-SV(0.9) beats SV(rw) at h = 2, 4, 8 at the 10% level (P). It does not beat SV, so it is not
  "better than both" and criterion 5 holds. This echoes CR's finding that VAR-ARSV(0.9) helps
  long-horizon UR and the bill (N08 §C5: −0.120 and −0.109 vs VAR-SV's −0.707 and −0.403).
- The quick run had AR-SV(0.9) better than both at 1/4 joint horizons (quick report, criterion 5).
- **The t variant is uninformative about estimated tails.** ν is fixed at 10 by
  `DiscreteUniform(10, 10)` (CR's fixed d = 10), so the warning "ν has posterior mass ≥ 0.5 at the
  grid bound 10 … mass [1.0, 1.0, 1.0, 1.0]" fires at all 106 origins *by construction*. Criterion
  5 tests CR's fixed-d variant, not the library's estimated-ν default (D-8).

*Table 7.5b. Structural-form pair `SV(rw) [structural]` vs `Constant [structural]`, full sample.
Source: `structural`. PROVISIONAL.*

| | GDP | INFL | UR | TBILL | joint |
| :- | -: | -: | -: | -: | -: |
| h = 1 ΔLPS | +0.103 | +0.232 | +0.128 | +0.315 | +0.751 |
| h = 2 | +0.107 | +0.260 | +0.197 | +0.060 | +0.589 |
| h = 8 | +0.109 | +0.282 | +1.754 | −0.239 | +2.913 |

The short-horizon ranking is unchanged in structural form, as N08 §C3 R2 expected (P).

### 7.6 Reliability of the RB log score (the methods finding)

- **Degenerate origins**, SV(rw) vs Constant (`rb_degenerate_origins`, `C`; P):

  | h | full sample | pre-crisis |
  | :- | :- | :- |
  | 1 | none | none |
  | 2 | 2008Q4 | none |
  | 4 | 2000Q4, 2007Q4, 2008Q1–2009Q1 (7) | 2000Q4, 2007Q4 (2) |
  | 8 | 1990Q3, 2000Q2, 2000Q4, 2001Q2, 2007Q1–2008Q4 (12) | 1990Q3, 2000Q2, 2000Q4, 2001Q2, 2007Q1–2007Q4 (8) |

  Their h = 8 targets (origin = last observation, docstring l. 42–43) are 2007Q1–2008Q4 + 8
  quarters = 2009Q1–2010Q4; 1990Q3 + 8 = 1992Q3; 2000Q2–2001Q2 + 8 = 2002Q2–2003Q2. (corrected by
  fact-check: the reading that these targets fall "in or around a recession" / at "the
  unemployment peak" has no repository source; moved to "Unsupported claims".)
- **Mechanism** (NOTES 8b; `RELIABILITY_NOTE`; DESIGN §2.10, WP23-review amendment (4), "known limitation", l. 616):
  - for h > 1 the SPEC §7 estimator averages `p(y_{T+h} | θ_s, path_s up to T+h−1)`;
  - for an outcome far in the predictive's tail, one draw whose simulated path wandered towards it
    carries all the weight;
  - the estimate is biased low by up to tens of nats, while the delta-method MCSE saturates
    near 1.
- **Numbers (recorded, from the WP23 review of the quick run, not re-run):** Constant's UR
  h = 8 score at 2008Q2 was −93.6 with 1000 draws, −46 to −63 with 8000 and −36 with 64,000,
  against −19.1 for the exact path-integrated mixture (NOTES 8b; `RELIABILITY_NOTE`); SV(rw)'s was
  −23.9 against about −12.7 (`RELIABILITY_NOTE` only). The review estimated that with the
  degenerate origins replaced by converged values, the quick run's full-sample UR h = 8 ΔLPS is
  about +0.23 (NOTES 8b; `RELIABILITY_NOTE`), against +0.157 at h = 1 (quick-run value;
  `RELIABILITY_NOTE` only). The quick-run UR h = 8 ΔLPS (+1.92) was inflated about 8× (NOTES 8b).
  (corrected by fact-check: the dossier attributed all of these to NOTES 8b; the SV(rw) pair and
  the +0.157 are only in `RELIABILITY_NOTE`, script l. 865–882 and the report, and the review
  numbers are for the quick run.)
- **Why it inflates SV's gain:** the bias is larger for the thinner-tailed Constant arm
  (pre-registration, T3, "Why there is a second run").
- **The degeneracy is visible as configuration noise.** Between the quick (4 × 2000) and full
  (4 × 5000) runs, the full-sample SV(rw) ΔLPS moved as follows:
  - at h = 1, by ≤ 0.003 in every marginal cell and by 0.002 joint;
  - in GDP and INFL, by ≤ 0.012 at every horizon;
  - in TBILL, by ≤ 0.036 (h = 4);
  - in UR, by +0.113 at h = 2 (one degenerate origin) and +0.161 at h = 4 (seven);
  - in the joint ΔLPD, by +0.351 at h = 4 and +0.210 at h = 8. (Computed from both JSONs, `C`;
  differences of PROVISIONAL / recorded numbers.)

### 7.7 Ordering robustness (criterion 6 and `ordering_sensitivity`)

*Table 7.7. Source: `orderings.per_ordering[*]` and `orderings.results[*].summary` in the full-run
JSON; ranges computed (`C`). PROVISIONAL.*

| quantity (h = 1) | GDP | INFL | UR | TBILL | joint |
| :- | -: | -: | -: | -: | -: |
| ΔLPS, identity ordering (main stage) | +0.126 | +0.231 | +0.159 | +0.378 | +0.809 |
| ΔLPS, min–max over 24 orderings | 0.091–0.147 | 0.209–0.236 | 0.153–0.212 | 0.351–0.428 | 0.807–0.887 |
| SV(rw) `lpd_range` over orderings | 0.057 | 0.029 | 0.061 | 0.077 | 0.083 |
| gain / range (identity gain ÷ `lpd_range`) | 2.2 | 7.8 | 2.6 | 4.9 | 9.8 |
| max one-sided DM p over orderings | 0.008 | 0.0004 | 0.021 | 3e-6 | 9e-7 |
| Constant `lpd_range` over orderings | 0.003 | 0.003 | 0.005 | 0.002 | 0.027 |

- **h = 1 ranking is order-robust** (P): SV(rw) beats Constant in every ordering in every variable,
  and every DM p is < 0.03. This matches N08 §B3's expectation that "the SV-beats-constant ranking
  should survive any ordering at h = 1".
- **Ordering spread vs the literature.** N08 §D item 11 lists the constant-A ordering spread for a
  4-variable VAR-SV as UNVERIFIED and expected to be below ARRS's ≈ 0.25 nats/quarter joint
  (time-varying `A_t`; Arias, Rubio-Ramírez & Shin 2023 [CITE?]). The full run gives 0.083 joint at
  h = 1 (P). If confirmed by the acceptance run, this is a small empirical contribution; its
  novelty is to be checked against the literature. The gain-to-spread ratios of 2.2–7.8 per
  variable overlap N08 §B3's 1.9–3.5 (computed from CR's gains and ARRS's spreads) only for GDP
  (2.2) and UR (2.6); INFL (7.8) and TBILL (4.9) are above that range. (corrected by fact-check:
  the dossier said the ratios "sit around" 1.9–3.5.)
- **Spread grows with h:** SV(rw) joint `lpd_range` is 0.083, 0.281, 1.414 and 5.173 at h = 1, 2,
  4, 8 (P). At h ≥ 2 that includes the degenerate-estimator noise.
- **Convergence across orderings**, SV(rw), share of 106 origins (P): 57% (ordering 9,
  INFL > UR > TBILL > GDP, 60/106) to 82% (ordering 18, TBILL > GDP > INFL > UR, 87/106).
  Constant converged at 100% in every ordering. `converged_orderings` counts orderings in which
  *all* origins converged (`ordering.py` l. 1504): 0 for SV(rw), 24 for Constant.
- **A direct seed-to-seed datum** (`C`; P). The main-stage SV(rw)-vs-Constant backtest and
  ordering 0 of the ordering stage are the same models in the same ordering
  (GDP > INFL > UR > TBILL) under different backtest seeds (§9.3).
  - The h = 1 marginal ΔLPS agree to ≤ 0.002: GDP +0.126 vs +0.124, INFL +0.231 vs +0.231, UR
    +0.159 vs +0.159, TBILL +0.378 vs +0.375.
  - The joint h = 1 ΔLPD differs by 0.029 (+0.809 vs +0.838), and SV(rw) converged at 72 vs 69
    of 106 origins.
  - 0.029 is the size of Constant's whole 24-ordering joint range (0.027), consistent with that
    range being Monte Carlo noise. It is also about a third of SV(rw)'s joint h = 1 range (0.083).
  - Yet that joint cell has a reported SNR of 16.6, from a noise floor with one replicate (one
    degree of freedom). The difference mixes both arms' noise, while `lpd_range` is computed on
    SV(rw)'s score alone, so it is indicative, not a measurement of `σ_MC`.
- **A false positive of the ordering diagnostic.** `Constant` is order-invariant in theory (Chan
  2022; the report says its spread "is Monte Carlo noise"), yet `ordering_sensitivity` returned
  `material = True` for it (P). Two flags fired: joint h = 2 (`lpd_range` 0.189, SNR 14.3) and UR
  h = 8 (1.233, SNR 7.5). Both are h > 1 cells, where the path-conditioned estimator is noisy, and
  the noise floor rests on one replicate (`noise_reps = 1`). The pre-registration keeps
  `noise_reps` unchanged. See the open questions.

### 7.8 Run-level warnings worth a sentence in the paper

From the report's "Warnings" section (P):

- **"Non-centred sign fold in more than 1% of sweeps"** for GDP and INFL (12–30 origins per SV
  arm; SV(rw) 15, SV 13, SV(rw,t10) 30, AR-SV(0.9) 12): these equations are near-homoskedastic,
  where the library's measure-preserving fold differs from stochvol/bayesianVARs (D-7).
  (corrected by fact-check: the SV(rw,t10) example message lists GDP, INFL **and UR**, so "GDP and
  INFL" holds for the other three arms' examples only; the report groups messages that differ in
  their variable lists, so per-variable counts are not available.) That it is the same region
  (GDP's σ → 0) as the crash in §8 is an inference from the warning text, not a measurement.
- **The R8 non-invariance warning** fires at all 106 origins of every SV arm (SPEC R8).
- **The own-lag-mean warning** fires at every origin of every arm (§3).

### 7.9 Baseline levels against CR

*Table 7.9. Source: `relative.full["SV(rw)"].variables[v]["1"].lps_base` and `.rmse_base` (the
Constant arm), extracted with `uv run --offline python -c '…'` (`C`); CR levels from N08 §C5
("Benchmark VAR levels (full sample, h = 1)"). PROVISIONAL.*

| h = 1, Constant / CR VAR | GDP | INFL | UR | TBILL |
| :- | -: | -: | -: | -: |
| average log score, ours (RB) | −1.115 | +0.138 | −0.024 | −0.906 |
| average log score, CR (Gaussian approx.) | −1.118 | −0.235 | −0.034 | −0.832 |
| RMSE, ours | 0.726 | 0.188 | 0.246 | 0.455 |
| RMSE, CR | 0.675 | 0.269 | 0.240 | 0.454 |

- GDP, UR and TBILL levels are close to CR's. Inflation forecasts are much more accurate in our
  data: RMSE 0.188 vs 0.269 (corrected by fact-check: was "much sharper"; RMSE measures accuracy,
  not sharpness).
- The two log-score rows use different estimators (RB vs CR's Gaussian approximation) and
  different data (final vintage vs real-time), so this table is a sanity check, not a parity
  test.
- This supports the paper's statement that final-vintage data mostly shift levels (N08 §C6), with
  inflation as the exception.

### 7.10 What the full run says, in one paragraph (draft, PROVISIONAL)

With the library's default SV priors, the RW-SV VAR improves on the constant-volatility VAR
at h = 1 and 2 in every variable, in both samples, in log score, CRPS and RMSE. At h = 1 the
log-score gain holds in all 24 variable orderings (full sample), with an ordering spread about a
tenth of the joint gain. At longer horizons the ranking depends on the scoring rule. Under CR's
Gaussian approximation the unemployment and bill-rate gains turn negative, as CR report. Under the
path-conditioned RB estimator the bill-rate gain turns negative too (−0.081 at h = 4, −0.207 at
h = 8), but the unemployment gain does not (+0.969, +1.954), and that estimator is unreliable
exactly at the crisis origins that decide those cells. `compare()` withholds a verdict because the
SV fits miss the R-hat < 1.01 / bulk ESS > 400 targets at 32–63% of origins at 4 × 5000 draws (the
reported worst quantities miss on R-hat). (corrected by fact-check: the draft said the RB gains of
both UR and TBILL "do not" turn negative, contradicting Table 7.2; it said the h = 1 result "holds
in all 24 orderings" for CRPS and RMSE too, but the ordering stage only records log scores, full
sample; and a fit fails on R-hat or ESS, per the report's warnings text and the last
`DEVIATIONS` entry, "R-hat/ESS targets of DESIGN §8.2".) **This paragraph must be
rewritten from the acceptance run.**

---

## 8. Run history and engineering findings

Timeline from T2 (issue #3; UTC, 2026-10-23), with what each step contributes to the paper:

1. **01:34Z:** item 7 added (full replication). The quick run's expected issues are noted:
   criterion 4 failed for UR (+1.92 at h = 8), and `compare()` is gated by R-hat ≈ 1.01–1.03.
2. **02:38Z, HOLD:** the independent WP23/WP24 review changed cache keys. MCMC backtest keys now
   carry `"seeding": "fit_seed-v1"`, and the Gaussian-moment files are keyed like the scores. The
   review also found the RB LPD degenerate at h ≥ 4 at the crisis origins, so the report gained
   the "RB degen." column and the reliability section (GO comment, 02:58Z, at `8a622c5`).
   *Paper point:* the cache key records everything that changes draws (DESIGN D-23).
3. **03:00Z:** full run #1 started on `8a622c5`, `--n-jobs 6`.
4. **Crash (04:10Z comment).** `MCMCNumericalError` in `EQ_VOL`, equation 0, chain 0, burn-in sweep
   318, code 2001, pivot 0. The fit was SV(rw), reduced form, origin 1987Q2, ordering GDP, TBILL,
   INFL, UR. It happened after the main stage and the Constant orderings had finished.
   - The failure replay itself crashed (`TracerArrayConversionError`), which was fixed in
     `93f2bcd` with a test.
   - Diagnosis: GDP's σ = 4.98e-9 is a valid state under `σ ~ ±N(0, B_σ)`. But in the scaled system
     `Q̃ = σ²Q` the terms `σ²D ≈ 1e-17` fall below the ulp of the Laplacian diagonal, and the level
     pivot `≈ σ²(1/V₀ + ΣD)` came out 0.
   - Fix `f60d736`: carry the pivots as their excess over the next edge,
     `q_t = δ_t + q_{t−1}/(1 + q_{t−1})`, which never cancels. Invariance: 639 of 656 arrays are
     bitwise identical (every non-RW composition); RW states agree to ≤ 3.3e-11 (3 sweeps) and
     ≤ 6.7e-10 (60-sweep fits).
   - **Paper status: verified** (NOTES 17; `tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma`,
     l. 292, against a 60-digit reference; `::test_rw_step_at_a_tiny_sigma_is_ok`, l. 341; DESIGN
     §3.7.4 step (a), l. 936–945).
   - The fix removes the crash, not the slow mixing in the σ → 0 region (NOTES 17).
5. **~04:12Z:** full run #2 on `4bb978a` with a fresh cache. Finished with exit 0, no crash, in
   1.10 h (05:17Z comment).
6. **Independent review of `f60d736`:** no defect. Follow-ups in `4fc0162` (RW `aux_log_likelihood` on
   the same factorisation, stronger tiny-σ tests); the 656-array invariance set is bitwise
   unchanged, "so the sampler output and this report stand" (05:17Z).
7. **05:46Z / 10:44Z:** PR #4 opened, then merged by the user (`62d15f6`).
8. **After the merge (T3 only, uncommitted):** the convergence diagnostic, the path-integrated
   estimator D-34, and the pre-registration of the acceptance run (§9).

The first run's main stage had convergence shares SV(rw) 65%, SV 41%, SV(rw,t10) 61%, AR-SV(0.9)
59% (04:10Z comment). These are close to run #2's 68/37/57/62%. The fit seeds are the same:
`src/rai_bvar/evaluation/backtest.py`, `src/rai_bvar/structural/ordering.py` and
`benchmarks/clark_ravazzolo.py` are identical between `8a622c5` and `4bb978a` (`git diff
8a622c5 4bb978a --stat` on those paths is empty; `C`). (corrected by fact-check) The code
differs by much more than the RW AWOL fix and the QR a-row draw: `git diff --stat 8a622c5
4bb978a` touches 41 files (3122 insertions), and `git merge-base --is-ancestor` shows that WP25 and
its follow-ups (`dd8244e`, `00524a9`, `2cd8ec4`), the failure-replay fix `93f2bcd`, the QR draw
`3d3f51b` and the RW fix `f60d736` are all in `4bb978a` and none in `8a622c5` (`C`). The lineage is
stated directly in the 05:17Z comment ("Code: `4bb978a`, the integration head: WP23/WP24 review
fixes `8a622c5`, WP25, the QR a-row draw and the RW AWOL fix") and in the `551f181` commit message;
the draft's "No single comment states it" was wrong. That the differences are rounding-level and
compound along chains is an *inference* from the invariance evidence (QR draw: ≤ 4.1e-10 after
60-sweep fits, T2 02:49Z; RW fix: 639 of 656 arrays bitwise, RW fits ≤ 6.7e-10, T2 04:10Z; WP25:
537 arrays bitwise without dummy rows, NOTES 16, recorded). Recorded.

---

## 9. The acceptance run (pre-registered; T3, to check)

Source for this section: `/Users/yigitokar/rai-bvar/benchmarks/reports/clark_ravazzolo_preregistration.md`
(untracked), "Written 2026-10-23, before any score of the acceptance run was computed". Everything
here is **to check** until it is committed and run.

### 9.1 What changes, and only this

1. **MCMC budget for every MCMC arm, in every stage:** 4 chains × 20,000 kept draws, burn 1000,
   thin 1 (from 4 × 5000).
2. **Primary LPD becomes the path-integrated RB estimator** (`lpd_method="integrated"`, D-34). It is
   identical to the old estimator at h = 1. The path-conditioned estimator is reported alongside as
   a secondary column that "never enters a rule".

### 9.2 What does not change

The pre-registration lists: data and checksums, the four series and transformations, the 1965Q1
start, all models and the mean prior, the library-default SV priors, origins, horizons, 1000
forecast draws, backtest seed 0, per-origin fit seeds, cold starts, the 24-ordering stage "and the
same noise replications", and the criteria rules and thresholds as coded. The rules include the
95% gate and the RB-degenerate flag, which stays descriptive.

- **CR's RW-SV σ² prior `IG(1/2, 0.0175)` is explicitly not adopted**: "adopting it now would be a
  modelling change decided after seeing results".
- **The run is honestly labelled non-blind:** the changes "are motivated by that failure analysis,
  so the run is not blind to the previous results. What is pre-registered is that the settings
  below were fixed before any score of the new run existed". The paper should use this framing
  verbatim or close to it.
- "No further run with different settings is made after this one's scores are seen, except to fix
  a bug."

### 9.3 Evidence for the new budget: the convergence diagnostic (T3, to check)

The diagnostic used fit-level diagnostics only, with no forecasts or scores, on the same data with
the CR main-stage fit seeds:

| SV(rw) fits | 4 × 5000 | burn 5000 | 8 chains | 4 × 20,000 |
| :- | -: | -: | -: | -: |
| identity ordering | 72/106 | 79/106 | 92/106 | **106/106** (max R-hat 1.0061, min bulk ESS 1763) |
| INFL > UR > GDP > TBILL | 71/106 | — | — | **106/106** (max R-hat 1.0049, min ESS 1689) |
| INFL > UR > TBILL > GDP | 63/106 | — | — | **106/106** (max R-hat 1.0046, min ESS 1717) |

(— = not reported in the pre-registration.) "All 520 + 424 diagnostic fits finished without error. Every base-budget failure is on a σ
quantity and is marginal (R-hat < 1.03), consistent with R-hat sampling noise at bulk ESS ~600, not
with a stuck sampler."

**Consistency check against the committed full run (`C`).**

- Identity ordering: 72/106 matches the main stage (`converged.full["SV(rw)"]` = 0.679 = 72/106).
- The two worst orderings do **not** match the ordering stage of the full run: there ordering 8
  converged at 63/106 and ordering 9 at 60/106, against 71/106 and 63/106 in the diagnostic.
  Ordering 0 (identity) was 69/106 in the ordering stage against 72/106 in the main stage.
- This is plausible rather than an error. The ordering stage draws its backtest seeds from
  `backtest_seeds` per ordering (`ordering_stage`, l. 799 and 814), while the main stage and the
  diagnostic use the main-stage seeds.
- It is itself evidence for the pre-registration's claim: at 4 × 5000 the converged count moves by
  up to 8 origins with the seed, which is R-hat sampling noise near the threshold.
- To check: the diagnostic should be committed (script and output) so that the paper's
  convergence-vs-budget table has a committed source.

**Not covered by the diagnostic:** SV, SV(rw,t10) and AR-SV(0.9) at 4 × 20,000. In the full run
these converged at 37%, 57% and 62% (P). Criteria 1–5 do not use the gate, and criterion 6 only
needs SV(rw), so this does not affect the criteria. It does affect whether the main-stage
`compare()` verdicts of those arms stay "inconclusive".

### 9.4 The integrated estimator (D-34; T3, to check)

From the uncommitted DESIGN §7.3 bullet, the D-34 row and the NOTES 8b addendum:

- **Conditioning.** Draw s is conditioned on its parameters and its drawn latent scale path only,
  and `y_{T+1..T+h−1}` is integrated out:
  `y_{T+h} | θ_s, λ_s ~ N(m_h, M_h M_hᵀ)`, with
  - `m_h = cond_mean_h − Σ_{k<h} Φ_{h−k} u_k` (the zero-shock mean);
  - `M_h = [Φ_{h−1}L_1 | … | Φ_0 L_h]`;
  - `Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}`.

  Subsets are factored by QR of `M_h[idx]ᵀ`; there are no random numbers. The MCSE uses the same
  delta-method formula.
- **Scope.**
  - The estimator needs the new forecast variable `lag_coefs`, which only MCMC forecasts of linear
    or zero means carry. Every arm in this script is MCMC with a linear mean (§4), so it applies to
    all of them.
  - Analytic, benchmark and ordering-pooled forecasts raise at h > 1, and so does
    `rb_transitory=True` at h > 1.
  - The SPEC §7 default (`"conditioned"`) is unchanged; D-34 is opt-in.
- **Tests:** `tests/unit/test_lpd_integrated.py` and `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`,
  both untracked. Claimed results:
  - 1e-10 against a companion-form oracle at h = 2..8 with 6-s.d. tails;
  - per-draw densities match `jax.jacfwd` of the forecast core to 1e-10;
  - on a synthetic persistent VAR at h = 8 and 6 s.d., the conditioned estimator gives −35.2
    (MCSE 1.0) against the exact −22.3.

  Status: the estimator is "verified (tests)" per the uncommitted NOTES text, but it is **to
  check** for the paper until committed.

### 9.5 Which numbers will change, and how

| Group | Numbers (§7 tables) | Expected after the acceptance run | Why |
| :- | :- | :- | :- |
| A. Changes systematically | every RB ΔLPS/ΔLPD at h ≥ 2 with a non-zero degen. count (full: all h = 2, 4, 8 cells; pre-crisis: h = 4, 8), the joint h = 4/8 ΔLPD, the DM p-values at h ≥ 2, the 28 RB/Gaussian sign-disagreement cells, criterion 4's UR cell and the pre-crisis GDP h = 8 cell, the variant comparisons at h ≥ 2 (criterion 5), the h ≥ 2 rows of the structural pair, `ordering_sensitivity` `lpd_range`/SNR/`material` at h ≥ 2, and the RB-degenerate counts themselves | new values; the size of the UR h = 4/8 cells is expected to fall sharply (review estimate for UR h = 8: about +0.23, NOTES 8b, recorded) | new primary estimator for h > 1 |
| B. No systematic change expected (Monte Carlo noise only) | all h = 1 LPD numbers (marginal and joint, both samples, all arms, all 24 orderings), CRPS and RMSE ratios at every h, the Gaussian-approximation scores, the DM p-values at h = 1 | small moves; the quick → full moves at h = 1 were ≤ 0.003 per marginal cell and 0.002 joint (§7.6) | the estimators coincide at h = 1 (D-34); CRPS, RMSE and the Gaussian moments use the draws only; but 4 × 20,000 chains give different posterior draws |
| C. Changes because of the budget | convergence shares, all gate-on `compare()` verdicts, criterion 6's gate-on count, the MC s.e. of the totals, runtime | SV(rw) expected at or near 106/106 (diagnostic, T3); other arms unknown | 4× longer chains |
| D. Does not change by design | data checks, models, priors, origins, rules, fit seeds, forecast draws | identical | pre-registration "What does not change" |

**Criterion-level expectations, stated before the run.** The first two lines are stated in the
pre-registration; the rest are this dossier's own reading of the full-run numbers.

- **Criterion 4 may still fail on UR** (pre-registration, verbatim intent). With the review's
  estimate of +0.23 at h = 8 against +0.157 at h = 1, the coded rule `ΔLPS(h=8) < ΔLPS(h=1)` breaks.
  If it fails, it is to be reported as a difference from CR and attributed with evidence where
  possible (final vintage, the `iw_equivalent` baseline, default SV priors), not tuned. **The
  paper's narrative must not bank on a pass.**
- **Criterion 6 is expected to pass if the convergence gate is met** (pre-registration): its h = 1
  scores are estimator-independent, and the ungated verdict was "better" in 24/24 orderings (P).
  The diagnostic covers 3 of the 24 orderings.
- **Criterion 1 at h = 2 is not guaranteed.** The thinnest full-sample h = 2 cells are UR (RB
  +0.266, Gaussian +0.043, DM p 0.090) and TBILL (+0.153 / +0.094, p 0.073) (P). The quick → full
  move of UR h = 2 (+0.152 → +0.266) shows how estimator-sensitive that cell is.
- **The pre-crisis half of criterion 4** needs GDP h = 8 > 0: RB +0.108, Gaussian −0.004, 8
  degenerate origins (P). It could flip; INFL h = 8 (+0.295 / +0.345) is safe.
- **Criteria 2 and 3** are estimator-independent and far from their thresholds: the largest h ≤ 2
  CRPS ratio is 0.936 and the largest RMSE ratio 0.970, against 1 and 1.02 (P).
- **Criterion 5** uses all horizons, so it will be recomputed. AR-SV(0.9) vs SV(rw) at h ≥ 2 is the
  cell to watch (Table 7.5a).

---

## 10. Proposed tables and figures for the paper

Principle: every table and figure is generated from a committed file of the acceptance run. Where
the committed JSON does not hold the data, the figure needs an export step that does not exist yet
(open question 1).

### 10.1 Tables

| # | Content | Data source (acceptance run unless stated) | Status of source |
| :- | :- | :- | :- |
| T-E1 | Data: CR definition, FRED-QD mnemonic, transformation, tcode override, sample, checks | script l. 118–125, `load_data`; N08 §C2; JSON `data_checks` | static + run JSON |
| T-E2 | Model arms: library spec (code), CR counterpart, priors | `models()`/`structural_models()` (l. 273–292); DESIGN D-13; N08 §C3; JSON `models` | static |
| T-E3 | The six criteria: N08 wording, operational rule, met/not met, numbers | JSON `criteria[*].rule/met/detail`; N08 §C5 | run JSON |
| T-E4 | **Headline**: SV(rw) vs Constant, full sample, 4 variables × h ∈ {1, 2, 4, 8}: ΔLPS (integrated, primary), ΔLPS (conditioned, secondary), ΔLPS (Gaussian), CRPS ratio, RMSE ratio, DM p, plus CR's three columns | JSON `relative.full["SV(rw)"]` (the secondary column's JSON key is to be fixed by the acceptance script; the uncommitted T3 script uses `lpd_marginal_conditioned`, corrected by fact-check); `CR_FULL` (l. 128–147) | run JSON + script constant |
| T-E5 | Pre-crisis compact: h = 1 and h = 8 vs CR | JSON `relative.pre_crisis["SV(rw)"]`; `CR_PRE_DLPS` (l. 148–153) | run JSON |
| T-E6 | All SV arms: joint ΔLPD by horizon, converged share, `compare()` verdicts (gate on / off), MCS membership | JSON `relative.*[arm].joint`, `converged`, `verdicts` | run JSON |
| T-E7 | Ordering robustness at h = 1: identity gain, min–max over 24 orderings, `lpd_range` for both arms, max DM p, converged share range | JSON `orderings.per_ordering[*]`, `orderings.results[*].summary` (h = 1 per-ordering values are in the JSON; for h > 1 the per-variable values exist only as `lpd_range`, but each ordering's `verdicts` list holds the joint ΔLPD total, its MCSE, DM p and MCS flags at h = 1, 2, 4, 8, gate on and off (corrected by fact-check: the draft said h > 1 only as `lpd_range`)) | run JSON |
| T-E8 (methods) | RB estimators at the diagnostic cell (Constant UR h = 8, 2008Q2): conditioned with S = 1000 / 8000 / 64,000 vs integrated vs exact; the synthetic h = 8, 6-s.d. case (−35.2 vs −22.3) | NOTES 8b (recorded; review computations); `tests/unit/test_lpd_integrated.py` (T3); the acceptance run's per-origin primary and secondary scores | **needs** a committed re-computation (NOTES 8b: "re-run with the library fix before publishing") |
| T-E9 | Convergence vs MCMC budget (4 × 5000, burn 5000, 8 chains, 4 × 20,000; identity and two worst orderings) | pre-registration (T3) | **needs** the diagnostic script and output committed |
| T-E10 (appendix) | Baseline levels vs CR (Table 7.9) | JSON `lps_base`/`rmse_base`; N08 §C5 levels | run JSON |
| T-E11 (appendix) | Compute: machine, `n_jobs`, stage times, fits, total wall time | JSON `seconds`, `versions`, `config`; stage times currently only in T2 | stage times **need** to go into the JSON or the report |
| T-E12 (appendix) | Deviations from CR | JSON `deviations` | run JSON |

### 10.2 Figures

| # | Content | Data source | Available? |
| :- | :- | :- | :- |
| F-E1 (methods, recommended) | Per-origin ΔLPS (SV(rw) − Constant), UR, h = 8 (and h = 1 for contrast), origins 1985Q1–2011Q2, **path-conditioned vs integrated**. The 2007Q4–2008Q3 spikes (+41.6, +41.4, +40.4 nats in the full run) should collapse under the integrated estimator. Recession bands from the target dates. | per-origin scores. The committed JSON holds only means, medians, the top-3 origins and the degenerate-origin lists. The per-origin series exist only in the machine-local backtest cache (`~/.cache/rai-bvar-cr-4bb978a` for the full run). | **No committed source.** The acceptance run must export per-origin scores, e.g. the backtest `scores` datasets as NetCDF or CSV. |
| F-E2 | Cumulative joint ΔLPD at h = 1 (and h = 4) over origins, SV(rw) vs Constant, with the pre-crisis boundary: the SPEC §9.5 `res.plot_cumulative_lpd()` plot, which also demonstrates the API | per-origin scores | **No committed source** (same export) |
| F-E3 | Ordering dot plot (ARRS Fig. 1 style; N08 §B4 `plot()`): h = 1 ΔLPS per variable for the 24 orderings, identity marked, with the noise floor. Shows the gain is ≫ the spread. | JSON `orderings.per_ordering[*].dlps_h1` (h = 1); for h > 1 the per-ordering per-variable values are not in the JSON, but the joint ΔLPD totals are (`per_ordering[*].verdicts[*].dlpd` by horizon; corrected by fact-check) | h = 1: yes (from the acceptance JSON); joint h > 1: yes |
| F-E4 | Converged share per ordering, 4 × 5000 vs 4 × 20,000 (dot pairs), with the 95% gate line | full-run JSON `orderings.per_ordering[*].converged_sv` (P) and the acceptance-run JSON | yes, once the acceptance JSON exists |
| F-E5 (optional) | The four series, 1965Q1–2026Q2, with the evaluation window shaded | FRED-QD 2026-09 via `rb.datasets.load` at build time (data not bundled, SPEC §5.8) | generated, not committed (licensing note in SPEC §5.8) |
| F-E6 (optional) | Posterior median and 68% band of SV(rw) volatilities at the last origin 2011Q2 | a fit at 2011Q2 | not available from the backtest (scores only) |

Recommended set for the main text: T-E3, T-E4, T-E7 and F-E1, with F-E2 or F-E3 as the second
figure. T-E8 and F-E1 belong to the methods section on the log score rather than the illustration,
if the paper keeps them separate.

---

## 11. Claims ready for the paper text, with status

| Claim | Evidence | Status |
| :- | :- | :- |
| The data loader verifies SHA-256 checksums of a pinned vintage and the script asserts the quarterly aggregation and no look-ahead at run time | script l. 179–262; SPEC §5.8, R9; JSON `data_checks` | verified per run; numbers PROVISIONAL |
| Five volatility arms and a structural pair, no prior tuned, library-default SV priors except the two CR variants (corrected by fact-check: the mean prior is CR's Minnesota with `hierarchy="fixed"` and zero own-lag means, `minnesota`, l. 268–270, which departs from the library defaults: `hierarchy="eb"` is the default, DESIGN D-12, and the zero means are `DEVIATIONS` item 1) | script l. 268–292, `DEVIATIONS` | verified (code) |
| SV(rw) beats Constant at h = 1, 2 in every variable, both samples (LPD, CRPS, RMSE) | full-run JSON | recorded, **PROVISIONAL** |
| The h = 1 ranking holds in all 24 orderings; the joint ordering spread is 0.083 nats/quarter | full-run JSON `orderings` | recorded, **PROVISIONAL** |
| The path-conditioned RB log score is unreliable in the tails at h > 1 | NOTES 8b; `RELIABILITY_NOTE`; DESIGN §2.10 l. 616 | recorded (review computations); re-run required |
| The path-integrated estimator is exact for one parameter draw and matches a companion-form oracle to 1e-10 | `tests/unit/test_lpd_integrated.py` (untracked) | **to check** (T3) |
| The scaled RW AWOL draw failed at σ ≈ 5e-9; the pivot recursion fixes it with bitwise invariance elsewhere | NOTES 17; `test_sv_core.py` l. 292, 341; T2 04:10Z | verified (tests); invariance counts recorded |
| 4 × 20,000 draws make SV(rw) converge at 106/106 origins | pre-registration | **to check** (T3) |
| The acceptance run's settings were fixed before any of its scores existed | pre-registration (dated 2026-10-23) | **to check**: commit the pre-registration *before* the run, so the commit order proves it |

---

## 12. Caveats a referee will raise (and the current answer)

1. **Real-time vs final data.** CR use real-time RTDSM vintages; we use the final vintage (D4).
   Answer: N08 §C6, final data shift score levels and should keep rankings; inflation is the
   exception in levels (§7.9). An exact-data mode built from the JAE archive files is possible
   (N08 §C2 "Optional exact-data mode", `data_at(origin)`, SPEC §5.5) but not implemented for this
   report.
2. **WP vs JAE numbers.** The comparison uses the WP's tables, and JAE numbers are UNVERIFIED
   (N08 §C1).
3. **Baseline.** `Constant("iw_equivalent")` is not CR's normal-diffuse VAR (N08 §C3).
4. **SV priors.** These are the library defaults, not CR's; CR's RW σ² prior is not adopted, by
   pre-registration.
5. **Significance.** DM/HLN p-values are descriptive under an expanding window (SPEC §9.3), and
   `compare()` decides through the MCS. The paper should not present DM p-values as valid tests.
6. **Operational criteria.** The thresholds are the project's own (§6); print them.
7. **Pre-registration is not blind.** Say so (pre-registration text, §9.2).
8. **One seed.** Backtest seed 0 only. The ordering stage gives a second seed set for SV(rw) at
   h = 1, and the convergence comparison in §9.3 shows the seed sensitivity at 4 × 5000.
9. **Estimator change between runs.** The primary LPD changes at h > 1 between the full and the
   acceptance run. The paper must report which estimator produced which number, and should show
   both (secondary column).

---

## Open questions

1. **Per-origin score export.** Will the acceptance run commit per-origin scores (both RB
   estimators, Gaussian, CRPS, squared errors, convergence flags) for every arm and ordering?
   Figures F-E1 and F-E2 have no committed data source without it.
2. **`--config acceptance`** is named in the pre-registration but is not in the committed
   `CONFIGS` (l. 166–173; corrected by fact-check, was 166–175). The uncommitted working-tree
   script in the other checkout already adds it (T3, §0.2), so the question is which commit lands
   it. Will the pre-registration be committed before the run starts (needed for the "fixed before
   any score existed" claim to be checkable)?
3. **Issue #3 vs the brief.** The convergence diagnostic and the acceptance plan are not in issue
   #3 (§0.1). Will they be posted there or committed (diagnostic script and output) so the paper can
   cite a durable source?
4. **Seed-dependence of convergence counts.** Pre-registration 71/106 and 63/106 vs ordering-stage
   63/106 and 60/106 (§9.3). Confirm the explanation (different backtest seeds per ordering) and
   say it in the paper.
5. **Convergence of SV, SV(rw,t10) and AR-SV(0.9) at 4 × 20,000** is undiagnosed (37/57/62% at
   4 × 5000). Is "inconclusive" acceptable for their gate-on main-stage verdicts?
6. **Ordering-diagnostic false positive.** `ordering_sensitivity` flags the order-invariant
   Constant arm as `material` (joint h = 2, UR h = 8). Is this the path-conditioned estimator, the
   one-replicate noise floor (`noise_reps = 1`), or both? The seed-to-seed datum in §7.7 (joint
   h = 1 ΔLPD differs by 0.029 between two seeds, against SV(rw)'s 24-ordering range of 0.083 and a
   reported SNR of 16.6) suggests the one-replicate floor understates the noise. Should `noise_reps` stay 1 in the
   acceptance run (the pre-registration keeps it)? Should the paper report `material` for the
   Constant arm as a known limitation of the diagnostic?
7. **DESIGN R6 promise** (Mode I vs II, data-centred vs stochvol `μ` priors in the CR report) is not
   delivered by the full run or the pre-registration. Drop it from DESIGN, or run it as a
   labelled post-hoc robustness check outside the pre-registered criteria?
8. **The RB/Gaussian split at h ≥ 4 for UR.** If the integrated estimator keeps UR h = 8 positive
   while the Gaussian score is negative, the paper needs a short explanation (tail shape vs
   variance). N08 §C4's prediction is the starting point; it needs evidence, e.g. per-origin
   contributions.
9. **Criterion 4's wording vs rule.** N08 says the gain "shrinks or turns negative". The coded rule
   compares h = 8 with h = 1 only. If UR fails at about +0.23 vs +0.16, should the paper discuss
   whether a rule based on the h = 4/8 trend would be more faithful? It must not be changed for the
   acceptance run (pre-registration).
10. **Inflation gains 1.6–3.2× CR's** (corrected by fact-check: was "2–3×"; §7.2 computes 3.1×,
    3.2×, 1.6× and 2.0× at h = 1, 2, 4, 8). Is this the final vintage, `GDPCTPI`, or the baseline? A
    real-time subsample check is outside Phase 2. Decide whether the paper speculates or just
    reports.
11. **The NOTES 8b numbers** (−93.6, −46 to −63, −36, −19.1, −23.9, −12.7, +0.23) are review
    computations. Will the acceptance run, or a committed script, reproduce them as NOTES 8b
    requires ("re-run with the library fix before publishing")?
12. **Stage timings and machine load** (649 / 982 / 2308 s; load ≈ 25) are only in T2. Should the
    script write them into the JSON for the compute table?
13. **Report "Generated" timestamp.** It is taken when the main stage finishes (the `rep` dict is
    built before the ordering stage, `main` l. 1262–1263), so it is not the run's end time.
    (corrected by fact-check: more precisely, it is taken after the five main-arm backtests and
    the Gaussian moments, but *before* the relative/variant tables, the structural pair's
    backtests and the ordering stage; `main` l. 1262–1263 vs the structural loop that follows.)
    Minor; clarify in the compute table.
14. **The weekly-tier heap-corruption crash** on the M5 Max (T2, 06:57Z and 07:10Z; Phase 1 EB test,
    numpy/scipy on Accelerate) did not touch this run. Should the paper's reproducibility section
    mention platform caveats, or keep them in the repository?

## [CITE?] references mentioned in this dossier

Bibliographic details are as best known. Every entry needs verification at the bibliography stage.
Where N08 §E records a Crossref/OpenAlex confirmation, that is noted, but the entry is still
flagged.

1. Clark, T. E. & Ravazzolo, F. (2015). "Macroeconomic Forecasting Performance under Alternative
   Specifications of Time-Varying Volatility." *Journal of Applied Econometrics* 30(4), 551–575.
   DOI 10.1002/jae.2379 (bibliographic data Crossref-confirmed per N08 §C1/§E; the title as given
   here is from memory). [CITE?]
2. Clark, T. E. & Ravazzolo, F. (2012). "The Macroeconomic Forecasting Performance of
   Autoregressive Models with Alternative Specifications of Time-Varying Volatility." Federal
   Reserve Bank of Cleveland WP 12-18 / Norges Bank WP 2012/09 (N08 §C1). [CITE?]
3. Adolfson, M., Lindé, J. & Villani, M. (2007). "Forecasting Performance of an Open Economy DSGE
   Model." *Econometric Reviews* 26(2–4), 289–328. [CITE?]
4. Chan, J. C. C. (2022). "Asymmetric Conjugate Priors for Large Bayesian VARs." *Quantitative
   Economics* 13(3), 1145–1169. DOI 10.3982/QE1381 (Crossref per N08 §E). [CITE?]
5. Carriero, A., Clark, T. E. & Marcellino, M. (2016). "Common Drifting Volatility in Large Bayesian
   VARs." *Journal of Business & Economic Statistics* 34(3), 375–390. [CITE?]
6. Carriero, A., Chan, J., Clark, T. E. & Marcellino, M. (2022). "Corrigendum to: Large Bayesian
   Vector Autoregressions with Stochastic Volatility and Non-Conjugate Priors." *Journal of
   Econometrics* 227(2), 506–512. [CITE?]
7. Carriero, A., Clark, T. E. & Marcellino, M. (2019). "Large Bayesian Vector Autoregressions with
   Stochastic Volatility and Non-Conjugate Priors." *Journal of Econometrics* 212(1), 137–154.
   [CITE?]
8. Arias, J. E., Rubio-Ramírez, J. F. & Shin, M. (2023). "Macroeconomic Forecasting and Variable
   Ordering in Multivariate Stochastic Volatility Models." *Journal of Econometrics* 235(2),
   1054–1086. DOI 10.1016/j.jeconom.2022.04.013 (Crossref per N08 §E). [CITE?]
9. Chan, J. C. C., Koop, G. & Yu, X. (2024). "Large Order-Invariant Bayesian VARs with Stochastic
   Volatility." *Journal of Business & Economic Statistics* 42(2), 825–837. DOI
   10.1080/07350015.2023.2252039 (Crossref per N08 §E). [CITE?]
10. Kadiyala, K. R. & Karlsson, S. (1997). "Numerical Methods for Estimation and Inference in
    Bayesian VAR-Models." *Journal of Applied Econometrics* 12(2), 99–132. [CITE?]
11. Jacquier, E., Polson, N. G. & Rossi, P. E. (1994). "Bayesian Analysis of Stochastic Volatility
    Models." *Journal of Business & Economic Statistics* 12(4), 371–389. [CITE?]
12. Jacquier, E., Polson, N. G. & Rossi, P. E. (2004). "Bayesian Analysis of Stochastic Volatility
    Models with Fat-Tails and Correlated Errors." *Journal of Econometrics* 122(1), 185–212 (N08
    §E). [CITE?]
13. Cogley, T. & Sargent, T. J. (2005). "Drifts and Volatilities: Monetary Policies and Outcomes in
    the Post WWII US." *Review of Economic Dynamics* 8(2), 262–302. [CITE?]
14. Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of Business &
    Economic Statistics* 13(3), 253–263. [CITE?]
15. Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the Equality of Prediction Mean
    Squared Errors." *International Journal of Forecasting* 13(2), 281–291. [CITE?]
16. Hansen, P. R., Lunde, A. & Nason, J. M. (2011). "The Model Confidence Set." *Econometrica*
    79(2), 453–497. [CITE?]
17. Clark, T. E. & McCracken, M. W. (2001). "Tests of Equal Forecast Accuracy and Encompassing for
    Nested Models." *Journal of Econometrics* 105(1), 85–110. [CITE?]
18. Giacomini, R. & White, H. (2006). "Tests of Conditional Predictive Ability." *Econometrica*
    74(6), 1545–1578. [CITE?]
19. Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and
    Estimation." *Journal of the American Statistical Association* 102(477), 359–378. [CITE?]
20. Krüger, F., Lerch, S., Thorarinsdottir, T. & Gneiting, T. (2021). "Predictive Inference Based on
    Markov Chain Monte Carlo Output." *International Statistical Review* 89(2), 274–301;
    arXiv:1608.06802 (N08 §E). [CITE?]
21. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
    "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of
    MCMC." *Bayesian Analysis* 16(2), 667–718. [CITE?]
22. McCracken, M. W. & Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research."
    *Journal of Business & Economic Statistics* 34(4), 574–589. [CITE?]
23. McCracken, M. W. & Ng, S. (2020). "FRED-QD: A Quarterly Database for Macroeconomic Research."
    NBER Working Paper 26872 (later *Federal Reserve Bank of St. Louis Review*, 2021). [CITE?]
24. Croushore, D. & Stark, T. (2001). "A Real-Time Data Set for Macroeconomists." *Journal of
    Econometrics* 105(1), 111–130 (the RTDSM that CR use). [CITE?]
25. Kastner, G. & Frühwirth-Schnatter, S. (2014). "Ancillarity-Sufficiency Interweaving Strategy
    (ASIS) for Boosting MCMC Estimation of Stochastic Volatility Models." *Computational Statistics
    & Data Analysis* 76, 408–423 (relevant to the sign-fold warning, D-7). [CITE?]
26. Chan, J. C. C. & Jeliazkov, I. (2009). "Efficient Simulation and Integrated Likelihood
    Estimation in State Space Models." *International Journal of Mathematical Modelling and
    Numerical Optimisation* 1(1/2), 101–120 (the precision-based "all without a loop" sampler
    behind the AWOL step; SPEC §8). [CITE?]
27. McCausland, W. J., Miller, S. & Pelletier, D. (2011). "Simulation Smoothing for State-Space
    Models: A Computational Efficiency Analysis." *Computational Statistics & Data Analysis* 55(1),
    199–212 (SPEC §8). [CITE?]
28. Clark, T. E. (2011). "Real-Time Density Forecasts from Bayesian Vector Autoregressions with
    Stochastic Volatility." *Journal of Business & Economic Statistics* 29(3), 327–341 (the 1965Q1
    start precedent, N08 §C2). [CITE?]
29. Dawid, A. P. & Sebastiani, P. (1999). "Coherent Dispersion Criteria for Optimal Experimental
    Design." *Annals of Statistics* 27(1), 65–81 (origin of the Dawid–Sebastiani score; cite only if
    the paper names the score). [CITE?]

## Unsupported claims (moved here by fact-check, 2026-10-23)

Claims below had no evidence in the repository, the issue #3 thread or a command run during the
fact-check. They are not to go into the paper until a source is added.

1. **"These are the origins whose h-step target falls in or around a recession … 2009Q1–2010Q4,
   the unemployment peak"** (formerly §7.6, degenerate-origin list). The target-date arithmetic is
   kept in §7.6. The reading that 1992Q3, 2002Q2–2003Q2 and 2009Q1–2010Q4 are recession or
   unemployment-peak quarters is an external macroeconomic fact. No committed file supports it
   (the FRED-QD data are not committed, SPEC §5.8). Source needed: NBER business-cycle dates or the
   `UNRATE` series at build time.

Everything else in the dossier traces to a T1 file, a T2 comment, a T3 file (labelled "to
check") or a `C` command. The inline "(corrected by fact-check)" notes give each fix.

## Fact-check record (2026-10-23)

Checked against the paper checkout at `62d15f6`: `benchmarks/reports/clark_ravazzolo.{md,json}`,
`clark_ravazzolo_quick.{md,json}`, `benchmarks/clark_ravazzolo.py`, `SPEC.md`,
`docs/phase2/DESIGN.md`, `docs/phase2/08_common_sv_and_ordering.md`, `docs/paper/NOTES.md`,
`src/rai_bvar/{api.py, forecast/result.py, evaluation/{backtest,compare,scores}.py,
structural/ordering.py}`, and the cited tests (by `grep -n`). T2 was re-read with `gh api
repos/yigitokar/rai-bvar/issues/3/comments` (17 comments) and `gh pr view 4 --json comments`
(1 comment). T3 was re-read read-only in `/Users/yigitokar/rai-bvar` (`git status`, `git diff`,
the pre-registration).

Every number in Tables 7.1–7.9 and in §7.6–§7.7 was either recomputed from the two JSON files with
`python3 -c 'json.load(…)'` or checked against the committed report, the script constants and
N08 §C5. Tables 7.4 (verdict totals), 7.5a/b and the CR columns were checked that second way. All
of them match. The recomputed numbers include:

- the 28 RB/Gaussian sign disagreements (marginal cells; 8 joint cells also disagree);
- the h = 1 marginal sum 0.894;
- the quick → full moves;
- the 24-ordering ranges and the gain/range ratios;
- the seed-to-seed datum;
- the converged counts 72/39/60/66 of 106.

The pre-registration quotes and the diagnostic table (§9.2–§9.3) match the T3 file verbatim. The
D-34 description (§9.4) matches the uncommitted DESIGN §7.3 bullet, the D-34 row and the NOTES 8b
addendum.
