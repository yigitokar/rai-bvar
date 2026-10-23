# Dossier 01: Positioning and contribution

*Scope.* What problem `rai-bvar` solves, who it is for, where it sits in the Python/R landscape, how it relates
to the reference packages it uses as numerical oracles, what it deliberately does not do, and what is new
relative to existing software and literature. It ends with candidate titles, a 200-word abstract draft, the
list of claimed contributions with evidence, open questions and the `[CITE?]` list.

*Evidence base.* Repository `/Users/yigitokar/rai-bvar-paper`, branch `paper-draft`, commit
`62d15f68c7e7a3de4ce83ee0098293b9c62111b0` (`62d15f6`, "Merge pull request #4 from
yigitokar/phase-2-integration"). Every pointer below is relative to the repository root and refers to that
commit. Commands I ran (all read-only):

- `uv run --frozen --offline pytest --collect-only -q -p no:cacheprovider` → "4964 tests collected"
  (collection only; nothing was executed, no cache written; `git status` clean afterwards).
- `find src -name '*.py' | xargs wc -l` → 27,347 lines; the same over `tests` → 48,013 lines.
- `git log` → 49 commits, dated 2026-10-21 (2), 2026-10-22 (34), 2026-10-23 (13).

*Status legend* (follows `docs/paper/NOTES.md`, header):
**verified** = reproducible from a committed test or fixture; **recorded** = numbers from a run written down
in a committed file but not re-run independently; **to check** = must be re-run, re-derived or looked up
before it goes into the paper. Literature is never cited from memory as fact: every outside reference is
marked `[CITE?]` and collected at the end for the bibliography stage.

*Superseded data.* The Clark–Ravazzolo numbers below come from the committed report
`benchmarks/reports/clark_ravazzolo.md` @ `62d15f6` (generated 2026-10-23 04:19 UTC, line 3 (corrected by fact-check: was "line 6"; every CR line number in this dossier was off by +3 against the committed 401-line file and has been corrected below)). An improved
acceptance run is in progress elsewhere and will replace that report; every CR number in this dossier is
tagged **[CR@62d15f6, superseded]** and must be refreshed. Nothing from that other run (or from the
uncommitted files of the development checkout) is used here.

---

## 1. The problem

### 1.1 The question the package is built around

SPEC §1.1 (SPEC.md lines 27–39) states the organising question verbatim:

> "Does added model complexity improve out-of-sample density forecasts once time-varying volatility is
> accounted for?"

The rationale is SPEC rule R2 "Volatility first (fair ablation)" (SPEC.md line 109): much of the apparent gain
from nonlinear or time-varying mean specifications comes from time-varying volatility and fat tails (the SPEC
cites Sims & Zha 2006 `[CITE?]` and Clark & Ravazzolo 2015 `[CITE?]`). Hence a mean effect must be measured
holding the volatility specification fixed (`BART × SV` vs `Linear × SV`, never vs `Linear × Constant`), and
`compare()` only issues a verdict between models that differ along exactly one axis (SPEC §9.5).

### 1.2 Why existing tooling makes this hard (the paper's motivation section)

Three practical obstacles, each tied to a design rule:

1. **Models come as separate classes with separate outputs.** A typical comparison stitches together an
   analytic conjugate BVAR from one package and an SV sampler from another, each with its own output format,
   and then writes ad hoc evaluation code. `rai-bvar` answers with R1 (composition over inheritance, SPEC.md
   line 107) and R3 (one predictive contract, SPEC.md line 115). *Status of the claim about other packages:*
   **to check** (a qualitative statement; the landscape table of §4 must back it).
2. **Density-forecast comparisons are easy to get subtly wrong.** Kernel-smoothed log scores, tests that are
   invalid for nested models under expanding windows, multiple comparisons over a grid, and look-ahead in
   data-dependent prior settings. Addressed by SPEC §7 (RB log score, never KDE), §9.3 (validity policy, MCS),
   R9 (no look-ahead). The SPEC's position that DM/AG p-values are formally valid only with rolling windows
   follows Giacomini & White (2006) `[CITE?]` and Clark & McCracken (2001) `[CITE?]` (SPEC §9.3).
3. **Samplers can be wrong without anyone noticing.** SPEC R6 (line 121) cites "the CCM (2019) episode, a
   published and widely used sampler that targeted the wrong posterior" (Carriero, Clark & Marcellino 2019
   `[CITE?]`; the error was shown by Bognanni 2022 `[CITE?]` and corrected by Carriero, Chan, Clark &
   Marcellino 2022 `[CITE?]`; SPEC §0 revision table and §15). The package's response is the validation suite
   (DESIGN §9) and the rule that no sampler is merged without Geweke and SBC tests.

A fourth, empirical, obstacle is **variable-ordering dependence** of triangular SV models (R8, SPEC.md
line 125; Arias, Rubio-Ramírez & Shin 2023 `[CITE?]`; Chan, Koop & Yu 2024 `[CITE?]`). `docs/phase2/08_common_sv_and_ordering.md`
§B3 compares magnitudes reported in those papers with the SV gains in Clark & Ravazzolo and concludes that the
T-bill and unemployment cells and every cell at h ≥ 4 are "close enough to the ordering spread that the
replication must report ordering sensitivity alongside the verdict". (Those magnitudes are third-party numbers
transcribed in note 08; the bibliography stage must verify them against the papers: **to check**.)

---

## 2. What `rai-bvar` is

One sentence for the paper (from `pyproject.toml` `description` and README line 3): *ablation-ready Bayesian
VARs: composable mean, volatility and prior components with a fair out-of-sample verdict engine.*

Five design commitments, each with its evidence:

| # | Commitment | Where it is specified | Where it is implemented / tested | Status |
| :- | :- | :- | :- | :- |
| 1 | A model is a composition `BVAR(mean=..., vol=..., lags=...)`; the engine is chosen by the composition, never by the user | SPEC R1, §4.5; DESIGN §3.8, §11.1, D-19, E20 | `src/rai_bvar/api.py`; `src/rai_bvar/engines/mcmc/compose.py`; `tests/unit/test_api_resolution.py` (e.g. `test_resolution_table`, `test_bvar_with_sv_resolves_to_structural_mode_one`) | verified (tests committed) |
| 2 | One predictive contract for both engines: joint path draws plus per-draw conditional mean and Cholesky factor, so the log predictive density is Rao–Blackwellised | SPEC R3, §7; DESIGN §7.2–7.3, D-20 | `src/rai_bvar/forecast/result.py` (`log_pred_density`, line 209 ff.); `tests/contract/test_forecast_contract.py` (`test_simulation_consistency`, `test_log_pred_density_against_the_mixture`, …), `tests/contract/test_forecast_contract_mcmc.py` (`test_simulation_consistency`, `test_log_pred_density`, `test_cond_chol_is_a_inverse_times_scales`) (corrected by fact-check: `test_log_pred_density` and `test_cond_chol_…` exist only in the MCMC file) | verified; limitation recorded (NOTES 8b, §8.4 below) |
| 3 | One backtest engine and one verdict engine; a verdict needs exactly one differing axis, a ΔLPD larger than 3 MC s.e., a valid test or an MCS separation, and ≥ 95% converged origins | SPEC §9.1–9.5; DESIGN §11.4 (grid harmonisation, E22), §11.6 | `src/rai_bvar/evaluation/compare.py` (`grid` line 45, `compare` line 69, `CompareResult.verdict` line 196); `tests/integration/test_compare.py` (`test_two_axis_differences_are_not_comparable`, `test_expanding_window_tests_are_descriptive`, `test_identical_models_are_inconclusive`); `tests/unit/test_grid_harmonise.py` | verified as implemented; statistical size/power of the rule **not** validated (§8.2) |
| 4 | Validation is part of the library, not an afterthought: Geweke, SBC, block-level identities, analytic twins, parity, mutation | SPEC R6, §10; DESIGN §9 | `src/rai_bvar/validation/` (shipped in the wheel; DESIGN Q9 "yes"); `tests/geweke/`, `tests/sbc/`, `tests/mutation/`, `tests/analytic/`, `tests/parity/` | verified (tests committed); full nightly/weekly results to be re-run on the release commit (NOTES "Before submission") |
| 5 | Clean-room Apache-2.0 implementation; GPL R packages are run only as numerical oracles | SPEC §2.1, D1; DESIGN front matter "Clean room" | `LICENSE`; `pyproject.toml` (`license = "Apache-2.0"`); `tools/make_r_fixtures.R`; committed fixtures under `tests/parity/fixtures/` | licence verified; practice nuanced, see §6 |

### 2.1 What exists today (Phases 0–2) versus what the architecture is ready for

The headline "mean × volatility × prior" describes the **architecture**. The grid actually delivered in
Phases 0–2 is narrower, and the paper must say so plainly.

**Analytic engine** (NumPy/SciPy; SPEC §4.6; `src/rai_bvar/engines/analytic.py`): `Linear(prior=Minnesota(form="conjugate"))
× Constant(prior="iw")` with SOC/SUR dummies, the GLP hierarchy (`"fixed"`, `"eb"`, `"mh"`), Lenza–Primiceri COVID
scaling, recursive IRFs/FEVD. Phase 1 status "met" (SPEC.md line 672). Benchmarks `RandomWalk`, `OLSVAR`
(`src/rai_bvar/models/benchmarks.py`).

**MCMC engine** (JAX, hand-written Gibbs; DESIGN §3–§5). The registered validation compositions
(`src/rai_bvar/validation/joint.py` lines 454–480) enumerate what exists:

| Mean / prior form | Volatility | DESIGN §9.5 rows |
| :- | :- | :- |
| `Linear`, structural form, Mode I (moment-matched independent Minnesota) | `Constant("ig")`, `SV`, `SV(outliers=True)`, `SV(tails="t")`, `SV(outliers=True, tails="t")`, `SV(rw=True)`, `SV(sigma2_prior=InvGamma)`, `SV(h0=B₀)` | 1–4, 12, 14–16 |
| `Linear`, structural form, Mode II (λ-scaled NIG, the MCMC twin of the conjugate model) | `Constant("iw")` | 5 |
| `Linear`, structural form | `Constant("iw_equivalent")` (Chan 2022 `[CITE?]` Prop. 1) | 6 |
| `Linear`, reduced form (corrected triangular algorithm, CCCM 2022 `[CITE?]`) | `Constant("ig")`, `SV`, `SV(outliers)`, `SV(t)`, `SV(outliers, t)`, `Constant("iw_equivalent")` | 7–11, 13 |
| `ZeroMean` (univariate SV test vehicle) | `SV` | 17 |
| with SOC/SUR dummy rows (WP25) | five of the above | 201–205 |

Prior family in Phases 0–2: **Minnesota only** (conjugate, structural, reduced forms; `hierarchy="fixed"`/`"eb"`
on MCMC, `"mh"` analytic only; `tests/unit/test_api_resolution.py::test_metropolis_hastings_hierarchy_is_rejected_on_mcmc`,
`::test_lenza_primiceri_scaling_is_rejected_on_mcmc`). Global-local priors, `CommonSV`, BART, TVP,
threshold/Markov switching, factor SV and order-invariant SV are **roadmap** (SPEC §13 Phases 3–5; DESIGN §3.9
"plug-in walk-through", which claims they plug in "no edit to `sweep.py`, `runner.py` or `forecast.py`": a
design claim, **to check** when Phase 3 lands).

Suggested wording for the paper: *"ablation-ready" refers to an architecture in which a new mean, volatility or
prior component is evaluated against every existing one through one contract and one verdict rule; Phases 0–2
populate the volatility axis (five volatility families, two structural and one reduced-form treatment of the
linear mean) and leave the mean and prior axes to later releases.*

---

## 3. Who it is for

| Audience | What they get | Evidence |
| :- | :- | :- |
| Applied macro forecasters (central banks, policy institutions, academic forecasters) who must justify a richer model with out-of-sample density evidence | A verdict that refuses to credit a mean or prior change with gains that come from volatility (R2), with MC error, validity flags and MCS membership in the ablation table | SPEC §9.5; `evaluation/compare.py` |
| Methodologists adding a component (a prior, a volatility process, a mean) | A protocol (`MeanBlock`, `VolBlock`, `CoefPrior`, `GlobalStep`, covariance structure) and a reference registry that gives the new block Geweke/SBC configurations | DESIGN §3, §9.1 (`register_reference`) |
| Python users who today go to R (`BVAR`, `stochvol`, `bayesianVARs`) or MATLAB for BVARs with SV | Numerical parity with those oracles where a mapping exists, in a NumPy/JAX/xarray stack | `tests/parity/*`; §5 below |
| Replicators | Look-ahead-free backtests, pinned FRED vintages with checksums, spec hashes, deterministic PRNG streams | SPEC R9, §5.8, §8 "Randomness"; DESIGN §5.6; CR report line 20 (corrected by fact-check: was "lines 21–23") (R9 asserted bitwise at three origins) |

Target scale (SPEC §1.2, lines 41–48): analytic engine up to ~100+ variables, MCMC engine up to ~30 with design
target `n = 20`; `p` up to 13; `T` 100–800; monthly or quarterly. Measured: analytic `n = 100, p = 13, T = 300`
fit 0.1 s (`fixed`) to ~100 s (`eb` over λ, μ, δ) on a laptop CPU (SPEC.md lines 674–680; **recorded**);
MCMC BENCH-P2 below (§8.4) (corrected by fact-check: was "§7", which is the non-goals section).

Python ≥ 3.12 (`pyproject.toml` `requires-python`); CI runs the fast suite on Ubuntu for 3.12, 3.13, 3.14 and
weekly on macOS (`.github/workflows/ci.yml`).

---

## 4. The landscape

### 4.1 Python (as recorded in SPEC §1.4, "October 2026")

SPEC §1.4 (SPEC.md lines 58–73) characterises the ecosystem as **"fragmented, not empty"** and explicitly
rejects the "first Python BVAR" positioning. Every row below is the SPEC's description; none was re-verified for
this dossier. Status of every row: **to check** (PyPI release history, GitHub, licence files, current feature
lists), because the dates are recent and the descriptions are the SPEC author's.

| Project (SPEC §1.4) | Licence (per SPEC) | Coverage (per SPEC) | Gap relative to `rai-bvar` (per SPEC) |
| :- | :- | :- | :- |
| Bank of England `bvar` (PyPI; "first release Aug 2026") | MIT | conjugate Minnesota, SOC/SUR, GLP-style hyperparameter optimisation, COVID dummies, conditional forecasts | no MH over hyperparameters, no SV, no global-local priors, no nonlinear models |
| Joshua Chan's `bvar-toolkit-python` (GitHub, "started Oct 2026") | MIT | port of his MATLAB toolkit (SV, shrinkage priors, marginal likelihoods), "about a quarter of functions ported" | early stage; no BART; no backtest/verdict engine |
| `bvhar` Python port (GitHub only) | MIT | Minnesota, SSVS, HS, NG, DL with SV (C++ core) | no hierarchical GLP, no nonlinear models, not on PyPI |
| `pymc-extras` `BayesianVARMAX` | Apache-2.0 | state-space VARMAX in PyMC | hand-set priors; no Minnesota/GLP, no SV |
| `impulso`, `srvar-toolkit` | MIT | small, specialised | experimental / narrow |
| `statsmodels` | BSD-3 | frequentist VAR, SVAR, VECM, LP | no Bayesian VARs |

Independent confirmation available in the repo for one row only: the BoE `bvar` 0.3.5 source was read and run
(`tests/parity/BOE_BVAR_NOTES.md` header; `pyproject.toml` dev group `bvar>=0.3.5`), and its notes confirm "There
is no MH step (GLP's MCMC over h is absent)" (§6) and list conditional forecasts and GIRFs among its features
(§10). **recorded** for those two features (corrected by fact-check: was "verified"; the statements come from
reading the v0.3.5 source as written down in the note, not from a committed test, so under the NOTES legend they are
"recorded"; GIRFs are not part of the SPEC §1.4 row).

Python libraries the package builds on rather than competes with: JAX (MCMC), NumPy/SciPy (analytic), xarray
(containers), `scoringrules` (CRPS/energy/variogram scores and test oracle), pandas (`pyproject.toml`
`dependencies`; SPEC §11). SPEC §11 also argues against a PPL in core ("NUTS mixes worse than precision-sampler
Gibbs + ASIS on SV/TVP states"): that is a design assertion without a committed benchmark in this repo, **to check**
or soften in the paper.

### 4.2 R (the oracles and the closest feature competitors)

SPEC §2.2 (lines 87–101) lists the R references. All are GPL (GPL-3, GPL ≥ 2 or GPL ≥ 3, per SPEC §2.1; **to check**
each licence on CRAN), with one exception in SPEC §2.2's own table: the BART-VAR row lists `mpfarrho/mf-bavart` as
Apache-2.0 ("GPL-3 / Apache-2.0"), so "all GPL" holds for the CRAN packages, not for every row (corrected by
fact-check). The table separates what was actually used from what is listed:

| Package | Authors (per SPEC) | Role per SPEC | Role in Phases 0–2 as committed |
| :- | :- | :- | :- |
| `BVAR` | Kuschnig & Vashold (2021) `[CITE?]` | oracle: GLP hierarchy, SOC/SUR, MH over hyperparameters | **run**: BVAR 1.0.5 on R 4.1.2 (`tests/parity/BVAR_PARAMETRIZATION.md` "Versions"); fixtures `r_bvar_*.json`; tests `tests/parity/test_r_bvar.py` |
| `stochvol` | Hosszejni & Kastner `[CITE?]` | oracle: univariate SV (AWOL, ASIS, OCSN mixture, t errors) | **run**: stochvol **3.2.1** (not the 3.2.9 named in DESIGN §9.9; `tests/parity/SV_PARAMETRIZATION.md` "Versions"); fixtures `stochvol_*.nc`; `tests/parity/test_stochvol.py` |
| `bayesianVARs` | Gruber, Haan & Kastner `[CITE?]` | oracle: reduced-form CCCM sampler with SV; later global-local priors | **not run: parity blocked** (bayesianVARs 0.1.8 could not be built, no gfortran; `SV_PARAMETRIZATION.md` §4; `fixtures/probe_bayesianvars.nc` records `bayesianvars_available = 0`). Replaced by an in-house exact system-wide Gibbs oracle written in R (`bvars_swa_*.nc`, `SV_PARAMETRIZATION.md` §4–§6) |
| `factorstochvol` | Kastner `[CITE?]` | oracle for `FactorSV` | Phase 5; not used |
| `bvars` | Woźniak `[CITE?]` | oracle for `CommonSV` | Phase 3; not used |
| `bvarsv` | Krüger `[CITE?]` | oracle for TVP-VAR-SV | Phase 5; not used (read for constants/units in `docs/phase2/02_sv_sampler.md` §8.2) |
| `bvartools` | Mohr `[CITE?]` | "architectural reference" (modular Gibbs samplers) | **not an oracle and never run**: `tools/make_r_fixtures.R` loads only `jsonlite`, `BVAR`, `coda`, `stochvol` (lines 87–92) and probes `bayesianVARs` (line 2120). It was, however, opened to confirm the mixture constants (`docs/phase2/01_omori_mixture.md` lines 9–11 and source S9 "bvartools 0.3.0", line 56) (corrected by fact-check) |
| `bsvars` / `bsvarSIGNs` | Woźniak; Wang & Woźniak `[CITE?]` | Phase 6 identification | not used |
| BART-VAR code (`BAVART`, `mf-bavart`) | Huber & Rossini 2022 `[CITE?]` | Phase 4, forecast-level only ("not posterior oracles") | not used |

**Correction to the task framing.** The task listed "the R packages used as numerical oracles: BVAR, the BoE bvar,
stochvol, bayesianVARs, bvartools". As committed: (i) the BoE `bvar` is a **Python** (MIT) package, used as the
Python-side oracle in CI (SPEC §10.4; D5); (ii) `bvartools` is an architectural reference only and was never run;
(iii) `bayesianVARs` parity is blocked and the exit criterion "bayesianVARs parity for the reduced-form sampler"
(SPEC §13 Phase 2) is met **by substitution** with an in-house R oracle, not as written; (iv) the stochvol oracle is
version 3.2.1. The paper must state all four.

**Closest feature competitors** (to be argued in the paper, each **to check** against current versions):
`bayesianVARs` (priors × SV/factor structures behind one interface, per SPEC §2.2), `BVAR` (hierarchical
Minnesota; JSS paper), BEAR toolbox for MATLAB (Dieppe, Legrand & van Roye 2016 `[CITE?]`; broad BVAR menu with
forecast evaluation; not in the repo, my addition), Chan's MATLAB toolkit (SPEC §1.4 mentions its Python port).
None is described in the repo as enforcing one-axis comparisons or shipping a sampler-validation suite; the paper
may say "to our knowledge" only after the prior-art search (open question Q3).

### 4.3 The venue's own prior art (JSS)

Two JSS software papers are already cited in the repo and are the natural comparators for structure and claims:

- Kuschnig, N. & Vashold, L. (2021), "BVAR: Bayesian vector autoregressions with hierarchical prior selection in
  R", *JSS* 100(14) (SPEC §15) `[CITE?]`. Phase 1's parity target and replication target (SPEC §10.7).
- Hosszejni, D. & Kastner, G. (2021), "Modeling univariate and multivariate stochastic volatility in R with
  stochvol and factorstochvol", *JSS* 100(12) (`docs/phase2/02_sv_sampler.md` references, line 584) `[CITE?]`.
  Note 02 records that stochvol uses "Geweke variant in unit tests" (§2.3 of that paper): **the use of Geweke
  tests in SV software is therefore not new**, and the paper must not imply it is.

Other JSS/R-Journal BVAR-family papers I believe exist but have not verified: `BGVAR` (Böck, Feldkircher & Huber
2022) `[CITE?]`, `mfbvar` (Ankargren & Yang) `[CITE?]`, `bsvars` (Woźniak) `[CITE?]`, R `scoringRules` (Jordan,
Krüger & Lerch 2019) `[CITE?]`.

---

## 5. Reference packages as numerical oracles: what parity actually shows

| Oracle | What is compared | Result in the committed notes | Status |
| :- | :- | :- | :- |
| R `BVAR` 1.0.5 | log ML at fixed hyperparameters (rtol 1e-8; "BVAR and the exact values agree to < 1e-11"), NIW posterior parameters (1e-8), EB modes (1e-3), MH and forecast quantiles within `4·√(MCSE_R² + MCSE_ours²)` | `BVAR_PARAMETRIZATION.md` "Tolerances" (lines 295–301); fixture script runs 319 internal consistency checks (line 281 (corrected by fact-check: was "line 280")) | verified (committed fixtures + `test_r_bvar.py`) |
| BoE `bvar` 0.3.5 (Python, runs in CI) | log ML at fixed h (rtol 1e-8, "measured ≤ 1.5e-11"), posterior B̄, S̄, ν̄, E[Σ], Ω̄; hyperparameter modes within 1e-3 of the polished mode of BoE's objective; forecast quantiles | `BOE_BVAR_NOTES.md` §14 | verified (`test_boe_bvar.py`) |
| Phase 1 summary | "Log marginal likelihoods match R `BVAR` (exact cases) and BoE `bvar` to ~1e-11 relative; posterior moments to ~1e-9; EB modes to 1e-3; MH quantiles within Monte Carlo error" | SPEC.md line 672 | recorded (summary sentence; the per-test numbers are verified) |
| `stochvol` 3.2.1 | quantiles of μ, φ, σ, h_t (and τ_t for SV-t) within 4 quantile-MCSE | main/defaults on four datasets: PR max \|z\| 1.63–2.59, nightly 1.64–3.57 (bound 4.00); SV-t fixture fails (\|z\| 28.9 / 62.5), replaced by an exact importance-sampling oracle (max \|z\| 1.83) | `SV_PARAMETRIZATION.md` §6 table: recorded (reference laptop, M1 Pro); tests committed |
| in-house R system-wide Gibbs (`bvars_swa_*`) | reduced-form CCCM sampler vs an O(n⁶) joint oracle | max \|z\| 2.76 (`n3p2_const`), 3.45 (`n3p2_sv`), 3.56 (`n7p4_sv`, 1,540 comparisons, bound 4.51) | `SV_PARAMETRIZATION.md` §6: recorded; `test_bayesianvars.py` committed |
| Analytic twins and exact oracles (A1–A7; A1–A5 use the Phase 1 engine or closed forms, A6 toy quadrature, A7 an O(n⁶) joint-draw oracle) | MCMC Mode II × `Constant("iw")` vs the NIW posterior, quantiles within 4 quantile-MCSE (A2 `n = 3`, A3 `n = 20`); per-equation quadrature for Mode I × Constant (A5) | DESIGN §9.8; `tests/analytic/test_mcmc_twin.py`, `test_mode1_quadrature.py` | tests committed; nightly results **to check** on the release commit |

What parity does **not** establish, to be stated in the paper: (a) no oracle exists for the default structural
Mode I × SV composition at VAR level (stochvol parity is univariate via `ZeroMean`; the SWA oracle is reduced
form); its correctness rests on Geweke/SBC and block-level tests (DESIGN §9.3–9.6). (b) Exact-mode SV
(`exact=True`, the default, D-1) has no external oracle except the optional ex-post-reweighted stochvol runs
(`stochvol_i_exact`, `stochvol_iv_exact`, nightly only; `SV_PARAMETRIZATION.md` §6).

---

## 6. Licensing and the clean-room rule

### 6.1 The rule as specified

SPEC §2.1 (lines 77–85): every R reference package is GPL; translating their source would make `rai-bvar` a
derivative work; therefore algorithms are implemented "from the papers in §15, not translated from R/C++
source"; R packages are used as numerical oracles whose "outputs are data, not code"; ambiguities are resolved
and recorded in module docstrings; licence Apache-2.0 (decision D1, SPEC §14: patent grant, compatibility with
JAX/ArviZ/`scoringrules` (Apache-2.0) and `bartz` (MIT)). MIT/Apache sources (BoE `bvar`, `bartz`, `bvhar`,
Chan's toolkit, `mf-bavart`) may be reused with attribution.

### 6.2 The rule as practised (the paper must use exact wording)

The committed notes show three different practices (table completed by fact-check: the original listed only
the first, second and last groups of notes below; rows for notes 01, 03, 04, 07, 08 and the bayesianVARs source reading in
note 02 were added (corrected by fact-check)):

| Note | What it says about GPL source |
| :- | :- |
| `tests/parity/BVAR_PARAMETRIZATION.md` lines 3–13 | behaviour "established by running the package and reading its documentation and source (`R/10_bvar.R`, …, `54_predict.R`, and the JSS article)"; "It is not a description of code to be ported" |
| `docs/phase2/02_sv_sampler.md` line 6 (and §7, §8.2, refs lines 597–598, cross-check line 608) | "GPL sources (`stochvol` on GitHub, CRAN manuals) were read only to confirm published defaults and behaviour … No code was transcribed"; `bvarsv` "read for constants only" / "for behaviour only"; also the **bayesianVARs 0.1.8 source** (`R/utility_functions.R`, `src/bvar_cpp.cpp`) "read only for behaviour and constants" (line 597); the cross-check read stochvol 3.2.9, bayesianVARs 0.1.8 and bvarsv 1.1 "only for constants and behaviour" (line 608) (corrected by fact-check) |
| `docs/phase2/01_omori_mixture.md` lines 9–11 | "GPL packages (stochvol, bayesianVARs, factorstochvol, bvartools, bvarsv, bsvars) were opened only to confirm published constants and observable behaviour … No code was transcribed" (corrected by fact-check: row added) |
| `docs/phase2/03_sv_outliers_t.md` lines 7–9 | GPL "code was consulted only to confirm constants or behaviour, never transcribed": `stochvol` C++ "greps only", `svsim` "one grep"; also the CCMM MATLAB replication code, which has **no licence** ("Greps only") (corrected by fact-check: row added) |
| `docs/phase2/07_jax_patterns.md` line 736 (and line 764) | no GPL source opened by the first author; in the cross-check the stochvol header `sv_utils_latent_states.h` "was searched only for the 10 published mixture constants" (corrected by fact-check: row added) |
| `tests/parity/SV_PARAMETRIZATION.md` lines 9–11; `docs/phase2/05_structural_equation_sampler.md` line 4; `docs/phase2/04_cccm_corrected.md` line 5; `docs/phase2/08_common_sv_and_ordering.md` line 13 | "Their GPL sources were not read for this note"; "No GPL source was read"; "No GPL source code was read, by the author or by the cross-checker"; "No GPL source code was opened" (corrected by fact-check: notes 04 and 08 added) |

**Accurate statement for the paper:** *No code was translated from GPL sources. Algorithms were implemented from
the published papers. GPL packages were run as numerical oracles, and in a few documented cases (named in the
repository notes) their source was read to pin down an oracle's behaviour, defaults or constants, never to write
implementation code.* Do not call it "clean room" in the strict sense (implementers never see the source) without
this qualification. Open question Q1.

**Unsubstantiated process claim.** SPEC §2.1 says "The PR template includes a checkbox: 'No code in this PR was
translated from GPL sources.'" No PR template exists in the repository (`.github/` contains only
`workflows/ci.yml`; `find . -iname '*pull_request*'` outside `.venv` returns nothing). Either add it or drop the
claim. **to check / fix before submission.**

**Fixture licensing.** The committed fixtures (`tests/parity/fixtures/*.json`, `*.nc`) are outputs of GPL programs.
The SPEC's position is that outputs are data, not code (SPEC §2.1). The paper can state the policy; whether it
needs a legal citation (e.g. the FSF GPL FAQ on program output `[CITE?]`) is open question Q2.

**Data licensing.** FRED-MD/QD are downloaded on demand, never bundled, because some series are third-party
copyrighted; committed fixtures use only U.S. federal-agency series (SPEC §5.8, decision D4). Worth one sentence in
the paper (McCracken & Ng 2016, 2021 `[CITE?]`).

---

## 7. Non-goals (and what is deferred)

From SPEC §1.3 (lines 50–56), non-goals for v1.0:

- structural identification beyond recursive (Cholesky): sign, zero, narrative restrictions, proxy SVARs,
  identification through heteroskedasticity (Phase 6, possibly a separate package);
- mixed-frequency data, missing values inside the sample, ragged edges (SPEC §5.7: v1 raises on NaN);
- real-time vintages (the data layer must not preclude them: `data_at(origin)` hook, SPEC §5.5);
- panel, global and factor-augmented VARs;
- frequentist VAR estimation ("use `statsmodels`").

Deliberate methodological exclusions (worth stating as positions):

- **No PSIS-LOO or WAIC for VARs** (SPEC §9.4): "LOO conditions on the future to predict the past, and pointwise
  log-likelihoods conditional on latent volatilities produce unreliable Pareto k̂". PSIS-LFO-CV (Bürkner, Gabry &
  Vehtari 2020 `[CITE?]`) is Phase 4.
- **No KDE log scores** (SPEC §7, §9.2).
- **No PPL in core** (R7; SPEC §11).
- **No silent jitter** on failed Cholesky factorisations (SPEC §8; DESIGN D-18; NOTES 10).

Phase 2 restrictions on the MCMC engine (DESIGN D-21; `api.py` lines 23, 166–169): Lenza–Primiceri scaling and
`hierarchy="mh"` are rejected with actionable errors (Phase 3). SOC/SUR dummies were delivered by WP25 (DESIGN
D-21 update, §13 WP25; NOTES 12–16).

Roadmap only (not claimed in the paper): global-local priors, asymmetric conjugate prior as a user-facing prior,
`CommonSV`, conditional forecasts (Phase 3); BART mean, GIRFs, PSIS-LFO (Phase 4); TVP, threshold, Markov
switching, factor SV, order-invariant SV (Phase 5); structural identification (Phase 6) (SPEC §13).

---

## 8. What is genuinely new, what is not, and what must not be over-claimed

### 8.1 (a) Software-level contributions

1. **A factorial composition API with engine resolution and an exact analytic/MCMC bridge.** The same composition
   language covers the conjugate model (analytic) and every non-conjugate model (MCMC); the MCMC structural Mode II
   × `Constant("iw")` composition is constructed to equal the Phase 1 NIW posterior at `cross = κ_a = 1`
   (DESIGN D-10), and the
   analytic twin turns the analytic engine into an exact end-to-end oracle for the MCMC engine at `n = 20`
   (DESIGN §9.8 A3). Novelty argument: other packages offer several models; the claim here is a single grammar in
   which "differs along exactly one axis" is a checkable property (`spec_hash`, `axes`;
   `tests/unit/test_api_resolution.py::test_sv_arms_differ_from_constant_along_the_vol_axis_only`). Prior-art
   search needed (`bayesianVARs`, BEAR): **to check**.
2. **One predictive contract carrying per-draw conditional Gaussian moments.** `cond_mean` and `cond_chol` sit
   beside the joint paths (SPEC §7), for analytic and MCMC models alike, so the LPD is a mixture of Gaussians
   evaluated exactly given each draw and variable subsets are marginalised analytically
   (`forecast/result.py::log_pred_density`). Mixture-of-normals log scores are standard in the literature (e.g.
   Geweke & Amisano 2010 `[CITE?]`; the CR report's Gaussian-approximation column follows Adolfson, Lindé &
   Villani 2007 `[CITE?]`, report line 24 (corrected by fact-check: was "line 27")); the contribution is making the moments part of a tested contract
   (invariant 1, "simulation consistency": `test_simulation_consistency` in both contract test files, over every
   analytic model in `REGISTERED_MODELS` of `tests/contract/test_forecast_contract.py` and over the 12 MCMC
   composition kinds of `tests/mcmc/forecast/mcmc_fits.py::COMPOSITIONS` (structural SV, SV(outliers), SV(t),
   SV(outliers,t), SV(rw), Constant ig/iw/iw_equivalent; reduced SV, SV(outliers), Constant ig/iw_equivalent), two of
   them in the fast tier and ten marked `slow` (corrected by fact-check: was "tested for every registered composition";
   the 22 validation rows of `validation/joint.py` are not all covered — e.g. reduced × SV(t), reduced × SV(outliers, t),
   SV(sigma2_prior=InvGamma), SV(h0=B₀) and the SOC/SUR dummy rows have no contract test)). **verified** as
   implemented; see the limitation in §8.4.
3. **A verdict engine that encodes "volatility first".** One-axis pairing (`verdict(axis=...)` compares each model
   with the neighbour that takes the baseline's value on that axis), `grid()` harmonisation of `auto` components
   so a volatility grid is genuinely one-axis (DESIGN §11.4, E22), the MC-error gate (|ΔLPD| > 3 MC s.e.), the
   validity policy (DM/AG p-values flagged valid only for rolling windows, SPEC §9.3), MCS membership (Hansen,
   Lunde & Nason 2011 `[CITE?]`), and a convergence gate (≥ 95% of origins). Each ingredient is existing
   methodology; the combination as an automatic rule in software is the claim. **verified** as implemented
   (`evaluation/compare.py` lines 196–288); **not** validated statistically (§8.2).
4. **A sampler-validation suite shipped with the library**, organised in tiers U (identities), B (block SBC and
   invariance), G (Geweke), S (SBC with predictive quantities), M (mutation), A (analytic twins), P (parity)
   (DESIGN §9), with a two-implementation rule: prior samplers, simulators and `log_joint` are written by work
   packages that wrote no kernel and never import kernel modules (DESIGN §9.1, D-22, E9). The individual tests are
   established (Geweke 2004 `[CITE?]`; Talts et al. 2018 `[CITE?]`; Modrák et al. 2025 `[CITE?]`; Grosse & Duvenaud
   2014 `[CITE?]`; Gandy & Scott 2021 `[CITE?]`). The claim is the systematic application across 22 compositions
   (rows 1–17, 201–205: `test_g_full` in `tests/geweke/test_geweke_compositions.py` docstring; SBC parametrised over
   `COMPOSITIONS` at `tests/sbc/test_sbc_compositions.py` line 137) plus seven block configurations under Geweke
   (`test_g_small_block` over `BLOCK_CONFIGS`, nightly; the block-level SBC/invariance checks of tier B live in
   `tests/sbc/test_block_tier.py` with their own parametrisations, not over the seven `BLOCK_CONFIGS`) (corrected by
   fact-check: clarified that the seven block configurations are Geweke-tested, not SBC-tested), and the release
   rule that every injected bug must be caught by some tier.
5. **A mutation suite for MCMC kernels.** 18 mutants (M1–M18) injected through production seams
   (`tests/mutation/mutants.py` docstring), each with recorded catching tiers. Evidence: "all 33 mutant × tier
   checks caught, every control passed" for M1–M16 on M1 Pro and independently on x86-64 Linux (mutants.py lines
   827–833; NOTES 8, status verified), plus four checks for M17–M18 recorded on Apple M5 Max and reproduced on
   x86-64 (same lines; NOTES 15). Note the count: NOTES 8 says 33/33 (pre-WP25); with M17–M18 it is 37 checks over
   18 mutants. Whether mutation testing of Bayesian samplers has been published as a methodology (e.g. in the
   probabilistic-programming testing literature, Dutta et al. 2018 `[CITE?]`) needs a prior-art search: **to check**.
6. **Ordering sensitivity as a built-in diagnostic** (`rb.ordering_sensitivity`, `src/rai_bvar/structural/ordering.py`;
   SPEC R8, §6.6; DESIGN §13 WP22; `tests/integration/test_ordering.py`, including detection power on the CKY DGP,
   weekly). The problem is documented in the literature (ARRS 2023, CKY 2024 `[CITE?]`); the contribution is the
   operational check and the warning attached to every non-invariant MCMC fit with n ≥ 4 variables
   (`src/rai_bvar/engines/mcmc/monitor.py` line 77 `_ORDERING_N = 4`, lines 695–699; CR report lines 386, 391, 395,
   400) (corrected by fact-check: was "every non-invariant fit (CR report line 389)"; the warning is gated on n ≥ 4).
   **verified** (tests);
   CR all-24-orderings run **[CR@62d15f6, superseded]**.
7. **Reproducibility engineering for JAX MCMC.** Deterministic key derivation per (chain, sweep, block, equation)
   (DESIGN D-16, §5.6), bitwise invariance across chunking, burn-in piece size, fused/unfused plans and chain
   prefixes with `chains="map"` (corrected by fact-check: DESIGN §5.6 makes chain-count prefixes bitwise only with
   `chains="map"` and toy kernels; under the default `vmap` they agree "to rounding only"), with documented limits
   (vmap batch size, platform BLAS, `shard_map`: compared at rtol ≈ 1e-12)
   (NOTES 9, verified in CI on Linux); failures never raise inside JIT and are replayed deterministically on the
   host (D-18).

### 8.2 What the verdict engine does not yet establish

- **Size and power of the verdict rule are not validated in Phases 0–2.** SPEC §10.5 specifies DGP tests
  ("Size (false `better` rate on null DGPs) and power are reported in the documentation. This turns the
  'volatility first' rule from advice into a tested property of the library"), and DESIGN §9.10 lists "SPEC §10.5
  DGP rows relevant to Phase 2" in the weekly tier, but no such test exists: there is no `tests/dgp/`, and
  `grep -ril dgp tests` finds only the ordering DGP (`tests/integration/test_ordering.py::cky_dgp`) and SV parity
  data. SPEC §13 places "DGP verdict-engine tests on linear DGPs" in the Phase 3 exit. What exists: DM size under
  the null and power (`tests/unit/test_significance.py::test_dm_size_under_null`, `::test_dm_power_and_direction`),
  MCS behaviour (`::test_mcs_eliminates_clearly_worse_models_first`, `::test_mcs_keeps_equivalent_models`) and
  verdict logic (`tests/integration/test_compare.py`). The paper must present the verdict rule's error rates as
  future work, or the DGP tests must be implemented before submission (open question Q4).
- **The convergence gate dominated the CR verdicts.** With the gate on, `compare()` returned "better" for SV(rw)
  vs Constant at h = 1 in **0 of 24** orderings; with the gate off in 24 of 24; SV(rw) converged at 57–82% of
  origins per ordering against the 95% threshold (CR report lines 14, 339–362 (corrected by fact-check: was "lines 17,
  340–365")) **[CR@62d15f6, superseded]**. In the
  main runs every SV arm vs Constant is "inconclusive" with the gate on at every horizon (lines 146–165, 281–300
  (corrected by fact-check: was "lines 149–168, 284–303")),
  although the joint h = 1 ΔLPD is +0.809 nats/quarter with one-sided DM/HLN p = 0.000 (line 9 (corrected by
  fact-check: was "line 12")). This is the gate
  doing its job (refusing to certify under 37–68% converged origins), but it is also a usability finding the paper
  must report: the CR run's MCMC budget (4 chains × 5000 kept draws, burn 1000, report line 22 (corrected by
  fact-check: was "line 25")) does not meet the R-hat < 1.01 / bulk ESS > 400 targets (DESIGN §8.2) for σ of some
  equations (warnings, report lines 388, 389, 397, 398 (corrected by fact-check: was "lines 391–401"); in the example
  messages it is R-hat that fails, 1.012–1.020, while bulk ESS is 537–743, and one primary quantity fails per
  flagged origin).

### 8.3 (b) Methods findings worth a section (statuses from NOTES)

| Finding | NOTES item | Status | Evidence |
| :- | :- | :- | :- |
| Exact-mode SV is exact for any mixture table and any indicator law; table/pmf mutants (M11, M12) are bugs of the auxiliary-mixture sampler only; M7 is invisible with OCSN-10 and detectable with a coarse one-component table | 3 | verified | `tests/mutation/mutants.py` docstring, `EVIDENCE` |
| Parameter ranks are blind to the CCM-2019 triangular-step error at N = 1000; the log-likelihood rank catches it in block SBC (p 2.8e-11 vs control min p 0.10) | 4 | block-SBC catch verified; whole-scan "not caught" recorded | `mutants.py` M1 evidence; `docs/phase2/06_validation_geweke_sbc.md` §7.3 |
| Per-replication SBC ESS cannot be certified from L = 99 kept draws; pooled-ESS gate added | 5 | verified (design erratum) | DESIGN §9.6 erratum |
| SBC thinning cap `t_max = 200` needed on slow outlier/t datasets | 6 | recorded | `tests/sbc/test_sbc_compositions.py` docstring |
| Geweke "error 2" (simulator draws fresh latents) caught, max \|z\| 23.5 vs control 2.59 | 7 | verified | `mutants.py` M3 |
| Path-conditioned RB log score breaks down in the tails at h > 1 (UR h = 8 at 2008Q2: −93.6 / −46…−63 / −36 at 1000 / 8000 / 64,000 draws vs −19.1 exact; MCSE ≈ 1 throughout); fix = integrate the path out | 8b | **recorded** ("re-run with the library fix before publishing") | NOTES 8b; CR report "Reliability of the RB log score" (lines 26–30 (corrected by fact-check: was "lines 29–33")) |
| Ill-conditioned `a`-row precision (cond 1.3e16) in the prior's extreme tail; QR draw of the whitened rows instead of jitter or a narrower test prior | 10 | verified | `tests/mcmc/blocks/test_contemp_lstsq.py` |
| Dummy rows in Mode II must enter the λ step; closed-form λ marginal and an exact rejection envelope under `iw_equivalent` for the Geweke reference; the O(q²) eigen fast path survives dense prior precision | 12, 13, 14 | 12 recorded; 13, 14 verified | `tests/analytic/test_mcmc_twin.py::test_a1_dummies_*`; `validation/dummies.py`; `tests/validation/test_reference_dummies.py`; `test_dense_eigen_path_*` |
| RW AWOL draw at σ ≈ 5e-9: cancellation-free pivot recursion | 17 | verified (Apple M5 Max) | `tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma` |
| ASIS non-centred move combined with the exact mixture correction: the `w`-ratio correction is "derived here, not published" | — (DESIGN R1; N01 §10.3) | derivation recorded; tested by toy quadrature (A6), MH-ratio identity, mutant M7 | `docs/phase2/01_omori_mixture.md` §10.3, §14 item 4; DESIGN §12.1 R1. Novelty: prior-art search needed `[CITE?]` (Del Negro & Primiceri 2015 Alg. 3; Stroud, Müller & Polson 2003; Kastner & Frühwirth-Schnatter 2014) |
| Standardized-t SV with a collapsed exact discrete Gibbs step for ν on a grid | — (DESIGN D-8; N03) | design; the standardized likelihood for ν is "derived here" | `docs/phase2/03_sv_outliers_t.md` line 209 |

These are candidates for the methods sections of other dossiers; for positioning they support the claim that the
validation suite finds non-obvious errors, not only that it exists.

### 8.4 Limitations that belong in the positioning section

- **The SPEC §7 LPD estimator conditions on the simulated path for h > 1** and is unreliable in the tails (NOTES 8b,
  recorded). In the CR report, cells with "RB degen." origins have inflated magnitudes, and RB and Gaussian-approx
  ΔLPS differ in sign in 28 listed cells (report line 30 (corrected by fact-check: was "line 33")) **[CR@62d15f6, superseded]**. The paper should either ship
  the path-integrated estimator and present this as a methods contribution, or restrict LPD claims to h = 1.
- **Phase 2 exit criteria are met with documented substitutions and failures:** bayesianVARs parity by an in-house
  oracle (§4.2); the stochvol SV-t row by an exact importance-sampling oracle (`SV_PARAMETRIZATION.md` §3.2, §6); CR
  criteria 4 and 6 "not met" (report lines 12, 14 (corrected by fact-check: was "lines 15, 17")), which WP23's acceptance allows if documented (DESIGN §13 WP23).
  **[CR@62d15f6, superseded]**.
- **Performance evidence is from one machine under load.** BENCH-P2 (`n = 20, p = 4, T = 300`, Linear × SV, 4 ×
  10,000 draws, 1 device): wall 52.3 s cold, 49.6 s warm, compile 3.46 s cold / 1.00 s warm, 3.90 ms per sweep
  (min of 3 × 1000), peak RSS 3.0 GB, against the 120 s target; Apple M5 Max, load average 6.3 / 4.7 / 3.6 at the
  start; `converged` false in all six fits (3 repeats × cold and warm; `summary.repeats` = 3) (corrected by fact-check: was "all six repeats"); primary-quantity ESS minimum 504 (σ) (`tests/bench/baseline.json`
  `summary`, `machine`; measured at code commit `deb3929`, baseline committed in `d490730`). **recorded.** DESIGN §10.1 and WP24 promise an x86 report; none is in
  `baseline.json`: **to check**.
- **Release state.** `pyproject.toml`: version 0.0.1, classifier "Development Status :: 2 - Pre-Alpha"; README's
  status banner still says "Status: Phase 1 (analytic engine) … MCMC models with stochastic volatility arrive in
  Phase 2" (README line 5), which is stale at `62d15f6`. No evidence of a PyPI release in the repo (open question Q5).

### 8.5 (c) Not new: methods the package implements and must attribute

GLP hierarchy (Giannone, Lenza & Primiceri 2015); Lenza–Primiceri COVID scaling (2022); conjugate NIW and
Kadiyala–Karlsson (1997); SOC/SUR dummies (Sims–Zha / Doan–Litterman–Sims lineage, via GLP); OCSN 10-component and
KSC 7-component mixtures (Omori et al. 2007; Kim, Shephard & Chib 1998); AWOL / precision samplers (Chan & Jeliazkov
2009; McCausland, Miller & Pelletier 2011); ASIS (Kastner & Frühwirth-Schnatter 2014); corrected step order (Del
Negro & Primiceri 2015); corrected triangular algorithm (Carriero, Chan, Clark & Marcellino 2022); IW-equivalent
structural prior and asymmetric conjugate prior (Chan 2022); outlier SV (Carriero, Clark, Marcellino & Mertens
2024); Student-t SV as a scale mixture (Jacquier, Polson & Rossi 2004); Geweke tests (2004); SBC (Talts et al. 2018;
Modrák et al. 2025); rank-normalised R-hat and ESS (Vehtari et al. 2021); DM test with HLN correction (Diebold &
Mariano 1995; Harvey, Leybourne & Newbold 1997); GW framework (Giacomini & White 2006); AG test (Amisano & Giacomini
2007); MCS (Hansen, Lunde & Nason 2011); proper scoring rules, qw/tw-CRPS, variogram score (Gneiting & Raftery 2007;
Gneiting & Ranjan 2011; Scheuerer & Hamill 2015). All `[CITE?]`; most are listed in SPEC §15.

### 8.6 Findings about other software (handle with care)

Recorded in committed notes; all version-pinned; NOTES §A requires re-testing on current versions and contacting
maintainers before any public claim.

| Package / version | Finding | Where | Status |
| :- | :- | :- | :- |
| stochvol 3.2.1 | ASIS non-centred step takes \|σ\| without flipping h̃; on a four-observation problem its h₂ quantiles deviate from an exact IS posterior; centred-only stochvol and `rai-bvar` match | NOTES 1; `SV_PARAMETRIZATION.md` §3.1 | recorded; to check on current stochvol |
| stochvol 3.2.1 | Student-t SV sampler misses the standardized-t posterior (μ and τ quantiles), with or without ASIS | NOTES 2; `SV_PARAMETRIZATION.md` §3.2 | recorded; "Report upstream first" |
| BoE `bvar` 0.3.5 | SOC/SUR initial mean ȳ₀ uses observations p+1…2p (GLP: first p); moves log p(Y\|h) by 0.1–1.1 nats on QD levels | `BOE_BVAR_NOTES.md` §4 | recorded (not in NOTES ledger) |
| BoE `bvar` 0.3.5 | NumPy optimiser with forward-difference gradients is inaccurate on level data (relative errors 2.8e-3 to 2.6e-2 vs the polished mode of its own objective) | `BOE_BVAR_NOTES.md` §6 | recorded (not in NOTES ledger) |
| BoE `bvar` 0.3.5 | six documentation-vs-code items: five discrepancies and one omission (the docs do not state the defaults `lambda_constant = 10`, `c3_sd = 0.5`) (corrected by fact-check: was "six documentation-vs-code discrepancies") | `BOE_BVAR_NOTES.md` §11 | recorded (not in NOTES ledger) |
| R `BVAR` 1.0.5 | `psi = "auto"` returns a standard deviation used where a variance belongs | `BVAR_PARAMETRIZATION.md` §3 | recorded (not in NOTES ledger) |

Recommendation for the paper: present these as illustrations that oracle-based testing cuts both ways (it validates
the new package and surfaces mapping conventions and deviations in the oracles), keep the tone factual, cite
versions, and add the BoE/BVAR items to `NOTES.md` §A if they are to appear.

---

## 9. Candidate titles

JSS titles usually follow "*pkg*: *What it does* in *Language*" (e.g. Kuschnig & Vashold 2021 `[CITE?]`).

1. **rai-bvar: Ablation-Ready Bayesian Vector Autoregressions in Python**
2. **rai-bvar: Composable Bayesian VARs with Stochastic Volatility and a Fair Out-of-Sample Verdict Engine in Python**
3. **Volatility First: Fair Ablation of Bayesian Vector Autoregressions with the Python Package rai-bvar**
4. **One Contract, One Verdict: Composable and Validated Bayesian VARs in Python and JAX**
5. **Does Complexity Pay? Testing Bayesian VAR Specifications Against Stochastic-Volatility Baselines with rai-bvar**

My ranking for JSS: 1 (house style, accurate for the architecture), 2 (more descriptive of Phase 2 content), 4 (if
the validation suite is to be foregrounded). Title 5 promises an answer the verdict engine has not yet been
validated to give (§8.2) and would need the DGP size/power study.

---

## 10. Abstract draft (≈ 200 words)

> Whether a richer Bayesian vector autoregression (BVAR) forecasts better is often decided without holding the
> volatility specification fixed, and the software makes fair comparisons laborious: each model class has its own
> estimator, output format and evaluation code. We present **rai-bvar**, an Apache-2.0 Python package in which a
> BVAR is a composition of a mean, a volatility and a prior component. Conjugate Minnesota models, with the
> Giannone–Lenza–Primiceri hierarchy and Lenza–Primiceri COVID scaling, are estimated by an analytic engine;
> compositions with stochastic volatility, outliers or Student-t errors by hand-written JAX Gibbs samplers in
> recursive structural or corrected reduced form. Both engines return one predictive contract, joint path draws
> with per-draw conditional Gaussian moments, so log predictive densities are Rao–Blackwellised rather than
> kernel-smoothed. One backtest engine scores every model, and a verdict engine declares a model better only if it
> differs from its reference along exactly one axis, its log-score gain exceeds three Monte Carlo standard errors, a
> valid test or the model confidence set separates them, and its chains converged. Samplers are checked by Geweke
> tests, simulation-based calibration, analytic twins, parity with R packages, and a mutation suite in
> which every injected bug must be caught; the analytic engine also matches R and Python packages. A replication of Clark and Ravazzolo (2015) illustrates the workflow.

(corrected by fact-check: the abstract said the *samplers* are checked by "parity with R and Python packages"; the only Python oracle, the BoE `bvar`, covers the analytic conjugate engine (`BOE_BVAR_NOTES.md` §14), while the MCMC samplers are compared with R `stochvol` and the in-house R system-wide Gibbs oracle (`SV_PARAMETRIZATION.md` §6); reworded. 213 words by `wc -w` after the change, was 206.) Every claim maps to §2 and §8.1. Deliberately absent: CR numbers (superseded), the 52 s benchmark (one
loaded machine, recorded), "first" or "only" (prior-art search pending), "size and power" (not validated). Optional
additions once verified on the release commit: "log marginal likelihoods agree with R `BVAR` and the Bank of
England's `bvar` to about 1e-11"; "a 20-variable SV-BVAR runs 4 × 10,000 draws in under a minute on a laptop".

---

## 11. Claimed contributions with evidence

| # | Contribution (paper wording) | Evidence pointer | Status |
| :- | :- | :- | :- |
| C1 | A composition grammar (mean × volatility × prior) with engine resolution; invalid combinations fail at construction with actionable errors | SPEC R1, §4.5; DESIGN §3.8, §11.1; `src/rai_bvar/api.py`; `tests/unit/test_api_resolution.py` | verified |
| C2 | Phase 2 populates the volatility axis: five volatility families (Constant with IG / IW / IW-equivalent priors; SV with AR(1) or RW log-variance; SV with outliers; SV with Student-t; both) across structural Mode I / Mode II and reduced (CCCM 2022) linear means, plus SOC/SUR dummies on MCMC | `src/rai_bvar/validation/joint.py` lines 454–480; DESIGN §9.5 table; NOTES 12–16 | verified (registered and tested compositions) |
| C3 | One predictive contract for analytic and MCMC models with per-draw conditional moments; Rao–Blackwellised LPD with analytic variable marginalisation; simulation-consistency invariant tested for every registered analytic model and the 12 MCMC composition kinds of `tests/mcmc/forecast/mcmc_fits.py::COMPOSITIONS` (corrected by fact-check: was "for every composition"; see §8.1 item 2) | SPEC §7; DESIGN §7.2–7.3; `forecast/result.py`; `tests/contract/test_forecast_contract*.py` | verified; tail limitation for h > 1 recorded (NOTES 8b) |
| C4 | A backtest engine with look-ahead-free refits (R9), caching by spec/data/origin/seed, per-origin fit seeds, opt-in warm starts | SPEC §9.1, R9; DESIGN §11.6, D-32; `evaluation/backtest.py`; `tests/integration/test_backtest*.py`; CR report line 20 (corrected by fact-check: was "line 23") (R9 bitwise at three origins) | verified (tests); CR assertion recorded |
| C5 | A verdict engine implementing "volatility first": one-axis pairing, grid harmonisation, MC-error gate, validity policy, MCS, convergence gate | SPEC R2, §9.3, §9.5; DESIGN §11.4; `evaluation/compare.py`; `tests/integration/test_compare.py` | verified as implemented; size/power **not validated** (SPEC §10.5 unimplemented) |
| C6 | Sampler validation shipped as library code with a two-implementation rule: 22 composition rows under Geweke and SBC and 7 block configurations under Geweke (block-tier SBC/invariance in `tests/sbc/test_block_tier.py`) (corrected by fact-check: was "22 composition rows and 7 block configurations under Geweke and SBC"), block identities, analytic twins | DESIGN §9.1–9.8; `src/rai_bvar/validation/`; `tests/geweke/`, `tests/sbc/`, `tests/analytic/` | tests verified to exist; pass status on the release commit **to check** |
| C7 | A mutation suite: 18 injected sampler bugs, each caught by at least one tier with passing controls, on two platforms | `tests/mutation/mutants.py` (docstring; lines 827–858); NOTES 8, 15 | verified (33 checks M1–M16, x86 re-run); M17–M18 4 checks verified on two platforms per NOTES 15 |
| C8 | Numerical parity with R `BVAR` 1.0.5 and the BoE `bvar` 0.3.5 (log ML ≈ 1e-11 relative), stochvol 3.2.1 (within 4 quantile-MCSE on four datasets) and an exact system-wide Gibbs oracle for the reduced form | `tests/parity/*.md` and tests; SPEC.md line 672 | verified (fixtures + tests); numbers recorded |
| C9 | Ordering-sensitivity diagnostic and ordering warnings for every non-invariant MCMC composition with n ≥ 4 (corrected by fact-check: warning gated on `_ORDERING_N = 4`, `engines/mcmc/monitor.py` lines 77, 695–699) | `structural/ordering.py`; `tests/integration/test_ordering.py`; CR report lines 333–366 (corrected by fact-check: was "lines 336–369") | verified (tests); CR results superseded |
| C10 | Methods findings from validation (NOTES 3, 4, 5, 7, 13, 14, 15) and numerics (NOTES 10, 17) | NOTES items and their pointers | per item (verified / recorded, §8.3) |
| C11 | A path-integrated Rao–Blackwellised log score for h > 1 that fixes the tail breakdown of the path-conditioned estimator | NOTES 8b; CR report lines 26–30 (corrected by fact-check: was "lines 29–33") | **recorded; library fix not committed at `62d15f6`**; include only once it lands |
| C12 | Reproducible JAX MCMC: deterministic keys, bitwise invariances with documented limits, deterministic failure replay, no silent jitter | DESIGN D-16, D-18, §5.6; NOTES 9; SPEC §8 | verified (CI, Linux) |
| C13 | Performance: BENCH-P2 in 52 s (target 120 s) on one CPU device | `tests/bench/baseline.json` | recorded (one machine, under load; no x86 entry; `converged` false) |
| C14 | Clean-room, permissively licensed implementation with GPL oracles used only as data | SPEC §2.1, D1; `LICENSE`; `tools/make_r_fixtures.R` | licence verified; wording per §6.2; PR-template claim unsubstantiated |
| C15 | Replication of Clark & Ravazzolo (2015) on FRED-QD with all 24 orderings | `benchmarks/clark_ravazzolo.py`; `benchmarks/reports/clark_ravazzolo.md` | **[CR@62d15f6, superseded]**: criteria 1, 2, 3, 5 met; 4, 6 not met (lines 9–14 (corrected by fact-check: was "lines 12–17")) |

---

## 12. Open questions

- **Q1 (clean room).** Adopt the wording of §6.2 ("no code translated; source read for behaviour in named notes")?
  Or re-establish oracle behaviour without reading GPL source (as SV_PARAMETRIZATION.md did) for BVAR and stochvol
  before submission?
- **Q2 (fixtures).** Do the committed GPL-oracle fixtures need a licensing statement in the repo and the paper?
- **Q3 (prior art).** A systematic search is needed before any "to our knowledge, first" claim: Python and R BVAR
  packages as of submission (verify every row of SPEC §1.4), MATLAB toolboxes (BEAR, Chan's toolkit, Dynare's BVAR),
  automated specification-comparison studies (e.g. Carriero, Clark & Marcellino 2015 on BVAR specification choices
  `[CITE?]`), mutation testing of MCMC samplers, the ASIS + exact-correction `w` ratio.
- **Q4 (verdict validation).** Implement SPEC §10.5's DGP size/power study for the Phase 2 rows (homoskedastic, SV,
  SV + outliers) before submission, or present the verdict rule's error rates as future work?
- **Q5 (release).** JSS reviews released, installable software with replication material `[CITE? JSS author
  guidelines]`. Plan: PyPI release, version number, updated README, a replication script per figure/table, archived
  fixtures and pinned FRED vintages (`rb.datasets.load("fred-qd", vintage="2026-09")`, SHA-256 checked, SPEC §5.8)?
- **Q6 (CR as the illustration).** Which run is the paper's illustration: the acceptance run in progress
  (replacing `clark_ravazzolo.md`)? Its convergence-gate outcome decides whether the illustration shows a "better"
  verdict or an "inconclusive" one, and the paper's narrative of the gate (feature or usability problem) depends on
  it.
- **Q7 (MCMC budget defaults).** Should default `MCMCSettings` for backtests be raised, given 37–68% converged origins
  in CR and `converged = false` in all six BENCH-P2 fits (3 cold, 3 warm; corrected by fact-check: was "all BENCH-P2 repeats"), or should the convergence gate be documented as strict by
  design?
- **Q8 (LPD estimator).** Ship the path-integrated RB estimator (NOTES 8b) as the default for h > 1 before the paper,
  and change SPEC §7 accordingly?
- **Q9 (third-party findings).** Re-test the stochvol findings on the current stochvol release and contact the
  maintainers (NOTES §A, "Before submission"); decide whether the BoE `bvar` and R `BVAR` convention findings belong
  in the paper and in NOTES §A.
- **Q10 (AI assistance).** NOTES "Before submission" requires an AI-assistance statement. The repository shows a
  multi-agent process (DESIGN front matter: three competing proposals ranked by two "judges"; work packages with
  separate owners; "independent x86 review") and 49 commits over three days. Who are the authors, and how is the
  process disclosed?
- **Q11 (scope of "ablation-ready").** Is it acceptable to submit while the mean and prior axes contain only the
  Minnesota/linear family, or wait for Phase 3 (global-local priors) so the paper shows a two-axis ablation?
- **Q12 (x86 benchmark).** Add the promised x86 BENCH-P2 entry (DESIGN §10.1, WP24) and a run on an idle machine.
- **Q13 (stochvol version).** DESIGN §9.9 names stochvol 3.2.9; the fixtures are 3.2.1. Regenerate (the unblock path
  in `SV_PARAMETRIZATION.md` §3.2) or correct DESIGN?
- **Q14 (PPL claim).** SPEC §11 says NUTS mixes worse than precision-sampler Gibbs + ASIS on SV states; no committed
  benchmark supports this. Cite literature `[CITE?]`, add a benchmark, or drop the claim.

---

## 13. References to verify (`[CITE?]`)

Format: author(s), year, title (as best known), and where the repo mentions it. All must be verified by the
bibliography stage; items marked "(not in repo)" are my additions.

**Software and JSS comparators**
- Kuschnig, N. & Vashold, L. (2021). BVAR: Bayesian vector autoregressions with hierarchical prior selection in R. *JSS* 100(14). (SPEC §15)
- Hosszejni, D. & Kastner, G. (2021). Modeling univariate and multivariate stochastic volatility in R with stochvol and factorstochvol. *JSS* 100(12). (`docs/phase2/02_sv_sampler.md` refs)
- Gruber, L., Haan, P. & Kastner, G. — `bayesianVARs` R package (CRAN). (SPEC §2.2)
- Kastner, G. — `factorstochvol` R package; Kastner (2019) Sparse Bayesian time-varying covariance estimation in many dimensions. *J. Econometrics* 210(1). (SPEC §2.2, §15)
- Woźniak, T. — `bvars`, `bsvars` R packages; Wang, X. & Woźniak, T. — `bsvarSIGNs`. (SPEC §2.2)
- Krüger, F. — `bvarsv` R package. (SPEC §2.2)
- Mohr, F. X. — `bvartools` R package. (SPEC §2.2)
- Bank of England — `bvar` Python package (PyPI, v0.3.5; github.com/bank-of-england/bvar). (SPEC §1.4; BOE notes)
- Chan, J. C. C. — BVAR toolkit (MATLAB) and `bvar-toolkit-python`. (SPEC §1.4)
- Kim, Y. G. — `bvhar` (R) and its Python port. (SPEC §1.4)
- `pymc-extras` `BayesianVARMAX`; `impulso`; `srvar-toolkit`. (SPEC §1.4)
- Seabold, S. & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. (not in repo)
- Dieppe, A., Legrand, R. & van Roye, B. (2016). The BEAR toolbox. ECB Working Paper 1934. (not in repo)
- Böck, M., Feldkircher, M. & Huber, F. (2022). BGVAR: Bayesian global vector autoregressions with shrinkage priors in R. *JSS* 104(9). (not in repo)
- Ankargren, S. & Yang, Y. — `mfbvar`: mixed-frequency Bayesian VAR models in R. (not in repo)
- Jordan, A., Krüger, F. & Lerch, S. (2019). Evaluating probabilistic forecasts with scoringRules. *JSS* 90(12). (not in repo)
- `scoringrules` (Python; Zanetta & Allen?) — authors and reference to verify. (SPEC §9.2, §11)
- Bradbury, J. et al. (2018). JAX: composable transformations of Python+NumPy programs. (not in repo)
- Hoyer, S. & Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *J. Open Research Software*. (not in repo)
- Kumar, R. et al. (2019). ArviZ: a unified library for exploratory analysis of Bayesian models in Python. *JOSS*. (SPEC §11 mentions ArviZ)
- Abril-Pla, O. et al. (2023). PyMC: a modern and comprehensive probabilistic programming framework in Python. (SPEC §11 mentions PyMC)
- Phan, D., Pradhan, N. & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. (SPEC §11 mentions NumPyro)
- Free Software Foundation. GPL FAQ, on the licence status of a program's output. (not in repo)
- Journal of Statistical Software, author guidelines. (not in repo)

**BVAR, priors and volatility**
- Giannone, D., Lenza, M. & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *REStat* 97(2). (SPEC §15)
- Lenza, M. & Primiceri, G. E. (2022). How to estimate a vector autoregression after March 2020. *JAE* 37(4). (SPEC §15)
- Kadiyala, K. R. & Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR-models. *JAE* 12(2). (SPEC §15)
- Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: five years of experience. *JBES*. (not in repo)
- Doan, T., Litterman, R. & Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*. (not in repo)
- Sims, C. A. & Zha, T. (1998). Bayesian methods for dynamic multivariate models. *IER*. (not in repo)
- Sims, C. A. & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *AER* 96(1). (SPEC R2, §15)
- Clark, T. E. & Ravazzolo, F. (2015). Macroeconomic forecasting performance under alternative specifications of time-varying volatility. *JAE* 30(4). (SPEC §15)
- Carriero, A., Clark, T. E. & Marcellino, M. (2015). Bayesian VARs: specification choices and forecast accuracy. *JAE*. (not in repo)
- Carriero, A., Clark, T. E. & Marcellino, M. (2016). Common drifting volatility in large Bayesian VARs. *JBES* 34(3). (SPEC §15)
- Carriero, A., Clark, T. E. & Marcellino, M. (2019). Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors. *J. Econometrics* 212(1). (SPEC §15)
- Bognanni, M. (2022). Comment on "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *J. Econometrics* 227(2). (SPEC §15)
- Carriero, A., Chan, J. C. C., Clark, T. E. & Marcellino, M. (2022). Corrigendum to "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *J. Econometrics* 227(2). (SPEC §15)
- Carriero, A., Clark, T. E., Marcellino, M. & Mertens, E. (2024). Addressing COVID-19 outliers in BVARs with stochastic volatility. *REStat* 106(5). (SPEC §15)
- Chan, J. C. C. (2022). Asymmetric conjugate priors for large Bayesian VARs. *Quantitative Economics* 13(3). (SPEC §15)
- Chan, J. C. C. (2023). BVARs and stochastic volatility (survey; arXiv:2310.14438). (`docs/phase2/08_common_sv_and_ordering.md` §B1)
- Arias, J. E., Rubio-Ramírez, J. F. & Shin, M. (2023). Macroeconomic forecasting and variable ordering in multivariate stochastic volatility models. *J. Econometrics* 235(2). (SPEC §15)
- Chan, J. C. C., Koop, G. & Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic volatility. *JBES* 42(2). (SPEC §15)
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *REStud* 72(3). (SPEC §15)
- Del Negro, M. & Primiceri, G. E. (2015). … A corrigendum. *REStud* 82(4). (SPEC §15)
- Cogley, T. & Sargent, T. J. (2005). Drifts and volatilities: monetary policies and outcomes in the post WWII US. *RED*. (not in repo; note 08 names "Cogley–Sargent")
- Kim, S., Shephard, N. & Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *REStud* 65(3). (SPEC §15)
- Omori, Y., Chib, S., Shephard, N. & Nakajima, J. (2007). Stochastic volatility with leverage: fast and efficient likelihood inference. *J. Econometrics* 140(2). (SPEC §15)
- Kastner, G. & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *CSDA* 76. (SPEC §15)
- Chan, J. C. C. & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *IJMMNO* 1(1/2). (SPEC §15)
- McCausland, W. J., Miller, S. & Pelletier, D. (2011). Simulation smoothing for state-space models: a computational efficiency analysis. *CSDA* 55(1). (SPEC §15)
- Stroud, J. R., Müller, P. & Polson, N. G. (2003). Nonlinear state-space models with state-dependent variances. *JASA*. (DESIGN D-1 cites "Stroud et al. 2003")
- Jacquier, E., Polson, N. G. & Rossi, P. E. (2004). Bayesian analysis of stochastic volatility models with fat-tails and correlated errors. *J. Econometrics*. (note 08 "JPR 2004")
- Huber, F. & Rossini, L. (2022). Inference in Bayesian additive vector autoregressive tree models. *AoAS* 16(1). (SPEC §15)
- Clark, T. E., Huber, F., Koop, G., Marcellino, M. & Pfarrhofer, M. (2023). Tail forecasting with multivariate Bayesian additive regression trees. *IER* 64(3). (SPEC §15)
- Gruber, L. & Kastner, G. (2025). Forecasting macroeconomic data with Bayesian VARs: sparse or dense? It depends! *IJF* 41(4). (SPEC §15)
- Bertsche, D. & Braun, R. (2020). Identification of structural vector autoregressions by stochastic volatility. (note 08 §B2)

**Forecast evaluation**
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3). (SPEC §15)
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF* 13(2). (SPEC §15)
- Clark, T. E. & McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. *J. Econometrics* 105(1). (SPEC §15)
- Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6). (SPEC §15)
- Amisano, G. & Giacomini, R. (2007). Comparing density forecasts via weighted likelihood ratio tests. *JBES* 25(2). (SPEC §15)
- Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2). (SPEC §15)
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477). (SPEC §15)
- Gneiting, T. & Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES* 29(3). (SPEC §15)
- Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *MWR* 143(4). (SPEC §15)
- Geweke, J. & Amisano, G. (2010). Comparing and evaluating Bayesian predictive distributions of asset returns. *IJF*. (not in repo)
- Adolfson, M., Lindé, J. & Villani, M. (2007). Forecasting performance of an open economy DSGE model. *Econometric Reviews*. (CR report line 24 "Adolfson et al. 2007"; corrected by fact-check: was "line 27")
- Bürkner, P.-C., Gabry, J. & Vehtari, A. (2020). Approximate leave-future-out cross-validation for Bayesian time series models. *JSCS* 90(14). (SPEC §15)

**Validation of samplers**
- Geweke, J. (2004). Getting it right: joint distribution tests of posterior simulators. *JASA* 99(467). (SPEC §15)
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788. (SPEC §15)
- Modrák, M. et al. (2025). Simulation-based calibration checking for Bayesian computation: the choice of test quantities shapes sensitivity. *Bayesian Analysis* 20(2). (SPEC §15)
- Säilynoja, T., Bürkner, P.-C. & Vehtari, A. (2022). Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. *Statistics and Computing*. (DESIGN §9.6 "Säilynoja recursion")
- Grosse, R. B. & Duvenaud, D. K. (2014). Testing MCMC code. (DESIGN §9.3 "Grosse & Duvenaud eq. 1")
- Gandy, A. & Scott, J. (2021). Unit testing for MCMC and other Monte Carlo methods. arXiv:2001.06465. (note 06 §4.6)
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis* 16(2). (SPEC §15)
- Dutta, S., Legunsen, O., Huang, Z. & Misailovic, S. (2018). Testing probabilistic programming systems. ESEC/FSE. (not in repo; for the mutation-testing prior-art search)

**Data**
- McCracken, M. W. & Ng, S. (2016). FRED-MD: a monthly database for macroeconomic research. *JBES* 34(4). (SPEC §15)
- McCracken, M. W. & Ng, S. (2021). FRED-QD: a quarterly database for macroeconomic research. *FRB St. Louis Review* 103(1). (SPEC §15)

---

## Unsupported claims (fact-check)

Added by the fact-check pass (2026-10-23) against the repository at `62d15f6`. Each item below is a statement
made in this dossier, or repeated from SPEC/DESIGN, for which no committed test, fixture, report or command output
supports it. They must not appear in the paper as facts unless evidence is added; most are already flagged
**to check** in the body and are collected here so the drafting stage cannot miss them.

1. **PR-template checkbox** (SPEC §2.1 line 85, §14 line 735 "PR checkbox"): no PR template exists (`.github/`
   holds only `workflows/ci.yml`; `find . -path ./.venv -prune -o -iname '*pull_request*' -print` returns nothing).
   §6.2 and C14 depend on it only as a process claim.
2. **"NUTS mixes worse than precision-sampler Gibbs + ASIS on SV/TVP states"** (SPEC §11 line 571; §4.1, Q14):
   no benchmark, test or note in `benchmarks/`, `tests/`, `src/` or `docs/phase2/` compares against NUTS
   (`grep -rni '\bnuts\b'` over those paths finds nothing).
3. **Phase 3–5 components plug in with "no edit to `sweep.py`, `runner.py` or `forecast.py`"** (DESIGN §3.9
   heading; §2.1): a design intention; no Phase 3–5 component exists to test it.
4. **The SPEC §1.4 Python landscape rows** (§4.1 table): licences, release dates ("first release Aug 2026",
   "started Oct 2026"), coverage and gaps of `bvar-toolkit-python`, `bvhar`, `pymc-extras`, `impulso`,
   `srvar-toolkit`, `statsmodels` are the SPEC author's descriptions; nothing in the repo checks them. Only the BoE
   `bvar` 0.3.5 features in §4.1 are backed (recorded) by `tests/parity/BOE_BVAR_NOTES.md`.
5. **Obstacle 1 of §1.2** ("a typical comparison stitches together an analytic conjugate BVAR from one package and
   an SV sampler from another …"): a qualitative statement about other software with no evidence in the repo.
6. **"None [of the closest competitors] is described in the repo as enforcing one-axis comparisons or shipping a
   sampler-validation suite"** and the BEAR toolbox description (§4.2): BEAR is not mentioned anywhere in the repo
   (dossier author's addition); the absence of a feature in competitors is unverified until the prior-art search (Q3).
7. **Third-party ordering magnitudes** transcribed in `docs/phase2/08_common_sv_and_ordering.md` §B3 (§1.2,
   obstacle 4): numbers from ARRS 2023 / CKY 2024 / CR 2015 as transcribed; not verifiable from the repo.
8. **Literature statements recorded only in notes** — e.g. that stochvol's JSS paper uses a "Geweke variant in unit
   tests" (`docs/phase2/02_sv_sampler.md` line 584; §4.3): recorded in a committed note, but the claim is about an
   external paper and is unverified here (bibliography stage).
9. **CI pass status of nightly/weekly tiers** (NOTES 9 "verified in CI (Linux)"; Geweke/SBC/mutation results on the
   release commit; C6, C12): the repo contains the tests and the NOTES status, but no CI logs or recorded JSON of a
   full nightly/weekly run at `62d15f6`; only test collection (4964 tests) was re-run for this dossier.
10. **Every Clark–Ravazzolo number** (§8.2, §8.4, C9, C11, C15): read correctly from the committed report, but that
    report is superseded by the acceptance run in progress; none may be used until refreshed.
