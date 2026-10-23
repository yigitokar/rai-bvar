# OUTLINE: rai-bvar paper (arXiv preprint, final synthesis)

Sole author: Yigit Okar (affiliation: TODO). LaTeX `article`, 11pt, natbib author-year, BibTeX
`docs/paper/refs.bib` (124 entries; every key below exists there). Main text 29.5 pages plus appendices.

---

## 0. Synthesis record (for the drafter; not part of the paper)

### 0.1 Winner and justification

Both judges named **Outline 1 (software paper first)** the winner. Summed totals: O1 77 (40 + 37),
O2 76 (40 + 36), O3 73 (37 + 36). Judge 1 scored O1 and O2 level at 40 and chose O1 on fit to the brief:
the paper is about the package, including its analytic engine, verdict engine and user API, and O1 is the
only outline that gives each of these its own section. The skeleton and the float IDs below are O1's.
The other two outlines contribute grafts, and every weakness the judges named is fixed (§0.3).

### 0.2 Grafts applied (source → destination)

| Graft | From | Lands in |
| :- | :- | :- |
| Six threats to a credible comparison, each pointing to a section; this becomes the paper's thesis and replaces O1's list of requirements | O3 §1 | §1 P2, §13 |
| Coverage statement: the CR primary arm reduced × SV(rw), and reduced × SV(rw, t), has no registered Geweke/SBC row and no contract fit | O3 §4.4 | §8.10, §11.10, §12 P6, TODO 3 |
| Evidence tags `[V]/[R]/[TC]/[REL]/[ACC]/[FULL]/[PREREG]` as a binding drafting rule | O2 | §0.5, every section |
| Harness principles: complementarity, two-implementation rule in covariance vs precision form, `t_{C−1}`, the r ≤ 5 rule (unverified for φ, σ, p, ν), the finite-variance rule, the data-only filter, Gandy–Scott constants, mixture transcription, exact rejection envelope | O2 §3–§4 | §8.2–§8.5, App. B.5, App. D |
| Mutation wording: "the four mutants without a U check"; coverage U 14/18, G 18/18, B 4, S 1; M17 control 1.94; mutant × tier heatmap; O(εκ) tolerance lesson | O2 §4.2, §5 | §8.4, §8.11, FigD1 |
| Generic "notes for other implementations" that names no package; it replaces O1's optional third-party Appendix I | O2 §11 P2 | §9.6 |
| Limitations: no relative pivot floor in the other formed-Gram draws; Geweke/SBC only at n ≤ 3; power only against 18 mutants; OCSN-10 visibility at G-full undetermined; no SBC at h > 1; prefix test bitwise on macOS, to rounding on x86 | O2 §10 | §12 |
| MCSE bound for non-negative weights (iid delta-method MCSE ≤ 1 nat), presented as an observation to confirm | O3 §5.3 (checked by both judges) | §6.4, App. B.4, TODO 14 |
| Reading rules fixed before the acceptance numbers (groups A–D) | O3 §6, dossier 06 §9.5 | top of §11.9 |
| Seed-to-seed datum (72 vs 69 converged origins; joint h = 1 dLPD +0.809 vs +0.838) | O3 §5.2 | §11.8, §12 P3 |
| Run-history timeline table | O3 T5 | TabF1 (App. F), §11.7 |
| Under expanding windows every "better" comes via the MCS; report the route per verdict; RB and Gaussian side by side with the `rb_gauss_disagree` count; order-invariant Constant arm as a negative control; C4's coded rule compares h = 8 with h = 1 only; why RB-degenerate origins are kept | O3 §2.4–§3.5, §6 | §6.6, §6.9, §11.5–§11.9 |
| Merged forecaster and implementer checklist | O3 §7 + O2 §11 | §13 |
| Cumulative joint dLPD figure (needs a per-origin export) | O3 F4 | FigF3 (App. F) |
| Float budget: about 25 main-text floats cut to 12 | O2 length plan; both judges | §14 |

Not grafted, on purpose: O3's "rtol ≈ 1e-12 across OpenBLAS/Accelerate, vmap and shard_map" and its
unqualified bitwise claims; O2's "no acceptance run is running"; O2's stochvol strict-xfail wording in a
main-text caption.

### 0.3 Weaknesses named by the judges and how each is fixed

1. **Uncited claim about others' practice in the abstract** (dossier 01, unsupported item 5). The abstract now opens with the package's design position (R2).
2. **Python/MATLAB landscape exposed to policy 2.** It is cut to one neutral sentence of names (§2 P2). No feature comparison remains.
3. **Coverage gap (reduced × SV(rw)) not stated.** It is stated in §8.10, §11.10 and §12 P6 (code check: `src/rai_bvar/validation/joint.py` rows 1–17 and 201–205 carry RW only in `structural_sv_rw`, row 14; `tests/mcmc/forecast/mcmc_fits.py` COMPOSITIONS has 12 entries and no reduced SV(rw)).
4. **`|y| ≈ 1e8`.** Corrected to `|y| ≈ 2e8` (test docstring `tests/mcmc/blocks/test_contemp_lstsq.py` l.4; commit 3d3f51b; dossier 05 l.133–134).
5. **Inconsistent DM wording.** One wording throughout, matching the code: a rectangular-kernel long-run variance with h − 1 autocovariances, with Bartlett (Newey–West) weights only if that estimate is not positive (`significance.py:35-53`).
6. **About 25 main-text floats in 29.5 pages.** Twelve stay in the main text and the rest move to the appendices (§14).
7. **No single thesis.** Added: "a comparison is only as credible as its weakest link; rai-bvar makes each link checkable" (the six threats in §1 P2), with a headline sentence before the contributions.
8. **Stale status labels.** Updated in §0.4. D-34 is committed (b9c66c0), and so are the pre-registration and the acceptance config (c823d20). The run is live from a clean worktree at c823d20.
9. **`[ACC: header / acceptance_gaps empty]` named a function, not a key.** Replaced by the JSON keys `config.name == "acceptance"`, `config.lpd_method == "integrated"`, `config.draws == 8000`, `config.mcmc`, `config.seed == 0`, `versions.git_commit`, `versions.git_dirty == false`, and the absence of `pending`. These are the fields read by `acceptance_gaps()` (`benchmarks/clark_ravazzolo.py` l.1022) and `is_acceptance()` (l.1058).
10. **CR script line anchors from 62d15f6.** Re-anchored at c823d20: CR_FULL l.147, CR_PRE_DLPS l.167, ACCEPTANCE config l.186–199, load_data l.236, minnesota/models/structural_models l.325/330/345, relative_table l.479, criteria l.666, RELIABILITY_NOTE l.953, RELIABILITY_NOTE_INTEGRATED l.976, acceptance_gaps l.1022, DEVIATIONS l.1380.
11. **Thinner validation depth than O2.** §8 is now 4.5 pages and includes the harness principles. The derivations are in App. B and App. D.
12. **Reproducibility over-statement risk.** §4.6 states each guarantee with its scope: toys or production kernels, and bitwise or to rounding (rtol 1e-9 for production vmap/shard_map; nothing across platforms).

### 0.4 Status at outline time (2026-10-23, 12:28 EDT)

- Paper worktree `/Users/yigitokar/rai-bvar-paper`: branch `paper-draft` at 62d15f6 (dossiers and refs.bib untracked).
- Main repo `/Users/yigitokar/rai-bvar`: branch `phase-2-cr-acceptance`, HEAD c823d20. Neither of the following commits is on `main` yet:
  - b9c66c0, "Path-integrated Rao–Blackwellised log predictive density (D-34, opt-in)". It adds `tests/unit/test_lpd_integrated.py`, `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`, `tests/integration/test_backtest_lpd_integrated.py` and `tests/unit/test_scores_lpd_method.py`, and changes NOTES 8b and DESIGN D-34 / §7.3. The commit message reports a companion-form oracle agreement of 1.8e-14 (test tolerance 1e-10) and "Fast suite 4655 passed".
  - c823d20, "Clark–Ravazzolo acceptance config and its pre-registration". It adds `benchmarks/reports/clark_ravazzolo_preregistration.md` and `tests/unit/test_clark_ravazzolo_script.py`, and the ACCEPTANCE config (4 × 20,000, burn 1000; 8000 forecast draws; `lpd_method="integrated"`).
- **The acceptance run is in progress.**
  - Launched 2026-10-23 12:10 EDT, after c823d20 was committed, from the detached worktree `/Users/yigitokar/rai-bvar-accept` at c823d20, with a clean tree (`git status` empty).
  - Command: `benchmarks/clark_ravazzolo.py benchmarks/reports/clark_ravazzolo_acceptance --config acceptance --n-jobs 10 --cache ~/.cache/rai-bvar-cr-c823d20`. Its log is at `/tmp/claude-501/cr_accept.log`, which is ephemeral (TODO 2).
  - No acceptance JSON exists yet. **No acceptance number appears anywhere in this outline.**
- **Still blocking for §11:** `docs/paper/NOTES.md` l.87 (main repo) and `docs/phase2/DESIGN.md` l.1674 say the acceptance config "keeps 1000 forecast draws". The committed pre-registration (item 3) and the script (l.195) use 8000. Fix both before the CR section cites either (TODO 1).

### 0.5 Evidence tags (binding drafting rule)

Every number in the draft carries one of these tags. The tags are removed at typesetting, and the
evidence goes to a footnote or to App. G.

- `[V]` verified: reproducible from a committed test or fixture, named in the sentence.
- `[R]` recorded: from a committed note, docstring, commit message or report. The source is named, and the number is never presented as the paper's own measurement.
- `[TC]` to check: excluded from the text, or kept as a placeholder and listed in the TODO.
- `[REL]` to be re-recorded on the release commit through `RAI_BVAR_VALIDATION_RECORD=docs/paper/data/release`. The files are `geweke.jsonl`, `geweke_full.jsonl`, `sbc.jsonl`, `block.jsonl`, `mutation.jsonl` and `mutation_x86.jsonl`, plus `collect.txt` from `pytest --collect-only`.
- `[ACC:<json key>]` placeholder filled from `benchmarks/reports/clark_ravazzolo_acceptance.json`, which must be committed to the main repo from the acceptance worktree.
- `[FULL:<json key>]` first full run: `benchmarks/reports/clark_ravazzolo.json`, code 4bb978a, report commit 551f181, path-conditioned estimator. History only, never presented as a result.
- `[PREREG]` numbers stated in the committed pre-registration (c823d20). They are `[R]`. The diagnostic script and its output are not committed (TODO 4).

Line numbers into dossiers and `src/` are at 62d15f6 unless stated. Line numbers into
`benchmarks/clark_ravazzolo.py` are at c823d20.

---

## Title

**rai-bvar: Composable Bayesian VARs with Stochastic Volatility, Validated Samplers and a Fair
Out-of-Sample Verdict Engine in Python**

(Shorter alternative: "rai-bvar: Composable Bayesian VARs with Stochastic Volatility and a Fair
Verdict Engine".)

## Abstract (draft, about 230 words; the last sentence is a placeholder)

> A density-forecast gain credited to a richer mean or prior in a Bayesian vector autoregression (BVAR) is
> interpretable only when the volatility specification is held fixed. We present rai-bvar, an Apache-2.0
> Python package built around this position. A BVAR is a composition of a mean, a volatility and a prior
> component, and the composition selects the estimation engine. Conjugate Minnesota models, with
> dummy-observation priors, the Giannone–Lenza–Primiceri hierarchy and Lenza–Primiceri COVID scaling, are
> estimated analytically. Compositions with stochastic volatility, outliers or Student-t errors run on
> hand-written JAX Gibbs samplers in recursive structural or corrected reduced form. Both engines return
> one predictive contract, joint path draws with per-draw conditional Gaussian moments. Log predictive
> densities are therefore Rao–Blackwellised. At multi-step horizons, an opt-in estimator integrates the
> intermediate outcomes out. A verdict engine calls a model better only if it differs from its reference
> along exactly one axis, its log-score gain exceeds three Monte Carlo standard errors, a valid test or the
> model confidence set separates the two, and both models converged at 95% or more of the forecast origins. The
> samplers are checked by Geweke tests, simulation-based calibration, analytic twins and parity with
> independent implementations. Eighteen deliberately injected sampler bugs measure the power of these
> checks: in each of 37 mutant-by-tier checks the bug is caught and the unmodified sampler passes. A
> pre-registered replication of Clark and Ravazzolo (2015) on FRED-QD illustrates the workflow. A first
> run missed two of six criteria; we diagnosed why and then fixed the estimator. [ACC: one sentence
> stating which criteria the acceptance run met, from `criteria[*].met`.]

Notes for the drafter:
- The first sentence is the package's design position (SPEC R2). The literature evidence that volatility is first order goes in §1 with citations.
- If D-34 is not on the release commit, delete "At multi-step horizons, an opt-in estimator integrates the intermediate outcomes out." and the last two clauses.
- "95% or more of the forecast origins" is the coded gate (`compare.py`).

## Headline (one sentence, used in §1 and §13)

A model comparison is only as credible as its weakest link: sampler, convergence, Monte Carlo error of the
score, variable ordering, test validity, or post-hoc choices. rai-bvar makes each link checkable, and the
paper shows the evidence for each. The econometrics is not new. The contribution is the integration, the
contracts and the evidence of correctness.

## Contributions (each with status and credit; used verbatim-ish in §1 P4)

- **C1 (software) [V]: a factorial composition grammar.**
  - `BVAR(mean, vol, prior)` resolves automatically to the analytic or the MCMC engine. A 9-row resolution table is encoded by a parametrised test, and invalid compositions fail at construction with actionable errors.
  - `spec()`, `spec_hash` and `axes` make "differs along exactly one axis" a checkable property.
  - Evidence: `src/rai_bvar/api.py`; `tests/unit/test_api_resolution.py::test_resolution_table`; dossier 02 §7.
  - No "first" claim: the prior-art search is TODO 11.
- **C2 (software) [V]: Phases 0–2 populate the volatility axis.**
  - Arms: Constant (IG / IW / iw_equivalent); SV with AR(1) or random-walk log-variance; SV with outliers; SV with Student-t errors; outliers plus t.
  - Forms: structural Mode I, structural Mode II (the exact MCMC twin of the conjugate NIW model) and the corrected reduced-form sampler, with SOC/SUR dummy rows on MCMC.
  - Models are credited to `chan2022asymmetric`, `carriero2022corrigendum`, `carriero2024addressing`, `jacquier2004bayesian`, `giannone2015prior`.
  - Evidence: `src/rai_bvar/validation/joint.py` l.454–480; dossier 01 §2.1.
- **C3 (software + methods): one predictive contract for both engines.**
  - The contract holds joint paths plus each draw's conditional mean and Cholesky factor. The log predictive density is a Gaussian mixture evaluated exactly given each draw, variable subsets are marginalised analytically, and the MCSE is ESS-adjusted.
  - The simulation-consistency invariant is tested for every registered analytic model and for 12 MCMC compositions [V].
  - A documented tail failure at h > 1 is diagnosed. The opt-in path-integrated estimator (D-34) integrates the intermediate outcomes out given each draw's parameters and volatility path [V: tests at b9c66c0; REL on the release commit].
  - Mixture log scores are credited to `geweke2010comparing` and `kruger2021predictive`. The Gaussian algebra is standard.
- **C4 (software; implemented, not statistically validated): a backtest engine and a verdict engine that encode "volatility first".**
  - Ingredients: look-ahead-free refits; one-axis pairing; grid harmonisation; an MC-error gate (|dLPD| > 3 MCSE); a validity policy (DM/HLN p-values valid only for rolling windows); MCS separation; a convergence gate (≥ 95% of origins).
  - Each ingredient is credited (`diebold1995comparing`, `harvey1997testing`, `clark2001tests`, `giacomini2006tests`, `hansen2011model`, `vehtari2021rank`). The claim is the combination into one automatic rule.
  - Its size and power are not studied (§12 P2).
- **C5 (software + validation methodology): sampler validation ships as library code.**
  - `src/rai_bvar/validation/` has 4,230 lines [V: `wc -l`, also at c823d20].
  - Tiers: identities U, block SBC/invariance B, Geweke G, SBC S, analytic twins A, parity P, mutation M. The two-implementation rule is enforced by a test, and the tiers cover 22 composition rows plus 7 block configurations.
  - Methods are credited to `geweke2004getting`, `cook2006validation`, `gelman2017correction`, `talts2018validating`, `modrak2025simulation`, `grosse2014testing`, `gandy2021unit`, `sailynoja2022graphical`.
  - The contribution is the systematic application and a mutation-measured power table. There are 18 mutants and 37 mutant × tier checks, all caught with passing controls [V: reproduced at 62d15f6 on Apple M5 Max; R: x86-64 Linux rerun of M1–M16; M17–M18 recorded on both; REL]. Mutation testing is credited to `demillo1978hints`, `jia2011analysis` and `dutta2018testing`.
- **C6 (methods observations; per-item status).**
  - Exact-mode SV is exact for any mixture table, so a deliberately coarse table gives Geweke tests power over the exact correction [V: M7 with COARSE1].
  - The log-likelihood used as a test quantity catches the CCM-2019 triangular-step error that parameter ranks miss [V: block SBC p 2.8e-11 vs control min p 0.10; the whole-scan "not caught" result is R]. This is consistent with `modrak2025simulation`.
  - Per-replication SBC ESS cannot be certified from L = 99 kept draws [V: design erratum].
  - Dummy rows have a closed-form λ marginal and an exact rejection envelope [V: NOTES 13].
- **C7 (engineering case studies; fixes verified).** Float64 failures were surfaced by validation and replication runs and fixed in the kernel, without jitter:
  - a Householder-QR a-row draw where the formed Gram loses an eigenvalue pair 1.9e17 : 14.8;
  - a cancellation-free excess-pivot recursion for the random-walk SV path Laplacian at σ ≈ 5e-9.

  The recursion is credited as an instance of GTH-type subtraction-free elimination (`grassmann1985regenerative`, `alfa2002accurate`, `ye2008computing`). The application is claimed, not the algorithm.
- **C8 (engineering; verified with documented limits): reproducible JAX MCMC.**
  - Keys are counter-based (`salmon2011parallel`), derived per (chain, absolute sweep, block, equation).
  - Bitwise invariance is tested on runner toys for chunking, burn pieces and fused/unfused plans, and on production `structural_sv` for chain prefixes under `chains="map"`. Production vmap and shard_map agree to rtol 1e-9. Nothing is guaranteed across platforms or JAX versions.
  - The engine's key layout avoids a partitionable-threefry key hazard that JAX's key-reuse checker cannot see [V].
- **C9 (empirical illustration): a pre-registered qualitative replication of `clark2015macroeconomic`** on FRED-QD with all 24 variable orderings.
  - First full run [FULL]: criteria 1, 2, 3 and 5 met; 4 and 6 not.
  - Diagnosis: the convergence gate bound on R-hat noise, and the path-conditioned RB score degenerated at crisis origins.
  - Library fixes: D-34 and the RW pivot recursion. A fit-level convergence diagnostic [PREREG] set the budget.
  - The acceptance run's settings were committed (c823d20) before it was launched. The framing is explicitly non-blind. All acceptance numbers are `[ACC]`.

**Explicitly not new** (one sentence in §1): the priors, SV samplers, mixture approximations, ASIS, CCCM,
precision samplers, tests of predictive ability, scoring rules, Geweke/SBC tests, mutation testing and
subtraction-free elimination are all existing methods, credited where used.

---

## 1. Introduction (2.5 pages)

**P1: the question (0.4 p).**
- Open with SPEC §1.1 verbatim: "Does added model complexity improve out-of-sample density forecasts once time-varying volatility is accounted for?"
- Time-varying volatility matters for macro density forecasts: `clark2011realtime`, `clark2015macroeconomic`, `carriero2016common`; drifting volatilities in `cogley2005drifts`, `primiceri2005time`, `sims2006were`; survey `chan2024bvars`; specification choices `carriero2015bayesian`.
- State rule R2 ("volatility first") as the package's design position, not as a literature finding: a mean or prior effect is measured holding the volatility arm fixed (e.g. Linear × SV vs Minnesota-variant × SV, never against Linear × Constant).

**P2: six threats to a credible comparison (the thesis; 0.8 p).** Each threat is stated as a requirement, not as a
claim about other tools, and points to the section that addresses it.
1. *The sampler may target the wrong posterior.* A widely used equation-by-equation algorithm did (`carriero2019large`, shown by `bognanni2022comment`, corrected by Carriero, Chan, Clark & Marcellino in `carriero2022corrigendum`; never write "with Chan & Yu"). → §8.
2. *Every origin is a separate MCMC fit that must converge.* → §6.8, §8.12, §11.8.
3. *The log score is itself a Monte Carlo estimate, and its error can be silent in the tails at h > 1.* → §6.3–§6.4.
4. *Triangular SV models depend on the variable ordering* (`arias2023macroeconomic`, `chan2024large`). → §6.9, §11.9.
5. *Test validity.* DM/HLN p-values are not formally valid for nested models under expanding windows (`clark2001tests`, `giacomini2006tests`), so a decision needs a second route, the MCS (`hansen2011model`), and proper scores (`gneiting2007strictly`). → §6.6.
6. *Analysis choices can be made after seeing scores.* → §11.8 (pre-registration).

Two infrastructure requirements follow: one estimation interface and output format across conjugate and
non-conjugate models (→ §4, §6.2), and no look-ahead in data-dependent prior settings (R9; → §6.7).

**P3: what rai-bvar is (0.5 p).**
- One-sentence description from `pyproject.toml`: "ablation-ready Bayesian VARs: composable mean, volatility and prior components with a fair out-of-sample verdict engine".
- Five design commitments:
  1. the composition chooses the engine;
  2. one predictive contract;
  3. one backtest and one verdict engine;
  4. validation shipped as library code;
  5. an Apache-2.0 implementation in which GPL packages are only run as oracles (qualified in §2 P4).
- Scope sentence (dossier 01 §2.1): "ablation-ready" describes the architecture. Phases 0–2 populate the volatility axis; the mean axis is linear and the prior axis is the Minnesota family.
- Release state: version 0.0.1, pre-alpha. Release commit [TODO 20].

**P4: contributions (0.6 p).**
- The headline sentence, then C1–C9 as bullets, one line each with its credit.
- Then the explicit "not new" sentence.
- C9 is described as an illustration, not as a test of the econometric claim.

**P5: map of the paper (0.2 p).**
- Footnotes point to the AI-assistance and process disclosure (App. J) and the licensing statement (App. H).

- **Evidence:** dossier 01 §1, §2, §2.1, §8.1, §11; SPEC §1.1 l.27–39, R2 l.109, R6 l.121, R8 l.125, R9; `pyproject.toml` (description, version 0.0.1, classifier "Development Status :: 2 - Pre-Alpha"); dossier 02 §1; dossier 99 §5 item 9.
- **Bib keys:** `clark2011realtime` `clark2015macroeconomic` `carriero2016common` `cogley2005drifts` `primiceri2005time` `sims2006were` `chan2024bvars` `carriero2015bayesian` `carriero2019large` `bognanni2022comment` `carriero2022corrigendum` `arias2023macroeconomic` `chan2024large` `clark2001tests` `giacomini2006tests` `hansen2011model` `gneiting2007strictly`
- **Floats:** none.

---

## 2. Related software, numerical oracles and licensing (1 page)

**P1: R and Python packages in their Phases 0–2 roles, as committed (0.35 p).**
- R BVAR 1.0.5 (`kuschnig2021bvar`, `kuschnig2024bvar`) was run as the oracle for the GLP hierarchy, SOC/SUR and MH.
- stochvol 3.2.1 (`kastner2016dealing`, `hosszejni2021modeling`) was run as the univariate SV oracle. DESIGN names 3.2.9; state the version actually used (`tests/parity/SV_PARAMETRIZATION.md` "Versions").
- bayesianVARs 0.1.8 (`gruber2026bayesianvars`) could not be built in this environment (no gfortran in the test environment; DESIGN Stage 4 amendment (2)), and no finding about it is stated. The reduced-form exit criterion is met by substitution, with an in-house exact system-wide Gibbs oracle written in R.
- The Bank of England bvar 0.3.5 (`labonne2026bvar`, MIT) is the analytic engine's second oracle.
- bvartools (`mohr2026bvartools`) is an architectural reference and was never run.
- Roadmap oracles, in one clause: `kastner2026factorstochvol`, `liu2026bvars`, `wozniak2026bsvars`, `wang2026bsvarsigns`, `krueger2015bvarsv`.

**P2: other software, one neutral sentence (0.1 p).**
- "Other open-source BVAR software exists in Python (`chan2026bvartoolkitpython`, `kim2026bvhar`, `pymc2026extras`, `pinder2026impulso`, `shaw2026srvar`; frequentist VARs in `seabold2010statsmodels`), MATLAB (`dieppe2016bear`, `chan2026bvartoolkit`) and R (`boeck2022bgvar`); we do not compare features."
- No characterisation of scope, gaps, licences or dates. The SPEC §1.4 rows are unverified (dossier 01, unsupported item 4); see TODO 12.

**P3: what the package builds on (0.15 p).**
- Libraries: JAX (`bradbury2018jax`, `frostig2018compiling`), NumPy (`harris2020array`), SciPy (`virtanen2020scipy`), xarray (`hoyer2017xarray`), scoringrules (`zanetta2024scoringrules`; the R analogue is `jordan2019evaluating`). ArviZ (`kumar2019arviz`) is used only as a source of reference values in diagnostics tests.
- Position on PPLs (`abrilpla2023pymc`, `phan2019composable`): the core is hand-written Gibbs by design (R7). The SPEC claim that NUTS mixes worse is **not** made (dossier 01, unsupported item 2).

**P4: licensing and clean-room practice, with the exact wording (0.4 p).**
- Quote dossier 01 §6.2 verbatim: "No code was translated from GPL sources. Algorithms were implemented from the published papers. GPL packages were run as numerical oracles, and in a few documented cases (named in the repository notes) their source was read to pin down an oracle's behaviour, defaults or constants, never to write implementation code."
- Footnote naming the notes:
  - `docs/phase2/02_sv_sampler.md` (stochvol 3.2.9, bayesianVARs 0.1.8 and bvarsv 1.1, for behaviour and constants);
  - `docs/phase2/01_omori_mixture.md` (mixture constants confirmed in six GPL packages);
  - `docs/phase2/03_sv_outliers_t.md` (stochvol greps, plus the unlicensed CCMM MATLAB code, greps only);
  - `docs/phase2/07_jax_patterns.md` (cross-check search of the stochvol header for the 10 constants);
  - `tests/parity/BVAR_PARAMETRIZATION.md` (BVAR 1.0.5 R sources read to establish behaviour).
- The fixtures are outputs of GPL programs and are treated as data. Citing `fsf2026gplfaq` as a web page, not legal advice, is the author's decision (TODO 17).
- FRED-MD/QD files are downloaded on demand and never bundled (`mccracken2016fredmd`, `mccracken2021fredqd`; cite the 2021 *Review*, not the 2020 WP).
- Never write "clean room" unqualified. Never mention a PR-template checkbox (none exists).

- **Evidence:** dossier 01 §4.1–§4.2, §5, §6.1–§6.2, unsupported items 1–6; dossier 99 §5 items 1, 3, 6, 7; `tests/parity/BOE_BVAR_NOTES.md`; `tests/parity/SV_PARAMETRIZATION.md` "Versions"; `tools/make_r_fixtures.R`; `LICENSE`; `pyproject.toml`.
- **Bib keys:** `kuschnig2021bvar` `kuschnig2024bvar` `kastner2016dealing` `hosszejni2021modeling` `gruber2026bayesianvars` `labonne2026bvar` `mohr2026bvartools` `kastner2026factorstochvol` `liu2026bvars` `wozniak2026bsvars` `wang2026bsvarsigns` `krueger2015bvarsv` `chan2026bvartoolkitpython` `kim2026bvhar` `pymc2026extras` `pinder2026impulso` `shaw2026srvar` `seabold2010statsmodels` `dieppe2016bear` `chan2026bvartoolkit` `boeck2022bgvar` `bradbury2018jax` `frostig2018compiling` `harris2020array` `virtanen2020scipy` `hoyer2017xarray` `zanetta2024scoringrules` `jordan2019evaluating` `kumar2019arviz` `abrilpla2023pymc` `phan2019composable` `fsf2026gplfaq` `mccracken2016fredmd` `mccracken2021fredqd`
- **Floats:** none.

---

## 3. The model framework (2.5 pages)

**3.1 Notation (0.2 p).**
- Regressors x_t = (d_t', y_{t−1}', …, y_{t−p}')', k = n_d + np, n_d ∈ {0, 1, 2} ('n', 'c', 'ct').
- Use κ for the Minnesota tightness (code field `lambda_`) to avoid a clash with λ_{i,t}. This resolves dossier 02 §2 / Q1.
- 1-based ι in prior formulas. IG(c, C) is shape/scale.
- The full table is in App. A.

**3.2 Recursive structural form, used by all MCMC models (0.4 p).**

  A y_t = Γ x_t + Λ_t^{1/2} ε_t, with A unit lower triangular;
  reduced form B' = A⁻¹Γ, Σ_t = A⁻¹Λ_t A⁻¹'.

- Log-likelihood (|A| = 1): Σ_i Σ_t [−½ log(2π λ_{i,t}) − e²_{i,t}/(2λ_{i,t})]. The same expression is used as the primary convergence monitor and as a test quantity in §8.
- Given the volatilities, the equations are conditionally independent, so one sweep costs O(n k³) = O(n⁴p³) instead of O(n⁶p³).
- The structural form avoids the CCM-2019 problem by construction. The reduced form uses the corrected triangular algorithm (`carriero2022corrigendum`; erroneous version `carriero2019large`).

**3.3 Priors (1.1 p).**

(a) **Conjugate NIW Minnesota**, analytic engine (`litterman1986forecasting`, `kadiyala1997numerical`, `banbura2010large`):
- Σ ~ IW(diag ψ, d = n + 2); vec B | Σ ~ N(vec b, Σ ⊗ Ω), with Ω_(l,j) = κ²(d − n − 1)/(l^α ψ_j).
- The own first lag has a transform-aware prior mean: 1 for level tcodes 1 and 4, 0 for differences.
- ψ is the AR(p) residual variance computed within the window.
- The Kronecker structure forces cross = 1.

(b) **SOC/SUR dummy rows** (`doan1984forecasting`, `sims1998bayesian`, as in `giannone2015prior`). The documented default is `soc_scope='persistent'`.

(c) **GLP hierarchy**, 'fixed' / 'eb' / 'mh' (`giannone2015prior`), with the NIW log ML.
- With dummies, log ML = ML(dummies + data) − ML(dummies).
- Every log-determinant comes from triangular factors.

(d) **Lenza–Primiceri COVID scaling** (`lenza2022estimate`), with Jacobian −n Σ_t log s_t. Analytic engine only.

(e) **Structural Minnesota with D3 centring**, E[γ_i | a_i] = M ã_i, sampled in δ-coordinates δ_i = γ_i − M_{<i} a_i, so y_i = (X M_{<i} − Y_{<i}) a_i + X δ_i + e_i.
- *Mode I* (moment-matched, independent) reproduces the NIW reduced-form mean, the marginal variances and E[Σ], without the 2^{i−1} variance growth noted by `chan2024large`.
- *Mode II* (λ-scaled) is exactly the structural form of the NIW prior at κ_c = κ_a = 1 (`chan2022asymmetric` Prop. 1; IW partition theorem, `gupta2000matrix`). At κ_c ≠ 1 it is Chan's asymmetric conjugate prior.
- *iw_equivalent* gives Σ ~ IW(diag ψ, n + 2) exactly and is order-invariant in reduced form.

(f) **Reduced-form prior** π_j ~ N(m_j, V_j), moment-matched to the NIW marginal at cross = 1.

(g) **Hyperparameters on MCMC.**
- 'fixed', or an 'eb' plug-in from the Phase 1 conjugate twin on the same window, so that every volatility arm gets the same mean prior (R2).
- This is type-II ML only for the Mode II twin, and a documented plug-in otherwise. Never call SV hyperparameters "EB-optimal".
- 'mh' and LP scaling are rejected on MCMC.

**3.4 Volatility arms (0.6 p).**

*Constant.*
- λ_i ~ IG(1 + ι/2, ψ_ι/2).
- Mode II and iw_equivalent couple through one "coupling channel" (n_active, quad) in the IG conditional.

*SV.*
- h_{i,t} = μ_i + φ_i(h_{i,t−1} − μ_i) + σ_i η_{i,t}, or a random walk.
- Default priors:
  - μ ~ N(log(ψ/ι), 10), centred on the data in each window (R9);
  - (φ + 1)/2 ~ Beta(20, 1.5);
  - σ² ~ B_σ χ²₁ with B_σ = 1 (0.1 for RW), optionally IG;
  - RW h_0 ~ N(log(ψ/ι), 4).

*Robust SV.*
- e = o √τ e^{h/2} ε.
- Outliers follow `carriero2024addressing`: o ∈ {1..20} on a grid in the spirit of `stock2016core`, P(o = 1) = 1 − p, and p ~ Beta from OutlierPrior(every_years, prior_years).
- Standardized Student-t: τ ~ IG(ν/2, (ν − 2)/2), with `t_scale='unit'` (`jacquier2004bayesian`) and ν ~ DiscreteUniform{3..40} by collapsed exact discrete Gibbs.
- Outliers and t can be combined.

**3.5 What is delivered (0.2 p).** Table 1 lists the components delivered against the roadmap. Each sentence cites
its committed test:
- Chib identity and dummy-difference log ML;
- Mode I moments;
- Mode II = NIW (KS);
- structural NIG log ML = Phase 1 to 1e-10;
- iw_equivalent Σ ~ IW.

- **Key equations:** structural form; likelihood; Ω; D3 centring; SV state equation; robust SV observation equation.
- **Evidence [V]:**
  - dossier 02 §2–§6, §14 (E1–E14);
  - `src/rai_bvar/priors/{minnesota,structural,reduced,sv_arrays}.py`;
  - `src/rai_bvar/engines/analytic.py` l.8–15, 425–483;
  - `tests/analytic/test_conjugate.py::test_log_ml_satisfies_the_chib_identity`;
  - `tests/analytic/test_structural_nig.py::test_a1_log_ml_equals_phase1`;
  - `tests/priors/test_structural_prior.py::test_mode_i_matches_the_niw_moments_exactly_and_by_monte_carlo`, `::test_mode_ii_prior_is_the_niw_prior`, `::test_iw_equivalent_sigma_is_inverse_wishart`;
  - `tests/analytic/test_eb_mcmc.py::test_eb_plug_in_equals_the_phase1_eb_mode_at_cross_1`.
- **Bib keys:** `carriero2022corrigendum` `carriero2019large` `litterman1986forecasting` `kadiyala1997numerical` `banbura2010large` `doan1984forecasting` `sims1998bayesian` `giannone2015prior` `lenza2022estimate` `chan2022asymmetric` `gupta2000matrix` `chan2024large` `carriero2024addressing` `stock2016core` `jacquier2004bayesian` `kastner2016dealing`
- **Floats:** **Tab1** (main).

---

## 4. Software architecture (2.5 pages)

**4.1 Design rules (0.3 p).**
- Rules that shape the API: R1 (composition over inheritance), R2, R3 (one predictive contract), R8 (ordering surfaced), R9 (no look-ahead).
- Rules that govern the internals: R4 (float64, no explicit inverses, precision samplers, log-space densities), R6 (Geweke/SBC before merge), R7 (JAX Gibbs, no PPL).
- **Fig1** shows the layers: specs → resolution → two engines → forecast contract → evaluation; the validation package wraps the production sweep.

**4.2 Composition and engine resolution (0.5 p).**
- `BVAR.__post_init__` runs in three stages:
  1. resolve the 'auto' defaults (`Minnesota.form`, `Constant.prior`);
  2. apply the 9 engine rules (**Tab2**);
  3. apply the MCMC-only rules and the JAX-free legality check.
- There is no user-settable engine field.
- `spec()` is plain JSON; `from_spec(spec())` round-trips (property test); `spec_hash` is stable; a golden test checks that the Phase 1 hashes are unchanged.
- `axes` = {mean, prior, vol, lags, deterministic, outliers}. Labels encode every non-default setting.
- `import rai_bvar` imports no JAX (tested in a subprocess), and the MCMC engine is imported lazily.

**4.3 The analytic engine, NumPy/SciPy (0.3 p).**
- The posterior comes from one QR of the stacked [X* | Y*] (Minnesota rows, dummy rows, data). Ω̄ is never formed.
- Draws use the Bartlett decomposition.
- EB uses L-BFGS-B, and the MH hierarchy follows GLP.
- Recursive IRFs and the FEVD are provided.

**4.4 The MCMC engine in six ideas (0.8 p).**
1. The static/traced split is the JIT cache key (StaticConfig × argument shapes).
2. Equations and chains are vmap axes; chains can optionally run under `lax.map` or `shard_map`.
3. The Gibbs order is data. A Plan of Steps declares read/write sets, which are checked against invariants and by a tree-diff of the actual writes. The mixture indicators are local to the vol step, so "nothing between s and h" (`delnegro2015time`) holds by construction.
4. Kernels are pure and take injected noise: core(noise, …) ∘ draw_noise(key, …). Every exact draw exposes its conditional, and every MH step exposes its log acceptance ratio. This enables both the identities (§8.4) and deterministic replay.
5. Randomness is counter-based (credit `salmon2011parallel`). fold_in chains over seed → stream → data salt → chain → absolute sweep → block → equation → sub-draw, with uint32 counters (erratum E7) and no key in any loop carry.
6. Failures are values. Each chain carries its first (code, it), with code = 1 + 1000·block + eq. The host replays the failing chunk deterministically and raises `MCMCNumericalError` with diagnostics recomputed from the exposed conditional. No jitter is added and no chain is dropped.

**4.5 Settings vs runtime (0.2 p).**
- `MCMCSettings` is part of the spec. Defaults: chains 4, draws 2000, burn 1000, store_max 2500, rhat_max 1.01, ess_min 400.
- `Runtime` is not part of the spec: devices, chunk, memory budget, storage dir, replay. Each knob except devices has an invariance test (scoped as in §4.6).
- `MCMCFit` exposes:
  - the posterior (xarray DataTree);
  - diagnostics;
  - `summary()`, with prior mode, hyperparameters, implied reduced-form moments and ordering;
  - `converged`, `warnings`, `forecast`, `irf`, `final_state` and timing.

**4.6 Reproducibility, stated exactly with its limits (0.3 p).** The full matrix is TabG1 in App. G.
- *Bitwise, runner toys:* chunk size and burn piece size; aligned thinning; fused vs unfused plans; vmap over chains vs a sequential loop.
- *Bitwise, production:* chain prefixes with `chains="map"` (production `structural_sv`); memmap vs memory store.
- *To rounding:*
  - production chain prefixes under vmap with SV/CCCM kernels, rtol 1e-9 (XLA vectorises reductions per batch size);
  - vmap vs `shard_map(4)` on production `structural_sv`, rtol 1e-9, short burn;
  - in-loop vs post-hoc forecasts, 1e-12.
- *Toy chain-prefix test:* bitwise on macOS, to rounding on x86 Linux (`test_chunking.py::test_chain_prefix_invariance` docstring).
- *Production chunk invariance* follows from the key layout but is `[TC]`.
- *No guarantee* across platforms (OpenBLAS vs Accelerate), JAX versions or PRNG implementations.
- Every fit records jax/jaxlib versions, PRNG implementation and partitionability, platform, device count, seed, salt, the resolved gram/const_path/T_pad, mixture, offset, the exact flag, ordering, spec hash and package version.
- One sentence on the partitionable-threefry hazard: words drawn directly from a parent key equal the key data of its fold_in children, which `debug_key_reuse` cannot see. The engine avoids it by layout [V].
- Do not quote NOTES 9's "rtol ≈ 1e-12" for production kernels.

**4.7 Extension protocol (0.1 p).** `MeanBlock`, `VolBlock`, `CoefPrior`, `GlobalStep` and the reference registry used by
validation. "Phase 3–5 components plug in without editing sweep.py, runner.py or forecast.py" is a design
intention, not a fact (dossier 01, unsupported item 3).

- **Evidence:**
  - dossier 02 §1, §7, §8;
  - dossier 03 §1, §3, §4, §5.1, §7.1–§7.4, §8, §9, §10;
  - `src/rai_bvar/api.py` l.68–191, 229–333;
  - `src/rai_bvar/settings.py` l.62–184;
  - `src/rai_bvar/engines/mcmc/{config,plan,keys,runner,results}.py`;
  - `tests/unit/test_api_resolution.py`, `test_phase1_hash_golden.py`, `test_spec_roundtrip_mcmc.py`;
  - `tests/mcmc/unit/test_import_no_jax.py`;
  - `tests/mcmc/runner/test_plan.py::test_plan_invariants_every_composition_with_tree_diff`;
  - `tests/mcmc/runner/{test_chunking,test_sweep,test_devices_bitwise,test_storage}.py`;
  - `tests/mcmc/forecast/test_forecast_core.py`;
  - `tests/mcmc/blocks/test_robust.py::test_ahead_and_transition_give_the_core_its_own_sub_key`.
- **Bib keys:** `delnegro2015time` `salmon2011parallel` `bradbury2018jax` `frostig2018compiling` `giannone2015prior` `hoyer2017xarray`
- **Floats:** **Fig1**, **Tab2** (main). The plan figure (FigC1), the contract table (TabC1) and the reproducibility matrix (TabG1) are in the appendices.

---

## 5. The MCMC samplers (2.5 pages)

**5.1 One sweep (Algorithm 1, compact box; 0.4 p).**
- Structural form, per equation under vmap: the joint Gaussian draw (δ_i, a_i) | λ_eff, then the residual, then the vol block, atomically.
- Inside the vol block the transitory steps run first (o → p → ν → τ). The indicators s come immediately before h, then θ_SV centred, then non-centred.
- Reduced form: a pre-phase CCCM step Π | A, λ, then a_j | Π, λ_j, then the vol step.
- Collapsed transitory steps are credited to `vandyk2008partially`.

**5.2 The structural coefficient draw (0.3 p).**
- A packed weighted Gram plus a Jacobi-scaled Cholesky two-solve precision draw.
- Exact constant-volatility fast paths are chosen by a deterministic shape rule: `eigen` (O(q²) per draw), `fixed_factor` (Mode II) and `cholesky_const` (iw_equivalent). They never change the target.
- Non-fusing means draw a_i by QR (forward reference to §9.2).

**5.3 The CCCM reduced-form step (0.2 p).** Following `carriero2022corrigendum`, with weights W = (1/Λ)(A ∘ A).
It is a block Gauss–Seidel scan on the joint precision of vec Π (`fox2017accelerated`).

**5.4 The SV core (Algorithm 2, compact; 0.8 p).**
- Working observation y* = log(ẽ² + c), with relative offset c = 1e-12 ψ.
- The OCSN 10-component mixture (`omori2007stochastic`); KSC-7 (`kim1998stochastic`) is an option. Indicators are drawn by inverse-CDF categorical draws.
- A (T + 1)-state AWOL tridiagonal precision draw (`chan2009efficient`, `mccausland2011simulation`, `rue2001fast`) in non-centred scaled form.
- Exact MH correction: w(h) = Σ_t m_t [log N(ẽ_t; 0, e^{h_t}) − log g(y*_t − h_t)] (`delnegro2015time` Alg. 3; precedent `stroud2003nonlinear`). It makes the chain exact for any mixture table and any offset.
- Then the centred (γ, φ) and σ² steps.
- Then the ASIS non-centred step (`yu2011center`, `kastner2014ancillarity`), with **one** accept/reject that combines the auxiliary ratio and the w-difference. That combination is derived in `docs/phase2/01_omori_mixture.md` §10.3 and is not claimed as new (TODO 11).
- On acceptance, a measure-preserving fold (σ, h̃) ↦ (−σ, −h̃) follows the sign-symmetry argument of `fruhwirth2010stochastic` [V: identity tests].
- The cold start draws h once by AWOL, because the pilot found σ = 0 absorbing when h is constant [R]. Pilot cost and acceptance numbers come from a prototype and go only to App. E.

**5.5 The RW path as a path Laplacian (Algorithm 3; 0.3 p).** Q̃ = σ²Q = Lap_w + diag(δ). The excess-pivot recursion:

  q_0 = δ_0,  q_t = δ_t + w_{t−1} q_{t−1}/(w_{t−1} + q_{t−1}),  p_t = q_t + w_t,
  log|Q̃| = Σ_t [log w_t + log1p(q_t/w_t)].

Every term is ≥ 0, so nothing cancels. The failure that motivated it is described in §9.3.

**5.6 Transitory components (0.3 p).**
- Outlier logits: l_tk = log w_k − log k − z²/(2k²), K = 20.
- p ~ Beta(a_p + N_1, b_p + T − N_1).
- Student-t: ν grid logits with τ integrated out; τ ~ IG((ν + 1)/2, (c + r²)/2).
- A vectorised Marsaglia–Tsang gamma sampler (`marsaglia2000simple`) replaces the batched rejection loop of `jax.random.loggamma`. The speed-up is a prototype number and is not quoted.

**5.7 Constant core (0.2 p).** λ_i | · ~ IG(c_i + T/2 + n_active/2, C_i + ½ Σ e² + quad), with the coupling channel
from the coefficient step.

- **Evidence:**
  - dossier 03 §2, §5.1–§5.3, §6.1–§6.10;
  - dossier 02 §6.4;
  - `src/rai_bvar/components/vol/{sv,transitory,constant,mixture}.py`;
  - `src/rai_bvar/engines/mcmc/linalg.py` l.425–480;
  - `tests/mcmc/blocks/test_sv_core.py::test_fold_is_measure_preserving_map`, `::test_nc_mu_sigma_log_accept_identity`, `::test_nc_accept_uses_w_of_the_post_h_step_state`, `::test_exact_h_acceptance_at_T300`, `::test_awol_rw_laplacian_is_the_rw_system`;
  - `docs/phase2/01_omori_mixture.md` §10.2–§10.3.
- **Bib keys:** `vandyk2008partially` `carriero2022corrigendum` `fox2017accelerated` `omori2007stochastic` `kim1998stochastic` `chan2009efficient` `mccausland2011simulation` `rue2001fast` `delnegro2015time` `stroud2003nonlinear` `yu2011center` `kastner2014ancillarity` `fruhwirth2010stochastic` `marsaglia2000simple` `jacquier2004bayesian` `carriero2024addressing`
- **Floats:** Algorithms 1–3 as compact boxes, counted in the page budget. The full pseudocode is in App. C.

---

## 6. Forecasts, the predictive contract and the verdict engine (3 pages)

**6.1 Joint path simulation (0.3 p).**
- For each selected draw, simulate the future volatilities (including o and τ). Then scan h = 1..H:
  - m_{s,h} = A⁻¹Γ x_{T+h};
  - L_{s,h} = A⁻¹ diag(exp(½ log λ^eff_{T+h}));
  - y = m + Lε.

  L is the conditional Cholesky factor, at no extra cost.
- Forecast draws are evenly spaced stored draws per chain. In-loop and post-hoc forecasts agree to 1e-12 [V].
- IRFs: Φ_h = Σ_l A_l Φ_{h−l}.
  - Unit-shock responses Φ_h A⁻¹ are date-free.
  - One-s.d. responses are date-specific and use the persistent variance by default.
  - The FEVD is computed from the IRFs.

**6.2 The contract (0.3 p; the field table is TabC1 in App. C).**
- Fields:
  - `paths` and `paths_level`: the pathwise inverse of the tcodes, never applied to quantiles;
  - `cond_mean`, `cond_chol`;
  - attrs: origin, model, engine, spec, spec_hash, tcodes, seed, converged, warnings. MCMC adds ordering and sampler metadata.
- Invariants:
  - L⁻¹(paths − cond_mean) is iid N(0, I);
  - draws are joint across horizons and variables;
  - stacking is chain-major.
- These are tested for every registered analytic model and for 12 MCMC compositions [V: `tests/contract/test_forecast_contract*.py`; `tests/mcmc/forecast/mcmc_fits.py` l.128–141]. Name the uncovered ones, including reduced × SV(rw), the CR primary arm.
- `lag_coefs` is an accelerator variable carried by MCMC forecasts (D-34), not a contract field.

**6.3 The RB log predictive density (0.4 p).**

  log p̂ = logsumexp_s log N(y; m_{s,h}, L_{s,h}L_{s,h}') − log S.

- Variable subsets are exact: a leading block uses L[:j, :j], and other subsets use QR.
- MCSE (delta method): sd(w)/(√S w̄), with w_s = exp(log p_s − max) and the sample sd (ddof = 1; `tests/unit/test_lpd_integrated.py` l.202). For MCMC draws it is multiplied by √(S/ESS).
- An optional transitory RB integrates o and τ out at h = 1.
- Credit mixture log scores to `geweke2010comparing` and `kruger2021predictive`.

**6.4 The h > 1 tail problem and the path-integrated estimator (the methods point; 0.8 p).**

*Failure.*
- For h > 1 the SPEC §7 estimator conditions on each draw's simulated path up to T + h − 1. For outcomes in the tail, one draw carries all the weight.
- Diagnostic cell (Constant, UR, h = 8, origin 2008Q2) [R: NOTES 8b, WP23 review; b9c66c0 commit message; REL: regenerate with `fig_lpd_estimators.py`]:

  | Draws | Path-conditioned estimate |
  | :- | :- |
  | 1000 | −93.6 |
  | 8000 | −46 to −63 |
  | 64,000 | −36 |

  The path-integrated mixture gives −19.1 over the same 1000 draws. Do not call −19.1 "exact": it is exact given those draws and is itself a Monte Carlo estimate.

*Observation, to be confirmed by the author (App. B.4; TODO 14).*
- For non-negative weights, Σw² ≤ (Σw)², so the sample variance satisfies s² ≤ S w̄². The iid delta-method MCSE s/(√S w̄) is therefore at most 1 nat, whatever the error, with equality if and only if one draw carries all the weight. With the population sd the bound is √((S − 1)/S).
- This explains why the reported MCSE "saturates near 1" and why the RB-degenerate flag sits at 0.9.
- The √(S/ESS) factor can raise the reported MCSE above 1 for MCMC draws.
- Lesson: an MCSE is a model of the error, not a certificate; inspect the weight ESS (`kish1965survey`).
- No "by construction" wording until the bound is written out and unit-tested.

*Fix: D-34, `lpd_method="integrated"` [V: b9c66c0 tests; REL].*

  y_{T+h} | θ_s, λ_s ~ N(m_h, M_h M_h'),
  m_h = cond_mean_h − Σ_{k<h} Φ_{h−k} u_k,  u_k = paths_k − cond_mean_k,
  M_h = [Φ_{h−1}L_1 | … | Φ_0 L_h],  Φ_0 = I,  Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}.

- Subset factors come from the QR of M_h[idx]'. M_h M_h' is never formed, and no extra random numbers are drawn. At h = 1 the conditioned code runs, bitwise.
- Exactness:
  - against a companion-form oracle at h = 2..8 (joint, leading and non-leading subsets, 6-s.d. tails): test tolerance 1e-10 [V]; measured 1.8e-14 [R: commit message];
  - unit equivariance at 1e±6 [V];
  - per-draw densities against `jax.jacfwd` of the forecast core on real MCMC fits [V: `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`].
- Synthetic persistent VAR at h = 8, 6 s.d.: the conditioned estimator gives −35.2 (MCSE 1.0) against an exact −22.3 [V: `tests/unit/test_lpd_integrated.py` l.169].

*Residual error [R: NOTES 8b addendum at b9c66c0; PREREG item 3].*
- The estimator is still an average over draws whose weights concentrate.
- At 2008Q2, UR h = 8 with 1000 draws (acceptance budget), Constant's weight ESS is 2–6. The 1000-draw estimate has sd 0.75–1.06 across thinning offsets and a downward bias of 0.3–0.6 nats. At 8000 draws the sd is 0.09–0.19 and the bias ≈ 0.
- The MCMC error remains: about 1.3 nats across fit seeds.

*Scope.*
- Opt-in; the SPEC default is unchanged.
- Covers MCMC forecasts of linear or zero means (they carry `lag_coefs`).
- Analytic, benchmark and pooled forecasts raise at h > 1, as does `rb_transitory`.

**6.5 Scores (0.2 p).** All on the transformed scale:
- joint and marginal LPD;
- fair CRPS via PWM (`zanetta2024scoringrules`);
- qw-CRPS (`gneiting2011comparing`) and tw-CRPS;
- energy and variogram (p = 0.5; `scheuerer2015variogram`) on a ≤ 1000-draw subsample;
- squared and absolute errors;
- PIT and coverage.

**6.6 Tests (0.3 p).**
- DM on loss differences (`diebold1995comparing`). The long-run variance uses a rectangular kernel with h − 1 autocovariances; Bartlett (Newey–West) weights (`newey1987simple`) are used only if that estimate is not positive. HLN correction with t_{P−1} (`harvey1997testing`).
  - State this as implemented; SPEC says Newey–West (dossier 02 D1).
- Validity policy: p-values are formally valid only for rolling or fixed windows and are labelled descriptive otherwise (`clark2001tests`, `giacomini2006tests`; density-forecast analogue `amisano2007comparing`).
- MCS: T_max on −lpd, circular block bootstrap with block length round(P^{1/3}), 2000 replications (`hansen2011model`).
- **Consequence stated explicitly:** under expanding windows a "better" verdict can only come through the MCS route. The comparison table reports two-sided p-values while the verdict uses one-sided ones (dossier 02 D8).

**6.7 Backtest (0.2 p).**
- Origins every step: Expanding (tests descriptive) or Rolling (tests valid).
- Each origin refits on `window.select(data.upto(origin))` (R9).
- Per-origin forecast RNG and fit seeds; a data salt keeps chains independent across windows and orderings.
- Scores are cached per origin, keyed by spec, data, outcomes, origin and settings, plus `fit_seed-v1` (and `integrated-v1` for D-34), so runs resume.
- The worker count does not change results [V: `test_spawn_pool_matches_the_serial_backtest`].
- Warm starts are opt-in and require increasing origins.

**6.8 `compare()` and the verdict (boxed rule; **Fig3**; 0.3 p).**

  better ⇔ exactly one differing axis ∧ ΔLPD > 0 ∧ |ΔLPD| > 3·MCSE,  MCSE = √(Σ_t (mcse²_{M,t} + mcse²_{B,t})),
           ∧ [(rolling ∧ one-sided DM-HLN p < 0.10) ∨ (M ∈ MCS_25% ∧ B ∉ MCS_25%)].

- The verdict is forced to "inconclusive" if either model converged at fewer than 95% of origins.
- A fit is converged when no chain raised, chains ≥ 2, EB converged (when used), and every primary quantity (D-33) has rank-normalised split R-hat (max of bulk and folded) < 1.01 and bulk ESS > 400 (`vehtari2021rank`).
- Multi-axis or unlabelled differences give "not comparable".
- Grid harmonisation: in a volatility grid the Constant baseline becomes MCMC Mode I `Constant('ig')`, not the analytic NIW model. Mode II is labelled a prior change.
- Forward reference: the rule's statistical properties are untested (§12 P2).

**6.9 Ordering sensitivity (0.2 p).**
- `rb.ordering_sensitivity(mode = prior | forecast | backtest; design = all | balanced Latin square | random)` recomputes everything data-dependent per permuted window.
- Noise floor σ_MC = max(sd across seeds, MCSE). A metric is flagged only when lpd_range exceeds 0.05 nats per variable (0.10 joint) and SNR > 3. Pooled orderings are optional.
- The R8 warning fires on every non-invariant MCMC fit with n ≥ 4.
- In Phase 2 the only order-invariant MCMC composition is reduced × Constant('iw_equivalent'), which therefore serves as a **negative control** (§11.9).

- **Evidence:**
  - dossier 02 §10, §11, §12, §15 D1–D4, D8;
  - `src/rai_bvar/forecast/result.py` l.96–288 (pre-D-34 anchors; re-anchor at release);
  - `src/rai_bvar/engines/mcmc/forecast.py`;
  - `src/rai_bvar/evaluation/{backtest,compare,scores,significance}.py`;
  - `src/rai_bvar/structural/{irf,ordering}.py`;
  - `tests/contract/test_forecast_contract.py`, `test_forecast_contract_mcmc.py`;
  - `tests/mcmc/forecast/test_lpd_mcmc.py::test_factored_lpd_equals_cond_chol_lpd`, `::test_ess_adjusted_mcse`;
  - D-34 tests at b9c66c0: `tests/unit/test_lpd_integrated.py`, `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`, `tests/unit/test_scores_lpd_method.py`, `tests/integration/test_backtest_lpd_integrated.py`;
  - `tests/integration/test_compare.py`, `tests/unit/test_grid_harmonise.py`, `tests/unit/test_significance.py`, `tests/integration/test_backtest*.py`, `tests/integration/test_ordering.py`;
  - DESIGN D-34 row (l.80) and §7.3;
  - NOTES 8b (main repo);
  - dossier 06 §9.4.
- **Bib keys:** `geweke2010comparing` `kruger2021predictive` `kish1965survey` `gneiting2007strictly` `gneiting2011comparing` `scheuerer2015variogram` `zanetta2024scoringrules` `newey1987simple` `diebold1995comparing` `harvey1997testing` `clark2001tests` `giacomini2006tests` `amisano2007comparing` `hansen2011model` `vehtari2021rank` `arias2023macroeconomic` `chan2024large` `chan2022asymmetric`
- **Floats:** **Fig3**, **Fig4** (main). TabC1 (the contract) is in App. C.

---

## 7. Using rai-bvar: a walk-through (2 pages)

**7.1 Data (0.3 p).**
- `VARData.from_frame(raw_levels, tcodes={...})`. Column order is the recursive ordering.
  - Gaps, duplicates and interior NaNs raise errors.
  - Leading and trailing incomplete periods are trimmed with a warning.
  - `upto`, `tail`, `select` and `realized` are provided.
- `rb.datasets.load('fred-qd', vintage='2026-09', variables='small'|'medium'|'large', tcodes='rai'|'official')` downloads the data, checks SHA-256 and caches it.
  - Files are not bundled. The named sets use only U.S. federal-agency series.
  - Interior publication gaps are interpolated with a listed warning (`fill_gaps=True`).

**7.2 Listing 1: `ex_a.py`, verbatim (0.7 p).** [V: verified at 62d15f6, bitwise re-run by the dossier 02 fact-check; REL re-run on the release commit]
- Data: a simulated 3-variable quarterly VAR stored as raw levels with tcodes {5, 5, 1}.
- Shown in order:
  1. Transform-aware own-lag means {0, 0, 1}.
  2. `BVAR(mean=Linear(prior=Minnesota(soc=1, sur=1)), lags=2)` resolves to "analytic", with EB over λ, soc and sur.
  3. Forecast quantiles on the model and level scales.
  4. `compare()` of RW, OLS-VAR(2) and BVAR(2) on rolling windows.
- The real outputs teach the design:
  - OLS-VAR(2) vs RW is "better" on the model axis: dLPD +14.45, MC s.e. 0.10, one-sided DM p 0.014, MCS separation.
  - BVAR(2) vs RW is "not comparable" because it differs along 7 axes (R2).
  - The table prints two-sided p-values; the verdict uses one-sided ones.
- The comment in `ex_a.py` says "valid DM/AG p-values". Keep the verbatim listing, and say in the text that the test is DM with the HLN correction (risk 10).

**7.3 Listing 2: `ex_b.py`, verbatim (0.7 p).** [V for the API; R for the MCMC numbers: one machine, bitwise only on the same platform]
- `BVAR(vol=SV(), lags=2, mcmc=MCMCSettings(4, 1000, 1000))` resolves to structural Mode I.
- EB comes from the conjugate twin (`eb_method 'conjugate_twin'`). `summary()` shows R-hat and ESS.
- The forecast container and the LPD API are identical to the analytic case (R3).
- The IRF shows a recursive zero impact for a shock to the rate, which is ordered last.
- A vol-axis comparison, `compare(Linear × Constant('ig') vs Linear × SV)`, on a homoskedastic DGP:
  - the SV arm fails the primary R-hat/ESS rule at all 11 origins;
  - the gate therefore forces "inconclusive";
  - this is intended: the gate refuses to certify an unconverged fit.

**7.4 Inspecting fits and failures (0.3 p).**
- `fit.plan.describe()` (FigC1 in App. C), diagnostics, warnings and timing.
- Failure reports: `MCMCNumericalError` with block, equation, iteration and conditional diagnostics.
- A vol-axis grid example with `rb.grid(BVAR, vol=[Constant(), SV(), SV(outliers=True)])` and `verdict(axis='vol')` is `[TC]`. Include it only if it is added to `run_examples.py` and run on the release commit.
- Do NOT print the stderr fold warning, whose text names other packages (TODO 16).

- **Evidence:**
  - dossier 02 §9, §13.1, §13.2 (scripts and outputs verbatim; fact-check re-run);
  - `src/rai_bvar/data/dataset.py`;
  - `src/rai_bvar/datasets/fred.py`.
  - The README quick start is stale (dossier 02 D13). Do not cite it.
- **Bib keys:** `mccracken2016fredmd` `mccracken2021fredqd` `giannone2015prior`
- **Floats:** **Lst1**, **Lst2** (main).

---

## 8. Validating the samplers (4.5 pages)

**8.1 Principle (0.2 p).**
- R6: no sampler is merged without passing Geweke and SBC tests. It is motivated by the CCM-2019 episode (`carriero2019large`, `bognanni2022comment`).
- A suite's "pass" is informative only if its power is known. Power is therefore measured by mutation (§8.11).

**8.2 Harness principles (0.6 p).**
- *Two-implementation rule (D-22).* Prior samplers, simulators and `log_joint` are written by work packages that wrote no kernel, and they never import kernel modules [V: `tests/validation/test_log_joint.py::test_references_never_import_kernels`, subprocess import test].
- *The algebra differs, not just the authors.* References are joint densities on natural parameters; kernels use conditional formulas.
  - For the SOC/SUR rows the reference is in covariance form (m × m systems, Matheron's rule; `journel1978mining`, `chiles2012geostatistics`; pathwise view `wilson2020efficiently`), while the kernels are in precision form (q × q Gram updates).
  - Mixture constants are transcribed a second time and must agree "on every printed digit" [V: `tests/validation/test_mixture_transcription.py`].
  - The rule exists because an earlier proposal reused the blocks' own prior samplers (erratum E9).
- *Complementarity, in one sentence:* an oracle shares any convention we copied, whereas Geweke and SBC share only our prior and simulator.
- *Test the production code path.* `JointSpec.step` is `prepare(data, prior)` followed by one production sweep, never a copy.
- *Configurations in which every term matters:*
  - proper, tight test priors;
  - M ≠ 0, so D3 centring is exercised;
  - outlier p ~ Beta(2, 18), so outliers occur at T = 20;
  - ν ∈ {5..12};
  - T_eff never a multiple of the 32-row bucket;
  - R9 recomputation off, so hyperparameters do not depend on y.
- *Failures fail and nothing is dropped.* A non-finite test function gives p = 0 and is counted. Only data-only filtering (finite ỹ ∧ max|ỹ| < 10³) is allowed, justified by the uniformity of ranks conditional on y [V: `tests/validation/test_geweke_selftest.py::test_non_finite_test_functions_fail_with_counts` and siblings].
- *Fixed seeds with a sequential wrapper.* Keys are `validation_key(seed, tier, crc32(name), chain)`.
  - The Gandy–Scott wrapper (`gandy2021unit`, Alg. 3): β₁ = α/k, γ = β₁^{1/k}, β_{i+1} = β_i/γ. Stage i fails if q_i ≤ β_i and passes if q_i > γ + β_i; otherwise a **fresh, independent** experiment, enlarged once by Δ, decides. P(reject) ≤ Σ γ^{i−1}β_i = α.
  - Per-PR settings α = 1e-3, k = 3, Δ = 4 give γ = 0.0693, β = (3.33e-4, 4.81e-3, 0.0693) and about 30% expected extra effort under H0 [R: N06 §4.6 table, recomputed there].
  - The global decision is Bonferroni on min p. Holm (`holm1979simple`) only lists the failing functions.

**8.3 Tier map, cadence and suite size (**Tab5**; 0.3 p).**
- Tiers U, B, G, S, A, P and M, each with its content and decision rule, plus the designed and the as-implemented cadence.
- As implemented: U and block-level G run per PR. Composition-level B, G-full, S, A3/A4/A7, SWA parity and M run in the CI job named "nightly", which runs **weekly**; write "weekly" [V: `.github/workflows/ci.yml`].
- Suite at 62d15f6: 4,964 parametrised test IDs (4,520 PR fast suite, 441 weekly, 3 bench) [V: collection reproduced by dossier 04]. b9c66c0 reports "Fast suite 4655 passed" [R]. Final counts come from `collect.txt` [REL].
- Markers and the `error::RuntimeWarning` filter are described here.

**8.4 Tier U: deterministic identities (0.3 p).**
- Density-ratio identity for block b with conditional q_b (`grosse2014testing`): log q_b(x₁) − log q_b(x₀) = log p(x₁, y) − log p(x₀, y), against the independent `log_joint`.
- MH identity: the computed log ratio before min(0, ·) equals log π(x′) + log q(x | x′) − log π(x) − log q(x′ | x). This includes the exact-mode w terms, the single-accept IG-σ² combination and the ASIS Jacobians. Tolerance: |LHS − RHS|/(1 + |RHS|) ≤ 1e-8.
- Dense oracles with injected noise: precision draw 1e-10, packed Gram 1e-13, tridiagonal SV 1e-12, CCCM vs a SUR oracle 1e-10.
- Tolerance lesson: two exact samplers of one Gaussian agree only to O(εκ), κ = cond. Path-vs-path envelopes are therefore 2e-12 + 5εκ(1 + t) [V: `test_linear_fastpaths.py`, `test_linalg.py`, `test_contemp_lstsq.py` docstring]. The "2,250 fresh problems" study is `[R]` (DESIGN §9.3 only; TODO 25).
- Infrastructure checks: float64 on every leaf, no-JAX import, typed keys, the key-reuse checker.

**8.5 Tier G: Geweke tests (0.6 p).**
- Statistic (`geweke2004getting`). Marginal-conditional draws: M₁ iid. Successive-conditional draws: C independent chains, each started at a prior draw with no burn-in.

  z_g = (ḡ_MC − ḡ_SC)/√(σ̂²_g/M₁ + s²_C/C),  p_g = 2P(t_{C−1} > |z_g|),  fail iff min p ≤ α/G.

- Methods points:
  - *Why prior-started chains suffice.* A prior-started chain is stationary from sweep 1 under H0, so the size needs only invariance, not ergodicity; ergodicity affects only power. Attributing the opposite concern to `talts2018validating` is `[TC]`; verify before writing (TODO 15).
  - *Why t_{C−1} critical values.* At G = 100 and α = 1e-3, the normal critical value has actual size 1.64e-3 at C = 200 and 1.11e-3 at C = 1000 [R: N06 §3.2].
  - *Test-prior design rule r = Var_prior/Var_post ≤ 5.* In the scalar conjugate model, IAT = 2r − 1. This is a recorded derivation and is **not verified for φ, σ, p and ν** [R]; the derivation is in App. B.6.
  - *Finite-variance rule.* Under the SV test prior E[y²] = ∞, so test functions are log-scale or bounded data functions, log λ is used instead of λ², and the Constant test prior is IG(5, 4).
  - Test functions also include the exact log-likelihood and Σ e²/λ.
- Sizes: G-small uses C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3; G-full uses C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2. Approximate `exact=False` rows are recorded but do not gate.
- Results on 15 PR compositions (TabD1) [V: reproduced at 62d15f6, M5 Max; REL]:
  - max |z| 3.03 against t_crit 4.10–4.50;
  - row 201 shows the sequential design: stage-1 q 0.0012 (|z| 4.21 on a log-scale data autocorrelation), then a pass at stage 2 with q = 1.0. This is deterministic on one machine for fixed code and seed, not a cross-machine reproduction;
  - minimum RNE 0.008–0.129, on intercepts, data functions and (zero_sv) h levels, not on φ or σ. Low RNE affects power, not size.
- Write "G-full runs weekly". Do NOT write "G-full green on all rows" until `geweke_full.jsonl` is committed [REL].
- Forward reference: the Geweke prior explores states ordinary data never reach (F1, §9.2).

**8.6 Tier S: SBC (0.4 p).**
- Credits: `cook2006validation`, `gelman2017correction`, `talts2018validating`, `modrak2025simulation`. The conditional variant `sailynoja2025posterior` is related work.
- Two-pass per-replication thinning:
  - the pilot chain starts from an independent prior draw;
  - t_i = min(⌈2·IAT_i⌉, t_max), using the Geyer IMS IAT (`geyer1992practical`) over the quantities and their tail indicators;
  - a fresh production chain then keeps L draws.
- Exact γ-statistic p-values (`sailynoja2022graphical`), computed by a forward recursion with tail accumulation [V: `tests/validation/test_uniformity.py`].
- ESS erratum: per-replication ESS cannot be certified from L = 99 kept draws. A pilot IAT predicts it, and a pooled kept-draw ESS gate with a Kish effective number of replications is added [V: design change]. The "76%" and "0.836L" numbers are docstring-only, `[R]` (TODO 25).
- t_max = 200 [R: `tests/sbc/test_sbc_compositions.py` docstring].
- Quantities include parameters, h at three dates, the total and per-equation log-likelihood, log p(θ), and the predictive unknowns h_{T+1} and y_{T+1} through the production forecast core.
  - Data-dependent quantities are needed: a posterior equal to the prior passes SBC on every data-free quantity (`modrak2025simulation`; check the theorem numbering, TODO 15).
- Config: N = 1000, L = 99, α = 1e-2, 22 compositions. Results are `[REL: sbc.jsonl]`. The only committed figure is row 205, min p 0.018 [V: NOTES 13].

**8.7 Tier B (0.2 p).**
- Block-conditional SBC (N = 200, L = 99) for exact-draw blocks, using ranks of the coordinates and of the exact log-likelihood.
- One-block invariance (N = 1000, L = 20 kernel applications, two-sample KS) for MH, ASIS and Gibbs-scan blocks.

**8.8 Tier A: analytic twins and exact oracles A1–A7 (TabD4 in App. D; 0.4 p).**
- A1: Chan's structural NIG log ML equals the Phase 1 log ML to 1e-10, with and without SOC/SUR.
- A2 / A3: MCMC Mode II × Constant('iw') against the Phase 1 NIW posterior, at n = 3 and at n = 20, p = 4, T = 300. A3 is the realistic size where Geweke/SBC are too expensive; its release result is `[REL]`.
- A4: Chan's asymmetric prior.
- A5: Mode I quadrature.
- A6: toy quadrature for the exact-h, ASIS and step-order kernels.
- A7: an exact system-wide joint draw against CTA.
- Methods details:
  - The CDF-form comparison statistic z = (F̂_MCMC(x_q) − q)/√(MCSE²(F̂) + q(1 − q)/N_ref) avoids the heavy-tailed order-statistic MCSE [R: `test_mcmc_twin.py` docstring; TODO 25].
  - Every twin has a negative control with measured power. A2 rejects one extra degree of freedom of the NIW posterior, and A5 rejects an IG shape off by ½ [V].

**8.9 Tier P: parity, neutral wording (TabD2; 0.3 p).**
- Statistic: z = (q̂_ours − q̂_ref)/√(mcse²_ours + mcse²_ref) over the 5/25/50/75/95% quantiles. Pass if |z| ≤ max(4, z_{1−0.01/(2m)}).
- *Analytic engine:* R BVAR 1.0.5 and BoE bvar 0.3.5.
  - Tests assert a log-ML rtol of 1e-8 [V]. Recorded agreement is < 1e-11 (BVAR, exact cases) and ≤ 1.5e-11 (BoE) [R].
  - Posterior moments agree to about 1e-9. EB modes agree to 1e-3. MH and forecast quantiles agree within MC error.
- *MCMC:* stochvol 3.2.1, via ZeroMean × SV (n = 1), main and default rows on four datasets: PR max |z| 1.63–2.59, nightly 1.64–3.57, bound 4.00 [R: `SV_PARAMETRIZATION.md` §6; tests committed].
- *Student-t SV:* our standardized-t sampler is checked against an exact importance-sampling posterior on a four-observation problem, max |z| 1.83 [V: `test_sv_t_matches_the_exact_importance_sampling_posterior`]. Methods point: when two implementations disagree, a tiny problem with an exact posterior decides.
  - No statement in the main text or in any caption about any other package's t sampler, or about the status of its fixture. App. D may mention it neutrally only after TODO 13.
- *Reduced form:* bayesianVARs was not run. The in-house exact system-wide Gibbs oracle gives max |z| 2.76 / 3.45 / 3.56 against bounds 4.05–4.51 [R]. Say plainly that the exit criterion is met against this substitute.
- *What parity does NOT establish:* there is no external VAR-level oracle for structural Mode I × SV, and no external oracle for exact mode beyond the reweighted stochvol rows.

**8.10 Coverage of the compositions the replication scores (0.2 p; blocking honesty statement).**
- The CR primary arm, **reduced-form Minnesota × SV(rw)**, and reduced × SV(rw, t) are **not** registered Geweke/SBC rows. `validation/joint.py` carries the random walk only in `structural_sv_rw`, row 14; the reduced rows 7–13 are AR-SV.
- They are also absent from the 12 contract-fit compositions in `tests/mcmc/forecast/mcmc_fits.py`.
- Their correctness currently rests on their validated parts:
  - the CCCM step (rows 7–13);
  - the RW block (row 14);
  - the t transitory component (rows 4 and 10);
  - D-34's MCMC tests on reduced CCCM fits [V: b9c66c0].
- Reduced × Constant('iw_equivalent') is row 11. AR-SV(0.9) shares the code path of row 8.
- TODO 3: register `reduced_sv_rw` (and the t variant) and add a contract fit before submission, then rewrite this paragraph.

**8.11 Tier M: the mutation suite (**Tab7**; FigD1 heatmap in App. D; 0.6 p).**
- Rules:
  - 18 mutants (M1–M18) are injected only through production seams: compose overrides, the vol slot, prior arrays, the simulator, the configuration.
  - A mutant is caught by a tier when the tier fails on the mutant **and** passes on the unmutated control, with common seeds.
  - `test_registry` requires every mutant to be registered with at least one confirmed catching tier.
- Coverage: U 14/18, G 18/18 (M7 with the coarse table, M11/M12 under the auxiliary sampler), B 4 (every mutant with a B check), S 1. That is 37 checks.
- Result: 37/37 caught with passing controls at 62d15f6 on Apple M5 Max, 177–182 s cold [V: reproduced by dossier 04; REL]. M1–M16 were re-run on x86-64 Linux, and M17–M18 were reproduced there ("4 passed") [R: `mutants.py` EVIDENCE header].
- Statistics agree across machines to two or three significant figures, e.g. M2: 702 / 2.48 vs 705 / 2.47 [R].
- Credits: `demillo1978hints`, `jia2011analysis`; for probabilistic programs `dutta2018testing`. The contribution is curating algorithm-level mutants, several taken from published errors, and gating releases on them. "To our knowledge" only after TODO 11.
- Lessons:
  1. **The four mutants without a U check** (M2, M3, M5, M16) are the non-local errors:
     - step order (M2, the stale indicators of `primiceri2005time` Alg. 1, corrected by `delnegro2015time`);
     - simulator (M3, Geweke's "error 2": G 23.5 vs 2.59);
     - sampler-vs-model prior mismatch (M5);
     - model approximation (M16).

     Every conditional-formula error has a deterministic U catch. So run both identity and joint tests. Never write "U cannot see them".
  2. **Exactness changes what counts as a bug.** M11 and M12 are bugs of the auxiliary sampler only. M7 is invisible with OCSN-10 and caught with a one-component COARSE1 table. The general device: degrade an approximation on purpose to test its exact correction [V: `test_sv_core.py::test_block_geweke_exact_mode_with_coarse_mixture`].
  3. **The log-likelihood as a test quantity** catches M1 (CCM-2019) in block SBC: p 2.8e-11 vs control min p 0.10 [V]. Whole-scan invariance does not catch it (q 0.29) [R].
  4. **Measure-preserving moves.** M6 takes |σ| without flipping h̃ and is caught by U (1.94 / 0) and G (7.4 / 2.34). Only the joint fold (σ, h̃) ↦ (−σ, −h̃) preserves the non-centred posterior (`fruhwirth2010stochastic`). Neutral wording; no package named.
  5. **Wrong dummy weights are loud:** M17 23.1 vs 1.94 (control on the dense eigen path; 2.44 on cholesky_const), and M18 99.7 vs 2.61 [V].

**8.12 Runtime convergence diagnostics and the gate (0.3 p).**
- Primary sets (D-33) at n = 20: 41 (Constant), 101 (SV), 61 (RW SV), 101 (SV-O-t) quantities [V: `test_monitored_set_sizes_at_n20`]. Secondary sets never gate.
- Estimators: rank-normalised split R-hat (max of bulk and folded), Geyer IPS/IMS ESS (`geyer1992practical`), tail ESS and MCSE (`vehtari2021rank`; batch-means background `flegal2010batch`).
- Tested against hard-coded ArviZ reference values (`kumar2019arviz`) to rtol 1e-10, plus AR(1) known-answer tests [V].
- What the gate does in practice:
  - the pilot predicted that SV fits at the defaults routinely fail through φ and σ [R];
  - BENCH-P2 has converged = False in all 6 runs [R: `baseline.json`];
  - in the first CR run the SV arms converged at 37–68% of main-stage origins, and SV(rw) at 57–82% across orderings [FULL] (§11.7).

**8.13 Cost (0.1 p).** Recorded on one machine, cold cache, under load: G-small for all PR rows takes about 95–100 s,
the mutation suite about 177–182 s, and the PR fast suite about 1,000 s [V: reproduced at 62d15f6; REL].

- **Evidence:**
  - dossier 04 §1–§14, §17;
  - dossier 01 §5;
  - `src/rai_bvar/validation/` (`joint.py`, `geweke.py`, `sbc.py`, `uniformity.py`, `sequential.py`, `reference.py`, `dummies.py`);
  - `tests/geweke/test_geweke_compositions.py`;
  - `tests/sbc/test_sbc_compositions.py`, `test_block_tier.py`;
  - `tests/analytic/test_mcmc_twin.py`, `test_mode1_quadrature.py`;
  - `tests/mcmc/blocks/test_sv_toy_quadrature.py`, `swa_oracle.py`;
  - `tests/parity/{test_r_bvar,test_boe_bvar,test_stochvol,test_bayesianvars}.py` with `*_PARAMETRIZATION.md` and `BOE_BVAR_NOTES.md`;
  - `tests/mutation/mutants.py` (`_ALL` l.767–826, EVIDENCE l.834–858), `tests/mutation/test_mutants.py`;
  - `tests/mcmc/diagnostics/test_monitor.py`, `test_diagnostics_ext.py`;
  - `.github/workflows/ci.yml`;
  - NOTES B.3–B.8, D.13.
- **Bib keys:** `carriero2019large` `bognanni2022comment` `journel1978mining` `chiles2012geostatistics` `wilson2020efficiently` `gandy2021unit` `holm1979simple` `grosse2014testing` `geweke2004getting` `talts2018validating` `cook2006validation` `gelman2017correction` `modrak2025simulation` `sailynoja2025posterior` `sailynoja2022graphical` `geyer1992practical` `chan2022asymmetric` `kastner2016dealing` `hosszejni2021modeling` `kuschnig2021bvar` `labonne2026bvar` `gruber2026bayesianvars` `demillo1978hints` `jia2011analysis` `dutta2018testing` `primiceri2005time` `delnegro2015time` `fruhwirth2010stochastic` `vehtari2021rank` `flegal2010batch` `kumar2019arviz` `kish1965survey`
- **Floats:** **Tab5**, **Tab7** (main). TabD1–TabD4, FigD1 and FigD2 are in App. D.

---

## 9. Numerical robustness: what the validation and replication runs found (1.5 pages)

**9.1 The pattern (0.15 p).**
- All three states are legitimate under the model, and none came from a user report.
- Assembly is the lossy step: the factorisations were correct for the matrices they received, but the information was lost when the matrix was formed. The fixes factor the data, never the assembled matrix.
- "Failures are values, no jitter" made the failures visible. That jitter would have biased results is a counterfactual, stated only as an argument.
- Status codes decode as 1003 = EQ_CONTEMP, equation 2, and 2001 = EQ_VOL, equation 0.
- The summary table of F1–F3 is TabE1 in App. E.

**9.2 F1: a Geweke chain in the test prior's tail (0.35 p).**
- *Where:* G-full `reduced_sv`, chain 3214 of 4000, sweep 194. The chain sat at h ≈ 31–37, so **|y| ≈ 2e8** [R: commit 3d3f51b; test docstring l.4].
- *Mechanism:* equation 2's a-row precision has exact eigenvalues 14.8 and 1.9e17 (cond 1.3e16), so forming Z'WZ loses the small one and the pivot is 0 (status 1003). The conditional itself is benign: mean (−0.17, −0.66), sd (0.20, 0.16) [R: NOTES 10].
- *Fix:* Householder QR of [diag(√prec); √w ⊙ Z] with diag(R) > 0. R is the upper Cholesky factor of K, so the noise map is the same.
- *Evidence:*
  - [V] `test_contemp_lstsq.py`: equality within the envelope at 20 states × 5 compositions; an extreme-scale state against a 60-digit exact conditional; the slow `::test_g_full_reduced_sv_failing_chain_is_fixed` replays chain 3214.
  - [R] The QR draw at the real state matches the exact conditional to 1.6e-8 sd.
- Narrowing the test prior was rejected because it would hide a real weakness and drop high-persistence coverage.
- Credit the normal-equations vs QR conditioning argument to `higham2002accuracy` and `bjorck1996numerical`. This is an application, not an algorithm.

**9.3 F2: the RW SV AWOL draw at σ = 4.98e-9 (0.45 p).**
- *Where:* in the first full CR attempt's ordering stage (8a622c5): SV(rw), reduced form, origin 1987Q2, ordering GDP, TBILL, INFL, UR, burn-in sweep 318.
- *Why the state is valid:* the non-centred prior has positive density at σ = 0, and the NC move of the same sweep returned σ to 0.031.
- *Mechanism:* σ²D ≈ 1e-17 is below the ulp of the Laplacian diagonal (2). The assembled matrix is therefore the singular path Laplacian, and the level pivot ≈ σ²(1/V₀ + ΣD) ~ 1e-15 came out 0 (status 2001).
- *Fix:* the excess-pivot recursion (§5.5) is the same factorisation without cancellation.
- *Evidence:*
  - [V on Apple M5 Max, against a 60-digit reference] `test_sv_core.py::test_rw_awol_at_a_tiny_sigma`, `::test_rw_step_at_a_tiny_sigma_is_ok`, `::test_rw_integrated_likelihood_at_a_tiny_sigma`, `::test_awol_rw_laplacian_is_the_rw_system`. A Linux re-run is TODO 23.
  - The old path at the test state gave min pivot 2.22e-16 vs an exact 5.308e-15, logdet −36.04 vs −32.8695, and h_0 −73.90 vs −3.024 [R: dossier 05 §2.3, scratch computation]. TabE1 recomputes these with `tab_rw_pivots.py`.
  - The RW `aux_log_likelihood` had the same flaw: 17534.8 instead of −1650.8 [R: commit 4fc0162]. The fix is [V].
  - Invariance: 639 of 656 arrays are bitwise identical (every non-RW composition) [R, one machine].
- *Positioning:* an instance of GTH-type subtraction-free elimination for diagonally dominant M-matrices (`grassmann1985regenerative`, `alfa2002accurate`, `ye2008computing`), within the GMRF precision-sampler setting (`rue2001fast`). Claim the application, not the algorithm.
- The fix removes the crash, not the slow mixing near σ → 0.
- The "silent wrong draw on the old path" observation (a positive rounding-noise pivot) stays out until a committed test exists (TODO 23).

**9.4 F3: the replay of F2 itself crashed (0.1 p).** It raised `TracerArrayConversionError` on host arrays inside the
production CCCM step, so the first crash's diagnostics were lost. Fixed in 93f2bcd, with a test that runs the
production program [V: `test_failures.py::test_replay_of_an_equation_failure_after_the_production_cccm_step`].
Lesson: test the failure path with the production program.

**9.5 SOC/SUR dummy rows on MCMC (0.2 p).**
- In Mode II the λ step must count the rows. A hand-injected omission was caught by the analytic twin and not by the dense oracles [R: NOTES 12]. The count "33 of 165" is withheld until the injection is committed (TODO 24).
- There is a closed-form λ marginal and an exact rejection envelope under iw_equivalent [V: NOTES 13; proof in App. B.5]:
  - the bound: g(λ) = (|S₀|/|S(λ)|)^{½} exp(−½(q − q∞)) ≤ 1;
  - KS against quadrature passes, Geweke row 205 gives G-small 2.24 and G-full 2.76, and SBC row 205 gives min p 0.018.
- The O(q²) eigen fast path survives the dense prior precision [V]. Its timing is under load and goes to App. E only.

**9.6 Notes for other implementations (0.15 p; names no package).** Generic lessons for any implementation of
the same algorithms, each phrased as "implementations that do X should check Y":
- formed Gram matrices lose small eigenvalues at extreme scales;
- scaled RW-SV tridiagonal systems cancel as σ → 0;
- a non-centred sign fold must flip (σ, h̃) jointly to be measure-preserving;
- path-conditioned RB log scores degrade in the tails at h > 1.

No version-specific observation about any package appears in the paper before TODO 13.

**9.7 Platform caveats (0.1 p).**
- On macOS with numpy/scipy on Accelerate, `slogdet`/`det` raised spurious RuntimeWarnings on well-conditioned SPD tridiagonals. The oracle now uses the Cholesky factor (dd8244e) [R].
- A Phase 1 EB test intermittently crashes the interpreter on macOS [R; root cause unknown; not tied to the MCMC code].
- Linux CI is the reference platform. Do not suggest that Accelerate is at fault.

- **Evidence:**
  - dossier 05 §1, §2.1–§2.6, §3, §4.1–§4.5, §5.6, §5.7, §6, §9;
  - NOTES C.10, D.12–D.16, E.17;
  - commits f60d736, 4fc0162, 93f2bcd, 3d3f51b, dd8244e;
  - `src/rai_bvar/validation/dummies.py`;
  - `tests/validation/test_reference_dummies.py`.
- **Bib keys:** `higham2002accuracy` `bjorck1996numerical` `grassmann1985regenerative` `alfa2002accurate` `ye2008computing` `rue2001fast` `journel1978mining` `chiles2012geostatistics` `wilson2020efficiently` `geweke2004getting` `fruhwirth2010stochastic`
- **Floats:** none in the main text. TabE1 and FigE1 are in App. E.

---

## 10. Performance (0.75 page)

**P1: target (0.2 p).** BENCH-P2:
- model: structural Linear × SV, n = 20, p = 4, T = 300 (a simulated VAR(4)-SV);
- run: 4 chains × (1000 burn + 10,000 draws), fixed hyperparameters, one CPU device;
- timing: end to end, including compilation, storage and diagnostics; medians over 3 cold and 3 warm fresh-process runs;
- pass if under 120 s.

**P2: result (0.35 p)** [R: `tests/bench/baseline.json`, measured at deb3929 on an Apple M5 Max at load 2–6, not idle; REL on an idle machine]:
- 52.3 s cold, 49.6 s warm;
- compile 3.46 s cold, 1.00 s warm;
- 3.90 ms per sweep;
- peak RSS 3,043 MiB.

Mixing: the slowest quantities are σ_i and φ_i (min ESS/s 10.2 and 11.2), against about 400 for the own-lag
coefficients. converged = False in all 6 runs; according to commit d490730, 4 of 101 primary quantities miss
R-hat 1.01. Full detail is in TabE2.

**P3: not measured on the production engine (0.2 p).**
- The 4-device stretch target.
- The reduced-form CCCM.
- SV(t) and SV-O-t.
- An idle x86 machine; the only x86 baseline is provisional and loaded.

Three sources of numbers (DESIGN stand-ins, pilot prototypes, the production baseline) are never blended, and
only the production baseline is quoted. Kernel microbenchmarks are in TabE3.

- **Evidence:**
  - dossier 03 §13.1–§13.10, §14 items 7–8;
  - `tests/bench/baseline.json`;
  - `benchmarks/phase2_perf.py`;
  - `tests/bench/test_perf_phase2.py`;
  - commits d490730, deb3929.
- **Bib keys:** `bradbury2018jax`
- **Floats:** none in the main text. TabE2 and TabE3 are in App. E.

---

## 11. Illustration: a pre-registered replication of Clark and Ravazzolo (2015) (4 pages)

**Number discipline for the whole section:**
- every acceptance-run number is `[ACC: <json path>]`, filled from `benchmarks/reports/clark_ravazzolo_acceptance.json`. JSON horizon keys are strings ("1", "2", "4", "8");
- first-full-run numbers appear only as labelled history, `[FULL: <key>]`;
- diagnostic numbers are `[PREREG]`;
- every number states which estimator produced it.

**11.1 Role (0.15 p).**
- This is a Phase 2 exit criterion: a **qualitative** ranking replication ("SV beats homoskedastic VARs in density forecasts"), not a number-for-number one.
- It exercises the whole stack on real data: pinned, checksummed data; the composition API; the backtest; scores; `compare()`; convergence diagnostics; `ordering_sensitivity`.

**11.2 Target (0.25 p).**
- Paper: `clark2015macroeconomic`. CR's numbers come from the working paper's Tables 1–3 (`clark2012macroeconomic`), as hard-coded in CR_FULL and CR_PRE_DLPS (l.147, l.167). The JAE tables were not accessed; say so (TODO 22).
- CR's VAR: Minnesota prior with θ = 0.2, 1/l² decay and a normal-diffuse Φ (`kadiyala1997numerical`).
- CR's VAR-SV: RW log-volatilities sampled by the JPR single-move step (`jacquier1994bayesian`). The variants are AR-SV and SV-t with fixed d.
- CR do not state a variable ordering, so all 24 orderings are run.

**11.3 Data (0.3 p; TabF2 in App. F).**
- Source: FRED-QD/MD vintage 2026-09 (`mccracken2021fredqd`, `mccracken2016fredmd`), SHA-256 checked.
- Series: GDPC1 and GDPCTPI as 100Δlog, non-annualised; UNRATE and TB3MS in levels.
- Estimation starts in 1965Q1 (precedent `clark2011realtime`).
- Pseudo real time on the final vintage, instead of the real-time vintages (`croushore2001realtime`).
- Run-time assertions:
  - quarterly aggregation of the monthly series [FULL: max |diff| 3.3e-5; ACC: `data_checks.aggregation`];
  - bitwise no look-ahead at 1985Q1, 2007Q4 and 2011Q2, **for the primary arm only** [ACC: `data_checks.in_window`];
  - checksums.

**11.4 Models (0.3 p; TabF3 in App. F).**
- Mean prior: `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, alpha=2, cross=1, mean=0)`, p = 4, sampled with CCCM (`carriero2022corrigendum`).
- Arms:
  - `Constant('iw_equivalent')`, the baseline (`chan2022asymmetric`);
  - SV(rw), the primary arm;
  - SV;
  - SV(rw, t10), with ν fixed at 10 (CR's fixed d), so its grid-bound warning fires by design;
  - AR-SV(0.9), with (φ + 1)/2 ~ Beta(71.25, 3.75), matching CR's a₁ ~ N(0.9, 0.05²) in mean and sd;
  - a structural pair: Mode I × Constant('ig') and × SV(rw).
- The library-default SV priors are used; CR's IG(1/2, 0.0175) is not adopted. "No prior was tuned."

**11.5 Evaluation design (0.4 p).**
- Expanding window over 106 origins (1985Q1–2011Q2), with a 92-origin pre-crisis subset; h = 1, 2, 4, 8.
- DM/HLN p-values are **descriptive** under an expanding window, so any "better" must come through the MCS route (§6.6). Report the route for each verdict.
- Scores:
  - *primary:* the RB LPD, path-integrated in the acceptance run;
  - *secondary:* the path-conditioned LPD, never used in a rule (`dlps_conditioned`, `dlpd_conditioned`);
  - *parity metric only:* the Gaussian-approximation LPS (`adolfson2007forecasting`; it equals −½ the Dawid–Sebastiani score, `dawid1999coherent`, which is proper but not strictly proper, `gneiting2007strictly`). It is reported side by side with the RB score, with the `rb_gauss_disagree` count per cell;
  - also CRPS and RMSE.
- `compare()` verdicts are reported with the gate on and off.
- The RB-degenerate flag (joint `lpd_mcse` ≥ 0.9; see the bound in §6.4) is descriptive. Flagged origins are **kept**, because dropping exactly the tail origins would bias the comparison.
- Ordering stage: `ordering_sensitivity(mode='backtest', design='all')` for SV(rw) and Constant, with `noise_reps = 1`. The order-invariant Constant arm is the **negative control**.

**11.6 Criteria (0.25 p; **Tab12**).**
- Print the N08 qualitative wording next to the operational rule as coded, verbatim from `criteria[*].rule` (code l.666).
- Say that the operationalisations are the project's own: "about 1" → 1.02; "systematically" → ≥ 3/4 joint horizons or ≥ 8/16 cells.
- Say that C4's coded rule compares h = 8 with h = 1 only, while N08's wording is "shrinks or turns negative".

**11.7 The first full run and what it showed (history; 0.5 p).** The run history (quick → 8a622c5 crash → 93f2bcd,
f60d736, 4fc0162 → 4bb978a / 551f181 → b9c66c0 → c823d20 → acceptance) is TabF1 in App. F.
- Settings: 4 × 5000, burn 1000, 1000 forecast draws, path-conditioned LPD.
- Outcome [FULL: `criteria[*].met`]: criteria 1, 2, 3 and 5 were met; 4 and 6 were not.
- Criterion 4 failed on one cell, full-sample UR: h = 8 +1.954 > h = 1 +0.159.
  - It was driven by 2007–08 origins: the three largest UR h = 8 origins contribute +41.6, +41.4 and +40.4 nats, and 12 origins are RB-degenerate.
  - RB and Gaussian ΔLPS differ in sign in 28 marginal cells and 8 joint cells, all at h ≥ 2.
- Criterion 6 failed on the gate alone: "better" in 0/24 orderings gated vs 24/24 ungated.
  - SV(rw) converged at 57–82% of origins across orderings; Constant converged at 100%.
  - Typical misses were one σ quantity with R-hat 1.012–1.020 and bulk ESS 537–743.
- The earlier attempt (8a622c5) had aborted in the ordering stage on F2 (§9.3).
- Possible causes of inflation gains 1.6–3.2× CR's are listed as hypotheses only: final-vintage GDPCTPI and the iw_equivalent baseline.

**11.8 Diagnosis and the pre-registration (0.75 p).**
- (a) **Fit-level convergence diagnostic** at the main-stage fit seeds, with no forecasts or scores [PREREG]:
  - SV(rw), identity ordering: 72/106 at 4 × 5000; 79/106 with burn 5000; 92/106 with 8 chains; 106/106 at 4 × 20,000 (max R-hat 1.0061, min bulk ESS 1763);
  - the two worst orderings go from 71 and 63 to 106/106;
  - all 520 + 424 diagnostic fits ran without error, and every base-budget failure was on a σ quantity with R-hat < 1.03.
  - Seed datum, the evidence of R-hat noise at the threshold: the same models and ordering converged at 72 (main-stage seeds) vs 69 (ordering-stage seeds) of 106 origins, with joint h = 1 dLPD +0.809 vs +0.838 [FULL; PREREG].
  - Lesson: plan the MCMC budget with fit-only diagnostics before any score exists. The script and its output are not committed (TODO 4).
- (b) **The estimator:** D-34 (§6.4).
- (c) **8000 forecast draws:** at the crisis cell the 1000-draw thinning error had sd 0.75–1.06 nats and a bias of 0.3–0.6; at 8000 the sd is 0.09–0.19 and the bias ≈ 0. About 1.3 nats of MCMC error remain [PREREG].
- What changes, and only this: 4 × 20,000 kept draws with burn 1000; the integrated primary LPD; 8000 forecast draws.
- What does not change: data, models, priors (CR's σ² prior explicitly not adopted), origins, horizons, seeds, the 24-ordering stage with its noise replications, the rules, the 95% gate and the descriptive RB-degenerate flag.
- Quote the non-blind framing verbatim: "These changes are motivated by that failure analysis, so the run is not blind to the previous results. What is pre-registered is that the settings below were fixed before any score of the new run existed, and that the criteria are evaluated on them as coded, whatever the outcome."
- Stated in advance:
  - criterion 4 may still fail on UR (review estimate +0.23 vs +0.157);
  - criterion 6 is expected to pass if the gate is met;
  - no further run, except to fix a bug.
- Provenance:
  - the pre-registration was committed in c823d20, and the run was launched afterwards from a detached worktree at that commit;
  - the report qualifies as the acceptance run only if `config.name == "acceptance"`, `config.lpd_method == "integrated"`, `config.draws == 8000`, `config.seed == 0`, `versions.git_commit` = c823d20…, `versions.git_dirty == false`, and there is no `pending` key [ACC: `config`, `versions`] (TODO 2).

**11.9 Results of the acceptance run (0.75 p; everything is a placeholder).**

*Opens with the reading rules fixed before the run* (dossier 06 §9.5):
- **A**, changes because of the estimator: every RB ΔLPS/ΔLPD at h ≥ 2, the DM p at h ≥ 2, the sign-disagreement cells, the C4 cells, the variant comparisons at h ≥ 2, and ordering `lpd_range`/SNR at h ≥ 2.
- **B**, MC noise only: all h = 1 LPD numbers, CRPS and RMSE ratios, Gaussian scores, and DM p at h = 1. The estimators coincide at h = 1; quick → full moves were ≤ 0.003 per marginal cell [FULL/R].
- **C**, changes because of the budget: convergence shares, gate-on verdicts, criterion 6's gate-on count, the MC s.e. of totals, and runtime.
- **D**, unchanged by design: data checks, models, priors, origins, rules, fit seeds.

*Then:*
- P1, criteria (Tab12: first full run vs acceptance) [ACC: `criteria[*].met`, `criteria[*].detail`].
- P2, headline SV(rw) vs Constant (**Tab13**) [ACC: `relative.full['SV(rw)'].variables[v][h].{dlps, dlps_conditioned, dlps_gauss, rb_gauss_disagree, rb_degenerate, dm_p_better, crps_ratio, rmse_ratio}` and `.joint[h].{dlpd, dlpd_conditioned, dlpd_gauss}`], with the CR columns from CR_FULL.
- P3, pre-crisis (TabF4) [ACC: `relative.pre_crisis['SV(rw)']`]. Flag the GDP h = 8 cell, which C4 needs to be positive (first run: RB +0.108, Gaussian −0.004) [FULL].
- P4, all arms (TabF5) [ACC: `relative.full[arm].joint`, `converged.full`, `verdicts.full`, gate on/off, MCS route].
  - The diagnostic covered only SV(rw), so SV, SV(rw, t10) and AR-SV(0.9) may stay "inconclusive". Say so before the numbers.
- P5, ordering robustness at h = 1 (**Fig6**) [ACC: `orderings.per_ordering[*].{dlps_h1, dlpd_h1, dm_p, dm_p_joint_h1, converged_sv, converged_constant, verdict_h1, verdict_h1_ungated}`; `orderings.results[*].{material, summary}`].
  - Report gain/spread. First-run reference: joint h = 1 range 0.083 nats against a gain of 0.809; gain/range 2.2–7.8 per variable [FULL].
  - Negative control: is the Constant arm flagged "material" [ACC: `orderings.results['Constant'].material`]? If so, report it as a limitation of the one-replicate noise floor (§12 P5).
- P6, convergence before and after (FigF2).
- P7, per-origin conditioned vs integrated scores at UR h = 8 (FigF1) and cumulative joint ΔLPD (FigF3). Both need the per-origin export (TODO 5); drop them if it is not committed.
- P8, interpretation, written in branches:
  - if C4 fails, report it as a difference from CR attributed per the pre-registration, without tuning;
  - check that the h = 1 numbers moved by MC noise only (group B);
  - report the run-level warnings [ACC: `warnings`].

**11.10 Caveats a referee will raise (0.35 p).**
- real-time vs final data;
- WP vs JAE numbers;
- the baseline is not CR's normal-diffuse VAR;
- SV priors;
- significance is descriptive, and verdicts come through the MCS;
- self-chosen operational thresholds;
- a non-blind pre-registration;
- one backtest seed;
- the estimator change between runs (say which estimator produced which number);
- the primary arm reduced × SV(rw) is validated through its parts only (§8.10).

- **Evidence:**
  - dossier 06 §0–§12 and Open questions;
  - `benchmarks/clark_ravazzolo.py` at c823d20: CONFIGS l.186–206; CR_FULL l.147; CR_PRE_DLPS l.167; relative_table l.479; criteria l.666; RELIABILITY_NOTE l.953 and _INTEGRATED l.976; acceptance_gaps l.1022; DEVIATIONS l.1380; report dict l.1530–1570;
  - `benchmarks/reports/clark_ravazzolo.{md,json}` (551f181; history);
  - `benchmarks/reports/clark_ravazzolo_preregistration.md` (c823d20);
  - `benchmarks/reports/clark_ravazzolo_acceptance.json` (to be committed);
  - `docs/phase2/08_common_sv_and_ordering.md` Part C (N08 §C1–§C6);
  - GitHub issue #3 (run log; recorded);
  - NOTES 8b, 17;
  - `tests/unit/test_clark_ravazzolo_script.py` (c823d20).
- **Bib keys:** `clark2015macroeconomic` `clark2012macroeconomic` `kadiyala1997numerical` `jacquier1994bayesian` `mccracken2021fredqd` `mccracken2016fredmd` `clark2011realtime` `croushore2001realtime` `chan2022asymmetric` `carriero2022corrigendum` `adolfson2007forecasting` `dawid1999coherent` `gneiting2007strictly` `kruger2021predictive` `diebold1995comparing` `harvey1997testing` `clark2001tests` `giacomini2006tests` `hansen2011model` `vehtari2021rank` `arias2023macroeconomic` `chan2024large` `carriero2016common`
- **Floats:** **Tab12**, **Tab13**, **Fig6** (main). TabF1–TabF8 and FigF1–FigF3 are in App. F.

---

## 12. Limitations (1.75 pages; mandatory)

**P1: scope (0.25 p).**
- Phases 0–2 only. The mean axis is linear, and the prior axis is the Minnesota family.
- Roadmap only:
  - Phase 3: global-local priors, CommonSV, conditional forecasts;
  - Phase 4: BART, GIRFs, PSIS-LFO (`burkner2020approximate`);
  - Phase 5: TVP, threshold and Markov switching, factor SV, order-invariant SV;
  - Phase 6: structural identification beyond recursive.
- Literature pointers: `huber2022inference`, `clark2023tail`, `gruber2025forecasting`, `kastner2019sparse`, `chan2024large`, `bertsche2022identification`.
- "Plug-in without editing core files" is untested.
- No in-sample missing values, no mixed frequency, and no shipped real-time vintages (API hooks only).

**P2: the verdict engine (0.3 p).**
- The size and power study on simulated DGPs (SPEC §10.5) is **not implemented** (no `tests/dgp/`), so no false-positive rate is claimed. Only DM size/power, MCS behaviour and verdict-logic unit tests exist.
- The MCSE of ΔLPD sums the two models' per-origin MCSEs in quadrature, although both models share the per-origin forecast seed. Whether 3 MCSE is conservative is unknown.
- The MCS route is accepted under expanding windows, where DM is descriptive; its validity there is open.
- The verdict is per horizon, with no multi-horizon rule.
- The DM long-run variance is rectangular-kernel with a Bartlett fallback, not the Newey–West that SPEC states (`newey1987simple`).

**P3: the convergence gate (0.2 p).**
- At the default budget, SV fits routinely fail R-hat 1.01 through φ and σ (pilot; BENCH-P2; the first CR run).
- Near the threshold, converged counts move by several origins with the seed (72 vs 69 of 106) [FULL; PREREG].
- The gate is strict by design, so users need larger budgets: 4 × 20,000 in the acceptance run [ACC: `converged.full`]. The default MCMCSettings (2000 draws) are unchanged; the policy for defaults is open (TODO 33).

**P4: log scores (0.2 p).**
- The path-conditioned estimator remains the SPEC default at h > 1.
- The integrated estimator is opt-in and limited to MCMC linear and zero-mean forecasts. It keeps residual MC error at tail origins (about 1.3 nats per arm at a crisis origin), which the delta-method MCSE understates.
- All scores are on the transformed scale.
- The MC noise of the energy/variogram subsample is not reported.
- There is no SBC of y_{T+h} at h > 1.

**P5: the ordering diagnostic (0.1 p).**
- The noise floor rests on one replicate (`noise_reps = 1`).
- In the first full run it flagged the order-invariant Constant arm as material at h > 1 [FULL]; see also [ACC: same field].

**P6: validation evidence (0.35 p).**
- Geweke and SBC run at n ≤ 3. The realistic size (n = 20) is covered only by analytic twins, and only for Constant compositions.
- There is no external VAR-level oracle for structural Mode I × SV.
- The CR primary arm reduced × SV(rw), and reduced × SV(rw, t), have no registered Geweke/SBC row and no contract fit (§8.10). The contract tests cover 12 MCMC compositions.
- G-full, SBC, composition-level B and weekly results have no committed run records at 62d15f6 [REL].
- The r ≤ 5 rule is unverified for φ, σ, p and ν, and low RNE lowers power for some functions.
- Power is measured only against 18 curated mutants.
- Whether the OCSN-10 approximation is visible at G-full with unknown SV parameters is undetermined.
- bayesianVARs parity is replaced by a substitute oracle. Third-party parity is on pinned versions (stochvol 3.2.1, BVAR 1.0.5, BoE bvar 0.3.5).

**P7: numerics, performance and platforms (0.25 p).**
- The other formed-Gram draws (the structural coefficient step, the CCCM Π step) still gate on pivot > 0 with no relative pivot floor. Whether reachable states expose them is unknown.
- Slow mixing of the RW σ near 0 remains.
- Performance evidence comes from one non-idle Apple M5 Max. There is no idle x86 measurement and no multi-device production timing.
- Bitwise reproducibility holds only within one platform and version set. Chunk invariance is tested with toys. The prefix test is bitwise on macOS and to rounding on x86.
- macOS Accelerate caveats (§9.7).

**P8: release state (0.1 p).** Version 0.0.1, pre-alpha. The README is to be updated, and the PyPI release is TODO 19.

- **Evidence:**
  - dossier 01 §7, §8.2, §8.4, Q4, Q7, Q11;
  - dossier 02 §12.4, §15, §17;
  - dossier 03 §13.8, §15;
  - dossier 04 §6.7, §12.5, §15;
  - dossier 05 §5.7, §7;
  - dossier 06 §7.7, §9.3, §12.
- **Bib keys:** `burkner2020approximate` `huber2022inference` `clark2023tail` `gruber2025forecasting` `kastner2019sparse` `chan2024large` `bertsche2022identification` `hansen2011model` `giacomini2006tests` `newey1987simple`
- **Floats:** none.

---

## 13. Conclusion, checklist and roadmap (1 page)

**P1 (0.2 p).** Restate the headline:
- what the package lets an applied forecaster do: compose, fit, score, and get a verdict that refuses to credit volatility gains to the mean or prior, with the evidence behind the samplers;
- the replication's answer [ACC: one clause]. No outcome-dependent claim beyond this slot.

**P2: checklist (0.5 p).** Merged from the forecaster and implementer lists; each item points to its section.
1. Write kernels as core(noise), with exposed conditionals and acceptance ratios (§4.4).
2. Write the references independently and enforce this with an import test (§8.2).
3. Check an identity at a joint draw for every conditional and every MH step. Run prior-started Geweke chains and SBC with data-dependent quantities, including the exact log-likelihood (§8.4–§8.6).
4. Measure power with mutants drawn from published errors, and degrade approximations on purpose to test their exact corrections (§8.11).
5. Treat failures as values, never jitter, and fix kernels rather than test priors (§9).
6. Gate verdicts on convergence at every origin. Plan budgets with fit-only diagnostics before any score exists (§6.8, §11.8).
7. Do not read the delta-method MCSE of an RB log score as a bound on its tail error; inspect the weight ESS. At h > 1, integrate the simulated path out and use enough forecast draws (§6.4).
8. Report RB and Gaussian scores side by side, the ordering spread next to the gain, and an order-invariant negative control (§11.5, §11.9).
9. Treat DM p-values under expanding windows as descriptive, and say which route produced each verdict (§6.6).
10. Pre-register settings as code, say how blind the pre-registration is, and make no-look-ahead checkable (§11.3, §11.8).

**P3: roadmap and availability (0.3 p).**
- Roadmap by phase (SPEC §13), with no promises of results. The next priority is the verdict-engine DGP size/power study, then the Phase 3 prior axis, so that a two-axis ablation can be shown.
- Availability: repository URL, licence, release tag, Zenodo DOI, and the one command that rebuilds every table and figure (`docs/paper/figures/Makefile`) [TODO 20].

- **Evidence:** SPEC §13; dossier 01 §7, Q4, Q11; Sections 4–11.
- **Bib keys:** `giannone2015prior` `clark2015macroeconomic`
- **Floats:** none.

**Page check:** §1 2.5 + §2 1 + §3 2.5 + §4 2.5 + §5 2.5 + §6 3 + §7 2 + §8 4.5 + §9 1.5 + §10 0.75 + §11 4 + §12 1.75 + §13 1 = **29.5 pages**, within the 25–30 budget. Appendices are about 20–25 pages.

---

## 14. Figure and table plan

Every float names its data source in the repo (or the acceptance JSON) and the script that generates it.
All scripts go under `docs/paper/figures/` and write LaTeX or PDF to `docs/paper/figures/out/`. One
`docs/paper/figures/Makefile` target, `make all`, regenerates everything from committed data. Validation
records are written to `docs/paper/data/release/` by `RAI_BVAR_VALIDATION_RECORD=docs/paper/data/release`
on the release commit. Repo paths below are in the main repo `/Users/yigitokar/rai-bvar` unless stated.

### 14.1 Main text (12 floats)

| ID | § | Caption | Data source | Script |
| :- | :- | :- | :- | :- |
| Fig1 | 4.1 | Architecture of rai-bvar. Specs and resolution select the analytic (NumPy/SciPy) or the MCMC (JAX) engine. Both return the same ForecastResult contract, which feeds the backtest, scores, `compare()` and `verdict()`. The validation package wraps the production sweep. | static; package layout `src/rai_bvar/` (`api.py`, `engines/analytic.py`, `engines/mcmc/`, `forecast/result.py`, `evaluation/`, `validation/`); dossier 02 §1, §7; dossier 03 §1 | `fig_architecture.py` (emits TikZ to `out/fig_architecture.tex`) |
| Tab1 | 3.5 | Components delivered in Phases 0–2 against the roadmap, by axis (mean, prior, volatility, COVID treatment, benchmarks), with the registered validation rows each composition appears in. | `src/rai_bvar/components/specs.py`; `src/rai_bvar/priors/minnesota.py`; `src/rai_bvar/validation/joint.py` (`_COMPOSITIONS`, `_BLOCKS`); SPEC §13 | `tab_components.py` (introspects the registry; no JAX work) |
| Tab2 | 4.2 | Engine resolution: the nine rules that map a resolved (prior form, volatility prior) to an engine and mode, with the errors raised at construction. | `src/rai_bvar/api.py` l.128–191; parametrised cases of `tests/unit/test_api_resolution.py::test_resolution_table` | `tab_resolution.py` (builds each case, records the resolved form, vol, engine and mode, or the error) |
| Fig3 | 6.8 | The verdict rule as a decision flow: axis check, sign, MC-error gate, valid-test or MCS route, convergence gate. The outcomes are better, worse, inconclusive and not comparable. | `src/rai_bvar/evaluation/compare.py` (`CompareResult.verdict`); `tests/integration/test_compare.py` | `fig_verdict_flow.py` (emits TikZ; static) |
| Fig4 | 6.4 | Path-conditioned vs path-integrated RB log score at h > 1. Left: synthetic persistent VAR at h = 8, outcome 6 s.d. out; conditioned estimate ± MCSE against the exact value. Right: Constant (and SV(rw)), UR, h = 8, origin 2008Q2, at the acceptance budget; conditioned with S = 1000 / 8000 / 64,000 against integrated, with the spread across fit and forecast seeds and the weight ESS. | Left: the synthetic case of `tests/unit/test_lpd_integrated.py` (b9c66c0). Right: a refit at 2008Q2 with `--config acceptance` settings and the `fit_seed` rule, written to `docs/paper/data/rb_tail_cell.json` (to create, TODO 6) | `rb_tail_cell.py` (computation) and `fig_lpd_estimators.py` |
| Lst1 | 7.2 | Listing 1: data, an analytic BVAR with SOC/SUR and GLP empirical Bayes, a forecast, and a rolling-window comparison with its verdicts (output verbatim). | `docs/paper/figures/ex_a.py` (verbatim from dossier 02 §13.1; simulated data inline, no network); stdout captured on the release commit | `run_examples.py` (runs `ex_a.py` and `ex_b.py`, captures stdout to `out/ex_a.txt` and `out/ex_b.txt`, records platform and versions) |
| Lst2 | 7.3 | Listing 2: the same model family on the MCMC engine. Summary, forecast LPD, IRF, and a volatility-axis comparison whose verdict the convergence gate forces to inconclusive (output verbatim; MCMC numbers platform-specific). | `docs/paper/figures/ex_b.py` (verbatim from dossier 02 §13.2); stdout on the release commit | `run_examples.py` |
| Tab5 | 8.3 | Validation tiers U, B, G, S, A, P and M: content, decision rule, designed vs as-implemented cadence (the CI job named "nightly" runs weekly), pytest marker, and test-ID counts on the release commit. | DESIGN §9.3–§9.10; `pyproject.toml` [tool.pytest.ini_options] markers; `.github/workflows/ci.yml`; `docs/paper/data/release/collect.txt` (`pytest --collect-only -q`, plus per-marker variants) | `tab_tiers.py` (parses ci.yml and collect.txt) |
| Tab7 | 8.11 | Mutation suite: 18 injected sampler bugs, with the injection seam, composition, catching tiers and mutant/control statistics, and the platform of each record. All 37 mutant × tier checks are caught with passing controls. | `tests/mutation/mutants.py` (`_ALL`, `EVIDENCE`); `docs/paper/data/release/mutation.jsonl` (Apple M5 Max) and `mutation_x86.jsonl` (x86-64 Linux), both from `pytest tests/mutation -m mutation` [REL] | `tab_mutation.py` |
| Tab12 | 11.6 | The six acceptance criteria: N08 qualitative wording, operational rule as coded, and outcome in the first full run (551f181, path-conditioned) vs the pre-registered acceptance run. [ACC placeholders] | `benchmarks/reports/clark_ravazzolo_acceptance.json` `criteria[*].{id,met,rule,detail}`; `benchmarks/reports/clark_ravazzolo.json` `criteria` (551f181); `docs/phase2/08_common_sv_and_ordering.md` §C5 | `tab_cr_criteria.py` |
| Tab13 | 11.9 | Headline: SV(rw) vs Constant, full sample, four variables and joint, h = 1, 2, 4, 8. Columns: ΔLPS integrated (primary), ΔLPS conditioned (secondary), ΔLPS Gaussian approximation, RB/Gaussian sign disagreement, RB-degenerate origins, descriptive one-sided DM/HLN p, CRPS and RMSE ratios, and CR's working-paper values. [ACC placeholders] | acceptance JSON `relative.full['SV(rw)'].variables[v][h].{dlps,dlps_conditioned,dlps_gauss,rb_gauss_disagree,rb_degenerate,dm_p_better,crps_ratio,rmse_ratio}` and `.joint[h].{dlpd,dlpd_conditioned,dlpd_gauss}`; CR columns from CR_FULL (`benchmarks/clark_ravazzolo.py` l.147) | `tab_cr_headline.py` |
| Fig6 | 11.9 | Ordering robustness at h = 1: ΔLPS (SV(rw) − Constant) per variable and joint for all 24 orderings, with the identity ordering marked and the noise floor shown. Inset: joint `lpd_range` by horizon for SV(rw) and for the order-invariant Constant arm (negative control). [ACC placeholders] | acceptance JSON `orderings.per_ordering[*].{ordering,label,dlps_h1,dlpd_h1,dm_p,dm_p_joint_h1,verdicts}`, `orderings.results[*].{material,summary}` | `fig_cr_orderings.py` |

### 14.2 Appendix floats

| ID | App. | Caption | Data source | Script |
| :- | :- | :- | :- | :- |
| TabA1 | A | Notation: κ vs `lambda_`, λ_{i,t}, ι vs i, IG(shape, scale), δ-coordinates, dimensions, tcodes 1–7 with pathwise inverses. | static; dossier 02 §2, §9.2 | `tab_notation.py` (static) |
| FigC1 | C | The Gibbs plan as data: `fit.plan.describe()` for structural SV(outliers, t) and reduced-form SV, with read/write sets and the indicators-before-h invariant. | output of `fit.plan.describe()` on the simulated data of `ex_b.py`; `src/rai_bvar/engines/mcmc/plan.py` | `fig_plan.py` |
| TabC1 | C | The predictive contract: ForecastResult fields, dimensions, meaning and invariants; MCMC accelerator variables (incl. `lag_coefs`, D-34), which are never contract fields. | `src/rai_bvar/forecast/result.py`; `src/rai_bvar/forecast/invariants.py`; `tests/contract/test_forecast_contract*.py`; `tests/mcmc/forecast/mcmc_fits.py` l.128–141 | `tab_contract.py` (introspects a ForecastResult from the ex_a and ex_b fits) |
| TabD1 | D | G-small Geweke results for the 15 PR compositions: G, t_{C−1} critical value, max \|z\| and argmax, Gandy–Scott stage and q, minimum RNE, time. | `docs/paper/data/release/geweke.jsonl` (`pytest tests/geweke -m 'geweke and not slow'`) [REL]; cross-check dossier 04 §6.6 (62d15f6) | `tab_geweke.py` |
| TabD2 | D | Parity summary: R BVAR 1.0.5 and BoE bvar 0.3.5 (analytic); stochvol 3.2.1 main/default rows; the exact importance-sampling oracle for our SV-t sampler; the in-house system-wide Gibbs oracle replacing bayesianVARs. Columns: statistic, bound, max \|z\| or rtol. Neutral wording. | `tests/parity/SV_PARAMETRIZATION.md` §6; `BVAR_PARAMETRIZATION.md` §12; `BOE_BVAR_NOTES.md` §14; `tests/parity/fixtures/`; release run with `-m parity_long` → `docs/paper/data/release/parity.jsonl` | `tab_parity.py` |
| TabD3 | D | G-full, the recorded `exact=False` rows, composition-level Tier B and full SBC per composition: max \|z\| / min p, stages, ESS gates, t_max hits. | `docs/paper/data/release/{geweke_full,sbc,block}.jsonl` [REL; no committed record exists yet] | `tab_geweke_full_sbc.py` |
| TabD4 | D | Analytic twins and exact oracles A1–A7: oracle, statistic and tolerance, negative control and its measured power, cadence, result. | `tests/analytic/test_mcmc_twin.py`, `test_mode1_quadrature.py`; `tests/mcmc/blocks/test_sv_toy_quadrature.py`, `swa_oracle.py`; release junit XML `docs/paper/data/release/twins.xml` | `tab_twins.py` |
| FigD1 | D | Mutation power heatmap: mutants M1–M18 × tiers U/B/G/S. Cell colour is log10(statistic / decision threshold) for the mutant, a marker shows the control, and grey means no check is registered. | `docs/paper/data/release/mutation.jsonl` (+ `mutation_x86.jsonl`) | `fig_mutation_heatmap.py` |
| FigD2 | D | Geweke max \|z\| per composition (controls: G-small and G-full) against the t_{C−1} critical band, with mutant G statistics overlaid on a log scale. | `docs/paper/data/release/{geweke,geweke_full,mutation}.jsonl` | `fig_geweke_dots.py` |
| TabE1 | E | Float64 failures F1–F3 (state, mechanism, how found, fix, evidence), and the RW SV AWOL step at σ = 5e-9: minimum pivot, log-determinant and h_0 for the assembled tridiagonal, the excess-pivot recursion and a 60-digit reference. | NOTES 10, 17; `tests/mcmc/blocks/test_sv_core.py` (`problem(4, T_eff=88, T_pad=96, rw=True, sigma=5e-9)` and its mpmath reference); `src/rai_bvar/engines/mcmc/linalg.py`; commits 3d3f51b, f60d736, 93f2bcd, 4fc0162 | `tab_rw_pivots.py` (recomputes all three columns; pairs with the old-path-fails test of TODO 23) |
| FigE1 | E | RW SV path system at small σ: level-direction pivot and h-level error vs σ for the assembled factorisation and the excess-pivot recursion, against a 60-digit reference. | new committed computation reusing the `problem()` constructor of `test_sv_core.py` → `docs/paper/data/laplacian_pivots.json` (to create) | `fig_laplacian_pivots.py` |
| TabE2 | E | BENCH-P2 on the production engine: wall time cold/warm, compile time, per-sweep time, peak RSS, ESS and ESS/s by quantity group, converged flag, and machine, load and commit. | `tests/bench/baseline.json` (deb3929, non-idle Apple M5 Max; re-measure on the release commit, TODO 9) | `tab_bench.py` |
| TabE3 | E | Kernel microbenchmarks (80 lanes): Gram, precision draw, SV indicators, AWOL, exact correction, SV core, sweep; XLA flops and bytes; DESIGN budget lines. | `tests/bench/baseline.json` `kernels`; `tests/bench/bench_kernels.py` | `tab_kernels.py` |
| TabF1 | F | Run history: quick run; full #1 (8a622c5, crash F2); fixes 93f2bcd, f60d736, 4fc0162; full #2 (4bb978a, report 551f181); D-34 (b9c66c0); pre-registration and acceptance config (c823d20); acceptance run. For each: commit, config, machine/load, wall time, outcome. | headers of `clark_ravazzolo_quick.json` and `clark_ravazzolo.json` (`config`, `versions`, `seconds`); GitHub issue #3 timestamps; commit metadata; acceptance JSON `config`, `versions`, `seconds`; acceptance run log (TODO 2) | `tab_runs.py` |
| TabF2 | F | CR data: CR definitions, FRED-QD 2026-09 mnemonics, transformations, tcode overrides, sample, and run-time data checks. | `benchmarks/clark_ravazzolo.py` (`load_data` l.236, `assert_quarterly_averages`, `assert_in_window`); acceptance JSON `data_checks` | `tab_cr_data.py` |
| TabF3 | F | CR model arms: library specification, CR counterpart, priors, and role in the criteria (including the Beta(71.25, 3.75) mapping for AR-SV(0.9)). | `benchmarks/clark_ravazzolo.py` (`minnesota` l.325, `models` l.330, `structural_models` l.345); acceptance JSON `models` | `tab_cr_models.py` |
| TabF4 | F | Pre-crisis (1985Q1–2007Q4): SV(rw) vs Constant at h = 1 and h = 8 (integrated, conditioned, Gaussian, degenerate count) against CR. [ACC] | acceptance JSON `relative.pre_crisis['SV(rw)']`; CR_PRE_DLPS (l.167) | `tab_cr_precrisis.py` |
| TabF5 | F | All SV arms against Constant: joint ΔLPD by horizon, share of converged origins, `compare()` verdict with gate on/off and the route (DM or MCS), MCS membership; variant comparisons for criterion 5. [ACC] | acceptance JSON `relative.{full,pre_crisis}[arm].joint`, `converged.{full,pre_crisis}`, `verdicts.{full,pre_crisis}`, `variants[variant][ref].joint` | `tab_cr_arms.py` |
| TabF6 | F | Convergence vs MCMC budget for SV(rw), fit-level only: 4 × 5000, burn 5000, 8 chains, 4 × 20,000; identity and the two worst orderings; converged origins, max R-hat, min bulk ESS; plus the main-stage vs ordering-stage seed comparison. [PREREG] | `benchmarks/reports/clark_ravazzolo_preregistration.md` item 1 (c823d20); the diagnostic script and output must be committed as `benchmarks/cr_convergence_diagnostic.py` and `benchmarks/reports/cr_convergence_diagnostic.json` (TODO 4) | `tab_cr_budget.py` |
| TabF7 | F | Constant-arm baseline levels (average log score and RMSE at h = 1) against CR's benchmark VAR, and the compute record of the acceptance run (machine, n_jobs = 10, stage times, wall time). [ACC] | acceptance JSON `relative.full['SV(rw)'].variables[v]['1'].{lps_base,rmse_base}`, `seconds`, `versions`, `config`; CR levels from N08 §C5 | `tab_cr_levels_compute.py` |
| TabF8 | F | First full run tables (history: 551f181, path-conditioned, 4 × 5000): headline, pre-crisis, arms, variants, structural pair, RB-degenerate origins, the 28 + 8 RB/Gaussian sign-disagreement cells. | `benchmarks/reports/clark_ravazzolo.json` (551f181) | `tab_first_run.py` |
| FigF1 | F | Per-origin ΔLPS (SV(rw) − Constant), UR, h = 8 (h = 1 for contrast), 1985Q1–2011Q2: path-conditioned vs path-integrated, from the same forecast draws. [ACC; needs export] | **not committed yet**: the per-origin scores (lpd, conditioned lpd, Gaussian, CRPS, squared error, converged, per arm and ordering) exist only in the machine-local cache `~/.cache/rai-bvar-cr-c823d20`. Export to `docs/paper/data/cr_acceptance_per_origin.csv` (TODO 5), or drop the figure | `export_per_origin.py`, `fig_cr_per_origin.py` |
| FigF2 | F | Share of converged origins per ordering for SV(rw) (and Constant): first full run (4 × 5000) vs acceptance run (4 × 20,000), with the 95% gate; gated and ungated h = 1 verdicts marked. [ACC] | `clark_ravazzolo.json` and acceptance JSON `orderings.per_ordering[*].{converged_sv,converged_constant,verdict_h1,verdict_h1_ungated}` | `fig_cr_convergence.py` |
| FigF3 | F | Cumulative joint ΔLPD over origins at h = 1 and h = 4, SV(rw) vs Constant, with the pre-crisis boundary marked. [ACC; needs export] | `docs/paper/data/cr_acceptance_per_origin.csv` (same export as FigF1) | `fig_cr_cumulative.py` |
| TabG1 | G | Reproducibility matrix: variation; guarantee (bitwise, to rounding, or none); the test that establishes it; scope (runner toys vs production kernels). | DESIGN §5.6; dossier 03 §10; `tests/mcmc/runner/{test_chunking,test_sweep,test_devices_bitwise,test_storage,test_runner}.py`; `tests/mcmc/forecast/test_forecast_core.py` | `tab_reproducibility.py` (static rows; checks that each cited test ID exists via `pytest --collect-only`) |
| TabG2 | G | Environment and regeneration record: versions (Python, numpy, scipy, jax/jaxlib, scoringrules, BLAS backend), machines, release commit, and the command per table and figure. | `docs/paper/data/release/env.json` (written by `run_examples.py`) | `tab_environment.py` |

---

## 15. Appendices (about 20–25 pages; not in the main-text budget)

- **A. Notation and conventions (1 p).**
  - TabA1: `lambda_` = tightness κ; λ_{i,t} = structural variance; 1-based ι vs 0-based i; IG(shape, scale); δ-coordinates; dimensions n, p, n_d, k, T, q.
  - The tcode table 1–7 with pathwise inverses. This resolves dossier 02 Q1.
- **B. Derivations, each with its test (5 p).**
  - B.1 The structural form of the NIW prior (Mode II = NIW at κ_c = κ_a = 1), via the IW partition theorem (`gupta2000matrix`; `chan2022asymmetric`).
  - B.2 Mode I moment matching and the implied reduced-form moments. Includes the cross-lag leak factor κ_c² + (1 − κ_c²) s_jm and the 2^{i−1} growth of the naive choice.
  - B.3 The exact mixture-correction weight w(h) and the single ASIS accept/reject (derived in `docs/phase2/01_omori_mixture.md` §10.3; novelty not claimed).
  - B.4 The path-integrated LPD: m_h, M_h, the Φ recursion and the QR subsets (D-34, b9c66c0). Then the weight bound: for w ≥ 0, sd(w)/w̄ ≤ √S with the sample sd (≤ √(S − 1) with the population sd), so the iid delta-method MCSE is ≤ 1 nat, with equality iff one draw carries all the weight; plus the effect of the √(S/ESS) factor. Status: derived for this outline; the author must confirm it and add a unit test (TODO 14).
  - B.5 The dummy-row λ marginal IG(c + m/2, …), and the proof that g(λ) ≤ 1 for the iw_equivalent rejection envelope (|S(λ)| ≥ |S₀|, q(λ) ≥ q∞).
  - B.6 The scalar conjugate successive-conditional chain as an AR(1), giving IAT = 2r − 1 and the r ≤ 5 rule. Labelled a derivation, not verified for φ, σ, p and ν.
  - B.7 Why the CCM-2019 factorisation is exact under a flat prior, so that its error is a prior effect that cancels in parameter ranks (N06 §7.2; recorded).
  - B.8 The excess-pivot recursion equals the Cholesky pivots of Lap_w + diag(δ) in exact arithmetic, with the log-determinant identity. The QR whitened draw equals the precision draw.
- **C. Algorithms (3 p).** Pseudocode from dossier 03:
  - Alg. 1 the sweep;
  - Alg. 2 the structural (δ, a) draw with D3 centring;
  - Alg. 3 the graded eigen fast path;
  - Alg. 4 the QR a-row draw;
  - Alg. 5 the CCCM step;
  - Alg. 6 the vol block;
  - Alg. 7 the SV core with exact correction, ASIS and the fold;
  - Alg. 8 the Laplacian sampler;
  - the transitory steps;
  - Alg. 9–10 the burn/sample bodies and the host driver;
  - Alg. 11 the failure path and replay;
  - Alg. 12 the forecast core.

  Also FigC1 and TabC1.
- **D. Validation detail (5 p).**
  - TabD1–TabD4, FigD1 and FigD2.
  - The full Geweke function families.
  - The SBC configuration and the ESS erratum.
  - The mutant catalogue: for each of M1–M18, the code seam, the mathematical statement of the error, its literature source where it has one, the catchers on both platforms, and why some tiers have no check (`u_check=None`).
  - Runtime-diagnostics tests against ArviZ reference values.
  - Suite runtimes with machine and load.
  - Parity detail in neutral wording. Any version-specific oracle deviation appears only after TODO 13.
- **E. Numerical robustness and performance detail (2 p).**
  - TabE1, FigE1, TabE2, TabE3.
  - The three sources of performance numbers, and why only the production baseline is quoted.
  - Unmeasured configurations.
  - The single-thread OpenBLAS rule while chains run (docstring numbers not quoted).
  - The dense-eigen timing (recorded under load).
- **F. Clark–Ravazzolo supplement (5 p).**
  - The pre-registration verbatim, with its commit hash (c823d20) and the acceptance report's commit and clean-tree fields.
  - The coded criteria rules verbatim.
  - The DEVIATIONS list (l.1380) plus `LPD_INTEGRATED_DEVIATION`.
  - TabF1–TabF8 and FigF1–FigF3.
  - The run log of the first attempt's crash and the fixes.
- **G. Reproducibility record (1.5 p).**
  - TabG1 and TabG2.
  - The release commit hash.
  - The `RAI_BVAR_VALIDATION_RECORD` paths.
  - The FRED vintage and its SHA-256 checksums.
  - `make all`.
  - Which numbers are bitwise-reproducible, and which only on the same platform and version set.
- **H. Licensing and clean-room practice (1 p).**
  - The verbatim statement from §2 P4.
  - A table of the repository notes in which GPL source was read, with the version and purpose: `docs/phase2/02_sv_sampler.md` (stochvol 3.2.9, bayesianVARs 0.1.8, bvarsv 1.1); `01_omori_mixture.md`; `03_sv_outliers_t.md` (greps; CCMM MATLAB code with no licence); `07_jax_patterns.md` (cross-check search of the 10 constants); `tests/parity/BVAR_PARAMETRIZATION.md` (BVAR 1.0.5 R sources).
  - The notes in which no GPL source was read: `SV_PARAMETRIZATION.md`, notes 04, 05, 08.
  - The fixtures-as-data position (optionally `fsf2026gplfaq`, as a web page, not legal advice).
  - FRED data are not bundled.
  - No PR-template checkbox is mentioned.
- **I. Design-review errata relevant to numerics and reproducibility (0.5 p).** E7 (uint32 counters), E9 (independent references), E14 (bitwise-neutral runtime knobs) and E23 (masking before log), as evidence that review caught defects before implementation (DESIGN §0.2; NOTES C.11). This appendix replaces O1's optional third-party appendix; the generic notes are in §9.6.
- **J. Development process and AI-assistance disclosure (0.5 p; the author writes and approves it).**
  - Design by competing proposals and independent reviews (errata E1–E28).
  - Work packages under the two-implementation rule.
  - Independent x86 review runs.
  - A statement of how AI assistance was used, worded to arXiv policy and to any later venue's. AI is not an author.
- **K. TODO for the author.** Kept out of the compiled paper; it mirrors §16.

---

## 16. TODO for the author (things only the author can decide, run or verify)

**Blocking before §11 is drafted**
1. Fix the stale text:
   - `docs/paper/NOTES.md` l.87 and `docs/phase2/DESIGN.md` l.1674 (main repo) say the acceptance config "keeps 1000 forecast draws". The committed pre-registration (item 3) and `benchmarks/clark_ravazzolo.py` l.195 use 8000.
   - In NOTES, also mark −19.1 as path-integrated over the same 1000 draws, never "exact".
   - Fix the paper-repo copy of NOTES (62d15f6), which lacks the D-34 addendum.
2. When the acceptance run finishes:
   - commit `clark_ravazzolo_acceptance.{json,md}` from `/Users/yigitokar/rai-bvar-accept` into the main repo;
   - verify `config.name == "acceptance"`, `config.lpd_method == "integrated"`, `config.draws == 8000`, `config.mcmc` = 4 × 20,000 with burn 1000, `config.seed == 0`, `versions.git_commit` = c823d20…, `versions.git_dirty == false`, no `pending` key, and that the .md is labelled as the acceptance run;
   - archive the run log `/tmp/claude-501/cr_accept.log` (ephemeral) with its UTC start time, as evidence that the launch followed c823d20.
3. Coverage gap: register `reduced_sv_rw`, and the reduced × SV(rw, t) variant, as Geweke/SBC rows in `src/rai_bvar/validation/joint.py`; add a `reduced-SV(rw)` contract fit to `tests/mcmc/forecast/mcmc_fits.py`; run them. Or keep the caveat in §8.10, §11.10 and §12 P6.
4. Commit the fit-level convergence-diagnostic script and output, e.g. `benchmarks/cr_convergence_diagnostic.py` and `benchmarks/reports/cr_convergence_diagnostic.json`. Until then the TabF6 numbers stay `[PREREG]`.
5. Add a per-origin score export to the CR script (`docs/paper/data/cr_acceptance_per_origin.csv`) for FigF1 and FigF3, or drop both figures.
6. Write `rb_tail_cell.py` and regenerate the NOTES 8b estimator numbers (−93.6 / −46…−63 / −36 / −19.1; weight ESS; thinning sd and bias) on the release commit, for Fig4.

**Release and reproducibility**
7. On the release commit, record the following with `RAI_BVAR_VALIDATION_RECORD=docs/paper/data/release`: G-small, G-full, SBC, composition-level B, `exact=False` rows, `parity_long`, the analytic twins, and the mutation suite on Apple M5 Max and on x86-64 Linux. Commit the JSONL files.
8. Fix count and label drift:
   - NOTES B.8 "33/33" → 37/37;
   - machine labels (M1 Pro vs M5 Max) in the `mutants.py` EVIDENCE header, `SV_PARAMETRIZATION.md` and the Geweke docstring;
   - the M17 control 1.94 (eigen path) vs 2.44 (cholesky_const);
   - NOTES 9 "rtol ≈ 1e-12" → production rtol 1e-9;
   - NOTES 10 and the Geweke docstring l.51, "|y| ≈ 1e8" → 2e8.
9. Re-measure BENCH-P2 on an idle machine on the release commit. Add an idle x86 entry and, optionally, the 4-device run.
10. Decide on the verdict-engine DGP size/power study (SPEC §10.5): implement it, or keep it as future work in §12 P2.
11. Run prior-art searches before any novelty or "to our knowledge" wording:
    - composition and one-axis verdict designs in other BVAR software;
    - mutation testing of MCMC samplers;
    - the ASIS + exact-correction w-ratio;
    - the collapsed standardized-t ν step;
    - the excess-pivot recursion vs GTH, Alfa–Xue–Ye, Ye 2008 and GMRF samplers;
    - whether stochvol's JSS paper uses Geweke-type unit tests (credit it if so).
12. Re-verify at submission the Python, MATLAB and R names cited in §2 P2 (names only; no feature claims). Check srvar-toolkit's first release date.
13. Third parties. No finding about other packages appears in the paper until these are done.
    - Re-test the stochvol D-7 fold and SV-t observations on the current stochvol release (the unblock path in `SV_PARAMETRIZATION.md` §3.2).
    - Re-test the BVAR 1.0.5 items (predict noise, ψ units, ȳ₀, dummy-only ML term) and the BoE bvar 0.3.5 items (ȳ₀, gradients, docs vs code) on current versions.
    - **Contact the maintainers** of stochvol/factorstochvol, R BVAR, the BoE bvar and bayesianVARs, using the maintainer fields on CRAN/PyPI.
    - Add the BVAR, BoE and Accelerate items to NOTES §A.
14. Derive the MCSE weight bound (App. B.4) carefully, including the ddof convention (`w.std(ddof=1)`, `tests/unit/test_lpd_integrated.py` l.202) and the √(S/ESS) factor. Add a unit test. Only then use it in §6.4 and §11.5.
15. Literature checks:
    - verify the attribution to Talts et al. (2018) of the concern that Geweke z-scores need converged chains;
    - check the Modrák et al. theorem and equation numbers against the journal version;
    - check the Alfa–Xue–Ye page range (217–236 vs 217–237);
    - keep N06's explanation of Gandy & Scott's Geweke size (r ≈ 2000) out of the paper;
    - section numbers in `chan2024bvars` refer to arXiv v1.
16. Reword the library's sign-fold warning, which names stochvol and bayesianVARs, before any listing prints stderr.
17. Licensing decisions: whether to cite `fsf2026gplfaq`; add a PR-template checkbox, or keep never mentioning one.
18. Resolve the SPEC↔code discrepancies D1–D15 of dossier 02 §15, or describe the code: the DM kernel; energy subsample; `store_moments='lazy'`; MCSE batch means; LP on MCMC; exogenous regressors; subsample analysis; ordering column; `estimate_cost` signature; ArviZ; the bayesianVARs exit; README; the FRED-QD year in `fred.py`; the analytic `ordering` attr.
19. Fix the README status banner and quick start. Release on PyPI with a version tag. Decide the version number (currently 0.0.1, pre-alpha).
20. Merge `phase-2-cr-acceptance` into `main` and fix the **release commit hash**. Mint a **Zenodo DOI** for the release plus the replication outputs. Add the repository URL. Re-run `ex_a.py` and `ex_b.py` on the release commit (D-34 touched `forecast.py`).
21. Confirm the official FRED-QD 2026-09 tcodes vs the 'rai' codes. Align `fred.py` to cite McCracken & Ng (2021).
22. Obtain the JAE tables of Clark & Ravazzolo (2015), or say explicitly that the working-paper numbers are used. Resolve N08's "every number re-read" (40) vs the 56 numbers quoted. Keep the CCM (2016) +0.967 and the ARRS/CKY ordering magnitudes out unless they are verified.
23. Re-run the tiny-σ tests and the QR replay test on Linux. Commit tests that assert the old-path failures at the R3 and R4 states (including the silent positive-pivot draw). Decide on a relative pivot floor for the other formed-Gram draws.
24. Make the NOTES 12 Mode II λ-count injection reproducible (an M19 seam or a committed scratch test), or keep "33 of 165" withheld.
25. Commit small tests or scripts for the numbers that are docstring-only: SBC 76% and 0.836L; the "2,250 fresh problems"; CDF-form vs order-statistic MCSE (−4.99, seed 221); the N06 explosive-mass bounds; the eigen-path accuracy envelope.
26. Investigate the Constant arm being flagged "material" by `ordering_sensitivity` (`noise_reps = 1`). Decide `noise_reps` for future runs, and report the flag if it recurs.
27. Remove the NUTS-mixing claim from SPEC §11, or support it.
28. Run the DESIGN R6 robustness items (Mode I vs II; data-centred vs stochvol μ prior) as labelled post-hoc checks, or drop them from DESIGN.
29. Decide whether to say that the committed quick report (before f60d736) may have been affected by the old RW path. Cite quick-run numbers only as context.
30. **Affiliation**; **AI-assistance disclosure wording** (App. J; arXiv policy; AI is not an author); the authorship statement; acknowledgements (independent reviewers).
31. Typesetting: `\usepackage[T1]{fontenc}` for the `kadiyala1997numerical` SICI DOI. Check the PDF.
32. Decide the arXiv category (stat.CO with cross-lists econ.EM and q-fin.ST, or econ.EM primary) and the target journal (e.g. JSS), which may impose its own software-paper format.
33. Decide the convergence-gate policy for SV at the default budget (default draws 2000; φ and σ are primary). This is an owner decision, documented in §12 P3.

---

## 17. Risk list

1. **Acceptance outcome unknown.** Every CR acceptance number is a placeholder, and no sentence may anticipate it.
   - Criterion 4 may fail on UR (pre-registered: about +0.23 vs +0.157).
   - The pre-crisis GDP h = 8 cell could flip (first run: RB +0.108, Gaussian −0.004).
   - Criterion 1 at h = 2 is thin for UR and TBILL (first-run DM p 0.090 and 0.073).
   - Gate-on verdicts for SV, SV(rw, t10) and AR-SV(0.9) may stay inconclusive: the diagnostic covered only SV(rw), and only 3 of 24 orderings at main-stage seeds.
2. **Provenance of "fixed before any score existed".**
   - It rests on c823d20 preceding the launch (the worktree was created at c823d20) and on the report header (`versions.git_dirty == false`, `config.name == "acceptance"`).
   - The run log is in `/tmp` and will be lost unless archived (TODO 2).
   - If a bug forces a rerun, the pre-registration requires full reruns of the affected stages with a fresh cache, and the paper must report it.
3. **Stale and contradictory ledger text.** NOTES l.87 and DESIGN l.1674 say 1000 forecast draws; the pre-registration says 8000. The paper-repo NOTES lacks the D-34 addendum. If they are not fixed, §11.8 contradicts the repository.
4. **The D-34 status depends on the release.** The tests are committed (b9c66c0) but not on `main`. The abstract, C3, §6.4 and Fig4 depend on D-34 landing on the release commit. If it does not, cut the abstract clause and §6.4's fix paragraph.
5. **Coverage gap on the scored composition.** reduced × SV(rw) is validated only through its parts. A referee will ask; register the rows (TODO 3) or keep the caveat in three places.
6. **The MCSE bound is new to this outline.** It is derived here and checked by two judges for the population sd, but it is not in the repository. Do not state it in the main text until TODO 14 is done.
7. **Clean-room overclaim.** GPL sources were read for constants and behaviour (stochvol 3.2.9, bayesianVARs 0.1.8, bvarsv 1.1, BVAR 1.0.5). Use the verbatim qualified statement. Never write "clean room" unqualified, and never cite a PR-template checkbox.
8. **Claims about other software.**
   - The stochvol D-7 and SV-t observations, the BVAR 1.0.5 items and the BoE bvar 0.3.5 items are recorded on pinned versions only. They are not publishable before re-tests and maintainer contact.
   - BVAR and stochvol are JSS packages.
   - The library's own warning names other packages.
   - Even a strict-xfail mention in a main-text caption implies a judgement, so keep it out.
9. **"First" or "only" claims.** No prior-art search has been done (TODO 11). The pivot recursion is an application of GTH-type elimination. The QR draw is textbook conditioning. The ASIS + exact-correction ratio is "derived in the design notes", not claimed as new.
10. **Test naming.** `ex_a.py`'s comment says "DM/AG p-values". The implemented test is DM with the HLN correction; Amisano–Giacomini is cited only as the density-forecast analogue. Make sure the text does not call the implemented test "Amisano–Giacomini".
11. **Verdict-engine statistics are unvalidated.** There is no DGP size/power study. The quadrature MCSE ignores the shared forecast seed, and the MCS route is used under expanding windows. Do not imply a controlled false-positive rate.
12. **Counts and coverage.**
    - The mutation suite has 37 checks (not 33) for 18 mutants.
    - The contract tests cover 12 MCMC compositions, not 22.
    - "G-full/SBC green on every composition" has no committed record.
    - The suite counts (4,964 at 62d15f6; 4655 passed at b9c66c0) must be refreshed at release.
13. **Reproducibility wording.** Chunk, burn-piece and fused/unfused invariance is tested with runner toys. Production vmap and shard_map agree to rtol 1e-9. Nothing holds across platforms. NOTES 9's 1e-12 is wrong for production kernels.
14. **Performance.** BENCH-P2 comes from one non-idle Apple M5 Max at deb3929, with converged = False in all runs and the R-hat detail only in a commit message. Never blend it with pilot or DESIGN numbers. Never present 52.3 s as a portable figure.
15. **Convergence-gate narrative.** In the first run the gate, not the scores, decided criterion 6. Present this as intended strictness plus a usability cost, without implying the defaults were tuned.
16. **Ordering diagnostic false positive.** The theoretically order-invariant Constant arm was flagged "material" in the first run. If this recurs, report it as a limitation of the noise floor.
17. **Replication target.**
    - The CR numbers come from the 2012 WP, not the JAE article.
    - N08's 40-vs-56 re-read count conflicts.
    - Third-party magnitudes (CCM 2016's +0.967, the ARRS/CKY ordering spreads) are unverified.
    - Final-vintage data, the iw_equivalent baseline and default SV priors are documented deviations. Do not attribute differences (e.g. inflation gains of 1.6–3.2× CR's in the first run) to any one cause without evidence.
18. **Recorded-only numbers must carry their status:** 537 and 639/656 arrays; the dense-eigen timing; "33 of 165"; whole-scan M1 "not caught"; the pilot cold-start shares; the 1.6e-8 sd QR accuracy at the real state. Prefer omitting them to presenting them as verified.
19. **Open questions about earlier runs.** Whether pre-fix SV(rw) runs were affected by the old RW path's silent mode is open; the committed quick report predates f60d736. Full run #2 differs from #1 in 41 files, so never describe it as differing "only by the fix".
20. **Platform caveats.** The macOS heap-corruption crash has an unknown root cause, and the Accelerate warnings have no minimal reproducer. State them only as caveats, with Linux CI as the reference.
21. **EB wording.** EB on MCMC is type-II ML only for the Mode II twin, and a plug-in for Mode I and SV. Never describe SV models as having EB-optimal hyperparameters.
22. **Citation hygiene.**
    - FRED-QD is `mccracken2021fredqd` (`fred.py` says 2020).
    - The corrigendum is Carriero, Chan, Clark & Marcellino ("Chan & Yu" is wrong).
    - bvars is Liu, Ramírez Hassan & Woźniak.
    - Cite stochvol via `hosszejni2021modeling` and `kastner2016dealing` and state version 3.2.1.
    - mfbvar is archived on CRAN and is not cited as current.
    - Cite only refs.bib keys.
23. **Length.** The 12 main-text floats are inside the 29.5-page budget only if §8 and §11 hold to 4.5 and 4 pages. If they overrun, move Tab5 to App. D and keep its content in prose.
24. **Disclosure.** The repository shows a multi-agent design and review process, and arXiv and most venues require an AI-use statement. The author must write it (TODO 30).

---

## 18. Bibliography keys used in this outline

All of the following exist in `docs/paper/refs.bib` (checked by script against the 124 keys). This section is
regenerated by the check in the drafting stage.

`abrilpla2023pymc` `adolfson2007forecasting` `alfa2002accurate` `amisano2007comparing` `arias2023macroeconomic` `banbura2010large` `bertsche2022identification` `bjorck1996numerical` `boeck2022bgvar` `bognanni2022comment` `bradbury2018jax` `burkner2020approximate` `carriero2015bayesian` `carriero2016common` `carriero2019large` `carriero2022corrigendum` `carriero2024addressing` `chan2009efficient` `chan2022asymmetric` `chan2024bvars` `chan2024large` `chan2026bvartoolkit` `chan2026bvartoolkitpython` `chiles2012geostatistics` `clark2001tests` `clark2011realtime` `clark2012macroeconomic` `clark2015macroeconomic` `clark2023tail` `cogley2005drifts` `cook2006validation` `croushore2001realtime` `dawid1999coherent` `delnegro2015time` `demillo1978hints` `diebold1995comparing` `dieppe2016bear` `doan1984forecasting` `dutta2018testing` `flegal2010batch` `fox2017accelerated` `frostig2018compiling` `fruhwirth2010stochastic` `fsf2026gplfaq` `gandy2021unit` `gelman2017correction` `geweke2004getting` `geweke2010comparing` `geyer1992practical` `giacomini2006tests` `giannone2015prior` `gneiting2007strictly` `gneiting2011comparing` `grassmann1985regenerative` `grosse2014testing` `gruber2025forecasting` `gruber2026bayesianvars` `gupta2000matrix` `hansen2011model` `harris2020array` `harvey1997testing` `higham2002accuracy` `holm1979simple` `hosszejni2021modeling` `hoyer2017xarray` `huber2022inference` `jacquier1994bayesian` `jacquier2004bayesian` `jia2011analysis` `jordan2019evaluating` `journel1978mining` `kadiyala1997numerical` `kastner2014ancillarity` `kastner2016dealing` `kastner2019sparse` `kastner2026factorstochvol` `kim1998stochastic` `kim2026bvhar` `kish1965survey` `krueger2015bvarsv` `kruger2021predictive` `kumar2019arviz` `kuschnig2021bvar` `kuschnig2024bvar` `labonne2026bvar` `lenza2022estimate` `litterman1986forecasting` `liu2026bvars` `marsaglia2000simple` `mccausland2011simulation` `mccracken2016fredmd` `mccracken2021fredqd` `modrak2025simulation` `mohr2026bvartools` `newey1987simple` `omori2007stochastic` `phan2019composable` `pinder2026impulso` `primiceri2005time` `pymc2026extras` `rue2001fast` `sailynoja2022graphical` `sailynoja2025posterior` `salmon2011parallel` `scheuerer2015variogram` `seabold2010statsmodels` `shaw2026srvar` `sims1998bayesian` `sims2006were` `stock2016core` `stroud2003nonlinear` `talts2018validating` `vandyk2008partially` `vehtari2021rank` `virtanen2020scipy` `wang2026bsvarsigns` `wilson2020efficiently` `wozniak2026bsvars` `ye2008computing` `yu2011center` `zanetta2024scoringrules`

Unused bib entries (deliberately not cited): `ankargren2021mfbvar` (archived on CRAN; mixed frequency is out of scope), `besag1989generalized` (no use identified), `mccracken2020fredqd` (cite the 2021 *Review* instead).
