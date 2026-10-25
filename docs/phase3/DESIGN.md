# Phase 3 design: shrinkage priors, CommonSV, the asymmetric conjugate prior and conditional forecasts

**Status.** Draft for review, written 2026-10-25 on branch `phase-3-design` from `main` at `2c71fb2` (release
v0.2.0 = Phase 2, tag at `4ac7969`, plus PR #11 and PR #12). It covers SPEC §13 Phase 3 and is the document the
Phase 3 work packages are implemented from. `docs/phase2/DESIGN.md` (cited as **P2**) stays authoritative for
everything this document does not change; where this document changes a Phase 2 contract it says so and names the
P2 section it amends.

**How it was produced.** One author wrote it after reading SPEC v2.3, P2 in full, the Phase 2 research notes
(N01–N09; N08 §A already specifies `CommonSV`), the code at `2c71fb2`, the manuals of `bayesianVARs` 0.1.8 and
`bvars` 1.1, the MIT source of the Bank of England `bvar` 0.3.5, and the papers of §14. Independent reviewers then
read the document cold, without the author's notes, under different briefs; every finding and its resolution is
in §0.4. The design was revised before any Phase 3 code was written.

**Reading guide.** §0 scope, decisions, errata · §1 module layout · §2 contracts (static config, state, prior
arrays, keys) · §3 global-local and selection priors · §4 the asymmetric conjugate prior · §5 `CommonSV` · §6
conditional forecasts · §7 API and construction-time validation · §8 MCMC budgets and the convergence gate · §9
validation plan · §10 performance · §11 risks · §12 open questions · §13 work packages and the exit-criteria map.

**Conventions.** As P2: code indices are 0-based (equation `i ∈ {0..n−1}`, `ι = i + 1`), `k = n_d + n·p`,
`q = k + n − 1`, `T_eff` real rows, `T_pad` compiled rows, `m_t = 1{t < T_eff}`; `IG(α, β)` has shape `α` and scale
`β`; `G(α, β)` is a gamma with shape `α` and **rate** `β`; `GIG(λ, ψ, χ)` has density `∝ x^{λ−1} exp(−(ψx + χ/x)/2)`
on `x > 0`; `Exp(r)` has rate `r`. `N08 §A5` means `docs/phase2/08_common_sv_and_ordering.md` §A5. Numbering
continues Phase 2 so cross-references stay unique (D-55): decisions **D-36…**, open questions **Q15…**, review
errata **E29…**, work packages **WP26…**, Geweke/SBC composition rows **301…**, mutants **M19…**, analytic oracles
**A8…**, `bayesianVARs` convention probes **P6–P10** (P1–P5 are P2 §9.9's), `bvars` probes **P11–P15**.

**Clean room** (SPEC §2.1). Apache-2.0. Every algorithm below is specified from papers (§14) and re-derived here
where marked *derived here*; each such derivation has a Tier U density-ratio or closed-form test before any kernel
relies on it. `bayesianVARs`, `bvars`, `stochvol` and `BVAR` are GPL: they are run only as black-box oracles by
`tools/make_r_fixtures.R`, outside CI, and only their manuals were read. Printing an object an R package returns
(its default settings, a grouping matrix) is running it; its output is data. The Bank of England `bvar` is MIT and
may be read and used as a Python-side oracle in CI. No code is translated from GPL sources.

---

## 0. Scope and decisions

### 0.1 What Phase 3 delivers

From SPEC §13 Phase 3:

1. Global-local and selection priors on the lag coefficients of `Linear`: `Horseshoe`, `NormalGamma`, `R2D2`,
   `DirichletLaplace`, `SSVS`, with semi-global coefficient groups (§3).
2. The user-facing asymmetric conjugate prior of Chan (2022), `AsymmetricConjugate`, on the analytic engine (§4).
3. `CommonSV`, Carriero, Clark & Marcellino (2016): a conjugate-speed SV screen for large `n` (D2), plus a
   per-equation variant for fair volatility-axis ablations (§5).
4. Conditional forecasts with hard conditions (SPEC §6.2; §6).

Exit criteria (SPEC §13): `bayesianVARs` parity for the global-local priors × SV and for the reduced-form sampler,
after the convention probes P1–P5 (and the Phase 3 probes P6–P10) are settled; `bvars` parity for `CommonSV`; DGP
verdict-engine tests on linear DGPs. The exit-criteria map is at the end of §13.

### 0.2 Inventory of every Phase 3 commitment in SPEC and P2

Every sentence in SPEC v2.3, P2, the Phase 2 notes and the code that assigns work to Phase 3, and what v0.3.0 does
with it. "SPEC edit" names the sentence WP42 changes when an item is deferred.

| # | Commitment | Where | v0.3.0 |
| :- | :--- | :--- | :--- |
| 1 | HS, NG, DL, R2D2, SSVS | SPEC §4.4, §13 | **In** (§3) |
| 2 | Coefficient groups: own lags, cross lags | SPEC §4.4 | **In**: groups over lag coefficients (D-38) |
| 3 | Group "deterministics" | SPEC §4.4 | **Not shrunk**: deterministic terms keep their fixed Normal prior (D-36); SPEC edit §4.4 |
| 4 | Group "contemporaneous" | SPEC §4.4 | **Deferred** (Q17): `a_i` keeps the Phase 2 Normal prior; SPEC edit §4.4 |
| 5 | `AsymmetricConjugate` prior and engine rule | SPEC §4.4, §4.5, P2 Q13 | **In** (§4) |
| 6 | `CommonSV` | SPEC §4.4, D2, N08 §A | **In** (§5) |
| 7 | Conditional forecasts | SPEC §6.2, §12 tree `forecast/conditional.py` | **In** (§6) |
| 8 | `bayesianVARs` parity: reduced-form sampler, GL × SV, probes P1–P5 | SPEC §10.4, §13; P2 §9.9, WP21 | **In** (§9.8, WP29, WP39) |
| 9 | `bvars` parity for `CommonSV` | SPEC §13, §2.2 | **In**, with probes P11–P15 deciding the comparison (§9.8) |
| 10 | DGP verdict tests on linear DGPs | SPEC §10.1, §10.5, §13 | **In**, built by the parallel DGP session; WP40 consumes its PR (§9.9) |
| 11 | Gruber & Kastner (2025) sparse-vs-dense replication | SPEC §10.7 | **In**, pre-registered (WP41) |
| 12 | `hierarchy="mh"` on the MCMC engine | SPEC §4.5, P2 D-12/§6.5, `api.py` message | **Deferred** (Q15); SPEC edit §4.5; the error message stops naming Phase 3 |
| 13 | Estimated Lenza–Primiceri scaling on the MCMC engine | SPEC §5.6, P2 D-21/§3.9, `api.py` message | **Deferred** (Q16); SPEC edit §5.6; message changed |
| 14 | Continuous ν prior | P2 Q8 | **Deferred** (Q18) |
| 15 | `solver="bcm"` (Bhattacharya et al. 2016) for `q > T` | P2 §3.9, N07 §5 | **Deferred** (Q19); the Cholesky path stays exact for `q > T`, only slower |
| 16 | A GIG sampler | N07 §9 (open), P2 §3.9 | **In** (D-42) |
| 17 | Analytic forecasts without `lag_coefs` | P2 D-34, §7.3 | **Changed**: analytic and structural-NIG forecasts attach `lag_coefs` (D-49) |
| 18 | The SV-t parity row stays an expected failure until a `stochvol` ≥ 3.2.5 fixture exists | P2 §2.10 Stage 4 (1) | **In if WP29's R environment has `stochvol` ≥ 3.2.5** (WP29 regenerates it); otherwise unchanged |
| 19 | Two-series SBC for VAR-SV | N06 §6 (optional) | Not planned |
| 20 | `HMP` (bayesianVARs' hierarchical Minnesota) | SPEC §2.2 table | Not in scope (Q20) |
| 21 | `CommonSV` with Student-t errors, random walk, outliers | N08 §A5.4, §A9 | **Deferred** (Q24) |
| 22 | Soft conditions, parameter-updating conditional forecasts | Waggoner & Zha (1999) | **Deferred** (Q25) |

### 0.3 Decision register

| ID | Decision | Source | Notes / supersedes |
| :--- | :--- | :--- | :--- |
| D-36 | **What is shrunk.** The global-local and SSVS priors govern the lag coefficients only (`n·p` per equation). Deterministic terms keep the fixed Normal prior of the form (structural Mode I `N(m, κ_d ψ_i/ι)`; reduced `N(0, κ_d ψ_j)`; `var_const`), and the contemporaneous `a_i` keep the Phase 2 prior (Mode I `N(0, κ_a ψ_i/(ι ψ_j))`, or λ-scaled under `iw_equivalent`). Shrinkage is towards the transform-aware prior mean (SPEC §5.3): the prior governs the **deviation** `d = coefficient − m` | GK (2025) §2 (intercept `N(0, 1000)` unshrunk); `bayesianVARs` manual (`prior_intercept` fixed) | SPEC §4.4 lists deterministics and contemporaneous as groups: §0.2 rows 3–4 |
| D-37 | **Scale.** Kernel variance `c_j·v_j`: `c_j = ψ_i/(ι ψ_m)` (structural) or `ψ_i/ψ_m` (reduced) for the coefficient of equation `i` on a lag of variable `m`, so the prior is in the units of the Mode I Minnesota prior at `λ = 1`, and the model is scale-equivariant (R9: ψ per window). `scale="none"` sets `c = 1` (raw coefficients; parity). No lag decay: lag-wise groups learn it | N05 §4.1 (Mode I scales); GK (2025) Minnesota scales | — |
| D-38 | **Groups.** `groups="olcl-lagwise"` (default; `2p` groups: own lags of lag `l` across equations, cross lags of lag `l` across equations), `"global"` (1), `"equation-wise"` (`n`: every lag coefficient of equation `i`), `"covariate-wise"` (`n·p`: regressor `(l, m)` across equations), or an explicit `(n, n·p)` integer array. Names follow `bayesianVARs`; their meaning there is probe P7 | GK (2025) §3.2: semi-global beats vanilla in every simulated design and in the application | — |
| D-39 | **Prior families and parametrisations** (§3.2): HS (Makalic & Schmidt 2016 auxiliary form), NG (`a, b, c`, GK/`bayesianVARs` meaning), R2D2 (`a, b`, GK gamma form), DL (`a`, gamma form), SSVS (`c0, c1, semiautomatic, p`). `a` is fixed or has a discrete hyperprior (default: GK's grid) | GK (2025) §2–3, App. B, re-derived here | — |
| D-40 | **One post-phase scale step** `G_COEF_GLOBAL` (`Block 17`, reserved by P2 §5.6) draws every local, group and hyper parameter of a GL prior, for the structural and the reduced form alike, after the equation phase. The coefficient-prior kernel declares `has_local_state = False`, so no `coef_local` equation step is planned (P2 §4.2 S3 stays omitted) | P2 §3.4, §4.8 item 3 (post steps may write `eq.coef`, `glob.*`) | P2 §3.4 sketched local scales in `sample_local_eq`; one step is simpler because semi-global groups cross equations and the reduced form's coefficients are global anyway (`ReducedEq.coefficients_eq` docstring) |
| D-41 | **Log space, no floors.** Every scale is stored as its logarithm; draws are log-gamma, log-inverse-gamma and log-GIG; the mean kernels draw GL coefficients in deviation coordinates with a log-space Jacobi scaling and return `log|d|` exactly (§3.4). No tolerance, floor, truncation or jitter is added to any prior | P2 D-14, D-18, N07 §9 (tiny shapes) | `bayesianVARs` floors draws (`PHI_tol`, `*_tol`); we do not, so the target stays the stated prior |
| D-42 | **GIG sampler**: exact rejection on the log scale. `u = log X` has the log-concave density `exp(λu − (ψeᵘ + χe⁻ᵘ)/2)`; the envelope is flat around the mode with tangent exponential tails at two points; `χ = 0` or `ψ = 0` draw log-gamma exactly (§3.5) | Devroye (2014); log-concavity re-derived here | — |
| D-43 | **Where GL priors run.** Structural Mode I or reduced form, with `Constant("ig")`, `Constant("iw_equivalent")`, every `SV(...)` and `CommonSV()`. Rejected at construction: the conjugate form, Mode II (`Constant("iw")` × structural, a λ-scaled coefficient prior), the analytic engine. `form="auto"` resolves to `"structural"` (D3). A GL prior varies, so the eigen and fixed-factor paths are off (P2 D-26) | P2 §3.8, §11.1 | — |
| D-44 | **SSVS scales.** `semiautomatic=True` (default): `τ_{0,j} = c0·ŝ_j`, `τ_{1,j} = c1·ŝ_j`, with `ŝ_j` the flat-prior posterior standard deviation of the coefficient on the window (structural: OLS of equation `i` on `[X | X M_c − Y_{<i}]`; reduced: OLS VAR), computed on the host per window (R9). Needs `T_eff > q` (structural) or `T_eff > k + n + 1` (reduced), else an actionable error. `semiautomatic=False`: `τ = c·√c_j` | George, Sun & Ni (2008); GK (2025) §2.4 | the exact `bayesianVARs` scaling is probe P9 |
| D-45 | **`AsymmetricConjugate`** is Chan's (2022) prior in our structural form (Mode II arrays with `cross` and `contemp`, δ-coordinates and D3 centring, P2 §6.1), fitted on the **analytic engine** by the existing closed form (`engines/structural_nig.py`): iid draws, exact log ML. `hierarchy`: `"fixed"`, `"eb"` (default; free `lambda` and `cross`, Chan's EB), `"mh"` (random-walk MH over log hyperparameters, as Phase 1). Only with `Constant` volatility (`"auto"` resolves to `"iw"`) | Chan (2022) §2–4; P2 Q13, WP16 | `Minnesota(form="structural") × Constant("iw")` stays the MCMC Mode II composition (P2 rule 4) |
| D-46 | **`CommonSV` model**: `u_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `E h_t = 0` (identification), stationary `h_1 ~ N(0, σ_h²/(1 − ρ²))`. Defaults `ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)`, `σ_h² ~ IG(5, 0.04)`; options `ρ: Beta` on `(ρ+1)/2` and `σ_h²: ScaledChi2` for parity | N08 §0, §A1–A3 (Chan 2020 priors) | — |
| D-47 | **Two `CommonSV` programs, one `h` kernel.** (b) **conjugate**: `Minnesota(form="conjugate") × CommonSV()`: a pre step draws `(Π, Σ) | h` jointly from the NIW posterior on GLS-rescaled rows (one QR, the Kronecker speed), a post step draws `h, ρ, σ_h | q`. (a) **per-equation**: structural, reduced or GL `Linear × CommonSV()`: per-equation Constant cores `λ_i`, `λ_{i,t} = λ_i e^{h_t}`, the same post step on `q_t = Σ_i e_{it}²/λ_i`. `form="auto"` with `CommonSV` resolves to `"conjugate"` (the D2 screen); grids with structural arms harmonise it (D-53). Mode II × CommonSV (engine-level) is the exact twin of (b) (A8) | N08 §A1, §A5–A6, §A11 item 2 | P2 §3.9 sketched only (a); (a) alone does not give conjugate speed |
| D-48 | **`h` step**: blocked exact independence MH (Newton mode from a start that does not depend on the current block, Gaussian proposal with the tridiagonal Hessian, blocks of `block = 100` dates with a uniformly random offset per sweep); `ρ` truncated-normal independence MH with the stationary-start factor; `σ_h²` exact (IG, or GIG under `ScaledChi2`) | N08 §A5.2–A5.3 (derived there; Geweke-gated here) | the Omori-mixture variant (CCM 2016) is not shipped: it is not exact (N08 §A11) |
| D-49 | **`lag_coefs` on every linear forecast.** Analytic (`AnalyticFit`) and structural-NIG forecasts attach the accelerator variable `lag_coefs` (P2 D-34), so the integrated LPD and conditional forecasts work on every engine. Contract fields unchanged | P2 D-34, §7.3 | amends P2 §7.3 "analytic-engine forecasts carry no `lag_coefs`" |
| D-50 | **Conditional forecasts**: hard conditions `y_{T+h, j} = r_{h,j}` on the model (transformed) scale; per draw, the exact Gaussian conditioning of the unconditional joint path given the parameter draw and its drawn volatility path (the conditions do not update the parameters; Waggoner & Zha 1999's per-draw step, as in BoE `bvar` and `bvars`); computed from an unconditional forecast's `paths`, `cond_mean`, `cond_chol` and `lag_coefs`, so it needs no new random numbers; returned as `ConditionalForecastResult` (§6) | Waggoner & Zha (1999) §3; Bańbura, Giannone & Lenza (2015) | parameter-updating and soft conditions deferred (Q25) |
| D-51 | **Budgets** (extends D-35): `CommonSV` is a stochastic-volatility composition for D-35 (default 4 × 10,000 kept draws, warning below). GL compositions: the default budget is frozen by the WP31 pilot (§8; Q21); until then they take the D-35 default of their volatility. The convergence gate (P2 §8.2) is unchanged | P2 D-35, Q14 | — |
| D-52 | **Monitors** (extends D-33): GL group-level log scales are **primary** when the number of groups `G ≤ max(2p, n)` (global, olcl-lagwise, equation-wise), secondary otherwise; local scales, SSVS indicators and hyperprior indices are secondary. `CommonSV`: `h_T`, `ρ`, `σ_h` primary, plus `log λ_i` (per-equation) or `log Σ_ii` (conjugate) | SPEC §10.6 ("monitored on hyperparameters") | — |
| D-53 | **Grid harmonisation** (extends D-19): if any arm resolves to MCMC through a non-conjugate volatility (`SV`) or a non-conjugate prior (GL), every `Minnesota(form="auto")` arm becomes `"structural"`, every `Constant("auto")` becomes `"ig"` and every `CommonSV` arm with an `auto` Minnesota form uses the per-equation program; a grid of only `Constant`/`CommonSV` arms with `auto` forms stays conjugate (analytic and program (b)), which is one-axis fair (same NIW prior) | P2 §11.4 | — |
| D-54 | **Oracle environment.** R fixtures are generated outside CI: by a manually dispatched GitHub Actions workflow (`.github/workflows/r-fixtures.yml`, `ubuntu-latest`, R from `r-lib/actions`, CRAN sources built with gfortran; it uploads the fixtures as an artifact), or locally in a pinned `rocker/r-ver` container. Fixtures are committed with R, package and container versions in their attributes; parity tests only read them (CI never installs R) | SPEC §10.4 | P2 could not build `bayesianVARs` (no gfortran); downloading the artifact or the image needs the owner's permission (Q22) |
| D-55 | **Numbering** continues Phase 2 (Conventions); SPEC §14 points at both registers | — | — |
| D-56 | **Validation designs for heavy-tailed coefficient priors.** GL composition rows run Geweke and SBC on an exogenous fixed design (`X ~ N(0, 1)` drawn once per configuration and held fixed, no lag feedback; N08 §A11), because HS/NG/DL/R2D2 coefficient priors put non-negligible mass on explosive lag dynamics; one lagged-design row per form keeps the lag path tested, with scales tight enough that explosive prior mass is ≤ 2e-4 (P2 §9.2). Test functions are bounded or have finite variance under the prior (§9.4) | N06 §3.4; N08 §A11 | — |
| D-57 | **Branches and PRs.** Each work package lands on its own branch from the previous work package's branch; its PR targets that branch, so the PR's diff is the work package alone (GitHub retargets it to `main` when its base merges). Merging is the owner's | task brief | — |

### 0.4 Review errata

Filled in by the review (§0 header): every finding, who raised it, and its resolution.

| # | Finding | Resolution | § |
| :- | :--- | :--- | :- |
| — | (pending review) | | |

---

## 1. Module layout and ownership

### 1.1 Import-time rules

P2 §1.1 holds unchanged: `import rai_bvar` imports no JAX; the new user-facing specs (`priors/global_local.py`,
`priors/asymmetric.py`, `components/specs.py` `CommonSV`) are pure Python; kernels import JAX and are loaded only
through `engines/mcmc/registry.py`. The analytic engine for `AsymmetricConjugate` is NumPy only.

### 1.2 Tree (new files and changed files; owners are the work packages of §13)

```text
src/rai_bvar/
├── __init__.py                          [WP27] + Horseshoe, NormalGamma, R2D2, DirichletLaplace, SSVS, DiscreteGrid,
│                                                 AsymmetricConjugate, CommonSV, conditional_forecast
├── api.py                               [WP27] resolution and engine table of §7.2, error table §7.3, harmonise (D-53)
├── components/specs.py                  [WP27] + CommonSV spec
├── components/coef/global_local.py new  [WP32] HS, NG, R2D2, DL, SSVS CoefPrior kernels + GLScaleStep (G_COEF_GLOBAL)
├── components/mean/linear.py            [WP32] deviation path (coef_log) for a varying coefficient prior (§3.4)
├── components/mean/linear_reduced.py    [WP32] CCCM deviation path; CCCMGlobal.dev_log; log prior from the scales
├── components/mean/conjugate.py   new   [WP33] ConjugateEq (per-equation no-op mean) + ConjugateNIWStep (G_CONJUGATE)
├── components/structure/kronecker.py new [WP33] Kronecker structure: Σ = CCᵀ in glob.struct
├── components/vol/common.py       new   [WP33] CommonSVStep (G_COMMON_SV), common_h kernels, CommonVol (conjugate)
├── components/vol/constant.py           [WP33] common-h weights for the per-equation program (gvol)
├── components/vol/robust.py             [WP33] passes gvol to the core; log_var adds h_t
├── engines/mcmc/{config,state,keys,registry,protocols}.py [WP26] contract additions of §2
├── engines/mcmc/linalg.py               [WP28] precision_draw_log (deviation, log-space Jacobi), tridiagonal Newton
├── engines/mcmc/random.py               [WP28] log_gig, log_gamma_rate, log_exponential, bernoulli_logit
├── engines/mcmc/resolve.py              [WP32, WP33] GL and CommonSV static configs and prior arrays
├── engines/mcmc/monitor.py              [WP32, WP33] D-52 monitors; warnings §8.3
├── engines/mcmc/forecast.py             [WP33] kronecker extraction; [WP35] nothing (conditional is post-hoc)
├── engines/mcmc/warmstart.py            [WP33] remap glob.vol.h by period
├── engines/analytic.py                  [WP34] forecast attaches lag_coefs (D-49); conditional_forecast method
├── engines/structural_nig.py            [WP34] StructuralNIGFit; EB with free cross; MH over hyperparameters
├── priors/global_local.py         new   [WP27] user-facing GL specs; [WP32] host builders (groups, log c, SSVS ŝ)
├── priors/asymmetric.py           new   [WP27] AsymmetricConjugate spec
├── priors/sv.py                         [WP27] + TruncatedNormal
├── forecast/conditional.py        new   [WP35] conditions parsing, the exact update, ConditionalForecastResult
├── forecast/result.py                   [WP35] subclass hook only (no contract change)
├── forecast/invariants.py               [WP35] check_conditional_structure
├── evaluation/compare.py                [WP27] grid() unchanged (harmonise_grid extended in api.py)
└── validation/                          [WP30] references of §9.1 (reference.py, simulate.py, log_joint.py,
                                                 testfns.py, joint.py: exogenous design, rows 301…)
tools/make_r_fixtures.R                  [WP29] bayesianVARs (P1–P10, parity runs), bvars (P11–P15, parity runs)
.github/workflows/r-fixtures.yml   new   [WP29] workflow_dispatch only; never on push, PR or schedule
.github/workflows/ci.yml                 [WP42] PR/weekly tier additions; tools/weekly_shards.py new shards (§9.11)
benchmarks/pilot/phase3_*.py       new   [WP31] pilot scripts; docs/phase3/pilot.md
benchmarks/gruber_kastner.py       new   [WP41] replication driver; benchmarks/reports/gruber_kastner_*.md
```

Files owned by the parallel sessions are not touched: `src/rai_bvar/validation/dgp.py`,
`src/rai_bvar/validation/__init__.py`, `tests/dgp/`, `benchmarks/dgp_study.py`,
`benchmarks/reports/dgp_study_*` (DGP harness) and every `_bart_spike` file (Phase 4 spike). New validation
helpers are imported from their modules, not re-exported through `validation/__init__.py`, until the DGP PR merges.

### 1.3 Test tree

```text
tests/mcmc/unit/          [WP26] contract additions (state templates, keys, config rules, registry)
                          [WP28] test_gig, test_linalg_log (precision_draw_log vs D-14 and a dense oracle), test_newton
tests/unit/               [WP27] test_specs_phase3, test_api_phase3 (resolution, every error message, hashes, labels,
                                 harmonise), test_spec_roundtrip_phase3
tests/priors/             [WP32] test_global_local_builders (groups, scales, SSVS ŝ, R9)
tests/mcmc/blocks/        [WP32] test_global_local (conditionals, density ratios, block Geweke)
                          [WP33] test_common_sv (h kernel, ρ, σ, NIW step, quadrature), test_kronecker
tests/analytic/           [WP34] test_asymmetric (closed form, EB, MH, forecasts) [WP38] A8–A13
tests/forecast/           [WP35] test_conditional (exact oracle, every engine, contract)
tests/validation/         [WP30] test_reference_phase3 (prior moments, simulators, log joints)
tests/geweke/, tests/sbc/ [WP37] test_*_p3.py (rows 301…), tests/mutation/ [WP37] M19…
tests/parity/             [WP39] test_bayesianvars (unskipped), test_bvars_commonsv, test_boe_conditional
tests/integration/        [WP40] the DGP study's Phase 3 reading (no harness code); [WP41] test_gruber_kastner_script
```

---

## 2. Contracts (WP26)

All additions keep every Phase 2 `StaticConfig`, state template, spec and spec hash unchanged: new fields default
to the Phase 2 value and new leaves are `Empty()` unless a Phase 3 component is present (P2 §2.1). A golden test
pins the state templates and spec hashes of every Phase 2 composition at `2c71fb2`.

### 2.1 Static configuration (`engines/mcmc/config.py`)

```python
StructKind = Literal["recursive", "kronecker"]                 # + kronecker: Σ = CCᵀ owned globally (§5.2)
MeanKind   = Literal["linear_structural", "linear_reduced", "zero", "linear_conjugate"]
CoefKind   = Literal["fixed_diagonal", "horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"]
Persistent = Literal["constant", "sv", "common"]                # common: no per-equation state (conjugate program)

@dataclass(frozen=True)
class MeanStatic:   ... ; coef_log: bool = False                # deviation path of §3.4 (set iff the coef prior varies)
@dataclass(frozen=True)
class CoefPriorStatic:
    kind: CoefKind = "fixed_diagonal"
    n_groups: int = 0                                           # G (shape-defining); 0 for fixed_diagonal
    a_grid: int = 0                                             # size of the discrete hyperprior grid on a (0 = fixed a)
    p_beta: bool = False                                        # SSVS: Beta hyperprior on p
@dataclass(frozen=True)
class VolStatic:    ... ; common: bool = False                  # λ_{i,t} = λ_i e^{h_t} with h in glob.vol (§5.3)
                        csv_block: int = 100                    # h-step MH block length (D-48)
                        csv_rho_prior: Literal["truncated_normal", "beta"] = "truncated_normal"
                        csv_sigma2_prior: Literal["inv_gamma", "scaled_chi2"] = "inv_gamma"
```

`COEF_VARIES` gains `True` for every GL kind. `validate_composition` adds, in order: (11) a varying coefficient
prior needs `coef_log = True` and `const_path = "weighted"`, and excludes `coef_scaling = "lambda"` (Mode II);
(12) `coef_log` needs a `linear_structural` or `linear_reduced` mean; (13) `common = True` needs
`persistent = "constant"` and `transitory = "none"`; (14) `persistent = "common"` needs `kind = "kronecker"`
structure and `linear_conjugate` mean, and conversely; (15) `n_groups ≥ 1` iff a GL kind; `a_grid > 0` only for NG,
R2D2 and DL; `p_beta` only for SSVS; (16) `csv_block ≥ 2`.

### 2.2 State (each new class registered with `register_dataclass`; every scale leaf is a log)

```text
LinearEq          delta: f64[n, k]; dev_log: f64[n, k] | Empty        log|δ_i − m_i| (§3.4); Empty when coef_log is off
CCCMGlobal        Pi: f64[k, n];    dev_log: f64[n, k] | Empty        log|π_j − m_j|, row j = equation j
HSEq              log_theta, log_nu: f64[n, k]                        local ϑ, auxiliary ν (0 on non-GL slots)
HSGlobal          log_zeta, log_tau: f64[G]
NGEq              log_w: f64[n, k]                                    ϖ
NGGlobal          log_xi: f64[G]; a_idx: i32[G] (grid) | a fixed      rate ξ; hyperprior index
R2D2Eq            log_psi, log_w: f64[n, k]                           ψ (exponential mixing), ϖ
R2D2Global        log_xi: f64[G]; a_idx: i32[G]
DLEq              log_theta, log_psi: f64[n, k]                       θ̃ (Laplace scale), ψ
DLGlobal          a_idx: i32[G]
SSVSEq            gamma: i32[n, k]; rb_incl: f64[n, k]                indicator; Rao–Blackwellised P(γ = 1 | ·) sum
SSVSGlobal        log_p, log1m_p: f64[G]
KronGlobal        C: f64[n, n]                                        lower Cholesky factor of Σ (glob.struct, §5.2)
CommonSVGlobal    h: f64[T_pad]; rho, sigma: f64[]; acc_h, acc_rho, acc_newton: f64[]; rb_h_sum, rb_h_sq: f64[T_pad]
```

`EqState.coef` holds the `*Eq` class, `GlobalState.coef` the `*Global` class, `GlobalState.struct` holds
`KronGlobal` and `GlobalState.vol` holds `CommonSVGlobal` (both `Empty` in Phase 2). Draw containers (P2 §2.10):
`GlobDraw.struct = KronDraw(C)`, `GlobDraw.vol = CommonSVDraw(h_T, rho, sigma)`; the GL scales are not part of
`ParamDraw` (they do not enter forecasts) but are stored in the `posterior` group as `log_scale_group` and, at
`store="paths"`, the local scales. Invariants (P2 §2.8) gain: every GL leaf finite on GL slots and exactly 0 on
other slots; `a_idx` in range; `γ ∈ {0, 1}`; `|ρ| < 1`, `σ_h > 0`; `C` lower triangular with positive diagonal.

### 2.3 Prior arrays (`resolve.py` from the WP32/WP33 builders; traced)

```text
CoefPriorArrays  ... + gl: GLArrays | Empty
GLArrays         mask: f64[n, k]        1 on GL-governed slots (lag coefficients), 0 elsewhere
                 group: i32[n, k]       group id g(j) on GL slots (0 elsewhere, masked)
                 log_c: f64[n, k]       log scale c_j (D-37); 0 when scale="none"
                 log_tau0, log_tau1: f64[n, k]    SSVS spike and slab log s.d. (Empty-valued zeros otherwise)
SharedPrior      ... + gl: GLShared | Empty;  niw: NIWPrior | Empty;  csv: CommonSVPrior | Empty
GLShared         b, c, s_global: f64[]  (per family; unused fields 0); log_a_fixed: f64[]
                 a_support, a_logw: f64[A]              discrete hyperprior on a (A = CoefPriorStatic.a_grid)
                 p_fixed, s1, s2: f64[]                 SSVS
NIWPrior         Yd: f64[m_d, n]; Xd: f64[m_d, k]       every prior row of the Phase 1 conjugate system (Minnesota, Ψ, SOC, SUR)
                 nu0: f64[]                              IW degrees of freedom of the stacked system (§5.2)
CommonSVPrior    rho_mean, rho_var, rho_a, rho_b: f64[]; s2_shape, s2_scale, s2_B: f64[]
```

### 2.4 Keys, registry, overridable internals

- `Block` (append-only, P2 §5.6): `G_COEF_GLOBAL = 17` (post, the GL scale step), `G_COMMON_SV = 18` (post),
  `G_CONJUGATE = 20` (pre, the NIW draw). 19 stays reserved for an estimated Lenza–Primiceri step (Q16).
- `Sub` (append-only): `GL_LOCAL = 15`, `GL_LOCAL_AUX = 16`, `GL_GROUP = 17`, `GL_GROUP_AUX = 18`, `GL_A = 19`,
  `CSV_OFFSET = 20`, `CSV_H = 21`, `CSV_H_ACC = 22`, `CSV_RHO = 23`, `CSV_RHO_ACC = 24`, `CSV_S2 = 25`,
  `NIW_SIGMA = 26`, `NIW_PI = 27`. Per-coordinate draws fold the flat coordinate index into the sub key
  (`fold_in(sub_key, i·k + r)`, uint32), per-group draws the group id, per-block `h` draws the block index.
- `registry.KERNELS` appends `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs`
  (`components.coef.global_local:{Horseshoe, NormalGamma, R2D2, DirichletLaplace, SSVS}`), `gl_scales`
  (`…:GLScaleStep`), `kronecker` (`components.structure.kronecker:Kronecker`), `linear_conjugate`
  (`components.mean.conjugate:ConjugateEq`), `conjugate_niw` (`…:ConjugateNIWStep`), `common`
  (`components.vol.common:CommonVol`), `common_sv` (`…:CommonSVStep`).
- `protocols.INTERNALS` appends `gl_local_conditional`, `gl_group_conditional`, `gl_a_conditional`,
  `ssvs_logodds`, `gl_deviation`, `csv_newton_start`, `csv_q`, `csv_rho_accept`, `niw_posterior`, `niw_draw`,
  `conditional_update` (test seam for the mutants of §9.6; production never sets them).
- `protocols.GaussianLogDiag(mean: f64[m], log_prec: f64[m], cmask: f64[m])`: what a varying `CoefPrior.moments_eq`
  returns. A mean receiving it takes the deviation path (static `coef_log`); `GaussianDiag` keeps the Phase 2 path.

---
## 3. Global-local and selection priors

### 3.1 Coordinates, deviation and kernel (D-36, D-37)

Lag slot `r ∈ {n_d, …, k−1}` of the `data/design.py` layout (deterministic terms, then the lag-1 block of `n`
variables, then lag 2, …) has lag `l(r) = 1 + ⌊(r − n_d)/n⌋` and variable `m(r) = (r − n_d) mod n`; it is an own lag of
equation `i` iff `m(r) = i`. A **GL coordinate** `j = (i, r)` is a lag slot of equation `i`; there are `N = n²p`.
The coefficient is `δ_{i,r}` (structural, δ-coordinates of D3: `γ_i = δ_i + M_c a_i`, P2 §2.3) or `π_{r,i}`
(reduced). Its prior mean `m_j` is the transform-aware own-lag mean at own lag 1 and 0 elsewhere (the Phase 2
arrays' `mean0`). The prior governs the deviation `d_j = coefficient − m_j`:

```text
d_j | v_j  ~ N(0, c_j v_j)  independently,      log c_j = log ψ_i − log ι − log ψ_m   (structural, scale="psi")
                                                log c_j = log ψ_i − log ψ_m           (reduced,    scale="psi")
                                                log c_j = 0                           (scale="none")
```

`v_j` is the family's mixing variance (§3.2). The group of `j` is `g(j)` (D-38; `n_g` counts the GL coordinates of
group `g`). Deterministic slots and `a_i` keep the Phase 2 Normal priors (D-36). Every density and conditional
below is written for `ℓ_j = log|d_j|`, which the mean kernels deliver exactly (§3.4); `d_j²/c_j` is always
`exp(2ℓ_j − log c_j)`.

### 3.2 The five families (D-39)

Each family lists its prior, its state (all logs; P2 §2.1 rules), its exact conditionals in update order, and its
`bayesianVARs` counterpart (mapping settled by probe P8). `lnΓ` is `gammaln`; `LG(α)` is a draw of `log G(α, 1)`
(`jax.random.loggamma`, exact for tiny `α`); every gamma and inverse gamma below is drawn as
`log x = LG(α) − log β` (gamma, rate `β`) or `log x = log β − LG(α)` (inverse gamma, scale `β`).

**Horseshoe** (Carvalho, Polson & Scott 2010; auxiliary form of Makalic & Schmidt 2016). `v_j = ϑ_j ζ_g`,
`√ϑ_j ~ C⁺(0, 1)`, `√ζ_g ~ C⁺(0, s)` (`s = global_scale`, default 1), written as
`ϑ_j | ν_j ~ IG(½, 1/ν_j)`, `ν_j ~ IG(½, 1)`, `ζ_g | τ_g ~ IG(½, 1/τ_g)`, `τ_g ~ IG(½, 1/s²)`. State
`HSEq(log_theta, log_nu)`, `HSGlobal(log_zeta, log_tau)`. Conditionals (*derived here*; GK App. B prints `τ_j`
where `ζ_g` belongs in the first line, which mutant M20 reproduces):

```text
ϑ_j | ν_j, ζ_g, d_j  ~ IG(1, 1/ν_j + d_j²/(2 c_j ζ_g))
ν_j | ϑ_j            ~ IG(1, 1 + 1/ϑ_j)
ζ_g | τ_g, ϑ, d      ~ IG((n_g + 1)/2, 1/τ_g + Σ_{j∈g} d_j²/(2 c_j ϑ_j))
τ_g | ζ_g            ~ IG(1, 1/s² + 1/ζ_g)
```

Log scales: `log(1/ν_j + d_j²/(2c_jζ_g)) = logaddexp(−log ν_j, 2ℓ_j − log 2 − log c_j − log ζ_g)`; the group sum is a
`segment_logsumexp`. No GIG is needed. `bayesianVARs`: `prior="HS"`.

**Normal-Gamma** (Griffin & Brown 2010; GK's parametrisation). User parameters `a, b, c`: `ϑ_j ~ G(a_g, a_g/2)`,
`ζ_g⁻¹ ~ G(b, c)`, `v_j = ϑ_j ζ_g`. Equivalently `v_j = ϖ_j | ξ_g ~ G(a_g, ξ_g)` with the rate `ξ_g = a_g/(2ζ_g) ~
G(b, 2c/a_g)`. `c="a/2"` ties `c` to `a` (GK's default), so `2c/a = 1`. State `NGEq(log_w)`,
`NGGlobal(log_xi, a_idx)`. Conditionals:

```text
ϖ_j | d_j, ξ_g     ~ GIG(a_g − ½, 2ξ_g, d_j²/c_j)
(a_g, ξ_g) | ϖ     collapsed: draw a_g from its marginal, then ξ_g | a_g, ϖ
  log P(a_g = ã | ϖ) = log π(ã) + (ã − 1) S_g − n_g lnΓ(ã) + b log(2c/ã) + lnΓ(n_g ã + b)
                       − (n_g ã + b) log(2c/ã + W_g) + const,   S_g = Σ_{j∈g} log ϖ_j,  W_g = Σ_{j∈g} ϖ_j
  ξ_g | a_g, ϖ    ~ G(n_g a_g + b, 2c/a_g + W_g)
```

With a fixed `a` only the `ξ_g` line runs. The collapsed `a`-weight keeps the `p(ξ_g | a)` factor `b log(2c/ã)`, which
GK's printed form omits (it is constant only when `c` is tied); mutant M22 drops it. `log W_g` is a
`segment_logsumexp`. `bayesianVARs`: `prior="NG"`, `NG_a, NG_b, NG_c`.

**R2D2** (Zhang, Naughton, Bondell & Reich 2022; GK's gamma form). User parameters `a, b`:
`v_j = ψ_j ϖ_j/2`, `ψ_j ~ Exp(½)`, `ϖ_j | ξ_g ~ G(a_g, ξ_g)`, `ξ_g ~ G(b, 1)` (with a Dirichlet split this is the
R2D2 prior with `R² ~ Beta(n_g a, b)` in each group). State `R2D2Eq(log_psi, log_w)`, `R2D2Global(log_xi, a_idx)`.

```text
ψ_j | d_j, ϖ_j        ~ GIG(½, 1, 2 d_j²/(c_j ϖ_j))
ϖ_j | d_j, ψ_j, ξ_g   ~ GIG(a_g − ½, 2ξ_g, 2 d_j²/(c_j ψ_j))
(a_g, ξ_g) | ϖ        log P(a_g = ã | ϖ) = log π(ã) + (ã − 1) S_g − n_g lnΓ(ã) + lnΓ(n_g ã + b) − (n_g ã + b) log(1 + W_g)
                      ξ_g | a_g, ϖ ~ G(n_g a_g + b, 1 + W_g)
```

`bayesianVARs`: `prior="R2D2"`, `R2D2_a, R2D2_b`.

**Dirichlet–Laplace** (Bhattacharya, Pati, Pillai & Dunson 2015; gamma form). User parameter `a`:
`d_j | ψ_j, θ̃_j ~ N(0, c_j ψ_j θ̃_j²)`, `ψ_j ~ Exp(½)`, `θ̃_j ~ G(a_g, ½)`. This is the DL prior: with
`θ̃_j = ϱ_j ω`, `ϱ ~ Dir(a, …, a)` and `ω ~ G(n_g a, ½)` the products are iid `G(a, ½)` (gamma–Dirichlet), and the
kernel `N(0, ψ θ̃²)` with `ψ ~ Exp(½)` is the Laplace density with scale `θ̃`. There is no global parameter
besides `a_g`. State `DLEq(log_theta, log_psi)`, `DLGlobal(a_idx)`.

```text
θ̃_j | d_j (ψ_j integrated)   ~ GIG(a_g − 1, 1, 2|d_j|/√c_j)
ψ_j | d_j, θ̃_j               ~ GIG(½, 1, d_j²/(c_j θ̃_j²))
a_g | θ̃                      log P(a_g = ã | θ̃) = log π(ã) − n_g ã log 2 + (ã − 1) Σ_{j∈g} log θ̃_j − n_g lnΓ(ã)
```

The first two lines are one exact block draw of `(θ̃_j, ψ_j) | d_j`. `bayesianVARs`: `prior="DL"`, `DL_a`.

**SSVS** (George & McCulloch 1993; George, Sun & Ni 2008). `d_j | γ_j ~ N(0, τ²_{γ_j, j})`, `γ_j | p_g ~
Bernoulli(p_g)`, `p_g` fixed or `~ Beta(s₁, s₂)`; the spike and slab standard deviations are those of D-44. State
`SSVSEq(gamma, rb_incl)`, `SSVSGlobal(log_p, log1m_p)`.

```text
logit P(γ_j = 1 | ·) = log p_g − log(1 − p_g) + log τ_{0j} − log τ_{1j} − ½ (d_j²/τ_{1j}² − d_j²/τ_{0j}²)
p_g | γ              ~ Beta(s₁ + N1_g, s₂ + n_g − N1_g),   N1_g = Σ_{j∈g} γ_j
```

`γ_j` is drawn as `log u < log σ(logit)` with one uniform; `rb_incl` accumulates `P(γ_j = 1 | ·)`. The kernel's
log precision is `−2 log τ_{γ_j, j}` (no `c_j`). `bayesianVARs`: `prior="SSVS"`, `SSVS_c0, SSVS_c1,
SSVS_semiautomatic, SSVS_p`.

**Hyperprior on `a`** (NG, R2D2, DL). `a` is a float (fixed) or a `DiscreteGrid(support, weights)`; the default
`DiscreteGrid.gruber_kastner()` is GK's: 1,000 equally spaced points on `[0.001, 1]` with weights proportional to
the `Exp(1)` density. The grid is data (`GLShared.a_support`, `a_logw`); only its size is static. Each group has
its own `a_g` (semi-global, GK `NG*_a`, `R2D2*_a`); the draw is one inverse-CDF categorical on the `A` log weights.

**Defaults** (Q23). `Horseshoe()`; `NormalGamma(a=DiscreteGrid.gruber_kastner(), b=0.5, c="a/2")`;
`R2D2(a=DiscreteGrid.gruber_kastner(), b=0.5)`; `DirichletLaplace(a=DiscreteGrid.gruber_kastner())`;
`SSVS(c0=0.01, c1=100.0, semiautomatic=True, p=0.5)`; all with `groups="olcl-lagwise"`, `scale="psi"`. These are GK's
best-performing configurations (semi-global, hyperprior on `a`); `bayesianVARs`' defaults (global grouping, fixed
`a`, `NG_c = 1`) are set explicitly in the parity runs.

### 3.3 The scale step `G_COEF_GLOBAL` and the sweep (D-40)

`GLScaleStep` is the `GlobalStep` of every GL coefficient prior (`CoefPrior.global_steps()` returns it; P2 §3.8
item 4 orders it). Phase `post`; `reads = {eq.mean, glob.mean, eq.coef, glob.coef}`; `writes = {eq.coef,
glob.coef}`; `marginal_of_indicators = True` (it never reads `eq.vol`). It takes `ℓ` from `eq.mean.dev_log`
(structural) or `glob.mean.dev_log` (reduced) and updates, in the family's order of §3.2, every local parameter (all
`N` coordinates at once, elementwise), then every group parameter (segment sums over `G` static groups), then the
hyperprior index. Sweeps:

```text
structural:  equation   (δ_i, a_i) | v, λ_eff,i  (deviation path, §3.4)    Block.EQ_MEAN    writes eq.mean, eq.struct.a
             residual · vol (P2 §4.4, unchanged)
             post       gl_scales  local | d → group | local → a | local    Block.G_COEF_GLOBAL   writes eq.coef, glob.coef
reduced:     pre        cccm       Π | A, λ_eff, v  (deviation path)         Block.G_CCCM     writes glob.mean
             contemp · residual · vol (unchanged)
             post       gl_scales  (as above, ℓ from glob.mean.dev_log)
```

Why it targets the posterior: the scale parameters enter only the coefficient prior, so given the coefficients they
are independent of the data, the volatility and the indicators; each line of §3.2 is an exact full conditional or
an exact block draw (the collapsed `(a_g, ξ_g)` and `(θ̃_j, ψ_j)` draws); the mean step conditions on the current
scales through `moments_eq`. Plan invariants (P2 §4.8) hold as stated there: post steps write only `glob.*` and
`eq.coef.*`, never a residual input. Density-ratio identities (Tier U) gate every line (§9.2).

`moments_eq` returns `GaussianLogDiag(mean0, log_prec, cmask)` with `log_prec_j = −log c_j − log v_j` on GL slots
(`log v_j` from the state: `log ϑ + log ζ`, `log ϖ`, `log ψ + log ϖ − log 2`, `log ψ + 2 log θ̃`, `−log c_j +
2 log τ_{γ_j j}` for SSVS) and `log prec0` elsewhere.

### 3.4 The deviation draw (D-41)

Shrunk deviations can be far below the resolution of `m + d` (an own-lag coefficient with `m = 1`) and below the
smallest float (a coefficient whose prior s.d. is `e^{−800}`); the scale step needs `log|d|` in both cases, and a
precision of `e^{1600}` overflows. The mean kernels therefore draw in deviation coordinates with the Jacobi
scaling done in log space (static `MeanStatic.coef_log`; `linalg.precision_draw_log`):

```text
K = diag(P) + G,   log P_j given (log_prec),   G ≥ 0 the data Gram (masked as in P2 §2.3)
b̃ = data right-hand side in deviation coordinates    (structural: F_θᵀ(w ⊙ (target − F_θ meanθ)); CCCM: Xᵀ(G_j + W_j ⊙ (f_j − X m_j)))
log s_j = −½ logaddexp(log P_j, log G_jj)                       never overflows
A = I + offdiag(exp(log s_i + log s_j) G_ij),  L = chol(A)       unit diagonal, as D-14's sKs
u = L⁻ᵀ(L⁻¹(s ⊙ b̃) + z),   d = exp(log s) ⊙ u,   dev_log = log s + log|u|,   coefficient = mean + d
```

This is D-14's draw (`θ = s ⊙ L⁻ᵀ(L⁻¹(s ⊙ b) + z)` with `b = P ⊙ mean + data`) rewritten with `b̃ = b − K·mean`, so
it is the same function of `z` to rounding wherever D-14 is finite (Tier U: 1e-12 relative at 1e8 disparities,
P2 §9.3). `d` may underflow to 0 harmlessly (its contribution to every residual is below rounding); `dev_log` stays
exact, so no scale conditional ever sees `χ = 0`. With `coef_log = False` the Phase 2 path runs (bitwise unchanged).
`StepInfo.ok` as D-14 (finite factor, positive diagonal). The structural fused draw applies it to the whole
`θ_i = [δ_i | a_i]` (fixed prior precisions on the non-GL coordinates), the CCCM step to each `π_j`.

### 3.5 Sampling the GIG on the log scale (D-42; `random.log_gig`)

`log_gig(key, lam, log_psi, log_chi) -> (log x, ok)`, one problem, batched by `vmap`:

1. `log χ = −∞`, `λ > 0`: `log x = LG(λ) − log(ψ/2)` (gamma). `log ψ = −∞`, `λ < 0`: `log x = log(χ/2) − LG(−λ)`
   (inverse gamma). Otherwise both are finite; an improper input (`χ = 0` with `λ ≤ 0`, or `ψ = 0` with `λ ≥ 0`)
   returns `ok = False` and becomes a status code (P2 D-18). By §3.4 it never occurs in a production sweep.
2. Two-parameter form: `x = η y`, `log η = ½(log χ − log ψ)`, `log ω = ½(log ψ + log χ)`; `u = log y` has density
   `∝ exp(f(u))`, `f(u) = λu − ω cosh u`, strictly concave. For `λ < 0` sample with `|λ|` and negate `u`.
3. Mode `u* = asinh(λ/ω)` (`log 2 + log λ − log ω` when `λ/ω > 1e8`); curvature `f''(u*) = −√(λ² + ω²)`. Work in
   `v = u − u*` with `f(u) − f(u*) = λv − ½[e^{log ω + u*} expm1(v) + e^{log ω − u*} expm1(−v)]` (no overflow for
   `log ω` down to −10⁴).
4. Points `v₋ < 0 < v₊` near `f = f(u*) − 1`: four Newton steps from `∓√2/(λ² + ω²)^{¼}`, clamped to their side.
   Envelope: `0` on `[v₋, v₊]`, tangent lines `f(v±) + f'(v±)(v − v±)` outside. By concavity the envelope bounds `f`
   for **any** such points, so exactness does not depend on the Newton accuracy.
5. Pick a piece with probabilities proportional to its mass (centre `v₊ − v₋`; tails `e^{f(v±)}/|f'(v±)|`), draw
   `v` in it (uniform or shifted exponential), accept iff `log U ≤ f(v) − envelope(v)`. Attempt `t` uses
   `fold_in(key, t)`; a `lax.while_loop` runs until every lane accepts or 64 attempts pass (then `ok = False`).
6. `log x = log η + sign·(u* + v)`.

Tier U (WP28): Kolmogorov–Smirnov against `scipy.stats.geninvgauss` on a parameter grid; `E log X` against the exact
`log η + ∂_λ log K_λ(ω)` (mpmath at 50 digits) for `λ ∈ [−50, 50]`, `log ω ∈ [−700, 12]`; the `χ → 0` and `ψ → 0`
limits against gamma and inverse-gamma draws; lane invariance under `vmap` (1e-15); determinism by key; mean
attempts reported (expected ≤ 1.5); no `ok = False` on the grid.

### 3.6 Host builders (`priors/global_local.py`; WP32)

`gl_arrays(prior, scales, *, n, lags, n_det, form) -> (GLArrays, GLShared)` per window (R9): the GL mask (lag
slots), group ids (D-38), `log c` (D-37, from the window's ψ), SSVS `log τ₀, log τ₁` (D-44: `ŝ` from one QR of the
window's design per equation, diagonal of `R⁻¹R⁻ᵀ` by triangular solves, no inverse; reduced: `ŝ²_{rj} =
Ŝ_jj/(T − k − n − 1)·[(XᵀX)⁻¹]_rr` from the flat-prior normal–Wishart posterior), and the hyperprior arrays. A custom
`groups` array is checked against `(n, n·p)` here (its shape is unknown at construction) with an actionable error.

### 3.7 Initialisation, log prior, storage

- **Init** (`Stream.INIT`, P2 §5.4): local scales at 1 (`log = 0`, HS `ν = 1`, DL `θ̃ = 2`, `ψ = 2`), group scales
  so that `c_j v_j` equals the Mode I Minnesota variance at `λ = 0.2` (`log ζ_g = log 0.04` for HS, the matching rate
  for NG/R2D2), `a_g` at the grid point nearest 0.1, SSVS `γ = 1`, `p_g = 0.5`; every group log scale jittered by
  `N(0, 0.5²)` per chain. The first coefficient draw then uses these moments (P2's structural and CCCM cold starts).
- **Log prior** (monitors and validation): `Σ_j log N(d_j; 0, c_j v_j)` plus the scales' log densities, computed
  from the full chain state; `CCCMStep.log_prior` keeps its fixed-moment form for `FixedDiagonal` only.
- **Storage**: `posterior/log_scale_group (chain, draw, group)`, `a_g`, SSVS `p_g` every stored draw;
  `latent_moments/incl_prob (equation, coefficient)` from `rb_incl` (SSVS); local scales only at `store="paths"`.

### 3.8 Labels and spec

`Horseshoe.label = "Horseshoe(structural, olcl-lagwise)"` plus every non-default setting (as `Minnesota.label`);
`axes["prior"]` is the label. Specs carry `"class": "Horseshoe"` etc. (absent `"class"` still means `Minnesota`), so
Phase 1 and Phase 2 hashes are unchanged (golden test). GL priors in structural form are not order-invariant: the
R8 note of P2 §8.3 applies.

---

## 4. The asymmetric conjugate prior (Chan 2022) on the analytic engine (D-45; WP34)

### 4.1 Prior

`AsymmetricConjugate(lambda_, cross, contemp, alpha, psi, mean, var_const, soc, sur, soc_scope, hierarchy, hyper,
hyperpriors, mh)`: per equation `i`, in δ-coordinates with D3 centring (P2 §6.1 Mode II), `θ_i = [δ_i | a_i] |
λ_i ~ N(μ₀, λ_i V₀)`, `λ_i ~ IG(1 + ι/2, ψ_i/2)`, with `V₀` the Mode II diagonal: own lag `l` `λ²(d−n−1)/(l^α ψ_i)`,
lag `l` of variable `m ≠ i` `λ² cross² (d−n−1)/(l^α ψ_m)`, deterministic `var_const`, `a_ij` `contemp/ψ_j`. At
`cross = contemp = 1` it is the Phase 1 NIW prior exactly (P2 A1). Equations are independent a priori and `|A| = 1`,
so the posterior factorises and is closed form (`engines/structural_nig.py`, P2 WP16).

### 4.2 Engine and fit

`BVAR(Linear(AsymmetricConjugate(...)), Constant())` resolves to `engine == "analytic"`, `Constant("auto") → "iw"`
(the NIG scaling), and fits `StructuralNIGFit` (new class in `engines/structural_nig.py`), with the interface of
`AnalyticFit`: `log_ml(hyper=None)`, `posterior(hyper=None)` (the per-equation `NIGPosterior`s),
`draw_parameters(size, rng)` (iid `(a, γ, λ)` → `B = (A⁻¹Γ)'`, `chol = A⁻¹Λ^{½}`), `forecast(horizon, draws, seed)`
(`simulate_linear_var` on iid draws; attrs as `AnalyticFit` plus `ordering`, and `lag_coefs`, D-49), `irf`,
`conditional_forecast` (§6), `summary()`, `hyper`, `converged`, `warnings` (Phase 1 setting checks; the R8 note for
`n ≥ 4`, since `cross ≠ 1` is not order-invariant, Chan Cor. 2).

### 4.3 Hyperparameters

- `"fixed"`: as given.
- `"eb"` (default; free set `hyper="auto"` = `("lambda", "cross")` plus `soc`/`sur` when on): maximise
  `Σ_i log p(y_i | h) + log p(h)` with `eb_structural`, extended so `cross` (and, if listed, `contemp`) can be free.
  `lambda`, `soc`, `sur` keep the GLP hyperpriors; `cross` and `contemp` have none (marginal-likelihood
  maximisation over `log cross ∈ [log 1e-3, log 10]`, as Chan 2022 §4), stated in the label. Start: the Phase 1 EB
  mode at `cross = 1` (P2 §6.5 step 2). `alpha` free when listed, as Phase 1.
- `"mh"`: random-walk Metropolis on the log free hyperparameters with Phase 1's `_metropolis` (proposal from the
  inverse Hessian at the EB mode, `MHSettings`, its convergence rule), then iid parameter draws per kept MH draw, as
  Phase 1's `hierarchy="mh"`.

### 4.4 What is rejected (construction time)

Any volatility other than `Constant("auto")`/`Constant("iw")` (`SV`, `CommonSV`, `Constant("ig")`,
`Constant("iw_equivalent")`); `outliers="lenza-primiceri"` (GLS rows would keep the closed form, but it is not in
v0.3.0); `psi` in the free set with `cross` free (as P2 §11.1). Messages in §7.3.

---

## 5. `CommonSV` (D-46 … D-48; WP33)

### 5.1 Model and priors

`u_t = y_t − Π'x_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `h_1 ~ N(0, σ_h²/(1 − ρ²))`, `E h_t = 0`
(N08 §A1–A2). In recursive form `Σ = A⁻¹ΛA⁻ᵀ` and `λ_{i,t} = λ_i e^{h_t}`. Given `(Π, Σ)` the data enter the `h`
conditional through `q_t = u_t'Σ⁻¹u_t` only (N08 §A4). Spec (`components/specs.py`, pure Python):

```python
@dataclass(frozen=True)
class CommonSV:
    rho_prior: TruncatedNormal | Beta = TruncatedNormal(0.9, 0.04)   # ρ ~ N(0.9, 0.2²)·1(|ρ|<1); Beta is on (ρ+1)/2
    sigma2_prior: InvGamma | ScaledChi2 = InvGamma(5.0, 0.04)       # σ_h²: E = 0.01 (Chan 2020)
    block: int = 100                                                # h-step MH block length (D-48)
```

`name = "CommonSV"`, `label` adds non-default settings; `TruncatedNormal(mean, var, low=-1.0, high=1.0)` joins
`rb.sv_priors`. CommonSV is a stochastic-volatility composition for D-35 (D-51).

### 5.2 Program (b): conjugate (`Minnesota(form="conjugate") × CommonSV()`)

```text
pre       conjugate_niw  G_CONJUGATE (20)   (Π, Σ) | h, Y        writes glob.mean.Pi, glob.struct.C
equation  residual · vol                    no-ops: the Kronecker structure has no per-equation state, the CommonVol block none
post      common_sv      G_COMMON_SV (18)   h, ρ, σ_h | q        writes glob.vol
```

- **`(Π, Σ) | h`** (N08 §A5.1; Chan 2020 §3): the Phase 1 conjugate system — Minnesota rows, Ψ rows, SOC/SUR rows,
  all **unweighted** — stacked on the data rows scaled by `w_t = m_t e^{−h_t/2}` (the Lenza–Primiceri GLS path of
  SPEC §4.6 with `s_t² = e^{h_t}`), one Householder QR, `B̄`, `Ŝ`, `ν̂ = ν₀ + T_eff` read off the factor as
  `ConjugateProblem._core` does; `Σ ~ IW(Ŝ, ν̂)` by Bartlett, `C = chol(Σ)`, `Π = B̄ + R_xx⁻¹ Z Cᵀ`. A JAX port of
  `ConjugateProblem._core` and of `rai_bvar.linalg`'s IW and matrix-normal draws (WP33), pinned to the NumPy engine
  at fixed `h` (posterior moments and `log|R|` to 1e-10; draws equal on injected noise). Not CCM (2016) eq. (23)
  (N08 §A10.1; mutant M32). Cost per sweep `O((T + m_d)(k + n)²)`: the Kronecker speed.
- **`q_t`** `= ‖C⁻¹u_t‖²`, formed as `2 log‖C⁻¹u_t‖` on real rows.
- **Prior hyperparameters**: `hierarchy="fixed"` exact; `"eb"` plugs in the Phase 1 EB mode of the conjugate twin at
  constant volatility (the D-12 rule, a documented plug-in approximation under CommonSV; N08 §A6 step 6 is
  deferred with Q15); `"mh"` is rejected (Q15).
- **Order invariance**: exact (N08 §A1), so no R8 note.

### 5.3 Program (a): per-equation (`Linear(structural | reduced | GL) × CommonSV()`)

`VolStatic(persistent="constant", common=True)`: each equation keeps its Constant core `λ_i ~ IG(1 + ι/2, ψ_i/2)`
(Mode I) with the conditional `IG(shape_i + T_eff/2 + n_active/2, scale_i + ½ Σ_t m_t exp(log e²_{it} − h_t) +
quad)`; `log_var_eq(vs, gvol) = log λ_i + h_t`, so `constant_weights()` is False and the mean takes the weighted
path (P2 §3.7.2). The post step forms `log q_t = logsumexp_i(log e²_{it} − log λ_i)` from `SweepCtx.eq_out` and the
cores. Order within a sweep: means and `a` given `λ_eff` → `λ_i | e_i, h` → `h, ρ, σ_h | q`: each an exact
conditional. Coupled cores (Mode II, `iw_equivalent`) are legal at engine level; user-facing `CommonSV` with a
structural or reduced Minnesota uses Mode I. Mode II × CommonSV at `cross = contemp = 1` targets exactly the
posterior of program (b) (N08 §A11 item 2): oracle A8. Not order-invariant (triangular), so the R8 note applies.

### 5.4 The `h, ρ, σ_h` kernel (shared; N08 §A5.2–A5.3)

Dates are 1-based in this section, as in N08 (`h_t` is stored at index `t − 1`; `t ≤ T_eff`).

```text
ℓ(h) = −(n/2) Σ_t m_t h_t − ½ Σ_t m_t exp(log q_t − h_t) − ½ hᵀK_h h,   K_h = σ_h⁻² tridiag([1, 1+ρ², …, 1+ρ², 1], −ρ) (masked)
blocks: offset o ~ U{0, …, block−1} (Sub.CSV_OFFSET); block b covers [o + (b−1)·block, o + b·block) ∩ [0, T_eff); ⌈T_pad/block⌉ + 1 blocks (static)
for each block B, given h outside B:
    Newton from h⁰_t = log q_t − log n on B (depends on q and the neighbours only), damped (halving ≤ 30), ≤ 50 iterations,
    stop when ‖Δ‖∞ < 1e-10; Hessian G_B = K_BB + diag(½ q e^{−h}) (tridiagonal, SPD)
    h* ~ N(ĥ_B, G_B⁻¹) by tridiag_precision_sample (Sub.CSV_H, block);  log α = ℓ(h*) − ℓ(h) + ½‖L_Bᵀ(h* − ĥ)‖² − ½‖L_Bᵀ(h − ĥ)‖²
    accept iff log U (Sub.CSV_H_ACC, block) < log α
ρ | h, σ_h:  K_ρ = Σ_{t=2}^{T_eff} h_{t−1}²/σ_h² (+ 1/V_ρ for TruncatedNormal),  ρ̂ = K_ρ⁻¹(Σ_{t=2}^{T_eff} h_{t−1}h_t/σ_h² (+ ρ₀/V_ρ))
             ρ* ~ TN(ρ̂, K_ρ⁻¹; −1, 1);  log α = log g(ρ*) − log g(ρ) (+ the Beta prior ratio),  g(ρ) = ½ log(1−ρ²) − (1−ρ²)h_1²/(2σ_h²)
σ_h² | h, ρ: InvGamma(c₀, C₀): IG(c₀ + T_eff/2, C₀ + ½S);  ScaledChi2(B): GIG(½ − T_eff/2, 1/B, S),
             S = (1−ρ²)h_1² + Σ_{t=2}^{T_eff} (h_t − ρh_{t−1})²
```

The proposal of each block depends only on `(q, ρ, σ_h, h outside the block)`, so it is an independence proposal
whatever Newton's accuracy (N08 §A5.2); the random offset is an auxiliary choice of kernel independent of the state.
Padded states (`t ≥ T_eff`) are held at exactly 0 and never enter a sum (invariant). Diagnostics: `acc_h` (block
average), `acc_rho`, and `acc_newton` (iterations that hit the cap; the warning of §8.3 if > 0, the draw stays
exact). There is no mixture indicator, so the DNP rule (P2 §4.6 item 4) does not arise.

### 5.5 Forecasting, warm starts, initialisation, monitors

- **Forecast**: `CommonSVStep.future(key, gd, H)` draws `h_{T+j} = ρ h_{T+j−1} + σ_h η_j` from `h_T = h[T_eff−1]`
  (P2 §3.6 hook); the volatility blocks add it: (a) `log λ_i + h_{T+j}`, (b) `h_{T+j}`. `Kronecker.chol(sd,
  log_lam) = C·diag(exp(½ log_lam))`, `mean_map(sd, g, f) = f = Π'x`; `lag_coefs` from `Π`. N08 §A8.
- **Warm start** (P2 §5.5): `glob.vol.h` is remapped by period like `eq.vol` `h` (dropped periods removed, new
  trailing ones simulated forward with the copied `(ρ, σ_h)`); `(Π, C)`, `λ_i` copied.
- **Init**: `h ≡ 0`, `ρ` = prior mean (clipped to (−0.99, 0.99)), `σ_h²` = prior mean, `log σ_h` jittered by
  `N(0, 0.25²)` per chain; (b) one NIW draw at `h = 0`; (a) the Phase 2 Constant init.
- **Monitors** (D-52): primary log-likelihood `Σ_t m_t[−½(n log 2π + n h_t + log|Σ| + q_t e^{−h_t})]`, own first-lag
  reduced-form coefficients, `h_T`, `ρ`, `σ_h`, and `log λ_i` (a) or `log Σ_ii` (b); secondary `h_1`, `h_mid`, the
  Cholesky correlations of `Σ`.

---

## 6. Conditional forecasts (D-50; WP35)

### 6.1 API

```python
fit.conditional_forecast(horizon, conditions, draws=1000, seed=None) -> ConditionalForecastResult
rb.conditional_forecast(fit, horizon, conditions, draws=1000, seed=None, base=None)  # base: an unconditional ForecastResult to condition
```

`conditions`: a `pandas.DataFrame` indexed by horizon `1..H` (ints) or by the forecast periods, columns a subset of
the variables, `NaN` = free, values on the model's (transformed) scale; a mapping `{variable: sequence}` is accepted
(`NaN` padding). Available on `AnalyticFit`, `StructuralNIGFit` and `MCMCFit` with a linear mean (structural,
reduced, both CommonSV programs; any volatility). Errors (§7.3): unknown variable; horizon outside `1..H`; no
condition; a non-finite condition; every cell conditioned; a forecast without `lag_coefs`; a non-linear mean.

### 6.2 The exact update

For each `(chain, draw)` slice of an unconditional forecast with the same `horizon`, `draws` and `seed` (or of
`base`): `u_k = paths_k − cond_mean_k`, `ε_k = L_k⁻¹u_k` (`L_k = cond_chol_k`, triangular solve), `Φ_0 = I`,
`Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}` from `lag_coefs`, the zero-shock path `μ_h = cond_mean_h − Σ_{k<h}
Φ_{h−k}u_k` (P2 D-34), and `M` (`nH × nH`, block lower triangular, `M_{h,k} = Φ_{h−k}L_k`), so that `Y = μ + Mε`
exactly (tested). With `R` the `m × nH` selection of conditioned cells and `r` their values:

```text
QR: (RM)ᵀ = Q₁R₁ (R₁ m × m upper, positive diagonal; full row rank because every L_k has a positive diagonal)
Y_c  = Y + M Q₁ R₁⁻ᵀ (r − RY)                         one exact draw from Y | θ_s, λ_s, RY = r
E_c  = μ + M Q₁ R₁⁻ᵀ (r − Rμ)                          conditional mean
Cov_c,h = N_hᵀN_h,  N_h = (I − Q₁Q₁ᵀ) M_hᵀ             M_h: the n rows of horizon h;  L^c_h from the QR of N_h
```

Conditioned cells of `Y_c` are then set to `r` exactly and the rows and columns of `L^c_h` that belong to them to
exactly 0 (rounding). `paths_level` is the pathwise inverse transform of `Y_c` (SPEC §5.2). No random number is drawn:
the conditional forecast is a deterministic function of the unconditional draws, so the same seed reproduces it.
Under SV, outliers or t the draws condition on their drawn scale paths (inside `L_k`); the conditions update neither
the parameters nor the volatilities (D-50).

### 6.3 `ConditionalForecastResult` and its contract

A subclass of `ForecastResult` (R3: same fields, methods and attrs) whose dataset adds `condition (horizon,
variable)` (`NaN` = free) and `conditioned (horizon, variable)` (bool), `attrs["conditional"] = "hard-v1"` and the
seed of the base forecast. `cond_mean`/`cond_chol` are `E_c` and `L^c` above: the moments of `y_{T+h}` given the
draw, its scale path and the conditions (integrated over the other shocks). `log_pred_density(y, horizon,
variables=None)` uses the free variables of that horizon by default and raises `ValueError` for a conditioned
`(horizon, variable)` or for `lpd_method="integrated"` (its moments are integrated already). `quantiles`, `mean`,
`to_dataframe` are inherited. `forecast/invariants.check_conditional_structure`: conditioned paths equal `r`
exactly, conditioned rows and columns of `cond_chol` are exactly 0, the free block is lower triangular with a
positive diagonal; simulation consistency (KS of the standardised free residuals) per horizon.

---

## 7. API and construction-time validation (WP27)

### 7.1 User-facing classes (pure Python, frozen dataclasses)

```python
rb.Horseshoe(global_scale=1.0, groups="olcl-lagwise", scale="psi", form="auto",
             mean=None, psi="ar", var_const=1e7, contemp=1.0)
rb.NormalGamma(a=rb.DiscreteGrid.gruber_kastner(), b=0.5, c="a/2", **common)
rb.R2D2(a=rb.DiscreteGrid.gruber_kastner(), b=0.5, **common)
rb.DirichletLaplace(a=rb.DiscreteGrid.gruber_kastner(), **common)
rb.SSVS(c0=0.01, c1=100.0, semiautomatic=True, p=0.5, **common)      # p: float in (0, 1) or sv_priors.Beta(s1, s2)
rb.DiscreteGrid(support, weights)                                     # normalised; DiscreteGrid.gruber_kastner()
rb.AsymmetricConjugate(lambda_=0.2, cross=0.5, contemp=1.0, alpha=2.0, psi="ar", mean=None, var_const=1e7,
                       soc=None, sur=None, soc_scope="persistent", hierarchy="eb", hyper="auto",
                       hyperpriors=GLPHyperpriors(), mh=MHSettings())
rb.CommonSV(rho_prior=sv_priors.TruncatedNormal(0.9, 0.04), sigma2_prior=sv_priors.InvGamma(5.0, 0.04), block=100)
rb.conditional_forecast(fit, horizon, conditions, draws=1000, seed=None, base=None)
```

`**common` are the fields of `Horseshoe` after `global_scale`. `Linear(prior=...)` accepts `Minnesota`, any GL prior
or `AsymmetricConjugate`; `BVAR(vol=...)` accepts `Constant`, `SV` or `CommonSV`.

### 7.2 Resolution and engine table (`BVAR.__post_init__`; extends P2 §11.1)

`auto` values first (stored resolved, D-19): `Minnesota(form="auto")` → `"conjugate"` iff `vol` is `Constant` with
prior `"auto"`/`"iw"` **or `CommonSV`**, else `"structural"`; a GL prior's `form="auto"` → `"structural"`;
`Constant("auto")` → `"iw"` iff the form is conjugate or the prior is `AsymmetricConjugate`, else `"ig"`. Then the
table (first match wins):

| # | Resolved composition | Engine / program |
| :- | :--- | :--- |
| 1 | conjugate Minnesota × `Constant("iw")` | `analytic` (Phase 1, unchanged) |
| 2 | conjugate Minnesota × `CommonSV` | `mcmc`, program (b) (§5.2) |
| 3 | conjugate Minnesota × anything else | `ValueError` (E-1) |
| 4–9 | structural / reduced Minnesota × `Constant(…)`, `SV(…)` | P2 §11.1 rows 3–9, unchanged |
| 10 | structural or reduced Minnesota × `CommonSV` | `mcmc`, program (a), Mode I (§5.3) |
| 11 | GL × `Constant("ig")`, `Constant("iw_equivalent")`, `SV(…)`, `CommonSV` | `mcmc` (structural or reduced; §3) |
| 12 | GL × `Constant("iw")` | `ValueError` (E-2) |
| 13 | `AsymmetricConjugate` × `Constant("iw")` | `analytic`, `StructuralNIGFit` (§4) |
| 14 | `AsymmetricConjugate` × anything else | `ValueError` (E-3) |

MCMC-only rules after the table: `outliers="lenza-primiceri"` → E-6 (with `CommonSV`: E-7); `hierarchy="mh"` on a
Minnesota MCMC composition → E-5; the JAX-free `config.validate_composition` (§2.1). `uses_fit_seed` and
`supports_warm_start` are `engine == "mcmc"` (program (b) included). `fit.engine` reports the engine;
`fit.static.struct.kind` tells the CommonSV programs apart.

### 7.3 Error messages (each raised at construction unless marked; tested verbatim in `test_api_phase3.py`)

| ID | Condition | Type | Message |
| :- | :--- | :--- | :--- |
| E-1 | conjugate Minnesota × `SV`/`Constant("ig"|"iw_equivalent")` | `ValueError` | "the conjugate Minnesota prior requires Constant(prior='iw') or CommonSV(); use form='structural' (the MCMC default) or 'reduced'" |
| E-2 | GL × `Constant("iw")` | `ValueError` | "Constant('iw') makes the coefficient prior λ-scaled (Mode II, the conjugate twin); global-local priors are independent priors: use Constant('ig'), Constant('iw_equivalent'), SV() or CommonSV()" |
| E-3 | `AsymmetricConjugate` × not `Constant("auto"|"iw")` | `ValueError` | "AsymmetricConjugate is conjugate only with constant volatility (Constant()); for stochastic volatility use Linear(prior=Minnesota(form='structural', cross=...)) with SV(), or CommonSV() with the conjugate Minnesota prior" |
| E-4 | GL `form="conjugate"` | `ValueError` | "global-local priors are not conjugate; use form='structural' (default) or 'reduced'" |
| E-5 | `Minnesota(hierarchy="mh")` on the MCMC engine | `NotImplementedError` | "hierarchy='mh' is not available on the MCMC engine in this version (DESIGN Q15); use 'eb' or 'fixed', or the analytic engine (Minnesota(form='conjugate') with Constant())" |
| E-6 | Lenza–Primiceri on the MCMC engine | `NotImplementedError` | "Lenza-Primiceri scaling runs on the analytic engine only (DESIGN Q16); use SV(outliers=True) for outlier-robust MCMC models" (P2's double-treatment clause kept for SV(outliers=True)) |
| E-7 | Lenza–Primiceri with `CommonSV` | `ValueError` | "CommonSV already scales every variance by e^{h_t}; adding Lenza-Primiceri scaling is a double treatment" |
| E-8 | `AsymmetricConjugate` with `outliers="lenza-primiceri"` | `ValueError` | "Lenza-Primiceri scaling is not available with AsymmetricConjugate in this version; use Minnesota(form='conjugate')" |
| E-9 | GL `scale` not in {"psi", "none"}; `groups` unknown string or not a rectangular non-negative integer array | `ValueError` | lists the allowed values |
| E-10 | NG/R2D2/DL `a ≤ 0` or a grid point outside `(0, 1]`; `b ≤ 0`; NG `c ≤ 0` and not `"a/2"`; HS `global_scale ≤ 0` | `ValueError` | names the parameter and its range |
| E-11 | SSVS `c0 ≤ 0`, `c1 ≤ c0`, `p ∉ (0, 1)` | `ValueError` | "SSVS needs 0 < c0 < c1 and p in (0, 1) or a Beta prior" |
| E-12 | SSVS `semiautomatic=True` and `T_eff ≤ q` (structural) / `T_eff ≤ k + n + 1` (reduced) (**at fit**) | `ValueError` | "SSVS(semiautomatic=True) scales by the flat-prior OLS posterior s.d., which needs more observations than coefficients (T = …, needed > …); use semiautomatic=False or fewer lags" |
| E-13 | custom `groups` shape ≠ `(n, n·p)` (**at fit**) | `ValueError` | "groups array has shape …; expected (n, n·lags) = (…)" |
| E-14 | `CommonSV.block < 2`; non-`TruncatedNormal`/`Beta` `rho_prior`; non-`InvGamma`/`ScaledChi2` `sigma2_prior` | `ValueError`/`TypeError` | names the field |
| E-15 | `AsymmetricConjugate` `cross`/`contemp`/`lambda_` ≤ 0; `psi` free with `cross` free | `ValueError` | as P2's structural rule |
| E-16 | `conditional_forecast` input errors (§6.1) (**at call**) | `ValueError`/`KeyError` | names the variable or horizon and the valid range |

### 7.4 Spec, hash, names, axes

`BVAR.spec()` serialises the new priors with a `"class"` key and the GL `a` grid as `{"class": "DiscreteGrid",
"support": [...], "weights": [...]}`; `CommonSV` as `{"class": "CommonSV", …}` with nested priors through
`priors.sv.prior_to_spec`. `from_spec` dispatches on `"class"` (absent means `Minnesota` / `Constant`). Every Phase 1
and Phase 2 spec and hash is unchanged (golden test at `2c71fb2`). `BVAR.name` and `axes` follow P2 §11.4; the
prior axis of a GL model is its label; `axes["vol"] = "CommonSV"`. Program (b) carries `axes["prior"]` of the
conjugate Minnesota (`Minnesota(conjugate, …)`), so a vol-axis verdict `CommonSV` vs `Constant` between programs (b)
and the analytic engine is one-axis fair (same NIW prior), and between program (a) and `SV` likewise (Mode I).

### 7.5 Grid harmonisation (D-53)

`BVAR.harmonise_grid(combos)`: if any combination has an `SV` volatility or a GL prior, every `Minnesota(form="auto")`
becomes `"structural"` and `Constant("auto")` becomes `"ig"` (CommonSV arms then use program (a)); otherwise
`auto` forms stay conjugate (analytic `Constant` arms and program (b) `CommonSV` arms). Explicit settings are never
changed; rewrites are listed in the info note (P2 §11.4).

---

## 8. MCMC budgets and the convergence gate (D-51, D-52)

- **Budget.** `MCMCSettings.with_default_draws(sv=…)` (D-35) treats `CommonSV` as stochastic volatility: 4 × 10,000
  kept draws by default, and the D-35 warning below 10,000 per chain, its message naming CommonSV. GL compositions
  take the D-35 default of their volatility until WP31 freezes their default (Q21). The pilot measures, at
  `n = 7, p = 4, T = 200` (FRED-QD medium set) and at the BENCH-P2 shape, the bulk ESS per 1,000 sweeps of every
  primary quantity of every family × {Constant, SV} and of CommonSV's `h_T`, `ρ`, `σ_h`, and recommends the smallest
  budget at which the median fit passes the gate. The owner's decision becomes D-58. No threshold of P2 §8.2 changes.
- **Gate.** P2 §8.2 unchanged. Primary set additions (D-52): GL group log scales (when `G ≤ max(2p, n)`), SSVS `p_g`
  under a Beta hyperprior; CommonSV `h_T`, `ρ`, `σ_h` and `log λ_i` / `log Σ_ii`.
- **Warnings** (P2 §8.3 table additions): GL group scales failing R-hat or ESS (named, primary or secondary);
  hyperprior index stuck at a grid end for > 50% of draws ("the grid may be too narrow"); SSVS inclusion probability
  in `(0.05, 0.95)` for fewer than 1% of coordinates is not a warning (expected sparsity); CommonSV `h` block
  acceptance below 0.1 ("decrease CommonSV(block=…)") and any Newton cap hit (informational; the draw is exact); the
  analytic `AsymmetricConjugate` EB or MH non-convergence (as Phase 1).

---

## 9. Validation plan (R6)

Every new block is tested in isolation (Tier U, B), every new composition end to end (G, S), every curated bug is
caught (M), and every comparison with "the truth" uses a truth written by someone else (P2 §9.1, D-22). The
references (WP30) and the composition suites (WP37) are written by an author who wrote no Phase 3 kernel, from this
document only.

### 9.1 References (`validation/reference.py`, `simulate.py`, `log_joint.py`, `testfns.py`, `joint.py`)

New registered kinds: `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs` (prior samplers of the scales
and deviations in log space, log densities of §3.2 written from the prior definitions, not from the conditionals);
`kronecker` + `linear_conjugate` (NIW prior sampler from explicit `(M, Ω, Ψ, ν₀)` with its own Bartlett draw);
`common_sv` (stationary AR(1) `h`, `ρ` and `σ_h²` priors); `constant_common` (Constant cores with the common `h`).
`FixedDesign` gains `exogenous: bool` (D-56): `X ~ N(0, 1)` of shape `(T_eff, k)` drawn once from the configuration
seed and held fixed, `Y` simulated given `θ` without lag feedback; the engine sees an ordinary `EngineData`. The OCSN
tables are unaffected. `simulate_obs` gains the Kronecker (`y_t = Π'x_t + e^{h_t/2} C ε_t`) and common-`h` paths.

### 9.2 Tier U (every PR; WP28, WP32–WP35)

1. **Numerics**: `log_gig` (§3.5 list); `precision_draw_log` equal to D-14's `precision_draw` on the same `z` (1e-12
   relative at precision disparities up to 1e8) and to a dense high-precision oracle at disparities up to 1e300
   (mpmath); `dev_log` equal to `log|θ − mean|` wherever that is representable; the JAX NIW port equal to
   `ConjugateProblem` (1e-10); the tridiagonal Newton (gradient below 1e-8, state independence of the proposal).
2. **Density-ratio identities** (P2 §9.3 item 2) at 20 random states for every line of §3.2 (local, group,
   collapsed `a`, SSVS `γ`, `p`), the deviation draw, the NIW step, each `h` block, the `ρ` and `σ_h²` steps, against
   the WP30 `log_joint`; MH steps: the computed log acceptance ratio equals target and proposal densities.
3. **Closed forms**: collapsed `a` weights against numerical integration over `ξ`; DL `θ̃ | d` against quadrature of
   the joint; SSVS log odds against brute force; `q_t e^{−h_t} | h ~ χ²_n` at the data-generating `θ` (simulator check,
   N08 §A11); conditional-forecast update against the exact oracle (A10).
4. **Infrastructure**: the Phase 2 golden state templates and spec hashes; every Phase 2 composition bitwise
   unchanged on a short run before/after WP26–WP33 (the `coef_log = False` path); float64 everywhere; plan
   invariants (P2 §4.8) with the write-set tree diff for every new composition; zero recompiles across windows of
   one bucket.

### 9.3 Tier B: block configurations (`BLOCK_CONFIGS`; every PR, in the kernel suites)

| Config | Live | Held | Exact |
| :--- | :--- | :--- | :--- |
| `gl_scales_only_<family>` (×5) | local, group and hyper parameters | coefficients (prior draws), data irrelevant | yes |
| `gl_mean_only` | coefficients (deviation path) | GL scales (prior draws), `λ` known | yes |
| `common_sv_only` | `h, ρ, σ_h` | `Π, Σ` known; `q` from the simulator | yes |
| `conjugate_niw_only` | `(Π, Σ)` | `h` known | yes |

### 9.4 Tier G: Geweke rows (P2 §9.5 machinery and sizes: G-small `C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3`;
G-full `C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`; Gandy–Scott reruns)

| Row | Composition | Design | n | Tier |
| :- | :--- | :--- | :- | :--- |
| 301 | structural Mode I × `Constant("ig")` × HS (olcl-lagwise) | exogenous | 2 / 3 | PR + weekly |
| 302 | structural × `SV` × HS | exogenous | 2 / 3 | weekly |
| 303, 304 | structural × `Constant("ig")` × NG (fixed `a`, fixed `c`; grid `a`, `c="a/2"`) | exogenous | 3 | weekly |
| 305, 306 | structural × `Constant("ig")` × R2D2 (fixed; grid) | exogenous | 3 | weekly |
| 307, 308 | structural × `Constant("ig")` × DL (fixed; grid) | exogenous | 3 | weekly |
| 309, 310 | structural × `Constant("ig")` × SSVS (fixed `p`, explicit `τ`; `p ~ Beta`) | exogenous | 3 | weekly |
| 311 | reduced × `SV` × HS (global) | exogenous | 3 | PR + weekly |
| 312–315 | reduced × `SV` × {NG, R2D2, DL, SSVS} (global, fixed `a`/`p`): the parity compositions | exogenous | 3 | weekly |
| 316 | reduced × `SV` × HS (olcl-lagwise), tight scales | lagged | 3 | weekly |
| 317 | structural × `SV(outliers=True)` × HS (equation-wise) | exogenous | 3 | weekly |
| 318 | structural × `Constant("iw_equivalent")` × HS | exogenous | 3 | weekly |
| 320 | conjugate Minnesota × `CommonSV` (program b) | exogenous (`k = 3`, N08 §A11) | 3 | PR + weekly |
| 321 | structural Mode I × `CommonSV` (program a) | lagged | 2 / 3 | PR + weekly |
| 322 | reduced × `CommonSV` | lagged | 3 | weekly |
| 323 | structural Mode II × `CommonSV` (engine level; the A8 twin of 320) | lagged | 3 | weekly |
| 324 | structural × `CommonSV` × HS | exogenous | 3 | weekly |
| 325 | conjugate × `CommonSV` with `ScaledChi2` σ_h² and `Beta` ρ (parity options) | exogenous | 3 | weekly |
| blocks | the eight configurations of §9.3 | — | — | PR |

**Test priors** (proper and tight, P2 §9.1): HS `global_scale = 0.1`; NG `a = 0.5, b = 2, c = 1`; R2D2 `a = 0.5,
b = 2`; DL `a = 0.5`; grids of five points `{0.1, 0.25, 0.5, 0.75, 1}` with equal weights; SSVS `τ₀ = 0.01·√c`,
`τ₁ = 0.5·√c`, `p = 0.3` or `Beta(2, 3)`; `ψ` fixed and unequal (so `c_j ≠ 1`, mutant M27); own-lag prior mean 0.4
(mutant M26); everything else P2 §9.2. CommonSV: `ρ ~ TN(0.8, 0.15²)`, `σ_h² ~ IG(5, 0.2)`; NIW `Ψ = 0.5 I`,
`ν₀ = n + 4`, own `0.1²`, cross `0.05²`, intercept 1. `T_eff = 20 / 50` (never a multiple of 32).
**Test functions**: bounded transforms of deviations (`arctan(d_j/√c_j)`, `1{|d_j| < 0.1√c_j}`) at fixed
coordinates, group log scales, two local log scales, `a`-grid indices, SSVS `N1_g` and `log p_g`, `h` at three dates
and `mean h`, `ρ`, `σ_h`, `log|Σ|`, `Σ₂₁`, the exact log-likelihood, `W = Σ_t q_t e^{−h_t}`, log-scale data functions.

### 9.5 Tier S: SBC (weekly; P2 §9.6 configuration: `N = 1000, L = 99`, ESS rules, γ test, Bonferroni `α = 1e-2`)

Every row of §9.4 and every block configuration. Quantities: the test functions above, the total and per-equation
log-likelihood, `log p(θ)`, `W` (N08 §A11), and the predictive `y_{T+1}` through the production forecast core.

### 9.6 Tier M: mutants (weekly; each must fail some tier; injected only through the P2 §4.9 seam)

| Mutant | Injected through | Expected catcher |
| :--- | :--- | :--- |
| M19 HS `ζ_g` shape `n_g/2` (the `+1` dropped) | `gl_group_conditional` | U (density ratio), G |
| M20 HS `ϑ_j` scale with `τ_g` where `ζ_g` belongs (GK App. B as printed) | `gl_local_conditional` | U, G |
| M21 NG `ϖ_j` GIG with `λ = a − 1` | `gl_local_conditional` | U, G |
| M22 NG collapsed `a`-weight without `b log(2c/ã)` (row 303's grid variant with fixed `c`) | `gl_a_conditional` | U, G |
| M23 R2D2 `ψ_j` GIG with `χ = d²/(cϖ)` (factor 2 dropped) | `gl_local_conditional` | U, G |
| M24 DL `θ̃_j` GIG with `λ = a − ½` | `gl_local_conditional` | U, G |
| M25 SSVS log odds without `log τ₀ − log τ₁` | `ssvs_logodds` | U, G |
| M26 deviation about 0 instead of the prior mean | `gl_deviation` | G (own-lag mean 0.4) |
| M27 scale step with `c_j = 1` while the mean uses `c_j` | `gl_local_conditional` | U, G |
| M28 CommonSV Newton started at the current `h`, one iteration (proposal depends on the state) | `csv_newton_start` | G, A11 (quadrature) |
| M29 `q_t` with `Σ` instead of `Σ⁻¹` | `csv_q` | U, G |
| M30 `ρ` acceptance without the stationary-start factor | `csv_rho_accept` | U, G |
| M31 CCM (2016) eq. (23) `Σ` draw (N08 §A10.1) | `niw_draw` | G, A11 |
| M32 prior dummy rows reweighted by `e^{−h_t/2}` | `niw_posterior` | U (A11 at fixed `h`), G |
| M33 conditional update with `Mᵀ` where `M` belongs | `conditional_update` | U (A10) |
| M34 GIG tails from secants instead of tangents (an envelope below `f`) | `gig_sampler` | U (KS, `E log X`) |

`gig_sampler` and the names of §2.4 join `protocols.INTERNALS`. The suite records the catching tier of each mutant
(`mutation.jsonl`); a mutant caught by none blocks the release.

### 9.7 Tier A: analytic twins and exact oracles

| Test | Oracle | Cadence |
| :- | :--- | :--- |
| A8 | program (b) vs structural Mode II × CommonSV (program a) at `cross = contemp = 1`: quantiles of `B`, `Σ`, `h_t`, `ρ`, `σ_h` within 4 quantile-MCSE (Holm); N08 §A11 item 2 | PR (`n = 3, p = 2, T = 60`); weekly (`n = 20, p = 4, T = 300`) |
| A9 | `StructuralNIGFit` at `cross = contemp = 1` equals `AnalyticFit` (log ML 1e-10, posterior moments of `B`, `Σ` 1e-10, EB `λ` 1e-6); EB with free `cross` at a stationary point; MH quantiles within MCSE of the closed-form posterior over a fixed hyperparameter grid; at `cross ≠ 1`, `StructuralNIGFit` forecasts vs MCMC Mode II within MCSE | PR; the MCMC comparison weekly |
| A10 | conditional forecasts: per draw against an independent stacked companion-form oracle with Schur-complement conditioning (SciPy), `E_c`, `Cov_c`, `Y_c` from the same `ε` to 1e-10, on analytic (constant `Σ`), MCMC SV and CommonSV draws; sample moments of conditional paths vs `E_c`, `Cov_c` | PR |
| A11 | CommonSV closed forms (N08 §A11 items 1, 3, 4): `(Π, Σ) | h` vs Phase 1 with `row_scales = e^{h/2}`; `log p(Y | h)` = Phase 1 log ML − `(n/2)Σh` (1e-10); `h` posterior for `T ∈ {2, 3}`, `n = 5` by tensor quadrature vs MH draws (KS); proposal state independence | PR |
| A12 | GL deviation path with frozen finite scales equals the Phase 2 path on the same noise (1e-12), structural and CCCM | PR |
| A13 | SSVS with `τ₀ = τ₁` equals the Normal prior (posterior moments within MCSE of `FixedDiagonal`) | PR |

### 9.8 Tier P: parity (fixtures by WP29; tests by WP39)

**`bayesianVARs` 0.1.8** (reduced form, through `fit_mcmc` with explicit arrays, P2 §9.9). Probes first, each a
small fixture whose result is recorded in `tests/parity/SV_PARAMETRIZATION.md` before any parity comparison: P1–P5
(P2 §9.9: `normal_sds` and `cholesky_normal_sds` variance vs s.d.; `U` vs `U⁻¹` storage and sign; intercept row;
`sv_para` layout; `cholesky_priorhomoscedastic` scale); **P6** internal standardisation (refit with one series ×100:
equivariance of `PHI`); **P7** the grouping matrices built for `"global"`, `"equation-wise"`, `"covariate-wise"`,
`"olcl-lagwise"` (printed from the prior object); **P8** the GL hyperparameters' meaning (prior-dominated runs, `T`
small and noise large, compared with WP30's prior sampler at the mapped parameters, KS on `log|φ|`; the value of
`DL_a = "1/K"`); **P9** the SSVS semi-automatic `τ` (printed); **P10** the intercept is outside the shrinkage and
`prior_intercept` is an s.d. Parity runs (one parameter set per run, both sides derived from it, E11's lesson):

| Run | Composition (`groups`, scales) | Data |
| :--- | :--- | :--- |
| R1 | reduced × SV, Normal coefficient prior: the Phase 2 criterion (P2 §9.9 parameter set) | FRED-MD `n ∈ {3, 7}`, `p ∈ {2, 4}` |
| R2 | reduced × `Constant("ig")`, Normal prior | `n = 3, p = 2` |
| R3–R7 | reduced × SV × {HS, NG, R2D2, DL, SSVS} (`"global"`, fixed `a`, fixed `p`, `semiautomatic=False`, `scale="none"`) | `n ∈ {3, 7}`, `p = 2` |
| R8 | reduced × SV × HS (`"olcl-lagwise"`) | `n = 7, p = 4` |
| R9 | reduced × SV × NG with a discrete grid on `a` | `n = 7, p = 2` |

Common settings: `cholesky_U_prior = "normal"` with `cholesky_normal_sds` from P1, the SV priors of P2 §6.3,
`expert_sv_offset = 0` ↔ `SV(offset=0.0, exact=False)`, `prior_intercept = s_c` ↔ `var_const = s_c²` (P10). Compare
5/25/50/75/95% quantiles of `Π`, `A`, `h_t` (`t ∈ {1, T/4, T/2, T}`), `μ, φ, σ` and the one-step predictive with the
Phase 2 rule (4 quantile-MCSE, Bonferroni-type bound of `tests/parity`). The GL scales themselves are compared only
through the coefficients they imply (their parametrisations may legitimately differ by P8).

**`bvars` 1.1** (program (b)). Probes **P11** the defaults of `specify_prior_bvar` (printed: `A`, `V`, `S`, `ν`, the
matrix-GIG `Ψ, Γ, λ`, `sv_a`, `sv_s`, `sigma2_omega`, `s_`); **P12** whether `V` can be held fixed (or the matrix-GIG
prior concentrated to a relative s.d. ≤ 1e-3 at our `Ω`: prior draws of `V`); **P13** the common-volatility
normalisation (`E h = 0`, initial condition) under `"cSV"` and `"ncSV"`; **P14** the `ρ` and `ω`/`σ_ω²` priors and
whether they can be fixed; **P15** the `Σ` convention (covariance vs scale), the prior mean (`stationary`) and the
intercept. Decision, recorded with the probe results before any comparison is run:

1. **Posterior parity** if `V` can be held fixed (P12) and the SV prior maps onto a `CommonSV` option (P13–P14, or a
   bvars setting that matches ours): quantiles of `Π`, `Σ`, `h_t`, `ρ`, `σ_h` within 4 MCSE, FRED-QD small and medium
   sets, `p = 4`.
2. **Otherwise forecast-distribution parity** (SPEC §10.4 last bullet): 1- and 4-step predictive quantiles on the same
   data, with the tolerance fixed in WP29's PR before the comparison runs, plus the exact internal oracles A8 and A11
   as the substitute for posterior parity, documented as Phase 2 did (P2 §2.10 Stage 4 amendment 2).

**BoE `bvar` 0.3.5** (Python, CI): conditional forecasts of the conjugate model at fixed hyperparameters (the Phase 1
mapping of `tests/parity`), conditions on one and two variables over 4–8 quarters, quantiles of the conditional
paths within 4 MCSE (iid draws).

### 9.9 DGP verdict tests (SPEC §10.5; exit criterion)

Built by the parallel DGP session: `validation/dgp.py`, `benchmarks/dgp_study.py`, `tests/dgp/` and the
pre-registration `benchmarks/reports/dgp_study_preregistration.md` (rows `linear-homoskedastic`, `linear-sv`,
`linear-sv-outliers`; size and power of `verdict(axis="vol")`, 200 replications per row, readings fixed in advance).
Phase 3 neither duplicates nor changes it. WP40 verifies its merged PR, receives (or runs, if asked) the registered
full study, and records its readings in SPEC §13 status. New Phase 3 rows (a CommonSV arm, GL prior arms) would be
rows of their own, pre-registered separately, and are not part of the exit.

### 9.10 Replication target (SPEC §10.7): Gruber & Kastner (2025), qualitative (WP41)

A pre-registration (`benchmarks/reports/gruber_kastner_preregistration.md`) is committed before any score exists. It
fixes: FRED-QD 2026-09, GK's 21 series mapped to FRED-QD mnemonics (substitutions for series not in FRED-QD listed),
GK's transformations, `p = 4`, the reduced form (GK's sampler), `SV()` defaults, expanding windows, origins and
horizons (1, 4), the arms (Minnesota EB, HS, NG, R2D2, DL, SSVS; semi-global, GK hyperpriors), the budget (D-51),
`lpd_method` per horizon, and four qualitative criteria: (1) at `h = 1` the best semi-global GL prior beats
Minnesota-EB in total joint log score; (2) at `h = 4` at least three of the four continuous GL priors beat
Minnesota-EB; (3) the leading prior changes between the pre-2008 and the 2009–2015 subperiods; (4) SSVS has the
lowest `h = 1` joint score among the GL priors. A criterion not met is reported, never tuned (as the Clark–Ravazzolo
acceptance did).

### 9.11 CI tiers

PR tier (fast suite): Tier U and B of §9.2–§9.3; G-small rows 301, 311, 320, 321; A8 (`n = 3`), A9, A10–A13;
contract tests for three new MCMC compositions (structural × SV × HS, conjugate × CommonSV, structural × CommonSV)
plus `StructuralNIGFit` and conditional results; fixture reads and short parity runs; BoE conditional parity.
Weekly tier: G-full and S for rows 301–325 and the blocks, M19–M34, A8 (`n = 20`), A9's MCMC leg, long parity runs.
`tools/weekly_shards.py` gains shards `geweke-p3` and `sbc-p3` (by file: `tests/geweke/test_geweke_p3.py`,
`tests/sbc/test_sbc_p3.py`, split further if a shard exceeds 200 min on a 2-vCPU EPYC 7763; the partition check
covers them). Weekly runs that a work package needs are dispatched by hand on its branch (`workflow_dispatch`).

---

## 10. Performance

Estimates to be replaced by WP31's measurements (`docs/phase3/pilot.md`):

- GL scale step: elementwise work on `n²p` coordinates (≤ 1.5 GIG attempts each), `G` segment sums and an `A × G`
  grid evaluation: ≈ 0.1–0.3 ms per sweep of 4 chains at `n = 20, p = 4`, against BENCH-P2's 3.90 ms. The deviation
  path costs D-14's flops plus elementwise logs. GL × Constant loses the eigen path (0.35 → ≈ 3 ms per sweep).
- CommonSV program (b): one QR of `(T + m_d) × (k + n)` per chain and sweep (`n = 20, p = 4`: 381 × 101, ≈ 8·10⁶
  flops) plus the `h` blocks: ≈ 0.5 ms per sweep of 4 chains, 8× below SV. `n = 100, p = 4, T = 300`: ≈ 3.5·10⁸
  flops per chain and sweep, ≈ 7–15 min for 4 × 11,000 sweeps on one device (the D2 screen; structural SV is
  infeasible there). Program (a) costs what structural SV costs.
- **BENCH-P3** (reported, not gated; SPEC sets no Phase 3 target): program (b) at `n = 100, p = 4, T = 300`,
  4 × 10,000 kept draws, and HS × SV at the BENCH-P2 shape (overhead against BENCH-P2), in `benchmarks/phase3_perf.py`.

---

## 11. Risks

| # | Risk | Mitigation |
| :- | :--- | :--- |
| R18 | GL mixing (HS global scale; small `a` on the grid) leaves fits unconverged | WP31 pilot; group scales primary (D-52); budget decision D-58; warnings |
| R19 | GIG sampler wrong at extreme parameters | log-space algorithm with an envelope valid by concavity; extreme-grid Tier U; M34 |
| R20 | `bayesianVARs` still cannot be built, or a probe stays ambiguous | D-54 Linux workflow (gfortran); an unresolved probe is recorded with its evidence and the comparison it blocks is replaced by the exact oracles, as in Phase 2 |
| R21 | `bvars`'s hierarchical `V` prevents posterior parity | probe-driven decision (§9.8) with a pre-fixed fallback |
| R22 | CommonSV `h` acceptance falls with `T` and `σ_h²` (N08 §A5.2 table) | blocks of 100 dates; acceptance warning; pilot at `T = 600` |
| R23 | Conditional forecasts near rank deficiency | QR with a positive-diagonal check; an actionable error, never jitter |
| R24 | CI time grows with 25 rows × (G, S) | new weekly shards; four PR rows; measured on the first dispatch |
| R25 | Contract additions change Phase 2 draws | `Empty` defaults; golden templates and hashes; bitwise before/after tests (WP26 acceptance) |
| R26 | Parallel sessions edit shared files | ownership of §1.2; rebase on their merges; `ci.yml` only in WP42 |
| R27 | GK series missing from FRED-QD or third-party | substitutions fixed in the pre-registration |
| R28 | GitHub Actions minutes for R builds | dispatch-only workflow with a cached R library |

---

## 12. Open questions for the owner

| ID | Question | Recommendation |
| :- | :--- | :--- |
| Q15 | `hierarchy="mh"` on the MCMC engine (SPEC §4.5, P2 D-12) | Defer past v0.3.0: an exact global step over `(λ, cross)` with GLP hyperpriors fits the D-40 machinery but is not in the SPEC §13 list; change the message (E-5) and SPEC §4.5 |
| Q16 | Estimated Lenza–Primiceri scaling on the MCMC engine (SPEC §5.6, P2 D-21) | Defer; `CommonSV` covers a common variance spike now, a common outlier component later (N08 §A9); message E-6, SPEC §5.6 edit |
| Q17 | GL shrinkage of the contemporaneous coefficients (SPEC §4.4) | Defer: it needs the structure's `contemp_moments` to read coefficient-prior state (D-28); parity uses `cholesky_U_prior = "normal"` |
| Q18 | Continuous ν prior (P2 Q8) | Defer |
| Q19 | `solver="bcm"` for `q > T` | Defer (the Cholesky path is exact, only slower) |
| Q20 | `HMP` (bayesianVARs' hierarchical Minnesota) | Not in v0.3.0; it is a GL-family member (group-scaled Normal with GIG steps) if wanted later |
| Q21 | Default MCMC budget of GL compositions | Decide from WP31 (D-58); gate unchanged |
| Q22 | Permission to download the R fixtures (the `r-fixtures` workflow artifact, or a `rocker/r-ver` image) | Allow the workflow artifact (smallest, reproducible from the workflow file) |
| Q23 | GL defaults: GK's semi-global with hyperpriors, or `bayesianVARs`' global with fixed `a` | GK's (§3.2): best in their simulations and application; parity sets `bayesianVARs`' values explicitly |
| Q24 | CommonSV extensions: Student-t (Chan 2020), random walk, outliers | Defer (N08 §A5.4, §A9) |
| Q25 | Conditional forecasts: soft conditions, parameter-updating (Waggoner–Zha Gibbs) | Defer |
| Q26 | Compute for the GK acceptance run (≈ 10–20 CPU hours on the reference laptop) | Run locally after the pre-registration, with a documented quick config first |

---

## 13. Work packages (dependency-ordered)

**Rules** (P2 §13 rules 1–5 apply). Each WP owns its files (§1.2) and tests (§1.3); the contracts of WP26/WP27 freeze
when their PR is open and change only through an amendment recorded here; kernel WPs pass Tier U and B before
their PR; WP30 and WP37 are written by an author who wrote no Phase 3 kernel; every WP keeps the Phase 1–2 suites
green; clean room throughout. Branches and PRs per D-57.

**Stages.** Stage 0 (after this document): WP26, WP27, WP28, WP29 (parallel; WP29 needs Q22). Stage 1: WP30 (after
WP26), WP32 and WP33 (after WP26, WP28), WP34 (after WP27). Stage 2: WP35 (after WP34), WP31 (after WP32, WP33).
Stage 3: WP37, WP38, WP39 (after WP29), WP40 (after the DGP PR), WP41. Stage 4: WP42. Critical path: **WP26 → WP28 →
WP32 → WP37 → WP42**, with WP29 → WP39 in parallel.

| WP | Content | Files (§1.2) | Acceptance |
| :- | :--- | :--- | :--- |
| WP26 | Contracts: static config, state, prior arrays, keys, registry, internals, `GaussianLogDiag` (§2) | `engines/mcmc/{config,state,keys,registry,protocols}.py`, `tests/mcmc/unit/*` | rules of §2.1 tested; Phase 2 templates, hashes and short runs bitwise unchanged |
| WP27 | Specs and API: GL classes, `DiscreteGrid`, `AsymmetricConjugate`, `CommonSV`, `TruncatedNormal`; resolution, engine table, errors E-1…E-16, spec/hash, labels, harmonise (§7) | `priors/{global_local,asymmetric,sv}.py`, `components/specs.py`, `api.py`, `__init__.py`, `tests/unit/*phase3*` | every row of §7.2 and message of §7.3 tested; Phase 1–2 golden hashes unchanged; `import rai_bvar` imports no JAX |
| WP28 | Numerics: `log_gig`, `precision_draw_log`, tridiagonal Newton, helpers (§3.4, §3.5, §5.4) | `engines/mcmc/{linalg,random}.py`, `tests/mcmc/unit/{test_gig,test_linalg_log,test_newton}.py` | §9.2 item 1 green |
| WP29 | Oracle environment and fixtures: `r-fixtures.yml`, `make_r_fixtures.R` sections, probes P6–P15 (and P1–P5), parity runs R1–R9 and bvars, the SV-t fixture if `stochvol` ≥ 3.2.5 | `tools/`, `.github/workflows/r-fixtures.yml`, `tests/parity/fixtures/*`, `SV_PARAMETRIZATION.md` | fixtures committed with versions; every probe answered or recorded unresolved with evidence; the bvars decision (§9.8) recorded |
| WP30 | References (§9.1), exogenous design, new `COMPOSITIONS`/`BLOCK_CONFIGS` | `validation/{reference,simulate,log_joint,testfns,joint}.py`, `tests/validation/test_reference_phase3.py` | prior moments vs closed forms; simulators vs NumPy loops (1e-12); `log_joint` vs SciPy (1e-10) |
| WP31 | Pilot: GL and CommonSV mixing and cost; budget recommendation (§8, §10) | `benchmarks/pilot/phase3_*.py`, `docs/phase3/pilot.md` | measurements with machine fingerprint; recommendation for D-58 |
| WP32 | GL kernels, `GLScaleStep`, deviation paths, host builders, resolve, monitors (§3) | §1.2 `[WP32]` | Tier U, B green; row 301/311 G-small green; A12, A13 green |
| WP33 | CommonSV: both programs, the `h` kernel, Kronecker structure, NIW port, forecasts, warm starts, monitors (§5) | §1.2 `[WP33]` | Tier U, B green; rows 320/321 G-small green; A8 (`n = 3`), A11 green |
| WP34 | `StructuralNIGFit`, EB with free `cross`, MH, `lag_coefs` on analytic forecasts (§4, D-49) | `engines/{structural_nig,analytic}.py`, `tests/analytic/test_asymmetric.py` | A9 green; Phase 1 tests unchanged |
| WP35 | Conditional forecasts (§6) | `forecast/{conditional,invariants,result}.py`, fit methods, `tests/forecast/test_conditional.py` | A10 green on every engine; contract checks |
| WP37 | Composition suites: rows 301–325, blocks, SBC, mutants M19–M34 (§9.4–§9.6) | `tests/{geweke,sbc,mutation}/*p3*` | G-small (PR rows), G-full and S (weekly dispatch) green; every mutant caught |
| WP38 | Twins at scale: A8 (`n = 20`), A9 (MCMC leg) | `tests/analytic/test_*_p3.py` | green weekly |
| WP39 | Parity tests (§9.8) | `tests/parity/{test_bayesianvars,test_bvars_commonsv,test_boe_conditional}.py` | exit criteria (map below) |
| WP40 | DGP exit: the DGP session's PR and study readings (§9.9) | SPEC §13 status only | readings recorded |
| WP41 | GK replication: pre-registration, then the acceptance run and report (§9.10) | `benchmarks/gruber_kastner.py`, `benchmarks/reports/gruber_kastner_*` | report against the four criteria (met or documented) |
| WP42 | Release: SPEC v2.4 status and the §0.2 SPEC edits, README roadmap, this document's exit map, `ci.yml` and shards, `pyproject` 0.3.0, release notes; tag only on the owner's approval | `SPEC.md`, `README.md`, `docs/phase3/DESIGN.md`, `.github/workflows/ci.yml`, `tools/weekly_shards.py`, `pyproject.toml` | CI green; weekly tier green on the release commit; status report before the tag |

**PR plan** (D-57): PR 1 this document; PR 2 WP26–WP28; PR 3 WP29; PR 4 WP30 + WP32; PR 5 WP33; PR 6 WP34 + WP35;
PR 7 WP37 + WP38; PR 8 WP39; PR 9 WP31 (+ D-58); PR 10 WP41; PR 11 WP40 + WP42.

### Exit-criteria map (SPEC §13 Phase 3)

| Criterion | Delivered by |
| :--- | :--- |
| `bayesianVARs` parity, global-local priors × SV | WP39 runs R3–R9 (fixtures WP29, probes P6–P10) |
| `bayesianVARs` parity, reduced-form sampler, after P1–P5 | WP39 runs R1–R2 (fixtures WP29) |
| `bvars` parity for `CommonSV` | WP39 per the §9.8 decision (probes P11–P15), with A8 and A11 |
| DGP verdict-engine tests on linear DGPs | WP40 (the DGP session's harness and pre-registered study) |
| R6: Geweke and SBC for every new composition | WP37 (references WP30) |
| Replication target (SPEC §10.7) | WP41 |

---

## 14. References (new in Phase 3; Phase 2's are in SPEC §15)

- Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario analysis with vector
  autoregressions for large cross-sections. *International Journal of Forecasting*, 31(3), 739–756.
- Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–Laplace priors for optimal shrinkage.
  *Journal of the American Statistical Association*, 110(512), 1479–1490.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*,
  97(2), 465–480.
- Chan, J. C. C. (2020). Large Bayesian VARs: A flexible Kronecker error covariance structure. *Journal of Business &
  Economic Statistics*, 38(1), 68–79.
- Devroye, L. (2014). Random variate generation for the generalized inverse Gaussian distribution. *Statistics and
  Computing*, 24(2), 239–246.
- George, E. I., & McCulloch, R. E. (1993). Variable selection via Gibbs sampling. *Journal of the American
  Statistical Association*, 88(423), 881–889.
- George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search for VAR model restrictions. *Journal of
  Econometrics*, 142(1), 553–580.
- Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma prior distributions in regression problems.
  *Bayesian Analysis*, 5(1), 171–188.
- Gruber, L., & Kastner, G. (2025). Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!
  *International Journal of Forecasting*, 41(4), 1589–1619 (arXiv:2206.04902, App. B).
- Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the horseshoe estimator. *IEEE Signal Processing
  Letters*, 23(1), 179–182.
- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate models. *Review of Economics and
  Statistics*, 81(4), 639–651.
- Zhang, Y. D., Naughton, B. P., Bondell, H. D., & Reich, B. J. (2022). Bayesian regression using a prior on the model
  fit: The R2-D2 shrinkage prior. *Journal of the American Statistical Association*, 117(538), 862–874.
