# Phase 3 design: shrinkage priors, CommonSV, the asymmetric conjugate prior and conditional forecasts

**Status.** Reviewed design, written 2026-10-25 on branch `phase-3-design` from `main` at `2c71fb2` (release v0.2.0
= Phase 2, tag at `4ac7969`, plus PR #11 and PR #12). It covers SPEC §13 Phase 3 and is the document the Phase 3
work packages are implemented from. `docs/phase2/DESIGN.md` (cited as **P2**) stays authoritative for everything
this document does not change; where this document changes a Phase 2 contract it says so and names the P2 section
it amends. Owner decisions still open are the questions of §12; every decision that depends on one says so.

**How it was produced.** One author wrote a first draft (`783190d`) after reading SPEC v2.3, P2 in full, the Phase 2
research notes (N01–N09; N08 §A already specifies `CommonSV`), the code at `2c71fb2`, the manuals of `bayesianVARs`
0.1.8 and `bvars` 1.1, the MIT source of the Bank of England `bvar` 0.3.5, and the papers of §14. Three reviewers
then read the draft cold, without the author's notes, under different briefs: **Ra** sampler mathematics and
numerics (with numerical checks of four claims), **Rb** engineering against the code at `2c71fb2`, **Rc** scope,
oracles, exit criteria and process. They raised 97 findings (9 blockers). Every finding and its resolution is in
§0.4 (errata E29–E86); the draft was revised before any Phase 3 code was written. The largest revisions: the
conjugate `CommonSV` became a shared-factor path of the per-equation sampler instead of a separate Kronecker program
(E29), the GIG sampler follows Devroye (2014) §6 (E31), every parity and DGP criterion is pass/fail with its
tolerance stated here (E34–E36), and conditional forecasts are labelled scenarios (E43).

**Reading guide.** §0 scope, inventory, decisions, errata · §1 module layout · §2 contracts · §3 global-local and
selection priors · §4 the asymmetric conjugate prior · §5 `CommonSV` · §6 conditional forecasts · §7 API and
construction-time validation · §8 MCMC budgets and the convergence gate · §9 validation plan · §10 performance ·
§11 risks · §12 open questions · §13 work packages and the exit-criteria map · §14 references.

**Conventions.** As P2: code indices are 0-based (equation `i ∈ {0..n−1}`, `ι = i + 1`), `k = n_d + n·p`,
`q = k + n − 1`, `T_eff` real rows, `T_pad` compiled rows, `m_t = 1{t < T_eff}`; `IG(α, β)` has shape `α` and scale
`β`; `G(α, β)` is a gamma with shape `α` and **rate** `β`; `GIG(λ, ψ, χ)` has density `∝ x^{λ−1} exp(−(ψx + χ/x)/2)`
on `x > 0`; `Exp(r)` has rate `r`. Inside §3.5 `(λ, ψ, χ, ω)` are the GIG's own parameters, not the Minnesota `λ`
and `ψ`. `N08 §A5` means `docs/phase2/08_common_sv_and_ordering.md` §A5. Numbering continues Phase 2 so
cross-references stay unique (D-55): decisions **D-36…**, open questions **Q15…**, review errata **E29…**, work
packages **WP26…**, Geweke/SBC composition rows **301…**, block configurations **401…**, mutants **M19…**, analytic
oracles **A8…**, `bayesianVARs` convention probes **P6–P10** (P1–P5 are P2 §9.9's), `bvars` probes **P11–P16**.
**The Phase 2 parity rule** (P2 §9.9, `tests/parity`): each of `m` compared quantiles passes when
`|z| ≤ max(4, z_{1−0.01/(2m)})`, with `z` the difference over the combined quantile-MCSE of both sides.

**Clean room** (SPEC §2.1). Apache-2.0. Every algorithm below is specified from papers (§14) and re-derived here
where marked *derived here*; each such derivation has a Tier U density-ratio or closed-form test before any kernel
relies on it. `bayesianVARs`, `bvars`, `stochvol` and `BVAR` are GPL: they are run only as black-box oracles by
`tools/make_r_fixtures.R`, outside every test workflow, and only their manuals were read. Running a package and
recording the **data** it returns (a default value, a grouping matrix, posterior draws) is allowed; printing an R
function, closure, R6 method or `body()` is not, because that shows GPL source. The Bank of England `bvar` is MIT
and may be read and used as a Python-side oracle in CI. No code is translated from GPL sources.

---

## 0. Scope and decisions

### 0.1 What Phase 3 delivers

From SPEC §13 Phase 3:

1. Global-local and selection priors on the lag coefficients of `Linear`: `Horseshoe`, `NormalGamma`, `R2D2`,
   `DirichletLaplace`, `SSVS`, with semi-global coefficient groups (§3).
2. The user-facing asymmetric conjugate prior of Chan (2022), `AsymmetricConjugate`, on the analytic engine (§4).
3. `CommonSV`, Carriero, Clark & Marcellino (2016): with the conjugate Minnesota prior a conjugate-speed SV screen
   for large `n` (D2); with any other prior a fair volatility-axis alternative to `SV` (§5).
4. Conditional forecasts with hard conditions (SPEC §6.2; §6).

Exit criteria (SPEC §13), each pass/fail with the rule stated in §9 before any comparison runs (D-59):
`bayesianVARs` parity for the global-local priors × SV and for the reduced-form sampler, after the convention probes
P1–P5 (and the Phase 3 probes P6–P10) are settled; `bvars` parity for `CommonSV`; DGP verdict-engine tests on linear
DGPs. The exit-criteria map is at the end of §13.

### 0.2 Inventory of every Phase 3 commitment in SPEC and P2

Every sentence in SPEC v2.3, P2, the Phase 2 notes and the code that assigns work to Phase 3, and what v0.3.0 does
with it. A deferral is a **recommendation pending its question** (§12); "SPEC edit" names the sentence WP41 changes
if the owner accepts it.

| # | Commitment | Where | v0.3.0 |
| :- | :--- | :--- | :--- |
| 1 | HS, NG, DL, R2D2, SSVS | SPEC §4.4, §13 | **In** (§3) |
| 2 | Coefficient groups: own lags, cross lags | SPEC §4.4 | **In**: groups over lag coefficients (D-38) |
| 3 | Group "deterministics" | SPEC §4.4 | **Not shrunk**: fixed Normal prior (D-36), pending Q30; SPEC edit §4.4 Priors table |
| 4 | Group "contemporaneous" | SPEC §4.4 | **Deferred**, pending Q17: `a_i` keeps the Phase 2 prior; SPEC edit §4.4 |
| 5 | `AsymmetricConjugate` prior and engine rule | SPEC §4.4, §4.5 last table row, P2 Q13 | **In** (§4) |
| 6 | `CommonSV` | SPEC §4.4, D2, N08 §A | **In** (§5) |
| 7 | Conditional forecasts | SPEC §6.2, §12 tree `forecast/conditional.py` | **In** (§6), per-draw conditioning, parameters not updated; SPEC edit §6.2 (states the convention) |
| 8 | `bayesianVARs` parity: reduced-form sampler, GL × SV, probes P1–P5 | SPEC §10.4, §13; P2 §9.9, WP21 | **In** (§9.8; WP29, WP38) |
| 9 | `bvars` parity for `CommonSV` | SPEC §13, §2.2 | **In**, probes P11–P16 decide which quantities gate (§9.8) |
| 10 | DGP verdict tests on linear DGPs | SPEC §10.1, §10.5, §13 | **In**: the parallel DGP session's harness and pre-registered study; WP39 holds the exit to its readings (§9.9) |
| 11 | "Recovery" in the DGP layer | SPEC §10.1 table | **Deferred**, pending Q29; SPEC edit §10.1 |
| 12 | Gruber & Kastner (2025) sparse-vs-dense replication | SPEC §10.7 | **In**, pre-registered (WP40) |
| 13 | `hierarchy="mh"` on the MCMC engine | SPEC §4.4 Minnesota row ("raises until Phase 3"), §4.5 last paragraph, P2 D-12/§6.5, `api.py` | **Deferred**, pending Q15; SPEC edits §4.4, §4.5; message E-5 |
| 14 | Estimated Lenza–Primiceri scaling on the MCMC engine | SPEC §4.5 last paragraph, §5.6, P2 D-21/§3.9, `api.py` | **Deferred**, pending Q16; SPEC edits §4.5, §5.6; message E-6 |
| 15 | Continuous ν prior | P2 Q8 | **Deferred**, pending Q18 |
| 16 | `solver="bcm"` (Bhattacharya et al. 2016) for `q > T` | P2 §3.9, N07 §5 | **Deferred**, pending Q19 (the Cholesky path stays exact for `q > T`, only slower) |
| 17 | A GIG sampler | N07 §9 (open), P2 §3.9 | **In** (D-42) |
| 18 | Analytic forecasts without `lag_coefs` | P2 D-34, §7.3 | **Changed**: analytic and structural-NIG forecasts attach `lag_coefs` (D-49) |
| 19 | The SV-t parity row stays an expected failure until a `stochvol` ≥ 3.2.5 fixture exists | P2 §2.10 Stage 4 (1) | **In** if WP29's R environment has `stochvol` ≥ 3.2.5 (regenerated); otherwise unchanged |
| 20 | Two-series SBC for VAR-SV | N06 §6 (optional) | Not planned |
| 21 | `HMP` (bayesianVARs' hierarchical Minnesota) | SPEC §2.2 table | Not in scope (Q20) |
| 22 | `bvars` as oracle for common SV **with t errors**; `CommonSV` t, random walk, outliers | SPEC §2.2, N08 §A5.4, §A9 | **Deferred**, pending Q24: the t part of the bvars oracle row waits with the t extension |
| 23 | Soft conditions; parameter-updating conditional forecasts | Waggoner & Zha (1999) | **Deferred**, pending Q25 |
| 24 | SPEC §10.6 default-budget text; §10.4 "not run in CI"; §5.8 "CI never touches the network" | SPEC | **Edits** in WP41: CommonSV counts as SV (D-51); the R workflow is outside the test workflows (D-54) |
| 25 | SPEC §12 tree paths (`vol/common_sv.py`, `components/priors/*`) | SPEC §12 | **Edit** in WP41 to the files of §1.2 |

### 0.3 Decision register

| ID | Decision | Source | Notes / supersedes |
| :--- | :--- | :--- | :--- |
| D-36 | **What is shrunk** (pending Q30). The global-local and SSVS priors govern the lag coefficients only (`n·p` per equation). Deterministic terms keep the fixed Normal prior of the form (structural Mode I `N(m, κ_d ψ_i/ι)`; reduced `N(0, κ_d ψ_j)`; `var_const`), the contemporaneous `a_i` keep the Phase 2 prior (Mode I `N(0, κ_a ψ_i/(ι ψ_j))`, or λ-scaled under `iw_equivalent`; Q17). Shrinkage is towards the transform-aware prior mean (SPEC §5.3): the prior governs the **deviation** `d = coefficient − m`. GL priors have no SOC/SUR dummy rows | GK (2025) §2 (intercept `N(0, 1000)`, unshrunk); `bayesianVARs` manual (`prior_intercept` fixed) | SPEC §4.4 lists deterministics and contemporaneous as groups (§0.2 rows 3–4) |
| D-37 | **Scale.** Kernel variance `r_j v_j`: `r_j = ψ_i/(ι ψ_m)` (structural) or `ψ_i/ψ_m` (reduced) for the coefficient of equation `i` on a lag of variable `m`, so the prior is in the units of the Mode I Minnesota prior at `λ = 1` and the model is scale-equivariant (ψ per window, R9). `scale="none"` sets `r = 1` (raw coefficients; parity). No lag decay: lag-wise groups learn it | N05 §4.1; GK (2025) `r̃_i` | — |
| D-38 | **Groups.** `groups="olcl-lagwise"` (default; `2p` groups: own lags of lag `l` across equations, cross lags of lag `l` across equations), `"global"` (1), `"equation-wise"` (`n`), `"covariate-wise"` (`n·p`: regressor `(l, m)` across equations), or an explicit `(n, n·p)` integer array. Names follow `bayesianVARs`; their meaning there is probe P7 | GK (2025) §3.2: semi-global beats vanilla in every simulated design and in the application | — |
| D-39 | **Prior families and parametrisations** (§3.2): HS (Makalic & Schmidt 2016), NG (`a, b, c`, GK/`bayesianVARs` meaning), R2D2 (`a, b`, GK gamma form), DL (`a`, gamma form), SSVS (`c0, c1, semiautomatic, p`). `a` is fixed or has a discrete hyperprior (default: GK's grid) | GK (2025) §2–3, App. B, re-derived here | — |
| D-40 | **One post-phase scale step** `G_COEF_GLOBAL` (`Block 17`, reserved by P2 §5.6) draws every local, group and hyper parameter of a GL prior, for the structural and the reduced form alike, after the equation phase, in the order **local → `a_g` (its rate integrated) → rate `ξ_g | a_g`** (NG, R2D2), local → group (HS, SSVS), local → `a_g` (DL). The coefficient-prior kernel declares `has_local_state = False`, so no `coef_local` equation step is planned | P2 §3.4, §4.8 item 3 (post steps may write `eq.coef`, `glob.*`) | P2 §3.4 sketched local scales in `sample_local_eq`; one step is simpler because semi-global groups cross equations and the reduced form's coefficients are global (`ReducedEq.coefficients_eq` docstring) |
| D-41 | **Log space, no floors.** Every scale is stored as its logarithm; draws are log-gamma, log-inverse-gamma and log-GIG; the mean kernels draw GL coefficients in deviation coordinates with a log-space Jacobi scaling and return `log|d|` exactly (§3.4). No tolerance, floor, truncation or jitter is added to any prior | P2 D-14, D-18, N07 §9 | `bayesianVARs` floors draws by default (`PHI_tol`, `*_tol`); its parity runs set them to 0 (§9.8) |
| D-42 | **GIG sampler**: Devroye (2014) §6 — the three-piece hat on `log X` with his explicit points `t, s` (uniform bound 3.4597 expected iterations, Theorem 2), every exponential term evaluated in log space; `χ = 0` or `ψ = 0` draw log-gamma exactly; attempt cap 512 (§3.5) | Devroye (2014) Theorems 1–2 | E31 |
| D-43 | **Where GL priors run.** Structural Mode I or reduced form, with `Constant("ig")`, `Constant("iw_equivalent")`, every `SV(...)` and `CommonSV()`. Rejected at construction: the conjugate form, Mode II (`Constant("iw")` × structural), the analytic engine. `form="auto"` resolves to `"structural"` (D3). A GL prior varies, so the eigen, fixed-factor and shared-factor paths are off (P2 D-26) | P2 §3.8, §11.1 | — |
| D-44 | **SSVS scales.** `semiautomatic=True` (default): `τ_{0,j} = c0·ŝ_j`, `τ_{1,j} = c1·ŝ_j`, `ŝ_j` the flat-prior posterior s.d. of the coefficient on the window (R9; §3.6 formulas). Needs `T_eff > q_i + 2` for every equation (structural) or `T_eff > k + n + 1` (reduced), else an actionable error at fit. `semiautomatic=False`: `τ = c·√r_j` | George, Sun & Ni (2008); GK (2025) §2.4 | the `bayesianVARs` scaling is probe P9 |
| D-45 | **`AsymmetricConjugate`** is Chan's (2022) prior in our structural form (Mode II arrays with `cross` and `contemp`, δ-coordinates and D3 centring, P2 §6.1), fitted on the **analytic engine** by the existing closed form (`engines/structural_nig.py`, through an adapter): iid draws, exact log ML. `hierarchy`: `"fixed"`, `"eb"` (default; free `lambda` and `cross`, Chan's EB), `"mh"` (random-walk MH over log hyperparameters, as Phase 1). Only with `Constant` volatility (`"auto"` resolves to `"iw"`). Its posterior equals the MCMC composition `Minnesota(form="structural", cross=…) × Constant("iw")` (P2 rule 4), which stays MCMC; a verdict between the two compares engines | Chan (2022) §2–4; P2 Q13, WP16 | — |
| D-46 | **`CommonSV` model**: `u_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `E h_t = 0` (identification), stationary `h_1 ~ N(0, σ_h²/(1 − ρ²))`. Defaults `ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)`, `σ_h² ~ IG(5, 0.04)`; options `ρ: Beta` on `(ρ+1)/2` and `σ_h²: ScaledChi2` (parity) | N08 §0, §A1–A3 (Chan 2020 priors) | — |
| D-47 | **One `CommonSV` program, per equation, with a conjugate fast path.** Every `Linear × CommonSV()` runs the per-equation sampler: Constant cores `λ_i`, `λ_{i,t} = λ_i e^{h_t}`, a post step `h, ρ, σ_h | q_t = Σ_i e_{it}²/λ_i`. With `Minnesota(form="conjugate")` the composition is structural **Mode II at `cross = contemp = 1`**, which is the NIW prior exactly (P2 §6.1, N05 §4.2), and the mean takes the **shared-factor path**: the λ-free prior precision and the weights `e^{−h_t}` are the same for every equation, so one Cholesky of the shared `q × q` matrix serves all equations through its leading blocks (§5.3): `O(T q² + q³ + n q²)` per sweep, the conjugate speed. `form="auto"` with `CommonSV` resolves to `"structural"` (D3 unchanged); the conjugate screen is explicit | N08 §A1, §A5, §A11 item 2; Chan (2022) Prop. 1 | the draft's Kronecker program needed hooks the engine lacks (E29); N08 §0's reduced-form NIW draw targets the same posterior |
| D-48 | **`h` step**: blocked exact independence MH, blocks scanned **sequentially** (each conditions on its updated neighbours), block layout 0-based and covering `[0, T_eff)` every sweep (§5.4), Newton mode from a start that depends only on `q`, the neighbours and `(ρ, σ_h)`; `ρ` truncated-normal independence MH with the stationary-start factor (a uniform proposal when the data part has zero precision); `σ_h²` exact (IG, or GIG under `ScaledChi2`) | N08 §A5.2–A5.3 (derived there; Geweke-gated here) | the Omori-mixture variant (CCM 2016) is not shipped: it is not exact (N08 §A11); see P16 |
| D-49 | **`lag_coefs` on every linear forecast.** Analytic (`AnalyticFit`) and structural-NIG forecasts attach the accelerator variable `lag_coefs` (P2 D-34), so the integrated LPD and conditional forecasts work on every engine. Contract fields unchanged; the Phase 2 test that expects analytic forecasts to raise is amended (E52) | P2 D-34, §7.3 | amends P2 §7.3 "analytic-engine forecasts carry no `lag_coefs`" |
| D-50 | **Conditional forecasts are scenarios.** Hard conditions `y_{T+h, j} = r_{h,j}` on the model scale; per draw, the exact Gaussian conditioning of the unconditional joint path given the parameter draw and its drawn volatility path; the conditions update neither parameters nor volatilities, and draws are not reweighted by how likely they make the conditions (the per-draw convention; the log weights `log N(r; Rμ_s, R₁ᵀR₁)` are stored for users who want to reweight). Computed from an unconditional forecast without new random numbers; returned as `ConditionalForecastResult`, which `score_forecast` refuses (§6) | Waggoner & Zha (1999) §3 per-draw step; Bańbura, Giannone & Lenza (2015) | BoE `bvar` instead re-draws parameters from the data and the conditional path (`tests/parity/BOE_BVAR_NOTES.md` §10), so it is an oracle only per draw (A10b); Q25 |
| D-51 | **Budgets** (extends D-35): `CommonSV` is a stochastic-volatility composition for D-35 (default 4 × 10,000 kept draws, warning below). GL compositions: the default is frozen by the WP35 pilot (§8; Q21, D-58); until then they take the D-35 default of their volatility. The convergence gate (P2 §8.2) is unchanged | P2 D-35, Q14 | — |
| D-52 | **Monitors** (extends D-33): GL group log scales are **primary** when the number of groups `G ≤ max(2p, n)` (global, olcl-lagwise, equation-wise), secondary otherwise; local scales, SSVS indicators and hyperprior indices are secondary. `CommonSV`: `h_T`, `ρ`, `σ_h` primary (with every `log λ_i`, as for `Constant`). They are monitor fields of `fit.posterior["monitor"]` | SPEC §10.6 ("monitored on hyperparameters") | — |
| D-53 | **Grid harmonisation** (extends D-19): if any arm resolves to MCMC through a non-conjugate volatility (`SV`, or `CommonSV` with an `auto` form) or a GL prior, every `Minnesota(form="auto")` arm becomes `"structural"` and its `Constant("auto")` becomes `"ig"`. GL arms are structural already; `AsymmetricConjugate` arms are never rewritten (they stay analytic) and do not trigger the rewrite | P2 §11.4 | — |
| D-54 | **Oracle environment** (Q22, resolved by the owner on 2026-10-25: "(a) CI job + artifact"). An input-gated job `r_fixtures` in the existing `ci.yml` (GitHub dispatches only workflows on the default branch), run by `workflow_dispatch` only (never on push, pull request or schedule), on `ubuntu-latest`, installs R and `bayesianVARs` 0.1.8, `bvars` 1.1 and `stochvol` from a pinned Posit Package Manager snapshot (CRAN sources built with gfortran), runs `tools/make_r_fixtures.R` and uploads the fixtures as an artifact, which is downloaded with `gh run download` and committed with R, package and snapshot versions in the fixtures' attributes. No local R and no container image. Test workflows never install R or touch the network (SPEC §5.8, §10.4 reworded to say so) | SPEC §10.4 | P2 failed on R 4.1.2 (no binary, no gfortran); E37 |
| D-55 | **Numbering** continues Phase 2 (Conventions); SPEC §14 points at both registers | — | — |
| D-56 | **Validation designs for heavy-tailed coefficient priors.** GL composition rows run Geweke and SBC on an exogenous fixed design: `X` of shape `(T_eff, k)` with a column of ones for each deterministic term and `N(0, 1)` entries elsewhere, drawn once from the configuration seed and held fixed, `Y` simulated given `θ` without lag feedback (N08 §A11). The lagged-path GL rows use families with exponential or Gaussian tails and a scale knob (NG, SSVS; DL has no scale parameter) with scales whose explosive prior mass, measured by prior simulation in WP30, is ≤ 2e-4 (P2 §9.2); the horseshoe's half-Cauchy tail makes that mass ≈ 0.4 at `global_scale = 0.1` (Ra8), so no lagged HS row runs with sampled scales | N06 §3.4; N08 §A11 | E46 |
| D-57 | **Branches and PRs.** One PR per work package. Each work-package branch starts from the branch of the work package it depends on (named in §13), and its PR targets that branch, so the diff is the work package alone; GitHub retargets it to `main` when its base merges. Merging is the owner's | task brief | E58 |
| D-58 | (reserved) the GL default budget, decided by the owner from WP35's pilot (Q21) | — | — |
| D-59 | **Every exit criterion is pass/fail.** An unresolved probe, a missing fixture, a skipped parity test or a comparison that fails means the criterion is **not met**; the owner then decides, and any substitution needs a named SPEC §13 amendment. No internal oracle stands in for an external one silently; the release run of WP41 must show the parity tests ran (no skips) | the project rule against changing criteria | E34 |
| D-60 | **Stated tolerances.** Quantile comparisons use the Phase 2 parity rule (Conventions); KS tests use `α = 1e-3`; moment comparisons use `|z| ≤ 4` with Bonferroni over the compared moments; every Tier U tolerance is stated with its test (§9.2). Nothing is compared before its rule is written here or in a pre-registration | P2 §9.9 | E38 |
| D-61 | **Separate Phase 3 validation registry**: `validation.joint` gains `COMPOSITIONS_P3`, `BLOCK_CONFIGS_P3` (and `joint_spec` accepts their names); the Phase 2 tuples and the suites that assert them stay unchanged | P2 §9.2 | E51 |
| D-62 | **Log-aware consumers.** Every reader of a coefficient prior's precision (failure diagnostics, implied prior moments, monitors) accepts `GaussianLogDiag`; implied reduced-form prior moments are reported "not applicable" under a GL prior (the horseshoe has no finite prior variance) | P2 §5.7, §8.4 | E53 |

### 0.4 Review errata (E29–E86)

`Ra`, `Rb`, `Rc` = the three reviewers (header), with their finding numbers. Every finding has a row; related findings
share one.

| # | Finding | Resolution | § |
| :- | :--- | :--- | :- |
| E29 | Rb1, Rb2, Rb3, Rb27, Rb28, Ra11, Rb31: the Kronecker program stores `C` where the forecast core cannot read it, breaks the many `.struct.a` reads, has no slot to compose its global steps, and leaves the NIW degrees of freedom ambiguous | Dropped. The conjugate CommonSV is structural Mode II at `cross = contemp = 1` with the shared-factor path (D-47, §5.3): Recursive structure, no new structure or mean kind, no `G_CONJUGATE` | D-47, §5.3 |
| E30 | Ra2: the scale-step order "local → group → a" draws `(a, ξ)` from the product of marginals | Order fixed: local → `a_g` with `ξ_g` integrated → `ξ_g | a_g`; mutant M35 reverses it; joint test function `a_g·log ξ_g` | D-40, §3.3, §9.6 |
| E31 | Ra1, Ra10, Ra17, Rb32: the GIG envelope points fail at `λ ≈ 0`, `NaN`s from `0·∞`, a 64-attempt cap without a uniform bound, a test grid that misses `λ = 0`, a cost estimate that ignores the slowest `vmap` lane | Devroye (2014) §6 with log-space evaluation, cap 512 (failure probability ≤ (1 − 1/3.46)⁵¹² ≈ 1e-76 per call), pinned test grid including `λ ∈ {0, ±1e-3, ±1e-8}`, every `a − ½`, `a − 1` of the GK grid and `½ − T_max/2`; worst-case attempts reported; the cost counts the slowest lane | D-42, §3.5, §10 |
| E32 | Ra3, Rb12: zeroing the conditioned row and column after a QR of `N_h` drops real covariance | QR of the free columns of `N_h` only, embedded in zeros; A10 includes a conditioned variable ordered before a free one | §6.2 |
| E33 | Ra4, Rb13: a conditional result inherits accelerator variables that make its LPD wrong, and fails `check_structure` | The result drops `cond_A`, `cond_scale`, `cond_scale_persistent`, `log_p`, `log1m_p`, `nu` and `lag_coefs`; its own structure check; `score_forecast` refuses it | §6.3 |
| E34 | Rc1: an unresolved probe would be "replaced by the exact oracles", and the bayesianVARs test skips when a fixture is missing | D-59: not met, owner decides, SPEC amendment named; the release run must show no skips | D-59, §9.8 |
| E35 | Rc2: the bvars fallback tolerance was deferred, and A8/A11 share the `h` kernel | Tolerance stated now (the Phase 2 rule on 1- and 4-step predictive quantiles and on `(Π, Σ)`); A8/A11 are evidence, never the criterion | §9.8 |
| E36 | Rc3: the DGP exit could not fail, and its full study may never run | Exit met only if the registered full study's readings pass (both null pairs and the not-worse pair "not above nominal" at `h = 1` and `h = 4`); compute question Q28; scope stated (one cell, volatility axis) | §9.9 |
| E37 | Rc15: CRAN now ships macOS binaries for R 4.5/4.6; the workflow must pin versions; SPEC wording | D-54 rewritten (a pinned snapshot; SPEC §5.8/§10.4 reworded); the owner chose the dispatch-only CI job with an artifact download (Q22) | D-54 |
| E38 | Rc9: A9, A10, A11, A13, P8 and GK had no tolerance | D-60 and each test's rule | §9 |
| E39 | Rc4: BoE `bvar` re-draws parameters from the conditional path, so it is no end-to-end oracle | BoE used per draw (`draw_constrained_forecasts` at fixed `(beta, sigma)`, A10b); "as in BoE/bvars" removed | D-50, §9.7 |
| E40 | Rc5: bvars samples the common SV with auxiliary mixture indicators (approximate), has hierarchical `σ_ω²` and `V`, one chain, undocumented initial condition | Probe P16 (is bvars exact?); P11–P15 reworded to data fields; the decision tree gates only the quantities an exact oracle can gate; R-side MCSE from `k` seeded runs | §9.8 |
| E41 | Rc6: `bayesianVARs` floors draws (`PHI_tol`, `*_tol`) | Parity runs set every tolerance to 0 (P10 checks 0 is accepted) and the derived call records it | §9.8 |
| E42 | Rc7, Rc8, Rc16, Rc24, Rc25: parity data, draws and horizons unstated; P6/P8/P10 without decision rules; GL scales compared only through coefficients; R9's NG settings; `expert_sv_offset` | Data `var_sv_n3/n7` (P2's simulated parity sets, a three-way check with the SWA oracle); 4 runs × 50,000 kept draws after 10,000 per side; horizons 1–4; P6 remedy (pre-standardised data on both sides); P8 by competing mappings; P6/P10 bands; SSVS inclusion probabilities, the `a`-grid posterior and the HS global scale compared where returned; R9 states `NG_b`, `NG_c` and the grid; `expert_sv_offset` left out | §9.8 |
| E43 | Ra5: the equal-weight mixture is not `p(y | data, RY = r)`; the moments change meaning | D-50: labelled scenarios, log weights stored, `score_forecast` refuses, moments documented | D-50, §6 |
| E44 | Ra6, Ra14, Ra16: block scan order unspecified; dates `[0, o)` never updated; index conventions mixed; interior blocks untested | Sequential scan; 0-based layout `B_0 = [0, o)`, `B_b = [o + (b−1)·block, o + b·block)`; one index convention; row 329 with `block = 7` | §5.4, §9.4 |
| E45 | Ra7: `Beta` ρ prior gives `0/0` at the cold start | `h` starts at the Newton mode of the initial `q`; uniform proposal when the data part of `K_ρ` is 0 | §5.4–§5.5 |
| E46 | Ra8, Rc12: a ≤ 2e-4 explosive mass forces a useless HS scale; structural GL rows were all exogenous | D-56: lagged rows with NG/SSVS (rows 316, 319, 330), HS only exogenous; coverage argument for the untested combinations | D-56, §9.4 |
| E47 | Ra9, Rb21: M22 had no Geweke row able to catch it | Row 326 (NG, grid `a`, fixed `c = 1`); Tier U closed form with fixed `c` | §9.4, §9.6 |
| E48 | Ra12: structural semi-automatic `ŝ` missed the residual-variance factor | `ŝ²_j = RSS_i/(T_eff − q_i − 2)·[(F_iᵀF_i)⁻¹]_jj`, condition `T_eff > q_i + 2` | D-44, §3.6 |
| E49 | Ra13, Rb33: the SBC predictive `y_{T+1}` needs lags | Only on lagged rows | §9.5 |
| E50 | Rb4: `constant_weights` stays true with a common `h`, so the eigen/fixed-factor paths would ignore `h` | `constant_weights` is false when `common`; rule (13) says so; the shared-factor path is the only Mode II path with common weights | §2.1 |
| E51 | Rb15: new rows in the shared registry break and duplicate the Phase 2 suites | D-61 separate registry | D-61 |
| E52 | Rb14: D-49 and the new error messages break Phase 2 tests | Amendments listed per WP (`tests/integration/test_backtest_lpd_integrated.py` uses `OLSVAR` for the raising case; `tests/unit/test_api_resolution.py` messages), as P2 WP2 did in `ed502d4` | §1.3, §13 |
| E53 | Rb25: `GaussianLogDiag` has no `.prec` for its readers | D-62 | D-62 |
| E54 | Rb5, Rb6, Rb26, Rb29: CommonSV's `rb_h_*` names crash storage; forecasts, IRFs and `draw_parameters` ignore the common `h`; cores receive no global vol state; futures are positional; budgets ignore CommonSV | Leaves `rb_ch_sum`, `rb_ch_sq`; layout `h[T_pad]` with its storage and warm-start handling; `PersistentCore` gains `uses_global_vol` and `gvol` arguments; futures looked up by block id; `_pool`, `_log_lam`, `_sv_budget`, `fit_mcmc` and `default_mcmc` count CommonSV | §2, §5.5, §8 |
| E55 | Rb7: files that must change were not listed | §1.2 lists every file with one owner | §1.2 |
| E56 | Rb8, Rb9, Ra20: GL scale storage, initialisation order, `dev_log` at init | Scales as monitor fields; the runner initialises `glob.coef` (`CoefPrior.init_global`) before the mean's cold draw, whose draw sets `dev_log` | §3.7 |
| E57 | Rb10, Rb11: builders would make `import rai_bvar` load JAX; an early export of a later module | Builders in `priors/global_local_arrays.py` (JAX side); `rb.conditional_forecast` through the lazy table, owned by WP34 | §1.2 |
| E58 | Rb23, Rc17, Rc21: dependencies, bundled PRs, WP30's independence, the missing WP36 | §13 redrawn: one PR per WP, consecutive numbers, named bases, WP30 and WP36 by an author who wrote no kernel (recorded in the PR) | §13 |
| E59 | Rb16: shard partition and timeouts; PR-tier budget | Two Phase 3 shards each for Geweke and SBC, the old shards `--ignore` the new files, shard edits in WP36, PR-tier increase measured and capped at +10 min per job | §9.11 |
| E60 | Rb17, Rb18, Ra23: the NIG code is Minnesota-only and cannot free `cross`; ψ free; flat prior under `"mh"` | Adapter to an internal structural `Minnesota`; `cross`, `contemp` join the free names; ψ rejected whenever `cross` is free or `≠ 1`; the flat prior on `log cross` over `[log 1e-3, log 10]` stated | §4 |
| E61 | Rb19: `structural/ordering.py` and `monitor.is_order_invariant` | Listed: prior mode via the NIG closed form, `_posterior_summary` for `StructuralNIGFit`, permuted custom groups; Mode II at `cross = contemp = 1` (with `Constant` or `CommonSV`) is order-invariant | §4.2, §5.3 |
| E62 | Rb20: construction-time validation would reject every GL model | `api.py` passes `coef_log`; `G` is resolved at fit (`resolve.py`); rule (15) runs at fit | §2.1, §7 |
| E63 | Rb22: no seam for M33 | `forecast/conditional.py` takes a keyword-only `_update` (tests only) | §6, §9.6 |
| E64 | Rb24: harmonisation would break `AsymmetricConjugate` arms | D-53 restricted to Minnesota arms | D-53 |
| E65 | Rb30: `const_path="cholesky"` before the varying-prior check | `resolve_const_path` checks `COEF_VARIES` first | §2.1 |
| E66 | Rb34: in-loop forecasts without `lag_coefs` | E-16 names it | §6.1 |
| E67 | Rb35: block configurations need ids; P2 runs block-configuration G-small weekly | Ids 401…; weekly, with the per-PR block Geweke in the kernel suites | §9.3 |
| E68 | Rb36: the default grid must be hashable and should not put 2,000 floats in every spec | `DiscreteGrid` holds tuples; the GK grid serialises as `{"class": "DiscreteGrid", "name": "gruber_kastner"}` | §7.4 |
| E69 | Rb37, Rb40: constructor defaults and template comparisons; `log_tau` leaves | Every new field defaults to `Empty()`; the golden test compares leaf paths, shapes and dtypes; SSVS `log_tau0/1` exist only for SSVS | §2 |
| E70 | Rb38: `CCCMStep.reads` omits `eq.coef`, `glob.coef` | Added | §2.4 |
| E71 | Rb39: AsymmetricConjugate equals the MCMC Mode II posterior | Stated in D-45 | D-45 |
| E72 | Rc10, Rc11, Rc26, Rc27: GK settings and criteria | §9.10 rewritten with GK's sample, prior mean 0, intercept prior, listed departures, global arms, directional criteria, named scores, "beats" defined; the cannot-fail acceptance is an owner decision (Q31); compute from the pilot (Q26) | §9.10 |
| E73 | Rc13: SPEC §10.1 "Recovery" | Q29, row 11 of §0.2 | §0.2 |
| E74 | Rc14: `auto` × CommonSV contradicted D3; unnamed SPEC edits; deferrals presented as settled | D3 kept (`auto` → structural); §0.2 rows marked pending and every SPEC sentence named | D-47, §0.2 |
| E75 | Rc18: `--sections=bvars` already means bayesianVARs | Woźniak's package: section `wbvars`, fixtures `wbvars_*.nc` | §9.8 |
| E76 | Rc19: printing R functions shows GPL source | Clean-room paragraph: data fields only | header |
| E77 | Rc20: §0.4 said "pending" while the header said "revised" | This table | §0.4 |
| E78 | Rc22: weekly contract tests for every MCMC composition | Every new MCMC composition in the weekly contract tier | §9.11 |
| E79 | Rc23: the DGP arms' spec hashes | Added to WP26/WP27 golden acceptance; the DGP readings refer to the pre-registration's commit | §2, §9.9 |
| E80 | Rc28, Rc29: §6.2 and §2.2 SPEC sentences | §0.2 rows 7, 22, 25 | §0.2 |
| E81 | Ra15, Ra16: `log g(ρ*) − log g(ρ)` of a log; `K_h`'s last diagonal | `g(ρ*) − g(ρ)`; last diagonal `1/σ_h²` at index `T_eff − 1`, padded states held at 0 | §5.4 |
| E82 | Ra18: §3.4 tolerances contradict `linalg.py`'s conditioning-aware rule | Path-vs-path `2e-12 + 5εκ(1 + t)` (P2 §9.3 item 1); the dense oracle takes log precisions | §3.4, §9.2 |
| E83 | Ra19: `b̃` needs the active-slot mask; dummy rows | Mask in the formula; GL priors carry no dummy rows (D-36); the shared-factor path's dummy rows are λ-free and not weighted by `h` | §3.4, §5.3 |
| E84 | Ra21, Ra22: `σ` not stored as a log; symbol overloads | `log_sigma`; the Minnesota scale is `r_j`, the exponential mixing variable `ϱ_j` | §2.2, §3 |
| E85 | Ra24, Ra25, Ra26, Ra27: A13 at engine level; the exogenous design's intercept; block-configuration draws; dummy rows under CommonSV | A13 runs engine-level arrays; D-56 states the ones column; "coefficients drawn from `p(d | current scales)`"; λ-free dummy weights in §5.3 | §9 |
| E86 | Ra28: M28 might be invisible at `T_eff = 20` | A11's quadrature (10⁵ draws, KS `α = 1e-3`) is M28's required catcher | §9.6 |

---

## 1. Module layout and ownership

### 1.1 Import-time rules

P2 §1.1 holds unchanged: `import rai_bvar` imports no JAX; the new user-facing specs (`priors/global_local.py`,
`priors/asymmetric.py`, `components/specs.py` `CommonSV`) are pure Python; array builders that return engine pytrees
live in `priors/global_local_arrays.py` and are imported only by `engines/mcmc/resolve.py`; kernels import JAX and
are loaded only through `engines/mcmc/registry.py`. `rb.conditional_forecast` is resolved through the lazy table of
`rai_bvar/__init__.py`. The analytic engine for `AsymmetricConjugate` is NumPy only.

### 1.2 Files (new or changed; the owning work package of each part, §13)

A file with two owners is changed by two work packages that are consecutive in §13's linear stack, each in its own
part of the file (named here), so two writers never touch a file at the same time.

```text
src/rai_bvar/__init__.py                         [WP27] exports (GL classes, DiscreteGrid, AsymmetricConjugate, CommonSV)
                                                 [WP34] lazy conditional_forecast
src/rai_bvar/api.py                              [WP27] resolution, engine table §7.2, errors §7.3, harmonise (D-53), coef_log
src/rai_bvar/components/specs.py                 [WP27] + CommonSV
src/rai_bvar/priors/global_local.py        new   [WP27] user-facing GL specs, DiscreteGrid
src/rai_bvar/priors/asymmetric.py          new   [WP27] AsymmetricConjugate spec
src/rai_bvar/priors/sv.py                        [WP27] + TruncatedNormal
src/rai_bvar/priors/global_local_arrays.py new   [WP31] host builders (groups, log r, SSVS ŝ, hyper arrays)
src/rai_bvar/priors/scales.py                    [WP33] adapter for AsymmetricConjugate (window_scales, conjugate_omega)
src/rai_bvar/engines/mcmc/config.py              [WP26] §2.1
src/rai_bvar/engines/mcmc/state.py               [WP26] §2.2–§2.3 classes, templates, invariants
src/rai_bvar/engines/mcmc/keys.py                [WP26] Block/Sub additions
src/rai_bvar/engines/mcmc/registry.py            [WP26] kinds, slot_kinds (global steps of vol blocks)
src/rai_bvar/engines/mcmc/protocols.py           [WP26] GaussianLogDiag, uses_global_vol, VolBlock.global_steps, INTERNALS
src/rai_bvar/engines/mcmc/compose.py             [WP26] collects VolBlock.global_steps(); capability rules
src/rai_bvar/engines/mcmc/linalg.py              [WP28] precision_draw_log, shared_factor_*, tridiagonal Newton
src/rai_bvar/engines/mcmc/random.py              [WP28] log_gig, bernoulli_logit
src/rai_bvar/engines/mcmc/runner.py              [WP31] init order (CoefPrior.init_global), log-aware failure_detail
                                                 [WP32] extract of glob.vol draws
src/rai_bvar/engines/mcmc/storage.py             [WP31] GL monitor fields  [WP32] CommonSV latent moments and h_T
src/rai_bvar/engines/mcmc/results.py             [WP31] implied moments n/a under GL  [WP32] CommonSV default draws
src/rai_bvar/engines/mcmc/forecast.py            [WP32] common h in _pool/_log_lam, futures by block id
src/rai_bvar/engines/mcmc/warmstart.py           [WP32] remap glob.vol.h
src/rai_bvar/engines/mcmc/resolve.py             [WP31] GL static parts and arrays  [WP32] CommonSV, shared-factor rule
src/rai_bvar/engines/mcmc/monitor.py             [WP31] GL monitors/warnings  [WP32] CommonSV monitors, order invariance
src/rai_bvar/components/coef/global_local.py new [WP31] HS, NG, R2D2, DL, SSVS + GLScaleStep (G_COEF_GLOBAL)
src/rai_bvar/components/mean/linear.py           [WP31] deviation path  [WP32] shared-factor path
src/rai_bvar/components/mean/linear_reduced.py   [WP31] CCCM deviation path, dev_log, log prior from the scales, reads
src/rai_bvar/components/vol/common.py      new   [WP32] CommonSVStep (G_COMMON_SV), h/ρ/σ kernels, futures
src/rai_bvar/components/vol/constant.py          [WP32] common-h conditional (uses_global_vol)
src/rai_bvar/components/vol/robust.py            [WP32] passes gvol; log_var adds h_t; global_steps()
src/rai_bvar/engines/structural_nig.py           [WP33] StructuralNIGFit, free cross/contemp, MH problem
src/rai_bvar/engines/analytic.py                 [WP33] forecast lag_coefs (D-49)  [WP34] conditional_forecast method
src/rai_bvar/structural/ordering.py              [WP33] StructuralNIGFit support, permuted custom groups
src/rai_bvar/forecast/conditional.py       new   [WP34] conditions, the exact update, ConditionalForecastResult
src/rai_bvar/forecast/invariants.py              [WP34] check_conditional_structure
src/rai_bvar/evaluation/scores.py                [WP34] refuses ConditionalForecastResult
src/rai_bvar/validation/{reference,simulate,log_joint,testfns,joint}.py  [WP30] §9.1 (separate registry, D-61)
tools/make_r_fixtures.R                          [WP29] sections bvars (bayesianVARs P1–P10, R1–R9), wbvars (P11–P16)
tools/make_sv_parity_data.py                     [WP29] packing of the new fixtures
.github/workflows/ci.yml                         [WP29] r_fixtures job (D-54)  [WP36] weekly shards  [WP41] tiers
tools/weekly_shards.py                           [WP36] Phase 3 shards
benchmarks/pilot/phase3_*.py               new   [WP35] pilot; docs/phase3/pilot.md
benchmarks/phase3_perf.py                  new   [WP35] BENCH-P3 (reported)
benchmarks/gruber_kastner.py               new   [WP40] replication; benchmarks/reports/gruber_kastner_*
SPEC.md, README.md, pyproject.toml, docs/phase3/DESIGN.md   [WP41]
```

Files owned by the parallel sessions are not touched: `src/rai_bvar/validation/dgp.py`,
`src/rai_bvar/validation/__init__.py`, `tests/dgp/`, `benchmarks/dgp_study.py`, `benchmarks/reports/dgp_study_*`
(DGP harness) and every `*_bart_spike*` file (Phase 4 spike). New validation helpers are imported from their modules,
not re-exported through `validation/__init__.py`.

### 1.3 Tests (owner)

```text
tests/mcmc/unit/      [WP26] test_contracts_p3 (rules, templates, Phase 2 goldens incl. the DGP arms' hashes)
                      [WP28] test_gig, test_linalg_log, test_shared_factor, test_newton
tests/unit/           [WP27] test_specs_p3, test_api_p3 (every §7.2 row, every §7.3 message, hashes, labels, harmonise)
                      [WP27] amendments: test_api_resolution.py messages (E52)
tests/priors/         [WP31] test_global_local_arrays
tests/mcmc/blocks/    [WP31] test_global_local (conditionals, density ratios, block Geweke)
                      [WP32] test_common_sv (h kernel, ρ, σ, shared factor, quadrature)
tests/analytic/       [WP33] test_asymmetric (A9)    [WP31] A12, A13    [WP32] A8 (n = 3), A11    [WP37] A8 (n = 20), A9 MCMC leg
tests/forecast/       [WP34] test_conditional (A10, A10b, contract)
                      [WP33] amendment: tests/integration/test_backtest_lpd_integrated.py raising case → OLSVAR (E52)
tests/validation/     [WP30] test_reference_p3
tests/geweke/test_geweke_p3*.py, tests/sbc/test_sbc_p3*.py, tests/mutation/test_mutants_p3.py   [WP36]
tests/parity/         [WP38] test_bayesianvars (fixtures present, no skip), test_wbvars_commonsv, test_boe_conditional
tests/contract/       [WP36] the new MCMC compositions (fast: three; weekly: all)
tests/integration/    [WP40] test_gruber_kastner_script
```

---

## 2. Contracts (WP26)

All additions keep every Phase 1 and Phase 2 `StaticConfig`, state template (leaf paths, shapes, dtypes), spec and
spec hash unchanged: new fields default to the Phase 2 value, new leaves default to `Empty()` (also in constructor
calls such as `CCCMGlobal(Pi=Pi)`), and new branches are taken only when a Phase 3 component is present (P2 §2.1).
A golden test pins the templates and hashes of every Phase 2 composition and of the DGP study's three arms
(`8aebe5adec44e966`, `b3daa828453542ed`, `7f0aeb733eb2d07e`) at `2c71fb2`, and short runs of every Phase 2
composition are bitwise unchanged before and after WP26–WP34.

### 2.1 Static configuration (`engines/mcmc/config.py`)

```python
ConstPath = Literal["weighted", "cholesky_const", "eigen", "fixed_factor", "shared_factor"]   # + shared_factor (§5.3)
CoefKind  = Literal["fixed_diagonal", "horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs"]

@dataclass(frozen=True)
class MeanStatic:      ... ; coef_log: bool = False          # deviation path of §3.4 (iff the coefficient prior varies)
@dataclass(frozen=True)
class CoefPriorStatic:
    kind: CoefKind = "fixed_diagonal"
    n_groups: int = 0                                         # G, resolved at fit (shape-defining); 0 for fixed_diagonal
    a_grid: int = 0                                           # size of the discrete hyperprior grid on a (0 = fixed a)
    p_beta: bool = False                                      # SSVS: Beta hyperprior on p
@dataclass(frozen=True)
class VolStatic:       ... ; common: bool = False             # λ_{i,t} = λ_i e^{h_t}, h in glob.vol (§5)
                           csv_block: int = 100               # h-step MH block length (D-48)
                           csv_rho_prior: Literal["truncated_normal", "beta"] = "truncated_normal"
                           csv_sigma2_prior: Literal["inv_gamma", "scaled_chi2"] = "inv_gamma"
    @property
    def constant_weights(self) -> bool: return persistent == "constant" and transitory == "none" and not common
```

`COEF_VARIES` gains `True` for every GL kind. `validate_composition` adds, in order: (11) a varying coefficient prior
needs `coef_log = True`, a `weighted` path and no λ-scaled coefficient block (Mode II); (12) `coef_log` needs a
`linear_structural` or `linear_reduced` mean; (13) `common = True` needs `persistent = "constant"` and
`transitory = "none"`, and makes `constant_weights` false; (14) `shared_factor` needs `linear_structural`, Mode II,
`common = True`, a non-varying coefficient prior, and (checked by `resolve.py` on the arrays, since it is data) an
equation-invariant λ-free prior precision; (15) `n_groups ≥ 1` iff a GL kind, `a_grid > 0` only for NG, R2D2 and DL,
`p_beta` only for SSVS (run when `resolve.py` builds the config, because `G` depends on `n`); (16) `csv_block ≥ 2`.
`api.py` passes `coef_log` to its construction-time check, so GL models pass rules (11)–(12) at construction.
`resolve.resolve_const_path` checks `COEF_VARIES` before an explicit `const_path="cholesky"` (E65).

### 2.2 State (each new class registered with `register_dataclass`; scales stored as logs)

```text
LinearEq          delta: f64[n, k]; dev_log: f64[n, k] | Empty = Empty()    log|δ_i − m_i| (§3.4) iff coef_log
CCCMGlobal        Pi: f64[k, n];    dev_log: f64[n, k] | Empty = Empty()    log|π_j − m_j|, row j = equation j
HSEq              log_theta, log_nu: f64[n, k]                              local ϑ, auxiliary ν (0 on non-GL slots)
HSGlobal          log_zeta, log_tau: f64[G]
NGEq              log_w: f64[n, k]                                          ϖ
NGGlobal          log_xi: f64[G]; a_idx: i32[G]                             rate ξ; grid index (0 with a fixed a)
R2D2Eq            log_varrho, log_w: f64[n, k]                              ϱ (exponential mixing), ϖ
R2D2Global        log_xi: f64[G]; a_idx: i32[G]
DLEq              log_theta, log_varrho: f64[n, k]                          θ̃ (Laplace scale), ϱ
DLGlobal          a_idx: i32[G]
SSVSEq            gamma: i32[n, k]; rb_incl: f64[n, k]                      indicator; Σ P(γ = 1 | ·) over sampling sweeps
SSVSGlobal        log_p, log1m_p: f64[G]
CommonSVGlobal    h: f64[T_pad]; rho: f64[]; log_sigma: f64[]               index t = data row t (no h_0), 0 for t ≥ T_eff
                  acc_h, acc_rho, acc_newton: f64[]; rb_ch_sum, rb_ch_sq: f64[T_pad]
```

`EqState.coef` holds the `*Eq` class, `GlobalState.coef` the `*Global` class, `GlobalState.vol` `CommonSVGlobal`
(all `Empty` in Phase 2). Draw containers (P2 §2.10): `GlobDraw.vol = CommonSVDraw(h_T, rho, sigma)`, filled by the
volatility block's new `extract_global(glob)` (runner `extract`, WP32); GL scales are not part of `ParamDraw` (they do
not enter forecasts) and are stored as monitor fields (D-52). Storage (WP32): `latent_moments/common_h_mean,
common_h_sd (chain, h_index)` from `rb_ch_*` with `h_index` = data rows, `posterior/common_h_T`, `rho`, `sigma`.
Invariants (P2 §2.8) gain: GL leaves finite on GL slots and exactly 0 elsewhere; `a_idx` in range; `γ ∈ {0, 1}`;
`|ρ| < 1`; `h[T_eff:] == 0`.

### 2.3 Prior arrays (`resolve.py` from the WP31/WP32 builders; traced)

```text
CoefPriorArrays  ... + gl: GLArrays | Empty = Empty()
GLArrays         mask: f64[n, k]        1 on GL-governed slots (lag coefficients), 0 elsewhere
                 group: i32[n, k]       group id on GL slots (0 elsewhere, masked)
                 log_r: f64[n, k]       log scale r_j (D-37); 0 when scale="none"
                 ssvs: SSVSArrays | Empty                 log_tau0, log_tau1: f64[n, k] (SSVS only)
SharedPrior      ... + gl: GLShared | Empty;  shared_prec: f64[q] | Empty;  csv: CommonSVPrior | Empty
GLShared         b, c, global_scale: f64[]; log_a_fixed: f64[]; a_support, a_logw: f64[A]; p_fixed, s1, s2: f64[]
shared_prec      the λ-free Mode II precision of θ = [δ | a] (equal across equations; shared-factor path only)
CommonSVPrior    rho_mean, rho_var, rho_a, rho_b: f64[]; s2_shape, s2_scale, s2_B: f64[]; n_eq: f64[]
```

### 2.4 Protocols, keys, registry, internals

- `protocols.GaussianLogDiag(mean: f64[m], log_prec: f64[m], cmask: f64[m])`: what a varying `CoefPrior.moments_eq`
  returns; a mean receiving it takes the deviation path (static `coef_log`). Readers per D-62.
- `PersistentCore.uses_global_vol: ClassVar[bool]` (default False); when True, `VolBlockImpl` passes `gvol` to the
  core's `sample`, `log_var`, `ahead` and `extract` (keyword `gvol`). `VolBlock.global_steps() -> tuple[GlobalStep,
  ...]` (default `()`) and `VolBlock.extract_global(glob)`; `compose` collects the vol block's global steps beside the
  mean's and the coefficient prior's (P2 §3.8 item 4).
- Global futures are passed as a mapping `{Block: future}`, so a block finds its own (WP32 changes `forecast.py`'s
  positional tuple; P2 §7.2).
- `CoefPrior.init_global(key, st, data, prior) -> ChainState` (default identity), called by the runner's cold start
  **before** the mean's cold draw (WP31).
- `Block` (append-only, P2 §5.6): `G_COEF_GLOBAL = 17` (post, GL scale step), `G_COMMON_SV = 18` (post). 19 stays
  reserved (Q16).
- `Sub` (append-only): `GL_LOCAL = 15`, `GL_LOCAL_AUX = 16`, `GL_GROUP = 17`, `GL_GROUP_AUX = 18`, `GL_A = 19`,
  `CSV_OFFSET = 20`, `CSV_H = 21`, `CSV_H_ACC = 22`, `CSV_RHO = 23`, `CSV_RHO_ACC = 24`, `CSV_S2 = 25`. Per-coordinate
  draws fold the flat coordinate index (`uint32`), per-group draws the group id, per-block `h` draws the block index.
- `registry.KERNELS` appends `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs`
  (`components.coef.global_local:*`), `gl_scales` (`…:GLScaleStep`), `common_sv` (`components.vol.common:CommonSVStep`).
- `protocols.INTERNALS` appends `gl_local_conditional`, `gl_group_conditional`, `gl_a_conditional`, `ssvs_logodds`,
  `gl_deviation`, `csv_newton_start`, `csv_q`, `csv_rho_accept`, `csv_lambda_conditional`, `shared_factor_rows`,
  `gig_sampler` (test seam for §9.6; production never sets them).
- `CCCMStep.reads` gains `eq.coef` and `glob.coef` (it reads them through `moments`).

---
## 3. Global-local and selection priors

### 3.1 Coordinates, deviation and kernel (D-36, D-37)

Lag slot `r ∈ {n_d, …, k−1}` of the `data/design.py` layout (deterministic terms, then the lag-1 block of `n`
variables, then lag 2, …) has lag `l(r) = 1 + ⌊(r − n_d)/n⌋` and variable `m(r) = (r − n_d) mod n`; it is an own lag of
equation `i` iff `m(r) = i`. A **GL coordinate** `j = (i, r)` is a lag slot of equation `i`; there are `N = n²p`.
The coefficient is `δ_{i,r}` (structural, δ-coordinates of D3: `γ_i = δ_i + M_c a_i`, P2 §2.3) or `π_{r,i}`
(reduced). Its prior mean `m_j` is the transform-aware own-lag mean at own lag 1 and 0 elsewhere (the Phase 2 arrays'
`mean0`). The prior governs the deviation `d_j = coefficient − m_j`:

```text
d_j | v_j  ~ N(0, r_j v_j)  independently,      log r_j = log ψ_i − log ι − log ψ_m   (structural, scale="psi")
                                                log r_j = log ψ_i − log ψ_m           (reduced,    scale="psi")
                                                log r_j = 0                           (scale="none")
```

`v_j` is the family's mixing variance (§3.2). The group of `j` is `g(j)` (D-38; `n_g` counts the GL coordinates of
group `g`). Deterministic slots and `a_i` keep the Phase 2 Normal priors (D-36). Every density and conditional below
is written for `ℓ_j = log|d_j|`, which the mean kernels deliver exactly (§3.4); `d_j²/r_j` is always
`exp(2ℓ_j − log r_j)`.

### 3.2 The five families (D-39)

Each family lists its prior, its state (all logs; P2 §2.1 rules), its exact conditionals, and its `bayesianVARs`
counterpart (mapping settled by probe P8). `lnΓ` is `gammaln`; `LG(α)` is a draw of `log G(α, 1)`
(`jax.random.loggamma`, exact for tiny `α`); gammas and inverse gammas are drawn as `log x = LG(α) − log β` (rate `β`)
and `log x = log β − LG(α)` (scale `β`).

**Horseshoe** (Carvalho, Polson & Scott 2010; auxiliary form of Makalic & Schmidt 2016). `v_j = ϑ_j ζ_g`,
`√ϑ_j ~ C⁺(0, 1)`, `√ζ_g ~ C⁺(0, s)` (`s = global_scale`, default 1), written as
`ϑ_j | ν_j ~ IG(½, 1/ν_j)`, `ν_j ~ IG(½, 1)`, `ζ_g | τ_g ~ IG(½, 1/τ_g)`, `τ_g ~ IG(½, 1/s²)`. State
`HSEq(log_theta, log_nu)`, `HSGlobal(log_zeta, log_tau)`. Conditionals (*derived here*; GK App. B prints `τ_j` where
`ζ_g` belongs in the first line, which mutant M20 reproduces), in this order:

```text
ϑ_j | ν_j, ζ_g, d_j  ~ IG(1, 1/ν_j + d_j²/(2 r_j ζ_g))
ν_j | ϑ_j            ~ IG(1, 1 + 1/ϑ_j)
ζ_g | τ_g, ϑ, d      ~ IG((n_g + 1)/2, 1/τ_g + Σ_{j∈g} d_j²/(2 r_j ϑ_j))
τ_g | ζ_g            ~ IG(1, 1/s² + 1/ζ_g)
```

Log scales: `log(1/ν_j + d_j²/(2r_jζ_g)) = logaddexp(−log ν_j, 2ℓ_j − log 2 − log r_j − log ζ_g)`; the group sum is a
`segment_logsumexp`. No GIG is needed. `bayesianVARs`: `prior="HS"`.

**Normal-Gamma** (Griffin & Brown 2010; GK's parametrisation). User parameters `a, b, c`: `ϑ_j ~ G(a_g, a_g/2)`,
`ζ_g⁻¹ ~ G(b, c)`, `v_j = ϑ_j ζ_g`. Equivalently `v_j = ϖ_j | ξ_g ~ G(a_g, ξ_g)` with the rate `ξ_g = a_g/(2ζ_g) ~
G(b, 2c/a_g)`. `c="a/2"` ties `c` to `a` (GK's default), so `2c/a = 1`. State `NGEq(log_w)`, `NGGlobal(log_xi, a_idx)`.
Conditionals, in this order:

```text
ϖ_j | d_j, ξ_g     ~ GIG(a_g − ½, 2ξ_g, d_j²/r_j)
a_g | ϖ            (ξ_g integrated)  log P(a_g = ã | ϖ) = log π(ã) + (ã − 1) S_g − n_g lnΓ(ã) + b log(2c/ã)
                                                          + lnΓ(n_g ã + b) − (n_g ã + b) log(2c/ã + W_g) + const
ξ_g | a_g, ϖ       ~ G(n_g a_g + b, 2c/a_g + W_g),        S_g = Σ_{j∈g} log ϖ_j,  W_g = Σ_{j∈g} ϖ_j
```

`a_g` is drawn from its ξ-marginal and **then** `ξ_g` given it: a partially collapsed Gibbs step that preserves the
posterior only in this order (E30; mutant M35 reverses it). With a fixed `a` only the `ξ_g` line runs. The `a`-weight
keeps the `p(ξ_g | a)` factor `b log(2c/ã)`, which GK's printed (uncollapsed) form omits (it is 0 when `c` is tied);
mutant M22 drops it. `log W_g` is a `segment_logsumexp`. `bayesianVARs`: `prior="NG"`, `NG_a, NG_b, NG_c`.

**R2D2** (Zhang, Naughton, Bondell & Reich 2022; GK's gamma form). User parameters `a, b`: `v_j = ϱ_j ϖ_j/2`,
`ϱ_j ~ Exp(½)`, `ϖ_j | ξ_g ~ G(a_g, ξ_g)`, `ξ_g ~ G(b, 1)` (with a Dirichlet split, `R² ~ Beta(n_g a, b)` in each group).
State `R2D2Eq(log_varrho, log_w)`, `R2D2Global(log_xi, a_idx)`. In this order:

```text
ϱ_j | d_j, ϖ_j         ~ GIG(½, 1, 2 d_j²/(r_j ϖ_j))
ϖ_j | d_j, ϱ_j, ξ_g    ~ GIG(a_g − ½, 2ξ_g, 2 d_j²/(r_j ϱ_j))
a_g | ϖ (ξ_g integrated)  log P(a_g = ã | ϖ) = log π(ã) + (ã − 1) S_g − n_g lnΓ(ã) + lnΓ(n_g ã + b) − (n_g ã + b) log(1 + W_g)
ξ_g | a_g, ϖ           ~ G(n_g a_g + b, 1 + W_g)
```

`bayesianVARs`: `prior="R2D2"`, `R2D2_a, R2D2_b`.

**Dirichlet–Laplace** (Bhattacharya, Pati, Pillai & Dunson 2015; gamma form). User parameter `a`:
`d_j | ϱ_j, θ̃_j ~ N(0, r_j ϱ_j θ̃_j²)`, `ϱ_j ~ Exp(½)`, `θ̃_j ~ G(a_g, ½)`. This is the DL prior: with
`θ̃_j = φ_j ω`, `φ ~ Dir(a, …, a)` and `ω ~ G(n_g a, ½)` the products are iid `G(a, ½)` (gamma–Dirichlet), and the
kernel `N(0, ϱ θ̃²)` with `ϱ ~ Exp(½)` is the Laplace density with scale `θ̃`. No global parameter besides `a_g`.
State `DLEq(log_theta, log_varrho)`, `DLGlobal(a_idx)`. In this order:

```text
θ̃_j | d_j (ϱ_j integrated)   ~ GIG(a_g − 1, 1, 2|d_j|/√r_j)
ϱ_j | d_j, θ̃_j               ~ GIG(½, 1, d_j²/(r_j θ̃_j²))
a_g | θ̃                      log P(a_g = ã | θ̃) = log π(ã) − n_g ã log 2 + (ã − 1) Σ_{j∈g} log θ̃_j − n_g lnΓ(ã)
```

The first two lines are one exact block draw of `(θ̃_j, ϱ_j) | d_j`. `bayesianVARs`: `prior="DL"`, `DL_a`.

**SSVS** (George & McCulloch 1993; George, Sun & Ni 2008). `d_j | γ_j ~ N(0, τ²_{γ_j, j})`, `γ_j | p_g ~
Bernoulli(p_g)`, `p_g` fixed or `~ Beta(s₁, s₂)`; spike and slab standard deviations per D-44. State
`SSVSEq(gamma, rb_incl)`, `SSVSGlobal(log_p, log1m_p)`. In this order:

```text
logit P(γ_j = 1 | ·) = log p_g − log(1 − p_g) + log τ_{0j} − log τ_{1j} − ½ (d_j²/τ_{1j}² − d_j²/τ_{0j}²)
p_g | γ              ~ Beta(s₁ + N1_g, s₂ + n_g − N1_g),   N1_g = Σ_{j∈g} γ_j
```

`γ_j` is drawn as `log u < log σ(logit)` with one uniform; `rb_incl` accumulates `P(γ_j = 1 | ·)`. The kernel's log
precision is `−2 log τ_{γ_j, j}`. `bayesianVARs`: `prior="SSVS"`, `SSVS_c0, SSVS_c1, SSVS_semiautomatic, SSVS_p`.

**Hyperprior on `a`** (NG, R2D2, DL). `a` is a float (fixed) or a `DiscreteGrid(support, weights)`; the default
`DiscreteGrid.gruber_kastner()` is GK's: 1,000 equally spaced points on `[0.001, 1]`, weights proportional to the
`Exp(1)` density. The grid is data (`GLShared.a_support`, `a_logw`); only its size is static. Each group has its own
`a_g` (semi-global, GK `NG*_a`, `R2D2*_a`); the draw is one inverse-CDF categorical on the `A` log weights.

**Defaults** (Q23). `Horseshoe()`; `NormalGamma(a=DiscreteGrid.gruber_kastner(), b=0.5, c="a/2")`;
`R2D2(a=DiscreteGrid.gruber_kastner(), b=0.5)`; `DirichletLaplace(a=DiscreteGrid.gruber_kastner())`;
`SSVS(c0=0.01, c1=100.0, semiautomatic=True, p=0.5)`; all with `groups="olcl-lagwise"`, `scale="psi"`: GK's
best-performing configurations (semi-global, hyperprior on `a`). `bayesianVARs`' defaults (global grouping, fixed `a`,
`NG_c = 1`, floors) are set explicitly in the parity runs.

### 3.3 The scale step `G_COEF_GLOBAL` and the sweep (D-40)

`GLScaleStep` is the `GlobalStep` of every GL coefficient prior (`CoefPrior.global_steps()` returns it; P2 §3.8 item 4
orders it). Phase `post`; `reads = {eq.mean, glob.mean, eq.coef, glob.coef}`; `writes = {eq.coef, glob.coef}`;
`marginal_of_indicators = True` (it never reads `eq.vol`). It takes `ℓ` from `eq.mean.dev_log` (structural) or
`glob.mean.dev_log` (reduced) and runs the family's lines of §3.2 **in the order printed there**: every local
parameter (all `N` coordinates at once, elementwise), then the group-level lines (segment sums over `G` static
groups). Sweeps:

```text
structural:  equation   (δ_i, a_i) | v, λ_eff,i  (deviation path, §3.4)    Block.EQ_MEAN    writes eq.mean, eq.struct.a
             residual · vol (P2 §4.4, unchanged)
             post       gl_scales  the §3.2 lines in order                    Block.G_COEF_GLOBAL   writes eq.coef, glob.coef
reduced:     pre        cccm       Π | A, λ_eff, v  (deviation path)         Block.G_CCCM     writes glob.mean
             contemp · residual · vol (unchanged)
             post       gl_scales  (as above, ℓ from glob.mean.dev_log)
```

Why it targets the posterior: the scale parameters enter only the coefficient prior, so given the coefficients they
are independent of the data, the volatility and the indicators; each line is an exact full conditional or, for the
collapsed lines (`a_g` with `ξ_g` integrated, then `ξ_g | a_g`; `(θ̃_j, ϱ_j)`), an exact block draw in the stated
order; the mean step conditions on the current scales through `moments_eq`. Plan invariants (P2 §4.8) hold: post steps
write only `glob.*` and `eq.coef.*`, never a residual input. Density-ratio identities (Tier U) gate every line.

`moments_eq` returns `GaussianLogDiag(mean0, log_prec, cmask)` with `log_prec_j = −log r_j − log v_j` on GL slots
(`log v_j` from the state: `log ϑ + log ζ`, `log ϖ`, `log ϱ + log ϖ − log 2`, `log ϱ + 2 log θ̃`; SSVS
`−2 log τ_{γ_j j}`) and `log prec0` elsewhere.

### 3.4 The deviation draw (D-41)

Shrunk deviations can be far below the resolution of `m + d` (an own-lag coefficient with `m = 1`) and below the
smallest float (a coefficient whose prior s.d. is `e^{−800}`); the scale step needs `log|d|` in both cases, and a
precision of `e^{1600}` overflows. The mean kernels therefore draw in deviation coordinates with the Jacobi scaling
done in log space (static `MeanStatic.coef_log`; `linalg.precision_draw_log`), with `c` the active-slot mask:

```text
K = diag(P) + G,   log P given (log_prec),   G = (c ⊗ c) ⊙ Gram ≥ 0 the masked data Gram (P2 §2.3)
b̃ = c ⊙ data right-hand side in deviation coordinates
      structural:  b̃ = c ⊙ F_θᵀ(w ⊙ (target − F_θ meanθ));   CCCM:  b̃_j = Xᵀ(G_j + W_j ⊙ (f_j − X m_j))
log s_j = −½ logaddexp(log P_j, log G_jj)                       never overflows
A = I + offdiag(exp(log s_i + log s_j) G_ij),  L = chol(A)       unit diagonal, as D-14's sKs
u = L⁻ᵀ(L⁻¹(s ⊙ b̃) + z),   d = exp(log s) ⊙ u,   dev_log = log s + log|u|,   coefficient = mean + d
```

This is D-14's draw (`θ = s ⊙ L⁻ᵀ(L⁻¹(s ⊙ b) + z)`, `b = P ⊙ mean + data`) rewritten with `b̃ = b − K·mean`, so it is
the same function of `z` to rounding wherever D-14 is finite (Tier U with P2 §9.3 item 1's conditioning-aware rule,
path-vs-path `2e-12 + 5εκ(1 + t)`). `d` may underflow to 0 harmlessly (its contribution to every residual is below
rounding); `dev_log` stays exact, so no scale conditional ever sees `χ = 0`. With `coef_log = False` the Phase 2 path
runs (bitwise unchanged). `StepInfo.ok` as D-14. The structural fused draw applies it to the whole `θ_i = [δ_i | a_i]`
(fixed prior precisions on the non-GL coordinates), the CCCM step to each `π_j`. GL compositions carry no SOC/SUR
rows (D-36), so `K` has no dummy term.

### 3.5 Sampling the GIG on the log scale (D-42; `random.log_gig`)

`log_gig(key, lam, log_psi, log_chi) -> (log x, ok)`, one problem, batched by `vmap` (the GIG's own parameters):

1. `log χ = −∞`, `λ > 0`: `log x = LG(λ) − log(ψ/2)` (gamma). `log ψ = −∞`, `λ < 0`: `log x = log(χ/2) − LG(−λ)`
   (inverse gamma). Any other infinite input is improper and returns `ok = False` (a status code, P2 D-18); by §3.4 it
   never occurs in a production sweep.
2. Reduction: `x = η y`, `log η = ½(log χ − log ψ)`, `log ω = ½(log ψ + log χ)`; `y` has density
   `∝ y^{λ−1} exp(−ω(y + 1/y)/2)`. For `λ < 0` sample with `|λ|` and return `−log y`.
3. Devroye (2014) §6 for `λ ≥ 0`, `ω > 0`: `log y = asinh(λ/ω) + X` (`asinh` taken as `log 2 + log λ − log ω` when
   `log λ − log ω > log 1e8`), with `X` drawn from the density `∝ exp(f(x))`,
   `f(x) = −α(cosh x − 1) − λ(eˣ − x − 1)`, `f'(x) = −α sinh x − λ(eˣ − 1)`,
   `α = √(ω² + λ²) − λ`, computed as `log α = 2 log ω − log(√(ω² + λ²) + λ)` (no cancellation).
   - Points: if `−f(1) ∈ [½, 2]` then `t = 1`; if `−f(1) > 2` then `t = √(2/(α + λ))`; else `t = log(4/(α + 2λ))`.
     If `−f(−1) ∈ [½, 2]` then `s = 1`; if `−f(−1) > 2` then `s = √(4/(α cosh 1 + λ))`; else
     `s = min(1/λ, log(1 + 1/α + √(1/α² + 2/α)))`.
   - Hat: `(η_D, ζ, θ, ξ) = (−f(t), −f'(t), −f(−s), f'(−s))`, `p = 1/ξ`, `r = 1/ζ`, `t' = t − rη_D`, `s' = s − pθ`,
     `q = t' + s'`; flat on `[−s', t']`, `exp(−η_D − ζ(x − t))` right of `t'`, `exp(−θ + ξ(x + s))` left of `−s'`.
   - Draw `U, V, W`: centre with probability `q/(p + q + r)` (`X = −s' + qV`), right with `r/(p + q + r)`
     (`X = t' + r log(1/V)`), left otherwise (`X = −s' − p log(1/V)`); accept iff `log W ≤ f(X) − log hat(X)`.
   - Every exponential is evaluated in logs: `α(cosh x − 1) = exp(log α + log 2 + 2 log|sinh(x/2)|)` with
     `log sinh y = y + log1p(−e^{−2y}) − log 2`; `λ(eˣ − x − 1) = exp(log λ + log(eˣ − x − 1))` only when `λ > 0` (else
     the term is 0); a term that overflows makes `f = −∞` and rejects. No product of an underflowed and an overflowed
     factor is ever formed.
   - Theorem 2: at most 3.4597 expected iterations, uniformly in `(λ, ω)`. Attempt `t` uses `fold_in(key, t)`; a
     `lax.while_loop` runs until every lane accepts or 512 attempts pass (`ok = False`, probability ≤ (1 − 1/3.46)⁵¹²
     ≈ 1e-76 per call).
4. `log x = log η ± log y`.

Tier U (WP28): Kolmogorov–Smirnov against `scipy.stats.geninvgauss` (`α = 1e-3`, 10⁴ draws) on a grid that pins
`λ ∈ {0, ±1e-8, ±1e-3, ±0.5, ±1, ±10, ±50}`, every `a − ½` and `a − 1` of the GK grid's first, middle and last
points, and `½ − T_max/2` for `T_max ∈ {200, 600}` (the ScaledChi2 `σ_h²` step), crossed with `log ω ∈ {−745, −300,
−50, −12, −5, 0, 5, 12}`; `E log X` against the exact `log η + ∂_λ log K_λ(ω)` (mpmath, 50 digits) within 4 Monte
Carlo s.e.; the `χ → 0` and `ψ → 0` limits against gamma and inverse-gamma draws; lane invariance under `vmap`
(1e-15); determinism by key; mean and worst-case attempts reported; no `ok = False` on the grid.

### 3.6 Host builders (`priors/global_local_arrays.py`; WP31)

`gl_arrays(prior, scales, *, n, lags, n_det, form, data) -> (GLArrays, GLShared, n_groups)` per window (R9): the GL
mask (lag slots), group ids (D-38; a custom array checked against `(n, n·p)` here, E-13), `log r` (D-37, from the
window's ψ), the hyperprior arrays, and SSVS `log τ₀, log τ₁` (D-44). Semi-automatic `ŝ` (one QR per equation of
the window's design, the diagonal of `R⁻¹R⁻ᵀ` by triangular solves, no inverse):

```text
structural, equation i:  ŝ²_j = RSS_i/(T_eff − q_i − 2) · [(F_iᵀF_i)⁻¹]_jj     (flat-prior NIG posterior variance; needs T_eff > q_i + 2)
reduced, equation j:     ŝ²_{rj} = Ŝ_jj/(T_eff − k − n − 1) · [(XᵀX)⁻¹]_rr     (flat-prior normal–Wishart; needs T_eff > k + n + 1)
```

### 3.7 Initialisation, log prior, storage

- **Init** (`Stream.INIT`, P2 §5.4): `CoefPrior.init_global` runs first (§2.4) and sets the group scales so that
  `r_j v_j` equals the Mode I Minnesota variance at `λ = 0.2` (`log ζ_g = log 0.04` for HS, the matching rate for
  NG/R2D2, `a_g` at the grid point nearest 0.1, SSVS `p_g = 0.5`), each group log scale jittered by `N(0, 0.5²)` per
  chain; local scales start at 1 (HS `ν = 1`, DL `θ̃ = 2`, `ϱ = 2`), SSVS `γ = 1`. The mean's cold draw (P2 §5.4) then
  uses these moments and sets `dev_log` (finite from the first state on).
- **Log prior** (monitors and validation): `Σ_j log N(d_j; 0, r_j v_j)` plus the scales' log densities, from the full
  chain state; `CCCMStep.log_prior` keeps its fixed-moment form for `FixedDiagonal` only.
- **Storage**: the group log scales, `a_g` and SSVS `p_g` are monitor fields (D-52) of `fit.posterior["monitor"]`;
  `latent_moments/incl_prob (equation, coefficient)` from `rb_incl`; local scales are not stored (they are latent).

### 3.8 Labels and spec

`Horseshoe.label = "Horseshoe(structural, olcl-lagwise)"` plus every non-default setting (as `Minnesota.label`);
`axes["prior"]` is the label. Specs carry `"class": "Horseshoe"` etc. (absent `"class"` still means `Minnesota`), so
Phase 1 and Phase 2 hashes are unchanged (golden test). GL priors in structural form are not order-invariant: the R8
note of P2 §8.3 applies; `ordering.permute_model` permutes a custom `groups` array with the variables (WP33).

---

## 4. The asymmetric conjugate prior (Chan 2022) on the analytic engine (D-45; WP33)

### 4.1 Prior

`AsymmetricConjugate(lambda_, cross, contemp, alpha, psi, mean, var_const, soc, sur, soc_scope, hierarchy, hyper,
hyperpriors, mh)`: per equation `i`, in δ-coordinates with D3 centring (P2 §6.1 Mode II), `θ_i = [δ_i | a_i] | λ_i ~
N(μ₀, λ_i V₀)`, `λ_i ~ IG(1 + ι/2, ψ_i/2)`, with `V₀` the Mode II diagonal: own lag `l` `λ²(d−n−1)/(l^α ψ_i)`, lag `l`
of variable `m ≠ i` `λ² cross² (d−n−1)/(l^α ψ_m)`, deterministic `var_const`, `a_ij` `contemp/ψ_j`. At
`cross = contemp = 1` it is the Phase 1 NIW prior exactly (P2 A1). Equations are independent a priori and `|A| = 1`,
so the posterior factorises and is closed form (`engines/structural_nig.py`).

### 4.2 Engine, adapter and fit

`BVAR(Linear(AsymmetricConjugate(...)), Constant())` resolves to `engine == "analytic"` and `Constant("auto") → "iw"`.
The NIG code takes a `Minnesota` (`_Window`, `eb_structural`, `window_scales` reject other priors; `conjugate_omega`
reads `prior.cross`), so the fit maps the prior onto an internal `Minnesota(form="structural", lambda_, alpha, psi,
mean, cross, contemp, var_const, soc, sur, soc_scope, hierarchy="fixed")` (the adapter, `priors/scales.py`), which
never reaches the user or a spec. `StructuralNIGFit` (new, `engines/structural_nig.py`) has the interface of
`AnalyticFit`: `log_ml(hyper=None)`, `posterior(hyper=None)` (the per-equation `NIGPosterior`s), `draw_parameters(size,
rng)` (iid `(a, γ, λ)` → `B = (A⁻¹Γ)'`, `chol = A⁻¹Λ^{½}`), `forecast(horizon, draws, seed)` (`simulate_linear_var`
on iid draws; attrs as `AnalyticFit` plus `ordering` and `lag_coefs`, D-49), `irf`, `conditional_forecast` (§6),
`summary()`, `hyper`, `converged`, `warnings` (Phase 1 setting checks; the R8 note for `n ≥ 4`: `cross ≠ 1` is not
order-invariant, Chan Cor. 2). `structural/ordering.py` supports it: the prior mode builds the NIG closed form, and
`_posterior_summary` reads `StructuralNIGFit`. Backtests treat it as analytic (`uses_fit_seed = False` unless
`hierarchy="mh"`, as Phase 1).

### 4.3 Hyperparameters

- `"fixed"`: as given.
- `"eb"` (default; free set `hyper="auto"` = `("lambda", "cross")` plus `soc`/`sur` when on): maximise
  `Σ_i log p(y_i | h) + log p(h)` with `eb_structural`, whose free names gain `cross` and `contemp` (its `_Window`
  takes them per evaluation). `lambda`, `soc`, `sur` keep the GLP hyperpriors; `cross` and `contemp` get a flat prior
  on their log over `[log 1e-3, log 10]` (marginal-likelihood maximisation, as Chan 2022 §4), stated in the label.
  Start: the Phase 1 EB mode at `cross = 1` (P2 §6.5 step 2). `alpha` free when listed, as Phase 1. `psi` is never free
  when `cross` is free or `≠ 1` (E-15).
- `"mh"`: random-walk Metropolis on the log free hyperparameters with Phase 1's `_metropolis`, through a small
  `_Free`-compatible problem class over the NIG log posterior (the same flat prior on `log cross` over the box), the
  proposal from the inverse Hessian at the EB mode, `MHSettings` and their convergence rule; then iid parameter draws
  per kept MH draw, as Phase 1's `hierarchy="mh"`.

### 4.4 What is rejected (construction time)

Any volatility other than `Constant("auto")`/`Constant("iw")`; `outliers="lenza-primiceri"` (GLS rows would keep the
closed form, but it is not in v0.3.0); `psi` free with `cross` free or `≠ 1`. Messages in §7.3.

---

## 5. `CommonSV` (D-46 … D-48; WP32)

### 5.1 Model and priors

`u_t = y_t − Π'x_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `h_0 ~ N(0, σ_h²/(1 − ρ²))` at the first data row,
`E h_t = 0` (N08 §A1–A2; this section's dates are 0-based: `h_t` belongs to data row `t`). In recursive form
`Σ = A⁻¹ΛA⁻ᵀ` and `λ_{i,t} = λ_i e^{h_t}`. Given the coefficients, the data enter the `h` conditional through
`q_t = Σ_i e_{it}²/λ_i` only (N08 §A4). Spec (`components/specs.py`, pure Python):

```python
@dataclass(frozen=True)
class CommonSV:
    rho_prior: TruncatedNormal | Beta = TruncatedNormal(0.9, 0.04)   # ρ ~ N(0.9, 0.2²)·1(|ρ|<1); Beta is on (ρ+1)/2
    sigma2_prior: InvGamma | ScaledChi2 = InvGamma(5.0, 0.04)       # σ_h²: E = 0.01 (Chan 2020)
    block: int = 100                                                # h-step MH block length (D-48)
```

`name = "CommonSV"`, `label` adds non-default settings; `TruncatedNormal(mean, var, low=-1.0, high=1.0)` joins
`rb.sv_priors`. CommonSV is a stochastic-volatility composition for D-35 (D-51).

### 5.2 The per-equation program (every `Linear × CommonSV()`)

`VolStatic(persistent="constant", common=True)`: each equation keeps its Constant core, `uses_global_vol = True`,
with prior `λ_i ~ IG(1 + ι/2, ψ_i/2)` (Mode I; coupled under Mode II) and the conditional

```text
λ_i | e_i, h, cpl ~ IG(shape_i + T_eff/2 + n_active/2,  scale_i + ½ Σ_t m_t exp(log e²_{it} − h_t) + quad)
```

`log_var_eq(vs, gvol) = log λ_i + h_t`, so `constant_weights()` is false and a Mode I or reduced mean takes the
weighted path (P2 §3.7.2). The post step `CommonSVStep` (`G_COMMON_SV`, `reads = {eq.vol, glob.vol}`, `writes =
{glob.vol}`) forms `log q_t = logsumexp_i(log e²_{it} − log λ_i)` from `SweepCtx.eq_out` and the cores and runs §5.4;
the vol block returns it from `global_steps()`. Order within a sweep: means and `a` given `λ_eff` → `λ_i | e_i, h` →
`h, ρ, σ_h | q`, each an exact conditional. GL priors compose unchanged (their scale step and this one are independent
post steps, ordered by block id). Triangular, so not order-invariant except at Mode II with `cross = contemp = 1`
(§5.3).

### 5.3 The conjugate prior: Mode II with the shared-factor path (`Minnesota(form="conjugate") × CommonSV()`)

The composition resolves to structural Mode II at `cross = contemp = 1` with coupled common cores: the NIW prior
`Σ ~ IW(diag ψ, n + 2)`, `B | Σ ~ MN(M, Σ ⊗ Ω)` exactly (P2 §6.1, N05 §4.2; with SOC/SUR rows as Phase 1), with
`Σ_t = e^{h_t}Σ`. For equation `i`, given `λ_i` and `h` (δ-coordinates, P2 §2.3, `θ_i = [δ_i | a_i]`):

```text
θ_i | λ_i, h ~ N(θ̂_i, λ_i K̃_i⁻¹),   K̃_i = P̃_i + F_iᵀ W F_i + F_dᵀF_d,   θ̂_i = K̃_i⁻¹(P̃_i μ_i + F_iᵀ W y_i + F_dᵀ y_{d,i}),
W = diag(m_t e^{−h_t})
```

At `cross = contemp = 1` the λ-free prior precision `P̃_i` is the same diagonal for every equation (Ω's rows are equal
bit for bit, `priors/scales.conjugate_omega`; the λ-scaled `a`-precision is `ψ_j/κ_a`), `F_i` is the active
(leading) columns of the shared design `F = [X | X M_c − Y_{:, :n−1}]`, and `W` does not depend on `i`. So `K̃_i` is
the leading `q_i × q_i` block of one matrix `K̃ = P̃ + FᵀWF + F_dᵀF_d`, and the Cholesky factor of a leading block is
the leading block of the Cholesky factor. Per sweep (`linalg.shared_factor_*`):

```text
G   = (√w F)ᵀ(√w F) + F_dᵀF_d          one GEMM per chain, w = m_t e^{−h_t}; dummy rows λ-free and not weighted by h
s   = diag(P̃ + G)^{−½},  L = chol(s(P̃ + G)s)      ONE Jacobi-scaled Cholesky per chain and sweep
per equation i (vmapped; L and s unbatched):  L_i = L on the leading q_i block, identity elsewhere
    θ̂_i = s ⊙ L_i⁻ᵀ L_i⁻¹ (s ⊙ c_i ⊙ (P̃μ_i + Fᵀ(w ⊙ y_i) + F_dᵀy_{d,i}))
    θ_i = (θ̂_i + √λ_i · s ⊙ L_i⁻ᵀ z) ⊙ c_i,      coupling n_active = q_i, quad = ½(θ_i − μ_i)ᵀP̃(θ_i − μ_i) (active)
```

Cost `O(T q² + q³ + n q²)` per chain and sweep instead of the weighted path's `O(n q³)`: one factorisation for all
equations, the conjugate speed of D2 (§10). `resolve.py` selects `const_path="shared_factor"` for this composition and
checks on the arrays that `P̃` is equation-invariant on active slots (else a `ValueError` naming `cross`/`contemp`).
The other paths agree with it to the conditioning-aware rule on the same noise (Tier U), and A8 compares it with the
weighted path end to end. The coupled common core draws `λ_i | θ_i, e_i, h` (§5.2), so the sampler is a two-block Gibbs
on `(θ_i, λ_i)`; the lag-1 autocorrelation of `λ_i` is about `q_i/(T_eff + q_i + 2·shape_i)` (*derived here*), which
WP35 measures (Q32 records the collapsed `λ_i` draw as a later improvement). The composition is order-invariant (the
NIW prior and a common scalar volatility), so no R8 note (`monitor.is_order_invariant` learns Mode II at
`cross = contemp = 1`, with `Constant` or `CommonSV`). `hierarchy="eb"` plugs in the Phase 1 EB mode of the conjugate
twin at constant volatility (the D-12 rule; a documented plug-in approximation under CommonSV); `"mh"` is E-5.

### 5.4 The `h, ρ, σ_h` kernel (shared; N08 §A5.2–A5.3)

Indices are 0-based; real states are `h_0 … h_{T−1}` with `T = T_eff`; padded states `h_t, t ≥ T` are held at exactly 0
and enter no sum.

```text
ℓ(h) = −(n/2) Σ_{t<T} h_t − ½ Σ_{t<T} exp(log q_t − h_t) − ½ hᵀK_h h,
K_h = σ_h⁻² · tridiag(diag = [1, 1+ρ², …, 1+ρ², 1], off = −ρ) on t < T (last diagonal 1/σ_h² at index T − 1)
blocks: offset o ~ U{0, …, block−1} (Sub.CSV_OFFSET);  B_0 = [0, o) ∩ [0, T),  B_b = [o + (b−1)·block, o + b·block) ∩ [0, T)
        for b = 1 … ⌈T_pad/block⌉ (static count); every real date lies in exactly one block every sweep; empty blocks are no-ops
scan b = 0, 1, … sequentially (lax.scan): block B conditions on the CURRENT h outside it (already-updated left neighbour)
    Newton from h⁰_t = log q_t − log n on B (depends on q, the neighbours and (ρ, σ_h) only), damped (halving ≤ 30),
    ≤ 50 iterations, stop when ‖Δ‖∞ < 1e-10; Hessian G_B = K_BB + diag(½ q e^{−h}) (tridiagonal, SPD)
    h* ~ N(ĥ_B, G_B⁻¹) by tridiag_precision_sample (Sub.CSV_H, b);  log α = ℓ(h*) − ℓ(h) + ½‖L_Bᵀ(h* − ĥ)‖² − ½‖L_Bᵀ(h − ĥ)‖²
    accept iff log U (Sub.CSV_H_ACC, b) < log α;  acc_h averages over non-empty blocks
ρ | h, σ_h:  K_ρ = Σ_{t=1}^{T−1} h_{t−1}²/σ_h² (+ 1/V_ρ for TruncatedNormal),  ρ̂ = K_ρ⁻¹(Σ_{t=1}^{T−1} h_{t−1}h_t/σ_h² (+ ρ₀/V_ρ))
             proposal ρ* ~ TN(ρ̂, K_ρ⁻¹; −1, 1), or U(−1, 1) when K_ρ = 0 (Beta prior, h ≡ 0; the proposal depends on h only)
             log α = g(ρ*) − g(ρ) (+ the Beta log-density ratio, and for U(−1, 1) the Gaussian factor's ratio),
             g(ρ) = ½ log(1 − ρ²) − (1 − ρ²) h_0²/(2σ_h²)
σ_h² | h, ρ: InvGamma(c₀, C₀): IG(c₀ + T/2, C₀ + ½S);  ScaledChi2(B): GIG(½ − T/2, 1/B, S),
             S = (1 − ρ²)h_0² + Σ_{t=1}^{T−1} (h_t − ρh_{t−1})²
```

The proposal of each block depends only on `(q, ρ, σ_h, h outside the block)`, so it is an independence proposal
whatever Newton's accuracy (N08 §A5.2); the random offset is an auxiliary choice of kernel independent of the state;
the sequential scan is a systematic-scan Gibbs over blocks. Diagnostics: `acc_h`, `acc_rho`, `acc_newton` (Newton
iterations that hit the cap; the warning of §8 if > 0, the draw stays exact). There is no mixture indicator, so the DNP
rule (P2 §4.6 item 4) does not arise.

### 5.5 Forecasting, warm starts, initialisation, monitors, storage

- **Forecast**: `CommonSVStep.future(key, gd, H)` draws `h_{T+j} = ρ h_{T+j−1} + σ_h η_j` from `h_T = h[T_eff−1]`
  (P2 §3.6 hook); the Constant cores add it (`log λ_i + h_{T+j}`) through the future mapping `{Block: future}` (§2.4).
  `lag_coefs` as for every linear mean. `forecast.py`'s `_pool` and `_log_lam` read the common `h` (`h_T` from the
  `CommonSVDraw`, other dates from stored paths), so `draw_parameters`, IRFs at a date and FEVD include `e^{h}`.
- **Warm start** (P2 §5.5): `glob.vol.h` is remapped by period like `eq.vol` `h` (dropped periods removed, new trailing
  ones simulated forward with the copied `(ρ, σ_h)`), with the 0-based layout of §2.2; `λ_i` copied.
- **Init**: `λ_i` from the Phase 2 Constant init; `ρ` = prior centre (clipped to `(−0.99, 0.99)`), `σ_h²` = prior mean,
  `log σ_h` jittered by `N(0, 0.25²)` per chain; `h` = the Newton mode of the initial `q` (so `K_ρ > 0` almost surely).
- **Monitors** (D-52): primary `h_T`, `ρ`, `σ_h` and every `log λ_i` (with the log-likelihood and own first lags);
  secondary `h_0`, `h_mid`. **Storage**: `latent_moments/common_h_mean, common_h_sd` from `rb_ch_*`,
  `posterior/common_h_T, rho, sigma`.

---

## 6. Conditional forecasts (D-50; WP34)

### 6.1 API

```python
fit.conditional_forecast(horizon, conditions, draws=1000, seed=None) -> ConditionalForecastResult
rb.conditional_forecast(fit, horizon, conditions, draws=1000, seed=None, base=None)  # base: an unconditional ForecastResult
```

`conditions`: a `pandas.DataFrame` indexed by horizon `1..H` (ints) or by the forecast periods, columns a subset of the
variables, `NaN` = free, values on the model's (transformed) scale; a mapping `{variable: sequence}` is accepted (`NaN`
padding). Available on `AnalyticFit`, `StructuralNIGFit` and `MCMCFit` with a linear mean (any volatility, including
CommonSV). Errors (E-16): unknown variable; horizon outside `1..H`; no condition; a non-finite condition; every cell
conditioned; a forecast without `lag_coefs` (a non-linear mean, a benchmark model, or an in-loop forecast whose draws
are not all stored); `R₁` with a diagonal below `1e-12·max|R₁|` (near-dependent conditions; never jittered).

### 6.2 The exact update

For each `(chain, draw)` slice of an unconditional forecast with the same `horizon`, `draws` and `seed` (or of `base`):
`u_k = paths_k − cond_mean_k`, `ε_k = L_k⁻¹u_k` (`L_k = cond_chol_k`, triangular solve), `Φ_0 = I`,
`Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}` from `lag_coefs`, the zero-shock path `μ_h = cond_mean_h − Σ_{k<h} Φ_{h−k}u_k`
(P2 D-34), and `M` (`nH × nH`, block lower triangular, `M_{h,k} = Φ_{h−k}L_k`), so that `Y = μ + Mε` exactly (tested).
With `R` the `m × nH` selection of conditioned cells and `r` their values:

```text
QR: (RM)ᵀ = Q₁R₁ (R₁ m × m upper, positive diagonal; full row rank because every L_k has a positive diagonal)
Y_c  = Y + M Q₁ R₁⁻ᵀ (r − RY)                       one exact draw from Y | θ_s, λ_s, RY = r
E_c  = μ + M Q₁ R₁⁻ᵀ (r − Rμ)                        conditional mean
Cov_c,h = N_hᵀN_h,  N_h = (I − Q₁Q₁ᵀ) M_hᵀ           M_h: the n rows of horizon h
L^c_h: QR of N_h[:, free_h] (the columns of the free variables of horizon h), Rᵀ embedded in an n × n zero matrix
log w_s = −(m/2) log 2π − Σ log R₁,ii − ½ ‖R₁⁻ᵀ(r − Rμ)‖²     log N(r; Rμ_s, R₁ᵀR₁), stored
```

The conditioned rows and columns of `Cov_c,h` are exactly zero, so factoring the free columns only is exact; the
conditioned cells of `Y_c` are set to `r` exactly (rounding). `paths_level` is the pathwise inverse transform of `Y_c`
(SPEC §5.2). No random number is drawn: the result is a deterministic function of the unconditional draws, so the same
seed reproduces it. The update function takes a keyword-only `_update` (test seam for M33).

### 6.3 `ConditionalForecastResult` (a scenario, not a forecast to score)

A subclass of `ForecastResult` (R3: same fields, inherited `quantiles`, `mean`, `to_dataframe`). Its dataset adds
`condition (horizon, variable)` (`NaN` = free), `conditioned (horizon, variable)` (bool) and `log_weight (chain,
draw)`; it **drops** the accelerator variables `cond_A`, `cond_scale`, `cond_scale_persistent`, `log_p`, `log1m_p`,
`nu` and `lag_coefs` (they describe the unconditional factors); `attrs["conditional"] = "hard-per-draw-v1"`, the base
forecast's seed, and the ESS of the stored weights. `cond_mean`/`cond_chol` are `E_c` and `L^c`: the moments of
`y_{T+h}` given the draw, its scale path and the conditions, integrated over the other shocks (not the one-step moments
of SPEC §7, which condition on the path). The equal-weight mixture over draws is the per-draw scenario density, not
`p(y_free | data, RY = r)` (that needs the weights `w_s`; Q25). `log_pred_density` therefore raises `ValueError` for
a conditioned `(horizon, variable)` and for `lpd_method="integrated"`, and documents what it computes for free
variables; `evaluation.scores.score_forecast` and the backtest refuse a `ConditionalForecastResult`.
`forecast/invariants.check_conditional_structure` replaces `check_structure` for it: conditioned paths equal `r`
exactly, conditioned rows and columns of `cond_chol` are exactly 0, the free block is lower triangular with a positive
diagonal; simulation consistency (KS, `α = 1e-3`) of the standardised free residuals per horizon.

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
rb.DiscreteGrid(support, weights)                                     # tuples, normalised; DiscreteGrid.gruber_kastner()
rb.AsymmetricConjugate(lambda_=0.2, cross=0.5, contemp=1.0, alpha=2.0, psi="ar", mean=None, var_const=1e7,
                       soc=None, sur=None, soc_scope="persistent", hierarchy="eb", hyper="auto",
                       hyperpriors=GLPHyperpriors(), mh=MHSettings())
rb.CommonSV(rho_prior=sv_priors.TruncatedNormal(0.9, 0.04), sigma2_prior=sv_priors.InvGamma(5.0, 0.04), block=100)
rb.conditional_forecast(fit, horizon, conditions, draws=1000, seed=None, base=None)       # lazy import (WP34)
```

`**common` are the fields of `Horseshoe` after `global_scale`. `Linear(prior=...)` accepts `Minnesota`, any GL prior or
`AsymmetricConjugate`; `BVAR(vol=...)` accepts `Constant`, `SV` or `CommonSV`.

### 7.2 Resolution and engine table (`BVAR.__post_init__`; extends P2 §11.1)

`auto` values first (stored resolved, D-19): `Minnesota(form="auto")` → `"conjugate"` iff `vol` is `Constant` with prior
`"auto"`/`"iw"` (unchanged; with `CommonSV` it resolves to `"structural"`, D3); a GL prior's `form="auto"` →
`"structural"`; `Constant("auto")` → `"iw"` iff the form is conjugate or the prior is `AsymmetricConjugate`, else `"ig"`.
Then the table (first match wins):

| # | Resolved composition | Engine / path |
| :- | :--- | :--- |
| 1 | conjugate Minnesota × `Constant("iw")` | `analytic` (Phase 1, unchanged) |
| 2 | conjugate Minnesota × `CommonSV` | `mcmc`: structural Mode II at `cross = contemp = 1`, coupled common cores, `const_path="shared_factor"` (§5.3) |
| 3 | conjugate Minnesota × anything else | `ValueError` (E-1) |
| 4–9 | structural / reduced Minnesota × `Constant(…)`, `SV(…)` | P2 §11.1 rows 3–9, unchanged |
| 10 | structural or reduced Minnesota × `CommonSV` | `mcmc`, per-equation program, Mode I (§5.2) |
| 11 | GL × `Constant("ig")`, `Constant("iw_equivalent")`, `SV(…)`, `CommonSV` | `mcmc` (structural or reduced; §3) |
| 12 | GL × `Constant("iw")` | `ValueError` (E-2) |
| 13 | `AsymmetricConjugate` × `Constant("iw")` | `analytic`, `StructuralNIGFit` (§4) |
| 14 | `AsymmetricConjugate` × anything else | `ValueError` (E-3) |

MCMC-only rules after the table: `outliers="lenza-primiceri"` → E-6 (with `CommonSV`: E-7); `hierarchy="mh"` on a
Minnesota MCMC composition → E-5; the JAX-free `config.validate_composition` with `coef_log` (§2.1). `uses_fit_seed`
and `supports_warm_start` are `engine == "mcmc"`. `MCMCSettings.with_default_draws` and `api.default_mcmc` count
`CommonSV` as SV (D-51).

### 7.3 Error messages (raised at construction unless marked; tested verbatim in `tests/unit/test_api_p3.py`)

| ID | Condition | Type | Message |
| :- | :--- | :--- | :--- |
| E-1 | conjugate Minnesota × `SV`/`Constant("ig"|"iw_equivalent")` | `ValueError` | "the conjugate Minnesota prior requires Constant(prior='iw') or CommonSV(); use form='structural' (the MCMC default) or 'reduced'" |
| E-2 | GL × `Constant("iw")` | `ValueError` | "Constant('iw') makes the coefficient prior λ-scaled (Mode II, the conjugate twin); global-local priors are independent priors: use Constant('ig'), Constant('iw_equivalent'), SV() or CommonSV()" |
| E-3 | `AsymmetricConjugate` × not `Constant("auto"|"iw")` | `ValueError` | "AsymmetricConjugate is conjugate only with constant volatility (Constant()); for stochastic volatility use Linear(prior=Minnesota(form='structural', cross=...)) with SV(), or Minnesota(form='conjugate') with CommonSV()" |
| E-4 | GL `form="conjugate"` | `ValueError` | "global-local priors are not conjugate; use form='structural' (default) or 'reduced'" |
| E-5 | `Minnesota(hierarchy="mh")` on the MCMC engine | `NotImplementedError` | "hierarchy='mh' is not available on the MCMC engine in this version (DESIGN Q15); use 'eb' or 'fixed', or the analytic engine (Minnesota(form='conjugate') with Constant())" |
| E-6 | Lenza–Primiceri on the MCMC engine | `NotImplementedError` | "Lenza-Primiceri scaling runs on the analytic engine only (DESIGN Q16); use SV(outliers=True) for outlier-robust MCMC models" (P2's double-treatment clause kept for `SV(outliers=True)`) |
| E-7 | Lenza–Primiceri with `CommonSV` | `ValueError` | "CommonSV already scales every variance by e^{h_t}; adding Lenza-Primiceri scaling is a double treatment" |
| E-8 | `AsymmetricConjugate` with `outliers="lenza-primiceri"` | `ValueError` | "Lenza-Primiceri scaling is not available with AsymmetricConjugate in this version; use Minnesota(form='conjugate')" |
| E-9 | GL `scale` not in {"psi", "none"}; `groups` an unknown string or not a rectangular non-negative integer array | `ValueError` | lists the allowed values |
| E-10 | NG/R2D2/DL `a ≤ 0` or a grid point outside `(0, 1]`; `b ≤ 0`; NG `c ≤ 0` and not `"a/2"`; HS `global_scale ≤ 0` | `ValueError` | names the parameter and its range |
| E-11 | SSVS `c0 ≤ 0`, `c1 ≤ c0`, `p ∉ (0, 1)` | `ValueError` | "SSVS needs 0 < c0 < c1 and p in (0, 1) or a Beta prior" |
| E-12 | SSVS `semiautomatic=True` with `T_eff ≤ q_i + 2` for some equation (structural) / `T_eff ≤ k + n + 1` (reduced) (**at fit**) | `ValueError` | "SSVS(semiautomatic=True) scales by the flat-prior posterior s.d., which needs more observations than coefficients (T = …, needed > …); use semiautomatic=False or fewer lags" |
| E-13 | custom `groups` shape ≠ `(n, n·p)` (**at fit**) | `ValueError` | "groups array has shape …; expected (n, n·lags) = (…)" |
| E-14 | `CommonSV.block < 2`; a `rho_prior` other than `TruncatedNormal`/`Beta`; a `sigma2_prior` other than `InvGamma`/`ScaledChi2` | `ValueError`/`TypeError` | names the field |
| E-15 | `AsymmetricConjugate` `cross`/`contemp`/`lambda_` ≤ 0; `psi` free while `cross` is free or `≠ 1` | `ValueError` | "empirical Bayes over psi is unavailable with cross-variable shrinkage (Chan 2022 marginal likelihood over lambda, cross, contemp and alpha only); remove 'psi' from hyper or fix psi" |
| E-16 | `conditional_forecast` input errors (§6.1) (**at call**) | `ValueError`/`KeyError` | names the variable or horizon and the valid range, or the missing `lag_coefs` and why |

Phase 2 test amendments (E52): `tests/unit/test_api_resolution.py` asserts the old E-1, E-5 and E-6 texts and is updated
to the new ones (WP27); `tests/integration/test_backtest_lpd_integrated.py` expects analytic forecasts to raise for the
integrated estimator at `h > 1` and switches that case to `OLSVAR`, which still carries no `lag_coefs` (WP33).

### 7.4 Spec, hash, names, axes

`BVAR.spec()` serialises the new priors with a `"class"` key; a `DiscreteGrid` as `{"class": "DiscreteGrid",
"support": [...], "weights": [...]}`, and the GK default as `{"class": "DiscreteGrid", "name": "gruber_kastner"}`;
`CommonSV` as `{"class": "CommonSV", …}` with nested priors through `priors.sv.prior_to_spec`. `from_spec` dispatches on
`"class"` (absent means `Minnesota` / `Constant`). Every Phase 1 and Phase 2 spec and hash is unchanged (golden test).
`BVAR.name` and `axes` follow P2 §11.4; a GL model's prior axis is its label; `axes["vol"] = "CommonSV"`. The
conjugate CommonSV composition keeps the conjugate Minnesota prior label (no `scaling=conjugate` suffix: its prior is
the NIW prior of the analytic arms), so `CommonSV` vs `Constant` is a one-axis comparison against the analytic
engine, and `CommonSV` vs `SV` likewise with Mode I priors.

### 7.5 Grid harmonisation (D-53)

`BVAR.harmonise_grid(combos)`: if any combination has an `SV` volatility, a `CommonSV` with an `auto` Minnesota form,
or a GL prior, every `Minnesota(form="auto")` becomes `"structural"` and its `Constant("auto")` becomes `"ig"`.
`AsymmetricConjugate` and GL arms are never rewritten. Explicit settings are never changed; rewrites are listed in the
info note (P2 §11.4).

---

## 8. MCMC budgets and the convergence gate (D-51, D-52)

- **Budget.** `MCMCSettings.with_default_draws(sv=…)` (D-35) treats `CommonSV` as stochastic volatility, and so do
  `resolve._sv_budget`, `fit_mcmc`'s default draws and `api.default_mcmc`: 4 × 10,000 kept draws by default, and the
  D-35 warning below 10,000 per chain, its message naming CommonSV. GL compositions take the D-35 default of their
  volatility until the owner sets D-58 from WP35. WP35 measures, at `n = 7, p = 4, T = 200` (FRED-QD medium set) and at
  the BENCH-P2 shape, the bulk ESS per 1,000 sweeps of every primary quantity of every family × {Constant, SV} and of
  CommonSV's `h_T`, `ρ`, `σ_h`, `λ_i` (including the shared-factor path's two-block autocorrelation), and recommends the
  smallest budget at which the median fit passes the gate. No threshold of P2 §8.2 changes.
- **Gate.** P2 §8.2 unchanged. Primary set additions (D-52): GL group log scales (when `G ≤ max(2p, n)`), SSVS `p_g`
  under a Beta hyperprior; CommonSV `h_T`, `ρ`, `σ_h`.
- **Warnings** (P2 §8.3 additions): GL group scales failing R-hat or ESS (named); a hyperprior index at a grid end for
  more than half the draws ("the grid may be too narrow"); CommonSV `h` block acceptance below 0.1 ("decrease
  CommonSV(block=…)") and any Newton cap hit (informational; the draw is exact); `AsymmetricConjugate` EB or MH
  non-convergence (as Phase 1).

---

## 9. Validation plan (R6)

Every new block is tested in isolation (Tier U, B), every new composition class end to end (G, S), every curated bug
is caught (M), and every comparison with "the truth" uses a truth written by someone else (P2 §9.1, D-22). WP30
(references) and WP36 (composition suites) are written by an author who wrote no Phase 3 kernel, from this document
only; the PR records the authorship. Every tolerance below is fixed now (D-60).

### 9.1 References (`validation/reference.py`, `simulate.py`, `log_joint.py`, `testfns.py`, `joint.py`)

New registered kinds: `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs` (prior samplers of the scales and
deviations in log space, log densities of §3.2 written from the prior definitions, not from the conditionals);
`constant_common` (Constant cores with the common `h`, Mode I or coupled) and `common_sv` (stationary AR(1) `h`, `ρ` and
`σ_h²` priors). `FixedDesign` gains `exogenous: bool` (D-56: a ones column per deterministic term, `N(0, 1)` elsewhere,
drawn once from the configuration seed; `Y` simulated given `θ` without lag feedback). `simulate_obs` gains the
common-`h` path. The Phase 3 rows and blocks live in `COMPOSITIONS_P3` and `BLOCK_CONFIGS_P3` (D-61).

### 9.2 Tier U (every PR; WP28, WP31–WP34)

1. **Numerics**: `log_gig` (§3.5 list); `precision_draw_log` against D-14's `precision_draw` on the same `z`
   (path-vs-path `2e-12 + 5εκ(1 + t)`, P2 §9.3 item 1) and against a dense oracle that takes log precisions up to
   `±1600` (mpmath, 1e-10 relative on `dev_log`); `dev_log` equal to `log|θ − mean|` wherever that is representable
   (1e-12); the shared-factor draw against the weighted path on the same noise (conditioning-aware rule); the
   tridiagonal Newton (gradient below 1e-8 at the returned mode; the proposal unchanged when the current block values
   change).
2. **Density-ratio identities** (P2 §9.3 item 2, relative 1e-8) at 20 random states for every line of §3.2 (local,
   group, collapsed `a` and then `ξ`, SSVS `γ`, `p`), the deviation draw, each `h` block, the `ρ` and `σ_h²` steps, and
   the coupled common core, against the WP30 `log_joint`; MH steps: the computed log acceptance ratio equals target and
   proposal densities (1e-10).
3. **Closed forms**: collapsed `a` weights against numerical integration over `ξ` with `c` fixed and tied (1e-9); DL
   `θ̃ | d` against quadrature of the joint (1e-8); SSVS log odds against brute force (1e-12); `q_t e^{−h_t} | h ~ χ²_n`
   at the data-generating `θ` (simulator check, KS); the conditional-forecast update against the exact oracle (A10).
4. **Infrastructure**: the golden templates and hashes (§2); every Phase 2 composition bitwise unchanged on a short run
   before/after; float64 everywhere; plan invariants (P2 §4.8) with the write-set tree diff for every new composition;
   zero recompiles across windows of one bucket.

### 9.3 Tier B: block configurations (`BLOCK_CONFIGS_P3`, ids 401–408)

Per PR as block Geweke tests in the kernel suites (`tests/mcmc/blocks/`); G-small of the configurations in the weekly
tier (P2 §9.10 amendment).

| Id | Config | Live | Held | Exact |
| :- | :--- | :--- | :--- | :--- |
| 401–405 | `gl_scales_only_<family>` | local, group and hyper parameters | coefficients drawn from `p(d | current scales)` each iteration; data irrelevant | yes |
| 406 | `gl_mean_only` | coefficients (deviation path) | GL scales (prior draws), `λ` known | yes |
| 407 | `common_sv_only` | `h, ρ, σ_h` | coefficients and `λ` known; `q` from the simulator | yes |
| 408 | `conjugate_niw_only` | `(θ_i, λ_i)` by the shared-factor path and the coupled common core | `h` known | yes |

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
| 316 | reduced × `SV` × NG (olcl-lagwise), tight scales | lagged | 3 | weekly |
| 317 | structural × `SV(outliers=True)` × HS (equation-wise) | exogenous | 3 | weekly |
| 318 | structural × `Constant("iw_equivalent")` × HS | exogenous | 3 | weekly |
| 319 | structural × `SV` × NG (olcl-lagwise), tight scales | lagged | 3 | weekly |
| 320 | conjugate Minnesota × `CommonSV`, shared-factor path, with SUR dummy rows | lagged | 3 | PR + weekly |
| 321 | structural Mode I × `CommonSV` | lagged | 2 / 3 | PR + weekly |
| 322 | reduced × `CommonSV` | lagged | 3 | weekly |
| 323 | as 320 with the weighted path forced (engine level; A8's twin) | lagged | 3 | weekly |
| 324 | structural × `CommonSV` × HS | exogenous | 3 | weekly |
| 325 | conjugate × `CommonSV` with `ScaledChi2` `σ_h²` and `Beta` ρ (parity options) | lagged | 3 | weekly |
| 326 | structural × `Constant("ig")` × NG (grid `a`, fixed `c = 1`): M22's catcher | exogenous | 3 | weekly |
| 327 | reduced × `Constant("ig")` × HS (global) | exogenous | 3 | weekly |
| 328 | reduced × `CommonSV` × NG (fixed) | exogenous | 3 | weekly |
| 329 | conjugate × `CommonSV` with `block = 7` (interior blocks with two neighbours) | lagged | 3 | weekly |
| 330 | structural × `SV` × SSVS (olcl-lagwise, `semiautomatic=False`), tight slab | lagged | 3 | weekly |
| blocks | 401–408 (§9.3) | — | — | weekly (G-small) |

**Coverage argument** (the R6 criterion read as "every new block and every new interaction class"). The GL step reads
only coefficients; a GL prior meets the volatility only through the weights of the mean draw. Rows 301–330 cover that
interaction with constant weights (301, 303–310, 326, 327), coupled `a` (318), time-varying SV weights (302, 311–316,
319, 330), transitory weights (317) and common weights (324, 328), in both forms; the volatility blocks themselves are
the Phase 2-validated ones. Compositions not listed (e.g. GL × `SV(tails="t")`, × `SV(rw=True)`) are covered by this
argument, not by a row of their own; the exit-criteria map says so.

**Test priors** (proper and tight, P2 §9.1): HS `global_scale = 0.1` (exogenous rows only, D-56); NG exogenous `a = 0.5,
b = 2, c = 1`, lagged rows `a = 0.5, b = 10, c = 0.05` (`E v ≈ 0.011`); R2D2 `a = 0.5, b = 2`; DL `a = 0.5`; grids of
five points `{0.1, 0.25, 0.5, 0.75, 1}` with equal weights; SSVS `τ₀ = 0.01·√r`, `τ₁ = 0.5·√r` (lagged: `0.1·√r`),
`p = 0.3` or `Beta(2, 3)`; `ψ` fixed and unequal (so `r_j ≠ 1`, mutant M27); own-lag prior mean 0.4 (mutant M26); the
lagged rows' explosive prior mass measured by WP30 and ≤ 2e-4; everything else P2 §9.2. CommonSV: `ρ ~ TN(0.8, 0.15²)`,
`σ_h² ~ IG(5, 0.2)`; conjugate rows: NIW `Ψ = 0.5 I`, `d = n + 4`, own `0.1²`, cross `0.1²` (cross = 1), intercept 1.
`T_eff = 20 / 50` (never a multiple of 32). **Test functions**: bounded transforms of deviations (`arctan(d_j/√r_j)`,
`1{|d_j| < 0.1√r_j}`) at fixed coordinates, group log scales, the joint `a_g·log ξ_g` (M35), two local log scales,
`a`-grid indices, SSVS `N1_g` and `log p_g`, `h` at three dates and `mean h`, `ρ`, `σ_h`, `log λ_i`, the exact
log-likelihood, `W = Σ_t q_t e^{−h_t}`, log-scale data functions.

### 9.5 Tier S: SBC (weekly; P2 §9.6 configuration: `N = 1000, L = 99`, the ESS rules, γ test, Bonferroni `α = 1e-2`)

Every row of §9.4. Quantities: the test functions above, the total and per-equation log-likelihood, `log p(θ)`, `W`
(N08 §A11), and on lagged rows the predictive `y_{T+1}` through the production forecast core (exogenous rows have no
`x_{T+1}`, E49).

### 9.6 Tier M: mutants (weekly; each must fail some tier; injected through the P2 §4.9 seam, or the `_update` argument)

| Mutant | Injected through | Expected catcher |
| :--- | :--- | :--- |
| M19 HS `ζ_g` shape `n_g/2` (the `+1` dropped) | `gl_group_conditional` | U (density ratio), G |
| M20 HS `ϑ_j` scale with `τ_g` where `ζ_g` belongs (GK App. B as printed) | `gl_local_conditional` | U, G |
| M21 NG `ϖ_j` GIG with `λ = a − 1` | `gl_local_conditional` | U, G |
| M22 NG collapsed `a`-weight without `b log(2c/ã)` | `gl_a_conditional` | U (fixed `c`), G (row 326) |
| M23 R2D2 `ϱ_j` GIG with `χ = d²/(rϖ)` (factor 2 dropped) | `gl_local_conditional` | U, G |
| M24 DL `θ̃_j` GIG with `λ = a − ½` | `gl_local_conditional` | U, G |
| M25 SSVS log odds without `log τ₀ − log τ₁` | `ssvs_logodds` | U, G |
| M26 deviation about 0 instead of the prior mean | `gl_deviation` | G (own-lag mean 0.4) |
| M27 scale step with `r_j = 1` while the mean uses `r_j` | `gl_local_conditional` | U, G |
| M28 CommonSV Newton started at the current block, one iteration (state-dependent proposal) | `csv_newton_start` | A11 (quadrature, 10⁵ draws, KS `α = 1e-3`; required), G |
| M29 `q_t` with `λ_i` instead of `1/λ_i` | `csv_q` | U, G |
| M30 `ρ` acceptance without the stationary-start factor | `csv_rho_accept` | U, G |
| M31 coupled common core without the `n_active/2` shape term | `csv_lambda_conditional` | U, G (row 320), A11 |
| M32 shared-factor dummy rows weighted by `e^{−h_t}` | `shared_factor_rows` | U (against the weighted path), G (row 320) |
| M33 conditional update with `Mᵀ` where `M` belongs | `_update` | U (A10) |
| M34 GIG tails from secants instead of tangents (a hat below `f`) | `gig_sampler` | U (KS, `E log X`) |
| M35 `ξ_g | a_old` drawn before `a_g` (the reversed collapsed order, E30) | `gl_a_conditional` | G (joint test function `a_g·log ξ_g`, rows 304, 306) |

The suite records the catching tier of each mutant (`mutation.jsonl`); a mutant caught by none blocks the release.

### 9.7 Tier A: analytic twins and exact oracles

| Test | Oracle | Rule | Cadence |
| :- | :--- | :--- | :--- |
| A8 | conjugate × CommonSV: shared-factor path vs the weighted path (row 323's config), quantiles of `B`, `Σ`, `h_t`, `ρ`, `σ_h` | Phase 2 parity rule | PR (`n = 3, p = 2, T = 60`); weekly (`n = 20, p = 4, T = 300`) |
| A9 | `StructuralNIGFit` at `cross = contemp = 1` equals `AnalyticFit`: log ML and posterior moments of `B`, `Σ` (1e-10), EB `λ` (1e-6); EB with free `cross`: gradient of the objective below 1e-6 at the returned mode; MH: quantiles vs the posterior on a hyperparameter grid by quadrature; at `cross ≠ 1`: forecast quantiles vs MCMC Mode II | 1e-10 / 1e-6 / Phase 2 parity rule | PR; the MCMC leg weekly |
| A10 | conditional forecasts per draw against an independent stacked companion-form oracle with Schur-complement conditioning (SciPy): `E_c`, `Cov_c`, `Y_c` from the same `ε`; analytic (constant `Σ`), MCMC SV and CommonSV draws; a conditioned variable ordered before a free one; sample moments of conditional paths vs `E_c`, `Cov_c` | 1e-10; `|z| ≤ 4` Bonferroni | PR |
| A10b | BoE `bvar` `draw_constrained_forecasts` at fixed `(beta, sigma)` (MIT, CI) vs our per-draw update on the same parameter draw: conditional mean (point output) and the covariance of 10⁴ draws | 1e-10; `|z| ≤ 4` Bonferroni | PR |
| A11 | CommonSV closed forms (N08 §A11): the stationary distribution of block 408 (`h` fixed) vs the Phase 1 NIW posterior with `row_scales = e^{h/2}` (quantiles of `B`, `Σ`); `log p(Y | h)` = Phase 1 log ML − `(n/2)Σh`; the `h` posterior for `T ∈ {2, 3}`, `n = 5` by tensor quadrature vs MH draws (10⁵); proposal state independence | Phase 2 parity rule / 1e-10 / KS `α = 1e-3` | PR |
| A12 | GL deviation path with frozen finite scales equals the Phase 2 path on the same noise, structural and CCCM | conditioning-aware rule | PR |
| A13 | SSVS with `τ₀ = τ₁` (engine-level arrays) equals the Normal prior (`FixedDiagonal`): posterior quantiles | Phase 2 parity rule | PR |

### 9.8 Tier P: parity (fixtures by WP29; tests by WP38; pass/fail, D-59)

**`bayesianVARs` 0.1.8** (reduced form, through `fit_mcmc` with explicit arrays, P2 §9.9). **Probes first**, each a small
fixture whose result is recorded in `tests/parity/SV_PARAMETRIZATION.md` before any parity comparison runs; a probe
that cannot be decided leaves the runs that depend on it **not met** (D-59).

| Probe | Question | Decision rule |
| :- | :--- | :--- |
| P1–P5 | P2 §9.9 (`normal_sds` variance vs s.d.; `U` vs `U⁻¹` and sign; intercept row; `sv_para` layout; `cholesky_priorhomoscedastic` scale) | the bands already in `tools/make_r_fixtures.R` |
| P6 | internal standardisation | refit with one series × 100: `PHI` rescales equivariantly within the Phase 2 rule ⇒ none; otherwise both sides receive pre-standardised data |
| P7 | the grouping matrices for `"global"`, `"equation-wise"`, `"covariate-wise"`, `"olcl-lagwise"` | recorded as data (the prior object's grouping field); must equal D-38's for the runs that use them |
| P8 | the meaning of the GL hyperparameters (and `DL_a = "1/K"`) | both samplers on the same tiny-`T` data under each candidate mapping (candidates at least × 2 apart); the mapping that passes the Phase 2 rule while every other fails is adopted; none or several ⇒ unresolved |
| P9 | the SSVS semi-automatic `τ` | the prior object's `τ₀, τ₁` fields equal D-44's formula to 1e-8, else the runs use `semiautomatic=False` with explicit `τ` on both sides and the difference is recorded |
| P10 | the intercept is outside the shrinkage, `prior_intercept` is an s.d., and every `*_tol` accepts 0 | prior-only run: intercept quantiles vs `N(0, s_c²)` within the Phase 2 rule; the call with all tolerances 0 succeeds |

Parity runs (one parameter set per run, both sides derived from it, E11's lesson); data: P2's simulated parity sets
`tests/parity/fixtures/data/var_sv_n3.csv`, `var_sv_n7.csv` (also the SWA oracle's, so a three-way check), standardised
if P6 says so.

| Run | Composition (`groups`, scales) | Data |
| :--- | :--- | :--- |
| R1 | reduced × SV, Normal coefficient prior: the Phase 2 criterion (P2 §9.9 parameter set) | `n ∈ {3, 7}`, `p ∈ {2, 4}` |
| R2 | reduced × `Constant("ig")`, Normal prior | `n = 3, p = 2` |
| R3–R7 | reduced × SV × {HS, NG, R2D2, DL, SSVS} (`"global"`, fixed `a`, fixed `p`, `semiautomatic=False`, `scale="none"`) | `n ∈ {3, 7}`, `p = 2` |
| R8 | reduced × SV × HS (`"olcl-lagwise"`) | `n = 7, p = 4` |
| R9 | reduced × SV × NG with a discrete grid on `a` (`NG_b`, `NG_c` and the grid matrix stated in the fixture) | `n = 7, p = 2` |

Common settings: `cholesky_U_prior = "normal"` with `cholesky_normal_sds` from P1; the SV priors of P2 §6.3 with
`SV(offset=0.0, exact=False)` (bayesianVARs' default offset 0; `expert_sv_offset` is not passed); `prior_intercept =
s_c` ↔ `var_const = s_c²` (P10); `PHI_tol = cholesky_U_tol = DL_tol = R2D2_tol = NG_tol = 0` (P10), recorded in the
derived call. Draws: on the R side 4 independent `bvar()` runs with recorded seeds, each 50,000 kept draws after 10,000
burn-in, MCSE from the per-run ESS; on our side 4 chains × 50,000 (weekly) and 4 × 6,000 (PR short run, P2 §9.10).
Compared with the Phase 2 rule: 5/25/50/75/95% quantiles of `Π`, `A`, `h_t` (`t ∈ {1, T/4, T/2, T}`), `μ, φ, σ`, the
predictive at horizons 1–4, and, where `bayesianVARs` returns them with the same meaning, the SSVS inclusion
probabilities (`|z| ≤ 4` Bonferroni with binomial MCSE), the HS global scale (`C⁺(0, 1)` on both sides) and R9's
posterior of `a`. The tests fail, never skip, when a fixture is missing in the release run (D-59).

**`bvars` 1.1** (Woźniak; section `wbvars` of `make_r_fixtures.R`, fixtures `wbvars_*.nc`; the conjugate CommonSV of
§5.3). Probes, recording **data fields only**: **P11** the defaults of `specify_prior_bvar` (`A`, `S`, `nu`, `Psi`,
`Gamma`, `lambda`, `sv_a`, `sv_s`) and of `specify_starting_values_bvar` (`V`, `sigma2_omega`, `s_`, `rho`, `omega`, `h`,
`S`); **P12** whether `V` can be held fixed: prior draws of `V` under a concentrated matrix-GIG setting must have
`sd(V_ij) ≤ 1e-3·√(Ω_ii Ω_jj)`; **P13** the common-volatility normalisation (`E h = 0`, initial condition) under `"cSV"`
and `"ncSV"`; **P14** the `ρ` and `ω`/`σ_ω²` priors and whether a setting fixes `σ_ω²`; **P15** the `Σ` convention
(covariance vs scale), the prior mean (`stationary`) and the intercept; **P16** whether bvars's `h` sampler is exact:
its draws on a `T ∈ {2, 3}`, `n = 5` dataset against A11's quadrature posterior (KS `α = 1e-3`, `k = 4` seeded runs
pooled). `estimate(spec, S, thin)` runs one chain without a burn-in argument: each R run is a 10,000-draw `estimate`
followed by a 50,000-draw `estimate` restarted from its `last_draw`, `k = 4` runs with recorded seeds, MCSE from the
per-run ESS. Data: FRED-QD small set (`n = 3`, federal series), `p = 4`. **Gating comparison**, decided by the probes
and recorded before it runs:

1. If P12 holds and P13–P14 map bvars's SV prior onto a `CommonSV` option (or a bvars setting onto ours): posterior
   quantiles of `B` and `Σ` under the Phase 2 rule, plus `h_t`, `ρ`, `σ_h` **if P16 shows bvars is exact**; if P16 shows
   an approximate sampler, `h_t`, `ρ`, `σ_h` are reported, not gating, and the 1- and 4-step predictive quantiles join
   the gate.
2. Otherwise: the 1- and 4-step predictive quantiles under the Phase 2 rule, same data and draw counts.

A failure under the selected comparison is **not met** (D-59); A8 and A11 are evidence, never a substitute (E35).

**BoE `bvar` 0.3.5**: the per-draw oracle A10b (its end-to-end conditional forecast re-draws the parameters, D-50).

### 9.9 DGP verdict tests (SPEC §10.5; exit criterion)

Built by the parallel DGP session: `validation/dgp.py`, `benchmarks/dgp_study.py`, `tests/dgp/` and the pre-registration
`benchmarks/reports/dgp_study_preregistration.md` (one design cell: `n = 3, p = 2, T = 200`, volatility axis only; rows
`linear-homoskedastic`, `linear-sv`, `linear-sv-outliers`; 200 replications per row; readings fixed in advance; arms
hash-pinned, and those hashes are in WP26's golden test, so Phase 3 changes no arm). Phase 3 neither duplicates nor
changes it. **The criterion is met** only if the registered full study (all 200 replications per row, from one clean
commit) has been run and its readings pass: both null pairs of `linear-homoskedastic` and the `not_worse` pair of
`linear-sv-outliers` are "not above nominal" (the registration's Wilson rule) at `h = 1` and at `h = 4`; power
(`linear-sv`) is reported, as registered. Otherwise not met (D-59). The full study takes about 10 h on the reference
laptop by the registration's own estimate (Q28). The readings refer to the registration's commit (Phase 2 code); WP39
records them in SPEC §13. Phase 3 rows of the study (CommonSV or GL arms) would be pre-registered separately and are not
part of the exit.

### 9.10 Replication target (SPEC §10.7): Gruber & Kastner (2025), qualitative (WP40)

A pre-registration (`benchmarks/reports/gruber_kastner_preregistration.md`) is committed before any score exists, with
the budget and compute estimate from WP35. It fixes:

- **Data and sample**: FRED-QD 2026-09; GK's 21 series (all present; `UMCSENTx` is missing in 1959Q1 and 1959Q3, so the
  estimation sample starts in 1960Q1, one quarter after GK's 1959Q4); GK's transformations (Δlog, rates in levels);
  expanding windows from 1980Q1; evaluation 1980Q2–2020Q1 (`h = 1`) and the matching `h = 4` origins; nothing after
  2020Q1.
- **Models**: reduced form (GK's sampler), `p = 4`, `SV()` defaults, prior mean 0 for every coefficient (`mean=0.0`),
  `scale="none"` (GK's raw coefficients); arms Minnesota-EB, `Horseshoe` (olcl-lagwise and global), `NormalGamma` (grid
  `a`; olcl-lagwise and global), `R2D2` (grid `a`), `DirichletLaplace` (grid `a`), `SSVS` (`p ~ Beta(1, 1)`).
- **Departures from GK, listed**: the Normal prior on `a_ij` (GK: horseshoe on `u`; Q17); Minnesota-EB instead of
  MP_LIT and SHM; the deterministic prior `var_const = 1000` scaled by `ψ_j` (GK: unscaled `N(0, 1000)`); the sample start.
- **Scores**: S1 the 21-variable joint log score, S2 the GDP/CPI/FFR joint log score; `lpd_method="conditioned"` at
  `h = 1`, `"integrated"` at `h = 4` (D-34).
- **"Beats"**: total ΔLPS > 0 and > 3 MCSE (the verdict's precision rule, SPEC §9.5 item 2).
- **Criteria** (S1 unless stated): C1 at `h = 1` semi-global HS beats Minnesota-EB; C2 at `h = 4` at least three of
  {HS, NG, R2D2, DL} (semi-global) beat Minnesota-EB; C3 at `h = 1` semi-global HS beats global HS; C4 over 2009Q1–2015Q4
  at `h = 1` the best semi-global GL prior beats Minnesota-EB; C5 over 1980Q2–2007Q4 at `h = 1` no semi-global GL prior
  beats Minnesota-EB; C6 at `h = 1` SSVS has the lowest score of the GL arms. S2 is reported for every criterion.

A criterion not met is reported, never tuned (as the Clark–Ravazzolo acceptance did). Whether the acceptance is "met or
documented" is the owner's decision (Q31).

### 9.11 CI tiers

- **PR tier** (fast suite): Tier U and the per-PR block Geweke; G-small rows 301, 311, 320, 321; A8 (`n = 3`), A9,
  A10, A10b, A11, A12, A13; contract tests for three new MCMC compositions (structural × SV × HS, conjugate × CommonSV,
  structural × CommonSV) plus `StructuralNIGFit` and conditional results; fixture reads and short parity runs. The
  increase is measured on the work-package PRs and capped at +10 min per cold job; rows beyond the cap move to the weekly
  tier (never dropped).
- **Weekly tier**: G-full and S for rows 301–330, G-small for 401–408, M19–M35, A8 (`n = 20`), A9's MCMC leg, the long
  parity runs, contract tests for every new MCMC composition.
- **Shards** (WP36, `tools/weekly_shards.py`): `geweke-p3a` (structural rows), `geweke-p3b` (reduced and CommonSV rows),
  `sbc-p3a`, `sbc-p3b` (same split, by file); the Phase 2 shards `--ignore` the Phase 3 files; each shard ≤ 200 min on a
  2-vCPU EPYC 7763, split further if not; the partition check covers them. Weekly runs a work package needs are
  dispatched by hand on its branch.

---

## 10. Performance

Estimates to be replaced by WP35's measurements (`docs/phase3/pilot.md`):

- **GL scale step**: elementwise work on `n²p` coordinates per chain plus `G` segment sums and an `A × G` grid. Under
  `vmap` the GIG `while_loop` runs to the slowest lane: with per-attempt acceptance ≥ 1/3.46, the maximum over 10⁴ lanes
  is about 27 attempts, so the step costs about 27 elementwise passes (≈ 0.5–1.5 ms per sweep of 4 chains at
  `n = 20, p = 4`, against BENCH-P2's 3.90 ms). The deviation path costs D-14's flops plus elementwise logs. GL ×
  Constant loses the eigen path (0.35 → ≈ 3 ms per sweep).
- **Conjugate CommonSV** (shared factor): per chain and sweep one Gram `O(T q²)`, one Cholesky `O(q³/3)` and `n`
  pairs of triangular solves `O(n q²)`. `n = 20, p = 4` (`q = 100`): ≈ 4·10⁶ flops, ≈ 0.3–0.6 ms per sweep of 4 chains
  with the `h` blocks; `n = 100, p = 4, T = 300` (`q = 500`): ≈ 2·10⁸ flops, ≈ 3–6 min for 4 × 11,000 sweeps on one
  device (the D2 screen; structural SV is `O(n q³) ≈ 1.25·10¹⁰` flops per sweep there). The per-equation program with
  other priors costs what structural SV costs.
- **BENCH-P3** (reported, not gated; SPEC sets no Phase 3 target), `benchmarks/phase3_perf.py`: the conjugate CommonSV
  at `n = 100, p = 4, T = 300`, 4 × 10,000 kept draws; HS × SV at the BENCH-P2 shape (overhead against BENCH-P2).

---

## 11. Risks

| # | Risk | Mitigation |
| :- | :--- | :--- |
| R18 | GL mixing (HS global scale; small `a` on the grid) leaves fits unconverged | WP35 pilot; group scales primary (D-52); budget D-58; warnings |
| R19 | GIG sampler wrong at extreme parameters | Devroye (2014) §6 with a uniform bound, log-space evaluation, the pinned Tier U grid, M34 |
| R20 | `bayesianVARs` cannot be built, or a probe stays undecided | D-54 (Linux with gfortran); an undecided probe is recorded with its evidence and its runs are **not met** (D-59); the owner decides |
| R21 | bvars is approximate (auxiliary mixture) or its `V` cannot be fixed | P16 and P12 select the gating comparison before it runs (§9.8) |
| R22 | CommonSV `h` acceptance falls with `T` and `σ_h²` (N08 §A5.2) | blocks of 100 dates; acceptance warning; pilot at `T = 600` |
| R23 | Conditional forecasts near rank deficiency | the `R₁` diagonal check of E-16; never jitter |
| R24 | CI time grows with 30 rows × (G, S) | four new weekly shards; four PR rows; the +10 min PR cap |
| R25 | Contract additions change Phase 2 draws | `Empty` defaults; golden templates and hashes; bitwise before/after tests |
| R26 | Parallel sessions edit shared files | ownership of §1.2; rebase on their merges; `ci.yml` edits named per WP |
| R27 | GK sample details (UMCSENTx gaps) | the sample start fixed in the pre-registration |
| R28 | GitHub Actions minutes for R builds | dispatch-only job with a cached library |
| R29 | The shared-factor path's two-block `(θ_i, λ_i)` Gibbs mixes slowly when `q_i ≈ T` | WP35 measures; Q32 (a collapsed `λ_i` draw) |
| R30 | The DGP full study's compute (≈ 10 h) is not scheduled | Q28 |

---

## 12. Open questions for the owner

| ID | Question | Recommendation |
| :- | :--- | :--- |
| Q15 | `hierarchy="mh"` on the MCMC engine (SPEC §4.4–4.5, P2 D-12) | Defer past v0.3.0: an exact global step over `(λ, cross)` fits the D-40 machinery but is not in the SPEC §13 list; change E-5 and SPEC §4.4–4.5 |
| Q16 | Estimated Lenza–Primiceri scaling on the MCMC engine (SPEC §4.5, §5.6, P2 D-21) | Defer; CommonSV covers a common variance spike now, a common outlier component later (N08 §A9); E-6, SPEC §4.5, §5.6 |
| Q17 | GL shrinkage of the contemporaneous coefficients (SPEC §4.4) | Defer: it needs the structure's `contemp_moments` to read coefficient-prior state (D-28); parity uses `cholesky_U_prior = "normal"` |
| Q18 | Continuous ν prior (P2 Q8) | Defer |
| Q19 | `solver="bcm"` for `q > T` | Defer (the Cholesky path is exact, only slower) |
| Q20 | `HMP` (bayesianVARs' hierarchical Minnesota) | Not in v0.3.0; a GL-family member later if wanted |
| Q21 | Default MCMC budget of GL compositions | Decide from WP35 (D-58); gate unchanged |
| Q22 | R environment for the oracles, and permission to download it | **Resolved (owner, 2026-10-25): "(a) CI job + artifact"**: the dispatch-only `r_fixtures` job and `gh run download` of its artifact; no Docker image, no local R (D-54) |
| Q23 | GL defaults: GK's semi-global with hyperpriors, or `bayesianVARs`' global with fixed `a` | GK's (§3.2); parity sets `bayesianVARs`' values explicitly |
| Q24 | CommonSV extensions (Student-t, random walk, outliers) and the t part of the bvars oracle row | Defer (N08 §A5.4, §A9) |
| Q25 | Conditional forecasts: soft conditions, parameter updating (Waggoner–Zha Gibbs), weighting by `w_s` | Defer; the weights are stored |
| Q26 | Compute for the GK acceptance run | Estimate from WP35; run locally after the pre-registration, with a documented quick config first |
| Q27 | Keep D3 for `Minnesota(form="auto") × CommonSV` (structural), or make the conjugate screen the `auto` default (D2's spirit, a SPEC D3 change) | Keep D3; the conjugate screen is one keyword away |
| Q28 | Compute for the DGP full study (≈ 10 h, the registration's estimate) | Run it on the reference laptop once the DGP PR merges |
| Q29 | "Recovery" in the DGP layer (SPEC §10.1) | Defer (posterior coverage of the true parameters needs fits the study does not keep); SPEC §10.1 edit |
| Q30 | Deterministic terms outside the shrinkage (D-36) | Keep (GK, `bayesianVARs`); SPEC §4.4 edit |
| Q31 | The GK acceptance cannot fail ("met or documented"), as with Clark–Ravazzolo | Accept, as in Phase 2 |
| Q32 | A collapsed `λ_i | h, y` draw for the shared-factor path (better mixing at `q ≈ T`) | Later, if WP35 shows the need; it needs the mean step to draw a vol-state leaf |

---

## 13. Work packages (dependency-ordered)

**Rules** (P2 §13 rules 1–5 apply). Each WP owns its files (§1.2) and tests (§1.3); the contracts of WP26/WP27 freeze
when their PR is open and change only through an amendment recorded here; kernel WPs pass Tier U and B before their PR;
WP30 and WP36 are written by an author who wrote no Phase 3 kernel (recorded in the PR); every WP keeps the Phase 1–2
suites green except the amendments named in §7.3; clean room throughout. One PR per WP (D-57); the base branch of each
is named below. The stack is linear because the WPs share files; only WP29 branches off independently.

| WP | Content | Depends on (base branch) | Acceptance |
| :- | :--- | :--- | :--- |
| WP26 | Contracts (§2): config rules, state classes, prior arrays, keys, registry, protocols, compose hooks, goldens | this design (`phase-3-design`) | rules tested; Phase 2 templates, hashes (and the DGP arms') and short runs bitwise unchanged |
| WP27 | Specs and API (§7): GL classes, `DiscreteGrid`, `AsymmetricConjugate`, `CommonSV`, `TruncatedNormal`; resolution, engine table, E-1…E-16, spec/hash, labels, harmonise; the `test_api_resolution.py` amendment | WP26 | every §7.2 row and §7.3 message tested; Phase 1–2 golden hashes unchanged; `import rai_bvar` imports no JAX |
| WP28 | Numerics (§3.4, §3.5, §5.3, §5.4): `log_gig`, `precision_draw_log`, `shared_factor_*`, tridiagonal Newton | WP27 | §9.2 item 1 green |
| WP29 | Oracle environment and fixtures (D-54, §9.8): `make_r_fixtures.R` sections `bvars` (P1–P10, R1–R9) and `wbvars` (P11–P16 and its runs), packing, the dispatch-only `r_fixtures` CI job, the SV-t fixture if `stochvol` ≥ 3.2.5 | this design (Q22 resolved) | fixtures committed with versions; every probe decided or recorded undecided with evidence; the bvars gating comparison recorded |
| WP30 | References (§9.1), the exogenous design, `COMPOSITIONS_P3`, `BLOCK_CONFIGS_P3`; explosive-mass measurement for the lagged rows | WP28 | prior moments vs closed forms; simulators vs NumPy loops (1e-12); `log_joint` vs SciPy (1e-10) |
| WP31 | GL kernels and `GLScaleStep`, deviation paths, builders, resolve, monitors, runner init order, storage (§3) | WP30 | Tier U, B green; rows 301 and 311 G-small green; A12, A13 green |
| WP32 | CommonSV (§5): common cores, `CommonSVStep`, the shared-factor path, forecasts, warm starts, monitors, storage | WP31 | Tier U, B green; rows 320 and 321 G-small green; A8 (`n = 3`), A11 green |
| WP33 | `StructuralNIGFit` and its adapter, EB with free `cross`, MH, `lag_coefs` on analytic forecasts (D-49), ordering support, the `test_backtest_lpd_integrated.py` amendment (§4) | WP32 | A9 green; Phase 1 tests unchanged |
| WP34 | Conditional forecasts (§6) | WP33 | A10, A10b green on every engine; contract checks |
| WP35 | Pilot: GL and CommonSV mixing and cost, BENCH-P3 (§8, §10) | WP34 | measurements with machine fingerprint; recommendation for D-58; compute estimates for Q26 and Q28 |
| WP36 | Composition suites (§9.4–§9.6): rows 301–330, blocks, SBC, mutants M19–M35, contract tests, weekly shards | WP35 | G-small (PR rows) green; G-full, S and M green on a dispatched weekly run; every mutant caught |
| WP37 | Twins at scale: A8 (`n = 20`), A9's MCMC leg | WP36 | green weekly |
| WP38 | Parity tests (§9.8) | WP37 with WP29 merged in | the parity exit criteria (map below) |
| WP39 | DGP exit (§9.9): the DGP session's merged PR and its full study's readings | WP38 (after the DGP PR merges) | the readings recorded; criterion met or not |
| WP40 | GK replication: pre-registration, then the acceptance run and report (§9.10) | WP39 | report against C1–C6 |
| WP41 | Release: SPEC v2.4 (status, the §0.2 edits the owner accepted), README roadmap, this document's exit map, `ci.yml` tiers, `pyproject` 0.3.0, release notes; tag only on the owner's approval after a status report | WP40 | CI green; the weekly tier green on the release commit with no skipped parity test |

### Exit-criteria map (SPEC §13 Phase 3)

| Criterion | Delivered by |
| :--- | :--- |
| `bayesianVARs` parity, global-local priors × SV | WP38 runs R3–R9 (fixtures WP29, probes P6–P10); pass/fail (D-59) |
| `bayesianVARs` parity, reduced-form sampler, after P1–P5 | WP38 runs R1–R2 (fixtures WP29); pass/fail |
| `bvars` parity for `CommonSV` | WP38, the comparison selected by P11–P16 (§9.8); pass/fail; A8 and A11 are evidence only |
| DGP verdict-engine tests on linear DGPs | WP39: the registered full study's readings (§9.9); pass/fail |
| R6: Geweke and SBC for every new block and interaction class | WP36 (references WP30), with the coverage argument of §9.4 |
| Replication target (SPEC §10.7) | WP40 (C1–C6; acceptance per Q31) |

---

## 14. References (new in Phase 3; Phase 2's are in SPEC §15)

- Bańbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario analysis with vector
  autoregressions for large cross-sections. *International Journal of Forecasting*, 31(3), 739–756.
- Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–Laplace priors for optimal shrinkage.
  *Journal of the American Statistical Association*, 110(512), 1479–1490.
- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2),
  465–480.
- Chan, J. C. C. (2020). Large Bayesian VARs: A flexible Kronecker error covariance structure. *Journal of Business &
  Economic Statistics*, 38(1), 68–79.
- Devroye, L. (2014). Random variate generation for the generalized inverse Gaussian distribution. *Statistics and
  Computing*, 24(2), 239–246.
- George, E. I., & McCulloch, R. E. (1993). Variable selection via Gibbs sampling. *Journal of the American Statistical
  Association*, 88(423), 881–889.
- George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search for VAR model restrictions. *Journal of
  Econometrics*, 142(1), 553–580.
- Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma prior distributions in regression problems.
  *Bayesian Analysis*, 5(1), 171–188.
- Gruber, L., & Kastner, G. (2025). Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!
  *International Journal of Forecasting*, 41(4), 1589–1619 (arXiv:2206.04902, App. B).
- Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the horseshoe estimator. *IEEE Signal Processing Letters*,
  23(1), 179–182.
- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate models. *Review of Economics and
  Statistics*, 81(4), 639–651.
- Zhang, Y. D., Naughton, B. P., Bondell, H. D., & Reich, B. J. (2022). Bayesian regression using a prior on the model
  fit: The R2-D2 shrinkage prior. *Journal of the American Statistical Association*, 117(538), 862–874.
