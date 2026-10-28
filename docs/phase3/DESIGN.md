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
§0.4 (errata E29–E86); the draft was revised (`6dbd426`), and two further reviewers, **R2a** (mathematics, with
numerical checks) and **R2b** (engineering and process), verified the revision and raised 44 findings (3 blockers),
resolved as errata E87–E116. No Phase 3 code was written before this version. The design was merged as PR #13
(`552fe30`). A merge-gate review (**R3**: four lenses, mathematics, engineering, scope and validation, each major
re-checked by a second reader) found no blocker, 12 majors and 26 minors; **Amendment 1** (errata E117–E159) applies
them before the work packages they gate, with the §9.8 probe rules as WP29 implements them. A merge-gate review of the
amendment and of WP26 (**R4**) found no blocker; its findings revised E124, E150, Q36 and row 28 in place and added
E153–E159, and a re-check of those fixes revised E123, E124, E154 and E157 again. Amendment 1 was merged as
PR #16 (`05757c1`). **Amendment 2** (errata E160–E169) records what the work packages WP29, WP31 and WP33 found
while implementing this design, the owner's decisions on Q35, Q38, Q39 and Q40, and the exact `CommonSV` oracle
(§9.8, WP38a) that Q40 adds. A merge-gate review of the amendment (**R5**) found no blocker; its findings revised
E163, E164, E167 and E168 in place (the oracle's fixed run length and its non-centred `σ_h` move, the decisive engine
run, the independence rule) and opened Q41, which the owner resolved (E169). Amendment 2 was merged as PR #24
(`49d41fe`). **Amendment 3** (errata E170–E174) records what WP32 found while implementing `CommonSV`, WP29's
regeneration of the `n = 7` bayesianVARs SV fixtures after PR #21's review, which moves §9.8's pin, and the owner's
decisions on Q42 (bayesianVARs' SV predictive at `h ≥ 2`, computed on the R side from its own draws) and Q43 (A10's
draw count on the analytic engines). Amendment 3 was merged as PR #26 (`80170af`). **Amendment 4** (errata E175–E178)
records the batch merge-gate review of the Phase 3 PRs (2026-10-26; **BR** below, since R1–R9 name bayesianVARs runs
and R6 a SPEC criterion): `CommonSV`'s cold start, the ordering prior
under `CommonSV`, WP32's and WP33's unrecorded choices, and the wording minors. **Amendment 5** (erratum E179)
records the owner's decision on Q44, the SBC cap of eight rows after WP35's weekly run, and the re-check of Amendment
4 (wording). **Amendment 6** (erratum E180) records the owner's decision on Q45: how the weekly tier presents SBC rows
308 and 325, which E179 records as not met (presentation only). The largest revisions: the
conjugate `CommonSV` became a shared-factor path of the per-equation sampler instead of a separate Kronecker program
(E29), the GIG sampler follows Devroye (2014) §6 (E31), every parity and DGP criterion is pass/fail with its
tolerance stated here (E34–E36), and conditional forecasts are labelled scenarios (E43); the second round made the
shared-factor weights reach the equation body unbatched (E87), fixed the dummy-row coupling (E88) and the GIG set-up in
log space (E89), and replaced the bvars probe that had no data (E91).

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
and may be read and used as a Python-side oracle in CI. No code is translated from GPL sources. A failed build or
install of a GPL package (D-54) is diagnosed from the exit status, the failing step and dependency or toolchain
messages only; compiler diagnostics that quote package source are not read (the job's log step filters them if
needed) (E143).

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
P1–P5 (and the Phase 3 probes P6–P10) are settled; `CommonSV` parity with an exact independent oracle (SPEC §13 as
amended by Q40: §9.8, WP38a, runs C1–C2; the `bvars` comparison is evidence, Q35); DGP verdict-engine tests on linear
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
| 5 | `AsymmetricConjugate` prior and engine rule | SPEC §4.4, §4.5 last table row, P2 Q13 | **In** (§4); SPEC edit §4.5: the row reads `× Constant("auto"\|"iw")` (E141) |
| 6 | `CommonSV` | SPEC §4.4, D2, N08 §A | **In** (§5); SPEC edit §4.4/§13: "Chan (2020) common SV, the CCM (2016) model up to the presample state, priors and sampler" (E141) |
| 7 | Conditional forecasts | SPEC §6.2, §12 tree `forecast/conditional.py` | **In** (§6), per-draw conditioning, parameters not updated; SPEC edit §6.2 (states the convention) |
| 8 | `bayesianVARs` parity: reduced-form sampler, GL × SV, probes P1–P5 | SPEC §10.4, §13; P2 §9.9, WP21 | **In** (§9.8; WP29, WP38) |
| 9 | `bvars` parity for `CommonSV` (SPEC §13 as first written) | SPEC §13, §10.4, §2.2 | **Changed** (Q35, Q40): the criterion is `CommonSV` parity with the exact oracle of §9.8 (WP38a, runs C1–C2); the `bvars` comparison selected by P11–P16 is reported as evidence (branch 2 for both volatilities: not met for want of an exact mapping, E167); SPEC §13, §10.4 and D2 are amended in Amendment 2 (E168) |
| 10 | DGP verdict tests on linear DGPs | SPEC §10.1, §10.5, §13 | **In**: the parallel DGP session's harness and pre-registered study; WP39 holds the exit to its readings (§9.9) |
| 11 | "Recovery" in the DGP layer | SPEC §10.1 table | **Deferred**, pending Q29; SPEC edit §10.1 |
| 12 | Gruber & Kastner (2025) sparse-vs-dense replication | SPEC §10.7 | **In**, pre-registered (WP40) |
| 13 | `hierarchy="mh"` on the MCMC engine | SPEC §4.4 Minnesota row ("raises until Phase 3"), §4.5 last paragraph, P2 D-12/§6.5, `api.py` | **Deferred**, pending Q15; SPEC edits §4.4, §4.5; message E-5 |
| 14 | Estimated Lenza–Primiceri scaling on the MCMC engine | SPEC §4.5 last paragraph, §5.6, P2 D-21/§3.9, `api.py` | **Deferred**, pending Q16; SPEC edits §4.5, §5.6; message E-6 |
| 15 | Continuous ν prior | P2 Q8 | **Deferred**, pending Q18 |
| 16 | `solver="bcm"` (Bhattacharya et al. 2016) for `q > T` | P2 §3.9, N07 §5 | **Deferred**, pending Q19 (the Cholesky path stays exact for `q > T`, only slower) |
| 17 | A GIG sampler | N07 §9 (open), P2 §3.9 | **In** (D-42) |
| 18 | Analytic forecasts without `lag_coefs` | P2 D-34, §7.3 | **Changed**: analytic and structural-NIG forecasts attach `lag_coefs` on request (`forecast(..., lag_coefs=True)`; D-49, E120) |
| 19 | The SV-t parity row stays an expected failure until a `stochvol` ≥ 3.2.5 fixture exists | P2 §2.10 Stage 4 (1) | **In** if WP29's R environment has `stochvol` ≥ 3.2.5 (regenerated); otherwise unchanged |
| 20 | Two-series SBC for VAR-SV | N06 §6 (optional) | Not planned |
| 21 | `HMP` (bayesianVARs' hierarchical Minnesota) | SPEC §2.2 table | Not in scope (Q20) |
| 22 | `bvars` as oracle for common SV **with t errors**; `CommonSV` t, random walk, outliers | SPEC §2.2, N08 §A5.4, §A9 | **Deferred**, pending Q24: the t part of the bvars oracle row waits with the t extension |
| 23 | Soft conditions; parameter-updating conditional forecasts | Waggoner & Zha (1999) | **Deferred**, pending Q25 |
| 24 | SPEC §10.6 default-budget text; §10.4 "not run in CI"; §5.8 "CI never touches the network" | SPEC | **Edits** in WP41: CommonSV counts as SV (D-51); the R job is a dispatch-only, non-test job, and §5.8/§10.4 take D-54's wording: no test job installs R, only the network-marked weekly tests download data, the PR tier never touches the network (E136, E159) |
| 25 | SPEC §12 tree paths (`vol/common_sv.py`, `components/priors/*`) | SPEC §12 | **Edit** in WP41 to the files of §1.2 |
| 26 | The CommonSV false-alarm control of the detection test (`test_detection_power_and_false_alarm_control` uses a stand-in until Phase 3) | N08 §B5 item 5 | **In** (WP32 replaces the stand-in; E140) |
| 27 | Ordering invariance within MCSE of the order-invariant CommonSV (conjugate × `CommonSV`), which `monitor.is_order_invariant` relies on | N08 §B5 item 2 | **In**: a weekly test that the user ordering and its reverse agree under the Phase 2 parity rule (WP35; E140) |
| 28 | `bayesianVARs` parity under the user ordering and its reverse | N08 §B5 item 7 | **Deferred**, pending Q37: the SPEC §13 Phase 3 exit requires `bayesianVARs` parity for GL × SV and for the reduced-form sampler, not per-ordering parity. Every one of R1–R9 is order-dependent (Cholesky SV, or `Constant("ig")` with an unscaled `a` prior; `monitor.is_order_invariant` is false for them), so a per-ordering comparison is a separate run. The ordering machinery is checked by rows 26 (WP32) and 27 (WP35) (E140, E156) |
| 29 | A qualitative CCM (2016) replication | N08 "Parity and replication (Phase 3)" | **Not planned**: the shipped model is Chan's (2020) specification (row 6); the GK replication (§9.10) is the Phase 3 replication target (E140) |

### 0.3 Decision register

| ID | Decision | Source | Notes / supersedes |
| :--- | :--- | :--- | :--- |
| D-36 | **What is shrunk** (pending Q30). The global-local and SSVS priors govern the lag coefficients only (`n·p` per equation). Deterministic terms keep the fixed Normal prior of the form (structural Mode I `N(m, κ_d ψ_i/ι)`; reduced `N(0, κ_d ψ_j)`; `var_const`), the contemporaneous `a_i` keep the Phase 2 prior (Mode I `N(0, κ_a ψ_i/(ι ψ_j))`, or λ-scaled under `iw_equivalent`; Q17). Shrinkage is towards the transform-aware prior mean (SPEC §5.3): the prior governs the **deviation** `d = coefficient − m`. GL priors have no SOC/SUR dummy rows | GK (2025) §2 (intercept `N(0, 1000)`, unshrunk); `bayesianVARs` manual (`prior_intercept` fixed) | SPEC §4.4 lists deterministics and contemporaneous as groups (§0.2 rows 3–4) |
| D-37 | **Scale.** Kernel variance `r_j v_j`: `r_j = ψ_i/(ι ψ_m)` (structural) or `ψ_i/ψ_m` (reduced) for the coefficient of equation `i` on a lag of variable `m`, so the prior is in the units of the Mode I Minnesota prior at `λ = 1` and the model is scale-equivariant (ψ per window, R9). `ι` is unit-free, so an equivariance test cannot see a dropped `ι`; the value test of §9.2 item 5 pins it (E126). `scale="none"` sets `r = 1` (raw coefficients; parity). No lag decay: lag-wise groups learn it | N05 §4.1; GK (2025) `r̃_i` | — |
| D-38 | **Groups.** `groups="olcl-lagwise"` (default; `2p` groups: own lags of lag `l` across equations, cross lags of lag `l` across equations), `"global"` (1), `"equation-wise"` (`n`), `"covariate-wise"` (`n·p`: regressor `(l, m)` across equations), or an explicit `(n, n·p)` integer array. Names follow `bayesianVARs`; their meaning there is probe P7 | GK (2025) §3.2: semi-global beats vanilla in every simulated design and in the application | — |
| D-39 | **Prior families and parametrisations** (§3.2): HS (Makalic & Schmidt 2016), NG (`a, b, c`, GK/`bayesianVARs` meaning), R2D2 (`a, b`, GK gamma form), DL (`a`, gamma form), SSVS (`c0, c1, semiautomatic, p`). `a` is fixed or has a discrete hyperprior (default: GK's grid) | GK (2025) §2–3, App. B, re-derived here | — |
| D-40 | **One post-phase scale step** `G_COEF_GLOBAL` (`Block 17`, reserved by P2 §5.6) draws every local, group and hyper parameter of a GL prior, for the structural and the reduced form alike, after the equation phase, in the order **local → `a_g` (its rate integrated) → rate `ξ_g | a_g`** (NG, R2D2), local → group (HS, SSVS), local → `a_g` (DL). The coefficient-prior kernel declares `has_local_state = False`, so no `coef_local` equation step is planned | P2 §3.4, §4.8 item 3 (post steps may write `eq.coef`, `glob.*`) | P2 §3.4 sketched local scales in `sample_local_eq`; one step is simpler because semi-global groups cross equations and the reduced form's coefficients are global (`ReducedEq.coefficients_eq` docstring) |
| D-41 | **Log space, no floors.** Every scale is stored as its logarithm; draws are log-gamma, log-inverse-gamma and log-GIG; the mean kernels draw GL coefficients in deviation coordinates with a log-space Jacobi scaling and return `log|d|` exactly (§3.4). No tolerance, floor, truncation or jitter is added to any prior | P2 D-14, D-18, N07 §9 | `bayesianVARs` floors draws by default (`PHI_tol`, `*_tol`); its parity runs set them to 0 (§9.8) |
| D-42 | **GIG sampler**: Devroye (2014) §6 — the three-piece hat on `log X` with his explicit points `t, s` (uniform bound 3.4597 expected iterations, Theorem 2), every exponential term evaluated in log space; `χ = 0` or `ψ = 0` draw log-gamma exactly; attempt cap 512 (§3.5) | Devroye (2014) Theorems 1–2 | E31 |
| D-43 | **Where GL priors run.** Structural Mode I or reduced form, with `Constant("ig")`, `Constant("iw_equivalent")`, every `SV(...)` and `CommonSV()`. Rejected at construction: the conjugate form, Mode II (`Constant("iw")` × structural), the analytic engine. `form="auto"` resolves to `"structural"` (D3). A GL prior varies, so the eigen, fixed-factor and shared-factor paths are off (P2 D-26) | P2 §3.8, §11.1 | — |
| D-44 | **SSVS scales.** `semiautomatic=True` (default): `τ_{0,j} = c0·ŝ_j`, `τ_{1,j} = c1·ŝ_j`, `ŝ_j` the flat-prior posterior s.d. of the coefficient on the window (R9; §3.6 formulas). Needs `T_eff > q_i + 2` for every equation (structural) or `T_eff > k + n + 1` (reduced), else an actionable error at fit. `semiautomatic=False`: `τ = c·√r_j` | George, Sun & Ni (2008); GK (2025) §2.4 | the `bayesianVARs` scaling is probe P9 |
| D-45 | **`AsymmetricConjugate`** is Chan's (2022) prior in our structural form (Mode II arrays with `cross` and `contemp`, δ-coordinates and D3 centring, P2 §6.1), fitted on the **analytic engine** by the existing closed form (`engines/structural_nig.py`, through an adapter): iid draws, exact log ML. `hierarchy`: `"fixed"`, `"eb"` (default; free `lambda` and `cross`, Chan's EB), `"mh"` (random-walk MH over log hyperparameters, as Phase 1). Only with `Constant` volatility (`"auto"` resolves to `"iw"`). Its posterior equals the MCMC composition `Minnesota(form="structural", cross=…) × Constant("iw")` (P2 rule 4), which stays MCMC; a verdict between the two compares engines | Chan (2022) §2–4; P2 Q13, WP16 | — |
| D-46 | **`CommonSV` model**: `u_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `E h_t = 0` (identification), stationary `h_0 ~ N(0, σ_h²/(1 − ρ²))` at the first data row (0-based). Defaults `ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)`, `σ_h² ~ IG(5, 0.04)`; options `ρ: Beta` on `(ρ+1)/2` and `σ_h²: ScaledChi2` (parity) | N08 §0, §A1–A3 (Chan 2020 priors) | — |
| D-47 | **One `CommonSV` program, per equation, with a conjugate fast path.** Every `Linear × CommonSV()` runs the per-equation sampler: Constant cores `λ_i`, `λ_{i,t} = λ_i e^{h_t}`, a post step `h, ρ, σ_h | q_t = Σ_i e_{it}²/λ_i`. With `Minnesota(form="conjugate")` the composition is structural **Mode II at `cross = contemp = 1`**, which is the NIW prior exactly (P2 §6.1, N05 §4.2), so the model is Chan's (2020) common-SV model with the Phase 1
prior, and the mean takes the **shared-factor path**: the λ-free prior precision and the weights `e^{−h_t}` are the same for every equation, so one Cholesky of the shared `q × q` matrix serves all equations through its leading blocks (§5.3): `O(T q² + q³ + n q²)` per sweep, the conjugate speed. `form="auto"` with `CommonSV` resolves to `"structural"` (D3 unchanged); the conjugate screen is explicit | N08 §A1, §A5, §A11 item 2; Chan (2022) Prop. 1 | the draft's Kronecker program needed hooks the engine lacks (E29); N08 §0's reduced-form NIW draw targets the same posterior |
| D-48 | **`h` step**: blocked exact independence MH, blocks scanned **sequentially** (each conditions on its updated neighbours), block layout 0-based and covering `[0, T_eff)` every sweep (§5.4), Newton mode from a start that depends only on `q`, the neighbours and `(ρ, σ_h)`; `ρ` truncated-normal independence MH with the stationary-start factor (a uniform proposal under the Beta prior when `K_ρ` has zero precision; under TruncatedNormal always the TN proposal; E117, E158); `σ_h²` exact (IG, or GIG under `ScaledChi2`) | N08 §A5.2–A5.3 (derived there; Geweke-gated here) | the Omori-mixture variant (CCM 2016) is not shipped: it is not exact (N08 §A11); see P16 |
| D-49 | **`lag_coefs` on request on every linear forecast** (amended by E120). `AnalyticFit.forecast` and `StructuralNIGFit.forecast` take a keyword-only `lag_coefs: bool = False`; when true, the accelerator variable `lag_coefs` (P2 D-34) is filled per chunk from the coefficient draws `_iter_draws` already yields (no new random numbers, so `paths`, `cond_mean` and `cond_chol` are bitwise the same either way). `models/base.py`'s forecast protocol gains the keyword (benchmarks accept and ignore it); `backtest` (with `estimate_cost`), `compare` and `ordering_sensitivity` request it when `lpd_method="integrated"` and some horizon is `> 1`; `score_forecast` makes no forecast and keeps raising (E52) for a forecast made without `lag_coefs=True` (E159); `conditional_forecast` builds its base forecast with it. So the integrated LPD and conditional forecasts work on every engine, while the default analytic forecast, its memory and its cache keys stay as in Phase 1. MCMC forecasts keep attaching it always (P2 D-34). Cost on request: `S·p·n²·8` bytes, `p` times the analytic `cond_chol`'s physical storage, which is a zero-stride view (1.04 GB at the SPEC §13 Phase 1 shape `n = 100, p = 13, S = 1000`, against a 0.45 GB peak) | P2 D-34, §7.3 | amends P2 §7.3 "analytic-engine forecasts carry no `lag_coefs`" for forecasts made with `lag_coefs=True` only |
| D-50 | **Conditional forecasts are scenarios.** Hard conditions `y_{T+h, j} = r_{h,j}` on the model scale; per draw, the exact Gaussian conditioning of the unconditional joint path given the parameter draw and its drawn volatility path; the conditions update neither parameters nor volatilities, and draws are not reweighted by how likely they make the conditions (the per-draw convention; the log weights `log N(r; Rμ_s, R₁ᵀR₁)` are stored for users who want to reweight). Computed from an unconditional forecast without new random numbers; returned as `ConditionalForecastResult`, which `score_forecast` refuses (§6) | Waggoner & Zha (1999) §3 per-draw step; Bańbura, Giannone & Lenza (2015) | BoE `bvar` instead re-draws parameters from the data and the conditional path (`tests/parity/BOE_BVAR_NOTES.md` §10), so it is an oracle only per draw (A10b); Q25 |
| D-51 | **Budgets** (extends D-35): `CommonSV` is a stochastic-volatility composition for D-35 (default 4 × 10,000 kept draws, warning below). GL compositions: the default is frozen by the WP36 pilot (§8; Q21, D-58); until then they take the D-35 default of their volatility. The convergence gate (P2 §8.2) is unchanged | P2 D-35, Q14 | — |
| D-52 | **Monitors** (extends D-33): GL group log scales are **primary** when the number of groups `G ≤ max(2p, n)` (global, olcl-lagwise, equation-wise), secondary otherwise; SSVS `p_g` is primary under a Beta hyperprior; hyperprior indices are secondary; local scales are latent, neither monitored nor stored (§3.7). `CommonSV`: `h_T`, `ρ`, `σ_h` primary (with every `log λ_i`, as for `Constant`). They are monitor fields of `fit.posterior["monitor"]`. The SSVS indicators `γ` are stored as `int8` with the stored parameter draws (thinned like them; `n·k` bytes per stored draw, §10; E123, E155) | SPEC §10.6 ("monitored on hyperparameters") | — |
| D-53 | **Grid harmonisation** (extends D-19 and keeps P2 §11.4's trigger; amended by E119): if any arm has a volatility other than `Constant` with prior `"auto"`/`"iw"` (that is, `Constant("ig")`, `Constant("iw_equivalent")`, any `SV(...)`, or a `CommonSV` whose Minnesota form is not explicitly `"conjugate"`), or has a GL prior, every `Minnesota(form="auto")` arm becomes `"structural"` and its `Constant("auto")` becomes `"ig"`. The one narrowing of P2 §11.4: a `CommonSV` arm with an explicit `Minnesota(form="conjugate")` (the conjugate screen) does not trigger, so it is compared against the analytic arms (§7.4). GL arms are structural already; `AsymmetricConjugate` arms are never rewritten (they stay analytic) and do not trigger the rewrite | P2 §11.4 | — |
| D-54 | **Oracle environment** (Q22, resolved by the owner on 2026-10-25: "(a) CI job + artifact"). An input-gated job `r_fixtures` in the existing `ci.yml` (GitHub dispatches only workflows on the default branch), run by `workflow_dispatch` only (never on push, pull request or schedule), on `ubuntu-latest`, installs R and `bayesianVARs` 0.1.8, `bvars` 1.1 and `stochvol` from a pinned Posit Package Manager snapshot (its Linux binaries; CRAN sources built with gfortran where no binary exists; E147), runs `tools/make_r_fixtures.R` and uploads the fixtures as an artifact, which is downloaded with `gh run download` and committed with R, package and snapshot versions in the fixtures' attributes. No local R and no container image. It is no test job: the weekly jobs are gated on `!inputs.r_fixtures` and the concurrency group includes the input. No test job installs R; only the network-marked weekly tests download data (the pinned, checksummed datasets of SPEC §5.8); the PR tier never touches the network (SPEC §5.8, §10.4 reworded to say so; E136, E159) | SPEC §10.4 | P2 failed on R 4.1.2 (no binary, no gfortran); E37 |
| D-55 | **Numbering** continues Phase 2 (Conventions); SPEC §14 points at both registers | — | — |
| D-56 | **Validation designs for heavy-tailed coefficient priors.** GL composition rows run Geweke and SBC on an exogenous fixed design: `X` of shape `(T_eff, k)` with a column of ones for each deterministic term and `N(0, 1)` entries elsewhere, drawn once from the configuration seed and held fixed, `Y` simulated given `θ` without lag feedback (N08 §A11). The lagged-path GL rows use families with exponential or Gaussian tails and a scale knob (NG, SSVS; DL has no scale parameter) with scales whose explosive prior mass, measured by prior simulation in WP30, is within Phase 3's target 2e-4 (E46, E93: the
level Phase 2 measured for its corrected prior at `n = 3, p = 2`, P2 §9.2; Phase 2's own bound is `P(ρ > 1) < 10⁻³`, N06 §9.1;
decided by E150's Monte Carlo rule, pending Q36(i)); the horseshoe's half-Cauchy tail makes that mass ≈ 0.4 at `global_scale = 0.1` (Ra8), so no lagged HS row runs with sampled scales | N06 §3.4; N08 §A11 | E46 |
| D-57 | **Branches and PRs.** One PR per work package. Each work-package branch starts from the branch of the work package it depends on (named in §13), and its PR targets that branch, so the diff is the work package alone; GitHub retargets it to `main` when its base merges. WP29 (fixtures) and WP38a (the exact `CommonSV` oracle) are independent of the stack, target `main` and are merged before WP38. Mechanics (E129): the repository does not delete
merged branches, so GitHub does not retarget a dependent PR; after a work package's PR merges, the next PR is
retargeted to `main` (`gh pr edit <next> --base main`) before it is merged. After WP29 merges, and again after WP38a
merges, `main` is merged into the lowest unmerged stack branch (and so carried up the stack), so WP38's diff stays its
own; the same holds for WP39 and
the DGP PR. Stacked PRs are merged with a merge commit (no squash or rebase merge), then the next PR is retargeted
(E159). Merging is the owner's | task brief | E58, E129, E159 |
| D-58 | (reserved) the GL default budget, decided by the owner from WP36's pilot (Q21) | — | — |
| D-59 | **Every exit criterion is pass/fail.** An unresolved probe, a missing fixture, a skipped parity test or a comparison that fails means the criterion is **not met**; the owner then decides, and any substitution needs a named SPEC §13 amendment. No internal oracle stands in for an external one silently; the release run of WP41 must show the parity tests ran (no skips) | the project rule against changing criteria | E34 |
| D-60 | **Stated tolerances.** Quantile comparisons use the Phase 2 parity rule (Conventions); KS tests use a **family-wise** `α = 1e-3`: a test function that runs `m` KS tests passes iff `min p > 1e-3/m` (Bonferroni, as `tests/mcmc/blocks/test_linear_fastpaths.py`), with `m` stated in the test (E124); moment comparisons use `|z| ≤ 4` with Bonferroni over the compared moments; every Tier U tolerance is stated with its test (§9.2). Nothing is compared before its rule is written here or in a pre-registration | P2 §9.9 | E38 |
| D-61 | **Separate Phase 3 validation registry**: `validation.joint` gains `COMPOSITIONS_P3`, `BLOCK_CONFIGS_P3` (and `joint_spec` accepts their names); the Phase 2 tuples and the suites that assert them stay unchanged | P2 §9.2 | E51 |
| D-62 | **Log-aware consumers.** Every reader of a coefficient prior's precision (failure diagnostics, implied prior moments, monitors) accepts `GaussianLogDiag`; implied reduced-form prior moments are reported "not applicable" under a GL prior (the horseshoe has no finite prior variance) | P2 §5.7, §8.4 | E53 |

### 0.4 Review errata (E29–E180)

`Ra`, `Rb`, `Rc` = the first-round reviewers and `R2a`, `R2b` the second-round reviewers (header), with their finding
numbers; `R3` = the merge-gate review of Amendment 1 (majors `R3-M1…M12`, each confirmed by a second reader; minors
`R3-m1…m26`, each checked against the code or the text before it was applied); `R4` = the merge-gate review of the
amendment (PR #16) and of WP26 (PR #17) (confirmed findings `R4-C1…C6`, minors `R4-m1…m24`). Every finding has a row;
related findings share one.

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
| E58 | Rb23, Rc17, Rc21: dependencies, bundled PRs, WP30's independence, a gap in the WP numbers | §13 redrawn: one PR per WP, consecutive numbers, named bases, WP30 and WP35 by an author who wrote no kernel (recorded in the PR) | §13 |
| E59 | Rb16: shard partition and timeouts; PR-tier budget | Two Phase 3 shards each for Geweke and SBC, the old shards `--ignore` the new files, shard edits in WP35, PR-tier increase measured and capped at +10 min per job | §9.11 |
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
| E86 | Ra28: M28 might be invisible at `T_eff = 20` | A11's quadrature (10⁵ draws, KS `α = 1e-3`) is M28's required catcher (superseded by E115: Tier U is) | §9.6 |
| E87 | R2b1, R2a14: the shared factor had no unbatched access to `e^{−h_t}`, and building `L_i` per equation costs `n·q²` memory | `EqContext.common_w` from the closed-over `glob.vol`; full `L` with masked forward-solve output and noise | §2.4, §5.3 |
| E88 | R2a1, R2b19: the shared-factor coupling dropped the dummy rows | `n_active = q_i + m`, the dummy sum of squares in `quad`, `coupling()` reused; mutant M36 | §5.3, §9.6 |
| E89 | R2a2, R2a7, R2a15: the GIG set-up was not in log space (NaN at seven grid points), small-argument log forms, `V = 0`, the double sign | `Λ`, `log α`, `t`, `s` and every term in logs; `lsh`, `le`; `V ∈ (0, 1]`; the sign once; grid to `log ω = −2000` | §3.5 |
| E90 | R2b2: many more Phase 2 tests need amendment than two | listed per WP in §1.3 | §1.3 |
| E91 | R2b3, R2a5: bvars returns no `h` draws; P16's quadrature was conditional, bvars's posterior joint | chained `estimate(·, S = 1)` extraction of data fields; joint 4-D quadrature under the P11–P14 mapping; a not-run branch; P12 on posterior `V` | §9.8 |
| E92 | R2a3: `scipy.stats.geninvgauss` is no valid oracle on the grid | KS of `log X − log η` against a log-space numerical CDF; mpmath in `dev` | §3.5 |
| E93 | R2a4: the lagged test priors exceeded the explosive-mass target 10–75× | NG `c = 0.01` (reduced), `0.005` (structural); SSVS slab `0.05√r`; ψ pinned; measured masses | §9.4 |
| E94 | R2a6: the λ autocorrelation omitted `m` and `−2` | `q_i/(T_eff + m + q_i + ι)` | §5.3 |
| E95 | R2b4: rule 5, the capability check, `resolve_const_path` and `_conditional` reject or mishandle `shared_factor` | exemptions, selection order, the weighted conditional branch, `const_path="cholesky"` → weighted | §2.1, §5.3 |
| E96 | R2b5: the API could not produce conjugate × CommonSV | `engine`, `static_parts` and the axes predicate spelled out | §7.2 |
| E97 | R2b6: GL resolution hits Minnesota-only code | WP31's adapter to an internal Minnesota; EB and dummies skipped | §3.6, §1.2 |
| E98 | R2b7: storage names collide or map wrongly | dimension `time`, explicit `glob.vol` mappings, `_accumulators` cases | §2.2 |
| E99 | R2b12: a static order-invariance rule cannot see `cross = contemp = 1` | keyed on `shared_factor`; `DiagnosticsInput.order_invariant` from resolve for Mode II × Constant | §5.3 |
| E100 | R2b8: the DGP exit gates size only | power and gate-pass share reported, as registered and as SPEC §10.5 words it; Q33 | §9.9, §12 |
| E101 | R2b9: the +10 min cap could not be measured and amended P2 silently | summed junit durations; priority list; P2 §9.10 amendment named | §9.11 |
| E102 | R2b10: bitwise checks only in WP26 | same-machine A/B in WP26–WP34 | §2, §13 |
| E103 | R2b11: WP31/WP32 acceptance used WP35's files | ad hoc runs through `validation.joint`, recorded in the PR | §13 |
| E104 | R2b13: ownership and branching contradictions; missing files | §1.2 note and files; WP29 merged to `main` before WP38; conditional tests in `tests/unit/` | §1.2, §13, D-57 |
| E105 | R2b14: a class-level `uses_global_vol` cannot depend on `common`; forecasts need the future | a property reading `static.common`; `gfut[G_COMMON_SV]` to `ahead` | §2.4 |
| E106 | R2b15: rule (15) "at fit" contradicted the single validation function | group rules in `CoefPriorStatic` and a resolve check; the four-argument signature kept | §2.1 |
| E107 | R2b16: monitor fields have no group kind | `group` kind; `monitor_fields` takes the coefficient-prior static | §2.4 |
| E108 | R2b17, R2b18: init order for the reduced form; `cccm_core`'s contract | `init_global` before every `global_pre` init; `cccm_core_log` as a new internal | §2.4, §3.7 |
| E109 | R2b20: warm start of `glob.vol.h` | `TIME_INDEXED`, key `eq_key(kc, n)` | §5.5 |
| E110 | R2b21: bayesianVARs pre-registration gaps | the P2 §9.9 data amendment named; P8 candidates listed; P7 mismatch → custom groups; P10 floor branch | §9.8 |
| E111 | R2b22: the R job in `ci.yml` would launch the weekly tier | weekly jobs gated on `!inputs.r_fixtures`; concurrency group | D-54 |
| E112 | R2b23: the pilot measured unvalidated kernels; stacking vs dependencies | suites (WP35) before the pilot (WP36); "Needs" and "Base" columns | §13 |
| E113 | R2b24, R2b25, R2b27: block-test file and shard; `rest` growth; SSVS `p_g` in D-52; cost comparison | named; measured; stated; Grams included | §9.4, §9.11, D-52, §5.3 |
| E114 | R2b26: the CommonSV level shift may mix slowly | R31; WP36 measures; Q34 | §5.3, §11 |
| E115 | R2a8, R2a9, R2a10, R2a11, R2a12, R2a13: ρ fallback threshold and ratio; A9 routes; M32's weight; M28's catcher; the conjugate test prior's reading; KS on correlated draws | each stated where it applies | §5.4, §9 |
| E116 | R2a16, R2a17: mixed index conventions; "exactly" citations | 0-based throughout (`h_0` the first data row); Chan (2020), not CCM (2016) | D-46, §5 |
| E117 | R3-M1, R3-m23: the `U(−1, 1)` fallback for `ρ` omits the TruncatedNormal prior ratio | The uniform branch is taken only under the Beta prior, when the full `K_ρ < 1e-8`; under TruncatedNormal the proposal is always the TN, so `log α = g(ρ*) − g(ρ)` is exact whatever `V_ρ` (revised by E158); pinned Tier U cases force the branch | D-48, §5.4, §9.2 |
| E118 | R3-M2, R3-m19: the bitwise A/B does not run the Phase 2 paths WP33/WP34 rewrite (the EB plug-in through `structural_nig.eb_structural`, the analytic engine) | The A/B covers the EB plug-in at the default `hierarchy="eb"`, with `fit.hyper` in the digest, `AnalyticFit` at `fixed`, `eb` and `mh`, and an analytic `ordering_sensitivity`, beside every reachable Phase 2 composition (§2; extended by E154). The free-cross refactor of `eb_structural`/`_Window` is bitwise at a fixed `cross` (the same arithmetic, `cross`/`contemp` passed per evaluation) | §2, §4.3, §13 |
| E119 | R3-M3: D-53's trigger list was narrower than P2 §11.4 (`Constant("ig")`, `Constant("iw_equivalent")` no longer triggered; `test_grid_harmonise[every-constant]` would fail) | P2's trigger kept; `CommonSV` triggers unless its form is explicitly conjugate; tests: the Phase 2 grids unchanged, `[Constant(), CommonSV()]` rewritten, `[conjugate × CommonSV, Constant()]` not | D-53, §7.5 |
| E120 | R3-M4, R3-M5: an always-on analytic `lag_coefs` multiplies large-`n` forecast memory by about `p` and moves the SPEC §13 Phase 1 n = 100 figure | Opt-in `lag_coefs=True` (option (a) of the review), requested by the integrated-LPD and conditional-forecast paths; a WP33 test that the default analytic forecast carries no `lag_coefs`, keeps `cond_chol.strides[2] == 0` and the same retained memory (tracemalloc current); E-16 names the keyword | D-49, §0.2 row 18, §6.1, §7.3, §13 |
| E121 | R3-M6: the bvars branch-2 gate compares two models that cannot be made equal; P13–P15 had no decision rule; a P16 failure was read as "approximate" | P13–P15 get stated rules (§9.8); a P16 failure is recorded as "mapping or sampler, undetermined" and cannot move the predictive into the gate; Q35 asks the owner what "bvars parity for CommonSV" means when branch 2 is selected (SPEC §10.4 prescribes comparing forecast distributions when no exact mapping exists; whether that comparison gates is the owner's) | §9.8, §12, §13 |
| E122 | R3-M7, R3-M10: the P1–P5 decision rules were deferred to the script, where P4 had none and P2/P3 always answered | Every probe rule (P1–P16) is written in §9.8 as WP29 implements it at `7f4e570` (re-pinned at `d281276` by E157); any later change needs an amendment before any parity comparison runs. P2 now separates `U⁻¹` from `2I − U` on a new `n = 3` dataset; P3 and P4 have bands and a unique-match rule. The rules were revised after discovery runs of the R side only, before any parity comparison; the run ids are in §9.8 | §9.8 |
| E123 | R3-M8: the SSVS inclusion comparison used a binomial MCSE (indicators are autocorrelated; the Rao–Blackwellised estimate is not binomial) | Each side's estimate is the mean of its per-draw indicator series `γ_j` with the MCSE of a mean, `sd/√ESS` (`diagnostics.mcse_mean_many`): the R side with the per-run ESS, our side with the chain ESS of the stored `γ` draws; Bonferroni as the Phase 2 rule; `p̂ ∈ {0, 1}` on both sides passes iff equal, on one side the other side's MCSE is used alone. The indicators `γ` are stored per draw (revised by E155: as `int8` with the stored parameter draws, not a monitor field; §3.7, D-52) | §3.7, §9.8 |
| E124 | R3-M9, R3-m24, R4-C2: the GIG KS grid (≈ 180 cells at `α = 1e-3` each) had no multiplicity rule (a correct sampler fails ≈ 16%); M34 was not specified | D-60's KS rule is family-wise (Bonferroni over the `m` tests of one test function); the §3.5 grid is one family whose cells each have their own random stream (the `−λ` mirrors included); the same rule for §6.3's per-horizon KS, A11 and P16 (its marginals × volatilities). M34 is defined (§9.6). **Order of events, recorded as found** (revised after R4): WP28 had already run the grid under the per-cell rule when the review's rule reached it, and one of 180 cells failed (`λ = 0, log ω = −50`, `p = 6.2e-4`), the ≈ 16% false failure the review predicted from the design alone (200 independent replications at that cell gave uniform p-values, KS `p = 0.49`; one 2·10⁶-draw sample `p = 0.67`). The grid's original keys, `fold_in(key(1), i·10⁴ + j)`, already gave every cell, the mirrors included, its own stream. WP28 first changed the keys (to `fold_in(fold_in(key(1), i), j)`) in the same commit as the rule (PR #19's "WP28 tests: the GIG KS grid as one Bonferroni family (review M9)", first pushed as `7da32d7`), and under those keys even the per-cell rule passed (min `p = 3.3e-3`), so the keys rather than the rule would have carried the outcome. The keys are restored to the original ones in PR #19's "WP28 tests: keep the GIG grid's original keys (E124)" (`6b8f433` on `phase-3-wp28` when this was written; stacked rebases change the hashes, the subjects stay): under them the family-wise rule passes (min `p = 6.2e-4 > 1e-3/180 = 5.6e-6`; `E log X` max `|z| = 3.95 < 4.03`) and the per-cell rule fails, so the only change after the failure is the review's rule, written from the design before the run was known to it. The owner decides on the rule with this record | D-60, §3.5, §6.3, §9.6, §9.7, §9.8 |
| E125 | R3-M11: P6's inference ran the wrong way (an internally standardising package is exactly equivariant, which the rule sent to "none") | P6 runs under a tight prior and compares the ×100 refit with (i) the equivariant image, (ii) the unchanged `PHI` and (iii) the image of a raw-unit-prior fit; (iii) ⇒ raw data, the SWA fixtures stand; (i) or (ii) ⇒ pre-standardised data in bayesianVARs' convention, which no probe establishes yet, so R1–R9 wait for a convention probe (an amendment); otherwise undecided | §9.8 |
| E126 | R3-M12: the default `scale="psi"` value of `r_j` had no catching tier independent of WP31's author | §9.2 item 5 (WP31's acceptance): `log r` against WP30's `gl_log_r_reference` (1e-14), `ŝ` against WP30's dense `ssvs_shat_reference` (1e-10) with the `T_eff` boundaries of D-44, and scale equivariance for every family in both forms. `ι` is unit-free, so only the value test pins it | D-37, §9.1, §9.2, §13 |
| E127 | R3-m1: A9's moment oracle cannot take posteriors (`implied_reduced_form_moments` is a prior-moment routine for diagonal arrays) | A9 compares `E[B] = (E[A]⁻¹E[Γ])ᵀ` (exact by row independence) and `E[Σ]` from a full-covariance recursion over the per-equation NIG posteriors (WP33) against `AnalyticFit` at 1e-10 | §9.7 |
| E128 | R3-m2: two Phase 2 contract tests changed without being listed | `tests/mcmc/unit/test_keys.py` (WP26, the frozen `Sub` values) and `tests/mcmc/blocks/test_cccm.py`'s `reads` assertion (WP31) join §1.3 | §1.3 |
| E129 | R3-m3 and the orchestrator: D-57's stacking mechanics were wrong (no automatic retargeting) | Retarget after each merge; carry `main` up the stack after WP29 | D-57 |
| E130 | R3-m4: E99 changes a Phase 2 warning (Mode II × `Constant` at `cross = contemp = 1`, `n ≥ 4`, loses the R8 ordering note) without listing it | Listed as an intended Phase 2 behaviour change of WP32, with a test that that fit no longer warns and that `cross ≠ 1` still does | §1.3, §13 |
| E131 | R3-m5: §4.2 gave `AsymmetricConjugate` the fit seed under `hierarchy="mh"`, against §7.2 and Phase 1 | Dropped: `uses_fit_seed` is `engine == "mcmc"` (the MH seed comes from `MHSettings` or `fit(seed=…)`, as Phase 1) | §4.2 |
| E132 | R3-m6: `CommonSVGlobal`'s scalar accumulators would be read as per-equation columns when `chains == n` (and `acc_h` collides with `SV`'s) | Renamed `acc_csv_h`, `acc_csv_rho`, `acc_csv_newton` (WP26); WP32 adds them to the monitor's per-chain scalars | §2.2, §5.4 |
| E133 | R3-m7: no seam gives `CommonSVStep.init` the residuals for the Newton start | `h` starts at 0 (a valid state; the first sweep's `h` step runs with fresh `q` before the `ρ` step). **Replaced by E175**: `h = 0` was not a safe start, and `h` now starts at a stationary prior draw | §5.5 |
| E134 | R3-m8: `pyproject.toml`/`uv.lock` (mpmath) and the `fit()` dispatch had no owner | `pyproject.toml` and `uv.lock` [WP28] (the lock regenerated, not hand-merged, on a rebase); `fit()` dispatches `AsymmetricConjugate` lazily to `StructuralNIGFit` by name [WP27], which WP33 provides | §1.2 |
| E135 | R3-m9: `ci.yml` was co-owned by an off-stack and an on-stack work package | `ci.yml` [WP29] (r_fixtures) and [WP41] (tiers) only; the Phase 3 shards live in `tools/weekly_shards.py` (WP35), which the plan job already reads | §1.2 |
| E136 | R3-m10, R3-m17: the planned SPEC §5.8/§10.4 wording ("CI never touches the network") is false for the weekly tier | WP41 writes: "No test job installs R; only the network-marked weekly tests download data (the pinned, checksummed datasets of §5.8); the PR tier never touches the network" | §0.2 row 24, D-54 |
| E137 | R3-m11: §1.1 did not say `forecast/conditional.py` is NumPy-only | Stated | §1.1 |
| E138 | R3-m12, R3-m14, R3-m26: Q28 was still open though the owner approved the full DGP run | Q28 resolved (owner, 2026-10-25); R30 mitigated; §9.9 names the registered run at the registration's commit; WP39 needs that run's report and the DGP PR merged; a harness change in the DGP PR before it merges triggers the registration's bug-fix rule | §9.9, §11, §12, §13 |
| E139 | R3-m13, R3-m18, R3-m22: internal staleness | §9.11 lists M19–M36; `benchmarks/reports/dgp_study*` protected; WP39 records the readings in this document's exit map (SPEC is WP41's); one dims name for `incl_prob`, `(variable, regressor)`; the Block ids 17–18 already exist, only `Sub` is appended | §1.2, §2.2, §2.4, §3.7, §9.9, §9.11 |
| E140 | R3-m15: four Phase 3 assignments of the Phase 2 notes were missing from §0.2 | Rows 26–29 | §0.2 |
| E141 | R3-m16: two deliveries depart from SPEC wording without a named edit; E-3 told a `Constant("ig")` user "only with Constant()" | SPEC edits named in §0.2 rows 5–6; E-3 reworded | §0.2, §7.3 |
| E142 | R3-m20: the DGP exit is read as met on one registered cell (`n = 3, p = 2, T = 200`) of SPEC §10.5's range without asking the owner | Q33 extended to that reading | §9.9, §12 |
| E143 | R3-m21: build logs of GPL packages can quote GPL source | The clean-room paragraph covers build failures | header |
| E144 | R3-m25: A10's oracle was owned by the author of the update it checks | WP30's author writes it (`validation/reference.py`, `conditional_forecast_reference`) | §9.1, §9.7, §13 |
| E145 | R3 (refuted, engineering): "the group rules cannot be met at construction" | No contract change: `api.py` passes a provisional `n_groups = 1` at construction (rules (11)–(14) do not read `G`) and `resolve.py` passes `n`; §2.1 now says only the value of `G` depends on `n`. The two other refuted findings (the R6 coverage reading; bvars branch 2 being SPEC-prescribed) change nothing; the second informs Q35 | §2.1 |
| E146 | WP29 discovery run 40226725087: bvars's ncSV sampler fails on the FRED-QD small set in levels (an `iwishrnd` error) | The bvars runs use `qd3_growth.csv` (growth rates: 4Δ of 100 × log `GDPC1`, `GDPCTPI`; `FEDFUNDS` in levels; 1960Q2–2019Q4), `n = 3`, `p = 4`; amends §9.8's "FRED-QD small set (federal series)" | §9.8 |
| E147 | WP29's implementation of §9.8 | Recorded: the R side installs the snapshot's Linux binaries (D-54); P12 holds trivially (bvars 1.1 never moves `V`, even under its default prior, so the runs keep the default starting `V = Ω`); under an untruncated `ρ` prior the P16 posterior at `T = 2` may be improper, so WP38 establishes propriety before any KS verdict; R7 uses SSVS `c1 = 1` (chosen from mixing before any comparison); at the discovery runs' size both volatilities take branch 2; the `r_fixtures` dispatch skips the `lint` and `test` jobs | D-54, §9.8 |
| E148 | WP30: under the §9.4 CommonSV test prior `ρ ~ N(0.8, 0.15²)·1(|ρ| < 1)` the density is positive at `ρ = ±1`, so `Var(h_t) = E[σ_h²/(1 − ρ²)]` is infinite | On every TruncatedNormal-`ρ` row (320–324, 328, 329, blocks 407, 408) the unbounded test functions (`h` at three dates, mean `h`, the exact log-likelihood, the log-scale data functions) are SBC-only (ranks need no moments); Geweke uses bounded transforms (`atan` of `h` at the three dates and of mean `h`) beside the other §9.4 functions. The PR-tier rows 320 and 321 are correspondingly weaker in `h` than §9.4 read | §9.4 |
| E149 | WP30: `compose` accepted a `VolStatic(common=True)` static and built a Phase 2 program without the common-`h` step, so before WP32 `BVAR(vol=CommonSV()).fit()` on the stacked branches would have run a Constant-volatility model under the CommonSV label | `compose` raises `NotImplementedError` unless the volatility block brings a `G_COMMON_SV` step (WP26) | §2.4 |
| E150 | WP30's explosive-mass measurement (10⁶ draws per size, seed 0; `validation.joint.EXPLOSIVE_MASS_P3`), revised after R4-C3. D-56's 2e-4 is Phase 3's target (E46, E93: the level Phase 2 measured for its corrected prior at `n = 3, p = 2`); Phase 2's bound is `P(ρ > 1) < 10⁻³` (N06 §9.1), which every row meets (largest 2.39e-4). D-56 states no Monte Carlo rule; the rule is now: a row exceeds the target iff its count is above the upper 99% Poisson limit at the target (234 of 10⁶ draws; ≈ 2.17e-4 at 4·10⁶). Rows 320 (238) and block 408 (239) exceed it, from the SOC/SUR rows of the conjugate prior (≈ 1.2e-4 without them, rows 325 and 329); row 321 (207; Phase 2's own test prior, reproducing Phase 2's measured level, 2.04e-4 ± 0.07e-4 at 4·10⁶) and row 323 (204) are within it. Rows 319 and 330 measure 0, against R2a's 4.6e-5 and 1.3e-5; an independent prior simulation in R4 confirms WP30's 0, so the gap lies in R2a's simulation | Not retuned: the test priors stay as §9.4 states; rows 320 and 408 are strict expected failures of WP30's target test; Q36 asks the owner, presenting the Monte Carlo rule as the definition of a criterion D-56 left incomplete (not a loosening); the target stays 2e-4 (moving it to `10⁻³` now that the results are known would loosen it). §9.4's figures for rows 319 and 330 are replaced by the measured 0 (R2a's kept as superseded) | §9.4, §12 |
| E151 | WP30: where §9.3–§9.4 were silent | Recorded as WP30 chose: row 325's parity priors are P2 §9.2's SV values (`Beta(10, 2)` on `(ρ + 1)/2`, `ScaledChi2(0.1)`); row 320's dummy rows are SOC for every variable plus SUR (the Phase 2 rows); blocks 401–405 and 407 run only the named production global step (`kernel="global_step"`; their held blocks run before the live post step, so P2's restore after a full sweep would not be exact), 401–405 redraw the coefficients from `p(d | scales)` with zero data, and the hyperprior variants of 401–405 and the families of 406–407 are named in `BLOCK_CONFIGS_P3`; D-56's "a column of ones for each deterministic term" would be collinear with a trend, and every row has `n_det = 1`. For WP32 to confirm: the common step receives `EqOut.log_lam = log λ_i + h_t`; the block-401–405 coefficient redraw keys on `Block.EQ_COEF_LOCAL` (unused by Phase 3 steps); the densities are with respect to the natural scales and `σ_h` | §9.3, §9.4 |
| E152 | WP28: design details | Recorded: the GK grid's "middle point" is both middle points (0.5 and 0.501); `E log X` uses `|z| ≤ max(4, z_{1−0.01/(2m)})` (= 4.03 at `m = 180`) as the Bonferroni rule over the compared moments; "1e-10 relative on `dev_log`" is relative, and "representable" is `ε(|θ| + |d|) ≤ 1e-13·|d|`; §10's GIG estimate (0.2–0.6 ms per sweep at `n = 20, p = 4`) is off: 7.6 ms per 6,400-lane call measured, 2.6 ms of it the limit-case gamma draw that `vmap` evaluates on every lane (WP31 may take the limits out of the production path; WP36 re-measures); §5.4's `K_h` diagonal at `T_eff = 1` is `(1 − ρ²)/σ_h²` (WP32); the `csv_newton_start` seam sets both the start and the iteration cap, as M28 changes both (WP32) | §3.5, §5.4, §9.2, §9.6, §10 |
| E153 | R4-m20: rule (11) did not reject SOC/SUR dummy rows with a varying coefficient prior (reachable through engine-level `StaticConfig`s, not the API) | Rule (11) also needs `MeanStatic.n_dummy = 0` (GL priors have none, D-36); tested in `test_contracts_p3` (WP26) | §2.1 |
| E154 | R4-C5, R4-C6, R4-m21–m24: the A/B tool ran 16 of the 23 Phase 2 composition rows while claiming "each Phase 2 MCMC composition"; the A/B record had no digests, machine or head; two contract defaults and resolve-level changes were unguarded; one Phase 2 test amendment was unlisted | `tools/phase3_bitwise_ab.py` builds its fixed-hierarchy cases from `validation.joint`'s registry (every row the public API can build: 22 of 23, row 17 `zero_sv` has no public mean) plus the option, EB-plug-in, default-`eb`, analytic and ordering cases of §2 (39 cases; `gram="batched"` runs on the two paths that read the Gram layout, after a re-check found it on the eigen path, where it equalled row 1, and `run` now refuses two MCMC cases with the same posterior digest), and a contract test checks the row coverage; the baseline is re-recorded twice on `552fe30` (sha256 `323d68b3…`); each PR records its head commit, the machine and the sha256 of both JSON files. New WP26 tests: every new leaf defaults to `Empty()` in constructor calls (`EqContext`, `CoefPriorArrays`, `SharedPrior`); a golden of the resolved `StaticConfig` of the MCMC golden models recorded at `552fe30`. §1.3 lists the `test_config.py` unknown-kind change | §1.3, §2 |
| E155 | R4-m18: E123's per-draw `γ` was unpriced, and D-52 listed local scales as secondary monitors while §3.7 said they are not stored | `γ` is stored as `int8` with the stored parameter draws (thinned like them; `n·k` bytes per stored draw, §10), and WP38's ESS-based MCSE uses those draws; local scales are latent, neither monitored nor stored | D-52, §3.7, §9.8, §10 |
| E156 | R4-C4: row 28 claimed the reduced-form runs R1–R9 are order-invariant (Cholesky SV and `Constant("ig")` are not), which dropped the per-ordering parity check without the owner question R3-m15 asked for | Row 28 deferred pending Q37 | §0.2, §12 |
| E157 | R4-C1, R4-m1, R4-m11, R4-m12: the §9.8 pin was not accurate about which runs an undecided probe blocks (the script stages R1/R2 when P6 is unresolved and wbvars_* whenever P15 returns a row map, and withholds R2 when P4 is undecided and the SV runs when P5 is), the P13 (c)/P14 rule accepted conditioning variants the row did not name, branch 1's settings were unstated, and one fixture attribute stated a band the code does not use | "Not met" is applied at comparison from each fixture's `probe_answers` and `SV_PARAMETRIZATION.md` (D-59), not by the script's staging. WP38 applies each probe row's own consequence (revised after the re-check, which found that a flat "P11–P16 for wbvars_*" list contradicted the P12–P14 and P16 rows and decided Q35 in advance): P1–P3 and P6 for every R run, P4 for R1 and R3–R9, P5 for R2, P7 for R8, P8 for R4–R6 and R9, P10 for R3–R9, P9 for semi-automatic SSVS runs only, P11 and P15 for wbvars_*; P12–P14 choose the branch (unmapped ⇒ branch 2, whose status waits on Q35); P16 decides only branch 1's `h_t, ρ, σ_h` part; P8 and P16 count as answered once their later decision is recorded in `SV_PARAMETRIZATION.md` (the recording work package amends WP29's schema test so that, for those two probes only, a fixture's "pending" may stand beside the recorded decision); the fixtures from the `d281276` final run replace the first run's in PR #21, which checks their decisions. Branch 1 and the P16 row now agree that an unresolved P16 leaves that part not met, stricter than the merged design's "reported, not gating"; R21 matches. The exit map reads "after P1–P6". WP29 restricts the P13 (c)/P14 decision to the variants the rows name (the others are evidence), passes the mapped branch-1 settings (the P12 `V`, the P14 `(sv_a, sv_s)`), corrects the P3 attribute text and the P6 staging note, gates staging on the rows' dependencies (P8 and P9 gate at comparison only), and re-pins at `d281276`; recomputed from the final run's recorded p-values, the P13/P14 decisions are unchanged (cSV: only `U(−1, 1)` with `h_0 = 0` passes, `p = 0.058`; ncSV: no candidate passes), and discovery run 40242687927 checks the new records | §9.8, §13 |
| E158 | R4-m2, R4-m3, R4-m16, R4-m17: "possible only under Beta" fails when `V_ρ > 10⁸`; the pinned case had `S_xy = 0`, so it could not catch a missing data term in the uniform branch | The uniform branch is a Beta-only branch (§5.4, D-48); two pinned Tier U cases: the all-zero `h` under both priors (TruncatedNormal shows the TN proposal), and under Beta a state with `S_xx/σ_h² = 10⁻¹⁰` and `S_xy/σ_h² = 10⁻²` (the uniform proposal with a non-zero data term) | D-48, §5.4, §9.2 |
| E159 | R4-m4/m14, m5, m6, m9, m10, m15, m19: text consistency | D-54 and §0.2 row 24 carry E136's wording; R21's mitigation names P12–P15 as selecting the comparison and P16 as deciding only the `h_t, ρ, σ_h` part; D-57 requires merge commits, and §13's WP26 and WP38 bases are corrected; §1.3 lists `tests/integration/test_ordering.py` (WP32, the CommonSV false-alarm control of row 26) and the weekly ordering-invariance test (WP35, row 27), which also join the WP32/WP35 rows and §9.11; `score_forecast` drops out of D-49's plumbing; P16's KS is one family; §3.5's `E log X` bound matches E152; §9.4's masses for rows 319 and 330 are the measured ones | D-49, D-54, D-57, §1.3, §3.5, §9.4, §9.8, §9.11, §11, §13 |
| E160 | WP33: A9's projected-gradient criterion (< 1e-6 at the EB mode with free `cross`) failed with Phase 2's L-BFGS-B tolerances (`ftol = 1e-12`, SciPy's default `gtol = 1e-5`): 4.5e-6 (`lambda+cross`) and 1.4e-6 (`+contemp`) on A9's window, while Phase 1's stationarity rule passed | **Q38, resolved by the owner (2026-10-25): "Toleransları sıkılaştır (Recommended)"** (tighten the tolerances). When `cross` or `contemp` is free, `eb_structural` runs L-BFGS-B with `ftol = 1e-15`, `gtol = 1e-9`; A9's projected gradients are 2.4e-8 and 7.3e-9 and the criterion stays 1e-6. The Phase 2 plug-in (both fixed) keeps its settings, bitwise (the 39-case A/B of WP33's PR) | §4.3, §9.7, §12 |
| E161 | WP33: with `contemp` free, the EB mode sat at the 1e-3 bound on 3 of 4 simulated windows (weak contemporaneous structure; A9's correlated window gave 0.18) | Recorded: the bound can bind, the projected gradient treats it as active, and the default free set (`lambda`, `cross`) excludes `contemp` | §4.3 |
| E162 | WP33's choices where §4 was silent | Recorded: (a) `MCMCFit.forecast` has no `lag_coefs` keyword (it always attaches them), so `backtest` and `ordering_sensitivity` pass the keyword through `models.base.forecast_options` only to forecasts that accept it; WP32 adds the keyword to `engines/mcmc/results.py`; (b) the EB start is the conjugate twin's Phase 1 mode for the GLP names and the spec's value for a free `cross`/`contemp`; (c) `psi` free is accepted only at `cross = contemp = 1`: E-15 rejects a free or non-unit `cross` at construction, and a free or non-unit `contemp` raises a `ValueError` at fit (moving that check to construction would change WP27's frozen E-15 text; a follow-up); (d) the R8 note fires for `n ≥ 4` when the fitted `cross ≠ 1` or `contemp ≠ 1` (the NIW correspondence needs both, as E99/E130 for MCMC); (e) the flat log prior is a constant in the EB objective and the natural-scale density in the MH target; (f) `lambda_alpha` accepts `cross`/`contemp` entries equal to the prior's; (g) new tests beyond §1.3: `tests/analytic/test_lag_coefs_on_request.py`, `tests/integration/test_ordering_p3.py`; A9's MCMC leg is in WP33 as `slow` (§1.3 named WP37) | §1.3, §4, §7.3 |
| E163 | WP31: the scale step costs 10–30 × §10's estimate; a kernel module computed with JAX at import time; slow mixing of a horseshoe group scale | Recorded: (a) per sweep of 4 chains at `n = 20, p = 4` (6,400 lanes): HS 6.0 ms, NG 7.8 ms, R2D2 17.6 ms, DL 15.5 ms, SSVS 0.28 ms (the equation phase 3.6 ms): each GIG line ≈ 7.4 ms, of which ≈ 2.6 ms is E152's limit-case gamma draw evaluated on every lane; HS dominated by its two shape-1 `loggamma` lines (≈ 2.6–3 ms each); measured on a shared machine (load ≈ 3). Two options in `random.py` (WP28's file), offered to the owner in PR #23 and not taken: (i) evaluate `log_gig`'s limit branches only on the lanes that need them (≈ 2.6 ms per GIG line); (ii) draw the shape-1 inverse gammas as exact exponentials, which needs a static shape that the seam's array parameters do not carry. WP36 re-measures on an idle machine whether or not either lands (§10). (b) §1.1 gains a rule: kernel modules compute nothing with JAX at import time (a module first imported inside a traced step otherwise fails; fixed in WP31 with a subprocess regression test). (c) The horseshoe's cross-lag-2 group scale mixed slowly (ESS 6 at 2 × 200 draws): R18, for WP36. (d) On E150/Q36: WP31's readings of NG's `c` and SSVS's `τ` agree with WP30's (density ratios on WP30's arrays, the monitored log prior equal to `log_joint.log_prior` to 1e-10, rows 319 and 330 green at G-small), so the gap to R2a's masses is not a disagreement between the references and the kernels | §1.1, §10, §11 |
| E164 | WP31's choices where §3 was silent | Recorded: (a) init (§3.7): "local scales start at 1" read as unit local multipliers, so `r_j v_j = 0.04 r_j`; the HS auxiliary `τ_g` starts at `1/s²`, jittered; (b) `gl_a_conditional` draws the whole collapsed block and also returns its log weights, so M22 (the weights) and M35 (the order) both reach it; (c) the SSVS indicators are stored in the `GlobDraw.coef` slot as `SSVSDraw(incl)` and saved as `posterior/ssvs_gamma` (`int8`, thinned like the stored draws, E155): the name `gamma` would collide with the structural coefficients in storage; `state.draw_template` (WP26) still shows `Empty` in that slot and gains `SSVSDraw` for the SSVS kind in WP32, which owns the next `state.py` change; (d) `CCCMStep.writes` is a property adding `glob.mean.dev_log` only under `coef_log`, so P2's `writes` assertion stands (the only `test_cccm.py` amendment is E128's `reads`); (e) the GL log prior is evaluated from the full chain state (`GLScaleStep.log_prior_state`), `log_prior_eq` returning 0, since the scale densities couple the equations; `CCCMStep.log_prior` refuses the deviation path; (f) monitor fields carry a `gl_` prefix (`gl_p`, `gl_a_idx` as a float index) where `a` and `log_p` already name Phase 2 fields; the HS `τ` is secondary; (g) fit-level tests stay in `tests/mcmc/blocks/test_global_local.py` (§1.3's names); WP31's 138 PR-tier tests add 183 s of summed junit time (Phase 3 so far: ≈ 4.9 of the 10 minutes of §9.11 without WP33's 10.8 s, ≈ 5.1 with it; WP29's PR-tier schema tests are not yet measured) | §1.3, §2.2, §3.3, §3.7, §9.11 |
| E165 | WP29's final fixture run (40243289623 at `d281276`) against the first (40234562983) | Recorded: every probe's status and answer and the gating are identical between the two runs; many fixture numbers differ at Monte Carlo level although the seeds, snapshot, R, BLAS and package versions are the same (the R side does not reproduce bit for bit from its seeds; threaded BLAS rounding amplified by the samplers is suspected, not checked). bvars's `Σ` fails its hard mixing check in both volatilities at §9.8's draw counts (cSV max R-hat 1.013, min ESS 570; ncSV 1.028, 198; the run exits 1); R9's `a` never left the grid point 0.05 in this run, so its fixture records it as constant (the schema test accepts both layouts); P14's fixing scan setting `sv_a × 10⁶, sv_s × 10⁶` now stops with an `iwishrnd` error for cSV. Q39 (resolved: option (b), E167) settles the bvars mixing | §9.8, §12 |
| E166 | Amendment 2's process | WP33 and WP31 were written in parallel on branches from WP30 (their files are disjoint) and rebased by the coordinator; WP33's PR targets `phase-3-wp30` until WP32 exists (D-57) | §13 |
| E167 | The owner's decisions on Q35 and Q39 (2026-10-25: "Q35 kanıt olsun, Q39 için (b)") | Q35: §9.8's branch 2 (the bvars CommonSV predictive comparison when no exact prior mapping exists) is reported as evidence and does not gate; under it the exit criterion "bvars parity for CommonSV" is not met for want of an exact mapping. Q39: the bvars runs keep §9.8's draw counts; WP38 treats a fixture whose hard mixing check fails as not met for the quantities it covers (at run 40243289623 both bvars volatilities fail it for `Σ`). Separately from Q39, which was asked about the bvars runs only: a compared quantity whose chain never moves (R9's `a` in that run, on one grid point) cannot pass §9.8's mixing rule and is not met (D-59); WP29 records it per quantity (the fixture's mixing record and `SV_PARAMETRIZATION.md` §7.1), and a fixture-level `mixing_ok` stays tied to the blocks it covers. These readings are of run 40243289623; PR #21 regenerates its `n = 7` fixtures after its review, and the amendment that records the new run restates them (R5) | §9.8, §12, §13 |
| E168 | Q40 (owner, 2026-10-25: "Bağımsız kesin oracle yaz (Recommended)", write an independent exact oracle): under Q35 the SPEC §13 "bvars parity for CommonSV" could not be met, since branch 2 applies to both volatilities | §9.8 specifies an exact `CommonSV` oracle (model; three exact blocks, each a different algorithm from the engine's, block 3 with a non-centred `σ_h` move; independence rules; its own Geweke and closed-form validation; runs C1–C2; the comparison and its tolerances), written in WP38a before WP38 by an author independent under §9.8's rule; SPEC §13, §10.4 and D2 are amended to name it, with bvars as evidence. **Revised after R5**, before any oracle or engine output existed: (a) the oracle runs one fixed length, 4 × 200,000 kept sweeps after 10,000, and the conditional four-fold lengthening is deleted (it contradicted §9.8's no-lengthening rule and was not put to the owner); (b) block 3 gains the non-centred `σ_h` move (R5's scratch oracle on C1 at 4 × 50,000: `σ_h` bulk ESS 62 and split R-hat 1.08 with the centred step alone, 420 and 1.012 with the move, so both are adopted); (c) the criterion is decided only on the engine's weekly run (never lengthened; a failing gate is not met; its length, 4 × 200,000 after 10,000, is the owner's answer to Q41, E169), and the PR short run is a smoke test; (d) the no-lengthening rule is stated for every §9.8 run; (e) WP30's author may not write the oracle, and WP35's only before reading the listed code; (f) the dummy rows, C2's generating process, the predictive simulation, the validation inputs and a Phase 1 cross-check of block 1 are stated | §0.1, §0.2, §1.2, §1.3, §9.8, §12, §13, D-57, SPEC §10.4, §13, §14 D2 |
| E169 | Q41 (owner, 2026-10-26: "Sabit 4 × 200.000, şimdiden (Recommended)", fixed 4 × 200,000 from now): at 4 × 50,000 the engine's decisive `CommonSV` run could fail §9.8's mixing gate for `σ_h` from mixing alone | C1 and C2 are decided on engine runs of 4 chains × 200,000 kept after 10,000, the oracle's length, fixed before any engine or oracle output exists, unconditional and never lengthened (D-35's precedent: a fixed higher count, not sampling until a gate passes); a gate failure on either side is not met (Q39(b)'s rule, §9.8). Cost ≈ 1.1 ms per 4-chain sweep at C1's size on the development machine, so ≈ 4 min per run; the 2-vCPU weekly runner is slower. **Finding, not a change:** the engine's centred `σ_h² \| h, ρ` step (§5.4) mixes slowly when `ρ` is near 1: in R5's scratch runs the centred `σ_h` IAT is ≈ 374–446 on C1 with a nearly exact `h` (50 elliptical slice updates per sweep), the centred floor; ≈ 580 on a C2-like posterior comes from a run with 5 updates per sweep, which the `h` update bounds, so it is not that floor; an input for WP36's pilot and Q34's interweaving move | §9.8, §12, §13 |
| E170 | WP32: the `CommonSV` step's cost, slow Mode I level mixing, and block 407's `W`/`rss_std` functions | Recorded: (a) at `n = 20, p = 4, T = 300`, 4 chains vmapped: the `h, ρ, σ_h` step costs 0.87 ms per sweep (5 blocks of 100 dates); the shared-factor equation phase costs 0.61 ms against 2.20 ms on the weighted path, so a standalone conjugate × CommonSV sweep costs 1.52 ms, against §10's 0.3–0.6 ms; inside the runner's `burn` loop it costs 3.88 ms (a Phase 2 runner property: Mode II × Constant 0.33 ms standalone, 1.22 ms in `burn`; larger here, presumably the Newton `while_loop`s). WP36 re-measures; (b) in 4 × 2,000 fits, Mode I × CommonSV has `log λ_i` bulk ESS ≈ 21 (R-hat 1.12) and `common_h_T` ESS 27, against ≈ 160 and 286 on the shared-factor path on the same data: R31's level shift mixes slowly in the per-equation program (WP36; Q34); (c) under the TruncatedNormal `ρ` prior `Var(h_t) = ∞` (E148), so block 407's `joint.W` and `eq*.joint.rss_std` have Geweke means of order 1e66–1e83 and `z ≡ 1`: they carry no power, and WP35 flags them as E148's SBC-only functions on the TN rows (WP30's harness; the bounded `atan` functions of `h` behave normally); (d) E158's second pinned `ρ` case passes at 1.5e-11 against the 1e-10 rule, within about 7× of float64's floor at that state (both evaluations share the large terms in the same order); nothing was changed | §5, §9.1, §9.6, §10, §11 |
| E171 | WP32's choices where §5 was silent | Recorded: (a) `csv_lambda_conditional(theta, rows) -> Coupling` lives on `LinearStructural`, where the θ part and the dummy-row part are still separate (the `Coupling` contract carries totals only, so a seam on the core could not drop the rows' part), and WP35 injects M31/M36 through the program's mean (`program.mean.impl`); (b) `VolBlockImpl.extract_global(glob, T_eff)` (the runner passes `T_eff`, since `h_T` is a dynamic index), and `CommonSVStep.log_prior_data(glob, prior, data)` as the monitor's third branch; (c) monitor names `common_h_T`, `common_rho`, `common_sigma` (primary) and `common_h_0`, `common_h_mid` (secondary); §2.2's `posterior/common_h_T, rho, sigma` read as three names; (d) `σ_h²` starts at the IG prior mean, or at its mode when `c₀ ≤ 1`; (e) a warm start's new first row with no old predecessor is drawn from the stationary law with the copied `(ρ, σ_h)`; (f) the SOC/SUR rows enter the shared factor scaled by `√dummy_w` (1 in Mode II), checked equal across equations with rule (14); (g) an unlisted WP26 test amendment: E149's guard test composes `CommonSV` with a block that brings no step, and checks that the production composition carries it; (h) `state.draw_template` shows `SSVSDraw` (E164(c)); (i) the R8 note no longer fires for Mode II × `Constant("iw")` at `cross = contemp = 1` (E130, intended); (j) `MCMCFit.forecast` accepts `lag_coefs` and changes nothing (E162(a)) | §1.3, §2, §5, §9.6 |
| E172 | WP29 after PR #21's review: `bv_svpara` recycled P4's `n = 3` equation order over seven columns, so the `sv_*` blocks of the eight `n = 7` bayesianVARs SV fixtures were wrong (their `Π`, `A`, `h_t` and predictive were not) | Recorded: `make_r_fixtures.R` changed after the `d281276` pin, in extraction and its records only (`73e3211`: the data order at any `n`, any other permutation stops the run, a dimension check, extraction warnings fatal; hard mixing checks of the hyperparameter scales, the SSVS indicators and R9's grid, with `mixing_covers` and `mixing_failed`; `git_commit` and `ci_run` per fixture). Run 40261716464 at `73e3211` regenerated those eight and `stochvol_v_t5` at unchanged draw counts. After E173, run 40351238342 at `56fcc5c` regenerated all 14 bayesianVARs SV fixtures, those eight among them. The fixtures of record are therefore 14 from 40243289623, `stochvol_v_t5` from 40261716464 and 14 from 40351238342 (`SV_PARAMETRIZATION.md` §7). §9.8's pin moves to `56fcc5c`. The probes gave the same status and answer for P1–P10. Outside the blocks each change touched, the runs pair up bit for bit: 40261716464 with the first final run 40234562983, and 40351238342 with 40243289623. The two pairs differ at Monte Carlo level (suspected to depend on the runner, not explained; E165). So the `n = 7` parameter blocks and the `h = 1` predictive of record are run 40351238342's realisation, a different Monte Carlo realisation from the `7d5aef0` fixtures of 40261716464. Run 40351238342 is the `n = 7` reference of record for WP38. R9's `a` visited two grid points without mixing in 40261716464 (indicator ESS 239, R-hat 1.015) and never left 0.05 in 40351238342, the run of record. Either way it is not met (E167). WP38 applies Q39(b) per entry of a fixture's `mixing_failed`, not to the whole fixture through `mixing_ok`: R9's `mixing_ok = 0` comes from `a` alone, so R9's `Π`, `A`, SV parameters, `h_t` and predictive are still compared. The Phase 2 weekly `test_bayesianvars_parity`, run unchanged on a local merge, fails `n3p2_sv` on 2 of 300 forecast quantiles (h4:y1 q75 `z = −6.35`, h3:y1 q75 `z = −4.82`). E173 traced this to bayesianVARs' `h ≥ 2` predictive. On the regenerated fixtures the test, unchanged and not skipped, passes all three of its fixtures: max `\|z\|` 3.34, 2.76 and 3.62 against bounds 4.05, 4.15 and 4.51 | §9.8, E157, E165, E167 |
| E173 | Q42 (owner, 2026-10-26: "Kendi çekimlerinden yol ile hesapla (Recommended)", compute it from its own draws with the path): the Phase 2 weekly `test_bayesianvars_parity`, run unchanged on a local merge of PR #21, failed `n3p2_sv` at `h ≥ 2` (E172) | Recorded in the order it happened. (1) The discrepancy was found first, before any Phase 3 parity comparison ran. (2) The orchestrator's diagnosis, confirmed by a second reader, placed it on the oracle side. bayesianVARs and Phase 2's exact SWA oracle `bvars_swa_n3p2_sv` (same data, md5 `7b588f5a…`, same prior) disagree on their own at h4:y1 (q75 `z = +4.52`, q25 `−4.17`), while every parameter block agrees. On the same rai-bvar draws the test passes against SWA (max `\|z\|` 3.1–3.5 over 4 seeds and at 4 × 50,000) and fails against bayesianVARs, with `z` growing as the Monte Carlo error shrinks (a fixed offset). The constant-volatility pair agrees (max `\|z\|` 2.94). Drawing each horizon's log-volatility from its `s`-step marginal, independently across horizons, reproduces bayesianVARs on all 200 cells, and the path recursion reproduces SWA. So `predict(…, simulate_predictive = TRUE)` appears not to return the joint-path predictive at `h ≥ 2` (a black-box inference, not confirmed). (3) The owner decided, and the rule is written here before any regenerated fixture is compared (D-59). For `h ≥ 2`, the R-side SV predictive of every bayesianVARs SV run is computed in `make_r_fixtures.R` from bayesianVARs' own posterior draws (per draw `Π`, `A`, the SV parameters and `h_T`) by the path recursion `h_{T+1..T+H}` and then the VAR recursion, as Phase 2's `swa_forecast` does. It is written from SPEC's model, not from rai-bvar's forecast code. `predict()`'s native output stays in the fixtures as evidence, not compared. `h = 1` and the parameter blocks are compared as before. The fixtures are regenerated through the dispatch-only `r_fixtures` job (D-54, option (a)). No tolerance, prior, draw count or test changes. The v0.2.0 release record and the paper are unaffected: Phase 2's parity of record is against SWA. (4) The rule was implemented in `56fcc5c`. It draws one path per kept draw, as `swa_forecast` does. The path's own `h = 1` is checked against `predict()`'s (max `\|z\|` 1.37–2.75, a hard check). That check treats the two Monte Carlo errors as independent, although both come from the same draws, so it is lenient. The independence of the `h ≥ 2` reference holds in the draws (bayesianVARs' own) but not in the recursion code, which mirrors Phase 2's `swa_forecast`. rai-bvar's separate JAX forecast agreeing with both closes that gap (the path against SWA: max `\|z\|` 2.13; `predict()` against SWA: 4.52, kept as evidence; PR #21's re-check). `predict()` against the path at `h ≥ 2`, kept as evidence only: max `\|z\|` 6.3–7.6 at `n = 3` and 2.7–4.9 at `n = 7`. The fixtures were regenerated by run 40351238342 with unchanged probe answers (E172). A follow-up adds two assertions to WP29's schema test: `forecast_h1_check_pass == 1`, and that the `h ≥ 2` `forecast_q` differs from `forecast_predict_q` | §9.8, §12 |
| E174 | Q43 (owner, 2026-10-26: "(b) Daha çok çekim, aynı seed (Recommended)", more draws, same seed): A10's statistical part failed on the analytic and structural-NIG setups at the pre-registered seed with 1,000 draws (WP34, PR #28) | Recorded in the order it happened. (1) Before the first run WP34's test fixed `SEED = 20261026`, `DRAWS = 1000`, the conditions, the statistic and the bound. (2) Both analytic setups failed on one cross moment (`rate` at `h = 2` × `gdp` at `h = 3`): max `\|z\|` 4.11 (`AnalyticFit`) and 4.35 (`StructuralNIGFit`) against 4.00. The two fits share the forecast's shocks, so it is one event; the exact parts (≤ 5.5e-14) and the MCMC setups passed, and the two cases became strict xfails. (3) An evidence run, not adopted: over 400 other seeds of the analytic setup the rule fails in 4 (1.0% family-wise; exact 95% interval 0.27–2.5%, one-sided `p ≈ 0.05` against ≈ 0.34% from normal theory at `m = 54`), with the product statistics mildly skewed at 1,000 draws. (4) The owner decided: the two analytic setups of A10 run at 10⁴ draws, as A10b does, with the seed 20261026, the statistic and the bound unchanged. The reason given is the evidence run's failure rate at 1,000 draws (4 of 400, above nominal but with an interval that contains it), not the failure itself. The outcome is unknown beforehand and binding: a setup that still fails at 10⁴ is not met for this check, with no further change. (5) The result, from WP34's `f390060`: max `\|z\|` 2.649 (`AnalyticFit`) and 2.432 (`StructuralNIGFit`), with `m = 54` and bound 4.00, so both pass and the strict xfails are removed. As implemented, every test of these two setups runs at 10⁴ draws through the shared `leg` fixture (stricter, not part of the decision): on both, the exact parts, §6.3's per-horizon KS check and the result-field checks; on the analytic setup also the M33 test, the determinism test and the base refusal. The MCMC setups stay at 1,000 draws. On each CI machine they are a fresh draw of the ≈ 1% false-failure risk (MCMC draws are not bitwise across CPU models); that is a standing risk, not decided here. The module's summed junit time is 110 s on a heavily loaded machine. Also from WP34: `engines/mcmc/results.py` gains `MCMCFit.conditional_forecast` (§6.1, §1.1), which §1.2 now lists | §1.2, §9.7, §12, §13 |
| E175 | BR (the batch review of the Phase 3 PRs, 2026-10-26), PR #25: with `sigma2_prior = ScaledChi2` a `CommonSV` fit failed at its cold start whenever the first `h` step rejected every block, at any `T_eff`: 13 of 40 and 8 of 20 fits at `T = 60` (below the block length) and 11 of 20 at `T = 150` (above it), 4 chains; `InvGamma` 0 of 40; `block = 7` 0 of 20. E133's `h = 0` start: when the first `h` step rejects every block, `S = 0`, the conditional `GIG(½ − T/2, 1/B, 0)` is improper and `log_gig` returns NaN (`MCMCNumericalError` in `G_COMMON_SV` at sweep 0) | The init is fixed, not the kernel (`S` is not clamped: the conditional really is improper there): `h` on the real dates is one draw of its stationary AR(1) prior given the initial `(ρ, log σ_h)` (`stationary_ar1_path`, keyed `Sub.CSV_H` of the init key), 0 on the padded rows. Tests: the cold-start state, the path's stationary law, and ten `ScaledChi2` fits at `T = 60` and at `T = 150` (the new fit test fails on the old code; the `T = 150` case is added by a follow-up of the re-check). Phase 2 is unaffected (the 39-case A/B, 39/39). PR #25, `5de678e` | §5.5, E133 |
| E176 | BR, PR #20 (live once WP32 made `CommonSV` reachable): `structural/ordering.py::_prior_fn` read `effective_log_var(st.eq.vol, static)` without the common `h`, which WP30's reference refuses under `CommonSV`, so `prior_covariance_draws` and `ordering_sensitivity(mode="prior")` raised on a `CommonSV` model | `_prior_fn` reads `state_log_var(st, static)`, which adds `glob.vol.h` under `CommonSV` and is the same function otherwise (Phase 2 bitwise). Test: both run on a `CommonSV` model (it fails on the old code). PR #25, `5de678e` | §7, §9.1 |
| E177 | BR: choices WP32 and WP33 recorded in their PRs but not here, and two fixes | Recorded. WP32 (adding to E171): (k) `acc_csv_newton` counts the non-empty blocks whose Newton did not report convergence (the cap or a stalled line search), a mean count per sweep (§5.4 says "iterations that hit the cap"); (l) `order_invariant` is also passed for conjugate × `CommonSV` with the weighted path forced (`const_path = "cholesky"`; the same posterior); (m) row 26's false-alarm control DGP is `cky_dgp(common = True)`; (n) the warm-start test now checks the remap's forward recursion, the stationary first row (E171(e)) and a window that starts earlier. WP33: the `analytic` case of `test_scoring_errors_name_the_model_and_the_origin` is removed (D-49; the OLS case keeps the contract), a Phase 2 test amendment §1.3 now lists; its other minors are fixed in PR #22 (`2e7333c`: the 2020 warning of `StructuralNIGFit` no longer advises `outliers='lenza-primiceri'`, which E-8 refuses; stale `lag_coefs` texts). **Fixes, not choices:** WP34's `conditional_forecast(..., base=…)` refuses a base whose origin is not the fit's last period (PR #28, `f21e040`; §6.1's E-16 list). A base from another vintage of the data with the same last period is not detected: recording the data fingerprint in the forecast's attributes is a follow-up (it would need its own A/B) | §1.3, §5.4, §9.4 |
| E178 | BR: wording and records | Recorded. (a) §9.8's per-entry reading of `mixing_failed` (E172) has no entries for the fixtures staged before `73e3211` (`wbvars_csv`, `wbvars_ncsv` of run 40243289623): for them the not-met quantities are those `SV_PARAMETRIZATION.md` §7.3 records as failing (`Σ`; evidence only under Q35). (b) E174's reason is worded with the evidence's count and interval (4 of 400; 95% interval 0.27–2.5%), and the A10 row and E174 say that the exact parts of the two analytic setups also run at 10⁴ draws. (c) WP28's PR #19: `atol = 1e-13` was added to the `min_pivot` equality checks of `test_linalg_log.py` and `test_shared_factor.py` after seed 11 failed by 2.2e-16 (relative 3.1e-9); pivot equality is outside §9.2, the unit-diagonal factor has an absolute rounding floor, and the change was not disclosed in the PR body. Only `test_linalg_log.py` needs it; the follow-up removes the one in `test_shared_factor.py`. The other review minors (PR bodies, test docstrings, coverage) are follow-ups | §9.2, §9.7, §9.8, E172, E174 |
| E179 | Q44 (owner, 2026-10-26: "Bu 8 satırda t_max = 400 (Recommended)", t_max = 400 for these 8 rows): in WP35's weekly run 40414682400 (head `bc44555`) eight SBC rows failed the per-replication ESS rule at the cap `t_max = 200` | Recorded in the order it happened. (1) WP35 registered `t_max = 200` for every row (P2's cap). (2) In run 40414682400 the γ test is clean on all 30 rows (min `p` 7.4e-4 against Bonferroni ≈ 1e-4), so this is mixing, not calibration. Rows 304, 306, 308, 311, 315, 325, 326 and 327 fail only the ESS rule: 98.8, 98.0, 72.3, 96.1, 97.7, 90.2, 98.5 and 96.2% of replications meet it against 99%, with pilot IATs at the 99th percentile of 183–347 (maximum 216–369) against the 182 the cap admits. 308 and 325 also fail the pooled-ESS gate. (3) The owner decided: these eight rows get `t_max = 400`, a per-row override like P2's row 18 (`SBC_OVERRIDES`). The γ test, both ESS gates, `N`, `L`, burn, pilot, the seed and every threshold are unchanged. 400 is fixed before the rerun, and the result is binding: a row that still fails is not met for R6, with no further change. (4) The result, from WP35's rerun 40430948633 at `a76bf1d` (the eight rows at `t_max = 400`, `SBC_OVERRIDES`; nothing else changed). Six rows pass: 304, 306, 311, 315 (at Gandy–Scott stage 2), 326 and 327. Each now meets the per-replication rule (99.6–100%) and the pooled-ESS gate (82.4–92.1 against 79.2). **Rows 308 and 325 are not met** for the R6 criterion: both now meet the per-replication rule (100%), but fail the pooled-ESS gate, P2's pilot blind spot. 308 reaches 70.7 for `log p(θ)` (the DL group-scale level). 325's pooled ESS is below 79.2 for `log p(θ)`, `mean \|h\|`, `σ_h` and `log σ_h` (minimum 68.4; the centred `σ_h²` step, E169). The γ test is clean on all 30 rows. Per the decision nothing further is changed and neither row is rerun. **Findings** (slow mixing, inputs for WP36, Q34 and R18; from WP35's local pilot runs of 200 replications per row, since the CI records carry no per-quantity IAT; the records' `argmin` is the γ test's smallest p-value, not a mixing measure): 325 `CommonSV`'s `σ_h` (E169's centred step), with `mean \|h\|` and `log p(θ)` (its pooled-gate list); 308 the DL group-scale level `mean_log_v[g]` (R18) and `log p(θ)`; 304, 306 and 326 (the grid `a`) the same, `mean_log_v[g]` and `log p(θ)`; 311 and 327 the horseshoe's global scale `log ζ`, with the reduced-form `Π` and the log-likelihood in the slow replications; 315 SSVS's indicators (`small_dev`, `gamma`, `N1`). P2's pilot blind spot (P2 §9.5, row 18) applies: a pass at 400 rests on the pilot IAT | §9.5, §12 |
| E180 | Q45 (owner, 2026-10-28: "Merge, sonra dar xfail (Recommended)", merge, then a narrow xfail): once WP35 (PR #31) is on `main`, SBC rows 308 and 325, which E179 records as not met, make the scheduled weekly run red every Sunday, in shards `sbc-p3a2` (308 alone) and `sbc-p3b4` (325 alone); the first such run is 2026-11-01 03:17 UTC | **Presentation only**, recorded in the order it happened. (1) E179's binding result: at `t_max = 400` both rows meet the per-replication ESS rule (100%) and have a clean γ test, but fail the pooled kept-draw ESS gate (70.7 and 68.4 against 79.2); they are not met for R6. (2) The question put to the owner on 2026-10-28: "#31 main'e girince, "karşılanmadı" kaydedilen 308 ve 325 satırları her pazar çalışan haftalık CI'ı kırmızı yapacak. Bu iki satırın her biri kendi shard'ında tek başına çalışıyor, o yüzden diğer testlerdeki bir hatayı gizlemiyorlar. Bunu nasıl ele alalım? (İlk haftalık koşu 1 Kasım Pazar.)" The options were "Merge, kırmızı kalsın", "#31'i beklet" and the chosen one: "#31 temizse şimdi merge edilir. Phase 3 session'ı pazardan önce küçük bir PR açar: 308 ve 325 yine çalışır ama strict xfail olur, gerekçe olarak E179 gösterilir. xfail yalnızca kaydedilen sebep (pooled-ESS) için geçerli olur: γ testi ya da replikasyon başına ESS kuralı bozulursa haftalık yine kırmızıya döner. Satır beklenmedik şekilde geçerse de koşu kırmızı olur. Karar Q45/E180 olarak kaydedilir; kriterler değişmez." (3) The rule: rows 308 and 325 still run in full at their E179 settings. Their test is **XFAIL** only when the row fails on the pooled kept-draw ESS gate alone, that is, the γ test is clean, no replication failed, the per-replication ESS rule is met, and every recorded reason is the pooled-ESS one. It **FAILS** on any other failure. It **FAILS** on a pass, since an unexpected pass needs a re-decision. Every other row is unchanged. No criterion, threshold, setting or rerun changes, and 308 and 325 stay not met for R6 (E179) | §9.5, §9.11, §12 |

---

## 1. Module layout and ownership

### 1.1 Import-time rules

P2 §1.1 holds unchanged: `import rai_bvar` imports no JAX; the new user-facing specs (`priors/global_local.py`,
`priors/asymmetric.py`, `components/specs.py` `CommonSV`) are pure Python; array builders that return engine pytrees
live in `priors/global_local_arrays.py` and are imported only by `engines/mcmc/resolve.py`; kernels import JAX and
are loaded only through `engines/mcmc/registry.py`, and compute nothing with JAX at import time (a kernel module first
imported inside a traced step would otherwise fail; E163). `rb.conditional_forecast` is resolved through the lazy table of
`rai_bvar/__init__.py`. The analytic engine for `AsymmetricConjugate` is NumPy only. `forecast/conditional.py` is
NumPy-only at import (like `structural/ordering.py`), and `MCMCFit.conditional_forecast` imports nothing from JAX at
module level, so `from rai_bvar import *` still loads no JAX (E137).

### 1.2 Files (new or changed; the owning work package of each part, §13)

A file with two or more owners is changed by work packages that come one after another in §13's stack (WP29 is merged
before the stack reaches WP38), each in its own part of the file (named here), so two writers never touch a file at
the same time. `engines/mcmc/sweep.py` needs no change: the shared factor's inputs reach the equation body through
`EqContext` and the closed-over `SharedPrior`/`FitCache` (§5.3).

```text
src/rai_bvar/__init__.py                         [WP27] exports (GL classes, DiscreteGrid, AsymmetricConjugate, CommonSV)
                                                 [WP34] lazy conditional_forecast
src/rai_bvar/api.py                              [WP27] resolution, engine table §7.2, errors §7.3, harmonise (D-53), coef_log,
                                                        fit() dispatch of AsymmetricConjugate to StructuralNIGFit by name (E134)
src/rai_bvar/components/specs.py                 [WP27] + CommonSV
src/rai_bvar/components/__init__.py              [WP27] re-exports CommonSV
src/rai_bvar/priors/global_local.py        new   [WP27] user-facing GL specs, DiscreteGrid
src/rai_bvar/priors/asymmetric.py          new   [WP27] AsymmetricConjugate spec
src/rai_bvar/priors/sv.py                        [WP27] + TruncatedNormal
src/rai_bvar/priors/global_local_arrays.py new   [WP31] host builders (groups, log r, SSVS ŝ, hyper arrays) and the GL
                                                        adapter to an internal Minnesota for ψ, means, deterministic and a arrays
src/rai_bvar/priors/scales.py                    [WP33] adapter for AsymmetricConjugate (window_scales, conjugate_omega)
src/rai_bvar/engines/mcmc/config.py              [WP26] §2.1
src/rai_bvar/engines/mcmc/state.py               [WP26] §2.2–§2.3 classes, templates, invariants
                                                 [WP32] draw_template shows SSVSDraw (E164)
src/rai_bvar/engines/mcmc/keys.py                [WP26] Block/Sub additions
src/rai_bvar/engines/mcmc/registry.py            [WP26] kinds, OWNERS, slot_kinds (global steps of vol blocks)
src/rai_bvar/engines/mcmc/protocols.py           [WP26] GaussianLogDiag, uses_global_vol, VolBlock.global_steps, INTERNALS
src/rai_bvar/engines/mcmc/compose.py             [WP26] collects VolBlock.global_steps(); capability rules
src/rai_bvar/engines/mcmc/linalg.py              [WP28] precision_draw_log, shared_factor_*, tridiagonal Newton
src/rai_bvar/engines/mcmc/random.py              [WP28] log_gig, bernoulli_logit
src/rai_bvar/engines/mcmc/runner.py              [WP31] init order (CoefPrior.init_global), log-aware failure_detail,
                                                        _accumulators (rb_incl)
                                                 [WP32] extract of glob.vol draws, _accumulators (rb_ch_*)
src/rai_bvar/engines/mcmc/storage.py             [WP31] GL monitor fields  [WP32] CommonSV latent moments and h_T
src/rai_bvar/engines/mcmc/results.py             [WP31] implied moments n/a under GL  [WP32] CommonSV default draws,
                                                        forecast(..., lag_coefs) keyword (E162)
                                                        [WP34] MCMCFit.conditional_forecast (§6.1; E174)
src/rai_bvar/engines/mcmc/forecast.py            [WP32] common h in _pool/_log_lam, futures by block id
src/rai_bvar/engines/mcmc/warmstart.py           [WP32] remap glob.vol.h
src/rai_bvar/engines/mcmc/resolve.py             [WP31] GL static parts and arrays  [WP32] CommonSV, shared-factor rule
src/rai_bvar/engines/mcmc/monitor.py             [WP31] GL monitors/warnings  [WP32] CommonSV monitors, order invariance
src/rai_bvar/components/coef/global_local.py new [WP31] HS, NG, R2D2, DL, SSVS + GLScaleStep (G_COEF_GLOBAL)
src/rai_bvar/components/mean/linear.py           [WP31] deviation path  [WP32] shared-factor path
src/rai_bvar/components/mean/linear_reduced.py   [WP31] CCCM deviation path, dev_log, log prior from the scales, reads
src/rai_bvar/components/vol/common.py      new   [WP32] CommonSVStep (G_COMMON_SV), h/ρ/σ kernels, futures
src/rai_bvar/components/vol/constant.py          [WP32] common-h conditional (uses_global_vol)
src/rai_bvar/components/structure/recursive.py   [WP32] EqContext.common_w from the closed-over glob.vol
src/rai_bvar/priors/sv_arrays.py                 [WP32] CommonSVPrior builder
src/rai_bvar/components/vol/robust.py            [WP32] passes gvol; log_var adds h_t; global_steps()
src/rai_bvar/engines/structural_nig.py           [WP33] StructuralNIGFit, free cross/contemp, MH problem
src/rai_bvar/engines/analytic.py                 [WP33] forecast(..., lag_coefs) on request (D-49)  [WP34] conditional_forecast method
src/rai_bvar/models/base.py                      [WP33] the lag_coefs keyword of the forecast protocol (D-49)
src/rai_bvar/evaluation/{backtest,compare}.py   [WP33] request lag_coefs for the integrated LPD (D-49)
src/rai_bvar/evaluation/scores.py                [WP34] refuses ConditionalForecastResult
src/rai_bvar/structural/ordering.py              [WP33] StructuralNIGFit support, permuted custom groups
src/rai_bvar/forecast/conditional.py       new   [WP34] conditions, the exact update, ConditionalForecastResult
src/rai_bvar/forecast/invariants.py              [WP34] check_conditional_structure
src/rai_bvar/evaluation/scores.py                [WP34] refuses ConditionalForecastResult
src/rai_bvar/validation/{reference,simulate,log_joint,testfns,joint}.py  [WP30] §9.1 (separate registry, D-61)
tools/make_r_fixtures.R                          [WP29] sections bvars (bayesianVARs P1–P10, R1–R9), wbvars (P11–P16)
tools/make_sv_parity_data.py                     [WP29] packing of the new fixtures
tools/make_commonsv_oracle.py              new   [WP38a] the C2 dataset (tests/parity/fixtures/data/commonsv_n3.csv) and
                                                        the oracle fixtures commonsv_oracle_*.nc (§9.8)
.github/workflows/ci.yml                         [WP29] r_fixtures job (D-54)  [WP41] tiers (E135)
tools/weekly_shards.py                           [WP35] Phase 3 shards (the plan job reads them; no ci.yml change)
pyproject.toml, uv.lock                          [WP28] mpmath in the dev group (the lock regenerated on a rebase)  [WP41] version
tools/phase3_bitwise_ab.py                 new   [WP26] the same-machine A/B of §2 (E118)
benchmarks/pilot/phase3_*.py               new   [WP36] pilot; docs/phase3/pilot.md
benchmarks/phase3_perf.py                  new   [WP36] BENCH-P3 (reported)
benchmarks/gruber_kastner.py               new   [WP40] replication; benchmarks/reports/gruber_kastner_*
SPEC.md, README.md, docs/phase3/DESIGN.md      [WP41]
```

Files owned by the parallel sessions are not touched: `src/rai_bvar/validation/dgp.py`,
`src/rai_bvar/validation/__init__.py`, `tests/dgp/`, `benchmarks/dgp_study*`, `benchmarks/reports/dgp_study*`
(DGP harness) and every `*_bart_spike*` file (Phase 4 spike). New validation helpers are imported from their modules,
not re-exported through `validation/__init__.py`.

### 1.3 Tests (owner)

```text
tests/mcmc/unit/      [WP26] test_contracts_p3 (rules, templates, Phase 2 goldens incl. the DGP arms' hashes)
                      [WP28] test_gig, test_linalg_log, test_shared_factor, test_newton
tests/unit/           [WP27] test_specs_p3, test_api_p3 (every §7.2 row, every §7.3 message, hashes, labels, harmonise)
                      [WP34] test_conditional_forecast (A10, A10b, contract)
tests/priors/         [WP31] test_global_local_arrays
tests/mcmc/blocks/    [WP31] test_global_local (conditionals, density ratios, block Geweke)
                      [WP32] test_common_sv (h kernel, ρ, σ, shared factor, quadrature)
tests/analytic/       [WP33] test_asymmetric (A9 and its MCMC leg, slow: E162), test_lag_coefs_on_request
                      [WP31] A12, A13    [WP32] A8 (n = 3), A11    [WP37] A8 (n = 20)
tests/validation/     [WP30] test_reference_p3
tests/geweke/test_geweke_p3*.py, tests/sbc/test_sbc_p3*.py, tests/mutation/test_mutants_p3.py   [WP35]
tests/integration/    [WP33] test_ordering_p3
tests/integration/    [WP35] the weekly ordering-invariance test of conjugate × CommonSV (row 27; E159)
tests/parity/oracles/ [WP38a] commonsv_exact.py (the exact CommonSV oracle, importing nothing from rai_bvar) and
                      its validation test_commonsv_exact: (b)–(d) in the PR tier (counted against §9.11), the
                      Geweke test (a) weekly
tests/parity/         [WP38] test_commonsv_oracle (C1–C2 against WP38a's fixtures, §9.8),
                      test_bayesianvars (fixtures present, no skip), test_wbvars_commonsv, test_boe_conditional
tests/contract/       [WP35] the new MCMC compositions (fast: three; weekly: all)
tests/integration/    [WP40] test_gruber_kastner_script
```

**Phase 2 test amendments** (E52, E90; each is a contract the work package extends on purpose, as P2 WP2 did in
`ed502d4`): WP26 — `tests/mcmc/unit/test_state.py` (the field lists of `SharedPrior`, `CoefPriorArrays`, `LinearEq`,
`CCCMGlobal`, `EqContext`; the exact `OVERRIDABLE` set), `tests/mcmc/unit/test_config.py` (the `COEF_VARIES` table;
the monkeypatched varying-prior case with Mode II `cholesky_const`, now rejected by rule (11); `registry.OWNERS ==
kinds`; the `slot_kinds` dict; the unknown-kind case now uses an unregistered kind, since a GL kind without `n_groups`
raises the `n_groups` message, E154), `tests/mcmc/unit/test_keys.py` (the frozen `Sub` values; E128). WP27 —
`tests/unit/test_api_resolution.py` (the E-1, E-5, E-6 texts; "Linear × Constant and Linear × SV" and "must be a
Minnesota prior") and `tests/unit/test_api.py` (the same messages, where its patterns no longer match);
`tests/unit/test_grid_harmonise.py` stays unchanged (E119). WP31 — `tests/mcmc/blocks/test_cccm.py`'s `reads` assertion
(`CCCMStep.reads` gains `eq.coef`, `glob.coef`; E128). WP32 — the R8 ordering note disappears for Mode II × `Constant`
at `cross = contemp = 1` (E99), an intended Phase 2 behaviour change, with a test that that fit no longer warns and that
`cross ≠ 1` still does (E130); `tests/integration/test_ordering.py` (the CommonSV false-alarm control replaces the Phase 2
stand-in, row 26; E159). WP33 — `tests/integration/test_backtest_lpd_integrated.py` (the analytic cases: a default
forecast still raises for the integrated estimator at `h > 1`; the backtest requests `lag_coefs=True`, E120; the
`analytic` case of `test_scoring_errors_name_the_model_and_the_origin` is removed, since an analytic integrated
backtest no longer raises, D-49, and the OLS case keeps the error contract; E177). No other
Phase 1–2 test changes; `tests/mcmc/diagnostics/test_monitor.py` stays as it is (order invariance keys on
`shared_factor`, E99).

---

## 2. Contracts (WP26)

All additions keep every Phase 1 and Phase 2 `StaticConfig`, state template (leaf paths, shapes, dtypes), spec and
spec hash unchanged: new fields default to the Phase 2 value, new leaves default to `Empty()` (also in constructor
calls such as `CCCMGlobal(Pi=Pi)`), and new branches are taken only when a Phase 3 component is present (P2 §2.1).
A golden test pins the templates and hashes of every Phase 2 composition and of the DGP study's three arms
(`8aebe5adec44e966`, `b3daa828453542ed`, `7f0aeb733eb2d07e`) at `2c71fb2`, and short runs of every Phase 2
composition are bitwise unchanged before and after each of WP26–WP34: a same-machine A/B of the parent commit and the
work package's head, recorded in its PR (as WP25 did), because committed draws cannot be bitwise across CPU models.
The A/B (`tools/phase3_bitwise_ab.py`, E118, E154) runs, at `hierarchy="fixed"`, every Phase 2 composition row the
public API can build (22 of the 23 rows of `validation.joint.COMPOSITIONS`, built from the registry itself; row 17
`zero_sv` has no public mean; a contract test checks the coverage); option cases (`const_path="cholesky"`, the branch
E65 changes, `gram="batched"` on the weighted structural and the reduced-form steps, the paths that read the Gram
layout, `store="paths"`, `deterministic="ct"`, `SV(exact=False)`, `SV(mixture="ksc7")`,
`SV(tails="t", t_scale="unit")`, a structural `cross ≠ 1`); the EB plug-in at the default `hierarchy="eb"` (structural
`cross ≠ 1` with Constant, SV, and Constant with SOC/SUR rows: the `eb_hyper → eb_structural` path WP33 refactors) and
the default MCMC path at `cross = 1`, with `fit.hyper` in the digest; `AnalyticFit` at `hierarchy` `fixed`, `eb` and
`mh` (`log_ml`, `hyper`, the MH chains, and the forecast's `paths`, `cond_mean`, `cond_chol`); and a forecast-mode
`ordering_sensitivity` of an analytic model (R3-m19's ordering item; an MCMC ordering case costs `n_perm` fits and no
work package touches that path): 39 cases, no two MCMC cases sharing a posterior digest, against a baseline recorded
twice on the unmodified code (`552fe30`). Each
PR records its head commit, the machine and the sha256 of both JSON files.

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

`COEF_VARIES` gains `True` for every GL kind. Rule 5 of P2 (a non-weighted path needs constant weights) and
`compose._check_capabilities` exempt `shared_factor`, and `resolve_const_path` selects it before the constant-weights
test. `validate_composition` keeps its four-argument signature and adds, in order: (11) a varying coefficient prior
needs `coef_log = True`, a `weighted` path and no λ-scaled coefficient block (Mode II); (12) `coef_log` needs a
`linear_structural` or `linear_reduced` mean; (13) `common = True` needs `persistent = "constant"` and
`transitory = "none"`, and makes `constant_weights` false; (14) `shared_factor` needs `linear_structural`, Mode II,
`common = True`, a non-varying coefficient prior, and (checked by `resolve.py` on the arrays, since it is data) an
equation-invariant λ-free prior precision; (15) `csv_block ≥ 2`. The group rules (`n_groups ≥ 1` iff a GL kind, `a_grid > 0` only for NG, R2D2 and DL,
`p_beta` only for SSVS) run in `CoefPriorStatic.__post_init__`; only the value of `G` depends on `n` (equation-wise
`G = n`, covariate-wise `G = n·p`), and `resolve.check_coef_groups` checks it at fit. `api.py` passes `coef_log` and a
provisional `n_groups = 1` to its construction-time check (no rule (11)–(14) reads `G`; E145), so GL models pass rules
(11)–(12) at construction; `resolve.py` passes `n` (`api.static_parts(model, n)`).
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
CommonSVGlobal    h: f64[T_pad]; rho: f64[]; log_sigma: f64[]               index t = data row t (no presample state), 0 for t ≥ T_eff
                  acc_csv_h, acc_csv_rho, acc_csv_newton: f64[] (E132); rb_ch_sum, rb_ch_sq: f64[T_pad]
```

`EqState.coef` holds the `*Eq` class, `GlobalState.coef` the `*Global` class, `GlobalState.vol` `CommonSVGlobal`
(all `Empty` in Phase 2). Draw containers (P2 §2.10): `GlobDraw.vol = CommonSVDraw(h_T, rho, sigma)`, filled by the
volatility block's new `extract_global(glob)` (runner `extract`, WP32); GL scales are not part of `ParamDraw` (they do
not enter forecasts) and are stored as monitor fields (D-52). Storage (WP32): `latent_moments/common_h_mean,
common_h_sd (chain, time)` from `rb_ch_*` (dimension `time` = data rows; P2's `h_index` has `T + 1` entries and is not
reused), `posterior/common_h_T`, `rho`, `sigma` (`glob.vol` leaves mapped explicitly, not onto P2's per-equation
`h_T`), with `runner._accumulators` special-casing `rb_ch_*` (WP32) and `rb_incl` (WP31; `latent_moments/incl_prob
(variable, regressor)`); `acc_csv_*` are per-chain scalars, which WP32 adds to the monitor's `_CHAIN_SCALARS` (E132).
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
- `PersistentCore.uses_global_vol` is a property (default False; `ConstantCore` returns `static.common`, since one class
  serves `Constant` and `CommonSV`); when True, `VolBlockImpl` passes `gvol` to the core's `sample` and `log_var`, and
  at forecast time the future `gfut[Block.G_COMMON_SV]` to `ahead`; `extract` needs neither.
- `EqContext.common_w: f64[T_pad] | Empty = Empty()`: `m_t e^{−h_t}`, built by `Recursive.context` from the
  closed-over `glob.vol` and `tmask` only, so it is unbatched inside the vmapped equation body (§5.3). `VolBlock.global_steps() -> tuple[GlobalStep,
  ...]` (default `()`) and `VolBlock.extract_global(glob)`; `compose` collects the vol block's global steps beside the
  mean's and the coefficient prior's (P2 §3.8 item 4).
- Global futures are passed as a mapping `{Block: future}`, so a block finds its own (WP32 changes `forecast.py`'s
  positional tuple; P2 §7.2).
- `CoefPrior.init_global(key, st, data, prior) -> ChainState` (default identity), called by the runner's cold start
  before every `global_pre` init (`CCCMStep.init` draws Π from `glob.coef`) and before the mean's cold draw (WP31);
  `GLScaleStep.init` is the identity; `CCCMStep.init`/`sample` pass `dev_log` under `coef_log`.
- A new internal `cccm_core_log` is the CCCM deviation path; `cccm_core` keeps its signature and 2-tuple return (mutants
  M1/M18 and `validation.joint.identity_cccm_core` use it).
- `monitor.MonitorField` gains a `group` kind (`(G,)` group scales, `a_g`, `p_g`); `monitor_fields` takes the
  coefficient-prior static.
- `Block` (append-only, P2 §5.6): `G_COEF_GLOBAL = 17` (post, GL scale step), `G_COMMON_SV = 18` (post), both
  already reserved in Phase 2 (no change); 19 stays reserved (Q16). Only `Sub` is appended (E139).
- `Sub` (append-only): `GL_LOCAL = 15`, `GL_LOCAL_AUX = 16`, `GL_GROUP = 17`, `GL_GROUP_AUX = 18`, `GL_A = 19`,
  `CSV_OFFSET = 20`, `CSV_H = 21`, `CSV_H_ACC = 22`, `CSV_RHO = 23`, `CSV_RHO_ACC = 24`, `CSV_S2 = 25`. Per-coordinate
  draws fold the flat coordinate index (`uint32`), per-group draws the group id, per-block `h` draws the block index.
- `registry.KERNELS` appends `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs`
  (`components.coef.global_local:*`), `gl_scales` (`…:GLScaleStep`), `common_sv` (`components.vol.common:CommonSVStep`).
- `protocols.INTERNALS` appends `gl_local_conditional`, `gl_group_conditional`, `gl_a_conditional`, `ssvs_logodds`,
  `gl_deviation`, `csv_newton_start`, `csv_q`, `csv_rho_accept`, `csv_lambda_conditional` (on the mean, E171),
  `shared_factor_rows`,
  `gig_sampler`, `cccm_core_log` (test seam for §9.6; production never sets them).
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

1. **Limits.** `log χ = −∞`, `λ > 0`: `log x = LG(λ) − log(ψ/2)` (gamma). `log ψ = −∞`, `λ < 0`: `log x = log(χ/2) −
   LG(−λ)` (inverse gamma). Any other infinite input is improper and returns `ok = False` (a status code, P2 D-18). In
   a production sweep it occurred only with E133's `h = 0` cold start (`S = 0` in the `ScaledChi2` `σ_h²` step, E175);
   with the stationary start it occurs with probability 0.
2. **Reduction.** `log η = ½(log χ − log ψ)`, `log ω = ½(log ψ + log χ)`; `y` has density `∝ y^{λ−1} exp(−ω(y +
   1/y)/2)`. For `λ < 0` draw with `|λ|` and set `sign = −1` (else `+1`); the sign is applied once, in step 4.
3. **Devroye (2014) §6** for `λ ≥ 0` (the transcription was checked against the paper by R2a): `log y = asinh(λ/ω) +
   X`, with `X` drawn from the density `∝ exp(f(x))`, `f(x) = −α(cosh x − 1) − λ(eˣ − x − 1)`, `f'(x) = −α sinh x −
   λ(eˣ − 1)`, `α = √(ω² + λ²) − λ`. The set-up is entirely in logs:
   - `Λ = logaddexp(½·logaddexp(2 log ω, 2 log λ), log λ)` (`= log(√(ω² + λ²) + λ)`), `log α = 2 log ω − Λ`;
     `asinh(λ/ω) = log 2 + log λ − log ω` when `log λ − log ω > log 1e8`, else `asinh(exp(log λ − log ω))` (0 at `λ = 0`).
   - Points: `−f(1) ∈ [½, 2]` ⇒ `t = 1`; `−f(1) > 2` ⇒ `t = √(2/(α + λ))`; else `t = log 4 − logaddexp(log α, log 2 +
     log λ)`. `−f(−1) ∈ [½, 2]` ⇒ `s = 1`; `−f(−1) > 2` ⇒ `s = √(4/(α cosh 1 + λ))`; else `s = min(1/λ, logaddexp(0,
     log1p(√(1 + 2α)) − log α))`, using `1/α + √(1/α² + 2/α) = (1 + √(1 + 2α))/α`; `1/λ = +∞` at `λ = 0`.
   - Hat: `(η_D, ζ, θ, ξ) = (−f(t), −f'(t), −f(−s), f'(−s))`, `p = 1/ξ`, `r = 1/ζ`, `t' = t − rη_D`, `s' = s − pθ`,
     `q = t' + s'`; flat on `[−s', t']`, `exp(−η_D − ζ(x − t))` right of `t'`, `exp(−θ + ξ(x + s))` left of `−s'`.
   - Draw `U, V, W` uniform on `(0, 1]`: centre with probability `q/(p + q + r)` (`X = −s' + qV`), right with
     `r/(p + q + r)` (`X = t' − r log V`), left otherwise (`X = −s' + p log V`); accept iff `log W ≤ f(X) − log hat(X)`.
   - Every term is evaluated in logs, with `lsh(y) = y − log 2 + log(−expm1(−2y))` for `y > 0`:
     `α(cosh x − 1) = exp(log α + log 2 + 2 lsh(|x|/2))`, `α sinh|x| = exp(log α + lsh(|x|))`; the `λ` terms only when
     `λ > 0`: `λ(eˣ − x − 1) = exp(log λ + le(x))` with `le(x) = log(x²/2 + x³/6 + x⁴/24)` for `|x| < 1e-3`,
     `x + log1p(−(1 + x)e^{−x})` for `x > 30` and `log(expm1(x) − x)` otherwise, and `λ(eˣ − 1) = sign(x)·exp(log λ +
     log|expm1(x)|)`. A term that overflows makes `f = −∞` (a rejection); no product of an underflowed and an overflowed
     factor is ever formed.
   - Theorem 2 bounds the expected number of iterations by 3.4597, uniformly in `(λ, ω)`; R2a measured a worst
     rejection constant of 1.40 on `λ ∈ [0, 10³]`, `ω ∈ [1e-5, 1e5]`. Attempt `t` uses `fold_in(key, t)`; a
     `lax.while_loop` runs until every lane accepts or 512 attempts pass (`ok = False`, probability ≤ (1 − 1/3.46)⁵¹²
     ≈ 1.6e-76 per call).
4. `log x = log η + sign·log y`.

Tier U (WP28): Kolmogorov–Smirnov (`α = 1e-3`, 10⁴ draws) of `Y = log X − log η` against the CDF of `exp(λy − ω cosh y)`
obtained by numerical integration in log space around the mode (`scipy.stats.geninvgauss` is not used as the oracle: it
returns NaN or wrong values on much of this grid, R2a); a grid that pins `λ ∈ {0, ±1e-8, ±1e-3, ±0.5, ±1, ±10, ±50}`,
every `a − ½` and `a − 1` at the GK grid's first, middle and last points, and `½ − T_max/2` for `T_max ∈ {200, 600}`,
crossed with `log ω ∈ {−2000, −1000, −745, −300, −50, −12, −5, 0, 5, 12}` (the §3.4 regime reaches `log ω ≈ −800`),
the whole grid one KS family under D-60's family-wise rule (`m` = its cell count, stated in the test; each cell its
own random stream, the `−λ` mirror cells included, since step 2's sign flip makes same-key mirrors identical; E124);
`E log X` against `log η + ∂_λ log K_λ(ω)` (mpmath at 50 digits; mpmath joins the `dev` group), Bonferroni over the
`m` cells: `|z| ≤ max(4, z_{1−0.01/(2m)})` Monte Carlo s.e. (E152, E159); the gamma and inverse-gamma limits at the extremes; lane invariance under `vmap` (1e-15); determinism by key;
mean and worst-case attempts reported; no `ok = False` on the grid.

### 3.6 Host builders (`priors/global_local_arrays.py`; WP31)

Resolution of a GL prior reuses the Phase 2 Minnesota code for everything the GL prior does not govern: WP31's adapter
maps it onto an internal `Minnesota(form, psi, mean, var_const, contemp, hierarchy="fixed")` for ψ
(`window_scales`), the own-lag means, the deterministic prior and the `a_i` arrays; EB and dummy rows are skipped (GL
priors have neither). Then
`gl_arrays(prior, scales, *, n, lags, n_det, form, data) -> (GLArrays, GLShared, n_groups)` per window (R9): the GL
mask (lag slots), group ids (D-38; a custom array checked against `(n, n·p)` here, E-13), `log r` (D-37, from the
window's ψ), the hyperprior arrays, and SSVS `log τ₀, log τ₁` (D-44). Semi-automatic `ŝ` (one QR per equation of
the window's design, the diagonal of `R⁻¹R⁻ᵀ` by triangular solves, no inverse):

```text
structural, equation i:  ŝ²_j = RSS_i/(T_eff − q_i − 2) · [(F_iᵀF_i)⁻¹]_jj     (flat-prior NIG posterior variance; needs T_eff > q_i + 2)
reduced, equation j:     ŝ²_{rj} = Ŝ_jj/(T_eff − k − n − 1) · [(XᵀX)⁻¹]_rr     (flat-prior normal–Wishart; needs T_eff > k + n + 1)
```

### 3.7 Initialisation, log prior, storage

- **Init** (`Stream.INIT`, P2 §5.4): `CoefPrior.init_global` runs first (§2.4: before every `global_pre` init and the
  mean's cold draw) and sets the group scales so that
  `r_j v_j` equals the Mode I Minnesota variance at `λ = 0.2` (`log ζ_g = log 0.04` for HS, the matching rate for
  NG/R2D2, `a_g` at the grid point nearest 0.1, SSVS `p_g = 0.5`), each group log scale jittered by `N(0, 0.5²)` per
  chain; local scales start at 1 (HS `ν = 1`, DL `θ̃ = 2`, `ϱ = 2`), SSVS `γ = 1`. The mean's cold draw (P2 §5.4) then
  uses these moments and sets `dev_log` (finite from the first state on).
- **Log prior** (monitors and validation): `Σ_j log N(d_j; 0, r_j v_j)` plus the scales' log densities, from the full
  chain state; `CCCMStep.log_prior` keeps its fixed-moment form for `FixedDiagonal` only.
- **Storage**: the group log scales, `a_g` and SSVS `p_g` are monitor fields (D-52) of `fit.posterior["monitor"]`;
  the SSVS indicators `γ` are stored as `int8` with the stored parameter draws (thinned like them; E123, E155);
  `latent_moments/incl_prob (variable, regressor)` from `rb_incl`; local scales are latent, neither monitored nor
  stored.

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
`_posterior_summary` reads `StructuralNIGFit`. Backtests treat it as analytic (`uses_fit_seed = False`, as every
analytic model in Phase 1; the MH seed comes from `MHSettings` or `fit(seed=…)`; E131).

### 4.3 Hyperparameters

- `"fixed"`: as given.
- `"eb"` (default; free set `hyper="auto"` = `("lambda", "cross")` plus `soc`/`sur` when on): maximise
  `Σ_i log p(y_i | h) + log p(h)` with `eb_structural`, whose free names gain `cross` and `contemp` (its `_Window`
  takes them per evaluation). `lambda`, `soc`, `sur` keep the GLP hyperpriors; `cross` and `contemp` get a flat prior
  on their log over `[log 1e-3, log 10]` (marginal-likelihood maximisation, as Chan 2022 §4), stated in the label.
  Start: the Phase 1 EB mode at `cross = 1` (P2 §6.5 step 2). `alpha` free when listed, as Phase 1. `psi` is never free
  when `cross` is free or `≠ 1` (E-15). The free-cross refactor of `eb_structural`/`_Window` is bitwise identical at a
  fixed `cross` (the same arithmetic, with `cross` and `contemp` passed per evaluation instead of read from the prior),
  which the §2 A/B checks on the Phase 2 EB plug-in (E118); any deviation is justified in WP33's PR. When `cross` or
  `contemp` is free, the L-BFGS-B tolerances are `ftol = 1e-15`, `gtol = 1e-9` (Q38, E160), so A9's projected gradient
  is below 1e-6; the plug-in keeps Phase 2's.
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

The composition resolves (§7.2) to `linear_structural` with `coef_scaling = contemp_scaling = "lambda"`,
`VolStatic(persistent="constant", coupled=True, common=True)` and `const_path="shared_factor"`: structural Mode II at
`cross = contemp = 1` with coupled common cores. Its prior is the Phase 1 NIW prior `Σ ~ IW(diag ψ, n + 2)`,
`B | Σ ~ MN(M, Σ ⊗ Ω)` exactly (P2 §6.1, N05 §4.2), with the optional SOC/SUR rows, and `Σ_t = e^{h_t}Σ`: Chan's (2020)
common-SV model with the Phase 1 prior. It is not CCM's (2016) specification, which has a presample `ln f_0`, other
priors and a mixture sampler (N08 §A3); CCM parity would need N08's CCM mode. For equation `i`, given `λ_i` and `h`
(δ-coordinates, P2 §2.3, `θ_i = [δ_i | a_i]`; `m` dummy rows with Mode II weight 1, `F_{d,i}` their active columns):

```text
θ_i | λ_i, h ~ N(θ̂_i, λ_i K̃_i⁻¹),   K̃_i = P̃_i + F_iᵀ W F_i + F_{d,i}ᵀF_{d,i},   θ̂_i = K̃_i⁻¹(P̃_i μ_i + F_iᵀ W y_i + F_{d,i}ᵀ y_{d,i}),
W = diag(m_t e^{−h_t})
λ_i | θ_i, e_i, h ~ IG(1 + ι/2 + (T_eff + q_i + m)/2,
                       ψ_i/2 + ½ Σ_t m_t e²_{it} e^{−h_t} + ½(θ_i − μ_i)ᵀP̃_i(θ_i − μ_i) + ½ Σ_d (y_{d,i} − F_{d,i}θ_i)²)
```

The `λ_i` line is P2's Mode II coupling with dummy rows (WP25 amendment (2): `n_active = q_i + m`, the dummy residual
sum of squares in `quad`; `LinearStructural.coupling` is reused) with the data residuals weighted by `e^{−h_t}` (E88).

At `cross = contemp = 1` the λ-free prior precision `P̃_i` is the same diagonal for every equation on its active slots
(Ω's rows are equal bit for bit, `priors/scales.conjugate_omega`; the λ-scaled `a`-precision is `ψ_j/κ_a`; the Mode II
dummy weight is 1), `F_i` is the active (leading) columns of the shared design `F = [X | X M_c − Y_{:, :n−1}]`, and `W`
does not depend on `i`. So `K̃_i` is the leading `q_i × q_i` block of one matrix `K̃ = P̃ + FᵀWF + F_dᵀF_d`, and the
Cholesky factor of a leading block is the leading block of the Cholesky factor.

**Unbatched inputs** (E87). `Recursive.context` builds `EqContext.common_w = m_t e^{−h_t}` from the closed-over
`glob.vol` and `tmask` only, and `shared_prec`, `F` and the dummy rows reach the body through the closed-over
`SharedPrior` and `FitCache` (P2 §4.5); values computed from closed-over arguments are not batched by `vmap`, so the
Gram, the scaling and the factor below are computed once per chain and sweep, outside the per-equation batch (under
`Batching.MAP` they would be recomputed per equation: correct, slower; no Phase 3 kernel declares `MAP`). Per sweep
(`linalg.shared_factor_*`):

```text
G = (√w ⊙ F)ᵀ(√w ⊙ F) + F_dᵀF_d,   s = diag(P̃ + G)^{−½},   L = chol(s(P̃ + G)s)        once per chain and sweep
per equation i (batched right-hand sides; c_i the active-slot mask):
    b_i = c_i ⊙ (P̃μ_i + Fᵀ(w ⊙ y_i) + F_dᵀ y_{d,i})
    v_i = c_i ⊙ L⁻¹(s ⊙ b_i)          the trailing part of the full forward solve (−L₂₂⁻¹L₂₁v₁) is masked to 0
    θ_i = c_i ⊙ s ⊙ L⁻ᵀ(v_i + √λ_i · (c_i ⊙ z_i))
```

With `v_i` and `z_i` zero beyond `q_i`, the back solve's leading block equals the leading-block solve exactly, so no
per-equation factor is built (it would take `n·q²` memory, 200 MB per chain at `n = 100`, R2a/R2b). **Gates** (E95):
`LinearStructural._conditional` (what `conditional_eq` and the `linear_conditional` internal return to Tier U) takes the
weighted branch for `shared_factor`; `MCMCSettings(const_path="cholesky")` on this composition forces the weighted path
(row 323, A8's twin). `resolve.py` checks on the arrays that `P̃` is equation-invariant on active slots (else a
`ValueError` naming `cross`/`contemp`).

**Cost** per chain and sweep: one Gram `O(T q²)`, one Cholesky `O(q³/3)`, the right-hand sides `O(T q n)` and `n` pairs of
triangular solves `O(n q²)`, against the weighted path's `n` Grams `O(n T q²)` and `n` factors `O(n q³/3)`: the
conjugate speed of D2 (§10).

**Mixing.** The sampler is a two-block Gibbs on `(θ_i, λ_i)`; the lag-1 autocorrelation of `λ_i` is exactly
`q_i/(T_eff + m + q_i + ι)` (*derived here*, corrected by R2a), about 0.5 at `q ≈ T`. The common shift
`(λ_i e^c, h − c)` is identified only through the zero-mean `h` prior, so for `ρ` near 1 `log λ_i` and `mean h` may mix
slowly. WP36 measures both (R29, R31); Q32 (a collapsed `λ_i` draw) and Q34 (an interweaving shift move) record the
remedies.

**Order invariance** is exact (the NIW prior and a common scalar volatility), so there is no R8 note:
`monitor.is_order_invariant` keys on `const_path == "shared_factor"` (which implies `cross = contemp = 1`; those values
are data, not static, D-23), and `resolve.py` passes `DiagnosticsInput.order_invariant` for Mode II × `Constant` at
`cross = contemp = 1` (E99). `hierarchy="eb"` plugs in the Phase 1 EB mode of the conjugate twin at constant
volatility (the D-12 rule; a documented plug-in approximation under CommonSV); `"mh"` is E-5.

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
    accept iff log U (Sub.CSV_H_ACC, b) < log α;  acc_csv_h averages over non-empty blocks
ρ | h, σ_h:  K_ρ = Σ_{t=1}^{T−1} h_{t−1}²/σ_h² (+ 1/V_ρ for TruncatedNormal),  ρ̂ = K_ρ⁻¹(Σ_{t=1}^{T−1} h_{t−1}h_t/σ_h² (+ ρ₀/V_ρ))
             proposal ρ* ~ TN(ρ̂, K_ρ⁻¹; −1, 1); under the Beta prior U(−1, 1) when K_ρ < 1e-8 (E117, E158; under
             TruncatedNormal always the TN) (the choice depends on h and σ_h only, so the kernel is valid);
             a draw with |ρ*| ≥ 1 rejects
             log α = g(ρ*) − g(ρ) (+ the Beta log-density ratio; for U(−1, 1) also the Gaussian factor's ratio
             (S_xy(ρ* − ρ) − ½S_xx(ρ*² − ρ²))/σ_h² from the raw sums S_xy = Σ h_{t−1}h_t, S_xx = Σ h_{t−1}²),
             g(ρ) = ½ log(1 − ρ²) − (1 − ρ²) h_0²/(2σ_h²)
σ_h² | h, ρ: InvGamma(c₀, C₀): IG(c₀ + T/2, C₀ + ½S);  ScaledChi2(B): GIG(½ − T/2, 1/B, S),
             S = (1 − ρ²)h_0² + Σ_{t=1}^{T−1} (h_t − ρh_{t−1})²
```

The proposal of each block depends only on `(q, ρ, σ_h, h outside the block)`, so it is an independence proposal
whatever Newton's accuracy (N08 §A5.2); the random offset is an auxiliary choice of kernel independent of the state;
the sequential scan is a systematic-scan Gibbs over blocks. Diagnostics: `acc_csv_h`, `acc_csv_rho`, `acc_csv_newton`
(the non-empty blocks whose Newton did not report convergence, the cap or a stalled line search, a mean count per
sweep, E177(k); the warning of §8 if > 0, the draw stays exact). There is no mixture indicator, so the DNP
rule (P2 §4.6 item 4) does not arise.

### 5.5 Forecasting, warm starts, initialisation, monitors, storage

- **Forecast**: `CommonSVStep.future(key, gd, H)` draws `h_{T+j} = ρ h_{T+j−1} + σ_h η_j` from `h_T = h[T_eff−1]`
  (P2 §3.6 hook); the Constant cores add it (`log λ_i + h_{T+j}`) through the future mapping `{Block: future}` (§2.4).
  `lag_coefs` as for every linear mean. `forecast.py`'s `_pool` and `_log_lam` read the common `h` (`h_T` from the
  `CommonSVDraw`, other dates from stored paths), so `draw_parameters`, IRFs at a date and FEVD include `e^{h}`.
- **Warm start** (P2 §5.5): `glob.vol.h` is remapped by period like `eq.vol` `h` (dropped periods removed, new trailing
  ones simulated forward with the copied `(ρ, σ_h)`, keyed `eq_key(kc, n)`, a slot beyond the equations), with the
  0-based layout of §2.2 (`glob.vol.h` joins `warmstart.TIME_INDEXED`); `λ_i` copied.
- **Init**: `λ_i` from the Phase 2 Constant init; `ρ` = prior centre (clipped to `(−0.99, 0.99)`), `σ_h²` = prior mean,
  `log σ_h` jittered by `N(0, 0.25²)` per chain; `h` on the real dates one draw of its stationary AR(1) prior given
  that `(ρ, σ_h)`, 0 on the padded rows (E175, replacing E133's `h = 0`: when the first `h` step rejects every block,
  `h = 0` leaves `S = 0`, and the `ScaledChi2` conditional `GIG(½ − T/2, 1/B, 0)` is improper).
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
conditioned; a forecast without `lag_coefs` (a non-linear mean, a benchmark model, an analytic forecast made without
`lag_coefs=True` (the message says "call forecast(..., lag_coefs=True)"; E120), or an in-loop forecast whose draws
are not all stored); a `base` whose origin is not the fit's last period (E177); `R₁` with a diagonal below
`1e-12·max|R₁|` (near-dependent conditions; never jittered).

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
diagonal; simulation consistency (KS, `α = 1e-3` family-wise over the horizons, D-60) of the standardised free
residuals per horizon.

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

Row 2 in `api.py` (E96): `engine` is `"analytic"` iff the form is conjugate **and** the volatility is a `Constant`;
`static_parts` maps the conjugate form with `CommonSV` to `mean_kind="linear_structural"` (not a lookup on the form),
`coef_scaling = contemp_scaling = "lambda"` and `VolStatic(persistent="constant", coupled=True, common=True)`; the
static Mode II predicate and the label predicate are split, so its `axes["prior"]` gains no `scaling=conjugate` (§7.4).
MCMC-only rules after the table: `outliers="lenza-primiceri"` → E-6 (with `CommonSV`: E-7); `hierarchy="mh"` on a
Minnesota MCMC composition → E-5; the JAX-free `config.validate_composition` with `coef_log` (§2.1). `uses_fit_seed`
and `supports_warm_start` are `engine == "mcmc"`. `MCMCSettings.with_default_draws` and `api.default_mcmc` count
`CommonSV` as SV (D-51).

### 7.3 Error messages (raised at construction unless marked; tested verbatim in `tests/unit/test_api_p3.py`)

| ID | Condition | Type | Message |
| :- | :--- | :--- | :--- |
| E-1 | conjugate Minnesota × `SV`/`Constant("ig"|"iw_equivalent")` | `ValueError` | "the conjugate Minnesota prior requires Constant(prior='iw') or CommonSV(); use form='structural' (the MCMC default) or 'reduced'" |
| E-2 | GL × `Constant("iw")` | `ValueError` | "Constant('iw') makes the coefficient prior λ-scaled (Mode II, the conjugate twin); global-local priors are independent priors: use Constant('ig'), Constant('iw_equivalent'), SV() or CommonSV()" |
| E-3 | `AsymmetricConjugate` × not `Constant("auto"|"iw")` | `ValueError` | "AsymmetricConjugate is conjugate only with Constant('iw') (Constant() resolves to it); for Constant('ig') or stochastic volatility use Linear(prior=Minnesota(form='structural', cross=...)) with that volatility, or Minnesota(form='conjugate') with CommonSV()" (E141) |
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

`BVAR.harmonise_grid(combos)` (E119): if any combination has a volatility other than `Constant` with prior
`"auto"`/`"iw"` (P2 §11.4's trigger: `Constant("ig")`, `Constant("iw_equivalent")`, any `SV`), a `CommonSV` whose
Minnesota form is not explicitly `"conjugate"`, or a GL prior, every `Minnesota(form="auto")` becomes `"structural"` and
its `Constant("auto")` becomes `"ig"`. A `CommonSV` arm with an explicit conjugate form does not trigger (the one
narrowing of P2 §11.4); `AsymmetricConjugate` arms neither trigger nor are rewritten, and GL arms are never rewritten.
Explicit settings are never changed; rewrites are listed in the info note (P2 §11.4). WP27 tests that the Phase 2
grids of `test_grid_harmonise.py` come out unchanged.

---

## 8. MCMC budgets and the convergence gate (D-51, D-52)

- **Budget.** `MCMCSettings.with_default_draws(sv=…)` (D-35) treats `CommonSV` as stochastic volatility, and so do
  `resolve._sv_budget`, `fit_mcmc`'s default draws and `api.default_mcmc`: 4 × 10,000 kept draws by default, and the
  D-35 warning below 10,000 per chain, its message naming CommonSV. GL compositions take the D-35 default of their
  volatility until the owner sets D-58 from WP36. WP36 measures, at `n = 7, p = 4, T = 200` (FRED-QD medium set) and at
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
(references) and WP35 (composition suites) are written by an author who wrote no Phase 3 kernel, from this document
only; the PR records the authorship. Every tolerance below is fixed now (D-60).

### 9.1 References (`validation/reference.py`, `simulate.py`, `log_joint.py`, `testfns.py`, `joint.py`)

New registered kinds: `horseshoe`, `normal_gamma`, `r2d2`, `dirichlet_laplace`, `ssvs` (prior samplers of the scales and
deviations in log space, log densities of §3.2 written from the prior definitions, not from the conditionals);
`constant_common` (Constant cores with the common `h`, Mode I or coupled) and `common_sv` (stationary AR(1) `h`, `ρ` and
`σ_h²` priors). `FixedDesign` gains `exogenous: bool` (D-56: a ones column per deterministic term, `N(0, 1)` elsewhere,
drawn once from the configuration seed; `Y` simulated given `θ` without lag feedback). `simulate_obs` gains the
common-`h` path. The Phase 3 rows and blocks live in `COMPOSITIONS_P3` and `BLOCK_CONFIGS_P3` (D-61). Three
references for other work packages' Tier U and Tier A tests (E126, E144): `gl_log_r_reference(psi, n, lags, n_det,
form, scale)`, the D-37 table on the §3.1 slot layout (0 on deterministic slots; exactly 0 under `scale="none"`);
`ssvs_shat_reference`, the §3.6 `ŝ` by a dense inverse (an error when a divisor is `≤ 0`); and
`conditional_forecast_reference`, A10's stacked companion-form oracle from plain arrays (Schur complement and
`log N(r; Rμ, RΣRᵀ)`).

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
   proposal densities (1e-10), including two pinned `ρ` cases at the branch of §5.4: `h_0 … h_{T−2} = 0` (`S_xx = 0`)
   under both priors, TruncatedNormal showing the TN proposal; and, under Beta, `h_0 … h_{T−3} = 0`, `h_{T−2} =
   10⁻⁵σ_h`, `h_{T−1} = 10³σ_h` (`S_xx/σ_h² = 10⁻¹⁰ < 10⁻⁸`, `S_xy/σ_h² = 10⁻²`): the uniform proposal with a non-zero
   data term, so a missing prior or data term fails (E117, E158).
3. **Closed forms**: collapsed `a` weights against numerical integration over `ξ` with `c` fixed and tied (1e-9); DL
   `θ̃ | d` against quadrature of the joint (1e-8); SSVS log odds against brute force (1e-12); `q_t e^{−h_t} | h ~ χ²_n`
   at the data-generating `θ` (simulator check, KS); the conditional-forecast update against the exact oracle (A10).
Every KS test on MCMC draws thins them to their ESS first (R2a).
4. **Infrastructure**: the golden templates and hashes (§2); every Phase 2 composition bitwise unchanged on a short run
   before/after; float64 everywhere; plan invariants (P2 §4.8) with the write-set tree diff for every new composition;
   zero recompiles across windows of one bucket.
5. **Host builders** (WP31, `tests/priors/test_global_local_arrays`; E126): (a) the builder's `log r` against WP30's
   `gl_log_r_reference` at 1e-14 on every slot (`n = 3, p = 2`, unequal ψ, deterministic terms, structural and reduced,
   and `scale="none"`, where `log r` is exactly 0); (b) `ŝ` against WP30's `ssvs_shat_reference` (1e-10), both forms,
   and the fit-time error at the `T_eff` boundaries of D-44; (c) equivariance: with `y_j → c_j y_j`, `c = (3, 0.1, 7)`,
   ψ rebuilt through `window_scales` and `gl_arrays`, the same key: `log r` shifts by `2(log c_i − log c_m)`, each GL
   deviation and `δ` coefficient after one equation phase and one scale step scales by `c_i/c_m`, the scale-step draws
   agree to 1e-9, and the semi-automatic `log τ₀`, `log τ₁` shift by `log c_i − log c_m`; every family, structural
   Mode I and reduced.

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
| blocks | 401–408 (§9.3; `tests/geweke/test_geweke_p3_blocks.py`) | — | — | weekly (G-small) |

**Coverage argument** (the R6 criterion read as "every new block and every new interaction class"). The GL step reads
only coefficients; a GL prior meets the volatility only through the weights of the mean draw. Rows 301–330 cover that
interaction with constant weights (301, 303–310, 326, 327), coupled `a` (318), time-varying SV weights (302, 311–316,
319, 330), transitory weights (317) and common weights (324, 328), in both forms; the volatility blocks themselves are
the Phase 2-validated ones. Compositions not listed (e.g. GL × `SV(tails="t")`, × `SV(rw=True)`) are covered by this
argument, not by a row of their own; the exit-criteria map says so.

**Test priors** (proper and tight, P2 §9.1): HS `global_scale = 0.1` (exogenous rows only, D-56); NG exogenous `a = 0.5,
b = 2, c = 1`, lagged rows `a = 0.5, b = 10` with `c = 0.01` (reduced, row 316: explosive mass 1.7e-5) and `c = 0.005`
(structural, row 319: 0 measured by WP30, R2a's 4.6e-5 superseded, E150) (R2a's measurements; the earlier `c = 0.05` gave 2e-3 to 1.5e-2); R2D2 `a = 0.5, b = 2`; DL `a = 0.5`; grids of
five points `{0.1, 0.25, 0.5, 0.75, 1}` with equal weights; SSVS `τ₀ = 0.01·√r`, `τ₁ = 0.5·√r` (lagged structural row 330: `0.05·√r`, mass 0 measured by WP30, R2a's 1.3e-5 superseded),
`p = 0.3` or `Beta(2, 3)`; `ψ = (0.5, 1, 2)` (n = 3) and `(0.5, 2)` (n = 2), fixed and unequal (so `r_j ≠ 1`, mutant M27); own-lag prior mean 0.4 (mutant M26); the
lagged rows' explosive prior mass measured by WP30 against the target 2e-4 (D-56, E150); everything else P2 §9.2. CommonSV: `ρ ~ TN(0.8, 0.15²)`,
`σ_h² ~ IG(5, 0.2)`; conjugate rows: NIW `Ψ = 0.5 I`, `d = n + 4`, `Ω_rr = 0.01` for every lag coefficient (own and cross, cross = 1; the
Ω entries, not the implied prior variance; explosive mass 1.2e-4 at `p = 2`), intercept `Ω = 1`.
`T_eff = 20 / 50` (never a multiple of 32). **Test functions**: bounded transforms of deviations (`arctan(d_j/√r_j)`,
`1{|d_j| < 0.1√r_j}`) at fixed coordinates, group log scales, the joint `a_g·log ξ_g` (M35), two local log scales,
`a`-grid indices, SSVS `N1_g` and `log p_g`, `h` at three dates and `mean h`, `ρ`, `σ_h`, `log λ_i`, the exact
log-likelihood, `W = Σ_t q_t e^{−h_t}`, log-scale data functions. On the TruncatedNormal-`ρ` CommonSV rows the
unbounded functions of `h` and the log-likelihood are SBC-only, and Geweke uses their `atan` transforms (E148). The
lagged rows' explosive masses are recorded in `validation.joint.EXPLOSIVE_MASS_P3`; rows 320 and block 408 exceed the
`2e-4` target under E150's Monte Carlo rule (pending Q36(i): WP30's test keeps rows 321 and 323 out of its strict
expected failures under that rule; Q36).

### 9.5 Tier S: SBC (weekly; P2 §9.6 configuration: `N = 1000, L = 99`, the ESS rules, γ test, Bonferroni `α = 1e-2`)

Every row of §9.4. Quantities: the test functions above, the total and per-equation log-likelihood, `log p(θ)`, `W`
(N08 §A11), and on lagged rows the predictive `y_{T+1}` through the production forecast core (exogenous rows have no
`x_{T+1}`, E49). The thinning cap is P2's `t_max = 200`, and 400 for rows 304, 306, 308, 311, 315, 325, 326 and 327
(a per-row override like P2's row 18; Q44, E179). Rows 308 and 325, not met by E179, run in full every week, and their
test is a narrow strict xfail (Q45, E180). It is XFAIL only when the row fails on the pooled kept-draw ESS gate alone,
FAIL on any other failure, and FAIL on a pass.

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
| M28 CommonSV Newton started at the current block, one iteration (state-dependent proposal) | `csv_newton_start` | U (the proposal-invariance test of §9.2 item 1; required, deterministic), A11, G |
| M29 `q_t` with `λ_i` instead of `1/λ_i` | `csv_q` | U, G |
| M30 `ρ` acceptance without the stationary-start factor | `csv_rho_accept` | U, G |
| M31 coupled common core without the `n_active/2` shape term | `csv_lambda_conditional` | U, G (row 320), A11 |
| M32 shared-factor dummy rows weighted by `e^{−h_0}` (the first real row's weight; padded rows have `h ≡ 0`) | `shared_factor_rows` | U (against the weighted path), G (row 320) |
| M33 conditional update with `Mᵀ` where `M` belongs | `_update` | U (A10) |
| M34 GIG tails from secants instead of tangents (a hat below `f`): the right tail is the chord of `f` through `(t, f(t))` and `(t + 1, f(t + 1))` extended beyond `t'`, the left the chord through `(−s, f(−s))` and `(−s − 1, f(−s − 1))` beyond `−s'` (E124) | `gig_sampler` | U (the §3.5 KS family or `E log X` at some grid cell) |
| M35 `ξ_g | a_old` drawn before `a_g` (the reversed collapsed order, E30) | `gl_a_conditional` | G (joint test function `a_g·log ξ_g`, rows 304, 306) |
| M36 coupled common core without the dummy rows (`n_active = q_i`, no dummy sum of squares; E88) | `csv_lambda_conditional` | U (density ratio at `m > 0`), G (row 320) |

The suite records the catching tier of each mutant (`mutation.jsonl`); a mutant caught by none blocks the release. The suite lists M19–M36.

### 9.7 Tier A: analytic twins and exact oracles

| Test | Oracle | Rule | Cadence |
| :- | :--- | :--- | :--- |
| A8 | conjugate × CommonSV: shared-factor path vs the weighted path (row 323's config), quantiles of `B`, `Σ`, `h_t`, `ρ`, `σ_h` | Phase 2 parity rule | PR (`n = 3, p = 2, T = 60`); weekly (`n = 20, p = 4, T = 300`) |
| A9 | `StructuralNIGFit` at `cross = contemp = 1` equals `AnalyticFit`: log ML (1e-10); posterior moments from the per-equation NIG posteriors against `AnalyticFit` (1e-10): `E[B] = (E[A]⁻¹E[Γ])ᵀ`, exact because the rows are independent (`(A⁻¹)_ij` does not involve row `j`, so `E[A⁻¹Γ] = E[A⁻¹]E[Γ]` and `E[A⁻¹] = E[A]⁻¹`), and `E[Σ]` from a full-covariance recursion over each equation's posterior second moments (WP33; `implied_reduced_form_moments` is a prior routine and is not used; E127); EB `λ` (1e-6); EB with free `cross`: projected gradient (the box boundary) below 1e-6 at the returned mode; MH: quantiles vs the posterior on a hyperparameter grid by quadrature; at `cross ≠ 1`: forecast quantiles vs MCMC Mode II | 1e-10 / 1e-6 / Phase 2 parity rule | PR; the MCMC leg weekly |
| A10 | conditional forecasts per draw (the analytic and structural-NIG setups, exact and statistical parts, at 10⁴ draws: Q43, E174) against an independent stacked companion-form oracle with Schur-complement conditioning (`validation.reference.conditional_forecast_reference`, written by WP30's author; E144): `E_c`, `Cov_c`, `Y_c` from the same `ε`; analytic (constant `Σ`), MCMC SV and CommonSV draws; a conditioned variable ordered before a free one; sample moments of conditional paths vs `E_c`, `Cov_c` | 1e-10; `|z| ≤ 4` Bonferroni | PR |
| A10b | BoE `bvar` `draw_constrained_forecasts` at fixed `(beta, sigma)` (MIT, CI) vs our per-draw update on the same parameter draw: conditional mean (point output) and the covariance of 10⁴ draws | 1e-10; `|z| ≤ 4` Bonferroni | PR |
| A11 | CommonSV closed forms (N08 §A11): the stationary distribution of block 408 (`h` fixed) vs the Phase 1 NIW posterior with `row_scales = e^{h/2}` (quantiles of `B`, `Σ`); `log p(Y | h)` = Phase 1 log ML − `(n/2)Σh`; the `h` posterior for `T ∈ {2, 3}`, `n = 5` by tensor quadrature vs MH draws (10⁵); proposal state independence | Phase 2 parity rule / 1e-10 / KS `α = 1e-3` family-wise (D-60) | PR |
| A12 | GL deviation path with frozen finite scales equals the Phase 2 path on the same noise, structural and CCCM | conditioning-aware rule | PR |
| A13 | SSVS with `τ₀ = τ₁` (engine-level arrays) equals the Normal prior (`FixedDiagonal`): posterior quantiles | Phase 2 parity rule | PR |

### 9.8 Tier P: parity (fixtures by WP29; tests by WP38; pass/fail, D-59)

**`bayesianVARs` 0.1.8** (reduced form, through `fit_mcmc` with explicit arrays, P2 §9.9). **Probes first**, each a small
fixture whose result is recorded in `tests/parity/SV_PARAMETRIZATION.md` before any parity comparison runs; a probe
that cannot be decided leaves the runs that depend on it **not met** (D-59).

The rules below are those `tools/make_r_fixtures.R` implements at `56fcc5c` (WP29; E122, E125, E157, E172, E173: `7f4e570`'s
rules with the P13 (c)/P14 decision restricted to the variants the rows name and E157's staging fixes, `73e3211`'s extraction and mixing records (E172) and `56fcc5c`'s `h ≥ 2` SV predictive (E173)); any later change to a rule
needs an amendment here before any parity comparison runs.
The configurations, comparisons and decisions below are the script's. The "its runs (…) not met" consequence is applied
at comparison and at the exit criterion, from the probe status recorded in each fixture's `probe_answers` and in
`SV_PARAMETRIZATION.md` (D-59), not by the script's staging (E157). WP38 applies each probe row's own consequence:
the rows are the dependency lists. A run is not met when a probe whose row names that run is not "answered": P1–P3
and P6 for every R run, P4 for R1 and R3–R9, P5 for R2, P7 for R8, P8 for R4–R6 and R9, P10 for R3–R9 (P9 only for
semi-automatic SSVS runs, of which there are none), P11 and P15 for wbvars_*. P12–P14 decide the gating branch, not a
run's status (unmapped or undecided ⇒ branch 1 is not available and branch 2 applies, which is evidence: Q35), and
P16 decides only whether branch 1's `h_t`, `ρ`, `σ_h` part is met. P8 and P16 are decided after the fixtures (by
WP30's comparison and WP38's quadrature) and stay "pending" in them; they count as answered once that decision is
recorded in `SV_PARAMETRIZATION.md`, and the work package that records it amends WP29's schema test (which today
requires a fixture's status to equal the document's, with P8 and P16 pending in both) so that, for those two probes
only, a fixture's "pending" may stand beside the document's recorded decision. PR #21's fixtures come from three runs:
- 40243289623 at `d281276` (14 fixtures, among them the probe fixtures, R2 and the bvars runs), whose decisions equal those of the
  first final run (40234562983 at `83c13b4`; E165);
- 40261716464 at `73e3211` (`stochvol_v_t5`; E172);
- 40351238342 at `56fcc5c` (the 14 bayesianVARs SV fixtures, after E172's extraction fix and E173's `h ≥ 2`
  predictive).

`SV_PARAMETRIZATION.md` §7 lists each fixture's run, and the schema test ties it to the fixture's recorded run and
commit (E157). The R side is R 4.5.3 with the Linux binaries of
the Posit Package Manager snapshot 2026-10-20 (bayesianVARs 0.1.8, bvars 1.1, bsvars 4.0, stochvol 3.2.9); the probe
datasets are `tests/parity/fixtures/data/*.csv` at that commit (`probe_corr3.csv`, `probe_gl.csv`, `probe_t2n5.csv` and
`qd3_growth.csv` are new in WP29, written by `tools/make_sv_parity_data.py`). Common definitions: **Phase 2 rule**: each of `m` compared quantiles passes when `|z| ≤ max(4, z_{1−0.01/(2m)})`, `z` = difference / combined quantile MCSE (Vehtari et al. 2021 §4.3 and batch MCSE, the larger), one-sided MCSE when one side is exact. **Loose prior** (P2, P3, P4): `prior = "normal", normal_sds = 100, priormean = 0, PHI_tol = 0`, `cholesky_normal_sds = 100`, `cholesky_U_tol = 0`, `prior_intercept = 100`; constant variances `cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))` unless SV is named. Probe fits use fixed seeds (`BV_PROBE_SEED = 20261400` plus a fixed offset per fit, `WB_SEED = 20261800` plus offsets; every seed is recorded in the fixture), one chain unless stated. **PIT test**: for a state component drawn from its full conditional, `u = F(x | conditioning values)` is iid U(0, 1); KS test against U(0, 1) at `α = 1e-3`, the CDF on a 4,001-point (ρ) or 8,001-point (log variance) grid by the trapezoid rule.

| Probe | Configuration | Comparison and bands | Decision |
| :- | :--- | :--- | :--- |
| P1 (`normal_sds`, `cholesky_normal_sds`: variance or s.d.; fill order) | `probe_tiny.csv` (n = 2, p = 1, T_eff = 4); `prior = "normal", normal_sds = 1e-3·(1, 2, 3, 4), priormean = 0, PHI_tol = 0`; `cholesky_normal_sds = 1e-3`, `cholesky_U_tol = 0`; constant variances IG(3, 2); `prior_intercept = 1e3`; 20,000 draws after 2,000 | the four lag-coefficient posterior variances `v` (the PHI row with the largest variance is the intercept); slot numbers `v/1e-3` (variance reading) and `√v/1e-3` (s.d. reading); a reading is valid iff its rounded slot numbers are a permutation of 1..4 and every raw/rounded ratio is within 10% of 1; the free U element's posterior variance within ±0.5 decades of 1e-3 (variance) or 1e-6 (s.d.) | decided iff exactly one reading is valid, its slot numbers are column-major or row-major over (PHI lag rows × equations), and the U reading is decided; `V_prior` is recorded as evidence; otherwise undecided ⇒ unresolved ⇒ its runs (R1–R9, P6) not met (D-59) |
| P2 (U vs U⁻¹, layout, sign) | (a) `var_sv_n7.csv`, p = 1, SV, loose prior, 1,000 draws after 300; (b) `probe_corr3.csv` (n = 3, p = 1, T = 600, A u = e with a21 = −0.6, a31 = 0.5, a32 = −0.7, e ~ N(0, diag(1, 0.5, 2))), loose prior, 5,000 draws; (c) `probe_corr.csv` (n = 2, corr(u1, u2) = +0.8), loose prior, 1,000 draws | (a) every candidate reading of the U draws (M × M matrix or free-element vector; upper or lower triangle, column- or row-major; A = t(U), 2I − t(U), solve(t(U)) or A = U, 2I − U, solve(U)) against the documented `vcov()` Σ_t at three (t, draw) pairs, Σ = A⁻¹ diag(exp(logvar)) A⁻ᵀ: match iff relative max error < 1e-8 and A unit lower triangular (1e-12); (b) under the selected layout, the posterior mean of every free a_ij under A = U′, 2I − U′ and (U′)⁻¹ against the A of the LDLᵀ factor of the OLS residual covariance: within 0.05 (max abs) or not; (c) posterior mean corr(u1, u2) by `vcov()` and by the selected mapping | decided iff (a) at least one reading matches and all matching readings give the same A on every checked draw (one mapping), (b) exactly the selected one of the three is within 0.05, and (c) both correlations are +0.8 ± 0.1; otherwise undecided ⇒ unresolved ⇒ its runs (R1–R9, P6) not met (D-59) |
| P3 (intercept row, lag order of PHI) | `probe_corr.csv` (p = 1) and `var_sv_n3.csv` (p = 2), loose prior, 5,000 draws each | the returned X columns against the rai-bvar design (const, lag-1 block, …, lag-p block) built from the CSV: exact match iff max abs difference < 1e-10, PHI row r taken to belong to X column r; the mapped posterior mean of every PHI entry against the OLS coefficients: max abs difference < 0.05 on both datasets | decided iff every X column matches exactly one design column, the intercept is the first or the last row on both datasets, the lag rows follow rai-bvar's order, and both OLS checks pass; every parity run re-derives its row map from its own X and stops if it differs from the predicted one; otherwise undecided ⇒ unresolved ⇒ its runs (R1–R9, P6–P9) not met (D-59) |
| P4 (`sv_para` layout) | `probe_sv.csv` (n = 3, p = 1, T = 400, μ = (−3, 0, 2), φ = (0.8, 0.9, 0.97), σ = (0.4, 0.3, 0.15)), SV, loose prior, 5,000 draws | for each of the 2 orientations × 6 permutations of the series axis: the posterior means of the slices against the true rows (μ, φ, σ, σ²) by max abs distance; a pair satisfies the rule iff every slice is nearest to exactly one target at less than 0.5 × its distance to the second-nearest target, and μ, φ and one of σ / σ² are each matched once (two slices near μ: the one most correlated with h_1 is h_0) | decided iff exactly one (orientation, permutation) pair satisfies the rule; σ vs σ² and any h_0 slice are recorded; otherwise undecided ⇒ unresolved ⇒ its runs (R1, R3–R9) not met (D-59) |
| P5 (`cholesky_priorhomoscedastic` scale; the NA default) | `probe_tiny.csv`, `normal_sds = 1`, `cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = cbind(rep(50, M), rep(100, M))`, `prior_intercept = 100`, 20,000 draws; a second fit with the default NA | the median over equations of the posterior mean of λ | decided: within ±0.2 decades of 100/49 ⇒ the inverse-gamma scale β₀; below 0.2 ⇒ the scale of a Gamma on 1/λ; the NA default's substituted field is recorded as data; otherwise undecided ⇒ unresolved ⇒ its runs (R2, P8, P10) not met (D-59) |
| P6 (internal standardisation) | `var_sv_n3.csv`, p = 2, SV defaults, `cholesky_normal_sds = 1e6`; prior P: Normal with s.d. 0.05 on every lag coefficient and 0.1 on every intercept (`priormean = 0`); fits B (data, P), S (series 1 × 100, P), T (data, the raw-unit image of P: s.d. × d_j/d_i, intercept s.d. / d_i); 4 runs × 10,000 draws after 2,000 per fit | S against (i) the equivariant image D·PHI_B·D⁻¹, (ii) PHI_B unchanged, (iii) the image of PHI_T, each over the 105 quantiles of PHI under the Phase 2 rule | decided iff exactly one of (i), (ii), (iii) passes: (i) or (ii) ⇒ both sides receive pre-standardised data in bayesianVARs' own convention (sample, divisor, demeaning) in R1–R9 and the SWA fixtures are regenerated on the same data — that convention is not established by any WP29 probe, so in that case the script runs none of R1–R9 and a convention probe (a design amendment) must precede them; (iii) ⇒ raw data, the committed SWA fixtures stand; otherwise undecided ⇒ unresolved ⇒ its runs (R1–R9) not met (D-59) |
| P7 (grouping matrices) | `var_sv_n3.csv`, p = 2, `specify_prior_phi(prior = "HS", global_grouping = g)` for g ∈ {global, equation-wise, covariate-wise, olcl-lagwise} | the integer-valued (≥ 0) fields of length lags·M² present for every g; the grouping field is the one whose number of distinct values varies with g; mapped to rai-bvar's order (P3, P1 fill order) and compared as a partition with D-38 | decided iff exactly one such field exists; the four matrices are recorded as data and a matrix that differs from D-38 is passed to rai-bvar as a custom `groups` array (not a failure); otherwise undecided ⇒ unresolved ⇒ its runs (R8) not met (D-59) |
| P8 (meaning of the GL hyperparameters) | `probe_gl.csv` (n = 3, p = 2, T_eff = 8), constant variances IG(3, 2), `cholesky_normal_sds = 1`, `prior_intercept = 10`, `priormean = 0`, every tol 0, global grouping; HS; NG (a, b, c) = (0.1, 1, 4) and (0.1, 1, 1); R2D2 (a, b) = (0.1, 4) and (0.1, 0.5); DL a = "1/K" and 0.5; 4 runs × 50,000 after 10,000 each | R side only (fixtures `probe_bvars_p8_*.nc`: PHI, A, λ summaries, all hyperparameter and `V_prior` draws thinned to 500 per run); WP30 runs rai-bvar under each candidate mapping of §9.8 on the same data, the Phase 2 rule on the PHI quantiles | data committed, decision pending WP30: the mapping that passes while every other candidate fails is adopted; none or several ⇒ undecided ⇒ unresolved ⇒ its runs (R4–R6, R9) not met (D-59) |
| P9 (SSVS semi-automatic τ) | `var_sv_n3.csv`, p = 2, `specify_prior_phi(prior = "SSVS", SSVS_c0 = 0.01, SSVS_c1 = 100, SSVS_semiautomatic = TRUE)` | every positive finite numeric field of length lags·M² (or a chunk of one up to 4 × that length) against c·ŝ, c·ŝ² and c²·ŝ² (c ∈ {c0, c1}) for ŝ = D-44's flat-prior normal–Wishart s.d. (df T − k − n − 1) and the variants df T − k, df T, and X without the intercept (df T − k + 1 − n − 1, T − k + 1), in rai-bvar → slot order (P1, P3): equal iff max relative difference < 1e-8 | decided iff a τ field exists: equal to D-44 ⇒ semi-automatic runs may be compared; otherwise the runs use `SSVS_semiautomatic = FALSE` with explicit τ on both sides (R7 does) and the difference (with any variant that matches) is recorded; no τ field ⇒ undecided ⇒ unresolved ⇒ semi-automatic runs not met (D-59; R7 is unaffected) |
| P10 (intercept outside the shrinkage; `prior_intercept` an s.d.; every tol accepts 0) | (a, b) `probe_tiny.csv`, `prior_intercept = 1e-3`, constant variances IG(3, 2) (P5), (a) `prior = "normal", normal_sds = 1`, (b) `prior = "HS"`, 2 runs × 20,000 each; (c) `var_sv_n3.csv`, p = 2, SV, 2,000 draws after 500, for HS, NG (0.1, 1, 1), R2D2 (0.1, 0.5), DL 0.5, SSVS (0.01, 100, FALSE, 0.5) | (a) the intercept quantiles of both equations against N(0, 1e-6) (s.d. reading) and N(0, 1e-3) (variance reading), Phase 2 rule with the R MCSE, m = 10; (b) the HS fit against the s.d. reading; (c) the call with `PHI_tol`, the family's `*_tol` and `cholesky_U_tol` all 0 succeeds with finite draws, else the smallest accepted of 1e-300, 1e-100, 1e-50, 1e-18 | decided iff exactly one reading passes (a) and (c) finds an accepted tolerance for every family; (b) passing ⇒ the intercept is outside the shrinkage; failing ⇒ inside, which no rai-bvar option reproduces, so R3–R9 are not met at comparison (their fixtures are still staged, flagged by the P10 answer); a run at a positive floor is valid only if no draw sits at it (`floor_fraction_phi`, `floor_fraction_u` recorded per run; any draw at the floor ⇒ that run not met); otherwise undecided ⇒ unresolved ⇒ its runs (R3–R9) not met (D-59) |

Parity runs (one parameter set per run, both sides derived from it, E11's lesson); data: P2's simulated parity sets
`tests/parity/fixtures/data/var_sv_n3.csv`, `var_sv_n7.csv`, raw or pre-standardised as P6 decides (E125). This amends P2 §9.9, which
named FRED-MD subsets for the bayesianVARs runs: the simulated sets are the SWA oracle's, so R1 and R2 at its shapes
(`n3p2`, `n7p4`) are a three-way check.

| Run | Composition (`groups`, scales) | Data |
| :--- | :--- | :--- |
| R1 | reduced × SV, Normal coefficient prior: the Phase 2 criterion (P2 §9.9 parameter set) | `n ∈ {3, 7}`, `p ∈ {2, 4}` |
| R2 | reduced × `Constant("ig")`, Normal prior | `n = 3, p = 2` |
| R3–R7 | reduced × SV × {HS, NG, R2D2, DL, SSVS} (`"global"`, fixed `a`, fixed `p`, `semiautomatic=False`, `scale="none"`; R7's SSVS `c0 = 0.01, c1 = 1`, since `c1 = 100` mixed badly in discovery run 40226725087, chosen before any comparison; E147) | `n ∈ {3, 7}`, `p = 2` |
| R8 | reduced × SV × HS (`"olcl-lagwise"`) | `n = 7, p = 4` |
| R9 | reduced × SV × NG with a discrete grid on `a` (`NG_b`, `NG_c` and the grid matrix stated in the fixture) | `n = 7, p = 2` |

Common settings: `cholesky_U_prior = "normal"` with `cholesky_normal_sds` from P1; the SV priors of P2 §6.3 with
`SV(offset=0.0, exact=False)` (bayesianVARs' default offset 0; `expert_sv_offset` is not passed); `prior_intercept =
s_c` ↔ `var_const = s_c²` (P10); `PHI_tol = cholesky_U_tol = DL_tol = R2D2_tol = NG_tol = 0` (P10), recorded in the
derived call. Draws: on the R side 4 independent `bvar()` runs with recorded seeds, each 50,000 kept draws after 10,000
burn-in, MCSE from the per-run ESS; on our side 4 chains × 50,000 (weekly), the run that decides the exit criterion
(the release run, D-59) and the only one the mixing rule below applies to on our side, and 4 × 6,000 (PR short run, P2
§9.10), a smoke test under the rule of the oracle paragraph below that never decides the criterion.
Compared with the Phase 2 rule: 5/25/50/75/95% quantiles of `Π`, `A`, `h_t` (`t ∈ {1, T/4, T/2, T}`), `μ, φ, σ`, the
predictive at horizons 1–4 (for the SV runs, `h = 1` from `predict()` and `h ≥ 2` computed on the R side from
bayesianVARs' own posterior draws by the path recursion, `predict()`'s native `h ≥ 2` output kept as evidence and not
compared: Q42, E173), and, where `bayesianVARs` returns them with the same meaning, the SSVS inclusion
probabilities (each side's mean of its per-draw indicator series `γ_j`, with the MCSE of a mean `sd/√ESS`: the R side
with the per-run ESS, our side with the chain ESS of the stored `γ` draws (E155); `z` the difference over the combined
MCSE, Bonferroni as the Phase 2 rule; `p̂ ∈ {0, 1}` on both sides passes iff equal, on one side the other side's MCSE
is used alone; E123), the HS global scale (`C⁺(0, 1)` on both sides) and R9's posterior of `a`. The tests fail, never skip, when a fixture is missing in the release run (D-59).

**`bvars` 1.1** (Woźniak; section `wbvars` of `make_r_fixtures.R`, fixtures `wbvars_*.nc`; the conjugate CommonSV of
§5.3). Probes record **data fields only** (rules at `d281276`, E121, E157). The documented output of `estimate()` has `A`,
`Sigma` and `V` only; `h`, `rho`, `omega` exist as the `last_draw` starting values, so P16's draws are extracted by
chained `estimate(post, S = 2)` calls (`estimate` refuses `S = 1`), recording those data fields after each.

| Probe | Configuration | Comparison and bands | Decision |
| :- | :--- | :--- | :--- |
| P11 (defaults) | `qd3_growth.csv` (FRED-QD small set: 4Δ of 100 × log GDPC1 and GDPCTPI, FEDFUNDS; 1960Q2–2019Q4), p = 4; `specify_bvar$new(data, p, common_volatility ∈ {homoskedastic, cSV, ncSV}, distribution = "norm")` | the `get_prior()` and `get_starting_values()` lists recorded as arrays (data fields only) | decided iff the fields are retrieved; otherwise undecided ⇒ unresolved ⇒ its runs (wbvars_*) not met (D-59) |
| P12 (can V be held fixed?) | `qd3_growth.csv`, p = 4, cSV; Ω = the default starting V; Psi = c·Ω, Gamma = c·Ω⁻¹ and the swapped orientation, c ∈ {1e4, 1e6, 1e8}, lambda = (K + 1)/2, starting V = Ω; 1,000 + 2,000 draws each; controls: default prior, and default prior with starting V = 2Ω | posterior draws of V: max_ij sd(V_ij)/√(Ω_ii Ω_jj) and max_ij \|mean V_ij − Ω_ij\|/√(Ω_ii Ω_jj) | holds iff some setting gives both ≤ 1e-3; does not hold iff every setting ran and none does; the controls are recorded (when V does not move under the default prior either, the runs keep the default starting V = Ω and the conjugate comparison uses V = Ω); otherwise undecided ⇒ unresolved ⇒ its runs (gating branch 1) not met (D-59) |
| P13 (E h = 0, initial condition) | (a) the length of h; (b) `qd3_growth.csv`, p = 4, data × 10 with S0 × 100 (the starting V, from AR fits, rescales itself) against the base fit, cSV and ncSV, 4 runs × 10,000 after 2,000 each; (c) the P16 chained draws (`probe_t2n5.csv`) | (b) A (intercept / 10) and Σ/100 quantiles of the scaled fit against the base fit, Phase 2 rule (m = 225); (c) PIT test of every ρ draw against its conditional given the same sweep's h and variance (cSV: σ_v², equal to ω² in the state; ncSV: unit variance on the stored h̃) under three candidates: ρ ~ U(−1, 1) with h_0 = 0 (h_1 ~ N(0, σ²)), ρ ~ U(−1, 1) with a stationary h_1 (N(0, σ²/(1 − ρ²))) — both rejected outright (p = 0) if any ρ draw lies outside (−1, 1) — and a flat prior on the real line with h_0 = 0 (the exact normal conditional), which passes only if, in addition, the observed number of draws with |ρ| ≥ 1 is consistent with the sum over draws of its conditional mass outside (−1, 1) (two-sided Poisson test, α = 1e-3; a pass reads "untruncated: flat, or a normal prior too diffuse to detect at this size"); Metropolis repeats: fraction of iterations with ρ unchanged ≤ 0.01 | decided iff (b) passes for both volatilities and, for each volatility, the repeat fraction is ≤ 0.01 and exactly one candidate has a passing same-sweep PIT; it maps onto CommonSV (stationary h_0 at the first data row, E h = 0) for a volatility iff that candidate's initial condition is the stationary one; otherwise undecided ⇒ unresolved ⇒ its runs (gating branch 1) not met (D-59) |
| P14 (ρ and ω / σ_ω² priors; does a setting fix σ_ω²?) | the P16 chained draws; fixing scan on `probe_t2n5.csv`: (sv_a, sv_s) multiplied by (1e6, 1e-6), (1e6, 1), (1e6, 1e6), (1e8, 1), (1e8, 1e-2), 2,000 burn-in + 200 calls of S = 10 | PIT tests (same sweep): ρ by the candidates of P13 (c), its prior being that of the one passing candidate; ncSV σ_ω² against the GIG conditional of ω ~ N(0, σ_ω²), σ_ω² ~ G(sv_a, scale s_) and G(sv_a, rate s_); cSV σ_v² against σ_v² ~ σ_ω² χ²₁ (h_0 = 0 and stationary) and σ_ω² against G(sv_a, s_) with w² = ω² or σ_v²; fixing: posterior CV of σ_ω² (ncSV) or σ_v² (cSV) ≤ 1e-3 | decided iff the ρ prior is decided (exactly one P13 (c) candidate passes) for both volatilities and one ncSV σ_ω² candidate passes; maps onto a CommonSV option for a volatility iff ρ ~ U(−1, 1) (= `Beta(1, 1)` on (ρ + 1)/2) and the variance prior has the family and every parameter of an option (ncSV: σ_ω² fixed by a setting ⇒ `ScaledChi2(σ_ω²)`; cSV: a fixed inverse gamma ⇒ `InvGamma`, which these candidates do not establish); otherwise undecided ⇒ unresolved ⇒ its runs (gating branch 1) not met (D-59) |
| P15 (Σ covariance or scale; prior mean; intercept) | `probe_corr.csv` (n = 2, p = 1), homoskedastic normal, 2,000 + 5,000 draws; `specify_prior_bvar$new(N = 3, p = 4, stationary = rep(FALSE / TRUE, 3))`; the X of the `qd3_growth` specification | posterior mean Σ against the OLS residual covariance (df T − k): max relative difference < 5% ⇒ covariance; A0 with stationary = FALSE: 1 on every own first lag and 0 elsewhere, with TRUE: 0 everywhere; the X row equal to the design's constant column (< 1e-10) is the intercept, checked by its A column: posterior mean within 1.5 of (5, −3), every entry above 2 in absolute value, every other entry below 2 | decided iff all three hold; otherwise undecided ⇒ unresolved ⇒ its runs (wbvars_*) not met (D-59) |
| P16 (is bvars' h sampler exact?) | `probe_t2n5.csv` (n = 5, p = 1, T = 2, iid N(0, 1), seed 20261032); the specification is built on a carrier (the data plus 40 N(0, 1) rows, because `specify_bvar$new` fits AR models) and its data matrices, prior and starting values are replaced by those of the data; cSV and ncSV, k = 4 seeded runs each: a 10,000-draw burn-in, then 10,000 chained `estimate(post, S = 2)` calls (`estimate` refuses S = 1), h, ρ, ω, σ_v², σ_ω², s_, S recorded after each; extraction check: 4 independent single runs (the same burn-in, then one `estimate()` of 10,000 draws), A and Σ quantiles against the chained runs' A and Σ draws, Phase 2 rule | WP38: the joint posterior of (h, ρ, σ_h) by 4-D quadrature with the closed-form p(Y \| h) (N08 §A4) under the P11–P14 reading, KS α = 1e-3 family-wise over its marginals × volatilities (D-60) on ESS-thinned draws pooled over the k = 4 runs | data committed, decision pending WP38: KS passes ⇒ exact; a KS failure is recorded as "mapping or sampler, undetermined", never as "approximate", and cannot move predictive quantiles into the gate; under an untruncated ρ prior (P13 (c), P14) the (h_1, ρ) posterior at T = 2 may be improper (∫ N(h̃_2; ρh̃_1, 1) dρ = 1/|h̃_1| is not integrable at h̃_1 = 0), so WP38 must establish propriety before any KS verdict, otherwise P16 for that volatility is "no proper posterior at T = 2" (the evidence records the fractions of draws with |ρ| > 1 and > 10); an extraction check failure ⇒ undecided; otherwise undecided ⇒ unresolved ⇒ its runs (the h_t, ρ, σ_h part of branch 1) not met (D-59) |

Parity runs: `estimate(spec, S, thin)` runs one chain without a burn-in argument, so each R run is a 10,000-draw
`estimate` followed by a 50,000-draw `estimate` restarted from its `last_draw`, `k = 4` runs with recorded seeds, MCSE
from the per-run ESS. Data: `qd3_growth.csv`, the FRED-QD small set as growth rates (4Δ of 100 × log `GDPC1` and
`GDPCTPI`, `FEDFUNDS` in levels; 1960Q2–2019Q4), `p = 4` (E146). **Selected comparison**, per volatility (`cSV`,
`ncSV`), decided by the probes and recorded before it runs; under Q40 neither branch gates the SPEC §13 `CommonSV`
criterion (the exact oracle below is the criterion), and both are reported as evidence (Q35, E168):

1. Branch 1 iff P12 holds, P13 and P14 map bvars's SV prior exactly onto a `CommonSV` option, and P15 is decided:
   posterior quantiles of `B` and `Σ` under the Phase 2 rule; `h_t`, `ρ`, `σ_h` are compared only if P16 passes
   (exact); if
   P16 fails ("mapping or sampler, undetermined"), is "no proper posterior at `T = 2`" or was not run, they are
   reported where available and that part of branch 1 is not met (the P16 row, D-59). Branch-1 runs use the mapped
   settings: the P12 `V` and the P14 `(sv_a, sv_s)` that fixes `σ_ω²`, recorded with their values (E157).
2. Otherwise (no exact mapping): the 1- and 4-step predictive quantiles under the Phase 2 rule, same data and draw
   counts, as SPEC §10.4 prescribes when no exact mapping exists, **reported as evidence, not gated** (Q35, resolved by
   the owner): "bvars parity for CommonSV" is then **not met** for want of an exact mapping (E121, E167).

Under Q39(b) (owner, asked about the bvars runs), a fixture whose hard mixing check fails (`mixing_ok = 0`, on either
side of a comparison) is **not met** for the quantities that check covers, read per entry of the fixture's
`mixing_failed` (E172). Fixtures staged before `73e3211` carry no `mixing_failed`; for them the not-met quantities are
those `SV_PARAMETRIZATION.md` §7.3 records as failing (for `wbvars_csv` and `wbvars_ncsv`, `Σ`; E178). The same rule
holds for every
run of §9.8
(bvars, `bayesianVARs`, the WP38a oracle and the engine's decisive weekly runs; E168): each run length is fixed here
before the run, and no run is lengthened after it is generated to make it pass (E165, E167).

The discovery runs whose R-side output informed the revisions of these rules (R side only; no rai-bvar code ran
against any R output, and no Phase 3 fixture was committed before the rules were written here; E122): 40226143296,
40226725087, 40227507036, 40228156925, 40231286955, 40232663417, 40233547243, and 40242687927 (the `d281276` records);
run 40228777915 used the earlier rules and counts as discovery only. At their (quick) size both volatilities take branch 2 (cSV resolves to `ρ ~ U(−1, 1)`
with `h_0 = 0`, which no `CommonSV` option has; ncSV's `ρ` prior fits no candidate); the final run's draw counts
decide (E147).

A failure under the selected comparison is **not met** (D-59); A8 and A11 are evidence, never a substitute (E35).

**Exact `CommonSV` oracle** (WP38a; Q40, resolved by the owner on 2026-10-25: "Bağımsız kesin oracle yaz
(Recommended)"; E168). §9.8's bvars comparison takes branch 2 for both volatilities, which Q35 makes evidence, so the
SPEC §13 `CommonSV` criterion is checked, as Phase 2 checked its reduced-form sampler (P2 §2.10 Stage 4 amendment 2),
against an exact sampler written independently of the engine. Everything below is fixed before any comparison runs
(D-59, D-60).

- **Model** (the composition `BVAR(Linear(Minnesota(form="conjugate", hierarchy="fixed", …)), CommonSV(), lags=p)`,
  §5.3, written in reduced form): `y_t = Bᵀx_t + u_t`, `u_t | h_t, Σ ~ N(0, e^{h_t}Σ)`; `Σ ~ IW(Ψ, d)`, `Ψ = diag(ψ)`,
  `d = n + 2`; `vec B | Σ ~ N(vec b, Σ ⊗ Ω)` with the Phase 1 prior of SPEC §4.6 (`Ω` diagonal: lag `l` of variable
  `j` `λ²(d − n − 1)/(l^α ψ_j)`, deterministic terms `var_const`; `b` zero except the own first lags, `mean_i`); the
  SOC/SUR dummy rows of SPEC §4.6 when the run has them, with weight 1 (not scaled by `e^{h}`; E88); `h_t = ρh_{t−1} +
  σ_h η_t`, `h_0 ~ N(0, σ_h²/(1 − ρ²))` at the first data row (0-based), `E h = 0`; `ρ ~ N(0.9, 0.2²)·1(|ρ| < 1)`,
  `σ_h² ~ IG(5, 0.04)` (CommonSV's defaults, D-46). The engine's Mode II at `cross = contemp = 1` is this prior exactly
  (P2 §6.1, N05 §4.2), so both sides target one posterior.
- **Sampler** (Gibbs over three exact blocks, each a different algorithm from the engine's; block 3 adds a
  non-centred `σ_h` move the engine does not have):
  1. `(B, Σ) | h, Y`: one joint matrix-normal–inverse-Wishart draw with row weights `w_t = e^{−h_t}` on the data rows
     and 1 on the dummy rows: `Ω̄ = (Ω⁻¹ + X̃ᵀX̃)⁻¹`, `B̄ = Ω̄(Ω⁻¹b + X̃ᵀỸ)`, `Ψ̄ = Ψ + ỸᵀỸ + bᵀΩ⁻¹b − B̄ᵀΩ̄⁻¹B̄`,
     `d̄ = d + T + m` (`X̃, Ỹ` the weighted rows stacked with the `m` dummy rows); `Σ ~ IW(Ψ̄, d̄)` by the Bartlett
     decomposition, then `B | Σ ~ MN(B̄, Ω̄, Σ)` (the engine draws the recursive structural `(θ_i, λ_i)` per equation).
  2. `h | B, Σ, ρ, σ_h, Y`: five elliptical slice sampling updates of the whole path per sweep (Murray, Adams &
     MacKay 2010), with the stationary AR(1) prior as the Gaussian prior (its draws by the AR recursion) and the
     log-likelihood `Σ_t [−(n/2)h_t − ½e^{−h_t}q_t]`, `q_t = u_tᵀΣ⁻¹u_t` (the engine: blocked independence MH with a
     tridiagonal Newton proposal).
  3. `ρ | h, σ_h`: univariate slice sampling (Neal 2003; stepping out with width 0.1, at most 50 steps, then
     shrinkage) on `(−1, 1)` of `log π(ρ) + ½ log(1 − ρ²) − (1 − ρ²)h_0²/(2σ_h²) − Σ_{t≥1}(h_t − ρh_{t−1})²/(2σ_h²)`
     (the engine: a truncated-normal independence MH); then `σ_h² | h, ρ ~ IG(5 + T/2, 0.04 + S/2)`, with
     `S = (1 − ρ²)h_0² + Σ_{t≥1}(h_t − ρh_{t−1})²`, the closed form, written independently; then a non-centred update
     of `σ_h` (Papaspiliopoulos, Roberts & Sköld 2007): with `x = h/σ_h` held fixed, univariate slice sampling as for
     `ρ` (width 0.5, at most 50 steps, then shrinkage) of `s = log σ_h` under `−2c₀s − C₀e^{−2s} + Σ_t[−(n/2)e^{s}x_t −
     ½e^{−e^{s}x_t}q_t]` (`c₀ = 5, C₀ = 0.04`: the `IG(c₀, C₀)` prior of `σ_h²` written in `s` with its Jacobian, times
     the likelihood at `h = e^{s}x`; the prior of `x` does not involve `σ_h`), then `h = e^{s}x`. Each move leaves the
     posterior invariant; the engine draws only the centred `σ_h² | h, ρ` (§5.4; Q34 is its interweaving remedy).
- **Rows**: `x_t = (1, y_{t−1}ᵀ, …, y_{t−p}ᵀ)ᵀ`, the engine's regressor order. With `ȳ₀` the mean of the `p`
  presample rows, one SOC row per variable `j` with a non-zero own-lag mean (`soc_scope="persistent"`; all three in
  C2), `y_d = e_jᵀȳ₀_j/μ` and `x_d = (0, y_d, …, y_d)`, then one SUR row `y_d = ȳ₀ᵀ/δ` and `x_d = (1/δ, y_d, …, y_d)`
  (SPEC §4.6, GLP 2015), so `m = 4` in C2.
- **Independence**: the oracle is a NumPy module under `tests/parity/oracles/` that imports nothing from `rai_bvar`
  (it lags the data, builds `Ω`, `b`, `Ψ` and the dummy rows from SPEC §4.6's formulas in its own code; SciPy only
  for special functions); its author wrote no Phase 3 kernel and reads neither the engine's CommonSV code
  (`components/vol/common.py`, the shared-factor path, `linalg.shared_factor_*`, `tridiag_newton`) nor the bodies of
  WP30's CommonSV references, only this design, SPEC §4.6 and, for (b)'s cross-check, the public interface of Phase
  1's analytic engine. This is stricter than E58: WP30's author may not write it, having written the `common_sv`
  reference bodies; WP35's author may write it only if WP38a's PR is opened before that author has read any of the
  code listed here, which the PR records; otherwise the author has no WP30 or WP35 role. Every prior
  number is stated in the run's fixture and read by both sides (E11: one parameter set, both sides derived from it).
- **Its own validation** (WP38a's PR, before any comparison): (a) a Geweke joint-distribution test of the oracle at
  P2 §9.5's G-small sizes (`C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3`, Bonferroni over its test functions) on `n = 2,
  p = 1, T_eff = 20` with SUR rows (`δ = 1`, presample row `(0.5, −0.5)`, `λ = 0.2`, `α = 2`, `ψ = (1, 1)`, own-lag
  means 0.5, `var_const = 1`, CommonSV's defaults; the non-centred move is part of the sweep it tests), the test
  functions bounded (E148: `atan` of the `B` entries, of `log Σ_ii`,
  of `h` at three dates and of mean `h`; `ρ`, `σ_h`); (b) block 1's moments against the closed-form
  matrix-normal–IW moments computed in the test (1e-10 for the posterior parameters; `|z| ≤ 4` Bonferroni for 10⁵
  draws), and a cross-check that is not circular: at `h ≡ 0`, and at one fixed non-zero `h` path with the data rows
  rescaled by `e^{−h_t/2}` and the dummy rows unscaled, block 1's `(B̄, Ω̄, Ψ̄, d̄)` equal Phase 1's conjugate NIW
  posterior to 1e-10 on C1's and C2's prior settings (the test calls Phase 1's analytic engine, never CommonSV engine
  code; the oracle module itself imports nothing from `rai_bvar`); (c) block 2 for `T ∈ {2, 3}`, `n = 2`, at `ρ =
  0.9`, `σ_h = 0.3` and `q = (1.5, 0.4, 3.0)` (its first `T` entries): the `h` posterior by tensor quadrature against
  its draws (KS, family-wise `α = 1e-3`, D-60); (d) block 3, on one `T = 20` path `h` drawn from the AR(1) at `ρ =
  0.9, σ_h = 0.3` (recorded seed), `n = 2`, `q_t = 2`: `p(ρ | h, σ_h = 0.3)` and the non-centred `p(log σ_h | x =
  h/0.3, q, ρ = 0.9)` by quadrature against their draws (KS).
- **Runs** (fixtures `tests/parity/fixtures/commonsv_oracle_*.nc`, written by `tools/make_commonsv_oracle.py` with
  recorded seeds, the commit and the NumPy/SciPy versions): **C1** `qd3_growth.csv` (`n = 3, p = 4`, the bvars data,
  E146) without dummy rows; **C2** a dataset simulated from this model (`commonsv_n3.csv`, `n = 3, p = 2`, `T = 200`
  data rows after the `p` presample rows; written by the same tool with NumPy `default_rng(20261026)`, the draw order
  the tool's and the committed file the dataset of record): `y_t = c + A₁y_{t−1} + A₂y_{t−2} + e^{h_t/2}Lε_t`, `c =
  (0.2, 0.1, 0.3)`, `A₁ = 0.5I`, `A₂ = 0.1I`, `Σ = LLᵀ` with unit variances and correlations 0.3, 0.1, 0.2 for the
  pairs (1, 2), (1, 3), (2, 3), `h_t = 0.9h_{t−1} + 0.15η_t` started from its stationary law, `y` started at 0 with
  the first 50 rows discarded; with SOC and SUR rows (`μ = δ = 1`). Prior
  numbers per run: `λ = 0.2`, `α = 2`, `ψ` and the own-lag means stated in the fixture (C1: the sample variances of the
  series and means 0; C2: `ψ = 1` and means 0.5), `var_const = 1e7`, CommonSV's defaults. Oracle: 4 independent runs ×
  200,000 kept sweeps after 10,000, MCSE from the per-run ESS, fixed here and never lengthened (at 4 × 50,000, R5's
  scratch oracle failed the gate for `σ_h` on C1 even with the non-centred move: E168; about 0.6 ms per sweep, so
  ≈ 9 min per run on one core). Predictive: one path per kept sweep, `h_T, …, h_{T+3}` by the AR(1) from `h_{T−1}`,
  `u ~ N(0, e^{h}Σ)`, the regressors from the last `p` data rows, from a separate recorded seed stream; the engine's
  `forecast(4)` with one path per kept draw. Engine: the run that decides is the weekly one, 4 chains × 200,000 kept
  after 10,000, the oracle's length (Q41, resolved by the owner on 2026-10-26: "Sabit 4 × 200.000, şimdiden
  (Recommended)"; E169): fixed here before any engine or oracle output exists, unconditional and never lengthened, as
  D-35 fixed a higher count instead of sampling until a gate passes. The PR short run
  (4 × 6,000 after 1,000, P2 §9.10) is a smoke test: every draw finite, every compared `z` finite, and split R-hat
  ≤ 1.05 for the coefficient and covariance quantities only (`B` and `Σ` here; `Π` and `A` for the `bayesianVARs`
  runs; Phase 2's `test_bayesianvars.py` guards `Π` the same way). The R-hat of `h_t`, `ρ`, `σ_h`, the
  hyperparameters and the predictive is reported, not gated: at this length the engine's centred `σ_h` step fails
  1.05 on C1 for most seeds (R5's re-check). Its comparison is reported, never gated by the rule below, and never
  decides the criterion.
- **Comparison** (WP38): the 5/25/50/75/95% quantiles of every `B` and `Σ` entry (reduced form; the engine's `Σ =
  A⁻¹ΛA⁻ᵀ`), `h_t` at `t ∈ {0, ⌊T/4⌋, ⌊T/2⌋, T − 1}`, `ρ`, `σ_h`, and the 1- and 4-step predictive quantiles of each
  variable, under the Phase 2 parity rule with the combined quantile-MCSE. A compared quantity whose split R-hat
  exceeds 1.01 or whose bulk ESS is below 400 on either side (the oracle's fixture, the engine's decisive run) is
  **not met** (Q39(b)'s rule, above). The criterion is met iff every compared quantity of C1 and C2 passes on the
  decisive run; a failure is not met (D-59) and is reported, never retuned.

**BoE `bvar` 0.3.5**: the per-draw oracle A10b (its end-to-end conditional forecast re-draws the parameters, D-50).

### 9.9 DGP verdict tests (SPEC §10.5; exit criterion)

Built by the parallel DGP session: `validation/dgp.py`, `benchmarks/dgp_study.py`, `tests/dgp/` and the pre-registration
`benchmarks/reports/dgp_study_preregistration.md` (one design cell: `n = 3, p = 2, T = 200`, volatility axis only; rows
`linear-homoskedastic`, `linear-sv`, `linear-sv-outliers`; 200 replications per row; readings fixed in advance; arms
hash-pinned, and those hashes are in WP26's golden test, so Phase 3 changes no arm). Phase 3 neither duplicates nor
changes it. **The criterion is met** only if the registered full study (all 200 replications per row, from one clean
commit) has been run and its readings pass: both null pairs of `linear-homoskedastic` and the `not_worse` pair of
`linear-sv-outliers` are "not above nominal" (the registration's Wilson rule) at `h = 1` and at `h = 4`; power
(`linear-sv`) and the gate-pass share are reported, not gated, as registered and as SPEC §10.5 asks ("Size … and power are
reported"); a power threshold set now would be post hoc (a pilot record exists), so Q33 asks the owner to confirm this
reading (and that the exit is met on the one registered cell `n = 3, p = 2, T = 200` of SPEC §10.5's range, other
cells left to a later registration; E142), and WP41 states it in SPEC §13's Phase 3 status. Otherwise not met (D-59).
The full study takes about 10 h on the reference laptop by the registration's own estimate; the owner approved it on
2026-10-25 (Q28, resolved), and the registered run is the one at the registration's commit `58ab09f`, after the pilot
(E138). The readings refer to that commit (Phase 2 code); if the DGP PR changes harness code before it merges, the
registration's bug-fix rule applies (the affected rows are rerun in full). WP39 records the readings in this document's
exit map; SPEC §13 is WP41's (E139). Phase 3 rows of the study (CommonSV or GL arms) would be pre-registered separately and are not
part of the exit.

### 9.10 Replication target (SPEC §10.7): Gruber & Kastner (2025), qualitative (WP40)

A pre-registration (`benchmarks/reports/gruber_kastner_preregistration.md`) is committed before any score exists, with
the budget and compute estimate from WP36. It fixes:

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
  increase is measured as the summed `junit.xml` durations of the new Phase 3 tests within each PR-tier job (wall-time
  deltas are noise: cold jobs vary 40–85 min by CPU model, P2 §9.10) and capped at +10 min; beyond the cap, tests move
  to the weekly tier from the end of this priority list: Tier U, A10/A10b, A11, A12, A13, per-PR block Geweke, A9,
  G-small 320, 301, 311, 321, A8 (`n = 3`), contract tests, short parity runs. This amends P2 §9.10's "no test leaves
  the PR tier to meet it" for Phase 3 tests only.
- **Weekly tier**: G-full and S for rows 301–330, G-small for 401–408, M19–M36, A8 (`n = 20`), A9's MCMC leg, the
  ordering-invariance test of conjugate × CommonSV (row 27; E159), the long
  parity runs, contract tests for every new MCMC composition.
- **Shards** (WP35, `tools/weekly_shards.py`): `geweke-p3a` (structural rows), `geweke-p3b` (reduced and CommonSV rows),
  `sbc-p3a`, `sbc-p3b` (same split, by file); the Phase 2 shards `--ignore` the Phase 3 files; each shard ≤ 200 min on a
  2-vCPU EPYC 7763, split further if not; the partition check covers them; the `rest` shard's growth (mutants, weekly
  contract tests, A8 at `n = 20`) is measured on the first dispatch and split the same way. Weekly runs a work package needs are
  dispatched by hand on its branch.

---

## 10. Performance

Estimates to be replaced by WP36's measurements (`docs/phase3/pilot.md`):

- **SSVS indicators** (E155): `γ` is stored as `int8` with the stored parameter draws, `n·k` bytes per stored draw
  (`n = 20, p = 4`: 1.6 kB per draw, 65 MB at 4 × 10,000 stored draws).

- **GL scale step**: elementwise work on `n²p` coordinates per chain plus `G` segment sums and an `A × G` grid. Under
  `vmap` the GIG `while_loop` runs to the slowest lane: with the measured rejection constant ≤ 1.40 (R2a; bound 3.46),
  the maximum over 10⁴ lanes is about 7 attempts (27 at the bound), so the step costs about 7 elementwise passes
  (≈ 0.2–0.6 ms per sweep of 4 chains at `n = 20, p = 4`, against BENCH-P2's 3.90 ms). Measured by WP31 (E163): 0.28 ms
  (SSVS) to 17.6 ms (R2D2), each GIG line ≈ 7.4 ms; WP36 re-measures on an idle machine, whether or not E163's
  `random.py` options land. The deviation path costs D-14's flops plus elementwise logs. GL ×
  Constant loses the eigen path (0.35 → ≈ 3 ms per sweep).
- **Conjugate CommonSV** (shared factor): per chain and sweep one Gram `O(T q²)`, one Cholesky `O(q³/3)` and `n`
  pairs of triangular solves `O(n q²)`. `n = 20, p = 4` (`q = 100`): ≈ 4·10⁶ flops, ≈ 0.3–0.6 ms per sweep of 4 chains
  with the `h` blocks (measured by WP32, E170: the `h` step 0.87 ms and the shared-factor equation phase 0.61 ms, so
  1.52 ms standalone and 3.88 ms inside the runner's loop; WP36 re-measures); `n = 100, p = 4, T = 300` (`q = 500`):
  ≈ 2·10⁸ flops, ≈ 3–6 min for 4 × 11,000 sweeps on one
  device (the D2 screen; structural SV is `O(n q³) ≈ 1.25·10¹⁰` flops per sweep there). The per-equation program with
  other priors costs what structural SV costs.
- **BENCH-P3** (reported, not gated; SPEC sets no Phase 3 target), `benchmarks/phase3_perf.py`: the conjugate CommonSV
  at `n = 100, p = 4, T = 300`, 4 × 10,000 kept draws; HS × SV at the BENCH-P2 shape (overhead against BENCH-P2).

---

## 11. Risks

| # | Risk | Mitigation |
| :- | :--- | :--- |
| R18 | GL mixing (HS global scale; small `a` on the grid) leaves fits unconverged | WP36 pilot; group scales primary (D-52); budget D-58; warnings |
| R19 | GIG sampler wrong at extreme parameters | Devroye (2014) §6 with a uniform bound, log-space evaluation, the pinned Tier U grid, M34 |
| R20 | `bayesianVARs` cannot be built, or a probe stays undecided | D-54 (Linux with gfortran); an undecided probe is recorded with its evidence and its runs are **not met** (D-59); the owner decides |
| R21 | bvars is approximate (auxiliary mixture) or its `V` cannot be fixed | P12–P15 select the comparison before it runs (§9.8); P16 decides whether branch 1's `h_t`, `ρ`, `σ_h` part is met; branch 2 is evidence (Q35, E157, E159, E167) |
| R22 | CommonSV `h` acceptance falls with `T` and `σ_h²` (N08 §A5.2) | blocks of 100 dates; acceptance warning; pilot at `T = 600` |
| R23 | Conditional forecasts near rank deficiency | the `R₁` diagonal check of E-16; never jitter |
| R24 | CI time grows with 30 rows × (G, S) | four new weekly shards; four PR rows; the +10 min PR cap |
| R25 | Contract additions change Phase 2 draws | `Empty` defaults; golden templates and hashes; bitwise before/after tests |
| R26 | Parallel sessions edit shared files | ownership of §1.2; rebase on their merges; `ci.yml` edits named per WP |
| R27 | GK sample details (UMCSENTx gaps) | the sample start fixed in the pre-registration |
| R28 | GitHub Actions minutes for R builds | dispatch-only job with a cached library |
| R29 | The shared-factor path's two-block `(θ_i, λ_i)` Gibbs mixes slowly when `q_i ≈ T` | WP36 measures; Q32 (a collapsed `λ_i` draw) |
| R30 | The DGP full study's compute (≈ 10 h) | Mitigated: approved by the owner (Q28, 2026-10-25); run at the registration's commit after the pilot (E138) |
| R31 | The CommonSV level shift `(λ_i e^c, h − c)` is identified only by the zero-mean `h` prior and may mix slowly for `ρ` near 1 | WP36 measures the ESS of `log λ_i` and `mean h`; Q34 |

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
| Q21 | Default MCMC budget of GL compositions | Decide from WP36 (D-58); gate unchanged |
| Q22 | R environment for the oracles, and permission to download it | **Resolved (owner, 2026-10-25): "(a) CI job + artifact"**: the dispatch-only `r_fixtures` job and `gh run download` of its artifact; no Docker image, no local R (D-54) |
| Q23 | GL defaults: GK's semi-global with hyperpriors, or `bayesianVARs`' global with fixed `a` | GK's (§3.2); parity sets `bayesianVARs`' values explicitly |
| Q24 | CommonSV extensions (Student-t, random walk, outliers) and the t part of the bvars oracle row | Defer (N08 §A5.4, §A9) |
| Q25 | Conditional forecasts: soft conditions, parameter updating (Waggoner–Zha Gibbs), weighting by `w_s` | Defer; the weights are stored |
| Q26 | Compute for the GK acceptance run | Estimate from WP36; run locally after the pre-registration, with a documented quick config first |
| Q27 | Keep D3 for `Minnesota(form="auto") × CommonSV` (structural), or make the conjugate screen the `auto` default (D2's spirit, a SPEC D3 change) | Keep D3; the conjugate screen is one keyword away |
| Q28 | Compute for the DGP full study (≈ 10 h, the registration's estimate) | **Resolved (owner, 2026-10-25): "evet 10 saatlik tam DGP koşusunu başlat"** (yes, start the full 10-hour DGP run): run at the registration's commit `58ab09f`, after the pilot (E138) |
| Q29 | "Recovery" in the DGP layer (SPEC §10.1) | Defer (posterior coverage of the true parameters needs fits the study does not keep); SPEC §10.1 edit |
| Q30 | Deterministic terms outside the shrinkage (D-36) | Keep (GK, `bayesianVARs`); SPEC §4.4 edit |
| Q31 | The GK acceptance cannot fail ("met or documented"), as with Clark–Ravazzolo | Accept, as in Phase 2 |
| Q32 | A collapsed `λ_i | h, y` draw for the shared-factor path (better mixing at `q ≈ T`) | Later, if WP36 shows the need; it needs the mean step to draw a vol-state leaf |
| Q33 | The DGP exit gates size (null pairs and the `not_worse` pair) and reports power and the gate-pass share, as registered and as SPEC §10.5 words it; and it is met on the one registered cell (`n = 3, p = 2, T = 200`) of SPEC §10.5's range, other cells left to a later registration (E142) | Confirm both; WP41 states them in SPEC §13 |
| Q34 | An interweaving (ASIS-type) shift move for the CommonSV level `(λ_i e^c, h − c)` | Later, if WP36 shows slow mixing of `log λ_i` or `mean h` |
| Q35 | What "bvars parity for CommonSV" (SPEC §13) means when §9.8's branch 2 is selected (no exact prior mapping: bvars's `V` cannot be held fixed, or its SV prior has no `CommonSV` counterpart). SPEC §10.4 prescribes comparing forecast distributions on the same data then; the question is whether that equality comparison of two different posteriors gates the criterion (E121) | **Resolved (owner, 2026-10-25): "Q35 kanıt olsun"** (let it be evidence): branch 2 is reported as evidence, not gated; under it "bvars parity for CommonSV" is **not met** for want of an exact mapping (E167) |
| Q36 | Rows 320 (238 of 10⁶ draws) and block 408 (239) exceed D-56's explosive-mass target `2e-4` under E150's Monte Carlo rule (a count above the upper 99% Poisson limit at the target, 234), from the SOC/SUR dummy rows of the conjugate test prior; every row meets Phase 2's bound `P(ρ > 1) < 10⁻³`. Two decisions: (i) accept the Monte Carlo rule as the definition of D-56's target, which states none (rows 321 and 323 are then within it); (ii) for rows 320 and 408: (a) tighten their lag prior (for example the conjugate `Ω_rr` from 0.01 to 0.008) until WP30's author measures them within the target at 4·10⁶ draws, before any Geweke run of them; (b) keep the priors and accept their masses (≈ 2.4e-4); (c) drop the dummy rows from 320 and 408 (which removes M32's and M36's G catcher) | (i) accept; (ii) (a): the target protects Geweke from explosive simulated paths, and no sampler result exists yet for these rows. Until answered the rows stay as stated and WP30's test marks them strict expected failures (E150) |
| Q37 | Add a reversed-ordering twin of R1 (reduced × SV, `n = 3, p = 2`, the same matched priors and the same Phase 2 criterion) to WP29's `bvars` section and WP38, per N08 §B5 item 7 (row 28, E156) | Defer past v0.3.0 (scope: SPEC §13 does not require per-ordering parity). The alternative is cheap: the R1 fixture machinery and the P3 row map serve a permuted dataset unchanged |
| Q38 | A9's projected-gradient criterion with free `cross`: keep Phase 1's stationarity rule as the criterion, or tighten the free-scale optimiser (E160) | **Resolved (owner, 2026-10-25): "Toleransları sıkılaştır (Recommended)"**: tighten the free-scale path's tolerances; the 1e-6 criterion stays |
| Q39 | bvars's `Σ` fails its hard mixing check in both volatilities at §9.8's draw counts, and the R fixtures do not reproduce bit for bit from their seeds (E165): (a) lengthen the bvars runs (for example × 4, ≈ 6 h of the dispatch-only job) in a WP29 follow-up before WP38; (b) keep them, and WP38 treats a fixture that fails its hard mixing check as not met (D-59) | **Resolved (owner, 2026-10-25): "Q39 için (b)"**: the bvars runs stay as they are, and WP38 treats a fixture that fails its hard mixing check as not met (E167) |
| Q40 | Under Q35 the SPEC §13 criterion "bvars parity for CommonSV" cannot be met (branch 2 for both volatilities): write an independent exact oracle, as Phase 2 did for its reduced-form sampler, or record the criterion as not met (E168) | **Resolved (owner, 2026-10-25): "Bağımsız kesin oracle yaz (Recommended)"**: the exact `CommonSV` oracle of §9.8, WP38a; SPEC §13 and §10.4 amended |
| Q41 | The engine's decisive `CommonSV` run (4 × 50,000 after 10,000, never lengthened) may fail §9.8's mixing gate for `σ_h` from mixing alone: its `σ_h² \| h, ρ` step is centred (§5.4), and R5's scratch oracle with a nearly exact `h` (50 elliptical slice updates per sweep) gives the centred floor on C1, `σ_h` IAT ≈ 446, so ESS ≈ 450 at 200,000 draws with split R-hat near 1.01; C2's posterior also puts `ρ` near 1. (a) keep it: a failing gate is not met; (b) fix one longer engine count for C1–C2 now, before any engine or oracle output (for example 4 × 200,000 after 10,000, as the oracle; ≈ 1.1 ms per 4-chain sweep at C1's size on the development machine, ≈ 4 min per run, the 2-vCPU weekly runner slower), as D-35 fixed a higher count instead of sampling until a gate passes | **Resolved (owner, 2026-10-26): "Sabit 4 × 200.000, şimdiden (Recommended)"** (fixed 4 × 200,000, from now): C1 and C2 are decided on engine runs of 4 × 200,000 kept after 10,000, the oracle's length (E169) |
| Q42 | bayesianVARs' SV predictive at `h ≥ 2` disagrees with the exact SWA oracle on the same data and prior while every parameter block agrees (E173): compare `predict()`'s output as it is, or compute the R-side predictive from bayesianVARs' own posterior draws with the path recursion | **Resolved (owner, 2026-10-26): "Kendi çekimlerinden yol ile hesapla (Recommended)"** (compute it from its own draws with the path): E173 |
| Q43 | A10's statistical part failed on the two analytic setups at the pre-registered seed with 1,000 draws (E174): (a) a new seed; (b) more draws, same seed; (c) the known-variance product statistic; (d) record them as not met | **Resolved (owner, 2026-10-26): "(b) Daha çok çekim, aynı seed (Recommended)"**: 10⁴ draws for the two analytic setups, seed unchanged; the outcome is binding (E174) |
| Q44 | Eight Phase 3 SBC rows fail the per-replication ESS rule at `t_max = 200` while the γ test is clean (E179): a per-row `t_max = 400`, or record those rows as not met | **Resolved (owner, 2026-10-26): "Bu 8 satırda t_max = 400 (Recommended)"**: `t_max = 400` for rows 304, 306, 308, 311, 315, 325, 326 and 327, fixed before the rerun; the result is binding (E179). Result: six of the eight pass; 308 and 325 are not met (pooled ESS) |
| Q45 | Once PR #31 is on `main`, SBC rows 308 and 325 (not met, E179) turn the weekly CI red every Sunday; each runs alone in its shard, so it hides no other failure (E180) | **Resolved (owner, 2026-10-28): "Merge, sonra dar xfail (Recommended)"**: a narrow strict xfail for the recorded reason only (pooled ESS); the criteria do not change (E180) |

---

## 13. Work packages (dependency-ordered)

**Rules** (P2 §13 rules 1–5 apply). Each WP owns its files (§1.2) and tests (§1.3); the contracts of WP26/WP27 freeze
when their PR is open and change only through an amendment recorded here; kernel WPs pass Tier U and B before their PR;
WP30 and WP35 are written by an author who wrote no Phase 3 kernel (recorded in the PR), and WP38a by an author
independent under §9.8's stricter rule; every WP keeps the Phase 1–2 suites green except the amendments of §1.3;
WP26–WP34 record the same-machine bitwise A/B of §2; clean room throughout. One PR per WP (D-57). "Needs" lists the real
dependencies; "Base" is the branch the PR targets: the stack is linear because the WPs share files, so a base can be a
stacking choice rather than a dependency. WP29 and WP38a target `main` directly and are merged before WP38 (D-57). WP31
and WP32 run their acceptance G-small rows ad hoc through `validation.joint` (the references of WP30) and record the
results in their PRs; the suite files that run them in CI are WP35's.

| WP | Content | Needs | Base | Acceptance |
| :- | :--- | :--- | :--- | :--- |
| WP26 | Contracts (§2): config rules, state classes, prior arrays, keys, registry, protocols, compose hooks, goldens, its §1.3 amendments | this design | `main` (`phase-3-design` merged as PR #13) | rules tested; Phase 2 templates, hashes (and the DGP arms') unchanged; bitwise A/B |
| WP27 | Specs and API (§7): GL classes, `DiscreteGrid`, `AsymmetricConjugate`, `CommonSV`, `TruncatedNormal`; resolution, engine table, E-1…E-16, spec/hash, labels, harmonise; its §1.3 amendments | WP26 | WP26 | every §7.2 row and §7.3 message tested; golden hashes unchanged; `import rai_bvar` imports no JAX |
| WP28 | Numerics (§3.4, §3.5, §5.3, §5.4): `log_gig`, `precision_draw_log`, `shared_factor_*`, tridiagonal Newton; mpmath in `dev` (`pyproject.toml`, `uv.lock`) | WP26 | WP27 | §9.2 item 1 green (the KS family rule of D-60); M34 caught by it (checked locally, E124) |
| WP29 | Oracle environment and fixtures (D-54, §9.8): `make_r_fixtures.R` sections `bvars` (P1–P10, R1–R9) and `wbvars` (P11–P16 and its runs), packing, the dispatch-only `r_fixtures` CI job, the SV-t fixture if `stochvol` ≥ 3.2.5 | this design (Q22 resolved) | `main` | fixtures committed with versions; every probe decided or recorded undecided with evidence; the bvars selected comparison recorded |
| WP30 | References (§9.1), the exogenous design, `COMPOSITIONS_P3`, `BLOCK_CONFIGS_P3`; explosive-mass measurement of the lagged rows; `gl_log_r_reference`, `ssvs_shat_reference`, `conditional_forecast_reference` (E126, E144) | WP26 | WP28 | prior moments vs closed forms; simulators vs NumPy loops (1e-12); `log_joint` vs SciPy (1e-10); the three references against direct formulas |
| WP31 | GL kernels and `GLScaleStep`, deviation paths, adapter and builders, resolve, monitors, runner init order, storage (§3) | WP28, WP30 | WP30 | Tier U (including §9.2 item 5, E126), B green; rows 301 and 311 G-small green (ad hoc); A12, A13 green; bitwise A/B |
| WP32 | CommonSV (§5): common cores, `CommonSVStep`, the shared-factor path, forecasts, warm starts, monitors, storage; the CommonSV false-alarm control (row 26) | WP28, WP30, WP31 (shared files) | WP31 | Tier U, B green; rows 320 and 321 G-small green (ad hoc); A8 (`n = 3`), A11 green; the R8-note test (E130); bitwise A/B |
| WP33 | `StructuralNIGFit` and its adapter, EB with free `cross`, MH, `lag_coefs` on request on analytic forecasts and its plumbing (D-49, E120), ordering support, its §1.3 amendment (§4) | WP27 | WP32 | A9 green; Phase 1 tests unchanged; the default analytic forecast carries no `lag_coefs` and keeps its memory (E120); bitwise A/B including the EB plug-in and analytic cases of §2 (E118) |
| WP34 | Conditional forecasts (§6) | WP32, WP33 | WP33 | A10, A10b green on every engine; contract checks; bitwise A/B |
| WP35 | Composition suites (§9.4–§9.6): rows 301–330, blocks, SBC, mutants M19–M36, contract tests, the ordering-invariance test (row 27), weekly shards | WP30–WP34 | WP34 | G-small (PR rows) green; G-full, S and M green on a dispatched weekly run; every mutant caught. Result: S green except rows 308 and 325, not met (E179) |
| WP36 | Pilot: GL and CommonSV mixing and cost on the validated kernels, BENCH-P3 (§8, §10) | WP35 | WP35 | measurements with machine fingerprint; recommendation for D-58; compute estimate for Q26 |
| WP37 | Twins at scale: A8 (`n = 20`) (A9's MCMC leg is WP33's, E162) | WP35 | WP36 | green weekly |
| WP38a | The exact `CommonSV` oracle (§9.8, Q40): `tests/parity/oracles/commonsv_exact.py`, its validation (a)–(d), `tools/make_commonsv_oracle.py`, the C2 dataset, fixtures C1–C2 | Amendment 2 | `main` (independent of the stack, like WP29) | the oracle's own validation green; fixtures committed with seeds and versions; the PR records the author's independence under §9.8 (no Phase 3 kernel written; none of the listed CommonSV code or WP30's CommonSV reference bodies read) |
| WP38 | Parity tests (§9.8), including CommonSV against WP38a's oracle | WP29 and WP38a (merged), WP37 | WP37 (`main` merged into the stack after WP29 and after WP38a, D-57) | the parity exit criteria (map below) |
| WP39 | DGP exit (§9.9): the DGP session's merged PR and its full study's readings | the registered run's report (at `58ab09f`) and the DGP PR merged (E138) | WP38 | the readings recorded in this document's exit map; criterion met or not |
| WP40 | GK replication: pre-registration, then the acceptance run and report (§9.10) | WP36 (budget), WP38 | WP39 | report against C1–C6 |
| WP41 | Release: SPEC v2.4 (status, the §0.2 edits the owner accepted), README roadmap, this document's exit map, `ci.yml` tiers, `pyproject` 0.3.0, release notes; tag only on the owner's approval after a status report | all | WP40 | CI green; the weekly tier green on the release commit with no skipped parity test |

### Exit-criteria map (SPEC §13 Phase 3)

| Criterion | Delivered by |
| :--- | :--- |
| `bayesianVARs` parity, global-local priors × SV | WP38 runs R3–R9 (fixtures WP29, probes P6–P10); pass/fail (D-59) |
| `bayesianVARs` parity, reduced-form sampler, after P1–P6 | WP38 runs R1–R2 (fixtures WP29); pass/fail |
| `CommonSV` parity with an exact independent oracle (SPEC §13 as amended by Q40; first written "`bvars` parity for `CommonSV`") | WP38 against WP38a's exact oracle (§9.8, runs C1–C2; pass/fail). The bvars comparison selected by P11–P16 stays as evidence: under branch 2 it is not met for want of an exact mapping and is reported (Q35, E167); A8 and A11 are evidence only |
| DGP verdict-engine tests on linear DGPs | WP39: the registered full study's readings (§9.9); pass/fail |
| R6: Geweke and SBC for every new block and interaction class | WP35 (references WP30), with the coverage argument of §9.4. SBC rows 308 and 325 not met (E179) |
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
