# Phase 2 design: the JAX MCMC engine (final)

**Status.** Final design for SPEC §13 Phase 2, written 2026-10-22 on branch `phase-2-mcmc` (Phase 0 and Phase 1
merged). It replaces the three competing proposals (correctness-, performance- and extensibility-first) and is
the document the implementing agents work from.

**How it was produced.** Both judges ranked the correctness-first proposal first (totals 48 and 45), so it is
the base: decision register, pure kernels with injected noise, the plan as data, the Mode I quadrature oracle,
the validation tiers. Onto that base this document grafts:

- from the performance proposal: the packed Khatri–Rao Gram (re-measured by both judges), the two exact
  Constant fast paths, reuse of the indicator `logsumexp` in the exact correction, EB through the Phase 1
  engine, the primary/secondary monitor split, the BENCH-P2 protocol and a pilot, device-side latent moments
  with a storage plan, and the factored LPD kept internal to `result.py`;
- from the extensibility proposal: a covariance structure that owns `a_i` and its prior, a separate
  coefficient-prior kernel, pre/post global steps with `eq_map`, `Batching.MAP`, the particle-filter hooks
  (`obs_logdensity_eq`, `transition_eq`), the in-loop forecast hook, a validation reference registry,
  deterministic replay for failure diagnostics, accumulators owned by components, and the Chan IW-equivalent
  prior under its own label.

Every technical error either judge listed is fixed; §0.2 maps each finding to its resolution.

**Reading guide.** §0 decisions and errata · §1 module layout · §2 state pytrees · §3 block protocols and the
Phase 2 kernels · §4 the Gibbs sweep · §5 chain runner, storage, PRNG · §6 priors · §7 forecasting, LPD, IRFs ·
§8 diagnostics · §9 validation · §10 performance · §11 API · §12 risks and open questions · §13 work packages
(dependency-ordered; the document ends there).

**Conventions.** `N05 §7.2` means `docs/phase2/05_structural_equation_sampler.md`, section 7.2; `SPEC §4.1`
means SPEC.md. Code indices are 0-based: equation `i ∈ {0..n−1}`, data row `t ∈ {0..T_pad−1}`. Prior formulas
quote the notes' 1-based equation number `ι = i + 1`. `k = n_d + n·p`; `q = k + n − 1` padded structural
coefficients; `P = q(q+1)/2`, `P_x = k(k+1)/2` packed Gram lengths; `T_eff` real rows; `T_pad =
t_bucket·⌈T_eff/t_bucket⌉` compiled rows (`t_bucket = 32`); `m_t = tmask_t = 1{t < T_eff}`. All floats are
float64 (R4); the only exception mechanism is a component's declared `float32_leaves` (empty in Phase 2).

**Clean room.** Apache-2.0. Every algorithm is specified by reference to the notes, which derive from papers
and package documentation. No code is translated from GPL sources. R packages (`stochvol`, `bayesianVARs`) are
numerical oracles only, run offline by `tools/make_r_fixtures.R` (SPEC §2.1, §10.4).

---

## 0. Decisions

### 0.1 Decision register

| ID | Decision | Source | Notes / supersedes |
| :--- | :--- | :--- | :--- |
| D-1 | **Exact mixture correction on by default** (`SV(exact=True)`): the `h` draw and the ASIS non-centred `(μ, σ)` move are MH steps with log weight `w(h) = Σ_t m_t[log N(ẽ_t; 0, e^{h_t}) − log g(y*_t − h_t)]`. `exact=False` is the stochvol/bayesianVARs parity mode | N01 §10.2–10.3; DNP 2015 Alg. 3 | N02 §2.4/§9 "exactness requires c = 0" holds only for N02's `log f − log g` form; in the Gaussian-`ẽ` form the invariant law is exact for any `c ≥ 0` |
| D-2 | Offset `c_i = c_rel·ψ_i`, `c_rel = 1e-12`, fixed per run and window (R9); `y*_t = logaddexp(log ẽ_t², log c_i)`; absolute offsets (`0`, `1e-4`, `1e-3`) are parity options | N01 §0 item 3, §7.2 | bias ≈ 2.5e-6; no `log 0` while `ψ_i > 0` |
| D-3 | OCSN 10-component mixture, effective means stored once (no extra −1.2704 shift); KSC-7 only as `mixture="ksc7"` | N01 §5.3, §6 | — |
| D-4 | Indicator draw by **inverse CDF** on max-subtracted log weights, one float64 uniform per `(eq, t)`, `≤` comparison, clamp to `J−1`; frozen | N07 §9.2 (S3) | 3.5–4.4× faster than Gumbel-max |
| D-5 | SV core: joint `(T+1)`-state AWOL system computed in **non-centred** form (`Q̃ = σ²Q`), two `lax.scan`s; GIS-C ASIS in the rotated order `s → h → θ_C → θ_NC` | N02 §3.2, §5.2–5.3, §8; N07 §8.3 | RW path amended (`f60d736`, `4fc0162`; WP25 queue item 7): still the scaled form `Q̃ = σ²Q`, but factored as a path-graph Laplacian plus a diagonal, with the pivots carried as their excess over the next edge, in **three** `lax.scan`s (`linalg.laplacian_precision_sample`: pivots, forward, reverse); the AR path keeps the two scans (§3.7.4) |
| D-6 | Centred `(γ, φ)` step: 2-block MH with **centred** cross-products for `b_T`, `B₀ = diag(1e12, 1e8)` used in proposal and ratio; `σ²` independence MH | N02 §6.1 | uncentred `b_T` is biased; KFS-printed 3-block φ ratio fails Geweke |
| D-7 | NC sign: measure-preserving fold `(σ, h̃) ↦ (−σ, −h̃)` when `σ < 0` | N02 §7 | stochvol's `|σ|` without the flip is a documented oracle deviation |
| D-8 | Student-t: **standardized** `τ_t ~ IG(ν/2, (ν−2)/2)`; ν on the static grid `{3..40}` with prior weights as data; collapsed exact discrete Gibbs (`ν | e,h,o` with τ integrated, then `τ | ν`). `t_scale="unit"` is a replication option. A continuous ν prior is deferred (open question Q8); SV-t parity uses a fixed ν | N03 §0, §4.3a, §4.4, §7.4 | N06 §2.1 writes `ω ~ Gamma(ν/2, rate ν/2)`; the validation simulators use the sampler's parametrisation |
| D-9 | Outliers: integer grid `o ∈ {1..K}`, `K = 20`, `w_1 = 1−p`, `w_k = p/(K−1)`; `p ~ Beta` from `(every_years, prior_years, freq)`; transitory order `o → p → ν → τ`, then `s → h → θ_SV` | N03 §0, §4.1–4.2, §5.1 | CCMM's `o`-after-`h` order is valid but depends on the approximation |
| D-10 | Structural prior **Mode I** (moment-matched, independent) for every vol block by default; **Mode II** (λ-scaled NIG) only for `Constant("iw")` × structural, where it equals the Phase 1 NIW posterior at `cross = κ_a = 1`; δ-coordinates; `cross` scales the s.d. | N05 §0, §4.2, §4.4 | — |
| D-11 | Reduced form (CCCM 2022): combined-weight conditional, precisions factored in parallel, sequential `fori_loop` over `j` for means; coefficient prior = moment-matched independent Minnesota; `(A, λ)` prior identical to the structural one, so the two forms differ only in the coefficient prior | N04 §5.3, §6.1–6.3 | deterministic prior variance `κ_d ψ_j` (NIW marginal), not raw `var_const` |
| D-12 | **Minnesota hyperparameters are fixed during MCMC.** `hierarchy="fixed"`: as given. `hierarchy="eb"` (default): plug-in mode from the **Phase 1 `AnalyticFit` on the conjugate twin at `cross = 1`**; Chan (2022) structural NIG ML only for `form="structural"`, `cross ≠ 1`; `form="reduced"`, `cross ≠ 1` uses the `cross = 1` twin with a recorded note (no error). `hierarchy="mh"` raises (Phase 3 global step) | N05 §9; §6.5 here | — |
| D-13 | SV defaults: `μ_i ~ N(log(ψ_ι/ι), 10)`, `(φ+1)/2 ~ Beta(20, 1.5)`, `σ² ~ 1·χ²₁`, stationary `h_0`; RW: `h_0 ~ N(log(ψ_ι/ι), 4)`, `σ² ~ 0.1·χ²₁`. stochvol/bayesianVARs priors by explicit arguments | N02 §8.2, §10.2; N05 §4.4; N01 §12.1 | open question Q3 |
| D-14 | Gaussian draws in precision form with exact Jacobi scaling and two triangular solves: `s = diag(K)^{−½}`, `L = chol(sKs)`, `θ = s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z)` | N05 §7.2; N07 §8.2 | One exception (`3d3f51b`; §2.10 WP25 amendment 6, §3.7.1): `Recursive.contemp_step` draws `a_i` by Householder QR of the whitened rows (`linalg.lstsq_draw`, `R` the upper Cholesky factor of `K_a`, so the same function of `z` to rounding) without forming `K_a`; it serves the non-fusing means (reduced form), and `precision_draw` is used again when `contemp_conditional` is overridden. The fused structural `(δ_i, a_i)` draw and CCCM keep D-14 |
| D-15 | Padding is exact by construction: masked coefficient slots get prior `N(0,1)`, zero data columns and are zeroed after the draw; padded rows have weight 0; padded SV states are decoupled `N(0,1)` dummies (right padding); **padded rows are masked before any `log`** | N07 §2.2–2.3, §9.4 item 3; N05 §7.1 | — |
| D-16 | PRNG: threefry2x32 typed keys; `fold_in` chain `seed → stream → data salt → chain → absolute sweep → block → equation → sub-draw`; **all counters uint32 with explicit casts**; no key in any carry; `rbg` forbidden; the backtest passes a per-origin fit seed in addition to the data salt | N07 §4 | — |
| D-17 | Every kernel = pure core with injected noise + thin key wrapper; every exact draw exposes its conditional parameters | N04 §6.4; N06 §5.1; N07 §13.1 | — |
| D-18 | Failures never raise inside JIT and never jitter: per-chain `(code, it)` in the carry; the host replays the failing chunk deterministically to `it − 1` and reports diagnostics recomputed from the failing block's inputs | N07 §8.4; N05 §7.3 | — |
| D-19 | `Minnesota(form="auto")` and `Constant(prior="auto")` are the new defaults, resolved in `BVAR.__post_init__` (conjugate/`"iw"` for the analytic case, structural/`"ig"` otherwise); resolved values are stored, so Phase 1 specs and hashes are unchanged; `grid()` harmonises `auto` forms across a grid so the SPEC §9.5 vol grid is a one-axis comparison | SPEC D3, §9.5 | — |
| D-20 | The LPD estimator of SPEC §7 is unchanged. Internal accelerations (leading-block slicing, optional factored `(A, ℓ)` variables) and the opt-in transitory Rao-Blackwellisation never change the contract fields | N03 §5.3 | — |
| D-21 | Lenza–Primiceri scaling, SOC/SUR dummies and `hierarchy="mh"` are rejected on the MCMC engine in Phase 2 with actionable errors (SOC/SUR is stretch WP25; **delivered by WP25, 2026-10-22: SOC/SUR now run on the MCMC engine**, §2.10 WP25 amendments) | N05 §7.7; SPEC §5.6 | — |
| D-22 | Validation code lives in `src/rai_bvar/validation/`; prior samplers, observable simulators and `log_joint` come from a **reference registry** written by work packages that write no kernels | N06 §2, §5.1; N03 §7.2 | — |
| D-23 | Only shapes and component choices are static; data, priors, `T_eff`, `thin`, hyperparameters are traced. `T_pad` is a deterministic function of `(T_eff, t_bucket)` only (never padded to a backtest maximum). Every choice that changes draws (`gram`, `const_path`, `t_bucket`) is in `MCMCSettings` and hence in the spec hash | N07 §7 | — |
| D-24 | Diagnostics use the in-repo NumPy `diagnostics.py`, extended with tail ESS, mean ESS, mean and quantile MCSE (Vehtari et al. 2021 §4.3–4.4); ArviZ optional | SPEC §10.6; N06 §10 | — |
| D-25 | **Packed Khatri–Rao weighted Gram** is the default Gram kernel: `FF[t,(r,s)] = F[t,r]·F[t,s]` (`r ≤ s`) built once per fit, `G_i = (w_i @ FF)[pos]`; the same for the CCCM precisions with `XX`. `packed_tiled` and `batched` are deterministic alternatives | performance P2; judges' re-measurement | measured 4.59 vs 8.16 ms per sweep |
| D-26 | Exact Constant fast paths behind `const_path`: **eigen** (Mode I × Constant, O(q²) per draw) and **fixed-factor** (Mode II × Constant, `θ = θ̂ + √λ·s⊙Lc⁻ᵀz`). The weighted Cholesky path is the reference oracle for both | N05 §7.4; performance P4 | 0.346 ms per sweep |
| D-27 | Burn-in is its own executable with a **traced sweep count** (exact `burn`, never rounded); sampling runs fixed-length chunks with a snapshot at the last kept draw; every `acc_*`/`rb_*` accumulator is zeroed at the burn-in → sampling boundary | judges' fix | kept draws and final state are chunk-invariant |
| D-28 | The **covariance structure owns `a_i` and its prior**; unified residual algebra `e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i` with `ỹ = Y` (structural means) or `Y − XΠ` (reduced). `Linear` fuses the `a_i` draw; non-fusing means (Phase 4 BART, reduced form) use the structure's contemporaneous step | extensibility A2–A3 | replaces a plan-inserted contemporaneous block |
| D-29 | Global steps run in a **pre** or **post** phase around the equation phase and get the program's `eq_map`; equations are batched by `vmap` (default) or `lax.map` (`Batching.MAP`, for non-vmappable kernels) | extensibility §3.7 | — |
| D-30 | One sanctioned prior–volatility interaction: the **coupling channel** (`Coupling(n_active, quad)` + `VolBlock.log_scale_eq`). Mode II scales every coefficient by `λ_i`; `Constant("iw_equivalent")` (Chan 2022 Prop. 1) scales only `a_i` | N05 §4.4, §5.2; N08 §0 | two labels, two priors |
| D-31 | `import rai_bvar` imports no JAX; `engines.mcmc` is imported lazily and enables x64 then | N07 §11 | subprocess test |
| D-32 | Backtest warm starts are opt-in in Phase 2 | SPEC §9.1 | open question Q6 |
| D-33 | `converged` is decided by a **primary** monitor set (log-likelihood, own first lag of `B`, SV parameters, `h_T`, `log λ`); every `a_ij`, `h_1`, `h_mid`, `p`, `ν` are secondary (reported, never gating) | performance §8.1; judges | avoids multiplicity false alarms |
| D-34 | **Opt-in path-integrated LPD** (`log_pred_density(…, lpd_method="integrated")`; the default `"conditioned"` is the SPEC §7 estimator, unchanged): draw `s` is conditioned on its parameters and its drawn latent scale path only, and `y_{T+1..T+h−1}` is integrated out analytically, `y_{T+h} | θ_s, λ_s ~ N(m_h, M_h M_hᵀ)` with `m_h = cond_mean_h − Σ_{k<h} Φ_{h−k} u_k`, `M_h = [Φ_{h−1} L_1 | … | Φ_0 L_h]`; subset factors by QR of `M_h[idx]ᵀ`. Needs the new accelerator variable `lag_coefs` (MCMC forecasts of linear or zero means); `h = 1` runs the conditioned code (bitwise); `rb_transitory` at `h > 1` and forecasts without `lag_coefs` raise. **Plumbing:** `score_forecast`, `backtest`, `estimate_cost`, `compare` and `ordering_sensitivity` take `lpd_method` (default `"conditioned"`: scores, layout and cache keys unchanged); `"integrated"` adds `"lpd": "integrated-v1"` to backtest cache keys and the conditioned estimator's columns to the scores (§11.6); the Clark–Ravazzolo `acceptance` config (4 × 20,000 draws) uses it in every stage. Review amendments (2026-10-23) in §7.3, §11.6 and §13 WP23 | WP23 review (2026-10-23); `docs/paper/NOTES.md` 8b; `benchmarks/reports/clark_ravazzolo_preregistration.md` | the conditioned estimator degenerates in the tails at `h > 1` (Constant UR h = 8 at 2008Q2: −93.6 with 1000 draws vs −19.1 path-integrated on 1000 draws) while its MCSE stays ≈ 1; the integrated one is exact given each draw's parameters and volatility path but still a Monte Carlo average over the draws (§7.3); contract fields and SPEC §7 unchanged; analytic, benchmark and ordering-pooled forecasts carry no `lag_coefs` (§7.3) |
| D-35 | **Composition-aware default MCMC budget and the SV budget warning** (resolves Q14, §12.2). `MCMCSettings.draws` defaults to `None`, which `BVAR.__post_init__` resolves and stores (the D-19 rule: no unresolved default survives construction): **4 chains × 10,000 kept draws** (`settings.SV_DEFAULT_DRAWS`) for every composition with stochastic volatility (`SV`, `SV(rw=True)`, `SV(outliers=True)`, `SV(tails="t")`, in structural or reduced form), 4 × 2,000 (`settings.DEFAULT_DRAWS`, the old value) for `Constant` volatility. An explicit `draws` (in an `MCMCSettings` or a mapping) is kept as given; settings that leave `draws` unset get the composition's default whatever else they set. `MCMCSettings.with_default_draws(sv=…)` is the one rule; `fit_mcmc` and `run_chains` apply it too (`sv` ⇔ `static.vol.persistent == "sv"`) for engine-level callers, and `MCMCSettings().store_thin` raises while `draws` is unset. **Warning** (§8.3): `resolve()` sets `warning_inputs["sv_budget"] = (chains, draws)` when an `SV` composition has fewer than 10,000 kept draws per chain (`draws`; `thin` and `chains` are not credited, and `chains < 2` keeps its own row), and `warning_catalogue` appends `monitor.sv_budget_message`, which names the Q14 evidence and the remedy `mcmc=MCMCSettings(draws=10_000)`; it reaches `fit.warnings`, a `UserWarning`, forecast attrs and the backtest aggregation like every §8.3 row. It never fires for `Constant` volatility, the analytic engine, the SV default or an explicit budget of 10,000 or more; engine-level fits (`fit_mcmc` called directly) carry no flag. **Hashes:** Phase 1 (analytic) specs and every `Constant` MCMC spec are unchanged (their `draws` resolves to 2000; pinned in `tests/unit/test_default_budget.py` with the hashes recorded at `ae5f8d6`). An SV model whose settings leave `draws` unset (the default, or e.g. `MCMCSettings(seed=1)`) now records `draws = 10000` in its spec, so its spec hash and every backtest cache key of it change: the one unavoidable change (scores cached at the old SV default are not served to the new default). `mcmc=MCMCSettings(draws=2000)` reproduces the old SV spec and hash exactly, so its caches are served; its name gains the `mcmc=` suffix because it is no longer the default, and conversely an SV model with an explicit `MCMCSettings(draws=10_000)` (BENCH-P2's budget) now equals the default and loses it (names label results and forecast attrs `model`; they enter no spec hash or cache key). Explicit budgets are unaffected: BENCH-P2 (4 × 10,000), the Clark–Ravazzolo `acceptance` (4 × 20,000), `full` (4 × 5,000) and `smoke` configs, the README `compare()` (`draws=20_000`); the CR `quick` config, which was `MCMCSettings()`, is now the explicit `MCMCSettings(draws=2000)`, so its models and cache keys stay as they were. The `converged` rule (§8.2), the SPEC §10.6 thresholds and `compare()`'s 95% rule are unchanged (Q14 options 3–5 not adopted) | owner decision 2026-10-24 (Q14 "as recommended"); evidence §12.2 Q14 (BENCH-P2, README quick start, Clark–Ravazzolo) | a default SV fit samples 5× as many kept draws as the old default (with the 1,000 burn-in sweeps, 11,000 against 3,000 sweeps, about 3.7×), and its diagnostics take about 7 s instead of 1 s at `n = 20` SV-O-t (§8.2); 4 × 10,000 raises the floor, it is not a convergence guarantee (BENCH-P2 still misses the gate on 4 of 101 primary quantities; the README example converges at 10 of 11 origins); `dataclasses.replace(model, vol=SV())` on a `Constant` MCMC model keeps its stored 2,000 draws (stored-resolved rule) and so warns |

### 0.2 Errata: every judge finding and its resolution

| # | Finding (who, on which proposal) | Resolution | § |
| :--- | :--- | :--- | :--- |
| E1 | D3 not applied: default `Minnesota.form="conjugate"` makes `BVAR(vol=SV())` raise (J1, J2 on correctness) | `form="auto"`, resolved to structural for MCMC compositions | 11.1 |
| E2 | Cost model of 20–25 ms per sweep contradicted by measurement (J1, J2 on correctness) | measured numbers (8.16 / 4.59 / 2.53 ms) replace it | 10.2 |
| E3 | `converged` gated on ~330 quantities including every `a_ij` (J1, J2 on correctness) | primary/secondary split | 8.1 |
| E4 | Burn-in rounded up to a chunk multiple contradicts chunk-invariance (J2 on all three) | burn executable with traced count | 5.1 |
| E5 | `form="reduced"`, `hierarchy="eb"`, `cross ≠ 1` raises although `eb` is the default (J2 on correctness) | plug-in from the `cross = 1` twin, recorded note | 6.5 |
| E6 | IG-σ² NC step is MH; combination with the exact-mode `w` ratio unspecified (J2 on correctness and performance) | single accept with `log α = log α_aux + w(h′) − w(h)` (N01 §10.3) | 3.7.4 |
| E7 | `it0 + j·thin + r` promotes to int64 under x64 (J2 on correctness) | all counters uint32 with explicit casts; host bound check | 5.6 |
| E8 | Accumulators (and performance's Welford moments) never reset after burn-in (J2) | zeroed at the boundary; reported values taken from the snapshot | 5.1 |
| E9 | Validation composes blocks' own `prior_sample`/`log_prior` (J1 on performance) | independent reference registry, separate owners | 9.1 |
| E10 | `Constant("iw_equivalent")` overloaded across forms (J1 on performance) | `"iw"` = Mode II (structural only); `"iw_equivalent"` = Chan `(a, λ)` prior (both forms) | 6.1 |
| E11 | bayesianVARs mapping inconsistent (R call vs table) (J1 on performance) | one parameter set, both sides derived through the mapping formulas | 9.9 |
| E12 | Factored `cond_A`/`cond_scale` changes the Phase 0 contract (J1 on performance) | contract fields unchanged; optional accelerator variables only | 7.3 |
| E13 | Type tags change every Phase 1 spec hash (J1, J2 on performance and extensibility) | no tags on Phase 1 components; Phase-2-only fields omitted at their defaults; golden hash test | 11.5 |
| E14 | "Runtime" knobs (`gram`, `pad_T`) are not bitwise-neutral (J2 on performance) | `gram` resolved by a fixed shape rule and stored in the spec; no pad-to-max; `Runtime` holds only knobs with a bitwise-invariance test | 5.6, 11.2 |
| E15 | `MCMCSettings` exported from a module whose package enables x64 at import (J2 on performance) | `settings.py` is pure Python; `import rai_bvar` imports no JAX | 1.1 |
| E16 | ν and `p` in the gating set (J2 on performance) | reported only (ν escape probability) | 8.1 |
| E17 | Large-offset mutant uncatchable under `exact=True` (J2 on performance and extensibility) | that mutant runs with `exact=False` | 9.7 |
| E18 | Fixed Lenza–Primiceri "supported" on MCMC without a mechanism; extensibility's `log_s2` incoherent (J1, J2) | LP rejected on MCMC in Phase 2; fast paths require time-constant weights | 11.1, 3.7.2 |
| E19 | Static `bool` inside a NamedTuple pytree (J1 on extensibility) | static flags live on kernel dataclasses (meta fields), never in leaves | 2.1 |
| E20 | User-settable `engine` field contradicts SPEC §4.5 (J1, J2 on extensibility) | no engine field; the twin is `Constant("iw")` × structural | 11.1 |
| E21 | Uncalibrated acceptance thresholds gate `converged` (J1, J2 on extensibility) | warnings only | 8.2–8.3 |
| E22 | Default grid not R2-fair under `auto` (J2 on extensibility and performance) | `grid()` harmonisation | 11.4 |
| E23 | `2 log|resid|` is `−inf` on padded rows → NaN (J2 on extensibility) | mask before the log | 3.7.4 |
| E24 | Mean state δ vs "γ stored"; `fitted_eq` without `a_i` (J2 on extensibility) | state stores `δ_i` and `a_i`; `fitted_eq(ms, a_i, …)` | 3.3 |
| E25 | EB re-implemented outside Phase 1 (J1 on extensibility) | Phase 1 `AnalyticFit` twin | 6.5 |
| E26 | `a`-slots first loses the padded-equals-unpadded oracle (J1 on extensibility) | layout `[δ | a]`, inactive slots trailing | 2.3 |
| E27 | Plan invariants forbid post-vol global steps (J1, J2 on correctness) | invariants restated around indicator locality | 4.8 |
| E28 | Thin Phase 3–5 story (J1 on correctness) | walk-through table | 3.9 |

---

## 1. Module layout and ownership

### 1.1 Import-time rules

1. `import rai_bvar` imports **no** `jax` module (asserted in a subprocess: `"jax" not in sys.modules`). Specs,
   settings and the analytic engine stay NumPy-only.
2. `BVAR.fit` imports the MCMC path lazily when `engine == "mcmc"`. `engines/mcmc/_jax.py`, imported first by
   every JAX-using engine module, calls `jax.config.update("jax_enable_x64", True)`; every public engine entry
   point also runs inside `with jax.enable_x64(True):` (N07 §3). `engines/mcmc/__init__.py`, `config.py` and
   `registry.py` import no JAX, so `api.py` can validate compositions at construction without loading JAX.
3. No module-level `jnp` array anywhere (Omori tables, ν grids and packed-index maps are NumPy and become
   constants at trace time). A module-level `jnp` array initialises the backend and makes
   `set_cpu_devices` impossible (N07 §11).
4. `rai_bvar.set_cpu_devices(k)` imports JAX inside the function, sets `jax_num_cpu_devices`, and raises an
   actionable error if a backend already exists.
5. `rai_bvar.components` imports only `components/specs.py`; kernel modules under
   `components/{structure,mean,coef,vol}/` import JAX and are loaded only by `engines/mcmc/registry.py`.

### 1.2 Tree

Owners are the work packages of §13. `new` marks new files; the others change. Every file has exactly one
owner; WP1 creates empty package `__init__.py` files and stub modules on day 1, and ownership of each stub
passes to the listed WP.

```text
src/rai_bvar/
├── __init__.py                          [WP2]  + SV, MCMCSettings, Runtime, FitPlan, set_cpu_devices,
│                                                 enable_compilation_cache, ordering_sensitivity, sv_priors
├── api.py                               [WP2]  auto resolution, engine rules, BVAR.mcmc, fit(seed, init, runtime, plan),
│                                                 harmonise_grid, spec/from_spec  [D-35] stores the resolved draws
├── settings.py                    new   [WP2]  MCMCSettings, Runtime, FitPlan (pure Python)
│                                        [D-35] draws=None → default_draws(sv): 10,000 with SV, else 2,000
├── _devices.py                    new   [WP1]  set_cpu_devices(k), enable_compilation_cache(path)
├── components/                          components.py becomes a package; import paths unchanged
│   ├── __init__.py                new   [WP2]  re-exports Linear, Constant, SV (no JAX)
│   ├── specs.py                   new   [WP2]  user-facing frozen dataclasses Linear, Constant, SV
│   ├── structure/__init__.py      new   [WP1]  empty
│   ├── structure/recursive.py     new   [WP10] Recursive covariance structure: a_i state, prior, contemp step, maps
│   ├── mean/__init__.py           new   [WP1]  empty
│   ├── mean/linear.py             new   [WP10] LinearStructural: fused (δ_i, a_i) draw, D3, Gram and constant paths
│   ├── mean/linear_reduced.py     new   [WP11] CCCMStep (pre global step) + ReducedEq (per-equation no-op mean)
│   ├── mean/zero.py               new   [WP10] ZeroMean (block-level stochvol parity, harness toys)
│   ├── coef/__init__.py           new   [WP1]  empty
│   ├── coef/fixed.py              new   [WP10] FixedDiagonal coefficient prior (Phase 3: global-local siblings)
│   └── vol/
│       ├── __init__.py            new   [WP1]  empty
│       ├── mixture.py             new   [WP12] OCSN-10 / KSC-7 NumPy tables, indicator logits, log g
│       ├── sv.py                  new   [WP12] SVCore (AR, RW): indicators, AWOL, exact MH, C-step, NC-step, fold
│       ├── constant.py            new   [WP13] ConstantCore: IG conditional with coupling
│       ├── transitory.py          new   [WP13] none / outlier / t / outlier+t transitory scales
│       └── robust.py              new   [WP13] VolBlockImpl = Transitory ∘ Persistent (atomic sample_eq)
├── priors/
│   ├── minnesota.py                     [WP2]  form="auto", contemp (κ_a); Phase 1 labels unchanged
│   ├── sv.py                      new   [WP2]  Normal, Beta, ScaledChi2, InvGamma, DiscreteUniform, OutlierPrior
│   ├── scales.py                  new   [WP6]  per-window ψ, own-lag means, frequency (reuses analytic helpers read-only)
│   ├── structural.py              new   [WP6]  Mode I / Mode II / iw_equivalent arrays (δ-coords), implied moments
│   ├── reduced.py                 new   [WP6]  moment-matched independent Minnesota for CCCM; explicit() parity hook
│   └── sv_arrays.py               new   [WP6]  ConstantPrior / SVPrior / TransitoryPrior arrays (data-centred, per window)
├── engines/
│   ├── analytic.py                      —      unchanged (read-only: _ar_residual_variances, AnalyticFit for EB;
│   │                                             Phase 2 closeout: + forecast attrs["ordering"], §2.10)
│   ├── structural_nig.py          new   [WP16] Chan (2022) per-equation NIG posterior, log ML, iid draws, EB objective
│   └── mcmc/
│       ├── __init__.py            new   [WP1]  lazy re-exports fit_mcmc, MCMCFit (imports no JAX)
│       ├── _jax.py                new   [WP1]  enables x64 once; imported first by every JAX-using module
│       ├── protocols.py           new   [WP1]  protocols, typed views, enums, overridable names
│       ├── state.py               new   [WP1]  pytree dataclasses for Phase 2 components, Empty, Status, helpers
│       ├── config.py              new   [WP1]  StaticConfig and sub-configs (frozen, hashable), bucket()
│       ├── keys.py                new   [WP1]  Stream / Block / Sub enums (append-only), key derivation
│       ├── registry.py            new   [WP1]  kind string → "module:Class" (lazy import strings)
│       ├── linalg.py              new   [WP4]  weighted Gram, precision draw, eigen and fixed-factor draws,
│       │                                         tridiagonal sampler, stable 2×2 regression
│       ├── random.py              new   [WP4]  inverse-CDF categorical, truncated normal, log-gamma family
│       ├── data.py                new   [WP5]  EngineData builders (NumPy and jnp), packed products, bucketing
│       ├── compose.py             new   [WP14] StaticConfig → Program (kernels, batching, global steps, caches)
│       ├── plan.py                new   [WP14] Plan / Step data, build_plan, invariants
│       ├── sweep.py               new   [WP14] make_sweep(program): the fixed skeleton
│       ├── runner.py              new   [WP14] prepare / init / burn / sample executables, host driver, devices
│       ├── storage.py             new   [WP14] storage plan, host store (preallocated or memmap), DataTree
│       ├── warmstart.py           new   [WP14] FinalState, remap onto a new window
│       ├── results.py             new   [WP14] MCMCFit, fit_mcmc (forecast/irf delegate to forecast.py)
│       ├── monitor.py             new   [WP15] monitored sets, converged rule, warning catalogue  [D-35] SV budget
│       ├── forecast.py            new   [WP17] forecast core, forecast / irf / draw_parameters from a fit
│       ├── resolve.py             new   [WP18] BVAR + VARData → (StaticConfig, EngineData, PriorArrays, hyper, notes)
│       │                                [D-35] warning_inputs["sv_budget"]
│       └── eb.py                  new   [WP18] EB via the Phase 1 twin and structural_nig
├── forecast/result.py                   [WP17] ESS-adjusted LPD MCSE, leading-block fast path, optional extra
│                                                 variables, opt-in transitory RB
├── diagnostics.py                       [WP15] + ess_tail, ess_mean, mcse_mean, mcse_quantile, *_many
├── evaluation/compare.py                [WP2]  grid() calls factory.harmonise_grid when present  [D-34] lpd_method
├── evaluation/backtest.py               [WP23] MCMC fit seeds, opt-in warm starts, spawn context, n_jobs warning
│                                        [D-34] lpd_method (cache tag "lpd"; scores.py layout)
├── models/base.py                       [WP2]  docstring of optional fit keywords (protocol unchanged)
├── structural/ordering.py         new   [WP22] ordering_sensitivity (N08 §B4)
└── validation/                    new
    ├── __init__.py                      [WP7]
    ├── geweke.py, sbc.py, uniformity.py, sequential.py, identities.py      [WP7]  statistics (model-agnostic)
    └── reference.py, simulate.py, log_joint.py, testfns.py, joint.py      [WP9]  independent references

tools/make_r_fixtures.R                  [WP8]  stochvol + bayesianVARs sections and probe fixtures
tools/make_sv_parity_data.py       new   [WP8]  NumPy generator of the parity datasets
benchmarks/pilot/                  new   [WP3]  clean-room pilot prototypes (timing only)
benchmarks/phase2_perf.py          new   [WP24] BENCH-P2 driver
benchmarks/clark_ravazzolo.py      new   [WP23] replication report (N08 §C)  [D-34] acceptance config, --lpd
                                         [D-35] quick config: explicit 4 × 2000
docs/phase2/09_pilot.md            new   [WP3]  pilot measurements and frozen defaults
pyproject.toml                           [WP1]  pytest markers only
.github/workflows/ci.yml                 [WP24] PR / nightly / weekly tiers
```

### 1.3 Responsibilities in one line each

- **protocols.py** is the only place the block interfaces are defined; every kernel type-checks against it.
- **config.py**: every static (shape- or branch-determining) choice as frozen dataclasses with value equality.
  The JIT cache key is exactly `(StaticConfig, argument shapes)`.
- **resolve.py** is the R9 boundary: ψ, own-lag means, EB values, offsets, SV level centres and the outlier
  prior are computed per window on the host and passed as arrays. Nothing inside JIT sees the calendar.
- **plan.py / sweep.py**: the Gibbs order as data (`Step` tuples) and its execution. `fit.plan.describe()`
  prints it; it is the single source of truth for the order.
- **reference.py** (validation): independent prior samplers, latent-variance maps and log densities per
  component kind, registered beside but never importing the kernels.

### 1.4 Test tree (owners)

```text
tests/mcmc/unit/          [WP1] test_keys, test_state, test_config, test_import_no_jax, test_devices
                          [WP4] test_linalg, test_random      [WP5] test_data
tests/priors/             [WP6] test_scales, test_structural_prior, test_reduced_prior, test_sv_arrays
tests/unit/               [WP2] test_specs_phase2, test_phase1_hash_golden, test_api_resolution, test_grid_harmonise,
                                test_spec_roundtrip_mcmc      [WP18] test_fit_mcmc_api   [D-34] test_lpd_integrated,
                                test_scores_lpd_method, test_clark_ravazzolo_script   [D-35] test_default_budget
tests/validation/         [WP7] test_uniformity, test_geweke_selftest, test_sbc_selftest, test_sequential, test_identities_toy
                          [WP9] test_reference_moments, test_simulate, test_log_joint, test_mixture_transcription
tests/mcmc/blocks/        [WP10] test_recursive, test_linear_structural, test_linear_fastpaths, test_zero_mean
                          [WP11] test_cccm, swa_oracle.py     [WP12] test_mixture, test_sv_core, test_sv_toy_quadrature
                          [WP13] test_constant, test_transitory, test_robust
tests/mcmc/runner/        [WP14] toys.py, test_plan, test_sweep, test_runner, test_chunking, test_recompile,
                                 test_failures, test_storage, test_warmstart, test_devices_bitwise
tests/mcmc/diagnostics/   [WP15] test_monitor, test_diagnostics_ext
tests/analytic/           [WP16] test_structural_nig     [WP18] test_eb_mcmc     [WP20] test_mcmc_twin, test_mode1_quadrature
tests/mcmc/forecast/      [WP17] test_forecast_core, test_irf_mcmc, test_lpd_mcmc   [D-34] test_lpd_integrated_mcmc
tests/contract/           [WP17] test_forecast_contract_mcmc.py (new file; the Phase 1 file is untouched)
tests/geweke/ tests/sbc/ tests/mutation/   [WP19]
tests/parity/             [WP21] test_stochvol, test_bayesianvars, SV_PARAMETRIZATION.md
                          [WP8] fixtures/*.nc, fixtures/data/*.csv, test_fixture_schema
tests/integration/        [WP22] test_ordering      [WP23] test_backtest_mcmc   [D-34] test_backtest_lpd_integrated
tests/bench/              [WP24] test_perf_phase2, bench_kernels, baseline.json
```

---

## 2. JAX state as pytrees

### 2.1 Conventions

- Every state, data, prior and cache class is registered with
  `jax.tree_util.register_dataclass(cls, data_fields=[...], meta_fields=[...])` (N07 §7.3). Meta fields are
  static and hashable.
- **Static flags never live in pytree leaves.** A choice such as "this prior block is λ-scaled" is a meta
  field of a frozen kernel dataclass or of `StaticConfig`, never a field of a NamedTuple that crosses
  `jit`/`vmap`/`scan` (fixes E19).
- **Fixed structure per StaticConfig.** No leaf is ever `None`; an absent part is the registered dataclass
  `Empty()` (no leaves).
- Per-equation quantities carry a leading equation axis `n`; kernels are written for **one** equation and
  batched by the program's `eq_map` (`vmap`, or `lax.map` when a kernel declares `Batching.MAP`). The runner
  adds a leading chain axis `C` (`vmap` or `shard_map`).
- The Phase 2 component states are defined in `engines/mcmc/state.py` so that the runner, the validation
  references and the kernels can be written in parallel against frozen shapes. Phase 3+ components define
  their own state classes in their own modules; the only shared struct is the tiny `Status`.
- Acceptance counters and Rao-Blackwellised sums belong to the component that produces them, as leaves named
  `acc_*` or `rb_*`. The runner zeroes every such leaf at the burn-in → sampling boundary (§5.1).

### 2.2 Dimensions

| Symbol | Meaning | Static |
| :--- | :--- | :--- |
| `C` | chains | yes |
| `n, p, n_d` | variables (= equations), lags, deterministic terms | yes |
| `k = n_d + n·p` | lagged regressors, `data/design.py` layout (deterministic, lag 1 block, …, lag p block) | yes |
| `q = k + n − 1` | padded structural coefficient vector `[δ_i | a_i]` | yes |
| `P = q(q+1)/2`, `P_x = k(k+1)/2` | packed Gram lengths (structural, CCCM) | yes |
| `T_eff`, `T_pad` | real rows (`int32` scalar, traced), compiled rows | `T_pad` yes |
| `J = 10` (`7`) | mixture components (OCSN, KSC) | yes |
| `K_o = 20` | outlier grid size | yes |
| `G = 38` | ν grid size (`{3..40}`) | yes |
| `H`, `S_c` | forecast horizon, forecast draws per chunk | yes |

Dtypes: float leaves `float64`; indicator and grid indices `int32` on device (`int8` on the host); error codes
and run lengths `int32`; iteration counters and `fold_in` data `uint32`; keys typed threefry.

### 2.3 Coefficient layout and padding (exact)

- Structural coefficient vector of equation `i`: `θ_i = [δ_i ∈ ℝ^k | a_i-slots ∈ ℝ^{n−1}]`, slot `k + j` holds
  `a_ij`, active iff `j < i`. `cmaskθ_i = [1_k | 1{j < i}]`, so the **inactive slots are trailing** and
  `q_i = k + i` coordinates are active (N07 §2.2; fixes E26).
- Inactive slots: prior mean 0, prior precision 1, data columns zeroed (rows and columns of the Gram and of
  `Fᵀ(w ⊙ y)` are masked), draw multiplied by the mask afterwards. The precision matrix is block-diagonal, so on
  the Cholesky path the active draw equals the unpadded draw computed with `z[:q_i]` (tested to 1e-12). Masked
  slots are excluded from log priors, from `n_active` in coupling, and from every log-determinant.
- Shared δ-coordinate design (N05 §4.1, §7.1): `F = [X | X M_c − Y_{:, :n−1}] ∈ ℝ^{T_pad × q}`, with
  `M_c = M[:, :n−1]` the reduced-form prior mean columns. Column `k + j` (`x_t'm_j − y_{j,t}`) does not depend on
  `i`, so one `F` (and one packed product matrix `FF`) serves all equations.
- Time padding (right): rows `t ≥ T_eff` of `Y`, `X`, `F`, `FF` are 0 and carry weight 0; SV states with
  index `> T_eff` are decoupled `N(0,1)` dummies (`Q = 1`, off-diagonal 0, `b = 0`), and the state at index
  `T_eff` uses the "last row" form (N07 §2.3). Every sum over `t` is masked with `m_t` (transitions with `m_t m_{t−1}`); IG shapes
  use `T_eff/2`; padded rows are masked **before** any `log` (N07 §9.4 item 3; fixes E23); forecasting reads
  `h[:, T_eff]` with a dynamic index, never `h[-1]`.
- Padding overhead at the reference size: 1.33× (Cholesky), 1.22× (Gram) (N07 §2.2). Two static equation
  buckets would cut this to 1.15×/1.10× at the best split (`s = 10`; the WP3 pilot's exact ratios, 09_pilot.md
  §10; the ≈1.08×/1.05× first quoted here was an underestimate); that is an optional lever kept only if
  measured faster (§10.4), and pilot decision 11 did not adopt it.

### 2.4 Static configuration (`engines/mcmc/config.py`, JAX-free)

```python
@dataclass(frozen=True)
class Shapes:
    n: int; lags: int; n_det: int; k: int; q: int; T_pad: int

@dataclass(frozen=True)
class StructStatic:
    kind: Literal["recursive"] = "recursive"
    contemp_scaling: Literal["none", "lambda"] = "none"     # λ-scaled a_i prior: Mode II and Constant("iw_equivalent")

@dataclass(frozen=True)
class MeanStatic:
    kind: Literal["linear_structural", "linear_reduced", "zero"]
    coef_scaling: Literal["none", "lambda"] = "none"        # λ-scaled δ_i prior: Mode II only
    const_path: Literal["weighted", "cholesky_const", "eigen", "fixed_factor"] = "weighted"   # resolved (§3.7.2)
    gram: Literal["packed", "packed_tiled", "batched"] = "packed"                              # resolved (§10.5)
    gram_tile: int = 0                                      # packed_tiled: columns of P per tile
    cccm_precompute: bool = True                            # reduced: all L_j before the j-loop (shape rule)

@dataclass(frozen=True)
class CoefPriorStatic:
    kind: Literal["fixed_diagonal"] = "fixed_diagonal"

@dataclass(frozen=True)
class VolStatic:
    persistent: Literal["constant", "sv"]
    rw: bool = False                                        # SV: φ ≡ 1 separate code path (N02 §8.1)
    h0: Literal["stationary", "fixed_b0"] = "stationary"    # AR SV initial state
    sigma2_prior: Literal["scaled_chi2", "inv_gamma"] = "scaled_chi2"
    transitory: Literal["none", "outlier", "t", "outlier_t"] = "none"
    outlier_K: int = 20
    nu_grid: tuple[int, ...] = tuple(range(3, 41))          # shape-defining; prior weights are data
    t_scale: Literal["standardized", "unit"] = "standardized"
    mixture: Literal["ocsn10", "ksc7"] = "ocsn10"
    exact: bool = True
    asis: bool = True
    zero_offset: bool = False                               # absolute c = 0 parity mode: floor-hit masking on
    coupled: bool = False                                   # Constant core consumes Coupling

@dataclass(frozen=True)
class StoreStatic:
    paths: bool = False                                     # h and log(o²τ) paths every path_thin stored draws
    in_loop_forecast: int = 0                               # FitPlan horizon H (0 = off), §5.8

@dataclass(frozen=True)
class StaticConfig:
    shapes: Shapes
    struct: StructStatic
    mean: MeanStatic
    coef: CoefPriorStatic
    vol: VolStatic
    store: StoreStatic
    batching: Literal["vmap", "map"] = "vmap"
    chunk: int = 250                                        # kept draws per sample executable (Runtime)
    float32_leaves: frozenset[str] = frozenset()

def bucket(T_eff: int, t_bucket: int) -> int:               # T_pad = t_bucket·⌈T_eff/t_bucket⌉
def validate_composition(struct: StructStatic, mean: MeanStatic, coef: CoefPriorStatic, vol: VolStatic) -> None
    # pure-Python legality rules of §3.8; api.py calls it at construction (no JAX import)
```

`StaticConfig` is the key of every `functools.lru_cache` factory (`compose`, `make_sweep`, `make_runner`,
`make_forecast`). Everything else is an argument. `engines/mcmc/__init__.py`, `config.py` and `registry.py`
import no JAX; JAX-using modules import `engines/mcmc/_jax.py` (WP1), which enables x64 once.

### 2.5 Data and per-fit cache

```python
@register_dataclass
class EngineData:              # one estimation window; shared by chains (in_axes=None)
    Y:      f64[T_pad, n]      # rows ≥ T_eff are 0
    X:      f64[T_pad, k]      # x_t = (d_t', y_{t−1}', …, y_{t−p}')' (data/design.py layout); rows ≥ T_eff are 0
    tmask:  f64[T_pad]         # m_t = 1{t < T_eff}
    T_eff:  i32[]
    hist:   f64[p, n]          # last p observations, most recent last (forecast start)
    t_last: f64[]              # 1-based sample position of the last observation (trend continuation)

@register_dataclass
class FitCache:                # computed once per fit by the jitted prepare(); per Geweke iteration in validation
    F:       f64[T_pad, q] | Empty     # structural δ-design [X | X M_c − Y_{:, :n−1}]
    FF:      f64[T_pad, P] | Empty     # packed products F[:, iu]·F[:, ju]   (gram="packed", const_path="weighted")
    XX:      f64[T_pad, P_x] | Empty   # packed products of X (reduced form, gram="packed")
    G_const: f64[q, q] | Empty         # Fᵀ D_m F       (Constant paths)
    FtY:     f64[q, n] | Empty         # Fᵀ D_m Y       (Constant paths)
    eq:      EigenCache | FixedFactorCache | Empty    # per-equation leaves (leading n)

@register_dataclass
class EigenCache:              # Mode I × Constant (§3.7.2)
    Q: f64[n, q, q]; g: f64[n, q]; d_half: f64[n, q]; u1: f64[n, q]; u2: f64[n, q]
@register_dataclass
class FixedFactorCache:        # Mode II × Constant (§3.7.2)
    Lc: f64[n, q, q]; s: f64[n, q]; theta_hat: f64[n, q]
```

The packed-index maps `(iu, ju, pos)` are NumPy constants built at trace time from the static `q` (and `k`);
they are never module-level `jnp` arrays.

### 2.6 Prior arrays (built on the host in `resolve.py` from WP6 builders; traced arguments)

```python
@register_dataclass
class CoefPriorArrays:         # δ_i (structural) or π_j (reduced); λ-free precisions
    mean0: f64[n, k]; prec0: f64[n, k]
    group: i32[n, k]           # 0 deterministic, 1 own lag, 2 cross lag (Phase 3 group hook)
@register_dataclass
class ContempPriorArrays:      # a_i; λ-free precisions; inactive slots: mean 0, prec 1
    mean0: f64[n, n−1]; prec0: f64[n, n−1]; cmask: f64[n, n−1]
@register_dataclass
class ConstantPrior:           # λ_i ~ IG(shape, scale)
    shape: f64[n]; scale: f64[n]
@register_dataclass
class SVPrior:
    b_mu: f64[n]; B_mu: f64[n]           # μ ~ N(b_μ, B_μ)            (B is a variance)
    a_phi: f64[n]; b_phi: f64[n]         # (φ+1)/2 ~ Beta(a_φ, b_φ)
    B_sigma: f64[n]                      # σ² ~ B_σ·χ²₁               (sigma2_prior="scaled_chi2")
    c0: f64[n]; C0: f64[n]               # σ² ~ IG(c0, C0)            (sigma2_prior="inv_gamma")
    B0: f64[n]                           # h0="fixed_b0": h_0 ~ N(μ, B0 σ²)
    m0: f64[n]; V0: f64[n]               # RW: h_0 ~ N(m0, V0)
    log_c: f64[n]                        # log offset (−inf only in zero_offset parity mode)
@register_dataclass
class TransitoryPrior:
    a_p: f64[n]; b_p: f64[n]             # p ~ Beta(a_p, b_p)
    log_nu_w: f64[n, G]                  # log prior weights on the ν grid
@register_dataclass
class EqPrior:                 # every leaf has leading axis n (vmapped)
    coef: CoefPriorArrays
    contemp: ContempPriorArrays
    vol_core: ConstantPrior | SVPrior
    transitory: TransitoryPrior | Empty
@register_dataclass
class SharedPrior:             # broadcast (in_axes=None)
    Mc: f64[k, n−1]            # D3 centring columns; zeros for reduced and zero means
@register_dataclass
class PriorArrays:
    eq: EqPrior; shared: SharedPrior
```

Which precisions are λ-scaled is static (`StructStatic.contemp_scaling`, `MeanStatic.coef_scaling`); the arrays
always hold the λ-free precision `P̃₀`, and kernels divide the scaled blocks by `λ_i = exp(log_scale)`.

### 2.7 Chain state

```python
@register_dataclass
class RecursiveEq:   a: f64[n, n−1]                        # row i: a_ij for j < i, exactly 0 for j ≥ i
@register_dataclass
class LinearEq:      delta: f64[n, k]                      # γ_i = δ_i + M_c a_i is derived, never stored
@register_dataclass
class ConstantEq:    log_lam: f64[n]
@register_dataclass
class SVEq:                                                 # AR SV (stationary or fixed-B0 h_0)
    h: f64[n, T_pad+1]                                     # index 0 = h_0; index t+1 = data row t
    mu: f64[n]; phi: f64[n]; sigma: f64[n]                 # σ > 0 stored (not σ²)
    acc_h: f64[n]; acc_cgp: f64[n]; acc_cs2: f64[n]; acc_ncphi: f64[n]; acc_ncms: f64[n]; acc_fold: f64[n]
    acc_floor: f64[n]                                      # zero_offset parity mode: masked exact zeros
    acc_rej_run: i32[n]; acc_rej_run_max: i32[n]           # exact-h consecutive rejections
    rb_h_sum: f64[n, T_pad+1]; rb_h_sq: f64[n, T_pad+1]    # latent moments without storing paths
@register_dataclass
class SVRWEq:                                               # RW SV: h_0 = h[:, 0]; no μ, φ
    h: f64[n, T_pad+1]; sigma: f64[n]
    acc_h, acc_cs2, acc_ncms, acc_fold, acc_floor: f64[n]; acc_rej_run, acc_rej_run_max: i32[n]
    rb_h_sum, rb_h_sq: f64[n, T_pad+1]
@register_dataclass
class OutlierEq:     o_idx: i32[n, T_pad]; log_p: f64[n]; log1m_p: f64[n]; rb_out: f64[n, T_pad]
@register_dataclass
class TEq:           log_tau: f64[n, T_pad]; nu_idx: i32[n]; acc_nu_move: f64[n]
@register_dataclass
class OutlierTEq:    o_idx: i32[n, T_pad]; log_p: f64[n]; log1m_p: f64[n]; log_tau: f64[n, T_pad]
                     nu_idx: i32[n]; rb_out: f64[n, T_pad]; acc_nu_move: f64[n]
@register_dataclass
class VolEq:         core: ConstantEq | SVEq | SVRWEq; trans: OutlierEq | TEq | OutlierTEq | Empty
@register_dataclass
class EqState:       struct: RecursiveEq; mean: LinearEq | Empty; coef: Empty; vol: VolEq
@register_dataclass
class CCCMGlobal:    Pi: f64[k, n]
@register_dataclass
class GlobalState:   mean: CCCMGlobal | Empty; coef: Empty; vol: Empty; struct: Empty
@register_dataclass
class Status:        code: i32[]; it: u32[]; min_pivot: f64[]
@register_dataclass
class ChainState:    eq: EqState; glob: GlobalState; status: Status
```

- The mixture indicators `s` are **never** state (N02 §3.2): they are local to `VolBlock.sample_eq`, which is
  how "nothing between `s` and `h`" is enforced structurally (N01 §9).
- `A = I + strict-lower(a)` is owned by the structure in both forms (§3.2).
- Helpers in `state.py`: `Empty`, `zero_accumulators(st) -> st` (every `acc_*`/`rb_*` leaf to 0),
  `assert_float64(tree, allow: frozenset[str])`, `tree_where(pred, a, b)`, `encode_error(block, eq) -> i32`
  (`1 + 1000·block + eq`), `decode_error(code) -> (Block, int)`.
- Memory per chain at `n = 20, k = 81, T_pad = 320`, SV-O-t: about 0.3 MB.

### 2.8 Invariants (asserted on the host after every executable call, and in tests)

| Invariant | Check |
| :--- | :--- |
| float64 | every float leaf is float64, except a component's declared `float32_leaves` (empty in Phase 2; reserved for `bartz`, SPEC §8) |
| `A` unit lower triangular | `max |a ⊙ (1 − cmask)| == 0` exactly |
| finiteness | all leaves finite and `status.code == 0` |
| padding inert | `o_idx[:, T_eff:] == 0`, `log_tau[:, T_eff:] == 0` |
| SV ranges | `σ > 0`; `|φ| < 1` (AR) |
| keys | typed keys only; no key in any state or carry |

### 2.9 Views passed between blocks

```python
@register_dataclass
class EqContext:               # built by the structure inside the equation body (one equation)
    i: i32[]                   # position in the ordering (keys, masks)
    target: f64[T_pad]         # ỹ_i (minus cols·a_i when handed to a non-fusing mean)
    cols: f64[T_pad, n−1]      # −ỹ_{·, :n−1}: shared, unmasked contemporaneous regressors
    cmask: f64[n−1]            # 1{j < i}
    weights: f64[T_pad]        # m_t · exp(−log λ_eff,i,t)   (never 1/exp)
    log_scale: f64[]           # log λ_i if any prior block is λ-scaled, else 0
    tmask: f64[T_pad]

@register_dataclass
class GaussianDiag:  mean: f64[m]; prec: f64[m]; cmask: f64[m]      # a block prior, scaling already applied
@register_dataclass
class GaussianPrec:  K: f64[m, m]; b: f64[m]                        # exposed exact conditional N(K⁻¹b, K⁻¹)
@register_dataclass
class Coupling:      n_active: f64[]; quad: f64[]                    # zeros unless coupled (§4.7)
@register_dataclass
class StepInfo:      ok: bool_[]; min_pivot: f64[]
@register_dataclass
class EqOut:         e: f64[n, T_pad]; log_lam: f64[n, T_pad]       # equation-phase outputs for post steps
@register_dataclass
class SweepCtx:      data: EngineData; prior: PriorArrays; cache: FitCache; eq_out: EqOut | Empty
```

### 2.10 Contract additions made during the WP1 freeze (§13 rule 2)

*Sign-off (closed at the Phase 2 closeout, 2026-10-23).* This heading first read "owner sign-off pending". No
separate sign-off was recorded; the contract below shipped as frozen here, with the Stage 1, 2 and 4, WP25 and
Phase 2 closeout amendments that follow, and Phase 2 closed on it (SPEC §13 Phase 2 status).

Draw containers (`state.py`). §3, §5.3 and §7.2 use these types without defining them:

```text
StructDraw:   a: f64[n, n−1]
LinearDraw:   gamma: f64[n, k]                                        γ_i = δ_i + M_c a_i
CCCMDraw:     Pi: f64[k, n]
GlobDraw:     mean: CCCMDraw | Empty; coef: Empty; vol: Empty; struct: Empty    mirrors GlobalState
ConstantDraw: log_lam: f64[n]
SVDraw:       h_T, mu, phi, sigma: f64[n]                             h_T = h[T_eff]
SVRWDraw:     h_T, sigma: f64[n]
OutlierDraw:  log_p, log1m_p: f64[n]
TDraw:        nu: f64[n]
OutlierTDraw: log_p, log1m_p, nu: f64[n]
VolDraw:      core: ConstantDraw | SVDraw | SVRWDraw; trans: OutlierDraw | TDraw | OutlierTDraw | Empty
ParamDraw:    glob: GlobDraw; mean: LinearDraw | Empty; struct: StructDraw; vol: VolDraw
PathDraw:     h: f64[n, T_pad+1]; log_scale2: f64[n, T_pad]           2 log o + log τ
```

- `MeanBlock.extract_global(glob) -> GlobDraw` fills `.mean` and leaves the other parts `Empty()`.
- `state.py` helpers: `state_template`, `prior_template`, `draw_template(static)` (structure, shapes and
  dtypes implied by a `StaticConfig`); `check_state_invariants(st, static, T_eff) -> list[str]` (§2.8; the only
  non-finite value allowed is `status.min_pivot = +inf`); `leaf_path`; `Coupling.zeros`, `Coupling.__add__`;
  `StepInfo.success`, `StepInfo.merge`; `Status.initial` (`min_pivot = +inf`); `FitCache.empty`.
- `protocols.py`: `KernelBase` is a frozen dataclass with a keyword-only field
  `overrides: tuple[(name, callable), ...]` (names from `INTERNALS`; part of equality and hashing, so an
  overridden kernel never shares a cache entry with the production kernel), `impl(name)` and
  `with_overrides(mapping)`. `OVERRIDABLE = KERNEL_SLOTS ∪ INTERNALS`. `SupportsCollapsedQuadratic`;
  `MCMCNumericalError(..., chain=None)` pickles; `Batching` is an `IntEnum`.
- `registry.py` freezes the class paths: `structure.recursive:Recursive`, `mean.linear:LinearStructural`,
  `mean.linear_reduced:{ReducedEq, CCCMStep}` (kinds `linear_reduced`, `cccm`), `mean.zero:ZeroMean`,
  `coef.fixed:FixedDiagonal`, `vol.constant:ConstantCore`, `vol.sv:SVCore` (AR and RW),
  `vol.transitory:{NoTransitory, OutlierScale, TScale, OutlierTScale}`, `vol.robust:VolBlockImpl` (kind
  `vol_block`).
- `keys.py` (§5.6): `PRNG_IMPL = "threefry2x32"` is passed to `random.key`, so `jax_default_prng_impl` cannot
  change a stream, and keys of any other implementation raise `TypeError`. `Sub.Z` is an alias of `Sub.S`
  (value 0). `Tier` (`U=0, B=1, G=2, S=3, M=4, A=5, P=6`, append-only) for `validation_key`. `forecast_key`
  always folds `i` (`i = 0` for draw-level randomness). `step_key` accepts only a uint32 counter (int32 and
  int64 arrays raise; E7); other indices of any integer dtype are cast to uint32; all arguments are scalars
  (batches through `vmap`).
- `config.py`: `Shapes.of(n, lags, n_det, T_pad)`, `Shapes.P`, `Shapes.P_x`; `n ≤ MAX_EQUATIONS = 999`
  (error codes). `validate_composition` adds to §3.8 item 2: `coupled` holds exactly when some prior block is
  λ-scaled; Mode II only with `linear_structural`; any `const_path` other than `weighted` needs
  `linear_structural` and constant weights; `fixed_factor` also needs `not coef.varies`; `rw` and
  `zero_offset` need the SV core; `outlier_K ≥ 2` and the `ν > 2` rule apply only when outliers or t are used.
  Field validation of `MeanStatic` requires `gram_tile > 0` exactly when `gram == "packed_tiled"`.

---

**Stage 1 amendments (orchestrator, 2026-10-22):** `FitCache.FF` exists only for `gram="packed"` with
`const_path="weighted"` (its only consumer); `contemp_arrays` takes `scaling=` (keyword-only, required) because
Mode I and Mode II/iw_equivalent need different contemporaneous prior scales; §9.3's eigen-vs-Cholesky tolerance
is conditioning-aware (see §9.3 item 1).

**Stage 2 amendments (orchestrator, 2026-10-22):** (1) `Runtime.devices` defaults to 1; chain-count prefixes and
`vmap` vs `shard_map` agree to rounding, not bitwise, with the production kernels (§5.2, §5.6). (2) The forecast
key's `d` is the kept-draw index (§5.6, §7.1). (3) In-loop forecasts select exactly the §7.1 draws (§5.8).
(4) The §8.2 cost figure is restated from measurements. (5) `fit.summary()`'s implied reduced-form `Σ` under SV
is the plug-in at `λ_i = exp(b_μ)` (RW: `exp(m_0)`) and says so in a `vol_mean` field.

**Stage 4 amendments (orchestrator, 2026-10-22):** (1) WP21's SV-t PR criterion is met by the exact
importance-sampling SV-t oracle (4 observations, max |z| ≈ 1.8): stochvol 3.2.1's Student-t sampler does not
target the standardized-t SV posterior (`tests/parity/SV_PARAMETRIZATION.md` §3.2), so the `stochvol_v_t5` row
stays a strict xfail (`raises=OracleDeviation`) until a fixture from stochvol ≥ 3.2.5 (ideally 3.2.9) is built.
(2) bayesianVARs parity stays deferred (no gfortran); the exact system-wide Gibbs oracle replaces it. (SPEC §13 was amended to match on 2026-10-23: the oracle is the Phase 2 reduced-form criterion, and bayesianVARs parity, after the probes P1–P5, is a Phase 3 exit criterion; §9.9, §9.10, WP21 and the exit-criteria map carry the same note.) (3) WP23 review: backtest cache keys of `uses_fit_seed` models carry `"seeding": "fit_seed-v1"` (analytic keys stay the Phase 1 keys), so scores cached before the per-origin fit seeds are never served; `warm_start=True` requires strictly increasing origins (a warm start from a later origin would carry later data into an earlier fit, R9). (4) Known limitation, not changed here: the SPEC §7 RB log predictive density for h > 1 conditions on the simulated path to T + h − 1; for outcomes far in a predictive's tail its weights collapse onto one draw, the estimate is biased low by up to tens of nats and the delta-method MCSE saturates near 1 (Clark–Ravazzolo crisis origins at h ≥ 4). The replication report flags such cells; a path-integrated estimator is an open SPEC-level decision. (Superseded 2026-10-23: D-34 adds it as an opt-in, `lpd_method="integrated"`, with the SPEC §7 default unchanged.)

**WP25 amendments (2026-10-22; SOC/SUR dummy rows on the MCMC engine, N05 §7.7, N04 §3.1):**
(1) Contract: `MeanStatic.n_dummy: int = 0` (the row count `m`, shape-defining; `> 0` forbids the
`zero` mean); `CoefPriorArrays.dummy_w: f64[n] | Empty` (λ-free row weight: `ι/ψ_i` in structural
Mode I and `iw_equivalent`, 1 in Mode II, `1/ψ_j` reduced); `SharedPrior.dummies: DummyRows(Y:
f64[m, n], X: f64[m, k]) | Empty` (the Phase 1 rows, shared by every equation, so they cannot live in
the vmapped `EqPrior`); `FitCache.dummy: DummyCache(F, G, FtY, w) | Empty` (structural `F_d = [X_d |
X_d M_c − Y_d[:, :n−1]]`, reduced `F_d = X_d`; `G = F_dᵀF_d`; `w` copied from `dummy_w` because the
mean kernels see an `EqContext`, not `EqPrior`). All four are `Empty`/0 without dummies, so no dummy
code is traced and pre-WP25 programs are unchanged. `prior_template` emits them from `n_dummy`.
(2) Kernels: the rows enter the structural `(δ_i, a_i)` Gram and right-hand side with weight `w_i`
(Mode I, `iw_equivalent`) or `w_i/λ_i` (Mode II), and the CCCM `K_j`, `b_j` with `w_j` (never `A`,
`U`, `E`, `W`); Mode II's coupling adds `m` to `n_active` and `½w_iΣ_d(y_{d,i} − F_dθ_i)²` to `quad`;
`cccm_core`/`cccm_conditional_j` take a keyword-only `dummy=(G, FtY, w)`; the monitor's log prior
adds the rows' factor (up to its constant). The rows are never time steps: not in `EngineData`,
`tmask`/`T_eff`, the likelihood or the LPD. (3) Path rule (§3.7.2) unchanged by the rows: Mode I ×
Constant keeps `eigen` on the dense prior precision `P = diag(precθ) + w_d(cθ⊗cθ ⊙ F_dᵀF_d)`
(`linalg.eigen_prepare_dense`: `G̃ = L_P⁻¹GL_P⁻ᵀ` by the graded decomposition, the cache stores
`L_P⁻ᵀQ` with `d_half = 1`, so `eigen_draw` is unchanged; N05 §7.7), equal to `cholesky_const`
within the §9.3 path-vs-path envelope and 0.46 vs 1.53 ms/sweep at `n = 20, p = 4, T = 300`
(Apple M5 Max); Mode II keeps `fixed_factor`. (4) EB
(§6.5): the conjugate twin keeps `soc`, `sur`, `soc_scope`, so their EB values are plugged into the
rows; `hierarchy="mh"` stays `NotImplementedError`. (5) Validation: the reference conditions the base
prior on the rows (Bayes' rule, covariance form; Matheron draws; Mode II integrates θ for
`λ | d ~ IG(c + m/2, ·)`; `iw_equivalent`, where `λ | d` is not IG, draws it by exact rejection from
`IG(c, C)` with acceptance `(|S₀|/|S(λ)|)^{½}exp(−½(q(λ) − q∞))`, and its log prior is exact up to a
constant). Geweke rows 201–205 (§9.5), A1 extension (§9.8), mutants M17–M18 (§9.7).
(6) Numerics (queue item 6): `Recursive.contemp_core` draws `a_i` by Householder QR of the
whitened rows instead of the Jacobi-scaled Cholesky of the formed `K_a` (§3.7.1). The
`test_g_full[reduced_sv]` chain that reached `h ≈ 31–37` had `cond(K_a) = 1.3e16` (exact
eigenvalues 14.8 and 1.9e17) and a zero pivot; the QR draw matches the exact conditional there to
1.6e-8 sd and agrees with the Cholesky draw to ≤ 3e-14 (sd units and relative) on ordinary states.
The test prior's `h` tail is unchanged.
(7) Numerics (queue item 7, `f60d736` and `4fc0162`; recorded here at the Phase 2 closeout): the RW AWOL draw
factors `Q̃ = σ²Q` as a path-graph Laplacian plus a diagonal with the pivots carried as their excess over the
next edge, in three scans (§3.7.4; D-5 Notes; NOTES §E 17). It removes the σ → 0 crash of the Clark–Ravazzolo
ordering stage; every non-RW composition stays bitwise unchanged.

**Phase 2 closeout amendments (2026-10-23).** From the design audit of `888f620`: its P0 and P1 documentation
items, the CI compilation-cache fix and the two missing contract rows. Each change is recorded where it applies;
this list is the index.

1. SPEC v2.2: §13 Phase 2 status per exit criterion; the reduced-form parity criterion is the exact system-wide
   Gibbs oracle, and bayesianVARs parity moves to the Phase 3 exit; Q12 executed (§4.2 settled protocols, §8
   AWOL attribution and the reproducibility caveat); §2.2, §4.4, §4.5, §5.4, §5.6, §7, §9.1–§9.3, §9.5, §10.1,
   §10.3, §10.4, §10.6, §11, §12 and §14 brought in line with the code, and the features the audit found
   missing marked as not implemented. Stage 4 amendment (2), §9.9, §9.10, WP21 and the exit-criteria map now
   agree with SPEC.
2. Register: the D-5 and D-14 Notes columns.
3. §2.3 and §10.4 two-bucket ratios, §10.3 exact-correction line (pilot decisions 2 and 11).
4. §5.4: the SV cold start as implemented (pilot decision 4).
5. §9.2 `hold_frozen`; §9.3 item 4 and the WP14 tests (bitwise or to rounding, as tested); §9.4 tiers as run;
   §9.7 M2 and M7 catch tiers; §9.9 fixture versions and the bayesianVARs replacement.
6. §9.10: cadence as run, the CI compilation cache (diagnosis, decision, residual misses), the PR-tier budget,
   the weekly validation record and the concurrency group.
7. §11.1: the SOC/SUR rejection rule is superseded by WP25.
8. WP2 acceptance (the Phase 1 test edits of `ed502d4`), WP10 timing result, WP17 contract
   coverage, WP21 as closed, WP24 acceptance as amended (x86 report, PR-tier budget).
9. Code: `engines/analytic.py` (listed "unchanged" in §1.2) gains `attrs["ordering"]` in
   `AnalyticFit.forecast` (SPEC R8, §7; WP17 amendment); the docstrings of `rai_bvar.set_cpu_devices` (sharded
   draws equal the one-device draws only to rounding, §5.2) and `engines/mcmc/_blas.py` (the largest BENCH-P2
   factor is the 100 × 100 structural precision, not 81 × 81) are corrected.
10. Open question Q14 (§12.2): the default convergence policy for SV fits (pilot decision 6). Resolved by the owner on
    2026-10-24: D-35 (SV compositions default to 4 × 10,000 kept draws and warn below it).
11. Repository hygiene: `docs/.DS_Store` (committed in `3802295`) is removed from the repository, and `.DS_Store`
    is in `.gitignore`.

## 3. Block protocols (`engines/mcmc/protocols.py`)

Mapping from the SPEC §4.2 sketch (settled here, open question Q12): `MeanBlock.sample(target, weights, glob)` →
`MeanBlock.sample_eq` with an `EqContext` carrying `target` and `weights`; `VolBlock.sample(resid)` →
`VolBlock.sample_eq` (atomic); `VolBlock.variances` → `log_var_eq` (log scale, `which=` persistent/effective);
`VolBlock.simulate_ahead` → `draw_ahead_eq`; `GlobalBlock.sample(glob, eqs)` → `GlobalStep.sample` (pre or post
phase). Two roles the sketch left implicit get their own protocols: `CovarianceStructure` (owns `a_i`, SPEC §4.1
"covariance structures") and `CoefPrior` (coefficient prior with Phase 3 local/global scales).

### 3.1 Rules every block obeys

1. **Pure.** No Python branching on traced values, no host callbacks, no state outside the arguments. Static
   choices are attributes of the frozen kernel dataclass. `lax.cond`/`switch` on traced flags are forbidden
   inside vmapped bodies (they lower to `select` and run both branches; N07 §2.1).
2. **Noise injection.** `sample*(key, …)` = `*_core(noise, …)` ∘ `draw_noise(key, …)`. Unit tests drive the
   cores with fixed noise against NumPy dense references.
3. **Conditional parameters exposed.** Every exact Gibbs draw has a `*_conditional(...)` returning its
   distribution parameters (`GaussianPrec`, IG `(shape, scale)`, categorical logits); every MH step exposes its
   log acceptance ratio as a pure function. Density-ratio tests (§9.3) call these.
4. **Failures are values** (`StepInfo.ok`), never exceptions.
5. **Keys.** A block receives one key per equation and derives sub-keys only by `fold_in(key, Sub.X)` with the
   append-only `Sub` enum.
6. **Overridable internals.** A kernel calls its named internal functions through `self.impl(name)`, which
   returns an override from `make_sweep(..., _overrides=...)` or the default. The names are frozen in
   `protocols.OVERRIDABLE` (§4.9).

### 3.2 CovarianceStructure

```python
class CovarianceStructure(Protocol):
    static: StructStatic                                   # meta
    has_contemp: ClassVar[bool]                            # Recursive: True; Phase 5 Factor: False
    supports_reduced_mean: ClassVar[bool]                  # Recursive: True
    def init_eq(self, key: Key, i: i32[], prior_i: EqPrior) -> RecursiveEq_i            # a_i = 0
    def pre_residual(self, glob: GlobalState, data: EngineData,
                     f_reduced: f64[T_pad, n] | None) -> f64[T_pad, n]                  # ỹ = Y or Y − XΠ, × tmask
    def context(self, i: i32[], ytil: f64[T_pad, n], eq_i: EqState_i, vol: VolBlock,
                data: EngineData, gvol: PyTree) -> EqContext
    def contemp_moments(self, ctx: EqContext, prior_i: EqPrior) -> GaussianDiag         # a_i prior (÷λ if scaled)
    def contemp_conditional(self, ctx: EqContext, g_i: f64[T_pad], cm: GaussianDiag) -> GaussianPrec
    def contemp_step(self, key: Key, s_i: RecursiveEq_i, ctx: EqContext, g_i: f64[T_pad],
                     cm: GaussianDiag) -> tuple[RecursiveEq_i, Coupling, StepInfo]       # non-fusing means only
    def residual(self, a_i: f64[n−1], ctx: EqContext, g_i: f64[T_pad]) -> f64[T_pad]    # e_i, 0 on padded rows
    def log_jacobian(self, glob: GlobalState, T_eff: i32[]) -> f64[]                     # 0 for unit-triangular A
    def extract(self, eq_struct: RecursiveEq, glob: GlobalState) -> StructDraw           # a f64[n, n−1]
    def mean_map(self, sd: StructDraw, g: f64[n], f: f64[n]) -> f64[n]                   # A⁻¹g + f
    def chol(self, sd: StructDraw, log_lam: f64[n]) -> f64[n, n]                         # A⁻¹ diag(e^{½log λ})
    def reduced_form(self, sd: StructDraw, gamma: f64[n, k]) -> f64[k, n]                # B = (A⁻¹Γ)'
    def realign(self, s_host: PyTree, drop_front: int, append: int) -> PyTree            # host; identity here
```

The residual algebra is the same for every Phase 2 composition (fixes E24):
`e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i = target − (cols ⊙ cmask)·a_i − g_i`, with `ỹ = Y` and `g_i = Xγ_i`
(structural Linear, D3: `Xγ_i = Xδ_i + X M_c a_i`), or `ỹ = Y − XΠ` and `g_i = 0` (reduced form,
`e = (UAᵀ)`).

### 3.3 MeanBlock

```python
class MeanBlock(Protocol):
    static: MeanStatic
    fuses_contemp: ClassVar[bool]          # LinearStructural True; ReducedEq, ZeroMean (and Phase 4 BART) False
    batching: ClassVar[Batching]           # Batching.VMAP in Phase 2
    float32_leaves: ClassVar[frozenset[str]]
    def prepare(self, data: EngineData, prior: PriorArrays, base: FitCache) -> FitCache          # fills mean caches
    def fitted_reduced(self, glob: GlobalState, data: EngineData) -> f64[T_pad, n] | None     # reduced: XΠ
    def init_eq(self, key, a_i: f64[n−1], ctx: EqContext, pm: GaussianDiag, cm: GaussianDiag,
                cache_i: PyTree, cache: FitCache, shared: SharedPrior) -> tuple[MeanEq_i, f64[n−1]]
    def sample_eq(self, key, ms: MeanEq_i, a_i: f64[n−1], ctx: EqContext, pm: GaussianDiag,
                  cm: GaussianDiag | None, cache_i, cache, shared
                  ) -> tuple[MeanEq_i, f64[n−1], Coupling, StepInfo]       # a_i returned unchanged if not fusing
    def conditional_eq(self, ms, a_i, ctx, pm, cm, cache_i, cache, shared) -> GaussianPrec
    def fitted_eq(self, ms, a_i, ctx, cache, shared) -> f64[T_pad]          # g_i(x_t) (D3 term included)
    def coefficients_eq(self, ms, a_i, shared) -> f64[k]                   # what its CoefPrior governs (Linear: δ_i)
    def log_prior_eq(self, ms, a_i, pm, cm, shared) -> f64[]               # active coordinates only
    def extract_eq(self, ms, a_i, shared) -> MeanDraw_i                    # Linear: gamma f64[k]
    def extract_global(self, glob: GlobalState) -> GlobDraw                # reduced: Pi f64[k, n]
    def future_eq(self, key, md: MeanDraw_i, H: int) -> MeanFut_i          # Empty in Phase 2 (TVP γ paths later)
    def predict_eq(self, md, fut, x: f64[k], h: i32[], gfut) -> f64[]      # g_i(x_{T+h})
    def predict_reduced(self, gd: GlobDraw, x: f64[k], h: i32[], gfut) -> f64[n]   # Π'x, or zeros
```

### 3.4 CoefPrior

```python
class CoefPrior(Protocol):
    static: CoefPriorStatic
    varies: ClassVar[bool]                 # False for FixedDiagonal: Constant fast paths allowed
    def init_eq(self, key, i: i32[], prior_i: EqPrior) -> CoefEq_i                      # Empty in Phase 2
    def moments_eq(self, cs_i, i: i32[], prior_i: EqPrior, glob_coef) -> GaussianDiag    # λ-free (mean0, prec0)
    def sample_local_eq(self, key, cs_i, coef_i: f64[k], prior_i, glob_coef) -> CoefEq_i # identity in Phase 2
    def log_prior_eq(self, cs_i, coef_i: f64[k], prior_i, glob_coef) -> f64[]
    def global_steps(self) -> tuple[GlobalStep, ...]                                     # () in Phase 2
```

Phase 3 global-local priors (HS, NG, DL, R2D2, SSVS) are sibling kernels: local scales in `eq.coef`
(`sample_local_eq`, batched), global and group scales in `glob.coef` updated by a post-phase `GlobalStep`;
`moments_eq` returns the per-coordinate precision; `varies = True` switches Linear off the eigen path. No mean
kernel changes.

### 3.5 VolBlock and its two sub-protocols

```python
class VolBlock(Protocol):
    static: VolStatic
    def constant_weights(self) -> bool     # persistent == "constant" and transitory == "none" (and no global vol)
    def init_eq(self, key, resid0: f64[T_pad], tmask, T_eff: i32[], prior_i: EqPrior) -> VolEq_i
    def sample_eq(self, key, vs: VolEq_i, resid: f64[T_pad], tmask, T_eff, prior_i: EqPrior,
                  cpl: Coupling, gvol) -> tuple[VolEq_i, StepInfo]                    # ATOMIC, order fixed (§4.4)
    def log_var_eq(self, vs, gvol, which: Literal["effective", "persistent"] = "effective") -> f64[T_pad]
    def log_scale_eq(self, vs) -> f64[]                    # log λ_i for a coupled Constant core, else 0.0
    def loglik_eq(self, vs, resid: f64[T_pad], tmask, gvol) -> f64[]     # Σ_t m_t log N(e_t; 0, λ_eff,t), exact
    def obs_logdensity_eq(self, e: f64[], log_lam_pers: f64[], vd: VolDraw_i) -> f64[]
        # transitory-integrated log density −½ log λ + log f(e·λ^{−½}) (N03 §2.2): RB LPD (§7.3), Phase 4 particle filter
    def transition_eq(self, key, log_lam_prev: f64[], vd: VolDraw_i) -> f64[]   # one-step persistent transition
    def extract_eq(self, vs, T_eff) -> VolDraw_i           # h_T, (μ, φ, σ) | σ (RW) | log λ; log p, ν
    def draw_ahead_eq(self, key, vd: VolDraw_i, prior_i, H: int, gfut) -> tuple[f64[H], f64[H]]
        # (log λ_pers, log λ_eff) for T+1..T+H
    def log_prior_eq(self, vs, prior_i, tmask) -> f64[]

class PersistentCore(Protocol):            # ConstantCore | SVCore(AR) | SVCore(RW)
    def init(self, key, log_e2: f64[T_pad], tmask, T_eff, prior_i) -> CoreEq
    def sample(self, key, cs, log_e2c: f64[T_pad], tmask, T_eff, prior_i, cpl: Coupling) -> tuple[CoreEq, StepInfo]
    def log_var(self, cs) -> f64[T_pad]                     # persistent log variance per data row
    def log_scale(self, cs) -> f64[]
    def extract(self, cs, T_eff) -> CoreDraw
    def ahead(self, key, cd: CoreDraw, H: int) -> f64[H]
    def transition(self, key, log_lam_prev: f64[], cd: CoreDraw) -> f64[]
    def log_prior(self, cs, prior_i, tmask) -> f64[]

class TransitoryScale(Protocol):           # none | outlier | t | outlier_t
    def init(self, tmask, prior_i) -> TransEq
    def sample(self, key, ts, log_e2: f64[T_pad], log_var_pers: f64[T_pad], tmask, T_eff, prior_i) -> TransEq
    def log_scale2(self, ts) -> f64[T_pad]                  # 2 log o_t + log τ_t (0 for none and padded rows)
    def extract(self, ts) -> TransDraw                      # log_p, nu (as float)
    def ahead(self, key, td: TransDraw, H: int) -> f64[H]   # future 2 log o + log τ
    def log_density(self, log_z2: f64[], td: TransDraw) -> f64[]   # log f(z), N03 §2.2 table, z² in logs
    def log_prior(self, ts, prior_i, tmask) -> f64[]
```

`VolBlockImpl` (`robust.py`) composes them; it is the only VolBlock in Phase 2. `log_e2` is formed as
`where(m_t > 0, log(where(m_t > 0, e_t², 1)), 0)`: padded rows never see `log 0` (E23); an exact zero in a real
row gives `−inf`, which the offset absorbs (`y* = log c`) and which outlier/t logits handle (`exp(−inf) = 0`).
In `zero_offset` parity mode (`c = 0`) such a row is masked in the SV block and counted in `acc_floor`
(N02 §2.4).

### 3.6 GlobalStep

```python
EqMap = Callable[[Callable], Callable]     # program's batching of a per-equation function (vmap or lax.map)

class GlobalStep(Protocol):
    block: ClassVar[Block]                 # keys.Block id (append-only)
    phase: ClassVar[Literal["pre", "post"]]
    reads: ClassVar[frozenset[str]]; writes: ClassVar[frozenset[str]]   # state paths; writes ⊆ glob.* (and eq.coef.* post)
    marginal_of_indicators: ClassVar[bool] # must be True; tested by a density-ratio identity
    def init(self, key, st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache) -> ChainState
    def sample(self, key, st: ChainState, ctx: SweepCtx, eq_map: EqMap) -> tuple[ChainState, StepInfo]
    def log_prior(self, glob: GlobalState, prior: PriorArrays) -> f64[]
    def future(self, key, gd: GlobDraw, H: int) -> GlobFut            # Empty in Phase 2 (regimes, CommonSV later)
```

Global steps get the kernels at construction (by `compose`), so a reduced-form step can ask the `CoefPrior` for
every equation's moments through `eq_map`, and a Phase 5 regime step can evaluate per-equation, per-regime
log-likelihoods without knowing the batching.

### 3.7 Phase 2 implementations

#### 3.7.1 `Recursive` (`components/structure/recursive.py`)

- `context`: `cols = −ỹ[:, :n−1]`; `weights = tmask·exp(−vol.log_var_eq(vs_i, gvol))`; `log_scale =
  vol.log_scale_eq(vs_i)` if `contemp_scaling == "lambda"` or `coef_scaling == "lambda"`, else 0.
- `contemp_moments`: `mean = prior_i.contemp.mean0`; `prec = prior_i.contemp.prec0 · exp(−log_scale)` on active
  slots when `contemp_scaling == "lambda"`, unscaled otherwise, 1 on inactive slots.
- `contemp_conditional` (N04 §6.2 with its minus sign; N05 §5.4): with `Z = cols ⊙ cmask`,
  `K_a = diag(prec) + Zᵀ diag(w) Z`, `b_a = prec ⊙ mean + Zᵀ(w ⊙ (target − g_i))`. Since `cols = −ỹ_{<}`, this
  is the regression of `ỹ_i − g_i` on `−ỹ_{<i}`.
- `contemp_step`: the draw of `N(K_a⁻¹b_a, K_a⁻¹)` as `linalg.lstsq_draw` on the whitened rows
  `[diag(√prec); √w ⊙ Z]` (QR, `diag(R) > 0`: the same function of `z` as `precision_draw(K_a, b_a, z)`
  to rounding, without forming `ZᵀWZ`; WP25 queue item 6), `precision_draw(K_a, b_a, z)` when
  `contemp_conditional` is overridden; then `a ⊙ cmask`; `Coupling(n_active = Σ cmask,
  quad = ½ Σ cmask·prec0·(a − mean0)²)` when scaled, zeros otherwise.
- `chol`, `mean_map`, `reduced_form`: `solve_triangular(A, ·, lower=True, unit_diagonal=True)`; no inverses
  (N05 §8). `chol` is lower triangular with diagonal `exp(½ log λ) > 0`.

#### 3.7.2 `LinearStructural` (`components/mean/linear.py`; D-10, D-14, D-25, D-26; N05 §5.1, §7.1–7.4)

Prior on `θ_i = [δ_i | a_i]`: `pm` (δ; ÷λ when `coef_scaling == "lambda"`) and `cm` (a; from the structure).
`fuses_contemp = True`. Four static paths, chosen in `resolve.py`:

```text
weighted (any time-varying weights; the default whenever vol.constant_weights() is False)
    G  = weighted_gram(w, FF)                      packed: (w @ FF)[pos] — under vmap one GEMM (C·n, T_pad) × (T_pad, P)
    K  = (cmaskθ ⊗ cmaskθ) ⊙ G + diag(precθ)       precθ = [pm.prec | cm.prec]
    r  = precθ ⊙ meanθ + cmaskθ ⊙ Fᵀ(w ⊙ target)
    θ  = precision_draw(K, r, z) ⊙ cmaskθ          Jacobi-scaled Cholesky + two triangular solves (D-14)
cholesky_const (Constant core, weights m_t/λ_i; any scaling; the oracle for the two fast paths)
    K  = (cmaskθ ⊗ cmaskθ) ⊙ G_const/λ_i + diag(precθ);   r = precθ ⊙ meanθ + cmaskθ ⊙ FtY[:, i]/λ_i
eigen (Mode I × Constant, coefficient prior not varying): per fit, with d = precθ (λ-free in Mode I)
    G̃ = d^{−½} (cmaskθ⊗cmaskθ ⊙ G_const) d^{−½} = Q diag(g) Qᵀ   (eigh, g clipped at 0)
    u1 = Qᵀ(d^{½} ⊙ meanθ);  u2 = Qᵀ(d^{−½} ⊙ cmaskθ ⊙ FtY[:, i])
  per sweep (O(q²), no factorisation):  v = 1/(1 + g/λ_i)
    θ = d^{−½} ⊙ Q (v ⊙ (u1 + u2/λ_i) + √v ⊙ z) ⊙ cmaskθ        exact: K = d^{½}(I + G̃/λ)d^{½} (N05 §7.4)
fixed_factor (Mode II × Constant): per fit, K̃ = cmaskθ⊗cmaskθ ⊙ G_const + diag(prec̃θ), s = diag(K̃)^{−½},
    Lc = chol(sK̃s), θ̂ = s ⊙ Lc⁻ᵀLc⁻¹(s ⊙ (prec̃θ ⊙ meanθ + cmaskθ ⊙ FtY[:, i]))
  per sweep:  θ = (θ̂ + √λ_i · s ⊙ Lc⁻ᵀ z) ⊙ cmaskθ                  (N05 §5.1, §7.4: θ̂ is λ-free)
then  δ_i = θ[:k], a_i = θ[k:];  γ_i = δ_i + M_c a_i;  e_i = y_i − Fθ  (via Recursive.residual)
```

- Path rule (`const_path="auto"`): Constant core, no transitory, no time-varying global volatility, coefficient
  prior not varying → Mode I: `eigen`; Mode II: `fixed_factor`; `iw_equivalent` (only `a` scaled):
  `cholesky_const`. Anything else: `weighted`. `const_path="cholesky"` forces `cholesky_const`. Phase 3 time
  variation in the weights (Lenza–Primiceri scaling, CommonSV) makes `constant_weights()` False and therefore
  selects `weighted` (no `log_s2` bypass; E18).
- Coupling: Mode II returns `n_active = k + i`, `quad = ½ Σ_active prec̃θ(θ − meanθ)²`; `iw_equivalent`
  returns the same sums over the active `a`-slots only.
- `log_prior_eq` sums Gaussian log densities over active coordinates, with the λ scaling where static flags say
  so. `ok = all(isfinite(L)) ∧ all(diag(L) > 0)` (eigen: `g ≥ −1e-10·max g`).
- Optional SYRK Gram (`lax.linalg.symmetric_product`) only behind a static flag after a 1e-12 agreement test
  (N07 §8.2: no measurable gain so far).

#### 3.7.3 Reduced form (`components/mean/linear_reduced.py`; D-11; N04 §5.3, §6.1–6.4)

`CCCMStep` (pre phase, `Block.G_CCCM`; reads `glob.mean.Pi`, `eq.struct.a`, `eq.vol`; writes
`glob.mean.Pi`), pure core

```python
def cccm_core(Pi: f64[k, n], a: f64[n, n−1], inv_lam: f64[n, T_pad], X: f64[T_pad, k], Y: f64[T_pad, n],
              XX: f64[T_pad, P_x] | Empty, mean0: f64[n, k], prec0: f64[n, k], Z: f64[n, k],
              *, gram: str, precompute: bool) -> tuple[f64[k, n], StepInfo]
```

```text
inv_lam = tmask · exp(−log λ_eff)  (n, T_pad), from eq_map(vol.log_var_eq);  A = I + strict-lower(a)
W   = inv_lamᵀ @ (A ∘ A)                          (T_pad, n): W_tj = Σ_i a_ij²/λ_it
K_j = (Wᵀ @ XX)[:, posx] + diag(prec0_j),  L_j = chol(Jacobi-scaled K_j)    all j in parallel (independent of Π)
U = Y − XΠ;  E = U Aᵀ                             recomputed from scratch every sweep (no drift; N04 §7)
fori_loop j = 0..n−1 (dynamic j; lower-triangular A masks i < j):
    f_j = XΠ[:, j];  G_j = (E ⊙ inv_lamᵀ) A[:, j];  b_j = prec0_j ⊙ mean0_j + Xᵀ(G_j + W[:, j] ⊙ f_j)
    π_j = s ⊙ L_j⁻ᵀ(L_j⁻¹(s ⊙ b_j) + Z[j]);  d = X(π_j − Π[:, j]);  U[:, j] −= d;  E −= d ⊗ A[:, j];  Π[:, j] = π_j
```

`cccm_precompute` is a deterministic shape rule (`n·k²·8·C ≤ 512 MiB`), so it is reproducible; otherwise
`K_j, L_j` are built inside the loop. The coefficient moments come from `CoefPrior.moments_eq` through `eq_map`
(Phase 3 global-local priors plug in unchanged). `ReducedEq` is the per-equation mean (`fuses_contemp = False`,
state `Empty`, `fitted_eq = 0`, `fitted_reduced = XΠ`, `predict_reduced = Π'x`). The `a_j | Π, λ_j` step is
`Recursive.contemp_step` on `ỹ = U` (N04 §6.2, minus sign), and the volatility blocks receive `e_j = (UAᵀ)_j`.

#### 3.7.4 `SVCore` (`components/vol/sv.py`; D-1…D-7; N01 §8, §10; N02 §3–§9)

Per equation, inside `PersistentCore.sample`, given `log_e2c` (log ẽ² after the transitory step):

```text
m      = tmask  (zero_offset mode: also 0 where log_e2c = −inf, counted in acc_floor)
y*_t   = where(m_t, logaddexp(log_e2c_t, log_c), 0)                                   D-2
(c) s: r_t = y*_t − h_{t+1};  ℓ_tj = MIX_LOGW_j − (r_t − MIX_M_j)²·MIX_HALF_INV_V_j      N01 §8.2
       s_t = categorical_inverse_cdf(u_t, ℓ_t·)                                        D-4 (Sub.S)
       keep log g(r_t) = logsumexp_j ℓ_tj − ½ log 2π                                   reused for w(h_cur)
(a) h: NC AWOL (N02 §5.3) on T_pad+1 states (state t+1 ↔ data row t; states > T_eff decoupled N(0,1)):
       Q̃_00 = ω₀ + φ²  (stationary ω₀ = 1−φ² ⇒ 1; fixed-B0 ω₀ = 1/B₀),
       Q̃_{t+1,t+1} = σ²D_t + 1 + φ²  (t+1 < T_eff),   Q̃_{T_eff,T_eff} = σ²D_{T_eff−1} + 1,   Q̃ = 1 beyond T_eff,
       Q̃_{t+1,t} = −φ  (t+1 ≤ T_eff, else 0),   b̃_0 = 0,   b̃_{t+1} = σ D_t (y*_t − MIX_M[s_t] − μ),
       D_t = m_t / MIX_V[s_t]
       x̃′ = tridiag_precision_sample(diag, off, b̃, z) (Sub.H);  h′ = μ + σx̃′
       exact: log α_h = w(h′) − w(h),  w(h) = Σ_t m_t[−½(log 2π + h_{t+1} + exp(log_e2c_t − h_{t+1})) − log g(y*_t − h_{t+1})]
              accept iff log u (Sub.H_ACC, u ≥ tiny) < log α_h; on rejection keep h (s stays as drawn)      N01 §10.2
       acc_h += accept; acc_rej_run ← (acc_rej_run + 1)(1 − accept); acc_rej_run_max ← max
(b) centred C-step on h (N02 §6.1): (γ, φ) independence MH with centred cross-products and the stable 2×2
       algebra, B₀ = diag(1e12, 1e8) in proposal and ratio, |φ*| ≥ 1 ⇒ reject (Sub.C_GP, C_GP_ACC);
       σ² independence MH with IG(T_eff/2, C_T) proposal and log R = (σ² − σ²*)/(2B_σ) (ScaledChi2), or exact
       IG(c₀ + (T_eff+1)/2, C₀ + C_T) (InvGamma) (Sub.C_S2, C_S2_ACC). Sums masked with m_{t−1}.
(b*) h̃ = (h − μ)/σ
(b**) NC (N02 §7): φ | h̃ MH (Sub.NC_PHI, NC_PHI_ACC); then (μ′, σ′) | h̃, s, y*:
       ScaledChi2: exact 2×2 Gaussian regression, prior N((b_μ, 0), diag(B_μ, B_σ)), log α_aux = 0
       InvGamma:   same regression with a flat σ prior, log α_aux = log p_ext(σ′) − log p_ext(σ)
       exact mode: ONE accept/reject with log α = log α_aux + w(μ′ + σ′h̃) − w(μ + σh̃)      N01 §10.3 (E6)
       non-exact:  accept iff log u < log α_aux (always for ScaledChi2)
       w(μ + σh̃) = w(h) is reused from (a); on accept, if σ′ < 0: (σ′, h̃) ← (−σ′, −h̃) (D-7; h unchanged),
       acc_fold += 1; on reject keep (μ, σ, h̃) and the φ draw.
(b***) h = μ + σh̃
```

- **RW path** (`rw=True`, N02 §8.1): centred AWOL entries of N02 §8.1 (`Q_00 = 1/V₀ + 1/σ²`, …); C-step `σ²`
  independence MH with `IG((T_eff−1)/2, S/2)` proposal (requires `T_eff ≥ 2`, asserted on the host) or exact IG;
  NC regression on `(h_0, σ)` with prior `N((m₀, 0), diag(V₀, B_σ))`, same exact-mode rule and fold.
  Step (a) draws in the scaled form `Q̃ = σ²Q`, `h = σx̃`, and factors `Q̃` as a path-graph Laplacian (unit
  edges on the live transitions) plus `diag(σ²/V₀, σ²D)` (`awol_rw_laplacian` →
  `linalg.laplacian_precision_sample`): the pivots are carried as their excess over the next edge,
  `q_0 = δ_0`, `q_t = δ_t + q_{t−1}/(1 + q_{t−1})` on a unit edge (`q_t = δ_t` after a padded one), which
  never cancels; `p_t = q_t + w_t`. Assembling `Q̃`
  first rounds `σ²D` away once it is below the ulp of 2 (`σ ≲ 1e-8`, positive prior density under `B_σχ²₁`),
  and the level direction's pivot `≈ σ²(1/V₀ + ΣD)` is then rounding noise (status 2001, pivot 0; the
  Clark–Ravazzolo ordering-stage crash, WP25 queue item 7). Same function of `z` as
  `tridiag_precision_sample` on the assembled system to rounding; `aux_log_likelihood` (RW) uses the same
  factorisation; the AR path is unchanged.
- Cost of the exact mode: two extra `(T_pad, J)` `logsumexp`s per equation and sweep (at `h′` and at the NC
  proposal), because `log g(y* − h_cur)` comes from the indicator step (performance §3.7.4).
- `ahead`: `h_{T+j} = μ + φ(h_{T+j−1} − μ) + ση_j` (RW: no μ), from `h[T_eff]`; `transition` is one such step.
- Exposed for tests: `indicator_logits`, `awol_system` (`diag, off, b̃`), `awol_rw_laplacian` (`δ, w, b̃`),
  `w_weight`, `h_log_accept`,
  `c_step_log_ratios`, `nc_regression`, `nc_log_accept`, `nc_fold`.

#### 3.7.5 `ConstantCore` (`components/vol/constant.py`; N05 §5.2; N03 §4.5)

`λ_i | · ~ IG(shape_i + T_eff/2 + cpl.n_active/2, scale_i + ½ Σ_t m_t exp(log_e2c_t) + cpl.quad)`, drawn as
`log λ = log(scale′) − loggamma(shape′)` (N07 §9.1). Mode I/`ig`: `cpl = 0`; Mode II and `iw_equivalent`: the
coupling terms of §4.7. `log_var` broadcasts `log λ_i`; `ahead` is constant.

#### 3.7.6 Transitory scales (`components/vol/transitory.py`; D-8, D-9; N03 §4.1–4.4, §5.1)

With `log z²_t = log e²_t − log λ^{pers}_t` (padded rows: `o = 1`, `τ = 1` deterministically, excluded from
`N_1`, the ν log-likelihood and every sum):

| Transitory | Steps, in order | Formulas |
| :--- | :--- | :--- |
| `outlier` | `o | e,h,p` → `p | o` | logits `log w_k − log k − exp(log z² − log τ − 2 log k)/2`, `log w_1 = log1m_p`, `log w_k = log_p − log(K−1)` (N03 §4.1, the `1/k` factor included); `p ~ Beta(a_p + N_1, b_p + T_eff − N_1)` via log-gamma; `rb_out += P(o_t > 1 | ·)` |
| `t` | `ν | e,h` (τ integrated) → `τ | e,h,ν` | grid logits `log π_g + T_eff·A_g − ((ν_g+1)/2) Σ_t m_t log1p(exp(log r²_t − log c_g))`, `A_g = lnΓ((ν_g+1)/2) − lnΓ(ν_g/2) − ½ log(π c_g)`, `c(ν) = ν − 2` (standardized) or `ν` (unit) (N03 §4.3a); `log τ_t = log((c(ν) + r²_t)/2) − loggamma((ν+1)/2)` (N03 §4.4) |
| `outlier_t` | `o | e,h,ν,p` (τ int.) → `p | o` → `ν | e,h,o` (τ int.) → `τ | e,h,o,ν` | collapsed pmf `log w_k − log k − ((ν+1)/2) log1p(z²/(c(ν)k²))` (N03 §4.1); then as above with `r² = z²/o²` |

`log_scale2 = 2 log o + log τ`; the cleaned input to the core is `log_e2c = log e² − log_scale2` (N03 §4.6).
`ahead` draws `o_{T+j}` with one uniform (`o = 1` if `u ≥ p`, else `2 + ⌊(u/p)(K−1)⌋`) and
`log τ = log(c(ν)/2) − loggamma(ν/2)` (N03 §5.2). `log_density` implements N03 §2.2:
`outlier: log Σ_k w_k k⁻¹ φ(z/k)`; `t: log t_{c(ν),ν}(z)`; `outlier_t: log Σ_k w_k k⁻¹ t_{c(ν),ν}(z/k)`.

#### 3.7.7 `VolBlockImpl`, `ZeroMean`, `FixedDiagonal`

- `VolBlockImpl.sample_eq` = `transitory.sample` → `log_e2c` → `core.sample`; nothing else, in that order.
  `log_var_eq(which="effective") = core.log_var + transitory.log_scale2`; `obs_logdensity_eq =
  −½ log λ^{pers} + transitory.log_density(log e² − log λ^{pers})`.
- `ZeroMean`: `g ≡ 0`, no coefficients, `fuses_contemp = False`. With `n = 1` it runs block-level stochvol
  parity through the production runner (N07 §13.4).
- `FixedDiagonal`: `moments_eq` returns `(prior_i.coef.mean0, prior_i.coef.prec0)`; `sample_local_eq` is the
  identity; `varies = False`.

### 3.8 Composition rules (`compose.py`, with JAX-free legality in `config.validate_composition`)

`compose(static: StaticConfig, *, overrides: Mapping[str, Any] | None = None) -> Program` (lru-cached when
`overrides is None`):

1. Look up kernel classes by kind through `registry.py` (static dict of `"module:Class"` strings; later work
   packages implement classes at the listed paths and never edit the registry).
2. Legality (also run at `BVAR` construction): reduced mean ⇒ `structure.supports_reduced_mean`; any λ
   scaling ⇒ `vol.persistent == "constant"` (`VolStatic.coupled`); `coef_scaling == "lambda"` ⇒
   `contemp_scaling == "lambda"` (Mode II scales both); `eigen` ⇒ Mode I ∧ constant weights ∧
   `not coef.varies`; `fixed_factor` ⇒ Mode II ∧ constant weights; `rw` ignores `h0` (the RW path has its own
   `h_0 ~ N(m₀, V₀)`); the ν grid must exclude `ν ≤ 2` when `t_scale="standardized"`; `outlier_K ≥ 2`.
3. `batching = max(component.batching)`; `float32_leaves = union`.
4. Global steps from the mean (CCCM) and the coefficient prior, sorted by `(phase, block)`; duplicate ids
   raise.
5. `Program` is a frozen dataclass of frozen kernels, hashable, and carries `eq_map`.

### 3.9 Phase 3–5 plug-in walk-through (no edit to `sweep.py`, `runner.py` or `forecast.py`)

| Phase | Component | Where it plugs in |
| :--- | :--- | :--- |
| 3 | HS / NG / DL / R2D2 / SSVS with groups | `CoefPrior` sibling: local scales in `eq.coef` (`sample_local_eq`), global/group scales in a post `GlobalStep` reading `eq.mean` and `eq.coef`; `group` arrays already in `CoefPriorArrays`; `varies=True` disables the eigen path; a static `solver="bcm"` flag in `LinearStructural` for `q > T` (N07 §8.2); GIG sampler added to `linalg.py` |
| 3 | `hierarchy="mh"` for Minnesota | the same global-scale post step (RW-MH on log κ) |
| 3 | `CommonSV` | VolBlock whose core is `Constant(σ_i²)` and whose `glob.vol` holds the common `h`; post `GlobalStep` uses `q_t = Σ_i e_it²/σ_i²` from `ctx.eq_out` (N08 §A5.2); `log_var_eq(vs, gvol)` adds `h_t`; conjugate Σ scaling through the coupling channel |
| 3 | Conditional forecasts | the forecast core is pure in injected shocks and future variances (§7.5); Waggoner–Zha replaces the Gaussian step only |
| 3 | Estimated Lenza–Primiceri on MCMC | `glob.vol` parameters with a post MH step; `log_var_eq` adds `log s_t²`; weights stop being constant, so Linear takes the weighted path automatically |
| 4 | BART via `bartz` | `MeanBlock` with `fuses_contemp=False`, `batching=Batching.MAP` if it does not vmap, `float32_leaves` listing bartz leaves; `sample_eq` writes `ctx.target` and `error_scale = exp(−½ log w)` into the bartz state through a private adapter; `Recursive.contemp_step` draws `a_i | g_i` with the same prior as Linear (R2); per-draw forests stored only through the in-loop forecast hook (§5.8) |
| 4 | GIRF with common random numbers | `simulate_paths(params, hist, eps, vol_future)` with injectable shocks (§7.5) |
| 4 | PSIS-LFO, state-integrated likelihood | per-equation bootstrap particle filter from `transition_eq` + `obs_logdensity_eq` (equations independent given structure and mean) |
| 5 | TVP mean | `MeanBlock` with state `(γ_{i,0}, √v_i, γ̃_{i,1:T})` (non-centred), block-tridiagonal sampler in `linalg.py`, `future_eq` simulates the γ path, `realign` shifts it; variance shrinkage = a `CoefPrior` on `√v` |
| 5 | Threshold / Markov switching | `glob.mean` regime path; pre `GlobalStep` (FFBS or grid) using per-equation, per-regime log-likelihoods from `eq_map`; regime-specific Gram `(w ⊙ 1{s_t = r}) @ FF` keeps the packed trick; `future` simulates regimes |
| 5 | `FactorSV` | `CovarianceStructure` with `has_contemp=False`, `pre_residual = ỹ − fL'`, pre steps for factors, loadings and factor SV (reusing `SVCore` under `vmap` over factors); `chol` = Cholesky of `LΩL' + H` (failure raises) |
| 5 | Order-invariant SV (CKY 2024) | structure with dense `B₀` in `glob.struct`, Waggoner–Zha row step needing the optional mean capability `collapsed_quadratic_eq(ctx, pm) -> f64[n, n]` (declared in `protocols.py`, implemented by Linear in Phase 5); `log_jacobian = T_eff log|det B₀|` |

---

## 4. The Gibbs sweep

### 4.1 Plan and Step (`engines/mcmc/plan.py`)

```python
@dataclass(frozen=True)
class Step:
    name: str                              # "cccm", "equation", "contemp", "residual", "coef_local", "vol", or a global step
    block: Block                           # key-derivation id (append-only enum, §5.6)
    phase: Literal["pre", "equation", "post"]
    reads: frozenset[str]                  # state paths, e.g. {"eq.vol", "eq.struct.a"}
    writes: frozenset[str]                 # paths it may modify; checked by a tree diff (§9.3)

@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    def describe(self) -> str              # printed by fit.plan.describe(); embedded in the docs

def build_plan(program: Program) -> Plan
def check_invariants(plan: Plan, program: Program) -> None      # §4.8; raises on violation
```

`build_plan` is the only place the order is decided; `sweep.py` executes it.

### 4.2 Structural form (`Linear(prior=Minnesota(form="structural"))`, Mode I, Mode II or `iw_equivalent`)

```text
equation phase (one eq_map body per equation i; keys fold_in(fold_in(it_key, block), i))
S1  equation    writes {eq.mean.delta, eq.struct.a}   (δ_i, a_i) | λ_eff,i  joint Gaussian         Block.EQ_MEAN
                weights from the CURRENT vol state; Mode II / iw_equivalent: prior blocks ÷ λ_i (current)
S2  residual    writes {}                              e_i = y_i − Fθ_i from the NEW θ_i
S3  coef_local  writes {eq.coef}                       identity in Phase 2 (omitted from the plan)       Block.EQ_COEF_LOCAL
S4  vol         writes {eq.vol}                        VolBlock.sample_eq(e_i, cpl_i)                     Block.EQ_VOL
post phase: none in Phase 2
```

### 4.3 Reduced form (`Linear(prior=Minnesota(form="reduced"))`, CCCM 2022)

```text
pre phase
G1  cccm        writes {glob.mean.Pi}                  Π | A, λ_eff, sequential over j (N04 §6.1)       Block.G_CCCM
equation phase
S1  contemp     writes {eq.struct.a}                   a_j | Π, λ_eff,j: regress u_j on −u_{<j}          Block.EQ_CONTEMP
S2  residual    writes {}                              e_j = (UAᵀ)_j
S3  vol         writes {eq.vol}                        VolBlock.sample_eq(e_j, cpl_j)                     Block.EQ_VOL
```

All parameters that enter the structural residuals (`Π`, then `a`) are drawn before any equation's indicators
(N01 §9 global-level rule; N04 §6.3).

### 4.4 Inside every `VolBlock.sample_eq` (fixed, not configurable)

```text
transitory:  o → p → ν → τ (as configured)        exact conditionals given h; marginal of s; PCG trimming
log ẽ² = log e² − 2 log o − log τ;  y* = logaddexp(log ẽ², log c)
s  | y*, h                                        immediately before h (DNP 2015)
h  | y*, s, θ_SV                                  exact=True: MH with w
θ_SV centred | h
θ_SV non-centred | h̃, s, y*                       exact=True: one MH with log α_aux + Δw; fold
```

### 4.5 The sweep skeleton (`engines/mcmc/sweep.py`)

```python
def make_sweep(program: Program) -> Callable:
    def sweep(ckey: Key, it: u32[], st: ChainState, data: EngineData, prior: PriorArrays,
              cache: FitCache) -> ChainState:
        kit = random.fold_in(ckey, it)
        ctx = SweepCtx(data, prior, cache, Empty())
        for g in program.global_pre:                                   # static tuple, unrolled at trace time
            st, info = g.sample(random.fold_in(kit, g.block), st, ctx, program.eq_map)
            st = record_status(st, info, g.block, None, it)
        ytil = program.structure.pre_residual(st.glob, data, program.mean.fitted_reduced(st.glob, data))

        def body(i, eq_i, prior_i, cache_i):                           # ONE equation
            key = lambda b: random.fold_in(random.fold_in(kit, b), i.astype(jnp.uint32))
            ctx_i = program.structure.context(i, ytil, eq_i, program.vol, data, st.glob.vol)
            cm = program.structure.contemp_moments(ctx_i, prior_i)
            pm = program.coef.moments_eq(eq_i.coef, i, prior_i, st.glob.coef)
            if program.mean.fuses_contemp:                             # static
                ms, a, cpl, i1 = program.mean.sample_eq(key(Block.EQ_MEAN), eq_i.mean, eq_i.struct.a, ctx_i,
                                                        pm, cm, cache_i, cache, prior.shared)
                ss, i2 = replace(eq_i.struct, a=a), OK
                g = program.mean.fitted_eq(ms, a, ctx_i, cache, prior.shared)
            else:                                                      # reduced form, ZeroMean, Phase 4 BART
                a0 = eq_i.struct.a
                ctx_m = replace(ctx_i, target=ctx_i.target - (ctx_i.cols * ctx_i.cmask) @ a0)
                ms, _, cpl_m, i1 = program.mean.sample_eq(key(Block.EQ_MEAN), eq_i.mean, a0, ctx_m,
                                                          pm, None, cache_i, cache, prior.shared)
                g = program.mean.fitted_eq(ms, a0, ctx_m, cache, prior.shared)
                ss, cpl_c, i2 = program.structure.contemp_step(key(Block.EQ_CONTEMP), eq_i.struct, ctx_i, g, cm)
                cpl = cpl_m + cpl_c
            e = program.structure.residual(ss.a, ctx_i, g)
            cs = program.coef.sample_local_eq(key(Block.EQ_COEF_LOCAL), eq_i.coef,
                                              program.mean.coefficients_eq(ms, ss.a, prior.shared),
                                              prior_i, st.glob.coef)
            vs, i3 = program.vol.sample_eq(key(Block.EQ_VOL), eq_i.vol, e, data.tmask, data.T_eff,
                                           prior_i, cpl, st.glob.vol)
            out = EqOut(e, program.vol.log_var_eq(vs, st.glob.vol))
            return EqState(ss, ms, cs, vs), out, (i1, i2, i3)

        eq_new, eq_out, infos = program.eq_map(body)(jnp.arange(n, dtype=jnp.int32), st.eq, prior.eq, cache.eq)
        st = record_eq_status(replace(st, eq=eq_new), infos, it)       # first (code, it) kept per chain
        ctx = replace(ctx, eq_out=eq_out)
        for g in program.global_post:
            st, info = g.sample(random.fold_in(kit, g.block), st, ctx, program.eq_map)
            st = record_status(st, info, g.block, None, it)
        return st
    return sweep
```

`eq_map` is `jax.vmap(body, in_axes=(0, 0, 0, 0))` (shared arguments are closed over inside `sweep`, i.e. traced
arguments of the enclosing jitted function, never HLO constants) or `lambda f: lambda *a: lax.map(lambda t:
f(*t), a)` when any kernel declares `Batching.MAP`.

### 4.6 Why this order targets the posterior

1. **Mean, contemporaneous and CCCM steps** condition on the exact Gaussian likelihood given `λ_eff` and are
   marginal of `s` (DNP 2015 Alg. 2: `θ | Σ` drawn marginal of `s`).
2. **CCCM** is a Gibbs scan over the full conditionals `π_j | π_{−j}` (CCCM 2022 eqs. 13–15), equal to the SUR
   conditional (N04 §5.3, verified to 3.5e-13); `K_j` does not depend on `π_{−j}`, so factoring all `K_j` first is
   exact.
3. **Transitory steps** are exact conditionals given `h`, marginal of `s`; the τ-collapsed steps are partially
   collapsed Gibbs blocks whose trimmed draws are never conditioned on before being redrawn (N03 §5.1;
   van Dyk & Park 2008; numerically verified there).
4. **`s` then `h`** with nothing in between (DNP Alg. 2); with `exact=True`, Alg. 3 (exact invariant law for any
   `c ≥ 0`, N01 §10.2). The centred θ step does not use `s`; the NC step uses `s` and `y*`, which have not changed
   since `s` was drawn (N02 §3.2). The ASIS NC correction is N01 §10.3 (derived there, toy-checked; gated by
   Geweke here).
5. **Equations are conditionally independent** given the global state in structural form (`|A| = 1`, priors
   independent across equations given hyperparameters; SPEC §4.1, N05 §0 item 1), so the batched equation phase
   is an exact block draw. In the reduced form the rows `a_j` are independent given `Π, Λ` (N04 §6.2).
6. **Pre and post global steps** are exact conditionals of the indicator-free target. Because `s` is local and is
   redrawn before `h` inside every equation's vol step, a global step may sit in either phase without violating
   the DNP rule (extensibility §3.7). Phase 2 has only the pre-phase CCCM step.
7. **Coupled λ step**: with `Coupling`, the Constant conditional is the exact NIG conditional (N05 §5.2,
   verified in N05 §13.2 check 3).

### 4.7 Coupling: Mode II and `iw_equivalent` made explicit (D-30)

Prior: `θ_i^{scaled} | λ_i ~ N(μ₀, λ_i P̃₀⁻¹)` on the scaled coordinates.

- Mode II (`Constant("iw")` × structural): all active coordinates (`δ_i` and `a_i`) scaled. The equation step uses
  `K = (P̃₀ + F_iᵀD_mF_i)/λ_i`; it returns `Coupling(n_active = q_i = k + i, quad = ½(θ_i − μ₀)ᵀP̃₀(θ_i − μ₀))` over
  active coordinates. The vol step draws `λ_i | θ_i ~ IG(c_i + (T_eff + q_i)/2, C_i + ½e_iᵀD_me_i + quad)` (N05 §5.2).
- `iw_equivalent` (`Constant("iw_equivalent")`, either form): only `a_i` scaled (`n_active = i`, quad over
  active `a`-slots). With `κ_a = 1` the implied `Σ ~ IW(diag ψ, n + 2)` exactly (Chan 2022 Prop. 1; N08 §0).
- Padded coordinates never enter `n_active` or `quad`. Coupling is legal only with a Constant persistent core
  (static check). It is the one sanctioned prior–volatility interaction; Phase 3 CommonSV with conjugate Σ
  scaling uses the same channel.

### 4.8 Plan invariants (checked by `check_invariants` for every composition; fixes E27)

1. Every step that writes an input of the structural residuals (`glob.mean.*`, `eq.mean.*`, `eq.struct.*`) runs
   in the pre phase or, within the equation body, before that equation's vol step.
2. Each equation has exactly one vol step, and the indicators are not part of any state, so no step can sit
   between `s` and `h`.
3. `eq.vol.*` is written only by the vol step. Pre steps write only `glob.*`; post steps write only `glob.*` and
   `eq.coef.*`.
4. Each step's actual modifications are a subset of its declared `writes` (tree diff at a random state).
5. Coupling non-zero ⇒ `VolStatic.coupled` ⇒ Constant core.
6. Every global step has `marginal_of_indicators = True` and passes its density-ratio identity against the
   non-augmented joint (§9.3).

### 4.9 Static configuration, compilation caching, test seam

- `compose`, `make_sweep`, `make_runner` and `make_forecast` are `lru_cache`d on `StaticConfig` (value equality).
  A new window, new hyperparameter values or new data in the same `T_pad` bucket reuse the executables (N07 §7).
- Component choices are Python branches at trace time. Data and priors are always arguments, never closed over
  (closing over arrays makes them HLO constants; N07 §7 rule 2).
- Opt-in persistent compilation cache: `rb.enable_compilation_cache(path, min_compile_secs=1.0)` (trusted
  directory only, N07 §7.6). CI sets `min_compile_secs=0` so small Geweke programs are cached.
- Recompile guard: two fits on different data of one bucket emit zero new `Compiling jit(sample_chunk…)` records
  under `jax_log_compiles` (filter on the runner's function names; N07 §7.7).
- Test seam: `compose(static, overrides=...)` and `make_sweep(..., _overrides=...)` replace named internals or
  whole kernel slots. Keys must be in `protocols.OVERRIDABLE`:
  `{"structure", "mean", "coef", "vol", "global_pre", "structural_gamma", "linear_conditional", "cccm_core",
  "contemp_conditional", "indicator_logits", "mixture_table", "h_accept", "c_step", "nc_step", "nc_accept",
  "nc_fold", "constant_conditional", "outlier_logits", "p_conditional", "nu_logits", "tau_conditional"}`.
  Overrides bypass every cache; production code never passes them. The mutation suite (§9.7) uses only this seam.

---

## 5. Chain runner (`engines/mcmc/runner.py`)

### 5.1 Executables and host loop (fixes E4, E8)

```python
class RunnerFns(NamedTuple):
    prepare: Callable   # (data, prior) -> FitCache                                            jit
    init:    Callable   # (init_keys[C], data, prior, cache) -> ChainState[C]                  jit
    burn:    Callable   # (ckeys[C], state, data, prior, cache, it0: u32[], n: i32[]) -> state jit, donate state
    sample:  Callable   # (ckeys[C], state, data, prior, cache, it0: u32[], thin: u32[], stop: i32[])
                        #   -> (state, snap, out[C, chunk])                                    jit, donate state
    extract: Callable   # (state, data, prior, cache) -> (Monitor, ParamDraw, PathDraw | Empty)   one chain

def make_runner(static: StaticConfig, *, fit_axis: bool = False) -> RunnerFns   # lru-cached when no overrides
```

```python
def burn_one_chain(ckey, st, data, prior, cache, it0, n):
    return lax.fori_loop(0, n, lambda r, s: sweep(ckey, it0 + r.astype(jnp.uint32), s, data, prior, cache), st)

def sample_one_chain(ckey, st, data, prior, cache, it0, thin, stop):
    def keep(carry, j):                                          # j: uint32
        st, snap = carry
        base = it0 + j * thin
        st = lax.fori_loop(0, thin.astype(jnp.int32),
                           lambda r, s: sweep(ckey, base + r.astype(jnp.uint32), s, data, prior, cache), st)
        snap = tree_where(j == stop.astype(jnp.uint32), st, snap)   # exact state at the last kept draw
        return (st, snap), extract(st, data, prior, cache)
    (st, snap), out = lax.scan(keep, (st, st), jnp.arange(static.chunk, dtype=jnp.uint32))
    return st, snap, out
```

Host driver `run_chains(fns, static, data, prior, settings, runtime, keys, init_state=None) -> RunOutput`:

1. `cache = fns.prepare(data, prior)`; `state = fns.init(...)` or the remapped warm state (§5.5).
2. **Burn-in, exact count.** Call `burn` in pieces of `m = min(remaining, runtime.chunk·thin)` sweeps with
   `it0 = it`. Because keys depend on the absolute sweep index only, any piece size gives bitwise-identical
   results, and exactly `burn` sweeps run (no rounding). Before each call the start state is copied to the host
   (≈0.5 MB per chain) for replay (§5.7). After each call: status check and state invariants.
3. **Boundary.** `state = zero_accumulators(state)` (every `acc_*`, `rb_*` leaf).
4. **Sampling.** `ceil(draws/chunk)` calls of `sample` with `it0 = burn + c·chunk·thin` and
   `stop = min(chunk, draws − c·chunk) − 1`. Outputs `out[:, :stop+1]` go to the host store; surplus records of the
   last chunk are dropped. The **final state is `snap` of the last call** (state after exactly
   `burn + draws·thin` sweeps); its accumulators therefore cover exactly the sampling sweeps, and status checks
   on the last chunk use `snap.status` so surplus sweeps can never raise a spurious error.
5. Results: draws, latent moments from `snap.rb_*` divided by `draws·thin`, acceptance rates from `snap.acc_*`,
   final state for warm starts.

Kept draws and the final state are bitwise independent of `runtime.chunk` and of the burn piece size (tested
with chunk ∈ {50, 250}). `thin` and `n` are traced (`fori_loop` with a dynamic trip count lowers to `while`): no
recompile across `thin`, `burn` or `warm_burn`. No `jax.debug_*`, callbacks or `block_until_ready` inside the
jitted functions.

### 5.2 Chains and devices

- Default: `vmap` over chains on one device (`in_axes=(0, 0, None, None, None, None, None, None)`).
- `Runtime.devices` defaults to `1` (Stage 2 amendment; was `"auto"`). With `devices="auto"` or an integer
  `D ≥ 2` dividing `C`, the same body runs under `jax.shard_map` over a 1-D `("chain",)` mesh with
  `NamedSharding`-placed keys and state (N07 §6 snippet). With the production SV and CCCM kernels, XLA
  vectorises reductions differently for different per-device batch sizes, so sharded output equals
  single-device `vmap` only to rounding (`rtol 1e-9` on short runs, tested), not bitwise. Sharding is therefore
  opt-in, and the device count and chain method are recorded in attrs (§5.6). The backtest never passes a
  `Runtime`, so cached backtest scores never depend on the machine's device count.
- `rb.set_cpu_devices(k)` must run before JAX initialises; otherwise it raises with an actionable message and
  the runner falls back to one device with a warning.
- Optional `fit_axis=True` adds a leading fit axis `F` to data, priors, caches, keys and state (one more `vmap`).
  It is used only by `ordering_sensitivity(batch=True)` and benchmarks, where the fixed per-iteration overhead
  (≈0.18 ms at `n = 3`) dominates; batched fits equal unbatched fits to 1e-10 (tested), not bitwise.

### 5.3 Storage and memory

| Stream | Content | Frequency | Size at n=20, T=300, 4 × 10,000 |
| :--- | :--- | :--- | :--- |
| `monitor` | primary and secondary monitors (§8.1): log-likelihood, `B_{(1,i),i}`, `μ, φ, σ, h_T, h_1, h_mid` or `log λ`, every `a_ij`, `log p`, `ν` | every kept draw | ≈ 330 scalars → ≈ 106 MB |
| `params` | `ParamDraw`: `gamma (n, k)` (structural) or `Pi (k, n)` (reduced), `a (n, n−1)`, `VolDraw` (`h_T, μ, φ, σ` / `σ` / `log λ`; `log p`, `ν`) | every `store_thin`-th kept draw, `store_thin = ⌈draws/store_max⌉`, `store_max = 2500` | ≤ 4 × 2500 × 15 KB = 150 MB |
| `paths` (opt-in, `store="paths"`) | `h (n, T_eff+1)`, `log(o²τ) (n, T_eff)` | every `path_thin`-th stored draw | user-controlled |
| latent moments (always) | mean and sd of `h_{i,t}`; `P(o_{i,t} > 1)`; from `rb_*` accumulators | once | O(n·T) |

- `storage.plan_storage(static, settings, runtime) -> StoragePlan(bytes_by_stream)` runs before sampling. If the
  total exceeds `runtime.memory_budget_gb` (default 2) and `runtime.storage_dir` is unset, it raises and suggests
  `store_max`, `thin` or `storage_dir`. The host store is a dict of preallocated NumPy arrays `(C, D, …)`, or
  `np.memmap` files under `storage_dir`. float32 storage is not offered (R4; N07 §10).
- `fit.posterior` is an `xarray.DataTree`: groups `posterior` (`gamma` or `Pi`, `a` compacted to the strict lower
  triangle, `mu`, `phi`, `sigma`, `h_T`, `log_lam`, `p_outlier`, `nu`), `monitor`, `sample_stats` (`loglik`,
  acceptance per sub-step, fold fraction, longest rejection run, floor hits), `latent_moments`, `constant_data`
  (ψ, M, hyperparameters, offsets, prior arrays) and attrs (§5.6); dims `(chain, draw, …)`. `B = (A⁻¹Γ)'` is
  derived lazily (`fit.reduced_form_draws()`), never stored.

### 5.4 Initialisation (cold start; keys from `Stream.INIT`)

| Component | Initial value (chain-specific, dispersed but not extreme) |
| :--- | :--- |
| structural `(δ_i, a_i)` | one exact draw from the equation conditional with weights `m_t·ι/ψ_ι` (the Mode I `E λ_i`) |
| reduced `Π`, `a` | `Π`: one CCCM draw with `A = I`, `λ_j = ψ_ι/ι` (equations decouple; N04 §8); `a = 0` |
| Constant | `log λ_i = log(Σ_t m_t e_t²/T_eff)` from the initial residuals |
| SV (AR) | `μ_i = log(Σ_t m_t ẽ_t²/T_eff) + N(0, 0.5²)`; `h ≡ μ_i`; `φ_i` = prior mean; `σ_i = √B_σ·exp(N(0, 0.25²))` |
| SV (RW) | `h ≡ log(Σ_t m_t ẽ_t²/T_eff)`; `σ_i = √B_σ·exp(N(0, 0.25²))` |
| outliers / t | `o ≡ 1`, `p` = prior mean; `τ ≡ 1`, `ν` = grid median |

**Amendment (WP3 pilot decision 4, as implemented in `components/vol/sv.py` `SVCore.init`; recorded at the
Phase 2 closeout, 2026-10-23).** The SV rows above are not the final state. With `h ≡ μ_i` and a rejected first
exact `h` step, `C_T = 0`, after which σ = 0 is accepted and absorbing (3–6% of chains with OCSN-10, 34–59% with
KSC-7; 09_pilot.md §3.1). The cold start therefore (1) sets the level `ℓ_i = log(mean_t (ẽ_t² + c_i))` over
the real rows (`y*` of D-2; `b_μ`, or `m₀` for RW, when no row is observed); (2) sets `μ_i = ℓ_i + N(0, 0.5²)`
and `φ_i` = prior mean `2a_φ/(a_φ + b_φ) − 1` (AR; RW: no `μ`, `φ`, level `ℓ_i`), and `σ_i = σ̄·exp(N(0, 0.25²))`
with `σ̄ = √B_σ` for the `ScaledChi2` prior and `σ̄ = √(C₀/(c₀ + 1))`, the IG mode, for `InvGamma`;
(3) draws the indicators at `h ≡` level; and (4) draws `h` once from the AWOL conditional given those
indicators (the auxiliary-model draw, accepted without an MH step), so `C_T > 0` almost surely. Keys:
`Sub.S` indicators, `Sub.H` AWOL normals, `Sub.NC_MS` the two jitter normals.

Exact mode can mix slowly from crude starts when `c/λ` is large (N01 §7.2); with `c_rel = 1e-12` this regime does
not arise, and the longest-rejection-run warning (§8.3) reports it if it does.

### 5.5 Warm starts (backtests; SPEC §9.1; opt-in, D-32)

`fit.final_state -> FinalState(state: ChainState (host NumPy), periods: pd.PeriodIndex, static: StaticConfig,
seed_meta: dict)`. `warmstart.remap(prev: FinalState, new_data: EngineData, new_periods, static_new: StaticConfig,
key: Key) -> ChainState`:

- coefficients, `a`, `Π`, `(μ, φ, σ)`, `log λ`, `p`, `ν`: copied (cold start with a warning if `n`, `p` or the
  composition differ);
- `h`, `o`, `τ`: aligned by period label; periods dropped by a rolling window are removed; new trailing periods get
  `h` forward-simulated from the last aligned value with the copied `(μ, φ, σ)` (RW: random walk) and `o = 1`,
  `τ = 1`; `h_0` is the aligned value just before the new first period (or `μ`);
- re-padded to the new `T_pad` with masks; status and accumulators reset.

Warm starts change only the initialisation; burn-in still runs (`MCMCSettings.warm_burn`, default
`max(250, burn // 4)`). The PRNG still starts at `it = 0` with the origin's own fit seed and data salt, so streams
never repeat across origins.

### 5.6 PRNG discipline and reproducibility (`engines/mcmc/keys.py`; fixes E7, E14)

```text
root      = random.key(seed)                                    threefry2x32, typed
salt      = crc32(data.fingerprint()) as uint32                 (variables are part of the fingerprint, so orderings differ)
fit_root  = fold_in(fold_in(root, Stream.MCMC), salt)
chain_key = fold_in(fit_root, c)                                c = 0..C−1 (prefix-stable)
it_key    = fold_in(chain_key, it)                              it: uint32 absolute sweep index, burn-in included
step_key  = fold_in(it_key, Block.X)
eq_key    = fold_in(step_key, i)                                i: uint32 position in the ordering
sub_key   = fold_in(eq_key, Sub.Y)
init      : fold_in(fold_in(fold_in(root, Stream.INIT), salt), c) → Block → i → Sub
forecast  : fold_in(fold_in(key(fseed), Stream.FORECAST), c) → d (kept-draw index) → Ahead.{VOL, SHOCK, MEAN, GLOBAL} → i
validation: fold_in(key(seed), Stream.VALIDATION) → tier → composition id → chain or replication
```

- Enums (append-only; renumbering changes every stream). `Stream`: `MCMC=0, INIT=1, FORECAST=2, VALIDATION=3,
  IRF=4`. `Block`: `EQ_MEAN=0, EQ_CONTEMP=1, EQ_VOL=2, EQ_COEF_LOCAL=3, G_CCCM=16`; 17–31 reserved for Phase 3
  global steps (`G_COEF_GLOBAL=17, G_COMMON_SV=18, G_LP_SCALE=19`), 32–63 for Phases 4–5 (`G_REGIME=32,
  G_FACTORS=33, G_LOADINGS=34, G_FACTOR_SV=35, G_B0_ROWS=36`). `Sub` (vol): `S=0, H=1, H_ACC=2, C_GP=3,
  C_GP_ACC=4, C_S2=5, C_S2_ACC=6, NC_PHI=7, NC_PHI_ACC=8, NC_MS=9, NC_MS_ACC=10, O=11, P=12, NU=13, TAU=14`;
  (mean, contemp, CCCM) `Z=0` (CCCM folds `j` before `Sub.Z`).
- **All counters are uint32 with explicit casts** (`it0 + j·thin + r` computed from `uint32` operands; the
  `fori_loop` index is cast before use). The host asserts `burn + ⌈draws/chunk⌉·chunk·thin < 2³²`.
- One consumer per key; uniforms used under a log use `minval=tiny`; `rbg`/`unsafe_rbg` forbidden (`keys.PRNG_IMPL`
  is pinned at root creation and other implementations raise; §2.10); a test runs a
  full sweep under `jax.debug_key_reuse(True)` (N07 §4.1).
- **Seeds.** `fit(seed=…)` overrides `MCMCSettings.seed`. The backtest passes a per-origin fit seed to MCMC models
  (§11.6), in addition to the data salt (which keeps chain noise independent across windows even when a user fits
  several windows with one seed, as `ordering_sensitivity` does). Analytic models are unaffected.
- **Bitwise guarantees** for the same `(spec, data, seed, Runtime-independent settings)` on one device count:
  across `runtime.chunk`, burn piece size, fused vs unfused plan execution. **To rounding only** (Stage 2
  amendment): across chain-count prefixes and `vmap` vs `shard_map`. The keys are prefix-stable, so chain `c`
  consumes the same random numbers whatever `C` is, but XLA's batch-size-dependent vectorisation of the SV and
  CCCM reductions changes last bits; over long runs a last-bit difference can flip an accept/reject decision,
  after which the chains are only equal in distribution. `chains` is a spec field (`MCMCSettings`), so the cache
  key already covers it; `devices` defaults to 1 (§5.2). The prefix property is still tested bitwise with
  `chains="map"` and with toy kernels. **Not across** JAX versions, platforms, PRNG implementations, `jax_threefry_partitionable`, or any spec-level choice:
  `gram`, `const_path`, `t_bucket` (and hence `T_pad`) are spec fields or deterministic functions of spec and data
  shape, so they can never differ silently between two runs with the same cache key.
- **Recorded in every fit and forecast**: `jax`/`jaxlib` versions, PRNG implementation,
  `jax_threefry_partitionable`, platform, device count and chain method, seed, salt, `it` range, resolved
  `gram`/`const_path`/`T_pad`, mixture, offset rule and value, `exact`, ordering, spec hash, package version.

### 5.7 Failures and deterministic replay (D-18)

Blocks return `StepInfo.ok`; the sweep keeps the first `(code, it)` per chain with
`code = 1 + 1000·block + eq` (N07 §8.4) and the smallest pivot seen. After each executable call the host checks
the codes. On failure it replays the chunk from the saved host copy of its start state with `burn(…, it0,
n = it_fail − it0)` (deterministic), then evaluates the failing block's exposed conditional eagerly for that
equation and reports, in `MCMCNumericalError(block, equation, iteration, detail)`: the diagonal-ratio lower bound
`max/min diag(L)²` on `cond₂`, the weight range `min/max w_i`, the minimum tridiagonal pivot (SV), and the prior
precision range (N05 §7.3, N02 §10.3). No jitter is ever added and no chain is dropped.

### 5.8 In-loop forecast hook (for Phases 4–5)

`fit(data, plan=FitPlan(horizon, draws, seed))` sets `StoreStatic.in_loop_forecast = horizon`: the `sample`
executable calls the forecast core (§7.2) on exactly the kept states that §7.1 would select post hoc (Stage 2
amendment; was "every `⌈C·D/S⌉`-th", which can return fewer than `S` paths): the host precomputes the selection
as a per-chain index mask, passed as a traced argument and used as the shared `cond` predicate. Keys are the same
as post-hoc forecasting (`Stream.FORECAST → c → d`, `d` the kept-draw index), and the `ForecastResult` arrays are
emitted directly.
`fit.forecast(h, draws, seed)` returns the precomputed result when `(h, draws, seed)` match the plan. Linear
models do not need it (their draws are small); it exists so BART forests and TVP paths never have to be stored. It
is tested in Phase 2 on Linear × SV: in-loop and post-hoc forecasts agree to 1e-12. The backtest passes a plan only
to models that declare `requires_in_loop_forecast = True` (none in Phase 2).

---

## 6. Priors and hyperparameters

Everything data-dependent is computed per window on the host (R9) by the WP6 builders, called from
`resolve.py`, reusing Phase 1 helpers read-only: ψ from `engines.analytic._ar_residual_variances`
(`psi="ar"`: AR(p) with constant, OLS residual variance), own-lag means from
`Minnesota.own_lag_means(variables, data.prior_means())`. Notation: ψ[i] (0-based), `ι = i + 1`, `d = n + 2`,
`κ = lambda_`, `α = alpha`, `κ_c = cross` (scales the **standard deviation**, N05 §2), `κ_d = var_const`,
`κ_a = contemp` (new `Minnesota` field, default 1).

### 6.1 Structural form (D-10, D-30; N05 §4.1–4.4)

**Mode I** (default for `Constant("ig")` and every SV block). Fixed scale `σ̄_i² = ψ[i]/(d − n + ι − 2) = ψ[i]/ι`.

| Coefficient | Prior |
| :--- | :--- |
| `a_ij`, `j < i` | `N(0, κ_a ψ[i] / (ι ψ[j]))` |
| `δ_i`, own lag `l` | `N(m_{i,l}, κ²/(ι l^α))`; `m_{i,1}` = transform-aware own-lag mean (SPEC §5.3), `m_{i,l>1} = 0` |
| `δ_i`, lag `l` of variable `m ≠ i` | `N(0, κ² κ_c² ψ[i] / (ι l^α ψ[m]))` |
| `δ_i`, deterministic terms | `N(0, κ_d ψ[i]/ι)` |
| `γ_i` | `δ_i + M_c a_i`, i.e. `E[γ_i | a_i] = M ã_i` (SPEC §4.1, D3) |
| `λ_i` (`Constant("ig")`) | `IG((d−n+ι)/2, ψ[i]/2) = IG(1 + ι/2, ψ[i]/2)`, mean `ψ[i]/ι` |

With `κ_c = κ_a = 1` Mode I reproduces exactly the NIW reduced-form prior mean (for every `A`), its marginal
variances and `E[Σ] = diag ψ` (N05 §4.3, proved algebraically); it is not the same distribution. With `κ_c < 1`
the implied reduced-form cross-lag variance leaks by `κ_c² + (1 − κ_c²)s_jm` (N05 §4.3);
`priors/structural.implied_reduced_form_moments(...)` returns the closed form (Monte Carlo for non-diagonal
cases) and `fit.summary()` reports it (R8; SPEC §14 "users expect reduced-form Minnesota").

**Mode II** (`Constant("iw")` × structural; the analytic twin): `λ_i ~ IG(1 + ι/2, ψ[i]/2)`,
`a_i | λ_i ~ N(0, λ_i κ_a diag(1/ψ[j]))`, `δ_i | λ_i ~ N(m_i, λ_i Ω^{(i)})` with
`Ω^{(i)}_{(l,m)} = κ² c_im (d−n−1)/(l^α ψ[m])`, `c_ii = 1`, `c_im = κ_c²`, `Ω^{(i)}_{det} = κ_d`. With
`κ_c = κ_a = 1` it **is** the Phase 1 NIW prior (N05 §4.2; log ML identical to 0.0, with SOC/SUR); with
`κ_c ≠ 1` it is Chan's (2022) asymmetric conjugate prior, whose closed form lives in
`engines/structural_nig.py`. Mode II changes the mean prior, so `axes["prior"]` carries `scaling=conjugate` and
`compare()` cannot issue a vol-axis verdict across Mode I and Mode II (R2).

**IW-equivalent** (`Constant("iw_equivalent")`, either form; Chan 2022 Prop. 1, N08 §0): `λ_i ~ IG(1 + ι/2,
ψ[i]/2)`, `a_ij | λ_i ~ N(0, λ_i κ_a/ψ[j])`; the coefficient prior is unchanged (Mode I in structural form,
§6.2 in reduced form). At `κ_a = 1` the implied `Σ ~ IW(diag ψ, n + 2)` exactly (N08 §B5 item 4). In reduced
form it is **order-invariant** (N08 §B2); in structural form it is not (the δ prior is per structural equation).
It is a covariance-prior choice, so it shows in `axes["vol"]` (`"Constant(iw_equivalent)"`), not in the prior
label (fixes E10).

Mode III (ψ-scaled, CKY convention) is not shipped; `implied_reduced_form_moments` reports its `(1 + κ_a)^{i−1}`
growth if a user builds it by hand (N05 §4.4).

### 6.2 Reduced form (D-11; N04 §3.1–3.2)

`π_j ~ N(m_j, V_j)` independent over `j`, `V_j` diagonal, moment-matched to the NIW marginal at `cross = 1`:
own lag `κ²/l^α`; lag `l` of variable `m ≠ j`: `κ² κ_c² ψ[j]/(l^α ψ[m])`; deterministic: `κ_d ψ[j]`
(`= κ_d ψ[j]/(d−n−1)`; N04 §3.1 writes raw `var_const`, immaterial at 1e7 but recorded as a decision). Means:
own first lag from tcodes, 0 elsewhere. The `(A, λ)` prior is the structural one (Mode I `a` and IG `λ`, or
`iw_equivalent`), so the two forms differ only in the coefficient prior (N04 §3.2). For parity,
`priors.reduced.explicit_reduced(mean0, var0, a_mean0, a_var0) -> (CoefPriorArrays, ContempPriorArrays)` builds
arrays from explicit numbers (engine-level, test-only).

### 6.3 Volatility priors and the stochvol / bayesianVARs mapping (D-13; N02 §1, §10.2, §13.6; N04 §3.3–3.4)

Parametrisation of stochvol: `μ ~ N(b_μ, B_μ)` (`B_μ` a variance), `(φ+1)/2 ~ Beta(a_φ, b_φ)`,
`σ² ~ B_σ·χ²₁ ⇔ σ ~ |N(0, B_σ)|`.

| Setting | `SV()` default | stochvol `svsample` | bayesianVARs `specify_prior_sigma(type="cholesky")` |
| :--- | :--- | :--- | :--- |
| `μ` | `N(log(ψ[i]/ι), 10)` (data-centred, per window) | `priormu = c(b_μ, sqrt(B_μ))` | `cholesky_priormu = c(b_μ, sqrt(B_μ))` (one value for every equation) |
| `(φ+1)/2` | `Beta(20, 1.5)` | `priorphi = c(a_φ, b_φ)` | `cholesky_priorphi = c(a_φ, b_φ)` |
| `σ²` (ScaledChi2) | `B_σ = 1` (RW: 0.1) | `priorsigma = B_σ` (**not** its square root) | `cholesky_priorsigma2 = c(0.5, 0.5/B_σ)`; shape must stay 0.5 |
| `σ²` (InvGamma) | option `InvGamma(c₀, C₀)` | `specify_priors(sigma2 = sv_inverse_gamma(c₀, C₀))` (general sampler; posteriors only) | not offered |
| `h_0` | stationary | `priorlatent0 = "stationary"` | `cholesky_priorh0 = "stationary"` (compare `h_{1:T}` only) |
| `h_0 ~ N(μ, B₀σ²)` | `SV(h0=B₀)` | `priorlatent0 = B₀` | `cholesky_priorh0 = sqrt(B₀)` |
| offset | relative `c_i = 1e-12·ψ[i]` | `c = 0` on data without zeros | `expert_sv_offset = 0` |
| exact correction | on | parity runs use `exact=False` | parity runs use `exact=False` |
| ASIS | GIS-C, 2-block C-step, fold | `expert = list(interweave = TRUE)` (takes `|σ|` without the flip) | via stochvol `update_fast_sv` |

```python
@dataclass(frozen=True)
class SV:                                                    # components/specs.py (pure Python)
    rw: bool = False
    outliers: bool = False
    tails: Literal["normal", "t"] = "normal"
    mu_prior: Normal | Literal["data"] = "data"              # "data": N(log(ψ_ι/ι), mu_var)
    mu_var: float = 10.0
    phi_prior: Beta = Beta(20.0, 1.5)                        # on (φ+1)/2; ignored if rw
    sigma2_prior: ScaledChi2 | InvGamma | None = None        # None → ScaledChi2(1.0); RW: ScaledChi2(0.1)
    h0: Literal["stationary"] | float = "stationary"         # float = B₀ multiplier (AR only)
    rw_h0_var: float = 4.0                                   # RW: h_0 ~ N(log(ψ_ι/ι), rw_h0_var)
    outlier_max: int = 20
    outlier_prior: OutlierPrior | Beta | None = None         # None → OutlierPrior(every_years=4, or 10 with t; prior_years=10)
    nu_prior: DiscreteUniform = DiscreteUniform(3, 40)
    t_scale: Literal["standardized", "unit"] = "standardized"
    mixture: Literal["ocsn10", "ksc7"] = "ocsn10"
    offset: Literal["relative"] | float = "relative"         # float: absolute c (0.0 = stochvol/bayesianVARs parity)
    exact: bool = True
    asis: bool = True
```

The prior classes of `priors/sv.py` are named by their natural parametrisation (`Normal(mean, var)`,
`Beta(a, b)`, `ScaledChi2(B)`, `InvGamma(shape, scale)`, `DiscreteUniform(low, high)`,
`OutlierPrior(every_years, prior_years)`), which avoids the shape/rate/sd confusions of N06 pitfall 3. To avoid
clashing with the GLP `BetaPrior`/`GammaPrior`, they are exported as `rb.sv_priors.*`.

### 6.4 Transitory priors (D-8, D-9; N03 §0, §3)

- Outliers: `OutlierPrior(every_years, prior_years)` resolves with the window's frequency `f` (12 monthly,
  4 quarterly) to `Beta(a, f·prior_years − a)`, `a = prior_years/every_years`. Defaults: SVO `Beta(2.5, 10f − 2.5)`;
  SVO-t `Beta(1, 10f − 1)`. Unknown frequency requires an explicit `Beta` (actionable error). `K = outlier_max`.
- Student-t: `ν ~ DiscreteUniform(3, 40)` on the static grid (`log_nu_w = −log G`); standardized
  `τ ~ IG(ν/2, (ν−2)/2)`. `t_scale="unit"` gives `IG(ν/2, ν/2)` (JPR, CCMM); it shifts `h` by `log(ν/(ν−2))`, so it
  is the same model only up to the `μ` prior (N03 §3.2).
- None of these hyperparameters depends on the data (no look-ahead).

### 6.5 Minnesota hyperparameters in MCMC (D-12; fixes E5, E25)

| `hierarchy` | MCMC behaviour |
| :--- | :--- |
| `"fixed"` | values as given; exact target; used by every Geweke/SBC test |
| `"eb"` (default) | plug-in mode computed per window before sampling (below) |
| `"mh"` | `NotImplementedError("hierarchical Minnesota inside MCMC arrives with the Phase 3 global-scale step; use 'eb' or 'fixed'")` at construction |

EB procedure (`engines/mcmc/eb.py`):

1. Build the conjugate twin `BVAR(Linear(Minnesota(form="conjugate", cross=1.0, contemp=1.0, <same lambda_,
   alpha, psi, mean, var_const, hyper, hyperpriors>, hierarchy="eb")), Constant("iw"), lags, deterministic)` and
   run the **Phase 1 `AnalyticFit`** on the same window (ψ default, COVID ψ rules, multistart — all unchanged)
   inside `warnings.catch_warnings(record=True)`.
2. `form="structural"`, `cross ≠ 1`: maximise Chan's (2022) per-equation NIG log ML (eq. 11; N05 §9) plus the
   GLP log hyperprior over the free set (`lambda`, optionally `alpha`) with `cross` fixed, starting from the twin's
   mode (L-BFGS-B on log values, as Phase 1). `psi` in the free set with `cross ≠ 1` is rejected at construction.
3. `form="reduced"`, `cross ≠ 1`: no closed-form objective exists; use the `cross = 1` twin mode and record the
   note "EB at cross=1 plugged into the reduced-form prior" in `fit.hyper["eb_note"]` and `summary()` (a note, not
   a warning, so backtests stay quiet).
4. Plug the EB values (λ, and ψ/α if free) into the WP6 builders. Propagate the twin's (or Chan EB's)
   `converged` flag into `MCMCFit.converged` and its optimisation and ψ messages into `fit.warnings`. Setting
   warnings (tcode clash, COVID coverage) are recomputed by the MCMC fit itself, because `SV(outliers=True)`
   counts as an outlier treatment.

At `κ_c = κ_a = 1` the NIW ML is exactly the marginal likelihood of the Mode II Constant model (N05 §4.2), so EB
is true type-II ML for the twin; for Mode I and SV it is a documented plug-in approximation (N05 §9 "Phase 2
shortcut").

**Why not MH-within-Gibbs on λ in Phase 2.**
1. *Target and tests.* Fixed hyperparameters keep the target fully specified; Geweke and SBC need
   hyperparameters that do not depend on `y` anyway (N06 §2.1).
2. *Verifiability.* The EB objective is the Phase 1 code, at parity with R `BVAR` and BoE `bvar` to ~1e-11.
3. *Fair ablation (R2).* EB λ depends on the data window only, so `Linear × Constant` and `Linear × SV` get the
   same mean prior; a sampled λ would move with the volatility arm.
4. *Mixing.* `λ | {≈1,600 structural coefficients}` is nearly degenerate; doing it properly needs the
   non-centred global-scale machinery that Phase 3 builds for global-local priors, behind the same `GlobalStep`.
5. *Cost.* No per-sweep cost, and the eigen fast path survives (it needs a fixed prior). λ-only EB costs seconds
   at `n = 20` (Phase 1: 2.1 s at `n = 100`).

### 6.6 Data-dependent quantities (R9)

Computed per window in `resolve.py` from the window only, passed as traced arrays (no recompile): ψ; own-lag
means `M`; EB values; offsets `c_i = c_rel ψ[i]`; data-centred SV levels `log(ψ[i]/ι)`; RW `h_0` means; the outlier
`Beta` from the frequency. Tests freeze them (N06 §2.1).

---

## 7. Forecasting, LPD and IRFs (`engines/mcmc/forecast.py`)

### 7.1 Draw selection and seeds

`fit.forecast(horizon, draws=1000, seed=None)`: `⌈draws/C⌉` stored parameter draws per chain, evenly spaced (the
`(chain, draw)` structure is kept; the total can exceed `draws` by less than `C`). Each selected draw is keyed by
its kept-draw index `d` (the `draw` coordinate of `fit.posterior`), not by its position in the store, so in-loop
(§5.8) and post-hoc forecasts agree whatever `store_thin` is. If that exceeds the stored
draws, each parameter draw gets `⌈draws/(C·D_stored)⌉` paths with independent shocks (each `(chain, draw)` slice
is still one path from one parameter draw). `seed` accepts `SeedLike`: `rng = np.random.default_rng(seed)`,
`label = seed_label(seed, rng)`, `fseed = int(rng.integers(0, 2**32))` (the backtest's generator advances, as with
the analytic engine).

### 7.2 Exact conditional moments (SPEC §6.1, §7; N03 §5.3; N05 §8)

Pure core, vmapped over draws in chunks of `S_c = 1024` and compiled once per `(StaticConfig, H, S_c)`:

```python
def forecast_core(draw: ParamDraw, hist: f64[p, n], t_last: f64[], shocks: ForecastShocks, program: Program
                  ) -> ForecastOut
# ParamDraw  = (glob: GlobDraw, mean: MeanDraw[n], struct: StructDraw, vol: VolDraw[n])
# ForecastShocks = (eps: f64[H, n], vol_keys: Key[n], mean_keys: Key[n], glob_key: Key)    injected randomness
# ForecastOut = (paths f64[H, n], cond_mean f64[H, n], cond_chol f64[H, n, n],
#                cond_scale f64[H, n], log_lam_pers f64[H, n])
```

```text
(log λ^pers_i, log λ^eff_i)_{T+1..T+H} = vol.draw_ahead_eq(vol_keys[i], vd_i, prior_i, H, gfut)
      SV: h_{T+j} = μ + φ(h_{T+j−1} − μ) + ση_j from h_T (RW: h_{T+j−1} + ση_j); Constant: log λ constant;
      outliers: one uniform → o; t: log τ = log(c(ν)/2) − loggamma(ν/2)                      (N03 §5.2)
fut_i = mean.future_eq(...) and gfut = global futures                                        (Empty in Phase 2)
scan h = 1..H (carry: lag window (p, n)):
    x = regressors_from_window(window, t_last + h, n_det)        jnp port of design.regressors_from_history
    g = [mean.predict_eq(md_i, fut_i, x, h, gfut)]_i             structural: x'γ_i;  reduced: 0
    f = mean.predict_reduced(gd, x, h, gfut)                     reduced: Π'x;       structural: 0
    m = structure.mean_map(sd, g, f)                             A⁻¹g + f  (= B'x)
    L = structure.chol(sd, log λ^eff_{·,T+h})                    A⁻¹ diag(exp(½ log λ^eff)): lower, positive diagonal
    y = m + L ε_h;  window ← [window[1:], y]
```

Given the draw, the path up to `T+h−1` and the drawn `λ_{T+h}` (including `o_{T+h}`, `τ_{T+h}`),
`y_{T+h} ~ N(m, LLᵀ)` exactly, and `L` is the Cholesky factor of the conditional covariance at no extra cost, so
SPEC §7 invariant 1 holds (conditionally on the drawn transitory scales, as SPEC §6.1 allows). Outputs go into the
unchanged `ForecastResult.from_arrays` with `paths_level = data.inverse_paths(paths)`; attrs as the analytic
engine (`model`, `spec`, `spec_hash`, `tcodes`, `seed`, `converged`, `warnings`, version) plus `engine="mcmc"`,
`ordering` (since the Phase 2 closeout the analytic engine writes `ordering` too, §2.10), `mixture`, `exact`,
`n_chains`, and §5.6 metadata. Keys come from `(fseed, c, d)`, so recomputing
moments reproduces the same `o_{T+h}`, `τ_{T+h}` and `ε` (N03 §5.3 requirement 1).

### 7.3 LPD (D-20; fixes E12)

`ForecastResult.log_pred_density` keeps its definition (SPEC §7). Changes in `forecast/result.py`:

- **MCSE for autocorrelated draws.** When `attrs["engine"] == "mcmc"`, `return_mcse=True` multiplies the iid
  delta-method MCSE by `√(S/ESS)`, with `ESS = diagnostics.ess_mean(w)` of the weights `w_s = exp(logp_s − max)` on
  the `(chain, draw)` array.
- **Leading-block fast path** (all engines, exact): when `variables` is a leading block `{0..j−1}` of the ordering,
  the marginal factor is `L[:j, :j]` (L is lower triangular), so no QR is needed.
- **Optional accelerator variables** (never contract fields): MCMC forecasts attach `cond_A (chain, draw, variable,
  variable_)` (unit lower) and `cond_scale (chain, draw, horizon, variable)`; when present and the subset is the
  full vector or a leading block, `log N = −½ j log 2π − Σ_{i<j} log ℓ_i − ½‖(A(y − m))_{:j} ⊘ ℓ_{:j}‖²` (exact,
  `(A⁻¹)₁₁ = (A₁₁)⁻¹`), O(n²) per draw. `from_arrays` gains `extra: Mapping[str, tuple[tuple[str, ...],
  np.ndarray]] | None = None`; `check_structure` and every existing consumer ignore extras. Tested equal to the
  `cond_chol` path to 1e-12.
- **Opt-in transitory Rao-Blackwellisation** (`rb_transitory=True`; stretch item of WP17): needs the extras
  `cond_scale_persistent`, `log_p`, `nu`; evaluates `log p(y_{1:j}) = Σ_{i≤j}[−½ log λ^{pers}_{i,T+h} + log f_i(η_i)]`,
  `η_i = e^{−½ log λ^{pers}}(A(y − m))_i` (N03 §5.3 item 2, CCMM S.11–S.16) for the full vector or a leading block,
  with `f_i` from `TransitoryScale.log_density`; other subsets warn and fall back. Default off; open question Q7.
- **Opt-in path-integrated estimator** (`lpd_method="integrated"`, D-34; default `"conditioned"`, the SPEC §7
  estimator above, whose values do not change). The conditioned estimator averages `p(y_{T+h} | θ_s,
  path_s up to T+h−1)`; for `h > 1` and an outcome in the predictive's tail one draw whose simulated path wandered
  towards it carries all the weight (Constant UR h = 8 at 2008Q2: −93.6 with 1000 draws against −19.1 for the
  path-integrated estimator on 1000 draws, MCSE ≈ 1). The integrated estimator conditions draw `s` only on its parameters and its drawn latent scale path (SV
  log-volatilities and transitory scales of `T+1..T+h`, already inside `L_k = cond_chol_{s,k} = A⁻¹ diag(cond_scale_k)`)
  and integrates `y_{T+1..T+h−1}` out: with `Φ_0 = I`, `Φ_j = Σ_{l=1}^{min(j,p)} A_l Φ_{j−l}` and the simulated
  shocks `u_k = paths_k − cond_mean_k`,
  `y_{T+h} | θ_s, λ_s ~ N(m_h, M_h M_hᵀ)`, `m_h = cond_mean_h − Σ_{k=1}^{h−1} Φ_{h−k} u_k` (the zero-shock mean;
  deterministic terms cancel), `M_h = [Φ_{h−1} L_1 | Φ_{h−2} L_2 | … | Φ_0 L_h]` (`n × hn`). Every subset (leading
  or not) gets the lower factor of `(M_h M_hᵀ)[idx, idx]` from the QR of `M_h[idx, :]ᵀ` (`linalg.marginal_chol`;
  `M_h M_hᵀ` is never formed), then `mvn_logpdf_chol`; the MCSE is the same delta-method formula (ESS-adjusted for
  MCMC). No additional random numbers, but still a Monte Carlo average over the forecast's parameter and
  volatility-path draws: exact given those draws (the exact predictive density only with one parameter draw and a
  constant volatility; under SV, exact given each drawn volatility path), and for tail outcomes its weights can still
  concentrate on a few draws, where the delta-method MCSE understates the spread (numbers in NOTES 8b). At `h = 1`
  the estimators coincide and the conditioned code runs (bitwise; `lag_coefs`
  not needed). For `h > 1` a forecast without `lag_coefs`, or with other dims, raises `ValueError` (no
  warn-and-fall-back: warnings are swallowed on the backtest path), and so does `rb_transitory=True` (the
  transitory scales of `T+1..T+h−1` would have to be integrated jointly; no closed form), whatever the
  `transitory` attribute. **Data:** MCMC forecasts attach the accelerator variable `lag_coefs (chain, draw, lag,
  variable, variable_)`, `lag_coefs[c, d, l−1, i, j] = A_l[i, j] = B[n_det + (l−1)·n + j, i]` with `B` the
  reduced-form `(k, n)` coefficients of the slot's parameter draw (replicate slots `R > 1` and in-loop forecasts
  included): `structure.reduced_form(a, Γ)` (linear structural), the CCCM `Π` (reduced) or zeros (zero mean), via
  the same map as `draw_parameters` (§7.4); other mean kinds attach nothing. The jitted core, `ForecastOut` and
  the in-loop storage stream are unchanged; an in-loop forecast whose draws are not all stored carries no
  accelerator variables. Analytic-engine, benchmark and ordering-pooled forecasts carry no `lag_coefs` in this
  change, so the integrated estimator raises for them at `h > 1`. **Size:** `lag_coefs` is attached to every such
  forecast and costs `S·p·n²·8` bytes, `(p/H)·nbytes(cond_chol)`, with each draw's `A_l` repeated for its replicate
  slots; the default estimator and `h = 1` never read it, and for `p > H` it is the largest variable of the dataset
  (a monthly `n = 7`, `p = 13`, `H = 12` forecast of 1000 draws: 5.1 MB against 4.7 MB of `cond_chol`). Storing
  it once per parameter draw, or only on request, was not done (the always-on attachment is the D-34 decision).
  Tested: exact against a companion-form oracle
  (one draw, constant `Σ`, `h = 2..8`, joint/leading/non-leading subsets, 6-s.d. tails; 1e-10) and against the
  mixture of exact densities (heterogeneous draws, time-varying factors), unit equivariance at 1e±6 scales, and on
  MCMC fits per slot against `jax.jacfwd` of `simulate_paths` at `ε = 0` (1e-10; `tests/unit/test_lpd_integrated.py`,
  `tests/mcmc/forecast/test_lpd_integrated_mcmc.py`).
  **Scoring** (`evaluation/scores.py`): `score_forecast(…, lpd_method=…)` passes it to the joint call (with MCSE)
  and to every per-variable call. The default output is the Phase 1 layout (`tests/unit/test_scores.py` pins the
  variable set). With `"integrated"`, `lpd`, `lpd_mcse` and `lpd_marginal` hold the integrated estimator, and the
  dataset also has `lpd_marginal_mcse (horizon, variable)` (its per-variable MCSE), the conditioned estimator on the
  same forecast (`lpd_conditioned`, `lpd_mcse_conditioned (horizon)`, `lpd_marginal_conditioned (horizon,
  variable)`) and `attrs["lpd_method"] = "integrated"`; the NaN pattern of missing outcomes is the default one and
  every non-LPD score is unchanged (no random numbers). An unknown method raises before anything is scored. The
  backtest plumbing is in §11.6; `ordering_sensitivity` forwards `lpd_method` in forecast and backtest modes
  (recorded in attrs) and rejects a non-default value in prior mode (no LPD there).
  **Review amendments (2026-10-23):** (1) `_lag_coefs` flattened the parameter leaves with `reshape(-1, …)`,
  which fails on the empty `a (C, D, 1, 0)` leaf of a univariate model, so every univariate MCMC forecast (post-hoc,
  in-loop and in backtests, whatever the estimator) raised; it now reshapes to the explicit `C·D` slots (tests:
  univariate fits, post-hoc with `R = 1, 3` and in-loop, against scalar recursions; synthetic `n = 1` draws).
  (2) `log_pred_density` raises `ValueError` for an empty or repeated `variables` subset, for either estimator (a
  repeated variable makes the subset covariance singular; the estimators returned NaN or finite nonsense, and an
  empty subset an `IndexError`); valid subsets are unchanged. (3) The residual Monte Carlo error of the integrated
  estimator at tail origins is stated in its docstring, NOTES 8b, the CR reliability note and the
  pre-registration; the CR acceptance config uses 8000 forecast draws for this reason (decided in the
  pre-registration, `c823d20`, before any acceptance score existed).

### 7.4 Parameters and IRFs

- `fit.draw_parameters(size, rng=None, date=None, which="persistent") -> (coefs (S, k, n), chol (S, n, n))`:
  `B = (A⁻¹Γ)'` (or `Π`), `chol = A⁻¹ diag(exp(½ log λ_date))`. `date=None` is the last in-sample period (`h_T` from
  the stored `VolDraw`); other dates need `store="paths"` and otherwise raise with that advice; Constant is
  date-free. Mirrors `AnalyticFit.draw_parameters`.
- `fit.irf(horizon=20, draws=1000, seed=None, shock="sd", date=None, which="persistent") -> IRFResult` through the
  existing `structural.irf.recursive_irf(coefs, chol, lags, n_det, horizon, shock)`. Unit shocks give
  `Φ_h A⁻¹` (date-free); one-s.d. shocks use the **persistent** variance by default (N03 §5.4: a "one-s.d. shock in
  March 2020" excludes the outlier scale); `which="effective"` is available. FEVD is date-specific under SV.

### 7.5 Hooks for later phases

`simulate_paths(params: ParamDraw, hist, t_last, eps: f64[H, n], log_lam_eff: f64[H, n], program) ->
(paths, cond_mean, cond_chol)` is public inside the engine: GIRF with common random numbers (Phase 4) reuses
`eps` and future variances across baseline and shocked paths; conditional forecasts (Phase 3) replace only the
Gaussian step (Waggoner–Zha); the in-loop hook (§5.8) calls it inside the scan; PSIS-LFO (Phase 4) builds a
per-equation particle filter from `transition_eq` and `obs_logdensity_eq` with the stored `(A, Γ, μ, φ, σ, p, ν,
h_T)`.

---

## 8. Diagnostics and the `converged` flag (`engines/mcmc/monitor.py`, `diagnostics.py`)

### 8.1 Monitored sets (D-33; fixes E3, E16)

| Set | Quantities | Count at `n = 20` |
| :--- | :--- | :--- |
| **Primary** (decides `converged`) | exact log-likelihood `Σ_i Σ_t m_t log N(e_it; 0, λ_eff,it)` given latents; per equation: own first-lag reduced-form coefficient `B_{(1,i),i}` (device: one triangular solve per kept draw); Constant: `log λ_i`; AR SV: `μ_i, φ_i, σ_i, h_{i,T}`; RW SV: `σ_i, h_{i,T}` | 41 (Constant), 101 (SV) |
| **Secondary** (reported; warning only) | every `a_ij`; `h_{i,1}`, `h_{i,⌈T/2⌉}`; intercepts; `log p_i`; `ν_i` with its escape probability `P(ν^{(m+1)} ≠ ν^{(m)})` (N03 §6); log prior | ≈ 290 |

This matches SPEC §10.6 ("hyperparameters, a subset of coefficients, log-volatilities at selected dates, and the
log-likelihood") while keeping the multiplicity of the gating set near 100.

### 8.2 The flag

```text
converged = (no chain raised an error)
          ∧ (chains ≥ 2)
          ∧ ∀ q ∈ primary: rhat(q) < settings.rhat_max (1.01)   rank-normalised split R-hat, max(bulk, folded)
          ∧ ∀ q ∈ primary: ess_bulk(q) > settings.ess_min (400)
          ∧ (hierarchy == "eb" ⇒ EB optimisation converged)
```

Tail ESS is reported, not gated. Acceptance rates never gate (E21). Single-chain runs are never converged (split
R-hat cannot see between-chain disagreement), consistent with the Phase 1 MH rule. The flag goes into forecast
attrs `converged`, so the backtest score `converged` and `compare()`'s "≥ 95% of origins" rule (SPEC §9.5) work
unchanged. Cost (Stage 2 amendment, measured): about 1 s CPU for flag, table and warnings at n = 20 SV-O-t
(392 quantities) and 4 × 2000 draws (the default before D-35); about 7 s at 4 × 10,000 draws, split between the rank
normalisation (two argsorts and a sort per quantity, ~40%) and the five FFT-based ESS per quantity (~25–40%). Each quantity is evaluated once per fit (content-addressed memo); a
memoised value equals a recomputation to rounding (the FFT's last bits depend on the batch it ran in).

With the pre-D-35 default of 4 × 2,000 kept draws this flag was routinely `False` for SV fits (Q14, §12.2). D-35
(2026-10-24) makes 4 × 10,000 the default of SV compositions and adds a warning below it (§8.3); the flag itself is
unchanged and can still be `False` at 4 × 10,000 (BENCH-P2: 4 of 101 primary quantities).

### 8.3 Warnings (`fit.warnings`, copied into forecast attrs, aggregated by the backtest)

| Condition | Message (abridged) |
| :--- | :--- |
| primary R-hat or ESS fails | five worst `(quantity, R-hat, ESS)`; "increase draws, burn or thin" |
| secondary: > 5% of `a_ij` with R-hat > 1.01 | fraction and worst three (informational) |
| `chains == 1` | split R-hat cannot detect between-chain disagreement |
| SV composition with fewer than 10,000 kept draws per chain (D-35; resolution-step input `sv_budget`, so `BVAR.fit` only) | the budget against the SV default, the Q14 evidence (4 × 2,000: README example 0 of 11 origins; CR `SV(rw)` 68% at 4 × 5,000, 106/106 at 4 × 20,000; BENCH-P2 4 of 101 at 4 × 10,000); "raise the budget with `mcmc=MCMCSettings(draws=10_000)` or more, or leave draws unset" |
| exact-`h` acceptance < 0.5, or longest rejection run > 50 sweeps, any equation | mixture-correction mixing (N01 §10.2): acceptance alone is not enough |
| NC fold fraction > 1% | near-homoskedastic equation; stochvol/bayesianVARs differ there (D-7) |
| ν posterior mass ≥ 0.5 at the grid bound 40 | "near-Gaussian; SV may suffice" (report only; mass at 3 is normal, N03 §6) |
| outlier scale at `K` for > 1% of rows | consider `outlier_max` |
| floor hits > 0 (`zero_offset` mode) | exact zero residuals masked (N02 §2.4) |
| non-order-invariant composition with `n ≥ 4` | one-line R8 note pointing to `ordering_sensitivity(mode="prior")` (N08 §B4) |
| sample covers 2020 without `SV(outliers=True)` | SPEC §5.6 (Phase 1 wording) |
| tcode / own-lag mean clash | Phase 1 check reused |
| EB optimisation not converged | propagated from the twin |

### 8.4 `fit.diagnostics` and `fit.summary()`

`fit.diagnostics` (DataFrame, one row per monitored quantity): mean, sd, MCSE of the mean and of the 5/50/95%
quantiles (Vehtari et al. 2021 §4.3–4.4), R-hat, bulk and tail ESS, `primary` flag; per-equation table: MH
acceptance per sub-step, longest exact-`h` rejection run, NC fold fraction, floor hits, smallest pivot,
Rao-Blackwellised outlier probabilities. `fit.summary()`: prior form and mode (Mode I / II / IW-equivalent),
hyperparameters in use and the EB note, implied reduced-form prior moments (R8), ordering, and the primary
diagnostics. `diagnostics.py` gains `ess_tail`, `ess_mean`, `mcse_mean`, `mcse_quantile`, and vectorised
`rhat_many`, `ess_bulk_many`, `ess_tail_many` over `(C, D, Q)` arrays (existing functions untouched).

---

## 9. Validation harness (`src/rai_bvar/validation/`) and test tiers

Every block is independently testable, every composition is tested end to end, and every test that compares a
sampler with "the truth" uses a truth written by someone else (R6; N06 §0).

### 9.1 Principles and the reference registry (D-22; fixes E9)

- **Two-implementation rule** (N06 §2, §5.1; N03 §7.2). Prior samplers, observable simulators and `log_joint` are
  written by WP9 from the parametrisations of §6 and the notes, never by kernel authors, and they never import
  kernel modules (`components/*`, `engines/mcmc/linalg.py`). Allowed imports: the state containers
  (`state.py`), `keys.py`, `config.py` and the jnp design builder `engines/mcmc/data.py` (itself tested against
  the NumPy `data/design.py`). The OCSN/KSC tables are transcribed a second time in `validation/reference.py`, and
  a unit test requires both transcriptions to agree on every printed digit.
- **Registry** mirroring the kernel registry, so Phase 3–5 components add a reference class and immediately get
  per-block and per-composition Geweke configurations:

```python
def register_reference(kind: str) -> Callable[[type], type]          # kinds: "recursive", "linear_structural",
                                                                      # "linear_reduced", "zero", "fixed_diagonal", "constant",
                                                                      # "sv", "sv_rw", "outlier", "t", "outlier_t", "none"
class Reference(Protocol):
    kind: ClassVar[str]
    def prior_sample_eq(self, key, i: i32[], prior_i: EqPrior, static: StaticConfig, tmask, T_eff) -> PyTree
    def prior_sample_global(self, key, prior: PriorArrays, static: StaticConfig) -> PyTree
    def log_prior_eq(self, part, prior_i: EqPrior, static: StaticConfig, tmask) -> f64[]
    def log_prior_global(self, part, prior: PriorArrays, static: StaticConfig) -> f64[]
    def latent_log_var_eq(self, vol_part, which: Literal["effective", "persistent"]) -> f64[T_pad]
    def test_functions_eq(self, part, tmask) -> dict[str, tuple[f64[], bool]]   # (value, finite variance under p(θ, y))
```

- Test priors are proper and tight (`r = Var_prior/Var_post ≤ 5`, N06 §3.3); test functions have finite variance
  (N06 §3.4: no raw data moments under SV, `log λ` rather than `λ²` unless shape > 4, `h` and `|h|` instead of `h²`
  under `Beta(·, 2)`); R9 recomputation is off; every configuration uses `T_eff` not a multiple of 32
  (20 → 32, 50 → 64) so masks are always exercised; the harness drives the **production** `make_sweep(static)` and
  `prepare`, never a copy.

### 9.2 `JointSpec` and configurations (`validation/joint.py`, `simulate.py`, `log_joint.py`, `testfns.py`)

```python
class JointModel(Protocol):                                   # validation/geweke.py (WP7): model-agnostic
    names: tuple[str, ...]
    def prior_sample(self, key) -> PyTree
    def simulate(self, key, state) -> PyTree
    def step(self, key, it: u32[], state, data) -> PyTree
    def test_functions(self, state, data) -> f64[G]

@dataclass(frozen=True)
class FixedDesign:
    presample: np.ndarray            # (p, n) fixed y_{1−p:0}
    T_eff: int; t0: float = 2.0

@dataclass(frozen=True)
class JointSpec:                                              # implements JointModel (WP9)
    name: str; static: StaticConfig; prior: PriorArrays; fixed: FixedDesign
    live: frozenset[str] = frozenset({"all"})                 # block-level configs: others held at their prior draws
    def prior_sample(self, key) -> ChainState                 # θ incl. h, o, τ, p, ν (never s)
    def simulate(self, key, st: ChainState) -> EngineData     # recursive in t, CURRENT latents (Geweke error 2)
    def step(self, key, it, st, data) -> ChainState           # prepare(data, prior) then the production sweep
    def test_functions(self, st, data) -> f64[G]

def simulate_obs(key, st: ChainState, static: StaticConfig, prior: PriorArrays, fixed: FixedDesign) -> EngineData
    # y_t = A⁻¹(Γx_t + exp(½ log λ_eff,t) ⊙ ε_t)  (reduced: Π'x_t + A⁻¹(…)); X rebuilt by data.build_engine_data
def log_prior(st: ChainState, prior: PriorArrays, static: StaticConfig, tmask) -> f64[]
def loglik(st: ChainState, data: EngineData, prior: PriorArrays, static: StaticConfig) -> f64[]   # exact, given latents
def log_joint(st, data, prior, static, s: i32[n, T_pad] | None = None) -> f64[]
    # s given: the augmented joint of the mixture model (for the s- and h|s-step identities)
def joint_spec(name: str, size: Literal["small", "full"] = "small", **overrides) -> JointSpec
COMPOSITIONS: tuple[str, ...]; BLOCK_CONFIGS: tuple[str, ...]
```

Block-level configurations (one block live, the others held at prior draws by identity kernels injected through
`_overrides`; they localise failures, N07 §13.2). **As implemented** (`validation/joint.py`, a documented
deviation recorded at the Phase 2 closeout, 2026-10-23): the production sweep runs and every leaf outside `live`
is then restored to its pre-sweep value (`hold_frozen`). This is an exact kernel for the live blocks given the
frozen ones whenever every frozen block runs after every live block within the sweep (vol after the equation
step; the persistent core after the transitory steps; `a` and vol after CCCM). A frozen block that runs before a
live one is held by an identity override of its named internal instead (`contemp_only`: `cccm_core`).

| Config | Live | Data | Exact |
| :--- | :--- | :--- | :--- |
| `aux_sv` | `s, h, θ_SV` | `y* = h + m_s + √v_s z` (auxiliary model) | yes (N01 §12.3(1)) |
| `gauss_sv_exact` | `s, h, θ_SV`, `exact=True` | `e = e^{h/2} ε` | yes |
| `transitory_known_lambda` | `o, p, ν, τ` with `λ̄` known | N03 §7.1 | yes |
| `constant_transitory` | Constant core + `o, τ` | N03 §4.5 | yes |
| `equation_only` | `(δ_i, a_i)`, `λ` known | regression design | yes |
| `cccm_only` | `Π`, `(A, λ)` known | regression design | yes |
| `contemp_only` | reduced `a` rows, `(Π, λ)` known | regression design | yes |

Test priors (N06 §9.1, corrected values): structural `a_ij ~ N(0, 0.7²)`; `M` own lag 1 = 0.4; δ intercept s.d. 1,
own s.d. `0.10/ℓ`, cross s.d. `0.05/ℓ` (explosive mass ≤ 2e-4); reduced form equation-asymmetric with the same
numbers; Constant `IG(5, 4)`; SV `μ ~ N(0, 0.5²)`, `(φ+1)/2 ~ Beta(10, 2)`, `B_σ = 0.1`; RW `h_0 ~ N(0, 1)`; outliers
`p ~ Beta(2, 18)`, `K = 20`; t `nu_grid = (5, …, 12)` uniform in both simulator and sampler (static grid, D-8),
standardized `τ`; PR size `n = 2` (structural) / `3` (reduced), `p = 1`, intercept, `T_eff = 20`, `y_{1−p:0} = 0`;
nightly `n = 3`, `p = 2`, `T_eff = 50`, fixed stored presample.

### 9.3 Tier U: deterministic and identity tests (every PR)

1. **Dense oracles with injected noise** (N07 §13.1): `precision_draw` vs dense `solve`/`chol` (1e-10); padding
   exactness (padded vs unpadded with `z[:q_i]`, 1e-12); packed and tiled Gram vs dense `Fᵀ diag(w) F` (1e-13
   relative); eigen and fixed-factor draws vs the `cholesky_const` path: identical mean and covariance to 1e-12
   is not achievable in general: both paths are accurate to O(ε·κ), κ = cond(sKs), so compare each path with a
   high-precision reference or path-vs-path with mean 2e-12 + 5εκ(1 + t) and covariance 2e-12 + 5εκ (WP4 fix-up,
   verified on 2,250 fresh problems); plus a KS test on 10⁴ draws; a weighted-path oracle builds its own
   cache with `const_path="weighted"` (only that path carries `FF`); tridiagonal sampler vs dense `Q` and `slogdet` (1e-12); SV `Q̃, b̃`
   builder vs `HᵀS⁻¹H + GᵀV⁻¹G` including masks and `T_eff`; CCCM vs the SUR oracle on the N04 §9.1 table
   (rtol 1e-10) and 20 random problems; A-row gradient/Hessian and sign check (N04 §9.4).
2. **Density-ratio identities** (Grosse & Duvenaud eq. 1; N06 §5.1) at 20 random states per block:
   `log q_b(θ′_b|·) − log q_b(θ_b|·) = log p(θ′_b, θ_−b, y) − log p(θ_b, θ_−b, y)` with the independent
   `log_joint` (relative 1e-8). `s` and `h | s` use the augmented joint; mean, contemporaneous, CCCM, transitory and
   global steps the non-augmented one. MH steps: computed log acceptance ratio equals
   `log π(x′) + log q(x|x′) − log π(x) − log q(x′|x)`, including the exact-mode `w` terms, the IG-σ² single-accept
   combination, and the ASIS Jacobian cancellation.
3. **Closed forms**: mixture constants and moments (mixture mean −1.2704 ± 1e-4; N01 §12.1); indicator pmf χ² and
   Bayes consistency (N01 §12.2); KSC re-sort; `o`, `p`, `τ`, ν-grid conditionals vs brute force (N03 §7.1 tests
   1–4); stable 2×2 algebra vs exact rationals (N02 T7); integrated likelihood vs dense MVN (N02 T6); Constant IG
   conditional including coupling (N05 §5.2).
4. **Infrastructure**: float64 on every leaf; `import rai_bvar` imports no JAX (subprocess); x64 enforced in a
   `JAX_ENABLE_X64=0` subprocess; typed keys; key-reuse checker; **bitwise** invariance across chunk {50, 250},
   burn piece size, thin {1, 3} with aligned sweeps, chain prefixes (first 3 of 5 chains = a 3-chain run), `vmap` vs
   sequential loop, `vmap` vs `shard_map` (4 CPU devices, subprocess); fused vs unfused plan execution; plan
   invariants (§4.8) including the write-set tree diff; zero recompiles across windows of one bucket; failure path
   (non-PD prior injected at sweep 7 → `MCMCNumericalError` with block, equation, iteration 7 and replay
   diagnostics); accumulators zero after burn-in; uint32 counter dtypes.
   **As implemented** (Stage 2 amendment 1; recorded here at the Phase 2 closeout, 2026-10-23): with the WP14
   toy kernels, bitwise across chunk size and burn piece size, thin with aligned sweeps, the exact final state,
   fused vs unfused plans, `vmap` vs a sequential loop, and chain prefixes and `shard_map` with
   `chains="map"`; `vmap` chain prefixes and `shard_map` with `chains="vmap"` to `rtol 1e-12`
   (`tests/mcmc/runner/test_chunking.py`, `test_sweep.py`, `test_devices_bitwise.py`). With the production
   kernels: chain prefixes bitwise with `chains="map"` (`structural_sv`; `structural_sv_outliers_t` in the slow
   tier), and to `rtol 1e-9` on short runs with `chains="vmap"` (`structural_sv`, `reduced_sv`;
   `structural_sv_outliers_t` in the slow tier) and under `shard_map` (4 devices, `structural_sv`), because XLA
   vectorises the SV and CCCM reductions differently for another batch size (§5.2, §5.6). Chunk, burn-piece and
   fused-vs-unfused invariance have no production-kernel test.

### 9.4 Tier B: block-conditional SBC and invariance (every PR)

Block-conditional SBC (N06 §5.2): draw `(θ̃, ỹ)` from the joint, draw `L = 99` values from the block's conditional
given `θ̃_−b`, rank `θ̃_b` and the block log-likelihood; `N = 200`; γ statistic with exact p-values; Bonferroni,
α = 1e-3; Gandy–Scott sequential (k = 3, Δ = 4, fresh keys per stage). MH and ASIS blocks: one-block invariance
(N06 §5.3) with `L = 20` kernel applications.

**As run** (recorded at the Phase 2 closeout, 2026-10-23): per PR, Tier B runs block by block in the kernel
work packages' suites (WP10–WP13: `test_block_sbc_*`, `test_block_conditional_sbc_of_lambda` and the block
Geweke tests in `tests/mcmc/blocks/`). The composition-level Tier B (`tests/sbc/test_block_tier.py`) is in the
weekly slow tier, not per PR.

### 9.5 Tier G: Geweke joint-distribution tests

`C` independent successive-conditional chains, each started at a prior draw, no burn-in;
`z_g = (ḡ_MC − ḡ_SC)/√(σ̂²_g/M₁ + s²_C/C)`; `t_{C−1}` critical values with Bonferroni over `G` (Holm only to list
failures); batch means as a cross-check; Gandy–Scott wrapper so a marginal p-value triggers a fresh 4× rerun
(N06 §3, D-2…D-5).

```python
@dataclass(frozen=True)
class GewekeConfig: C: int; M_c: int; M1: int; alpha: float; seed: int; batch_means: bool = True
def geweke(model: JointModel, cfg: GewekeConfig) -> GewekeResult   # z, p, t_crit, passed, rne per g, failing names
```

Test functions: every scalar parameter, latent states at fixed dates, squares only where the variance is finite,
within-block cross-products (`a_ij·γ_{i,(lag1,j)}`, `μφ`, `φσ`, `p·N_1`, `ν·mean log τ`), cross-equation products
`π_{1,l}π_{2,l}` (reduced form), the exact log-likelihood, `Σ_t e²_it/λ_it`, and log-scale data functions
(`mean_t log y²`, sample correlations, first autocorrelations).

| # | Composition | n | Tier |
| :--- | :--- | :--- | :--- |
| 1–4 | structural Mode I × {`Constant("ig")`, `SV`, `SV(outliers)`, `SV(t)`} | 2 (PR), 3 (nightly) | PR + nightly |
| 5 | structural Mode II × `Constant("iw")` | 2 / 3 | PR + nightly |
| 6 | structural × `Constant("iw_equivalent")` | 3 | nightly |
| 7–10 | reduced × {`Constant("ig")`, `SV`, `SV(outliers)`, `SV(t)`} | 3 | PR + nightly |
| 11 | reduced × `Constant("iw_equivalent")` | 3 | nightly |
| 12–13 | structural / reduced × `SV(outliers=True, tails="t")` | 3 | nightly |
| 14–16 | structural × `SV(rw=True)`; × `SV(sigma2_prior=InvGamma)`; × `SV(h0=B₀)` | 2 | nightly |
| 17 | `ZeroMean × SV`, `n = 1` | 1 | PR |
| 18 | reduced × `SV(rw=True)` (the Clark–Ravazzolo primary SV arm) | 3 | PR + nightly |
| inexact | the SV rows of 1–16 and 18 with `exact=False` (same ids; formerly labelled "18+") | — | nightly, **recorded, not gating** (N06 §8; N01 §12.3(3)) |
| 201–205 | SOC/SUR dummy rows (WP25): structural Mode I × `Constant("ig")`; Mode II × `Constant("iw")`; reduced × `Constant("ig")`; structural Mode I × `SV`; structural × `Constant("iw_equivalent")` | 2 / 3 (reduced, iw_equivalent 3) | PR + nightly |
| blocks | the seven block-level configurations of §9.2 | — | PR |

Sizes: G-small `C = 1000`, `M_c = 100`, `M₁ = 2·10⁵`, α = 1e-3; G-full `C = 4000`, `M_c = 500`, `M₁ = 10⁶`,
α = 1e-2.

**Row 18 (added 2026-10-23, Clark–Ravazzolo acceptance).** The CR replication's primary SV arm,
`BVAR(mean=Linear(prior=Minnesota("reduced")), vol=SV(rw=True), lags=4)` (`benchmarks/clark_ravazzolo.py`
`SV(rw)`), had no row of its own: it was validated only through its parts (row 8, reduced × AR SV; row 14,
structural × RW SV; the CCCM block tests run with `rw=True`). Row 18 (`reduced_sv_rw`) is that composition. Its
`StaticConfig` equals the CR arm's in every component field (`recursive` structure without contemporaneous
scaling; `linear_reduced` mean, `coef_scaling="none"`, `const_path="weighted"`, `gram="packed"`,
`cccm_precompute`; `fixed_diagonal` coefficient prior; SV core with `rw=True`, `sigma2_prior="scaled_chi2"`,
no transitory scale, `ocsn10`, `exact=True`, `asis=True`, relative offset, uncoupled). What differs is what
every row changes by construction: the shapes (`n = 3` at both sizes like the other reduced rows, so CCCM's
middle column `j` is exercised; `p = 1 / 2`, `T_eff = 20 / 50` against CR's `n = 4`, `p = 4` and
`T_eff = 77` (origin 1985Q1) to `182` (2011Q2) on its expanding windows, measured through `resolve()` on the
FRED-QD 2026-09 vintage), the inert `nu_grid` (no transitory scale) and the §9.2 test priors (RW
`h_0 ~ N(0, 1)` against the library default `N(log(ψ_i/ι), 4)`; `B_σ = 0.1` in both, D-13; the tight test
Minnesota prior against CR's θ = 0.2, `1/l²`). It is tiered like its reduced-form neighbours 7–10 (G-small
per PR, G-full and S nightly; the inexact variant recorded), and the reduced × SV(rw) fit joins the MCMC
forecast-contract compositions (`tests/contract`, slow) and the `BVAR.fit → resolve()` compositions of
`tests/unit/test_fit_mcmc_api.py` (slow). At the default `h_0 ~ N(0, 1)` a `V0` used as a precision (or a
dropped `m0`) in the RW AWOL step or NC regression is numerically invisible, while CR runs at `V0 = 4` and
data-centred `m0` (−3.8 to −0.5); so the RW rows also run G-small at `h_0 ~ N(−1, 4)`
(`test_g_small_rw_h0_prior`: row 18 per PR, row 14 nightly; the row's own prior stays the shared §9.2 one).
There a `V0`-as-precision mutant of the RW AWOL step fails (max |z| 13.8 at `log p(θ)` for row 18, 11.3 for
row 14) while production passes (2.33, 1.39; t_crit 4.42, 4.29). The row needs no new reference code:
`references_for` composes `LinearReducedRef` with `SVRWRef` slot by slot. Its SBC thins with a cap of
`t_max = 400` (every other row: 200; both deviate from §9.6's 50 by the ESS rule's own remedy, §9.6 erratum,
documented in `tests/sbc/test_sbc_compositions.py`) at the pre-registered `pilot = 1000`: at 200 the γ test was
clean but 10 of 997 replications had pilot IATs of 187–319, one more than the 99% rule allows; at 400 the
rule holds for every pilot IAT observed. **That pass rests on the rule's blind spot** (a pilot that
underestimates the IAT): at `pilot = 4000` the same run fails the ESS rule alone (16 of 997 replications;
pilot IAT 99th percentile 530, maximum 1438 against 319), with the γ test (min p 0.019) and the pooled-ESS
gate (min 98.7) clean. The slow quantities are the reduced-form coefficients, not `σ`: a `Π` element (an
intercept in 19 of 35) is the pilot-IAT argmax in every replication with IAT > 91, and extending the 20
slowest chains to 16000 sweeps leaves the IAT estimate growing in 10 of them (up to 4376). About 1–2% of
prior-drawn `T = 50` datasets therefore mix with IATs in the hundreds to thousands of sweeps under reduced ×
RW SV; the γ test sees no calibration error. Evidence (2026-10-23, Apple M5 Max): G-small max |z| 1.98,
G-full 1.90, SBC min p 0.019 (59 quantities), all at stage 1; the inexact variant's G-full max |z| 2.49.

### 9.6 Tier S: simulation-based calibration (nightly)

`sbc(model, quantities, SBCConfig(N=1000, L=99, burn=300, pilot=1000, t_max=50, alpha=1e-2, seed))` (N06 §4):
random initialisation from `Stream.INIT` (never at `θ̃`); two-pass per-replication thinning (pilot IAT → fresh chain
thinned by `t_i = ⌈2·IAT⌉`, `thin` traced, replications bucketed by `t_i` to bound the worst `vmap` lane); random
tie-breaking `n_less + ⌊u(n_tie + 1)⌋` with one `u` per quantity (N07 §13.3); data-only filtering
(`finite(ỹ) ∧ max|ỹ| < 1e3`); failed fits fail the test, never dropped. Quantities: parameters, `h` at three dates and
`mean h`, **total and per-equation log-likelihood**, `log p(θ)`, `‖Γ − AM′‖²`, transitory-marginal log-likelihood,
`N_1`, `Σ log τ`, and the predictive unknowns `h_{T+1}`, `y_{T+1}` through the production forecast core. Pass:
γ-test p-values from the exact Säilynoja recursion, Bonferroni over quantities, α = 1e-2; per-replication thinned
ESS ≥ 0.8 L for ≥ 99% of replications. **Erratum (orchestrator, 2026-10-22):** the per-replication ESS is predicted
from the pilot IAT (an AR(1) form, `ess_thin`), because L = 99 kept draws cannot certify 0.8 L for one replication;
that rule cannot see a pilot that underestimates the IAT, so a second gate requires the replication-pooled kept-draw
ESS (`ess_pooled`, Kish-weighted ECDF-indicator IAT) ≥ 0.8 L for every quantity (iid minimum 0.836 L at n = 40).
**Erratum (2026-10-23).** `t_max` bounds the work, not the gate (the pass rule is in ESS): the live caps are 200 for
every other row and 400 for row 18, each raised from 50 by the ESS rule's own remedy, at the pre-registered `pilot = 1000`
(evidence in `tests/sbc/test_sbc_compositions.py`). Row 18 shows the blind spot above is live: at `pilot = 4000` it
fails the per-replication rule (98.4%) with both other gates clean (§9.5 row 18 note).

### 9.7 Tier M: mutation suite (weekly; every mutant must fail some tier; fixes E17)

| Mutant | Injected through | Expected catcher |
| :--- | :--- | :--- |
| M1 CCM-2019 triangular step instead of CTA | `cccm_core` | U (density ratio), B, G (N06 §7: z = 15–140) |
| M2 stale indicators (Primiceri Alg. 1: `s` carried across sweeps, drawn after `h`) | `vol` slot (mutant kernel in `tests/mutation/mutants.py`) | G / divergence (N02 §3.3); **confirmed G**: G-small max \|z\| 702 at the log-likelihood vs control 2.48, run with `exact=False` (`mutants.py` `EVIDENCE`) |
| M3 simulator draws fresh latents | `JointSpec(fresh_latents=True)` | G (Geweke error 2) |
| M4 τ conditional with dof `ν` instead of `ν + 1` | `tau_conditional` | U, G |
| M5 `B_σ` × 2 in the sampler only | sampler prior arrays | G |
| M6 NC `|σ|` without flipping `h̃` | `nc_fold` | G (N02 T16: z = 16–19) |
| M7 uncorrected NC move in exact mode | `nc_accept` | G (N01 §10.3 toy: z = −12.9); **confirmed U and G**: U 2.87 vs control 0; G only with the deliberately coarse one-component mixture table (`COARSE1`, through `mixture_table`; max \|z\| 77.6 vs 2.28), because with OCSN-10 the correction is too small to see (WP3 pilot §3.4; `mutants.py` `EVIDENCE`) |
| M8 D3 centring dropped (`γ_i = δ_i`) | `structural_gamma` | G (`a·γ` cross moment), S (`‖Γ − AM′‖²`) |
| M9 outlier pmf without `1/k` | `outlier_logits` | U, G (N03: z ≈ 392) |
| M10 `b_p + T` in the `p` update | `p_conditional` | U, G |
| M11 indicator pmf without `σ_j⁻¹` | `indicator_logits` | U (Bayes consistency), B |
| M12 double −1.2704 shift | `mixture_table` | U (mixture mean), B |
| M13 IG shape with `T_pad` instead of `T_eff` | `constant_conditional` | G (masks exercised by design) |
| M14 A-row regression with `+u_{<j}` | `contemp_conditional` | U (N04 §9.4), G |
| M15 ASIS map without the `h̃` rescale | `nc_step` | B (one-block invariance), G |
| M16 offset `c = 0.1` at `λ ≈ 0.01`, **`exact=False`** (μ prior `N(log 0.01, 0.5²)`) | `SV(offset=0.1, exact=False)` config | G. Under `exact=True` this is not a bug (exact for any `c`, N01 §10.2), so the mutant is defined only with `exact=False` |
| M17 Mode I SOC/SUR rows weighted by the current `1/λ_i` instead of `1/σ̄_i²` (WP25) | `linear_conditional` | U (density ratio), G |
| M18 SOC/SUR rows dropped from the CCCM `Π` step (WP25) | `cccm_core` | U (density ratio), G |

The suite records which tier caught each mutant; a mutant caught by none blocks the release.

### 9.8 Tier A: analytic twins and exact oracles

| Test | Oracle | Cadence |
| :--- | :--- | :--- |
| A1 | `structural_nig` log ML = Phase 1 `AnalyticFit.log_ml()` (Mode II, `κ_c = κ_a = 1`, `d = n + 2`), with and without SOC/SUR in the analytic engine, to 1e-10; EB λ from `structural_nig` = Phase 1 EB λ at `cross = 1` (1e-6). **WP25 extension:** MCMC Mode II × `Constant("iw")` with SOC/SUR rows vs the Phase 1 NIW posterior with the same rows (A2 comparison, 4 quantile-MCSE; EB λ, μ, δ equal to the Phase 1 modes), with a negative control without the rows | PR (`n = 3`); other paths and `n = 20` nightly |
| A2 | MCMC structural Mode II × `Constant("iw")` vs the Phase 1 NIW posterior: quantiles of `B`, `Σ`, `A` within 4 quantile-MCSE (Holm); `n = 3, p = 2, T = 60` | PR |
| A3 | as A2 at `n = 20, p = 4, T = 300` (SPEC §10.3 realistic-size oracle) | nightly |
| A4 | MCMC Mode II with `κ_c = 0.5` vs Chan closed-form iid draws | nightly |
| A5 | **Mode I × Constant per-equation 1-D quadrature**: `p(λ_i | y) ∝ IG(λ_i; c_i, C_i)·N(y_i; F_iμ₀, λ_i I + F_iV₀F_iᵀ)` (determinant lemma and Woodbury in `q×q`), giving exact `E[λ_i]`, `E[θ_i]`, `Var(θ_i)`; MCMC within 4 MCSE (equations are independent, so it is exact at any `n`) | PR (`n = 3`), nightly (`n = 20`) |
| A6 | toy posteriors by quadrature (N01 §12.6): step order, exact-mode invariance, ASIS correction | PR (owned by WP12) |
| A7 | reduced × Constant SWA oracle (O(n⁶) joint draw in NumPy test code) vs CTA Gibbs, `n ∈ {2, 3, 4}` (N04 §9.5) | nightly (owned by WP11) |

### 9.9 Tier P: parity (fixtures generated offline by `tools/make_r_fixtures.R`; fixes E11)

**stochvol 3.2.9**, block level through the production runner with `ZeroMean`, `n = 1`, data generated in NumPy
and committed as CSV without zeros. (As built: the committed fixtures come from stochvol 3.2.1 under R 4.1.2,
recorded in every fixture's `stochvol_version` and `stochvol_version_note` attributes; the SV-t run is
replaced by an exact importance-sampling oracle until a fixture from stochvol ≥ 3.2.5 exists, §2.10 Stage 4
amendment 1.) (i) `T = 300, μ = −1, φ = 0.95, σ = 0.2`; (ii) `φ = 0.5, σ = 0.5`;
(iii) `φ = 0.99, σ = 0.1`; (iv) standardised AR(4) residuals of a FRED-QD growth series; (v) (i) with
standardized-t errors, ν = 5. Each side is derived from one parameter set through the §6.3 formulas:

| Run | rai-bvar | R |
| :--- | :--- | :--- |
| main | `SV(mu_prior=Normal(0.0, 100.0), phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(0.1), offset=0.0, exact=False)` | `svsample(y, draws=50000, burnin=5000, n_chains=4, priormu=c(0, 10), priorphi=c(20, 1.5), priorsigma=0.1, priorlatent0="stationary")` |
| defaults | `SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(5, 1.5), sigma2_prior=ScaledChi2(1.0), offset=0.0, exact=False)` | `svsample(y, draws=50000, burnin=5000, n_chains=4)` |
| SV-t, fixed ν | main priors + `tails="t", nu_prior=DiscreteUniform(5, 5)` | `svsample(y, …, priorspec=specify_priors(mu=sv_normal(0, 10), phi=sv_beta(20, 1.5), sigma2=sv_gamma(0.5, 0.5/0.1), nu=sv_constant(5)))` |
| exact (optional) | main with `exact=True` | main + `expert=list(correct_model_misspecification=TRUE)` (ex-post SIR; quantiles only, MCSE inflated by the weight ESS) |

Compare 5/25/50/75/95% quantiles of `μ, φ, σ` (and `τ_t` at the three largest `|y_t|` for SV-t) and of `h_t` at
`t ∈ {0, 1, T/4, T/2, T}` within 4 quantile-MCSE. Fixtures where our NC fold fraction exceeds 0.5% are excluded
(stochvol takes `|σ|` without the flip, D-7).

**bayesianVARs 0.1.8**, reduced form, through `fit_mcmc` with explicit arrays (`priors.reduced.explicit_reduced`).
(Not run in Phase 2: bayesianVARs could not be installed, so the exact system-wide Gibbs oracle
`bvars_swa_*` replaced it with the same parameter set and comparison, §2.10 Stage 4 amendment 2; SPEC §13 moved
bayesianVARs parity, with the probes below, to the Phase 3 exit on 2026-10-23.) One parameter set: `π_j ~ N(m_j, diag v_π)`, intercept variance `s_c²`, `a_ij ~ N(0, v_a)`, and
`SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(1.0), h0="stationary", offset=0.0,
exact=False)`. The R call is derived from it:

```r
bvar(data, lags = p, draws = D, burnin = B, thin = 1, sv_keep = "all",
     prior_intercept = s_c,                                          # documented as an s.d.
     prior_phi = specify_prior_phi(data = data, lags = p, prior = "normal",
                                   normal_sds = <v_π or sqrt(v_π), settled by probe P1>, priormean = m_own),
     prior_sigma = specify_prior_sigma(data = data, type = "cholesky",
                     cholesky_U_prior = "normal", cholesky_normal_sds = <v_a or sqrt(v_a), probe P1>,
                     cholesky_heteroscedastic = TRUE,
                     cholesky_priormu = c(0, 100),        # c(b_μ, sqrt(B_μ)), B_μ = 1e4
                     cholesky_priorphi = c(20, 1.5),      # c(a_φ, b_φ)
                     cholesky_priorsigma2 = c(0.5, 0.5),  # c(0.5, 0.5/B_σ), B_σ = 1
                     cholesky_priorh0 = "stationary"))
```

Constant variant: `cholesky_heteroscedastic = FALSE`, `cholesky_priorhomoscedastic = cbind(shape, scale)` (always
explicit) ↔ `Constant("ig")` with an explicit `ConstantPrior`. **Probe fixtures first** resolve the UNVERIFIED
conventions (N04 §3.4, §9.8, §10): P1 variance vs s.d. of `normal_sds` and `cholesky_normal_sds` (tiny `T`,
prior-dominated posterior); P2 `U` vs `U⁻¹` storage and sign (`corr(u₁, u₂) = +0.8` fixture, `A = Uᵀ`); P3 intercept
row placement in `PHI`; P4 `sv_para` layout; P5 the scale convention of `cholesky_priorhomoscedastic`. Results are
recorded in `tests/parity/SV_PARAMETRIZATION.md`. Compare quantiles of `Π`, `A`, `h_t` (selected `t ≥ 1`), `μ, φ, σ`
and forecast quantiles within 4 MCSE on FRED-MD subsets `n ∈ {3, 7}`, `p ∈ {2, 4}`, same ordering; exclude series
where our fold fraction is material.

The Phase 1 engines remain parity-tested as before; A2/A3 tie MCMC to them transitively.

### 9.10 Test layout and CI tiers

Markers (added to `pyproject.toml` by WP1): `geweke`, `sbc`, `mutation`, `bench`, `parity_long`, `weekly`; MCMC
test modules add `filterwarnings = "error:Explicitly requested dtype"` (x64 truncation must fail tests, N07 §3).

| Tier | Contents | When | Budget |
| :--- | :--- | :--- | :--- |
| PR | U, B, contract (MCMC compositions at `n = 2`, short chains), G-small (compositions 1–5, 7–10, 17, 18, 201–205, row 18 at the RW `h_0` prior `N(−1, 4)` and the block configs), A1, A2, A5 (`n = 3`), A6, stochvol parity (fixture read + our run), runner/plan/recompile/failure tests | every PR | cold ≤ 90 min per job; warm recorded per run (amendment below; first stated as ≈ 10–15 min with the persistent compilation cache) |
| Nightly | G-full (all rows, `exact=False` recorded), row 14's G-small at the RW `h_0` prior `N(−1, 4)`, S, A3, A4, A5 (`n = 20`), A7, bayesianVARs parity (moved to the Phase 3 exit; in Phase 2 the long runs against the system-wide Gibbs oracle, §9.9), ordering invariance MC tests | weekly (amendment below) | ≤ 3 h (CI job limit 6 h) |
| Weekly | M, SPEC §10.5 DGP rows relevant to Phase 2 (homoskedastic, SV, SV + outliers; **not implemented** at the Phase 2 closeout, and SPEC §13 places DGP verdict tests at the Phase 3 exit), ordering detection power (N08 §B5 item 5) | weekly | — |
| Pre-release | BENCH-P2 (§10), Clark–Ravazzolo replication (N08 §C5 criteria 1–6) | release (BENCH-P2: the manual CI bench job) | — |

**Amendment (Phase 2 closeout, 2026-10-23): cadence and tiers as run.** `.github/workflows/ci.yml` runs the PR
tier as the fast suite (`-m "not slow and not network and not bench"`, Python 3.12, 3.13 and 3.14 on
`ubuntu-latest`) on every pull request and every push to `main`. The nightly and weekly tiers run **together,
weekly**: one job (still named `nightly`) on Sundays at 03:17 UTC with `-m "(slow or network) and not bench"`,
plus the macOS fast suite; the cron became weekly in `e3b8fa0`. The bench tier (`-m bench`, BENCH-P2 and the
WP10 timing gate) runs only when dispatched by hand (bench = true). Two PR-row items run elsewhere, as their
test modules document: the block configurations' G-small in `tests/geweke` is slow (their per-PR coverage is the
block Geweke tests of the kernel suites, `tests/mcmc/blocks/`), and the composition-level Tier B is weekly
(§9.4). The `pyproject.toml` marker descriptions still say "nightly".

**Amendment (Phase 2 closeout, 2026-10-23): CI compilation cache.** *Diagnosis.* Through `888f620` no
fast-suite job restored the persistent XLA cache: 21 of 21 jobs logged "Cache not found" (runs 39864205394,
39866597126, 39907612654, 39941239235, 40015549161, 40019667165, 40020554571). The restore prefix
`xla-<os>-py<ver>-<md5(model name)>-<uv.lock>-` was stable; three causes compounded. (1) Partition: 3 Pythons ×
6 runner CPU models (AMD EPYC 7763, 9V74, 9V45; Intel Xeon 8370C, 8573C; one unidentified) gave 18 buckets, and
no bucket repeated within a PR (PR #1: 9 jobs, 9 buckets). (2) Scope: an Actions cache is readable only from its
own ref, the PR's base and the default branch; PR entries (`refs/pull/N/merge`) are invisible to other PRs, and
`refs/heads/main` held no `xla` entry until run 40020554571 saved three. (3) Churn: every job saved 206–276 MiB
whether or not it compiled anything (21 entries, 5199 MiB, against the 10 GB repository quota, which evicts the
least recently used entries and expires entries unread for 7 days). *Decision* (ci.yml, steps "XLA cache key",
"Restore XLA cache", "XLA cache size check", "XLA cache report" and "Save XLA cache" of the `test` and `nightly`
jobs). Key: the bucket prefix `xla-<os>-py<ver>-<isa>-<uv.lock>-` (weekly job: `xla-slow-…-py3.12-…`) plus
`<run_id>-<run_attempt>`; restore-keys is the prefix. `<isa>` is the md5 of the `/proc/cpuinfo` "model name" and
"flags" lines: XLA:CPU compiles for the host instruction set, so an entry is never loaded on another ISA, and
the model name alone would miss a host that exposes different features under the same model. Python stays in
the key because jax 0.12.0.1 compresses entries with stdlib zstd on 3.14 and zlib on 3.12/3.13 while its cache
key names "zlib" in both cases, so mixed payloads would fail to decompress. Save policy: save only when the run
wrote new entries (a failed or cancelled run still saves). Size cap: a restored cache over 1 GiB is wiped first,
because entries are never evicted (a cold fast suite writes about 0.25 GiB). Measurement: each fast-suite job
writes the runner CPU, vCPU count, matched key, entries restored and written, and the wall time to the run
summary. A hit loads the executable the same program compiles to and changes no result; no test or threshold
changed. *Residual misses.* A job hits only if its own PR or `main` holds an entry for its (Python, ISA) bucket.
With the observed runner mix (21 jobs: 6/5/4/2/2/2 per CPU model) the projected per-job hit rate after `k`
main-scope runs (a merge push or the weekly run, one sample per Python bucket each) is 20%, 36%, 57%, 79% and
89% for `k` = 1, 2, 4, 8 and 12; the PR tier's wall time is its slowest job, so a whole run is warm only if all
three jobs hit (about 0.89³, or 70%, at `k` = 12). Other misses: an entry unread for 7 days expires; a `uv.lock`
change resets every bucket; kernels a PR changes always recompile; a CPU model new to the pool starts cold. The
new fingerprint resets `main` to `k` = 0 (the three entries saved at `888f620` do not match the new prefix).
*Status:* the changed workflow has not run on GitHub yet (no push from the closeout session); the first PR run
and the first `main` run after the merge give the first measurements. *Not adopted* (owner decisions): sharing
buckets between 3.12 and 3.13 (both write zlib payloads; relies on JAX-internal codec behaviour), and
`XLA_FLAGS=--xla_cpu_max_isa=AVX2` (expected to make entries portable across x86 runners; it does not, and was
reverted after a brief adoption on 2026-10-24: amendment below).

**Amendment (Phase 2 closeout, 2026-10-23): PR-tier budget.** The 10–15 min figure assumed a warm persistent
cache and was never measured in CI, where every PR-tier job through `888f620` ran cold. Measured cold, per job
on GitHub's x86-64 `ubuntu-latest` runner (pytest wall time, 4387–4666 tests, the seven runs from 39864205394 to
40020554571, read from the job logs): 40–85 min for 20 of 21 jobs, median 63 min, plus one 141 min outlier
(run 40015549161, Python 3.13). The CPU model dominates: EPYC 9V45 40–49 min, Xeon 8573C 58–62, EPYC 9V74 60–76,
Xeon 8370C 76–80, EPYC 7763 67–85. The tier's wall time is its slowest of three jobs: 60–85 min per run (141 min
once). Warm, projected (not measured): about one third of a job's cold time, from the local ratio on
`tests/mcmc/runner` (405 s to 137 s, `0941b57`), so roughly 15–30 min per job, which misses 10–15 min on most
runner classes. The measurement is the run summary's XLA cache block, which requires `main` to hold an entry
for the job's (Python, ISA) bucket first; the first warm runs replace the projection. The budget is not a gate:
no test leaves the PR tier to meet it. If warm runs miss, the levers are sharding (pytest-xdist, or one Python
running the MCMC-heavy modules) or pinning `--xla_cpu_max_isa` (adopted 2026-10-24, but it is no lever: JAX
keys entries on the runner's CPU features; amendment below).

**Amendment (2026-10-24): AVX2 cap adopted, then reverted.** The owner approved `XLA_FLAGS=--xla_cpu_max_isa=AVX2`
for the `test` and `nightly` jobs on the expectation that it would make cache entries portable across the x86
runner pool, at the cost of AVX-512 code-generation coverage. The review below showed that the cap gives no
caching benefit, so it was reverted before merging: the jobs compile for the host ISA again (AVX-512 coverage
kept), and the cache buckets stay per CPU model. The analysis is kept for the record.
What the cap does (XLA `d8679698`, the revision jax 0.12.0.1 pins; `xla/backends/cpu/codegen/cpu_features.cc`,
`xla/backends/cpu/target_machine_options.cc`): XLA:CPU removes the host's AVX-512, AMX, AVX10.2 and APX features
from the LLVM target, and on a host that has any of them (every pool model except the EPYC 7763) it also sets the
target CPU to `haswell`; extensions that XLA does not order under an ISA level (GFNI, VAES, AVX-VNNI, vendor
extensions) stay enabled. *Review finding (2026-10-24): the cap does not make cache entries portable.* The
decision expected one cache bucket for every x86 runner. In jax 0.12.0.1 the persistent-cache key of an
executable includes the CPU topology fingerprint (`jax._src.cache_key._hash_accelerator_config`), and XLA's CPU
topology serialises the target machine it compiles for: triple, target CPU name and the host's whole `+`/`-`
feature list after the cap's filtering (`xla/pjrt/plugin/xla_cpu/cpu_topology.cc`; `xla/pjrt/cpu/cpu_client.cc`
builds it from `XLA_FLAGS`). The pool's models still differ in that list under the cap (GFNI, for one: the EPYC
7763 lacks it; the 7763 also keeps its own target CPU, `znver3`), so JAX serves an entry only to a runner with the
same target, whatever the Actions bucket; by the same mechanism a load across CPU models was never possible,
with or without the cap. A single shared `avx2cap` bucket would therefore have gained no hits and accumulated every
model's entries (about 0.25 GiB each) in one Actions entry, tripping the 1 GiB wipe and saving up to 1 GiB per job
against the 10 GB quota. With no caching benefit left, the cap would only have made every runner compile for the
same ISA level at the cost of AVX-512 coverage, so it was reverted: `<isa>` is the per-CPU md5 of the "model
name" and "flags" lines, and the bucket count and the hit-rate arithmetic above apply. The run summary's XLA
cache block prints the ISA bucket and `XLA_FLAGS`, and the weekly `meta.json` records `xla_flags` (both empty
unless a job sets the flag).

**Amendment (Phase 2 closeout, 2026-10-23): weekly validation record and concurrency.** The weekly job sets
`RAI_BVAR_VALIDATION_RECORD=$RUNNER_TEMP/validation-record`; the tests append `geweke.jsonl`, `sbc.jsonl` and
`mutation.jsonl` there (`tests/geweke/test_geweke_compositions.py`, `tests/sbc/test_sbc_compositions.py`,
`tests/mutation/test_mutants.py`). A `meta.json` records the commit, ref, event, run URL and attempt, start
time, marker expression (`TIER_MARKERS`, defined once), runner platform, CPU and logical CPUs, Python, and the
`jax`/`jaxlib`/`numpy`/`scipy` versions; pytest adds `junit.xml`. The directory is uploaded with `if: always()`
as the artifact `validation-record-<sha>-<attempt>`, kept 90 days. The test step has `timeout-minutes: 330`
inside the job's 360, so a run near the 6 h limit still uploads every finished row. The record is committed
from that artifact together with its `meta.json`. Concurrency: the group is now
`${workflow}-${event_name}-${ref}`. Before, a push to `main` and the weekly schedule shared a group with
`cancel-in-progress`, so a merge during the Sunday run cancelled it. A new push to a PR still cancels that PR's
older run, and two manual dispatches on one ref still share a group, as before.

**Amendment (2026-10-24): weekly tier sharded.** *Measurement.* The single weekly job did not finish on x86-64
Linux: run 40052069815 (`779957f`, AMD EPYC 7763, 2 vCPU) hit the 330 min step timeout after 260 of 482 tests
(53%); run 40106532730 (`8f2c969`, EPYC 9V45, 4 vCPU) passed all 483 in 273 min. Minutes per directory, cold
XLA cache, from the progress-line timestamps (7763) and `junit.xml` (9V45), next to the local release record
(`benchmarks/reports/validation/7f880e1_apple-m5-max/summary.txt`):

| Directory | EPYC 7763, 2 vCPU | EPYC 9V45, 4 vCPU | Apple M5 Max |
| :- | -: | -: | -: |
| `tests/geweke` | 273.6 | 95.7 | 83.5 |
| `tests/sbc` | not reached | 83.8 | 82.3 |
| `tests/unit` | not reached | 52.9 | excluded (issue #3) |
| `tests/analytic` | 23.6 | 12.0 | 5.5 |
| `tests/mcmc` | 14.3 (to `runner/test_chunking.py`) | 11.2 | 3.4 (`blocks` only) |
| `tests/integration` | 10.5 | 4.8 | — |
| `tests/contract` | 7.3 | 2.8 | 0.5 |
| `tests/mutation` | not reached | 7.0 | 2.5 |
| `tests/parity` | not reached | 3.1 | 3.8 |

The 7763 takes 2.0–2.9× the 9V45's time per directory. Gandy–Scott reruns (§9.5) add to that and depend on the
CPU's rounding: `test_g_full_inexact_recorded[structural_sv_outliers_t]` ran 2 stages on the 7763 (69.7 min),
1 on the 9V45 (6.4 min) and 2 on the M5 Max (22.3 min); `test_g_full[structural_sv]` ran 1, 2 and 3. Geweke alone
on the 7763 (274 min) leaves no room under the 6 h job limit. *Decision.* The `nightly` job is a matrix of six
shards, defined in `tools/weekly_shards.py`: `tests/geweke` and `tests/sbc` each split by `-k structural` /
`-k "not structural"`, `tests/unit`, and the rest (`tests` with those three directories `--ignore`d, so a new
directory lands there). On the 9V45 the shards take 55, 40, 43, 41, 53 and 41 min. On the 2-vCPU 7763 the two
Geweke shards would take 160 and 113 min, from run 40052069815's per-row times. A `nightly-plan` job collects the tier and every shard with `pytest --collect-only`, fails
unless every test of the tier is in exactly one shard, and passes the matrix on. Each shard runs the same
`-m "$TIER_MARKERS"` (now set once at the workflow level) with `fail-fast: false`. Its timeouts are 240 min for
the test step and 270 min for the job. Its XLA cache prefix carries the shard name (`xla-slow-<shard>-…`):
the programs differ by shard, and concurrent shards cannot save under one key. Its validation record is the
artifact `validation-record-<sha>-<attempt>-<shard>`, whose `meta.json` adds `shard` and `shard_args`. A
`nightly-record` job merges the shard records into `validation-record-<sha>-<attempt>`, one directory per shard,
because two shards each write `geweke.jsonl` or `sbc.jsonl` and each shard ran on its own CPU. No test, prior,
setting, tolerance or criterion changed. This replaces the 330/360 min timeouts of the previous amendment and
the nightly tier's single-job budget of ≤ 3 h, which no x86 runner met. *First sharded run* (run 40140397848,
`b1e3801`; every shard on an EPYC 7763 with 4 vCPU, cold cache): all six shards passed, with 483 tests, each in
exactly one shard (481 passed, 1 skipped, 1 xfailed). The jobs took 120 (geweke-structural), 82, 79, 72, 73 and
74 min in the order above, and the tier took 121 min from dispatch to the merged record. The two
`G-full-inexact` reruns of run 40052069815 (`reduced_sv_outliers`, `structural_sv_outliers_t`) recurred on this
ISA, and they account for most of geweke-structural's lead.

---

## 10. Performance plan

### 10.1 Target definition (BENCH-P2; open question Q2)

- **Model**: `BVAR(Linear(Minnesota(form="structural", hierarchy="fixed")), SV(), lags=4,
  mcmc=MCMCSettings(chains=4, burn=1000, draws=10_000, thin=1))`.
- **Data**: simulated stationary VAR(4)-SV DGP, `n = 20`, `T = 300` (`T_pad = 320`), seed 0, committed generator.
- **Measured**: end-to-end `fit()` wall time including compilation, storage and diagnostics, excluding data
  loading; compile time reported separately (cold and from the persistent cache).
- **Machines**: an idle Apple M-series laptop (8+ performance cores) and a 4–8 core x86 laptop (AVX2).
- **Pass**: < 120 s on one device with default settings. **Stretch**: < 60 s with `set_cpu_devices(4)`.
- **Also reported**: ms per sweep (minimum of 3 × 1000 sweeps), peak RSS, compile time, and **ESS per second** for
  the log-likelihood, `h_{·,T}`, `μ`, `σ` and one cross-lag coefficient (raw speed that buys worse mixing is not a
  win).

### 10.2 Evidence (both judges re-measured the performance proposal's scratch scripts; M1 Pro, jax 0.12, float64)

| Configuration (`k = 81`, `q = 100`, `T = 300`, 4 chains; ms per sweep of all chains) | 1 device (`vmap`) | 4 devices (`shard_map`) |
| :--- | ---: | ---: |
| full Linear × SV sweep, batched einsum Gram (N07 design) | 8.07–8.16 | 3.28 |
| full Linear × SV sweep, **packed Gram** | **4.56–4.60** | **2.47–2.53** |
| Gram kernel alone, 80 equation-chains: batched vs packed | 4.18–4.26 vs 1.00–1.06 (rel. error 2.4e-16) | — |
| batched Cholesky, 80 × 100² | 1.11 | — |
| Mode I × Constant, eigen path | 0.343–0.346 | — |
| `n = 50` (`q = 250`), packed | 45.9 | 24.6 |
| `n = 20, p = 13` (`q = 280`), packed | 22.6 | — |
| `n = 3, p = 2, T = 100` | 0.18 (fixed per-iteration overhead) | — |

Caveats: the sweep prototype uses a stand-in SV-parameter block (no ASIS, no C/NC MH, no exact correction),
closes over data as constants, and runs `T = 300` rather than `T_pad = 320`. The correctness proposal's 20–25 ms
cost model was based on N07's heavily loaded machine and is superseded (E2).

### 10.3 Budget per sweep (4 chains, one device)

| Item | ms |
| :--- | ---: |
| packed Gram + gather | 1.1 |
| batched Cholesky, two triangular solves, right-hand-side and residual GEMMs, normals | 1.9 |
| SV indicators + AWOL | 1.5 |
| exact correction (two extra `(T_pad, 10)` `logsumexp`s, with reuse) | 0.4 (pilot: ≈ 1) |
| real centred/NC/ASIS steps over the stand-in | 0.3 (pilot) |
| accumulators, status, `T_pad = 320` padding (+7% on `T`-linear kernels) | 0.4 |
| **total** | **≈ 5.6** |

**Pilot amendment (WP3, recorded at the Phase 2 closeout, 2026-10-23).** The WP3 pilot measured the exact
correction at 0.7–1.2 ms of the full sweep (1.1 ms in isolation) instead of the 0.4 ms budgeted above, and a
total of 6.3–6.7 ms per sweep under load 6–10 on the M1 Pro (09_pilot.md decision 2). The table keeps the
original budget for the record. The measured BENCH-P2 sweep on the reference laptop is 3.90 ms (minimum of
3 × 1000 sweeps, Apple M5 Max, `tests/bench/baseline.json`), inside the budget.

11,000 sweeps ≈ 62 s, plus compile 1.5–2.8 s (0.37 s cached) and 2–3 s of host work: **≈ 67 s on one device**
(margin ≈ 1.8×); ≈ 3.1 ms per sweep and ≈ 38 s with 4 devices. Other compositions: Linear × Constant (eigen)
≈ 4 s; reduced form (CCCM) estimated +1–1.5 ms per sweep for the sequential `j` loop (to measure); SV(t) adds the
ν grid, `T·G·n·C ≈ 0.9·10⁶` `log1p` per sweep (≈ 1–5 ms; up to 2× the SV sweep; the pilot decides, §12 risk R5);
SV(outliers) adds `K·T` logits per equation, about one indicator step.

### 10.4 Levers, in order (none changes the target distribution)

1. Packed Khatri–Rao Gram (adopted, D-25): 8.1 → 4.6 ms.
2. Chains on CPU devices (`set_cpu_devices`): 4.6 → 2.5 ms; opt-in because it must run before JAX initialises.
3. Constant fast paths (adopted, D-26): ≈ 12× over Cholesky for Linear × Constant.
4. Inverse-CDF categorical (adopted, D-4): 3.5–4.4× on indicators, 10× fewer uniforms.
5. Two-solve precision draw with Jacobi scaling (adopted, D-14).
6. `logsumexp` reuse in the exact correction (adopted).
7. No recompiles (D-23) plus the persistent compilation cache (1.5–2.8 s → 0.37 s).
8. State donation; asynchronous host copy of chunk `c` while chunk `c+1` runs.
9. Pilot-measured options (WP3): `philox4x32` (≈4× faster uniforms; would become a spec field, so not a Phase 2
   default), SYRK Gram, scan `unroll`, two static equation buckets (padding 1.33×/1.22× → 1.15×/1.10×, pilot §10;
   not adopted, pilot decision 11), ν-sampler alternatives.
10. Not pursued: associative-scan SV (slower on CPU), `tridiagonal_solve` perturbation (slower), CG samplers
    (inexact), float32 (R4).

No lever may change a default prior, the mixture, the step order or float64 storage. A target miss is escalated
(Q2), never fixed by flipping `exact`.

### 10.5 Deterministic size rules (resolved in `resolve.py`, recorded in attrs)

- `gram="auto"` → `"packed"` if `T_pad·P·8 ≤ 256 MiB`, else `"packed_tiled"` with the largest tile (multiple of
  512 columns) such that `T_pad·tile·8 ≤ 64 MiB`. Examples: `n = 20, p = 4`: 12.9 MB, packed; `n = 50, p = 4`:
  80 MB, packed; `n = 30, p = 13, T = 600`: 422 MB, tiled. `"batched"` only on explicit request (oracle).
  These are shape rules, not machine-memory rules, so the same `(spec, data)` always resolves the same way (E14).
- `cccm_precompute` ⇔ `n·k²·8·C ≤ 512 MiB`.
- Storage plan: §5.3 (`memory_budget_gb` only decides whether to raise, never what is computed).

### 10.6 Measurement and pilot

- **WP3 pilot (stage 0)**: clean-room timing prototypes in `benchmarks/pilot/`, run on an idle machine: the real
  SV-parameter block and exact-mode cost; the joint `T = 300` exact-`h` acceptance (N01 §14 item 5, UNVERIFIED);
  ν-grid cost; philox vs threefry; chunk size; tile threshold; scan unroll; two equation buckets. Results and the
  defaults they freeze go to `docs/phase2/09_pilot.md`.
- **WP24 benchmark**: `benchmarks/phase2_perf.py` (AOT `jit(...).lower(...).compile()` to separate compile time,
  `block_until_ready`, minimum of 3 repeats, per-kernel microbenchmarks with `compiled.cost_analysis()` and
  `memory_analysis()`); JSON baseline with machine fingerprint, load average and versions in
  `tests/bench/baseline.json`; pre-release gate: +20% time or −30% ESS/s against the baseline fails.

---

## 11. API

### 11.1 Composition and engine resolution (`api.py`; D-19, D-21; fixes E1, E10, E18, E20)

```python
import rai_bvar as rb
rb.set_cpu_devices(4)                                     # optional; before anything touches JAX
model = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota()),     # form="auto" → "structural" because vol is SV (D3)
                vol=rb.SV(outliers=True),                 # SV(), SV(tails="t"), SV(rw=True), Constant("ig"), …
                lags=4,
                mcmc=rb.MCMCSettings(chains=4, draws=2000, burn=1000))
model.engine                                              # "mcmc"
fit = model.fit(data)                                     # MCMCFit
fc = fit.forecast(horizon=8, draws=1000, seed=1)          # ForecastResult, same contract as the analytic engine
irf = fit.irf(horizon=20, shock="sd")
res = rb.ordering_sensitivity(model, data, n_perm=10, mode="forecast")
```

`BVAR.__post_init__` resolves `auto` values first, then applies the engine rules (first match wins), then the
MCMC-only rules; all errors are raised at construction (SPEC §4.5). There is **no** user-settable engine field.

1. `Minnesota.form == "auto"` → `"conjugate"` iff `vol` is a `Constant` with prior `"auto"` or `"iw"`, else
   `"structural"`.
2. `Constant.prior == "auto"` → `"iw"` iff the resolved form is `"conjugate"`, else `"ig"`.
3. The resolved components are stored (`object.__setattr__`), so `BVAR()` keeps its Phase 1 meaning, spec and hash.

| # | Resolved composition | Engine |
| :--- | :--- | :--- |
| 1 | `conjugate` × `Constant("iw")` | `"analytic"` (Phase 1, unchanged) |
| 2 | `conjugate` × anything else | `ValueError`: "the conjugate Minnesota prior requires Constant(prior='iw'); use form='structural' (the MCMC default) or 'reduced'" |
| 3 | `structural` × `Constant("ig")` | `"mcmc"`, Mode I |
| 4 | `structural` × `Constant("iw")` | `"mcmc"`, Mode II (the NIW twin at `cross = contemp = 1`); `axes["prior"]` gains `scaling=conjugate` |
| 5 | `structural` × `Constant("iw_equivalent")` | `"mcmc"`, Mode I δ prior + Chan `(a, λ)` prior |
| 6 | `structural` × `SV(...)` | `"mcmc"`, Mode I |
| 7 | `reduced` × `Constant("ig")` or `SV(...)` | `"mcmc"`, CCCM |
| 8 | `reduced` × `Constant("iw_equivalent")` | `"mcmc"`, CCCM + Chan `(a, λ)` prior (order-invariant) |
| 9 | `reduced` × `Constant("iw")` | `ValueError`: "Constant('iw') is the conjugate NIW prior; with form='reduced' use 'iw_equivalent' (Chan 2022) or 'ig'" |

MCMC-only rules: `outliers="lenza-primiceri"` → `NotImplementedError` (Phase 3), with `SV(outliers=True)` also
"double treatment"; `soc`/`sur` not `None` → `NotImplementedError` (Phase 3 or stretch WP25; **superseded**: WP25
delivered SOC/SUR on the MCMC engine on 2026-10-22, D-21, so `api.py` no longer rejects them); `hierarchy="mh"` →
`NotImplementedError` (Phase 3); `form="structural"`, `cross ≠ 1`, `hierarchy="eb"`, `"psi"` in the free set →
`ValueError`; `contemp ≠ 1` with `form="conjugate"` → `ValueError` (Minnesota validation, like `cross`);
`config.validate_composition` (JAX-free, §3.8). `mcmc` given for an analytic composition is normalised to `None`
(documented; the spec stays unchanged).

### 11.2 Settings (`rai_bvar/settings.py`, pure Python; fixes E14, E15)

```python
@dataclass(frozen=True)
class MCMCSettings:                 # part of the spec: results depend on every field
    chains: int = 4
    draws: int | None = None        # kept draws per chain; None → 10,000 with SV, else 2,000 (D-35)
    burn: int = 1000                # exact number of burn-in sweeps (never rounded)
    thin: int = 1
    warm_burn: int | None = None    # None → max(250, burn // 4); warm starts only
    seed: int = 0
    store: Literal["params", "paths"] = "params"
    store_max: int = 2500           # stored parameter draws per chain (monitors keep every kept draw)
    path_thin: int = 10
    t_bucket: int = 32
    gram: Literal["auto", "packed", "packed_tiled", "batched"] = "auto"   # resolved by the §10.5 shape rule
    const_path: Literal["auto", "cholesky"] = "auto"
    rhat_max: float = 1.01
    ess_min: float = 400.0

@dataclass(frozen=True)
class Runtime:                      # NOT in the spec: knobs with an invariance test (§5.6)
    devices: int | Literal["auto"] = 1   # Stage 2 amendment: sharding is opt-in (§5.2)
    chunk: int = 250
    memory_budget_gb: float = 2.0   # only decides whether the storage plan raises
    storage_dir: str | None = None  # memmap host store
    replay: bool = True             # host copy of each chunk's start state for failure replay
    progress: bool = False

@dataclass(frozen=True)
class FitPlan:                      # in-loop forecasting (§5.8)
    horizon: int
    draws: int = 1000
    seed: int = 0
```

`BVAR.mcmc: MCMCSettings | None = None`; it resolves to `MCMCSettings()` for MCMC compositions, with an unset `draws`
resolved per composition (`MCMCSettings.with_default_draws`, D-35: 10,000 kept draws per chain with `SV`, 2,000
with `Constant`), and is stored resolved, so the spec contains it.

### 11.3 `fit` and `MCMCFit`

```python
BVAR.fit(data: VARData, *, seed: int | None = None, init: FinalState | MCMCFit | None = None,
         runtime: Runtime | None = None, plan: FitPlan | None = None) -> AnalyticFit | MCMCFit
```

- Analytic compositions: `seed` keeps its Phase 1 meaning (MH seed); `init`, `runtime`, `plan` must be `None`.
- MCMC: `seed` overrides `MCMCSettings.seed`; `init` is a warm start (§5.5); `runtime` defaults to `Runtime()`.
- The `Model` protocol keeps `fit(data)`. `BVAR` declares `uses_fit_seed` and `supports_warm_start` (both
  `engine == "mcmc"`) and `requires_in_loop_forecast = False`; the backtest passes keywords only to models that
  declare them.
- Engine-level entry for validation and parity: `engines.mcmc.fit_mcmc(static: StaticConfig, data: EngineData,
  prior: PriorArrays, settings: MCMCSettings, runtime: Runtime, *, seed: int, salt: int, meta: FitMeta,
  init: ChainState | None = None) -> MCMCFit`; the `BVAR` path calls `resolve.resolve(...)` and then `fit_mcmc`.

```python
class MCMCFit:
    engine = "mcmc"
    model: BVAR; data: VARData; static: StaticConfig; plan: Plan
    hyper: dict[str, Any]                 # λ, α, cross, κ_a, ψ, own-lag means, offsets, SV level centres, eb_note
    psi: np.ndarray
    warnings: list[str]
    converged: bool                       # §8.2
    posterior: xr.DataTree                # §5.3
    diagnostics: pd.DataFrame             # §8.4
    final_state: FinalState               # §5.5
    timing: dict[str, float]              # compile, burn, sample, host seconds; ms per sweep
    ordering: tuple[str, ...]
    def forecast(self, horizon: int, draws: int = 1000, seed: SeedLike = None) -> ForecastResult
    def irf(self, horizon: int = 20, draws: int = 1000, seed: SeedLike = None, shock: str = "sd",
            date: pd.Period | str | None = None, which: str = "persistent") -> IRFResult
    def draw_parameters(self, size: int, rng: SeedLike = None, date=None, which: str = "persistent"
                        ) -> tuple[np.ndarray, np.ndarray]      # coefs (S, k, n), chol (S, n, n)
    def reduced_form_draws(self) -> xr.DataArray                 # B per stored draw, computed lazily
    def summary(self) -> pd.DataFrame
```

`MCMCFit.forecast/irf/draw_parameters` delegate to `engines.mcmc.forecast.{forecast, irf, draw_parameters}(fit,
…)` (WP17), so `results.py` (WP14) and `forecast.py` (WP17) have disjoint owners.

### 11.4 Labels, axes and grid harmonisation (fixes E22)

- `SV.name`: `"SV"`, `"SV(rw)"`, `"SV(outliers)"`, `"SV(t)"`, `"SV(outliers,t)"` (and `rw` combinations);
  `SV.label` adds non-default settings (short hashes for nested priors, as `Minnesota.label` does).
- `Constant.name`: `"Constant"` for `"iw"` and `"ig"` (Phase 1 unchanged), `"Constant(iw_equivalent)"`.
- `BVAR.axes["prior"] = prior.label` plus `", scaling=conjugate"` for Mode II; `axes["vol"] = vol.name`; Phase 1
  names and axes are unchanged (golden test).
- `BVAR.harmonise_grid(combos: list[dict[str, Any]]) -> list[dict[str, Any]]` (classmethod): if any combination
  resolves to MCMC because of its volatility (a vol that is not a `Constant` with prior `"auto"`/`"iw"`), every
  combination whose mean prior has `form="auto"` gets `form="structural"`, and `Constant(prior="auto")` becomes
  `"ig"`. Explicit settings are never changed; harmonised arms are listed in an info note.
- `evaluation.compare.grid(factory, **axes)` calls `factory.harmonise_grid(combos)` when the factory defines it
  (five lines). The SPEC §9.5 grid `mean=[Linear(prior=Minnesota())]`, `vol=[Constant(), SV(),
  SV(outliers=True)]` then runs structural Mode I in all three arms, so the vol verdict compares one axis. A grid
  with only `Constant()` arms stays on the analytic engine.

### 11.5 `spec()` and `from_spec` (fixes E13)

- `BVAR.spec()`: `{"class": "BVAR", "mean": {"prior": {...}}, "vol": {...}, "lags", "deterministic", "outliers"}`,
  plus `"mcmc": asdict(settings)` only when `engine == "mcmc"`.
- Phase 1 spec hashes are unchanged: no type tags are added to Phase 1 components, and `Minnesota` fields new in
  Phase 2 (`contemp`) are omitted when equal to their defaults. A golden test pins the hashes of the Phase 1
  contract-registry models.
- The SV dict carries `"class": "SV"`; nested prior objects serialise as `{"class": "Normal", "mean": …, "var": …}`
  etc. `from_spec` dispatches the vol dict on `"class"` (absent → `Constant`), rebuilds nested SV priors with
  `priors.sv.prior_from_spec`, `MCMCSettings` from `"mcmc"`, and defaults missing Phase 2 fields. A hypothesis
  property test checks `from_spec(m.spec()) == m` and stable `spec_hash` over random valid compositions.

### 11.6 Backtest integration (`evaluation/backtest.py`, WP23)

- **Fit seeds.** For models with `uses_fit_seed`, `_run_origin` derives
  `fit_seed = int(np.random.default_rng([seed, crc32(origin), 0x4D434D43]).integers(2**31))` and calls
  `model.fit(sample, seed=fit_seed)`; other models are called exactly as in Phase 1. The forecast generator is
  unchanged. The seed is already in the cache key, and the MCMC settings are in the spec.
- **Padding.** `T_pad = bucket(T_eff)` per origin; no padding to a backtest maximum, because results must depend
  only on `(spec, data, seed)` for the cache key to be valid (E14). An expanding monthly backtest compiles about
  once per 32 origins (cheap with the persistent cache).
- **Processes.** With an MCMC model and `n_jobs > 1`: a `UserWarning` (XLA is already multi-threaded; no verified
  single-thread recipe, N07 §6), and the pool uses `multiprocessing.get_context("spawn")` (forking after JAX
  initialises is unsafe).
- **Warm starts** (opt-in `warm_start=False` default): origins of an MCMC model run sequentially in one process;
  each fit gets `init=<previous final_state>`; final states are cached next to the scores (`<key>.state.npz`) so
  a resumed backtest continues warm; the cache key gains `"init": "warm:<previous origin key>"`.
- `estimate_cost` is unchanged (it times one origin) apart from forwarding `lpd_method`.
- **LPD estimator** (D-34): `backtest(…, lpd_method="conditioned")` validates the value up front and threads the
  plain string (picklable for the `spawn` pool) through `_run_origin`, `_run_warm` and `_fit_and_score` to
  `score_forecast` (§7.3). The default leaves scores, layout and cache keys exactly as before: `_cache_key` adds
  `"lpd": "<method>-v1"` only for another method, for every model (the estimator changes the scores, not the fit;
  precedent: `"seeding"`), so the two estimators never share cached scores, and a warm key carries both tags.
  `_fit_and_score` receives `lpd_method=` only when it is not the default, so the default call is the WP23 one;
  it re-adds `attrs["lpd_method"]` after replacing the attrs, so cached scores record their estimator. The
  integrated estimator at `h > 1` needs MCMC forecasts: an analytic or benchmark model raises `ValueError` at its
  first origin (no fallback), after that origin's fit; `_fit_and_score` re-raises any scoring `ValueError` with
  `"<model name> at origin <origin>: "` in front (the cause chained; a plain string, so it pickles from a worker),
  because `compare` runs its backtests one after another and the failing model need not be the last one fitted.
  No up-front check: whether a forecast carries `lag_coefs` is known only after fitting in general (wrappers such as
  the CR `MomentsProbe` have no engine attribute). `compare` forwards it to every backtest. Changing `lpd_method` on
  an existing cache refits every origin (and every warm chain), although fits and forecasts do not depend on it;
  `"integrated"` stores the conditioned estimator too, so one run gives both.

---

## 12. Risks and open questions

### 12.1 Risks

| # | Risk | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| R1 | The ASIS NC `w`-correction is derived, not published (N01 §10.3) | a wrong exact mode would make strict tests meaningless | toy quadrature (A6), MH-ratio identity (U), G-full with `exact=True`, mutant M7 |
| R2 | Exact-mode joint-path acceptance at `T = 300` and mixing from crude starts unverified (N01 §14 item 5) | slow mixing, `converged = False` | WP3 pilot; acceptance and longest-rejection-run warnings; tiny relative offset; residual-scale initialisation |
| R3 | Performance evidence comes from a stand-in SV-parameter block on one machine | target missed | budget margin ≈ 1.8×; WP3 pilot with the real block; x86 benchmark; ESS/s gate; device lever |
| R4 | `set_cpu_devices` must run before JAX initialises | lost 1.8× for users who touched JAX first | `import rai_bvar` imports no JAX; actionable error; one device still meets the target |
| R5 | SV(t) ν-grid cost `T·G` per equation and sweep (N03 §4.3c) | SV(t) far above the SV budget | pilot; exact fallbacks with a decision record: uncollapsed `ν | τ` (mixes ≈40× worse) or a coarser grid declared as a prior choice |
| R6 | Mode I default and the data-centred SV level prior are design proposals (N05 §14 item 6) | forecast quality unknown | DGP tests; Clark–Ravazzolo report includes Mode I vs II and data-centred vs stochvol `μ` priors |
| R7 | bayesianVARs ambiguities (s.d. vs variance, `U` vs `U⁻¹`, intercept row, `h_0` mean) and stochvol's `|σ|` | false parity failures | probe fixtures first; exclude near-homoskedastic fixtures; compare `h_{1:T}` only |
| R8 | Masks everywhere | silent padding bugs | `T_eff` never a multiple of 32 in tests; mutant M13; padding-exactness tests; dtype and invariant asserts after every call |
| R9 | Ordering dependence of every triangular composition (R8) | ordering artefacts read as model differences | `ordering_sensitivity`; R8 warning for `n ≥ 4`; replication over all 24 orderings |
| R10 | Warm starts interact with caching and parallelism | stale or irreproducible backtests | opt-in; cache key carries init provenance; sequential origins; warm-vs-cold equality test |
| R11 | JAX version drift changes random streams (N07 §4.1) | golden bitwise tests break on upgrade | pin `jax`/`jaxlib`; regenerate goldens on upgrade; distributional tests are version-agnostic |
| R12 | CI cost is compile-dominated | slow PRs | persistent cache in CI (`min_compile_secs=0`), shared executables per configuration, G-small sizes of N06 §9.2 |
| R13 | Process-pool backtests oversubscribe cores | slow parallel backtests | warning, `spawn`; prefer `fit_axis` batching inside one process |
| R14 | Geweke power against M2/M7 unverified (N06 §12 item 6) | weak guard on step order and ASIS | structural enforcement (`s` local); quadrature toys; catch tiers recorded |
| R15 | Eigen path on near-singular `G̃` (many `g ≈ 0`) | rounding differences vs Cholesky | exact algebra; deterministic mean/covariance test at 1e-12; `const_path="cholesky"` escape hatch |
| R16 | Phase 3 CommonSV needs a JAX NIW draw (Phase 1 engine is NumPy); Phase 4 `bartz` may not vmap and has float32 leaves | rewrite pressure | `GlobalStep`, coupling channel, `Batching.MAP`, `float32_leaves`, in-loop forecast hook exist now |
| R17 | Host storage at 4 × 10,000 draws (the SV default since D-35) | memory pressure | `store_max`, latent moments, memmap, storage plan raises early |

### 12.2 Open questions for the orchestrator

**Decisions (2026-10-22):** every recommendation below is accepted as stated. Q2: the target is defined as in §10.1. Q12: the SPEC owner updates SPEC §4.2/§8 in the Phase 2 PR. (Q12 was executed at the Phase 2 closeout, 2026-10-23, in SPEC v2.2. Q14, added then, was resolved by the owner on 2026-10-24 as recommended: D-35.)

| ID | Question | Recommendation |
| :--- | :--- | :--- |
| Q1 | `SV(exact=True)` as the default? | Yes (D-1): it makes R6's Geweke/SBC strict; WP3 measures the cost; a target miss is escalated, not solved by flipping it |
| Q2 | Exact performance-target definition (kept vs total draws, compile included, devices) | §10.1 |
| Q3 | SV level prior: data-centred `N(log(ψ_ι/ι), 10)` vs stochvol `N(0, 100²)` | data-centred (scale-equivariant, R9, consistent with Mode I); stochvol priors for parity runs |
| Q4 | `Constant()` and `Minnesota()` defaulting to `"auto"` plus grid harmonisation | yes |
| Q5 | SOC/SUR on the MCMC engine in Phase 2 (stretch WP25) | Phase 3 unless WP25 is free — **resolved: delivered by WP25 (2026-10-22)** |
| Q6 | Warm starts as the backtest default | opt-in in Phase 2; revisit after WP23 shows warm/cold score equality and the speed gain |
| Q7 | Transitory Rao-Blackwellisation in the LPD as the default for outlier/t models | opt-in in Phase 2; contract unchanged |
| Q8 | Continuous ν prior (stochvol `ShiftedExponential`, RW-MH on `log(ν−2)` adapted in warm-up only) | Phase 3; Phase 2 parity uses a fixed ν |
| Q9 | Validation harness shipped under `src/rai_bvar/validation` | yes (Phases 3–5 and user-defined priors reuse it) |
| Q10 | Fit seed salted by the data fingerprint plus a per-origin backtest seed | yes |
| Q11 | `Constant` with transitory components user-facing | engine supports it (test vehicle); keep it engine-level in Phase 2 |
| Q12 | Replace the SPEC §4.2 interface sketch by §3 of this design; correct SPEC §8's "all without a loop" attribution (N07 §8.3) | yes, by the SPEC owner |
| Q13 | Pull `AsymmetricConjugate` (Chan 2022) into Phase 2 as an analytic composition, since `structural_nig.py` exists | user-facing prior and engine rule in Phase 3; the closed form ships in Phase 2 |
| Q14 | Default convergence policy for SV fits (WP3 pilot decision 6, escalated; added at the Phase 2 closeout, 2026-10-23) | **Resolved (owner, 2026-10-24): D-35.** Option 2 below as a composition-dependent default (SV compositions: 4 chains × 10,000 kept draws; `Constant`: 4 × 2,000, unchanged) plus a fit warning for SV fits below it; options 3–5 not adopted, the gate unchanged. Owners of the change: WP2 (settings), WP15 (monitors), WP18 |

**Q14 in detail** (the evidence as escalated; "the defaults" are the pre-D-35 ones). With the defaults (`MCMCSettings`: 4 chains × 2,000 kept draws, `rhat_max = 1.01`,
`ess_min = 400`) and the SV parameters `φ_i`, `σ_i` in the primary monitor set (D-33, §8.1), `converged = False`
is routine for SV fits, and `compare()` then returns `inconclusive` for any SV comparison (its gate needs
converged fits at ≥ 95% of origins, SPEC §9.5). Evidence:

- WP3 pilot (09_pilot.md decision 6, §3.3; M1 Pro, loaded): `φ` and `σ` have ESS ≈ 22–28 per 1,000 sweeps per
  chain, so 4 × 2,000 gives bulk ESS ≈ 200 < 400, and split R-hat reaches 1.03–1.06.
- BENCH-P2 at 4 × 10,000 (`tests/bench/baseline.json`, `d490730`): `converged = False` in 6 of 6 runs, with 4 of
  101 primary quantities missing R-hat < 1.01 (worst `σ` 1.016; the minimum `σ` ESS is 504).
- Clark–Ravazzolo (`benchmarks/reports/clark_ravazzolo_preregistration.md`): `SV(rw)` (identity ordering)
  converges at 72 of 106 origins at 4 × 5,000 and at 106 of 106 at 4 × 20,000, the budget the acceptance run
  uses.
- The README quick start (this closeout; FRED-QD 2026-09 small set, `n = 3`, `p = 4`, `Rolling(120)`, 11 origins
  2016Q1–2026Q1 every 4 quarters, structural `Constant("ig")` vs `SV(outliers=True)`; Apple M5 Max): at the
  default 4 × 2,000 the SV arm converges at 0 of 11 origins (4–8 of 16 primary quantities fail per origin; the
  worst is `σ`, the log-likelihood or `h_T`, R-hat 1.019–1.047, ESS 104–225), so both horizons are
  `inconclusive` although ΔLPD is +85.2 (MCSE 1.1) at h = 1 and +20.5 (1.5) at h = 4; at 4 × 10,000 it
  converges at 10 of 11 (h = 1 `inconclusive` at 91%, h = 4 `better`); at 4 × 20,000 at 11 of 11 (`better` at
  both).

Options (not exclusive):

1. **Keep the defaults and the gate, and document them.** SV fits need more draws (4 × 20,000 at these sizes);
   `verdict(min_converged=...)` and `MCMCSettings` already let a user relax or raise them. No code change, no hash
   change; default SV verdicts stay `inconclusive`.
2. **Raise the default draws for SV compositions** (a composition-dependent default, or a higher global one).
   `MCMCSettings` is part of the spec, so the spec hash and every backtest cache key of an MCMC model with
   default settings change, and default fits cost proportionally more; even 4 × 10,000 does not pass BENCH-P2's
   gate.
3. **Demote `φ_i` and `σ_i` to the secondary set** (amend D-33). The flag then gates on the log-likelihood,
   `B_own1`, `μ_i` and `h_T`; the README example shows the log-likelihood and `h_T` also failing at the default
   draws, so this alone may not suffice.
4. **Gate the SV parameters on bulk ESS only, or with a looser R-hat threshold** than the SPEC §10.6 default of
   1.01. A SPEC-level change of the convergence definition for those quantities.
5. **Sample until the gate passes**, up to a cap (an adaptive number of kept draws). The draws then depend on a
   stopping rule; it needs a design for reproducibility, storage and cache keys.

**Decision (owner, 2026-10-24): option 2, composition-dependent, with a warning (D-35).** `MCMCSettings.draws=None`
resolves to 10,000 kept draws per chain for SV compositions and to 2,000 for `Constant` volatility, so only SV
models whose settings leave `draws` unset (the default, or e.g. `MCMCSettings(seed=1)`) change spec hash and
backtest cache keys (an explicit `MCMCSettings(draws=2000)` reproduces the old ones); an SV fit below 10,000 kept draws per chain warns with this evidence and the remedy. The
README example's SV arm, which converged at 10 of 11 origins at 4 × 10,000, keeps its explicit 4 × 20,000 for the
verdict.

---

## 13. Implementation plan: dependency-ordered work packages

**Rules.**

1. Each work package (WP) owns its files exclusively and adds tests only under its own test paths (§1.4). WP1
   creates empty package files and stub modules on day 1; ownership of each stub passes to the WP listed in §1.2.
2. The contracts of WP1 (pytrees, protocols, keys, static config) and WP2 (user-facing specs and settings) are
   frozen at the end of stage 0. A later change goes through the WP1 owner and updates this document.
3. Kernel WPs (WP10–WP13) must pass tiers U and B for their blocks before merging. The composition-level Geweke/SBC
   gate (R6) is WP19, owned by someone who wrote no kernel; the reference implementations (WP9) likewise.
4. Clean room: no code translated from GPL sources; R packages are run as oracles only (WP8).
5. Every WP keeps the Phase 1 test suite green (golden spec hashes, contract, parity).

**Stages and critical path.** Stage 0: WP1, WP2, WP3 (parallel). Stage 1 (after WP1): WP4, WP5, WP6, WP7, WP8,
WP9. Stage 2 (after WP4, WP5): WP10–WP16. Stage 3: WP17, WP18. Stage 4: WP19–WP24. Stretch: WP25. Critical path:
**WP1 → WP4 → WP12 → WP13 → WP14 → WP17 → WP18 → WP19**. WP14 develops against toy kernels, so it runs in parallel
with WP10–WP13; WP9 runs in parallel with the kernels.

### Stage 0

#### WP1 — Core contracts (interfaces freeze)

- **Files owned**: `engines/mcmc/{__init__, _jax, protocols, state, config, keys, registry}.py`, `_devices.py`,
  `components/{structure, mean, coef, vol}/__init__.py`, all day-1 stubs, `pyproject.toml` (markers only),
  `tests/mcmc/unit/{test_keys, test_state, test_config, test_import_no_jax, test_devices}.py`.
- **Consumes**: nothing.
- **Provides**:
  ```python
  # _jax.py (imported by every JAX-using module; enables x64 once)
  def x64() -> contextlib.AbstractContextManager
  # keys.py
  class Stream(IntEnum); class Block(IntEnum); class Sub(IntEnum); class Ahead(IntEnum)   # values of §5.6
  def data_salt(fingerprint: str) -> int
  def fit_root(seed: int, salt: int) -> Key
  def chain_keys(root: Key, C: int) -> Key                      # shape (C,)
  def init_keys(seed: int, salt: int, C: int) -> Key            # shape (C,)
  def step_key(chain_key: Key, it: u32[], block: Block) -> Key
  def eq_key(step_key: Key, i: i32[]) -> Key
  def sub_key(eq_key: Key, sub: Sub) -> Key
  def forecast_key(fseed: int, c: Array, d: Array, kind: Ahead, i: Array) -> Key
  def validation_key(seed: int, tier: int, composition: int, rep: Array) -> Key
  # state.py: every class of §2.5–§2.7 and §2.9, plus
  class Empty
  def zero_accumulators(st: ChainState) -> ChainState
  def assert_float64(tree: PyTree, allow: frozenset[str] = frozenset()) -> None
  def tree_where(pred: bool_[], a: PyTree, b: PyTree) -> PyTree
  def encode_error(block: Block, eq: i32[]) -> i32[]
  def decode_error(code: int) -> tuple[Block, int]
  def check_state_invariants(st: ChainState, static: StaticConfig, T_eff: int) -> list[str]
  # config.py (JAX-free): Shapes, StructStatic, MeanStatic, CoefPriorStatic, VolStatic, StoreStatic, StaticConfig
  def bucket(T_eff: int, t_bucket: int) -> int
  def validate_composition(struct: StructStatic, mean: MeanStatic, coef: CoefPriorStatic, vol: VolStatic) -> None
  # protocols.py: CovarianceStructure, MeanBlock, CoefPrior, VolBlock, PersistentCore, TransitoryScale, GlobalStep,
  #               Batching, EqMap, OVERRIDABLE, class KernelBase: def impl(self, name: str) -> Callable
  class MCMCNumericalError(RuntimeError): block: str; equation: int; iteration: int; detail: dict
  # registry.py (JAX-free)
  KERNELS: dict[str, str]                                       # kind → "module:Class"
  def load(kind: str) -> type
  # _devices.py
  def set_cpu_devices(k: int) -> None
  def enable_compilation_cache(path: str | os.PathLike, min_compile_secs: float = 1.0) -> None
  ```
- **Tests**: `fold_in` golden values (enum stability), chain-prefix and `(c, it, block, i)` determinism; counters
  are uint32; every state class flattens/unflattens with the documented leaves; `Empty` has no leaves;
  `zero_accumulators` touches exactly the `acc_*`/`rb_*` leaves; error encode/decode; the
  `validate_composition` rule table (§3.8); subprocess tests: `import rai_bvar` and
  `import rai_bvar.engines.mcmc.config` load no JAX; `set_cpu_devices` works before backend initialisation and
  raises after; x64 enforced under `JAX_ENABLE_X64=0`.
- **Acceptance**: tests green; interfaces reviewed and signed off by the owners of WP4–WP18 in the PR; stubs exist
  for every file of §1.2.

#### WP2 — User-facing specs, settings and composition resolution (pure Python)

- **Files owned**: `components.py` → `components/{__init__, specs}.py`; `priors/minnesota.py`; `priors/sv.py`;
  `settings.py`; `api.py`; `evaluation/compare.py` (grid hook only); `models/base.py` (docstring only);
  `rai_bvar/__init__.py`; `tests/unit/{test_specs_phase2, test_phase1_hash_golden, test_api_resolution,
  test_grid_harmonise, test_spec_roundtrip_mcmc}.py`.
- **Consumes**: `config.validate_composition`, `StructStatic`, `MeanStatic`, `CoefPriorStatic`, `VolStatic` (WP1).
- **Provides**:
  ```python
  class Linear(prior: Minnesota = Minnesota())
  class Constant(prior: Literal["auto", "iw", "ig", "iw_equivalent"] = "auto")
  class SV(...)                                                  # fields of §6.3; name, label, to_spec(), from_spec()
  class Normal(mean, var); class Beta(a, b); class ScaledChi2(B); class InvGamma(shape, scale)
  class DiscreteUniform(low, high); class OutlierPrior(every_years=4.0, prior_years=10.0): def resolve(freq: str) -> Beta
  def prior_to_spec(obj) -> dict; def prior_from_spec(d: dict) -> object
  class Minnesota(..., form="auto", contemp: float = 1.0)       # Phase 1 fields unchanged
  class MCMCSettings; class Runtime; class FitPlan               # §11.2
  class BVAR(mean, vol, lags, deterministic, outliers, mcmc: MCMCSettings | None = None):
      engine: str; name: str; axes: dict[str, str]
      uses_fit_seed: bool; supports_warm_start: bool; requires_in_loop_forecast: bool
      def spec(self) -> dict; @classmethod def from_spec(cls, spec: dict) -> BVAR
      @classmethod def harmonise_grid(cls, combos: list[dict]) -> list[dict]
      def fit(self, data, *, seed=None, init=None, runtime=None, plan=None) -> AnalyticFit | MCMCFit
          # MCMC branch: lazy `from rai_bvar.engines.mcmc.resolve import fit_bvar` (WP18)
  def static_parts(model: BVAR) -> StaticParts   # (StructStatic, mean kind, coef_scaling, CoefPriorStatic, VolStatic); JAX-free
  ```
- **Tests**: resolution table of §11.1 including every error message; `BVAR()` and every Phase 1 registry model keep
  their spec hashes, names and axes (golden); `BVAR(vol=SV())` resolves to structural Mode I; Mode II label carries
  `scaling=conjugate`; `harmonise_grid` makes the SPEC §9.5 grid one-axis and leaves an all-`Constant()` grid
  analytic; hypothesis round trip `from_spec(m.spec()) == m` with stable hashes; settings validation; `import
  rai_bvar` still imports no JAX.
- **Acceptance**: all Phase 1 tests green unchanged; new tests green; `fit` of an MCMC composition raises a clear
  "engine not yet available" error until WP18 lands. **Amendment (recorded at the Phase 2 closeout, 2026-10-23):**
  two Phase 1 tests in `tests/unit/test_api.py` were changed in `ed502d4`, as intended by D-19 and E1 (recorded
  then only in the commit message). `test_non_conjugate_forms_need_the_mcmc_engine` (which expected
  `NotImplementedError`) became `test_non_conjugate_forms_resolve_to_the_mcmc_engine`: structural and reduced
  forms resolve to the MCMC engine, reduced × `Constant("iw")` raises `ValueError`, and `hierarchy="mh"` still
  raises `NotImplementedError`. `test_conjugate_form_requires_the_inverse_wishart_covariance` now builds the
  conjugate form explicitly, because `BVAR(vol=Constant("ig"))` resolves to the structural form under `auto`.
  No other Phase 2 commit changes a Phase 1 test file (`git log bce0db3..888f620` over the 35 files under
  `tests/` at `bce0db3`, the last Phase 1 commit, lists only `ed502d4`). `bce0db3` itself, which makes the
  test-side Beta-by-mode-and-sd oracle helper solve by bracketing instead of `fsolve`, is part of Phase 1.

#### WP3 — Pilot measurements (clean-room prototypes)

- **Files owned**: `benchmarks/pilot/*.py`, `docs/phase2/09_pilot.md`.
- **Consumes**: nothing (standalone scratch prototypes, never imported by the package).
- **Provides**: measured numbers and frozen defaults: exact-mode cost with the real SV-parameter block; joint
  `T = 300` exact-`h` acceptance for `(φ, σ) ∈ {(0.95, 0.2), (0.5, 0.05), (0.99, 0.1)}`; ν-grid cost; threefry vs
  philox uniforms; chunk size ∈ {50, 125, 250, 500}; packed-tile threshold; scan `unroll`; two equation buckets.
- **Tests**: none in CI (scripts run with `python -I`).
- **Acceptance**: `09_pilot.md` with machine fingerprint, load average, versions and one decision line per default;
  if projected BENCH-P2 exceeds 100 s on one device, Q1/Q2 are escalated before stage 2 freezes the defaults.

### Stage 1

#### WP4 — JAX numerics

- **Files owned**: `engines/mcmc/{linalg, random}.py`, `tests/mcmc/unit/{test_linalg, test_random}.py`.
- **Consumes**: WP1 (`StepInfo`, `EigenCache`, `FixedFactorCache`, `_jax`).
- **Provides** (pure, one problem; batched by `vmap`):
  ```python
  def weighted_gram(w: f64[T], *, F: f64[T, q] | None, FF: f64[T, P] | None,
                    mode: Literal["packed", "packed_tiled", "batched"], q: int, tile: int = 0) -> f64[q, q]
  def gram_from_packed(v: f64[P], q: int) -> f64[q, q]
  def precision_draw(K: f64[m, m], b: f64[m], z: f64[m]) -> tuple[f64[m], StepInfo]          # D-14
  def precision_moments(K: f64[m, m], b: f64[m]) -> tuple[f64[m], f64[m, m]]                 # mean, covariance (tests)
  def eigen_prepare(G: f64[q, q], d: f64[q], mean0: f64[q], fty: f64[q], cmask: f64[q]) -> EigenCache_i
  def eigen_draw(c: EigenCache_i, lam: f64[], z: f64[q]) -> tuple[f64[q], StepInfo]
  def fixed_factor_prepare(K_tilde: f64[q, q], rhs: f64[q]) -> FixedFactorCache_i
  def fixed_factor_draw(c: FixedFactorCache_i, lam: f64[], z: f64[q]) -> f64[q]
  def tridiag_precision_sample(diag: f64[m], off: f64[m−1], b: f64[m], z: f64[m]) -> tuple[f64[m], f64[], f64[]]
      # (x, log|Q|, min pivot); two lax.scans (N07 §8.3)
  def stable_ar1_regression(x: f64[T], z: f64[T], m: f64[T], eps1: float, eps2: float) -> tuple[f64[2], f64[2, 2]]
      # centred cross-products: (b_T, chol(B_T)) (N02 §6.1)
  def solve_unit_lower(A: f64[n, n], B: Array) -> Array
  # random.py
  def uniform_tiny(key: Key, shape) -> Array
  def categorical_inverse_cdf(u: f64[...], logw: f64[..., J]) -> i32[...]
  def categorical(key: Key, logw: f64[..., J]) -> i32[...]
  def log_inv_gamma(key: Key, shape: Array, scale: Array) -> Array          # log(scale) − loggamma(shape)
  def log_beta(key: Key, a: Array, b: Array) -> tuple[Array, Array]          # (log p, log(1 − p))
  def truncated_std_normal(u: Array, lo: Array, hi: Array) -> Array          # N07 §9.3
  ```
- **Tests**: dense oracles with injected noise (1e-10); padded vs unpadded Cholesky draw with `z[:q_i]` (1e-12);
  packed and tiled Gram vs dense (1e-13 relative); eigen and fixed-factor mean/covariance equal the Cholesky path
  (1e-12) plus KS on 10⁴ draws; tridiagonal sampler vs dense `Q` and `slogdet` (1e-12) with decoupled padded rows;
  stable 2×2 vs exact `Fraction`s (N02 T7); categorical χ² with log weights shifted by +700 (N07 §9.2); truncated
  normal far tails; `log_inv_gamma` and `log_beta` at shape 1e-3; Jacobi scaling with 1e8 precision disparities;
  `vmap` lanes equal unbatched calls (1e-14).
- **Acceptance**: tests green; a non-gating timing table at the reference shapes is printed by the tests.

#### WP5 — Data and design (JAX)

- **Files owned**: `engines/mcmc/data.py`, `tests/mcmc/unit/test_data.py`.
- **Consumes**: WP1 (`EngineData`, `FitCache`, `StaticConfig`); `data/design.py` (read-only oracle).
- **Provides**:
  ```python
  def packed_index(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]     # iu, ju, pos (q, q) int32
  def packed_products(F: f64[T, q]) -> f64[T, P]
  def build_engine_data(y: f64[T_pad, n], presample: f64[p, n], T_eff: i32[], *, n_det: int, t0: float) -> EngineData
  def engine_data_from_vardata(data: VARData, lags: int, deterministic: str, t_bucket: int
                               ) -> tuple[EngineData, pd.PeriodIndex]
  def regressors_from_window(window: f64[p, n], t_next: f64[], n_det: int) -> f64[k]
  def structural_design(data: EngineData, Mc: f64[k, n−1]) -> f64[T_pad, q]
  def base_cache(data: EngineData, Mc: f64[k, n−1] | None, static: StaticConfig) -> FitCache   # F, FF, XX, G_const, FtY
  ```
- **Tests**: equality with `lagged_design` and `regressors_from_history` (1e-15), including trend continuation
  (`t0 = lags + 1`, `t_last = len(values)`); padded rows exactly zero; packed products vs explicit products; `XM_c`
  identity; `build_engine_data` under `jit` with traced `T_eff` (no recompile within a bucket).
- **Acceptance**: tests green.

#### WP6 — Host prior builders (NumPy)

- **Files owned**: `priors/{scales, structural, reduced, sv_arrays}.py`, `tests/priors/*`.
- **Consumes**: WP1 prior pytree classes; WP2 specs; `engines.analytic._ar_residual_variances` and
  `ConjugateProblem` (read-only, the latter in tests).
- **Provides**:
  ```python
  @dataclass(frozen=True)
  class WindowScales: psi: np.ndarray; own_means: np.ndarray; freq: str; notes: tuple[str, ...]
  def window_scales(data: VARData, prior: Minnesota, lags: int, deterministic: str,
                    psi: np.ndarray | None = None) -> WindowScales
  def contemp_arrays(scales: WindowScales, prior: Minnesota, n: int, *, scaling: Scaling) -> ContempPriorArrays  # scaling = StructStatic.contemp_scaling (Mode I vs Mode II/iw_equivalent); WP16 and WP18 pass it
  def structural_arrays(scales, prior, hyper: Mapping[str, float], *, n: int, lags: int, n_det: int,
                        mode: Literal["I", "II"]) -> tuple[CoefPriorArrays, SharedPrior]       # δ prior, M_c
  def constant_prior(scales: WindowScales, n: int) -> ConstantPrior                             # IG(1 + ι/2, ψ/2)
  def implied_reduced_form_moments(coef, contemp, shared, vol_mean: np.ndarray, *, mode: str, draws: int = 0,
                                   rng: np.random.Generator | None = None) -> ImpliedMoments
  def reduced_arrays(scales, prior, hyper, *, n: int, lags: int, n_det: int) -> tuple[CoefPriorArrays, SharedPrior]
  def explicit_reduced(mean0: np.ndarray, var0: np.ndarray, a_mean0: np.ndarray, a_var0: np.ndarray
                       ) -> tuple[CoefPriorArrays, ContempPriorArrays]
  def sv_prior_arrays(sv: SV, scales: WindowScales) -> SVPrior
  def transitory_arrays(sv: SV, scales: WindowScales, nu_grid: tuple[int, ...]) -> TransitoryPrior | Empty
  ```
- **Tests**: `Ω` equals `ConjugateProblem.omega` at `cross = 1`; Mode I moments by Monte Carlo at
  `κ_c = κ_a = 1` (`Var(B)/NIW ∈ [0.98, 1.02]`, `E[Σ_ii]/ψ_i ∈ [0.95, 1.05]`); leak-ratio closed form vs Monte Carlo
  (N05 §4.3); Mode II prior equals NIW (KS on `Σ` functionals); IW-equivalent `(a, λ)` gives `Σ ~ IW(diag ψ, n+2)`
  (KS on `log|Σ|`, `Σ_ii`, correlations; N08 §B5 item 4); reduced `V_j` formulas; SV arrays (data-centred `μ`,
  `log_c = log(1e-12 ψ)`), outlier `Beta` per frequency and the unknown-frequency error; R9 (arrays computed from
  `data.upto(origin)` ignore later data).
- **Acceptance**: tests green.

#### WP7 — Validation statistics (model-agnostic)

- **Files owned**: `validation/{__init__, geweke, sbc, uniformity, sequential, identities}.py`,
  `tests/validation/{test_uniformity, test_geweke_selftest, test_sbc_selftest, test_sequential, test_identities_toy}.py`.
- **Consumes**: nothing from the engine (JAX for vectorised drivers).
- **Provides**:
  ```python
  class JointModel(Protocol): ...                                                 # §9.2
  @dataclass(frozen=True) class GewekeConfig: C: int; M_c: int; M1: int; alpha: float; seed: int; batch_means: bool = True
  @dataclass(frozen=True) class GewekeResult: names; z; p; t_crit: float; passed: bool; rne; failing: tuple[str, ...]
  def geweke(model: JointModel, cfg: GewekeConfig) -> GewekeResult
  @dataclass(frozen=True) class SBCConfig: N: int = 1000; L: int = 99; burn: int = 300; pilot: int = 1000; t_max: int = 50
                                            alpha: float = 1e-2; seed: int = 0
  def sbc(model: JointModel, quantities: Callable, names: tuple[str, ...], cfg: SBCConfig) -> SBCResult
  def block_sbc(joint_sample: Callable, conditional_draws: Callable, stat: Callable, N: int = 200, L: int = 99,
                seed: int = 0) -> SBCResult
  def rank_with_ties(truth: f64[Q], draws: f64[L, Q], u: f64[Q]) -> i32[Q]
  def gamma_statistic(ranks: np.ndarray, L: int) -> float
  def gamma_pvalue(gamma: float, N: int, L: int) -> float                         # exact Säilynoja recursion
  def gandy_scott(run: Callable[[int], float], alpha: float, k: int = 3, delta: int = 4) -> SequentialResult
  def density_ratio_error(log_q: Callable, log_joint: Callable, x0: PyTree, x1: PyTree) -> float
  def mh_ratio_error(log_alpha: Callable, log_target: Callable, log_prop: Callable, x: PyTree, x_prop: PyTree) -> float
  ```
- **Tests**: γ exact recursion vs 10⁵-draw simulation (N06 §0 table); Geweke and SBC on a conjugate scalar-normal
  toy hold the nominal level over 20 seeds; a deliberately wrong toy fails; Gandy–Scott decisions; identities on a
  Gaussian toy.
- **Acceptance**: tests green.

#### WP8 — R fixtures (offline)

- **Files owned**: `tools/make_r_fixtures.R` (new stochvol and bayesianVARs sections), `tools/make_sv_parity_data.py`
  (new), `tests/parity/fixtures/{stochvol_*, bvars_*, probe_*}.nc`, `tests/parity/fixtures/data/*.csv`,
  `tests/parity/test_fixture_schema.py`.
- **Consumes**: the parameter sets of §9.9 (documentation only).
- **Provides**: committed NetCDF fixtures with attrs (R and package versions, exact call strings, seeds) and NumPy
  generated CSV data; probe results P1–P5 stored as fixture attrs.
- **Tests**: fixture schema (files, attrs, shapes, no zeros in SV data).
- **Acceptance**: fixtures committed; P1–P5 answered, or recorded as unresolved with the evidence.

#### WP9 — Validation reference implementations (owner writes no kernel)

- **Files owned**: `validation/{reference, simulate, log_joint, testfns, joint}.py`,
  `tests/validation/{test_reference_moments, test_simulate, test_log_joint, test_mixture_transcription}.py`.
- **Consumes**: WP1 state/config/keys; WP5 `build_engine_data`; WP7 `JointModel`; the production `make_sweep` and
  `prepare` (WP14) only through `JointSpec.step`.
- **Provides**: `register_reference`, `Reference` and references for every Phase 2 kind (§9.1); `simulate_obs`,
  `log_prior`, `loglik`, `log_joint`, `test_functions`, `FixedDesign`, `JointSpec`, `joint_spec`, `COMPOSITIONS`,
  `BLOCK_CONFIGS` (§9.2).
- **Tests**: prior-sampler moments vs closed forms (IG, Beta, Normal; D3 `E[γ | a] = Mã`; stationary `h`
  moments; outlier frequencies; standardized `E τ = 1`); `simulate_obs` vs a NumPy loop with the same noise
  (1e-12); `log_joint` vs `scipy.stats` sums (1e-10); second transcription of the mixture tables vs N01 §5.3 and
  vs `components/vol/mixture.py` once WP12 lands; finite-variance flags.
- **Acceptance**: tests green; `joint_spec(name)` builds for every name in `COMPOSITIONS` and `BLOCK_CONFIGS`.

### Stage 2

#### WP10 — Recursive structure, structural Linear, ZeroMean, FixedDiagonal

- **Files owned**: `components/structure/recursive.py`, `components/mean/{linear, zero}.py`,
  `components/coef/fixed.py`, `tests/mcmc/blocks/{test_recursive, test_linear_structural, test_linear_fastpaths,
  test_zero_mean}.py`.
- **Consumes**: WP1 protocols and state; WP4 `weighted_gram`, `precision_draw`, `eigen_*`, `fixed_factor_*`,
  `solve_unit_lower`; WP5 `base_cache`; WP6 arrays (tests); WP9 references (block Geweke).
- **Provides**: `class Recursive(CovarianceStructure)` (§3.2, §3.7.1), `class LinearStructural(MeanBlock)` with the
  four paths (§3.7.2), `class ZeroMean(MeanBlock)`, `class FixedDiagonal(CoefPrior)`; overridable internals
  `structural_gamma`, `linear_conditional`, `contemp_conditional`.
- **Tests**: N05 §13.1 items 1–7 (precision draw, γ/δ invariance, masking, `n = 1` dense GLS, `B′ = A⁻¹Γ`,
  log-likelihood identity, scale equivariance); the four Linear paths agree (mean/covariance 1e-12, KS); padded =
  unpadded (1e-12); coupling values for Mode II and IW-equivalent vs hand formulas; contemporaneous sign check
  (N04 §9.4); density-ratio identities for the equation and contemporaneous steps against `validation.log_joint`;
  block Geweke `equation_only`, `contemp_only`.
- **Acceptance**: U and B tiers green for these blocks; weighted-path time at the reference shape within 25% of the
  WP3 pilot number.
- **Timing result (Phase 2 closeout, 2026-10-23)**: met under both readings. The gate is
  `tests/mcmc/blocks/test_linear_fastpaths.py::test_weighted_path_timing_reference_shape` (markers `bench` and
  `slow`, so it runs only in the manual bench job or by hand). It times our weighted path (packed Gram, Jacobi
  Cholesky, two solves and the residual; `n = 20`, `p = 4`, `T = 300 → T_pad = 320`, `q = 100`, 4 chains = 80
  lanes, float64) against the pilot's own one-bucket prototype (`pilot_common.mean_draw`) in the same process,
  interleaved, as the minimum over 10 rounds of 50 draws, and asserts `ours ≤ 1.25 × pilot`; it does not gate on
  the pilot's absolute 3.30 ms (09_pilot.md §10, measured on a loaded M1 Pro). Three runs on 888f620 (Apple M5
  Max, 18 cores, 64 GB, Darwin 25.6.0, `jax`/`jaxlib` 0.12.0.1, NumPy 2.5.4, one CPU device; 1-minute load average
  2.56–3.48 with no other Python or pytest work running), ms per mean draw at 80 lanes, ours vs the prototype:
  2.940 vs 2.939 (ratio 1.000), 3.079 vs 3.020 (1.019), 2.971 vs 2.944 (1.009). All three are also below the
  absolute 3.30 ms. No idle or x86 run.

#### WP11 — Reduced form (CCCM 2022)

- **Files owned**: `components/mean/linear_reduced.py`, `tests/mcmc/blocks/{test_cccm.py, swa_oracle.py}`.
- **Consumes**: WP1; WP4 (`weighted_gram`, `precision_draw`); WP5 (`XX`); WP10 (`Recursive.contemp_step`,
  integration tests only); WP9 (block Geweke).
- **Provides**: `cccm_core(...)` (§3.7.3), `cccm_conditional_j(Pi, a, inv_lam, X, Y, mean0, prec0, j: int) ->
  GaussianPrec`, `class CCCMStep(GlobalStep)`, `class ReducedEq(MeanBlock)`; overridable `cccm_core`.
- **Tests**: N04 §9.1 table (rtol 1e-10); 20 random problems vs the SUR oracle; kernel invariance (N04 §9.2:
  extract `M, c, N`, Lyapunov solve, `M` = block Gauss–Seidel); CCM-2019 regression guard (mutant differs);
  `A = I` decoupling; packed vs batched `K_j`; precompute vs in-loop (1e-12); SWA oracle vs CTA Gibbs
  `n ∈ {2, 3, 4}` (A7, nightly); block Geweke `cccm_only`.
- **Acceptance**: U and B green; A7 green nightly.

#### WP12 — SV core

- **Files owned**: `components/vol/{mixture, sv}.py`, `tests/mcmc/blocks/{test_mixture, test_sv_core,
  test_sv_toy_quadrature}.py`.
- **Consumes**: WP1; WP4 (`tridiag_precision_sample`, `stable_ar1_regression`, `categorical_inverse_cdf`,
  `log_inv_gamma`, `uniform_tiny`).
- **Provides**:
  ```python
  OCSN10: MixtureTable; KSC7: MixtureTable          # NumPy: p, m, v, logw, half_inv_v (effective means, D-3)
  def indicator_logits(r: f64[T], table) -> f64[T, J]
  def log_g(r: f64[T], table) -> f64[T]
  class SVCore(PersistentCore)                       # statics: rw, h0, sigma2_prior, mixture, exact, asis, zero_offset
  def awol_system(y_star, s, tmask, T_eff, mu, phi, sigma, prior_i, table) -> tuple[f64[m], f64[m−1], f64[m]]
  def w_weight(h: f64[T_pad+1], log_e2c: f64[T_pad], y_star, tmask, table, log_g_cur: f64[T_pad] | None) -> f64[]
  def h_log_accept(...) -> f64[]; def c_step(...) -> ...; def nc_regression(...) -> GaussianPrec
  def nc_log_accept(...) -> f64[]; def nc_fold(sigma: f64[], h_tilde: f64[T_pad+1]) -> tuple[f64[], f64[T_pad+1]]
  ```
- **Tests**: N01 §12.1–12.2 (constants, mixture mean −1.2704 ± 1e-4, pmf χ², Bayes consistency); N02 T1–T9 (AWOL
  vs dense, `Q̃ = σ²Q`, integrated likelihood); kernel invariance (N02 §13.2); prior recovery with all rows masked
  (N02 §13.3); toy quadrature A6 (N01 §12.6: step order, exact-mode invariance, ASIS correction, IG-σ² single
  accept); regression tests T13 (stale order diverges) and T16 (`|σ|` mutant); RW path vs dense; exact-`h`
  acceptance ≈ 0.96–0.98 at `T = 300`; one-block invariance for C and NC steps; block Geweke `aux_sv`,
  `gauss_sv_exact`.
- **Acceptance**: U and B green; A6 green.

#### WP13 — Constant core, transitory scales, VolBlock composition

- **Files owned**: `components/vol/{constant, transitory, robust}.py`, `tests/mcmc/blocks/{test_constant,
  test_transitory, test_robust}.py`.
- **Consumes**: WP1; WP4 (`categorical`, `log_beta`, `log_inv_gamma`); WP12 `SVCore` (a stub core until WP12
  lands).
- **Provides**: `class ConstantCore(PersistentCore)`; `class NoTransitory, OutlierScale, TScale,
  OutlierTScale(TransitoryScale)`; `class VolBlockImpl(VolBlock)`; overridable `constant_conditional`,
  `outlier_logits`, `p_conditional`, `nu_logits`, `tau_conditional`.
- **Tests**: N03 §7.1 tests 1–9 and 12 (categoricals vs brute force, Beta, IG, ν grid vs `scipy.stats.t`, quadrature
  posteriors of `p` and `ν`, PCG equivalence, collapsed vs uncollapsed ESS ratio > 3, limits, `draw_ahead` moments
  with `E λ = E[e^h](1 + 150p)`); Constant IG conditional with coupling; the internal order of `sample_eq` (spy
  through overrides); `log e²` masking (padded rows never `−inf`; zero-offset floor masking); `obs_logdensity_eq` vs
  Monte Carlo (N03 §5.3 numbers); block Geweke `transitory_known_lambda`, `constant_transitory`.
- **Acceptance**: U and B green.

#### WP14 — Engine assembly: compose, plan, sweep, runner, storage, warm starts, results

- **Files owned**: `engines/mcmc/{compose, plan, sweep, runner, storage, warmstart, results}.py`,
  `tests/mcmc/runner/*` (incl. `toys.py`).
- **Consumes**: WP1; WP4; WP5; kernels through `registry.load` (toys until WP10–WP13 land); WP15 `monitor_values`,
  `converged`, `diagnostics_table`; WP17 `forecast/irf/draw_parameters` (delegation only).
- **Provides**:
  ```python
  def compose(static: StaticConfig, *, overrides: Mapping[str, Any] | None = None) -> Program
  def build_plan(program: Program) -> Plan
  def check_invariants(plan: Plan, program: Program) -> None
  def make_sweep(program: Program) -> Callable   # (ckey, it: u32[], st, data, prior, cache) -> ChainState
  def make_runner(static: StaticConfig, *, fit_axis: bool = False,
                  overrides: Mapping[str, Any] | None = None) -> RunnerFns
  def run_chains(fns: RunnerFns, static: StaticConfig, data: EngineData, prior: PriorArrays,
                 settings: MCMCSettings, runtime: Runtime, *, seed: int, salt: int,
                 init_state: ChainState | None = None) -> RunOutput
  def plan_storage(static: StaticConfig, settings: MCMCSettings, runtime: Runtime) -> StoragePlan
  class HostStore: def append(self, out, stop: int) -> None; def to_datatree(self, meta: FitMeta) -> xr.DataTree
  @dataclass class FinalState: state: ChainState; periods: pd.PeriodIndex; static: StaticConfig; seed_meta: dict
  def remap(prev: FinalState, new_data: EngineData, new_periods: pd.PeriodIndex, static_new: StaticConfig,
            key: Key) -> ChainState
  @dataclass(frozen=True)
  class FitMeta: model: Any; vardata: VARData | None; periods: pd.PeriodIndex; variables: tuple[str, ...]
                 hyper: dict; notes: tuple[str, ...]; eb_converged: bool | None
  class MCMCFit                                                  # §11.3
  def fit_mcmc(static, data, prior, settings, runtime, *, seed: int, salt: int, meta: FitMeta,
               init: ChainState | None = None) -> MCMCFit
  ```
- **Tests**: §9.3 item 4 (bitwise invariance across chunk, burn piece size, thin, chain prefix, `vmap` vs loop,
  fused vs unfused; `shard_map` in a subprocess; as amended there: chain prefixes under `chains="vmap"` and
  `shard_map` are equal to rounding, not bitwise, and the bitwise cases are tested with the toy kernels, plus
  the production chain prefix with `chains="map"`); plan invariants for every composition (§4.8) with the write-set
  tree diff; zero recompiles across windows of one bucket; failure injected at sweep 7 raises with block, equation,
  iteration and replay diagnostics; accumulators zero after burn-in; `snap` equals the state after exactly
  `burn + draws·thin` sweeps; storage plan raises over budget; memmap store equals the in-memory store; warm-start
  remap (expanding, rolling, bucket change); toy posteriors recovered within MCSE; `fit_axis` equals looped fits
  (1e-10).
- **Acceptance**: green on toys; after WP10–WP13 merge, a smoke test runs every composition of §9.5 (2 chains × 50
  draws) without error.

#### WP15 — Diagnostics and monitors

- **Files owned**: `engines/mcmc/monitor.py`, `diagnostics.py` (additions only), `tests/mcmc/diagnostics/*`.
- **Consumes**: WP1 (state, static config).
- **Provides**:
  ```python
  # diagnostics.py (NumPy; existing functions untouched)
  def ess_tail(chains) -> float; def ess_mean(chains) -> float; def mcse_mean(chains) -> float
  def mcse_quantile(chains, q: float) -> float
  def rhat_many(x: np.ndarray) -> np.ndarray; def ess_bulk_many(x) -> np.ndarray; def ess_tail_many(x) -> np.ndarray
  # monitor.py
  @dataclass(frozen=True) class MonitorSpec: primary: tuple[str, ...]; secondary: tuple[str, ...]
  def monitor_spec(static: StaticConfig, variables: tuple[str, ...]) -> MonitorSpec
  def monitor_values(st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache,
                     program: Program) -> dict[str, Array]            # jnp; called by the runner's extract
  def converged(mon: Mapping[str, np.ndarray], spec: MonitorSpec, settings: MCMCSettings, *, error: bool,
                eb_converged: bool | None) -> tuple[bool, list[str]]
  def warning_catalogue(inputs: DiagnosticsInput) -> list[str]
  def diagnostics_table(mon: Mapping[str, np.ndarray], spec: MonitorSpec, acc: Mapping[str, np.ndarray]) -> pd.DataFrame
  ```
- **Tests**: AR(1) chains with known IAT; batched equals scalar functions; tail ESS and quantile MCSE against
  reference values (ArviZ when installed); flag cases (failure in primary → False; secondary only → True plus a
  warning; one chain → False; error → False; EB failure → False); warning texts.
- **Acceptance**: tests green; existing `tests/unit/test_diagnostics.py` unchanged and green.

#### WP16 — Structural NIG closed form (Chan 2022)

- **Files owned**: `engines/structural_nig.py`, `tests/analytic/test_structural_nig.py`.
- **Consumes**: WP6 builders (Mode II arrays), Phase 1 `AnalyticFit` (tests).
- **Provides**:
  ```python
  @dataclass(frozen=True)
  class NIGPosterior: theta_hat: np.ndarray; R: np.ndarray; nu_hat: float; S_hat: float; log_ml: float
  def nig_posterior(Z: np.ndarray, y: np.ndarray, mu0: np.ndarray, var0: np.ndarray, nu0: float, S0: float) -> NIGPosterior
  def structural_log_ml(data: VARData, prior: Minnesota, hyper: Mapping[str, float], lags: int, deterministic: str) -> float
  def draw_structural_nig(posts: Sequence[NIGPosterior], size: int, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]           # a (S, n, n−1), gamma (S, n, k), lam (S, n)
  def eb_structural(data, prior, lags, deterministic, start: Mapping[str, float]) -> EBResult
  ```
- **Tests**: A1 (log ML = Phase 1 to 1e-10, with SOC/SUR as dummy rows; EB λ equals the Phase 1 EB λ at
  `cross = 1` to 1e-6); posterior identities (N05 §13.2); iid-draw moments; residual-form `Ŝ` vs the naive form on a
  cancellation case.
- **Acceptance**: A1 green on every PR.

### Stage 3

#### WP17 — Forecasting, IRFs and LPD

- **Files owned**: `engines/mcmc/forecast.py`, `forecast/result.py`, `tests/mcmc/forecast/*`,
  `tests/contract/test_forecast_contract_mcmc.py`.
- **Consumes**: WP10–WP14 (kernels' `draw_ahead_eq`, `predict_*`, `mean_map`, `chol`; `MCMCFit` store); WP5
  `regressors_from_window`; `structural.irf.recursive_irf`.
- **Provides**:
  ```python
  def forecast_core(draw: ParamDraw, hist: f64[p, n], t_last: f64[], shocks: ForecastShocks, program: Program) -> ForecastOut
  def simulate_paths(params: ParamDraw, hist, t_last, eps: f64[H, n], log_lam_eff: f64[H, n], program: Program
                     ) -> tuple[Array, Array, Array]
  def make_forecast(static: StaticConfig, H: int, S_c: int) -> Callable
  def forecast(fit: MCMCFit, horizon: int, draws: int = 1000, seed: SeedLike = None) -> ForecastResult
  def irf(fit: MCMCFit, horizon: int = 20, draws: int = 1000, seed: SeedLike = None, shock: str = "sd",
          date=None, which: str = "persistent") -> IRFResult
  def draw_parameters(fit: MCMCFit, size: int, rng: SeedLike = None, date=None, which: str = "persistent"
                      ) -> tuple[np.ndarray, np.ndarray]
  ForecastResult.from_arrays(..., extra: Mapping[str, tuple[tuple[str, ...], np.ndarray]] | None = None)
  ForecastResult.log_pred_density(y, horizon, variables=None, *, return_mcse=False, rb_transitory=False)
  ```
- **Tests**: contract invariants (structure, simulation consistency) for every MCMC composition; structural
  recursion equals `B′x + A⁻¹(Λ^{½}ε)` with the same `ε` (1e-12); `chol` equals `np.linalg.cholesky(A⁻¹ΛA⁻ᵀ)`
  (1e-12); `draw_ahead` moments; forecast-key reproducibility (same `fseed` → identical paths, `o`, `τ`);
  factored LPD equals the `cond_chol` LPD (1e-12) for full and leading blocks; leading-block slicing equals
  `marginal_chol` (1e-12, analytic forecasts too); ESS-adjusted MCSE ≥ iid MCSE on autocorrelated draws; IRF via
  `recursive_irf` equals direct `Φ_h A⁻¹ Λ^{½}`; date errors of `draw_parameters`; in-loop forecast (FitPlan)
  equals post-hoc (1e-12); stretch: RB-transitory LPD vs Monte Carlo (N03 §5.3).
- **Acceptance**: Phase 1 contract and LPD tests unchanged and green; MCMC contract tests green.
- **Amendment (Phase 2 closeout, 2026-10-23): contract coverage.** The MCMC contract compositions
  (`tests/mcmc/forecast/mcmc_fits.py` `COMPOSITIONS`) lacked the two reduced-form Student-t compositions; they gain
  `"reduced-SV(t)"` (`("reduced", SV(tails="t"))`) and `"reduced-SV(outliers,t)"` (`("reduced", SV(outliers=True,
  tails="t"))`), from 13 to 15 entries (8 structural, 7 reduced). `tests/contract/test_forecast_contract_mcmc.py`
  now collects 150 tests: 20 per PR (the fast allow-list is unchanged: `structural-SV(outliers)`,
  `reduced-Constant(ig)`) and 130 in the slow tier, the two new rows slow like their reduced-SV neighbours. All
  150 pass on `888f620` plus this change. The analytic forecast now records `attrs["ordering"]` (SPEC R8, §7
  attrs), the same JSON list the MCMC engine writes, through a one-line change to `engines/analytic.py`
  (`tests/unit/test_forecast_ordering_attr.py`; the Phase 1 contract file stays untouched). Benchmark-model
  forecasts (`models/benchmarks.py`) still carry no `ordering`.

#### WP18 — MCMC fit assembly and EB plumbing

- **Files owned**: `engines/mcmc/{resolve, eb}.py`, `tests/unit/test_fit_mcmc_api.py`,
  `tests/analytic/test_eb_mcmc.py`.
- **Consumes**: WP2 (`BVAR`, `static_parts`, settings); WP5 (`engine_data_from_vardata`, `base_cache`); WP6
  builders; WP14 (`fit_mcmc`, `FitMeta`); WP15; WP16 (`eb_structural`); WP17; Phase 1 `AnalyticFit`.
- **Provides**:
  ```python
  @dataclass(frozen=True)
  class Resolved: static: StaticConfig; data: EngineData; prior: PriorArrays; periods: pd.PeriodIndex
                  hyper: dict; notes: tuple[str, ...]; eb_converged: bool | None; salt: int
  def resolve(model: BVAR, data: VARData, runtime: Runtime) -> Resolved      # gram / const_path / T_pad rules (§10.5)
  def fit_bvar(model: BVAR, data: VARData, *, seed: int | None, init: FinalState | MCMCFit | None,
               runtime: Runtime | None, plan: FitPlan | None) -> MCMCFit
  @dataclass(frozen=True)
  class EBOutcome: hyper: dict; converged: bool; warnings: tuple[str, ...]; note: str | None
  def eb_hyper(model: BVAR, data: VARData) -> EBOutcome                         # §6.5
  ```
- **Tests**: `BVAR(vol=SV()).fit(data)` end to end (structural Mode I static config); Mode II and IW-equivalent
  static configs; reduced form; EB plug-in equals the Phase 1 EB mode (1e-8) at `cross = 1`; structural
  `cross ≠ 1` uses `eb_structural`; reduced `cross ≠ 1` records the note and does not raise; R9 (prior arrays depend
  only on the window); resolution of `gram`, `const_path`, `T_pad`; setting warnings (COVID warning suppressed by
  `SV(outliers=True)`); `converged` includes the EB flag; seed override; salts differ across windows and orderings.
- **Acceptance**: tests green; `fit()` works for every composition of §11.1.

### Stage 4 (integration and exit criteria)

#### WP19 — Composition validation suites (owner writes no kernel)

- **Files owned**: `tests/geweke/*`, `tests/sbc/*`, `tests/mutation/{mutants.py, test_mutants.py}`.
- **Consumes**: WP7 (`geweke`, `sbc`, `gandy_scott`, `block_sbc`); WP9 (`joint_spec`, references); WP14
  (`compose(..., overrides)`); WP17 (predictive SBC quantities).
- **Tests**: §9.4 (block SBC, one-block invariance), §9.5 (G-small, G-full, `exact=False` recorded), §9.6 (SBC),
  §9.7 (mutants M1–M16 with recorded catch tiers).
- **Acceptance (exit criterion)**: G-small green per PR for compositions 1–5, 7–10, 17 and the block configs;
  G-full and S green nightly for every composition; every mutant caught by some tier. (Row 18, added with the
  Clark–Ravazzolo acceptance work, joins the PR G-small set, §9.5.)

#### WP20 — Analytic twins at scale

- **Files owned**: `tests/analytic/{test_mcmc_twin, test_mode1_quadrature}.py`.
- **Consumes**: WP16; WP18 (`fit_bvar`, `fit_mcmc`); Phase 1 `AnalyticFit`.
- **Tests**: A2, A3, A4, A5 (§9.8).
- **Acceptance**: A2 and A5 (`n = 3`) green per PR; A3, A4, A5 (`n = 20`) green nightly.

#### WP21 — Parity tests

- **Files owned**: `tests/parity/{test_stochvol, test_bayesianvars}.py`, `tests/parity/SV_PARAMETRIZATION.md`.
- **Consumes**: WP8 fixtures; WP14 `fit_mcmc` with `ZeroMean`; WP6 `explicit_reduced`; WP18.
- **Tests**: §9.9 comparisons with the single parameter sets and derived R calls.
- **Acceptance (exit criteria)**: stochvol main, defaults and SV-t runs within 4 quantile-MCSE (PR);
  bayesianVARs within 4 MCSE (nightly) after P1–P5 are settled.
- **As closed (2026-10-23)**: the stochvol clause is met with the SV-t run replaced by the exact
  importance-sampling oracle (§2.10 Stage 4 amendment 1). The bayesianVARs clause is met in Phase 2 by the exact
  system-wide Gibbs oracle (`bvars_swa_*`, max |z| 2.75–3.56 against bounds 4.05–4.51; §2.10 Stage 4 amendment
  2); SPEC §13 now names that oracle as the Phase 2 criterion and moves bayesianVARs parity, after P1–P5, to the
  Phase 3 exit. `test_bayesianvars_parity` stays in place and skips until a `bvars_bayesianvars_*.nc` fixture
  exists.

#### WP22 — Ordering sensitivity

- **Files owned**: `structural/ordering.py`, `tests/integration/test_ordering.py`.
- **Consumes**: WP2 (`BVAR`), WP18 (MCMC fits), WP14 (`fit_axis`, optional), Phase 0 backtest.
- **Provides**:
  ```python
  def ordering_sensitivity(model: Model, data: VARData, n_perm: int = 10, pinned: Sequence[str] = (),
                           mode: Literal["prior", "forecast", "backtest"] = "forecast",
                           design: Literal["balanced", "random", "all"] = "balanced",
                           horizons: Sequence[int] = (1, 4, 8), origins=None, noise_reps: int = 3,
                           quantiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95), pool: bool = False,
                           seed: int = 0, batch: bool = False) -> OrderingSensitivityResult      # N08 §B4
  def permute_model(model: Model, order: Sequence[str]) -> Model   # ψ tuples and per-variable means follow variables
  ```
- **Tests**: N08 §B5 items 1 (analytic exact invariance, 1e-10, hypothesis), 2 (reduced × IW-equivalent invariance
  within MCSE, nightly), 3 (prior-mode `2^{i−1}` growth; IW-equivalent KS), 4 (IW equivalence), 6 (un-permutation
  with ×10ⁱ scales); item 5 (detection power) weekly.
- **Acceptance (exit criterion)**: tests green; `material` flag and summary table documented.

#### WP23 — Backtest integration and the Clark–Ravazzolo replication

- **Files owned**: `evaluation/backtest.py`, `benchmarks/clark_ravazzolo.py`, `tests/integration/test_backtest_mcmc.py`.
- **Consumes**: WP2 (`uses_fit_seed`, `supports_warm_start`), WP18, WP22 (24 orderings).
- **Provides**: §11.6 (`backtest(..., warm_start: bool = False)`; fit seeds; spawn context; `n_jobs` warning; warm
  state cache).
- **Tests**: analytic backtests bitwise unchanged (golden); fit seeds reach MCMC models only; resumed warm backtest
  equals an uninterrupted one (bitwise); warm vs cold scores statistically equal; cache semantics with init
  provenance; spawn context; one compile per `T` bucket.
- **Acceptance (exit criterion)**: tests green; the replication report meets N08 §C5 criteria 1–6 (including all 24
  orderings) or documents each failure.
- **Acceptance result (2026-10-23, `65dd212`)**: `benchmarks/reports/clark_ravazzolo_acceptance.{md,json,log}`, run at
  `c823d20` on a clean tree (3.58 h; every arm converged at 100% of origins). Criteria 1, 2, 3, 5 and 6 met (6: verdict
  'better' in 24/24 orderings with the convergence gate on); criterion 4 not met and documented: full-sample UR
  ΔLPS h1 +0.160 → h8 +0.273 (the rule needs h8 < h1; TBILL and the pre-crisis cells pass). The path-conditioned
  estimator on the same forecasts gives UR h8 +1.309; the largest UR h8 origins fall from about +41 nats (first run)
  to +7.6. The remaining gap to CR's shrinking UR gain is reported as a difference from CR (pseudo-real-time final
  vintage, the iw_equivalent baseline, library-default SV priors), not tuned.
- **Acceptance run (D-34, 2026-10-23)**: `--config acceptance` (4 × 20,000 draws, `lpd_method="integrated"`,
  `benchmarks/reports/clark_ravazzolo_preregistration.md`); the criteria rules are unchanged. Review amendments to
  `benchmarks/clark_ravazzolo.py`: the report records the commit and whether tracked files differ from it (`git
  status --porcelain --untracked-files=no`, with the SHA-256 of `git diff HEAD`), the cache directory and the
  origins it served per backtest (`rep["cached"]`; cache keys do not change with the code, so the documented command
  uses a commit-suffixed cache and writes `clark_ravazzolo_acceptance.{md,json}` next to the full run's report); it
  is labelled the acceptance run only for the registered config, estimator and backtest seed 0 with the structural
  pair and all 24 orderings run on a clean tree at a known commit (`acceptance_gaps`, listed in the banner
  otherwise); the main-stage report written before the ordering stage is labelled pending (title, banner, and
  criterion 6's detail text "pending" instead of "skipped"; `criteria()` itself is unchanged); `--note` text is
  printed for every config; the acceptance report cites the pre-registration and its two departures from the
  full run, and the header says how the stored draws are thinned; the startup log and the `--only-ordering`
  completion line name the config, estimator, budget, seed and cache; the Gaussian-moments store is written to a
  temporary file and renamed (two runs sharing a store never read a partial file).

#### WP24 — Performance benchmark and CI tiers

- **Files owned**: `benchmarks/phase2_perf.py`, `tests/bench/{test_perf_phase2, bench_kernels}.py`,
  `tests/bench/baseline.json`, `.github/workflows/ci.yml`.
- **Consumes**: WP18 (`fit`), WP3 (pilot numbers), WP19 (tier markers).
- **Provides**: BENCH-P2 driver and per-kernel microbenchmarks (§10.1, §10.6); CI tiers of §9.10 with the persistent
  compilation cache.
- **Tests**: `test_perf_phase2` (marker `bench`, pre-release).
- **Acceptance (exit criterion)**: BENCH-P2 < 120 s on one device on the idle reference laptop, reported on x86;
  baseline JSON committed; PR tier within budget.
- **Acceptance as amended (Phase 2 closeout, 2026-10-23)**: BENCH-P2 < 120 s on one device on the idle reference
  laptop (asserted where `tests/bench/baseline.json` has `reference: true`); baseline JSON committed; an x86 run
  reported (the CI bench job, amendment below; reported, not gated); the PR tier measured against the §9.10
  budget as amended there (cold ≤ 90 min per job, warm recorded per run).
- **Amendment (Phase 2 closeout, 2026-10-23): reported on x86.** The x86 report is the `bench-p2` artifact of the
  CI bench job (Actions > CI > Run workflow, bench = true), run on GitHub's x86-64 `ubuntu-latest` runner:
  `benchmarks/phase2_perf.py --kernels` writes `bench-p2.json` with ms per sweep, compile time, peak RSS, ESS/s
  and wall time, plus the machine fingerprint (`machine_info`: CPU model, logical CPUs, load average, versions).
  That run is reported, not gated. A shared cloud runner (expected 2 vCPU for a private repository, 4 for a
  public one; the CPU model is drawn from a mixed EPYC/Xeon pool) is not the §10.1 x86 machine, an idle 4–8 core
  AVX2 laptop, so the 120 s target is neither required nor asserted there: `test_perf_phase2` compares only
  against a baseline with the same fingerprint and asserts 120 s only on the machine of a baseline with
  `reference: true`.
  The earlier x86 datum (the provisional baseline of `d37e059`: 311 s on a loaded 4-core Xeon) is superseded by
  the first CI bench report at the release commit. A run on an idle 4–8 core x86 laptop stays open under Q2.
- **Status (2026-10-23)**: BENCH-P2 52.3 s cold, 49.6 s warm on the reference laptop (Apple M5 Max, `reference:
  true`; not idle, 1-minute load average 2.4–6.3; every timed run `converged = False`, Q14, resolved by D-35 without
  changing the gate); baseline committed;
  the bench job has never been dispatched, so the x86 report does not exist yet; the PR tier ran cold through
  `888f620` (40–85 min per job), and the cache fix of §9.10 is not yet verified in CI.

#### WP25 — Stretch: SOC/SUR dummies on the MCMC engine (after the exit criteria)

Dummy rows in structural form (fixed weights `1/σ̄_i²` in Mode I, `1/λ_i` in Mode II; N05 §7.7) and reduced form
(fixed variance `ψ_j`; N04 §3.1), with their own Geweke configurations and an A1 extension. Done jointly by the
owners of WP6, WP10 and WP11 after Phase 2 exits (it touches their files), behind a new `FitCache`/`CoefPriorArrays`
field added by the WP1 owner. **Delivered** (2026-10-22): contract and decisions in the WP25 amendments of §2.10;
builders `priors/dummies.py`; reference `validation/dummies.py`; tests `tests/mcmc/blocks/test_dummies.py`,
`tests/validation/test_reference_dummies.py`, Geweke rows 201–205, `test_a1_dummies_*`, mutants M17–M18.

### Exit-criteria map (SPEC §13 Phase 2)

| Criterion | Delivered by |
| :--- | :--- |
| Geweke and SBC green for every composition | WP19 (with block-level tiers in WP10–WP13 and references in WP9) |
| stochvol parity | WP21 (fixtures WP8); SV-t by the exact importance-sampling oracle (§2.10 Stage 4 amendment 1) |
| reduced-form sampler parity with an exact oracle (SPEC §13 as amended 2026-10-23; first written "bayesianVARs parity for the reduced-form sampler") | WP21 (fixtures WP8): the exact system-wide Gibbs oracle `bvars_swa_*` (§2.10 Stage 4 amendment 2); bayesianVARs parity moved to the Phase 3 exit |
| performance target (`n = 20, p = 4, T = 300`, Linear × SV, 4 × 10,000 draws < 2 min) | WP24 (pilot WP3) |
| ordering-sensitivity check | WP22 |
| runtime convergence diagnostics | WP15, wired by WP14 |
| Clark–Ravazzolo-style replication | WP23 |
| analytic twin (SPEC §10.3), pulled into Phase 2 | WP16 (A1), WP20 (A2–A5) |
