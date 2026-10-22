# 06 · Validating the Phase 2 Gibbs samplers: Geweke joint-distribution tests, SBC and block-level tests

*Research note for `rai-bvar` Phase 2 (SPEC §3 R6, §8, §10.3, §13). Prepared 2026-10-21; independently cross-checked the same day (corrections are listed with sources in the **Cross-check log** at the end, §14). Clean-room: every procedure below is specified from papers. No GPL source was read. The `stochvol` facts come from its CRAN reference manual (documentation) only.*

**What this note gives.** It is an implementation-grade specification of the correctness tests that SPEC R6 requires before any Phase 2 sampler is merged. It covers:

- Geweke (2004) marginal-conditional vs successive-conditional tests;
- simulation-based calibration (SBC; Talts et al. 2018, Modrák et al. 2025) with ECDF-envelope uniformity tests (Säilynoja, Bürkner & Vehtari 2022);
- the cheaper block-level tests that catch most errors earlier: the density-ratio identity (Grosse & Duvenaud 2014), block-conditional SBC, and the exact tests of Gandy & Scott (2021);
- a power study run for this note, showing how many replications or iterations each test needs to detect the CCM (2019) error;
- a concrete test plan with run-time estimates for `Linear × {Constant, SV, SV(outliers), SV(t)}` in structural and reduced form.

---

## 0. Decisions and key numbers

| # | Recommendation | Evidence |
|---|---|---|
| D-1 | Run four test layers per composition: (i) **density-ratio identities** and closed-form checks for every block (deterministic or MCSE-based, every PR); (ii) **block-conditional SBC** (every PR); (iii) **Geweke** with C parallel successive-conditional chains (small config every PR, full config nightly); (iv) **full SBC** with the log-likelihood among the test quantities (nightly). | §5, §7 |
| D-2 | The Geweke SE comes from **C ≥ 1000 independent SC chains** (C ≥ 200 only with t_{C−1} critical values), each started at a prior draw and run with **no burn-in**. Under H0 every chain is exactly stationary from its first iteration, so z = (ḡ_MC − ḡ_SC)/√(σ̂²/M₁ + s²_chains/C). Batch means serve as a cross-check. Every test function must have **finite variance under p(θ, y)** (Geweke eq. 2); raw data moments fail this in the SV compositions (§3.4). | §3.2, §3.4; calibrated in E1/E2 |
| D-3 | Choose test configurations so that **r = Var_prior/Var_post ≤ ~5** for every scalar parameter. The θ-chain of the SC simulator then has lag-1 autocorrelation κ = 1 − 1/r and integrated autocorrelation time IAT = 2r − 1. In practice this means small T (20–50) and tight, proper priors. | §3.3 (derived and re-derived in the cross-check; checked: predicted κ 0.71 vs observed 0.71) |
| D-4 | Use the same small-T, informative-prior regime for SBC and Geweke when hunting CCM-type errors. The CCM (2019) triangular algorithm is **exact under a flat prior**: the error is entirely a prior effect, so it is invisible when the data dominate. | §7.2 (derivation + numeric check: max mean shift 0.015 posterior s.d. under a flat prior vs 1.49 under a tight prior) |
| D-5 | Use Bonferroni over the G test functions for the pass/fail decision. Holm gives the same global decision, so use Holm only to list which functions fail. Set the family α to 1e-3 for per-PR runs and 1e-2 for nightly runs. Wrap tests in the Gandy–Scott sequential scheme (k = 3, Δ = 4) so that marginal p-values trigger a 4× rerun (with fresh, independent seeds) rather than a red build. | §3.5, §4.6 |
| D-6 | The SBC uniformity test is the **γ statistic** (Modrák et al. eq. 7 = Säilynoja et al. eq. 13). Its null distribution for discrete ranks is computed **exactly** with the Säilynoja et al. forward recursion (eqs 19–23), which is needed for the far-tail p-values that Bonferroni at α/Q requires; simulation is a cross-check. In our experiments it was more powerful than χ² with 20 bins in every configuration where either test had power above nominal. | §4.3, Table 7.3 |
| D-7 | Always include the **log-likelihood** among the SBC test quantities. Against the CCM error with Σ known, parameter ranks had ≤ 26% power at N = 2000, while the log-likelihood reached 93–100% at N = 500–1000. In the full MCMC (E2) the log-likelihood reached 98.5% at N = 1000, against 13% for parameter ranks. **Thin per replication**: a single global thinning of 10 made the log-likelihood test reject a correct sampler 26.5% of the time at N = 4000. | Table 7.3 |
| D-8 | SV-block tests are run on data from the **auxiliary mixture model** (exact target) **and** on data from the Gaussian model. The OCSN 10-component approximation (KL ≈ 7.9e-5 per observation) was undetectable by SBC at N = 4000, and so was KSC-7 (KL ≈ 2.7e-3). The full-composition tests should therefore not raise false alarms from the approximation at the planned sizes. This was shown for latent states with known SV parameters; with unknown parameters it is UNVERIFIED. | §8 |
| D-9 | Keep a **mutation suite** of deliberately wrong samplers (CCM-2019 TA, wrong DP-2015 order, data simulator using fresh latents, dof off by one, …). Every mutant must fail the suite. This mirrors Geweke (2004, §3). | §9.5 |

**Key constants.**

- Bonferroni critical values z\* = Φ⁻¹(1 − α/(2G)): G = 20 → 3.481; G = 50 → 3.719; G = 92 → 3.870; G = 150 → 3.988; G = 190 → 4.044; G = 300 → 4.149 (α = 0.01). At α = 0.001: G = 60 → 4.305; G = 90 → 4.394 (Geweke's value); G = 100 → 4.417; G = 120 → 4.456; G = 500 → 4.753. (All recomputed with `scipy.stats.norm.isf` in the cross-check.) With C parallel chains use t_{C−1} instead: at G = 100, α = 1e-3 the critical value is 4.534 for C = 200 and 4.440 for C = 1000 (§3.2).
- Null quantiles of γ for L = 99 posterior draws, computed **exactly** with the forward recursion (Säilynoja et al. 2022, eqs 19–23; §4.3). Simulation with 10⁵ draws agrees at q0.05 and q0.01 within Monte Carlo error.

| N | q0.05 | q0.01 | q0.001 | q0.0001 |
|---|---|---|---|---|
| 100 | 4.1e-3 | 6.8e-4 | 6.6e-5 | 5.5e-6 |
| 200 | 3.5e-3 | 5.7e-4 | 5.2e-5 | 4.6e-6 |
| 500 | 3.0e-3 | 4.9e-4 | 4.1e-5 | 3.7e-6 |
| 1000 | 2.7e-3 | 4.6e-4 | 3.7e-5 | 3.4e-6 |

  γ is discrete, so a quantile is defined only up to the nearest atom; use p-values (§4.3), not these quantiles, for decisions. The first version of this table was simulated with 2×10⁴ draws; its q0.001 entries were off by up to 20% (cross-check log).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `n, p, k` | variables, lags, regressors per equation (`k = np + n_d`) |
| `y_t, x_t` | observation and regressor vectors; `y_{1−p:0}` are fixed initial conditions (conditioned on, never simulated) |
| `Π = [π^(1) … π^(n)]` | reduced-form coefficients (k × n); `π^(j)` = equation j |
| `A` (`a_ij`), `γ_i` | unit lower-triangular contemporaneous matrix; structural coefficients, `Γ = AΠ'` |
| `λ_{i,t} = e^{h_{i,t}}` | structural variance; SV state `h`, parameters `μ_i, φ_i, σ_i` |
| `o_{i,t}`, `ω_{i,t}`, `ν_i` | outlier scale (SV-outliers); Student-t mixing weight and dof (SV-t) |
| `s_{i,t}` | auxiliary mixture indicator (KSC/OCSN). **Not** part of the model θ |
| `θ` | every unobservable of the *model*: parameters plus latent states `h, o, ω` (excluding `s`) |
| `p(θ)`, `p(y∣θ)` | proper test prior; observation density given θ and `y_{1−p:0}` |
| `q(θ∣θ', y)` | the transition kernel of one Gibbs sweep under test |
| `g(θ, y)` | Geweke test function; `G` = number of test functions |
| `M₁`; `C`, `M_c` | marginal-conditional draws; number of parallel SC chains and length of each (M₂ = C·M_c) |
| `σ²_g`, `τ²_g` | iid variance of g; long-run variance of g along the SC chain; RNE = σ²/τ² |
| `r`, `κ` | prior/posterior variance ratio; lag-1 autocorrelation of the SC chain (κ = 1 − 1/r) |
| `N, L, Q` | SBC replications (Modrák's S); posterior draws per replication (Modrák's M); test quantities |
| `f(θ, y)` | SBC test quantity; rank `r_i ∈ {0,…,L}` |
| `γ`, `γ̄_α` | Säilynoja/Modrák ECDF statistic and its α-quantile under H0 |

---

## 2. The joint model under test and its two simulators

Geweke tests and SBC need two simulators that are independent of the posterior sampler:

- a **prior simulator**, θ ~ p(θ);
- an **observables simulator**, y ~ p(y | θ).

Both are much simpler than the sampler, so they are written separately and reviewed line by line. Geweke (2004, §2) stresses that a failure can come from any of the three pieces.

### 2.1 Generative specification (structural form, SPEC §4.1)

The prior simulator draws θ in this order:

1. `a_i ~ N(m_a, V_a,i)`.
2. `γ_i | a_i ~ N(M ã_i, V_γ,i)`. This is the D3 conditional centring; the test prior **must use M ≠ 0**, or the coupling term is never exercised.
3. Volatility parameters and the latent paths:
   - `Constant`: `λ_i ~ IG(α₀, β₀)`.
   - `SV`: `(μ_i, φ_i, σ²_i)`, then `h_{i,·}` from its AR(1) prior. The initial-state convention (`h_{i,0}` or `h_{i,1}` from the stationary distribution, or a fixed `h_0` prior for random-walk SV) must be **bit-identical** to the convention the sampler's prior uses.
   - `SV(outliers)`: additionally `p_i` and `o_{i,t}` from their priors (`o = 1` w.p. `1 − p`, else uniform on the integer grid `{2, …, K}`, K = 20, as fixed in note 03).
   - `SV(t)`: additionally `ν_i` (discrete prior on the sampler's grid, note 03) and `ω_{i,t} ~ Gamma(ν_i/2, rate ν_i/2)`. The unit-scale vs standardised-t convention (note 03, `t_scale`) must be the same in the observables simulator and the sampler.

The observables simulator then runs, for t = 1..T:

```text
x_t   = (d_t', y_{t-1}', …, y_{t-p}')'                 # uses the fixed y_{1-p:0} for t ≤ p
e_it  = sqrt(λ_it) · ξ_it · ε_it,  ε_it ~ N(0,1)       # ξ = 1 | o_it | ω_it^{-1/2}
y_it  = −Σ_{j<i} a_ij y_jt + x_t'γ_i + e_it            # i = 1..n in order
```

In reduced form this is `y_t = Π'x_t + A⁻¹e_t`.

Rules (each is a known failure mode, §6):

- The observables simulator uses the **current** latent states `h, o, ω` from θ. It never draws fresh ones; Geweke (2004, §3, error 2) shows that tests detect this as an error.
- It never uses the auxiliary indicators `s`.
- The initial conditions `y_{1−p:0}` are a fixed constant of the test configuration.
- Everything data-derived inside the sampler (X, y\* = log(u² + c), cross-products, residual caches) is recomputed after each new y.
- Prior hyperparameters may not depend on y. The Minnesota ψ from AR fits, standardisation, and the ȳ₀ of SOC/SUR dummies are frozen constants in tests; R9's per-window recomputation is switched off.

### 2.2 Proper priors are mandatory

The joint p(θ, y) must exist. Geweke (2004, §2) makes a proper prior a precondition of the joint-distribution tests. In §5 of the same paper, a reduced-rank sampler coded with prior precision τ = 0 (an improper prior) made the successive-conditional simulator diverge: it stopped with an exponent overflow after a few thousand iterations, because no invariant joint distribution exists. Improper-prior models are tested as the limit of proper ones (Geweke 2004, §4 replaced p(σ) ∝ 1/σ by 0.01/σ² ~ χ²(4)).

---

## 3. Geweke (2004) joint-distribution tests

Source: Geweke, "Getting it right" (JASA 99:799–804). Equation numbers refer to the Feb-2004 preprint (`http://qed.econ.queensu.ca/pub/faculty/ferrall/quant/papers/04_04_29_geweke.pdf`).

### 3.1 The two simulators

**Marginal-conditional (MC)**, Geweke eqs (3)–(4), iid:

```text
θ^(m) ~ p(θ),   y^(m) ~ p(y | θ^(m)),   g^(m) = g(θ^(m), y^(m)),   m = 1..M₁
ḡ^(M₁) = M₁⁻¹ Σ g^(m),     σ̂²_g = M₁⁻¹ Σ (g^(m))² − (ḡ^(M₁))²
```

**Successive-conditional (SC)**, Geweke §2:

```text
θ̃^(0) ~ p(θ);  for m = 1..M₂:
    ỹ^(m) ~ p(y | θ̃^(m−1))
    θ̃^(m) ~ q(θ | θ̃^(m−1), ỹ^(m))        # one sweep of the posterior sampler, data = ỹ^(m)
    g̃^(m) = g(θ̃^(m), ỹ^(m))
```

Under H0 the SC chain is a Gibbs sampler on (θ, y) with invariant distribution p(θ, y). Geweke notes that ergodicity of the SC chain usually follows from the posterior sampler's ergodicity with little or no extra work.

Important consequence: if θ̃^(0) is drawn from the prior, the pair (θ̃^(0), ỹ^(1)) is already a joint draw. **The SC chain is stationary from m = 1 under H0, so no burn-in is needed.** Discarding iterations only lowers power, because under H1 the drift away from p(θ, y) is the signal. (This also answers the objection of Talts et al. 2018, §3, that Geweke's z-scores "will be meaningful only once the Gibbs sampler has converged": a prior-started SC chain needs no convergence under H0. Grosse & Duvenaud 2014, §4, make the same point: "each step of this chain is a perfect sample from p(θ, x)".)

### 3.2 Test statistic and standard errors

Geweke eq. (6):

```text
z_g = (ḡ^(M₁) − g̃̄^(M₂)) / ( σ̂²_g / M₁ + τ̂²_g / M₂ )^{1/2}  →  N(0,1)  under H0
```

Here τ²_g is the long-run variance (2π × spectral density at frequency zero) of g along the SC chain. The paper delegates its computation to the BACC software ("computes both τ̂²_g and the test statistic (6)", §2) and does not state the estimator in the text. **UNVERIFIED:** BACC's estimator is believed to be a tapered spectral estimate with 4/8/15% tapers, in the Geweke (1992) style. (Verified only for a different source: the CCCM 2022 corrigendum, Fig. 2 caption, reports inefficiency factors with 4, 8 and 15 percent tapering.) Relative numerical efficiency is RNE = σ̂²/τ̂² (Geweke 2004, §5 text and Table 4).

**Recommended estimators for `rai-bvar`, in order:**

1. **Parallel chains (primary).** Run C independent SC chains, each of length M_c and each started at an independent prior draw. Use

   ```text
   g̃̄ = C⁻¹ Σ_c ḡ_c,    Var(g̃̄) = s²_C / C,    s²_C = (C−1)⁻¹ Σ_c (ḡ_c − g̃̄)²
   ```

   This is valid for any M_c ≥ 1 because each chain is stationary under H0. It needs no spectral or bandwidth choices and maps directly onto `vmap(scan(...))`. It does not even need ergodicity: the chain means are iid with mean E_p[g] whenever the sweep leaves p(θ | y) invariant (pitfall 14).

   **Use t_{C−1} critical values, not normal ones.** The t correction is not negligible at Bonferroni-sized tails. With Gaussian chain means, the normal-critical-value test has the following actual size:

   | G, α_family | C = 200 | C = 1000 | C = 4000 |
   |---|---|---|---|
   | G = 100, α = 1e-3 (z\* = 4.417) | 1.64e-3 (t\* = 4.534) | 1.11e-3 (t\* = 4.440) | 1.03e-3 |
   | G = 200, α = 1e-2 (z\* = 4.056) | 1.43e-2 (t\* = 4.146) | 1.08e-2 (t\* = 4.073) | 1.02e-2 |

   Recommendation: C ≥ 1000, and use t_{C−1} critical values. The MC term adds degrees of freedom, so this is slightly conservative.

   *Calibration (this note, E1/E2):* with C = 1000–2000 and M_c ∈ {25, 50, 100, 200, 400, 800}, the exact and CTA samplers never exceeded the Bonferroni critical value. The largest |z| observed was 3.76 against a critical value of 3.87 (G = 92) and 3.12 against 4.04 (G = 190).
2. **Batch means (cross-check)** within the pooled chain. Use B batches of size b = ⌊M^{1/2}⌋ and

   ```text
   τ̂² = b/(B−1) Σ_k (ḡ_k − ḡ)²
   ```

   Flegal & Jones (2010, *Ann. Statist.* 38(2):1034–1070) prove strong consistency of batch means and spectral variance estimators as batch size and number of batches grow, and mean-square consistency for batch means, which yields an optimal batch-size rate. b = ⌊M^{1/2}⌋ is the conventional choice. The exact consistency conditions (mixing rate, moment conditions) are not restated here.
3. **Geyer (1992) initial positive / initial monotone sequence** estimator on the autocorrelations: τ̂ = −1 + 2 Σ_{t'=0}^{k} P̂_{t'}, with P̂_{t'} = ρ̂_{2t'} + ρ̂_{2t'+1} (Vehtari et al. 2021, §3.2, eq. 13). ArviZ's `ess(method="mean")` / `mcse(method="mean")` is stated to implement this; the ArviZ source was not re-read in the cross-check. Use it for diagnostics and RNE reports.

Store running sums Σg and Σg² per chain and, for batch means, per batch. Do not store the draws themselves.

### 3.3 Mixing of the SC chain: closed form and design rule

This result is derived here. It is consistent with Geweke's choice of T = 6 and T = 10 (Geweke gives no rationale for these values). Grosse & Duvenaud (2014, §4) state the principle directly: fewer data points help the chains mix faster, because "the fewer data points there are, the weaker the coupling between θ and x". It also explains why Gandy & Scott (2021, Table 1) saw inflated false-rejection rates when θ and y are strongly coupled. In their model y = θ₁ + θ₂ + ε with θ_i ~ N(0, 10²) and σ²_ε = 0.1, the sum θ₁ + θ₂ has r = 200/0.09995 ≈ 2000. So IAT ≈ 4000 even with an exact posterior sampler, against their 3000 sweeps (600 draws × thinning 5).

Take the scalar conjugate model θ ~ N(0, τ²), y | θ ~ N(θ, s²), where s² stands for the data variance (for T observations, s² = s₀²/T). Let κ = τ²/(τ² + s²). A sweep gives

```text
θ_{m+1} = κ (θ_m + e_m) + sqrt(κ s²) z_m,   e_m ~ N(0,s²), z_m ~ N(0,1)
```

which is an AR(1) process with coefficient κ and stationary variance τ², as it must be. With r = Var_prior/Var_post = 1/(1 − κ):

```text
κ = 1 − 1/r,   IAT(θ) = (1+κ)/(1−κ) = 2r − 1,   IAT(θ²) = (1+κ²)/(1−κ²),   RNE ≈ 1/(2r−1)
```

(Re-derived in the cross-check. The posterior variance is τ²s²/(τ²+s²) = κs², and the stationary variance is κ²(τ²+s²) + κs² = τ². The IAT(θ²) formula assumes prior mean 0; with mean μ₀ ≠ 0, θ² also carries the linear term 2μ₀θ, which has IAT 2r − 1.)

**If the posterior sampler is itself autocorrelated, the IATs roughly multiply.** Derivation (scalar case): let the inner kernel be θ' = m + ρ(θ − m) + √(v(1−ρ²)) z, which leaves N(m, v) invariant. Then the SC chain is AR(1) with coefficient κ_tot = 1 − (1−κ)(1−ρ). For κ and ρ near 1 this gives IAT_tot ≈ 2/((1−κ)(1−ρ)) ≈ IAT_SC · IAT_inner / 2.

*Empirical check (E1 Geweke, VAR with known Σ_t):*

| Configuration | Approx. r | Predicted κ | Observed lag-1 autocorrelation |
|---|---|---|---|
| T = 50, own-lag prior s.d. 0.2 | 3.4 | 0.71 | 0.71 |
| T = 20, s.d. 0.4 | 4.8 | 0.79 | 0.74 |
| T = 50, s.d. 0.1 | 1.5 | 0.33 | 0.42 |

Median RNE was 0.15–0.25 in the first two configurations and about 0.6 in the third.

**Design rule.** For every scalar block parameter, aim for r ≤ 5 (IAT ≤ 9). Achieve it with small T (20 per PR, 50 nightly) and with proper priors that are tight relative to the information in T observations.

| Parameter | Information per observation | Implication |
|---|---|---|
| SV level μ | at most 1/4.93 per observation (var log χ²₁ = π²/2). With persistent h it is roughly 1/(π²/2 + σ²/(1−φ)²), because the sample mean of h is itself noisy | r = 1 + B_μ²·I_μ with I_μ ≤ T/4.93. The stochvol default μ ~ N(0, 100²) gives r up to about 2000·T (≈ 240·T at φ = 0.95, σ = 0.3): unusable. The test prior μ ~ N(0, 0.5²) gives r ≤ 2.0 at T = 20 and r ≤ 3.5 at T = 50 |
| φ, σ | weak at T ≤ 50 | r ≈ 1–2 expected; **UNVERIFIED**, measure in the pilot |

### 3.4 Test functions

- **Geweke's choices.** In §3: the 5 first and 15 second moments (squares and cross-products) of the parameter vector, 20 in total. In §4: the 9 parameters, 3 functions of interest, and all squares and cross-products of these 12, giving 90. In §5: 46 functions, including the **ordered eigenvalues λ₁ ≥ … ≥ λ₄ of Y'Y**, which are pure data functions and were among the most sensitive. In Table 6 (after the first fix) they reached z = 16.5, 20.3, 16.3 and 6.6, against 11.4–15.0 for the Σθ² functions. In Table 5 (original code) λ₂ had the largest statistic, 7.76. Note that Geweke's §5 model has Gaussian errors with a Wishart prior, so these data moments have finite variance there; this does not carry over to SV (next bullet).
- **Finite variance is a precondition (Geweke 2004, eq. 2: σ²_g = var[g(θ, y)] < ∞).** Several "natural" test functions violate it under the priors of this note, and then z is not asymptotically N(0, 1). Cross-check simulation (`xc06_check/heavy_g.py`, 10 batches of 2×10⁵ prior–data draws each, T = 50):
  - **Raw data moments in SV compositions.** Under μ ~ N(0, 0.5²), (φ+1)/2 ~ Beta(10, 2), σ² ~ 0.1·χ²₁ and stationary h₁, E[y²] = E[e^h] is **infinite**. E[exp(cσ²/(1−φ²))] diverges because σ² has a χ² tail and φ has mass near 1; any unbounded-support σ² prior or any φ prior reaching 1 does this. Batch means of mean_t y²_t ranged from 2.3 to 5.2×10³ and batch s.d.s from 244 to 2.3×10⁶. By contrast mean_t log y²_t was stable: −1.27 in every batch, s.d. 0.66. So tr(Y'Y), eigenvalues of Y'Y, Σ y₁y₂, max|y| and mean y² are **not valid** Geweke functions there. Use log-scale or bounded data functions instead: mean_t log y²_{i,t}; sample correlations and autocorrelations, which lie in [−1, 1]; tanh or arctan of standardised data.
  - **Squares of IG-distributed λ.** For λ ~ IG(α, β), var(λᵏ) < ∞ iff α > 2k. IG(3, 2), used in E2 and first proposed in §9.1, gives var(λ²) = ∞: batch s.d. of λ² ranged from 11.6 to 299, against 2.5–3.3 for IG(5, 4). Use log λ in g, or a shape > 4.
  - **Squares of h under a Beta(a, 2) prior on (φ+1)/2.** E[h⁴] ∝ E[(1−φ)⁻²], which diverges (logarithmically) for b ≤ 2. This is borderline in practice (batch s.d. of mean h² 1.2–1.7). Use b > 2, or h and |h| rather than h².
  - **Student-t errors.** E[y^{2k}] < ∞ iff ν > 2k, so data functions of y² need ν > 4 on the whole prior support.
  - SBC is unaffected: ranks need no moments.
- **Grosse & Duvenaud (2014, §4)** illustrate the amplification effect. A 0.51-instead-of-0.5 factor in the inverse-gamma scale (a within-cluster variance about 2% too large) compounds through the data resampling: after each SC step the data are generated with a slightly larger variance. Their §4 figure shows the sampled within-cluster variance σ²_n rising from about 1 to about 10⁵–10⁶ within 1000 steps.

Recommended g-vector for a VAR-SV composition (size G ≈ 60–120 at n = 2–3, p = 1):

1. All scalar parameters: a_ij, γ_i or π^(j), log λ_i (Constant) or (μ_i, φ_i, σ_i), p_i and ν_i where present.
2. Latent states at fixed dates: h_{i,1}, h_{i,⌈T/2⌉}, h_{i,T}, and mean_t h_{i,t}. For outliers, add the share of o_{i,t} ≠ 1; for t errors, mean_t log ω_{i,t}.
3. Squares of all items in 1–2, **only where var(g) < ∞** under the test prior (see the finite-variance bullet above). Cross-products **within** blocks: (γ_ij, γ_ik), (a_ij, γ_i·), (μ_i, φ_i), (φ_i, σ_i), (μ_i, h_{i,t}).
4. Joint functions:
   - log p(y | θ), the exact Gaussian likelihood given the latent states (its moments are finite, because at a joint draw the standardised residuals are exactly N(0, 1));
   - log p(θ);
   - Σ_t ε̂²_{i,t}, the standardised residual sum of squares per equation;
   - one-step predictive mean Π'x_{T+1} and log λ_{i,T+1} (the latter if `simulate_ahead` is part of the sweep).
5. Data functions:
   - `Constant` compositions with IG shape > 2: tr(Y'Y)/T, eigenvalues of Y'Y/T, cross-moments Σ y_{1,t}y_{2,t}/T and first autocovariances. Their finite variance needs E[λ²] < ∞.
   - `SV`, `SV(outliers)` and `SV(t)` compositions: **log-scale or bounded functions only**: mean_t log y²_{i,t}, sample cross-correlations and first autocorrelations of y_i and of log y²_i, and mean_t tanh(y_{i,t}). Raw moments and max|y| have infinite mean there.

Data functions test the observables simulator jointly with the sampler; Geweke (2004, §2) notes that an observables-simulator error shows up differently in MC and SC.

### 3.5 Multiplicity

- **Geweke (2004, §4)** uses Bonferroni: at the 0.1% level with 90 test functions, the critical value is Φ⁻¹[.001/(2·90)] = 4.394. This value is reproduced here (§0).
- **Holm vs Bonferroni.** For the *global* null ("the sampler is correct"), Holm rejects exactly when min p ≤ α/G, which is the Bonferroni condition. Holm only adds power for identifying *which* g's fail. So the SPEC's "Holm correction" equals Bonferroni for the pass/fail decision; use Holm-adjusted p-values for reporting.
- **Dependence.** The g's are strongly positively correlated (squares, cross-products), so Bonferroni is conservative. An optional sharper alternative is a max-|z| test with a critical value simulated from N(0, R̂), where R̂ is the correlation of the chain-mean vectors across the C chains. This is a parametric Westfall–Young-style max-T test (not from the cited papers).
- **CI false alarms.** Tests use fixed seeds, so a "false" failure is deterministic for given code and recurs only when the RNG consumption changes. Use α_family = 1e-3 per composition per PR and 1e-2 nightly, plus the sequential rerun (§4.6).

### 3.6 Geweke's settings as a calibration of what is realistic

| Study | T | Simulator sizes | Result |
|---|---|---|---|
| §3 t-mixture | 6 | M₁ = M₂ = 2.5×10⁵, under a minute | Five intentionally introduced errors (wrong Beta prior for p; observables simulator using fresh ω; zero conditional variance; dof 5 instead of 6; ω drawn several steps after s instead of immediately) failed 2–9 of the 20 tests at the 0.001 level (4–11 at 0.05). The correct samplers failed none (Table 1) |
| §4 unit-root model | 10 | M₁ = 10⁶ prior draws, M₂ = 4×10⁶ with every 400th draw kept | Found a coding error in a published 1994 paper: the (a₁…a₄) and (γ, δ) draws came from the *marginal* of the joint conditional with σ, not the conditional. The Bayes factor for difference stationarity was too high by a factor of ten. 37 of 90 statistics exceeded 4.394 |
| §5 reduced rank | 6 | M₁ = 10⁶, M₂ = 4×10⁶ with every 40th draw kept | Found two errors in a published 1996 paper: prior precision τ = 0 in the Ψ\* conditional, and a τ = 0 joint conditional for (Φ, Σ) retained when τ > 0 |

Error 5 of the t-mixture study is the same class as the Del Negro & Primiceri (2015) mixture-indicator ordering issue (SPEC §4.3): the tests detect step-order errors.

**Numbering caveat (verified in the preprint).** Table 1 swaps the labels of errors 3 and 4 relative to the text. In the text, error 3 is "variance of the (μ₁, μ₂) conditional set to zero" and error 4 is "ω dof 5 instead of 6". Table 1 lists "3. ω degrees of freedom" (5/3/3/3 failures at .05/.01/.005/.001) and "4. µ variance" (11/10/10/9). This note uses the **text** numbering, so the dof error failed 3 of 20 tests at 0.01.

---

## 4. Simulation-based calibration (SBC)

### 4.1 Algorithm and ranks

Talts et al. (2018, arXiv:1804.06788v2, §4.1: Theorem 1 and Algorithm 1; proof in Appendix B). For i = 1..N:

1. Draw θ̃_i ~ p(θ) and ỹ_i ~ p(y | θ̃_i).
2. Obtain L draws θ_{i,1..L} from the sampler for ỹ_i.
3. For each test quantity f compute the rank

   ```text
   r_{i,f} = Σ_l 1[ f(θ_{i,l}, ỹ_i) < f(θ̃_i, ỹ_i) ]  ∈ {0, …, L}
   ```

Under a correct sampler with independent draws, r_{i,f} ~ Uniform{0, …, L}.

**Ties** (discrete quantities such as the outlier share or o_{i,t} = 1, ν on a grid, or MH rejections that repeat the same value). Modrák et al. (2025, §1.2): with N_less = #{f(θ_l) < f(θ̃)} and N_eq = #{f(θ_l) = f(θ̃)}, set rank = N_less + K, where K ~ Uniform{0, …, N_eq}.

### 4.2 MCMC specifics

- **Thinning (Talts et al., §5.1, Algorithm 2).** If ESS < L, run longer and thin by ⌈n_draws/ESS⌉, using one thinning value for all quantities (the largest needed). Talts et al. compute the ESS of the ECDF indicators I(f ≤ f\*) at 19 equispaced empirical quantiles and take the minimum, because the rank is a CDF statistic. Evidence that autocorrelation breaks uniformity tests:
  - Säilynoja et al. (2022, §4.2.3): rejection rates of their multi-chain comparison test rise with AR(1) autocorrelation, and "the same holds for other common uniformity tests".
  - Säilynoja et al. (2022, Appendix A): extreme order statistics of autocorrelated draws are biased towards the centre, which produces the 0/L edge spikes. They compare ESS-based thinning rules, including the Talts quantile rule and min(bulk-ESS, tail-ESS).
  - Gandy & Scott (2021, §4.1): with an over-estimated ESS for a slowly mixing Gibbs sampler, Talts' procedure lost all control of the false-rejection rate (it rejected every scenario, including the correct sampler, in their Table 2).
  - *Practice (clean two-pass version of Talts' Algorithm 2).*
    1. **Pass 1 (pilot).** For each replication run a pilot chain with its own key (for example 300 burn-in + 1000 sweeps) and estimate max_f IAT_{i,f}. Use, per f, the larger of the mean-IAT and the Talts quantile-indicator IAT (or tail-ESS, Vehtari et al. 2021, §4.3).
    2. **Pass 2 (production).** Run a **fresh** chain with thinning t_i = ⌈2·max_f IAT_{i,f}⌉, capped at t_max, and keep L draws.

    The thinning decision then depends only on (ỹ_i, pilot), which is independent of θ̃_i given ỹ_i, so uniformity is preserved. Replications that hit t_max are rerun with a longer fresh chain, never dropped. Afterwards, check that the per-replication ESS of the thinned draws is at least 0.8 L for at least 99% of replications. Do **not** use the pooled lag-1 autocorrelation: a few large-variance replications dominate it (0.73 in E2 even though the ranks were calibrated).

    A single global thinning factor is not enough. In E2 the per-replication IAT had median 2 and maximum above 400, and thinning 10 produced false alarms (§7.3).
- **Initialisation.** Initialise each replication **independently of θ̃_i**, for example from a separate prior draw, and use a burn-in. Starting the chain at θ̃_i makes ranks non-uniform: the chain draws are not exchangeable with θ̃ (Gandy & Scott 2021, §2.2). That procedure is valid only with the Besag–Clifford backward/forward construction (§4.6), which requires a reversible kernel.
- **Failed fits.** Do not drop replications because the sampler failed or did not converge; that is selection on sampler output and invalidates uniformity. Record the failure and fail the test.
- **Data-only filtering is valid.** Replications may be discarded **based on ỹ alone**, for example non-finite values or max|ỹ| > 10³ from explosive VAR draws. Ranks are uniform conditionally on y for a correct posterior: this follows from Modrák et al. (2025) eq. (3), and the text after Theorem 5 (§3.2) says ranks are uniform "even when we only consider simulations that yielded data in some Ȳ ⊂ Y", calling the procedure a form of rejection sampling. So selecting on y preserves uniformity. The check then covers only the retained region of data space. Filtering on θ̃ does **not** preserve uniformity. Used in E1/E2 (1.7% of datasets dropped at T = 50; null calibration intact).

### 4.3 Uniformity tests

**γ statistic** (Säilynoja et al. 2022, eq. 13; Modrák et al. 2025, eq. 7). For the N ranks of one test quantity, each in {0..L}, define for i = 1..L:

```text
z_i = i/(L+1),   R_i = #{ranks < i}
γ = 2 · min_{i=1..L} min{ F_Bin(R_i; N, z_i),  1 − F_Bin(R_i − 1; N, z_i) }
```

The point i = L + 1 is trivial. Reject uniformity at level α if γ < γ̄_α, the α-quantile of γ under H0.

- **Null distribution: exact recursion (recommended).** With evaluation points z_i = i/(L+1), the counts R_i of discrete Uniform{0..L} ranks have exactly the same joint law as the ECDF counts of continuous uniform PITs at z_i. So the Säilynoja et al. (2022, §2.3, eqs 15–23) recursion is exact for SBC ranks:
  - Start with R₀ = 0. Step i (i = 0..L−1): R_{i+1} − R_i | R_i ~ Bin(N − R_i, 1/(L+1−i)), their eq. 20 with z̃_{i+1} = (z_{i+1} − z_i)/(1 − z_i).
  - After each step, zero the mass outside the interior I_{i+1}(γ) = [Bin⁻¹(γ/2 | N, z_{i+1}), Bin⁻¹(1 − γ/2 | N, z_{i+1})] (their eq. 15).
  - The surviving mass is P(T(γ)) = P₀(γ_null ≥ γ), up to boundary ties.
  - The p-value is p = 1 − P(T(γ_obs⁺)), with γ_obs⁺ slightly above γ_obs, because γ ≥ g holds exactly on the interior T(g) up to ties. The cost is O(L·N²) per evaluation: a few seconds for N = 1000, L = 99 with a naive SciPy loop. It vectorises as a banded matrix–vector product per step, and is computed once per (N, L, γ_obs).
  - Implemented and checked in the cross-check (`xc06_check/gamma_exact.py`); it reproduces 10⁵-draw simulated quantiles within Monte Carlo error. Quantiles for L = 99 are in §0.
  - **Why not simulation alone.** Bonferroni at α = 1e-3 over Q ≈ 20–40 quantities needs per-quantity tails near 3e-5–5e-5. A 10⁵-draw null table holds only 3–5 draws beyond such a threshold. Simulate (10⁵ draws, cached per (N, L) with its seed) only as a cross-check of the recursion.
- **Combining quantities.** Bonferroni: fail if min_f p_f ≤ α/Q.
- **Graphics.** Draw ECDF-difference plots with simultaneous bands L_i, U_i = Bin⁻¹(γ̄/2 | N, z_i)/N and Bin⁻¹(1 − γ̄/2 | N, z_i)/N, minus z_i (Säilynoja et al. eqs 11–12). γ̄ is the adjusted coverage parameter, found by Brent's method on the recursion (eq. 24) or by simulation (§2.2 of the paper).
- **Alternatives.**
  - Talts' histogram band (§4.1): the 0.5% and 99.5% quantiles of Binomial(N, 1/B) per bin (Binomial(N, (L+1)⁻¹) before rebinning). For N = 1000 and B = 20 it is [33, 69] around 50 (recomputed). It is a graphic, not a simultaneous test.
  - χ² on B bins: Talts et al. (§7) found global χ² tests "did not perform particularly well". In our experiments it was dominated by γ; for example 0.32 vs 0.77 power at N = 500 (Table 7.3).
  - Modrák et al. report log(γ/γ̄₀.₀₅) and note that γ is usually more sensitive than Kolmogorov–Smirnov or χ² tests.
- **Reading histogram shapes (Talts et al., §4.2, Figs 4–7).**

| Shape | Meaning |
|---|---|
| ∩ | sampler over-dispersed |
| ∪ | sampler under-dispersed |
| skewed | biased: draws biased low give high ranks |
| spikes at 0 and L | residual autocorrelation |

### 4.4 Test quantities

**Modrák et al. (2025)**

- **Theorem 7 (§3.4).** A "posterior" that ignores the data (equals the prior) passes SBC for **every** test quantity that does not depend on y. Data-dependent quantities are therefore required.
- **Theorem 6.** The density ratio p(θ | y)/φ(θ | y) as test quantity is complete, but impractical.
- **Heuristic (§3.3).** The likelihood p(y | θ) is often a useful proxy for it. In their case studies it detected every discrepancy studied.
- **Recommendation (§6.1).** Use individual parameters, the joint log-likelihood, likelihoods of data subsets where partial data use is a concern, the log prior when the prior implementation is uncertain, products or differences of parameters for dependence, and quantities tied to inferential goals.

**For VAR-SV in `rai-bvar` (Q ≈ 20–40 at n = 2–3)**

- All structural parameters and SV parameters.
- h_{i,t} at three dates and mean_t h_{i,t}.
- **log p(y | θ)** in total and per equation (conditional on latents). The per-equation term is the "subset likelihood" that targets coupling errors such as the CCM error.
- log p(θ).
- **Predictive unknowns.** If the sweep or forecast path simulates h_{i,T+1} and y_{T+1}, treat them as additional unknowns: the generative step draws y_{T+1} ~ p(y_{T+1} | θ̃, y_{1:T}), and the sampler's predictive draws are ranked against it. This is valid because (θ̃, y_{T+1}) | y_{1:T} follows the posterior predictive. It checks the §7 forecast contract end to end.

**Invalid test quantities.** A function of simulated data *not* given to the sampler, such as log p(y_{T+1} | θ) with y_{T+1} held out, is invalid. θ̃ and y_{T+1} are dependent given y_{1:T}, but the posterior draws are not, so ranks are non-uniform even for a correct sampler.

### 4.5 How many replications (literature)

- **Talts et al.:** L = 100 in the experiments (§6). Choose L + 1 divisible by a power of 2 for rebinning, for example L = 1023 (§4.1). N = 10⁴ in §6.1–6.3 and 10³ for INLA (§6.4). Choose B such that N/B ≈ 20 (§4.1).
- **Modrák et al. (§6.2):** use roughly as many simulated datasets as posterior draws per dataset, with at least about 100 effective draws per fit. Adding test quantities raises the false-alarm risk. Their violations were similar across quantities built from the same inputs, which in their view makes a multiple-comparison correction less pressing. Extra quantities do not help much against *small imprecision* in a nearly correct posterior.
- **Posterior SBC** (Säilynoja, Schmitt, Bürkner & Vehtari 2025, arXiv v1): 250–500 iterations in the case studies (§3.1: 500; §3.2: 250 prior SBC vs 500 posterior SBC).
  - *Procedure:* draw θ′ ~ p(θ | y_obs) from the sampler, simulate y_rep ~ p(y | θ′), refit on (y_obs, y_rep) jointly, and rank θ′. This concentrates the check on the region of the observed data.
  - *For VAR-SV* it needs a sampler that accepts two independent series with shared parameters and separate latent paths. Phase 3 at the earliest; optional.
- **Relation to Geweke (Modrák et al. §3.5).** SBC is *not* equivalent to the data-averaged-posterior (self-consistency) identity p(θ) = ∫∫ p(θ | y) p(y | θ̃) p(θ̃) dy dθ̃, which Modrák et al. associate with Geweke (2004, their eq. 1). They construct posteriors that satisfy one and fail the other. Geweke's SC test with joint test functions g(θ, y) probes the whole stationary joint of the (θ, y) chain, which is more than that identity, but it is still a different check from SBC. Run both; §7 shows they have very different power profiles.
- **Our power numbers for the CCM error** are in §7.

### 4.6 Exact variants and the sequential wrapper (Gandy & Scott 2021, arXiv:2001.06465v2)

- **Exact two-sample test (Algorithm 1, §2.1).** Draw (θ̃, ỹ) from the joint and run **L sweeps starting at θ̃** with ỹ fixed. Compare the *fitted* pairs (θ_L, ỹ) with independent *direct* pairs (θ, y) ~ p(θ) p(y | θ). The test statistics are functions of (θ, y), so they include the likelihood p(y | θ). The paper suggests two-sample KS, Cramér–von Mises or Wilcoxon for continuous statistics, LR or χ² for discrete ones, or a one-sample test against a simple prior. "γ on prior-CDF PITs" is this note's addition. In their Table 1 the likelihood p(y | θ) was the **only** statistic that detected the variance error (power 1.000 vs ≤ 0.010 for the parameter statistics). This needs no burn-in, no ergodicity assumption and no thinning, and samples are independent across replications. Variant: refresh ỹ ~ p(y | θ) after each sweep, which is a short SC chain. In their §4.2 example (wrong prior mean μ = 10) it improved power from 30.8% / 29.7% to 99.2% / 97.2% (two-sample / rank variants). At that budget Geweke's test had nominal size and 100% power.
- **Exact rank test (Algorithm 2, Proposition 2.3).** Draw M ~ Uniform{1..L}, set θ_M = θ̃, run the kernel M − 1 steps backward and L − M steps forward, and use rank R_M. It is exactly uniform if K_y is π(· | y)-reversible.
  - Systematic-scan Gibbs is not reversible: the exact rank test with systematic scan rejected 77% of the time under H0 (their Table 1).
  - Use it for single reversible blocks (MH steps), or for a palindromic scan. Applying it to the whole PCG-ordered SV sweep is UNVERIFIED and not recommended.
- **Sequential wrapper (Algorithm 3, Theorem 3.1).** With overall α, k stages and sample-size multiplier Δ, set β₁ = α/k and γ = β₁^{1/k}. At stage i, let q_i = d · min p (Bonferroni); fail if q_i ≤ β_i; pass if q_i > γ + β_i; otherwise set β_{i+1} = β_i/γ and continue. The sample size is multiplied by Δ **once**, after stage 1 (Algorithm 3, line 9: "if i = 1 then n = Δn"). Theorem 3.1 requires the p-value vectors of different stages to be **independent**, so every stage uses fresh keys and new samples, never an extension of the previous stage's chains. The expected extra effort under H0 is Δγ(1 − γ^{k−1})/(1 − γ) for exactly uniform p-values (their eq. 1), with upper bound Δ Σ_{i=2}^{k} Π_{j<i}(γ + β_j) (their eq. 2).

| Setting | γ | β sequence | Expected extra effort, eq. 1 / eq. 2 |
|---|---|---|---|
| `mcunit` defaults: α = 10⁻⁵, k = 7, Δ = 4 | 0.146 | 1.43e-6, 9.77e-6, 6.68e-5, 4.57e-4, 3.13e-3, 0.0214, 0.146 | 68.5% / 68.5% (paper: "around 68.5%") |
| Recommended per-PR setting: α = 10⁻³, k = 3, Δ = 4 | 0.0693 | 3.33e-4, 4.81e-3, 0.0693 | 29.7% / 29.9% |
| Nightly: α = 10⁻², k = 3, Δ = 4 | 0.149 | 3.33e-3, 0.0223, 0.149 | 68.7% / 71.6% |

  (All recomputed in the cross-check.) With the per-PR setting, a correct sampler is rerun at 4× the size with probability ≤ γ ≈ 7%. Bonferroni q_i makes this conservative. Nightly, the probability is about 15%.
- **Power caveat (Gandy & Scott §4.1, Table 1).** Errors that change every conditional but leave the joint unchanged (their truncated conditionals) are invisible to the joint tests (two-sample and Geweke) but detected by rank tests on single kernels. Errors that accumulate in the joint favour Geweke. For the wrong prior mean (Table 2, μ = 10), Geweke's power was 0.909, against at most 0.018 for the exact tests. Their Geweke implementation, however, had inflated size: 0.44–0.52 for the Bonferroni "All" row in Table 1 (6×10² samples, thinning 5) and 0.10 in Table 2 (10³ samples, thinning 50). Gandy & Scott attribute this to "the correlation between the data and parameters". In §3.3 terms, r ≈ 2000 for θ₁ + θ₂, so their 3000 sweeps cover less than one IAT. The parallel-chain design with r ≤ 5 (§3.2–3.3) avoids this. With a longer run their Geweke test achieved nominal size and 100% power (end of their §4.2). Run both kinds of test.

---

## 5. Block-level unit tests: cheaper, and they localise the bug

Integration tests say *that* something is wrong, not *where* (Grosse & Duvenaud 2014, §4). Every Gibbs block in `rai-bvar` exposes its conditional in closed form (Gaussian mean and precision, IG shape and scale, categorical probabilities), so three cheap tests apply.

### 5.1 Density-ratio identity

Grosse & Duvenaud (2014, eq. 1). For a block b with implemented conditional q_b(θ_b | θ_−b, y) and two values θ_b, θ'_b:

```text
log q_b(θ'_b | θ_−b, y) − log q_b(θ_b | θ_−b, y)  ==  log p(θ'_b, θ_−b, y) − log p(θ_b, θ_−b, y)
```

This holds exactly for a correct conditional. Check it at random states with relative tolerance 1e-8 × (1 + |RHS|). Grosse & Duvenaud suggest a precision of about 10⁻¹⁰ in log space; 1e-8 relative leaves room for float64 round-off in long sums over T.

- **Requirement.** `log_joint` must be an **independent** implementation, written as prior log-densities plus the likelihood, not assembled from the block's own posterior formulas. Otherwise the test is circular.
- **Partially collapsed steps.** The SV sweep of SPEC §4.3 draws the mean block given h with s marginalised. Its identity must use the *non-augmented* joint p(Π, A, h, y). The s-step uses the augmented joint under the mixture model, and the h | s step uses that same augmented joint.
- **MH steps.** Check that the computed log acceptance ratio equals log π(x') + log q(x | x') − log π(x) − log q(x' | x).
- **E1b result.** For the CTA step of π^(1) the maximum deviation was 5e-10 (median 1e-14). For the CCM-2019 step the median deviation was 1.1–4.4 nats. A single random pair detects the CCM error.

### 5.2 Block-conditional SBC

Draw (θ̃, ỹ) from the joint, draw L values from block b's conditional sampler **given the true θ̃_−b**, and rank θ̃_b (and the block's log-likelihood) among them. For exact-draw blocks the L draws are iid, so no MCMC issues arise. E1b detected the CCM-2019 π^(1) step with N = 20:

| Coupling | Power at N = 20 | Power at N = 50 |
|---|---|---|
| a_ij ~ N(0, 1), constant volatility | 99.5% | 100% |
| a_ij ~ N(0, 1), SV-like volatility | 98.5% | 100% |
| a_ij ~ N(0, 0.5²), SV-like volatility | 45% | 95% |

The correct step held the nominal 1% level.

### 5.3 Block invariance tests

Use the Gandy–Scott two-sample construction with **one block only**. Draw (θ, y) from the joint, apply K_b L times, and require the joint to be preserved. Compare the distributions of block-relevant quantities between the input and output samples.

- **Exact-conditional blocks:** L = 1 suffices.
- **MH and ASIS blocks** (SV parameters, ν, outlier probability; Kastner & Frühwirth-Schnatter 2014): L = 10–50. This is how ASIS re-parametrisation and Jacobian errors are found.

### 5.4 Closed-form checks (fixed data, deterministic or MCSE-based)

- **CTA Π block given Σ_{1:T}.** The stationary law must be the SUR Gaussian

  ```text
  vec(Π) | y, Σ_{1:T} ~ N(P⁻¹ b, P⁻¹)
  P = V⁻¹ + Σ_t Σ_t⁻¹ ⊗ x_t x_t'
  b = V⁻¹ μ + vec_by_equation( Σ_t x_t y_t' Σ_t⁻¹ )
  ```

  Run C = 4000 parallel CTA chains for 60 sweeps on one dataset. Check means within 4 MCSE and each variance ratio within 1 ± 4·√(2/C) ≈ 1 ± 0.09. In this note's check (`check_cta.py`): mean error ≤ 0.03 s.d. and variance ratios 0.97–1.03.
- **Structural `Linear × Constant` with independent normal–inverse-gamma priors per equation.** Exact iid posterior (Chan 2022), which is the SPEC §10.3 analytic twin.
- **Banded precision sampler** for h | s. The mean from the banded Cholesky equals the dense `cho_solve` to 1e-10. The sample covariance of C draws matches the dense inverse within Monte Carlo error. Check that the precision is assembled exactly as derived: `diag = (1+φ²)/σ²` inside, `1/σ²` at both ends under the stationary-h₁ convention (an h₀ convention changes the first entry), `off = −φ/σ²`, plus `1/v_{s_t}` on the diagonal. (Re-derived: the first entry is (1−φ²)/σ² from the stationary prior plus φ²/σ² from the t = 2 transition, which is 1/σ².) Check the linear term too: b = Q_prior·μ1 + (y\*_t − m_{s_t})/v_{s_t}. Q_prior·μ1 has entries μ(1−φ)/σ² at both ends and μ(1−φ)²/σ² inside. With stochvol's convention (h₀ stationary and sampled, h₁ | h₀ AR(1)), the marginal law of h_{1:T} is the same as with stationary h₁. Only the state dimension differs.
- **Indicator step.** Probabilities are computed in log space with `logsumexp` and sum to 1. Each probability equals the direct evaluation p_j N(y\*_t; h_t + m_j, v_j)/Σ_k(·) to 1e-12.

---

## 6. Pitfalls checklist

1. **Improper or vague priors.** No joint exists, and the SC chain diverges (Geweke §5). Vague proper priors make r enormous: the SC chain barely moves, and simulated data are absurd, for example λ = e^{±200} under μ ~ N(0, 100²).
2. **Observables simulator inside SC.** It must use the sampler's *current* latent states (h, o, ω). Drawing fresh ones is Geweke's error 2, detected by 9 of 20 tests at 0.01. It must not use the auxiliary s. It must rebuild the lagged regressors X from the new y with the *fixed* y_{1−p:0}, and invalidate every cache (X'X, y\*, residuals, scaled designs).
3. **Prior–sampler mismatch.** This includes h₀/h₁ conventions, Beta on (φ+1)/2 versus Beta on φ, σ² ~ Bσ·χ²₁ versus Gamma, and IG shape/scale versus shape/rate. Geweke's error 1 (Beta(1,1) vs Beta(2,2)) failed 3 of 20 tests at 0.01.
4. **Data-dependent priors** (SPEC R9: ψ from AR fits, standardisation, SOC/SUR ȳ₀). In tests they must be functions of the fixed presample only, or constants.
5. **Explosive VAR draws.** Under own-lag s.d. 0.4 and cross-lag s.d. 0.2 with zero mean (n = 3, p = 1), P(spectral radius > 1) = 4.2% (re-simulated: 4.18%, 2×10⁵ draws). At T = 50, 1.7% of datasets have max|y| > 10³; at T = 200, 3.4%. Non-finite values arise.
   - SBC: drop on y only (§4.2).
   - Geweke: use priors with negligible explosive mass. With zero mean and s.d. ≤ 0.2 for n = 3, P(radius > 1) is 0 in 2×10⁵ draws. Alternatively use a stationarity-truncated prior that is implemented identically in the prior simulator and in the sampler (accept/reject inside the Gaussian step); the truncation then becomes part of the model under test.
   - A random-walk prior mean (1 on the own lag) is unusable for Geweke. **Corrected:** own-lag mean 0.5 with s.d. 0.15 (cross 0.075), as first proposed, violates the §9.1 bound P(ρ > 1) < 10⁻³. Simulated P(ρ > 1), 4×10⁵ draws (`xc06_check/explosive2.py`):

     | Prior | n = 2, p = 1 | n = 3, p = 1 | n = 3, p = 2 |
     |---|---|---|---|
     | Reduced form, mean 0.5, s.d. 0.15/ℓ (cross 0.075/ℓ) | 0.10% | 0.15% | 0.48% |
     | Structural D3 with a_ij ~ N(0, 0.7²), same M and V | 0.26% | 0.97% | 1.8% |

     A⁻¹ amplifies the structural form, because B' = M' + A⁻¹E with the rows of E ~ N(0, V). **Use own-lag mean 0.4, own s.d. 0.10/ℓ, cross s.d. 0.05/ℓ.** With a_ij ~ N(0, 0.7²) this gives ≤ 1e-6, 7e-5 and 2e-4 for the three cases, and still exercises the M ≠ 0 coupling.
6. **Step order.** The DP-2015 order "mean block → s → h" must be exercised. Geweke's error 5 shows that step-order errors are detectable; include a mis-ordered mutant (§9.5).
7. **The mixture approximation means the sampler does not target the exact posterior** (§8). SV-block tests are run on auxiliary-model data, where they are exact. Full-model tests on Gaussian data did not detect the approximation at N = 4000 in E3 (latent states, known SV parameters).
8. **Offset c in log(u² + c).** Any fixed c is a model change whose size depends on c/λ. Primiceri-style c = 0.001 (Primiceri 2005, appendix: "c is the offset constant (set to 0.001)") with λ ≈ 1e-4 (decimal growth rates) dominates u². Tests must cover λ ranges where c/λ is not negligible, or use a scale-relative offset. The SV note fixes the choice; tests assert that it is documented and that its bias at the test scales is below MCSE.
9. **Initialising SBC chains at θ̃** (§4.2), or dropping failed fits.
10. **Holdout-dependent test quantities** (§4.4).
11. **Residual autocorrelation in SBC.** Its signature is edge spikes. In E2 a *correct* sampler thinned by 10 failed the log-likelihood test 26.5% of the time at N = 4000. The cause was a heavy right tail of per-dataset IATs (median 1.9, maximum 433). Thin per replication by its own ESS and rerun low-ESS replications with fresh chains. Thinning 40 restored nominal size (≤ 1% at N ≤ 2000).
12. **float32.** JAX's default would add noise and real bias; the tests assert float64 (SPEC §8) before anything else runs.
13. **Configurations in which a term vanishes hide its bugs.** Examples: a_ij = 0; M = 0 (the D3 centring term); a flat prior (the CCM error vanishes, §7.2); outlier probability p ≈ 0 (the outlier code path is never used); ν large (t equals Gaussian). Every test configuration must make every term matter.
14. **Ergodicity.** With a single long SC chain (Geweke's original design), z needs an ergodic SC chain. With the parallel, prior-started design of §3.2, the **size** of the test needs only invariance: if the sweep leaves p(θ | y) invariant, every chain is stationary, and the chain means are iid with mean E_p[g]. Ergodicity then affects only **power**: a reducible chain may hide errors in regions it never visits. Reducible samplers, such as ones stuck in label permutations for regime models (Phase 5), therefore still need care.
15. **Passing is not proof.** The tests have power only against errors that change the joint distribution or the conditionals at the tested configurations (§4.6 power caveat). Keep the mutation suite (§9.5) to know the suite's power.
16. **Infinite-variance test functions** (§3.4). Raw data moments under SV priors, λ² under IG(α ≤ 4), and y² under Student-t with ν ≤ 4 violate Geweke's eq. (2). Use log-scale or bounded functions.

---

## 7. Power study: how many replications detect a CCM (2019)-type error?

All experiments were run for this note in NumPy, float64. Scripts are in `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/gsbc_exp/`: `common.py`, `e1_sbc.py`, `e1_sbc_fixed.py`, `e1b_block.py`, `e1_geweke.py`, `e2_full.py`, `e3_sv_sbc.py`, `check_flat.py`.

### 7.1 Setup

- **Model.** n = 3, p = 1, intercept, so k = 4. Reduced-form prior π^(j) ~ N(0, V) with intercept s.d. 1, own-lag s.d. 0.4·τ and cross-lag s.d. 0.2·τ; τ is the tightness multiplier in the tables.
- **Covariance.** Σ_t = A⁻¹Λ_tA⁻¹′. A is either random with a_ij ~ N(0, s_a²), or fixed as a₂₁ = 0.8s, a₃₁ = −0.6s, a₃₂ = 0.7s. Λ_t is constant, or an "SV-like" known log-normal AR(1) path (φ = 0.95, innovation s.d. 0.3).
- **Samplers for Π.**
  - *Exact.* Joint Gaussian with precision V⁻¹ + Σ_t Σ_t⁻¹ ⊗ x_tx_t′.
  - *CTA* (correct; CCCM 2022 eqs 13–15). A Gibbs step for π^(j) | π^(−j) using equations j..n:

    ```text
    Ω̄⁻¹ = Ω⁻¹ + Σ_{i=j}^{n} a²_ij Σ_t x_t x_t'/λ_it
    μ̄   = Ω̄ (Ω⁻¹ μ + Σ_{i=j}^{n} a_ij Σ_t x_t z_it/λ_it)
    z_it = ỹ_it − Σ_{l≤i, l≠j} a_il x_t'π^(l),   ỹ_t = A y_t,   a_jj = 1
    ```

    Checked against the exact posterior: mean error ≤ 0.03 s.d., variance ratios 0.97–1.03.
  - *TA-2019* (wrong; CCCM 2022 eqs 5–6). Draw π^(1) | y^(1), then π^(2) | π^(1), y^(1:2), and so on, using equation j only. This is a direct (non-Markov) draw from a wrong Gaussian.
- **Experiments.**

| Experiment | Σ_t | What is tested |
|---|---|---|
| E1 | known | prior SBC and Geweke |
| E1b | known | block level |
| E2 | A and λ unknown; λ_i ~ IG(3, 2), a_ij ~ N(0, 1) | full Gibbs, conjugate (a, λ) steps; SBC with burn-in 300 and thinning 10 (L = 99); Geweke with C = 2000 chains |

- **Tests.** γ test with α_family = 0.01 and Bonferroni over Q = 13 (E1) or 19 (E2) quantities. Power is estimated from 200–400 random subsets of a 20 000-replication pool.

### 7.2 Why the error is a prior effect (derived here)

With a flat prior on (π^(1), π^(2)), the map (π^(1), π^(2)) → (π^(1), γ₂ = a₂₁π^(1) + π^(2)) has unit Jacobian. Equation 2 of CCCM's system (12), ỹ_{2,t} = a₂₁x_t'π^(1) + x_t'π^(2) + λ_{2,t}^{1/2}ε_{2,t}, depends only on γ₂ for two reasons: all equations share the regressors x_t, and **A is time-invariant** (true in CCMM, CCCM and `rai-bvar`). The volatility weights do not matter. The posterior of π^(1) therefore depends on equation 1 alone, and the CCM-2019 factorisation is **exact**. The missing term p(y^(2) | y^(1), π^(1), …), CCCM eq. (7), equals ∫ N(ỹ₂ | X(a₂₁π^(1) + π^(2)), Λ₂) p(π^(2)) dπ^(2). Under a flat p(π^(2)) it is constant in π^(1). Under a proper p(π^(2)) = N(μ₂, V₂) it is N(ỹ₂ | X(a₂₁π^(1) + μ₂), XV₂X' + Λ₂), which depends on π^(1) whenever a₂₁ ≠ 0. The cross-check re-derived this. With a time-varying A_t (Primiceri-type TVP-VAR), the shift a_{21,t}x_t'π^(1) cannot be absorbed into one γ₂, so the flat-prior exactness fails.

Numeric check (`check_flat.py`, fixed strong A, T = 50, 20 000 TA draws; re-run in the cross-check with identical output):

| Prior | Max \|mean shift\| (posterior s.d.) | s.d. ratio |
|---|---|---|
| flat, constant volatility | 0.015 | 0.99–1.01 |
| flat, SV-like volatility | 0.021 | 1.00–1.01 |
| default | 0.22 | 1.02–1.24 |
| tight (τ = 0.25) | **1.49** | 0.99–1.18 |

**Consequences.**

- (a) In large VARs with tight Minnesota priors the error matters; with long T and loose priors it nearly vanishes. This is consistent with CCCM (§5): "The empirical results presented in the paper did not change appreciably after re-estimating the model using the correct algorithm."
- (b) Tests must use informative, equation-asymmetric priors relative to T. This is the same regime that makes the Geweke SC chain mix (§3.3).
- (c) Bognanni (2022) states that the TA and the system-wide algorithm "generally sample from different distributions" (abstract, via IDEAS/RePEc; the full text was not accessed). The derivation shows that the difference is present for **every proper prior that is independent across reduced-form equations** (the CCMM/CCCM setting, CCCM footnote 2) whenever some a_ij ≠ 0, and absent for a flat prior. It is not a heteroskedasticity effect. No claim is made here for priors that are dependent across equations. In `rai-bvar`'s structural form (D3) the prior is independent across *structural* equations, so the triangular factorisation is exact by construction (SPEC §4.1).

### 7.3 Results

**Prior SBC, E1 (Σ_t known, random A per replication, L = 99).** Power at α_family = 1%. Columns are N; "loglik" is the log-likelihood quantity alone at 1%; "params" is Bonferroni over the 12 coefficients; "any" is the γ test over all 13 quantities with Bonferroni. The exact sampler was at nominal size throughout (0–2.5% rejection over all rows and N).

| Configuration | Test | N = 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| s_a = 1, T = 50, constant volatility, τ = 1 | any γ | 0.05 | 0.28 | 0.77 | **1.00** | 1.00 |
| | χ² (20 bins) | 0.01 | 0.05 | 0.32 | 0.93 | 1.00 |
| | loglik | 0.21 | 0.50 | 0.93 | 1.00 | 1.00 |
| | params | 0.01 | 0.02 | 0.04 | 0.09 | 0.25 |
| s_a = 1, T = 50, SV-like, τ = 1 | any γ / loglik | 0.03 / 0.11 | 0.12 / 0.26 | 0.46 / 0.74 | 0.89 / 1.00 | 1.00 / 1.00 |
| | params | 0.01 | 0.03 | 0.02 | 0.06 | 0.19 |
| s_a = 1, T = 20, SV-like, τ = 1 | any γ / loglik | 0.17 / 0.41 | 0.55 / 0.81 | 0.98 / 1.00 | 1.00 | 1.00 |
| | params | 0.02 | 0.03 | 0.04 | 0.10 | 0.26 |
| s_a = 0.5, T = 50, SV-like, τ = 1 | any γ / loglik | 0.01 / 0.02 | 0.02 / 0.03 | 0.02 / 0.03 | 0.02 / 0.10 | 0.04 / 0.11 |
| s_a = 1, T = 200, SV-like, τ = 1 | any γ / loglik | 0.02 / 0.02 | 0.01 / 0.02 | 0.02 / 0.02 | 0.01 / 0.07 | 0.06 / 0.10 |
| s_a = 1, T = 50, constant, **τ = 0.25** (prior-dominated) | any γ / loglik | 0.22 / 0.54 | 0.72 / 0.89 | **1.00 / 1.00** | 1.00 | 1.00 |
| | params | 0.01 | 0.01 | 0.02 | 0.01 | **0.01** |
| Fixed strong A, T = 50, SV-like, τ = 0.5 (the Geweke configuration) | any γ / loglik | 0.08 / 0.29 | 0.39 / 0.64 | 0.92 / 0.99 | 1.00 / 1.00 | — |
| | params | 0.01 | 0.04 | 0.04 | 0.10 | — |
| Fixed **weak** A (×0.25), T = 50, SV-like, τ = 0.5 | any γ / loglik | 0.01 / 0.02 | 0.01 / 0.01 | 0.01 / 0.01 | 0.00 / 0.02 | — |

Two observations matter most:

- **Parameter ranks alone are blind to this error in the prior-dominated regime.** At τ = 0.25 their power stayed at the nominal 1% for N = 2000. The per-dataset errors (mean shifts of up to 1.5 posterior s.d.) cancel when averaged over datasets, which is the situation of Modrák et al.'s Theorem 5. The log-likelihood detects the same error with N ≈ 200.
- **Prior SBC cannot see weak coupling at N ≤ 1000**, while Geweke detects it at |z| = 15 with 2.5×10⁴ sweeps (Geweke table below).

**Block-conditional SBC, E1b (π^(1) step, N replications).**

| Configuration | CTA step size | TA-2019 step power: N = 20 | 50 | 100 |
|---|---|---|---|---|
| s_a = 1, constant volatility | 0.3–1.8% | 0.995 | 1.00 | 1.00 |
| s_a = 1, SV-like | 0.5–1.5% | 0.985 | 1.00 | 1.00 |
| s_a = 0.5, SV-like | 0.5–1.5% | 0.45 | 0.95 | 1.00 |

**Geweke, E1 (Σ_t fixed and known; C = 1000 parallel SC chains; M₁ = 2×10⁵; G = 92: 12 coefficients, 78 products, log-likelihood, mean y²; Bonferroni critical value 3.87).** Entries are max |z|, with z of the log-likelihood in brackets.

| Configuration | Sampler | M_c = 25 (M₂ = 2.5×10⁴) | M_c = 100 | M_c = 400 |
|---|---|---|---|---|
| strong A, T = 50, SV-like, τ = 0.5 | exact | 3.20 | 2.35 | 2.20 |
| | CTA | 2.14 | 2.03 | 2.66 |
| | **TA-2019** | **41.9** (10.6) | 82.9 (17.0) | 138.8 (23.9) |
| strong A, T = 20, SV-like, τ = 1 | exact / CTA | 2.69 / 2.36 | 3.76 / 2.82 | 2.91 / 2.89 |
| | **TA-2019** | **35.1** (15.9) | 64.1 | 114.2 |
| strong A, T = 50, constant, τ = 0.25 | exact / CTA | 3.28 / 2.79 | 2.50 / 2.67 | 2.52 / 2.56 |
| | **TA-2019** | **37.1** (9.6) | 69.4 | 108.7 |
| A × 0.5, T = 50, SV-like, τ = 0.5 | **TA-2019** | **28.9** (−0.8) | 54.0 | 95.4 |
| A × 0.25, T = 50, SV-like, τ = 0.5 | **TA-2019** | **15.3** (2.1) | 29.6 | 51.9 |

**Geweke, E2 (full Gibbs: Π, A, λ unknown; T = 50, τ = 0.5; C = 2000; G = 190; critical value 4.04).** *Cross-check caveat:* G includes λ_i² and λ_iλ_j. Under λ_i ~ IG(3, 2) these have infinite variance (§3.4), so their z is not formally N(0, 1). The CTA's largest |z| at M_c = 200 (3.12) was in fact λ₃². The conclusions are unaffected, but the production suite must use log λ or a shape > 4 (§9.1).

| Sampler | M_c = 50 (M₂ = 10⁵) | M_c = 200 | M_c = 800 |
|---|---|---|---|
| CTA | 2.90 | 3.12 | 2.83 |
| **TA-2019** | **30.5** (log-lik z = 13.4) | 50.6 (29.0) | 80.4 (53.0) |

**SBC, E2 (full Gibbs, A and λ unknown; T = 50, τ = 1, a_ij ~ N(0, 1); burn-in 300, thinning 10, L = 99; pool of 7859 kept replications out of 8000).** Power, or size for the CTA, at α = 1%:

| Sampler | Test | N = 100 | 200 | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|
| CTA (correct; size) | any γ | 0.00 | 0.005 | **0.04** | 0.02 | 0.02 | 0.03 |
| | loglik alone | 0.02 | 0.005 | 0.035 | 0.035 | **0.115** | **0.265** |
| | params | 0.00 | 0.01 | 0.035 | 0.005 | 0.01 | 0.02 |
| TA-2019 | any γ | 0.01 | 0.04 | 0.33 | 0.76 | 1.00 | 1.00 |
| | loglik alone | 0.10 | 0.22 | 0.65 | 0.985 | 1.00 | 1.00 |
| | params | 0.005 | 0.01 | 0.055 | 0.13 | 0.43 | 0.975 |

**The CTA rows show a real pitfall: SBC false alarms from residual autocorrelation.**

- After thinning by 10, the pooled lag-1 autocorrelation of the CTA draws was 0.29 for log λ₁ and **0.61 for the log-likelihood**; most other quantities were below 0.1. The log-likelihood test then rejected the *correct* sampler 11.5% of the time at N = 2000 and 26.5% at N = 4000.
- The TA-2019 draws, being direct conditional draws of Π, had lag-1 autocorrelation ≈ −0.01. CCCM (2022, §2 and §4) state that the CTA's equation-by-equation blocking makes its draws more autocorrelated than the TA's. Their Fig. 2 compares CTA mixing per unit of time with the system-wide algorithm, not with the TA.
- A separate IAT study (`e2_iat.py`: 300 replications, 2000 unthinned sweeps, Geyer initial-positive-sequence IAT) explains the pattern. It was re-run in the cross-check and reproduced exactly (seeded):

| Quantity | Median IAT | 90th percentile | Maximum |
|---|---|---|---|
| Coefficients | 2–3 | 5–15 | 14–131 |
| Log-likelihood | 1.9 | 3.3 | **433** |
| a_ij and log λ_i | ≈ 1.2 | — | up to 639 |

  A handful of slowly mixing datasets dominate the pooled autocorrelation.

**Rule.** Choose thinning per replication from that replication's own pilot-chain IAT (two-pass procedure in §4.2, a clean version of Talts et al.'s Algorithm 2), with a cap t_max. Replications whose ESS is still below L are **rerun with a fresh, independent chain**. Never drop them, and do not extend the same chain after looking at it: dropping, and arguably extending, condition on sampler output.

**Rerun with thinning 40** (CTA, 3927 kept replications, burn-in 300, L = 99, `e2_full.py sbc_thin 50 1.0 40`): the size was back at or below nominal. Rejection rates for any γ / loglik / params:

| N | any γ | loglik | params |
|---|---|---|---|
| 100 | 0.010 | 0.005 | 0.005 |
| 200 | 0.010 | 0.010 | 0.010 |
| 500 | 0.005 | 0.005 | 0.005 |
| 1000 | 0.000 | 0.005 | 0.000 |
| 2000 | 0.000 | 0.000 | 0.000 |

The *pooled* lag-1 autocorrelation of the thinned log-likelihood was nevertheless 0.73, against 0.61 at thinning 10. A few sticky replications with large variance dominate that statistic, so it is a poor diagnostic. Use per-replication ESS of the thinned draws instead (rule in §4.2).

### 7.4 Conclusions

| Test | Cost to detect the CCM-2019 error | Comment |
|---|---|---|
| Density-ratio identity (§5.1) | **1 evaluation** | deterministic |
| Block-conditional SBC (§5.2) | **N = 20–50** | iid draws, so no MCMC cost |
| Geweke | **M₂ ≈ 2.5×10⁴ SC sweeps** (1000 chains × 25) gives \|z\| = 15–42 even for weak coupling (\|a\| ≈ 0.2) | Amplification: errors compound through y-resampling. Also ≈ 10⁵ sweeps with A and λ unknown (\|z\| = 30) |
| Prior SBC, with log-likelihood | **N ≈ 200–500** in the informative-prior, strong-coupling regime (Σ known); **N ≈ 1000** for the full MCMC (E2) | Undetectable with weak coupling (s_a = 0.5 at N = 2000; fixed A × 0.25 at N = 1000) or long T (T = 200). With MCMC, residual autocorrelation can create false alarms at the same N |
| Prior SBC, parameter ranks only | ≤ 26% power at N = 2000 | Nominal (no power) in the prior-dominated regime |

For this error class Geweke is about 20× cheaper than prior SBC in MCMC sweeps. In the matched configuration (fixed strong A, T = 50, SV-like, τ = 0.5), SBC needed N ≈ 500 replications × L = 99 ≈ 5×10⁴ exact draws, against 2.5×10⁴ SC sweeps for Geweke. That is only 2× in raw draws. With MCMC it becomes about 500 × (300 burn-in + 99 × 10) ≈ 6.5×10⁵ sweeps, about 25×. Geweke's margin is also much larger (|z| = 41.9 against a critical value of 3.87), so far fewer than 2.5×10⁴ sweeps would have sufficed. Gandy & Scott (2021, §4.2) likewise found Geweke strongest for errors that accumulate in the joint distribution.

SBC remains necessary for errors that leave the joint intact but distort conditionals, and for checking the predictive and latent-state paths.

---

## 8. Will the tests flag the SV mixture approximation or the offset?

**Per-observation divergence** between log χ²₁ and the mixtures, computed here by numerical integration; constants from Omori et al. 2007, Table 1:

| Mixture | Mean | Variance | KL(f‖g) | KL(g‖f) | Var_f(log f − log g) | max\|f − g\| |
|---|---|---|---|---|---|---|
| exact log χ²₁ | ψ(½) + log 2 = −1.2704 | π²/2 = 4.9348 | — | — | — | — |
| OCSN 10-component | −1.27028 | 4.93373 | 7.9e-5 | 6.8e-5 | 2.4e-4 | 3.8e-4 |
| KSC 7-component | −1.2704 | 4.93485 | 2.7e-3 | 2.4e-3 | 6.7e-3 | 1.0e-2 |

The max|f − g| values agree with Omori et al.'s Fig. 1, whose KSC panel has a vertical scale of ±0.010. Cross-check: the mixture constants were re-read from Omori et al. Table 1 (K = 10, and the KSC columns already shifted by −1.2704) and from KSC (1998) Table 4 (unshifted m_i, to which m_i − 1.2704 is applied). Both sources agree. KL(f‖g) and KL(g‖f) were recomputed by adaptive quadrature (`scipy.integrate.quad`, `xc06_check/kl_quad.py`): 7.94e-5 / 6.80e-5 (K = 10) and 2.69e-3 / 2.43e-3 (KSC-7), matching the grid values above. The exact log χ²₁ mean is ψ(½) + log 2 = −1.27036.

**SBC, E3.** Univariate SV with known (μ = 0, φ = 0.95, σ = 0.3), T = 100, N = 4000, L = 99, thinning 10. The sampler is s → h with a tridiagonal precision sampler, y\* = log y² with no offset. Test quantities: h₁, h_{T/2}, h_T, mean h, and the exact Gaussian log-likelihood.

| Sampler / data | Posterior Var(log w), mean (median) | Rejection rate (Bonferroni, 1%) at N = 500 / 1000 / 2000 | Full-pool p-values (N = 4000) |
|---|---|---|---|
| OCSN-10, Gaussian data | 0.0025 (0.0008) | 0.015 / 0.01 / 0.00 | min 0.040 |
| KSC-7, Gaussian data | 0.166 (0.154) | 0.01 / 0.00 / 0.01 | min 0.058 |
| OCSN-10, auxiliary-model data (exact control) | 0.0024 | 0.01 / 0.01 / 0.00 | min 0.19 |

Here w is the importance weight Π_t f(y\*_t − h_t)/g(y\*_t − h_t), the reweighting of Omori et al.

**Conclusions.**

- Full-model SBC at N ≤ 4000 will not flag the OCSN-10 approximation, or even KSC-7, on the latent states.
- With unknown (μ, φ, σ) the effect on σ could be larger. This is **UNVERIFIED**; the nightly suite reports the SV parameters separately.
- For Geweke with M₂ ≳ 10⁶ the KSC-7 error might become visible. Use OCSN-10 (as the SPEC does), and run SV-block tests with auxiliary-model data, where the target is exact.
- `stochvol` defaults to `correct_model_misspecification = FALSE`, i.e. auxiliary mixture sampling without correction (CRAN `svsample` documentation). Parity tests against `stochvol` therefore compare like with like.
- **Offset.** E3 used c = 0. The bias from a fixed offset grows with c/λ and must be tested where λ is small (pitfall 8).

---

## 9. Concrete test plan for Phase 2

### 9.1 Test configurations (all priors proper; values chosen for r ≤ 5 at the given T)

| Item | Per-PR ("small") | Nightly ("full") |
|---|---|---|
| n, p, deterministic terms | 2, 1, intercept (n = 3 for the reduced-form CTA compositions, see §11) | 3, 2, intercept |
| T; initial conditions | 20; y_{1−p:0} = 0 (fixed) | 50; fixed draw stored in the fixture |
| Structural mean | a_ij ~ N(0, 0.7²); γ_i \| a_i ~ N(Mã_i, V); M: own-lag (lag 1) **0.4**, else 0; V: intercept s.d. 1, own s.d. **0.10/ℓ**, cross s.d. **0.05/ℓ** (corrected from 0.5 / 0.15 / 0.075, which violate the explosive-mass bound; pitfall 5) | same, with ℓ = 1, 2 |
| Reduced form (CTA) | π^(j) ~ N(M_j, V) with the same M and V; **equation-asymmetric** V (own ≠ cross) | same |
| Constant | λ_i ~ **IG(5, 4)** (shape, scale; mean 1, s.d. 0.58), so that λ_i² has finite variance (shape > 4); alternatively keep IG(3, 2) and use only log λ_i in g (§3.4) | same |
| SV | μ_i ~ N(0, 0.5²); (φ_i + 1)/2 ~ Beta(10, 2) (φ mean 0.67, s.d. 0.21); σ²_i ~ 0.1·χ²₁ (stochvol-type parametrisation with B_σ = 0.1); h initial value from the stationary law (whatever convention the sampler uses). Simulated P(max_t \|h_t\| > 8) = 2e-4 at T = 20 and 50. Geweke g: no raw data moments, no h² unless b > 2 (§3.4) | same, plus a random-walk-SV variant with φ = 1 and h₀ ~ N(0, 1) |
| SV(outliers) | as SV, plus outlier probability p_i ~ Beta(2, 18) (mean 0.1) so that outliers occur at T = 20; o on the integer grid {2, …, 20} as in note 03 | same |
| SV(t) | as SV, plus ν_i discrete uniform on a sub-grid of the sampler's grid {3, …, 40} (note 03), restricted to **ν ≥ 5**, e.g. {5, …, 12} (the first version proposed a continuous U(3, 20), which does not match the discrete-grid sampler). Mass on ν ≤ 12 keeps the t path active. ν > 4 keeps var(ε²/ω) = var(e²/λ) finite, which matters if Σ_t ε̂²_{i,t} in g is standardised by λ only | same |
| Explosive mass | P(spectral radius > 1) < 10⁻³; checked in the fixture by simulation. With the corrected M and V and a_ij ~ N(0, 0.7²): ≤ 1e-6 (n = 2, p = 1), 7e-5 (n = 3, p = 1), 2e-4 (n = 3, p = 2), from 4×10⁵ draws in the cross-check | same |

The (φ, σ, μ) prior *types* follow `stochvol`:

- μ ~ N(b_μ, B_μ²);
- (φ + 1)/2 ~ Beta(a₀, b₀);
- σ² ~ B_σ·χ²₁.

The `stochvol` *defaults* (b_μ, B_μ) = (0, 100), (a₀, b₀) = (5, 1.5) and B_σ = 1 (CRAN reference manual, `svsample` usage and arguments; verified) are **too vague for Geweke** (§3.3). They may be used for SBC with data-only filtering. The Beta(10, 2) and B_σ = 0.1 values are design proposals to be tuned in the pilot so that r ≤ 5 for φ and σ (UNVERIFIED until the pilot). The λ range part is verified by simulation (row above).

### 9.2 Tiers

| Tier | Content | Size | Pass rule |
|---|---|---|---|
| **U** (unit, every PR) | Density-ratio identities for every block × composition (mean/contemporaneous, s, h, SV params incl. ASIS map, o, ω, ν, CTA π^(j)); MH acceptance-ratio identities; closed-form checks (§5.4); dtype is float64 | 20 random states per block; C = 4000 chains for the CTA closed-form check | relative error ≤ 1e-8; MCSE bands for §5.4 |
| **B** (block, every PR) | Block-conditional SBC (§5.2) for exact-draw blocks; one-block invariance tests (§5.3) for MH/ASIS blocks with L = 20 | N = 200, L = 99 | γ test with exact-recursion p-values (§4.3), Bonferroni, α = 1e-3, Gandy–Scott sequential (k = 3, Δ = 4, fresh keys per stage) |
| **G-small** (every PR) | Geweke for each composition: structural `Linear` × {Constant, SV, SV-outliers, SV-t} **and** reduced-form CTA × {Constant, SV, SV-outliers, SV-t}. That is 8 compositions, because SPEC §13 Phase 2 requires "Geweke and SBC green for every composition"; the first version listed only reduced × {Constant, SV}. Test functions obey the finite-variance rule (§3.4) | small config; C = 1000, M_c = 100 (10⁵ sweeps); M₁ = 2×10⁵; G ≈ 60–100 | Bonferroni α = 1e-3 with t_{C−1} critical values (t\* ≈ 4.44 at G = 100, C = 1000) + sequential |
| **G-full** (nightly) | as above | full config; C = 4000, M_c = 500; M₁ = 10⁶; G ≈ 150–300 | α = 1e-2 (z\* ≈ 3.99–4.15; t\* is larger by < 0.01 at C = 4000) |
| **S** (SBC, nightly) | Every composition with the log-likelihood, per-equation log-likelihoods, latent states, log prior and predictive unknowns h_{T+1}, y_{T+1} | full config; N = 1000, L = 99, burn-in 300; per-replication thinning from a pilot pass (§4.2), t_max = 50 | γ test (exact-recursion p-values), Bonferroni over Q, α = 1e-2; per-replication thinned ESS ≥ 0.8 L for ≥ 99% of replications |
| **M** (mutation, weekly) | Each mutant (§9.5) must fail the U, B or G tier | — | all mutants caught |
| **A** (analytic twin, weekly; SPEC §10.3) | MCMC `Linear(structural) × Constant` vs Chan (2022) exact iid at n = 20 | — | within 4 MCSE |

### 9.3 Run-time estimates

Measured by the first author with NumPy prototypes on an Apple M1 Pro, vectorised across chains or replications, with other jobs running concurrently. The component timings were not re-measured in the cross-check. The E2 total (190 s) matches the log `e2_geweke_T50.log`, and the re-run of `e2_iat.py` (300 chains × 2300 sweeps) took 74 s CPU, i.e. about 110 µs per chain-sweep including test-quantity evaluation.

| Component (n = 3, p = 1, T = 50) | Cost per chain per sweep |
|---|---|
| CTA Π block + conjugate (a, λ) | ≈ 75 µs |
| Simulate y | ≈ 6 µs |
| Tridiagonal h sampler (3 series) | ≈ 7 µs |
| 10-component indicator draw (3 series) | ≈ 40 µs |
| **VAR-SV sweep total** | **≈ 140 µs** (SV-parameter MH steps not measured) |

E2 Geweke (CTA): 1.6×10⁶ chain-sweeps plus 4×10⁵ MC draws took 190 s. E3 (univariate SV, T = 100): about 35 µs per series-iteration.

| Tier (per composition) | Sweeps | NumPy bound | JAX estimate (UNVERIFIED) |
|---|---|---|---|
| G-small (n = 2, T = 20; cost ≈ ⅓ of the above) | 10⁵ | ≈ 5 s | ≈ 1–3 s + 10–30 s compile |
| G-full | 2×10⁶ | ≈ 5 min | ≈ 1–2 min |
| S (pilot 1300 sweeps + production 300 + 99·t_i; worst case t_i = t_max = 50 for all) | ≈ 1.3×10⁶ + up to 5.3×10⁶ | ≈ 3 + up to 12 min (bucket replications by t_i to cut the worst case) | ≈ 1–5 min |
| B | 200 × 99 exact draws per block | < 1 s | — |

**Totals.** Eight compositions give about 1.5–4 minutes per PR (dominated by JIT compiles; corrected from "six compositions, about 1 minute"). Nightly runs take up to about 2.7 hours in NumPy terms in the worst case: 8 × (G-full ≈ 5 min + S ≤ 15 min). JAX `jit(vmap(scan))` should be 2–5× faster than these NumPy vectorised loops for such small ops (UNVERIFIED); calibrate in the Phase-2 pilot.

### 9.4 Harness design (JAX)

```python
# all functions pure, float64, keys split deterministically: fold_in(base, (tier, composition, chain|rep))
prior_sample(key, cfg)            -> State     # θ incl. latent h, o, ω (no auxiliary s)
simulate_obs(key, state, cfg)     -> Data      # uses state's latents, fixed y_init, rebuilds X
sweep(key, state, data)           -> State     # the production kernel, same code path as fit()
log_prior(state, cfg), loglik(state, data)     # independent reference implementations (§5.1)
test_functions(state, data)       -> (G,)      # §3.4; reused as SBC test quantities (§4.4)

geweke:  vmap over C chains of  lax.scan(m=1..M_c){ y ← simulate_obs; θ ← sweep; acc += g, g² }
         MC:  vmap over M₁ of  (prior_sample → simulate_obs → g), accumulated in chunks
sbc:     vmap over N reps of  { θ̃ ← prior; ỹ ← simulate_obs; keep = finite(ỹ) & max|ỹ|<1e3;
                                θ ← prior (independent init); scan(burn + L·thin) collecting f at thinned steps;
                                ranks with random tie-breaking; flag non-finite / Cholesky failure }
```

- Failures inside `vmap` cannot raise. Carry a per-chain error flag (failed Cholesky, non-finite state). A set flag **fails** the test with diagnostics, consistent with SPEC §8 (no silent jitter, no dropping).
- Keep accumulators and draws in float64.
- SBC memory is N × L × Q floats, for example 1000 × 99 × 40 × 8 B ≈ 32 MB.
- Compute γ p-values with the exact recursion (§4.3). Optionally cache a simulated null table per (N, L) under `tests/fixtures/`, with its seed, as a cross-check.
- Geweke accumulators: keep per-chain Σg (C × G) for the across-chain variance s²_C, plus per-batch sums for the batch-means cross-check.

### 9.5 Mutation suite (must all FAIL)

1. TA-2019 instead of CTA for `form="reduced"`. Caught by U (density ratio), B and G (§7).
2. DP-2015 order violated: s drawn before the mean block, or another block placed between s and h. Expected to be caught by G (Geweke error-5 class); **UNVERIFIED**, demonstrate in the pilot.
3. Observables simulator drawing fresh h (or ω, o) instead of using the state's latents (Geweke error 2).
4. Student-t ω conditional with dof ν instead of ν + 1 (Geweke error 4 analogue).
5. SV σ² prior scale B_σ off by a factor of 2 in the sampler only (prior mismatch; Geweke error 1 analogue).
6. ASIS step missing the Jacobian, or using the wrong centred/non-centred map.
7. The D3 conditional centring E[γ_i | a_i] = Mã_i replaced by M e_i, i.e. ignoring a_i.
8. Offset c = 0.1 in log(u² + c) at λ ≈ 0.01.

Expected-catch tiers for mutants 2–8 are design targets and must be confirmed empirically.

---

## 10. Parity oracles (test-relevant mapping only)

- **`stochvol`** (CRAN `svsample` documentation):
  - priors μ ~ N(priormu[1], priormu[2]²) (the second element is a **standard deviation**), (φ+1)/2 ~ Beta(priorphi), σ² ~ priorsigma·χ²₁ (equivalently σ ~ |N(0, priorsigma)|, i.e. σ² ~ Gamma(½, rate 1/(2·priorsigma)));
  - defaults: `priormu = c(0, 100)`, `priorphi = c(5, 1.5)`, `priorsigma = 1`, `priornu = 0` (no t), `priorlatent0 = "stationary"`, `thin = 1`, `burnin = 1000`, `draws = 10000`. `interweave = TRUE` (ASIS) and `correct_model_misspecification = FALSE` are `expert` list elements. All verified in the reference manual;
  - **initial state:** `priorlatent0 = "stationary"` puts the stationary law on **h₀**, which stochvol samples as an extra latent, with h₁ | h₀ AR(1). The implied law of h_{1:T} equals that of a stationary-h₁ convention, so compare h_{1:T} only. `priorlatent0 = B` gives h₀ ~ N(μ, Bσ²);
  - `y` must not contain zeros.

  Parity runs (SPEC §10.4) use the same prior values in both implementations, and compare posterior quantiles within 4 MCSE. **Corrected:** quantile MCSE must use the quantile-specific method of Vehtari et al. (2021, §4.3–4.4, eq. 20): take the ESS of the indicator I(θ ≤ θ̂_α), and the Beta(S_eff α + 1, S_eff(1−α) + 1) quantiles mapped through the empirical CDF. Bulk ESS is wrong for quantiles, because tail quantiles are often much less efficient than the mean. Use bulk-ESS MCSE only for posterior means.
- **`bayesianVARs`** implements the corrected triangular algorithm (SPEC §2.2). Its prior parametrisation mapping belongs to the reduced-form note. Geweke/SBC on our side do not depend on it.
- **Oracles vs tests.** Oracles check agreement on a few datasets. Geweke/SBC check correctness over the prior. They complement each other: an oracle shares any modelling convention we copied, while Geweke/SBC share only our prior and observables simulators.

---

## 11. Edge cases specific to the components

- **Discrete or mixed quantities** (o_{i,t} ∈ {1} ∪ continuous support, ν on a grid, outlier counts): use random tie-breaking in ranks (§4.1). For Geweke use indicator means and second moments.
- **Random-walk SV** (φ = 1): needs a proper h₀ prior. Geweke r for h₀ is driven by the h₀ prior variance, so keep it at about 1.
- **Reduced form with n = 2**: the CTA for j = 1 uses equations 1–2. Test n ≥ 3, so that the j = 1 step uses three equations and the "middle" equation case (j = 2 < n) exists.
- **SV-t with ν → ∞, or outliers with p → 0**: the code path is unexercised. The test priors keep these mechanisms active.
- **Ordering**: tests run for the default ordering and one permutation, because the recursive form is ordering-dependent (SPEC R8). Geweke/SBC validity does not depend on the ordering, but bugs may be index-specific (e.g. off-by-one in `j..n`).

---

## 12. UNVERIFIED items

1. The BACC estimator of τ²_g used by Geweke (2004). It is believed to be a tapered spectral estimate with 4/8/15% tapers, but the paper text does not state it. The 4/8/15% tapers are verified only for the CCCM (2022) Fig. 2 inefficiency factors.
2. *(Resolved in the cross-check.)* Flegal & Jones (2010) and Geyer (1992) bibliographic details were verified via Crossref and Project Euclid (§13). The precise consistency conditions of Flegal & Jones are not restated.
3. JAX run-times: §9.3 gives NumPy measurements; the JAX figures are estimates for the Phase-2 pilot to confirm. The NumPy component timings were not re-measured.
4. Test-prior values are design proposals:
   - SV: Beta(10, 2) for (φ+1)/2, B_σ = 0.1, μ s.d. 0.5.
   - SV(outliers): Beta(2, 18) outlier probability, o on {2, …, 20}.
   - SV(t): ν discrete uniform on {5, …, 12}.
   - Constant: IG(5, 4).
   - Mean: the corrected mean prior.

   Explosive mass and the λ range were checked by simulation in the cross-check. That r ≤ 5 holds for φ, σ, p and ν is **not** verified (pilot).
5. The effect of the mixture approximation with **unknown** SV parameters (σ in particular) on SBC/Geweke was not measured.
6. Expected catching tiers for mutants 2–8 (§9.5); only mutant 1 was measured.
7. Applicability of the Gandy–Scott exact rank test (which needs reversibility) to the PCG-ordered SV sweep or a palindromic scan.
8. The power numbers in §7–§8 come from this note's NumPy prototypes (single seed per configuration; power from 200–400 subsamples of one pool, so the estimates for large N share most replications). They show orders of magnitude, not exact rates for the JAX implementation. The cross-check re-ran `check_flat.py`, `check_cta.py`, `e3_mixture_kl.py` and `e2_iat.py`, which reproduced exactly. It compared every table in §5.2, §7.3 and §8 against the logs, and they match. The other experiments were not re-run.
9. *(Resolved.)* Talts et al. section numbers are now given (§4.1, §4.2, §5.1, §6, §7; arXiv v2). Theorem and equation numbers for Modrák et al. are from arXiv v3 (19 Oct 2023); the journal version (*Bayesian Analysis* 20(2):461–488, 2025) may renumber.
10. ArviZ's `ess(method="mean")` implementing Geyer's initial monotone sequence: the ArviZ source was not re-read. The estimator itself is specified from Vehtari et al. (2021, §3.2, eq. 13).
11. The full text of Bognanni (2022) was not accessed; only its abstract (IDEAS/RePEc) was read.
12. The exact-recursion γ p-value (§4.3) was implemented and checked against simulation at q0.05 and q0.01. Its far-tail values (q0.001, q0.0001) have no independent simulation check beyond the earlier 2×10⁴-draw table.

---

## 13. References (with the locations used)

- Geweke, J. (2004). Getting it right: Joint distribution tests of posterior simulators. *JASA* 99(467):799–804. DOI 10.1198/016214504000001132. Preprint (Feb 2004) used for equation numbers: http://qed.econ.queensu.ca/pub/faculty/ferrall/quant/papers/04_04_29_geweke.pdf. Used: §2 eqs (1)–(6), §3 Table 1, §4 Table 2 (Bonferroni 4.394), §5 Tables 4–6.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788v2: https://arxiv.org/abs/1804.06788. Used: §3 (critique of Geweke), §4.1 Theorem 1, Algorithm 1 and binomial band, §4.2 histogram shapes, §5.1 Algorithm 2 and thinning, §6 settings, §7 (χ² remark).
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejsková, K., Gelman, A., Vehtari, A. (2025). Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. *Bayesian Analysis* 20(2):461–488. DOI 10.1214/23-BA1404. arXiv:2211.02383v3: https://arxiv.org/abs/2211.02383. Volume/issue/pages verified via Crossref (published June 2025). Used: §1.1 eqs 1–3, §1.2 ranks and ties, eq. 7 (γ), Theorems 5–7 (§3.2–3.4), §3.5, §6.1–6.2.
- Säilynoja, T., Bürkner, P.-C., Vehtari, A. (2022). Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison. *Statistics and Computing* 32(2), article 32. DOI 10.1007/s11222-022-10090-6 (verified via Crossref). arXiv:2103.10522v2: https://arxiv.org/abs/2103.10522. Used: eqs 8, 10–13 (§2.1–2.2), 15–24 (§2.3, recursion and Brent), §3, §4.2.3, Appendix A.
- Säilynoja, T., Schmitt, M., Bürkner, P.-C., Vehtari, A. (2025). Posterior SBC: Simulation-based calibration checking conditional on data. arXiv:2502.03279: https://arxiv.org/abs/2502.03279.
- Gandy, A., Scott, J. A. (2021). Unit testing for MCMC and other Monte Carlo methods. arXiv:2001.06465v2: https://arxiv.org/abs/2001.06465. Not found in Crossref as a journal article; cited as arXiv v2. Used: Algorithms 1–3 (§2.1, §2.2, §3), Proposition 2.3, Theorem 3.1, eqs (1)–(2) of §3, §4.1–4.2, Tables 1–2.
- Grosse, R. B., Duvenaud, D. K. (2014). Testing MCMC code. arXiv:1412.5218: https://arxiv.org/abs/1412.5218. Used: eq. (1), §4.
- Cook, S. R., Gelman, A., Rubin, D. B. (2006). Validation of software for Bayesian models using posterior quantiles. *JCGS* 15(3):675–692, DOI 10.1198/106186006X136976. Gelman, A. (2017). Correction to Cook, Gelman, and Rubin (2006). *JCGS* 26(4):940, DOI 10.1080/10618600.2017.1377082. Both verified via Crossref; background only.
- Besag, J., Clifford, P. (1989). Generalized Monte Carlo significance tests. *Biometrika* 76(4):633–642, DOI 10.1093/biomet/76.4.633 (verified via Crossref).
- Carriero, A., Chan, J., Clark, T. E., Marcellino, M. (2022). Corrigendum to "Large Bayesian VARs with stochastic volatility and non-conjugate priors". *J. Econometrics* 227(2):506–512, DOI 10.1016/j.jeconom.2021.11.010 (verified via Crossref). WP version (Cleveland Fed WP 16-17C, Nov 2021, DOI 10.26509/frbc-wp-201617corrigendum): https://www.clevelandfed.org/publications/working-paper/2021/wp-1617c-large-vector-autoregressions-with-stochastic-volatility-and-flexible-priors. Used: eqs (3)–(8), (12)–(15), footnote 2; §2 and §4 (CTA about 3.25× slower than TA at N = 20; CTA draws more autocorrelated than TA; Fig. 2 IF tapers 4/8/15%); §5 (empirical results "did not change appreciably").
- Bognanni, M. (2022). Comment on "Large Bayesian VARs with stochastic volatility and non-conjugate priors". *J. Econometrics* 227(2):498–505, DOI 10.1016/j.jeconom.2021.10.008 (verified via Crossref). Abstract only, via https://ideas.repec.org/a/eee/econom/v227y2022i2p498-505.html.
- Omori, Y., Chib, S., Shephard, N., Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *J. Econometrics* 140(2):425–449, DOI 10.1016/j.jeconom.2006.07.008 (verified via Crossref). WP: https://www.nuff.ox.ac.uk/economics/papers/2004/W19/mixture28.pdf. Used: Table 1 constants (KSC and K = 10) and Fig. 1.
- `stochvol` `svsample` reference manual (CRAN): https://search.r-project.org/CRAN/refmans/stochvol/html/svsample.html. Used: default arguments, prior definitions (`priormu`, `priorphi`, `priorsigma`, `priorlatent0`) and `expert` defaults (read from the reference-manual PDF in the scratchpad, `dl_stochvol_docs/refman.txt`).
- Kastner, G., Frühwirth-Schnatter, S. (2014); Del Negro, M., Primiceri, G. (2015): as in SPEC §15.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis* 16(2):667–718, DOI 10.1214/20-BA1221 (verified via Crossref). arXiv:1903.08008v5. Used: §3.2 eq. 13 (Geyer initial positive/monotone sequence), §4.3 (quantile ESS, tail-ESS), §4.4 eq. 20 (quantile MCSE).
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Review of Economic Studies* 72(3):821–852, DOI 10.1111/j.1467-937X.2005.00353.x (verified via Crossref). Used: appendix, offset c = 0.001 (read from the copy in `dl_primiceri/`).
- Flegal, J. M., Jones, G. L. (2010). Batch means and spectral variance estimators in Markov chain Monte Carlo. *Ann. Statist.* 38(2):1034–1070, DOI 10.1214/09-AOS735 (verified via Crossref and Project Euclid; arXiv:0811.1729).
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statist. Sci.* 7(4):473–483, DOI 10.1214/ss/1177011137 (verified via Crossref and Project Euclid; 473–511 including the discussion).

---

## 14. Cross-check log

This is an independent cross-check, done 2026-10-21. I read every primary source again from the PDF or text:
- the Geweke 2004 preprint;
- Talts et al. arXiv v2;
- Modrák et al. arXiv v3;
- Säilynoja et al. arXiv v2;
- Gandy & Scott arXiv v2;
- Grosse & Duvenaud arXiv v1;
- Säilynoja et al. 2025 arXiv v1;
- the CCCM corrigendum (Cleveland Fed WP 16-17C);
- Vehtari et al. arXiv v5;
- Omori et al. WP Table 1 and KSC 1998 Table 4;
- the `stochvol` CRAN reference manual;
- Primiceri 2005.

Downloads are in `phase2-research/xc06_dl/`. Bibliographic data comes from Crossref (`xc06_dl/crossref/`), with web search as the second source. Check scripts are in `phase2-research/xc06_check/`.

### 14.1 Verified without change

**Geweke (2004)**
- eq. (6) and the MC/SC simulators (eqs 3–5);
- the proper-prior condition and the 0.01/σ² ~ χ²(4) replacement;
- §3 settings: T = 6, ν = 5, 2.5×10⁵ draws each, Table 1 counts;
- §4 settings: T = 10, M₁ = 10⁶, M₂ = 4×10⁶ every 400th, 90 functions, 4.394, factor-of-ten Bayes factor;
- §5 settings: T = 6, every 40th of 4×10⁶, τ = 0 overflow, two errors.

**Constants and derivations**
- All Bonferroni critical values in §0, recomputed.
- The κ = 1 − 1/r, IAT = 2r − 1 derivation, re-derived.
- Holm equals Bonferroni for the global decision.

**Talts et al.**
- Theorem 1, Algorithms 1–2, thinning by ⌈N/N_eff⌉ with the largest value over quantities;
- the band [33, 69] for N = 1000, B = 20;
- L = 100, N = 10⁴ / 10³, N/B ≈ 20, L = 1023;
- the histogram shapes.

**Modrák et al.**
- ties via K ~ U{0..N_eq} (§1.2);
- eq. 7 for γ, with i = L + 1 trivial;
- Theorems 5, 6 and 7 as described;
- the §6.1–6.2 recommendations;
- the §3.5 non-equivalence with the data-averaged posterior.

**Säilynoja et al.**
- eq. 13 (γ), eqs 11–12 (bands).

**Gandy & Scott**
- Algorithms 1–3, Proposition 2.3;
- the systematic-scan rank-test size of 0.769;
- Geweke size 0.441–0.523 (Table 1) and 0.101 (Table 2), and power 0.909 against ≤ 0.018 (Table 2);
- the 30% to 97–99% power gain;
- `mcunit` defaults, γ ≈ 0.146, the β sequence and 68.5%.

**Grosse & Duvenaud, CCCM and Omori**
- Grosse & Duvenaud eq. (1).
- CCCM eqs (5)–(8) and (13)–(15), the CTA formula in §7.1, and "about 3.25×" at N = 20.
- The CTA closed-form SUR precision and linear term (§5.4), re-derived.
- Flat-prior exactness of TA-2019 (§7.2), re-derived; `check_flat.py` re-run gives identical numbers.
- `check_cta.py` re-run.
- Omori K = 10 and KSC-7 constants, from two primary tables.
- KL values, by quadrature.

**stochvol, priors and experiment logs**
- `stochvol` defaults, from the reference manual.
- Primiceri c = 0.001.
- P(ρ > 1) = 4.2% for the E1/E2 prior.
- IG(3, 2) has mean 1; Beta(2, 18) has mean 0.1.
- Every number in the §3.3, §5.2, §7.3 and §8 tables against the experiment logs (`e1_*.log`, `e1b.log`, `e2_*.log`, `e2_*.json`, `e3_sv_sbc.log`).
- `e2_iat.py` re-run reproduced the IAT table exactly.

### 14.2 Corrections

| # | Where | Was | Now | Source / evidence |
|---|---|---|---|---|
| 1 | §3.4, §6 (16), §9.1, D-2 | Raw data functions (tr(Y'Y), eigenvalues, y², max\|y\|) and squares of all parameters recommended for every composition | **Finite-variance rule.** Under the SV test prior, E[y²] = ∞, because σ² has a χ² tail and φ has mass near 1. λ² under IG(3, 2) has infinite variance (needs shape > 4). h² under Beta(·, 2) is borderline. Student-t needs ν > 4. Use log-scale or bounded data functions in SV compositions, and IG(5, 4) or log λ | Geweke (2004) eq. (2): σ²_g < ∞ is required. `xc06_check/heavy_g.py`: batch means of mean y² from 2.3 to 5.2×10³; IG(3, 2) λ² batch s.d. from 11.6 to 299 |
| 2 | §3.2, §0, D-2 | "C ≥ 200: the t_{C−1} correction is negligible; use the normal" | Use t_{C−1} critical values and C ≥ 1000. At C = 200 the normal test has 1.4–1.6× the nominal Bonferroni size | `xc06_check` computation with `scipy.stats.t` |
| 3 | §6 (5), §9.1 | Own-lag mean 0.5, s.d. 0.15/0.075 meets P(ρ > 1) < 10⁻³ | It does not: 0.10–0.48% in reduced form, 0.26–1.8% in structural D3 (A⁻¹ amplification). Corrected to mean 0.4, s.d. 0.10/ℓ (cross 0.05/ℓ), giving ≤ 2e-4 | `xc06_check/explosive.py`, `explosive2.py` (4×10⁵ draws) |
| 4 | §0, §4.3, D-6 | γ null quantiles from 2×10⁴ simulations; "with discrete ranks, simulation is simplest and exact" | Exact forward recursion (Säilynoja eqs 15–23) for discrete ranks. New table: old q0.001 entries were off by up to 20% (N = 200: 4.3e-5 → 5.2e-5; N = 1000: 4.2e-5 → 3.7e-5), and N = 200 q0.01 was 5.4e-4 → 5.7e-4. Simulation cannot resolve the α/Q tails | Säilynoja et al. 2022 §2.3; `xc06_check/gamma_exact.py`, and `constants.py` (10⁵-draw simulation) |
| 5 | §4.6 | Two-sample test "compare the final θ_L with fresh prior draws" | It compares joint (θ, y) samples. In Table 1 only the likelihood statistic detected the variance error | Gandy & Scott §2.1, Algorithm 1, Table 1 |
| 6 | §4.6 | Sample size multiplied by Δ at each stage (ambiguous); extra effort "—" | Δ is applied once, after stage 1. Stages must be independent (fresh keys). Extra effort 29.7% / 29.9% (per PR) and 68.7% / 71.6% (nightly), from eqs 1–2 | Gandy & Scott Algorithm 3 line 9, Theorem 3.1, §3 eqs (1)–(2) |
| 7 | §4.6 | Inflated Geweke size in G&S "reflects naive SEs and short runs" (interpretation) | Added G&S's own attribution (data–parameter correlation), the r ≈ 2000 quantification, and their result that a longer Geweke run had nominal size and 100% power | Gandy & Scott §4.1 text, end of §4.2; derivation §3.3 |
| 8 | §3.6 | (no caveat) | Table 1 swaps the labels of errors 3 and 4 relative to the text | Geweke 2004 preprint pp. 4–5 |
| 9 | §3.4 | Eigenvalues of Y'Y "most sensitive (z = 16–20)" | z = 16.5, 20.3, 16.3, 6.6 in Table 6; λ₂ = 7.76 largest in Table 5 | Geweke 2004 Tables 5–6 |
| 10 | §3.4 | G&D: "SC chain's variance grows by orders of magnitude" | The sampled σ²_n grows from about 1 to about 10⁵–10⁶ in 1000 steps; the bug is 0.51 vs 0.5 in the IG scale | Grosse & Duvenaud §3.1, §4 |
| 11 | §3.3 | "explains why Geweke used T = 6 and 10" | "consistent with" (Geweke gives no rationale), plus the G&D quote on fewer data points | Geweke 2004; Grosse & Duvenaud §4 |
| 12 | §3.3 | μ information "≈ 1/4.93", r ≈ 10⁴T/4.93 | Upper bound I_μ ≤ T/4.93; ≈ T/(π²/2 + σ²/(1−φ)²) for persistent h; r = 1 + B_μ² I_μ | Derivation (log χ²₁ variance π²/2) |
| 13 | §3.3 | "the two IATs roughly multiply" | Derived: κ_tot = 1 − (1−κ)(1−ρ), IAT_tot ≈ IAT_SC · IAT_inner / 2. IAT(θ²) formula needs prior mean 0 | Derivation |
| 14 | §4.2 | Säilynoja §4.2.3 cited for SBC autocorrelation | §4.2.3 is the multi-chain test (it says the same holds for other uniformity tests); added Appendix A and Talts' quantile-indicator ESS rule | Säilynoja et al. §4.2.3, Appendix A; Talts §5.1 |
| 15 | §4.2 | Data-only filtering justified by "Theorem 5" | Justified by eq. (3) and the text after Theorem 5 ("rejection sampling") | Modrák et al. §1.1, §3.2 |
| 16 | §7.3 | "CCCM (§4) report that the correct algorithm mixes worse than the original" | CCCM state that CTA draws are more autocorrelated than TA's; their Fig. 2 compares CTA with the system-wide algorithm, not with TA | CCCM 2022 §2, §4, Fig. 2 |
| 17 | §7.2 | Flat-prior exactness without conditions; (c) "Bognanni's finding … holds for any proper prior" | Requires shared regressors **and time-invariant A**; explicit form of the missing term added. (c) restricted to priors independent across reduced-form equations, with some a_ij ≠ 0; Bognanni quoted from the abstract only | CCCM eqs (7), (12), footnote 2; Bognanni (2022) abstract (IDEAS/RePEc); derivation |
| 18 | §7.4 | "Geweke about 20× cheaper" | Clarified: 2× in raw draws, about 25× in MCMC sweeps, plus a large z margin | Arithmetic on §7.3 settings |
| 19 | §9.1, §2.1 | ν ~ U(3, 20) (continuous); outlier support unspecified | ν discrete uniform on {5, …, 12} ⊂ sampler grid {3, …, 40}; o on {2, …, 20} | Note 03, §0 decisions table, rows "Prior on ν" and "Outlier scale `o_t`" |
| 20 | §9.1 | λ_i ~ IG(3, 2) | IG(5, 4), or log λ in g | Correction 1 |
| 21 | §9.2–9.3 | 6 compositions (reduced × {Constant, SV} only); "about 1 minute per PR" | 8 compositions; about 1.5–4 min per PR; nightly up to about 2.7 h (NumPy terms) | SPEC §13 Phase 2: "Linear (structural form, and reduced form …) × {Constant, SV, SV(outliers), SV(t)}"; exit "Geweke and SBC green for every composition" |
| 22 | §10 | Quantile MCSE "from bulk ESS" | Quantile-specific MCSE: ESS of I(θ ≤ θ̂_α) and Beta quantiles | Vehtari et al. 2021 §4.3–4.4, eq. 20 |
| 23 | §10, §5.4 | stochvol initial-state convention not stated | `priorlatent0 = "stationary"` applies to h₀, which is sampled; the law of h_{1:T} is the same as with stationary h₁; `interweave` and `correct_model_misspecification` are `expert` elements | `stochvol` reference manual, `svsample` arguments |
| 24 | §6 (14) | "Geweke's z needs an ergodic SC chain" | With parallel prior-started chains, size needs only invariance; ergodicity affects power | Derivation from §3.1–3.2 |
| 25 | §3.2 | Batch means "consistent for b ∝ M^{1/2} (from memory)" | What Flegal & Jones prove (strong and mean-square consistency, optimal batch-size rate); b = ⌊M^{1/2}⌋ as convention; citation verified | Flegal & Jones 2010 (abstract via Project Euclid/arXiv), Crossref |
| 26 | §12, §13 | Flegal & Jones, Geyer marked UNVERIFIED; several DOIs missing | Verified: Geweke, Modrák 20(2):461–488, Säilynoja 32(2) art. 32, Bognanni 227(2):498–505, CCCM 227(2):506–512 (DOI 10.1016/j.jeconom.2021.11.010), Cook 2006, Gelman 2017, Besag & Clifford, Omori, Vehtari, Flegal & Jones, Geyer, Primiceri. Gandy & Scott: no journal version found | Crossref API (`xc06_dl/crossref/*.json`), web search |
| 27 | §5.4 | Banded h-sampler: precision only | Added the linear term Q·μ1 + (y\* − m_s)/v_s and the first-entry derivation | Derivation |
| 28 | §7.3 E2 | (no caveat) | E2's G includes infinite-variance λ² terms; the CTA's maximum \|z\| at M_c = 200 was λ₃² | Correction 1; argmax index 188 = λ₃·λ₃ |
| 29 | §4.3–4.5 | Talts section numbers missing | §4.1, §4.2, §5.1, §6, §7 added | Talts et al. arXiv v2 |

### 14.3 Still UNVERIFIED after the cross-check

See §12. In short:
- the BACC τ² estimator;
- JAX run-times;
- r ≤ 5 for the SV, outlier and t test priors;
- the mixture-approximation effect with unknown SV parameters;
- the catching tier of mutants 2–8;
- the Gandy–Scott rank test on the PCG-ordered SV sweep or a palindromic scan;
- single-seed experiment rates (most were not re-run);
- the ArviZ implementation detail;
- the Bognanni full text;
- the far-tail γ quantiles beyond the recursion itself.
