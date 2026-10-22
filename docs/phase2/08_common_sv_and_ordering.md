# 08: Common drifting volatility (`CommonSV`), variable-ordering dependence, and the Clark–Ravazzolo (2015) replication target

Phase 2 research note for `rai-bvar`. Relevant SPEC sections: §3 R2/R8/R9, §4.1–4.6, §6.6, §7, §8, §10.3–10.7, §13 Phases 2–3, decisions D2/D3.

It covers three topics:

- **(A)** Carriero, Clark & Marcellino (2016, JBES) common stochastic volatility, `Σ_t = f_t Σ`: model, priors, conditionals, sampler, tests. `CommonSV` ships in Phase 3 (decision D2). It is specified here because its `(Π, Σ) | h` block is the Phase 1 analytic engine with GLS row weights, and because it is the cheapest **order-invariant** SV model.
- **(B)** What depends on variable ordering in triangular (Cholesky) SV models, following Arias, Rubio-Ramírez & Shin (2023) and Chan, Koop & Yu (2024): how large the effects are empirically, and a concrete specification of `rb.ordering_sensitivity` (SPEC §6.6).
- **(C)** The Clark & Ravazzolo (2015) set-up behind the Phase 2 acceptance target (SPEC §10.7): data, models, evaluation design, scores, and the headline numbers to reproduce qualitatively.

**Clean-room status.** Everything below is specified from papers (author-hosted working papers or accepted manuscripts) and package *documentation*:

- No GPL source code was opened. `bvars` and `bayesianVARs` were consulted only through their manuals and READMEs.
- Formulas marked *derived here* are my own algebra and need a test (§A11, §B5) before anyone relies on them.
- Anything I could not verify is marked **UNVERIFIED** and collected in §D.
- An independent cross-check pass (2026-10-21) re-verified every constant and citation against the primary texts and corrected the note in place. Its changes are listed in the **Cross-check log** at the end.

---

## 0. Summary of decisions (recommended defaults)

| Item | Recommendation | Source |
| :--- | :--- | :--- |
| `CommonSV` model | `u_t ~ N(0, e^{h_t} Σ)`, `h_t = ρ h_{t−1} + σ_h η_t`, `E[h_t] = 0` (identification), stationary start `h_1 ~ N(0, σ_h²/(1−ρ²))` | Chan (2020) eq. (2)–(3) and fn. 1; CCM (2016) eq. (12)–(14) (no intercept, AR(1)) |
| `(Π, Σ)` prior | Natural-conjugate NIW from Phase 1: `Σ ~ IW(S₀, ν₀)`, `vec Π | Σ ~ N(vec M, Σ ⊗ V)`, Minnesota `V` with `cross = 1`, `ν₀ = n + 2`, `S₀ = diag(ψ)` from in-window AR(p) residual variances | CCM (2016) §2.4, §3.1; Chan (2020) §4.2; GLP (2015) |
| `(Π, Σ) | h` step | **Exact joint NIW draw** on GLS-rescaled rows `ỹ_t = e^{−h_t/2} y_t`, `x̃_t = e^{−h_t/2} x_t`, reusing the Phase 1 QR/dummy-observation engine (the Lenza–Primiceri row-scaling path with `s_t² = e^{h_t}`). Draw `Σ` marginally, then `Π | Σ` | Chan (2020) §3, eq. (5) ff. Do **not** implement CCM (2016) eq. (23) as printed (see §A10) |
| `h` step | **Exact independence-chain MH** with a Gaussian (Laplace) proposal from Newton iterations on a tridiagonal precision, using the sufficient statistic `q_t = u_t'Σ⁻¹u_t`. Newton must start from a point that does not depend on the current `h` (e.g. `log(q_t/n)`), so the proposal is a genuine independence proposal. For long samples (`T ≳ 300`) or large `σ_h²`, acceptance falls (simulated here: about 0.3 at `T = 300, σ_h² = 0.1`; 0.05–0.2 at `T = 600–1000`), so use the **blocked** version (§A5.2). Exactly order-invariant and exact in Geweke tests. The Omori-mixture variant with `n` pseudo-observations per `t` (CCM 2016) is an optional parity mode | Chan (2020) §4.3 ("independence-chain Metropolis-Hastings step as in Chan (2017)"); Online App. A2, Example 2 describes the `h` step as an "acceptance-rejection Metropolis-Hastings" (ARMH) step with the same Gaussian proposal `N(ĥ, K_h⁻¹)`. Plain independence MH, as specified here, is also exact; ARMH typically accepts more often at extra cost; CCM (2016) §2.6 |
| `ρ`, `σ_h²` priors | `ρ ~ N(0.9, 0.2²)·1(|ρ|<1)`; `σ_h² ~ IG(5, 0.04)` (shape, scale), so `E σ_h² = 0.01` | Chan (2020) §4.2. CCM (2016) §3.1 uses the same centre for `ρ` (their `ψ`, no truncation stated) and `φ ~ IG_Clark(4·0.01, 4)` |
| Ordering in triangular SV | Always recorded (R8). Ship `ordering_sensitivity` with modes `"prior"` (cheap), `"forecast"` (default), `"backtest"`; balanced Latin-square permutation design; a Monte Carlo noise floor from repeated seeds; optional `pool=True` that mixes predictives over orderings | §B4 (design is *recommended here*, not taken from a paper) |
| Order-invariant constant-volatility prior | Offer `Constant(prior="iw_equivalent")`: `λ_i ~ IG((ν₀+i−n)/2, s_i²/2)`, `a_ij | λ_i ~ N(0, λ_i/s_j²)`. This induces `Σ ~ IW(ν₀, diag s²)` exactly | Chan (2022, QE) Prop. 1 and Cor. 1–3 |
| Phase 2 replication | US quarterly `[GDP growth, GDP-deflator inflation, unemployment, 3-m T-bill]`, p = 4. `Linear × Constant` vs `Linear × SV` (RW and AR) vs `Linear × SV(tails="t")`. Recursive origins 1985Q1–2011Q2 (also 1985Q1–2007Q4), h = 1, 2, 4, 8. Scores: RMSE, LPD, CRPS | Clark & Ravazzolo WP 12-18 §2–§4; JAE data archive |

---

## 1. Notation

| Symbol | Meaning | Shape |
| :--- | :--- | :--- |
| `n, p, k` | number of variables, lags, regressors per equation (`k = np + n_d`) | ints |
| `T` | effective estimation sample length | int |
| `Y, X` | stacked data, rows `y_t'`, `x_t'` | `T×n`, `T×k` |
| `Π` | reduced-form coefficients (`= B` in SPEC §4.1, CCM's `Π`, Chan's `A`) | `k×n` |
| `u_t` | reduced-form residual `y_t − Π'x_t` | `n` |
| `Σ` | constant covariance scale matrix in `CommonSV` | `n×n` |
| `L_Σ` | lower Cholesky factor, `Σ = L_Σ L_Σ'` | `n×n` |
| `h_t` | common log-volatility, `f_t = e^{h_t}` (CCM write `λ_t` or `f_t`) | `T` |
| `ρ, σ_h²` | AR(1) persistence and innovation variance of `h` (CCM: `ψ`, `φ`) | scalars |
| `D` | `diag(e^{h_1}, …, e^{h_T})` (Chan's `Ω`) | `T×T` |
| `q_t` | `u_t' Σ⁻¹ u_t = ‖L_Σ⁻¹ u_t‖²`. In structural form, `Σ_i e_{i,t}² / λ_i` | `T` |
| `M, V` | NIW prior mean and column covariance of `Π` | `k×n`, `k×k` |
| `S₀, ν₀` | IW prior scale and degrees of freedom | `n×n`, scalar |
| `K, Π̂, Ŝ, ν̂` | NIW posterior quantities given `h` (§A5.1) | — |
| `A, λ_{i,t}` | unit-lower-triangular `A` and structural variances (SPEC §4.1): `Σ_t = A⁻¹ Λ_t A⁻¹'` | — |
| `a_i` | free elements of row `i` of `A` (`i−1` of them) | — |
| `π` | a permutation of `{1, …, n}`; `P_π` is its permutation matrix | — |
| `IG(α, β)` | inverse gamma, density `β^α/Γ(α) · x^{−α−1} e^{−β/x}` (shape α, scale β), as in the sibling notes | — |
| `IG_Clark(a, d)` | Clark/CCM convention, "scale term `a`, degrees of freedom `d`": `a/x ~ χ²_d`, equivalently `IG(d/2, a/2)` | — |
| `IW(S, ν)` | inverse Wishart, density `∝ |Σ|^{−(ν+n+1)/2} exp(−½ tr(S Σ⁻¹))`, `E Σ = S/(ν−n−1)` | — |
| `MN(M, Σ, V)` | matrix normal: `vec Π ~ N(vec M, Σ ⊗ V)`, where `Σ` is the covariance across equations (columns) and `V` across regressors (rows) | — |

---

# Part A: Common drifting volatility (CCM 2016)

## A1. Model

Published version (CCM 2016 JBES, accepted manuscript §2.2, eq. (1), (11)–(14)):

```
y_t = Π_0 + Π(L) y_{t−1} + v_t,     v_t = Ã⁻¹ f_t^{1/2} ε_t,  ε_t ~ iid N(0, I_n)
Var(v_t) = Σ_t = f_t Σ,             Σ = Ã⁻¹ Ã⁻¹'        (Ã lower triangular, non-unit diagonal)
ln f_t = ψ ln f_{t−1} + u_t,        Var(u_t) = φ        (scalars; ln f has zero unconditional mean)
```

Chan (2020, JBES 38(1)) writes the same model in Kronecker form, eq. (1)–(3) with fn. 1:

```
Y = XΠ + U,    vec(U) ~ N(0, Σ ⊗ D),    D = diag(e^{h_1}, …, e^{h_T})
h_t = ρ h_{t−1} + ε^h_t,  ε^h_t ~ N(0, σ_h²),  |ρ| < 1,  h_1 ~ N(0, σ_h²/(1 − ρ²))
```

**In `rai-bvar`'s structural form (SPEC §4.1)** *(derived here)*. Factor `Σ = A⁻¹ Λ A⁻¹'` with `A` unit lower triangular and `Λ = diag(λ_1, …, λ_n)`. Then `CommonSV` is the recursive model with

```
λ_{i,t} = λ_i e^{h_t}    for every i: one shared state, per-equation constant scale.
```

Consequences for the architecture:

- `CommonSV` cannot be a per-equation `VolBlock` (SPEC §4.2). It is a **`GlobalBlock`**: it owns `h_{1:T}`, `ρ`, `σ_h²` and receives `q_t = Σ_i e_{i,t}²/λ_i`, a sum over equations of standardized squared structural residuals (one `vmap` and one reduction). Per-equation weights are `1/(λ_i e^{h_t})`.
- The fast path (§A5.1) bypasses the per-equation mean blocks entirely. The conditional posterior of `(Π, Σ)` given `h` is natural-conjugate NIW in the **reduced form**, so `fit.engine` for `Linear(prior=Minnesota(form="conjugate")) × CommonSV` should be a *hybrid*: MCMC over `(h, ρ, σ_h²)` with exact iid `(Π, Σ) | h` draws.

Working-paper history. In the 2012 version (Cleveland Fed WP 12-06, §2.1, eq. (1)–(3)) the model was `v_t = λ_t^{0.5} A⁻¹ S^{1/2} ε_t` with a **random-walk** `log λ_t`, unit-lower-triangular `A`, `S = diag(1, s_2, …, s_n)` with IG priors, and a JPR (1994) single-move Metropolis step for `λ_t`. The published version replaced this with IW `Σ`, AR(1) `h`, and KSC/Omori mixtures. Implement the **published** version. Keep the RW variant only as an option (`rho=1.0` fixed, with `h_0 = 0` pinned; see §A9).

`CommonSV` is order-invariant: CKY (2024) §2 and Chan's survey (arXiv:2310.14438, §3.1) both say so. This holds if the IW scale and the Minnesota scales are permuted consistently with the data, and if the `h` step does not depend on a particular square root of `Σ` (see §A5.2 for why the Omori variant is only approximately invariant).

## A2. Identification

- The scale of `Σ` and the level of `h` trade off one for one: `e^{h_t} Σ = e^{h_t + c} (e^{−c} Σ)`. Fix it with **zero unconditional mean** of `h` (AR(1) without intercept), as in both CCM (2016, §2.1: "assume ln f_t to have zero unconditional mean") and Chan (2020, fn. 1).
- Under a random walk (`ρ = 1`) the unconditional mean does not exist. Pin `h_0 = 0`, a known initial value. Otherwise the posterior of `(Σ, h)` is improper along that direction when the prior on `Σ` is flat-ish.
- The factor loadings are all 1 (CCM restriction (10): `β_j = 1`, `h_{j,t} = 1`). That restriction is the model; it is not a normalisation.

## A3. Priors: who uses what

| Parameter | CCM (2016 JBES) | Chan (2020 JBES) | `bvars` (Woźniak; GPL ≥ 3), docs only | **`rai-bvar` default** |
| :--- | :--- | :--- | :--- | :--- |
| `Π | Σ` | `N(vec μ_Π, Σ ⊗ Ω₀)`, μ_Π = 0. `Ω₀`: lag-`l` coefficient on variable `j` has variance `θ²/(l² σ_j²)` with θ = 0.2; intercept variance `ε²` with ε = 1000. σ_j² from AR(4) residual variances on the estimation sample (§3.1 eq. (34)–(35)) | `N(vec A₀, Σ ⊗ V_A)`, A₀ = 0, `v_A,ii = κ₁/(l² ŝ_r)` with κ₁ = 0.2²; intercept `κ₂ = 10²`. `ŝ_r` = AR(4) residual variance (§4.2) | `MN(A̲, Σ, V)` with a hierarchical **matrix GIG** prior on `V` (Ψ₀, Γ₀, λ₀). Starting values `kappa = c(0.2², 10²)`; `stationary` flag picks RW or white-noise prior mean | Phase 1 conjugate Minnesota (λ = 0.2, α = 2, `cross = 1`, transform-aware prior mean §5.3, optional SOC/SUR dummies), ψ in-window (R9) |
| `Σ` | `IW(d_s·Σ̲, d_s)`, d_s = n + 2. `Σ̲` = OLS residual covariance of a VAR(1) on a 40-quarter training sample before the estimation sample (§2.4 eq. (18), §3.1). See convention note 1 | `IW(S₀ = I_n, ν₀ = n + 3)` (§4.2) | `IW(S̲, ν̲)` (defaults not stated in the manual) | `IW(S₀ = diag(ψ), ν₀ = n + 2)`, so `E Σ = diag(ψ)` (GLP convention, same as Phase 1) |
| persistence | `ψ ~ N(0.9, 0.2²)`, no truncation stated (§3.1). The appendix uses sd 0.3 for the GFSV model (§6.1) | `ρ ~ N(0.9, 0.2²)·1(|ρ|<1)` | `rho` (prior not stated in the manual) | `ρ ~ N(0.9, 0.2²)·1(|ρ|<1)` |
| innovation variance | `φ ~ IG_Clark(d_φ φ̲, d_φ)`, φ̲ = 0.01, d_φ = 4, i.e. `IG(2, 0.02)`: mean 0.02, mode 0.0067. The paper calls 0.01 the "mean" (§3.1); see note 2 | `σ_h² ~ IG(ν_h0 = 5, S_h0 = 0.04)`: mean 0.01 (§4.2) | centred and non-centred SV. Non-centred: `ω ~ N(0, σ²_ω)` with a hierarchical gamma prior (`sv_a` shape, `sv_s` scale) on `σ²_ω` (manual fields; equations **UNVERIFIED**) | `σ_h² ~ IG(5, 0.04)` |
| initial state | `ln f_0` "mean 0, variance 0.5" in the Kalman filter (§3.2 Step 1) | `h_1 ~ N(0, σ_h²/(1−ρ²))` | not documented | stationary `h_1` |
| t errors (extension) | — | `λ_t ~ IG(ν/2, ν/2)` (Σ is the *scale*); `ν ~ U(2, 100)` (§2 item 1, §4.2) | `λ_t ~ IG2(ν−2, ν)`, documented as "inverse gamma 2" (Σ is the *covariance*, `E λ_t = 1`; parametrisation **UNVERIFIED**) | `tails="t"`: standardized `λ_t ~ IG(ν/2, (ν−2)/2)`, matching the sibling note 03, so Σ stays the covariance |

Convention notes:

1. **IW "prior mean".** CCM call `Σ̲` the prior mean of `Σ ~ IW(d_s Σ̲, d_s)`. Under the IW above, `E Σ = d_s Σ̲/(d_s − n − 1) = (n+2) Σ̲` when `d_s = n + 2`, while `E[Σ⁻¹]⁻¹ = Σ̲`. Their `Σ̲` is a precision-centred value, not the mean. For parity with CCM, set `S₀ = (n+2) Σ̲_CCM`. For `rai-bvar`, specify `S₀` directly so that `E Σ = S₀` at `ν₀ = n + 2`.
2. **`IG_Clark(a, d)` "mean".** CCM report "a mean of 0.01 and … 4 degrees of freedom" for `φ`. With their posterior update `IG(d_φ φ̲ + Σ u_t², d_φ + T)` (eq. (25)), the prior is `IG(shape = 2, scale = 0.02)`, whose mean is 0.02. Treat `φ̲` as the "centre", as CCM do (§2.4, IG definition).

## A4. Likelihood, sufficient statistic, integrated likelihood

```
log p(Y | Π, Σ, h) = −(nT/2) log 2π − (T/2) log|Σ| − (n/2) Σ_t h_t − ½ Σ_t q_t e^{−h_t},   q_t = u_t'Σ⁻¹u_t
```

Given `(Π, Σ)`, the data enter the `h`-conditional only through `q_{1:T}`. Moreover, `q_t e^{−h_t} | h ~ χ²_n` independently over `t`. This is also a good SBC test statistic (§A11).

**Integrated likelihood given `h`** (closed form; Chan 2020 §3 gives the NIW posterior, the marginal is standard and **derived here**). With `Ỹ = D^{−1/2} Y`, `X̃ = D^{−1/2} X` and the quantities of §A5.1:

```
log p(Y | h) = −(n/2) Σ_t h_t − (nT/2) log π + log Γ_n(ν̂/2) − log Γ_n(ν₀/2)
              + (ν₀/2) log|S₀| − (ν̂/2) log|Ŝ| − (n/2)(log|V| + log|K|)
```

The first term is the Jacobian `|D|^{−n/2}`. The rest is the Phase 1 matrix-t log-ML evaluated on the rescaled rows, so the Phase 1 code is reused verbatim plus the Jacobian term. All log-determinants come from triangular factors (`log|K| = 2 Σ log|R_ii|` from the QR of `[X_d; X̃]`).

**Cross-check (numerical).** The formula matches the basic marginal-likelihood identity `log p(Y|Π,Σ,h) + log p(Π,Σ) − log p(Π,Σ|Y,h)`, evaluated at an arbitrary `(Π,Σ)` with SciPy's `invwishart`/`matrix_normal`, to `6e−14` (`n = 3, k = 4, T = 25`; `scratchpad/xc08_checks/checks.py`). SciPy's `invwishart(df, scale)` uses the same `|Σ|^{−(ν+n+1)/2}` convention as this note.

Chan (2020, Online App. A) uses the complementary integrated likelihood `p(Y | Π, Σ, ρ, σ_h²)`, with `h` integrated out by importance sampling from the same Gaussian approximation `N(ĥ, K_h⁻¹)`. That quantity is for Chib-style model comparison and is not used by the sampler.

Uses:

- an SBC/Geweke test quantity;
- an optional collapsed sampler (MH on `h` from `p(Y|h) p(h)`, marginal of `(Π, Σ)`);
- Chib/predictive-likelihood model comparison.

## A5. Conditional posteriors

### A5.1 `(Π, Σ) | h, Y` (exact NIW; Chan 2020 §3, eq. (4)–(5) and the display after it)

```
X̃ = D^{−1/2} X,   Ỹ = D^{−1/2} Y                         (row t multiplied by e^{−h_t/2})
K  = V⁻¹ + X̃'X̃
Π̂  = K⁻¹ (V⁻¹ M + X̃'Ỹ)
Ŝ  = S₀ + M'V⁻¹M + Ỹ'Ỹ − Π̂' K Π̂        (form as S₀ + Ê'Ê on the dummy-augmented system, never by subtraction)
ν̂  = ν₀ + T
Σ | Y, h      ~ IW(Ŝ, ν̂)                                  (marginal of Π)
Π | Σ, Y, h   ~ MN(Π̂, Σ, K⁻¹):  Π = Π̂ + R⁻¹ Z L_Σ',  Z ~ N(0,1)^{k×n}
```

Here `K = R'R` comes from the thin QR of the stacked `[X_d; X̃]`, exactly as in SPEC §4.6. Check: `vec(R⁻¹ Z L_Σ') = (L_Σ ⊗ R⁻¹) vec Z`, whose covariance is `Σ ⊗ K⁻¹`.

Implementation: call the Phase 1 engine with observation weights `s_t = e^{h_t/2}` (the Lenza–Primiceri GLS path, SPEC §4.6). Draw `Σ` by Bartlett, then `Π`. Cost per sweep is `O(T k² + k³ + n³)`, compared with `O(n⁶ p³)` without the Kronecker structure (CCM §2.5, App. 6.3; Chan 2020 §3). The Phase 1 dummy observations (Minnesota, SOC, SUR) are **not** reweighted. They are prior rows and carry no `h_t`.

### A5.2 `h | Π, Σ, ρ, σ_h², Y`

**Recommended: exact independence MH.** The approach is Chan (2017), as used in Chan (2020) §4.3. The construction below follows Chan's Gaussian-approximation papers; the exact algebra is *derived here*.

The prior is `H_ρ h ~ N(0, S_h)`, with `H_ρ` unit lower bidiagonal (`−ρ` on the subdiagonal) and `S_h = diag(σ_h²/(1−ρ²), σ_h², …, σ_h²)`. The prior precision is tridiagonal:

```
K_h = H_ρ' S_h⁻¹ H_ρ = (1/σ_h²) · tridiag( diag = [1, 1+ρ², …, 1+ρ², 1],  offdiag = −ρ )
```

The log target, its gradient and the negative Hessian are:

```
ℓ(h)  = −(n/2) Σ_t h_t − ½ Σ_t q_t e^{−h_t} − ½ h'K_h h
g(h)  = −(n/2)·1 + ½ q ⊙ e^{−h} − K_h h
G(h)  = K_h + diag(½ q ⊙ e^{−h})        (tridiagonal, SPD, so ℓ is strictly concave)
```

1. Newton with backtracking: `h ← h + G⁻¹g`, halving the step until `ℓ` increases. Stop when `‖Δ‖_∞ < 1e−8` or at `max_iter` (20). Each solve is a tridiagonal Cholesky, `O(T)`.
   **Start Newton from a point that does not depend on the current draw `h`**, for example `h⁽⁰⁾_t = log(q_t/n)` (the per-`t` likelihood mode), or `0`. Then `ĥ` is a function of `(q, ρ, σ_h²)` only and the proposal is a true independence proposal, whatever the convergence tolerance. A warm start from the current `h` is exact only if Newton converges exactly. `ℓ` is strictly concave, so the converged mode is unique. In the simulations below Newton from `log(q/n)` converged to `‖Δ‖_∞ < 1e−10` in 5–8 iterations. Raise (or count as a diagnostic) if `max_iter` is hit.
2. Proposal `h* ~ N(ĥ, G(ĥ)⁻¹)` with the precision sampler: `G = L L'`, `h* = ĥ + L'⁻¹ z` (SPEC §8).
3. Accept with
   `log α = ℓ(h*) − ℓ(h) + ½ (h* − ĥ)'G(h* − ĥ) − ½ (h − ĥ)'G(h − ĥ)`.
   The log-determinants cancel.
4. **Acceptance rate (simulated here; corrected).** The first draft said acceptance should rise with `n`. Simulation does not support that. The acceptance rate is roughly **flat in `n`** and **falls with `T` and with `σ_h²`**. The reason: the log-error of a joint Laplace proposal adds up over `T` dates. Per-date non-Gaussianity is largest when the likelihood and prior precisions are comparable, so neither small nor large `n` is the bad case. Simulated values: `q_t` drawn from the model at a stationary `h`, 1,500 MH iterations, script `scratchpad/xc08_checks/checks.py`:

   | `ρ`, `σ_h²` | `T` | `n = 1` | `n = 3` | `n = 5` | `n = 20` | `n = 50` |
   | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
   | 0.95, 0.02 | 100 / 300 | 0.82 / 0.71 | 0.83 / 0.67 | 0.81 / 0.65 | 0.78 / 0.64 | 0.79 / 0.60 |
   | 0.90, 0.10 | 100 / 300 | 0.64 / 0.31 | 0.50 / 0.34 | 0.58 / 0.36 | 0.52 / 0.30 | 0.56 / 0.31 |
   | 0.98, 0.05 | 100 / 300 | 0.66 / 0.41 | 0.64 / 0.50 | 0.64 / 0.46 | 0.68 / 0.48 | 0.68 / 0.48 |

   At `T = 600 / 1000` (monthly data): about 0.5 / 0.4 for `σ_h² = 0.02`, but 0.15–0.2 / 0.05–0.11 for `σ_h² = 0.1`. Chan's own acceptance rates remain **UNVERIFIED**: Chan (2020) reports only inefficiency factors. Report the acceptance rate as a diagnostic.

   **Blocked variant (recommended default when `T > 200`; exact).** Split `1..T` into contiguous blocks `B` of about 100 dates. Update each block by the same independence MH, conditional on the other blocks. Because `K_h` is tridiagonal, the block target is
   `ℓ_B(h_B) = −(n/2)Σ_{t∈B} h_t − ½Σ_{t∈B} q_t e^{−h_t} − ½ h_B'K_{BB}h_B − h_B'K_{B,¬B}h_{¬B}`,
   and `K_{B,¬B}h_{¬B}` has only two non-zero entries, `−(ρ/σ_h²)h_{s−1}` at the first index `s` of the block and `−(ρ/σ_h²)h_{e+1}` at the last index `e`. The gradient gains the term `−K_{B,¬B}h_{¬B}`, and the Hessian is `K_{BB} + diag(½ q e^{−h})`. Start Newton from `log(q_B/n)`, as above. This remains exactly order-invariant. Randomise block boundaries per iteration (a seeded random offset) to avoid fixed-boundary stickiness. *Derived here*; Geweke-test it like the full-block version.

There is no mixture indicator, so the Del Negro–Primiceri ordering constraint (SPEC §4.3) does not apply to this block. The step depends on `Σ` only through `q_t`, so it is invariant to any choice of square root of `Σ` and to variable permutations.

**Optional: Omori-mixture variant (CCM 2016 §2.6, eq. (33)).**

1. Let `ẽ_t = L_Σ⁻¹ u_t` (Cholesky; CCM's `ṽ_t = Ã v_t`) and `y*_{j,t} = log(ẽ_{j,t}² + c̄)` for `j = 1..n`. These are `nT` pseudo-observations of the single state `h_t`.
2. Draw indicators `s_{j,t}` with `P(s_{j,t} = k) ∝ π_k N(y*_{j,t}; h_t + m_k, v_k²)` (Omori 10-component constants, see the SV note).
3. Draw `h | s ~ N(G⁻¹b, G⁻¹)` with

```
G = K_h + diag( Σ_j 1/v²_{s_{j,t}} )
b_t = Σ_j (y*_{j,t} − m_{s_{j,t}}) / v²_{s_{j,t}}
```

Step order must follow DP2015: `Θ | h → s | Θ, h → h | s, Θ` (CCM 2016 §3.2 "Step 2e" reorders accordingly). This variant targets a mixture-approximated posterior. Strictly, its `(Π,Σ)` and `h` conditionals are mutually incompatible, so it is approximate, not exact (see §A11, Geweke). Because the approximation is applied to the components of one particular rotation `L_Σ⁻¹u_t`, it is order-invariant only up to approximation error (*derived here*). The value of `c̄` used by CCM is not stated: **UNVERIFIED**.

### A5.3 `ρ, σ_h² | h` (stationary start; Chan & Hsiao 2014 style, *derived here*)

```
K_ρ = 1/V_ρ + Σ_{t=2}^T h_{t−1}²/σ_h²,     ρ̂ = K_ρ⁻¹ ( ρ₀/V_ρ + Σ_{t=2}^T h_{t−1} h_t / σ_h² )
propose ρ* ~ N(ρ̂, K_ρ⁻¹) truncated to (−1, 1);  accept with min{1, g(ρ*)/g(ρ)},
g(ρ) = (1−ρ²)^{1/2} exp( −(1−ρ²) h_1² / (2σ_h²) )
σ_h² | h, ρ ~ IG( a_σ + T/2,  b_σ + ½[ (1−ρ²) h_1² + Σ_{t=2}^T (h_t − ρ h_{t−1})² ] )
```

The truncation constants cancel because the proposal is the truncated conjugate part of the target. Draw the truncated normal by inverse CDF in log space (`log_ndtr` differences).

**CCM parity mode** (`h_0 ~ N(0, 0.5)` fixed, untruncated `ψ`, Clark-IG `φ`; CCM eq. (24)–(25), (28)–(29)). Add `h_0` as a state, so `K_h` becomes `(T+1)×(T+1)`:

- `K_h[0,0] = 1/0.5 + ψ²/φ`;
- `K_h[t,t] = (1+ψ²)/φ` for `1 ≤ t < T`;
- `K_h[T,T] = 1/φ`;
- off-diagonal entries `−ψ/φ`.

Checked numerically against `H'S⁻¹H`. Then:

```
ψ | h ~ N(μ̄_ψ, Ω̄_ψ),  Ω̄_ψ = (Ω_ψ⁻¹ + φ⁻¹ Σ_{t=1}^T h_{t−1}²)⁻¹,  μ̄_ψ = Ω̄_ψ(Ω_ψ⁻¹ μ_ψ + φ⁻¹ Σ_{t=1}^T h_{t−1} h_t)
φ | h, ψ ~ IG( (d_φ + T)/2,  (d_φ φ̲ + Σ_{t=1}^T (h_t − ψ h_{t−1})²)/2 )
```

Whether CCM's "initial value … 0 and 0.5" refers to `t = 0` or `t = 1` is **UNVERIFIED**; implement it as `h_0`.

### A5.4 Student-t extension (Chan 2020: BVAR-t-CSV)

With `Ω = diag(λ_t e^{h_t})`, the extension is as follows *(derived here; Chan 2020 §2 item 1, §4.2)*:

- **`λ_t | ·`**: under Chan's unit-scale prior `λ_t ~ IG(ν/2, ν/2)`, `λ_t | · ~ IG((ν+n)/2, (ν + q_t e^{−h_t})/2)`. Under the standardized prior (§A3), replace `ν` with `ν − 2` in the scale only: `IG((ν+n)/2, (ν − 2 + q_t e^{−h_t})/2)`.
- **`ν | λ`**: use the collapsed griddy Gibbs of sibling note 03 or bounded MH. Chan uses `U(2, 100)` and samples `ν` by an independence-chain MH with proposal `N(ν̂, K_ν⁻¹)` at the mode of `log p(ν | ·)` (Chan 2020, Online App. A2, Example 1).
- **Closed form for tests** (Chan 2020, Online App. A, integrated-likelihood section, BVAR-t-CSV): `λ` integrates out analytically. `p(Y | h, Π, Σ, ν)` is a product over `t` of multivariate-t densities with scale `e^{h_t}Σ` and `ν` degrees of freedom (Chan's unit-scale convention). This is a useful SBC/Geweke test quantity.
- **Other steps**: in A5.1 and A5.2 replace `e^{h_t}` with `λ_t e^{h_t}`, i.e. row weights `(λ_t e^{h_t})^{−1/2}` and `q_t/λ_t` in the `h` step.

## A6. Sweep order (one iteration)

```
1. (Σ, Π) | h, [λ], Y              exact NIW draw (A5.1)
2. h | Π, Σ, ρ, σ_h², [λ], Y       exact MH (A5.2). Omori variant: 2a s | ·, then 2b h | s, with nothing in between
3. ρ | h, σ_h²                     truncated-normal MH (A5.3)
4. σ_h² | h, ρ                     inverse gamma
5. [t only] λ_t | ·, then ν | λ    (A5.4)
6. [Phase 1 hierarchy, optional]   Minnesota hyperparameters | h via MH on log p(Y|h) + log p(hyper)
                                   (closed form, A4). This is GLP conditional on h (CCM 2016 §3.1 mention it, untested)
```

Mixing benchmarks in CCM 2016 Table 1 (14 variables, 55,000 draws, burn 5,000, thin 5): the median inefficiency factors are Π 0.95, Σ 2.65, φ 5.06, ψ 2.99, Λ 3.14. With 4 variables, Σ has the worst median IF, 11.8.

## A7. Numerical tricks

- **No `K⁻¹`, no `Ω̄_Π`.** QR of the dummy-augmented, row-weighted design (Phase 1). For `n = 100, p = 4, T = 300`, `k = 401`. The design has `T + k_d` rows, where `k_d ≥ k` counts the Minnesota dummy rows. A Householder QR then costs about `2(T + k_d)k² − (2/3)k³ ≈ 1.8·10⁸` flops per sweep. The draws add `O(k²n + kn²)`. This is dominated by `k³`, not `T k²`, when `T < k` (corrected; the first draft quoted `T k² ≈ 5·10⁷`). Chan (2020) Table 2: 20 variables, 10,000 draws of BVAR-CSV in 0.46 min (MATLAB).
- **Row weights in log space.** `x̃_t = exp(−h_t/2) x_t`. Overflow is impossible in the realistic range, but compute `q_t e^{−h_t}` as `exp(log q_t − h_t)` with `log q_t` from `2·log‖L_Σ⁻¹u_t‖`.
- **Tridiagonal kernels in JAX.** Write a `lax.scan` tridiagonal Cholesky, forward solve and backward solve, shared with the SV block. Run Newton under `lax.while_loop` with `max_iter` and a backtracking inner loop. There is no Python branching inside the sweep.
- **Damped Newton** guarantees monotone ascent. `ℓ` is strictly concave and its Hessian is bounded away from 0 by `K_h`, so convergence is global.
- **Determinism.** Split keys per (chain, iteration, block): one key for `Z` (Π), one for Bartlett (Σ), one for the `h` proposal (one per MH block), one for the uniform(s), one for the `ρ` proposal and its uniform, one for `σ_h²`, and one for the random block offset.
- **float64 throughout.** `q_t` can be tiny (`~1e−6`). Use `log q_t` with an `eps` only for exact zeros, and raise if any `q_t == 0` (measure zero; it signals a degenerate design).

## A8. Forecasting (SPEC §6.1, §7)

For each draw:

```
h_{T+j} = ρ h_{T+j−1} + σ_h η_j
cond_mean[j] = Π' x_{T+j}
cond_chol[j] = e^{h_{T+j}/2} L_Σ      (× √λ_{T+j} with t; λ_{T+j} ~ prior)
```

`cond_chol` is lower triangular, so it satisfies the contract directly. The RW variant uses `ρ = 1`.

## A9. Edge cases

- **`ρ → 1` or the RW variant**: pin `h_0 = 0`. Under RW the `K_h` diagonal becomes `[2, …, 2, 1]/σ_h²` with `h_0 = 0` known.
- **`T` small relative to `n`**: `ν̂ = ν₀ + T` must exceed `n − 1` (always true with `ν₀ = n + 2`). For `Ŝ`, raise on a failed Cholesky with diagnostics; never jitter (SPEC §8).
- **COVID**: `CommonSV` absorbs a common variance spike, but the AR(1) `h` reverts slowly. For samples with 2020, compare against the Lenza–Primiceri scaling (Phase 1) and `CommonSV + t`. A common outlier component is a natural extension: multiply `D` by `o_t²`, as in the CCMM 2024 structure, applied to the common scale (*suggestion*).
- **Explosive draws**: CCM (2012 WP fn. 6) reject explosive VAR draws. `rai-bvar` should **not** reject inside the sampler, because that changes the posterior and breaks Geweke. Flag it at forecast time instead.
- **Phase 1 hierarchy (GLP) with `CommonSV`**: MH on hyperparameters given `h` uses `log p(Y|h)` (A4). R9: recompute ψ, `S₀` and the training-sample quantities inside each window.

## A10. Known paper issues (verified against the accepted manuscript text)

1. **CCM 2016 eq. (23)**, `Σ | Π, f, y ~ IW(d_s Σ̲ + Σ_t f_t⁻¹ v_t v_t', d_s + T)`, is not the exact full conditional under their own conditional prior `vec Π | Σ ~ N(·, Σ ⊗ Ω₀)` (eq. (17), (21)). The full conditional also contains `(Π − μ_Π)'Ω₀⁻¹(Π − μ_Π)` and `k` more degrees of freedom. It is also not the `Π`-marginal, which needs `Ŝ` and `ν₀ + T` as in A5.1. Use the exact joint NIW draw (A5.1). Whether CCM's RATS code used the correct form is **UNVERIFIED**.
2. **"Prior mean" conventions.** See §A3 notes 1–2.
3. **`ψ` prior standard deviation.** It is 0.2 in §3.1 and 0.3 for GFSV in App. 6.1.
4. **Offset `c̄`** for `log(ṽ² + c̄)` is named but not given (§2.6): **UNVERIFIED**.

## A11. How to test it

**Closed form (unit/analytic layer, every PR)**

1. *`h` fixed (vol block frozen).* Draws of `(Π, Σ)` must match the analytic NIW on rescaled rows. Check `E Π = Π̂` and `E Σ = Ŝ/(ν̂−n−1)` within MCSE, and run a KS test on `log|Σ|`. Also compare `log p(Y | h)` (A4) against the Phase 1 log-ML on manually rescaled data plus `−(n/2)Σh` to `1e−10`.
2. *Equivalence of conjugate and structural forms* (*derived here* from Chan 2022, Prop. 1). Take `Σ ~ IW(ν₀, diag s²)` and `Π | Σ ~ MN(M, Σ, V)`. This equals, in structural form, `λ_i ~ IG((ν₀+i−n)/2, s_i²/2)`, `a_ij | λ_i ~ N(0, λ_i/s_j²)`, `γ_i | a_i, λ_i ~ N(M ã_i, λ_i V)`, with `ã_i` = row `i` of `A`, and `Γ' = Π A'` so `γ_i = Π ã_i` (`vec(ΠA') = (A ⊗ I_k) vec Π ~ N(vec(MA'), Λ ⊗ V)`). Given `h`, per-equation conjugate draws with weights `e^{−h_t}` must reproduce the Kronecker draws (two-sample tests on `Σ`, `Π`).
3. *Brute-force `h` posterior* for `T ∈ {2, 3}` and `n = 5`. Tensor-grid quadrature over `h` (e.g. `200³` points) of `exp ℓ(h)` gives exact marginals of `h_t`, to compare with MH draws: KS test and quantiles within MCSE.
4. *MH kernel sanity.* With `q_t` generated from the model at known `h`, compare the acceptance rate with the reference table in §A5.2 step 4 (soft check: at `ρ = 0.95, σ_h² = 0.02, T = 100`, expect about 0.8; flag below 0.6). Do **not** use an `n`-based threshold: acceptance is roughly flat in `n`. `ĥ` from Newton must satisfy `‖g(ĥ)‖ < 1e−8`. Assert that `ĥ` does not change when the current `h` passed to the kernel changes (state independence of the proposal).
5. *Permutation invariance.* Permute the variables, `S₀` and the Minnesota scales. Analytic pieces must match to `1e−10` after un-permuting. MCMC posteriors of `h_t`, `ρ`, `σ_h²` must match within 4 MCSE (two independent runs). Run the same test on the Omori variant and expect small but non-zero discrepancies at large draws. That characterises the approximation; it is not a failure.

**Geweke (SPEC §10.3)**

- Marginal-conditional simulator: `ρ ~ TN`, `σ_h² ~ IG`, `h ~` AR(1) stationary, `Σ ~ IW`, `Π | Σ ~ MN`, then `Y | θ`.
- To avoid explosive lag dynamics, test with **exogenous fixed regressors** `X ~ N(0,1)` (`k = 3`), `n = 3`, `T = 40`. The samplers condition on `X`; lag construction is unit-tested separately.
- Test functions: `ρ, σ_h², h_1, h_{T/2}, h_T, mean(h), log|Σ|, Σ_11, Σ_21, Π_11, Π_11², ℓ(Y|θ)`.
- The exact-MH variant (full or blocked) should pass at any length.
- The **Omori variant is not exact under any single data-generating model** (corrected). Its `(Π, Σ)` block uses the Gaussian likelihood, while its `h` block uses the 10-component mixture for `log ε²` and the offset `c̄`. The conditionals are therefore incompatible, and a full-sweep Geweke test will show a small bias at high power. Two options:
  - A *block-level* Geweke test that holds `(Π, Σ)` fixed, sets `c̄ = 0`, and generates `ε_{j,t} = ±exp(z/2)` (random sign), `z ~` the Omori mixture, rotated by `L_Σ`. This is exact for the `(s, h, ρ, σ_h²)` sub-sampler.
  - A full-sweep test with a pre-registered tolerance, documented as approximate.

  Because SPEC R6 requires passing Geweke/SBC, ship the Omori variant only as a parity option labelled approximate.

**SBC (nightly)**

- Parameters: `ρ, σ_h², h_t` (3 dates), `log|Σ|`, `Π` entries.
- Data-dependent quantities: `log p(Y | Π, Σ, h)`, `log p(Y | h)` (A4), and `W(θ, Y) = Σ_t q_t e^{−h_t}`, a sharp check of the `h`–`Σ` coupling. Use `W` like any SBC quantity: take the rank of `W(θ_true, Y)` among `W(θ̃_s, Y)` over posterior draws `θ̃_s`.
- Corrected: `W` is **not** `χ²_{nT}` across posterior draws for fixed `Y`. It is `χ²_{nT}` only at the data-generating `θ` (prior-predictive simulation), which makes it a cheap unit test of the data simulator, separate from SBC.
- 500 replications; thin to an ESS of about 100 per replication.

**Parity and replication (Phase 3)**

- **`bvars`**: `e_t ~ N(0, σ_t² λ_t Σ)` with MN–IW and a hierarchical matrix-GIG prior on `V`. Exact prior mapping is **not available**: the hierarchical `V` and the SV prior equations are undocumented and **UNVERIFIED**. So:
  - Fix `V` in `bvars` if its API allows. If not, compare forecast distributions only (SPEC §10.4 last bullet).
  - Map `Σ_bvars(t-errors) = Σ` (covariance) to `rai-bvar`'s standardized-t `Σ`, and to Chan's scale `Σ_Chan` via `Σ_bvars = (ν/(ν−2)) Σ_Chan`. This assumes `IG2(ν−2, ν)` has mean 1.
    - That holds under the Bauwens–Lubrano–Richard "inverted gamma-2" convention `IG2(s, ν)`: density `∝ x^{−(ν+2)/2} e^{−s/(2x)}`, i.e. `IG(ν/2, s/2)`, mean `s/(ν−2)`. Clark & Ravazzolo use the same convention (WP 12-18 App. 6.4, eq. (26): `IG-2(ω̄, ν)` with mean `ω̄/(ν−2)`).
    - Whether `bvars` follows this convention is **UNVERIFIED**: its manual names "inverse gamma 2" but does not define it.
    - `bvars` cites Chan (2020) as its model reference (manual, References).
- **CCM 2016 replication (qualitative)**, US, 4 and 14 variables, estimation from 1965Q1, evaluation 1985Q1–2013Q4, h = 1, 2, 4, 8, 12:
  - CSV beats BVAR in average log score at h = 1, 2 for all variables. Example, 4 variables (Table 6): joint +0.930 at h = 1 and +0.862 at h = 2, against BVAR-SV +0.967 and +0.831.
  - Log-ML (Table 3): BVAR −1097.5, BVAR-CSV −946.4 (Chib); 14 variables: BVAR −5644.1, BVAR-CSV −5419.9.
  - CSV beats BVAR in RMSE in 51 of 70 cases with 14 variables (§4.5.1).

---

# Part B: Variable-ordering dependence of triangular SV models

## B1. What exactly depends on the ordering

**(i) Constant `Σ`, likelihood.** `Σ ↦ (A, Λ)` (LDL') is a bijection for every ordering. Hence the likelihood of a constant-volatility VAR is ordering-invariant up to relabelling, and **only the prior** can break invariance:

- CCM (2019 WP 16-17 §3.1) say the problem is "entirely absent in the case of the homoskedastic model", because there the prior can be placed on the whole matrix (e.g. IW).
- With independent priors on `a_i` and `λ_i` (as in `rai-bvar`'s structural form), the induced prior on `Σ` is ordering-dependent. CKY (2024 §2) give an example: `a_ij ~ N(0, 1)` iid and `ε ~ N(0, I)` imply `E u_{i,t}² = 2^{i−1}`. The variance grows exponentially with position.
- **Exact fix (Chan 2022, QE, Prop. 1, Cor. 1–3).** `σ_i² ~ IG((ν₀+i−n)/2, s_i²/2)` and `A_ij | σ_i² ~ N(0, σ_i²/s_j²)` induce `Σ ~ IW(ν₀, diag(s²))`. For a permutation π, use `s_{π(i)}` (Cor. 2). Non-diagonal `S` gives correlated rows (Cor. 3). The **shape depends on position**, `(ν₀ − n + i)/2`, which is `1 + i/2` for `ν₀ = n + 2`. An "identical prior for every equation" is therefore *not* order-invariant.
- **Coefficients.** Under the structural-form Minnesota (SPEC D3), the prior on `B = (A⁻¹Γ)'` depends on the ordering even with the IW-equivalent `(a, λ)` prior. The exception is when `γ_i`'s prior covariance is scaled by `λ_i` in the conjugate way **with the same `V` in every equation** (`γ_i | a_i, λ_i ~ N(M ã_i, λ_i V)`; §A11 test 2). That implies `Π | Σ ~ MN(M, Σ, V)`, and the Jacobian of `Π ↦ Γ = AΠ'` is `|A|^k = 1`. Equation-specific `V_i` (structural cross-variable shrinkage, as in Chan's asymmetric conjugate prior) breaks the Kronecker form and hence order invariance of the induced prior on `B`. Chan's survey (arXiv:2310.14438 §4.2.3; the first draft said §4.3): normal priors on structural coefficients "often imply order-dependent priors on the reduced-form parameters"; the remedy is to elicit on the reduced form and derive the structural prior (Chan 2022).

**(ii) Time-varying `Σ_t = A⁻¹ Λ_t A⁻¹'` (Cogley–Sargent; CCM 2019; CCCM 2022; `rai-bvar` `SV`).** Here the **model class itself** depends on the ordering, not just the prior. *(Derived here; consistent with ARRS §2.2 and CKY §6.1.)*

- `Var_t(u_1) = λ_{1,t}`: a log-AR(1) (log-normal) process. In general `Var_t(u_i) = Σ_{j≤i} (A⁻¹)_{ij}² λ_{j,t}`, a sum of log-normals.
- Correlations move in an ordering-specific way. With `n = 2`, `A⁻¹ = [[1,0],[−a,1]]` gives `Corr_t(u_1,u_2) = −a λ_{1,t} / √(λ_{1,t}(a² λ_{1,t} + λ_{2,t}))`. Its magnitude rises when the **first-ordered** variable becomes volatile. Under the reverse ordering, it rises when the other one does. The two models are non-nested.
- ARRS (§2.2) give the analytic example `σ_{1,t} = σ_{2,t} = 1` and `α_21 ~ N(0, 1)`: the predictive density of `y_1` is Gaussian, while that of `y_2` is non-Gaussian with fatter tails (a normal-product mixture).
- CKY (§6.1) simulate from a non-triangular `B_0` with SV. The CS-VAR-SV estimates of the time-varying covariances are "flat" and miss the true dynamics.

**(iii) Downstream objects that inherit (ii):**

| Object | Ordering effect | Evidence |
| :--- | :--- | :--- |
| Reduced-form `B | Σ_{1:T}` | Invariant conditional on `Σ_{1:T}` (reduced-form prior). It differs marginally through the GLS weights `Σ_t⁻¹`, and under structural priors also through the prior | CCM 2019 WP §3.1; CKY §2 |
| Point forecasts | Small | ARRS: RMSE range within about ±5% (10% for 8-q inflation). The max-pairwise-DM test rejects in 2 of 12 cases. CKY Table 5: RMSFE equal to 3 decimals for IP, UR and PCE inflation. For the FFR they differ by up to 2.2% (h = 12: 2.216 vs 2.264), and the DM test rejects in a few cells |
| Predictive variances, intervals, LPS | **Material** | ARRS Tables 2–5, Fig. 2–5; CKY Table 5 |
| Conditional forecasts | **Amplified** | ARRS §3.5: one-quarter GDP RMSE range about 0.3 pp conditional vs <0.1 pp unconditional |
| Hyperparameters | Can be large | CKY Table 2: cross-lag shrinkage κ₂ 2.34e−6 vs 4.30e−6 (+84%) after reversing the order |
| Marginal likelihood | Large | CKY Table 3: −26,301 vs −26,709 (408 log points) for the two orderings of CS-VAR-SV (20 variables) |
| Recursive IRFs | Ordering is the identification by design | SPEC §6.3 |
| Size dependence | Worse as `n` grows | CKY §2 (`2^{i−1}`); ARRS §1 (20! ≈ 2.4·10¹⁸ orderings) |
| Time-varying `A_t` (Primiceri) | Same issue, can be extreme | CKY §7.2 Table 6: reversing the order of a 4-variable TVP-VAR-SV blows up long-horizon forecasts (IP 12-month RMSFE 6.217 vs 0.073) |

Note that CCM (2019 WP §3.1) argued the problem "will be mitigated" with constant `A`, because the likelihood dominates the prior. ARRS and CKY show that empirically it is not negligible for density forecasts, and CKY's evidence uses constant `B_0`.

## B2. Which `rai-bvar` compositions are order-invariant

| Composition | Invariant? | Notes |
| :--- | :--- | :--- |
| Analytic NIW (Phase 1) | **Yes** (exact) | Provided Minnesota scales, ψ, SOC/SUR are permuted with the data |
| `CommonSV` (Phase 3) | **Yes** with the exact `h` step | Only approximately with the Omori variant (§A5.2) |
| `Linear(form="reduced") × Constant(prior="iw_equivalent")` | **Yes** | Chan 2022 Prop. 1 plus a reduced-form prior on `B`. Recommend adding it as an option |
| `Linear(form="structural") × Constant` | No (prior) | Even with IW-equivalent `(a, λ)`, unless `γ_i` is prior-scaled by `λ_i` (conjugate) |
| `Linear × SV`, `SV(outliers=True)`, `SV(tails="t")` (any form) | **No** (model class) | The outlier and t components act per *structural* equation, so they are ordering-specific too |
| `FactorSV` (Phase 5) | Yes, if loadings are not triangularly restricted | Kastner (2019) does not restrict them (as stated by CKY §2) |
| Order-invariant SV (Phase 5) | Yes | CKY (2024): unrestricted `B_0` identified by SV (Bertsche–Braun 2020) |
| DW-SV, DSC-SV | Yes | ARRS §4. DW-SV underperforms; the DSC variants are comparable to the median CSP-SV ordering. Not on the roadmap |

## B3. Empirical magnitudes (what to expect from the diagnostic)

**ARRS (2023; WP 21-21 revised Mar 2022)**

Set-up:

- **Model**: TVP-VAR-SV in the Cogley–Sargent–Primiceri style (time-varying `B_t`, `A_t`, RW log-vol).
- **Data**: 4 variables (real GDP growth, core PCE inflation, 3-m T-bill, unemployment), FRED-QD, 1970Q3–2016Q4, p = 2, prior training sample 1960Q3–1970Q2.
- **Evaluation**: all 24 orderings, recursive; 120 origins from 1987Q1; h = 1, 4, 8; 10,000 retained draws (110,000, burn 10,000, thin 10); 10 paths per draw.

Results (WP §3, Tables 1–4):

- **Joint log predictive score.** The 1-q sum ranges from −379.44 to −350.06, a spread of 29.4 over 120 origins (≈ 0.25 nats per quarter). The spread is ≈ 40 at 4 q and ≈ 52 at 8 q.
- **Marginal 1-q spreads**: T-bill 18.1 (0.15/qtr), unemployment 9.1, output 7.0, inflation 2.6.
- **Tests.** The Amisano–Giacomini max-pairwise test (Hansen–Lunde–Nason p-values) rejects equal LPS in 9 of 15 cases at 10%.
- **Best ordering** differs by variable, horizon and date. ARRS Table 4 gives the Spearman correlations between the ranking of orderings by 1-q output-growth LPS and the rankings by the other variable × horizon LPS sums. They range from −0.08 to 0.45; the first draft said "mostly 0.05–0.45".
- **Robustness.** With 30 independent chains of 20,000 draws, the best–worst ordering gap in the predictive density far exceeds Monte Carlo noise (Fig. 3). The diagnostic needs that comparison too.

**CKY (2024; accepted manuscript)**

Set-up:

- **Data**: FRED-MD, 20 monthly variables, 1959:03–2019:12, CCM-2019 transformations; evaluation from 1970:03; h = 1, 6, 12.
- **Orderings**: (1) IP, UNRATE, PCE inflation, FFR first, then the CCM 2019 order; (2) the reverse.
- **Prior (both models)**: `φ ~ N(0.95, 0.05²)`, `σ² ~ IG(3, 0.1)`. The CS model has `B_0` lower-triangular elements `N(0, 1)`; the OI model has diagonal `N(1, 1)`. Minnesota-type horseshoe on the VAR coefficients. CS is estimated with the CCCM 2022 sampler.

Average log predictive likelihood (ALPL), CS-1 vs CS-2 (Table 5):

| Variable | h = 1 | h = 12 |
| :--- | :--- | :--- |
| IP | 2.071 vs 1.522 (Δ 0.55/month) | 2.266 vs 1.767 |
| PCE inflation | 2.210 vs 1.811 (Δ 0.40) | 2.425 vs 2.024 |
| Unemployment | −0.005 vs −0.124 | −0.707 vs −0.742 |
| FFR | 0.222 vs −0.327 (Δ 0.55) | −16.704 vs −12.474 (the *reverse* ordering is better here) |

- **The order-invariant model dominates**: IP 3.660, UR 0.463, PCE 4.821, FFR 0.294 at h = 1.
- **RMSFE** is essentially identical across orderings.

**Comparison with SV gains** (useful for the Phase 2 replication):

- CR WP Table 2 (VAR-SV − VAR, 1-q average log score): GDP +0.193, inflation +0.075, unemployment +0.162, T-bill +0.284. That sums to ≈ 0.71 nats/quarter across marginals.
- CCM 2016 Table 6 (BVAR-SV − BVAR, joint): +0.967.

So the SV gain is ≈ 3–4× the ARRS ordering spread (≈ 0.25/quarter joint) on comparable 4-variable US data. Caveats added in cross-check:

1. CR's 0.71 is a sum of *marginal* score differences, not a joint score difference. Only CCM's +0.967 is joint, and that is for a different variable set (FFR, not the T-bill) and sample.
2. ARRS's spread comes from a TVP-VAR-SV with **time-varying `A_t`**. CKY Tables 5–6 show this amplifies ordering effects relative to constant `A`. For `rai-bvar`'s constant-`A` `SV`, the 4-variable spread is expected to be smaller (*expected*, **UNVERIFIED**).
3. Per variable, compare like with like. ARRS 1-q *marginal* spreads per quarter are GDP 0.058, inflation 0.021, T-bill 0.151 and unemployment 0.076. CR's VAR-SV gains at h = 1 are 0.193, 0.075, 0.284 and 0.162, i.e. ratios of 3.3, 3.5, 1.9 and 2.1. The first draft compared the inflation cell (0.075) with the *joint* spread; that comparison was misleading.

The SV-beats-constant ranking should survive any ordering at h = 1. The T-bill and unemployment cells (ratio ≈ 2), and every cell at h ≥ 4 where gains shrink, are close enough to the ordering spread that the replication must report ordering sensitivity alongside the verdict.

## B4. Recommended diagnostic: `rb.ordering_sensitivity` (SPEC §6.6)

This design is a *recommendation*. Elements traceable to ARRS and CKY are cited.

```python
rb.ordering_sensitivity(
    model, data,
    n_perm=10,                      # number of orderings (incl. the user's and its reverse)
    pinned=(),                      # leading variables kept fixed, in order
    mode="forecast",                # "prior" | "forecast" | "backtest"
    design="balanced",              # "balanced" (Latin square) | "random" | "all"
    horizons=(1, 4, 8),
    origins=None,                   # backtest mode: subset of origins (default: every 4th)
    noise_reps=3,                   # extra seeds for the user's ordering (Monte Carlo noise floor)
    quantiles=(0.05, 0.16, 0.5, 0.84, 0.95),
    pool=False,                     # also return the equal-weight mixture over orderings
    seed=...,
) -> OrderingSensitivityResult
```

**Permutation design**

- Let `F` be the free (unpinned) variables and `m = |F|`. If `m! ≤ n_perm`, or `design="all"`, enumerate all orderings (e.g. 24 for `m = 4`, as in ARRS).
- Otherwise always include the **user ordering** and its **reverse** on `F` (CKY's comparison).
- Fill the remaining slots with rows of a random **Latin square** of order `m`, so each variable occupies each free position about equally often. The ordering effect is a *position* effect (§B1 ii), and this balances it. If `n_perm > m + 2`, top up with uniform random permutations without replacement, seeded.

**Per ordering**

- Fit on `data[pinned + π(F)]` with the same hyperparameter rules. R9: everything data-dependent is recomputed, and per-variable hyperparameters (ψ, Minnesota means from tcodes) follow their variables.
- Map every output back to the original labels with the inverse permutation, and assert round-trip correctness.
- Store per variable and horizon:
  - predictive mean, sd and quantiles;
  - central interval lengths (68%, 90%);
  - marginal and joint LPD (Rao–Blackwellised, SPEC §7) at the realised value when available;
  - CRPS;
  - posterior means of `Σ_T`, of `B`, and of the hyperparameters;
  - in backtest mode, per-origin scores.

**Noise floor**

Refit the user ordering `noise_reps` times with different seeds. For every statistic `s`, compute `σ_MC(s)` as the across-seed sd, and combine it with the ArviZ MCSE where available (SPEC §10.6). ARRS's 30-chain check (Fig. 3) is the precedent.

**Summary metrics** (per variable × horizon; joint where defined)

| Metric | Definition | Default flag (*heuristic, recommended here*) |
| :--- | :--- | :--- |
| `sd_ratio` | max/min of predictive sd across orderings | > 1.10 |
| `q_spread` | `(max − min)` of each quantile / median predictive sd | > 0.25 |
| `lpd_range` | `(max − min)` of LPD (forecast mode) or of the average LPS per origin (backtest) | > 0.05 nats per variable, > 0.10 joint |
| `signal_to_noise` | sd across orderings / `σ_MC` | > 3 (required for any flag) |
| `mean_shift` | `(max − min)` of the predictive mean / median predictive sd | > 0.10 |
| `rank_corr` | Spearman correlation of ordering rankings across variables and horizons (backtest) | reported only (ARRS Table 4) |
| equality test | max-pairwise DM/AG statistic with bootstrap p-value (ARRS used HLN 2011), or the Phase 0 MCS over orderings at 10% | reported |

`result.material` is true if any metric exceeds its threshold *and* `signal_to_noise > 3`. The thresholds are calibrated against the ARRS magnitudes: 0.25 nats/quarter joint, interval lengths visibly different (Fig. 5).

**Modes**

- **`"prior"`** (seconds, no posterior fit). Simulate from the prior of `(A, λ_{·,1:T}, B)` under each ordering. Report the implied prior median and 90% band of `Var(u_{i,t})`, `Corr(u_i, u_j)` and the prior-predictive sd by **position**. This exposes CKY's `2^{i−1}` inflation and confirms that the IW-equivalent prior removes it. It is useful as a fit-time warning (SPEC R8).
- **`"forecast"`** (default): one full-sample fit per ordering and forecasts from the last origin. Cost is `n_perm + noise_reps` fits.
- **`"backtest"`**: the Phase 0 backtest per ordering over `origins`. This is the only mode that can say an ordering *forecasts* better. Use the caching and `estimate_cost()`.

**`pool=True`** returns a `ForecastResult` whose draws concatenate equal-weighted draws from all orderings. Its LPD is the Rao–Blackwellised mixture: `logsumexp` over the pooled `(ordering, draw)` slices minus `log(total)`. With `design="all"` this predictive is exactly order-invariant; with sampled orderings it is invariant in expectation. This is a practical mitigation until Phase 5 (*suggestion*; ARRS and CKY rank orderings but do not pool).

**Output**

`OrderingSensitivityResult` is an xarray `Dataset` with dims `(ordering, variable, horizon, stat)`, plus `orderings` (a list of tuples), `noise`, `summary()` (a table with flags), `plot()` (an ARRS-Fig.1-style dot plot per variable relative to the user ordering), and the provenance attributes (seeds, design, spec hash).

**Integration**

- `fit()` emits a one-line R8 warning for non-invariant compositions with `n ≥ 4`, pointing to `ordering_sensitivity(mode="prior")`.
- The Phase 2 replication report includes a `mode="backtest"` run with `n_perm = 24` (all orderings of 4 variables).

## B5. How to test the ordering machinery

1. **Exact invariance (analytic).** For the Phase 1 NIW with permuted data and hyperparameters, all outputs after un-permuting must equal the identity-ordering outputs to `1e−10`. Property-based test (hypothesis) over random permutations and `n ≤ 6`.
2. **Invariance within MCSE (MCMC).** Run `Linear(form="reduced") × Constant(prior="iw_equivalent")` and `CommonSV` (exact `h`). After un-permuting, two-sample tests on posterior summaries and forecast quantiles must not reject at 1% (Holm-corrected).
3. **Prior-mode closed form.** With `a_ij ~ N(0, 1)`, `λ ≡ 1` and no SV, `mode="prior"` must return `E u_i² = 2^{i−1}` within Monte Carlo error (CKY §2; reproduced here by simulation: 1.00, 1.99, 4.00, 8.01, 16.01, 32.2 for `i = 1..6`).
   With the IW-equivalent prior, the marginal is `Σ_ii ~ IG((ν₀−n+1)/2, s_i²/2)`, the same for every position. At `ν₀ = n + 2` this is `IG(3/2, s_i²/2)`, with mean `s_i²` but **infinite variance**. So do not test the sample mean against `s_i²` (corrected). Instead:
   - KS-test `Σ_ii` against `IG(3/2, s_i²/2)`; or
   - compare medians or `log Σ_ii`; or
   - test with `ν₀ ≥ n + 4`, where the variance is finite.
4. **IW equivalence** (Chan 2022 Prop. 1). Sample `(A, λ)` from the IW-equivalent prior, form `Σ`, and compare it with direct `IW(ν₀, S)` draws: KS on `log|Σ|`, on `Σ_ii`, and on the correlations. Prototype run here (`n = 5`, 40,000 draws each, SciPy `invwishart`): KS p-values 0.24–0.94 for `log|Σ|`, `log Σ_11`, `log Σ_55`, `corr(5,1)` and `corr(3,2)`, and `Σ_55` vs `IG(3/2, s²/2)` p = 0.78.
5. **Detection power (DGP, weekly).**
   - Simulate from a CS-SV DGP with a strongly non-triangular covariance dynamic (CKY §6.1 `B_0 = [[1,−.8,−.8],[.8,1,−.8],[.8,.8,1]]`, `φ = 0.95`, `ω² = 0.05`, `T = 500`).
   - `ordering_sensitivity` on `Linear × SV` must flag `material=True`.
   - On a `CommonSV` DGP fitted with `CommonSV`, it must not flag, which controls the false-alarm rate.
6. **Un-permutation bugs.** Use synthetic data where each variable has a distinct scale (×10ⁱ). Any label mix-up shows up as order-of-magnitude errors in predictive sd.
7. **Parity per ordering.** `bayesianVARs` implements CCCM-2022 Cholesky SV. Run it under the user ordering and the reverse with matched priors (sibling note 04). Its per-ordering predictive quantiles must match `rai-bvar`'s `Linear(form="reduced") × SV` within 4 MCSE. This implies the sensitivity metrics match.

---

# Part C: Clark & Ravazzolo (2015): Phase 2 replication target

## C1. Versions and what was verified

- **Published**: *J. Applied Econometrics* 30(4), 551–575 (2015), DOI 10.1002/jae.2379. Bibliographic data confirmed via Crossref and the JAE archive readme.
  - Full text not accessible. The **abstract is now verified** via OpenAlex (`api.openalex.org/works/doi:10.1002/jae.2379`): the published paper evaluates "real-time point and density forecasts of key macroeconomic time series for the USA". It covers random-walk SV, stationary AR SV, SV with fat tails, GARCH and mixture-of-innovation models in AR and VAR form. Conventional SV "dominate[s]" for point forecasts "to some degree" and for density forecasts "to a greater degree".
  - So the JAE version is **US-only**, consistent with the archive. The WP's US conclusions carry over qualitatively.
  - JAE-specific *numbers* (tables, samples) remain **UNVERIFIED**.
- **Working paper**: Cleveland Fed WP 12-18 / Norges Bank WP 2012/09 (Sept. 2012), "The Macroeconomic Forecasting Performance of Autoregressive Models with Alternative Specifications of Time-Varying Volatility". Read in full; all specifics below come from it unless stated otherwise.
- **JAE data archive** (verified, re-read in cross-check): `readme.cr.txt`, `cr-data-csv.zip` at http://qed.econ.queensu.ca/jae/2015-v30.4/clark-ravazzolo/. It contains **US data only**; the UK appears in the WP, and the JAE abstract confirms US-only. The readme states that quarterly UR and T-bill are "simple averages of monthly observations within each quarter". Files:
  - `currentdata.csv`: UR, ITB3M, 1947Q1–2011Q3 (259 obs);
  - `realtimeGDP.csv`, `realtimeGDPpriceindex.csv`: 148 vintages each, **1975Q1–2011Q4**, from RTDSM (Feb/May/Aug/Nov vintages, ALFRED fill-ins).
- The WP uses vintages 1985Q1–2011Q4 for evaluation. Why the JAE archive starts at 1975 vintages is unknown (**UNVERIFIED**; possibly a longer evaluation window or the prior training).

## C2. Data (WP §2) and the FRED-QD mapping

| CR variable | CR definition | Source in CR | FRED-QD 2026-09 mnemonic (checked in the local cache) | FRED-QD default tcode | **Override for replication** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GDP growth | `100·Δlog` real GDP/GNP | RTDSM real-time | `GDPC1` | 5 | 5 (`100·Δlog`) |
| Inflation | `100·Δlog` GDP price index/deflator | RTDSM real-time | `GDPCTPI` (chain-type price index; FRED-QD has no `GDPDEF` column) | **6** | **5** |
| Unemployment | level, quarterly average of monthly | FAME, current vintage | `UNRATE` | **2** | **1** |
| 3-m T-bill | level, quarterly average of monthly | FAME, current vintage | `TB3MS` | **2** | **1** |

Details:

- **Scale.** The WP table notes say "annualized percentage points", but three pieces of evidence point to **non-annualized** quarterly `100·Δlog`:
  - WP §2 itself defines GDP growth and inflation as "100 times the log difference";
  - the reported RMSEs (VAR GDP 0.675 at h = 1, full sample) are about a quarter (ratio 3.9) of CCM 2016's annualized 2.622, for a similar model, data and evaluation period;
  - the VAR GDP average log score of −1.118 is consistent with a predictive sd of about 0.6 (≈ −0.92 + 0.51 − 0.5 ≈ −0.9 for a Gaussian), whereas the annualized CCM level is −2.42.

  Use `100·Δlog`. Rankings, LPS *differences*, CRPS *ratios* and RMSE *ratios* are invariant to rescaling a variable: the Jacobian cancels in LPS differences and CRPS scales linearly. Only score *levels* shift, so this does not affect the verdict. Annualization in the JAE version is **UNVERIFIED**.
- **Sample.** CR: estimation start fixed at **1955Q1**, with 1948–54 used for some priors. FRED-QD starts in 1959Q1, so `rai-bvar` must start later.
  - Recommended: estimation start **1965Q1** (as CCM 2016 / Clark 2011), keeping a 1959Q2–1964Q4 training window available for any training-sample prior. It must precede the window (R9).
  - Alternative closer to CR: start 1960Q2 (first usable observation after differencing, plus p = 4 lags).
- **Origins.** Recursive expanding window, origins 1985Q1–2011Q2 (US full sample) and 1985Q1–2007Q4 (pre-crisis). Optional extension to 2019Q4 (pre-COVID).
- **Horizons**: h = 1, 2, 4, 8, quarterly rates at each horizon (not 4-quarter averages; WP Table 1 note 1).
- **Real-time vs final.** CR's "1-quarter ahead" is the current-quarter nowcast: vintage `t` holds data through `t−1`. Actuals are the **second release** (vintage `τ+2` for target quarter `τ`) for GDP and inflation, and current data for UR and the T-bill (WP §2). The `rai-bvar` default (FRED-QD pinned, D4) is pseudo-real-time on final data. Map h = 1 to the first quarter after the last observation.
  - Optional exact-data mode: `data_at(origin)` (SPEC §5.5) built from the JAE archive files, following the vintage and actuals rules above. The data are downloaded on demand, never bundled.
- **Ordering.** The WP does not state the Cholesky ordering of the VAR. The data section lists GDP, inflation, unemployment, bill rate, which is also the table order: **UNVERIFIED** as the model ordering. Use that order and report `ordering_sensitivity(mode="backtest", design="all")` (24 orderings).

## C3. Models in CR (WP §3, App. 6) and their `rai-bvar` counterparts

Common to all VARs: p = 4. Minnesota prior **without cross-variable shrinkage**:

- `E[B_l^{ij}] = 0` (all variables are stationary transforms);
- `V[B_l^{ij}] = θ² σ_i² / (l² σ_j²)` for `l > 0`, with θ = 0.2;
- intercept variance `ε² σ_i²`, with ε = 1000;
- `σ_i²` = AR(4) residual variance on the estimation sample (App. 6.1, eq. (14)–(15), (22)–(23)).

| CR model | Specification (WP) | Prior / algorithm (WP App.) | `rai-bvar` composition (Phase 2) |
| :--- | :--- | :--- | :--- |
| VAR | `y_t = B(L) y_{t−1} + v_t`, `v_t ~ N(0, Φ)` | Normal-diffuse (Kadiyala–Karlsson 1997, eq. (16a/b)); Gibbs; 30,000 draws, burn 5,000, thin 5 | `Linear(prior=Minnesota(form="reduced", lambda_=0.2, cross=1, mean=0)) × Constant`. CR's diffuse `Φ` is approximated by the weakly informative `Constant` prior. Ideally use `iw_equivalent` with small `ν₀` |
| VAR-SV | `v_t = A⁻¹ Λ_t^{0.5} ε_t`, `log λ_{i,t} = log λ_{i,t−1} + ν_{i,t}`, `ν ~ N(0, φ_i)` | `A`: `N(0, 1000² I)`. `φ_i ~ IG_Clark(0.035·1, 1)` ("mean 0.035, 1 d.o.f."), i.e. `IG(1/2, 0.0175)` in shape–scale form. That density has **no finite mean**: "mean" is again Clark's name for the centre `a/d`. `log λ_{i,0} ~ N(log σ̂²_{i,0}, 4)` from AR(4) residuals regressed on earlier residuals over a 36-obs training sample of the last vintage. JPR (1994) single-move Metropolis (Cogley–Sargent 2005); 45,000 draws, burn 5,000, thin 8 | `Linear(form="reduced") × SV(phi=1 fixed)` (RW). Also run `× SV()` with the default AR prior; CR's own "VAR-ARSV" row is the counterpart |
| VAR-ARSV (a₁ = 0.9 / 0.8 / 0.5) | `log λ_{i,t} = a_{0,i} + a_{1,i} log λ_{i,t−1} + ν_{i,t}` | `a_{0,i} ~ N(0, 0.4²)`, `a_{1,i} ~ N(a₁, 0.05²)` (nearly fixed), independent | `× SV(phi_prior=Normal(0.9, 0.05))`, if `SV` exposes a Normal φ prior; otherwise a tight Beta. Mapping to `rai-bvar`'s centred SV (`μ`, `φ`): `φ = a₁`, `μ = a₀/(1 − a₁)`. CR's intercept prior therefore implies roughly `μ ~ N(0, (0.4/(1−a₁))²)`, about `N(0, 4²)` at `a₁ = 0.9`. That is not the `stochvol`-style `μ` prior, so set `μ`'s prior explicitly for parity (*derived here*) |
| VAR-SVt (d = 5 / 10 / 15, **fixed**) | `v_t = A⁻¹ Q_t^{0.5} Λ_t^{0.5} ε_t`, `d/q_{i,t} ~ χ²_d` (JPR 2004) | `q_{i,t} | · ~ IG`. The WP text says `(ỹ²/λ + d)/χ²(d)`. The correct full conditional is `(d + ỹ²/λ)/χ²_{d+1}`, i.e. `IG((d+1)/2, (d + ỹ²/λ)/2)`. The dof `d + 1` follows from `q ~ IG(d/2, d/2)` and one Gaussian observation (*derived here*), and is **confirmed by JPR (2004) eq. (6)**, `p(λ_t | y_t, h_t, ν) ~ IG((ν+1)/2, ·)`. So the WP text is a typo or an error; what the code did is **UNVERIFIED** | `× SV(tails="t", nu=fixed)` with the **unit-scale** t (`t_scale="unit"`, sibling note 03) for parity. The default estimates ν |
| VAR-GARCH, VAR-TVP-SV, AR-* and AR-mixture | GARCH(1,1) per orthogonalized error; Cogley–Sargent TVP | various | Out of scope in Phase 2 (TVP is Phase 5; no GARCH block) |

The AR benchmarks use 2 lags for GDP and 4 for the others (WP §3.1). They can be run as `n = 1` VARs if useful.

**R2 (fair ablation).** The Phase 2 verdict compares `Linear × SV` with `Linear × Constant` under the *same* `Linear` prior and form. CR also do this: both VAR and VAR-SV use the same Minnesota prior. Use `form="reduced"` for closest parity (CR's Minnesota is on the reduced form). Also report the default `form="structural"` pair; the ranking should not change.

## C4. Evaluation design and scores

- **Recursive estimation** at each origin. 5,000 retained draws in CR; `rai-bvar` uses 4 chains with an ESS target.
- **RMSE** of the posterior **mean** point forecast (WP App. 6.2–6.3: "RMSE based on the posterior mean"). The WP is internally inconsistent: App. 6.1 (constant-volatility VAR) says "based on the posterior median" (**UNVERIFIED** which was used). Use the mean for all models; the difference is negligible for near-symmetric predictives.
- **Average log predictive score.** CR use the **Gaussian approximation** of Adolfson, Lindé & Villani (2007), eq. (12): `s = −0.5[n log 2π + log|V| + (y − ȳ)'V⁻¹(y − ȳ)]`, with `ȳ, V` the mean and variance of the simulated predictive. Univariate scores are reported per variable.
  - `rai-bvar`'s primary LPD is the Rao–Blackwellised mixture (SPEC §7). It is exact and rewards SV's scale-mixture tails, so expect **larger SV gains** than CR's Gaussian-approximated LPS, especially in 2008–09.
  - For comparability, also compute `lpd_gauss` from path moments as an extra score. It equals −½ × the Dawid–Sebastiani score (plus a constant), which is **proper but not strictly proper**: it depends only on the first two moments (corrected; the first draft said "not a proper scoring rule"). See Krüger, Lerch, Thorarinsdottir & Gneiting, arXiv:1608.06802 (*Int. Stat. Rev.* 2021), and Gneiting & Raftery (2007). It cannot reward SV's tail shape beyond the variance, so keep it as a parity metric; the Rao–Blackwellised LPD stays primary.
- **CRPS** from draws, energy form `E|Y − y| − ½ E|Y − Y'|` (eq. (13)). `rai-bvar` uses `scoringrules` (validated in Phase 0).
- **Significance.** The WP tables report no tests. CCM 2016 use DM with the HLN small-sample adjustment, a rectangular kernel with h−1 lags, and **one-sided** tests against the constant-volatility benchmark. Use the Phase 0 DM/HLN and MCS (SPEC §9.3).

## C5. Headline results to reproduce qualitatively (WP 12-18, US, VAR block)

Table 1 (RMSE ratio vs VAR), Table 2 (average log score difference vs VAR), Table 3 (CRPS ratio vs VAR). Full sample 1985Q1–2011Q2:

| Variable | Metric | h = 1 | h = 2 | h = 4 | h = 8 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GDP growth | VAR-SV ΔLPS | **+0.193** | +0.133 | +0.090 | +0.060 |
| | VAR-SV CRPS ratio | 0.864 | 0.856 | 0.890 | 0.927 |
| | VAR-SV RMSE ratio | 0.896 | 0.878 | 0.932 | 1.007 |
| Inflation | ΔLPS | +0.075 | +0.083 | +0.155 | +0.138 |
| | CRPS ratio | 0.954 | 0.944 | 0.895 | 0.887 |
| | RMSE ratio | 0.966 | 0.949 | 0.915 | 0.883 |
| Unemployment | ΔLPS | +0.162 | +0.098 | **−0.182** | **−0.707** |
| | CRPS ratio | 0.922 | 0.897 | 0.906 | 1.001 |
| | RMSE ratio | 0.967 | 0.930 | 0.914 | 0.971 |
| 3-m bill | ΔLPS | **+0.284** | +0.043 | −0.304 | −0.403 |
| | CRPS ratio | 0.822 | 0.915 | 1.010 | 1.059 |
| | RMSE ratio | 0.933 | 0.944 | 0.974 | 0.990 |

Pre-crisis 1985Q1–2007Q4, VAR-SV ΔLPS at h = 1 / h = 8:

| Variable | h = 1 | h = 8 |
| :--- | :--- | :--- |
| GDP | +0.222 | +0.214 |
| Inflation | +0.077 | +0.152 |
| Unemployment | +0.187 | −0.065 |
| Bill | +0.316 | −0.261 |

Further patterns:

- VAR-ARSV (a₁ = 0.9) helps long-horizon unemployment and the bill rate in the full sample: ΔLPS at h = 8 is −0.120 and −0.109 respectively, against VAR-SV's −0.707 and −0.403.
- VAR-SVt (d = 10) is similar to or slightly below VAR-SV (GDP h = 1: +0.190).
- VAR-GARCH is dominated (inflation ΔLPS about −0.3 at all h).

Benchmark VAR levels (full sample, h = 1): average log score GDP −1.118, inflation −0.235, UR −0.034, bill −0.832; RMSE GDP 0.675, inflation 0.269, UR 0.240, bill 0.454.

**Phase 2 acceptance criteria (qualitative; proposed)**

1. `Linear × SV` beats `Linear × Constant` in average LPD at h = 1 and 2 for all four variables, in both samples. The joint LPD gain at h = 1 should be clearly positive. CR's marginals sum to about +0.7 nats; CCM 2016 report +0.97 joint for 1985–2013.
2. CRPS ratio < 1 at h ≤ 2 for all four variables.
3. RMSE ratios ≤ about 1 at short horizons (small gains: GDP and bill 5–10%).
4. In the full sample, the long-horizon (h = 8) LPD gain shrinks or turns negative for unemployment and the bill rate (the Great Recession effect, WP §4.1). It stays positive pre-crisis for GDP and inflation.
5. Neither `SV(tails="t")` nor the AR-SV variant systematically beats RW/AR `SV`.
6. The `compare()` verdict for `SV` vs `Constant` is "better" (DM/HLN one-sided at 10% for at least GDP and the bill at h = 1). This holds in **all 24 orderings** (§B4 backtest mode), or the report states which orderings fail.

## C6. Pitfalls specific to this replication

- **LPD estimator mismatch** (C4): report both. The rankings should agree.
- **Final vs real-time data**: SV gains in CR partly reflect volatility shifts that are robust to revisions, so expect qualitatively the same ranking on final data. Level differences are expected.
- **`GDPCTPI` (chain-type)** vs CR's GDP/GNP deflator or fixed-weight index: same concept, different vintages.
- **FRED-QD `UNRATE` and `TB3MS`** are quarterly averages of monthly values in FRED-QD, as in CR (**verified empirically in cross-check**).
  - FRED-QD 2026-09 `TB3MS` equals CR's archive `ITB3M` (within-quarter averages, per the JAE readme) to ≤ 0.0034 over all 211 overlapping quarters, 1959Q1–2011Q3.
  - `UNRATE` matches to a mean absolute difference of 0.004 (max 0.13), consistent with seasonal-adjustment revisions since 2011.
  - The loader should still assert this for future vintages.
- **Prior differences**:
  - CR's RW-SV prior on `φ_i` (IG with 1 d.o.f., scale 0.035) is very diffuse, unlike `rai-bvar`'s stochvol-style defaults. For parity runs, expose `SV(rw=True, sigma2_prior=IG_Clark(0.035, 1))`.
  - CR's prior on the initial volatility uses a training sample from the *last* vintage for all origins (App. 6.2). That is mild look-ahead and violates R9. Do **not** replicate it; use in-window quantities.
- **Ordering**: see §B3. At h = 1 the 4-variable SV gains are about 3× the joint ordering spread, and 1.9–3.5× the per-variable marginal spreads (smallest for the T-bill and unemployment). At h ≥ 4 the gains shrink to within the spread.

---

## D. UNVERIFIED items

1. **JAE 2015 published version (CR)**: evaluation sample, exact model list, annualization, presence of significance tests, and the reason the archive vintages start in 1975Q1. The abstract is now verified (OpenAlex): US-only, and conventional SV dominates. The full text is still unread.
2. **CR model ordering** of the Cholesky VAR (WP does not state it).
3. **CR VAR-SVt `q_{i,t}` draw**: the WP text says `χ²(d)`; the correct dof is `d + 1`, derived here and confirmed by JPR 2004 eq. (6). What their code did is unknown. Also unknown: whether CR's RMSE used the posterior mean or median (App. 6.1 vs 6.2–6.3).
4. **CCM 2016 offset `c̄`** for `log(ṽ² + c̄)`, and whether "initial value of ln f_t: mean 0, variance 0.5" refers to `t = 0` or `t = 1`.
5. **CCM 2016 eq. (23)**: whether their code used the printed (incomplete) `Σ` conditional or the exact one.
6. **`bvars`**:
   - the SV state equations and prior densities (only field names are documented: `rho`, `omega`, `sigma2_omega`, `sv_a`, `sv_s`, `S` a T-vector of mixture indicators, suggesting **one** mixture indicator per `t`, i.e. a mixture approximation of a `log χ²_n`-type quantity);
   - the `IG2(ν−2, ν)` parametrisation for the t errors. Mean 1 under the Bauwens–Lubrano–Richard / Clark–Ravazzolo `IG-2(s, ν)` convention (mean `s/(ν−2)`), but `bvars` does not define it;
   - whether `V` can be fixed (needed for exact parity).
7. **Chan (2017/2020) MH acceptance rates** for the common-volatility `h` step. Chan (2020) reports only inefficiency factors (h: 75% below 10, max below 30). Cross-check simulation (§A5.2 step 4) gives our own reference values: flat in `n`, decreasing in `T` and `σ_h²`. The blocked variant is *derived here* and untested against a published reference.
8. **ARRS published (JoE 2023) numbers vs WP (rev. Mar 2022)**: the numbers quoted are from the WP; I assume they are identical in print. **CKY** numbers are from the Strathclyde accepted manuscript. They match the March 2022 WP for Table 5 (ALPL); the log-ML (Table 3) and the m-SV table appear only in the accepted manuscript.
9. **CCM 2019 supplementary appendix §C** (1,000 random orderings for one period), cited by ARRS §1: not read.
10. ~~FRED-QD quarterly aggregation~~: **resolved** in cross-check. `UNRATE` and `TB3MS` are within-quarter averages (§C6).
11. **Constant-`A` ordering spread for a 4-variable VAR-SV** (as opposed to ARRS's time-varying `A_t`): no published number found. Expect it to be smaller than ARRS's 0.25 nats/quarter (§B3 caveat 2).

## E. Sources (URL; sections and equations used)

- **Carriero, Clark & Marcellino (2016)**, JBES 34(3), 375–390. Accepted manuscript: https://cris.unibo.it/retrieve/e1dcb339-5848-7715-e053-1705fe0a6cc9/carriero.pdf. Used: §2.2 eq. (10)–(14); §2.4 eq. (17)–(21); §2.5 eq. (22)–(32); §2.6 eq. (33); §3.1 priors; §3.2 sampler; Tables 1–7; App. 6.1–6.3.
  Working paper: Cleveland Fed WP 12-06, https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2012/wp1206.pdf, §2.1–3.3 (earlier A/S/RW variant).
- **Chan (2020)**, "Large Bayesian VARs: A flexible Kronecker error covariance structure", JBES 38(1), 68–79. https://opus.lib.uts.edu.au/bitstream/10453/139991/1/BVAR.pdf. Used: §2 eq. (1)–(3) and fn. 1; §3 eq. (4)–(5) (NIW given Ω, sampling); §4.2 priors (`ν₀ = n+3`, `S₀ = I`, `κ = (0.2², 10²)`, `σ_h² ~ IG(5, 0.04)` with stated prior mean 0.1², `ρ ~ N(0.9, 0.2²)1(|ρ|<1)`, `ν ~ U(2,100)`); §4.3 and Table 2 (independence-chain MH for `h`; BVAR-CSV 0.46 min); Online App. A2 Examples 1–2 (samplers, ARMH wording); Online App. A integrated-likelihood section (BVAR-t-CSV `p(Y|h,A,Σ,ν)`).
- **Chan (2022)**, "Asymmetric conjugate priors for large Bayesian VARs", QE 13, 1145–1169. arXiv:2111.07170, https://arxiv.org/pdf/2111.07170v1. Used: §2 Prop. 1, Cor. 1–3 (IW ⇔ position-dependent NIG), baseline `ν_i = 1 + i/2`.
- **Chan (2023)**, "BVARs and Stochastic Volatility" (handbook chapter), arXiv:2310.14438, https://arxiv.org/pdf/2310.14438. Used: §2.1 (CSV), §3.1 (order invariance), §4.2.1 (DP2015 order), §4.2.3 (structural priors are order-dependent; the first draft said §4.3).
- **Arias, Rubio-Ramírez & Shin (2023)**, JoE 235(2), 1054–1086. WP 21-21 (rev. Mar 2022): https://philadelphiafed.org/-/media/FRBP/Assets/working-papers/2021/wp21-21.pdf. Used: §2.1 priors; §2.2 analytic example; §3.1 set-up; §3.2–3.5 Tables 1–6, Figs. 1–6; §4 order-invariant alternatives.
- **Chan, Koop & Yu (2024)**, JBES 42(2), 825–837.
  Accepted manuscript: https://strathprints.strath.ac.uk/85925/1/Chan_etal_JBES_2023_Large_order_invariant_bayesian_VARs_with_stochastic_volatilitypdf.pdf. Used: §2 (the `2^{i−1}` example); §3 Props. 1–2; §4.2 prior and fn. 6; §6.1–6.3 (Tables 2–4); §7 Tables 5–6; Appendix A data.
  Also the March 2022 version: https://economics.princeton.edu/wp-content/uploads/2021/12/OISV-Chan.pdf.
- **Carriero, Clark & Marcellino (2019) WP 16-17** (local copy from the sibling research run), §3.1 "The role of variables ordering".
- **Clark & Ravazzolo (2012)**, Cleveland Fed WP 12-18. https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2012/wp1218.pdf. Used: §2 data; §3.1–3.3 models; §4 eq. (12)–(13), Tables 1–3; App. 6.1–6.6.
- **Clark & Ravazzolo (2015)**, JAE 30(4), 551–575. Data archive: http://qed.econ.queensu.ca/jae/2015-v30.4/clark-ravazzolo/ (`readme.cr.txt`, `cr-data-csv.zip`, inspected).
- **`bvars` manual** (documentation only): https://bsvars.r-universe.dev/bvars/doc/manual.html. README: https://github.com/bsvars/bvars.
- **FRED-QD 2026-09** (`rai-bvar` local cache): tcodes for GDPC1 (5), GDPCTPI (6), UNRATE (2), TB3MS (2); data from 1959Q1. There is no `GDPDEF` column (re-checked in cross-check).
- **Clark & Ravazzolo (2015) abstract**: OpenAlex `https://api.openalex.org/works/doi:10.1002/jae.2379`. Bibliographic data from Crossref: JAE 30(4) 551–575, DOI 10.1002/jae.2379.
- **Jacquier, Polson & Rossi (2004)**, "Bayesian analysis of stochastic volatility models with fat-tails and correlated errors", *J. Econometrics* 122(1), 185–212. Used: eq. (5)–(6) (`λ_t` conditional `IG((ν+1)/2, ·)`).
- **Krüger, Lerch, Thorarinsdottir & Gneiting (2021)**, "Predictive inference based on Markov chain Monte Carlo output", *Int. Stat. Rev.*; arXiv:1608.06802. Used: the Dawid–Sebastiani score is proper but not strictly proper.
- **Crossref** (`api.crossref.org`): confirmed ARRS JoE 235(2) 1054–1086, DOI 10.1016/j.jeconom.2022.04.013; CKY JBES 42(2) 825–837, DOI 10.1080/07350015.2023.2252039; Chan (2020) JBES 38(1) 68–79, DOI 10.1080/07350015.2018.1451336; Chan (2022) QE 13(3) 1145–1169, DOI 10.3982/QE1381.
- **Cross-check scripts**: `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc08_checks/checks.py` (integrated likelihood, `K_h`, Chan Prop. 1, CKY `2^{i−1}`, MH acceptance).

---

## Cross-check log (independent verification pass, 2026-10-21)

**Method.**
- Re-extracted every primary PDF with `pdftotext -layout` into `phase2-research/xc08/` and read the cited sections directly, rather than relying on the first author's text dumps:
  - CCM 2016 accepted manuscript (AAM) and Cleveland Fed WP 12-06;
  - Chan 2020 (Dec 2017 accepted version, including the Online Appendix);
  - Chan 2022 (arXiv v1);
  - Chan 2023 survey;
  - CKY AAM (June 2023) and the March 2022 WP;
  - ARRS WP 21-21 (rev. Mar 2022);
  - CR WP 12-18;
  - CCM 2019 WP 16-17;
  - JPR 2004.
- Re-fetched the JAE archive readme. Re-read the `bvars` manual (documentation only).
- Confirmed bibliographic data through Crossref and the CR (2015) abstract through OpenAlex.
- Re-derived every formula, then checked the key ones numerically (`scratchpad/xc08_checks/checks.py`). No GPL code was opened.

**Verified as written (no change needed).**
- Model and identification: CCM eq. (10)–(14), (17)–(21), (22)–(29), (33)–(35).
- CCM priors: θ = 0.2, ε = 1000, `d_s = n+2` with a VAR(1) on a 40-obs training sample, `ψ` mean 0.9 / sd 0.2 (0.3 for GFSV), `φ` "mean 0.01, 4 dof", `ln f` initial (0, 0.5).
- Clark-convention IG: "scale term a, dof b" means `IG(b/2, a/2)`, so `φ ~ IG(2, 0.02)` with mean 0.02.
- IW mean `(n+2)Σ̲` and `E[Σ⁻¹]⁻¹ = Σ̲`.
- The CCM eq. (23) defect (missing `(Π−μ)'Ω₀⁻¹(Π−μ)` and `k` dof).
- CCM DP2014 reordering and step order.
- CCM Tables 1, 3, 6 numbers, "51 of 70", and the one-sided HLN DM tests.
- Chan 2020: eq. (1)–(5), fn. 1, NIW-given-Ω posterior, §4.2 priors (`ν₀ = n+3`, `S₀ = I`, `κ = (0.2², 10²)`, `IG(5, 0.04)`, `N(0.9, 0.2²)1(|ρ|<1)`, `U(2,100)`), §4.3 sampler differences, Table 2 (0.46 min), IFs (h: 75% < 10, max < 30).
- Chan 2022 Prop. 1 and Cor. 1–3, independently re-derived from the IW block-marginal / Bartlett results and checked by simulation.
- CKY: §2 `2^{i−1}` (re-derived and simulated), priors `N(0.95, 0.05²)` and `IG(3, 0.1)` (shape–scale), Tables 2, 3, 5, 6 and the §6.1 DGP.
- ARRS: setup, draws, Tables 1–2 ranges and spreads, AG test 9/15, DM test 2/12, 30 × 20,000 chains, §3.5 conditional forecasts, 20! ≈ 2.43·10¹⁸.
- CCM 2019 §3.1 quotes.
- CR: data, models, priors, draws, and every Table 1–3 number quoted in §C5 (all 40 cells re-read).
- JAE archive contents.
- FRED-QD tcodes.
- Algebra:
  - the likelihood and `q_t`, the gradient and Hessian, `K_h`, and the independence-MH acceptance ratio;
  - the ρ truncated-normal MH with `g(ρ)`, the `σ_h²` IG update, and the CCM-parity `ψ`/`φ` updates;
  - the t-error `λ_t` conditionals (unit and standardized);
  - the structural ⇔ NIW mapping `vec(ΠA') = (A⊗I)vec Π ~ N(·, Λ⊗V)`;
  - the n = 2 correlation formula and the RW `K_h` (with `h_0 = 0`).
- The integrated likelihood `log p(Y|h)` (§A4) matches the basic marginal-likelihood identity to 6e−14.

**Corrections made (with source).**
1. **§A5.2 step 4 / §0 / §A11 test 4: the MH acceptance claim was wrong.** The draft said acceptance rises with `n` and set a soft check of "> 0.5 for `n ≥ 3`". Simulation shows acceptance roughly flat in `n` and falling with `T` and `σ_h²`: 0.6–0.8 at `T = 100–300, σ_h² = 0.02`; about 0.3 at `T = 300, σ_h² = 0.1`; 0.05–0.2 at `T = 600–1000, σ_h² = 0.1`. Added a reference table, an exact blocked independence-MH variant (*derived here*), and a recalibrated soft check. Source: own simulation, `xc08_checks/checks.py`.
2. **§A5.2 step 1: Newton start point.** The draft started Newton "from the current draw", which makes the proposal state-dependent unless Newton converges exactly. Now it starts from `log(q_t/n)` or 0. Source: independence-MH theory (the proposal must not depend on the current state). Converges in 5–8 iterations in simulation.
3. **§0 / §A5.2: Chan's own description.** Chan (2020) main text §4.3 says "independence-chain MH"; Online App. A2, Example 2 says "acceptance-rejection MH" with the same `N(ĥ, K_h⁻¹)` proposal. Both are now cited, noting that both are exact.
4. **§A11 Geweke: the Omori variant is never exact under a full sweep.** The draft claimed a mixture-generated DGP makes it exact. In fact the `(Π,Σ)` block uses the Gaussian likelihood and the `h` block uses the mixture and offset, so the conditionals are incompatible. Exactness holds only in a block-level test with `(Π,Σ)` fixed and `c̄ = 0`. Source: CCM 2016 §2.6 and §3.2 sampler structure; derivation.
5. **§A11 SBC: `Σ_t q_t e^{−h_t}` statement.** It is `χ²_{nT}` at the data-generating `θ`, **not** across posterior draws for fixed `Y` (the draft said "under the posterior predictive"). Source: derivation.
6. **§A7: flop count.** QR of the dummy-augmented design is about `2(T+k_d)k² − (2/3)k³ ≈ 1.8·10⁸`, not `T k² ≈ 5·10⁷`, at `n = 100, p = 4, T = 300`. Source: standard Householder QR cost. Also added missing RNG keys (`σ_h²`, block offset).
7. **§A5.3: CCM-parity `K_h` leading entry.** Spelled out `K_h[0,0] = 1/0.5 + ψ²/φ` and the rest, checked numerically against `H'S⁻¹H`. The draft's wording ("prior precision 1/0.5") omitted the `ψ²/φ` term.
8. **§B1: coefficient-prior invariance.** Added the requirement of a **common** `V` across equations (equation-specific `V_i` breaks the Kronecker form), plus the Jacobian `|A|^k = 1`. Source: derivation; Chan survey §4.2.3.
9. **§B1 and §E: wrong section citation.** Chan survey: "order-dependent priors" is in **§4.2.3**, not §4.3. Source: arXiv:2310.14438 text.
10. **§B1(iii) table: CKY RMSFE.** Equal to 3 decimals only for IP, UR and PCE. FFR differs by up to 2.2% (2.216 vs 2.264 at h = 12). Source: CKY AAM Table 5.
11. **§B3: ARRS Spearman range.** −0.08 to 0.45 (correlations with the 1-q output-growth ranking), not "mostly 0.05–0.45". Source: ARRS WP Table 4.
12. **§B3: SV gain vs ordering spread.** Added caveats:
    - CR's 0.71 is a sum of marginal differences, not a joint difference;
    - ARRS uses time-varying `A_t` (CKY Tables 5–6 show this amplifies ordering effects);
    - the draft compared the CR inflation cell with the *joint* spread. The per-variable ratios are 1.9–3.5.

    Sources: ARRS Table 2; CR Table 2; CKY Tables 5–6.
13. **§B5 test 3: infinite variance.** `Σ_ii ~ IG(3/2, s_i²/2)` at `ν₀ = n+2` has infinite variance, so a sample-mean test against `s_i²` is invalid. Replaced with KS, median or log tests, or `ν₀ ≥ n+4`. Source: IW marginal (derivation); KS p = 0.78 in simulation.
14. **§C1: the JAE abstract is available.** It confirms a US-only published paper and its headline conclusion. Source: OpenAlex `works/doi:10.1002/jae.2379`; Crossref.
15. **§C2: annualization evidence strengthened.** WP §2 explicitly defines `100×Δlog`. The VAR GDP log-score level (−1.118) also implies non-annualized data. Source: CR WP §2, Table 2.
16. **§C3: VAR-SV and VAR-SVt rows.**
    - `IG_Clark(0.035, 1) = IG(1/2, 0.0175)` has no finite mean.
    - The `d+1` dof is confirmed by JPR (2004) eq. (6), so it is no longer just "derived here".
    - Added the CR `(a₀, a₁)` → `rai-bvar` `(μ, φ)` prior mapping.

    Sources: CR WP App. 6.2–6.3; JPR 2004 eq. (6).
17. **§C4: RMSE point forecast.** The CR WP is inconsistent (App. 6.1 says median; App. 6.2–6.3 say mean). Source: CR WP App. 6.1–6.3.
18. **§C4: Gaussian-approximation log score.** It is the Dawid–Sebastiani score, which is **proper** (not strictly proper). The draft said "not a proper scoring rule". Source: Krüger et al. (2021, arXiv:1608.06802); Gneiting & Raftery (2007).
19. **§C6 / §D10: FRED-QD aggregation verified and resolved.** `TB3MS` matches CR's within-quarter averages to ≤ 0.0034 over 211 quarters; `UNRATE` matches to a mean of 0.004. Source: JAE readme; FRED-QD 2026-09 local cache vs `currentdata.csv`.
20. **§A4 and §A5.4: added complementary Chan (2020) results.** Online App. A gives the `h`-integrated likelihood by importance sampling and the closed-form `λ`-integrated BVAR-t-CSV likelihood; App. A2, Example 1 gives the `ν` sampler.
21. **§A11 parity (`bvars`): added a second source for the `IG2(s, ν)` mean-`s/(ν−2)` convention.** Source: CR WP App. 6.4 eq. (26). `bvars`' own convention remains UNVERIFIED.

**Still UNVERIFIED after the cross-check** (see §D):
- JAE-specific CR numbers, and CR's VAR ordering;
- CR's code for the `q` draw, and mean vs median for RMSE;
- CCM's offset `c̄`, the `h_0` vs `h_1` timing, and whether their code used the printed eq. (23);
- `bvars` SV equations, its IG2 convention, and whether its `V` can be fixed;
- Chan's own acceptance rates;
- ARRS print vs WP numbers;
- the CCM 2019 supplementary appendix;
- the constant-`A` ordering spread for 4 variables;
- the blocked MH variant (derived here, not yet Geweke-tested).
