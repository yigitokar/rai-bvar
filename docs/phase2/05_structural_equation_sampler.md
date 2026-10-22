# 05: Recursive structural-form equation sampler (SPEC §4.1, §4.3, decision D3)

Status: research note for Phase 2. Clean-room: every formula below is derived from the papers
cited or derived here, and checked numerically by our own scripts (§13.2). No GPL source was read.
Anything not confirmed from a primary source or by our own check is marked **UNVERIFIED**.

---

## 0. Key results (read this first)

1. **Equation `i` is a weighted Gaussian regression** of `y_i` on `(−y_{<i}, x)` with weights
   `w_{i,t} = 1/(λ_{i,t} ω_{i,t})`. Because `A` is unit lower triangular, `|A| = 1`, so the likelihood
   factorises over equations exactly. With priors independent across equations (given global
   hyperparameters), the `n` equation posteriors are conditionally independent and `vmap` is exact,
   unlike the reduced-form case (CCCM 2022).
2. **D3 prior.** `a_i ~ N(m_a, V_a)` and `γ_i | a_i ~ N(M ã_i, V_γ)`. This is jointly Gaussian. The
   cleanest implementation reparametrises to `δ_i = γ_i − M_{<i} a_i` (unit Jacobian). Then
   `(a_i, δ_i)` are a priori independent with a **block-diagonal (diagonal) prior precision**, and
   the regressors become `[X M_{<i} − Y_{<i}, X]`, which are minus the prior-mean residuals
   ("quasi-differences") of earlier variables.
3. **D3 centring reproduces the conjugate Minnesota prior exactly.** The Phase 1 NIW prior
   (`Σ ~ IW(diag ψ, d)`, `B|Σ ~ MN(M, Σ⊗Ω)`) is identical to the structural normal-inverse-gamma
   (NIG) prior with:
   - `σ_i² ~ IG((d−n+i)/2, ψ_i/2)`,
   - `a_i | σ_i² ~ N(0, σ_i² diag(1/ψ_j))`,
   - `γ_i | a_i, σ_i² ~ N(M ã_i, σ_i² Ω)`.

   Our script confirms this to machine precision: the log marginal likelihood difference is
   `0.0`, and the posterior identities hold to `≤ 1e-14`. The cross-check repeated this against
   the actual Phase 1 engine (`rai_bvar.engines.analytic`, with SOC and SUR dummies, `n = 4`):
   the difference is again `0.0` (§13.2). So D3 centring is not ad hoc. It is the structural form
   of the conjugate prior. Sims & Zha use the same conditional centring: `A_+ = D + S A_0` with
   `S = [I_n; 0]` and a zero-mean prior on `D` (Sims & Zha 2006, eq. 7). That is our δ-coordinates
   up to normalisation. Chan (2022) uses only an *unconditional* mean (his Appendix D; Online
   Supplement §2 in the QE version).
4. **Analytic oracle in Phase 2 already.** In "conjugate mode" (prior covariances scaled by `λ_i`),
   `Linear(form="structural") × Constant` with `cross = 1` and `κ_a = 1` has *the Phase 1 NIW posterior*
   as its exact target. This gives an end-to-end oracle at any `n` without waiting for Phase 3. With
   `cross ≠ 1` the target is Chan's (2022) per-equation NIG closed form, which takes about 60 lines
   (§9).
5. **Recommended default (independent mode, "moment-matched").** For `d = n+2`:
   - `Var(a_ij) = κ_a ψ_i/(i ψ_j)`,
   - `V_γ,i = (ψ_i/i)·Ω^{(i)}`,
   - `Constant`: `λ_i ~ IG(1 + i/2, ψ_i/2)`.

   With `cross = 1` this reproduces the conjugate prior's reduced-form mean (exactly, for every
   `A`), its marginal variances and `E[Σ] = diag ψ`. These equalities are exact; the cross-check
   proved them algebraically (§4.3). Monte Carlo check: ratio `1.00 ± 0.01`.
   The naive `a_ij ~ N(0, ψ_i/ψ_j)` gives `E[Σ_ii] = 2^{i−1} ψ_i` (CKY 2024 §2; our MC check gives
   `1, 2, 4`).
6. **Back to reduced form:**
   - `B' = A⁻¹Γ` (triangular solve), with `B − M = (Δ − M')' A⁻¹'`.
   - `chol(Σ_t) = A⁻¹Λ_t^{1/2}`, so no Cholesky is needed.
   - Forecasts use the structural recursion directly.
   - IRFs are `Φ_h A⁻¹` for unit shocks and `Φ_h A⁻¹ Λ_t^{1/2}` for one-sd shocks.

---

## 1. Sources

| Ref | What we use | Where |
| :--- | :--- | :--- |
| Chan (2022), *Asymmetric conjugate priors for large Bayesian VARs*, QE 13(3) 1145–1169, DOI 10.3982/QE1381 | Structural NIG prior per equation (eq. 4), posterior (eq. 10), closed-form ML (eq. 11), IW↔NIG mapping (Prop. 1, Cor. 1–3, eqs. 6–9), defaults (`ν_i = 1+i/2`, `S_i = s_i²/2`, `V_α,i = diag(1/s_j²)`, `κ_3 = 100`, `s_i²` from AR(4)), App. D (structural vs reduced-form prior moments), EB values `κ̃_1 = 0.058`, `κ̃_2 = 0.0043` (§4.2, Table 2) | arXiv:2111.07170v1 (https://arxiv.org/abs/2111.07170), §2.1–2.4, App. B–D. Published PDF: https://www.econometricsociety.org/publications/quantitative-economics/2022/07/01/Asymmetric-conjugate-priors-for-large-Bayesian-VARs/file/QE1381.pdf. **Cross-checked against the QE PDF:** eqs. (1)–(11), Prop. 1, Cor. 1–3, `κ_3 = 100` and Table 2 are identical. The arXiv App. B (ML derivation) is App. B in QE. The arXiv App. C (proofs) and App. D (implied structural hyperparameters) are §1 and §2 of the QE *Online Supplementary Material* (https://www.econometricsociety.org/publications/quantitative-economics/2022/07/01/Asymmetric-conjugate-priors-for-large-Bayesian-VARs/supp/1638-7628-1-SP.pdf) |
| Chan, Koop & Yu (2024), *Large order-invariant Bayesian VARs with SV*, JBES 42(2) 825–837 | Order-dependence illustration `Var(u_i) = 2^{i−1}` (§2); OI model with dense `B_0` (eq. 4), SV zero-mean AR(1) (eq. 3), identification (Prop. 1, citing Bertsche & Braun 2020), invariance (Prop. 2), Waggoner–Zha/Villani row sampler (§4.3, eq. 12), priors (§4.2, eqs. 5–8; §6) | arXiv:2111.07225v1 (https://arxiv.org/abs/2111.07225); DOI 10.1080/07350015.2023.2252039. **Cross-checked against the JBES accepted author manuscript** (https://strathprints.strath.ac.uk/85925/): eqs. (3), (4), (5), (8), (12), Props. 1–2, the `2^{i−1}` example (§2), the `N(0,1)`/`N(1,1)` `B_0` priors (§6) and the positive-diagonal sign normalisation (§3.2) all match. The typeset version of record was not seen |
| Carriero, Chan, Clark & Marcellino (2022), Corrigendum, JoE 227(2) 506–512, DOI 10.1016/j.jeconom.2021.11.010 | Corrected triangular algorithm: eqs. (11)–(20), especially (14)–(15); footnote 2 (independence across equations, and how to relax it by a triangular factorisation of the prior); speed comparison §4 (`3.25×` slower than the flawed algorithm and `13×` faster than system-wide at `N = 20`) | Cleveland Fed WP 16-17R corrigendum, DOI 10.26509/frbc-wp-201617corrigendum, https://www.clevelandfed.org/publications/working-paper/2021/wp-1617c-large-vector-autoregressions-with-stochastic-volatility-and-flexible-priors. Equation and footnote numbers are those of the WP version (the JoE typeset numbering was not seen) |
| Sims & Zha (1998), IER 39(4) 949–968 | Precedent for the conditional prior `f_i | a_i ~ N(P̄ a_i, H̄_i)` (structural lag coefficients centred linearly on the contemporaneous row) | https://ideas.repec.org/a/ier/iecrev/v39y1998i4p949-68.html. Form confirmed via MSBVAR docs https://rdocumentation.org/packages/MSBVAR/versions/0.9-2/topics/szbsvar. The 1998 paper itself was not obtained, so its equation number is **UNVERIFIED** |
| Sims & Zha (2006), *Were there regime switches in U.S. monetary policy?*, AER 96(1) 54–81 | Explicit random-walk form of the Sims–Zha centring: `A_+ = D + S A_0`, `S = [I_n; 0_{(m−n)×n}]`, zero-mean prior on `D`, "centered on the same reduced-form random walk model … (Sims and Zha 1998a)" (eq. 7); unit-variance structural shocks; `x_t` ordered lags first, constant last | NBER Summer Institute 2004 draft, https://conference.nber.org/confer/2004/si2004/me/sims.pdf, §II eqs. (1)–(7). The AER equation numbering was not seen |
| Inverse-Wishart partition theorem | `Σ ~ IW(Ψ,ν)`: `Σ_11 ~ IW(Ψ_11, ν−p_2)`; `Σ_22·1 ~ IW(Ψ_22·1, ν)`; `Σ_11⁻¹Σ_12 | Σ_22·1 ~ MN(Ψ_11⁻¹Ψ_12, Ψ_11⁻¹ ⊗ Σ_22·1)`; independent of `Σ_11` | Standard result (e.g. Gupta & Nagar 2000, *Matrix Variate Distributions*; the theorem number is **UNVERIFIED**). Chan (2022) Prop. 1 is the diagonal-`Ψ` case. **Verified numerically here** (§13.2) |
| `bayesianVARs` reference manual, v0.1.8 (docs only) | `Σ_t = U'⁻¹ D_t U⁻¹` with `U⁻¹` upper unitriangular; `cholesky_U_prior` options (HS listed first); `cholesky_normal_sds = 10`, which the docs describe as *prior variances* despite the name; `prior_phi` default `"HS"`; `prior_intercept = 10` (documented as a standard deviation); **`bvar()` defaults to `specify_prior_sigma(type = "factor")`**; the package states it implements the corrected CCCM algorithm | https://cran.r-project.org/web/packages/bayesianVARs/refman/bayesianVARs.html |
| `bvartools` reference manual, v0.3.0 (docs only) | Structural model `A_0 y_t = Σ A_i y_{t−i} + … + u_t`, with `A_0` unit lower triangular, drawn jointly with the other coefficients in SUR form; default `coef = list(v_i = 1, v_i_det = 0.1, …)` (precisions); "for structural models or models with stochastic volatility only a gamma prior specification is allowed" for the errors | https://cran.r-project.org/web/packages/bvartools/refman/bvartools.html |
| `stochvol` reference manual v3.2.9 (docs only) and Hosszejni & Kastner (2021), *Modeling univariate and multivariate stochastic volatility in R with stochvol and factorstochvol*, JSS 100(12), arXiv:1906.12123 | Model `y_t = x_t β + exp(h_t/2) ε_t`, `h_{t+1} = μ + φ(h_t − μ) + σ η_t`, `h_0 ~ N(μ, σ²/(1−φ²))` (JSS eq. 1); **t errors are `t_ν(0, 1)` with unit *variance*** (JSS eq. 2); defaults `specify_priors(mu = sv_normal(0, sd = 100), phi = sv_beta(5, 1.5), sigma2 = sv_gamma(0.5, 0.5), beta = sv_multinormal(mean = 0, sd = 10000))`; `svlm` (formula wrapper) | https://cran.r-project.org/web/packages/stochvol/refman/stochvol.html; https://arxiv.org/abs/1906.12123 §2.1–2.2 |
| Geweke (2004); Talts et al. (2018); Modrák et al. (2025); Säilynoja, Bürkner & Vehtari (2022, *Stat. Comput.* 32:32, ECDF envelope) | Test methodology | SPEC §15 |
| Del Negro & Primiceri (2015) | Order of indicator and log-volatility draws | SPEC §4.3 |

---

## 2. Notation

Indices are 1-based in the maths and 0-based in code. The rai-bvar regressor layout is
`x_t = (d_t', y_{t−1}', …, y_{t−p}')'`: lag `l` of variable `j` is row `n_d + (l−1)n + j` (0-based `j`).
This matches `structural/irf.py::_lag_matrices`.

| Symbol | Shape | Meaning |
| :--- | :--- | :--- |
| `n, p, n_d, T` | scalars | variables, lags, deterministic terms, effective sample size |
| `k = n_d + np` | scalar | regressors per equation (excluding contemporaneous terms) |
| `Y`, `X` | `T×n`, `T×k` | stacked `y_t'`, `x_t'` |
| `y_i`, `Y_{<i}` | `T`, `T×(i−1)` | column `i` of `Y`; its first `i−1` columns |
| `A` | `n×n` | unit lower triangular; `A_ij = a_ij` for `j<i` |
| `a_i` | `i−1` | `(a_{i1},…,a_{i,i−1})'` (structural form: `y_i = −Y_{<i} a_i + …`) |
| `ã_i` | `n` | `(a_i', 1, 0,…,0)'` = row `i` of `A`, transposed |
| `Γ`, `γ_i` | `n×k`, `k` | structural coefficients; `γ_i'` = row `i` of `Γ` |
| `B`, `β_j` | `k×n`, `k` | reduced form `y_t = B'x_t + u_t`; `β_j` = column `j` |
| `M`, `m_i`, `M_{<i}` | `k×n`, `k`, `k×(i−1)` | prior mean of `B`; `m_i = M e_i`; first `i−1` columns |
| `δ_i`, `Δ` | `k`, `n×k` | `δ_i = γ_i − M_{<i} a_i`; rows `δ_i'` |
| `θ_i` | `q_i = i−1+k` | `(a_i', γ_i')'` (γ-coordinates) or `(a_i', δ_i')'` (δ-coordinates) |
| `λ_{i,t}`, `Λ_t` | | structural variance of equation `i`; `diag(λ_{1,t},…,λ_{n,t})` |
| `ω_{i,t}` | | outlier / Student-t scale multiplier (1 if absent) |
| `w_{i,t}`, `W_i` | | `1/(λ_{i,t}ω_{i,t})`; `diag(w_{i,1:T})` |
| `ψ_j` | | Minnesota scale (variance units). Phase 1 default: AR(p)+constant OLS residual variance (`psi="ar"`); Chan uses AR(4) |
| `d` | | IW degrees of freedom of the conjugate counterpart (Phase 1: `d = n+2`) |
| `κ` | | overall tightness = `Minnesota.lambda_`. Written `κ` here to avoid a clash with `λ_{i,t}` |
| `α`, `κ_c`, `κ_d`, `κ_a` | | lag decay (variance ∝ `l^{−α}`), `cross`, `var_const`, contemporaneous scale multiplier |
| `c_ij` | | `1` if `j = i`, else `κ_c²`. **Decision:** `cross` multiplies the *standard deviation*, like `lambda_` |
| `σ̄_i²` | | fixed structural scale (independent mode) `= ψ_i/(d−n+i−2)` (= `ψ_i/i` for `d = n+2`) |
| `P_0, μ_0` | | prior precision and mean of `θ_i` |
| `K_i, b_i` | | posterior precision and linear term: `θ_i | · ~ N(K_i⁻¹b_i, K_i⁻¹)` |
| `IG(c, C)` | | inverse gamma, shape `c`, **scale** `C`: density ∝ `λ^{−c−1} e^{−C/λ}`; draw `λ = C / Gamma(c, 1)`. Same as Chan's `IG(ν, S)`. `IW_1(ψ, ν) = IG(ν/2, ψ/2)` |

---

## 3. Model

```
y_{i,t} = −Σ_{j<i} a_ij y_{j,t} + x_t'γ_i + sqrt(λ_{i,t} ω_{i,t}) ε_{i,t},    ε_{i,t} ~ iid N(0,1)
⇔  A y_t = Γ x_t + (Λ_t Ω_t)^{1/2} ε_t
```

**Likelihood.** `p(y_t | x_t) = |A| Π_i N(ỹ_{i,t}; x_t'γ_i, λ_{i,t}ω_{i,t})` with `ỹ_t = A y_t` and
`|A| = 1`. Therefore

```
log p(Y | A, Γ, λ, ω) = Σ_i Σ_t [ −½ log(2π λ_{i,t} ω_{i,t}) − ½ e_{i,t}² /(λ_{i,t} ω_{i,t}) ],
e_{i,t} = y_{i,t} + Σ_{j<i} a_ij y_{j,t} − x_t'γ_i.
```

This is the same density as the reduced form `N(B'x_t, Σ_t)` with `Σ_t = A⁻¹Λ_tΩ_tA⁻¹'`. There is no
Jacobian term. Use it for the SBC log-likelihood test quantity and for diagnostics.

Contemporaneous `y_{<i,t}` are *data* on the right-hand side. This is legitimate because the joint
density factorises as `Π_i p(y_{i,t} | y_{<i,t}, x_t)`. (Missing values would make them latent. Not
supported in Phase 2; raise.)

---

## 4. Prior (decision D3)

### 4.1 Structure

```
a_i ~ N(m_a,i, V_a,i)                       (weakly informative; m_a,i = 0 by default)
γ_i | a_i ~ N(M ã_i, V_γ,i) = N(m_i + M_{<i} a_i, V_γ,i)
independent across i (given global hyperparameters)
```

**Joint prior in γ-coordinates** `θ_i = (a_i, γ_i)`, with `M := M_{<i}`:

```
μ_0 = ( m_a ;  m_i + M m_a )

V_0 = [ V_a        V_a M'           ]          log|V_0| = log|V_a| + log|V_γ|
      [ M V_a      V_γ + M V_a M'   ]

P_0 = V_0⁻¹ = [ V_a⁻¹ + M'V_γ⁻¹M    −M'V_γ⁻¹ ]      P_0 μ_0 = ( V_a⁻¹m_a − M'V_γ⁻¹m_i ;  V_γ⁻¹ m_i )
              [ −V_γ⁻¹M              V_γ⁻¹    ]
```

The closed form `P_0` was checked by block multiplication. The covariance term `Cov(a_ij, γ_{i,r}) =
(V_a M')_{jr}` is non-zero exactly at the rows `r` where `M_{rj} ≠ 0`. For example, with a random-walk
`M`, `γ_{i,(lag1, j)}` is centred on `a_ij`.

**δ-coordinates (recommended for implementation).** `δ_i = γ_i − M_{<i} a_i` has unit Jacobian, so

```
a_i ~ N(m_a, V_a)  ⊥  δ_i ~ N(m_i, V_γ,i)
y_i = Z_i (a_i ; δ_i) + e_i,      Z_i = [ X M_{<i} − Y_{<i} ,  X ]          (T × q_i)
γ_i = δ_i + M_{<i} a_i;   in matrix form  Γ = Δ + (A − I) M'
```

`(X M_{<i} − Y_{<i})_{t,j} = −(y_{j,t} − m_j'x_t)` is minus the prior-mean residual of variable `j`. For a
random-walk mean it is minus the first difference. So equation `i` regresses `y_i` on the
quasi-differenced earlier variables and on `x_t`, with **independent** priors. When `V_a` and `V_γ`
are diagonal the prior precision is diagonal. This has three practical benefits:

- the `O(q²)` constant-volatility trick applies (§7.4);
- Phase 3 global-local shrinkage composes directly, since it shrinks `δ_i` towards `m_i`, i.e.
  shrinkage towards `M ã_i`;
- the QR form of the analytic twin needs only a diagonal `V_0^{−1/2}`.

### 4.2 Why this centring: exact equivalence with the conjugate prior

**Proposition (NIW ⇔ D3-centred structural NIG).** Let `Σ ~ IW(Ψ, d)` with `Ψ = diag(ψ)`, and let
`vec(B) | Σ ~ N(vec M, Σ ⊗ Ω)` (the Phase 1 prior). Write `Σ⁻¹ = A'D⁻¹A` with `A` unit lower triangular
and `D = diag(σ_i²)`, and set `Γ = AB'`. Then, independently across `i`:

```
σ_i² ~ IG( (d − n + i)/2 , ψ_i/2 )
a_i | σ_i² ~ N( 0 , σ_i² diag(1/ψ_1, …, 1/ψ_{i−1}) )
γ_i | a_i, σ_i² ~ N( M ã_i , σ_i² Ω )
```

*Proof sketch.*
- The first two lines come from the IW partition theorem, used twice:
  - the marginal of the leading `i×i` block is `Σ_{1:i} ~ IW(Ψ_{1:i}, d−n+i)` (drop `p_2 = n−i`
    rows and columns);
  - partition that block as `(i−1, 1)`. Then `σ_i² = Σ_{ii·<i} ~ IW_1(ψ_i, d−n+i)`, and
    `a_i = −Σ_{<i}⁻¹Σ_{<i,i} | σ_i² ~ N(0, σ_i² Ψ_{<i}⁻¹)` (the sign flip is irrelevant for a
    zero-mean law). Here `a_i` solves `ã_i'Σ e_j = 0` for `j < i`, which is what `AΣA' = D` requires.
  - This is Chan (2022) Cor. 1 (the IW ⇒ NIG direction) with eqs. 6–7; Prop. 1 is the converse.
- **Independence across `i`** (missing from the original sketch): the partition theorem also makes
  `(a_i, σ_i²)` independent of `Σ_{1:i−1}`, and every `(a_j, σ_j²)` with `j < i` is a function of
  `Σ_{1:i−1}`.
- For the third line, `γ_i = Bã_i` gives `E[γ_i|A,D] = Mã_i`, and
  `Cov(Bã_i, Bã_j | Σ) = (ã_i'Σã_j) Ω = δ_ij σ_i² Ω` because `AΣA' = D`. The `γ_i` are jointly
  Gaussian given `Σ`, so they are independent across `i`, and each depends only on `(a_i, σ_i²)`.
- The map `(B, Σ) ↔ (A, D, Γ)` is a bijection and the likelihoods coincide (`|A| = 1`), so the
  posteriors coincide. ∎

The cross-check also verified the prior statement by Monte Carlo (`n = 4`, `d = 6`, 4·10⁵ IW
draws; §13.2 check B):
- `E[1/σ_i²]` and `E[log σ_i²]` match the IG values to 0.2%;
- `a_ij √(ψ_j/σ_i²)` has mean 0, variance 1 and kurtosis 3, and is uncorrelated with `log σ_i²`;
- all cross-equation correlations are below 0.004;
- `(γ_i − Mã_i)/(σ_i Ω^{1/2})` is iid `N(0, 1)`.

Consequences:

- With `d = n+2` (Phase 1 default): `σ_i² ~ IG(1 + i/2, ψ_i/2)` (Chan's baseline `ν_i = 1+i/2`,
  `S_i = s_i²/2`), and `E[σ_i²] = ψ_i/i`.
- **Dummy-observation view.** Phase 1 enters the Minnesota prior as dummy rows `Y_d = Ω^{−1/2}M`,
  `X_d = Ω^{−1/2}`. Transformed into structural equation `i` (target `Ω^{−1/2}m_i`, regressors
  `[−Ω^{−1/2}M_{<i}, Ω^{−1/2}]`), the residual is `Ω^{−1/2}(Mã_i − γ_i)`. So D3 centring is literally what
  the Minnesota dummies say in structural form. SOC/SUR dummies transform the same way (§7.7).
- **Uniqueness.** `E[B'|A] = A⁻¹E[Γ|A] = M'` for every `A` if and only if `E[Γ|A] = AM'`. D3 centring is
  the only linear centring that keeps the reduced-form prior mean at `M` conditionally on `A`.
- **Contrast with Chan (2022).** Chan's default sets `m_β,i` to the reduced-form mean *unconditionally*
  (zero, or one on the own first lag for levels; §2.2 and App. D). App. D shows `E[B] = M` only after
  averaging over `A` (using `E[α_i] = 0`). It then inflates the structural variances approximately:
  `Var((B_k)_ij) ≈ σ_i²[V_β̃,i + Σ_{l<i}(V_β̃,l + s_l⁻² m_β̃,l²)]`. The `s_l⁻² m²` term is exactly the
  `M V_a M'` block that D3 represents exactly, including the `a`–`γ` covariance that Chan drops by
  using a block-diagonal `V_i`.
  - To reach this approximation, Chan "replace[s] σ_l² by its prior mean s_l²" (Supplement §2).
    Under eq. 6 with `ν_0 = n+2`, the prior mean of the *structural* `σ_l²` is actually `s_l²/l`;
    `s_l²` is the prior mean of the reduced-form `Σ̃_ll`.
  - So each summand `V_β̃,l` is inflated by a factor `l` relative to the exact moment.
    This matters only when replicating Chan's tables, which must use his formula as printed. Conditional on `A`, Chan's prior implies `E[B'|A] = A⁻¹M'`. For example,
  with `n = 2` and random-walk `M`, the reduced-form equation 2 is centred on a coefficient of
  `−a_21` on `y_{1,t−1}`.
- **Precedent.** Sims & Zha (1998) put `f_i | a_i ~ N(P̄_i a_i, H̄_i)` on structural lag coefficients
  given the contemporaneous row. This is the same linear conditional centring, under their
  normalisation (unit-variance shocks, free `A(0)` diagonal).
  - Map: their `a_i = ã_i/√λ_i`, `f_i = γ_i/√λ_i`, `P̄_i = M`.
  - The random-walk case is explicit in Sims & Zha (2006, eq. 7): `A_+ = D + S A_0` with
    `S = [I_n; 0]` (their `x_t` lists lags first and the constant last) and a zero-mean prior on `D`.
    Column `i` of `D` is `(γ_i − Mã_i)/√λ_i = (δ_i − m_i)/√λ_i`, so their `D` is exactly our
    δ-coordinate deviation.
  - Their fixed `H̄_i` in normalised units means `Var(γ_i | a_i, λ_i) = λ_i H̄_i`. That is
    conjugate-type (Mode II) scaling.

### 4.3 Implied reduced-form prior

From `Γ = Δ + (A − I)M'`:

```
B' = A⁻¹Γ = M' + A⁻¹(Δ − M')       ⇒   B − M = (Δ − M')' A⁻¹'
E[B | A] = M  (exactly, every A)
Cov(β_j, β_l | A) = Σ_{i ≤ min(j,l)} (A⁻¹)_{ji} (A⁻¹)_{li} V_γ,i
  → if V_γ,i = σ_i² Ω:  = Σ_jl Ω     (Kronecker, i.e. the NIW conditional)
E[Σ_ii] = E[λ̄_i] + Σ_{j<i} E[a_ij²] E[Σ_jj]      (a_i independent of u_{<i} under the prior; λ̄_i = E-level of λ_{i,t})
```

The `E[Σ_ii]` recursion also needs `m_a = 0` and a diagonal `V_a`, so that the cross terms
`E[a_ij a_il] E[u_j u_l]` vanish. With a non-zero `m_a` (for example from Chan's Cor. 3), add
`m_a'E[u_{<i}u_{<i}']m_a`.

**Exactness of the Mode I moment match (cross-check, algebraic).**
- Under Mode I with `κ_c = κ_a = 1`, `Var(β_j | A) = (Σ_i (A⁻¹)_{ji}² σ̄_i²) Ω`, and
  `c_j := E[Σ_i (A⁻¹)²_{ji} σ̄_i²]` satisfies the same recursion as `E[Σ_jj]`:
  `c_j = σ̄_j² + Σ_{l<j} (σ̄_j²/ψ_l) c_l`.
- With `σ̄_j² = ψ_j/(d−n+j−2)`, induction gives `c_j = ψ_j/(d−n−1)` for **every** `d`. So
  `Var(B) = E_IW[Σ_jj] Ω` (the NIW value) and `E[Σ] = Ψ/(d−n−1)` hold exactly, not only in Monte
  Carlo.
- The off-diagonal `E[Σ_jl] = 0` follows from the sign symmetry of the zero-mean rows `a_j`.

**Ordering pitfall (CKY 2024 §2).** `a_ij ~ N(0, ψ_i/ψ_j)` with `E[λ_i] = ψ_i` gives `E[Σ_ii] = 2^{i−1}ψ_i`.
At `n = 20` the last variable's implied prior variance is 5·10⁵ times too large. Chan's moments,
`E[a_ij²] = E[σ_i²]/ψ_j` with `E[σ_i²] = ψ_i/i`, give `E[Σ_ii] = ψ_i/i + (i−1)ψ_i/i = ψ_i` for every `i`.
Both are confirmed by Monte Carlo (§13.2, check 4: `[1.00, 2.00, 4.00]` vs `[0.97, 0.99, 0.98]`; the
first entry is noisy because `IG(1.5)` has infinite variance).

**Cross-variable shrinkage is order-dependent in reduced form.** Under D3 with `κ_c < 1`, own-lag
variances of earlier equations leak into the cross-lag variances of later reduced-form equations
through `(A⁻¹)_{ji}²`. Monte Carlo with `n = 4`, `κ_c = 0.5` (variance factor 0.25) and the default
scales gives the implied reduced-form variance as a fraction of the NIW value:

- for the own first lag of equation `j` = 1..4: `1.00, 0.63, 0.50, 0.44` (**corrected**; the
  first draft read the `0.37` from the wrong cell);
- for the coefficient of equation 2 on variable 1: `0.63` instead of `0.25`.

**Closed form (cross-check; exact, no Monte Carlo needed).** Take Mode I or Mode II with
`κ_a = 1`, `m_a = 0`, diagonal `V_a` and `d = n+2`, and 1-based `j` (equation) and `m` (variable).
Then

```
Var_D3(B_{(l,m),j}) / [κ² ψ_j / (l^α ψ_m)] = κ_c² + (1 − κ_c²) · s_{jm},
s_{jm} = 1/j            if m = j      (own lag)
       = 1/(m(m+1))     if m < j      (lags of earlier variables)
       = 0              if m > j      (lags of later variables)
```

- `s_{jm} = E[(A⁻¹)²_{jm}] σ̄_m²/ψ_j` is the prior share of structural shock `m` in `Var(u_j)`, and
  `Σ_m s_{jm} = 1`.
- Proof: the shares satisfy `r_{j,m} = (1/j) Σ_{l=m}^{j−1} r_{l,m}` with `r_{m,m} = 1/m`. Induction
  gives `r_{j,m} = 1/(m(m+1))` for all `j > m`.
- The same holds in Mode II (Chan's structural κ₁/κ₂ route with `V_α = diag(1/s_j²)`), because
  `(A⁻¹)_{ji}` does not involve row `i` of `A`, so `E[(A⁻¹)²_{ji} λ_i] = E[(A⁻¹)²_{ji}] E[λ_i]`.
- So **every** later equation's coefficients on variable 1's lags get relative variance
  `κ_c² + (1−κ_c²)/2`, which is `0.625` at `κ_c = 0.5` instead of the intended `0.25`.
- Checked by Monte Carlo (`n = 5`): the max deviation is 0.003 in Mode I and 0.014 in Mode II
  (heavy-tailed IG shapes 1.5–3.5).
- This formula is what `implied_reduced_form_moments()` should return analytically for the
  diagonal case.

`Linear(form="structural")` and `form="reduced"` are therefore **different priors** (SPEC §4.1).
Phase 2 should ship `implied_reduced_form_moments(prior, draws=…)` (Monte Carlo over the prior) and
report it in `fit.summary()`, in line with R8.

### 4.4 Exact parametrisations and recommended defaults

There are three modes. Each element is diagonal unless stated. `Ω^{(i)}` is the Minnesota pattern for
equation `i`:

```
Ω^{(i)}_{(l, j)} = κ² c_ij (d − n − 1) / (l^α ψ_j)     lag l of variable j  (c_ii = 1, c_ij = κ_c² otherwise)
Ω^{(i)}_{det}    = κ_d                                deterministic terms (Phase 1 var_const, default 1e7)
```

With `κ_c = 1` this is Phase 1's `Ω = κ²/(l^α ψ_j/(d−n−1))`.

**Mode I: independent, "moment-matched" (RECOMMENDED D3 DEFAULT for every volatility block).**
This is the rai-bvar design. The moments are verified numerically; the mode itself is not taken from
a paper. Fixed scale `σ̄_i² = E_IW[σ_i²] = ψ_i/(d−n+i−2)`:

```
a_i ~ N(0, κ_a σ̄_i² diag(1/ψ_j)_{j<i})        d = n+2:  Var(a_ij) = κ_a ψ_i / (i ψ_j)
δ_i ~ N(m_i, σ̄_i² Ω^{(i)})                    d = n+2:  own lag l: κ²/(i l^α); cross: κ²κ_c² ψ_i/(i l^α ψ_j); det: κ_d ψ_i / i
Constant:  λ_i ~ IG((d−n+i)/2, ψ_i/2)         d = n+2:  IG(1 + i/2, ψ_i/2), mean ψ_i/i
SV:        see the SV note; moment-matching suggests centring the level prior so E[λ_{i,t}] ≈ ψ_i/i (UNVERIFIED as a recommendation)
defaults:  κ = 0.2, α = 2, κ_c = 1 (κ_c < 1 allowed), κ_d = 1e7, κ_a = 1, d = n + 2, ψ = "ar"
```

- With `κ_c = 1`, `κ_a = 1` it has the *same reduced-form mean (exact) and marginal second moments*
  as the Phase 1 conjugate prior: `Var(B_{(l,m),j}) = κ²ψ_j/(l^αψ_m)`, `E[Σ] = diag ψ`. It is not the
  same distribution, since there is no `σ²`-scale mixing.
- The same mean prior is used for `Constant`, `SV`, `SV(outliers)` and `SV(tails="t")`. This keeps the
  volatility-axis ablation clean (R2): `Linear×SV` vs `Linear×Constant` changes only the volatility
  block. Do **not** use conjugate scaling for the `Constant` default, because the mean prior would
  then differ between the two arms.
- `κ` is calibrated like the analytic engine's `lambda_`. EB values from Phase 1 transfer at `κ_c = 1`.

**Mode II: conjugate (λ-scaled NIG; Chan 2022 family; analytic-twin / test mode, Constant only).**

```
λ_i ~ IG((d−n+i)/2, ψ_i/2)
a_i | λ_i ~ N(0, λ_i κ_a diag(1/ψ_j))
δ_i | λ_i ~ N(m_i, λ_i Ω^{(i)})
```

With `κ_c = κ_a = 1` this equals the Phase 1 NIW prior exactly (§4.2). The mapping to Chan's
hyperparameters is:

- `κ_1 = κ²(d−n−1)`, `κ_2 = κ²κ_c²(d−n−1)`, `κ_3 = κ_d` (Chan: 100);
- `s_j² = ψ_j` (Chan: AR(4); pass `psi` explicitly to match);
- `α = 2`; `ν_i = (d−n+i)/2`; `S_i = ψ_i/2`; `V_α,i = κ_a diag(1/ψ_j)`.

Chan's own default is block-diagonal `V_i` with an unconditional `m_β`. rai-bvar's twin uses D3
centring. Offer `centre="unconditional"` only to replicate Chan's tables.

**Mode III: independent, "ψ-scaled" (CKY 2024 / CCM 2019 convention transplanted to structural coefficients).**

```
σ̄_i² = ψ_i
Var(a_ij) = κ_a ψ_i/ψ_j
own lag κ²/l^α, cross κ²κ_c²ψ_i/(l^αψ_j)
```

This matches CKY's `C_ij`, but CKY place it on *reduced-form* coefficients.

Warn about `Σ_ii` growth. If the volatility prior is centred so that `E[λ_i] = σ̄_i² = ψ_i`, then
`E[Σ_ii] = (1+κ_a)^{i−1}ψ_i` exactly, from the recursion `r_i = 1 + κ_a Σ_{j<i} r_j`. With the Mode I
volatility prior `IG((d−n+i)/2, ψ_i/2)` (mean `ψ_i/i`), the growth rate is still `(1+κ_a)` per
equation, but with a smaller constant.

Not the default.

Deterministic-term prior means are `m_{det,i}` = row block of `M` (0 by default). The own-lag means
come from transform codes (SPEC §5.3; `Minnesota.own_lag_means`).

### 4.5 Which software uses which

| Source | Where the prior lives | Coefficient scaling | Contemporaneous prior | Centring |
| :--- | :--- | :--- | :--- | :--- |
| Phase 1 / R `BVAR` / BoE `bvar` (GLP 2015) | reduced form, NIW | `Σ ⊗ Ω` (conjugate) | implied `IW(diag ψ, n+2)` ⇒ Mode II with `κ_c = κ_a = 1` | `M` |
| Chan (2022) | structural NIG per equation | `σ_i²·κ_1/(l²s_i²)` own, `σ_i²·κ_2/(l²s_j²)` cross, `σ_i²κ_3` intercept (`κ_3 = 100`) | `α_ij | σ_i² ~ N(0, σ_i²/s_j²)`, `σ_i² ~ IG(1+i/2, s_i²/2)` | unconditional `m_β`; App. D variance approximation |
| Chan, Koop & Yu (2024) | reduced-form `A` (coefficients) + dense `B_0` (errors) | independent, `C_ij = 1/l²` own, `s_i²/(l²s_j²)` other, horseshoe on `κ_1, κ_2`, local `ψ_ij` | rows of `B_0`: off-diagonal `N(0,1)`, diagonal `N(1,1)` (§6; the same in the JBES accepted manuscript); positive-diagonal sign normalisation | 0, or 1 on own lag 1 |
| CCCM (2022) | reduced form, independent across equations | any Gaussian | (not stated in corrigendum) | any |
| `bayesianVARs` | reduced-form `Φ`; Cholesky `U` (`Σ_t = U'⁻¹D_tU⁻¹`). **`bvar()` defaults to the factor structure; pass `specify_prior_sigma(type = "cholesky")`** | default `prior_phi = "HS"`; `"normal"` with `normal_sds = 10` (documented as prior *variances*; **UNVERIFIED** whether sd or variance); `"HMP"`; `prior_intercept = 10` (documented as sd) | `cholesky_U_prior` (HS listed first, i.e. the default under R's `match.arg`; **UNVERIFIED**); `"normal"` with `cholesky_normal_sds = 10` (also documented as variances) | `priormean = 0` (scalar = own first lag) |
| `bvartools` | structural `A_0 y_t = …` (unit lower, all lower entries free) and coefficients, joint SUR draw | normal, precision `v_i = 1` (variance 1), deterministic `v_i_det = 0.1` (variance 10) | same `N(0, 1)` (the default precision covers "structural terms") | 0 |
| MSBVAR (Sims–Zha 1998) | structural, unit-variance shocks | `H̄_i` (λ0…λ5, μ5, μ6) | `a_i ~ N(0, S̄_i)` | `P̄_i a_i` (conditional) |
| **rai-bvar D3** | structural | Mode I (default) / II (twin) / III | Mode-dependent, §4.4 | `M ã_i` (conditional) |

---

## 5. Conditional posteriors

### 5.1 `(a_i, γ_i) | λ_{i,·}, ω_{i,·}, y` (joint Gaussian, one step)

The two coordinate systems give the same posterior. Use the δ form.

```
δ-coordinates:  Z_i = [X M_{<i} − Y_{<i}, X],   P_0 = blockdiag(V_a⁻¹, V_γ⁻¹),   μ_0 = (m_a, m_i)
γ-coordinates:  X̃_i = [−Y_{<i}, X],            P_0, μ_0 as in §4.1

K_i = P_0 + Z_i' W_i Z_i
b_i = P_0 μ_0 + Z_i' W_i y_i
θ_i ~ N(K_i⁻¹ b_i, K_i⁻¹)
then γ_i = δ_i + M_{<i} a_i     (δ-coordinates only)
```

**Mode II (conjugate):**

- precision `K_i = (P̃_0 + Z_i'Z_i)/λ_i`, linear term `(P̃_0μ_0 + Z_i'y_i)/λ_i`, where the prior
  covariance is `λ_i P̃_0⁻¹`;
- mean `θ̂_i = (P̃_0 + Z'Z)⁻¹(P̃_0μ_0 + Z'y)`, which is **independent of `λ_i`**;
- covariance `λ_i (P̃_0 + Z'Z)⁻¹`.

### 5.2 Constant `λ_i`

| Mode | Full conditional | Note |
| :--- | :--- | :--- |
| I / III (independent) | `λ_i | θ_i, y ~ IG( c_i + T/2 , C_i + ½ Σ_t e_{i,t}²/ω_{i,t} )` | Default `c_i = (d−n+i)/2`, `C_i = ψ_i/2` |
| II (conjugate), Gibbs | `λ_i | θ_i, y ~ IG( c_i + (T + q_i)/2 , C_i + ½ e_i'e_i + ½ (θ_i − μ_0)' P̃_0 (θ_i − μ_0) )` | `q_i = i−1+k` counts **active** coordinates only (exclude padding, §7.1). Verified, §13.2 check 3 |
| II, collapsed (exact iid) | `λ_i | y ~ IG( c_i + T/2 , Ŝ_i )`, then `θ_i | λ_i, y ~ N(θ̂_i, λ_i (P̃_0 + Z'Z)⁻¹)` | `Ŝ_i` from §9 (residual form) |

### 5.3 SV, SV with outliers, SV with Student-t

The mean block receives only `w_{i,t} = 1/(λ_{i,t} ω_{i,t})`:

- `λ_{i,t} = exp(h_{i,t})`;
- `ω_{i,t}` = outlier scale `o_{i,t}²` (CCMM 2024) or the t-mixing variable (`ν`-dependent).

  **Parity caveat (cross-check).** `stochvol` defines its t errors as `t_ν(0, 1)` with unit
  *variance* (Hosszejni & Kastner 2021, eq. 2). So its `exp(h_t)` is the conditional variance, and
  equals our `λ_{i,t} E[ω]` only if our mixing variable is normalised to `E[ω] = 1`. With the
  textbook `ω ~ IG(ν/2, ν/2)`, `E[ω] = ν/(ν−2)`, so map `h^{stochvol} = h + log(ν/(ν−2))`. The
  exact normalisation belongs in the SV-with-t note.

It returns `θ_i`. The volatility block receives the **raw** residual
`e_{i,t} = y_{i,t} + Y_{<i,t}a_i − x_t'γ_i` (SPEC §4.2: `resid_t = √λ_t ε_t`, with `ω` handled inside the
volatility block). The mean step is identical for all four volatility blocks; only `W_i` changes.

### 5.4 `a_i` alone (for nonlinear `g_i`, or if someone splits the block)

With target `r_i = y_i − g_i(X)`, and since `r_i = −Y_{<i}a_i + e_i`:

```
K_a = V_a⁻¹ + Y_{<i}' W_i Y_{<i}
b_a = V_a⁻¹ m_a − Y_{<i}' W_i r_i
```

For `Linear` with D3, the prior of `a_i` given `γ_i` is
`N((V_a⁻¹ + M'V_γ⁻¹M)⁻¹(V_a⁻¹m_a + M'V_γ⁻¹(γ_i − m_i)), (V_a⁻¹ + M'V_γ⁻¹M)⁻¹)` with `M = M_{<i}`. Splitting
is valid but mixes worse. Draw jointly (SPEC §4.3).

For BART and other nonparametric `g_i` there is no `Mã_i` centring. The marginal prior of `a_i` is the
same `N(m_a, V_a)` as in `Linear`, so `Linear × SV` vs `BART × SV` share the `a_i` prior (R2).

---

## 6. Step order (one sweep)

```text
glob ← GlobalBlock.sample(...)                       # Phase 2: hyperparameters fixed (or EB, §9.4); nothing else global
vmap over i = 1..n  (keys split per (chain, iter, block, eq)):
    w_i      ← 1 / (λ_{i,1:T} · ω_{i,1:T})            # from current vol state
    θ_i      ← N(K_i⁻¹ b_i, K_i⁻¹)                     # §5.1, one joint draw of (a_i, δ_i)
    γ_i      ← δ_i + M_{<i} a_i
    e_i      ← y_i + Y_{<i} a_i − X γ_i               # raw residual, recomputed from the NEW θ_i
    vol_i    ← VolBlock.sample(e_i)                   # Constant: λ_i ~ IG (§5.2)
                                                      # SV: [ω (outlier/t) →] mixture indicators s → h (precision sampler) → (μ, φ, σ) (ASIS)
                                                      # DNP 2015: s drawn after θ_i and immediately before h; nothing in between
store (A, Γ, λ paths, vol params)                      # B is derived on demand (§8)
```

Equations are independent given `glob`. `θ_i` needs only the volatility of equation `i`, and the
volatility of equation `i` needs only `e_i`. So the `vmap` is an exact Gibbs block, not an
approximation.

---

## 7. Numerical implementation (JAX, float64)

### 7.1 Shared design, masks, static shapes

- One design matrix `F = [X M − Y, X] ∈ ℝ^{T×(n+k)}` (δ-coordinates) is shared by all equations. In
  γ-coordinates it is `[−Y, X]`.
- `mask_i ∈ {0,1}^{n+k}`: the first `n` entries are `1{j < i}`, the rest are 1. Equation `i` uses
  `F_i = F ⊙ mask_i`. The padded dimension is `q̄ = n + k`; slot `j` holds `a_ij` (the own slot `j = i`
  is always masked).
- Masked coordinates get prior mean 0 and prior precision 1. With their columns zeroed, the posterior
  there is `N(0,1)` and decoupled. **After the draw, multiply `a` by the mask** so that `A` stays unit
  lower triangular.
- Exclude masked coordinates from:
  - the log prior;
  - `q_i` in the Mode II `λ` update (§5.2);
  - log-determinants in any marginal likelihood.
- `n = 1`: the `a` block has static size `n` with every entry masked, so no special case is needed.
- Exclusion restrictions (a zero `a_ij` or zero `γ` entry) use the same mask mechanism. Never use a
  zero prior variance (infinite precision).

### 7.2 The Gaussian draw (SPEC §8, precision form, exact Jacobi preconditioning)

```
Fw   = F_i * sqrt(w_i)[:, None]
K    = diag(P0_i) + Fw' Fw                    # (δ-coords: P0 diagonal)
b    = P0_i * mu0_i + Fw' (sqrt(w_i) * y_i)
s    = 1 / sqrt(diag(K))                      # Jacobi scaling: exact change of variables
L    = cholesky(s[:,None] * K * s[None,:])    # lower
mu~  = cho_solve(L, s * b)
th~  = mu~ + solve_triangular(L', z, lower=False)     # z ~ N(0, I_q̄)
th   = s * th~                                         # = N(K⁻¹b, K⁻¹) exactly
```

`S(SKS)⁻¹S = K⁻¹`, so the scaling is exact. It is not jitter (SPEC §8 forbids jitter). It tames the
range from intercept prior precision `1/(κ_dσ̄²) ≈ 1e−7` to data precision `T/λ`, which can reach `1e8`
under SV.

### 7.3 Failure handling inside `jit`

`jnp.linalg.cholesky` returns NaN and does not raise. Carry `ok_i = all(isfinite(L)) & all(diag(L) > 0)`
and the first failing `(iter, eq)` in the scan carry. After the jitted segment, raise on the host with
diagnostics:

- `cond(K̃)` estimate from `L` (max/min `diag(L)²`). This is a *lower bound* on `cond₂(K̃)`, since
  `cond₂(L) ≥ max|L_ii|/min|L_jj|`;
- `min/max w_i`;
- the offending equation.

`jax.experimental.checkify` is an alternative. Never fall back silently.

### 7.4 Constant-volatility fast paths

| Case | Per-sweep cost per equation | Trick |
| :--- | :--- | :--- |
| Mode I/III × Constant | `O(q̄³)`, or `O(q̄²)` | `W_i = I/λ_i` ⇒ `Z_i'W_iZ_i = G_i/λ_i` with `G_i = mask(F'F)`: **one Gram for all equations**, precomputed once. With diagonal `P_0` (δ-coordinates, no SOC/SUR rows), precompute `eigh(P_0^{−1/2} G_i P_0^{−1/2}) = Q_i diag(g_i) Q_i'`. Then `K⁻¹ = P_0^{−1/2} Q (I + g/λ_i)⁻¹ Q' P_0^{−1/2}` and `θ = P_0^{−1/2} Q[(1 + g/λ)⁻¹ ⊙ (Q'P_0^{−1/2}b) + (1 + g/λ)^{−1/2} ⊙ z]`: `O(q̄²)` per draw, with no explicit inverse |
| Mode II × Constant | `O(q̄²)` | `K̃ = P̃_0 + G_i` does not depend on `λ_i`. Factor once; `θ = θ̂ + √λ_i · L'⁻¹z`. Or draw collapsed iid (§5.2) |
| SV / outliers / t | `O(Tq̄² + q̄³)` | Weighted Gram each sweep; it dominates when `T > q̄/3` |

### 7.5 Cost model for the Phase 2 target (UNVERIFIED until the pilot)

Setup: `n = 20`, `p = 4`, `k = 81`, `q̄ = 101`, `T = 300`.

- Weighted Gram: `2·T·q̄² ≈ 6.1e6` flop per equation.
- Cholesky plus three triangular solves: `≈ 3.7e5` flop.
- Total: `≈ 1.3e8` flop per sweep; × 4 chains × 12,500 sweeps ≈ `6.5e12` flop.

That is about 65 s at 100 GFLOP/s or about 220 s at 30 GFLOP/s of float64 on a laptop CPU, so the
2-minute target depends on how well XLA batches small matmuls. Possible savings:

- No padding: the average `q` is 90.5, saving about 20%.
- Batched matmul `Fw: (n,T,q̄)` instead of `einsum('tr,ts,et->ers')`, which may materialise
  `T×q̄×q̄`.
- Memory is small: `n·T·q̄·8 B ≈ 4.8 MB` per chain.

### 7.6 Other numerics

- **Residual-form sums of squares.** Never compute `y'y + μ_0'P_0μ_0 − θ̂'Kθ̂`, which suffers
  catastrophic cancellation. Use `Ŝ = S + ½[(y−Zθ̂)'(y−Zθ̂) + (θ̂−μ_0)'P_0(θ̂−μ_0)]`; the identity
  holds because `Kθ̂ = P_0μ_0 + Z'y`.
- **Log densities.**
  - log likelihood: §3;
  - log prior: Gaussian log densities evaluated through `P_0` (diagonal in δ-coordinates);
  - `IG` log densities in log space (`gammaln`);
  - with every `log|·|` taken from Cholesky diagonals.
- **IG draws.** Compute `λ = C/Gamma(c)` in float64 and assert `λ > 0` and finite. A tiny `C` (perfect
  fit) produces huge weights; §7.3 catches the failed Cholesky.
- **Data-dependent quantities** (`ψ`, `M` from transform codes, presample `ȳ_0` for dummies) are
  computed outside `jit`, per backtest window (R9), and passed in as arrays.

### 7.7 SOC/SUR dummy rows in structural form (optional in Phase 2)

A reduced-form dummy row `(y_d', x_d')` becomes, in equation `i`, the row with target `y_{d,i}` and
regressors `[x_d'M_{<i} − y_{d,<i}', x_d']` (δ-coordinates).

- **Mode II:** give the row weight `1/λ_i`. This is exactly the NIW-plus-dummies posterior. The
  marginal likelihood is `ML(dummies+data) − ML(dummies)`, per equation (Phase 1 convention).
- **Modes I/III:** use a **fixed** weight `1/σ̄_i²`, which makes the rows part of the Gaussian prior.
  Do **not** weight them by a volatility parameter such as `exp(μ_i)` unless the volatility block's
  conditional also includes these rows, or the sampler targets the wrong posterior.
- Dummy rows break the diagonal `P_0`, so the eigen trick of §7.4 then needs a dense
  `P_0^{−1/2}` (Cholesky). It is still exact.

---

## 8. Mapping back to the reduced form, forecasting, IRFs

All formulas use triangular solves (`solve_triangular(A, ·, lower=True, unit_diagonal=True)`). There
are no inverses.

```
B'   = A⁻¹ Γ                              (n×k)      equivalently  B = M + (Δ − M')' A⁻¹'
A⁻¹  = solve_triangular(A, I)            (only if needed for IRFs)
Σ_t  = A⁻¹ Λ_t Ω_t A⁻¹'                  chol(Σ_t) = A⁻¹ (Λ_tΩ_t)^{1/2}    (lower, positive diagonal; no Cholesky needed)
```

- **Forecast simulation** (per draw, `h = 1..H`). Build `x_{T+h}` from the simulated lags, draw
  `λ_{T+h}` (and `ω_{T+h}`) from `VolBlock.simulate_ahead`, then set
  `y_{T+h} = solve_triangular(A, Γx_{T+h} + (λ_{T+h}ω_{T+h})^{1/2} ⊙ ε)`. Equivalently,
  `y_{i} = −Σ_{j<i}a_ij y_j + γ_i'x + √(λ_iω_i) ε_i`, recursively over `i`. `B` is never formed.
- **Conditional moments** for the `ForecastResult` invariants (SPEC §7): mean `B'x_{T+1}`, covariance
  `A⁻¹Λ_{T+1}Ω_{T+1}A⁻¹'` given the drawn `λ_{T+1}`. Test invariant 1 with the same `ε`: the recursion
  output must equal `B'x + A⁻¹(Λ^{1/2}ε)` to `1e−12`.
- **IRFs** (`structural/irf.py` already documents this):
  - `Φ_h` from `B` (`ma_coefficients`);
  - unit structural shock `Θ_h = Φ_h A⁻¹`, time-invariant in shape;
  - one-sd shock at date `t`: `Φ_h A⁻¹ Λ_t^{1/2}`. Pass `chol = A⁻¹Λ_t^{1/2}` per draw.
  - FEVD is date-specific under SV.

  Recursive identification inherits the ordering (R8).
- **Storage.** Store `A` (strict lower triangle), `Γ` (or `Δ`), volatility states and parameters.
  Derive `B` lazily. It costs one `n×n` triangular solve against `n×k` per draw.

---

## 9. Analytic twin: Chan (2022) closed form (Phase 3, or pull into Phase 2)

Per equation `i`, conjugate Mode II in δ-coordinates (so `V_0 = blockdiag(κ_a diag(1/ψ_j)_{j<i}, Ω^{(i)})`,
which is diagonal):

```
Z̃ = [ Z_i ; V_0^{−1/2} ],   ỹ = [ y_i ; V_0^{−1/2} μ_0 ]          (stack, (T + q_i) rows)
QR:  Z̃ = Q R
θ̂_i  = R⁻¹ Q'ỹ                     (triangular solve)        ⇔ K_i = V_0⁻¹ + Z_i'Z_i = R'R
Ŝ_i  = S_i + ½ ‖ỹ − Z̃ θ̂_i‖²        (residual form)
ν̂_i  = ν_i + T/2
posterior: λ_i ~ IG(ν̂_i, Ŝ_i),  θ_i | λ_i ~ N(θ̂_i, λ_i (R'R)⁻¹)   ⇒ draw θ_i = θ̂_i + √λ_i R⁻¹ z
```

This is Chan (2022) §2.3, eq. 10: `K_θi = V_i⁻¹ + X_i'X_i`, `θ̂_i = K⁻¹(V⁻¹m + X'y)`,
`Ŝ_i = S_i + (y'y + m'V⁻¹m − θ̂'Kθ̂)/2`, posterior `NIG(θ̂, K⁻¹, ν_i + T/2, Ŝ_i)`.

**Log marginal likelihood** (Chan 2022 §2.4, eq. 11; our derivation and numerical check agree):

```
log p(Y) = Σ_{i=1}^{n} [ −(T/2) log(2π) − ½ log|V_0,i| − log|R_i|
                          + lnΓ(ν_i + T/2) − lnΓ(ν_i) + ν_i ln S_i − (ν_i + T/2) ln Ŝ_i ]
```

Here `log|R_i| = Σ log|R_jj| = ½ log|K_i|`, and `log|V_0,i| = Σ_{j<i} log(κ_a/ψ_j) + Σ_r log Ω^{(i)}_r`
(D3 centring does not change it, because the block-triangular map has unit Jacobian). With
`ν_i = (d−n+i)/2`, `S_i = ψ_i/2`, `κ_c = κ_a = 1`, this equals the Phase 1 NIW log ML **exactly**
(§13.2: difference 0.0 on `n = 3`, `p = 2`, `T = 60`, with a non-trivial `M`).

**Back-mapping:** `a_i = θ_a`, `γ_i = θ_δ + M_{<i}a_i`, then §8.

**Hyperparameters (Chan §3–4).**
- EB: maximise `log p(Y | κ, κ_c, κ_a)` over log hyperparameters. Each evaluation costs `n` QRs.
  Chan's application gives `κ̃_1 = 0.058`, `κ̃_2 = 0.0043` (reduced-form-route values, App. D).
- Full Bayes: Chan uses a collapsed sampler (MH on `κ` using eq. 11).
- Phase 2 shortcut for MCMC `hierarchy="eb"`: optimise the Mode II ML (which at `κ_c = 1` *is* the
  Phase 1 NIW ML) and plug the result into Mode I. This is approximate for Mode I; document it.

**Chan's propositions to carry into the docs.**
- Prop. 1 / Cor. 1: the diagonal-`S` IW ↔ NIG mapping above.
- Cor. 2: permuting variables permutes `s_i²`. The implied *reduced-form covariance prior* is
  order-invariant. The coefficient prior with `κ_2 ≠ κ_1` is **not**.
- Cor. 3 (general SPD scale `R`): with `R⁻¹ = L'S⁻¹L`, `σ_i² ~ IG((ν_0+i−n)/2, s_i²/2)` and
  `c_i | σ_i² ~ N(l_i, σ_i² L_{1:i−1}'S_{1:i−1}⁻¹L_{1:i−1})`. This is the route to a non-zero prior mean
  `m_a` for `a_i` (for example from a training sample).

---

## 10. Relation to the reduced-form prior and the corrected triangular algorithm (CCCM 2022)

In the reduced form, the prior is on the columns `β_j` of `B`, independently across `j`. In structural
form `γ_i = Bã_i = β_i + Σ_{j<i} a_ij β_j`, so `β_j` appears in equations `j..n`.

CCCM (2022) eq. (12): `ỹ_{i,t} = Σ_{j<i} a_ij x_t'β_j + x_t'β_i + λ_{i,t}^{1/2} ε_{i,t}` with `ỹ = Ay`. The
full conditional (their eqs. 14–15) is

```
Ω̄⁻¹ = Ω⁻¹ + Σ_{i=j}^{n} a_ij² Σ_t x_t x_t'/λ_{i,t},
μ̄ = Ω̄(Ω⁻¹μ + Σ_{i=j}^{n} a_ij Σ_t x_t z_{i,t}/λ_{i,t}),
```

where `z_{i,t} = ỹ_{i,t} − Σ_{l≠j, l≤i} a_il x_t'β_l` and `a_jj = 1`. Equivalent stacked form: eqs.
(16)–(20).

Differences from D3:

- The `n` coefficient blocks are a Gibbs *sequence*. This is a `GlobalBlock`, not `vmap`. It is about
  3× slower than the flawed 2019 version at `n = 20` and more autocorrelated, but still 13× faster than
  the system-wide sampler (CCCM §4).
- `a_i` must be drawn in a separate step given `B` (regress `u_i = y_i − Xβ_i` on `−u_{<i}`).
- CCCM footnote 2 (WP pp. 5–6) says independence across equations "covers most of the practical
  implementations of VARs". It "could be relaxed while maintaining the same triangular
  structure", via a triangular factorisation of the prior in which `π^{(j)}` appears only in
  equations `j..N`.

**D3 needs none of this.** Its prior is on `(a_i, γ_i)` per equation, so the posterior factorises
exactly.

**Corrected statement.** The first draft said that "with `cross = 1` and conjugate scaling the two
prior families coincide (NIW)". That holds only for D3. CCCM's algorithm, as published, assumes
coefficient priors `π^{(j)} ~ N(μ, Ω)` that are independent across equations and do not depend
on `(A, Λ)`. The NIW prior violates both assumptions: `Cov(β_j, β_l | Σ) = Σ_jl Ω`.

What is true:
- **D3 Mode II at `κ_c = κ_a = 1` is NIW.**
- Writing the NIW prior kernel per structural equation, `Σ_i (γ_i − Mã_i)'Ω⁻¹(γ_i − Mã_i)/λ_i` with
  `γ_i − Mã_i = Σ_{j≤i} a_ij(β_j − m_j)`, is exactly the footnote-2 triangular factorisation:
  `β_j` appears only in the terms `i ≥ j`.
- `form="reduced"` with an independent Minnesota prior is never NIW, even at `cross = 1`.

---

## 11. Relation to Chan, Koop & Yu (2024)

- **Model.** `y_t = a + Σ_l A_l y_{t−l} + B_0⁻¹ ε_t` (eq. 4), `ε_t ~ N(0, D_t)`,
  `D_t = diag(e^{h_{i,t}})`. `h_{i,t}` follows a **zero-mean** stationary AR(1) (eq. 3), with
  `h_{i,1} ~ N(0, ω_i²/(1−φ_i²))`. `B_0` is non-singular and otherwise unrestricted: no unit diagonal,
  only a positive-diagonal sign normalisation. Identification comes from SV: Prop. 1, citing Bertsche
  & Braun (2020), requires `φ_i ≠ 0` for all `i`, which is stronger than needed. Order invariance of
  the likelihood is Prop. 2. Posterior invariance needs a *permutation-equivariant* prior: the prior
  of equation `i` in one ordering must become the prior of equation `j` when variable `i` moves to
  position `j` (CKY §3.2). Signs are normalised by a positive `B_0` diagonal, and label switching is
  discouraged by a prior mean `I` on `B_0` (AAM §3.2).
- **Why triangular SV is order-dependent (§2).** With unit-variance priors on the triangular
  elements, `Var(u_{i,t}) = 2^{i−1}`. This is the reason for our Mode I scaling (§4.3, §4.4).
- **Their sampler (§4.3).**
  - Rows of `B_0`: `p(b_i|·) ∝ |det B_0|^T exp{−½(b_i − b̂_i)'K_{b_i}(b_i − b̂_i)}` (eq. 12). Sampled
    with Waggoner & Zha (2003), extended by Villani (2009): an orthonormal basis, then an
    absolute-normal draw for the first coordinate.
  - Coefficients by equation, using all `n` equations: `W^i = X ⊗ B_{0,i}`,
    `z^i_t = B_0(y_t − A_{i=0}'x_t)`. This costs `O(n⁴)`, like CCCM.
  - SV via KSC mixture and the Chan–Jeliazkov precision sampler.

  Their Minnesota-type horseshoe sits on the reduced-form coefficients and does not involve `B_0`.
- **Implication for rai-bvar.**
  - D3 is the triangular special case, with a unit diagonal and the scale in the SV level `μ_i`.
    Normalisation map: `B_0 = diag(e^{−μ_i/2}) A` with zero-mean `h`.
  - Ordering dependence stays and is surfaced through R8 and §6.6.
  - Phase 5 order-invariant SV replaces `A` by a dense `B_0`. **Design hint (ours, not in the paper,
    UNVERIFIED):** D3-style centring `γ_i | b_{0,i} ~ N(M b_{0,i}, V_γ,i)` keeps the structural
    equations conditionally independent *given* `B_0`, with target `b_{0,i}'y_t`. The row `b_{0,i}`, with
    `γ_i` integrated out analytically, then has the Waggoner–Zha form `|det B_0|^T × Gaussian kernel`,
    so the same row sampler applies.

---

## 12. Edge cases

| Case | Handling |
| :--- | :--- |
| `i = 1` / `n = 1` | No `a`; all `a` slots masked (§7.1). `n = 1` × SV is the `stochvol::svlm` parity case (§13.5) |
| `T < q_i` (more regressors than observations) | Fine: the proper prior keeps `K_i` positive definite |
| Collinear or constant columns (a dummy that is zero in a window) | `K` stays positive definite through the prior, but with `κ_d = 1e7` the coefficient is drawn with sd ≈ 3·10³σ̄. Warn when a deterministic column has zero variance in the window |
| Extreme SV weights (`w` up to `e^{±20}`) | Jacobi scaling (§7.2); failure flag (§7.3) |
| Padded coordinates | Mean 0, precision 1, zero columns; zeroed after the draw; excluded from `q_i`, the log prior and log determinants |
| `κ_c < 1` | Allowed in Modes I–III. **Conjugate NIW (Phase 1) forbids it.** Mode II with `κ_c ≠ 1` is Chan's asymmetric prior: still analytic, but no longer NIW |
| `Σ_ii` prior blow-up | Mode III or a large `κ_a` ⇒ geometric growth. `implied_reduced_form_moments()` reports it |
| IG shapes `1 + i/2 ≤ 2` for `i ≤ 2` (infinite prior variance of `λ_i`) | Fine for inference. In Geweke/SBC use `log λ_i` as the test function, or test configurations with shape ≥ 3 (§13.3) |
| Explosive prior draws in Geweke data simulation | Lags are simulated recursively, so explosive `B` draws can overflow. Use tight `κ` and short `T`. Never discard draws (that biases the test) |
| Data-dependent prior (`ψ`, transform-based `M`, `ȳ_0`) | Recompute per window (R9). **Fixed** in Geweke/SBC: if `ψ` is recomputed from simulated data, the joint-distribution identity fails |
| Zero restrictions | Mask; never zero variance |
| Missing values | Not supported in Phase 2. Raise (SPEC §5.7) |

---

## 13. How to test it

### 13.1 Unit / closed-form tests (every PR)

1. **Precision draw.** For random SPD `K` and `b`, the `μ` returned by the §7.2 routine must equal
   `np.linalg.solve(K, b)` (1e−10). The sample covariance of 1e5 draws must equal `inv(K)` within MC
   error. With Jacobi scaling on vs off, use the same `z`; the draws differ only by rounding (1e−10).
2. **Coordinate invariance.** γ- and δ-coordinates give the same posterior mean and covariance of
   `(a_i, γ_i)`: transform with `T = [[I,0],[M_{<i},I]]` (1e−10).
3. **Masking.** Padded `vmap` equals unpadded per-equation solves, for mean, covariance and draws with
   matched `z` on active coordinates (1e−12). Masked `a` entries are exactly 0.
4. **Single equation vs dense GLS.** `n = 1`, fixed weights: posterior equals the dense formula
   `(P_0 + X'WX)⁻¹(P_0μ_0 + X'Wy)` (1e−10).
5. **Mapping.**
   - `B' = A⁻¹Γ` vs `np.linalg.inv(A) @ Γ` (test-only inverse; 1e−12).
   - `A⁻¹Λ^{1/2}` equals `np.linalg.cholesky(A⁻¹ΛA⁻¹')` (1e−12).
   - Structural forecast recursion equals the reduced-form simulation with the same `ε` (1e−12).
   - `B − M = (Δ − M')'A⁻¹'`.
6. **Log-likelihood.** The structural `Σ_i Σ_t` formula (§3) equals `Σ_t log N(y_t; B'x_t, Σ_t)` (1e−10).
7. **Scale equivariance** (Modes I/II × Constant). Rescale `y_j → c_j y_j`. Then `ψ_j → c_j²ψ_j`, `M` is
   rescaled accordingly, and the deterministic prior is rescaled too. With the same seed, `a_ij`
   draws scale by `c_i/c_j`, `γ` and `B` scale consistently, and `λ_i → c_i²λ_i`, all to about 1e−10
   (Cholesky of `DKD` is `DL`). Under SV this holds only if the SV level prior is centred on `log ψ_i`.

### 13.2 Exact oracle: D3 conjugate ≡ Phase 1 NIW (implemented and run here)

Script: `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/verify_05/check_niw_structural.py`
(`n = 3`, `p = 2`, `T = 60`, `ψ = (0.8, 1.3, 0.6)`, `d = n+2`, `κ = 0.3`, `κ_d = 10`, `M` with own-lag
means 1 and 0.5 and a non-zero intercept mean). Results:

| Check | Result |
| :--- | :--- |
| 1. NIW (GLP formula) log ML vs Σ_i structural NIG log ML (γ-coordinates) | −248.3952005459069 both; diff **0.0** |
| 1b. Same in δ-coordinates (block-diagonal prior, regressors `XM_{<i} − Y_{<i}`) | diff **0.0** |
| 2. Posterior identities: `ν̂_i = (d+T−n+i)/2`; `2Ŝ_i = S̄_ii − S̄_{i,<i}S̄_{<i}⁻¹S̄_{<i,i}`; `E[a_i] = −S̄_{<i}⁻¹S̄_{<i,i}`; `Cov(a_i)/σ_i² = S̄_{<i}⁻¹`; `γ_i|a_i` covariance `Ω̄` with mean `B̄ã_i` | all ≤ **7e−15** |
| 3. Mode II non-collapsed Gibbs (40k draws) vs NIW posterior `E[B]`, `E[Σ]` | max \|z\| = 2.55 (B, 21 entries), 2.72 (Σ, 9 entries), batch-means MCSE |
| 4. Implied `E[Σ_ii]/ψ_i`: Mode I moments vs `Var(a_ij) = ψ_i/ψ_j`, `E[λ_i] = ψ_i` | `[0.97, 0.99, 0.98]` vs `[1.00, 2.00, 4.00]` |

Script: `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/verify_05/check_implied_prior.py`.
With Mode I, `κ_c = 1` (`n = 4`, 2·10⁵ prior draws): implied `Var(B)/NIW` lies in 0.991–1.005 for every
entry, and `|E[B] − M| ≤ 0.02` (MC). With `κ_c = 0.5`, the asymmetric pattern of §4.3 appears.

**Independent cross-check.** Script:
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/xc05_check/xcheck_05.py`.
It was written from the formulas in this note, not from the first author's script. It uses the
repository's Phase 1 engine read-only. Both first-author scripts were re-run and reproduce the
table above exactly.

| Check | Result |
| :--- | :--- |
| A1. **Phase 1 engine** (`rb.BVAR(...).fit(...).log_ml()`; `n = 4`, `p = 2`, `T = 90`, own means `(1, 0.5, 0, 0.9)`, **SOC μ = 1.5 and SUR δ = 2 dummies**, `var_const = 10`) vs `Σ_i [ML_i(dummies+data) − ML_i(dummies)]`. Per equation: δ-coordinates, QR, residual-form `Ŝ`, Chan eq. 11 | −536.9458857495 both; diff **0.0** |
| A2. 2·10⁵ iid draws from the per-equation structural posterior (collapsed `λ` then `θ`; dummies as data rows) vs the engine's `E[B]` and `E[Σ]` | max \|z\| = 2.53 (36 entries), 2.14 (16 entries) |
| B. Prior MC of the §4.2 proposition (`n = 4`, `d = 6`, 4·10⁵ IW draws) | moments, kurtosis and independence as stated in §4.2 |
| C. Closed-form leak ratio (§4.3) vs MC, `n = 5`, `κ_c = 0.5` | max abs diff 0.003 (Mode I), 0.014 (Mode II, heavy tails) |
| D. `P_0` closed form; eigen fast path (§7.4); Jacobi-scaled draw (§7.2) with a `1e8` scale disparity | `1e−15`; `2e−16`; `4e−16` relative |
| E. Chan unconditional centring, `n = 2`, `a_21 = 0.7`, random-walk `M` | `E[B'|A]` lag-1 block `[[1, 0], [−0.7, 1]]`, as claimed in §4.2 |

**CI version (Phase 2).**

- (a) The analytic part (checks 1, 2): exact equality at 1e−10 against `rai_bvar.engines.analytic`
  itself, not a re-implementation, including SOC/SUR dummies via the per-equation
  `ML(dummies+data) − ML(dummies)`.
- (b) The JAX Mode II Gibbs vs the analytic posterior: quantiles of `B`, `Σ` and `A` within 4 MCSE.
  Run at `n = 3` per PR and `n = 20`, `p = 4` nightly. This is the SPEC §10.3 analytic-twin test,
  available in Phase 2.
- (c) Permutation test: under Mode II with `κ_c = κ_a = 1`, permuting the variables (and `ψ`, `M`)
  leaves the reduced-form posterior unchanged (NIW is permutation-equivariant). Check exactly through
  ML and posterior moments. Under Mode I or `κ_c < 1` the difference is expected; it feeds the
  ordering-sensitivity check (§6.6).

### 13.3 Geweke joint-distribution tests

- **Configuration (small, every PR).**
  - `n = 3`, `p = 1`, `T = 25`, constant only.
  - `κ = 0.3`, `α = 2`, `κ_c = 0.5` (exercises asymmetry), `κ_d = 1` (proper, informative enough for
    power), `κ_a = 1`.
  - Fixed `ψ = (1, 0.5, 2)`, fixed `M` with own-lag means `(0.5, 0, 0.9)`, so `M_{<i} ≠ 0` and the
    centring is exercised.
  - Fixed presample.
  - Constant volatility: `λ_i ~ IG(3, ψ_i)` for the test (finite variance).
  - Run each mode separately: Mode I, Mode II, and Mode I × each volatility block.
- **Marginal-conditional simulator.** Draw `a_i ~ N(0, V_a)`, `δ_i ~ N(m_i, V_γ)`, `γ_i = δ_i + M_{<i}a_i`,
  the volatility states from their priors, then `Y` by **recursive** simulation from the presample.
- **Successive-conditional simulator.** One full sweep (§6) given `Y`, then a fresh `Y | θ`. Use
  2·10⁴ iterations per PR and 2·10⁵ nightly.
- **Test functions** (means and second moments, Holm-corrected z-tests with spectral or batch-means
  MCSE):
  - every `a_ij` and `a_ij²`;
  - `δ` entries (or `γ`) and their squares;
  - `log λ_i` (`log h`-path averages under SV);
  - the **cross-moment `a_ij · γ_{i,(lag1,j)}`**, which detects a missing or wrong centring;
  - `B` entries (nonlinear in `A`);
  - `Σ_ij` (use `log Σ_ii` if tails are heavy);
  - the log-likelihood `ℓ(θ; Y)`.
- **Pitfalls.**
  - Proper priors only.
  - `ψ` and `M` fixed (§12).
  - IG shapes: a z-test needs the test function to have finite variance under the prior. For
    `IG(c, ·)`:
    - `E[λ]` needs `c > 2`;
    - `E[λ²]` and `E[Σ_ii²]` need `c > 4`;
    - in Mode II, `E[a_ij²]` needs `c > 2`, because `Var(a_ij²) ∝ E[λ_i²]`.

    `IG(3, ·)` therefore supports first moments of `λ`, `a` and `Σ` but **not** second moments of
    `λ` or `Σ`. Test those through `log λ` and `log Σ_ii`, or use shape 5 (cross-check refinement).
  - The data step must use the same deterministic terms and presample as the sampler.

### 13.4 SBC (nightly)

- 500 replications (1000 pre-release), `T = 50`, `n = 3`, `p = 1`, plus one `n = 5` case. Use 999
  draws thinned to near-independence (thin by `⌈1/ESS-fraction⌉`).
- Rank statistics for: each `a_ij`; selected `γ`/`δ` entries (own lag 1, one cross lag, intercept);
  `log λ_i` (Constant) or `μ_i`, `φ_i`, `σ_i`, and `h` at three dates (SV); reduced-form `B` own lag 1
  and `Σ_21`; the **log-likelihood** `ℓ(θ; ỹ)` (Modrák et al. 2025); and `‖Γ − AM'‖²`, the deviation
  norm, which is sensitive to the centring.
- Pass: no rejection at 1% under the ECDF-envelope test (Säilynoja et al. 2022), Holm across
  quantities.

### 13.5 Parity oracles (with exact mappings)

| Oracle | Our configuration | Mapping | Compare |
| :--- | :--- | :--- | :--- |
| Phase 1 analytic NIW (in-repo) | Mode II, `κ_c = κ_a = 1`, `d = n+2`, same `ψ`, `M`, `κ`, `α`, `κ_d`, dummies | exact (§4.2) | log ML 1e−10; posterior moments 1e−10; Gibbs within 4 MCSE |
| R `BVAR`, BoE `bvar` | as above | transitively via the Phase 1 parity (SPEC §10.4) | as Phase 1 |
| Chan (2022) closed form (`AsymmetricConjugate`, §9) | Mode II, any `κ_c`; `centre="unconditional"` + block-diagonal `V` to replicate Chan's own tables | `κ_1 = κ²(d−n−1)`, `κ_2 = κ²κ_c²(d−n−1)`, `κ_3 = κ_d`, `s_j² = ψ_j` (Chan: AR(4)), `ν_i = 1+i/2`, `S_i = ψ_i/2`, `V_α = diag(1/ψ_j)` | exact iid vs MCMC within MCSE; log ML exact |
| `stochvol::svlm` (n = 1) | `Linear × SV`, `n = 1` (no `a`), Gaussian prior `γ ~ N(0, 10⁸ I)` (sd 10⁴; bypass the Minnesota scaling), SV priors mapped per the SV note | `stochvol` `beta = sv_multinormal(mean = 0, sd = 10000)`. Model `y_t = x_t β + exp(h_t/2) ε_t`, `h_{t+1} = μ + φ(h_t − μ) + σ η_t`, `h_0 ~ N(μ, σ²/(1−φ²))` (**verified**: Hosszejni & Kastner 2021, JSS eq. 1), so `λ_t = exp(h_t)`. Defaults: `μ ~ N(0, 100²)`, `(φ+1)/2 ~ Beta(5, 1.5)`, `σ² ~ Gamma(0.5, rate 0.5)` (= `χ²₁`), `latent0_variance = "stationary"`. With `heavytails = TRUE` the t errors have **unit variance** (§5.3 caveat) | posterior quantiles of `β`, `μ`, `φ`, `σ`, `h_t` within 4 MCSE. stochvol's `h_0` is a latent state one period before the first observation: match it or compare `h_{1:T}` only |
| `bvartools` structural (Constant) | Mode-free independent prior: `M = 0`, `V_a = I`, `V_γ = I` (deterministic `10·I`), `λ_i` with IG(shape, rate) ↔ their Gamma(shape, rate) on `1/λ_i` | `A_0` unit lower, `A_0 y_t = …` (same *model* sign as ours). **UNVERIFIED:** whether the stored `A0` draws are `A_0` entries or the SUR regression coefficients on `y_{<i,t}` (which are `−A_0` entries). Check on a fixture (`E[Σ]` from `A0`) | posterior moments within 4 MCSE. **UNVERIFIED:** exact default handling, whether `structural = TRUE` combines with `sv = TRUE` (`gen_var` exposes both flags, and the docs neither show nor forbid the combination), and the gamma shape/rate convention |
| `bayesianVARs` | **Not** a D3 oracle: reduced-form prior. Use only for `form="reduced"` (other note) and, with near-flat priors and large `T`, forecast distributions. Set `specify_prior_sigma(type = "cholesky")`: the `bvar()` default is the factor structure | `Σ_t = U'⁻¹D_tU⁻¹` ⇒ `Σ_t⁻¹ = U D_t⁻¹ U'`. The factorisation `Σ⁻¹ = A'Λ⁻¹A` with unit-lower `A` is unique, so **`A = U'` and `λ_{i,t} = (D_t)_ii` follow algebraically**, with no sign freedom (cross-check). **UNVERIFIED** only: whether the stored draws hold `U` or `U⁻¹`. Confirm on a fixture by comparing `E[Σ_t]` | forecast quantiles only |

### 13.6 Property tests

- Implied prior moments (§4.3): Mode I with `κ_c = 1` gives `Var(B)/NIW ∈ [0.98, 1.02]` and
  `E[Σ_ii]/ψ_i ∈ [0.95, 1.05]` (MC, fixed seed).
- Scale equivariance (§13.1-7).
- Permutation invariance in Mode II with `cross = 1` (§13.2 (c)).
- Determinism: the same seed gives bitwise-identical draws (SPEC §8).
- dtype float64 asserted on all state arrays.

---

## 14. UNVERIFIED items and open decisions

**UNVERIFIED**

1. ~~Chan (2022) equation numbers~~ **Resolved by the cross-check.** The QE PDF has the same
   eqs. (1)–(11), Prop. 1, Cor. 1–3, `κ_3 = 100` and Table 2. The arXiv App. C/D are §1/§2 of the
   QE Online Supplement.
2. Chan, Koop & Yu (2024). **Mostly resolved:** the `B_0` priors, eq. numbers, the `2^{i−1}` example,
   sign normalisation and sampler steps match the JBES *accepted manuscript*. Still unverified:
   the typeset version-of-record numbering, and the Villani (2009) absolute-normal construction,
   which is cited as in the paper and was not checked.
3. Sims & Zha (1998): the paper itself was not obtained. **Partly resolved:** the explicit
   random-walk form `A_+ = D + S A_0`, `S = [I; 0]` is confirmed from Sims & Zha (2006, NBER SI 2004
   draft, eq. 7), which attributes it to Sims & Zha (1998a). The conditional-linear form is
   confirmed via the MSBVAR docs.
4. IW partition theorem: book and theorem number. The *content* is verified numerically to ≤ 1e−14
   (§13.2) and by the cross-check's prior Monte Carlo (check B).
5. Software conventions, all from documentation only and to be confirmed on fixtures:
   - `bayesianVARs`: whether the stored draws are `U` or `U⁻¹`. `A = U'` itself follows
     algebraically from the documented `Σ_t`. Also whether `normal_sds` and
     `cholesky_normal_sds` are sds (by name) or variances (by the docs).
   - `bvartools`: the sign convention of the stored `A0` draws, the structural+SV combination, and
     the gamma-prior convention.
   - `stochvol`: the regression form is **now verified** from the JSS paper (eq. 1), as is the
     unit-variance t (eq. 2).
6. Mode I as the default is a rai-bvar design proposal, not a published prior. Its moment
   properties are verified by our Monte Carlo and, after the cross-check, proved algebraically
   (§4.3). Its *suitability as a default*, for example forecast performance against Chan's
   structural κ₁/κ₂ route, is untested. Likewise the recommendation to centre the SV level prior
   so that `E[λ_{i,t}] ≈ ψ_i/i`.
7. Performance numbers (§7.5) are flop estimates, not measurements.
8. Carriero, Clark & Marcellino (2019) state only a generic `A ~ N(μ_A, Ω_A)` (eq. 7 of the JoE
   accepted manuscript, https://hdl.handle.net/11585/713469). The numerical values are deferred to
   their online Appendix A, which was not obtained. Omitted.
9. The Phase 5 hint (D3-style centring with a dense `B_0`, collapsing `γ_i` into a Waggoner–Zha row
   draw) is our derivation sketch, not from CKY.
   - The cross-checker re-derived the marginalisation on paper. Integrating
     `γ_i ~ N(M b_{0,i}, V_γ)` out of `b_{0,i}'y_t = γ_i'x_t + ε_{i,t}` leaves
     `|det B_0|^T exp{−½ b'(Y−XM)'(W_i⁻¹ + X V_γ X')⁻¹(Y−XM) b}`, which is Gaussian in `b`, so the
     form is right.
   - Not numerically tested.
   - The `T×T` inverse needs Woodbury, giving a `k×k` solve.

**Decisions to record in module docstrings (SPEC §2.1)**

- D3 default = Mode I (moment-matched independent prior), the same for all volatility blocks; Mode II
  is available for `Constant` (test/twin).
- `cross` scales the standard deviation (variance × `cross²`), consistently with `lambda_`.
- δ-coordinates internally. Store `A` and `Γ`.
- SOC/SUR in MCMC: fixed weights `1/σ̄_i²` (Modes I/III), `1/λ_i` (Mode II). Can be deferred to Phase 3.
- `hierarchy="eb"` for MCMC uses the Mode II / NIW marginal likelihood (documented approximation for
  Mode I).

---

## Cross-check log

Cross-checked by a second agent, working clean-room: papers and package *documentation* only,
with no GPL source read. Every formula in §§3–10 was re-derived by hand.

New script:
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/xc05_check/xcheck_05.py`.
It runs the repository's Phase 1 engine read-only; no repository file was edited. Downloads are in
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/xc05_dl/`.

### Verified, no change needed

- **§4.2 proposition (NIW ⇔ D3 structural NIG).** Re-derived by hand. The structural log ML equals
  the actual `rai_bvar` Phase 1 engine's `log_ml()` **with SOC and SUR dummies** to `0.0` (check
  A1). iid structural-posterior draws reproduce the engine's `E[B]` and `E[Σ]` (|z| ≤ 2.53, check
  A2). The prior law was checked by Monte Carlo (check B).
- **Chan (2022) content.**
  - eq. 4 prior `θ_i|σ_i² ~ N(m_i, σ_i²V_i)`, `σ_i² ~ IG(ν_i, S_i)` (shape–scale, as in §2);
  - eqs. 6–7: `σ_i² ~ IG((ν_0+i−n)/2, s_i²/2)`, `A_ij|σ_i² ~ N(0, σ_i²/s_j²)`;
  - eq. 9 (Cor. 3);
  - eq. 10: `K = V⁻¹+X'X`, `Ŝ = S + (y'y + m'V⁻¹m − θ̂'Kθ̂)/2`, `ν + T/2`;
  - eq. 11 (log ML);
  - baseline `ν_i = 1+i/2`, `S_i = s_i²/2`, `V_α,i = diag(1/s_j²)`, `m_α = 0`, AR(4) `s_i²`,
    `κ_3 = 100`, own lag `κ_1/(l²s_i²)`, cross `κ_2/(l²s_j²)` (all relative to `σ_i²`);
  - App. D formula; Table 2 values `0.058` / `0.0043`.

  Checked in both the arXiv v1 and the published QE PDF.
- **Mode II ↔ Chan mapping** (`κ_1 = κ²(d−n−1)`, `κ_2 = κ²κ_c²(d−n−1)`, `κ_3 = κ_d`): consistent
  with `Minnesota.omega` in `src/rai_bvar/engines/analytic.py`
  (`Ω = λ²/(l^α ψ_j/(d−n−1))`, `d = n+2`).
- **Chan's unconditional centring gives `E[B'|A] = A⁻¹M'`** (check E).
- **Mode I scalings.** `Var(a_ij) = κ_aψ_i/(iψ_j)`, `V_γ = (ψ_i/i)Ω^{(i)}`, `IG(1+i/2, ψ_i/2)`, and
  `E[Σ] = diag ψ`. `2^{i−1}` under `E[λ_i] = ψ_i` (CKY §2 example; `2^{19} ≈ 5.2·10⁵` at `n = 20`).
- **§4.1 / §5 algebra.** `P_0`, `P_0μ_0`, δ-regressors `[XM_{<i} − Y_{<i}, X]`, `Γ = Δ + (A−I)M'`,
  `B' = M' + A⁻¹(Δ−M')`, `Cov(β_j, β_l|A)`, conditional prior of `a_i | γ_i`, Mode II
  `λ`-conditional `IG(c + (T+q_i)/2, C + ½e'e + ½(θ−μ_0)'P̃_0(θ−μ_0))`, collapsed draw.
- **§7 numerics.** Jacobi scaling, the eigen fast path, residual-form `Ŝ`, and the cost model
  arithmetic (`k = 81`, `q̄ = 101`, `6.1e6` flop Gram, `1.3e8` per sweep, `6.5e12` in total, average
  `q = 90.5`, 4.8 MB).
- **§8 back-mapping.** `chol(Σ_t) = A⁻¹(ΛΩ)^{1/2}` by uniqueness of the Cholesky factor; forecast
  recursion; IRF formulas. Also checked against `structural/irf.py` and the `lagged_design`
  layout.
- **CCCM corrigendum** (WP version): eqs. (12), (14)–(15) as transcribed (`z_{j+l,t}` definition,
  `a_ii = 1`); footnote 2 (relaxation via a triangular factorisation of the prior is indeed
  there, continued on p. 6); `3.25×` / `13×` in §4.
- **CKY (2024)**, against arXiv v1 and the JBES accepted manuscript: eqs. (3), (4), (12),
  Props. 1–2, `C_ij` (eq. 8), horseshoe on `κ_1, κ_2` and local `ψ_ij`, row sampler (WZ +
  Villani, absolute normal), `W_i = X ⊗ B_{0,i}`, KSC + Chan–Jeliazkov, positive-diagonal sign
  normalisation, §6 `B_0` priors.
- **Step order (§6) with `ω`.** Drawing `ω` (outlier/t) between `θ` and `s` is valid under Del
  Negro & Primiceri (2015): `ω` is drawn from its `s`-free conditional, and `s → h` stay adjacent.
- **Säilynoja, Bürkner & Vehtari (2022)**, Stat. Comput. 32(2):32, DOI 10.1007/s11222-022-10090-6.
- **Bibliographic data.** CKY JBES 42(2) 825–837; Chan QE 13(3) 1145–1169; CCCM JoE 227(2)
  506–512.

### Corrections made

1. **§4.3: wrong Monte Carlo number.** The own-first-lag ratio for equation 3 at `κ_c = 0.5` is
   `0.50`, not `0.37`. `0.37` is the coefficient of equation 3 on variable 2.
   - Source: re-run of `verify_05/check_implied_prior.py` (row "lag1 of var 3", column "eq 3" =
     0.500) and the new closed form (`1/3 + 0.25·2/3 = 0.5`).
2. **§4.3: closed form added.** The leak ratio is `κ_c² + (1−κ_c²)s_{jm}`, with `s_{jm} = 1/j`
   (own), `1/(m(m+1))` (`m < j`), `0` (`m > j`).
   - Exact for Mode I and Mode II, so it also holds for Chan's structural κ₁/κ₂ route.
   - Proof in §4.3; Monte Carlo check C.
   - The first draft had only Monte Carlo numbers.
3. **§4.3 / §0.5: Mode I moment match made exact.** The draft said "verified numerically". It is
   now proven for every `d`, and the conditions `m_a = 0` and diagonal `V_a` are now stated for the
   `E[Σ_ii]` recursion.
4. **§4.2 proof sketch: two fixes.**
   - The independence-across-`i` step was missing; added.
   - Corrected the citation to Chan **Cor. 1** for the IW ⇒ NIG direction (Prop. 1 is the
     converse; Chan, QE 2022, §2.2).
5. **§10: false statement fixed.** "With `cross = 1` and conjugate scaling, the two prior
   families coincide (NIW)" was wrong for CCCM.
   - CCCM's algorithm assumes priors independent across equations and independent of `(A, Λ)`;
     NIW has `Cov(β_j, β_l|Σ) = Σ_jl Ω`.
   - Only D3 Mode II equals NIW. NIW is exactly CCCM footnote 2's triangular prior factorisation.
   - Source: CCCM corrigendum WP §3, eq. (14) and footnote 2.
6. **§4.2 Precedent and §0.3: Sims–Zha random-walk form verified, mapping refined.**
   - Verified from Sims & Zha (2006), eq. 7: `A_+ = D + S A_0`, `S = [I_n; 0]`.
   - Their `D` equals our δ-deviation.
   - Their fixed `H̄_i` implies `λ_i`-scaled (Mode II-type) `Var(γ_i|a_i)`.
   - Source: https://conference.nber.org/confer/2004/si2004/me/sims.pdf §II.
7. **§4.2 Contrast with Chan: approximation flagged.** Chan's App. D replaces structural `σ_l²` by
   `s_l²`, but its prior mean under eq. 6 is `s_l²/l`.
   - Source: Chan QE Supplement §2 and eq. 6 (IG mean `(s_l²/2)/((ν_0+l−n)/2 − 1)`).
8. **§1 sources: citations updated.**
   - Chan QE DOI 10.3982/QE1381; arXiv App. C/D = QE Supplement §1/§2.
   - CKY verified against the AAM.
   - CCCM DOI 10.1016/j.jeconom.2021.11.010, WP numbering noted.
   - Sims & Zha (2006) row added.
   - stochvol JSS paper added (Hosszejni & Kastner 2021, JSS 100(12),
     DOI 10.18637/jss.v100.i12).
9. **§1, §4.5, §13.5: `bayesianVARs` details.**
   - `bvar()` defaults to `type = "factor"`, so parity runs must set `type = "cholesky"`.
   - `normal_sds` and `cholesky_normal_sds` are documented as *variances*.
   - `prior_intercept` is documented as an sd.
   - `A = U'` follows algebraically from the documented `Σ_t` (the unit-triangular `Σ⁻¹`
     factorisation is unique). The "sign UNVERIFIED" flag is replaced by "storage of `U` vs `U⁻¹`
     UNVERIFIED".
   - Source: bayesianVARs 0.1.8 reference manual, `bvar` Usage/Details and
     `specify_prior_sigma` Arguments.
10. **§13.5 `stochvol` row: model form verified.** `y_t = x_tβ + exp(h_t/2)ε_t`, `h_0` stationary
    (JSS eq. 1). Default priors written out (`specify_priors`, stochvol 3.2.9 manual).
    - **New caveat (§5.3):** stochvol's t errors have unit variance (JSS eq. 2). Map
      `h^{sv} = log λ + log(ν/(ν−2))` when our `ω ~ IG(ν/2, ν/2)`.
11. **§13.5 `bvartools` row: claim downgraded.** "Same sign as ours" now applies to the *model*
    `A_0 y_t = …` only. Whether stored `A0` draws are `A_0` entries or the negated SUR
    coefficients is UNVERIFIED (bvartools 0.3.0 manual, `bvar` Details, `gen_var` Details).
12. **§13.3 Geweke: IG shape conditions made precise.** `IG(3)` does not give finite variance for
    `λ²` or `Σ_ii²` test functions; that needs shape > 4.
13. **§7.3: wording.** The `diag(L)` ratio is a lower bound on `cond₂`.
14. **§4.4 Mode III: wording.** `(1+κ_a)^{i−1}` is exact only when `E[λ_i] = ψ_i`; with the
    Mode I `λ` prior it is the growth rate only.
15. **§11: wording.** "Exchangeable prior" became "permutation-equivariant prior" (CKY §3.2).
    Added the `B_0` prior-mean-`I` remark on label switching (AAM §3.2).

### Still UNVERIFIED after the cross-check

- Sims & Zha (1998) itself: the paper was not obtained (the Atlanta Fed WP 96-13 link returned
  404). Its equation numbers are unknown.
- Typeset version-of-record numbering:
  - CKY: only the accepted manuscript was checked;
  - CCCM: only the WP was checked;
  - Sims & Zha (2006): only the NBER draft was checked.
- Software conventions:
  - `bayesianVARs`: `U` vs `U⁻¹` storage, sd vs variance for `*_normal_sds`, `match.arg` default;
  - `bvartools`: sign of stored `A0`, the structural+SV combination, gamma shape/rate convention.
- Design proposals:
  - Mode I as the default: a design choice whose moments are proven, but whose forecast merit is
    untested;
  - the SV level centring `E[λ] ≈ ψ_i/i`.
- The §7.5 timing is a flop estimate.
- CCM (2019) online-appendix prior values on `A`.
- The IW partition theorem's textbook theorem number.
- The Villani (2009) absolute-normal details.
