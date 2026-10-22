# 04 — Corrected triangular algorithm (CCCM 2022) for `Linear(prior=Minnesota(form="reduced"))`

Scope: the reduced-form-prior variant of the Phase 2 MCMC engine (SPEC §4.1 "Reduced-form priors", §4.3 last bullet, §10.4 `bayesianVARs` parity). It covers the reduced-form VAR with a recursive (Cholesky-type) covariance, Normal priors on the columns of Π that are independent across equations, the exact full conditional of each column, the A block, the hand-off to the volatility blocks, and how to test all of it.

Clean-room status: everything below comes from the papers and from **package documentation** (argument names, defaults, and the model as documented). No GPL source code was read, by the author or by the cross-checker. The CCCM Matlab replication code (Bocconi link in the corrigendum) was **not** opened: its licence is unknown, so treat it as off-limits.

Cross-checked 2026-10-21 by a second agent; see the "Cross-check log" at the end. Independent check scripts are in `xc04_check/`, and the re-downloaded primary sources are in `xc04_dl/` (both under this folder).

---

## 0. Sources actually read

| Tag | Source | What was used |
|---|---|---|
| **CCCM-WP** | Carriero, Chan, Clark & Marcellino, "Corrigendum to: Large Bayesian VARs with SV and non-conjugate priors", FRB Cleveland WP 16-17C, Nov 2021, DOI 10.26509/frbc-wp-201617corrigendum. PDF (23 pp.; the online appendix is pp. 12–23 of the same file): https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2021/wp1617c.pdf . Published as J. Econometrics 227(2), 506–512 (2022), DOI 10.1016/j.jeconom.2021.11.010 (Crossref-checked) | §1 eqs (1)–(8) (the problem); §2 eqs (9)–(10) (the fix); §3 eqs (11)–(20) (implementation); §4 + Fig. 1 timings; online appendix §2 (Fig. 6) and §3 eqs (1)–(4) (homoskedastic variant) |
| **CCCM-AAM** | Authors' accepted manuscript of the journal corrigendum, open access (CC BY-NC-ND) at https://cris.unibo.it/handle/11585/900983 (file `Corrigendumfinal_ourposting.pdf`) | Used to confirm that the **accepted journal text has the same equation numbers (5)–(20) and the same footnotes 1–2 as the WP**. The typeset journal PDF itself was not seen. |
| **CCM-WP** | Carriero, Clark & Marcellino, FRB Cleveland WP 16-17, June 2016, **WP title "Large Vector Autoregressions with Stochastic Volatility and Flexible Priors"** (the copy with the 12/2021 note), https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2016/wp1617.pdf . Published as J. Econometrics 212(1), 137–154 (2019), DOI 10.1016/j.jeconom.2019.04.024 | §2 eqs (1)–(14) (model, priors, system-wide step); §3 eqs (27)–(38) (the flawed triangular step); fn 7 (within-equation prior correlation such as SOC); fn 9; §3.1 (ordering); §3.2.1 (Gibbs step order); appendix §8.1–8.2 (prior settings, KSC volatility) |
| **CCM-AAM** | Authors' accepted manuscript of the 2019 journal article, open access at https://cris.unibo.it/handle/11585/713469 | Journal numbering is **shifted**: WP (27)–(33) = journal (24)–(30); WP (34)–(38) = journal (31)–(35). Journal eqs (29)–(30) print the weight as σ*^{−1}_{j,j,t} (the typo of CCCM fn 1). The journal moved the prior settings (WP §8.1) to an online Appendix A, which was **not** accessible. |
| **Bognanni** | M. Bognanni, "Comment on 'Large Bayesian VARs with SV and non-conjugate priors'", J. Econometrics 227(2), 498–505 (2022), DOI 10.1016/j.jeconom.2021.10.008 (Crossref-checked). Abstract via https://ideas.repec.org/a/eee/econom/v227y2022i2p498-505.html | **Abstract only** (paywalled; no open copy found). Its detailed argument is UNVERIFIED; CCCM-WP §1 restates the core of it. |
| **Chan23** | J. C. C. Chan, "BVARs and Stochastic Volatility", arXiv:2310.14438 (Oct 2023) | §2.2 eqs (4)–(5) (Cholesky SV, AR(1) log-vols; note Chan puts the stationary prior on **h_{i,1}**, not on an h_{i,0}); §4.2.2 incl. fn 12 (CCM 2019 "can only be viewed as an approximation"; CCCM is exact at the same complexity) |
| **GK** | Gruber & Kastner, "Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!", arXiv:2206.04902v5 (IJF 2025, DOI 10.1016/j.ijforecast.2025.02.001) | §2 eqs (1)–(5): Σ_t = U'^{-1} D_t U^{-1}, ξ_t = U'ε_t, h_{i0} ~ N(μ_i, σ_i²/(1−ρ_i²)); §2.4: SV priors and HS prior on u; Appendix B: U sampled by equation-wise regressions (**printed without the minus sign; see §3.4**); Φ by the CCCM algorithm |
| **bVARs-doc** | `bayesianVARs` 0.1.8 reference manual (CRAN, https://cran.r-project.org/web/packages/bayesianVARs/bayesianVARs.pdf ; byte-identical to the copy in `dl_bvars_docs/bv.pdf`) | Model statement, argument names and defaults (documentation only) |
| **sv-doc** | `stochvol` 3.2.9 reference manual (already in `dl_stochvol_docs/refman.txt`) | `svsample` defaults; `priorsigma` meaning: "sigma^2 ~ priorsigma * chisq(df = 1)" |
| **Gibbs = Gauss–Seidel** | Fox & Parker (2017), "Accelerated Gibbs sampling of normal distributions using matrix splittings and polynomials", Bernoulli 23(4B), 3711–3743, DOI 10.3150/16-BEJ863, arXiv:1505.03512; Amit (1991), J. Multivariate Anal. 38(1), 82–99; Roberts & Sahu (1997), JRSS-B 59(2), 291–317 | The Fox & Parker abstract states the identity used in §9.2: Gibbs for a Gaussian with precision Q has the same iteration operator and convergence factor as Gauss–Seidel on Q. Amit and Roberts & Sahu were checked bibliographically only. |

---

## 1. Notation

| Symbol | Shape | Meaning | CCCM-WP | SPEC §4.1 | `bayesianVARs` |
|---|---|---|---|---|---|
| n | | number of variables | N | n | M |
| p | | lags | p | p | lags |
| k | | regressors per equation, k = np + n_d | k = Np+1 | k | K (+ intercept ι, see §3.4) |
| T | | effective sample size | T | T | T |
| y_t | n | data | y_t | y_t | y_t |
| x_t | k | (d_t', y_{t-1}', …, y_{t-p}')' | x_t | x_t | x_t |
| Y, X | T×n, T×k | rows y_t', x_t' | y, X | | Y, X |
| Π | k×n | reduced-form coefficients, y_t = Π'x_t + u_t | Π | B | Φ |
| π_j | k | column j of Π (equation j) | π^{(j)} | | |
| A | n×n | unit lower triangular, A u_t = Λ_t^{1/2} ε_t | A | A | U' (see §3.4) |
| a_ij | | element (i, j) of A, free for i > j, a_ii = 1, a_ij = 0 for i < j | a_{i,j} | a_ij | u_ji |
| **a**_j | j−1 | free elements of row j: (a_j1, …, a_{j,j−1})' | | a_i | |
| λ_it | | effective variance of structural shock i at t (exp(h_it) × any scale mixture) | λ_{i,t} | λ_{i,t} | exp(h_it) |
| Λ_t | n×n | diag(λ_1t, …, λ_nt) | Λ_t | Λ_t | D_t |
| Σ_t | n×n | A^{-1} Λ_t A^{-1}' | Σ_t | Σ_t | Σ_t |
| Ω_t = Σ_t^{-1} | n×n | A' Λ_t^{-1} A; ω_{jl,t} = Σ_{i ≥ max(j,l)} a_ij a_il / λ_it | | | U D_t^{-1} U' |
| u_t | n | reduced-form residual y_t − Π'x_t | (unnamed; v_t in CCM-WP) | u_t | ε_t |
| e_t | n | structural residual A u_t (= Λ_t^{1/2} ε_t) | Λ^{0.5} ε | | ξ_t |
| ỹ_t | n | A y_t | ỹ_t | | |
| m_j, V_j | k, k×k | prior mean and covariance of π_j; P_j = V_j^{-1} | μ_π(j), Ω_π(j) | | |
| m_{a,j}, V_{a,j} | j−1 | prior of **a**_j | μ_A, Ω_A | | |
| W | T×n | W_tj = ω_{jj,t} = Σ_{i ≥ j} a_ij² / λ_it ("combined weights") | | | |

Convention: vec stacks columns, so vec(Π) = (π_1', …, π_n')'.

---

## 2. Model

Reduced form (CCCM-WP eq. 1):

  y_t = Π' x_t + A^{-1} Λ_t^{1/2} ε_t,  ε_t ~ iid N(0, I_n),  t = 1..T.

Multiplying by A gives the recursive structural representation (CCCM-WP eqs 11–12):

  ỹ_{i,t} = Σ_{l ≤ i} a_il x_t'π_l + λ_it^{1/2} ε_it,  i = 1..n,  ỹ_{i,t} = Σ_{l ≤ i} a_il y_{l,t}.

So equation i contains π_l for **every l ≤ i**. In SPEC notation this is A y_t = Γ x_t + Λ_t^{1/2} ε_t with Γ = A Π' (so B = Π and B' = A^{-1}Γ). The likelihood (|A| = 1) is

  p(Y | Π, A, Λ_{1:T}) ∝ Π_t |Λ_t|^{-1/2} exp{ −½ Σ_t (A u_t)' Λ_t^{-1} (A u_t) } = Π_t |Λ_t|^{-1/2} exp{ −½ Σ_t Σ_i e_it² / λ_it }.

The difference from the structural form (SPEC D3): the prior sits on the columns π_j of the **reduced-form** Π and is independent across j. In the structural representation that prior couples the equations, so π_j cannot be drawn from equation j alone.

---

## 3. Priors (exact parametrisations)

### 3.1 VAR coefficients Π

  π_j ~ N(m_j, V_j), independent across j = 1..n, with V_j positive definite (diagonal for Minnesota).

CCCM-WP assumes independence across equations (text before eq. 13). Its footnote 2 says the assumption "covers most of the practical implementations", and adds that it *could be relaxed*: one would derive a triangular factorisation of the prior in which π_j enters only equations j..n, with the large inversions done once outside the MCMC loop. rai-bvar v1 does not need this; see §8 for the plain Gibbs alternative. Correlation *within* an equation is allowed. CCM-WP fn 7 (journal fn 11) names the sum-of-coefficients and unit-root priors of Sims (1993) and Sims & Zha (1998) as priors with within-equation but no cross-equation correlation, which supports the SOC/SUR handling below.

**Recommended rai-bvar default: independent ("reduced-form") Minnesota**, built from the same hyperparameters as `priors/minnesota.py`:

- m_j: 0 everywhere, except the own first lag, which takes the transform-aware value δ_j (SPEC §5.3).
- V_j, diagonal. For the coefficient on lag l of variable i in equation j:
  - own lag (i = j): λ² / l^α
  - cross lag (i ≠ j): λ² · cross² · ψ_j / (l^α ψ_i)
  - deterministic terms: `var_const`

  Here ψ is the variance-unit scale (`psi="ar"`). Whether `cross` multiplies the s.d. (squared here) or the variance must be fixed once in `minnesota.py` and documented (it is currently only validated, because `cross` must equal 1 in conjugate form).
- SOC/SUR dummies, if enabled: add the dummy rows to *each equation separately* as extra observations with fixed variance ψ_j. This gives a Gaussian prior that is independent across equations (correlated within an equation), which the CTA supports. Do **not** scale the dummies by the current Σ_t: that would make the prior depend on A and change the A full conditional. (Design recommendation; not from a paper.)

What others use:

| Source | Prior on Π |
|---|---|
| CCM-WP §8.1 eq (42) (read from the rendered PDF, p. 28) | Minnesota with cross-variable shrinkage, printed as **Var(Π_l^{(ij)}) = λ1² λ2 σ_i / (l^{λ3} σ_j)**, l = 1..p. Π_l^{(ij)} is row i (equation), column j (variable) of the lag-l matrix in y_t = Π_0 + Π(L)y_{t−1} + v_t. Settings: λ1 = 0.05 (overall), λ2 = 0.5 ("additional shrinkage on lags of other variables"; own lags implicitly 1), λ3 = 2 ("quadratic decay"), intercept variance 100, σ_i = residual s.d. of a univariate AR. Prior mean 0, except 1 or 0 on the own first lag by persistence. So λ1 enters **squared** (an s.d.-scale tightness: own first-lag prior s.d. 0.05), while λ2 and the ratio σ_i/σ_j enter the **variance unsquared** as printed. The unsquared s.d. ratio is dimensionally odd for a variance and may be a typo for σ_i²/σ_j². The journal version moved this to an online appendix, which I could not access, so there is no second source (**UNVERIFIED**). This prior is not a rai-bvar target. |
| `bayesianVARs` `specify_prior_phi(prior="normal", normal_sds=10, priormean=0)` | Normal prior. `priormean` exists for all priors (bVARs-doc): a scalar sets the own-first-lag mean (all other means 0), a length-M vector sets per-equation own-first-lag means, and a (lags·M)×M matrix gives every mean. The default 0 means a zero prior mean. `normal_sds` is *named* "sds" but documented as "prior variances", a vector of length lags·M² (intercept excluded). GK §2 also writes the generic prior as φ ~ N(0, V) with V holding variances. Which one the code uses is **UNVERIFIED; resolve with a fixture**. The default prior is `"HS"`, not normal. Intercept: `bvar(prior_intercept=10)`, documented as prior **standard deviations** (variance 100). |

### 3.2 Contemporaneous coefficients A

  **a**_j ~ N(m_{a,j}, V_{a,j}), independent across rows j = 2..n and independent of Π and of the volatilities.

| Source | Prior on the free elements of A |
|---|---|
| CCM-WP §8.1 | N(0, 10^6 I): "virtually flat". **Not allowed in rai-bvar tests** (Geweke/SBC need proper, non-degenerate priors). |
| `bayesianVARs` | `cholesky_U_prior` ∈ {HS (default), DL, R2D2, NG, SSVS, normal, HMP}; `cholesky_normal_sds = 10`, documented as "prior variances for the free off-diagonal elements in U" (**name vs doc conflict: UNVERIFIED**) |
| GK §2.4 | Horseshoe on u_ij |
| rai-bvar recommendation | Reuse the structural-form contemporaneous prior (SPEC D3) so the two forms differ only in the Π/Γ prior. If that note leaves it open: a_ij ~ N(0, κ_A ψ_i/ψ_j) with κ_A = 10 (scale-aware and weakly informative; a_ij has units σ_i/σ_j). This is a recommendation, not literature. |

### 3.3 Volatility

The CTA only needs the T×n array λ_it of **effective** variances. Each volatility block (specified in the SV notes of this folder) receives e_{j,1:T} = ((A u_t)_j)_t. For reference:

- `Constant`: λ_jt ≡ λ_j ~ IG(α0, β0), with density ∝ λ^{−α0−1} e^{−β0/λ}. Full conditional: IG(α0 + T/2, β0 + ½ Σ_t e_jt²). `bayesianVARs`: `cholesky_heteroscedastic=FALSE`, `cholesky_priorhomoscedastic` is an M×2 matrix of (shape, scale) per equation (bVARs-doc). Its default is `NA`, and the value the package substitutes is undocumented, so fixtures must pass it explicitly. The convention that "scale" is β0 in the density above is **UNVERIFIED**. The factor-model twin argument is documented as (shape, **rate**).
- `SV`: h_jt = μ_j + φ_j(h_{j,t−1} − μ_j) + σ_j η_jt, h_j0 ~ N(μ_j, σ_j²/(1−φ_j²)) (stationary), λ_jt = exp(h_jt). GK (§2, text after eq. 4) uses exactly this h_0 prior. Chan23 eq. 5 instead puts the stationary law on h_j1, which gives the same marginal law of h_{j,1:T}; only a stored h_0 differs.
  - `bayesianVARs` defaults (bVARs-doc; GK §2.4 states the same values): μ ~ N(0, 100²) (`cholesky_priormu = c(0, 100)` is mean and s.d.), (φ+1)/2 ~ Beta(20, 1.5), σ² ~ Gamma(shape 0.5, rate 0.5) (`cholesky_priorsigma2 = c(0.5, 0.5)`, "shape and the rate"), `cholesky_priorh0 = "stationary"`. bVARs-doc words the stationary h_0 prior as "Gaussian with **mean 0** and variance σ²/(1−φ²)", whereas GK says mean μ_i. Treat the doc wording as **UNVERIFIED** and compare only h_{1:T}.
  - `stochvol::svsample` defaults (sv-doc): `priormu = c(0, 100)`, `priorphi = c(5, 1.5)`, `priorsigma = 1` (σ² ~ 1·χ²₁ = Gamma(½, rate ½)), `priornu = 0` (ν fixed at ∞), `priorlatent0 = "stationary"`.
- CCM 2019 itself used random-walk log-vols (WP eq. 5; journal version too) with a full innovation covariance Φ ~ IW(d_Φ·Φ̲, d_Φ) (WP eq. 41) and h_0 ~ N(0, 100) (WP §8.2). §8.1 sets d_Φ = N + 2 and says "the scale matrix is set to an identity matrix". Whether that means Φ̲ = I or d_Φ·Φ̲ = I is ambiguous. This is **not** a rai-bvar target; parity is with `bayesianVARs` (independent AR(1)).
- `SV(outliers=True)`, `SV(tails="t")`: λ_jt = exp(h_jt)·o_jt² or exp(h_jt)·ω_jt. The CTA is unchanged; only W uses the effective λ.

### 3.4 Exact mapping to `bayesianVARs` (Cholesky type)

bVARs-doc/GK eq. 2: Σ_t = U'^{-1} D_t U^{-1} with U^{-1} (hence U) upper unitriangular, and ξ_t = U'ε_t ~ N(0, D_t). Ours: Σ_t = A^{-1} Λ_t A^{-1}' with A u_t ~ N(0, Λ_t). Hence

  **A = U'** (a_ij = U[j, i] for i > j, same sign), Λ_t = D_t, h_it = log d_ii,t, Π = Φ (intercept handled separately as ι; the row placement of the intercept in the returned `PHI` draws is **UNVERIFIED**).

**Sign caveat.** The mapping A = U' follows from the documented model, but GK Appendix B prints the U-step without a minus sign. It regresses ε_j e^{−h_j/2} on **+**ε_{<j} e^{−h_j/2}, giving μ_u = V̄_u X̃'ỹ. From ξ_t = U'ε_t we have ε_jt = −Σ_{i<j} u_ij ε_it + ξ_jt, so that printed form is the conditional of **−u**, not u. Either the appendix has a typo or the package stores the opposite sign. This is **UNVERIFIED**, and the GPL code must not be read to settle it.

**Fixture check (sign-sensitive).** A check that U'Σ_tU = D_t using `vcov()` would be circular, because `vcov()` is presumably built from the same U and D. Instead, simulate a large-T dataset whose residuals are strongly correlated, for example corr(u_1, u_2) = +0.8. Map every draw with A = U', so that Σ_t = A^{-1}D_tA^{-1}'. Then assert that the posterior mean of Σ_{12,t}/√(Σ_{11,t}Σ_{22,t}), averaged over t, is close to +0.8 and not −0.8. If it comes out at −0.8, use A = −(strict upper part of U)' + I.

The *storage layout* of the returned `U` draws is undocumented (**UNVERIFIED**): the docs only say it is "a matrix". It could be a vector of free elements or a matrix, in some order. GK §2 writes u = (u_12, u_13, …, u_{(M−1)M})', which reads as row-wise; App. B works column-wise with u^{(j)} = (u_1j, …, u_{j−1,j})'.

---

## 4. Why CCM (2019) was wrong

**What CCM 2019 did** (CCM-WP §3, eqs 27–33). Write v_t = A^{-1}Λ_t^{1/2}ε_t, with CCM's a*_{j,i} denoting elements of A^{-1}, and the generic equation

  y_{j,t} − (a*_{j,1} λ_1t^{1/2} ε_1t + … + a*_{j,j−1} λ_{j−1,t}^{1/2} ε_{j−1,t}) = x_t'π_j + λ_jt^{1/2} ε_jt.

Given π_1..π_{j−1}, the left side y*_{j,t} is "known", because the shocks of earlier equations are e_{<j,t} = (A u_t)_{<j}. CCM drew π_j ~ N(μ̄_j, Ω̄_j) with

  Ω̄_j^{-1} = V_j^{-1} + Σ_t x_t x_t' / λ_jt,  μ̄_j = Ω̄_j (V_j^{-1} m_j + Σ_t x_t y*_{j,t} / λ_jt)

for j = 1, 2, …, n in sequence, and claimed that the product is one exact joint draw of Π | y, A, Λ (CCM-WP eqs 30–31). The WP says (p. 13) this "will produce draws numerically identical to those that would be obtained using system-wide estimation".

Aside on the typo (CCCM-WP fn 1). The *journal* version printed the weight as σ*^{−1}_{j,j,t}, where σ*_{j,j,t} = λ_jt^{1/2} is the diagonal element of the lower Cholesky factor of Σ_t. It should be σ*^{−2}_{j,j,t} = 1/λ_jt (journal eqs 29–30, seen in CCM-AAM). The code was correct. The 2016 WP prints λ_{j,t}^{−1} in eqs (32)–(33), which is correct, but its eq. (28) has the typo λ_{j,t}ε_{j,t} for λ_{j,t}^{0.5}ε_{j,t}.

**What the step actually samples** (CCCM-WP eqs 5–6, crediting Bognanni). CCCM eq. (5) as printed conditions on y^{(1)}, …, y^{(j−1)}. That is a typo: eq. (6) and the text both condition on y^{(1)}, …, y^{(j)}, and the CCCM-AAM has the same typo. The correct statement is

  q(Π) = Π_{j=1}^n p(π_j | π_{1:j−1}, y^{(1:j)}, x, Σ_{1:T}),

where y^{(1:j)} is the data of variables 1..j only. The target factorisation is p(Π | y, Σ_{1:T}) = Π_j p(π_j | π_{1:j−1}, y^{(1:n)}, x, Σ_{1:T}). For n = 2 (CCCM-WP eqs 7–8):

  p(π_1, π_2 | y, Σ) ∝ p(π_1 | y^{(1)}) · **p(y^{(2)} | y^{(1)}, π_1, x, Σ)** · p(π_2 | y^{(1)}, y^{(2)}, π_1).

The bold factor depends on π_1 and was dropped. In the structural representation (§2), equation i contains a_ij x_t'π_j for every i > j. The contemporaneous values of variables j+1..n therefore carry information about π_j, and CCM's step ignored it. Only the last equation (j = n) was right.

**When the 2019 step happens to be exact (derived here, verified numerically in `xc04_check/xcheck2.py`).**
- With constant Σ, identical regressors and the **Kronecker (natural-conjugate) prior** vec(Π) ~ N(vec M0, Σ⊗Ω0), the 2019 step is exact. In this setting the TA uses CCM's conditional priors (CCM-WP eqs 34–38). The posterior is MN(M̄, Σ⊗Ω̄), and the columns 1..j of M̄ depend only on y^{(1:j)}, so p(π_j | π_{<j}, y^{(1:j)}) = p(π_j | π_{<j}, y). Numerically the error is 1e-15.
- With constant Σ and a flat prior the step is also exact: identical-regressor SUR is equation-by-equation OLS. Numerically the relative error is 1.5e-8 with prior variances of 1e8.
- With an **independent** prior, even one identical across equations and with constant Σ, the step is **not** exact (mean error 0.37 in one random case). This is the case CCM's Minnesota prior with cross-variable shrinkage falls into.

**Why it went unnoticed.** CCCM report that the empirical results "did not change appreciably" (§5). Their appendix Fig. 6 shows the SWA and CTA impulse responses coinciding, with the TA showing "some occasional deviations".

The error does not vanish with T in general. The measurement here: random problems with n = 3 and k = 3, independent priors, A entries drawn N(0,1), 5 replications per cell (`xc04_check/xcheck4.py`). For each, take the worst of |s.d. ratio − 1| and the mean shift in posterior s.d.s, then the median over replications:

| T | homoskedastic | log-normal time-varying λ |
|---|---|---|
| 10 | 0.36 | 0.84 |
| 40 | 0.49 | 0.51 |
| 160 | 0.14 | 0.41 |

The error falls clearly at large T only in the homoskedastic case. So CCCM's small empirical impact is application-specific.

CCM-WP fn 9 reported that, with the same random seed, TA and SWA results "exactly coincide besides minimal numerical errors". That cannot hold in general given CCCM eqs (5)–(8), and I cannot explain the reported coincidence.

**Bognanni (2022), per its abstract (IDEAS, paraphrased):** the paper shows *analytically* that the triangular and system-wide algorithms in general sample from different distributions, which contradicts a central claim of CCM. Substituting the triangular step into a VAR-SV Gibbs sampler is therefore an ad hoc change to the transition kernel, and the formal link between the simulations and the posterior is unknown. Whatever analytic construction Bognanni uses beyond this is **UNVERIFIED** (paywalled). Chan23 fn 12: CCM 2019 "can only be viewed as an approximation". GK §2.5 likewise calls the 2019 algorithm "an approximation".

**Size of the error on the §9.1 toy problem.** Computed in `cccm_check/` and re-derived independently in `xc04_check/xcheck.py`. In the CCM-2019 step's implied joint distribution:
- Posterior s.d.s are **2.7–5.6 % too wide** (ratios 1.027–1.056).
- Means are shifted by up to 0.040 in absolute terms. Measured in posterior s.d.s, the largest shift is **0.09**, on the first element of π_2.
- The precision blocks for j = 1 and j = 2 are off by up to 2.82 and 5.98.
- The rows for j = n agree exactly.

**Two ways to fix it.** CCCM-WP §2 says the missing terms could be added back, but that "might get messy" and costs O(N⁵). Using the CCM draw as an independence Metropolis–Hastings proposal would also work; that is my remark, not from the papers, and it is not recommended. The CCCM fix instead turns one "exact joint draw" into **n Gibbs steps over the full conditionals π_j | π_{−j}**, each built from equations j..n.

---

## 5. The corrected full conditional of π_j

### 5.1 CCCM form (CCCM-WP eqs 13–15)

With priors independent across equations, equations 1..j−1 do not contain π_j. Collecting the terms of the joint kernel (CCCM-WP display after eq. 12) that involve π_j, from equations i = j..n:

  π_j | y, A, Λ_{1:T}, π_{−j} ~ N(μ̄_j, K_j^{-1}),

  **K_j = V_j^{-1} + Σ_{i=j}^{n} a_ij² Σ_{t=1}^{T} λ_it^{-1} x_t x_t'**  (14)

  **μ̄_j = K_j^{-1} ( V_j^{-1} m_j + Σ_{i=j}^{n} a_ij Σ_{t=1}^{T} λ_it^{-1} x_t z_{i,t}^{(j)} )**  (15)

  **z_{i,t}^{(j)} = ỹ_{i,t} − Σ_{l ≤ i, l ≠ j} a_il x_t'π_l**,  a_jj = 1.

Here z^{(j)} is "equation i with every term except the one in π_j moved to the left". Equivalently z_{i,t}^{(j)} = e_it + a_ij x_t'π_j^{cur}, where e_t = A(y_t − Π^{cur}'x_t) uses the *current* Π (including the current π_j).

### 5.2 Stacked "regression" form (CCCM-WP eqs 16–20)

Let Π^{[j=0]} be Π with column j zeroed. Then

  Y^{(j)} = vec((Y − XΠ^{[j=0]}) A_{(j:n, 1:n)}') ./ vec(Λ^{1/2}_{(1:T, j:n)}),  T(n−j+1) × 1

  X^{(j)} = (A_{(j:n, j)} ⊗ X) ./ vec(Λ^{1/2}_{(1:T, j:n)}),  T(n−j+1) × k

  K_j = V_j^{-1} + X^{(j)'}X^{(j)},  μ̄_j = K_j^{-1}(V_j^{-1}m_j + X^{(j)'}Y^{(j)}).

This is a homoskedastic regression with T(n−j+1) rows. It explains the 3.25× slowdown versus the old TA at n = 20: CCCM-WP §4 says each step uses T(n−j+1) observations instead of T. The cross-product cost is Σ_j T(n−j+1)k² = O(T n² k²).

### 5.3 Combined-weight form (recommended for rai-bvar; derived here, verified numerically in §9)

Because A is lower triangular (a_ij = 0 for i < j), (14) collapses to a **single weighted regression with T rows**:

  K_j = V_j^{-1} + X' diag(W_{·j}) X,  W_tj = Σ_i a_ij² / λ_it = (Σ_t^{-1})_{jj} = ω_{jj,t}  →  W = (1/Λ) · (A∘A)  (T×n times n×n)

  μ̄_j = K_j^{-1}( V_j^{-1} m_j + X'( G_{·j} + W_{·j} ∘ X π_j^{cur} ) ),  G_tj = Σ_i a_ij e_it / λ_it = (Σ_t^{-1} u_t)_j  →  G = (E ⊘ Λ) A,  E = U A'.

Independent re-check: the 14–15 and combined-weight forms agree with the joint SUR oracle on 125 random problems (n ∈ {1, 2, 3, 5, 7}, k = 4, T = 9), with a worst relative error of 3.5e-13 (`xc04_check/xcheck2.py`). On the toy problem they agree with an oracle built differently, by exact polarisation of the log kernel with Σ_t^{-1} formed explicitly, to 1e-14 (`xc04_check/xcheck.py`).

Proof sketch: Σ_i a_ij z_it/λ_it = Σ_i a_ij (e_it + a_ij x_t'π_j)/λ_it = (A'Λ_t^{-1}A u_t)_j + ω_{jj,t} x_t'π_j. Equivalently, (14)–(15) are *exactly* the textbook SUR Gibbs conditional:

  K_j = V_j^{-1} + Σ_t ω_{jj,t} x_t x_t',  b_j = V_j^{-1}m_j + Σ_t x_t (ω_{jj,t} y_jt + Σ_{l≠j} ω_{jl,t} u_lt).

The "triangular" structure is how the needed entries of Σ_t^{-1} are obtained cheaply, from A and Λ_t, without forming or inverting Σ_t. This is why the conditional can be tested against the system-wide Gaussian (§9).

**Key implementation consequence:** K_j depends only on (A, Λ, X, prior), **not on π_{−j}**. All n precisions and their Cholesky factors can be computed in parallel (`vmap` over j) before the sequential scan. Only the cheap mean and draw step (O(Tk + Tn + k²)) is sequential.

### 5.4 Complexity per Π-block sweep (k ≈ np)

| Version | Cross-products | Factorisations | Sequential part |
|---|---|---|---|
| System-wide (SWA) | O(T n² k²) | O((nk)³) = O(n⁶p³) | — |
| CCCM stacked (5.2) | O(T n² k²) | O(n k³) | O(n k³) (all inside the loop) |
| Combined weights (5.3) | O(T n k²) + O(T n²) | O(n k³), parallel | O(n(Tk + Tn + k²)) |

All CTA variants are O(n⁴p³) in the factorisations, which is the CCCM headline. CCCM-WP §4/Fig. 1 timings (Matlab, 3.5 GHz Intel Core i7, p = 13; each figure is the average over 10 independent chains of the time to make 10 draws; the values are read from the data tips in Fig. 1): n = 20: SWA 33.1 s vs CTA 2.54 s (≈13×); n = 40: SWA 1181.7 s vs CTA 27.3 s (≈43×); CTA ≈ 3.25× slower than the old TA at n = 20. The combined-weight form should bring the cross-product cost back to the old TA's O(T n k²). That is a derived claim, **not benchmarked**.

Optional for k ≫ T (very large p): Chan23 §4.2.2 notes the Bhattacharya–Chakraborty–Mallick (2016) sampler costs O(r²m) with r rows and m regressors. With combined weights r = T (not T(n−j+1)), giving O(T²k) per equation. This needs the precision to be "prior + low-rank", which (5.3) is. Not needed for v1.

---

## 6. Algorithm

### 6.1 Π block (one CTA sweep)

```
inputs: X (T,k), Y (T,n), Pi (k,n) current, A (n,n), inv_lam (T,n) = 1/λ (computed as exp(-h) / mixture scales),
        prior precision P_j and linear term P_j m_j for each j, key
1. W  = inv_lam @ (A*A)                               # (T,n)  O(T n^2)
2. for all j in parallel (vmap):                       # independent of Pi
       K_j = P_j + (X * W[:,j,None]).T @ X             # (k,k)  O(T k^2)
       L_j = cholesky(K_j)                             # O(k^3); failure -> raise with cond(K_j), j
3. U = Y - X @ Pi;  E = U @ A.T                        # (T,n)
4. Z = normal(key, (n,k))                              # injected noise (see 6.4)
5. for j = 1..n sequentially (lax.scan over j; fori_loop with dynamic index):
       f_j   = X @ Pi[:,j]                             # current fitted values  O(Tk)
       G_j   = (E * inv_lam) @ A[:,j]                  # (Σ_t^{-1} u_t)_j for all t  O(Tn)
       b_j   = P_j m_j + X.T @ (G_j + W[:,j] * f_j)    # O(Tk)
       mu_j  = cho_solve(L_j, b_j)                     # O(k^2)
       new_j = mu_j + solve_triangular(L_j.T, Z[j], lower=False)
       d     = X @ (new_j - Pi[:,j])                   # O(Tk)
       U[:,j] -= d;   E -= d[:,None] * A[:,j][None,:]  # A[:,j] is zero above the diagonal: rows i<j untouched
       Pi[:,j] = new_j
6. return Pi, (U, E reusable by the A and vol blocks)
```

- **Cholesky failure under `jit`:** `jnp.linalg.cholesky` does not raise; it returns NaNs. To meet SPEC §8 ("a failed Cholesky raises with diagnostics"), check `isfinite(L_j)` with `jax.experimental.checkify`, or return a per-j failure flag in the sweep output. Raise on the host with j, cond(K_j) and the range of W_{·j}. A Python `try/except` inside the jitted sweep never fires.
- Scan order: j = 1..n (as in CCCM-WP §2). Any fixed or random order is valid, since this is a Gibbs scan over full conditionals. Keep 1..n for reproducibility.
- Memory: step 2 stores n·k² doubles per chain (n = 20, p = 4, k = 81: 1 MB; n = 100, p = 13, k = 1301: 1.35 GB, too much). Provide a switch that computes K_j and L_j *inside* the scan when n·k²·8·chains exceeds a budget. The result is identical, only slower.
- SPEC §4.3 places this block in the `GlobalBlock` because the equations are coupled. The A and volatility blocks remain `vmap`-over-equations.

### 6.2 A block (rows independent given Π, Λ)

With u_t from the Π block, row j's equation is u_jt = −Σ_{i<j} a_ji u_it + λ_jt^{1/2} ε_jt. This is a weighted regression of u_j on the regressors **−u_{<j}**. The minus sign matters: it was checked numerically in §9.4, and independently by exact polarisation in `xc04_check/xcheck.py`, where the plus-sign variant is off by 1.5–2.9. GK App. B prints this step without the minus sign (§3.4).

  K_{a,j} = V_{a,j}^{-1} + Σ_t λ_jt^{-1} u_{<j,t} u_{<j,t}',  μ_{a,j} = K_{a,j}^{-1}( V_{a,j}^{-1} m_{a,j} − Σ_t λ_jt^{-1} u_{<j,t} u_jt ).

This is the same as the structural-form contemporaneous step with x_t'γ_j replaced by the reduced-form fitted part (CCM-WP eq. 11; GK App. B; Cogley & Sargent 2005).

- `vmap` padding: embed row j in a length-(n−1) regression whose columns i ≥ j are masked to zero, with prior precision 1 on masked coordinates. Draw, then multiply by the mask.
- After the A draw, recompute E = U A' (O(Tn²)). The volatility blocks get e_j = E[:, j].

### 6.3 Full sweep and step order

```
for each iteration:
  (0) global hyperparameters (Minnesota λ etc., if hierarchical; Phase 3)
  (1) Π   | A, Λ_{1:T}, y           CTA sweep (6.1)
  (2) A   | Π, Λ_{1:T}, y           rows in parallel (6.2)
  (3) vmap over j:  scale-mixture latents (outlier o_jt / t ω_jt) | e_j, h_j
                    mixture indicators s_jt | e_j, h_j               (immediately before h)
                    h_{j,1:T} | s_j, e_j, (μ,φ,σ)_j                  (tridiagonal precision sampler)
                    (μ,φ,σ)_j | h_j  (ASIS)
```

This respects the Del Negro & Primiceri (2015) order required by SPEC §4.3. The indicators are drawn after (Π, A) and immediately before h, and no block sits between them. CCM-WP §3.2.1 has the same structure: step 2d draws s after Π and A, and step 1 draws the volatilities. The CTA itself is independent of the volatility specification: it consumes 1/λ_it only.

### 6.4 Pure-kernel design for testability

Implement `cta_sweep(Pi, A, inv_lam, X, Y, prior, Z) -> Pi_new` as a pure function of the injected standard normals Z (n×k). The JAX wrapper draws Z from the per-(chain, iteration, block) key. The deterministic tests in §9.2 need this.

### 6.5 Homoskedastic IW variant (optional; π_j step from CCCM-WP appendix §3)

For `Linear(Minnesota(form="reduced")) × Constant(prior="iw")`, the classic independent Normal–inverse-Wishart model. CCCM's appendix §3 gives only the π_j conditional given Σ. The Σ step is the standard independent Normal–IW one (CCM-WP eqs 21 and 23, §3.2.2):

1. Σ | Π, y ~ IW(S0 + Σ_t u_t u_t', ν0 + T), with prior density p(Σ) ∝ |Σ|^{−(ν0+n+1)/2} exp{−½ tr(S0 Σ^{-1})}.
2. Factor Σ = C C' (lower Cholesky). Set A = diag(c_11, …, c_nn) C^{-1} and λ_j = c_jj² (the LDL factorisation, CCCM-WP App. eq. 1 and fn 1). Alternatively use Ã = C^{-1} with unit variances (App. eqs 3–4). Both give the same conditional for π_j: re-verified to 2e-15 in `xc04_check/xcheck.py`.
3. CTA sweep with λ_jt ≡ λ_j.

This variant is order-invariant because the prior is on Σ, not on (A, Λ).

---

## 7. Numerical rules (SPEC §8 applied)

- **No inverses.** A^{-1} is never formed in the sampler: Σ_t^{-1} entries come from A'Λ^{-1}A via W and G. Forecasting uses `solve_triangular(A, ·, unit_diagonal=True)`.
- **Precision sampler.** Use `cholesky(K_j)`, then `cho_solve` for the mean and `solve_triangular(L', z)` for the noise (SPEC §8).
- **Log space.** Compute inv_lam = exp(−h) (times mixture precisions) directly; never 1/exp(h). Keep h in float64. W can range over many orders of magnitude, and that is fine in float64. No clipping and no jitter (SPEC §8). A failed Cholesky raises with j, cond(K_j), min/max of W_{·j}.
- **Residual bookkeeping.** U and E are updated incrementally inside the scan (rank-1 in time). Recompute them from scratch once per sweep (step 3) to stop round-off drift.
- **dtype.** Everything is float64. The test asserts `Pi.dtype == jnp.float64`.
- **Scaling.** The intercept prior variance (`var_const` = 1e7) makes K_j badly scaled when X has a constant column and data are in large units. That is acceptable in float64 because the prior adds only 1e-7 to that diagonal entry. Monitor cond(K_j) in a debug mode.

---

## 8. Edge cases

| Case | Behaviour |
|---|---|
| n = 1 | A = [1] has no free elements and W = 1/λ. The CTA is a single heteroskedastic regression and exact in one step. |
| j = n | K_n and μ̄_n equal CCM 2019's formulas (only equation n). Useful sanity check. |
| A = I (or all a_ij = 0 for i > j) | The equations decouple: W_tj = 1/λ_tj, G_j = e_j/λ_j. The Gibbs map has M = 0 (exact independent draws in one sweep). Test this. |
| Highly correlated residuals | Posterior correlation between π_j and π_l grows, and the Gibbs rate ρ(M) (the spectral radius of the block Gauss–Seidel iteration matrix) approaches 1, so mixing slows. CCCM-WP §1, §4 concede "more correlated draws". The structural-form default (SPEC D3) avoids this because its equations are conditionally independent. |
| k > T | K_j is still PD thanks to the proper prior. Cost is dominated by O(k³). |
| Cross-equation prior correlation | Not supported in v1. CCCM fn 2 notes it could be done via a triangular factorisation of the prior. A simpler route, standard Gibbs algebra, keeps everything else unchanged: replace V_j^{-1}m_j by (P m)_j − Σ_{l≠j} P_jl π_l and V_j^{-1} by P_jj, where P is the joint prior precision. Note that this term then depends on π_{l>j} too. |
| Missing values | Out of scope (SPEC §5.7). |
| Ordering | Given Σ_{1:T}, the conditional posterior of Π is ordering-invariant (CCM-WP §3.1; journal §3.1), provided the Π prior is permuted with the variables. The priors on (A, Λ) are not invariant, so the joint posterior depends on ordering (SPEC R8). CCM call this the "prior ordering problem". Record the ordering. When Λ_t varies over time, re-factoring a permuted PΣ_tP' gives a **time-varying** A_{P,t}: the constant-A model is not closed under permutation. |
| Random-walk SV (φ = 1) | Unchanged for the CTA; handled in the SV block. |

---

## 9. How to test it

### 9.1 Worked numerical check (unit test, deterministic)

Reference script: `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/cccm_check/cccm_check.py` (NumPy only). n = 3, k = 2, T = 5:

```
X = [[1, 0.5], [1, -1], [1, 2], [1, 0], [1, 1.5]]                      # T x k
Y = [[1.0, 0.2, -0.5], [-0.3, 1.1, 0.4], [2.0, -0.7, 1.3],
     [0.1, 0.5, -0.2], [1.2, 0.9, 0.8]]                                 # T x n
A = [[1, 0, 0], [0.5, 1, 0], [-0.3, 0.8, 1]]
lam = [[1, 0.5, 2], [2, 1, 0.5], [0.5, 1.5, 1], [1, 0.8, 1.2], [1.5, 0.6, 0.7]]   # T x n
m   = [[0, 0.1, -0.1], [0.5, 0, 0.2]]          # k x n, column j = m_j
V   = diag per column: [[4, 1, 2], [0.25, 0.5, 1]]   # k x n prior variances
Pi_cur = [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]]  # state at which the conditionals are evaluated
```

**Oracle (system-wide):** Q = blkdiag(V_j^{-1}) + Σ_t Σ_t^{-1} ⊗ x_t x_t', b = blkdiag(V_j^{-1}) vec(m) + Σ_t (Σ_t^{-1} y_t) ⊗ x_t. Conditional of π_j: precision Q_jj, mean Q_jj^{-1}(b_j − Σ_{l≠j} Q_jl π_l).

Expected values (|CCCM − oracle| ≤ 4e-15 in float64; use rtol = 1e-10):

| | precision K_j | mean μ̄_j at Pi_cur |
|---|---|---|
| j = 1 | [[7.5810714286, 6.1736904762], [6.1736904762, 17.069702381]] | [0.4584459034, 0.5225169483] |
| j = 2 | [[11.270952381, 5.3647619048], [5.3647619048, 15.8938095238]] | [0.697919255, 0.1574959305] |
| j = 3 | [[6.2619047619, 2.3928571429], [2.3928571429, 10.3392857143]] | [0.9656974281, −0.4844619264] |
| CCM 2019, j = 1 (wrong) | [[5.4166666667, 5], [5, 14.25]] | [0.4347305389, 0.6790419162] |
| CCM 2019, j = 2 (wrong) | [[7.5833333333, 3.8333333333], [3.8333333333, 9.9166666667]] | [0.5588890164, −0.0404108803] |
| CCM 2019, j = 3 | identical to the correct row | identical |

Joint posterior of vec(Π) = (π_1', π_2', π_3')':
- mean [0.4459332838, 0.6669945043, 0.5244110404, −0.1678802503, 0.5050315562, 0.1254023495]
- s.d. [0.4950517687, 0.3086496251, 0.4314114021, 0.3527802469, 0.5687897169, 0.4303835395]

Assertions:
1. CCCM (5.1), stacked (5.2) and combined-weight (5.3) conditionals all equal the oracle for every j. Also test 20 random draws of (A, λ, Pi_cur) from seeded generators, with n ∈ {1, 2, 3, 5}. Verified so far:
   - `cccm_check/random_check.py`: worst relative error 1.8e-15 for the stacked form over 80 random problems.
   - `xc04_check/xcheck2.py`: worst 3.5e-13 for forms (5.1) and (5.3) over 125 random problems with n up to 7.
   - Every number in the table above was reproduced independently by `xc04_check/xcheck.py`, which uses an oracle built by polarisation rather than the Kronecker formula. Agreement is to the printed digits.
2. **Regression guard:** the CCM-2019 conditional differs from the oracle for j < n (max |ΔK| = 2.82 and 5.98 here) and equals it for j = n.
3. The JAX kernel equals the NumPy reference to 1e-12, under `jit` and under `vmap` over chains.

### 9.2 Kernel-invariance test (deterministic; catches exactly a CCM-type bug)

For fixed (A, Λ), one CTA sweep is affine in (vec Π_old, vec Z): vec Π_new = M vec Π_old + c + N vec Z. Extract M, c, N **from the implementation** by calling `cta_sweep` on basis vectors with Z = 0, and on Z = basis vectors with Π = 0. Then solve

  mean* = (I − M)^{-1} c,  vec S* = (I − M⊗M)^{-1} vec(N N')

and assert mean* = Q^{-1}b and S* = Q^{-1} (rtol 1e-10).

Toy results:
- CTA: ρ(M) = 0.5037, |mean* − μ| ≤ 4e-16, |S* − Q^{-1}| ≤ 2e-16.
- CCM 2019: ρ(M) = 0 (it ignores Π_old), |mean* − μ| = 0.040, |S* − Q^{-1}| = 0.028, and the implied s.d.s are 1.027–1.056× too wide.
- Re-verified with a different solver: fixed-point iteration S ← MSM' + NN' instead of the Kronecker solve (`xc04_check/xcheck.py`).

Also assert the zero-noise iteration (Z ≡ 0) converges to the joint mean (block Gauss–Seidel): 200 sweeps give an error of 2e-16. A sharper assertion is that the extracted M equals the block Gauss–Seidel iteration matrix −(D+L)^{-1}U of the joint precision Q = D + L + U, split by equation blocks. This holds to 4e-16 here.

The identity is published: Fox & Parker (2017, Bernoulli 23(4B), 3711–3743, abstract) show that Gibbs sampling of a Gaussian with precision Q and Gauss–Seidel on Q have identical iteration operators and convergence factors. Amit (1991, JMVA 38(1), 82–99) and Roberts & Sahu (1997, JRSS-B 59(2), 291–317) give the earlier Gaussian Gibbs rate results; their bibliographic details were checked, their contents only through citing abstracts.

### 9.3 Monte Carlo check of the Π block alone

With fixed (A, Λ), run 60,000 CTA sweeps from the toy model. Measured: mean errors ≤ 0.005 and s.d. ratios within 0.997–1.005 of the oracle. These figures are seed-specific: an independent re-run with another seed (`xc04_check/xcheck3.py`) gave mean errors ≤ 0.0039 and s.d. ratios 0.994–1.006, so do not hard-code them. In CI, compare means and covariances with the oracle using MCSE-based z-tests (4 MCSE, Holm). This is cheap and isolates the block.

### 9.4 A-block check

The quadratic in **a**_j of −½Σ_t e_jt²/λ_jt − ½(**a**_j − m_a)'V_a^{-1}(**a**_j − m_a) has zero gradient at the §6.2 mean, and its negative Hessian equals K_{a,j} (numerically 1e-8 with finite differences; script `cccm_check/a_check.py`). Unit-test the sign convention with this. Also test n = 1 and 2.

### 9.5 Internal end-to-end oracle: system-wide algorithm (SWA)

Implement, in NumPy test code only, the O(n⁶) SWA step (joint Gaussian draw of vec Π from Q, b) and swap it into the full Gibbs sampler. For n ∈ {2, 3, 4}, p ∈ {1, 2} and T = 60, run CTA-Gibbs and SWA-Gibbs (Constant and SV) and compare posterior means, s.d.s and 5/50/95 % quantiles of Π, A, h-path summaries and SV parameters within 4 MCSE. The two samplers share the target, and SWA is correct by construction (CCM-WP eqs 13–14).

### 9.6 Geweke joint-distribution test (SPEC §10.3)

- **Configurations:**
  - (i) "Regression" config: x_t is a fixed exogenous design (constant plus two N(0,1) columns), n = 3, T = 25, so data can be regenerated without feedback.
  - (ii) VAR config: p = 1, n = 3, T = 25, fixed y_0 and tight lag prior (s.d. 0.3) so paths rarely explode. Regenerate the whole path y_{1:T} | θ in the successive-conditional simulator.
  - Vol ∈ {Constant (IG(3, 2)), SV (μ ~ N(0, 1), (φ+1)/2 ~ Beta(20, 1.5), σ² ~ Gamma(½, rate ½/0.1)), SV(t), SV(outliers)}. All priors proper; A prior N(0, 1).
- **Test functions:** each π_j entry and its square; **cross-equation products** π_{1,l}π_{2,l} and π_{1,l}π_{3,l} (most sensitive to a CCM-type error, which distorts the cross-equation dependence); a_ij and a_ij²; μ_j, φ_j, σ_j²; mean_t h_jt; log-likelihood log p(y | θ); Σ_t e_jt²/λ_jt.
- z-tests with MCSE (batch means), Holm correction, α = 0.01.
- **Power check (strongly recommended):** run the same harness with the CCM-2019 step plugged in, and confirm that it *fails* for some configuration. This proves the harness can detect the bug class R6 is about. Choose the configuration with §4 in mind:
  - The 2019 step is exact with a flat or Kronecker prior and constant Σ, and its error shrinks with T when volatility is constant. So use **informative independent priors that differ across equations, small T, and time-varying λ** (an SV configuration).
  - Use a non-trivial A (|a_ij| of order 1).
  - Do **not** expect larger T to help in the Constant configuration.

### 9.7 SBC (nightly)

200–1000 replications on configuration (i) and (ii), each thinned to about 100 near-independent draws. Rank statistics for π entries, a_ij, (μ, φ, σ) per equation, h_{j,T}, and the data-dependent quantities log p(y | θ) and Σ_t e_jt²/λ_jt (Modrák et al. 2025). Use an ECDF-envelope uniformity test at 1 % after multiplicity correction.

### 9.8 Parity with `bayesianVARs` (fixtures, §10.4)

R call (to be written in `tools/make_r_fixtures.R`; argument names from bVARs-doc 0.1.8):

```r
bvar(data, lags = p, draws = D, burnin = B, thin = 1,
     sv_keep = "all",                                     # default "last" stores only h_T
     prior_intercept = s_c,                               # documented as prior SDs of the intercept
     prior_phi   = specify_prior_phi(data = data, lags = p, prior = "normal", normal_sds = v_pi,
                                     priormean = m_own),  # scalar/length-M: own first-lag mean; or (lags*M) x M matrix
     prior_sigma = specify_prior_sigma(data = data, type = "cholesky",
                     cholesky_U_prior = "normal", cholesky_normal_sds = v_a,
                     cholesky_heteroscedastic = TRUE,
                     cholesky_priormu = c(0, 100), cholesky_priorphi = c(20, 1.5),
                     cholesky_priorsigma2 = c(0.5, 0.5), cholesky_priorh0 = "stationary"))
```

| rai-bvar | `bayesianVARs` |
|---|---|
| m_j | `specify_prior_phi(priormean=...)` (bVARs-doc): a scalar or length-M vector sets the own-first-lag means and zeros elsewhere; a (lags·M)×M matrix sets every mean. Default 0. It is documented generically, not only for `prior="normal"`; that it is honoured under `"normal"` is to be confirmed by a probe fixture. |
| V_j = diag(v) on lag coefficients | `normal_sds = v` if the argument is a variance (as documented), or `sqrt(v)` if it is an s.d. (**UNVERIFIED; settle with a probe fixture**: tiny T, huge λ so the prior dominates, then read off the posterior variance) |
| intercept variance s_c² | `prior_intercept = s_c` (documented as s.d.) |
| a_ij ~ N(0, v_a) | `cholesky_U_prior="normal"`, `cholesky_normal_sds` (same variance-vs-s.d. ambiguity) |
| A | t(U) per draw (§3.4); run the sign-sensitive correlation check of §3.4 (not the circular U'Σ_tU = D_t check) |
| h_it, (μ, φ, σ) | `logvar` (T × M per draw **only with `sv_keep = "all"`**; with the default `"last"` only h_T is stored), `sv_para` (layout **UNVERIFIED**) |
| SV priors | as in the call (μ ~ N(0, 100²); (φ+1)/2 ~ Beta(20, 1.5); σ² ~ Gamma(0.5, rate 0.5) = stochvol `priorsigma = 1`; stationary h_0, whose documented mean is 0 vs GK's μ_i, see §3.3) |
| Constant variant | `cholesky_heteroscedastic = FALSE`, `cholesky_priorhomoscedastic = cbind(alpha0, beta0)` (IG shape, scale), always passed explicitly (default `NA`) |

Compare posterior quantiles of Π, A, h_{·,t} (selected t), μ, φ, σ, plus forecast quantiles, within 4 MCSE on FRED-MD subsets n ∈ {3, 7}, p = 2–4, with the same variable ordering. `bayesianVARs` draws the volatilities with `stochvol`'s `update_fast_sv` (ASIS etc. per its defaults) and its own SV offset (`expert_sv_offset`, "Do not use!"). Posterior agreement is therefore expected only up to MCMC error, not draw-by-draw.

### 9.9 Ordering check (optional unit test)

Take Σ_t = A^{-1}Λ_tA^{-1}' and permute the variables with a permutation matrix P (Y → YP', columns of m and V permuted likewise). Re-factor PΣ_tP' by LDL **for each t** into (A_{P,t}, Λ_{P,t}). When Λ_t varies, A_{P,t} varies with t, so a constant-A CTA kernel cannot represent the permuted problem. Run §9.2 with a test-only kernel whose combined weights take per-t A_{P,t}: W_tj = Σ_i (A_{P,t})_ij²/λ_{P,it} and G_t = A_{P,t}'Λ_{P,t}^{-1}A_{P,t}u_t. The stationary law of the permuted Π columns must match the permuted oracle, because the conditional of Π given Σ_{1:T} is ordering-invariant (CCM-WP §3.1). Verified with n = 4 and a non-trivial permutation: errors 2e-15 (mean) and 6e-16 (covariance) in `xc04_check/xcheck2.py`.

---

## 10. UNVERIFIED items (collected)

1. Bognanni (2022): only the abstract was read (IDEAS; no open copy found). His analytic construction and any proposed alternative are unverified.
2. *(Resolved in the cross-check.)* CCCM journal numbering: the accepted manuscript (CCCM-AAM) has the same eqs (5)–(20) and footnotes as the WP. Only the typeset journal PDF was not seen.
3. CCM-WP eq. (42), now read from the rendered PDF: Var = λ1²λ2σ_i/(l^{λ3}σ_j), so λ1 = 0.05 is s.d.-scale. **Still UNVERIFIED:** whether the unsquared λ2 and σ_i/σ_j are intended or a typo for squared terms. No second source exists, because the journal moved the priors to an inaccessible online appendix. Also unresolved is the IW scale for Φ (Φ̲ = I or d_Φ·Φ̲ = I).
4. `bayesianVARs`:
   - whether `normal_sds` and `cholesky_normal_sds` are variances (as documented) or s.d.s (as named);
   - whether `priormean` (which exists, per the docs) is honoured under `prior="normal"`;
   - the layout and order of the stored `U`, `PHI` (intercept row) and `sv_para` draws;
   - whether `cholesky_priorhomoscedastic` "scale" means β0 in p(λ) ∝ λ^{−α−1}e^{−β/λ}, and the value it falls back to when left as `NA`;
   - whether the stationary h_0 prior has mean 0 (doc wording) or μ (GK);
   - **the sign convention of the stored U**, given that GK App. B prints the U-step without the minus sign (§3.4);
   - the Gibbs block order inside `bayesianVARs`.
5. The claim that the combined-weight form (5.3) runs at roughly the old TA's cost is derived, not benchmarked. Its *equality* with CCCM (14)–(15) and with the SUR oracle is verified numerically by both authors.
6. *(Resolved in the cross-check.)* Gibbs = Gauss–Seidel is published (Fox & Parker 2017, Bernoulli). Amit (1991) and Roberts & Sahu (1997) were checked bibliographically only.
7. The recommended defaults κ_A = 10 (A prior) and handling SOC/SUR dummies with fixed per-equation variance ψ_j are design recommendations, not literature. CCM-WP fn 7 supports only the general point that SOC-type priors are within-equation.
8. CCM-WP fn 9 (same-seed TA = SWA "exactly") is reported but is mathematically unexplained.

---

## 11. Further pointers (not needed for v1)

- Chopin, Fulop, Huo & Simoni (2026), arXiv:2609.27431. They reparametrise the model so the n volatility paths are conditionally independent (structural-form-like) and use Particle Gibbs. They report about 14× ESS/s for VAR coefficients and about 3.4× for volatilities versus the CTA at K = 15 (abstract only).
- Carriero, Clark & Marcellino, corrigendum to "Measuring uncertainty and its impact on the economy" (JAE): the same flaw affected that paper; changes were reported as small. Seen only in search results: https://journaldata.zbw.eu/dataset/6552f3b9-06ff-446a-b409-cd2e6622a91f/resource/545140b4-0d37-453b-b60c-4bcf52ea3864/download/corrigendum-ccm-jae-2020.pdf (not read).
- Chan23 §4.2.3: the recursive structural reparametrisation (Chan 2021; Gefang, Koop & Poon 2023) is the route rai-bvar already uses by default (SPEC D3).

---

## 12. Cross-check log

Done independently from primary sources (rendered PDFs read page by page, not text extractions) and from new NumPy scripts that do not reuse the first author's code. Scripts are in `xc04_check/` (`xcheck.py`, `xcheck2.py`, `xcheck3.py`, `xcheck4.py`); re-downloaded sources are in `xc04_dl/`. No GPL code and no CCCM Matlab code was opened.

**Verified as written (no change needed)**
- **Model.** CCCM eqs (1), (11)–(12), and Σ_t = A^{-1}Λ_tA^{-1}' with A unit lower triangular. Source: CCCM-WP pp. 1–5.
- **Corrected conditional.**
  - Eqs (13)–(15), including the definition of z (a_ii = 1), and the stacked form (16)–(20). Source: CCCM-WP pp. 6–7.
  - The combined-weight form (5.3) equals the SUR conditional. My oracle is built by polarisation with Σ_t^{-1} formed explicitly; it matches the toy problem to 1e-14 and 125 random problems to 3.5e-13.
  - Every number in the §9.1 table was reproduced to the printed digits.
- **The 2019 step.** CCM's flawed step, CCM-WP eqs (29), (32)–(33), uses the weight σ*^{−2}_{j,j,t} = 1/λ_jt. Rebuilt from an explicit Cholesky of Σ_t, it reproduces the "CCM 2019" rows of the table, and it is exact for j = n.
- **Kernel-invariance test.**
  - ρ(M) = 0.5037; mean and covariance recovered to 1e-15. Re-checked with a fixed-point Lyapunov solver.
  - M equals the block Gauss–Seidel matrix −(D+L)^{-1}U to 4e-16.
  - For the CCM-2019 step: ρ = 0, |Δmean| = 0.0395, |ΔS| = 0.028.
- **A-block.** The minus sign is right: checked by exact polarisation; the plus-sign variant is off by 1.5–2.9.
- **Edge cases.** A = I gives M = 0. The LDL and unit-variance Cholesky forms of the homoskedastic variant give identical π_j conditionals (CCCM App. §3 eqs 1–4).
- **Timings.** Fig. 1 data tips read 33.0975 / 2.542 s at N = 20 and 1181.7306 / 27.3167 s at N = 40. The text gives 3.25× (CTA vs TA), 13× and 43× (vs SWA). Source: CCCM-WP pp. 7–9.
- **Citations.** DOIs, volumes and pages of CCM (2019), CCCM (2022) and Bognanni (2022) were confirmed via Crossref. Chopin–Fulop–Huo–Simoni arXiv:2609.27431 was confirmed: authors, and the abstract's ≈14× and ≈3.4× ESS/s at K = 15.
- **Package defaults.**
  - `stochvol` 3.2.9 defaults and the meaning of `priorsigma` (sv-doc).
  - `bayesianVARs` 0.1.8: argument names and defaults, `prior_intercept` given as s.d., `normal_sds` and `cholesky_normal_sds` documented as variances, cholesky SV defaults c(0,100), c(20,1.5), c(0.5,0.5) (shape, rate), "stationary". Source: bVARs-doc, re-downloaded from CRAN (byte-identical).
  - The same SV values appear in GK §2.4, so each constant has two sources.
- **Chan23 and GK.** Chan23 fn 12 and §4.2.2 (BCM O(r²m), r = Tn) are as cited; GK §2 (A = U') is as cited.
- **Ordering invariance (§9.9).** Verified numerically (2e-15), with the correction in item 13 below.

**Corrected**
1. **§0 sources.**
   - Added the accepted manuscripts: CCCM-AAM, with the same equation numbers as the WP, which resolves old UNVERIFIED item 2; and CCM-AAM, whose journal numbering is shifted by −3 in §3.
   - Noted the different WP title of CCM 2016.
   - Added the Gibbs = Gauss–Seidel references.
   - Sources: cris.unibo.it 11585/900983 and 11585/713469.
2. **§3.1 independence assumption.** "Footnote 2 *requires* independence" was too strong. CCCM fn 2 says the assumption could be relaxed through a triangular factorisation of the prior (CCCM-WP p. 6, fn 2, continued). Added CCM-WP fn 7, which classifies SOC and unit-root priors as within-equation.
3. **§3.1 CCM eq. (42).**
   - The parameters are λ1, λ2, λ3, not θ1–θ3.
   - Read from the rendered PDF, the formula is Var = λ1²λ2σ_i/(l^{λ3}σ_j). So λ1 = 0.05 is s.d.-scale, which resolves the old ambiguity.
   - Flagged the unsquared λ2 and σ_i/σ_j as a possible typo with no second source.
   - Source: CCM-WP p. 28 (PDF p. 31).
4. **§3.1 and §9.8 `priormean`.** "Whether a non-zero prior mean can be set is UNVERIFIED" was wrong: `specify_prior_phi(priormean=...)` is documented (scalar, vector or matrix forms). Added it to the R call. Source: bVARs-doc, `specify_prior_phi`.
5. **§9.8 `sv_keep`.** The note assumed `logvar` holds T × M per draw. The default `sv_keep = "last"` stores only h_T, so `sv_keep = "all"` is now in the R call. Source: bVARs-doc, `bvar` arguments.
6. **§3.4 and §6.2 sign of U.**
   - GK App. B prints the U-step regression without the minus sign, which contradicts ξ_t = U'ε_t. Added a caveat. Source: GK arXiv v5 p. 44.
   - Replaced the circular "U'Σ_tU = D_t via vcov()" check with a sign-sensitive residual-correlation fixture check.
7. **§3.3 h_0 prior.**
   - Chan23 eq. 5 puts the stationary prior on h_1, not h_0; the marginal law of h_{1:T} is the same.
   - bVARs-doc says the stationary h_0 has "mean 0", while GK says μ_i; flagged.
   - `cholesky_priorhomoscedastic` defaults to `NA`, so it must be passed explicitly.
   - Sources: Chan23 p. 6; GK p. 5; bVARs-doc pp. 34–35.
8. **§3.3 CCM volatility prior.** "Φ ~ IW(I_N, N+2)" is replaced by the paper's IW(d_Φ·Φ̲, d_Φ) with d_Φ = N+2, keeping the paper's ambiguous "scale matrix set to an identity matrix" wording. Source: CCM-WP eq. 41, pp. 28–29.
9. **§4 typo footnote.** The typo concerned σ*_{j,j,t} (the Cholesky diagonal element = λ_jt^{1/2}), printed to the power −1 in the **journal** eqs (29)–(30). It did not concern "λ*". The 2016 WP prints the correct λ_{j,t}^{−1} but has λ_{j,t}ε_{j,t} in eq. (28). Also noted that CCCM eq. (5) as printed conditions on y^{(1:j−1)}, a typo relative to eq. (6) and the text. Sources: CCCM-WP fn 1 and pp. 2–3; CCM-AAM p. 12; CCM-WP pp. 11–12.
10. **§4 "why unnoticed".**
    - "Symmetric prior (conjugate case)" was imprecise. The 2019 step is exact for the Kronecker prior Σ⊗Ω0 with CCM's conditional priors, or for a flat prior. It is **not** exact for an independent prior that is identical across equations (error 0.37).
    - The claim that "large T makes differences small" holds only for constant volatility; under time-varying λ the error persisted at T = 160.
    - All numerical (`xcheck2.py`, `xcheck4.py`).
11. **§4 error size.** The largest mean shift is 0.09 posterior s.d., not 0.08, and the s.d. ratios reach 1.0555 (≈5.6%).
12. **§9.6 power check.** "Increase T" is wrong for constant-volatility configurations. Use informative independent priors, small T, time-varying λ and non-trivial A (item 10).
13. **§9.9 and §8 ordering.** After permuting a constant-A model with time-varying Λ_t, the LDL factor A_{P,t} is time-varying, so the test needs a per-t-A kernel. Verified numerically.
14. **§6.1 Cholesky failure.** Under `jit`, `jnp.linalg.cholesky` returns NaN rather than raising. Added a checkify or flag-based failure path so the SPEC §8 "raise with diagnostics" rule can be met.
15. **§6.5 IW Σ-step.** This step comes from CCM-WP eqs 21–23 and §3.2.2, not from the CCCM appendix, which gives only the π_j conditional. The prior density convention is now stated.
16. **§9.2 citations.** No longer UNVERIFIED: Fox & Parker (2017) states the identity in its abstract; Amit (1991) and Roberts & Sahu (1997) were checked bibliographically.
17. **§9.3 Monte Carlo figures.** These are seed-specific; a re-run gave 0.0039 and 0.994–1.006.

**Still UNVERIFIED** — see §10. The open items:
- Bognanni's full argument.
- The squaring in CCM (42) and the Φ IW scale.
- `bayesianVARs`: variance vs s.d. in `*_normal_sds`, whether `priormean` applies under `"normal"`, storage layouts, the IG "scale" meaning and its `NA` default, the h_0 mean, the sign of the stored U, and the internal block order.
- The cost claim for the combined-weight form.
- The design defaults (κ_A, the SOC/SUR treatment).
- CCM fn 9.
