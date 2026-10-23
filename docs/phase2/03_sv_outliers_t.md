# 03: Robust SV variants: outlier-adjusted SV (SVO) and SV with Student-t errors (SV-t, SVO-t)

Phase 2 research note for `rai-bvar` (SPEC §3, §4.1–4.4, §6.1, §7, §8, §10.3, §13 Phase 2).
Covers `SV(outliers=True)`, `SV(tails="t")` and their combination `SV(outliers=True, tails="t")`.
The base SV machinery (Omori 10-component mixture, tridiagonal precision sampler for `h`, ASIS for `(μ, φ, σ)`, the log-square offset) belongs to the companion SV note. Here it is used as a black box, "`SVcore`".

**Clean-room status.** Every algorithm below comes from papers or package *documentation* (cited with URL plus section or equation). GPL or unlicensed code was consulted only to confirm constants or behaviour, never transcribed:
- CCMM MATLAB replication code: no licence (GitHub API reports `license: null`). Greps only: prior constants, the outlier grid, the KSC table it loads, and the constant term of its ν log-likelihood (to confirm it uses the collapsed JPR form).
- `stochvol` C++ (GPL ≥ 2): greps only, to confirm the ν proposal type and that `τ` is updated before `ν`. `stochvol` R (`svsim`): one grep, to settle which t scaling the simulator uses (§7.4).
- Statements that could not be verified are marked **UNVERIFIED**.
- This note was independently cross-checked (formulas re-derived, constants checked against a second source, numerical checks run). See the **Cross-check log** at the end.

---

## 0. Summary of decisions (recommended defaults)

| Item | Recommended `rai-bvar` default | Source of the constant |
| :--- | :--- | :--- |
| Outlier scale `o_t` (multiplies the **std. dev.**) | Discrete: `o_t = 1` w.p. `1−p`; `o_t = k` w.p. `p/(K−1)` for `k ∈ {2,…,K}`, with `K = 20` (`K ≥ 2` required) | CCMM (2024) DP §3.1 item 3 = REStat §III.A ("U(2,20)"; the REStat text adds that the discrete grid is a convenient, tractable approximation). Implemented as the integer grid `s_k = k`, `w_1 = 1−p`, `w_k = p/(K−1)`: CCMM online appendix II.iv, fn. 8; replication code `SVOstates.values = 1:SVOmaxscale` with `SVOmaxscale = 20` (`goVARSVO.m` l. 36). Second source for the grid: Chan (arXiv 2208.13255, §6.3) discretises `U(2,20)` "using a grid", following SW (2016) and CCMM |
| Outlier probability prior, SVO | `p ~ Beta(2.5, 10f − 2.5)`. Prior mean `1/(4f)` (one outlier every 4 years), prior sample size `10f` (10 years). `f` = periods per year: monthly `Beta(2.5, 117.5)`, quarterly `Beta(2.5, 37.5)` | CCMM DP §3.2 = REStat §III.B. Code: `SVOalpha = 1/(4·np)·10·np`, `SVObeta = 10·np − SVOalpha` (`goVARSVO.m` l. 177–178; `np = 4` quarterly, `12` monthly). Second source for the "every 4 years" mean: Chan (arXiv 2208.13255, §6.3; hyperparameters not printed there) |
| Outlier probability prior, SVO-t | `p ~ Beta(1, 10f − 1)` (one outlier every 10 years): monthly `Beta(1, 119)`, quarterly `Beta(1, 39)` | CCMM DP §3.2 = REStat §III.B. Code: `SVOalpha = 1/(10·np)·SVOpriorobs`, `SVOpriorobs = 10·np` (`goVARSVOt.m` l. 177–179) |
| t mixing variable | `τ_t = q_t² ~ IG(ν/2, (ν−2)/2)` (**standardized**: `E τ = 1`, so `e^{h_t}` stays the conditional variance). Option `t_scale="unit"` gives `IG(ν/2, ν/2)` (JPR 2004, CCMM 2024) | stochvol ≥ 3.0 (NEWS 3.0.0; `update_t_error` docs); JPR §2.2.2 eq. (5); CCMM §3.1 item 4 |
| Prior on ν | Discrete uniform on `{3, 4, …, 40}` | JPR §2.2.2; CCMM DP §3.2 = REStat §III.B (code `tdofGrid = 3:40`, `goVARSVOt.m` l. 37) |
| ν sampler | **Collapsed exact discrete Gibbs** (not "griddy": the prior itself is discrete, so nothing is approximated): draw ν from `p(ν | e, h, o)` with τ integrated out (exact categorical over 38 values), then τ given ν | JPR §2.2.3 eq. (7) (block draw `p(λ, ν | ·) = p(ν | ·) p(λ | ν, ·)`). Numerically validated in the cross-check (ESS/N for ν ≈ 0.90 collapsed vs 0.02 uncollapsed) |
| Optional continuous ν prior (stochvol parity) | `ν − 2 ~ Exp(rate = 0.1)`; collapsed random-walk MH on `log(ν − 2)` | stochvol `svtsample(priornu = 0.1)`, `specify_priors(nu = sv_exponential(rate))`, "prior for nu−2" |
| Where these steps sit in the sweep | After the mean block, **before** the Omori indicators: `o → p → ν → τ → s → h → (μ,φ,σ)` | SPEC §4.3 (DNP 2015: nothing may sit between `s` and `h`); §5.1 below (validity argument and numerical check) |
| What `VolBlock.variances` returns | Effective variance `λ_t = e^{h_t} o_t² τ_t` | so mean-block weights `1/λ_t` down-weight outliers |
| Forecasting | `simulate_ahead` draws future `o_{T+j}` iid from `p`, `τ_{T+j}` iid from `IG`, and `h` from the SV law. `cond_chol = A⁻¹ diag(√λ_{T+h})` with the effective λ | SPEC §6.1; CCMM appendix II |

---

## 1. Notation

All quantities refer to one equation `i` of the recursive structural form (SPEC §4.1). Within the vol block the index `i` is dropped.

| Symbol | Meaning | Shape (per equation) | Suggested code name |
| :--- | :--- | :--- | :--- |
| `T` | estimation sample length | — | `T` |
| `e_t` | structural residual passed to the VolBlock: `e_t = y_{i,t} + Σ_{j<i} a_{ij} y_{j,t} − g_i(x_t)` | `(T,)` | `resid` |
| `h_t` | persistent log-variance (SV state); constant `h` for `Constant` | `(T,)` | `h` |
| `μ, φ, σ` | SV level, persistence, vol-of-vol | scalars | `mu, phi, sigma` |
| `K` | largest outlier scale (default 20) | static int | `outlier_max` |
| `𝒦 = {1,…,K}` | support of `o_t`, index `k` | static | `scales = arange(1, K+1)` |
| `o_t` | outlier scale on the **standard deviation** (`o_t = 1`: regular observation) | `(T,)` int | `o_idx` (store the index `k−1`, int8) |
| `w_k(p)` | prior pmf of `o_t`: `w_1 = 1−p`, `w_k = p/(K−1)` for `k ≥ 2` | `(K,)` | `log_w` |
| `p` | outlier probability | scalar | `p` (store `log_p`, `log1m_p`) |
| `a_p, b_p` | Beta prior on `p` | static | `p_prior` |
| `f` | periods per year (12 monthly, 4 quarterly), from `VARData` | static | `freq` |
| `τ_t = q_t²` | Student-t variance-mixing variable | `(T,)` | `tau` (store `log_tau`) |
| `ν` | degrees of freedom | scalar | `nu` (grid index `nu_idx` under the discrete prior) |
| `c(ν)` | t scale constant: `ν − 2` (standardized, default) or `ν` (`t_scale="unit"`) | scalar | `c_nu` |
| `𝒢 = {3,…,40}` | ν grid (G = 38 points) | static | `nu_grid` |
| `λ_t` | effective variance `e^{h_t} o_t² τ_t` (`τ ≡ 1` without t, `o ≡ 1` without outliers) | `(T,)` | `variances()` |
| `z_t²` | `e_t² e^{−h_t}` (residual standardised by the persistent part only) | `(T,)` | `log_z2 = log e² − h` |
| `ẽ_t` | cleaned residual `e_t / (o_t √τ_t)`, which feeds `SVcore` | `(T,)` | `log_e2_clean` |
| `s_t` | Omori mixture indicator (10 components) | `(T,)` | `s` |
| `c_off` | offset in `log(ẽ² + c_off)` (see the SV note) | static | `log_offset` |
| `IG(α, β)` | inverse gamma, density `β^α/Γ(α) x^{−α−1} e^{−β/x}` (shape α, scale β, which equals the rate of `1/x`) | — | — |
| `t_{c,ν}(x)` | density of `√τ ε` with `τ ~ IG(ν/2, c/2)`: `Γ((ν+1)/2) / (Γ(ν/2) √(πc)) · (1 + x²/c)^{−(ν+1)/2}`. `c = ν`: standard t; `c = ν−2`: unit-variance t | — | — |

---

## 2. Model

### 2.1 Observation and latent structure (one equation)

$$
e_t = e^{h_t/2}\, o_t\, \sqrt{\tau_t}\;\varepsilon_t,\qquad \varepsilon_t \overset{iid}{\sim} N(0,1),\qquad
\lambda_t \equiv \operatorname{Var}(e_t\mid h_t,o_t,\tau_t)=e^{h_t}o_t^2\tau_t .
$$

- **Persistent part.** `h_t = μ + φ(h_{t−1} − μ) + σ η_t`, with the random walk as the special case `φ = 1`. This is the SPEC §4.4 SV block, unchanged. `Constant` is the special case `h_t ≡ h`.
- **Outlier part (SVO).** The `o_t` are iid over `t`, independent of `h`, `τ` and of the other equations. `P(o_t = 1) = 1 − p` and `P(o_t = k) = p/(K−1)` for `k = 2,…,K`.
  - CCMM (2024) write `o_{j,t} = 1` w.p. `1−p_j` and `U(2, 20)` w.p. `p_j` (Bundesbank DP 13/2022, §3.1 item 3; identical in REStat 106(5), §III.A).
  - The lower bound 2 is "motivated by seeing outliers as events firmly outside the typical mass" (fn. 4).
  - The implementation is discrete: the online appendix (II.iv, fn. 8) states `s_k = k`, `w_1 = 1 − p`, `w_k = p/(K−1)` with `K = 20`, and the replication code uses the value grid `1:20`. The published REStat text (§III.A) states this explicitly: the authors regard the discrete grid as a convenient, tractable approximation and consider the rare-and-large shape, not the grid, to be what matters.
  - **Decision:** use the integer grid `{2,…,20}`. Every conditional is then an exact categorical draw. A continuous `U(2,20)` would need an extra MH or slice step.
  - **`K ≥ 2` is required.** `w_k = p/(K−1)` is undefined at `K = 1`; validate `outlier_max ≥ 2` at construction and point users to `outliers=False` instead.
- **t part (SV-t).** The `τ_t` are iid `IG(ν/2, c(ν)/2)`, independent of everything else.
  - With `c = ν` (JPR 2004 eq. (5); CCMM §3.1 item 4, "`q²_{j,t} ~ IG(d_j/2, d_j/2)`"), `√τ ε ~ t_ν` and `Var(e | h) = e^h ν/(ν−2)`.
  - With `c = ν − 2` (stochvol ≥ 3.0, NEWS 3.0.0: "normalized Student's t-distribution, i.e. its standard deviation is 1"; `update_t_error` docs: `tau_i ~ InvGamma(.5*nu, .5*(nu-2))`), `Var(e | h) = e^h`.
  - **Decision:** use `c = ν − 2` by default, so `e^{h_t}` keeps its meaning as the conditional variance of the structural shock (the stochvol JSS 2021 §2.2 rationale: "to ascertain interpretability of the scaling exp(h_t/2)"). That matters for one-s.d. IRFs (§6.3), GIRF scaling (§6.5) and any informative prior on `μ`. Requires `ν > 2`, which `𝒢 = {3,…,40}` guarantees.
- **SVO-t.** Both parts together: `λ_t = e^{h_t} o_t² τ_t` (CCMM §3.1 item 4: `Σ_t = A⁻¹ O_t Q_t Λ_t Q_t' O_t' A⁻¹'`).
- **Cross-equation structure.** `o_{i,t}`, `τ_{i,t}`, `p_i` and `ν_i` are equation-specific and independent across `i` ("mutually i.i.d. over all j and t", CCMM §3.1). Equation independence (SPEC §4.1) is therefore preserved and everything runs under `vmap`. A *common* outlier or t scale across equations (Lenza–Primiceri-like, or multivariate t) would couple equations and need a `GlobalBlock`; that is out of scope here.

### 2.2 Implied marginal (transitory-integrated) density of `e_t`

Given `h_t`, `p` and `ν`, integrating out `o_t` and `τ_t`:

$$
p(e_t\mid h_t,p,\nu)=e^{-h_t/2}\,f\!\left(z_t\right),\qquad z_t=e_te^{-h_t/2},
$$

| Model | `f(z)` |
| :--- | :--- |
| SV | `φ(z)` (standard normal pdf) |
| SVO | `Σ_k w_k(p) k⁻¹ φ(z/k)` |
| SV-t | `t_{c(ν),ν}(z)` |
| SVO-t | `Σ_k w_k(p) k⁻¹ t_{c(ν),ν}(z/k)` |

This density is used for (i) the collapsed ν step, (ii) the Rao-Blackwellised predictive density (§5.3), and (iii) the "marginal log-likelihood" test quantity in SBC (§7.3).

### 2.3 Prior moments (exact, for the integer grid)

- `E[o | o > 1] = (Σ_{k=2}^{20} k)/19 = 11`, so `E[o] = 1 + 10p`.
- `E[o² | o > 1] = (Σ_{k=2}^{20} k²)/19 = 2869/19 = 151`, so `E[o²] = 1 + 150p` and `Var(o) = 130p − 100p²`.
- General `K` (for tests with `outlier_max ≠ 20`): `E[o | o > 1] = (K+2)/2` and `E[o² | o > 1] = (2K² + 5K + 6)/6`.
- `E[τ] = 1` (standardized) or `ν/(ν−2)` (unit).
- At the prior means:
  - monthly SVO (`p = 1/48`): `E[o²] = 4.125`, `Var(o) = 2.665`;
  - monthly SVO-t (`p = 1/120`): `E[o²] = 2.25`.
- For comparison, the continuous `U(2,20)` mixture gives `E[o²] = 1 + 147p` and `Var(o) = 127p − 100p²` (4.0625 and 2.602 at `p = 1/48`).
- The CCMM appendix III calibration formula `Var(o) ≈ (1−p) + p·(20−2)²/12` (giving 1.54 at `p = 1/48`) is a heuristic: it mixes 1 with the variance of the continuous `U(2,20)` (27). It is neither `Var(o)` nor `E[o²]` of the grid or of the continuous mixture. Do not use it in tests.
  - Its SVO-t companion, as printed in the text extraction, reads `(1−p) + p·(18²/12)·d/(d−2)`. That gives 1.28 at `p = 1/120`, `d = 9`, not the reported 1.56. The reported 1.56 matches the product form `[(1−p) + p·18²/12]·d/(d−2)`. The SV-t value 1.67 at `d = 5` is `d/(d−2)`, i.e. `E[q²]` under the unit scale.

---

## 3. Priors, defaults, and who uses what

### 3.1 Outlier component

| Spec | Support of `o_t` | Probability | Prior on `p` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Stock & Watson NBER WP 21282 (2015), eq. (3) and p. 6 | `{1, 5, 10}` | **fixed** `(.975, 1/60, 1/120)` | none | `s_t` multiplies the transitory s.d. in a UC-SV inflation model; Omori 10-component mixture with the DNP correction (p. 9). URL: https://www.nber.org/papers/w21282.pdf |
| Stock & Watson (2016, REStat 98(4), 770–784, doi:10.1162/REST_a_00608) | `U(2,10)`, discretised on a grid: **second-hand only** | estimated | Beta, hyperparameters **UNVERIFIED** (CCMM attribute their Beta prior on `p` to SW 2016) | Published text not accessed (Princeton and Harvard hosts return 403). Second-hand evidence: Chan (arXiv 2310.14438, fn. 6) says the original SW (2016) uniform runs from 2 to 10; Chan (arXiv 2208.13255, §6.3) says SW (2016) and CCMM discretise it on a grid. Consistent with the CCMM code fallback `SVOmaxscale = 10` (`mcmcVARSVO.m` l. 175–176) |
| **CCMM SVO** (DP 13/2022 §3.1–3.2 = REStat 106(5) §III.A–B; appendix II.iv) | `{1} ∪ {2,…,20}` | `w_k = p/19` | `Beta(a, b)`, `a = (1/(4f))·10f = 2.5`, `b = 10f − 2.5` | Prior mean: one outlier every 4 years in monthly data. Prior precision: equivalent to 10 years of prior observations (paraphrase of DP §3.2) |
| **CCMM SVO-t** | same | same | `Beta(1, 10f − 1)` | Prior mean: one outlier every 10 years; same 10-year precision |
| `stochvol`, `bayesianVARs` 0.1.8, `bvartools` 0.3.0 | — | — | — | No outlier component in their documentation (reference manuals grepped). **There is no R oracle for SVO**, see §7.4 |

`rai-bvar` API proposal: `SV(outliers=True, outlier_max=20, outlier_prior=OutlierPrior(every_years=4, prior_years=10))`. This resolves to `Beta(f·prior_years/(f·every_years), f·prior_years − a)`. Because `f·prior_years/(f·every_years) = prior_years/every_years`, `a` does not depend on frequency. The frequency comes from `VARData`; raise an actionable error for unknown frequency unless `Beta(a, b)` is given explicitly. With `tails="t"` the default becomes `every_years=10`. These hyperparameters are not data-dependent, so there is no look-ahead (R9).

Prior moments of `p`:

| Prior | Mean | s.d. |
| :--- | :--- | :--- |
| `Beta(2.5, 117.5)` | 0.0208 | 0.0130 |
| `Beta(1, 119)` | 0.0083 | 0.0083 |
| `Beta(2.5, 37.5)` | 0.0625 | 0.0378 |
| `Beta(1, 39)` | 0.0250 | 0.0244 |

### 3.2 Student-t component

| Spec | `τ_t` law | ν prior | ν sampler |
| :--- | :--- | :--- | :--- |
| Jacquier, Polson & Rossi (2004, J. Econometrics 122(1), 185–212, doi:10.1016/j.jeconom.2003.09.001), §2.2.2 eq. (5), §2.2.3 eqs. (6)–(7), §5.1 (draft at https://people.bu.edu/jacquier/papers/jpr.je2004.pdf). JPR write `IG(ν/2, 2/ν)` in the convention where the second argument is the inverse scale; the density `∝ λ^{−(ν/2+1)} e^{−ν/(2λ)}` is shape `ν/2`, scale `ν/2` in ours | `IG(ν/2, ν/2)` (unit) | discrete uniform on `[3, 40]`; bounds 30 and 60 tried in §5.1 | **Blocked** draw `p(λ, ν | h, y) = p(ν | h, y) p(λ | ν, h, y)` with `ν` drawn from the product of t ordinates (λ integrated out). This avoids ν getting absorbed at the lower bound (Eraker et al. 1998). Reported ν autocorrelation about 0.6 |
| CCMM (2024) SVO-t, SV-t (DP §3.1 item 4, §3.2, fn. 9 = REStat §III.A–B, fn. 9; fn. 7 is only the prior-calibration remark) | `q² ~ IG(d/2, d/2)` (unit) | discrete uniform `{3,…,40}` (code `tdofGrid = 3:40`) | Follows JPR. The constant term of the code's ν log-likelihood (`tdof.loglike0`, grep only) is the t-ordinate form of JPR eq. (7), i.e. the collapsed draw. `q` is sampled **before** the mixture states (fn. 9) |
| Geweke (1993, JAE 8 S19–S40), via JPR §2.2.2 and stochvol JSS §2.2 | — | `p(ν) ∝ e^{−gν}` | — |
| Kastner (2015) "Heavy-tailed innovations in the R package stochvol" (vignette `heavytails.pdf` in stochvol 1.3.3, https://cran.r-project.org/src/contrib/Archive/stochvol/stochvol_1.3.3.tar.gz): eq. (1), §3.1, §3.2 eqs. (4)–(7). Applies to stochvol 1.1–2.x | `IG(ν/2, ν/2)` (unit) | `ν ~ U(a, b)` (example `priornu = c(2, 100)`, following Chib, Nardari & Shephard 2002) | `τ_t | · ~ IG((ν+1)/2, (ν + y_t² e^{−h_t})/2)`. ν **conditional on τ** by independence MH, with a Gaussian proposal at the mode (Newton–Raphson) and variance `−1/ℓ''`. The vignette warns that ν | τ "can be very inefficient if ν becomes large" |
| **stochvol ≥ 3.0** (current 3.2.9). Manual: `specify_priors`, `svtsample`, `update_t_error`. JSS 100(12) §2.1 eq. (2), §2.2–2.3 (https://doi.org/10.18637/jss.v100.i12) | `IG(ν/2, (ν−2)/2)` (**standardized**; NEWS 3.0.0, `update_t_error` docs, `svsim` source) | `ν − 2 ~ Exp(λ_ν)` (JSS §2.2, following Geweke 1993). `svtsample` default `priornu = 0.1`; `svsample` default `priornu = 0` (Gaussian). `specify_priors` default `nu = sv_infinity()` (Gaussian) | Independence MH **conditional on τ** with a Newton–Raphson/Laplace Gaussian proposal (JSS §2.3 refers to Kastner 2015; confirmed by grep of `single_update.cc`: the `τ_i` loop runs first, then a `newton_raphson` call on τ's sufficient statistics; no transcription) |
| Chan & Hsiao (2014), *Bayesian Inference in the Social Sciences*, ch. 6 (Wiley), pp. 155–176, doi:10.1002/9781118771051.ch6. Read as CAMA WP 74/2013, §4 eqs. (16)–(18), §4.1–4.2 (Wayback copy of https://cama.crawford.anu.edu.au/sites/default/files/publication/cama_crawford_anu_edu_au/2013-11/74_2013_chan_hsiao.pdf) | `λ_t ~ IG(ν/2, ν/2)` (unit), eq. (18) | `ν ~ U(0, ν̄)` (lower bound **0**, not 2), with `ν̄ = 50` in the §4.2 silver example | **Uncollapsed** `ν | λ`: independence-chain MH with a `N(ν̂, K_ν⁻¹)` proposal, mode `ν̂` found by Newton–Raphson using the same gradient and Hessian as Kastner (2015) eqs. (6)–(7). `λ_t | · ~ IG((ν+1)/2, (ν + e^{−h_t}(y_t−μ)²)/2)` |

**Recommended defaults:**
- `SV(tails="t")`: `t_scale="standardized"`, `nu_prior=DiscreteUniform(3, 40)`, collapsed exact discrete Gibbs (§4.3a).
- Option `nu_prior=ShiftedExponential(rate=0.1, shift=2)` (stochvol parity), sampled by collapsed RW-MH on `log(ν−2)` (§4.3b).
- Option `t_scale="unit"` (JPR/CCMM). With a discrete ν it shifts `h` (and hence `μ`) by the ν-dependent constant `log(ν/(ν−2))` (mapping in §7.4). The two parametrisations are therefore the same model only up to the prior on `μ` (or, for `Constant`, the IG prior on `λ̄`), which is not shift-invariant. The difference is negligible with `μ ~ N(0, 100²)`; it is not negligible with an informative `μ` prior.

---

## 4. Full conditional posteriors

Everything below is conditional on the current mean or contemporaneous draw, through `e_{1:T}`, and on `h_{1:T}`. All are exact: no mixture approximation enters them. Write `ℓ` for unnormalised log-densities.

### 4.1 Outlier scale `o_t` (SVO; or SVO-t conditional on `τ_t`)

The `o_t` are conditionally independent over `t`:

$$
P(o_t=k\mid e_t,h_t,\tau_t,p)\;\propto\; w_k(p)\,\frac1k\exp\!\Big(-\frac{z_t^2}{2k^2\tau_t}\Big),\qquad k=1,\dots,K.
$$

Logits: `ℓ_{t,k} = log w_k − log k − exp(log_z2_t − log_tau_t − 2 log k)/2`. The `1/k` factor (the Gaussian normalising constant `(k²τ e^h)^{−1/2}` with its `k`-free part dropped) is easy to drop by mistake; a Geweke mutation test targets exactly that bug (§7.2).

**Collapsed over τ (SVO-t, recommended).** Given ν, τ integrated out:

$$
P(o_t=k\mid e_t,h_t,\nu,p)\propto w_k(p)\,\frac1k\Big(1+\frac{z_t^2}{c(\nu)k^2}\Big)^{-(\nu+1)/2}.
$$

Rao-Blackwellised outlier probability per date, for reporting (accumulate in the scan carry; no need to store `o` draws):
`π_t = Σ_{k≥2} softmax(ℓ_{t,·})_k`.

### 4.2 Outlier probability `p`

With `N_1 = Σ_t 1{o_t > 1}` (the value of `o` given `o > 1` is uniform and does not depend on `p`):

$$
p\mid o_{1:T}\sim \mathrm{Beta}(a_p+N_1,\; b_p+T-N_1).
$$

### 4.3 Degrees of freedom `ν` (τ integrated out)

Let `r_t² = z_t²/o_t²` (`o_t ≡ 1` without outliers). The `o`- and `h`-dependent Jacobian terms do not involve ν and drop out:

$$
\ell(\nu)=\log\pi(\nu)+T\Big[\log\Gamma\big(\tfrac{\nu+1}{2}\big)-\log\Gamma\big(\tfrac{\nu}{2}\big)-\tfrac12\log\big(\pi c(\nu)\big)\Big]-\frac{\nu+1}{2}\sum_{t=1}^T\log\Big(1+\frac{r_t^2}{c(\nu)}\Big).
$$

With `c = ν` this is JPR eq. (7).

**(a) Discrete prior on 𝒢 (default).** Evaluate `ℓ` on all 38 grid points: a `(T, G)` `log1p` matrix, summed over `t`. Draw `ν` by `categorical(ℓ)`. This is exact, needs no tuning, and has no rejection, which suits `vmap`.

**(b) Continuous prior, `ν − 2 ~ Exp(λ_ν)` (stochvol parity).** Random-walk MH on `u = log(ν − 2)`, with target `ℓ(ν(u)) + u`; the `+u` is the Jacobian and `log π(ν) = log λ_ν − λ_ν(ν − 2)`.
- Step size is adapted during warmup only, then frozen. Adaptation must be off in Geweke and SBC runs.
- **Validity of MH inside the partially collapsed sweep.** van Dyk & Jiao (2015, JCGS, doi:10.1080/10618600.2014.930041) show that replacing a reduced-conditional draw in a PCG sampler with an MH update can change the stationary distribution. This scheme is safe for two reasons:
  - the MH kernel's input is `(o, p, ν)` only, never the trimmed `τ`, and it targets `p(ν | e, h, o)` with `τ` integrated out;
  - it is followed immediately by an exact `τ | ν, o` draw.
  Hence, if `(o, p, ν)` enters with its τ-marginal posterior, `(o, p, ν', τ')` leaves with the joint posterior. The cross-check ran this exact sweep (SVO-t, `ν − 2 ~ Exp(0.2)`, `h` known, `T = 40`, 200k sweeps) against a 2-D quadrature posterior: `E[ν]` 8.592 vs 8.605 (z = −0.46), `E[p]` 0.1026 vs 0.1027 (z = −0.51), and the ν deciles agree. **Do not** insert any step that uses `τ` between the MH step and the `τ` redraw.
- The stochvol-style alternative (independence MH on `ν | τ` with a Laplace proposal) is valid but mixes worse, because it is uncollapsed. If implemented, get gradient and Hessian from `jax.grad`/`jax.hessian` of the standardized log-target. For reference only, the unit-scale target is Kastner (2015) eq. (5): `ℓ(ν | τ) = (Tν/2) log(ν/2) − T log Γ(ν/2) − (ν/2) Σ(log τ_t + 1/τ_t)`, with derivatives in eqs. (6)–(7). Chan & Hsiao (2014, §4.1) give the same target and derivatives.
- **The standardized analogue is derived here, not taken from a source:** `ℓ(ν | τ) = T[(ν/2) log((ν−2)/2) − log Γ(ν/2)] − (ν/2) Σ log τ_t − ((ν−2)/2) Σ 1/τ_t`.
  - Cross-check: it equals `Σ_t log IG(τ_t; ν/2, (ν−2)/2)` from `scipy.stats.invgamma` up to a ν-free constant, to 1e-8 over `ν ∈ [2.5, 30]`.
  - Derivatives (each checked against finite differences):
    - `∂ℓ/∂ν = (T/2)[log((ν−2)/2) + ν/(ν−2) − ψ(ν/2)] − ½Σ log τ_t − ½Σ 1/τ_t`;
    - `∂²ℓ/∂ν² = T[1/(2(ν−2)) − 1/(ν−2)²] − (T/4) ψ'(ν/2)`.
  - The ν-dependence enters only through `Σ log τ_t` and `Σ 1/τ_t`, so the uncollapsed step costs O(T + iterations). This is its one advantage over the collapsed step (O(T·G)).

**(c) Fully collapsed variant (SVO-t, optional).** Integrate out both `o` and `τ`:
`ℓ(ν) = log π(ν) + Σ_t log Σ_k w_k k⁻¹ t_{c(ν),ν}(z_t/k)`.
Step order: `ν | e, h, p` (o and τ integrated) → `o | e, h, ν, p` (§4.1, collapsed) → `τ | e, h, o, ν` → `p | o`. This is a block draw of `(ν, o, τ)` given `p`, followed by `p | o`.

It mixes best, but costs `T·G·K` per equation and sweep: 228k log-evaluations at `T = 300`, about 4.6M per sweep at `n = 20`. The arithmetic for the SPEC Phase 2 benchmark (4 chains × 20 equations × 10,000 draws) is:

| Step | Evaluations per run | Time at 11 ns/eval | Time at 1 ns/eval |
| :--- | :--- | :--- | :--- |
| Fully collapsed ν (`T·G·K`) | 1.8·10¹¹ `log1p` | about 34 min | about 3 min |
| Partially collapsed ν (`T·G`) | 9.1·10⁹ `log1p` | about 100 s | about 9 s |

- 11 ns/eval was measured for NumPy float64 `log1p` on an Apple M1 Pro. XLA-CPU throughput is **UNVERIFIED**; 1 ns/eval is an optimistic bound.
- So `collapse="full"` breaks the 2-minute budget at that configuration. Keep it for small problems and tests.
- The partial step's cost is material, so benchmark it in the Phase 2 pilot. A fallback is the uncollapsed `ν | τ` step, which costs O(T) but loses about 40× in ESS for ν in the cross-check.

### 4.4 Mixing variable `τ_t`

$$
\tau_t\mid e_t,h_t,o_t,\nu\sim \mathrm{IG}\Big(\frac{\nu+1}{2},\;\frac{c(\nu)+z_t^2/o_t^2}{2}\Big)\quad\text{independently over }t
$$

With `c = ν` and `o ≡ 1` this is JPR eq. (6) and Kastner (2015) §3.1.

### 4.5 Constant variance with transitory components (test configuration; optionally user-facing)

With `h_t ≡ log λ̄` and `λ̄ ~ IG(α_0, β_0)`:

$$
\bar\lambda\mid e,o,\tau\sim \mathrm{IG}\Big(\alpha_0+\tfrac T2,\;\beta_0+\tfrac12\sum_t \frac{e_t^2}{o_t^2\tau_t}\Big).
$$

This gives an SV-free configuration in which every step is exact (no Omori approximation). It is the main vehicle for the closed-form tests in §7.1. The transitory components are best written as a `TransitoryScale` sub-component (`none | outlier | t | outlier+t`) composed with a persistent part (`Constant | SV`), per rule R1. `Constant(tails="t")` (homoskedastic BVAR with t errors) then comes for free.

### 4.6 SV core on cleaned data

Given `o`, `τ`: `log ẽ_t² = log e_t² − 2 log o_t − log τ_t`, which is exact. Then `SVcore` draws `s_t` from the 10-component Omori posterior using `log(ẽ_t² + c_off)`, then `h_{1:T}` (tridiagonal precision sampler), then `(μ, φ, σ)` (ASIS). No modification is needed. Under the offset, the offset scale applies to `ẽ`, not to `e`; document this in the SV module docstring.

### 4.7 Mean block

It receives `weights = 1/λ_t = exp(−h_t − 2 log o_t − log τ_t)`. Outliers and t-tail observations are down-weighted automatically. The reduced-form CCCM (2022) global block uses the same per-equation weights; nothing else changes.

---

## 5. Step order, validity, and predictive simulation

### 5.1 Sweep order (per equation, inside `VolBlock.sample`)

```text
mean / contemporaneous block (given λ_{1:T} from the previous sweep)          [SPEC §4.3]
e_t ← structural residuals
--- transitory block (conditions on h, never on s) ---
SVO   : o | e,h,τ,p (§4.1)        → p | o (§4.2)
SV-t  : ν | e,h (§4.3, τ integrated) → τ | e,h,ν (§4.4)
SVO-t : o | e,h,ν,p (§4.1, τ integrated) → p | o → ν | e,h,o (§4.3, τ integrated) → τ | e,h,o,ν
--- SV core (SPEC §8) ---
s | ẽ,h   →   h | s,ẽ,(μ,φ,σ)   →   (μ,φ,σ) | h (ASIS)                        # nothing between s and h
```

**Why this order is valid.**
1. *DNP (2015).* Every transitory step conditions on `h` with the **exact** Gaussian or t likelihood, so `s` is marginalised in those steps. `s` is drawn after them and immediately before `h`, which is the SPEC §4.3 rule. Within the DNP scheme the transitory states belong to the "θ | Σ" group.
2. *Partially collapsed Gibbs* (van Dyk & Park 2008, "Partially collapsed Gibbs samplers: Theory and methods", JASA 103(482), 790–796, doi:10.1198/016214508000000409: the tools are marginalise, permute, trim).
   - The ν step integrates `τ` out and is followed immediately by a fresh `τ` draw. Together they form an exact block draw of `(ν, τ) | e, h, o`, which is the JPR construction.
   - In SVO-t, the `o` step integrates `τ` out. The `τ` it implicitly draws is discarded, which is valid because no later step conditions on `τ` before `τ` is redrawn (`p | o` does not involve `τ`; the ν step integrates it out).
   - Equivalently, the sweep is a Gibbs sampler on the blocks `{(o, τ)}`, `{p}`, `{(ν, τ)}`, `{s, h, …}` with a redundant draw trimmed.
   - **Numerically verified in the cross-check.** The SVO-t transitory block ran with `h` known, `T = 40`, 3 planted outliers, `p ~ Beta(2, 18)`, `ν ~ DU{3,…,12}` standardized and 58k retained sweeps. It was compared with the exact `(p, ν)` posterior from quadrature over `p` and enumeration over ν:
     - `E[p]` 0.0846 vs 0.0845 (z = +0.74); `E[ν]` 3.931 vs 3.940 (z = −1.20); total variation on the ν pmf 0.003.
     - Mutations are caught: dropping `1/k` gives z(p) ≈ +392; `b_p + T` gives z(p) ≈ −30.
     - The uncollapsed `ν | τ` variant is also correct but has ν ESS/N 0.023 vs 0.90.
3. **Difference from CCMM.** CCMM fn. 9 (DP and REStat) samples `q` before the mixture states but draws `o` **after** `h`, conditional on the KSC mixture states. Their code uses the 7-component KSC table (`getKSC7values`, confirmed by grep in `mcmcVARSVO.m` l. 184 and `mcmcVARSVOt.m` l. 191), and the `o` step runs inside `StochVolOutlierKSC…`. That is also a valid Gibbs scheme for the *approximate* augmented model, but it makes `o` depend on the mixture approximation. Our exact `o` step does not, and it keeps the "nothing between `s` and `h`" rule trivially.
   - A third valid option, documented but not recommended: draw `(o_t, s_t)` jointly from the `K × 10` table `∝ w_k π_j N(log e_t² − 2 log k − log τ_t; h_t + m_j, v_j²)` immediately before `h`. Note that `ẽ` depends on `k`; there is no Jacobian, because `log k` is a shift. Under the offset, use `log(e_t² k⁻² τ_t⁻¹ + c_off)` instead.

### 5.2 `simulate_ahead(key, state, H)` (SPEC §4.2, §6.1)

For each posterior draw, for `j = 1..H`:
1. `h_{T+j} = μ + φ(h_{T+j−1} − μ) + σ η_j` (persistent part only; `Constant`: `h_{T+j} = h`).
2. SVO: `u ~ U(0,1)`. Set `o_{T+j} = 1` if `u ≥ p`, else uniform on `{2,…,K}`. The posterior draw of `p` is used, so parameter uncertainty about the outlier rate propagates.
3. SV-t: `τ_{T+j} = (c(ν)/2) / Gamma(ν/2, 1)`; in logs, `log τ = log(c/2) − loggamma(ν/2)`.
4. Return `λ_{T+j} = exp(h_{T+j} + 2 log o_{T+j} + log τ_{T+j})`.

Future `o` and `τ` are iid and **transitory**: they scale only that period's shock. Any persistence comes through the VAR dynamics of `y`, not through the volatility. This is the whole point of SVO: a COVID spike does not get extrapolated as a persistent rise in uncertainty, unlike plain SV (CCMM §5, §7).

Forecasting needs only `h_T`, `(μ, φ, σ)`, `p` and `ν` from the posterior. **The in-sample `o_{1:T}` and `τ_{1:T}` are not needed**, so their storage can be optional (`store_latent=False`). Keep the Rao-Blackwellised `π_t` and `E[o_t² τ_t | y]` as running means for diagnostics.

### 5.3 Conditional Gaussian moments for the predictive contract (SPEC §7)

Given draw `s`, the simulated path up to `T+h−1`, and the latent `(h, o, τ)_{T+h}` (all equations):

$$
m_{s,h}=A^{-1}\Gamma x_{T+h},\qquad L_{s,h}=A^{-1}\operatorname{diag}\big(e^{h_{i,T+h}/2}\,o_{i,T+h}\sqrt{\tau_{i,T+h}}\big)_{i=1}^n,\qquad y_{T+h}=m_{s,h}+L_{s,h}\varepsilon .
$$

`L` is lower triangular with a positive diagonal, so it is a valid Cholesky factor (SPEC §8). Invariant 1 (paths distributed as `N(m, LL')`) holds **conditional on the drawn transitory scales**. SPEC §6.1 already says "Student-t errors are conditionally Gaussian given the latent scale draw".

**Two further requirements:**
1. With `store_moments="lazy"`, recomputing the moments must reproduce the *same* `o_{T+h}` and `τ_{T+h}`. Derive their keys deterministically from `(chain, draw, horizon, equation)` per SPEC §8, or store them.
2. **Rao-Blackwellising the transitory scales in `log_pred_density`** (recommended extension; a contract change, flagged for review).
   - The §7 estimator averages `N(y; m, LL')` over drawn `o` and `τ`. It is unbiased in the density and consistent in the log, but very noisy when `y_obs` is extreme. At `p ≈ 1/48`, only about 2% of draws carry `o > 1` for a given variable and horizon.
   - CCMM (appendix II, pp. S.15–S.16, paraphrased): only the persistent SV states need to be simulated one step ahead; the transitory outlier and t states are integrated out analytically. This Rao-Blackwellisation greatly improves the accuracy of the predictive likelihood, especially for extreme observations. The main text (DP §5.3 "Model fit" = REStat §V.C) says the same: integrating out future transitory states, rather than simulating their arrival, proved particularly important.
   - Exact formula, valid for the **full vector, or any leading block `{1,…,j}` of the ordering**, at any `h`, conditional on the path up to `T+h−1`. Since `|A| = 1`:

     $$
     \log p(y_{1:j,T+h}\mid\cdot)=\sum_{i\le j}\Big[-\tfrac12 h_{i,T+h}+\log f_i\big(\eta_i\big)\Big],\quad \eta_i=e^{-h_{i,T+h}/2}\big(A(y-m)\big)_i,
     $$

     with `f_i` from §2.2. The structural residual `(A(y − m))_i` depends only on `y_{1:i}`. This is CCMM eqs. (S.11), (S.14)–(S.16). Cost: `n × K` per draw and horizon.
     - Cross-check (`n = 3`, random unit-lower-triangular `A`, standardized SVO-t, residual of −6 in equation 2, 2·10⁶ Monte Carlo draws of `(o, τ)`): the RB value matches the Monte Carlo log-density within 0.006 for the full vector (−10.149 vs −10.155) and for the leading block `j = 2` (−8.557 vs −8.562; Gaussian marginalised explicitly). The remaining gap is Monte Carlo noise from the extreme residual, which is exactly the problem RB removes.
   - For non-leading subsets (for example variable 3 alone), the marginal is a convolution of mixtures (`K^j` configurations, about `6.6·10^20` at `K = 20`, `n = 16`, per CCMM p. S.20). There, fall back to the drawn-scale estimator.
   - **Proposal:** store `cond_chol_persistent` (the `e^{h/2}` part only), or recompute it lazily, plus per-draw `(p_i, ν_i)`. Expose `log_pred_density(..., rb_transitory=True)`, valid when `variables` is `None` or a leading block; otherwise warn and fall back.
   - Optionally use `J > 1` inner draws of `h_{T+1}` per posterior draw (CCMM eq. (S.9), "J draws of Λ_{t+1}").
3. **Predictive variance sanity check (h = 1, standardized t, SVO-t):**
   `E[λ_{T+1} | θ, h_T] = E[e^{h_{T+1}}] · (1 + 150p) · 1`, with `E[e^{h_{T+1}}] = exp(μ + φ(h_T − μ) + σ²/2)`.

### 5.4 IRF / GIRF scaling

The impact matrix `A⁻¹Λ_t^{1/2}` (SPEC §6.3) should use the **persistent** variance `e^{h_t}` by default. With standardized t this is the shock variance net of outliers. Under `t_scale="unit"` it is `e^{h} ν/(ν−2)`. Offer `VolBlock.variances(which="persistent" | "effective")`. Document that a "one-s.d. shock in March 2020" means the persistent part, not `o`.

---

## 6. Numerics and implementation details (JAX, float64)

- **Static configuration.** `has_outliers`, `has_t`, `K`, the ν grid, `t_scale` and `collapse` are Python-level static. Compile one kernel per configuration (SPEC §14: "no Python-level branching inside sweeps"). Shapes per equation: `(T,)`, `(T, K)`, `(T, G)`, all fixed, so `vmap` over equations and chains and `lax.scan` over iterations are straightforward.
- **Log space everywhere.**
  - Keep `log_z2 = log(e²) − h` and `log_tau`, never `exp(h)`.
  - Categorical draws use `jax.random.categorical(key, logits)` (Gumbel-max). It accepts `−inf` logits; check that the logits are float64.
  - Precompute `log k`, `log w_k = [log1m_p, log_p − log(K−1), …]` per sweep.
  - With `log_p = −inf`, the logits are `−inf` for `k ≥ 2`, which is fine. Never compute `(−inf) − (−inf)`.
- **Beta draw.** Draw via `jax.random.loggamma`: `G_a = loggamma(a')`, `G_b = loggamma(b')`, `log p = G_a − logaddexp(G_a, G_b)`, `log(1−p) = G_b − logaddexp(G_a, G_b)`. This avoids underflow when `a'` is small (the default `a' ≥ 1`, but user priors may not be).
- **Inverse-gamma draw.** `log τ = log(scale) − loggamma(shape)`, with `shape = (ν+1)/2 ≥ 2`. Huge `z²` (an outlier with `o = 1` in the current state) gives a huge `τ`, which float64 handles.
- **ν grid.** Precompute per grid point `A_g = lgamma((ν_g+1)/2) − lgamma(ν_g/2) − ½ log(π c(ν_g))`. Then `ℓ_g = log π_g + T·A_g − ((ν_g+1)/2) Σ_t log1p(exp(log_r2_t − log c_g))`. Cost `T × 38` `log1p` calls per equation and sweep; see the cost table in §4.3(c), since this step is material against the Phase 2 budget. With `t_scale="unit"`, `ℓ_g` equals JPR eq. (7) up to a ν-free constant (checked to 1e-9).
- **Cleaned residual for the Omori step.** `log(ẽ² + c_off) = logaddexp(log_e2 − 2 log o − log τ, log c_off)`. Exact zeros in `e` (`log e² = −inf`) are harmless in §4.1 and §4.4 (`z² = 0`). Only the Omori step needs the offset.
- **Keys.** Split per (chain, iteration, block, equation) as SPEC §8 requires. Sub-keys inside the vol block: `o`, `p`, `ν`, `τ`, `s`, `h`, `asis`. `simulate_ahead` keys: per (chain, draw, horizon, equation, {h, o_u, o_k, τ}).
- **Initial state.** `o ≡ 1`, `τ ≡ 1`, `p = a_p/(a_p+b_p)`, `ν` = grid median (about 21) or 10. Burn-in handles the rest. For COVID samples, an optional warm start sets `o_t = min(K, max(1, round(|e_t|/σ̂_robust)))` using an IQR-based scale.
- **Storage.** Per draw: `p`, `ν` (`(chain, draw, n)`). Running Rao-Blackwellised means of `π_t` and `E[o_t²τ_t]` go in the scan carry. `o` and `τ` draws are optional (`(chain, draw, n, T)`, the same size as `h`).
- **Diagnostics (SPEC §10.6).** Add `p_i`, `ν_i`, `N_1,i` and `π_t` at flagged dates to the monitored set. ν is discrete: rank-normalised R-hat still works, but report the "escape probability" `P(ν^{(m+1)} ≠ ν^{(m)})` as well (JPR §5.2). *Report* (do not warn as an error) when the ν posterior piles up at 40 (near-Gaussian, so SV suffices) or at 3. Mass at the lower bound is normal for macro data: in CCMM's SVO-t on 16 FRED-MD variables, the posterior mean of ν is 3 for about half the variables and above 20 for six (DP §5.1 = REStat §V.A).

### Edge cases

| Case | Behaviour / handling |
| :--- | :--- |
| Residual larger than `K·e^{h/2}` (e.g. April 2020 payrolls: Δlog ≈ −0.147, tens of pre-COVID s.d.; the exact multiple is **UNVERIFIED**) | `o` saturates at `K`; `h_t` must absorb the rest, so a persistent (RW) SV inherits part of the spike. Expose `outlier_max`. How often the cap binds in general is **UNVERIFIED**. Partial evidence from CCMM DP §5.1 / REStat §V.A, Fig. 2 (SVO-t, payrolls, posterior medians): total residual volatility peaks near 90 against a persistent component near 12. The implied transitory multiplier, about 7.5, is below 20, yet the persistent part still rose about 8-fold over 2020 |
| `outlier_max = 1` | Undefined (`w_k = p/(K−1)`). Reject at construction; use `outliers=False` |
| No outliers in the window | `p` posterior ≈ prior; forecasts still carry outlier risk at rate `E[p | y]`. This is intended |
| SVO-t: `o` and `τ` both explain a large residual | Weakly identified split. Report the product `o²τ` and `π_t`, not `o` alone |
| ν at grid bounds | Mass at 40: suggest `tails=None`. Mass at 3: heavy tails. If `outliers=False`, suggest `outliers=True`. With SVO-t it is expected (CCMM: about half of 16 variables), so report it, do not warn. Lowering the bound below 3 is not allowed under `t_scale="standardized"` (needs `ν > 2`, and `c(ν) → 0` as `ν → 2`) |
| `t_scale="unit"` with an informative `μ` prior | `μ` shifts by `log(ν/(ν−2))`. Document it, or use the standardized scale |
| Exact zero residuals | Only the Omori step needs `c_off`; the transitory steps are exact at `z = 0` |
| Short `T` or unknown frequency | Require an explicit `Beta(a, b)` |
| `SV(outliers=True)` combined with `outliers="lenza-primiceri"` | Reject at construction (double treatment) |
| COVID warning (SPEC §5.6) | `SV(outliers=True)` counts as "an option selected"; no warning |

---

## 7. How to test it

### 7.1 Closed-form and exact checks (unit layer, every PR)

All of these use the SV-free configuration of §4.5 (`h` known or `Constant`), so no Omori approximation is involved.

1. **`o_t` categorical (T = 1).** For fixed `(e, h, τ, p)`, compare `10^6` draws of §4.1 with the brute-force pmf computed in NumPy from `w_k N(e; 0, e^h k² τ)`. Chi-square test. Repeat for the collapsed-τ version against `w_k k⁻¹ t_{c,ν}(z/k)` evaluated with `scipy.stats.t` (`scale = sqrt(c/ν)`).
2. **`p | o`.** Mean and variance of the draws vs `Beta(a+N_1, b+T−N_1)`, plus a KS test.
3. **`τ_t`.** Draws vs `IG((ν+1)/2, (c + z²/o²)/2)`: mean `β/(α−1)`, variance `β²/((α−1)²(α−2))`, KS test. The variance is finite only for `α > 2`, i.e. `ν > 3`; at `ν = 3` (`α = 2`) use the mean and KS only, or test `log τ` moments.
4. **ν grid posterior.** For fixed `(e, h, o)`, the categorical pmf from §4.3 must equal `π(ν) Π_t t_{c,ν}(r_t)` computed independently with `scipy.stats.t.logpdf` (tolerance 1e-10 on normalised log-probabilities). Chi-square on draws.
5. **SVO with known `λ̄`: exact posterior of `p` by 1-D quadrature.**
   `p(p | e) ∝ Beta(p; a, b) Π_t [(1−p) φ(e_t/√λ̄) + (p/(K−1)) Σ_{k≥2} k⁻¹ φ(e_t/(k√λ̄))]`.
   Also `P(o_t > 1 | e) = ∫ π_t(p) p(p | e) dp`. The Gibbs sampler `(o, p)` must match the posterior mean, s.d. and 5/50/95% quantiles of `p`, and every `P(o_t > 1 | e)`, within 4 MCSE. Use `T = 50` with 3 planted outliers.
6. **SV-t with known `λ̄`: exact ν posterior.** `p(ν | e) ∝ π(ν) Π_t t_{c,ν}(e_t/√λ̄)/√λ̄`, which is exact on the grid. Compare draw frequencies by chi-square. With unknown `λ̄ ~ IG(α_0, β_0)`: do 1-D quadrature over `λ̄` per grid ν, giving the exact joint `(λ̄, ν)` posterior.
7. **Partially collapsed vs uncollapsed.** On the setting of test 6, run (a) collapsed ν then τ and (b) plain Gibbs `τ | ν`, `ν | τ` for long chains. Their stationary distributions must agree (KS on ν and `λ̄`).
   - (a) must have higher ESS per draw for ν (regression guard). Use a fixed seed and a margin, e.g. ratio > 3. In the cross-check's SVO-t setting the ratio was about 39 (0.90 vs 0.023).
   - A ready-made exact reference for the SVO-t transitory block is the cross-check's quadrature-over-`p` × enumeration-over-ν posterior (§5.1).
8. **Limits.**
   - `p` fixed at 0 (static flag) ⇒ SVO posterior = SV posterior (distributional equality within MCSE).
   - ν fixed at 10⁴ ⇒ SV-t ≈ SV.
   - (`K = 1` is not a valid limit: `w_k` is undefined. Use the `p = 0` flag instead, and test that `outlier_max = 1` raises.)
9. **Prior draws of `simulate_ahead`.** For fixed `(p, ν)`:
   - the frequency of `o > 1` is `Binomial(N, p)`;
   - `o | o > 1` is uniform on `{2,…,K}` (chi-square);
   - `τ ~ IG(ν/2, c/2)` (KS);
   - `E[λ] = E[e^h](1 + 150p)` for `K = 20`, and `E[e^h]·[1 + p((2K²+5K+6)/6 − 1)]` for general `K`. The `K = 20` case, with `E[e^{h_{T+1}}] = exp(μ + φ(h_T−μ) + σ²/2)`, was checked to < 1% by 2·10⁶ draws in the cross-check.
10. **Contract.** Invariant 1 (simulation consistency) for every registered composition `Linear × {SV(outliers), SV(t), SV(outliers, t)}`. Lazy moments must reproduce the same `o_{T+h}` and `τ_{T+h}`.
11. **Rao-Blackwellised LPD (if adopted).**
    - (a) For a single draw with fixed `(θ, h_{T+1})`, the RB density must equal a `10^6`-draw Monte Carlo average over `(o, τ)` (relative error < 1e-3).
    - (b) On simulated data with a planted outlier at `T+1`, RB and drawn-scale estimators must agree within combined MCSE, and the RB variance across seeds must be lower (assert a ratio < 0.5).
    - (c) The leading-block marginal must equal the full-vector computation when `j = n`.
12. **Scale equivariance (approximate).** Multiply `e` by `c`. Then `h` shifts by `2 log c` and the posteriors of `o`, `p`, `ν`, `τ` are unchanged (with a flat-ish `μ` prior and `c_off` scaled by `c²`).

### 7.2 Geweke (2004) joint-distribution tests (SPEC §10.3)

- **Marginal-conditional simulator.** Written independently of the sampler, in NumPy and not sharing code with the kernels, so that mirrored bugs are not masked. Draw `(μ, φ, σ, h_{0:T}, p, o_{1:T}, ν, τ_{1:T})` from the prior, then `e_t = e^{h_t/2} o_t √τ_t ε_t`. For composition-level tests, draw `Γ` and `A` too and generate `y` recursively.
- **Successive-conditional simulator.** Alternate one full sweep (§5.1) with `e | latent` (or `y | θ, latent`).
- **Configurations.** Small and proper priors with enough outliers for power:
  - `T = 40–60`, `p ~ Beta(2, 18)`;
  - `ν ~ DU{3,…,12}` (small grid for power), or `ν − 2 ~ Exp(0.2)` for the MH variant;
  - SV priors as in the SV note, with `φ` informative (e.g. `(φ+1)/2 ~ Beta(20, 1.5)`).
  - Run each of: (i) known `λ̄` + O, (ii) known `λ̄` + t, (iii) `Constant(IG)` + O + t, (iv) SV + O, (v) SV + t, (vi) SV + O + t, (vii) `Linear × SV(outliers=True, tails="t")` with `n = 2`, `p = 1`, `T = 30`.
- **Test functions `g`** (first and second moments; z-tests with MCSE; Holm correction across all `g`):
  - `p`, `ν`, `1{ν ≤ 5}`, `N_1/T`, `mean(log o)`, `mean(log τ)`;
  - `μ`, `φ`, `σ²`, `h_1`, `h_T`, `mean(h)`;
  - `Σ_t log N(e_t; 0, λ_t)`, `Σ_t e_t²/λ_t`, and the transitory-marginal log-likelihood `Σ_t log f(e_t | h_t, p, ν)`;
  - cross-products `p·N_1` and `ν·mean(log τ)`.
- **Mutation tests (power check, run occasionally).** Each injected bug must make Geweke fail:
  - drop the `1/k` factor in §4.1;
  - use `b_p + T` instead of `b_p + T − N_1`;
  - use `c = ν` in §4.4 while the prior uses `c = ν − 2`;
  - insert the `o` step between `s` and `h`. The power of Geweke against this DNP-type error, given the Omori tolerance, is **UNVERIFIED**; if it does not fail, keep it as a documented limitation rather than dropping the rule.
  - The cross-check confirmed that the first two mutations are detected by an exact-posterior comparison (§5.1). Geweke power for them is expected but not run.
- **Mixture-approximation caveat.** Configurations (iv)–(vii) inherit the Omori approximation in `h | s`. Apply whatever tolerance or correction the SV note adopts. Configurations (i)–(iii) are exact and must pass at the full test size.

### 7.3 SBC (Talts et al. 2018; Modrák et al. 2025)

- 500–1000 replications. `T = 120` at the block level and `T = 100`, `n = 2` at composition level. Priors as in §7.2 but closer to defaults (e.g. `p ~ Beta(2, 38)`). 4 chains, thinned to 99 posterior draws, giving ranks in `{0,…,99}`. ECDF-envelope uniformity test at 1% after multiplicity correction.
- **Discrete quantities** (`ν`, `N_1`) need **randomised tie-breaking**: rank = `#{post < true} + U{0, …, #{post = true}}`. Without it, uniformity fails spuriously.
- **Test quantities:**
  - parameters: `p`, `ν`, `μ`, `φ`, `σ`, `h_{T/2}`, `h_T`, `N_1`, `Σ log τ_t`;
  - data-dependent (Modrák et al.): the transitory-marginal log-likelihood `Σ_t log f(e_t | h_t, p, ν)`, the conditional log-likelihood `Σ_t log N(e_t; 0, λ_t)`, and `Σ_t 1{o_t > 1} z_t²` (sensitive to mis-assigned outliers);
  - composition level: `log p(y | Γ, A, λ)` as well.

### 7.4 Parity oracles

**stochvol (GPL ≥ 2, oracle only) for SV-t.** Current version 3.2.9 (https://cran.r-project.org/web/packages/stochvol/). Run **block-level** (feed `e = y`, no mean), or with `designmatrix = "ar0"` against our `Linear` with intercept only and prior `N(0, 10000²)` (`priorbeta = c(0, 10000)`).

| stochvol ≥ 3.0 | `rai-bvar` (`t_scale="standardized"`) | `rai-bvar` (`t_scale="unit"`) |
| :--- | :--- | :--- |
| `y_t = e^{h_t/2} ε_t`, `ε_t ~ t_ν(0, 1)` with **variance 1** (JSS eq. (2)); `τ_t ~ IG(ν/2, (ν−2)/2)` | identical | `h^{ours} = h^{sv} − log(ν/(ν−2))`; `τ^{ours} = τ^{sv}·ν/(ν−2)` |
| `mu = sv_normal(0, 100)` (`priormu = c(0, 100)`, sd 100) | `μ ~ N(0, 100²)` | `μ^{ours} = μ^{sv} − log(ν/(ν−2))` (the prior shift is negligible at sd 100) |
| `phi = sv_beta(5, 1.5)` on `(φ+1)/2` (`priorphi = c(5, 1.5)`) | same | same |
| `sigma2 = sv_gamma(0.5, 0.5)` ⇔ `σ² ~ 1·χ²_1` ⇔ `σ ~ |N(0, 1)|` (`priorsigma = 1`) | same | same |
| `nu = sv_exponential(0.1)`: prior on `ν − 2` (`svtsample(priornu = 0.1)`) | `nu_prior = ShiftedExponential(0.1)` (continuous, §4.3b) | same |
| `latent0_variance = "stationary"`: `h_0 ~ N(μ, σ²/(1−φ²))` | same | same |
| `keeptau = TRUE` | store `τ` | — |

- **Compare** 5/25/50/75/95% posterior quantiles of `μ`, `φ`, `σ`, `ν`, `h_t` (several `t`) and `τ_t` at the 3 largest `|y_t|`, within 4 MCSE. Use long runs on both sides (e.g. `draws = 50000`).
- **Discrete ν prior.** stochvol cannot express it. Instead run `nu = sv_constant(ν_0)` (e.g. ν₀ = 5) on both sides and compare `(μ, φ, σ, h, τ)`; plus the exact grid checks in §7.1.
- **`svsim` (resolved in the cross-check).** Its manual is internally inconsistent:
  - the `tau` entry says "Gamma⁻¹(shape = ν/2, rate = ν/2 − 1)" (standardized);
  - the "Note" says `y <- exp(-h/2)*rt(h, df=nu)` with volatility `sqrt(nu/(nu−2)) exp(h/2)` (unit).

  The 3.2.9 R source (`R/simulation.R`, `svsim`, grep only) does `tau <- 1/rgamma(len, shape=nu/2, rate=nu/2-1)` and `y <- exp(h/2)*sqrt(tau)*eps`. The simulator is therefore **standardized**, consistent with the sampler; the man-page Note is stale (it predates 3.0.0). `svsim` also asserts `nu > 2` and draws `h_0` from the stationary law. It is usable for fixtures, but simulating in Python and writing the data to the fixture is still preferred, so that the DGP lives in our code.
- **Older versions.** stochvol 1.1.0–2.0.5 (all pre-3.0 versions with t errors; 1.1.0 introduced them) used the unit scale with a `U(a, b)` ν prior (Kastner 2015; NEWS 1.1.0 and 3.0.0). It is usable as an oracle for `t_scale="unit"` with a continuous uniform prior, but pinning an old version is not recommended.
- **bayesianVARs** 0.1.8 and **bvartools** 0.3.0: Gaussian SV only (manuals grepped for t, outlier and degrees-of-freedom options). They remain the parity oracle for `Linear(form="reduced") × SV`, not for these variants.

**SVO: no package oracle.** Use, in order of strength:
1. The exact quadrature checks of §7.1 (tests 5, 6).
2. An **independent slow reference sampler** in NumPy for `T ≤ 100`. Use single-site or block MH on `h` with the *exact* likelihood (no Omori), plus the exact `o`, `p`, `ν`, `τ` steps. It shares no code and no approximation with the JAX kernel. Compare posterior quantiles within 4 MCSE; this also bounds the Omori-approximation effect.
3. Optionally, the CCMM MATLAB replication code (https://github.com/elmarmertens/CCMMoutlierVAR-code) as a *forecast-level* comparison on FRED-MD. Run it, never read it for algorithms.
   - It has no licence (GitHub API `license: null`, checked) and needs MATLAB.
   - Its model differs: RW SV with correlated innovations `Φ ~ IW`, 7-component KSC, `o` drawn after `h` conditional on mixture states, CCM prior on `Π`.
   - The repository also ships AR(1)-SV drivers (`goVARSVOar1.m`, `goVARSVOtar1.m`, matching the appendix's AR(1) robustness check). These are closer to our AR(1) SV.
   - Running it is **UNVERIFIED**; use it for qualitative comparison only.

### 7.5 DGP and replication (SPEC §10.5, §10.7)

- DGP "Linear, SV + outliers": the integer grid with `p = 0.02`, `T = 300`. Expect `SV(outliers=True) ≥ SV` on LPD; the false-positive rate on a no-outlier SV DGP should be at or below the nominal level.
- Add a t-DGP (ν = 5): expect `SV(t) ≥ SV`.
- **Qualitative replication of CCMM.** 16-variable FRED-MD (April 2021 vintage), 1959:3–2021:3, `p = 12` lags, 1,200 draws with 200 burn-in (DP §3.2, §4 = REStat §III.B, §IV).
  - SVO and SVO-t predictive densities after March 2020 should be much less dispersed than SV.
  - In cumulative 1-step log predictive score over 2020:03–2021:02, SVO should beat SV. Reference values (Table 2, identical in DP and REStat; log BF vs SV): SVO +232.59, SVO-t +225.33; full window 1975:01–2021:02: SVO +334.84, SVO-t +244.11.
  - Posterior means of `p` in SVO range from about 0.3% (housing starts) to 3.19% (real income); ν posterior mean is 3 for about half the variables in SVO-t (DP §5.1).
  - Our SV is independent AR(1) or RW per equation, not RW with correlated `Φ`, so the match is qualitative only (signs and rough magnitudes, not the numbers).

---

## 8. API / state sketch (for the Phase 2 implementer)

```python
SV(outliers=True, tails="t",                 # SVO-t
   outlier_max=20,                           # K; grid {1} ∪ {2..K}
   outlier_prior=OutlierPrior(every_years=10, prior_years=10),   # → Beta(1, 10f−1); SVO default every_years=4
   nu_prior=DiscreteUniform(3, 40),          # or ShiftedExponential(rate=0.1) for stochvol parity
   t_scale="standardized",                   # or "unit" (JPR/CCMM)
   collapse="partial")                       # "partial" (default, §5.1) | "full" (§4.3c) | "none" (tests only)

class RobustSVState(NamedTuple):
    h: Array          # (T,)
    mu: Array; phi: Array; sigma: Array
    o_idx: Array      # (T,) int8, k-1        [outliers]
    log_p: Array; log1m_p: Array              # [outliers]
    log_tau: Array    # (T,)                  [t]
    nu_idx: Array     # () int   (discrete)   / nu: Array () (continuous) [t]
    rb_out_prob: Array  # (T,) running mean of π_t (carry; diagnostics)
```

`variances(state, which="effective")` returns `exp(h + 2·log(o) + log_tau)`. `simulate_ahead` follows §5.2.

---

## 9. UNVERIFIED items

Status after the cross-check. Items marked RESOLVED are kept so the trail stays visible.

1. **Stock & Watson (2016, REStat 98(4)) published outlier specification: still UNVERIFIED from the primary source.**
   - Only NBER WP 21282 was read: fixed `{1, 5, 10}` with probabilities `.975, 1/60, 1/120`.
   - Second-hand: Chan (arXiv 2310.14438, fn. 6) gives `U(2,10)`; Chan (arXiv 2208.13255, §6.3) says SW (2016) discretise it on a grid. The Beta-prior attribution comes from CCMM.
   - The Beta hyperparameters of SW (2016) are unknown. Nothing in `rai-bvar` depends on them.
2. **RESOLVED: CCMM REStat 106(5), 1403–1417 (2024).** The published text matches the DP on every constant used here: `U(2,20)`, the Beta prior calibration, the DU{3..40} ν prior, `IG(d/2, d/2)`, fn. 9 step order, 1,200/200 draws, and Table 2. It adds an explicit sentence that the outlier values are on a discrete grid.
3. **RESOLVED: Chan & Hsiao (2014).** Read via a Wayback copy of CAMA WP 74/2013: unit `IG(ν/2, ν/2)`, `ν ~ U(0, ν̄)` with `ν̄ = 50` in the example, and uncollapsed independence-chain MH with a Newton–Raphson proposal. The published Wiley chapter text was not compared.
4. **RESOLVED: stochvol `svsim`.** The 3.2.9 source simulates the standardized t; the man-page "Note" is stale (§7.4).
5. **Partly resolved: stochvol 3.x ν sampler.**
   - Confirmed by grep: `τ` updated first, then a Newton–Raphson-centred independence MH conditional on `τ`.
   - The standardized log-target and its derivatives (§4.3b) are now checked numerically against `scipy.stats.invgamma` and finite differences.
   - Still not checked: the exact proposal variance and any truncation/rejection details inside stochvol. These are irrelevant for parity, which compares posteriors.
6. **Cost of the ν steps** (§4.3c). The arithmetic is verified, and NumPy throughput was measured (11 ns per float64 `log1p` on an M1 Pro). JAX/XLA-CPU throughput is **UNVERIFIED**. Conclusion: `collapse="full"` breaks the 2-minute budget; the partial step is material and must be benchmarked in the pilot.
7. **How often `o` saturates at `K = 20`** in COVID-era FRED-MD data: **UNVERIFIED**. Partial evidence for payrolls (SVO-t) in the §6 edge-case table suggests the cap did not bind there.
8. **Feasibility of the CCMM MATLAB code** as a forecast-level oracle: **UNVERIFIED** (no licence, confirmed; needs MATLAB).
9. **RESOLVED: van Dyk & Park (2008)** JASA 103(482), 790–796, doi:10.1198/016214508000000409. The van Dyk & Jiao (2015) JCGS citation is verified by DOI only (10.1080/10618600.2014.930041); volume and pages not checked.
10. **NEW: Geweke power against the "o between s and h" mutation** (§7.2): **UNVERIFIED**.
11. **NEW: April 2020 payrolls as a multiple of pre-COVID residual s.d.** (§6 edge cases): order of magnitude only, **UNVERIFIED**.

---

## 10. Sources

- Carriero, Clark, Marcellino & Mertens, "Addressing COVID-19 outliers in BVARs with stochastic volatility":
  - *Review of Economics and Statistics* 106(5), 1403–1417 (2024), doi:10.1162/rest_a_01213. §III.A (model; discrete-grid remark), §III.B (priors, fn. 7–9), §V.A, §V.C, Table 2. Open-access copy: https://cris.unibo.it/retrieve/84935685-6fb6-40ab-bcf0-3a25062471b7/rest_a_01213.pdf
  - Deutsche Bundesbank DP 13/2022: §3.1 (items 3–4, fn. 4–5), §3.2 (fn. 7–9), §4, §5.1, §5.3, Table 2. https://epflicht-hessen.hebis.de/ubffm/download/pdf/6249236
  - Online appendix (dated 9 Aug 2021) to Cleveland Fed WP 21-02R: appendix II (eqs. S.1–S.16, fn. 8) and III. https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2021/wp2102r_appendix.pdf
  - Landing page: https://www.clevelandfed.org/publications/working-paper/2021/wp-2102r-covid19-outliers-in-bvars-with-stochastic-volatility
  - Replication code, no licence, greps only:
    - `goVARSVO.m` l. 36 and 177–178;
    - `goVARSVOt.m` l. 36–37 and 177–179;
    - `mcmcVARSVO.m` l. 175–180 and 184;
    - `mcmcVARSVOt.m` l. 174–186 and 191.
    https://github.com/elmarmertens/CCMMoutlierVAR-code
- Stock & Watson, "Core inflation and trend inflation":
  - NBER WP 21282 (2015): §2 eqs. (1)–(3), p. 5–6, estimation paragraph (Omori mixture, DNP correction). https://www.nber.org/papers/w21282.pdf
  - Published as REStat 98(4), 770–784 (2016), doi:10.1162/REST_a_00608. Not accessed.
- Chan, J. C. C., "BVARs and Stochastic Volatility", arXiv:2310.14438 (2023), §3.2 and fn. 6. https://arxiv.org/abs/2310.14438
- Chan, J. C. C., "Comparing Stochastic Volatility Specifications for Large Bayesian VARs", arXiv:2208.13255 (2022), §6.3. https://arxiv.org/abs/2208.13255
- Chan, J. C. C. & Hsiao, C. Y. L. (2014), "Estimation of stochastic volatility models with heavy tails and serial dependence", in *Bayesian Inference in the Social Sciences* (Wiley), ch. 6, pp. 155–176, doi:10.1002/9781118771051.ch6. Read as CAMA WP 74/2013, §4–4.2: https://cama.crawford.anu.edu.au/sites/default/files/publication/cama_crawford_anu_edu_au/2013-11/74_2013_chan_hsiao.pdf (fetched via web.archive.org).
- Jacquier, Polson & Rossi (2004), "Bayesian analysis of stochastic volatility models with fat-tails and correlated errors", *J. Econometrics* 122(1), 185–212, doi:10.1016/j.jeconom.2003.09.001. Draft: §2.2.2 eq. (5), §2.2.3 eqs. (6)–(7), §5.1–5.2. https://people.bu.edu/jacquier/papers/jpr.je2004.pdf
- Kastner (2015), "Heavy-tailed innovations in the R package stochvol", vignette in stochvol 1.3.3: eqs. (1)–(3), §3.1–3.2 eqs. (4)–(7), §4. https://cran.r-project.org/src/contrib/Archive/stochvol/stochvol_1.3.3.tar.gz
- Hosszejni & Kastner (2021), JSS 100(12): §2.1 eq. (2), §2.2, §2.3. https://cran.r-project.org/web/packages/stochvol/vignettes/article2.pdf (https://doi.org/10.18637/jss.v100.i12)
- stochvol 3.2.9:
  - Reference manual (`specify_priors`, `svsample`/`svtsample`, `svsim`, `update_t_error`). https://cran.r-project.org/web/packages/stochvol/stochvol.pdf
  - NEWS.md (1.1.0, 3.0.0).
  - Source tarball, greps only (`R/simulation.R`, `src/single_update.cc`). https://cran.r-project.org/src/contrib/stochvol_3.2.9.tar.gz
- van Dyk, D. A. & Park, T. (2008), "Partially collapsed Gibbs samplers: Theory and methods", *JASA* 103(482), 790–796, doi:10.1198/016214508000000409.
- van Dyk, D. A. & Jiao, X. (2015), "Metropolis-Hastings within partially collapsed Gibbs samplers", *JCGS*, doi:10.1080/10618600.2014.930041. https://www.ma.imperial.ac.uk/~dvandyk/Research/15-jcgs-mhpcg.pdf
- bayesianVARs 0.1.8 reference manual (`specify_prior_sigma`). https://cran.r-project.org/web/packages/bayesianVARs/bayesianVARs.pdf
- bvartools 0.3.0 reference manual. https://cran.r-project.org/web/packages/bvartools/bvartools.pdf
- Del Negro & Primiceri (2015), Geweke (2004), Talts et al. (2018), Modrák et al. (2025), Omori et al. (2007), Kastner & Frühwirth-Schnatter (2014): as listed in SPEC §15.
- Geweke (1993), "Bayesian treatment of the independent Student-t linear model", *JAE* 8(S1), S19–S40, doi:10.1002/jae.3950080504. Cited for the exponential ν prior via JPR §2.2.2 and JSS §2.2; not read directly.
- Cited second-hand only:
  - Eraker, Jacquier & Polson (1998) for ν absorption at the lower bound;
  - Chib, Nardari & Shephard (2002) for the uniform ν prior;
  - Cúrdia, Del Negro & Greenwald (2014, *JAE* 29, 1031–1052) for t-SV step order.

---

## Cross-check log

Independent cross-check of this note (second agent, 2026-10-21). Every formula in §2, §4 and §5 was re-derived by hand. Constants were checked against the primary source and, where possible, a second one.

Scripts (NumPy/SciPy, written from the formulas, not from any package code), in `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/xc03_scripts/` (run with the repo venv Python, `-I`):
- `verify03.py` (all PASS): moments, Beta table, `o` conditionals vs quadrature, ν grid vs SciPy and JPR eq. (7), ν | τ targets and derivatives, the unit↔standardized mapping, RB leading-block LPD vs Monte Carlo, and predictive variance.
- `pcg_check.py`: exact-posterior check of the discrete-ν sweep, with mutations.
- `pcg_mh_check.py`: exact-posterior check of the continuous-ν MH variant.

### Verified without change

| Item | Verification |
| :--- | :--- |
| `o_t` scales the s.d.; grid `s_k = k`, `w_1 = 1−p`, `w_k = p/(K−1)`, `K = 20` | CCMM appendix II.iv eqs. (S.14)–(S.16), fn. 8; DP/REStat §3.1/§III.A; code `goVARSVO.m` l. 36, `mcmcVARSVO.m` l. 178–179; second source Chan arXiv 2208.13255 §6.3 |
| SVO prior `Beta(2.5, 10f−2.5)`; SVO-t prior `Beta(1, 10f−1)` | DP §3.2 = REStat §III.B text; code `goVARSVO.m` l. 177–178, `goVARSVOt.m` l. 177–179; "every 4 years" also in Chan 2208.13255 §6.3 |
| Beta moments table (0.0208/0.0130, 0.0083/0.0083, 0.0625/0.0378, 0.0250/0.0244) | `scipy.stats.beta` |
| `E[o] = 1+10p`, `E[o²] = 1+150p`, `Var(o) = 130p−100p²`; 4.125 / 2.665 / 2.25 | Exact sums |
| `o` conditional (Gaussian and τ-collapsed) | Brute-force quadrature over τ, to 1e-8 |
| `p | o ~ Beta(a+N₁, b+T−N₁)`; `τ | · ~ IG((ν+1)/2, (c + z²/o²)/2)` | Re-derived; JPR eq. (6); Kastner 2015 §3.1; Chan & Hsiao §4.1 |
| `t_{c,ν}` density and `scipy.stats.t(scale = sqrt(c/ν))` mapping | Quadrature of the scale mixture |
| ν collapsed target (§4.3) = JPR eq. (7) for `c = ν` | Numerically, to 1e-9 |
| DU{3..40} ν prior; bounds 30/60 sensitivity; ν ACF ≈ 0.6; escape probability | JPR §2.2.2, §5.1, §5.2; CCMM DP §3.2 / REStat §III.B; code l. 37 |
| CCMM step order (q before mixture states; o, p after the SV steps) and 7-component KSC table | DP/REStat fn. 9; code grep (`getKSC7values`) |
| stochvol 3.x: standardized `IG(ν/2,(ν−2)/2)`, `ν−2 ~ Exp`, `svtsample` `priornu = 0.1`, `specify_priors` defaults, `σ² ~ χ²₁ ⇔ σ ~ |N(0,1)|`, stationary `h₀` | Reference manual; NEWS 3.0.0; JSS §2.1–2.3 |
| Kastner (2015) target and derivatives eqs. (5)–(7), U(a,b) prior, warning on large ν | Vignette text; finite differences |
| Unit ↔ standardized mapping `h^u = h^s − log(ν/(ν−2))`, `τ^u = τ^s ν/(ν−2)` | Derivation; two-sample KS on simulated τ |
| RB leading-block LPD formula (§5.3) | Monte Carlo with 2·10⁶ draws (full vector and `j = 2`) |
| PCG validity of the §5.1 sweep | Exact posterior comparison; mutations detected |
| Predictive-variance identity (§5.3 item 3) | Monte Carlo with 2·10⁶ draws, < 1% |
| SW WP 21282 `{1,5,10}`, `(.975, 1/60, 1/120)`, Omori mixture plus DNP correction | WP text |
| bayesianVARs 0.1.8 and bvartools 0.3.0 have no t/outlier option | Manuals grepped for student/outlier/heavy/fat-tail: no hits |

### Corrections made

| # | Where | Was | Now | Source |
| :--- | :--- | :--- | :--- | :--- |
| 1 | §3.2 Chan & Hsiao row; §9 item 3 | UNVERIFIED, "believed `U(2, ν̄)`" | `ν ~ U(0, ν̄)` (lower bound 0), `ν̄ = 50` in example; uncollapsed `ν | λ` independence-chain MH with Newton–Raphson proposal; unit `IG(ν/2, ν/2)` | CAMA WP 74/2013 §4, eq. (18), §4.1–4.2 (Wayback copy) |
| 2 | §7.4 `svsim`; §9 item 4 | "which one 3.2.9 does is UNVERIFIED; do not use svsim" | 3.2.9 `svsim` is standardized (`tau <- 1/rgamma(shape=nu/2, rate=nu/2-1)`, `y <- exp(h/2)*sqrt(tau)*eps`); man-page Note is stale; svsim usable | stochvol 3.2.9 tarball `R/simulation.R` (grep) |
| 3 | §7.4 older versions | "stochvol ≤ 2.0.4" | 1.1.0–2.0.5 (last pre-3.0 release is 2.0.5) | NEWS.md 1.1.0, 2.0.5, 3.0.0 |
| 4 | §2.1, §7.1 test 8, §6 edge cases | "`K = 1` ⇒ SVO = SV" listed as a limit test | `K = 1` is invalid (`w_k = p/(K−1)` divides by zero): validate `outlier_max ≥ 2`, test that it raises | Derivation |
| 5 | §9 item 2; §0, §2.1, §3 citations | REStat text "UNVERIFIED, assumed identical" | Verified identical; REStat §III.A adds the explicit discrete-grid remark; REStat section numbers and DOI added | REStat 106(5) 1403–1417 (open-access copy) |
| 6 | §3.1 SW 2016 row; §9 item 1 | "U(2,10) hinted by code fallback" | Added second-hand evidence: Chan arXiv 2310.14438 fn. 6 (U(2,10)); Chan arXiv 2208.13255 §6.3 (discretised grid). Still UNVERIFIED from the primary source | Chan (2023) |
| 7 | §5.1 item 2; §9 item 9 | van Dyk & Park "from memory" | JASA 103(482), 790–796, doi:10.1198/016214508000000409 | Imperial PDF / search |
| 8 | §4.3(b) | No statement on MH within PCG | Added validity argument (van Dyk & Jiao 2015 caveat), exact-posterior numerical check, standardized derivatives (verified) | van Dyk & Jiao (2015); `pcg_mh_check.py`; `verify03.py` |
| 9 | §4.3(c), §6, §9 item 6 | "probably breaks the budget (UNVERIFIED)" | Arithmetic for the SPEC benchmark: full 1.8·10¹¹ vs partial 9.1·10⁹ `log1p`; NumPy measured at 11 ns/eval; `collapse="full"` breaks the 2-min budget; partial step must be benchmarked. Also made the `p | o` placement explicit in the fully collapsed order | SPEC §13 Phase 2; measurement |
| 10 | §2.3 | CCMM appendix III remark only | Added general-`K` moments, continuous-U(2,20) moments, and that the printed SVO-t formula (sum form, 1.28) does not reproduce the reported 1.56 (product form does) | Exact sums; CCMM appendix III |
| 11 | §6 diagnostics and edge cases | "Warn if ν piles up at 3 … consider outliers=True" | Report, do not warn: CCMM find posterior-mean ν = 3 for about half of the 16 variables *with* outliers (SVO-t) | DP §5.1 = REStat §V.A |
| 12 | §6 edge cases | Payroll "roughly 100 s.d."; cap binding UNVERIFIED | Magnitude marked UNVERIFIED; added CCMM Fig. 2 evidence (total ≈ 90 vs persistent ≈ 12 at peak) | DP §5.1 / REStat §V.A |
| 13 | §0 (sampler name) | "Collapsed griddy Gibbs" | "Collapsed exact discrete Gibbs" (griddy Gibbs means a grid approximation of a continuous conditional; here the prior is discrete) | Terminology |
| 14 | §3.2 CCMM row | Cited "fn. 7" for the sampler | fn. 9 is step order, fn. 7 is only the calibration remark; added that the code's ν likelihood constant matches the collapsed JPR eq. (7) | DP/REStat footnotes; code grep |
| 15 | §5.1 item 3 | Joint `(o, s)` table written with `ẽ` | Made the `k`-dependence of `ẽ` explicit (`log e² − 2 log k − log τ`, no Jacobian) | Derivation |
| 16 | §3.2 recommendation bullet | "With a discrete ν it changes only the meaning of `h`" | Equivalent only up to the `μ` prior (or the IG prior on `λ̄` for `Constant`), which is not shift-invariant | Derivation |
| 17 | §0 / §10 | Code line numbers `goVARSVO.m` l. 176–178 | l. 36 (`SVOmaxscale`) and 177–178 (`SVOalpha`, `SVObeta`) | Code grep |
| 18 | §3.1, §5.3 | Long verbatim quotes from CCMM | Paraphrased (copyright hygiene); added main-text DP §5.3 / REStat §V.C support for RB of transitory states | CCMM |
| 19 | §7.5 | Qualitative only | Added Table 2 reference values (SVO +232.59, SVO-t +225.33 over 2020:03–2021:02; full window +334.84 / +244.11) and posterior ranges of `p` | DP/REStat Table 2, §5.1 |
| 20 | §7.1 test 7 | "strictly higher ESS" | Fixed seed plus margin (ratio > 3); observed ratio ≈ 39 | `pcg_check.py` |
| 21 | §7.1 test 3 | IG variance check at all ν | The IG variance is infinite at `ν = 3` (`α = 2`); use mean/KS or `log τ` there | IG moments |
| 22 | §4.1 | "`1/k` Jacobian" | It is the Gaussian normalising factor, not a Jacobian (wording only) | Derivation |
| 23 | §0 table | Pointer "§3.3 below" (no such section) | "§5.1 below" | Internal consistency |
