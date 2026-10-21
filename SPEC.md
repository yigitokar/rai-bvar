# `rai-bvar`: Technical Architecture & Engineering Specification

**Version:** 2.1 (revision of v1) · **Status:** Draft; decisions D1–D5 settled (§14) · **Date:** 2026-10-21

---

## 0. Revision notes (v1 → v2)

| Area | v1 | v2 |
| :--- | :--- | :--- |
| Model API | One class per model (`linear.py`, `linear_sv.py`, `tvp_bvar.py`, …) | Composition: `Mean × Volatility × Prior`. Ablation is a factorial grid, not a ladder (§4) |
| Equation-by-equation sampling | Carriero, Clark & Marcellino (2019) | The 2019 algorithm conditions each equation's coefficients on the first `j` equations only, ignoring the information about them in later equations, so it samples from the wrong distribution (Bognanni 2022). v2 uses the recursive structural form (equations independent by construction) or the corrected algorithm of Carriero, Chan, Clark & Marcellino (2022). Ordering dependence is documented and checked (§3 R8, §6.6) |
| Conjugate Minnesota | Tunes λ1, λ2, λ3 by marginal likelihood | The conjugate NIW prior cannot express cross-variable shrinkage, so there is no λ2 there. Hyperparameters follow GLP (2015), with both empirical-Bayes (mode) and fully hierarchical (MH) modes (§4.6) |
| Predictive contract | Draw array `(n_draws, horizon, n_vars)` | Joint path draws **plus** per-draw conditional Gaussian moments, so the log predictive density is computed by Rao-Blackwellisation, not by KDE (§7) |
| In-sample screening | PSIS-LOO, WAIC | Removed: both are wrong for time series and unreliable with latent volatilities. Replaced by PSIS-LFO-CV with state-integrated likelihoods (later phase) and exact log-ML for conjugate models (§9.4) |
| Forecast tests | DM / Amisano-Giacomini p-values | Formally valid only with rolling windows (Giacomini-White framework). HAC + HLN correction. Model Confidence Set for the ablation grid (§9.3) |
| Backends | NumPy/SciPy, JAX/NumPyro, PyMC | NumPy/SciPy for the analytic engine. Hand-written JAX Gibbs for MCMC. No probabilistic programming language in core (§11) |
| Reference packages | "Port/adapt" R/C++ source | Clean-room reimplementation from papers (most R references are GPL). R packages serve as numerical oracles only (§2.1) |
| Missing in v1 | — | Validation strategy (Geweke tests, SBC, R parity, verdict-engine size/power on simulated DGPs), COVID/outlier handling, transform-aware Minnesota prior, look-ahead-free backtests, conditional forecasts, phased roadmap |
| Positioning | "Python lacks a unified SOTA BVAR ecosystem" | Fragmented, not absent: several Python BVAR projects appeared in 2026. Differentiation is the ablation/verdict engine, the factorial composition and the validation suite (§1.4) |
| Errors fixed | `bayesm` listed as a Primiceri implementation; sign restrictions attributed to `bsvars`; "BART-VAR" listed as a package; "standard IRFs are invalid in TVP models" | Corrected (§2.2, §6.4) |

---

## 1. Vision & scope

### 1.1 Value proposition

`rai-bvar` is a Python library for Bayesian vector autoregressions built around one question:

> **Does added model complexity improve out-of-sample density forecasts once time-varying volatility is accounted for?**

Every model is a composition of three independent components:

- **Mean:** how the conditional mean depends on lagged data (linear, BART, TVP, threshold / Markov-switching).
- **Volatility:** how the error covariance evolves (constant, stochastic volatility, SV with outliers or fat tails).
- **Prior:** how the mean coefficients are shrunk (Minnesota, global-local).

An ablation is a factorial grid over these components. One backtest engine and one predictive contract evaluate every model in the grid.

### 1.2 Target scale

| | Analytic (conjugate) engine | MCMC engine |
| :--- | :--- | :--- |
| Variables `n` | up to ~100+ | up to ~30 (design target `n = 20`) |
| Lags `p` | up to 13 (monthly) | up to 13 |
| Observations `T` | 100–800 | 100–800 |
| Frequency | monthly, quarterly | monthly, quarterly |

### 1.3 Non-goals for v1.0

- Structural identification beyond the recursive (Cholesky) scheme: sign, zero and narrative restrictions, proxy SVARs, identification through heteroskedasticity (Phase 6, possibly a separate package).
- Mixed-frequency data, missing values inside the sample, ragged edges.
- Real-time data vintages. The data layer must not preclude them (§5.5).
- Panel, global and factor-augmented VARs.
- Frequentist VAR estimation (use `statsmodels`).

### 1.4 Python landscape (October 2026)

The Python BVAR ecosystem is **fragmented, not empty**:

| Project | Licence | Coverage | Gap relative to `rai-bvar` |
| :--- | :--- | :--- | :--- |
| `bvar` (Bank of England, PyPI, first release Aug 2026) | MIT | Conjugate Minnesota, SOC/SUR dummies, GLP-style hyperparameter optimisation, COVID dummies, conditional forecasts | No MH over hyperparameters, no SV, no global-local priors, no nonlinear models |
| Joshua Chan's `bvar-toolkit-python` (GitHub, started Oct 2026) | MIT | Port of his MATLAB toolkit (SV, shrinkage priors, marginal likelihoods); about a quarter of functions ported so far | Early stage; no BART; no backtest/verdict engine |
| `bvhar` Python port (GitHub only) | MIT | Minnesota, SSVS, HS, NG, DL priors with SV (C++ core) | No hierarchical GLP, no nonlinear models, not on PyPI |
| `pymc-extras` `BayesianVARMAX` | Apache-2.0 | State-space VARMAX in PyMC | Hand-set priors; no Minnesota/GLP, no SV |
| `impulso`, `srvar-toolkit` | MIT | Small, specialised (PyMC/NUTS Minnesota; shadow-rate VAR with SV) | Experimental / narrow |
| `statsmodels` | BSD-3 | Frequentist VAR, SVAR, VECM, local projections | No Bayesian VARs |

No maintained package combines hierarchical GLP with MH, global-local shrinkage, SV / factor SV and BART behind one interface, and none has a model-comparison engine built around fair ablation. **That engine, the factorial composition and the validation suite are `rai-bvar`'s differentiation, not "the first Python BVAR".** The MIT-licensed projects above are potential collaborators, Python-side oracles (§10.4) and, with attribution, sources of reusable code.

---

## 2. Reference implementations & licensing

### 2.1 Licensing policy (clean-room rule)

Every R reference package in §2.2 is GPL-licensed (GPL-3, GPL ≥ 2 or GPL ≥ 3). Translating their source into `rai-bvar` would make `rai-bvar` a derivative work under the GPL. MIT- and Apache-licensed sources (`bartz`, `bvhar`, Chan's toolkit, the Bank of England `bvar`, `mf-bavart`) may be reused with attribution. Unless the project deliberately chooses the GPL:

- Algorithms are implemented **from the papers** in §15, not translated from R/C++ source.
- R packages are used as **numerical oracles**: we run them to generate test fixtures (§10.4). Their outputs are data, not code.
- Where a paper is ambiguous, the implementation decision and its reasoning are recorded in the module docstring.
- Licence of `rai-bvar`: **Apache-2.0** (decision D1).
- The PR template includes a checkbox: "No code in this PR was translated from GPL sources."

### 2.2 Reference packages (numerical oracles)

| Package | Authors | Licence | Oracle for |
| :--- | :--- | :--- | :--- |
| `BVAR` | Kuschnig & Vashold (2021) | GPL-3 | Hierarchical Minnesota prior (GLP 2015), SOC/SUR dummy observations, MH over hyperparameters → Phase 1 analytic engine. (Internally it uses cross-products, eigenvalues and Cholesky, not QR; compare outputs, not numerics) |
| `stochvol` | Hosszejni & Kastner | GPL ≥ 2 | Univariate SV: AWOL (tridiagonal Cholesky) sampling of log-variances, ASIS, Omori et al. 10-component mixture, Student-t errors, leverage → SV block |
| `bayesianVARs` | Gruber, Haan & Kastner | GPL ≥ 3 | Global-local priors (`HS`, `R2D2`, `NG`, `DL`, `SSVS`, `HMP`) with SV in a Cholesky structure (using the corrected CCCM triangular algorithm) or a factor structure → Phase 2 reduced-form sampler, Phase 3 shrinkage |
| `factorstochvol` | Kastner | GPL ≥ 2 | Factor SV with interweaving → Phase 5 `FactorSV` |
| `bvars` | Woźniak | GPL ≥ 3 | Large BVAR with Minnesota prior, common SV and t errors → `CommonSV` (Phase 3) |
| `bvarsv` | Krüger | GPL ≥ 2 | Primiceri (2005) TVP-VAR-SV with the Del Negro & Primiceri (2015) corrigendum → Phase 5 TVP |
| `bvartools` | Mohr | GPL ≥ 2 | Modular Gibbs samplers: Durbin-Koopman simulation smoother for TVP, Chan-Jeliazkov precision sampler, Omori-mixture SV (architectural reference) |
| `bsvars` / `bsvarSIGNs` | Woźniak / Wang & Woźniak | GPL ≥ 3 | Identification through heteroskedasticity, non-normality and exclusion restrictions (`bsvars`); sign, zero and narrative restrictions, without proxy/IV identification (`bsvarSIGNs`) → Phase 6 |
| BART-VAR code: `EoghanONeill/BAVART` (Huber & Rossini 2022), `mpfarrho/mf-bavart` | — | GPL-3 / Apache-2.0 | BART-VAR with SV → Phase 4. There is no CRAN package and no public replication repository for Clark et al. (2023); pin a commit. These are **not posterior oracles**: Huber & Rossini use a reduced-form, CCM-2019-style residual adjustment (approximate), and the main Clark et al. specification uses factor SV. Parity is checked on forecast performance only |

Not a reference: `bayesm` (Rossi's Bayesian marketing/micro-econometrics package; unrelated to Primiceri 2005).

---

## 3. Core design rules (read before coding)

**R1. Composition over inheritance.** A model is `BVAR(mean=..., vol=..., lags=...)`. There are no model-specific classes such as `LinearBVAR_SV`. Adding a component must not require touching other components.

**R2. Volatility first (fair ablation).** Much of the apparent gain from nonlinear or time-varying mean specifications comes from time-varying volatility and fat tails (Sims & Zha 2006; Clark & Ravazzolo 2015). Therefore:

- The effect of a mean specification is measured **holding the volatility specification fixed**: `BART × SV` vs `Linear × SV`, never `BART × SV` vs `Linear × Constant`.
- The default baseline for any non-linearity claim is `Linear × SV`. For samples that include 2020, it is `Linear × SV(outliers=True)`.
- `compare()` only issues a verdict between models that differ along exactly one axis (§9.5).

**R3. One predictive contract.** Analytic and MCMC models return the same `ForecastResult` (§7). Evaluation and structural analysis never depend on how a model was estimated.

**R4. Numerical rules.** No explicit inverses, float64 everywhere (JAX defaults to float32; one documented exception inside `bartz`), precision-based samplers, log-space densities (§8).

**R5. Clean-room implementation** (§2.1).

**R6. No sampler is merged without Geweke and SBC tests passing** (§10.3). The CCM (2019) episode, a published and widely used sampler that targeted the wrong posterior, is the reason for this rule.

**R7. One MCMC backend:** JAX with hand-written Gibbs kernels. No PPL (PyMC, NumPyro) in core.

**R8. Ordering dependence is surfaced, not hidden.** Any model that uses a triangular factorisation of `Σ_t` depends on the variable ordering (Arias, Rubio-Ramírez & Shin 2023; Chan, Koop & Yu 2024). Every result records the ordering, and the library ships an ordering-sensitivity check (§6.6).

**R9. No look-ahead.** Everything data-dependent (prior scales `ψ`, standardisation, initial-observation means for dummies, hyperparameter optimisation) is recomputed inside each backtest window from data available at the forecast origin.

---

## 4. Model framework

### 4.1 Common representation

Let `y_t ∈ ℝⁿ` and `x_t = (d_t', y_{t-1}', …, y_{t-p}')' ∈ ℝᵏ`, where `d_t` holds deterministic terms (constant, trend, exogenous regressors) and `k = np + n_d`.

All MCMC models are written in **recursive structural form**:

$$
y_{i,t} = -\sum_{j<i} a_{ij}\, y_{j,t} + g_i(x_t;\theta_i) + \sqrt{\lambda_{i,t}}\;\varepsilon_{i,t},
\qquad \varepsilon_{i,t} \overset{iid}{\sim} N(0,1).
$$

Equivalently `A y_t = g(x_t) + Λ_t^{1/2} ε_t`, with `A` unit lower triangular and `Λ_t = diag(λ_{1,t}, …, λ_{n,t})`. The reduced form is

$$
y_t = A^{-1} g(x_t) + u_t, \qquad u_t \sim N(0,\Sigma_t), \qquad \Sigma_t = A^{-1}\Lambda_t A^{-1\prime}.
$$

For a linear mean, `g(x_t) = Γ x_t` and the reduced-form coefficients are `B' = A⁻¹Γ`.

**Why this form.** Conditional on the volatilities, and with priors independent across equations, the `n` equations have independent posteriors. Equation `i` is a heteroskedastic univariate regression with `k + i − 1` regressors (`x_t` plus `y_{<i,t}`). One sweep then costs `O(n k³) = O(n⁴p³)` instead of `O((nk)³) = O(n⁶p³)` for joint sampling, and the equations run in parallel under `jax.vmap`. This is the structural-form route to the CCM speed-up, and it is correct by construction because the equations really are conditionally independent.

**Reduced-form priors.** Users who want the prior on the reduced-form `B` (classic Minnesota with cross-variable shrinkage) break this independence: in `A y_t`, equation `i > j` contains `a_{ij} x_t'β_j`, so equation `j`'s coefficients appear in every later equation. That case uses the corrected triangular algorithm of Carriero, Chan, Clark & Marcellino (2022): `Linear(prior=Minnesota(form="reduced"))`. It is a Gibbs sampler over `β_j | y, Σ_{1:T}, β_{−j}` in which step `j` uses equations `j..n`. It keeps the `O(n⁴)` cost, but at `n = 20` it is roughly 3× slower than the flawed 2019 version and its draws are more autocorrelated, while still about 13× faster than system-wide sampling. It assumes coefficient priors independent across equations.

**Structural-form Minnesota (default for MCMC, decision D3).** `form="structural"` places a weakly informative Gaussian prior on `a_i` and the Minnesota-type prior on `γ_i`. To keep the familiar reduced-form prior **mean**, `γ_i` is centred conditionally on `a_i`. Since `Γ = AB'`, a reduced-form prior mean `E[B] = M` (`k × n`; e.g. random walk for level series) maps to

$$
\mathbb{E}[\gamma_i \mid a_i] = M\,\tilde a_i, \qquad \tilde a_i = (a_{i1}, \dots, a_{i,i-1}, 1, 0, \dots, 0)' \;(\text{row } i \text{ of } A).
$$

This is linear in `a_i`, so the joint prior on `(a_i, γ_i)` is Gaussian and the linear mean block still draws both in one Gaussian step. Both forms share the prior mean on `B`; their prior **covariances** differ, because the structural form places them on structural coefficients. They are therefore different priors: the documentation must say so, and parity tests must use matching forms. The same structural treatment of `a_i` is used by the BART and TVP mean blocks, so `Linear × SV` vs `BART × SV` differs only in `g_i`.

**Covariance structures.** The recursive structure above (`A`, `Λ_t`) is the default. Factor stochastic volatility, `Σ_t = L Ω_t L' + H_t` (Kastner 2019; the main specification of Clark et al. 2023), is the Phase 5 alternative: given the factors, equations are again conditionally independent, and no triangular factorisation of `Σ_t` is involved. Mean blocks never see which structure is in use; they only receive a per-equation `target` and `weights` (§4.2).

### 4.2 Component interfaces

Sketch only; the final signatures are settled in Phase 0.

```python
class MeanBlock(Protocol):
    """g_i(x_t; θ_i) for one equation, together with its prior."""
    def init_state(self, key, eq: EquationData) -> MeanState: ...
    def sample(self, key, state: MeanState, eq: EquationData,
               target: Array,    # y_i, adjusted by the covariance structure, shape (T,)
                                 #   (recursive: y_{<i,t} enter as extra regressors; factor: y_i − L_i f_t)
               weights: Array,   # 1 / λ_{i,t}, shape (T,)
               glob: GlobalState) -> MeanState: ...
    def predict(self, state: MeanState, x: Array) -> Array: ...   # g_i(x) at new x
    def log_prior(self, state: MeanState, glob: GlobalState) -> Array: ...

class VolBlock(Protocol):
    """λ_{i,1:T} for one equation."""
    def init_state(self, key, T: int) -> VolState: ...
    def sample(self, key, state: VolState, resid: Array) -> VolState: ...   # resid_t = √λ_t ε_t
    def variances(self, state: VolState) -> Array: ...                     # λ_{i,1:T}
    def simulate_ahead(self, key, state: VolState, H: int) -> Array: ...   # λ_{i,T+1:T+H}

class GlobalBlock(Protocol):
    """Parameters shared across equations: hyperparameters, global shrinkage
    scales, regime indicators, reduced-form coupling."""
    def sample(self, key, glob: GlobalState, eqs: EquationStates) -> GlobalState: ...
```

The `weights` argument is how volatility reaches the mean block: every mean block must support heteroskedastic (weighted) likelihoods. This is a hard requirement for the BART backend (§13, Phase 4).

### 4.3 Gibbs sweep

```text
for each iteration:
    glob ← GlobalBlock.sample(... | all equation states)          # hyperparameters, regimes, global scales
    vmap over equations i = 1..n:                                  # independent given glob
        (θ_i, a_i) ← MeanBlock.sample(... | λ_{i,1:T}, glob)       # a_i: Gaussian regression on y_{<i,t}
        λ_{i,1:T}  ← VolBlock.sample(... | residuals of eq. i)     # mixture indicators → log-vols → SV params (ASIS)
    store (thinned) draws
```

- For a linear mean, `a_i` and `γ_i` are drawn jointly in one Gaussian step. For nonlinear `g_i`, `a_i` gets its own Gaussian step conditional on `g_i`.
- Within the SV block, the mixture indicators are drawn **after** the mean / contemporaneous block and immediately before the log-volatilities. This is the corrected step order of Del Negro & Primiceri (2015): `Σ_{1:T} | θ, s` → `θ | Σ_{1:T}` → `s | Σ_{1:T}, θ`. No other block may be inserted between the indicator draw and the log-volatility draw.
- The `Linear(form="reduced")` sampler is implemented as a global block, because its equations are coupled.

### 4.4 Component catalogue

**Mean blocks**

| Block | `g_i(x_t)` | Prior(s) | Phase |
| :--- | :--- | :--- | :--- |
| `Linear` | `x_t'γ_i` | Minnesota (structural, reduced or conjugate form); global-local | 1–3 |
| `BART` | `Σ_{m=1}^{M} tree_m(x_t)` | Chipman, George & McCulloch (2010) tree prior, leaf scale calibrated to data | 4 |
| `TVP` | `x_t'γ_{i,t}`, random-walk states, non-centred | Shrinkage on state variances (Huber, Koop & Onorante 2021) | 5 |
| `Threshold` | `x_t'γ_i^{(s_t)}`, `s_t = 1{z_{t-d} > τ}` | Minnesota per regime; `τ, d` sampled in a global block | 5 |
| `MarkovSwitching` | `x_t'γ_i^{(s_t)}`, latent Markov `s_t` | Minnesota per regime; identification constraint against label switching | 5 |

**Volatility blocks**

| Block | `λ_{i,t}` | Phase |
| :--- | :--- | :--- |
| `Constant` | `λ_i` constant, inverse-gamma prior. `Constant(prior="iw")` marks the conjugate NIW case (§4.6) | 1–2 |
| `SV` | `log λ_{i,t} = μ_i + φ_i (log λ_{i,t-1} − μ_i) + σ_i η_{i,t}`; random walk as the special case `φ_i = 1` | 2 |
| `SV(outliers=True)` | SV plus a mixture outlier component (Carriero, Clark, Marcellino & Mertens) | 2 |
| `SV(tails="t")` | SV with Student-t errors as a scale mixture, `ν` estimated | 2 |
| `CommonSV` | `Σ_t = f_t Σ`, a single common volatility factor (Carriero, Clark & Marcellino 2016); keeps the conjugate structure, so it is a fast SV screen for large `n` | 3 |
| `FactorSV` | Factor stochastic volatility (Kastner 2019) | 5 |
| Order-invariant SV | Chan, Koop & Yu (2024) | 5 |

**Priors**

| Prior | Parameters | Phase |
| :--- | :--- | :--- |
| `Minnesota` | `lambda_` (overall tightness), `alpha` (lag decay; fixed at 2, i.e. `1/s²`, as in GLP, or optionally a hyperparameter), `cross` (cross-variable tightness; must be 1 in conjugate form), `psi` (scales; default: residual s.d. of univariate AR(p)), `soc` (μ), `sur` (δ), `mean` (prior mean of own first lag; default from transform codes, §5.3), `form` (`"conjugate"`, `"structural"`, `"reduced"`), `hierarchy` (`"fixed"`, `"eb"`, `"mh"`) | 1–2 |
| `Horseshoe`, `NormalGamma`, `DirichletLaplace`, `R2D2`, `SSVS` | Global-local shrinkage with coefficient groups (own lags, cross lags, deterministics, contemporaneous) | 3 |
| `AsymmetricConjugate` | Chan (2022): cross-variable shrinkage with conjugate-like speed | 3 |

### 4.5 Engine resolution

The engine is chosen by the composition, not by the user. `fit.engine` reports the choice.

| Composition | Engine |
| :--- | :--- |
| `Linear(prior=Minnesota(form="conjugate"))` × `Constant(prior="iw")` | Analytic (§4.6) |
| `Linear(prior=AsymmetricConjugate(...))` × `Constant` | Analytic (Phase 3). Chan (2022) works in exactly our recursive structural form with independent normal-inverse-gamma priors per equation, so the posterior factorises and iid draws are exact. It is the analytic twin of the MCMC model `Linear(form="structural") × Constant` |
| Anything else | MCMC (§4.3) |

Invalid combinations fail at construction time with an actionable message, for example `Minnesota(form="conjugate", cross=0.5)` → "cross-variable shrinkage requires form='structural' or 'reduced'".

### 4.6 Analytic engine (fast path)

Conjugate Normal-Inverse-Wishart: `B | Σ ~ MN(B̄, Σ ⊗ Ω̄)`, `Σ ~ IW(S̄, ν̄)`. The Kronecker structure forces the same prior tightness on own and cross lags (Kadiyala & Karlsson 1997), so `cross = 1`.

Implementation:

- The prior is expressed as dummy observations `(Y_d, X_d)` (Minnesota, SOC, SUR) stacked on the data: `Y* = [Y_d; Y]`, `X* = [X_d; X]`. SOC/SUR dummies use `ȳ_0`, the mean of the first `p` observations of the window.
- Thin QR: `X* = QR`. Posterior mean `B̄ = R⁻¹Q'Y*` via a triangular solve. `Ω̄ = (R'R)⁻¹` is never formed.
- Draws: `Σ ~ IW(S̄, ν̄)` via the Bartlett decomposition, then `B = B̄ + R⁻¹ Z C'` with `Σ = CC'` and `Z` a `k × n` standard normal matrix (two triangular solves).
- Closed-form log marginal likelihood (matrix-t), with all log-determinants taken from triangular factors (`log|X*'X*| = 2 Σ log|R_ii|`).
- **Hierarchy (GLP 2015).** Hyperparameters `h = (λ, μ, δ, ψ, [α])`. Default hyperpriors follow GLP: Gamma on `λ`, `μ`, `δ` with modes 0.2 / 1 / 1 and standard deviations 0.4 / 1 / 1; Inverse-Gamma(0.02², 0.02²) on `ψ_j/(d − n − 1)`; IW degrees of freedom `d = n + 2`.
  - `hierarchy="eb"`: maximise `log p(Y | h) + log p(h)` with L-BFGS-B on log-transformed `h`.
  - `hierarchy="mh"`: random-walk Metropolis on `h`, proposal covariance from the inverse Hessian at the mode; then draw `(B, Σ) | h, Y` directly.
- **COVID scaling (Lenza & Primiceri 2022).** `y_t = B'x_t + s_t ε_t`, so the residual covariance is `s_t² Σ`, with one scalar common to all variables. `s_t = 1` before `t*`; `s_{t*}, s_{t*+1}, s_{t*+2}` are free (March, April, May 2020 in monthly data); afterwards the excess over one decays geometrically, `s_{t*+j} = 1 + (s_{t*+2} − 1) ρ^{j−2}`. Rows of `Y` and `X` are divided by `s_t` (GLS), which preserves conjugacy. `(s_{t*}, s_{t*+1}, s_{t*+2}, ρ)` join `h`, with Pareto(1, 1) priors on the scales and a Beta prior (mode 0.8, s.d. 0.2) on `ρ`.
- Given `h`, draws are iid. Only the MH chain over `h` needs convergence diagnostics.

---

## 5. Data layer

### 5.1 Input

A pandas `DataFrame` with a regular `DatetimeIndex` or `PeriodIndex`. Column order is the variable ordering of the recursive form (R8). Lags and deterministic terms belong to the model (`BVAR(..., lags=4, deterministic="c")`), not to the data, so lag length can be part of an ablation.

### 5.2 Transforms

A registry of FRED-MD transformation codes (McCracken & Ng 2016):

| tcode | Transform |
| :--- | :--- |
| 1 | level |
| 2 | Δx |
| 3 | Δ²x |
| 4 | log x |
| 5 | Δ log x |
| 6 | Δ² log x |
| 7 | Δ(x_t / x_{t−1} − 1) |

Each transform has a forward map and a **pathwise inverse**. Inverse transforms are applied to each joint path using the last observed levels, never to quantiles or means: nonlinear transforms do not commute with expectations. Optional annualisation is a display concern only.

### 5.3 Transform-aware Minnesota prior mean

The prior mean of the own first lag is tied to the transform: 1 for persistent level series (tcodes 1 and 4 by default), 0 for differenced series (tcodes 2, 3, 5, 6, 7). Users override per variable with `Minnesota(mean={...})`. A level series with prior mean 0, or a growth rate with prior mean 1, is a common silent error; `fit()` warns when the user-specified means contradict the tcodes.

### 5.4 Deterministic terms

`deterministic="c"` (default), `"ct"` or `"n"`. Exogenous regressors require future values (or a scenario) at forecast time; forecasting raises if they are missing.

### 5.5 Vintages

`VARData` carries an optional `vintage` attribute. The backtest accepts either a single `VARData` or a callable `data_at(origin) -> VARData`, so real-time vintages can be added later without changing the API.

### 5.6 Outliers and COVID

`outliers=` on the model:

- `"lenza-primiceri"`: residual-variance scaling (analytic engine and constant-volatility MCMC models).
- `SV(outliers=True)`: outlier component inside the volatility block.
- `"exclude"`: masks periods out of the likelihood. A simple fallback, documented as ad hoc.

The default is none, but `fit()` warns if the estimation sample includes March–December 2020 and no option is selected.

### 5.7 Missing values

v1 raises on any NaN inside the estimation sample. Leading NaNs are trimmed with a warning.

### 5.8 Reference datasets (decision D4)

- **FRED-QD** (quarterly) is the default for documentation, examples and replication; **FRED-MD** (monthly) covers monthly models and the COVID-period tests.
- Three standard sizes, defined in Phase 0: **small** (3 variables: real GDP growth, inflation, policy rate), **medium** (about 7) and **large** (about 20).
- Datasets are **not bundled** in the package. Some FRED-MD/QD series are third-party copyrighted (e.g. S&P indices). `rb.datasets.load("fred-qd", vintage="2026-09")` downloads a pinned vintage on demand, verifies a SHA-256 checksum and caches it locally.
- **Committed test fixtures** (§10.4) use only series whose source is a U.S. federal agency (BEA, BLS, Federal Reserve Board). CI never touches the network.
- **Publication gaps.** Real vintages contain missing values inside a series' span (e.g. October 2025 for BLS series after the federal shutdown). `load(..., fill_gaps=True)` (default) interpolates them, in logs for log-transformed series, and warns listing every filled value; `fill_gaps=False` keeps them missing, and `VARData` then raises (§5.7).

---

## 6. Forecasting & structural analysis

### 6.1 Forecast simulation

For each posterior draw `s`:

1. Simulate future volatilities `λ_{T+1:T+H}` (`VolBlock.simulate_ahead`).
2. For `h = 1..H`: build `x_{T+h}` from the simulated path, compute the conditional mean `m_{s,h} = A⁻¹ g(x_{T+h})` and the factor `L_{s,h} = A⁻¹ Λ_{T+h}^{1/2}`, then draw `y_{T+h} = m_{s,h} + L_{s,h} ε`.

`L_{s,h}` is lower triangular, so it is the Cholesky factor of the conditional covariance at no extra cost. Both `m` and `L` are stored (§7). Every model in scope is conditionally Gaussian given the parameters, the simulated path and the volatilities (Student-t errors are conditionally Gaussian given the latent scale draw).

### 6.2 Conditional forecasts (Phase 3)

Hard conditions on the future paths of a subset of variables (Waggoner & Zha 1999), for linear mean models.

### 6.3 IRFs and FEVD (linear mean)

Recursive identification is native to the representation (`A`, `Λ`). For SV models the impact matrix `A⁻¹Λ_t^{1/2}` depends on `t`. Responses to unit structural shocks are time-invariant in shape; responses to one-s.d. shocks are reported at user-chosen dates.

### 6.4 TVP models

Default: time-varying IRFs with coefficients frozen at date `t` (the Primiceri 2005 convention). GIRFs (§6.5) are available as the more complete alternative that accounts for future parameter drift. Standard IRFs are not "invalid" for TVP models; they answer a narrower question.

### 6.5 Generalised impulse responses (nonlinear models, Phase 4)

$$
\mathrm{GIRF}(h,\delta,\Omega_{t-1}) = \mathbb{E}[y_{t+h} \mid \varepsilon_{j,t} = \delta,\ \Omega_{t-1}] - \mathbb{E}[y_{t+h} \mid \Omega_{t-1}]
$$

(Koop, Pesaran & Potter 1996.)

- The shock is defined on the recursive structural innovation `ε_{j,t}`; `δ` is in standard-deviation units, scaled by `λ_{j,t}^{1/2}`.
- Monte Carlo over posterior draws (subsample of size `S_g`), histories `t`, and `R` replications. Baseline and shocked paths use **common random numbers** (identical future shocks). Without them, Monte Carlo noise swamps the response.
- Output dims: `(draw, history, horizon, variable)`. Helpers average over history sets (recession vs expansion, high vs low uncertainty) and compare `+δ` vs `−δ` and 1 vs 3 s.d. (sign and size asymmetry).
- Cost scales with `S_g × |histories| × R × H`. Defaults: `S_g = 500`, `R = 100`; documented with timings.

### 6.6 Ordering sensitivity

`rb.ordering_sensitivity(model, data, n_perm=10, pinned=[...])` refits under random permutations of the variable ordering, keeping user-pinned leading variables fixed, and reports the dispersion of forecast LPD and quantiles. This is the operational answer to R8 until order-invariant models land in Phase 5.

---

## 7. Predictive contract

`ForecastResult` is an xarray-based container:

| Field | Dims | Meaning |
| :--- | :--- | :--- |
| `paths` | `(chain, draw, horizon, variable)` | Joint simulated paths on the model (transformed) scale. Each `(chain, draw)` slice is one joint path generated from one parameter draw |
| `paths_level` | same | Pathwise inverse-transformed paths on the original scale (§5.2) |
| `cond_mean` | `(chain, draw, horizon, variable)` | `m_{s,h}`: conditional mean given the draw, the simulated path up to `T+h−1` and `λ_{T+h}` |
| `cond_chol` | `(chain, draw, horizon, variable, variable_)` | `L_{s,h}`: lower-triangular factor of the conditional covariance |
| attrs | — | Forecast origin, horizon dates, variable ordering, transforms, model spec hash, engine, seed, package version |

**Invariants**, tested for every registered composition:

1. `paths[s, h]` is distributed as `N(cond_mean[s, h], cond_chol[s, h] cond_chol[s, h]')` (simulation-consistency test).
2. Draws are joint across horizons and variables, never assembled from marginals.
3. Stacking `(chain, draw)` into `sample` gives the v1 shape `(n_draws, horizon, n_vars)`.

**Methods**

- `log_pred_density(y_obs, horizon, variables=None)`: the Rao-Blackwellised estimator
  `log p̂(y_{T+h} | Y) = logsumexp_s log N(y; m_{s,h}, L_{s,h}L_{s,h}') − log S`.
  Variable subsets marginalise the Gaussian (select rows of `m`, form the sub-covariance). This is the `h`-step marginal predictive density of `y_{T+h}`, the standard target in the forecasting literature.
- `quantiles(q)`, `mean()`, `median()`, `to_dataframe()`.

**Memory.** `cond_chol` has size `S × H × n²`; for `S = 4000`, `H = 12`, `n = 20` that is about 150 MB in float64. `store_moments="lazy"` recomputes moments from posterior draws on demand, chunked over draws.

---

## 8. Numerical rules

- **No explicit inverses.** Use `cho_factor` / `cho_solve`, `solve_triangular` and QR.
- **Gaussian draws from precision form.** For `β ~ N(K⁻¹b, K⁻¹)`: `L = chol(K)`, `μ = cho_solve(L, b)`, `β = μ + solve_triangular(L', z)` with `z ~ N(0, I)`.
- **SV states.** Log-volatility paths have a banded (tridiagonal) precision, so they are sampled in `O(T)` with a precision sampler (Chan & Jeliazkov 2009; "all without a loop", McCausland, Miller & Pelletier 2011). No Kalman-filter loops for SV.
- **Mixture approximation.** `log ε²` is approximated by the 10-component mixture of Omori, Chib, Shephard & Nakajima (2007), an extension of Kim, Shephard & Chib (1998). The offset added before taking logs is documented and tested.
- **ASIS** (Kastner & Frühwirth-Schnatter 2014) interweaves the centred and non-centred parametrisations of the SV parameters.
- **float64.** The MCMC engine calls `jax.config.update("jax_enable_x64", True)` on import, and tests assert dtypes. JAX defaults to float32. **One documented exception:** `bartz` computes and samples leaf full conditionals in float32 even with x64 enabled. This is confined to the BART block, and its Geweke and SBC tests must pass with it.
- **Log space.** Densities are evaluated in log space; mixtures use `logsumexp`.
- **Positive-definiteness by construction.** Covariances come from triangular factors with positive diagonals. Adding silent jitter is forbidden; a failed Cholesky raises with diagnostics (condition number, offending block).
- **Randomness.** JAX keys are split deterministically per (chain, iteration, block, equation); the analytic engine uses a NumPy `Generator`. The same seed gives bitwise-identical results on the same platform and versions.

---

## 9. Evaluation & verdict engine

### 9.1 Backtest

- **Origins:** `rb.Origins(start, end, step)`. **Windows:** `rb.Expanding()` or `rb.Rolling(size)`.
- **At each origin:** build the data available at that origin (vintage-ready, §5.5), recompute data-dependent prior settings (R9), fit, forecast `H` steps, score against realised values.
- **Warm starts:** MCMC chains start from the last state of the previous origin. Burn-in is still applied (shorter by default); warm starts are an initialisation, not a shortcut around burn-in.
- **Parallelism:** across origins with a process pool (each process owns its JAX runtime); within a fit with `vmap` over chains and equations.
- **Caching:** results are keyed by `(model spec hash, data hash, origin, seed)` on disk. Backtests are resumable, and adding a model to a grid only fits the new model.
- **Budget:** `rb.estimate_cost(grid, data, origins)` extrapolates from a short pilot run before anything expensive starts.

### 9.2 Scores (per origin, horizon and model)

| Score | Scope | Notes |
| :--- | :--- | :--- |
| Log predictive density (LPD) | joint and per variable | Rao-Blackwellised (§7). **Primary score** |
| CRPS | per variable | Sample-based fair estimator, `O(S log S)` via sorting |
| Tail-weighted CRPS | per variable | Quantile-weighted CRPS (Gneiting & Ranjan 2011; left-tail, right-tail and centre weightings) implemented in-house, plus threshold-weighted CRPS from `scoringrules`. Targets the "crisis regime" question |
| Energy score | joint | The `O(S²)` pairwise term is estimated from random pairs |
| Variogram score (p = 0.5) | joint | Scheuerer & Hamill (2015). Sensitive to correlation misspecification, which the energy score barely detects |
| RMSE / MAE | per variable | Posterior mean / median |
| PIT, interval coverage | per variable and horizon | Calibration diagnostics |

`scoringrules` (Apache-2.0) provides `crps_ensemble` (with a fair estimator), `twcrps_ensemble`, `es_ensemble` and `vs_ensemble` with NumPy, Numba and JAX backends. It is a dependency for these scores and the test oracle for anything implemented in-house. It has no quantile-weighted CRPS, and its sample-based log score uses KDE; `rai-bvar` never uses KDE for LPD (§7).

### 9.3 Statistical tests

- Loss differential `d_t = L_{A,t} − L_{B,t}`, per horizon.
- **Diebold-Mariano** statistic with HAC variance (Newey-West, bandwidth at least `h − 1`) and the Harvey-Leybourne-Newbold small-sample correction, using Student-t critical values.
- **Validity policy.** For nested models with expanding windows, DM-type statistics are not asymptotically standard normal (Clark & McCracken 2001). The tests are formally valid in the Giacomini-White (2006) framework, which requires rolling (or fixed) estimation windows; Amisano-Giacomini (2007), for density forecasts, sits in the same framework. `compare()` therefore flags p-values `valid=True` only for rolling windows; expanding-window p-values are labelled descriptive.
- **Model Confidence Set** (Hansen, Lunde & Nason 2011) over the whole grid at α ∈ {0.10, 0.25}, with a block bootstrap. This handles the multiple-comparison problem that a factorial grid creates.
- **Subsample analysis:** scores split by user-defined periods. Default for US data: pre-2008, 2008–09, 2010–19, 2020–21, 2022 onwards.

### 9.4 In-sample screening (fast, never a substitute for the backtest)

- **Exact log marginal likelihood** for conjugate models.
- **PSIS-LOO and WAIC are not provided for VARs.** LOO conditions on the future to predict the past, and pointwise log-likelihoods conditional on latent volatilities produce unreliable Pareto `k̂`. (ArviZ 1.0 has dropped WAIC as well.)
- **PSIS-LFO-CV** (Bürkner, Gabry & Vehtari 2020), Phase 4: approximate leave-future-out using the one-step predictive density with latent states integrated out (particle filter for SV), with exact refits when `k̂ > 0.7`. Built on `arviz_stats.lfo_cv` (available since arviz-stats 1.3.3), with `rai-bvar` supplying the state-integrated log-likelihood and a `SamplingWrapper` for refits.
- **Marginal likelihoods for SV/TVP models** via integrated-likelihood importance sampling (Chan & Eisenstat 2018): research backlog.

### 9.5 `compare()` and the verdict

```python
res = rb.compare(
    rb.grid(mean=[rb.Linear(prior=rb.Minnesota()), rb.BART(n_trees=200)],
            vol=[rb.Constant(), rb.SV(), rb.SV(outliers=True)]),
    data,
    origins=rb.Origins(start="2000Q1", step=1),
    window=rb.Rolling(size=120),
    horizons=[1, 4, 8],
)
res.table()                      # ablation table
res.plot_cumulative_lpd()        # cumulative LPD difference vs reference over time
res.verdict(axis="mean")         # mean effect, holding each volatility spec fixed
res.verdict(axis="vol")          # volatility effect, holding each mean spec fixed
```

**Ablation table columns:** model, axis values (mean / vol / prior / lags), total and mean ΔLPD vs reference (per horizon), Monte Carlo standard error of ΔLPD, mean CRPS and qwCRPS (relative), energy and variogram scores, RMSE / MAE (relative), DM/AG p-value with validity flag, membership of the 10% and 25% MCS, convergence flag, ordering sensitivity (if run).

**Verdict rule.** A model `M` beats a reference `B` along axis `X` only if all of the following hold:

1. `M` and `B` differ only along `X` (R2).
2. ΔLPD(`M − B`) > 0 at the horizons of interest, and |ΔLPD| > 3 × its Monte Carlo standard error (batch means across chains and draws).
3. A valid (rolling-window) AG/GW test rejects at 10%, **or** `B` is excluded from the 25% MCS while `M` is included.
4. Convergence checks passed at ≥ 95% of origins; failures are listed.

Output per comparison: `better` / `worse` / `inconclusive`, with the reasons. For the headline non-linearity claim, the reference is `Linear × SV` (or `Linear × SV(outliers=True)` for post-2020 samples).

---

## 10. Validation strategy

### 10.1 Test layers

| Layer | Content | Cadence |
| :--- | :--- | :--- |
| Unit | Linear algebra, transform round-trips, dummy observations, shapes and dtypes | every PR |
| Analytic | Conjugate posterior and log-ML vs brute force (§10.2) | every PR |
| Contract | `ForecastResult` invariants (§7) for every registered composition | every PR |
| Geweke | Joint-distribution tests for every block and composition | every PR (small config), nightly (full) |
| SBC | Rank uniformity for parameters and test quantities | nightly |
| Parity | Against committed R fixtures (§10.4) | every PR |
| DGP | Recovery and verdict-engine size/power (§10.5) | weekly and pre-release |
| Benchmarks | Speed and memory against targets (§13) | pre-release |

### 10.2 Analytic correctness

- `n = 1, p = 1, T = 20`: posterior moments and log-ML vs numerical quadrature, relative tolerance 1e-6.
- QR path vs naive normal equations on well-conditioned data (1e-10); on designs with condition number ~1e12 the QR path stays accurate where the naive path fails.
- Posterior with dummy observations equals the posterior with the equivalent explicit prior moments.

### 10.3 Sampler correctness

- **Geweke (2004) joint-distribution tests.** Compare the marginal-conditional simulator (draw `θ` from the prior, then `y | θ`) with the successive-conditional simulator (alternate one Gibbs sweep with `y | θ`). Test first and second moments of parameter functions with z-tests using MCSE and a Holm correction. Requires proper priors; all test configurations use them.
- **Simulation-based calibration** (Talts et al. 2018; Modrák et al. 2025). 200–1000 replications; rank statistics for each parameter and for data-dependent test quantities, including the log-likelihood, which Modrák et al. (2025) show increases sensitivity. Thinned to reduce autocorrelation. Pass: no rejection of uniformity at 1% after multiplicity correction (ECDF-envelope test).
- These tests catch exactly the CCM-type failure: a sampler that converges to a distribution close to, but not equal to, the posterior.
- **Analytic twin.** Once the asymmetric conjugate prior exists, MCMC `Linear(form="structural") × Constant` with the same normal-inverse-gamma priors must match its exact iid posterior within MCSE. This is an end-to-end oracle for the MCMC engine at realistic sizes (`n = 20+`), where Geweke tests and SBC are too expensive.
- **Simulation consistency:** forecast draws vs stored conditional moments (§7 invariant 1).

### 10.4 Parity with R (numerical oracles)

- `tools/make_r_fixtures.R` runs the R packages on fixed datasets (a FRED-QD subset, a FRED-MD subset, simulated data) and writes posterior summaries and forecast quantiles to NetCDF files under `tests/parity/fixtures/`. It is not run in CI; R and package versions are stored in the fixture attributes.
- Comparisons:
  - `BVAR`: log-ML at fixed hyperparameters (deterministic; equal up to a documented constant, relative tolerance 1e-8); posterior quantiles of `B`, `Σ` and hyperparameters within 4 MCSE.
  - `stochvol`: posteriors of `μ`, `φ`, `σ` and log-volatility path quantiles within 4 MCSE.
  - `bayesianVARs`: `Linear(form="reduced")` × SV, since it implements the corrected CCCM triangular algorithm (Phase 2); global-local priors × SV (Phase 3).
  - `factorstochvol`: `FactorSV` (Phase 5).
  - `bvarsv`: TVP-VAR-SV, small `n` (Phase 5).
- **Python-side oracle:** the Bank of England `bvar` (MIT, pip-installable) covers the conjugate GLP path. It runs directly in CI as a second check on Phase 1, without R.
- Prior parametrisations are mapped explicitly. Where no exact mapping exists, only forecast distributions on the same data are compared.

### 10.5 DGP tests: validating the verdict engine

Simulate from known DGPs (`n = 3–7`, `T = 200–400`, many replications) and run the full `compare()` pipeline:

| DGP | Expected verdict |
| :--- | :--- |
| Linear, homoskedastic | Nothing beats `Linear × Constant` |
| Linear, SV | `SV` beats `Constant`; `BART × SV` does **not** beat `Linear × SV` (false-positive rate ≤ nominal level) |
| Linear, SV + outliers | `SV(outliers=True)` ≥ `SV` |
| Threshold, SV | `Threshold × SV` and `BART × SV` beat `Linear × SV` (power) |
| TVP with sizeable drift, SV | `TVP × SV` beats `Linear × SV` |

Size (false `better` rate on null DGPs) and power are reported in the documentation. This turns the "volatility first" rule from advice into a tested property of the library.

### 10.6 Runtime convergence diagnostics

- Rank-normalised split R-hat, bulk and tail ESS (Vehtari et al. 2021) via ArviZ, monitored on hyperparameters, a subset of coefficients, log-volatilities at selected dates, and the log-likelihood. MCSE for forecast quantiles and for ΔLPD.
- Defaults: 4 chains, R-hat < 1.01, bulk ESS > 400. Warnings attach to results and aggregate across backtest origins.

### 10.7 Replication targets (phase acceptance)

- **Phase 1:** the Kuschnig & Vashold (2021) example and GLP (2015) hyperparameter posteriors on their data.
- **Phase 2:** a Clark & Ravazzolo (2015)-style ranking: SV beats homoskedastic models on density forecasts (FRED-QD).
- **Phase 3:** the qualitative sparse-vs-dense findings of Gruber & Kastner (2025).
- **Phase 4:** Clark et al. (2023): BART-VAR improves tail forecasts (qwCRPS) relative to linear SV models on their sample. Qualitative only: their main specification uses factor SV, so a closer replication waits for `FactorSV` (Phase 5).

---

## 11. Tech stack

| Concern | Choice | Notes |
| :--- | :--- | :--- |
| Analytic engine | `numpy`, `scipy.linalg`, `scipy.optimize` | float64 |
| MCMC engine | `jax` (x64 enabled) | Hand-written Gibbs; `lax.scan` over iterations; `vmap` over chains and equations |
| BART | `bartz` (MIT, JAX, pre-1.0) | Low-level `mcmcstep` / `mcmcloop` API with per-observation `error_scale`. Version pinned behind an adapter; gated by the Phase 4 spike (§13); fallback is an in-house JAX BART |
| Containers | `xarray.DataTree` (core since xarray 2024.10) | Posterior and forecast storage; the native format of ArviZ 1.x |
| Diagnostics | `arviz-base`, `arviz-stats` (ArviZ ≥ 1.0) | R-hat, ESS, MCSE, LFO-CV; behind a thin adapter, versions pinned |
| Scores | `scoringrules` (Apache-2.0) | CRPS, threshold-weighted CRPS, energy and variogram scores (§9.2) |
| Data | `pandas` | |
| Plots | `matplotlib` | Optional extra |
| Testing | `pytest`, `hypothesis` | |
| Not in core | PyMC, NumPyro, `pymc-bart` | NUTS mixes worse than precision-sampler Gibbs + ASIS on SV/TVP states; `pymc-bart` (now on PyMC 6 and the Rust `bartrs` backend) runs only inside a PyMC model |

Extras: `rai-bvar[bart]`, `rai-bvar[plot]`, `rai-bvar[dev]`. **Python ≥ 3.12**, which current JAX, ArviZ 1.x and `scoringrules` already require; afterwards follow SPEC 0.

---

## 12. Package layout

```text
src/rai_bvar/
├── __init__.py               # public API: BVAR, VARData, compare, grid, components
├── api.py                    # BVAR façade: validates composition, resolves engine, fit() / forecast()
├── linalg.py                 # NumPy linear algebra: QR least squares, IW / matrix-normal draws  [P0]
├── data/
│   ├── dataset.py            # VARData: raw levels + tcodes → sample; ordering; vintage hook   [P0]
│   ├── transforms.py         # tcode registry; forward and pathwise inverse transforms         [P0]
│   ├── design.py             # lagged design matrices and regressor vectors                    [P0]
│   └── outliers.py           # Lenza-Primiceri scaling, exclusion masks, COVID warnings
├── datasets/
│   └── fred.py               # FRED-MD/QD loaders: pinned vintages, checksums, variable sets   [P0]
├── models/
│   ├── base.py               # Model / FittedModel protocols, spec hashing                     [P0]
│   └── benchmarks.py         # RandomWalk, OLSVAR (plug-in or flat-prior draws)                [P0]
├── components/
│   ├── mean/
│   │   ├── linear.py         # structural-form linear mean (independent equations)
│   │   ├── linear_reduced.py # reduced-form prior; corrected triangular sampler (CCCM 2022)
│   │   ├── bart.py           # sum-of-trees mean via bartz            [P4]
│   │   ├── tvp.py            # non-centred TVP with shrinkage         [P5]
│   │   └── regime.py         # threshold and Markov-switching         [P5]
│   ├── vol/
│   │   ├── constant.py       # constant λ_i (IG); conjugate IW marker
│   │   ├── sv.py             # AR(1)/RW log-vol: mixture, precision sampler, ASIS
│   │   ├── sv_robust.py      # outlier component, Student-t scale mixture
│   │   ├── common_sv.py      # common drifting volatility             [P3]
│   │   └── factor_sv.py      # factor SV                              [P5]
│   └── priors/
│       ├── minnesota.py      # hyperparameters, scaling, transform-aware prior mean
│       ├── dummies.py        # Minnesota, SOC and SUR dummy observations
│       ├── hierarchy.py      # hyperpriors; EB (mode) and MH over hyperparameters
│       ├── global_local.py   # HS, NG, DL, R2D2, SSVS with coefficient groups [P3]
│       └── asymmetric.py     # asymmetric conjugate prior (Chan 2022)        [P3]
├── engines/
│   ├── analytic.py           # conjugate NIW: QR posterior, log-ML, iid draws
│   └── mcmc/
│       ├── sweep.py          # global blocks → vmap(equations) → hyperparameter blocks
│       ├── state.py          # PyTree sampler state; PRNG key discipline
│       ├── linalg.py         # precision samplers; banded / tridiagonal solvers
│       └── runner.py         # chains, burn-in, thinning, warm starts, diagnostics hooks
├── forecast/
│   ├── result.py             # ForecastResult: the predictive contract (§7)                    [P0]
│   ├── simulate.py           # joint path simulation and conditional moments                  [P0]
│   ├── invariants.py         # contract checks every model must pass                           [P0]
│   └── conditional.py        # conditional forecasts                  [P3]
├── structural/
│   ├── irf.py                # recursive IRF / FEVD; time-varying IRFs
│   ├── girf.py               # GIRF with common random numbers        [P4]
│   └── ordering.py           # ordering-sensitivity check
├── evaluation/
│   ├── scores.py             # LPD, CRPS, qwCRPS, twCRPS, energy, variogram, RMSE/MAE, PIT   [P0]
│   ├── significance.py       # DM-HAC + HLN, MCS (block bootstrap)                             [P0]
│   ├── screening.py          # exact log-ML; PSIS-LFO                 [P4]
│   ├── backtest.py           # origins, windows, warm starts, caching, parallelism            [P0]
│   └── compare.py            # compare(), grid(), tables, plots, verdict()                     [P0]
└── diagnostics.py            # R-hat, ESS, MCSE (ArviZ adapter)

tests/
├── unit/   contract/   integration/                                  [P0]
├── analytic/   geweke/   sbc/   dgp/   bench/
├── fixtures/                 # synthetic FRED-format CSVs                                       [P0]
└── parity/fixtures/          # NetCDF fixtures generated from R

tools/
└── make_r_fixtures.R         # regenerates parity fixtures; not run in CI
```

---

## 13. Phased roadmap

Each phase ends with explicit exit criteria. Phases 1 and 2 can run in parallel once Phase 0 is done; Phases 3–5 need Phase 2.

### Phase 0: Foundations

- Data layer, transforms, `VARData`.
- `ForecastResult` with its invariants.
- Scores (validated against `scoringrules`), statistical tests, backtest engine with caching.
- Two trivial benchmark models, random walk and OLS VAR, to exercise the pipeline end to end.
- Reference dataset loaders (FRED-QD, FRED-MD) with pinned vintages and checksums; small / medium / large variable sets (§5.8).
- CI with the test layers of §10.1.

**Exit:** `compare()` runs end to end on random walk vs OLS VAR and produces the ablation table, DM/MCS output and cumulative-LPD plot.

### Phase 1: Analytic engine

- Conjugate Minnesota with SOC/SUR dummies; GLP hierarchy (EB and MH).
- Lenza-Primiceri COVID scaling.
- Recursive IRFs and FEVD.

**Exit:** parity with R's `BVAR` and with the Bank of England `bvar` (§10.4, decision D5); analytic tests green; GLP / Kuschnig-Vashold replication; `n = 100` fit within a documented time budget.

### Phase 2: MCMC engine and the volatility axis

- JAX Gibbs runner (`sweep`, `state`, `runner`, `linalg`).
- `Linear` (structural form, and reduced form via the corrected CCCM 2022 sampler) × {`Constant`, `SV`, `SV(outliers=True)`, `SV(tails="t")`}.
- Ordering-sensitivity check; runtime convergence diagnostics.

**Exit:** Geweke and SBC green for every composition; `stochvol` parity; `bayesianVARs` parity for the reduced-form sampler; performance target (to be calibrated in a pilot, initial goal: `n = 20`, `p = 4`, `T = 300`, `Linear × SV`, 4 chains × 10,000 draws in under 2 minutes on a laptop CPU); Clark-Ravazzolo-style replication.

### Phase 3: Shrinkage and practical features

- Horseshoe, Normal-Gamma, Dirichlet-Laplace, R2D2, SSVS.
- Asymmetric conjugate prior (Chan 2022). Candidate to pull into Phase 2: it is the analytic twin of `Linear(form="structural") × Constant` and gives an exact oracle for the MCMC engine (§10.3).
- `CommonSV` (Carriero, Clark & Marcellino 2016): a conjugate-speed SV screen for large `n` (decision D2).
- Conditional forecasts. (The Model Confidence Set landed in Phase 0.)

**Exit:** `bayesianVARs` parity; `bvars` parity for `CommonSV`; DGP verdict-engine tests on linear DGPs.

### Phase 4: Nonlinear mean

- **Week-1 spike (decision gate).** `bartz` already accepts per-observation `error_scale` and exposes `mcmcstep.step`, but it has no public setter for changing `y` / `error_scale` between steps. The spike answers:
  1. Can the residual target and `1/λ_{i,t}` be swapped into `State` each sweep (fields `resid`, `error_scale` and the derived precision terms) inside a jitted loop, and stay correct across `bartz` releases? Upstream a public setter if possible.
  2. Does the float32 leaf computation pass Geweke and SBC for the BART block at realistic `T` and volatility ranges?
  3. Is a fixed error precision (`Wishart(value=...)`, which skips the σ update) the right hook to leave volatility entirely to `rai-bvar`'s SV block?

  If any answer is no, decide between a fork and an in-house JAX BART before continuing.
- `BART` mean block, GIRFs with common random numbers, PSIS-LFO screening.

**Exit:** DGP power test (threshold DGP) and false-positive test (linear-SV DGP) pass; qualitative replication of Clark et al. (2023).

### Phase 5: Time variation and regimes

- `TVP` (non-centred, with shrinkage), `Threshold`, `MarkovSwitching`.
- Order-invariant SV (Chan, Koop & Yu 2024); factor SV.

**Exit:** `bvarsv` parity for small `n`; DGP tests for TVP and regime models.

### Phase 6: Structural identification

- Sign and zero restrictions (Arias, Rubio-Ramírez & Waggoner 2018), narrative restrictions, proxy SVARs.
- Decide whether this lives in `rai-bvar` or a separate `rai-bvar-structural` package.

---

## 14. Risks & decisions

| Risk | Mitigation |
| :--- | :--- |
| `bartz` embedding relies on undocumented `State` fields; pre-1.0 API; float32 leaf updates | Phase 4 spike; adapter with a pinned version; upstream a public setter; fallback in-house JAX BART (significant effort, budget accordingly) |
| Overlap with emerging Python projects (Bank of England `bvar`, Chan's toolkit port, `bvhar`) | Differentiate on the ablation/verdict engine, factorial composition and validation (§1.4); reuse MIT code where it helps; consider interop or contributing instead of duplicating |
| Backtest compute cost (grid × origins × chains) | Caching, warm starts, `estimate_cost()`, analytic models as a first screen |
| Variable-ordering dependence | R8, ordering-sensitivity check, order-invariant models in Phase 5 |
| ArviZ API churn | Thin adapter module; pinned versions |
| Users expect reduced-form Minnesota but get the structural-form default | Both forms offered; default stated in the docs and in `fit.summary()` |
| GPL contamination | Clean-room rule (§2.1); PR checkbox |
| JAX compile times across many compositions | Cache compiled sweeps by static configuration; no Python-level branching inside sweeps |

**Decisions (2026-10-21)**

| ID | Question | Decision | Rationale |
| :--- | :--- | :--- | :--- |
| D1 | Licence | **Apache-2.0** | Patent grant, enterprise-friendly, compatible with all core dependencies (JAX, ArviZ, `scoringrules`: Apache-2.0; `bartz`: MIT). Cost: GPL R code cannot be translated, so the clean-room rule applies (§2.1) |
| D2 | Where does `CommonSV` go? | **Phase 3** | Phase 2 is already heavy and its baseline is `Linear × SV`. `CommonSV` earns its place as a fast SV screen for large `n`, with `bvars` as oracle |
| D3 | Default Minnesota form for MCMC models | **Structural**, with the prior mean of `γ_i` centred on `a_i` so the reduced-form prior mean is preserved (§4.1) | Equations stay independent and parallel; `Linear` and `BART` share one structural treatment of `a_i`, keeping the mean-axis ablation clean; Chan (2022) gives an analytic twin. `form="reduced"` (CCCM 2022) remains available |
| D4 | Reference datasets | **FRED-QD + FRED-MD**, small / medium / large sets, downloaded on demand with pinned vintages (§5.8) | Literature standard (GLP, CCM, Clark-Ravazzolo); not bundled because of third-party copyrights; fixtures use federal-agency series only |
| D5 | Relationship with the Bank of England `bvar` | **Build our own analytic engine; use BoE `bvar` as a Python-side parity oracle in CI** | The conjugate model is the baseline of every ablation, so its contract (conditional moments), MH over hyperparameters, Lenza-Primiceri scaling and look-ahead-free refits must stay under our control. The engine is small (about 1–2 weeks) |

---

## 15. References

### Priors and mean specifications
- Chan, J. C. C. (2022). Asymmetric conjugate priors for large Bayesian VARs. *Quantitative Economics*, 13(3), 1145–1169.
- Chan, J. C. C., Eisenstat, E., & Strachan, R. W. (2020). Reducing the state space dimension in a large TVP-VAR. *Journal of Econometrics*, 218(1), 105–118.
- Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). BART: Bayesian additive regression trees. *Annals of Applied Statistics*, 4(1), 266–298.
- Clark, T. E., Huber, F., Koop, G., Marcellino, M., & Pfarrhofer, M. (2023). Tail forecasting with multivariate Bayesian additive regression trees. *International Economic Review*, 64(3), 979–1022.
- Del Negro, M., & Primiceri, G. E. (2015). Time varying structural vector autoregressions and monetary policy: A corrigendum. *Review of Economic Studies*, 82(4), 1342–1345.
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *Review of Economics and Statistics*, 97(2), 436–451.
- Gruber, L., & Kastner, G. (2025). Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends! *International Journal of Forecasting*, 41(4), 1589–1619.
- Huber, F., Koop, G., & Onorante, L. (2021). Inducing sparsity and shrinkage in time-varying parameter models. *Journal of Business & Economic Statistics*, 39(3), 669–683.
- Huber, F., & Rossini, L. (2022). Inference in Bayesian additive vector autoregressive tree models. *Annals of Applied Statistics*, 16(1), 104–123.
- Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR-models. *Journal of Applied Econometrics*, 12(2), 99–132.
- Kuschnig, N., & Vashold, L. (2021). BVAR: Bayesian vector autoregressions with hierarchical prior selection in R. *Journal of Statistical Software*, 100(14), 1–27.
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Review of Economic Studies*, 72(3), 821–852.
- Zhang, Y. D., Naughton, B. P., Bondell, H. D., & Reich, B. J. (2022). Bayesian regression using a prior on the model fit: The R2-D2 shrinkage prior. *Journal of the American Statistical Association*, 117(538), 862–874.

### Volatility, large VARs and ordering
- Arias, J. E., Rubio-Ramírez, J. F., & Shin, M. (2023). Macroeconomic forecasting and variable ordering in multivariate stochastic volatility models. *Journal of Econometrics*, 235(2), 1054–1086.
- Bognanni, M. (2022). Comment on "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *Journal of Econometrics*, 227(2), 498–505.
- Carriero, A., Chan, J. C. C., Clark, T. E., & Marcellino, M. (2022). Corrigendum to "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *Journal of Econometrics*, 227(2), 506–512.
- Carriero, A., Clark, T. E., & Marcellino, M. (2016). Common drifting volatility in large Bayesian VARs. *Journal of Business & Economic Statistics*, 34(3), 375–390.
- Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors. *Journal of Econometrics*, 212(1), 137–154.
- Carriero, A., Clark, T. E., Marcellino, M., & Mertens, E. (2024). Addressing COVID-19 outliers in BVARs with stochastic volatility. *Review of Economics and Statistics*, 106(5), 1403–1417.
- Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *International Journal of Mathematical Modelling and Numerical Optimisation*, 1(1/2), 101–120.
- Chan, J. C. C., Koop, G., & Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic volatility. *Journal of Business & Economic Statistics*, 42(2), 825–837.
- Clark, T. E., & Ravazzolo, F. (2015). Macroeconomic forecasting performance under alternative specifications of time-varying volatility. *Journal of Applied Econometrics*, 30(4), 551–575.
- Kastner, G. (2019). Sparse Bayesian time-varying covariance estimation in many dimensions. *Journal of Econometrics*, 210(1), 98–115.
- Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data Analysis*, 76, 408–423.
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361–393.
- Lenza, M., & Primiceri, G. E. (2022). How to estimate a vector autoregression after March 2020. *Journal of Applied Econometrics*, 37(4), 688–699.
- McCausland, W. J., Miller, S., & Pelletier, D. (2011). Simulation smoothing for state-space models: A computational efficiency analysis. *Computational Statistics & Data Analysis*, 55(1), 199–212.
- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2), 425–449.
- Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *American Economic Review*, 96(1), 54–81.

### Forecast evaluation
- Amisano, G., & Giacomini, R. (2007). Comparing density forecasts via weighted likelihood ratio tests. *Journal of Business & Economic Statistics*, 25(2), 177–190.
- Bürkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out cross-validation for Bayesian time series models. *Journal of Statistical Computation and Simulation*, 90(14), 2499–2523.
- Chan, J. C. C., & Eisenstat, E. (2018). Bayesian model comparison for time-varying parameter VARs with stochastic volatility. *Journal of Applied Econometrics*, 33(4), 509–532.
- Clark, T. E., & McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. *Journal of Econometrics*, 105(1), 85–110.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545–1578.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Gneiting, T., & Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *Journal of Business & Economic Statistics*, 29(3), 411–422.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Scheuerer, M., & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review*, 143(4), 1321–1334.

### Structural analysis and conditional forecasts
- Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018). Inference based on structural vector autoregressions identified with sign and zero restrictions: Theory and applications. *Econometrica*, 86(2), 685–720.
- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. *Journal of Econometrics*, 74(1), 119–147.
- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate models. *Review of Economics and Statistics*, 81(4), 639–651.

### Validation and diagnostics
- Geweke, J. (2004). Getting it right: Joint distribution tests of posterior simulators. *Journal of the American Statistical Association*, 99(467), 799–804.
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejsková, K., Gelman, A., & Vehtari, A. (2025). Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. *Bayesian Analysis*, 20(2), 461–488.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.

### Data
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- McCracken, M. W., & Ng, S. (2021). FRED-QD: A quarterly database for macroeconomic research. *Federal Reserve Bank of St. Louis Review*, 103(1), 1–44.
