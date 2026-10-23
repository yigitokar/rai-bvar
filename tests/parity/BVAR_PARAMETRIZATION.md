# How the R package `BVAR` parametrises the GLP model

This note pins down exactly what `BVAR` 1.0.5 (Kuschnig & Vashold, CRAN, GPL-3) computes, so
that its outputs can serve as numerical oracles for the rai-bvar analytic engine (SPEC §4.6,
§10.4). It was established by running the package and reading its documentation and source
(`R/10_bvar.R`, `11_input.R`, `12_aux.R`, `13_mvtnorm.R`, `20_ml.R`, `21_draw.R`, `22_calc.R`,
`30_metropolis_setup.R`, `40_priors_setup.R`–`43_sur_soc.R`, `51_fcast_compute.R`,
`54_predict.R`, and the JSS article). Every formula below was also checked numerically by
`tools/make_r_fixtures.R`, which reproduces BVAR's numbers from an independent implementation.

Clean-room rule (SPEC §2.1): this note describes behaviour in mathematical terms so that tests
can be written. It is not a description of code to be ported. rai-bvar implements the model
from GLP (2015) and the textbook conjugate results.

Versions used for the committed fixtures: R 4.1.2 (x86_64-apple-darwin17.0), BVAR 1.0.5,
mvtnorm 1.1-3 (CRAN macOS binary), jsonlite 1.8.3, coda 0.19-4.1.

Notation: `n` variables (BVAR: `M`), `p` lags, `T` raw observations, `N = T − p` equations
after dropping the first `p` rows, `k = 1 + n p` regressors (BVAR: `K`), `d` inverse-Wishart
degrees of freedom.

## 1. Sample, regressors and the constant

- `y_t = c + A_1 y_{t−1} + … + A_p y_{t−p} + u_t`, `u_t ~ N(0, Σ)`, written `Y = X B + U`
  with `B` of size `k × n`.
- Rows: the first `p` raw observations are only used as lags, so `Y` holds raw rows
  `p+1, …, T` (`N = T − p`).
- Columns of `X`: **constant first, then the lag-1 block (variables in data order), …, then the
  lag-p block**: `x_t = (1, y_{t−1}', …, y_{t−p}')'`. This is exactly rai-bvar's layout
  (`deterministic="c"`), so `B` matrices compare element by element. BVAR has no other
  deterministic terms.

## 2. The conjugate Minnesota prior

```
Σ ~ IW(Ψ, d),          Ψ = diag(ψ_1, …, ψ_n),   d = n + 2   (hard-coded)
vec(B) | Σ ~ N(vec(b), Σ ⊗ Ω)      i.e.  B | Σ ~ MN(b, Ω, Σ)
```

- IW density `∝ |Σ|^{−(d+n+1)/2} exp(−tr(Ψ Σ⁻¹)/2)`, so `E[Σ] = Ψ / (d − n − 1) = Ψ`.
- Prior mean `b` (`bv_mn(b = ...)`, default `1`; `"auto"` also means 1): `k × n` zeros except
  the own first lags, `b[1 + i, i] = b_i` (scalar or length-`n` vector), or a full `k × n`
  matrix.
- `Ω = diag(ω)` (`k × k`):
  - constant: `ω_1 = var`, default `var = 1e7`. With the Kronecker structure the prior
    variance of the intercept of equation `i` is `Σ_ii · 1e7`.
  - lag `l` of variable `j`: `ω = λ² / (l^α · ψ_j)`.

  This is GLP's `λ² / l^α · Σ_ii / (ψ_j / (d − n − 1))` with `d − n − 1 = 1`. `ψ` enters
  linearly: it is never squared or square-rooted inside the model. Mathematically `ψ_j` is on
  the **variance** scale (it is the IW scale diagonal and `Σ_ii / ψ_j` is a ratio of variances).
- Defaults: `λ` mode 0.2, `α` mode 2 (both fixed unless hierarchical), `var = 1e7`.

## 3. ψ: what BVAR's `psi = "auto"` does

`bv_psi(scale = 0.004, shape = 0.004, mode = "auto", min = "auto", max = "auto")`.
With `mode = "auto"`, for each variable `j`:

```
ψ_j = sqrt( sigma2 of stats::arima(Y[, j], order = c(p, 0, 0)) )
```

- `Y` is the **trimmed** sample (raw rows `p+1..T`), not the full series.
- `arima` defaults: `method = "CSS-ML"` (conditional sum of squares for starting values, then
  exact Gaussian ML), `include.mean = TRUE`. `sigma2` is the ML innovation variance.
- On a warning the fit is repeated with warnings suppressed (this happens for the
  `qd_small_levels` GDP and deflator series: "NaNs produced"). On an error an ARIMA(p, 1, 0)
  is fitted instead.
- The result is a **standard deviation**, yet it is used as the IW scale diagonal and in `Ω`,
  i.e. where a variance belongs. (NEWS for 1.0.4: an earlier version took the square root
  twice; the documented intent is "the square root of the innovations variance".)
- Bounds when ψ is hierarchical: `min = mode / 100`, `max = mode · 100`.

GLP (2015) treat `ψ_j` as a variance (the IW scale diagonal). rai-bvar's `psi` is in variance
units, so **to reproduce a BVAR run pass BVAR's ψ numbers verbatim** (`psi=(…)`), whatever
BVAR called them. The fixtures avoid the ambiguity by fixing ψ explicitly: `psi` =
`arima_ml_variance` (the square of BVAR's auto value). `r_bvar_logml.json["psi_conventions"]`
also lists BVAR's auto value (`bvar_auto_sd`) and OLS AR(p) residual variances on the full
series with (`ols_ar_variance_dof`, `SSR / (T − p − (p + 1))`) and without
(`ols_ar_variance_mle`, `SSR / (T − p)`) a degrees-of-freedom correction.

## 4. SOC and SUR dummy observations

Both are optional, named in `bv_priors(..., soc = bv_soc(), sur = bv_sur())`, and stacked on top
of the data, `Y* = [Y_d; Y]`, `X* = [X_d; X]`, in the order given.

- **Sum of coefficients** (`soc`, GLP's `μ`), `n` rows:
  `Y_soc = diag(ȳ0) / μ`,  `X_soc = [0_{n×1}, Y_soc, Y_soc, …, Y_soc]` (`p` copies).
- **Single unit root / dummy initial observation** (`sur`, GLP's `δ`), 1 row:
  `Y_sur = ȳ0' / δ`,  `X_sur = [1/δ, ȳ0'/δ, …, ȳ0'/δ]` (`p` copies).
- Smaller `μ`, `δ` mean tighter priors; `μ, δ → ∞` switch them off.

**Initial-observation mean `ȳ0` (deviation).** BVAR's built-in `bv_soc()`/`bv_sur()` receive the
trimmed `Y` and use `ȳ0 = colMeans(Y[1:p, ])`, i.e. the mean of **raw observations
`p+1, …, 2p`** (for `p = 1`, raw observation 2). GLP (2015), and SPEC §4.6, use the mean of the
first `p` observations, raw rows `1..p` (the initial conditions). The two give different log
MLs (e.g. qd, `λ = 0.2`, SOC+SUR: −593.2554 with GLP's `ȳ0`, −593.6052 with BVAR's).

The fixtures therefore use GLP's `ȳ0` as the primary convention, implemented inside BVAR through
its documented custom-dummy interface (`bv_dummy(fun = …)`, a closure over the raw first `p`
rows); entries with `"ybar0": "bvar"` use BVAR's built-in constructors and are kept for
diagnosis only.

## 5. Marginal likelihood

For any `(Y, X)` with `N` rows, BVAR evaluates (with `d = n + 2`)

```
L(Y, X) = −(N n / 2) log π + Σ_{i=0}^{n−1} [ lnΓ((N + d − i)/2) − lnΓ((d − i)/2) ]
          − (N/2) Σ_j log ψ_j
          − (n/2)       Σ log(1 + eig(Ω^{1/2} X'X Ω^{1/2}))
          − ((N + d)/2) Σ log(1 + eig(Ψ^{−1/2} Ŝ Ψ^{−1/2}))

B̂ = (X'X + Ω⁻¹)⁻¹ (X'Y + Ω⁻¹ b),   Ŝ = (Y − X B̂)'(Y − X B̂) + (B̂ − b)' Ω⁻¹ (B̂ − b)
```

(eigenvalues below 1e-12 are set to 0 before adding 1). This is the standard matrix-t marginal
likelihood

```
log p(Y) = −(Nn/2) log π + log Γ_n((N+d)/2) − log Γ_n(d/2) − (n/2) log|Ω| − (n/2) log|X'X + Ω⁻¹|
           + (d/2) log|Ψ| − ((N+d)/2) log|Ψ + Ŝ| .
```

**With dummies** (GLP): `log p(Y | h) = L(Y*, X*) − L_d(Y_d, X_d)`, the marginal likelihood of
the data given the prior updated by the dummy observations (`N* = N + N_d` rows in the first
term, `N_d` in the second).

**Deviation in the dummy-only term.** BVAR evaluates `L_d` with `B̂ = b` (the prior mean)
instead of the dummy posterior mean `(X_d'X_d + Ω⁻¹)⁻¹(X_d'Y_d + Ω⁻¹b)`, i.e. with
`Ŝ_d = (Y_d − X_d b)'(Y_d − X_d b)`. This is exact if and only if `Y_d = X_d b`, which holds for
SOC/SUR when `b` is a random walk (`b_i = 1` on every own first lag, 0 elsewhere; then
`X_d b = Y_d`). For any other prior mean BVAR's `log p(Y | h)` is wrong: on `sim_var3` with
`b = 0`, by 0.3–5 log points with SOC and by a constant 6.6 log points with SUR, and its EB
objective with SOC+SUR becomes degenerate (mode at the lower bounds `μ = δ = 1e-4`, log
posterior −565.5 instead of −748.5 at the exact interior mode). rai-bvar must use the exact
dummy-only term; the fixtures carry both (`log_ml` exact, `log_ml_bvar`, `bvar_is_exact`).

**What BVAR reports.** `bv_ml()` returns, and `bvar()` stores in `$ml` and `$optim$value`, the
**log posterior kernel**

```
log p(Y | h) + Σ_{h_i hierarchical} log p(h_i)
```

with natural-scale hyperprior densities, no Jacobian, no normalisation for the truncation to the
bounds, and nothing for hyperparameters that are held fixed. There is no other additive
constant: `L(Y, X)` above is the complete normalised density. Hence rai-bvar's
`fit.log_ml()` must equal BVAR's `ml` minus the hyperprior terms, and `fit.log_posterior()`
must equal BVAR's `ml`.

## 6. Hyperpriors

| Name | BVAR constructor (defaults) | Density | Bounds (default) |
| :--- | :--- | :--- | :--- |
| `lambda` (λ) | `bv_lambda(mode = 0.2, sd = 0.4, min = 1e-4, max = 5)` | Gamma(k, θ) | [1e-4, 5] |
| `alpha` (α) | `bv_alpha(mode = 2, sd = 0.25, min = 1, max = 3)` | Gamma(k, θ) | [1, 3] |
| `soc` (μ) | `bv_soc(mode = 1, sd = 1, min = 1e-4, max = 50)` | Gamma(k, θ) | [1e-4, 50] |
| `sur` (δ) | `bv_sur(mode = 1, sd = 1, min = 1e-4, max = 50)` | Gamma(k, θ) | [1e-4, 50] |
| `psi1..psin` (ψ_j) | `bv_psi(scale = 0.004, shape = 0.004)` | Inverse-Gamma(shape a, scale b) | [mode/100, mode·100] |

- Gamma from mode `m` and s.d. `s` (shape `k`, scale `θ`, density
  `x^{k−1} e^{−x/θ} / (Γ(k) θ^k)`): with `r = m²/s²`,
  `k = (2 + r + sqrt((4 + r) r)) / 2`, `θ = sqrt(s² / k)` (solves `(k − 1)θ = m`, `k θ² = s²`).
  Defaults: λ: `k = 1.6403882032`, `θ = 0.3123105626`; μ, δ: `k = 2.6180339887`,
  `θ = 0.6180339887`; α: `k = 65.984845005`, `θ = 0.0307764064`.
  rai-bvar's `GammaPrior(mode, sd, lower, upper)` uses the same formula (verified by the tests).
- Inverse-Gamma: `log p(ψ) = a log b − (a + 1) log ψ − b/ψ − lnΓ(a)`, applied to each `ψ_j` in
  its own (BVAR) units. **Deviation:** BVAR's default `a = b = 0.004`; GLP use `0.02² = 0.0004`
  on `ψ_j / (d − n − 1) = ψ_j`. To mimic BVAR in rai-bvar use
  `InvGammaPrior(scale=0.004, shape=0.004)`. (Not exercised by the fixtures: ψ is fixed.)
- `bv_priors(hyper = "auto")` means `λ` plus the hyperparameter of every dummy prior supplied
  (here `soc`, `sur`). `"full"` adds `α` and all `ψ_j`. Otherwise a character vector. A
  hyperparameter that is not hierarchical is fixed at its `mode`.
- Proposals or optimiser steps outside the bounds get log posterior `−1e18`, so all targets are
  truncated to the bounds.

## 7. Posterior mode ("empirical Bayes")

`stats::optim(par = modes, fn = log posterior, method = "L-BFGS-B", lower = min, upper = max,
control = list(fnscale = −1))` on the **natural scale**, numerical gradient (`ndeps = 1e-3`),
default `factr = 1e7`, starting at the hyperprior modes. BVAR never uses the mode for inference
by itself: it only centres the MH chain. Its `optim` result is accurate to roughly 1e-5
relative (e.g. qd, λ only: 0.2966692 vs 0.2966674 at a tight optimum).

The fixtures report BVAR's raw `optim` result and the maximiser of the same objective refined
to ~1e-6 (`mode`, `log_posterior`). Maximising `log p(Y|h) + log p(h)` over `log h` *without* a
Jacobian (as rai-bvar does) has the same maximiser.

## 8. Metropolis-Hastings

`bvar(n_draw = 10000, n_burn = 5000, n_thin = 1, mh = bv_mh(scale_hess = 0.01,
adjust_acc = FALSE, adjust_burn = 0.75, acc_lower = 0.25, acc_upper = 0.45, acc_change = 0.01))`.

- **Target**: the log posterior kernel of §5 on the natural scale of `h`, truncated to the
  bounds. Gaussian random walk on the natural scale, accept if `U < exp(lp' − lp)` (symmetric
  proposal, no Jacobian).
- **Proposal covariance is not an inverse Hessian** (despite the documentation). No Hessian is
  computed. With `m` at the mode (natural scale),
  `J_i = e^{m_i} / (1 + e^{m_i})² · (max_i − min_i)` (the derivative of a logistic map, but
  evaluated at the natural-scale mode) and `H = diag(scale_hess)`:
  - several hyperparameters: proposal covariance `J H J'`, i.e. independent coordinates with
    s.d. `|J_i| · sqrt(scale_hess_i)`;
  - a single hyperparameter: proposal **s.d.** `J² · scale_hess` (the variance formula is used as
    a standard deviation).
- `adjust_acc = TRUE`: every 10 iterations during the first `(1 − adjust_burn)` share of the
  burn-in (25 % by default), with the cumulative acceptance rate since the start: below
  `acc_lower` the proposal scale (covariance eigenvalues, or the s.d. in the one-dimensional
  case) is multiplied by `1 − acc_change`, above `acc_upper` by `1 + acc_change`.
- The chain starts at a draw from the proposal centred at the mode (redrawn until inside the
  bounds). `n_draw` includes `n_burn`; `n_draw − n_burn` iterations are kept (thinned).
- **Coefficients given `h`**: at every kept iteration, one exact draw from the conjugate
  posterior at the current `h`, with the dummies stacked (`N* = N + N_d` rows):

  ```
  S̄ = Ψ + (Y* − X* B̂)'(Y* − X* B̂) + (B̂ − b)' Ω⁻¹ (B̂ − b),    ν̄ = N* + n + 2
  Σ ~ IW(S̄, ν̄)            (W = η'η, rows of η ~ N(0, S̄⁻¹), Σ = W⁻¹)
  B | Σ ~ MN(B̂, (X*'X* + Ω⁻¹)⁻¹, Σ)
  ```

  so `E[B | Y, h] = B̂`, `E[Σ | Y, h] = S̄ / (ν̄ − n − 1) = S̄ / (N* + 1)`. `B̂` is computed from
  the normal equations (`solve(X'X + Ω⁻¹, …)`), which loses up to ~1e-8 absolute accuracy on the
  levels data (condition number ~2.5e10); rai-bvar's QR path is more accurate, so the fixtures'
  targets are QR-based and BVAR's own numbers are kept as `*_bvar`.
- The fixture chains use `adjust_acc = TRUE` with hand-tuned `scale_hess` (acceptance 0.26–0.44),
  4 chains × 20 000 kept draws after 5 000 burn-in.

## 9. Forecasts (`predict.bvar`) — two bugs

For each stored draw `(B, Σ)` BVAR iterates the companion form from the last `p` observations
**without shocks** and then adds independent noise `Σ z_h`, `z_h ~ N(0, I_n)`, to each horizon:

1. the noise is multiplied by `Σ`, not by a Cholesky factor of `Σ`, so its covariance is `Σ²`;
2. the shocks are not propagated through the dynamics, so `h`-step uncertainty is not the
   VAR's `Σ_{s<h} Φ_s Σ Φ_s'`.

On `qd_small_levels` the 90 % band at `h = 8` is about 3 times too narrow. The fixtures
therefore simulate forecasts from BVAR's posterior draws with the correct recursion,
`y_{T+h} = B'x_{T+h} + chol(Σ) z`, one path per draw (`forecast`), and keep BVAR's own output
as `forecast_bvar_native` for reference only. Quantiles are R type 7 (= numpy `"linear"`).

## 10. Where BVAR deviates from GLP (2015)

| Item | GLP (2015) | BVAR 1.0.5 | Fixtures / rai-bvar |
| :--- | :--- | :--- | :--- |
| ψ units | variance (IW scale diagonal) | auto value is an s.d., used as a variance | ψ fixed explicitly, variance units |
| ψ default | estimated (IG hyperprior); starting values from univariate AR residual variances | `sqrt(arima ML σ²)`, AR(p), trimmed sample | fixed explicitly; candidates exported |
| ψ hyperprior | IG(0.02², 0.02²) | IG(0.004, 0.004) | ψ not hierarchical in fixtures |
| ȳ0 for SOC/SUR | mean of raw rows 1..p | mean of raw rows p+1..2p | GLP (custom dummies in BVAR) |
| dummy-only marginal likelihood | exact | prior mean plugged in; exact only for random-walk `b` | exact (`log_ml`), BVAR kept as `log_ml_bvar` |
| optimiser | numerical maximisation of the posterior | L-BFGS-B, natural scale, numerical gradient, loose tolerance | refined maximiser exported |
| MH proposal | inverse Hessian at the mode | `diag(J² · scale_hess)` (no Hessian) | only the target matters |
| forecast noise | — | `Σ z`, not propagated | correct recursion from BVAR draws |
| constant | diffuse | `var = 1e7` | `var_const = 1e7` |
| α | 2 (fixed) | 2 by default; Gamma(2, 0.25) on [1, 3] if hierarchical | fixed at 2 |
| IW d.o.f. | `n + 2` | `n + 2` | `n + 2` |
| `ml` reported | — | log ML + hyperprior log densities | split into `log_ml_bvar`, `log_hyperprior` |

## 11. Mapping to the rai-bvar API

```python
prior = rb.Minnesota(lambda_=λ, alpha=2.0, psi=tuple(ψ), soc=μ or None, sur=δ or None,
                     mean=b, var_const=1e7, hierarchy="fixed" | "eb" | "mh",
                     hyper=("lambda",) or ("lambda", "soc", "sur"),
                     hyperpriors=rb.GLPHyperpriors(
                         lambda_=rb.GammaPrior(0.2, 0.4, 1e-4, 5.0),
                         soc=rb.GammaPrior(1.0, 1.0, 1e-4, 50.0),
                         sur=rb.GammaPrior(1.0, 1.0, 1e-4, 50.0)))
model = rb.BVAR(mean=rb.Linear(prior=prior), vol=rb.Constant(prior="iw"), lags=4,
                deterministic="c")
```

- `fit.log_ml()` ↔ `log_ml` (= BVAR `ml` − hyperprior terms when `bvar_is_exact`).
- `fit.log_posterior()` ↔ `log_posterior_bvar` / BVAR `ml`.
- `fit.posterior().mean / .scale / .df / .omega()` ↔ `B̂ / S̄ / ν̄ / (X*'X* + Ω⁻¹)⁻¹`.
- Data: the CSVs as-is, `VARData.from_frame(frame, tcodes=1)`; `mean=1` for `qd`, `mean=0` (or 1
  where stated) for `sim`.

## 12. Fixtures

Generated by `Rscript tools/make_r_fixtures.R` (about 6 minutes; fixed seeds; exits non-zero if any of its 319 internal consistency checks fails). Matrices are
row-major nested lists in the layout of §1; `null` marks an absent dummy.

| File | Content |
| :--- | :--- |
| `r_bvar_logml.json` | (a) `cases[]`: `qd_p4_rw` (`b = 1`), `sim_p4_zero` (`b = 0`), `sim_p4_rw` (`b = 1`), `qd_p4_rw_bvarpsi` (ψ = BVAR's auto s.d. values, λ = 0.2). `grid[]` over λ ∈ {0.1, 0.2, 0.5} × {none, SOC, SUR, SOC+SUR} (μ = δ = 1) × ȳ0 ∈ {glp, bvar}: `log_ml`, `log_ml_full`, `log_ml_dummy` (exact), `log_ml_bvar`, `log_ml_full_bvar`, `log_ml_dummy_bvar`, `bvar_is_exact`, `log_hyperprior{lambda, soc, sur}`, `log_posterior_bvar`. Also `psi_conventions`, `hyperpriors`, dataset MD5s. |
| `r_bvar_posterior.json` | (b) `qd_p4_rw`, `sim_p4_zero`: λ ∈ {0.1, 0.2, 0.5} × {none, SOC+SUR}: `mean` (k × n), `scale` S̄, `df` ν̄, `omega` (k × k), `sigma_mean`, BVAR's `*_bvar` and their deviation `bvar_vs_exact`, an MC check of `BVAR:::draw_post`; `forecast`: quantiles (5/50/95 %, h = 1..8) from 100 000 iid posterior draws at λ = 0.2 with MCSE. |
| `r_bvar_eb.json` | (c) λ only and λ + SOC + SUR on both datasets: BVAR's `optim` result, the refined mode of BVAR's objective, and the refined `mode` / `log_posterior` / `log_ml` / `log_hyperprior` / `laplace_sd` of the exact objective. |
| `r_bvar_mh.json` | (d) `qd_p4_rw_lambda_soc_sur_glp`, `qd_p4_rw_lambda`, `sim_p4_zero_lambda`: 4 × 20 000 draws; quantiles with MCSE of the hyperparameters, nine coefficients, four elements of Σ, and forecasts h = 1..8 for every variable; acceptance, ESS, R-hat. |

**MCSE** of a quantile is the largest of (i) the spread of batch quantiles, (ii) the delta method
(sparsity `dQ/dp` times the batch-means standard error of `1{x ≤ q}`), and (iii) the
between-chain spread; the Python tests estimate their own MCSE with (i) and (ii).

**Tolerances** (`tests/parity/test_r_bvar.py`): log ML at fixed `h` 1e-8 relative (BVAR and the
exact values agree to < 1e-11); NIW parameters 1e-8 relative to the largest entry of each matrix
(BVAR's own normal-equation numbers deviate by up to 6e-9 on that scale); hyperparameter modes
1e-3 relative (targets accurate to ~1e-5); Monte Carlo quantiles within
`4 · sqrt(MCSE_R² + MCSE_ours²)`. At `h = 1` the conjugate predictive is an exact multivariate t
(`ν̄ − n + 1` d.o.f., location `B̂'x`, scale `S̄ (1 + x'Ω̄x) / (ν̄ − n + 1)`); both the R
simulation and rai-bvar's forecasts are also checked against it.
