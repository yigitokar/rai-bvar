# Bank of England `bvar` as a parity oracle

Notes behind `test_boe_bvar.py` (SPEC §1.4, §4.6, §10.4; decision D5). Everything below was read
from the source of **`bvar` 0.3.5** (PyPI, MIT; github.com/bank-of-england/bvar, tag `v0.3.5`)
and checked numerically. Where the BoE documentation and code disagree, the code is
authoritative (see §11).

- Install: `uv pip install --python .venv/bin/python bvar` (not added to `pyproject.toml`).
- Runtime dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `numba` (pulls in
  `llvmlite`). Optional extra `bvar[jax]` adds a JAX backend for the optimiser; it is not
  installed here, and the tests force the NumPy backend.
- Installing it does not change numpy 2.5.4 / scipy 1.18.1 / pandas 3.0.6.
- `test_boe_version_is_mapped` xfails when the installed version differs, as a reminder to
  re-audit these notes.

Notation: `n` variables, `p` lags, `T` effective observations, `k = 1 + np` regressors,
`d = ν₀` IW degrees of freedom.

## 1. Model setup and regressor layout

`y_t = c + A_1 y_{t−1} + … + A_p y_{t−p} + ε_t`, `ε_t ~ N(0, Σ)`, written as `Y = Z A + E`.

- `construct_Y_Z`: `Y = data[p:]`, `Z_t = [1, y_{t−1}', …, y_{t−p}', covid dummies']`. The
  constant comes first, then lag 1 … lag p, each block in variable order. This is **the same
  layout as `rai-bvar`** (`deterministic="c"`), with `A` (k × n) = our `B`.
- Coefficient draws are stored as `beta`, shape `(draws, n·k)`, equation-major:
  `beta.reshape(n, k).T` gives `(k, n)`. Σ is stored as `sigma`, shape `(draws, n²)`.
- The only deterministic term is a constant (no `"ct"` or `"n"`). At least two variables are
  required. The input must be a DataFrame with a regular `PeriodIndex` or `DatetimeIndex`.
- `stationary` is one flag for the whole system. `False` puts prior mean 1 on every own first
  lag, and `True` puts 0 on every one. There are no per-variable means, so mixed I(0)/I(1)
  systems cannot be mapped. With `stationary=True` the SOC/SUR dummies are silently disabled.

## 2. Minnesota prior covariance (conjugate model, `models/conjugate/minnesota.py`)

`vec(A) | Σ ~ N(β₀, Σ ⊗ V_A)`, with `V_A` diagonal (k × k) and the same for every equation.
The prior covariance of each coefficient is the corresponding `V_A` entry times `Σ_ii`.

| Regressor | `V_A` entry (a variance) | BoE default |
| :--- | :--- | :--- |
| constant | `lambda_constant` | **10** |
| lag `l` of variable `j` | `c1² / (l^{c3} · s_j)`, `s_j = S₀[j,j] / (ν₀ − n − 1)` | `c1 = 0.2`, `c3 = 2` |
| COVID dummy | `lambda_covid` | 10 000 |

- `c1` = λ (overall tightness, entering as λ²), `c3` = α (lag decay `l^{−α}`).
- `s_j` is ψ_j / (d − n − 1) on the regressor variable `j`, so this is exactly GLP's
  `λ² / (l^α ψ_j/(d−n−1))` and our parametrisation.
- Own and cross lags have the same tightness. The docs say cross lags are shrunk harder, but
  that applies only to `IndependentNIW` (§10).
- `minnesota=False` gives a flat prior: `β₀ = 0`, `V_A⁻¹ = 1e-10·I`.

Pinned by `test_boe_minnesota_prior`, which uses ν₀ ≠ n+2 to expose the `(ν₀ − n − 1)` scaling.

## 3. Inverse-Wishart prior and ψ

`Σ ~ IW(S₀, ν₀)` with `ν₀ = n + 2` and `S₀ = diag(s) · (ν₀ − n − 1)`, so `E[Σ] = diag(s)`.

- `s` is **always** the per-variable AR(1) residual MSE (`models/common.py::ar1_mse`): OLS of
  `y_t` on `(1, y_{t−1})` over all `T_total − 1` pairs of the supplied data, including the `p`
  initial observations. The divisor is `T_total − 1`, i.e. a mean of squares with no degrees-of-
  freedom correction. Pinned by `test_boe_psi_is_ar1_mse`.
- `sample()` and `optimise_hyperparameters()` both recompute `S₀` from the data, so **ψ cannot
  be set by the user and is never estimated**.
- `ν₀` can be changed (`model.pars.nu_0 = …` after `set_priors`). `S₀` then rescales so that
  `E[Σ]` stays `diag(s)`.
- Mapping: our ψ (the IW scale diagonal, in variance units) is `diag(S₀) = s · (ν₀ − n − 1)`,
  which equals `s` at the default `d = n + 2`. The tests read `diag(S₀)` from BoE and pass it
  as `Minnesota(psi=...)`.
- Our `psi="ar"` is the AR(**p**) residual variance with a dof-corrected divisor `T − p − (p+1)`,
  computed on the estimation window. It is not BoE's value, so ψ is always passed explicitly.

## 4. SOC / SUR dummy observations (`dummy_observations.py`)

| Dummy | `Y_d` | `Z_d` |
| :--- | :--- | :--- |
| SOC, `n` rows, `mu` = μ | `diag(ȳ₀)/μ` | `[0, diag(ȳ₀)/μ, …, diag(ȳ₀)/μ, 0_covid]` |
| SUR, 1 row, `theta` = δ | `ȳ₀'/δ` | `[1/δ, ȳ₀'/δ, …, ȳ₀'/δ, 0_covid]` |

The functional form is GLP's. The rows are stacked as `[SUR; SOC; data]`; the order is
irrelevant.

**Difference: the initial mean ȳ₀.** BoE's builders receive the post-lag `Y = data[p:]` and
average its first `p` rows:

- BoE: **ȳ₀ = mean of observations p+1 … 2p**.
- GLP (2015) and `rai-bvar` (`ConjugateProblem.ybar0 = values[:p].mean(0)`, SPEC §4.6): the
  first `p` observations, i.e. the pre-sample initial conditions.

For variables not in levels BoE uses the full-sample mean, but SOC/SUR are disabled in that case
anyway. `test_boe_dummy_initial_mean_convention` pins BoE's convention.

The difference is material. On the QD levels data it moves log p(Y | h) by 0.1–1.1 nats
(qd-p4-sur −0.10, soc-sur +0.23, p2-soc-sur +0.70, soc +1.07). Every SOC/SUR comparison
therefore wraps BoE's two builders (`monkeypatch` on `bvar.dummy_observations`) so that they
receive `data[:p]`. BoE's own code then builds the rows, and BoE's sampler, ML objective and
optimiser all use them, because `stack_dummies` resolves the builders through the module
namespace. Nothing else in BoE is modified.

## 5. Marginal likelihood (`models/conjugate/marginal_likelihood.py`)

`log_marginal_likelihood` is exactly the closed form in GLP (2015, App. A) (`d = ν₀`, `Ψ = S₀`,
`Ω = V_A`, `B̂` the posterior mean, `ε̂` its residuals):

```
log p(Y|h) = −nT/2·log π + Σ_{i=0}^{n−1}[lnΓ((T+d−i)/2) − lnΓ((d−i)/2)] − T/2·Σ log ψ
             − n/2·log|I + V_A Z'Z| − (T+d)/2·log|I + Ψ^{−1/2}(ε̂'ε̂ + (B̂−b)'V_A⁻¹(B̂−b))Ψ^{−1/2}|
```

- With dummies, `objective_function` computes `log p([Y_d; Y]) − log p(Y_d)`, as GLP do. The
  `−nT/2·log π` terms difference to the data `T`.
- No hyperprior terms are included unless `add_priors=True` (see §6).
- **Documented constant between BoE and `rai-bvar`: zero.** `test_boe_log_ml_is_glp_formula`
  checks BoE against an independent QR plus `multigammaln` implementation to 1e-9 (measured
  ≈ 1e-11). Our `fit.log_ml()` matches BoE to ≤ 1.5e-11 relative.
- It is computed with Cholesky factors and residuals, so it is accurate even on level data.
- On a failed factorisation it returns the sentinel **−1e5**. The tests assert this value
  never appears.
- There is no public "log ML at given h" API. The tests call
  `objective_function(inv_softplus(h), …)`, which is the function BoE's optimiser minimises.

## 6. Hyperparameter optimisation (`optimisation_method="ml"`, `tune_priors`)

- **Free hyperparameters**: always `c1` (λ) **and `c3` (α)**, plus `mu` if `model.soc` and
  `theta` if `model.sur`. The layout follows the model flags, so with `stationary=True` the μ
  and δ parameters still exist and simply sit at their hyperprior modes. **α cannot be held
  fixed**, and **ψ is never estimated**.
- **Hyperpriors** (`add_priors=True` by default): normalised Gamma log densities on the natural
  scale, with no Jacobian, parametrised by mode and sd using GLP's `GammaCoef`:

  | | λ (`c1`) | α (`c3`) | μ | δ (`theta`) |
  | :--- | :--- | :--- | :--- | :--- |
  | BoE (mode, sd) | 0.2, 0.4 | **2, 0.5** | 1, 1 | 1, 1 |
  | `GLPHyperpriors` default | 0.2, 0.4 | 2, **0.25** | 1, 1 | 1, 1 |

  BoE has no ψ prior. GLP's IG(0.02², 0.02²) on ψ/(d−n−1) is irrelevant because BoE does not
  estimate ψ.
- **Optimiser**: `scipy.optimize.minimize(method="BFGS", maxiter=1000)`, unbounded, on
  `x = softplus⁻¹(h)`.
  - Default start: `softplus(0 + N(0, 0.1²)) ≈ 0.69` for every hyperparameter (random unless
    `initial_values` or `random_state` is given). `nb_restart` adds perturbed restarts.
  - Backend `"auto"` uses JAX autodiff when JAX is installed. Otherwise it uses NumPy with
    **forward-difference gradients**.
  - There is no MH step (GLP's MCMC over h is absent) and no bounds. `rai-bvar` uses L-BFGS-B
    on log h with bounds (λ ∈ [1e-4, 5], α ∈ [1, 3], μ, δ ∈ [1e-4, 50]).
- **Accuracy of BoE's NumPy optimiser on level data is poor.**
  - Cause: the log-ML has rounding noise of about 5e-9 nats because cond(V⁻¹ + Z'Z) ≈ 1e10. A
    forward difference with step 1.5e-8 turns that into gradient noise of order 0.3, and BFGS
    stops with "precision loss".
  - Measured relative error of BoE's result against the polished mode of its own objective:
    QD p=4, (λ, α): 2.8e-3. With SOC+SUR: λ 5.9e-3, α 6.1e-3, μ 1.1e-2, δ 2.6e-2.
  - On the simulated data it is accurate (< 1e-6).
- The tests therefore compare our mode with the **mode of BoE's objective**. They run BoE's own
  optimiser, then polish its result with restarted Nelder–Mead on log h, which is reproducible
  to about 3e-5. A wide check (10%) confirms that BoE's raw answer is the same mode.
- Two configurations are tested:
  1. GLP baseline: `hyper="auto"`, α = 2 fixed, free set (λ[, μ, δ]). BoE's objective is
     maximised with `c3` held at 2, because `tune_priors` cannot do this itself.
  2. BoE default: `hyper=("lambda", "alpha"[, "soc", "sur"])` with
     `GLPHyperpriors(alpha=GammaPrior(2, 0.5, lower=0.1, upper=10))`.
- `optimisation_method="cross_validation"` (grid search on out-of-sample predictive likelihood
  over c1, c3, μ, δ) has no `rai-bvar` counterpart and is not compared.

## 7. Posterior and sampling (`NaturalConjugate._sample_direct`)

- `K_A = V_A⁻¹ + Z'Z` and `A_post = K_A⁻¹(V_A⁻¹A₀ + Z'Y)` (normal equations plus Cholesky).
  `S_post = S₀ + A₀'V_A⁻¹A₀ + Y'Y − A_post'K_A A_post` and `ν_post = ν₀ + T + (#dummy rows)`.
  The dummy rows count as observations, as in GLP.
- `beta_point` = `A_post`. `sigma_point` = `S_post / (ν_post − n − 1)`, the posterior mean of Σ.
- Draws: `Σ ~ invwishart.rvs(ν_post, S_post)`, then `A = A_post + chol(K_A)'⁻¹ U chol(Σ)'`. The
  draws are iid.
- Mapping: `post.mean ↔ A_post`, `post.scale ↔ S_post`, `post.df ↔ ν_post`, and
  `post.omega() ↔ K_A⁻¹`.
- Numerical accuracy on level data (QD, cond(K_A) ≈ 3e10):
  - BoE's `A_post` is correct to about 2e-9 per column (normwise). Its elementwise error reaches
    3e-6 on near-zero entries. A QR solve of the prior-augmented system is within 5e-10 of an
    exact rational solve.
  - BoE's `S_post` formula cancels: up to 4e-6 relative error elementwise, 4e-9 on the
    correlation scale.
  - The tests therefore compare:
    - B̄ with rtol 1e-8 plus a column-scaled atol of 1e-8·max|B̄_{·j}| (measured ≤ 2.9e-9).
    - S̄ with the residual form `S₀ + Ê'Ê + (B̂−A₀)'V⁻¹(B̂−A₀)` built from BoE's own `A_post`,
      prior and design, which is first-order insensitive to BoE's rounding: rtol 1e-8
      (measured ≤ 6e-11).
    - ν̄ exactly.
    - `sigma_point` on the correlation scale at 1e-7.
    - Ω̄ against `inv(K_A)` on the correlation scale at 1e-8 (measured ≤ 4e-9).
  - On the simulated (stationary) data everything agrees to about 1e-15.

## 8. Forecast simulation (`forecast/mixin.py`, `forecast/matrices.py`)

- `forecast(H, N_draws)` with no constraints draws **one path per posterior draw**, using
  Gaussian shocks `N(0, Σ^{(i)})` in companion form (`b + M'ε`, numba-compiled). The paths
  capture both parameter and shock uncertainty, which is the same predictive distribution as
  `fit.forecast`.
- `recursive_forecast` is an equivalent loop. `point_only=True` gives a plug-in path at the
  posterior means.
- Output `forecast_unconditional` has shape `(draws, T + H, n)`, with history first. Data
  transforms (`data_transformation`, qoq, yoy) are a post-processing layer and are not used.
- Compared: quantiles (0.05, 0.16, 0.5, 0.84, 0.95) at h = 1…8 for 5000 BoE paths against 5000
  of ours.
  - The tolerance is |z| < 5, using the standard error of a difference of two sample
    quantiles, with 1/density estimated by Siddiqui's difference quotient (±0.025) on BoE's
    paths.
  - Calibration: BoE against BoE with different seeds gave max|z| of 2.0–3.7 over the
    120 comparisons. Ours against BoE gave 2.7 (QD p4 SOC+SUR) and 3.5 (sim p2).
- Numba compiles on the first `forecast` call (about 3.5 s). That is most of the module's
  runtime of about 8–10 s.

## 9. COVID dummies

`covid=True` adds one impulse-dummy regressor per period in `covid_dates` (default
2020-01-01 … 2021-12-31, so 8 quarterly dummies), following Cascaldi-Garcia (2022).

- Each dummy has prior variance `lambda_covid = 1e4` (× Σ_ii) and prior mean 0. The dummies
  also enter the AR(1) regressions that set ψ.
- `rai-bvar` instead uses Lenza–Primiceri scaling (`s_t² Σ`, SPEC §4.6) or `"exclude"`. Impulse
  dummies with a finite prior variance still count in ν̄ and in the ML, and their observations
  remain lags for later rows, so neither option is an exact map.
- **Not compared.** Both fixtures end in 2019Q4.

## 10. Other BoE features (not compared; no `rai-bvar` counterpart yet)

- Conditional forecasts: hard and soft constraints, plus skew-normal constraints.
  - Methods: Waggoner–Zha / Andersson et al. (default), Antolín-Díaz et al., Labonne–Renzetti.
  - Gibbs over (forecast path, parameters), with the posterior re-drawn from the augmented data.
  - `rai-bvar` conditional forecasts are Phase 3 (SPEC §6.2).
- Generalised IRFs, forecast revision / counterfactual analysis, and nowcast-uncertainty priors.
- `IndependentNIW`: an independent Normal–IW Gibbs sampler.
  - Prior variances are **not** scaled by Σ: `c1²·(σ_i/σ_j)²·(1 own | c2 cross)·l^{−c3}`, with
    `c2 = 0.5` and a Gamma(c2, 0.5) hyperprior. The constant gets `lambda_constant`.
  - It is closest to our MCMC `Linear(form="reduced") × Constant`, but the priors differ and
    there is no ML. Not compared.
- `nu_0 ≠ n + 2`: our conjugate engine fixes `d = n + 2`.

## 11. BoE docs vs code (code wins)

- The docs say "cross-variable coefficients are shrunk more aggressively". This is false for
  `NaturalConjugate` (Kronecker structure, §2).
- `methods/bvar_framework.md` writes `[V_A⁻¹] = c1²/(ℓ^{c3} s_i)`, i.e. a precision indexed by the
  equation `i`. The code sets the **variance** to `c1²/(ℓ^{c3} s_j)`, indexed by the regressor
  variable `j`.
- The docs give the SUR dummy `Z_d` as `[1/θ, ȳ' … ȳ']/θ`. The code uses `[1/θ, ȳ'/θ, …]` (§4),
  and the SOC row is consistent with the docs.
- The docs say "initial sample means" for ȳ. The code uses observations p+1…2p (§4).
- The docs say "L-BFGS … over a bounded parameter space" and that the free hyperparameters are
  (λ, μ, θ). The code uses unbounded BFGS in softplus space and always includes `c3` (§6).
- The docs do not state the defaults `lambda_constant = 10` or `c3_sd = 0.5`.

## 12. Parametrisation map

| `rai-bvar` | BoE `bvar` 0.3.5 | Notes |
| :--- | :--- | :--- |
| `VARData.from_frame(frame, tcodes=1)` | `frame` (PeriodIndex) | BoE takes data as supplied |
| `lags=p, deterministic="c"` | `BVAR(n_lags=p)` | constant only |
| `Minnesota(mean=1.0)` / `mean=0.0` | `stationary=False` / `True` | system-wide only |
| `lambda_` | `set_priors(c1=…)` | |
| `alpha` | `set_priors(c3=…)` | always estimated by BoE's ML |
| `var_const` | `set_priors(lambda_constant=…)` | **defaults differ: 1e7 vs 10** |
| `soc=μ` / `None` | `NaturalConjugate(soc=True/False)`, `set_priors(mu=μ)` | BoE default `soc=True`; ȳ₀ differs (§4) |
| `sur=δ` / `None` | `NaturalConjugate(sur=True/False)`, `set_priors(theta=δ)` | BoE default `sur=True`; ȳ₀ differs (§4) |
| `psi=(ψ…)` | not settable: `diag(S₀)` = AR(1) MSE·(ν₀−n−1) | pass BoE's value explicitly |
| `d = n + 2` | `nu_0 = n + 2` (settable) | compare at n+2 only |
| `hierarchy="fixed"` | `optimisation_method="none"` + `set_priors` | |
| `hierarchy="eb"`, `hyper=("lambda","alpha"[,"soc","sur"])` | `optimisation_method="ml"` | α prior: set `sd=0.5` on ours |
| `hierarchy="eb"`, `hyper="auto"` (α fixed) | no equivalent | BoE objective with c3 held fixed |
| `hierarchy="mh"` | no equivalent | |
| `fit.log_ml(h)` | `−objective_function(…, add_priors=False)` | constant = 0 |
| `fit.log_posterior(h)` | `−objective_function(…, add_priors=True)` | needs the same free set and α prior |
| `post.mean / scale / df / omega()` | `beta_point.reshape(n,k).T` / `S_post` / `ν₀+T*` / `K_A⁻¹` | `sigma_point = S_post/(ν_post−n−1)` |
| `fit.draw_parameters` | `beta`, `sigma` after `sample()` | iid in both |
| `fit.forecast(H, draws)` | `forecast(H, N_draws)` → `forecast_unconditional[:, -H:]` | one path per draw |
| `outliers="lenza-primiceri"` | `covid=True` (impulse dummies) | not mappable |

## 13. Differences from GLP (2015), summarised

| | GLP (2015) | BoE `bvar` | `rai-bvar` |
| :--- | :--- | :--- | :--- |
| ȳ₀ for SOC/SUR | first `p` obs | **obs p+1…2p** | first `p` obs |
| ψ | estimated, IG(0.02², 0.02²) on ψ/(d−n−1) | **fixed**, AR(1) MSE (divisor T−1) | `"ar"` = AR(p) (dof-corrected), or given, or estimated |
| α | fixed at 2 (baseline) | **estimated**, Gamma(2, 0.5) | fixed by default; Gamma(2, 0.25) when free |
| constant variance | 1e7 (flat) | **10** (× Σ_ii) | 1e7 |
| search | max posterior, then MH | BFGS max posterior only, unbounded | L-BFGS-B max posterior (bounded); MH |
| prior mean | per variable | all 1 or all 0 | per variable (tcode-aware) |
| COVID | n/a (Lenza–Primiceri 2022 later) | impulse dummies | Lenza–Primiceri scaling |

## 14. What the tests do and do not compare

Compared (both datasets; lags 4 and 2; prior means 1 and 0; SOC, SUR and both on the QD level
data; non-default λ, α, μ, δ and var_const in the fixed cases):

- Log ML at fixed h with explicit ψ: rtol 1e-8 (measured ≤ 1.5e-11). A 1e-6 relative change in
  ψ already moves the log ML by 2.6e-8, so the check has power.
- `log_posterior` and `log_ml` at given h with BoE's hyperpriors (rtol 1e-8).
- Posterior B̄, S̄, ν̄, E[Σ], Ω̄ (§7).
- Draws: the means of 4000 of our iid draws against BoE's B̄ and E[Σ] within 5 MCSE.
- Hyperparameter modes: λ (and μ, δ; and α in the BoE-default configuration) within 1e-3
  relative of the mode of BoE's objective. In addition, BoE's objective at our mode may not
  exceed BoE's minimum by more than 1e-4.
- Forecast quantiles within Monte Carlo tolerance (§8).

Not compared, with reasons:

- BoE's SOC/SUR **as shipped** (different ȳ₀; pinned instead).
- `psi="ar"` defaults (different AR order and divisor).
- BoE's raw optimiser output at 1e-3 (it is not that accurate on level data; checked at 10%).
- COVID dummies, conditional forecasts, GIRFs, cross-validation, `IndependentNIW`, MH over h,
  and `nu_0 ≠ n+2`.

Margins worth watching:

- B̄ on level data (measured 2.9e-9 against a 1e-8 column-scaled tolerance; limited by BoE's
  normal equations).
- δ in the BoE-default SOC+SUR mode: our optimum is 6.5e-4 relative from the polished mode,
  against a tolerance of 1e-3. The gap comes from the engine's L-BFGS-B stopping rule (its
  objective is 1.6e-6 nats short of the maximum), not from BoE.
