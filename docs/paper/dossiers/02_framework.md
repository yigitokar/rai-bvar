# Dossier 02: Model framework and user-facing API

Scope: the composition model (mean × volatility × prior), the priors, the volatility arms, engine
resolution, the data layer, forecasting and structural analysis, the predictive contract and the
evaluation/verdict layer of `rai-bvar`, Phases 0–2. Later phases appear only as roadmap. Other
dossiers cover the samplers' validation (Geweke/SBC/mutation), performance and the empirical
replication in depth; this one points to them where the framework depends on them.

---

## 0. Provenance and conventions

**Code base.** Repository `/Users/yigitokar/rai-bvar-paper`, branch `paper-draft`, commit
`62d15f68c7e7a3de4ce83ee0098293b9c62111b0` (2026-10-23 06:44 −0400). Every file path below is
relative to the repository root, and `file:a-b` means lines a–b of that file at this commit.

**Environment of every command run for this dossier.** macOS-26.6.2-arm64 (Apple M5 Max), Python
3.12.13, rai-bvar 0.0.1, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, xarray 2026.9.0, scoringrules
0.11.0, jax 0.12.0.1. Commands were run as
`cd /Users/yigitokar/rai-bvar-paper && PYTHONDONTWRITEBYTECODE=1 uv run --frozen …`, with
`pytest -p no:cacheprovider`: no tracked file was written, bytecode and pytest caches were
disabled, and `git status` afterwards showed only the untracked dossier directory (`.hypothesis/`
and `.venv/` pre-existed and are gitignored).

**Status labels.** They follow `docs/paper/NOTES.md` (header):

- **verified**: reproducible from a committed test or fixture. I add *(re-run here)* when I ran
  that test on 2026-10-23 for this dossier (§0.1).
- **recorded**: a number or claim from a run that is written down in a committed file but was not
  re-run here.
- **to check**: must be re-run or re-derived, or is a SPEC/code discrepancy to resolve before
  publication.

Plain descriptions of what the API does carry a code citation `[path:lines]`. These are read
from the source at the commit above. They carry a status label only when a number or a
non-obvious property is claimed.

### 0.1 Test runs done for this dossier (2026-10-23)

| Command (all with `PYTHONDONTWRITEBYTECODE=1 uv run --frozen pytest -p no:cacheprovider -q`) | Result |
| :--- | :--- |
| `tests/unit/test_api_resolution.py tests/unit/test_grid_harmonise.py tests/contract/test_forecast_contract.py tests/contract/test_forecast_contract_mcmc.py tests/analytic/test_structural_nig.py tests/priors/test_structural_prior.py tests/integration/test_compare.py tests/unit/test_significance.py tests/unit/test_scores.py tests/analytic/test_conjugate.py tests/analytic/test_mcmc_twin.py tests/unit/test_dataset.py` | 577 passed, 110 skipped (`slow`), 44.9 s |
| `-m slow tests/contract/test_forecast_contract_mcmc.py` | 100 passed, 34.2 s: 10 contract checks × the 10 slow-tier compositions; the other two compositions (`structural-SV(outliers)`, `reduced-Constant(ig)`, the `FAST` tuple at `tests/contract/test_forecast_contract_mcmc.py:46`) are not `slow` and ran in the first command, so the two commands together cover all 12 (corrected by fact-check; `--collect-only -m slow` gives 100/120 collected, 20 deselected) |
| `tests/parity/test_r_bvar.py tests/parity/test_boe_bvar.py tests/analytic/test_eb_mcmc.py tests/integration/test_backtest.py tests/integration/test_backtest_mcmc.py tests/unit/test_outliers.py tests/unit/test_irf.py tests/mcmc/forecast/test_lpd_mcmc.py tests/mcmc/forecast/test_irf_mcmc.py tests/unit/test_phase1_hash_golden.py tests/unit/test_spec_roundtrip_mcmc.py tests/unit/test_specs_phase2.py` | 694 passed, 9 skipped, 70.9 s |

The slow tiers of `test_mcmc_twin.py` (A2 other paths, A3/A4 realistic size, A1-dummies at
realistic size) and the slow tests of `test_backtest_mcmc.py` were **not** re-run.

---

## 1. Design principles that shape the API

SPEC §3 states nine rules. Five of them shape the user-facing API directly:

| Rule | Content | Where it lives in code |
| :--- | :--- | :--- |
| R1 Composition over inheritance | `BVAR(mean=..., vol=..., lags=...)`; no model-specific classes | `src/rai_bvar/api.py:68-83` (the frozen dataclass `BVAR` with fields `mean`, `vol`, `lags`, `deterministic`, `outliers`, `mcmc`) |
| R2 Volatility first (fair ablation) | Mean effects are measured holding the volatility arm fixed; `compare()` issues a verdict only between models that differ along exactly one axis | `src/rai_bvar/evaluation/compare.py:212-244` (pairing and the "not comparable" rule) |
| R3 One predictive contract | Analytic and MCMC models return the same `ForecastResult` | `src/rai_bvar/forecast/result.py:96-103`; MCMC attaches only optional extras (`result.py:8-19`) |
| R8 Ordering dependence surfaced | Every triangular factorisation depends on the variable ordering; results record the ordering and an ordering-sensitivity check ships | `src/rai_bvar/structural/ordering.py:1-73`; the `ordering` attribute of every `IRFResult` (`src/rai_bvar/structural/irf.py:11-17`, `:195`), of MCMC forecasts (`src/rai_bvar/engines/mcmc/forecast.py:309`) and of `MCMCFit.summary().attrs` (`results.py:443`). Analytic forecasts have **no** `ordering` attr (`analytic.py:696-707`); there the order is only the `variable` coordinate (corrected by fact-check; see §15 D15) |
| R9 No look-ahead | Everything data-dependent is recomputed inside each backtest window | `src/rai_bvar/data/dataset.py:123-134` (`upto`), `src/rai_bvar/engines/mcmc/resolve.py:4-15`, `src/rai_bvar/priors/scales.py:1-15` |

R4 (float64, no explicit inverses, precision samplers, log-space densities), R5 (clean room),
R6 (Geweke and SBC before merge) and R7 (JAX Gibbs, no PPL) govern the internals (SPEC §3).

**Import-time rule.** `import rai_bvar` imports no JAX. Specs, settings and the analytic engine
are NumPy-only, and the MCMC engine is imported lazily when an MCMC composition is fitted
(`src/rai_bvar/__init__.py:1-9`, `:48-82`; DESIGN D-31). This is tested in a subprocess
(`tests/unit/test_api_resolution.py::test_mcmc_compositions_are_built_and_described_without_jax`;
`tests/mcmc/unit/test_import_no_jax.py`). Status: verified (the first test was re-run here,
the second was not).

---

## 2. Notation (and a conflict the paper must resolve)

The code and DESIGN use different symbols from SPEC in two places:

| Concept | SPEC | DESIGN / code / notes | Code field |
| :--- | :--- | :--- | :--- |
| Minnesota overall tightness | `λ` (`lambda_`) | `κ` (N05 §2: "written κ here to avoid a clash with λ_{i,t}") | `Minnesota.lambda_` |
| Structural variance of equation i at t | `λ_{i,t}` | `λ_{i,t}` (`h_{i,t} = log λ_{i,t}` under SV) | state `log_lam`, `h` |
| Lag decay | `alpha` | `α` | `Minnesota.alpha` |
| Cross-lag tightness (scales the **s.d.**) | `cross` | `κ_c` | `Minnesota.cross` |
| Contemporaneous prior scale | n/a | `κ_a` | `Minnesota.contemp` |
| Deterministic-term prior variance | n/a | `κ_d` | `Minnesota.var_const` (default `1e7`) |
| Equation index | 1-based `i` | 0-based `i` in code, 1-based `ι = i + 1` in prior formulas | (DESIGN "Conventions", lines 28-33; corrected by fact-check, was "26-31") |
| IW degrees of freedom of the conjugate prior | `d = n + 2` | `d = n + 2` | `ConjugateProblem.d` (`src/rai_bvar/engines/analytic.py:202`) |
| Inverse gamma | n/a | `IG(c, C)` shape `c`, **scale** `C`, density ∝ `λ^{−c−1}e^{−C/λ}` | N05 §2 table |

Below I write `κ` for the tightness, as the code does, and give the `lambda_` field name the
first time. The paper must choose one convention (open question Q1).

Dimensions: `n` variables, `p` lags, `n_d ∈ {0, 1, 2}` deterministic terms (`"n"`, `"c"`,
`"ct"`; `src/rai_bvar/data/design.py:13`), `k = n_d + np` regressors, `T` effective
observations.

---

## 3. The model

### 3.1 Regressors

$$
x_t = (d_t', y_{t-1}', \dots, y_{t-p}')' \in \mathbb{R}^k, \qquad k = n_d + np .
$$

Deterministic terms come first, then lags 1..p, each lag block in variable order. The trend is
the 1-based position in the estimation sample and continues into the forecast
(`src/rai_bvar/data/design.py:1-5`, `:31-44`). Lags and deterministic terms belong to the model,
not the data, so lag length can be an ablation axis (SPEC §5.1).

### 3.2 Recursive structural form (all MCMC models) and reduced form

SPEC §4.1; N05 §3. For equation `i`:

$$
y_{i,t} = -\sum_{j<i} a_{ij}\, y_{j,t} + x_t'\gamma_i + \sqrt{\lambda_{i,t}}\,\varepsilon_{i,t},
\qquad \varepsilon_{i,t} \overset{iid}{\sim} N(0,1),
$$

that is, `A y_t = Γ x_t + Λ_t^{1/2} ε_t` with `A` unit lower triangular and
`Λ_t = diag(λ_{1,t}, …, λ_{n,t})`. The reduced form is

$$
y_t = B'x_t + u_t,\qquad u_t \sim N(0, \Sigma_t),\qquad B' = A^{-1}\Gamma,\qquad \Sigma_t = A^{-1}\Lambda_t A^{-1\prime}.
$$

Under SV with transitory scales, `λ_{i,t}` is the **effective** variance
`e^{h_{i,t}} o_{i,t}^2 τ_{i,t}` (§6). In code, the mean block's weights are `1/λ_eff`
(`DESIGN §2.9`, `EqContext.weights`). Phase 2 has only the linear mean `g_i(x_t) = x_t'γ_i`.
SPEC writes the general `g_i(x_t; θ_i)` that BART/TVP will fill in later.

### 3.3 Likelihood

N05 §3. With `|A| = 1`,

$$
\log p(Y \mid A, \Gamma, \lambda) = \sum_{i=1}^{n}\sum_{t=1}^{T}\Big[-\tfrac12\log(2\pi\lambda_{i,t}) - \tfrac12\, e_{i,t}^2/\lambda_{i,t}\Big],
\qquad e_{i,t} = y_{i,t} + \sum_{j<i} a_{ij} y_{j,t} - x_t'\gamma_i .
$$

This is the same density as the reduced form `N(B'x_t, Σ_t)`, with no Jacobian term. It is the
**primary convergence monitor** (`src/rai_bvar/engines/mcmc/monitor.py:3-6`) and an SBC test
quantity.

### 3.4 Why this form

With priors independent across equations given the hyperparameters, and conditional on the
volatilities, the `n` equation posteriors are independent. Equation `i` is a heteroskedastic
regression with `k + i − 1` regressors, so one sweep costs `O(n k³) = O(n⁴p³)` instead of
`O(n⁶p³)` for joint sampling, and the equations run under `jax.vmap` (SPEC §4.1; N05 §0 item 1).
The 2019 equation-by-equation algorithm of Carriero, Clark & Marcellino conditioned equation `j`'s
coefficients only on the first `j` equations. SPEC §0 and §4.1 record that it targets the wrong
distribution (Bognanni 2022 [CITE?]). The package therefore uses this structural form, which is
correct by construction, or the corrected algorithm of Carriero, Chan, Clark & Marcellino (2022)
[CITE?] for reduced-form priors (§5.6).

---

## 4. Component catalogue delivered in Phases 0–2

User-facing specifications are frozen, pure-Python dataclasses
(`src/rai_bvar/components/specs.py:1-15`). The MCMC kernels live under
`src/rai_bvar/components/{structure,mean,coef,vol}/` and are loaded lazily through
`src/rai_bvar/engines/mcmc/registry.py` (DESIGN §1.1 rule 5).

| Axis | Delivered | Spec class / field | Roadmap (SPEC §4.4, §13) |
| :--- | :--- | :--- | :--- |
| Mean | `Linear` (`g(x_t) = B'x_t`) | `Linear(prior=Minnesota(...))` (`specs.py:40-53`) | BART (P4), TVP, Threshold, Markov-switching (P5) |
| Prior | `Minnesota` in three forms: `conjugate`, `structural`, `reduced` (plus `auto`); SOC/SUR dummies; hierarchy `fixed`/`eb`/`mh` | `src/rai_bvar/priors/minnesota.py:79-95` | Horseshoe, NG, DL, R2D2, SSVS, AsymmetricConjugate (P3) |
| Volatility | `Constant` with prior `auto`/`iw`/`ig`/`iw_equivalent`; `SV` (AR or RW) with optional outliers and/or Student-t | `specs.py:56-85` (`Constant`), `:107-153` (`SV`) | CommonSV (P3), FactorSV, order-invariant SV (P5) |
| COVID treatment | Lenza–Primiceri scaling (analytic engine only); `SV(outliers=True)` (MCMC) | `src/rai_bvar/data/outliers.py:27-36`; `specs.py:120-122` | LP scaling on MCMC (P3; DESIGN §3.9) |
| Benchmarks | `RandomWalk(drift=False)`, `OLSVAR(lags, uncertainty="flat"|"none")` | `src/rai_bvar/models/benchmarks.py:37-91` | — |

`rai_bvar.components` re-exports only `Linear`, `Constant` and `SV`
(`src/rai_bvar/components/__init__.py:8-10`). The SV hyperprior classes are exported as
`rb.sv_priors.*` (`Normal`, `Beta`, `ScaledChi2`, `InvGamma`, `DiscreteUniform`,
`OutlierPrior`). This avoids a clash with the GLP `BetaPrior`/`GammaPrior`
(`src/rai_bvar/__init__.py:44`; DESIGN §6.3).

---

## 5. Priors

### 5.1 Minnesota in conjugate form: the analytic engine (Phase 1)

`src/rai_bvar/priors/minnesota.py:1-17`; `src/rai_bvar/engines/analytic.py:1-32`, `:378-382`.

$$
\Sigma \sim IW(\Psi, d),\quad \Psi = \mathrm{diag}(\psi),\ d = n+2;\qquad
\mathrm{vec}(B)\mid\Sigma \sim N(\mathrm{vec}(b),\ \Sigma\otimes\Omega),
$$

with `Ω` diagonal. For the coefficient on lag `l` of variable `j`:

$$
\Omega_{(l,j)} = \frac{\kappa^2 (d-n-1)}{l^{\alpha}\,\psi_j}\ \ (= \kappa^2/(l^\alpha\psi_j)\ \text{at}\ d=n+2),\qquad
\Omega_{\det} = \kappa_d .
$$

`b` is zero except on the own first lag of each variable, whose mean is transform-aware (§9.3).
The Kronecker structure forces equal tightness on own and cross lags (Kadiyala & Karlsson 1997
[CITE?]), so `cross ≠ 1` and `contemp ≠ 1` raise in conjugate (and unresolved `auto`) form
(`minnesota.py:110-123`).

**Defaults** (`minnesota.py:81-95`): `lambda_=0.2`, `alpha=2.0`, `psi="ar"`, `soc=None`,
`sur=None`, `soc_scope="persistent"`, `mean=None` (from tcodes), `cross=1.0`, `contemp=1.0`,
`var_const=1e7`, `form="auto"`, `hierarchy="eb"`, `hyper="auto"`.

**ψ default.** `psi="ar"` is the residual variance of a univariate AR(p) with a constant, by
least squares on the window (`analytic.py:996-1041`). With LP scaling active it uses the periods
before `t*` only, provided at least `2p + 2` such periods exist; otherwise it uses the whole window
without the regression rows that touch `t* … t*+2`, and records a note (`analytic.py:253-271`;
fallback added by fact-check). It raises on a constant series or an exact AR fit, and
notes when fewer than 10 degrees of freedom remain.

**Posterior by one QR.** The Minnesota part enters as `k` dummy rows `Ω^{−1/2}(B − b)`, stacked
on `[X | Y]` with the SOC/SUR rows. One QR of `[X* | Y*]` gives `B̄ = R⁻¹Q'Y*` and
`S̄ = Ψ + R_yy'R_yy` (as a Cholesky factor). `Ω̄ = (R'R)⁻¹` is never formed, and the posterior
degrees of freedom are `ν̄ = d + rows` (`analytic.py:425-449`). Rank deficiency raises
(`analytic.py:436-437`). Draws use the Bartlett decomposition for `Σ` and `B = B̄ + R⁻¹ZC'`
(SPEC §4.6).

**Marginal likelihood** (GLP 2015 appendix; `analytic.py:8-15`):

$$
\log p(Y\mid h) = -\tfrac{nT}{2}\log\pi + \log\Gamma_n\!\big(\tfrac{T+d}{2}\big) - \log\Gamma_n\!\big(\tfrac d2\big)
- \tfrac n2\log|\Omega| - \tfrac n2\log|X'X+\Omega^{-1}| + \tfrac d2\log|\Psi| - \tfrac{T+d}{2}\log|\bar S|,
$$

with `S̄ = Ψ + ε̂'ε̂ + (B̂ − b)'Ω⁻¹(B̂ − b)`. All log-determinants come from triangular factors.

- With dummies: `ML(dummies + data) − ML(dummies)`. The dummies-only term uses the determinant
  lemma on its `m ≤ n + 1` rows (`analytic.py:451-467`, `:477-483`).
- With LP scaling: rows divided by `s_t`, plus the Jacobian `−n Σ_t log s_t` (`analytic.py:483`).

Evidence (all verified *(re-run here)*):

- `tests/analytic/test_conjugate.py::test_log_ml_satisfies_the_chib_identity`
- `::test_log_ml_with_dummies_is_the_difference_of_two_marginal_likelihoods`
- `::test_lenza_primiceri_jacobian_by_the_heteroskedastic_chib_identity`
- `::test_n1_p1_log_ml_and_moments_by_quadrature`

**Parity with R `BVAR` and the Bank of England `bvar`.** SPEC §13 "Status (2026-10-22)" reports
log marginal likelihoods equal to R `BVAR` (exact cases) and BoE `bvar` to about 1e-11 relative,
posterior moments to about 1e-9, EB modes to 1e-3, and MH quantiles within Monte Carlo error.
Status: the parity tests pass *(re-run here: `tests/parity/test_r_bvar.py`,
`tests/parity/test_boe_bvar.py`)*. The tolerance numbers are **recorded** (SPEC §13); I did not
read them back from the tests.

### 5.2 SOC and SUR dummy observations

`src/rai_bvar/engines/analytic.py:384-399`; `src/rai_bvar/priors/dummies.py:1-24`, `:52-83`.
Let `ȳ₀` be the mean of the first `p` observations of the window (GLP), `μ = soc` and `δ = sur`.

- **Sum of coefficients:** one row per SOC variable `j`, `y_d = e_j ȳ₀_j/μ`,
  `x_d = (0_{n_d}, y_d, …, y_d)`.
- **Single unit root:** one row `y_d = ȳ₀'/δ`, `x_d = (1/δ, y_d, …, y_d)`, where `1/δ` sits on the
  constant only (a trend gets 0).

**Design choice (`soc_scope`).** With `soc_scope="persistent"` (the default), SOC rows exist only
for variables whose own-lag prior mean is non-zero. A differenced variable is then not pulled
towards a unit root. `soc_scope="all"` keeps GLP's rows for every variable, as R `BVAR` and BoE
`bvar` do (SPEC §4.6; `minnesota.py:10-14`). The analytic engine's docstring lists it under
"Conventions (documented choices)" (`analytic.py:18-28`; SPEC §4.6 describes the rule without that
wording; attribution corrected by fact-check). It changes results relative to the reference packages
whenever the SOC dummy is on and differenced variables are present.

**On the MCMC engine (WP25).** The same rows are appended to every equation's regression as
prior rows (never as time steps). Their fixed weight depends on the prior mode:

| Mode | Row weight |
| :--- | :--- |
| Mode I, `iw_equivalent` | `1/σ̄_i² = ι/ψ_i` |
| Mode II | `1/λ_i`, like data rows, and the λ step counts the rows |
| reduced form | `1/ψ_j` |

Sources: `src/rai_bvar/priors/dummies.py:14-23`; DESIGN §2.10 "WP25 amendments". Evidence:

| Finding | NOTES item | Status |
| :--- | :--- | :--- |
| λ step must count the rows in Mode II | D.12 | recorded |
| closed-form/rejection reference for the dummy prior | D.13 | verified |
| O(q²) eigen path survives dense prior precision | D.14 | verified |
| "wrong weight" mutants are loud | D.15 | verified |
| bitwise neutrality without dummies | D.16 | recorded |

`tests/unit/test_api_resolution.py::test_soc_sur_dummies_run_on_mcmc` passes *(re-run here)*.

### 5.3 The GLP (2015) hierarchy: empirical Bayes and Metropolis–Hastings

**Hyperpriors** (`src/rai_bvar/priors/hyperpriors.py:206-217`):

| Hyperparameter | Prior (mode, s.d.) | Bounds |
| :--- | :--- | :--- |
| `λ` (κ) | Gamma(mode 0.2, s.d. 0.4) | [1e-4, 5] |
| `μ` (SOC) | Gamma(mode 1, s.d. 1) | [1e-4, 50] |
| `δ` (SUR) | Gamma(mode 1, s.d. 1) | [1e-4, 50] |
| `ψ_j/(d−n−1)` | Inverse-Gamma(0.02², 0.02²) | default mode/100 to mode·100 (`analytic.py:286-287`) |
| `α` (only when estimated) | Gamma(mode 2, s.d. 0.25) | [1, 3] |

Gamma priors are parametrised by mode and s.d. (`hyperpriors.py:35-65`). The free set is
`hyper="auto"`, i.e. `λ` plus whichever of `soc`/`sur` is switched on, or an explicit subset of
`("lambda", "alpha", "soc", "sur", "psi")` (`minnesota.py:43`, `:167-174`).

**EB (`hierarchy="eb"`, the default).** Maximise `log p(Y | h) + log p(h)` by L-BFGS-B on log
values (logit for the COVID decay ρ), with central-difference gradients
(`analytic.py:742-772`). Two further rules:

- Convergence is judged by a scale-free predicted Newton gain `½ Σ g²/H_ii < 1e-6`
  (`analytic.py:68`, `:830-836`), not by a gradient threshold.
- A deterministic multistart moves one axis at a time along `soc`, `sur` and `covid_rho`, where
  secondary modes were found in review (`analytic.py:791-812`; SPEC §13 Phase 1 note).

**MH (`hierarchy="mh"`).** Random-walk Metropolis on the unconstrained hyperparameters, with the
log-Jacobian in the target:

- The proposal starts from the inverse Hessian at the mode. The eigenvalues of the Hessian of
  the negative log target are floored at 1, so no direction gets a proposal s.d. above one unit on
  the unconstrained scale; with more than 20 free hyperparameters only the diagonal curvature is
  used (`analytic.py:957-993`; wording corrected by fact-check, the floor is on the Hessian, i.e. a
  cap on the inverse Hessian). The proposal is re-estimated at ½ and ¾ of burn-in with shrinkage,
  only when `burn ≥ 200` (`analytic.py:886-887`). The step adapts to a 0.3 target acceptance during
  burn-in only (`analytic.py:865-918`, `:934-955`).
- Given each `h` draw, `(B, Σ)` are drawn exactly.
- Converged means at least two chains (`chains > 1`) and every free hyperparameter has
  rank-normalised split R-hat < 1.01 and bulk ESS > 400 (`MHSettings`, `minnesota.py:46-61`;
  `analytic.py:921-929`; the `chains > 1` condition added by fact-check).

**Phase 1 cost** (SPEC §13 Phase 1 table; simulated `n = 100`, `p = 13`, `T = 300`, laptop CPU):

| Hierarchy | Fit | Forecast (1000 draws × 8 steps) |
| :--- | :--- | :--- |
| `fixed` | 0.1 s | 3.7 s |
| `eb`, λ only | 2.1 s | 3.6 s |
| `eb`, λ, μ, δ | ~100 s (≈35 s on FRED-QD) | 3.6 s |

Status: recorded.

### 5.4 Lenza–Primiceri COVID scaling (analytic engine)

`src/rai_bvar/data/outliers.py:1-12`, `:79-88`; SPEC §4.6.

$$
y_t = B'x_t + s_t\varepsilon_t,\qquad
s_t = \begin{cases}1 & t<t^*\\ s_0, s_1, s_2 & t = t^*, t^*{+}1, t^*{+}2\\ 1+(s_2-1)\rho^{\,j-2} & t = t^*{+}j,\ j\ge 3\end{cases}
$$

- `s_t` is one scalar common to all variables. Rows of `Y` and `X` are divided by `s_t` (GLS),
  which preserves conjugacy.
- `(s_0, s_1, s_2, ρ)` join `h`. The scales have Pareto(1, 1) priors truncated at
  `ParetoPrior.upper = 1000`; ρ has a Beta prior with mode 0.8 and s.d. 0.2
  (`outliers.py:33-36`; `hyperpriors.py:170-203`).
- A parameter that no sample period informs is not estimated. EB keeps it at the prior mode; MH
  draws it from the truncated prior when forecasting, so the truncation sets the prior
  predictive variance there (`E[s²] = upper`; `hyperpriors.py:172-177`).
- `t*` defaults to the period containing March 2020 (`outliers.py:64-72`).
- With `hierarchy="fixed"`, LP needs `fixed=(s0, s1, s2, ρ)` (`api.py:107-114`).
- **On MCMC compositions LP raises `NotImplementedError`** (Phase 3). With `SV(outliers=True)`
  the message adds "double treatment" (`api.py:159-168`).

### 5.5 Minnesota in structural form: D3 centring, Mode I, Mode II and `iw_equivalent`

`src/rai_bvar/priors/structural.py:1-23`; DESIGN §6.1; N05 §4. This is the default for MCMC
(SPEC decision D3).

**D3 centring.** The familiar reduced-form prior mean `E[B] = M` is kept by centring `γ_i`
conditionally on `a_i`:

$$
\mathbb{E}[\gamma_i\mid a_i] = M\tilde a_i,\qquad \tilde a_i = (a_{i1},\dots,a_{i,i-1},1,0,\dots,0)'.
$$

The code works in **δ-coordinates**, `δ_i = γ_i − M_{<i} a_i` (unit Jacobian). There `a_i ⊥ δ_i`
a priori with diagonal precisions, and equation `i` becomes the regression

$$
y_i = (XM_{<i} - Y_{<i})\,a_i + X\delta_i + e_i,
$$

whose contemporaneous regressors are minus the prior-mean residuals of the earlier variables
(N05 §4.1). One shared design `F = [X | X M_c − Y_{:, :n−1}]` serves all equations (DESIGN §2.3).
N05 §4.2 shows that D3 is the **only** linear centring that keeps `E[B'|A] = M'` for every `A`.

**Mode I** (independent, moment-matched; the default for `Constant("ig")` and every SV arm). The
fixed scale is `σ̄_ι² = ψ_ι/ι`.

| Coefficient | Prior |
| :--- | :--- |
| `a_ij`, `j < ι` | `N(0, κ_a ψ_ι/(ι ψ_j))` |
| `δ_i`, own lag `l` | `N(m_{i,l}, κ²/(ι l^α))`, with `m_{i,1}` the transform-aware mean and `m_{i,l>1} = 0` |
| `δ_i`, lag `l` of variable `m ≠ i` | `N(0, κ² κ_c² ψ_ι/(ι l^α ψ_m))` |
| `δ_i`, deterministic terms | `N(0, κ_d ψ_ι/ι)` |
| `λ_i` (Constant) | `IG(1 + ι/2, ψ_ι/2)`, mean `ψ_ι/ι` |

Sources: DESIGN §6.1 table; `structural.py:9-12`; `src/rai_bvar/priors/sv_arrays.py:38-41`.

With `κ_c = κ_a = 1`, Mode I reproduces exactly three moments of the NIW prior: its reduced-form
mean (for every `A`), its marginal variances and `E[Σ] = diag ψ`. It is **not** the same
distribution (DESIGN §6.1; N05 §4.3). The "naive" choice `a_ij ~ N(0, ψ_i/ψ_j)` would instead
give `E[Σ_ii] = 2^{i−1}ψ_i` (N05 §0 item 5, after Chan, Koop & Yu 2024 [CITE?]). Evidence:

- `tests/priors/test_structural_prior.py::test_mode_i_matches_the_niw_moments_exactly_and_by_monte_carlo`:
  verified *(re-run here)*.
- `tests/integration/test_ordering.py::test_prior_mode_reproduces_the_cky_two_to_the_i_growth`:
  verified (in the fast suite; not re-run here).

With `κ_c < 1` the implied reduced-form cross-lag variance "leaks" by a factor
`κ_c² + (1 − κ_c²)s_jm`. `fit.summary()` reports the implied moments
(`structural.implied_reduced_form_moments`; `src/rai_bvar/engines/mcmc/results.py:417-447`;
`test_leak_ratio_closed_form_formula_and_monte_carlo`, verified *(re-run here)*).

**Mode II** (`structural × Constant("iw")`, the "NIW twin"):

$$
\lambda_i\sim IG(1+\iota/2,\ \psi_\iota/2),\quad a_i\mid\lambda_i\sim N\big(0,\ \lambda_i\kappa_a\,\mathrm{diag}(1/\psi_j)\big),\quad
\delta_i\mid\lambda_i\sim N(m_i,\ \lambda_i\Omega^{(i)}),
$$

with `Ω^{(i)}_{(l,m)} = κ² c_im (d−n−1)/(l^α ψ_m)`, `c_ii = 1`, `c_im = κ_c²` and
`Ω^{(i)}_det = κ_d` (`structural.py:13-15`). Three properties:

- **At `κ_c = κ_a = 1` it is the Phase 1 NIW prior.** This is the structural form of the
  conjugate prior (N05 §4.2, a proposition proved via the IW partition theorem; Chan 2022
  Prop. 1/Cor. 1 [CITE?]): under NIW, independently over `i`,
  `σ_i² ~ IG((d−n+i)/2, ψ_i/2)`, `a_i | σ_i² ~ N(0, σ_i² diag(1/ψ_{1..i−1}))` and
  `γ_i | a_i, σ_i² ~ N(Mã_i, σ_i²Ω)`.
- With `κ_c ≠ 1` it is Chan's (2022) asymmetric conjugate prior. Its closed form is in
  `src/rai_bvar/engines/structural_nig.py`.
- Mode II changes the **mean prior**. The prior axis label therefore gains `", scaling=conjugate"`
  (`api.py:215-219`), and `compare()` will not pair a Mode I and a Mode II arm along `vol`
  (`test_mode_two_prior_axis_carries_the_conjugate_scaling`, verified *(re-run here)*).

Evidence for the equivalence:

- `tests/analytic/test_structural_nig.py::test_a1_log_ml_equals_phase1`: the structural NIG log
  ML equals the Phase 1 NIW log ML to `abs=1e-10` across five cases (constant; SOC+SUR with
  own means (1, .5, 0, .9); trend+SUR; no deterministics + `soc_scope="all"`; n = 1) and
  perturbed hyperparameters (`test_structural_nig.py:255-297`). Verified *(re-run here)*.
- `tests/priors/test_structural_prior.py::test_mode_ii_prior_is_the_niw_prior`: KS tests on Σ
  functionals and coefficient marginals, 20,000 draws, `min p > 1e-3`. Verified *(re-run here)*.
- N05 §0 item 3 records a log-ML difference of exactly `0.0` against the Phase 1 engine with SOC
  and SUR at `n = 4`. Recorded.

This equivalence is what makes the end-to-end MCMC oracle (analytic twin A2) possible. It is
covered in the validation dossier.

**`iw_equivalent`** (`Constant("iw_equivalent")`, either form; Chan 2022 Prop. 1; N08 §0). It
combines the Mode II `a` prior, `a_ij | λ_i ~ N(0, λ_i κ_a/ψ_j)` and `λ_i ~ IG(1+ι/2, ψ_ι/2)`,
with the **Mode I** δ prior in structural form or the CCCM coefficient prior in reduced form
(`structural.py:16-17`).

- At `κ_a = 1` the implied `Σ ~ IW(diag ψ, n+2)` exactly
  (`test_iw_equivalent_sigma_is_inverse_wishart`, verified *(re-run here)*).
- In reduced form it is **order-invariant**; in structural form it is not, because the δ prior
  is per structural equation (DESIGN §6.1).
- It is a covariance-prior choice, so it appears on the `vol` axis as
  `"Constant(iw_equivalent)"`, not in the prior label (`specs.py:75-85`; erratum E10).

A "Mode III" (ψ-scaled, Chan–Koop–Yu convention) is not shipped. The implied-moments helper
reports its `(1 + κ_a)^{i−1}` growth (DESIGN §6.1; `test_mode_iii_growth_is_reported`).

### 5.6 Minnesota in reduced form: the corrected triangular sampler (CCCM 2022)

`src/rai_bvar/priors/reduced.py:1-16`; `src/rai_bvar/components/mean/linear_reduced.py`; N04
§2–§6; DESIGN §3.7.3, §4.3.

Model: `y_t = Π'x_t + A⁻¹Λ_t^{1/2}ε_t`. Prior: `π_j ~ N(m_j, V_j)` independently over equations
`j`. `V_j` is diagonal and moment-matched to the NIW marginal at `cross = 1`:

- own lag: `κ²/l^α`;
- lag `l` of variable `m ≠ j`: `κ² κ_c² ψ_j/(l^α ψ_m)`;
- deterministic terms: `κ_d ψ_j`.

The `(A, λ)` prior is the structural one (Mode I `a` and IG `λ`, or `iw_equivalent`), so **the two
forms differ only in the coefficient prior** (D-11; N04 §3.2).

Because `π_j` enters every later structural equation, the equations are coupled. `π_j` is drawn
from its exact full conditional in a Gibbs scan over `j` (N04 §5.3, the "combined-weight form"):

$$
K_j = V_j^{-1} + X'\mathrm{diag}(W_{\cdot j})X,\quad W_{tj}=\sum_i a_{ij}^2/\lambda_{it} = (\Sigma_t^{-1})_{jj};\qquad
b_j = V_j^{-1}m_j + X'\big(G_{\cdot j} + W_{\cdot j}\circ X\pi_j^{\text{cur}}\big),\quad G = (E\oslash\Lambda)A,\ E = UA'.
$$

`K_j` does not depend on `π_{−j}`, so all `n` precisions are factored in parallel before the
sequential scan (N04 §5.3; DESIGN §3.7.3). Then the rows `a_j | Π, λ_j` are independent
regressions of `u_j` on `−u_{<j}` (DESIGN §4.3).

The combined-weight form agrees with the joint SUR oracle on 125 random problems to a worst
relative error of 3.5e-13 (N04 §5.3). Status: recorded (cross-check script not committed as a
test). The kernel itself is tested against an exact system-wide oracle (`tests/mcmc/blocks/test_cccm.py`,
`swa_oracle.py`; validation dossier).

SPEC §4.1 states the cost relative to the flawed 2019 version: "roughly 3× slower … while still
about 13× faster than system-wide sampling" at `n = 20`. N04 §5.2 and §5.4 cite CCCM-WP §4/Fig. 1
for `3.25×` and `≈13×` (`docs/phase2/04_cccm_corrected.md` lines 220, 248; cross-check log line
522); N04 §0 only lists "§4 + Fig. 1 timings" among the sources read (section pointer corrected by
fact-check). Status: recorded (from the paper, not measured here).

`bayesianVARs` parity for the reduced-form sampler is **deferred** (no gfortran). The exact
system-wide Gibbs oracle replaces it (DESIGN §2.10, Stage 4 amendment (2)). Status: to check
before claiming parity in the paper.

### 5.7 Minnesota hyperparameters on the MCMC engine

DESIGN §6.5; D-12; `src/rai_bvar/engines/mcmc/eb.py`. Hyperparameters are **fixed during MCMC**:

| `hierarchy` | Behaviour on MCMC |
| :--- | :--- |
| `"fixed"` | values as given |
| `"eb"` (default) | plug-in mode from the **Phase 1 `AnalyticFit` on the conjugate twin** at `cross = contemp = 1`, on the same window |
| `"mh"` | `NotImplementedError` at construction (`api.py:169-173`) |

Two special cases of `"eb"`:

- `form="structural"` with `cross ≠ 1`: Chan's per-equation NIG log ML is maximised instead.
  `psi` in the free set raises (`api.py:174-184`).
- `form="reduced"` with `cross ≠ 1`: the `cross = 1` twin mode is used, and a note is recorded in
  `fit.hyper["eb_note"]` (erratum E5).

The rationale (DESIGN §6.5) bears on R2: the EB `λ` depends only on the data window, so
`Linear × Constant` and `Linear × SV` get the **same mean prior**. A sampled `λ` would move with
the volatility arm. At `κ_c = κ_a = 1` EB is true type-II ML for the Mode II twin; for Mode I and
SV it is a documented plug-in approximation (DESIGN §6.5 last paragraph). Evidence:
`tests/analytic/test_eb_mcmc.py::test_eb_plug_in_equals_the_phase1_eb_mode_at_cross_1`, verified
*(re-run here)*. The runnable example (§13) shows the plug-in in `fit.summary().attrs["hyper"]`
(`'eb_method': 'conjugate_twin'`).

---

## 6. Volatility arms

### 6.1 `Constant`

`specs.py:56-85`; `src/rai_bvar/components/vol/constant.py`; DESIGN §3.7.5.

- Mode I (`"ig"`): `λ_i ~ IG(1 + ι/2, ψ_ι/2)`, independent of the coefficients.
- Mode II / `iw_equivalent`: the λ-scaled prior blocks enter the IG conditional through one
  sanctioned "coupling channel":

$$
\lambda_i\mid\cdot \sim IG\Big(c_i + \tfrac{T}{2} + \tfrac{n_{\text{active}}}{2},\ C_i + \tfrac12\sum_t e_{i,t}^2 + \text{quad}\Big),
$$

  with `n_active = k + i` (Mode II) or `i` (`iw_equivalent`), and `quad` the prior quadratic
  form of the scaled block (DESIGN §4.7, D-30).

Under constant weights the structural draw uses exact fast paths: `eigen` (Mode I, `O(q²)` per
draw), `fixed_factor` (Mode II) and `cholesky_const` (`iw_equivalent`; also what
`const_path="cholesky"` gives with time-constant weights). A deterministic shape rule chooses among
them (DESIGN D-26; `resolve.py:18-31`; path-to-mode mapping and line range corrected by
fact-check). They are internals and do not change the target.

### 6.2 `SV`: AR(1) or random-walk log-volatility

`specs.py:107-153`; `sv_arrays.py:1-14`, `:44-88`; DESIGN §6.3; N02. For each structural
equation:

$$
h_{i,t} = \log\lambda^{\text{pers}}_{i,t},\qquad h_{i,t} = \mu_i + \phi_i(h_{i,t-1}-\mu_i) + \sigma_i\eta_{i,t},\ \ \eta_{i,t}\sim N(0,1);\qquad
\texttt{rw=True}:\ h_{i,t} = h_{i,t-1}+\sigma_i\eta_{i,t}.
$$

Defaults (stochvol's parametrisation; `specs.py:137-153`, `:276-283`; `sv_arrays.py:59-79`):

| Parameter | Default prior |
| :--- | :--- |
| `μ_i` | `N(log(ψ_ι/ι), 10)` (data-centred on the Mode I variance scale, per window, R9; `mu_prior="data"`, `mu_var=10`) |
| `(φ_i+1)/2` | `Beta(20, 1.5)` (prior mean φ ≈ 0.86, `specs.py:97`) |
| `σ_i²` | `B_σ·χ²₁` with `B_σ = 1`; RW: `B_σ = 0.1` (`ScaledChi2`, i.e. `σ ~ |N(0, B_σ)|`); option `InvGamma(c₀, C₀)` |
| `h_{i,0}` | stationary `N(μ, σ²/(1−φ²))`, or `N(μ, B₀σ²)` with `h0=B₀`; RW: `N(log(ψ_ι/ι), rw_h0_var = 4)` |

Settings a variant does not use are validated and then reset to their defaults, so one model has
one spec, label and hash (`specs.py:129-134`, `:185-190`). The data-centred μ prior versus
stochvol's `N(0, 100²)` was DESIGN Q3, whose recommendation (data-centred; stochvol priors for
parity runs) was accepted on 2026-10-22 (DESIGN §12.2, "every recommendation below is accepted as
stated"). What stays open is risk R6: the forecast quality of the Mode I default and of the
data-centred level prior is unknown (DESIGN §12.1) (corrected by fact-check: a decided question with
an open risk, not an open question).

**Sampler** (details in the SV/validation dossiers; DESIGN §3.7.4, §4.4):

- The working observation is `y*_t = log(ẽ_t² + c_i)`, with relative offset
  `c_i = 1e-12·ψ_i` (`sv_arrays.py:35`, `:78-79`; D-2).
- `log χ²₁` is approximated by the Omori et al. (2007) [CITE?] 10-component mixture. `KSC-7` is
  available as `mixture="ksc7"`.
- An exact MH correction is on by default (`exact=True`). Its log weight is
  `w(h) = Σ_t m_t[log N(ẽ_t; 0, e^{h_t}) − log g(y*_t − h_t)]`, which makes the target exact for
  any mixture table (D-1; NOTES B.3).
- The `(T+1)`-state AWOL tridiagonal precision draw runs in non-centred form.
- ASIS interweaving (Kastner & Frühwirth-Schnatter 2014 [CITE?]) uses the measure-preserving sign
  fold `(σ, h̃) ↦ (−σ, −h̃)` (D-7; NOTES A.1).
- The RW path uses a cancellation-free path-Laplacian factorisation (NOTES E.17).
- `exact=False` and `offset=0.0` are the stochvol/bayesianVARs parity modes
  (`specs.py:125-127`).

### 6.3 Robust SV: outliers and Student-t

`specs.py:120-124`; `sv_arrays.py:91-143`; `src/rai_bvar/components/vol/transitory.py`; N03 §0;
DESIGN §3.7.6, §6.4.

$$
e_{i,t} = o_{i,t}\sqrt{\tau_{i,t}}\,e^{h_{i,t}/2}\,\varepsilon_{i,t},\qquad \lambda^{\text{eff}}_{i,t} = e^{h_{i,t}}\,o_{i,t}^2\,\tau_{i,t}.
$$

- **Outliers** (`outliers=True`; Carriero, Clark, Marcellino & Mertens 2024 [CITE?]):
  `o_t ∈ {1, …, K}` with `K = outlier_max = 20`, `P(o_t = 1) = 1 − p` and `P(o_t = k) = p/(K−1)`.
  The prior on `p` is `OutlierPrior(every_years, prior_years)`, resolved with `f` periods per year
  to `Beta(a, f·prior_years − a)`, `a = prior_years/every_years`. The defaults are SVO
  `Beta(2.5, 10f − 2.5)` (one outlier every 4 years over 10 prior years) and SVO-t
  `Beta(1, 10f − 1)` (every 10 years) (`specs.py:281-283`; `sv_arrays.py:91-102`). Frequencies
  other than monthly and quarterly need an explicit `Beta`.
- **Student-t** (`tails="t"`): the default is **standardized**, `τ_t ~ IG(ν/2, (ν−2)/2)`, so that
  `E τ = 1` and `e^{h}` stays the conditional variance. `t_scale="unit"` gives `IG(ν/2, ν/2)`
  (JPR 2004 [CITE?], CCMM). `ν ~ DiscreteUniform{3, …, 40}` on a static grid, sampled by
  collapsed exact discrete Gibbs (D-8). A standardized t with `ν ≤ 2` raises (`specs.py:191-195`).
- **Outliers + t**: both; transitory step order `o → p → ν → τ`, then `s → h → θ_SV`
  (D-9; DESIGN §4.4).

The **transitory-integrated** shock density. `transitory_log_density`
(`src/rai_bvar/forecast/result.py:42-93`) implements it for the opt-in RB-transitory LPD. The same
density is the `VolBlock.obs_logdensity_eq` hook (`src/rai_bvar/components/vol/robust.py:27-28`;
DESIGN §3.5, line 763-764), which is meant for the Phase 4 particle filter. That hook is a separate
engine-side function, not `result.py` (attribution corrected by fact-check):

| Variant | `log f(z)` |
| :--- | :--- |
| outlier | `log Σ_k w_k k⁻¹ φ(z/k)` |
| t | `log t_{c,ν}(z)`, `c = ν−2` (standardized) or `ν` (unit) |
| outlier + t | `log Σ_k w_k k⁻¹ t_{c,ν}(z/k)` |

`stochvol` 3.2.1's Student-t SV sampler does not target the standardized-t posterior that this
package's sampler matches (NOTES A.2). Status: recorded for stochvol 3.2.1 only; report upstream
first.

### 6.4 One Gibbs sweep (fixed order)

DESIGN §4.2–4.6. The order is decided in one place, `build_plan`, and printed by
`fit.plan.describe()` (DESIGN §1.3, §4.1).

**Structural form**, per equation (vmapped):

1. `(δ_i, a_i) | λ_eff` — one joint Gaussian draw;
2. the residual `e_i` from the new coefficients;
3. `VolBlock.sample_eq`, atomically: transitory `o → p → ν → τ`; then `s` (the mixture
   indicators) immediately before `h`; then `θ_SV` centred, then non-centred (ASIS).

**Reduced form**: a pre-phase CCCM step `Π | A, λ`, then per equation `a_j | Π, λ_j`, then the
same vol step.

The mixture indicators are **never part of the state**: they are local to the vol step. So the
Del Negro & Primiceri (2015) [CITE?] rule "nothing between `s` and `h`" is enforced by
construction, not by convention (DESIGN §2.7, §4.8 invariant 2).

---

## 7. Engine resolution: the composition chooses the engine

SPEC §4.5; DESIGN §11.1; `src/rai_bvar/api.py:1-27`, `:128-154`, `:156-184`, `:188-191`.

There is **no user-settable engine field** (erratum E20). `BVAR.__post_init__` runs three stages,
and every error is raised at construction:

1. **Resolve the `auto` defaults** (D-19):
   - `Minnesota.form == "auto"` resolves to `"conjugate"` iff `vol` is a `Constant` with prior
     `"auto"` or `"iw"`, else to `"structural"`.
   - Then `Constant.prior == "auto"` resolves to `"iw"` iff the form is `"conjugate"`, else to
     `"ig"`.
   - The resolved components are stored (`object.__setattr__`), so `BVAR()` keeps its Phase 1
     meaning, spec and hash.
2. **Apply the engine rules** (first match wins):

| # | Resolved composition | Engine / mode |
| :--- | :--- | :--- |
| 1 | `conjugate × Constant("iw")` | analytic (Phase 1) |
| 2 | `conjugate × anything else` | `ValueError` |
| 3 | `structural × Constant("ig")` | MCMC, Mode I |
| 4 | `structural × Constant("iw")` | MCMC, Mode II (NIW twin); prior axis `scaling=conjugate` |
| 5 | `structural × Constant("iw_equivalent")` | MCMC, Mode I δ + Chan `(a, λ)` |
| 6 | `structural × SV(...)` | MCMC, Mode I |
| 7 | `reduced × Constant("ig")` or `SV(...)` | MCMC, CCCM |
| 8 | `reduced × Constant("iw_equivalent")` | MCMC, CCCM + Chan `(a, λ)` (order-invariant) |
| 9 | `reduced × Constant("iw")` | `ValueError` |

3. **Apply the MCMC-only rules** (`api.py:156-184`): LP scaling raises; `hierarchy="mh"` raises;
   EB over `psi` with structural `cross ≠ 1` raises. Then comes the JAX-free legality check
   `engines.mcmc.config.validate_composition` (`api.py:118-126`; DESIGN §3.8).

`BVAR.engine` returns `"analytic"` iff the resolved form is `"conjugate"` (`api.py:188-191`).
`mcmc` settings given to an analytic composition are normalised to `None`, so the Phase 1 spec
is unchanged (`api.py:105-106`).

**Evidence.** `tests/unit/test_api_resolution.py::test_resolution_table` encodes the table row by
row: 16 parametrised `(form, vol)` cases → (resolved form, resolved vol, engine, static scaling
mode) (`test_api_resolution.py:46-141`). Further tests check the error messages: rows 2 and 9,
LP, `mh`, the double-treatment message, `contemp ≠ 1` in conjugate form. Verified *(re-run
here)*.

**The Phase 2 user API surface** (DESIGN §11.1–11.3; `api.py:337-381`;
`src/rai_bvar/settings.py:62-184`):

- `BVAR.fit(data, *, seed=None, init=None, runtime=None, plan=None)` returns `AnalyticFit` or
  `MCMCFit`. `init` (warm start), `runtime` and `plan` are MCMC-only and raise on analytic
  models (`api.py:354-362`).
- `MCMCSettings` **is part of the spec** (every field can change the draws): `chains=4`,
  `draws=2000`, `burn=1000` (exact, never rounded), `thin=1`, `warm_burn=None →
  max(250, burn//4)`, `seed=0`, `store="params"`, `store_max=2500`, `path_thin=10`,
  `t_bucket=32`, `gram="auto"`, `const_path="auto"`, `rhat_max=1.01`, `ess_min=400`
  (`settings.py:62-88`).
- `Runtime` is **not** part of the spec. Its fields are `devices=1`, `chunk=250`,
  `memory_budget_gb=2.0`, `storage_dir`, `replay`, `progress` (`settings.py:130-150`). `chunk`,
  `memory_budget_gb`, `storage_dir` and `replay` never change the draws, and their
  bitwise invariance is tested. `devices ≥ 2` (sharding under `shard_map`) matches the one-device draws
  only to rounding (`rtol 1e-9` on short runs, tested), not bitwise (`settings.py:132-143`; DESIGN
  Stage 2 amendment (1)) (corrected by fact-check: not every knob is bitwise-invariant).
- `FitPlan(horizon, draws, seed)` requests in-loop forecasts (for Phase 4–5 models whose draws
  cannot be stored; `settings.py:168-176`).
- `MCMCFit` exposes:

| Attribute | Content |
| :--- | :--- |
| `posterior` | `xarray.DataTree` |
| `diagnostics` | DataFrame |
| `summary()` | primary diagnostics, with `attrs` for prior form/mode, hyperparameters and EB note, implied reduced-form moments, ordering, converged |
| `converged` | the convergence flag |
| `warnings` | warning messages |
| `forecast`, `irf`, `draw_parameters`, `reduced_form_draws` | forecasting and parameter access |
| `final_state` | warm-start state |
| `timing` | timings |

  Sources: `results.py:203-480`; DESIGN §11.3.

**Spec, hash and labels.** The spec, hash and label machinery is what makes ablation tables
self-describing. Fields:

- `BVAR.spec()` is plain JSON. `contemp` is omitted at its default and `mcmc` appears only on
  MCMC models, so Phase 1 spec hashes are unchanged (`api.py:229-245`; golden test
  `tests/unit/test_phase1_hash_golden.py`, verified *(re-run here)*).
- `from_spec(spec())` round-trips (`api.py:247-284`; property test
  `tests/unit/test_spec_roundtrip_mcmc.py`, verified *(re-run here)*).
- `BVAR.axes = {mean, prior, vol, lags, deterministic, outliers}` (`api.py:215-227`).
- SV names: `"SV"`, `"SV(rw)"`, `"SV(outliers)"`, `"SV(t)"` and combinations. `label` adds
  every non-default setting: nested prior objects appear as 6-character hashes, scalars and strings
  as their values. For example, the CR report's `SV(rw,t, nu_prior=7eca60, t_scale=unit)`
  (`specs.py:199-218`; `minnesota.py:218-227` `_short`; corrected by fact-check: only nested priors
  are hashed).

---

## 8. Fair-ablation machinery (R2 in code)

1. **Grid harmonisation** (`api.py:288-333`; `src/rai_bvar/evaluation/compare.py:45-66`;
   DESIGN §11.4; erratum E22). Without it, the SPEC §9.5 grid
   `vol=[Constant(), SV(), SV(outliers=True)]` would resolve its `Constant()` arm to the analytic
   NIW model and its SV arms to structural Mode I. The arms would then differ in their mean
   prior, not only in volatility.

   `grid(BVAR, …)` calls `BVAR.harmonise_grid`. If any arm is MCMC because of its volatility,
   every `form="auto"` becomes `"structural"`. Then, in every grid, `Constant("auto")` becomes `"ig"`
   in each arm whose form (after that rewrite) is `"structural"` or `"reduced"`. An arm with an
   explicit `form="conjugate"` keeps `Constant("auto")`, and `BVAR` then resolves it to `"iw"`
   (`api.py:302-322`; rule stated precisely by fact-check). Explicit settings are never changed, and
   the rewritten arms are listed in an INFO log.
   Evidence: `tests/unit/test_grid_harmonise.py::test_spec_grid_becomes_a_one_axis_volatility_comparison`
   and `::test_all_constant_grids_stay_analytic_and_unchanged`, verified *(re-run here)*.

   **Paper point:** the "Constant" baseline of a volatility ablation is the **MCMC Mode I
   `Constant("ig")`** model, not the analytic NIW BVAR. The Clark–Ravazzolo report instead uses
   `Constant(iw_equivalent)` with `form='reduced'` for all arms
   (`benchmarks/reports/clark_ravazzolo.md`, "Design"). The paper should say which baseline each
   table uses.

2. **Mode II is a prior change**, labelled `scaling=conjugate`, so it is never paired with Mode I
   along `vol` (§5.5).

3. **One mean prior across volatility arms.** EB from the conjugate twin depends only on the
   window (§5.7).

4. **`verdict()` refuses multi-axis comparisons.** The runnable example (§13) shows
   `BVAR(2) vs RW` → `not comparable` ("differs along ['deterministic', 'lags', 'mean', 'model',
   'outliers', 'prior', 'vol']").

---

## 9. Data layer

### 9.1 `VARData`

`src/rai_bvar/data/dataset.py:20-35`.

- `VARData` holds the raw levels (a `PeriodIndex`; `from_frame` accepts a `DatetimeIndex` with
  inferable frequency), the transformation codes, an optional `vintage` label and an optional
  `start` that trims the transformed sample while keeping earlier raw values for pathwise
  inversion.
- Column order **is** the recursive ordering (`dataset.py:3-5`).
- Gaps in the index, duplicates, infinite values and log transforms of non-positive values raise
  (`dataset.py:43-44`, `:267-285`; `transforms.py:49-53`).

Sub-samples:

| Method | What it does | Lines |
| :--- | :--- | :--- |
| `upto(origin)` | data available at the origin (R9) | `dataset.py:123-134` |
| `tail(n_obs)` | rolling windows | `:136-146` |
| `select(variables)` | subsetting and reordering, used by `ordering_sensitivity` | `:148-160` |
| `realized(origin, h)` | future outcomes on the transformed scale, NaN when unobserved | `:162-171` |

**Missing values** (SPEC §5.7). Leading and trailing incomplete periods are trimmed with a warning
(a trailing trim is the "ragged edge"). Missing values **inside** the sample raise
(`dataset.py:215-243`; `tests/unit/test_dataset.py::test_interior_missing_value_raises`, verified
*(re-run here)*).

### 9.2 Transforms

`src/rai_bvar/data/transforms.py:1-26`, `:68-100`. The FRED-MD/QD codes of McCracken & Ng
(2016) [CITE?]:

| tcode | Transform |
| :--- | :--- |
| 1 | level |
| 2 | Δx |
| 3 | Δ²x |
| 4 | log x |
| 5 | Δ log x |
| 6 | Δ² log x |
| 7 | Δ(x_t/x_{t−1} − 1) |

Each code has a forward map and a **pathwise inverse**. Inverses apply to each joint simulated
path using the last observed levels, never to quantiles or means, because nonlinear transforms do
not commute with expectations (SPEC §5.2). `ForecastResult.paths_level` is that inverse
(`dataset.py:182-192`). Evidence: `tests/unit/test_transforms.py::test_inverse_is_pathwise_not_applied_to_means`
and the round-trip property tests (in the fast suite; not re-run here because they use
hypothesis); `tests/contract/test_forecast_contract*.py::test_paths_level_is_the_pathwise_inverse`
(verified *(re-run here)*).

### 9.3 Transform-aware Minnesota prior mean

SPEC §5.3. The own-first-lag prior mean is 1 for persistent level series (tcodes 1 and 4) and 0
for differenced series (tcodes 2, 3, 5, 6, 7) (`transforms.py:25-26`, `:40-42`;
`dataset.py:117-119`). Users override per variable with `Minnesota(mean={...})` or with one
scalar (`minnesota.py:207-215`).

If user-specified means contradict the tcodes, `fit()` warns: "levels with mean < 0.5 or
differences with mean ≥ 0.5" (`analytic.py:1043-1059`). The MCMC fit recomputes the same check
(DESIGN §6.5 step 4). The SOC default `soc_scope="persistent"` follows the same logic (§5.2).

### 9.4 Vintages and data sources

- `VARData.vintage` is a label (SPEC §5.5).
- `backtest(data=…)` accepts either one `VARData` (pseudo real time: each origin sees
  `data.upto(origin)`) or a callable `origin → VARData`. With a callable, the realised outcomes
  must be passed as `realized=` (`src/rai_bvar/evaluation/backtest.py:195-237`, `:365-369`;
  `tests/integration/test_backtest.py::test_callable_data_source_needs_realized`, verified
  *(re-run here)*).
- Real-time vintages are therefore API-ready but not shipped (SPEC §1.3 non-goal for v1.0).

### 9.5 Reference datasets

`src/rai_bvar/datasets/fred.py:1-13`, `:56`, `:71-74`, `:76-163`, `:169-247`.

- `rb.datasets.load("fred-qd" | "fred-md", vintage="2026-09", variables="small"|"medium"|"large"|[...], tcodes="rai"|"official"|{...})`
  downloads a pinned vintage on demand, verifies a SHA-256 checksum from `REGISTRY` and caches it
  under `$RAI_BVAR_DATA` or `~/.cache/rai-bvar`.
- The files are **not bundled**, because some FRED series are third-party copyrighted. The
  small/medium/large sets use **only U.S. federal-agency series** (BEA, BLS, Census, Fed), in the
  recursive ordering given.
- Sizes: QD small = GDPC1, GDPCTPI, FEDFUNDS; medium 7 series; large 20 series. MD likewise
  (`fred.py:76-153`).
- `fill_gaps=True` (default) interpolates **interior** publication gaps (in logs for log series)
  with a warning listing every filled value, e.g. October 2025 for BLS series after the federal
  shutdown. With `False` they are kept and `VARData` raises.

**`"rai"` vs `"official"` tcodes.** The default `tcodes="rai"` uses the codes of the named set:

- QD small: GDPC1 5, GDPCTPI 5, FEDFUNDS 1 (`fred.py:77-81`);
- QD large: also TCU 1, UNRATE 1, HOUST 4, GS1/GS10 1.

The committed synthetic fixture `tests/fixtures/fred_qd_sample.csv` lists `transform` codes 6 for
GDPCTPI and 2 for FEDFUNDS, i.e. its "official" codes differ from the `rai` ones. Whether the
2026-09 FRED-QD file's official codes are 6/2 for these series is **to check**. The paper must
state the transformation used: inflation as Δ log, the policy rate in levels.

---

## 10. Forecasting and structural analysis

### 10.1 Joint path simulation

SPEC §6.1; DESIGN §7.1–7.2; `src/rai_bvar/engines/mcmc/forecast.py:1-44`. For each selected
parameter draw:

1. simulate future volatilities `(log λ^pers, log λ^eff)_{T+1..T+H}`, including future `o` and
   `τ`;
2. scan `h = 1..H`: build `x_{T+h}` from the simulated path, then compute

$$
m_{s,h} = A^{-1}\Gamma x_{T+h} = B'x_{T+h},\qquad L_{s,h} = A^{-1}\,\mathrm{diag}\big(\exp(\tfrac12\log\lambda^{\text{eff}}_{T+h})\big),\qquad
y_{T+h} = m_{s,h} + L_{s,h}\varepsilon_h .
$$

`L` is lower triangular with a positive diagonal, so it is the Cholesky factor of the conditional
covariance at no extra cost. `y_{T+h} | draw, path_{<T+h}, λ_{T+h} ~ N(m, LL')` holds exactly,
conditional on the drawn transitory scales (`forecast.py:15-19`).

The analytic engine draws `(B, Σ)` from the NIW posterior (integrating over MH hyperparameter
draws when `hierarchy="mh"`, `analytic.py:589-599`) and uses `L = chol(Σ)`. Under LP, `L` is
scaled by the future `s_{T+h}` (`outliers.scales(future, covid)`, `analytic.py:673`), so
`cond_chol` is then horizon-specific (`analytic.py:664-683`).

Draw selection (`forecast.py:20-33`; DESIGN §7.1):

- `fit.forecast(H, draws, seed)` takes `⌈draws/C⌉` evenly spaced stored draws per chain and keeps
  the `(chain, draw)` structure.
- When more paths are requested than draws are stored, each stored draw gives
  `⌈draws/(C·D)⌉` paths with independent shocks.
- Keys derive from `(fseed, chain, kept-draw index)`, so in-loop and post-hoc forecasts agree,
  and recomputing the moments reproduces the same `o`, `τ` and `ε`.

### 10.2 IRFs and FEVD

`src/rai_bvar/structural/irf.py:1-25`, `:58-89`, `:90-133`, `:136`; DESIGN §7.4.

$$
\Phi_0 = I,\qquad \Phi_h = \sum_{l=1}^{\min(h,p)} A_l\Phi_{h-l};\qquad
\Theta_h^{\text{sd}} = \Phi_h L,\qquad \Theta_h^{\text{unit}} = \Phi_h L\,\mathrm{diag}(L)^{-1} = \Phi_h A^{-1}.
$$

- In the recursive structural form, `L = A⁻¹Λ^{1/2}`. Unit-shock responses are therefore
  **date-free**; one-s.d. responses depend on the date at which `Λ` is evaluated.
- `fit.irf(horizon=20, draws=1000, seed=None, shock="sd", date=None, which="persistent")` uses
  the **persistent** variance by default: "a one-s.d. shock in March 2020" excludes the outlier
  scale. `which="effective"` includes it. `date=None` is the last in-sample period; other dates
  need `store="paths"` (`forecast.py:35-44`).
- `IRFResult.fevd()` computes the FEVD; it is date-specific under SV (DESIGN §7.4).

Evidence: `tests/mcmc/forecast/test_irf_mcmc.py::test_irf_equals_direct_phi_ainv_lambda`,
`::test_unit_shock_irf_is_phi_ainv_and_date_free`;
`tests/unit/test_irf.py::test_fevd_matches_forecast_error_covariance`. Verified *(re-run here)*.

### 10.3 Ordering sensitivity (R8)

`src/rai_bvar/structural/ordering.py:1-73`, `:1102-1183`; DESIGN §13 WP22; N08 §B4.

`rb.ordering_sensitivity(model, data, n_perm=10, pinned=(), mode="prior"|"forecast"|"backtest", design="balanced"|"random"|"all", noise_reps=3, pool=False, batch=False, …)`
refits under permutations of the free variables, recomputing every data-dependent quantity per
permuted window (R9). It maps outputs back by label and compares their dispersion with a Monte
Carlo noise floor `σ_MC = max(sd across seeds, MCSE)`.

- **Permutation design.** All orderings when they fit in `n_perm`; otherwise the user ordering
  and its reverse, then the rows of a random Latin square (each variable in each free position
  equally often), then random permutations.
- **Thresholds.** Metrics such as `sd_ratio > 1.10` and `lpd_range > 0.05` nats are flagged only
  when the signal-to-noise ratio is above 3.
- **Pooling.** `pool=True` returns the equal-weight mixture over orderings, which is exactly
  order-invariant with `design="all"`.

In Phase 2 only `Linear(form="reduced") × Constant("iw_equivalent")` is order-invariant, besides
the analytic NIW engine (`ordering.py:5-7`).

*Note added by fact-check.* The committed CR report runs `ordering_sensitivity` on exactly this
order-invariant `Constant` arm and gets `material = True`, with joint LPD ranges over orderings of
0.027 (h=1), 0.189 (h=2), 0.768 (h=4) and 2.280 (h=8) nats per origin
(`benchmarks/reports/clark_ravazzolo.md`, "All 24 orderings", line 364). The report calls that
spread Monte Carlo noise (line 335), and the h ≥ 4 values involve the degenerate RB cells of §11.3.
Before the paper presents the check's false-alarm behaviour, it should reconcile this with the
order-invariance claim. Status: recorded; to check.

Evidence (`tests/integration/test_ordering.py`, not re-run here):

| Test | Tier |
| :--- | :--- |
| `test_analytic_engine_is_exactly_order_invariant` | fast suite (no `slow` marker; `test_ordering.py:246`) |
| `test_reduced_iw_equivalent_is_order_invariant_within_mcse` | `@pytest.mark.slow` (nightly suite; `test_ordering.py:735`) |
| `test_prior_mode_shows_mode_one_sv_position_effects_and_no_iw_equivalent_ones` | fast suite (no `slow` marker; `test_ordering.py:503`) |

---

## 11. The predictive contract (SPEC §7)

### 11.1 Fields and invariants

`src/rai_bvar/forecast/result.py:1-19`, `:36-37`, `:96-162`. `ForecastResult` wraps an
`xarray.Dataset`:

| Field | Dims | Meaning |
| :--- | :--- | :--- |
| `paths` | `(chain, draw, horizon, variable)` | joint simulated paths, transformed scale; each slice is one path from one parameter draw |
| `paths_level` | same | pathwise inverse to raw levels |
| `cond_mean` | same | `m_{s,h}`: mean of `y_{T+h}` given the draw, the simulated path to `T+h−1` and `λ_{T+h}` |
| `cond_chol` | `(…, variable, variable_)` | `L_{s,h}`: lower-triangular factor of the conditional covariance |
| attrs | — | every engine: `origin`, `freq` (`result.py:157`), `model`, `engine`, `spec`, `spec_hash`, `tcodes`, `seed`, `converged`, `warnings`, `rai_bvar_version`; analytic adds `mh_seed` (`analytic.py:696-707`); MCMC adds `ordering`, `n_chains`, `mixture`, `exact`, `transitory`, `outlier_K`, `t_scale`, `fseed` and the fit's DESIGN §5.6 metadata (`engines/mcmc/forecast.py:291-321`, `:447`). Analytic forecasts carry **no** `ordering` attr (corrected by fact-check; see §15 D15) |

Optional MCMC **accelerator** variables (`cond_A`, `cond_scale`, `cond_scale_persistent`,
`log_p`, `log1m_p`, `nu`) are never contract fields. `check_structure` and every consumer ignore
them (`result.py:8-19`; D-20, erratum E12).

**Invariants** (SPEC §7), checked by the contract test layer for every registered composition
(`src/rai_bvar/forecast/invariants.py`):

1. simulation consistency: `L⁻¹(paths − cond_mean)` is iid `N(0, I)`;
2. draws are joint across horizons and variables (`cond_mean` is the VAR mean of the simulated
   path);
3. stacking `(chain, draw)` gives the v1 shape `(n_draws, H, n)` in chain-major order.

The MCMC file also checks `cond_chol = A⁻¹ diag(cond_scale)`. Evidence:

- `tests/contract/test_forecast_contract.py` (RW, OLS-VAR, analytic BVARs);
- `tests/contract/test_forecast_contract_mcmc.py` over **12 MCMC compositions**: structural × {SV,
  SV(outliers), SV(t), SV(outliers,t), SV(rw), Constant(ig), Constant(iw),
  Constant(iw_equivalent)} and reduced × {SV, SV(outliers), Constant(ig),
  Constant(iw_equivalent)} (`tests/mcmc/forecast/mcmc_fits.py:128-141`).

Verified *(re-run here: 100 slow + fast-tier tests passed)*. Note that `reduced × SV(t)`,
`reduced × SV(rw)` and `reduced × SV(outliers,t)` are **not** in the contract list (open question
Q14).

### 11.2 The Rao-Blackwellised log predictive density

`result.py:209-279`:

$$
\log\hat p(y_{T+h}\mid Y) = \operatorname{logsumexp}_s\ \log N\big(y;\ m_{s,h},\ L_{s,h}L_{s,h}'\big) - \log S .
$$

- **Variable subsets.** A subset marginalises the Gaussian. A leading block `{0..j−1}` of the
  ordering (in any order) uses the exact factor `L[:j, :j]`; other subsets factor `L[idx]` by QR
  (`linalg.marginal_chol`). With the MCMC accelerators, the factored density
  `−½ j log 2π − Σ log ℓ_i − ½‖(A(y − m))_{:j} ⊘ ℓ_{:j}‖²` costs `O(n²)` per draw. Tested equal
  to the `cond_chol` path (`tests/mcmc/forecast/test_lpd_mcmc.py::test_factored_lpd_equals_cond_chol_lpd`,
  verified *(re-run here)*).
- **MCSE** (`return_mcse=True`). The delta-method value is `sd(w)/(√S·mean(w))` for
  `w_s = exp(log p_s − max)`. For MCMC forecasts it is multiplied by `√(S/ESS)` with
  `ESS = ess_mean(w)` on the `(chain, draw)` array (`result.py:275-288`; DESIGN §7.3;
  `test_ess_adjusted_mcse`, verified *(re-run here)*).
- **Opt-in transitory Rao-Blackwellisation** (`rb_transitory=True`). It integrates
  `o_{T+h}, τ_{T+h}` out analytically with the densities of §6.3 (`result.py:231-236`,
  `:312-328`). It is off by default (DESIGN Q7); `test_rb_transitory_matches_monte_carlo` is
  verified *(re-run here)*.
- **Not implemented:** SPEC §7's `store_moments="lazy"` (recompute moments on demand) appears
  nowhere in `src/` (grep at this commit). Status: to check (drop from SPEC or implement).

### 11.3 A known limitation of the contract at h > 1 (methods point)

For `h > 1` the estimator conditions on each draw's **simulated path** up to `T+h−1`. For outcomes
far in a predictive's tail, one draw carries all the weight. The estimate is then biased low by
up to tens of nats, while the delta-method MCSE saturates near 1 (NOTES B.8b; DESIGN §2.10 Stage 4
amendment (4)).

Constant's US unemployment `h = 8` score at 2008Q2:

| Forecast draws | RB score |
| :--- | :--- |
| 1000 | −93.6 |
| 8000 | −46 to −63 |
| 64,000 | −36 |
| exact path-integrated mixture | −19.1 |

This inflated the quick Clark–Ravazzolo UR `h = 8` ΔLPS about 8× (+1.92 against ≈ +0.23
converged). The committed full report flags degenerate cells ("RB degen." column; e.g. full
SV(rw) UR h8: ΔLPS +1.954 RB vs −0.499 Gaussian-approx, 12 degenerate origins;
`benchmarks/reports/clark_ravazzolo.md`, "Reliability of the RB log score" and the full-sample
table). Status: recorded. Re-run with the library fix before publishing.

The remedy, integrating the path out (Gaussian given θ and the volatility path), is an open
SPEC-level decision (DESIGN Stage 4 amendment (4)). An implementation is in progress outside this
branch and is not part of commit `62d15f6`. The paper must not describe it as the contract until
it is committed (to check; open question Q2).

---

## 12. Evaluation layer

### 12.1 Backtest

`src/rai_bvar/evaluation/backtest.py:1-37`, `:71-122`, `:195-293`, `:335-419`. The signature is
`rb.backtest(model, data, *, origins, window=Expanding(), horizon=8, draws=1000, seed=0, cache=None, n_jobs=1, realized=None, name=None, axes=None, warm_start=False)`
(`backtest.py:195-210`; `name` and `axes` added by fact-check).

- **Origins** run from `start` to `end` (default: the second-to-last period) every `step`.
- **Windows.** `Expanding(min_size)` sets `valid_tests=False`; `Rolling(size)` sets
  `valid_tests=True` (`backtest.py:71-106`).
- **At each origin** the model sees `window.select(data.upto(origin))` and is fit afresh (R9).
  It forecasts `H` steps and is scored against `realized.realized(origin, H)`.
- **Seeds.**
  - Each origin has its own forecast generator `default_rng([seed, crc32(origin)])`, so results
    do not depend on which other origins run.
  - Models that declare `uses_fit_seed` (MCMC) are fit with
    `fit_seed = default_rng([seed, crc32(origin), 0x4D434D43]).integers(2**31)`
    (`backtest.py:335-341`, `:387-398`).
  - The MCMC data salt `crc32(data.fingerprint())` keeps chains independent across windows and
    orderings (`resolve.py:33-38`, `:294`; DESIGN §5.6).
- **Cache.** Per-origin netCDF files are keyed by (spec, estimation sample, realised outcomes,
  origin, settings); MCMC keys add `"seeding": "fit_seed-v1"`. Backtests therefore resume, and
  adding a model only fits the new model (`backtest.py:4-6`, `:31-36`, `:490-510`).
- **Warm starts** are opt-in (`warm_start=True`) and need strictly increasing origins, because a
  warm start from a later origin would leak later data (R9) (`backtest.py:244-248`). Separately, a
  warm origin's result depends on the previous origins, so its cache key records
  `"init": "warm:<previous origin key>"`; the first origin keeps the cold key (`backtest.py:23-33`,
  `:476-479`; the "so" linking the two was a non-sequitur, corrected by fact-check).
- **Processes.** `n_jobs > 1` with MCMC warns and uses `spawn`.

Evidence (`tests/integration/test_backtest.py`, `tests/integration/test_backtest_mcmc.py`; verified
*(re-run here, fast tier)*):

- `test_no_look_ahead_and_window_sizes`
- `test_results_do_not_depend_on_other_origins`
- `test_cache_roundtrip_and_invalidation`
- `test_analytic_backtests_match_the_golden`
- `test_resumed_warm_backtest_equals_an_uninterrupted_one`
- `test_one_compile_per_t_bucket`

The slow `test_warm_and_cold_scores_are_statistically_equal` was not re-run.

`rb.estimate_cost(model, data, origins=…)` times one origin (the last) and extrapolates
(`backtest.py:296-332`). SPEC §9.1 describes it as taking a grid (to check).

### 12.2 Scores

`src/rai_bvar/evaluation/scores.py:1-136`. Per origin, horizon and model. All scores are on the
**transformed (model) scale**; no level-scale scoring exists (open question Q18).

| Score | Scope | Implementation |
| :--- | :--- | :--- |
| `lpd` (primary), `lpd_mcse` | joint | RB, §11.2 |
| `lpd_marginal` | per variable | RB marginal |
| `crps` | per variable | `scoringrules` fair CRPS via the PWM form, `O(S log S)` (`scores.py:18`, `:100-104`) |
| `qwcrps_{center,left,right}` | per variable | in-house quantile-weighted CRPS (Gneiting & Ranjan 2011 [CITE?]), midpoint rule over 99 levels, weights `a(1−a)`, `(1−a)²`, `a²` (`scores.py:20-47`) |
| `twcrps` | per variable, optional | `scoringrules`, with user `tw_bounds` |
| `energy` | joint | `scoringrules` fair energy score on a random subsample of ≤ 1000 draws (`scores.py:61-63`, `:87-95`) |
| `variogram` | joint | `scoringrules`, p = 0.5, same subsample |
| `sq_err`, `abs_err` | per variable | squared error of the posterior mean; absolute error of the median |
| `pit`, `cover50`, `cover90` | per variable | calibration |

Evidence: `tests/unit/test_scores.py::test_unweighted_qwcrps_matches_fair_crps`,
`::test_crps_against_gaussian_closed_form`, `::test_gaussian_forecast_scores_are_exact`. Verified
*(re-run here)*.

### 12.3 Tests of equal predictive ability

`src/rai_bvar/evaluation/significance.py:1-137`.

**Diebold–Mariano** on `d_t = L_A − L_B` (NaN pairs dropped):

$$
\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}/P}},\quad
\widehat{\mathrm{LRV}} = \hat\gamma_0 + 2\sum_{k=1}^{h-1}\hat\gamma_k\ \text{(rectangular kernel)};\qquad
\mathrm{DM}_{\text{HLN}} = \mathrm{DM}\sqrt{\frac{P+1-2h+h(h-1)/P}{P}}\ \sim t_{P-1}.
$$

- Newey–West (Bartlett) weights are used only if the rectangular estimate is not positive
  (`significance.py:35-53`, `:71-77`).
- `alternative="less"` tests whether A has the smaller expected loss.
- The validity policy follows Clark & McCracken (2001) and Giacomini & White (2006) [CITE?]:
  p-values are formally valid only with rolling (or fixed) windows. Expanding-window p-values are
  labelled descriptive (`significance.py:3-6`; `compare.py:187`, `:252`).

**Model Confidence Set** (Hansen, Lunde & Nason 2011 [CITE?]):

- `T_max` statistic on log-score losses (`−lpd`, joint only);
- circular block bootstrap with block length `max(1, round(P^{1/3}))` and `n_boot = 2000`;
- the model with the largest standardised loss relative to the average of the survivors is
  eliminated; p-values are running maxima (`significance.py:90-137`; `compare.py:146-151`).

Evidence: `tests/unit/test_significance.py::test_dm_size_under_null`, `::test_dm_hln_uses_student_t`,
`::test_mcs_eliminates_clearly_worse_models_first`. Verified *(re-run here)*.

### 12.4 `compare()`, the ablation table and the verdict

`src/rai_bvar/evaluation/compare.py:69-108`, `:155-194`, `:196-288`, `:331-336`. The signature
is `rb.compare(models, data, *, origins, window=Expanding(), horizons=(1,), draws=1000, seed=0, baseline=None, cache=None, n_jobs=1, realized=None)`.
It backtests every model on identical origins, windows and outcomes. Every statistic is computed
on the origins where **all** models have a finite joint LPD (`compare.py:132-144`).

**`table()` columns.**

| Group | Columns |
| :--- | :--- |
| Identity | model, horizon, `axis:*` labels, origins |
| Log score | total LPD; ΔLPD (total and mean); `dlpd_mcse` |
| CRPS family | `crps`, `crps_rel`, `qwcrps_left`, `qwcrps_right` |
| Multivariate | energy, variogram |
| Point errors | relative RMSE and MAE |
| Tests | `dm_stat`, `dm_pvalue` (two-sided), `tests_valid` |
| MCS | `mcs_pvalue`, `in_mcs_10`, `in_mcs_25` |
| Convergence | `converged` (the fraction of origins) |

**`verdict(axis=None, horizons=None, *, test_level=0.10, mcs_alpha=0.25, mc_multiple=3.0, min_converged=0.95)`.**
Pairing:

- Without `axis`, every model is paired with the baseline.
- With `axis="vol"`, every model is paired with the arm that has the baseline's value on that
  axis and agrees on all other axes (`compare.py:213-223`).

For each pair `(M, B)` and horizon `h`, with ΔLPD summed over the common origins
(`compare.py:245-287`):

$$
\text{better} \iff \underbrace{|\mathcal A(M)\,\triangle\,\mathcal A(B)| = 1}_{\text{R2}}
\ \wedge\ \Delta\mathrm{LPD}>0\ \wedge\ |\Delta\mathrm{LPD}| > 3\,\mathrm{MCSE}
\ \wedge\ \big[(\text{rolling} \wedge p^{\text{DM-HLN}}_{\text{one-sided}} < 0.10)\ \vee\ (M\in\mathcal M_{25\%} \wedge B\notin\mathcal M_{25\%})\big]
$$

- `MCSE = sqrt(Σ_origins (mcse_{M,t}² + mcse_{B,t}²))` (`compare.py:331-332`).
- `worse` is the mirror image.
- **Convergence gate:** if `min(conv_M, conv_B) < 0.95` (the fraction of origins whose forecast
  attr `converged` is true), the verdict is forced to `inconclusive`, with the reason "converged
  at only x% of origins" (`compare.py:254`, `:266-267`, `:274-275`).
- Two models with identical specs and no differing axis are compared and come out
  `inconclusive`. Differing specs with no declared differing axis, or more than one differing
  axis, give `not comparable` (`compare.py:235-244`).

The **per-origin `converged` flag** of an MCMC fit comes from DESIGN §8.2 and
`src/rai_bvar/engines/mcmc/monitor.py:22-28`:

```text
converged = no chain raised ∧ chains ≥ 2 ∧ (EB converged, when EB was used)
            ∧ ∀ q ∈ primary: R-hat(q) < rhat_max (1.01) ∧ bulk ESS(q) > ess_min (400)
```

R-hat is the rank-normalised split R-hat, `max(bulk, folded)`. The primary set:

- the log-likelihood;
- the own-first-lag reduced-form coefficients `B_{(1,i),i}`;
- `log λ_i` (Constant), or `μ_i, φ_i, σ_i, h_{i,T}` (AR SV), or `σ_i, h_{i,T}` (RW SV).

At `n = 20` this is 41 quantities with Constant and 101 with SV (`monitor.py:3-9`; recorded).
Secondary quantities never gate. The analytic EB/MH flag is `AnalyticFit.converged`
(`analytic.py:549-552`). Benchmarks carry no flag and count as converged: the backtest stores
`attrs.get("converged", 1)` (`backtest.py:407`; `compare.py:335-336`).

Evidence:

- `tests/integration/test_compare.py::test_two_axis_differences_are_not_comparable`,
  `::test_expanding_window_tests_are_descriptive`, `::test_identical_models_are_inconclusive`:
  verified *(re-run here)*.
- `tests/mcmc/diagnostics/test_monitor.py` covers the flag (not re-run here).
- **Size and power of the verdict engine on simulated DGPs (SPEC §10.5) are not tested yet.**
  There is no `tests/dgp/`, and SPEC §13 assigns the DGP verdict tests to the Phase 3 exit. Status:
  to check. The paper cannot claim a controlled false-positive rate.

**A finding about the gate (paper-worthy).** The committed Clark–Ravazzolo report
(`benchmarks/reports/clark_ravazzolo.md`, acceptance criterion 6) states that SV(rw) vs Constant
at `h = 1` is `better` in **0 of 24 orderings with the gate on, and in 24 of 24 with the gate
off**. In the ordering stage (106 origins, 1985Q1–2011Q2, expanding window), SV(rw) converged at
57–82% of origins depending on the ordering, Constant at 100% (report section "All 24
orderings"). In the full-sample stage it converged at 68%, with 4 chains × 5000 kept draws.
Status: recorded.

The replication script builds its `CompareResult` with `Expanding()` and calls
`verdict(min_converged=0.95)` (gate on) or `verdict(min_converged=0.0)` (gate off)
(`benchmarks/clark_ravazzolo.py:501-515`). With expanding windows `tests_valid` is false, so
every gate-off `better` came through the MCS route (`m_in and not r_in`,
`compare.py:255-259`). This bears on Q6. The report "will be replaced" by an improved acceptance run
(task statement), so re-read the new report before citing any number.

The runnable example (§13, snippet B) reproduces the mechanism on a homoskedastic DGP: the SV arm
fails the primary R-hat/ESS rule at all 11 origins, and the verdict is forced to `inconclusive`.
Open question Q3 asks whether this strictness (every primary quantity at every one of ≥ 95% of
origins, at R-hat 1.01) is the intended operating point.

---

## 13. Minimal runnable example (verified against commit `62d15f6`)

Both scripts were run on 2026-10-23 with the environment of §0, from the repository root:
`PYTHONDONTWRITEBYTECODE=1 uv run --frozen python ex_a.py` (12.3 s wall) and `… ex_b.py` (23.1 s
wall, JIT compilation included). They use no network. *Fact-check re-run (2026-10-23, same machine and
versions; scripts copied verbatim from this dossier into a scratch directory, run from the repository
root):* both reproduced every printed number bitwise, apart from the two timing lines of snippet B
(see §13.2). The wall times above are recorded and were not re-measured; they depend on the run. The data are simulated inline from a
3-variable VAR(1) that mirrors `tests/conftest.py:24-36`, stored as raw levels with tcodes
{5, 5, 1}.

- Analytic outputs are deterministic given the seeds (NumPy).
- MCMC outputs are bitwise reproducible only on the same platform and versions (DESIGN §5.6;
  NOTES C.9). Status of the MCMC numbers: recorded on this machine.

### 13.1 Snippet A: data, analytic BVAR, forecast, comparison

```python
# ex_a.py (verbatim)
import numpy as np
import pandas as pd

import rai_bvar as rb

# A tiny quarterly data set: a stationary VAR(1) in (GDP growth, inflation, rate),
# stored as raw levels (two log-levels, one rate) with FRED-style transformation codes.
rng = np.random.default_rng(0)
A = np.array([[0.4, 0.0, -0.001], [0.1, 0.6, 0.0], [15.0, 25.0, 0.85]])
mu = np.array([0.0075, 0.006, 4.0])
sd = np.array([0.007, 0.003, 0.25])
corr = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.2], [0.3, 0.2, 1.0]])
L = np.linalg.cholesky(corr * np.outer(sd, sd))
y, Y = mu.copy(), []
for _ in range(200 + 240):
    y = (np.eye(3) - A) @ mu + A @ y + L @ rng.standard_normal(3)
    Y.append(y)
Y = np.array(Y[200:])
raw = pd.DataFrame(
    {"gdp": 100 * np.exp(np.cumsum(Y[:, 0])), "prices": 50 * np.exp(np.cumsum(Y[:, 1])),
     "rate": Y[:, 2]},
    index=pd.period_range("1960Q1", periods=240, freq="Q"),
)
data = rb.VARData.from_frame(raw, tcodes={"gdp": 5, "prices": 5, "rate": 1})
print(data)
print("prior means of own first lag:", data.prior_means())

# Analytic engine: conjugate Minnesota with SOC/SUR dummies and GLP empirical Bayes.
bvar = rb.BVAR(mean=rb.Linear(prior=rb.Minnesota(soc=1.0, sur=1.0)), lags=2)
print(bvar.engine, "|", bvar.name)
fit = bvar.fit(data)
print(fit.summary())
fc = fit.forecast(horizon=4, draws=2000, seed=0)
print(fc)
print(fc.quantiles([0.05, 0.5, 0.95]).sel(horizon=1).round(4).to_pandas())
print("paths_level h=1 median:", fc.median(level=True).sel(horizon=1).round(3).values)

# Pseudo out-of-sample comparison on rolling windows (valid DM/AG p-values).
res = rb.compare(
    {"RW": rb.RandomWalk(), "OLS-VAR(2)": rb.OLSVAR(lags=2), "BVAR(2)": bvar},
    data,
    origins=rb.Origins(start="2010Q1", end="2019Q3"),
    window=rb.Rolling(size=120),
    horizons=[1, 4],
    baseline="RW",
)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 200)
cols = ["origins", "dlpd", "dlpd_mcse", "crps_rel", "dm_pvalue", "tests_valid", "in_mcs_25", "converged"]
print(res.table()[cols].round(3))
v = res.verdict()
print(v[["model", "reference", "axis", "horizon", "verdict"]])
for r in v.itertuples():
    print(f"{r.model} vs {r.reference} h={r.horizon}: {r.reasons}")
```

Output, verbatim (stdout; no warnings were emitted):

```text
VARData(variables=['gdp', 'prices', 'rate'], T=239, 1960Q2..2019Q4, freq='Q-DEC', vintage=None)
prior means of own first lag: {'gdp': 0.0, 'prices': 0.0, 'rate': 1.0}
analytic | BVAR(Minnesota(conjugate, eb, SOC+SUR), p=2)
            mode
lambda  0.268149
soc     1.119513
sur     1.176805
ForecastResult(origin=2019Q4, horizon=4, variables=['gdp', 'prices', 'rate'], samples=2000)
variable     gdp  prices    rate
quantile                        
0.05     -0.0053  0.0027  4.2629
0.50      0.0053  0.0073  4.6754
0.95      0.0158  0.0120  5.1097
paths_level h=1 median: [349.266 182.507   4.675]
                    origins    dlpd  dlpd_mcse  crps_rel  dm_pvalue  tests_valid  in_mcs_25  converged
model      horizon                                                                                    
RW         1             39   0.000      0.000     1.000        NaN         True      False        1.0
OLS-VAR(2) 1             39  14.447      0.102     0.907      0.029         True       True        1.0
BVAR(2)    1             39  15.368      0.094     0.893      0.020         True       True        1.0
RW         4             36   0.000      0.000     1.000        NaN         True      False        1.0
OLS-VAR(2) 4             36  33.191      0.884     0.860      0.001         True       True        1.0
BVAR(2)    4             36  35.051      0.946     0.842      0.002         True       True        1.0
        model reference                                              axis  horizon         verdict
0  OLS-VAR(2)        RW                                             model        1          better
1     BVAR(2)        RW  deterministic,lags,mean,model,outliers,prior,vol        1  not comparable
2  OLS-VAR(2)        RW                                             model        4          better
3     BVAR(2)        RW  deterministic,lags,mean,model,outliers,prior,vol        4  not comparable
OLS-VAR(2) vs RW h=1: ΔLPD=+14.45 (MC s.e. 0.10); DM p=0.014 (valid: rolling window); MCS25: model in, reference out
BVAR(2) vs RW h=1: differs along ['deterministic', 'lags', 'mean', 'model', 'outliers', 'prior', 'vol']; a verdict needs exactly one axis (SPEC R2)
OLS-VAR(2) vs RW h=4: ΔLPD=+33.19 (MC s.e. 0.88); DM p=0.000 (valid: rolling window); MCS25: model in, reference out
BVAR(2) vs RW h=4: differs along ['deterministic', 'lags', 'mean', 'model', 'outliers', 'prior', 'vol']; a verdict needs exactly one axis (SPEC R2)
```

What it shows:

- tcode 5 drops the first period (T = 239 from 240 raw rows).
- The transform-aware own-lag means are 0, 0, 1.
- `form="auto"` with the default `Constant()` resolves to the analytic engine; EB estimates λ,
  μ (SOC) and δ (SUR).
- The forecast is reported on both the transformed and the level scale.
- Rolling windows mark the tests `valid`. The table's `dm_pvalue` is **two-sided** (0.029),
  while the verdict reports the **one-sided** p-value (0.014) (`compare.py:166`, `:250-251`,
  `:270-271`).
- R2 makes `BVAR(2)` vs `RW` "not comparable". The README quick start
  (`README.md`, "Quick start") calls `res.verdict()` on exactly such a grid, so its output would be
  "not comparable" for the BVAR.

### 13.2 Snippet B: the same model family on the MCMC engine

```python
# ex_b.py (verbatim)
import platform
import time

import numpy as np
import pandas as pd

import rai_bvar as rb

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 200)

# Same simulated data as ex_a.py
rng = np.random.default_rng(0)
A = np.array([[0.4, 0.0, -0.001], [0.1, 0.6, 0.0], [15.0, 25.0, 0.85]])
mu = np.array([0.0075, 0.006, 4.0])
sd = np.array([0.007, 0.003, 0.25])
corr = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.2], [0.3, 0.2, 1.0]])
L = np.linalg.cholesky(corr * np.outer(sd, sd))
y, Y = mu.copy(), []
for _ in range(200 + 240):
    y = (np.eye(3) - A) @ mu + A @ y + L @ rng.standard_normal(3)
    Y.append(y)
Y = np.array(Y[200:])
raw = pd.DataFrame(
    {"gdp": 100 * np.exp(np.cumsum(Y[:, 0])), "prices": 50 * np.exp(np.cumsum(Y[:, 1])),
     "rate": Y[:, 2]},
    index=pd.period_range("1960Q1", periods=240, freq="Q"),
)
data = rb.VARData.from_frame(raw, tcodes={"gdp": 5, "prices": 5, "rate": 1})
print(platform.platform(), "| rai-bvar", rb.__version__)

# MCMC engine: same Linear mean, stochastic volatility; form="auto" resolves to "structural".
mc = rb.MCMCSettings(chains=4, draws=1000, burn=1000)
sv = rb.BVAR(vol=rb.SV(), lags=2, mcmc=mc)
print(sv.engine, "|", sv.name, "|", sv.mean.prior.form, "|", sv.axes)
t0 = time.perf_counter()
fit = sv.fit(data.upto("2019Q3"))
print(f"fit seconds (incl. compile): {time.perf_counter() - t0:.1f}")
s = fit.summary()
print("prior mode:", s.attrs["prior_mode"], "| hyper:", s.attrs["hyper"], "| converged:", fit.converged)
print(s[["mean", "sd", "rhat", "ess_bulk"]].round(3))
fc = fit.forecast(horizon=4, draws=2000, seed=0)
print(fc, "| engine attr:", fc.dataset.attrs["engine"])
y_obs = data.values[-1]  # 2019Q4, transformed scale
print("joint LPD h=1 and MCSE:", fc.log_pred_density(y_obs, horizon=1, return_mcse=True))
print("marginal LPD (rate):", fc.log_pred_density(y_obs[2:], horizon=1, variables=["rate"]))
irf = fit.irf(horizon=8, draws=500, seed=0)
print(irf)
print(irf.median().sel(horizon=[0, 4, 8], shock="rate").round(4).to_pandas())

# Volatility-axis comparison: both arms structural Mode I, differing only in vol (R2).
models = {
    "Linear x Constant": rb.BVAR(vol=rb.Constant("ig"), lags=2, mcmc=mc),
    "Linear x SV": sv,
}
for k, m in models.items():
    print(k, m.axes)
t0 = time.perf_counter()
res = rb.compare(
    models, data,
    origins=rb.Origins(start="2017Q1", end="2019Q3"),
    window=rb.Rolling(size=120),
    horizons=[1],
    draws=1000,
    baseline="Linear x Constant",
)
print(f"compare seconds: {time.perf_counter() - t0:.1f}")
print(res.table()[["origins", "dlpd", "dlpd_mcse", "dm_pvalue", "tests_valid", "in_mcs_25", "converged"]].round(3))
v = res.verdict(axis="vol")
print(v[["model", "reference", "axis", "horizon", "verdict"]])
for r in v.itertuples():
    print(r.reasons)
```

Output on stdout, verbatim. The two timing lines (`fit seconds …: 5.6`, `compare seconds: 14.1`)
depend on the run. The fact-check re-run printed 4.7 and 11.9 and every other line identically
(timing caveat added by fact-check):

```text
macOS-26.6.2-arm64-arm-64bit | rai-bvar 0.0.1
mcmc | BVAR(Minnesota(structural, eb), SV, p=2, mcmc=45b567) | structural | {'mean': 'Linear', 'prior': 'Minnesota(structural, eb)', 'vol': 'SV', 'lags': '2', 'deterministic': 'c', 'outliers': 'none'}
fit seconds (incl. compile): 5.6
prior mode: Mode I | hyper: {'hierarchy': 'eb', 'lambda': 0.24549046712778935, 'alpha': 2.0, 'cross': 1.0, 'contemp': 1.0, 'var_const': 10000000.0, 'eb_method': 'conjugate_twin', 'eb_converged': True, 'offset_rule': 'relative', 'c_rel': 1e-12} | converged: False
                    mean     sd   rhat  ess_bulk
quantity                                        
loglik          1928.304  7.948  1.031   179.391
B_own1[gdp]        0.461  0.069  1.000  3010.071
B_own1[prices]     0.558  0.065  0.999  2735.739
B_own1[rate]       0.948  0.054  1.001  3537.035
mu[gdp]          -10.108  0.129  1.003   665.300
mu[prices]       -11.834  0.160  1.001   757.856
mu[rate]          -2.892  0.155  1.002   692.202
phi[gdp]           0.806  0.118  1.035   131.873
phi[prices]        0.831  0.111  1.034   125.015
phi[rate]          0.861  0.110  1.017   197.937
sigma[gdp]         0.156  0.112  1.036   134.415
sigma[prices]      0.200  0.113  1.042   109.911
sigma[rate]        0.088  0.062  1.008   362.438
h_T[gdp]         -10.226  0.356  1.009   756.713
h_T[prices]      -11.766  0.361  1.010  1264.266
h_T[rate]         -2.831  0.246  1.002   887.965
ForecastResult(origin=2019Q3, horizon=4, variables=['gdp', 'prices', 'rate'], samples=2000) | engine attr: mcmc
joint LPD h=1 and MCSE: (8.575356603927622, 0.007996504228586188)
marginal LPD (rate): -0.4000308979305567
IRFResult(horizon=8, variables=['gdp', 'prices', 'rate'], shock_size='sd', samples=500)
response     gdp  prices    rate
horizon                         
0         0.0000     0.0  0.2397
4        -0.0002     0.0  0.1230
8        -0.0001    -0.0  0.0456
Linear x Constant {'mean': 'Linear', 'prior': 'Minnesota(structural, eb)', 'vol': 'Constant', 'lags': '2', 'deterministic': 'c', 'outliers': 'none'}
Linear x SV {'mean': 'Linear', 'prior': 'Minnesota(structural, eb)', 'vol': 'SV', 'lags': '2', 'deterministic': 'c', 'outliers': 'none'}
compare seconds: 14.1
                           origins   dlpd  dlpd_mcse  dm_pvalue  tests_valid  in_mcs_25  converged
model             horizon                                                                         
Linear x Constant 1             11  0.000      0.000        NaN         True       True        1.0
Linear x SV       1             11 -0.107      0.102      0.842         True       True        0.0
         model          reference axis  horizon       verdict
0  Linear x SV  Linear x Constant  vol        1  inconclusive
ΔLPD=-0.11 (MC s.e. 0.10, not precise enough); DM p=0.421 (valid: rolling window); MCS25: model in, reference in; converged at only 0% of origins
```

`UserWarning`s on stderr (file/line prefixes removed; the first two verbatim, the `compare()` ones
summarised):

```text
UserWarning: MCMC not converged: 7 of 16 primary quantities fail R-hat < 1.01 or bulk ESS > 400 (worst: sigma[prices] R-hat 1.042, ESS 110; sigma[gdp] R-hat 1.036, ESS 134; phi[gdp] R-hat 1.035, ESS 132; phi[prices] R-hat 1.034, ESS 125; loglik R-hat 1.031, ESS 179); increase draws, burn or thin
UserWarning: non-centred sign fold in more than 1% of sweeps for ['gdp', 'prices', 'rate'] (fractions [0.0555, 0.0135, 0.0795]): near-homoskedastic equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)
[compare(): one "Linear x SV: MCMC not converged: 6–8 of 16 primary quantities fail …" message per origin (11 origins, 2017Q1..2019Q3), and one aggregated fold warning "Linear x SV: non-centred sign fold … (and 10 variants with other numbers) [at 11 of 11 origins: 2017Q1..2019Q3]"]
```

What it shows:

- The same `Linear(prior=Minnesota())` resolves to structural Mode I once the volatility is
  `SV()`.
- EB is plugged in from the conjugate twin (`eb_method: conjugate_twin`).
- The forecast container and its LPD API are identical to the analytic one (R3; `attrs["engine"]`
  differs).
- With rate ordered last, the rate shock has no impact effect on GDP and prices.
- On a homoskedastic DGP the SV arm's φ and σ mix slowly (bulk ESS 110–362 from 4000 kept draws) and
  the fold warning fires. The convergence gate then forces `inconclusive`, independently of the
  (imprecise, negative) ΔLPD.

---

## 14. Key equations the paper must state (checklist, in code notation)

| # | Equation | Source |
| :--- | :--- | :--- |
| E1 | Regressor vector `x_t = (d_t', y_{t−1}', …, y_{t−p}')'`, `k = n_d + np` | §3.1; `design.py:1-5` |
| E2 | Recursive structural form `A y_t = Γx_t + Λ_t^{1/2}ε_t`; `B' = A⁻¹Γ`; `Σ_t = A⁻¹Λ_tA⁻¹'` | §3.2; SPEC §4.1 |
| E3 | Likelihood with `|A| = 1` (equation-by-equation Gaussian) | §3.3; N05 §3 |
| E4 | Conjugate NIW prior with `Ω_{(l,j)} = κ²(d−n−1)/(l^αψ_j)`, `d = n+2`, transform-aware `b` | §5.1; `analytic.py:378-382` |
| E5 | NIW log marginal likelihood; dummy correction `ML(d+Y) − ML(d)`; LP Jacobian `−nΣ log s_t` | §5.1; `analytic.py:8-15` |
| E6 | SOC and SUR rows (with `soc_scope`) | §5.2; `dummies.py:8-10` |
| E7 | GLP hyperpriors and the EB objective `log p(Y|h) + log p(h)` | §5.3; `hyperpriors.py:206-217` |
| E8 | LP scale path `s_t` and its priors | §5.4; `outliers.py:3-7` |
| E9 | D3 centring `E[γ_i|a_i] = Mã_i`; δ-coordinate regression `y_i = (XM_{<i} − Y_{<i})a_i + Xδ_i + e_i` | §5.5; N05 §4.1 |
| E10 | Mode I table; Mode II (λ-scaled) prior; `iw_equivalent`; the NIW ⇔ structural-NIG proposition | §5.5; DESIGN §6.1; N05 §4.2 |
| E11 | Reduced-form prior `π_j ~ N(m_j, V_j)` and the CCCM conditional `(K_j, b_j)` with `W = (1/Λ)(A∘A)` | §5.6; N04 §5.3 |
| E12 | Constant `λ_i ~ IG(1+ι/2, ψ_ι/2)` and the coupled IG conditional | §6.1; DESIGN §4.7 |
| E13 | SV law (AR, RW) and default priors; the mixture observation `y* = log(ẽ² + c)` and the exact-correction weight `w(h)` | §6.2; DESIGN §3.7.4 |
| E14 | Robust SV: `e = o√τ e^{h/2}ε`, outlier pmf, `p` prior, standardized t, `ν` grid; transitory-integrated `f(z)` | §6.3; N03 §0; `result.py:42-61` |
| E15 | Forecast recursion `m = A⁻¹Γx`, `L = A⁻¹ diag(e^{½ log λ^eff})` | §10.1; `forecast.py:5-19` |
| E16 | IRF recursion `Φ_h`, `Θ_h = Φ_hL`, unit shock `Φ_hA⁻¹`; FEVD | §10.2; `irf.py:1-25` |
| E17 | RB LPD estimator and its ESS-adjusted MCSE | §11.2; `result.py:218-224` |
| E18 | DM-HLN statistic and the rectangular-kernel LRV; the MCS `T_max` | §12.3 |
| E19 | Verdict rule including the convergence gate | §12.4; `compare.py:245-287` |

---

## 15. SPEC ↔ code discrepancies (to resolve before submission)

| # | SPEC says | Code at `62d15f6` | Status |
| :--- | :--- | :--- | :--- |
| D1 | §9.3: DM "with HAC variance (Newey-West, bandwidth at least h − 1)" | rectangular kernel with `h−1` autocovariances; Bartlett (NW) only if that is not positive (`significance.py:35-53`) | to check (amend SPEC or code; the paper must state the kernel) |
| D2 | §9.2: energy score pairwise term "estimated from random pairs" | fair estimator on a random subsample of ≤ 1000 draws (`scores.py:61-63`, `:87-95`) | to check |
| D3 | §7: `store_moments="lazy"` | not implemented (no occurrence in `src/`) | to check |
| D4 | §9.5: "Monte Carlo standard error of ΔLPD (batch means across chains and draws)" | per-origin delta-method MCSE (× `√(S/ESS)` for MCMC), summed in quadrature over origins and the two models (`result.py:275-288`; `compare.py:331-332`) | to check (independence across models is questionable: both models use the same per-origin forecast seed, `backtest.py:387`) |
| D5 | §5.6: `"lenza-primiceri"` for "analytic engine and constant-volatility MCMC models"; option `"exclude"` | LP rejected on MCMC (`api.py:159-168`; D-21); `outliers` accepts only `None`, `"lenza-primiceri"` or `LenzaPrimiceri()` (`api.py:96-101`) | to check (amend SPEC; LP-on-MCMC is Phase 3) |
| D6 | §5.4: exogenous regressors with future values at forecast time | only `"n"`, `"c"`, `"ct"` (`design.py:13`) | roadmap; amend SPEC |
| D7 | §9.3: subsample analysis (pre-2008, 2008–09, …) in `compare()` | not in `compare.py` | to check |
| D8 | §9.5 table: "ordering sensitivity (if run)" column; "DM/AG p-value with validity flag" | no ordering column; table p-value two-sided, verdict one-sided (`compare.py:166`, `:250-251`) | to check (document) |
| D9 | §9.1: `rb.estimate_cost(grid, data, origins)` | `estimate_cost(model, data, *, origins, …)`, one model (`backtest.py:296-332`) | minor; amend SPEC |
| D10 | §4.2 interface sketch; §8 "all without a loop" attribution | superseded by DESIGN §3; attribution to be corrected (DESIGN Q12, N07 §8.3) | cite DESIGN, not SPEC §4.2 |
| D11 | §10.6: diagnostics "via ArviZ" | in-repo NumPy `diagnostics.py` (D-24); ArviZ optional | amend SPEC |
| D12 | §13 Phase 2 exit: `bayesianVARs` parity for the reduced-form sampler | deferred; the system-wide Gibbs oracle replaces it (DESIGN Stage 4 amendment (2)) | to check |
| D13 | README: "Status: Phase 1 (analytic engine)"; quick-start `res.verdict()` on RW vs BVAR | Phase 2 merged; that verdict is "not comparable" (§13.1) | fix README before release; do not cite README for status |
| D14 | `fred.py` docstring cites "McCracken & Ng 2016, 2020" | SPEC §15 cites FRED-QD as McCracken & Ng (2021) | citation to check |
| D15 | §7: `ForecastResult` attrs include "variable ordering, transforms, …"; R8: "every result records the ordering" | MCMC forecasts and every `IRFResult` carry an `ordering` attr (`engines/mcmc/forecast.py:309`; `structural/irf.py:195`); analytic forecasts do not (`analytic.py:696-707`), and their order lives only in the `variable` coordinate. Transforms are recorded as `tcodes` | to check (add the attr to analytic forecasts, or state that the coordinate order is the record) (row added by fact-check) |

---

## 16. Paper-worthy contributions of the framework (candidate claims with status)

1. **The composition chooses the engine.** One `BVAR(mean, vol, prior)` façade with automatic
   analytic/MCMC resolution, an explicit 9-row resolution table and errors at construction (§7).
   Verified *(re-run here)*.
2. **Structural-form Minnesota with D3 centring is the structural form of the conjugate NIW
   prior.** The log ML equals the Phase 1 engine's to `1e-10` with SOC/SUR, trend and n = 1
   (§5.5). Verified *(re-run here)*. It underpins the analytic twin oracle (validation dossier).
3. **Moment-matched independent prior (Mode I)** that reproduces the NIW reduced-form mean,
   marginal variances and `E[Σ]` without the `2^{i−1}` growth (§5.5). Verified.
4. **Fair-ablation machinery:** axis labels, grid harmonisation, Mode II flagged as a prior
   change, EB from the conjugate twin so the mean prior is identical across volatility arms
   (§5.7, §8). Verified (tests); the R2 rationale is a design argument (DESIGN §6.5).
5. **A predictive contract with conditional Gaussian moments** that gives RB log scores without
   KDE, exact marginal subsets and ESS-aware MCSE (§11). Verified. Its documented **h > 1 tail
   failure** is a methods point in itself (§11.3). Recorded; re-run with the fix.
6. **A verdict engine that combines precision, a valid test or the MCS, and a convergence gate.**
   The CR replication shows the gate decides the headline verdict (24/24 → 0/24) (§12.4).
   Recorded; the report is to be replaced. Size and power on DGPs: to check (not yet tested).
7. **Ordering dependence made measurable:** an ordering-sensitivity check with a noise floor, an
   order-invariant reduced × `iw_equivalent` option and pooled orderings (§10.3). Verified (fast
   tests in the suite); the nightly invariance test was not re-run.
8. **No look-ahead as an API property:** per-window recomputation of ψ, own-lag means, EB values,
   SV level centres and dummy rows. The CR report asserts bitwise-identical resolved priors when
   the vintage is cut at the origin (report "Design", R9 assertion at 1985Q1, 2007Q4, 2011Q2).
   Recorded.

---

## 17. Open questions

- **Q1 Notation.** Unify `κ` (code/DESIGN) vs `λ` (SPEC) for the Minnesota tightness, `ι` vs `i`,
  and the IG (shape, scale) convention. Note also that `cross` scales the **s.d.** while ψ is in
  variance units. A notation table should go in the paper's §2.
- **Q2 h > 1 LPD.** Adopt the path-integrated estimator as the contract (a SPEC change) or keep the
  path-conditioned estimator and flag degenerate cells? The fix is not on `paper-draft`. Which
  committed commit will the paper cite?
- **Q3 Convergence gate.** Is "all primary quantities at R-hat < 1.01 and bulk ESS > 400 at
  ≥ 95% of origins for both models" the intended operating point? The CR report and the example
  show it can override every other evidence for SV models, especially near-homoskedastic
  equations (fold warnings). Options: report gate-on and gate-off verdicts; gate on a smaller set;
  adaptive draws per origin.
- **Q4 MCSE of ΔLPD.** The quadrature sum assumes independence across origins and between the two
  models' estimates, but both models share the per-origin forecast seed. Is the 3·MCSE rule
  conservative or anti-conservative? It needs a derivation or a simulation.
- **Q5 DM kernel.** Rectangular with Bartlett fallback (code) vs Newey–West (SPEC). Decide, and
  state it in the paper.
- **Q6 MCS under expanding windows.** The verdict accepts the MCS route even when DM p-values are
  "descriptive" (expanding windows, as in the CR replication). Is the MCS route valid there?
- **Q7 Horizons.** The verdict is per horizon; SPEC says "at the horizons of interest". Should a
  multi-horizon rule exist?
- **Q8 Defaults to justify.** ψ by AR(p) (Chan uses AR(4)); α fixed at 2; `var_const = 1e7`; the
  data-centred SV μ prior with variance 10 (DESIGN Q3, risk R6); `Beta(20, 1.5)` on `(φ+1)/2`;
  RW `B_σ = 0.1`; `soc_scope="persistent"` vs GLP/R `BVAR`'s `"all"`.
- **Q9 EB as plug-in for SV models.** It is type-II ML only for the Mode II twin. Does the plug-in
  bias SV vs Constant comparisons? (A sampled λ would break R2 per DESIGN §6.5; quantify instead.)
- **Q10 Which "Constant" baseline the paper uses.** MCMC Mode I `Constant("ig")` (grid
  harmonisation), Mode II, analytic NIW, or reduced `Constant(iw_equivalent)` (the CR report)?
- **Q11 `rai` vs official tcodes.** Confirm the official 2026-09 FRED-QD codes for GDPCTPI and
  FEDFUNDS, and state the transformations in the paper.
- **Q12 Unimplemented SPEC items** (D3, D5–D7). Drop from SPEC, move to the roadmap, or implement
  before submission?
- **Q13 README.** Update the status and the quick start so the documented verdict is meaningful
  (e.g. a vol-axis grid).
- **Q14 Contract coverage.** Add `reduced × SV(t)`, `reduced × SV(rw)` and
  `reduced × SV(outliers,t)` to `tests/mcmc/forecast/mcmc_fits.py::COMPOSITIONS`, or state that
  the contract test covers 12 compositions.
- **Q15 DGP size/power** of the verdict engine (SPEC §10.5) is a Phase 3 exit item. Can the paper
  make any statement about false-positive control without it?
- **Q16 bayesianVARs parity** for the reduced form is deferred (no gfortran). Is the system-wide
  oracle enough for the paper's claim?
- **Q17 Third-party claims** (NOTES A.1–A.2: stochvol ASIS sign handling; stochvol 3.2.1 t-SV).
  Re-test on current stochvol and contact the maintainers before mentioning them.
- **Q18 Scores on the transformed scale only.** Should level-scale (`paths_level`) scoring be
  offered, given the pathwise inverse exists?
- **Q19 Energy/variogram subsample of 1000 draws.** Is its Monte Carlo noise reported anywhere? It
  is not in `table()`.

---

## 18. References mentioned (all [CITE?]: the bibliography stage must verify)

Entries marked "SPEC §15" are listed with full details in `SPEC.md` §15; the others are cited in
the notes named, or are standard software references I added.

- Adolfson, M., Lindé, J., & Villani, M. (2007). Forecasting performance of an open economy DSGE model. *Econometric Reviews*, 26(2–4), 289–328. (Gaussian-approximation log score in the CR report.) [CITE?]
- Amisano, G., & Giacomini, R. (2007). Comparing density forecasts via weighted likelihood ratio tests. *JBES*, 25(2), 177–190. (SPEC §15) [CITE?]
- Arias, J. E., Rubio-Ramírez, J. F., & Shin, M. (2023). Macroeconomic forecasting and variable ordering in multivariate stochastic volatility models. *J. Econometrics*, 235(2), 1054–1086. (SPEC §15) [CITE?]
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions. *J. Applied Econometrics*, 25(1), 71–92. (Dummy-observation implementation of the Minnesota prior; not in SPEC.) [CITE?]
- Bognanni, M. (2022). Comment on "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *J. Econometrics*, 227(2), 498–505. (SPEC §15) [CITE?]
- Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs (software). [CITE?]
- Carriero, A., Chan, J. C. C., Clark, T. E., & Marcellino, M. (2022). Corrigendum to "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors". *J. Econometrics*, 227(2), 506–512. (SPEC §15) [CITE?]
- Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors. *J. Econometrics*, 212(1), 137–154. (SPEC §15) [CITE?]
- Carriero, A., Clark, T. E., Marcellino, M., & Mertens, E. (2024). Addressing COVID-19 outliers in BVARs with stochastic volatility. *REStat*, 106(5), 1403–1417. (SPEC §15) [CITE?]
- Chan, J. C. C. (2022). Asymmetric conjugate priors for large Bayesian VARs. *Quantitative Economics*, 13(3), 1145–1169. (SPEC §15) [CITE?]
- Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *IJMMNO*, 1(1/2), 101–120. (SPEC §15) [CITE?]
- Chan, J. C. C., Koop, G., & Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic volatility. *JBES*, 42(2), 825–837. (SPEC §15) [CITE?]
- Clark, T. E., & McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. *J. Econometrics*, 105(1), 85–110. (SPEC §15) [CITE?]
- Clark, T. E., & Ravazzolo, F. (2015). Macroeconomic forecasting performance under alternative specifications of time-varying volatility. *J. Applied Econometrics*, 30(4), 551–575. (SPEC §15) [CITE?]
- Del Negro, M., & Primiceri, G. E. (2015). Time varying structural vector autoregressions and monetary policy: A corrigendum. *RES*, 82(4), 1342–1345. (SPEC §15) [CITE?]
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263. (SPEC §15) [CITE?]
- Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*, 3(1), 1–100. (Origin of the sum-of-coefficients prior; not in SPEC.) [CITE?]
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545–1578. (SPEC §15) [CITE?]
- Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *REStat*, 97(2), 436–451. (SPEC §15) [CITE?]
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378. (SPEC §15) [CITE?]
- Gneiting, T., & Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES*, 29(3), 411–422. (SPEC §15) [CITE?]
- Gupta, A. K., & Nagar, D. K. (2000). *Matrix Variate Distributions*. Chapman & Hall/CRC. (IW partition theorem, N05 §1.) [CITE?]
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497. (SPEC §15) [CITE?]
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF*, 13(2), 281–291. (SPEC §15) [CITE?]
- Hosszejni, D., & Kastner, G. (2021). Modeling univariate and multivariate stochastic volatility in R with stochvol and factorstochvol. *JSS*, 100(12). (N05 §1; stochvol parametrisation.) [CITE?]
- Hoyer, S., & Hamman, J. (2017). xarray: N-D labeled arrays and datasets in Python. *J. Open Research Software*, 5(1). (Software; not in SPEC.) [CITE?]
- Jacquier, E., Polson, N. G., & Rossi, P. E. (2004). Bayesian analysis of stochastic volatility models with fat-tails and correlated errors. *J. Econometrics*, 122(1), 185–212. (Student-t SV, ν grid; N03.) [CITE?]
- Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for estimation and inference in Bayesian VAR-models. *J. Applied Econometrics*, 12(2), 99–132. (SPEC §15) [CITE?]
- Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *CSDA*, 76, 408–423. (SPEC §15) [CITE?]
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *RES*, 65(3), 361–393. (SPEC §15) [CITE?]
- Kuschnig, N., & Vashold, L. (2021). BVAR: Bayesian vector autoregressions with hierarchical prior selection in R. *JSS*, 100(14), 1–27. (SPEC §15; the JSS precedent for this paper.) [CITE?]
- Lenza, M., & Primiceri, G. E. (2022). How to estimate a vector autoregression after March 2020. *J. Applied Econometrics*, 37(4), 688–699. (SPEC §15) [CITE?]
- Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: Five years of experience. *JBES*, 4(1), 25–38. (Origin of the Minnesota prior; not in SPEC.) [CITE?]
- McCausland, W. J., Miller, S., & Pelletier, D. (2011). Simulation smoothing for state-space models: A computational efficiency analysis. *CSDA*, 55(1), 199–212. (SPEC §15) [CITE?]
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *JBES*, 34(4), 574–589. (SPEC §15) [CITE?]
- McCracken, M. W., & Ng, S. (2021). FRED-QD: A quarterly database for macroeconomic research. *FRB St. Louis Review*, 103(1), 1–44. (SPEC §15; `fred.py` says 2020; see D14.) [CITE?]
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. (Bartlett fallback; not in SPEC.) [CITE?]
- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *J. Econometrics*, 140(2), 425–449. (SPEC §15) [CITE?]
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *RES*, 72(3), 821–852. (SPEC §15) [CITE?]
- Scheuerer, M., & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review*, 143(4), 1321–1334. (SPEC §15) [CITE?]
- Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic multivariate models. *International Economic Review*, 39(4), 949–968. (Conditional centring precedent, N05 §1; also the usual SUR-prior citation.) [CITE?]
- Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S. monetary policy? *AER*, 96(1), 54–81. (SPEC §15; N05 §4.2 for `A_+ = D + SA_0`.) [CITE?]
- Stock, J. H., & Watson, M. W. (2016). Core inflation and trend inflation. *REStat*, 98(4), 770–784. (Discrete outlier-scale grid, cited in N03 §0 as "SW (2016)".) [CITE?]
- `scoringrules` (Python package, Apache-2.0; version 0.11.0 used here). The authors and the citable reference are to be identified. [CITE?]
- van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers: Theory and methods. *JASA*, 103(482), 790–796. (Collapsed transitory steps, DESIGN §4.6 item 3.) [CITE?]
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718. (SPEC §15) [CITE?]

---

## 19. Fact-check record (adversarial pass, 2026-10-23)

Scope: every factual claim and number about the package in §0–§17 was checked against commit
`62d15f6` (`git rev-parse HEAD` = `62d15f68c7e7a3de4ce83ee0098293b9c62111b0`, branch `paper-draft`).
Literature citations were not checked; a later stage does that.

**Re-run by the fact-checker** (same environment as §0: `platform.platform()` and package versions
re-read and identical, Apple M5 Max):

| What | Result |
| :--- | :--- |
| §0.1 command 1 | 577 passed, 110 skipped (matches) |
| §0.1 command 2 (`-m slow` contract) | 100 passed, 20 deselected (matches; composition count corrected in §0.1) |
| §0.1 command 3 | 694 passed, 9 skipped (matches) |
| §13.1 snippet A | stdout identical line for line; no stderr |
| §13.2 snippet B | stdout identical except the two timing lines (4.7 s, 11.9 s); stderr warnings identical in content |
| Every test name cited in §1–§12 (44 names) | all exist at the cited files (`grep "def <name>"`) |

**Opened and confirmed** (no change needed). These are the cited line ranges of `api.py`,
`__init__.py`, `components/specs.py`, `priors/{minnesota,hyperpriors,dummies,structural,reduced,
sv,sv_arrays,scales}.py`, `data/{dataset,design,transforms,outliers}.py`, `datasets/fred.py`,
`engines/analytic.py`, `engines/mcmc/{forecast,monitor,resolve,results}.py`, `forecast/result.py`,
`forecast/invariants.py`, `evaluation/{backtest,compare,scores,significance}.py`,
`structural/{irf,ordering}.py`, `settings.py`, `models/benchmarks.py`. Also: the SPEC sections cited
(§0, §1.3, §3, §4.1–4.6, §5, §6.1, §6.6, §7, §8, §9, §10.5–10.6, §13, §15 Data); DESIGN §0.1–0.2,
§2.9–2.10, §4.1–4.8, §6, §7, §8, §11, §12; N04 §0, §5; N05 §0, §2–§4.2; NOTES A–E with their
statuses; the CR report (acceptance criteria, Design, Reliability, SV(rw) full table, verdict table,
24-ordering table); `benchmarks/clark_ravazzolo.py:493-515`; `tests/fixtures/fred_qd_sample.csv`
header; `tests/conftest.py:23-41`; `tests/mcmc/forecast/mcmc_fits.py:128-141`; and the absence of
`tests/dgp/` and of `store_moments`/`lazy` in `src/`.

**Corrections made inline** (each marked "corrected by fact-check" or "added by fact-check"). The
§0.1 contract-test composition count; R8 and §11.1 (`ordering` attr not on analytic forecasts; new
§15 D15); the §2 DESIGN line range; ψ under LP (fallback rule); §5.2 "documented choice" attribution;
§5.3 MH proposal floor wording, the refit condition and `chains > 1`; the §5.6 N04 section pointer;
the §6.1 fast path of `iw_equivalent`; the §6.2 Q3 status (decided, R6 open); the §6.3 particle-filter
hook attribution; §7 `Runtime` invariance (devices only to rounding) and SV label hashing; the §8
grid-harmonisation rule; §10.3 a note on the CR report's `material = True` for the order-invariant
arm; the §12.1 backtest signature and the warm-start cache-key reasoning; §13 timing caveats.

**Process note.** `.hypothesis/` (gitignored) is dated 10:35, after the 10:32 checkout of `.venv/`,
and `.hypothesis/constants/` 10:59. So "pre-existed" in §0 holds only relative to that run, and the
hypothesis-based tests of command 3 (`test_spec_roundtrip_mcmc.py`) may have written to that
gitignored cache. No tracked file changed: `git status --short` shows only `?? docs/paper/dossiers/`,
before and after this fact-check. The fact-checker did not run the further hypothesis-based tests
(`test_transforms.py`, fast `test_ordering.py`); their status stays "not re-run here".

## Unsupported claims

No claim about the package remains unsupported after the inline corrections. The two claims that
did not match the code were fixed in place rather than moved here: the `ordering` attr on analytic
forecasts (§1, §11.1) and the "12 compositions" description of the slow contract run (§0.1).

These statements rest on the task statement or the session environment, not on a committed file in
this repository. Keep them out of the paper unless a committed source appears:

- §11.3: "An implementation [of the path-integrated estimator] is in progress outside this branch."
  Evidence: the task statement and an uncommitted working tree elsewhere. Nothing at `62d15f6`.
- §12.4: the CR report "will be replaced" by an improved acceptance run. Evidence: the task statement
  only.
- §13: the wall-clock times (12.3 s, 23.1 s, 5.6 s, 14.1 s). They were recorded on one run and are
  not reproducible numbers. The fact-check re-run gave 4.7 s and 11.9 s for the two printed timings.
