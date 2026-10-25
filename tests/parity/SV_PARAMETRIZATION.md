# Stochastic-volatility parity: stochvol, bayesianVARs and what the tests compare

This note records how the R packages `stochvol` and `bayesianVARs` parametrise the
stochastic-volatility (SV) model, how rai-bvar's `SV` maps onto them, and what the Phase 2 parity
tests (`test_stochvol.py`, `test_bayesianvars.py`; DESIGN §9.9, work package WP21) found. The
fixtures are generated offline by WP8 (`tools/make_sv_parity_data.py`, `tools/make_r_fixtures.R`;
layout in `test_fixture_schema.py`).

Clean room (SPEC §2.1): the R packages were only *run* as numerical oracles. Their GPL sources
were not read for this note; every statement about their behaviour below comes from their manuals,
from N02/N04 (which cite them), or from experiments with the installed package.

Versions: R 4.1.2 (x86_64-apple-darwin17.0), stochvol 3.2.1 (DESIGN §9.9 names 3.2.9; 3.2.1 is
the installed version). bayesianVARs 0.1.8 is **not installed** (§4). Phase 3 (§7): R 4.5.3,
stochvol 3.2.9, bayesianVARs 0.1.8 and bvars 1.1, from the dispatch-only CI job `r-fixtures`.

## 1. The SV model and the parameter mapping (DESIGN §6.3; N02 §1, §10.2, §13.6)

One series, `h_t` the log variance, `t = 0` the initial state:

```
y_t = exp(h_t/2) ε_t,                    ε_t ~ N(0, 1)                     t = 1..T
h_t = μ + φ(h_{t−1} − μ) + σ η_t,        h_0 ~ N(μ, σ²/(1 − φ²))  (stationary)
μ ~ N(b_μ, B_μ)    (φ + 1)/2 ~ Beta(a_φ, b_φ)    σ² ~ B_σ·χ²₁  (⇔ σ ~ |N(0, B_σ)|)
```

Student-t errors (`tails="t"`, D-8): `y_t = exp(h_t/2) √τ_t ε_t` with the **standardized**
`τ_t ~ IG(ν/2, (ν − 2)/2)` (`E τ = 1`, `Var(y_t | h_t) = e^{h_t}`). stochvol's simulator
`svsim(…, nu = 5)` uses the same convention (checked: mean of `τ` 0.997, `var(y e^{−h/2})` 0.995
over 200,000 draws).

| rai-bvar (`rb.sv_priors`) | `svsample()` | `specify_priors()` | bayesianVARs `specify_prior_sigma(type = "cholesky")` |
| :--- | :--- | :--- | :--- |
| `mu_prior=Normal(b_μ, B_μ)` (`B_μ` a variance) | `priormu = c(b_μ, √B_μ)` | `mu = sv_normal(b_μ, √B_μ)` | `cholesky_priormu = c(b_μ, √B_μ)` |
| `phi_prior=Beta(a_φ, b_φ)` (on `(φ+1)/2`) | `priorphi = c(a_φ, b_φ)` | `phi = sv_beta(a_φ, b_φ)` | `cholesky_priorphi = c(a_φ, b_φ)` |
| `sigma2_prior=ScaledChi2(B_σ)` | `priorsigma = B_σ` (not `√B_σ`) | `sigma2 = sv_gamma(0.5, 0.5/B_σ)` | `cholesky_priorsigma2 = c(0.5, 0.5/B_σ)` (shape must stay 0.5) |
| `h0="stationary"` | `priorlatent0 = "stationary"` | `latent0_variance = "stationary"` | `cholesky_priorh0 = "stationary"` (compare `h_{1:T}` only) |
| `offset=0.0` (zero-offset parity mode) | `c = 0` on data without exact zeros | `myoffset = 0` | `expert_sv_offset = 0` |
| `exact=False` | the mixture sampler | | |
| `exact=True` | `expert = list(correct_model_misspecification = TRUE)` (ex-post reweighting) | | not offered |
| `tails="t", nu_prior=DiscreteUniform(ν, ν)` | — | `nu = sv_constant(ν)` | — |

`priorsigma = B_σ` and `sv_gamma(0.5, 0.5/B_σ)` are the same prior: `σ² ~ B_σ·χ²₁` is a Gamma with
shape ½ and rate `1/(2B_σ)`. stochvol's own print-out of the prior it used (fixture attribute
`priors`) is checked against the derived values, e.g. `sigma^2 ~ Gamma(shape = 0.5, rate = 5)` for
`B_σ = 0.1`.

## 2. stochvol parity (`test_stochvol.py`)

**Setup.** stochvol `svsample` runs, 4 chains × 50,000 draws after 5,000 burn-in (`thinlatent =
5`), on five committed datasets: (i) `T = 300, μ = −1, φ = 0.95, σ = 0.2`; (ii) `φ = 0.5, σ = 0.5`;
(iii) `φ = 0.99, σ = 0.1`; (iv) the standardised AR(4) OLS residuals of annualised real-GDP growth
(`T = 235`); (v) the `h` and `ε` paths of (i) with standardized `t_5` errors. Our side is the
production runner `fit_mcmc` on `ZeroMean × SV` with `n = 1` (so `e = y`), the `SV` static part
from `api.static_parts` and the prior arrays from `priors.sv_arrays`.

**One parameter set per run, both sides derived** (E11):

| run | parameter set `(b_μ, B_μ, a_φ, b_φ, B_σ)` | rai-bvar | R |
| :--- | :--- | :--- | :--- |
| main | `(0, 100, 20, 1.5, 0.1)` | `SV(mu_prior=Normal(0.0, 100.0), phi_prior=Beta(20, 1.5), sigma2_prior=ScaledChi2(0.1), offset=0.0, exact=False)` | `priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1, priorlatent0 = "stationary"` |
| defaults | `(0, 1e4, 5, 1.5, 1)` (stochvol's defaults) | `SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(5, 1.5), sigma2_prior=ScaledChi2(1.0), offset=0.0, exact=False)` | no prior argument |
| t5 | main, `ν = 5` | main + `tails="t", nu_prior=DiscreteUniform(5, 5)` | `specify_priors(mu = sv_normal(0, 10), phi = sv_beta(20, 1.5), sigma2 = sv_gamma(0.5, 0.5/0.1), nu = sv_constant(5))` |
| exact | main | main with `exact=True` | main + `correct_model_misspecification = TRUE`, summarised by self-normalised importance weights of the raw chains (no resampling) |

The test parses the fixture's `rai_bvar_spec` attribute (a literal evaluator, never `eval`) and
requires it to equal the spec derived from the parameter set, and requires the derived R arguments
in the recorded call.

**Time convention.** `h_t` belongs to `y_t`; `t = 0` is `h_0` (stochvol's `latent0`), our state
index 0. Our `h` path has `T_pad + 1` states with index `t` ↔ `h_t`; `τ` of observation `t` is
`exp(PathDraw.log_scale2[t − 1])` (no outliers, so `log_scale2 = log τ`).

**Compared quantities.** The 5/25/50/75/95% quantiles of `μ, φ, σ`, of `h_t` at
`t ∈ {0, 1, T/4, T/2, T}`, and for SV-t of `τ_t` at the three largest `|y_t|`.

**Statistic.** `z = (q̂_ours − q̂_R)/√(mcse_ours² + mcse_R²)` with type-7 quantiles of the pooled
chains and `mcse = max(Vehtari et al. 2021 §4.3 quantile MCSE, batch-quantile MCSE with 20 batches
per chain)` on both sides (the fixtures' method). Pass: `|z| ≤ max(4, z_{1 − 0.01/(2m)})` over the
`m` comparisons of a fixture. The second term is a Bonferroni guard at family level 1%; it exceeds
4 only for `m > 157`, so every stochvol bound is exactly 4 quantile-MCSE (DESIGN §9.9). A gross
mixing guard requires split R-hat < 1.05 for our `μ, φ, σ` (the MCSE accounts for the rest).

**D-7 exclusion.** stochvol's non-centred (ASIS) step takes `|σ|` without flipping `h̃`, which does
not preserve the posterior; ours folds `(σ, h̃) ↦ (−σ, −h̃)`. DESIGN §9.9 excludes fixtures where our
fold fraction (sweeps with an accepted non-centred `σ' < 0`) exceeds 0.5%. The test still compares
them, and a failure on an excluded fixture is reported as a skip rather than an error.

## 3. Findings

### 3.1 D-7 confirmed against an exact oracle

On a four-observation problem, `y = (0.4, −2.5, 0.3, 0.6)`, `μ ~ N(−1, 0.25)`, `(φ+1)/2 ~ Beta(20,
1.5)`, `σ² ~ 0.1·χ²₁`, the posterior is computed exactly by self-normalised importance sampling from
the prior (4 × 10⁶ draws, weight ESS about half of them). Quantiles 5/25/50/75/95% of `h_2`:

| source | `h_2` |
| :--- | :--- |
| exact (importance sampling) | −0.663, −0.259, 0.077, 0.486, 1.230 |
| rai-bvar, `exact=False` / `exact=True` | −0.663, −0.264, 0.074, 0.480, 1.225 / −0.665, −0.260, 0.078, 0.486, 1.236 |
| stochvol, centred parametrisation only | −0.659, −0.265, 0.071, 0.478, 1.215 |
| stochvol, non-centred only | −0.843, −0.303, 0.056, 0.469, 1.210 |
| stochvol, default (ASIS) | −0.837, −0.300, 0.059, 0.473, 1.218 |

(stochvol: 2 × 200,000 draws; rai-bvar: 4 × 25,000 here, 4 × 50,000 in the table of §3.2.) Our
fold fraction on this problem is 27%: with four observations `σ` keeps most of its prior mass near
0. stochvol is exact without the non-centred step and visibly biased with it, as N02 §7 predicts.
On the committed `T = 300` datasets our fold fraction is 0–1.3% (§6), and the parity holds there.

### 3.2 SV-t: stochvol 3.2.1 misses the standardized-t posterior (oracle deviation)

The SV-t fixture (`stochvol_v_t5`) disagrees with our sampler far beyond Monte Carlo error: our
posterior of `μ` is about 0.10 higher (`z` = 29 at 4 × 6,000 kept draws, 62 at 4 × 40,000), our
`τ_t` at the largest observations has a heavier upper tail (95% quantile 7.5 against 5.4), and the
median of `τ_t` over all `t` is 0.72 against 0.81. The same four-observation problem with `t_5` errors (`ν` fixed), whose exact
posterior is again computed by importance sampling with `τ` integrated out
(`y_t | h_t ~ t_ν(0, e^{h_t}(ν − 2)/ν)`), decides which side is right:

| source | `μ` | `h_2` | `τ_2` |
| :--- | :--- | :--- | :--- |
| exact (importance sampling) | −1.495, −1.042, −0.732, −0.425, 0.017 | −1.291, −0.759, −0.353, 0.124, 1.022 | 0.681, 1.361, 2.263, 3.888, 9.171 |
| rai-bvar, `exact=False` | −1.495, −1.044, −0.734, −0.426, 0.018 | −1.292, −0.759, −0.352, 0.126, 1.030 | 0.682, 1.358, 2.251, 3.871, 9.115 |
| rai-bvar, `exact=True` | −1.494, −1.044, −0.734, −0.426, 0.020 | −1.292, −0.759, −0.352, 0.125, 1.025 | 0.683, 1.359, 2.251, 3.871, 9.124 |
| stochvol, default (ASIS) | −1.549, −1.069, −0.743, −0.423, 0.029 | −1.630, −0.826, −0.340, 0.162, 1.018 | 0.697, 1.249, 1.971, 3.275, 7.681 |
| stochvol, centred only | −1.469, −1.016, −0.735, −0.422, 0.032 | −1.193, −0.691, −0.292, 0.178, 1.041 | 0.689, 1.211, 1.852, 2.907, 5.923 |

The unit-scale alternative (`τ ~ IG(ν/2, ν/2)`) does not explain stochvol's numbers either (its
exact posterior has `μ` median −0.82 and `τ_2` median 3.10). The deviation persists without the
non-centred step, so it is not D-7. Conclusion: stochvol 3.2.1's Student-t sampler does not target
the standardized-t SV posterior that its own simulator uses; rai-bvar matches the exact answer.
The reason inside stochvol was not investigated (that would mean reading GPL code).

Consequence for the tests: the SV-t row of DESIGN §9.9 cannot serve as an exit criterion with this
oracle. The PR tier instead runs `test_sv_t_matches_the_exact_importance_sampling_posterior`: our
SV-t on the four-observation problem against the exact importance-sampling posterior (10⁶ draws),
quantiles of `μ, φ, σ`, `h_0..h_4` and `τ_2` within 4 MCSE. It runs at the fixtures' `T_pad = 320`
(316 inert padded steps, D-15), so it shares one compiled program with the SV-t row.

The SV-t row itself (`test_stochvol_parity[stochvol_v_t5]` in the PR tier,
`test_stochvol_parity_long[stochvol_v_t5]` nightly) is a strict `xfail` narrowed to the oracle
deviation (`raises=OracleDeviation`). Before the stochvol mismatch is accepted, our `T = 300` run
must pass three checks:

- the R-hat guard;
- finite `z` on every comparison;
- a DGP-recovery guard: the true `μ, φ, σ` lie inside our central 99.8% posterior interval, and our
  pointwise 90% bands cover at least 70% of the true `h_t` and `log τ_t`.

Measured values of the DGP guard: posterior ranks of the truth 0.58, 0.55 and 0.75; coverage 0.85
(`h`) and 0.92 (`τ`), the same at both sizes. A crash, a NaN, non-mixing chains or a failed guard
therefore fail the test. An unexpected pass fails it too, since that would mean the oracle or our
sampler changed. Each of these four outcomes was checked with a mutated scratch copy.

**Version caveat.** Every statement in this section is about stochvol **3.2.1**, the version that
produced the committed fixtures. DESIGN §9.9 names 3.2.9; whether 3.2.9 (or later) still deviates
was not checked, because R is not available where this phase was finished. The finding is a
property of one fixture, not of stochvol in general.

**Unblock path for the SV-t row.** The strict `xfail` is meant to be removed, not kept:

1. Regenerate the one fixture under stochvol 3.2.9 or later, same seeds and dataset (v):
   `Rscript tools/make_r_fixtures.R --sections=stochvol --sv-only=stochvol_v_t5` (the script
   records `stochvol_version` in the fixture attributes).
2. Run the four-observation exact check of the table above against that version (a few lines of R
   with `svsample(…, priornu = …)`/`specify_priors(nu = sv_constant(5))`). If stochvol now matches
   the importance-sampling posterior, `test_stochvol_parity[stochvol_v_t5]` passes, the strict
   `xfail` turns that pass into a failure, and the `xfail` mark (and `OracleDeviation`) is deleted
   in the same change; the row then gates like the other fixtures.
3. If stochvol still deviates, keep the `xfail`, update the version and the table here, and report
   the four-observation example upstream; the exact importance-sampling test stays the SV-t exit
   criterion.

## 4. bayesianVARs: blocked in Phase 2 (superseded by §7)

**Phase 2 record.** Phase 3 installed bayesianVARs 0.1.8 and decided P1–P5; their outcomes are
in §7.1. The rest of this section describes the Phase 2 state.

**Status in Phase 2: bayesianVARs parity was blocked.** bayesianVARs 0.1.8 (first CRAN release 2023) has no
CRAN macOS binary for R 4.1, and a source build links its RcppArmadillo code against R's `FLIBS`,
i.e. against gfortran, which is not installed (`R CMD config FC` names `gfortran`; the `FLIBS`
directories `/usr/local/gfortran/lib…` do not exist). `fixtures/probe_bayesianvars.nc` records this
evidence (`bayesianvars_available = 0`). Without the package the probes cannot run:

| probe | question | documented | status |
| :--- | :--- | :--- | :--- |
| P1 | are `normal_sds` and `cholesky_normal_sds` variances (documented) or standard deviations (named)? | variances | unresolved (Phase 2) |
| P2 | is `U` stored as `U` or `U⁻¹`, and with which sign (`A = Uᵀ` or `2I − Uᵀ`)? | `Σ_t = U'⁻¹ D_t U⁻¹`, hence `A = Uᵀ`; GK App. B prints the U step without the minus sign | unresolved (Phase 2) |
| P3 | which row of the returned `PHI` draws holds the intercept? | undocumented | unresolved (Phase 2) |
| P4 | what is the layout of the returned `sv_para` draws? | undocumented | unresolved (Phase 2) |
| P5 | is `cholesky_priorhomoscedastic = cbind(shape, scale)` the IG scale `β₀` of `λ^{−α₀−1}e^{−β₀/λ}`, and what replaces `NA`? | (shape, scale), default `NA` with an undocumented substitute | unresolved (Phase 2) |

**Derived calls** (E11; recorded in the probe fixture's `parity_calls` and checked by
`test_bayesianvars_calls_derive_from_the_parameter_set`). From the one parameter set of §5 below:
`prior_intercept = 10` (an s.d.), `priormean = 0.5`, `cholesky_U_prior = "normal"`,
`cholesky_normal_sds = 1` (`1 = √1`, the same under both P1 readings), `sv_keep = "all"`; SV:
`cholesky_heteroscedastic = TRUE, cholesky_priormu = c(0, 100), cholesky_priorphi = c(20, 1.5),
cholesky_priorsigma2 = c(0.5, 0.5), cholesky_priorh0 = "stationary"`; Constant:
`cholesky_heteroscedastic = FALSE, cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))`. The
lag-coefficient `normal_sds` take the prior variances or their square roots by P1. Draws map back
by `A = t(U)` (P2), dropping the intercept row of `PHI` (P3) and reading `sv_para` by P4.

**Replacement oracle.** WP8 wrote an exact system-wide Gibbs sampler in R (`bvars_swa_*.nc`,
N04 §9.5): `vec Π | A, Λ` jointly from its Gaussian full conditional (precision
`P₀ + Σ_t Ω_t ⊗ x_t x_t'`, `Ω_t = A'Λ_t⁻¹A`), the rows of `A` by Bayesian regression, and each
equation's SV state by one sweep of stochvol's fast sampler on `e_t = A u_t` at offset 0 (what
bayesianVARs does through stochvol), or `λ_i | e ~ IG(3 + T/2, 2 + Σe²/2)`. It uses rai-bvar's own
conventions (`A u_t = e_t`, `A` unit lower triangular, regressors `const, lag-1 block, …`), so P1–P5
do not arise, and it checks itself (dense-precision identity 1e-16, `A`-row gradient 2e-11, the
single-step stochvol chain against `svsample` within `|z| < 2.2`). `test_bayesianvars.py` compares
our reduced-form CCCM sampler with it (§5).

**To unblock** (offline, as WP8): install gfortran so that bayesianVARs builds, rerun
`Rscript tools/make_r_fixtures.R --sections=bvars` (the probes run and record their answers),
then write the parity runs as `fixtures/bvars_bayesianvars_<name>.nc` in the frozen `bvars_swa`
format with `kind = "bvars_bayesianvars"`, the draws already mapped by the P1–P5 answers, and the
exact R call in the `call` attribute. `test_bayesianvars_parity` picks such files up by name and
runs them nightly once every probe is answered; until then it skips.

## 5. The reduced-form comparison (`test_bayesianvars.py`)

**Parameter set** (DESIGN §9.9): `π_j ~ N(m_j, diag v_j)` with `m = 0.5` on the own first lag and 0
elsewhere, `v = (0.4/l)²` for the own lag `l`, `(0.2/l)²` for cross lags, `10²` for the intercept;
`a_ij ~ N(0, 1)`; `SV(mu_prior=Normal(0.0, 1e4), phi_prior=Beta(20, 1.5),
sigma2_prior=ScaledChi2(1.0), h0="stationary", offset=0.0, exact=False)`; Constant
`λ_i ~ IG(3, 2)`. The fixture arrays are `(regressor, equation)`; ours (`explicit_reduced`) are
their equation-major transposes; `a_free` lists `a_ij` row-major over `i > j`, our state holds them
as `a[i, j]` (`j < i`) with the same sign.

**Compared quantities**: quantiles of `Π`, the free `a_ij`, `(μ, φ, σ)` per equation and `h_t` at
`t_sel = {1, T/4, T/2, 3T/4, T}` (not `h_0`: bayesianVARs documents a mean-0 stationary `h_0`, N04
§3.3), or `λ_i`, and of the 1–4 step forecasts (one path per stored draw, `fit.forecast`). Same
statistic as §2 over the `m` comparisons of a fixture (195 for `n3p2_const`, bound 4.05; 300 for
`n3p2_sv`, bound 4.15; 1,540 for `n7p4_sv`, bound 4.51). D-7 is applied per series: failing SV
quantities of a series whose fold fraction exceeds 0.5% are excluded (the oracle's SV step is
stochvol's).

## 6. Results

Measured on the reference laptop (M1 Pro, jax 0.12, float64; shared with other jobs, so times
are upper bounds). `max |z|` is over all compared quantiles of a fixture; fold is our fold
fraction (per series for VARs). The PR tier is the exit criterion (DESIGN §13 WP21); the nightly
tier repeats it with 4 × 40,000 (stochvol) or 4 × 20,000 (VAR) kept draws, where our MCSE is
2–3 times smaller and the test correspondingly sharper.

| fixture | m | bound | PR: max \|z\| (4 × 6,000) | nightly: max \|z\| | fold (PR / nightly) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| stochvol_i_main | 40 | 4.00 | 1.86 | 3.57 | 0 / 0 |
| stochvol_i_defaults | 40 | 4.00 | 1.66 | 2.12 | 0.01% / 0.03% |
| stochvol_ii_main | 40 | 4.00 | 1.63 | 2.53 | 0.17% / 0.12% |
| stochvol_ii_defaults | 40 | 4.00 | 1.95 | 2.75 | 0.06% / 0.04% |
| stochvol_iii_main | 40 | 4.00 | 2.34 | 2.92 | 1.03% / 0.72% (D-7-excludable, passes) |
| stochvol_iii_defaults | 40 | 4.00 | 2.59 | 2.59 | 1.22% / 1.07% (D-7-excludable, passes) |
| stochvol_iv_main | 40 | 4.00 | 1.96 | 1.76 | 0 / 0 |
| stochvol_iv_defaults | 40 | 4.00 | 1.79 | 1.64 | 0 / 0 |
| stochvol_i_exact (`exact=True`) | 40 | 4.00 | nightly only | 2.14 | — / 0.01% |
| stochvol_iv_exact (`exact=True`) | 40 | 4.00 | nightly only | 2.56 | — / 0 |
| stochvol_v_t5 (strict xfail, DGP guard passes) | 55 | 4.00 | **28.9** (oracle deviation, §3.2) | **62.5** | 0.72% / 0.65% |
| SV-t vs exact importance sampling (4 obs., `T_pad = 320`) | 45 | 4.00 | 1.83 | — | 26% |
| bvars_swa_n3p2_const | 195 | 4.05 | 2.76 (4 × 3,000) | 2.75 | — |
| bvars_swa_n3p2_sv | 300 | 4.15 | nightly only | 3.45 | 0, 1.06%, 0.55% |
| bvars_swa_n7p4_sv | 1,540 | 4.51 | nightly only | 3.56 | up to 7.1% (series 7) |

Every comparison passes without invoking the D-7 exclusion, including the series whose fold
fraction exceeds 0.5%: at these sample sizes stochvol's `|σ|` shortcut is not visible within the
fixtures' MCSE (in contrast with the four-observation problem of §3.1). The exit criterion "stochvol
main, defaults and SV-t within 4 quantile-MCSE" is met for main and defaults on all four datasets;
the SV-t row is replaced by the exact oracle because the stochvol fixture is wrong (§3.2); the
bayesianVARs row is blocked (§4) and replaced by the system-wide Gibbs oracle, which our reduced
form matches on all three fixtures.

**Runtime of the PR tier** (both files, this machine under load): about 50–55 s. The parts:

- 12 s for the SV-t row, mostly compiling the Student-t composition at `T_pad = 320`;
- 7 s for the exact-oracle test, which reuses that program;
- 8 s for the first Gaussian stochvol fixture, mostly compiling (all datasets share one program
  with `T_pad = 320`);
- 2–4 s for each further stochvol fixture;
- 5–6 s for the `n3p2_const` VAR.

The same Student-t program compiled at `T_pad = 32` takes about 28 s, against 6 s at
`T_pad = 320`, so the exact-oracle test uses the larger padding. The nightly tier takes about
6 minutes.

## 7. Phase 3: bayesianVARs 0.1.8 and bvars 1.1 (work package WP29)

Phase 3 DESIGN §9.8 runs the two GPL oracles that Phase 2 could not install. The fixtures come from
the dispatch-only job `r-fixtures` of `.github/workflows/ci.yml` (D-54): R 4.5.3 on `ubuntu-latest`
(Ubuntu 24.04, `noble`) with the Linux binaries of the Posit Package Manager snapshot
**2026-10-20** (`https://packagemanager.posit.co/cran/__linux__/noble/2026-10-20`), every
package installed as a binary: bayesianVARs 0.1.8, bvars 1.1, bsvars 4.0, stochvol 3.2.9,
factorstochvol 1.1.2, GIGrvg 0.8, RcppArmadillo 15.6.2-1, Rcpp 1.1.2, RcppProgress 0.4.2,
RcppTN 0.2-2, R6 2.6.1, generics 0.1.4, lpSolveAPI 5.5.2.0-17.15, mvtnorm 1.4-2, jsonlite 2.0.0,
coda 0.19-4.1. Every fixture records them (`packages`, `repos`, `ppm_snapshot`, `r_version`,
`ci_run`). The job's artifact was downloaded with `gh run download` and committed.

Clean room. The packages were run as black boxes. `tools/make_r_fixtures.R` records only the data
they return (numbers, dimensions, dimnames, field names): `as_data()` drops every function, R6
method, call or formula before anything is described or staged, and warnings and errors are
reported by their message only. Build logs are filtered of compiler diagnostics that would quote
package source (none was needed: every package came as a binary).

**Rules first.** The decision rule of every probe was fixed before any comparison on the rai-bvar
side, after discovery runs of the R side only (their run ids and what they changed are in the WP29
PR); the rules are implemented at commit `1c2fb9d` and carried into the design by the errata PR
(E117+). A probe that cannot be decided is recorded as `unresolved` with its evidence, and the
runs that depend on it are **not met** (D-59). P8 and P16 need rai-bvar-side code that does not
exist yet (WP30, WP38): their R data are committed and their status is `pending`.

The rules themselves are in `docs/phase3/DESIGN.md` §9.8 as amended by the errata PR #16
(Amendment 1, commit `224fb86`, E117–E147), pinned to `tools/make_r_fixtures.R` at `7f4e570`;
that PR is not merged yet. If the owner's review of it changes a rule, the affected probes are
rerun and this section is updated. Here each probe's outcome under its rule is recorded, from the
final run of the job.

The fixtures and every number below are from the final run of the job, **run 40234562983**
(commit `83c13b4`; `tools/make_r_fixtures.R` identical to `7f4e570`), downloaded with
`gh run download` and committed unchanged. The job ended with exit status 1 for one recorded
reason: the hard mixing check of `wbvars_ncsv` (§7.3); every fixture was staged and packed.

### 7.1 bayesianVARs 0.1.8: probes P1–P10

| probe | question | outcome under its rule | status |
| :--- | :--- | :--- | :--- |
| P1 | `normal_sds`, `cholesky_normal_sds`: variance or s.d.; fill order | both are **prior standard deviations** (the manual calls `cholesky_normal_sds` variances); `normal_sds` fills **column-major** over (PHI lag rows × equations). Slot numbers as s.d. 0.99, 1.99, 2.97, 4.01 (as variance 0.001–0.016); free-U posterior variance 1.0e-6; `V_prior` = (1e-6, 4e-6, 9e-6, 1.6e-5) and 1e6 for the intercept | answered |
| P2 | U vs U⁻¹, layout, sign | U is the **vector of its free elements** (names `yi~yj`: (2,1), (3,1), (3,2), …), **A = t(U)** (A u_t = e_t): (a) relative error of Σ_t against `vcov()` 4.4e-16 for this reading (and its identical lower-row-major twin), ≥ 0.74 for every other; (b) `probe_corr3`: 0.001 for A = U′ against 1.51 for 2I − U′ and (U′)⁻¹; (c) corr(u1, u2) = 0.806 by `vcov()` and by the mapping | answered |
| P3 | intercept row, lag order | PHI row r is X column r; the **intercept is the last row**; the lag rows follow rai-bvar's order (lag 1 of y1..yM, …); OLS check 0.015 (probe_corr) and 0.0035 (var_sv_n3), limit 0.05 | answered |
| P4 | `sv_para` layout | (parameter, equation, draw), equations in data order: slice 1 = μ, 2 = φ, 3 = **σ (the s.d.)**, **no h_0 slice**; 1 of 12 (orientation, permutation) pairs satisfies the rule | answered |
| P5 | `cholesky_priorhomoscedastic` | column 2 is the **inverse-gamma scale** (posterior mean λ = 2.01 against 100/49 = 2.04); the NA default substitutes `cbind(0.01, 0.01)` (the prior object's field) | answered |
| P6 | internal standardisation | **none**: the scaled fit matches the image of the raw-unit-prior fit (max \|z\| 2.81, bound 4.00) and neither the equivariant image (956) nor the unchanged fit (620); both sides use the data as committed and the SWA fixtures stand | answered |
| P7 | grouping matrices | the field `general_settings$i_mat` holds the group ids; global (1 group), equation-wise (3), covariate-wise (6), olcl-lagwise (4 at p = 2) all **equal D-38** as partitions (`probe_bayesianvars_p3.nc` `p7_groups`) | answered |
| P8 | meaning of the GL hyperparameters | data committed (`probe_bvars_p8_*.nc`, all mixed: R-hat ≤ 1.001, ESS ≥ 7,334); the decision is WP30's | pending |
| P9 | SSVS semi-automatic τ | **differs from D-44** and from every variant tried (τ₀/(c₀ ŝ) ranges 7.4–24 across coefficients); the runs use `SSVS_semiautomatic = FALSE` with explicit τ (R7) | answered |
| P10 | intercept, `prior_intercept`, floors | `prior_intercept` is an **s.d.** (max \|z\| 1.95 against N(0, s²), 5,421 against the variance reading); the intercept is **outside the shrinkage** (HS: 1.38); **every tolerance accepts 0** (no draw at a floor in any family; in every parity run `floor_fraction_phi = floor_fraction_u = 0`) | answered |

The **derived calls** of the R1 and R2 fixtures follow from these answers. They are
`normal_sds = bv_normal_sds`, the prior s.d.s in P1's column-major order and P3's rows (stored as
the variable `bv_normal_sds`), plus `priormean = 0.5`, `prior_intercept = 10`,
`cholesky_normal_sds = 1` and, for the constant run, `cholesky_priorhomoscedastic = cbind(rep(3, M), rep(2, M))`.
The draws are mapped by A = t(U), the PHI rows by P3, and `sv_para` by P4. Each fixture's `call`
attribute is the code that ran.

**Parity fixtures** (4 independent `bvar()` runs × 50,000 draws after 10,000, seeds in
`chain_seeds`; quantiles, MCSE (the larger of Vehtari et al. §4.3 and the batch MCSE), ESS, R-hat
of Π, A, h_t (t ≥ 1), μ, φ, σ or λ and the 1–4-step predictive):

| run | fixture | mixing (max R-hat / min ESS over the compared summaries) |
| :--- | :--- | :--- |
| R1 | `bvars_bayesianvars_n3p2_sv`, `bvars_bayesianvars_n7p4_sv` | 1.001 / 4,538; 1.000 / 16,175 |
| R2 | `bvars_bayesianvars_n3p2_const` | 1.000 / 155,657 |
| R3 HS | `bvars_gl_r3_hs_global_n{3,7}p2` | 1.001 / 4,564; 1.000 / 15,260 |
| R4 NG (0.1, 1, 1) | `bvars_gl_r4_ng_global_n{3,7}p2` | 1.002 / 4,484; 1.001 / 5,907 |
| R5 R2D2 (0.1, 0.5) | `bvars_gl_r5_r2d2_global_n{3,7}p2` | 1.001 / 4,586; 1.001 / 5,710 |
| R6 DL (0.5) | `bvars_gl_r6_dl_global_n{3,7}p2` | 1.001 / 4,715; 1.000 / 16,431 |
| R7 SSVS (0.01, 1, fixed p = 0.5) | `bvars_gl_r7_ssvs_global_n{3,7}p2` | 1.001 / 4,364; 1.003 / 2,767 |
| R8 HS olcl-lagwise | `bvars_gl_r8_hs_olcl_n7p4` | 1.003 / 2,569 |
| R9 NG, grid on a | `bvars_gl_r9_nggrid_global_n7p2` | 1.002 / 1,806 |

The GL fixtures also hold:
- the thinned draws of every `phi_hyperparameter` column (`hyper_draws`, 500 per run);
- summaries of the global scales (`hypersel`: HS ζ_g, NG and R2D2 ξ_g);
- the SSVS inclusion probabilities (`incl_prob`, all lags·M² indicators, MCSE sd/√ESS);
- R9's posterior of a on its grid (`grid_prob`: 0.9915 at a = 0.05, 0.0085 at 0.1).

R7 uses c1 = 1 (raw s.d.), not bayesianVARs' default 100, which mixed badly with
`semiautomatic = FALSE` in a discovery run. That choice was made before any comparison.

### 7.2 bvars 1.1: probes P11–P16

Data: `data/qd3_growth.csv` (the FRED-QD small set as 400 Δlog GDPC1, 400 Δlog GDPCTPI and
FEDFUNDS, 1960Q2–2019Q4), p = 4. On the levels data bvars' ncSV sampler stopped with an
`iwishrnd` error (E146).

| probe | question | outcome under its rule | status |
| :--- | :--- | :--- | :--- |
| P11 | defaults | recorded as data (`p11_*` variables): A0 = 1 on the own first lags, S0 = I₃, ν₀ = 6, Ψ, Γ (K × K), λ = 4, sv_a = 1, sv_s = 0.1; starting values ρ = 0.5, ω = 0.1, σ_v² = 0.01, σ_ω² = 1, s_ = 0.05, h a T-vector, V the Minnesota V (K × K) | answered |
| P12 | can V be held fixed? | **holds, trivially**: the V draws equal the starting value exactly (sd 0) under every concentrated matrix-GIG setting and under the default prior; a starting V of 2Ω stays at 2Ω, so bvars 1.1 never moves V. The runs keep the default starting V and record it (`V_mean`, `V_sd` = 0) | answered |
| P13 | E h = 0, initial condition | **undecided**: h has T entries (no h_0 state); the scale experiment passes for ncSV (max \|z\| 2.25) but **fails for cSV** (8.93, bound 4.08); cSV's ρ conditional fits ρ ~ U(−1, 1) with **h_0 = 0** (PIT p = 0.058; stationary p = 3e-62; flat rejected: 0 draws outside (−1, 1) against 25,552 expected); for ncSV no candidate fits (below) | unresolved |
| P14 | ρ, ω / σ_ω² priors; fixing σ_ω² | **undecided**: cSV ρ ~ U(−1, 1); **ncSV ρ is not confined to (−1, 1)** (0.93% of draws with \|ρ\| > 1, 0.15% > 10, max 1.7e5), yet a flat prior predicts far more draws outside (25,495 expected, 371 observed), so no candidate fits; ncSV σ_ω² \| ω, s_ fits σ_ω² ~ G(sv_a, scale s_) with s_ a sampled state (p = 0.057); cSV σ_v²'s prior matches none of the candidates; no setting fixes the variance that drives h: for ncSV σ_ω² keeps CV 0.021 and 0.195, and three settings stopped with a GIG parameter error; for cSV σ_v² keeps CV > 1 in every setting, although cSV's σ_ω² reaches CV 2e-4 at sv_a × 1e8 | unresolved |
| P15 | Σ, prior mean, intercept | Σ is the **error covariance** (posterior mean within 1.0% of the residual covariance); stationary = FALSE gives A0 = 1 on every own first lag, TRUE gives 0; the **intercept is the last row of X** (column K of A; posterior mean (3.72, −3.05) on probe_corr) | answered |
| P16 | is bvars' h sampler exact? | data committed (`probe_wbvars_p16.nc`): 4 runs × 10,000 chained `estimate(post, S = 2)` calls per volatility (bvars refuses S = 1), h_1, h_2, ρ, ω, σ_v², σ_ω², s_ and S after each; the extraction check passes (max \|z\| 2.74 cSV, 3.34 ncSV, bound 4.08); the decision is WP38's, which must first establish that the T = 2 posterior is proper under ncSV's untruncated ρ | pending |

**Gating comparison** (§9.8, per volatility): both cSV and ncSV take **branch 2**. P12 holds and P15
is decided, but P13 and P14 do not map bvars' SV prior onto a CommonSV option. The initial
condition is h_0 = 0 (cSV) or undecided (ncSV), and the ρ and σ² priors do not map. Branch 2
compares the 1- and 4-step predictive quantiles under the Phase 2 rule; its role is pending the
owner's answer to Q35 (review M6's recommended reading: evidence, not an equality gate).

### 7.3 bvars parity fixtures

`wbvars_csv`, `wbvars_ncsv`: 4 runs, each a 10,000-draw `estimate()` and then a 50,000-draw
`estimate()` restarted from its `last_draw`, followed by `forecast(post, horizon = 4)`. They hold:
- quantiles, MCSE, ESS and R-hat of B (= t(A), rai-bvar regressor order), Σ (unique entries) and
  the 1–4-step predictive;
- the prior arrays: B0, S0, ν₀, V (`V_mean`, `V_sd`);
- from 4 separate state runs (1,000 chained calls of S = 50 each), h_t at t_sel and the SV
  states, with a check that their B and Σ agree with the main runs (max |z| 2.55 and 2.65, pass).

Mixing: `wbvars_csv` 1.005 / 484 (Σ). **`wbvars_ncsv` does not meet the mixing requirement**:
Σ has max split R-hat 1.013 and min ESS 190. The draw counts are the design's, and the fixture
records `mixing_ok = 0`. A comparison that needs Σ of ncSV is not met at these draw counts
(D-59). B (7,405) and the predictive (196,375) mix.

### 7.4 stochvol 3.2.9: the SV-t row

`stochvol_v_t5.nc` is regenerated with stochvol 3.2.9 (same seed, dataset (v); P2 §2.10 Stage 4
amendment 1). It also records stochvol's posterior on the four-observation SV-t problem of
§3.2, 2 × 200,000 draws:

| | μ | h_2 | τ_2 |
| :--- | :--- | :--- | :--- |
| exact (importance sampling, §3.2) | −1.495, −1.042, −0.732, −0.425, 0.017 | −1.291, −0.759, −0.353, 0.124, 1.022 | 0.681, 1.361, 2.263, 3.888, 9.171 |
| stochvol 3.2.9 | −1.545, −1.068, −0.743, −0.422, 0.028 | −1.618, −0.822, −0.335, 0.172, 1.036 | 0.692, 1.244, 1.962, 3.250, 7.577 |

stochvol 3.2.9 still misses the standardized-t posterior, as 3.2.1 did (§3.2): its quantiles
are those of 3.2.1 within Monte Carlo error. Step 3 of §3.2's unblock path therefore applies. The
strict `xfail` of `test_stochvol_parity[stochvol_v_t5]` stays, and the exact importance-sampling
test remains the SV-t criterion; the fixture now records 3.2.9.
