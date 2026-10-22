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
the installed version). bayesianVARs 0.1.8 is **not installed** (§4).

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

## 4. bayesianVARs: blocked

**Status: bayesianVARs parity is blocked.** bayesianVARs 0.1.8 (first CRAN release 2023) has no
CRAN macOS binary for R 4.1, and a source build links its RcppArmadillo code against R's `FLIBS`,
i.e. against gfortran, which is not installed (`R CMD config FC` names `gfortran`; the `FLIBS`
directories `/usr/local/gfortran/lib…` do not exist). `fixtures/probe_bayesianvars.nc` records this
evidence (`bayesianvars_available = 0`). Without the package the probes cannot run:

| probe | question | documented | status |
| :--- | :--- | :--- | :--- |
| P1 | are `normal_sds` and `cholesky_normal_sds` variances (documented) or standard deviations (named)? | variances | unresolved |
| P2 | is `U` stored as `U` or `U⁻¹`, and with which sign (`A = Uᵀ` or `2I − Uᵀ`)? | `Σ_t = U'⁻¹ D_t U⁻¹`, hence `A = Uᵀ`; GK App. B prints the U step without the minus sign | unresolved |
| P3 | which row of the returned `PHI` draws holds the intercept? | undocumented | unresolved |
| P4 | what is the layout of the returned `sv_para` draws? | undocumented | unresolved |
| P5 | is `cholesky_priorhomoscedastic = cbind(shape, scale)` the IG scale `β₀` of `λ^{−α₀−1}e^{−β₀/λ}`, and what replaces `NA`? | (shape, scale), default `NA` with an undocumented substitute | unresolved |

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
