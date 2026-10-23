# 01: Normal-mixture approximations of log χ²₁ (KSC 1998, 7 components; OCSN 2007, 10 components)

Research note for rai-bvar Phase 2 (SPEC §4.3, §4.4 volatility blocks, §8 "Mixture approximation", §10.3, §13 Phase 2).
Scope: the auxiliary-mixture step of every `SV` variant (`SV`, `SV(outliers=True)`, `SV(tails="t")`): the constants, the
`log(e² + c)` transformation, the centring convention, the indicator full conditional, exactness, and how to test it.
The log-volatility precision sampler, SV-parameter priors and ASIS are covered in other notes. They appear here only
where they touch the mixture.

Clean-room status: every formula below comes from the papers cited. GPL packages (stochvol, bayesianVARs, factorstochvol,
bvartools, bvarsv, bsvars) were opened only to confirm **published constants and observable behaviour** (which table is
used, which offset, whether a shift is applied). No code was transcribed. The pseudo-code in §8 is original.

---

## 0. TL;DR: decisions to take into code

1. **Default mixture: OCSN 10-component** (Omori, Chib, Shephard & Nakajima 2007, Table 1, right-hand block). Keep the KSC
   7-component table only as a parity option (`mixture="ksc7"`) for Primiceri/bvarsv/CCMM/Chan-style replications.
2. **Centring.** The OCSN 10-component means are the component means of log χ²₁ itself. **Do not shift them.** The KSC
   7-component means *as printed in KSC Table 4* are centred at 0, so the component means are `m_i − 1.2704`.
   OCSN's Table 1 *left-hand block* reprints the KSC table **already shifted and re-sorted**. Subtracting 1.2704 again is a
   double-shift bug. Guard: unit test that the mixture mean equals −1.2704 ± 1e-4 (§12.1).
3. **Offset.** Use `y*_t = log(ẽ_t² + c)`, where ẽ_t is the residual after dividing by any mixing scale (√τ_t for t errors,
   o_t for outliers), with a **scale-relative** `c = c_rel·ŝ²`, c_rel = 1e-12 by default (bias ≈ √(2π·c_rel) ≈ 2.5e-6 in
   log-variance units; §7). (Called κ in the first draft; renamed because κ_j denotes the log indicator weights, §2.)
   Absolute parity modes: `c = 0` (stochvol/bayesianVARs), `c = 1e-4` (OCSN, bvartools, Chan),
   `c = 1e-3` (KSC, Primiceri, bvarsv, CCMM). Record `(mixture, offset rule, c)` in the fit metadata.
4. **Indicator draw.** The indicators are independent over (t, equation, chain):
   `Pr(s_t = j | ·) ∝ p_j σ_j⁻¹ exp{−(y*_t − h_t − μ_j)²/(2σ_j²)}`. Compute it in log space and draw with
   `jax.random.categorical` (Gumbel-max) or inverse CDF with one uniform. The indicators are drawn **after the mean
   block and immediately before h** (Del Negro & Primiceri 2015). Nothing may run between the s draw and the h draw.
5. **Exactness.** The mixture sampler targets an *approximate* posterior. Ship an `exact=True` option that turns the h draw
   into a Metropolis–Hastings step with acceptance ratio `exp{w(h') − w(h)}`, where w is a *log* weight (the first draft
   wrote `w(h')/w(h)`, which is inconsistent with the definition; corrected by the cross-check), and
   `w(h) = Σ_t [log N(ẽ_t; 0, e^{h_t}) − log g(y*_t − h_t)]` (Stroud, Müller & Polson 2003; DNP 2015, Algorithm 3 and
   Appendix B.2 eq. 7). Strict Geweke/SBC tests run with `exact=True`. The default `exact=False` matches the stochvol and
   bayesianVARs defaults. The exact mode has the right invariant distribution for any c ≥ 0 and either table, but its
   *mixing* depends on how well the auxiliary model covers the exact conditional: with c/λ not small it can converge
   extremely slowly while still showing a high acceptance rate (cross-check toy, §10.2). Use OCSN10 with the tiny relative
   offset in exact mode too.

---

## 1. Sources consulted

| # | Source | What it confirms | Where |
|---|---|---|---|
| S1 | Kim, Shephard & Chib (1998), *RES* 65(3) 361–393. Read as the final working paper, Nuffield, 14 July 1997: https://www.nuffield.ox.ac.uk/economics/papers/1996/w26/Chib8.pdf | 7-component table; means written as `m_i − 1.2704`; offset `c = 0.001`; indicator conditional; reweighting | §2.1 eq. (2) (E log ε² = −1.2704, Var = 4.93); §3.1 eqs. (9)–(10), "we will set c = 0.001", moment-matching method; **Table 4**; §3.2 (indicator pmf); §3.4 eq. (12) (reweighting) |
| S2 | Omori, Chib, Shephard & Nakajima (2007), *J. Econometrics* 140(2) 425–449, doi:10.1016/j.jeconom.2006.07.008 (bibliographic data confirmed on Crossref). Read as (a) Nuffield WP 2004-W19, 22 Aug 2004: https://www.nuff.ox.ac.uk/economics/papers/2004/W19/mixture28.pdf (byte-identical to the ORA deposit https://ora.ox.ac.uk/objects/uuid:e9199508-29f0-4865-aa3e-18b06e25934f), (b) the revision of Sept 2005: https://www.economics.uci.edu/files/docs/colloqpapers/w06/chib.pdf, and (c) **added by cross-check:** CIRJE-F-297, "August 2004: Revised in April 2006", https://www.cirje.e.u-tokyo.ac.jp/research/dp/2004/2004cf297.pdf, the last pre-publication revision | 10-component table **Table 1**; KSC block reprinted shifted; offset `c = 0.0001`; indicator conditional (§2.3.2), sampled "by the inverse distribution method"; reweighting (§2.4); sd of log-weights, K = 7 vs K = 10 (§3.1, Fig. 6) | 2005 rev.: §2.1 eq. (5), Table 1, §2.2 (a_i, b_i), §2.3.2, §2.4, §3.1. The April 2006 revision has the same Table 1 (all 50 entries), the same §2.3.2 pmf (with the v_j⁻¹ factor), the same §3.1 c = 0.0001 sentence and the same Fig. 6 values (0.917, 0.0535). It uses index j and calls the log χ²₁ variable ε*_t. The 2004 WP numbers the mixture eq. (4), not (5). |
| S3 | Hiraki, Chib & Omori (2024), arXiv:2404.13986v2, https://arxiv.org/pdf/2404.13986 | Reprints the OCSN 10-component table (p, m, v², a, b) | §2, eq. (8), Table 1 |
| S4 | Primiceri (2005), *RES* 72(3) 821–852. Read as the final version, July 2004: https://faculty.wcas.northwestern.edu/~gep575/tvsvar_final_july_04.pdf | KSC 7 with `m_j − 1.2704`; offset `c = 0.001` on the structural residual; `y** = 2h + e` with `h = log σ`; indicator pmf; no reweighting (fn. 26) | App. A.3 eqs. (17)–(19); App. A.7 (table) |
| S5 | Del Negro & Primiceri (2015), *RES* 82(4) 1342–1345. Read as NY Fed Staff Report 619 (rev. Oct 2014, with online appendix): https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr619.pdf | Step order (Algorithm 2: indicators drawn after θ and before Σ); exact MH correction (Algorithm 3) and its acceptance ratio; Geweke tests showing Alg. 1 wrong, Alg. 2 approximately right, Alg. 3 exact; `y** = log(y*² + 0.001)` | §§2–5; App. B.1, B.2 eq. (7) |
| S6 | Stroud, Müller & Polson (2003), *JASA* 98(462) 377–386 | MH correction of mixture approximations, eq. (11) (cited by S5) | **UNVERIFIED (not opened)**; the formula is re-derived in S5 App. B.2 and in §10 below |
| S7 | stochvol 3.2.9 (CRAN mirror github.com/cran/stochvol; GPL ≥ 2). Constants and behaviour only. Vignette (JSS 2016): https://cran.r-project.org/web/packages/stochvol/vignettes/article.pdf | Uses OCSN 10 unshifted; precomputes `log p_j − ½ log v_j²`. Offset 0 unless **no design matrix is given and** some `y² == 0`; then `sd(y)/10000` is added to y². With t errors the offset goes on `y²/τ`. **Cross-check additions:** `svsample` floors `log(y²/τ + offset)` at −100 ("−100 ≈ log(4e-44)") before the mixture step. Optional misspecification correction (default off). The fast mixture sampler is used only when ρ = 0, μ is normal or fixed at 0, (φ+1)/2 is beta, σ² ~ Gamma(½, ·), and, if the correction is on, there are no t errors and no regression. With the correction on, fast SV computes `w = Σ log N(y_t; 0, e^{h_t}) − Σ log g(y*_t − h_t)` per stored draw and **resamples with replacement** in R. Otherwise the general sampler applies an MH step. Fast-SV sweep order: s → h → (μ, φ, σ) → τ → β, with y* recomputed after β and τ, which is the DNP Algorithm 2 order. | `src/utils_latent_states.h`, `src/utils_main.h` (`clamp_log_data2`), `src/utils_main.cc` (`compute_correction_weight`), `src/sampling_main.cc`, `src/single_update.cc`, `R/wrappers.R` (svsample data prep and sampler choice); vignette §3.1 |
| S8 | bayesianVARs 0.1.8 (GPL ≥ 3) | SV through stochvol's C++ fast-SV update (OCSN 10); `log(str_resid² + offset)` with `expert_sv_offset = 0` ("Do not use!"). **Cross-check:** it calls `stochvol::update_fast_sv`, whose `ExpertSpec_FastSV` (stochvol `inst/include/expert.hpp`) has no correction option, so no mixture correction is possible on this path. There is also no −100 floor on this path. Order: Φ → U → structural residual → (s → h → θ), which is DNP-consistent. | `R/utility_functions.R` (`specify_prior_sigma`), `src/bvar_cpp.cpp` |
| S9 | bvartools 0.3.0 (GPL ≥ 2) | Both tables (`stochvol_ksc1998` subtracts 1.2704; `stochvol_ocsn2007` does not); `y* = ln(y² + constant)`, default constant 1e-4. The docs state `y_t = e^{h_t/2} ε_t` with a random-walk h. **Cross-check:** the indicator log-pmf uses `log_normpdf(y*, h + μ_j, sqrt(σ²_j))`, so h is the log-variance. | `src/stochvol_ksc1998.cpp`, `src/stochvol_ocsn2007.cpp`, `R/add_priors.bvarmodel.R` |
| S10 | bvarsv 1.1 (GPL ≥ 2) | KSC 7 with ±1.2704; offset 0.001. **Cross-check:** the internal state `Sigtdraw` is the **log-variance** (sd = `exp(0.5·Sigtdraw)`), not Primiceri's log-sd. | `R/bvarsv.R`, `src/bvarsv.cpp` |
| S11 | factorstochvol 1.1.2 (GPL ≥ 2) | stochvol update; offset `sd(y)/10000` only if zeros and `factors == 0` | `R/wrappers.R`, `src/update_fsv.cpp` |
| S12 | bsvars 4.0 (GPL ≥ 3; CRAN, published 2026-08-22) | OCSN 10 constants; transform `log((u + 1e-9)²)`, an offset on u rather than u². **Cross-check:** in CRAN 4.0 the indicator kernel is `exp(p_j − (r − m_j)²/v_j²)`, with no log, no v_j⁻¹ and no ½. That is **not** the OCSN pmf. The development version 4.0.0.9000 fixes it (commit 5d6ac212ae, 2026-08-07, "[B04] … with the probability and variance terms included"; NEWS: PR #160). bsvars 4.0 is therefore no mixture oracle. | `src/sv.cpp` (CRAN tarball and GitHub master) |
| S15 | **Added by cross-check:** Kastner & Frühwirth-Schnatter (2014), *CSDA* 76, 408–423, doi:10.1016/j.csda.2013.01.002. Read as arXiv:1706.05280v1 ("with minor editorial changes, this article is published as…") | Auxiliary model `ỹ_t = m_{r_t} + h_t + ε_t`, ε ~ N(0, s²_{r_t}) (eq. 7). Non-centred regression for (μ, σ) given h̃ and r (eq. 10). Offsets mentioned: 10⁻³ (KSC) or 10⁻⁴ (OCSN). Indicator pmf computed in log scale with max-normalisation, then inverse transform (§2.6). | §2.2 eqs. (6)–(7), §2.5 eq. (10), §2.6 |
| S13 | Chan, statespace-toolkit (MIT): https://github.com/joshuaccchan/statespace-toolkit | KSC 7 with the −1.2704 shift ("means already adjusted"); `c = 1e-4`; reweighting example with the exact density of `log(y²+c)` | `core/+ssm/ksc_rw_h0.m`, `examples/ex07_aux_mixture.m`; cites Chan, *Bayesian Macroeconometrics*, §10.1.1–10.1.2, Table 10.1 |
| S14 | Mertens, CCMM (2024) replication code https://github.com/elmarmertens/CCMMoutlierVAR-code (no licence: do not copy) and em-matlabbox `emgibbsbox/getKSC{7,10}values.m` | CCMM uses KSC 7 (−1.2704), `c = 0.001`; the 10-component helper uses `c = 0.0001` with the comment "see OCSN page 436" | constants only |

---

## 2. Notation

| Symbol | Meaning | Code name (suggested) |
|---|---|---|
| i = 1..n, t = 1..T | equation, time | `i`, `t` |
| e_{i,t} | structural residual `y_{i,t} + Σ_{j<i} a_ij y_{j,t} − g_i(x_t)` (SPEC §4.1); equals `√λ_{i,t} ε_{i,t}` | `resid` |
| λ_{i,t} = exp(h_{i,t}) | conditional variance; **h is the log-variance** (as in stochvol) | `h` |
| τ_{i,t} | Student-t scale-mixture variable, `e = √(λτ) ε` (`SV(tails="t")`) | `tau` |
| o_{i,t} ≥ 1 | outlier scale on the **standard deviation**, `e = o √λ ε` (CCMM, `SV(outliers=True)`) | `o` |
| ẽ_{i,t} | normalised residual `e/(√τ · o)`; ẽ = √λ ε | `resid_n` |
| c | offset in `log(ẽ² + c)` | `offset` |
| c_rel, ŝ² | relative offset multiplier and per-equation scale, `c = c_rel·ŝ²` (§7.2) | `offset_rel`, `scale2` |
| y*_{i,t} | `log(ẽ²_{i,t} + c)`, the "auxiliary observation" | `ystar` |
| ξ_t | `log ε_t²`, distributed log χ²₁ | |
| r_t | `y*_t − h_t` (the argument of the mixture density) | `r` |
| K | number of components (7 or 10) | `K` |
| p_j, μ_j, σ_j² | component weight, **effective** mean and variance of component j | `MIX_P`, `MIX_MU`, `MIX_VAR` |
| m_j | mean as printed in the source table (μ_j = m_j for OCSN; μ_j = m_j − 1.2704 for KSC Table 4) | |
| κ_j | `log p_j − ½ log σ_j²` (log of the unnormalised indicator weight, without the 2π constant) | `MIX_LOGW` |
| s_t ∈ {1..K} | mixture indicator | `s` |
| g(·) | mixture density `Σ_j p_j N(·; μ_j, σ_j²)` | `log_mix_pdf` |
| f(·) | exact log χ²₁ density | `log_logchi2_pdf` |

Primiceri (S4) writes `h^P = log σ` (log standard deviation), so `y** = 2h^P + e`. Here `h = 2h^P`, and the state-innovation
variance in our units is 4 × Primiceri's `W`. The mixture itself is unchanged. (bvarsv, S10, already stores the
log-variance internally. See §12.5.)

Symbol clash (cross-check): μ, φ, σ **without** a component subscript are the SV level, persistence and vol-of-vol of
equation i (other note). μ_j, σ_j² and μ_{s_t}, σ²_{s_t} **with** a component subscript are mixture constants. Keep the
subscript in code names too (`MIX_MU`, `MIX_VAR` against `sv_mu`, `sv_sigma`).

---

## 3. Where the mixture enters the model

Conditional on the mean block, equation i's residuals satisfy `ẽ_{i,t} = exp(h_{i,t}/2) ε_{i,t}`, ε ~ N(0,1). Squaring and
taking logs (S1 §2.1 eq. 2; S2 eqs. 3–4):

$$ y^*_{i,t} \equiv \log(\tilde e_{i,t}^2 + c) \;\approx\; h_{i,t} + \xi_{i,t}, \qquad \xi_{i,t} = \log\varepsilon_{i,t}^2 \sim \log\chi^2_1 \quad(\text{exact when } c=0). $$

The sign of ε is ancillary for h when there is no leverage (S2 §2.1). Leverage is out of scope for Phase 2. ξ is replaced
by the mixture, with indicators (S1 eq. 10; S2 eq. 5):

$$ \xi_t \mid s_t = j \sim N(\mu_j, \sigma_j^2), \qquad \Pr(s_t=j)=p_j . $$

Given s, `y*_t − μ_{s_t} = h_t + σ_{s_t} z_t` is linear Gaussian in h. h is then drawn by the banded precision sampler
(other note) with observation vector `ỹ_t = y*_t − μ_{s_t}` and observation precision `1/σ²_{s_t}`.

Variants (t errors, outliers): the mixture always applies to the **normalised** residual ẽ, i.e. after dividing by √τ_t and
o_t. stochvol applies the offset after dividing by τ (S7: `log(y²/τ + offset)`), and so do we. When c = 0 this is the same
as using `log e² − log τ − 2 log o`. When c > 0 the two differ slightly, so the choice is a documented design decision.
(Cross-check: the CCMM code, S14, does it the other way round. It computes `log((A·resid)² + 0.001)` from the raw structural
residual **before** its joint SV/outlier routine. A c = 1e-3 CCMM parity run therefore differs from ours by an
offset-placement term. The term vanishes as c → 0.) Compute y* in log space,
`y* = logaddexp(2·log|e| − log τ − 2·log o, log c)`, so that ẽ² never underflows. The companion note 03 uses the same
form.

---

## 4. The target: log χ²₁

Density, for ξ ∈ ℝ (S2 2005 rev. §2.1):

$$ f(\xi) = \frac{1}{\sqrt{2\pi}} \exp\Big\{\frac{\xi - e^{\xi}}{2}\Big\}, \qquad \log f(\xi) = -\tfrac12\log 2\pi + \tfrac12 \xi - \tfrac12 e^{\xi}. $$

Exact moments (polygamma identities; computed in float64 with scipy and checked):

| Quantity | Exact value | KSC/OCSN quote |
|---|---|---|
| E ξ = ψ(½) + log 2 = −γ − log 2 | **−1.2703628454614782** | −1.2704 (S1 §2.1) |
| Var ξ = ψ′(½) = π²/2 | **4.934802200544679** | 4.93 (S1 §2.1) |
| skewness = ψ″(½)/ψ′(½)^{3/2} | −1.535142 | |
| kurtosis = ψ‴(½)/ψ′(½)² + 3 | **7** (excess kurtosis exactly 4) | |
| q₀.₀₁, q₀.₉₉ of ξ | −8.759, 1.892 | range used in S2 Fig. 1 |

The shift constant is **1.2704 exactly, to 4 decimals** in every source that uses the KSC table (S1, S4, S9, S10, S13,
S14; the OCSN10 users S7, S8, S11 and S12 apply no shift). Use the literal `1.2704`, not the digamma value, when building
the KSC table. Σ q_i m_i = 8.5e-7 for the printed means, which suggests KSC fitted centred means and then shifted them by
that literal. That is an inference; KSC do not say so. The literal differs from the exact value by 3.7e-5, and the parity
oracles use the literal.

---

## 5. Constant tables (full published precision: 5 decimals in every source)

### 5.1 KSC 7-component, as printed in KSC Table 4 (S1; reprinted in S4 App. A.7)

Column headers in S1: ω, Pr(ω = i), μ_i, σ_i². In S4: ω, q_j = Pr(ω = j), m_j, v_j². The third column is a **variance**.

| i | p_i (q_i) | m_i (as printed; **centred at 0**) | σ_i² (variance) | effective mean μ_i = m_i − 1.2704 |
|---|---|---|---|---|
| 1 | 0.00730 | −10.12999 | 5.79596 | −11.40039 |
| 2 | 0.10556 | −3.97281 | 2.61369 | −5.24321 |
| 3 | 0.00002 | −8.56686 | 5.17950 | −9.83726 |
| 4 | 0.04395 | 2.77786 | 0.16735 | 1.50746 |
| 5 | 0.34001 | 0.61942 | 0.64009 | −0.65098 |
| 6 | 0.24566 | 1.79518 | 0.34023 | 0.52478 |
| 7 | 0.25750 | −1.08819 | 1.26261 | −2.35859 |

Σ p_i = 1.00000. Σ p_i m_i = +0.000001, so the printed means are centred at zero. S1 §3.1: "f(z_t) = Σ q_i f_N(z_t | m_i −
1.2704, v_i²)"; §3.2: "Pr(s_t = i | y*_t, h_t) ∝ q_i f_N(y*_t | h_t + m_i − 1.2704, v_i²)". KSC fitted these by matching the first
four moments of the implied log-normal (to χ²₁) and of f (to log χ²₁) by nonlinear least squares (S1 §3.1).

### 5.2 KSC 7-component as reprinted in OCSN Table 1, left-hand block (S2; **already shifted, re-sorted by mean**)

| i | p_i | m_i (= μ_i, shift already applied) | v_i² |
|---|---|---|---|
| 1 | 0.04395 | 1.50746 | 0.16735 |
| 2 | 0.24566 | 0.52478 | 0.34023 |
| 3 | 0.34001 | −0.65098 | 0.64009 |
| 4 | 0.25750 | −2.35859 | 1.26261 |
| 5 | 0.10556 | −5.24321 | 2.61369 |
| 6 | 0.00002 | −9.83726 | 5.17950 |
| 7 | 0.00730 | −11.40039 | 5.79596 |

Every row equals a §5.1 row after subtracting 1.2704 (checked for all 7 rows). **Trap:** code copied from OCSN's left block
must not subtract 1.2704 again.

### 5.3 OCSN 10-component (S2 Table 1, right-hand block; **no shift**)

| j | p_j | m_j = μ_j | v_j² = σ_j² (variance) | a_j (leverage only) | b_j (leverage only) |
|---|---|---|---|---|---|
| 1 | 0.00609 | 1.92677 | 0.11265 | 1.01418 | 0.50710 |
| 2 | 0.04775 | 1.34744 | 0.17788 | 1.02248 | 0.51124 |
| 3 | 0.13057 | 0.73504 | 0.26768 | 1.03403 | 0.51701 |
| 4 | 0.20674 | 0.02266 | 0.40611 | 1.05207 | 0.52604 |
| 5 | 0.22715 | −0.85173 | 0.62699 | 1.08153 | 0.54076 |
| 6 | 0.18842 | −1.97278 | 0.98583 | 1.13114 | 0.56557 |
| 7 | 0.12047 | −3.46788 | 1.57469 | 1.21754 | 0.60877 |
| 8 | 0.05591 | −5.55246 | 2.54498 | 1.37454 | 0.68728 |
| 9 | 0.01575 | −8.68384 | 4.16591 | 1.68327 | 0.84163 |
| 10 | 0.00115 | −14.65000 | 7.33342 | 2.50097 | 1.25049 |

Σ p_j = 1.00000; Σ p_j m_j = −1.270280, so the means are already the log χ²₁ component means. a_j and b_j belong to the
leverage extension (S2 §2.2: a_i = exp(v_i²/8), b_i = ½ exp(v_i²/8)). They are not needed in Phase 2. Recomputing b from the
rounded v² differs from the printed b in the 5th decimal for j = 1, 8, 9, 10, so use the printed values if leverage is ever
added. The fitting criterion for the 10-component table is not stated in the 2004, 2005 or April 2006 WP text: **UNVERIFIED**.
(The a, b formulas were re-derived in the cross-check. ξ − m = v z gives a = E e^{vz/2} = e^{v²/8}, and by Stein's lemma
b = E[z e^{vz/2}]/v = ½ e^{v²/8}. All 10 printed a_j match exp(v_j²/8) to 5 decimals.)

### 5.4 Derived constants (computed here; useful as transcription checks)

`κ_j = log p_j − ½ log v_j²` (unnormalised log weight of the indicator pmf; the common −½ log 2π cancels):

- OCSN10: `[−4.00937239, −2.17845316, −1.37686428, −1.12572770, −1.24873234, −1.66194609, −2.34338373, −3.35107342, −4.86438228, −7.76421433]`.
  stochvol 3.2.9 stores the same numbers (first five seen to ≥ 15 digits: −4.0093723912…, −2.1784531553…, −1.3768642766…,
  −1.1257277037…, −1.2487323430…). This is an independent check of the p and v² transcription.
- KSC7 (Table-4 order): `[−5.79846149, −2.72885728, −11.64213255, −2.23086873, −0.85570701, −0.86474010, −1.47332606]`.

Mixture moments versus exact (computed; script in §12.1):

| | mean | variance | skew | kurt | E[e^ξ], E[e^{2ξ}], E[e^{3ξ}], E[e^{4ξ}] (χ²₁: 1, 3, 15, 105) |
|---|---|---|---|---|---|
| exact log χ²₁ | −1.270363 | 4.934802 | −1.5351 | 7.0000 | 1, 3, 15, 105 |
| KSC7 (μ = m − 1.2704) | −1.270399 | 4.934854 | −1.5352 | 7.0006 | 1.0000, 2.9999, 14.9993, 104.993 |
| OCSN10 | −1.270280 | 4.933731 | −1.5343 | 6.9931 | 1.0000, 3.0000, 15.0005, 105.006 |
| KSC7 **without** shift (bug) | +0.000001 | 4.934854 | | | 3.56, 38.1, … |
| OCSN10 **with** extra −1.2704 (bug) | −2.540680 | 4.933731 | | | 0.281, 0.236, … |

Density accuracy (computed on a fine grid, float64):

| | max abs(f − g) on [q₀.₀₁, q₀.₉₉] | TV(f, g) | KL(f‖g) per obs | sd_f[log f − log g] per obs | log f − log g at ξ = −20 / −15 / −10 |
|---|---|---|---|---|---|
| KSC7 | 1.03e-2 | 1.79e-2 | 2.69e-3 | 0.0820 | +2.18 / −0.58 / +0.68 |
| OCSN10 | 3.8e-4 | 9.0e-4 | 7.9e-5 | 0.0155 | −0.28 / +0.13 / −0.05 |

This agrees with S2 Fig. 1 (KSC errors up to about ±0.01; K = 10 much closer). S2 §3.1 / Fig. 6 report that the sd of the
posterior log importance weights in a simulated T = 1000, ρ = 0 example is 0.917 for K = 7 and 0.0535 for K = 10. The text
rounds these to 0.92 and 0.05. That experiment used c = 0.0001 ("Throughout we use…"), φ = 0.97, exp(μ/2) = 0.65 and
σ = 0.15, so c/λ ≈ 2.4e-4 and the offset bias (§7.2, ≈ 0.039) is part of those weights. All numbers in this subsection were
recomputed independently in the cross-check (`xc01_checks/xc_mixture.py`) and agree to the digits shown.

### 5.5 Cross-source verification of the constants

| Source | Independent of OCSN authors? | OCSN10 p / m / v² | KSC7 p / m / v² |
|---|---|---|---|
| S2(a) OCSN WP Aug 2004, Table 1 | (reference) | ✔ | ✔ (shifted, re-sorted) |
| S2(b) OCSN WP rev. Sept 2005, Table 1 | no | ✔ identical | ✔ identical |
| S2(c) OCSN CIRJE-F-297 rev. April 2006, Table 1 (cross-check) | no | ✔ identical (incl. a, b) | ✔ identical (shifted, re-sorted) |
| S3 Hiraki–Chib–Omori 2024, Table 1 | no (Omori, Chib) | ✔ identical (incl. a, b) | n/a |
| S7 stochvol 3.2.9 | **yes** (Kastner / Hosszejni) | ✔ identical (incl. a, b) | n/a |
| S9 bvartools 0.3.0 | **yes** (Mohr) | ✔ identical | ✔ identical (raw + shift) |
| S12 bsvars 4.0 | **yes** (Woźniak) | ✔ identical constants, but the CRAN 4.0 *kernel* is wrong (S12 row in §1) | n/a |
| S14 Mertens em-matlabbox | **yes** | ✔ identical | ✔ identical (raw + shift) |
| S1 KSC WP July 1997, Table 4 | n/a | n/a | ✔ (reference) |
| S4 Primiceri July 2004, App. A.7 | yes | n/a | ✔ identical |
| S10 bvarsv 1.1 / S13 Chan toolkit (MIT) | yes | n/a | ✔ identical |
| Published JoE 140(2) Table 1 / published RES 65(3) Table 4 | n/a | **UNVERIFIED (paywalled, not opened)**. The last pre-publication revision (CIRJE-F-297, April 2006, found by the cross-check) and a 2024 reprint by the same authors agree, and Mertens cites "OCSN page 436" for c = 1e-4. | **UNVERIFIED (not opened)**. The July 1997 WP is the final revision. |

Conclusion: the OCSN 10-component table is confirmed by the authors' own working papers (2004, 2005) and their 2024 reprint,
and by **four independent transcriptions** (stochvol, bvartools, bsvars, Mertens). All agree to the last printed digit.

---

## 6. Centring convention: the −1.2704 question

| Table | What the printed means are | Effective component mean |
|---|---|---|
| KSC Table 4 (S1), Primiceri A.7 (S4), Chan Table 10.1 (S13), bvarsv, bvartools `ksc1998`, Mertens `getKSC7values` | centred at 0 (Σ p m ≈ 0) | **m_i − 1.2704** |
| OCSN Table 1 left block (S2) | KSC means, already shifted | m_i as printed |
| OCSN Table 1 right block (10 components), stochvol, bayesianVARs, bvartools `ocsn2007`, bsvars, Mertens `getKSC10values` | log χ²₁ component means (Σ p m ≈ −1.2703) | **m_j as printed** |

Software applies the KSC shift in different places, all of them equivalent. Some subtract it from the table means (Chan:
`m − 1.2704`, with the comment "means already adjusted"; bvartools; Mertens). Others add +1.2704 to the residual inside the
density (bvarsv). Recommendation: store **effective** means `MIX_MU` (already shifted) as the single source of truth, together
with a `centred_source_means` field for documentation.

Consequence of getting it wrong: h is shifted by ∓1.2704, so λ is scaled by e^{±1.27} ≈ 3.56. The bug is invisible in
convergence diagnostics but obvious in forecast-density calibration and in stochvol parity for μ (the SV level).

---

## 7. Offset c in `y* = log(ẽ² + c)`

### 7.1 Who uses what

| Source | c | Applied to | Notes |
|---|---|---|---|
| KSC 1998 (S1 §3.1) | **0.001** | y_t² (returns in %) | "introduced into the SV literature by Fuller (1996, pp. 494–7)… it is possible to let c depend on the actual value taken by y_t²" |
| OCSN 2007 (S2 §3.1) | **0.0001** | y_t² | "Because our ten component mixture approximation provides an improved fit to the left tail… we set c equal to 0.0001 which is smaller than the value c = 0.001 used by KSC" |
| Primiceri 2005 (S4 A.3) / DNP 2015 (S5 App. B.2) | **0.001** | squared **structural** residual `[A_t(y_t − X_t'B_t)]²_i` | "c is the offset constant (set to 0.001)" |
| CCMM 2024 code (S14) | 0.001 | structural residual², **before** outlier rescaling | KSC 7 |
| Kastner & Frühwirth-Schnatter 2014 (S15 §2.2) | mentions 10⁻³ (KSC) or 10⁻⁴ (OCSN) | (y − ȳ)² or y² | descriptive, no recommendation |
| stochvol 3.2.9 (S7) | **0**, or `sd(y)/10000` if no design matrix and any `y² == 0` | y²/τ (t errors) or y² | message to the console; JSS vignette §3.1 recommends demeaning instead. `svsample` then floors `log(y²/τ + offset)` at **−100** |
| factorstochvol 1.1.2 (S11) | 0, or `sd(y)/10000` if zeros and `factors == 0` | y² | |
| bayesianVARs 0.1.8 (S8) | **0** (`expert_sv_offset`, "Do not use!") | structural residual² | per equation, vector allowed |
| bvartools 0.3.0 (S9) | user `constant`; default **1e-4** (warning if unset) | y² | both mixtures |
| bvarsv 1.1 (S10) | 0.001 | residual² | Primiceri replication |
| Chan statespace-toolkit (S13) | **1e-4** | y² | |
| bsvars 4.0 (S12) | `log((u + 1e-9)²)` | **u**, before squaring | sign-asymmetric. This is not the KSC convention. |

### 7.2 Bias from the offset (derived here; checked numerically)

For ẽ ~ N(0, λ) and δ = c/λ, the bias `E log(ẽ² + c) − E log ẽ² = E log(1 + δ/x²)`, x ~ N(0,1), depends **only on c/λ**.
For small δ, `E log(1+δ/x²) ≈ φ(0) ∫ log(1 + δ/x²) dx = φ(0)·2π√δ = √(2πδ)`. Exact values by quadrature:

| c/λ | 1e-6 | 1e-4 | 1e-3 | 1e-2 | 1 | 10 |
|---|---|---|---|---|---|---|
| bias in y* (≈ in posterior h) | 0.0025 | 0.0250 | 0.0783 | 0.2411 | 1.8038 | 3.6614 |

At c/λ = 1e-12 the quadrature gives 2.5064e-6, against √(2πδ) = 2.5066e-6.

Implications:

- An **absolute** offset is not scale-equivariant. With c = 0.001, data in percent (λ ≈ 1) give a bias of about 0.08 in h. The
  same data in decimal units (λ ≈ 1e-4) give a bias of about 1.8–3.7 in h, which is catastrophic. Rescaling the data changes
  the posterior.
- Hence **recommended default: c_i = c_rel·ŝ_i²**. ŝ_i² is a per-window scale of equation i's residual variance (reuse the
  Minnesota ψ_i, the AR(p) residual variance recomputed inside each backtest window per R9). c_rel = 1e-12 gives a bias of
  about 2.5e-6 (quadrature re-checked: 2.506627e-6), so it is effectively c = 0 and gives stochvol/bayesianVARs parity. It
  prevents `log(0)` provided ŝ_i² > 0. Assert ŝ_i² > 0 and finite: a constant series gives ŝ² = 0 and hence c = 0. When
  ŝ² > 0, an exactly zero residual gives r ≈ log c_rel ≈ −27.6, which lands in component 10's tail. (ψ_i is the AR(p)
  variance of y_i, which is larger than the structural λ_i. With c_rel this small the difference does not matter.)
- With `exact=True` (§10) any c ≥ 0 gives the exact posterior as the **invariant** distribution. c does not affect the target,
  but it can affect **mixing** drastically, not only the acceptance rate. y* ≥ log c, so a large c/λ hides small residuals
  from the auxiliary proposal. The importance ratio e^{w} then becomes huge exactly where the exact posterior has mass, and
  the chain converges very slowly. Cross-check toy (`xc01_checks/dbg2.py`): one observation e = 0.02, h ~ N(−0.2, 1), KSC7,
  c = 0.5. The acceptance rate was 0.80, yet after 30 / 300 / 3000 sweeps from the prior E h was off by z = +15 / +5.5 / −0.1.
  A chain started at the exact posterior stayed there (z < 1). With c = 1e-3 the same chain was exact after 100 sweeps.
- Fuller's (1996) refinement of the offset (KSC footnote reference) was not checked: **UNVERIFIED / not needed**.

---

## 8. Indicator full conditional and how to draw it

### 8.1 Formula

For each equation i, time t (and chain), independently (S1 §3.2; S2 §2.3.2 with ρ = 0; S4 A.3):

$$ \Pr(s_{i,t}=j \mid y^*_{i,t}, h_{i,t}) = \frac{p_j\,\sigma_j^{-1}\exp\{-(r_{i,t}-\mu_j)^2/(2\sigma_j^2)\}}{\sum_{k=1}^K p_k\,\sigma_k^{-1}\exp\{-(r_{i,t}-\mu_k)^2/(2\sigma_k^2)\}}, \qquad r_{i,t}=y^*_{i,t}-h_{i,t}. $$

The conditional depends on the data and h only through r, and on nothing else: not on μ, φ, σ, β or the other equations.

### 8.2 Log-space computation (float64)

```
# constants (float64, module level): MIX_LOGW[j] = log p_j - 0.5*log σ_j²,  MIX_MU[j],  MIX_HALF_INV_VAR[j] = 0.5/σ_j²
r      = ystar - h                                   # (T,)
logits = MIX_LOGW[None, :] - (r[:, None] - MIX_MU[None, :])**2 * MIX_HALF_INV_VAR[None, :]   # (T, K)
s      = jax.random.categorical(key, logits, axis=-1)                                         # (T,) int32
mu_s, var_s = MIX_MU[s], MIX_VAR[s]                  # handed to the precision sampler for h
```

- `jax.random.categorical` uses Gumbel-max (K Gumbels per t). It handles `-inf` and underflowed logits robustly. It is exact
  only up to the resolution of the uniforms behind the Gumbel noise. The JAX docstring (checked on jax 0.12.0.1) says that
  with the default `mode="low"` "in float32 sampling will be biased for events with probability less than about 1E-7"
  (`mode="high"` pushes this to about 1E-14). With x64 enabled the noise is float64. A float64 uniform has 53 bits, so the
  largest Gumbel value is about −log(2⁻⁵³) ≈ 36.7. Only events with probability below about 1e-16 are distorted, which is
  irrelevant here. Still: pass float64 logits, assert the dtype, and never let the indicator draw run in float32.
  An alternative is inverse CDF with one uniform per t (S2 §2.3.2 "inverse distribution method"; stochvol does this on
  cumulative unnormalised weights): `w = exp(logits − max_j logits)`, `cw = cumsum(w)`,
  `s = min(Σ_j 1{cw_j ≤ u·cw_K}, K−1)`. Use `≤` so that zero-weight components are never selected even when u = 0. Clamp
  because `u·cw_K` can round up to `cw_K`. Pick one method and freeze it, since bitwise reproducibility (SPEC §8) depends on
  the RNG call pattern.
- Cost is O(T·K) per equation: 240k logits (about 1–2 Mflop) per sweep for 4 chains × n = 20 × T = 300 × K = 10. This is
  negligible next to the mean block.
- Shapes for vmap: write the kernel for one equation, (T,), then `vmap` over equations and chains. Keys are split per
  (chain, iteration, block, equation) per SPEC §8.
- Missing observations (mask m_t = 0): set observation precision `1/σ²_{s_t}` to 0 in the h sampler and draw s_t from its prior
  `p` (or keep it; it is irrelevant). Drawing from the prior keeps Geweke tests clean. In `exact=True` mode, masked t must
  also be dropped from both sums in w(h) (§10.2).
- Do not evaluate the weights in linear space. stochvol 3.2.9 computes `exp(κ_j − (r − m_j)²/(2v_j²))` without
  max-subtraction (S7, `find_mixture_indicator_cdf`). If every weight underflows (|r| ≳ 90–120), its inverse-CDF search
  returns component 1. That is harmless at realistic r, but the log-space form above has no such regime.

### 8.3 Common bugs this formula guards against (quantified)

Draw r from the mixture itself, so that the true marginal of s is p. Then `E_r[Pr(s = j | r)] = p_j` must hold (Bayes
consistency). Results with 2·10⁶ draws, OCSN10:

| p_j | .0061 .0478 .1306 .2067 .2272 .1884 .1205 .0559 .0158 .0012 |
|---|---|
| correct conditional | .0061 .0477 .1304 .2065 .2273 .1886 .1205 .0560 .0158 .0011 |
| **missing σ_j⁻¹ factor** | .0047 .0404 .1181 .1980 .2289 .1979 .1307 .0622 .0179 .0013 |
| **sd √(v_j²) used where the variance is expected** (row mislabelled in the first draft as "column used as sd") | .0039 .0426 .1332 .2167 .2313 .1840 .1151 .0548 .0169 .0015 |
| **variance column v_j² passed to an sd argument**, so the effective variance is (v_j²)² (cross-check) | .0138 .0571 .1217 .1869 .2184 .1966 .1319 .0592 .0139 .0007 |
| **bsvars 4.0 kernel** `exp(p_j − (r − m_j)²/v_j²)` (cross-check, S12) | .0221 .0635 .1238 .1811 .2038 .1805 .1276 .0691 .0253 .0031 |

Cross-check: the cross-check reran this test with a fresh seed. The correct row reproduced p_j within Monte Carlo error. The
other four rows all fail at 2·10⁶ draws.

---

## 9. Step order and the interface to the rest of the sweep

The SPEC §4.3 order is correct and is exactly DNP's Algorithm 2 (S5 §3): blocks Σ (here h) and (θ, s). θ (mean block, a_i,
γ_i, τ, o, SV parameters) is drawn from its conditional *marginalised over s*, using the exact Gaussian likelihood. Then s is
drawn given θ, and h is drawn given s.

```
mean block (a_i, γ_i) | h, τ, o          # exact Gaussian likelihood, no s
τ (t errors), o (outliers) | resid, h    # exact likelihoods, no s
--- VolBlock.sample(resid) ---
ẽ  = resid / (sqrt(τ)·o);  y* = log(ẽ² + c)
s  | y*, h                               # §8, immediately before h
h  | y*, s, (μ, φ, σ)                    # banded precision sampler (other note); MH-corrected if exact=True (§10)
(μ, φ, σ) | h   [centred]                # no s needed
ASIS non-centred step | h̃, y*, s         # uses the s-conditional Gaussian likelihood (see §10.3)
```

Original Primiceri (2005) order (s drawn *after* h, using stale θ) is wrong even with an exact mixture (S5 §2). Keep it as a
**negative control** in the Geweke suite (§12.3). Cross-check toy (`xc01_checks/toy_alg123.py`, §12.6) puts a size on it. With
y_t = β + e^{h_t/2}ε_t, T = 3 and 4·10⁵ chains, Algorithm 1 misses the quadrature posterior by |z| = 27–224 with both tables
(c = 1e-3 and 0.5). Algorithms 2 and 3 pass at c = 1e-3, except KSC7 in Algorithm 2 with E h₁ at z = +5.1, which is the
approximation error showing up through a small residual (y₁ − β ≈ 0.05).

For `Linear(form="reduced")` (a global block, SPEC §4.3) the same rule applies at the global level. All mean and covariance
parameters that enter the structural residuals must be drawn before **any** equation's s. Otherwise the s of a later equation
is stale. bayesianVARs does exactly this (Φ → U → residuals → per-equation s → h; S8), and stochvol's regression sampler also
draws s right before h (S7).

Where τ, o or the mean block are drawn between the s and h draws, the s draw is stale. That is invalid: SPEC §4.3 forbids
inserting any block between them. Enforce it structurally: the `VolBlock.sample` call draws s and h together and returns
neither separately.

---

## 10. Exactness: the mixture is an approximation

### 10.1 What the default sampler targets

With the mean block using the exact likelihood and the vol block using the mixture likelihood, the sweep alternates between
two likelihoods. It is exactly correct only if g = f (DNP Algorithm 2: "a sampler from an approximate posterior", S5 §3). The
approximation error is tiny for OCSN10: KL 7.9e-5 nats per observation. DNP's Primiceri application shows Algorithms 2 and 3
"indistinguishable" (S5 §5). Still, SPEC R6 requires that sampler bugs be detectable, so we provide an exact mode.

### 10.2 Exact correction (DNP 2015 Algorithm 3; Stroud et al. 2003)

Augmented target: `p(θ, h | y) · p̃(s | h, θ, y)`, where `p̃(s | ·) = p̃(y* | h, s) π(s) / c(h)` and
`c(h) = Π_t g(y*_t − h_t)` (S5 App. B.2 eqs. 4–5). Propose `h' ~ p̃(h | y*, s, θ)`, the usual Gaussian precision-sampler draw,
and accept with

$$ \log\alpha = w(h') - w(h), \qquad w(h) = \sum_t \Big[\log N(\tilde e_t;\,0,\,e^{h_t}) - \log g(y^*_t - h_t)\Big] $$

(S5 App. B.2 eq. 7, in log-variance units with per-equation factorisation.) Notes:

- `log N(ẽ; 0, e^h) = −½(log 2π + h + ẽ² e^{−h})`. Compute it from ẽ, not from y*. The Jacobian dy*/dẽ does not depend on h, so
  it cancels (S1 §3.4: "w = const + log f(y|h) − log k(y*|h)").
- It is valid for **any** c ≥ 0, meaning the invariant distribution is exact. c (and the table) change the proposal
  quality, and with it the **mixing**, possibly drastically (§7.2 toy: acceptance 0.80, yet thousands of sweeps to forget a
  prior start at c/λ ≈ 0.5). The importance ratio e^{w} is unbounded in regions the auxiliary model under-covers, so the
  independence-type MH step need not be geometrically ergodic (cross-check observation; the general theory is Mengersen &
  Tweedie 1996, not re-read here). A Geweke chain sits near stationarity and may not show this. A backtest started from a
  crude h initialisation can.
- Cross-check numerical confirmation. Started at the exact posterior, the MH chain stays there at c = 0.5 with KSC7
  (`xc01_checks/dbg2.py`, `dbg3.py`; |z| ≤ 1.1). From a prior start, both tables approach the quadrature answer, slowly at
  c = 0.5 (`toy_alg123.py`: |z| ≤ 3.4 after 60 sweeps) and quickly at c = 1e-3 (|z| ≤ 2.8).
- It is per equation and needs no cross-equation terms, so it vmaps.
- On rejection, keep h. s stays as drawn. It is valid because s is part of the (θ, s) block drawn before.
- Initial state h₀ (stochvol-style AR(1) with a free h₀): either make the MH step act on the joint proposal (h₀, h_{1:T}),
  or propose h_{1:T} with h₀ integrated out and then draw h₀ | h₁, θ exactly. w does not involve h₀.
- Report the acceptance rate as a diagnostic, together with the longest run of consecutive rejections and the spread of
  w(h) across stored draws. The acceptance rate alone is not enough (toy above). Expect a rate close to 1 for OCSN10 and
  noticeably lower for KSC7. Heuristic, not verified: if the h proposal behaved like an independence proposal with
  log-weights ≈ N(·, s²), acceptance would be about 2Φ(−s/√2). The S2 Fig. 6 posterior sds (s = 0.05 for K = 10, 0.92 for
  K = 7, T = 1000) then give about 0.97 and 0.51. The cross-check toy, which has independent h_t and per-t MH, measured
  per-t acceptance of 0.999 (OCSN10) and 0.981 (KSC7) at c = 1e-3. The joint T = 300 path rate is **UNVERIFIED**; measure
  it in the pilot.

Alternative: ex-post importance reweighting (S1 §3.4 eq. 12; S2 §2.4) with weights `w_j ∝ exp(w(θ^j, h^j))` summed over
equations. It is cheaper but turns every downstream statistic (forecasts, scores) into a weighted one. The `ForecastResult`
contract (SPEC §7) assumes equally weighted draws, so prefer the MH correction. stochvol offers both through
`correct_model_misspecification` (default FALSE): resampling for fast SV, MH for general SV. This is now confirmed from
stochvol's behaviour, not only its docs (S7). The fast-SV weight is exactly the w of this section. The resampling is
`sample.int(…, replace = TRUE, prob = w)` over the stored draws, so a "corrected" stochvol fixture contains duplicated draws
and its MCSE/ESS are not comparable to an MCMC chain.

### 10.3 Interaction with ASIS (DERIVED here, not from a paper; re-derived and toy-checked in the cross-check; still verify by Geweke test)

The non-centred ASIS step redraws (μ, σ) given h̃ = (h − μ)/σ from the *s-conditional* Gaussian regression
`y*_t − μ_{s_t} = μ + σ h̃_t + σ_{s_t} z_t`. This form follows Kastner & Frühwirth-Schnatter (2014). **Verified by the
cross-check:** S15 eq. (7) and §2.5 eq. (10) regress (ỹ_t − m_{r_t})/s_{r_t} on (1/s_{r_t}, h̃_t/s_{r_t}) with a joint
Gaussian prior on (μ, σ). That is an exact Gibbs step for the approximate augmented target. Under the exact target (10.2),
using that conditional as an MH proposal gives the acceptance ratio `exp{w(h_new) − w(h_old)}` with `h = μ + σ h̃` (the same
w). The derivation mirrors S5 App. B.2. The exact augmented target is π = π̃·exp{w(h)}, where π̃ is the approximate augmented
target, so the target is proportional to the proposal times `exp{w(h)}·const`. The reparametrisation Jacobian σ^{−T} (or
σ^{−(T+1)} with h₀) appears in both π and π̃ and cancels. So in `exact=True` mode, apply the same w-ratio accept/reject to
the ASIS move. The centred (μ, φ, σ) | h draw needs no correction because it does not move h, and p̃(s | h, y) does not
involve (μ, φ, σ).

Two precisions added by the cross-check:

- If the non-centred step is itself a Metropolis–Hastings step and not an exact Gibbs draw (for example when the σ prior is
  not the Gaussian/half-normal one, or φ is moved jointly), use **one** accept/reject with
  `log α = log α_aux + w(h_new) − w(h_old)`, where α_aux is the step's own ratio under π̃. Two sequential tests
  (delayed acceptance) are also valid, but they are a different kernel, so pick one and freeze it.
- Numerical check (`xc01_checks/dbg3.py`) on a toy with h_t = μ + h̃_t, μ ~ N(0, 1), T = 3 and KSC7. Chains were started at
  the exact posterior (SIR from 8·10⁶ prior draws), then run 40 sweeps of [s → h-MH → centred μ → NC μ-move]. With the
  w-correction on the NC move, E μ and E μ² stay at the quadrature values (|z| ≤ 1.1 at c = 0.5 and c = 1e-3). Without the
  correction (the uncorrected NC move inside an otherwise exact sampler), E μ² is off by z = −12.9 at c = 1e-3 and E μ by
  z = +281 at c = 0.5. Both the correction and the claim that it is needed are therefore confirmed on a toy. A Geweke test of
  the real kernel (§12.3(2)) is still required.

---

## 11. Edge cases

1. **ẽ_t = 0 exactly** (constant series segments, duplicated data, a perfect fit): `log 0 = −∞` gives NaN. The relative offset
   (§7.2) prevents it, provided ŝ² > 0. A fully constant series gives ŝ² = 0, so raise on it. Add an assertion test with
   an exactly zero residual. (stochvol's −100 floor, S7, is its own guard. bayesianVARs has none at c = 0.)
2. **Tiny |ẽ| relative to √λ** (far-left tail, r ≲ −15): KSC7 log-density errors reach about ±2 (§5.4). OCSN10 stays within
   about 0.3 down to r = −20. Use OCSN10 by default; exact mode removes the issue.
3. **Large positive r** (bad initial h, or a huge outlier with o fixed at 1): the pmf moves to the *high-variance* components,
   not to component 1 (largest mean). Computed for OCSN10: the mode is component 6 at r = 4, component 8 at r = 6, component 9
   at r = 10 (0.905) and component 10 for r ≥ 20. This is harmless but slows burn-in. Initialise h at a constant path
   `log ŝ_i²` per equation.
4. **Double shift / missing shift** (§6): caught by the mean test.
5. **Variance vs sd confusion**: every source prints variances (σ_i², v_i²). Caught by the §8.3 consistency test.
6. **Tiny weights**: KSC component 3 has p = 0.00002 (log p = −10.8) and OCSN component 10 has p = 0.00115. Both are fine in
   log space and float64. Never compute `p·N(·)` in linear space without max-subtraction.
7. **Absolute offset in parity mode with rescaled data**: document that `offset=1e-3` reproduces Primiceri/bvarsv only on data
   with the same units as theirs.
8. **t errors / outliers**: apply the offset after normalisation (§3). Recorded in metadata.
9. **float32 leakage**: all mixture constants must be created as float64 arrays after `jax_enable_x64`. Test the dtypes
   (SPEC §8).
10. **Masked t**: see §8.2.

---

## 12. How to test it

### 12.1 Closed-form and constant checks (fast, CI)

Script used for every number in §5.4 and §7.2 (original):
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/scripts/mixture_check.py`
(plus `indicator_bugs.py` for §8.3). An independent re-implementation from the cross-check, written from the papers, is
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc01_checks/xc_mixture.py`
(`bias_tiny.py` gives the log-substitution quadrature for tiny c/λ). Port both as tests:

| Test | Assertion |
|---|---|
| weights | `abs(sum(p) − 1) < 1e-12` (both tables) |
| centring | `abs(Σ p μ − (−1.2703628)) < 2e-4` (KSC: −1.270399; OCSN: −1.270280) |
| variance | `abs(Σ p(σ² + (μ − mean)²) − π²/2) < 2e-3` (4.934854; 4.933731) |
| log-normal moments | E[e^ξ] = 1 ± 1e-3, E[e^{2ξ}] = 3 ± 1e-2 |
| density | OCSN10: max abs(f − g) on [−8.759, 1.892] < 5e-4; KSC7 < 1.1e-2; KL(f‖g) < 1e-4 (OCSN) |
| transcription | `MIX_LOGW` equals the §5.4 vectors to 1e-8 (OCSN vector independently matches stochvol) |
| KSC re-sort | sorted(μ_KSC) equals OCSN Table-1 left block to 1e-10 |
| offset bias | quadrature value for c/λ = 1e-4 equals 0.02497 ± 1e-4; bias(c, λ) depends only on c/λ |
| scale equivariance | with the relative offset, scaling the data by k shifts the h draws by exactly 2 log k (same seed, to fp tolerance). **Cross-check correction:** this holds only if every prior is scale-equivariant as well. The SV-level prior (μ ~ N(b_μ, B_μ)) must be shifted by 2 log k, for example through a data-based b_μ = log ŝ² recomputed per window, and the mean-block prior scales must follow ŝ (Minnesota ψ does). With a fixed b_μ, the test must fail. Test it at kernel level instead: the s-kernel and the h-given-s kernel with μ shifted by 2 log k. |
| dtype | all constants and outputs are float64 |

### 12.2 Indicator kernel tests

- **Exact pmf**: for a grid of r ∈ {−25, −15, −8, −3, −1.27, 0, 1, 3}, draw 10⁶ indicators and run a χ² goodness-of-fit test
  against the §8.1 probabilities. Reference values for OCSN10: at r = −8 the pmf is ≈ (0, 0, 0, 0, 0, 0, .0077, .5916, .3995,
  .0011); at r = 1 it is ≈ (.0009, .1904, .5228, .2364, .0440, .0051, .0004, 0, 0, 0).
- **Bayes consistency** (§8.3): r drawn from the mixture, so that `E[Pr(s = j | r)] = p_j` within MC error.
- **Unit Geweke for the indicator step** with h fixed. Marginal-conditional: s ~ p, y* ~ N(h + μ_s, σ_s²). Successive-conditional:
  alternate s | y*, h (our kernel) with y* | s, h. Test functions: 1{s = j} for each j, y*, y*². This catches every bug in §8.3
  and §11.5 in seconds.

### 12.3 Geweke (2004) joint-distribution tests (SPEC §10.3)

1. **Auxiliary-model test (exact by construction).** Model: θ_sv ~ prior, h ~ AR(1), s_t ~ Cat(p),
   y*_t = h_t + μ_{s_t} + σ_{s_t} z_t. Sampler: [s | y*, h] → [h | y*, s, θ] → [θ | h] → ASIS. Data step: y* | h, s. Test
   functions: μ, φ, σ, σ², h_1, h_T, mean_t h_t, mean_t μ_{s_t}, 1{s_t = j}. This must pass at any size. It isolates the
   mixture, precision-sampler and ASIS logic from the approximation question.
2. **Full model, exact mode.** Data: e_t = exp(h_t/2) ε_t with Gaussian ε (and τ, o for the variants). Sampler with
   `exact=True` (§10.2–10.3). Must pass strictly, with Holm correction (SPEC §10.3). Cross-check notes:
   (a) Use OCSN10 with the relative offset here, so that the MH step mixes well (§10.2).
   (b) A data-dependent c (c_rel·ŝ²(y)) is fine. For each fixed y the kernel leaves p(θ, h | y) invariant, and the stale s
   carried across the data step is redrawn before it is used, because the mean block marginalises s.
   (c) Do **not** use test functions of s in this full-model test. After the data step s no longer follows p̃(s | h, θ, y_new),
   and the marginal-conditional simulator does not generate s.
3. **Full model, approximate mode** (`exact=False`, OCSN10). Expected to pass at CI sizes, since the bias is of order KL ≈ 1e-4
   per observation. Whether a CI-sized Geweke run can detect the approximation is **UNVERIFIED**. Record the z-scores and treat a
   failure as a signal to inspect, not as proof of a bug. DNP (S5 §5, online appendix) found Alg. 2 "approximately correct"
   by Geweke tests with KSC7. One data point from the cross-check (a posterior check on fixed data, not a Geweke test): with
   4·10⁵ independent draws, KSC7 Algorithm 2 at c = 1e-3 was detectably off (E h₁, z = +5.1) on data containing one small
   residual, while OCSN10 was not (|z| ≤ 1.7).
4. **Negative control**: Primiceri's Algorithm 1 order (s drawn after h, θ not conditioned on s) **must fail** (S5 §5 reports
   this). This demonstrates the suite has power. KSC7 in approximate mode at large T is a second power check: the posterior
   log-weight sd of about 0.9 (S2 Fig. 6) may be detectable (**UNVERIFIED**).

Proper priors are required. Use tight-ish SV priors so that h stays in a numerically sane range during the
successive-conditional simulator, for example φ ∈ (0.5, 0.95) and σ² around 0.05.

### 12.4 SBC (Talts et al. 2018; Modrák et al. 2025)

Test quantities per equation (thin to reduce autocorrelation; randomise ties for discrete quantities):

- h_t at t ∈ {1, ⌊T/2⌋, T}; mean_t h_t; μ, φ, σ.
- **Exact log-likelihood** `Σ_t log N(ẽ_t; 0, e^{h_t})` (Modrák et al.: likelihood-type quantities increase sensitivity).
- **Mixture log-likelihood** `Σ_t log g(y*_t − h_t)`, and their difference, the **log-weight** `w(h)` of §10.2. In approximate
  mode, mis-calibration shows up first in w. In exact mode, w is just another function and must be uniform.
- Auxiliary-model SBC only: `Σ_t μ_{s_t}` and `Σ_t log σ²_{s_t}` (functions of the latent s).

### 12.5 Parity oracles (SPEC §10.4): exact mapping

| Oracle | Mixture | Offset to set on our side | Correction | Parametrisation mapping (mixture-relevant only) |
|---|---|---|---|---|
| **stochvol** 3.2.9 `svsample` | OCSN10 unshifted (identical constants) | `c = 0`. The fixture generator must assert that y has no exact zeros (more precisely `y² == 0` with no design matrix), since stochvol otherwise uses `sd(y)/10000` (store `myoffset` in the fixture attributes). Our side must also reproduce or avoid the **−100 floor** on `log(y²/τ + offset)`: generate fixtures whose min `log(y²)` is well above −100. | stochvol default `correct_model_misspecification = FALSE` ⇒ compare with our `exact=False`. **The fixture must hit stochvol's fast sampler:** ρ = 0, μ normal (or 0), (φ+1)/2 beta, σ² ~ Gamma(½, ·), which is stochvol's default `priorsigma`. Other σ² priors switch to the general (leverage-mixture) sampler, a different algorithm. Optionally a second fixture with TRUE vs our `exact=True`. With TRUE and no t/regression, stochvol *resamples* the stored draws (SIR, duplicates), so compare quantiles only, not MCSE-based bands. With TRUE plus t errors or regression it uses the general sampler with MH. | h is log-variance in both. stochvol `y_t = exp(h_t/2) ε_t` equals our univariate ẽ. With t errors the offset sits on y²/τ in both. (Prior mapping: other note.) |
| **bayesianVARs** 0.1.8 | OCSN10 via stochvol C++ | `c = 0` (`expert_sv_offset` default) | none. **Verified by the cross-check:** `stochvol::update_fast_sv` has no correction option (`ExpertSpec_FastSV`), and stochvol's correction lives only in `svsample`'s loop and R wrapper | applied to the structural residual of its Cholesky factorisation. Map the factorisation in the CCCM note. No −100 floor on this path. |
| **bvartools** 0.3.0 | user's choice (`stochvol_ksc1998` / `stochvol_ocsn2007`) | `c = constant` (default 1e-4) | none | `y* = ln(y² + constant)`; h is log-variance (docs, and **verified by the cross-check** from the indicator density `N(y*; h + μ_j, σ_j²)`); random-walk h with innovation **variance** argument `sigma` |
| **bvarsv** 1.1 (Phase 5) | KSC7 (raw m, +1.2704 inside the density) | `c = 1e-3` absolute | none | **Corrected by the cross-check:** bvarsv's internal state `Sigtdraw` is already the **log-variance** (our h). Its sd is `exp(0.5·Sigtdraw)` and its `W` is the innovation covariance of the log-variance. Primiceri's paper uses `h^P = log σ` (our h = 2h^P), but bvarsv's code does not. Whether bvarsv's default W prior equals Primiceri's prior × 4 is **UNVERIFIED** (Phase 5). |
| **Chan toolkit** (MIT, Python-side spot check) | KSC7 shifted | `c = 1e-4` | reweighting example in ex07. Its weight uses the exact density of `y* = log(e^h ε² + c)`, which equals our w up to an h-free Jacobian | h is log-variance, random-walk SV |
| **bsvars** 4.0 | **not an oracle** (cross-check): wrong indicator kernel in CRAN 4.0 (S12) | — | — | Phase 6 only. Use version ≥ the 2026-08-07 fix if it is ever used for SV parity. |

Fixtures compare posterior quantiles within 4 MCSE (SPEC §10.4). Because the RNG streams differ, parity is distributional
only. The indicator-sampling method (Gumbel vs inverse CDF) does not matter for parity.

### 12.6 Posterior-vs-quadrature toy tests (added by the cross-check; fast, CI-sized after shrinking N)

These are cheap end-to-end tests of step order and exactness that need neither Geweke nor R. Prototypes (numpy, original):
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc01_checks/toy_alg123.py`,
`dbg2.py` and `dbg3.py` in the same folder.

| Test | Model | Pass criterion | Observed (4·10⁵ chains unless stated) |
|---|---|---|---|
| step order | y_t = β + e^{h_t/2}ε_t, β ~ N(0,1), h_t iid N(0,1), T = 3, y = (0.05, −1.5, 0.3); exact posterior by 2-D quadrature | From a prior start (60–120 sweeps), Alg. 2 at c = 1e-3 with OCSN10 and Alg. 3 at c ≤ 1e-3 must give \|z\| < 4 on E β, E β², E h₁, E h₁². At large c, use the invariance test below: from a prior start Alg. 3 at c = 0.5 still showed \|z\| up to 3.4 after 60 sweeps. Alg. 1 must fail. | Alg. 1: \|z\| = 27–224. Alg. 2: OCSN10 \|z\| ≤ 1.7; KSC7 E h₁ z = +5.1. Alg. 3 at c = 1e-3: \|z\| ≤ 2.8. (P(β > 0) carries a +1 to +2 z bias from the quadrature grid; use a finer grid or drop it.) |
| exact-mode invariance | same, or one observation | started at the exact posterior (SIR), 30–40 sweeps of Alg. 3 at c = 0.5: moments unchanged | \|z\| ≤ 1.1 (`dbg2.py`, `dbg3.py`) |
| ASIS correction | h_t = μ + h̃_t, μ ~ N(0,1), T = 3 | with the w-corrected NC move, invariance as above; without it, must fail | with: \|z\| ≤ 1.1; without: z = −12.9 (c = 1e-3), +281 (c = 0.5) |
| exact-mode burn-in (diagnostic, not pass/fail) | one observation, KSC7, c = 0.5 | documents slow mixing at large c/λ | from the prior: z = +15 / +5.5 / −0.1 after 30 / 300 / 3000 sweeps (acceptance 0.80) |

---

## 13. Software summary: which uses which

| Package / paper | Components | Shift handling | Offset c | Correction |
|---|---|---|---|---|
| KSC 1998 | 7 | m − 1.2704 | 0.001 | reweighting §3.4 |
| OCSN 2007 | 10 (+ a, b for leverage) | none | 0.0001 | reweighting §2.4 |
| Primiceri 2005 | 7 | m − 1.2704 | 0.001 | none (fn. 26) |
| Del Negro–Primiceri 2015 | 7 | m − 1.2704 | 0.001 | Alg. 2 (approx.), Alg. 3 (exact MH) |
| CCMM 2024 code | 7 | m − 1.2704 | 0.001 | none seen |
| stochvol 3.2.9 | 10 | none | 0 / sd(y)/1e4 if zeros and no regressors; log floored at −100 | optional (default off): SIR resampling (fast SV) or MH (general SV) |
| factorstochvol 1.1.2 | 10 (stochvol) | none | 0 / sd(y)/1e4 if zeros and 0 factors | not checked |
| bayesianVARs 0.1.8 | 10 (stochvol) | none | 0 | none (verified via the `update_fast_sv` API, cross-check) |
| bvartools 0.3.0 | 7 and 10 | 7: m − 1.2704; 10: none | user, default 1e-4 | none |
| bvarsv 1.1 | 7 | +1.2704 on the residual side | 0.001 | none |
| bsvars 4.0 | 10 | none | `(u + 1e-9)²` | none seen. **CRAN 4.0 indicator kernel is wrong** (no log p, no v⁻¹, no ½); fixed in dev 4.0.0.9000 (cross-check, S12) |
| Chan toolkit (MIT) | 7 | m − 1.2704 | 1e-4 | reweighting (example) |
| **rai-bvar (proposed)** | **10 default**, 7 optional | stores effective means | **relative c_rel = 1e-12** (parity: 0, 1e-4, 1e-3) | `exact=False` default; `exact=True` = DNP Alg. 3 + ASIS w-ratio |

---

## 14. UNVERIFIED items

1. The published JoE 2007 Table 1 and the published RES 1998 Table 4 were not opened (paywall / the fetched copy was HTML;
   the cross-check's attempt at the Harvard-hosted copy returned 403). The constants are verified against the authors'
   working papers, including the last pre-publication revision (CIRJE-F-297, April 2006), the authors' 2024 reprint, and
   four independent software transcriptions. Mertens' code cites "OCSN page 436" for c = 1e-4, which suggests the published
   §3.1 keeps the WP text.
2. The fitting criterion for the OCSN 10-component table (not stated in the 2004, 2005 or 2006 WP text).
3. Stroud, Müller & Polson (2003) eq. (11) was not opened (bibliographic data confirmed on Crossref:
   doi:10.1198/016214503000161). The MH ratio is taken from DNP App. B.2 eq. (7) and re-derived in §10.2.
4. The ASIS w-ratio correction in §10.3 is derived here, not published. The cross-check re-derived it and confirmed it on a
   toy (§12.6). It must still be validated on the real kernel by Geweke test 12.3(2).
5. Whether CI-sized Geweke/SBC runs can detect the OCSN10 or KSC7 approximation in `exact=False` mode. Also the joint-path
   (T = 300) MH acceptance rate and burn-in of `exact=True`, for KSC7 and for OCSN10 (§10.2 gives only a heuristic and toy
   per-t rates).
6. ~~bayesianVARs and bsvars: absence of any misspecification correction (grep only). bvartools h-parametrisation (docs
   only).~~ Resolved by the cross-check: bayesianVARs has no correction (API-level, S8), bvartools' h is the log-variance
   (S9), and bsvars 4.0 has no correction plus a wrong kernel (S12). Remaining: in which CRAN release bsvars' dev fix
   (2026-08-07) will appear. CRAN 4.0, published 2026-08-22, still has the old kernel.
7. Fuller (1996) offset refinement: not checked, not used.
8. Section and page numbers of the published KSC/OCSN/Primiceri articles. The citations above use working-paper section
   numbers (KSC July 1997; OCSN Sept 2005 revision; Primiceri July 2004). The volume, issue and page ranges of KSC (RES 65(3)
   361–393), Primiceri (RES 72(3) 821–852), DNP (RES 82(4) 1342–1345), OCSN (JoE 140(2) 425–449), Stroud et al.
   (JASA 98(462) 377–386) and KFS (CSDA 76 408–423) were confirmed on Crossref by the cross-check.
9. (Cross-check) bvarsv's default W prior and its relation to Primiceri's log-sd W (factor 4?). Phase 5.
10. (Cross-check) The non-geometric-ergodicity remark in §10.2 cites Mengersen & Tweedie (1996) from memory. The practical
    claim (slow mixing at large c/λ) is backed by the toy runs.

---

## Cross-check log

An independent cross-check re-derived the formulas, recomputed every constant-derived number, re-read the primary sources
and confirmed software behaviour (behaviour only; no GPL code transcribed). Scripts are original and live in
`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc01_checks/`:
`xc_mixture.py`, `bias_tiny.py`, `toy_alg123.py`, `toy_asis_exact.py`, `dbg1.py`, `dbg2.py`, `dbg3.py`, `is_check.py`.

**Verified without change**

- KSC Table 4 (p, m, σ²), the §2.1 constants (−1.2704, 4.93), §3.1 c = 0.001 and its Fuller quote, the §3.1 moment-matching
  method, the §3.2 pmf and the §3.4 eq. (12) reweighting. Source: S1 WP pp. 11–13, 17–18.
- OCSN Table 1, both blocks, all entries. Checked against the 2004 WP, the 2005 revision, the **April 2006 CIRJE revision
  (new)** and Hiraki–Chib–Omori 2024 Table 1. The left block equals KSC − 1.2704, re-sorted, to 1e-15. Also verified: §2.3.2
  pmf with the v⁻¹ factor, "inverse distribution method", §2.4 reweighting, §3.1 c = 0.0001, and the Fig. 6 sd values
  0.917 / 0.0535.
- Exact log χ²₁ moments: E = ψ(½) + log 2 = −γ − log 2 = −1.2703628454614782, Var = π²/2, skew −1.53514, kurtosis 7.
  The density formula f(ξ) = (2π)^{-1/2} exp{(ξ − e^ξ)/2} was re-derived by change of variables.
- Every number in §5.4 (mixture moments, log-normal moments, max |f − g|, TV, KL, sd of log f/g, tail log-ratios, κ_j
  vectors), the §7.2 bias table, the √(2πδ) asymptotic (re-derived: ∫ log(1 + 1/u²) du over ℝ = 2π), and the §11.3 and §12.2
  pmf examples. The κ_j vector also matches stochvol's precomputed `mix_pre` to 15 digits.
- Primiceri App. A.3 eqs. (17)–(19), "c … (set to 0.001)", A.7 table, fn. 26. DNP Algorithms 1–3, App. B.2 eqs. (4)–(7)
  with `y** = log(y*² + 0.001)`, and §5 Geweke statements.
- Software constants and offsets: bvartools (KSC with −1.2704, OCSN unshifted, default 1e-4 with warning), bvarsv (+1.2704
  inside the density, 0.001, the 1/√(2πu²) factor), Mertens' helpers (0.001 / 0.0001), the Chan MIT toolkit (KSC − 1.2704,
  c = 1e-4; repository exists, MIT licence), factorstochvol (sd(y)/1e4 only with zeros and 0 factors), bayesianVARs
  (`expert_sv_offset` = 0, "Do not use!").
- Bibliographic data for S1, S2, S4, S5, S6 and S15 via Crossref.

**Corrections and additions** (each with its source)

1. §1 S2, §5.5, §14: added the April 2006 CIRJE-F-297 revision, the last pre-publication version. It is identical on every
   item this note uses. Source: https://www.cirje.e.u-tokyo.ac.jp/research/dp/2004/2004cf297.pdf, Table 1, §2.3.2, §3.1,
   Fig. 6.
2. §0, §2, §7.2, §13: the offset multiplier was called κ, which clashes with κ_j (log indicator weights). Renamed to c_rel.
   Added a remark on the μ/σ symbol clash between mixture constants and SV parameters.
3. §7.2 and §10.2: "c then affects only the MH acceptance rate" was **wrong**. c leaves the invariant distribution exact but
   can make the exact-mode MH mix extremely slowly while the acceptance rate looks healthy. Source: cross-check toy
   `dbg2.py` (acceptance 0.80; z = +15 / +5.5 / −0.1 after 30 / 300 / 3000 sweeps from the prior; z < 1 from the exact
   posterior). Theory: unbounded importance ratio (Mengersen & Tweedie 1996, cited from memory). Added the diagnostics
   (longest rejection run, spread of w), an acceptance heuristic, and h₀ handling.
4. §8.3: the row "table column used as sd, not variance" was **mislabelled**. Its numbers come from using the sd √v² as the
   variance. Relabelled it, and added the opposite bug (variance passed as sd) with recomputed numbers. Source:
   `xc_mixture.py`; first author's `indicator_bugs.py` (`VV = np.sqrt(V)` used as variance).
5. §1 S12, §5.5, §8.3, §12.5, §13: **bsvars 4.0 (CRAN) uses a wrong indicator kernel**, `exp(p_j − (r − m_j)²/v_j²)`.
   Fixed upstream in 4.0.0.9000. bsvars is not a mixture oracle, and the first draft's "constants identical" remark hid
   this. Sources: CRAN `bsvars_4.0.tar.gz` `src/sv.cpp` (Date/Publication 2026-08-22); GitHub bsvars/bsvars master
   `src/sv.cpp`, commit 5d6ac212ae (2026-08-07), NEWS.md (PR #160).
6. §1 S7, §7.1, §12.5, §13: stochvol details. The offset rule also requires "no design matrix". `log(y²/τ + offset)` is
   floored at −100 (`clamp_log_data2`). The fast sampler is used only under ρ = 0, normal μ, beta φ, Gamma(½, ·) σ², and with
   the correction on only without t errors or regression. The correction is SIR resampling with replacement (fast SV) or
   MH (general SV). Fast-SV order: s → h → θ → τ → β (DNP-consistent). Source: stochvol 3.2.9 `src/utils_main.h`,
   `src/utils_main.cc`, `src/sampling_main.cc`, `src/single_update.cc`, `R/wrappers.R` (behaviour only).
7. §1 S8, §12.5, §13, §14: bayesianVARs having no correction is now **verified** at API level, not by grep. It calls
   `stochvol::update_fast_sv`, and stochvol's `ExpertSpec_FastSV` (`inst/include/expert.hpp`) has no correction field. Order
   Φ → U → residual → SV is DNP-consistent. Source: bayesianVARs `src/bvar_cpp.cpp`; stochvol `inst/include/expert.hpp`.
8. §1 S9, §12.5, §14: bvartools' h is the log-variance, now **verified** from the indicator density `N(y*; h + μ_j, σ_j²)`,
   not only from the docs. Source: bvartools `src/stochvol_ocsn2007.cpp`.
9. §1 S10, §2, §12.5: bvarsv's internal state is the **log-variance** (`sigt = exp(0.5·Sigtdraw)`). The first draft mapped
   bvarsv as if it used Primiceri's log-sd. Corrected. The W-prior mapping is flagged for Phase 5. Source: bvarsv
   `src/bvarsv.cpp` (`sigmahelper2`).
10. §10.3, §1 S15: the Kastner & Frühwirth-Schnatter non-centred regression form is now **verified** (arXiv:1706.05280,
    eqs. (7), (10), §2.5–2.6). The w-correction was re-derived. Added the combined ratio for a non-Gibbs NC step and the
    Jacobian remark. Toy confirmation in `dbg3.py`: corrected |z| ≤ 1.1; uncorrected z = −12.9 (c = 1e-3) and +281 (c = 0.5).
11. §12.1: the scale-equivariance test as written would fail with a fixed SV-level prior. It now requires scale-equivariant
    priors (data-based b_μ) or a kernel-level test.
12. §12.3: added that exact-mode Geweke tests should use OCSN10 with the tiny offset, that a data-dependent c is valid, and
    that test functions of s are invalid in the full-model test. Added the toy evidence on approximation detectability.
13. §8.2: the claim that `jax.random.categorical` is "exact" was qualified with the JAX docstring's resolution caveat
    (float32 "low" mode biased below about 1e-7). It is negligible in float64; assert the dtype. The "240k fused flops" was
    reworded as 240k logits (about 1–2 Mflop). Added that masked t must be dropped from w, and noted the linear-space
    underflow behaviour of stochvol's kernel. Source: jax 0.12.0.1 `jax.random.categorical` / `gumbel` docstrings.
14. §3, §7.1: CCMM applies the offset to the raw structural residual **before** outlier rescaling, unlike our normalised ẽ.
    Recommended computing y* with `logaddexp`. Source: CCMM `mcmcVARSVO.m` (`logy2 = log((RESID*A_').^2 + logy2offset)`),
    behaviour only.
15. §4: "1.2704 in every source (S1, S4, S7–S14)" was inaccurate. S7, S8, S11 and S12 use OCSN10 with no shift. The claim
    that KSC "fitted with that literal" is marked as an inference.
16. §5.3: re-derived a_j = e^{v²/8} and b_j = ½e^{v²/8}. Confirmed the first author's remark on the 5th-decimal b differences
    (j = 1, 8, 9, 10).
17. §9 and §12.6 (new): DNP Algorithm 1 vs 2 vs 3 measured against quadrature on a regression toy. Added a CI-able
    posterior-vs-quadrature test table, and the reduced-form global-block rule (all mean parameters before any s).
18. §11.1, §7.2: the relative offset needs ŝ² > 0. Raise on constant series.

**Still UNVERIFIED after the cross-check:** published JoE/RES page images (paywalled); the OCSN 10-component fitting
criterion; Stroud et al. (2003) eq. (11) text; the joint-path exact-mode acceptance and burn-in at T = 300; CI-scale Geweke
power against the approximation; bvarsv's W-prior factor; the Mengersen–Tweedie citation; Chan's book §10.1 / Table 10.1
(cited from code comments only).
