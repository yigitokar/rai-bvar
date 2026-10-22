# 02 — Univariate stochastic-volatility block (AWOL precision sampler + ASIS)

Phase 2 research note for `rai-bvar` (`SV`, `SV(phi=1)`; the base used by `SV(outliers=True)` and `SV(tails="t")`).
SPEC references: §2.1 (clean room), §3 R4/R6, §4.1–4.4 (VolBlock, Gibbs sweep, step order), §8 (numerical rules), §10.3–10.4 (Geweke/SBC/parity), §13 Phase 2.

**Clean-room status.** Every algorithm below is specified from papers: Kastner & Frühwirth-Schnatter (2014, "KFS"), Kastner (2016), Hosszejni & Kastner (2021), Omori, Chib, Shephard & Nakajima (2007, "OCSN"), Kim, Shephard & Chib (1998, "KSC"), Del Negro & Primiceri (2015, "DNP"), Chan (2023), Chan & Jeliazkov (2009, "CJ"), McCausland, Miller & Pelletier (2011, "MMP"). Where a formula is not printed in a paper, it was **derived here** and then **checked numerically** against dense linear algebra and by Geweke tests in a clean-room NumPy prototype (§13.0). GPL sources (`stochvol` on GitHub, CRAN manuals) were read only to confirm published defaults and behaviour (default constants, step order, offset rule, `latent0` semantics). No code was transcribed. Items not confirmed are marked **UNVERIFIED**.

**Cross-checked (2026-10-21) by a second agent.** Every formula was re-derived and re-checked numerically with independent scripts (`…/phase2-research/xc02_check/`, not the first author's prototype). Constants and priors were checked against the primary sources and, for constants, against a second source. Corrections are made in place and listed in the **Cross-check log** at the end. The main changes are: (i) the `b_T` formula in §6.1 loses up to 5 digits and was replaced by a centred form; (ii) stochvol's NC `|σ|` behaviour is now confirmed from its source, and a clean-room reimplementation of it fails Geweke (§7); (iii) the KFS three-block `φ` ratio question is settled by a Geweke test (§6.3); (iv) bvarsv's code works in log-variance units, not log-sd, so the prior mapping for that oracle was corrected (§8.2); (v) several former UNVERIFIED items are now resolved (§14).

---

## 0. Executive summary (what to build)

1. **Model:** `e_t = exp(h_t/2) ε_t`, `h_t = μ + φ(h_{t−1} − μ) + σ η_t` (`t = 1..T`), `h_0 ~ N(μ, σ²/ω₀)` with `ω₀ = 1 − φ²` (stationary, default) or `ω₀ = 1/B₀` (fixed multiplier). Random walk (`φ = 1`) is a separate static code path with parameters `(h_0, σ)`.
2. **Per sweep, inside the SV block, in this order:** (c) indicators `r | y*, h`, then (a) `h_{0:T} | y*, r, θ` by a tridiagonal precision sampler (one `lax.scan` forward, one backward, O(T)), then (b) `θ` in the centred parametrisation (stochvol-default 2-block MH), then (b\*\*) `θ` again in the non-centred parametrisation (ASIS, "GIS-C"). The indicators are **not** part of the persistent state. This order satisfies Del Negro & Primiceri (2015), and it is the order `stochvol`'s single-update routine uses (confirmed in its source).
3. **Mixture:** the 10-component OCSN table (§2.3, constants verified from the OCSN working paper). Offset `c = 0` by default (bayesianVARs default), with a floor against `log 0`.
4. **Priors (stochvol/bayesianVARs parametrisation):** `μ ~ N(b_μ, B_μ)`, `(φ+1)/2 ~ Beta(a_φ, b_φ)`, `σ² ~ B_σ·χ²₁ = Gamma(½, rate 1/(2B_σ))`, which is equivalent to `σ ~ |N(0, B_σ)|`. Recommended defaults = bayesianVARs Cholesky-SV defaults: `b_μ = 0, B_μ = 100²`, `Beta(20, 1.5)`, `B_σ = 1`, stationary `h_0`.
5. **Optional exact correction** of the mixture approximation, by two cheap MH accept/reject steps (§9). With it, the composed sampler targets the exact Gaussian-SV posterior (DNP "Algorithm 3"). At `T = 300` the prototype accepted about 96–98% of `h` moves and about 99% of NC `(μ, σ)` moves. **Recommendation:** default ON (it makes R6 Geweke/SBC exact), with OFF available for stochvol/bayesianVARs-style runs. This is a decision for the orchestrator.

---

## 1. Notation

| Symbol | Meaning | Notes / units |
| :--- | :--- | :--- |
| `T` | observations seen by the SV block (per equation) | `t = 1..T`; state index `t = 0..T` |
| `e_t` | residual passed to the SV block | structural residual of equation `i`, divided by `√τ_t` (t-tails) or `o_t` (outliers) when those are active |
| `c` | offset in `y*_t = log(e_t² + c)` | default `0` (§2.4) |
| `y*_t` | log squared residual | observation of the linearised model |
| `h_t` | log variance `log λ_{i,t}` | SPEC §4.4 `log λ`; **not** log std. dev. (Primiceri's *paper* uses log std. dev.; bvarsv's *code* uses log variance, §8.2) |
| `μ, φ, σ` | level, persistence, volatility of log variance | `σ > 0`; `σ²` is the innovation variance |
| `γ` | `μ(1 − φ)`, AR intercept | used by the C-step regression |
| `h̃_t` | `(h_t − μ)/σ`, non-centred (NC) state | `h̃_t = φ h̃_{t−1} + η_t` |
| `ξ_t` | `log ε_t²`, `ε_t ~ N(0,1)` | `log χ²₁`: mean `ψ(½) + log 2 = −1.270363`, variance `π²/2 = 4.934802` |
| `K` | number of mixture components | 10 (OCSN), or 7 (KSC) |
| `p_k, m_k, v_k` | mixture weight, mean, **variance** | KFS write `s²_k` for `v_k` |
| `r_t` | mixture indicator, `∈ {1..K}` | KFS `r_t`; DNP `s_t` |
| `ω₀(φ)` | prior precision factor of `h_0`: `h_0 ~ N(μ, σ²/ω₀)` | stationary: `1 − φ²`; fixed: `1/B₀` |
| `b_μ, B_μ` | prior mean and **variance** of `μ` | stochvol's `priormu` is (mean, **sd**) |
| `a_φ, b_φ` | Beta shapes for `(φ+1)/2` | KFS `a₀, b₀` (renamed here to avoid a clash with the RW `h_0` prior) |
| `B_σ` | `σ² ~ B_σ χ²₁` | stochvol `priorsigma` |
| `c₀, C₀` | `σ² ~ IG(c₀, C₀)` (shape, scale) | density `∝ x^{−c₀−1} e^{−C₀/x}` |
| `m₀, V₀` | RW initial state prior `h_0 ~ N(m₀, V₀)` | only for `φ = 1` |
| `B₀¹¹, B₀²²` | KFS auxiliary prior variances of `(γ, φ)` | `10¹²` (γ), `10⁸` (φ), KFS §2.4. These are proposal-only: they do not change the target. stochvol's `proposal_intercept_var = 1e12`, `proposal_phi_var = 1e8` carry the same values; bayesianVARs passes them swapped (§13.6) |
| `Q, c` | posterior precision and canonical mean of `h_{0:T}` | `h ~ N(Q⁻¹c, Q⁻¹)`. This `c` is unrelated to the offset `c` of §2.4; the context tells them apart. In code, name them `offset` and `canon` |
| `L` | lower bidiagonal Cholesky factor, `Q = L L'` | diagonal `l_t`, subdiagonal `o_t` |

---

## 2. Model, linearisation, mixture

### 2.1 Stationary AR(1) log volatility (default `SV()`)

$$
e_t = e^{h_t/2}\varepsilon_t,\qquad h_t = \mu + \phi(h_{t-1}-\mu) + \sigma\eta_t,\qquad \varepsilon_t,\eta_t \overset{iid}{\sim} N(0,1),\quad t=1,\dots,T,
$$
$$
h_0\mid\mu,\phi,\sigma \sim N\!\big(\mu,\ \sigma^2/\omega_0(\phi)\big),\qquad \omega_0=\begin{cases}1-\phi^2 & \text{"stationary" (default; KFS eq. 2 and following text; stochvol default)}\\ 1/B_0 & \text{fixed: } h_0\sim N(\mu, B_0\sigma^2)\end{cases}
$$

- The convention "`h_0` has no observation, `y_1` loads on `h_1`" is the KFS/stochvol convention (KFS eqs. 1–2; Kastner 2016 eqs. 1–3). KSC (1997 WP §2.2, eq. 4) and Chan (2023, eq. 5) instead put the stationary prior on `h_1`, which has an observation, and have no `h_0`. **Under the stationary prior the two models coincide**: integrating `h_0` out of the KFS model gives `h_1 ~ N(μ, σ²/(1−φ²))`. They differ only for a non-stationary `h_0` prior. When comparing `σ²` conditionals, KFS has `T+1` states and KSC has `T`, so the IG shape differs by ½ (KSC eq. 4: `(n + σ_r)/2`; §6.1 step 5 here: `c₀ + (T+1)/2`). (Cross-check addition.)
- `stochvol` numeric `priorlatent0 = B` means `h_0 ~ N(μ, B·σ²)`. This is confirmed in the C++ source, which uses standard deviation `√B·σ`. The `specify_priors` help text says "`σ²/v`", which is **inconsistent** with the code. Use the variance-multiplier reading. (Cross-check: re-confirmed. `determine_Bh0inv` in `src/utils.h` returns `1 − φ²` or `1/value`, the latent-state sampler uses that as `ω₀`, and the `svsample` help itself says "`h_0 ~ N(μ, Bσ²)`".)
- A **diffuse / fixed-scale** alternative `h_0 ~ N(m₀, V₀)`, with a prior that does not scale with `σ`, is natural for the RW case (§8). For the AR case it is supported by the same formulas with `ω₀` replaced by an additive pseudo-observation (§5.4).

### 2.2 Linearisation

`y*_t = log(e_t² + c)`. With `c = 0`: `y*_t = h_t + ξ_t`, where `ξ_t = log ε_t²` has exact density
`f(ξ) = (2π)^{−1/2} exp{(ξ − e^ξ)/2}` (OCSN WP §2.1). Conditional on `r_t = k`: `ξ_t ~ N(m_k, v_k)`, so the model is linear Gaussian in `h` (KFS eq. 7).

### 2.3 Mixture tables (verified from OCSN 2004 WP, Table 1; second sources listed below)

**OCSN K = 10 (default).** These are means and variances of `log ε²` directly (no −1.2704 shift needed). Second source: all 30 numbers are identical to the constants `mix_prob`, `mix_mean`, `mix_var` in stochvol 3.2.9 `src/utils_latent_states.h`, read for the constants only.

| k | p_k | m_k | v_k |
| ---: | ---: | ---: | ---: |
| 1 | 0.00609 | 1.92677 | 0.11265 |
| 2 | 0.04775 | 1.34744 | 0.17788 |
| 3 | 0.13057 | 0.73504 | 0.26768 |
| 4 | 0.20674 | 0.02266 | 0.40611 |
| 5 | 0.22715 | −0.85173 | 0.62699 |
| 6 | 0.18842 | −1.97278 | 0.98583 |
| 7 | 0.12047 | −3.46788 | 1.57469 |
| 8 | 0.05591 | −5.55246 | 2.54498 |
| 9 | 0.01575 | −8.68384 | 4.16591 |
| 10 | 0.00115 | −14.65000 | 7.33342 |

(The `(a_k, b_k)` columns of OCSN Table 1 are for the leverage model only and are not needed.)

**KSC K = 7** (for Primiceri/bvarsv-style replication). The values are as reproduced in OCSN Table 1 (left block), with the means already shifted. **Verified against KSC itself** (1997 Nuffield WP, https://www.nuffield.ox.ac.uk/economics/papers/1996/w26/Chib8.pdf, §3.1 eqs. 9–10 and Table 4). KSC print unshifted `m_i` in a different row order and use `N(m_i − 1.2704, v_i²)`. All seven shifted means reproduce the column below exactly, e.g. 2.77786 − 1.2704 = 1.50746 and −10.12999 − 1.2704 = −11.40039. A third source agrees: Chan's MIT-licensed toolkit (`ksc_ar1_mean.m`: same weights, unshifted means, `− 1.2704`).

| k | p_k | m_k | v_k |
| ---: | ---: | ---: | ---: |
| 1 | 0.04395 | 1.50746 | 0.16735 |
| 2 | 0.24566 | 0.52478 | 0.34023 |
| 3 | 0.34001 | −0.65098 | 0.64009 |
| 4 | 0.25750 | −2.35859 | 1.26261 |
| 5 | 0.10556 | −5.24321 | 2.61369 |
| 6 | 0.00002 | −9.83726 | 5.17950 |
| 7 | 0.00730 | −11.40039 | 5.79596 |

**Computed checks** (script `mix_check.py`; these are the unit-test targets):

| Quantity | OCSN-10 | exact `log χ²₁` | KSC-7 |
| :--- | ---: | ---: | ---: |
| `Σ p_k` | 1.00000 | 1 | 1.00000 |
| mean | −1.270280 | −1.270363 | −1.270399 |
| variance | 4.933731 | 4.934802 | 4.934854 |
| skewness | −1.5343 | −1.5351 | −1.5352 |
| excess kurtosis | 3.9931 | 4.0000 | 4.0006 |
| max abs density error on [−20, 5] | 3.84e−4 (at ξ ≈ 1.68) | — | 1.03e−2 (at ξ ≈ 0.38) |
| max `\|log f/g\|` on [−12, 3] | 0.0675 (at ξ ≈ −10.6) | — | — |
| max `\|log f/g\|` between the 1st and 99th percentiles of `log χ²₁` (−8.759, 1.892) | 0.0123 | — | 0.473 |

(Cross-check: every entry was reproduced by an independent script, `xc02_check/c1_mixture.py`. The KSC-7 skewness, kurtosis and density-error entries were added. The exact moments are `ψ(½) + log 2`, `ψ′(½) = π²/2`, `ψ″(½)/ψ′(½)^{3/2}` and `ψ‴(½)/ψ′(½)² = 4`.)

Store the constants as float64 module-level arrays, with `log p_k` and `log v_k` precomputed.

### 2.4 Offset `c` (SPEC §8: "documented and tested")

| Source | Offset | Status |
| :--- | :--- | :--- |
| KSC (1998) | `log(y_t² + c)` on mean-corrected returns, `c = 0.001` | verified in KSC 1997 WP §3.1 ("Throughout we will set c = 0.001", after eq. 9; offset attributed to Fuller 1996); also cited by KFS §2.2 and OCSN WP §3 |
| OCSN (2007) | `log(y_t² + c)`, `c = 10⁻⁴`, set smaller than KSC's because the ten-component mixture fits the left tail better | verified, OCSN WP §3 |
| Primiceri (2005) | `log[(y*)² + c̄]`, `c̄ = 0.001` | **verified** (cross-check): Primiceri 2005 (July 2004 version) Appendix A.3; DNP App. B.2 eq. (7) uses `log(y*² + 0.001)`; bvarsv 1.1 C++ also uses 0.001 |
| CCM (2016 WP §8.2) | `ln(ṽ² + c)`, "offset constant used in the KSC algorithm" | the WP states **no value** (checked) |
| `stochvol` ≥ 3 (`svsample`) | `c = 0`, unless some `y_t² == 0` exactly and there is no design matrix; then `c = sd(y)/10000` is added to `y²` with a message | confirmed (`R/wrappers.R` of 3.2.9; Kastner 2016 §3.1, where it is called a warning) |
| `stochvol` C++ API | `myoffset = 0` default | confirmed (manual) |
| `bayesianVARs` | `expert_sv_offset = 0` ("Do not use!") | confirmed (manual v0.1.8) |

**Recommendation.** Default `c = 0`, which matches bayesianVARs and stochvol on data without exact zeros. Compute `y*_t = log(max(e_t², e_min²))` with `e_min²` = the smallest normal float64 (about 2.2e−308). The floor then activates only on an exact zero. Count activations as a diagnostic, outside `jit`. (Cross-check caution: a floor hit gives `y*_t ≈ −708`, which is about 255 standard deviations below mixture component 10 (`m = −14.65`, `v = 7.33`). The AWOL draw then pulls `h_t` towards very negative values, and that `h_t` feeds the next mean-block weight `exp(−h_t)`. A more benign choice is to **mask** a floor-hit observation in the SV block, `m_t = 0` as in §10.3 item 7, and report it. An exact zero has probability zero under the model, so either rule is a convention: document it in the module docstring.) The offset is part of the target definition, so it must be a fixed constant for the whole run, never adapted per iteration. An optional user-set absolute `c` should be documented in data units, since an absolute offset is scale dependent. With `c > 0` the exact correction of §9 is no longer exact; require `c = 0` when `mixture_correction=True`.

**Edge case seen in the prototype.** If `exp(h_t/2)` falls below the rounding error of the mean fit (≈1e−16·|y_t|), `e_t = y_t − x_t'β` is pure rounding noise, possibly exactly 0. The DNP-order sampler never got there (min `h` ≈ −11 over 50k sweeps). The wrong-order sampler did, within 48–485 sweeps (§3.3). Data should be on a sensible scale. SPEC's standardisation, if any, should happen before the MCMC.

---

## 3. Sampler architecture and step order

### 3.1 Del Negro & Primiceri (2015) in one paragraph

With the KSC/OCSN auxiliary mixture, the indicators `s` are augmented data. Primiceri's original order ("Algorithm 1": `Σ | θ, s` → `s | Σ, θ` → `θ | Σ`) is wrong, because `θ` is drawn without conditioning on `s`. That is not a valid Gibbs step once `s` exists, even with an exact mixture. The fix ("Algorithm 2") is a 2-block Gibbs sampler on `Σ` and `(θ, s)`: draw `θ | Σ` (marginal of `s`, i.e. the true Gaussian likelihood), then `s | Σ, θ`, then `Σ | θ, s`. **The indicators must be drawn immediately before the volatilities and after everything that changes the residuals.** "Algorithm 3" additionally turns the volatility draw into an MH step, which removes the mixture approximation error (Stroud, Müller & Polson 2003). Sources: DNP staff-report version, §§2–4 and Appendix B; Chan (2023) §4.2.1 and footnotes 8–9. (Cross-check: re-read in full. Algorithm 1 is `Σ | θ, s` → `s | Σ, θ` → `θ | Σ`. Algorithm 2 is `Σ | θ, s` → [`θ | Σ` then `s | Σ, θ`]. Algorithm 3 is Algorithm 2 with step 1 turned into an MH step; its acceptance ratio is App. B.2 eq. (7), which follows Stroud et al. 2003 eq. (11). DNP App. D reports Geweke tests: Algorithm 1 incorrect, Algorithm 2 "approximately correct", Algorithm 3 "fully correct".) Footnote 9: with several conditionally independent log-volatility series, pairing each series' indicators and volatilities is valid. That is exactly our per-equation `vmap`.

### 3.2 Order inside a `rai-bvar` sweep (refines SPEC §4.3)

```text
glob ← GlobalBlock.sample(...)      # hyperparameters; CCCM reduced-form β block if form="reduced"
vmap over equations i:
    (a_i, γ_i) ← MeanBlock.sample(weights = exp(−h_{i,1:T}))   # true Gaussian likelihood, marginal of r
    e_i ← y_i + Σ_{j<i} a_ij y_j − X γ_i
    [SV(tails="t"): τ_i | e_i, h_i, ν_i ;  SV(outliers=True): o_i | e_i, h_i, p_i]   # exact conditionals, marginal of r
    ẽ_i ← e_i / scale_i ;  y* ← log(ẽ_i² + c)
    ── SV block: one VolBlock.sample call ──
    (c)   r ← indicators | y*, h                       # FIRST: residuals are now final for this sweep
    (a)   h_{0:T} ← AWOL | y*, r, μ, φ, σ              # immediately after r (optional exact-MH, §9)
    (b)   (μ, φ, σ) ← C-step | h                       # does not depend on r or y*
    (b*)  h̃ ← (h − μ)/σ
    (b**) (μ, φ, σ, h̃) ← NC-step | h̃, r, y*          # depends on r: valid because nothing changed y* since (c)
    (b***) h ← μ + σ h̃
    [ν_i | τ_i ; p_i | o_i]                            # depend only on τ / o, can go anywhere
```

- `(c)(a)(b)(b*)(b**)(b***)` is a cyclic rotation of KFS's GIS-C `(a)(b)(b*)(b**)(b***)(c)` (KFS Algorithm 2). For a stand-alone SV model the two are equivalent. With a mean block, only the rotated order is valid.
- `stochvol`'s single-step updater (`update_fast_sv`, used by `bayesianVARs` and `factorstochvol`) runs indicators → latent vector → parameters (C then NC). This is confirmed in its source, so it has the same order as here. (Cross-check: re-confirmed in stochvol 3.2.9 `src/single_update.cc`, whose comments read "Step (c)", then "Step (a)", then "Step (b)". bayesianVARs 0.1.8 calls `stochvol::update_fast_sv` in `src/bvar_cpp.cpp`, and factorstochvol's `update_fsv.cpp` references it too.)
- The `t`/outlier scale draws and the mean draw are all "marginal of `r`" collapsed steps. Validity follows from partially collapsed Gibbs reasoning: each collapsed draw is followed by a draw of `r` before any step conditions on `r`.
- `r` is **not** stored between sweeps. VolState holds `h_{0:T}, μ, φ, σ` (plus `τ`, `o`, `ν`, `p` in the extended blocks).

### 3.3 Prototype evidence on step order (regression × SV)

The prototype model is `y_t = βx_t + e^{h_t/2}ε_t`, `T = 20`, with Geweke successive-conditional regeneration of `y`:

- DNP order (as §3.2): passes Geweke at 200k sweeps, both with and without the exact correction (§13.4).
- **Stale-indicator order** (Primiceri Algorithm 1: `β | h` → `h | r_old, y*(β_new)` → `θ` → `r | h, β`) **diverged in 5 of 5 seeds within 48–485 sweeps**. `h_t` ran to about −720 and `exp(−h)` overflowed. The DNP order with identical code and seeds ran 50,000 sweeps with `min h ≈ −6 … −11`. The mechanism is positive feedback: a stale low-variance component pins `h_t` to a too-small value, the mean block then interpolates `y_t`, `e_t → 0`, and `y*_t` falls further. This is a stronger failure than the bias DNP report for real data, and it makes a good regression test (§13.4, T13).

---

## 4. Step (c): mixture indicators

For `t = 1..T` and `k = 1..K`, with `ξ_t = y*_t − h_t` (C) or `ξ_t = y*_t − μ − σh̃_t` (NC):

$$
\log \pi_{tk} = \log p_k - \tfrac12\log v_k - \frac{(\xi_t - m_k)^2}{2v_k} + \text{const},\qquad r_t \sim \text{Categorical}(\operatorname{softmax}_k \log\pi_{tk})
$$

(KFS §2.6; OCSN.) KFS use log-scale normalisation and inverse-transform sampling. In JAX use `jax.random.categorical(key, logits, axis=-1)` on a `(T, K)` array. That is the Gumbel-max trick: exact, branch-free, and it needs no normalisation. The cost is `T·K` exponentials-equivalent; KFS §2.6 call this step "computationally rather expensive". (Cross-check, JAX docs: `categorical` has a `mode` argument, `"low"` by default or `"high"`, that controls how many bits the Gumbel sampler uses. In **float32**, `"low"` is biased for events with probability below about 1e−7, and `"high"` lowers that to about 1e−14 at roughly twice the cost. With x64 enabled the logits are float64. The bias threshold for float64 is not documented (**UNVERIFIED**). Pass float64 logits and add a χ² test, T8.) Missing observations (§10.3) have no `y*_t`: draw anything and mask.

---

## 5. Step (a): AWOL / precision sampler for `h_{0:T}`

### 5.1 General precision form (CJ 2009; Rue 2001; MMP 2011)

Stack `h = (h_0, …, h_T)'`. The prior is `H h = δ + u`, with `H` lower bidiagonal (1 on the diagonal, `−φ` below), `δ = (μ, γ, …, γ)'`, `u ~ N(0, S)`, `S = diag(σ²/ω₀, σ², …, σ²)`. So `K_h = H'S⁻¹H` (tridiagonal) and the prior mean is `μ1`. The observation is `y* − m_r = G h + ε`, with `G = [0 | I_T]` and `ε ~ N(0, V_r)`, `V_r = diag(v_{r_1}, …)`. Then

$$
h \mid y^*, r, \theta \sim N(Q^{-1}c,\ Q^{-1}),\qquad Q = K_h + G'V_r^{-1}G,\qquad c = K_h\,\mu\mathbf 1 + G'V_r^{-1}(y^*-m_r).
$$

### 5.2 Explicit entries, centred, joint `(T+1)` system (derived; checked against dense algebra to 4e−15)

| entry | value |
| :--- | :--- |
| `Q_00` | `(ω₀ + φ²)/σ²` (stationary: `1/σ²`) |
| `Q_tt`, `1 ≤ t ≤ T−1` | `1/v_{r_t} + (1 + φ²)/σ²` |
| `Q_TT` | `1/v_{r_T} + 1/σ²` |
| `Q_{t−1,t} = Q_{t,t−1}`, `1 ≤ t ≤ T` | `−φ/σ²` |
| `c_0` | `μ(ω₀ − φ(1 − φ))/σ²` (stationary: `μ(1 − φ)/σ²`) |
| `c_t`, `1 ≤ t ≤ T−1` | `(y*_t − m_{r_t})/v_{r_t} + μ(1 − φ)²/σ²` |
| `c_T` | `(y*_T − m_{r_T})/v_{r_T} + μ(1 − φ)/σ²` |

These coincide with the entries in `stochvol` ≥ 3, which also samples the joint `(h_0, …, h_T)` system. That was confirmed by reading the source after deriving them.

**KFS's printed version (§2.3)** samples only `h_{1:T}` with `h_0` integrated out, then `h_0 | h_1`. In the stationary case, `Ω_11 = 1/v_{r_1} + 1/σ²` and `c_1 = (y*_1 − m_{r_1})/v_{r_1} + μ(1−φ)/σ²`, other entries as above, followed by `h_0 | h_1 ~ N(μ + φ(h_1 − μ), σ²)`. General `ω₀` (derived, verified as the Schur complement): `Ω_11 = 1/v_{r_1} + 1/(σ²(1 + φ²B₀)) + φ²/σ²`, `c_1 = (y*_1 − m_{r_1})/v_{r_1} + μ[1/(σ²(1 + φ²B₀)) + (φ² − φ)/σ²]`, and
`h_0 | h_1 ~ N( μ + φB₀(h_1 − μ)/(1 + φ²B₀),  σ²B₀/(1 + φ²B₀) )` with `B₀ = 1/ω₀`.
**Recommendation:** use the joint `(T+1)` system. It is one code path for stationary, fixed-`B₀`, diffuse and RW initial conditions, and it gives `h_0` without an extra step.

### 5.3 Non-centred system and an identity

For `h̃ = (h − μ)/σ`: `Q̃ = σ²Q`, exactly. Entries: `Q̃_00 = ω₀ + φ²`, `Q̃_tt = σ²/v_{r_t} + 1 + φ²`, `Q̃_TT = σ²/v_{r_T} + 1`, off-diagonal `−φ`; `c̃_0 = 0`, `c̃_t = σ(y*_t − m_{r_t} − μ)/v_{r_t}` (KFS §2.3, NC display). The draw `h = μ + σh̃` has exactly the C conditional distribution (verified to 2e−15). Since `Q̃ = σ²Q`, both systems have identical conditioning. Computing the draw in the NC form avoids forming `1/σ²` when `σ` is tiny, so it is the **recommended** way to compute step (a) whichever baseline is used.

### 5.4 Fixed-scale (diffuse) `h_0 ~ N(m₀, V₀)` in the AR model

Replace the `h_0` row: `Q_00 = 1/V₀ + φ²/σ²`, `c_0 = m₀/V₀ − φμ(1−φ)/σ²`, other rows unchanged. In the NC parametrisation `h_0` then depends on `(μ, σ)`, so the NC regression of §7 gains one pseudo-observation row `m₀ = μ + σh̃_0 + e_0`, `e_0 ~ N(0, V₀)`. The C-step drops the `h_0` term from `σ²` and `(γ, φ)`. Not prototyped. Prefer the stationary default.

### 5.5 O(T) banded Cholesky, forward and back substitution

For tridiagonal SPD `Q` with diagonal `d_0..d_n` and subdiagonal `e_0..e_{n−1}` (`n = T`):

```text
l_0 = sqrt(d_0);  v_0 = c_0 / l_0
for t = 1..n:                      # lax.scan, carry (l_{t-1}, v_{t-1})
    o_{t-1} = e_{t-1} / l_{t-1}
    piv     = d_t − o_{t-1}²        # must be > 0 (SPD); record min(piv) for diagnostics
    l_t     = sqrt(piv)
    v_t     = (c_t − o_{t-1} v_{t-1}) / l_t      # forward solve L v = c
w = v + z,  z ~ N(0, I_{n+1})
x_n = w_n / l_n
for t = n−1..0:                    # lax.scan(reverse=True)
    x_t = (w_t − o_t x_{t+1}) / l_t              # back solve L' x = w
return x                           # x ~ N(Q⁻¹c, Q⁻¹)
```

- This is KFS §2.3 ("compute `Ω = LL'` … solve `La = c` … `L'h = a + ε`") and SPEC §8's "Gaussian draws from precision form", specialised to bandwidth 1. MMP's "CFA" method and CJ's precision sampler are the same computation for this model. The equation numbers of those two papers are **UNVERIFIED**: both are paywalled and were not accessed.
- `log|Q| = 2 Σ_t log l_t` comes for free.
- JAX: two `lax.scan`s of length `T+1` per series under `vmap` over (chains × equations). Use `unroll=4–16` to cut while-loop overhead on CPU, and benchmark it. There is no silent jitter (SPEC §8). Return `min(piv)` and `isfinite` flags in the state, and raise outside `jit`.
- **Alternative without Cholesky ("perturb-then-solve"):** `u = H'S^{−1/2}z₁ + G'V_r^{−1/2}z₂` has `Cov(u) = Q`, so `x = Q⁻¹(c + u) ~ N(Q⁻¹c, Q⁻¹)` with one tridiagonal solve (for example `jax.lax.linalg.tridiagonal_solve`). It needs `2T+1` normals and gives no log-det. (Cross-check, JAX docs, latest: the signature is `jax.lax.linalg.tridiagonal_solve(dl, d, du, b, *, perturb_singular=False)` with `dl[0] = 0` and `du[m−1] = 0` by convention, and it uses a partially pivoted LU factorisation; leading batch dimensions are allowed. The repository does not pin JAX yet. Float64 CPU performance, and whether it beats two `lax.scan`s under `vmap`, remain **UNVERIFIED**; decide in the Phase 2 pilot.)
- Parallel-in-time (associative scan) Cholesky is possible for GPU and is not needed at `T ≈ 300`.

### 5.6 Integrated likelihood given `r` (CJ 2009 idea; formula derived, verified to 6e−14 against a dense MVN)

$$
\log \tilde p(y^*\mid r,\theta) = -\tfrac T2\log 2\pi - \tfrac12\sum_t \log v_{r_t} - \tfrac12\sum_t \frac{(y^*_t-m_{r_t})^2}{v_{r_t}} + \tfrac12\big[\log\omega_0 - (T+1)\log\sigma^2\big] - \sum_{t=0}^{T}\log l_t - \frac{\mu^2}{2\sigma^2}\big[\omega_0 + T(1-\phi)^2\big] + \tfrac12\lVert v\rVert^2,
$$

where `v` is the forward-solve vector of §5.5. Use it as a deterministic oracle against a Kalman-filter log likelihood (§13.2) and as an SBC test quantity.

---

## 6. Step (b): `θ = (μ, φ, σ²)` in the centred parametrisation (KFS §2.4)

Write the state equation as the regression `h_t = γ + φh_{t−1} + σ η_t`, `t = 1..T`, with `X = [1, h_{t−1}]_{t=1..T}` (`T×2`) and `z = h_{1:T}`.
- Implied prior: `γ | φ ~ N(b_μ(1−φ), B_μ(1−φ)²)`. This includes the Jacobian of `μ ↦ γ`; KFS §2.4.
- Auxiliary conjugate prior (proposal only): `(γ, φ) | σ² ~ N(0, σ²B₀)`, `B₀ = diag(10¹², 10⁸)`, and `p_aux(σ²) ∝ σ⁻¹ = (σ²)^{−1/2}`, an improper `IG(−½, 0)`.
- Auxiliary posterior: `(γ, φ) | h, σ² ~ N(b_T, σ²B_T)`, `B_T = (X'X + B₀⁻¹)⁻¹`, `b_T = B_T X'z` (KFS eq. 8).
- `ℓ₀(μ, φ, σ²) := log N(h_0; μ, σ²/ω₀(φ))`, `log p_φ(φ) = (a_φ−1)log(1+φ) + (b_φ−1)log(1−φ) + const` on `|φ| < 1`, else `−∞` (KFS eq. 5; Kastner 2016 eq. 4).

### 6.1 Two-block sampler (stochvol default: `mh_blocking_steps = 2`, confirmed)

Block 1, `(γ, φ) | h, σ²` (independence MH):
1. Propose `(γ*, φ*) ~ N(b_T, σ²B_T)` at the current `σ²`. If `|φ*| ≥ 1`, reject. This is stochvol's default `proposal_phi = "immediate acceptance-rejection"`. The alternative, re-proposing until `|φ*| < 1`, is also valid **here**: it samples the truncated proposal, whose normalising constant `P(|φ*| < 1 | h, σ²)` is the same for the old and the new point within this block, so it cancels. (Cross-check caveat: it does **not** cancel in the one-block sampler of §6.2, where `σ²` is proposed too. There, re-proposing needs the extra factor `Z(σ²_old)/Z(σ²*)`, or use immediate rejection.)
2. With `μ* = γ*/(1−φ*)`:
$$
\log R = \Big[\ell_0(\mu^*,\phi^*,\sigma^2) + \log N\big(\gamma^*; b_\mu(1-\phi^*), B_\mu(1-\phi^*)^2\big) + \log p_\phi(\phi^*) + \tfrac{1}{2\sigma^2}\big(\tfrac{\gamma^{*2}}{B_0^{11}} + \tfrac{\phi^{*2}}{B_0^{22}}\big)\Big] - \Big[\text{same at }(\gamma,\phi,\mu)\Big].
$$
The last term is `−log p_aux(γ, φ | σ²)` up to a `σ²`-constant (KFS §2.4, 2-block ratio).

Block 2, `σ² | h, μ, φ` (independence MH under the Gamma prior):
3. `C_T = ½[ Σ_{t=1}^T (h_t − μ − φ(h_{t−1} − μ))² + ω₀(φ)(h_0 − μ)² ]`, `c_T = T/2` (KFS eq. 9). The `h_0` density contributes `(σ²)^{−1/2}` because its variance scales with `σ²`, both for stationary and for fixed `B₀`.
4. Propose `σ²* = C_T / G` with `G ~ Gamma(c_T, 1)`, i.e. `σ²* ~ IG(c_T, C_T)`. Accept with `log R = (σ² − σ²*)/(2B_σ)`. This is the prior/aux ratio for `σ² ~ Gamma(½, 1/(2B_σ))`.
5. Under an `IG(c₀, C₀)` prior on `σ²` this step is exact Gibbs: `σ² ~ IG(c₀ + (T+1)/2, C₀ + C_T)`. The `+½` is for the `h_0` term; it becomes `c₀ + T/2` if the `h_0` prior does not scale with `σ²`.

**Numerically stable 2×2 algebra (implementation trick; corrected in the cross-check).** With `x_t = h_{t−1}`, `z_t = h_t` (`t = 1..T`), `x̄ = mean(x)`, `z̄ = mean(z)`, `S_c = Σ(x_t − x̄)²`, `S_xz = Σ(x_t − x̄)(z_t − z̄)` (**centred** cross-product), `ε₁ = 1/B₀¹¹`, `ε₂ = 1/B₀²²`:
`A = X'X + B₀⁻¹ = [[T+ε₁, Tx̄], [Tx̄, S_c + Tx̄² + ε₂]]`, `det A = T·S_c + T·ε₂ + ε₁·A₂₂`, with no cancellation.
`φ̂ = [T·S_xz + ε₁(S_xz + T x̄ z̄)] / det`, `γ̂ = [T z̄ (S_c + ε₂) − T x̄ S_xz] / det`, i.e. `b_T = (γ̂, φ̂)`. These are algebraically equal to `(1/det)[A₂₂ s_z − A₁₂ s_xz, −A₁₂ s_z + A₁₁ s_xz]` with uncentred `s_z = Σz_t`, `s_xz = Σx_t z_t`. **Do not use the uncentred form.** Its numerators cancel catastrophically when `h` is large and nearly constant. Measured against exact rational arithmetic (`xc02_check/c3_2x2.py`, `T = 300`), the uncentred `b_T` had relative error 7e−6 / 6e−6 (φ̂ / γ̂) at `μ = −25, φ = 0.5, σ = 1e−4`, and 7e−5 / 1.6e−4 at `μ = −9, φ = 0.99, σ = 1e−6`. The centred form was exact to ≤ 1.5e−16 in all cases. An inaccurate `b_T` makes the proposal differ from the density used in the MH ratio, so it biases the chain slightly, not just the efficiency.
`chol(B_T) = [[√(A₂₂/det), 0], [−A₁₂/√(det·A₂₂), 1/√A₂₂]]` (verified, relative error ≤ 4e−16).
The naive `cholesky(X'X + B₀⁻¹)` failed in the prototype when `h` was large and nearly constant. Compute `C_T` from squared residuals as in step 3, never as `½(z'z − b_T'X'z)`. The two are algebraically equal (`z'z − b'X'z = ‖z − Xb‖² + b'B₀⁻¹b`), but the second form cancels catastrophically.

### 6.2 One-block sampler (KFS §2.4; Geweke-checked in the cross-check)

`σ²* ~ IG((T−1)/2, C_T^aux)` with `C_T^aux = ½[‖z − Xb_T‖² + b_T'B₀⁻¹b_T]`, then `(γ*, φ*) | σ²* ~ N(b_T, σ²*B_T)`. With `log p_aux(γ,φ|σ²) = −log σ² − (γ²/B₀¹¹ + φ²/B₀²²)/(2σ²)` and `log p_aux(σ²) = −½ log σ²`:
`log R = [ℓ₀(θ*) + log p(γ*|φ*) + log p_φ(φ*) + log p_σ²(σ²*) − log p_aux(γ*,φ*|σ²*) − log p_aux(σ²*)] − [same, old]`. For the Gamma prior, `log p_σ² − log p_aux(σ²) = −σ²/(2B_σ) + const`. KFS report 1- and 2-block perform similarly. (Cross-check: implemented exactly as written, with centred `b_T` and immediate rejection for `|φ*| ≥ 1`, in `xc02_check/c4_geweke.py`, variant `cblock1`, C-only, same settings as §6.3. **It passes Geweke** in 2/2 seeds, max |z| = 2.26 and 1.80.)

### 6.3 Three-block sampler (KFS §2.4; not recommended)

KFS find it "deteriorates due to massive overconditioning". The proposals are `φ ~ N(m_φ, v_φ)` with `m_φ = [Σ_{t=1}^T h_{t−1}h_t − γΣ_{t=0}^{T−1}h_t]/[Σ_{t=0}^{T−1}h_t² + 1/B₀²²]`, `v_φ = σ²/[Σ_{t=0}^{T−1}h_t² + 1/B₀²²]`; and `γ ~ N([Σ_{t=1}^T h_t − φΣ_{t=0}^{T−1}h_t]/(T + 1/B₀¹¹), σ²/(T + 1/B₀¹¹))`. KFS print the `φ` acceptance ratio with `p(h_0|φ, μ, σ²)p(φ)` only. When `γ` is held fixed, `μ = γ/(1−φ)` moves with `φ`, and the correct target also contains `p(γ|φ)`. **Use the derived version**, `log R = ℓ₀(γ/(1−φ*), φ*, σ²) + log p(γ|φ*) + log p_φ(φ*) − old − [log N(φ*;0,σ²B₀²²) − log N(φ;0,σ²B₀²²)]`. This ambiguity in the paper is UNVERIFIED (it may be a typo); it is irrelevant if the 2-block sampler is used. **Cross-check: settled empirically.** In an independent Geweke test (`xc02_check/c4_geweke.py`; `T = 20`, `N(0,1)`, `Beta(20,1.5)`, `B_σ = 0.1`, 3×10⁵ sweeps, C-only three-block, γ held fixed during the φ move), the **derived** ratio passed twice (max |z| = 2.22, 2.16). The ratio **as printed by KFS** (no `p(γ|φ)`) was rejected under Holm both times (max |z| = 10.7 and 11.9, on `φ` and `φ²`). For our purposes the printed three-block φ ratio is wrong; use the derived one. KFS's printed γ ratio, which does contain `p(γ|φ)`, is consistent with the derivation. KFS's own conclusion stands: three-block "deteriorates due to massive overconditioning" (§4.2), and one- and two-block have very similar inefficiency factors.

---

## 7. Step (b\*\*): `θ` in the non-centred parametrisation (KFS §2.5)

NC model: `y*_t − m_{r_t} = μ + σh̃_t + ε_t`, `ε_t ~ N(0, v_{r_t})`; `h̃_t = φh̃_{t−1} + η_t`; `h̃_0 ~ N(0, 1/ω₀)`. The prior of `h̃` is free of `(μ, σ)` for both the stationary and the fixed-`B₀` initial condition.

**φ | h̃** (MH, flat auxiliary prior): propose `φ* ~ N(Σ_{t=1}^T h̃_{t−1}h̃_t / Σ_{t=1}^T h̃²_{t−1},  1/Σ_{t=1}^T h̃²_{t−1})`. Reject if `|φ*| ≥ 1`. Otherwise `log R = log N(h̃_0; 0, 1/ω₀(φ*)) + log p_φ(φ*) − [same at φ]`. For fixed `B₀`, the `h̃_0` term cancels.

**(μ, σ) | h̃, r, y\*** (exact Gibbs, joint 2-block as in KFS eq. 10): weighted regression with `w_t = 1/v_{r_t}`, `X_t = (1, h̃_t)`, `ỹ_t = y*_t − m_{r_t}`, `t = 1..T`, prior `N((b_μ, 0)', diag(B_μ, B_σ))`:
`P = diag(1/B_μ, 1/B_σ) + Σ_t w_t X_t X_t'`, `(μ, σ) ~ N(P⁻¹[(b_μ/B_μ, 0)' + Σ_t w_t X_t ỹ_t], P⁻¹)`. This is exact because `σ² ~ B_σχ²₁` is equivalent to `±σ ~ N(0, B_σ)` (KFS §2.1; Frühwirth-Schnatter & Wagner 2010).

**Sign handling (derived, required).** `σ` is drawn on ℝ. The extended posterior is invariant under `(σ, h̃) ↦ (−σ, −h̃)`, so if `σ < 0`, set `σ ← −σ` **and** `h̃ ← −h̃`. Then `h = μ + σh̃` is unchanged and the state stays valid. Taking `|σ|` **without** flipping `h̃` is not measure preserving, because it reflects `h` about `μ`.
> Observation (**CONFIRMED** in the cross-check, behaviour read from stochvol 3.2.9 source, nothing transcribed): `draw_theta_2block` in `src/utils_parameters.cc` draws `(σ, μ)` jointly from the bivariate Gaussian and returns `std::fabs(sigma_new)`. `update_fast_sv` in `src/single_update.cc` then sets `h = μ + σh̃` with the **unflipped** `h̃`. bayesianVARs and factorstochvol inherit this through `update_fast_sv`.
>
> **Consequence, demonstrated.** A clean-room reimplementation of GIS-C that differs only in this detail (`xc02_check/c4_geweke.py`, variant `abs`; `T = 20`, `μ ~ N(0,1)`, `Beta(20,1.5)`, `B_σ = 0.1`, 3×10⁵ sweeps) **fails Geweke decisively in 3/3 seeds** (max |z| = 16.0, 18.8, 19.2; worst on `μ²`, `σ`, `φ`). The sign-flip version passes in 3/3 seeds (max |z| = 2.16, 1.25, 1.18). Taking `|σ|` without the flip is therefore a real, if usually small, error.
>
> **Size on realistic data** (GIS-C with flip, `T = 300`, `b_μ = 0, B_μ = 100`, `Beta(20,1.5)`, `B_σ = 0.1`, 5,000 sweeps, `xc02_check/c5_negfrac.py`). The table gives the fraction of NC draws with `σ < 0`, which is where the two rules differ:
>
> | True `(μ, φ, σ)` | Fraction of NC draws with `σ < 0` |
> | :--- | ---: |
> | (−1, 0.95, 0.2) | 0 |
> | (−1, 0.99, 0.1) | 0 |
> | (−1, 0.5, 0.5) | 0.24% |
> | (−1, 0.95, 0.05) | 4.2% |
> | (−1, 0.5, 0.05) | 9.1% |
>
> Parity against stochvol or bayesianVARs is meaningful only where this fraction is ≈ 0. Record it as a diagnostic, and exclude near-homoskedastic fixtures (§13.6). Frühwirth-Schnatter & Wagner (2010, accepted manuscript §2.1, eqs. 6–8) state the non-identification of the sign of `±√θ` together with the non-centred state. Their sampler keeps signed values and adds a random sign switch of `(√θ, {µ̃_t})` (§2.4 step (c)). The deterministic fold used here is the measure-preserving equivalent.

**Alternative 3-block NC** (KFS): `μ | · ~ N(b, B)`, `B = 1/(Σ_t w_t + 1/B_μ)`, `b = B(Σ_t w_t(ỹ_t − σh̃_t) + b_μ/B_μ)`; `σ | · ~ N(b, B)`, `B = 1/(Σ_t w_t h̃_t² + 1/B_σ)`, `b = B Σ_t w_t h̃_t(ỹ_t − μ)`.

**IG prior on σ²** (not stochvol's fast path). The NC conditional is no longer Gaussian. Use an independence MH with proposal `(μ*, σ*)` from the regression above, but with a flat prior on `σ` (precision entry `0 + Σw h̃²`). Accept with `log R = log p_ext(σ*) − log p_ext(σ)`, where `p_ext(s) ∝ |s|^{−2c₀−1} exp(−C₀/s²)` is the symmetric extension to ℝ of the density of `σ` implied by `σ² ~ IG(c₀, C₀)`. Then apply the sign flip. This passed Geweke for the RW model (§13.4).

---

## 8. ASIS interweaving (KFS §3, Algorithms 2–3; Yu & Meng 2011)

- **Why.** In C, `h` is a sufficient augmentation for `(μ, σ)`; in NC, `h̃` is ancillary. C fails when `σ` is small (and `φ` small). NC fails when `φ → 1` (spurious-regression-like). Interweaving both is robust over the whole range at little extra cost: KFS §4 report 2.3 s (C) to 2.4 s (GIS-NC) per 1000 draws at `T = 5000`, i.e. **≈ 4%** (corrected from "about 2%"). Yu & Meng bound the rate by `R·√(r_C r_NC)`.
- **GIS-C (default; stochvol `baseline_parameterization = "centered"`, `interweave = TRUE`).** (a) `h` in C; (b) `θ` in C; (b\*) `h̃ = (h − μ)/σ`; (b\*\*) `θ` in NC (stochvol redraws `μ, φ, σ` in both parametrisations, confirmed; KFS say redrawing only `μ, σ` in NC is equivalent in efficiency); (b\*\*\*) `h = μ + σh̃`; (c) `r`. Rotate to `(c)` first inside a composed sweep (§3.2).
- **GIS-NC.** (a) and (c) in NC, parameters in NC then C. KFS find the baseline matters little.
- **Validity.** Every move is an exact Gibbs or MH conditional of the same joint target in one of two parametrisations, linked by a bijection. Sufficiency/ancillarity affects only efficiency. Interweaving is not the same as alternating parametrisations by iteration (KFS §3).

### 8.1 Random-walk special case `φ = 1` (Primiceri 2005, in log std. dev.; CCM 2019 AAM eq. 5, `ln λ_{j,t} = ln λ_{j,t−1} + e_{j,t}`, in log variance)

(Cross-check: the earlier heading also cited "Chan 2023 §2.2". That section has the stationary AR(1) of eq. 5 with `h_{i,1} ~ N(μ_i, σ_i²/(1−φ_i²))` and does not mention the random walk, so the citation was removed.)

`h_t = h_{t−1} + σ η_t`, `h_0 ~ N(m₀, V₀)`. There is no `μ`: it is not identified when `φ = 1`, so this is a separate static configuration.
- **AWOL (C, T+1):** `Q_00 = 1/V₀ + 1/σ²`, `Q_tt = 1/v_{r_t} + 2/σ²` (`1 ≤ t ≤ T−1`), `Q_TT = 1/v_{r_T} + 1/σ²`, off-diagonal `−1/σ²`; `c_0 = m₀/V₀`, `c_t = (y*_t − m_{r_t})/v_{r_t}` (verified against dense).
- **C-step σ²:** with `S = Σ_{t=1}^T (h_t − h_{t−1})²`:
  - Gamma(½, 1/(2B_σ)) prior: independence MH, propose `σ²* ~ IG((T−1)/2, S/2)`, accept with `exp((σ² − σ²*)/(2B_σ))`. This needs `T ≥ 2`: at `T = 1` the proposal shape is 0, which is improper. (Cross-check: re-derived; the target is `(σ²)^{−(T+1)/2} e^{−S/(2σ²)} e^{−σ²/(2B_σ)}`.) Exact alternative: `σ² | h ~ GIG(λ = (1−T)/2, χ = S, ψ = 1/B_σ)`, density `∝ x^{λ−1}exp(−(χ/x + ψx)/2)`; JAX has no GIG sampler.
  - IG(c₀, C₀) prior: exact, `σ² ~ IG(c₀ + T/2, C₀ + S/2)`.
- **NC:** `h_t = h_0 + σh̃_t`, `h̃_0 = 0`, `h̃_t = h̃_{t−1} + η_t`. The NC step draws `(h_0, σ) | h̃, r, y*` by the §7 regression, with `μ ↦ h_0` and prior `N((m₀, 0)', diag(V₀, B_σ))`. Under the IG prior use the §7 MH variant. Then flip sign and map back. This is the non-centred RW of Frühwirth-Schnatter & Wagner (2010). **Verified in the cross-check** against the accepted manuscript (SSOAR, https://ssoar.info/ssoar/bitstream/document/26176/1/ssoar-jecon-2009-1-fruhwirth-schnatter_et_al-stochastic_model_specification_search_for.pdf). Its §2.1 eqs. (6)–(8) write `µ_t = µ_0 + √θ₁ µ̃_t`, with `µ̃_t = µ̃_{t−1} + ω̃_t`, `µ̃_0 = 0`, and the initial value moved into the observation equation. That is exactly `h_t = h_0 + σh̃_t` here. Its §2.3 normal prior on `±√θ` (scaled by `σ_ε²` in their Gaussian model) is the `σ ~ N(0, B_σ)` used here, and its §2.4 step (c) is the random sign switch. The ASIS here interweaves `σ` (C: `σ² | h_{0:T}`) with `(h_0, σ)` (NC); both moves are valid conditionals (Geweke-verified, §13.4).
- **No stochvol oracle:** `stochvol` requires a constant `φ` to lie in `(−1, 1)` (confirmed in its prior validation). Use `bvarsv` (Primiceri TVP-VAR-SV with DNP order, KSC-7, offset 0.001; its code works in **log-variance** units, §8.2) only at VAR level in Phase 5; for the univariate RW block rely on §13.1–13.5.

### 8.2 Prior conventions for the RW case and unit mapping

- **Primiceri (2005)** works in log standard deviations: `log σ_t = log σ_{t−1} + η_t`, `η ~ N(0, W)`. In our units `h = 2 log σ_t`, so **for the paper's prior** `σ²_h = 4W_ii` and `h_0 ~ N(2 log σ̂_OLS, 4)`. bvarsv's `k_sig` scales the `h_0` variance, but bvarsv's units differ from the paper's (next bullet). `bvarsv` defaults are `k_sig = 1`, `k_W = 0.01` (confirmed, manual v1.1 and the `bvar.sv.tvp` signature). **The IW form is now verified**: Primiceri 2005 §4.1 (July 2004 version) has `log σ_0 ~ N(log σ̂_OLS, I_n)` and `W ~ IW(k_W²·4·I_n, 4)` for `n = 3`, with degrees of freedom "one plus the dimension of each matrix" and `k_W = 0.01`. bvarsv generalises this to `W_prmean = k_W²(M+1)·I`, `W_prvar = M+1` (`R/bvarsv.R`, read for constants only). So `W ~ IW(k_W²(1 + n)I, 1 + n)`, and the univariate marginal is `W_ii ~ IG(1, k_W²(1+n)/2)`, i.e. `σ²_h ~ IG(1, 2k_W²(1+n))` in the paper's log-sd units (derived; uses `IW(Ψ, ν)` ⇒ `W_ii ~ IG((ν−n+1)/2, Ψ_ii/2)`). This is very tight.
- **Units: the paper and bvarsv differ (cross-check correction).** Primiceri's paper defines the state as `h_t = log σ_t`, with `y** = 2h_t + e_t` (App. A.3 eqs. 18–19, eq. 7). **bvarsv's code works in log variance.** Its SV step builds `yss = log(y² + 0.001)`, uses `yss − m_KSC + 1.2704 = state + noise`, and sets `σ_t = exp(state/2)`. It puts `sigma_prmean = log(σ̂²_OLS)`, `sigma_prvar = k_sig·I` and `W ~ IW(k_W²(M+1)I, M+1)` directly on that log-variance state (`src/bvarsv.cpp`, `sigmahelper2`, and `R/bvarsv.R`, read for behaviour only). For **parity with bvarsv** use, in our units: `h_0 ~ N(log σ̂²_OLS, k_sig)`, `σ²_h = W_ii`, `W ~ IW(k_W²(n+1)I, n+1)`, i.e. `σ²_h ~ IG(1, k_W²(n+1)/2)`. **Do not** apply the ×4 of the paper's log-sd units. With the defaults, bvarsv's increment variance is therefore 4× smaller in `h` units than the paper's prior.
- **CCM (2016 WP §8.1–8.2, verified in the WP text):** RW log volatilities, in **log-variance** units (no factor 4), with a full innovation covariance `Φ ~ IW(d_Φ·Φ̲, d_Φ)`, `d_Φ = N + 2`, `Φ̲ = I`; initial log volatilities `N(0, 100)`. The 2019 AAM (eqs. 5 and 8) has the same functional form and an "uninformative Normal" for the initial states, but the main text gives **no numeric values**, so the published 2019 values remain **UNVERIFIED**. Implied univariate marginal of the WP prior, derived with the same IW convention: `Φ_jj ~ IG(3/2, (N+2)/2)`, prior mean `N + 2` (22 at `N = 20`). That is extremely diffuse for a vol-of-vol variance, so do not copy it as a default. A full `Φ` couples the equations' volatility shocks and is outside the per-equation block. Diagonal `Φ` is the CCM 2019 footnote-2 alternative (Cogley & Sargent 2005).
- **KSC (1998) σ² prior (verified in the cross-check):** `σ² ~ IG(σ_r/2, S_σ/2)` with `σ_r = 5` and `S_σ = 0.01·σ_r`, i.e. **`IG(2.5, 0.025)`** (shape, scale; prior mean 0.0167). Source: KSC 1997 WP §2.2, eq. (4) and the sentence after it. The same section has `φ`: `Beta(20, 1.5)`, prior mean 0.86.
- **Proposed rai-bvar default for `SV(phi=1)`** (our proposal, not from the literature; calibrate in the Phase 2 pilot): `h_0 ~ N(log ψ_i, 4)`, with `ψ_i` the Minnesota AR(p) residual variance computed inside each window (R9; the variance 4 is the equivalent of Primiceri's paper prior `log σ_0 ~ N(log σ̂, 1)`; bvarsv's code instead implies variance `k_sig = 1` in `h` units, §8.2), and `σ² ~ Gamma(½, 1/(2·0.1))`, i.e. `σ ~ |N(0, 0.1)|` with prior median ≈ 0.21. Offer `σ² ~ IG(c₀, C₀)` for replication.

---

## 9. Optional exact correction of the mixture approximation (DNP Algorithm 3; Stroud et al. 2003; derived here)

Augment with `s` (= `r`) through `q(r_t | h_t, y*_t) = p_k N(y*_t − h_t; m_k, v_k)/g(y*_t − h_t)`, where `g = Σ_k p_k N(·; m_k, v_k)`. Then `Π_t p(y_t|h_t) q(r_t|·)` integrates over `r` to the exact likelihood. With `log w(ξ) := log f(ξ) − log g(ξ)` (`f` = exact `log χ²₁` density, `g` = mixture; use `logsumexp`):

- **(c) `r` step:** unchanged.
- **(a) `h` step:** the AWOL draw `h'` is an independence proposal. Accept with `log α = Σ_{t=1}^T [log w(y*_t − h'_t) − log w(y*_t − h_t)]`; otherwise keep `h`.
- **(b) C step:** unchanged (depends on `h` only).
- **(b\*\*) NC `(μ, σ)` step:** accept the Gaussian draw `(μ', σ')` with `log α = Σ_t [log w(y*_t − μ' − σ'h̃_t) − log w(y*_t − μ − σh̃_t)]`. The `φ` NC step is unchanged. The sign flip does not change `μ + σh̃`.
- **Mean block, `τ`, outliers:** unchanged. They already use the exact likelihood, marginal of `r`.

Prototype acceptance at `T = 300`: `h` 0.96 (φ = 0.95, σ = 0.2), 0.98 (φ = 0.5, σ = 0.05), 0.98 (φ = 0.99, σ = 0.1); NC `(μ, σ)` 0.99–0.995. For comparison, C `(γ, φ)` acceptance was 0.73–0.87 and C `σ²` ≈ 1.0. Cost: two extra `T`-vectors of `log w` (K = 10 `logsumexp`) per sweep. Exactness requires `c = 0`. stochvol's `correct_model_misspecification` offers a related correction ("reweighting after the fact or an online MH step", manual), not used by default. (Cross-check, behaviour from stochvol 3.2.9 source: in the **fast** sampler it is **ex-post sampling-importance-resampling**, not MH. Each stored draw gets the log weight `Σ_t [log f − log g]`. The weights are normalised per chain, and `sample.int(..., replace = TRUE, prob = w)` resamples the `para` and `latent` draws (`R/wrappers.R`; `res$resampled = TRUE`; ESS is not computed afterwards). This is the KSC (1998) / OCSN (2007, WP §2.3) reweighting idea. The R wrapper allows it in the fast sampler only without t-errors and without a regression. The online MH variant belongs to the general sampler.) DNP's Algorithm 3 ratio is printed as App. B.2 eq. (7): Gaussian density of the residuals at the new and old volatilities, divided by the KSC mixture density at the same points. That is the `log w` difference above, up to DNP's 0.001 offset.

(Cross-check re-derivation of the augmented target: `π(h, θ, r | y) ∝ p(y | h) p(h | θ) p(θ) Π_t q(r_t | ξ_t)`, with `q(k | ξ) = p_k N(ξ; m_k, v_k)/g(ξ)`. Summing over `r` gives the exact posterior. The AWOL density is `∝ p(h|θ) Π_t p_{r_t} N(ξ_t; m_{r_t}, v_{r_t})`, so the independence-MH ratio is `Π_t w(ξ′_t)/w(ξ_t)`. This holds as written, and also for the NC `(μ, σ)` move on the sign-extended space followed by the fold (§7). A rejected NC move keeps the `φ` draw, since `φ | h̃` does not involve the likelihood. Independent prototype: `h`-move acceptance at `T = 300` was 0.964 / 0.979 / 0.961 for `(φ, σ) = (0.95, 0.2) / (0.5, 0.05) / (0.99, 0.1)` (`xc02_check/c6_accept.py`), in line with the 0.96–0.98 above.)

---

## 10. Interfaces, defaults and edge cases

### 10.1 VolBlock mapping (SPEC §4.2)

- `init_state(key, T)` → `h_{0:T} = log(ŝ²)`, constant, with `ŝ²` the OLS residual variance (stochvol-like: it initialises `μ` from `log y_t² = μ + ξ_t`, `ξ ~ N(−1.27, 4.934)`, sets all `h_t = μ`, and draws indicators as `r_t = 5`; Hosszejni & Kastner 2021 §4.6). Initialise `φ` and `σ` at the prior means or, for over-dispersed chains, at random draws from a moderately tightened prior. (Cross-check: stochvol 3.2.9's defaults are `φ = 2E[(φ+1)/2] − 1`, `σ = √E[σ²]` (= √B_σ under the Gamma prior), `μ` from H&K eq. (14), all `h_t = μ`, and `init_indicators = 5` from `get_default_fast_sv()`. H&K §4.6 says only "prior mean".)
- `sample(key, state, resid)` → §3.2 steps (c)…(b\*\*\*). Keys are split per (chain, iteration, block, equation, sub-step): `r`, `z` for AWOL, two MH uniforms plus proposal normals for the C step, and the same for NC.
- `variances(state)` → `exp(h_{1:T})`. Weights for the mean block are `exp(−h_{1:T})`, computed directly, never as `1/exp(h)`.
- `simulate_ahead(key, state, H)` → `h_{T+j} = μ + φ(h_{T+j−1} − μ) + σ η_j` (RW: `h_{T+j−1} + σ η_j`), `λ = exp(h)`. This is the source of future-volatility uncertainty in forecast draws (SPEC §7).
- Store `σ` (not `σ²`) and `h_0` (stochvol reports `sigma` and `latent0`) for parity.

### 10.2 Recommended defaults

| Setting | `SV()` default | Source |
| :--- | :--- | :--- |
| `μ` | `N(0, 100²)` | stochvol `priormu = c(0, 100)` (mean, **sd**); bayesianVARs `cholesky_priormu = c(0, 100)` |
| `(φ+1)/2` | `Beta(20, 1.5)` (E φ = 0.860, sd 0.107) | bayesianVARs default; KSC choice (Kastner 2016 §2.2: mean 0.86, sd 0.11). stochvol default is `Beta(5, 1.5)` (E 0.54, sd 0.31) |
| `σ²` | `Gamma(½, rate ½)`, i.e. `B_σ = 1` | stochvol `priorsigma = 1`; bayesianVARs `cholesky_priorsigma2 = c(0.5, 0.5)` (shape, rate). Verified in the cross-check: `specify_prior_sigma` defaults in bayesianVARs 0.1.8 `R/utility_functions.R` are `cholesky_priormu = c(0,100)`, `cholesky_priorphi = c(20, 1.5)`, `cholesky_priorsigma2 = c(0.5, 0.5)`, `cholesky_priorh0 = "stationary"`, `expert_sv_offset = 0` |
| `h_0` | stationary | both |
| offset `c` | 0 (with float floor) | bayesianVARs `expert_sv_offset = 0` |
| mixture | OCSN 10 | stochvol, KFS |
| ASIS | on, GIS-C | stochvol `interweave = TRUE`, `baseline_parameterization = "centered"` |
| C blocking | 2-block; `B₀¹¹ = 1e12` (γ), `B₀²² = 1e8` (φ), as in KFS §2.4, used **consistently** in both the proposal and the MH ratio; σ² independence proposal; φ out-of-range ⇒ reject | stochvol `get_default_fast_sv()` (confirmed: `mh_blocking_steps = 2`, `proposal_intercept_var = 1e12`, `proposal_phi_var = 1e8`, `proposal_sigma2 = "independence"`, `proposal_phi = "immediate acceptance-rejection"`, `init_indicators = 5`). Cross-check notes on the oracles, all proposal-only and all leaving the target unchanged: bayesianVARs passes `(1e−8, 1e−12)` as `(B011inv, B022inv)`, i.e. swapped. In stochvol 3.2.9's centred regression (`regression_aggregates`) the `intercept` precision is added to the `Σh²` (φ) diagonal and the `phi` precision to the `T` (γ) diagonal, while `acceptance_rate_beta` divides by `N(φ; 0, σ²·phi_var)·N(γ; 0, σ²·intercept_var)`. That inconsistency changes `log R` by about `1e−8·Δ(φ² − γ²)/(2σ²)`, which is negligible unless `σ ≲ 1e−4` |
| exact correction | **ON** (recommendation; orchestrator decision) | DNP Alg. 3 |

The `μ` prior is in log-variance units of the data scale. If `rai-bvar` standardises series, `N(0, 100²)` is effectively flat. A window-specific `N(log ψ_i, 10)` is a reasonable optional alternative (R9-compliant).

### 10.3 Edge cases and numerical rules

1. **Exact zero / tiny residuals:** float floor (§2.4); count floor hits; never adapt `c` within a run.
2. **`|φ*| ≥ 1` proposals:** reject via `log p_φ = −∞`. Compute it with `jnp.where(|φ| < 1, …, −inf)` and make sure `log1p(−φ)` of an unselected branch cannot turn the comparison into NaN (old values are always valid).
3. **`σ → 0` (near-homoskedastic):** C mixes badly, ASIS fixes it. Compute AWOL in the NC form (§5.3). The NC regression handles `σ ≈ 0` and its sign flip (§7).
4. **`φ → 1`:** stationary `Var(h_0) = σ²/(1−φ²)` becomes huge. All terms are in log space. NC mixes badly, ASIS fixes it.
5. **Large, nearly constant `h`:** use the stable 2×2 algebra **with centred cross-products** (§6.1). The prototype's naive Cholesky failed here, and the uncentred `b_T` loses digits.
6. **Cholesky pivots:** SPD is guaranteed mathematically (`K_h ≻ 0` for `|φ| < 1`, `σ > 0`, and for RW with `V₀ < ∞`; observation precisions ≥ 1/7.33). A non-positive pivot means NaN or Inf upstream. Report `min(piv)` and the offending `t`, raise outside `jit`, and never jitter (SPEC §8).
7. **Missing observations** (SPEC §5.7): use a mask `m_t ∈ {0,1}`. The observation precision becomes `m_t/v_{r_t}` (and is 0 in `c_t`), the NC regression weights become `m_t w_t`, the `log w` correction sums use the mask, and `r_t` is ignored where `m_t = 0`.
8. **Small `T`** (e.g. a backtest window start): the regressions stay proper (`B₀` regularises C; the NC prior regularises `(μ, σ)`). The NC `φ` proposal needs `Σ h̃²_{t−1} > 0`, which holds almost surely.
9. **Scaled residuals for `t`/outliers:** `y*_t = log e_t² − log τ_t` (t) or `− 2 log o_t` (outliers). Draw `τ`/`o` before `r` (§3.2).
10. **Overflow in weights:** `exp(−h_t)` overflows for `h_t < −709`. Only the wrong step order reached that in the prototype (§3.3). Assert `isfinite` per sweep.
11. **float64:** all of the above (SPEC §8). The prototype ran in float64; the 1e−12 cross-checks would fail in float32.

---

## 11. Cost and performance notes

Per equation per sweep: indicators `O(TK)` (`K = 10`; KFS §2.6: "computationally rather expensive" because of the `T × 10` exponentials), AWOL `2 × (T+1)` scalar scan steps, C-step `O(T)` sums plus a 2×2 solve, NC-step `O(T)` sums plus a 2×2 solve. This is negligible next to the mean block's `O(k³)` at `n = 20`, `p = 4`. The real risk is scan latency: about 600 sequential steps per sweep at `T = 300`. Mitigate with `unroll`, by batching all (chain × equation) series in one `vmap`ed scan, or with the perturb-then-solve alternative (§5.5). Measure in the Phase 2 pilot against the "4 chains × 10,000 draws < 2 min" target.

---

## 12. What differs from the papers / decisions to record in module docstrings (SPEC §2.1)

1. Joint `(T+1)` AWOL system instead of KFS's `T` system plus `h_0 | h_1` (equivalent; verified).
2. Step order rotated to `(c)` first (DNP; equivalent stand-alone, required when composed).
3. NC sign handling via the measure-preserving flip `(σ, h̃) ↦ (−σ, −h̃)`. KFS are silent on it; FS&W (2010) §2.4(c) use the equivalent random sign switch. stochvol **does** differ: it takes `|σ|` without the flip, which is confirmed and fails Geweke (§7).
4. Derived 3-block `φ` acceptance including `p(γ|φ)` (§6.3). The KFS-printed ratio fails Geweke.
5. Stable 2×2 formulas, with **centred** cross-products for `b_T` (cross-check correction, §6.1) and residual-form `C_T`.
6. Optional exact MH correction (§9), recommended ON.
7. Offset default 0 with a float floor, and a documented rule for floor hits (masking suggested, §2.4).
8. RW variant with `(h_0, σ)` ASIS and the proposed priors (§8.1–8.2).

---

## 13. How to test it

### 13.0 Prototype (clean room, written for this note)

`/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/phase2-research/_sv_proto/`:
Independent cross-check scripts, also clean room, are in `…/phase2-research/xc02_check/`: `c1_mixture.py`, `c2_linalg.py`, `c3_2x2.py`, `c4_geweke.py` (variants `flip`, `abs`, `conly`, `cblock1`, `cblock3_kfs`, `cblock3_derived`), `c5_negfrac.py`, `c6_accept.py` and `c7_reg.py` (regression × SV, `approx`/`exact`).
First author's prototype: `svproto.py` (sampler), `test_proto.py` (dense checks + Geweke), `comp.py` (regression × SV, exact correction, stale order, RW Gamma/IG), `intlik.py`, `marg_check.py`, `accept.py`, `stale_probe.py`, `mix_check.py`. Run them with the repo venv: `…/.venv/bin/python -I <script> <dir> <variant>`. They are NumPy, not JAX, and serve as an executable specification and an independent oracle for the JAX kernels.

### 13.1 Deterministic unit tests (every PR)

| ID | Test | Tolerance / prototype result |
| :--- | :--- | :--- |
| T1 | Mixture constants: `Σp = 1`; mean −1.270280; variance 4.933731; KSC-7 mean −1.270399 | 1e−12 vs stored values; 2e−3 vs exact `log χ²₁` moments |
| T2 | AWOL mean `Q⁻¹c` and covariance (columns from unit `z`) vs dense `numpy.linalg.solve`/`inv`, C and NC, stationary, fixed `B₀`, RW, `T = 40` | 3.6e−15 (mean), 5.8e−16 (cov) |
| T3 | `Q̃ = σ²Q`; `μ + σ·mean_NC = mean_C` | 2.4e−15 |
| T4 | `2Σ log l_t = log\|Q\|` vs `slogdet` | 0 (exact to print) |
| T5 | Marginalised `T` system = Schur complement of `(T+1)` system; `h_0\|h_1` formula | 3.6e−15 |
| T6 | Integrated likelihood §5.6 vs dense MVN; also vs a Kalman-filter log-likelihood of the conditionally Gaussian model with time-varying observation variance `v_{r_t}` (statsmodels `MLEModel`/`CFASimulationSmoother`, BSD-3 licensed: usable in CI) | 6e−14 vs MVN |
| T7 | Stable 2×2 vs `numpy.linalg.inv` (typical data), **and** `b_T` vs exact `fractions.Fraction` arithmetic on large, nearly constant `h` (`μ = −25, σ = 1e−4`; `μ = −9, φ = 0.99, σ = 1e−6`; cross-check) | 5e−13; centred `b_T` ≤ 1e−15 relative (the uncentred form fails at 1e−5 to 1e−4) |
| T8 | Indicator logits vs direct computation; categorical sampler χ² goodness-of-fit on 10⁶ draws for a fixed `ξ` | p > 1e−3 |
| T9 | Shapes, dtypes (float64), `vmap`-batch = loop, bitwise reproducibility per seed | exact |

### 13.2 Kernel-invariance tests (every PR, small)

For each MH or Gibbs kernel in isolation, draw the input from its exact full conditional, apply the kernel once, and test that the output has the same distribution (KS / ECDF; repeat with 10⁴ replications):
- `(γ, φ) | h, σ²`: exact conditional on a 2-D grid (quadrature) for `T = 10`.
- `σ² | h, μ, φ`: exact Gamma×likelihood; the 1-D quadrature density equals the GIG above.
- NC `φ | h̃`: 1-D quadrature.
- NC `(μ, σ)`: exactly Gaussian. Test its mean and covariance deterministically, and test that the sign flip leaves the law of `μ + σh̃` unchanged.
- Corrected AWOL (§9): for `T = 3`, compare against the exact `p(h | y, θ)` by importance sampling.

### 13.3 Prior recovery (every PR)

Mask every observation (`m_t = 0`). The chain must sample the prior: `μ ~ N(b_μ, B_μ)`, `(φ+1)/2 ~ Beta`, `σ² ~ B_σχ²₁`, `h | θ` AR(1). Test via CDF-transform to uniforms (this is stochvol's own Geweke-variant unit test: CDF → normal quantiles → Shapiro test; Hosszejni & Kastner 2021 §2.3), using thinned draws and Holm-corrected KS tests.

### 13.4 Geweke (2004) joint-distribution tests (SPEC §10.3)

Configuration used in the prototype: `T = 20`, `b_μ = 0, B_μ = 1, Beta(20, 1.5), B_σ = 0.1`. For compositions, use `Beta(10, 3)` so that prior-predictive `h` stays moderate (see §2.4). Marginal-conditional: 20,000 iid draws. Successive-conditional: 200,000 sweeps; regenerate `y*_t = h_t + ξ_t` with `ξ_t ~` mixture (SV-only, approximate-model space), or `y_t = βx_t + e^{h_t/2}ε_t` (composition). Ten test functions: `μ, φ, σ, σ², μ², φ², h_0, h_T, mean(h), var(h)` (composition: plus `β, β²`). z-statistics use batch-means MCSE (100 batches). Results:

| Variant | max \|z\| | Verdict (Holm, α = 0.05) |
| :--- | ---: | :--- |
| SV, GIS-C, stationary `h_0` (2 runs) | 1.24 / 1.04 | pass |
| SV, GIS-NC | 1.47 | pass |
| SV, C only (no ASIS) | 2.28 | pass |
| SV, GIS-C, fixed `B₀ = 2` (2 runs) | 1.08 / 1.27 | pass |
| RW-SV, Gamma prior, ASIS | 0.94 | pass |
| RW-SV, IG(3, 0.2) prior, ASIS (200k / 400k) | 2.50 / 1.54 | pass / pass |
| regression × SV, DNP order, exact correction | 0.72 | pass |
| regression × SV, DNP order, no correction | 0.83 | pass (approximation error not detectable at this size) |
| regression × SV, **stale** order | — | **diverged** in 48–485 sweeps (5/5 seeds) |

(Cross-check note on the table above: the first author's earlier run `_sv_proto/comp_approx.log`, regression × SV without correction under `Beta(20, 1.5)`, had max |z| = 2.82, `p = 0.0048`. That is just below Holm's first threshold 0.005, so it was a **borderline Holm rejection**, not a pass. The table reports only the later `Beta(10, 3)` run.)

**Independent re-run (cross-check; separate clean-room code `xc02_check/c4_geweke.py`, `c7_reg.py`; `T = 20`, `μ ~ N(0,1)`, `Beta(20,1.5)`, `B_σ = 0.1`, 3×10⁵ sweeps, marginal-conditional 3×10⁴, 100 batch means, 10 functions, Holm α = 0.05):**

| Variant | Runs | max \|z\| per run | Verdict |
| :--- | ---: | :--- | :--- |
| SV, GIS-C with sign flip (§7) | 12 | 0.75–2.17 | 12/12 pass |
| SV, C only | 2 | 1.14, 0.78 | pass |
| SV, C one-block (§6.2) | 2 | 2.26, 1.80 | pass |
| SV, GIS-C with stochvol-style `\|σ\|`, no flip | 3 | 16.0, 18.8, 19.2 | **3/3 reject** |
| SV, C three-block, derived φ ratio (§6.3) | 2 | 2.22, 2.16 | pass |
| SV, C three-block, KFS-printed φ ratio | 2 | 10.7, 11.9 | **2/2 reject** |
| regression × SV, DNP order, no correction (exact `y`-space data) | 9 | 1.14–2.88 | 8/9 pass; seed 39 borderline (2.88, `var h`) |
| regression × SV, DNP order, exact correction (§9) | 9 | 1.17–3.13 | 7/9 pass; seeds 34 (3.13, `μ²`) and 38 (2.85, `φ`) borderline |

The true defects give |z| of 10–19. The composition rejections are borderline, fall on different functions with different signs, and total 3/18. A correct test at nominal 5% familywise would reject at least 3 of 18 with probability ≈ 0.06. They are therefore consistent with noise, or with a batch-means MCSE that is slightly anti-conservative for the more autocorrelated composition chain. **Longer re-runs at 1.2×10⁶ sweeps all pass.** Seeds 34, 38, 50 and 51 with the exact correction gave max |z| = 1.52, 1.43, 1.42 and 1.16; seed 39 without the correction gave 1.79. The borderline 3×10⁵ results were therefore noise and did not grow with run length. **Implementation advice for the CI harness:** compute MCSE with a spectral or overlapping-batch estimator, or with ≥ 30 batches and `t` quantiles. Calibrate the false-rejection rate once by running the correct sampler over ≥ 20 seeds. Fail CI only when a rejection reproduces on a second seed. As in DNP App. D, treat the uncorrected sampler as "approximately correct": no rejection beyond the noise level was found here.

Recommended CI settings: SV-only, `T = 20`, `2×10⁵` sweeps (≈ 50 s in NumPy; far less in JAX) on every PR. Nightly: `T = 100`, and every composition of Phase 2 (`Linear × {SV, SV(outliers), SV(t)}`, structural and reduced form). Run the composition tests with `mixture_correction=True`, which makes the sampler exact. Also run them with `False` and record the `|z|` values, so the size of the approximation error is visible.

### 13.5 SBC (Talts et al. 2018; Modrák et al. 2025)

1000 replications, `T = 100`. Prior as in production but with `B_μ = 1`, so that `exp(h)` stays representable. 4 chains × 1000 kept draws, thinned to about 100 near-independent draws. Rank statistics for `μ, φ, σ, h_0, h_{T/2}, h_T, mean(h)`, plus data-dependent quantities:
- `log p(y | h)` (exact Gaussian; Modrák et al. recommend the log-likelihood),
- `log p(h | θ)`,
- `log p̃(y* | r, θ)` (§5.6).

Use the ECDF-envelope test with multiplicity correction (SPEC §10.3). Generate data in `y`-space with `mixture_correction=True`, or in `y*`-space for the stand-alone block. Otherwise SBC tests the approximate posterior against exact data.

### 13.6 Parity with `stochvol` (SPEC §10.4; fixtures from `tools/make_r_fixtures.R`, not run in CI)

**Parameter mapping (exact):**

| rai-bvar | `svsample()` argument | `specify_priors()` | bayesianVARs `specify_prior_sigma(type="cholesky")` |
| :--- | :--- | :--- | :--- |
| `μ ~ N(b_μ, B_μ)` (variance) | `priormu = c(b_μ, sqrt(B_μ))` | `mu = sv_normal(mean = b_μ, sd = sqrt(B_μ))` | `cholesky_priormu = c(b_μ, sqrt(B_μ))` |
| `(φ+1)/2 ~ Beta(a_φ, b_φ)` | `priorphi = c(a_φ, b_φ)` | `phi = sv_beta(a_φ, b_φ)` | `cholesky_priorphi = c(a_φ, b_φ)` |
| `σ² ~ B_σχ²₁` | `priorsigma = B_σ` (**not** `sqrt`) | `sigma2 = sv_gamma(shape = 0.5, rate = 0.5/B_σ)` (svsample converts exactly so; re-confirmed in `R/wrappers.R`) | `cholesky_priorsigma2 = c(0.5, 0.5/B_σ)` (shape, rate). **Keep shape = 0.5**: stochvol's fast sampler reads only the rate (`B_σ = 0.5/rate`) and ignores the shape, so any other shape is silently treated as ½ (cross-check, `src/utils_parameters.cc`) |
| `σ² ~ IG(c₀, C₀)` | — | `sigma2 = sv_inverse_gamma(shape = c₀, scale = C₀)`; routes to stochvol's *general* sampler (confirmed: `use_fast_sv` requires `sv_gamma` with shape 0.5), so compare posteriors only | not offered (gamma shape/rate only; see the shape caveat above) |
| stationary `h_0` | `priorlatent0 = "stationary"` | `latent0_variance = "stationary"` | `cholesky_priorh0 = "stationary"` |
| `h_0 ~ N(μ, B₀σ²)` | `priorlatent0 = B₀` (svsample help: "`h_0 ~ N(μ, Bσ²)`") | `latent0_variance = sv_constant(B₀)`. Variance multiplier per source (`determine_Bh0inv` returns `1/value`); the `specify_priors` help text "σ²/v" is inconsistent with the code | `cholesky_priorh0 = sqrt(B₀)`. Resolved from source in the cross-check: the R side **squares** the numeric value (`cholesky_priorh0^2`, `R/utility_functions.R`) and passes it to stochvol as `Latent0(Constant(v²))`, i.e. `h_0 ~ N(μ, v²σ²)`. The manual's "sd = value·σ" matches; its "mean 0" does not, since stochvol centres `h_0` at `μ` |
| `c = 0` | automatic unless exact zeros (then `sd(y)/1e4` added) — supply data without zeros | C++ `myoffset` | `expert_sv_offset = 0` |
| ASIS GIS-C, 2-block | defaults (`expert = list(interweave = TRUE)`) | — | uses stochvol `update_fast_sv` |
| outputs | `para` (`mu, phi, sigma`), `latent` (`h_1..h_T`), `latent0` (`h_0`); per-chain `mcmc.list` | | |

The stochvol output structure is confirmed in `man/svsample.Rd` (3.2.9): `para`, `latent` and `latent0` are `mcmc.list` objects. Still check exact column names with `str(res)` in the fixture script. The fast sampler is used only if `ρ = 0`, `μ` is normal or fixed at 0, `φ` has a Beta prior and `σ²` has a Gamma(shape 0.5) prior, **and**, when `correct_model_misspecification = TRUE`, only without t-errors and without a regression (confirmed in the R wrapper, `use_fast_sv`). Our default priors meet these conditions.

**Known oracle deviations (cross-check).** (a) stochvol's NC step takes `|σ|` without flipping `h̃` (§7). Keep only fixtures where rai-bvar's diagnostic "fraction of NC draws with σ < 0" is ≈ 0. That holds for (i) and (iii) below, is 0.24% for (ii), and is 4–9% for `σ_true = 0.05`, which must not be used. (b) The auxiliary `B₀` values are placed inconsistently between stochvol's proposal and ratio, and bayesianVARs swaps them (§10.2). The effect is negligible. (c) For an **exact-posterior** oracle (rai-bvar with `mixture_correction=True`), also run stochvol with `expert = list(correct_model_misspecification = TRUE)`. That is ex-post SIR (§9), so compare quantiles only, with MCSE from the resampled draws inflated by the weight ESS.

**Fixture recipe (R side; run offline):** use datasets generated in NumPy and committed as CSV (no zeros):
(i) `T = 300`, `μ = −1, φ = 0.95, σ = 0.2`;
(ii) `T = 300`, `φ = 0.5, σ = 0.5`;
(iii) `T = 300`, `φ = 0.99, σ = 0.1`;
(iv) standardised AR(4) OLS residuals of a FRED-QD growth series.
Call `svsample(y, draws = 50000, burnin = 5000, n_chains = 4, priormu = c(0, 10), priorphi = c(20, 1.5), priorsigma = 0.1, priorlatent0 = "stationary")`, plus one run at the defaults. Record `packageVersion("stochvol")` (3.2.9 on CRAN as of 2026-03) and R version in the NetCDF attributes. Compare the 5/25/50/75/95% quantiles of `μ, φ, σ` and of `h_t` at `t ∈ {0, 1, T/4, T/2, T}` within 4 MCSE (ArviZ ESS on our side; `posterior::mcse_quantile` or `coda` on theirs). Run with `mixture_correction=False` for strict parity. With it on, differences should still be inside MCSE (test both). Avoid `σ ≈ 0` datasets, or loosen tolerances for them (§7 note).

**bayesianVARs (VAR level, reduced form):** map the SV priors as above. The coefficient-prior mapping belongs in the CCCM note.

### 13.7 Other oracles

- **statsmodels** `CFASimulationSmoother` (BSD-3, McCausland et al. "Cholesky Factor Algorithm") and the Kalman smoother. Conditional on `r`, the model is linear Gaussian, so the smoothed means and variances must equal `Q⁻¹c` and `diag(Q⁻¹)` to 1e−10. This is a CI-friendly Python oracle for step (a). (Cross-check, statsmodels docs: the CFA smoother cites MMP 2011 and CJ 2009. It "does not yet allow diffuse initialization", does not support degenerate covariances, and "produces simulated states only for exactly the observations in the model". The oracle therefore covers `h_{1:T}` only. Initialise it as known, with `h_1 ~ N(μ, σ²/(1−φ²))` (stationary) or `N(μ, σ²(1+φ²B₀))` (fixed `B₀`), put `μ(1−φ)` in the state intercept, and compare against the `h_{1:T}` block of the `(T+1)` system, which equals the marginal T-system of §5.2. Whether the CFA class accepts a time-varying observation variance `v_{r_t}` is **UNVERIFIED**; if not, use the Kalman smoother, which does.)
- **Dense NumPy** for everything at `T ≤ 50` (T2–T7).
- **bvarsv** (GPL, R) for RW-SV inside Primiceri's TVP-VAR (Phase 5): KSC-7 mixture, offset 0.001, DNP order. Its code works in **log-variance** units, unlike the paper, so map priors as in §8.2.

### 13.8 Regression tests worth keeping

- T13: the stale-indicator order must be *rejected*. In the regression × SV toy above, a deliberately mis-ordered sweep must diverge or fail Geweke while the correct order passes. This guards SPEC §4.3's "no block between indicator and log-vol draw".
- T14: GIS-C and GIS-NC give identical posteriors, within MCSE, on fixture (i).
- T15: the inefficiency factor for `μ` and `σ` with ASIS is no worse than min(C, NC) on the KFS grid corners (`φ ∈ {0.5, 0.99}`, `σ ∈ {0.1, 0.5}`). This is a qualitative KFS §4 replication, nightly.
- T16 (cross-check): a deliberately wrong NC fold, `|σ|` without flipping `h̃`, must **fail** Geweke at `T = 20`, `B_σ = 0.1` (expected max |z| > 10), while the flip passes. This guards §7 and also shows that the Geweke harness has power.
- T17 (cross-check): the KFS-printed three-block `φ` ratio must fail Geweke and the derived one must pass (§6.3), if the three-block sampler is ever implemented.

---

## 14. UNVERIFIED items (updated by the cross-check)

**Still UNVERIFIED**

1. CJ (2009) and MMP (2011) equation and section numbers. Both are paywalled; the ANU copy of CJ is embargoed until 2037. The bibliographic details are confirmed by the statsmodels CFA documentation, which cites both, and by the Inderscience listing. The algorithms are confirmed through KFS §2.3, Chan (2023) §4.2.1 and the statsmodels documentation.
2. The numeric prior values of the published CCM (2019). The accepted manuscript gives only the form `Φ ~ IW(d_Φ Φ̲, d_Φ)` and an "uninformative Normal" for the initial states. The 2016 WP values (`d_Φ = N+2`, `Φ̲ = I`, `N(0,100)`) are verified.
3. JAX: float64 CPU performance of `tridiagonal_solve` (§5.5), and the bias threshold of `jax.random.categorical` in float64 under `mode="low"` (§4).
4. Whether statsmodels' `CFASimulationSmoother` accepts a time-varying observation variance (§13.7).
5. The proposed RW defaults (§8.2) and the "correction ON" default (§0.5). These are design proposals, not literature values.

**Resolved in the cross-check** (details in the Cross-check log)

- Primiceri's `W ~ IW(k_W²(1+n)I, 1+n)` and the offset 0.001: verified (Primiceri 2005 §4.1 and App. A.3; DNP App. B.2 eq. 7; bvarsv code). New finding: bvarsv's code works in log-variance units, unlike the paper (§8.2).
- KSC σ² prior: `IG(2.5, 0.025)` (KSC 1997 WP §2.2 eq. 4).
- FS&W (2010) non-centred RW and sign handling: verified (accepted manuscript §2.1, §2.4(c)).
- stochvol: the NC `|σ|` without the `h̃` flip is confirmed in the source, is not compensated elsewhere, and fails Geweke in a clean-room reimplementation (§7). The `svsample` output structure is confirmed (`svsample.Rd`). `correct_model_misspecification` in the fast sampler is ex-post SIR (§9).
- bayesianVARs numeric `cholesky_priorh0`: `h_0 ~ N(μ, v²σ²)` (§13.6).
- KFS three-block `φ` ratio: the printed version fails Geweke; the derived one passes (§6.3).

---

## 15. References (accessed 2026-10-21)

- Kastner, G. & Frühwirth-Schnatter, S. (2014). ASIS for boosting MCMC estimation of SV models. *CSDA* 76, 408–423. arXiv:1706.05280, https://arxiv.org/abs/1706.05280: §1 eqs. 1–4, §2.1 eq. 5 (priors), §2.2 eqs. 6–7 and Algorithm 1, §2.3 (AWOL matrices), §2.4 eqs. 8–9 (C-step 1/2/3-block), §2.5 eq. 10 (NC), §2.6 (indicators), §3 Algorithms 2–3 (GIS-C/GIS-NC), §4 (simulation settings).
- Kastner, G. (2016). Dealing with SV in time series using the R package stochvol. *JSS* 69(5). arXiv:1906.12134, https://arxiv.org/abs/1906.12134: §2.1 eqs. 1–3, §2.2 eq. 4 (priors, Beta moments, KSC `(20, 1.5)`), §2.3, §3.1 (offset `sd(y)/10000`), §3.2 (`priormu` = (mean, sd), e.g. `c(0, 100)`; `priorphi` default `c(5, 1.5)`, prior mean 0.54 and sd 0.31; `priorsigma` = `B_σ`, default 1), §4 (embedding in Gibbs). [Cross-check: the earlier text read "`priormu` … default `c(5, 1.5)`", which mixed up the `priormu` and `priorphi` defaults.]
- Hosszejni, D. & Kastner, G. (2021). Modeling univariate and multivariate SV in R with stochvol and factorstochvol. *JSS* 100(12). arXiv:1906.12123, https://arxiv.org/abs/1906.12123: §2.2 (priors; `σ ~ |N(0, B_σ)|`; `h_0` options), §2.3 (estimation; Geweke variant in unit tests), §4.5 (`specify_priors` defaults; `priormu`, `priorphi`, `priorsigma` defaults `c(0,100)`, `c(5,1.5)`, `1`), §4.6 (initial values, eq. 14; misspecification correction is off by default).
- Omori, Y., Chib, S., Shephard, N. & Nakajima, J. (2007). SV with leverage: fast and efficient likelihood inference. *J. Econometrics* 140, 425–449. WP version (2004): https://www.nuff.ox.ac.uk/economics/papers/2004/W19/mixture28.pdf: eqs. 2–4, Table 1 (K = 10 constants and shifted KSC-7), §3 (offset `c = 10⁻⁴`).
- Kim, S., Shephard, N. & Chib, S. (1998). *Rev. Econ. Stud.* 65(3), 361–393, doi:10.1111/1467-937X.00050. Read in the July 1997 Nuffield WP version, https://www.nuffield.ox.ac.uk/economics/papers/1996/w26/Chib8.pdf: §2.2 eq. (4) (`σ² ~ IG(σ_r/2, S_σ/2)`, `σ_r = 5`, `S_σ = 0.01σ_r`; `Beta(20, 1.5)`), §3.1 eqs. (9)–(10) (offset `c = 0.001`, shift `−1.2704`), Table 4 (7-component mixture, unshifted means). The published RES pagination was not checked against the WP.
- Del Negro, M. & Primiceri, G. (2015). Corrigendum. *Rev. Econ. Stud.* 82(4), 1342–1345; FRBNY Staff Report 619 (May 2013, rev. Oct 2014), PDF https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr619.pdf (verified in the cross-check: downloaded, "May 2013; revised October 2014"): §2 Algorithm 1, §3 Algorithm 2, §4 Algorithm 3, App. B.1–B.2 (B.2 eq. 7: Algorithm 3 acceptance ratio), App. D (Geweke tests).
- Chan, J. C. C. (2023). BVARs and stochastic volatility. arXiv:2310.14438, https://arxiv.org/abs/2310.14438: §2.2 eqs. 4–5, §4.2.1 and footnotes 8–9 (DNP ordering; per-series pairing).
- Chan, J. C. C. & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *IJMMNO* 1(1/2), 101–120 (not accessed; bibliographic details confirmed via the statsmodels CFA documentation and the Inderscience listing).
- McCausland, W., Miller, S. & Pelletier, D. (2011). Simulation smoothing for state-space models: a computational efficiency analysis. *CSDA* 55(1), 199–212 (not accessed; details confirmed via the statsmodels CFA documentation).
- Rue, H. (2001). Fast sampling of Gaussian Markov random fields. *JRSS-B* 63(2), 325–338 (cited by KFS §2.3).
- Yu, Y. & Meng, X.-L. (2011). To center or not to center: that is not the question. *JCGS* 20(3), 531–570.
- Frühwirth-Schnatter, S. & Wagner, H. (2010). Stochastic model specification search for Gaussian and partial non-Gaussian state space models. *J. Econometrics* 154(1), 85–100, doi:10.1016/j.jeconom.2009.07.003. Accepted manuscript (cross-check): https://ssoar.info/ssoar/bitstream/document/26176/1/ssoar-jecon-2009-1-fruhwirth-schnatter_et_al-stochastic_model_specification_search_for.pdf: §2.1 eqs. (6)–(8) (non-centred RW, sign non-identification), §2.3 (normal prior on `±√θ`), §2.4 step (c) (random sign switch).
- Stroud, J., Müller, P. & Polson, N. (2003). Nonlinear state-space models with state-dependent variances. *JASA* 98, 377–386 (via DNP §4).
- Carriero, A., Clark, T. & Marcellino, M. (2016 WP 16-17), Cleveland Fed, https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2016/wp1617.pdf: eq. 5, §2.2, §3.2.1 (steps 1–2d), Appendix §8.1–8.2 (priors, offset). Corrigendum: doi:10.26509/frbc-wp-201617corrigendum.
- stochvol 3.2.9 manual: https://search.r-project.org/CRAN/refmans/stochvol/html/svsample.html and https://ftp.fau.de/cran/web/packages/stochvol/refman/stochvol.html. Source, read only to confirm behaviour: https://github.com/gregorkastner/stochvol (`R/wrappers.R`: `get_default_fast_sv`, offset rule, fast-sampler conditions, `priorsigma` conversion; `R/util.R`: `specify_priors`, constant-`φ` range; `src/single_update.cc`: update order; `src/sampling_latent_states.cc`: `h_0` row; `src/utils_parameters.cc`: `latent0` variance multiplier, `fabs(σ)` in `draw_theta_2block`, `regression_aggregates`/`acceptance_rate_beta`, shape-agnostic Gamma rate; `src/utils_latent_states.h`: mixture constants; `man/svsample.Rd`: outputs). Cross-check: the CRAN tarball https://cran.r-project.org/src/contrib/stochvol_3.2.9.tar.gz, published 2026-03-07, was read for the same purpose only.
- bayesianVARs 0.1.8 manual (`specify_prior_sigma`; `bvar` details: CCCM triangular algorithm with stochvol `update_fast_sv`), CRAN, published 2026-02-19. Source, read only for behaviour and constants (https://cran.r-project.org/src/contrib/bayesianVARs_0.1.8.tar.gz): `R/utility_functions.R` (defaults; `cholesky_priorh0` squared), `src/bvar_cpp.cpp` (`ExpertSpec_FastSV` with `B011inv = 1e−8`, `B022inv = 1e−12`, 2 MH blocks, centred baseline, interweaving).
- bvarsv 1.1 manual (`bvar.sv.tvp` defaults; DNP corrigendum incorporated), CRAN. Source, read only for behaviour and constants: `R/bvarsv.R` (`k_W`, `k_sig`, `Wprior = M+1`, `sigma_prmean = log σ̂²`) and `src/bvarsv.cpp` (`sigmahelper2`: offset 0.001, log-variance state).
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *Rev. Econ. Stud.* 72(3), 821–852. Read in the July 2004 version: eq. (7) (`log σ_t` random walk), §4.1 (priors: `log σ_0 ~ N(log σ̂_OLS, I_n)`, `W ~ IW(k_W²·4·I_n, 4)`, `k_W = 0.01`), App. A.3 eqs. (18)–(19) (`y** = 2h + e`, `c̄ = 0.001`).
- Carriero, A., Clark, T. & Marcellino, M. (2019). Large Bayesian VARs with stochastic volatility and non-conjugate priors. *J. Econometrics* 212(1), 137–154, doi:10.1016/j.jeconom.2019.04.024. Read in the accepted manuscript from the Università di Bologna institutional archive: eq. (5) (RW log variance), eq. (8) (`Φ ~ IW(d_Φ Φ̲, d_Φ)`), footnote 2.
- JAX documentation (latest, accessed 2026-10-21): `jax.lax.linalg.tridiagonal_solve`, https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html; `jax.random.categorical`, https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html.
- statsmodels `CFASimulationSmoother`: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.cfa_simulation_smoother.CFASimulationSmoother.html (limitations quoted in §13.7).

---

## Cross-check log (second agent, 2026-10-21)

**Method.** Primary sources were downloaded independently into `phase2-research/xc02_dl/`: KFS arXiv:1706.05280, Kastner arXiv:1906.12134, Hosszejni & Kastner arXiv:1906.12123, the OCSN 2004 WP, DNP SR 619, Chan arXiv:2310.14438 and the FS&W 2010 accepted manuscript. Copies already in the scratchpad were used for the KSC 1997 WP (its md5 is identical to the Nuffield URL), the Primiceri 2005 July-2004 version, CCM 2016 WP, the CCM 2019 AAM and Chan's MIT code. GPL packages (stochvol 3.2.9, bayesianVARs 0.1.8, bvarsv 1.1) were read **only** for constants and behaviour, in `phase2-research/xc02_gpl_readonly/` and `dl_gpl_constants_only/`; nothing was transcribed into the note. All numerical checks use new, independent scripts in `phase2-research/xc02_check/` (`c1_mixture.py`, `c2_linalg.py`, `c3_2x2.py`, `c4_geweke.py`, `c5_negfrac.py`, `c6_accept.py`, `c7_reg.py`), run with `…/rai-bvar/.venv/bin/python -I`.

**Verified without change.** Each item lists its source or check.
- The OCSN-10 table (OCSN WP Table 1, plus stochvol `utils_latent_states.h`). The KSC-7 table (OCSN Table 1 left block, KSC WP Table 4 and Chan's `ksc_ar1_mean.m`). All mixture moments and density errors in §2.3 were reproduced to the printed digits.
- All AWOL entries of §5.2 (stationary and fixed `B₀`), §5.4 (fixed-scale `h_0`) and §8.1 (RW). Checked against dense `H'S⁻¹H + G'V⁻¹G`, with maximum absolute error ≤ 7e−15. The KFS §2.3 printed `T`-system matches.
- The NC identity `Q̃ = σ²Q` and the NC mean map (≤ 6e−15). The Schur complement `Ω₁₁`, `c₁` and `h_0 | h_1` for general `B₀` (≤ 8e−15).
- The banded forward and back substitution, mean, covariance and `log|Q| = 2Σ log l_t`.
- The §5.6 integrated likelihood against a dense MVN log density: 0 (stationary) and 9e−14 (`B₀ = 2`).
- Priors: `σ² ~ B_σχ²₁ = G(½, 1/(2B_σ)) ⇔ ±σ ~ N(0, B_σ)` (KFS §2.1; Kastner 2016 §2.2; H&K §2.2). Beta moments (Kastner 2016 §2.2): `Beta(20,1.5)` gives mean 0.8605 and sd 0.1074, `Beta(5,1.5)` gives 0.538 and 0.308.
- stochvol: `priormu = (mean, sd)`; `priorsigma = B_σ` (`rate = 0.5/priorsigma` in `R/wrappers.R`); `priorlatent0 = B` ⇒ `N(μ, Bσ²)`; the offset rule; `get_default_fast_sv()` values; update order (c)(a)(b); constant `φ ∈ (−1,1)`; C++ `myoffset = 0`.
- KFS formulas: eq. 8, the 1-block `c_T = (T−1)/2`, the 2-block `c_T = T/2` with `C_T` including `(1−φ²)(h_0−μ)²`, the `σ²` acceptance ratio, the NC `φ` proposal and ratio, and the NC regression eq. 10 and its 3-block variant.
- DNP Algorithms 1–3 and App. B; Chan 2023 §4.2.1 and footnotes 8–9; the H&K §2.3 Geweke-variant unit test; the bayesianVARs `specify_prior_sigma` defaults.

**Corrections and additions**

| # | Where | What changed | Source / evidence |
| ---: | :--- | :--- | :--- |
| 1 | §6.1 | The `b_T` formula with uncentred sums loses up to ~5 significant digits (relative error 6e−6 to 1.6e−4) when `h` is large and nearly constant, and that biases the MH chain. **Replaced** by a centred cross-product form, exact to ≤ 1.5e−16. The earlier claim "verified to 5e−13" held only for typical data | `c3_2x2.py`, against exact `fractions.Fraction` arithmetic |
| 2 | §7 | stochvol's NC `\|σ\|` without flipping `h̃` moved from UNVERIFIED to **confirmed**: `draw_theta_2block` returns `fabs(sigma_new)` and `update_fast_sv` maps back with the unflipped `h̃`. It is also **demonstrated non-invariant**: a clean-room reimplementation fails Geweke with max \|z\| 16–19 (3/3 seeds), while the flip passes. Added the table of how often `σ < 0` on realistic data | stochvol 3.2.9 `src/utils_parameters.cc`, `src/single_update.cc` (behaviour only); `c4_geweke.py` (`abs`/`flip`); `c5_negfrac.py` |
| 3 | §6.3 | The KFS three-block `φ` ratio moved from UNVERIFIED to **settled**: as printed (no `p(γ\|φ)`) it fails Geweke (max \|z\| 10.7, 11.9); the derived ratio passes (2.22, 2.16) | `c4_geweke.py` (`cblock3_kfs`/`cblock3_derived`); KFS §2.4 |
| 4 | §6.1 | Caveat added: re-proposing until `\|φ*\| < 1` is valid in the 2-block sampler but needs a `Z(σ²)` correction in the 1-block sampler | derivation |
| 5 | §6.2 | The one-block sampler was "not prototyped". It now **passes Geweke** in 2/2 seeds (max \|z\| 2.26, 1.80), so its formulas are confirmed | `c4_geweke.py` (`cblock1`) |
| 6 | §8 | ASIS overhead "about 2%" corrected to **≈ 4%** (2.3 s vs 2.4 s per 1000 draws at `T = 5000`) | KFS §4 |
| 7 | §8.1 | Removed the citation "Chan 2023 §2.2 notes [RW] as the common variant": that section has the stationary AR(1), eq. 5. Added CCM 2019 AAM eq. 5 (RW in log variance) | Chan arXiv:2310.14438 §2.2; CCM 2019 AAM |
| 8 | §8.1 | FS&W 2010 NC random walk and sign handling moved from UNVERIFIED to **verified** | FS&W accepted manuscript §2.1 eqs. 6–8, §2.3, §2.4(c) |
| 9 | §8.1 | The RW Gamma-prior `σ²` proposal needs `T ≥ 2` | derivation |
| 10 | §8.2 | Primiceri's `W ~ IW(k_W²(1+n)I, 1+n)`, `k_W = 0.01`, `log σ_0 ~ N(log σ̂, I)`: **verified** | Primiceri 2005 §4.1; bvarsv `R/bvarsv.R` |
| 11 | §8.2, §8.1, §13.7, §1 | **Unit correction:** bvarsv's code works in **log variance**, not log-sd as the note had said. Its `h_0 ~ N(log σ̂², k_sig)` and `W` sit directly on `h`, so for parity there is no ×4 | bvarsv `src/bvarsv.cpp` `sigmahelper2` (`σ_t = exp(state/2)`, `yss − m + 1.2704 = state`); Primiceri App. A.3 eqs. 18–19 for the paper's log-sd units |
| 12 | §8.2 | CCM: log-variance units stated. The 2019 AAM gives only the functional form (values still UNVERIFIED). Added the derived implied marginal `IG(3/2, (N+2)/2)` of the WP prior as a caution | CCM 2016 WP §8.1–8.2; CCM 2019 AAM eqs. 5, 8 and footnote 2 |
| 13 | §8.2 | The KSC σ² prior moved from UNVERIFIED to **verified**: `IG(2.5, 0.025)` | KSC 1997 WP §2.2 eq. (4) |
| 14 | §2.3 | KSC-7 is now verified against KSC itself and a third source. Added KSC-7 skewness and kurtosis (−1.5352, 4.0006), density error (1.03e−2) and the 1st–99th-percentile \|log f/g\| (OCSN 0.0123, KSC 0.473) | KSC WP Table 4; `c1_mixture.py` |
| 15 | §2.4 | The KSC `c = 0.001` was verified directly. The Primiceri `c̄ = 0.001` moved from UNVERIFIED to **verified**. CCM 2016 WP gives **no** value. Added a caution about the size of a float-floor hit (`y* ≈ −708`) and the masking alternative | KSC WP §3.1; Primiceri App. A.3; DNP App. B.2 eq. (7); bvarsv C++ |
| 16 | §2.1 | Added that the KSC and Chan convention (stationary `h_1`, no `h_0`) is the same model as KFS under the stationary prior; the IG shape differs by ½ | KSC WP §2.2; Chan eq. 5; derivation |
| 17 | §3.1 | DNP algorithms restated precisely; App. B.2 eq. (7) and the App. D Geweke results added; DNP PDF URL verified | DNP SR 619 (May 2013, rev. Oct 2014) |
| 18 | §4, §11 | "dominates" corrected to KFS's "computationally rather expensive". Added the JAX `categorical` `mode` caveat | KFS §2.6; JAX docs |
| 19 | §5.5 | Added the JAX `tridiagonal_solve` signature and conventions (`dl[0] = du[m−1] = 0`, partially pivoted LU) | JAX docs |
| 20 | §9 | The algorithm behind stochvol's `correct_model_misspecification` moved from UNVERIFIED to **known**: in the fast sampler it is ex-post SIR, not MH. Added DNP eq. (7) and a re-derivation; independent acceptance rates were 0.961–0.979 | stochvol `R/wrappers.R`, `src/sampling_main.cc` (behaviour); `c6_accept.py` |
| 21 | §10.1 | Recorded stochvol's actual starting values | stochvol `R/wrappers.R` |
| 22 | §1, §10.2 | `B₀¹¹` is γ and `B₀²²` is φ. bayesianVARs passes them swapped, and stochvol places them inconsistently between proposal and ratio (effect negligible). The bayesianVARs defaults were verified from source | bayesianVARs `src/bvar_cpp.cpp`; stochvol `regression_aggregates` and `acceptance_rate_beta` (behaviour) |
| 23 | §13.6 | Mapping fixes: (a) bayesianVARs numeric `cholesky_priorh0` moved from UNVERIFIED to **resolved**: the value is squared, giving `h_0 ~ N(μ, v²σ²)`, so map `cholesky_priorh0 = sqrt(B₀)`; (b) `cholesky_priorsigma2` must keep shape 0.5, because the fast sampler ignores the shape; (c) the fast-sampler condition was completed with the misspecification-correction restriction; (d) the stochvol output structure moved from UNVERIFIED to confirmed; (e) added an oracle-deviation paragraph and the SIR exact-posterior oracle | bayesianVARs `R/utility_functions.R`; stochvol `src/utils_parameters.cc`, `R/wrappers.R`, `man/svsample.Rd` |
| 24 | §13.4 | The first author's `comp_approx.log` (max \|z\| 2.82, `p = 0.0048`) was a **borderline Holm rejection**, not a pass. Added an independent Geweke table: 41 runs at 3×10⁵ sweeps, plus 5 re-runs at 1.2×10⁶ sweeps of the borderline or new seeds, all passing with max \|z\| ≤ 1.79. Added harness advice on anti-conservative batch-means MCSE | `_sv_proto/comp_approx.log`; `c4_geweke.py`, `c7_reg.py` |
| 25 | §13.7 | statsmodels CFA limitations: no diffuse initialisation, and states only at observation times, so compare `h_{1:T}`. Support for a time-varying observation variance is UNVERIFIED | statsmodels documentation |
| 26 | §15 | Kastner 2016 §3.2 entry fixed: it had mixed up the `priormu` and `priorphi` defaults. Added DOIs and URLs (KSC WP, DNP PDF, FS&W AAM, Primiceri, CCM 2019, JAX, the source tarballs). Added the CJ and MMP titles | sources above |
| 27 | §14 | Rewritten: 7 items resolved, 5 remain | — |
