# Paper notes: findings worth writing up

A running ledger of non-obvious results found while building `rai-bvar`, kept for a methods /
software paper. Each entry says what was found, where the evidence lives in the repository, and
whether the claim has been verified enough to publish. Add an entry when a test, a fixture or a
review turns up something a reader would not expect; never delete one, mark it superseded.

Status: **verified** (reproducible from a committed test or fixture), **recorded** (numbers from a
run that is written down but not re-run independently), **to check** (must be re-run or
re-derived before it goes into the paper).

## A. Findings about other software (handle with care: re-check against current versions)

1. **stochvol's ASIS step is biased in small samples (D-7).** stochvol takes `|σ|` in the
   non-centred step without flipping `h̃`; on a four-observation problem its `h_2` quantiles
   deviate from an exact importance-sampling posterior, while the centred-only sampler and ours
   (which folds `(σ, h̃) ↦ (−σ, −h̃)`) match. Evidence: `tests/parity/SV_PARAMETRIZATION.md` §3.1.
   Status: recorded (stochvol 3.2.1). To check against the current stochvol release before
   publication.
2. **stochvol 3.2.1's Student-t SV sampler misses the standardized-t posterior.** Our sampler
   matches the exact posterior; stochvol's `μ` and `τ` quantiles do not, with or without ASIS.
   Evidence: `SV_PARAMETRIZATION.md` §3.2; `test_sv_t_matches_the_exact_importance_sampling_posterior`.
   Status: recorded for 3.2.1 only; the unblock path in §3.2 must be run before any public claim.
   Report upstream first.

## B. Validation methodology (the core of a methods paper)

3. **Exact-mode SV is exact for any mixture table and any indicator law,** so mutants that corrupt
   the table or the indicator pmf (M11, M12) are not bugs there; they are bugs of the
   auxiliary-mixture sampler only. Likewise the uncorrected non-centred move (M7) is invisible with
   the 10-component table at test sizes and becomes detectable with a deliberately coarse
   one-component table. Evidence: `tests/mutation/mutants.py` module docstring and `EVIDENCE`.
   Status: verified (mutation suite re-run on x86-64 Linux, 2026-10-22: M11/M12 caught under
   `exact=False`, M7 caught with `COARSE1`).
4. **Parameter ranks are blind to the CCM-2019 triangular-step error; the log-likelihood rank is
   not.** Whole-scan one-block invariance does not catch M1 at N = 1000, block-conditional SBC of
   one column with the log-likelihood as a statistic does. Evidence: `mutants.py` (M1 evidence),
   N06 §7.3. Status: block SBC catch verified (log-likelihood p 2.8e-11 vs control min p 0.10);
   the whole-scan "not caught" observation is recorded, not re-run.
5. **Per-replication SBC ESS cannot be certified from L = 99 kept draws.** The design was amended to
   predict per-replication ESS from the pilot IAT and add a pooled kept-draw ESS gate. Evidence:
   DESIGN §9.6 erratum. Status: verified (design change; the reasoning is in the erratum).
6. **SBC thinning cap.** With `t_max = 50` the ESS rule failed on slow-mixing datasets (outlier/t
   rows) while the γ test was clean; `t_max = 200` covers every replication observed. Evidence:
   `tests/sbc/test_sbc_compositions.py` docstring. Status: recorded.
7. **Geweke "error 2" (simulator draws fresh latents)** is caught by Geweke tests (M3, max |z|
   23.5 vs control 2.59). Status: verified.
8. **Every mutant must be caught by some tier** (DESIGN §9.7): the table of mutant × catching tier
   with statistics is a natural paper table. Status: verified, 33/33 mutant × tier checks caught
   with passing controls (`tests/mutation`, ~10 min with a warm compilation cache). Re-run on
   the release commit for the paper table (`RAI_BVAR_VALIDATION_RECORD=<dir>` writes the JSON).

8b. **The path-conditioned RB log score breaks down in the tails at h > 1.** SPEC §7's estimator
   averages p(y_{T+h} | θ_s, simulated path_s up to T+h−1). For outcomes far in the predictive's tail
   (US unemployment 2008–10 at h = 8) one draw carries all the weight: Constant's UR h = 8 score at
   2008Q2 was −93.6 (1000 draws), −46 to −63 (8000), −36 (64,000) against −19.1 for the exact
   path-integrated mixture, while the delta-method MCSE stayed at ≈ 1. It inflated the quick
   Clark–Ravazzolo UR h = 8 ΔLPS about 8× (+1.92 vs ≈ +0.23 converged). A methods point worth a
   section: integrate the path out (Gaussian given θ and the volatility path) instead of conditioning
   on it. Evidence: WP23 review (2026-10-23), `benchmarks/clark_ravazzolo.py` RELIABILITY_NOTE.
   Status: recorded (review computations; re-run with the library fix before publishing).

## C. Engineering lessons

9. **Bitwise reproducibility has limits.** Results are bitwise identical across chunk sizes,
   burn-in piece sizes, fused/unfused plans and `chains='map'`, but not across `vmap` batch sizes,
   platforms (OpenBLAS vs Accelerate) or `shard_map`; those are compared at rtol ≈ 1e-12.
   Evidence: tests under `tests/mcmc/` (invariance tests). Status: verified in CI (Linux).
10. **Float64 failure in the prior's extreme tail.** One Geweke chain out of 4000 reached `h₀ ≈ 30`
    where `|y| ≈ 1e8` made a Cholesky pivot zero; the engine reports it (no silent jitter).
    Evidence: `tests/geweke/test_geweke_compositions.py` docstring. Status: recorded.
    **Resolved by a QR draw (2026-10-23):** at the failing state (chain 3214, sweep 194; equation 0
    at `h ≈ 31–37`) equation 2's `a`-row precision has exact eigenvalues 14.8 and 1.9e17
    (`cond = 1.3e16`), so forming `ZᵀWZ` loses the small one; the posterior itself is benign (mean
    (−0.17, −0.66), sd (0.20, 0.16)). Householder QR of `[diag(√prec); √w ⊙ Z]` (`R` = the upper
    Cholesky factor of `K`, so the same noise map) gets it to 1.6e-8 sd and agrees with the
    Cholesky draw to ≤ 3e-14 elsewhere. The alternative — narrowing the test prior's `h` tail
    (`(φ+1)/2 ~ Beta(10, 3)` gives `P(max|h| > 30) < 5e-7` → 0 in 2e6 draws, from 5e-7) — would have
    hidden a real numerical weakness and dropped high-persistence coverage. Status: verified
    (`tests/mcmc/blocks/test_contemp_lstsq.py`).
11. **The design-review errata (E1–E28)** in DESIGN §0.2 document what independent reviewers caught
    in the proposals before implementation. Status: verified (in repository).

## D. SOC/SUR dummy rows on the MCMC engine (WP25)

12. **In conjugate (Mode II) structural form the λ step must count the dummy rows.** The rows are
    weighted `1/λ_i` like data rows, so `λ_i | ·` gains `m/2` in its shape and the rows' λ-free
    squared residuals in its scale (`IG(c + (T + q_i + m)/2, …)`); this is what makes the MCMC
    posterior the NIW-plus-dummies posterior. Omitting it is easy to do (the coefficient step
    alone looks right) and it is not caught by the dense oracles of the coefficient step; the
    MCMC-vs-Phase-1 twin catches it at once (33 of 165 quantiles outside 4 MCSE, n = 3, p = 2,
    T = 60). Evidence: `test_a1_dummies_*` in `tests/analytic/test_mcmc_twin.py` (the omission
    was injected by hand; not a registered mutant, because the coupling is not an overridable
    internal). Status: recorded.
13. **The dummy-augmented Mode II prior has a closed-form λ marginal,** `λ_i | d ~ IG(c + m/2,
    C + ½ r'(F_d P̃⁻¹F_dᵀ + I/w)⁻¹r)` with `r = y_d − F_dμ₀`, so the Geweke reference can sample
    the prior exactly (integrate θ, then condition θ on the rows by Matheron's rule). Under
    `iw_equivalent` (only `a_i` λ-scaled, rows with fixed weight) `p(d | λ) = N(r; 0, S₀ + λUUᵀ)`
    is not inverse-gamma, but the IG base prior is an exact rejection envelope: `|S(λ)| ≥ |S₀|`
    and `q(λ) = r'S(λ)⁻¹r ≥ q∞ = min_u (r − Uu)'S₀⁻¹(r − Uu)` (the GLS residual), so
    `g(λ) = (|S₀|/|S(λ)|)^{½}exp(−½(q − q∞)) ≤ 1`; acceptance 0.6–1.0 on the test rows. The log
    prior is exact up to the (intractable) evidence. Evidence: `validation/dummies.py`,
    `tests/validation/test_reference_dummies.py` (KS against the quadrature CDF), Geweke row 205
    (G-small max |z| 2.24, G-full 2.76), SBC row 205 (min p 0.018). Status: verified.
14. **Dummy rows break the diagonal prior precision, but the O(q²) eigen fast path survives**: with
    `P = L_P L_Pᵀ` dense, `K = L_P(I + L_P⁻¹GL_P⁻ᵀ/λ)L_Pᵀ`, one per-fit eigendecomposition again
    gives every λ in O(q²), and storing `L_P⁻ᵀQ` keeps the per-sweep draw code unchanged. At
    `n = 20, p = 4, T = 300` (Mode I × Constant) it is 0.46 ms/sweep against 1.53 for the
    per-sweep Cholesky; the two agree within DESIGN §9.3's rounding envelope. Mode II keeps its
    fixed-factor path (rows λ-free like the data). Status: verified (`test_dense_eigen_path_*`;
    Geweke G-small over 7 seeds and G-full on the eigen path; timing on an Apple M5 Max under a
    concurrent load of 3–5).
15. **A "wrong weight" dummy bug is loud.** Weighting Mode I rows by the current `1/λ_i` instead of
    the fixed `1/σ̄_i²` (N05 §7.7's warning; M17) gives Geweke max |z| 23 (control 2.4); dropping
    the rows from the CCCM step (M18) gives 100 (control 2.6), at 3–6 s per G-small run. The
    reference is written in covariance form (`m × m` systems), the kernels in precision form
    (`q × q` Gram updates), which keeps the two implementations algebraically distinct. Status:
    verified (`tests/mutation`; Apple M5 Max and, independently, x86-64 Linux: M17 U 0.0602, G
    23.15; M18 U 0.405, G 99.7). The independent x86 review also caught four further injected
    bugs (Mode II rows weighted 1/σ̄², rows counted twice in the λ step, CCCM rows weighted by
    `(Σ⁻¹)_jj`, fixed-factor weight 2w).
16. **Adding the feature left every existing composition bitwise unchanged.** With
    `MeanStatic.n_dummy = 0` no dummy code is traced; 537 arrays (prior draws, simulated data,
    caches, three production sweeps of all 17 compositions and 7 block configurations, and 7
    end-to-end EB fits with forecasts) were identical bit for bit before and after the change on
    one machine. Evidence: WP25 report; `test_dummy_path_is_closed_without_dummies`. Status:
    recorded (Apple M5 Max, one run).

## E. Numerical robustness found by the Clark–Ravazzolo replication (WP25 queue item 7)

17. **A legitimate state the scaled RW AWOL draw could not represent.** In the Clark–Ravazzolo
    ordering stage (SV(rw), reduced form, origin 1987Q2, ordering GDP, TBILL, INFL, UR) GDP's
    volatility reached `σ = 4.98e-9` at burn-in sweep 318. The non-centred prior `σ ~ ±N(0, B_σ)`
    has positive density at 0, so the state is valid, and the NC move of the same sweep took σ
    back to 0.031. But in the scaled system `Q̃ = σ²Q` the observation terms `σ²D ≈ 1e-17` are
    below the ulp of the Laplacian's diagonal (2): the assembled matrix is the singular path
    Laplacian, and the level direction's pivot (exactly `≈ σ²(1/V₀ + ΣD)`, of order 1e-15) came
    out 0 (status 2001; the run aborted). Carrying the pivots as their excess over the next edge
    (`q_t = δ_t + q_{t−1}/(1 + q_{t−1})`, all terms ≥ 0) is the same factorisation without the
    cancellation: at ordinary σ it agrees with the old draw to ~1e-13 and every non-RW composition
    stays bitwise unchanged. The trigger is the σ → 0 region where the centred σ² step mixes
    slowly and only the NC move escapes; the fix removes the crash, not the slow mixing.
    Evidence: `tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma` (60-digit
    reference) and `::test_rw_step_at_a_tiny_sigma_is_ok`. Status: verified (Apple M5 Max).

## Before submission

- Re-run every "recorded" / "to check" item on the release commit and cite the commit hash.
- Disclose AI assistance as the target venue requires (most venues do not accept an AI system as
  an author; they require a statement of how it was used).
- Third-party claims (section A): re-test on current versions and contact the maintainers first.
