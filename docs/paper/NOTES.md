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
   **Row 18 (reduced × RW SV, the Clark–Ravazzolo primary SV arm; added 2026-10-23)** failed the
   same rule at 200 by one replication (987 of 997 at ≥ 0.8 L; pilot IATs up to 319, γ clean, min
   p 0.019 over 59 quantities) and passes at a per-row cap of 400 (ESS rule 100%, same min p).
   That pass rests on the rule's blind spot: with a 4000-draw pilot the same run fails the ESS rule
   alone (98.4%; pilot IATs up to 1438, γ and pooled-ESS gates clean). The slow quantities are the
   reduced-form coefficients (intercepts most often), not σ, and in half of the 20 slowest chains
   the IAT estimate still grows at 16000 sweeps (up to 4376): about 1–2% of prior-drawn T = 50
   datasets mix very slowly under reduced × RW SV. The pre-registered pilot (1000) is kept.
   Evidence: the same docstring; DESIGN §9.5 row 18 note and §9.6 erratum. Status: recorded (one
   run each, Apple M5 Max under load); the paper should state the row's SBC pass with this caveat.
7. **Geweke "error 2" (simulator draws fresh latents)** is caught by Geweke tests (M3, max |z|
   23.5 vs control 2.59). Status: verified.
8. **Every mutant must be caught by some tier** (DESIGN §9.7): the table of mutant × catching tier
   with statistics is a natural paper table. Status: verified, 33/33 mutant × tier checks caught
   with passing controls (`tests/mutation`, ~10 min with a warm compilation cache; machine not
   recorded). Re-run on the release commit for the paper table (`RAI_BVAR_VALIDATION_RECORD=<dir>`
   writes the JSON). **Count superseded (2026-10-23):** 33 was M1–M16 only. WP25 added M17–M18
   (U and G each), so the suite has 37 mutant × tier checks (`pytest --collect-only -m mutation
   tests/mutation` collects the 37 `test_mutant_caught` cases), all 37 caught on
   `408a7f4`; the weekly CI job now records them in `mutation.jsonl`.

8b. **The path-conditioned RB log score breaks down in the tails at h > 1.** SPEC §7's estimator
   averages p(y_{T+h} | θ_s, simulated path_s up to T+h−1). For outcomes far in the predictive's tail
   (US unemployment 2008–10 at h = 8) one draw carries all the weight: Constant's UR h = 8 score at
   2008Q2 was −93.6 (1000 draws), −46 to −63 (8000), −36 (64,000) against −19.1 for the
   path-integrated mixture over the same 1000 draws (exact given those draws; itself a Monte Carlo
   estimate, see the residual error below), while the delta-method MCSE stayed at ≈ 1. It inflated the quick
   Clark–Ravazzolo UR h = 8 ΔLPS about 8× (+1.92 vs ≈ +0.23 converged). A methods point worth a
   section: integrate the path out (Gaussian given θ and the volatility path) instead of conditioning
   on it. Evidence: WP23 review (2026-10-23), `benchmarks/clark_ravazzolo.py` RELIABILITY_NOTE.
   Status: recorded (review computations; re-run with the library fix before publishing).
   **Library fix (D-34, 2026-10-23):** `log_pred_density(…, lpd_method="integrated")` (opt-in; the
   SPEC §7 default is unchanged) conditions each draw on its parameters and drawn volatility path
   only: `y_{T+h} | θ_s, λ_s ~ N(m_h, M_h M_hᵀ)` with `m_h` the zero-shock mean (recovered from the
   contract fields as `cond_mean_h − Σ_{k<h} Φ_{h−k} u_k`) and `M_h = [Φ_{h−1}L_1 | … | Φ_0 L_h]`,
   factored by QR. What it fixes: the weights no longer depend on where a draw's simulated path went,
   so the tail estimate is no longer carried by one path. Exactness: with one parameter draw and a
   constant volatility it is the exact predictive density (1e-10 against a companion-form oracle at
   h = 2..8, 6-s.d. tails, zero MCSE); under SV it is exact given each drawn volatility path; on
   MCMC fits each draw's density matches the moments from `jax.jacfwd` of the
   forecast core to 1e-10 (1e-13 against exact rational arithmetic in a one-off check). A
   synthetic persistent VAR reproduces the pathology: at h = 8 and 6 s.d. the conditioned
   estimator gives −35.2 with MCSE 1.0 against the exact −22.3
   (`tests/unit/test_lpd_integrated.py`). Scope: MCMC forecasts of linear or zero means (they carry
   the new `lag_coefs` variable); analytic, benchmark and ordering-pooled forecasts raise at h > 1;
   no closed form with the transitory Rao-Blackwellisation at h > 1. Residual Monte Carlo error:
   it uses no extra random numbers but is still an average over the forecast's parameter and
   volatility-path draws, whose weights still concentrate in the tails. At the acceptance budget
   (4 × 20,000) and 2008Q2, UR h = 8, with 1000 forecast draws: Constant's weight ESS is 2–6 draws
   and its estimate −19.0 against −17.8 with 8000 (fit seed 1); over thinning offsets the
   1000-draw estimate has sd ≈ 0.75–1.06 and a downward log-of-mean bias of ≈ 0.3–0.6 nats
   (SV(rw): ≈ 0.4–0.7), and SV(rw)'s spread over forecast seeds (sd 1.4) exceeds its delta-method
   MCSE (0.45–0.94). The heaviest weights sit on consecutive draws of one chain, so more forecast
   draws remove the thinning error, not the MCMC error (80,000-draw values still differ by
   ≈ 1.3 nats across fit seeds). Evidence: D-34 review (2026-10-23, finding math-2). The
   pre-registered CR acceptance config therefore uses 8000 forecast draws (thinning error sd
   0.09–0.19 nats, bias ≈ 0 in the same review). Status: estimator verified
   (tests). **Acceptance run (`65dd212`, pre-registered at `c823d20`):** UR h8 ΔLPS +0.273 with the
   integrated estimator against +1.309 with the conditioned one on the same forecasts (h1 +0.160
   for both); the largest UR h8 origin contributions fall from about +41 nats to +7.6. Criterion 4
   still fails (+0.273 > +0.160), as the pre-registration allowed; criteria 1, 2, 3, 5, 6 are met. Plumbing:
   `score_forecast`, `backtest`, `compare` and `ordering_sensitivity` take `lpd_method`; integrated
   backtest scores also carry the conditioned estimator on the same forecast (so the paper can show
   both from one run) and have their own cache keys. The pre-registered CR `acceptance` config
   (`benchmarks/clark_ravazzolo.py --config acceptance`, 4 × 20,000 draws) scores with it in every
   stage and reports the conditioned ΔLPS for the criterion-4 cells as a secondary number.

## C. Engineering lessons

9. **Bitwise reproducibility has limits.** Results are bitwise identical across chunk sizes,
   burn-in piece sizes, fused/unfused plans and `chains='map'`, but not across `vmap` batch sizes,
   platforms (OpenBLAS vs Accelerate) or `shard_map`; those are compared at rtol ≈ 1e-12.
   Evidence: tests under `tests/mcmc/` (invariance tests). Status: verified in CI (Linux).
   **Tolerance corrected (2026-10-23):** rtol 1e-12 holds for the toy kernels; the production SV
   and CCCM kernels are compared at rtol 1e-9 on short runs (chain-count prefixes under `vmap`,
   `shard_map`; `tests/mcmc/runner/test_chunking.py`, `test_devices_bitwise.py`, DESIGN §5.2), and
   the bitwise cases above are tested with the toy kernels, plus production chain prefixes with
   `chains="map"`.
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
    **Replay replaced (2026-10-24, after release 0.2.0; PR #11, merge `3644c4b`):** the release's
    slow replay of chain 3214 (194 sweeps with the old Cholesky draw, then status 1003 from it at
    sweep 194) is not portable: at `cond(K_a) ≈ 1/ε` whether the old draw's Jacobi-scaled pivot is
    0 or a few ε depends on the last bits of its inputs, and on x86-64 Linux the old-draw trajectory
    already fails before sweep 194 (all 8 x86-64 jobs of CI runs 40105877002 and 40106105193; sweep
    170 on the AMD EPYC 7763, 162 on the 9V45), while on macOS arm64 it is bitwise the committed one.
    The chain's sweep-194 state is committed (`tests/fixtures/f1_reduced_sv_chain3214_sweep194.npz`;
    bitwise the state replayed with 3d3f51b's code) and two slow tests use it and the chain's prior
    draw; `tools/make_f1_state.py` keeps the replay. The numbers above (14.8 and 1.9e17) are those of
    `K_a` after sweep 194's CCCM step; the test forms `K_a` entering the sweep (13.7 and 1.92e17).
    Status: verified (PR #11 tests on 8 x86-64 Linux and 2 macOS arm64 jobs).
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
    (`q_t = δ_t + q_{t−1}/(1 + q_{t−1})` on unit edges, all terms ≥ 0) is the same factorisation without the
    cancellation: at ordinary σ it agrees with the old draw to ~1e-13 and every non-RW composition
    stays bitwise unchanged. The trigger is the σ → 0 region where the centred σ² step mixes
    slowly and only the NC move escapes; the fix removes the crash, not the slow mixing.
    Evidence: `tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma` (60-digit
    reference) and `::test_rw_step_at_a_tiny_sigma_is_ok`. Status: verified (Apple M5 Max).

## F. Defaults and the convergence gate

18. **A convergence-gated verdict needs a composition-aware MCMC budget.** With one default for
    every composition (4 chains × 2,000 kept draws) the SV fits routinely missed the R-hat < 1.01 /
    bulk ESS > 400 gate, so `compare()` answered `inconclusive` for every SV comparison although
    ΔLPD was large (README example: +85.2, MCSE 1.1, at h = 1; the SV arm converged at 0 of 11
    origins). The SV parameters drive it: φ and σ have ESS ≈ 22–28 per 1,000 sweeps per chain (WP3
    pilot). The budget, not the gate, was changed (D-35): SV compositions default to 4 × 10,000,
    and an SV fit below that warns. Even that raises the floor without guaranteeing the gate
    (README example: 10 of 11 origins at 4 × 10,000, 11 of 11 at 4 × 20,000; Clark–Ravazzolo
    `SV(rw)`: 72 of 106 at 4 × 5,000, 106 of 106 at 4 × 20,000; BENCH-P2, n = 20, T = 300: 4 of 101
    primary quantities still miss at 4 × 10,000; the README's full-sample fit: 1 of 16, σ of GDP
    at R-hat 1.012). A methods point: the gate does its job (it refuses verdicts the draws cannot
    support), and the default budget has to follow the slowest-mixing block of the composition.
    Evidence: DESIGN §12.2 Q14 and D-35; `tests/unit/test_default_budget.py` for the policy.
    Status: recorded (Apple M5 Max, one run each; BENCH-P2 under load).

## G. The weekly tier on Linux (after release 0.2.0)

19. **The weekly tier passes on x86-64 Linux.** The one weekly run before the release
    (40052069815, on a 2-vCPU AMD EPYC 7763) stopped at its 330-minute limit about half-way. After the release, with `src/` unchanged from 4ac7969 and the
    F1 replay replaced by PR #11's tests, the whole tier passed twice: run 40106532730 in one job on
    a 4-vCPU AMD EPYC 9V45 (274 min) and run 40140397848 in six shards on 4-vCPU AMD EPYC 7763s
    (PR #12, `tools/weekly_shards.py`; 72–120 min each, 121 min end to end). Each: 483 tests, 481
    passed, 1 skipped (no bayesianVARs fixture), 1 xfailed (stochvol Student-t). The slow
    empirical-Bayes tests of `tests/unit`, left out of the macOS release record because of the
    macOS heap corruption (cause unknown, issue #3), passed in one process there. Evidence:
    `figures/data/provenance/ci_runs_40106532730_40140397848_weekly_x86.txt`; DESIGN §9.10 at
    2c71fb2. Status: verified (CI logs; the validation-record artifacts were not downloaded).

## Before submission

- Re-run every "recorded" / "to check" item on the release commit and cite the commit hash.
- Disclose AI assistance as the target venue requires (most venues do not accept an AI system as
  an author; they require a statement of how it was used).
- Third-party claims (section A): re-test on current versions and contact the maintainers first.
