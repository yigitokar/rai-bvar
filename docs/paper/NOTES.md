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

## C. Engineering lessons

9. **Bitwise reproducibility has limits.** Results are bitwise identical across chunk sizes,
   burn-in piece sizes, fused/unfused plans and `chains='map'`, but not across `vmap` batch sizes,
   platforms (OpenBLAS vs Accelerate) or `shard_map`; those are compared at rtol ≈ 1e-12.
   Evidence: tests under `tests/mcmc/` (invariance tests). Status: verified in CI (Linux).
10. **Float64 failure in the prior's extreme tail.** One Geweke chain out of 4000 reached `h₀ ≈ 30`
    where `|y| ≈ 1e8` made a Cholesky pivot zero; the engine reports it (no silent jitter).
    Evidence: `tests/geweke/test_geweke_compositions.py` docstring. Status: recorded.
11. **The design-review errata (E1–E28)** in DESIGN §0.2 document what independent reviewers caught
    in the proposals before implementation. Status: verified (in repository).

## Before submission

- Re-run every "recorded" / "to check" item on the release commit and cite the commit hash.
- Disclose AI assistance as the target venue requires (most venues do not accept an AI system as
  an author; they require a statement of how it was used).
- Third-party claims (section A): re-test on current versions and contact the maintainers first.
