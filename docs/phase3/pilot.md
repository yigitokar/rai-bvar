# Phase 3 pilot: GL and CommonSV mixing, costs, BENCH-P3 (WP36)

**Status.** Draft, written before the mixing campaign ran: this version states the method and the decision
rule only; the measurements follow in later commits.

## Decision rule (fixed before any ladder result)

- **Gate.** P2 §8.2 unchanged: every primary quantity (P2 D-33 with P3 D-52's additions) has rank-normalised split
  R-hat < 1.01 and bulk ESS > 400, 4 chains, no numerical failure. It is evaluated by the production function
  `engines.mcmc.monitor.converged`.
- **Fits at budget `D`.** One run per (configuration, size, seed): 4 chains, 1,000 burn-in sweeps, `D_max` kept draws
  (`thin = 1`). The first `D` kept draws of every chain are exactly the draws of a fit with `draws = D` and the same
  seed, because the sweep keys depend only on the absolute sweep index (checked against `BVAR.fit`:
  `phase3_mixing.py --check`). Each seed is one independent fit at every `D` of the ladder
  `D ∈ {2,000, 5,000, 10,000, 20,000, 50,000}` kept draws per chain.
- **"The median fit passes" at `D`** ⇔ at least ⌈R/2⌉ of the `R` seeds pass the gate at `D` (`R = 5` at `n = 7`,
  `R = 3` at the BENCH-P2 shape). The recommended budget for D-58 is the smallest ladder `D` at which the median fit
  passes and keeps passing at every larger ladder `D`, per family × volatility × size. No budget is inferred from ESS
  alone: R-hat does not extrapolate, so a recommendation is always a `D` that was evaluated. Where no ladder `D` passes,
  the recommendation is "above 50,000", with the ESS-based projection stated as a projection.
- **ESS per 1,000 sweeps** = bulk ESS of the 4 chains ÷ (4·D/1000), that is per 1,000 sweeps of one chain; the median
  over seeds is reported, with the slowest primary quantity named.

## Sizes, configurations and seeds

- `qd7`: FRED-QD medium set (`rai_bvar.datasets`, vintage 2026-09, the verified local cache), 1969Q1–2019Q4, `n = 7`,
  `p = 4`, `T_eff = 200`; seeds 0–4.
- `bp2`: BENCH-P2's shape and data (`simulate_var_sv(20, 4, 300, seed = 0)`, P2 §10.1); seeds 0–2.
- Configurations: the five family defaults (`Horseshoe()`, `NormalGamma()`, `R2D2()`, `DirichletLaplace()`, `SSVS()`)
  and `SSVS(p = Beta(1, 1))` × {`Constant()`, `SV()`}; `Minnesota(form="structural", hierarchy="fixed")` × {`Constant()`,
  `SV()`} as the Phase 2 reference; `Minnesota(form="conjugate", hierarchy="fixed") × CommonSV()` (the shared-factor path),
  `Minnesota(form="structural", hierarchy="fixed") × CommonSV()` (the per-equation program), and both with row 325's
  CommonSV priors (`(ρ+1)/2 ~ Beta(10, 2)`, `σ_h² ~ 0.1·χ²₁`).
