# Phase 4 week-1 spike: can the BART mean block embed `bartz`? (decision gate)

**Status.** Decision memo for the week-1 spike of SPEC §13 Phase 4, written 2026-10-25 on branch
`claude/phase4-bartz-spike` from `main` at `2c71fb2`. It answers the spike's three questions with
evidence from a prototype that runs inside the production sweep. Nothing here is public API: there
is no `rb.BART`, no `MeanKind`, and no registry entry.

**bartz.** 0.13.0 (PyPI, MIT, released 2026-10-04), pinned in the new optional extra `bart`
(`uv sync --group dev --extra bart`). CI installs only the dev group, so the spike's tests skip
there (`pytest.importorskip`). Only MIT sources were read: the bartz wheels 0.7.0–0.13.0 and the
bartz `main` branch at `19682e7`. `mpfarrho/mf-bavart` (Apache-2.0) was not needed. No GPL code was
read (SPEC §2.1).

**Machine.** Apple M5 Max (18 cores, 64 GB), macOS, JAX 0.12.0.1 on the CPU. **The machine was not
idle.** Another session's DGP-study pilot (12 worker processes near 100% CPU each) and this spike's
own validation jobs kept the 1-minute load average at 120–195 for every run. Each record and each
timing row carries `os.getloadavg()`. Wall times are therefore upper bounds. The cost table (§4)
compares the two samplers with both arms timed the same way, by minimum wall time and by process
CPU time. Neither measure is load-invariant: at lower load XLA uses more threads, so the CPU time
rises while the wall time falls (§4).

---

## 0. Decision

**Recommendation: build the Phase 4 `BART` mean block as an adapter around a pinned `bartz`
(0.13.0 now). No fork and no in-house JAX BART.** The adapter is about 150 lines
(`set_data`, the template state and the step wrapper), and every check below passes with it. Three
conditions go with the recommendation:

1. **Amend SPEC §8 and R4.** SPEC says bartz "computes and samples leaf full conditionals in float32
   even with x64 enabled". That is not true of any release from 0.7.0 to `main`: under
   `jax_enable_x64` bartz's `step` does not even trace (§Q2.1). The prototype runs bartz with x64
   switched off for the duration of its trace. The exception is therefore **the whole bartz
   sub-state is float32**: residuals, precision scales and caches as well as leaves. Everything
   outside it stays float64. A float32 error study (§Q2.4) shows this costs nothing measurable
   against the float64-residual alternative, as long as targets are standardised.
2. **Standardise each equation's target per window (R9).** The float32 leaf conditional is accurate
   to ≤ 2·10⁻⁴ posterior s.d. when the level the trees have not fitted is within 10 noise s.d.
   The error grows linearly with that ratio and reaches about 1 s.d. at 10⁵ (§Q2.4). The ratio of
   the volatility extremes does not matter (tested up to e¹⁶).
3. **Pin bartz exactly and re-run the adapter suite on every bump.** The fields the setter writes
   changed at 4 of the 5 minor-version steps from 0.7 to 0.12 (0.8, 0.10, 0.11 and 0.12; §Q1.3),
   and 0.13.0 changed the `init` argument that builds the template. The suite (`tests/mcmc/blocks/test_bart_spike.py`, about
   1.5 min) catches such a change bitwise.

**Upstream** (three small requests; issue texts in §5.2):

- (a) export bartz's new private `mcmcstep._rebuild.rebuild_derived` (on `main`, unreleased)
  together with an error-scale setter, which would replace most of rai-bvar's `set_data`;
- (b) a one-line int32 cast so that `step` traces under x64. The cast-before-scatter half of this
  fix is already on `main`;
- (c) let `init` accept a prior-free error precision for continuous outcomes, which `step` already
  supports.

None of these blocks Phase 4.

**The three questions in one line each.**

| Question | Answer | Evidence |
| :--- | :--- | :--- |
| 1. Swap the residual target and `1/λ_{i,t}` into `State` each sweep, in a jitted loop, correctly across releases? Public setter? | **Yes, with a caveat on churn.** `set_data` rewrites 13 fields and rebuilds the incremental per-leaf precision cache. It is bitwise equal to `init` (all written fields, and one `step` afterwards), and it reproduces bartz's own caches on grown forests. It is unchanged and exact on 0.12.0, 0.12.1, 0.13.0 and `main`. There is no public setter; `main` adds a private `rebuild_derived` that covers most of it | §Q1 |
| 2. Does the float32 leaf step pass Geweke and SBC at realistic T and volatility ranges? | **Yes, but only with bartz traced under x64 off** (or with a 2-line patch; both pass). Geweke G-small passes for every row at T = 50–400 (n = 1–3, p = 1–2), including a wide-volatility prior and path (b), and G-full passes at T = 200; two mutants are caught. **SBC still running** when the PR opened (§Q2.3) | §Q2 |
| 3. Is a fixed error precision (`Wishart(value=...)`) the right hook to leave volatility to the SV block? | **Yes.** `step` skips the σ update exactly when `error_cov_inv.nu is None`, and the per-observation weights are then the exact leaf-likelihood precisions (verified against the exact conditional). But `init` refuses that `Wishart` for continuous outcomes, so the adapter swaps it in after `init` | §Q3 |
| Cost per sweep at n = 3–7, T = 200–400, 200 trees | 6.5–19 ms per sweep on one chain (minimum wall time, load 37–49), **13–17× `Linear × SV`** timed the same way (14–29× across the other measures and loads). The bartz tree loop dominates; `set_data` adds 3–10%; `vmap` gives no CPU parallelism. An idle-machine re-run is open (§5.1a) | §4 |

---

## 1. What was built

| File | What |
| :--- | :--- |
| `src/rai_bvar/components/mean/_bart_spike.py` | `BartPrior` (static tree-prior choices, fixed cutpoint grid); `template_state` (a root-only bartz `State` with dtypes normalised to `step`'s output); **`set_data`** (the setter); `bartz_step` (bartz under the right x64 setting); `BartSpikeMean`, a `MeanBlock` with `fuses_contemp=False`, `has_state=True`, `Batching.VMAP` |
| `src/rai_bvar/validation/_bart_spike.py` | `BartJoint` (a `JointModel` for Geweke and SBC: the BART tree prior on top of the `zero_sv` composition's structure, SV block, test prior and fixed design). Independent box-form references: `forest_prior_sample`, `forest_eval`, `tree_log_prior`. Plus `bartz_x64_patch` (the path-(b) experiment) |
| `tests/mcmc/blocks/test_bart_spike.py` | 13 block tests: Q1 setter exactness, Q3 precision hook, the x64 facts, the reference model against bartz |
| `tests/geweke/test_geweke_bart_spike.py` | Geweke rows (G-small, G-full) with `test_geweke_compositions`' sizes, rule and record |
| `tests/sbc/test_sbc_bart_spike.py` | SBC with `test_sbc_compositions`' sizes, rule and record |
| `benchmarks/phase4_bart_spike_cost.py` | Per-sweep cost (§4) |
| `benchmarks/phase4_float32_leaf_error.py` | Float32 leaf-conditional error study (§Q2.4) |
| `pyproject.toml`, `uv.lock` | Optional extra `bart = ["bartz==0.13.0"]` |

**How the prototype sits in the engine.** The production sweep is used unchanged. The model
`BART × SV` is `compose(static, overrides={"mean": BartSpikeMean(...)})`, the kernel-slot test seam
of DESIGN §4.9, with a `MeanStatic("zero")` placeholder; `make_sweep` then gives the plan
`mean → contemp → residual → vol`. That is DESIGN §4.3's non-fusing, stateful-mean order. Each step
does the following:

- **`mean`** (`BartSpikeMean.sample_eq`): `set_data` swaps in the target `ỹ_i + Σ_{j<i} a_ij ỹ_j`
  (that is, `ctx.target − Z a_i` from `sweep.mean_context`) and the weights
  `w_t = m_t·exp(−log λ_eff,i,t)`, then one `bartz.mcmcstep.step` updates all trees.
- **`contemp`**: `Recursive.contemp_step` draws `a_i | g_i`, with the same prior as `Linear`
  (SPEC §4.3, R2).
- **`vol`**: the existing SV block (`VolBlockImpl` with `SVCore`) runs on `e_i = target − Z a_i − g_i`.

`eq.mean` holds the bartz `State` of each equation, batched over equations (and over chains) by
`vmap`. DESIGN §3.8 reserved `Batching.MAP` in case bartz did not vmap; it does. The binned lag
block reaches the kernel through `FitCache.eq` (`BartSpikeMean.prepare`).

**A fixed cutpoint grid.** The tree prior must not depend on the data for Geweke and SBC to be
valid. The covariates are therefore the lag block of the design row (the intercept column has no
cutpoint), binned on a fixed grid of 31 cutpoints on [−3, 3]. bartz's own quantile binning
(`prepcovars`) moves with the data; production can bin per backtest window (R9), but the test
model cannot.

---

## Q1. Swapping the target and the weights into bartz's `State`

### Q1.1 What has to change, and why it is more than two fields

`init` derives five fields from `error_scale`: `inv_sdev_scale`, `prec_scale`, `inv_sdev_unit`,
`n_non_missing` and `sum_diag_prec_scale`. The values are stored in power-of-two units
(`inv_sdev_unit`, `resid_unit`). On top of that, `step` keeps a per-node cache of summed precisions,
`Forest.prec_tree`, and updates it **only at the nodes a move touches** (`_compute_count_or_prec_tree`).
After a change of `prec_scale`, every other leaf of every tree is stale. Swapping `resid` and
`error_scale` alone would therefore give a silently wrong likelihood.

`set_data(state, X, target, weights)` (in `components/mean/_bart_spike.py`) writes the 13 fields of
`SET_FIELDS` and reads 7 more:

- `X` and `forest.leaf_indices`, by re-traversing the trees, with `forest.to_prune` cleared.
  `split_tree` already reflects a pending prune, so the fold at the start of the next `step` becomes
  a no-op. Re-traversal is needed because Geweke's successive-conditional simulator redraws `y`, and
  hence the lags, every sweep. In production the data are fixed, so the harness exercises the
  adapter harder than production will.
- `resid = (y − forest(X))/resid_unit`, with bartz's sanitised `y` (the offset on zero-weight rows).
  It is computed from the trees, so no running-update drift accumulates across sweeps, and
  `resid_inexact_integral` is reset to 0.
- The five scale fields, derived from `w` as `init` derives them. Zero weights on padded rows drop
  out like bartz's `missing`.
- `forest.prec_tree`, rebuilt for **every** node (bottom-up sums). This makes it valid where 0.13.0
  needs it (the leaves) and where `main`'s new one-bin update needs it (the parent of a grow).
- `y` and `error_scale`, kept consistent (the continuous `step` reads neither).

All of it is jit- and vmap-safe and runs inside the Geweke and SBC `lax.scan` loops over 1000 chains.

### Q1.2 Exactness (`tests/mcmc/blocks/test_bart_spike.py`)

| Test | Result |
| :--- | :--- |
| `test_set_data_on_a_stump_equals_init_bitwise`: `set_data` on a template versus `mcmcstep.init` on the same data, with power-of-4 weights (so that `1/error_scale` is exact in float32) and 14 padded rows | All 13 written fields bitwise equal; **one `step` from either state is bitwise identical in every leaf** |
| `test_set_data_reproduces_bartz_caches_on_a_grown_forest`: after 100 steps on grown trees (asserted: more than 10 internal nodes over the 20 trees), `set_data` with unchanged data versus bartz's own caches | Leaf indices equal after the pending-prune fold; `prec_tree` at actual leaves within 2·10⁻⁶ relative (float32 summation order); residuals within 10⁻⁵ (float32 running updates); fitted values within 10⁻¹² |
| `test_swapping_weights_changes_only_the_written_fields` | A swap changes only `SET_FIELDS` (a guard against releases that add derived fields) |
| `test_padded_rows_never_enter_the_leaf_likelihood` | Targets and bins of zero-weight rows change no cache that reaches the likelihood |

### Q1.3 Across releases

Fields and arguments the adapter touches, by release (from the wheels' sources; `main` at
`19682e7`, 2026-10-24):

| | 0.7.0 | 0.8.0 | 0.9.0 | 0.10.0 | 0.11.0 | 0.12.0 | 0.12.1 | 0.13.0 | `main` |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `State.resid`, `prec_scale`, `Forest.leaf_indices`, `affluence_tree` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| error precision | `sigma2` | `error_cov_inv` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `State.inv_sdev_scale` | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Forest.prec_tree` / `count_tree` (incremental caches) | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `error_scale`, `inv_sdev_unit`, `n_non_missing`, `sum_diag_prec_scale`, `resid_unit`, `resid_eff_scale`, `resid_inexact_integral`, `to_prune`, `move_node`, `leaf_unit`, `offset`; `init(resid_dtype=, prec_scale_dtype=, leaf_dtype=)` | — | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| leaf prior precision in `init` | `sigma_mu2` | array | array | array | array | array | array | `Wishart` | `Wishart` |
| `step` skips the σ update for a prior-free precision | — | — | `error_cov_df is None` | `error_cov_df is None` | `nu is None` | `nu is None` | `nu is None` | `nu is None` | `nu is None` |
| `init` asserts `rate is not None` (continuous) | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `step` asserts int32 `lrt_nodes` (fails under x64) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| private `mcmcstep._rebuild.rebuild_derived` | — | — | — | — | — | — | — | — | ✓ |

Run against each release (`benchmarks`-style scratch venvs; the commands are in §6):

- **0.12.0 and 0.12.1**: the setter is exact, unchanged. All written fields equal `init`, one
  `step` is bitwise identical, and the caches are reproduced. Only the template's `init` call needs
  the 0.12 argument, `leaf_prior_cov_inv` as an array.
- **0.13.0**: all 13 block tests pass.
- **`main` @`19682e7`** (0.13.1.dev194, 194 commits ahead): 12 of 13 pass. The setter tests (bitwise
  included) and the x64 and `init` findings are unchanged. The failure is the test-side oracle
  `test_reference_prior_sampler_matches_bartz_debug`, because `bartz.debug.sample_prior` changed its
  signature. On `main` the sparsity fields moved into `Forest.sparsity`, and `StepConfig` gained
  `one_bin_prec_reduction`.

Releases come every 1–2 months: 0.8 (Feb 2026) to 0.13 (Oct 2026) is six minor releases in eight
months. So the adapter has been stable since 0.12.0 (Jul 2026), but the step before that (0.11 → 0.12)
rewrote the units and half the fields. The pin and the bitwise suite are the mitigation (§0
condition 3).

### Q1.4 Public setter

There is none in any release, and no upstream issue or PR asks for one (searched `bartz-org/bartz`
for setter, `error_scale`, Gibbs, x64 and float64). Two things make the request natural:

- bartz 0.13.0 already does this swap **internally** for its BCF model: `bcf/_loop.py` replaces
  `resid` and `prec_scale` between the μ and τ forests, and has a private
  `recompute_prec_trees` ("Rebuild `Forest.prec_tree` from scratch for a new `prec_scale`").
- `main` adds a private `rebuild_derived(state)`. It recomputes `leaf_indices`, `count_tree`,
  `prec_tree`, `affluence_tree` and `resid` from the trees and `y`, and drops pending prunes. It does
  **not** recompute the scale fields, which it treats as fixed by `init`.

The upstream ask (§5.2 a) is therefore small: export `rebuild_derived` plus a `set_data` that takes
`y`, `error_scale` and `X`. rai-bvar's `set_data` would then shrink to the scale fields and a call.

---

## Q2. float32 under Geweke and SBC

### Q2.1 bartz does not run with `jax_enable_x64` on

rai-bvar enables x64 at import (SPEC §8). Under x64:

- `_propose_moves` computes `node << jnp.array([1, 1, 0])` as int64, and
  `accept_move_and_sample_leaves` asserts `lrt_nodes.dtype == jnp.int32`. So `step` raises
  `AssertionError` at trace time (`test_bartz_step_does_not_trace_under_x64`). The assert is present
  in every release from 0.7.0 to `main`.
- With float64 residuals, `BatchedReduction._reduce` (`mcmcstep/_reduction.py:200`) scatters
  float64 values into the float32 per-leaf sums, which JAX warns "will result in an error" in future
  releases. `main` fixes this ("fix: cast values to dtype in BatchedReduction").

There are two ways around it:

| Path | How | float32 in the bartz state | Status |
| :--- | :--- | :--- | :--- |
| **(a)** (default) | `init` and `step` traced inside `jax.enable_x64(False)`; the state's dtypes normalised to `step`'s output; the adapter computes in float64 and rounds at the boundary; fitted values returned as float64 sums of the float32 leaves | every float: residuals, precision scales, caches, leaves | unmodified bartz 0.13.0; no dtype warning survives (the tests turn "Explicitly requested dtype" into an error) |
| **(b)** | bartz traced under x64 with float64 residuals and precision scales, float32 leaves (SPEC §8's stated exception), after the two one-line fixes (`validation._bart_spike.bartz_x64_patch`, monkeypatch, experiment only) | leaves and the per-leaf reductions | needs the int32 cast upstream (§5.2 b) |

Two side observations about path (a). Switching x64 off inside an outer x64 trace
(`jax.enable_x64(False)` within the jitted sweep) was **observed** to work in JAX 0.12.0.1; it is
not documented behaviour, and path (a) rests on it, so a JAX upgrade must re-run the block suite. And bartz's `step` donates its argument when called eagerly, so the template is kept as
host NumPy arrays.

### Q2.2 Geweke (`tests/geweke/test_geweke_bart_spike.py`)

The joint model is `A y_t = g(x_t) + Λ_t^{½} ε_t`. The tree prior has bartz's convention:
`P(split | depth δ, a rule available) = 0.95/(1 + δ)²`, the variable is uniform over the available
ones, the cutpoint uniform over the available ones, and leaves are `N(0, (0.5/√M)²)`. The rest is the
`zero_sv` test prior (`a_ij ~ N(0, 0.7²)`, the §9.2 SV prior). The forests have 20 trees of depth
≤ 5; exactness of the swap and of the float32 step does not depend on the number of trees.

The rule and sizes are DESIGN §9.5's. G-small is `C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3` and
G-full is `C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`, both with Bonferroni over the test functions
and the Gandy–Scott wrapper. The test functions are:

- the `zero_sv` reference functions that do not involve the mean (`a`, SV parameters and states,
  data functions);
- per equation: `g` at three fixed probe points and their squares; mean and mean square of `g(x_t)`;
  `g` at three dates; the leaf count and its square; mean depth; split counts per variable; leaf sum
  of squares; and the exact tree log-prior;
- `loglik`, `Σe²/λ` and `log p(θ)`, recomputed with the BART residuals.

The tree prior sampler, the forest evaluator and the log-prior are independent box-form
implementations (DESIGN D-22), checked against bartz's own sampler (two-sample χ²) and against
exact enumeration (§Q2.5).

| Row | n, p, T | What it tests | Tier | Result (max \|z\|, argmax; `t_crit`; G functions) | Wall time |
| :-- | :-- | :-- | :-- | :-- | --: |
| `block_T200` | 1, 1, 200 | BART block alone (`live = eq.mean`; SV held at its prior draw) | G-small | **pass**, 1.80 (`data.acf1_log_y2[0]`); 4.23; 39 | 277 s |
| `bart_sv_T50` | 2, 1, 50 | BART × SV, full sweep | G-small (PR) | **pass**, 2.03 (`eq1.mean.tree_log_prior`); 4.40; 83 | 321 s |
| `bart_sv_T200` | 2, 1, 200 | same at a realistic T | G-small | **pass**, 2.46 (`eq1.mean.n_split_var[0]`); 4.40; 83 | 690 s |
| `bart_sv_n3p2_T200` | 3, 2, 200 | 6 covariates, three equations | G-small | **pass**, 2.53 (`data.acf1_y[2]`); 4.51; 140 | 1433 s |
| `bart_sv_T200_x64` | 2, 1, 200 | path (b) | G-small | **pass**, 1.93 (`eq0.core.sigma^2`); 4.40; 83 | 1006 s |
| `bart_sv_wide_T400` | 2, 1, 400 | **wider SV prior** (`μ ~ N(0, 2²)`, `(φ+1)/2 ~ Beta(20, 1.5)`, `B_σ = 0.5`) | G-small | **pass**, 1.54 (`eq1.core.h0`); 4.40; 83 | 1615 s |
| `bart_sv_T200` | 2, 1, 200 | as above | **G-full** | **pass**, 1.48 (`data.acf1_y[1]`); 3.85; 83 | 6041 s |

**Can the harness see the bug the adapter prevents?** Two mutants of the kernel
(`test_g_small_bart_mutant_fails`, test-file subclasses only, DESIGN §9.7 convention), at the
`bart_sv_T50` sizes and seed:

| Mutant | Result |
| :--- | :--- |
| `stale_prec_tree`: the naive swap (residuals and error-scale fields written, `forest.prec_tree` left as the previous sweep left it) | **fails**, but only at stage 2. Stage 1 was marginal (max \|z\| 3.54 at `eq1.joint.rss_std` against 4.40; Bonferroni p 0.034, which triggers the 4× rerun). Stage 2 (C = 4000, M₁ = 8·10⁵) gave `loglik` \|z\| 4.88 against 4.38, p 9·10⁻⁵. The minimum relative numerical efficiency was 10⁻¹⁵–10⁻²²: some chains become extremely heavy-tailed under the stale cache, which costs the z-test power |
| `sqrt_weights`: the precisions passed as `√w` | **fails** at stage 1: max \|z\| 12.4 (`eq1.core.mean_abs_h`), p 4·10⁻³¹, 17 of 83 functions failing (SV parameters and states, `log p(θ)`, `Σe²/λ`) |

So the G-small rows can detect a wrong likelihood, and they detect the stale cache only through
the Gandy–Scott rerun. The bitwise block tests (§Q1.2) are therefore the primary guard for the
cache, and Geweke the backstop.

**Volatility ranges covered** (within-sample `max_t λ_t / min_t λ_t` per equation, 4000 prior draws):
the default SV test prior at T = 200 gives a median of 5 (90th percentile 66, 99th 2·10³). The wide
prior at T = 400 gives a median of 2.4·10² (interquartile range 13–1.7·10⁴, 90th percentile 3·10⁶).

Every row passed at stage 1 of the Gandy–Scott wrapper (no rerun was needed). The wide-prior row
makes the test harder, not easier; no prior, setting or criterion was changed to make a row pass.

### Q2.3 SBC (`tests/sbc/test_sbc_bart_spike.py`)

The settings are those of `test_sbc_compositions` (DESIGN §9.6 with that module's `t_max = 200`):
`N = 1000`, `L = 99`, `burn = 300`, `pilot = 1000`, `t_max = 200`, `α = 1e-2`. That means the
data-only filter `max|ỹ| < 10³`, both ESS gates and the Gandy–Scott wrapper. They were
pre-registered from a pilot before any SBC run: on 16 prior-drawn `T = 100` datasets × 3000 sweeps,
the largest per-quantity IAT was 43 (`φ`; tree split counts 20–38, median over quantities ≈ 20),
inside what `t_max = 200` covers (IAT ≤ 182).

The quantities are the 53 test functions minus data functions, squares and cross-products: `a`, the
SV parameters and states, `g` at probe points and dates, `g` mean and mean square, leaf count, mean
depth, split counts, leaf sum of squares, tree log-prior, the log-likelihoods and `log p(θ)`. There
is no predictive quantity, because forecasting is outside the spike.

| Row | n, p, T | Result | Wall time |
| :-- | :-- | :-- | --: |
| `bart_sv_T100` | 2, 1, 100 | **running** (started 2026-10-25 12:48 on this PR's `src` and `tests/sbc`, both unchanged since; the result is appended here when it completes) | |

### Q2.4 How large is the float32 error? (`benchmarks/phase4_float32_leaf_error.py`)

Geweke and SBC have little power against a bias of 10⁻³ s.d. So the leaf conditional was also
checked directly. One leaf holds `n = 400` observations with weights `w_t = e^{u_t}`,
`u_t ~ U(−R, R)` (a within-sample variance ratio of `e^{2R}`), and targets
`y_t = c·σ̄ + ε_t/√w_t`. Here `c` is the level the trees have not yet fitted, in noise units. The
exact conditional is `N(Σwy/P, 1/P)` with `P = p₀ + Σw`. bartz's value is recomputed with its own
internals on the state the adapter builds: `sum_resid`, `prec_tree`, `scaled_error_cov_inv` and
`_precompute_leaf_terms_uv`. Worst case over 20 replications, mean error in posterior s.d. units
(the variance's relative error is < 10⁻⁶ everywhere):

| variance ratio `e^{2R}` | c = 0 | c = 10 | c = 10³ | c = 10⁵ |
| --: | --: | --: | --: | --: |
| 1 | 1.8·10⁻⁷ | 1.9·10⁻⁵ | 2.0·10⁻³ | 0.18 |
| 55 | 4.8·10⁻⁷ | 4.7·10⁻⁵ | 5.4·10⁻³ | 0.57 |
| 3.0·10³ | 4.5·10⁻⁷ | 6.0·10⁻⁵ | 5.6·10⁻³ | 0.77 |
| 1.6·10⁵ | 5.1·10⁻⁷ | 7.5·10⁻⁵ | 8.6·10⁻³ | 0.90 |
| 8.9·10⁶ | 1.2·10⁻⁶ | 1.8·10⁻⁴ | 1.8·10⁻² | **1.8** |

Path (b), with float64 residuals and precision scales, gives the same numbers to two digits: the
error comes from the float32 per-leaf reduction and `mean_factor`, which both paths share, not from
the residual storage. Two consequences follow:

- The volatility range is not the risk. The unexplained level relative to the noise is, so targets
  must be standardised, as BART practice does anyway (§0 condition 2).
- Path (b) buys conformance with SPEC's wording, not accuracy.

### Q2.5 The reference model

| Test | Result |
| :--- | :--- |
| `test_reference_prior_sampler_matches_bartz_debug`: 20,000 trees, depth 5, 3 variables, 4 cutpoints | internal-node counts, root variables and root cutpoints agree with `bartz.debug.sample_prior` (two-sample χ², p > 10⁻³); leaf s.d. within 2% |
| `test_reference_tree_log_prior_matches_sampler_frequencies`: depth 3, 2 variables, 3 cutpoints, 60,000 trees | the exact structure log-prior matches the sampling frequencies (χ² goodness of fit) and sums to 1 within 0.01 |
| `test_reference_evaluator_matches_bartz` | box membership = heap descent (kernel `traverse`) to 10⁻¹² = `bartz.grove.evaluate_forest` to float32 |

---

## Q3. A fixed error precision as the hook

- `step` runs `step_error_cov_inv` only if `state.error_cov_inv.nu is not None` (0.11.0 onwards;
  `error_cov_df is not None` in 0.9–0.10).
  With `Wishart(None, None, value=1)` the per-observation precision is
  `error_cov_inv · prec_scale · inv_sdev_unit² = w_t`, so the SV block's `1/λ_{i,t}` is the whole
  error precision. bartz learns no global scale on top of it, and there is no double counting with
  the SV level `μ`.
- `test_fixed_precision_and_weights_are_exact_leaf_precisions`: one tree is held at the root
  (`min_points_per_leaf > T`), with weights spanning e⁶ ≈ 400. 4000 independent `step`s draw the leaf
  from `N(Σwy/(p₀ + Σw), 1/(p₀ + Σw))`, with z-tests on the mean and the variance (|z| < 4).
  `error_cov_inv` stays exactly 1 and the tree never splits.
- **Caveat.** `init` asserts `rate is not None` for continuous outcomes (0.11.0 to `main`), although
  `Wishart`'s docstring offers `nu = rate = None` for "a precision held fixed". The adapter builds
  the state with a dummy prior and swaps the fixed precision in with `dataclasses.replace`
  (`test_init_rejects_a_prior_free_error_precision`). That uses a documented `Wishart` mode outside
  the place `init` allows it; it is a one-line upstream change (§5.2 c).
- Two consequences are worth stating:
  - `resid_eff_scale` is updated only inside `step_error_cov_inv`, so it stays at its initial value.
    It matters only with `leaf_quantization`, which the adapter leaves off.
  - The leaf prior precision is held fixed the same way (`leaf_prior_cov_inv = Wishart(None, None,
    1/σ_μ²)`): the CGM calibration is a per-window constant (R9).

---

## 4. Cost per sweep

`benchmarks/phase4_bart_spike_cost.py`. Setup: 200 trees of depth ≤ 6, p = 4 (so `n·p` = 12–28
covariates), 31 cutpoints, path (a), one chain. Each figure is the production sweep timed after
50 warm-up sweeps, as the median of 7 timings of a jitted 20-sweep `lax.scan`. `Linear × SV` is
the production `structural_sv` sweep at the same `n`, `p` and `T`. Both arms are timed the same
way in every row. Compare like with like: minimum wall time against minimum wall time, and CPU
time against CPU time.

**Table A: lower load** (load average 37–49, during a gap between the DGP pilot and the full DGP
study; still not idle):

| n | T | BART × SV wall ms (min) | BART × SV wall ms (median) | BART × SV CPU ms | Linear × SV wall ms (min) | Linear × SV wall ms (median) | Linear × SV CPU ms | ratio wall (min) | ratio CPU | bartz step, one equation, wall ms (min) | `set_data`, one equation, wall ms (min) | load |
| -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 3 | 200 | 6.5 | 8.1 | 15.0 | 0.43 | 0.58 | 0.76 | 15× | 20× | 2.07 | 0.09 | 45 |
| 3 | 400 | 10.3 | 11.0 | 23.2 | 0.62 | 0.71 | 1.04 | 17× | 22× | 2.95 | 0.11 | 43 |
| 5 | 200 | 9.7 | 10.5 | 20.5 | 0.61 | 0.84 | 0.91 | 16× | 23× | 1.96 | 0.08 | 49 |
| 5 | 400 | 13.4 | 14.2 | 33.7 | 0.98 | 1.11 | 1.81 | 14× | 19× | 2.69 | 0.10 | 41 |
| 7 | 200 | 11.9 | 13.6 | 28.1 | 0.90 | 0.97 | 1.51 | 13× | 19× | 2.27 | 0.08 | 40 |
| 7 | 400 | 19.3 | 25.3 | 45.2 | 1.42 | 1.90 | 2.48 | 14× | 18× | 3.23 | 0.22 | 43 |

**Table B: heavy load** (load average 130–150, the DGP pilot running), with the path-(b) and
4-chain columns:

| n | T | BART × SV CPU ms | BART × SV wall ms (min) | Linear × SV CPU ms | Linear × SV wall ms (min) | ratio CPU | ratio wall (min) | bartz step, one equation, CPU ms | `set_data`, one equation, CPU ms | path (b) BART × SV CPU ms | 4 chains BART × SV CPU ms | 4 chains Linear × SV CPU ms | compile s | load |
| -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 3 | 200 | 12.2 | 9.4 | 0.83 | 0.60 | 15× | 16× | 4.2 | 0.38 | 10.8 | 41 | 2.0 | 8 | 150 |
| 3 | 400 | 19.9 | 17.2 | 1.02 | 0.60 | 19× | 29× | 6.5 | 0.39 | 16.5 | 67 | 2.8 | 7 | 146 |
| 5 | 200 | 19.6 | 13.9 | 1.12 | 0.88 | 18× | 16× | 5.3 | 0.31 | — | — | — | 8 | 132 |
| 5 | 400 | 29.0 | 19.0 | 1.88 | 1.15 | 15× | 17× | 5.8 | 0.44 | — | — | — | 8 | 137 |
| 7 | 200 | 23.7 | 16.8 | 1.48 | 0.92 | 16× | 18× | 3.9 | 0.30 | 22.0 | 93 | 3.9 | 7 | 136 |
| 7 | 400 | 36.2 | 25.3 | 2.52 | 1.32 | 14× | 19× | 5.5 | 0.31 | 33.9 | 160 | 6.4 | 6 | 139 |

What the tables say:

- **BART × SV costs 13–17× `Linear × SV` per sweep on minimum wall time** at lower load: 6.5–19 ms
  against 0.4–1.4 ms. Under heavy load it is 16–29×, and on CPU time 14–23× under either load.
- **The cost is almost all bartz's step.** Its accept/reject stage is a `lax.scan` over the 200
  trees that cannot be parallelised across trees. So the cost per equation depends only weakly on
  `T`: about 2–3 ms of wall time for one equation alone.
- **The adapter is cheap**: `set_data` is 0.08–0.2 ms wall (0.2–0.4 ms CPU) per equation, 3–10%
  of the step, including the re-traversal that production could skip when `X` is unchanged.
- **Path (b) costs the same as path (a)**: within 10%, inside the noise.
- **`vmap` does not parallelise on the CPU.** Under `vmap` the sweep's CPU time is about `n` × the
  per-equation step. Four chains under `vmap` cost 3.3–4.4× one chain. More cores need more XLA CPU
  devices (one chain per device, `rai_bvar.set_cpu_devices`), which BENCH-P2 already supports for
  chains.

**What a fit costs**, from Table A. At n = 7, T = 400, 4 chains × (1,000 burn-in + 10,000 kept)
sweeps is about 4 × 11,000 × 19 ms ≈ 14 min of wall time on one device (`vmap` gives no saving).
That is about 3.5 min on 4 CPU devices if the per-device cost holds, against about 1 min for
`Linear × SV` on one device. A backtest of 100 origins at that budget is of the order of 6–24 h
on this laptop. That
argues for a smaller kept-draw budget in BART backtests than the SV default of D-35, and for GPU
runs, which bartz is built for. It is a scheduling cost, not a reason against bartz: an
in-house BART would have the same sequential tree loop.

The `Batching.MAP` variant (equations one after another under `lax.map`) is in §4.1.

### 4.1 `Batching.MAP` against `vmap`

`--batching map`, one chain, load average 120–143. Times are CPU ms per sweep, then wall ms
(median / minimum):

| n | T | MAP | `vmap` |
| -: | -: | :-- | :-- |
| 3 | 200 | 12.0; 10.8 / 8.4 | 12.2; 14.5 / 9.4 |
| 3 | 400 | 15.1; 13.8 / 9.8 | 19.9; 18.9 / 17.2 |
| 7 | 200 | 28.2; 23.5 / 19.4 | 23.7; 18.4 / 16.8 |
| 7 | 400 | 38.1; 33.5 / 27.0 | 36.2; 32.2 / 25.3 |

The two are equal within the noise. The prototype keeps `Batching.VMAP` (DESIGN §3.8's `MAP` hedge
is not needed). Neither gives parallelism across equations on one XLA CPU device.

---

## 5. Recommendation and next steps

### 5.1 Phase 4 implementation, in order

1. The `bart` kind proper: `MeanKind`, a registry entry, `state_template`/`draw_template` for the
   bartz sub-state (built by `init` under path (a), like `template_state`), `float32_leaves` in the
   class variable (they are per-shape today), and `rb.BART` with CGM defaults.
2. Per-window standardisation of the targets and leaf-scale calibration (R9; §0 condition 2), and
   quantile cutpoints per window.
3. Forecasting: `predict_eq` by forest evaluation at `x_{T+h}`. Forests are stored only through the
   in-loop forecast hook (DESIGN §5.8).
4. A CI job with the `bart` extra that runs the block suite (about 1.5 min) on every PR touching
   the adapter. The Geweke and SBC rows need their own weekly shard:
   - On this machine, under load, they took about 85 min for the five slow G-small rows, plus the
     SBC and the G-full run (§Q2).
   - The weekly shards have a 240-min test-step timeout (DESIGN §9.10) and the `geweke-other`
     shard already takes 82 min on the 4-vCPU EPYC 7763, so the rows cannot simply be appended
     there.
   - `bart_sv_T50` (G-small, about 5 min) is PR-tier only by its sizes; it should become nightly
     when the job exists.
5. Skip the re-traversal when `X` has not changed (production); measure the gain.

### 5.1a Open items of the spike itself

- **Re-run the single-chain cost table on an idle machine**, after the DGP study (12 workers,
  about 10 h) ends, with both arms timed the same way:
  `uv run python benchmarks/phase4_bart_spike_cost.py --chains 1 --out <file>`. §4 has the table at
  load averages of 37–49 (Table A) and 130–150 (Table B), each labelled.

### 5.2 Upstream requests (drafts; not filed, since filing is the owner's call)

- **(a) Public setter.** "`mcmcstep.set_data(state, *, y=None, error_scale=None, X=None)` for Gibbs
  embedding: write the data, recompute the scale fields as `init` does
  (`compute_scale_related_attrs`), then `rebuild_derived`. Use case: BART as one block of a larger
  Gibbs sampler (a VAR with stochastic volatility: the target and the per-observation precisions
  change every sweep). bcf already does this internally (`recompute_prec_trees`)."
- **(b) x64.** "`step` fails under `jax_enable_x64`: `_propose_moves` builds `lrt_nodes` as int64
  (`node << jnp.array([1, 1, 0])`) and `accept_move_and_sample_leaves` asserts int32. Fix:
  `.astype(jnp.int32)` on `lrt_nodes` (the cast before the scatter in `BatchedReduction` is already
  on `main`)."
- **(c) Fixed error precision in `init`.** "Allow `error_cov_inv=Wishart(None, None, value)` for
  continuous outcomes (`init_shape_shifting_parameters` asserts `rate is not None`); `step` already
  skips the update when `nu is None`."

### 5.3 When to reconsider

Consider a fork, or an in-house BART, if either of these happens:

- a bartz release changes `step`'s semantics in a way the bitwise suite cannot absorb with a small
  adapter change (cache invariants, unit scheme);
- SPEC is held to "float64 residuals" and (b) is refused upstream. Even then, §Q2.4 says the change
  would be cosmetic.

---

## 6. Reproduce

```bash
uv sync --group dev --extra bart
uv run pytest tests/mcmc/blocks/test_bart_spike.py -q                       # Q1, Q3, x64, references (~1.5 min)
uv run pytest tests/geweke/test_geweke_bart_spike.py -m "not slow"          # bart_sv_T50 (~5 min)
RAI_BVAR_VALIDATION_RECORD=rec uv run pytest tests/geweke/test_geweke_bart_spike.py -m slow
RAI_BVAR_VALIDATION_RECORD=rec uv run pytest tests/sbc/test_sbc_bart_spike.py -m slow
uv run python benchmarks/phase4_float32_leaf_error.py
uv run python benchmarks/phase4_bart_spike_cost.py --out bart_cost.json
```

Other releases: create a venv with `uv venv`, then
`uv pip install -e . "bartz==0.12.1" pytest hypothesis` (or
`"bartz @ git+https://github.com/bartz-org/bartz@19682e7"`), and run the block suite. On 0.12.x
the template needs `leaf_prior_cov_inv` as an array; the setter itself needs no change.
