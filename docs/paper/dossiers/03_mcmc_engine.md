# Dossier 03: the JAX MCMC engine (architecture, algorithms, performance)

Input for the methods/software paper on `rai-bvar` (target style: *Journal of Statistical
Software*). Topic: the Phase 2 JAX Gibbs engine. Everything below is taken from files in the
repository snapshot `/Users/yigitokar/rai-bvar-paper` (branch `paper-draft`, HEAD `62d15f6`) or from
commands stated in §0.3. Later phases are mentioned only as roadmap hooks.

---

## 0. Conventions

### 0.1 Status tags

The ledger statuses come from `docs/paper/NOTES.md` (header). This dossier adds two tags for
descriptions of code and design:

| Tag | Meaning |
| :--- | :--- |
| **[code]** | Describes committed source at `62d15f6`, read directly (file and symbol cited). Not a measured claim. |
| **[design]** | Stated in `docs/phase2/DESIGN.md` and not separately confirmed in code or tests by this dossier. |
| **[verified]** | Reproducible from a committed test or fixture (test named). The dossier author did **not** re-run tests: pytest writes caches into the repository. |
| **[recorded]** | A number from a run that is written down in a committed file or commit message, not re-run. |
| **[to check]** | Must be re-run or re-derived before it goes into the paper. |

### 0.2 Citation shorthand

`DESIGN §x` means `docs/phase2/DESIGN.md`. `N0k §x` means `docs/phase2/0k_*.md` (N01 Omori mixture,
N02 SV sampler, N03 outliers/t, N04 CCCM, N05 structural sampler, N06 validation, N07 JAX patterns,
N08 common SV/ordering). `pilot §x` means `docs/phase2/09_pilot.md`. `NOTES n` means entry `n` of
`docs/paper/NOTES.md`. `SPEC §x` means `SPEC.md`. Code paths are relative to `src/rai_bvar/`
unless they start with `tests/` or `benchmarks/`.

### 0.3 Commands run by the dossier author

All were run on the dossier machine: Apple M5 Max, 18 logical CPUs (`sysctl`), 1-minute load
average 4.95 at the time. They change no file.

1. `cd /Users/yigitokar/rai-bvar-paper && PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -c '…'`.
   Output: `decode_error(2001) = (Block.EQ_VOL, 0)`; `jax.config.jax_threefry_partitionable = True`;
   `jax 0.12.0.1`; at `q = 100`, `P = q(q+1)/2 = 5050` and `T_pad·P·8 = 12,928,000` bytes.
2. `git show d37e059:tests/bench/baseline.json`: the provisional x86 baseline, which the M5 Max
   reference baseline replaced (§13.7).
3. `git log -- tests/bench/baseline.json`, `git show d490730 -s`, `git show d37e059 -s`: the
   baseline commit messages (§13).
4. `git diff --stat deb3929 HEAD -- src/rai_bvar/engines/mcmc src/rai_bvar/components` and the
   diffs of `components/vol/sv.py`, `engines/mcmc/_blas.py`, `engines/mcmc/runner.py`: what changed
   after the baseline commit (§13.10).
5. `PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -c 'from rai_bvar.engines.mcmc import _blas;
   print(_blas.openblas())'` printed `None` (SciPy 1.18.1 in the repository's `.venv`): no OpenBLAS
   thread API is found in this environment, so `_blas.single_thread()` is a no-op here (§8.4).

---

## 1. What the engine is, in one page

`rai-bvar` fits every non-conjugate model (stochastic volatility, outliers, Student-t, reduced-form
priors) with one hand-written Gibbs engine in JAX. SPEC rule R7 excludes probabilistic programming
languages from the core (SPEC §3, R7). The engine is built on six ideas. Each is a numbered
decision in DESIGN §0.1.

1. **The static/traced split is the cache key.** Every choice that sets a shape or a branch lives in
   a frozen, hashable `StaticConfig`. Data, priors, hyperparameters, `T_eff`, `thin`, `burn` and
   seeds are traced arguments. The JIT cache key is exactly `(StaticConfig, argument shapes)`
   (D-23; DESIGN §1.3, §2.4, §4.9). **[code]** `engines/mcmc/config.py`.
2. **Equations and chains are `vmap` axes.** In recursive structural form the equations are
   conditionally independent, so one kernel is written for one equation and batched (SPEC §4.1;
   DESIGN §4.6 item 5). Chains are an outer `vmap`, an opt-in `shard_map` or a `lax.map`. **[code]**
   `engines/mcmc/compose.py` (`eq_map_for`), `engines/mcmc/runner.py` (`chain_map`, `_sharded`).
3. **The Gibbs order is data.** A `Plan` of `Step`s, each with declared read and write sets, is
   built in one place and checked against invariants that encode the Del Negro–Primiceri step-order
   rule (DESIGN §4.1, §4.8). **[code]** `engines/mcmc/plan.py`.
4. **Pure kernels with injected noise.** Every block is `core(noise, …) ∘ draw_noise(key, …)`, and
   every exact draw exposes its conditional parameters (D-17; DESIGN §3.1 rules 2–3). **[code]**
   for example `SVCore.draw_noise` and `SVCore.sample_core` (`components/vol/sv.py`).
5. **Counter-based randomness.** Keys are `fold_in` chains over seed → stream → data salt → chain →
   absolute sweep → block → equation → sub-draw, with uint32 counters and no key in any loop
   carry (D-16). **[code]** `engines/mcmc/keys.py`.
6. **Failures are values.** No exception inside JIT and no jitter. Each chain carries the first
   `(code, it)` of a failure. The host replays the failing chunk deterministically and reports
   diagnostics recomputed from the failing block's exposed conditional (D-18; DESIGN §5.7).
   **[code]** `engines/mcmc/runner.py` (`_Run.check`, `replay_to`, `failure_detail`).

All floats are float64 (SPEC R4, §8; DESIGN conventions). The only exception mechanism is a
component's declared `float32_leaves`, which is empty in Phase 2 (DESIGN §2.8).

---

## 2. Model recap (only what the engine needs)

Recursive structural form (SPEC §4.1):

```
A y_t = g(x_t) + Λ_t^{½} ε_t,   A = I + strict-lower(a) (unit lower triangular),
Λ_t = diag(λ_{1,t}, …, λ_{n,t}),   ε_t ~ N(0, I),   reduced form Σ_t = A⁻¹Λ_tA⁻ᵀ.
```

Equation `i` is a heteroskedastic univariate regression of `y_i` on `x_t` and `y_{<i,t}`. Because
`|A| = 1` and the priors are independent across equations, the posterior factorises over equations
given the hyperparameters. One sweep costs `O(n k³) = O(n⁴p³)` rather than `O((nk)³)` for joint
sampling (SPEC §4.1, "Why this form"; DESIGN §4.6 item 5). **[design]**

Two coefficient-prior forms run on the engine:

- **structural** (`Minnesota(form="structural")`, the MCMC default; SPEC decision D3): the
  Minnesota prior sits on structural coefficients `γ_i`, centred on `a_i` so that the reduced-form
  prior *mean* is the familiar `M`: `E[γ_i | a_i] = M ã_i` (SPEC §4.1). In δ-coordinates
  `γ_i = δ_i + M_c a_i` (DESIGN §2.3).
- **reduced** (`Minnesota(form="reduced")`): a prior on reduced-form `Π`, which couples the
  equations. It is sampled by the corrected triangular algorithm of Carriero, Chan, Clark &
  Marcellino (2022) [CITE?] as a pre-phase global step (SPEC §4.1, §4.3; DESIGN §3.7.3, §4.3).

Volatility blocks: `Constant` (inverse-gamma λ_i), `SV` (AR or random-walk log variance), each
optionally with transitory outlier and/or Student-t scales (DESIGN §3.7.4–3.7.6).

---

## 3. Static configuration and pytrees

### 3.1 `StaticConfig`

**[code]** `engines/mcmc/config.py` (JAX-free, so `api.py` can validate compositions without
importing JAX; DESIGN §1.1 rule 2, §2.4):

```python
@dataclass(frozen=True)
class StaticConfig:
    shapes: Shapes          # n, lags, n_det, k = n_det + n·lags, q = k + n − 1, T_pad
    struct: StructStatic    # kind="recursive", contemp_scaling ∈ {none, lambda}
    mean:   MeanStatic      # kind, coef_scaling, const_path, gram, gram_tile, cccm_precompute, n_dummy
    coef:   CoefPriorStatic # kind="fixed_diagonal"
    vol:    VolStatic       # persistent ∈ {constant, sv}, rw, h0, sigma2_prior, transitory,
                            # outlier_K, nu_grid, t_scale, mixture, exact, asis, zero_offset, coupled
    store:  StoreStatic     # paths, in_loop_forecast (horizon H, 0 = off)
    batching: Literal["vmap", "map"] = "vmap"
    chunk: int = 250        # kept draws per compiled sample call
    float32_leaves: frozenset[str] = frozenset()
```

- `T_pad = t_bucket·⌈T_eff/t_bucket⌉` (`config.bucket`) depends only on `(T_eff, t_bucket)`. The
  default `t_bucket` is 32 (`settings.MCMCSettings.t_bucket`). It is never padded to a backtest
  maximum (D-23). **[code]**
- `n ≤ MAX_EQUATIONS = 999`, because error codes encode the equation in three digits
  (`config.py` line 36). **[code]**
- `validate_composition` holds the legality rules: λ-scaling only with a Constant core, Mode II
  only with `linear_structural`, fast paths only with constant weights, and others (DESIGN §3.8
  item 2, §2.10). **[code]**
- Every choice that changes draws (`gram`, `const_path`, `t_bucket`) is an `MCMCSettings` field, so
  it is in the spec hash. Shape rules resolve `"auto"` deterministically from shapes, never from
  machine memory (D-23, E14; DESIGN §10.5). **[design]**; resolution happens in
  `engines/mcmc/resolve.py`.

### 3.2 Pytree conventions

**[code]** `engines/mcmc/state.py`; DESIGN §2.1:

- Every state, data, prior and cache class is registered with
  `jax.tree_util.register_dataclass(cls, data_fields=…, meta_fields=…)`. Meta fields are static
  and hashable.
- **No static flag ever lives in a leaf.** Flags are meta fields of frozen kernel dataclasses or of
  `StaticConfig` (erratum E19: a `bool` inside a NamedTuple crossing `jit`/`vmap`/`scan`).
- **Fixed structure per `StaticConfig`.** No leaf is `None`. An absent part is the registered
  dataclass `Empty()`, which has no leaves. For example `FitCache.FF` is `Empty` unless
  `gram="packed"` and `const_path="weighted"` (DESIGN §2.10, Stage 1 amendments;
  `engines/mcmc/data.base_cache` docstring).
- Kernels are written for **one equation**. Per-equation leaves carry a leading axis `n`, and the
  runner adds a leading chain axis `C` (and optionally a fit axis `F`).
- Acceptance counters and Rao–Blackwellised sums belong to the component that produces them, as
  leaves named `acc_*` / `rb_*`. `state.zero_accumulators` zeroes every such leaf by name at the
  burn-in → sampling boundary (DESIGN §2.1, §5.1; erratum E8).
- Mixture indicators `s` are **never state**. They are local to `VolBlock.sample_eq`, which is how
  "nothing between `s` and `h`" is enforced structurally (DESIGN §2.7; N01 §9).

Main chain-state classes (DESIGN §2.7; **[code]** `state.py`):

```text
ChainState = (eq: EqState[n], glob: GlobalState, status: Status(code i32, it u32, min_pivot f64))
EqState    = (struct: RecursiveEq(a f64[n, n−1]), mean: LinearEq(delta f64[n, k]) | Empty,
              coef: Empty, vol: VolEq(core, trans))
VolEq.core = ConstantEq(log_lam) | SVEq(h[T_pad+1], mu, phi, sigma, acc_*, rb_h_sum, rb_h_sq)
             | SVRWEq(h, sigma, …)
VolEq.trans= Empty | OutlierEq(o_idx i32[T_pad], log_p, log1m_p, rb_out) | TEq(log_tau, nu_idx)
             | OutlierTEq(…)
GlobalState.mean = CCCMGlobal(Pi f64[k, n]) | Empty
```

`γ_i = δ_i + M_c a_i` is derived and never stored (erratum E24). DESIGN §2.7 puts memory per chain
at about 0.3 MB at `n = 20, k = 81, T_pad = 320` for SV-O-t. **[design]**

### 3.3 Coefficient layout and exact padding (D-15)

**[code]** DESIGN §2.3; `components/mean/linear.py` module docstring; `linalg.precision_draw`
docstring:

- `θ_i = [δ_i ∈ ℝ^k | a_i-slots ∈ ℝ^{n−1}]`. Slot `k + j` holds `a_ij` and is active iff `j < i`,
  so inactive slots are **trailing** and `q_i = k + i` coordinates are active (erratum E26).
- Inactive slots get prior mean 0, prior precision 1 and zero data columns, and are zeroed after
  the draw. The precision is block diagonal, so the active draw equals the unpadded draw computed
  with `z[:q_i]` (DESIGN §9.3 item 1: tested to 1e-12). **[verified]**
  `tests/mcmc/unit/test_linalg.py::test_padding_is_exact`.
- Time padding is on the right. Rows `t ≥ T_eff` carry weight 0. SV states with index `> T_eff`
  are decoupled `N(0, 1)` dummies. Every sum over `t` is masked, and padded rows are masked
  **before** any `log` (erratum E23). Forecasting reads `h[:, T_eff]` with a dynamic index, never
  `h[-1]`. **[code]** `SVCore.extract` uses `lax.dynamic_index_in_dim`.
- Padding overhead at the reference size is 1.333× (Cholesky) and 1.215× (Gram). Two static
  equation buckets would give 1.152×/1.100× at the best split. They were not adopted: −2% to −4% of
  the full sweep, about the noise, and the draws would not be bitwise equal to one bucket. Sources:
  pilot §10, decision 11 (prototype, loaded machine). DESIGN §2.3 still quotes "≈ 1.08×/1.05×" for
  two buckets, which pilot §10 corrects (see Open questions).

### 3.4 Import-time rules and float64

**[code]** DESIGN §1.1; `engines/mcmc/_jax.py`; `_devices.py`:

- `import rai_bvar` imports no JAX module (D-31; asserted in a subprocess by
  `tests/mcmc/unit/test_import_no_jax.py`). The first JAX-using engine module imports
  `engines/mcmc/_jax.py`, which enables x64.
- No module-level `jnp` array (DESIGN §1.1 rule 3). Mixture tables, ν grids and packed-index maps
  are NumPy and become constants at trace time (`components/vol/mixture.py` docstring;
  `linalg._packed_index`). A module-level `jnp` array would initialise the backend and make
  `rai_bvar.set_cpu_devices(k)` impossible.
- MCMC test modules turn the warning `"Explicitly requested dtype"` into an error, so a silent x64
  truncation fails the tests (DESIGN §9.10). **[code]**, for example `tests/mcmc/runner/test_recompile.py`
  `pytestmark`.

---

## 4. Batching: equations, chains, devices, fits

### 4.1 Equations

**[code]** `engines/mcmc/compose.py` (`eq_map_for`, `lax_map_eq`); DESIGN §4.5:

```
eq_map = jax.vmap                                      (Batching.VMAP, every Phase 2 kernel)
       | λ f: λ *args: lax.map(λ t: f(*t), args)       (Batching.MAP, for non-vmappable kernels)
```

`Program.batching` is the maximum over components (DESIGN §3.8 item 3). Shared arguments (data,
caches, shared priors) are closed over inside the jitted sweep. They are traced arguments of the
enclosing executable, never HLO constants (DESIGN §4.5, §4.9; N07 §7 rule 2). `Batching.MAP`
exists for Phase 4 BART (`bartz`) if it does not `vmap` (DESIGN §3.9). **[design]** for the Phase 4
use.

Kernels must not branch on traced values inside vmapped bodies, because `lax.cond`/`switch` lower
to `select` and run both branches (DESIGN §3.1 rule 1; N07 §2.1). **[design]**

### 4.2 Chains, devices, fits

**[code]** `engines/mcmc/runner.py`:

| Mode | Mechanism | Determinism (see §10) |
| :--- | :--- | :--- |
| default `chains="vmap"` | `jax.vmap(f, in_axes=_CHAIN_AXES[...])`: keys and state batched, data/prior/cache shared (`None`) | chain-count prefixes equal **to rounding** with the production SV and CCCM kernels |
| `chains="map"` | `lax.map` over chains: one identical one-chain program per chain | chain-count prefixes **bitwise** |
| `Runtime(devices=D)` | the same body under `jax.shard_map` over a 1-D `("chain",)` mesh, `NamedSharding`-placed keys and state (`_sharded`) | equal to one device to rounding, so opt-in; `Runtime.devices` defaults to 1 |
| `fit_axis=True` | one more `vmap` over a leading fit axis `F` of data, priors, caches, keys and state | equal to looped fits to 1e-10 |

`rai_bvar.set_cpu_devices(k)` (`_devices.py`) sets `jax_num_cpu_devices` and must run before JAX
initialises. Otherwise it raises an actionable `RuntimeError`, and `resolve_devices` falls back to
one device with a warning (DESIGN §5.2; R4).

### 4.3 Why batched equations are an exact block draw

In structural form the equations are conditionally independent given the global state: `|A| = 1`,
and priors are independent across equations given hyperparameters (DESIGN §4.6 item 5; SPEC §4.1;
N05 §0 item 1). So the batched equation phase is an exact Gibbs block. In the reduced form the
rows `a_j` are independent given `Π, Λ` (N04 §6.2), and `Π` itself is drawn in the pre phase.
**[design]**; validated empirically by the Geweke/SBC tiers (see dossier on validation).

---

## 5. The Gibbs sweep

### 5.1 The order as data

**[code]** `engines/mcmc/plan.py` (module docstring; `Step`, `Plan`, `build_plan`,
`check_invariants`). `build_plan` is the only place where the order is decided, and
`fit.plan.describe()` prints it (DESIGN §1.3, §4.1). The two Phase 2 plans:

| Form | Phase | Step | Key block | Draw | Writes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| structural | equation | `equation` | `EQ_MEAN` | `(δ_i, a_i) \| λ_eff,i`, joint Gaussian (fused contemporaneous draw) | `eq.mean`, `eq.struct.a` |
| | equation | `residual` | `EQ_MEAN` (draws nothing) | `e_i = y_i − Fθ_i` | — |
| | equation | `vol` | `EQ_VOL` | `VolBlock.sample_eq` (atomic) | `eq.vol` |
| reduced (CCCM) | pre | `cccm` | `G_CCCM` | `Π \| A, λ_eff`, sequential over `j` | `glob.mean.Pi` |
| | equation | `contemp` | `EQ_CONTEMP` | `a_j \| Π, λ_eff,j` (regress `u_j` on `−u_{<j}`) | `eq.struct.a` |
| | equation | `residual` | `EQ_MEAN` | `e_j = (UAᵀ)_j` | — |
| | equation | `vol` | `EQ_VOL` | `VolBlock.sample_eq` | `eq.vol` |

A `coef_local` step (Phase 3 local shrinkage scales) and a `mean` step (Phase 4 non-fusing mean
with state) are inserted only when the corresponding state has leaves. In Phase 2 both are omitted
(`plan.py` docstring; `tests/mcmc/runner/test_plan.py::test_zero_mean_plan_omits_stateless_mean_and_coef_steps`,
`::test_stateful_non_fusing_mean_gets_a_mean_step`). **[verified]**

### 5.2 Algorithm 1: one sweep of one chain

From `engines/mcmc/sweep.py` (`_sweep_fn`, `equation_phase`, `EQUATION_STEPS`). DESIGN §4.5 is the
specification; the code expresses the equation body as a table of step functions that pass an
environment `env = {g, e, lv, cpl, infos}`.

```text
Algorithm 1  SWEEP(ckey, it: uint32, st, data, prior, cache)          # one chain; it = absolute sweep index
 1  for g in program.global_pre:                                       # static tuple, unrolled at trace time
 2      st, info ← g.sample(step_key(ckey, it, g.block), st, ctx, eq_map)        # reduced form: CCCM
 3      st ← record_status(st, info, g.block, eq = 0, it)
 4  ỹ ← structure.pre_residual(st.glob, data, mean.fitted_reduced(st.glob, data)) # Y, or Y − XΠ; × tmask
 5  (eq′, eq_out, infos) ← eq_map(BODY)(i = 0..n−1, st.eq, prior.eq, cache.eq)
 6  st ← record_eq_status(st with eq = eq′, infos, plan.equation, it)
 7  for g in program.global_post: …                                    # none in Phase 2; get eq_out = (e, log λ_eff)
 8  return st

    BODY(i, eq_i, prior_i, cache_i):                                   # ONE equation; env ← empty
 9  for step in plan.equation:
10      ctx_i ← structure.context(i, ỹ, eq_i, vol, data)               # rebuilt per step from the CURRENT eq_i:
                                                                       #   target ỹ_i, cols −ỹ_{·,:n−1}, cmask 1{j<i},
                                                                       #   weights m_t·exp(−log λ_eff,i,t), log_scale
11      key_b ← eq_key(step_key(ckey, it, step.block), i)
12      case step.name of
13        "equation": (δ_i, a_i, cpl, info) ← mean.sample_eq(key_b, …, ctx_i, pm, cm)   # Alg. 2/3
14                    g_i ← mean.fitted_eq(…)                           # Xγ_i, D3 term included
15        "contemp":  (a_i, cpl, info) ← structure.contemp_step(key_b, …, ctx_i, g_i, cm)  # Alg. 4
16        "residual": e_i ← target − (cols ⊙ cmask)·a_i − g_i           # 0 on padded rows
17        "vol":      (vol_i, info) ← vol.sample_eq(key_b, vol_i, e_i, tmask, T_eff, prior_i, cpl)  # Alg. 6
18                    lv_i ← vol.log_var_eq(vol_i)
19  return eq_i, env
```

Notes:

- `record_status` keeps the **first** non-zero `code = 1 + 1000·block + eq` per chain and the
  running minimum pivot (`sweep._set_status`, `state.encode_error`). For the equation phase,
  `record_eq_status` takes the first failing step in plan order, then its lowest failing equation
  (`jnp.argmax(bad)`). **[code]**
- DESIGN §4.5 passes `None` as the equation of global steps; the code passes `0`
  (`sweep._sweep_fn` line 300). Global failures are then located by replay (§9). **[code]**
- Every step rebuilds `EqContext` from the state at the time it runs, so steps are pure functions
  of `(state, env)`. That is what lets the same plan run **fused** (all steps in one `eq_map`
  body, the production default) or **unfused** (one `eq_map` per step, with `env` batched over
  equations). The two are bitwise identical (`sweep.py` docstring). **[verified]** with the test
  toys: `tests/mcmc/runner/test_sweep.py::test_fused_equals_unfused_bitwise` (3 configurations:
  structural SV + outliers, Mode II Constant, reduced SV).

### 5.3 Why the order targets the posterior; invariants

DESIGN §4.6 (summarised; **[design]**, with the empirical evidence in the validation dossier):

1. Mean, contemporaneous and CCCM steps condition on the exact Gaussian likelihood given `λ_eff`
   and are marginal of the mixture indicators `s` (Del Negro & Primiceri 2015 [CITE?], Alg. 2).
2. CCCM is a Gibbs scan over the full conditionals `π_j | π_{−j}` (CCCM 2022 eqs. 13–15). `K_j`
   does not depend on `π_{−j}`, so factoring all `K_j` first is exact (§6.4).
3. Transitory steps are exact conditionals given `h`. The τ-collapsed ones are partially collapsed
   Gibbs blocks whose trimmed draws are never conditioned on before being redrawn (van Dyk & Park
   2008 [CITE?]; N03 §5.1).
4. `s` is drawn immediately before `h` with nothing in between (DNP Alg. 2). With `exact=True` this
   is DNP Alg. 3, whose invariant law is exact for any offset `c ≥ 0` (N01 §10.2).
5. Equations are conditionally independent (§4.3).
6. Because `s` is local and redrawn before `h` inside every equation's vol step, a global step may
   sit in either the pre or the post phase without violating the DNP rule.
7. The coupled λ step with `Coupling` is the exact NIG conditional (N05 §5.2).

Plan invariants checked by `check_invariants` for every composition (DESIGN §4.8, which fixes
erratum E27; **[code]** `plan.py`):

1. Every step that writes an input of the structural residuals (`glob.mean.*`, `eq.mean.*`,
   `eq.struct.*`) runs in the pre phase or, within the body, before that equation's vol step.
2. Each equation has exactly one vol step, and indicators are not state.
3. `eq.vol.*` is written only by the vol step. Pre steps write only `glob.*`; post steps write
   only `glob.*` and `eq.coef.*`.
4. Each step's actual modifications are a subset of its declared `writes`. This is checked by a
   **tree diff** at a random state (`sweep.trace_steps` runs the plan unfused, one jitted step at a
   time).
5. A non-zero `Coupling` implies `VolStatic.coupled`, which implies a Constant core.
6. Every global step declares `marginal_of_indicators = True` (and passes a density-ratio
   identity, DESIGN §9.3).

**[verified]** `tests/mcmc/runner/test_plan.py::test_plan_invariants_every_composition_with_tree_diff`,
`::test_plan_invariants_production_kernels`, `::test_tree_diff_catches_undeclared_write`,
`::test_post_step_writing_residual_inputs_is_rejected`, `::test_coupling_without_coupled_core_is_rejected`.

---

## 6. The blocks

### 6.1 Algorithm 2: structural `(δ_i, a_i)` draw with D3 centring (weighted path)

**[code]** `components/mean/linear.py` (`LinearStructural._conditional`, `draw_core`,
`sample_eq`); `engines/mcmc/linalg.py` (`weighted_gram`, `precision_draw`); DESIGN §3.7.2, D-14,
D-25.

D3 in δ-coordinates (DESIGN §2.3; N05 §4.1, §7.1). With `γ_i = δ_i + M_c a_i`, where `M_c =
M[:, :n−1]` holds the reduced-form prior-mean columns, equation `i` becomes

```
y_i = X δ_i + Σ_{j<i} a_ij (X m_j − y_j) + e_i = F θ_i + e_i,
F = [X | X M_c − Y_{:, :n−1}] ∈ ℝ^{T_pad × q}.
```

Column `k + j` does not depend on `i`, so **one design `F` and one packed product matrix `FF` serve
all equations**. The prior on `δ_i` is centred on the own column of `M`. The resulting prior
satisfies `E[γ_i | a_i] = M ã_i` (SPEC §4.1 D3).

```text
Algorithm 2  STRUCTURAL_DRAW(key, ctx_i, pm, cm, cache)        # one equation; q = k + n − 1
   per fit:  FF[t, c] = F[t, r_c]·F[t, s_c], r_c ≤ s_c          # packed Khatri–Rao, T_pad × P, P = q(q+1)/2
 1  c_θ ← [1_k | 1{j < i}];  mean_θ ← [pm.mean | cm.mean];  prec_θ ← [pm.prec·e^{−log λ_i·1{Mode II}} | cm.prec]
    (inactive slots: mean 0, precision 1)
 2  w_t ← m_t·exp(−log λ_eff,i,t)                               # ctx_i.weights, masked before the exp
 3  G ← (w @ FF)[pos]                                          # q × q; under vmap one GEMM (C·n, T_pad)×(T_pad, P)
 4  f ← Fᵀ(w ⊙ y_i)
 5  K ← (c_θ c_θᵀ) ⊙ G + diag(prec_θ);   b ← prec_θ ⊙ mean_θ + c_θ ⊙ f
 6  z ~ N(0, I_q)  from sub_key(key, Sub.Z)
 7  s ← diag(K)^{−½};   L ← chol(s K s)                         # Jacobi scaling: an exact change of variables
 8  θ ← s ⊙ L⁻ᵀ(L⁻¹(s ⊙ b) + z)                                 # ~ N(K⁻¹b, K⁻¹), two triangular solves
 9  ok ← all(diag L finite and > 0) ∧ all(θ finite);   min_pivot ← min(diag L²)   (NaN pivot counts as 0)
10  θ ← θ ⊙ c_θ;   δ_i ← θ[:k];   a_i ← θ[k:]
11  cpl ← 0 (Mode I) | (k + i, ½Σ_active P̃_θ(θ − mean_θ)²) (Mode II) | same over the a-slots (iw_equivalent)
12  return (δ_i, a_i, cpl, ok, min_pivot)
```

Then `γ_i = δ_i + M_c a_i`, `g_i = Xγ_i` and `e_i = y_i − Fθ_i` (`Recursive.residual`). The
precision draw (step 8) is `s(sKs)⁻¹s = K⁻¹` (`linalg.py` docstring); only the lower triangle of
`K` is read. **[verified]** `tests/mcmc/unit/test_linalg.py::test_precision_draw_matches_dense_oracle`,
`::test_precision_draw_reads_only_the_lower_triangle`,
`::test_jacobi_scaling_with_large_precision_disparities`,
`::test_weighted_gram_modes_match_dense`.

The Gram has three deterministic modes (DESIGN §10.5, `linalg.weighted_gram`):

- `packed`: the default when `T_pad·P·8 ≤ 256 MiB`.
- `packed_tiled`: tiles of `T_pad·tile·8 ≤ 64 MiB` (multiples of 512 columns), with `FF` never
  stored.
- `batched`: `Fᵀ(w ⊙ F)` with the lower triangle mirrored. This is the oracle, used only on
  explicit request.

At `n = 20, p = 4` (`q = 100`, `P = 5050`, `T_pad = 320`), `FF` takes 12.9 MB (command 1, §0.3;
DESIGN §10.5). The packed order is column-major (`c = s(s+1)/2 + r`), so a prefix of `FF` is the
Gram of a leading block. This keeps the two-bucket lever available (pilot §10; `data.packed_index`).
**[verified]** `tests/mcmc/unit/test_linalg.py::test_packed_index_is_column_major`,
`::test_packed_order_matches_data_module`.

### 6.2 Constant-volatility fast paths (D-26)

When the weights are constant over time (Constant core, no transitory scale, a coefficient prior
that does not vary), `K(λ_i) = diag(d) + G_const/λ_i`. The path rule (DESIGN §3.7.2; `resolve.py`;
`const_path="cholesky"` forces the oracle):

| Composition | Path | Per-fit work | Per-sweep work |
| :--- | :--- | :--- | :--- |
| Mode I × Constant | `eigen` | one graded eigendecomposition per equation | `O(q²)`, no factorisation |
| Mode II × Constant | `fixed_factor` | one Cholesky of the λ-free `K̃` and its mean `θ̂` | `θ = θ̂ + √λ_i·s ⊙ Lc⁻ᵀz`: one triangular solve |
| `iw_equivalent` × Constant | `cholesky_const` | `G_const = FᵀD_mF`, `FtY` | Cholesky as in Alg. 2 with `G = G_const/λ_i` |
| anything time-varying | `weighted` | `FF` | Alg. 2 |

```text
Algorithm 3  EIGEN PATH (Mode I × Constant; linalg.eigen_prepare / eigen_draw; N05 §7.4)
  per fit, per equation:
 1  d ← prec_θ (λ-free in Mode I);   G̃ ← d^{−½}(c_θ c_θᵀ ⊙ G_const)d^{−½}
 2  graded eigendecomposition G̃ = Q diag(g) Qᵀ:
      δ ← diag(G̃)^{½};  A ← δ⁻¹G̃δ⁻¹ (unit diagonal);  L ← chol(A)   (if it breaks down: Lᵀ ← diag(√a⁺)Vᵀ from eigh(A))
      Bᵀ ← Lᵀ ⊙ δ (columns), sorted by decreasing δ;  Bᵀ = Q_r R (Householder QR)
      Rᵀ = U S Wᵀ by QR-iteration SVD (not divide and conquer);  Q ← U (rows unsorted);  g ← S²
 3  u1 ← Qᵀ(d^{½} ⊙ mean_θ);   u2 ← Qᵀ(d^{−½} ⊙ c_θ ⊙ FtY[:, i])
  per sweep (λ = λ_i, the variance):
 4  v ← λ/(λ + g)   (g clipped at 0)
 5  θ ← d^{−½} ⊙ Q(v ⊙ u1 + u2/(λ + g) + √v ⊙ z)               # exact: K = d^{½}(I + G̃/λ)d^{½}
 6  ok ← g finite ∧ g ≥ −1e-10·max|g| ∧ θ finite
```

Why the graded decomposition (from the `linalg.eigen_prepare` docstring): the default intercept
prior `var_const = 1e7` makes one diagonal entry of `G̃` about 1e9 times the others. A plain `eigh`
has absolute eigenvalue error `ε‖G̃‖` and then loses up to 7 digits in the other directions.
**[code]**. Accuracy is **observed, not proven** (same docstring). Against a 40–45-digit solve over
about 1,500 random Minnesota level-data problems (4,400 problem–λ pairs), with `κ = cond(sKs)`
and `t` the largest posterior t-ratio, the covariance error is `≤ 1e-12 + 0.8εκ` and the mean error
`≤ 1e-12 + 1.5εκ(1 + t)`. The largest mean error seen is 9e-5 sd. **[recorded]** (docstring);
**[verified]** for the committed cases:
`tests/mcmc/unit/test_linalg.py::test_constant_paths_on_level_data_vs_45_digits`,
`::test_eigen_prepare_graded_choices_are_pinned`, `::test_eigen_path_ks_on_10k_draws`.
DESIGN §9.3 item 1 states the path-vs-path test envelope: mean `2e-12 + 5εκ(1 + t)`, covariance
`2e-12 + 5εκ`, "verified on 2,250 fresh problems". This replaced a 1e-12 tolerance that is not
achievable in general. It is a paper-worthy point: two exact samplers of the same Gaussian agree
only to `O(εκ)`.

**Dense eigen path with SOC/SUR dummy rows** (WP25; NOTES 14). Dummy rows make the λ-free prior
precision dense: `P = diag(prec_θ) + w_d(c_θc_θᵀ ⊙ F_dᵀF_d)`. With `P = L_PL_Pᵀ`,
`K = L_P(I + L_P⁻¹GL_P⁻ᵀ/λ)L_Pᵀ`, so one per-fit decomposition again gives every λ in `O(q²)`.
The cache stores `L_P⁻ᵀQ` as `Q` and `d_half = 1`, so `eigen_draw` is unchanged
(`linalg.eigen_prepare_dense`). **[code]**. Status per NOTES 14 is verified:
`tests/mcmc/blocks/test_dummies.py::test_dense_eigen_path_with_dummies_equals_cholesky_const_test_prior`,
`::test_dense_eigen_path_with_dummies_equals_cholesky_const_minnesota`, plus Geweke G-small over 7
seeds and G-full. The timing (0.46 vs 1.53 ms/sweep, §13.9) is **[recorded]**. (Note added by
fact-check: NOTES 14 lists the timing inside its "verified" status line; treating a one-run timing
under load as recorded follows NOTES's own definition, but the ledger itself should say so.)

### 6.3 Algorithm 4: the contemporaneous `a_i` step by QR (non-fusing means)

The reduced form uses this step; in Phase 4 BART will too. **[code]**
`components/structure/recursive.py` (`contemp_core`, `contemp_lstsq_draw`); `linalg.lstsq_draw`;
DESIGN §3.7.1, §2.10 WP25 amendment (6).

```text
Algorithm 4  CONTEMP_DRAW(z, ctx_i, g_i, cm)      # a_i | g_i, λ:  ỹ_i − g_i = −ỹ_{<i} a_i + e_i
 1  Z ← cols ⊙ cmask   (cols = −ỹ_{·,:n−1});   √w ← √(m_t e^{−log λ_eff,t})
 2  A ← [diag(√prec) ; √w ⊙ Z];   rhs ← [√prec ⊙ mean ; √w ⊙ (target − g_i)]   # AᵀA = K_a, Aᵀrhs = b_a
 3  A = QR (Householder), sign-fixed so diag(R) > 0              # R = upper Cholesky factor of K_a
 4  a ← R⁻¹(Qᵀrhs + z)                                          # = K_a⁻¹b_a + R⁻¹z: same function of z as precision_draw
 5  min_pivot ← min_j (R_jj/‖A_·j‖)²;   ok ← diag(R) > 0 ∧ a finite
 6  a ← a ⊙ cmask;   cpl ← (Σcmask, ½Σ cmask·P̃(a − m)²) if λ-scaled else 0
```

The reason for step 3, and a finding: forming `ZᵀWZ` squares the condition number. A Geweke chain
of `reduced_sv` reached `h ≈ 31–37` in equation 0, which makes one regressor about 1e8 times the
others. Equation 2's `a`-row precision then had exact eigenvalues 14.8 and 1.9e17
(`cond = 1.3e16`), and the Cholesky hit a zero pivot. The posterior itself is benign. The QR draw
matches the exact conditional there to 1.6e-8 sd and agrees with the Cholesky draw to ≤ 3e-14 on
ordinary states. NOTES 10 marks this verified via `tests/mcmc/blocks/test_contemp_lstsq.py`
(`::test_g_full_reduced_sv_failing_chain_is_fixed`, `::test_extreme_scale_state`,
`::test_lstsq_draw_equals_the_cholesky_draw`). It is a methods point: the alternative was to narrow
the test prior's `h` tail, which "would have hidden a real numerical weakness" (NOTES 10).

### 6.4 Algorithm 5: the CCCM reduced-form step

**[code]** `components/mean/linear_reduced.py` (`cccm_core`, module docstring); DESIGN §3.7.3; N04
§5.3. It is a pre-phase `GlobalStep` (`Block.G_CCCM`, reads `glob.mean.Pi`, `eq.struct.a`, `eq.vol`;
writes `glob.mean.Pi`; `marginal_of_indicators = True`).

```text
Algorithm 5  CCCM(Π, a, inv_lam, X, Y, XX, mean0, prec0, Z)    # one corrected-triangular scan
 1  A ← I + strict-lower(a);   inv_lam_{i,t} ← m_t·exp(−log λ_eff,i,t)   (masked before the exp)
 2  W ← inv_lamᵀ(A ∘ A)                         # (T_pad, n): W_tj = Σ_i a_ij²/λ_it = (Σ_t⁻¹)_jj
 3  for all j in parallel (precompute):          # K_j does not depend on Π
        K_j ← (W_·jᵀ XX)[pos] + diag(prec0_j);   (L_j, s_j) ← Jacobi-scaled chol(K_j)
 4  E ← (Y − XΠ)Aᵀ                               # recomputed from scratch each sweep (no drift)
 5  for j = 0..n−1  (lax.fori_loop, dynamic j):
 6      f ← Xπ_j;   G ← (E ⊙ inv_lamᵀ)A_·j      # (Σ_t⁻¹u_t)_j
 7      b ← prec0_j ⊙ mean0_j + Xᵀ(G + W_·j ⊙ f)
 8      π_j′ ← s_j ⊙ L_j⁻ᵀ(L_j⁻¹(s_j ⊙ b) + Z_j)     # ~ N(K_j⁻¹b, K_j⁻¹)
 9      d ← X(π_j′ − π_j);   E ← E − d A_·jᵀ;   Π[:, j] ← π_j′    # rank-1 update in time
10  ok ← all factorisations ok ∧ Π finite
```

This is the textbook SUR Gibbs conditional with `Ω_t = AᵀΛ_t⁻¹A`, so the scan is block
Gauss–Seidel on the joint precision of `vec Π` (Fox & Parker 2017 [CITE?]; N04 §9.2). `A` is never
inverted (`linear_reduced.py` docstring). `cccm_precompute` is the shape rule
`n·k²·8·C ≤ 512 MiB`, which is reproducible (DESIGN §10.5). The noise row `Z_j` comes from
`sub_key(eq_key(step_key, j), Sub.Z)`: CCCM folds `j` before `Sub.Z` (`cccm_noise`). **[code]**.
The CCM (2019) step that drew `π_j` from equation `j` alone is exact only for `j = n−1` (N04 §4).
Mutant M1 injects it; the dossier on validation covers how it is caught (NOTES 4).

### 6.5 Algorithm 6: the volatility block (transitory ∘ persistent, atomic)

**[code]** `components/vol/robust.py` (`VolBlockImpl.sample_eq`, module docstring); DESIGN §3.7.7,
§4.4. This is the only `VolBlock` in Phase 2.

```text
Algorithm 6  VOL_SAMPLE_EQ(key, vs, e, tmask, T_eff, prior_i, cpl)
 1  log e²_t ← where(m_t > 0, log(where(m_t > 0, e_t², 1)), 0)     # padded rows never see log 0 (E23)
 2  trans ← transitory.sample(key, trans, log e², core.log_var(core), …)   # o → p → ν → τ given CURRENT persistent λ
 3  log ẽ²_t ← where(m_t > 0, log e²_t − (2 log o_t + log τ_t), 0)  # cleaned residual
 4  core, info ← core.sample(key, core, log ẽ², tmask, T_eff, prior_i, cpl)  # Constant: Alg. 8;  SV: Alg. 7
 5  ok ← info.ok ∧ every float leaf of trans finite
```

Nothing runs in between. The indicators, which are local to the SV core, are therefore drawn
immediately before `h` (DESIGN §4.8 invariant 2). An exact zero residual in a real row gives
`log e² = −∞`. The transitory logits handle it (`z² = 0`), the Constant core gets a zero
contribution, and the SV core absorbs it through its offset (`y* = log c`). In `zero_offset` parity
mode the row is instead masked and counted in `acc_floor` (N02 §2.4). **[code]**.

### 6.6 Algorithm 7: the SV core (AR), with exact correction and ASIS

**[code]** `components/vol/sv.py` (module docstring; `SVCore.h_move`, `nc_move`, `sample_core`,
`c_step`, `nc_step`, `awol_system`, `w_weight`); DESIGN §3.7.4, D-1…D-7. The step labels (c), (a),
(b), (b**) follow DESIGN §3.7.4.

Model: `h_j = μ + φ(h_{j−1} − μ) + ση_j`. State `j = t + 1` corresponds to data row `t`, and `j = 0`
is `h_0`. `h_0 ~ N(μ, σ²/ω₀)` with `ω₀ = 1 − φ²` (stationary) or `1/B₀`. Priors (D-13):
`μ ~ N(b_μ, B_μ)`, `(φ+1)/2 ~ Beta(a_φ, b_φ)`, `σ² ~ B_σχ²₁` or `IG(c₀, C₀)`. The mixture is OCSN-10
with effective means stored once (D-3; `components/vol/mixture.py`).

```text
Algorithm 7  SV_UPDATE(noise, (h, μ, φ, σ), log ẽ², tmask, T_eff, prior)   # noise injected (SVNoise)
    m_t ← tmask_t;   y*_t ← logaddexp(log ẽ²_t, log c_i) on observed rows, 0 elsewhere     # D-2: c_i = 1e-12·ψ_i
 (c) indicators
 1  r_t ← y*_t − h_{t+1};   ℓ_tj ← log p_j − ½log v_j − (r_t − m_j)²/(2v_j)
 2  s_t ← inverse-CDF categorical(u_t, ℓ_t·)  (one uniform per row; max-subtracted; ≤ comparison; clamp J−1)  # D-4
 3  keep log g(r_t) = logsumexp_j ℓ_tj − ½ log 2π                 # reused for w(h_cur): exact-mode lever
 (a) log-volatility path (non-centred AWOL system, Q̃ = σ²Q, h = μ + σx̃)
 4  D_t ← m_t / v_{s_t}
    Q̃_00 = ω₀ + φ² (stationary: 1);  Q̃_jj = σ²D_{j−1} + 1 + φ² (1 ≤ j < T_eff);  Q̃_{T_eff,T_eff} = σ²D + 1;
    Q̃ = 1 beyond T_eff;  Q̃_{j,j−1} = −φ (j ≤ T_eff, else 0);  b̃_0 = 0;  b̃_j = σD_{j−1}(y*_{j−1} − m_{s} − μ)
 5  x̃′ ← TRIDIAG_SAMPLE(Q̃, b̃, z_h)  (bidiagonal Cholesky fused with the forward solve in one lax.scan,
                                     back substitution in a reverse scan; log|Q̃| and min pivot for free)
 6  h′ ← μ + σx̃′
 7  if exact:  w(h) = Σ_t m_t[−½(log 2π + h_{t+1} + exp(log ẽ²_t − h_{t+1})) − log g(y*_t − h_{t+1})]
               accept h′ iff log u_h < w(h′) − w(h)        (DNP 2015 Alg. 3)           # D-1
               acc_rej_run ← (acc_rej_run + 1)(1 − accept)   (s stays as drawn on rejection)
 (b) centred step on h
 8  (γ′, φ′) ~ N(b_T, σ²B_T) from centred cross-products (stable 2×2 algebra), aux prior B₀ = diag(1e12, 1e8)
    independence MH;  |φ′| ≥ 1 ⇒ reject                                                 # D-6
 9  C_T ← ½[Σ m(h_j − μ − φ(h_{j−1} − μ))² + ω₀(h_0 − μ)²]
    B_σχ²₁: σ²′ = C_T/G, G ~ Gamma(T_eff/2), accept with log R = (σ² − σ²′)/(2B_σ);  IG prior: exact IG draw
 (b**) non-centred (ASIS) step                                                          # D-5
10  h̃ ← (h − μ)/σ;   φ′ ~ N(Σh̃_{j−1}h̃_j/Σh̃²_{j−1}, 1/Σh̃²_{j−1}), MH on the prior and the h̃_0 term
11  (μ′, σ′) ~ exact 2×2 Gaussian regression of y*_t − m_{s_t} on (1, h̃_{t+1}), weights 1/v_{s_t},
               prior N((b_μ, 0), diag(B_μ, B_σ))             (σ is unrestricted in sign here)
12  exact: ONE accept/reject with log α = log α_aux + w(μ′ + σ′h̃) − w(h)   (log α_aux = 0 for B_σχ²₁)   # E6
    w(h) is the value after step 7 (a stale w is a bug, pilot §3.4)
13  on accept: if σ′ < 0 then (σ′, h̃) ← (−σ′, −h̃)   (measure-preserving fold; μ + σh̃ unchanged)        # D-7
               h ← μ′ + σ′h̃;   on reject keep (μ, σ, h) and the φ draw
14  padded states j > T_eff ← μ + σ·z_h,j  (decoupled dummies, refreshed every sweep)
15  ok ← AWOL pivot > 0 ∧ C_T > 0 ∧ σ > 0 ∧ |φ| < 1 ∧ 2×2 NC regression PD ∧ new state finite
```

Points for the paper (all **[code]** unless marked):

- **Exactness.** With `exact=True` the `h` draw and the NC `(μ, σ)` move are MH steps whose
  invariant law is exact for any mixture table and any offset `c ≥ 0`, in the Gaussian-`ẽ` form
  (D-1; N01 §10.2–10.3). The ASIS NC correction is derived in N01 §10.3, not published. DESIGN
  §12.1 lists it as risk R1, gated by toy quadrature, MH-ratio identities, Geweke with
  `exact=True` and mutant M7.
- Exact mode passes a Geweke test with a deliberately **coarse one-component mixture**, where
  `exact=False` fails (z ≈ 10). With OCSN-10, Geweke cannot see the mixture error at CI sizes.
  Pilot §3.4 records this on the prototype. For the production code NOTES 3 marks it verified with
  the mutation suite as evidence (`tests/mutation/mutants.py`: M11/M12 caught under `exact=False`,
  M7 caught with `COARSE1`; re-run on x86-64 Linux 2026-10-22). Separately (not cited by NOTES 3),
  `tests/mcmc/blocks/test_sv_core.py::test_block_geweke_exact_mode_with_coarse_mixture` (marker
  `geweke`) asserts that exact mode passes and `exact=False` and M7 fail with the coarse table
  (corrected by fact-check: the dossier attributed this test to NOTES 3).
- **The fold (D-7).** stochvol takes `|σ|` in the NC step without flipping `h̃`. That is a
  documented oracle deviation (NOTES 1, **[recorded]** for stochvol 3.2.1, **[to check]** against
  the current release). **[verified]** that the fold is measure-preserving and the identities
  hold: `test_sv_core.py::test_fold_is_measure_preserving_map`,
  `::test_nc_mu_sigma_log_accept_identity`, `::test_nc_accept_uses_w_of_the_post_h_step_state`,
  `::test_block_geweke_catches_abs_sigma_without_flip`.
- **Cost of exact mode.** Two extra `(T_pad, J)` `logsumexp`s per equation and sweep, at `h′` and
  at the NC proposal (DESIGN §3.7.4). Prototype measurements: +12% to +22% of the full sweep and
  ≤ 10% of ESS (pilot §1 decision 1, §3.3). The production kernel `sv_exact` takes 0.363 ms for
  80 lanes on the M5 Max (§13.5). **[recorded]**
- **Exact-`h` acceptance at `T = 300`** with OCSN-10: prototype 0.973 / 0.983 / 0.979 for three
  `(φ, σ)` settings, worst chain 0.925, longest rejection run ≤ 5 (pilot §3, **[recorded]**;
  KSC-7 gives 0.75–0.82). Production band test:
  `test_sv_core.py::test_exact_h_acceptance_at_T300` **[verified]**.
- **Cold-start defect found by the pilot.** With `h ≡ μ` and a rejected first exact `h` step,
  `C_T = 0`, the σ² proposal is the point 0 and is accepted, and σ = 0 is absorbing. No status code
  fires. This hit 3–6% of chains with OCSN-10 and 34–59% with KSC-7 (pilot §3.1, **[recorded]**).
  The fix: one unconditional AWOL draw of `h` after the initial values (`SVCore.init`, which calls
  `awol_system` + `tridiag_precision_sample`; `sv.py` module docstring "Cold start (DESIGN §5.4 plus
  the pilot §3.1 amendment)"; recommended in pilot §3.1 and §11). DESIGN §5.4 itself was **not**
  amended: it still lists `h ≡ μ_i` with no AWOL draw (corrected by fact-check: the dossier cited a
  "DESIGN §5.4 amendment" that does not exist in the committed DESIGN.md; see Open questions).
  `C_T > 0` and `σ > 0` are now part of `StepInfo.ok` (step 15; `sv.py` docstring "Failures").
  **[code]**

### 6.7 Algorithm 8: the random-walk SV path, factored as a path Laplacian

**[code]** `sv.awol_rw_laplacian`, `linalg.laplacian_precision_sample`; DESIGN §3.7.4 "RW path";
NOTES 17.

For `rw=True`, `h_j = h_{j−1} + ση_j` and `h_0 ~ N(m₀, V₀)`. The scaled centred system is
`Q̃ = σ²Q = Lap_w + diag(δ)`: a path-graph Laplacian with unit edges on the live transitions, plus
`δ_0 = σ²/V₀` and `δ_j = σ²D_j`.

```text
Algorithm 8  LAPLACIAN_SAMPLE(δ, w, b, z)       # Q = Lap_w + diag(δ):  Q_tt = δ_t + w_{t−1} + w_t,  Q_{t,t−1} = −w_{t−1}
 1  q_0 ← δ_0;   q_t ← δ_t + w_{t−1} q_{t−1}/(w_{t−1} + q_{t−1})   (q_t = δ_t after a zero edge)   # pivot less next edge
 2  p_t ← q_t + w_t;   ℓ_t ← √p_t;   c_t ← −w_{t−1}/ℓ_{t−1}                              # all terms ≥ 0: no cancellation
 3  forward:  v_t ← (b_t − c_t v_{t−1})/ℓ_t;   backward: x_t ← (v_t + z_t − c_{t+1}x_{t+1})/ℓ_t
 4  log|Q| ← Σ_t [log w_t + log1p(q_t/w_t)]  (log q_t where no next edge);   min pivot ← min p_t
```

The finding (NOTES 17, verified on Apple M5 Max). In the Clark–Ravazzolo ordering stage (SV(rw),
reduced form, origin 1987Q2), GDP's volatility reached `σ = 4.98e-9` at burn-in sweep 318. That is
a legitimate state: the NC prior `σ ~ ±N(0, B_σ)` has positive density at 0, and the NC move of
the same sweep returned σ to 0.031. But once the scaled system is assembled, `σ²D ≈ 1e-17` lies
below the ulp of the Laplacian's diagonal (2). The matrix becomes the singular path Laplacian, and
the level direction's pivot (`≈ σ²(1/V₀ + ΣD)`, of order 1e-15) comes out 0. The run aborted with
status 2001, which `decode_error` maps to `(EQ_VOL, equation 0)` (command 1, §0.3). Carrying pivots
as their excess over the next edge is the same factorisation without the cancellation. It agrees
with the old draw to about 1e-13 at ordinary σ, and every non-RW composition stays bitwise
unchanged. Evidence: `tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma` (60-digit
reference), `::test_rw_step_at_a_tiny_sigma_is_ok`, `::test_rw_integrated_likelihood_at_a_tiny_sigma`,
`::test_awol_rw_laplacian_is_the_rw_system`. **[verified]**. The fix removes the crash, not the slow
mixing of the centred σ² step near σ → 0 (NOTES 17).

### 6.8 Transitory components

**[code]** `components/vol/transitory.py` (module docstring); DESIGN §3.7.6, D-8, D-9; N03.
`log z²_t = log e²_t − log λ^pers_t`.

| Kind | Steps, in order | Conditionals (all exact) |
| :--- | :--- | :--- |
| `outlier` | `o \| e,h,p` → `p \| o` | `ℓ_tk = log w_k − log k − ½z²_t/k²`, `w_1 = 1 − p`, `w_k = p/(K−1)`, `K = 20`; `p ~ Beta(a_p + N₁, b_p + T_eff − N₁)` |
| `t` | `ν \| e,h` (τ integrated) → `τ \| e,h,ν` | grid logits `log π_g + T_eff A_g − ((ν_g+1)/2)Σ_t m_t log1p(r²_t/c_g)` on `ν ∈ {3..40}`; `τ_t ~ IG((ν+1)/2, (c(ν) + r²_t)/2)`, `c(ν) = ν − 2` (standardized) |
| `outlier_t` | `o \| e,h,ν,p` (τ int.) → `p \| o` → `ν \| e,h,o` (τ int.) → `τ \| e,h,o,ν` | collapsed pmf `log w_k − log k − ((ν+1)/2) log1p(z²/(c k²))` |

Numerics (pilot decisions 12–13, adopted in the code docstring):

- The ν sums use `log(c_g + r²) − log c_g` with `r²` computed once per row; rows with
  `log r² > 700` use `log r²`.
- Gaussian outlier logits clamp `log z²` at 700.
- Padded rows hold `o = 1`, `τ = 1` and are excluded from every sum.

The τ draw (one log-gamma per row) dominated SV(t). `jax.random.loggamma` under `vmap` runs one
batched rejection loop over all `C·n·T` elements. The engine's vectorised Marsaglia–Tsang sampler
(§6.10) cut the prototype SV(t) sweep from 19.6 to 9.9 ms (pilot §1 decision 12, §5;
**[recorded]**, prototype, loaded M1 Pro).

### 6.9 Constant core and the coupling channel (D-30)

**[code]** `components/vol/constant.py`; DESIGN §3.7.5, §4.7.

```
λ_i | · ~ IG(c_i + T_eff/2 + n_active/2,  C_i + ½Σ_t m_t ẽ²_t + quad),   log λ = log(scale′) − loggamma(shape′)
```

`Coupling(n_active, quad)` is returned by the mean or contemporaneous step and consumed by the vol
step of the same equation. It is zero in Mode I. In Mode II (`Constant("iw")` × structural, the
Phase 1 NIW twin at `cross = κ_a = 1`) it is `(k + i, ½(θ − μ₀)ᵀP̃₀(θ − μ₀))`. For
`iw_equivalent` (Chan 2022 Prop. 1 [CITE?]) it covers the `a`-slots only. This is "the one
sanctioned prior–volatility interaction" (DESIGN §4.7), legal only with a Constant core (static
check, plan invariant 5). It is also the channel Phase 3 CommonSV with conjugate Σ scaling is
designed to use (DESIGN §3.9). **[design]** for Phase 3.

### 6.10 Random-variate kernels

**[code]** `engines/mcmc/random.py`:

- `uniform_tiny`: `U[tiny, 1)`, for every uniform whose log is taken (`log u ≥ −708.4`).
- `categorical_inverse_cdf_lse` (D-4): one float64 uniform per row on max-subtracted weights, with
  a `≤` comparison and a clamp to `J−1`. It also returns the row `logsumexp`, which the exact SV
  correction reuses. DESIGN D-4 records it as 3.5–4.4× faster than Gumbel-max (N07 §9.2,
  **[recorded]**).
- `loggamma`: Marsaglia & Tsang (2000) [CITE?] with two pre-drawn candidates per element and a
  `while_loop` for elements that rejected both. The first accepted candidate of an i.i.d. sequence
  is the rejection sampler, so it is exact. For `a < 1`, `log G_a = log G_{a+1} + log U/a` in log
  space. Pilot §5 records it as 5.8× faster than `jax.random.loggamma` under `vmap`, with KS
  p-values 0.52–0.67 at four shapes (**[recorded]**).
- `log_beta` via two log-gammas with `softplus`, stable for `a, b ≪ 1`. `truncated_std_normal` uses
  a log-space inverse CDF with Newton refinement and `log_ndtr(series_order=10)`, because the
  default order-3 series is off by 4e-9 at `x = −20`.

---

## 7. Randomness: injected noise and key layout (D-16, D-17)

### 7.1 Key derivation

**[code]** `engines/mcmc/keys.py` (module docstring and functions); DESIGN §5.6:

```text
root      = random.key(seed, impl="threefry2x32")                  # pinned: jax_default_prng_impl cannot change it
salt      = crc32(VARData.fingerprint()) & 0xFFFFFFFF              # windows and orderings get distinct streams
fit_root  = fold_in(fold_in(root, Stream.MCMC), salt)
chain_key = fold_in(fit_root, c)                                    # c = 0..C−1, prefix-stable
step_key  = fold_in(fold_in(chain_key, it), Block.X)               # it: uint32 ABSOLUTE sweep index, burn-in included
eq_key    = fold_in(step_key, i)                                    # i: position in the ordering
sub_key   = fold_in(eq_key, Sub.Y)                                  # one consumer per key
init      : fold_in(fold_in(fold_in(root, Stream.INIT), salt), c) → Block → i → Sub
forecast  : fold_in(fold_in(key(fseed), Stream.FORECAST), c) → d (kept-draw index) → Ahead.{VOL,SHOCK,MEAN,GLOBAL} → i
validation: fold_in(key(seed), Stream.VALIDATION) → tier → composition id → chain or replication
```

Append-only enums (renumbering changes every stream):

- `Stream`: `MCMC=0, INIT=1, FORECAST=2, VALIDATION=3, IRF=4`.
- `Block`: `EQ_MEAN=0, EQ_CONTEMP=1, EQ_VOL=2, EQ_COEF_LOCAL=3, G_CCCM=16`; members already defined
  for later phases: `G_COEF_GLOBAL=17, G_COMMON_SV=18, G_LP_SCALE=19` (Phase 3) and
  `G_REGIME=32 … G_B0_ROWS=36` (Phases 4–5). The `Block` docstring reserves the ranges 17–31 for
  Phase 3 global steps and 32–63 for Phases 4–5 (corrected by fact-check: the dossier gave 17–19 and
  32–36 as the reserved ranges).
- `Sub`: `S=0, H=1, H_ACC=2, C_GP=3, C_GP_ACC=4, C_S2=5, C_S2_ACC=6, NC_PHI=7, NC_PHI_ACC=8, NC_MS=9,
  NC_MS_ACC=10, O=11, P=12, NU=13, TAU=14`, with `Z=0` an alias of `S` (mean, contemp and CCCM
  use only `Z`).
- `Ahead`: `VOL=0, SHOCK=1, MEAN=2, GLOBAL=3`.
- `Tier`: `U..P = 0..6`.

Consequences, by construction:

- **No key is ever carried in a loop state.** Keys depend only on `(seed, salt, chain, it, block,
  equation, sub-draw)`, so draws cannot depend on chunk size, burn piece size or thinning alignment
  (N07 §4.2 rule 1).
- **The backtest passes a per-origin fit seed** in addition to the data salt. The salt alone keeps
  chain noise independent across windows even when a user fits several windows with one seed, as
  `ordering_sensitivity` does (DESIGN §5.6 "Seeds"; §11.6). **[design]**
- **Forecast randomness is keyed by the kept-draw index `d`**, not by store position. So in-loop
  and post-hoc forecasts agree whatever `store_thin` is. Replicate `r ≥ 1` of a draw uses the
  virtual index `d + r·D_kept` (`forecast.py` docstring; `storage.forecast_selection`).
  **[verified]** `tests/mcmc/runner/test_in_loop_forecast.py::test_forecast_key_is_the_kept_draw_index_with_store_thin`,
  `::test_path_keys_stride_by_the_kept_draws_with_store_thin`.

### 7.2 uint32 counters (erratum E7)

Under x64, `it0 + j·thin + r` would promote to int64. Every counter is computed from uint32
operands with explicit casts. `keys.step_key` **raises** on a non-uint32 sweep counter instead of
wrapping (`keys._counter`). The host asserts `burn + ⌈draws/chunk⌉·chunk·thin < 2³²` before running
(`runner.run_chains`). **[verified]** `tests/mcmc/runner/test_sweep.py::test_counter_must_be_uint32`.

### 7.3 Injected noise (D-17)

Pattern: `sample(key, …) = sample_core(draw_noise(key, …), …)`. Unit tests drive the cores with
fixed noise against NumPy dense references (DESIGN §3.1 rule 2). Example, the SV core's `SVNoise`
(`components/vol/sv.py`), with one sub-key per field:

| field | shape | key |
| :--- | :--- | :--- |
| `u_s` indicator uniforms | `T_pad` | `Sub.S` |
| `z_h` AWOL normals | `T_pad + 1` | `Sub.H` |
| `u_h` exact-`h` accept | scalar, `U[tiny,1)` | `Sub.H_ACC` |
| `z_gp`, `u_gp` centred `(γ, φ)` | 2, scalar | `Sub.C_GP`, `C_GP_ACC` |
| `lg_s2`, `u_s2` σ² proposal, accept | scalar | `Sub.C_S2`, `C_S2_ACC` |
| `z_phi`, `u_phi` NC φ | scalar | `Sub.NC_PHI`, `NC_PHI_ACC` |
| `z_ms`, `u_ms` NC `(μ, σ)` | 2, scalar | `Sub.NC_MS`, `NC_MS_ACC` |

Every exact draw also exposes its conditional (`*_conditional` returning `GaussianPrec(K, b)`,
IG `(shape, scale)` or categorical logits), and every MH step exposes its log acceptance ratio
(DESIGN §3.1 rule 3). Density-ratio identity tests (Grosse & Duvenaud 2014 [CITE?] eq. 1) and the
failure replay (§9) are built on these functions. **[code]**. The identities are **[verified]**,
for example `test_sv_core.py::test_exact_h_log_accept_identity`,
`::test_centred_gamma_phi_log_ratio_identity`, `::test_sigma2_log_ratio_identity_and_ig_gibbs`.

### 7.4 One consumer per key, and a threefry hazard the key-reuse checker cannot see

- A full sweep runs under `jax.debug_key_reuse(True)` (DESIGN §5.6). **[verified]**
  `tests/mcmc/runner/test_sweep.py::test_key_reuse_checker_passes_a_full_sweep` (with the runner
  toys, `reduced_sv` toy configuration, one sweep; scope added by fact-check). Also
  `tests/mcmc/blocks/test_sv_core.py::test_key_derivation_uses_one_sub_key_per_consumer` and
  `tests/mcmc/blocks/test_robust.py::test_no_key_reuse`.
- **Finding (not in the NOTES ledger yet).** Under partitionable threefry
  (`jax_threefry_partitionable = True`, the default in jax 0.12.0.1 per command 1, §0.3), the
  words that `jr.normal(key, (H,))` consumes are `threefry(key, (0, j))`. These equal the key data
  of `fold_in(key, j)`. So a kernel that drew directly from a parent key whose children are derived
  by `fold_in` would put `η_12` on the `Sub.O` key (value 11) and `η_15` on the `Sub.TAU` key
  (value 14) once `H ≥ 12` / `H ≥ 15`. `debug_key_reuse` cannot detect this, because `fold_in` is
  not a consuming use. The engine therefore gives the persistent core `fold_in(key, Sub.H)` in
  `draw_ahead_eq` and `transition_eq` (`components/vol/robust.py` docstring).
  **[verified]** `tests/mcmc/blocks/test_robust.py::test_ahead_and_transition_give_the_core_its_own_sub_key`.
  Under partitionable threefry the test asserts both that the hazard is real
  (`raw[Sub.O] == key_word(sub_key(k, Sub.O))`) and that the production layout avoids it.
  Candidate NOTES entry (see Open questions).

### 7.5 Philox vs threefry

The pilot measured philox4x32 as 2.4× faster on the sweep's RNG work (0.70 → 0.29 ms per sweep),
6–8% of a full sweep (pilot §6, decision 7; **[recorded]**, prototype, loaded M1 Pro). threefry is
kept: the gain is small, philox's statistical quality is untested here, and switching changes
every stream, so it would be a spec field. `rbg`/`unsafe_rbg` are forbidden: under `vmap` they
generate a whole batch from the first key (N07 §4.1). `keys._typed` raises on any implementation
other than `threefry2x32`. **[code]**

---

## 8. Runner, storage and initialisation

### 8.1 Executables

**[code]** `engines/mcmc/runner.py` (module docstring; `_make_runner`); DESIGN §5.1:

| Executable | Signature | JIT options |
| :--- | :--- | :--- |
| `prepare` | `(data, prior) → FitCache` (`F`, `FF` / `XX`, `G_const`, `FtY`, per-equation eigen/fixed-factor caches) | `jit` |
| `init` | `(init_keys[C], data, prior, cache) → ChainState[C]` | `jit` |
| `burn` | `(ckeys[C], state, data, prior, cache, it0: u32, n: i32) → state` | `jit`, state donated |
| `sample` | `(ckeys[C], state, …, it0: u32, thin: u32, stop: i32[, fkeys, d0, sel, D]) → (state, snap, out[C, chunk])` | `jit`, state donated, static `reps` |
| `extract` | `(state, data, prior, cache) → (monitor, ParamDraw, PathDraw \| Empty)` | `jit`, one chain |

`make_runner` is `lru_cache`d on `(StaticConfig, fit_axis, chains)` when no test overrides are
passed. Overrides bypass every cache (DESIGN §4.9). **[code]**

### 8.2 Algorithm 9: burn and sample bodies (one chain)

From `runner._bodies`:

```text
Algorithm 9a  BURN(ckey, st, data, prior, cache, it0: u32, n: i32)
 1  return fori_loop(r = 0..n−1, st ↦ SWEEP(ckey, it0 + u32(r), st, data, prior, cache))
     # traced trip count → lowered to `while`: no recompile across burn, warm_burn or piece sizes (D-27)

Algorithm 9b  SAMPLE(ckey, st, data, prior, cache, it0: u32, thin: u32, stop: i32 [, fkey, d0, sel, D])
 1  (st, snap), out ← scan over j = 0..chunk−1 (u32) with carry (st, snap ← st):
 2      base ← it0 + j·thin                                         # uint32 throughout (E7)
 3      st ← fori_loop(r = 0..thin−1, s ↦ SWEEP(ckey, base + u32(r), s, …))
 4      snap ← tree_where(j == u32(stop), st, snap)                 # exact state at the last kept draw
 5      rec ← EXTRACT(st)                                            # monitors, ParamDraw, optional paths
 6      if in-loop forecasts: rec ← (rec, cond(sel[j], FORECAST_CORE(params, d0 + j), zeros))   # §11
 7  return (st, snap, out)
```

`sel` is shared by all chains, so the `lax.cond` stays a real branch under `vmap` and unselected
draws skip the forecast core (`runner.py` docstring;
`tests/mcmc/runner/test_in_loop_forecast.py::test_unselected_draws_skip_the_forecast_core`).
**[verified]**

### 8.3 Algorithm 10: host driver

From `runner.run_chains`; DESIGN §5.1:

```text
Algorithm 10  RUN_CHAINS(fns, static, data, prior, settings, runtime, seed, salt [, init_state, plan])
 1  with x64, one OpenBLAS thread (§8.4), compile timer:
 2    check: static == fns.program.static; runtime.chunk == static.chunk; store mode; plan horizon;
             burn + ⌈draws/chunk⌉·chunk·thin < 2³²
 3    ckeys ← chain_keys(fit_root(seed, salt), C);   ikeys ← init_keys(seed, salt, C)
 4    cache ← prepare(data, prior);   state ← init(ikeys, …)  or the remapped warm state (§8.6)
 5    storage plan from the abstract record shapes (jax.eval_shape); raise StorageBudgetError before sampling
 6    BURN-IN, exact count: it ← 0; while it < burn:
         m ← min(burn − it, chunk·thin);   start ← host copy of state (if runtime.replay)
         state ← burn(ckeys, state, …, u32(it), i32(m));   CHECK(host(state), start, it, m);   it += m
 7    state ← zero_accumulators(state)                       # every acc_*, rb_* leaf (D-27, E8)
 8    for c = 0..⌈draws/chunk⌉−1:
         it0 ← burn + c·chunk·thin;   stop ← min(chunk, draws − c·chunk) − 1
         start ← host copy of state (if runtime.replay)
         (state, snap, out) ← block_until_ready(sample(ckeys, state, …, u32(it0), u32(thin), i32(stop)))
         CHECK(last ? host(snap) : host(state), start, it0, (stop+1)·thin);   store.append(out[:, :stop+1])
 9    final state ← snap of the last call (exactly burn + draws·thin sweeps); surplus sweeps never raise
10    acceptance ← acc_*/(draws·thin);   latent moments ← rb_*/(draws·thin)
```

`CHECK` reads the status codes and the state invariants of DESIGN §2.8: float64, `A` unit lower
triangular exactly, finiteness, inert padding, `σ > 0`, `|φ| < 1`, typed keys
(`state.check_state_invariants`). **[code]**.

**[verified]** in `tests/mcmc/runner/test_chunking.py`:
`::test_final_state_is_state_after_exactly_burn_plus_draws_thin_sweeps`,
`::test_accumulators_cover_exactly_the_sampling_sweeps`,
`::test_accumulators_with_thinning_cover_every_sampling_sweep` (with the test toys).

### 8.4 One OpenBLAS thread while the chains run (WP24 finding)

**[code]** `engines/mcmc/_blas.py`; `run_chains` wraps everything in `_blas.single_thread()`. On CPU,
jaxlib lowers `cholesky` and `triangular_solve` to LAPACK/BLAS custom calls into SciPy's library
(OpenBLAS on Linux/Windows wheels, Accelerate on macOS). XLA already runs those calls on its own
worker threads, and OpenBLAS additionally threads a level-3 call with busy-waiting workers. On a
machine whose cores are all busy, the hand-off costs whole scheduler time slices.

Recorded numbers (module docstring only; **[recorded]**, machine described only as "an idle 4-core
x86 machine" and "load 12"): in the BENCH-P2 `sample` executable, the monitor's `B = (A⁻¹Γ)ᵀ`
(four `20 × 20` solves with 81 right-hand sides per kept draw) took 9 ms per draw idle and 100 ms
per draw at load 12, against 0.7 ms single-threaded, while the sweep itself costs about 16 ms.

Claimed result invariance: below OpenBLAS's threading threshold, a triangular-solve column and a
Cholesky smaller than `128 × 128` use the same kernel sequence whatever the thread count. This was
"checked bitwise on BENCH-P2-shaped fits". At `k ≥ 128` the threaded Cholesky is blocked
differently, so single-threading makes such runs more reproducible across core counts, not less.
**[recorded]**. The committed tests check only set/restore semantics and thread safety:
`tests/mcmc/runner/test_blas_threads.py` (3 tests). The function is a no-op where no OpenBLAS with a
thread API is loaded (macOS Accelerate, MKL). In the dossier environment (Apple M5 Max, SciPy
1.18.1) `_blas.openblas()` returns `None` (command 5, §0.3), so the function does nothing there.
The reference baseline ran on macOS arm64 with "numpy 2.5.4 on Accelerate" (commit d490730 message;
`baseline.json` `machine.platform`), and the `_blas.py` docstring states that SciPy uses Accelerate
on macOS, so the baseline did not exercise the OpenBLAS path. **[recorded]** (corrected by
fact-check: was **[to check]**; the commit message records the environment. The SciPy version of
the reference run is not in `baseline.json`.) The real gap is the result-invariance claim itself
(Open question 19).

### 8.5 Storage and memory

**[code]** `engines/mcmc/storage.py` (module docstring; `plan_storage`, `HostStore`); DESIGN §5.3;
`settings.MCMCSettings`, `settings.Runtime`:

| Stream | Content | Stored |
| :--- | :--- | :--- |
| `monitor` | primary and secondary monitors (§8.1 of DESIGN; corrected by fact-check, was §8.2): log-likelihood, own first-lag `B`, SV parameters, `h_T`, `h_1`, `h_mid`, every `a_ij`, `log p`, `ν` | every kept draw |
| `params` | `ParamDraw`: `γ (n, k)` or `Π (k, n)`, `a`, `VolDraw` | every `store_thin = ⌈draws/store_max⌉`-th kept draw (`store_max = 2500`) |
| `paths` (opt-in `store="paths"`) | `h` and `2 log o + log τ` | every `path_thin`-th stored draw (default 10) |
| `forecast` (in-loop plan) | `ForecastOut` of the §7.1 selection | the selected draws |
| latent moments | mean/sd of `h`, `P(o_t > 1)` from `rb_*` | once |

- The host store holds preallocated NumPy arrays `(C, D, …)`, or `np.memmap` files under
  `Runtime.storage_dir` (**[verified]**
  `tests/mcmc/runner/test_storage.py::test_memmap_store_equals_memory_store`).
- `Runtime.memory_budget_gb` (default 2.0) only decides whether to raise before sampling, never what
  is computed (E14; `::test_run_raises_over_budget_before_sampling`).
- float32 storage is not offered (R4).
- `fit.posterior` is an `xarray.DataTree` with groups `posterior`, `monitor`, `sample_stats`,
  `latent_moments`, `constant_data` and attrs. `B = (A⁻¹Γ)ᵀ` is derived lazily, never stored.
- DESIGN sizes at `n = 20, T = 300`, 4 × 10,000: monitor ≈ 330 scalars ≈ 106 MB; params
  ≤ 4 × 2500 × 15 KB = 150 MB (DESIGN §5.3; **[design]**). The BENCH-P2 peak RSS was 3,043 MiB
  (≈ 3.19 GB; corrected by fact-check: the dossier wrote "3.04 GB", but `peak_rss_mb` is in MiB) on
  the M5 Max (§13.3), well above these stream sizes. Which part of the process dominates is not
  documented (Open questions).

### 8.6 Initialisation and warm starts

- **Cold start** (DESIGN §5.4, with the pilot §3.1 amendment, which is recorded in `SVCore.init` and
  the `sv.py` docstring but not in DESIGN §5.4 itself (corrected by fact-check); `runner._init_fn`,
  `SVCore.init`; **[code]**). First `a_i = 0`. Then one exact equation draw with provisional weights
  `m_t·exp(−log E λ_i)` from the volatility prior (`runner.initial_log_variance`: Mode I `ψ_ι/ι`;
  SV `b_μ` or `m₀`). Then the vol block's `init_eq`, which for SV ends with the unconditional AWOL
  draw. The init stream is `Stream.INIT`, which has no sweep index.
- **Warm starts** (backtests; opt-in, D-32; DESIGN §5.5; `engines/mcmc/warmstart.py`; **[code]**).
  Parameters are copied. `h`, `o` and `τ` are aligned by period label. New trailing periods get `h`
  forward-simulated and `o = τ = 1`. The state is re-padded to the new `T_pad`, and status and
  accumulators are reset. Burn-in still runs (`warm_burn = max(250, burn // 4)`), and the PRNG
  starts at `it = 0` with the origin's own seed and salt. `warm_start=True` requires strictly
  increasing origins (DESIGN §2.10 Stage 4 amendment (3), R9). **[verified]**
  `tests/mcmc/runner/test_warmstart.py` (expanding, rolling, bucket change, RW `h`, incompatible
  composition, `warm_burn`).

---

## 9. Failures as values and deterministic replay (D-18)

**[code]** `engines/mcmc/linalg.py` (pivot conventions), `sweep.record_status`,
`state.encode_error`/`decode_error`, `runner._Run` (`check`, `replay_to`, `failure`, `nonfinite`),
`runner.failure_detail`, `_equation_detail`, `_global_detail`; DESIGN §5.7.

```text
Algorithm 11  FAILURE PATH
  inside JIT (every block):
 1  StepInfo(ok, min_pivot): ok ⇔ factorisation pivots finite and > 0 ∧ draws finite (∧ block-specific checks);
    a NaN pivot counts as 0, so min_pivot ≤ 0 exactly when a factorisation failed, and the value stays finite
 2  sweep keeps the FIRST (code = 1 + 1000·block + eq, it) per chain and the running min pivot; nothing raises, nothing jitters
  on the host, after every executable call:
 3  if any status.code ≠ 0: pick the chain with the smallest failing `it`
 4      (block, eq) ← decode_error(code)
 5      before ← replay_to(start copy, it0, n = it_fail − it0)          # deterministic: keys depend on it only
 6      failure_detail: re-run the steps of sweep it_fail that precede the failing one (with that sweep's keys),
        then evaluate the failing block's EXPOSED conditional eagerly:
          equation/mean/contemp steps: K of conditional_eq / contemp_conditional → dimension, non-finite entries,
             non-positive diagonal, min eigenvalue of sKs, Cholesky status, diag-ratio bound max/min diag(L)² on cond₂,
             weight range min/max w, prior precision range;   vol step: residual range
          global steps (CCCM, which record equation 0): conditional_j for every j → the first failing j
             (equation_located = True) or the worst-conditioned one
 7      raise MCMCNumericalError(block, equation, iteration, detail, chain)      (picklable)
 8  elif state invariants are violated without a code: bisect the replay for the first offending sweep
```

- Replay costs one host copy of the start state of every executable call (≈ 0.5 MB per chain,
  DESIGN §5.1). `Runtime(replay=False)` disables it. **[code]**
- No jitter is ever added and no chain is dropped (DESIGN §5.7; SPEC §8 "Adding silent jitter is
  forbidden").
- **[verified]** `tests/mcmc/runner/test_failures.py` (with the runner toys `toys.DriftMean` /
  `toys.DriftCCCM`, except the last test listed, which uses the production CCCM step; scope added by
  fact-check). A non-PD prior is injected at sweep 7 and
  reported as `("EQ_MEAN", equation 1, iteration 7, chain 0)`, with code `1 + 1000·0 + 1`, a failed
  Cholesky, a negative minimum eigenvalue and an infinite diagonal ratio, in burn-in and in sampling
  (`::test_failure_in_burn_in_is_replayed`, `::test_failure_in_sampling_is_replayed`); without a
  replay copy only `(block, equation, iteration)` is checked and the detail says replay is disabled
  (`::test_failure_without_replay_copy`); a failure after the last kept draw is ignored
  (`::test_failure_after_the_last_kept_draw_is_ignored`). The other cases report differently
  (corrected by fact-check: the dossier said the same report "holds" for all of them): a global-step
  failure at sweep 7 is `("G_CCCM", equation 1, iteration 7, chain 0)`, code `1 + 1000·16 + 0`, with
  the failing column located by replay (`equation_located = True`;
  `::test_global_step_failure_is_located_by_replay`); an unflagged NaN is reported with block
  `"state"`, equation 1, iteration 7, found by bisection (`::test_unflagged_non_finite_state_is_located`);
  and with the production CCCM step the replay of a reduced-form `EQ_VOL` failure now works on host
  copies of the state (it raised `TracerArrayConversionError` before the fix, found in the full
  Clark–Ravazzolo run; `::test_replay_of_an_equation_failure_after_the_production_cccm_step`).
- **Field examples.** The two real failures in NOTES were both reported, not hidden, and both led
  to a better algorithm rather than a jitter. One was the Geweke chain whose `a`-row precision had
  `cond = 1.3e16` (NOTES 10, §6.3). The other was the Clark–Ravazzolo RW crash with status 2001 =
  `(EQ_VOL, 0)` (NOTES 17, §6.7).

---

## 10. Reproducibility matrix

DESIGN §5.6 (with the Stage 2 amendment of §5.2/§5.6) and NOTES 9. "Toys" means the runner test
doubles in `tests/mcmc/runner/toys.py`, which implement the block protocols but are not the
production kernels.

| Variation (same spec, data, seed, one device count) | Guarantee | Evidence | Status |
| :--- | :--- | :--- | :--- |
| `Runtime.chunk` (5 vs 8 in the test; DESIGN cites {50, 250}) and burn piece size | bitwise | `test_chunking.py::test_chunk_and_burn_piece_size_invariance` (toys) | verified (toys); production: follows from the key layout, **to check** |
| thinning with aligned sweeps | bitwise | `::test_thinning_with_aligned_sweeps` (toys) | verified (toys) |
| fused vs unfused plan execution | bitwise | `test_sweep.py::test_fused_equals_unfused_bitwise` (toys, 3 configs) | verified (toys) |
| `vmap` over chains vs a sequential loop | bitwise | `test_sweep.py::test_vmap_over_chains_equals_sequential_loop` (toys) | verified (toys) |
| chain-count prefix, `chains="map"` | bitwise | `::test_chain_prefix_invariance_with_mapped_chains`, `::test_mapped_chains_production_prefix` (production `structural_sv`), `::test_production_chain_prefix_3_of_5_with_vmapped_chains_to_rounding` (map part, SV-O-t) | verified |
| chain-count prefix, `chains="vmap"`, production SV / CCCM kernels | to rounding, `rtol 1e-9` (short runs) | `::test_production_chain_prefix_with_vmapped_chains_to_rounding`, `::…_reduced_form_to_rounding` | verified |
| `vmap` vs `shard_map(4)` | toys `rtol 1e-12` (`vmap`), bitwise (`map`); production `rtol 1e-9` | `test_devices_bitwise.py::test_shard_map_matches_vmap_in_subprocess`: toys, plus a production part in the same subprocess script (`joint_spec("structural_sv")`, 4 chains under `_sharded(…, 4, chains)` for both `"vmap"` and `"map"`, a 2-sweep burn, `assert_allclose(rtol=1e-9, atol=1e-12)`; for `"map"` bitwise equality is only printed, not asserted; the test also accepts `PRODUCTION-SKIPPED` if `compose` raises, which it does not at HEAD) | verified (toys; production `structural_sv` to `rtol 1e-9` on a short burn) (corrected by fact-check: the dossier said production was only **[design]**) |
| leading fit axis vs looped fits | `rtol 1e-10` | `test_runner.py::test_fit_axis_equals_looped_fits` (toys) | verified (toys) |
| in-loop vs post-hoc forecasts | 1e-12 | `tests/mcmc/forecast/test_forecast_core.py::test_in_loop_forecast_equals_post_hoc_sv`, `::…_constant`, `::…_with_thinned_storage` | verified |
| memmap vs in-memory store | equal | `test_storage.py::test_memmap_store_equals_memory_store` | verified |
| OpenBLAS thread count (factors `< 128 × 128`) | bitwise | `_blas.py` docstring | recorded |
| adding WP25 dummy code, `n_dummy = 0` | bitwise, 537 arrays | NOTES 16; `test_dummy_path_is_closed_without_dummies` | recorded (one run) |
| platforms (OpenBLAS vs Accelerate), JAX versions, PRNG impl, `jax_threefry_partitionable`, spec fields (`gram`, `const_path`, `t_bucket`) | none | DESIGN §5.6 | design |

Why `vmap` breaks bitwise prefix equality: chain `c` consumes the same random numbers whatever `C`
is, but XLA vectorises the reductions of the production SV and CCCM kernels differently for
different batch sizes. Over long runs, a last-bit difference can flip an accept/reject decision,
after which chains are equal only in distribution (DESIGN §5.6). This is why `Runtime.devices`
defaults to 1 and the backtest never passes a `Runtime`, so cached scores cannot depend on the
machine's device count (DESIGN §5.2). **[design]**. Note: N07 §6 reported a bitwise
`shard_map`-vs-`vmap` match on a snippet. The Stage 2 amendment supersedes that for the production
kernels.

Recorded in every fit (`engines/mcmc/results.py` `_metadata`, lines 140–192; DESIGN §5.6): `jax`/`jaxlib`
versions, `prng_impl`, `jax_threefry_partitionable`, platform, device count, chain method, local
device count, seed, salt, the resolved `gram`/`const_path`/`T_pad`, mixture, offset,
`exact`, ordering, spec hash (lines 186–189, when the model's spec serialises) and package version.
The `it` range is not in `_metadata`: it is written to the store attrs as `it_first = 0`,
`it_last = burn + draws·thin − 1` (`runner.run_chains`, line 1123), and `burn`/`draws`/`thin`/`sweeps`
are in `fit.final_state.seed_meta` (`results.py` line 247). **[code]** for all of these
(corrected by fact-check: the dossier gave lines 144–167 and said the rest was per DESIGN only).

---

## 11. Forecast core and the in-loop forecast hook

**[code]** `engines/mcmc/forecast.py` (module docstring; `forecast_core`, `simulate_paths`,
`make_forecast`); DESIGN §7.1–7.2, §5.8.

```text
Algorithm 12  FORECAST_CORE(draw, hist (p, n), t_last, shocks = (eps (H, n), vol_keys, mean_keys, glob_key))
 1  (log λ^pers_i, log λ^eff_i)_{T+1..T+H} ← vol.draw_ahead_eq(vol_keys[i], vd_i, H)
       SV: h_{T+j} = μ + φ(h_{T+j−1} − μ) + ση_j from h_T;  outliers: one uniform → o;  t: log τ = log(c(ν)/2) − loggamma(ν/2)
 2  scan h = 1..H, carry = lag window:
 3      x ← regressors_from_window(window, t_last + h, n_det)
 4      g ← [mean.predict_eq(md_i, x)]_i  (structural: Γx);   f ← mean.predict_reduced(gd, x)  (reduced: Π′x)
 5      m ← A⁻¹g + f   (= B′x; unit-triangular solve);   L ← A⁻¹ diag(exp(½ log λ^eff_{·,T+h}))
 6      y ← m + L ε_h;   window ← [window[1:], y]
 7  return (paths, cond_mean = m, cond_chol = L, cond_scale, log_lam_pers)
```

Given the draw, the path up to `T + h − 1` and the drawn `λ_{T+h}`, `y_{T+h} ~ N(m, LLᵀ)` exactly,
and `L` is the Cholesky factor of the conditional covariance at no extra cost (DESIGN §7.2).
Forecast draws are vmapped in chunks of `S_c = min(1024, 2^⌈log₂ S⌉)`, compiled once per
`(program, H, S_c)` (`forecast.py` docstring). **[verified]**
`tests/mcmc/forecast/test_forecast_core.py::test_recursion_equals_reduced_form_with_the_same_shocks`
(1e-12), `::test_chunked_executable_equals_the_core_and_checks_its_size`,
`::test_forecast_shocks_follow_the_key_layout`.

**In-loop hook** (DESIGN §5.8). `fit(data, plan=FitPlan(horizon, draws, seed))` compiles the
forecast core into the `sample` executable (Alg. 9b, step 6). It runs on exactly the kept states
that post-hoc forecasting would select (`storage.forecast_selection`: `⌈S/C⌉` evenly spaced stored
draws per chain, or every stored draw with `R = ⌈S/(C·D_s)⌉` paths), with the same keys. The hook
exists so that Phase 4 BART forests and Phase 5 TVP paths never have to be stored. **No Phase 2
model declares `requires_in_loop_forecast = True`, so the backtest never passes a plan in Phase 2**
(DESIGN §5.8 last sentence); a user can still pass `FitPlan` to `fit` on a Phase 2 model, which is
how the tests exercise it (corrected by fact-check: the dossier said "No Phase 2 model uses it"). It is exercised by tests: with toys in
`tests/mcmc/runner/test_in_loop_forecast.py` (14 test functions), and with the production forecast core in
`test_forecast_core.py::test_in_loop_forecast_equals_post_hoc_sv` and the related tests listed in
§10. **[verified]**

Cross-reference: the SPEC §7 path-conditioned Rao–Blackwellised log score breaks down in the tails
at `h > 1` (NOTES 8b, **[recorded]**). A path-integrated estimator is in progress outside this
snapshot. That belongs to the evaluation dossier, not the engine.

---

## 12. Compilation and caching

- **In-process cache.** `compose`, `make_sweep`, `make_runner` and `make_forecast` are
  `lru_cache`d on `StaticConfig` (value equality). A new window, new hyperparameters or new data
  in the same `T_pad` bucket reuse the executables (DESIGN §4.9; D-23). Component choices are
  Python branches at trace time. Data and priors are always arguments, never closed over. Closing
  over arrays would make them HLO constants and force a retrace per dataset (N07 §7 rule 2).
  **[code]**
- **Recompile guard.** Two fits on different windows of one bucket (`T_eff` 20 and 27, both
  `T_pad = 32`), with new data, `burn`, `thin`, `draws` and seeds, emit zero new backend
  compilations of `mcmc_*` executables. Detection uses `jax.monitoring` events and `jax_log_compiles`
  records (DESIGN §4.9; N07 §7.7). **[verified]**
  `tests/mcmc/runner/test_recompile.py::test_zero_recompiles_across_windows_of_one_bucket` (toys).
  The executables are named (`jit(mcmc_prepare)`, `jit(mcmc_init)`, `jit(mcmc_burn)`,
  `jit(mcmc_sample_chunk)`), so the filter can match them.
- **Persistent cache** (opt-in). `rai_bvar.enable_compilation_cache(path, min_compile_secs=1.0)`
  (`_devices.py`) sets `jax_compilation_cache_dir` and
  `jax_persistent_cache_min_compile_time_secs`. It must point to a trusted directory (N07 §7.6). CI
  uses a threshold of 0 so that small validation programs, which compile in under 1 s, are
  cached too (DESIGN §4.9; R12). **[code]**; in CI this is set through the environment
  (`JAX_COMPILATION_CACHE_DIR`, `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS: "0"`,
  `.github/workflows/ci.yml` lines 60–61 and 103–104), not through `enable_compilation_cache`
  (precision added by fact-check).
- **Expanding-window backtests** recompile about once every 32 origins at monthly frequency, and
  with the persistent cache only the first run pays (N07 §7 rule 5). **[design]**
- **Measured compile times.** Production BENCH-P2 (M5 Max): 3.46 s cold, 1.00 s from the
  persistent cache in a fresh process (§13.3, **[recorded]**). Prototype full sweep (N07 §7):
  1.5–2.8 s cold, 0.37 s cached (**[recorded]**, loaded machine).

---

## 13. Performance: BENCH-P2

### 13.1 Target definition

DESIGN §10.1; `benchmarks/phase2_perf.py` docstring:

- **Model**: `BVAR(Linear(Minnesota(form="structural", hierarchy="fixed")), SV(), lags=4,
  mcmc=MCMCSettings(chains=4, burn=1000, draws=10_000, thin=1))`. That is 4 × 11,000 sweeps.
- **Data**: the simulated stationary VAR(4)-SV DGP `benchmarks/pilot/pilot_common.simulate_var_sv(
  n=20, p=4, T=300, seed=0)`: own lag `l` coefficient `0.5/l`, small cross lags, spectral radius
  ≤ 0.9, unit lower-triangular `A`, AR(1) log variances with `φ = 0.95`, `σ = 0.2` (pilot §0). Shape:
  `k = 81`, `q = 100`, `P = 5050`, `T_pad = 320`, `B = 80` equation-chain lanes.
- **Measured**: end-to-end `BVAR.fit()` wall time, including compilation, storage and
  diagnostics, excluding data loading. Each fit runs in a fresh process. One repeat is a cold run
  (empty persistent cache) and a warm run (cache populated). The summary reports medians over 3
  repeats. ESS per second = bulk ESS / end-to-end wall time, taken over the warm runs.
- **Pass**: < 120 s on one device with default settings. **Stretch**: < 60 s with
  `set_cpu_devices(4)`.
- **Gate**: +20% time or −30% ESS/s against `tests/bench/baseline.json` on a matching machine fails
  (DESIGN §10.6; `phase2_perf.TIME_TOL`, `ESS_TOL`; `tests/bench/test_perf_phase2.py::test_bench_p2`,
  markers `bench`, `slow`). The 120 s assertion runs only on the reference machine.

### 13.2 Reference machine and protocol (as recorded)

Source: `tests/bench/baseline.json` (`machine`, `kernel_machine`, `label`, `settings`) and commit
`d490730` (message). **[recorded]**

| Field | Value as recorded |
| :--- | :--- |
| `label` | "Apple M5 Max (18 cores, 64 GB), macOS, plugged in, desktop apps open (load ~2), reference" |
| `cpu` / `physical_cores` / `perf_cores` / `logical_cpus` | Apple M5 Max / "18" / "6" / 18 |
| `mem_bytes` | `null`. The 64 GB appears only in the free-text label. |
| `platform` | macOS-26.6.2-arm64-arm-64bit |
| `python` / `numpy` / `jax` / `jaxlib` | 3.12.13 / 2.5.4 / 0.12.0.1 / 0.12.0.1 (NumPy on Accelerate, per the commit message) |
| `xla_flags` | empty |
| `commit` (code measured) | `deb3929` |
| `date` | 2026-10-22T21:33:01 (kernels 21:33:12) |
| `machine.loadavg` (1, 5, 15 min, at write time) | 6.26, 4.66, 3.56 |
| `summary.loadavg_1min` (1-min load before each of the 6 runs) | 2.36, 5.38, 6.12, 4.65, 5.08, 4.66 |
| commit message | "1-min load 1.95 before the run"; "desktop apps open" |
| `settings` | n 20, lags 4, T 300, chains 4, burn 1000, draws 10000, thin 1, devices 1, data_seed 0, sweep_block 1000, sweep_reps 3 |

The machine was **not idle**, although DESIGN §10.1 asks for "an idle Apple M-series laptop" and the
WP24 acceptance criterion (DESIGN §13 WP24) for "the idle reference laptop" (wording per section
corrected by fact-check). DESIGN §10.1 also specifies "8+
performance cores", while the JSON records `perf_cores: "6"`. Both are listed under Open questions.

### 13.3 End-to-end results (production engine, one device)

`tests/bench/baseline.json` → `summary`. **[recorded]** `wall_*` and `compile_*` are medians over the 3
cold or the 3 warm runs; `run_s`, `post_runner_s`, `runner_ms_per_sweep`, `sweep_ms_min3x1000` and
`peak_rss_mb` are medians over all 6 runs, cold and warm (`phase2_perf.summarise`) (corrected by
fact-check: the dossier said "medians over 3 repeats" for every row).

| Quantity (JSON field) | Value |
| :--- | ---: |
| wall time, cold cache (`wall_cold_s`) | **52.28 s** |
| wall time, warm cache (`wall_warm_s`) | **49.57 s** |
| compile, cold (`compile_cold_s`) | 3.46 s |
| compile, warm / cache load (`compile_warm_s`) | 1.00 s |
| run time = wall − compile (`run_s`) | 48.74 s |
| after the runner: fit assembly, diagnostics (`post_runner_s`) | 4.02 s |
| runner ms per sampling sweep, 4 chains (`runner_ms_per_sweep`) | 4.07 ms |
| ms per sweep, minimum of 3 × 1000 sweeps of the compiled `burn` executable (`sweep_ms_min3x1000`) | 3.90 ms |
| peak RSS (`peak_rss_mb`, computed as `ru_maxrss` bytes / 2²⁰, i.e. MiB) | 3,043 MiB ≈ 3.19 GB ≈ 2.97 GiB (corrected by fact-check: unit) |
| `pass_120s_cold` / `pass_120s_warm` | true / true |
| `converged` (6 runs) | false, false, false, false, false, false |

Derived (dossier arithmetic): `runner_ms_per_sweep` is measured over the 10,000 **sampling** sweeps
only (`runner.run_chains` timing: `(t_sample − t_host − sample_compile)/(draws·thin)`; `phase2_perf.py`
docstring), so 10,000 × 4.07 ms = 40.7 s is the measured sampling share; extrapolating the same rate
to the 1,000 burn sweeps gives ≈ 44.8 s of the 48.7 s run time, but the burn-in rate is not
recorded separately (corrected by fact-check: the dossier multiplied by 11,000 without this
caveat). The 120 s target is met with a margin of 2.3× cold. The one-device time is already below the 60 s
stretch target, which was defined for 4 devices. No production 4-device measurement exists
(§13.8).

### 13.4 Mixing efficiency at BENCH-P2

`summary.ess` (bulk ESS over 4 × 10,000 kept draws; ESS/s over the warm-run wall time; for each
group, the minimum and median over its quantities, then the median over runs). **[recorded]**

| Group | `ess_min` | `ess_per_s_min` | `ess_per_s_median` |
| :--- | ---: | ---: | ---: |
| log-likelihood (1 quantity) | 1,562 | 31.5 | 31.5 |
| `h_T` (20) | 1,870 | 37.7 | 134.6 |
| `μ` (20) | 3,835 | 77.4 | 165.9 |
| `σ` (20) | 504 | 10.2 | 13.7 |
| `φ` (20) | 554 | 11.2 | 17.3 |
| own first-lag `B` (`B_own1`, 20) | 19,997 | 403.4 | 494.5 |
| primary set (101) | 504 | 10.2 | 115.2 |
| cross-lag `B[y1.L1, y0]` (stored draws, `store_max = 2500` thins by 4; JSON fields `ess` and `ess_per_s` of one quantity, not a group minimum) | 8,968 | 180.9 | — |

`converged = False` in every run. The commit message records the reason: "4 of 101 primary
quantities miss R-hat < 1.01 (worst sigma[y7] 1.016, sigma[y1] 1.015)". **[recorded]**: the
R-hat values are in the commit message only. `phase2_perf._ess_summary` computes `rhat_max` but
`summarise` does not keep it in the baseline JSON. The recorded `ess_min` values are medians over
the warm runs of each run's minimum, and all are above `ess_min = 400`; the commit message gives
R-hat as the reason for `converged = False`. `test_perf_phase2.py` (line 155) reports `converged` but does not gate it.

This is consistent with pilot decision 6 (escalated, not resolved in this snapshot). φ and σ mix
at ESS ≈ 22–28 per 1,000 sweeps per chain (prototype). With the default `draws = 2000`
(`settings.MCMCSettings`), 4 chains give bulk ESS ≈ 180–220 < 400 and split-R̂ 1.03–1.06 (pilot
§3.3), so `converged = False` will be routine for SV fits at the defaults. BENCH-P2's 5× more
draws (10,000 vs the default 2,000; corrected by fact-check, was "10×") clear the ESS rule but still
miss R-hat on a few σ_i. The defaults are unchanged:
`draws = 2000`, `ess_min = 400`, `rhat_max = 1.01`, and AR SV `μ, φ, σ, h_T` are still primary
(`engines/mcmc/monitor.py` docstring). **[code]**

### 13.5 Kernel microbenchmarks (production kernels, 80 lanes, M5 Max)

`tests/bench/baseline.json` → `kernels` (AOT-compiled, `block_until_ready`, 50 repetitions, 5 for `sweep`; `sweep`
is the runner's compiled `burn` executable over 200 sweeps from the cold state, normalised per
sweep). Kernel definitions and budget lines are from the `tests/bench/bench_kernels.py` docstring.
**[recorded]**

| Kernel | Production code | `ms_min` | `ms_median` | compile s | XLA `flops` | bytes accessed | temp bytes | DESIGN §10.3 budget |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gram_packed` | `weighted_gram(mode="packed")` | 0.467 | 0.488 | 0.019 | 2.59e8 | 2.60e7 | 3.23e6 | 1.1 |
| `gram_batched` | `weighted_gram(mode="batched")` (oracle) | 1.050 | 1.168 | 0.017 | 5.17e8 | 6.09e7 | 2.69e7 | — |
| `precision_draw` | Jacobi-scaled Cholesky + 2 solves | 1.167 | 1.193 | 0.053 | 4.93e6 | 3.95e7 | 1.29e7 | 1.9 (with GEMMs) |
| `sv_indicators` | inverse-CDF indicators | 0.270 | 0.295 | 0.032 | 6.02e6 | 1.46e7 | 4.10e6 | } 1.5 |
| `sv_awol` | AWOL system + tridiagonal sampler | 0.141 | 0.149 | 0.079 | 5.15e5 | 8.22e6 | 1.44e6 | } |
| `sv_exact` | exact `h` correction: one extra `log g` + two `w` | 0.363 | 0.384 | 0.028 | 2.30e6 | 1.11e7 | 2.87e6 | 0.4 (pilot: 1.1) |
| `sv_core` | whole SV update | 1.802 | 1.888 | 0.520 | 2.62e7 | 6.38e7 | 6.16e6 | ≈ 3.6 per the `bench_kernels.py` docstring only; DESIGN §10.3 has no `sv_core` line, and its SV lines sum to 1.5 + 0.4 + 0.3 = 2.2 (note added by fact-check) |
| `sweep` | compiled `burn`, per sweep | 3.867 | 3.877 | 0.844 | 3.07e8 | 1.46e8 | 1.75e7 | 5.6 |

Observations (dossier arithmetic on these numbers):

- The packed Gram (D-25) halves the flops of the batched oracle (2.59e8 vs 5.17e8), cuts bytes
  accessed by 2.3× and temporaries by 8.3×, and is 2.25× faster here (`ms_min`).
- `gram_packed` + `precision_draw` + `sv_core` = 3.44 ms of the 3.87 ms sweep.
- Every production kernel is under its DESIGN §10.3 budget line on this machine (`sv_core` also
  under the 2.2 ms sum of DESIGN's SV lines; `sv_indicators + sv_awol` = 0.41 ms vs 1.5), but the
  budget was set for an M1 Pro (DESIGN §10.2), so machine and load differ.

### 13.6 Three sources of performance numbers: do not blend them

| Source | Code | Machine and load | Example numbers | Status |
| :--- | :--- | :--- | :--- | :--- |
| DESIGN §10.2 judges' re-measurement; D-25, D-26 | proposal scratch scripts: a **stand-in SV-parameter block** (no ASIS, no C/NC MH, no exact correction), data closed over as constants, `T = 300` not 320 | M1 Pro, jax 0.12, float64 (load not stated) | full sweep 8.07–8.16 ms (batched Gram) vs 4.56–4.60 ms (packed), 2.47–2.53 ms with 4 devices; eigen path 0.343–0.346 ms | recorded (prototype) |
| pilot (`09_pilot.md`) | clean-room prototypes in `benchmarks/pilot/` with the real SV block | M1 Pro, 1-min load 4.7–10.8 on 10 cores ("not idle", pilot §0) | full sweep 6.27–6.65 ms; BENCH-P2 projected 74.8–80.4 s on 1 device, 47–51 s on 4 devices; exact mode +12% to +22% | recorded (prototype, upper bounds under load) |
| `tests/bench/baseline.json` | **the production engine** through `BVAR.fit` | M5 Max, load ≈ 2–6 (§13.2) | 52.3 s cold, 3.90 ms/sweep | recorded |

Only the third row is the engine's performance. The first two document how defaults were chosen
(DESIGN §10.4 levers; pilot decisions 1–13).

DESIGN §10.4 levers, in order, none of which changes the target distribution: packed Gram;
chains on CPU devices; Constant fast paths; inverse-CDF categorical; two-solve Jacobi-scaled
precision draw; `logsumexp` reuse in the exact correction; no recompiles plus the persistent cache;
state donation. Not pursued: associative-scan SV (slower on CPU), CG samplers (inexact), float32
(R4). "A target miss is escalated (Q2), never fixed by flipping `exact`" (DESIGN §10.4).

### 13.7 x86 (provisional, loaded; from git history only)

The baseline that `d490730` replaced (`git show d37e059:tests/bench/baseline.json`, command 2,
§0.3; label "x86 cloud, loaded (load avg ~10 on 4 cores), provisional, not the reference"; code at
commit `1093932`, which already contains the OpenBLAS single-thread change). **[recorded]**

| Field | Value |
| :--- | :--- |
| CPU | Intel Xeon Processor @ 2.10 GHz (AVX, AVX2, AVX-512F, FMA), 4 physical = 4 logical, 16.9 GB, Linux 6.18 x86_64, Python 3.13.16, NumPy 2.5.4, jax/jaxlib 0.12.0.1 |
| load average at write time | 11.7, 11.1, 10.0 |
| `wall_cold_s` / `wall_warm_s` | 311.0 s / 284.8 s (fails 120 s) |
| `compile_cold_s` / `compile_warm_s` | 29.8 s / 3.9 s |
| `runner_ms_per_sweep` / `sweep_ms_min3x1000` | 24.4 / 21.3 ms |
| `peak_rss_mb` | 1,972 (MiB) |
| kernels `ms_min`: gram_packed / gram_batched / precision_draw / sv_core / sweep | 2.82 / 6.89 / 5.04 / 5.22 / 23.0 ms |
| ESS: `σ` `ess_min`, primary `ess_min` | 446, 446 (ESS/s 1.57) |
| `converged` | false in all 6 runs |

This was measured under heavy load on 4 cores (1-minute load 5.8–11.2 before the six runs,
`summary.loadavg_1min`; 11.7/11.1/10.0 at write time; label "load avg ~10") and is not a fair x86
figure (corrected by fact-check: the dossier said "load 10–12"). DESIGN §13 WP24 asks for
BENCH-P2 to be "reported on x86". The `_blas.py` docstring mentions an "idle 4-core x86 machine"
with a sweep of ≈ 16 ms, but no committed JSON holds an idle x86 run. **[to check]**

### 13.8 What has not been measured on the production engine

- The **stretch target** (< 60 s with `set_cpu_devices(4)`). The only numbers are pilot
  projections, 47–51 s (prototype). `phase2_perf.py --devices 4` exists, but no output is
  committed. **[to check]**
- The **reduced form (CCCM)**. DESIGN §10.3 says "estimated +1–1.5 ms per sweep … (to measure)".
  Pilot §12: "not a WP3 item". N07 §12 has a loaded-machine proxy of 9.1–13.3 ms per sweep for the
  coefficient block. **[to check]**
- **SV(t) and SV-O-t** on the production engine. Pilot only, and for SV(t) only: 9.9 ms vs 6.3 ms
  per sweep, "≈ 120 s at the BENCH-P2 shape", not part of the target (pilot §5). No committed file
  holds any SV-O-t sweep timing (corrected by fact-check: the dossier attached the pilot SV(t)
  number to SV-O-t as well). **[to check]**
- **Linear × Constant** end to end. DESIGN §10.3 estimates ≈ 4 s (prototype). The production
  number is NOTES 14's 0.46 ms/sweep for the dense eigen path with dummies (§13.9). **[to check]**
  without dummies.

### 13.9 Other recorded timings

- Mode I × Constant **with SOC/SUR dummies**, `n = 20, p = 4, T = 300`: dense eigen path 0.46
  ms/sweep vs 1.53 ms/sweep for the per-sweep Cholesky (`cholesky_const`). Measured on an Apple M5
  Max "under a concurrent load of 3–5" (NOTES 14; DESIGN §2.10 WP25 amendment (3)). **[recorded]**;
  not in `baseline.json`.
- Weighted-path timing guard at the BENCH-P2 shape:
  `tests/mcmc/blocks/test_linear_fastpaths.py::test_weighted_path_timing_reference_shape` compares
  with the pilot prototype in the same process. The pilot's 3.30 ms per mean draw for 80 lanes is
  machine- and load-specific (that test's docstring). **[code]**
- Diagnostics cost: about 1 s CPU for the flag, table and warnings at `n = 20` SV-O-t (392
  quantities) and 4 × 2000 draws, and about 7 s at 4 × 10,000 (DESIGN §8.2, Stage 2 amendment).
  **[recorded]**

### 13.10 Is the baseline still valid at HEAD?

The baseline measured commit `deb3929`; HEAD is `62d15f6`. Between them
(command 4, §0.3), the engine and component sources changed in 14 files (587 insertions,
77 deletions):

- WP25 SOC/SUR dummies (`linear.py`, `linear_reduced.py`, `data.py`, `state.py`, `resolve.py`). No
  dummy code is traced with `n_dummy = 0`, and NOTES 16 records bitwise-unchanged output.
- QR contemporaneous draw (`recursive.py`). This is used only by non-fusing means; structural
  Linear fuses the `a_i` draw.
- RW Laplacian factorisation (`sv.py`, `linalg.py`). The diff shows the change confined to
  `vol.rw` branches.
- `_blas.py` docstring wording.
- Failure-replay host conversion (`runner.py`). This is on the failure path only.
- `monitor.py`, `config.py`, `resolve.py`, `results.py` and `eb.py`: WP25 plumbing, gated by
  `n_dummy` (per the WP25 amendments of DESIGN §2.10).

By this reading none touches the BENCH-P2 path (structural Linear, weighted path, AR SV). That is
an argument, not a measurement. **[to check]**: NOTES "Before submission" already requires
re-running every recorded item on the release commit.

---

## 14. Candidate claims for the paper (with status)

1. The engine expresses a multi-block Gibbs sampler for BVARs with SV, outliers and Student-t as
   pure JAX kernels batched by `vmap` over equations and chains, with the step order as checked
   data. **[code]**; invariants **[verified]** (§5).
2. Counter-based keys over the absolute sweep index make draws bitwise independent of chunking,
   burn pieces, thinning alignment and fused/unfused execution. They are *not* bitwise independent
   of the `vmap` batch size or device layout with production kernels, which agree there only to
   rounding. **[verified]** with toys and, for prefixes and for `vmap` vs `shard_map(4)` (production
   `structural_sv`, short burn; corrected by fact-check), production kernels; the chunk invariance
   with production kernels is **[to check]** (§10).
3. A partitionable-threefry hazard (direct draws from a parent key collide with its `fold_in`
   children) is invisible to JAX's key-reuse checker and is avoided by layout. **[verified]**
   (§7.4).
4. Failures are values with deterministic replay. The two production-engine numerical failures
   recorded in NOTES (entries 10 and 17) were reported with diagnostics and led to better algorithms
   (QR `a`-row draw; Laplacian RW pivots) instead of jitter. **[verified]** (§6.3, §6.7, §9).
   (Corrected by fact-check: the dossier said "the two real numerical failures of Phase 2"; the
   repository supports only "the two recorded in NOTES". The pilot's σ = 0 absorbing cold-start
   defect, §6.6, was a further failure, on the prototype, that no status code caught.)
5. The exact mixture correction (DNP Alg. 3) is the default and is validated with a coarse-mixture
   Geweke configuration, because OCSN-10 Geweke tests cannot see it. **[verified]** (NOTES 3) (§6.6).
6. Two exact Gaussian samplers of the same conditional (eigen vs Cholesky) agree only to
   `O(ε·cond)`. The test envelope is conditioning-aware. **[verified]** for the committed cases;
   the bound is observed, not proven (§6.2).
7. BENCH-P2: 52.3 s cold and 49.6 s warm end to end for 4 × 11,000 sweeps of a 20-variable VAR(4)
   with SV on one CPU device (3.90 ms per sweep). **[recorded]** on a non-idle Apple M5 Max at
   commit `deb3929`; **[to check]** on the release commit, an idle machine and x86 (§13).
8. At BENCH-P2, SV σ_i and φ_i are the slowest-mixing quantities (ESS/s ≈ 10–11 vs ≈ 400 for own
   first-lag coefficients), and the default convergence rule fails on R-hat for 4 of 101 primary
   quantities. **[recorded]** (§13.4).

---

## 15. Open questions

**Performance evidence**

1. **Idle reference machine.** DESIGN §10.1 requires an idle Apple M-series laptop with 8+
   performance cores, and WP24 "the idle reference laptop" (the 8+ cores are in §10.1 only;
   corrected by fact-check). The baseline records `perf_cores = "6"`, `loadavg` 6.26/4.66/3.56 at write time, per-run
   1-minute loads 2.36–6.12, "desktop apps open", and "1.95 before the run" (commit d490730). Re-run
   idle, or restate the target machine?
2. **x86.** The only x86 BENCH-P2 is the replaced provisional baseline (1-minute load 5.8–11.2
   before the runs, 10.0–11.7 at write time, on 4 cores; 311 s) (corrected by fact-check). An idle x86 run is needed for the "reported on x86" exit criterion and for a portability
   statement.
3. **Stretch target** (< 60 s, 4 devices) has no production measurement. Run
   `benchmarks/phase2_perf.py --devices 4`.
4. **Reduced-form (CCCM), SV(t), SV-O-t and Linear × Constant** end-to-end timings on the
   production engine are missing (§13.8). A composition table would strengthen the paper.
5. **Peak RSS 3,043 MiB** (≈ 3.19 GB, M5 Max) vs 1,972 MiB (≈ 2.07 GB, x86 provisional), against
   stream sizes of about 256 MB (DESIGN §5.3) (corrected by fact-check: the dossier wrote 3.04 GB /
   1.97 GB, but `peak_rss_mb` is bytes / 2²⁰). What dominates it, and why does it differ by platform?
6. **Baseline at `deb3929` vs HEAD `62d15f6`.** Re-run on the release commit (NOTES "Before
   submission"). The argument of §13.10 is not a measurement.
7. **R-hat values** behind `converged = False` are only in a commit message. Should
   `phase2_perf.summarise` keep `rhat_max` (computed by `_ess_summary`) in the JSON?
8. **Default convergence rule for SV.** Pilot decision 6 (default `draws = 2000`, `ess_min = 400`,
   SV parameters primary) is still escalated. At BENCH-P2, even 10,000 draws per chain fail R-hat on
   a few σ_i. The paper needs a stated policy: raise the SV default draws, move φ and σ to the
   secondary set, or report.

**Documentation inconsistencies to fix before quoting**

9. DESIGN §2.3 two-bucket padding ratios (≈ 1.08×/1.05×) vs pilot §10 exact values (1.152×/1.100×).
   Pilot §11 asks the design owner to amend.
10. The DESIGN §10.3 exact-correction budget line (0.4 ms) vs the pilot measurement (≈ 1.1 ms in
    isolation, 0.7–1.2 ms in the full sweep). `bench_kernels.py` already annotates "0.4 (pilot
    1.1)". The production `sv_exact` (0.363 ms on M5 Max) is a different machine.
11. DESIGN §9.3 item 4 still lists "bitwise" invariance across chain prefixes and `vmap` vs
    `shard_map`, which the Stage 2 amendment (§5.2, §5.6) reduced to "to rounding".
    `_devices.set_cpu_devices` docstring says "Draws do not depend on the device count
    (DESIGN §5.2)", which contradicts the amendment.
12. NOTES 9 says the rounding comparisons are at "rtol ≈ 1e-12". The tests use `rtol 1e-12` for
    toys but `1e-9` for the production SV/CCCM kernels (`test_chunking.py`,
    `test_devices_bitwise.py` docstrings).
13. The `_blas.py` docstring says BENCH-P2's "largest factor is 81 × 81". The structural precision
    draw factors `q × q = 100 × 100` (`q = k + n − 1`). Is the batched Cholesky not routed through
    LAPACK under `vmap`, or is the docstring imprecise? The claim that thread count changes no
    result rests on it. (Added by fact-check: at the measured commit `deb3929` the same docstring
    said "a `≤ 100 × 100` Cholesky"; the "smaller than `128 × 128`" / "`81 × 81`" wording was
    introduced after the baseline, per `git diff deb3929 HEAD -- src/rai_bvar/engines/mcmc/_blas.py`.)
14. README.md still says "Status: Phase 1 (analytic engine)". It needs updating before release.
15. SPEC §8 attributes "all without a loop" to McCausland, Miller & Pelletier (2011). N02/N07 and
    DESIGN Q12 say the name comes from Kastner & Frühwirth-Schnatter (2014) describing the
    Rue/McCausland method. Use the corrected attribution.
16. SPEC §8 "Randomness" ("the same seed gives bitwise-identical results on the same platform and
    versions") needs the batch-size/device caveat of DESIGN §5.6.
    - (Added by fact-check.) DESIGN §5.4 was never amended for the SV cold-start fix: its table
      still initialises AR SV with `h ≡ μ_i` and RW SV with a constant `h`, with no unconditional
      AWOL draw, although pilot §1 decision 4 and §11 ask the design owner to amend it and the code
      does the draw (`SVCore.init`; `sv.py` docstring "Cold start"). Amend DESIGN §5.4 before
      quoting it.

**Evidence gaps (tests)**

17. Chunk-size, burn-piece and fused/unfused **bitwise** invariance is tested with the runner toys
    only. Add one production-kernel check (for example `structural_sv` with chunk 5 vs 8), or state
    the toy scope in the paper. NOTES 9 says "verified in CI (Linux)" without that scope.
18. ~~`vmap` vs `shard_map` with production kernels … Locate or add the production test.~~ Resolved
    (corrected by fact-check): `test_devices_bitwise.py::test_shard_map_matches_vmap_in_subprocess`
    already runs production `structural_sv` (both `"vmap"` and `"map"`, 2 burn sweeps) to `rtol 1e-9`
    after its toy block. Remaining gap: only a 2-sweep burn and one composition (no CCCM / SV-O-t
    under `shard_map`), and `"map"` bitwise equality is printed, not asserted.
19. The `_blas.single_thread` result-invariance claim ("checked bitwise on BENCH-P2-shaped fits")
    has no committed test or record. Only set/restore semantics are tested.

**Candidate NOTES ledger entries (not present yet)**

20. The partitionable-threefry key hazard (§7.4; `test_robust.py::test_ahead_and_transition_give_the_core_its_own_sub_key`).
21. The OpenBLAS oversubscription finding: monitor solves 9 ms → 100 ms per draw under load vs
    0.7 ms single-threaded (§8.4; `_blas.py`). It needs a committed measurement before the paper.
22. The pilot's absorbing σ = 0 cold-start defect (§6.6; pilot §3.1). No status code fired; it is a
    good example for "failures as values" needing `C_T > 0, σ > 0` in `StepInfo.ok`.
23. The Marsaglia–Tsang vectorised gamma sampler vs `jax.random.loggamma` under `vmap` (one batched
    rejection loop across all lanes: 12.1 → 2.1 ms in the prototype τ draw, pilot §5). It is a
    general JAX lesson.

**Third-party claims**

24. stochvol's ASIS `|σ|` without flip (D-7; NOTES 1) and its Student-t sampler (NOTES 2) are
    recorded for 3.2.1 only. Re-test on the current release and contact the maintainers before any
    statement in the paper (NOTES "Before submission").

---

## 16. [CITE?] references (bibliography stage must verify every entry)

Strings are taken from the research notes' source lists (N01–N08, SPEC §15) where available,
otherwise from memory. All are flagged.

1. [CITE?] Carriero, A., Chan, J. C. C., Clark, T. E., & Marcellino, M. (2022). Corrigendum to
   "Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors".
   *Journal of Econometrics*, 227(2), 506–512. doi:10.1016/j.jeconom.2021.11.010 (per N04/N06).
2. [CITE?] Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian vector
   autoregressions with stochastic volatility and non-conjugate priors. *Journal of Econometrics*,
   212(1), 137–154.
3. [CITE?] Del Negro, M., & Primiceri, G. E. (2015). Time varying structural vector
   autoregressions and monetary policy: a corrigendum. *Review of Economic Studies*, 82(4),
   1342–1345.
4. [CITE?] Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with
   leverage: fast and efficient likelihood inference. *Journal of Econometrics*, 140(2), 425–449.
5. [CITE?] Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: likelihood inference
   and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361–393.
6. [CITE?] Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving
   strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational
   Statistics & Data Analysis*, 76, 408–423.
7. [CITE?] Kastner, G. (2016). Dealing with stochastic volatility in time series using the R
   package stochvol. *Journal of Statistical Software*, 69(5).
8. [CITE?] Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood
   estimation in state space models. *International Journal of Mathematical Modelling and Numerical
   Optimisation*, 1(1), 101–120.
9. [CITE?] McCausland, W. J., Miller, S., & Pelletier, D. (2011). Simulation smoothing for
   state–space models: a computational efficiency analysis. *Computational Statistics & Data
   Analysis*, 55(1), 199–212.
10. [CITE?] Rue, H. (2001). Fast sampling of Gaussian Markov random fields. *Journal of the Royal
    Statistical Society B*, 63(2), 325–338 (from memory).
11. [CITE?] Chan, J. C. C. (2022). Asymmetric conjugate priors for large Bayesian VARs.
    *Quantitative Economics*, 13(3), 1145–1169. doi:10.3982/QE1381.
12. [CITE?] Marsaglia, G., & Tsang, W. W. (2000). A simple method for generating gamma variables.
    *ACM Transactions on Mathematical Software*, 26(3), 363–372.
13. [CITE?] Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel random
    numbers: as easy as 1, 2, 3. *Proceedings of SC11* (from memory; N07 §4.2 cites "Salmon et al.
    (2011)").
14. [CITE?] van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers: theory and
    methods. *Journal of the American Statistical Association*, 103(482), 790–796 (from memory).
15. [CITE?] Fox, C., & Parker, A. (2017). Accelerated Gibbs sampling of normal distributions using
    matrix splittings and polynomials. *Bernoulli*, 23(4B), 3711–3743.
16. [CITE?] Jacquier, E., Polson, N. G., & Rossi, P. E. (2004). Bayesian analysis of stochastic
    volatility models with fat-tails and correlated errors. *Journal of Econometrics*, 122(1),
    185–212.
17. [CITE?] Carriero, A., Clark, T. E., Marcellino, M., & Mertens, E. (2022 or 2024). Addressing
    COVID-19 outliers in BVARs with stochastic volatility. *Review of Economics and Statistics*
    (year and venue to verify; N03 cites it as "CCMM").
18. [CITE?] Geweke, J. (2004). Getting it right: joint distribution tests of posterior simulators.
    *Journal of the American Statistical Association*, 99(467), 799–804.
19. [CITE?] Grosse, R. B., & Duvenaud, D. K. (2014). Testing MCMC code. arXiv:1412.5218.
20. [CITE?] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
    Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of
    MCMC. *Bayesian Analysis*, 16(2), 667–718 (from memory).
21. [CITE?] Stroud, J. R., Müller, P., & Polson, N. G. (2003). Nonlinear state-space models with
    state-dependent variances. *JASA*, 98(462), 377–386 (MH correction of mixture approximations;
    N01 marks it "not opened").
22. [CITE?] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D.,
    Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX:
    composable transformations of Python+NumPy programs. Software, http://github.com/jax-ml/jax
    (from memory; the version used is jax 0.12.0.1).
23. [CITE?] Frostig, R., Johnson, M. J., & Leary, C. (2018). Compiling machine learning programs
    via high-level tracing. *SysML* (from memory).
24. [CITE?] Clark, T. E., & Ravazzolo, F. (2015). Macroeconomic forecasting performance under
    alternative specifications of time-varying volatility. *Journal of Applied Econometrics*,
    30(4), 551–575 (the replication target behind NOTES 17).
25. [CITE?] Hosszejni, D., & Kastner, G. (2021). Modeling univariate and multivariate stochastic
    volatility in R with stochvol and factorstochvol. *Journal of Statistical Software*, 100(12)
    (the oracle package; N02 lists "Kastner (2021)" for the same JSS article: verify the author
    list).
26. [CITE?] Frühwirth-Schnatter, S., & Wagner, H. (2010). Stochastic model specification search
    for Gaussian and partial non-Gaussian state space models. *Journal of Econometrics*, 154(1),
    85–100 (sign non-identification of the non-centred scale, background for the fold, D-7).

---

## 17. Unsupported claims (fact-check, 2026-10-23)

Fact-check scope: every package claim and number in §0–§15 was traced to the cited file, test,
commit or command (literature entries of §16 were not checked; another stage does that). All 73
cited test functions (and every cited test module) exist under `tests/`; every number in the `baseline.json` tables (§13.2–§13.5),
the provisional x86 JSON (`git show d37e059:tests/bench/baseline.json`, §13.7), the two commit
messages, the pilot tables (`docs/phase2/09_pilot.md` §0–§12) and the DESIGN tables (§2.3, §2.7,
§5.3, §8.2, §9.3, §10.1–§10.5) matched, and the dossier's rendering of the NOTES entries it cites
(1–4, 8b, 9, 10, 14, 16, 17) matches the ledger text (the entries' own sources were not re-traced,
and NOTES 5–8, 11–13 and 15, which the dossier does not cite, were not checked). All five
commands of §0.3 were re-run and reproduce (`decode_error(2001) = (EQ_VOL, 0)`, partitionable threefry `True` with a plain
`import jax`, `jax 0.12.0.1`, `P = 5050`, `T_pad·P·8 = 12,928,000`, `_blas.openblas() = None`
with SciPy 1.18.1; diff stat 14 files, 587 insertions, 77 deletions); `sysctl` confirms Apple M5 Max,
18 logical CPUs. Corrections are marked "(corrected by fact-check)" in place.

**Claims moved here:** none. No claim of §0–§15 lacked evidence entirely; the wrong ones were
corrected in place.

**Claims that stand, but only on a docstring, a commit message or one run (recorded, not
re-runnable from a test).** Listed so the paper does not quote them as verified:

1. The OpenBLAS numbers (9 ms idle / 100 ms at load 12 vs 0.7 ms single-threaded per kept draw;
   sweep ≈ 16 ms) and "checked bitwise on BENCH-P2-shaped fits" (`_blas.py` docstring only; §8.4,
   Open questions 19, 21).
2. The R-hat values behind `converged = False` (worst `sigma[y7]` 1.016, `sigma[y1]` 1.015; "4 of
   101") are in the commit message of `d490730` only (§13.4, Open question 7).
3. The eigen-path accuracy envelope over ≈ 1,500 problems / 4,400 problem–λ pairs and the "2,250
   fresh problems" of DESIGN §9.3 (docstring / DESIGN only; the case counts in the committed tests
   were not tallied by this fact-check) (§6.2).
4. NOTES 16's "537 arrays bitwise identical" (one run, WP25 report; §10, §13.10).
5. The `0.46 vs 1.53 ms/sweep` dense-eigen timing (NOTES 14, DESIGN §2.10 WP25 amendment (3),
   `resolve.py` docstring; one machine under load 3–5) (§6.2, §13.9).
6. Every pilot number (prototype code, loaded M1 Pro): exact-mode cost, acceptance rates, cold-start
   absorption shares, Marsaglia–Tsang speed-up, philox, buckets, projections (§6, §7.5, §13.6).
7. The dossier-machine 1-minute load "4.95" of §0.3 is a transient reading and cannot be re-checked.
