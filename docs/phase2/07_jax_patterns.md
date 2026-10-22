# 07 · JAX implementation patterns for the Phase-2 Gibbs engine

Scope: how to write the `engines/mcmc/` code (`sweep.py`, `state.py`, `linalg.py`, `runner.py`) so that the Gibbs sampler for `Linear × {Constant, SV, SV(outliers), SV(t)}` (structural and CCCM-2022 reduced form) is **exact, float64, reproducible and fast** in current JAX. This note covers the computational layer. Model-specific priors and conditionals (SV parametrisation, ASIS, Omori offset, outlier and t blocks, CCCM conditionals) belong to the sibling notes in this folder. Here they appear only as the *kernel shapes* the JAX code has to implement.

**Environment actually tested** (21 Oct 2026): `jax`/`jaxlib` **0.12.0.1** (PyPI, released 2026-10-21; 0.12.0 on 2026-10-19), CPU wheel, Python 3.12.7, NumPy 2.5.4, SciPy 1.18.1, Apple M1 Pro (8 performance and 2 efficiency cores, 16 GB). Every API claim marked *(verified)* was executed against that install. The scripts are in `scratchpad/jaxbench/` (absolute path: `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/jaxbench/`), and `snippets.py` there runs the self-checks for every snippet in this note.

**Benchmark caveat:** other workflow agents shared the machine during the runs (load average 25–50 on 10 cores). Every timing is the minimum over repeats and should be read as an **upper bound**. Re-run `run_all.sh` on an idle machine during the Phase-2 pilot.

**Independently cross-checked** on 21 Oct 2026 against the JAX source and docs, the papers and fresh runs on the same 0.12.0.1 install. Corrections are marked "cross-check" inline and listed in the **Cross-check log** at the end. The cross-check scripts are in `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc07/`.

---

## 0. Decisions in one table

| Topic | Recommendation | Why / evidence |
| :--- | :--- | :--- |
| float64 | `jax.config.update("jax_enable_x64", True)` at import of `engines.mcmc` (SPEC §8), **plus** `with jax.enable_x64(True):` around every public engine entry, plus dtype asserts | Global flag (SPEC); the context manager has been non-experimental since 0.8.0. *(verified)* jit caches float32 and float64 traces separately |
| PRNG impl | Default `threefry2x32`, typed keys from `jax.random.key(seed)` | Per the JAX docs, only Threefry/Philox are "identical across CPU/GPU/TPU" and give "exact `jax.vmap` over keys". `rbg`/`unsafe_rbg` generate a whole vmapped batch from the first key: **forbidden** |
| Key derivation | `fold_in` chain: `root → stream → chain c → absolute iteration it → block id → equation i → sub-draw j` | Same draws regardless of chunking, thinning, chain count or vmap-vs-loop *(verified)*; no key in the loop carry |
| Iterations | `lax.scan` over saved draws × inner `lax.fori_loop(0, thin)`; Python loop over fixed-size **chunks** of one compiled executable | One compile; bounded memory; host-side diagnostics between chunks |
| Chains | `vmap` on one device, or `shard_map` over `jax_num_cpu_devices = n_chains` CPU devices when the user opts in before JAX initialises | 4 CPU devices: 1.7–2.2× faster than single-device vmap for SV, 2.8–3.7× for Constant (§12); sharded draws are bitwise equal to single-device vmap *(verified, cross-check)* |
| Equations | `vmap` over equations with **padded, masked** regressors (`K_max = k + n − 1`); padded coefficients get an independent N(0,1) prior and are zeroed afterwards | Exact (block-diagonal posterior precision, §2.2); one batched Cholesky |
| Gaussian blocks | Precision form: Cholesky of `K = P_0 + Z'WZ`, then **two** triangular solves `β = L⁻ᵀ(L⁻¹r + z)`. Optionally form the Gram matrix with `lax.linalg.symmetric_product` (SYRK, lower triangle) and use `cholesky(·, symmetrize_input=False)` | SPEC §8 (no inverses); one solve fewer than `cho_solve` + noise solve. SYRK is 20% faster in isolation but showed no measurable gain in the full sweep |
| SV states | Tridiagonal precision sampler as two `lax.scan`s (forward Cholesky and forward solve fused, then back-substitution); returns `log|Q|` for free | Fastest of the 3 variants tried on CPU: 0.21 ms for 80 series × T=300 vs 0.74–1.19 ms (`associative_scan`) and 2.1–3.1 ms (`tridiagonal_solve` + perturbation, RNG included) |
| Discrete draws | Own inverse-CDF categorical from log-weights (max-subtract, `cumsum`, one uniform per row) | `jax.random.categorical` is Gumbel-max, 3.5–4.4× slower here (7.0 vs 1.6–2.0 ms for 80×300×10) |
| Truncated normal | Own log-space inverse-CDF with Newton refinement (§9.3) | `jax.random.truncated_normal` returns `1.797e308` on (9, ∞) *(verified)* |
| Gamma family | `jax.random.gamma(..., method="exact")` (the default); `loggamma` for shapes < ~0.1; `beta` and `dirichlet` are already log-space | `method="approximate"` is documented as biased: forbidden. `gamma(1e-3)` is exactly 0 in ≈49% of draws, because `U^{1/a}` underflows with probability ≈ `exp(−744.4·a)` *(verified, cross-check)* |
| Failures | Carry an int error code in the state (iteration, block, equation); raise on the host after each chunk; never add jitter | SPEC §8. JAX Cholesky returns NaNs and does not raise *(verified)* |
| Recompilation | Data and hyperparameter values as array args (pytree); only shapes and component choices static; T bucketed to a multiple of 32 with masks; opt-in persistent compilation cache | Second-process compile 1.56 s → 0.37 s with the CPU persistent cache *(verified)* |
| Storage | Coefficients and scalar vol parameters every kept draw; log-vol paths thinned or summarised; float64 | 4 chains × 10k draws of full SV paths at n=20, T=300 is about 1.9 GB |

---

## 1. Notation

| Symbol | Meaning | JAX array (one chain) |
| :--- | :--- | :--- |
| `n, p, k` | variables, lags, regressors per equation (`k = np + n_d`) | static ints |
| `K_max` | padded regressor count per equation, `k + n − 1` | static |
| `T`, `T_eff`, `T_pad` | sample length; effective length; padded (bucketed) length | `T_pad` static, `T_eff` traced int32 |
| `X` | `T_pad × k` lagged design (rows `t ≥ T_eff` arbitrary but finite) | `(T_pad, k)` |
| `Y` | `T_pad × n` targets | `(T_pad, n)` |
| `Z_i` | regressors of structural equation `i`: `[X, −Y_{<i}]`, padded | `(T_pad, K_max)` built on the fly |
| `θ_i` | `(γ_i, a_i)` padded to `K_max` | `(n, K_max)` |
| `λ_{i,t} = exp(h_{i,t})` | structural variances | `(n, T_pad)` |
| `w_{i,t}` | likelihood weight `= m_t / λ_{i,t}` with time mask `m_t = 1{t < T_eff}` | `(n, T_pad)` |
| `s_{i,t}` | Omori mixture indicator in `{0..9}` | `int32 (n, T_pad)` |
| `μ_i, φ_i, σ_i²` | SV level, persistence, innovation variance | `(n,)` each |
| `P_0, m_0` | prior precision and mean of `θ_i` (padded) | `(n, K_max, K_max)` or diagonal `(n, K_max)` |
| `key_c` | chain key | typed key `key<fry>[]` |
| `it` | absolute sweep index (burn-in counts) | `uint32` |
| `C` | number of chains | static |

JAX conventions: `jax.random.gamma(key, a)` is **Gamma(shape a, rate 1)**. So Gamma(a, rate b) = `gamma(key,a)/b`, Gamma(a, scale θ) = `θ·gamma(key,a)`, and IG(a, b) (density ∝ x^{−a−1}e^{−b/x}) = `b/gamma(key,a)`. R's `rgamma(n, shape, rate)` maps to `gamma(key, shape)/rate`. Prior parametrisations of R oracles (stochvol's `priorsigma`, `priorphi`, …) are mapped in the SV note, not here.

---

## 2. Model and the kernels the JAX code must provide

### 2.1 Structural form (SPEC §4.1)

`y_{i,t} = −Σ_{j<i} a_ij y_{j,t} + x_t'γ_i + √λ_{i,t} ε_{i,t}`. Given the volatilities and with priors independent across equations, the `n` equations are conditionally independent, which is what makes `vmap` over equations correct. The sweep order is fixed by SPEC §4.3:

```text
for it in 0 .. n_iter-1:                         # lax.scan / fori_loop, §5
    glob ← GlobalBlock(key(it, GLOBAL))          # hyperparameters; reduced-form CCCM block (§2.7)
    vmap over i:                                 # §6
        θ_i ← Gaussian block (2.2)               key(it, MEAN, i)
        r_i = y_i − Z_i θ_i ; y*_i = log(r_i² + c)    (offset c: SV note)
        s_i ← indicators | y*_i, h_i (2.4)       key(it, VOL_INDICATORS, i)
        h_i ← tridiagonal block (2.3)            key(it, VOL_STATES, i)     # nothing between s and h (DNP 2015)
        (μ_i, φ_i, σ_i) ← SV params (+ASIS)      key(it, VOL_PARAMS, i)
        tails/outliers (SV note)                 key(it, TAILS, i)
    save every `thin`-th state
```

The volatility variant is a **static** choice made in Python at trace time. Never use `lax.cond`/`lax.switch` on a traced flag inside the vmapped body: under `vmap` a batched predicate turns `cond` into `select`, so both branches run. *(Verified in the cross-check from the jaxpr on 0.12.0.1: `vmap(cond)` with a batched predicate lowers to `select_n` with both branch bodies inlined; with an unbatched predicate it stays `cond`. The runtime cost was not benchmarked.)*

### 2.2 Gaussian regression block (precision form)

For equation `i` with prior `θ_i ~ N(m_0, P_0⁻¹)` and weights `w_t`:

```
K_i = P_0 + Z_i' W_i Z_i ,   r_i = P_0 m_0 + Z_i' W_i y_i ,   W_i = diag(w_{i,1..T_pad})
θ_i | · ~ N(K_i⁻¹ r_i, K_i⁻¹)
L = chol(K_i) (lower),   v = L⁻¹ r_i,   θ_i = L⁻ᵀ (v + z),  z ~ N(0, I_{K_max})
```

Check: `E = L⁻ᵀL⁻¹ r = K⁻¹ r` and `Cov = L⁻ᵀ L⁻¹ = K⁻¹`. This uses two triangular solves instead of the three in "`cho_solve` for the mean, then one more solve for the noise".

**Padding is exact.** Order the columns as `[x_t (k) | −y_{1,t} … −y_{n−1,t} (n−1)]` and multiply column `k+j` by `1{j < i}`. For equation `i` the inactive columns are then the trailing `n−1−(i−1)` ones. Give them prior mean 0 and prior precision 1, with zero prior cross-precision to active coordinates. Their data columns are identically zero, so `K_i = blockdiag(K_i^{active}, I)` and `r_i = (r_i^{active}, 0)`. The Cholesky factor is block-diagonal too, so the active draw equals the unpadded draw computed with `z[:K_i]`, and the padded coordinates are iid N(0,1). Multiply them by the mask to store zeros. The padding cost is `n·K_max³ / Σ_i K_i³` in the Cholesky and `n·K_max² / Σ_i K_i²` in the Gram matrix, with `K_i = k + i − 1`. At n=20, k=81 these are 1.33 and 1.22. The rough form `(K_max/K̄)³` gives 1.35.

**Conditional centring (structural Minnesota, SPEC §4.1, D3)** fits the same kernel. With `E[γ_i | a_i] = M ã_i = M_{:,i} + M_{:,<i} a_i`, reparametrise `δ_i = γ_i − M_{:,<i} a_i`. Assuming the conditional covariance of `γ_i` does not depend on `a_i` (check against the Minnesota note), `(a_i, δ_i)` have independent Gaussian priors and the likelihood becomes `y_i = X δ_i + (X M_{:,<i} − Y_{<i}) a_i + e`. The kernel is unchanged: only the last `n−1` columns of `Z_i` change from `−Y_{<i}` to `X M_{:,<i} − Y_{<i}`, and they are masked the same way. Recover `γ_i = δ_i + M_{:,<i} a_i` when storing.

**Constant volatility shortcut:** `w_{i,t} = m_t/λ_i`, so `Z_i'W_iZ_i = (Z_i' D_m Z_i)/λ_i` with the time mask `D_m = diag(m_1..m_{T_pad})`. All cross-products are sub-blocks of `G = [X, Y]' D_m [X, Y]`, which is computed **once per dataset** (outside the iteration loop). Under conditional centring, `Z_i = [X, Y]·T_i` for a fixed `(k+n) × K_max` matrix `T_i`, built from the Minnesota mean `M_{:,<i}`, `−I` and the equation mask. The cross-product is then the congruence `T_iᵀ G T_i`, which is still computed once per dataset as long as `M` is fixed. Each sweep then costs only the `K_max³/3` Cholesky. Measured: 3.7–4.4 ms per iteration (single device) vs 14.5–18.0 ms with SV at n=20, 4 chains.

### 2.3 SV state block (tridiagonal precision)

Centred deviations `x_t = h_t − μ`, AR(1) `x_t = φ x_{t−1} + σ η_t`, initial `x_1 ~ N(0, V_1)`. The SV note fixes `V_1` and the h₀ convention. Stationary start `V_1 = σ²/(1−φ²)`. Conditional on the mixture indicators, `y*_t − μ − m_{s_t} = x_t + ζ_t` with `ζ_t ~ N(0, v²_{s_t})`. Let `D_t = m_t / v²_{s_t}` (0 on padded rows). Then `x | · ~ N(Q⁻¹b, Q⁻¹)` with tridiagonal `Q`:

```
Q_11 = 1/V_1 + φ²/σ² + D_1                         (= 1/σ² + D_1 under the stationary start)
Q_tt = (1 + φ²)/σ² + D_t                            1 < t < T_eff
Q_{T_eff,T_eff} = 1/σ² + D_{T_eff}
Q_{t,t−1} = −φ/σ²                                   2 ≤ t ≤ T_eff
b_t = D_t · (y*_t − μ − m_{s_t})
Padded t > T_eff: Q_tt = 1, Q_{t,t−1} = 0, b_t = 0  → padded states are iid N(0,1) dummies
```

Right-padding is exact for any Markov prior (stationary or random walk), because unobserved future states marginalise out without touching the past. The decoupled-identity choice above is also exact and keeps the padded states out of the `(μ, φ, σ)` sufficient statistics: mask every sum over `t` with `m_t` and `m_{t−1}`. Left-padding that extends the AR chain into the past is exact only under the stationary start. Left-padding with decoupled dummies is exact but needs a dynamic position for the initial-state row. Right-padding avoids both, so use it.

**Explicit `h_0` (stochvol-style convention, used in the SV note).** If `x_0 = h_0 − μ ~ N(0, σ²/(1−φ²))` is kept as a sampled state, run the same kernel on `T_eff + 1` states with an extra leading row: `Q_00 = (1−φ²)/σ² + φ²/σ² = 1/σ²`, `D_0 = 0`, `b_0 = 0`, `Q_{1,0} = −φ/σ²`. Row 1 then becomes an interior row, `Q_11 = (1+φ²)/σ² + D_1`. Integrating `x_0` out gives back the stationary `V_1` above (derivation re-done in the cross-check). Sampler: snippet S4 (§8.3). The `log|Q| = 2Σ log l_t` it returns is needed by integrated-likelihood MH steps (if the SV note uses any) and by tests.

### 2.4 Mixture indicators (10-component Omori et al. 2007)

`Pr(s_t = j | ·) ∝ π_j N(y*_t; h_t + m_j, v_j²)`, so the log-weight is

```
ℓ_tj = c_j − κ_j (y*_t − h_t − m_j)²,   c_j = log π_j − ½ log v_j²,   κ_j = 1/(2 v_j²)
```

`c_j` and `κ_j` are precomputed constants of shape `(10,)`. Draw with S3 (§9.2). The constants used in the benchmarks were checked against Table 1 (K = 10 block) of the Omori, Chib, Shephard & Nakajima working paper in `phase2-research/dl_omori/nuff.txt` (lines 142–151). Those `m_i` already include the KSC −1.2704 shift and approximate log χ²₁ directly:

```
π  = [0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115]
m  = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000]
v² = [0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342]
```

*Cross-check:* the three rows are identical in E. Mertens's `getKSC10values.m` and in stochvol's `sv_utils_latent_states.h` (constants only). The working-paper table's KSC half lists KSC's means already shifted by −1.2704 (for example 2.77786 − 1.2704 = 1.50746), which confirms that the K = 10 means approximate log χ²₁ with no further shift. Moment checks: Σπ = 1.00000; mixture mean −1.27028 vs E[log χ²₁] = ψ(½) + log 2 = −1.27036; mixture variance 4.93373 vs π²/2 = 4.93480.

The authoritative values and the offset `c` are set in the SV note.

### 2.5 Discrete-grid blocks (Student-t ν, outlier scales)

Any `p(g | ·) ∝ exp(ℓ_g)` on a finite grid is drawn with S3 on log-weights. Where a normaliser is needed, use `jax.nn.logsumexp`. Cost scales with `T × (#categories)`: a *joint* draw of (Omori component × outlier-scale grid of G points) costs `10·G` exponentials per `t`. At G = 20 that is 4.8M log-weights per sweep for 80 series × T=300. At the measured 6.7–8.3 ns per category element (1.6–2.0 ms for 240k elements) that is about 32–40 ms. Prefer the blocked conditional structure the outlier note derives, or collapse the grid analytically where the paper allows.

### 2.6 Scalar conjugate and MH steps

- `λ_i | · ~ IG(a_0 + T_eff/2, b_0 + SSR_i/2)` (Constant vol) → `(b_0 + SSR/2) / gamma(key, a_0 + T_eff/2)`.
- For σ², μ, φ, ν, use the SV/t notes' exact conditionals. JAX patterns for MH inside a vmapped Gibbs body: compute the log acceptance ratio in float64, accept with `jnp.log(u) < log_alpha` where `u = uniform(key, minval=tiny)`, and select with `jnp.where`. There is no Python branching.

### 2.7 Reduced-form prior (CCCM 2022 corrected triangular algorithm) in JAX

The coefficient steps for `j = 1..n` are **sequential**: `β_j` is drawn given the updated `β_{<j}` and current `β_{>j}`. So equations cannot be vmapped and the loop is `lax.fori_loop(0, n, body, B)` with dynamic indexing (`B.at[:, j].set(...)`, `jnp.where(i >= j, …)` masks). Chains are still vmapped or sharded. Equation `j`'s coefficients enter rows `i ≥ j` with loading `a_ij` (`a_jj = 1`), so its likelihood precision is a **single** weighted Gram matrix with combined weights `w_t^{(j)} = Σ_{i≥j} a_ij²/λ_{i,t}`. This is CCCM corrigendum eq. (14): `Ω̄⁻¹ = Ω⁻¹ + Σ_{i=j}^{N} a_{i,j}² Σ_t λ_{i,t}⁻¹ x_t x_t'`. Their `A` (in `ỹ_t = A y_t`) has the same entries as SPEC's `a_ij`. The mean is eq. (15), `Ω̄(Ω⁻¹μ + Σ_{i≥j} a_{i,j} Σ_t λ_{i,t}⁻¹ x_t z_{i,t})`, where `z_{i,t}` is `ỹ_{i,t}` minus the other equations' terms. The CCCM note has the details. The cost per `j` is one `T k²` Gram matrix, one `k³/3` Cholesky and two solves, which keeps the `O(n⁴)` scaling. Measured proxy (coefficient block only, §12): 9.1–13.3 ms per iteration at n=20, k=81, T=300, 4 chains. At CCCM's empirical size (n=20, p=13 ⇒ k=261, T≈647) it is 24–52 ms per iteration for 1 chain and 32 ms for 4 vmapped chains. For comparison, the CCCM corrigendum (Fig. 1, §4) reports **2.542 s per 10 draws** for the correct triangular algorithm (CTA) at N=20 and 33.10 s for the system-wide algorithm. That is MATLAB on a 3.5 GHz Intel Core i7, averaged over 10 chains, on FRED-MD monthly data for 1960–2014 with 13 lags. The text does not say whether the volatility steps are included in the timed draws.

---

## 3. float64

Mechanisms (JAX docs "Default dtypes and the X64 flag"): env var `JAX_ENABLE_X64=1`; `jax.config.update("jax_enable_x64", True)` (docs: "set at the top of your main file"); context manager `jax.enable_x64(True)` (non-experimental since 0.8.0, which deprecated `jax.experimental.enable_x64`). With the flag off, asking for a 64-bit dtype does not raise. It emits a `UserWarning` ("Explicitly requested dtype float64 … will be truncated to dtype float32") and returns float32 values. Tests should therefore turn that warning into an error. The docs present the context manager as a tool for scoped use and recommend the global flag for the whole program.

Verified behaviour (0.12.0.1):
- `jax.config.jax_enable_x64` is `False` by default.
- A function jitted while the flag is off returns float32 outside `with jax.enable_x64(True):` and float64 inside. Each mode gets its own cache entry.
- With x64 on, `random.normal` and `random.gamma` return float64, and default integers are int64.

Pattern:

```python
# engines/mcmc/__init__.py
import jax
jax.config.update("jax_enable_x64", True)        # SPEC §8; documented global side effect

def fit_mcmc(spec, data, *, seed, ...):
    with jax.enable_x64(True):                   # belt and braces if the user flips the flag later
        data = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), data)
        state = init_state(...)
        _assert_dtypes(state)                    # every float leaf float64, every key typed
        ...
```

- Use explicit integer dtypes: `uint32` for counters fed to `fold_in`, `int32` for indicators (halves storage vs int64).
- The flag is process-global, so enabling it changes the user's other JAX code. Document this. The `bartz` float32 exception (SPEC §8) lives inside its adapter.
- Tests: assert that all float leaves of state and outputs are float64. Run one test in a subprocess with `JAX_ENABLE_X64=0` set **before** import, to prove the engine enforces x64 itself.

---

## 4. PRNG key discipline

### 4.1 Facts (JAX docs + verified)

- Typed keys from `jax.random.key(seed)` are recommended over legacy `PRNGKey` (uint32 arrays). `jax.random.Key` is a type alias new in 0.12.0; the changelog says that from v0.13.0 it becomes the return type of `key()`, `split()` and `fold_in()`. Convert with `key_data` / `wrap_key_data` for persistence.
- Available impls (config `jax_default_prng_impl`): `threefry2x32` (default), `threefry4x32`, `philox2x32`, `philox4x32`, `rbg`, `unsafe_rbg` *(verified enum)*. Docs table: only Threefry and Philox are "identical across CPU/GPU/TPU" and have "exact `jax.vmap` over keys". For `rbg`, "the entire batch of output random numbers is generated from only the first key in the input key batch", so it must **never** be used with vmapped chains or equations. `philox2x32` has only a 32-bit key space.
- `jax_threefry_partitionable` defaults to `True` (since 0.5.0) *(verified True)*. Changing it changes the streams.
- API compatibility policy: "For a fixed PRNG key input, the outputs of pseudorandom functions in jax.random may vary across JAX versions … The compatibility policy applies only to the output distribution." Bitwise reproducibility (SPEC §8) therefore holds only for pinned `jax`/`jaxlib` on one platform. Record the versions, the PRNG impl and the partitionable flag in the fit metadata.
- `fold_in(key, data)` takes `data` in uint32 range. A Python int `2**32` raises `OverflowError` *(verified)*.
- Verified invariances on 0.12.0.1 (`snippets.py`):
  - vmapped `fold_in` equals looped `fold_in`.
  - vmapped `normal` equals per-key `normal`, bitwise.
  - jit equals eager, bitwise.
  - Chunked runs equal one long run, bitwise.
  - The first 3 chains of a 5-chain run equal a 3-chain run, bitwise.
- Key-reuse checker (experimental): `with jax.debug_key_reuse(True):` or config `jax_debug_key_reuse`. *Verified:* it raises `KeyReuseError` when one key feeds two `normal` calls, under jit and eager. It does **not** flag several `fold_in`s from the same parent, so the scheme below passes it.

### 4.2 Derivation scheme (S1)

```python
import enum, jax, jax.numpy as jnp
from jax import random

class Block(enum.IntEnum):          # append-only: never renumber (would change all streams)
    GLOBAL = 0; MEAN = 1; VOL_INDICATORS = 2; VOL_STATES = 3; VOL_PARAMS = 4; TAILS = 5

STREAM_MCMC, STREAM_INIT, STREAM_FORECAST = 0, 1, 2

def chain_keys(seed: int, n_chains: int):
    base = random.fold_in(random.key(seed), STREAM_MCMC)
    return jax.vmap(lambda c: random.fold_in(base, c))(jnp.arange(n_chains, dtype=jnp.uint32))

def block_key(chain_key, it, block: Block):          # it: absolute sweep index, uint32
    return random.fold_in(random.fold_in(chain_key, it), int(block))

def eq_keys(bkey, n_eq: int):                         # per-equation keys for the vmapped body
    return jax.vmap(lambda i: random.fold_in(bkey, i))(jnp.arange(n_eq, dtype=jnp.uint32))
```

Rules:
1. **Absolute iteration index.** Derive the iteration key from the sweep counter including burn-in, never by splitting a key carried in the loop state. Draws then do not depend on chunk size, thinning, warm restarts or burn-in/sampling phase boundaries.
2. **One `fold_in` per consumer.** A block that needs several independent draws per equation uses `fold_in(eq_key, j)` with fixed small `j` constants, or `split(eq_key, m)` with a static `m`. Never pass one key to two samplers.
3. **Equation index = position in the model ordering.** In the ordering-sensitivity check (SPEC §6.6), the same seed under a permuted ordering gives different streams by design.
4. **Separate streams** for initialisation, MCMC and forecasting (`STREAM_*`), so that adding forecast draws never perturbs posterior draws.
5. **Persist** only `(seed, it)` plus the version metadata. Keys are re-derivable. If raw keys must be stored, `key_data(k)` gives a uint32 `(..., 2)` array for threefry2x32.
6. Rejection samplers inside JAX (gamma) consume extra randomness internally from their key. This is fine because the key is used by that one call only.

Statistical note: `fold_in` hashes the data into the key with the Threefry function, as `split` does; Salmon et al. (2011) designed Threefry as a counter-based PRF. Using `fold_in` with distinct counters is the standard JAX idiom for step-indexed streams. JAX's docs give no ranking of split vs fold_in.

---

## 5. Loop structure: scan, thinning, chunks, donation

```python
def make_runner(sweep, extract, n_save_chunk: int):
    """sweep(chain_key, it, state, data) -> state ; extract(state) -> pytree of saved values."""
    def run_chunk(ckey, state, data, it0, thin):
        def save_step(state, j):
            def one(i, s):
                return sweep(ckey, (it0 + j * thin + i).astype(jnp.uint32), s, data)
            state = lax.fori_loop(0, thin, one, state)        # dynamic trip count -> while loop
            return state, extract(state)
        return lax.scan(save_step, state, jnp.arange(n_save_chunk, dtype=jnp.uint32))
    return jax.jit(jax.vmap(run_chunk, in_axes=(0, 0, None, None, None)),   # chains batched, data shared
                   donate_argnums=(1,))

# host side
runner = make_runner(sweep, extract, n_save_chunk=250)        # compiled once per (spec, shape bucket)
it = 0
for c in range(n_burn_chunks):                                 # burn-in: same executable, outputs dropped
    state, _ = runner(keys, state, data, jnp.uint32(it), thin); it += 250 * thin
for c in range(n_keep_chunks):
    state, draws = runner(keys, state, data, jnp.uint32(it), thin); it += 250 * thin
    store.append(jax.device_get(draws))                        # numpy; (C, 250, ...)
    if c % check_every == 0: diagnostics(store)                # R-hat/ESS via ArviZ on host (SPEC §10.6)
```

- `lax.scan` "is lowered to a single WhileOp" and its carry must keep a fixed shape and dtype (docs). `ys` are stacked on a new leading axis, so the output buffer is `n_save_chunk × draw_size`, and the chunk size bounds memory.
- `fori_loop` with a traced bound lowers to `while` *(verified)*, so `thin` can be dynamic without recompiling.
- `unroll`: no gain on CPU in our tridiagonal test. Over two runs, unroll 1/8/32 took 0.21–0.22 / 0.27–0.39 / 0.29–0.31 ms, and compile time rose from 0.18 s to 0.9–1.3 s to 1.4–1.8 s. Keep `unroll=1` and re-measure on an idle machine.
- Donation (`donate_argnums`) works on CPU in 0.12 *(verified: the donated input `is_deleted()`)*. After a call, the caller must drop the old state object.
- Keep host syncs (`jax.debug.print`, `io_callback`, `.block_until_ready()`) out of the jitted loop. Use them only between chunks.
- Warm starts (backtest, SPEC R9) pass the previous window's final state as `state` and keep `it` running. With the absolute-index scheme the stream remains well defined.

The verified skeleton is `make_runner` in `snippets.py` (chunking and chain-count invariance asserted).

---

## 6. vmap over equations and chains; multi-device CPU

- **Inner `vmap` over equations** with `in_axes` that batch per-equation arrays (`θ_i`, `h_i`, priors, masks) and broadcast shared ones (`X`, `Y`, constants: `in_axes=None`). Build `Z_i` inside the vmapped function from `X`, `Y` and the equation mask, rather than materialising `(n, T, K_max)` once per dataset. XLA may still materialise a transient batch, which is fine. It matters for SBC with hundreds of replications in one call.
- **Outer `vmap` over chains** batches only `(key, state)`. Data uses `in_axes=None`, so it is not copied per chain.
- Global blocks (hierarchy, CCCM reduced form) live outside the equation vmap and inside the chain vmap.
- Under `vmap`, a `while_loop` (for example the gamma rejection sampler) runs until **all** lanes finish. That is correct, just slower in the worst lane.
- NaNs do not cross lanes, but a corrupted chain stays corrupted, so error flags are required (§8.4).

**Several CPU devices** (opt-in). Giving each chain its own CPU device helps; the likely reason is XLA's limited intra-op parallelism on one program (a hypothesis, not checked). Measured: Linear×SV 14.5–18.0 → 8.2–8.6 ms per iteration; Linear×Constant 3.7–4.4 → 1.2–1.3 ms per iteration (4 chains, §12). *Cross-check:* with the snippet below, the sharded output was bitwise identical to single-device `vmap` over the same keys. Draws therefore do not depend on the device layout, at least on CPU 0.12.0.1.

```python
import jax
jax.config.update("jax_num_cpu_devices", 4)   # must run before ANY JAX operation (verified: RuntimeError otherwise)
from jax.sharding import NamedSharding, PartitionSpec as P
mesh = jax.make_mesh((4,), ("chain",))
shd  = NamedSharding(mesh, P("chain"))
keys, state = jax.device_put(keys, shd), jax.device_put(state, shd)   # required in 0.12 (verified)
body = jax.vmap(run_chunk, in_axes=(0, 0, None, None, None))
run  = jax.jit(jax.shard_map(lambda k, s, d, i0, th: body(k, s, d, i0, th), mesh=mesh,
               in_specs=(P("chain"), P("chain"), P(), P(), P()), out_specs=P("chain"),
               check_vma=False))
```

Passing unsharded inputs raised `ValueError: in_specs … does not match the specs of the input` in 0.12. `XLA_FLAGS=--xla_force_host_platform_device_count=4` is the older equivalent. Since the library cannot change the device count after the user's code has touched JAX, expose `rai_bvar.set_cpu_devices(k)` ("call right after import") and fall back to single-device vmap. The number of chains must be a multiple of the device count. For process-level parallelism in backtests (many fits at once), oversubscription is a real risk, and **no verified way to make a JAX 0.12 CPU process single-threaded was found**. The often-quoted `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` does not do what it appears to (cross-check, 0.12.0.1):
- `intra_op_parallelism_threads=1` has no leading `--`, so XLA silently ignores it. Any token without dashes is ignored: `bogus_flag_xyz=1` is "accepted" too. Written as `--intra_op_parallelism_threads=1`, it aborts with `Unknown flag in XLA_FLAGS`.
- `--xla_cpu_multi_thread_eigen=false` is a recognised flag. On a batched Gram + Cholesky workload under load, it only moved the CPU-time/wall-time ratio from 2.17 to 1.94, so the process stayed multi-threaded.

Prefer fewer processes, each running several fits through the chain/vmap axis, and measure (UNVERIFIED).

---

## 7. Avoiding recompilation

The jit cache key covers function identity, static arguments (by `__hash__`/`__eq__`), the abstract signature of array arguments (shape, dtype, weak type, sharding), the pytree structure, and tracing-relevant config such as x64.

1. **Factory per model spec.** Build the step and runner functions once per frozen, hashable spec dataclass (`functools.lru_cache` on a `make_runner(spec, shape_bucket)` factory). A re-created closure is a new function, which means a retrace and possibly a new compile.
2. **Data and hyperparameter values are arguments**, as a pytree. Never close over arrays: they become HLO constants, which makes compiles slow and makes every new dataset need a new trace. Pass Minnesota `λ`, prior scales, `T_eff`, `it0` and `thin` as arrays or ints, so EB/MH hyperparameter changes and new windows do not recompile.
3. **Static only what changes shapes or Python control flow:** `n, p, K_max, T_pad, n_save_chunk, C`, component choices (vol type, tails), the number of mixture components and grid sizes. Use `jax.tree_util.register_dataclass(..., data_fields=[...], meta_fields=[...])` for state classes *(verified)*. Meta fields are static.
4. **Fixed pytree structure.** No `Optional` leaf that is sometimes `None`. Use zero arrays plus flags.
5. **Bucket `T`.** `T_pad = 32·⌈T/32⌉` with the masks of §2.2–2.3, which are exact. An expanding-window monthly backtest then recompiles about once every 32 origins, and with the persistent cache only the first run pays.
6. **Persistent compilation cache** (opt-in, trusted directory only, per the docs): `jax.config.update("jax_compilation_cache_dir", path)`. Entries are written only if compile time exceeds `jax_persistent_cache_min_compile_time_secs` (docs default 1.0 s, verified as the config default). Small Geweke/SBC configurations compile in under 1 s and are **not** cached unless that threshold is lowered (the debug log says "Not writing persistent cache entry … because it took < 1.00 seconds to compile"). Per the docs, the key covers the unoptimised HLO, jaxlib version, relevant XLA flags, device configuration (number and topology), compression algorithm and an optional custom hook. The docs also warn that "the compilation cache is considered trusted". *Verified on CPU:* full-sweep compile 1.56 s → 0.37 s in a second process.
7. **Detect recompiles in tests.** Set `jax_log_compiles=True` and count log records across two fits on new data of the same bucket (expect 0 new compiles). `jax_explain_cache_misses=True` explains misses. Both config options exist in 0.12 *(verified)*.
   - *Cross-check, 0.12.0.1:* each compile emits a WARNING record from logger `jax._src.interpreters.pxla` with message `Compiling jit(<fn name>) with global shapes and types (…)`, and a `Finished XLA compilation of jit(<fn name>) in … sec` record from `jax._src.dispatch`.
   - A same-shape second call emitted no new records, and a new shape emitted them again.
   - Eager `jnp` calls outside jit also emit `Compiling jit(broadcast_in_dim)`-style records, so filter on the runner's function name rather than counting all records.
   - `f._cache_size()` works *(verified)*, but it is private API: use it only in tests.
8. **AOT for benchmarks:** `jax.jit(f).lower(*args).compile()` separates compile time from run time (used in §12). *Cross-check on CPU:* `compiled.memory_analysis()` returns a `CompiledMemoryStats` with `argument_size_in_bytes`, `output_size_in_bytes`, `temp_size_in_bytes`, `alias_size_in_bytes`, `generated_code_size_in_bytes` and `host_*` fields. `compiled.cost_analysis()` returns a dict with `flops` and `bytes accessed` keys; it gave 250,000 flops for a 50×50 matmul, which is correct.

Measured compile times (full Linear×SV sweep, n=20, K_max=100, T=300, 4 chains): **1.5–2.8 s** cold, **0.37 s** from the persistent cache.

---

## 8. Linear algebra in JAX

### 8.1 What exists (0.12.0.1, verified listing)

- `lax.linalg`: `cholesky`, `cholesky_update`, `triangular_solve`, `symmetric_product`, `tridiagonal`, `tridiagonal_solve`, `qr`, `eigh`, `lu`, `svd`, `schur`, `hessenberg`, `householder_product`, `ormqr`, `qdwh`.
- `jax.scipy.linalg`: `cho_factor`, `cho_solve`, `cholesky`, `solve_triangular`, `eigh_tridiagonal`, `lu_factor`, `lu_solve`, `qr`, `qr_multiply`, … There is **no** `solve_banded` or `cholesky_banded`.
- CPU lowerings (read from `jax/_src/lax/linalg.py`, Apache-2.0): `cholesky` → LAPACK `potrf_ffi`, `triangular_solve` → BLAS `trsm_ffi`, `tridiagonal_solve` → LAPACK `gtsv_ffi` (partial-pivoting LU).
- `jax.lax.linalg.tridiagonal_solve(dl, d, du, b, *, perturb_singular=False)`: `dl[0]` and `du[m−1]` must be 0, and `b` has shape `[..., m, k]`. It is batched over leading dims and works under vmap *(verified shapes)*. `perturb_singular` was added in 0.10.0.
- Changelog 0.10.0: "LAPACK operations are now parallelized along the batch dimension on CPU". Also from 0.10.0: `cho_solve`, `lu_solve` and `solve_triangular` **warn** about batched 1-D solves with `b.ndim > 1`, which will later be treated as batched 2-D solves. Avoid the ambiguity by calling the 1-D form inside `vmap`, or use `lax.linalg.triangular_solve` with an explicit `(K, 1)` right-hand side (as in S2).

### 8.2 Gaussian draw from a precision matrix (S2)

```python
def weighted_precision(Z, w, prior_prec_diag):
    """Lower triangle of Z' diag(w) Z + diag(prior_prec) via SYRK."""
    Zw_t = (Z * jnp.sqrt(w)[:, None]).T                                       # (K, T)
    return lax.linalg.symmetric_product(Zw_t, jnp.diag(prior_prec_diag), alpha=1.0, beta=1.0)

def gaussian_from_precision(key, K_lower, r):
    """beta ~ N(K^{-1} r, K^{-1}); only the lower triangle of K is read. Returns (beta, ok)."""
    L = lax.linalg.cholesky(K_lower, symmetrize_input=False)
    z = random.normal(key, r.shape, dtype=r.dtype)
    v = lax.linalg.triangular_solve(L, r[:, None], left_side=True, lower=True)            # L v = r
    beta = lax.linalg.triangular_solve(L, v + z[:, None], left_side=True, lower=True,
                                       transpose_a=True)[:, 0]                             # L' b = v + z
    return beta, jnp.all(jnp.isfinite(jnp.diagonal(L)))
```

- `symmetric_product` computes `αAAᵀ + βC` and guarantees only the lower triangle. The default `cholesky(symmetrize_input=True)` averages `A` with `Aᴴ` and would read the garbage upper triangle, so pass `symmetrize_input=False`. Alternatively pass `symmetrize_output=True` to `symmetric_product`. *Verified* against a dense NumPy draw built from the same `z`.
- Gram matrix timing for B=80, K=100, T=300: matmul 5.9 ms vs `symmetric_product` 4.7 ms in isolation (≈20% saving, same run). The Gram matrix dominates the coefficient block (≈60% of it). In the full sweep, the SYRK + two-solve variant (`sweep_opt.py`) was **not** measurably faster than plain matmul + `cholesky` + `cho_solve` + solve (`sweep.py`): 18.0 vs 16.6 ms per iteration single-device and 8.2 vs 8.4 ms sharded, which is within the load noise. Treat SYRK as optional. The two-solve form is still preferred because it does less work.
- Cholesky timing: 80 × (100×100) in 2.3–2.7 ms, the same as a SciPy per-matrix loop under load. The batch-parallel speed-up from the changelog was not observed (UNVERIFIED on an idle machine).
- **Large K (Phase 3, K ≫ T):** use the Bhattacharya, Chakraborty & Mallick (2016) algorithm (`O(T²K)`) instead of `O(K³)`. It is not needed for Phase 2.

### 8.3 Tridiagonal precision sampler (S4)

Rue (2001) and Chan & Jeliazkov (2009) sample banded Gaussian Markov random fields through a banded Cholesky, and McCausland, Miller & Pelletier (2011) do the same for state-space models. The name "all without a loop" (AWOL) is from Kastner & Frühwirth-Schnatter (2014, §2.2–2.3), who use it for the Rue / McCausland et al. method. It is not McCausland et al.'s own term, although SPEC §8 attributes the phrase to them. For bandwidth 1 the factor is lower-bidiagonal: `l_1 = √Q_11`, `c_t = Q_{t,t−1}/l_{t−1}`, `l_t = √(Q_tt − c_t²)`. One forward scan fuses the factorisation with `L v = b`, and one reverse scan solves `Lᵀx = v + z`.

```python
def tridiag_precision_sample(a, e, b, z):
    """Q = tridiag(e[1:], a, e[1:]) SPD (e[t] = Q[t,t-1], e[0] unused).
    Returns x = Q^{-1} b + L^{-T} z  (~ N(Q^{-1}b, Q^{-1}) for z ~ N(0,I)) and log|Q|."""
    def fwd(carry, inp):
        l_prev, v_prev = carry
        a_t, e_t, b_t = inp
        c_t = e_t / l_prev
        l_t = jnp.sqrt(a_t - c_t * c_t)          # NaN iff Q not PD -> error flag
        v_t = (b_t - c_t * v_prev) / l_t
        return (l_t, v_t), (l_t, c_t, v_t)
    l0 = jnp.sqrt(a[0]); v0 = b[0] / l0
    _, (l, c, v) = lax.scan(fwd, (l0, v0), (a[1:], e[1:], b[1:]))
    l = jnp.concatenate([l0[None], l]); v = jnp.concatenate([v0[None], v])
    c = jnp.concatenate([jnp.zeros((1,), a.dtype), c])
    w = v + z
    def bwd(x_next, inp):
        w_t, l_t, c_next = inp
        x_t = (w_t - c_next * x_next) / l_t
        return x_t, x_t
    xT = w[-1] / l[-1]
    _, xs = lax.scan(bwd, xT, (w[:-1], l[:-1], c[1:]), reverse=True)
    return jnp.concatenate([xs, xT[None]]), 2.0 * jnp.sum(jnp.log(l))
```

Verified against dense NumPy (`x` to 3e-16, `log|Q|` to 6e-14) at T=50 with SV-like `Q` (φ=0.95, σ²=0.04, v² ∈ [0.1, 7]). Alternatives measured at B=80 series × T=300:

| Variant | Idea | ms per call (two runs) | Notes |
| :--- | :--- | ---: | :--- |
| **A: two `lax.scan`s** (above) | sequential Cholesky | **0.21–0.22** | fastest on CPU; gives `log|Q|`; `unroll=8/32` slower (0.27–0.39 ms) and compiles 5–10× longer |
| C: `lax.associative_scan` | Möbius 2×2 products for the pivots `d_t = a_t − e_t²/d_{t−1}` (Stone 1973 recursive doubling), renormalised by the max entry at each combine; affine-map scans for both substitutions | 0.74–1.19 | O(log T) depth; consider on GPU or for very long T; accurate to 3e-15 |
| B: perturbation + `tridiagonal_solve` | `x = Q⁻¹(b + η)` with `η = HᵀΩ^{−1/2}z₁ + D^{1/2}z₂ ~ N(0, Q)` (Papandreou & Yuille 2010) | 2.09–3.09 (incl. 2T normals) | exact but needs 2× the normals and a factorised `Q`; no `log|Q|` |

Variant B's covariance was checked by Monte Carlo (2·10⁵ draws, max relative error 0.7%). It is shown because `tridiagonal_solve` is the only built-in banded solver. It is not recommended on CPU.

**Wider bands** (Phase 5 TVP states, AR(2) log-vols): there is no banded Cholesky in JAX, so write a block-tridiagonal Cholesky as a `lax.scan` over `t` with `k×k` blocks (`L_tt = chol(Q_tt − C_t C_tᵀ)`, `C_t = Q_{t,t−1} L_{t−1,t−1}⁻ᵀ`). It costs `O(T k³)`. This is outside Phase 2.

### 8.4 Failure handling (no jitter)

JAX returns NaNs from a failed Cholesky *(verified: `[[nan, 0], [nan, nan]]`)*. Pattern:

```python
# state.err: int32 scalar per chain, 0 = ok; encode (block, equation) in the first failing sweep
ok_i = jnp.all(jnp.isfinite(jnp.diagonal(L)))                       # per equation (vmapped)
code = jnp.where(ok_i, 0, 1 + int(Block.MEAN) * 1000 + eq_index)
err_new = jnp.where(state.err == 0, jnp.max(code), state.err)       # keep the first error
err_it  = jnp.where((state.err == 0) & (err_new != 0), it, state.err_it)
```

After each chunk the host checks `err`. On failure it recomputes the offending `K_i` in NumPy from the saved pre-failure state (or from the iteration's inputs) and raises with the condition number and block/equation (SPEC §8). `jax_debug_nans` is a debugging aid only.

---

## 9. Sampling distributions in JAX

### 9.1 Built-ins (0.12.0.1)

| Need | Use | Notes (verified unless marked) |
| :--- | :--- | :--- |
| N(0,1) | `random.normal(key, shape)` | `√2·erfinv(U(−1+ε, 1))`. Costs 1–2× a uniform: 0.48–0.55 ms vs 0.26–0.49 ms per 24k float64 draws, and the two runs disagree. 52-bit uniforms cap \|z\| at ≈8.1 (negligible) |
| U(0,1) | `random.uniform(key, shape)` | `[0, 1)` on a 2⁻⁵² grid, so `u = 0` has probability 2⁻⁵² ≈ 2.2·10⁻¹⁶ per draw. The reference run uses ≈10⁹ uniforms (4 chains × 10⁴ sweeps × 80 × 300 indicator draws), so P(some `u = 0`) ≈ 2·10⁻⁷ per run. Use `minval=jnp.finfo(jnp.float64).tiny` whenever `log u` is taken |
| Gamma(a, rate b) | `random.gamma(key, a) / b` | `method="exact"` (default) is a Marsaglia–Tsang (2000) rejection loop, with `U^{1/a}` boosting for a < 1. `method="approximate"` is "loop-free and faster but carries a small bias": **never use** |
| log Gamma, small a | `random.loggamma` | `gamma(key, a)` draws `G_{a+1}·U^{1/a}` and underflows to exact 0 with probability ≈ `exp(−744.4·a)`. Cross-check, 2·10⁵ draws: 49% zeros at a=1e-3, 0.09% at a=0.01, none at a=0.1. `loggamma` stays finite. Rule: use `loggamma` whenever a < 0.1 is possible |
| IG(a, b) | `b / random.gamma(key, a)` | or `exp(log b − loggamma)` |
| Beta(a, b) | `random.beta` | two `loggamma` draws, normalised by their max (stable) |
| Dirichlet | `random.dirichlet` | `softmax` of `loggamma` draws (stable for small α) |
| χ²_ν, Student-t | `random.chisquare`, `random.t` | `chisquare` also takes `method` |
| Categorical | **own S3**; not `random.categorical` | Built-in = Gumbel-max ("mode=low/high" bit depth); 7.0 ms vs 1.6 ms for 80×300×10 |
| Truncated normal | **own S6**; not `random.truncated_normal` | Built-in is erf/erfinv inversion: returned `1.797e308` for (9, ∞) |
| MVN | own precision sampler (S2) | `random.multivariate_normal` takes a covariance (cholesky/eigh/svd), which violates SPEC §8's precision-form rule |
| Bernoulli | `uniform < p` or `random.bernoulli` | `bernoulli` docs warn about `mode="low"` for small p in float32 |
| GIG, Pólya-Gamma | not in `jax.random` | Phase 3 (NG/HS) will need an own GIG sampler (UNVERIFIED which algorithm) |

### 9.2 Categorical from log-weights (S3)

```python
def categorical_from_logits(key, logw):
    """One uniform per row, exact up to rounding. logw: (..., J) unnormalised log-probs."""
    J = logw.shape[-1]
    w = jnp.exp(logw - jnp.max(logw, axis=-1, keepdims=True))      # the largest weight is exactly 1
    cw = jnp.cumsum(w, axis=-1)
    u = random.uniform(key, logw.shape[:-1], dtype=logw.dtype) * cw[..., -1]
    return jnp.minimum(jnp.sum(cw <= u[..., None], axis=-1), J - 1)
```

`<=` skips zero-weight categories when `u = 0`. The `minimum` guards against `u·cw[-1]` rounding up to `cw[-1]`. Frequency check on log-weights shifted by +700: target (0.001, 0.2, 0.3, 0.499), observed (0.00106, 0.2005, 0.3002, 0.4983) with 4·10⁵ draws *(verified)*.

### 9.3 Tail-safe truncated normal (S6)

Needed for φ-type conditionals whose untruncated mean lies outside (−1, 1) by many standard deviations. In that case both bounds sit in the same far tail and the built-in fails.

```python
from jax.scipy.special import log_ndtr, ndtri
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

def _log_ndtri(lp, newton_steps=4):
    """Solve log Phi(x) = lp (lp <= log 0.5): ndtri start when exp(lp) is representable,
    else the Mills-ratio asymptote; then Newton on g(x) = log_ndtr(x) - lp, g' = phi/Phi."""
    t = jnp.sqrt(-2.0 * lp)
    x_asym = -(t - (jnp.log(t) + _LOG_SQRT_2PI) / t)
    p = jnp.exp(lp)
    x = jnp.where(p > 1e-300, ndtri(jnp.clip(p, 1e-300, 1.0)), x_asym)
    for _ in range(newton_steps):
        lphi = -0.5 * x * x - _LOG_SQRT_2PI
        x = x - (log_ndtr(x) - lp) * jnp.exp(log_ndtr(x) - lphi)
    return x

def truncated_std_normal(key, a, b):
    """Z ~ N(0,1) restricted to (a, b); reflect into the lower half-line, invert in log space."""
    flip = a > 0
    lo, hi = jnp.where(flip, -b, a), jnp.where(flip, -a, b)
    la, lb = log_ndtr(lo), log_ndtr(hi)
    u = random.uniform(key, jnp.shape(lo), minval=jnp.finfo(jnp.float64).tiny)
    lp = lb + jnp.log(u + (1.0 - u) * jnp.exp(la - lb))       # log(Phi(lo) + u (Phi(hi) - Phi(lo)))
    x = jnp.where(lp < np.log(0.5), _log_ndtri(jnp.minimum(lp, np.log(0.5))), ndtri(jnp.exp(lp)))
    x = jnp.clip(x, lo, hi)
    return jnp.where(flip, -x, x)
```

Tested against `scipy.stats.truncnorm` on (−1,1), (2,3), (9,∞), (−∞,−40), (−410,−10), (30,30.5), (−0.5,∞) and (−∞,∞): all draws in bounds, and means match to ≤5·10⁻³. Across 40 seeds × 4 intervals the KS p-values were uniform (KS-of-p-values p = 0.28; min p 0.018). One seed alone gave p = 0.004 on every interval. Its raw uniforms had KS p = 0.0044, so that was the stream's luck and not a bias. The cross-check confirmed this: `key(0)` uniforms give KS p = 0.0044. The KS statistic is invariant under monotone maps, so equal p-values on every interval are exactly what exact inversion predicts.

*Cross-check (same-u test):* the S6 output was compared with `scipy.stats.truncnorm.ppf` evaluated at the same uniforms, with the same reflection. The maximum relative error was ≤ 1·10⁻¹⁴ on all 10 intervals tried: the 8 above plus (−3, 50) and (37, 38). Over 60 runs (20 seeds × 3 intervals, 5·10⁴ draws each), the KS p-values had min 0.032 and a uniformity p of 0.089.

Robert (1995) exponential rejection is the classic alternative, but it needs a `while_loop`.

### 9.4 Numerical stability of log-mixture sampling (checklist)

1. Work with log-weights `ℓ_tj = c_j − κ_j d_tj²` (§2.4) and never exponentiate raw densities. COVID-size residuals give `d²` in the hundreds, so `exp` of a raw density underflows to 0 for every component and normalising gives NaN.
2. Subtract the row max before `exp`. The largest weight is then exactly 1, so the sum is ≥ 1 and never 0.
3. `y* = log(r² + c)`: apply the time mask **before** the log, because padded rows can have `r = 0` and `log 0 = −inf` turns the weights into NaN. Real rows rely on the SV note's offset `c`.
4. Precompute `log π_j`, `½ log v_j²` and `1/(2v_j²)` as float64 constants. Do not recompute logs of constants inside the loop.
5. Normalising constants (grid posteriors, integrated likelihoods): `jax.nn.logsumexp` / `jax.scipy.special.logsumexp`.
6. MH: compare `log u` (with `u ≥ tiny`) to the log ratio. Never compute `exp(log_ratio)`.
7. Under vmap one lane's NaN does not spread, but it poisons that chain. Add the indicator draw to the error-flag check (`isfinite` of the row sums).

---

## 10. Memory for stored draws

Per chain and kept draw (float64, 8 bytes):

| Item | Count | n=20, p=4, T=300 |
| :--- | :--- | ---: |
| `γ` (n·k) + `a` (n(n−1)/2), compact | 1620 + 190 | 14.5 KB |
| SV params (μ, φ, σ) and `h_T` | 4n | 0.6 KB |
| t / outlier scalars (ν, p_o, …) | O(n) | <1 KB |
| full `h` path | nT | 48 KB |
| indicators `s` (int32), if stored | nT | 24 KB (do not store) |

At 4 chains × 10,000 kept draws this is **0.6 GB** for coefficients and scalars and **+1.9 GB** with full log-vol paths. Recommendations:

- By default, store coefficients, scalar vol parameters and the last `h_{T}` (and `h_{T−p+1..T}` if a block needs them), which is all forecasting needs (SPEC §6.1).
- Store full paths thinned (`path_thin`, for example every 10th kept draw), or accumulate running moments in the carry (Welford mean/var: O(1) memory). Quantiles need draws, so store thinned ones.
- Store the padded `θ` compactly: drop the masked entries on the host when converting to `xarray.DataTree` `(chain, draw, …)`.
- The benchmark's 1000 kept draws of `(n×K_max)` coefficients plus 4n scalars took 66.6 MB for 4 chains, which matches the formula.
- During sampling the scan output buffer is `C × n_save_chunk × draw_size`. With chunks of 250 it is about 17 MB at n=20, which is negligible.
- float32 *storage* of paths would halve memory, but it conflicts with the letter of R4 (SPEC §3). Not recommended without a decision record.

---

## 11. Edge cases (implementation checklist)

- **`T_eff < T_pad`:** apply the masks everywhere. That covers mean weights `w = m/λ`, `D_t`, `b_t`, the `Q` decoupling, SV sufficient statistics (`Σ_t m_t m_{t−1}(…)`), IG shape `T_eff/2`, and the log-likelihood used in SBC. The forecast state is `h[T_eff − 1]` via a dynamic index, not `h[-1]`.
- **Equation 1** has no `a`. Its mask row is all zeros and its padded coordinates are pure N(0,1) dummies.
- **Exact zero residuals** (constant series, padded rows) are handled by masking before the log and by the SV note's offset.
- **Very small σ² or φ → 1:** `Q` entries reach 1/σ² ≈ 1e8. Cholesky on SPD tridiagonal needs no pivoting and is backward stable, and float64 is fine. `Q` stays PD for any φ (including the random walk) because `HᵀΩ⁻¹H` is PD.
- **Gamma with tiny shape** (Phase 3 global-local): use `loggamma`.
- **Truncation** in the far tail: use S6.
- **`u = 0`** from `uniform`: use `minval=tiny` wherever `log u` or `u^{1/a}` is taken.
- **`fold_in` data** must be `0 ≤ data < 2³²` (uint32). 10⁹ iterations fit.
- **Donated state** is invalid after the call. Tests must not reuse it.
- **x64 toggled off** by the user after import: the engine context manager plus dtype asserts catch it.
- **`jax_num_cpu_devices`** set after JAX initialised raises `RuntimeError` *(verified)*. Fall back to vmap and warn.
- **Import-time backend initialisation.** Any module-level `jnp` array, such as Omori constants stored as `jnp.array`, initialises the CPU backend during `import rai_bvar`. After that, `set_cpu_devices` can never work. *Cross-check:* `jax.config.update("jax_enable_x64", True)` plus NumPy constants at import, followed by `jax_num_cpu_devices=4`, works. A module-level `jnp.array(...)` before it raises `RuntimeError`. Keep module-level constants as NumPy and convert them inside the jitted functions.
- **Chains vs devices:** `C % n_devices == 0` for `shard_map`.
- **CCCM loop** uses dynamic `j` inside `fori_loop`: masks such as `jnp.arange(n) >= j` replace Python slicing `[j:]`, which would need static bounds.
- **Recompile storms:** a new spec object per backtest window (not equal-hashing) or changing pytree structure. Test for both (§13.1).
- **Version drift:** changes to JAX's random streams appear only in the changelog. Bitwise golden tests must carry the pinned version and are regenerated on upgrade. Distributional tests are version-agnostic.

---

## 12. Benchmarks to expect on CPU

Machine: Apple M1 Pro, JAX 0.12.0.1 CPU, float64. **Shared and heavily loaded** (load average 25–50); minimum of repeats. The full sweep is in `jaxbench/sweep.py` and `sweep_opt.py`. It runs the true Gaussian, indicator and tridiagonal kernels; the SV-parameter step is a cheap stand-in (conjugate σ², truncated-normal φ, Gaussian μ, with no ASIS or MH), so expect +10–20% for the real block.

**Full sweep, `Linear × SV`, n=20, p=4 (k=81, K_max=100), T=300, 4 chains (SPEC Phase-2 reference config):**

Ranges below cover two separate runs. The second is `jaxbench/run_all.log`, run at load average 25–34.

| Configuration | ms per iteration (all 4 chains) | 10k iterations | compile |
| :--- | ---: | ---: | ---: |
| single device, vmap chains | 14.5–18.0 | 145–180 s | 1.5–2.8 s (0.37 s cached) |
| 4 CPU devices, `shard_map` over chains | **8.2–8.6** | **82–86 s** | ≈2 s (in first call) |
| `Linear × Constant`, single device (precomputed cross-products) | 3.7–4.4 | 37–44 s | 1.0–1.8 s |
| `Linear × Constant`, 4 CPU devices | 1.2–1.3 | 12–13 s | — |

The SPEC target (4 × 10,000 draws in under 2 minutes) is **met with 4 CPU devices** (≈85 s, plus ≈10–20% for the real SV-parameter block) and **missed by ≈20–50% with single-device vmap**, on this loaded machine. Re-measure idle; the multi-device runner should be the default when the user opts in.

**Component microbenchmarks** (B = 80 lanes = 4 chains × 20 equations, T=300):

| Kernel | ms |
| :--- | ---: |
| Weighted Gram matrix 80 × (300×100)ᵀ(300×100): einsum / matmul / `symmetric_product` | 6.0–8.0 / 5.9–8.1 / 4.7 |
| Batched Cholesky 80 × 100² | 2.3–2.7 (SciPy per-matrix loop: 2.3–2.7; NumPy batched: 2.7–4.1) |
| `cho_solve` / one triangular solve (80 × 100) | 0.9–1.8 / 0.8–1.4 |
| Full Gaussian draw (Gram + Cholesky + solves + RNG) | 10.2–12.0 |
| Tridiagonal sampler A / C / B (§8.3) | 0.21–0.22 / 0.74–1.19 / 2.1–3.1 |
| 24,000 float64 normals: threefry2x32 / philox4x32 | 0.48–0.55 / 0.42–0.70 |
| 24,000 float64 uniforms: threefry2x32 / philox4x32 / rbg | 0.26–0.49 / 0.07–0.12 / 0.18–0.27 |
| Indicators 80×300×10: inverse-CDF S3 / `random.categorical` | 1.6–2.0 / 7.0 |
| 80 exact gamma draws (while-loop overhead dominated) | 0.13–0.2 |
| SV block alone (indicators → h → params) per iteration | 3.1–3.3 (0.13–0.14 µs per series·t) |

Single series, T=500, 1000 SV iterations: **0.10 s** in JAX (both runs). For orientation only (different hardware, older CPU): Kastner & Frühwirth-Schnatter (2014, §4) report 0.23–0.24 s per 1000 draws at T=500 for stochvol's C implementation on a 2.67 GHz Intel i5 M560 (one core).

**Reduced form (CCCM sequential coefficient block only, §2.7):** n=20, k=81, T=300, 4 chains: 9.1–13.3 ms per iteration. n=20, k=261, T=647: 24–52 ms (1 chain) and 32 ms (4 chains vmapped, one run). Published reference: ≈254 ms per draw (2.542 s per 10 draws) for the correct triangular algorithm in MATLAB (CCCM corrigendum Fig. 1, 3.5 GHz Core i7, N=20, 13 lags). Whether that timing includes the volatility steps is not stated, so treat the comparison as indicative.

Rules of thumb (cost model per sweep, per chain):
- Gram matrices: `n · 2T·K_max²` flops (half that with SYRK). Measured on M1 Pro with the multi-threaded XLA dot: ≈60–80 GF/s at K=100 (5.9–8.1 ms for 480 MFLOP), ≈100–110 GF/s at K=200 and only ≈20–25 GF/s at K=50 (`run_all.log`). This term dominates for SV.
- Cholesky: `n · K_max³/3`, roughly 5–15 GF/s for 100² batches (LAPACK per matrix).
- SV: ≈0.13 µs per (series × t) including RNG.
- RNG: ≈23 ns per float64 normal and ≈11 ns per uniform (threefry, CPU).
- Constant vol removes the Gram term entirely.

---

## 13. How to test it

### 13.1 Deterministic kernel tests (every PR, fast)

**Pattern: separate randomness from transformation.** Each kernel has a pure core `kernel(params, z_or_u)` and a thin wrapper that draws `z`/`u` from the key. Tests feed the same `z` to the JAX core and to a dense NumPy/SciPy reference:

| Kernel | Oracle | Tolerance |
| :--- | :--- | :--- |
| S2 Gaussian block | `solve(K, r) + solve(chol(K).T, z)` dense | 1e-10 relative (verified at K=6) |
| Padding exactness | Unpadded `K_i`, same `z[:K_i]` | 1e-12; padded coords equal `z` tail |
| S4 tridiagonal | dense `Q`, `slogdet` | 1e-12 (verified 3e-16 / 6e-14) |
| SV `Q, b` builder | dense `HᵀΩ⁻¹H + D` from definitions, incl. masks and `T_eff` | exact |
| S3 categorical | `u`-inversion by hand; frequency χ² test over 10⁶ draws | χ² p > 1e-3 |
| S6 truncated normal | `scipy.stats.truncnorm` KS over several seeds and intervals (including far-tail) | KS p-values uniform |
| Gamma/IG/Beta wrappers | moments vs closed form (10⁶ draws) | 4 MC s.e. |

**Infrastructure tests:**
- dtype: all float leaves float64; keys typed.
- Key discipline: the same seed gives bitwise-identical draws for chunk sizes {50, 250}, thin {1, 3} with appropriate `it` alignment, `C` ∈ {3, 5} (prefix chains equal), and vmap vs a Python loop over chains (snippets S1/S5 verified this).
- Key reuse: one full sweep under `jax.debug_key_reuse(True)` raises nothing. A deliberately broken sweep that reuses a key raises.
- Recompilation: two fits with new data and hyperparameter values in the same `T` bucket give 0 new compiles (`jax_log_compiles` counting).
- Failure path: a non-PD prior injected at iteration 7 makes the host raise with block, equation and iteration 7.
- Reproducibility metadata: `jax.__version__`, jaxlib, PRNG impl and `jax_threefry_partitionable` are recorded.

### 13.2 Geweke (2004) joint-distribution tests in JAX

The JAX-specific design requirement is that the **data must be an argument of the sweep**, and the VAR design builder (lags, masks) must be written in `jnp` so it can run inside the compiled loop. The successive-conditional simulator redraws `y`, and with it `X` (its lags), every iteration, conditioning on fixed initial observations `y_{1−p..0}`.

```python
def geweke_marginal_conditional(key, M, prior_draw, simulate_y, g):
    def one(k):
        th = prior_draw(random.fold_in(k, 0)); y = simulate_y(random.fold_in(k, 1), th)
        return g(th, y)
    return jax.vmap(one)(random.split(key, M))                      # iid -> plain s.e.

def geweke_successive_conditional(key, M, th0, y0, sweep, simulate_y, g):
    def body(carry, it):
        th, y = carry
        kit = random.fold_in(key, it)
        th = sweep(random.fold_in(kit, 0), it, th, build_data(y))  # one full Gibbs sweep
        y = simulate_y(random.fold_in(kit, 1), th)
        return (th, y), g(th, y)
    return lax.scan(body, (th0, y0), jnp.arange(M, dtype=jnp.uint32))[1]   # autocorrelated -> MCSE
```

- Test functions `g`: every scalar parameter and its square, selected cross-products (`a_ij γ_i`, `μ_i φ_i`), log-vol path means and the last state `h_T`, the count of each mixture component, and the log-likelihood.
- z-statistic: `(ḡ_MC − ḡ_SC)/√(s²_MC/M + MCSE²_SC)`, with MCSE from batch means or ArviZ ESS. Apply a Holm correction across all `g`.
- Small config for every PR (n=2, p=1, T=30, proper and moderately tight priors, M≈2·10⁵ SC sweeps, which take seconds). Larger config nightly.
- Also run per-block Geweke (other blocks fixed at prior draws) to localise failures.
- The CCM-2019 failure mode (SPEC R6) is already present at **n = 2**. The CCCM corrigendum (§1, eqs. (7)–(8)) shows the missing term `p(y^{(2)} | y^{(1)}, π^{(1)}, x, Σ_T)` with "the simplest case N = 2": equation 1's coefficients enter equation 2.
  - The reduced-form block therefore needs its own Geweke config with `n ≥ 2` to detect the flaw. Use `n = 3` as the PR config so that a middle `j` with two later equations and the dynamic `i ≥ j` masks are also exercised.
  - The missing term changes both the location and the spread of `π^{(1)} | y`. So include second moments, and also parameter–data cross moments such as `β_1 · y_{2,t}`. Gandy & Scott (2020, §4.1) find that statistics on both data and parameters are needed to detect some errors.
  - The flawed sampler can be kept as a *negative control*: the Geweke test must reject it. That this rejection happens at the PR-config size is UNVERIFIED and should be measured.

### 13.3 SBC (Talts et al. 2018; Modrák et al. 2025) in JAX

```python
def sbc_rank(key, L, thin):
    th = prior_draw(random.fold_in(key, 0)); y = simulate_y(random.fold_in(key, 1), th)
    draws = run_chain(random.fold_in(key, 2), build_data(y), n_keep=L, thin=thin)  # random init + burn-in
    q_true = test_quantities(th, y)                       # params + log-likelihood (Modrák et al.)
    q_draw = jax.vmap(lambda d: test_quantities(d, y))(draws)
    n_less = jnp.sum(q_draw < q_true, axis=0)
    n_tie = jnp.sum(q_draw == q_true, axis=0)
    u = random.uniform(random.fold_in(key, 3), n_less.shape)     # ONE uniform per quantity
    return n_less + jnp.floor(u * (n_tie + 1)).astype(n_less.dtype)   # rank in 0..L, uniform over ties
ranks = jax.jit(jax.vmap(sbc_rank, in_axes=(0, None, None)), static_argnums=(1, 2))(
            random.split(random.key(seed), N_rep), L, thin)
```

- **Tie-breaking (corrected in the cross-check).** The first version flipped an independent coin per tied draw. That gives a Binomial(#ties, ½) offset instead of a uniform one, and it makes ranks of *discrete* test quantities non-uniform even for a correct sampler: mixture-component counts, outlier counts, or ν on a grid.
  - Simulation with 2·10⁵ replications, L=9 and exchangeable draws on {0,1,2}: coin-flip ranks gave χ² p = 0 (edge bins 0.066 vs 0.1), while the uniform-over-ties rule above gave p = 0.26.
  - For continuous quantities ties have probability 0 and both rules agree.
- Return only ranks (O(N_rep × #q) memory), and shard replications across CPU devices.
- Test on the host with the ECDF-envelope test (Säilynoja, Bürkner & Vehtari 2022), Holm across quantities, at 1% (SPEC §10.3).
- **Thinning.** Autocorrelated draws make ranks non-uniform. Talts et al. (2018, §5.1 and Algorithm 2) thin by `⌈N/N_eff⌉`, taking the largest such factor over all quantities, so that `L` roughly independent draws remain. Pick `thin` from a pilot ESS per test configuration, not per replication, so the compiled shape stays fixed.
- **Initialisation.** Do **not** start the chain at the true `θ̃` and run it only forward. Gandy & Scott (2020, arXiv:2001.06465v2, §2.2) state that for a chain "initiated at θ₁ = θ … the null distribution of R₁(θ_{1:L}) is generally not uniform", because the sequence is not exchangeable. (This confirms the first author's reasoning, which had been marked UNVERIFIED.) The default is random initialisation (`STREAM_INIT`), burn-in, then thinning as above. This is approximate, with residual autocorrelation.
- **Exact alternative** (Gandy & Scott 2020, Algorithm 2 and Prop. 2.3, after Besag & Clifford 1989):
  - Draw `M ~ U{1..L}` and set `θ_M = θ̃`. Run the kernel forward `L − M` steps and *backward* `M − 1` steps. The rank of `θ_M` is then exactly uniform if the kernel is π-reversible.
  - A systematic-scan Gibbs sweep is not reversible, and Gandy & Scott's own systematic-scan experiment fails the test (§4.1). The backward steps would have to use the time-reversed sweep, which runs the blocks in reverse order and keeps the `(θ, s) | h` block intact.
  - That this makes the construction exact for our sweep is my derivation and is UNVERIFIED in the literature. Optional; not needed for v1.

### 13.4 Parity and end-to-end oracles

- The SV kernel (`Linear` mean switched off) is compared with **stochvol** posterior quantiles of μ, φ, σ and the h path within 4 MCSE (SPEC §10.4). The prior mapping, offset and initial-state convention are in the SV note. JAX-specific point: drive the comparison through the *same* public runner with a "zero mean" spec, so that the runner, key discipline and masks are exercised as well.
- The reduced form is compared with **bayesianVARs** (SPEC §10.4). Mapping is in the CCCM/Minnesota notes.
- Analytic twin (Phase 3 candidate, SPEC §10.3): `Linear(form="structural") × Constant` vs the exact iid normal-inverse-gamma posterior at n=20. This is the only realistic-size correctness check, and it costs ~40 s of MCMC.
- Performance regression test (`tests/bench`): the §12 reference config, recording ms per iteration and compile time with a threshold, run pre-release (SPEC §10.1).

---

## 14. UNVERIFIED items

1. All absolute timings were taken on a **heavily loaded shared machine**, so treat them as upper bounds. Re-run `jaxbench/run_all.sh` idle.
2. The batch-parallel CPU LAPACK (0.10.0 changelog, verified to exist) showed no speed-up under load; behaviour on an idle machine is unknown.
3. How to make a JAX 0.12 CPU process single-threaded for multi-process backtests (§6). `intra_op_parallelism_threads=1` without dashes is silently ignored, and `--xla_cpu_multi_thread_eigen=false` did not make the process single-threaded in the cross-check.
4. *(Resolved in cross-check)* `vmap(cond)` with a batched predicate lowers to `select_n` (jaxpr inspected). Only the runtime cost remains unmeasured.
5. *(Resolved in cross-check)* `jax_log_compiles` text and the `memory_analysis`/`cost_analysis` fields on CPU (§7).
6. The +10–20% estimate for the real SV-parameter block (ASIS, MH) over the benchmark's stand-in.
7. Philox4x32 statistical quality was not tested by us. JAX's docs list it as cross-backend identical and vmap-exact. JAX says its `rbg` variants were "not … subject to empirical randomness testing"; no such statement covers Philox. Not recommended for v1 (threefry is the default).
8. *(Resolved in cross-check)* The SBC "do not initialise at θ̃ and run only forward" caution is stated by Gandy & Scott (2020, §2.2). Still UNVERIFIED: whether the reverse-order-scan variant of their exact rank test is exact for our sweep (§13.3).
9. Which GIG algorithm Phase 3 should use (Hörmann & Leydold 2014 or Devroye 2014) was not researched here.
10. *(Largely resolved in cross-check)* The Omori constants were checked against three independent copies:
    - the working-paper Table 1 (`dl_omori/nuff.txt`);
    - E. Mertens's MATLAB `getKSC10values.m` (constants only);
    - the stochvol header `sv_utils_latent_states.h` (constants only, GPL, read to confirm published constants as SPEC §2.1 allows).

    The mixture's moments also match log χ²₁: mean −1.27028 vs ψ(½)+log 2 = −1.27036, variance 4.93373 vs π²/2 = 4.93480, Σπ = 1.00000. The printed J. Econometrics table itself was not opened. The SV note stays authoritative.
11. The claim that a Geweke test at the PR config (n=2–3, T=30) actually rejects the CCM-2019 flawed sampler, to be measured with the negative control in §13.2. Also the "M ≈ 2·10⁵ SC sweeps take seconds" estimate.
12. Whether the CCCM corrigendum's Fig. 1 timings include the volatility steps.

---

## 15. Sources

JAX (Apache-2.0; reading the source is permitted):
- PyPI release history (0.12.0.1, 2026-10-21; requires Python ≥ 3.12): https://pypi.org/project/jax/
- Changelog (x64 context manager 0.8.0; `random.Key` 0.12.0; LAPACK batch parallelism and `tridiagonal_solve(perturb_singular)` 0.10.0; batched-1-D-solve warnings 0.10.0): https://docs.jax.dev/en/latest/changelog.html
- `jax.random` module docs (typed keys, impl table, rbg vmap caveat): https://docs.jax.dev/en/latest/jax.random.html
- API compatibility (random outputs may vary across versions; only distributions are covered): https://docs.jax.dev/en/latest/api_compatibility.html
- Default dtypes and the X64 flag: https://docs.jax.dev/en/latest/101/default_dtypes.html
- Persistent compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- `lax.scan` (unroll, fixed carry, single WhileOp): https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
- `lax.linalg.tridiagonal_solve`: https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html
- Key reuse checker: https://docs.jax.dev/en/latest/jax.experimental.key_reuse.html
- Installed source `jax/_src/random/core.py` (`_truncated_normal`, `_gamma_one`, `_beta`, `_dirichlet`, `_normal_real`) and `jax/_src/lax/linalg.py` (`potrf_ffi`, `trsm_ffi`, `gtsv_ffi` lowerings), jax 0.12.0.1.

Papers:
- Salmon, Moraes, Dror & Shaw (2011), "Parallel random numbers: as easy as 1, 2, 3", SC'11 (Threefry/Philox).
- Marsaglia & Tsang (2000), "A simple method for generating gamma variables", ACM TOMS 26(3), 363–372.
- Robert (1995), "Simulation of truncated normal variables", Statistics and Computing 5(2), 121–125.
- Kool, van Hoof & Welling (2019), Gumbel-top-k, ICML (cited in the `jax.random.categorical` docstring).
- Rue (2001), "Fast sampling of Gaussian Markov random fields", JRSS-B 63(2), 325–338.
- Chan & Jeliazkov (2009), "Efficient simulation and integrated likelihood estimation in state space models", IJMMNO 1(1), 101–120.
- McCausland, Miller & Pelletier (2011), "Simulation smoothing for state–space models: a computational efficiency analysis", CSDA 55(1), 199–212.
- Stone (1973), "An efficient parallel algorithm for the solution of a tridiagonal linear system of equations", J. ACM 20(1), 27–38 (recursive doubling, used in variant C).
- Papandreou & Yuille (2010), "Gaussian sampling by local perturbations", NeurIPS (variant B).
- Bhattacharya, Chakraborty & Mallick (2016), "Fast sampling with Gaussian scale mixture priors in high-dimensional regression", Biometrika 103(4), 985–991.
- Omori, Chib, Shephard & Nakajima (2007), "Stochastic volatility with leverage: fast and efficient likelihood inference", J. Econometrics 140(2), 425–449. Table 1 checked in the working-paper text `phase2-research/dl_omori/nuff.txt`, lines 140–151.
- Kastner & Frühwirth-Schnatter (2014), "Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models", CSDA 76, 408–423, §4 timing paragraph (`phase2-research/_sv_papers/kfs2014_asis.txt`, lines 687–695).
- Carriero, Chan, Clark & Marcellino (2022), "Corrigendum to: Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors", J. Econometrics 227(2), 506–512, doi:10.1016/j.jeconom.2021.11.010. The working-paper version is FRB Cleveland WP 16-17C (Nov 2021), doi:10.26509/frbc-wp-201617corrigendum (`phase2-research/dl_cccm/wp1617c.txt`). Used here:
  - §1, eqs. (7)–(8): the N = 2 illustration of the missing term;
  - eqs. (14)–(15): the combined-weight precision and mean;
  - §4 and Fig. 1, lines 355–430: CTA 2.542 s and SWA 33.10 s per 10 draws at N=20; CTA ≈3.25× the flawed TA and ≈13× faster than SWA; FRED-MD 1960–2014, 13 lags, 3.5 GHz Core i7.
- Del Negro & Primiceri (2015), "Time varying structural vector autoregressions and monetary policy: a corrigendum", REStud 82(4), 1342–1345 (step order, SPEC §4.3).
- Geweke (2004), "Getting it right: joint distribution tests of posterior simulators", JASA 99(467), 799–804.
- Talts, Betancourt, Simpson, Vehtari & Gelman (2018), "Validating Bayesian inference algorithms with simulation-based calibration", arXiv:1804.06788 (§5.1 and Algorithm 2: thinning by ⌈N/N_eff⌉).
- Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková, Gelman & Vehtari (2025), "Simulation-based calibration checking for Bayesian computation: the choice of test quantities shapes sensitivity", Bayesian Analysis 20(2), 461–488, doi:10.1214/23-BA1404 (advance online 2023; volume and pages from Crossref).
- Säilynoja, Bürkner & Vehtari (2022), "Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison", Statistics and Computing 32(2), article 32, doi:10.1007/s11222-022-10090-6.
- Gandy & Scott (2020; v2 Sept 2021), "Unit testing for MCMC and other Monte Carlo methods", arXiv:2001.06465v2. Used here: §2.2 (a chain started at θ̃ does not give uniform ranks; exact rank test via random position plus forward/backward runs, Algorithm 2 and Prop. 2.3, after Besag & Clifford 1989) and §4.1 (systematic scan fails the exact rank test; data–parameter statistics are needed).
- Besag & Clifford (1989), "Generalized Monte Carlo significance tests", Biometrika 76(4), 633–642 (cited via Gandy & Scott; not opened).
- Kastner & Frühwirth-Schnatter (2014), §2.2–2.3 (`kfs2014_asis.txt`, lines 114–120 and 220–230): origin of the name "all without a loop" (AWOL) for the Rue (2001) / McCausland et al. (2011) method.
- Omori et al. (2007), 10-component constants, second and third copies: `phase2-research/dl_ccmm_constants_only/getKSC10values.m` (E. Mertens; constants only) and `phase2-research/dl_gpl_constants_only/sv_utils_latent_states.h`, lines 45–47 (stochvol; constants only).

Clean-room statement: every snippet above is original. No GPL source (stochvol, bayesianVARs, bvartools) was opened by the first author. In the cross-check, the stochvol header `sv_utils_latent_states.h` was searched only for the 10 published mixture constants (lines 45–47), which SPEC §2.1 allows ("confirm published constants"). No code was transcribed. JAX's own Apache-2.0 source was read only to confirm the behaviour of its samplers, docstrings and lowerings.

---

## Cross-check log

Independent cross-check, 21 Oct 2026. Same JAX 0.12.0.1 / jaxlib 0.12.0.1 CPU install, NumPy 2.5.4, SciPy 1.18.1. Scripts are in `/private/tmp/claude-501/-Users-yigitokar-rai-bvar/2cd712bc-93c9-47b9-880a-ba6190d87527/scratchpad/xc07/`: `api_check.py`, `tn_check.py`, `shard_check.py` and `threads_check.py`. The rest were run inline. `jaxbench/snippets.py` and `jaxbench/truncnorm.py` were re-run, and all their self-checks pass.

### Corrections

| # | Where | Was | Now | Evidence / source |
| :-- | :-- | :-- | :-- | :-- |
| 1 | §13.2 | "Reduced-form Geweke config needs `n ≥ 3`, because the coupling only appears with at least two later equations" | The CCM-2019 flaw is already present at **n = 2**. Use n ≥ 2, with n = 3 as the PR config for mask coverage. Added a flawed-sampler negative control and data–parameter cross moments | CCCM corrigendum, WP 16-17C §1, eqs. (7)–(8): "consider the simplest case N = 2"; missing term `p(y^{(2)} \| y^{(1)}, π^{(1)}, x, Σ_T)` |
| 2 | §13.3 SBC snippet | Ties broken by an independent coin per tied draw | `n_less + floor(u·(n_tie+1))` with one `u` per quantity | Simulation, 2·10⁵ reps, L=9, discrete exchangeable draws: coin rule χ² p = 0 (edge bins 0.066 vs 0.10); corrected rule p = 0.26 (NumPy) and p = 0.20 / 0.49 (JAX version of the snippet) |
| 3 | §13.3 | "Do not start at θ̃" marked as the author's own UNVERIFIED reasoning | Supported in the literature. Added the Talts thinning rule and the exact Besag–Clifford alternative, with the systematic-scan caveat | Gandy & Scott, arXiv:2001.06465v2, §2.2 ("initiated at θ₁ = θ … generally not uniform"), Algorithm 2, Prop. 2.3, §4.1; Talts et al. 2018, §5.1 |
| 4 | §6 | `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` makes processes single-threaded; "both flags are accepted" | `intra_op_parallelism_threads=1` (no `--`) is silently ignored, like any non-dashed token; with `--` it aborts as an unknown flag. `--xla_cpu_multi_thread_eigen=false` left the CPU/wall ratio at 1.94 (2.17 without it). No verified single-thread recipe | Runs with `XLA_FLAGS` set to `bogus_flag_xyz=1` (accepted) and `--intra_op_parallelism_threads=1` (fatal: `Unknown flag in XLA_FLAGS`); `threads_check.py` |
| 5 | §0, §9.1 | `gamma(1e-3)` exact 0 "in 4 of 5 draws" | ≈49% of draws; P ≈ `exp(−744.4·a)`; 0.09% at a=0.01, none at 0.1 | 2·10⁵ draws per shape; derivation from `_gamma_one` (`G_{a+1}·U^{1/a}`, jax/_src/random/core.py) and the smallest float64 subnormal (log ≈ −744.4) |
| 6 | §8.3 | "McCausland, Miller & Pelletier (2011) … ('all without a loop')" | The AWOL name is from Kastner & Frühwirth-Schnatter (2014, §2.2–2.3), describing the Rue / McCausland et al. method. SPEC §8 has the same misattribution, so flag it to the SPEC owner | `kfs2014_asis.txt` lines 116–120 and 222–226 |
| 7 | §2.5 | Grid joint draw "about 30 ms at ~6–7 ns per element" | 32–40 ms at 6.7–8.3 ns per element | 1.6–2.0 ms / 240k elements (§12 table and `run_all.log`) |
| 8 | §0, §6 | Multi-device "1.7–3× faster" | SV 1.7–2.2×; Constant 2.8–3.7× | Ratios of the §12 table entries |
| 9 | §12 | Gram matrices "roughly 70–120 GF/s" | ≈60–80 GF/s at K=100; ≈100–110 at K=200; ≈20–25 at K=50 | `run_all.log` (60 GF/s at K=100, 112 at K=200, 21–25 at K=50); 480 MFLOP / 5.9 ms = 81 GF/s |
| 10 | §2.7, §12 | CCCM Fig. 1 = "full corrected sampler" / "whole corrected sampler" | Correct triangular algorithm, 10 draws, average over 10 chains, FRED-MD 1960–2014, 13 lags. Whether SV steps are included is not stated | WP 16-17C §4 text and the Fig. 1 caption |
| 11 | §3 | x64 off: "silently produces 32-bit values with a warning" | Emits a `UserWarning` and truncates; tests should turn the warning into an error | Run on 0.12.0.1; docs page "Default dtypes and the X64 flag" |
| 12 | §2.2 | Padding cost "(K_max/K̄)³ ≈ 1.35" | Exact ratios 1.33 (Cholesky) and 1.22 (Gram) | `n·K_max³/Σ K_i³` with `K_i = 81 + i − 1` |
| 13 | §5 | unroll times "0.21/0.27/0.29 ms", compile "0.88/1.38 s" | Two-run ranges 0.21–0.22 / 0.27–0.39 / 0.29–0.31 ms; compile 0.18 / 0.9–1.3 / 1.4–1.8 s | `run_all.log` (0.221 / 0.392 / 0.305 ms; 0.18 / 1.27 / 1.83 s) |
| 14 | §9.1 | Normal "≈2× the cost of a uniform" | 1–2×; the two runs disagree | `run_all.log`: threefry normal 0.478 ms vs uniform 0.491 ms |
| 15 | §15 | Modrák et al. pages UNVERIFIED | Bayesian Analysis 20(2), 461–488, doi:10.1214/23-BA1404 | Crossref record for that DOI |
| 16 | §15 | Säilynoja et al. "Statistics and Computing 32, 32" | 32(2), article 32, doi:10.1007/s11222-022-10090-6 | Search results / Aalto repository |
| 17 | §15 | Clean-room statement said no GPL source was opened | Discloses that the cross-check searched the stochvol header for the published constants only | SPEC §2.1 permits confirming constants |

### Additions (not errors in the original)

- §2.2: the constant-vol cross-product shortcut under conditional centring is the congruence `T_iᵀ G T_i`. The time mask was renamed `D_m` to avoid a clash with the Minnesota mean `M`.
- §2.3: the explicit-`h_0` state variant (`Q_00 = 1/σ²`, `D_0 = 0`) and the left-padding nuance. Both were re-derived from `HᵀΩ⁻¹H`.
- §2.4: Omori constants confirmed in two more independent copies, plus moment checks against log χ²₁.
- §2.7: the combined-weight precision matched to CCCM eqs. (14)–(15).
- §6: sharded `shard_map` output is bitwise identical to single-device `vmap` (`shard_check.py`).
- §7: exact `jax_log_compiles` record text and logger name; a warning that eager ops also log compiles; persistent-cache 1.0 s threshold and its effect on small test configs; cache-key components and the docs' trust warning; CPU fields of `memory_analysis` and `cost_analysis`.
- §9.3: same-uniform comparison of S6 with `scipy.stats.truncnorm.ppf`, with max relative error ≤ 1e-14 on 10 intervals including (37, 38) and (−3, 50).
- §11: module-level `jnp` constants initialise the backend and break `set_cpu_devices`; keep them as NumPy (verified).

### Verified without change

- **Precision-form draw.** `θ = L⁻ᵀ(L⁻¹r + z)` with `K = P_0 + Z'WZ` and `r = P_0 m_0 + Z'Wy` gives mean `K⁻¹r` and covariance `K⁻¹`. It is algebraically identical to SPEC §8's `cho_solve` + `solve_triangular` form, with one solve fewer. S2 matches a dense draw (`snippets.py`).
- **Padding exactness.** Inactive trailing columns, `n − i` of them, give a block-diagonal `K` and `L`. The active draw equals the unpadded draw with `z[:K_i]`.
- **Conditional centring.** `M ã_i = M_{:,i} + M_{:,<i} a_i`, and `y_i = Xδ_i + (X M_{:,<i} − Y_{<i}) a_i + e`.
- **Tridiagonal SV precision.** All entries re-derived from `HᵀΩ⁻¹H + D`, and `Q_11 = 1/σ² + D_1` under the stationary start.
- **Bidiagonal Cholesky recursions.** `c_t = e_t/l_{t−1}`, `l_t = √(a_t − c_t²)`, plus the forward and back substitutions. S4 matches dense `solve`/`slogdet`.
- **Other formulas:**
  - Variant B covariance `HᵀΩ⁻¹H + D`;
  - Variant C pivot recursion (Möbius form);
  - block-tridiagonal Cholesky formulas;
  - constant-vol IG conditional `IG(a_0 + T_eff/2, b_0 + SSR/2)`;
  - JAX gamma/IG parametrisation mapping;
  - Omori log-weight `c_j − κ_j d²`.
- **JAX facts** (source, docstrings, changelog or runs):
  - PRNG impl table and the rbg/unsafe_rbg vmap caveat (module docstring in `jax/random.py`);
  - philox2x32 32-bit key space;
  - `jax_threefry_partitionable` default since 0.5.0;
  - API-compatibility quote;
  - `jax.random.Key` added in 0.12.0 (to become the return type in 0.13.0);
  - `jax.enable_x64` non-experimental since 0.8.0;
  - LAPACK batch parallelism, `perturb_singular` and the batched-1-D-solve warning, all in 0.10.0;
  - `fold_in` = `threefry_2x32(key, threefry_seed(data))`, with `OverflowError` at 2³²;
  - key-reuse checker: errors on double use, passes repeated `fold_in` from one parent;
  - vmap/loop and jit/eager bitwise equality for `normal` and `gamma`;
  - chunk and chain-count invariance;
  - Cholesky NaN on non-PD input;
  - `symmetric_product` lower-triangle guarantee and `cholesky(symmetrize_input=False)` reading only the lower triangle;
  - the `potrf_ffi`/`trsm_ffi`/`gtsv_ffi` lowerings;
  - `tridiagonal_solve` signature;
  - no banded solver in `jax.scipy.linalg`;
  - `fori_loop` lowering to `while` with a traced bound (and to `scan` with a static one);
  - donation on CPU;
  - `jax_num_cpu_devices` `RuntimeError` after backend initialisation;
  - `shard_map` `ValueError` on unsharded inputs;
  - `scan` docstring "lowered to a single WhileOp";
  - `gamma` `'approximate'` docstring ("small bias");
  - built-in `truncated_normal` returning 1.797e308 on (9, ∞);
  - categorical implemented as Gumbel-max (Kool et al. reference in its docstring);
  - `bernoulli` `mode='low'` float32 warning;
  - persistent-cache default 1.0 s;
  - JAX 0.12.0.1 released 2026-10-21 and requiring Python ≥ 3.12 (PyPI); 0.12.0 released 2026-10-19 (changelog).
- **Numbers:**
  - CCCM 2.542 s / 33.10 s / 3.25× / 13×;
  - KFS 2014 timing of 0.23–0.24 s per 1000 draws at T=500 on an i5 M560;
  - memory table (14.5 KB, 48 KB, 0.6 GB and 1.9 GB at 4×10k; 66.6 MB benchmark buffer);
  - ≈1e9 uniforms per reference run.
- **Citations checked against Crossref or the paper text:** Stone 1973; McCausland et al. 2011; Chan & Jeliazkov 2009; Rue 2001; Bhattacharya et al. 2016; Marsaglia & Tsang 2000; Geweke 2004; Kastner & Frühwirth-Schnatter 2014; Robert 1995; Besag & Clifford 1989; CCCM 2022 (J. Econometrics 227(2), 506–512); Kool et al. 2019 (docstring).

### Still UNVERIFIED after the cross-check

See §14, items 1–3, 6–7, 9, 11–12, and the reverse-scan part of item 8.
- **Timings:** absolute timings and LAPACK batch speed-up, both measured on a loaded machine.
- **Process threading:** a single-thread recipe for multi-process backtests.
- **SV cost estimate:** the +10–20% for the real SV-parameter block.
- **Philox quality:** Philox4x32 statistical quality.
- **Phase 3:** the GIG algorithm choice.
- **Geweke power:** whether the PR-size Geweke config rejects the CCM-2019 sampler, and the "seconds" runtime estimate.
- **CCCM timing scope:** whether CCCM Fig. 1 includes the SV steps.
- **SBC reverse scan:** exactness of the reverse-scan Besag–Clifford variant for our systematic-scan sweep.
- **Omori journal table:** the printed J. Econometrics table was not opened; three other copies and the moment checks agree.
