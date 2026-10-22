"""Ordering-sensitivity check (SPEC §6.6, R8; N08 §B4; DESIGN §13 WP22).

Every composition with a triangular factorisation ``Σ_t = A⁻¹Λ_tA⁻¹'`` depends on the variable
ordering: through the prior (constant volatility, N08 §B1 (i)) or through the model class itself
(SV, outliers, t; §B1 (ii)). In Phase 2 only ``Linear(form="reduced") ×
Constant("iw_equivalent")`` is order-invariant (Chan 2022, Prop. 1; N08 §B2), besides the analytic
NIW engine. :func:`ordering_sensitivity` measures how much a model's results move when the
variables are reordered, against a Monte Carlo noise floor, and :func:`permute_model` makes the
per-variable settings follow their variables.

**Permutation design** (:func:`permutation_design`; N08 §B4). ``F`` are the free (unpinned)
variables, ``m = |F|``. With ``m! ≤ n_perm`` or ``design="all"`` every ordering of ``F`` is used
(24 for ``m = 4``, as in ARRS). Otherwise the user ordering and its reverse on ``F`` (CKY's
comparison) come first, then the rows of a random Latin square of order ``m`` (``"balanced"``:
each variable in each free position equally often, since the ordering effect is a *position*
effect; for ``m = 3``, where every Latin square holds the user ordering or its reverse, the user
ordering's cyclic shifts), then seeded uniform permutations without replacement; ``"random"``
skips the square.
Pinned variables lead every ordering, in the given order.

**Per ordering** the model is refit on ``data[pinned + π(F)]`` with :func:`permute_model` (R9:
ψ, own-lag means, EB values and every other data-dependent quantity are recomputed from the
permuted window; a ψ tuple follows its variables) and every output is mapped back to the
original labels (labels, never positions; the round trip is asserted).

**Noise floor.** The user ordering is refit ``noise_reps`` more times with other seeds. For each
statistic ``s``, ``σ_MC(s) = max(sd_rep(s), MCSE(s))``: the across-seed sd (``ddof = 1`` over the
user fit and its replicates) and, where it exists, the MCSE of the user fit (mean and quantiles,
Vehtari et al. 2021 §4.3–4.4; ``sd`` by the delta method ``MCSE(s²)/(2s)``; interval lengths
``√(MCSE_lo² + MCSE_hi²)``; LPD as ``ForecastResult.log_pred_density(return_mcse=True)``). The
maximum is the conservative choice: noise is never understated (ARRS's 30-chain check, Fig. 3).
Backtest scores carry no MCSE, so ``mode="backtest"`` needs ``noise_reps ≥ 1`` (otherwise no
signal-to-noise ratio exists and "not material" would mean "undetermined").

**Metrics** (per variable × horizon, joint where defined; over orderings; heuristic thresholds
of N08 §B4 in :data:`THRESHOLDS`):

* ``sd_ratio``: ``max/min`` of the predictive sd (prior mode: of ``pred_sd``), ``> 1.10``;
* ``q_spread``: ``(max − min)`` of a predictive quantile / median predictive sd, ``> 0.25``;
* ``mean_shift``: ``(max − min)`` of the predictive mean / median predictive sd, ``> 0.10``;
* ``lpd_range``: ``(max − min)`` of the LPD at the realised value (backtest: of the average log
  score per origin), ``> 0.05`` nats per variable, ``> 0.10`` joint (``lpd_range_joint``);
* ``var_ratio`` (prior mode; added here): ``max/min`` of the prior median of ``Var(u_i,T)``,
  ``> 1.21 = 1.10²`` (the ``sd_ratio`` threshold on the variance scale);
* ``signal_to_noise``: ``sd over orderings / σ_MC`` of the metric's statistic, ``> 3``.

A metric is flagged when it exceeds its threshold **and** its statistic's signal-to-noise ratio
exceeds 3; ``result.material`` is true when any metric is flagged. Backtest mode also reports
the Spearman rank correlations of the orderings' rankings across variables and horizons (ARRS
Table 4) and the Model Confidence Set over orderings at 10% (Hansen, Lunde & Nason 2011; the
Phase 0 :func:`~rai_bvar.evaluation.model_confidence_set`).

**Modes.** ``"prior"`` simulates the prior of ``(A, λ_{·,T}, λ_{·,T+1})`` per ordering (no
posterior fit; :func:`prior_covariance_draws`) and reports, per variable and by position, the
prior median and 90% band of ``Var(u_i,T)``, ``Corr(u_i,T, u_j,T)`` and the robust
prior-predictive sd of the one-step-ahead reduced-form shock ``u_{T+1} = A⁻¹Λ_{T+1}^{½}ε``
(``(q84 − q16)/2``, finite even when the variance is not, e.g. ``IG(3/2, ·)`` marginals at
``ν₀ = n + 2``). It exposes CKY's ``E u_i² = 2^{i−1}`` growth under independent ``N(0, 1)``
``a_ij`` and confirms that the IW-equivalent prior removes it. ``"forecast"`` (default) fits
each ordering once and forecasts from the origin (``n_perm + noise_reps`` fits). ``"backtest"``
runs the Phase 0 :func:`~rai_bvar.evaluation.backtest` per ordering over ``origins`` (the only
mode that can say an ordering *forecasts* better).

``pool=True`` (forecast mode) returns the equal-weight mixture over orderings as a
``ForecastResult`` (``result.pooled``): draws of every ordering concatenated along ``chain``, its
LPD the Rao–Blackwellised mixture ``logsumexp_{(o,s)} log N − log(total)``. With
``design="all"`` that predictive is exactly order-invariant; with sampled orderings it is
invariant in expectation (a practical mitigation until Phase 5; N08 §B4).

``batch=True`` (forecast mode, MCMC models) runs every fit of the check in one program with a
leading fit axis (DESIGN §5.2; ``make_runner(static, fit_axis=True)``). Batched fits equal
separate fits to rounding (``1e-10``), not bitwise (different ``vmap`` batch sizes).
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import math
import warnings
import zlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar._version import __version__
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import regressor_names
from rai_bvar.forecast.result import ForecastResult
from rai_bvar.models.base import Model, spec_hash

__all__ = [
    "MAX_ALL_ORDERINGS",
    "THRESHOLDS",
    "OrderingSensitivityResult",
    "PriorDraws",
    "ordering_sensitivity",
    "permutation_design",
    "permute_model",
    "prior_covariance_draws",
    "prior_draws_from_arrays",
]

THRESHOLDS: dict[str, float] = {
    "sd_ratio": 1.10,
    "q_spread": 0.25,
    "mean_shift": 0.10,
    "lpd_range": 0.05,
    "lpd_range_joint": 0.10,
    "var_ratio": 1.21,
    "signal_to_noise": 3.0,
}
"""Default flag thresholds (heuristic, calibrated on the ARRS magnitudes; N08 §B4)."""

MAX_ALL_ORDERINGS = 5040
"""``design="all"`` enumerates at most ``7!`` orderings of the free variables."""

MODES = ("prior", "forecast", "backtest")
DESIGNS = ("balanced", "random", "all")
PRIOR_DRAWS = 10_000  # default prior draws per ordering (mode="prior")
FORECAST_DRAWS = 1000  # default forecast draws per ordering (the ForecastResult default)
_PRIOR_CHUNK = 2048  # prior draws per compiled call (fixed shape: one compile per StaticConfig)
_T_BUCKET = 32  # row bucket when the prior mode needs one more row than the window's T_pad
_PRIOR_STATS = ("var_mean", "var_q05", "var_q50", "var_q95", "pred_sd")
_CORR_STATS = ("q05", "q50", "q95")
_BACKTEST_STATS = ("lpd", "crps", "rmse")
_SCORES = ("lpd", "crps", "sq_err")
_SCORE_VARS = {"lpd": "lpd_marginal", "crps": "crps", "sq_err": "sq_err"}  # backtest names
_JSTATS = ("lpd",)

# (metric, kind, statistics): kind "ratio" = max/min, "spread" = (max − min)/median sd,
# "range" = max − min, over orderings; "quantiles" stands for every q-statistic
_METRICS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "forecast": (
        ("sd_ratio", "ratio", ("sd",)),
        ("q_spread", "spread", "quantiles"),
        ("mean_shift", "spread", ("mean",)),
        ("lpd_range", "range", ("lpd",)),
    ),
    "backtest": (("lpd_range", "range", ("lpd",)),),
    "prior": (("sd_ratio", "ratio", ("pred_sd",)), ("var_ratio", "ratio", ("var_q50",))),
}
_JOINT_METRICS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "forecast": (("lpd_range_joint", "range", ("lpd",)),),
    "backtest": (("lpd_range_joint", "range", ("lpd",)),),
    "prior": (),
}


# =========================================================================== permutations


def permute_model(
    model: Model, order: Sequence[str] | Sequence[int], *, variables: Sequence[str] | None = None
) -> Model:
    """``model`` with its positional per-variable settings moved to the new ordering.

    ``order`` lists the variables in their new order, by name (then ``variables``, the current
    ordering, is needed whenever the model has positional settings) or by current position
    (``order[i]`` is the current index of the variable that moves to position ``i``). Positional
    settings are the ``Minnesota.psi`` tuple; settings keyed by name (``Minnesota(mean={...})``)
    and data-dependent ones (``psi="ar"``, own-lag means from tcodes, EB values) follow their
    variables by construction (R9), and scalar settings are order-free. Models without
    positional settings (including non-``BVAR`` models) are returned unchanged.
    """
    from rai_bvar.api import BVAR

    order = list(order)
    by_index = all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in order)
    n = len(order)
    if by_index:
        perm = [int(v) for v in order]
        if sorted(perm) != list(range(n)):
            raise ValueError(f"order {perm} is not a permutation of 0..{n - 1}")
    else:
        names = [str(v) for v in order]
        if len(set(names)) != n:
            raise ValueError(f"order {names} repeats a variable")
        perm = None
        if variables is not None:
            current = [str(v) for v in variables]
            if sorted(current) != sorted(names):
                raise ValueError(f"order {names} is not a permutation of the variables {current}")
            perm = [current.index(v) for v in names]
    if not isinstance(model, BVAR):
        return model
    prior = model.mean.prior
    if isinstance(prior.psi, str):  # "ar": recomputed per variable from the permuted window
        return model
    if perm is None:
        raise ValueError(
            "permute_model needs variables= (the current ordering) to move the positional "
            "Minnesota psi tuple when the new order is given by name"
        )
    if len(prior.psi) != n:
        raise ValueError(f"Minnesota psi has {len(prior.psi)} values for {n} variables")
    psi = tuple(prior.psi[i] for i in perm)
    return dataclasses.replace(
        model, mean=dataclasses.replace(model.mean, prior=dataclasses.replace(prior, psi=psi))
    )


def permutation_design(
    variables: Sequence[str],
    n_perm: int = 10,
    pinned: Sequence[str] = (),
    design: Literal["balanced", "random", "all"] = "balanced",
    seed: int = 0,
) -> list[tuple[str, ...]]:
    """The orderings of an ordering-sensitivity check (N08 §B4), the user ordering first.

    The user ordering is ``pinned + F`` with the free variables ``F`` in their order in
    ``variables``. All ``m!`` orderings of ``F`` when ``m! ≤ n_perm`` or ``design="all"``;
    otherwise the user ordering, its reverse on ``F``, the rows of a random Latin square of order
    ``m`` (``"balanced"``), then uniform permutations without replacement. Deterministic in
    ``seed``. For ``m ≥ 4`` the square is re-drawn until no row repeats the first two, so with
    ``n_perm ≥ m + 2`` every free variable occupies every free position exactly once among the
    square's rows. For ``m = 3`` no such square exists: the rows of a 3×3 Latin square are the
    three even or the three odd permutations, and the user ordering is even, its reverse (a
    transposition) odd. The square of the user ordering's cyclic shifts is used instead, so with
    ``n_perm ≥ 4`` the user ordering and its two shifts are balanced (``n_perm ≥ 6 = 3!`` takes
    every ordering).
    """
    variables = tuple(str(v) for v in variables)
    if len(set(variables)) != len(variables):
        raise ValueError(f"variables {list(variables)} repeat a name")
    pinned = _pinned(pinned, variables)
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    n_perm = _positive_int(n_perm, "n_perm")
    free = tuple(v for v in variables if v not in pinned)
    m = len(free)
    if m < 2:
        raise ValueError(
            f"ordering sensitivity needs at least two free (unpinned) variables, got {list(free)}"
        )
    total = math.factorial(m)
    if design == "all" or total <= n_perm:
        if total > MAX_ALL_ORDERINGS:
            raise ValueError(
                f"design='all' with {m} free variables means {total} orderings (at most "
                f"{MAX_ALL_ORDERINGS}); pin leading variables or use design='balanced'"
            )
        rest = [p for p in itertools.permutations(free) if p != free]
        return [(*pinned, *p) for p in [free, *rest]]
    if n_perm < 2:
        raise ValueError("n_perm must be at least 2 (the user ordering and one other)")
    rng = np.random.default_rng([_seed_int(seed), _tag("design")])
    chosen: list[tuple[str, ...]] = [free]
    seen = {free}

    def add(p: tuple[str, ...]) -> None:
        if p not in seen and len(chosen) < n_perm:
            chosen.append(p)
            seen.add(p)

    add(free[::-1])
    if design == "balanced":
        for row in _latin_square(m, rng, avoid=frozenset(seen), free=free):
            add(tuple(free[j] for j in row))
    while len(chosen) < n_perm:
        add(tuple(free[j] for j in rng.permutation(m)))
    return [(*pinned, *p) for p in chosen]


def _latin_square(
    m: int, rng: np.random.Generator, *, avoid: frozenset, free: tuple[str, ...]
) -> np.ndarray:
    """A random Latin square: the cyclic square ``(r + c) mod m`` with rows, columns and symbols
    permuted; re-drawn (at most 1000 times) until no row is in ``avoid``. ``m = 3``: the user
    ordering's cyclic shifts in random row order (every 3×3 square holds the identity or the
    reverse, so no redraw can avoid both; :func:`permutation_design`)."""
    base = (np.arange(m)[:, None] + np.arange(m)[None, :]) % m
    if m == 3:
        return rng.permutation(base)  # rows: the identity and its two cyclic shifts
    for _ in range(1000):
        square = rng.permutation(m)[base[rng.permutation(m)][:, rng.permutation(m)]]
        if not any(tuple(free[j] for j in row) in avoid for row in square):
            break
    return square


def _pinned(pinned: Sequence[str] | str, variables: tuple[str, ...]) -> tuple[str, ...]:
    pinned = (pinned,) if isinstance(pinned, str) else tuple(str(v) for v in pinned)
    unknown = [v for v in pinned if v not in variables]
    if unknown:
        raise ValueError(f"pinned variables {unknown} are not in the data {list(variables)}")
    if len(set(pinned)) != len(pinned):
        raise ValueError(f"pinned {list(pinned)} repeats a variable")
    return pinned


def _index(order: Sequence[str], variables: Sequence[str]) -> np.ndarray:
    """Positions in ``order`` of ``variables`` (label lookup), with the round trip asserted."""
    order = list(order)
    idx = np.array([order.index(v) for v in variables], dtype=np.int64)
    if [order[i] for i in idx] != list(variables) or sorted(idx) != list(range(len(order))):
        raise RuntimeError(f"un-permutation failed: {order} → {variables}")  # pragma: no cover
    return idx


# =========================================================================== seeds


def _tag(name: str) -> int:
    return zlib.crc32(name.encode())


def _seed_int(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"seed must be a non-negative int, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return int(seed)


def _derive(seed: int, name: str, index: int) -> int:
    """An int seed in ``[0, 2³¹)`` from ``(seed, stream name, index)`` (SeedSequence hashing)."""
    state = np.random.SeedSequence([seed, _tag(name), int(index)]).generate_state(1, np.uint32)
    return int(state[0]) & 0x7FFFFFFF


def _noise_fit_seeds(seed: int, reps: int) -> list[int]:
    """Fit seeds of the ``reps`` noise replicates: derived from ``(seed, "noise", r)``, distinct
    from ``seed`` (the orderings' fit seed) and from each other (a hash collision is skipped)."""
    out: list[int] = []
    used, r = {seed}, 1
    while len(out) < reps:
        s = _derive(seed, "noise", r)
        r += 1
        if s not in used:
            out.append(s)
            used.add(s)
    return out


# =========================================================================== prior mode


class PriorDraws(NamedTuple):
    """Prior draws of one ordering: ``sigma (S, n, n)``, the implied ``Σ_T = A⁻¹Λ_TA⁻¹'``;
    ``shock (S, n)``, one-step-ahead reduced-form shocks ``u_{T+1} = A⁻¹Λ_{T+1}^{½}ε``; in the
    ordering ``variables``."""

    sigma: np.ndarray
    shock: np.ndarray
    variables: tuple[str, ...]


def prior_covariance_draws(
    model: Model, data: VARData, draws: int = PRIOR_DRAWS, seed: int = 0
) -> PriorDraws:
    """Prior draws of ``Σ_T`` and of the one-step-ahead shock under ``model`` on ``data``.

    MCMC compositions: the engine's prior arrays for the window (``resolve``: ψ, EB values, SV
    level centres; R9) and the independent reference prior sampler (``validation.simulate.
    prior_sample``, DESIGN §9.1) of ``(a, λ / h, o, τ)``; ``Λ_t`` is the effective variance
    ``e^{h_t}o_t²τ_t`` at the last sample row ``T`` and at ``T + 1`` (one step of the volatility
    process). Analytic compositions: ``Σ ~ IW(diag ψ, n + 2)`` (Phase 1; time-invariant).
    """
    from rai_bvar.api import BVAR

    if not isinstance(model, BVAR):
        raise TypeError(f"prior mode needs a BVAR, got {type(model).__name__}")
    draws = _positive_int(draws, "draws")
    seed = _seed_int(seed)
    variables = tuple(data.variables)
    n = len(variables)
    if model.engine == "analytic":
        from rai_bvar.engines.analytic import AnalyticFit, ConjugateProblem
        from rai_bvar.linalg import sample_inverse_wishart_chol

        prior = model.mean.prior
        if "psi" in prior.free:
            psi = AnalyticFit(model, data, None).psi
        else:
            problem = ConjugateProblem(data, prior, model.lags, model.deterministic, model.outliers)
            psi = problem.default["psi"]
        rng = np.random.default_rng([seed, _tag("analytic-prior")])
        chol = sample_inverse_wishart_chol(np.diag(np.sqrt(psi)), n + 2.0, draws, rng)
        shock = np.einsum("sij,sj->si", chol, rng.standard_normal((draws, n)))
        return PriorDraws(chol @ np.swapaxes(chol, -1, -2), shock, variables)

    from rai_bvar.engines.mcmc import _jax
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.settings import Runtime

    with _jax.x64():
        r = resolve(model, data, Runtime())
        sigma, shock = prior_draws_from_arrays(
            r.static, r.prior, int(np.asarray(r.data.T_eff)), draws=draws, seed=seed
        )
    return PriorDraws(sigma, shock, variables)


def prior_draws_from_arrays(
    static: Any, prior: Any, T_eff: int, *, draws: int = PRIOR_DRAWS, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``(Σ_T (S, n, n), u_{T+1} (S, n))`` from ``PriorArrays`` of a ``StaticConfig`` (the
    engine-level entry of :func:`prior_covariance_draws`; tests pass hand-built arrays).

    ``T_pad`` is re-bucketed to hold ``T_eff + 1`` real rows, so ``h`` (and the transitory
    scales) are drawn for ``t = 1..T+1`` from their prior (stationary, fixed-``B₀`` or RW start).
    Draws run in fixed chunks of 2048 (one compile per configuration)."""
    import jax
    from jax import random

    from rai_bvar.engines.mcmc import _jax
    from rai_bvar.engines.mcmc.config import Shapes, bucket
    from rai_bvar.engines.mcmc.keys import PRNG_IMPL

    draws = _positive_int(draws, "draws")
    T_eff = _positive_int(T_eff, "T_eff")
    s = static.shapes
    if T_eff >= s.T_pad:  # room for row T + 1 (only shapes change; prior arrays are per equation)
        static = dataclasses.replace(
            static, shapes=Shapes.of(s.n, s.lags, s.n_det, bucket(T_eff + 1, _T_BUCKET))
        )
    fn = _prior_fn(static)
    with _jax.x64():
        prior = jax.tree.map(np.asarray, prior)
        root = random.fold_in(random.key(_seed_int(seed), impl=PRNG_IMPL), _tag("prior-mode"))
        sig, sho = [], []
        for start in range(0, draws, _PRIOR_CHUNK):
            idx = np.arange(start, start + _PRIOR_CHUNK, dtype=np.uint32)
            keys = jax.vmap(lambda i: random.fold_in(root, i))(idx)
            a, b = fn(keys, prior, np.int32(T_eff))
            take = min(_PRIOR_CHUNK, draws - start)
            sig.append(np.asarray(a)[:take])
            sho.append(np.asarray(b)[:take])
    return np.concatenate(sig), np.concatenate(sho)


@functools.lru_cache(maxsize=32)
def _prior_fn(static: Any) -> Any:
    """Compiled ``(keys[B], prior, T_eff) → (Σ_T[B], u_{T+1}[B])`` for one ``StaticConfig``."""
    import jax
    import jax.numpy as jnp
    from jax import random
    from jax.scipy.linalg import solve_triangular

    from rai_bvar.validation.reference import effective_log_var
    from rai_bvar.validation.simulate import prior_sample, unit_lower

    n = static.shapes.n

    def one(key: Any, prior: Any, T_eff: Any) -> tuple[Any, Any]:
        st = prior_sample(random.fold_in(key, 0), prior, static, T_eff + 1)
        A = unit_lower(st.eq.struct.a)
        lv = effective_log_var(st.eq.vol, static)  # (n, T_pad)
        lv_T = jnp.take(lv, T_eff - 1, axis=1)
        lv_next = jnp.take(lv, T_eff, axis=1)
        L_T = solve_triangular(A, jnp.diag(jnp.exp(0.5 * lv_T)), lower=True, unit_diagonal=True)
        L_next = solve_triangular(
            A, jnp.diag(jnp.exp(0.5 * lv_next)), lower=True, unit_diagonal=True
        )
        eps = random.normal(random.fold_in(key, 1), (n,), jnp.float64)
        return L_T @ L_T.T, L_next @ eps

    return jax.jit(jax.vmap(one, in_axes=(0, None, None)))


def _prior_stats(
    pd_: PriorDraws, variables: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-variable prior statistics ``(V, |_PRIOR_STATS|)``, correlation quantiles
    ``(V, V, 3)`` and their MCSE ``(V, |_PRIOR_STATS|)``, in the order ``variables``."""
    from rai_bvar.diagnostics import mcse_mean_many, mcse_quantile_many

    idx = _index(pd_.variables, variables)
    sigma = pd_.sigma[:, idx][:, :, idx]
    shock = pd_.shock[:, idx]
    var = np.diagonal(sigma, axis1=1, axis2=2)  # (S, V)
    q = np.quantile(var, [0.05, 0.5, 0.95], axis=0)
    s16, s84 = np.quantile(shock, [0.16, 0.84], axis=0)
    values = np.stack([var.mean(axis=0), q[0], q[1], q[2], 0.5 * (s84 - s16)], axis=-1)
    sd = np.sqrt(var)
    corr = sigma / (sd[:, :, None] * sd[:, None, :])
    cq = np.moveaxis(np.quantile(corr, [0.05, 0.5, 0.95], axis=0), 0, -1)
    one = var[None]  # (1 chain, S, V)
    m16, m84 = (mcse_quantile_many(shock[None], p) for p in (0.16, 0.84))
    mcse = np.stack(
        [
            mcse_mean_many(one),
            mcse_quantile_many(one, 0.05),
            mcse_quantile_many(one, 0.5),
            mcse_quantile_many(one, 0.95),
            0.5 * np.sqrt(m16**2 + m84**2),
        ],
        axis=-1,
    )
    return values, cq, mcse


# =========================================================================== forecast statistics


def _stat_names(quantiles: Sequence[float]) -> tuple[str, ...]:
    return ("mean", "sd", *(_qname(q) for q in quantiles), "len68", "len90", "lpd", "crps")


def _qname(q: float) -> str:
    return f"q{q:g}"


def _forecast_stats(
    fc: ForecastResult,
    order: Sequence[str],
    variables: Sequence[str],
    horizons: Sequence[int],
    quantiles: Sequence[float],
    realized: np.ndarray | None,
    *,
    with_mcse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(values (V, H, S), joint (H, 1), mcse (V, H, S), mcse_joint (H, 1))`` of one forecast,
    un-permuted to ``variables``; ``realized (H_max, V)`` is in the order ``variables``."""
    import scoringrules as sr

    if list(fc.variables) != list(order):
        raise RuntimeError(f"forecast labelled {fc.variables}, fitted on {list(order)}")
    idx = _index(fc.variables, variables)
    hsel = np.asarray(horizons, dtype=np.int64) - 1
    full = fc.paths.values  # (C, D, H, n)
    x4 = full[:, :, hsel][..., idx]  # (C, D, Hs, V)
    x = x4.reshape(-1, *x4.shape[2:])  # (S, Hs, V)
    probs = sorted({*quantiles, 0.05, 0.16, 0.84, 0.95})
    qv = dict(zip(probs, np.quantile(x, probs, axis=0), strict=True))
    V, Hs = len(variables), len(horizons)
    lpd = np.full((Hs, V), np.nan)
    lpd_mcse = np.full((Hs, V), np.nan)
    crps = np.full((Hs, V), np.nan)
    joint = np.full((Hs, 1), np.nan)
    joint_mcse = np.full((Hs, 1), np.nan)
    if realized is not None:
        obs = realized[hsel]  # (Hs, V)
        for a, h in enumerate(horizons):
            for v, name in enumerate(variables):
                if np.isfinite(obs[a, v]):
                    lpd[a, v], lpd_mcse[a, v] = fc.log_pred_density(
                        obs[a, v : v + 1], int(h), [name], return_mcse=True
                    )
            if np.isfinite(obs[a]).all():
                y_fc = obs[a][_index(variables, fc.variables)]  # into the forecast's ordering
                joint[a, 0], joint_mcse[a, 0] = fc.log_pred_density(y_fc, int(h), return_mcse=True)
        ok = np.isfinite(obs)
        if ok.any():
            ens = np.moveaxis(x, 0, -1)  # (Hs, V, S)
            score = sr.crps_ensemble(np.where(ok, obs, 0.0), ens, estimator="pwm")
            crps = np.where(ok, score, np.nan)
    cols = [x.mean(axis=0), x.std(axis=0, ddof=1)]
    cols += [qv[q] for q in quantiles]
    cols += [qv[0.84] - qv[0.16], qv[0.95] - qv[0.05], lpd, crps]
    values = np.moveaxis(np.stack(cols, axis=-1), 0, 1)  # (V, Hs, S)
    if not with_mcse:
        return values, joint, None, None
    from rai_bvar.diagnostics import mcse_mean_many, mcse_quantile_many

    mq = {q: mcse_quantile_many(x4, q) for q in probs}  # (Hs, V)
    sd = cols[1]
    dev2 = (x4 - x4.mean(axis=(0, 1), keepdims=True)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_mcse = np.where(sd > 0, mcse_mean_many(dev2) / (2.0 * sd), np.nan)
    mc = [mcse_mean_many(x4), sd_mcse, *(mq[q] for q in quantiles)]
    mc += [np.hypot(mq[0.84], mq[0.16]), np.hypot(mq[0.95], mq[0.05]), lpd_mcse]
    mc += [np.full((Hs, V), np.nan)]  # CRPS: noise floor only
    mcse = np.moveaxis(np.stack(mc, axis=-1), 0, 1)
    return values, joint, mcse, joint_mcse


def _posterior_summary(
    fit: Any, order: Sequence[str], variables: Sequence[str], lags: int, det: str
) -> dict[str, Any] | None:
    """Posterior means of ``B`` (``(R, V)``, original regressor labels) and ``Σ_T`` (``(V, V)``),
    the scalar hyperparameters and ψ, un-permuted. ``None`` for fits other than ``AnalyticFit``
    and ``MCMCFit``."""
    from rai_bvar.engines.analytic import AnalyticFit

    if isinstance(fit, AnalyticFit):
        post = fit.posterior()
        B, sigma = np.asarray(post.mean), np.asarray(post.sigma_mean)
    elif getattr(fit, "engine", None) == "mcmc" and hasattr(fit, "param_draws"):
        lead = np.shape(next(iter(_leaves(fit.param_draws))))[:2]
        coefs, chol = fit.draw_parameters(int(np.prod(lead)))
        B = coefs.mean(axis=0)
        sigma = np.einsum("sij,skj->ik", chol, chol) / chol.shape[0]
    else:
        return None
    idx = _index(order, variables)
    regs_fit = regressor_names(list(order), lags, det)
    regs = regressor_names(list(variables), lags, det)
    ridx = _index(regs_fit, regs)
    hyper = getattr(fit, "hyper", {}) or {}
    scalars = {
        k: float(v)
        for k, v in hyper.items()
        if isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool)
    }
    psi = hyper.get("psi")
    psi = np.full(len(variables), np.nan) if psi is None else np.asarray(psi, np.float64)[idx]
    return {"B": B[ridx][:, idx], "sigma": sigma[idx][:, idx], "hyper": scalars, "psi": psi}


def _leaves(tree: Any) -> list[Any]:
    import jax

    return jax.tree.leaves(tree)


# =========================================================================== fitting


def _with_seed(model: Model, seed: int) -> Model:
    """``model`` whose own sampler seed is ``seed`` (MCMC ``BVAR``: ``MCMCSettings.seed``;
    analytic ``BVAR`` with ``hierarchy="mh"``: ``MHSettings.seed``); other models unchanged.
    Backtests fit without a seed argument, so this is how backtest-mode replicates differ in
    their MCMC noise (on top of the per-origin fit seeds the backtest derives from its seed)."""
    from rai_bvar.api import BVAR

    if not isinstance(model, BVAR):
        return model
    if model.engine == "mcmc":
        return dataclasses.replace(model, mcmc=dataclasses.replace(model.mcmc, seed=seed))
    prior = model.mean.prior
    if prior.hierarchy == "mh":
        prior = dataclasses.replace(prior, mh=dataclasses.replace(prior.mh, seed=seed))
        return dataclasses.replace(model, mean=dataclasses.replace(model.mean, prior=prior))
    return model


def _fit_kwargs(model: Model, seed: int) -> dict[str, Any]:
    """``seed=`` for models that take a fit seed (MCMC ``BVAR``; analytic ``BVAR`` with MH)."""
    from rai_bvar.api import BVAR

    if getattr(model, "uses_fit_seed", False):
        return {"seed": seed}
    if isinstance(model, BVAR) and model.mean.prior.hierarchy == "mh":
        return {"seed": seed}
    return {}


class _Warnings:
    """Collects the ``UserWarning`` messages of every fit (re-emitted once, aggregated)."""

    def __init__(self) -> None:
        self.by_message: dict[str, list[str]] = {}
        self.passthrough: list[Any] = []

    def record(self, label: str, caught: list[Any]) -> None:
        for w in caught:
            if issubclass(w.category, UserWarning):
                runs = self.by_message.setdefault(str(w.message), [])
                if label not in runs:
                    runs.append(label)
            else:
                self.passthrough.append(w)

    def emit(self, total: int) -> None:
        for w in self.passthrough:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        for message, runs in self.by_message.items():
            warnings.warn(
                f"ordering_sensitivity: {message} [in {len(runs)} of {total} runs: "
                f"{', '.join(runs[:6])}{', …' if len(runs) > 6 else ''}]",
                UserWarning,
                stacklevel=3,
            )


def _run_fits(
    models: list[Model],
    datas: list[VARData],
    seeds: list[int],
    labels: list[str],
    log: _Warnings,
    batch: bool,
) -> list[Any]:
    """Fit every ``(model, data)`` pair: looped, or as one fit-axis program (``batch``)."""
    if batch:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits = _batched_fits(models, datas, seeds)
        log.record("batched fits", caught)
        return fits
    fits = []
    for m, d, s, label in zip(models, datas, seeds, labels, strict=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits.append(m.fit(d, **_fit_kwargs(m, s)))
        log.record(label, caught)
    return fits


def _batched_fits(models: list[Any], datas: list[VARData], seeds: list[int]) -> list[Any]:
    """Every fit in one program with a leading fit axis (DESIGN §5.2): the windows' engine
    inputs (``resolve``) are stacked, ``run_chains`` runs them together, and each fit's slice of
    the run becomes an ``MCMCFit``."""
    import jax

    from rai_bvar.engines.mcmc import _jax
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.results import FitMeta, MCMCFit
    from rai_bvar.engines.mcmc.runner import make_runner, run_chains
    from rai_bvar.settings import Runtime

    runtime = Runtime()
    settings = models[0].mcmc
    with _jax.x64():
        resolved = [resolve(m, d, runtime) for m, d in zip(models, datas, strict=True)]
        static = resolved[0].static
        if any(r.static != static for r in resolved):
            raise ValueError(
                "batch=True needs one StaticConfig for every fit (same shapes and components)"
            )
        fns = make_runner(static, fit_axis=True)
        stack = lambda *xs: np.stack([np.asarray(x) for x in xs])  # noqa: E731
        data = jax.tree.map(stack, *[r.data for r in resolved])
        prior = jax.tree.map(stack, *[r.prior for r in resolved])
        run = run_chains(
            fns, static, data, prior, settings, runtime,
            seed=list(seeds), salt=[r.salt for r in resolved],
        )  # fmt: skip
        fits = []
        for f, (m, d, r) in enumerate(zip(models, datas, resolved, strict=True)):
            meta = FitMeta(
                model=m, vardata=d, periods=r.periods, variables=tuple(d.variables),
                hyper=r.hyper, notes=r.notes, eb_converged=r.eb_converged,
                warning_inputs=dict(r.warning_inputs),
            )  # fmt: skip
            fits.append(
                MCMCFit(
                    static=static,
                    run=_slice_run(run, f, int(np.asarray(r.data.T_eff))),
                    program=fns.program,
                    data=r.data,
                    prior=r.prior,
                    settings=settings,
                    meta=meta,
                )
            )
    return fits


def _slice_run(run: Any, f: int, T_eff: int) -> Any:
    """The ``RunOutput`` of fit ``f`` of a fit-axis run (leading ``(F, C)`` → ``(C,)``)."""
    import jax

    from rai_bvar.engines.mcmc.storage import HostStore

    store = run.store
    F = store.plan.lead[0]
    plan = dataclasses.replace(
        store.plan,
        lead=tuple(store.plan.lead[1:]),
        bytes_by_stream={k: v // F for k, v in store.plan.bytes_by_stream.items()},
    )
    take = lambda x: np.asarray(x)[f]  # noqa: E731
    sliced = HostStore(
        plan=plan,
        arrays={name: [take(a) for a in arrs] for name, arrs in store.arrays.items()},
        kept=store.kept,
        acceptance={k: take(v) for k, v in store.acceptance.items()},
        latent={k: take(v) for k, v in store.latent.items()},
        min_pivot=take(store.min_pivot),
        constant=dict(store.constant),
        attrs=dict(store.attrs),
        T_eff=T_eff,
        static=store.static,
    )
    return dataclasses.replace(
        run,
        store=sliced,
        final_state=jax.tree.map(take, run.final_state),
        acceptance={k: take(v) for k, v in run.acceptance.items()},
        latent={k: take(v) for k, v in run.latent.items()},
        min_pivot=take(run.min_pivot),
        seed=int(run.seed[f]),
        salt=int(run.salt[f]),
        plan=plan,
    )


# =========================================================================== pooling


def _pooled(
    forecasts: list[ForecastResult],
    orders: list[tuple[str, ...]],
    variables: Sequence[str],
    attrs: Mapping[str, Any],
) -> ForecastResult:
    """Equal-weight mixture over orderings (N08 §B4): each ordering's draws un-permuted to
    ``variables`` (``cond_chol`` re-factored for the new order by QR, ``linalg.marginal_chol``)
    and concatenated along ``chain``."""
    from rai_bvar.linalg import marginal_chol

    parts: dict[str, list[np.ndarray]] = {"paths": [], "cond_mean": [], "cond_chol": []}
    levels: list[np.ndarray] = []
    draws = {fc.dataset.sizes["draw"] for fc in forecasts}
    if len(draws) != 1:
        raise ValueError(f"pooling needs equal draws per ordering, got {sorted(draws)}")
    for fc, order in zip(forecasts, orders, strict=True):
        if list(fc.variables) != list(order):
            raise RuntimeError(f"forecast labelled {fc.variables}, fitted on {list(order)}")
        idx = _index(order, variables)
        parts["paths"].append(fc.paths.values[..., idx])
        parts["cond_mean"].append(fc.cond_mean.values[..., idx])
        parts["cond_chol"].append(marginal_chol(fc.cond_chol.values, idx))
        if fc.paths_level is not None:
            levels.append(fc.paths_level.values[..., idx])
    first = forecasts[0]
    return ForecastResult.from_arrays(
        **{k: np.concatenate(v, axis=0) for k, v in parts.items()},
        variables=list(variables),
        origin=first.origin,
        paths_level=np.concatenate(levels, axis=0) if len(levels) == len(forecasts) else None,
        attrs=attrs,
    )


# =========================================================================== result


class OrderingSensitivityResult:
    """Outcome of :func:`ordering_sensitivity` (N08 §B4): an ``xarray.Dataset`` plus helpers.

    ``dataset`` variables (``ordering`` 0 is the user ordering; ``variable`` in data order):

    * ``value (ordering, variable, horizon, stat)``: per-ordering statistics (prior mode:
      ``(ordering, variable, stat)``); ``joint (ordering, horizon, jstat)``: joint LPD;
    * ``noise`` / ``noise_joint`` (``rep`` first; rep 0 is the user-ordering run itself) and
      ``mcse`` / ``mcse_joint`` of the user ordering: the noise floor;
    * ``position (ordering, variable)``: each variable's 0-based position;
      ``converged (ordering)``; forecast mode: posterior means ``B_post (ordering, regressor,
      variable)``, ``sigma_post (ordering, variable, variable_)`` (``Σ_T``), ``psi (ordering,
      variable)``, ``hyper_post (ordering, hyper)``; backtest mode: ``origin_scores (ordering,
      origin, variable, horizon, score)``, ``origin_joint_lpd (ordering, origin, horizon)``,
      ``rank_corr (cell, cell_)`` and ``mcs_pvalue (ordering, horizon)``; prior mode: ``corr
      (ordering, variable, variable_, cstat)``;
    * attrs: mode, design, orderings (JSON), seeds, thresholds, spec hash, versions.
    """

    def __init__(self, dataset: xr.Dataset, pooled: ForecastResult | None = None):
        self._ds = dataset
        self.pooled = pooled

    # ------------------------------------------------------------------ accessors

    @property
    def dataset(self) -> xr.Dataset:
        return self._ds

    @property
    def mode(self) -> str:
        return str(self._ds.attrs["mode"])

    @property
    def orderings(self) -> list[tuple[str, ...]]:
        return [tuple(o) for o in json.loads(self._ds.attrs["orderings"])]

    @property
    def variables(self) -> list[str]:
        return [str(v) for v in self._ds["variable"].values]

    @property
    def noise(self) -> xr.Dataset:
        """The noise-floor runs of the user ordering and the user fit's MCSE."""
        names = [k for k in ("noise", "noise_joint", "mcse", "mcse_joint") if k in self._ds]
        return self._ds[names]

    @property
    def material(self) -> bool:
        """Whether any metric exceeds its default threshold with signal-to-noise above 3."""
        return self.is_material()

    def is_material(self, thresholds: Mapping[str, float] | None = None) -> bool:
        m = self.metrics(thresholds)
        return bool(any(bool(m[k].values.any()) for k in m.data_vars if k.endswith("_flag")))

    # ------------------------------------------------------------------ noise and metrics

    def sigma_mc(self, joint: bool = False) -> np.ndarray:
        """``σ_MC = max(sd over noise replicates, MCSE)`` per statistic (NaN-aware)."""
        noise = self._ds["noise_joint" if joint else "noise"].values
        mcse = self._ds["mcse_joint" if joint else "mcse"].values
        sd = _sd(noise, axis=0)
        return np.fmax(sd, mcse)

    def signal_to_noise(self, joint: bool = False) -> np.ndarray:
        """``sd over orderings / σ_MC`` per statistic (``inf`` when σ_MC = 0 < sd)."""
        value = self._ds["joint" if joint else "value"].values
        return _snr(_sd(value, axis=0), self.sigma_mc(joint))

    def metrics(self, thresholds: Mapping[str, float] | None = None) -> xr.Dataset:
        """Metric values, their statistic's signal-to-noise ratio and flags (module docstring)."""
        t = _thresholds(thresholds)
        out: dict[str, xr.DataArray] = {}
        for joint, specs in ((False, _METRICS[self.mode]), (True, _JOINT_METRICS[self.mode])):
            if not specs:
                continue
            da = self._ds["joint" if joint else "value"]
            stats = [str(s) for s in da[da.dims[-1]].values]
            value = da.values
            snr = self.signal_to_noise(joint)
            norm = self._median_sd() if not joint else None
            dims = da.dims[1:-1]
            coords = {d: da[d].values for d in dims}
            for name, kind, wanted in specs:
                cols = [s for s in stats if s.startswith("q")] if wanted == "quantiles" else wanted
                cols = [s for s in cols if s in stats]
                if not cols:
                    continue
                k = [stats.index(s) for s in cols]
                vals = _dispersion(value[..., k], kind, norm)  # (..., |k|)
                s2n = snr[..., k]
                with np.errstate(invalid="ignore"):
                    flags = (vals > t[name]) & (s2n > t["signal_to_noise"])
                best = _nanargmax(vals)
                metric = np.take_along_axis(vals, best[..., None], axis=-1)[..., 0]
                metric_snr = np.take_along_axis(s2n, best[..., None], axis=-1)[..., 0]
                out[name] = xr.DataArray(metric, dims=dims, coords=coords)
                out[f"{name}_snr"] = xr.DataArray(metric_snr, dims=dims, coords=coords)
                out[f"{name}_flag"] = xr.DataArray(flags.any(axis=-1), dims=dims, coords=coords)
        return xr.Dataset(out)

    def _median_sd(self) -> np.ndarray | None:
        """Median over orderings of the predictive sd ``(variable, horizon)`` (spread metrics)."""
        da = self._ds["value"]
        stats = [str(s) for s in da["stat"].values]
        if "sd" not in stats:
            return None
        return np.median(da.values[..., stats.index("sd")], axis=0)

    def summary(self, thresholds: Mapping[str, float] | None = None) -> pd.DataFrame:
        """One row per variable × horizon (prior mode: per variable) plus ``joint`` rows: every
        metric with its signal-to-noise ratio and flag, and ``material`` (any flag in the row)."""
        m = self.metrics(thresholds)
        rows: list[dict[str, Any]] = []
        per_var = [k for k in m.data_vars if "variable" in m[k].dims]
        joint = [k for k in m.data_vars if "variable" not in m[k].dims]
        if per_var:
            frame = m[per_var].to_dataframe()
            for key, row in frame.iterrows():
                rec = dict(
                    zip(frame.index.names, key if isinstance(key, tuple) else (key,), strict=True)
                )
                rec.update(row.to_dict())
                rows.append(rec)
        if joint:
            frame = m[joint].to_dataframe()
            for key, row in frame.iterrows():
                rec = {"variable": "joint", "horizon": key}
                rec.update({k.replace("_joint", ""): v for k, v in row.to_dict().items()})
                rows.append(rec)
        table = pd.DataFrame(rows)
        index = [c for c in ("variable", "horizon") if c in table.columns]
        flags = [c for c in table.columns if c.endswith("_flag")]
        for c in flags:  # joint rows have no per-variable metric (NaN): not flagged
            table[c] = [bool(x) if isinstance(x, bool | np.bool_) else False for x in table[c]]
        table["material"] = table[flags].any(axis=1) if flags else False
        table = table.set_index(index)
        table.attrs.update(
            {
                "mode": self.mode,
                "thresholds": _thresholds(thresholds),
                "material": bool(table["material"].any()),
                "orderings": len(self.orderings),
            }
        )
        return table

    def by_position(self, name: str = "value") -> xr.DataArray:
        """``name`` (with dims ``(ordering, variable, …)``) indexed by position in each ordering:
        dims ``(ordering, position, …)``, with the variable at each position as a coordinate."""
        da = self._ds[name]
        if da.dims[:2] != ("ordering", "variable"):
            raise ValueError(f"{name!r} has dims {da.dims}, not (ordering, variable, …)")
        pos = self._ds["position"].values
        out = np.empty_like(da.values)
        names = np.empty(pos.shape, dtype=object)
        for k in range(pos.shape[0]):
            out[k, pos[k]] = da.values[k]
            names[k, pos[k]] = self.variables
        coords = {d: da[d].values for d in da.dims[2:] if d in da.coords}
        coords |= {"position": np.arange(pos.shape[1]), "ordering": da["ordering"].values}
        coords["variable_at"] = (("ordering", "position"), names.astype(str))
        return xr.DataArray(out, dims=("ordering", "position", *da.dims[2:]), coords=coords)

    # ------------------------------------------------------------------ plotting

    def plot(self, stat: str | None = None, *, relative: bool = True, ax: Any = None) -> Any:
        """ARRS-Fig.1-style dot plot (needs ``matplotlib``): one panel per variable, a column of
        dots per horizon (prior mode: per variable), one dot per ordering, relative to the user
        ordering (``relative=True``) with a ``±2σ_MC`` band. Returns the figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - optional dependency
            raise ImportError("plot() needs matplotlib: pip install 'rai-bvar[plot]'") from err
        da = self._ds["value"]
        stats = [str(s) for s in da["stat"].values]
        if stat is None:
            stat = next(
                (s for s in ("lpd", "pred_sd", "sd") if s in stats and np.isfinite(
                    da.sel(stat=s).values).any()),
                stats[0],
            )  # fmt: skip
        if stat not in stats:
            raise ValueError(f"stat must be one of {stats}, got {stat!r}")
        values = da.sel(stat=stat).values  # (K, V[, H])
        if values.ndim == 2:
            values = values[..., None]
        band = 2.0 * self.sigma_mc()[..., stats.index(stat)]
        band = band if band.ndim == 2 else band[..., None]
        V = values.shape[1]
        if ax is None:
            fig, axes = plt.subplots(1, V, figsize=(3.0 * V, 3.0), squeeze=False)
            axes = axes[0]
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
        xs = self._ds["horizon"].values if "horizon" in da.dims else np.array([0])
        for v, a in enumerate(axes[:V]):
            base = values[0, v] if relative else np.zeros(values.shape[2])
            for k in range(1, values.shape[0]):
                a.plot(xs, values[k, v] - base, "o", color="0.45", ms=3)
            a.plot(xs, values[0, v] - base, "D", color="C3", ms=4, label="user ordering")
            centre = 0.0 if relative else values[0, v]
            a.fill_between(xs, centre - band[v], centre + band[v], color="C0", alpha=0.15)
            a.set_title(self.variables[v])
            a.set_xlabel("horizon" if "horizon" in da.dims else "")
        axes[0].set_ylabel(f"{stat} − user" if relative else stat)
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"OrderingSensitivityResult(mode={self.mode!r}, orderings={len(self.orderings)}, "
            f"variables={self.variables}, material={self.material})"
        )


def _sd(x: np.ndarray, axis: int) -> np.ndarray:
    """``ddof = 1`` sd along ``axis``; NaN with fewer than two values (no warning)."""
    if x.shape[axis] < 2:
        return np.full(np.delete(x.shape, axis), np.nan)
    with np.errstate(invalid="ignore"):
        return np.std(x, axis=axis, ddof=1)


def _snr(sd: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sigma > 0, sd / sigma, np.where(sd > 0, np.inf, 0.0))
    return np.where(np.isnan(sd) | np.isnan(sigma), np.nan, out)


def _dispersion(value: np.ndarray, kind: str, norm: np.ndarray | None) -> np.ndarray:
    """``max/min`` (ratio), ``(max − min)/norm`` (spread) or ``max − min`` (range) over the
    leading ordering axis; NaN where any ordering is NaN."""
    hi, lo = value.max(axis=0), value.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "ratio":
            return np.where(lo > 0, hi / lo, np.nan)
        if kind == "spread":
            if norm is None:
                return np.full(hi.shape, np.nan)
            n = norm[..., None]
            return np.where(n > 0, (hi - lo) / n, np.nan)
        return hi - lo


def _nanargmax(x: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), -np.inf, x)
    return np.argmax(filled, axis=-1)


def _thresholds(thresholds: Mapping[str, float] | None) -> dict[str, float]:
    t = dict(THRESHOLDS)
    for k, v in (thresholds or {}).items():
        if k not in THRESHOLDS:
            raise KeyError(f"unknown threshold {k!r}; expected some of {sorted(THRESHOLDS)}")
        t[k] = float(v)
    return t


# =========================================================================== entry point


def ordering_sensitivity(
    model: Model,
    data: VARData,
    n_perm: int = 10,
    pinned: Sequence[str] = (),
    mode: Literal["prior", "forecast", "backtest"] = "forecast",
    design: Literal["balanced", "random", "all"] = "balanced",
    horizons: Sequence[int] = (1, 4, 8),
    origins: Any = None,
    noise_reps: int = 3,
    quantiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95),
    pool: bool = False,
    seed: int = 0,
    batch: bool = False,
    *,
    draws: int | None = None,
    window: Any = None,
    cache: Any = None,
    n_jobs: int = 1,
) -> OrderingSensitivityResult:
    """Refit ``model`` under ``n_perm`` orderings of ``data`` and measure the dispersion of its
    results against a Monte Carlo noise floor (SPEC §6.6; N08 §B4; module docstring).

    ``pinned`` variables lead every ordering. ``origins``: forecast mode, ``None`` (the last
    period) or one forecast origin (outcomes after it are scored: LPD, CRPS); backtest mode, an
    ``Origins``, a sequence of periods or ``None`` (every 4th period from the middle of the sample
    to the second-to-last period). ``noise_reps``: extra fits of the user ordering for the noise
    floor (backtest mode: at least 1). ``draws``: forecast draws per run (default 1000) or prior
    draws per ordering (default 10,000). ``window``, ``cache`` and ``n_jobs`` go to
    :func:`~rai_bvar.evaluation.backtest`. Seeds: every ordering is fit with ``seed`` (data salts
    keep MCMC chains independent across orderings, DESIGN §5.6; in backtest mode ``seed`` becomes
    the model's own sampler seed, ``MCMCSettings.seed`` or ``MHSettings.seed``, since backtests
    fit without a seed argument); forecast, backtest, prior and noise-replicate seeds (each noise
    replicate's fit seed differs from ``seed`` and from the others) are derived from ``(seed,
    stream, index)`` and recorded in attrs (``fit_seeds``, ``forecast_seeds``,
    ``backtest_seeds``, ``prior_seeds``, ``noise_seeds``).
    """
    if not isinstance(data, VARData):
        raise TypeError(f"data must be a VARData, got {type(data).__name__}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    seed = _seed_int(seed)
    noise_reps = _nonnegative_int(noise_reps, "noise_reps")
    horizons = _horizons(horizons)
    quantiles = _quantiles(quantiles)
    for name, flag in (("pool", pool), ("batch", batch)):
        if not isinstance(flag, bool | np.bool_):
            raise TypeError(f"{name} must be a bool, got {flag!r}")
    if pool and mode != "forecast":
        raise ValueError("pool=True mixes forecasts: it needs mode='forecast'")
    if batch:
        if mode != "forecast":
            raise ValueError("batch=True batches the fits of mode='forecast'")
        if getattr(model, "engine", None) != "mcmc":
            raise ValueError("batch=True batches MCMC fits; this model is not on the MCMC engine")
    if mode != "backtest" and (window is not None or cache is not None or n_jobs != 1):
        raise ValueError("window, cache and n_jobs apply to mode='backtest' only")
    if mode == "backtest" and noise_reps < 1:
        raise ValueError(
            "mode='backtest' needs noise_reps >= 1: backtest scores carry no MCSE, so the "
            "replicates of the user ordering are the only noise floor"
        )
    variables = tuple(data.variables)
    orders = permutation_design(variables, n_perm, pinned, design, seed)
    ctx = _Context(
        model=model, data=data, variables=variables, orders=orders, mode=mode, design=design,
        n_perm=int(n_perm), pinned=_pinned(pinned, variables), horizons=horizons,
        quantiles=quantiles, noise_reps=noise_reps, seed=seed,
    )  # fmt: skip
    if mode == "prior":
        return _prior_mode(ctx, draws)
    if mode == "forecast":
        return _forecast_mode(ctx, origins, draws, pool, batch)
    return _backtest_mode(ctx, origins, draws, window, cache, n_jobs)


@dataclasses.dataclass(frozen=True)
class _Context:
    model: Any
    data: VARData
    variables: tuple[str, ...]
    orders: list[tuple[str, ...]]
    mode: str
    design: str
    n_perm: int
    pinned: tuple[str, ...]
    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]
    noise_reps: int
    seed: int

    def models(self) -> list[Any]:
        return [permute_model(self.model, o, variables=self.variables) for o in self.orders]

    def attrs(self, **extra: Any) -> dict[str, Any]:
        model = self.model
        try:
            shash = spec_hash(model)
        except Exception:  # a model whose spec does not serialise
            shash = "unknown"
        out = {
            "mode": self.mode,
            "design": self.design,
            "n_perm": self.n_perm,
            "orderings": json.dumps([list(o) for o in self.orders]),
            "pinned": json.dumps(list(self.pinned)),
            "noise_reps": self.noise_reps,
            "seed": self.seed,
            "horizons": json.dumps(list(self.horizons)),
            "quantiles": json.dumps(list(self.quantiles)),
            "thresholds": json.dumps(THRESHOLDS),
            "model": str(getattr(model, "name", type(model).__name__)),
            "engine": str(getattr(model, "engine", "unknown")),
            "spec_hash": shash,
            "data_fingerprint": self.data.fingerprint(),
            "rai_bvar_version": __version__,
        }
        for k, v in extra.items():
            out[k] = v if isinstance(v, str | int | float) else json.dumps(v, default=str)
        return out

    def positions(self) -> np.ndarray:
        return np.stack([_index(o, self.variables) for o in self.orders])


def _finish(ds: xr.Dataset, pooled: ForecastResult | None = None) -> OrderingSensitivityResult:
    result = OrderingSensitivityResult(ds, pooled)
    ds.attrs["material"] = int(result.material)
    return result


# --------------------------------------------------------------------------- prior mode


def _prior_mode(ctx: _Context, draws: int | None) -> OrderingSensitivityResult:
    draws = PRIOR_DRAWS if draws is None else _positive_int(draws, "draws")
    log = _Warnings()
    runs: list[tuple[str, Any, VARData, int]] = []
    for j, (m, o) in enumerate(zip(ctx.models(), ctx.orders, strict=True)):
        runs.append((f"ordering {j}", m, ctx.data.select(o), _derive(ctx.seed, "prior", j)))
    m0, d0 = runs[0][1], runs[0][2]
    for r in range(1, ctx.noise_reps + 1):
        runs.append((f"noise {r}", m0, d0, _derive(ctx.seed, "noise-prior", r)))
    K = len(ctx.orders)
    values, corrs, mcse = [], [], None
    for j, (label, m, d, s) in enumerate(runs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pd_ = prior_covariance_draws(m, d, draws, s)
        log.record(label, caught)
        v, c, mc = _prior_stats(pd_, ctx.variables)
        values.append(v)
        corrs.append(c)
        if j == 0:
            mcse = mc
    log.emit(len(runs))
    vals = np.stack(values)  # (K + R, V, S)
    noise = np.concatenate([vals[:1], vals[K:]])
    var = list(ctx.variables)
    coords = {
        "ordering": np.arange(K),
        "variable": var,
        "variable_": var,
        "stat": list(_PRIOR_STATS),
        "cstat": list(_CORR_STATS),
        "rep": np.arange(ctx.noise_reps + 1),
        "jstat": list(_JSTATS),
        "label": ("ordering", [" > ".join(o) for o in ctx.orders]),
    }
    ds = xr.Dataset(
        {
            "value": (("ordering", "variable", "stat"), vals[:K]),
            "corr": (("ordering", "variable", "variable_", "cstat"), np.stack(corrs[:K])),
            "noise": (("rep", "variable", "stat"), noise),
            "mcse": (("variable", "stat"), mcse),
            "joint": (("ordering", "jstat"), np.full((K, 1), np.nan)),  # no joint statistic
            "noise_joint": (("rep", "jstat"), np.full((ctx.noise_reps + 1, 1), np.nan)),
            "mcse_joint": (("jstat",), np.full(1, np.nan)),
            "position": (("ordering", "variable"), ctx.positions()),
        },
        coords=coords,
        attrs=ctx.attrs(
            draws=draws,
            prior_seeds=[_derive(ctx.seed, "prior", j) for j in range(K)],
            noise_seeds=[_derive(ctx.seed, "noise-prior", r) for r in range(1, ctx.noise_reps + 1)],
            fit_warnings=log.by_message,
        ),
    )
    return _finish(ds)


# --------------------------------------------------------------------------- forecast mode


def _forecast_mode(
    ctx: _Context, origins: Any, draws: int | None, pool: bool, batch: bool
) -> OrderingSensitivityResult:
    draws = FORECAST_DRAWS if draws is None else _positive_int(draws, "draws")
    data = ctx.data
    origin = _forecast_origin(origins, data)
    window = data if origin == data.last else data.upto(origin)
    H = max(ctx.horizons)
    realized = None
    if origin < data.last:
        realized = data.realized(origin, H)  # (H, n) in data order
        if not np.isfinite(realized).any():
            realized = None
    K, R = len(ctx.orders), ctx.noise_reps
    models = ctx.models()
    run_models = models + [models[0]] * R
    run_orders = list(ctx.orders) + [ctx.orders[0]] * R
    run_datas = [window.select(o) for o in run_orders]
    fit_seeds = [ctx.seed] * K + _noise_fit_seeds(ctx.seed, R)
    fc_seeds = [_derive(ctx.seed, "forecast", j) for j in range(K)]
    fc_seeds += [_derive(ctx.seed, "noise-forecast", r) for r in range(1, R + 1)]
    labels = [f"ordering {j}" for j in range(K)] + [f"noise {r}" for r in range(1, R + 1)]
    log = _Warnings()
    fits: list[Any] | None = (
        _run_fits(run_models, run_datas, fit_seeds, labels, log, True) if batch else None
    )

    lags = getattr(ctx.model, "lags", None)
    det = getattr(ctx.model, "deterministic", None)
    values, joints, posts, conv, forecasts = [], [], [], [], []
    mcse = mcse_joint = None
    for j in range(K + R):
        if fits is not None:
            fit = fits[j]
        else:
            fit = _run_fits(
                [run_models[j]], [run_datas[j]], [fit_seeds[j]], [labels[j]], log, False
            )[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fc = fit.forecast(H, draws, seed=fc_seeds[j])
        log.record(labels[j], caught)
        v, jt, mc, mcj = _forecast_stats(
            fc, run_orders[j], ctx.variables, ctx.horizons, ctx.quantiles, realized,
            with_mcse=j == 0,
        )  # fmt: skip
        values.append(v)
        joints.append(jt)
        conv.append(int(bool(getattr(fit, "converged", True))))
        if j == 0:
            mcse, mcse_joint = mc, mcj
        if j < K:
            if lags is not None and det is not None:
                posts.append(_posterior_summary(fit, run_orders[j], ctx.variables, lags, det))
            if pool:
                forecasts.append(fc)
        if fits is not None:
            fits[j] = None  # release the draws
    log.emit(K + R)
    vals, jts = np.stack(values), np.stack(joints)
    var = list(ctx.variables)
    stats = list(_stat_names(ctx.quantiles))
    coords: dict[str, Any] = {
        "ordering": np.arange(K),
        "variable": var,
        "variable_": var,
        "horizon": list(ctx.horizons),
        "stat": stats,
        "jstat": list(_JSTATS),
        "rep": np.arange(R + 1),
        "label": ("ordering", [" > ".join(o) for o in ctx.orders]),
    }
    data_vars: dict[str, Any] = {
        "value": (("ordering", "variable", "horizon", "stat"), vals[:K]),
        "joint": (("ordering", "horizon", "jstat"), jts[:K]),
        "noise": (("rep", "variable", "horizon", "stat"), np.concatenate([vals[:1], vals[K:]])),
        "noise_joint": (("rep", "horizon", "jstat"), np.concatenate([jts[:1], jts[K:]])),
        "mcse": (("variable", "horizon", "stat"), mcse),
        "mcse_joint": (("horizon", "jstat"), mcse_joint),
        "position": (("ordering", "variable"), ctx.positions()),
        "converged": (("ordering",), np.array(conv[:K])),
        "noise_converged": (("rep",), np.array(conv[:1] + conv[K:])),
    }
    if posts and all(p is not None for p in posts):
        hyper_names = list(dict.fromkeys(k for p in posts for k in p["hyper"]))
        coords["regressor"] = regressor_names(var, lags, det)
        coords["hyper"] = hyper_names
        data_vars["B_post"] = (
            ("ordering", "regressor", "variable"), np.stack([p["B"] for p in posts])
        )  # fmt: skip
        data_vars["sigma_post"] = (
            ("ordering", "variable", "variable_"), np.stack([p["sigma"] for p in posts])
        )  # fmt: skip
        data_vars["psi"] = (("ordering", "variable"), np.stack([p["psi"] for p in posts]))
        data_vars["hyper_post"] = (
            ("ordering", "hyper"),
            np.array([[p["hyper"].get(h, np.nan) for h in hyper_names] for p in posts]),
        )
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs=ctx.attrs(
            origin=str(origin),
            realized=int(realized is not None),
            draws=draws,
            batch=int(batch),
            fit_seeds=fit_seeds,
            forecast_seeds=fc_seeds,
            fit_warnings=log.by_message,
        ),
    )
    pooled = None
    if pool:
        first = forecasts[0].dataset.attrs
        keep = ("model", "engine", "spec", "spec_hash", "tcodes", "rai_bvar_version")
        attrs = {k: first[k] for k in keep if k in first}
        attrs |= {
            "ordering": json.dumps(var),
            "pooled_orderings": json.dumps([list(o) for o in ctx.orders]),
            "design": ctx.design,
            "converged": int(all(conv[:K])),
            "seed": json.dumps(fc_seeds[:K]),
        }
        pooled = _pooled(forecasts, list(ctx.orders), var, attrs)
    return _finish(ds, pooled)


def _forecast_origin(origins: Any, data: VARData) -> pd.Period:
    if origins is None:
        return data.last
    if isinstance(origins, Sequence) and not isinstance(origins, str):
        if len(origins) != 1:
            raise ValueError("mode='forecast' takes one origin (or None for the last period)")
        origins = origins[0]
    origin = pd.Period(origins, freq=data.index.freq)
    if origin < data.first or origin > data.last:
        raise ValueError(f"origin {origin} outside the sample {data.first}..{data.last}")
    return origin


# --------------------------------------------------------------------------- backtest mode


def _backtest_mode(
    ctx: _Context, origins: Any, draws: int | None, window: Any, cache: Any, n_jobs: int
) -> OrderingSensitivityResult:
    from scipy import stats as sps

    from rai_bvar.evaluation.backtest import Expanding, Origins, backtest
    from rai_bvar.evaluation.significance import model_confidence_set

    draws = FORECAST_DRAWS if draws is None else _positive_int(draws, "draws")
    data = ctx.data
    index = data.index
    if origins is None:
        if len(index) < 4:
            raise ValueError("the sample is too short for default backtest origins")
        origin_list = list(index[len(index) // 2 : len(index) - 1 : 4])
    elif isinstance(origins, Origins):
        origin_list = origins.resolve(index)
    else:
        seq = [origins] if isinstance(origins, str | pd.Period) else list(origins)
        origin_list = [pd.Period(o, freq=index.freq) for o in seq]
    if not origin_list:
        raise ValueError("no backtest origins")
    window = Expanding() if window is None else window
    H = max(ctx.horizons)
    K, R = len(ctx.orders), ctx.noise_reps
    fit_seeds = [ctx.seed] * K + _noise_fit_seeds(ctx.seed, R)
    models = [_with_seed(m, ctx.seed) for m in ctx.models()]
    runs = [(f"ordering {j}", models[j], ctx.orders[j], _derive(ctx.seed, "backtest", j))
            for j in range(K)]  # fmt: skip
    runs += [
        (f"noise {r}", _with_seed(models[0], fit_seeds[K + r - 1]), ctx.orders[0],
         _derive(ctx.seed, "noise-backtest", r))
        for r in range(1, R + 1)
    ]  # fmt: skip
    var = list(ctx.variables)
    log = _Warnings()
    values, joints, origin_scores, origin_joint, conv = [], [], [], [], []
    for label, m, order, s in runs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bt = backtest(
                m, data.select(order), origins=origin_list, window=window, horizon=H,
                draws=draws, seed=s, cache=cache, n_jobs=n_jobs,
            )  # fmt: skip
        log.record(label, caught)
        sc = bt.scores.sel(horizon=list(ctx.horizons))
        if sorted(map(str, sc["variable"].values)) != sorted(var):  # pragma: no cover
            raise RuntimeError(f"backtest scored {list(sc['variable'].values)}, expected {var}")
        per = np.stack(  # (origin, V, Hs, score), un-permuted by label
            [sc[_SCORE_VARS[name]].transpose("origin", "variable", "horizon")
             .sel(variable=var).values for name in _SCORES], axis=-1,
        )  # fmt: skip
        jl = sc["lpd"].transpose("origin", "horizon").values  # (origin, Hs)
        avg = _nanmean(per, axis=0)  # (V, Hs, score)
        stat = np.stack([avg[..., 0], avg[..., 1], np.sqrt(avg[..., 2])], axis=-1)
        values.append(stat)
        joints.append(_nanmean(jl, axis=0)[:, None])
        origin_scores.append(per)
        origin_joint.append(jl)
        conv.append(int(np.all(sc["converged"].values == 1)) if "converged" in sc else 1)
    log.emit(len(runs))
    vals, jts = np.stack(values), np.stack(joints)
    ojl = np.stack(origin_joint[:K])  # (K, origin, Hs)
    # rank correlations of the orderings' rankings across cells (ARRS Table 4)
    cells = [f"{v}@h{h}" for v in var for h in ctx.horizons] + [f"joint@h{h}" for h in ctx.horizons]
    mat = np.concatenate([vals[:K, :, :, 0].reshape(K, -1), jts[:K, :, 0]], axis=1)
    rank_corr = np.full((len(cells), len(cells)), np.nan)
    if K >= 3:
        finite = np.isfinite(mat).all(axis=0)
        ok = finite & (np.ptp(np.where(finite, mat, 0.0), axis=0) > 0)  # constant cells: NaN
        if ok.sum() >= 2:
            rho = np.asarray(sps.spearmanr(mat[:, ok]).statistic)
            if rho.ndim == 0:  # two cells: spearmanr returns the coefficient
                rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
            sub = np.ix_(np.flatnonzero(ok), np.flatnonzero(ok))
            rank_corr[sub] = rho
    # Model Confidence Set over orderings (loss = −joint log score), per horizon
    mcs = np.full((K, len(ctx.horizons)), np.nan)
    for a in range(len(ctx.horizons)):
        losses = -ojl[:, :, a].T  # (origin, K)
        if np.isfinite(losses).all(axis=1).sum() >= 3:
            mcs[:, a] = model_confidence_set(losses, seed=ctx.seed).pvalues
    nan_mcse = np.full((len(var), len(ctx.horizons), len(_BACKTEST_STATS)), np.nan)
    ds = xr.Dataset(
        {
            "value": (("ordering", "variable", "horizon", "stat"), vals[:K]),
            "joint": (("ordering", "horizon", "jstat"), jts[:K]),
            "noise": (("rep", "variable", "horizon", "stat"), np.concatenate([vals[:1], vals[K:]])),
            "noise_joint": (("rep", "horizon", "jstat"), np.concatenate([jts[:1], jts[K:]])),
            "mcse": (("variable", "horizon", "stat"), nan_mcse),
            "mcse_joint": (("horizon", "jstat"), np.full((len(ctx.horizons), 1), np.nan)),
            "position": (("ordering", "variable"), ctx.positions()),
            "converged": (("ordering",), np.array(conv[:K])),
            "origin_scores": (
                ("ordering", "origin", "variable", "horizon", "score"),
                np.stack(origin_scores[:K]),
            ),
            "origin_joint_lpd": (("ordering", "origin", "horizon"), ojl),
            "rank_corr": (("cell", "cell_"), rank_corr),
            "mcs_pvalue": (("ordering", "horizon"), mcs),
        },
        coords={
            "ordering": np.arange(K),
            "variable": var,
            "horizon": list(ctx.horizons),
            "stat": list(_BACKTEST_STATS),
            "jstat": list(_JSTATS),
            "rep": np.arange(R + 1),
            "origin": [str(o) for o in origin_list],
            "score": list(_SCORES),
            "cell": cells,
            "cell_": cells,
            "label": ("ordering", [" > ".join(o) for o in ctx.orders]),
        },
        attrs=ctx.attrs(
            draws=draws,
            window=window.describe() if hasattr(window, "describe") else str(window),
            fit_seeds=fit_seeds,
            backtest_seeds=[s for *_, s in runs],
            mcs_alpha=0.10,
            mcs_size=[int(np.sum(mcs[:, a] >= 0.10)) for a in range(len(ctx.horizons))],
            fit_warnings=log.by_message,
        ),
    )
    return _finish(ds)


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """``nanmean`` without the empty-slice warning (NaN where every value is NaN)."""
    ok = np.isfinite(x)
    count = ok.sum(axis=axis)
    total = np.where(ok, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# --------------------------------------------------------------------------- validation


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return int(value)


def _horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    hs = (horizons,) if isinstance(horizons, int | np.integer) else tuple(horizons)
    if not hs:
        raise ValueError("horizons must not be empty")
    out = tuple(_positive_int(h, "horizons") for h in hs)
    if len(set(out)) != len(out):
        raise ValueError(f"horizons {list(out)} repeat a value")
    return tuple(sorted(out))


def _quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if not qs or not all(0.0 < q < 1.0 for q in qs):
        raise ValueError(f"quantiles must lie in (0, 1), got {list(qs)}")
    if len({_qname(q) for q in qs}) != len(qs):
        raise ValueError(f"quantiles {list(qs)} repeat a value")
    return tuple(sorted(qs))
