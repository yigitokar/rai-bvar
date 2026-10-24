"""``BVAR`` + ``VARData`` → engine inputs, and ``BVAR.fit`` for MCMC compositions (DESIGN §1.3,
§6, §10.5, §11.3; WP18).

This module is the R9 boundary: everything data-dependent is computed here, per estimation
window, from the window alone, on the host, and handed to the engine as arrays (nothing inside JIT
sees the calendar):

* EB plug-in hyperparameters (``engines.mcmc.eb``, DESIGN §6.5) or the fixed values;
* ψ, own-lag means and the frequency (``priors.scales.window_scales``);
* the prior arrays of §2.6 from the WP6 builders: Mode I / Mode II δ arrays with the D3 centring
  columns ``M_c`` (``priors.structural``) or the CCCM arrays (``priors.reduced``), the SOC/SUR
  dummy rows from the window's first ``p`` observations and their weights (``priors.dummies``,
  WP25), the ``a_i`` arrays at the composition's ``contemp_scaling``, and the volatility arrays
  (``priors.sv_arrays``: IG ``λ``, data-centred SV levels ``log(ψ_i/ι)``, offsets ``c_i = c_rel ψ_i``,
  the outlier ``Beta`` from the frequency, the ν weights);
* the padded window (``data.engine_data_from_vardata``) and the static configuration.

Deterministic size rules (DESIGN §10.5; shape rules, never machine rules, so the same ``(spec,
data)`` always resolves the same way, E14):

    T_pad      = t_bucket·⌈T_eff/t_bucket⌉
    gram       "auto" → "packed" if T_pad·P·8 ≤ 256 MiB, else "packed_tiled" with the largest
               tile (a multiple of 512 packed columns) such that T_pad·tile·8 ≤ 64 MiB;
               P = q(q+1)/2 (structural δ-design) or k(k+1)/2 (reduced form, packed X)
    const_path "auto" → structural Linear × Constant without a transitory scale and with a
               non-varying coefficient prior: Mode I "eigen" (with SOC/SUR dummy rows on the
               dense prior precision; N05 §7.7, WP25), Mode II "fixed_factor", iw_equivalent
               "cholesky_const"; anything else "weighted" (§3.7.2).
               "cholesky" → "cholesky_const" where weights are time-constant, else "weighted"
               (the general Cholesky path): the fast paths are off either way
    cccm_precompute ⇔ n·k²·8·C ≤ 512 MiB

``fit_bvar`` is the lazy entry point of ``BVAR.fit`` (DESIGN §1.1 rule 2): it resolves, derives the
data salt ``crc32(data.fingerprint())`` (variables are part of the fingerprint, so windows and
orderings get distinct streams; §5.6), applies the seed override, remaps a warm start onto the
window (§5.5) and calls ``fit_mcmc``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pandas as pd
from jax import random

from rai_bvar.components import SV
from rai_bvar.data import transforms
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import n_deterministic
from rai_bvar.engines.analytic import _COVID_WINDOW
from rai_bvar.engines.mcmc import _jax, eb, keys
from rai_bvar.engines.mcmc.config import (
    COEF_VARIES,
    MeanStatic,
    Shapes,
    StaticConfig,
    StoreStatic,
    bucket,
    scaling_mode,
)
from rai_bvar.engines.mcmc.data import engine_data_from_vardata
from rai_bvar.engines.mcmc.state import (
    ChainState,
    Empty,
    EngineData,
    EqPrior,
    PriorArrays,
)
from rai_bvar.priors.dummies import dummy_arrays, dummy_hyper
from rai_bvar.priors.reduced import reduced_arrays
from rai_bvar.priors.scales import WindowScales, window_scales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.priors.sv_arrays import (
    C_REL,
    constant_prior,
    outlier_beta,
    sv_prior_arrays,
    transitory_arrays,
)
from rai_bvar.settings import SV_DEFAULT_DRAWS, FitPlan, MCMCSettings, Runtime

if TYPE_CHECKING:
    from rai_bvar.api import BVAR
    from rai_bvar.engines.mcmc.results import MCMCFit
    from rai_bvar.engines.mcmc.warmstart import FinalState

__all__ = [
    "Resolved",
    "fit_bvar",
    "resolve",
    "resolve_cccm_precompute",
    "resolve_const_path",
    "resolve_gram",
]

_MiB = 2**20
PACKED_MAX_BYTES = 256 * _MiB  # T_pad·P·8 bound of the packed Gram (DESIGN §10.5)
TILE_MAX_BYTES = 64 * _MiB  # T_pad·tile·8 bound of one packed tile
TILE_UNIT = 512  # tiles are multiples of 512 packed columns
CCCM_MAX_BYTES = 512 * _MiB  # n·k²·8·C bound of the precomputed CCCM precisions


@dataclass(frozen=True)
class Resolved:
    """Engine inputs of one window (DESIGN §13 WP18).

    ``static``: the full ``StaticConfig`` (shapes with ``T_pad``, resolved ``gram`` /
    ``const_path`` / ``cccm_precompute``, storage, ``chunk = runtime.chunk``); ``data``: the padded
    window (NumPy leaves); ``prior``: ``PriorArrays`` (NumPy leaves); ``periods``: the labels of the
    ``T_eff`` target rows; ``hyper``: the hyperparameters in use (λ, α, cross, κ_a, κ_d, ψ, own-lag
    means, SV level centres and offsets, the outlier ``Beta``, EB method, free set, note);
    ``notes``: informational notes (the reduced-form EB note); ``eb_converged``: the EB flag
    (``None`` with ``hierarchy="fixed"``); ``salt``: ``crc32(data.fingerprint())``.

    ``warning_inputs`` (an addition to the DESIGN fields) carries the resolution step's inputs of
    the DESIGN §8.3 warning table to ``FitMeta.warning_inputs``: ``covers_2020``, ``tcode_clash``
    (the Phase 1 checks, recomputed for the MCMC fit), ``eb_warnings`` (EB optimisation and ψ
    messages) and ``sv_budget`` (``(chains, draws)`` of an SV composition below the SV default
    budget, else ``None``; D-35).
    """

    static: StaticConfig
    data: EngineData
    prior: PriorArrays
    periods: pd.PeriodIndex
    hyper: dict[str, Any]
    notes: tuple[str, ...]
    eb_converged: bool | None
    salt: int
    warning_inputs: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- size rules (§10.5)


def resolve_gram(setting: str, T_pad: int, P: int) -> tuple[str, int]:
    """``(gram, gram_tile)`` from ``MCMCSettings.gram`` by the DESIGN §10.5 shape rule.

    ``"auto"``: ``"packed"`` when ``T_pad·P·8 ≤ 256 MiB``, else ``"packed_tiled"``. A tiled Gram
    (``"auto"`` or explicit) gets the largest tile, a multiple of 512 packed columns, with
    ``T_pad·tile·8 ≤ 64 MiB`` (at least 512). ``"packed"`` and ``"batched"`` are taken as given.
    """
    if setting not in ("auto", "packed", "packed_tiled", "batched"):
        raise ValueError(f"unknown gram setting {setting!r}")
    if setting == "auto":
        if T_pad * P * 8 <= PACKED_MAX_BYTES:
            return "packed", 0
        setting = "packed_tiled"
    if setting == "packed_tiled":
        units = TILE_MAX_BYTES // (T_pad * 8) // TILE_UNIT
        return "packed_tiled", TILE_UNIT * max(1, units)
    return setting, 0


def resolve_const_path(setting: str, static_parts: Any) -> str:
    """``MeanStatic.const_path`` from ``MCMCSettings.const_path`` (DESIGN §3.7.2 path rule).

    ``static_parts`` is ``api.static_parts(model)``. The fast paths need the structural ``Linear``
    mean, time-constant weights (Constant core, no transitory scale) and a non-varying
    coefficient prior; ``"cholesky"`` turns them off. SOC/SUR dummy rows (WP25) do not change the
    path: with them the Mode I prior precision is dense and ``eigen`` factors it
    (``linalg.eigen_prepare_dense``, N05 §7.7; 0.46 vs 1.53 ms/sweep for ``cholesky_const`` at
    ``n = 20, p = 4, T = 300``), and Mode II's ``fixed_factor`` holds the λ-free rows.
    """
    if setting not in ("auto", "cholesky"):
        raise ValueError(f"unknown const_path setting {setting!r}")
    if static_parts.mean_kind != "linear_structural" or not static_parts.vol.constant_weights:
        return "weighted"
    if setting == "cholesky":
        return "cholesky_const"
    if COEF_VARIES[static_parts.coef.kind]:
        return "weighted"
    mean = MeanStatic(kind=static_parts.mean_kind, coef_scaling=static_parts.coef_scaling)
    mode = scaling_mode(static_parts.struct, mean)
    return {"I": "eigen", "II": "fixed_factor", "iw_equivalent": "cholesky_const"}[mode]


def resolve_cccm_precompute(n: int, k: int, chains: int) -> bool:
    """Every CCCM precision factored before the ``j``-loop iff ``n·k²·8·C ≤ 512 MiB`` (§10.5)."""
    return n * k * k * 8 * chains <= CCCM_MAX_BYTES


# --------------------------------------------------------------------------- resolution


def resolve(
    model: BVAR, data: VARData, runtime: Runtime, *, plan: FitPlan | None = None
) -> Resolved:
    """Engine inputs of ``model`` on the window ``data`` (DESIGN §1.3, §6, §10.5).

    Reads ``data`` only (R9): a window ``data.upto(origin)`` yields arrays that do not depend on
    later observations. ``plan`` (a keyword addition to the DESIGN signature) sets
    ``StaticConfig.store.in_loop_forecast = plan.horizon`` (§5.8).
    """
    from rai_bvar.api import static_parts

    if getattr(model, "engine", None) != "mcmc":
        raise ValueError("resolve() serves MCMC compositions; analytic models use AnalyticFit")
    if not isinstance(data, VARData):
        raise TypeError(f"data must be a VARData, got {type(data).__name__}")
    if not isinstance(runtime, Runtime):
        raise TypeError(f"runtime must be a Runtime, got {runtime!r}")
    if plan is not None and not isinstance(plan, FitPlan):
        raise TypeError(f"plan must be a FitPlan or None, got {plan!r}")
    settings: MCMCSettings = model.mcmc
    prior, vol = model.mean.prior, model.vol
    lags, det = model.lags, model.deterministic
    parts = static_parts(model)

    # hyperparameters: EB plug-in (DESIGN §6.5) or the fixed values
    outcome = eb.eb_hyper(model, data) if prior.hierarchy == "eb" else None
    psi = outcome.hyper["psi"] if outcome is not None and "psi" in prior.free else None
    scales = window_scales(data, prior, lags, det, psi=psi)
    if outcome is None:
        lam, alpha = prior.lambda_, prior.alpha
    else:
        lam, alpha = outcome.hyper["lambda"], outcome.hyper["alpha"]

    # window and static configuration
    engine_data, periods = engine_data_from_vardata(data, lags, det, settings.t_bucket)
    n, n_det = data.n_vars, n_deterministic(det)
    structural = parts.mean_kind == "linear_structural"
    mode = scaling_mode(
        parts.struct, MeanStatic(kind=parts.mean_kind, coef_scaling=parts.coef_scaling)
    )
    dummies, dummy_w = dummy_arrays(
        data.values, scales, prior, None if outcome is None else outcome.hyper,
        lags=lags, n_det=n_det, mode=mode if structural else "reduced",
    )  # fmt: skip
    n_dummy = 0 if isinstance(dummies, Empty) else int(dummies.Y.shape[0])
    T_eff = int(np.asarray(engine_data.T_eff))
    shapes = Shapes.of(n, lags, n_det, bucket(T_eff, settings.t_bucket))
    gram, tile = resolve_gram(settings.gram, shapes.T_pad, shapes.P if structural else shapes.P_x)
    mean_static = MeanStatic(
        kind=parts.mean_kind,
        coef_scaling=parts.coef_scaling,
        const_path=resolve_const_path(settings.const_path, parts),
        gram=gram,
        gram_tile=tile,
        cccm_precompute=resolve_cccm_precompute(n, shapes.k, settings.chains),
        n_dummy=n_dummy,
    )
    static = StaticConfig(
        shapes=shapes,
        struct=parts.struct,
        mean=mean_static,
        coef=parts.coef,
        vol=parts.vol,
        store=StoreStatic(
            paths=settings.store == "paths",
            in_loop_forecast=0 if plan is None else plan.horizon,
        ),
        chunk=runtime.chunk,
    )

    # prior arrays (DESIGN §2.6) from the WP6 builders
    values = {"lambda": lam, "alpha": alpha}
    if structural:
        coef, shared = structural_arrays(
            scales, prior, values, n=n, lags=lags, n_det=n_det, mode="II" if mode == "II" else "I"
        )
    else:
        coef, shared = reduced_arrays(scales, prior, values, n=n, lags=lags, n_det=n_det)
    if n_dummy:  # SOC/SUR prior rows (WP25)
        coef = dataclasses.replace(coef, dummy_w=dummy_w)
        shared = dataclasses.replace(shared, dummies=dummies)
    contemp = contemp_arrays(scales, prior, n, scaling=parts.struct.contemp_scaling)
    if parts.vol.persistent == "constant":
        core, trans = constant_prior(scales, n), Empty()
    else:
        core = sv_prior_arrays(vol, scales)
        trans = transitory_arrays(vol, scales, parts.vol.nu_grid)
    arrays = PriorArrays(
        eq=EqPrior(coef=coef, contemp=contemp, vol_core=core, transitory=trans), shared=shared
    )

    hyper = _hyper(model, scales, lam, alpha, outcome, core)
    if n_dummy:  # the SOC/SUR weights in use (EB values when free) and the row count (WP25)
        hyper.update(dummy_hyper(prior, None if outcome is None else outcome.hyper))
        hyper["n_dummy"] = n_dummy
    notes = () if outcome is None or outcome.note is None else (outcome.note,)
    messages = tuple(scales.notes) + (() if outcome is None else outcome.warnings)
    return Resolved(
        static=static,
        data=engine_data,
        prior=arrays,
        periods=periods,
        hyper=hyper,
        notes=notes,
        eb_converged=None if outcome is None else outcome.converged,
        salt=keys.data_salt(data.fingerprint()),
        warning_inputs={
            "covers_2020": _covers_2020(periods),
            "tcode_clash": _tcode_clash(model, data, scales),
            "eb_warnings": tuple(dict.fromkeys(messages)),
            "sv_budget": _sv_budget(model),
        },
    )


def _sv_budget(model: BVAR) -> tuple[int, int] | None:
    """``(chains, draws)`` of a stochastic-volatility composition fitted with fewer kept draws per
    chain than ``SV_DEFAULT_DRAWS`` (the D-35 warning), else ``None``: never for ``Constant``
    volatility, nor for an explicit budget at or above the default."""
    settings: MCMCSettings = model.mcmc
    if isinstance(model.vol, SV) and settings.draws < SV_DEFAULT_DRAWS:
        return settings.chains, settings.draws
    return None


def _hyper(
    model: BVAR,
    scales: WindowScales,
    lam: float,
    alpha: float,
    outcome: eb.EBOutcome | None,
    core: Any,
) -> dict[str, Any]:
    """The hyperparameters in use (``MCMCFit.hyper``; DESIGN §11.3)."""
    prior, vol = model.mean.prior, model.vol
    hyper: dict[str, Any] = {
        "hierarchy": prior.hierarchy,
        "lambda": float(lam),
        "alpha": float(alpha),
        "cross": prior.cross,
        "contemp": prior.contemp,
        "var_const": prior.var_const,
        "psi": scales.psi.copy(),
        "own_means": scales.own_means.copy(),
    }
    if outcome is not None:
        hyper["eb_method"] = outcome.hyper["eb_method"]
        hyper["eb_free"] = outcome.hyper["eb_free"]
        hyper["eb_converged"] = bool(outcome.converged)
        if outcome.note is not None:
            hyper["eb_note"] = outcome.note
    if not isinstance(vol, SV):  # Constant
        return hyper
    hyper["sv_level"] = np.array(core.m0 if vol.rw else core.b_mu)  # log(ψ_i/ι) unless given
    if vol.offset == "relative":
        hyper["offset_rule"], hyper["c_rel"] = "relative", C_REL
    else:
        hyper["offset_rule"] = "zero" if vol.offset == 0.0 else "absolute"
        hyper["offset"] = float(vol.offset)
    hyper["offsets"] = np.exp(np.asarray(core.log_c, dtype=np.float64))
    if vol.outliers:
        beta = outlier_beta(vol, scales.freq)
        hyper["outlier_beta_a"], hyper["outlier_beta_b"] = beta.a, beta.b
    return hyper


def _covers_2020(periods: pd.Index) -> bool:
    """The estimation periods overlap March–December 2020 (the Phase 1 SPEC §5.6 check)."""
    start, end = _COVID_WINDOW
    return any(p.end_time >= start and p.start_time <= end for p in periods)


def _tcode_clash(model: BVAR, data: VARData, scales: WindowScales) -> tuple[str, ...]:
    """Variables whose explicit own-lag prior mean contradicts its transformation code (levels
    with mean < 0.5 or differences with mean ≥ 0.5; the Phase 1 SPEC §5.3 check)."""
    if model.mean.prior.mean is None:
        return ()
    return tuple(
        v
        for v, m in zip(data.variables, scales.own_means, strict=True)
        if (data.tcodes[v] in transforms.PERSISTENT_TCODES) != (m >= 0.5)
    )


# --------------------------------------------------------------------------- BVAR.fit


WARM_SLOT = 2**32 - 1
"""The init-stream slot of the warm-start key: the largest uint32, which no chain index reaches."""


def warm_key(seed: int, salt: int) -> jax.Array:
    """Key of the warm-start remap (forward-simulated ``h``; DESIGN §5.5)::

        warm_key = key(seed) → Stream.INIT → salt → WARM_SLOT → Block.EQ_VOL

    The cold-start keys of the same stream are ``key(seed) → Stream.INIT → salt → c → Block → i →
    Sub`` (``keys.init_keys``) with a chain index ``c < C``. :data:`WARM_SLOT` ``= 2**32 − 1`` sits
    where they put ``c`` and is never a chain index (``c`` runs over ``0..C−1`` with ``C`` far below
    ``2**32 − 1``: a chain axis that long cannot be allocated), so the fold path of the warm key,
    and of the keys the remap folds from it (``→ c → i``), leaves every cold-start path of every
    chain at the slot after the salt: no warm-start key is a cold-start key. (Folding
    ``Block.EQ_VOL`` straight after the salt would not do: ``salt → 2`` *is* the cold init key of
    chain 2.)"""
    root = random.key(int(seed), impl=keys.PRNG_IMPL)
    base = random.fold_in(random.fold_in(root, int(keys.Stream.INIT)), np.uint32(salt))
    return keys.block_key(random.fold_in(base, np.uint32(WARM_SLOT)), keys.Block.EQ_VOL)


def _warm_state(
    init: FinalState | MCMCFit | None, resolved: Resolved, seed: int, chains: int
) -> ChainState | None:
    """``init`` remapped onto the resolved window, or ``None`` (cold start; a ``UserWarning``
    names the reason when ``init`` cannot be used)."""
    import warnings

    from rai_bvar.engines.mcmc.results import MCMCFit
    from rai_bvar.engines.mcmc.warmstart import FinalState, warm_state

    if init is None:
        return None
    prev = init.final_state if isinstance(init, MCMCFit) else init
    if not isinstance(prev, FinalState):
        raise TypeError(f"init must be an MCMCFit, a FinalState or None, got {type(init).__name__}")
    if prev.chains < chains:
        warnings.warn(
            f"warm start ignored (it has {prev.chains} chains, the settings ask for {chains}); "
            "cold start",
            UserWarning,
            stacklevel=4,
        )
        return None
    key = warm_key(seed, resolved.salt)
    return warm_state(prev, resolved.data, resolved.periods, resolved.static, key)


def fit_bvar(
    model: BVAR,
    data: VARData,
    *,
    seed: int | None,
    init: FinalState | MCMCFit | None,
    runtime: Runtime | None,
    plan: FitPlan | None,
) -> MCMCFit:
    """``BVAR.fit`` for MCMC compositions (DESIGN §11.3): :func:`resolve`, then ``fit_mcmc``.

    ``seed`` overrides ``MCMCSettings.seed``; ``init`` (an ``MCMCFit`` or its ``final_state``)
    warm-starts the chains (§5.5; ``MCMCSettings.warm_burn`` sweeps of burn-in; a cold start with
    a warning when the composition, ``n``, the lags or the chain count do not fit); ``runtime``
    defaults to ``Runtime()``; ``plan`` requests in-loop forecasts (§5.8).
    """
    from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc

    runtime = Runtime() if runtime is None else runtime
    settings: MCMCSettings = model.mcmc
    seed = settings.seed if seed is None else int(seed)
    with _jax.x64():
        resolved = resolve(model, data, runtime, plan=plan)
        init_state = _warm_state(init, resolved, seed, settings.chains)
        meta = FitMeta(
            model=model,
            vardata=data,
            periods=resolved.periods,
            variables=tuple(data.variables),
            hyper=resolved.hyper,
            notes=resolved.notes,
            eb_converged=resolved.eb_converged,
            warning_inputs=dict(resolved.warning_inputs),
        )
        return fit_mcmc(
            resolved.static,
            resolved.data,
            resolved.prior,
            settings,
            runtime,
            seed=seed,
            salt=resolved.salt,
            meta=meta,
            init=init_state,
            plan=plan,
        )
