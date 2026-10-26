"""``fit_mcmc`` and the fitted MCMC model ``MCMCFit`` (DESIGN §11.3, §5.3, §5.6, §8).

``fit_mcmc(static, data, prior, settings, runtime, *, seed, salt, meta)`` is the engine-level entry
point (validation, parity, and ``BVAR.fit`` through ``resolve.fit_bvar``): it builds the runner of
``static`` (``make_runner``), runs the chains (``run_chains``), and assembles :class:`MCMCFit`:
the posterior ``DataTree`` (§5.3), the monitored-quantity diagnostics and the ``converged`` flag
(``engines.mcmc.monitor``, WP15; §8.1–§8.2), the final state for warm starts (§5.5) and the
reproducibility metadata of §5.6. ``forecast``, ``irf`` and ``draw_parameters`` delegate to
``engines.mcmc.forecast`` (WP17), so this module and that one have disjoint owners.

For the forecast engine (WP17) a fit also exposes, on the host: ``program`` (the composed
kernels), ``engine_data`` (``EngineData`` of the window: ``hist``, ``t_last``), ``prior``
(``PriorArrays``), ``param_draws`` (the stored ``ParamDraw`` with leaves ``(chain, draw, …)``),
``path_draws`` (``PathDraw`` or ``None``), ``draw_index`` (kept-draw index of each stored draw),
``seed`` / ``salt``, and ``in_loop_forecast`` (:class:`InLoopForecast`, the forecasts computed
inside the sampling loop for a ``FitPlan``, DESIGN §5.8; ``None`` without a plan):
``engines.mcmc.forecast.forecast`` returns them as a ``ForecastResult`` when ``(horizon, draws,
seed)`` match the plan (:meth:`InLoopForecast.matches`).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar.engines.mcmc import _jax
from rai_bvar.engines.mcmc import monitor as monitor_mod
from rai_bvar.engines.mcmc.config import StaticConfig, scaling_mode
from rai_bvar.engines.mcmc.keys import PRNG_IMPL
from rai_bvar.engines.mcmc.plan import Plan, build_plan
from rai_bvar.engines.mcmc.runner import RunOutput, make_runner, run_chains
from rai_bvar.engines.mcmc.state import (
    ChainState,
    ConstantPrior,
    EngineData,
    PriorArrays,
    SVPrior,
    leaf_path,
)
from rai_bvar.engines.mcmc.storage import regressor_names
from rai_bvar.engines.mcmc.warmstart import FinalState

if TYPE_CHECKING:
    from rai_bvar.data.dataset import VARData
    from rai_bvar.forecast.result import ForecastResult
    from rai_bvar.models.base import SeedLike
    from rai_bvar.priors.structural import ImpliedMoments
    from rai_bvar.settings import FitPlan, MCMCSettings, Runtime

__all__ = ["FitMeta", "InLoopForecast", "MCMCFit", "fit_mcmc"]


@dataclass(frozen=True)
class FitMeta:
    """What the engine needs to label a fit (DESIGN §13 WP14): the user model and data when the
    fit comes from ``BVAR.fit`` (``None`` at the engine level), the periods of the ``T_eff`` rows,
    the variable ordering, the hyperparameters in use (ψ, λ, offsets, …), notes (e.g. the EB note)
    and the EB convergence flag (``None`` without EB).

    ``warning_inputs`` (an addition to the DESIGN fields, default empty) carries the resolution
    step's inputs of the DESIGN §8.3 warning table to ``monitor.DiagnosticsInput``: e.g.
    ``covers_2020``, ``tcode_clash``, ``eb_warnings``, ``outlier_at_max``."""

    model: Any = None
    vardata: VARData | None = None
    periods: pd.Index | None = None
    variables: tuple[str, ...] = ()
    hyper: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    eb_converged: bool | None = None
    warning_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class InLoopForecast:
    """Forecasts computed inside the sampling loop for a ``FitPlan`` (DESIGN §5.8).

    Every chain forecasts from exactly the draws DESIGN §7.1 selects post hoc
    (:func:`storage.forecast_selection`): ``per_chain = ⌈S/C⌉`` stored parameter draws, evenly
    spaced, one path each (``C·per_chain ≥ S`` paths, fewer than ``C`` extra); when that exceeds
    the stored draws, every stored draw with ``paths_per_draw = ⌈S/(C·D_stored)⌉`` paths. ``out``:
    the forecast core's output (``ForecastOut``) with leaves ``(chain, M, …)``, ``M = per_chain ·
    paths_per_draw``, each ``(chain, m)`` slice one path from one parameter draw: the kept draw
    ``draw_index[m]`` (the ``draw`` coordinate of ``fit.posterior``), path ``replicate[m]``.
    Shocks are those of :func:`runner.forecast_shocks` keyed by ``(fseed, chain, d)`` with ``d =
    draw_index[m] + replicate[m]·D`` (``D`` kept draws per chain; ``d`` is the kept-draw index
    for single paths), ``fseed`` from ``seed`` as in DESIGN §7.1."""

    horizon: int
    draws: int
    seed: int
    fseed: int
    per_chain: int
    paths_per_draw: int
    kept_draws: int  # D, kept draws per chain
    draw_index: np.ndarray
    replicate: np.ndarray
    out: Any

    @property
    def key_index(self) -> np.ndarray:
        """The forecast key's ``d`` of each ``(chain, m)`` slice (DESIGN §5.6)."""
        return self.draw_index + self.replicate * int(self.kept_draws)

    def matches(self, horizon: int, draws: int, seed: Any) -> bool:
        """Whether ``fit.forecast(horizon, draws, seed)`` asks for exactly these forecasts."""
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            return False
        return (int(horizon), int(draws), int(seed)) == (self.horizon, self.draws, self.seed)


def _monitor_fn(name: str) -> Any:
    fn = getattr(monitor_mod, name, None)
    if fn is None:
        raise NotImplementedError(
            f"engines.mcmc.monitor.{name} (WP15) is not available yet (DESIGN §8)"
        )
    return fn


def _forecast_fn(name: str) -> Any:
    from rai_bvar.engines.mcmc import forecast as forecast_mod

    fn = getattr(forecast_mod, name, None)
    if fn is None:
        raise NotImplementedError(
            f"engines.mcmc.forecast.{name} (WP17) is not available yet (DESIGN §7)"
        )
    return fn


def _metadata(
    static: StaticConfig, run: RunOutput, meta: FitMeta, variables: tuple[str, ...]
) -> dict:
    """Reproducibility metadata recorded in every fit (DESIGN §5.6)."""
    import jaxlib

    from rai_bvar._version import __version__

    out: dict[str, Any] = {
        "engine": "mcmc",
        "package_version": __version__,
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
        "prng_impl": PRNG_IMPL,
        "jax_threefry_partitionable": int(bool(jax.config.jax_threefry_partitionable)),
        "platform": jax.default_backend(),
        # the devices the chains ran on and how (DESIGN §5.2, §5.6): draws are bitwise
        # reproducible for one device count and chain method, and equal across device counts,
        # chain methods and chain-count prefixes only to rounding (Stage 2 amendment)
        "device_count": int(run.devices),
        "chain_method": run.chain_method,
        "local_device_count": int(jax.local_device_count()),
        "reproducibility": (
            "bitwise across Runtime.chunk, burn piece size and fused/unfused plan execution; "
            "to rounding across chain-count prefixes and vmap vs shard_map (DESIGN §5.6)"
        ),
        "seed": run.seed if isinstance(run.seed, int) else str(run.seed),
        "salt": run.salt if isinstance(run.salt, int) else str(run.salt),
        "T_pad": static.shapes.T_pad,
        "gram": static.mean.gram,
        "const_path": static.mean.const_path,
        "prior_mode": scaling_mode(static.struct, static.mean),
        "mean_kind": static.mean.kind,
        "persistent": static.vol.persistent,
        "transitory": static.vol.transitory,
        "mixture": static.vol.mixture,
        "exact": int(static.vol.exact),
        "ordering": list(variables),
    }
    hyper = meta.hyper or {}
    for key in ("offset", "offset_rule", "c_rel"):
        if key in hyper:
            value = hyper[key]
            out[key] = value if isinstance(value, int | float | str) else str(value)
    model = meta.model
    if model is not None and hasattr(model, "spec"):
        from rai_bvar.models.base import spec_hash

        try:
            out["spec_hash"] = spec_hash(model)
        except Exception:  # a model whose spec does not serialise: record nothing
            pass
    return out


def _constant_data(prior: PriorArrays) -> dict[str, np.ndarray]:
    host = jax.tree.map(np.asarray, jax.device_get(prior))
    return {
        "prior_" + leaf_path(p).replace(".", "_"): x
        for p, x in jax.tree_util.tree_flatten_with_path(host)[0]
    }


class MCMCFit:
    """A fitted MCMC model (DESIGN §11.3). Built by :func:`fit_mcmc`."""

    engine = "mcmc"

    def __init__(
        self,
        *,
        static: StaticConfig,
        run: RunOutput,
        program: Any,
        data: EngineData,
        prior: PriorArrays,
        settings: MCMCSettings,
        meta: FitMeta,
    ) -> None:
        self.static = static
        self.program = program
        self.settings = settings
        self.meta = meta
        self.model = meta.model
        self.data = meta.vardata
        self.engine_data: EngineData = jax.tree.map(np.asarray, jax.device_get(data))
        self.prior: PriorArrays = jax.tree.map(np.asarray, jax.device_get(prior))
        n = static.shapes.n
        self.ordering: tuple[str, ...] = (
            tuple(meta.variables) if meta.variables else tuple(f"y{i}" for i in range(n))
        )
        self.hyper: dict[str, Any] = dict(meta.hyper or {})
        self.notes: tuple[str, ...] = tuple(meta.notes)
        self.plan: Plan = build_plan(program)
        self.seed, self.salt = run.seed, run.salt
        self.timing: dict[str, float] = dict(run.timing)
        self.acceptance = run.acceptance
        self.latent = run.latent
        self._run = run
        T_eff = int(np.asarray(self.engine_data.T_eff))
        periods = meta.periods if meta.periods is not None else pd.RangeIndex(T_eff)
        self.periods = periods
        self.final_state = FinalState(
            state=run.final_state,
            periods=periods,
            static=static,
            seed_meta={
                "seed": run.seed, "salt": run.salt, "burn": run.burn, "draws": run.draws,
                "thin": run.thin, "sweeps": run.sweeps, "warm_start": run.warm,
            },
        )  # fmt: skip
        store = run.store
        self.param_draws = store.stream("params")
        self.draw_index = store.plan.streams["params"].draw_index()
        self.path_draws = store.stream("paths") if "paths" in store.plan.streams else None
        self.monitor: dict[str, np.ndarray] = store.monitor()
        self.in_loop_forecast: InLoopForecast | None = None
        if run.forecast is not None and "forecast" in store.plan.streams:
            fc = run.forecast
            R = int(fc["paths_per_draw"])
            index = store.plan.streams["forecast"].draw_index()
            lead = len(store.plan.lead)  # (chain, M, R, …) → (chain, M·R, …)
            out = jax.tree.map(
                lambda x: np.asarray(x).reshape(*x.shape[:lead], -1, *x.shape[lead + 2 :]),
                store.stream("forecast"),
            )
            self.in_loop_forecast = InLoopForecast(
                horizon=int(fc["horizon"]), draws=int(fc["draws"]), seed=int(fc["seed"]),
                fseed=int(fc["fseed"]), per_chain=int(fc["per_chain"]), paths_per_draw=R,
                kept_draws=int(run.draws), draw_index=np.repeat(index, R),
                replicate=np.tile(np.arange(R, dtype=np.int64), index.size), out=out,
            )  # fmt: skip

        spec = _monitor_fn("monitor_spec")(static, self.ordering)
        self.monitor_spec = spec
        converged, messages = _monitor_fn("converged")(
            self.monitor, spec, settings, error=False, eb_converged=meta.eb_converged
        )
        self.converged = bool(converged)
        # per-equation table inputs (DESIGN §8.4): acceptance rates, the smallest pivot, and the
        # Rao-Blackwellised outlier probabilities over the T_eff real rows
        acc = {**run.acceptance, "min_pivot": run.min_pivot, "T_eff": np.asarray(T_eff)}
        if "p_outlier" in run.latent:
            acc["p_outlier"] = run.latent["p_outlier"]
        self.diagnostics: pd.DataFrame = _monitor_fn("diagnostics_table")(self.monitor, spec, acc)
        catalogue = getattr(monitor_mod, "warning_catalogue", None)
        inputs_cls = getattr(monitor_mod, "DiagnosticsInput", None)
        if catalogue is not None and inputs_cls is not None:  # the DESIGN §8.3 table
            messages = catalogue(
                inputs_cls(
                    static=static, settings=settings, variables=self.ordering, mon=self.monitor,
                    spec=spec, acc=acc, error=False, eb_converged=meta.eb_converged,
                    **dict(meta.warning_inputs or {}),
                )
            )  # fmt: skip
        self.warnings: list[str] = []
        for message in messages:
            self._warn(message)

        attrs = _metadata(static, run, meta, self.ordering)
        attrs["converged"] = int(self.converged)
        store.attrs.update(attrs)
        store.constant.update(_constant_data(self.prior))
        self.metadata: dict[str, Any] = {**store.attrs}
        self.posterior: xr.DataTree = store.to_datatree(
            dataclasses.replace(meta, periods=periods, variables=self.ordering, hyper=self.hyper)
        )  # fmt: skip

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=4)

    # ------------------------------------------------------------------ inspection
    @property
    def psi(self) -> np.ndarray | None:
        psi = self.hyper.get("psi")
        return None if psi is None else np.asarray(psi)

    @property
    def store(self) -> Any:
        """The host store of the draws (in memory or memory-mapped)."""
        return self._run.store

    def reduced_form_draws(self) -> xr.DataArray:
        """``B = (A⁻¹Γ)'`` (structural) or ``Π`` (reduced) per stored draw, computed lazily:
        dims ``(chain, draw, regressor, variable)``."""
        p = self.param_draws
        s = self.static.shapes
        n = s.n
        if hasattr(p.glob.mean, "Pi"):
            B = np.asarray(p.glob.mean.Pi)
        else:
            gamma = np.asarray(p.mean.gamma)  # (..., n, k)
            a = np.asarray(p.struct.a)  # (..., n, n−1)
            A = np.broadcast_to(np.eye(n), (*a.shape[:-2], n, n)).copy()
            rows, cols = np.tril_indices(n, -1)
            A[..., rows, cols] = a[..., rows, cols]
            B = np.swapaxes(np.linalg.solve(A, gamma), -1, -2)  # A is unit lower triangular
        regs = regressor_names(self.ordering, s.lags, s.n_det)
        lead = ("chain", "draw") if B.ndim == 4 else ("fit", "chain", "draw")
        return xr.DataArray(
            B,
            dims=(*lead, "regressor", "variable"),
            coords={"draw": self.draw_index, "regressor": regs, "variable": list(self.ordering)},
            name="B",
        )

    def prior_vol_mean(self) -> np.ndarray:
        """``λ_i`` (n,) behind the implied moments: the IG prior mean ``C/(c − 1)`` (Constant;
        ``ψ_ι/ι`` in Mode I, DESIGN §6.1), or under SV the plug-in at the data-centred level
        ``exp(b_μ)`` (``exp(m₀)`` with a random walk) ``= ψ_ι/ι`` (§6.3; Stage 2 amendment: a
        plug-in, not ``E[λ_i] = E[exp h]``, which exceeds it by Jensen's inequality); the
        cold-start centre of ``runner``. :meth:`vol_mean_label` names the choice."""
        core = self.prior.eq.vol_core
        if isinstance(core, ConstantPrior):
            c, C = np.asarray(core.shape, np.float64), np.asarray(core.scale, np.float64)
            return C / np.where(c > 1.0, c - 1.0, c)
        if isinstance(core, SVPrior):
            return np.exp(np.asarray(core.m0 if self.static.vol.rw else core.b_mu, np.float64))
        raise TypeError(f"no prior mean of λ for {type(core).__name__}")

    def vol_mean_label(self) -> str:
        """How :meth:`prior_vol_mean` sets ``λ_i`` (the ``vol_mean`` field of ``summary()``)."""
        if self.static.vol.common:  # P3 §5: Σ_t = e^{h_t}Σ; the moments are those of Σ (h = 0)
            return "C/(c-1) prior mean (CommonSV: the base Σ at h = 0)"
        if self.static.vol.persistent == "constant":
            return "C/(c-1) prior mean (Constant)"
        return "exp(m0) plug-in (SV, RW)" if self.static.vol.rw else "exp(b_mu) plug-in (SV)"

    def implied_prior_moments(self) -> ImpliedMoments | None:
        """Implied reduced-form prior moments of ``B = (A⁻¹Γ)'`` and ``Σ`` (R8; N05 §4.3, DESIGN
        §6.1): ``priors.structural.implied_reduced_form_moments`` of the fit's prior arrays at the
        composition's scaling mode with ``E[λ_i]`` from :meth:`prior_vol_mean`. ``None`` unless
        the mean is structural (the reduced-form coefficient prior is already on ``B``, §6.2) and
        the coefficient prior fixed: under a global-local prior they are not applicable (P3 D-62:
        the horseshoe has no finite prior variance)."""
        if self.static.mean.kind != "linear_structural" or self.static.coef.varies:
            return None
        from rai_bvar.priors.structural import implied_reduced_form_moments

        p = self.prior
        mode = scaling_mode(self.static.struct, self.static.mean)
        return implied_reduced_form_moments(
            p.eq.coef, p.eq.contemp, p.shared, self.prior_vol_mean(), mode=mode
        )

    def _implied_summary(self) -> dict[str, Any] | str:
        if self.static.coef.varies:
            return (
                f"not applicable: the {self.static.coef.kind} global-local prior has no fixed "
                "prior variances (the horseshoe's are infinite; P3 D-62)"
            )
        if self.static.mean.kind != "linear_structural":
            return (
                "not applicable: the reduced-form coefficient prior is already a prior on B "
                "(DESIGN §6.2)"
                if self.static.mean.kind == "linear_reduced"
                else f"not applicable to mean {self.static.mean.kind!r}"
            )
        try:
            im = self.implied_prior_moments()
        except (ValueError, TypeError) as err:
            return f"not available: {err}"
        assert im is not None
        s = self.static.shapes
        regs = regressor_names(self.ordering, s.lags, s.n_det)
        var = list(self.ordering)

        def table(x: np.ndarray, rows: list[str], cols: list[str]) -> dict[str, dict[str, float]]:
            return {c: {r: float(x[a, b]) for a, r in enumerate(rows)} for b, c in enumerate(cols)}

        out = {
            "method": im.method,
            "vol_mean": self.vol_mean_label(),
            "mean": table(im.mean, regs, var),
            "var": table(im.var, regs, var),
            "sigma_mean": table(im.sigma_mean, var, var),
            "shares": table(im.shares, var, var),
        }
        if self.static.mean.n_dummy:  # WP25: the closed form covers the Minnesota part only
            out["dummies"] = (
                f"the {self.static.mean.n_dummy} SOC/SUR dummy rows are not included in these "
                "implied moments (Minnesota part of the prior only)"
            )
        return out

    def summary(self) -> pd.DataFrame:
        """Primary diagnostics (one row per gating quantity), DESIGN §8.4. ``attrs`` hold the
        prior form and mode, the hyperparameters in use (with the EB note), the implied
        reduced-form prior moments (R8, :meth:`implied_prior_moments`, as ``{column: {row:
        value}}`` tables of ``E[B]``, ``Var(B)`` (regressor × variable), ``E[Σ]`` and the shock
        shares, with ``vol_mean`` naming the ``λ_i`` used: under SV ``Σ`` is the plug-in at
        ``λ_i = exp(b_μ)`` (RW: ``exp(m₀)``), DESIGN Stage 2 amendment; a note when not
        applicable), the ordering and the convergence flag."""
        diag = self.diagnostics
        frame = diag
        if isinstance(diag, pd.DataFrame) and "primary" in diag.columns:
            frame = diag[diag["primary"].astype(bool)]
        frame = frame.copy()
        frame.attrs.update(
            {
                "engine": "mcmc",
                "prior_form": {"linear_structural": "structural", "linear_reduced": "reduced"}.get(
                    self.static.mean.kind, self.static.mean.kind
                ),
                "prior_mode": {"I": "Mode I", "II": "Mode II", "iw_equivalent": "IW-equivalent"}[
                    scaling_mode(self.static.struct, self.static.mean)
                ],
                "hyper": {k: v for k, v in self.hyper.items() if np.ndim(v) == 0},
                "eb_note": self.hyper.get("eb_note"),
                "notes": list(self.notes),
                "implied_reduced_form": self._implied_summary(),
                "ordering": list(self.ordering),
                "converged": self.converged,
            }
        )
        return frame

    # ------------------------------------------------------------------ delegation (WP17)
    def forecast(
        self, horizon: int, draws: int = 1000, seed: SeedLike = None, *, lag_coefs: bool = True
    ) -> ForecastResult:
        """Predictive paths and conditional moments (DESIGN §7); ``engines.mcmc.forecast``.

        MCMC forecasts always attach the accelerator variable ``lag_coefs`` (P2 D-34); the
        keyword-only ``lag_coefs`` is accepted for symmetry with the analytic forecasts'
        ``forecast(..., lag_coefs=True)`` protocol (P3 D-49, E162(a)) and changes nothing."""
        if not isinstance(lag_coefs, bool):
            raise TypeError(f"lag_coefs must be a bool, got {lag_coefs!r}")
        return _forecast_fn("forecast")(self, horizon, draws, seed)

    def irf(
        self,
        horizon: int = 20,
        draws: int = 1000,
        seed: SeedLike = None,
        shock: str = "sd",
        date: pd.Period | str | None = None,
        which: str = "persistent",
    ) -> Any:
        """Recursive impulse responses (DESIGN §7.4); ``engines.mcmc.forecast.irf``."""
        return _forecast_fn("irf")(
            self, horizon=horizon, draws=draws, seed=seed, shock=shock, date=date, which=which
        )

    def draw_parameters(
        self, size: int, rng: SeedLike = None, date: Any = None, which: str = "persistent"
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(coefs (S, k, n), chol (S, n, n))`` (DESIGN §7.4); ``engines.mcmc.forecast``."""
        return _forecast_fn("draw_parameters")(self, size, rng, date=date, which=which)

    def __repr__(self) -> str:
        c = self._run.chains
        return (
            f"MCMCFit({self.static.mean.kind} × {self.static.vol.persistent}"
            f"{'' if self.static.vol.transitory == 'none' else '+' + self.static.vol.transitory}, "
            f"chains={c}, draws={self._run.draws}, converged={self.converged})"
        )


def fit_mcmc(
    static: StaticConfig,
    data: EngineData,
    prior: PriorArrays,
    settings: MCMCSettings,
    runtime: Runtime,
    *,
    seed: int,
    salt: int,
    meta: FitMeta,
    init: ChainState | None = None,
    plan: FitPlan | None = None,
    _overrides: Mapping[str, Any] | None = None,
) -> MCMCFit:
    """Fit one window with the MCMC engine (DESIGN §11.3).

    ``init`` is a warm-start state (``warmstart.remap`` of a previous ``fit.final_state``; leading
    chain axis, at least ``settings.chains`` chains, the first ones are used); it replaces the cold
    initialisation and ``settings.resolved_warm_burn`` replaces ``settings.burn``. ``plan`` (a
    ``FitPlan``; keyword addition) requests in-loop forecasts and needs
    ``static.store.in_loop_forecast = plan.horizon`` (DESIGN §5.8; ``fit.in_loop_forecast``).
    ``_overrides`` is the test seam of DESIGN §4.9 (kernel slots or internals); production code
    never passes it. ``settings.draws=None`` (the composition default) is resolved from
    ``static.vol`` as ``BVAR`` resolves it (D-35).
    """
    if getattr(settings, "draws", 0) is None:  # CommonSV counts as SV (P3 D-51)
        settings = settings.with_default_draws(
            sv=static.vol.persistent == "sv" or static.vol.common
        )
    with _jax.x64():
        fns = make_runner(static, overrides=_overrides)
        if init is not None:
            C = settings.chains
            have = int(np.shape(init.status.code)[0])
            if have < C:
                raise ValueError(f"init has {have} chains, the settings ask for {C}")
            init = jax.tree.map(lambda x: np.asarray(x)[:C], init)
        run = run_chains(
            fns, static, data, prior, settings, runtime, seed=seed, salt=salt, init_state=init,
            plan=plan,
        )  # fmt: skip
        return MCMCFit(
            static=static, run=run, program=fns.program, data=data, prior=prior,
            settings=settings, meta=meta,
        )  # fmt: skip
