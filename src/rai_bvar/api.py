"""``BVAR``: the model façade. A model is a composition of components (SPEC §3 R1, §4).

The engine is chosen by the composition, not by the user (SPEC §4.5). ``BVAR.__post_init__``
(DESIGN §11.1; D-19, D-21) first resolves the ``auto`` defaults,

1. ``Minnesota(form="auto")`` → ``"conjugate"`` iff ``vol`` is ``Constant`` with prior ``"auto"``
   or ``"iw"``, else ``"structural"``;
2. ``Constant(prior="auto")`` → ``"iw"`` iff the resolved form is ``"conjugate"``, else ``"ig"``;

and stores the resolved components, so ``BVAR()`` keeps its Phase 1 meaning, spec and hash. The
engine rules then apply (first match wins):

* ``conjugate`` × ``Constant("iw")``: analytic (Phase 1);
* ``conjugate`` × anything else: ``ValueError``;
* ``structural`` × ``Constant("ig")`` or ``SV(...)``: MCMC, Mode I (independent, moment-matched);
* ``structural`` × ``Constant("iw")``: MCMC, Mode II (λ-scaled NIG; the NIW twin at
  ``cross = contemp = 1``; the prior axis gains ``scaling=conjugate``);
* ``structural`` × ``Constant("iw_equivalent")``: MCMC, Mode I δ prior + Chan (2022) ``(a, λ)``;
* ``reduced`` × ``Constant("ig")`` or ``SV(...)``: MCMC, CCCM (2022);
* ``reduced`` × ``Constant("iw_equivalent")``: MCMC, CCCM + Chan ``(a, λ)`` (order-invariant);
* ``reduced`` × ``Constant("iw")``: ``ValueError``;

and finally the MCMC-only rules (Lenza–Primiceri scaling and ``hierarchy="mh"`` arrive in Phase 3;
ψ cannot be estimated by EB with ``cross ≠ 1`` in structural form) and the JAX-free legality check
``engines.mcmc.config.validate_composition`` (DESIGN §3.8). Every error is raised at construction.
SOC/SUR dummies run on both engines (MCMC: DESIGN §13 WP25; N05 §7.7, N04 §3.1).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from rai_bvar.components import SV, Constant, Linear
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import n_deterministic
from rai_bvar.data.outliers import LenzaPrimiceri
from rai_bvar.engines.analytic import AnalyticFit
from rai_bvar.priors.hyperpriors import (
    BetaPrior,
    GammaPrior,
    GLPHyperpriors,
    InvGammaPrior,
    ParetoPrior,
)
from rai_bvar.priors.minnesota import MHSettings, Minnesota, _short
from rai_bvar.priors.sv import InvGamma
from rai_bvar.settings import FitPlan, MCMCSettings, Runtime

if TYPE_CHECKING:
    from rai_bvar.engines.mcmc.config import (
        CoefPriorStatic,
        StructStatic,
        VolStatic,
    )

logger = logging.getLogger(__name__)

_CONJUGATE_VOL = ("auto", "iw")  # Constant priors under which form="auto" means "conjugate"
_RESOLVE_MODULE = "rai_bvar.engines.mcmc.resolve"  # WP18: fit_bvar(model, data, *, seed, init, …)


@dataclass(frozen=True)
class BVAR:
    """``BVAR(mean, vol, lags, deterministic, outliers, mcmc)``: a model specification.

    ``mean`` and ``vol`` are stored resolved (no ``auto`` survives construction). ``mcmc`` holds
    the sampler settings of MCMC compositions (``None`` → ``MCMCSettings()``, a mapping is
    accepted) and is part of their spec; on an analytic composition it is normalised to ``None``
    so the Phase 1 spec is unchanged.
    """

    mean: Linear = field(default_factory=Linear)
    vol: Constant | SV = field(default_factory=Constant)
    lags: int = 1
    deterministic: str = "c"
    outliers: LenzaPrimiceri | str | None = None
    mcmc: MCMCSettings | None = None

    requires_in_loop_forecast: ClassVar[bool] = False  # no Phase 2 model needs a FitPlan

    def __post_init__(self) -> None:
        if (
            isinstance(self.lags, bool)
            or not isinstance(self.lags, int | np.integer)
            or self.lags < 1
        ):
            raise ValueError("lags must be a positive integer")
        object.__setattr__(self, "lags", int(self.lags))
        n_deterministic(self.deterministic)
        if isinstance(self.outliers, str):
            if self.outliers != "lenza-primiceri":
                raise ValueError("outliers must be None, 'lenza-primiceri' or a LenzaPrimiceri()")
            object.__setattr__(self, "outliers", LenzaPrimiceri())
        elif self.outliers is not None and not isinstance(self.outliers, LenzaPrimiceri):
            raise TypeError("outliers must be None, 'lenza-primiceri' or a LenzaPrimiceri()")
        self._resolve_auto()
        settings = _mcmc_settings(self.mcmc)
        prior = self.mean.prior
        if self.engine == "analytic":
            object.__setattr__(self, "mcmc", None)  # unused by the analytic engine (spec unchanged)
            if (
                isinstance(self.outliers, LenzaPrimiceri)
                and self.outliers.fixed is None
                and prior.hierarchy == "fixed"
            ):
                raise ValueError(
                    "Lenza-Primiceri scaling needs fixed=(s0, s1, s2, rho) with hierarchy='fixed'"
                )
            return
        object.__setattr__(self, "mcmc", settings or MCMCSettings())
        self._check_mcmc_rules()
        from rai_bvar.engines.mcmc.config import MeanStatic, validate_composition

        parts = static_parts(self)
        validate_composition(
            parts.struct,
            MeanStatic(kind=parts.mean_kind, coef_scaling=parts.coef_scaling),
            parts.coef,
            parts.vol,
        )

    def _resolve_auto(self) -> None:
        """Resolve ``form="auto"`` and ``Constant("auto")``, then apply the engine rules."""
        if not isinstance(self.mean, Linear) or not isinstance(self.vol, Constant | SV):
            raise NotImplementedError(
                "only Linear × Constant and Linear × SV compositions exist so far (Phase 2)"
            )
        prior = self.mean.prior
        if not isinstance(prior, Minnesota):
            raise TypeError("Linear(prior=...) must be a Minnesota prior")
        vol = self.vol
        if prior.form == "auto":
            conjugate = isinstance(vol, Constant) and vol.prior in _CONJUGATE_VOL
            prior = replace(prior, form="conjugate" if conjugate else "structural")
            object.__setattr__(self, "mean", replace(self.mean, prior=prior))
        if isinstance(vol, Constant) and vol.prior == "auto":
            vol = replace(vol, prior="iw" if prior.form == "conjugate" else "ig")
            object.__setattr__(self, "vol", vol)
        if prior.form == "conjugate" and not (isinstance(vol, Constant) and vol.prior == "iw"):
            raise ValueError(
                "the conjugate Minnesota prior requires Constant(prior='iw'); use "
                "form='structural' (the MCMC default) or 'reduced'"
            )
        if prior.form == "reduced" and isinstance(vol, Constant) and vol.prior == "iw":
            raise ValueError(
                "Constant('iw') is the conjugate NIW prior; with form='reduced' use "
                "'iw_equivalent' (Chan 2022) or 'ig'"
            )

    def _check_mcmc_rules(self) -> None:
        """Compositions the Phase 2 MCMC engine rejects (DESIGN §11.1, D-21, §6.5)."""
        prior = self.mean.prior
        if self.outliers is not None:
            double = (
                " (and SV(outliers=True) already models outliers: a double treatment)"
                if isinstance(self.vol, SV) and self.vol.outliers
                else ""
            )
            raise NotImplementedError(
                "Lenza-Primiceri scaling runs on the analytic engine only; on the MCMC engine it "
                f"arrives in Phase 3{double}. Use SV(outliers=True) for outlier-robust MCMC models"
            )
        if prior.hierarchy == "mh":
            raise NotImplementedError(
                "hierarchical Minnesota inside MCMC arrives with the Phase 3 global-scale step; "
                "use 'eb' or 'fixed'"
            )
        if (
            prior.form == "structural"
            and prior.cross != 1.0
            and prior.hierarchy == "eb"
            and "psi" in prior.free
        ):
            raise ValueError(
                "empirical Bayes over psi is not available for the structural Minnesota prior "
                "with cross != 1 (Chan 2022 NIG marginal likelihood over lambda and alpha only); "
                "remove 'psi' from hyper or fix psi"
            )

    # ------------------------------------------------------------------ identity

    @property
    def engine(self) -> str:
        """``"analytic"`` for the conjugate form (always with ``Constant("iw")``), else ``"mcmc"``."""
        return "analytic" if self.mean.prior.form == "conjugate" else "mcmc"

    @property
    def uses_fit_seed(self) -> bool:
        """Whether the backtest passes a per-origin fit seed (``fit(seed=…)``, DESIGN §11.6)."""
        return self.engine == "mcmc"

    @property
    def supports_warm_start(self) -> bool:
        """Whether ``fit(init=…)`` accepts the previous origin's final state (DESIGN §5.5)."""
        return self.engine == "mcmc"

    @property
    def name(self) -> str:
        extra = "" if self.deterministic == "c" else f", det={self.deterministic}"
        if self.outliers is not None:
            label = _outlier_label(self.outliers).removeprefix("lenza-primiceri")
            extra += ", COVID" + (f" {label}" if label else "")
        if self.engine == "analytic":
            return f"BVAR({self.mean.prior.label}, p={self.lags}{extra})"
        if self.mcmc != MCMCSettings():
            extra += f", mcmc={_short(self.mcmc)}"
        return f"BVAR({self.mean.prior.label}, {_vol_label(self.vol)}, p={self.lags}{extra})"

    @property
    def axes(self) -> dict[str, str]:
        prior = self.mean.prior.label
        if _mode_ii(self):
            prior += ", scaling=conjugate"  # Mode II changes the mean prior (R2)
        return {
            "mean": self.mean.name,
            "prior": prior,
            "vol": self.vol.name,
            "lags": str(self.lags),
            "deterministic": self.deterministic,
            "outliers": "none" if self.outliers is None else _outlier_label(self.outliers),
        }

    def spec(self) -> dict[str, Any]:
        """Plain-JSON description (DESIGN §11.5). Phase 1 compositions keep their Phase 1 spec:
        ``Minnesota.contemp`` is omitted at its default and ``mcmc`` appears only on MCMC models."""
        prior = asdict(self.mean.prior)
        if prior["contemp"] == 1.0:
            del prior["contemp"]
        spec: dict[str, Any] = {
            "class": "BVAR",
            "mean": {"prior": prior},
            "vol": self.vol.to_spec() if isinstance(self.vol, SV) else asdict(self.vol),
            "lags": self.lags,
            "deterministic": self.deterministic,
            "outliers": None if self.outliers is None else asdict(self.outliers),
        }
        if self.engine == "mcmc":
            spec["mcmc"] = asdict(self.mcmc)
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> BVAR:
        """Rebuild a model from :meth:`spec` (the inverse of ``spec()``)."""
        spec = {k: v for k, v in spec.items() if k != "class"}
        prior = dict(spec["mean"]["prior"])
        hp = prior.get("hyperpriors", {})
        prior["hyperpriors"] = GLPHyperpriors(
            **{k: (InvGammaPrior if k == "psi" else GammaPrior)(**v) for k, v in hp.items()}
        )
        prior["mh"] = MHSettings(**prior.get("mh", {}))
        for key in ("psi", "hyper"):
            if isinstance(prior.get(key), list):
                prior[key] = tuple(prior[key])
        outliers = spec.get("outliers")
        if isinstance(outliers, dict):
            outliers = LenzaPrimiceri(
                start=outliers["start"],
                fixed=None if outliers["fixed"] is None else tuple(outliers["fixed"]),
                scale_prior=ParetoPrior(**outliers["scale_prior"]),
                rho_prior=BetaPrior(**outliers["rho_prior"]),
            )
        vol_spec = dict(spec.get("vol", {}))
        vol_class = vol_spec.pop("class", "Constant")
        if vol_class == "SV":
            vol = SV.from_spec(vol_spec)
        elif vol_class == "Constant":
            vol = Constant(**vol_spec)
        else:
            raise ValueError(f"unknown volatility class {vol_class!r} in spec")
        mcmc = spec.get("mcmc")
        return cls(
            mean=Linear(prior=Minnesota(**prior)),
            vol=vol,
            lags=spec["lags"],
            deterministic=spec["deterministic"],
            outliers=outliers,
            mcmc=None if mcmc is None else MCMCSettings(**mcmc),
        )

    # ------------------------------------------------------------------ grids

    @classmethod
    def harmonise_grid(cls, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve ``auto`` components the same way in every arm of a grid (DESIGN §11.4, D-19).

        If any combination runs on the MCMC engine because of its volatility (a ``vol`` that is
        not ``Constant`` with prior ``"auto"``/``"iw"``), every arm whose Minnesota prior has
        ``form="auto"`` gets ``form="structural"`` (a missing ``mean`` counts as the default
        ``Linear()``). Then ``Constant(prior="auto")`` becomes ``"ig"`` in every arm whose form
        is ``"structural"`` or ``"reduced"`` (what ``BVAR`` resolves anyway, so the grid entry is
        labelled ``Constant(ig)`` and named apart from a Mode II ``Constant("iw")`` arm). The
        SPEC §9.5 grid ``vol=[Constant(), SV(), SV(outliers=True)]`` then compares one axis
        (structural Mode I in every arm); a grid of ``Constant()`` arms stays analytic. Explicit
        settings are never changed; the rewritten arms are listed in an INFO note.
        """
        out = [dict(c) for c in combos]
        mcmc_grid = not all(_conjugate_vol(c.get("vol", Constant())) for c in out)
        notes = []
        for i, c in enumerate(out):
            rewrites = []
            mean = c.get("mean", Linear())
            prior = getattr(mean, "prior", None)
            if not isinstance(prior, Minnesota):
                continue  # BVAR rejects it at construction
            if mcmc_grid and prior.form == "auto":
                prior = replace(prior, form="structural")
                c["mean"] = replace(mean, prior=prior)
                rewrites.append("Minnesota(form='auto' → 'structural')")
            vol = c.get("vol", Constant())
            if (
                isinstance(vol, Constant)
                and vol.prior == "auto"
                and prior.form in ("structural", "reduced")
            ):
                c["vol"] = replace(vol, prior="ig")
                rewrites.append("Constant('auto' → 'ig')")
            if rewrites:
                notes.append(f"arm {i}: {', '.join(rewrites)}")
        if notes:
            logger.info(
                "grid(): %d of %d arms resolved 'auto' components so every arm resolves alike "
                "(DESIGN §11.4): %s",
                len(notes),
                len(out),
                "; ".join(notes),
            )
        return out

    # ------------------------------------------------------------------ estimation

    def fit(
        self,
        data: VARData,
        *,
        seed: int | None = None,
        init: Any = None,
        runtime: Runtime | None = None,
        plan: FitPlan | None = None,
    ) -> Any:
        """Estimate on ``data`` (DESIGN §11.3).

        Analytic compositions: ``seed`` overrides the MH seed of ``hierarchy="mh"`` (recorded in
        forecast attrs as ``mh_seed``); ``init``, ``runtime`` and ``plan`` must be ``None``.
        MCMC compositions: ``seed`` overrides ``MCMCSettings.seed``; ``init`` (a ``FinalState`` or
        an ``MCMCFit``) warm-starts the chains; ``runtime`` defaults to ``Runtime()``; ``plan``
        requests in-loop forecasts. Returns an ``AnalyticFit`` or an ``MCMCFit``.
        """
        if self.engine == "analytic":
            given = [
                k
                for k, v in (("init", init), ("runtime", runtime), ("plan", plan))
                if v is not None
            ]
            if given:
                raise ValueError(f"{given} apply to MCMC compositions only; this model is analytic")
            return AnalyticFit(self, data, seed)
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int | np.integer)
            or not 0 <= seed < 2**63
        ):
            raise TypeError(f"seed must be an int in [0, 2**63) or None, got {seed!r}")
        if runtime is not None and not isinstance(runtime, Runtime):
            raise TypeError(f"runtime must be a Runtime or None, got {runtime!r}")
        if plan is not None and not isinstance(plan, FitPlan):
            raise TypeError(f"plan must be a FitPlan or None, got {plan!r}")
        fit_bvar = _mcmc_entry_point()
        return fit_bvar(
            self,
            data,
            seed=None if seed is None else int(seed),
            init=init,
            runtime=runtime if runtime is not None else Runtime(),
            plan=plan,
        )


class StaticParts(NamedTuple):
    """The data-independent part of an MCMC ``StaticConfig`` (DESIGN §2.4); ``resolve.py`` adds
    shapes, the resolved ``gram``/``const_path`` and storage (WP18)."""

    struct: StructStatic
    mean_kind: str  # "linear_structural" | "linear_reduced"
    coef_scaling: str  # "lambda" in Mode II only
    coef: CoefPriorStatic
    vol: VolStatic


def static_parts(model: BVAR) -> StaticParts:
    """Static choices of an MCMC composition (JAX-free; DESIGN §2.4, §4.7, §6.1).

    Mode II (``structural`` × ``Constant("iw")``) λ-scales both ``a_i`` and ``δ_i``;
    ``Constant("iw_equivalent")`` λ-scales ``a_i`` only; both couple the Constant core.
    """
    from rai_bvar.engines.mcmc.config import CoefPriorStatic, StructStatic, VolStatic

    if model.engine != "mcmc":
        raise ValueError(f"{model.name} runs on the analytic engine: it has no MCMC static parts")
    prior, vol = model.mean.prior, model.vol
    coupled = isinstance(vol, Constant) and vol.prior in ("iw", "iw_equivalent")
    struct = StructStatic(kind="recursive", contemp_scaling="lambda" if coupled else "none")
    mean_kind = {"structural": "linear_structural", "reduced": "linear_reduced"}[prior.form]
    coef_scaling = "lambda" if _mode_ii(model) else "none"
    if isinstance(vol, Constant):
        vstatic = VolStatic(persistent="constant", coupled=coupled)
    else:
        transitory = {
            (False, False): "none",
            (True, False): "outlier",
            (False, True): "t",
            (True, True): "outlier_t",
        }[(vol.outliers, vol.tails == "t")]
        vstatic = VolStatic(
            persistent="sv",
            rw=vol.rw,
            h0="stationary" if vol.rw or vol.h0 == "stationary" else "fixed_b0",
            sigma2_prior=(
                "inv_gamma" if isinstance(vol.resolved_sigma2_prior(), InvGamma) else "scaled_chi2"
            ),
            transitory=transitory,
            outlier_K=vol.outlier_max,
            nu_grid=vol.nu_prior.support,
            t_scale=vol.t_scale,
            mixture=vol.mixture,
            exact=vol.exact,
            asis=vol.asis,
            zero_offset=vol.offset == 0.0,
            coupled=False,
        )
    return StaticParts(struct, mean_kind, coef_scaling, CoefPriorStatic(), vstatic)


def _mode_ii(model: BVAR) -> bool:
    """Structural Minnesota × ``Constant("iw")``: the λ-scaled NIG prior (D-10)."""
    vol = model.vol
    return model.mean.prior.form == "structural" and isinstance(vol, Constant) and vol.prior == "iw"


def _conjugate_vol(vol: Any) -> bool:
    return isinstance(vol, Constant) and vol.prior in _CONJUGATE_VOL


def _vol_label(vol: Constant | SV) -> str:
    return f"Constant({vol.prior})" if isinstance(vol, Constant) else vol.label


def _mcmc_settings(value: Any) -> MCMCSettings | None:
    if value is None or isinstance(value, MCMCSettings):
        return value
    if isinstance(value, Mapping):
        return MCMCSettings(**value)
    raise TypeError(f"mcmc must be an MCMCSettings, a mapping or None, got {value!r}")


def _mcmc_entry_point():
    """``engines.mcmc.resolve.fit_bvar`` (WP18), imported lazily so ``import rai_bvar`` loads no
    JAX (D-31). Raises ``NotImplementedError`` while the engine is not part of the build; any
    other import error (a broken engine module) propagates."""
    try:
        module = importlib.import_module(_RESOLVE_MODULE)
    except ModuleNotFoundError as err:
        missing = err.name or ""
        if not (_RESOLVE_MODULE == missing or _RESOLVE_MODULE.startswith(missing + ".")):
            raise
        module = None
    fit_bvar = getattr(module, "fit_bvar", None)
    if fit_bvar is None:
        raise NotImplementedError(
            f"the MCMC engine is not yet available in this build ({_RESOLVE_MODULE}.fit_bvar "
            "is missing); only analytic compositions (Minnesota(form='conjugate') × "
            "Constant('iw')) can be fitted"
        )
    return fit_bvar


def _outlier_label(outliers: LenzaPrimiceri) -> str:
    default = LenzaPrimiceri()
    changed = [
        f.name
        for f in fields(LenzaPrimiceri)
        if getattr(outliers, f.name) != getattr(default, f.name)
    ]
    return "lenza-primiceri" + (
        f"({','.join(f'{c}={getattr(outliers, c)}' for c in changed)})" if changed else ""
    )
