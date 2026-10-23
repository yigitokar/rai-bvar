"""Production MCMC fits for the forecast tests (DESIGN §13 WP17), without ``resolve.py`` (WP18).

:func:`build_fit` is a minimal local double of ``resolve.fit_bvar``: it assembles the engine
inputs of a ``BVAR`` with ``hierarchy="fixed"`` from the frozen builders (WP5
``engine_data_from_vardata``, WP6 prior arrays, ``api.static_parts``) and calls
``results.fit_mcmc`` with the production kernels. ``const_path`` defaults to ``"weighted"``
(legal for every composition).
"""

from __future__ import annotations

import warnings

from rai_bvar import BVAR, SV, Constant, Linear, Minnesota
from rai_bvar.api import static_parts
from rai_bvar.data import VARData
from rai_bvar.data.design import n_deterministic
from rai_bvar.engines.mcmc.config import MeanStatic, Shapes, StaticConfig, StoreStatic, bucket
from rai_bvar.engines.mcmc.data import engine_data_from_vardata
from rai_bvar.engines.mcmc.keys import data_salt
from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
from rai_bvar.engines.mcmc.state import Empty, EqPrior, PriorArrays
from rai_bvar.priors.reduced import reduced_arrays
from rai_bvar.priors.scales import window_scales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.priors.sv_arrays import constant_prior, sv_prior_arrays, transitory_arrays
from rai_bvar.settings import FitPlan, MCMCSettings, Runtime

SETTINGS = MCMCSettings(chains=2, draws=60, burn=60)


def model(form: str = "structural", vol=None, lags: int = 1, deterministic: str = "c", **mcmc):
    """A ``BVAR`` with a fixed-hyperparameter Minnesota prior on the MCMC engine."""
    settings = MCMCSettings(**{**SETTINGS.__dict__, **mcmc})
    return BVAR(
        mean=Linear(prior=Minnesota(form=form, hierarchy="fixed")),
        vol=SV() if vol is None else vol,
        lags=lags,
        deterministic=deterministic,
        mcmc=settings,
    )


def prior_arrays(m: BVAR, data: VARData, static: StaticConfig) -> PriorArrays:
    parts = static_parts(m)
    prior = m.mean.prior
    n, lags, n_det = data.n_vars, m.lags, n_deterministic(m.deterministic)
    scales = window_scales(data, prior, lags, m.deterministic)
    if parts.mean_kind == "linear_structural":
        mode = "II" if parts.coef_scaling == "lambda" else "I"
        coef, shared = structural_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det, mode=mode)
    else:
        coef, shared = reduced_arrays(scales, prior, {}, n=n, lags=lags, n_det=n_det)
    contemp = contemp_arrays(scales, prior, n, scaling=parts.struct.contemp_scaling)
    if isinstance(m.vol, Constant):
        core, trans = constant_prior(scales, n), Empty()
    else:
        core = sv_prior_arrays(m.vol, scales)
        trans = transitory_arrays(m.vol, scales, static.vol.nu_grid)
    eq = EqPrior(coef=coef, contemp=contemp, vol_core=core, transitory=trans)
    return PriorArrays(eq=eq, shared=shared)


def build_fit(
    m: BVAR,
    data: VARData,
    *,
    seed: int = 1,
    chunk: int = 50,
    plan: FitPlan | None = None,
    const_path: str = "weighted",
    quiet: bool = True,
):
    """``MCMCFit`` of ``m`` on ``data`` through ``fit_mcmc`` (store paths per ``m.mcmc.store``)."""
    settings = m.mcmc
    parts = static_parts(m)
    ed, periods = engine_data_from_vardata(data, m.lags, m.deterministic, settings.t_bucket)
    shapes = Shapes.of(
        data.n_vars, m.lags, n_deterministic(m.deterministic),
        bucket(int(ed.T_eff), settings.t_bucket),
    )  # fmt: skip
    static = StaticConfig(
        shapes=shapes,
        struct=parts.struct,
        mean=MeanStatic(
            kind=parts.mean_kind, coef_scaling=parts.coef_scaling, const_path=const_path
        ),
        coef=parts.coef,
        vol=parts.vol,
        store=StoreStatic(
            paths=settings.store == "paths",
            in_loop_forecast=0 if plan is None else plan.horizon,
        ),
        chunk=chunk,
    )
    meta = FitMeta(model=m, vardata=data, periods=periods, variables=data.variables)
    with warnings.catch_warnings():
        if quiet:  # short chains: convergence warnings are expected
            warnings.simplefilter("ignore", UserWarning)
        return fit_mcmc(
            static, ed, prior_arrays(m, data, static), settings, Runtime(devices=1, chunk=chunk),
            seed=seed, salt=data_salt(data.fingerprint()), meta=meta, plan=plan,
        )  # fmt: skip


_CACHE: dict = {}


def cached_fit(m: BVAR, data: VARData, **kw):
    """:func:`build_fit`, memoised per process (fits are shared read-only across test modules;
    a test that changes a fit works on ``copy.copy(fit)``)."""
    key = (repr(m.spec()), data.fingerprint(), repr(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = build_fit(m, data, **kw)
    return _CACHE[key]


def shared_model() -> BVAR:
    """The fast-tier composition: structural Mode I × SV with outliers, latent paths stored."""
    return model("structural", SV(outliers=True), store="paths", path_thin=2)


def shared_fit(data: VARData):
    """The fast-tier fit of :func:`shared_model` (one compile per process)."""
    return cached_fit(shared_model(), data)


COMPOSITIONS = {
    "structural-SV": ("structural", SV()),
    "structural-SV(outliers)": ("structural", SV(outliers=True)),
    "structural-SV(t)": ("structural", SV(tails="t")),
    "structural-SV(outliers,t)": ("structural", SV(outliers=True, tails="t")),
    "structural-SV(rw)": ("structural", SV(rw=True)),
    "structural-Constant(ig)": ("structural", Constant("ig")),
    "structural-Constant(iw)": ("structural", Constant("iw")),
    "structural-Constant(iw_equivalent)": ("structural", Constant("iw_equivalent")),
    "reduced-SV": ("reduced", SV()),
    "reduced-SV(outliers)": ("reduced", SV(outliers=True)),
    "reduced-SV(rw)": ("reduced", SV(rw=True)),  # the Clark–Ravazzolo primary SV arm
    "reduced-Constant(ig)": ("reduced", Constant("ig")),
    "reduced-Constant(iw_equivalent)": ("reduced", Constant("iw_equivalent")),
}
"""Every Phase 2 MCMC composition kind of DESIGN §11.1 (rows 3–8) and the SV variants."""
