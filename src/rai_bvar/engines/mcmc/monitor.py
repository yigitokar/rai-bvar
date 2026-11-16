"""Monitored quantities, the ``converged`` flag, warnings and the diagnostics table (DESIGN §8).

Monitored sets (D-33; DESIGN §8.1; fixes E3, E16). The **primary** set decides ``converged``:

    loglik = Σ_i Σ_t m_t log N(e_it; 0, λ_eff,it) (+ log|det A|·T_eff, 0 for unit-triangular A)
    B_own1[i] = B_{(1,i),i}, the own first-lag reduced-form coefficient, B = (A⁻¹Γ)' or Π
    Constant: log λ_i;   AR SV: μ_i, φ_i, σ_i, h_{i,T};   RW SV: σ_i, h_{i,T}

(41 quantities at ``n = 20`` with Constant, 101 with SV). The **secondary** set is reported and
never gates: every active ``a_ij``; ``h_{i,1}``, ``h_{i,⌈T/2⌉}``; the reduced-form intercepts
``B_{0,i}``; ``log p_i``; ``ν_i`` (with its escape probability in the table); the log prior.

Phase 3 (P3 D-52, §3.7; global-local and SSVS priors): ``group`` fields (``(G,)``, one value per
coefficient group): the group log scales (HS ``gl_log_zeta``, NG/R2D2 ``gl_log_xi``) are primary
when ``G ≤ max(2p, n)`` (global, olcl-lagwise, equation-wise groups) and secondary otherwise; SSVS
``gl_p`` (the inclusion probability ``p_g``) is primary under a Beta hyperprior; the HS auxiliary
``gl_log_tau`` and the hyperprior index ``gl_a_idx`` (with a grid on ``a``) are secondary. Local
scales are latent: neither monitored nor stored (§3.7). Under a varying coefficient prior the log
prior reads the deviations' ``dev_log`` and the scales from the full chain state.

Phase 3 CommonSV (P3 D-52, §5.5): the common ``h`` at the last data row ``common_h_T``, its
persistence ``common_rho`` and scale ``common_sigma`` are primary scalars (with every ``log λ_i``
of the Constant cores, as for ``Constant``); ``common_h_0`` and ``common_h_mid`` (data rows ``0``
and ``⌈T_eff/2⌉ − 1``) are secondary. The log prior adds the step's ``log p(h, ρ, σ_h)``, which
needs the data mask (``CommonSVStep.log_prior_data``). The sampler's per-chain scalars
``acc_csv_h``, ``acc_csv_rho``, ``acc_csv_newton`` (E132) go to the table's attrs, never to
per-equation columns, and the warnings of P3 §8 name a low ``h`` block acceptance and any Newton
cap hit.

On device, :func:`monitor_values` returns one array per *field* (``"mu"``: one value per
equation, ``"a"``: the strict lower triangle in row-major order, ``"loglik"``: a scalar); the
runner stacks them to ``(chains, draws, …)``. On the host a :class:`MonitorSpec` names every scalar
quantity (``"mu[INDPRO]"``, ``"a[CPI,INDPRO]"``) and maps it to its field and position. With
outliers two per-equation *auxiliary* fields (:func:`auxiliary_fields`) ride along that are no
monitored quantity (no R-hat, never gate, no table row): ``outlier_share = Σ_t m_t 1{o_t > 1}/T_eff``
and ``outlier_at_max = Σ_t m_t 1{o_t = K}/T_eff``; their posterior means feed the per-equation
table and the "outlier scale at K" warning (DESIGN §8.3, §8.4).

The flag (DESIGN §8.2)::

    converged = no chain raised ∧ chains ≥ 2 ∧ (EB converged, when EB was used)
                ∧ ∀ q ∈ primary: R-hat(q) < rhat_max ∧ bulk ESS(q) > ess_min

with the rank-normalised split R-hat ``max(bulk, folded)`` and the bulk ESS of
``rai_bvar.diagnostics``. Tail ESS, acceptance rates and every secondary quantity are reported
only (E21). :func:`warning_catalogue` assembles every message of the DESIGN §8.3 table. R-hat and
bulk ESS are memoised by the content of each quantity's draws, so the flag, the table and the
warnings of one fit evaluate them once per quantity (:func:`rhat_ess`); a memoised value equals
a recomputation to rounding, being that of the batch in which the quantity was first evaluated.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax

from rai_bvar import diagnostics as dg
from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)
from rai_bvar.engines.mcmc.config import CoefPriorStatic, StaticConfig, VolStatic
from rai_bvar.engines.mcmc.state import (
    ChainState,
    EngineData,
    FitCache,
    GaussianDiag,
    PriorArrays,
)
from rai_bvar.settings import SV_DEFAULT_DRAWS, MCMCSettings

Array = jax.Array
Program = Any  # compose.Program (WP14): structure, mean, coef, vol, global_pre, global_post, eq_map

_LOG_2PI = float(np.log(2.0 * np.pi))
_LINEAR_MEANS = ("linear_structural", "linear_reduced")
_WORST_PRIMARY = 5  # quantities named in the primary-failure warning (DESIGN §8.3)
_WORST_SECONDARY = 3
_SECONDARY_A_FRACTION = 0.05  # more a_ij than this with R-hat ≥ rhat_max: informational
_H_ACCEPT_MIN = 0.5  # exact-h acceptance below this, or ...
_REJECTION_RUN_MAX = 50  # ... a longer run of rejections (sweeps), any equation: warning
_FOLD_MAX = 0.01  # NC sign-fold fraction (D-7)
_NU_BOUND_MASS = 0.5  # ν posterior mass at the grid's upper bound
_OUTLIER_AT_K_MAX = 0.01  # fraction of rows whose outlier scale sits at K
_ORDERING_N = 4  # R8 note from this many variables on
_CHAIN_SCALARS = (  # per-chain sampler scalars, not per equation (P3 E132: the CommonSV rates)
    "min_pivot",
    "csv_h",
    "csv_rho",
    "csv_newton",
)
_CSV_H_ACCEPT_MIN = 0.1  # CommonSV h block acceptance below this: "decrease CommonSV(block=…)"
_AUX_OUTLIER = ("outlier_share", "outlier_at_max")  # auxiliary per-equation monitor fields
_STATS_CACHE_SIZE = 1 << 14  # memoised (R-hat, bulk ESS) pairs, keyed by content (≈ 3 MB)


# --------------------------------------------------------------------------- monitored sets


class MonitorField(NamedTuple):
    """One device-side monitor array: ``kind`` ``"scalar"`` (shape ``()``), ``"equation"``
    (``(n,)``) or ``"pair"`` (``(n(n−1)/2,)``, the active ``a_ij`` in row-major order)."""

    name: str
    kind: Literal["scalar", "equation", "pair", "group"]
    primary: bool


def gl_fields(coef: CoefPriorStatic | None, n: int, lags: int) -> tuple[MonitorField, ...]:
    """The ``group`` fields of a global-local or SSVS coefficient prior (P3 D-52; module
    docstring); ``()`` for a fixed prior."""
    if coef is None or not coef.varies:
        return ()
    primary = coef.n_groups <= max(2 * lags, n)
    out: list[MonitorField] = []
    if coef.kind == "horseshoe":
        out += [MonitorField("gl_log_zeta", "group", primary)]
        out += [MonitorField("gl_log_tau", "group", False)]
    elif coef.kind in ("normal_gamma", "r2d2"):
        out.append(MonitorField("gl_log_xi", "group", primary))
    elif coef.kind == "ssvs" and coef.p_beta:
        out.append(MonitorField("gl_p", "group", True))
    if coef.a_grid:
        out.append(MonitorField("gl_a_idx", "group", False))
    return tuple(out)


def monitor_fields(
    mean_kind: str,
    vol: VolStatic,
    n: int,
    n_det: int,
    *,
    coef: CoefPriorStatic | None = None,
    lags: int = 1,
) -> tuple[MonitorField, ...]:
    """The monitor fields of a composition, primary fields first (DESIGN §8.1). Shared by
    :func:`monitor_spec` (host) and :func:`monitor_values` (device), so the two always agree.
    ``coef`` (the coefficient-prior static) and ``lags`` add the ``group`` fields of a
    global-local prior (:func:`gl_fields`, P3 D-52) before the log prior."""
    linear = mean_kind in _LINEAR_MEANS
    out = [MonitorField("loglik", "scalar", True)]
    if linear:
        out.append(MonitorField("B_own1", "equation", True))
    if vol.persistent == "constant":
        out.append(MonitorField("log_lam", "equation", True))
    else:
        core = ("sigma", "h_T") if vol.rw else ("mu", "phi", "sigma", "h_T")
        out.extend(MonitorField(name, "equation", True) for name in core)
    if vol.common:  # CommonSV (P3 D-52): h_T, ρ, σ_h primary
        out.extend(MonitorField(name, "scalar", True) for name in _CSV_PRIMARY)
    if n >= 2:
        out.append(MonitorField("a", "pair", False))
    if vol.persistent == "sv":
        out.extend(MonitorField(name, "equation", False) for name in ("h_1", "h_mid"))
    if vol.common:  # secondary: h at the first and the middle data row
        out.extend(MonitorField(name, "scalar", False) for name in _CSV_SECONDARY)
    if linear and n_det >= 1:  # the constant is the first deterministic column (data/design.py)
        out.append(MonitorField("intercept", "equation", False))
    if vol.has_outliers:
        out.append(MonitorField("log_p", "equation", False))
    if vol.has_t:
        out.append(MonitorField("nu", "equation", False))
    out.extend(gl_fields(coef, n, lags))
    out.append(MonitorField("log_prior", "scalar", False))
    return tuple(out)


_CSV_PRIMARY = ("common_h_T", "common_rho", "common_sigma")
_CSV_SECONDARY = ("common_h_0", "common_h_mid")


def auxiliary_fields(vol: VolStatic) -> tuple[str, ...]:
    """The auxiliary per-equation (``(n,)``) fields :func:`monitor_values` adds to the
    :func:`monitor_fields`: with outliers, the share of real rows with ``o_t > 1`` and with
    ``o_t = K`` in the draw. They carry no label, so they never gate and get no table row."""
    return _AUX_OUTLIER if vol.has_outliers else ()


@dataclass(frozen=True)
class MonitorSlot:
    """Where a named scalar quantity lives: ``mon[field]`` (``index = −1``) or
    ``mon[field][..., index]``; ``variable`` is its equation (``"y_i,y_j"`` for ``a_ij``)."""

    label: str
    field: str
    index: int = -1
    variable: str = ""


@dataclass(frozen=True)
class MonitorSpec:
    """Names of the primary (gating) and secondary (reported) monitored quantities.

    ``slots`` maps labels to monitor fields (:func:`monitor_spec` fills it); a label without a
    slot names a monitor array of shape ``(chains, draws)`` itself, so test doubles can build a
    spec from labels alone. ``variables`` labels the per-equation acceptance table.
    """

    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    slots: tuple[MonitorSlot, ...] = ()
    variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("primary", "secondary", "variables"):
            object.__setattr__(self, name, tuple(str(v) for v in getattr(self, name)))
        object.__setattr__(self, "slots", tuple(self.slots))
        labels = self.primary + self.secondary
        if len(set(labels)) != len(labels):
            dup = sorted({x for x in labels if labels.count(x) > 1})
            raise ValueError(f"monitored labels must be unique, repeated: {dup}")
        unknown = sorted({s.label for s in self.slots} - set(labels))
        if unknown:
            raise ValueError(f"slots name labels that are not monitored: {unknown}")

    @property
    def labels(self) -> tuple[str, ...]:
        """Every monitored quantity, primary first."""
        return self.primary + self.secondary

    @functools.cached_property
    def _where(self) -> dict[str, MonitorSlot]:
        return {s.label: s for s in self.slots}

    def slot(self, label: str) -> MonitorSlot:
        """The slot of ``label`` (a slot naming ``mon[label]`` itself when none was given)."""
        return self._where.get(label, MonitorSlot(label, label))

    def field_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-draw shape of every monitor field the labels read (storage planning)."""
        top: dict[str, int] = {}
        for label in self.labels:
            s = self.slot(label)
            top[s.field] = max(top.get(s.field, -1), s.index)
        return {name: () if i < 0 else (i + 1,) for name, i in top.items()}

    def stack(self, mon: Mapping[str, Any], labels: Sequence[str] | None = None) -> np.ndarray:
        """``(chains, draws, Q)`` array of the quantities ``labels`` (default: all) from a monitor
        store ``mon`` of ``(chains, draws, …)`` arrays."""
        labels = self.labels if labels is None else tuple(labels)
        cols, cache = [], {}
        for label in labels:
            s = self.slot(label)
            if s.field not in mon:
                raise KeyError(f"monitor field {s.field!r} (for {label!r}) is not in the store")
            if s.field not in cache:
                cache[s.field] = np.asarray(mon[s.field], dtype=np.float64)
            arr = cache[s.field]
            col = arr if s.index < 0 else arr[..., s.index]
            if col.ndim != 2:
                raise ValueError(
                    f"monitor {label!r} must be (chains, draws) after selection, got {col.shape}"
                )
            cols.append(col)
        if not cols:
            return np.empty((0, 0, 0))
        shapes = {c.shape for c in cols}
        if len(shapes) != 1:
            raise ValueError(f"monitors disagree on (chains, draws): {sorted(shapes)}")
        return np.stack(cols, axis=-1)

    def select(self, mon: Mapping[str, Any], labels: Sequence[str] | None = None) -> dict:
        """``{label: (chains, draws) array}`` for ``labels`` (default: all)."""
        labels = self.labels if labels is None else tuple(labels)
        x = self.stack(mon, labels)
        return {label: x[..., j] for j, label in enumerate(labels)}

    def field_labels(self, name: str, *, primary: bool | None = None) -> tuple[str, ...]:
        """Labels whose slot reads field ``name`` (optionally only primary or secondary)."""
        pool = {True: self.primary, False: self.secondary, None: self.labels}[primary]
        return tuple(label for label in pool if self.slot(label).field == name)


def monitor_spec(static: StaticConfig, variables: tuple[str, ...]) -> MonitorSpec:
    """The monitored quantities of ``static`` labelled with the variable names (DESIGN §8.1)."""
    variables = tuple(str(v) for v in variables)
    n = static.shapes.n
    if len(variables) != n:
        raise ValueError(f"need {n} variable names (one per equation), got {len(variables)}")
    if len(set(variables)) != n:
        raise ValueError(f"variable names must be unique, got {variables}")
    slots: list[MonitorSlot] = []
    primary: list[str] = []
    fields = monitor_fields(
        static.mean.kind, static.vol, n, static.shapes.n_det, coef=static.coef,
        lags=static.shapes.lags,
    )  # fmt: skip
    for f in fields:
        if f.kind == "scalar":
            new = [MonitorSlot(f.name, f.name)]
        elif f.kind == "group":
            new = [MonitorSlot(f"{f.name}[{g}]", f.name, g) for g in range(static.coef.n_groups)]
        elif f.kind == "equation":
            new = [MonitorSlot(f"{f.name}[{v}]", f.name, i, v) for i, v in enumerate(variables)]
        else:
            rows, cols = np.tril_indices(n, -1)
            new = [
                MonitorSlot(
                    f"{f.name}[{variables[r]},{variables[c]}]",
                    f.name,
                    j,
                    f"{variables[r]},{variables[c]}",
                )
                for j, (r, c) in enumerate(zip(rows.tolist(), cols.tolist(), strict=True))
            ]
        slots.extend(new)
        if f.primary:
            primary.extend(s.label for s in new)
    secondary = tuple(s.label for s in slots if s.label not in set(primary))
    return MonitorSpec(tuple(primary), secondary, tuple(slots), variables)


# --------------------------------------------------------------------------- device side


def gaussian_diag_logpdf(x: Array, prior: GaussianDiag) -> Array:
    """``Σ_j m_j [½ log(p_j/2π) − ½ p_j (x_j − μ_j)²]`` over the active slots (``m = cmask``)."""
    terms = 0.5 * (jnp.log(prior.prec) - _LOG_2PI) - 0.5 * prior.prec * (x - prior.mean) ** 2
    return jnp.sum(jnp.where(prior.cmask > 0, terms, 0.0))


def monitor_values(
    st: ChainState, data: EngineData, prior: PriorArrays, cache: FitCache, program: Program
) -> dict[str, Array]:
    """The monitor fields of one chain's state (jnp; jit- and vmap-safe; DESIGN §8.1, §5.1).

    Equations are batched with ``program.eq_map``, exactly as in the sweep (DESIGN §4.5): per
    equation the structure's context, the mean's fit ``g_i`` and the residual ``e_i`` give the
    exact log-likelihood ``vol.loglik_eq`` and the log prior

        log p = Σ_i [mean.log_prior_eq(δ_i, a_i | pm, cm) (+ log N(a_i; cm) for a non-fusing mean)
                     + coef.log_prior_eq + vol.log_prior_eq] + Σ_g global_step.log_prior,

    plus, with SOC/SUR dummy rows (WP25), their factor ``Σ_d log N(y_d; F_dθ_i, 1/w_d)`` per
    structural equation (``LinearStructural.dummy_log_density_eq``; reduced form: inside the
    CCCM step's ``log_prior``), i.e. ``log p`` up to the rows' normalising constant,

    with ``cm = structure.contemp_moments`` (λ-scaled where static) and ``pm =
    coef.moments_eq`` divided by ``λ_i`` when ``coef_scaling == "lambda"`` (Mode II): a
    ``GaussianDiag`` carries its scaling (DESIGN §2.9), and ``coef.log_prior_eq`` is the density
    of the coefficient prior's own state (zero for ``FixedDiagonal``). ``B = (A⁻¹Γ)'`` comes
    from ``structure.reduced_form`` (one triangular solve) or ``B = Π`` (reduced form);
    ``h_1``, ``h_mid`` are rows ``0`` and ``⌈T_eff/2⌉ − 1`` of the persistent log variance.
    The :func:`auxiliary_fields` read the outlier indices ``o_t = o_idx_t + 1`` of the state:
    ``outlier_share_i = Σ_t m_t 1{o_it > 1}/T_eff``, ``outlier_at_max_i = Σ_t m_t 1{o_it = K}/T_eff``.
    """
    structure, mean, coef, vol = program.structure, program.mean, program.coef, program.vol
    st, data, prior, cache = jax.tree.map(jnp.asarray, (st, data, prior, cache))  # host inputs
    n = data.Y.shape[-1]
    lags = data.hist.shape[0]
    n_det = data.X.shape[-1] - n * lags
    coef_static = getattr(coef, "static", None)
    coef_static = coef_static if isinstance(coef_static, CoefPriorStatic) else None
    coef_log = bool(getattr(mean.static, "coef_log", False))
    fields = monitor_fields(mean.static.kind, vol.static, n, n_det, coef=coef_static, lags=lags)
    names = {f.name for f in fields}
    glob, gvol, shared = st.glob, st.glob.vol, prior.shared
    fuses = bool(mean.fuses_contemp)
    scale_coef = mean.static.coef_scaling == "lambda"
    structural = mean.static.kind == "linear_structural"
    ytil = structure.pre_residual(glob, data, mean.fitted_reduced(glob, data))

    def body(i: Array, eq_i: Any, prior_i: Any, cache_i: Any) -> dict[str, Any]:
        del cache_i
        a = eq_i.struct.a
        ctx = structure.context(i, ytil, eq_i, vol, data, gvol)
        if fuses:
            ctx_m = ctx
        else:  # as in the sweep: a non-fusing mean sees the target net of the contemporaneous term
            ctx_m = dataclasses.replace(ctx, target=ctx.target - (ctx.cols * ctx.cmask) @ a)
        g = mean.fitted_eq(eq_i.mean, a, ctx_m, cache, shared)
        e = structure.residual(a, ctx, g)
        cm = structure.contemp_moments(ctx, prior_i)
        pm = coef.moments_eq(eq_i.coef, i, prior_i, glob.coef)
        if scale_coef:
            pm = dataclasses.replace(pm, prec=pm.prec * jnp.exp(-ctx.log_scale))
        log_prior = mean.log_prior_eq(eq_i.mean, a, pm, cm if fuses else None, shared)
        if structural and mean.static.n_dummy:  # SOC/SUR prior rows (WP25)
            log_prior = log_prior + mean.dummy_log_density_eq(eq_i.mean, a, ctx, cache, shared)
        if not fuses:
            log_prior = log_prior + gaussian_diag_logpdf(a, cm)
        coefs = mean.coefficients_eq(eq_i.mean, a, shared)
        log_prior = log_prior + coef.log_prior_eq(eq_i.coef, coefs, prior_i, glob.coef)
        log_prior = log_prior + vol.log_prior_eq(eq_i.vol, prior_i, data.tmask)
        out = {
            "loglik": vol.loglik_eq(eq_i.vol, e, data.tmask, gvol),
            "log_prior": log_prior,
            "vol": vol.extract_eq(eq_i.vol, data.T_eff),
        }
        if "h_1" in names:
            lv = vol.log_var_eq(eq_i.vol, gvol, "persistent")
            mid = (data.T_eff + 1) // 2 - 1  # data row of the state h_{⌈T/2⌉}
            out["h_1"] = lv[0]
            out["h_mid"] = lax.dynamic_index_in_dim(lv, mid, keepdims=False)
        if structural and "B_own1" in names:
            out["gamma"] = mean.extract_eq(eq_i.mean, a, shared).gamma
        return out

    per = program.eq_map(body)(jnp.arange(n, dtype=jnp.int32), st.eq, prior.eq, cache.eq)
    log_prior = jnp.sum(per["log_prior"])
    for step in (*getattr(program, "global_pre", ()), *getattr(program, "global_post", ())):
        data_fn = getattr(step, "log_prior_data", None)
        state_fn = getattr(step, "log_prior_state", None) if coef_log else None
        if data_fn is not None:  # P3 §5.5: the CommonSV h prior needs the data mask
            log_prior = log_prior + data_fn(glob, prior, data)
        elif state_fn is not None:  # P3 §3.7: a varying prior's density needs the full state
            log_prior = log_prior + state_fn(st, prior, program.eq_map)
        else:
            log_prior = log_prior + step.log_prior(glob, prior)
    core, trans = per["vol"].core, per["vol"].trans
    values: dict[str, Any] = {
        "loglik": jnp.sum(per["loglik"]) + structure.log_jacobian(glob, data.T_eff),
        "log_prior": log_prior,
    }
    if "B_own1" in names:
        if structural:
            B = structure.reduced_form(structure.extract(st.eq.struct, glob), per["gamma"])
        else:
            B = mean.extract_global(glob).mean.Pi
        values["B_own1"] = B[n_det + np.arange(n), np.arange(n)]
        if "intercept" in names:
            values["intercept"] = B[0]
    for name in ("log_lam", "mu", "phi", "sigma", "h_T"):
        if name in names:
            values[name] = getattr(core, name)
    for name in ("h_1", "h_mid"):
        if name in names:
            values[name] = per[name]
    if "a" in names:
        rows, cols = np.tril_indices(n, -1)
        values["a"] = structure.extract(st.eq.struct, glob).a[rows, cols]
    if "common_h_T" in names:  # CommonSV (P3 D-52): h at data rows T_eff − 1, 0, ⌈T_eff/2⌉ − 1
        gv = glob.vol
        T_eff = jnp.asarray(data.T_eff, jnp.int32)
        values["common_h_T"] = lax.dynamic_index_in_dim(gv.h, T_eff - 1, keepdims=False)
        values["common_rho"] = gv.rho
        values["common_sigma"] = jnp.exp(gv.log_sigma)
        values["common_h_0"] = gv.h[0]
        mid = (T_eff + 1) // 2 - 1
        values["common_h_mid"] = lax.dynamic_index_in_dim(gv.h, mid, keepdims=False)
    if "log_p" in names:
        values["log_p"] = trans.log_p
    if "nu" in names:
        values["nu"] = trans.nu
    gcoef = glob.coef
    for name, leaf in (("gl_log_zeta", "log_zeta"), ("gl_log_tau", "log_tau"),
                       ("gl_log_xi", "log_xi"), ("gl_a_idx", "a_idx")):  # fmt: skip
        if name in names:
            values[name] = getattr(gcoef, leaf)
    if "gl_p" in names:
        values["gl_p"] = jnp.exp(gcoef.log_p)
    aux = auxiliary_fields(vol.static)
    if aux:
        o_idx = st.eq.vol.trans.o_idx  # (n, T_pad), o = o_idx + 1; padded rows are masked
        live = data.tmask > 0
        rows = jnp.sum(data.tmask)  # T_eff
        values["outlier_share"] = jnp.sum(live & (o_idx > 0), axis=-1) / rows
        top = vol.static.outlier_K - 1
        values["outlier_at_max"] = jnp.sum(live & (o_idx == top), axis=-1) / rows
    names_out = (*(f.name for f in fields), *aux)
    return {name: jnp.asarray(values[name], dtype=jnp.float64) for name in names_out}


# --------------------------------------------------------------------------- the flag


def _fmt(x: float, digits: int) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{digits}f}"


def _chains(mon: Mapping[str, Any], spec: MonitorSpec) -> int:
    for label in spec.labels:
        s = spec.slot(label)
        if s.field in mon:
            return int(np.shape(mon[s.field])[0])
    for value in mon.values():
        return int(np.shape(value)[0])
    return 0


_stats_cache: OrderedDict[bytes, tuple[float, float]] = OrderedDict()
_stats_lock = threading.Lock()


def _content_key(draws: np.ndarray) -> bytes:
    """SHA-256 of a C-contiguous float64 ``(chains, draws)`` array and of its shape."""
    digest = hashlib.sha256(np.asarray(draws.shape, dtype=np.int64).tobytes())
    digest.update(draws)
    return digest.digest()


def _rhat_ess_once(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R-hat, bulk ESS)`` of every column of ``x`` (``(chains, draws, Q)``), memoised by the
    content of each column. R-hat and bulk ESS depend on a quantity's own draws only, so a hit
    equals a recomputation to rounding: memoised values are those of the batch in which the
    quantity was first evaluated (the FFT's last bits depend on the batch it ran in, DESIGN
    §8.2). A fit runs :func:`converged`, :func:`diagnostics_table` and
    :func:`warning_catalogue` (which runs :func:`converged` again) on the same draws; the cache
    makes that one evaluation per quantity (DESIGN §8.2). Least recently used pairs are dropped
    beyond ``_STATS_CACHE_SIZE``; an array changed in place has new content, hence a new key."""
    q = x.shape[-1]
    rhat, ess = np.full(q, np.nan), np.full(q, np.nan)
    if x.size == 0:
        return rhat, ess
    cols = np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float64), -1, 0))
    keys = [_content_key(c) for c in cols]
    todo: list[int] = []
    with _stats_lock:
        for j, key in enumerate(keys):
            hit = _stats_cache.get(key)
            if hit is None:
                todo.append(j)
            else:
                rhat[j], ess[j] = hit
                _stats_cache.move_to_end(key)
    if todo:
        rhat[todo], ess[todo] = dg.rhat_ess_bulk_many(np.moveaxis(cols[todo], 0, -1))
        with _stats_lock:
            for j in todo:
                _stats_cache[keys[j]] = (float(rhat[j]), float(ess[j]))
            while len(_stats_cache) > _STATS_CACHE_SIZE:
                _stats_cache.popitem(last=False)
    return rhat, ess


def rhat_ess(mon: Mapping[str, Any], spec: MonitorSpec, labels: Sequence[str]) -> tuple:
    """``(R-hat, bulk ESS)`` arrays of the quantities ``labels`` (``rhat_ess_bulk_many``),
    memoised by the content of each quantity's draws (see ``_rhat_ess_once``)."""
    x = spec.stack(mon, labels)
    if x.size == 0:
        return np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    return _rhat_ess_once(x)


def _severity(rhat: np.ndarray, ess: np.ndarray, settings: MCMCSettings) -> np.ndarray:
    """How far beyond the thresholds: ``max((R̂ − 1)/(R̂_max − 1), ESS_min/ESS)``, NaN → ∞."""
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.maximum((rhat - 1.0) / (settings.rhat_max - 1.0), settings.ess_min / ess)
    return np.where(np.isnan(score), np.inf, score)


def converged(
    mon: Mapping[str, np.ndarray],
    spec: MonitorSpec,
    settings: MCMCSettings,
    *,
    error: bool,
    eb_converged: bool | None,
) -> tuple[bool, list[str]]:
    """The ``converged`` flag of a run and the warnings explaining it (DESIGN §8.2, §8.3).

    ``mon`` is the monitor store (fields → ``(chains, draws, …)`` arrays). ``error``: some chain
    reported a numerical failure. ``eb_converged``: the EB optimisation's flag, ``None`` when the
    hyperparameters were fixed. Single-chain runs are never converged (split R-hat cannot see
    between-chain disagreement). Secondary quantities only produce the informational ``a_ij``
    warning (more than 5% with R-hat ≥ ``rhat_max``) and never change the flag.
    """
    ok, out = True, []
    chains = _chains(mon, spec)
    if error:
        ok = False
        out.append(
            "a chain reported a numerical failure (MCMCNumericalError), so fit.converged is False"
        )
    if chains < 2:
        ok = False
        out.append(
            f"{'single chain' if chains == 1 else 'no chains'}: split R-hat cannot detect "
            "between-chain disagreement, so fit.converged is False; use chains ≥ 2"
        )
    if spec.primary:
        rhat, ess = rhat_ess(mon, spec, spec.primary)
        good = (rhat < settings.rhat_max) & (ess > settings.ess_min)
        if not good.all():
            ok = False
            bad = np.flatnonzero(~good)
            order = bad[np.argsort(-_severity(rhat[bad], ess[bad], settings), kind="stable")]
            worst = "; ".join(
                f"{spec.primary[j]} R-hat {_fmt(rhat[j], 3)}, ESS {_fmt(ess[j], 0)}"
                for j in order[:_WORST_PRIMARY]
            )
            out.append(
                f"MCMC not converged: {bad.size} of {len(spec.primary)} primary quantities fail "
                f"R-hat < {settings.rhat_max:g} or bulk ESS > {settings.ess_min:g} (worst: "
                f"{worst}); increase draws, burn or thin"
            )
    a_labels = spec.field_labels("a", primary=False) or tuple(
        label for label in spec.secondary if label.startswith("a[") and label not in spec._where
    )
    if a_labels:
        rhat, _ = rhat_ess(mon, spec, a_labels)
        bad = np.flatnonzero(~(rhat < settings.rhat_max))
        if bad.size > _SECONDARY_A_FRACTION * len(a_labels):
            order = bad[
                np.argsort(-np.where(np.isnan(rhat[bad]), np.inf, rhat[bad]), kind="stable")
            ]
            worst = ", ".join(f"{a_labels[j]} {_fmt(rhat[j], 3)}" for j in order[:_WORST_SECONDARY])
            frac = bad.size / len(a_labels)
            out.append(
                f"secondary (informational; does not gate converged): {frac:.0%} of the "
                f"{len(a_labels)} contemporaneous coefficients a_ij have R-hat ≥ "
                f"{settings.rhat_max:g} (worst R-hat: {worst})"
            )
    if eb_converged is False:
        ok = False
        out.append(
            "the empirical-Bayes optimisation of the Minnesota hyperparameters did not converge, "
            "so fit.converged is False"
        )
    return ok, out


# --------------------------------------------------------------------------- warnings


def is_order_invariant(static: StaticConfig) -> bool:
    """Whether the composition's posterior is invariant to the variable ordering (N08 §B2):
    ``Linear(form="reduced") × Constant(prior="iw_equivalent")`` (with ``κ_a = 1``, Chan 2022
    Prop. 1) and, in Phase 3, the shared-factor path (the conjugate prior × ``CommonSV``: the NIW
    prior with a common scalar volatility, exactly invariant; P3 §5.3, E99). Every triangular SV,
    outlier or t composition is not. Mode II × ``Constant`` at ``cross = contemp = 1`` is
    invariant too, but those values are data, not static: the resolution step passes
    ``DiagnosticsInput.order_invariant`` for it (E99, E130)."""
    if static.mean.const_path == "shared_factor":
        return True
    return (
        static.mean.kind == "linear_reduced"
        and static.struct.contemp_scaling == "lambda"
        and static.vol.persistent == "constant"
        and static.vol.transitory == "none"
        and not static.vol.common
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class DiagnosticsInput:
    """Host-side inputs of :func:`warning_catalogue` (DESIGN §8.3); unknown inputs stay ``None``.

    ``acc`` holds per-chain, per-equation rates from the final state's accumulators divided by
    the sampling sweeps, keyed by leaf name with or without the ``acc_`` prefix, arrays ``(chains,
    n)`` or ``(n,)``: ``h`` (exact-h acceptance), ``rej_run_max`` (longest run of exact-h
    rejections, in sweeps), ``fold`` (NC sign-fold fraction), ``floor`` (masked exact zeros,
    ``zero_offset`` mode). ``outlier_at_max``: per equation, the posterior mean fraction of real
    rows whose outlier scale sat at ``K`` (``(n,)`` or ``(chains, n)``); ``None`` averages the
    monitor's auxiliary ``outlier_at_max`` field over chains and draws. ``covers_2020`` and
    ``tcode_clash`` come from the resolution step (the Phase 1 checks); ``eb_warnings`` are the
    conjugate twin's messages, appended verbatim. ``sv_budget`` (``(chains, draws)``, from the
    resolution step) is set only for a stochastic-volatility composition fitted with fewer kept
    draws per chain than ``SV_DEFAULT_DRAWS`` (D-35); engine-level fits leave it ``None``.
    """

    static: StaticConfig
    settings: MCMCSettings
    variables: tuple[str, ...] = ()
    mon: Mapping[str, np.ndarray] | None = None
    spec: MonitorSpec | None = None
    acc: Mapping[str, np.ndarray] = field(default_factory=dict)
    chains: int | None = None
    error: bool = False
    eb_converged: bool | None = None
    eb_warnings: tuple[str, ...] = ()
    outlier_at_max: np.ndarray | None = None
    covers_2020: bool = False
    tcode_clash: tuple[str, ...] = ()
    order_invariant: bool | None = None
    sv_budget: tuple[int, int] | None = None


def _acc_names(acc: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for key, value in acc.items():
        name = key[4:] if key.startswith("acc_") else key
        out[name] = np.asarray(value, dtype=np.float64)
    return out


def _per_equation(x: np.ndarray, n: int, how: Literal["mean", "max", "min", "sum"]) -> np.ndarray:
    """Reduce a ``(chains, n)`` (or ``(n,)``) array over chains."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[-1] != n:
        raise ValueError(f"expected per-equation values with last axis {n}, got {x.shape}")
    x = x.reshape(-1, n)
    return {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}[how](x, axis=0)


def _mean_above(x: np.ndarray, n: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Per equation, the mean of ``x`` (``(…, n)``, every leading axis averaged) and whether it
    is strictly above ``threshold``. The test reads the sign of the mean excess ``x − threshold``,
    which is exactly zero when every value sits at the threshold: a plain float mean of copies of
    ``0.01`` can round to either side of it (e.g. ``0.010000000000000503`` over 4 × 2000 draws)."""
    x = np.asarray(x, dtype=np.float64)
    return _per_equation(x, n, "mean"), _per_equation(x - threshold, n, "mean") > 0.0


def _names(variables: tuple[str, ...], mask: np.ndarray) -> list[str]:
    return [v for v, m in zip(variables, mask, strict=True) if m]


def _values(x: np.ndarray, mask: np.ndarray, digits: int = 3) -> list[float]:
    return [round(float(v), digits) for v in x[mask]]


def sv_budget_message(chains: int, draws: int, *, common: bool = False) -> str:
    """The D-35 warning of a stochastic-volatility fit below the SV default budget (DESIGN §8.3):
    the evidence behind the default (DESIGN Q14) and how to raise the draws. ``common`` names
    ``CommonSV``, which counts as stochastic volatility for D-35 (P3 D-51, §8)."""
    kind = "CommonSV stochastic-volatility" if common else "stochastic-volatility"
    return (
        f"{kind} fit with {chains} × {draws:,} kept draws, fewer than the SV "
        f"default of {SV_DEFAULT_DRAWS:,} per chain (D-35): the SV parameters mix slowly. In the "
        "Phase 2 evidence (DESIGN Q14) SV fits at 4 × 2,000 missed the R-hat/ESS gate routinely "
        "(the README example at 0 of 11 origins), Clark–Ravazzolo SV(rw) converged at 68% of "
        "origins at 4 × 5,000 and at all 106 at 4 × 20,000, and BENCH-P2 (n = 20, T = 300) "
        "still misses R-hat < 1.01 on 4 of 101 primary quantities at 4 × 10,000. Raise the "
        f"budget with mcmc=MCMCSettings(draws={SV_DEFAULT_DRAWS:_}) or more, or leave draws "
        "unset for the default"
    )


_GL_SCALE_FIELDS = ("gl_log_zeta", "gl_log_xi", "gl_p")
_A_GRID_END_MASS = 0.5  # draws of a_g at a grid end: "the grid may be too narrow" (P3 §8)


def _gl_warnings(
    mon: Mapping[str, Any], spec: MonitorSpec, static: StaticConfig, settings: MCMCSettings
) -> list[str]:
    """P3 §8 warnings of a global-local prior: secondary group scales (``G > max(2p, n)``) that
    fail R-hat or ESS (informational, named; primary ones are in the convergence message), and a
    hyperprior index ``a_idx`` at an end of its grid for more than half the draws."""
    out: list[str] = []
    labels = tuple(
        label
        for name in _GL_SCALE_FIELDS
        for label in spec.field_labels(name, primary=False)
        if spec.slot(label).field in mon
    )
    if labels:
        rhat, ess = rhat_ess(mon, spec, labels)
        bad = np.flatnonzero(~((rhat < settings.rhat_max) & (ess > settings.ess_min)))
        if bad.size:
            order = bad[np.argsort(-_severity(rhat[bad], ess[bad], settings), kind="stable")]
            worst = "; ".join(
                f"{labels[j]} R-hat {_fmt(rhat[j], 3)}, ESS {_fmt(ess[j], 0)}"
                for j in order[:_WORST_PRIMARY]
            )
            out.append(
                f"secondary (informational; does not gate converged): {bad.size} of "
                f"{len(labels)} global-local group scales fail R-hat < {settings.rhat_max:g} or "
                f"bulk ESS > {settings.ess_min:g} (worst: {worst})"
            )
    A = static.coef.a_grid
    if A and "gl_a_idx" in mon:
        idx = np.asarray(mon["gl_a_idx"], dtype=np.float64)  # (chains, draws, G)
        G = idx.shape[-1]
        flat = idx.reshape(-1, G)
        mass = ((flat == 0) | (flat == A - 1)).mean(axis=0) if flat.size else np.zeros(G)
        bad = mass > _A_GRID_END_MASS
        if bad.any():
            groups = [int(g) for g in np.flatnonzero(bad)]
            out.append(
                f"the hyperprior on a puts more than {_A_GRID_END_MASS:.0%} of its draws at an end "
                f"of its {A}-point grid for coefficient group(s) {groups} (mass "
                f"{[round(float(m), 3) for m in mass[bad]]}): the grid may be too narrow"
            )
    return out


def _csv_warnings(acc: Mapping[str, np.ndarray]) -> list[str]:
    """P3 §8 CommonSV warnings from the per-chain rates (E132): an ``h`` block acceptance below
    0.1 in some chain, and any Newton iteration cap hit (informational: the draw is exact)."""
    out: list[str] = []
    rate = acc.get("csv_h")
    if rate is not None and np.size(rate):
        low = float(np.min(rate))
        if low < _CSV_H_ACCEPT_MIN:
            out.append(
                f"CommonSV h block acceptance {low:.3f} is below {_CSV_H_ACCEPT_MIN} (lowest "
                "chain): the blocks are too long for the data; decrease CommonSV(block=…)"
            )
    cap = acc.get("csv_newton")
    if cap is not None and np.size(cap) and float(np.max(cap)) > 0.0:
        out.append(
            f"CommonSV h step: the Newton search for a block's proposal mode hit its iteration "
            f"cap or stalled ({float(np.max(cap)):.3g} blocks per sweep in the worst chain); "
            "informational: the draw is exact, only the proposal is less efficient"
        )
    return out


def warning_catalogue(inputs: DiagnosticsInput) -> list[str]:
    """Every warning of the DESIGN §8.3 table that applies, in the table's order.

    The convergence messages come from :func:`converged` when monitors are given (otherwise only
    the chain-count, error and EB reasons, from ``chains`` or ``settings.chains``); acceptance
    rates and the order-invariance, 2020 and transformation checks never change the flag.
    """
    static, vol, settings = inputs.static, inputs.static.vol, inputs.settings
    n = static.shapes.n
    spec = inputs.spec
    variables = tuple(inputs.variables) or (spec.variables if spec is not None else ())
    if not variables:
        variables = tuple(f"y{i + 1}" for i in range(n))
    if len(variables) != n:
        raise ValueError(f"need {n} variable names, got {len(variables)}")
    out: list[str] = []
    if inputs.mon is not None:
        spec = spec if spec is not None else monitor_spec(static, variables)
        out += converged(
            inputs.mon, spec, settings, error=inputs.error, eb_converged=inputs.eb_converged
        )[1]
    else:
        chains = settings.chains if inputs.chains is None else inputs.chains
        empty = MonitorSpec((), ())
        out += converged(
            {"_": np.zeros((chains, 0))},
            empty,
            settings,
            error=inputs.error,
            eb_converged=inputs.eb_converged,
        )[1]
    if inputs.sv_budget is not None:
        out.append(sv_budget_message(*inputs.sv_budget, common=vol.common))
    if inputs.mon is not None and spec is not None:
        out += _gl_warnings(inputs.mon, spec, static, settings)
    acc = _acc_names(inputs.acc)
    if vol.persistent == "sv" and vol.exact and ("h" in acc or "rej_run_max" in acc):
        rate = _per_equation(acc["h"], n, "min") if "h" in acc else np.full(n, np.nan)
        run = _per_equation(acc["rej_run_max"], n, "max") if "rej_run_max" in acc else np.zeros(n)
        bad = (rate < _H_ACCEPT_MIN) | (run > _REJECTION_RUN_MAX)
        if bad.any():
            out.append(
                f"exact-h acceptance below {_H_ACCEPT_MIN} or a run of more than "
                f"{_REJECTION_RUN_MAX} rejected sweeps for {_names(variables, bad)} (acceptance "
                f"{_values(rate, bad)}, longest run {_values(run, bad, 0)}): the mixture "
                "correction mixes poorly there (N01 §10.2); acceptance alone is not enough, check "
                "the h diagnostics and consider a longer burn-in"
            )
    if vol.common:
        out += _csv_warnings(acc)
    if vol.persistent == "sv" and "fold" in acc:
        fold, bad = _mean_above(acc["fold"], n, _FOLD_MAX)
        if bad.any():
            out.append(
                f"non-centred sign fold in more than {_FOLD_MAX:.0%} of sweeps for "
                f"{_names(variables, bad)} (fractions {_values(fold, bad, 4)}): near-homoskedastic "
                "equation(s), where stochvol and bayesianVARs differ from rai-bvar (D-7)"
            )
    if vol.has_t and inputs.mon is not None and "nu" in inputs.mon:
        nu = np.asarray(inputs.mon["nu"], dtype=np.float64)
        top = max(vol.nu_grid)
        mass = (nu == top).reshape(-1, n).mean(axis=0) if nu.size else np.zeros(n)
        bad = mass >= _NU_BOUND_MASS
        if bad.any():
            out.append(
                f"Student-t ν has posterior mass ≥ {_NU_BOUND_MASS} at the grid bound {top} for "
                f"{_names(variables, bad)} (mass {_values(mass, bad)}): near-Gaussian tails, SV "
                f"may suffice (report only; mass at ν = {min(vol.nu_grid)} is normal, N03 §6)"
            )
    at_max = inputs.outlier_at_max
    if vol.has_outliers and at_max is None and inputs.mon is not None:
        draws = np.asarray(inputs.mon.get("outlier_at_max", ()), dtype=np.float64)
        at_max = draws.reshape(-1, n) if draws.size else None  # (chains, draws, n) → (C·D, n)
    if vol.has_outliers and at_max is not None:  # "> 1% of rows": exactly 1% is quiet
        frac, bad = _mean_above(at_max, n, _OUTLIER_AT_K_MAX)
        if bad.any():
            out.append(
                f"outlier scale at its maximum K = {vol.outlier_K} for more than "
                f"{_OUTLIER_AT_K_MAX:.0%} of rows in {_names(variables, bad)} (fractions "
                f"{_values(frac, bad, 4)}): consider a larger outlier_max"
            )
    if vol.zero_offset and "floor" in acc:
        hits = _per_equation(acc["floor"], n, "sum")
        bad = hits > 0
        if bad.any():
            out.append(
                f"zero_offset parity mode: exact zero residuals were masked from the SV step in "
                f"{_names(variables, bad)} (floor hits {_values(hits, bad, 4)}; N02 §2.4)"
            )
    invariant = (
        is_order_invariant(static) if inputs.order_invariant is None else inputs.order_invariant
    )
    if not invariant and n >= _ORDERING_N:
        out.append(
            f"this composition is not order-invariant: with {n} variables its results depend on "
            "their ordering (R8; N08 §B2); gauge how much with "
            "rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)"
        )
    if inputs.covers_2020 and not vol.has_outliers:
        out.append(
            "the estimation sample covers 2020 but no outlier treatment is set; consider "
            "SV(outliers=True) (SPEC §5.6)"
        )
    if inputs.tcode_clash:
        out.append(
            f"own-lag prior means contradict the transformation codes for "
            f"{list(inputs.tcode_clash)} (levels with mean < 0.5 or differences with mean ≥ 0.5)"
        )
    out += [w for w in dict.fromkeys(inputs.eb_warnings) if w not in out]
    return out


# --------------------------------------------------------------------------- tables


def acceptance_table(
    acc: Mapping[str, np.ndarray],
    variables: tuple[str, ...],
    mon: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-equation sampler statistics (DESIGN §8.4): one row per variable.

    From ``acc`` (``acc_`` prefix dropped): one column per per-equation accumulator (``(chains,
    n)`` or ``(n,)``) averaged over chains (``*_max``: the maximum, ``floor``: the sum);
    ``p_outlier``, the Rao-Blackwellised ``P(o_t > 1 | y)`` per row (``(chains, n, T)``, the
    runner's ``latent["p_outlier"]``), averaged over chains and the first ``acc["T_eff"]`` rows
    (every row without ``T_eff``). Per-chain scalars (``min_pivot``, ``(chains,)``) go to
    ``attrs`` (``min_*``: the minimum over chains); other shapes (raw ``rb_*`` sums) are
    ignored. From ``mon``: the posterior means of the :func:`auxiliary_fields`, ``outlier_share``
    (mean over real rows of ``P(o_t > 1 | y)``, estimated from the draws of ``o``) and
    ``outlier_at_max``.
    """
    n = len(variables)
    cols: dict[str, np.ndarray] = {}
    attrs: dict[str, float] = {}
    raw = _acc_names(acc)
    T_eff = raw.pop("T_eff", None)
    for name, value in raw.items():
        how = "max" if name.endswith("max") else "sum" if name == "floor" else "mean"
        if name == "p_outlier":
            if value.ndim >= 2 and value.shape[-2] == n and value.shape[-1] >= 1:
                rows = value if T_eff is None else value[..., : int(T_eff)]
                cols[name] = rows.reshape(-1, n, rows.shape[-1]).mean(axis=(0, 2))
        elif name in _CHAIN_SCALARS or value.ndim == 0:
            attrs[name] = float(np.min(value) if name.startswith("min") else np.mean(value))
        elif value.ndim <= 2 and value.shape[-1] == n:
            cols[name] = _per_equation(value, n, how)
    for name in _AUX_OUTLIER:
        draws = np.asarray(mon[name], dtype=np.float64) if mon is not None and name in mon else None
        if draws is not None and draws.size and draws.shape[-1] == n:
            cols[name] = draws.reshape(-1, n).mean(axis=0)
    table = pd.DataFrame(cols, index=pd.Index(variables, name="variable"))
    table.attrs.update(attrs)
    return table


_TABLE_COLUMNS = (
    *("mean", "sd", "mcse_mean", "q5", "q50", "q95", "mcse_q5", "mcse_q50", "mcse_q95"),
    *("rhat", "ess_bulk", "ess_tail"),
)


def diagnostics_table(
    mon: Mapping[str, np.ndarray], spec: MonitorSpec, acc: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """``fit.diagnostics`` (DESIGN §8.4): one row per monitored quantity, primary first.

    Columns: ``field``, ``variable``, ``primary``; posterior ``mean``, ``sd``, quantiles
    ``q5``/``q50``/``q95``; MCSE of the mean and of those quantiles (Vehtari et al. 2021
    §4.3–4.4); ``rhat`` (rank-normalised split, max of bulk and folded), ``ess_bulk``,
    ``ess_tail``; ``escape``, for ν, the per-sweep probability ``P(ν^{(m+1)} ≠ ν^{(m)})`` (N03
    §6): the ``nu_move`` rate of ``acc`` (the runner's ``acc_nu_move``) averaged over chains, or
    without it the share of consecutive kept draws in which ν moved (the same only when
    ``thin = 1``). ``attrs["equations"]`` holds the per-equation :func:`acceptance_table` (with
    the monitored outlier shares) as ``{column: {variable: value}}``, its attrs (``min_pivot``,
    the smallest pivot over chains) are copied into ``attrs``, and ``attrs["chains"]``,
    ``attrs["draws"]`` give the run size. R-hat and bulk ESS come from the memoised
    :func:`rhat_ess`, the other statistics from ``diagnostics.summary_many``.
    """
    labels = spec.labels
    x = spec.stack(mon, labels)
    chains, draws = (x.shape[0], x.shape[1]) if x.ndim == 3 else (0, 0)
    if labels:
        stats = dg.summary_many(x, rank_normalised=False)
        stats["rhat"], stats["ess_bulk"] = _rhat_ess_once(x)
    else:
        stats = {c: np.empty(0) for c in _TABLE_COLUMNS}
    slots = [spec.slot(label) for label in labels]
    move = _acc_names(acc).get("nu_move")
    per_sweep = None
    if move is not None and move.ndim >= 1 and move.size:
        per_sweep = _per_equation(move, move.shape[-1], "mean")
    escape = np.full(len(labels), np.nan)
    for j, s in enumerate(slots):
        if s.field != "nu":
            continue
        if per_sweep is not None and 0 <= s.index < per_sweep.size:
            escape[j] = per_sweep[s.index]
        elif draws >= 2:
            escape[j] = float(np.mean(x[:, 1:, j] != x[:, :-1, j]))
    data: dict[str, Any] = {
        "field": [s.field for s in slots],
        "variable": [s.variable for s in slots],
        "primary": [label in set(spec.primary) for label in labels],
    }
    data.update({c: stats[c] for c in _TABLE_COLUMNS})
    data["escape"] = escape
    table = pd.DataFrame(data, index=pd.Index(list(labels), name="quantity"))
    variables = spec.variables
    if variables:
        eq = acceptance_table(acc, variables, mon=mon)
        table.attrs["equations"] = {c: {v: float(eq.at[v, c]) for v in eq.index} for c in eq}
        table.attrs.update(eq.attrs)
    elif acc:
        table.attrs.update(acceptance_table(acc, ()).attrs)  # the per-chain scalars
    table.attrs["chains"], table.attrs["draws"] = int(chains), int(draws)
    return table


__all__ = [
    "DiagnosticsInput",
    "MonitorField",
    "MonitorSlot",
    "MonitorSpec",
    "acceptance_table",
    "auxiliary_fields",
    "converged",
    "diagnostics_table",
    "gaussian_diag_logpdf",
    "gl_fields",
    "is_order_invariant",
    "monitor_fields",
    "monitor_spec",
    "monitor_values",
    "rhat_ess",
    "warning_catalogue",
]
