"""Independent reference implementations of every Phase 2 component kind (DESIGN §9.1, D-22).

**Two-implementation rule** (N06 §2, §5.1; N03 §7.2). Prior samplers, latent-variance maps and
log densities are written here from the parametrisations of DESIGN §6 and the notes, never by the
kernel authors, and this module never imports a kernel module (``components/*``,
``engines/mcmc/linalg.py``, ``engines/mcmc/random.py``). Allowed engine imports: the state
containers (``state.py``), ``keys.py``, ``config.py`` and the jnp design builder ``data.py``.
The OCSN/KSC mixture tables are transcribed here a second time (N01 §5.1, §5.3); a unit test
requires both transcriptions to agree on every printed digit.

**Registry.** :func:`register_reference` mirrors the kernel registry: a Phase 3–5 component adds a
reference class under its kind and :func:`references_for` assigns it to a composition slot, which
gives it per-block and per-composition Geweke configurations (``validation/joint.py``).

Conventions
-----------
* One equation per call (the ``_i`` slices of DESIGN §2.7); ``simulate.prior_sample`` batches the
  equations with ``vmap``. Global parts (``glob.*``) have their own methods.
* **Densities** are w.r.t. Lebesgue measure on the *natural* parameters, i.e. the coordinates in
  which the notes state the priors and the exact conditionals: ``a``, ``δ``, ``Π``, ``λ``
  (Constant; the state stores ``log λ``), ``μ``, ``φ``, ``σ`` (σ > 0; prior stated on σ²),
  ``h``, ``p`` (the state stores ``log p``), ``τ`` (the state stores ``log τ``), and counting
  measure on ``o`` and ``ν``. A density-ratio identity (N06 §5.1) for an exact Gibbs draw compares
  the conditional and the joint in these same coordinates.
* λ-scaled blocks (Mode II, ``iw_equivalent``; DESIGN §4.7): ``prior_sample_eq`` and
  ``log_prior_eq`` take ``log_scale = log λ_i`` of the Constant core; a reference applies it only
  when its own static flag (``StructStatic.contemp_scaling``, ``MeanStatic.coef_scaling``) says so.
* Active slots of ``a_i`` are those with ``prior_i.contemp.cmask = 1`` (``1{j < i}``); inactive
  slots are exactly 0 and carry no density.
* **Padding** (DESIGN §2.3): every sum over data rows is masked with ``m_t``, every transition of
  ``h`` with ``m_{t−1}``; padded rows hold ``o_idx = 0`` and ``log τ = 0``. Padded SV states
  (index ``> T_eff``) carry no model content: the prior sampler fills them with independent
  ``N(0, 1)`` dummies and no density includes them.
* Mixture indicators ``s`` index components in **descending order of the effective mean** (the
  OCSN Table 1 order, for both mixtures; N01 §5.2–§5.3).
* **Finite-variance flags** of the test functions (N06 §3.4) depend on the prior's tails, which
  :class:`PriorTails` summarises (:func:`prior_tails`, host side). No test function reads a padded
  state: the kernel's padded states are decoupled dummies whose law differs from these ``N(0, 1)``
  ones (DESIGN §3.7.4: ``h′ = μ + σx̃′``, then moved by the C and NC steps).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.special import gammaln, logsumexp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    ConstantEq,
    Empty,
    EqPrior,
    LinearEq,
    OutlierEq,
    OutlierTEq,
    PriorArrays,
    RecursiveEq,
    SVEq,
    SVRWEq,
    TEq,
    VolEq,
)
from rai_bvar.validation.dummies import reduced_dummy_log_factor, reduced_dummy_prior_sample

Array = jax.Array
PyTree = Any
Which = Literal["effective", "persistent"]
TestFns = dict[str, tuple[Array, bool]]

LOG_2PI = math.log(2.0 * math.pi)
_F64 = jnp.float64

# --------------------------------------------------------------------------- mixture tables
# Second transcription (DESIGN §9.1). Five printed decimals, as in every source (N01 §5).

# OCSN (2007) Table 1, right-hand block (N01 §5.3; N02 §2.3): means are the log χ²₁ component
# means, NO −1.2704 shift. Columns: p_j, m_j, v_j² (a variance).
_OCSN10_PRINTED: tuple[tuple[float, float, float], ...] = (
    (0.00609, 1.92677, 0.11265),
    (0.04775, 1.34744, 0.17788),
    (0.13057, 0.73504, 0.26768),
    (0.20674, 0.02266, 0.40611),
    (0.22715, -0.85173, 0.62699),
    (0.18842, -1.97278, 0.98583),
    (0.12047, -3.46788, 1.57469),
    (0.05591, -5.55246, 2.54498),
    (0.01575, -8.68384, 4.16591),
    (0.00115, -14.65000, 7.33342),
)

# KSC (1998) Table 4 in its printed order (N01 §5.1): means centred at 0; the effective component
# mean is m_i − 1.2704 (KSC §3.1; N01 §4, §6: the literal 1.2704, not ψ(½) + log 2).
_KSC7_TABLE4_PRINTED: tuple[tuple[float, float, float], ...] = (
    (0.00730, -10.12999, 5.79596),
    (0.10556, -3.97281, 2.61369),
    (0.00002, -8.56686, 5.17950),
    (0.04395, 2.77786, 0.16735),
    (0.34001, 0.61942, 0.64009),
    (0.24566, 1.79518, 0.34023),
    (0.25750, -1.08819, 1.26261),
)
KSC_SHIFT = 1.2704


class MixtureTableRef(NamedTuple):
    """A normal mixture for ``log χ²₁`` with **effective** means (D-3), components sorted by
    descending mean. ``v`` holds variances."""

    name: str
    p: np.ndarray
    m: np.ndarray
    v: np.ndarray


def _table(
    name: str, rows: tuple[tuple[float, float, float], ...], shift: float
) -> MixtureTableRef:
    arr = np.array(rows, dtype=np.float64)
    p, m, v = arr[:, 0], arr[:, 1] - shift, arr[:, 2]
    order = np.argsort(-m, kind="stable")
    return MixtureTableRef(name, p[order], m[order], v[order])


def mixture_table(name: Literal["ocsn10", "ksc7"]) -> MixtureTableRef:
    """The reference transcription of ``"ocsn10"`` (no shift) or ``"ksc7"`` (KSC Table 4 means
    minus 1.2704, re-sorted by descending mean; this reproduces OCSN Table 1, left block)."""
    if name == "ocsn10":
        return _table("ocsn10", _OCSN10_PRINTED, 0.0)
    if name == "ksc7":
        return _table("ksc7", _KSC7_TABLE4_PRINTED, KSC_SHIFT)
    raise ValueError(f"unknown mixture {name!r}; expected 'ocsn10' or 'ksc7'")


def log_mixture_components(x: Array, table: MixtureTableRef) -> Array:
    """``log p_j + log N(x; m_j, v_j)`` for every component; shape ``(*x.shape, J)``."""
    x = jnp.asarray(x, _F64)[..., None]
    p, m, v = (jnp.asarray(a) for a in (table.p, table.m, table.v))
    return jnp.log(p) - 0.5 * (LOG_2PI + jnp.log(v) + (x - m) ** 2 / v)


def log_mixture_density(x: Array, table: MixtureTableRef) -> Array:
    """``log g(x) = log Σ_j p_j N(x; m_j, v_j)`` (N01 §2)."""
    return logsumexp(log_mixture_components(x, table), axis=-1)


def log_logchi2_density(x: Array) -> Array:
    """Exact ``log χ²₁`` density ``log f(ξ) = −½ log 2π + ½ ξ − ½ e^ξ`` (N01 §4)."""
    x = jnp.asarray(x, _F64)
    return -0.5 * LOG_2PI + 0.5 * x - 0.5 * jnp.exp(x)


# --------------------------------------------------------------------------- log densities


def log_normal(x: Array, mean: Array, var: Array) -> Array:
    """``log N(x; mean, var)`` (``var`` is a variance)."""
    return -0.5 * (LOG_2PI + jnp.log(var) + (x - mean) ** 2 / var)


def log_inv_gamma_from_log(log_x: Array, shape: Array, scale: Array) -> Array:
    """``log IG(x; shape, scale)`` at ``x = exp(log_x)``, density ``∝ x^{−shape−1} e^{−scale/x}``."""
    return shape * jnp.log(scale) - gammaln(shape) - (shape + 1.0) * log_x - scale * jnp.exp(-log_x)


def log_beta_fn(a: Array, b: Array) -> Array:
    """``log B(a, b) = lnΓ(a) + lnΓ(b) − lnΓ(a + b)``. (``jax.scipy.special.betaln`` is off by up to
    6e-7 in float64 at ``(10, 2)`` in jax 0.12, so it is not used.)"""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def log_beta_from_logs(log_u: Array, log1m_u: Array, a: Array, b: Array) -> Array:
    """``log Beta(u; a, b)`` from ``log u`` and ``log(1 − u)``."""
    return (a - 1.0) * log_u + (b - 1.0) * log1m_u - log_beta_fn(a, b)


def _log_phi_prior(phi: Array, a: Array, b: Array) -> Array:
    """Density of φ when ``(φ+1)/2 ~ Beta(a, b)``: ``Beta((φ+1)/2)·½``; ``−∞`` for ``|φ| ≥ 1``."""
    inside = jnp.abs(phi) < 1.0
    phi_s = jnp.where(inside, phi, 0.0)
    lp = log_beta_from_logs(
        jnp.log1p(phi_s) - math.log(2.0), jnp.log1p(-phi_s) - math.log(2.0), a, b
    ) - math.log(2.0)
    return jnp.where(inside, lp, -jnp.inf)


def _log_sigma_prior(sigma: Array, core: Any, kind: str) -> Array:
    """Density of σ > 0: ``σ² ~ B_σ·χ²₁ ⇔ σ ~ |N(0, B_σ)|`` (scaled_chi2), or
    ``σ² ~ IG(c₀, C₀)`` mapped to σ (Jacobian ``2σ``; inv_gamma). ``−∞`` for σ ≤ 0."""
    pos = sigma > 0.0
    s = jnp.where(pos, sigma, 1.0)
    if kind == "scaled_chi2":
        lp = math.log(2.0) + log_normal(s, 0.0, core.B_sigma)
    else:
        lp = log_inv_gamma_from_log(2.0 * jnp.log(s), core.c0, core.C0) + math.log(2.0) + jnp.log(s)
    return jnp.where(pos, lp, -jnp.inf)


# --------------------------------------------------------------------------- draws


def _normal(key: Array, shape: tuple[int, ...] = ()) -> Array:
    return random.normal(key, shape, dtype=_F64)


def _log_gamma(key: Array, a: Array) -> Array:
    """``log G``, ``G ~ Gamma(a, 1)`` (accurate for small shapes)."""
    return random.loggamma(key, jnp.asarray(a, _F64), dtype=_F64)


def _log_beta_pair(key: Array, a: Array, b: Array) -> tuple[Array, Array]:
    """``(log u, log(1 − u))`` for ``u ~ Beta(a, b)``, from two log-gamma draws."""
    g1 = _log_gamma(random.fold_in(key, 0), a)
    g2 = _log_gamma(random.fold_in(key, 1), b)
    tot = jnp.logaddexp(g1, g2)
    return g1 - tot, g2 - tot


def _draw_sigma(key: Array, core: Any, kind: str) -> Array:
    """σ from its prior: ``√B_σ·|z|`` or ``exp(½(log C₀ − log G))``, ``G ~ Gamma(c₀)``."""
    if kind == "scaled_chi2":
        return jnp.sqrt(core.B_sigma) * jnp.abs(_normal(key))
    return jnp.exp(0.5 * (jnp.log(core.C0) - _log_gamma(key, core.c0)))


def _masked_path(
    key: Array, h0: Array, step: Callable[[Array, Array], Array], tmask: Array
) -> Array:
    """States ``h_0..h_{T_pad}``: ``h_t = step(h_{t−1}, η_t)`` while ``m_{t−1} = 1``, else an
    independent ``N(0, 1)`` dummy."""
    eta = _normal(key, tmask.shape)

    def body(h_prev: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
        e, m = inp
        h = jnp.where(m > 0, step(h_prev, e), e)
        return h, h

    _, path = jax.lax.scan(body, h0, (eta, tmask))
    return jnp.concatenate([h0[None], path])


def t_scale_constant(nu: Array, static: StaticConfig) -> Array:
    """``c(ν) = ν − 2`` (standardized) or ``ν`` (unit) (N03 §1, D-8)."""
    return nu - 2.0 if static.vol.t_scale == "standardized" else nu


def nu_grid(static: StaticConfig) -> Array:
    """The static ν grid as float64 (built at trace time, never a module-level array)."""
    return jnp.asarray(np.asarray(static.vol.nu_grid, dtype=np.float64))


# --------------------------------------------------------------------------- small helpers


@dataclass(frozen=True)
class PriorTails:
    """Tail indices of the prior that decide the finite-variance flags (N06 §3.4; minimum over
    equations): ``E[λ^k] < ∞`` iff ``k < lam`` (IG shape of the Constant core);
    ``E[(σ²)^k] < ∞`` iff ``k < sigma2`` (``c₀`` under ``inv_gamma``, ``∞`` under scaled χ²);
    ``E[h_t^{2k}] < ∞`` iff ``k < h2``. A λ-scaled coefficient ``x = m + √λ·s·z`` has
    ``E[x^{2k}] < ∞`` iff ``k < lam``. The default (all 0) assumes nothing: every prior-dependent
    function is then flagged infinite."""

    lam: float = 0.0
    sigma2: float = 0.0
    h2: float = 0.0


_NO_TAILS = PriorTails()


def _host_min(x: Any, label: str) -> float:
    try:
        return float(np.min(np.asarray(x)))
    except jax.errors.TracerArrayConversionError:
        raise ValueError(
            f"prior_tails needs concrete prior arrays ({label} is traced); pass tails= explicitly"
        ) from None


def prior_tails(static: StaticConfig, prior: PriorArrays) -> PriorTails:
    """:class:`PriorTails` of a concrete ``prior`` (host side).

    Constant: ``lam = min shape``, ``h2 = ∞`` (``log λ`` has every moment). SV: ``sigma2 = min c₀``
    (``inv_gamma``) or ``∞``; stationary ``h_t | θ ~ N(μ, σ²/(1 − φ²))`` and
    ``E[(1 − φ²)^{−k}] = E[(4u(1 − u))^{−k}] < ∞`` iff ``k < min(a_φ, b_φ)``, so
    ``h2 = min(sigma2, a_φ, b_φ)``; fixed ``B₀`` and random walk: ``Var(h_t | θ) ≤ σ²(B₀ + t)``
    (or ``V₀ + tσ²``), so ``h2 = sigma2``."""
    core, vol = prior.eq.vol_core, static.vol
    if vol.persistent == "constant":
        return PriorTails(lam=_host_min(core.shape, "shape"), sigma2=math.inf, h2=math.inf)
    sigma2 = math.inf if vol.sigma2_prior == "scaled_chi2" else _host_min(core.c0, "c0")
    if vol.rw or vol.h0 == "fixed_b0":
        h2 = sigma2
    else:
        h2 = min(sigma2, _host_min(core.a_phi, "a_phi"), _host_min(core.b_phi, "b_phi"))
    return PriorTails(lam=math.inf, sigma2=sigma2, h2=h2)


def _t_eff(tmask: Array) -> Array:
    return jnp.round(jnp.sum(tmask)).astype(jnp.int32)


def _masked_mean(x: Array, tmask: Array) -> Array:
    return jnp.sum(jnp.where(tmask > 0, x, 0.0)) / jnp.sum(tmask)


def _sv_dates(h: Array, tmask: Array) -> dict[str, Array]:
    """Latent states at fixed dates (N06 §3.4): ``h_0, h_1, h_⌈T/2⌉, h_T`` and the path mean."""
    T = _t_eff(tmask)
    path = h[1:]
    return {
        "h0": h[0],
        "h1": h[1],
        "hmid": jnp.take(h, (T + 1) // 2),
        "hT": jnp.take(h, T),
        "mean_h": _masked_mean(path, tmask),
        "mean_abs_h": _masked_mean(jnp.abs(path), tmask),
    }


# --------------------------------------------------------------------------- the protocol


@runtime_checkable
class Reference(Protocol):
    """An independent reference for one component kind (DESIGN §9.1)."""

    kind: ClassVar[str]

    def prior_sample_eq(
        self,
        key: Array,
        i: Array,
        prior_i: EqPrior,
        static: StaticConfig,
        tmask: Array,
        T_eff: Array,
        *,
        log_scale: Array | float = 0.0,
    ) -> PyTree: ...

    def prior_sample_global(
        self, key: Array, prior: PriorArrays, static: StaticConfig
    ) -> PyTree: ...

    def log_prior_eq(
        self,
        part: PyTree,
        prior_i: EqPrior,
        static: StaticConfig,
        tmask: Array,
        *,
        log_scale: Array | float = 0.0,
    ) -> Array: ...

    def log_prior_global(self, part: PyTree, prior: PriorArrays, static: StaticConfig) -> Array: ...

    def latent_log_var_eq(
        self, vol_part: PyTree, which: Which = "effective", *, T_pad: int | None = None
    ) -> Array:
        """``f64[T_pad]``; a time-constant core returns a scalar unless ``T_pad`` is given."""
        ...

    def test_functions_eq(
        self, part: PyTree, tmask: Array, *, i: int | None = None, tails: PriorTails = ...
    ) -> TestFns:
        """``{name: (value, finite variance under p(θ, y))}``; ``i`` (static) drops inactive slots;
        the flags follow ``tails`` (default: nothing assumed)."""
        ...

    def test_functions_global(self, part: PyTree, *, tails: PriorTails = ...) -> TestFns: ...


REFERENCES: dict[str, type] = {}


def register_reference(kind: str) -> Callable[[type], type]:
    """Class decorator: register ``cls`` as the reference of component ``kind`` (append-only)."""

    def deco(cls: type) -> type:
        if kind in REFERENCES:
            raise ValueError(f"a reference for kind {kind!r} is already registered")
        cls.kind = kind  # type: ignore[attr-defined]
        REFERENCES[kind] = cls
        return cls

    return deco


def reference(kind: str, static: StaticConfig | None = None) -> Reference:
    """An instance of the reference registered for ``kind`` (``static`` is needed only by the t
    references' test functions)."""
    try:
        return REFERENCES[kind](static)
    except KeyError:
        raise KeyError(
            f"no reference for kind {kind!r}; registered: {sorted(REFERENCES)}"
        ) from None


class _Base:
    """Defaults for parts without state, density, latent variance or test functions.

    Instances are stateless apart from an optional ``StaticConfig`` (needed by the t references
    for the static ν grid in their test functions)."""

    kind: ClassVar[str]

    def __init__(self, static: StaticConfig | None = None) -> None:
        self._static = static

    def _need(self) -> StaticConfig:
        if self._static is None:
            raise ValueError(
                f"the {self.kind!r} reference needs the StaticConfig: reference(kind, static)"
            )
        return self._static

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0) -> PyTree:
        return Empty()

    def prior_sample_global(self, key, prior, static) -> PyTree:
        return Empty()

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0) -> Array:
        return jnp.zeros((), _F64)

    def log_prior_global(self, part, prior, static) -> Array:
        return jnp.zeros((), _F64)

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad: int | None = None) -> Array:
        return jnp.zeros((), _F64) if T_pad is None else jnp.zeros((T_pad,), _F64)

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS) -> TestFns:
        return {}

    def test_functions_global(self, part, *, tails=_NO_TAILS) -> TestFns:
        return {}

    def _scaled(self, which: Literal["contemp", "coef"]) -> bool:
        """Whether this composition λ-scales the ``a`` (``contemp``) or ``δ`` (``coef``) prior."""
        st = self._need()
        if which == "contemp":
            return st.struct.contemp_scaling == "lambda"
        return st.mean.coef_scaling == "lambda"


# --------------------------------------------------------------------------- structure and means


@register_reference("recursive")
class RecursiveRef(_Base):
    """``a_ij ~ N(m, 1/P̃)`` for ``j < i`` (DESIGN §6.1); ``× λ_i`` when
    ``contemp_scaling="lambda"`` (Mode II, ``iw_equivalent``; Chan 2022 Prop. 1)."""

    @staticmethod
    def _var(prior_i: EqPrior, static: StaticConfig, log_scale: Array | float) -> Array:
        var = 1.0 / prior_i.contemp.prec0
        return var * jnp.exp(log_scale) if static.struct.contemp_scaling == "lambda" else var

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        c = prior_i.contemp
        z = _normal(key, c.mean0.shape)
        a = (c.mean0 + jnp.sqrt(self._var(prior_i, static, log_scale)) * z) * c.cmask
        return RecursiveEq(a=jnp.where(c.cmask > 0, a, 0.0))

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        c = prior_i.contemp
        lp = log_normal(part.a, c.mean0, self._var(prior_i, static, log_scale))
        return jnp.sum(jnp.where(c.cmask > 0, lp, 0.0))

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        scaled = self._scaled("contemp")  # a = √λ·s·z: Var(a) needs E λ, Var(a²) needs E λ²
        out: TestFns = {}
        for j in range(part.a.shape[-1]):
            if i is not None and j >= i:
                continue
            out[f"a{j}"] = (part.a[j], not scaled or tails.lam > 1.0)
            out[f"a{j}^2"] = (part.a[j] ** 2, not scaled or tails.lam > 2.0)
        return out


@register_reference("linear_structural")
class LinearStructuralRef(_Base):
    """``δ_i ~ N(m_i, diag(1/P̃))`` in δ-coordinates (D3: ``γ_i = δ_i + M_c a_i``; N05 §4.1); the
    moments are those of the ``fixed_diagonal`` coefficient prior; ``× λ_i`` in Mode II."""

    @staticmethod
    def _var(prior_i: EqPrior, static: StaticConfig, log_scale: Array | float) -> Array:
        var = 1.0 / prior_i.coef.prec0
        return var * jnp.exp(log_scale) if static.mean.coef_scaling == "lambda" else var

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        cp = prior_i.coef
        z = _normal(key, cp.mean0.shape)
        return LinearEq(delta=cp.mean0 + jnp.sqrt(self._var(prior_i, static, log_scale)) * z)

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        var = self._var(prior_i, static, log_scale)
        return jnp.sum(log_normal(part.delta, prior_i.coef.mean0, var))

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        scaled = self._scaled("coef")  # Mode II: δ − m = √λ·s·z
        out: TestFns = {}
        for r in range(part.delta.shape[-1]):
            out[f"delta{r}"] = (part.delta[r], not scaled or tails.lam > 1.0)
            out[f"delta{r}^2"] = (part.delta[r] ** 2, not scaled or tails.lam > 2.0)
        return out


@register_reference("linear_reduced")
class LinearReducedRef(_Base):
    """Reduced form (CCCM 2022; N04 §3.1): ``π_j ~ N(m_j, diag(1/P̃_j))`` independent over ``j``;
    global state ``Π`` (``k × n``), no per-equation mean state. With SOC/SUR dummy rows (WP25)
    each ``π_j`` is conditioned on its rows (``validation.dummies``)."""

    def prior_sample_global(self, key, prior, static):
        if static.mean.n_dummy:  # π_j conditioned on the SOC/SUR rows (WP25; N04 §3.1)
            return reduced_dummy_prior_sample(key, prior)
        cp = prior.eq.coef
        z = _normal(key, cp.mean0.shape)
        return CCCMGlobal(Pi=(cp.mean0 + jnp.sqrt(1.0 / cp.prec0) * z).T)

    def log_prior_global(self, part, prior, static):
        cp = prior.eq.coef
        out = jnp.sum(log_normal(part.Pi.T, cp.mean0, 1.0 / cp.prec0))
        if static.mean.n_dummy:  # + log p(d | Π) − log p(d) (Bayes' rule; WP25)
            out = out + reduced_dummy_log_factor(part.Pi, prior)
        return out

    def test_functions_global(self, part, *, tails=_NO_TAILS):
        Pi = part.Pi  # Gaussian prior, never λ-scaled (CCCM 2022): every moment finite
        k, n = Pi.shape
        out: TestFns = {}
        for j in range(n):
            for r in range(k):
                out[f"Pi[{r},{j}]"] = (Pi[r, j], True)
                out[f"Pi[{r},{j}]^2"] = (Pi[r, j] ** 2, True)
        for j1 in range(n):  # cross-equation products π_{l,j1}·π_{l,j2} (DESIGN §9.5)
            for j2 in range(j1 + 1, n):
                for r in range(k):
                    out[f"Pi[{r},{j1}]*Pi[{r},{j2}]"] = (Pi[r, j1] * Pi[r, j2], True)
        return out


@register_reference("zero")
class ZeroRef(_Base):
    """``ZeroMean``: ``g ≡ 0``, no coefficients."""


@register_reference("fixed_diagonal")
class FixedDiagonalRef(_Base):
    """``FixedDiagonal`` coefficient prior: no state of its own; its moments ``(mean0, prec0)``
    are the δ / π prior used by the mean references."""


# --------------------------------------------------------------------------- persistent cores


@register_reference("constant")
class ConstantRef(_Base):
    """``λ_i ~ IG(shape, scale)`` (DESIGN §6.1; density of λ, state ``log λ``)."""

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        cp = prior_i.vol_core
        return ConstantEq(log_lam=jnp.log(cp.scale) - _log_gamma(key, cp.shape))

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        cp = prior_i.vol_core
        return log_inv_gamma_from_log(part.log_lam, cp.shape, cp.scale)

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        lv = jnp.asarray(vol_part.log_lam, _F64)
        return lv if T_pad is None else jnp.broadcast_to(lv, (T_pad,))

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        # log λ = log scale − log G, G ~ Gamma: every moment finite for any shape
        return {"log_lam": (part.log_lam, True), "log_lam^2": (part.log_lam**2, True)}


def _zeros_like_sv(n_states: int) -> dict[str, Array]:
    z = jnp.zeros((), _F64)
    zi = jnp.zeros((), jnp.int32)
    zh = jnp.zeros((n_states,), _F64)
    return {"f": z, "i": zi, "h": zh}


@register_reference("sv")
class SVRef(_Base):
    """Stationary AR(1) SV (N02 §2.1, D-13): ``μ ~ N(b_μ, B_μ)``, ``(φ+1)/2 ~ Beta(a_φ, b_φ)``,
    ``σ² ~ B_σ·χ²₁`` or ``IG(c₀, C₀)``, ``h_0 ~ N(μ, σ²/ω₀)`` with ``ω₀ = 1 − φ²`` (stationary) or
    ``1/B₀`` (``h0="fixed_b0"``), ``h_t = μ + φ(h_{t−1} − μ) + ση_t`` for ``t = 1..T_eff``."""

    @staticmethod
    def _h0_var(phi: Array, sigma: Array, prior_i: EqPrior, static: StaticConfig) -> Array:
        if static.vol.h0 == "stationary":
            return sigma**2 / ((1.0 - phi) * (1.0 + phi))
        return prior_i.vol_core.B0 * sigma**2

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        cp = prior_i.vol_core
        k = [random.fold_in(key, j) for j in range(5)]
        mu = cp.b_mu + jnp.sqrt(cp.B_mu) * _normal(k[0])
        log_u, log1m_u = _log_beta_pair(k[1], cp.a_phi, cp.b_phi)
        phi = jnp.exp(log_u) - jnp.exp(log1m_u)  # 2u − 1, accurate near ±1
        sigma = _draw_sigma(k[2], cp, static.vol.sigma2_prior)
        h0 = mu + jnp.sqrt(self._h0_var(phi, sigma, prior_i, static)) * _normal(k[3])
        h = _masked_path(k[4], h0, lambda hp, e: mu + phi * (hp - mu) + sigma * e, tmask)
        z = _zeros_like_sv(h.shape[0])
        return SVEq(
            h=h, mu=mu, phi=phi, sigma=sigma,
            acc_h=z["f"], acc_cgp=z["f"], acc_cs2=z["f"], acc_ncphi=z["f"], acc_ncms=z["f"],
            acc_fold=z["f"], acc_floor=z["f"], acc_rej_run=z["i"], acc_rej_run_max=z["i"],
            rb_h_sum=z["h"], rb_h_sq=z["h"],
        )  # fmt: skip

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        cp = prior_i.vol_core
        mu, phi, sigma, h = part.mu, part.phi, part.sigma, part.h
        lp = log_normal(mu, cp.b_mu, cp.B_mu)
        lp += _log_phi_prior(phi, cp.a_phi, cp.b_phi)
        lp += _log_sigma_prior(sigma, cp, static.vol.sigma2_prior)
        stationary_ok = jnp.abs(phi) < 1.0
        phi_s = jnp.where(stationary_ok, phi, 0.0)
        sig_s = jnp.where(sigma > 0, sigma, 1.0)
        lp += log_normal(h[0], mu, self._h0_var(phi_s, sig_s, prior_i, static))
        trans = log_normal(h[1:], mu + phi_s * (h[:-1] - mu), sig_s**2)
        lp += jnp.sum(jnp.where(tmask > 0, trans, 0.0))
        return lp

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        return vol_part.h[1:]

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        mu, phi, sigma = part.mu, part.phi, part.sigma
        d = _sv_dates(part.h, tmask)
        s1, s2 = tails.sigma2 > 1.0, tails.sigma2 > 2.0  # E σ², E σ⁴ finite
        out: TestFns = {
            "mu": (mu, True), "phi": (phi, True), "sigma": (sigma, s1),
            "mu^2": (mu**2, True), "phi^2": (phi**2, True), "sigma^2": (sigma**2, s2),
            "mu*phi": (mu * phi, True), "phi*sigma": (phi * sigma, s1),
        }  # fmt: skip
        out.update({name: (val, tails.h2 > 1.0) for name, val in d.items()})
        out["mu*hT"] = (mu * d["hT"], tails.h2 > 1.0)  # hT − μ is independent of μ
        # E[h⁴] ∝ E[(1−φ)^{−2}] diverges for b_φ ≤ 2 (N06 §3.4): then h² is not a valid Geweke
        # function (the test prior has b_φ = 2).
        out["hT^2"] = (d["hT"] ** 2, tails.h2 > 2.0)
        return out


@register_reference("sv_rw")
class SVRWRef(_Base):
    """Random-walk SV (N02 §8.1): ``h_0 ~ N(m₀, V₀)``, ``h_t = h_{t−1} + ση_t``, σ as for AR SV."""

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        cp = prior_i.vol_core
        k = [random.fold_in(key, j) for j in range(3)]
        sigma = _draw_sigma(k[0], cp, static.vol.sigma2_prior)
        h0 = cp.m0 + jnp.sqrt(cp.V0) * _normal(k[1])
        h = _masked_path(k[2], h0, lambda hp, e: hp + sigma * e, tmask)
        z = _zeros_like_sv(h.shape[0])
        return SVRWEq(
            h=h, sigma=sigma, acc_h=z["f"], acc_cs2=z["f"], acc_ncms=z["f"], acc_fold=z["f"],
            acc_floor=z["f"], acc_rej_run=z["i"], acc_rej_run_max=z["i"], rb_h_sum=z["h"],
            rb_h_sq=z["h"],
        )  # fmt: skip

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        cp = prior_i.vol_core
        sigma, h = part.sigma, part.h
        sig_s = jnp.where(sigma > 0, sigma, 1.0)
        lp = log_normal(h[0], cp.m0, cp.V0) + _log_sigma_prior(sigma, cp, static.vol.sigma2_prior)
        trans = log_normal(h[1:], h[:-1], sig_s**2)
        return lp + jnp.sum(jnp.where(tmask > 0, trans, 0.0))

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        return vol_part.h[1:]

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        sigma = part.sigma
        d = _sv_dates(part.h, tmask)
        s1, s2 = tails.sigma2 > 1.0, tails.sigma2 > 2.0
        out: TestFns = {"sigma": (sigma, s1), "sigma^2": (sigma**2, s2)}
        out.update({name: (val, tails.h2 > 1.0) for name, val in d.items()})
        out["sigma*hT"] = (sigma * d["hT"], s2)  # E σ²h_T² = E σ²h₀² + T·E σ⁴ (+ cross terms)
        return out


# --------------------------------------------------------------------------- transitory scales


def outlier_log_weights(log_p: Array, log1m_p: Array, K: int) -> Array:
    """Prior pmf of ``o ∈ {1..K}`` in logs: ``w_1 = 1 − p``, ``w_k = p/(K−1)`` (N03 §2.1, D-9)."""
    rest = jnp.broadcast_to(log_p - math.log(K - 1), (*jnp.shape(log_p), K - 1))
    return jnp.concatenate([jnp.asarray(log1m_p)[..., None], rest], axis=-1)


def _outlier_sample(key: Array, prior_i: EqPrior, static: StaticConfig, tmask: Array):
    tp = prior_i.transitory
    log_p, log1m_p = _log_beta_pair(random.fold_in(key, 0), tp.a_p, tp.b_p)
    logw = outlier_log_weights(log_p, log1m_p, static.vol.outlier_K)
    o_idx = random.categorical(random.fold_in(key, 1), logw, shape=tmask.shape).astype(jnp.int32)
    return log_p, log1m_p, jnp.where(tmask > 0, o_idx, 0)


def _outlier_log_prior(o_idx, log_p, log1m_p, prior_i, static, tmask) -> Array:
    tp = prior_i.transitory
    lp = log_beta_from_logs(log_p, log1m_p, tp.a_p, tp.b_p)
    lw = jnp.where(o_idx == 0, log1m_p, log_p - math.log(static.vol.outlier_K - 1))
    return lp + jnp.sum(jnp.where(tmask > 0, lw, 0.0))


def _t_sample(key: Array, prior_i: EqPrior, static: StaticConfig, tmask: Array):
    tp = prior_i.transitory
    nu_idx = random.categorical(random.fold_in(key, 0), tp.log_nu_w).astype(jnp.int32)
    nu = nu_grid(static)[nu_idx]
    c = t_scale_constant(nu, static)
    lg = _log_gamma(random.fold_in(key, 1), jnp.broadcast_to(0.5 * nu, tmask.shape))
    log_tau = jnp.log(0.5 * c) - lg  # τ ~ IG(ν/2, c(ν)/2) (D-8)
    return nu_idx, jnp.where(tmask > 0, log_tau, 0.0)


def _t_log_prior(log_tau, nu_idx, prior_i, static, tmask) -> Array:
    tp = prior_i.transitory
    nu = nu_grid(static)[nu_idx]
    c = t_scale_constant(nu, static)
    lp = tp.log_nu_w[nu_idx] - logsumexp(tp.log_nu_w)
    lt = log_inv_gamma_from_log(log_tau, 0.5 * nu, 0.5 * c)
    return lp + jnp.sum(jnp.where(tmask > 0, lt, 0.0))


def _outlier_fns(o_idx, log_p, tmask) -> TestFns:
    p = jnp.exp(log_p)
    n_out = jnp.sum(jnp.where(tmask > 0, (o_idx > 0).astype(_F64), 0.0))
    log_o = jnp.log(o_idx.astype(_F64) + 1.0)
    return {
        "p": (p, True), "p^2": (p**2, True), "n_out": (n_out, True),
        "share_out": (n_out / jnp.sum(tmask), True), "p*n_out": (p * n_out, True),
        "mean_log_o": (_masked_mean(log_o, tmask), True),
        "mean_o": (_masked_mean(o_idx.astype(_F64) + 1.0, tmask), True),
    }  # fmt: skip


def _t_fns(log_tau, nu_idx, static: StaticConfig, tmask) -> TestFns:
    nu = nu_grid(static)[nu_idx]
    mlt = _masked_mean(log_tau, tmask)
    return {
        "nu": (nu, True), "nu^2": (nu**2, True), "nu_le_5": ((nu <= 5.0).astype(_F64), True),
        "mean_log_tau": (mlt, True), "nu*mean_log_tau": (nu * mlt, True),
        "mean_inv_tau": (_masked_mean(jnp.exp(-log_tau), tmask), True),
    }  # fmt: skip


@register_reference("none")
class NoTransitoryRef(_Base):
    """No transitory scale: ``o ≡ 1``, ``τ ≡ 1``."""


@register_reference("outlier")
class OutlierRef(_Base):
    """Outliers (N03 §2.1, D-9): ``p ~ Beta(a_p, b_p)``; ``o_t = 1`` w.p. ``1 − p``, else uniform
    on ``{2..K}``; state ``o_idx = o − 1``. Effective log-variance contribution ``2 log o``."""

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        log_p, log1m_p, o_idx = _outlier_sample(key, prior_i, static, tmask)
        return OutlierEq(
            o_idx=o_idx, log_p=log_p, log1m_p=log1m_p, rb_out=jnp.zeros(tmask.shape, _F64)
        )

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        return _outlier_log_prior(part.o_idx, part.log_p, part.log1m_p, prior_i, static, tmask)

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        if which == "persistent":
            return jnp.zeros(vol_part.o_idx.shape, _F64)
        return 2.0 * jnp.log(vol_part.o_idx.astype(_F64) + 1.0)

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        return _outlier_fns(part.o_idx, part.log_p, tmask)


@register_reference("t")
class TRef(_Base):
    """Student-t (N03 §2.1, D-8): ``ν ~ π`` on the static grid; ``τ_t ~ IG(ν/2, c(ν)/2)``,
    ``c(ν) = ν − 2`` (standardized, ``E τ = 1``) or ``ν`` (unit). Contribution ``log τ``."""

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        nu_idx, log_tau = _t_sample(key, prior_i, static, tmask)
        return TEq(log_tau=log_tau, nu_idx=nu_idx, acc_nu_move=jnp.zeros((), _F64))

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        return _t_log_prior(part.log_tau, part.nu_idx, prior_i, static, tmask)

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        if which == "persistent":
            return jnp.zeros(vol_part.log_tau.shape, _F64)
        return vol_part.log_tau

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        return _t_fns(part.log_tau, part.nu_idx, self._need(), tmask)


@register_reference("outlier_t")
class OutlierTRef(_Base):
    """Outliers and Student-t together (N03 §2.1): ``λ_eff = e^h o² τ``."""

    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        log_p, log1m_p, o_idx = _outlier_sample(random.fold_in(key, 0), prior_i, static, tmask)
        nu_idx, log_tau = _t_sample(random.fold_in(key, 1), prior_i, static, tmask)
        return OutlierTEq(
            o_idx=o_idx, log_p=log_p, log1m_p=log1m_p, log_tau=log_tau, nu_idx=nu_idx,
            rb_out=jnp.zeros(tmask.shape, _F64), acc_nu_move=jnp.zeros((), _F64),
        )  # fmt: skip

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        return _outlier_log_prior(
            part.o_idx, part.log_p, part.log1m_p, prior_i, static, tmask
        ) + _t_log_prior(part.log_tau, part.nu_idx, prior_i, static, tmask)

    def latent_log_var_eq(self, vol_part, which="effective", *, T_pad=None):
        if which == "persistent":
            return jnp.zeros(vol_part.log_tau.shape, _F64)
        return 2.0 * jnp.log(vol_part.o_idx.astype(_F64) + 1.0) + vol_part.log_tau

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        out = _outlier_fns(part.o_idx, part.log_p, tmask)
        out.update(_t_fns(part.log_tau, part.nu_idx, self._need(), tmask))
        return out


# --------------------------------------------------------------------------- composition


@dataclass(frozen=True)
class SlotReferences:
    """The references of one composition, by slot (DESIGN §3.8 slots)."""

    structure: Reference
    mean: Reference
    coef: Reference
    core: Reference
    trans: Reference


def core_kind(static: StaticConfig) -> str:
    """Reference kind of the persistent core: ``"constant"``, ``"sv"`` or ``"sv_rw"``."""
    if static.vol.persistent == "constant":
        return "constant"
    return "sv_rw" if static.vol.rw else "sv"


def references_for(static: StaticConfig) -> SlotReferences:
    """Instantiate the references a ``StaticConfig`` composes."""

    def make(kind: str) -> Reference:
        if kind not in REFERENCES:
            raise KeyError(f"no reference registered for kind {kind!r} (DESIGN §9.1)")
        return REFERENCES[kind](static)

    return SlotReferences(
        structure=make(static.struct.kind),
        mean=make(static.mean.kind),
        coef=make(static.coef.kind),
        core=make(core_kind(static)),
        trans=make(static.vol.transitory),
    )


def effective_log_var(
    vol: VolEq, static: StaticConfig, which: Which = "effective", refs: SlotReferences | None = None
) -> Array:
    """``log λ_eff,i,t = log λ^pers_i,t + 2 log o_i,t + log τ_i,t`` (``which="effective"``) or the
    persistent part only; shape ``(..., n, T_pad)`` for a batched ``VolEq`` (leading equation axis),
    computed equation by equation with the references."""
    refs = refs or references_for(static)
    T_pad = static.shapes.T_pad

    def one(v: VolEq) -> Array:
        lv = refs.core.latent_log_var_eq(v.core, which, T_pad=T_pad)
        return lv + refs.trans.latent_log_var_eq(v.trans, which, T_pad=T_pad)

    return jax.vmap(one)(vol)


def coupling_log_scale(vol_core: Any, static: StaticConfig) -> Array | float:
    """``log λ_i`` that scales the λ-scaled prior blocks (Constant core, ``coupled``), else 0."""
    if static.vol.coupled:
        return vol_core.log_lam
    return 0.0


def transitory_log_density(log_z2: Array, trans: Any, static: StaticConfig) -> Array:
    """Transitory-integrated ``log f(z)`` of N03 §2.2 (written from the table, in ``log z²``):
    none ``φ(z)``; outlier ``Σ_k w_k k⁻¹ φ(z/k)``; t ``t_{c,ν}(z)``; outlier-t
    ``Σ_k w_k k⁻¹ t_{c,ν}(z/k)``. Shapes broadcast over ``log_z2``."""
    kind = static.vol.transitory
    log_z2 = jnp.asarray(log_z2, _F64)
    if kind in ("outlier", "outlier_t"):
        K = static.vol.outlier_K
        ks = jnp.arange(1, K + 1, dtype=_F64)
        logw = outlier_log_weights(trans.log_p, trans.log1m_p, K)
        log_zk2 = log_z2[..., None] - 2.0 * jnp.log(ks)  # log (z/k)²
    else:
        log_zk2 = log_z2[..., None]
    if kind in ("t", "outlier_t"):
        nu = nu_grid(static)[trans.nu_idx]
        c = t_scale_constant(nu, static)
        dens = (
            gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * jnp.log(jnp.pi * c)
            - 0.5 * (nu + 1.0) * jnp.log1p(jnp.exp(log_zk2) / c)
        )  # fmt: skip
    else:
        dens = -0.5 * LOG_2PI - 0.5 * jnp.exp(log_zk2)
    if kind in ("outlier", "outlier_t"):
        return logsumexp(logw - jnp.log(ks) + dens, axis=-1)
    return dens[..., 0]


__all__ = [
    "KSC_SHIFT",
    "LOG_2PI",
    "REFERENCES",
    "MixtureTableRef",
    "PriorTails",
    "Reference",
    "SlotReferences",
    "core_kind",
    "coupling_log_scale",
    "effective_log_var",
    "log_beta_fn",
    "log_beta_from_logs",
    "log_inv_gamma_from_log",
    "log_logchi2_density",
    "log_mixture_components",
    "log_mixture_density",
    "log_normal",
    "mixture_table",
    "nu_grid",
    "outlier_log_weights",
    "prior_tails",
    "reference",
    "references_for",
    "register_reference",
    "t_scale_constant",
    "transitory_log_density",
]
