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

Phase 3 (``docs/phase3/DESIGN.md`` §9.1, WP30; written from that document only, by an author who
wrote no Phase 3 kernel)
----------------------------------------------------------------------------------------------
* **Global-local and SSVS priors** (P3 §3.1–§3.2, D-36…D-39, D-44): kinds ``horseshoe``,
  ``normal_gamma``, ``r2d2``, ``dirichlet_laplace``, ``ssvs``. They govern the **deviations**
  ``d_j = coefficient − m_j`` of the lag slots (``CoefPriorArrays.gl.mask``), ``d_j | v_j ~
  N(0, r_j v_j)`` (SSVS: ``N(0, τ²_{γ_j, j})``); deterministic slots keep ``N(m, 1/prec0)``.
  Samplers draw every scale in log space (``log x = LG(α) − log β`` for a gamma with rate ``β``,
  ``log β − LG(α)`` for an inverse gamma with scale ``β``) and the deviations as
  ``log|d| = ½ log(r v) + log|z|``. Densities are those of the **prior definitions** of P3 §3.2
  (never the full conditionals), w.r.t. Lebesgue measure on the natural scales (HS ``ϑ, ν, ζ, τ``;
  NG ``ϖ, ξ``; R2D2 ``ϱ, ϖ, ξ``; DL ``θ̃, ϱ``; SSVS ``p``; the state stores their logs) and on
  the coefficients, with counting measure on the grid index ``a_idx`` and on ``γ``. The deviation
  density is evaluated from ``ℓ = dev_log`` (P3 §3.1: ``d²/(r v) = exp(2ℓ − log r − log v)``).
  Their per-equation ``Reference`` methods raise: the scales couple equations through the groups,
  so :mod:`validation.simulate` and :mod:`validation.log_joint` call the family API of
  :class:`GLReference` on the whole state.
* **CommonSV** (P3 §5.1, D-46): ``constant_common`` (Constant cores whose log variance gains the
  common ``h_t``; the ``λ_i`` prior is the Constant one, Mode I or coupled) and ``common_sv``
  (``glob.vol``: ``h_0 ~ N(0, σ_h²/(1 − ρ²))`` at the first data row, ``h_t = ρh_{t−1} + σ_hη_t``
  for ``1 ≤ t < T_eff``, ``h_t = 0`` for ``t ≥ T_eff``; ``ρ ~ N(ρ₀, V_ρ)·1(|ρ| < 1)`` or
  ``(ρ+1)/2 ~ Beta(a, b)``; ``σ_h² ~ IG(c₀, C₀)`` or ``B·χ²₁``). Its density is w.r.t. ``h``, ``ρ``
  and ``σ_h > 0`` (the state stores ``log σ_h``; the prior is stated on ``σ_h²``, as for SV's σ).
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
from jax.scipy.special import gammaln, logsumexp, ndtr

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    CoefPriorArrays,
    CommonSVGlobal,
    CommonSVPrior,
    ConstantEq,
    DLEq,
    DLGlobal,
    Empty,
    EqPrior,
    GLArrays,
    GLShared,
    HSEq,
    HSGlobal,
    LinearEq,
    NGEq,
    NGGlobal,
    OutlierEq,
    OutlierTEq,
    PriorArrays,
    R2D2Eq,
    R2D2Global,
    RecursiveEq,
    SSVSEq,
    SSVSGlobal,
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
    function is then flagged infinite.

    Phase 3: ``coef`` is the tail index of the deviations of a varying (global-local or SSVS)
    coefficient prior, ``E|d_j|^k < ∞`` iff ``k < coef`` (HS 1; NG and R2D2 ``2b``; DL and SSVS
    ``∞``); it is 0 and never read for a fixed prior. Under CommonSV ``sigma2`` and ``h2`` describe
    ``σ_h²`` and the common ``h``."""

    lam: float = 0.0
    sigma2: float = 0.0
    h2: float = 0.0
    coef: float = 0.0


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
    (or ``V₀ + tσ²``), so ``h2 = sigma2``.

    Phase 3. CommonSV: ``sigma2`` from the ``σ_h²`` prior; the stationary ``h_t | ρ, σ_h ~
    N(0, σ_h²/(1 − ρ²))`` has ``E h^{2k} < ∞`` iff ``k < sigma2`` and ``E[(1 − ρ²)^{−k}] < ∞``: under
    ``(ρ+1)/2 ~ Beta(a, b)`` iff ``k < min(a, b)``; under the truncated normal, whose density is
    positive at ``ρ = ±1``, iff ``k < 1``. Varying coefficient priors: ``coef`` (class docstring)."""
    core, vol = prior.eq.vol_core, static.vol
    coef = _coef_tail(static, prior)
    if vol.persistent == "constant":
        lam = _host_min(core.shape, "shape")
        if vol.common:
            sigma2, h2 = _common_tails(static, prior.shared.csv)
            return PriorTails(lam=lam, sigma2=sigma2, h2=h2, coef=coef)
        return PriorTails(lam=lam, sigma2=math.inf, h2=math.inf, coef=coef)
    sigma2 = math.inf if vol.sigma2_prior == "scaled_chi2" else _host_min(core.c0, "c0")
    if vol.rw or vol.h0 == "fixed_b0":
        h2 = sigma2
    else:
        h2 = min(sigma2, _host_min(core.a_phi, "a_phi"), _host_min(core.b_phi, "b_phi"))
    return PriorTails(lam=math.inf, sigma2=sigma2, h2=h2, coef=coef)


def _coef_tail(static: StaticConfig, prior: PriorArrays) -> float:
    """Tail index of the GL deviations (``PriorTails.coef``; 0 for a fixed prior). HS: ``√ϑ`` and
    ``√ζ`` are half-Cauchy, so ``E|d|^k < ∞`` iff ``k < 1``. NG: ``v = ϑζ`` with ``ζ⁻¹ ~ G(b, c)``;
    R2D2: ``v = ϱϖ/2`` with ``ϖ | ξ ~ G(a, ξ)``, ``ξ ~ G(b, 1)``; in both ``E v^m < ∞`` iff
    ``m < b``, so ``k < 2b``. DL (gamma and exponential mixing) and SSVS (two normals): every moment."""
    kind = static.coef.kind
    if not static.coef.varies:
        return 0.0
    if kind == "horseshoe":
        return 1.0
    if kind in ("normal_gamma", "r2d2"):
        return 2.0 * _host_min(prior.shared.gl.b, "b")
    return math.inf


def _common_tails(static: StaticConfig, csv: CommonSVPrior) -> tuple[float, float]:
    """``(sigma2, h2)`` of the CommonSV prior (:func:`prior_tails`)."""
    vol = static.vol
    if vol.csv_sigma2_prior == "scaled_chi2":
        sigma2 = math.inf
    else:
        sigma2 = _host_min(csv.s2_shape, "s2_shape")
    if vol.csv_rho_prior == "beta":
        rho = min(_host_min(csv.rho_a, "rho_a"), _host_min(csv.rho_b, "rho_b"))
    else:
        rho = 1.0
    return sigma2, min(sigma2, rho)


def _ok(heavy: bool, tails: PriorTails, k: float) -> bool:
    """Finite-variance flag of a test function whose variance needs ``E|d|^k < ∞`` when it reads
    a GL deviation (``heavy``); ``True`` otherwise (P3)."""
    return (not heavy) or tails.coef > k


def _fixed_prior_only(static: StaticConfig, kind: str) -> None:
    """The Phase 2 mean references draw and score the coefficients from the fixed moments; under
    a varying (GL) prior the coefficients belong to :class:`GLReference` (P3 §3.1)."""
    if static.coef.varies:
        raise ValueError(
            f"the {kind!r} reference uses the fixed coefficient moments; under the varying "
            f"{static.coef.kind!r} prior use validation.simulate.prior_sample / "
            "validation.log_joint.log_prior (GLReference)"
        )


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
        _fixed_prior_only(static, "linear_structural")
        cp = prior_i.coef
        z = _normal(key, cp.mean0.shape)
        return LinearEq(delta=cp.mean0 + jnp.sqrt(self._var(prior_i, static, log_scale)) * z)

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        _fixed_prior_only(static, "linear_structural")
        var = self._var(prior_i, static, log_scale)
        return jnp.sum(log_normal(part.delta, prior_i.coef.mean0, var))

    def test_functions_eq(self, part, tmask, *, i=None, tails=_NO_TAILS):
        scaled = self._scaled("coef")  # Mode II: δ − m = √λ·s·z
        gl = self._gl_lags()  # P3: lag slots carry the GL deviations' tails
        out: TestFns = {}
        for r in range(part.delta.shape[-1]):
            heavy = r >= gl
            out[f"delta{r}"] = (
                part.delta[r],
                (not scaled or tails.lam > 1.0) and _ok(heavy, tails, 2),
            )
            out[f"delta{r}^2"] = (
                part.delta[r] ** 2,
                (not scaled or tails.lam > 2.0) and _ok(heavy, tails, 4),
            )
        return out

    def _gl_lags(self) -> int:
        """First GL slot (``n_det``) under a varying coefficient prior, else past every slot."""
        st = self._static
        if st is None or not st.coef.varies:
            return 1 << 30
        return st.shapes.n_det


@register_reference("linear_reduced")
class LinearReducedRef(_Base):
    """Reduced form (CCCM 2022; N04 §3.1): ``π_j ~ N(m_j, diag(1/P̃_j))`` independent over ``j``;
    global state ``Π`` (``k × n``), no per-equation mean state. With SOC/SUR dummy rows (WP25)
    each ``π_j`` is conditioned on its rows (``validation.dummies``)."""

    def prior_sample_global(self, key, prior, static):
        _fixed_prior_only(static, "linear_reduced")
        if static.mean.n_dummy:  # π_j conditioned on the SOC/SUR rows (WP25; N04 §3.1)
            return reduced_dummy_prior_sample(key, prior)
        cp = prior.eq.coef
        z = _normal(key, cp.mean0.shape)
        return CCCMGlobal(Pi=(cp.mean0 + jnp.sqrt(1.0 / cp.prec0) * z).T)

    def log_prior_global(self, part, prior, static):
        _fixed_prior_only(static, "linear_reduced")
        cp = prior.eq.coef
        out = jnp.sum(log_normal(part.Pi.T, cp.mean0, 1.0 / cp.prec0))
        if static.mean.n_dummy:  # + log p(d | Π) − log p(d) (Bayes' rule; WP25)
            out = out + reduced_dummy_log_factor(part.Pi, prior)
        return out

    def test_functions_global(self, part, *, tails=_NO_TAILS):
        Pi = part.Pi  # Gaussian prior, never λ-scaled (CCCM 2022): every moment finite
        k, n = Pi.shape
        st = self._static  # P3: lag rows of Π carry the GL deviations' tails
        gl = st.shapes.n_det if st is not None and st.coef.varies else k
        out: TestFns = {}
        for j in range(n):
            for r in range(k):
                out[f"Pi[{r},{j}]"] = (Pi[r, j], _ok(r >= gl, tails, 2))
                out[f"Pi[{r},{j}]^2"] = (Pi[r, j] ** 2, _ok(r >= gl, tails, 4))
        for j1 in range(n):  # cross-equation products π_{l,j1}·π_{l,j2} (DESIGN §9.5)
            for j2 in range(j1 + 1, n):
                for r in range(k):
                    # GL: the two share group scales, so E[π₁²π₂²] needs E v² < ∞ (coef > 4)
                    fin = _ok(r >= gl, tails, 4)
                    out[f"Pi[{r},{j1}]*Pi[{r},{j2}]"] = (Pi[r, j1] * Pi[r, j2], fin)
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


# --------------------------------------------------------------------------- P3: GL priors

LOG_2 = math.log(2.0)


def log_gamma_from_logs(log_x: Array, shape: Array, log_rate: Array) -> Array:
    """``log G(x; shape, rate)`` at ``x = exp(log_x)`` (density of ``x``), rate given in logs."""
    return shape * log_rate - gammaln(shape) + (shape - 1.0) * log_x - jnp.exp(log_rate + log_x)


def log_inv_gamma_from_logs(log_x: Array, shape: Array, log_scale: Array) -> Array:
    """``log IG(x; shape, scale)`` at ``x = exp(log_x)`` (density of ``x``), scale given in logs."""
    return shape * log_scale - gammaln(shape) - (shape + 1.0) * log_x - jnp.exp(log_scale - log_x)


def _on(mask: Array, x: Array) -> Array:
    """``x`` on the GL slots, exactly 0 elsewhere (P3 §2.2 invariant)."""
    return jnp.where(mask > 0, x, jnp.zeros_like(x))


def _full(shape: tuple[int, ...], value: float) -> Array:
    return jnp.full(shape, value, _F64)


def a_values(a_idx: Array, gl: GLShared, static: StaticConfig) -> Array:
    """``a_g`` per group: the grid point ``a_support[a_idx]`` (``CoefPriorStatic.a_grid > 0``) or
    the fixed ``exp(log_a_fixed)``."""
    if static.coef.a_grid:
        return gl.a_support[a_idx]
    return jnp.broadcast_to(jnp.exp(gl.log_a_fixed), a_idx.shape)


def _a_log_prior(a_idx: Array, gl: GLShared, static: StaticConfig) -> Array:
    """``Σ_g log π(a_g)`` of the discrete grid hyperprior (normalised log weights); 0 for a fixed
    ``a``. Counting measure on the index."""
    if not static.coef.a_grid:
        return jnp.zeros((), _F64)
    return jnp.sum(gl.a_logw[a_idx] - logsumexp(gl.a_logw))


def _sample_a_idx(key: Array, gl: GLShared, static: StaticConfig) -> Array:
    G = static.coef.n_groups
    if not static.coef.a_grid:
        return jnp.zeros((G,), jnp.int32)
    return random.categorical(key, gl.a_logw, shape=(G,)).astype(jnp.int32)


def _segment_sum(x: Array, arr: GLArrays, G: int) -> Array:
    """``Σ_{j ∈ g} x_j`` over the GL slots of every equation (``x`` of shape ``(n, k)``)."""
    vals = jnp.where(arr.mask > 0, x, 0.0).reshape(-1)
    return jax.ops.segment_sum(vals, arr.group.reshape(-1), num_segments=G)


class GLReference(_Base):
    """Base of the global-local and SSVS references (P3 §3.2; module docstring).

    Family API, written elementwise so that it serves one equation (leaves of shape ``(k,)``,
    under ``vmap``) or every equation (``(n, k)``); ``arr`` is ``CoefPriorArrays.gl`` (or its
    equation slice), ``gl`` is ``SharedPrior.gl`` and ``glob`` the ``GlobalState.coef`` part:

    * :meth:`sample_global` / :meth:`sample_local`: the group-level and local scales;
    * :meth:`log_kernel_var`: ``log(r_j v_j)`` (SSVS ``2 log τ_{γ_j, j}``), 0 off the GL slots;
    * :meth:`local_log_density` (elementwise, 0 off the GL slots) and :meth:`global_log_density`;
    * :meth:`local_logs` (the local log scales for the test functions) and
      :meth:`group_functions` (group-level test functions, all finite: logs of gamma or inverse
      gamma variables, bounded indices and counts).
    """

    local_names: ClassVar[tuple[str, ...]] = ()

    # ---- Reference protocol: the per-equation methods cannot see the group scales
    def prior_sample_eq(self, key, i, prior_i, static, tmask, T_eff, *, log_scale=0.0):
        raise TypeError(
            f"the {self.kind!r} reference draws local scales given the group scales: use "
            "sample_local (validation.simulate.prior_sample composes it)"
        )

    def log_prior_eq(self, part, prior_i, static, tmask, *, log_scale=0.0):
        raise TypeError(
            f"the {self.kind!r} reference scores the scales jointly: use local_log_density / "
            "global_log_density (validation.log_joint.log_prior composes them)"
        )

    def prior_sample_global(self, key, prior, static):
        return self.sample_global(key, prior.shared.gl, static)

    def log_prior_global(self, part, prior, static):
        return self.global_log_density(part, prior.shared.gl, static)

    # ---- family API (overridden)
    def sample_global(self, key: Array, gl: GLShared, static: StaticConfig) -> PyTree:
        raise NotImplementedError

    def sample_local(
        self, key: Array, arr: GLArrays, glob: PyTree, gl: GLShared, static: StaticConfig
    ) -> PyTree:
        raise NotImplementedError

    def log_kernel_var(
        self, part: PyTree, arr: GLArrays, glob: PyTree, gl: GLShared, static: StaticConfig
    ) -> Array:
        raise NotImplementedError

    def local_log_density(
        self, part: PyTree, arr: GLArrays, glob: PyTree, gl: GLShared, static: StaticConfig
    ) -> Array:
        raise NotImplementedError

    def global_log_density(self, glob: PyTree, gl: GLShared, static: StaticConfig) -> Array:
        raise NotImplementedError

    def local_logs(self, part: PyTree) -> dict[str, Array]:
        return {name: getattr(part, name) for name in self.local_names}

    def group_functions(
        self, part: PyTree, glob: PyTree, arr: GLArrays, gl: GLShared, static: StaticConfig
    ) -> TestFns:
        return {}

    def _mean_log_v(self, part, glob, arr, gl, static) -> TestFns:
        """``mean_{j∈g} log v_j`` (``v`` without ``r``): a group summary of every local scale."""
        G = static.coef.n_groups
        log_v = self.log_kernel_var(part, arr, glob, gl, static) - arr.log_r
        tot = _segment_sum(log_v, arr, G) / _segment_sum(jnp.ones_like(log_v), arr, G)
        return {f"mean_log_v[{g}]": (tot[g], True) for g in range(G)}


@register_reference("horseshoe")
class HorseshoeRef(GLReference):
    """Horseshoe (Carvalho, Polson & Scott 2010) in the auxiliary form of Makalic & Schmidt (2016),
    P3 §3.2: ``v_j = ϑ_j ζ_g``, ``ϑ_j | ν_j ~ IG(½, 1/ν_j)``, ``ν_j ~ IG(½, 1)``,
    ``ζ_g | τ_g ~ IG(½, 1/τ_g)``, ``τ_g ~ IG(½, 1/s²)`` (``s = global_scale``), i.e.
    ``√ϑ_j ~ C⁺(0, 1)`` and ``√ζ_g ~ C⁺(0, s)``."""

    local_names = ("log_theta", "log_nu")

    def sample_global(self, key, gl, static):
        G = static.coef.n_groups
        log_tau = -2.0 * jnp.log(gl.global_scale) - _log_gamma(
            random.fold_in(key, 0), _full((G,), 0.5)
        )
        log_zeta = -log_tau - _log_gamma(random.fold_in(key, 1), _full((G,), 0.5))
        return HSGlobal(log_zeta=log_zeta, log_tau=log_tau)

    def sample_local(self, key, arr, glob, gl, static):
        shape = arr.mask.shape
        log_nu = -_log_gamma(random.fold_in(key, 0), _full(shape, 0.5))
        log_theta = -log_nu - _log_gamma(random.fold_in(key, 1), _full(shape, 0.5))
        return HSEq(log_theta=_on(arr.mask, log_theta), log_nu=_on(arr.mask, log_nu))

    def log_kernel_var(self, part, arr, glob, gl, static):
        return _on(arr.mask, arr.log_r + part.log_theta + glob.log_zeta[arr.group])

    def local_log_density(self, part, arr, glob, gl, static):
        lp = log_inv_gamma_from_logs(part.log_nu, 0.5, 0.0) + log_inv_gamma_from_logs(
            part.log_theta, 0.5, -part.log_nu
        )
        return _on(arr.mask, lp)

    def global_log_density(self, glob, gl, static):
        lp = log_inv_gamma_from_logs(glob.log_zeta, 0.5, -glob.log_tau)
        lp = lp + log_inv_gamma_from_logs(glob.log_tau, 0.5, -2.0 * jnp.log(gl.global_scale))
        return jnp.sum(lp)

    def group_functions(self, part, glob, arr, gl, static):
        G = static.coef.n_groups
        out: TestFns = {}
        for g in range(G):
            out[f"log_zeta[{g}]"] = (glob.log_zeta[g], True)
            out[f"log_tau[{g}]"] = (glob.log_tau[g], True)
        out.update(self._mean_log_v(part, glob, arr, gl, static))
        return out


def _ng_log_rate(a: Array, gl: GLShared) -> Array:
    """``log(2c/a)``, the log rate of ``ξ_g ~ G(b, 2c/a_g)``; 0 when ``c`` is tied to ``a/2``
    (encoded ``c < 0``, ``GLShared`` docstring)."""
    tied = gl.c < 0
    log_c = jnp.log(jnp.where(tied, 1.0, gl.c))
    return jnp.where(tied, 0.0, LOG_2 + log_c - jnp.log(a))


@register_reference("normal_gamma")
class NormalGammaRef(GLReference):
    """Normal-Gamma (Griffin & Brown 2010; GK's parametrisation), P3 §3.2: ``ϑ_j ~ G(a_g, a_g/2)``,
    ``ζ_g⁻¹ ~ G(b, c)``, ``v_j = ϑ_j ζ_g``, written as ``ϖ_j = v_j | ξ_g ~ G(a_g, ξ_g)`` with the
    rate ``ξ_g = a_g/(2ζ_g) ~ G(b, 2c/a_g)``; ``a_g`` fixed or on the discrete grid."""

    local_names = ("log_w",)

    def sample_global(self, key, gl, static):
        a_idx = _sample_a_idx(random.fold_in(key, 0), gl, static)
        a = a_values(a_idx, gl, static)
        lg = _log_gamma(random.fold_in(key, 1), jnp.broadcast_to(gl.b, a.shape))
        return NGGlobal(log_xi=lg - _ng_log_rate(a, gl), a_idx=a_idx)

    def sample_local(self, key, arr, glob, gl, static):
        a = a_values(glob.a_idx, gl, static)[arr.group]
        log_w = _log_gamma(key, a) - glob.log_xi[arr.group]
        return NGEq(log_w=_on(arr.mask, log_w))

    def log_kernel_var(self, part, arr, glob, gl, static):
        return _on(arr.mask, arr.log_r + part.log_w)

    def local_log_density(self, part, arr, glob, gl, static):
        a = a_values(glob.a_idx, gl, static)[arr.group]
        return _on(arr.mask, log_gamma_from_logs(part.log_w, a, glob.log_xi[arr.group]))

    def global_log_density(self, glob, gl, static):
        a = a_values(glob.a_idx, gl, static)
        lp = log_gamma_from_logs(glob.log_xi, gl.b, _ng_log_rate(a, gl))
        return jnp.sum(lp) + _a_log_prior(glob.a_idx, gl, static)

    def group_functions(self, part, glob, arr, gl, static):
        return _xi_group_functions(self, part, glob, arr, gl, static)


def _xi_group_functions(ref, part, glob, arr, gl, static) -> TestFns:
    """NG and R2D2: ``log ξ_g``; with a grid also ``a_idx_g`` and the joint ``a_g·log ξ_g`` (M35)."""
    G = static.coef.n_groups
    out: TestFns = {}
    a = a_values(glob.a_idx, gl, static)
    for g in range(G):
        out[f"log_xi[{g}]"] = (glob.log_xi[g], True)
        if static.coef.a_grid:
            out[f"a_idx[{g}]"] = (glob.a_idx[g].astype(_F64), True)
            out[f"a*log_xi[{g}]"] = (a[g] * glob.log_xi[g], True)
    out.update(ref._mean_log_v(part, glob, arr, gl, static))
    return out


@register_reference("r2d2")
class R2D2Ref(GLReference):
    """R2D2 (Zhang, Naughton, Bondell & Reich 2022; GK's gamma form), P3 §3.2:
    ``v_j = ϱ_j ϖ_j/2``, ``ϱ_j ~ Exp(½)``, ``ϖ_j | ξ_g ~ G(a_g, ξ_g)``, ``ξ_g ~ G(b, 1)``."""

    local_names = ("log_varrho", "log_w")

    def sample_global(self, key, gl, static):
        a_idx = _sample_a_idx(random.fold_in(key, 0), gl, static)
        G = static.coef.n_groups
        log_xi = _log_gamma(random.fold_in(key, 1), jnp.broadcast_to(gl.b, (G,)))
        return R2D2Global(log_xi=log_xi, a_idx=a_idx)

    def sample_local(self, key, arr, glob, gl, static):
        shape = arr.mask.shape
        a = a_values(glob.a_idx, gl, static)[arr.group]
        log_w = _log_gamma(random.fold_in(key, 0), a) - glob.log_xi[arr.group]
        log_varrho = _log_gamma(random.fold_in(key, 1), _full(shape, 1.0)) + LOG_2  # rate ½
        return R2D2Eq(log_varrho=_on(arr.mask, log_varrho), log_w=_on(arr.mask, log_w))

    def log_kernel_var(self, part, arr, glob, gl, static):
        return _on(arr.mask, arr.log_r + part.log_varrho + part.log_w - LOG_2)

    def local_log_density(self, part, arr, glob, gl, static):
        a = a_values(glob.a_idx, gl, static)[arr.group]
        lp = log_gamma_from_logs(part.log_varrho, 1.0, -LOG_2)
        lp = lp + log_gamma_from_logs(part.log_w, a, glob.log_xi[arr.group])
        return _on(arr.mask, lp)

    def global_log_density(self, glob, gl, static):
        lp = log_gamma_from_logs(glob.log_xi, gl.b, 0.0)
        return jnp.sum(lp) + _a_log_prior(glob.a_idx, gl, static)

    def group_functions(self, part, glob, arr, gl, static):
        return _xi_group_functions(self, part, glob, arr, gl, static)


@register_reference("dirichlet_laplace")
class DirichletLaplaceRef(GLReference):
    """Dirichlet–Laplace (Bhattacharya, Pati, Pillai & Dunson 2015; gamma form), P3 §3.2:
    ``d_j | ϱ_j, θ̃_j ~ N(0, r_j ϱ_j θ̃_j²)``, ``ϱ_j ~ Exp(½)``, ``θ̃_j ~ G(a_g, ½)``; no global
    parameter besides ``a_g``."""

    local_names = ("log_theta", "log_varrho")

    def sample_global(self, key, gl, static):
        return DLGlobal(a_idx=_sample_a_idx(random.fold_in(key, 0), gl, static))

    def sample_local(self, key, arr, glob, gl, static):
        shape = arr.mask.shape
        a = a_values(glob.a_idx, gl, static)[arr.group]
        log_theta = _log_gamma(random.fold_in(key, 0), a) + LOG_2  # rate ½
        log_varrho = _log_gamma(random.fold_in(key, 1), _full(shape, 1.0)) + LOG_2
        return DLEq(log_theta=_on(arr.mask, log_theta), log_varrho=_on(arr.mask, log_varrho))

    def log_kernel_var(self, part, arr, glob, gl, static):
        return _on(arr.mask, arr.log_r + part.log_varrho + 2.0 * part.log_theta)

    def local_log_density(self, part, arr, glob, gl, static):
        a = a_values(glob.a_idx, gl, static)[arr.group]
        lp = log_gamma_from_logs(part.log_theta, a, -LOG_2)
        lp = lp + log_gamma_from_logs(part.log_varrho, 1.0, -LOG_2)
        return _on(arr.mask, lp)

    def global_log_density(self, glob, gl, static):
        return _a_log_prior(glob.a_idx, gl, static)

    def group_functions(self, part, glob, arr, gl, static):
        out: TestFns = {}
        if static.coef.a_grid:
            for g in range(static.coef.n_groups):
                out[f"a_idx[{g}]"] = (glob.a_idx[g].astype(_F64), True)
        out.update(self._mean_log_v(part, glob, arr, gl, static))
        return out


def _ssvs_log_p(glob: SSVSGlobal, gl: GLShared, static: StaticConfig) -> tuple[Array, Array]:
    """``(log p_g, log(1 − p_g))``: the state's under a Beta hyperprior, else the fixed ``p``."""
    if static.coef.p_beta:
        return glob.log_p, glob.log1m_p
    G = static.coef.n_groups
    return (
        jnp.broadcast_to(jnp.log(gl.p_fixed), (G,)),
        jnp.broadcast_to(jnp.log1p(-gl.p_fixed), (G,)),
    )


@register_reference("ssvs")
class SSVSRef(GLReference):
    """SSVS (George & McCulloch 1993; George, Sun & Ni 2008), P3 §3.2: ``d_j | γ_j ~
    N(0, τ²_{γ_j, j})`` with ``log τ_{0,j}, log τ_{1,j}`` from ``GLArrays.ssvs``, ``γ_j | p_g ~
    Bernoulli(p_g)``, ``p_g`` fixed (``GLShared.p_fixed``; the state then holds ``log p_fixed``) or
    ``~ Beta(s₁, s₂)`` (density w.r.t. ``p``)."""

    local_names = ("gamma",)

    def sample_global(self, key, gl, static):
        G = static.coef.n_groups
        if static.coef.p_beta:
            log_p, log1m_p = _log_beta_pair(
                key, jnp.broadcast_to(gl.s1, (G,)), jnp.broadcast_to(gl.s2, (G,))
            )
            return SSVSGlobal(log_p=log_p, log1m_p=log1m_p)
        return SSVSGlobal(
            log_p=jnp.broadcast_to(jnp.log(gl.p_fixed), (G,)),
            log1m_p=jnp.broadcast_to(jnp.log1p(-gl.p_fixed), (G,)),
        )

    def sample_local(self, key, arr, glob, gl, static):
        log_p, _ = _ssvs_log_p(glob, gl, static)
        u = random.uniform(key, arr.mask.shape, dtype=_F64)
        gamma = (jnp.log(u) < log_p[arr.group]).astype(jnp.int32)
        return SSVSEq(gamma=_on(arr.mask, gamma), rb_incl=jnp.zeros(arr.mask.shape, _F64))

    def log_kernel_var(self, part, arr, glob, gl, static):
        s = arr.ssvs
        return _on(arr.mask, 2.0 * jnp.where(part.gamma == 1, s.log_tau1, s.log_tau0))

    def local_log_density(self, part, arr, glob, gl, static):
        log_p, log1m_p = _ssvs_log_p(glob, gl, static)
        lp = jnp.where(part.gamma == 1, log_p[arr.group], log1m_p[arr.group])
        return _on(arr.mask, lp)

    def global_log_density(self, glob, gl, static):
        if not static.coef.p_beta:
            return jnp.zeros((), _F64)
        return jnp.sum(log_beta_from_logs(glob.log_p, glob.log1m_p, gl.s1, gl.s2))

    def local_logs(self, part):
        return {"gamma": part.gamma.astype(_F64)}

    def group_functions(self, part, glob, arr, gl, static):
        G = static.coef.n_groups
        n1 = _segment_sum(part.gamma.astype(_F64), arr, G)
        out: TestFns = {f"N1[{g}]": (n1[g], True) for g in range(G)}
        if static.coef.p_beta:
            out.update({f"log_p[{g}]": (glob.log_p[g], True) for g in range(G)})
        return out


def gl_deviation_draw(key: Array, cp: CoefPriorArrays, log_kv: Array) -> tuple[Array, Array]:
    """One equation's coefficients from ``p(coefficients | scales)`` (P3 §3.1): on GL slots
    ``d_j = exp(½ log(r_j v_j))·z_j``, elsewhere ``N(m, 1/prec0)``. Returns
    ``(m + d, log|d|)`` with ``log|d| = log sd + log|z|`` exact (never ``log|m + d − m|``)."""
    z = _normal(key, cp.mean0.shape)
    mask = cp.gl.mask
    log_sd = jnp.where(mask > 0, 0.5 * log_kv, -0.5 * jnp.log(cp.prec0))
    return cp.mean0 + jnp.exp(log_sd) * z, log_sd + jnp.log(jnp.abs(z))


def gl_coefficient_log_density(
    coef: Array, dev_log: Array, cp: CoefPriorArrays, log_kv: Array
) -> Array:
    """Elementwise ``log p(coefficient | scales)``: ``log N(d; 0, r v)`` from ``ℓ = dev_log`` on GL
    slots (``d²/(r v) = exp(2ℓ − log r v)``, P3 §3.1), ``log N(coef; m, 1/prec0)`` elsewhere."""
    mask = cp.gl.mask
    lkv = jnp.where(mask > 0, log_kv, 0.0)
    gl = -0.5 * (LOG_2PI + lkv + jnp.exp(2.0 * dev_log - lkv))
    fixed = log_normal(coef, cp.mean0, 1.0 / cp.prec0)
    return jnp.where(mask > 0, gl, fixed)


# --------------------------------------------------------------------------- P3: CommonSV


class _SigmaPrior(NamedTuple):
    """The ``σ²``-prior fields under the Phase 2 names (``_log_sigma_prior``)."""

    B_sigma: Array
    c0: Array
    C0: Array


def _csv_sigma_prior(csv: CommonSVPrior) -> _SigmaPrior:
    return _SigmaPrior(B_sigma=csv.s2_B, c0=csv.s2_shape, C0=csv.s2_scale)


def _log_rho_prior(rho: Array, csv: CommonSVPrior, kind: str) -> Array:
    """``ρ ~ N(ρ₀, V_ρ)·1(|ρ| < 1)`` (normalised by ``Φ(b) − Φ(a)``) or ``(ρ+1)/2 ~ Beta(a, b)``
    (density of ρ); ``−∞`` for ``|ρ| ≥ 1``."""
    if kind == "beta":
        return _log_phi_prior(rho, csv.rho_a, csv.rho_b)
    sd = jnp.sqrt(csv.rho_var)
    lo, hi = (-1.0 - csv.rho_mean) / sd, (1.0 - csv.rho_mean) / sd
    lp = log_normal(rho, csv.rho_mean, csv.rho_var) - jnp.log(ndtr(hi) - ndtr(lo))
    return jnp.where(jnp.abs(rho) < 1.0, lp, -jnp.inf)


def _draw_rho(key: Array, csv: CommonSVPrior, kind: str) -> Array:
    if kind == "beta":
        log_u, log1m_u = _log_beta_pair(key, csv.rho_a, csv.rho_b)
        return jnp.exp(log_u) - jnp.exp(log1m_u)  # 2u − 1
    sd = jnp.sqrt(csv.rho_var)
    lo, hi = (-1.0 - csv.rho_mean) / sd, (1.0 - csv.rho_mean) / sd
    z = random.truncated_normal(key, lo, hi, dtype=_F64)  # inverse CDF on [Φ(lo), Φ(hi)]
    inner = jnp.nextafter(jnp.asarray(1.0, _F64), 0.0)
    return jnp.clip(csv.rho_mean + sd * z, -inner, inner)


def _draw_log_sigma_h(key: Array, csv: CommonSVPrior, kind: str) -> Array:
    """``log σ_h``: ``σ_h² ~ IG(c₀, C₀)`` (``½(log C₀ − LG(c₀))``) or ``B·χ²₁`` (``σ_h = √B|z|``)."""
    if kind == "scaled_chi2":
        return 0.5 * jnp.log(csv.s2_B) + jnp.log(jnp.abs(_normal(key)))
    return 0.5 * (jnp.log(csv.s2_scale) - _log_gamma(key, csv.s2_shape))


@register_reference("constant_common")
class ConstantCommonRef(ConstantRef):
    """Constant cores under CommonSV (P3 §5.1–§5.2): ``λ_{i,t} = λ_i e^{h_t}`` with the Constant
    ``λ_i ~ IG(shape, scale)`` (Mode I, or coupled in Mode II: the λ-scaled blocks read ``log λ_i``
    as before). :meth:`latent_log_var_eq` is ``log λ_i``; :func:`effective_log_var` adds ``h_t``."""


@register_reference("common_sv")
class CommonSVRef(_Base):
    """The common log volatility ``glob.vol`` (P3 §5.1, D-46; 0-based dates, ``h_t`` belongs to
    data row ``t``): stationary AR(1) without intercept, ``h_0 ~ N(0, σ_h²/(1 − ρ²))``,
    ``h_t = ρ h_{t−1} + σ_h η_t`` for ``1 ≤ t < T_eff``, padded states exactly 0 and carrying no
    density. ``prior_sample_global``, ``log_prior_global`` and ``test_functions_global`` need the
    data mask (keyword ``tmask``)."""

    def prior_sample_global(self, key, prior, static, *, tmask=None):
        if tmask is None:
            raise ValueError("the common_sv reference needs tmask= (h is 0 on padded rows)")
        csv, vol = prior.shared.csv, static.vol
        T_pad = static.shapes.T_pad
        k = [random.fold_in(key, j) for j in range(4)]
        rho = _draw_rho(k[0], csv, vol.csv_rho_prior)
        log_sigma = _draw_log_sigma_h(k[1], csv, vol.csv_sigma2_prior)
        sigma = jnp.exp(log_sigma)
        h0 = sigma / jnp.sqrt((1.0 - rho) * (1.0 + rho)) * _normal(k[2])
        eta = _normal(k[3], (T_pad,))

        def body(h_prev: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
            e, m = inp
            h = jnp.where(m > 0, rho * h_prev + sigma * e, 0.0)
            return h, h

        _, rest = jax.lax.scan(body, h0, (eta[1:], tmask[1:]))
        h = jnp.where(tmask > 0, jnp.concatenate([h0[None], rest]), 0.0)
        z, zt = jnp.zeros((), _F64), jnp.zeros((T_pad,), _F64)
        # positional (field order h, ρ, log σ, three scalar accumulators, rb_ch_sum, rb_ch_sq):
        # independent of the accumulators' names (WP26 erratum E132 renames them)
        return CommonSVGlobal(h, rho, log_sigma, z, z, z, zt, zt)

    def log_prior_global(self, part, prior, static, *, tmask=None):
        if tmask is None:
            raise ValueError("the common_sv reference needs tmask=")
        csv, vol = prior.shared.csv, static.vol
        rho, h = part.rho, part.h
        sigma = jnp.exp(part.log_sigma)
        lp = _log_rho_prior(rho, csv, vol.csv_rho_prior)
        lp = lp + _log_sigma_prior(sigma, _csv_sigma_prior(csv), vol.csv_sigma2_prior)
        inside = jnp.abs(rho) < 1.0
        rho_s = jnp.where(inside, rho, 0.0)
        lp = lp + log_normal(h[0], 0.0, sigma**2 / ((1.0 - rho_s) * (1.0 + rho_s)))
        trans = log_normal(h[1:], rho_s * h[:-1], sigma**2)
        return lp + jnp.sum(jnp.where(tmask[1:] > 0, trans, 0.0))

    def test_functions_global(self, part, *, tails=_NO_TAILS, tmask=None):
        if tmask is None:
            raise ValueError("the common_sv reference needs tmask=")
        T = _t_eff(tmask)
        h, rho, log_sigma = part.h, part.rho, part.log_sigma
        sigma = jnp.exp(log_sigma)
        dates = {
            "h0": h[0],
            "hmid": jnp.take(h, T // 2),
            "hT": jnp.take(h, T - 1),
            "mean_h": _masked_mean(h, tmask),
        }
        h_ok = tails.h2 > 1.0  # Var(h_t) = E σ² · E (1 − ρ²)⁻¹ (infinite under the TN ρ prior)
        out: TestFns = {name: (v, h_ok) for name, v in dates.items()}
        out["mean_abs_h"] = (_masked_mean(jnp.abs(h), tmask), h_ok)
        # bounded transforms (WP30): the h path stays testable when Var(h) = ∞
        out.update({f"atan_{name}": (jnp.arctan(v), True) for name, v in dates.items()})
        s1, s2 = tails.sigma2 > 1.0, tails.sigma2 > 2.0
        out.update({
            "rho": (rho, True), "rho^2": (rho**2, True), "sigma": (sigma, s1),
            "sigma^2": (sigma**2, s2), "log_sigma": (log_sigma, True),
            "rho*log_sigma": (rho * log_sigma, True),
        })  # fmt: skip
        return out


# --------------------------------------------------------------------------- P3: review references
# NumPy only, from docs/phase3/DESIGN.md (review items M12, M12(b), m25 of the design review): the
# value references WP31 (GL builders) and WP34 (conditional forecasts, A10) are tested against.


def gl_log_r_reference(
    psi: Any,
    n: int,
    lags: int,
    n_det: int,
    form: Literal["structural", "reduced", "linear_structural", "linear_reduced"],
    scale: Literal["psi", "none"] = "psi",
) -> np.ndarray:
    """``log r_j`` of the GL kernel ``d_j | v_j ~ N(0, r_j v_j)`` (P3 D-37, §3.1), shape ``(n, k)``,
    row ``i`` = equation ``i`` in both forms, column ``r`` = regressor of the ``data/design.py``
    layout; 0 on the deterministic slots ``r < n_det`` and everywhere under ``scale="none"``.

    Lag slot ``r ∈ {n_det, …, k−1}`` has lag ``l(r) = 1 + ⌊(r − n_det)/n⌋`` and variable
    ``m(r) = (r − n_det) mod n`` (§3.1); equation ``i`` (0-based) has ``ι = i + 1``, the index
    convention of the Phase 2 Mode I prior (P2 §6.1: ``a_ij ~ N(0, κ_a ψ[i]/(ι ψ[j]))``, cross lag
    ``κ²κ_c²ψ[i]/(ι l^α ψ[m])``; ``priors/structural.py``). Structural: ``log ψ_i − log ι −
    log ψ_m``; reduced (``priors/reduced.py``: row ``j`` of the arrays is equation ``j``, lag ``l``
    of variable ``m`` has variance ``κ²κ_c²ψ_j/(l^α ψ_m)``): ``log ψ_i − log ψ_m``. So ``r_j`` is the
    Mode I (structural) or reduced Minnesota variance at ``κ = 1``, ``κ_c = 1`` without lag decay;
    there is no ``l`` term (D-37)."""
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (n,) or not np.all(np.isfinite(psi) & (psi > 0)):
        raise ValueError(f"psi must be {n} positive finite values, got {psi}")
    if form not in ("structural", "reduced", "linear_structural", "linear_reduced"):
        raise ValueError(f"form must be 'structural' or 'reduced', got {form!r}")
    if scale not in ("psi", "none"):
        raise ValueError(f"scale must be 'psi' or 'none', got {scale!r}")
    k = n_det + n * lags
    out = np.zeros((n, k))
    if scale == "none":
        return out
    structural = form in ("structural", "linear_structural")
    for i in range(n):
        for r in range(n_det, k):
            m = (r - n_det) % n
            out[i, r] = math.log(psi[i]) - math.log(psi[m])
            if structural:
                out[i, r] -= math.log(i + 1.0)
    return out


def ssvs_shat_reference(
    Y: Any,
    X: Any,
    form: Literal["structural", "reduced", "linear_structural", "linear_reduced"],
    *,
    Mc: Any = None,
) -> np.ndarray:
    """Semi-automatic SSVS scales ``ŝ_j`` (P3 D-44, §3.6), shape ``(n, k)`` (row = equation, column
    = regressor), by a dense oracle (``np.linalg.inv``) on the real rows ``Y`` (``T_eff × n``) and
    ``X`` (``T_eff × k``).

    Structural, equation ``i``: the flat-prior NIG posterior variance of ``δ_i`` in the regression
    of ``y_i`` on ``F_i = [X | X M_c[:, :i] − Y_{:, :i}]`` (the active block of the Phase 2 δ-design
    ``F = [X | X M_c − Y_{:, :n−1}]``, ``q_i = k + i`` columns; ``Mc`` the D3 centring columns,
    ``SharedPrior.Mc``): ``ŝ²_j = RSS_i/(T_eff − q_i − 2)·[(F_iᵀF_i)⁻¹]_jj`` for the ``k`` δ columns.
    Reduced: ``ŝ²_{rj} = Ŝ_jj/(T_eff − k − n − 1)·[(XᵀX)⁻¹]_rr`` with ``Ŝ`` the OLS residual
    sum-of-squares matrix (flat-prior normal–Wishart). ``ValueError`` when a divisor is ≤ 0."""
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim != 2 or X.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"Y (T, n) and X (T, k) must share T, got {Y.shape} and {X.shape}")
    T, n = Y.shape
    k = X.shape[1]
    out = np.empty((n, k))
    if form in ("reduced", "linear_reduced"):
        dof = T - k - n - 1
        if dof <= 0:
            raise ValueError(
                f"reduced-form semi-automatic SSVS needs T_eff > k + n + 1 = {k + n + 1}"
            )
        G = np.linalg.inv(X.T @ X)
        E = Y - X @ (G @ (X.T @ Y))
        S = E.T @ E
        return np.sqrt(np.diag(S)[:, None] / dof * np.diag(G)[None, :])
    if form not in ("structural", "linear_structural"):
        raise ValueError(f"form must be 'structural' or 'reduced', got {form!r}")
    Mc = np.asarray(Mc, dtype=np.float64) if Mc is not None else None
    if Mc is None or Mc.shape != (k, n - 1):
        raise ValueError(
            f"the structural form needs the D3 centring columns Mc of shape ({k}, {n - 1})"
        )
    for i in range(n):
        F = np.hstack([X, X @ Mc[:, :i] - Y[:, :i]])
        dof = T - (k + i) - 2
        if dof <= 0:
            raise ValueError(
                f"structural semi-automatic SSVS needs T_eff > q_i + 2 = {k + i + 2} (equation {i})"
            )
        G = np.linalg.inv(F.T @ F)
        resid = Y[:, i] - F @ (G @ (F.T @ Y[:, i]))
        out[i] = np.sqrt(resid @ resid / dof * np.diag(G)[:k])
    return out


class ConditionalForecastReference(NamedTuple):
    """Result of :func:`conditional_forecast_reference` (stacked index ``(h − 1)·n + j``)."""

    mean: np.ndarray  # (H, n) E_c: conditional mean of the path
    cov: np.ndarray  # (nH, nH) Cov_c: conditional covariance of the stacked path
    log_weight: float  # log N(r; Rμ, RΣRᵀ)
    path: np.ndarray | None  # (H, n) Y_c from the given ε (None without ε)
    uncond_mean: np.ndarray  # (H, n) μ
    uncond_cov: np.ndarray  # (nH, nH) Σ = MMᵀ
    M: np.ndarray  # (nH, nH) block lower triangular, M_{h,k} = Φ_{h−k} L_k
    index: np.ndarray  # (m,) stacked indices of the conditioned cells
    values: np.ndarray  # (m,) r


def companion_impulse(lag_coefs: Any, H: int) -> np.ndarray:
    """``Φ_0 … Φ_{H−1}`` (``(H, n, n)``) as the leading block of the powers of the companion
    matrix ``C = [[A_1 … A_p], [I 0]]`` (``lag_coefs[l−1] = A_l``, the Phase 2 D-34 layout
    ``A_l[i, j]`` = coefficient of ``y_{t−l, j}`` in equation ``i``)."""
    A = np.asarray(lag_coefs, dtype=np.float64)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"lag_coefs must have shape (p, n, n), got {A.shape}")
    p, n, _ = A.shape
    C = np.zeros((n * p, n * p))
    C[:n] = np.concatenate(list(A), axis=1)
    if p > 1:
        C[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    out = np.empty((H, n, n))
    P = np.eye(n * p)
    for j in range(H):
        out[j] = P[:n, :n]
        P = C @ P
    return out


def zero_shock_path(
    coefs: Any, hist: Any, H: int, *, n_det: int, t_last: float = 0.0
) -> np.ndarray:
    """The unconditional (zero-shock) mean path ``μ`` (``(H, n)``) of a linear VAR
    ``y_t = Bᵀx_t + u_t`` (``coefs = B``, ``k × n``, the ``data/design.py`` regressor layout) from
    the last ``p`` observations ``hist`` (``p × n``, most recent last): ``x_{T+h} = (d_{T+h}',
    y_{T+h−1}', …)'`` with ``d = (1, t_last + h)[:n_det]`` and the earlier means in place of future
    values."""
    B = np.asarray(coefs, dtype=np.float64)
    hist = np.asarray(hist, dtype=np.float64)
    p, n = hist.shape
    if B.shape != (n_det + n * p, n):
        raise ValueError(f"coefs must have shape ({n_det + n * p}, {n}), got {B.shape}")
    window = list(hist)
    out = np.empty((H, n))
    for h in range(1, H + 1):
        det = [1.0, t_last + h][:n_det]
        x = np.concatenate([det, *[window[-lag] for lag in range(1, p + 1)]])
        out[h - 1] = B.T @ x
        window.append(out[h - 1])
    return out


def conditional_forecast_reference(
    lag_coefs: Any,
    chol: Any,
    mean: Any,
    conditions: Any,
    *,
    eps: Any = None,
) -> ConditionalForecastReference:
    """Exact hard-condition update of one draw's forecast path (P3 §6.2, D-50; the oracle of A10),
    in stacked companion form with a dense Schur complement.

    Inputs (plain arrays): ``lag_coefs`` ``(p, n, n)`` (``A_l``, D-34 layout), ``chol`` ``(H, n, n)``
    (``L_k``, the shock factor of horizon ``k``, ``u_k = L_k ε_k``; it may vary with ``k``),
    ``mean`` ``(H, n)`` the unconditional zero-shock path ``μ`` (e.g. :func:`zero_shock_path`) and
    ``conditions`` an iterable of ``(h, j, value)`` with ``h ∈ 1..H`` and variable ``j ∈ 0..n−1``.
    With ``Φ`` from :func:`companion_impulse`, ``Y = μ + Mε`` (``M_{h,k} = Φ_{h−k}L_k``, ``k ≤ h``),
    ``Σ = MMᵀ``, ``R`` the selection of the conditioned cells and ``S = RΣRᵀ``:

        E_c = μ + ΣRᵀS⁻¹(r − Rμ),   Cov_c = Σ − ΣRᵀS⁻¹RΣ,   log w = log N(r; Rμ, S),

    and, given ``eps`` (``(H, n)``), the exact draw ``Y_c = Y + ΣRᵀS⁻¹(r − RY)`` from the same
    shocks."""
    Phi_lags = np.asarray(lag_coefs, dtype=np.float64)
    L = np.asarray(chol, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    if L.ndim != 3 or L.shape[1] != L.shape[2]:
        raise ValueError(f"chol must have shape (H, n, n), got {L.shape}")
    H, n, _ = L.shape
    if mu.shape != (H, n) or Phi_lags.shape[1:] != (n, n):
        raise ValueError(f"mean must be ({H}, {n}) and lag_coefs (p, {n}, {n})")
    cells: dict[int, float] = {}
    for h, j, value in conditions:
        if int(h) != h or not 1 <= h <= H or int(j) != j or not 0 <= j < n:
            raise ValueError(
                f"condition ({h}, {j}) outside horizons 1..{H} and variables 0..{n - 1}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"condition ({h}, {j}) is not finite")
        idx = (int(h) - 1) * n + int(j)
        if idx in cells:
            raise ValueError(f"condition ({h}, {j}) given twice")
        cells[idx] = float(value)
    if not cells or len(cells) == n * H:
        raise ValueError("need at least one condition and at least one free cell")
    Phi = companion_impulse(Phi_lags, H)
    M = np.zeros((n * H, n * H))
    for h in range(H):
        for k in range(h + 1):
            M[h * n : (h + 1) * n, k * n : (k + 1) * n] = Phi[h - k] @ L[k]
    Sigma = M @ M.T
    idx = np.fromiter(cells, dtype=np.int64)
    r = np.fromiter(cells.values(), dtype=np.float64)
    mu_s = mu.reshape(-1)
    S = Sigma[np.ix_(idx, idx)]
    K = np.linalg.solve(S, Sigma[idx, :]).T  # ΣRᵀS⁻¹
    dev = r - mu_s[idx]
    cond_mean = mu_s + K @ dev
    cond_cov = Sigma - K @ Sigma[idx, :]
    _, logdet = np.linalg.slogdet(S)
    log_w = -0.5 * (idx.size * LOG_2PI + logdet + dev @ np.linalg.solve(S, dev))
    path = None
    if eps is not None:
        e = np.asarray(eps, dtype=np.float64)
        if e.shape != (H, n):
            raise ValueError(f"eps must have shape ({H}, {n}), got {e.shape}")
        Yu = mu_s + M @ e.reshape(-1)
        path = (Yu + K @ (r - Yu[idx])).reshape(H, n)
    return ConditionalForecastReference(
        mean=cond_mean.reshape(H, n), cov=cond_cov, log_weight=float(log_w), path=path,
        uncond_mean=mu, uncond_cov=Sigma, M=M, index=idx, values=r,
    )  # fmt: skip


# --------------------------------------------------------------------------- composition


@dataclass(frozen=True)
class SlotReferences:
    """The references of one composition, by slot (DESIGN §3.8 slots)."""

    structure: Reference
    mean: Reference
    coef: Reference
    core: Reference
    trans: Reference
    gvol: Reference | None = None  # P3: the global volatility state (CommonSV), else None


def core_kind(static: StaticConfig) -> str:
    """Reference kind of the persistent core: ``"constant"``, ``"sv"`` or ``"sv_rw"``; P3:
    ``"constant_common"`` (Constant cores with the common ``h``)."""
    if static.vol.persistent == "constant":
        return "constant_common" if static.vol.common else "constant"
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
        gvol=make("common_sv") if static.vol.common else None,
    )


def effective_log_var(
    vol: VolEq,
    static: StaticConfig,
    which: Which = "effective",
    refs: SlotReferences | None = None,
    *,
    gvol: Any = None,
) -> Array:
    """``log λ_eff,i,t = log λ^pers_i,t + 2 log o_i,t + log τ_i,t`` (``which="effective"``) or the
    persistent part only; shape ``(..., n, T_pad)`` for a batched ``VolEq`` (leading equation axis),
    computed equation by equation with the references.

    P3 CommonSV (``static.vol.common``): the persistent log variance is ``log λ_i + h_t`` with
    ``h`` from the global state ``gvol = st.glob.vol`` (required; :func:`state_log_var`)."""
    refs = refs or references_for(static)
    T_pad = static.shapes.T_pad

    def one(v: VolEq) -> Array:
        lv = refs.core.latent_log_var_eq(v.core, which, T_pad=T_pad)
        return lv + refs.trans.latent_log_var_eq(v.trans, which, T_pad=T_pad)

    out = jax.vmap(one)(vol)
    if static.vol.common:
        if not isinstance(gvol, CommonSVGlobal):
            raise ValueError("CommonSV log variances need the common h: pass gvol=st.glob.vol")
        out = out + gvol.h[None, :]
    return out


def state_log_var(
    st: Any, static: StaticConfig, which: Which = "effective", refs: SlotReferences | None = None
) -> Array:
    """:func:`effective_log_var` of a chain state (its ``eq.vol`` and, under CommonSV, the common
    ``h`` of ``glob.vol``); ``n × T_pad``."""
    return effective_log_var(st.eq.vol, static, which, refs, gvol=st.glob.vol)


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
    "LOG_2",
    "LOG_2PI",
    "REFERENCES",
    "CommonSVRef",
    "ConditionalForecastReference",
    "ConstantCommonRef",
    "DirichletLaplaceRef",
    "GLReference",
    "HorseshoeRef",
    "MixtureTableRef",
    "NormalGammaRef",
    "PriorTails",
    "R2D2Ref",
    "Reference",
    "SSVSRef",
    "SlotReferences",
    "a_values",
    "companion_impulse",
    "conditional_forecast_reference",
    "core_kind",
    "coupling_log_scale",
    "effective_log_var",
    "gl_coefficient_log_density",
    "gl_deviation_draw",
    "gl_log_r_reference",
    "log_beta_fn",
    "log_beta_from_logs",
    "log_gamma_from_logs",
    "log_inv_gamma_from_log",
    "log_inv_gamma_from_logs",
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
    "ssvs_shat_reference",
    "state_log_var",
    "t_scale_constant",
    "transitory_log_density",
    "zero_shock_path",
]
