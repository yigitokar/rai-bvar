"""Normal mixtures for ``log χ²₁``: OCSN-10 (default) and KSC-7 (D-3; N01 §5, §6, §8; N02 §2.3).

With ``ẽ_t = exp(h_t/2)ε_t``, ``y*_t = log ẽ_t² = h_t + ξ_t`` and ``ξ_t = log ε_t² ~ log χ²₁``, whose exact
density is ``f(ξ) = (2π)^{−½} exp{(ξ − e^ξ)/2}`` (N01 §4). The samplers replace ``f`` by
``g(ξ) = Σ_j p_j N(ξ; m_j, v_j)`` with an indicator ``s_t`` per row (N01 §3).

* **OCSN-10** (Omori, Chib, Shephard & Nakajima 2007, Table 1 right block; N01 §5.3): the printed
  means are the component means of ``log χ²₁`` itself (``Σ p m = −1.270280``); **no shift**.
* **KSC-7** (Kim, Shephard & Chib 1998, Table 4; N01 §5.1): printed means are centred at 0; the
  effective means are ``m_i − 1.2704`` (the literal, N01 §4). Stored here already shifted and
  re-sorted by descending mean, which reproduces OCSN Table 1, left block (N01 §5.2). A table is
  never shifted twice: every :class:`MixtureTable` holds **effective** means (D-3; N01 §6).

Indicator full conditional (N01 §8.1; KFS §2.6), in logs with ``r_t = y*_t − h_t``::

    ℓ_tj = logw_j − (r_t − m_j)²·half_inv_v_j,   logw_j = log p_j − ½ log v_j,   half_inv_v_j = 1/(2v_j)
    P(s_t = j | ·) = softmax_j ℓ_tj,             log g(r_t) = logsumexp_j ℓ_tj − ½ log 2π

The ``σ_j⁻¹`` factor sits in ``logw`` (its omission is mutant M11, N01 §8.3); ``v`` is a variance in
every source (N01 §11 item 5). The tables are NumPy (no module-level ``jnp`` array, DESIGN §1.1
rule 3); they become constants when a kernel is traced. Clean room: every constant is
transcribed from the papers (N01 §5.5 lists four independent software transcriptions that agree).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)

Array = jax.Array

LOG_2PI = math.log(2.0 * math.pi)
KSC_SHIFT = 1.2704  # KSC (1998) §3.1: N(m_i − 1.2704, v_i²); the literal, not ψ(½) + log 2 (N01 §4)
LOGCHI2_MEAN = -1.2703628454614782  # E log χ²₁ = ψ(½) + log 2 (N01 §4)
LOGCHI2_VAR = math.pi**2 / 2.0  # Var log χ²₁ = ψ′(½) (N01 §4)

__all__ = [
    "COARSE1",
    "KSC7",
    "KSC_SHIFT",
    "LOGCHI2_MEAN",
    "LOGCHI2_VAR",
    "LOG_2PI",
    "OCSN10",
    "MixtureTable",
    "indicator_logits",
    "log_g",
    "log_g_from_logits",
    "log_logchi2_density",
    "mixture_table",
]


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class MixtureTable:
    """A normal mixture for ``log χ²₁`` with **effective** means (D-3).

    ``p`` weights, ``m`` effective means, ``v`` variances (float64, read-only NumPy);
    ``logw = log p − ½ log v`` and ``half_inv_v = 1/(2v)`` are derived once. Components may be in
    any order (the kernel's indicator indices refer to this order; both shipped tables are sorted
    by descending mean, the OCSN Table 1 order).
    """

    name: str
    p: np.ndarray
    m: np.ndarray
    v: np.ndarray
    logw: np.ndarray = field(init=False, repr=False)
    half_inv_v: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p, m, v = _ro(self.p), _ro(self.m), _ro(self.v)
        if not (p.ndim == m.ndim == v.ndim == 1 and p.shape == m.shape == v.shape and p.size):
            raise ValueError("p, m and v must be non-empty vectors of equal length")
        if not (np.all(p > 0) and np.all(v > 0) and np.all(np.isfinite(m))):
            raise ValueError("mixture weights and variances must be positive, means finite")
        if abs(float(p.sum()) - 1.0) > 1e-9:
            raise ValueError(f"mixture weights must sum to 1, got {p.sum()!r}")
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "logw", _ro(np.log(p) - 0.5 * np.log(v)))
        object.__setattr__(self, "half_inv_v", _ro(0.5 / v))

    @property
    def J(self) -> int:
        """Number of components."""
        return int(self.p.size)

    @property
    def mean(self) -> float:
        """``Σ p_j m_j`` (OCSN-10: −1.270280; KSC-7: −1.270399; N01 §5.4)."""
        return float(np.dot(self.p, self.m))

    @property
    def variance(self) -> float:
        """``Σ p_j(v_j + (m_j − mean)²)`` (OCSN-10: 4.933731; KSC-7: 4.934854)."""
        return float(np.dot(self.p, self.v + (self.m - self.mean) ** 2))

    def shifted(self, delta: float, name: str | None = None) -> MixtureTable:
        """The same table with every mean moved by ``delta`` (test device: M12 is ``−1.2704``)."""
        return MixtureTable(name or f"{self.name}{delta:+g}", self.p, self.m + delta, self.v)


def _from_rows(
    name: str, rows: tuple[tuple[float, float, float], ...], shift: float
) -> MixtureTable:
    """``(p, m_printed, v)`` rows → effective means ``m − shift``, sorted by descending mean."""
    arr = np.array(rows, dtype=np.float64)
    p, m, v = arr[:, 0], arr[:, 1] - shift, arr[:, 2]
    order = np.argsort(-m, kind="stable")
    return MixtureTable(name, p[order], m[order], v[order])


# OCSN (2007) Table 1, right-hand block (N01 §5.3; N02 §2.3): p_j, m_j, v_j² (a variance).
_OCSN10_ROWS: tuple[tuple[float, float, float], ...] = (
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

# KSC (1998) Table 4 as printed (N01 §5.1): q_i, m_i (centred at 0), v_i² (a variance).
_KSC7_TABLE4_ROWS: tuple[tuple[float, float, float], ...] = (
    (0.00730, -10.12999, 5.79596),
    (0.10556, -3.97281, 2.61369),
    (0.00002, -8.56686, 5.17950),
    (0.04395, 2.77786, 0.16735),
    (0.34001, 0.61942, 0.64009),
    (0.24566, 1.79518, 0.34023),
    (0.25750, -1.08819, 1.26261),
)

OCSN10: MixtureTable = _from_rows("ocsn10", _OCSN10_ROWS, 0.0)
"""Default table (D-3): OCSN 10 components, unshifted."""

KSC7: MixtureTable = _from_rows("ksc7", _KSC7_TABLE4_ROWS, KSC_SHIFT)
"""Parity option ``mixture="ksc7"``: KSC Table 4 means minus 1.2704, sorted by descending mean."""

COARSE1: MixtureTable = MixtureTable("coarse1", [1.0], [LOGCHI2_MEAN], [LOGCHI2_VAR])
"""Validation device only (pilot §3.4): one component ``N(ψ(½) + log 2, π²/2)``. Exact mode is exact
for any table (N01 §10.2), so a deliberately poor table gives Geweke and invariance tests power
over the exact correction: ``exact=False`` and M7 fail the PR-size block Geweke with it (and pass
with OCSN-10). A stale ``w`` is subtler: it passes the PR-size block Geweke (``M_c = 100``), is
borderline at 10× that chain length and fails for every seed tried from 20× (the nightly test runs
80×), so the PR tier pins it with a deterministic test of the ``w`` the NC accept uses. Inject it
through the ``mixture_table`` override; it is not a user option."""


def mixture_table(name: Literal["ocsn10", "ksc7"]) -> MixtureTable:
    """The table of ``VolStatic.mixture`` (the default of the overridable ``mixture_table``)."""
    if name == "ocsn10":
        return OCSN10
    if name == "ksc7":
        return KSC7
    raise ValueError(f"unknown mixture {name!r}; expected 'ocsn10' or 'ksc7'")


def indicator_logits(r: Array, table: MixtureTable) -> Array:
    """Unnormalised log indicator weights ``ℓ[..., j] = logw_j − (r − m_j)²/(2v_j)`` (N01 §8.2),
    shape ``r.shape + (J,)``. The common ``−½ log 2π`` is omitted (it cancels in the pmf)."""
    r = jnp.asarray(r, dtype=jnp.float64)
    d = r[..., None] - table.m
    return table.logw - d * d * table.half_inv_v


def log_g_from_logits(logits: Array) -> Array:
    """``log g = logsumexp_j ℓ_j − ½ log 2π`` from :func:`indicator_logits` (N01 §10.2)."""
    return logsumexp(logits, axis=-1) - 0.5 * LOG_2PI


def log_g(r: Array, table: MixtureTable) -> Array:
    """Log mixture density ``log g(r) = log Σ_j p_j N(r; m_j, v_j)`` (max-subtracted, float64)."""
    return log_g_from_logits(indicator_logits(r, table))


def log_logchi2_density(xi: Array) -> Array:
    """Exact ``log f(ξ) = −½ log 2π + ½ξ − ½e^ξ`` of ``ξ = log ε²``, ``ε ~ N(0, 1)`` (N01 §4)."""
    xi = jnp.asarray(xi, dtype=jnp.float64)
    return -0.5 * LOG_2PI + 0.5 * xi - 0.5 * jnp.exp(xi)
