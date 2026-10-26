"""Warm starts: the final state of one fit as the initial state of the next (DESIGN §5.5, D-32).

``fit.final_state`` is a :class:`FinalState`: the host state after exactly ``burn + draws·thin``
sweeps (``snap``), the periods of its ``T_eff`` data rows, its ``StaticConfig`` and the seed
metadata. :func:`remap` moves it onto a new estimation window:

* parameters (``δ``, ``a``, ``Π``, ``(μ, φ, σ)``, ``log λ``, ``log p``, ``ν``) are copied;
* time-indexed latents (``h``, ``o``, ``τ``) are aligned **by period label**: a new row whose period
  exists in the old window takes the old value (``h`` state ``t + 1`` belongs to row ``t``); every
  other row gets ``o = 1``, ``τ = 1`` and an ``h`` simulated forward from the previous state with the
  copied ``(μ, φ, σ)`` (``h' = μ + φ(h − μ) + ση``; random walk ``h' = h + ση``); ``h_0`` is the old
  state just before the new first period when it exists, else ``μ`` (random walk: the old ``h_0``);
* the result is padded to the new ``T_pad`` (padded rows inert), accumulators are zero and the
  status is reset;
* CommonSV (P3 §5.5, E109): the common ``glob.vol.h`` (index ``t`` = data row ``t``, no presample
  state, exactly 0 on padded rows) is aligned by period in the same way, with every new row
  simulated forward with the copied ``(ρ, σ_h)``: ``h_t = ρh_{t−1} + σ_hη_t`` from the old state
  just before it (for a new first row, the old row of the previous period when it exists, else a
  stationary ``N(0, σ_h²/(1 − ρ²))`` draw); the normals come from ``eq_key(eq_key(key, c), n)``,
  a slot beyond the equations. ``ρ``, ``log σ_h`` and the ``λ_i`` are copied.

A warm start changes only the initialisation: burn-in still runs (``MCMCSettings.warm_burn``) and
the PRNG starts at ``it = 0`` with the new fit's own seed and data salt, so streams never repeat
across origins. When ``n``, the lags or the composition differ, :func:`incompatibility` names the
reason and the caller cold-starts with a warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import pandas as pd
from jax import random

from rai_bvar.engines.mcmc import (
    _jax,  # noqa: F401  (enables float64 before any computation)
    keys,
)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import ChainState, EngineData, leaf_path, state_template

__all__ = ["TIME_INDEXED", "FinalState", "incompatibility", "remap", "warm_state"]

Key = jax.Array

TIME_INDEXED: frozenset[str] = frozenset(
    {"eq.vol.core.h", "eq.vol.trans.o_idx", "eq.vol.trans.log_tau", "glob.vol.h"}
)
"""Leaves aligned by period label (every other non-accumulator leaf is copied)."""


@dataclass
class FinalState:
    """The final state of a fit (host NumPy leaves with a leading chain axis) and its window."""

    state: ChainState
    periods: pd.Index
    static: StaticConfig
    seed_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chains(self) -> int:
        return int(np.shape(self.state.status.code)[0])


def incompatibility(prev: FinalState, static_new: StaticConfig) -> str | None:
    """Why ``prev`` cannot warm-start a fit of ``static_new`` (``None`` if it can)."""
    a, b = prev.static, static_new
    for name in ("n", "lags", "n_det"):
        if getattr(a.shapes, name) != getattr(b.shapes, name):
            return f"{name} differs ({getattr(a.shapes, name)} → {getattr(b.shapes, name)})"
    if a.struct != b.struct or a.coef != b.coef or a.vol != b.vol:
        return "the composition differs"
    if (a.mean.kind, a.mean.coef_scaling) != (b.mean.kind, b.mean.coef_scaling):
        return "the mean differs"
    return None


def _row_map(old: pd.Index, new: pd.Index) -> np.ndarray:
    """``m[t'] = t`` when new row ``t'`` has the period of old row ``t``, else −1."""
    try:
        return np.asarray(pd.Index(old).get_indexer(pd.Index(new)), dtype=np.int64)
    except (TypeError, ValueError):  # incomparable labels (e.g. different frequencies)
        return np.full(len(new), -1, dtype=np.int64)


def _remap_latents(
    prev: dict[str, np.ndarray], tmpl: dict[str, np.ndarray], rows: np.ndarray, T_new: int,
    rw: bool, key: Key,
) -> dict[str, np.ndarray]:  # fmt: skip
    """New ``h``, ``o_idx``, ``log_tau`` for every chain and equation (module docstring rules)."""
    out: dict[str, np.ndarray] = {}
    for name in ("eq.vol.trans.o_idx", "eq.vol.trans.log_tau"):
        if name in tmpl:
            new = np.zeros_like(tmpl[name])
            have = rows >= 0
            new[..., np.nonzero(have)[0]] = prev[name][..., rows[have]]
            out[name] = new
    name = "eq.vol.core.h"
    if name not in tmpl:
        return out
    old_h = prev[name]
    C, n = old_h.shape[:2]
    new_h = np.zeros_like(tmpl[name])
    sigma = prev["eq.vol.core.sigma"]
    mu = None if rw else prev["eq.vol.core.mu"]
    phi = None if rw else prev["eq.vol.core.phi"]
    first = rows[0] if T_new else -1
    for c in range(C):
        kc = keys.eq_key(key, c)
        for i in range(n):
            eta = np.asarray(random.normal(keys.eq_key(kc, i), (max(T_new, 1),)), dtype=np.float64)
            if first >= 0:
                h_prev = old_h[c, i, first]  # state just before the new first period
            else:
                h_prev = old_h[c, i, 0] if rw else mu[c, i]
            new_h[c, i, 0] = h_prev
            for t in range(T_new):
                if rows[t] >= 0:
                    h_t = old_h[c, i, rows[t] + 1]
                elif rw:
                    h_t = h_prev + sigma[c, i] * eta[t]
                else:
                    h_t = mu[c, i] + phi[c, i] * (h_prev - mu[c, i]) + sigma[c, i] * eta[t]
                new_h[c, i, t + 1] = h_t
                h_prev = h_t
    out[name] = new_h
    return out


def _remap_common_h(
    prev: dict[str, np.ndarray], tmpl: dict[str, np.ndarray], rows: np.ndarray, T_new: int,
    before: int, key: Key,
) -> np.ndarray:  # fmt: skip
    """The new ``glob.vol.h`` (``(C, T_pad)``) of every chain (module docstring). ``before`` is the
    old row of the period just before the new first period (−1 when the old window lacks it)."""
    old_h = prev["glob.vol.h"]
    C = old_h.shape[0]
    n = int(prev["eq.vol.core.log_lam"].shape[1])
    rho = prev["glob.vol.rho"]
    sigma = np.exp(prev["glob.vol.log_sigma"])
    new_h = np.zeros_like(tmpl["glob.vol.h"])
    for c in range(C):
        kc = keys.eq_key(key, c)
        eta = np.asarray(random.normal(keys.eq_key(kc, n), (max(T_new, 1) + 1,)), np.float64)
        r, s = float(rho[c]), float(sigma[c])
        if before >= 0:
            h_prev = float(old_h[c, before])
        else:  # no old state before the window: the stationary law, scaled by the last normal
            h_prev = None
        for t in range(T_new):
            if rows[t] >= 0:
                h_t = float(old_h[c, rows[t]])
            elif h_prev is None:  # the new first row: h_0 ~ N(0, σ_h²/(1 − ρ²))
                h_t = s / np.sqrt((1.0 - r) * (1.0 + r)) * eta[-1]
            else:
                h_t = r * h_prev + s * eta[t]
            new_h[c, t] = h_t
            h_prev = h_t
    return new_h


def _period_before(old: pd.Index, new: pd.Index) -> int:
    """The old row of the period immediately before ``new[0]``, or −1."""
    if not len(new):
        return -1
    try:
        first = new[0]
        prev_period = first - 1
        loc = pd.Index(old).get_indexer(pd.Index([prev_period]))
        return int(loc[0])
    except (TypeError, ValueError, AttributeError):
        return -1


def remap(
    prev: FinalState,
    new_data: EngineData,
    new_periods: pd.Index,
    static_new: StaticConfig,
    key: Key,
) -> ChainState:
    """``prev`` moved onto the window of ``new_data`` (host NumPy, leading chain axis).

    ``new_periods`` labels the ``T_eff`` data rows of ``new_data``; ``key`` (a typed key) drives the
    forward simulation of ``h`` (folded with the chain, then the equation). Raises ``ValueError``
    when :func:`incompatibility` finds a reason.
    """
    reason = incompatibility(prev, static_new)
    if reason is not None:
        raise ValueError(f"cannot warm-start: {reason}")
    T_new = int(np.asarray(new_data.T_eff))
    if len(new_periods) != T_new:
        raise ValueError(f"new_periods has {len(new_periods)} labels for T_eff = {T_new}")
    C = prev.chains
    tmpl_tree = state_template(static_new)
    flat_t, treedef = jax.tree_util.tree_flatten_with_path(tmpl_tree)
    tmpl = {leaf_path(p): np.broadcast_to(x, (C, *np.shape(x))).copy() for p, x in flat_t}
    prev_flat = {
        leaf_path(p): np.asarray(x) for p, x in jax.tree_util.tree_flatten_with_path(prev.state)[0]
    }
    if set(prev_flat) != set(tmpl):
        raise ValueError("the final state does not have the structure of the new StaticConfig")
    rows = _row_map(prev.periods, new_periods)
    latents = _remap_latents(prev_flat, tmpl, rows, T_new, static_new.vol.rw, key)
    if "glob.vol.h" in tmpl:  # CommonSV (P3 §5.5, E109)
        before = _period_before(prev.periods, new_periods)
        latents["glob.vol.h"] = _remap_common_h(prev_flat, tmpl, rows, T_new, before, key)
    leaves = []
    for p, _ in flat_t:
        name = leaf_path(p)
        last = name.split(".")[-1]
        if name in latents:
            leaves.append(latents[name])
        elif name.startswith("status."):
            leaves.append(tmpl[name])  # code 0, it 0, min_pivot +inf
        elif last.startswith(("acc_", "rb_")):
            leaves.append(np.zeros_like(tmpl[name]))
        else:
            value = prev_flat[name]
            if value.shape != tmpl[name].shape:
                raise ValueError(
                    f"{name}: shape {value.shape} cannot be copied to {tmpl[name].shape}"
                )
            leaves.append(value.astype(tmpl[name].dtype, copy=True))
    return jax.tree.unflatten(treedef, leaves)


def warm_state(
    prev: FinalState | None,
    new_data: EngineData,
    new_periods: pd.Index,
    static_new: StaticConfig,
    key: Key,
) -> ChainState | None:
    """:func:`remap`, or ``None`` (a cold start) with a ``UserWarning`` naming the reason when
    ``prev`` cannot warm-start this fit (DESIGN §5.5: ``n``, ``p`` or the composition differ)."""
    if prev is None:
        return None
    reason = incompatibility(prev, static_new)
    if reason is not None:
        warnings.warn(f"warm start ignored ({reason}); cold start", UserWarning, stacklevel=2)
        return None
    return remap(prev, new_data, new_periods, static_new, key)
