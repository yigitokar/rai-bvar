"""Density-ratio and Metropolis–Hastings identities (Grosse & Duvenaud 2014 eq. 1; N06 §5.1;
DESIGN §9.3 item 2).

**Exact conditionals.** For a block ``b`` with implemented conditional ``q_b(θ_b | θ_−b, y)`` and two
states ``x0, x1`` that differ only in ``θ_b``::

    log q_b(x1) − log q_b(x0)  ==  log p(x1, y) − log p(x0, y)

because ``p(θ_b | θ_−b, y) ∝ p(θ, y)``. The joint must be an independent implementation (prior
log-densities plus the likelihood; ``validation.log_joint``), never the block's own formulas.
Partially collapsed and augmented steps use the joint they target (N06 §5.1: the ``s`` and
``h | s`` steps use the augmented joint, the others the non-augmented one).

**MH steps.** The computed log acceptance ratio (before ``min(0, ·)``) must equal

    log π(x′) + log q(x | x′) − log π(x) − log q(x′ | x)

including every correction folded into it (exact-mode ``w`` terms, the single-accept IG-σ²
combination, ASIS Jacobians).

Both functions return the scaled error ``|LHS − RHS| / (1 + |RHS|)``; DESIGN §9.3 asserts
``≤ 1e-8`` (N06 §5.1: relative 1e-8 leaves room for float64 round-off in sums over ``T``).
Equal infinities (both sides ``−∞``) give 0; any other non-finite mismatch gives ``inf``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["density_ratio_error", "mh_ratio_error", "scaled_error"]


def _scalar(value: Any, name: str) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"{name} must return a scalar, got shape {arr.shape}")
    return float(arr.reshape(()))


def _difference(a: float, b: float) -> float:
    """``a − b`` with ``(−∞) − (−∞)`` and ``∞ − ∞`` treated as an exact 0 (same infinite term)."""
    if math.isinf(a) and a == b:
        return 0.0
    return a - b


def scaled_error(lhs: float, rhs: float) -> float:
    """``|lhs − rhs| / (1 + |rhs|)``; 0 for identical infinities, ``inf`` for other non-finite."""
    lhs, rhs = float(lhs), float(rhs)
    if math.isnan(lhs) or math.isnan(rhs):
        return math.inf
    if math.isinf(lhs) or math.isinf(rhs):
        return 0.0 if lhs == rhs else math.inf
    return abs(lhs - rhs) / (1.0 + abs(rhs))


def density_ratio_error(
    log_q: Callable[[Any], Any], log_joint: Callable[[Any], Any], x0: Any, x1: Any
) -> float:
    """Scaled error of ``log q(x1) − log q(x0) = log p(x1, y) − log p(x0, y)``.

    ``log_q(x)`` is the block's implemented conditional log-density at the block value of ``x``
    (normalised or not: constants cancel); ``log_joint(x)`` the independent joint log-density.
    ``x0`` and ``x1`` are full states (PyTrees) that differ only in the block.
    """
    lhs = _difference(_scalar(log_q(x1), "log_q"), _scalar(log_q(x0), "log_q"))
    rhs = _difference(_scalar(log_joint(x1), "log_joint"), _scalar(log_joint(x0), "log_joint"))
    return scaled_error(lhs, rhs)


def mh_ratio_error(
    log_alpha: Callable[[Any, Any], Any],
    log_target: Callable[[Any], Any],
    log_prop: Callable[[Any, Any], Any],
    x: Any,
    x_prop: Any,
) -> float:
    """Scaled error of an MH log acceptance ratio against its definition.

    ``log_alpha(x, x_prop)`` is the kernel's computed log ratio (before ``min(0, ·)``),
    ``log_target(x)`` the target log-density (up to a constant) and ``log_prop(to, frm)`` the
    proposal log-density ``log q(to | frm)``. The reference is
    ``log π(x_prop) + log q(x | x_prop) − log π(x) − log q(x_prop | x)``.
    """
    lhs = _scalar(log_alpha(x, x_prop), "log_alpha")
    forward = _scalar(log_target(x_prop), "log_target") + _scalar(log_prop(x, x_prop), "log_prop")
    backward = _scalar(log_target(x), "log_target") + _scalar(log_prop(x_prop, x), "log_prop")
    return scaled_error(lhs, _difference(forward, backward))
