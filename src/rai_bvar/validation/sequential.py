"""Multiplicity corrections and the Gandy–Scott sequential wrapper (N06 §3.5, §4.6; DESIGN §9.4–9.5).

**Multiplicity** (N06 §3.5). The global decision over ``G`` tests is Bonferroni: fail iff
``min_g p_g ≤ α/G``, i.e. iff the Bonferroni-adjusted ``q = min(1, G·min p) ≤ α``. Holm rejects
exactly when Bonferroni does at the first step, so it is used only to *list* the failing tests.

**Sequential wrapper** (Gandy & Scott 2021, arXiv:2001.06465v2, Algorithm 3 and Theorem 3.1). With
family level ``α``, ``k`` stages and sample-size multiplier ``Δ``::

    β₁ = α/k,   γ = β₁^{1/k},   β_{i+1} = β_i/γ        (so β_k = γ)
    stage i:  q_i = Bonferroni-adjusted p-value of a FRESH, independent experiment
              q_i ≤ β_i          → fail
              q_i > γ + β_i      → pass
              otherwise          → continue (at the last stage: pass)

Under H0 each stage continues with probability at most ``γ`` and rejects with probability at most
``β_i``, so ``P(reject) ≤ Σ_i γ^{i−1}β_i = kβ₁ = α`` (Theorem 3.1). The sample size is multiplied by
``Δ`` once, after stage 1 (Algorithm 3, line 9); the expected extra effort under H0 is
``Δγ(1 − γ^{k−1})/(1 − γ)`` for exactly uniform p-values (eq. 1), bounded by
``Δ Σ_{i=2..k} Π_{j<i}(γ + β_j)`` (eq. 2). Theorem 3.1 needs the stages' p-values to be independent:
every stage must draw fresh keys and new samples (use :func:`stage_seed`), never extend the
previous stage's chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SequentialResult",
    "bonferroni",
    "expected_extra_effort",
    "gandy_scott",
    "holm_reject",
    "stage_multiplier",
    "stage_seed",
    "thresholds",
]


def bonferroni(p) -> float:
    """``min(1, G·min_g p_g)``; NaN p-values count as 0 (a non-finite statistic fails the test)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == 0:
        return 1.0
    p = np.where(np.isnan(p), 0.0, p)
    return float(min(1.0, p.size * p.min()))


def holm_reject(p, alpha: float) -> np.ndarray:
    """Holm step-down rejections at family level ``alpha`` (NaN p-values are rejected)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.where(np.isnan(p), 0.0, p)
    G = p.size
    reject = np.zeros(G, dtype=bool)
    for step, idx in enumerate(np.argsort(p, kind="stable")):
        if p[idx] > alpha / (G - step):
            break
        reject[idx] = True
    return reject


def _check(alpha: float, k: int, delta: int) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if isinstance(delta, bool) or int(delta) != delta or delta < 1:
        raise ValueError(f"delta must be a positive integer, got {delta!r}")


def thresholds(alpha: float, k: int = 3) -> tuple[float, tuple[float, ...]]:
    """``(γ, (β_1..β_k))`` of Algorithm 3."""
    _check(alpha, k, 1)
    beta1 = alpha / k
    gamma = beta1 ** (1.0 / k)
    return gamma, tuple(beta1 / gamma**i for i in range(k))


def expected_extra_effort(alpha: float, k: int = 3, delta: int = 4) -> tuple[float, float]:
    """Expected extra effort under H0 relative to stage 1: (eq. 1 exact-uniform, eq. 2 bound)."""
    _check(alpha, k, delta)
    gamma, betas = thresholds(alpha, k)
    exact = delta * gamma * (1.0 - gamma ** (k - 1)) / (1.0 - gamma)
    bound = delta * sum(float(np.prod([gamma + b for b in betas[:i]])) for i in range(1, k))
    return float(exact), float(bound)


def stage_multiplier(stage: int, delta: int = 4) -> int:
    """Sample-size multiplier of 0-based ``stage``: 1 at stage 0, ``Δ`` afterwards (line 9)."""
    return 1 if stage == 0 else int(delta)


def stage_seed(seed: int, stage: int) -> int:
    """Seed of 0-based ``stage``: ``seed`` itself at stage 0, an independent mix afterwards."""
    if stage == 0:
        return int(seed)
    words = np.random.SeedSequence([int(seed), int(stage)]).generate_state(2, dtype=np.uint32)
    return (int(words[0]) << 31 | int(words[1]) >> 1) & (2**63 - 1)


@dataclass(frozen=True)
class SequentialResult:
    """Outcome of :func:`gandy_scott`; ``q`` holds the adjusted p-value of each stage run."""

    passed: bool
    stage: int  # 1-based stage at which the decision was taken
    q: tuple[float, ...]
    betas: tuple[float, ...]
    gamma: float
    multipliers: tuple[int, ...]
    alpha: float
    k: int
    delta: int


def gandy_scott(
    run: Callable[[int], float], alpha: float, k: int = 3, delta: int = 4
) -> SequentialResult:
    """Gandy–Scott sequential test (Algorithm 3). ``run(stage)`` is called with the 0-based stage
    index and must return the Bonferroni-adjusted p-value of a fresh experiment at
    :func:`stage_multiplier` ``(stage, delta)`` times the base size, with keys from
    :func:`stage_seed` (independent across stages; Theorem 3.1)."""
    _check(alpha, k, delta)
    gamma, betas = thresholds(alpha, k)
    qs: list[float] = []
    for i in range(k):
        q = float(run(i))
        if np.isnan(q):
            q = 0.0  # a non-finite experiment fails
        qs.append(q)
        last = i == k - 1
        if q <= betas[i] or last or q > gamma + betas[i]:
            passed = q > betas[i]
            break
    stages = len(qs)
    return SequentialResult(
        passed=bool(passed),
        stage=stages,
        q=tuple(qs),
        betas=betas,
        gamma=float(gamma),
        multipliers=tuple(stage_multiplier(s, delta) for s in range(stages)),
        alpha=float(alpha),
        k=int(k),
        delta=int(delta),
    )
