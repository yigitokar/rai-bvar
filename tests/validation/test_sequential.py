"""Multiplicity and the Gandy–Scott sequential wrapper (N06 §3.5, §4.6; Gandy & Scott 2021 Alg. 3)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from rai_bvar.validation.sequential import (
    bonferroni,
    expected_extra_effort,
    gandy_scott,
    holm_reject,
    stage_multiplier,
    stage_seed,
    thresholds,
)


@pytest.mark.parametrize(
    ("alpha", "k", "gamma", "betas"),
    [
        (1e-5, 7, 0.146, (1.43e-6, 9.77e-6, 6.68e-5, 4.57e-4, 3.13e-3, 0.0214, 0.146)),
        (1e-3, 3, 0.0693, (3.33e-4, 4.81e-3, 0.0693)),
        (1e-2, 3, 0.149, (3.33e-3, 0.0223, 0.149)),
    ],
)
def test_thresholds_match_n06_table(alpha, k, gamma, betas):
    g, b = thresholds(alpha, k)
    assert g == pytest.approx(gamma, rel=5e-3)
    np.testing.assert_allclose(b, betas, rtol=5e-3)
    assert b[-1] == pytest.approx(g, rel=1e-12)  # β_k = γ


@pytest.mark.parametrize(
    ("alpha", "k", "eq1", "eq2"),
    [(1e-5, 7, 0.685, 0.685), (1e-3, 3, 0.297, 0.299), (1e-2, 3, 0.687, 0.716)],
)
def test_expected_extra_effort_matches_n06_table(alpha, k, eq1, eq2):
    exact, bound = expected_extra_effort(alpha, k, delta=4)
    assert exact == pytest.approx(eq1, abs=1.5e-3)
    assert bound == pytest.approx(eq2, abs=1.5e-3)
    assert bound >= exact


def scripted(qs):
    calls = []

    def run(stage):
        calls.append(stage)
        return qs[stage]

    return run, calls


def test_decisions_per_stage():
    gamma, betas = thresholds(1e-3, 3)
    cases = [
        ((1e-4, 0.5, 0.5), False, 1),  # q ≤ β₁: fail at once
        ((0.5, 0.5, 0.5), True, 1),  # q > γ + β₁: pass at once
        ((0.01, 1e-3, 0.5), False, 2),  # marginal, then q ≤ β₂
        ((0.01, 0.2, 0.5), True, 2),  # marginal, then clear pass
        ((0.01, 0.01, 0.05), False, 3),  # last stage: q ≤ β₃ = γ fails
        ((0.01, 0.01, 0.1), True, 3),  # last stage: q > β₃ passes (always decides)
    ]
    for qs, passed, stage in cases:
        run, calls = scripted(qs)
        res = gandy_scott(run, 1e-3, k=3, delta=4)
        assert (res.passed, res.stage) == (passed, stage), qs
        assert calls == list(range(stage))  # fresh experiment per stage, 0-based index
        assert res.q == qs[:stage]
        assert res.multipliers == (1, 4, 4)[:stage]  # Δ applied once, after stage 1
        assert res.gamma == pytest.approx(gamma) and res.betas == betas


def test_boundaries_are_inclusive_for_failure():
    gamma, betas = thresholds(1e-2, 3)
    assert not gandy_scott(lambda s: betas[0], 1e-2).passed
    res = gandy_scott(scripted((gamma + betas[0], 1.0, 1.0))[0], 1e-2)
    assert res.stage == 2  # q = γ + β₁ is still marginal
    assert not gandy_scott(lambda s: float("nan"), 1e-2).passed  # non-finite experiment fails


def test_type_one_error_and_effort_under_uniform_pvalues():
    # Theorem 3.1: P(reject) ≤ α; eq. 1: E[extra effort] = Δγ(1 − γ^{k−1})/(1 − γ).
    rng = np.random.default_rng(2026)
    alpha, k, delta, reps = 0.05, 3, 4, 40_000
    rejected, effort = 0, 0.0
    for _ in range(reps):
        res = gandy_scott(lambda s: rng.uniform(), alpha, k, delta)
        rejected += not res.passed
        effort += sum(res.multipliers) - 1
    rate = rejected / reps
    assert rate <= alpha + 4 * np.sqrt(alpha * (1 - alpha) / reps)
    assert rate > 0.8 * alpha  # uniform p-values: the bound is nearly attained
    assert effort / reps == pytest.approx(expected_extra_effort(alpha, k, delta)[0], rel=0.05)


def test_power_against_small_pvalues():
    rng = np.random.default_rng(1)
    res = [gandy_scott(lambda s: rng.uniform() * 1e-3, 1e-2).passed for _ in range(200)]
    assert not any(res)


def test_bonferroni_and_holm():
    p = np.array([0.001, 0.02, 0.04, 0.3])
    assert bonferroni(p) == pytest.approx(0.004)
    assert bonferroni([0.5, 0.9]) == 1.0
    assert bonferroni([0.1, np.nan]) == 0.0
    np.testing.assert_array_equal(holm_reject(p, 0.05), [True, False, False, False])
    # Holm steps down: 0.01 ≤ 0.05/4, 0.015 ≤ 0.05/3, 0.02 ≤ 0.05/2, 0.3 > 0.05
    np.testing.assert_array_equal(
        holm_reject([0.3, 0.02, 0.01, 0.015], 0.05), [False, True, True, True]
    )
    np.testing.assert_array_equal(holm_reject([np.nan, 0.9], 0.05), [True, False])
    # Holm's global decision equals Bonferroni's (N06 §3.5)
    rng = np.random.default_rng(4)
    for _ in range(500):
        q = rng.uniform(0, 0.1, size=7)
        assert holm_reject(q, 0.05).any() == (bonferroni(q) <= 0.05)


def test_stage_seeds_and_multipliers():
    assert stage_seed(123, 0) == 123
    seeds = {stage_seed(123, s) for s in range(1, 50)} | {stage_seed(124, s) for s in range(1, 50)}
    assert len(seeds) == 98 and all(0 <= s < 2**63 for s in seeds)
    assert stage_seed(123, 2) == stage_seed(123, 2)
    assert [stage_multiplier(s, 4) for s in range(3)] == [1, 4, 4]


def test_validation():
    with pytest.raises(ValueError, match="alpha"):
        gandy_scott(lambda s: 0.5, 1.5)
    with pytest.raises(ValueError, match="k must"):
        gandy_scott(lambda s: 0.5, 0.05, k=0)
    with pytest.raises(ValueError, match="delta"):
        gandy_scott(lambda s: 0.5, 0.05, delta=0)


def test_package_import_is_lazy_and_jax_free():
    code = textwrap.dedent(
        """
        import sys
        import rai_bvar.validation as v
        assert "jax" not in sys.modules, "importing rai_bvar.validation loaded JAX"
        assert {"geweke", "sbc", "block_sbc", "gandy_scott", "gamma_pvalue"} <= set(v.__all__)
        from rai_bvar.validation import gandy_scott, geweke
        assert gandy_scott.__module__ == "rai_bvar.validation.sequential"
        assert geweke.__module__ == "rai_bvar.validation.geweke"
        import jax
        assert jax.config.jax_enable_x64  # the drivers enable float64 on import (SPEC §8)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
