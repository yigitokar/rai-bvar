"""Tier G: Geweke joint-distribution tests of every composition through the production sweep
(DESIGN §9.5, §13 WP19; N06 §3).

Each test builds ``validation.joint.joint_spec(name, size)`` (WP9), whose ``step`` is
``prepare(data, prior)`` followed by one **production** sweep (``compose`` → ``make_sweep``, WP14),
and runs :func:`~rai_bvar.validation.geweke.geweke_sequential` (WP7): ``C`` prior-started
successive-conditional chains of ``M_c`` sweeps without burn-in against ``M₁`` iid
marginal-conditional draws,

    z_g = (ḡ_MC − ḡ_SC)/√(σ̂²_g/M₁ + s²_C/C),   fail iff some p_g ≤ α/G (Bonferroni, t_{C−1}),

inside the Gandy–Scott wrapper (``k = 3``, ``Δ = 4``): a marginal Bonferroni p-value triggers a
fresh run with ``C`` and ``M₁`` four times larger instead of a red build (family level α).

=========================  =================================================  ==================
Test                       Sizes                                              Tier
=========================  =================================================  ==================
``test_g_small``           rows 1–5, 7–10, 17; ``size="small"``;              PR
                           ``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``
``test_g_small_block``     the seven block configurations of §9.2 through     nightly (``slow``)
                           ``JointSpec.step`` (their PR-tier G-small runs
                           live with the kernel owners, see below)
``test_g_full``            rows 1–17; ``size="full"``;                        nightly (``slow``)
                           ``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2``
``test_g_full_inexact``    rows 18+: the SV rows of 1–16 with                 nightly (``slow``),
                           ``exact=False``, G-full sizes                       recorded, not gating
=========================  =================================================  ==================

Deviations from DESIGN §9.10:

* the block configurations already run at G-small on every PR in the kernel work packages'
  suites — ``equation_only`` (``tests/mcmc/blocks/test_linear_structural.py``, also through the
  production sweep), ``contemp_only`` (``test_recursive.py``), ``cccm_only`` (``test_cccm.py``,
  also through the production sweep), ``aux_sv`` and ``gauss_sv_exact`` (``test_sv_core.py``),
  ``transitory_known_lambda`` and ``constant_transitory`` (``test_robust.py``) — so their
  ``JointSpec.step`` versions here are nightly;
* rows 18+ are recorded for the SV rows only: with a Constant core ``exact`` changes nothing.

Measured locally (M1 Pro, 10 cores, other jobs running): G-small costs 5–55 s per composition
including compilation (the ``t`` rows dominate: their prior sampler and ``τ`` step draw ``T_pad``
gamma variates per equation). The ten DESIGN rows took 144 s with a cold compilation cache and
85 s with a warm persistent cache (``JAX_COMPILATION_CACHE_DIR``), 220 s under heavy load; the
nine PR rows take about 2 min cold. G-full takes 34 s (row 17) to 9 min (rows 12–13) per
composition at stage 1, about 1 h for all 17 rows, more when the Gandy–Scott wrapper reruns a
stage at 4× (``structural_sv``: 13 min over three stages); the nightly job can be sharded by
composition (``-k <name>``; every test id carries the name).

Known nightly finding (2026-10-22): ``test_g_full[reduced_sv]`` failed on one SC chain of 4000
(sweep 194), which had reached the prior's extreme tail (``h₀ ≈ 30–36``, ``φ ≈ 0.999``,
``σ ≈ 0.9``; ``P(max h > 25) ≈ 1e-6`` per prior draw) where ``|y| ≈ 10⁸`` makes the
contemporaneous precision of equation 2 singular in float64 (Cholesky pivot 0; status code 1003,
``EQ_CONTEMP``, equation 2). The engine reports the failure as designed (D-18, no jitter) and the
test counts it; the remedy belongs to the kernel (a QR form of the A-row draw) or to the test
prior's ``h`` tail, not to this harness.

Seeds are fixed (one per tier); the Geweke keys fold the CRC-32 of the composition name
(``validation_key(seed, Tier.G, crc32(name), chain)``), so compositions use disjoint streams.
Setting ``RAI_BVAR_VALIDATION_RECORD=<dir>`` appends one JSON line per run to
``<dir>/geweke.jsonl`` (the record of the non-gating rows and the evidence for the others).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rai_bvar.validation.geweke import GewekeConfig, GewekeResult, geweke_sequential
from rai_bvar.validation.joint import BLOCK_CONFIGS, COMPOSITIONS, composition_id, joint_spec
from rai_bvar.validation.sequential import SequentialResult

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]

G_SMALL: dict[str, Any] = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}
G_FULL: dict[str, Any] = {"C": 4000, "M_c": 500, "M1": 1_000_000, "alpha": 1e-2}
SEED_SMALL = 20261022
SEED_FULL = 20261023
SEED_INEXACT = 20261024
SEED_BLOCK = 20261025

#: DESIGN §9.5 and the WP19 acceptance: rows 1–5, 7–10 and 17 at G-small on every PR
PR_ROWS = frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 17})
PR_COMPOSITIONS = tuple(n for n in COMPOSITIONS if composition_id(n) in PR_ROWS)
#: rows 18+: rows 1–16 with exact=False; only the SV rows differ from their exact versions
INEXACT_ROWS = frozenset({2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16})
INEXACT_COMPOSITIONS = tuple(n for n in COMPOSITIONS if composition_id(n) in INEXACT_ROWS)


# --------------------------------------------------------------------------- helpers


def record(kind: str, payload: dict[str, Any]) -> None:
    """Append ``payload`` to ``$RAI_BVAR_VALIDATION_RECORD/<kind>.jsonl`` when the variable is set."""
    root = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{kind}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=float) + "\n")


def summary(name: str, seq: SequentialResult, results: list[GewekeResult]) -> dict[str, Any]:
    """The evidence of one sequential Geweke run (every stage)."""
    stages = [
        {
            "C": r.C, "M_c": r.M_c, "M1": r.M1, "G": len(r.names), "t_crit": r.t_crit,
            "max_abs_z": float(np.max(np.abs(r.z))), "argmax": r.names[int(np.argmax(np.abs(r.z)))],
            "p_bonferroni": r.p_bonferroni, "failing": list(r.failing),
            "n_nonfinite": int(np.sum(r.n_nonfinite)), "min_rne": float(np.nanmin(r.rne)),
        }
        for r in results
    ]  # fmt: skip
    return {"name": name, "passed": seq.passed, "stage": seq.stage, "q": list(seq.q),
            "stages": stages}  # fmt: skip


def explain(seq: SequentialResult, results: list[GewekeResult]) -> str:
    """Failure message: the sequential decision and the largest ``|z|`` of the last stage."""
    last = results[-1]
    return f"stages q={seq.q} betas={seq.betas}\n{last.table().head(12)}"


def run_geweke(name: str, size: str, sizes: dict[str, Any], seed: int, **overrides: Any):
    spec = joint_spec(name, size, **overrides)
    t0 = time.perf_counter()
    seq, results = geweke_sequential(spec, GewekeConfig(seed=seed, **sizes))
    out = summary(name, seq, results)
    out.update(size=size, seed=seed, overrides=sorted(overrides), seconds=time.perf_counter() - t0)
    return spec, seq, results, out


# --------------------------------------------------------------------------- PR tier


def test_pr_rows_are_the_design_rows():
    """The PR G-small set is rows 1–5, 7–10 and 17 of DESIGN §9.5, resolved by row to the
    compositions the table names."""
    assert {composition_id(n) for n in PR_COMPOSITIONS} == PR_ROWS
    by_row = {composition_id(n): n for n in COMPOSITIONS}
    assert [by_row[r] for r in (7, 8, 9, 10, 17)] == [
        "reduced_constant", "reduced_sv", "reduced_sv_outliers", "reduced_sv_t", "zero_sv"
    ]  # fmt: skip
    assert {composition_id(n) for n in COMPOSITIONS} == set(range(1, 18))
    assert len(BLOCK_CONFIGS) == 7
    for name in COMPOSITIONS:  # rows 18+ are exactly the SV rows among 1–16
        sv = joint_spec(name).static.vol.persistent == "sv"
        assert (composition_id(name) in INEXACT_ROWS) == (sv and composition_id(name) <= 16), name


@pytest.mark.parametrize("name", PR_COMPOSITIONS)
def test_g_small(name):
    """G-small (DESIGN §9.5) of one composition through the production sweep: exact mode,
    ``T_eff = 20`` (``T_pad = 32``: masks exercised), the PR design of §9.2."""
    spec, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_SMALL)
    record("geweke", {"tier": "G-small", **out})
    assert spec.static.vol.exact and spec.fixed.T_eff % 32 != 0
    assert len(spec.names) >= 20
    assert seq.passed, explain(seq, results)


# --------------------------------------------------------------------------- nightly


@pytest.mark.slow
@pytest.mark.parametrize("name", BLOCK_CONFIGS)
def test_g_small_block(name):
    """G-small of a block configuration (DESIGN §9.2 table) through ``JointSpec.step`` (production
    sweep, frozen blocks restored or held by identity internals)."""
    _, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_BLOCK)
    record("geweke", {"tier": "G-small-block", **out})
    assert seq.passed, explain(seq, results)


@pytest.mark.slow
@pytest.mark.parametrize("name", COMPOSITIONS)
def test_g_full(name):
    """G-full (DESIGN §9.5): ``n = 3`` (rows 14–16: 2; row 17: 1), ``p = 2``, ``T_eff = 50``
    (``T_pad = 64``), fixed stored presample; ``C = 4000``, ``M_c = 500``, ``M₁ = 10⁶``,
    ``α = 1e-2``."""
    spec, seq, results, out = run_geweke(name, "full", G_FULL, SEED_FULL)
    record("geweke", {"tier": "G-full", **out})
    assert spec.static.shapes.lags == 2 and spec.fixed.T_eff == 50
    assert seq.passed, explain(seq, results)


@pytest.mark.slow
@pytest.mark.parametrize("name", INEXACT_COMPOSITIONS)
def test_g_full_inexact_recorded(name, record_property):
    """Rows 18+ (DESIGN §9.5; N06 §8; N01 §12.3(3)): the mixture-approximation sampler
    (``exact=False``) at G-full sizes. **Recorded, not gating**: the statistic is written to the
    record (and the JUnit properties); only a sampler failure (a non-finite test function) fails,
    because that is a bug whatever the approximation."""
    _, seq, results, out = run_geweke(name, "full", G_FULL, SEED_INEXACT, exact=False)
    out["gating"] = False
    record("geweke", {"tier": "G-full-inexact", **out})
    record_property("geweke_inexact", json.dumps(out, default=float))
    assert all(int(np.sum(r.n_nonfinite)) == 0 for r in results), explain(seq, results)
