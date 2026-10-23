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

============================  ==============================================  ==================
Test                          Sizes                                           Tier
============================  ==============================================  ==================
``test_g_small``              rows 1–5, 7–10, 17, 18 and the SOC/SUR dummy    PR
                              rows 201–205 (WP25); ``size="small"``;
                              ``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``
``test_g_small_rw_h0_prior``  the random-walk SV rows at the RW ``h₀`` prior  PR (row 18),
                              ``N(−1, 4)`` (:data:`RW_H0_PRIOR`), G-small     nightly (row 14)
                              sizes
``test_g_small_block``        the seven block configurations of §9.2 through  nightly (``slow``)
                              ``JointSpec.step`` (their PR-tier G-small runs
                              live with the kernel owners, see below)
``test_g_full``               rows 1–18, 201–205; ``size="full"``;            nightly (``slow``)
                              ``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2``
``test_g_full_inexact``       the inexact variants: the SV rows of 1–16 and   nightly (``slow``),
                              18 with ``exact=False``, G-full sizes           recorded, not gating
============================  ==============================================  ==================

Deviations from DESIGN §9.10:

* the block configurations already run at G-small on every PR in the kernel work packages'
  suites — ``equation_only`` (``tests/mcmc/blocks/test_linear_structural.py``, also through the
  production sweep), ``contemp_only`` (``test_recursive.py``), ``cccm_only`` (``test_cccm.py``,
  also through the production sweep), ``aux_sv`` and ``gauss_sv_exact`` (``test_sv_core.py``),
  ``transitory_known_lambda`` and ``constant_transitory`` (``test_robust.py``) — so their
  ``JointSpec.step`` versions here are nightly;
* the inexact variants are recorded for the SV rows only: with a Constant core ``exact`` changes
  nothing (DESIGN §9.5 labelled them "rows 18+" before row 18 existed).

Measured locally (M1 Pro, 10 cores, other jobs running): G-small costs 5–55 s per composition
including compilation (the ``t`` rows dominate: their prior sampler and ``τ`` step draw ``T_pad``
gamma variates per equation). The ten DESIGN rows took 144 s with a cold compilation cache and
85 s with a warm persistent cache (``JAX_COMPILATION_CACHE_DIR``), 220 s under heavy load; the
nine PR rows of the WP19 set took about 2 min cold (measured before the dummy rows 201–205 and
row 18 joined the PR tier). G-full takes 34 s (row 17) to 9 min (rows 12–13) per composition at
stage 1, about 1 h for rows 1–17, more when the Gandy–Scott wrapper reruns a stage at 4×
(``structural_sv``: 13 min over three stages); the nightly job can be sharded by composition
(``-k <name>``; every test id carries the name).

Known nightly finding (2026-10-22): ``test_g_full[reduced_sv]`` failed on one SC chain of 4000
(sweep 194), which had reached the prior's extreme tail (``h₀ ≈ 30–36``, ``φ ≈ 0.999``,
``σ ≈ 0.9``; ``P(max h > 25) ≈ 1e-6`` per prior draw) where ``|y| ≈ 10⁸`` makes the
contemporaneous precision of equation 2 singular in float64 (Cholesky pivot 0; status code 1003,
``EQ_CONTEMP``, equation 2). The engine reported the failure as designed (D-18, no jitter).
**Resolved** (WP25 queue item 6, 2026-10-23) in the kernel, not the harness or the test prior: the
``a``-row draw is now the QR form of the whitened rows (``linalg.lstsq_draw``), which never forms
the Gram (``cond(K_a) = 1.3e16`` at that state, exact eigenvalues 14.8 and 1.9e17); the replay of
that chain is ``tests/mcmc/blocks/test_contemp_lstsq.py::test_g_full_reduced_sv_failing_chain_is_fixed``.

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

#: WP25: SOC/SUR dummy rows — structural Mode I × Constant (eigen, dense prior precision) and × SV
#: (weighted), Mode II × Constant("iw") (fixed_factor, coupled λ), reduced × Constant (CCCM) and
#: structural × Constant("iw_equivalent") (cholesky_const, λ-scaled a; rejection reference for λ | d)
DUMMY_ROWS = frozenset({201, 202, 203, 204, 205})
#: DESIGN §9.5 and the WP19 acceptance: rows 1–5, 7–10 and 17 at G-small on every PR, plus the
#: WP25 dummy rows (3–6 s each) and row 18 (reduced × RW SV: the Clark–Ravazzolo primary SV arm,
#: tiered like its reduced-form neighbours 7–10)
PR_ROWS = frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 17, 18}) | DUMMY_ROWS
PR_COMPOSITIONS = tuple(n for n in COMPOSITIONS if composition_id(n) in PR_ROWS)
#: the inexact variants: rows 1–16 and 18 with exact=False; only the SV rows differ from their
#: exact versions (row 17 and the dummy rows have none)
INEXACT_ROWS = frozenset({2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 16, 18})
INEXACT_COMPOSITIONS = tuple(n for n in COMPOSITIONS if composition_id(n) in INEXACT_ROWS)
#: the random-walk h₀ prior N(m0, V0) away from the §9.2 default N(0, 1), at which V0 used as a
#: precision (or m0 dropped) in the RW AWOL proposal or NC regression is numerically invisible; the
#: Clark–Ravazzolo arm runs at V0 = 4 with data-centred m0 (DESIGN §9.5, row 18 note)
RW_H0_PRIOR: dict[str, float] = {"V0": 4.0, "m0": -1.0}


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
    """The PR G-small set is rows 1–5, 7–10, 17 and 18 of DESIGN §9.5, resolved by row to the
    compositions the table names, plus the WP25 dummy rows 201–205."""
    assert {composition_id(n) for n in PR_COMPOSITIONS} == PR_ROWS
    by_row = {composition_id(n): n for n in COMPOSITIONS}
    assert [by_row[r] for r in (7, 8, 9, 10, 17, 18)] == [
        "reduced_constant", "reduced_sv", "reduced_sv_outliers", "reduced_sv_t", "zero_sv",
        "reduced_sv_rw",
    ]  # fmt: skip
    st18 = joint_spec("reduced_sv_rw").static  # the CR primary SV arm: reduced CCCM × RW SV
    assert (st18.mean.kind, st18.vol.persistent, st18.vol.rw, st18.vol.transitory) == (
        "linear_reduced", "sv", True, "none"
    )  # fmt: skip
    assert {composition_id(n) for n in COMPOSITIONS} == set(range(1, 19)) | DUMMY_ROWS
    for r in DUMMY_ROWS:
        assert by_row[r].endswith("_dummies") and joint_spec(by_row[r]).static.mean.n_dummy > 0
    for r in range(1, 19):
        assert joint_spec(by_row[r]).static.mean.n_dummy == 0
    assert len(BLOCK_CONFIGS) == 7
    for name in COMPOSITIONS:  # the inexact variants are exactly the SV rows among 1–16 and 18
        sv = joint_spec(name).static.vol.persistent == "sv"
        row = composition_id(name)
        assert (row in INEXACT_ROWS) == (sv and row != 17 and row < 200), name


@pytest.mark.parametrize("name", PR_COMPOSITIONS)
def test_g_small(name):
    """G-small (DESIGN §9.5) of one composition through the production sweep: exact mode,
    ``T_eff = 20`` (``T_pad = 32``: masks exercised), the PR design of §9.2."""
    spec, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_SMALL)
    record("geweke", {"tier": "G-small", **out})
    assert spec.static.vol.exact and spec.fixed.T_eff % 32 != 0
    assert len(spec.names) >= 20
    assert seq.passed, explain(seq, results)


@pytest.mark.parametrize(
    "name", ["reduced_sv_rw", pytest.param("structural_sv_rw", marks=pytest.mark.slow)]
)
def test_g_small_rw_h0_prior(name):
    """G-small of the random-walk SV rows (18: PR; 14: nightly) at the RW ``h₀`` prior
    :data:`RW_H0_PRIOR` instead of the §9.2 default ``N(0, 1)``. At the default a ``V0``-as-
    precision mutant of the RW AWOL step passes like production (row 18: max ``|z|`` 1.98, the
    row's own value); here it fails at ``log_prior`` (row 18: 13.8 against this test's 2.33;
    row 14: 11.3 against 1.39; t_crit 4.42 / 4.29; DESIGN §9.5 row 18 note, 2026-10-23)."""
    kw = {"prior_kwargs": RW_H0_PRIOR}
    spec, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_SMALL, **kw)
    record("geweke", {"tier": "G-small-rw-h0", "prior_kwargs": RW_H0_PRIOR, **out})
    core = spec.prior.eq.vol_core
    assert spec.static.vol.rw and spec.static.vol.exact
    assert np.all(np.asarray(core.V0) == RW_H0_PRIOR["V0"])
    assert np.all(np.asarray(core.m0) == RW_H0_PRIOR["m0"])
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
    """The inexact variants (DESIGN §9.5; N06 §8; N01 §12.3(3)): the mixture-approximation sampler
    (``exact=False``) at G-full sizes. **Recorded, not gating**: the statistic is written to the
    record (and the JUnit properties); only a sampler failure (a non-finite test function) fails,
    because that is a bug whatever the approximation."""
    _, seq, results, out = run_geweke(name, "full", G_FULL, SEED_INEXACT, exact=False)
    out["gating"] = False
    record("geweke", {"tier": "G-full-inexact", **out})
    record_property("geweke_inexact", json.dumps(out, default=float))
    assert all(int(np.sum(r.n_nonfinite)) == 0 for r in results), explain(seq, results)
