"""Shared machinery of the Phase 3 Geweke suites (docs/phase3/DESIGN.md §9.3–§9.4; WP35).

The rows 301–330 (``validation.joint.COMPOSITIONS_P3``) and the block configurations 401–408
(``BLOCK_CONFIGS_P3``) are WP30's joint models; this module only sizes and runs them through the
Phase 2 machinery (``geweke_sequential``: ``C`` prior-started successive-conditional chains of
``M_c`` production sweeps against ``M₁`` marginal-conditional draws, Bonferroni over the test
functions, Gandy–Scott reruns at ``Δ = 4``), at the Phase 2 sizes (P3 §9.4):

* G-small ``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3`` (``size="small"``: ``p = 1``,
  ``T_eff = 20``);
* G-full ``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`` (``size="full"``: ``p = 2``, ``T_eff = 50``).

**Seeds.** The Phase 2 seeds of ``tests/geweke`` (one per tier: G-small 20261022, G-full
20261023, blocks 20261025); every Geweke key folds the CRC-32 of the row's name, so the Phase 3
rows run on streams of their own (an open choice of WP35: the design names no Phase 3 seed).

**SBC-only functions on the TruncatedNormal-ρ CommonSV rows** (E148, E170(c)). Under
``ρ ~ TN(0.8, 0.15²)`` the density is positive at ``ρ = ±1``, so ``Var(h_t) = ∞``: WP30's
harness already flags the ``h`` dates, ``mean h``, the log-likelihoods and the log-scale data
functions infinite (their ``atan`` transforms stay). E170(c) adds ``joint.W = Σ_t q_t e^{−h_t}`` and
``eq*.joint.rss_std``: extreme prior paths of ``h`` make the residuals reconstructed from simulated
data lose all precision (block 407: Geweke means of order 1e66–1e83, ``z ≡ 1``), so they carry no
power, and WP35 flags them as SBC-only on every TN row (rows 320–324, 328, 329 and blocks 407,
408; not row 325, whose ρ prior is the Beta). :func:`p3_spec` applies the flag test-side
(:class:`SpecP3`), so WP30's ``testfns`` is unchanged; the SBC quantities (``all_names``) keep
both functions.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from rai_bvar.validation.geweke import GewekeConfig, GewekeResult, geweke_sequential
from rai_bvar.validation.joint import (
    BLOCK_CONFIGS_P3,
    COMPOSITIONS_P3,
    JointSpecP3,
    composition_id,
    joint_spec,
)
from rai_bvar.validation.sequential import SequentialResult

G_SMALL: dict[str, Any] = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}
G_FULL: dict[str, Any] = {"C": 4000, "M_c": 500, "M1": 1_000_000, "alpha": 1e-2}
SEED_SMALL = 20261022
SEED_FULL = 20261023
SEED_BLOCK = 20261025

#: P3 §9.4 / §9.11: the rows at G-small on every PR (and G-full weekly with every other row)
PR_ROWS_P3 = frozenset({301, 311, 320, 321})
ROW_IDS = tuple(range(301, 331))
BLOCK_IDS = tuple(range(401, 409))
BY_ID: dict[int, str] = {composition_id(n): n for n in (*COMPOSITIONS_P3, *BLOCK_CONFIGS_P3)}
#: the G-full rows of each weekly shard file (P3 §9.11: ``geweke-p3a`` the structural rows,
#: ``geweke-p3b`` the reduced and CommonSV rows; ``tools/weekly_shards.py``)
FULL_ROWS_A = (301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 317, 318, 319, 326, 330)
FULL_ROWS_B = (311, 312, 313, 314, 315, 316, 327, 320, 321, 322, 323, 324, 325, 328, 329)

#: E170(c): the functions with no power under the TruncatedNormal ρ prior (SBC-only there)
_NO_POWER_TN = re.compile(r"^(joint\.W|eq\d+\.joint\.rss_std)$")


def truncated_normal_rho(spec: JointSpecP3) -> bool:
    """A CommonSV row or block whose ``ρ`` prior is the TruncatedNormal (E148)."""
    vol = spec.static.vol
    return bool(vol.common) and vol.csv_rho_prior == "truncated_normal"


@dataclass(frozen=True, eq=False)
class SpecP3(JointSpecP3):
    """A :class:`~rai_bvar.validation.joint.JointSpecP3` with extra test functions flagged
    SBC-only (``sbc_only``; module docstring): they leave :attr:`names` (the Geweke set) and
    stay in :attr:`all_names` (the SBC quantities)."""

    sbc_only: frozenset[str] = frozenset()

    @cached_property
    def function_flags(self) -> dict[str, bool]:
        flags = JointSpecP3.function_flags.func(self)
        return {k: fin and k not in self.sbc_only for k, fin in flags.items()}


def p3_spec(name: str, size: str = "small", **overrides: Any) -> JointSpecP3:
    """``joint_spec(name, size, **overrides)`` with E170(c)'s SBC-only functions on the
    TruncatedNormal-ρ rows (module docstring); other rows are returned unchanged."""
    spec = joint_spec(name, size, **overrides)
    if not truncated_normal_rho(spec):
        return spec
    drop = frozenset(k for k in spec.all_names if _NO_POWER_TN.match(k))
    fields = {f.name: getattr(spec, f.name) for f in dataclasses.fields(JointSpecP3)}
    return SpecP3(**fields, sbc_only=drop)


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
    """The evidence of one sequential Geweke run (every stage): ``G``, ``t_crit``, max ``|z|``
    and its function, min ``p`` and the failing functions."""
    stages = [
        {
            "C": r.C, "M_c": r.M_c, "M1": r.M1, "G": len(r.names), "t_crit": r.t_crit,
            "max_abs_z": float(np.max(np.abs(r.z))), "argmax": r.names[int(np.argmax(np.abs(r.z)))],
            "min_p": float(np.min(r.p)), "p_bonferroni": r.p_bonferroni, "failing": list(r.failing),
            "n_nonfinite": int(np.sum(r.n_nonfinite)), "min_rne": float(np.nanmin(r.rne)),
        }
        for r in results
    ]  # fmt: skip
    return {"name": name, "row": composition_id(name), "passed": seq.passed, "stage": seq.stage,
            "q": list(seq.q), "stages": stages}  # fmt: skip


def explain(seq: SequentialResult, results: list[GewekeResult]) -> str:
    """Failure message: the sequential decision and the largest ``|z|`` of the last stage."""
    return f"stages q={seq.q} betas={seq.betas}\n{results[-1].table().head(12)}"


def run_geweke(
    name: str, size: str, sizes: dict[str, Any], seed: int, **overrides: Any
) -> tuple[JointSpecP3, SequentialResult, list[GewekeResult], dict[str, Any]]:
    """Build :func:`p3_spec` and run ``geweke_sequential``; ``out`` is the record."""
    spec = p3_spec(name, size, **overrides)
    t0 = time.perf_counter()
    seq, results = geweke_sequential(spec, GewekeConfig(seed=seed, **sizes))
    out = summary(name, seq, results)
    out.update(size=size, seed=seed, sbc_only=sorted(getattr(spec, "sbc_only", ())),
               seconds=time.perf_counter() - t0)  # fmt: skip
    return spec, seq, results, out


def emit(kind: str, tier: str, out: dict[str, Any], record_property: Any) -> None:
    """Write the record (jsonl, junit property, and the job log's summary section)."""
    payload = {"tier": tier, **out}
    record(kind, payload)
    record_property("p3_record", json.dumps(payload, default=float))


__all__ = [
    "BLOCK_IDS",
    "BY_ID",
    "G_FULL",
    "G_SMALL",
    "PR_ROWS_P3",
    "ROW_IDS",
    "SEED_BLOCK",
    "SEED_FULL",
    "SEED_SMALL",
    "SpecP3",
    "emit",
    "explain",
    "p3_spec",
    "record",
    "run_geweke",
    "summary",
    "truncated_normal_rho",
]
