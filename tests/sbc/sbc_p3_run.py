"""The SBC run of one Phase 3 row (docs/phase3/DESIGN.md §9.5; WP35), shared by the shard files
``test_sbc_p3_<group>.py``: ``sbc_sequential`` on :func:`sbc_p3.sbc_model` at
:data:`sbc_p3.SBC_CONFIG` (``N = 1000, L = 99``, burn 300, pilot 1000, ``t_max = 200``, and 400 for
the eight rows of :data:`sbc_p3.SBC_OVERRIDES` by Q44, E179; γ test with Bonferroni at
``α = 1e-2``, both ESS gates, Gandy–Scott), and its record (``sbc.jsonl``, the JUnit property and
the session's "Phase 3 validation records")."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pytest
from sbc_p3 import (
    NOT_MET_ROWS,
    SBC_CONFIG,
    SBC_N,
    SBC_OVERRIDES,
    SEED,
    iat_summary,
    nanmin,
    sbc_model,
)

from rai_bvar.validation.joint import composition_id
from rai_bvar.validation.sbc import SBCConfig, sbc_sequential


def record(payload: dict) -> None:
    root = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if root:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "sbc.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")


def run_sbc(name: str, record_property: Any) -> tuple[bool, dict[str, Any], Any]:
    """SBC of row ``name``: ``(passed, record, last stage)``."""
    model, quantities, names = sbc_model(name, "full")
    sizes = {**SBC_CONFIG, **SBC_OVERRIDES.get(composition_id(name), {})}  # Q44, E179
    cfg = SBCConfig(N=SBC_N, seed=SEED, **sizes)
    t0 = time.perf_counter()
    seq, results = sbc_sequential(model, quantities, names, cfg)
    last = results[-1]
    out = {
        "tier": "S", "name": name, "row": composition_id(name), "model": type(model).__name__,
        "passed": seq.passed, "stage": seq.stage, "q": list(seq.q), "N": last.N,
        "t_max": cfg.t_max, "pilot": cfg.pilot, "Q": len(names), "reasons": list(last.reasons),
        "failing": list(last.failing), "min_p": float(np.min(last.p)),
        "argmin": names[int(np.argmin(last.p))], "ess_ok": last.ess_ok,
        "thin_max": int(np.max(last.thin)) if last.N else 0,
        "ess_pooled_min": nanmin(last.ess_pooled), **iat_summary(last, cfg.min_ess, cfg.L),
        "seconds": time.perf_counter() - t0, "ess_coverage": cfg.ess_coverage,
    }  # fmt: skip
    record(out)
    record_property("p3_record", json.dumps(out, default=float))
    return seq.passed, out, last


#: The opening of the reason ``validation.sbc`` gives when the pooled kept-draw ESS gate fails.
POOLED_REASON = "pooled kept-draw ESS below"


def recorded_not_met(last: Any, ess_coverage: float) -> bool:
    """P3 E180: the final SBC stage ``last`` fails on the pooled kept-draw ESS gate alone. The γ test
    is clean, no replication failed, the per-replication ESS rule is met (at least
    ``ess_coverage`` of the replications), and every recorded reason is the pooled-ESS one."""
    reasons = tuple(last.reasons)
    return (
        bool(reasons)
        and not last.failing
        and not np.any(last.failed)
        and last.ess_ok >= ess_coverage
        and all(r.startswith(POOLED_REASON) for r in reasons)
    )


def check_sbc(name: str, passed: bool, out: dict[str, Any], last: Any) -> None:
    """Row ``name``'s outcome: it must pass, except the rows E179 records as not met
    (:data:`sbc_p3.NOT_MET_ROWS`; Q45, E180). Such a row is XFAIL when it fails on the pooled
    kept-draw ESS gate alone (:func:`recorded_not_met`), FAILS on any other failure, and FAILS on a
    pass (an unexpected pass needs a re-decision)."""
    row = composition_id(name)
    if row not in NOT_MET_ROWS:
        assert passed, (out, last.table().head(10))
        return
    if passed:
        pytest.fail(
            f"row {row} passed although E179 records it not met: an unexpected pass needs a "
            "re-decision (Q45, E180)"
        )
    if not recorded_not_met(last, out["ess_coverage"]):
        pytest.fail(
            f"row {row} failed for a reason E179 does not record (Q45, E180): {list(last.reasons)}"
        )
    pytest.xfail(
        f"not met (E179): row {row} fails on the pooled kept-draw ESS gate alone (Q45, E180): "
        f"{list(last.reasons)}"
    )
