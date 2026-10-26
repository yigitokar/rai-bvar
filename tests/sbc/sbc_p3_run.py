"""The SBC run of one Phase 3 row (docs/phase3/DESIGN.md §9.5; WP35), shared by the shard files
``test_sbc_p3_a.py`` and ``test_sbc_p3_b.py``: ``sbc_sequential`` on :func:`sbc_p3.sbc_model` at
:data:`sbc_p3.SBC_CONFIG` (``N = 1000, L = 99``, burn 300, pilot 1000, ``t_max = 200``, γ test with
Bonferroni at ``α = 1e-2``, both ESS gates, Gandy–Scott), and its record (``sbc.jsonl``, the JUnit
property and the session's "Phase 3 validation records")."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from sbc_p3 import SBC_CONFIG, SBC_N, SEED, iat_summary, nanmin, sbc_model

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
    cfg = SBCConfig(N=SBC_N, seed=SEED, **SBC_CONFIG)
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
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    record(out)
    record_property("p3_record", json.dumps(out, default=float))
    return seq.passed, out, last
