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
from rai_bvar.validation.sequential import bonferroni, thresholds


def record(payload: dict) -> None:
    root = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if root:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "sbc.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")


def sbc_config(name: str) -> SBCConfig:
    """Row ``name``'s SBC configuration: :data:`sbc_p3.SBC_CONFIG` with the row's
    :data:`sbc_p3.SBC_OVERRIDES` (Q44, E179), ``N = SBC_N`` and the seed :data:`sbc_p3.SEED`."""
    sizes = {**SBC_CONFIG, **SBC_OVERRIDES.get(composition_id(name), {})}  # Q44, E179
    return SBCConfig(N=SBC_N, seed=SEED, **sizes)


def run_sbc(name: str, record_property: Any) -> tuple[bool, dict[str, Any], Any]:
    """SBC of row ``name``: ``(passed, record, last stage)``."""
    model, quantities, names = sbc_model(name, "full")
    cfg = sbc_config(name)
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
        "min_ess": cfg.min_ess, "L": cfg.L, "alpha": cfg.alpha,
    }  # fmt: skip
    record(out)
    record_property("p3_record", json.dumps(out, default=float))
    return seq.passed, out, last


#: The opening of the reason ``validation.sbc`` gives when the pooled kept-draw ESS gate fails.
POOLED_REASON = "pooled kept-draw ESS below"


def recorded_not_met(last: Any, out: dict[str, Any]) -> bool:
    """P3 E180: the run has E179's recorded shape, a failure on the pooled kept-draw ESS gate
    alone. Every condition below must hold:

    - **the shape:** a stage-1 decision (``out["stage"] == 1``, ``out["q"] == [0.0]``);
    - **the γ test clean:** no γ rejection (``last.failing`` empty), and the Bonferroni γ value
      alone above Gandy–Scott's stage-1 pass bound ``γ + β₁``, so no γ result in the continue band
      can hide behind the ESS failure;
    - **no failed replication**;
    - **the per-replication ESS rule met** (at least ``ess_coverage`` of the replications);
    - **the pooled gate itself below** ``min_ess·L`` for some quantity, and the non-γ checks failed
      (``not last.checks_ok``);
    - **every recorded reason the pooled-ESS one**.

    Anything else is not E179's recorded outcome."""
    reasons = tuple(last.reasons)
    gamma, betas = thresholds(out["alpha"])
    pooled = last.ess_pooled
    pooled_low = pooled is not None and bool(
        np.any(np.isfinite(pooled) & (np.asarray(pooled) < out["min_ess"] * out["L"]))
    )
    return (
        out["stage"] == 1
        and list(out["q"]) == [0.0]
        and bool(reasons)
        and not last.checks_ok
        and not last.failing
        and bonferroni(last.p) > gamma + betas[0]
        and not np.any(last.failed)
        and last.ess_ok >= out["ess_coverage"]
        and pooled_low
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
    if not recorded_not_met(last, out):
        pytest.fail(
            f"row {row} failed for a reason E179 does not record (Q45, E180): {list(last.reasons)}"
        )
    pytest.xfail(
        f"not met (E179): row {row} fails on the pooled kept-draw ESS gate alone (Q45, E180): "
        f"{list(last.reasons)}"
    )
