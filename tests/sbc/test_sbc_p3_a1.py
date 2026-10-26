"""Tier S of Phase 3, weekly shard ``sbc-p3a1``: SBC of rows 303, 317 (docs/phase3/DESIGN.md §9.5,
§9.11; WP35).

Rows ``sbc_p3.SBC_ROWS["a1"]``, a part of the structural rows (``sbc-p3a`` of §9.11, split further:
estimated 80 min before the first dispatch, ``sbc_p3.SBC_ESTIMATE_MIN``), at the P2 §9.6
configuration (``N = 1000, L = 99``; ``sbc_p3.SBC_CONFIG``) with the joint models of ``sbc_p3.py``
(exogenous rows without a predictive; lagged rows with ``y_{T+1}`` through the production forecast
core). The pilot-IAT tail behind the ESS rule is recorded (``iat_*``, ``ess_fail``).
"""

from __future__ import annotations

import pytest
from sbc_p3 import BY_ID, SBC_ROWS
from sbc_p3_run import run_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


@pytest.mark.slow
@pytest.mark.sbc
@pytest.mark.parametrize("name", [BY_ID[r] for r in SBC_ROWS["a1"]])
def test_sbc(name, record_property):
    """Full SBC (P3 §9.5) of one row."""
    passed, out, last = run_sbc(name, record_property)
    assert passed, (out, last.table().head(10))
