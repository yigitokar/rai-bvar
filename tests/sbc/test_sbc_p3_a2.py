"""Tier S of Phase 3, weekly shard ``sbc-p3a2``: SBC of the structural rows 309, 310, 317–319, 326, 330
(docs/phase3/DESIGN.md §9.5, §9.11; WP35).

Rows ``sbc_p3.SBC_ROWS["a2"]``: SSVS (309 fixed ``p``, 310 ``p ~ Beta``), the horseshoe with
``SV(outliers=True)`` (317) and ``Constant("iw_equivalent")`` (318), NG with grid ``a`` and ``c =
1`` (326), and the lagged rows 319 (``SV`` × NG) and 330 (``SV`` × SSVS) with ``y_{T+1}`` and
``h_{T+1}`` through the production forecast core. The P2 §9.6 configuration (``N = 1000, L = 99``;
``sbc_p3.SBC_CONFIG``); the pilot-IAT tail behind the ESS rule is recorded (``iat_*``,
``ess_fail``).
"""

from __future__ import annotations

import pytest
from sbc_p3 import BY_ID, SBC_ROWS
from sbc_p3_run import run_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


@pytest.mark.slow
@pytest.mark.sbc
@pytest.mark.parametrize("name", [BY_ID[r] for r in SBC_ROWS["a2"]])
def test_sbc(name, record_property):
    """Full SBC (P3 §9.5) of one row."""
    passed, out, last = run_sbc(name, record_property)
    assert passed, (out, last.table().head(10))
