"""Tier S of Phase 3, weekly shard ``sbc-p3a2``: SBC of rows 308 (docs/phase3/DESIGN.md §9.5, §9.11;
WP35).

Rows ``sbc_p3.SBC_ROWS["a2"]``, a part of the structural rows (``sbc-p3a`` of §9.11, split further;
``tools/weekly_shards.py``), at the P2 §9.6 configuration (``N = 1000, L = 99``;
``sbc_p3.SBC_CONFIG``). Row 308 runs at ``t_max = 400`` (Q44, E179; ``sbc_p3.SBC_OVERRIDES``). The
joint models are ``sbc_p3.py``'s (exogenous rows without a predictive; lagged rows with ``y_{T+1}``
through the production forecast core). The pilot-IAT tail behind the ESS rule is recorded
(``iat_*``, ``ess_fail``).
"""

from __future__ import annotations

import pytest
from sbc_p3 import BY_ID, SBC_ROWS
from sbc_p3_run import check_sbc, run_sbc

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


@pytest.mark.slow
@pytest.mark.sbc
@pytest.mark.parametrize("name", [BY_ID[r] for r in SBC_ROWS["a2"]])
def test_sbc(name, record_property):
    """Full SBC (P3 §9.5) of one row; the row E179 records as not met is a narrow strict xfail (Q45,
    E180; ``sbc_p3_run.check_sbc``)."""
    passed, out, last = run_sbc(name, record_property)
    check_sbc(name, passed, out, last)  # Q45, E180: a narrow strict xfail for the not-met row
