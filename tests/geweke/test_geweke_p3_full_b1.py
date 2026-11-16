"""Tier G of Phase 3, weekly shard ``geweke-p3b1``: G-full of the reduced-form rows 311–316, 327
(docs/phase3/DESIGN.md §9.4, §9.11; WP35).

Rows ``geweke_p3.FULL_ROWS["b1"]``: reduced × ``SV`` with every family and global groups (311–315,
the parity compositions), the lagged 316 (``SV`` × NG, tight scales) and reduced ×
``Constant("ig")`` × HS (327). ``size="full"``: ``n = 3``, ``p = 2``, ``T_eff = 50``; G-full ``C =
4000, M_c = 500, M₁ = 10⁶, α = 1e-2`` inside the Gandy–Scott wrapper. The machinery and the record
are ``geweke_p3.py``'s.
"""

from __future__ import annotations

import pytest
from geweke_p3 import BY_ID, FULL_ROWS, G_FULL, SEED_FULL, emit, explain, run_geweke

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name", [BY_ID[r] for r in FULL_ROWS["b1"]])
def test_g_full(name, record_property):
    """G-full (P3 §9.4) of one row through the production sweep."""
    spec, seq, results, out = run_geweke(name, "full", G_FULL, SEED_FULL)
    emit("geweke", "G-full", out, record_property)
    assert spec.static.shapes.lags == 2 and spec.fixed.T_eff == 50
    assert seq.passed, explain(seq, results)
