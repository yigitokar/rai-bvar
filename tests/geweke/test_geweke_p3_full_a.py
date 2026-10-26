"""Tier G of Phase 3, weekly shard ``geweke-p3a``: G-full of the structural rows without CommonSV
(docs/phase3/DESIGN.md §9.4, §9.11; WP35).

Rows 301–310, 317–319, 326 and 330 (``geweke_p3.FULL_ROWS_A``): structural Mode I (and 318's
``Constant("iw_equivalent")``) with every global-local family and SSVS, on the exogenous design
(D-56) except the lagged rows 319 (NG, tight scales) and 330 (SSVS, tight slab). ``size="full"``:
``n = 3`` (§9.4's "2 / 3" rows run ``n = 3`` here), ``p = 2``, ``T_eff = 50``; G-full
``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`` inside the Gandy–Scott wrapper. The machinery and the
record are ``geweke_p3.py``'s.
"""

from __future__ import annotations

import pytest
from geweke_p3 import BY_ID, FULL_ROWS_A, G_FULL, SEED_FULL, emit, explain, run_geweke

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name", [BY_ID[r] for r in FULL_ROWS_A])
def test_g_full(name, record_property):
    """G-full (P3 §9.4) of one row through the production sweep."""
    spec, seq, results, out = run_geweke(name, "full", G_FULL, SEED_FULL)
    emit("geweke", "G-full", out, record_property)
    assert spec.static.shapes.lags == 2 and spec.fixed.T_eff == 50
    assert seq.passed, explain(seq, results)
