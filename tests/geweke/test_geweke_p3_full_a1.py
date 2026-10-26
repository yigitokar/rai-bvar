"""Tier G of Phase 3, weekly shard ``geweke-p3a1``: G-full of the structural rows 301–308
(docs/phase3/DESIGN.md §9.4, §9.11; WP35).

Rows ``geweke_p3.FULL_ROWS["a1"]``: the global-local families on the exogenous design (D-56): the
horseshoe with ``Constant("ig")`` (301) and ``SV`` (302), NG (303 fixed ``a``, 304 grid ``a`` with
``c = a/2``), R2D2 (305, 306) and DL (307, 308), fixed and grid ``a``. ``size="full"``: ``n = 3``,
``p = 2``, ``T_eff = 50``; G-full ``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2`` inside the
Gandy–Scott wrapper. The machinery and the record are ``geweke_p3.py``'s.
"""

from __future__ import annotations

import pytest
from geweke_p3 import BY_ID, FULL_ROWS, G_FULL, SEED_FULL, emit, explain, run_geweke

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name", [BY_ID[r] for r in FULL_ROWS["a1"]])
def test_g_full(name, record_property):
    """G-full (P3 §9.4) of one row through the production sweep."""
    spec, seq, results, out = run_geweke(name, "full", G_FULL, SEED_FULL)
    emit("geweke", "G-full", out, record_property)
    assert spec.static.shapes.lags == 2 and spec.fixed.T_eff == 50
    assert seq.passed, explain(seq, results)
