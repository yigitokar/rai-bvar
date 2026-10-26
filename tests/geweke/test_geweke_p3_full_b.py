"""Tier G of Phase 3, weekly shard ``geweke-p3b``: G-full of the reduced-form and the CommonSV
rows (docs/phase3/DESIGN.md §9.4, §9.11; WP35).

Rows 311–316 and 327 (reduced form × SV or ``Constant("ig")`` with the global-local families and
SSVS; 316 lagged with tight NG scales) and 320–325, 328, 329 (``CommonSV``: the conjugate
shared-factor path with SOC/SUR rows (320), its weighted twin (323), the parity options (325,
Beta ρ and ``ScaledChi2`` σ_h²), interior blocks of 7 dates (329), structural Mode I (321, and ×
HS: 324), reduced (322, and × NG: 328)) (``geweke_p3.FULL_ROWS_B``). ``size="full"``: ``n = 3``,
``p = 2``, ``T_eff = 50``; G-full ``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2``. On the
TruncatedNormal-ρ rows the unbounded functions of ``h``, the log-likelihoods, ``W`` and
``rss_std`` are SBC-only (E148, E170(c); ``geweke_p3.py``).

Row 320 sits above D-56's explosive-mass target by E150's Monte Carlo rule (238 of 10⁶ prior
draws against the limit 234; Q36, pending; rows 321 and 323 are within it): that is WP30's
strict expected failure of the target test (``tests/validation``), not of this row, which runs
as stated.
"""

from __future__ import annotations

import pytest
from geweke_p3 import BY_ID, FULL_ROWS_B, G_FULL, SEED_FULL, emit, explain, run_geweke

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name", [BY_ID[r] for r in FULL_ROWS_B])
def test_g_full(name, record_property):
    """G-full (P3 §9.4) of one row through the production sweep."""
    spec, seq, results, out = run_geweke(name, "full", G_FULL, SEED_FULL)
    emit("geweke", "G-full", out, record_property)
    assert spec.static.shapes.lags == 2 and spec.fixed.T_eff == 50
    assert seq.passed, explain(seq, results)
