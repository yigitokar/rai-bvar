"""Tier B of Phase 3 at the weekly cadence: G-small of the block configurations 401–408
(docs/phase3/DESIGN.md §9.3, §9.4 "blocks"; WP35).

=====  ================================  ================================================
Id     Configuration                     Live (held)
=====  ================================  ================================================
401    ``gl_scales_only_horseshoe``      local, group and hyper parameters of the family;
402    ``gl_scales_only_normal_gamma``   the coefficients redrawn from ``p(d | scales)``
403    ``gl_scales_only_r2d2``           each iteration; data irrelevant (WP30: only the
404    ``gl_scales_only_dirichlet_…``    production ``G_COEF_GLOBAL`` step runs; grid ``a``
405    ``gl_scales_only_ssvs``           with ``c = 1`` (NG), grids (R2D2, DL), Beta ``p``)
406    ``gl_mean_only``                  coefficients by the deviation path (HS scales held)
407    ``common_sv_only``                ``h, ρ, σ_h`` (``G_COMMON_SV`` alone; coefficients,
                                         ``λ`` held, ``q`` from the simulator)
408    ``conjugate_niw_only``            ``(θ_i, λ_i)`` by the shared-factor path and the
                                         coupled common core (``h`` held)
=====  ================================  ================================================

The per-PR block Geweke runs of the same configurations live in the kernel suites
(``tests/mcmc/blocks/test_global_local.py``: 401–406, WP31; ``test_common_sv.py``: 407–408,
WP32; P3 §9.3); here they run at G-small sizes (``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``) on
the PR design (``size="small"``), the Phase 2 precedent (``tests/geweke``'s
``test_g_small_block``), with the block seed 20261025. On 407 and 408 (TruncatedNormal ρ) ``W``
and ``rss_std`` are SBC-only (E170(c); ``geweke_p3.py``).
"""

from __future__ import annotations

import pytest
from geweke_p3 import BLOCK_IDS, BY_ID, G_SMALL, SEED_BLOCK, emit, explain, run_geweke

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name", [BY_ID[b] for b in BLOCK_IDS])
def test_g_small_block(name, record_property):
    """G-small of one block configuration (P3 §9.3) through ``JointSpecP3.step``."""
    spec, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_BLOCK)
    emit("geweke", "G-small-block", out, record_property)
    assert len(spec.names) >= 20
    assert seq.passed, explain(seq, results)
