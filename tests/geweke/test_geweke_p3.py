"""Tier G of Phase 3: the PR-tier rows and the registry (docs/phase3/DESIGN.md §9.4, §9.11; WP35).

============================  ================================================  ==============
Test                          Rows                                              Tier
============================  ================================================  ==============
``test_g_small_p3``           301, 311, 320, 321 (§9.4 "PR + weekly");          PR
                              ``size="small"``; ``C = 1000, M_c = 100,
                              M₁ = 2·10⁵, α = 1e-3``
``test_g_full`` (rows)        301–330, ``size="full"``; ``C = 4000,             weekly
                              M_c = 500, M₁ = 10⁶, α = 1e-2``                   (``slow``)
                              (``test_geweke_p3_full_a1.py``, ``…_a2.py``:
                              the structural rows; ``…_b1.py``, ``…_b2.py``:
                              the reduced and CommonSV rows; one shard each,
                              P3 §9.11)
``test_g_small_block``        401–408 (``test_geweke_p3_blocks.py``)            weekly
============================  ================================================  ==============

The joint models, test priors and test functions are WP30's (``validation.joint``,
``COMPOSITIONS_P3``); the shared runner, sizes, seeds and E170(c)'s SBC-only functions are in
``geweke_p3.py``. WP31 and WP32 ran rows 301, 311, 319, 330 and 320, 321 ad hoc at G-small
(PRs #23, #25); these files are the suites that run them in CI (P3 §13 rules).

``RAI_BVAR_VALIDATION_RECORD=<dir>`` appends one JSON line per run to ``<dir>/geweke.jsonl``; the
same record is a JUnit property and is printed in the session's "Phase 3 validation records".
"""

from __future__ import annotations

import pytest
from geweke_p3 import (
    BLOCK_IDS,
    BY_ID,
    FULL_ROWS,
    G_SMALL,
    PR_ROWS_P3,
    ROW_IDS,
    SEED_SMALL,
    emit,
    explain,
    p3_spec,
    run_geweke,
    truncated_normal_rho,
)

from rai_bvar.validation.joint import BLOCK_CONFIGS_P3, COMPOSITIONS_P3, composition_id

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]

PR_COMPOSITIONS_P3 = tuple(BY_ID[r] for r in sorted(PR_ROWS_P3))


def test_registry_is_the_design_table():
    """The Phase 3 rows are 301–330 and the blocks 401–408 (P3 §9.3–§9.4); the PR rows are 301,
    311, 320, 321 and resolve to the compositions §9.4 names; the weekly shard files cover every
    row exactly once; E170(c)'s SBC-only functions are exactly ``joint.W`` and the ``rss_std`` of
    the TruncatedNormal-ρ rows."""
    assert [composition_id(n) for n in COMPOSITIONS_P3] == list(ROW_IDS)
    assert [composition_id(n) for n in BLOCK_CONFIGS_P3] == list(BLOCK_IDS)
    assert sorted(r for rows in FULL_ROWS.values() for r in rows) == list(ROW_IDS)  # once each
    assert PR_COMPOSITIONS_P3 == (
        "structural_constant_hs", "reduced_sv_hs_global", "conjugate_commonsv_dummies",
        "structural_commonsv",
    )  # fmt: skip
    for r in FULL_ROWS["a1"] + FULL_ROWS["a2"]:  # geweke-p3a*: structural rows without CommonSV
        st = p3_spec(BY_ID[r]).static
        assert st.mean.kind == "linear_structural" and not st.vol.common, r
    for r in FULL_ROWS["b1"] + FULL_ROWS["b2"]:  # geweke-p3b*: the reduced and CommonSV rows
        st = p3_spec(BY_ID[r]).static
        assert st.mean.kind == "linear_reduced" or st.vol.common, r
    s320 = p3_spec(BY_ID[320]).static
    assert (s320.mean.const_path, s320.mean.n_dummy > 0, s320.coef.kind) == (
        "shared_factor", True, "fixed_diagonal"
    )  # fmt: skip
    assert p3_spec(BY_ID[323]).static.mean.const_path == "weighted"
    assert p3_spec(BY_ID[329]).static.vol.csv_block == 7
    s325 = p3_spec(BY_ID[325]).static.vol
    assert (s325.csv_rho_prior, s325.csv_sigma2_prior) == ("beta", "scaled_chi2")
    tn = [r for r in (*ROW_IDS, *BLOCK_IDS) if truncated_normal_rho(p3_spec(BY_ID[r]))]
    assert tn == [320, 321, 322, 323, 324, 328, 329, 407, 408]
    for r in (320, 325, 407):  # the flags (each traces the test functions once)
        spec = p3_spec(BY_ID[r], "full")
        dropped = {k for k in spec.all_names if k not in spec.names}
        no_power = {k for k in spec.all_names if k == "joint.W" or k.endswith(".joint.rss_std")}
        assert len(no_power) == 4
        assert (no_power <= dropped) == (r != 325), r
        assert set(getattr(spec, "sbc_only", ())) == (no_power if r != 325 else set()), r


@pytest.mark.parametrize("name", PR_COMPOSITIONS_P3)
def test_g_small_p3(name, record_property):
    """G-small (P3 §9.4) of one PR row through the production sweep: ``T_eff = 20``
    (``T_pad = 32``: masks exercised), ``p = 1``, the §9.4 test prior."""
    spec, seq, results, out = run_geweke(name, "small", G_SMALL, SEED_SMALL)
    emit("geweke", "G-small", out, record_property)
    assert spec.fixed.T_eff % 32 != 0 and spec.static.shapes.lags == 1
    assert len(spec.names) >= 20
    assert seq.passed, explain(seq, results)
