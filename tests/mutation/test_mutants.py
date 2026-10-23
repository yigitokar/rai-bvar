"""Tier M: every mutant M1–M18 must fail some tier (DESIGN §9.7, §9.10, §13 WP19; N06 §9.5).

The mutants, their injection seams, controls and tier checks live in ``mutants.py``.
``test_mutant_caught`` runs each confirmed catching tier of each mutant (``MUTANTS[id].catchers``)
on the mutant and on its control and requires the tier to catch the mutant (the control passes,
the mutant fails). Every tier that has a check for a mutant is a confirmed catcher (the registry's
``evidence`` records the statistics, and the observations that are not catches: M1 under
whole-scan invariance, M11 and M12 under the exact sampler). Weekly (markers ``mutation``,
``weekly`` and ``slow``). ``test_registry`` (every PR, no JAX work) requires a confirmed catching
tier for every mutant: a mutant caught by none blocks the release (DESIGN §9.7).

``RAI_BVAR_VALIDATION_RECORD=<dir>`` appends one JSON line per tier run to
``<dir>/mutation.jsonl``.
"""

from __future__ import annotations

import json
import os

import pytest
from mutants import MUTANTS, TIERS, available_tiers, run_tier

CAUGHT = [(mid, tier) for mid, m in MUTANTS.items() for tier in m.catchers]
DTYPE = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def record(payload: dict) -> None:
    root = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if root:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "mutation.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")


def test_registry():
    """M1–M18 are registered (M17–M18: the WP25 dummy-row mutants), each with at least one
    confirmed catching tier that has a check."""
    assert list(MUTANTS) == [f"M{i}" for i in range(1, 19)]
    for m in MUTANTS.values():
        assert m.catchers, f"{m.id} is caught by no tier"
        assert set(m.catchers) == set(available_tiers(m)) and set(m.catchers) <= set(TIERS), m.id
        assert m.design and m.injected and m.title and m.evidence, m.id


@pytest.mark.mutation
@pytest.mark.weekly
@pytest.mark.slow
@DTYPE
@pytest.mark.parametrize(("mid", "tier"), CAUGHT, ids=[f"{m}-{t}" for m, t in CAUGHT])
def test_mutant_caught(mid, tier, record_property):
    """The tier catches the mutant: its control passes and the mutant fails."""
    out = run_tier(MUTANTS[mid], tier)
    record(out)
    record_property("mutation", json.dumps(out, default=float))
    assert out["caught"], out
