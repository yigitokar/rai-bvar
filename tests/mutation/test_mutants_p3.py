"""Tier M of Phase 3: every mutant M19–M36 must fail some tier (docs/phase3/DESIGN.md §9.6;
WP35).

The mutants, their seams and their checks live in ``mutants_p3.py``. ``test_mutant_caught_p3``
runs each confirmed check of each mutant (``MUTANTS[id].checks``) on the mutant and on its
control and requires the check to catch the mutant (the control passes, the mutant fails).
Weekly (markers ``mutation``, ``weekly`` and ``slow``). ``test_registry_p3`` (every PR, no JAX
work) requires M19–M36 to be registered, each with at least one check of a tier §9.6 names: a
mutant caught by none blocks the release (§9.6).

``RAI_BVAR_VALIDATION_RECORD=<dir>`` appends one JSON line per check to
``<dir>/mutation.jsonl``; the same record is a JUnit property and is printed in the session's
"Phase 3 validation records".
"""

from __future__ import annotations

import json
import os

import pytest
from mutants_p3 import MUTANTS, run_check

from rai_bvar.engines.mcmc.protocols import INTERNALS

CAUGHT = [(mid, k) for mid, m in MUTANTS.items() for k in range(len(m.checks))]
IDS = [f"{mid}-{MUTANTS[mid].checks[k].label}" for mid, k in CAUGHT]
DTYPE = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def record(payload: dict) -> None:
    root = os.environ.get("RAI_BVAR_VALIDATION_RECORD")
    if root:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "mutation.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")


def test_registry_p3():
    """M19–M36 are registered, each injected through the seam §9.6 names (a kernel internal of
    ``protocols.INTERNALS``, or the ``_update`` argument for M33) and with at least one check of
    a tier §9.6 names."""
    assert list(MUTANTS) == [f"M{i}" for i in range(19, 37)]
    for m in MUTANTS.values():
        assert m.checks, f"{m.id} is caught by no tier"
        assert m.title and m.injected and m.design, m.id
        assert set(m.internals) <= INTERNALS, m.id
        assert m.injected.split()[0] in (*m.internals, "_update"), m.id
        for c in m.checks:
            assert c.tier in m.design, (m.id, c.label)
            assert (c.u_check is None) == (c.tier == "G"), (m.id, c.label)
    assert not MUTANTS["M33"].internals and MUTANTS["M33"].injected == "_update"


@pytest.mark.mutation
@pytest.mark.weekly
@pytest.mark.slow
@DTYPE
@pytest.mark.parametrize(("mid", "k"), CAUGHT, ids=IDS)
def test_mutant_caught_p3(mid, k, record_property):
    """The check catches the mutant: its control passes and the mutant fails."""
    out = run_check(MUTANTS[mid], k)
    record(out)
    record_property("p3_record", json.dumps(out, default=float))
    assert out["caught"], out
