"""``grid()`` harmonisation of ``auto`` components (DESIGN §11.4, D-19; fixes E22).

* The SPEC §9.5 grid ``mean=[Linear(prior=Minnesota())]``, ``vol=[Constant(), SV(),
  SV(outliers=True)]`` becomes a one-axis comparison: structural Mode I in every arm, so the
  arms differ along ``vol`` only and pair in ``verdict(axis="vol")``.
* A grid of ``Constant()`` arms stays on the analytic engine, unchanged.
* Explicit settings are never changed; inputs are not mutated; the rewritten arms are listed in
  an INFO note.
* ``grid()`` calls ``factory.harmonise_grid`` only when the factory defines it, and labels the
  entries with the rewritten components; Mode I (``Constant(ig)``) and Mode II (``Constant``)
  arms get distinct entry names, so ``compare()`` accepts the grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

import rai_bvar as rb
from rai_bvar.evaluation import Entry, grid
from rai_bvar.evaluation.compare import _entries
from rai_bvar.models import spec_hash

AUTO_MEAN = rb.Linear(prior=rb.Minnesota())
STRUCTURAL = rb.Linear(prior=rb.Minnesota(form="structural"))


def _differing(a: Entry, b: Entry) -> list[str]:
    return sorted(k for k in a.axes if a.axes[k] != b.axes[k])


def test_spec_grid_becomes_a_one_axis_volatility_comparison():
    entries = grid(rb.BVAR, mean=[AUTO_MEAN], vol=[rb.Constant(), rb.SV(), rb.SV(outliers=True)])
    models = [e.model for e in entries]
    assert [m.engine for m in models] == ["mcmc"] * 3
    assert {m.mean.prior.form for m in models} == {"structural"}
    assert models[0].vol == rb.Constant("ig")  # Mode I, not the conjugate NIW
    assert {m.axes["prior"] for m in models} == {"Minnesota(structural, eb)"}
    assert [m.axes["vol"] for m in models] == ["Constant", "SV", "SV(outliers)"]
    for e in entries[1:]:
        assert _differing(e, entries[0]) == ["vol"]
        assert e.axes == e.model.axes
    assert [e.name for e in entries] == [
        "Linear[Minnesota(structural, eb)] × Constant(ig)",
        "Linear[Minnesota(structural, eb)] × SV",
        "Linear[Minnesota(structural, eb)] × SV(outliers)",
    ]
    # the Constant arm is the same model as a hand-built structural Mode I BVAR
    assert models[0] == rb.BVAR(mean=STRUCTURAL, vol=rb.Constant("ig"))
    assert spec_hash(models[1]) == spec_hash(rb.BVAR(mean=STRUCTURAL, vol=rb.SV()))


def test_vol_arms_pair_with_the_constant_reference_along_the_vol_axis():
    """The pairing rule of ``CompareResult.verdict(axis="vol")``: the reference takes the
    baseline's vol value and agrees on every other axis."""
    entries = grid(
        rb.BVAR,
        mean=[AUTO_MEAN, rb.Linear(rb.Minnesota(lambda_=0.1))],
        vol=[rb.Constant(), rb.SV()],
    )
    base_vol = entries[0].axes["vol"]
    for e in entries:
        if e.axes["vol"] != base_vol:
            target = {**e.axes, "vol": base_vol}
            refs = [f for f in entries if dict(f.axes) == target]
            assert len(refs) == 1 and refs[0].model.vol == rb.Constant("ig")


def test_grid_without_a_mean_axis_inserts_the_harmonised_default_mean():
    entries = grid(rb.BVAR, vol=[rb.Constant(), rb.SV(tails="t")], lags=[1, 2])
    assert [e.name for e in entries] == [
        "Constant(ig) × 1",
        "Constant(ig) × 2",
        "SV(t) × 1",
        "SV(t) × 2",
    ]
    assert {e.model.mean.prior.form for e in entries} == {"structural"}
    assert {e.model.engine for e in entries} == {"mcmc"}


def test_all_constant_grids_stay_analytic_and_unchanged():
    combos = [
        {"mean": AUTO_MEAN, "vol": rb.Constant()},
        {"mean": rb.Linear(rb.Minnesota(lambda_=0.1)), "vol": rb.Constant("iw")},
        {"lags": 2},
    ]
    assert rb.BVAR.harmonise_grid(combos) == combos
    entries = grid(rb.BVAR, mean=[AUTO_MEAN], vol=[rb.Constant()], lags=[1, 4])
    assert {e.model.engine for e in entries} == {"analytic"}
    assert entries[0].model == rb.BVAR()
    assert entries[0].name == "Linear[Minnesota(conjugate, eb)] × Constant × 1"


def test_explicit_settings_are_never_changed():
    reduced = rb.Linear(rb.Minnesota(form="reduced"))
    conjugate = rb.Linear(rb.Minnesota(form="conjugate"))
    combos = [
        {"mean": reduced, "vol": rb.Constant()},
        {"mean": AUTO_MEAN, "vol": rb.Constant("iw")},
        {"mean": AUTO_MEAN, "vol": rb.Constant("iw_equivalent")},
        {"mean": conjugate, "vol": rb.Constant()},
        {"mean": AUTO_MEAN, "vol": rb.SV()},
    ]
    out = rb.BVAR.harmonise_grid(combos)
    assert out[0]["mean"] is reduced and out[0]["vol"] == rb.Constant("ig")
    assert out[1]["mean"] == STRUCTURAL and out[1]["vol"] == rb.Constant("iw")  # → Mode II
    assert out[2]["vol"] == rb.Constant("iw_equivalent")
    assert out[3]["mean"] is conjugate and out[3]["vol"] == rb.Constant()  # explicit conjugate
    assert out[4]["mean"] == STRUCTURAL and out[4]["vol"] == rb.SV()
    mode2 = rb.BVAR(**out[1])
    assert mode2.axes["prior"].endswith("scaling=conjugate")  # R2 keeps it off the vol axis
    assert rb.BVAR(**out[3]).engine == "analytic"


def test_explicit_conjugate_arm_in_an_mcmc_grid_fails_at_construction():
    with pytest.raises(ValueError, match=r"requires Constant\(prior='iw'\)"):
        grid(rb.BVAR, mean=[rb.Linear(rb.Minnesota(form="conjugate"))], vol=[rb.SV()])


def test_harmonise_grid_does_not_mutate_its_input():
    combos = [{"mean": AUTO_MEAN, "vol": rb.Constant()}, {"mean": AUTO_MEAN, "vol": rb.SV()}]
    snapshot = [dict(c) for c in combos]
    out = rb.BVAR.harmonise_grid(combos)
    assert combos == snapshot
    assert out is not combos and all(o is not c for o, c in zip(out, combos, strict=True))


def test_harmonised_arms_are_listed_in_the_info_note(caplog):
    with caplog.at_level(logging.INFO, logger="rai_bvar.api"):
        rb.BVAR.harmonise_grid(
            [
                {"vol": rb.Constant()},
                {"mean": STRUCTURAL, "vol": rb.Constant("iw")},
                {"mean": AUTO_MEAN, "vol": rb.SV()},
                {"mean": rb.Linear(rb.Minnesota(form="reduced")), "vol": rb.Constant()},
            ]
        )
    [record] = caplog.records
    message = record.getMessage()
    assert "3 of 4 arms" in message
    assert (
        "arm 0: Minnesota(form='auto' → 'structural'), Constant('auto' → 'ig'); "
        "arm 2: Minnesota(form='auto' → 'structural'); "
        "arm 3: Constant('auto' → 'ig')"
    ) in message
    assert "arm 1" not in message  # explicit settings are never changed
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rai_bvar.api"):
        rb.BVAR.harmonise_grid([{"vol": rb.Constant()}, {"mean": AUTO_MEAN, "lags": 2}])
    assert caplog.text == ""  # an analytic grid: nothing resolved


def test_explicit_form_resolves_constant_auto_in_an_analytic_vol_grid(caplog):
    """No vol arm forces MCMC, but an explicit structural mean already does: its Constant() is
    Mode I (what BVAR resolves anyway), labelled apart from the Mode II Constant("iw") arm."""
    with caplog.at_level(logging.INFO, logger="rai_bvar.api"):
        entries = grid(rb.BVAR, mean=[STRUCTURAL], vol=[rb.Constant(), rb.Constant("iw")])
    assert "arm 0: Constant('auto' → 'ig')" in caplog.text
    assert [e.model for e in entries] == [
        rb.BVAR(mean=STRUCTURAL, vol=rb.Constant()),
        rb.BVAR(mean=STRUCTURAL, vol=rb.Constant("iw")),
    ]
    assert [e.name for e in entries] == [
        "Linear[Minnesota(structural, eb)] × Constant(ig)",
        "Linear[Minnesota(structural, eb)] × Constant",
    ]


@pytest.mark.parametrize(
    "axes",
    [
        {"mean": [STRUCTURAL], "vol": [rb.Constant("ig"), rb.Constant("iw")]},
        {"vol": [rb.Constant("ig"), rb.Constant("iw"), rb.SV()]},
        {"mean": [AUTO_MEAN], "vol": [rb.Constant(), rb.Constant("iw"), rb.SV()]},
        {"vol": [rb.Constant("ig"), rb.Constant("iw"), rb.Constant("iw_equivalent")]},
    ],
    ids=["explicit-structural", "no-mean", "auto-mean", "every-constant"],
)
def test_mode_one_and_mode_two_arms_are_named_apart(axes):
    """Before ``Constant.label``, both arms were named "... × Constant" and ``compare()`` rejected
    the grid ("model names must be unique")."""
    entries = grid(rb.BVAR, **axes)
    names = [e.name for e in entries]
    assert len(set(names)) == len(names), names
    assert _entries(entries) == entries  # what compare() runs first
    vols = [e.model.vol.prior for e in entries if isinstance(e.model.vol, rb.Constant)]
    assert {"ig", "iw"} <= set(vols)
    mode1, mode2 = (next(e for e in entries if e.model.vol == rb.Constant(p)) for p in ("ig", "iw"))
    assert mode1.model.engine == mode2.model.engine == "mcmc"
    assert mode2.axes["prior"].endswith("scaling=conjugate")  # R2: they differ on prior, not vol
    assert mode1.axes["vol"] == mode2.axes["vol"] == "Constant"


def test_grid_calls_the_factory_hook_and_labels_rewritten_components():
    @dataclass(frozen=True)
    class Toy:
        size: str = "auto"
        kind: str = "a"

        @property
        def axes(self) -> dict[str, str]:
            return {"model": "toy"}

        @classmethod
        def harmonise_grid(cls, combos):
            big = any(c["kind"] == "b" for c in combos)
            return [{**c, "size": "big" if big else c["size"]} for c in combos]

    entries = grid(Toy, size=["auto"], kind=["a", "b"])
    assert [e.model for e in entries] == [Toy("big", "a"), Toy("big", "b")]
    assert [e.name for e in entries] == ["big × a", "big × b"]
    assert [dict(e.axes) for e in entries] == [
        {"size": "big", "kind": "a"},
        {"size": "big", "kind": "b"},
    ]


def test_factories_without_the_hook_are_unchanged():
    entries = grid(rb.OLSVAR, lags=[1, 2])
    assert [e.model for e in entries] == [rb.OLSVAR(1), rb.OLSVAR(2)]
