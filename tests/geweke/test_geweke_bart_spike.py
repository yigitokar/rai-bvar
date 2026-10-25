"""Phase 4 week-1 spike, question 2 (SPEC §13): Geweke tests of the prototype BART mean block
through the production sweep (``validation._bart_spike.BartJoint``; ``docs/phase4/SPIKE.md``).

The sizes and the decision rule are those of ``test_geweke_compositions`` (DESIGN §9.5): G-small
``C = 1000, M_c = 100, M₁ = 2·10⁵, α = 1e-3``, G-full ``C = 4000, M_c = 500, M₁ = 10⁶,
α = 1e-2``, inside the Gandy–Scott wrapper. Skipped without the optional ``bart`` extra.

=========================  =========================================================  ===========
Row                        Composition                                                Tier
=========================  =========================================================  ===========
``block_T200``             BART only (``live = eq.mean``), n = 1, T = 200; SV held    nightly
                           at its prior draw (it runs after the mean, so restoring
                           it is an exact kernel for the BART block)
``bart_sv_T50``            BART × SV, n = 2, p = 1, T = 50                            PR
``bart_sv_T200``           BART × SV, n = 2, p = 1, T = 200                           nightly
``bart_sv_n3p2_T200``      BART × SV, n = 3, p = 2, T = 200 (6 covariates)            nightly
``bart_sv_T200_x64``       ``bart_sv_T200`` on path (b): bartz traced under x64        nightly
                           with float64 residuals (``bartz_x64_patch``)
``bart_sv_wide_T400``      BART × SV, n = 2, T = 400, a **wider** SV prior            nightly
                           (``μ ~ N(0, 2²)``, ``(φ+1)/2 ~ Beta(20, 1.5)``,
                           ``B_σ = 0.5``): within-sample max/min variance ratio median 2.4·10², IQR
                           13–1.7·10⁴, 90th percentile 3·10⁶ (default prior at T = 200: median 5,
                           90th percentile 66)
=========================  =========================================================  ===========

Forests: 20 trees of depth ≤ 5 on a 31-cutpoint grid (the exactness of the swap and of the
float32 leaf step does not depend on the number of trees; the cost table in the memo uses 200).
Every row gates; ``RAI_BVAR_VALIDATION_RECORD=<dir>`` appends the evidence to ``geweke.jsonl``.

``bart_sv_T50`` is PR-tier by its sizes (G-small) but takes about 5 minutes, against 3–55 s for the
existing PR rows; it runs on PRs only where the ``bart`` extra is installed (CI installs the dev
group only, so today it never runs there).

**Mutants** (``test_g_small_bart_mutant_fails``, DESIGN §9.7 convention): the harness must catch
the failure the adapter exists to prevent. ``stale_prec_tree``: the naive swap that writes the
residuals and the error-scale fields but leaves bartz's per-node precision cache as the previous
sweep left it. ``sqrt_weights``: the weights passed as ``√w`` instead of ``w = 1/λ``. Both must
fail G-small at the ``bart_sv_T50`` sizes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("bartz", reason="optional extra 'bart' (bartz) not installed")

from test_geweke_compositions import G_FULL, G_SMALL, explain, record, summary

from rai_bvar.components.mean._bart_spike import (
    BartPrior,
    BartSpikeMean,
    bartz_step,
    set_data,
)
from rai_bvar.engines.mcmc.state import Coupling, StepInfo
from rai_bvar.validation._bart_spike import BartJoint, bart_joint, bartz_x64_patch
from rai_bvar.validation.geweke import GewekeConfig, geweke_sequential
from rai_bvar.validation.joint import joint_spec

pytestmark = [
    pytest.mark.geweke,
    pytest.mark.filterwarnings("error:Explicitly requested dtype"),
]

SEED_SMALL = 20261101
SEED_FULL = 20261102
FOREST = {"num_trees": 20, "max_depth": 5}
WIDE_SV = {"B_mu": 4.0, "a_phi": 20.0, "b_phi": 1.5, "B_sigma": 0.5}

ROWS: dict[str, dict[str, Any]] = {
    "block_T200": {"n": 1, "T_eff": 200, "live": ("eq.mean",)},
    "bart_sv_T50": {"n": 2, "T_eff": 50},
    "bart_sv_T200": {"n": 2, "T_eff": 200},
    "bart_sv_n3p2_T200": {"n": 3, "lags": 2, "T_eff": 200},
    "bart_sv_T200_x64": {"n": 2, "T_eff": 200, "x64": True},
    "bart_sv_wide_T400": {"n": 2, "T_eff": 400, "prior_kwargs": WIDE_SV},
}
PR_ROWS = ("bart_sv_T50",)


def spike_joint(row: str):
    kw = dict(ROWS[row])
    x64 = kw.pop("x64", False)
    return bart_joint(f"bart_spike_{row}", bart=BartPrior(**FOREST, x64=x64), **kw), x64


def run_geweke(row: str, sizes: dict[str, Any], seed: int):
    with bartz_x64_patch() if ROWS[row].get("x64") else contextlib.nullcontext():
        spec, _ = spike_joint(row)
        t0 = time.perf_counter()
        seq, results = geweke_sequential(spec, GewekeConfig(seed=seed, **sizes))
    out = summary(spec.name, seq, results)
    out.update(row=row, seed=seed, seconds=time.perf_counter() - t0, forest=FOREST, **{
        k: v for k, v in ROWS[row].items() if k != "live"
    })  # fmt: skip
    return spec, seq, results, out


@pytest.mark.parametrize(
    "row", [r if r in PR_ROWS else pytest.param(r, marks=pytest.mark.slow) for r in ROWS]
)
def test_g_small_bart(row):
    """G-small of one spike row through the production sweep (module table)."""
    spec, seq, results, out = run_geweke(row, G_SMALL, SEED_SMALL)
    record("geweke", {"tier": "G-small-bart-spike", **out})
    assert spec.plan.names == ("mean", "contemp", "residual", "vol")
    assert len(spec.names) >= 25
    assert seq.passed, explain(seq, results)


@pytest.mark.slow
@pytest.mark.parametrize("row", ["bart_sv_T200"])
def test_g_full_bart(row):
    """G-full of the main spike row (``C = 4000, M_c = 500, M₁ = 10⁶, α = 1e-2``)."""
    _, seq, results, out = run_geweke(row, G_FULL, SEED_FULL)
    record("geweke", {"tier": "G-full-bart-spike", **out})
    assert seq.passed, explain(seq, results)


def test_rows_are_distinct_streams():
    """Each row has its own name, hence its own Geweke key stream (CRC-32 of the name)."""
    names = {spike_joint(r)[0].name for r in ROWS if not ROWS[r].get("x64")}
    assert len(names) == len(ROWS) - 1
    assert np.all([n.startswith("bart_spike_") for n in names])


# --------------------------------------------------------------------------- mutants


@dataclasses.dataclass(frozen=True)
class _StalePrecTree(BartSpikeMean):
    """Mutant: the naive swap; ``forest.prec_tree`` keeps the previous sweep's values."""

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        st = set_data(ms, cache_i.Xbin, ctx.target, ctx.weights)
        st = dataclasses.replace(
            st, forest=dataclasses.replace(st.forest, prec_tree=ms.forest.prec_tree)
        )
        st = bartz_step(key, st, self.prior.x64)
        return st, a_i, Coupling.zeros(), StepInfo.success()


@dataclasses.dataclass(frozen=True)
class _SqrtWeights(BartSpikeMean):
    """Mutant: the per-observation precisions passed as ``√w``."""

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        ctx = dataclasses.replace(ctx, weights=jnp.sqrt(ctx.weights))
        return super().sample_eq(key, ms, a_i, ctx, pm, cm, cache_i, cache, shared)


MUTANTS = {"stale_prec_tree": _StalePrecTree, "sqrt_weights": _SqrtWeights}


@dataclasses.dataclass(frozen=True, eq=False)
class _MutantJoint(BartJoint):
    mutant: type = BartSpikeMean

    @cached_property
    def kernel(self) -> BartSpikeMean:
        return self.mutant.for_static(self.static, self.bart)


@pytest.mark.slow
@pytest.mark.parametrize("mutant", list(MUTANTS))
def test_g_small_bart_mutant_fails(mutant):
    """Each mutant fails G-small at the ``bart_sv_T50`` sizes (module docstring)."""
    base = joint_spec("zero_sv", "small", n=2, T_eff=50)
    spec = _MutantJoint(
        name=f"bart_spike_mutant_{mutant}", base=base, bart=BartPrior(**FOREST),
        mutant=MUTANTS[mutant],
    )  # fmt: skip
    t0 = time.perf_counter()
    seq, results = geweke_sequential(spec, GewekeConfig(seed=SEED_SMALL, **G_SMALL))
    out = summary(spec.name, seq, results)
    out.update(mutant=mutant, seconds=time.perf_counter() - t0)
    record("geweke", {"tier": "G-small-bart-spike-mutant", **out})
    assert not seq.passed, explain(seq, results)
