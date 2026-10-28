"""Phase 4 week-1 spike, question 2 (SPEC §13): simulation-based calibration of the prototype
BART mean block through the production sweep (``validation._bart_spike.BartJoint``;
``docs/phase4/SPIKE.md``). Skipped without the optional ``bart`` extra.

``sbc_sequential`` with the sizes of ``test_sbc_compositions`` (DESIGN §9.6 and that module's
``t_max = 200`` deviation): ``SBCConfig(N=1000, L=99, burn=300, pilot=1000, t_max=200,
alpha=1e-2)``, the data-only filter ``max|ỹ| < 10³``, both ESS gates, the Gandy–Scott wrapper.

Pre-registered (before any SBC run) from a pilot of 16 prior-drawn ``T = 100`` datasets × 3000
sweeps: the largest per-quantity IAT was 43 (``φ``; tree split counts ≈ 20–38, median over
quantities ≈ 20), inside what ``t_max = 200`` covers (IAT ≤ 182). Quantities: every test function
of :class:`BartJoint` except data functions, squares and cross-products (``sbc_names``): ``a``,
the SV parameters and states, ``g`` at probe points and dates, ``g`` mean and mean square, the leaf
count, mean depth, split counts, leaf sum of squares, tree log-prior, the log-likelihoods and
``log p(θ)``. No predictive quantity: forecasting is outside the spike (``predict_eq`` raises).

=================  ======================================================  =========
Row                Composition                                             Tier
=================  ======================================================  =========
``bart_sv_T100``   BART × SV, n = 2, p = 1, T = 100; 20 trees, depth ≤ 5    nightly
=================  ======================================================  =========
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("bartz", reason="optional extra 'bart' (bartz) not installed")

from test_sbc_compositions import _iat_summary, _nanmin, record

from rai_bvar.components.mean._bart_spike import BartPrior
from rai_bvar.validation._bart_spike import bart_joint
from rai_bvar.validation.sbc import SBCConfig, sbc_sequential

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

SEED = 20261103
SBC_N = int(os.environ.get("RAI_BVAR_SBC_N", "1000"))
SIZES = {"L": 99, "burn": 300, "pilot": 1000, "t_max": 200, "alpha": 1e-2}
ROWS = {"bart_sv_T100": {"n": 2, "T_eff": 100}}
FOREST = {"num_trees": 20, "max_depth": 5}


def spike_joint(row: str):
    return bart_joint(f"bart_spike_sbc_{row}", bart=BartPrior(**FOREST), **ROWS[row])


def test_sbc_quantity_selection():
    """Parameters, latents, the BART summaries, log-likelihoods and ``log p(θ)``; no data
    functions, squares or cross-products."""
    names = spike_joint("bart_sv_T100").sbc_names
    assert not any(k.startswith("data.") or "^2" in k or "_sq" in k or "*" in k for k in names)
    for want in ("loglik", "log_prior", "eq1.joint.loglik", "eq1.struct.a0", "eq0.core.phi",
                 "eq0.mean.g_probe[0]", "eq1.mean.n_leaves", "eq0.mean.tree_log_prior"):  # fmt: skip
        assert want in names, (want, names)


@pytest.mark.slow
@pytest.mark.sbc
@pytest.mark.parametrize("row", list(ROWS))
def test_sbc_bart(row, record_property):
    """Full SBC of one spike row (module docstring)."""
    model = spike_joint(row)
    cfg = SBCConfig(N=SBC_N, seed=SEED, **SIZES)
    seq, results = sbc_sequential(model, model.sbc_quantities, model.sbc_names, cfg)
    last = results[-1]
    out = {
        "name": model.name, "row": row, "passed": seq.passed, "stage": seq.stage,
        "q": list(seq.q), "N": last.N, **SIZES, "forest": FOREST, "Q": len(model.sbc_names),
        "reasons": list(last.reasons), "failing": list(last.failing),
        "min_p": float(np.min(last.p)), "argmin": model.sbc_names[int(np.argmin(last.p))],
        "ess_ok": last.ess_ok, "thin_max": int(np.max(last.thin)) if last.N else 0,
        "ess_pooled_min": _nanmin(last.ess_pooled), **_iat_summary(last, cfg),
    }  # fmt: skip
    record(out)
    record_property("sbc", json.dumps(out, default=float))
    assert seq.passed, (out, last.table().head(10))
