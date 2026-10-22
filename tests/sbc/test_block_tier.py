"""Tier B at the composition level: block-conditional SBC and one-block invariance through the
production sweep (DESIGN §9.4, §13 WP19; N06 §5.2–5.3).

**Block-conditional SBC** (exact-draw blocks; ``block_sbc_sequential``, ``N = 200``, ``L = 99``,
Bonferroni at ``α = 1e-3``, Gandy–Scott ``k = 3``, ``Δ = 4``, fresh keys per stage): ``(θ̃, ỹ)``
from the joint, ``L`` iid draws of the block from one production sweep with every other block
restored (``JointSpec.live``), ranks of the block's coordinates and of the exact log-likelihood.
Exact because the live block runs before every frozen one and draws from its exact conditional:

* the fused structural equation step ``(δ_i, a_i) | λ_eff, y`` in every structural composition
  (rows 1–6, 12, 14–16: the weights ``m_t e^{−log λ_eff}`` of every vol block, Mode II, the
  λ-scaled ``iw_equivalent`` ``a`` prior and the fast paths ``eigen``/``fixed_factor``/
  ``cholesky_const``);
* the reduced-form contemporaneous step ``a_i | Π, λ_eff, y`` in every reduced composition
  (rows 7–11, 13), with ``Π`` held by the identity ``cccm_core`` (the ``contemp_only`` mechanism);
* each column step ``π_j | π_{−j}, A, λ_eff, y`` of the program's ``cccm_core`` in every reduced
  composition (column ``j`` of one call with fresh noise, the other columns restored).

**One-block invariance** (MH, ASIS and Gibbs-scan blocks, whose single application is not an iid
draw; N06 §5.3): ``N = 1000`` joint draws, ``L = 20`` applications of the live blocks with ``ỹ``
fixed, against ``N`` independent joint draws; two-sample KS per block statistic, Bonferroni at
``α = 1e-3`` inside the Gandy–Scott wrapper (``N × 4`` after stage 1):

* the CCCM ``Π`` scan (block Gauss–Seidel over ``π_j``) in every reduced composition;
* the volatility block (indicators, exact-mode ``h`` MH, centred ``(γ, φ)`` and ``σ²`` MH, ASIS
  NC move and fold; transitory ``o``, ``p``, ``ν``, ``τ``) in zero-mean compositions, where it is
  the only block: ``zero_sv`` and its RW, ``InvGamma``, fixed-``B₀``, outlier, ``t`` and
  outlier-``t`` variants, and the Constant core with outlier-``t``.

All of this tier is nightly (``slow``): the per-PR Tier B of every block runs in the kernel work
packages' suites (WP10–WP13), and the PR budget of this work package is spent on G-small.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from sbc_models import (
    block_conditional_draws,
    block_joint_sample,
    cccm_column_draws,
    invariance_pvalue,
)

from rai_bvar.validation.joint import COMPOSITIONS, composition_id, identity_cccm_core, joint_spec
from rai_bvar.validation.log_joint import loglik
from rai_bvar.validation.sbc import block_sbc_sequential
from rai_bvar.validation.sequential import gandy_scott, stage_multiplier, stage_seed

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("error:Explicitly requested dtype")]

SEED = 20261027
ALPHA = 1e-3
STRUCTURAL = tuple(n for n in COMPOSITIONS if n.startswith("structural"))
REDUCED = tuple(n for n in COMPOSITIONS if n.startswith("reduced"))
EQ_LIVE = frozenset({"eq.mean", "eq.struct"})
VOL_CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "sv": ("zero_sv", {}),
    "sv_rw": ("zero_sv", {"rw": True}),
    "sv_invgamma": ("zero_sv", {"sigma2_prior": "inv_gamma"}),
    "sv_fixed_b0": ("zero_sv", {"h0": "fixed_b0"}),
    "sv_outliers": ("zero_sv", {"transitory": "outlier"}),
    "sv_t": ("zero_sv", {"transitory": "t"}),
    "sv_outliers_t": ("zero_sv", {"transitory": "outlier_t"}),
    "constant_outliers_t": ("constant_transitory", {}),
}


def _check_sbc(seq, results) -> None:
    last = results[-1]
    assert seq.passed, (seq.q, last.reasons, last.table().head(8))


# --------------------------------------------------------------------------- block SBC


@pytest.mark.parametrize("name", STRUCTURAL)
def test_block_sbc_structural_equation_step(name):
    """``(δ_i, a_i) | λ_eff, y`` (exact; DESIGN §4.2) inside composition ``name``."""
    spec = joint_spec(name, live=EQ_LIVE)
    n, k = spec.static.shapes.n, spec.static.shapes.k

    def stat(st, data):
        vals = [st.eq.mean.delta[i, r] for i in range(n) for r in range(k)]
        vals += [st.eq.struct.a[i, j] for i in range(n) for j in range(i)]
        return jnp.stack([*vals, loglik(st, data, spec.prior, spec.static)])

    seq, results = block_sbc_sequential(
        block_joint_sample(spec), block_conditional_draws(spec), stat, N=200, L=99, seed=SEED,
        alpha=ALPHA, composition=composition_id(name),
    )  # fmt: skip
    _check_sbc(seq, results)


@pytest.mark.parametrize("name", REDUCED)
def test_block_sbc_reduced_contemporaneous_step(name):
    """``a_i | Π, λ_eff, y`` (exact; N04 §6.2) inside composition ``name``, ``Π`` held."""
    spec = joint_spec(
        name, live=frozenset({"eq.struct"}), kernel_overrides={"cccm_core": identity_cccm_core}
    )
    n = spec.static.shapes.n

    def stat(st, data):
        vals = [st.eq.struct.a[i, j] for i in range(n) for j in range(i)]
        return jnp.stack([*vals, loglik(st, data, spec.prior, spec.static)])

    seq, results = block_sbc_sequential(
        block_joint_sample(spec), block_conditional_draws(spec), stat, N=200, L=99, seed=SEED,
        alpha=ALPHA, composition=composition_id(name),
    )  # fmt: skip
    _check_sbc(seq, results)


@pytest.mark.parametrize("name", REDUCED)
def test_block_sbc_cccm_columns(name):
    """Each column step ``π_j | π_{−j}, A, λ_eff, y`` of the program's ``cccm_core`` (exact; the
    scan's building block, N04 §5.3) inside composition ``name``: ranks of ``π_j`` and of the
    exact log-likelihood (the statistic that sees the CCM-2019 error, N06 §7.3)."""
    spec = joint_spec(name)
    for j in range(spec.static.shapes.n):

        def stat(st, data, j=j):
            ll = loglik(st, data, spec.prior, spec.static)
            return jnp.concatenate([st.glob.mean.Pi[:, j], ll[None]])

        seq, results = block_sbc_sequential(
            block_joint_sample(spec), cccm_column_draws(spec, j), stat, N=200, L=99,
            seed=SEED + j, alpha=ALPHA, composition=composition_id(name),
        )  # fmt: skip
        _check_sbc(seq, results)


# --------------------------------------------------------------------------- one-block invariance


def _invariance(spec, stat, composition: int, *, N: int = 1000, L: int = 20):
    stages: list[tuple[float, Any]] = []

    def run(stage: int) -> float:
        p, ps, _, _ = invariance_pvalue(
            spec, stat, N * stage_multiplier(stage), L, stage_seed(SEED, stage),
            composition=composition,
        )  # fmt: skip
        stages.append((p, ps))
        return p

    return gandy_scott(run, ALPHA), stages


@pytest.mark.parametrize("name", REDUCED)
def test_invariance_cccm_scan(name):
    """The CCCM ``Π`` scan (``L = 20`` applications, ``A`` and ``λ`` held) preserves the joint."""
    spec = joint_spec(name, live=frozenset({"glob.mean"}))
    k, n = spec.static.shapes.k, spec.static.shapes.n

    def stat(st, data):
        Pi = st.glob.mean.Pi
        vals = [Pi[r, j] for j in range(n) for r in range(k)]
        return jnp.stack([*vals, loglik(st, data, spec.prior, spec.static)])

    seq, stages = _invariance(spec, stat, composition_id(name))
    assert seq.passed, (seq.q, stages[-1][1])


@pytest.mark.parametrize("case", list(VOL_CASES))
def test_invariance_vol_block(case):
    """The atomic ``VolBlock.sample_eq`` (MH/ASIS SV core and transitory steps) preserves the
    joint (``L = 20``); statistics: every test function of the block and the exact
    log-likelihood."""
    name, kw = VOL_CASES[case]
    spec = joint_spec(name, **kw)
    names = tuple(g for g in spec.all_names if (".core." in g or ".trans." in g or g == "loglik"))
    idx = jnp.asarray([spec.all_names.index(g) for g in names])

    def stat(st, data):
        return spec.quantities(st, data)[idx]

    seq, stages = _invariance(spec, stat, composition_id(name))
    assert seq.passed, (seq.q, dict(zip(names, stages[-1][1], strict=True)))
