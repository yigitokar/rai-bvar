"""Phase 3 references (docs/phase3/DESIGN.md §9.1, §13 WP30; written by an author who wrote no
Phase 3 kernel, from the design document only).

Acceptance of WP30:

1. **prior moments against closed forms**: every global-local family (HS, NG with a tied and a
   fixed ``c``, R2D2, DL, SSVS with fixed and Beta ``p``), the discrete ``a`` grid, the CommonSV
   ``ρ``/``σ_h²``/``h`` priors (both options) and the conjugate (NIW) prior of the CommonSV rows,
   checked against the **prior definitions** of P3 §3.2 / §5.1 (e.g. the R2D2 ``R² ~ Beta(n_g a,
   b)``, the DL Laplace and gamma–Dirichlet identities, the IW diagonal marginals), including
   the dependence of every latent on the stored parameters it is drawn from;
2. **simulators against plain NumPy loops** (1e-12): the exogenous design (D-56; structural and
   reduced, no lag feedback) and the common-``h`` path (lagged and exogenous, Mode I, reduced and
   conjugate);
3. **log_joint against SciPy densities** (1e-10), with group ids, ``r_j`` and the SSVS ``τ`` built
   independently here from D-37, D-38 and §9.4.

Plus the separate registry (D-61): rows 301–330 and blocks 401–408 with the §9.4 test priors, the
Phase 2 registries unchanged, the P3 test functions and their finite-variance flags, the
:class:`JointSpecP3` kernels (against a test double of the production engine, which WP31/WP32
provide), and the explosive prior mass of the lagged rows (recorded in
:data:`EXPLOSIVE_MASS_P3`, each ≤ 2e-4).

Monte Carlo checks use fixed seeds (deterministic): means within 5 standard errors, KS tests at
``α = 1e-3`` on independent draws.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import special, stats

from rai_bvar.engines.mcmc.keys import Block, step_key
from rai_bvar.engines.mcmc.state import (
    StepInfo,
    check_state_invariants,
    leaf_path,
)
from rai_bvar.validation import joint as J
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation import reference as R
from rai_bvar.validation import simulate as S
from rai_bvar.validation import testfns as TF

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

N = 10_000
K_SE = 5.0
ALPHA = 1e-3

EXPLOSIVE_DRAWS = J.EXPLOSIVE_DRAWS
EXPLOSIVE_MASS_P3 = J.EXPLOSIVE_MASS_P3


# --------------------------------------------------------------------------- helpers


def _np(tree):
    return jax.tree.map(np.asarray, tree)


@functools.cache
def _sampler(static, T_eff: int):
    """One compile per StaticConfig: the prior is an argument, so rows that share a static
    (e.g. 304 and 326) share it."""
    return jax.jit(jax.vmap(lambda k, pr: S.prior_sample(k, pr, static, T_eff), in_axes=(0, None)))


_DRAWN: set = set()


@functools.cache
def draws(name: str, size: str = "small", n_draws: int = N, seed: int = 0):
    """``(spec, st)``: ``n_draws`` prior draws of a row (NumPy leaves, leading draw axis)."""
    _DRAWN.add((name, size, n_draws, seed))
    spec = J.joint_spec(name, size)
    keys = jax.vmap(lambda i: random.fold_in(random.key(seed), i))(
        jnp.arange(n_draws, dtype=jnp.uint32)
    )
    return spec, _np(_sampler(spec.static, spec.fixed.T_eff)(keys, spec.prior))


def draws_with_prior(name: str, prior, n_draws: int = N, seed: int = 0):
    spec = J.joint_spec(name)
    keys = jax.vmap(lambda i: random.fold_in(random.key(seed), i))(
        jnp.arange(n_draws, dtype=jnp.uint32)
    )
    return _np(_sampler(spec.static, spec.fixed.T_eff)(keys, prior))


def one(tree, c: int):
    return jax.tree.map(lambda x: x[c], tree)


@functools.cache
def _simulator(name: str, size: str):
    return jax.jit(J.joint_spec(name, size).simulate)


def a_dataset(name: str, size: str = "small", c: int = 0):
    """``(spec, st, data)`` with ``data ~ p(y | st)`` (jitted, cached per row)."""
    spec, st = a_draw(name, size, c)
    return spec, st, _simulator(name, size)(random.key(1000 + c), st)


def a_draw(name: str, size: str = "small", c: int = 0):
    """``(spec, st)``: draw ``c`` of the row's cached draws (no new compile when the row's sampler
    is cached); rows outside the moment tests get a small batch."""
    n_draws = N if (name, size, N, 0) in _DRAWN else 4
    spec, st = draws(name, size, n_draws)
    return spec, jax.tree.map(jnp.asarray, one(st, c))


def assert_mean(x, mean, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean() - mean) < K_SE * se + 1e-12, f"{label}: mean {x.mean()} vs {mean} (se {se})"


def assert_var(x, var, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    s2 = x.var(ddof=1)
    se = math.sqrt(max(np.mean((x - x.mean()) ** 4) - s2**2, 0.0) / x.size)
    assert abs(s2 - var) < K_SE * se, f"{label}: var {s2} vs {var} (se {se})"


def assert_ks(x, cdf, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    p = stats.kstest(x, cdf).pvalue
    assert p > ALPHA, f"{label}: KS p = {p} (n = {x.size})"


def assert_uncorrelated(x, others: dict, label=""):
    x = np.asarray(x, dtype=np.float64).ravel()
    for name, y in others.items():
        r = np.corrcoef(x, np.asarray(y, dtype=np.float64).ravel())[0, 1]
        assert abs(r) < K_SE / math.sqrt(x.size), f"{label}: corr with {name} = {r}"


def gl_slots(spec):
    """GL coordinates (i, r): the lag slots (D-36), built here, not from the prior arrays."""
    s = spec.static.shapes
    return [(i, r) for i in range(s.n) for r in range(s.n_det, s.k)]


def group_of(spec, groups: str, i: int, r: int) -> int:
    """D-38, written again: own/cross lags of lag l (olcl-lagwise), global, equation-wise."""
    s = spec.static.shapes
    lag, m = 1 + (r - s.n_det) // s.n, (r - s.n_det) % s.n
    return {
        "olcl-lagwise": 2 * (lag - 1) + (0 if m == i else 1),
        "global": 0,
        "equation-wise": i,
    }[groups]


def log_r_of(spec, i: int, r: int) -> float:
    """D-37 with the §9.4 ψ: log ψ_i − log ι − log ψ_m (structural) or without log ι."""
    s = spec.static.shapes
    psi = {2: (0.5, 2.0), 3: (0.5, 1.0, 2.0)}[s.n]
    m = (r - s.n_det) % s.n
    out = math.log(psi[i]) - math.log(psi[m])
    if spec.static.mean.kind == "linear_structural":
        out -= math.log(i + 1.0)
    return out


def deviations(spec, st):
    """``(d, dev_log)`` with row i the equation: ``d = coefficient − m`` (both forms)."""
    m = np.asarray(spec.prior.eq.coef.mean0)
    if spec.static.mean.kind == "linear_structural":
        return st.eq.mean.delta - m, st.eq.mean.dev_log
    return np.swapaxes(st.glob.mean.Pi, -1, -2) - m, st.glob.mean.dev_log


def standardized_deviations(spec, st, log_v):
    """``d_j/√(r_j v_j)`` on every GL slot from ``dev_log`` and the sign of ``coefficient − m``,
    with ``log v`` (draws × n × k) supplied by the caller from the stored scales."""
    d, dev_log = deviations(spec, st)
    out = []
    for i, r in gl_slots(spec):
        lr = log_r_of(spec, i, r)
        out.append(np.sign(d[:, i, r]) * np.exp(dev_log[:, i, r] - 0.5 * (lr + log_v[:, i, r])))
    return np.stack(out, axis=1)


def assert_invariants(spec, st, count: int = 4):
    for c in range(count):
        bad = check_state_invariants(one(st, c), spec.static, spec.fixed.T_eff)
        assert bad == [], (spec.name, bad)


def assert_dev_log_exact(spec, st):
    """``dev_log = log|coefficient − m|`` wherever ``|d|`` is resolved by ``m + d`` (and finite)."""
    d, dev_log = deviations(spec, st)
    m = np.abs(np.asarray(spec.prior.eq.coef.mean0))
    ok = np.abs(d) > 1e-6 * np.maximum(1.0, m)
    assert ok.mean() > 0.5
    np.testing.assert_allclose(dev_log[ok], np.log(np.abs(d[ok])), rtol=1e-9, atol=1e-9)
    assert np.all(np.isfinite(dev_log))


def assert_deterministic_slots(spec, st):
    """Deterministic slots keep N(m, 1/prec0) (D-36): the intercept test prior N(0, 1)."""
    d, _ = deviations(spec, st)
    for i in range(spec.static.shapes.n):
        assert_mean(d[:, i, 0], 0.0, f"intercept {i}")
        assert_var(d[:, i, 0], 1.0, f"intercept {i}")


def grid_a(spec, a_idx):
    return np.asarray(J.TEST_A_GRID)[a_idx]


# --------------------------------------------------------------------------- 1. registry


def test_phase2_registries_are_unchanged():
    assert len(J.COMPOSITIONS) == 23 and len(J.BLOCK_CONFIGS) == 7  # D-61
    assert [J.composition_id(c) for c in J.COMPOSITIONS] == [*range(1, 19), *range(201, 206)]
    p2 = set(J.COMPOSITIONS) | set(J.BLOCK_CONFIGS)
    p3 = set(J.COMPOSITIONS_P3) | set(J.BLOCK_CONFIGS_P3)
    assert not p2 & p3
    for name in ("structural_sv", "aux_sv", "structural_constant_dummies"):
        spec = J.joint_spec(name)
        assert type(spec) is J.JointSpec and spec.static.coef.kind == "fixed_diagonal"
        assert not spec.fixed.exogenous and spec.fixed.X is None
        assert not spec.static.vol.common and not spec.static.mean.coef_log


def test_p3_registry_ids_and_dispatch():
    assert [J.composition_id(c) for c in J.COMPOSITIONS_P3] == list(range(301, 331))
    assert [J.composition_id(c) for c in J.BLOCK_CONFIGS_P3] == list(range(401, 409))
    for name in J.COMPOSITIONS_P3 + J.BLOCK_CONFIGS_P3:
        spec = J.joint_spec(name)  # joint_spec accepts the P3 names (D-61)
        assert isinstance(spec, J.JointSpecP3) and spec.name == name
    with pytest.raises(KeyError, match="unknown joint spec"):
        J.joint_spec("structural_bart")
    with pytest.raises(KeyError, match="unknown Phase 3 joint spec"):
        J.joint_spec_p3("structural_sv")
    with pytest.raises(TypeError, match="unknown joint_spec overrides"):
        J.joint_spec("structural_constant_hs", groups="global")
    assert J.LAGGED_P3 == (
        "reduced_sv_ng_lagged", "structural_sv_ng_lagged", "conjugate_commonsv_dummies",
        "structural_commonsv", "reduced_commonsv", "conjugate_commonsv_dummies_weighted",
        "conjugate_commonsv_parity", "conjugate_commonsv_block7", "structural_sv_ssvs_lagged",
        "common_sv_only", "conjugate_niw_only",
    )  # fmt: skip


# (mean, persistent, transitory, coef, groups G at p = 1, a_grid, p_beta, common, const_path,
#  design, n small/full) per row of P3 §9.4
_S, _R = "linear_structural", "linear_reduced"
ROWS = {
    301: (_S, "constant", "none", "horseshoe", 2, 0, False, False, "weighted", "exo", (2, 3)),
    302: (_S, "sv", "none", "horseshoe", 2, 0, False, False, "weighted", "exo", (2, 3)),
    303: (_S, "constant", "none", "normal_gamma", 2, 0, False, False, "weighted", "exo", (3, 3)),
    304: (_S, "constant", "none", "normal_gamma", 2, 5, False, False, "weighted", "exo", (3, 3)),
    305: (_S, "constant", "none", "r2d2", 2, 0, False, False, "weighted", "exo", (3, 3)),
    306: (_S, "constant", "none", "r2d2", 2, 5, False, False, "weighted", "exo", (3, 3)),
    307: (_S, "constant", "none", "dirichlet_laplace", 2, 0, False, False, "weighted", "exo", (3, 3)),
    308: (_S, "constant", "none", "dirichlet_laplace", 2, 5, False, False, "weighted", "exo", (3, 3)),
    309: (_S, "constant", "none", "ssvs", 2, 0, False, False, "weighted", "exo", (3, 3)),
    310: (_S, "constant", "none", "ssvs", 2, 0, True, False, "weighted", "exo", (3, 3)),
    311: (_R, "sv", "none", "horseshoe", 1, 0, False, False, "weighted", "exo", (3, 3)),
    312: (_R, "sv", "none", "normal_gamma", 1, 0, False, False, "weighted", "exo", (3, 3)),
    313: (_R, "sv", "none", "r2d2", 1, 0, False, False, "weighted", "exo", (3, 3)),
    314: (_R, "sv", "none", "dirichlet_laplace", 1, 0, False, False, "weighted", "exo", (3, 3)),
    315: (_R, "sv", "none", "ssvs", 1, 0, False, False, "weighted", "exo", (3, 3)),
    316: (_R, "sv", "none", "normal_gamma", 2, 0, False, False, "weighted", "lag", (3, 3)),
    317: (_S, "sv", "outlier", "horseshoe", 3, 0, False, False, "weighted", "exo", (3, 3)),
    318: (_S, "constant", "none", "horseshoe", 2, 0, False, False, "weighted", "exo", (3, 3)),
    319: (_S, "sv", "none", "normal_gamma", 2, 0, False, False, "weighted", "lag", (3, 3)),
    320: (_S, "constant", "none", "fixed_diagonal", 0, 0, False, True, "shared_factor", "lag", (3, 3)),
    321: (_S, "constant", "none", "fixed_diagonal", 0, 0, False, True, "weighted", "lag", (2, 3)),
    322: (_R, "constant", "none", "fixed_diagonal", 0, 0, False, True, "weighted", "lag", (3, 3)),
    323: (_S, "constant", "none", "fixed_diagonal", 0, 0, False, True, "weighted", "lag", (3, 3)),
    324: (_S, "constant", "none", "horseshoe", 2, 0, False, True, "weighted", "exo", (3, 3)),
    325: (_S, "constant", "none", "fixed_diagonal", 0, 0, False, True, "shared_factor", "lag", (3, 3)),
    326: (_S, "constant", "none", "normal_gamma", 2, 5, False, False, "weighted", "exo", (3, 3)),
    327: (_R, "constant", "none", "horseshoe", 1, 0, False, False, "weighted", "exo", (3, 3)),
    328: (_R, "constant", "none", "normal_gamma", 2, 0, False, True, "weighted", "exo", (3, 3)),
    329: (_S, "constant", "none", "fixed_diagonal", 0, 0, False, True, "shared_factor", "lag", (3, 3)),
    330: (_S, "sv", "none", "ssvs", 2, 0, False, False, "weighted", "lag", (3, 3)),
}  # fmt: skip


@pytest.mark.parametrize("name", J.COMPOSITIONS_P3)
def test_rows_match_the_design_table(name):
    row = J.composition_id(name)
    mean, core, trans, coef, G1, a_grid, p_beta, common, path, design, ns = ROWS[row]
    for size, n in zip(("small", "full"), ns, strict=True):
        spec = J.joint_spec(name, size)
        st = spec.static
        p, T = (1, 20) if size == "small" else (2, 50)
        assert (st.shapes.n, st.shapes.lags, spec.fixed.T_eff) == (n, p, T)
        assert spec.fixed.T_eff % 32 != 0 and st.shapes.n_det == 1
        assert (st.mean.kind, st.vol.persistent, st.vol.transitory, st.coef.kind) == (
            mean, core, trans, coef,
        )  # fmt: skip
        varies = coef != "fixed_diagonal"
        G = {1: 1, 2: 2 * p, 3: n}[G1] if varies else 0  # global, olcl-lagwise (2p), eq-wise
        assert (st.coef.n_groups, st.coef.a_grid, st.coef.p_beta) == (G, a_grid, p_beta)
        assert st.mean.coef_log == varies and st.vol.common == common
        assert st.mean.const_path == path
        assert spec.fixed.exogenous == (design == "exo")
        mode2 = row in (320, 323, 325, 329)
        assert (st.mean.coef_scaling == "lambda") == mode2
        assert (st.struct.contemp_scaling == "lambda") == (mode2 or row == 318)
        assert st.vol.coupled == (mode2 or row == 318)
        assert st.mean.n_dummy == (n + 1 if row in (320, 323) else 0)
        assert st.vol.csv_block == (7 if row == 329 else 100)
        assert st.vol.csv_rho_prior == ("beta" if row == 325 else "truncated_normal")
        assert st.vol.csv_sigma2_prior == ("scaled_chi2" if row == 325 else "inv_gamma")
        assert spec.kernel == "sweep" and spec.refresh == "none" and not spec.data_irrelevant
        assert spec.live == J.ALL


def test_block_configurations_401_408():
    kinds = ("horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs")
    for number, kind in zip(range(401, 406), kinds, strict=True):
        spec = J.joint_spec(f"gl_scales_only_{kind}")
        assert J.composition_id(spec.name) == number and spec.static.coef.kind == kind
        assert spec.kernel == "global_step" and spec.global_block == Block.G_COEF_GLOBAL
        assert spec.refresh == "gl_coefficients" and spec.data_irrelevant
        assert spec.live == frozenset({"eq.coef", "glob.coef"})
        assert not any(TF_ for TF_ in spec.names if J._data_dependent(TF_))
    assert J.joint_spec("gl_scales_only_normal_gamma").static.coef.a_grid == 5
    assert J.joint_spec("gl_scales_only_ssvs").static.coef.p_beta
    mean_only = J.joint_spec("gl_mean_only")
    assert mean_only.live == frozenset({"eq.mean", "eq.struct"}) and mean_only.kernel == "sweep"
    assert mean_only.static.coef.kind == "horseshoe" and mean_only.fixed.exogenous
    csv = J.joint_spec("common_sv_only")
    assert csv.live == frozenset({"glob.vol"}) and csv.global_block == Block.G_COMMON_SV
    assert csv.static.vol.common and not csv.fixed.exogenous and csv.kernel == "global_step"
    niw = J.joint_spec("conjugate_niw_only")
    assert niw.live == frozenset({"eq.mean", "eq.struct", "eq.vol"}) and niw.kernel == "sweep"
    assert niw.static.mean.const_path == "shared_factor" and niw.static.mean.n_dummy == 4


def test_test_priors_of_section_9_4():
    """The arrays of the §9.4 test priors, rebuilt here from D-37, D-38 and §9.4."""
    for name, size in (("structural_constant_hs", "small"), ("reduced_sv_ssvs_global", "full"),
                       ("structural_sv_ssvs_lagged", "small"), ("structural_constant_ng_grid", "small"),
                       ("structural_sv_outliers_hs_eqwise", "small")):  # fmt: skip
        spec = J.joint_spec(name, size)
        s, gl = spec.static.shapes, spec.prior.eq.coef.gl
        groups = {"reduced_sv_ssvs_global": "global"}.get(name, "olcl-lagwise")
        if name == "structural_sv_outliers_hs_eqwise":
            groups = "equation-wise"
        mask = np.zeros((s.n, s.k))
        mask[:, s.n_det :] = 1.0
        np.testing.assert_array_equal(np.asarray(gl.mask), mask)
        for i in range(s.n):
            for r in range(s.k):
                if r < s.n_det:
                    assert int(gl.group[i, r]) == 0 and float(gl.log_r[i, r]) == 0.0
                    continue
                assert int(gl.group[i, r]) == group_of(spec, groups, i, r)
                assert math.isclose(float(gl.log_r[i, r]), log_r_of(spec, i, r), abs_tol=1e-15)
        if spec.static.coef.kind == "ssvs":
            slab = 0.05 if name == "structural_sv_ssvs_lagged" else 0.5
            on = mask > 0
            np.testing.assert_allclose(
                np.asarray(gl.ssvs.log_tau0)[on], math.log(0.01) + 0.5 * np.asarray(gl.log_r)[on]
            )
            np.testing.assert_allclose(
                np.asarray(gl.ssvs.log_tau1)[on], math.log(slab) + 0.5 * np.asarray(gl.log_r)[on]
            )
        # everything else is the Phase 2 §9.2 test prior: own-lag mean 0.4, intercept s.d. 1
        m = np.asarray(spec.prior.eq.coef.mean0)
        assert m[0, s.n_det] == 0.4 and np.isclose(float(spec.prior.eq.coef.prec0[0, 0]), 1.0)
    sh = J.joint_spec("structural_constant_hs").prior.shared.gl
    assert float(sh.global_scale) == 0.1
    ng = J.joint_spec("structural_constant_ng").prior.shared.gl
    assert (float(ng.b), float(ng.c), float(ng.log_a_fixed)) == (2.0, 1.0, math.log(0.5))
    tied = J.joint_spec("structural_constant_ng_grid").prior.shared.gl
    assert float(tied.c) < 0 and float(tied.b) == 2.0  # c = a/2 (encoded c < 0)
    np.testing.assert_allclose(np.asarray(tied.a_support), [0.1, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(np.asarray(tied.a_logw), np.full(5, -math.log(5.0)))
    assert float(J.joint_spec("structural_constant_ng_grid_c1").prior.shared.gl.c) == 1.0
    for name, c in (("reduced_sv_ng_lagged", 0.01), ("structural_sv_ng_lagged", 0.005)):
        g = J.joint_spec(name).prior.shared.gl
        assert (float(g.b), float(g.c), float(np.exp(g.log_a_fixed))) == (10.0, c, 0.5)
    r2 = J.joint_spec("structural_constant_r2d2").prior.shared.gl
    assert (float(r2.b), float(np.exp(r2.log_a_fixed))) == (2.0, 0.5)
    assert np.isclose(
        float(np.exp(J.joint_spec("structural_constant_dl").prior.shared.gl.log_a_fixed)), 0.5
    )
    assert float(J.joint_spec("structural_constant_ssvs").prior.shared.gl.p_fixed) == 0.3
    b = J.joint_spec("structural_constant_ssvs_beta").prior.shared.gl
    assert (float(b.s1), float(b.s2)) == (2.0, 3.0)
    # CommonSV: ρ ~ TN(0.8, 0.15²), σ_h² ~ IG(5, 0.2); Constant λ ~ IG(5, 4) (P2 §9.2)
    csv = J.joint_spec("structural_commonsv").prior
    c = csv.shared.csv
    assert (float(c.rho_mean), float(c.rho_var), float(c.s2_shape), float(c.s2_scale)) == (
        0.8, 0.15**2, 5.0, 0.2,
    )  # fmt: skip
    assert float(c.n_eq) == 2.0
    np.testing.assert_array_equal(np.asarray(csv.eq.vol_core.shape), [5.0, 5.0])
    # conjugate rows: NIW Ψ = 0.5 I, d = n + 4, Ω = 0.01 on every lag, 1 on the intercept
    conj = J.joint_spec("conjugate_commonsv_dummies", "full")
    s = conj.static.shapes
    pr = conj.prior
    np.testing.assert_allclose(np.asarray(pr.eq.vol_core.shape), (4.0 + np.arange(1, 4)) / 2.0)
    np.testing.assert_allclose(np.asarray(pr.eq.vol_core.scale), 0.25)
    prec = np.asarray(pr.eq.coef.prec0)
    assert np.all(prec[:, 0] == 1.0) and np.all(prec[:, 1:] == 100.0)
    cm = np.asarray(pr.eq.contemp.cmask)
    np.testing.assert_allclose(np.asarray(pr.eq.contemp.prec0)[cm > 0], 0.5)
    sp = np.asarray(pr.shared.shared_prec)
    assert sp.shape == (s.q,) and np.allclose(sp[: s.k], prec[0]) and np.allclose(sp[s.k :], 0.5)
    np.testing.assert_allclose(np.asarray(pr.eq.coef.dummy_w), 1.0)  # Mode II weight
    assert isinstance(J.joint_spec("conjugate_commonsv_dummies_weighted").prior.shared.shared_prec,
                      type(R.Empty()))  # fmt: skip


def test_exogenous_design():
    """D-56: a ones column per deterministic term, N(0, 1) elsewhere, drawn once from the
    configuration seed (row, size) and held fixed; lagged rows have no fixed X."""
    a = J.joint_spec("structural_constant_hs")
    b = J.joint_spec("structural_constant_hs")
    X = a.fixed.X
    assert X.shape == (20, a.static.shapes.k) and not X.flags.writeable
    np.testing.assert_array_equal(X[:, 0], 1.0)
    np.testing.assert_array_equal(X, b.fixed.X)
    ref = np.random.default_rng((301, 0)).standard_normal((20, 3))
    np.testing.assert_array_equal(X[:, 1:], ref[:, 1:])
    other = J.joint_spec("structural_sv_hs")
    assert not np.array_equal(other.fixed.X[:, 1:], X[:, 1:])
    full = J.joint_spec("structural_constant_hs", "full").fixed.X
    assert full.shape == (50, 7)
    assert abs(full[:, 1:].mean()) < 0.2 and 0.8 < full[:, 1:].std() < 1.2
    reseeded = J.joint_spec("structural_constant_hs", design_seed=7).fixed.X
    assert not np.array_equal(reseeded, X)
    assert J.joint_spec("structural_commonsv").fixed.X is None
    d1 = J.FixedDesign.exogenous_design(np.zeros((1, 2)), 20, n_det=1, k=3, seed=1)
    assert d1 == J.FixedDesign.exogenous_design(np.zeros((1, 2)), 20, n_det=1, k=3, seed=1)
    assert d1 != J.FixedDesign(np.zeros((1, 2)), 20)
    with pytest.raises(ValueError, match="needs its fixed X"):
        J.FixedDesign(np.zeros((1, 2)), 20, exogenous=True)
    with pytest.raises(ValueError, match="belongs to an exogenous design"):
        J.FixedDesign(np.zeros((1, 2)), 20, X=np.zeros((20, 3)))
    with pytest.raises(ValueError, match="finite"):
        J.FixedDesign(np.zeros((1, 2)), 20, exogenous=True, X=np.zeros((19, 3)))


# --------------------------------------------------------------------------- 2. prior moments


def test_horseshoe_prior_against_closed_forms():
    """√ϑ ~ C⁺(0, 1), √ζ ~ C⁺(0, s), the auxiliary IG(½, ·) layers and d | v ~ N(0, r v)."""
    spec, st = draws("structural_constant_hs")
    s = 0.1
    G = spec.static.coef.n_groups
    psi_half, tri_half = special.digamma(0.5), special.polygamma(1, 0.5)
    lt, ln = st.eq.coef.log_theta, st.eq.coef.log_nu
    slots = gl_slots(spec)
    for i, r in slots:
        assert_mean(ln[:, i, r], -psi_half, "log ν")  # ν ~ IG(½, 1)
        assert_var(ln[:, i, r], tri_half, "log ν")
        assert_mean(lt[:, i, r], 0.0, "log ϑ")  # log ϑ = log ν' − log ν, ν' ⊥ ν
        assert_var(lt[:, i, r], 2.0 * tri_half, "log ϑ")
    all_theta = np.stack([lt[:, i, r] for i, r in slots], axis=1)
    assert_ks(np.exp(0.5 * all_theta), stats.halfcauchy.cdf, "√ϑ ~ C⁺(0, 1)")
    for g in range(G):
        assert_mean(st.glob.coef.log_tau[:, g], -2 * math.log(s) - psi_half, "log τ")
        assert_ks(np.exp(0.5 * st.glob.coef.log_zeta[:, g]) / s, stats.halfcauchy.cdf, "√ζ/s")
    # the local scale is drawn given the stored ν: ν·ϑ ~ IG(½, 1) independent of ν
    prod = np.stack([lt[:, i, r] + ln[:, i, r] for i, r in slots], axis=1)
    assert_ks(np.exp(-prod), stats.gamma(0.5).cdf, "1/(νϑ) ~ G(½)")
    assert_uncorrelated(prod[:, 0], {"log ν": ln[:, slots[0][0], slots[0][1]]}, "νϑ")
    # deviations given the stored scales (groups of D-38)
    log_v = np.zeros_like(lt)
    for i, r in slots:
        log_v[:, i, r] = (
            lt[:, i, r] + st.glob.coef.log_zeta[:, group_of(spec, "olcl-lagwise", i, r)]
        )
    z = standardized_deviations(spec, st, log_v)
    assert_ks(z, stats.norm.cdf, "d/√(rϑζ)")
    assert_uncorrelated(z[:, 0] ** 2, {"log ϑ": lt[:, 0, 1], "log ζ": st.glob.coef.log_zeta[:, 0]})
    assert_deterministic_slots(spec, st)
    assert_dev_log_exact(spec, st)
    for leaf in (lt, ln):
        assert np.all(leaf[:, :, 0] == 0.0)  # exactly 0 off the GL slots (P3 §2.2)
    assert_invariants(spec, st)


def _ng_checks(spec, st, c_tied: bool):
    b = 2.0
    a = grid_a(spec, st.glob.coef.a_idx)  # (draws, G)
    log_rate = 0.0 if c_tied else np.log(2.0 * 1.0 / a)
    u = st.glob.coef.log_xi + log_rate  # log G(b)
    assert_ks(u, stats.loggamma(b).cdf, "log ξ + log(2c/a) ~ log G(b)")
    counts = np.bincount(st.glob.coef.a_idx.ravel(), minlength=5) / st.glob.coef.a_idx.size
    se = math.sqrt(0.2 * 0.8 / st.glob.coef.a_idx.size)
    assert np.all(np.abs(counts - 0.2) < K_SE * se), counts  # equal grid weights
    slots = gl_slots(spec)
    lw = st.eq.coef.log_w
    for k, a_val in enumerate(J.TEST_A_GRID):
        pit, logv = [], []
        for i, r in slots:
            g = group_of(spec, "olcl-lagwise", i, r)
            sel = st.glob.coef.a_idx[:, g] == k
            pit.append(stats.gamma(a_val).cdf(np.exp(lw[sel, i, r] + st.glob.coef.log_xi[sel, g])))
            logv.append(lw[sel, i, r])
        assert_ks(np.concatenate(pit), stats.uniform.cdf, f"ϖξ | a = {a_val}")
        # from the definition v = ϑζ, ϑ ~ G(a, a/2), ζ⁻¹ ~ G(b, c): E log v | a
        c = a_val / 2.0 if c_tied else 1.0
        mean = special.digamma(a_val) - math.log(a_val / 2.0) + math.log(c) - special.digamma(b)
        assert_mean(np.concatenate(logv), mean, f"E log v | a = {a_val}")
    z = standardized_deviations(spec, st, lw)
    assert_ks(z, stats.norm.cdf, "d/√(rϖ)")
    assert_dev_log_exact(spec, st)
    assert_invariants(spec, st)


def test_normal_gamma_prior_with_grid_and_fixed_c():
    """Rows 326 (c = 1) and 304 (c = a/2) share a static: the same compiled sampler, two priors."""
    spec, st = draws("structural_constant_ng_grid_c1")
    _ng_checks(spec, st, c_tied=False)
    tied = J.joint_spec("structural_constant_ng_grid")
    assert tied.static == spec.static
    _ng_checks(tied, draws_with_prior("structural_constant_ng_grid", tied.prior, seed=1), True)


def test_r2d2_prior_against_closed_forms():
    """ϱ ~ Exp(½); ϖ | ξ ~ G(a, ξ), ξ ~ G(b, 1); the Dirichlet split: R²_g = W_g/(1 + W_g) ~
    Beta(n_g a_g, b) given a_g (GK's gamma form, P3 §3.2)."""
    spec, st = draws("structural_constant_r2d2_grid")
    b = 2.0
    slots = gl_slots(spec)
    vr = np.stack([st.eq.coef.log_varrho[:, i, r] for i, r in slots], axis=1)
    assert_ks(np.exp(vr), stats.expon(scale=2.0).cdf, "ϱ ~ Exp(½)")
    assert_ks(st.glob.coef.log_xi, stats.loggamma(b).cdf, "log ξ ~ log G(b)")
    a = grid_a(spec, st.glob.coef.a_idx)
    for g in range(spec.static.coef.n_groups):
        members = [(i, r) for i, r in slots if group_of(spec, "olcl-lagwise", i, r) == g]
        W = sum(np.exp(st.eq.coef.log_w[:, i, r]) for i, r in members)
        pit = stats.beta.cdf(W / (1.0 + W), len(members) * a[:, g], b)
        assert_ks(pit, stats.uniform.cdf, f"R² of group {g}")
    log_v = st.eq.coef.log_varrho + st.eq.coef.log_w - math.log(2.0)
    assert_ks(standardized_deviations(spec, st, log_v), stats.norm.cdf, "d/√(rϱϖ/2)")
    assert_dev_log_exact(spec, st)
    assert_invariants(spec, st)


def test_dirichlet_laplace_prior_against_closed_forms():
    """θ̃ ~ G(a, ½); ϱ ~ Exp(½); d/√r | θ̃ is Laplace(θ̃) (ϱ integrated); Σ_{j∈g} θ̃_j ~
    G(n_g a, ½) (gamma–Dirichlet: θ̃ = φω, φ ~ Dir(a…a), ω ~ G(n_g a, ½))."""
    spec, st = draws("structural_constant_dl_grid")
    slots = gl_slots(spec)
    a = grid_a(spec, st.glob.coef.a_idx)
    lt, lv = st.eq.coef.log_theta, st.eq.coef.log_varrho
    d, _ = deviations(spec, st)
    pit, lap = [], []
    for i, r in slots:
        g = group_of(spec, "olcl-lagwise", i, r)
        pit.append(stats.gamma.cdf(np.exp(lt[:, i, r]), a[:, g], scale=2.0))
        lap.append(
            np.abs(d[:, i, r]) / (math.exp(0.5 * log_r_of(spec, i, r)) * np.exp(lt[:, i, r]))
        )
    assert_ks(np.concatenate(pit), stats.uniform.cdf, "θ̃ | a ~ G(a, ½)")
    assert_ks(np.concatenate(lap), stats.expon.cdf, "|d|/(√r θ̃) ~ Exp(1) (Laplace)")
    vr = np.stack([lv[:, i, r] for i, r in slots], axis=1)
    assert_ks(np.exp(vr), stats.expon(scale=2.0).cdf, "ϱ ~ Exp(½)")
    for g in range(spec.static.coef.n_groups):
        members = [(i, r) for i, r in slots if group_of(spec, "olcl-lagwise", i, r) == g]
        omega = sum(np.exp(lt[:, i, r]) for i, r in members)
        assert_ks(stats.gamma.cdf(omega, len(members) * a[:, g], scale=2.0), stats.uniform.cdf,
                  f"ω of group {g}")  # fmt: skip
    log_v = lv + 2.0 * lt
    assert_ks(standardized_deviations(spec, st, log_v), stats.norm.cdf, "d/√(rϱθ̃²)")
    assert_dev_log_exact(spec, st)
    assert_invariants(spec, st)


def test_ssvs_prior_against_closed_forms():
    """p_g ~ Beta(2, 3) (row 310) or fixed 0.3 (row 309); γ_j | p_g ~ Bernoulli(p_g); d_j | γ_j ~
    N(0, τ²_{γ_j}) with τ₀ = 0.01√r, τ₁ = 0.5√r."""
    spec, st = draws("structural_constant_ssvs_beta")
    slots = gl_slots(spec)
    p = np.exp(st.glob.coef.log_p)
    np.testing.assert_allclose(p + np.exp(st.glob.coef.log1m_p), 1.0, rtol=1e-12)
    assert_ks(p, stats.beta(2.0, 3.0).cdf, "p ~ Beta(2, 3)")
    gam = st.eq.coef.gamma
    resid, z = [], []
    d = deviations(spec, st)[0]
    for i, r in slots:
        pg = p[:, group_of(spec, "olcl-lagwise", i, r)]
        resid.append(gam[:, i, r] - pg)  # E[γ − p] = 0 and γ − p ⊥ p
        tau = np.where(gam[:, i, r] == 1, 0.5, 0.01) * math.exp(0.5 * log_r_of(spec, i, r))
        z.append(d[:, i, r] / tau)
        assert_uncorrelated(gam[:, i, r] - pg, {"p": pg}, "γ | p")
    assert_mean(np.concatenate(resid), 0.0, "γ − p")
    assert_ks(np.concatenate(z), stats.norm.cdf, "d/τ_γ")
    assert set(np.unique(gam)) == {0, 1} and np.all(gam[:, :, 0] == 0)
    assert np.all(st.eq.coef.rb_incl == 0.0)
    assert_invariants(spec, st)
    fixed_spec, fixed = draws("structural_constant_ssvs")
    gf = np.stack([fixed.eq.coef.gamma[:, i, r] for i, r in gl_slots(fixed_spec)], axis=1)
    assert_mean(gf, 0.3, "γ ~ Bernoulli(0.3)")
    np.testing.assert_allclose(fixed.glob.coef.log_p, math.log(0.3), rtol=1e-15)


def test_common_sv_prior_truncated_normal_and_inverse_gamma():
    """ρ ~ N(0.8, 0.15²)·1(|ρ| < 1), σ_h² ~ IG(5, 0.2), stationary h_0 at the first data row,
    h_t = ρh_{t−1} + σ_hη_t for 1 ≤ t < T_eff, padded states 0 (P3 §5.1, D-46); λ_i ~ IG(5, 4)."""
    spec, st = draws("structural_commonsv")
    gv = st.glob.vol
    T = spec.fixed.T_eff
    sd = 0.15
    tn = stats.truncnorm((-1 - 0.8) / sd, (1 - 0.8) / sd, loc=0.8, scale=sd)
    assert_ks(gv.rho, tn.cdf, "ρ ~ TN")
    assert np.all(np.abs(gv.rho) < 1.0)
    sigma = np.exp(gv.log_sigma)
    assert_ks(sigma**2, stats.invgamma(5.0, scale=0.2).cdf, "σ_h² ~ IG(5, 0.2)")
    h = gv.h
    z0 = h[:, 0] * np.sqrt(1 - gv.rho**2) / sigma
    eta = (h[:, 1:T] - gv.rho[:, None] * h[:, : T - 1]) / sigma[:, None]
    for label, x in (("z_0", z0), ("η", eta)):
        assert_mean(x, 0.0, label)
        assert_var(x, 1.0, label)
    for label, x in (("z_0²", z0**2), ("mean η²", (eta**2).mean(1)), ("mean η", eta.mean(1))):
        assert_uncorrelated(x, {"ρ": gv.rho, "log σ": gv.log_sigma}, label)
    assert np.all(h[:, T:] == 0.0)
    for leaf in (gv.acc_h, gv.acc_rho, gv.acc_newton, gv.rb_ch_sum, gv.rb_ch_sq):
        assert np.all(leaf == 0.0)
    for i in range(spec.static.shapes.n):
        assert_ks(np.exp(st.eq.vol.core.log_lam[:, i]), stats.invgamma(5.0, scale=4.0).cdf, "λ")
    assert_invariants(spec, st)


def test_conjugate_common_sv_prior_is_niw_with_beta_rho_and_scaled_chi2():
    """Row 325: (ρ+1)/2 ~ Beta(10, 2), σ_h² ~ 0.1·χ²₁; the Mode II arrays give Σ ~ IW(0.5 I,
    n + 4) (each Σ_jj ~ IG((d − n + 1)/2, ψ/2), E Σ_ij = 0) and B | Σ ~ MN(M, Σ ⊗ Ω)."""
    spec, st = draws("conjugate_commonsv_parity")
    gv = st.glob.vol
    assert_ks((gv.rho + 1) / 2, stats.beta(10.0, 2.0).cdf, "(ρ+1)/2 ~ Beta(10, 2)")
    assert_ks(np.exp(2 * gv.log_sigma) / 0.1, stats.chi2(1).cdf, "σ_h²/B ~ χ²₁")
    s = spec.static.shapes
    n, nd, k = s.n, s.n_det, s.k
    a = st.eq.struct.a
    lam = np.exp(st.eq.vol.core.log_lam)
    A = np.broadcast_to(np.eye(n), (a.shape[0], n, n)).copy()
    for i in range(n):
        A[:, i, :i] = a[:, i, :i]
    Ainv = np.linalg.inv(A)
    Sigma = np.einsum("cij,cj,ckj->cik", Ainv, lam, Ainv)
    for j in range(n):
        assert_ks(Sigma[:, j, j], stats.invgamma(2.5, scale=0.25).cdf, f"Σ_{j}{j}")
    for i in range(n):
        for j in range(i):
            assert_mean(Sigma[:, i, j], 0.0, f"Σ_{i}{j}")
    B = np.asarray(jax.vmap(lambda x: J.reduced_form_coefficients(x, spec.prior, spec.static))(
        jax.tree.map(jnp.asarray, st)))  # fmt: skip
    M = np.asarray(spec.prior.eq.coef.mean0).T  # (k, n)
    omega = np.where(np.arange(k) < nd, 1.0, 0.01)
    D = B - M
    for j in range(n):
        assert_ks(D[:, :, j] / np.sqrt(Sigma[:, j, j, None] * omega), stats.norm.cdf, f"B[:, {j}]")
    for i in range(n):  # Kronecker: Cov(B_ri, B_rj | Σ) = Σ_ij Ω_rr
        for j in range(i):
            assert_mean(D[:, :, i] * D[:, :, j] / omega - Sigma[:, i, j, None], 0.0, f"Σ⊗Ω {i}{j}")


def test_reduced_gl_with_common_h():
    """Row 328: reduced form × CommonSV × NG (fixed a = 0.5, b = 2, c = 1): Π's lag rows have
    d | ϖ ~ N(0, rϖ) with the reduced r (no ι), ξ ~ G(b, 2c/a), intercept rows N(0, 1)."""
    spec, st = draws("reduced_commonsv_ng")
    assert_ks(st.glob.coef.log_xi + math.log(4.0), stats.loggamma(2.0).cdf, "log ξ")
    G = spec.static.coef.n_groups
    pit = []
    for i, r in gl_slots(spec):
        g = group_of(spec, "olcl-lagwise", i, r)
        assert g < G
        pit.append(
            stats.gamma(0.5).cdf(np.exp(st.eq.coef.log_w[:, i, r] + st.glob.coef.log_xi[:, g]))
        )
    assert_ks(np.concatenate(pit), stats.uniform.cdf, "ϖξ ~ G(a)")
    assert_ks(standardized_deviations(spec, st, st.eq.coef.log_w), stats.norm.cdf, "π deviations")
    assert_deterministic_slots(spec, st)
    assert_dev_log_exact(spec, st)
    assert isinstance(st.eq.mean, type(R.Empty())) and st.glob.mean.dev_log.shape[1:] == (3, 4)
    assert np.all(np.abs(st.glob.vol.rho) < 1) and np.all(st.glob.vol.h[:, 20:] == 0)
    assert_invariants(spec, st)


def test_gl_references_refuse_per_equation_use():
    spec = J.joint_spec("structural_constant_hs")
    ref = R.reference("horseshoe", spec.static)
    pi = jax.tree.map(lambda x: x[0], spec.prior.eq)
    with pytest.raises(TypeError, match="sample_local"):
        ref.prior_sample_eq(random.key(0), 0, pi, spec.static, spec.tmask, 20)
    with pytest.raises(TypeError, match="local_log_density"):
        ref.log_prior_eq(None, pi, spec.static, spec.tmask)
    lin = R.reference("linear_structural", spec.static)
    with pytest.raises(ValueError, match="fixed coefficient moments"):
        lin.prior_sample_eq(random.key(0), 0, pi, spec.static, spec.tmask, 20)
    _, st = a_draw(spec.name)
    with pytest.raises(ValueError, match="couples equations"):
        LJ.log_prior_eq(one(st.eq, 0), pi, spec.static, spec.tmask)
    csv = J.joint_spec("structural_commonsv")
    _, st = a_draw(csv.name)
    with pytest.raises(ValueError, match="common h"):
        R.effective_log_var(st.eq.vol, csv.static)
    with pytest.raises(ValueError, match="tmask"):
        R.reference("common_sv").prior_sample_global(random.key(0), csv.prior, csv.static)
    assert R.core_kind(csv.static) == "constant_common"
    assert set(R.REFERENCES) >= {
        "horseshoe", "normal_gamma", "r2d2", "dirichlet_laplace", "ssvs", "constant_common",
        "common_sv",
    }  # fmt: skip


# --------------------------------------------------------------------------- 3. simulators


def numpy_simulate(spec, st, eps):
    """``y_t`` by a Python loop with dense solves: lagged (recursive in t) or exogenous (fixed
    ``x_t``, no feedback), ``e_{i,t} = exp(½(log λ_i + h_t))ε`` under CommonSV."""
    st = _np(st)
    sh = spec.static.shapes
    n, p, T = sh.n, sh.lags, spec.fixed.T_eff
    a = st.eq.struct.a
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = a[i, :i]
    vol = spec.static.vol
    if vol.persistent == "constant":
        lv = np.repeat(st.eq.vol.core.log_lam[:, None], sh.T_pad, axis=1)
    else:
        lv = st.eq.vol.core.h[:, 1:].copy()
        if vol.transitory in ("outlier", "outlier_t"):
            lv = lv + 2.0 * np.log(st.eq.vol.trans.o_idx + 1.0)
    if vol.common:
        lv = lv + st.glob.vol.h[None, :]
    e = np.exp(0.5 * lv.T) * eps
    if spec.static.mean.kind == "linear_structural":
        G = st.eq.mean.delta + a @ np.asarray(spec.prior.shared.Mc).T
    else:
        G = st.glob.mean.Pi.T
    hist = list(np.asarray(spec.fixed.presample))
    Y = np.zeros((sh.T_pad, n))
    for t in range(T):
        if spec.fixed.exogenous:
            x = spec.fixed.X[t]
        else:
            x = np.concatenate([[1.0], *[hist[-lag] for lag in range(1, p + 1)]])
        if spec.static.mean.kind == "linear_reduced":
            Y[t] = G @ x + np.linalg.solve(A, e[t])
        else:
            Y[t] = np.linalg.solve(A, G @ x + e[t])
        hist.append(Y[t])
    return Y


@pytest.mark.parametrize(
    ("name", "size"),
    [("structural_constant_hs", "small"), ("structural_sv_outliers_hs_eqwise", "small"),
     ("reduced_commonsv_ng", "small"), ("structural_commonsv", "small"),
     ("structural_commonsv", "full"), ("reduced_commonsv", "small"),
     ("conjugate_commonsv_parity", "small")],
)  # fmt: skip
def test_simulate_obs_matches_numpy_loop(name, size):
    spec, st = a_draw(name, size)
    sh = spec.static.shapes
    eps = np.asarray(random.normal(random.key(12), (sh.T_pad, sh.n), dtype=jnp.float64))
    data = _np(jax.jit(lambda s, e: S.simulate_obs_core(e, s, spec.static, spec.prior, spec.fixed))(
        st, eps))  # fmt: skip
    T, p = spec.fixed.T_eff, sh.lags
    ref = numpy_simulate(spec, st, eps)
    np.testing.assert_allclose(data.Y[:T], ref[:T], rtol=1e-12, atol=1e-12)
    assert np.all(data.Y[T:] == 0.0) and np.all(data.X[T:] == 0.0)
    np.testing.assert_array_equal(data.tmask, (np.arange(sh.T_pad) < T).astype(float))
    assert int(data.T_eff) == T and float(data.t_last) == spec.fixed.t0 + T - 1
    stacked = np.vstack([spec.fixed.presample, data.Y[:T]])
    np.testing.assert_array_equal(data.hist, stacked[-p:])
    if spec.fixed.exogenous:  # the fixed design, not the lags of y (D-56)
        np.testing.assert_array_equal(data.X[:T], spec.fixed.X)
    else:
        from rai_bvar.data.design import lagged_design

        _, Xd = lagged_design(stacked, p, "c")
        np.testing.assert_allclose(data.X[:T], Xd, rtol=0, atol=1e-15)
    if spec.static.vol.common:  # q_t e^{−h_t} = Σ_i ε²_{it} at the data-generating θ (N08 §A4)
        e = np.asarray(LJ.residuals(st, jax.tree.map(jnp.asarray, data), spec.prior, spec.static))
        lam = np.exp(np.asarray(st.eq.vol.core.log_lam))
        q = np.sum(e[:, :T] ** 2 / lam[:, None], axis=0)
        np.testing.assert_allclose(q * np.exp(-np.asarray(st.glob.vol.h)[:T]),
                                   np.sum(eps[:T] ** 2, axis=1), rtol=1e-10)  # fmt: skip


def test_data_irrelevant_spec_simulates_zero_targets():
    spec, _, data = a_dataset("gl_scales_only_horseshoe")
    assert np.all(np.asarray(data.Y) == 0.0)
    np.testing.assert_array_equal(np.asarray(data.X)[:20], spec.fixed.X)
    assert spec.names and all(not J._data_dependent(k) for k in spec.all_names)
    assert "log_prior" in spec.names and not spec.raw_moments


# --------------------------------------------------------------------------- 4. log_joint vs SciPy


def scipy_gl_log_prior(spec, st) -> float:
    """``Σ log p(coefficient | scales) + log p(scales)`` from the §3.2 definitions with SciPy,
    groups and ``r`` built here; GL deviations from ``dev_log`` (P3 §3.1)."""
    kind = spec.static.coef.kind
    groups = {"structural_sv_outliers_hs_eqwise": "equation-wise"}.get(spec.name, "olcl-lagwise")
    if spec.name.endswith("_global"):
        groups = "global"
    G = spec.static.coef.n_groups
    params = dict(J._P3_DEFS[spec.name].gl)
    grid = params.get("a") == "grid"
    a_g = (np.asarray(J.TEST_A_GRID)[st.glob.coef.a_idx] if grid
           else np.full(G, params.get("a", 1.0)))  # fmt: skip
    d, dev_log = deviations(spec, st)
    m = np.asarray(spec.prior.eq.coef.mean0)
    coef = d + m
    prec0 = np.asarray(spec.prior.eq.coef.prec0)
    total = 0.0
    for i in range(spec.static.shapes.n):
        total += stats.norm.logpdf(coef[i, 0], m[i, 0], 1.0 / math.sqrt(prec0[i, 0]))
    ex = lambda x: math.exp(float(x))  # noqa: E731
    for i, r in gl_slots(spec):
        g, r_j = group_of(spec, groups, i, r), ex(log_r_of(spec, i, r))
        c = spec.static.coef
        sc = st.eq.coef
        if kind == "horseshoe":
            th, nu = ex(sc.log_theta[i, r]), ex(sc.log_nu[i, r])
            total += stats.invgamma.logpdf(th, 0.5, scale=1 / nu) + stats.invgamma.logpdf(nu, 0.5)
            var = r_j * th * ex(st.glob.coef.log_zeta[g])
        elif kind == "normal_gamma":
            w, xi = ex(sc.log_w[i, r]), ex(st.glob.coef.log_xi[g])
            total += stats.gamma.logpdf(w, a_g[g], scale=1 / xi)
            var = r_j * w
        elif kind == "r2d2":
            rho, w, xi = ex(sc.log_varrho[i, r]), ex(sc.log_w[i, r]), ex(st.glob.coef.log_xi[g])
            total += stats.expon.logpdf(rho, scale=2.0) + stats.gamma.logpdf(
                w, a_g[g], scale=1 / xi
            )
            var = r_j * rho * w / 2.0
        elif kind == "dirichlet_laplace":
            th, rho = ex(sc.log_theta[i, r]), ex(sc.log_varrho[i, r])
            total += stats.gamma.logpdf(th, a_g[g], scale=2.0) + stats.expon.logpdf(rho, scale=2.0)
            var = r_j * rho * th**2
        else:
            p = ex(st.glob.coef.log_p[g]) if c.p_beta else params["p"]
            gam = int(sc.gamma[i, r])
            total += stats.bernoulli.logpmf(gam, p)
            var = (params["tau1"] if gam else params["tau0"]) ** 2 * r_j
        total += stats.norm.logpdf(np.sign(d[i, r]) * ex(dev_log[i, r]), 0.0, math.sqrt(var))
    for g in range(G):
        if kind == "horseshoe":
            z, t = ex(st.glob.coef.log_zeta[g]), ex(st.glob.coef.log_tau[g])
            s = params["global_scale"]
            total += stats.invgamma.logpdf(z, 0.5, scale=1 / t)
            total += stats.invgamma.logpdf(t, 0.5, scale=1 / s**2)
        elif kind == "normal_gamma":
            c = a_g[g] / 2.0 if params["c"] == "a/2" else params["c"]
            total += stats.gamma.logpdf(
                ex(st.glob.coef.log_xi[g]), params["b"], scale=a_g[g] / (2 * c)
            )
        elif kind == "r2d2":
            total += stats.gamma.logpdf(ex(st.glob.coef.log_xi[g]), params["b"])
        elif kind == "ssvs" and spec.static.coef.p_beta:
            total += stats.beta.logpdf(ex(st.glob.coef.log_p[g]), params["s1"], params["s2"])
        if grid:
            total += math.log(1.0 / len(J.TEST_A_GRID))
    return total


def scipy_rest_log_prior(spec, st) -> float:
    """The Constant-core compositions' other parts: λ_i ~ IG, a_ij ~ N(0, 0.7²) (×λ_i when λ-scaled;
    conjugate: N(0, λ_i/ψ)), Mode I / Mode II δ (non-GL rows), and CommonSV's (ρ, σ_h, h)."""
    s, static = spec.static.shapes, spec.static
    pr = _np(spec.prior)
    total = 0.0
    mode2 = static.mean.coef_scaling == "lambda"
    for i in range(s.n):
        lam = math.exp(float(st.eq.vol.core.log_lam[i]))
        total += stats.invgamma.logpdf(lam, pr.eq.vol_core.shape[i], scale=pr.eq.vol_core.scale[i])
        scale_a = lam if static.struct.contemp_scaling == "lambda" else 1.0
        for j in range(i):
            sd = (math.sqrt(1 / 0.5) if mode2 else 0.7) * math.sqrt(scale_a)
            total += stats.norm.logpdf(float(st.eq.struct.a[i, j]), 0.0, sd)
        if not static.coef.varies and static.mean.kind == "linear_structural":
            omega = (
                np.where(np.arange(s.k) < s.n_det, 1.0, 0.01) if mode2 else 1 / pr.eq.coef.prec0[i]
            )
            total += np.sum(stats.norm.logpdf(st.eq.mean.delta[i], pr.eq.coef.mean0[i],
                                              np.sqrt(omega * (lam if mode2 else 1.0))))  # fmt: skip
    if not static.coef.varies and static.mean.kind == "linear_reduced":
        sd = 1 / np.sqrt(pr.eq.coef.prec0.T)
        total += np.sum(stats.norm.logpdf(st.glob.mean.Pi, pr.eq.coef.mean0.T, sd))
    if static.vol.common:
        gv, T = st.glob.vol, spec.fixed.T_eff
        rho, sig = float(gv.rho), math.exp(float(gv.log_sigma))
        if static.vol.csv_rho_prior == "beta":
            total += stats.beta.logpdf((rho + 1) / 2, 10.0, 2.0) - math.log(2.0)
        else:
            total += stats.truncnorm.logpdf(rho, -1.8 / 0.15, 0.2 / 0.15, loc=0.8, scale=0.15)
        if static.vol.csv_sigma2_prior == "scaled_chi2":
            total += stats.halfnorm.logpdf(sig, scale=math.sqrt(0.1))
        else:
            total += stats.invgamma.logpdf(sig**2, 5.0, scale=0.2) + math.log(2 * sig)
        h = gv.h
        total += stats.norm.logpdf(h[0], 0.0, sig / math.sqrt(1 - rho**2))
        total += np.sum(stats.norm.logpdf(h[1:T], rho * h[: T - 1], sig))
    return float(total)


def scipy_loglik(spec, st, data) -> float:
    st, data = _np(st), _np(data)
    T, n = spec.fixed.T_eff, spec.static.shapes.n
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = st.eq.struct.a[i, :i]
    Y, X = data.Y[:T], data.X[:T]
    if spec.static.mean.kind == "linear_structural":
        Gam = st.eq.mean.delta + st.eq.struct.a @ np.asarray(spec.prior.shared.Mc).T
        E = Y @ A.T - X @ Gam.T
    else:
        E = (Y - X @ st.glob.mean.Pi) @ A.T
    lv = st.eq.vol.core.log_lam[:, None] + (
        st.glob.vol.h[None, :T] if spec.static.vol.common else 0.0
    )
    return float(np.sum(stats.norm.logpdf(E.T, 0.0, np.exp(0.5 * lv))))


LOG_JOINT_SPECS = (
    "structural_constant_hs", "structural_constant_ng_grid", "structural_constant_ng_grid_c1",
    "structural_constant_ng", "structural_constant_r2d2_grid", "structural_constant_dl_grid",
    "structural_constant_ssvs", "structural_constant_ssvs_beta", "structural_iw_equivalent_hs",
    "reduced_commonsv_ng", "structural_commonsv",
    "conjugate_commonsv_parity", "conjugate_commonsv_block7", "reduced_commonsv",
)  # fmt: skip


@pytest.mark.parametrize("name", LOG_JOINT_SPECS)
def test_log_joint_matches_scipy(name):
    spec = J.joint_spec(name)

    @jax.jit
    def evaluate(st, key):
        data = spec.simulate(key, st)
        lp = LJ.log_prior(st, spec.prior, spec.static, spec.tmask)
        return data, lp, LJ.loglik(st, data, spec.prior, spec.static), spec.log_joint(st, data)

    for c in range(2):
        _, st = a_draw(name, c=c)
        data, got, ll, lj = evaluate(st, random.key(100 + c))
        host = _np(st)
        ref = scipy_rest_log_prior(spec, host)
        if spec.static.coef.varies:
            ref += scipy_gl_log_prior(spec, host)
        assert abs(float(got) - ref) <= 1e-10 * (1 + abs(ref)), (name, c, float(got), ref)
        ll_ref = scipy_loglik(spec, st, data)
        assert abs(float(ll) - ll_ref) <= 1e-10 * (1 + abs(ll_ref)), (name, c, float(ll), ll_ref)
        assert abs(float(lj) - (ref + ll_ref)) <= 1e-10 * (1 + abs(ref + ll_ref))


def test_gl_log_prior_terms_and_padding():
    """The GL decomposition plus the Phase 2 and CommonSV parts is log p; padded h never enters;
    the deviation density reads dev_log (exact for deviations far below the resolution of m + d)."""
    spec, st = a_draw("reduced_commonsv_ng")
    dev, local, group = LJ.gl_log_prior_terms(st, spec.prior, spec.static)
    mask = np.asarray(spec.prior.eq.coef.gl.mask)
    assert np.all(np.asarray(local)[mask == 0] == 0.0)
    rest = scipy_rest_log_prior(spec, _np(st))  # λ, a, the deterministic π and CommonSV
    total = float(jnp.sum(dev) + jnp.sum(local) + group) + rest
    lp = jax.jit(lambda s: LJ.log_prior(s, spec.prior, spec.static, spec.tmask))
    assert math.isclose(float(lp(st)), total, rel_tol=1e-12)
    T = spec.fixed.T_eff  # moving a padded h state changes nothing
    moved = jax.tree_util.tree_map_with_path(
        lambda p, x: x.at[T:].set(5.0) if leaf_path(p) == "glob.vol.h" else x, st
    )
    assert float(lp(st)) == float(lp(moved))
    # a deviation of e^{−800}: dev_log carries it, the density stays finite and exact
    lv = R.reference("normal_gamma", spec.static).log_kernel_var(
        st.eq.coef, spec.prior.eq.coef.gl, st.glob.coef, spec.prior.shared.gl, spec.static
    )
    coef, dev_log = LJ.gl_coefficients(st, spec.static)
    d2 = R.gl_coefficient_log_density(coef, dev_log.at[0, 1].set(-800.0), spec.prior.eq.coef, lv)
    expect = -0.5 * (R.LOG_2PI + float(lv[0, 1]) + math.exp(-1600.0 - float(lv[0, 1])))
    assert math.isclose(float(d2[0, 1]), expect, rel_tol=1e-14)


# --------------------------------------------------------------------------- 5. test functions


def numpy_gl_functions(spec, st) -> dict[str, float]:
    """The P3 §9.4 GL test functions written again from their definitions."""
    st = _np(st)
    kind, s = spec.static.coef.kind, spec.static.shapes
    d, dev_log = deviations(spec, jax.tree.map(lambda x: x[None], st))
    d, dev_log = d[0], dev_log[0]
    out = {}
    for i, r in gl_slots(spec):
        lr = log_r_of(spec, i, r)
        out[f"eq{i}.coef.atan_dev[{r}]"] = math.atan(d[i, r] / math.exp(0.5 * lr))
        out[f"eq{i}.coef.small_dev[{r}]"] = float(
            math.exp(dev_log[i, r]) < 0.1 * math.exp(0.5 * lr)
        )
    names = {"horseshoe": ("log_theta", "log_nu"), "normal_gamma": ("log_w",),
             "r2d2": ("log_varrho", "log_w"), "dirichlet_laplace": ("log_theta", "log_varrho"),
             "ssvs": ("gamma",)}[kind]  # fmt: skip
    for i, r in ((0, s.n_det), (s.n - 1, s.n_det)):
        for nm in names:
            out[f"eq{i}.coef.{nm}[{r}]"] = float(getattr(st.eq.coef, nm)[i, r])
    G = spec.static.coef.n_groups
    groups = "global" if spec.name.endswith("_global") else "olcl-lagwise"
    members = {
        g: [(i, r) for i, r in gl_slots(spec) if group_of(spec, groups, i, r) == g]
        for g in range(G)
    }
    gc = st.glob.coef
    for g in range(G):
        if kind == "horseshoe":
            out[f"glob.coef.log_zeta[{g}]"] = gc.log_zeta[g]
            out[f"glob.coef.log_tau[{g}]"] = gc.log_tau[g]
        if kind in ("normal_gamma", "r2d2"):
            out[f"glob.coef.log_xi[{g}]"] = gc.log_xi[g]
        if spec.static.coef.a_grid:
            out[f"glob.coef.a_idx[{g}]"] = float(gc.a_idx[g])
            if kind in ("normal_gamma", "r2d2"):
                out[f"glob.coef.a*log_xi[{g}]"] = J.TEST_A_GRID[gc.a_idx[g]] * gc.log_xi[g]
        if kind == "ssvs":
            out[f"glob.coef.N1[{g}]"] = float(sum(st.eq.coef.gamma[i, r] for i, r in members[g]))
            if spec.static.coef.p_beta:
                out[f"glob.coef.log_p[{g}]"] = gc.log_p[g]
        else:
            c = st.eq.coef
            vals = []
            for i, r in members[g]:
                if kind == "horseshoe":
                    vals.append(c.log_theta[i, r] + gc.log_zeta[g])
                elif kind == "normal_gamma":
                    vals.append(c.log_w[i, r])
                elif kind == "r2d2":
                    vals.append(c.log_varrho[i, r] + c.log_w[i, r] - math.log(2))
                else:
                    vals.append(c.log_varrho[i, r] + 2 * c.log_theta[i, r])
            out[f"glob.coef.mean_log_v[{g}]"] = np.mean(vals)
    return {k: float(v) for k, v in out.items()}


@pytest.mark.parametrize(
    "name",
    ["structural_constant_hs", "structural_constant_ng_grid", "structural_constant_r2d2_grid",
     "structural_constant_dl_grid", "structural_constant_ssvs_beta", "reduced_sv_hs_global"],
)  # fmt: skip
def test_gl_test_functions_match_numpy(name):
    spec = J.joint_spec(name)
    refs = R.references_for(spec.static)
    for seed in (0, 1):
        _, st = a_draw(name, c=seed)
        got = {
            k: float(v)
            for k, (v, fin) in TF._gl_functions(st, spec.prior, spec.static, refs).items()
        }
        flags = {fin for _, fin in TF._gl_functions(st, spec.prior, spec.static, refs).values()}
        ref = numpy_gl_functions(spec, st)
        assert sorted(got) == sorted(ref), sorted(set(got) ^ set(ref))
        bad = {
            k: (got[k], ref[k])
            for k in ref
            if not np.isclose(got[k], ref[k], rtol=1e-12, atol=1e-12)
        }
        assert not bad and flags == {True}, (name, bad)


def test_common_sv_test_functions_match_numpy():
    spec = J.joint_spec("structural_commonsv")
    spec, st, data = a_dataset(spec.name)
    fns = TF.test_functions(st, data, spec.prior, spec.static)
    host = _np(st)
    gv, T = host.glob.vol, spec.fixed.T_eff
    h = gv.h
    expect = {"h0": h[0], "hmid": h[T // 2], "hT": h[T - 1], "mean_h": h[:T].mean(),
              "mean_abs_h": np.abs(h[:T]).mean(), "rho": gv.rho, "rho^2": gv.rho**2,
              "sigma": math.exp(gv.log_sigma), "sigma^2": math.exp(2 * gv.log_sigma),
              "log_sigma": gv.log_sigma, "rho*log_sigma": gv.rho * gv.log_sigma}  # fmt: skip
    expect |= {f"atan_{k}": math.atan(expect[k]) for k in ("h0", "hmid", "hT", "mean_h")}
    for k, v in expect.items():
        assert math.isclose(
            float(fns[f"glob.vol.{k}"][0]), float(v), rel_tol=1e-12, abs_tol=1e-14
        ), k
    e = np.asarray(LJ.residuals(st, data, spec.prior, spec.static))[:, :T]
    q = np.sum(e**2 / np.exp(host.eq.vol.core.log_lam)[:, None], axis=0)
    W = float(np.sum(q * np.exp(-h[:T])))
    assert math.isclose(float(fns["joint.W"][0]), W, rel_tol=1e-12)
    ll = scipy_loglik(spec, st, data)
    assert math.isclose(float(fns["loglik"][0]), ll, rel_tol=1e-12)


def test_finite_variance_flags_follow_the_p3_tails():
    """HS: E|d| = ∞ (tail 1); NG/R2D2 with b = 2: tail 4 (d finite variance, d² not); DL, SSVS:
    every moment; CommonSV under the TN ρ prior: Var(h_t) = ∞ (density positive at ρ = ±1), under
    Beta(10, 2): finite; bounded transforms always finite."""
    hs = J.joint_spec("structural_constant_hs")
    assert hs.tails.coef == 1.0
    fl = hs.function_flags
    assert not fl["eq0.mean.delta1"] and fl["eq0.mean.delta0"] and fl["eq0.mean.delta0^2"]
    assert not fl["eq1.joint.B_own1"] and not fl["joint.gamma_minus_AM_sq"]
    assert fl["eq1.coef.atan_dev[1]"] and fl["glob.coef.log_zeta[0]"] and fl["loglik"]
    assert not hs.raw_moments
    ng = J.joint_spec("structural_constant_ng")
    assert ng.tails.coef == 4.0
    assert ng.function_flags["eq0.mean.delta1"] and not ng.function_flags["eq0.mean.delta1^2"]
    lagged = J.joint_spec("structural_sv_ng_lagged")
    assert lagged.tails.coef == 20.0 and lagged.function_flags["eq0.mean.delta1^2"]
    dl = J.joint_spec("structural_constant_dl")
    assert dl.tails.coef == math.inf and dl.function_flags["joint.gamma_minus_AM_sq"]
    assert dl.raw_moments  # Constant core, uncoupled, E λ² < ∞, E d⁴ < ∞
    red = J.joint_spec("reduced_constant_hs_global")
    assert not red.function_flags["glob.mean.Pi[1,0]"] and red.function_flags["glob.mean.Pi[0,0]"]
    assert not red.function_flags["glob.mean.Pi[1,0]*Pi[1,1]"]
    tn = J.joint_spec("structural_commonsv")
    assert (tn.tails.sigma2, tn.tails.h2) == (5.0, 1.0)
    f = tn.function_flags
    assert not f["glob.vol.hT"] and not f["loglik"] and not f["data.mean_log_y2[0]"]
    assert f["glob.vol.atan_hT"] and f["joint.W"] and f["glob.vol.rho"] and f["glob.vol.sigma^2"]
    assert f["eq0.core.log_lam"] and f["log_prior"] and not tn.raw_moments
    beta = J.joint_spec("conjugate_commonsv_parity")
    assert (beta.tails.sigma2, beta.tails.h2) == (math.inf, 2.0)
    assert beta.function_flags["glob.vol.hT"] and beta.function_flags["loglik"]
    assert beta.tails.lam == 2.5  # (d − n + 1)/2: λ-scaled δ has finite variance
    assert beta.function_flags["eq0.mean.delta1^2"]
    # Phase 2 tails are unchanged (coef = 0 is never read for a fixed prior)
    assert J.joint_spec(
        "structural_mode2_constant", prior_kwargs={"ig_shape": 2.0}
    ).tails == R.PriorTails(lam=2.0, sigma2=math.inf, h2=math.inf)


# --------------------------------------------------------------------------- 6. JointSpecP3 kernels


class _FakeStep:
    block = Block.G_COEF_GLOBAL

    def __init__(self, calls, block):
        self.calls, self.block = calls, block

    def sample(self, key, st, ctx, eq_map):
        self.calls.append(("sample", ctx))
        bump = lambda x: x + 1.0 if jnp.issubdtype(x.dtype, jnp.floating) else x  # noqa: E731
        return jax.tree.map(bump, st), StepInfo.success()


class _FakeProgram:
    def __init__(self, static, calls, block):
        self.static, self.global_steps = static, (_FakeStep(calls, block),)
        self.eq_map = jax.vmap


def _fake_program(calls):
    def factory(static, overrides=()):
        calls.append(("program", dict(overrides)))
        block = Block.G_COMMON_SV if static.vol.common else Block.G_COEF_GLOBAL
        return _FakeProgram(static, calls, block), (lambda data, prior: "cache")

    return factory


def _moved(old, new):
    out = set()
    for (path, a), b in zip(
        jax.tree_util.tree_flatten_with_path(old)[0], jax.tree.leaves(new), strict=True
    ):
        if jnp.issubdtype(a.dtype, jnp.floating) and bool(jnp.any(a != b)):
            out.add(leaf_path(path))
    return out


def test_global_step_kernel_with_refresh(monkeypatch):
    calls: list = []
    monkeypatch.setattr(J, "production_program", _fake_program(calls))
    spec, st, data = a_dataset("gl_scales_only_horseshoe")
    new = spec.step(random.key(2), jnp.uint32(4), st, data)
    moved = _moved(st, new)
    # the scale step's writes are kept, every other leaf is restored, then d | scales is redrawn
    assert {
        "eq.coef.log_theta",
        "eq.coef.log_nu",
        "glob.coef.log_zeta",
        "glob.coef.log_tau",
    } <= moved
    assert {"eq.mean.delta", "eq.mean.dev_log"} <= moved
    assert not any(p.startswith(("eq.vol", "eq.struct")) for p in moved)
    ctx = next(c for c in calls if c[0] == "sample")[1]
    np.testing.assert_allclose(
        np.asarray(ctx.eq_out.e), np.asarray(LJ.residuals(st, data, spec.prior, spec.static))
    )
    # the refresh is a draw from p(d | the new scales)
    bumped = jax.tree.map(lambda x: x + 1.0 if jnp.issubdtype(x.dtype, jnp.floating) else x, st)
    rkey = step_key(random.key(2), jnp.uint32(4), Block.EQ_COEF_LOCAL)
    ref = S.gl_coefficient_draw(rkey, J.hold_frozen(bumped, st, spec.live), spec.prior, spec.static)
    np.testing.assert_array_equal(np.asarray(new.eq.mean.delta), np.asarray(ref.eq.mean.delta))
    np.testing.assert_array_equal(np.asarray(new.eq.mean.dev_log), np.asarray(ref.eq.mean.dev_log))


def test_common_sv_only_runs_the_common_step_alone(monkeypatch):
    calls: list = []
    monkeypatch.setattr(J, "production_program", _fake_program(calls))
    spec = J.joint_spec("common_sv_only")
    spec, st, data = a_dataset(spec.name)
    new = spec.step(random.key(2), jnp.uint32(0), st, data)
    moved = _moved(st, new)
    assert moved and all(p.startswith("glob.vol") for p in moved)
    ctx = next(c for c in calls if c[0] == "sample")[1]
    lv = np.asarray(ctx.eq_out.log_lam)
    expect = np.asarray(st.eq.vol.core.log_lam)[:, None] + np.asarray(st.glob.vol.h)[None, :]
    np.testing.assert_allclose(lv, expect, rtol=1e-15)


def test_p3_steps_wait_for_the_kernels():
    """Building a P3 spec never composes; stepping composes and, until WP31/WP32 land, reports
    the missing kernels (the GL kinds through the registry; CommonSV through the program's missing
    G_COMMON_SV step, which compose itself does not detect)."""
    for name in ("structural_constant_hs", "structural_commonsv", "conjugate_commonsv_parity",
                 "common_sv_only"):  # fmt: skip
        spec, st, data = a_dataset(name)
        try:
            spec.step(random.key(2), jnp.uint32(0), st, data)
        except NotImplementedError as err:
            assert "WP31" in str(err) or "WP32" in str(err), err
        else:  # pragma: no cover - once the kernels exist this is the production sweep
            pytest.skip("Phase 3 kernels available")


# --------------------------------------------------------------------------- 7. explosive mass


def test_explosive_mass_is_recorded_for_every_lagged_row():
    assert set(EXPLOSIVE_MASS_P3) == set(J.LAGGED_P3)
    over = {}
    for name, by_size in EXPLOSIVE_MASS_P3.items():
        assert set(by_size) == {"small", "full"}
        for size, count in by_size.items():
            if count / EXPLOSIVE_DRAWS > J.EXPLOSIVE_BOUND:
                over[f"{name}/{size}"] = count / EXPLOSIVE_DRAWS
    assert not over, f"lagged rows above the explosive-mass bound {J.EXPLOSIVE_BOUND}: {over}"


def test_spectral_radius_and_reduced_form():
    B = np.zeros((5, 2))  # n = 2, p = 2, intercept
    B[1:3] = np.array([[0.5, 0.0], [0.0, 0.2]]).T
    B[3:5] = np.array([[0.3, 0.0], [0.0, 0.0]]).T
    rho = J.spectral_radius(B, 2, 2, 1)
    lam = (0.5 + math.sqrt(0.25 + 1.2)) / 2  # x² = 0.5x + 0.3
    assert math.isclose(float(rho), lam, rel_tol=1e-12)
    spec, st = a_draw("structural_commonsv", "full")
    Bj = np.asarray(J.reduced_form_coefficients(st, spec.prior, spec.static))
    host = _np(st)
    n = 3
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = host.eq.struct.a[i, :i]
    Gam = host.eq.mean.delta + host.eq.struct.a @ np.asarray(spec.prior.shared.Mc).T
    np.testing.assert_allclose(Bj, np.linalg.solve(A, Gam).T, rtol=1e-12, atol=1e-14)


def test_explosive_mass_measurement_reproduces_the_record():
    """A fresh 2·10⁵-draw measurement (the first draws of the recorded stream) is consistent with
    the recorded mass (binomial, 5 s.d.) for a conjugate row."""
    name = "conjugate_commonsv_block7"
    spec = J.joint_spec(name, "full")
    c, d = J.explosive_mass(spec, 200_000, seed=0, chunk=100_000)
    p = EXPLOSIVE_MASS_P3[name]["full"] / EXPLOSIVE_DRAWS
    assert abs(c - d * p) <= 5.0 * math.sqrt(d * p) + 1.0, (c, d * p)


@pytest.mark.slow
@pytest.mark.parametrize("name", J.LAGGED_P3)
def test_explosive_mass_full_measurement(name):
    """The recorded measurement itself (10⁶ draws per size, seed 0; weekly tier)."""
    for size in ("small", "full"):
        c, d = J.explosive_mass(J.joint_spec(name, size), EXPLOSIVE_DRAWS, seed=0, chunk=50_000)
        assert abs(c - EXPLOSIVE_MASS_P3[name][size]) <= 2, (name, size, c)
        assert c / d <= J.EXPLOSIVE_BOUND, (name, size, c / d)


# --------------------------------------------------------------------------- 8. review references
# (design review M12, M12(b), m25: value references for WP31's builders and WP34's A10)


def test_gl_log_r_reference_values_and_conventions():
    """D-37 by hand at n = 3, p = 2, and against the Phase 2 Minnesota builders: r_j is the Mode I
    (structural, ι = i + 1) or reduced lag variance at κ = 1, cross = 1 times l^α (no lag decay)."""
    from rai_bvar.priors.minnesota import Minnesota
    from rai_bvar.priors.reduced import reduced_arrays
    from rai_bvar.priors.scales import WindowScales, regressor_layout
    from rai_bvar.priors.structural import structural_arrays

    psi = np.array([0.5, 1.0, 2.0])
    n, p, nd = 3, 2, 1
    s = R.gl_log_r_reference(psi, n, p, nd, "structural")
    red = R.gl_log_r_reference(psi, n, p, nd, "reduced")
    assert s.shape == red.shape == (3, 7)
    assert np.all(s[:, 0] == 0.0) and np.all(red[:, 0] == 0.0)
    # equation 2 (ι = 3), lag 2 of variable 0 (slot 1 + 3 + 0 = 4): log(2/(3·0.5))
    assert s[2, 4] == math.log(2.0) - math.log(0.5) - math.log(3.0)
    assert red[2, 4] == math.log(2.0) - math.log(0.5)
    assert s[0, 1] == 0.0 and red[1, 2] == 0.0  # own lags: ψ_i/ψ_i (ι = 1 for equation 0)
    np.testing.assert_array_equal(R.gl_log_r_reference(psi, n, p, nd, "structural", "none"), 0.0)
    np.testing.assert_array_equal(R.gl_log_r_reference(psi, n, p, nd, "linear_reduced"), red)
    scales = WindowScales(psi=psi, own_means=np.zeros(n), freq="Q")
    lag_of, _ = regressor_layout(n, p, nd)
    for form, ref in (("structural", s), ("reduced", red)):
        prior = Minnesota(lambda_=1.0, alpha=2.0, cross=1.0, form=form, hierarchy="fixed")
        if form == "structural":
            coef, _ = structural_arrays(scales, prior, {}, n=n, lags=p, n_det=nd, mode="I")
        else:
            coef, _ = reduced_arrays(scales, prior, {}, n=n, lags=p, n_det=nd)
        var = 1.0 / np.asarray(coef.prec0)
        np.testing.assert_allclose(np.log(var[:, nd:] * lag_of[nd:] ** 2), ref[:, nd:], atol=1e-14)
    np.testing.assert_array_equal(J.gl_log_r(psi, n, p, nd, "linear_structural"), s)
    with pytest.raises(ValueError, match="positive"):
        R.gl_log_r_reference(np.array([1.0, -1.0, 1.0]), n, p, nd, "structural")
    with pytest.raises(ValueError, match="form"):
        R.gl_log_r_reference(psi, n, p, nd, "conjugate")


def test_ssvs_shat_reference():
    """D-44 / §3.6: ŝ² = RSS/(T − q_i − 2)·[(FᵀF)⁻¹]_jj (structural, D3 δ-design) and
    Ŝ_jj/(T − k − n − 1)·[(XᵀX)⁻¹]_rr (reduced), against a statsmodels-free OLS written with lstsq
    and the textbook covariance; divisors ≤ 0 raise."""
    rng = np.random.default_rng(3)
    T, n, k = 30, 3, 4
    X = np.hstack([np.ones((T, 1)), rng.standard_normal((T, k - 1))])
    Y = X @ rng.standard_normal((k, n)) * 0.3 + rng.standard_normal((T, n))
    Mc = np.zeros((k, n - 1))
    Mc[1, 0], Mc[2, 1] = 0.4, 0.4
    got = R.ssvs_shat_reference(Y, X, "structural", Mc=Mc)
    for i in range(n):
        F = np.hstack([X, X @ Mc[:, :i] - Y[:, :i]])
        _, rss, *_ = np.linalg.lstsq(F, Y[:, i], rcond=None)
        cov = float(rss[0]) / (T - F.shape[1] - 2) * np.linalg.pinv(F.T @ F)
        np.testing.assert_allclose(got[i], np.sqrt(np.diag(cov)[:k]), rtol=1e-10)
    # the δ-coordinates matter: without the D3 columns the a-columns differ, and so does ŝ(δ)
    assert not np.allclose(got[1:], R.ssvs_shat_reference(Y, X, "structural", Mc=0 * Mc)[1:])
    red = R.ssvs_shat_reference(Y, X, "reduced")
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    for j in range(n):
        v = (E[:, j] @ E[:, j]) / (T - k - n - 1) * np.diag(np.linalg.pinv(X.T @ X))
        np.testing.assert_allclose(red[j], np.sqrt(v), rtol=1e-10)
    with pytest.raises(ValueError, match="T_eff > q_i \\+ 2"):
        R.ssvs_shat_reference(Y[:8], X[:8], "structural", Mc=Mc)  # 8 − (4 + 2) − 2 = 0
    with pytest.raises(ValueError, match="k \\+ n \\+ 1"):
        R.ssvs_shat_reference(Y[:8], X[:8], "reduced")
    with pytest.raises(ValueError, match="Mc"):
        R.ssvs_shat_reference(Y, X, "structural")


def _cf_problem(seed: int = 0, n: int = 3, p: int = 2, H: int = 4):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, n, n)) * 0.25
    L = np.tril(rng.standard_normal((H, n, n)) * 0.3)
    idx = np.arange(n)
    L[:, idx, idx] = np.abs(L[:, idx, idx]) + 0.2  # positive diagonal; varies with the horizon
    B = np.concatenate([rng.standard_normal((1, n)), *[A[lag].T for lag in range(p)]])
    hist = rng.standard_normal((p, n))
    mu = R.zero_shock_path(B, hist, H, n_det=1, t_last=10.0)
    return A, L, B, hist, mu


def test_conditional_forecast_reference_companion_and_schur():
    """A10's oracle: M from the companion powers reproduces the recursion ``y = Bᵀx + L_hε_h``
    exactly; E_c, Cov_c equal the precision-form conditioning; conditioned cells are r; log w is
    SciPy's multivariate normal; Y_c from the same ε is the exact draw (linear in ε with
    covariance Cov_c). A conditioned variable is ordered before a free one."""
    A, L, B, hist, mu = _cf_problem()
    H, n = L.shape[:2]
    eps = np.random.default_rng(9).standard_normal((H, n))
    conds = [(1, 0, 0.7), (2, 2, -0.3), (4, 1, 1.1)]  # variable 0 conditioned, 1 and 2 free at h=1
    ref = R.conditional_forecast_reference(A, L, mu, conds, eps=eps)
    # the stacked companion form equals the recursion with the same shocks
    window, path = list(hist), []
    for h in range(H):
        x = np.concatenate([[1.0], *[window[-lag] for lag in range(1, A.shape[0] + 1)]])
        y = B.T @ x + L[h] @ eps[h]
        path.append(y)
        window.append(y)
    np.testing.assert_allclose(
        mu.reshape(-1) + ref.M @ eps.reshape(-1), np.ravel(path), rtol=1e-12, atol=1e-12
    )
    # precision form (an independent algebra of the same Gaussian conditioning)
    Sig, m = ref.uncond_cov, mu.reshape(-1)
    c = ref.index
    f = np.setdiff1d(np.arange(n * H), c)
    P = np.linalg.inv(Sig)
    cov_f = np.linalg.inv(P[np.ix_(f, f)])
    mean_f = m[f] - cov_f @ P[np.ix_(f, c)] @ (ref.values - m[c])
    np.testing.assert_allclose(ref.mean.reshape(-1)[f], mean_f, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ref.cov[np.ix_(f, f)], cov_f, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ref.mean.reshape(-1)[c], ref.values, rtol=0, atol=1e-12)
    assert np.max(np.abs(ref.cov[c])) < 1e-12 and np.max(np.abs(ref.cov[:, c])) < 1e-12
    lw = stats.multivariate_normal(m[c], Sig[np.ix_(c, c)]).logpdf(ref.values)
    assert math.isclose(ref.log_weight, float(lw), rel_tol=1e-10)
    # Y_c: conditioned cells are r, and Y_c − E_c = (I − K R) M ε with covariance Cov_c
    np.testing.assert_allclose(ref.path.reshape(-1)[c], ref.values, atol=1e-12)
    K = Sig[:, c] @ np.linalg.inv(Sig[np.ix_(c, c)])
    G = (np.eye(n * H) - K @ np.eye(n * H)[c]) @ ref.M
    np.testing.assert_allclose(
        ref.path.reshape(-1) - ref.mean.reshape(-1), G @ eps.reshape(-1), atol=1e-12
    )
    np.testing.assert_allclose(G @ G.T, ref.cov, atol=1e-12)
    # Monte Carlo of Y_c over fresh shocks (|z| ≤ 4 per free cell, D-60)
    many = np.random.default_rng(1).standard_normal((400, H, n))
    draws_ = np.stack(
        [R.conditional_forecast_reference(A, L, mu, conds, eps=e).path.reshape(-1) for e in many]
    )
    z = (draws_[:, f].mean(0) - ref.mean.reshape(-1)[f]) / np.sqrt(np.diag(ref.cov)[f] / 400)
    assert np.max(np.abs(z)) < 4.0, z


def test_conditional_forecast_reference_errors_and_impulses():
    A, L, _, _, mu = _cf_problem(n=2, p=1, H=3)
    Phi = R.companion_impulse(A, 3)
    np.testing.assert_allclose(Phi[0], np.eye(2))
    np.testing.assert_allclose(Phi[2], A[0] @ A[0], rtol=1e-14)
    for bad, msg in (([(0, 0, 1.0)], "outside"), ([(1, 2, 1.0)], "outside"),
                     ([(1, 0, 1.0), (1, 0, 2.0)], "twice"), ([(1, 0, math.nan)], "finite"),
                     ([], "at least one"),
                     ([(h, j, 0.0) for h in (1, 2, 3) for j in (0, 1)], "free cell")):  # fmt: skip
        with pytest.raises(ValueError, match=msg):
            R.conditional_forecast_reference(A, L, mu, bad)
