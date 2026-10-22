"""Per-window scales (DESIGN §6, §6.6; R9): ψ and own-lag means equal the Phase 1 conjugate
problem's, explicit and EB overrides, validation, the shared layout helpers, and the R9 property
that every prior array built from ``data.upto(origin)`` ignores later observations."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import rai_bvar as rb
from rai_bvar.components import SV
from rai_bvar.data import VARData
from rai_bvar.data.design import regressor_names
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.priors.reduced import reduced_arrays
from rai_bvar.priors.scales import (
    WindowScales,
    coefficient_groups,
    conjugate_omega,
    lambda_alpha,
    prior_mean_matrix,
    regressor_layout,
    window_scales,
)
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.priors.sv_arrays import constant_prior, sv_prior_arrays, transitory_arrays

from .helpers import make_data, make_frame


@pytest.mark.parametrize("lags", [1, 2, 4])
def test_psi_and_means_equal_the_phase1_conjugate_problem(data4, lags):
    prior = rb.Minnesota()
    scales = window_scales(data4, prior, lags, "c")
    problem = ConjugateProblem(data4, prior, lags, "c", None)
    np.testing.assert_array_equal(scales.psi, problem.default["psi"])  # same helper, same ψ
    np.testing.assert_array_equal(scales.own_means, problem.means)
    np.testing.assert_array_equal(scales.own_means, [1.0, 0.0, 0.0, 1.0])  # tcodes 1, 5, 2, 4
    assert scales.freq == data4.freq == "Q-DEC"
    assert scales.n == 4
    np.testing.assert_array_equal(scales.iota, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(scales.sigma2_bar, scales.psi / np.arange(1, 5), rtol=0)
    # distinct variable scales: ψ grows by about 10 per two variables (label mix-ups are visible)
    assert scales.psi[2] > 10 * scales.psi[0]


def test_psi_sources_explicit_override_and_mean_settings(data4):
    explicit = (1.0, 2.0, 3.0, 4.0)
    prior = rb.Minnesota(psi=explicit, mean={"v1": 0.5})
    scales = window_scales(data4, prior, 2, "c")
    np.testing.assert_array_equal(scales.psi, explicit)
    np.testing.assert_array_equal(scales.own_means, [1.0, 0.5, 0.0, 1.0])
    eb = np.array([0.5, 0.6, 0.7, 0.8])
    over = window_scales(data4, prior, 2, "c", psi=eb)
    np.testing.assert_array_equal(over.psi, eb)
    eb[0] = 99.0  # the scales hold a copy
    assert over.psi[0] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        over.psi[0] = 1.0
    assert scales.notes == ()


def test_short_windows_record_the_phase1_notes():
    data = make_data(2, 14, seed=3)
    scales = window_scales(data, rb.Minnesota(), 2, "c")
    assert any("fewer than 10 degrees of freedom" in note for note in scales.notes)


def test_window_scales_validation(data4):
    prior = rb.Minnesota()
    with pytest.raises(ValueError, match="more than 4 observations"):
        window_scales(data4.tail(4), prior, 4, "c")
    with pytest.raises(ValueError, match="deterministic"):
        window_scales(data4, prior, 2, "q")
    with pytest.raises(ValueError, match="lags"):
        window_scales(data4, prior, 0, "c")
    with pytest.raises(TypeError, match="Minnesota"):
        window_scales(data4, SV(), 2, "c")
    with pytest.raises(ValueError, match="4 positive finite"):
        window_scales(data4, prior, 2, "c", psi=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="4 positive finite"):
        window_scales(data4, prior, 2, "c", psi=np.array([1.0, 2.0, -3.0, 1.0]))
    with pytest.raises(ValueError, match="needs 4 positive"):
        window_scales(data4, rb.Minnesota(psi=(1.0, 2.0)), 2, "c")
    frame = make_frame(2, 40, seed=4)
    frame["v0"] = 3.0  # constant: the Phase 1 AR(p) helper refuses it
    with pytest.raises(ValueError, match="constant"):
        window_scales(VARData.from_frame(frame, 1), prior, 2, "c")
    with pytest.raises(ValueError, match="positive"):
        WindowScales(psi=np.array([1.0, 0.0]), own_means=np.zeros(2), freq="Q")
    with pytest.raises(ValueError, match="own_means"):
        WindowScales(psi=np.ones(2), own_means=np.zeros(3), freq="Q")


def test_import_loads_no_jax():
    # The scales are pure NumPy; only the array builders import the WP1 pytree classes.
    code = """
        import sys
        import rai_bvar.priors.scales
        assert not [m for m in sys.modules if m.split(".")[0] == "jax"], "jax imported"
        print("ok")
    """
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


# --------------------------------------------------------------------------- layout helpers


@pytest.mark.parametrize("deterministic", ["n", "c", "ct"])
def test_layout_matches_the_design_matrix_names(deterministic):
    variables, lags = ("x", "y", "z"), 3
    names = regressor_names(variables, lags, deterministic)
    n_det = {"n": 0, "c": 1, "ct": 2}[deterministic]
    lag_of, var_of = regressor_layout(3, lags, n_det)
    for r, name in enumerate(names):
        if r < n_det:
            assert (lag_of[r], var_of[r]) == (0.0, -1)
        else:
            var, lag = name.split(".L")
            assert (lag_of[r], var_of[r]) == (float(lag), variables.index(var))
    groups = coefficient_groups(3, lags, n_det)
    assert groups.dtype == np.int32 and groups.shape == (3, n_det + 9)
    for i in range(3):
        for r in range(n_det + 9):
            expected = 0 if var_of[r] < 0 else (1 if var_of[r] == i else 2)
            assert groups[i, r] == expected


def test_prior_mean_matrix_and_omega_shapes(data4):
    prior = rb.Minnesota(cross=0.5, form="structural")
    scales = window_scales(data4, prior, 2, "ct")
    M = prior_mean_matrix(scales, 4, 2, 2)
    expected = np.zeros((10, 4))
    expected[2 + np.arange(4), np.arange(4)] = scales.own_means
    np.testing.assert_array_equal(M, expected)
    omega = conjugate_omega(scales, prior, 0.3, 1.5, 4, 2, 2)
    assert omega.shape == (4, 10)
    np.testing.assert_array_equal(omega[:, :2], prior.var_const)
    # equation 0, lag 2 of variable 1 (cross): κ² κ_c² (d−n−1)/(l^α ψ_1)
    np.testing.assert_allclose(omega[0, 2 + 4 + 1], 0.3**2 * 0.25 / (2**1.5 * scales.psi[1]))
    with pytest.raises(ValueError, match="4 variables"):
        prior_mean_matrix(scales, 3, 2, 2)


def test_lambda_alpha_reads_eb_values_and_rejects_inconsistencies(data4):
    prior = rb.Minnesota(lambda_=0.2, alpha=2.0)
    scales = window_scales(data4, prior, 2, "c")
    assert lambda_alpha(prior, {}, scales) == (0.2, 2.0)
    phase1 = {"lambda": 0.35, "alpha": 1.5, "soc": None, "sur": None, "psi": scales.psi.copy()}
    assert lambda_alpha(prior, phase1, scales) == (0.35, 1.5)
    assert lambda_alpha(prior, {"lambda": np.float64(0.1), "covid": None}, scales) == (0.1, 2.0)
    with pytest.raises(KeyError, match="unknown hyperparameters"):
        lambda_alpha(prior, {"lamda": 0.1}, scales)
    with pytest.raises(ValueError, match="psi"):
        lambda_alpha(prior, {"psi": 2.0 * scales.psi}, scales)
    with pytest.raises(ValueError, match="positive"):
        lambda_alpha(prior, {"lambda": -0.1}, scales)
    with pytest.raises(TypeError, match="number"):
        lambda_alpha(prior, {"alpha": "2"}, scales)


# --------------------------------------------------------------------------- R9


def _all_arrays(data: VARData, lags: int = 2) -> list[np.ndarray]:
    prior = rb.Minnesota(form="structural", cross=0.5, contemp=2.0)
    scales = window_scales(data, prior, lags, "c")
    sv = SV(outliers=True, tails="t")
    trees = [
        scales,
        structural_arrays(scales, prior, {"lambda": 0.3}, n=4, lags=lags, n_det=1, mode="I"),
        structural_arrays(scales, prior, {}, n=4, lags=lags, n_det=1, mode="II"),
        contemp_arrays(scales, prior, 4, scaling="none"),
        contemp_arrays(scales, prior, 4, scaling="lambda"),
        reduced_arrays(scales, prior, {}, n=4, lags=lags, n_det=1),
        constant_prior(scales, 4),
        sv_prior_arrays(sv, scales),
        transitory_arrays(sv, scales, sv.nu_prior.support),
    ]
    import jax  # the arrays are WP1 pytrees; flatten them to compare leaf by leaf

    leaves = []
    for tree in trees:
        if isinstance(tree, WindowScales):
            leaves += [tree.psi, tree.own_means, np.array([tree.freq])]
        else:
            leaves += [np.asarray(x) for x in jax.tree.leaves(tree)]
    return leaves


def test_r9_arrays_from_a_window_ignore_later_data(data4):
    origin = data4.index[80]
    frame = data4.raw.copy()
    later = frame.index > origin
    frame.loc[later] = frame.loc[later] * 1.7 + 0.3  # rewrite everything after the origin
    changed = VARData.from_frame(frame, data4.tcodes)
    before = _all_arrays(data4.upto(origin))
    after = _all_arrays(changed.upto(origin))
    assert len(before) == len(after) > 20
    for x, y in zip(before, after, strict=True):
        np.testing.assert_array_equal(x, y)
    # sanity: once the later data are inside the window, ψ does change
    full = window_scales(changed, rb.Minnesota(), 2, "c").psi
    assert not np.allclose(full, window_scales(data4, rb.Minnesota(), 2, "c").psi)


def test_scales_follow_the_window_frequency(data4_monthly):
    scales = window_scales(data4_monthly, rb.Minnesota(), 3, "c")
    assert scales.freq == "M"
    annual = VARData.from_frame(
        make_frame(2, 60, "Y", seed=5), {"v0": 1, "v1": 5}
    )  # frequency only matters for the outlier prior (test_sv_arrays)
    assert window_scales(annual, rb.Minnesota(), 1, "c").freq.startswith("Y")
