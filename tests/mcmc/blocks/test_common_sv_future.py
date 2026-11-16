"""``CommonSV``'s future step, tested directly (P3 D-63, E182; the merge review of WP35, minor 3).

The lagged CommonSV rows of the SBC rank ``y_{T+1}`` but not ``h_{T+1}``, and ranking it would
change the registered rows (D-63). So the step ``h_{T+j} = ρh_{T+j−1} + σ_hη_j`` of
``CommonSVStep.future`` is checked here in the PR tier, at fixed ``ρ = 0.83``, ``σ_h = 0.37`` and
``h_T = 0.61`` (an in-sample path whose mean is not ``h_T``, padded rows non-zero), on rows 320,
321 and 322 (the conjugate shared-factor, structural Mode I and reduced forms):

* **exact** (1e-12): the innovations ``η_j`` recovered from the key (``fold_in(glob_key,
  G_COMMON_SV)``, then ``Sub.CSV_H``) give the recursion from ``h_T`` computed here; the draw's
  ``h_T`` is the in-sample ``h[T_eff − 1]``; ``CommonSVStep.future`` and the forecast core's
  ``log λ^eff_{T+j} − log λ_i`` both equal the recursion; and every ``y_{T+j}`` uses that
  ``h_{T+j}``: ``cond_chol·cond_cholᵀ = A⁻¹ diag(λ_i e^{h_{T+j}}) A⁻ᵀ`` with ``A`` from the state
  (relative 1e-12), and ``paths − cond_mean = cond_chol·ε``;
* **distributional** (20,000 draws, seed 20261028): ``h_{T+1}`` against ``N(ρh_T, σ_h²)`` and
  ``h_{T+2}`` against ``N(ρ²h_T, σ_h²(1 + ρ²))``, mean and variance ``|z| ≤ 4`` (four moments) and
  KS ``min p > 1e-3/m`` with ``m = 2`` (D-60).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.forecast import ForecastShocks, forecast_core
from rai_bvar.engines.mcmc.keys import Block, Sub, sub_key
from rai_bvar.engines.mcmc.runner import extract_params
from rai_bvar.validation.joint import composition_id, joint_spec
from rai_bvar.validation.simulate import unit_lower

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

ROWS = ("conjugate_commonsv_dummies", "structural_commonsv", "reduced_commonsv")
RHO, SIGMA, H_T, H = 0.83, 0.37, 0.61, 4
N_DIST, SEED = 20_000, 20261028
_F64 = jnp.float64


def _setup(name: str):
    """Row ``name``'s spec, program and a prior state with the fixed ``(h, ρ, σ_h)`` of the module
    docstring, with its stored parameters and data."""
    spec = joint_spec(name, "small")
    program = (
        compose(spec.static, overrides=dict(spec.overrides))
        if spec.overrides
        else compose(spec.static)
    )
    key = random.key(SEED)
    st = spec.prior_sample(random.fold_in(key, 0))
    data = spec.simulate(random.fold_in(key, 1), st)
    T, T_pad = int(data.T_eff), int(st.glob.vol.h.shape[0])
    assert T_pad > T, name  # padded rows exist, so h[T_eff − 1] differs from h[-1]
    h = np.linspace(-0.4, 0.3, T_pad)
    h[T - 1] = H_T
    h[T:] = 1.7  # never a state of the T-row model: a draw that read it would be wrong
    assert abs(h[:T].mean() - H_T) > 0.5
    gv = dataclasses.replace(
        st.glob.vol,
        h=jnp.asarray(h, _F64),
        rho=jnp.asarray(RHO, _F64),
        log_sigma=jnp.asarray(math.log(SIGMA), _F64),
    )
    st = dataclasses.replace(st, glob=dataclasses.replace(st.glob, vol=gv))
    params = extract_params(program, st, data, spec.prior)
    step = next(g for g in program.global_steps if g.block == Block.G_COMMON_SV)
    return program, st, data, params, step


def _recursion(eta: np.ndarray) -> np.ndarray:
    """``h_{T+j} = ρh_{T+j−1} + σ_hη_j`` from ``h_T``, in NumPy."""
    out, prev = np.empty(eta.shape[0]), H_T
    for j, e in enumerate(eta):
        prev = RHO * prev + SIGMA * e
        out[j] = prev
    return out


@pytest.mark.parametrize("name", ROWS, ids=lambda nm: f"{composition_id(nm)}-{nm}")
def test_future_step_is_the_ar1_recursion_and_y_uses_it(name):
    """D-63 (1): the exact checks of the module docstring."""
    program, st, data, params, step = _setup(name)
    n = program.static.shapes.n
    cd = params.glob.vol
    assert float(cd.h_T) == H_T  # h[T_eff − 1]: not the path's mean, not a padded row
    np.testing.assert_allclose([float(cd.rho), float(cd.sigma)], [RHO, SIGMA], rtol=1e-15)

    k_glob, k_vol, k_mean, k_eps = random.split(random.key(SEED + 1), 4)
    shocks = ForecastShocks(
        eps=random.normal(k_eps, (H, n), dtype=_F64),
        vol_keys=random.split(k_vol, n),
        mean_keys=random.split(k_mean, n),
        glob_key=k_glob,
    )
    k_csv = random.fold_in(k_glob, int(Block.G_COMMON_SV))
    eta = np.asarray(random.normal(sub_key(k_csv, Sub.CSV_H), (H,), dtype=_F64))
    want = _recursion(eta)
    np.testing.assert_allclose(np.asarray(step.future(k_csv, params.glob, H)), want, atol=1e-12)

    out = forecast_core(params, data.hist, data.t_last, shocks, program)
    log_lam = np.asarray(st.eq.vol.core.log_lam)
    common = 2.0 * np.log(np.asarray(out.cond_scale)) - log_lam[None, :]  # (H, n)
    np.testing.assert_allclose(common, np.repeat(want[:, None], n, axis=1), atol=1e-12)
    a_inv = np.linalg.inv(np.asarray(unit_lower(st.eq.struct.a)))
    chol = np.asarray(out.cond_chol)
    for j in range(H):
        cov = a_inv @ np.diag(np.exp(log_lam + want[j])) @ a_inv.T
        np.testing.assert_allclose(
            chol[j] @ chol[j].T, cov, rtol=1e-12, atol=1e-12 * np.abs(cov).max()
        )
    shock = np.einsum("hij,hj->hi", chol, np.asarray(shocks.eps))
    np.testing.assert_allclose(np.asarray(out.paths) - np.asarray(out.cond_mean), shock, atol=1e-12)


def test_future_step_has_the_ar1_law():
    """D-63 (2): the distributional checks of the module docstring (row 321's step)."""
    _, _, _, params, step = _setup("structural_commonsv")
    keys = random.split(random.key(SEED), N_DIST)
    draws = np.asarray(jax.jit(jax.vmap(lambda k: step.future(k, params.glob, 2)))(keys))
    laws = (
        (draws[:, 0], RHO * H_T, SIGMA**2),
        (draws[:, 1], RHO**2 * H_T, SIGMA**2 * (1.0 + RHO**2)),
    )
    z, pvals = [], []
    for x, m, v in laws:
        z.append((x.mean() - m) / math.sqrt(v / N_DIST))
        z.append((x.var(ddof=1) - v) / (v * math.sqrt(2.0 / (N_DIST - 1))))
        pvals.append(stats.kstest(x, stats.norm(m, math.sqrt(v)).cdf).pvalue)
    assert max(abs(v) for v in z) <= 4.0, z
    assert min(pvals) > 1e-3 / len(pvals), pvals
