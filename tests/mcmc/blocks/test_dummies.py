"""SOC/SUR dummy rows on the MCMC engine (DESIGN §13 WP25; N05 §7.7; N04 §3.1).

Tier U (DESIGN §9.3):

* **builders** — the MCMC rows are the Phase 1 rows bit for bit (``ConjugateProblem.dummies`` and
  ``structural_nig._Window.dummies``), with the weights of DESIGN §6.1/§6.2 (Mode I and
  ``iw_equivalent`` ``ι/ψ_i``, Mode II 1, reduced ``1/ψ_j``) and the EB values in use;
* **dense oracles** (item 1) — the structural conditional of ``θ_i = [δ_i | a_i]`` at a fixed
  ``λ_i`` against the dense stacked rows of the Phase 1 twin (``_Window.equations``: data rows,
  dummy rows, Mode II prior), on the weighted and constant-weight paths; the ``fixed_factor`` cache
  against a dense solve; the CCCM ``π_j`` conditional against the system-wide SUR conditional;
  the Mode II coupling against its hand formula;
* **density-ratio identities** (item 2; Grosse & Duvenaud eq. 1) of the equation step, the
  coupled ``λ`` step and the CCCM step against the independent ``validation.log_joint`` with the
  rows (``validation.dummies``: covariance form, Bayes' rule), at 20 random states, 1e-8;
* **closed path** — without dummies no dummy code is traced (every dummy helper raises) and the
  cache and prior trees carry ``Empty`` (the bitwise invariance of every pre-WP25 composition; the
  cross-commit comparison is in the WP25 report);
* **monitor** — the rows change ``log_prior`` (by their factor) and never the log-likelihood.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import rai_bvar as rb
from rai_bvar.components.mean import linear as linear_mod
from rai_bvar.components.mean import linear_reduced as reduced_mod
from rai_bvar.components.mean.linear_reduced import cccm_conditional_j, cccm_core, unit_lower
from rai_bvar.engines.analytic import ConjugateProblem
from rai_bvar.engines.mcmc import data as data_mod
from rai_bvar.engines.mcmc import runner
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.config import MeanStatic, validate_composition
from rai_bvar.engines.mcmc.linalg import precision_draw
from rai_bvar.engines.mcmc.monitor import monitor_values
from rai_bvar.engines.mcmc.resolve import resolve
from rai_bvar.engines.mcmc.state import (
    CCCMGlobal,
    ConstantEq,
    DummyCache,
    DummyRows,
    Empty,
    LinearEq,
    RecursiveEq,
    prior_template,
    state_template,
)
from rai_bvar.engines.structural_nig import _Window
from rai_bvar.priors.dummies import dummy_arrays, dummy_rows, dummy_weights, soc_rows
from rai_bvar.priors.scales import window_scales
from rai_bvar.validation import log_joint as lj_mod
from rai_bvar.validation import reference as ref_mod
from rai_bvar.validation import simulate as sim_mod
from rai_bvar.validation import testfns as tf_mod
from rai_bvar.validation.identities import density_ratio_error
from rai_bvar.validation.joint import (
    COMPOSITIONS,
    JointSpec,
    composition_id,
    joint_spec,
    production_sweep,
)
from rai_bvar.validation.log_joint import log_joint, residuals

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261025, impl="threefry2x32")
PRE_WP25 = tuple(n for n in COMPOSITIONS if composition_id(n) < 200)  # DESIGN §9.5 rows 1–17
DUMMY_SPECS = (
    "structural_constant_dummies",
    "structural_mode2_constant_dummies",
    "reduced_constant_dummies",
    "structural_sv_dummies",
)


def eq_slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def problem(spec: JointSpec, seed: int):
    k = random.fold_in(KEY, seed)
    st = spec.prior_sample(random.fold_in(k, 0))
    return st, spec.simulate(random.fold_in(k, 1), st)


def views(spec: JointSpec, st, data):
    """Program kernels, the production cache and the batched ``(ctx, pm, cm)`` of the sweep."""
    program = compose(spec.static)
    prepare, _ = production_sweep(spec.static)
    cache = prepare(data, spec.prior)
    S, M, C = program.structure, program.mean, program.coef
    ytil = S.pre_residual(st.glob, data, M.fitted_reduced(st.glob, data))

    def one(i, eq_i, prior_i):
        ctx = S.context(i, ytil, eq_i, program.vol, data, st.glob.vol)
        return (
            ctx,
            C.moments_eq(eq_i.coef, i, prior_i, st.glob.coef),
            S.contemp_moments(ctx, prior_i),
        )

    n = spec.static.shapes.n
    ctx, pm, cm = jax.vmap(one)(jnp.arange(n, dtype=jnp.int32), st.eq, spec.prior.eq)
    return program, cache, ctx, pm, cm


def var_data(n: int = 3, T: int = 70, seed: int = 11) -> rb.VARData:
    """A small stationary sample with non-zero levels (``ȳ₀`` far from 0) and tcode 1."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    A = np.diag(np.linspace(0.8, 0.4, n)) + 0.03 * rng.standard_normal((n, n))
    mean = np.linspace(2.0, -1.0, n)
    y, out = mean.copy(), np.empty((T, n))
    for t in range(T + 50):
        y = mean + A @ (y - mean) + 0.3 * rng.standard_normal(n)
        if t >= 50:
            out[t - 50] = y
    index = pd.period_range("1980Q1", periods=T, freq="Q")
    return rb.VARData.from_frame(
        pd.DataFrame(out, index=index, columns=[f"v{j}" for j in range(n)]), 1
    )


# =========================================================================== builders


BUILDER_CASES = {
    "soc+sur": dict(n=3, lags=2, det="c", soc=1.3, sur=0.7),
    "soc-only-all": dict(n=3, lags=1, det="c", soc=0.4, soc_scope="all", mean=0.0),
    "sur-trend": dict(n=2, lags=3, det="ct", sur=2.0),
    "soc-no-det": dict(n=4, lags=2, det="n", soc=1.1, mean={"v1": 0.0}),
    "n1": dict(n=1, lags=2, det="c", soc=0.9, sur=0.5),
}


@pytest.mark.parametrize("case", list(BUILDER_CASES), ids=list(BUILDER_CASES))
def test_rows_equal_the_phase1_rows_bitwise(case):
    """``priors.dummies`` reproduces ``ConjugateProblem.dummies`` and ``_Window.dummies`` bit for
    bit (N05 §7.7: "the MCMC rows are the same dummies")."""
    c = dict(BUILDER_CASES[case])
    n, lags, det = c.pop("n"), c.pop("lags"), c.pop("det")
    data = var_data(n=n, seed=3)
    prior = rb.Minnesota(**c)
    h = {"lambda": 0.2, "alpha": 2.0, "soc": prior.soc, "sur": prior.sur, "psi": np.ones(n)}
    Y1, X1 = ConjugateProblem(data, prior, lags, det, None).dummies(h)
    window = _Window(data, prior, lags, det)
    Y2, X2 = window.dummies(h)
    scales = window_scales(data, prior, lags, det)
    rows = soc_rows(scales.own_means, prior.soc_scope)
    n_det = {"n": 0, "c": 1, "ct": 2}[det]
    Y3, X3 = dummy_rows(
        data.values[:lags].mean(axis=0), rows, prior.soc, prior.sur, lags=lags, n_det=n_det
    )
    for a, b in ((Y1, Y3), (X1, X3), (Y2, Y3), (X2, X3)):
        assert a.shape == b.shape and a.tobytes() == b.tobytes()
    mode = "reduced"
    rows_arr, w = dummy_arrays(data.values, scales, prior, None, lags=lags, n_det=n_det, mode=mode)
    assert rows_arr.Y.tobytes() == Y3.tobytes() and rows_arr.X.tobytes() == X3.tobytes()
    np.testing.assert_array_equal(w, 1.0 / scales.psi)


def test_weights_hyper_and_errors():
    data = var_data()
    prior = rb.Minnesota(soc=1.0, sur=2.0, form="structural")
    scales = window_scales(data, prior, 2, "c")
    iota = np.arange(1.0, 4.0)
    np.testing.assert_array_equal(dummy_weights(scales, 3, "I"), 1.0 / (scales.psi / iota))
    np.testing.assert_array_equal(dummy_weights(scales, 3, "iw_equivalent"), iota / scales.psi)
    np.testing.assert_array_equal(dummy_weights(scales, 3, "II"), np.ones(3))
    np.testing.assert_array_equal(dummy_weights(scales, 3, "reduced"), 1.0 / scales.psi)
    rows, _ = dummy_arrays(data.values, scales, prior, {"soc": 4.0}, lags=2, n_det=1, mode="II")
    ybar0 = data.values[:2].mean(axis=0)
    np.testing.assert_array_equal(rows.Y[:3], np.diag(ybar0) / 4.0)  # EB value used
    np.testing.assert_array_equal(rows.Y[3], ybar0 / 2.0)
    off = rb.Minnesota(form="structural")
    assert dummy_arrays(data.values, scales, off, {"soc": 4.0}, lags=2, n_det=1, mode="I") == (
        Empty(),
        Empty(),
    )
    diff = rb.Minnesota(soc=1.0, mean=0.0, form="structural")
    with pytest.raises(ValueError, match="non-zero own-lag prior mean"):
        dummy_arrays(
            data.values, window_scales(data, diff, 2, "c"), diff, None, lags=2, n_det=1, mode="I"
        )


@pytest.mark.parametrize(
    ("vol", "form", "path", "mode"),
    [
        (rb.Constant("ig"), "structural", "cholesky_const", "I"),
        (rb.Constant("iw"), "structural", "fixed_factor", "II"),
        (rb.Constant("iw_equivalent"), "structural", "cholesky_const", "iw_equivalent"),
        (rb.SV(), "structural", "weighted", "I"),
        (rb.Constant("ig"), "reduced", "weighted", "reduced"),
    ],
    ids=["modeI", "modeII", "iw_eq", "sv", "reduced"],
)
def test_resolve_builds_the_rows(vol, form, path, mode):
    """``resolve``: ``n_dummy``, the path rule (Mode I → ``cholesky_const``), the arrays of the
    builders and ``hyper`` (``soc``, ``sur``, ``n_dummy``); no dummies → ``Empty`` everywhere."""
    data = var_data()
    prior = rb.Minnesota(form=form, soc=1.5, sur=0.8, hierarchy="fixed")
    model = rb.BVAR(mean=rb.Linear(prior=prior), vol=vol, lags=2)
    r = resolve(model, data, rb.Runtime())
    assert r.static.mean.n_dummy == 4 and r.static.mean.const_path == path
    scales = window_scales(data, prior, 2, "c")
    rows, w = dummy_arrays(data.values, scales, prior, None, lags=2, n_det=1, mode=mode)
    np.testing.assert_array_equal(r.prior.shared.dummies.Y, rows.Y)
    np.testing.assert_array_equal(r.prior.shared.dummies.X, rows.X)
    np.testing.assert_array_equal(r.prior.eq.coef.dummy_w, w)
    assert (r.hyper["soc"], r.hyper["sur"], r.hyper["n_dummy"]) == (1.5, 0.8, 4)
    assert jax.tree.structure(r.prior) == jax.tree.structure(prior_template(r.static))
    plain = rb.BVAR(mean=rb.Linear(prior=replace(prior, soc=None, sur=None)), vol=vol, lags=2)
    r0 = resolve(plain, data, rb.Runtime())
    assert r0.static.mean.n_dummy == 0 and "n_dummy" not in r0.hyper
    assert isinstance(r0.prior.shared.dummies, Empty) and isinstance(
        r0.prior.eq.coef.dummy_w, Empty
    )


def test_contract_and_legality():
    """Static legality (DESIGN §3.8 rule 6 with WP25) and the closed contract fields."""
    spec = joint_spec("structural_constant")
    assert spec.static.mean.n_dummy == 0
    tmpl = prior_template(spec.static)
    assert isinstance(tmpl.shared.dummies, Empty) and isinstance(tmpl.eq.coef.dummy_w, Empty)
    on = joint_spec("structural_constant_dummies")
    tmpl = prior_template(on.static)
    assert tmpl.shared.dummies.Y.shape == (3, 2) and tmpl.shared.dummies.X.shape == (3, 3)
    assert tmpl.eq.coef.dummy_w.shape == (2,)
    st = on.static
    with pytest.raises(ValueError, match="dense"):
        validate_composition(st.struct, replace(st.mean, const_path="eigen"), st.coef, st.vol)
    with pytest.raises(ValueError, match="zero mean"):
        MeanStatic(kind="zero", n_dummy=2)
    _, data = problem(spec, 0)
    with pytest.raises(ValueError, match="n_dummy is 0"):
        data_mod.base_cache(
            data, spec.prior.shared.Mc, spec.static, dummies=on.prior.shared.dummies,
            dummy_w=on.prior.eq.coef.dummy_w,
        )  # fmt: skip
    with pytest.raises(ValueError, match="needs the dummy rows"):
        data_mod.base_cache(data, on.prior.shared.Mc, on.static)


def test_dummy_cache_products():
    """``FitCache.dummy``: ``F_d = [X_d | X_d M_c − Y_d[:, :n−1]]`` (structural) or ``X_d``
    (reduced), ``G = F_dᵀF_d`` exactly symmetric, ``FtY = F_dᵀY_d``, the weights; no ``tmask``."""
    for name in ("structural_mode2_constant_dummies", "reduced_constant_dummies"):
        spec = joint_spec(name)
        st, data = problem(spec, 1)
        _, cache, *_ = views(spec, st, data)
        d, rows = cache.dummy, spec.prior.shared.dummies
        assert isinstance(d, DummyCache)
        Y, X = np.asarray(rows.Y), np.asarray(rows.X)
        n = Y.shape[1]
        if spec.static.mean.kind == "linear_structural":
            F = np.hstack([X, X @ np.asarray(spec.prior.shared.Mc) - Y[:, : n - 1]])
        else:
            F = X
        np.testing.assert_allclose(d.F, F, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(d.G, F.T @ F, rtol=1e-13, atol=1e-13)
        assert np.array_equal(np.asarray(d.G), np.asarray(d.G).T)
        np.testing.assert_allclose(d.FtY, F.T @ Y, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(d.w, spec.prior.eq.coef.dummy_w)


# =========================================================================== dense oracles


def _fixed_state(static, log_lam: np.ndarray):
    st = jax.tree.map(jnp.asarray, state_template(static))
    core = ConstantEq(log_lam=jnp.asarray(log_lam))
    return replace(st, eq=replace(st.eq, vol=replace(st.eq.vol, core=core)))


@pytest.mark.parametrize(
    ("vol", "path"),
    [
        (rb.Constant("iw"), "weighted"),
        (rb.Constant("iw"), "cholesky_const"),
        (rb.Constant("ig"), "weighted"),
        (rb.Constant("ig"), "cholesky_const"),
        (rb.Constant("iw_equivalent"), "cholesky_const"),
    ],
    ids=["II-weighted", "II-cholesky", "I-weighted", "I-cholesky", "iweq-cholesky"],
)
def test_structural_conditional_equals_the_phase1_stacked_rows(vol, path):
    """At a fixed ``λ`` the kernel's ``θ_i = [δ_i | a_i]`` conditional equals the dense system of
    the Phase 1 twin's stacked rows (``structural_nig._Window.equations``: data rows ``Z``, dummy
    rows ``Z_d``, the Mode II prior): Mode II ``K = (V₀⁻¹ + Z_dᵀZ_d + ZᵀZ)/λ_i``; Mode I and
    ``iw_equivalent`` the same rows with fixed weight ``ι/ψ_i`` on ``Z_d`` (1e-10)."""
    data = var_data()
    lags, det = 2, "c"
    prior = rb.Minnesota(form="structural", soc=1.3, sur=0.7, hierarchy="fixed", lambda_=0.3)
    model = rb.BVAR(mean=rb.Linear(prior=prior), vol=vol, lags=lags)
    r = resolve(model, data, rb.Runtime())
    static = dataclasses.replace(r.static, mean=replace(r.static.mean, const_path=path))
    window = _Window(data, prior, lags, det)
    h = window.complete({"lambda": 0.3, "soc": 1.3, "sur": 0.7})
    rows = list(window.equations(h))
    lam = np.array([0.7, 1.9, 0.4])
    st = _fixed_state(static, np.log(lam))
    spec = JointSpec("oracle", static, jax.tree.map(jnp.asarray, r.prior), _fixed_design(r))
    program, cache, ctx, pm, cm = views(spec, st, jax.tree.map(jnp.asarray, r.data))
    psi = window.scales.psi
    k = static.shapes.k
    for i, (Z, y, Zd, yd, mu0, var0, _, _) in enumerate(rows):
        cond = program.mean.conditional_eq(
            None, None, eq_slice(ctx, i), eq_slice(pm, i), eq_slice(cm, i),
            eq_slice(cache.eq, i) if not isinstance(cache.eq, Empty) else Empty(), cache,
            spec.prior.shared,
        )  # fmt: skip
        q = k + i
        K, b = np.asarray(cond.K)[:q, :q], np.asarray(cond.b)[:q]
        if static.mean.coef_scaling == "lambda":
            Kd = (np.diag(1.0 / var0) + Zd.T @ Zd + Z.T @ Z) / lam[i]
            bd = (mu0 / var0 + Zd.T @ yd + Z.T @ y) / lam[i]
        else:
            prec = np.concatenate([np.asarray(r.prior.eq.coef.prec0)[i],
                                   np.asarray(r.prior.eq.contemp.prec0)[i, :i]])  # fmt: skip
            if static.struct.contemp_scaling == "lambda":
                prec[k:] /= lam[i]
            mean = np.concatenate([np.asarray(r.prior.eq.coef.mean0)[i], np.zeros(i)])
            w = (i + 1.0) / psi[i]
            Kd = np.diag(prec) + w * Zd.T @ Zd + Z.T @ Z / lam[i]
            bd = prec * mean + w * Zd.T @ yd + Z.T @ y / lam[i]
        np.testing.assert_allclose(K, Kd, rtol=1e-10, atol=1e-10 * np.abs(Kd).max())
        np.testing.assert_allclose(b, bd, rtol=1e-10, atol=1e-10 * np.abs(bd).max())


def _fixed_design(r):
    from rai_bvar.validation.joint import FixedDesign

    s = r.static.shapes
    return FixedDesign(np.zeros((s.lags, s.n)), int(r.data.T_eff))


def test_fixed_factor_cache_with_dummies_is_the_dense_posterior():
    """Mode II ``fixed_factor``: ``θ̂ = K̃⁻¹r̃`` and ``(s ⊙ Lc⁻ᵀ)(s ⊙ Lc⁻ᵀ)ᵀ = K̃⁻¹`` with the
    λ-free ``K̃ = P̃ + FᵀD_mF + w F_dᵀF_d`` and ``r̃ = P̃μ₀ + FᵀD_my_i + w F_dᵀy_{d,i}`` (1e-10)."""
    spec = joint_spec("structural_mode2_constant_dummies", n=3)
    st, data = problem(spec, 4)
    assert spec.static.mean.const_path == "fixed_factor"
    _, cache, *_ = views(spec, st, data)
    T = int(data.T_eff)
    X, Y = np.asarray(data.X)[:T], np.asarray(data.Y)[:T]
    Mc = np.asarray(spec.prior.shared.Mc)
    Yd, Xd = np.asarray(spec.prior.shared.dummies.Y), np.asarray(spec.prior.shared.dummies.X)
    k = spec.static.shapes.k
    for i in range(3):
        F = np.hstack([X, X @ Mc[:, :i] - Y[:, :i]])
        Fd = np.hstack([Xd, Xd @ Mc[:, :i] - Yd[:, :i]])
        w = float(spec.prior.eq.coef.dummy_w[i])
        prec = np.concatenate([np.asarray(spec.prior.eq.coef.prec0)[i],
                               np.asarray(spec.prior.eq.contemp.prec0)[i, :i]])  # fmt: skip
        mean = np.concatenate([np.asarray(spec.prior.eq.coef.mean0)[i], np.zeros(i)])
        Kt = np.diag(prec) + F.T @ F + w * Fd.T @ Fd
        rt = prec * mean + F.T @ Y[:, i] + w * Fd.T @ Yd[:, i]
        c = eq_slice(cache.eq, i)
        q = k + i
        np.testing.assert_allclose(c.theta_hat[:q], np.linalg.solve(Kt, rt), rtol=1e-10)
        R = (np.asarray(c.s)[:, None] * np.linalg.inv(np.asarray(c.Lc)).T)[:q, :q]
        np.testing.assert_allclose(R @ R.T, np.linalg.inv(Kt), rtol=1e-9, atol=1e-12)


def test_cccm_conditional_with_dummies_is_the_sur_conditional():
    """CCCM with dummy rows: ``cccm_conditional_j`` equals the conditional of the system-wide
    Gaussian of ``vec Π`` (precision ``⊕_j(P_j + w_j X_dᵀX_d) + Σ_t Ω_t ⊗ x_tx_tᵀ``, the rows
    entering equation ``j`` only, untouched by ``A``), and one ``cccm_core`` scan with injected
    noise equals the sequential dense draws (precompute on and off; 1e-10)."""
    spec = joint_spec("reduced_constant_dummies")
    st, data = problem(spec, 5)
    _, cache, *_ = views(spec, st, data)
    n, k = spec.static.shapes.n, spec.static.shapes.k
    T = int(data.T_eff)
    X, Y = np.asarray(data.X)[:T], np.asarray(data.Y)[:T]
    lam = np.exp(np.asarray(st.eq.vol.core.log_lam))
    A = np.asarray(unit_lower(st.eq.struct.a))
    Omega = A.T @ np.diag(1.0 / lam) @ A
    c = spec.prior.eq.coef
    w = np.asarray(c.dummy_w)
    Xd, Yd = np.asarray(spec.prior.shared.dummies.X), np.asarray(spec.prior.shared.dummies.Y)
    Q = np.kron(Omega, X.T @ X)  # vec Π stacks the columns π_j
    bvec = (X.T @ Y @ Omega).T.reshape(-1)
    for j in range(n):
        sl = slice(j * k, (j + 1) * k)
        Q[sl, sl] += np.diag(np.asarray(c.prec0)[j]) + w[j] * Xd.T @ Xd
        bvec[sl] += np.asarray(c.prec0)[j] * np.asarray(c.mean0)[j] + w[j] * Xd.T @ Yd[:, j]
    inv_lam = np.where(np.arange(spec.static.shapes.T_pad)[None, :] < T, 1.0 / lam[:, None], 0.0)
    dummy = (cache.dummy.G, cache.dummy.FtY, cache.dummy.w)
    Pi = np.asarray(st.glob.mean.Pi)
    for j in range(n):
        cond = cccm_conditional_j(Pi, st.eq.struct.a, inv_lam, data.X, data.Y, c.mean0, c.prec0,
                                  j, dummy=dummy)  # fmt: skip
        sl = slice(j * k, (j + 1) * k)
        others = np.concatenate([Pi.T.reshape(-1)[: j * k], Pi.T.reshape(-1)[(j + 1) * k :]])
        Qo = np.delete(Q[sl], np.s_[j * k : (j + 1) * k], axis=1)
        np.testing.assert_allclose(cond.K, Q[sl, sl], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(cond.b, bvec[sl] - Qo @ others, rtol=1e-10, atol=1e-9)
    Z = random.normal(random.fold_in(KEY, 9), (n, k), dtype=jnp.float64)
    for precompute in (True, False):
        got, info = cccm_core(st.glob.mean.Pi, st.eq.struct.a, jnp.asarray(inv_lam), data.X,
                              data.Y, cache.XX, c.mean0, c.prec0, Z, gram="packed",
                              precompute=precompute, dummy=dummy)  # fmt: skip
        assert bool(info.ok)
        ref = Pi.copy()
        for j in range(n):
            cond = cccm_conditional_j(ref, st.eq.struct.a, inv_lam, data.X, data.Y, c.mean0,
                                      c.prec0, j, dummy=dummy)  # fmt: skip
            ref[:, j] = np.asarray(precision_draw(cond.K, cond.b, Z[j])[0])
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_mode2_coupling_counts_the_rows():
    """Mode II (N05 §7.7): ``n_active = k + i + m`` and
    ``quad = ½[(θ − μ)ᵀP̃(θ − μ) + w_i Σ_d (y_{d,i} − F_dθ_i)²]``; Mode I and ``iw_equivalent``
    leave the rows out of the coupling (fixed weights)."""
    spec = joint_spec("structural_mode2_constant_dummies", n=3)
    st, data = problem(spec, 6)
    program, cache, ctx, pm, cm = views(spec, st, data)
    m, k = spec.static.mean.n_dummy, spec.static.shapes.k
    Yd, Xd = np.asarray(spec.prior.shared.dummies.Y), np.asarray(spec.prior.shared.dummies.X)
    Mc = np.asarray(spec.prior.shared.Mc)
    for i in range(3):
        d, a = st.eq.mean.delta[i], st.eq.struct.a[i]
        cpl = program.mean.coupling(d, a, eq_slice(ctx, i), eq_slice(pm, i), eq_slice(cm, i),
                                    cache=cache, shared=spec.prior.shared)  # fmt: skip
        theta = np.concatenate([np.asarray(d), np.asarray(a)[:i]])
        mu = np.concatenate([np.asarray(spec.prior.eq.coef.mean0)[i], np.zeros(i)])
        P = np.concatenate([np.asarray(spec.prior.eq.coef.prec0)[i],
                            np.asarray(spec.prior.eq.contemp.prec0)[i, :i]])  # fmt: skip
        Fd = np.hstack([Xd, Xd @ Mc[:, :i] - Yd[:, :i]])
        r = Yd[:, i] - Fd @ theta
        w = float(spec.prior.eq.coef.dummy_w[i])
        assert float(cpl.n_active) == k + i + m
        np.testing.assert_allclose(
            cpl.quad, 0.5 * (np.sum(P * (theta - mu) ** 2) + w * r @ r), rtol=1e-12
        )
    for name in ("structural_constant_dummies",):
        spec1 = joint_spec(name, n=3)
        st1, data1 = problem(spec1, 7)
        program1, cache1, ctx1, pm1, cm1 = views(spec1, st1, data1)
        cpl = program1.mean.coupling(st1.eq.mean.delta[2], st1.eq.struct.a[2], eq_slice(ctx1, 2),
                                     eq_slice(pm1, 2), eq_slice(cm1, 2), cache=cache1,
                                     shared=spec1.prior.shared)  # fmt: skip
        assert float(cpl.n_active) == 0.0 and float(cpl.quad) == 0.0


# =========================================================================== identities


def _equation_identity(spec: JointSpec, rep: int, state_spec: JointSpec | None = None) -> float:
    """Worst density-ratio error of every equation's ``θ_i`` conditional at one random state."""
    static, prior = spec.static, spec.prior
    n, k = static.shapes.n, static.shapes.k
    st, data = problem(state_spec or spec, 3000 + rep)
    program, cache, ctx, pm, cm = views(spec, st, data)
    worst = 0.0
    for i in range(n):
        cache_i = Empty() if isinstance(cache.eq, Empty) else eq_slice(cache.eq, i)
        cond = program.mean.conditional_eq(None, None, eq_slice(ctx, i), eq_slice(pm, i),
                                           eq_slice(cm, i), cache_i, cache, prior.shared)  # fmt: skip
        mask = jnp.concatenate([eq_slice(pm, i).cmask, eq_slice(cm, i).cmask])
        th0 = jnp.concatenate([st.eq.mean.delta[i], st.eq.struct.a[i]])
        th1 = jnp.where(
            mask > 0, th0 + 0.05 * random.normal(random.fold_in(KEY, 50 * rep + i), th0.shape), 0.0
        )

        def with_theta(th, i=i):
            eq = replace(
                st.eq,
                mean=LinearEq(delta=st.eq.mean.delta.at[i].set(th[:k])),
                struct=RecursiveEq(a=st.eq.struct.a.at[i].set(th[k:])),
            )
            return replace(st, eq=eq)

        def log_q(th, cond=cond):
            return -0.5 * th @ cond.K @ th + cond.b @ th

        err = density_ratio_error(
            log_q, lambda th: log_joint(with_theta(th), data, prior, static), th0, th1
        )
        worst = max(worst, err)
    return worst


@pytest.mark.parametrize(
    "name",
    ["structural_constant_dummies", "structural_mode2_constant_dummies", "structural_sv_dummies",
     "iw_equivalent"],
)  # fmt: skip
def test_density_ratio_equation_step_with_dummies(name):
    """Equation step ``θ_i | λ_eff,i`` with the rows vs the independent ``log_joint`` (Bayes'
    rule with the rows; for ``iw_equivalent`` exact up to a constant), 20 states, 1e-8."""
    if name == "iw_equivalent":  # its prior sampler with rows does not exist: states without rows
        base = joint_spec("structural_iw_equivalent")
        spec = _with_dummies(base)
        state_spec = base
    else:
        spec, state_spec = joint_spec(name, n=3), None
    worst = max(_equation_identity(spec, rep, state_spec) for rep in range(20))
    assert worst <= 1e-8, worst


def _with_dummies(base: JointSpec) -> JointSpec:
    """``base`` with the test SOC/SUR rows (the ``joint_spec`` construction of WP25)."""
    from rai_bvar.validation.joint import reference_prior

    n = base.static.shapes.n
    mean = replace(base.static.mean, n_dummy=n + 1)
    static = replace(base.static, mean=mean)
    return JointSpec(base.name + "+dummies", static, reference_prior(static), base.fixed)


def test_density_ratio_coupled_lambda_step_with_dummies():
    """Mode II ``λ_i | θ_i, y`` (Constant core with the mean's coupling, rows included) vs
    ``log_joint``: ``IG(c + (T + k + i + m)/2, C + ½e'e + quad)`` (20 states, 1e-8)."""
    spec = joint_spec("structural_mode2_constant_dummies", n=3)
    static, prior = spec.static, spec.prior
    worst = 0.0
    for rep in range(20):
        st, data = problem(spec, 4000 + rep)
        program, cache, ctx, pm, cm = views(spec, st, data)
        e = residuals(st, data, prior, static)
        for i in range(3):
            cpl = program.mean.coupling(st.eq.mean.delta[i], st.eq.struct.a[i], eq_slice(ctx, i),
                                        eq_slice(pm, i), eq_slice(cm, i), cache=cache,
                                        shared=prior.shared)  # fmt: skip
            live = data.tmask > 0
            log_e2 = jnp.where(live, jnp.log(jnp.where(live, e[i] ** 2, 1.0)), 0.0)
            shape, scale = program.vol.core.conditional(log_e2, data.tmask, data.T_eff,
                                                        eq_slice(prior.eq, i), cpl)  # fmt: skip
            ll = st.eq.vol.core.log_lam

            def at(d, i=i, ll=ll, st=st):
                core = ConstantEq(log_lam=ll.at[i].add(d))
                return replace(st, eq=replace(st.eq, vol=replace(st.eq.vol, core=core)))

            def log_q(d, i=i, shape=shape, scale=scale, ll=ll):
                x = ll[i] + d
                return -(shape + 1.0) * x - scale * jnp.exp(-x)

            for d0, d1 in ((-0.3, 0.4), (0.2, -0.5)):
                err = density_ratio_error(
                    log_q, lambda d, data=data: log_joint(at(d), data, prior, static), d0, d1
                )
                worst = max(worst, err)
    assert worst <= 1e-8, worst


def test_density_ratio_cccm_step_with_dummies():
    """``π_j | π_{−j}, A, λ, y`` with the rows vs ``log_joint`` (20 states, every ``j``, 1e-8)."""
    spec = joint_spec("reduced_constant_dummies")
    static, prior = spec.static, spec.prior
    k = static.shapes.k
    worst = 0.0
    for rep in range(20):
        st, data = problem(spec, 5000 + rep)
        program, cache, *_ = views(spec, st, data)
        from rai_bvar.engines.mcmc.state import SweepCtx

        sctx = SweepCtx(data=data, prior=prior, cache=cache, eq_out=Empty())
        step = program.global_pre[0]
        Pi = st.glob.mean.Pi
        for j in range(static.shapes.n):
            cond = step.conditional_j(st, sctx, j)
            p0 = Pi[:, j]
            p1 = p0 + 0.05 * random.normal(random.fold_in(KEY, 70 * rep + j), (k,))

            def with_pi(p, j=j, st=st, Pi=Pi):
                return replace(st, glob=replace(st.glob, mean=CCCMGlobal(Pi=Pi.at[:, j].set(p))))

            def log_q(p, cond=cond):
                return -0.5 * p @ cond.K @ p + cond.b @ p

            err = density_ratio_error(
                log_q, lambda p, data=data: log_joint(with_pi(p), data, prior, static), p0, p1
            )
            worst = max(worst, err)
    assert worst <= 1e-8, worst


# =========================================================================== closed path


def _raise(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("dummy code traced in a composition without dummy rows")


def test_dummy_path_is_closed_without_dummies(monkeypatch):
    """Every pre-WP25 composition (no dummy rows) traces prepare, the sweep and the monitor with
    every dummy helper replaced by a raising stub, and carries ``Empty`` in ``FitCache.dummy``,
    ``SharedPrior.dummies`` and ``CoefPriorArrays.dummy_w``: the WP25 code is never part of their
    programs (their draws are bitwise those of the parent commit; WP25 report)."""
    monkeypatch.setattr(linear_mod.LinearStructural, "dummy_terms", _raise)
    monkeypatch.setattr(linear_mod.LinearStructural, "_dummy_rows", _raise)
    monkeypatch.setattr(linear_mod.LinearStructural, "dummy_log_density_eq", _raise)
    monkeypatch.setattr(reduced_mod.CCCMStep, "dummy_terms", _raise)
    monkeypatch.setattr(data_mod, "dummy_cache", _raise)
    for module, fn in (
        (sim_mod, "structural_dummy_prior_sample_eq"),
        (lj_mod, "structural_dummy_log_factor_eq"),
        (ref_mod, "reduced_dummy_prior_sample"),
        (ref_mod, "reduced_dummy_log_factor"),
        (tf_mod, "structural_rows_eq"),
    ):
        monkeypatch.setattr(module, fn, _raise)
    jax.clear_caches()
    runner.clear_caches()
    try:
        for name in PRE_WP25:
            spec = joint_spec(name)
            assert spec.static.mean.n_dummy == 0, name
            st, data = problem(spec, 0)
            prepare, sweep = production_sweep.__wrapped__(spec.static, spec.overrides)
            cache = jax.eval_shape(prepare, data, spec.prior)
            assert isinstance(cache.dummy, Empty), name
            assert isinstance(spec.prior.shared.dummies, Empty), name
            assert isinstance(spec.prior.eq.coef.dummy_w, Empty), name
            cache = prepare(data, spec.prior)
            jax.eval_shape(sweep, KEY, jnp.uint32(0), st, data, spec.prior, cache)
            program = compose(spec.static)

            def monitor(s, d, c, program=program, prior=spec.prior):
                return monitor_values(s, d, prior, c, program)

            jax.eval_shape(monitor, st, data, cache)
            jax.eval_shape(spec.test_functions, st, data)  # reference log prior and functions
        for name in DUMMY_SPECS:  # positive control: with rows, the stubs are reached
            spec = joint_spec(name)
            st = jax.tree.map(jnp.asarray, state_template(spec.static))
            data = tf_mod.engine_data_template(spec.static, spec.fixed.T_eff)
            prepare, _ = production_sweep.__wrapped__(spec.static, spec.overrides)
            for fn, args in ((prepare, (data, spec.prior)), (spec.prior_sample, (KEY,)),
                             (spec.test_functions, (st, data))):  # fmt: skip
                with pytest.raises(AssertionError, match="dummy code traced"):
                    jax.eval_shape(fn, *args)
    finally:
        jax.clear_caches()
        runner.clear_caches()


# =========================================================================== monitor


@pytest.mark.parametrize("name", ["structural_mode2_constant_dummies", "reduced_constant_dummies"])
def test_monitor_loglik_unchanged_and_log_prior_up_to_a_constant(name):
    """The rows are prior rows: at a fixed state and dataset the monitor's log-likelihood is that
    of the composition without rows (1e-12), and its ``log_prior`` moves between two states as the
    reference ``log p(θ)`` with the rows (Bayes' rule; equal up to the rows' constant, 1e-8)."""
    spec = joint_spec(name)
    off = _without_dummies(spec)
    st, data = problem(spec, 8)
    st2, _ = problem(spec, 9)
    program, cache, *_ = views(spec, st, data)
    program0, cache0, *_ = views(off, st, data)
    on1 = monitor_values(st, data, spec.prior, cache, program)
    off1 = monitor_values(st, data, off.prior, cache0, program0)
    np.testing.assert_allclose(on1["loglik"], off1["loglik"], rtol=1e-12)
    on2 = monitor_values(st2, data, spec.prior, cache, program)
    from rai_bvar.validation.log_joint import log_prior

    d_mon = float(on2["log_prior"] - on1["log_prior"])
    d_ref = float(log_prior(st2, spec.prior, spec.static, data.tmask)
                  - log_prior(st, spec.prior, spec.static, data.tmask))  # fmt: skip
    assert abs(d_mon - d_ref) <= 1e-8 * (1.0 + abs(d_ref))


def _without_dummies(spec: JointSpec) -> JointSpec:
    mean = replace(spec.static.mean, n_dummy=0)
    if mean.kind == "linear_structural" and spec.static.vol.constant_weights:
        mean = replace(mean, const_path=spec.static.mean.const_path)
    static = replace(spec.static, mean=mean)
    prior = replace(
        spec.prior,
        eq=replace(spec.prior.eq, coef=replace(spec.prior.eq.coef, dummy_w=Empty())),
        shared=replace(spec.prior.shared, dummies=Empty()),
    )
    return JointSpec(spec.name + "-off", static, prior, spec.fixed)


def test_dummy_rows_shared_structure():
    """``DummyRows`` is the reduced-form layout ``(Y_d, X_d)`` the builders and the test prior
    share."""
    spec = joint_spec("structural_constant_dummies")
    rows = spec.prior.shared.dummies
    assert isinstance(rows, DummyRows)
    n, k = spec.static.shapes.n, spec.static.shapes.k
    assert rows.Y.shape == (n + 1, n) and rows.X.shape == (n + 1, k)


# =========================================================================== EB and end to end


@pytest.mark.parametrize(("form", "cross"), [("structural", 1.0), ("structural", 0.5),
                                             ("reduced", 0.5)])  # fmt: skip
def test_eb_plugs_in_the_dummy_weights(form, cross):
    """DESIGN §6.5 with dummies: the conjugate twin keeps ``soc``/``sur`` (λ, μ, δ free), Chan's
    structural EB estimates them at ``cross ≠ 1``; the values in use are those modes."""
    from rai_bvar.engines import structural_nig
    from rai_bvar.engines.analytic import AnalyticFit
    from rai_bvar.engines.mcmc import eb

    data = var_data()
    prior = rb.Minnesota(form=form, cross=cross, soc=1.0, sur=1.0)
    model = rb.BVAR(mean=rb.Linear(prior=prior), vol=rb.Constant("ig"), lags=2)
    twin_model = eb.conjugate_twin(model)
    assert (twin_model.mean.prior.soc, twin_model.mean.prior.sur) == (1.0, 1.0)
    out = eb.eb_hyper(model, data)
    assert out.hyper["eb_free"] == ("lambda", "soc", "sur")
    twin = AnalyticFit(twin_model, data, None)
    if form == "structural" and cross != 1.0:
        ref = structural_nig.eb_structural(data, prior, 2, "c", start=twin.hyper).hyper
    else:
        ref = twin.hyper
    for key in ("lambda", "soc", "sur"):
        assert out.hyper[key] == pytest.approx(ref[key], rel=1e-12)


def test_fit_with_dummies_end_to_end():
    """A short Mode I fit with dummies: hyper, metadata, summary note, forecasts and LPD run."""
    data = var_data()
    settings = rb.MCMCSettings(chains=2, draws=100, burn=50, store_max=100, seed=3)
    model = rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form="structural", soc=1.0, sur=1.0)),
        vol=rb.SV(),
        lags=2,
        mcmc=settings,
    )
    with pytest.warns(UserWarning):
        fit = model.fit(data)
    assert fit.hyper["n_dummy"] == 4 and fit.metadata["const_path"] == "weighted"
    assert "dummy rows" in fit.summary().attrs["implied_reduced_form"]["dummies"]
    fc = fit.forecast(2, draws=20, seed=1)
    assert np.all(np.isfinite(np.asarray(fc.draws.values if hasattr(fc, "draws") else fc.paths)))
