"""Observable simulators and :class:`JointSpec` (DESIGN §9.2, §13 WP9; N06 §2.1, §6 pitfall 2).

* ``simulate_obs`` equals a plain NumPy loop driven by the same noise (1e-12), recursively in t
  with the state's current latents; ``X`` equals ``data.design.lagged_design`` on the stacked
  series and padded rows are exactly zero;
* the auxiliary-model simulator reproduces ``y* = h + m_s + √v_s z`` exactly;
* ``joint_spec(name)`` builds for every name in ``COMPOSITIONS`` and ``BLOCK_CONFIGS`` (acceptance)
  and its methods trace under ``jit``/``vmap``;
* ``JointSpec.step`` plumbing (frozen blocks, overrides, sampler-only prior) against a test double
  of the production sweep, and the real sweep when WP14 provides it;
* every Geweke/SBC test function equals a NumPy/scipy oracle (names and values, several prior
  draws per composition, both sizes for a subset), so a wrong date, mask, Jacobian or index in
  ``testfns``/``reference`` cannot pass unnoticed.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import stats

from rai_bvar.data.design import lagged_design
from rai_bvar.engines.mcmc import data as engine_data
from rai_bvar.engines.mcmc.state import (
    EngineData,
    FitCache,
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


def _build_engine_data_double(y, presample, T_eff, *, n_det, t0):
    """Local test double of ``data.build_engine_data`` (WP5), used only if WP5 is absent."""
    T_pad = y.shape[0]
    p = presample.shape[0]
    live = jnp.arange(T_pad) < T_eff
    Y = jnp.where(live[:, None], y, 0.0)
    full = jnp.concatenate([presample, Y])
    det = [jnp.ones(T_pad), t0 + jnp.arange(T_pad, dtype=jnp.float64)][:n_det]
    lags = [full[p - lag : p - lag + T_pad] for lag in range(1, p + 1)]
    X = jnp.where(live[:, None], jnp.concatenate([jnp.stack(det, 1), *lags], 1), 0.0)
    return EngineData(Y, X, live.astype(jnp.float64), jnp.asarray(T_eff, jnp.int32),
                      jax.lax.dynamic_slice_in_dim(full, T_eff, p), t0 + T_eff - 1.0)  # fmt: skip


@pytest.fixture(autouse=True)
def _engine_data_builder(monkeypatch):
    if not hasattr(engine_data, "build_engine_data"):  # pragma: no cover - depends on WP5
        monkeypatch.setattr(
            engine_data, "build_engine_data", _build_engine_data_double, raising=False
        )


def _np(tree):
    return jax.tree.map(np.asarray, tree)


def _numpy_log_var(st, static):
    """Effective log variance (T_pad × n) from the state, written directly in NumPy."""
    vol = st.eq.vol
    T_pad = static.shapes.T_pad
    if static.vol.persistent == "constant":
        lv = np.repeat(vol.core.log_lam[:, None], T_pad, axis=1)
    else:
        lv = vol.core.h[:, 1:].copy()
    if static.vol.transitory in ("outlier", "outlier_t"):
        lv = lv + 2.0 * np.log(vol.trans.o_idx + 1.0)
    if static.vol.transitory in ("t", "outlier_t"):
        lv = lv + vol.trans.log_tau
    return lv.T


def numpy_simulate(spec: J.JointSpec, st, eps: np.ndarray) -> np.ndarray:
    """``y_t`` by a Python loop over t with dense solves (reference for the jnp recursion)."""
    st = _np(st)
    sh = spec.static.shapes
    n, p, T = sh.n, sh.lags, spec.fixed.T_eff
    a = st.eq.struct.a
    A = np.eye(n)
    for i in range(n):
        for j in range(i):
            A[i, j] = a[i, j]
    e = np.exp(0.5 * _numpy_log_var(st, spec.static)) * eps
    kind = spec.static.mean.kind
    if kind == "linear_structural":
        G = st.eq.mean.delta + a @ np.asarray(spec.prior.shared.Mc).T
    elif kind == "linear_reduced":
        G = st.glob.mean.Pi.T
    else:
        G = np.zeros((n, sh.k))
    hist = [row for row in np.asarray(spec.fixed.presample)]
    Y = np.zeros((sh.T_pad, n))
    for t in range(T):
        x = np.concatenate([[1.0], *[hist[-lag] for lag in range(1, p + 1)]])
        if kind == "linear_reduced":
            y = G @ x + np.linalg.solve(A, e[t])
        else:
            y = np.linalg.solve(A, G @ x + e[t])
        Y[t] = y
        hist.append(y)
    return Y


@pytest.mark.parametrize(
    "name",
    ["structural_sv_outliers_t", "reduced_sv_t", "structural_mode2_constant", "zero_sv",
     "reduced_iw_equivalent", "structural_sv_rw"],
)  # fmt: skip
@pytest.mark.parametrize("size", ["small", "full"])
def test_simulate_obs_matches_numpy_loop(name, size):
    spec = J.joint_spec(name, size)
    st = spec.prior_sample(random.key(11))
    sh = spec.static.shapes
    eps = np.asarray(random.normal(random.key(12), (sh.T_pad, sh.n), dtype=jnp.float64))
    data = jax.jit(lambda s, e: S.simulate_obs_core(e, s, spec.static, spec.prior, spec.fixed))(
        st, eps
    )
    data = _np(data)
    T, p = spec.fixed.T_eff, sh.lags
    ref = numpy_simulate(spec, st, eps)
    np.testing.assert_allclose(data.Y[:T], ref[:T], rtol=1e-12, atol=1e-12)
    assert np.all(data.Y[T:] == 0.0) and np.all(data.X[T:] == 0.0)  # padding exact (D-15)
    # X rebuilt from the simulated y equals the Phase 1 design on the stacked series (read-only)
    stacked = np.vstack([spec.fixed.presample, data.Y[:T]])
    Yd, Xd = lagged_design(stacked, p, "c")
    np.testing.assert_array_equal(data.Y[:T], Yd)
    np.testing.assert_allclose(data.X[:T], Xd, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(data.tmask, (np.arange(sh.T_pad) < T).astype(float))
    assert int(data.T_eff) == T
    np.testing.assert_array_equal(data.hist, stacked[-p:])
    assert float(data.t_last) == spec.fixed.t0 + T - 1
    # the regressors used inside the recursion are those of the rebuilt data
    shocks = S.gaussian_shocks(jnp.asarray(eps), st, spec.static)
    Y2, X2 = S.simulate_targets(shocks, st, spec.static, spec.prior, spec.fixed)
    np.testing.assert_allclose(np.asarray(X2), data.X, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(Y2), data.Y, rtol=0, atol=1e-15)


def test_simulate_uses_current_latents():
    spec = J.joint_spec("zero_sv")
    st = spec.prior_sample(random.key(1))
    key = random.key(2)
    d1 = spec.simulate(key, st)
    core = st.eq.vol.core
    st2 = jax.tree_util.tree_map_with_path(
        lambda path, x: x + 1.0 if leaf_path(path) == "eq.vol.core.h" else x, st
    )
    d2 = spec.simulate(key, st2)
    T = spec.fixed.T_eff
    # e = exp(h/2)·ε: shifting h by 1 scales the targets by e^{1/2}
    np.testing.assert_allclose(np.asarray(d2.Y[:T]), np.exp(0.5) * np.asarray(d1.Y[:T]), rtol=1e-12)
    assert core.h.shape == (1, spec.static.shapes.T_pad + 1)
    # mutant M3 (fresh latents): the data ignore the state's h
    m3 = J.joint_spec("zero_sv", fresh_latents=True)
    np.testing.assert_array_equal(
        np.asarray(m3.simulate(key, st).Y), np.asarray(m3.simulate(key, st2).Y)
    )


def test_aux_model_simulator_reproduces_y_star():
    spec = J.joint_spec("aux_sv")  # ZeroMean × SV, n = 1, zero offset, exact = False
    assert spec.static.vol.zero_offset and not spec.static.vol.exact and spec.obs == "aux"
    st = spec.prior_sample(random.key(3))
    sh = spec.static.shapes
    shape = (sh.T_pad, sh.n)
    table = R.mixture_table("ocsn10")
    s_idx = np.asarray(random.randint(random.key(4), shape, 0, 10), dtype=np.int32)
    z = np.asarray(random.normal(random.key(5), shape, dtype=jnp.float64))
    u = np.asarray(random.uniform(random.key(6), shape, dtype=jnp.float64))
    e, ystar = S.aux_shocks(jnp.asarray(s_idx), jnp.asarray(z), jnp.asarray(u), st, spec.static)
    h = np.asarray(st.eq.vol.core.h)[0, 1:]
    ref = h + table.m[s_idx[:, 0]] + np.sqrt(table.v[s_idx[:, 0]]) * z[:, 0]
    np.testing.assert_allclose(np.asarray(ystar)[:, 0], ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.log(np.asarray(e) ** 2)[:, 0], ref, rtol=1e-12, atol=1e-12)
    assert np.all(np.sign(np.asarray(e)) == np.where(u < 0.5, -1.0, 1.0))
    Y, _ = S.simulate_targets(e, st, spec.static, spec.prior, spec.fixed)
    data = S._engine_data(Y, spec.static, spec.fixed)
    T = spec.fixed.T_eff
    ys = np.asarray(LJ.y_star(st, data, spec.prior, spec.static))[0]  # log c = −∞ (zero offset)
    np.testing.assert_allclose(ys[:T], ref[:T], rtol=1e-12, atol=1e-12)
    assert np.all(ys[T:] == 0.0)
    # the jitted simulator draws all of this from one key
    d = jax.jit(spec.simulate)(random.key(7), st)
    assert np.all(np.isfinite(np.asarray(d.Y))) and np.all(np.asarray(d.Y)[T:] == 0)


def test_aux_simulator_requires_sv_core():
    spec = J.joint_spec("structural_constant")
    st = spec.prior_sample(random.key(0))
    z = jnp.zeros((spec.static.shapes.T_pad, 2))
    with pytest.raises(ValueError, match="SV persistent core"):
        S.aux_shocks(z.astype(jnp.int32), z, z, st, spec.static)
    with pytest.raises(ValueError, match="SV persistent core"):
        J.joint_spec("structural_constant", obs="aux")


# --------------------------------------------------------------------------- test-function oracle


def _numpy_residuals(spec, st, data):
    """Structural residuals ``e`` (``n × T``) from the state, in NumPy (D3 Γ built directly)."""
    n, T = spec.static.shapes.n, spec.fixed.T_eff
    A = np.eye(n)
    for i in range(n):
        A[i, :i] = st.eq.struct.a[i, :i]
    Y, X = data.Y[:T], data.X[:T]
    kind = spec.static.mean.kind
    if kind == "linear_structural":
        E = Y @ A.T - X @ _numpy_gamma(spec, st).T
    elif kind == "linear_reduced":
        E = (Y - X @ st.glob.mean.Pi) @ A.T
    else:
        E = Y @ A.T
    return E.T, A


def _numpy_gamma(spec, st):
    return st.eq.mean.delta + st.eq.struct.a @ np.asarray(spec.prior.shared.Mc).T


def _acf1(x):
    xc = x - x.mean()
    return float(np.sum(xc[1:] * xc[:-1]) / np.sum(xc**2))


def _marginal_density(z, spec, tr):
    """Transitory-integrated density of ``z = e/√λ_pers`` (N03 §2.2) by direct scipy mixtures."""
    vol = spec.static.vol
    if vol.transitory in ("t", "outlier_t"):
        nu = float(vol.nu_grid[int(tr.nu_idx)])
        c = nu - 2.0 if vol.t_scale == "standardized" else nu
        base = lambda x: stats.t.pdf(x, nu, scale=math.sqrt(c / nu))  # noqa: E731
    else:
        base = stats.norm.pdf
    if vol.transitory in ("outlier", "outlier_t"):
        K, p = vol.outlier_K, math.exp(float(tr.log_p))
        ks = np.arange(1, K + 1)
        w = np.where(ks == 1, 1.0 - p, p / (K - 1))
        return np.sum(w / ks * base(z[:, None] / ks), axis=1)
    return base(z)


def numpy_test_functions(spec: J.JointSpec, st, data, raw_moments: bool) -> dict[str, float]:
    """Every test function of ``testfns.test_functions`` written again from its definition."""
    st, data = _np(st), _np(data)
    static, vol = spec.static, spec.static.vol
    n, nd, k, T = static.shapes.n, static.shapes.n_det, static.shapes.k, spec.fixed.T_eff
    out: dict[str, float] = {}
    lv = _numpy_log_var(st, static).T[:, :T]  # (n, T) effective log variance
    for i in range(n):
        for j in range(i):
            a = st.eq.struct.a[i, j]
            out[f"eq{i}.struct.a{j}"], out[f"eq{i}.struct.a{j}^2"] = a, a**2
        if static.mean.kind == "linear_structural":
            for r in range(k):
                d = st.eq.mean.delta[i, r]
                out[f"eq{i}.mean.delta{r}"], out[f"eq{i}.mean.delta{r}^2"] = d, d**2
        core, pre = jax.tree.map(lambda x, i=i: x[i], st.eq.vol.core), f"eq{i}.core."
        if vol.persistent == "constant":
            out[pre + "log_lam"], out[pre + "log_lam^2"] = core.log_lam, core.log_lam**2
        else:
            h, s = core.h, core.sigma
            dates = {"h0": h[0], "h1": h[1], "hmid": h[math.ceil(T / 2)], "hT": h[T],
                     "mean_h": np.mean(h[1 : T + 1]), "mean_abs_h": np.mean(np.abs(h[1 : T + 1]))}  # fmt: skip
            out.update({pre + key: val for key, val in dates.items()})
            out[pre + "sigma"], out[pre + "sigma^2"] = s, s**2
            if vol.rw:
                out[pre + "sigma*hT"] = s * h[T]
            else:
                mu, phi = core.mu, core.phi
                out.update({pre + "mu": mu, pre + "phi": phi, pre + "mu^2": mu**2,
                            pre + "phi^2": phi**2, pre + "mu*phi": mu * phi,
                            pre + "phi*sigma": phi * s, pre + "mu*hT": mu * h[T],
                            pre + "hT^2": h[T] ** 2})  # fmt: skip
        tr, pre = jax.tree.map(lambda x, i=i: x[i], st.eq.vol.trans), f"eq{i}.trans."
        if vol.transitory in ("outlier", "outlier_t"):
            p, o = math.exp(float(tr.log_p)), tr.o_idx[:T] + 1.0
            n_out = float(np.sum(o > 1))
            out.update({pre + "p": p, pre + "p^2": p**2, pre + "n_out": n_out,
                        pre + "share_out": n_out / T, pre + "p*n_out": p * n_out,
                        pre + "mean_log_o": np.mean(np.log(o)), pre + "mean_o": np.mean(o)})  # fmt: skip
        if vol.transitory in ("t", "outlier_t"):
            nu, lt = float(vol.nu_grid[int(tr.nu_idx)]), tr.log_tau[:T]
            out.update({pre + "nu": nu, pre + "nu^2": nu**2, pre + "nu_le_5": float(nu <= 5),
                        pre + "mean_log_tau": np.mean(lt), pre + "nu*mean_log_tau": nu * np.mean(lt),
                        pre + "mean_inv_tau": np.mean(np.exp(-lt))})  # fmt: skip
    if static.mean.kind == "linear_reduced":
        Pi = st.glob.mean.Pi
        for j in range(n):
            for r in range(k):
                out[f"glob.mean.Pi[{r},{j}]"], out[f"glob.mean.Pi[{r},{j}]^2"] = (
                    Pi[r, j],
                    Pi[r, j] ** 2,
                )
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                for r in range(k):
                    out[f"glob.mean.Pi[{r},{j1}]*Pi[{r},{j2}]"] = Pi[r, j1] * Pi[r, j2]
    E, A = _numpy_residuals(spec, st, data)
    if static.mean.kind == "linear_structural":
        G = _numpy_gamma(spec, st)
        B = np.linalg.solve(A, G).T  # reduced-form coefficients (k × n)
        for i in range(n):
            for j in range(i):
                out[f"eq{i}.joint.gamma_lag1_{j}"] = G[i, nd + j]
                out[f"eq{i}.joint.a{j}*gamma_lag1_{j}"] = st.eq.struct.a[i, j] * G[i, nd + j]
            out[f"eq{i}.joint.B_own1"] = B[nd + i, i]
        out["joint.gamma_minus_AM_sq"] = np.sum((G - A @ np.asarray(spec.prior.eq.coef.mean0)) ** 2)
    if static.mean.n_dummy:  # SOC/SUR rows (WP25): Σ_d (y_d − F_dθ)²
        Yd = np.asarray(spec.prior.shared.dummies.Y)
        Xd = np.asarray(spec.prior.shared.dummies.X)
        if static.mean.kind == "linear_structural":
            Mc = np.asarray(spec.prior.shared.Mc)
            for i in range(n):
                Fd = np.hstack([Xd, Xd @ Mc[:, :i] - Yd[:, :i]])
                theta = np.concatenate([st.eq.mean.delta[i], st.eq.struct.a[i, :i]])
                out[f"eq{i}.joint.dummy_ss"] = np.sum((Yd[:, i] - Fd @ theta) ** 2)
        else:
            R = Yd - Xd @ np.asarray(st.glob.mean.Pi)
            for j in range(n):
                out[f"glob.mean.dummy_ss[{j}]"] = np.sum(R[:, j] ** 2)
    ll = stats.norm.logpdf(E, 0.0, np.exp(0.5 * lv))
    out["loglik"] = np.sum(ll)
    out["log_prior"] = float(
        LJ.log_prior(st, spec.prior, static, data.tmask)
    )  # vs scipy: test_log_joint
    for i in range(n):
        out[f"eq{i}.joint.rss_std"] = np.sum(E[i] ** 2 / np.exp(lv[i]))
        if n > 1:
            out[f"eq{i}.joint.loglik"] = np.sum(ll[i])
    if vol.transitory != "none":
        lp = lv.copy()
        if vol.transitory in ("outlier", "outlier_t"):
            lp -= 2.0 * np.log(st.eq.vol.trans.o_idx[:, :T] + 1.0)
        if vol.transitory in ("t", "outlier_t"):
            lp -= st.eq.vol.trans.log_tau[:, :T]
        for i in range(n):
            tr = jax.tree.map(lambda x, i=i: x[i], st.eq.vol.trans)
            f = _marginal_density(E[i] / np.exp(0.5 * lp[i]), spec, tr)
            out[f"eq{i}.joint.loglik_marginal"] = np.sum(np.log(f) - 0.5 * lp[i])
    Y = data.Y[:T]
    for i in range(n):
        out[f"data.mean_log_y2[{i}]"] = np.mean(np.log(Y[:, i] ** 2))
        out[f"data.mean_tanh_y[{i}]"] = np.mean(np.tanh(Y[:, i]))
        out[f"data.acf1_y[{i}]"] = _acf1(Y[:, i])
        out[f"data.acf1_log_y2[{i}]"] = _acf1(np.log(Y[:, i] ** 2))
        for j in range(i + 1, n):
            out[f"data.corr_y[{i},{j}]"] = np.corrcoef(Y[:, i], Y[:, j])[0, 1]
        if raw_moments:
            out[f"data.mean_y2[{i}]"] = np.mean(Y[:, i] ** 2)
            out[f"data.acov1_y[{i}]"] = np.sum(Y[1:, i] * Y[:-1, i]) / T
            for j in range(i + 1, n):
                out[f"data.mean_yy[{i},{j}]"] = np.mean(Y[:, i] * Y[:, j])
    return {key: float(val) for key, val in out.items()}


_ORACLE_FULL = ("structural_sv_outliers_t", "reduced_sv_t", "structural_mode2_constant",
                "structural_sv_rw", "reduced_sv_outliers")  # fmt: skip


@pytest.mark.parametrize(
    ("name", "size"),
    [(nm, "small") for nm in J.COMPOSITIONS + J.BLOCK_CONFIGS]
    + [(nm, "full") for nm in _ORACLE_FULL],
)
def test_test_functions_match_numpy_oracle(name, size):
    """Names and values of every test function on 3 prior draws (and simulated data) per spec."""
    spec = J.joint_spec(name, size)
    keys = jax.vmap(lambda i: random.fold_in(random.key(31), i))(jnp.arange(3, dtype=jnp.uint32))
    sts = jax.jit(jax.vmap(spec.prior_sample))(keys)
    datas = jax.jit(jax.vmap(spec.simulate))(keys, sts)
    fns = jax.jit(lambda s, d: {k: v for k, (v, _) in spec._functions(s, d).items()})
    for c in range(3):
        st, data = (jax.tree.map(lambda x, c=c: x[c], t) for t in (sts, datas))
        got = {k: float(v) for k, v in fns(st, data).items()}
        ref = numpy_test_functions(spec, st, data, spec.raw_moments)
        assert sorted(got) == sorted(ref), sorted(set(got) ^ set(ref))
        assert set(got) == set(spec.all_names)  # (a jitted dict comes back with sorted keys)
        bad = {
            k: (got[k], ref[k])
            for k in ref
            if not np.isclose(got[k], ref[k], rtol=1e-10, atol=1e-10)
        }
        assert not bad, (name, size, c, bad)


def test_gamma_minus_AM_is_delta_deviation_under_d3():
    """``‖Γ − AM'‖² = Σ_i ‖δ_i − m_i‖²`` under D3 (``γ_i = δ_i + M_c a_i``; DESIGN §9.6, §9.7 M8)."""
    for name, size in (("structural_sv", "small"), ("structural_sv_outliers_t", "full")):
        spec = J.joint_spec(name, size)
        st = spec.prior_sample(random.key(4))
        d = TF.test_functions(st, spec.simulate(random.key(5), st), spec.prior, spec.static)
        dev = np.asarray(st.eq.mean.delta) - np.asarray(spec.prior.eq.coef.mean0)
        np.testing.assert_allclose(
            float(d["joint.gamma_minus_AM_sq"][0]), np.sum(dev**2), rtol=1e-12
        )
        assert np.any(np.asarray(st.eq.struct.a) != 0.0)  # the centring term is really exercised


# --------------------------------------------------------------------------- JointSpec


@pytest.mark.parametrize("name", J.COMPOSITIONS + J.BLOCK_CONFIGS)
@pytest.mark.parametrize("size", ["small", "full"])
def test_joint_spec_builds_for_every_name(name, size):
    spec = J.joint_spec(name, size)
    sh = spec.static.shapes
    assert spec.fixed.T_eff % 32 != 0 and sh.T_pad == 32 * -(-spec.fixed.T_eff // 32)
    assert len(spec.names) == len(set(spec.names)) > 0
    assert J.composition_id(name) in (*range(1, 18), *range(101, 108), *range(201, 206))

    def run(keys):  # the JointModel methods, batched over chains (one compile)
        sts = jax.vmap(spec.prior_sample)(keys)
        datas = jax.vmap(spec.simulate)(keys, sts)
        return jax.vmap(spec.test_functions)(sts, datas), jax.vmap(spec.quantities)(sts, datas)

    keys = jax.vmap(lambda i: random.fold_in(random.key(0), i))(jnp.arange(3, dtype=jnp.uint32))
    if size == "full":  # nightly sizes: trace only (the small size runs the same code)
        g, q = jax.eval_shape(run, keys)
    else:
        sts = jax.jit(jax.vmap(spec.prior_sample))(keys)
        for c in range(3):
            one = jax.tree.map(lambda x, c=c: np.asarray(x[c]), sts)
            assert check_state_invariants(one, spec.static, spec.fixed.T_eff) == [], name
        g, q = jax.jit(run)(keys)
        assert bool(jnp.all(jnp.isfinite(g))), name
    assert g.shape == (3, len(spec.names)) and g.dtype == jnp.float64
    assert q.shape == (3, len(spec.all_names))


def test_composition_table_matches_design():
    assert len(J.COMPOSITIONS) == 22 and len(J.BLOCK_CONFIGS) == 7
    assert [J.composition_id(c) for c in J.COMPOSITIONS] == [*range(1, 18), *range(201, 206)]
    # WP25 dummy rows: (n_dummy small, const_path, coef_scaling)
    for name, (m, path, cs) in {
        "structural_constant_dummies": (3, "eigen", "none"),
        "structural_mode2_constant_dummies": (3, "fixed_factor", "lambda"),
        "reduced_constant_dummies": (4, "weighted", "none"),
        "structural_sv_dummies": (3, "weighted", "none"),
        "structural_iw_equivalent_dummies": (4, "cholesky_const", "none"),
    }.items():
        st = J.joint_spec(name).static
        assert (st.mean.n_dummy, st.mean.const_path, st.mean.coef_scaling) == (m, path, cs), name
    for name in J.COMPOSITIONS[:17] + J.BLOCK_CONFIGS:
        assert J.joint_spec(name).static.mean.n_dummy == 0, name
    expect = {
        "structural_constant": ("linear_structural", "constant", "none", "eigen", "none"),
        "structural_mode2_constant": ("linear_structural", "constant", "none", "fixed_factor",
                                      "lambda"),
        "structural_iw_equivalent": ("linear_structural", "constant", "none", "cholesky_const",
                                     "none"),
        "reduced_sv_outliers_t": ("linear_reduced", "sv", "outlier_t", "weighted", "none"),
        "zero_sv": ("zero", "sv", "none", "weighted", "none"),
    }  # fmt: skip
    for name, (mean, core, trans, path, coef_scaling) in expect.items():
        st = J.joint_spec(name).static
        assert (st.mean.kind, st.vol.persistent, st.vol.transitory) == (mean, core, trans)
        assert (st.mean.const_path, st.mean.coef_scaling) == (path, coef_scaling)
    assert J.joint_spec("structural_mode2_constant").static.struct.contemp_scaling == "lambda"
    assert J.joint_spec("reduced_iw_equivalent").static.vol.coupled
    assert J.joint_spec("structural_sv_rw").static.vol.rw
    assert J.joint_spec("structural_sv_invgamma").static.vol.sigma2_prior == "inv_gamma"
    assert J.joint_spec("structural_sv_fixed_b0").static.vol.h0 == "fixed_b0"
    small, full = J.joint_spec("structural_sv"), J.joint_spec("structural_sv", "full")
    assert (small.static.shapes.n, small.static.shapes.lags, small.fixed.T_eff) == (2, 1, 20)
    assert (full.static.shapes.n, full.static.shapes.lags, full.fixed.T_eff) == (3, 2, 50)
    assert np.all(small.fixed.presample == 0) and np.any(full.fixed.presample != 0)
    assert J.joint_spec("reduced_sv").static.shapes.n == 3  # middle j of CCCM exercised
    assert J.joint_spec("structural_sv", exact=False).static.vol.exact is False  # rows 18+
    # the n column of the DESIGN §9.5 table, (small, full), row by row
    n_by_row = {1: (2, 3), 2: (2, 3), 3: (2, 3), 4: (2, 3), 5: (2, 3), 6: (3, 3), 7: (3, 3),
                8: (3, 3), 9: (3, 3), 10: (3, 3), 11: (3, 3), 12: (3, 3), 13: (3, 3), 14: (2, 2),
                15: (2, 2), 16: (2, 2), 17: (1, 1),
                201: (2, 3), 202: (2, 3), 203: (3, 3), 204: (2, 3), 205: (3, 3)}  # fmt: skip
    for name in J.COMPOSITIONS:
        ns = tuple(J.joint_spec(name, size).static.shapes.n for size in ("small", "full"))
        assert ns == n_by_row[J.composition_id(name)], name
    for name in J.COMPOSITIONS + J.BLOCK_CONFIGS:  # nightly lags and T_eff for every row
        full = J.joint_spec(name, "full")
        assert (full.static.shapes.lags, full.fixed.T_eff) == (2, 50), name


def test_joint_spec_overrides_and_errors():
    with pytest.raises(KeyError, match="unknown joint spec"):
        J.joint_spec("structural_bart")
    with pytest.raises(TypeError, match="unknown joint_spec overrides"):
        J.joint_spec("structural_sv", tails="t")
    with pytest.raises(ValueError, match="live paths"):
        J.joint_spec("structural_sv", live={"eq.nonsense"})
    spec = J.joint_spec(
        "structural_sv",
        T_eff=37,
        mixture="ksc7",
        prior_updates={"eq.vol_core.b_mu": np.log(0.01), "eq.vol_core.log_c": np.log(0.1)},
        sampler_prior_updates={"eq.vol_core.B_sigma": 0.2},
    )  # mutants M16 (offset) and M5 (sampler-only B_σ) are expressible
    assert spec.static.shapes.T_pad == 64 and spec.static.vol.mixture == "ksc7"
    np.testing.assert_allclose(np.asarray(spec.prior.eq.vol_core.log_c), np.log(0.1))
    np.testing.assert_allclose(np.asarray(spec.prior.eq.vol_core.B_sigma), 0.1)
    np.testing.assert_allclose(np.asarray(spec.sampler_prior.eq.vol_core.B_sigma), 0.2)
    with pytest.raises(KeyError, match="no prior leaves"):
        J.update_prior(spec.prior, {"eq.vol_core.nonsense": 1.0})
    a = J.FixedDesign(np.zeros((1, 2)), 20)
    assert a == J.FixedDesign(np.zeros((1, 2)), 20, 2.0) and hash(a) == hash(
        J.FixedDesign(np.zeros((1, 2)), 20)
    )
    assert a != J.FixedDesign(np.ones((1, 2)), 20)
    with pytest.raises(ValueError, match="T_eff"):
        J.FixedDesign(np.zeros((1, 2)), 0)


def _fake_production(calls):
    def factory(static, overrides=()):
        calls.append(dict(overrides))

        def prepare(data, prior):
            return FitCache.empty()

        def sweep(key, it, st, data, prior, cache):
            calls.append({"prior": prior, "it": it})
            bump = lambda x: x + 1 if jnp.issubdtype(x.dtype, jnp.floating) else x  # noqa: E731
            return jax.tree.map(bump, st)

        return prepare, sweep

    return factory


@pytest.mark.parametrize(
    ("name", "live"),
    [("equation_only", ("eq.mean", "eq.struct")), ("cccm_only", ("glob.mean",)),
     ("contemp_only", ("eq.struct",)), ("transitory_known_lambda", ("eq.vol.trans",)),
     ("structural_sv", ("eq", "glob"))],
)  # fmt: skip
def test_step_holds_frozen_blocks(monkeypatch, name, live):
    calls: list = []
    monkeypatch.setattr(J, "production_sweep", _fake_production(calls))
    spec = J.joint_spec(name)
    st = spec.prior_sample(random.key(0))
    data = spec.simulate(random.key(1), st)
    new = jax.jit(spec.step)(random.key(2), jnp.uint32(5), st, data)
    old_leaves = jax.tree_util.tree_flatten_with_path(st)[0]
    for (path, old), new_leaf in zip(old_leaves, jax.tree.leaves(new), strict=True):
        p = leaf_path(path)
        if not jnp.issubdtype(old.dtype, jnp.floating) or p.startswith("status"):
            continue
        moved = bool(jnp.any(new_leaf != old))
        assert moved == any(p == x or p.startswith(x + ".") for x in live), p
    if name == "contemp_only":
        assert calls[0] == {"cccm_core": J.identity_cccm_core}
    else:
        assert calls[0] == {}


def test_identity_cccm_core_and_sampler_prior(monkeypatch):
    Pi = jnp.arange(6.0).reshape(3, 2)
    out, info = J.identity_cccm_core(Pi, jnp.zeros((2, 1)), gram="packed", precompute=True)
    assert out is Pi and bool(info.ok) and isinstance(info, StepInfo)
    out2, _ = J.identity_cccm_core(Pi=Pi, a=None)
    assert out2 is Pi
    calls: list = []
    monkeypatch.setattr(J, "production_sweep", _fake_production(calls))
    spec = J.joint_spec("zero_sv", sampler_prior_updates={"eq.vol_core.B_sigma": 0.2})
    st = spec.prior_sample(random.key(0))
    spec.step(random.key(1), jnp.uint32(0), st, spec.simulate(random.key(2), st))
    used = calls[1]["prior"]
    np.testing.assert_allclose(np.asarray(used.eq.vol_core.B_sigma), 0.2)
    np.testing.assert_allclose(np.asarray(spec.prior.eq.vol_core.B_sigma), 0.1)
    assert calls[1]["it"].dtype == jnp.uint32


def test_hold_frozen_all_is_identity():
    spec = J.joint_spec("structural_sv")
    a = spec.prior_sample(random.key(0))
    b = spec.prior_sample(random.key(1))
    assert J.hold_frozen(a, b, J.ALL) is a


def test_production_step_when_available():
    spec = J.joint_spec("structural_constant")
    try:
        J.production_sweep(spec.static, spec.overrides)
    except NotImplementedError as err:
        pytest.skip(f"production sweep not available yet (WP10–WP14): {err}")
    st = spec.prior_sample(random.key(0))
    step = jax.jit(spec.step)
    for it in range(3):
        data = spec.simulate(random.fold_in(random.key(1), it), st)
        st = step(random.key(2), jnp.uint32(it), st, data)
    assert check_state_invariants(_np(st), spec.static, spec.fixed.T_eff) == []
