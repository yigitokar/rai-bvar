"""Engine data and design caches (DESIGN §2.3, §2.5; WP5): equality with the Phase 1 design
(``data/design.py`` is the oracle), exact zero padding, packed products, the D3 ``X M_c`` identity,
and jit with a traced ``T_eff`` (one compile per bucket)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rai_bvar.data import VARData
from rai_bvar.data.design import lagged_design, n_deterministic, regressors_from_history
from rai_bvar.engines.mcmc import data as D
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
    bucket,
)
from rai_bvar.engines.mcmc.state import Empty, EngineData, FitCache, assert_float64

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

DETERMINISTIC = ["n", "c", "ct"]
LEAVES = ("Y", "X", "tmask", "T_eff", "hist", "t_last")


def _np(ed: EngineData) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(ed, name)) for name in LEAVES}


def _series(T: int, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, n))


def _window(series: np.ndarray, p: int, T_eff: int, T_pad: int, n_det: int) -> EngineData:
    """``build_engine_data`` on ``series[:p + T_eff]`` with junk (later values, NaN) in the padding."""
    y = np.full((T_pad, series.shape[1]), np.nan)
    take = min(T_pad, series.shape[0] - p)
    y[:take] = series[p : p + take]
    return D.build_engine_data(y, series[:p], T_eff, n_det=n_det, t0=p + 1.0)


def _vardata(values: np.ndarray, start: str = "1990Q1") -> VARData:
    index = pd.period_range(start, periods=values.shape[0], freq="Q")
    frame = pd.DataFrame(values, index=index, columns=[f"v{j}" for j in range(values.shape[1])])
    return VARData.from_frame(frame)


# --------------------------------------------------------------------------- packed index / products


@pytest.mark.parametrize("q", [1, 2, 5, 13])
def test_packed_index_is_column_major(q):
    iu, ju, pos = D.packed_index(q)
    P = q * (q + 1) // 2
    assert iu.shape == ju.shape == (P,) and pos.shape == (q, q)
    assert iu.dtype == ju.dtype == pos.dtype == np.int32
    assert (iu <= ju).all()
    pairs = list(zip(ju.tolist(), iu.tolist(), strict=True))
    assert pairs == sorted(pairs) and len(set(pairs)) == P  # sorted by s, then r; each pair once
    np.testing.assert_array_equal(np.arange(P), ju * (ju + 1) // 2 + iu)  # c = s(s+1)/2 + r
    np.testing.assert_array_equal(pos, pos.T)
    np.testing.assert_array_equal(pos[iu, ju], np.arange(P))


def test_packed_index_prefix_covers_leading_block():
    iu, ju, pos = D.packed_index(9)
    for q in range(1, 9):
        P = q * (q + 1) // 2
        jiu, jju, jpos = D.packed_index(q)
        np.testing.assert_array_equal(iu[:P], jiu)
        np.testing.assert_array_equal(ju[:P], jju)
        np.testing.assert_array_equal(pos[:q, :q], jpos)


def test_packed_index_is_cached_and_read_only():
    a, b = D.packed_index(6), D.packed_index(6)
    assert all(x is y for x, y in zip(a, b, strict=True))
    for arr in a:
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1


@pytest.mark.parametrize("q", [0, -1, 2.0, True, None])
def test_packed_index_rejects_bad_q(q):
    with pytest.raises(ValueError, match="positive integer"):
        D.packed_index(q)


def test_packed_products_equal_explicit_products():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((37, 7))
    F[30:] = 0.0
    FF = np.asarray(D.packed_products(F))
    iu, ju, pos = D.packed_index(7)
    assert FF.shape == (37, 28) and FF.dtype == np.float64
    for r in range(7):
        for s in range(7):
            np.testing.assert_array_equal(FF[:, pos[r, s]], F[:, r] * F[:, s])
    np.testing.assert_array_equal(FF, F[:, iu] * F[:, ju])
    assert (FF[30:] == 0).all()


def test_packed_products_batch_axes_and_vmap():
    F = np.random.default_rng(2).standard_normal((3, 11, 5))
    batched = np.asarray(D.packed_products(F))
    vmapped = np.asarray(jax.vmap(D.packed_products)(F))
    assert batched.shape == (3, 11, 15)
    for b in range(3):
        np.testing.assert_array_equal(batched[b], np.asarray(D.packed_products(F[b])))
    np.testing.assert_array_equal(vmapped, batched)


def test_packed_gram_matches_dense_weighted_gram():
    rng = np.random.default_rng(3)
    F = rng.standard_normal((64, 12)) * np.logspace(-3, 3, 12)
    w = rng.uniform(0.0, 5.0, 64)
    w[50:] = 0.0
    _, _, pos = D.packed_index(12)
    G_packed = (w @ np.asarray(D.packed_products(F)))[pos]
    G_dense = F.T @ (w[:, None] * F)
    np.testing.assert_allclose(G_packed, G_dense, rtol=0, atol=1e-13 * np.abs(G_dense).max())
    np.testing.assert_array_equal(G_packed, G_packed.T)


def test_packed_products_agree_with_the_linalg_unpacking():
    """``FF`` (this module) and ``gram_from_packed`` (WP4) share one packed order."""
    from rai_bvar.engines.mcmc import linalg

    gram_from_packed = getattr(linalg, "gram_from_packed", None)
    if gram_from_packed is None:
        pytest.skip("linalg.gram_from_packed is not available yet (WP4)")
    rng = np.random.default_rng(12)
    F = rng.standard_normal((40, 9))
    w = rng.uniform(0.0, 2.0, 40)
    G = np.asarray(gram_from_packed(w @ np.asarray(D.packed_products(F)), 9))
    G_dense = F.T @ (w[:, None] * F)
    np.testing.assert_allclose(G, G_dense, rtol=0, atol=1e-13 * np.abs(G_dense).max())


# --------------------------------------------------------------------------- host builder


@pytest.mark.parametrize("t_bucket", [1, 32])
@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize("lags", [1, 2, 4])
def test_engine_data_from_vardata_matches_lagged_design(sim_data, lags, deterministic, t_bucket):
    values = sim_data.values
    Y, X = lagged_design(values, lags, deterministic)
    ed, index = D.engine_data_from_vardata(sim_data, lags, deterministic, t_bucket)
    T_eff = Y.shape[0]
    T_pad = bucket(T_eff, t_bucket)
    arr = _np(ed)
    assert all(isinstance(getattr(ed, name), np.ndarray) for name in LEAVES)  # host leaves
    assert arr["Y"].shape == (T_pad, values.shape[1]) and arr["X"].shape == (T_pad, X.shape[1])
    np.testing.assert_array_equal(arr["Y"][:T_eff], Y)
    np.testing.assert_array_equal(arr["X"][:T_eff], X)
    assert (arr["Y"][T_eff:] == 0).all() and (arr["X"][T_eff:] == 0).all()
    np.testing.assert_array_equal(arr["tmask"], np.arange(T_pad) < T_eff)
    assert arr["T_eff"].dtype == np.int32 and arr["T_eff"].shape == () and arr["T_eff"] == T_eff
    np.testing.assert_array_equal(arr["hist"], values[-lags:])
    assert arr["t_last"].shape == () and arr["t_last"] == len(values)
    assert index.equals(sim_data.index[lags:]) and len(index) == T_eff
    assert index[0] == sim_data.first + lags and index[-1] == sim_data.last
    assert_float64(ed)


def test_engine_data_from_vardata_rolling_window_restarts_the_trend(sim_data):
    window = sim_data.tail(50)
    ed, index = D.engine_data_from_vardata(window, 2, "ct", 32)
    _, X = lagged_design(window.values, 2, "ct")
    np.testing.assert_array_equal(ed.X[:48], X)
    np.testing.assert_array_equal(ed.X[:48, 1], np.arange(3.0, 51.0))  # t0 = lags + 1
    assert ed.t_last == 50.0 and ed.Y.shape[0] == 64
    assert index.equals(window.index[2:])


def test_engine_data_from_vardata_validation(sim_data):
    with pytest.raises(ValueError, match="lags"):
        D.engine_data_from_vardata(sim_data, 0, "c", 32)
    with pytest.raises(TypeError, match="lags"):
        D.engine_data_from_vardata(sim_data, True, "c", 32)
    with pytest.raises(ValueError, match="deterministic"):
        D.engine_data_from_vardata(sim_data, 1, "trend", 32)
    with pytest.raises(ValueError, match="t_bucket"):
        D.engine_data_from_vardata(sim_data, 1, "c", 0)
    short = _vardata(_series(3, 2))
    with pytest.raises(ValueError, match="more than"):
        D.engine_data_from_vardata(short, 3, "c", 32)


# --------------------------------------------------------------------------- jnp builder


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize("lags", [1, 3])
def test_build_engine_data_equals_host_builder(sim_data, lags, deterministic):
    host, _ = D.engine_data_from_vardata(sim_data, lags, deterministic, 32)
    values = sim_data.values
    T_eff = int(host.T_eff)
    dev = _window(values, lags, T_eff, host.Y.shape[0], n_deterministic(deterministic))
    got, want = _np(dev), _np(host)
    for name in LEAVES:
        assert got[name].dtype == want[name].dtype, name
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    assert_float64(dev)


def test_build_engine_data_padding_is_exactly_zero_whatever_the_input():
    series = _series(30, 3, seed=4)
    y = np.zeros((32, 3))
    y[:20] = series[2:22]
    y[20:] = np.array([np.nan, np.inf, -np.inf])  # junk a simulator may leave behind
    ed = D.build_engine_data(y, series[:2], 20, n_det=2, t0=3.0)
    Y, X = np.asarray(ed.Y), np.asarray(ed.X)
    assert np.isfinite(Y).all() and np.isfinite(X).all()
    assert (Y[20:] == 0).all() and (X[20:] == 0).all()
    assert not np.signbit(Y[20:]).any() and not np.signbit(X[20:]).any()  # +0.0, not −0.0
    want_Y, want_X = lagged_design(series[:22], 2, "ct")
    np.testing.assert_array_equal(Y[:20], want_Y)
    np.testing.assert_array_equal(X[:20], want_X)


def test_build_engine_data_hist_reaches_into_the_presample():
    presample = np.arange(12.0).reshape(4, 3)
    y = 100.0 + np.arange(24.0).reshape(8, 3)
    ed = D.build_engine_data(y, presample, 2, n_det=1, t0=5.0)
    np.testing.assert_array_equal(ed.hist, np.vstack([presample[2:], y[:2]]))
    assert float(ed.t_last) == 6.0
    np.testing.assert_array_equal(
        ed.X[1], [1.0, *y[0], *presample[3], *presample[2], *presample[1]]
    )


def test_build_engine_data_traced_T_eff_compiles_once_per_bucket():
    p, n, T_pad = 2, 3, 32
    series = _series(p + 2 * T_pad, n, seed=5)
    traces: list[int] = []

    def build(y, presample, T_eff, t0):
        traces.append(T_pad)
        return D.build_engine_data(y, presample, T_eff, n_det=2, t0=t0)

    jitted = jax.jit(build)
    y = series[p : p + T_pad]  # rows ≥ T_eff hold later values: they must be ignored
    for T_eff in range(1, T_pad + 1):
        ed = jitted(y, series[:p], jnp.int32(T_eff), float(p + 1))
        want_Y, want_X = lagged_design(series[: p + T_eff], p, "ct")
        np.testing.assert_array_equal(ed.Y[:T_eff], want_Y)
        np.testing.assert_array_equal(ed.X[:T_eff], want_X)
        assert (np.asarray(ed.Y[T_eff:]) == 0).all() and (np.asarray(ed.X[T_eff:]) == 0).all()
        np.testing.assert_array_equal(ed.hist, series[T_eff : T_eff + p])
        assert float(ed.t_last) == p + T_eff  # = len(values) of the window
        np.testing.assert_array_equal(_window(series, p, T_eff, T_pad, 2).X, ed.X)  # = eager
    assert len(traces) == 1  # one compile for the whole bucket (T_eff and t0 traced)
    jitted(series[p : p + 2 * T_pad], series[:p], jnp.int32(40), float(p + 1))
    assert len(traces) == 2  # the next bucket is a new shape


def test_build_engine_data_vmaps_over_T_eff():
    p, T_pad = 3, 32
    series = _series(p + T_pad, 2, seed=6)
    T_effs = jnp.array([1, 5, 31, 32], dtype=jnp.int32)
    batched = jax.vmap(lambda T: D.build_engine_data(series[p:], series[:p], T, n_det=1, t0=4.0))(
        T_effs
    )
    for b, T_eff in enumerate(T_effs.tolist()):
        one = _np(D.build_engine_data(series[p:], series[:p], T_eff, n_det=1, t0=4.0))
        for name in LEAVES:
            np.testing.assert_array_equal(np.asarray(getattr(batched, name))[b], one[name])


def test_build_engine_data_validation():
    y, pre = np.zeros((8, 2)), np.zeros((2, 2))
    for T_eff in (0, 9):
        with pytest.raises(ValueError, match="T_eff"):
            D.build_engine_data(y, pre, T_eff, n_det=1, t0=3.0)
    with pytest.raises(TypeError, match="T_eff"):
        D.build_engine_data(y, pre, 4.0, n_det=1, t0=3.0)
    with pytest.raises(TypeError, match="T_eff"):
        D.build_engine_data(y, pre, np.array([4]), n_det=1, t0=3.0)
    with pytest.raises(ValueError, match="presample"):
        D.build_engine_data(y, np.zeros((2, 3)), 4, n_det=1, t0=3.0)
    with pytest.raises(ValueError, match="p >= 1"):
        D.build_engine_data(y, np.zeros((0, 2)), 4, n_det=1, t0=3.0)
    for n_det in (3, -1, True, "c"):
        with pytest.raises(ValueError, match="n_det"):
            D.build_engine_data(y, pre, 4, n_det=n_det, t0=3.0)
    with pytest.raises(ValueError, match="t0"):
        D.build_engine_data(y, pre, 4, n_det=1, t0=np.array([3.0]))


# --------------------------------------------------------------------------- forecast regressors


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize("lags", [1, 2, 4])
def test_regressors_from_window_rebuild_every_design_row(lags, deterministic):
    values = _series(25, 3, seed=7)
    _, X = lagged_design(values, lags, deterministic)
    n_det = n_deterministic(deterministic)
    for row in range(X.shape[0]):
        t = row + lags  # 0-based period of the target; trend = t + 1 = t0 + row
        window = values[t - lags : t]
        x = np.asarray(D.regressors_from_window(window, lags + 1.0 + row, n_det))
        np.testing.assert_array_equal(x, X[row])
        np.testing.assert_array_equal(x, regressors_from_history(window, t + 1, deterministic))


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_regressors_from_window_continue_the_trend(deterministic):
    """``x_{T+h}`` from ``(hist, t_last + h)`` equals the Phase 1 regressors (SPEC §4.1, §5.4)."""
    values = _series(20, 3, seed=8)
    lags, n_det = 2, n_deterministic(deterministic)
    ed, _ = D.engine_data_from_vardata(_vardata(values), lags, deterministic, 32)
    extended = np.vstack([values, np.full((1, 3), np.nan)])
    _, X_ext = lagged_design(extended, lags, deterministic)
    x_next = np.asarray(D.regressors_from_window(ed.hist, ed.t_last + 1, n_det))
    np.testing.assert_array_equal(x_next, X_ext[-1])
    for h in range(1, 6):
        np.testing.assert_array_equal(
            np.asarray(D.regressors_from_window(ed.hist, ed.t_last + h, n_det)),
            regressors_from_history(values[-lags:], len(values) + h, deterministic),
        )
    if deterministic == "ct":
        assert x_next[1] == 21.0


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_regressors_from_window_batched_and_vmapped(deterministic):
    history = np.random.default_rng(9).standard_normal((4, 5, 2, 3))  # (chain, draw, p, n)
    n_det = n_deterministic(deterministic)
    out = np.asarray(D.regressors_from_window(history, 31.0, n_det))
    assert out.shape == (4, 5, n_det + 6)
    np.testing.assert_array_equal(out, regressors_from_history(history, 31, deterministic))
    t_next = 31.0 + np.arange(5.0)
    vm = jax.vmap(jax.vmap(lambda w, t: D.regressors_from_window(w, t, n_det)), (0, None))(
        history, t_next
    )
    for d in range(5):
        np.testing.assert_array_equal(
            np.asarray(vm)[:, d], regressors_from_history(history[:, d], t_next[d], deterministic)
        )


def test_regressors_from_window_validation():
    with pytest.raises(ValueError, match="n_det"):
        D.regressors_from_window(np.zeros((2, 3)), 1.0, 3)
    with pytest.raises(ValueError, match="window"):
        D.regressors_from_window(np.zeros(3), 1.0, 1)


# --------------------------------------------------------------------------- structural design


def _structural_case(n=4, p=2, n_det=1, T_eff=27, T_pad=32, seed=10):
    rng = np.random.default_rng(seed)
    series = rng.standard_normal((p + T_pad, n)) * np.array([1.0, 30.0, 0.01, 3.0])[:n]
    ed = _window(series, p, T_eff, T_pad, n_det)
    k = n_det + n * p
    Mc = rng.standard_normal((k, n - 1)) * 0.3
    return ed, Mc, k


def test_structural_design_layout_and_padding():
    ed, Mc, k = _structural_case()
    F = np.asarray(D.structural_design(ed, Mc))
    X, Y, m = np.asarray(ed.X), np.asarray(ed.Y), np.asarray(ed.tmask)
    n = Y.shape[1]
    assert F.shape == (32, k + n - 1) and F.dtype == np.float64
    np.testing.assert_array_equal(F[:, :k], X)
    want = (X @ Mc - Y[:, : n - 1]) * m[:, None]
    np.testing.assert_allclose(F[:, k:], want, rtol=1e-14, atol=1e-14 * np.abs(want).max())
    assert (F[27:] == 0).all() and not np.signbit(F[27:]).any()


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_structural_design_XMc_identity(i):
    """``Fθ_i = Xγ_i − Y_{:, :n−1}a_i`` with ``γ_i = δ_i + M_c a_i``, so ``y_i − Fθ_i`` is the
    structural residual ``y_i + Σ_{j<i} a_ij y_j − x'γ_i`` (D3; DESIGN §3.2)."""
    ed, Mc, k = _structural_case()
    rng = np.random.default_rng(11 + i)
    F = np.asarray(D.structural_design(ed, Mc))
    X, Y = np.asarray(ed.X), np.asarray(ed.Y)
    n = Y.shape[1]
    delta = rng.standard_normal(k)
    a = rng.standard_normal(n - 1) * (np.arange(n - 1) < i)  # active slots j < i only
    theta = np.concatenate([delta, a])
    gamma = delta + Mc @ a
    fitted = F @ theta
    want = X @ gamma - Y[:, : n - 1] @ a
    np.testing.assert_allclose(fitted, want, rtol=0, atol=1e-12 * np.abs(want).max())
    resid = Y[:, i] - fitted
    structural = Y[:, i] + Y[:, :i] @ a[:i] - X @ gamma
    np.testing.assert_allclose(resid, structural, rtol=0, atol=1e-12 * np.abs(structural).max())


def test_structural_design_single_variable_is_X():
    ed, _, k = _structural_case(n=1)
    F = np.asarray(D.structural_design(ed, np.zeros((k, 0))))
    np.testing.assert_array_equal(F, np.asarray(ed.X))


def test_structural_design_rejects_bad_Mc():
    ed, Mc, _ = _structural_case()
    with pytest.raises(ValueError, match="Mc"):
        D.structural_design(ed, Mc[:, :-1])


# --------------------------------------------------------------------------- base cache


def _static(n, p, n_det, T_pad, *, mean, vol, struct=None):
    return StaticConfig(
        shapes=Shapes.of(n, p, n_det, T_pad),
        struct=struct or StructStatic(),
        mean=mean,
        coef=CoefPriorStatic(),
        vol=vol,
    )


SV = VolStatic(persistent="sv")
CONST = VolStatic(persistent="constant")
CONST_COUPLED = VolStatic(persistent="constant", coupled=True)
LAMBDA = StructStatic(contemp_scaling="lambda")

# (mean, vol, struct) → present FitCache leaves (DESIGN §2.5)
COMPOSITIONS = {
    "structural_sv_packed": (MeanStatic("linear_structural"), SV, StructStatic(), {"F", "FF"}),
    "structural_sv_tiled": (
        MeanStatic("linear_structural", gram="packed_tiled", gram_tile=512),
        SV,
        StructStatic(),
        {"F"},
    ),
    "structural_sv_batched": (
        MeanStatic("linear_structural", gram="batched"),
        SV,
        StructStatic(),
        {"F"},
    ),
    # the Constant fast paths read only G_const and FtY: no FF even with gram="packed" (§3.7.2)
    "mode1_eigen": (
        MeanStatic("linear_structural", const_path="eigen"),
        CONST,
        StructStatic(),
        {"F", "G_const", "FtY"},
    ),
    "mode2_fixed_factor": (
        MeanStatic("linear_structural", coef_scaling="lambda", const_path="fixed_factor"),
        CONST_COUPLED,
        LAMBDA,
        {"F", "G_const", "FtY"},
    ),
    "iw_equivalent_cholesky_packed": (
        MeanStatic("linear_structural", const_path="cholesky_const"),
        CONST_COUPLED,
        LAMBDA,
        {"F", "G_const", "FtY"},
    ),
    "iw_equivalent_cholesky_batched": (
        MeanStatic("linear_structural", const_path="cholesky_const", gram="batched"),
        CONST_COUPLED,
        LAMBDA,
        {"F", "G_const", "FtY"},
    ),
    "structural_constant_weighted": (
        MeanStatic("linear_structural"),
        CONST,
        StructStatic(),
        {"F", "FF"},
    ),
    "reduced_sv_packed": (MeanStatic("linear_reduced"), SV, StructStatic(), {"XX"}),
    "reduced_sv_batched": (MeanStatic("linear_reduced", gram="batched"), SV, StructStatic(), set()),
    "zero_sv": (MeanStatic("zero"), SV, StructStatic(), set()),
}
CACHE_LEAVES = ("F", "FF", "XX", "G_const", "FtY")


@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_base_cache_leaves_by_composition(name):
    mean, vol, struct, present = COMPOSITIONS[name]
    ed, Mc, k = _structural_case()
    static = _static(4, 2, 1, 32, mean=mean, vol=vol, struct=struct)
    cache = D.base_cache(ed, Mc if mean.kind == "linear_structural" else None, static)
    assert isinstance(cache, FitCache) and isinstance(cache.eq, Empty)
    got = {leaf for leaf in CACHE_LEAVES if not isinstance(getattr(cache, leaf), Empty)}
    assert got == present
    s = static.shapes
    shapes = {
        "F": (s.T_pad, s.q),
        "FF": (s.T_pad, s.P),
        "XX": (s.T_pad, s.P_x),
        "G_const": (s.q, s.q),
        "FtY": (s.q, s.n),
    }
    for leaf in present:
        assert getattr(cache, leaf).shape == shapes[leaf], leaf
    assert_float64(cache)
    # the reduced and zero means ignore Mc (SharedPrior.Mc is zeros there)
    if mean.kind != "linear_structural":
        again = D.base_cache(ed, np.zeros((k, 3)), static)
        assert jax.tree.structure(again) == jax.tree.structure(cache)


def test_base_cache_values_against_dense_oracles():
    mean, vol, struct, _ = COMPOSITIONS["mode1_eigen"]
    ed, Mc, _ = _structural_case()
    static = _static(4, 2, 1, 32, mean=mean, vol=vol, struct=struct)
    cache = D.base_cache(ed, Mc, static)
    F = np.asarray(cache.F)
    np.testing.assert_array_equal(F, np.asarray(D.structural_design(ed, Mc)))
    # the weighted oracle of the same composition (const_path="weighted") carries FF instead
    oracle_static = _static(
        4, 2, 1, 32, mean=dataclasses.replace(mean, const_path="weighted"), vol=vol, struct=struct
    )
    oracle = D.base_cache(ed, Mc, oracle_static)
    np.testing.assert_array_equal(np.asarray(oracle.F), F)
    assert isinstance(oracle.G_const, Empty) and isinstance(oracle.FtY, Empty)
    np.testing.assert_array_equal(np.asarray(oracle.FF), np.asarray(D.packed_products(F)))
    m, Y = np.asarray(ed.tmask), np.asarray(ed.Y)
    G = np.asarray(cache.G_const)
    G_dense = F.T @ np.diag(m) @ F
    np.testing.assert_allclose(G, G_dense, rtol=0, atol=1e-13 * np.abs(G_dense).max())
    np.testing.assert_array_equal(G, G.T)  # exactly symmetric (eigh / Cholesky inputs)
    FtY_dense = F.T @ np.diag(m) @ Y
    np.testing.assert_allclose(
        np.asarray(cache.FtY), FtY_dense, rtol=0, atol=1e-13 * np.abs(FtY_dense).max()
    )
    _, _, pos = D.packed_index(static.shapes.q)
    np.testing.assert_allclose(
        (m @ np.asarray(oracle.FF))[pos], G_dense, rtol=0, atol=1e-13 * np.abs(G_dense).max()
    )


def test_packed_products_of_F_start_with_those_of_X():
    """Column-major order: ``FF[:, :P_x]`` is exactly the reduced-form ``XX`` (``F[:, :k] = X``)."""
    ed, Mc, _ = _structural_case()
    structural = _static(4, 2, 1, 32, mean=MeanStatic("linear_structural"), vol=SV)
    reduced = _static(4, 2, 1, 32, mean=MeanStatic("linear_reduced"), vol=SV)
    FF = np.asarray(D.base_cache(ed, Mc, structural).FF)
    XX = np.asarray(D.base_cache(ed, None, reduced).XX)
    P_x = structural.shapes.P_x
    np.testing.assert_array_equal(FF[:, :P_x], XX)
    np.testing.assert_array_equal(XX, np.asarray(D.packed_products(ed.X)))
    assert (XX[27:] == 0).all() and (FF[27:] == 0).all()


def test_base_cache_under_jit_with_traced_T_eff():
    mean, vol, struct, _ = COMPOSITIONS["mode2_fixed_factor"]
    static = _static(4, 2, 1, 32, mean=mean, vol=vol, struct=struct)
    traces: list[int] = []

    def prepare(ed, Mc):
        traces.append(1)
        return D.base_cache(ed, Mc, static)

    jitted = jax.jit(prepare)
    for T_eff in (9, 20, 32):
        ed, Mc, _ = _structural_case(T_eff=T_eff)
        got = jitted(ed, Mc)
        want = D.base_cache(ed, Mc, static)
        assert jax.tree.structure(got) == jax.tree.structure(want)
        for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_allclose(g, w, rtol=1e-14, atol=1e-14 * (np.abs(w).max() + 1))
    assert len(traces) == 1


def test_base_cache_validation():
    ed, Mc, _ = _structural_case()
    static = _static(4, 2, 1, 32, mean=MeanStatic("linear_structural"), vol=SV)
    with pytest.raises(ValueError, match="Mc"):
        D.base_cache(ed, None, static)
    with pytest.raises(ValueError, match=r"StaticConfig\.shapes"):
        D.base_cache(ed, Mc, _static(4, 2, 1, 64, mean=MeanStatic("linear_structural"), vol=SV))
    with pytest.raises(ValueError, match="hist"):
        D.base_cache(dataclasses.replace(ed, hist=ed.hist[:1]), Mc, static)
    with pytest.raises(ValueError, match="X"):
        D.base_cache(ed, Mc, _static(4, 2, 2, 32, mean=MeanStatic("linear_structural"), vol=SV))


def test_end_to_end_window_to_cache(sim_data):
    """Host window → cache, the path ``resolve.py`` takes (WP18)."""
    ed, index = D.engine_data_from_vardata(sim_data, 2, "c", 32)
    n = sim_data.n_vars
    static = _static(n, 2, 1, ed.Y.shape[0], mean=MeanStatic("linear_structural"), vol=SV)
    Mc = np.zeros((static.shapes.k, n - 1))
    Mc[1 : 1 + n, :] = 0.9 * np.eye(n)[:, : n - 1]  # m_j: 0.9 on the first lag of variable j
    cache = jax.jit(lambda d, M: D.base_cache(d, M, static))(ed, Mc)
    F, X, Y = np.asarray(cache.F), ed.X, ed.Y
    k = static.shapes.k
    want = 0.9 * X[:, 1:n] - Y[:, : n - 1]  # x_t'm_j − y_{j,t} = 0.9·y_{j,t−1} − y_{j,t}
    np.testing.assert_allclose(F[:, k:], want, rtol=0, atol=1e-14 * np.abs(want).max())
    assert len(index) == int(ed.T_eff)


def test_import_leaves_the_backend_uninitialised():
    """No module-level ``jnp`` array (DESIGN §1.1 rule 3): ``set_cpu_devices`` still works after
    importing this module, and the packed maps are NumPy constants."""
    code = """
        import numpy as np
        from rai_bvar.engines.mcmc import data
        from rai_bvar._devices import set_cpu_devices
        assert isinstance(data.packed_index(5)[2], np.ndarray)
        set_cpu_devices(2)
        import jax
        assert jax.local_device_count() == 2, jax.devices()
        print("ok")
        """
    skip = {"JAX_ENABLE_X64", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES"}
    env = {k: v for k, v in os.environ.items() if k not in skip} | {"JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stdout + proc.stderr
