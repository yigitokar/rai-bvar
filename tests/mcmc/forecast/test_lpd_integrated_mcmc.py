"""The path-integrated LPD on MCMC forecasts and the ``lag_coefs`` extra (DESIGN §7.3, D-34).

``lag_coefs[c, d, l−1] = A_l`` of the parameter draw behind every forecast slot: equal to
``fit.reduced_form_draws`` mapped through the §7.1 draw selection (post-hoc, replicate slots
``R > 1`` and in-loop forecasts; 1e-11) and, on synthetic draws, to ``B`` written out with NumPy
for the structural, reduced (CCCM) and zero means; other mean kinds attach nothing. Per slot,
the integrated moments of ``y_{T+h}`` equal a brute-force oracle built from the forecast core
alone: the zero-shock path ``simulate_paths(ε = 0)`` and ``M_h = ∂paths_h/∂ε_{1..h}``
(``jax.jacfwd``) at the slot's drawn scale path, so ``Cov = M_h M_hᵀ`` (1e-10 in log density);
the forecast's integrated LPD is the logsumexp of those densities. Fits: structural × SV with
outliers (``p = 1``), reduced CCCM × SV with outliers (``p = 2``, constant and trend) and
structural × Constant (``p = 2``, constant and trend, thinned storage, in-loop forecasts). The
default estimator is bitwise unchanged by ``lag_coefs``, and ``h = 1`` is the conditioned
estimator bitwise (``rb_transitory`` included).
"""

from __future__ import annotations

import functools
import types

import jax
import jax.numpy as jnp
import mcmc_fits as mf
import numpy as np
import pytest
from scipy.special import logsumexp

from rai_bvar import SV, Constant
from rai_bvar.engines.mcmc import forecast as F
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.state import (
    CCCMDraw,
    ConstantDraw,
    Empty,
    GlobDraw,
    LinearDraw,
    ParamDraw,
    StructDraw,
    VolDraw,
)
from rai_bvar.forecast import ForecastResult
from rai_bvar.forecast.result import LAG_DIMS
from rai_bvar.settings import FitPlan

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

PLAN = FitPlan(horizon=4, draws=30, seed=5)
FITS = {
    "structural-SV(outliers)-p1-c": lambda data: mf.shared_fit(data),
    "reduced-SV(outliers)-p2-ct": lambda data: mf.cached_fit(
        mf.model("reduced", SV(outliers=True), lags=2, deterministic="ct"), data
    ),
    # the thinned trend fit of test_forecast_core.py (one compile per process)
    "structural-Constant(ig)-p2-ct": lambda data: mf.cached_fit(
        mf.model("structural", Constant("ig"), lags=2, deterministic="ct", store_max=20),
        data,
        plan=PLAN,
    ),
}
SUBSETS = [None, ["gdp"], ["gdp", "prices"], ["prices"], ["rate"], ["rate", "prices"]]


@pytest.fixture(scope="module", params=list(FITS))
def fit(request, sim_data):
    return FITS[request.param](sim_data)


def _lags(B: np.ndarray, lags: int, n_det: int) -> np.ndarray:
    """``A_l[i, j] = B[n_det + (l−1)·n + j, i]`` for ``B (..., k, n)``."""
    n = B.shape[-1]
    out = np.empty((*B.shape[:-2], lags, n, n))
    for lag in range(lags):
        for i in range(n):
            for j in range(n):
                out[..., lag, i, j] = B[..., n_det + lag * n + j, i]
    return out


def _expected_lag_coefs(fit, slots: np.ndarray) -> np.ndarray:
    """``A_l`` of stored draws ``slots`` (positions, one per forecast slot) of every chain."""
    s = fit.static.shapes
    return _lags(fit.reduced_form_draws().values[:, slots], s.lags, s.n_det)


def test_lag_coefs_are_the_reduced_form_draws_of_each_slot(fit):
    """Post-hoc slot ``j·R + r`` of chain ``c`` comes from stored draw ``pos_j`` (DESIGN §7.1),
    with ``R = 1`` and ``R = 3`` (replicates share their draw's ``A_l``)."""
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    for draws in (10, 2 * C * D + 1):
        fr = fit.forecast(3, draws=draws, seed=3)
        assert fr.dataset.attrs["in_loop"] == 0
        da = fr.dataset["lag_coefs"]
        assert da.dims == LAG_DIMS and da.dtype == np.float64
        pos, R = F.select_draws(D, C, draws)
        assert da.shape == (C, pos.size * R, fit.static.shapes.lags, 3, 3)
        np.testing.assert_allclose(
            da.values, _expected_lag_coefs(fit, np.repeat(pos, R)), rtol=1e-11, atol=1e-13
        )


def _in_loop_slots(fit) -> np.ndarray:
    """Stored-draw positions of the in-loop forecast slots (all stored for these fits)."""
    kept = np.asarray(fit.in_loop_forecast.draw_index).reshape(-1)
    where = np.searchsorted(fit.draw_index, kept)
    np.testing.assert_array_equal(np.asarray(fit.draw_index)[where], kept)
    return where


def test_lag_coefs_of_in_loop_forecasts(sim_data):
    """In-loop forecasts (DESIGN §5.8) carry the ``A_l`` of the kept draws they came from."""
    fit = FITS["structural-Constant(ig)-p2-ct"](sim_data)
    fr = fit.forecast(PLAN.horizon, draws=PLAN.draws, seed=PLAN.seed)
    assert fr.dataset.attrs["in_loop"] == 1
    np.testing.assert_allclose(
        fr.dataset["lag_coefs"].values,
        _expected_lag_coefs(fit, _in_loop_slots(fit)),
        rtol=1e-11,
        atol=1e-13,
    )


# --------------------------------------------------------------------------- univariate fits

UNIVARIATE = {  # n = 1: struct.a is (C, D, 1, 0), an empty leaf (review finding math-1)
    "structural-SV(outliers)-p2-c": ("structural", SV(outliers=True)),
    "reduced-SV-p2-c": ("reduced", SV()),
}
PLAN_1 = FitPlan(horizon=3, draws=10, seed=5)


@pytest.fixture(scope="module", params=list(UNIVARIATE))
def fit_1(request, sim_data_stationary):
    form, vol = UNIVARIATE[request.param]
    return mf.cached_fit(mf.model(form, vol, lags=2), sim_data_stationary.select(["rate"]),
                         plan=PLAN_1)  # fmt: skip


def _scalar_integrated(fr: ForecastResult, y: float, h: int) -> float:
    """The integrated LPD of a univariate forecast with scalar recursions written out:
    ``φ_0 = 1``, ``φ_j = Σ_{l ≤ min(j, p)} a_l φ_{j−l}``, ``m = cond_mean_h − Σ_{k<h} φ_{h−k}
    u_k``, ``v = Σ_{k ≤ h} φ_{h−k}² L_k²`` per slot, then the logsumexp over slots."""
    a = fr.stacked("lag_coefs")[:, :, 0, 0]
    mean, paths = fr.stacked("cond_mean")[..., 0], fr.stacked("paths")[..., 0]
    L = fr.stacked("cond_chol")[..., 0, 0]
    logp = []
    for s in range(a.shape[0]):
        phi = [1.0]
        for j in range(1, h):
            phi.append(
                sum(a[s, lag - 1] * phi[j - lag] for lag in range(1, min(j, a.shape[1]) + 1))
            )
        m = mean[s, h - 1] - sum(
            phi[h - k] * (paths[s, k - 1] - mean[s, k - 1]) for k in range(1, h)
        )
        v = sum(phi[h - k] ** 2 * L[s, k - 1] ** 2 for k in range(1, h + 1))
        logp.append(-0.5 * (np.log(2.0 * np.pi * v) + (y - m) ** 2 / v))
    return float(logsumexp(logp) - np.log(len(logp)))


def test_univariate_forecasts_carry_lag_coefs(fit_1, monkeypatch):
    """``n = 1`` (post-hoc, ``R = 1`` and ``R = 3``, and in-loop): the forecast succeeds,
    ``lag_coefs`` is ``(C, D', p, 1, 1)`` and equals the reduced-form draws of each slot; the
    default estimator equals the forecast without ``lag_coefs`` (the pre-D-34 extras) bitwise;
    the integrated LPD at ``h = 2, 3`` equals the scalar recursions."""
    C, D = np.shape(fit_1.param_draws.struct.a)[:2]
    assert np.shape(fit_1.param_draws.struct.a)[2:] == (1, 0)
    cases = [((3,), {"draws": 10, "seed": 3}), ((3,), {"draws": 2 * C * D + 1, "seed": 3}),
             ((PLAN_1.horizon,), {"draws": PLAN_1.draws, "seed": PLAN_1.seed})]  # fmt: skip
    for args, kw in cases:
        fr = fit_1.forecast(*args, **kw)
        da = fr.dataset["lag_coefs"]
        assert da.dims == LAG_DIMS and da.shape[2:] == (2, 1, 1)
        if fr.dataset.attrs["in_loop"]:
            slots = _in_loop_slots(fit_1)
        else:
            pos, R = F.select_draws(D, C, kw["draws"])
            slots = np.repeat(pos, R)
        assert da.shape[:2] == (C, slots.size)
        np.testing.assert_allclose(
            da.values, _expected_lag_coefs(fit_1, slots), rtol=1e-11, atol=1e-13
        )
        with monkeypatch.context() as mp:  # the extras of the pre-D-34 code: no lag_coefs
            mp.setattr(F, "_lag_coefs", lambda fit, params: None)
            plain = fit_1.forecast(*args, **kw)
        assert set(fr.dataset.data_vars) == set(plain.dataset.data_vars) | {"lag_coefs"}
        for h in (1, 2, 3):
            y = (
                fr.stacked("cond_mean")[:, h - 1, 0].mean()
                + 2.0 * fr.stacked("paths")[:, h - 1, 0].std()
            )
            want = plain.log_pred_density([y], h, return_mcse=True)
            assert fr.log_pred_density([y], h, return_mcse=True) == want
            assert fr.log_pred_density([y], h, ["rate"], return_mcse=True) == want
            if h > 1:
                lpd, mcse = fr.log_pred_density([y], h, return_mcse=True, lpd_method="integrated")
                assert np.isfinite(lpd) and np.isfinite(mcse) and mcse >= 0
                np.testing.assert_allclose(
                    lpd, _scalar_integrated(fr, y, h), rtol=1e-12, atol=1e-12
                )


# --------------------------------------------------------------------------- synthetic draws

N, LAGS, NDET = 3, 2, 2


def _static(kind: str, n: int = N) -> StaticConfig:
    return StaticConfig(
        shapes=Shapes.of(n, LAGS, NDET, 32),
        struct=StructStatic(),
        mean=MeanStatic(kind=kind),
        coef=CoefPriorStatic(),
        vol=VolStatic(persistent="constant", transitory="none"),
    )


def _draws(kind: str, C: int = 2, D: int = 3, n: int = N) -> ParamDraw:
    """Random ``ParamDraw``\\ s with leaves ``(C, D, …)`` (``n = 1``: ``a`` is ``(C, D, 1, 0)``)."""
    rng = np.random.default_rng(4)
    k = NDET + n * LAGS
    a = rng.normal(0.0, 0.5, (C, D, n, n - 1)) * (np.arange(n - 1) < np.arange(n)[:, None])
    gamma, Pi = rng.normal(0.0, 0.3, (C, D, n, k)), rng.normal(0.0, 0.3, (C, D, k, n))
    return ParamDraw(
        glob=GlobDraw(
            mean=CCCMDraw(Pi=Pi) if kind == "linear_reduced" else Empty(),
            coef=Empty(),
            vol=Empty(),
            struct=Empty(),
        ),
        mean=LinearDraw(gamma=gamma) if kind == "linear_structural" else Empty(),
        struct=StructDraw(a=a),
        vol=VolDraw(core=ConstantDraw(log_lam=rng.normal(-1.0, 0.3, (C, D, n))), trans=Empty()),
    )


@pytest.mark.parametrize("n", [N, 1])
@pytest.mark.parametrize("kind", ["linear_structural", "linear_reduced", "zero"])
def test_lag_coefs_of_synthetic_draws(kind, n):
    """``B = (A⁻¹Γ)ᵀ`` (NumPy solve), ``Π`` or zeros, mapped to ``A_l`` element by element
    (``n = 1`` included: its empty ``a`` leaf must not break the flattening)."""
    static = _static(kind, n)
    params = _draws(kind, n=n)
    got = F._lag_coefs(types.SimpleNamespace(static=static, program=compose(static)), params)
    if kind == "linear_structural":
        A = np.broadcast_to(np.eye(n), (2, 3, n, n)).copy()
        rows, cols = np.tril_indices(n, -1)
        A[..., rows, cols] = np.asarray(params.struct.a)[..., rows, cols]
        B = np.swapaxes(np.linalg.solve(A, np.asarray(params.mean.gamma)), -1, -2)
    elif kind == "linear_reduced":
        B = np.asarray(params.glob.mean.Pi)
    else:
        B = np.zeros((2, 3, NDET + n * LAGS, n))
    assert got.shape == (2, 3, LAGS, n, n)
    np.testing.assert_allclose(got, _lags(B, LAGS, NDET), rtol=1e-12, atol=1e-15)
    if kind == "linear_reduced":  # Π is copied, not recomputed
        np.testing.assert_array_equal(got, _lags(B, LAGS, NDET))


def test_lag_coefs_are_not_attached_for_other_mean_kinds():
    fake = types.SimpleNamespace(static=types.SimpleNamespace(mean=types.SimpleNamespace(
        kind="bart")))  # fmt: skip
    assert F._lag_coefs(fake, _draws("zero")) is None


# --------------------------------------------------------------------------- integrated moments

H_ORACLE = 5


@functools.cache
def _oracle_fns(program):
    """Jitted zero-shock path and ``∂paths/∂ε`` of :func:`F.simulate_paths` (forecast core)."""

    def paths(params, hist, t_last, eps, log_lam):
        return F.simulate_paths(params, hist, t_last, eps, log_lam, program)[0]

    return jax.jit(paths), jax.jit(jax.jacfwd(paths, argnums=3))


def _oracle(fit, fr: ForecastResult):
    """Per slot ``(c, j)``: the zero-shock mean ``(H, n)`` and the Jacobian ``(H, n, H, n)`` at
    its stored draw and its drawn effective scale path ``2 log cond_scale``."""
    C, D = np.shape(fit.param_draws.struct.a)[:2]
    pos, R = F.select_draws(D, C, fr.n_samples)
    assert R == 1 and fr.dataset.attrs["in_loop"] == 0
    zero_path, jac = _oracle_fns(fit.program)
    hist, t_last = fit.engine_data.hist, fit.engine_data.t_last
    eps = jnp.zeros((fr.horizon, 3))
    out = {}
    for c in range(C):
        for j, w in enumerate(pos):
            params = jax.tree.map(lambda x, c=c, w=w: np.asarray(x)[c, w], fit.param_draws)
            log_lam = 2.0 * np.log(fr.dataset["cond_scale"].values[c, j])
            m = np.asarray(zero_path(params, hist, t_last, eps, log_lam))
            J = np.asarray(jac(params, hist, t_last, eps, log_lam))
            out[c, j] = m, J
    return out


def _oracle_logpdf(m, J, h: int, idx, y) -> float:
    """``log N(y; m_h[idx], M Mᵀ)``, ``M = (∂y_{T+h}/∂ε_{1..h})[idx]``, from the SVD ``M = U diag(s)
    Vᵀ`` (``log det = 2 Σ log s``, quadratic form ``‖s⁻¹ ⊙ Uᵀ(y − m)‖²``). Forming ``M Mᵀ``
    (``scipy.stats.multivariate_normal``) squares the condition number of these fits' scales
    (gdp and prices ~1e-3, rate ~0.25) and was off by up to 4e-9 against exact rational
    arithmetic on the same ``(m, M)``, where this oracle is within 1e-12 and the estimator within
    1e-13 (a one-off check, 2026-10-23)."""
    M = J[h - 1, :, :h, :].reshape(3, 3 * h)[idx]
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    z = (U.T @ (y - m[h - 1, idx])) / s
    return float(-0.5 * (idx.size * np.log(2.0 * np.pi) + z @ z) - np.log(s).sum())


def test_integrated_moments_equal_the_jacobian_oracle(fit):
    """~20 slots, ``h = 2..5``, joint, leading and non-leading subsets: one-slot forecasts give
    the oracle density at the oracle mean and 3 s.d. away (1e-10), and the whole forecast gives
    their logsumexp; ``lpd_method="integrated"`` needs no random numbers (repeatable)."""
    fr = fit.forecast(H_ORACLE, draws=20, seed=17)
    oracle = _oracle(fit, fr)
    names = fr.variables
    for h in range(2, H_ORACLE + 1):
        for subset in SUBSETS:
            idx = np.arange(3) if subset is None else np.array([names.index(v) for v in subset])
            centre = np.mean([m[h - 1, idx] for m, _ in oracle.values()], axis=0)
            sd = fr.stacked("paths")[:, h - 1].std(axis=0)[idx]
            for y in (centre, centre + 3.0 * sd * np.array([1.0, -1.0, 0.5])[: idx.size]):
                logp = []
                for (c, j), (m, J) in oracle.items():
                    want = _oracle_logpdf(m, J, h, idx, y)
                    one = ForecastResult(fr.dataset.isel(chain=[c], draw=[j]))
                    got = one.log_pred_density(y, h, subset, lpd_method="integrated")
                    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-10)
                    logp.append(want)
                lpd = fr.log_pred_density(y, h, subset, lpd_method="integrated")
                np.testing.assert_allclose(
                    lpd, logsumexp(logp) - np.log(len(logp)), rtol=1e-12, atol=1e-10
                )
                again, mcse = fr.log_pred_density(
                    y, h, subset, return_mcse=True, lpd_method="integrated"
                )
                assert again == lpd and np.isfinite(mcse) and mcse >= 0


def test_oracle_mean_is_the_zero_shock_mean_of_the_simulated_path(fit):
    """Cross-check of the oracle itself: the simulated paths are the zero-shock path plus
    ``Σ_k (∂paths/∂ε_k) ε_k`` (linear in ``ε`` given the scale path), so ``paths − J·ε`` is
    ``m`` (1e-10) with ``ε = L⁻¹(paths − cond_mean)`` recovered from the contract fields."""
    fr = fit.forecast(H_ORACLE, draws=20, seed=17)
    oracle = _oracle(fit, fr)
    diff = fr.dataset["paths"].values - fr.dataset["cond_mean"].values
    for (c, j), (m, J) in oracle.items():
        L = fr.dataset["cond_chol"].values[c, j]
        eps = np.stack([np.linalg.solve(L[h], diff[c, j, h]) for h in range(H_ORACLE)])
        np.testing.assert_allclose(
            fr.dataset["paths"].values[c, j] - np.einsum("hikn,kn->hi", J, eps), m,
            rtol=1e-10, atol=1e-10,
        )  # fmt: skip


# --------------------------------------------------------------------------- default unchanged


def _y(fr: ForecastResult, h: int) -> np.ndarray:
    return fr.stacked("cond_mean")[:, h - 1].mean(axis=0) + fr.stacked("paths")[:, h - 1].std(
        axis=0
    ) * np.array([1.5, -2.0, 0.5])


def test_default_estimator_is_bitwise_unchanged_by_lag_coefs(fit):
    """The default (conditioned) estimator never reads ``lag_coefs``: the same values with and
    without it, MCSE included, for every subset and horizon (and the transitory RB)."""
    fr = fit.forecast(4, draws=60, seed=8)
    plain = ForecastResult(fr.dataset.drop_vars("lag_coefs"))
    names = fr.variables
    for h in range(1, 5):
        y = _y(fr, h)
        for subset in SUBSETS:
            idx = np.arange(3) if subset is None else np.array([names.index(v) for v in subset])
            want = plain.log_pred_density(y[idx], h, subset, return_mcse=True)
            assert fr.log_pred_density(y[idx], h, subset, return_mcse=True) == want
        if fr.dataset.attrs["transitory"] != "none":
            assert fr.log_pred_density(y, h, rb_transitory=True) == plain.log_pred_density(
                y, h, rb_transitory=True
            )


def test_horizon_one_integrated_is_the_conditioned_estimator(fit):
    fr = fit.forecast(4, draws=60, seed=8)
    y = _y(fr, 1)
    for subset in SUBSETS:
        idx = slice(None) if subset is None else [fr.variables.index(v) for v in subset]
        want = fr.log_pred_density(y[idx], 1, subset, return_mcse=True)
        got = fr.log_pred_density(y[idx], 1, subset, return_mcse=True, lpd_method="integrated")
        assert got == want
    if fr.dataset.attrs["transitory"] != "none":  # allowed at h = 1: the conditioned RB runs
        assert fr.log_pred_density(y, 1, rb_transitory=True, lpd_method="integrated") == (
            fr.log_pred_density(y, 1, rb_transitory=True)
        )
        with pytest.raises(ValueError, match="rb_transitory"):
            fr.log_pred_density(_y(fr, 2), 2, rb_transitory=True, lpd_method="integrated")
