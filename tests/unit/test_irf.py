"""Recursive impulse responses and FEVD (SPEC §6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from rai_bvar.structural import IRFResult, fevd, ma_coefficients, recursive_irf

VARIABLES = ["gdp", "prices", "rate"]


def _stable_lags(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Random lag matrices ``(p, n, n)`` scaled so the VAR is comfortably stationary."""
    return rng.normal(scale=0.3 / (n * p) ** 0.5, size=(p, n, n))


def _coefs(lag_mats: np.ndarray, det: np.ndarray | None = None) -> np.ndarray:
    """``B`` in the ``lagged_design`` layout from ``A_l`` (``(p, n, n)``) and deterministic rows."""
    n = lag_mats.shape[-1]
    det = np.zeros((0, n)) if det is None else det
    return np.vstack([det, *[a.T for a in lag_mats]])


def _random_chol(rng: np.random.Generator, n: int) -> np.ndarray:
    G = rng.normal(size=(n, n))
    return np.linalg.cholesky(G @ G.T + n * np.eye(n))


def _draws(seed: int, S: int, n: int = 3, p: int = 2, n_det: int = 1):
    rng = np.random.default_rng(seed)
    coefs = np.stack(
        [_coefs(_stable_lags(rng, n, p), rng.normal(size=(n_det, n))) for _ in range(S)]
    )
    chol = np.stack([_random_chol(rng, n) for _ in range(S)])
    return coefs, chol


# ---------------------------------------------------------------------- MA coefficients


def test_var1_closed_form(sim_truth):
    A = sim_truth.coefs
    phi = ma_coefficients(sim_truth.B, lags=1, n_det=1, horizon=8)
    assert phi.shape == (9, 3, 3)
    for h in range(9):
        np.testing.assert_allclose(phi[h], np.linalg.matrix_power(A, h), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("p", [2, 3])
def test_varp_matches_companion_form(p):
    rng = np.random.default_rng(1)
    n, H = 3, 12
    A = _stable_lags(rng, n, p)
    companion = np.zeros((n * p, n * p))
    companion[:n] = np.concatenate(list(A), axis=1)
    companion[n:, :-n] = np.eye(n * (p - 1))
    phi = ma_coefficients(_coefs(A, rng.normal(size=(2, n))), lags=p, n_det=2, horizon=H)
    for h in range(H + 1):
        expected = np.linalg.matrix_power(companion, h)[:n, :n]
        np.testing.assert_allclose(phi[h], expected, rtol=1e-12, atol=1e-14)


def test_lag_layout_row_i_is_equation():
    """``A_l[i, j] = coefs[n_det + (l−1)·n + j, i]``: Φ_1 = A_1 read straight from ``B``."""
    n, p, n_det = 3, 2, 2
    coefs = np.arange((n_det + n * p) * n, dtype=np.float64).reshape(n_det + n * p, n)
    phi = ma_coefficients(coefs, lags=p, n_det=n_det, horizon=1)
    for i in range(n):
        for j in range(n):
            assert phi[1, i, j] == coefs[n_det + j, i]


def test_deterministic_terms_are_ignored():
    rng = np.random.default_rng(2)
    A = _stable_lags(rng, 3, 2)
    reference = ma_coefficients(_coefs(A), lags=2, n_det=0, horizon=10)
    for n_det in (1, 2):
        det = rng.normal(scale=100.0, size=(n_det, 3))
        phi = ma_coefficients(_coefs(A, det), lags=2, n_det=n_det, horizon=10)
        np.testing.assert_array_equal(phi, reference)


def test_horizon_zero_is_identity():
    coefs, _ = _draws(3, S=4)
    phi = ma_coefficients(coefs, lags=2, n_det=1, horizon=0)
    assert phi.shape == (4, 1, 3, 3)
    np.testing.assert_array_equal(phi[:, 0], np.broadcast_to(np.eye(3), (4, 3, 3)))


def test_bad_shapes_raise():
    coefs, chol = _draws(4, S=2)
    with pytest.raises(ValueError, match="expected n_det"):
        ma_coefficients(coefs, lags=2, n_det=0, horizon=4)
    with pytest.raises(ValueError, match="horizon"):
        ma_coefficients(coefs, lags=2, n_det=1, horizon=-1)
    with pytest.raises(ValueError, match="chol"):
        recursive_irf(coefs, chol[..., :2, :2], lags=2, n_det=1, horizon=4)
    with pytest.raises(ValueError, match="shock"):
        recursive_irf(coefs, chol, lags=2, n_det=1, horizon=4, shock="one")


# ---------------------------------------------------------------------- impulse responses


def test_impact_is_chol_for_sd_shocks():
    coefs, chol = _draws(5, S=6)
    irf = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6)
    phi = ma_coefficients(coefs, lags=2, n_det=1, horizon=6)
    assert irf.shape == (6, 7, 3, 3)
    np.testing.assert_allclose(irf[:, 0], chol, rtol=1e-14)
    np.testing.assert_allclose(irf, phi @ chol[:, None], rtol=1e-12)
    assert np.all(np.triu(irf[:, 0], k=1) == 0.0)  # shock j does not move earlier variables


def test_unit_shocks_have_unit_own_impact():
    coefs, chol = _draws(6, S=6)
    sd = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, shock="sd")
    unit = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, shock="unit")
    np.testing.assert_allclose(np.diagonal(unit[:, 0], axis1=-2, axis2=-1), 1.0, rtol=1e-14)
    diag = np.diagonal(chol, axis1=-2, axis2=-1)
    np.testing.assert_allclose(unit, sd / diag[:, None, None, :], rtol=1e-12)


def test_unit_shocks_equal_phi_times_inverse_structural_matrix():
    """With ``L = A⁻¹Λ^{1/2}``, unit-shock responses are ``Φ_h A⁻¹`` whatever ``Λ`` is."""
    rng = np.random.default_rng(7)
    a_inv = np.tril(rng.normal(size=(3, 3)), k=-1) + np.eye(3)
    coefs = _coefs(_stable_lags(rng, 3, 2), rng.normal(size=(1, 3)))
    phi = ma_coefficients(coefs, lags=2, n_det=1, horizon=5)
    for lam in (np.array([1.0, 2.0, 0.5]), np.array([9.0, 0.1, 3.0])):
        chol = a_inv * np.sqrt(lam)[None, :]
        unit = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=5, shock="unit")
        np.testing.assert_allclose(unit, phi @ a_inv, rtol=1e-12, atol=1e-14)


def test_batching_over_draws_and_chains_matches_loops():
    coefs, chol = _draws(8, S=12)
    batched = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=8)
    chains = recursive_irf(
        coefs.reshape(3, 4, *coefs.shape[1:]),
        chol.reshape(3, 4, 3, 3),
        lags=2,
        n_det=1,
        horizon=8,
    )
    assert chains.shape == (3, 4, 9, 3, 3)
    for s in range(12):
        single = recursive_irf(coefs[s], chol[s], lags=2, n_det=1, horizon=8)
        assert single.shape == (9, 3, 3)
        np.testing.assert_allclose(batched[s], single, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(chains[s // 4, s % 4], single, rtol=1e-13, atol=1e-15)
    shares = fevd(chains)
    for s in range(12):
        np.testing.assert_allclose(shares[s // 4, s % 4], fevd(batched[s]), rtol=1e-13)


def test_scale_multiplies_responses():
    coefs, chol = _draws(9, S=5)
    base = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6)
    np.testing.assert_allclose(
        recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, scale=2.5), 2.5 * base, rtol=1e-13
    )
    per_draw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaled = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, scale=per_draw)
    np.testing.assert_allclose(scaled, per_draw[:, None, None, None] * base, rtol=1e-13)
    dates = np.array([1.0, 3.0])[:, None]  # (dates, 1) broadcasts against (S,)
    by_date = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, scale=dates)
    assert by_date.shape == (2, 5, 7, 3, 3)
    np.testing.assert_allclose(by_date[1], 3.0 * base, rtol=1e-13)
    unit = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, shock="unit")
    unit_scaled = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=6, shock="unit", scale=dates)
    np.testing.assert_allclose(unit_scaled, np.broadcast_to(unit, by_date.shape), rtol=1e-13)


# ---------------------------------------------------------------------- FEVD


def test_fevd_properties():
    coefs, chol = _draws(10, S=20)
    shares = fevd(recursive_irf(coefs, chol, lags=2, n_det=1, horizon=10))
    assert shares.shape == (20, 11, 3, 3)
    np.testing.assert_allclose(shares.sum(axis=-1), 1.0, rtol=1e-14)
    assert np.all((shares >= 0.0) & (shares <= 1.0))
    # recursive identification: only the first shock moves the first variable on impact
    np.testing.assert_allclose(shares[:, 0, 0], np.broadcast_to([1.0, 0.0, 0.0], (20, 3)))


def test_fevd_matches_forecast_error_covariance(sim_truth):
    """The total ``Σ_s Θ_s Θ_s'`` diagonal is the ``(h+1)``-step forecast-error variance."""
    H = 6
    irf = recursive_irf(sim_truth.B, sim_truth.chol, lags=1, n_det=1, horizon=H)
    shares = fevd(irf)
    A, sigma = sim_truth.coefs, sim_truth.sigma
    mse = np.zeros((3, 3))
    for h in range(H + 1):
        Ah = np.linalg.matrix_power(A, h)
        mse += Ah @ sigma @ Ah.T
        contrib = (irf[: h + 1] ** 2).sum(axis=0)
        np.testing.assert_allclose(contrib.sum(axis=-1), np.diag(mse), rtol=1e-12)
        np.testing.assert_allclose(shares[h], contrib / np.diag(mse)[:, None], rtol=1e-12)


def test_fevd_is_invariant_to_a_common_scale():
    coefs, chol = _draws(11, S=4)
    base = fevd(recursive_irf(coefs, chol, lags=2, n_det=1, horizon=5))
    scaled = fevd(recursive_irf(coefs, chol, lags=2, n_det=1, horizon=5, scale=7.0))
    np.testing.assert_allclose(scaled, base, rtol=1e-12)


# ---------------------------------------------------------------------- IRFResult


@pytest.mark.parametrize("with_chain", [False, True])
def test_irf_result_dims_and_coords(with_chain):
    coefs, chol = _draws(12, S=8)
    irf = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=10)
    arr = irf.reshape(2, 4, *irf.shape[1:]) if with_chain else irf
    result = IRFResult.from_arrays(arr, VARIABLES)
    da = result.data
    assert da.dims == ("chain", "draw", "horizon", "response", "shock")
    assert da.shape == ((2, 4) if with_chain else (1, 8)) + (11, 3, 3)
    np.testing.assert_array_equal(da["horizon"].values, np.arange(11))
    assert list(da["response"].values) == VARIABLES
    assert list(da["shock"].values) == VARIABLES
    assert result.variables == VARIABLES
    assert result.horizon == 10
    assert da.attrs["identification"] == "recursive"
    assert da.attrs["ordering"] == '["gdp", "prices", "rate"]'
    np.testing.assert_array_equal(da.values.reshape(irf.shape), irf)


def test_irf_result_summaries():
    coefs, chol = _draws(13, S=40)
    irf = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=4)
    result = IRFResult.from_arrays(irf, VARIABLES, attrs={"model": {"lags": 2}})
    assert result.data.attrs["model"] == '{"lags": 2}'

    qs = result.quantiles([0.1, 0.9])
    assert qs.dims == ("quantile", "horizon", "response", "shock")
    np.testing.assert_allclose(qs.values, np.quantile(irf, [0.1, 0.9], axis=0))
    assert result.quantiles(0.5).dims == ("horizon", "response", "shock")
    np.testing.assert_allclose(result.median().values, np.median(irf, axis=0))
    np.testing.assert_allclose(result.mean().values, irf.mean(axis=0))

    shares = result.fevd()
    assert shares.dims == result.data.dims
    assert shares.name == "fevd"
    np.testing.assert_allclose(shares.values[0], fevd(irf))
    assert list(shares["shock"].values) == VARIABLES

    frame = result.to_dataframe()
    assert list(frame.columns) == [0.05, 0.5, 0.95]
    assert frame.index.names == ["horizon", "response", "shock"]
    assert len(frame) == 5 * 3 * 3
    assert list(frame.index[:3]) == [(0, "gdp", "gdp"), (0, "gdp", "prices"), (0, "gdp", "rate")]
    np.testing.assert_allclose(
        frame.loc[(2, "rate", "prices")].to_numpy(),
        np.quantile(irf[:, 2, 2, 1], [0.05, 0.5, 0.95]),
    )


def test_irf_result_validation():
    coefs, chol = _draws(14, S=3)
    unit = recursive_irf(coefs, chol, lags=2, n_det=1, horizon=3, shock="unit")
    with pytest.raises(ValueError, match="one-standard-deviation"):
        IRFResult.from_arrays(unit, VARIABLES, shock="unit").fevd()
    with pytest.raises(ValueError, match="variables"):
        IRFResult.from_arrays(unit, VARIABLES[:2])
    sd = IRFResult.from_arrays(unit, VARIABLES)
    with pytest.raises(ValueError, match="horizon"):
        IRFResult(sd.data.isel(horizon=slice(1, None))).fevd()
    with pytest.raises(ValueError, match="dims"):
        IRFResult(sd.data.isel(chain=0))
