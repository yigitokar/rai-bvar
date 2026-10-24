"""The bound on the delta-method MCSE of ``log_pred_density`` (paper App. B, ``prop:mcse-bound``).

``return_mcse`` reports ``sd(w)/(√S·mean(w))`` with ``w_s = exp(log p_s − max_r log p_r)`` and the
sample standard deviation (``ddof=1``), times ``√(S/ESS)`` for MCMC forecasts only. For any
``w_1..w_S ≥ 0``, not all zero (``S ≥ 2``), with Kish ``ESS_K = (Σw)²/Σw²``, the proposition says
``s_w²/(S·w̄²) = (S/ESS_K − 1)/(S − 1) ≤ 1``, with equality if and only if exactly one ``w_s`` is
non-zero, and with the population variance ``σ_w/(√S·w̄) ≤ √((S − 1)/S)``. Checked here on the
iid MCSE (no ``engine`` attribute, so no ESS factor): one-hot weights (first, middle, last draw)
give 1 to rounding (and 1, not ``√((S−1)/S)``: the code uses ``ddof=1``), equal weights give 0, two
non-zero weights give less than 1, ``S = 2`` matches the closed form ``|1 − a|/(1 + a)``, and
random weights (with underflowed zeros) satisfy the bound and the identity. The ``√(S/ESS)``
factor of MCMC forecasts is outside the proposition and can lift the reported value above 1.

The weights are prescribed: one variable, ``cond_chol = 1`` and ``cond_mean_s = √(−2 ℓ_s)`` with
``max ℓ = 0`` give ``log p_s − max_r log p_r = ℓ_s`` at ``y = 0``. The MCSE is invariant to
rescaling the weights, so ``w/max(w)`` covers every non-negative weight vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rai_bvar.diagnostics import ess_mean
from rai_bvar.forecast import ForecastResult
from rai_bvar.linalg import mvn_logpdf_chol

Y = np.array([0.0])


def _forecast(logw: np.ndarray, *, chains: int = 1, engine: str | None = None) -> ForecastResult:
    """A one-variable ``h = 1`` forecast whose density weights at ``y = 0`` are ``exp(logw)``
    (``logw ≤ 0`` with maximum 0; ``−1e6`` underflows to an exact zero weight)."""
    logw = np.asarray(logw, dtype=np.float64).reshape(chains, -1)
    mean = np.sqrt(-2.0 * logw)[:, :, None, None]
    attrs = {} if engine is None else {"engine": engine}
    return ForecastResult.from_arrays(
        paths=mean, cond_mean=mean, cond_chol=np.ones((*logw.shape, 1, 1, 1)),
        variables=["x"], origin=pd.Period("2000Q1", "Q"), attrs=attrs,
    )  # fmt: skip


def _weights(fr: ForecastResult) -> np.ndarray:
    """The weights ``exp(log p_s − max)`` the estimator works with, recomputed from the draws."""
    logp = mvn_logpdf_chol(Y, fr.stacked("cond_mean")[:, 0], fr.stacked("cond_chol")[:, 0])
    return np.exp(logp - logp.max())


def _kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def _one_hot(S: int, at: int) -> np.ndarray:
    logw = np.full(S, -1e6)
    logw[at] = 0.0
    return logw


@pytest.mark.parametrize("S", [2, 3, 10, 1000, 4000])
def test_one_draw_carrying_all_the_weight_reaches_the_bound(S):
    """``ESS_K = 1``: the iid MCSE is 1 to rounding wherever the draw sits, with ``ddof=1`` (the
    population formula would give ``√((S−1)/S)``, 0.707 at ``S = 2``)."""
    for at in (0, S // 2, S - 1):
        fr = _forecast(_one_hot(S, at))
        w = _weights(fr)
        assert np.count_nonzero(w) == 1 and _kish(w) == 1.0
        _, mcse = fr.log_pred_density(Y, 1, return_mcse=True)
        np.testing.assert_allclose(mcse, 1.0, rtol=1e-14)
        np.testing.assert_allclose(mcse, w.std(ddof=1) / (np.sqrt(S) * w.mean()), rtol=1e-14)
        population = w.std(ddof=0) / (np.sqrt(S) * w.mean())
        np.testing.assert_allclose(population, np.sqrt((S - 1) / S), rtol=1e-14)


@pytest.mark.parametrize("S", [2, 3, 10, 1000])
def test_equal_weights_give_zero(S):
    """``ESS_K = S``: ``(S/ESS_K − 1)/(S − 1) = 0`` and the MCSE is exactly 0."""
    fr = _forecast(np.zeros(S))
    assert _kish(_weights(fr)) == pytest.approx(S, rel=1e-14)
    _, mcse = fr.log_pred_density(Y, 1, return_mcse=True)
    assert mcse == 0.0


@pytest.mark.parametrize("S", [3, 10, 1000])
def test_two_non_zero_weights_stay_below_the_bound(S):
    """The "only if" direction: with two non-zero weights ``ESS_K > 1`` and the MCSE is below 1
    (two equal weights among zeros: ``ESS_K = 2``, MCSE² ``= (S/2 − 1)/(S − 1)``)."""
    for a in (1e-3, 0.5, 1.0):
        logw = _one_hot(S, 0)
        logw[S - 1] = np.log(a)
        fr = _forecast(logw)
        w = _weights(fr)
        _, mcse = fr.log_pred_density(Y, 1, return_mcse=True)
        assert mcse < 1.0
        np.testing.assert_allclose(mcse**2, (S / _kish(w) - 1) / (S - 1), rtol=1e-10)
        if a == 1.0:
            np.testing.assert_allclose(mcse**2, (S / 2 - 1) / (S - 1), rtol=1e-12)


def test_two_draws_closed_form():
    """``S = 2``, ``w = (1, a)``: MCSE ``= |1 − a|/(1 + a)``, from 1 at ``a = 0`` to 0 at ``a = 1``."""
    for a in (0.0, 1e-300, 1e-12, 1e-3, 0.25, 0.5, 0.9, 1.0):
        logw = np.array([0.0, np.log(a) if a > 0 else -1e6])
        fr = _forecast(logw)
        w = _weights(fr)
        _, mcse = fr.log_pred_density(Y, 1, return_mcse=True)
        assert mcse <= 1.0 + 1e-14
        np.testing.assert_allclose(mcse, abs(1 - w[1]) / (1 + w[1]), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(mcse**2, 2 / _kish(w) - 1, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("S", [2, 3, 5, 10, 100, 1000])
@pytest.mark.parametrize("sigma", [0.1, 1.0, 5.0, 30.0, 300.0])
def test_random_weights_satisfy_the_identity_and_the_bound(S, sigma):
    """Log-normal weights of every spread (``σ = 300`` underflows most of them to exact zeros),
    some with a random subset forced to zero, on one and on four chains:
    ``mcse² = (S/ESS_K − 1)/(S − 1) ≤ 1`` (1e-10), the code equals the ``ddof=1`` formula, and
    the population version is at most ``√((S − 1)/S)``."""
    rng = np.random.default_rng([S, int(sigma * 10)])
    for trial in range(20):
        z = rng.standard_normal(S)
        logw = sigma * (z - z.max())
        if trial % 2:
            zero = rng.random(S) < 0.5
            zero[np.argmax(z)] = False  # the largest weight (1) stays: not all zero
            logw[zero] = -1e6
        chains = 4 if S % 4 == 0 and trial % 3 == 0 else 1
        fr = _forecast(logw, chains=chains)
        w = _weights(fr)
        _, mcse = fr.log_pred_density(Y, 1, return_mcse=True)
        assert 0.0 <= mcse <= 1.0 + 1e-12
        ratio = (S / _kish(w) - 1) / (S - 1)
        assert ratio <= 1.0 + 1e-12
        np.testing.assert_allclose(mcse**2, ratio, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(mcse, w.std(ddof=1) / (np.sqrt(S) * w.mean()), rtol=1e-12)
        population = w.std(ddof=0) / (np.sqrt(S) * w.mean())
        assert population <= np.sqrt((S - 1) / S) * (1.0 + 1e-12)
        if np.count_nonzero(w) > 1 and _kish(w) > 1.0 + 1e-6:
            assert mcse < 1.0


def test_the_mcmc_ess_factor_is_outside_the_bound():
    """Consequence (iii): MCMC forecasts multiply the iid value by ``√(S/ESS)`` (``ess_mean`` on
    the ``(chain, draw)`` array), which the proposition does not cover. Five adjacent draws of
    one chain carrying all the weight: iid 0.44, reported above 1."""
    logw = np.full((4, 50), -1e6)
    logw[0, :5] = 0.0
    iid = _forecast(logw, chains=4)
    mc = _forecast(logw, chains=4, engine="mcmc")
    _, se_iid = iid.log_pred_density(Y, 1, return_mcse=True)
    _, se_mc = mc.log_pred_density(Y, 1, return_mcse=True)
    w = _weights(iid)
    np.testing.assert_allclose(se_iid**2, (w.size / 5 - 1) / (w.size - 1), rtol=1e-12)
    np.testing.assert_allclose(
        se_mc, se_iid * np.sqrt(w.size / ess_mean(w.reshape(4, 50))), rtol=1e-12
    )
    assert se_iid < 1.0 < se_mc
