import numpy as np
import pytest

from rai_bvar.evaluation.significance import (
    circular_block_bootstrap,
    diebold_mariano,
    model_confidence_set,
)


def _ma_losses(rng, P, h, shift=0.0):
    """Loss differentials with MA(h−1) dependence, as for h-step forecast errors."""
    e = rng.standard_normal(P + h - 1)
    d = np.convolve(e, np.ones(h), mode="valid") / np.sqrt(h)
    base = rng.standard_normal(P)
    return base + d + shift, base


def test_dm_identical_losses_is_undefined():
    x = np.arange(10.0)
    result = diebold_mariano(x, x)
    assert np.isnan(result.statistic) and np.isnan(result.pvalue)


def test_dm_drops_nan_pairs():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    a[[3, 7]] = np.nan
    assert diebold_mariano(a, b).n == 48


@pytest.mark.parametrize("h", [1, 4])
def test_dm_size_under_null(h):
    rng = np.random.default_rng(h)
    rejections = [
        diebold_mariano(*_ma_losses(rng, 80, h), horizon=h).pvalue < 0.05 for _ in range(400)
    ]
    assert 0.01 < np.mean(rejections) < 0.12


def test_dm_power_and_direction():
    rng = np.random.default_rng(1)
    worse, better = _ma_losses(rng, 120, 1, shift=0.5)  # loss_a larger → A worse
    two = diebold_mariano(worse, better)
    assert two.pvalue < 0.01 and two.mean_diff > 0
    assert diebold_mariano(worse, better, alternative="greater").pvalue < 0.01
    assert diebold_mariano(worse, better, alternative="less").pvalue > 0.99
    assert diebold_mariano(better, worse, alternative="less").pvalue < 0.01


def test_dm_hln_uses_student_t():
    rng = np.random.default_rng(2)
    a, b = _ma_losses(rng, 30, 1, shift=0.3)
    with_hln, without = diebold_mariano(a, b, hln=True), diebold_mariano(a, b, hln=False)
    assert abs(with_hln.statistic) < abs(without.statistic)
    assert with_hln.pvalue > without.pvalue


def test_dm_rejects_unknown_alternative():
    with pytest.raises(ValueError):
        diebold_mariano(np.arange(5.0), np.arange(5.0)[::-1], alternative="bigger")


def test_circular_block_bootstrap_indices():
    idx = circular_block_bootstrap(17, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 17)
    assert idx.min() >= 0 and idx.max() < 17
    # within a block, indices advance by one (mod n)
    assert np.all((idx[:, 1] - idx[:, 0]) % 17 == 1)


def test_mcs_eliminates_clearly_worse_models_first():
    rng = np.random.default_rng(3)
    P = 150
    common = rng.standard_normal(P)
    losses = np.column_stack(
        [
            common + 0.2 * rng.standard_normal(P),
            common + 1.5 + rng.standard_normal(P),
            common + 0.6 + rng.standard_normal(P),
        ]
    )
    result = model_confidence_set(losses, n_boot=1000, seed=0)
    assert result.elimination_order[0] == 1
    assert result.pvalues[0] == 1.0
    assert result.pvalues[1] < 0.05
    ordered = [result.pvalues[i] for i in result.elimination_order]
    assert ordered == sorted(ordered)
    assert result.included(0.10)[0] and not result.included(0.10)[1]


def test_mcs_keeps_equivalent_models():
    rng = np.random.default_rng(4)
    P = 200
    common = rng.standard_normal(P)
    losses = np.column_stack([common + 0.5 * rng.standard_normal(P) for _ in range(3)])
    assert model_confidence_set(losses, n_boot=1000, seed=0).included(0.10).all()


def test_mcs_degenerate_inputs():
    assert model_confidence_set(np.ones((10, 1))).pvalues.tolist() == [1.0]
    losses = np.ones((10, 2))
    losses[:, 1] = np.nan
    assert model_confidence_set(losses).pvalues.tolist() == [1.0, 1.0]
