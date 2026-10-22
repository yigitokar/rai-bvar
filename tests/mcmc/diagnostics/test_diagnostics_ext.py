"""Tail ESS, mean ESS, MCSE of means and quantiles, and the batched diagnostics (DESIGN §8.4,
D-24; Vehtari et al. 2021 §4.3–4.4; WP15).

Reference values come from ArviZ: ``az.ess(x, method="tail" | "mean")`` and
``az.mcse(x, method="mean" | "quantile", prob=p)`` of ``arviz`` 0.23.4, and the same statistics
from ``arviz_stats.base.array_stats`` of ``arviz-stats`` 1.3.3, which agree to the last digit on
every case below. They were computed once, in a throwaway virtualenv (numpy 2.5.4, scipy 1.18.1),
on the exact arrays the builders in this file produce, and are hard-coded: ArviZ is not a test
dependency (a live comparison runs when it is installed). Both are deterministic implementations
of the same estimators, so the tolerance only allows for summation order (``rtol=1e-10``).

Known-answer tests use stationary Gaussian AR(1) chains, whose integrated autocorrelation times
are known: ``τ = (1 + ρ)/(1 − ρ)`` for the mean, and for the indicator ``1{x ≤ q_p}``
``τ_p = 1 + 2 Σ_k [Φ₂(z_p, z_p; ρ^k) − p²]/(p(1 − p))``, with the bivariate normal orthant
probability from Plackett's identity ``∂Φ₂/∂r = φ₂``. Statistical bounds were calibrated over 30
seed sets of the same designs and sit at least 3.5 sampling s.d. from the observed means (the
calibration extremes are quoted next to each bound).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import integrate, stats

from rai_bvar import diagnostics as dg

# --------------------------------------------------------------------------- chain builders
# (the reference script exec'd exactly this block in the ArviZ venv; keep it self-contained)


def ar1_chains(rng: np.random.Generator, rho: float, chains: int, draws: int) -> np.ndarray:
    """Stationary Gaussian AR(1) chains with unit innovation variance."""
    e = rng.standard_normal((chains, draws))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1.0 - rho**2)
    for t in range(1, draws):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def rwm_chains(rng: np.random.Generator, starts, draws: int, step: float = 2.4) -> np.ndarray:
    """Random-walk Metropolis chains targeting N(0, 1): rejections produce repeated values."""
    cur = np.asarray(starts, dtype=np.float64).copy()
    out = np.empty((cur.size, draws))
    for t in range(draws):
        prop = cur + step * rng.standard_normal(cur.size)
        accept = np.log(rng.random(cur.size)) < 0.5 * (cur**2 - prop**2)
        cur = np.where(accept, prop, cur)
        out[:, t] = cur
    return out


CASE_SEEDS = {
    "iid": 201,
    "iid_odd": 202,
    "shifted": 203,
    "scaled": 204,
    "ar_0.5": 205,
    "ar_0.9": 206,
    "ar_-0.3": 207,
    "cauchy": 208,
    "lognormal": 209,
    "ties": 210,
    "rwm": 211,
    "single_chain": 212,
    "many_chains": 213,
    "grid_atom": 214,
    "short": 215,
    "infinities": 216,
}


def build(name: str) -> np.ndarray:
    """The ``(chains, draws)`` array of reference case ``name``."""
    rng = np.random.default_rng(CASE_SEEDS[name])
    if name == "iid":
        return rng.standard_normal((4, 1000))
    if name == "iid_odd":  # odd length: the split drops the middle draw
        return rng.standard_normal((4, 1001))
    if name == "shifted":  # one chain shifted by one s.d.
        return rng.standard_normal((4, 1000)) + np.array([[0.0], [0.0], [0.0], [1.0]])
    if name == "scaled":  # one chain with three times the s.d.
        return rng.standard_normal((4, 1000)) * np.array([[1.0], [1.0], [1.0], [3.0]])
    if name.startswith("ar_"):
        return ar1_chains(rng, float(name[3:]), 4, 2000)
    if name == "cauchy":
        return rng.standard_cauchy((4, 1000))
    if name == "lognormal":  # skewed: the two tails differ
        return np.exp(ar1_chains(rng, 0.7, 4, 1500))
    if name == "ties":  # discrete draws
        return rng.poisson(2.0, (4, 500)).astype(np.float64)
    if name == "rwm":  # dispersed starts, no burn-in
        return rwm_chains(rng, [-3.0, -1.0, 1.0, 3.0], 1000)
    if name == "single_chain":
        return rng.standard_normal((1, 1000))
    if name == "many_chains":  # 16 short chains with small chain-specific offsets
        return rng.standard_normal((16, 64)) + 0.2 * rng.standard_normal((16, 1))
    if name == "grid_atom":  # a nu-like grid draw with > 5% of the mass at the upper bound 40
        return np.minimum(np.round(rng.gamma(4.0, 6.0, (4, 800))) + 3.0, 40.0)
    if name == "short":  # split halves of 5 draws: the shortest accepted length
        return rng.standard_normal((4, 10))
    if name == "infinities":  # tails only: the tail statistics stay defined
        x = rng.standard_normal((4, 100))
        x[1, 5], x[2, 50] = np.inf, -np.inf
        return x
    raise KeyError(name)


# --------------------------------------------------------------------------- end of builders

NAN = float("nan")
# ArviZ 0.23.4 == arviz-stats 1.3.3 on every entry (printed with repr, i.e. full precision).
# ``infinities``: the mean statistics need finite draws (ArviZ: nan as well); the tail ones do not.
# ``grid_atom``: more than 5% of the draws sit at the maximum, so 1{x ≤ q̂_0.95} never changes and
# its ESS is the number of split draws (3200), the tail ESS is that minimum.
ARVIZ = {
    "iid": {
        "ess_tail": 3843.8227301925517,
        "ess_mean": 3746.159729109136,
        "mcse_mean": 0.01641934974575674,
        "mcse_q0.05": 0.027858491152244147,
        "mcse_q0.5": 0.020244964768271607,
        "mcse_q0.95": 0.029919790057448714,
    },
    "iid_odd": {
        "ess_tail": 3905.051579415536,
        "ess_mean": 3721.854714122347,
        "mcse_mean": 0.016417206574336814,
        "mcse_q0.05": 0.02858609893289854,
        "mcse_q0.5": 0.017279558197362353,
        "mcse_q0.95": 0.02982947092282573,
    },
    "shifted": {
        "ess_tail": 156.70539140752555,
        "ess_mean": 26.5717002548103,
        "mcse_mean": 0.20402770782332114,
        "mcse_q0.05": 0.04102286375613595,
        "mcse_q0.5": 0.18790764738574992,
        "mcse_q0.95": 0.17405146966166396,
    },
    "scaled": {
        "ess_tail": 38.413627740045946,
        "ess_mean": 4318.56923021901,
        "mcse_mean": 0.025940765624525863,
        "mcse_q0.05": 0.7591921388630557,
        "mcse_q0.5": 0.020078124792335176,
        "mcse_q0.95": 0.7440864996426626,
    },
    "ar_0.5": {
        "ess_tail": 4300.603872413455,
        "ess_mean": 2749.3058751379867,
        "mcse_mean": 0.022236989076017892,
        "mcse_q0.05": 0.0423117426685965,
        "mcse_q0.5": 0.02380528501116197,
        "mcse_q0.95": 0.03190562742273917,
    },
    "ar_0.9": {
        "ess_tail": 1003.2270359574468,
        "ess_mean": 393.19620722294445,
        "mcse_mean": 0.11918848400194743,
        "mcse_q0.05": 0.1363397288814101,
        "mcse_q0.5": 0.11231224379164356,
        "mcse_q0.95": 0.1475030546689977,
    },
    "ar_-0.3": {
        "ess_tail": 7907.055384039804,
        "ess_mean": 14169.22159968271,
        "mcse_mean": 0.008850920449084577,
        "mcse_q0.05": 0.03144073985207907,
        "mcse_q0.5": 0.014651466757077729,
        "mcse_q0.95": 0.02431379911637266,
    },
    "cauchy": {
        "ess_tail": 4010.5113984491345,
        "ess_mean": 4025.220742333073,
        "mcse_mean": 0.48830199404263597,
        "mcse_q0.05": 0.38798253177978426,
        "mcse_q0.5": 0.0268327458981144,
        "mcse_q0.95": 0.41205485181625745,
    },
    "lognormal": {
        "ess_tail": 2088.375225328775,
        "ess_mean": 1730.293354629695,
        "mcse_mean": 0.16873849464974525,
        "mcse_q0.05": 0.006832510955485191,
        "mcse_q0.5": 0.05097971349844255,
        "mcse_q0.95": 0.7923569277919436,
    },
    "ties": {
        "ess_tail": 1918.2279781207833,
        "ess_mean": 1986.153763038249,
        "mcse_mean": 0.031246175530863007,
        "mcse_q0.05": 0.0,
        "mcse_q0.5": 0.0,
        "mcse_q0.95": 0.5,
    },
    "rwm": {
        "ess_tail": 888.645884205243,
        "ess_mean": 746.3138047065861,
        "mcse_mean": 0.03753211047617504,
        "mcse_q0.05": 0.06643501518915862,
        "mcse_q0.5": 0.0514930712636045,
        "mcse_q0.95": 0.08135980048434854,
    },
    "single_chain": {
        "ess_tail": 983.2831256179077,
        "ess_mean": 1121.656460611712,
        "mcse_mean": 0.030104172886235063,
        "mcse_q0.05": 0.05670813982683498,
        "mcse_q0.5": 0.04213694056472389,
        "mcse_q0.95": 0.07282793168104984,
    },
    "many_chains": {
        "ess_tail": 945.3071608646427,
        "ess_mean": 888.3607629476893,
        "mcse_mean": 0.03508908267109363,
        "mcse_q0.05": 0.06991812521856722,
        "mcse_q0.5": 0.041827218400532364,
        "mcse_q0.95": 0.09647825975833624,
    },
    "grid_atom": {
        "ess_tail": 3200.0,
        "ess_mean": 3146.97956831466,
        "mcse_mean": 0.16846213160064524,
        "mcse_q0.05": 0.0,
        "mcse_q0.5": 0.5,
        "mcse_q0.95": 0.0,
    },
    "short": {
        "ess_tail": 46.57534246575342,
        "ess_mean": 46.945303260399875,
        "mcse_mean": 0.12833562059186385,
        "mcse_q0.05": 0.3621030561912093,
        "mcse_q0.5": 0.19428059939965758,
        "mcse_q0.95": 0.36241755457068364,
    },
    "infinities": {
        "ess_tail": 361.1660224368452,
        "ess_mean": NAN,
        "mcse_mean": NAN,
        "mcse_q0.05": 0.24509675705621958,
        "mcse_q0.5": 0.05561662890508798,
        "mcse_q0.95": 0.129050266979433,
    },
}
REF_RTOL = 1e-10

STATISTICS = {
    "ess_tail": dg.ess_tail,
    "ess_mean": dg.ess_mean,
    "mcse_mean": dg.mcse_mean,
    "mcse_q0.05": lambda x: dg.mcse_quantile(x, 0.05),
    "mcse_q0.5": lambda x: dg.mcse_quantile(x, 0.5),
    "mcse_q0.95": lambda x: dg.mcse_quantile(x, 0.95),
}
BATCHED = {
    "ess_tail": dg.ess_tail_many,
    "ess_mean": dg.ess_mean_many,
    "mcse_mean": dg.mcse_mean_many,
    "mcse_q0.05": lambda x: dg.mcse_quantile_many(x, 0.05),
    "mcse_q0.5": lambda x: dg.mcse_quantile_many(x, 0.5),
    "mcse_q0.95": lambda x: dg.mcse_quantile_many(x, 0.95),
    "rhat": dg.rhat_many,
    "ess_bulk": dg.ess_bulk_many,
}
SCALAR = {**STATISTICS, "rhat": dg.rhat, "ess_bulk": dg.ess_bulk}


def same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


# --------------------------------------------------------------------------- ArviZ parity


@pytest.mark.parametrize("stat", sorted(STATISTICS))
@pytest.mark.parametrize("case", sorted(ARVIZ))
def test_matches_arviz(case, stat):
    value = STATISTICS[stat](build(case))
    assert isinstance(value, float)
    ref = ARVIZ[case][stat]
    if np.isnan(ref):
        assert np.isnan(value)
    else:
        np.testing.assert_allclose(value, ref, rtol=REF_RTOL, atol=1e-15)


@pytest.mark.parametrize("case", ["iid", "ar_0.9", "lognormal", "ties", "rwm", "grid_atom"])
def test_matches_installed_arviz(case):
    az = pytest.importorskip("arviz")
    x = build(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = {
            "ess_tail": float(az.ess(x, method="tail")),
            "ess_mean": float(az.ess(x, method="mean")),
            "mcse_mean": float(az.mcse(x, method="mean")),
            **{
                f"mcse_q{p}": float(az.mcse(x, method="quantile", prob=p))
                for p in (0.05, 0.5, 0.95)
            },
        }
    for stat, fn in STATISTICS.items():
        np.testing.assert_allclose(fn(x), ref[stat], rtol=REF_RTOL, atol=1e-15, err_msg=stat)


# --------------------------------------------------------------------------- batched == scalar


def stacked(cases: list[str], draws: int) -> np.ndarray:
    """``(4, draws, len(cases))``: the first ``draws`` draws of 4-chain cases."""
    return np.stack([build(c)[:4, :draws] for c in cases], axis=-1)


BATCH_CASES = [
    "iid",
    "shifted",
    "scaled",
    "ar_0.9",
    "cauchy",
    "lognormal",
    "ties",
    "rwm",
    "grid_atom",
]


@pytest.mark.parametrize("stat", sorted(BATCHED))
def test_batched_equals_scalar(stat):
    """Every ``*_many`` equals its scalar function column by column; for ``rhat``/``ess_bulk``
    (the functions predating Phase 2) to rounding, bitwise in practice."""
    x = stacked(BATCH_CASES, 500)
    x = np.concatenate([x, np.full((4, 500, 1), 2.5)], axis=-1)  # a constant column
    got = BATCHED[stat](x)
    assert got.shape == (x.shape[-1],) and got.dtype == np.float64
    for j in range(x.shape[-1]):
        ref = SCALAR[stat](x[:, :, j])
        if np.isnan(ref):
            assert np.isnan(got[j]), (stat, j)
        else:
            np.testing.assert_allclose(got[j], ref, rtol=1e-12, err_msg=f"{stat}[{j}]")


@pytest.mark.parametrize("stat", sorted(BATCHED))
def test_batched_trailing_shapes(stat):
    rng = np.random.default_rng(31)
    x = rng.standard_normal((3, 120, 2, 3)) + rng.standard_normal((3, 1, 2, 3))
    got = BATCHED[stat](x)
    assert got.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert same(got[i, j], SCALAR[stat](x[:, :, i, j]))
    zero_d = BATCHED[stat](x[:, :, 0, 0])
    assert zero_d.shape == () and same(float(zero_d), SCALAR[stat](x[:, :, 0, 0]))
    assert BATCHED[stat](np.zeros((4, 50, 0))).shape == (0,)


def test_a_bad_column_leaves_the_others_alone():
    x = stacked(["iid", "ar_0.5", "rwm"], 400)
    clean = {k: f(x) for k, f in BATCHED.items()}
    dirty = np.concatenate([x, x[..., :2]], axis=-1)
    dirty[1, 7, 3] = np.nan
    dirty[2, 9, 4] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, fn in BATCHED.items():
            out = fn(dirty)
            np.testing.assert_array_equal(out[:3], clean[name], err_msg=name)
            assert np.isnan(out[3]), name  # a NaN draw
    assert np.isnan(dg.ess_mean_many(dirty)[4]) and np.isfinite(dg.ess_tail_many(dirty)[4])


def test_summary_equals_the_individual_functions():
    x = stacked(BATCH_CASES, 500)
    s = dg.summary_many(x, quantiles=(0.05, 0.5, 0.95, 0.25))
    for name, fn in BATCHED.items():
        key = name.replace("q0.05", "q5").replace("q0.5", "q50").replace("q0.95", "q95")
        np.testing.assert_array_equal(s[key], fn(x), err_msg=name)
    np.testing.assert_array_equal(s["mcse_q25"], dg.mcse_quantile_many(x, 0.25))
    flat = x.reshape(-1, x.shape[-1])
    np.testing.assert_allclose(s["mean"], flat.mean(axis=0), rtol=1e-13)
    np.testing.assert_allclose(s["sd"], flat.std(axis=0, ddof=1), rtol=1e-13)
    np.testing.assert_allclose(s["q25"], np.quantile(flat, 0.25, axis=0), rtol=1e-13)
    assert set(s) == {
        "mean", "sd", "mcse_mean", "ess_mean", "rhat", "ess_bulk", "ess_tail",
        *(f"{a}q{q}" for a in ("", "mcse_") for q in (5, 50, 95, 25)),
    }  # fmt: skip
    rest = dg.summary_many(x, quantiles=(0.05, 0.5, 0.95, 0.25), rank_normalised=False)
    assert set(rest) == set(s) - {"rhat", "ess_bulk"}  # left to rhat_ess_bulk_many
    for key, value in rest.items():
        np.testing.assert_array_equal(value, s[key], err_msg=key)


# --------------------------------------------------------------------------- known answers (AR(1))


def ar1_batch(rng: np.random.Generator, rho: float, shape: tuple[int, ...]) -> np.ndarray:
    """Stationary unit-innovation AR(1) along axis 1 of ``shape = (chains, draws, ...)``."""
    e = rng.standard_normal(shape)
    x = np.empty(shape)
    x[:, 0] = e[:, 0] / np.sqrt(1.0 - rho**2)
    for t in range(1, shape[1]):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def indicator_iat(rho: float, p: float) -> float:
    """``τ_p = 1 + 2 Σ_k corr(1{x_0 ≤ q_p}, 1{x_k ≤ q_p})`` for a Gaussian AR(1), with
    ``Φ₂(z, z; r) − p² = ∫_0^r exp(−z²/(1 + s)) / (2π√(1 − s²)) ds`` (Plackett)."""
    z = stats.norm.ppf(p)
    total, k = 0.0, 1
    while abs(rho) ** k > 1e-14:
        cov = integrate.quad(
            lambda s: np.exp(-z * z / (1.0 + s)) / (2.0 * np.pi * np.sqrt(1.0 - s * s)),
            0.0,
            rho**k,
            epsabs=1e-15,
            epsrel=1e-12,
        )[0]
        total += cov / (p * (1.0 - p))
        k += 1
    return 1.0 + 2.0 * total


def test_indicator_iat_oracle():
    assert indicator_iat(0.0, 0.05) == 1.0
    # ρ → the orthant probability: Φ₂(0, 0; r) = 1/4 + arcsin(r)/(2π) (Sheppard)
    z = 0.5
    direct = sum(np.arcsin(0.6**k) / (2 * np.pi) / 0.25 for k in range(1, 200))
    np.testing.assert_allclose(indicator_iat(0.6, z), 1 + 2 * direct, rtol=1e-10)
    assert 1.0 < indicator_iat(0.9, 0.05) < indicator_iat(0.9, 0.5) < 19.0


# Mean of the ratio estimate/theory over 40 replications of 4 × 2000 draws. Calibration over 30
# seed sets (min–max of the 40-replication means): ess_mean 0.962–1.036, ess_tail 0.918–1.008
# (the minimum of two estimates is biased low by 2–5%), mcse_mean 0.985–1.034, mcse_q5
# 0.944–1.098, mcse_q50 0.965–1.067; s.d. ≤ 0.019, 0.017, 0.012, 0.028, 0.019.
AR_BOUNDS = {
    "ess_mean": (0.92, 1.08),
    "ess_tail": (0.87, 1.04),
    "mcse_mean": (0.95, 1.05),
    "mcse_q5": (0.88, 1.14),
    "mcse_q50": (0.92, 1.09),
}


@pytest.mark.parametrize(("rho", "seed"), [(0.0, 1), (0.5, 2), (0.9, 3), (-0.3, 4)])
def test_ar1_known_autocorrelation_times(rho, seed):
    chains, draws, reps = 4, 2000, 40
    total = chains * draws
    x = ar1_batch(np.random.default_rng(seed), rho, (chains, draws, reps))
    tau, sd = (1.0 + rho) / (1.0 - rho), 1.0 / np.sqrt(1.0 - rho**2)
    tau05, tau50 = indicator_iat(rho, 0.05), indicator_iat(rho, 0.5)
    theory = {
        "ess_mean": total / tau,
        "ess_tail": total / tau05,
        "mcse_mean": sd * np.sqrt(tau / total),
        # asymptotic s.d. of a sample quantile: √(τ_p p(1−p)/N) / f(q_p), f the N(0, sd²) density
        "mcse_q5": np.sqrt(tau05 * 0.05 * 0.95 / total) * sd / stats.norm.pdf(stats.norm.ppf(0.05)),
        "mcse_q50": np.sqrt(tau50 * 0.25 / total) * sd / stats.norm.pdf(0.0),
    }
    summary = dg.summary_many(x)
    for name, (lo, hi) in AR_BOUNDS.items():
        ratio = float(np.mean(summary[name] / theory[name]))
        assert lo < ratio < hi, (name, ratio)


def test_mcse_covers_the_truth():
    """95% intervals ``estimate ± 1.96 MCSE`` cover the true mean, 5% and 50% quantiles of an
    AR(1) with ρ = 0.9 in 95% of 400 replications (calibration: 0.912–0.978 over 30 seed sets)."""
    rho = 0.9
    x = ar1_batch(np.random.default_rng(77), rho, (4, 2000, 400))
    s = dg.summary_many(x)
    sd = 1.0 / np.sqrt(1.0 - rho**2)
    for name, truth in (("mean", 0.0), ("q5", sd * stats.norm.ppf(0.05)), ("q50", 0.0)):
        cover = float(np.mean(np.abs(s[name] - truth) < 1.96 * s[f"mcse_{name}"]))
        assert 0.90 < cover < 0.985, (name, cover)


def test_tail_ess_sees_what_bulk_ess_misses():
    """A chain three times as dispersed as the others leaves bulk ESS high but wrecks the tails."""
    x = build("scaled")
    assert dg.ess_bulk(x) > 3000 and dg.ess_tail(x) < 100
    assert dg.mcse_quantile(x, 0.95) > 20 * dg.mcse_quantile(x, 0.5)


# --------------------------------------------------------------------------- invariances and edges


@pytest.mark.parametrize("case", ["iid", "ar_0.5", "rwm", "lognormal"])
def test_tail_and_quantile_ess_invariant_to_increasing_maps(case):
    x = build(case)
    for f in (np.exp, lambda v: 3.0 * v - 7.0, np.arctan):
        assert dg.ess_tail(f(x)) == dg.ess_tail(x)
    if case != "rwm":  # without ties, 1{−x ≤ q̂} = 1 − 1{x ≤ q̂′}: the same autocorrelations
        np.testing.assert_allclose(dg.ess_tail(-x), dg.ess_tail(x), rtol=1e-12)


@pytest.mark.parametrize("case", ["iid", "ar_0.9", "rwm"])
def test_mcse_equivariance(case):
    x = build(case)
    np.testing.assert_allclose(dg.mcse_mean(2.5 * x - 4.0), 2.5 * dg.mcse_mean(x), rtol=1e-10)
    np.testing.assert_allclose(dg.mcse_mean(-x), dg.mcse_mean(x), rtol=1e-10)
    np.testing.assert_allclose(dg.ess_mean(2.5 * x - 4.0), dg.ess_mean(x), rtol=1e-10)
    for p in (0.05, 0.5, 0.95):
        np.testing.assert_allclose(
            dg.mcse_quantile(2.5 * x - 4.0, p), 2.5 * dg.mcse_quantile(x, p), rtol=1e-10
        )


@pytest.mark.parametrize("draws", [0, 1, 4, 9])
def test_too_short(draws):
    """Split halves of fewer than 5 draws: every new statistic is NaN (no warnings)."""
    x = np.random.default_rng(draws).standard_normal((4, draws))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, fn in STATISTICS.items():
            assert np.isnan(fn(x)), name
    assert np.isfinite(dg.ess_tail(np.random.default_rng(0).standard_normal((4, 10))))


@pytest.mark.parametrize("value", [0.0, 1 / 3, -7.3, 1e10])
@pytest.mark.parametrize(("chains", "draws"), [(4, 1000), (3, 999), (1, 50)])
def test_constant_chains_give_nan_without_warnings(value, chains, draws):
    """Chains that never move carry no evidence (as for ``ess_bulk``; ArviZ returns the number
    of draws, or an MCSE of 0, instead)."""
    x = np.full((chains, draws), value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, fn in STATISTICS.items():
            assert np.isnan(fn(x)), name
        s = dg.summary_many(x[:, :, None])
    np.testing.assert_allclose([s["mean"][0], s["q50"][0]], value, rtol=1e-14)
    assert s["sd"][0] <= 1e-14 * abs(value)


def test_nan_draws_give_nan():
    x = build("iid")
    x[3, 500] = np.nan
    for name, fn in STATISTICS.items():
        assert np.isnan(fn(x)), name


def test_infinite_draws_keep_the_tail_statistics():
    x = build("infinities")
    assert np.isnan(dg.ess_mean(x)) and np.isnan(dg.mcse_mean(x))
    assert np.isfinite(dg.ess_tail(x)) and np.isfinite(dg.mcse_quantile(x, 0.5))


def test_degenerate_indicator_counts_every_split_draw():
    x = build("grid_atom")
    assert np.mean(x == 40.0) > 0.05
    assert dg.ess_tail(x) == x.shape[0] * (x.shape[1] // 2) * 2
    assert dg.mcse_quantile(x, 0.95) == 0.0


def test_inputs_lists_and_shapes():
    x = build("rwm")
    before = x.copy()
    for fn in STATISTICS.values():
        fn(x)
    np.testing.assert_array_equal(x, before)
    for name, fn in STATISTICS.items():
        assert same(fn(x.tolist()), fn(x)), name
        assert same(fn(x[0]), fn(x[:1])), name  # a 1-D input is one chain
    counts = np.random.default_rng(2).poisson(3.0, (4, 200))
    assert dg.ess_tail(counts) == dg.ess_tail(counts.astype(np.float64))
    with pytest.raises(ValueError, match="chains, draws"):
        dg.ess_tail(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="chains, draws"):
        dg.rhat_many(np.zeros(5))
    for q in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            dg.mcse_quantile(x, q)


def test_existing_functions_are_unchanged():
    """The Phase 1 functions keep their behaviour (tests/unit/test_diagnostics.py pins them to
    ArviZ); the batched kernels reproduce them bit for bit on these cases."""
    for case in ("iid", "ar_0.9", "ties", "rwm", "single_chain", "many_chains"):
        x = build(case)
        assert dg.rhat_many(x[:, :, None])[0] == dg.rhat(x)
        assert dg.ess_bulk_many(x[:, :, None])[0] == dg.ess_bulk(x)
        r, e = dg.rhat_ess_bulk_many(x[:, :, None])
        assert r[0] == dg.rhat(x) and e[0] == dg.ess_bulk(x)


# --------------------------------------------------------------------------- nightly (slow)


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, -0.3])
def test_mcse_coverage_over_many_replications(rho):
    """2000 replications of 4 × 2000 draws: 95% MCSE intervals cover the truth 93–97% of the
    time for the mean and the 5/50/95% quantiles (s.e. of a coverage ≈ 0.5%)."""
    sd = 1.0 / np.sqrt(1.0 - rho**2)
    truths = {
        "mean": 0.0,
        "q5": sd * stats.norm.ppf(0.05),
        "q50": 0.0,
        "q95": sd * stats.norm.ppf(0.95),
    }
    hits = {k: [] for k in truths}
    for block in range(5):
        x = ar1_batch(np.random.default_rng(1000 + block), rho, (4, 2000, 400))
        s = dg.summary_many(x)
        for name, truth in truths.items():
            hits[name].append(np.abs(s[name] - truth) < 1.96 * s[f"mcse_{name}"])
    for name, h in hits.items():
        cover = float(np.mean(np.concatenate(h)))
        assert 0.925 < cover < 0.97, (name, cover)


@pytest.mark.slow
def test_batched_cost_at_the_monitored_size():
    """≈430 quantities × 4 × 10,000 draws (DESIGN §8.2): the whole table's statistics in one
    pass, finite everywhere."""
    x = np.random.default_rng(5).standard_normal((4, 10_000, 430))
    s = dg.summary_many(x)
    assert all(np.isfinite(v).all() for v in s.values())
