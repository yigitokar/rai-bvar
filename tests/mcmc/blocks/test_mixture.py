"""Mixture tables and the indicator kernel (DESIGN §13 WP12, §9.3 item 3; N01 §5, §6, §8, §12.1–12.2;
N02 §2.3 T1, T8).

Constants against the papers' printed digits and the derived N01 §5.4 vectors, the −1.2704
centring (mutant M12 shows up in the mean), log-normal and density accuracy, the offset bias,
the indicator pmf against reference values, χ² goodness of fit of the inverse-CDF draw, Bayes
consistency (the missing ``σ_j⁻¹`` factor of mutant M11 fails it) and a unit Geweke test of the
indicator step with ``h`` fixed (N01 §12.2). The N01 §12.1 "scale equivariance" row needs the
kernel (s-kernel and ``h | s`` kernel with the levels shifted by ``2 log k``), so it lives in
``test_sv_core.py::test_scale_equivariance_at_kernel_level``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from scipy import integrate, special, stats

from rai_bvar.components.vol import mixture as MX
from rai_bvar.components.vol import sv as SV
from rai_bvar.validation.geweke import GewekeConfig, geweke

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")
TABLES = {"ocsn10": MX.OCSN10, "ksc7": MX.KSC7}

# N01 §5.4: κ_j = log p_j − ½ log v_j (OCSN order j = 1..10; KSC in its Table-4 order)
KAPPA_OCSN10 = [-4.00937239, -2.17845316, -1.37686428, -1.12572770, -1.24873234, -1.66194609,
                -2.34338373, -3.35107342, -4.86438228, -7.76421433]  # fmt: skip
KAPPA_KSC7_TABLE4 = [-5.79846149, -2.72885728, -11.64213255, -2.23086873, -0.85570701,
                     -0.86474010, -1.47332606]  # fmt: skip
KSC7_TABLE4_MEANS = [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]
# OCSN (2007) Table 1, left block: KSC already shifted and re-sorted (N01 §5.2)
OCSN_LEFT_BLOCK = np.array(
    [
        [0.04395, 1.50746, 0.16735],
        [0.24566, 0.52478, 0.34023],
        [0.34001, -0.65098, 0.64009],
        [0.25750, -2.35859, 1.26261],
        [0.10556, -5.24321, 2.61369],
        [0.00002, -9.83726, 5.17950],
        [0.00730, -11.40039, 5.79596],
    ]
)


def _logf(xi):
    """Exact ``log χ²₁`` log density (NumPy oracle, from scipy's χ²₁ in logs)."""
    xi = np.asarray(xi, dtype=np.float64)
    return stats.chi2.logpdf(np.exp(xi), 1) + xi


def _logg(xi, tab):
    comps = np.log(tab.p) + stats.norm.logpdf(np.asarray(xi)[..., None], tab.m, np.sqrt(tab.v))
    return special.logsumexp(comps, axis=-1)


def _f(xi):
    return np.exp(_logf(xi))


def _g(xi, tab):
    return np.exp(_logg(xi, tab))


# --------------------------------------------------------------------------- constants (N01 §12.1)


@pytest.mark.parametrize("name", ["ocsn10", "ksc7"])
def test_weights_dtype_and_layout(name):
    tab = TABLES[name]
    assert abs(tab.p.sum() - 1.0) < 1e-12
    for a in (tab.p, tab.m, tab.v, tab.logw, tab.half_inv_v):
        assert a.dtype == np.float64 and not a.flags.writeable
    assert np.all(np.diff(tab.m) < 0)  # descending effective means (OCSN Table 1 order)
    np.testing.assert_array_equal(tab.logw, np.log(tab.p) - 0.5 * np.log(tab.v))
    np.testing.assert_array_equal(tab.half_inv_v, 0.5 / tab.v)
    assert MX.mixture_table(name) is tab


def test_transcription_of_log_weights():
    np.testing.assert_allclose(MX.OCSN10.logw, KAPPA_OCSN10, atol=1e-8)
    # KSC: map back to Table-4 order through the (unique) effective means
    eff = np.array(KSC7_TABLE4_MEANS) - MX.KSC_SHIFT
    order = [int(np.argmin(np.abs(MX.KSC7.m - m))) for m in eff]
    np.testing.assert_allclose(MX.KSC7.logw[order], KAPPA_KSC7_TABLE4, atol=1e-8)


def test_ksc_resort_equals_ocsn_left_block():
    np.testing.assert_allclose(MX.KSC7.p, OCSN_LEFT_BLOCK[:, 0], atol=1e-10)
    np.testing.assert_allclose(MX.KSC7.m, OCSN_LEFT_BLOCK[:, 1], atol=1e-10)
    np.testing.assert_allclose(MX.KSC7.v, OCSN_LEFT_BLOCK[:, 2], atol=1e-10)


@pytest.mark.parametrize(("name", "mean", "var"), [("ocsn10", -1.270280, 4.933731),
                                                   ("ksc7", -1.270399, 4.934854)])  # fmt: skip
def test_centring_and_variance(name, mean, var):
    tab = TABLES[name]
    assert tab.mean == pytest.approx(mean, abs=1e-6)
    assert tab.variance == pytest.approx(var, abs=1e-6)
    assert abs(tab.mean - MX.LOGCHI2_MEAN) < 2e-4  # −1.2704 ± 1e-4 (DESIGN §13 WP12)
    assert abs(tab.variance - math.pi**2 / 2) < 2e-3


def test_double_shift_is_visible_in_the_mean():
    """Mutant M12 (an extra −1.2704 on OCSN-10) moves the mean to −2.540680 (N01 §5.4)."""
    bad = MX.OCSN10.shifted(-MX.KSC_SHIFT)
    assert bad.mean == pytest.approx(-2.540680, abs=1e-6)
    assert abs(bad.mean - MX.LOGCHI2_MEAN) > 1.0


@pytest.mark.parametrize("name", ["ocsn10", "ksc7"])
def test_log_normal_moments(name):
    """``E e^{kξ}`` of ``log χ²₁`` are the χ²₁ moments 1, 3, 15, 105 (N01 §5.4)."""
    tab = TABLES[name]
    mom = [float(np.sum(tab.p * np.exp(k * tab.m + 0.5 * k * k * tab.v))) for k in (1, 2, 3, 4)]
    assert mom[0] == pytest.approx(1.0, abs=1e-3)
    assert mom[1] == pytest.approx(3.0, abs=1e-2)
    assert mom[2] == pytest.approx(15.0, abs=2e-2)
    assert mom[3] == pytest.approx(105.0, abs=2e-2)


@pytest.mark.parametrize(("name", "tol"), [("ocsn10", 5e-4), ("ksc7", 1.1e-2)])
def test_density_accuracy(name, tol):
    tab = TABLES[name]
    xi = np.linspace(-8.759, 1.892, 20001)  # 1st–99th percentiles of log χ²₁
    assert np.max(np.abs(_f(xi) - _g(xi, tab))) < tol
    if name == "ocsn10":
        kl, _ = integrate.quad(lambda x: _f(x) * (_logf(x) - _logg(x, tab)), -60, 8, limit=400)
        assert 0 < kl < 1e-4  # 7.9e-5 per observation (N01 §5.4)


def test_jax_densities_match_scipy():
    x = np.linspace(-30.0, 6.0, 721)
    for tab in TABLES.values():
        np.testing.assert_allclose(np.asarray(MX.log_g(jnp.asarray(x), tab)), _logg(x, tab),
                                   rtol=1e-12, atol=1e-12)  # fmt: skip
        logits = np.asarray(MX.indicator_logits(jnp.asarray(x), tab))
        direct = np.log(tab.p) + stats.norm.logpdf(x[:, None], tab.m, np.sqrt(tab.v))
        np.testing.assert_allclose(logits - 0.5 * MX.LOG_2PI, direct, rtol=1e-12, atol=1e-12)
        assert logits.dtype == np.float64 and logits.shape == (x.size, tab.J)
    np.testing.assert_allclose(np.asarray(MX.log_logchi2_density(jnp.asarray(x))), _logf(x),
                               rtol=1e-12, atol=1e-12)  # fmt: skip


def test_unknown_mixture_and_validation():
    with pytest.raises(ValueError, match="unknown mixture"):
        MX.mixture_table("ksc10")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sum to 1"):
        MX.MixtureTable("bad", [0.5, 0.4], [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        MX.MixtureTable("bad", [0.5, 0.5], [0.0, 1.0], [1.0, -1.0])
    assert MX.COARSE1.J == 1 and MX.COARSE1.mean == pytest.approx(MX.LOGCHI2_MEAN, abs=1e-15)


# --------------------------------------------------------------------------- offset (N01 §7.2, D-2)


def test_y_star_is_log_of_e2_plus_c():
    e2 = np.array([0.0, 1e-300, 1e-30, 1e-12, 0.3, 1.0, 1e5, 1e250])
    for c in (1e-12, 1e-4, 1.0):
        with np.errstate(divide="ignore"):
            le2 = np.log(e2)
        ys = np.asarray(SV.y_star(jnp.asarray(le2), jnp.log(c), jnp.ones(e2.size)))
        np.testing.assert_allclose(ys, np.logaddexp(le2, np.log(c)), rtol=1e-15, atol=1e-27)
    # masked rows are 0 whatever they hold, and an exact zero with c = 0 never leaks into them
    ys = SV.y_star(jnp.asarray([-np.inf, 2.0]), -jnp.inf, jnp.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(ys), [0.0, 2.0])


def _offset_bias(delta):
    """``E log(1 + δ/x²)``, ``x ~ N(0, 1)`` (substitution ``x = e^u`` near 0; N01 §7.2)."""
    f = lambda u: 2 * stats.norm.pdf(np.exp(u)) * np.log1p(delta * np.exp(-2 * u)) * np.exp(u)  # noqa: E731
    return integrate.quad(f, -40, 5, limit=400, points=[0.5 * np.log(delta)])[0]


def test_offset_bias_depends_on_c_over_lambda_only():
    assert _offset_bias(1e-4) == pytest.approx(0.02497, abs=1e-4)
    assert _offset_bias(1e-12) == pytest.approx(math.sqrt(2 * math.pi * 1e-12), rel=1e-3)
    # same c/λ, different scales: the y* shift equals the bias in log-variance units
    r = np.random.default_rng(0)
    x = r.normal(size=400_000)
    for lam in (1e-4, 1.0, 1e4):
        e2 = lam * x * x
        ys = np.asarray(SV.y_star(jnp.log(e2), jnp.log(1e-3 * lam), jnp.ones(x.size)))
        assert np.mean(ys - np.log(e2)) == pytest.approx(_offset_bias(1e-3), abs=5e-4)


# --------------------------------------------------------------------------- indicator pmf (N01 §12.2)


def _pmf(r, tab):
    lw = np.asarray(MX.indicator_logits(jnp.asarray(r, dtype=jnp.float64), tab))
    w = np.exp(lw - lw.max(-1, keepdims=True))
    return w / w.sum(-1, keepdims=True)


def test_pmf_reference_values():
    """OCSN-10 pmf at ``r = −8`` and ``r = 1`` (N01 §12.2, 4 printed decimals)."""
    np.testing.assert_allclose(_pmf(-8.0, MX.OCSN10),
                               [0, 0, 0, 0, 0, 0, 0.0077, 0.5916, 0.3995, 0.0011], atol=6e-5)  # fmt: skip
    np.testing.assert_allclose(_pmf(1.0, MX.OCSN10),
                               [0.0009, 0.1904, 0.5228, 0.2364, 0.0440, 0.0051, 0.0004, 0, 0, 0],
                               atol=6e-5)  # fmt: skip
    # large positive r moves to the high-variance components (N01 §11 item 3)
    assert np.argmax(_pmf(10.0, MX.OCSN10)) == 8 and _pmf(10.0, MX.OCSN10)[8] > 0.9


@pytest.mark.parametrize("name", ["ocsn10", "ksc7"])
def test_indicator_draw_chi2(name):
    """Inverse-CDF draws (D-4) against the exact pmf on a grid of ``r`` (10⁶ draws each)."""
    tab, N = TABLES[name], 1_000_000
    r_grid = [-25.0, -15.0, -8.0, -3.0, -1.27, 0.0, 1.0, 3.0]
    draw = jax.jit(lambda u, r: SV.draw_indicators(u, jnp.full(u.shape, r), tab))
    for k, r in enumerate(r_grid):
        u = random.uniform(random.fold_in(KEY, k), (N,), dtype=jnp.float64)
        s, lg = draw(u, r)
        assert s.dtype == jnp.int32 and lg.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(lg[:1]), _logg(np.array([r]), tab), rtol=1e-12)
        counts = np.bincount(np.asarray(s), minlength=tab.J)
        p = _pmf(r, tab)
        keep = p * N >= 5  # pool tiny cells
        obs = np.append(counts[keep], counts[~keep].sum())
        exp = np.append(p[keep], p[~keep].sum()) * N
        obs, exp = (obs, exp) if exp[-1] > 0 else (obs[:-1], exp[:-1])
        assert counts[p == 0].sum() == 0
        pval = stats.chisquare(obs, exp).pvalue
        assert pval > 1e-4, (name, r, pval)


def _logits_without_sd_factor(r, table):
    """Mutant M11: the pmf without ``σ_j⁻¹`` (N01 §8.3)."""
    d = jnp.asarray(r)[..., None] - table.m
    return jnp.log(table.p) - d * d * table.half_inv_v


def test_bayes_consistency_and_m11_power():
    """``E_r[P(s = j | r)] = p_j`` for ``r`` drawn from the mixture; the M11 kernel fails it."""
    tab, N = MX.OCSN10, 2_000_000
    k1, k2 = random.split(random.fold_in(KEY, 99))
    j = random.categorical(k1, jnp.log(tab.p), shape=(N,))
    r = jnp.asarray(tab.m)[j] + jnp.sqrt(jnp.asarray(tab.v))[j] * random.normal(k2, (N,))

    def mean_probs(fn):
        lw = fn(r, tab)
        return np.asarray(jnp.mean(jax.nn.softmax(lw, axis=-1), axis=0))

    good = mean_probs(MX.indicator_logits)
    se = np.sqrt(tab.p * (1 - tab.p) / N)  # conservative (Rao–Blackwellised means vary less)
    assert np.max(np.abs(good - tab.p) / se) < 5
    bad = mean_probs(_logits_without_sd_factor)
    assert np.max(np.abs(bad - tab.p) / se) > 20


# --------------------------------------------------------------------------- unit Geweke (N01 §12.2)


@dataclass(frozen=True, eq=False)
class IndicatorJoint:
    """``s_t ~ p``, ``y*_t ~ N(h_t + m_s, v_s)`` with ``h`` fixed; the sampler is the production
    indicator draw ``s | y*, h`` (the WP7 ``JointModel`` protocol)."""

    table: MX.MixtureTable
    logits: object = MX.indicator_logits
    name: str = "wp12-indicator"
    h: tuple[float, ...] = (-3.0, -1.0, 0.0, 0.5, 2.0)

    @property
    def names(self):
        return (*(f"1{{s={j}}}" for j in range(self.table.J)), "y*", "y*^2", "y*·s0")

    def prior_sample(self, key):
        s = random.categorical(key, jnp.log(self.table.p), shape=(len(self.h),))
        return s.astype(jnp.int32)

    def simulate(self, key, s):
        z = random.normal(key, (len(self.h),), dtype=jnp.float64)
        return (
            jnp.asarray(self.h)
            + jnp.asarray(self.table.m)[s]
            + jnp.sqrt(jnp.asarray(self.table.v))[s] * z
        )

    def step(self, key, it, s, ys):
        u = random.uniform(random.fold_in(key, it), (len(self.h),), dtype=jnp.float64)
        return SV.draw_indicators(u, ys - jnp.asarray(self.h), self.table, self.logits)[0]

    def test_functions(self, s, ys):
        onehot = jnp.mean(jax.nn.one_hot(s, self.table.J, dtype=jnp.float64), axis=0)
        return jnp.concatenate([onehot, jnp.stack([jnp.mean(ys), jnp.mean(ys**2), ys[0] * s[0]])])


@pytest.mark.geweke
def test_indicator_step_unit_geweke():
    cfg = GewekeConfig(C=500, M_c=100, M1=100_000, alpha=1e-3, seed=3)
    assert geweke(IndicatorJoint(MX.OCSN10), cfg).passed
    bad = geweke(IndicatorJoint(MX.OCSN10, logits=_logits_without_sd_factor, name="m11"), cfg)
    assert not bad.passed  # mutant M11 is caught in seconds (N01 §12.2)
