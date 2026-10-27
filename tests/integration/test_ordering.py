"""Ordering sensitivity (SPEC §6.6, R8; N08 §B4–B5; DESIGN §13 WP22).

N08 §B5 items and their tiers:

1. exact invariance of the analytic NIW engine under permuted data and hyperparameters (1e-10;
   ``hypothesis``) — PR;
2. invariance within MCSE of ``Linear(form="reduced") × Constant("iw_equivalent")`` (two-sample
   tests, Holm at 1%) — nightly (``slow``);
3. prior mode: CKY's ``E u_i² = 2^{i−1}`` under ``a_ij ~ N(0, 1)``, ``λ ≡ 1``; the IW-equivalent
   prior's ``Σ_ii ~ IG(3/2, ψ_i/2)`` at every position (KS) — PR;
4. IW equivalence (Chan 2022, Prop. 1): ``(A, λ)`` draws against direct ``IW(n + 2, diag ψ)``
   draws (KS on ``log|Σ|``, ``log Σ_ii``, correlations) — PR;
5. detection power on CKY's §6.1 DGP and the false-alarm control — weekly (``slow``, ``weekly``);
6. un-permutation with ``×10ⁱ`` variable scales — PR.

PR-tier MCMC runs share one ``StaticConfig`` (``n = 3``, ``T = 80``, 2 chains), so the engine
compiles once per process.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats
from scipy.special import logsumexp

import rai_bvar as rb
from rai_bvar import BVAR, SV, Constant, Linear, MCMCSettings, Minnesota, VARData
from rai_bvar.data.design import regressor_names
from rai_bvar.evaluation.backtest import backtest
from rai_bvar.forecast import check_structure
from rai_bvar.models import OLSVAR
from rai_bvar.structural import ordering as so
from rai_bvar.structural.ordering import (
    THRESHOLDS,
    OrderingSensitivityResult,
    ordering_sensitivity,
    permutation_design,
    permute_model,
    prior_covariance_draws,
    prior_draws_from_arrays,
)

MCMC_SMALL = MCMCSettings(chains=2, draws=60, burn=60)


def var_data(
    n: int = 3,
    T: int = 80,
    seed: int = 0,
    scales=None,
    names=None,
    tcodes: int = 1,
    persistence: float = 0.5,
) -> VARData:
    """A stationary VAR(1) with correlated shocks; variable ``i`` has scale ``scales[i]``
    (``A = D A₀ D⁻¹``, shocks ``D C z``)."""
    rng = np.random.default_rng(seed)
    s = np.ones(n) if scales is None else np.asarray(scales, dtype=np.float64)
    A0 = persistence * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / n
    A = (s[:, None] * A0) / s[None, :]
    C = np.linalg.cholesky(0.5 * np.eye(n) + 0.5 / n)
    y = np.zeros((T + 50, n))
    for t in range(1, T + 50):
        y[t] = A @ y[t - 1] + s * (C @ rng.standard_normal(n))
    names = [f"y{i}" for i in range(n)] if names is None else list(names)
    frame = pd.DataFrame(
        y[50:], index=pd.period_range("1990Q1", periods=T, freq="Q"), columns=names
    )
    return VARData.from_frame(frame, tcodes=tcodes)


def analytic(**prior) -> BVAR:
    return BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed", **prior)), lags=1)


def mcmc_model(form: str = "reduced", vol=None, **prior) -> BVAR:
    return BVAR(
        mean=Linear(prior=Minnesota(form=form, hierarchy="fixed", **prior)),
        vol=Constant("iw_equivalent") if vol is None else vol,
        lags=1,
        mcmc=MCMC_SMALL,
    )


@pytest.fixture(scope="module")
def data3() -> VARData:
    return var_data(3, 80, seed=1, names=["gdp", "infl", "rate"])


@pytest.fixture(scope="module")
def scaled4() -> VARData:
    """Item 6: variable ``i`` has scale ``10^i``, so a label mix-up is an order of magnitude."""
    return var_data(4, 80, seed=2, scales=[1.0, 10.0, 100.0, 1000.0], names=list("abcd"))


@pytest.fixture(scope="module")
def scaled3() -> VARData:
    return var_data(3, 80, seed=3, scales=[1.0, 10.0, 100.0], names=list("abc"))


# =========================================================================== permute_model


def test_permute_model_moves_the_psi_tuple_with_its_variables():
    m = analytic(psi=(1.0, 2.0, 3.0))
    by_name = permute_model(m, ["c", "a", "b"], variables=["a", "b", "c"])
    by_index = permute_model(m, [2, 0, 1])
    assert by_name.mean.prior.psi == by_index.mean.prior.psi == (3.0, 1.0, 2.0)
    assert by_name.engine == "analytic" and by_name.lags == m.lags
    # the inverse permutation restores the model
    assert permute_model(by_name, ["a", "b", "c"], variables=["c", "a", "b"]) == m


def test_permute_model_leaves_name_keyed_and_data_dependent_settings_alone():
    for m in (analytic(), analytic(mean={"a": 0.3}), mcmc_model(), OLSVAR(1)):
        assert permute_model(m, ["b", "a"], variables=["a", "b"]) is m
    # names without the current ordering are enough when nothing is positional
    assert permute_model(analytic(), ["b", "a"]) == analytic()


@pytest.mark.parametrize(
    ("order", "kwargs", "match"),
    [
        (["a", "a"], {}, "repeats"),
        (["a", "c"], {"variables": ["a", "b"]}, "not a permutation"),
        ([0, 2], {}, "not a permutation"),
        (["b", "a"], {}, "needs variables="),
    ],
)
def test_permute_model_rejects_bad_orders(order, kwargs, match):
    with pytest.raises(ValueError, match=match):
        permute_model(analytic(psi=(1.0, 2.0)), order, **kwargs)


# =========================================================================== design


def test_all_orderings_when_they_fit_in_n_perm():
    orders = permutation_design(["a", "b", "c"], n_perm=10)
    assert orders[0] == ("a", "b", "c") and len(orders) == 6 == len(set(orders))
    assert set(orders) == set(itertools.permutations("abc"))
    all24 = permutation_design(list("abcd"), n_perm=3, design="all")  # ARRS: all 24
    assert len(all24) == 24 and all24[0] == tuple("abcd")


def test_balanced_design_is_a_latin_square_after_the_user_ordering_and_its_reverse():
    free = list("abcde")
    for seed in range(5):
        orders = permutation_design(free, n_perm=7, seed=seed)
        assert orders[0] == tuple(free) and orders[1] == tuple(reversed(free))
        assert len(set(orders)) == 7
        square = np.array([[free.index(v) for v in o] for o in orders[2:]])
        for col in square.T:  # every variable in every position exactly once
            assert sorted(col) == list(range(5))
        assert permutation_design(free, n_perm=7, seed=seed) == orders  # deterministic
    assert permutation_design(free, 7, seed=0) != permutation_design(free, 7, seed=1)


def test_balanced_design_with_three_free_variables_uses_the_cyclic_shifts():
    """Every 3×3 Latin square holds the even or the odd permutations, so the user ordering or its
    reverse: the user ordering and its two cyclic shifts form the balanced square."""
    assert all(  # the parity argument, by enumeration
        {(0, 1, 2), (2, 1, 0)} & set(rows)
        for rows in itertools.permutations(itertools.permutations(range(3)), 3)
        if all(sorted(col) == [0, 1, 2] for col in zip(*rows, strict=True))
    )
    for pinned, free in (((), list("abc")), (("p",), list("abc"))):
        for seed in range(6):
            for n_perm in (4, 5):
                orders = permutation_design([*pinned, *free], n_perm, pinned, seed=seed)
                assert len(orders) == n_perm == len(set(orders))
                assert all(o[: len(pinned)] == pinned for o in orders)
                rows = [o[len(pinned) :] for o in orders]
                assert rows[:2] == [tuple(free), tuple(reversed(free))]
                square = np.array([[free.index(v) for v in r] for r in (rows[0], *rows[2:4])])
                for col in square.T:  # every variable in every position exactly once
                    assert sorted(col) == [0, 1, 2]
    three = permutation_design(list("abc"), n_perm=3, seed=0)  # one cyclic shift fits
    assert three[2] in {tuple("bca"), tuple("cab")}


def test_random_design_tops_up_and_pinned_variables_lead():
    orders = permutation_design(list("abcdef"), n_perm=12, pinned=["d", "a"], design="random")
    assert len(orders) == 12 == len(set(orders))
    assert all(o[:2] == ("d", "a") for o in orders)
    assert orders[0] == ("d", "a", "b", "c", "e", "f")
    assert orders[1] == ("d", "a", "f", "e", "c", "b")
    big = permutation_design(list("abcdef"), n_perm=40, seed=3)  # Latin rows + uniform top-up
    assert len(big) == 40 == len(set(big))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"pinned": ["zz"]}, ValueError, "not in the data"),
        ({"pinned": ["a", "b"]}, ValueError, "two free"),
        ({"design": "latin"}, ValueError, "design must be"),
        ({"n_perm": 0}, ValueError, "positive int"),
        ({"n_perm": 1}, ValueError, "at least 2"),
    ],
)
def test_design_validation(kwargs, error, match):
    with pytest.raises(error, match=match):
        permutation_design(["a", "b", "c", "d"][: 3 if "pinned" in kwargs else 4], **kwargs)


def test_design_all_has_a_size_cap():
    with pytest.raises(ValueError, match="orderings"):
        permutation_design([f"v{i}" for i in range(8)], design="all")


# =========================================================================== item 1: analytic exact


def _unpermute_posterior(post, order, variables, lags, det):
    idx = [list(order).index(v) for v in variables]
    regs_fit = regressor_names(list(order), lags, det)
    ridx = [regs_fit.index(r) for r in regressor_names(list(variables), lags, det)]
    return (
        post.mean[np.ix_(ridx, idx)],
        post.scale[np.ix_(idx, idx)],
        post.omega()[np.ix_(ridx, ridx)],
    )


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(2, 6),
    lags=st.integers(1, 2),
    det=st.sampled_from(["c", "ct", "n"]),
    psi=st.sampled_from(["ar", "tuple"]),
    soc=st.sampled_from([None, 1.0]),
    seed=st.integers(0, 2**16),
    perm_seed=st.integers(0, 2**16),
)
def test_analytic_engine_is_exactly_order_invariant(n, lags, det, psi, soc, seed, perm_seed):
    """N08 §B5 item 1: permuted data and hyperparameters, un-permuted outputs equal to 1e-10."""
    data = var_data(n, 50, seed=seed)
    psi_value = "ar" if psi == "ar" else tuple(np.linspace(0.5, 2.0, n))
    prior = {"psi": psi_value, "soc": soc, "sur": soc, "mean": {data.variables[0]: 0.7}}
    model = BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed", **prior)), lags=lags,
                 deterministic=det)  # fmt: skip
    order = tuple(np.random.default_rng(perm_seed).permutation(data.variables))
    fit0 = model.fit(data)
    fit1 = permute_model(model, order, variables=data.variables).fit(data.select(order))
    p0, p1 = fit0.posterior(), fit1.posterior()
    B1, S1, O1 = _unpermute_posterior(p1, order, data.variables, lags, det)
    for a, b in ((p0.mean, B1), (p0.scale, S1), (p0.omega(), O1)):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10 * np.abs(a).max())
    assert p1.df == p0.df
    assert fit1.log_ml() == pytest.approx(fit0.log_ml(), rel=1e-10, abs=1e-10)
    idx = [list(order).index(v) for v in data.variables]
    np.testing.assert_allclose(fit1.psi[idx], fit0.psi, rtol=1e-12)


def test_ordering_sensitivity_unpermutes_analytic_posteriors_exactly(scaled4):
    """The check's own un-permutation: with fixed hyperparameters every ordering's posterior
    means of ``B`` and ``Σ_T`` and ψ agree to 1e-10 (item 1 through ``ordering_sensitivity``)."""
    res = ordering_sensitivity(analytic(), scaled4, n_perm=6, noise_reps=1, draws=200)
    ds = res.dataset
    for name in ("B_post", "sigma_post", "psi"):
        x = ds[name].values
        np.testing.assert_allclose(
            x, np.broadcast_to(x[:1], x.shape), rtol=1e-10, atol=1e-10 * np.abs(x).max()
        )
    assert ds["hyper_post"].sel(hyper="lambda").values.tolist() == [0.2] * 6
    regs = regressor_names(list("abcd"), 1, "c")
    assert list(ds["regressor"].values) == regs and list(ds["variable"].values) == list("abcd")


def test_eb_hyperparameters_are_order_invariant_to_optimiser_tolerance(scaled3):
    res = ordering_sensitivity(rb.BVAR(lags=1), scaled3, n_perm=6, noise_reps=0, draws=100)
    lam = res.dataset["hyper_post"].sel(hyper="lambda").values
    np.testing.assert_allclose(lam, lam[0], rtol=1e-5)


# =========================================================================== item 6: ×10ⁱ scales


def _scale_checks(res: OrderingSensitivityResult, scales: np.ndarray) -> None:
    sd = res.dataset["value"].sel(stat="sd", horizon=1).values  # (ordering, variable)
    ratio = sd / scales[None, :]
    assert (ratio > 0.3).all() and (ratio < 3.0).all(), ratio
    sig = np.diagonal(res.dataset["sigma_post"].values, axis1=1, axis2=2)
    assert ((sig / scales**2 > 0.3) & (sig / scales**2 < 3.0)).all()
    mean = res.dataset["value"].sel(stat="mean").values
    assert (np.abs(mean) < 5 * scales[None, :, None]).all()


def test_unpermutation_with_order_of_magnitude_scales_analytic(scaled4):
    res = ordering_sensitivity(analytic(), scaled4, n_perm=6, noise_reps=2, draws=400)
    _scale_checks(res, 10.0 ** np.arange(4))
    pos = res.dataset["position"].values
    for k, order in enumerate(res.orderings):
        assert [order[p] for p in pos[k]] == list("abcd")
    assert res.orderings[0] == tuple("abcd") and res.orderings[1] == tuple("dcba")
    # an invariant model: the noise floor explains the dispersion, nothing is flagged
    assert not res.material
    assert res.dataset.attrs["mode"] == "forecast" and json.loads(res.dataset.attrs["orderings"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_unpermutation_with_order_of_magnitude_scales_mcmc(scaled3):
    res = ordering_sensitivity(mcmc_model(), scaled3, n_perm=6, noise_reps=2, draws=120)
    _scale_checks(res, 10.0 ** np.arange(3))
    psi = res.dataset["psi"].values  # ψ = AR residual variance of the variable itself: bitwise
    assert (psi == psi[:1]).all()
    assert ((psi[0] / 100.0 ** np.arange(3) > 0.3) & (psi[0] / 100.0 ** np.arange(3) < 3)).all()
    assert res.dataset["converged"].dims == ("ordering",)
    # noise replicates refit with their own seeds, distinct from the orderings' and each other's
    fit_seeds = json.loads(res.dataset.attrs["fit_seeds"])
    fc_seeds = json.loads(res.dataset.attrs["forecast_seeds"])
    assert len(fit_seeds) == 8 and fit_seeds[:6] == [0] * 6
    assert len(set(fit_seeds[6:])) == 2 and 0 not in fit_seeds[6:]
    # ... and those recorded seeds are the ones used: replaying the last replicate reproduces it
    fc = mcmc_model().fit(scaled3, seed=fit_seeds[-1]).forecast(8, 120, seed=fc_seeds[-1])
    x = fc.paths.sel(horizon=[1, 4, 8], variable=list("abc")).values.reshape(-1, 3, 3)
    np.testing.assert_allclose(
        res.dataset["noise"].isel(rep=-1).sel(stat="sd").values, x.std(axis=0, ddof=1).T,
        rtol=1e-12,
    )  # fmt: skip


def test_pooled_predictive_is_the_equal_weight_mixture(scaled3):
    origin = scaled3.index[-5]
    res = ordering_sensitivity(
        analytic(), scaled3, n_perm=6, noise_reps=1, draws=150, origins=str(origin), pool=True
    )
    pooled = res.pooled
    assert pooled.variables == list("abc") and pooled.n_samples == 6 * 150
    assert pooled.dataset.sizes["chain"] == 6
    assert json.loads(pooled.dataset.attrs["pooled_orderings"]) == [list(o) for o in res.orderings]
    check_structure(pooled)
    y = scaled3.realized(origin, 4)
    # mixture LPD = logsumexp over the per-ordering draws − log(total) (N08 §B4)
    per = res.dataset["joint"].sel(jstat="lpd")  # (ordering, horizon)
    for h in (1, 4):
        mix = logsumexp(per.sel(horizon=h).values) - math.log(per.sizes["ordering"])
        assert pooled.log_pred_density(y[h - 1], h) == pytest.approx(mix, abs=1e-9)
    # each ordering's own forecast lives in the pool, un-permuted (paths by label)
    paths = pooled.paths.values  # (6, 150, H, 3)
    assert np.isfinite(paths).all() and paths.shape[-1] == 3
    sd = paths.std(axis=(0, 1))[0]
    assert (sd / 10.0 ** np.arange(3) > 0.3).all() and (sd / 10.0 ** np.arange(3) < 3).all()


def test_forecast_scores_at_an_inner_origin(scaled3):
    """Item 6 for the scores: with ``×10ⁱ`` scales a label mix-up in the per-variable LPD or
    CRPS is off by orders of magnitude. The analytic model is exactly order-invariant, so every
    ordering scores alike up to forecast noise; each ordering's scores equal a direct scoring of
    its own forecast by label; the sd's MCSE is the iid delta-method value (analytic draws are
    independent)."""
    import scoringrules as sr

    origin, hz, draws = scaled3.index[-9], (1, 2, 8), 400
    res = ordering_sensitivity(
        analytic(), scaled3, n_perm=6, noise_reps=1, draws=draws, origins=[origin], horizons=hz
    )
    ds, v = res.dataset, res.dataset["value"]
    assert np.isfinite(v.sel(stat=["lpd", "crps"]).values).all()
    assert np.isfinite(ds["joint"].values).all()
    assert np.isfinite(ds["mcse"].sel(stat=["mean", "sd", "q0.5", "len90", "lpd"])).all()
    assert ds.attrs["realized"] == 1 and ds.attrs["origin"] == str(origin)
    # invariance: LPD within forecast noise (ptp ≈ 0.07–0.12 nats; scoring another variable's
    # density moves it by ≈ 4 nats), CRPS within a factor 2 and of the variable's scale
    lpd, crps = v.sel(stat="lpd").values, v.sel(stat="crps").values  # (ordering, V, H)
    assert np.ptp(lpd, axis=0).max() < 0.5, np.ptp(lpd, axis=0)
    assert (crps.max(axis=0) / crps.min(axis=0)).max() < 2.0
    rel = crps / (10.0 ** np.arange(3))[None, :, None]
    assert ((rel > 0.05) & (rel < 5.0)).all(), rel
    assert not res.material, res.summary()
    # each ordering's scores are its own forecast's, by label (the reversed ordering)
    window, y = scaled3.upto(origin), scaled3.realized(origin, max(hz))
    seeds = json.loads(ds.attrs["forecast_seeds"])
    j, order = 1, res.orderings[1]
    fc = permute_model(analytic(), order, variables=scaled3.variables).fit(window.select(order))
    fc = fc.forecast(max(hz), draws, seed=seeds[j])
    for i, name in enumerate(scaled3.variables):
        for h in hz:
            cell = v.sel(ordering=j, variable=name, horizon=h)
            want_lpd = fc.log_pred_density(y[h - 1, i : i + 1], h, [name])
            ens = fc.paths.sel(horizon=h, variable=name).values.ravel()
            want_crps = sr.crps_ensemble(y[h - 1, i], ens, estimator="pwm")
            assert cell.sel(stat="lpd").item() == pytest.approx(want_lpd, rel=1e-12)
            assert cell.sel(stat="crps").item() == pytest.approx(float(want_crps), rel=1e-12)
    # MCSE of the sd (delta method) against the iid value √Var((x − x̄)²)/(2·sd·√S) of the
    # user ordering's own draws; the ESS estimate moves it by ≈ ±25%
    x = analytic().fit(window).forecast(max(hz), draws, seed=seeds[0]).paths
    x = x.sel(horizon=list(hz), variable=list(scaled3.variables)).values.reshape(-1, 3, 3)
    sd = x.std(axis=0, ddof=1)  # (H, V)
    iid = ((x - x.mean(axis=0)) ** 2).std(axis=0, ddof=1) / (2.0 * sd * math.sqrt(len(x)))
    np.testing.assert_allclose(v.sel(ordering=0, stat="sd").values, sd.T, rtol=1e-12)
    np.testing.assert_allclose(ds["mcse"].sel(stat="sd").values, iid.T, rtol=0.35)
    no_truth = ordering_sensitivity(analytic(), scaled3, n_perm=3, noise_reps=0, draws=100)
    assert np.isnan(no_truth.dataset["value"].sel(stat="lpd").values).all()


# =========================================================================== batch


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_batched_fits_equal_looped_fits(scaled3):
    """DESIGN §5.2: one fit-axis program equals separate fits to rounding, 1e-10 (observed
    ≈ 3e-14), not bitwise."""
    kw = {"n_perm": 3, "noise_reps": 1, "draws": 120}
    loop = ordering_sensitivity(mcmc_model(), scaled3, **kw)
    batch = ordering_sensitivity(mcmc_model(), scaled3, batch=True, **kw)
    for name in ("value", "noise", "B_post", "sigma_post", "mcse"):
        a, b = loop.dataset[name].values, batch.dataset[name].values
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12 * np.nanmax(np.abs(a)))
    assert batch.dataset.attrs["batch"] == 1


# =========================================================================== items 3 and 4: prior


def test_prior_mode_reproduces_the_cky_two_to_the_i_growth():
    """N08 §B5 item 3: ``a_ij ~ N(0, 1)``, ``λ ≡ 1``, no SV ⇒ ``E u_i² = 2^{i−1}`` (CKY §2)."""
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.state import ConstantPrior, ContempPriorArrays
    from rai_bvar.settings import Runtime

    n = 6
    data = var_data(n, 60, seed=4)
    model = mcmc_model("structural", Constant("ig"))
    r = resolve(model, data, Runtime())
    eq = r.prior.eq
    c = eq.contemp
    eq = replace(
        eq,
        contemp=ContempPriorArrays(
            mean0=np.zeros_like(c.mean0), prec0=np.ones_like(c.prec0), cmask=c.cmask
        ),
        vol_core=ConstantPrior(shape=np.full(n, 1e10), scale=np.full(n, 1e10)),  # λ ≈ 1 ± 1e-5
    )
    sigma, shock = prior_draws_from_arrays(
        r.static, replace(r.prior, eq=eq), int(r.data.T_eff), draws=40_000, seed=7
    )
    var = np.diagonal(sigma, axis1=1, axis2=2)
    se = var.std(axis=0, ddof=1) / math.sqrt(var.shape[0])
    target = 2.0 ** np.arange(n)
    assert (np.abs(var.mean(axis=0) - target) < 5 * se).all(), (var.mean(axis=0), se)
    assert (np.abs(shock.var(axis=0) / target - 1) < 0.1).all()


@pytest.fixture(scope="module")
def iw_equivalent_prior():
    """Item 4 set-up: reduced × ``Constant("iw_equivalent")``, ψ fixed, ``n = 5``."""
    data = var_data(5, 60, seed=5)
    psi = (1.0, 2.0, 0.5, 4.0, 3.0)
    model = mcmc_model(psi=psi)
    draws = prior_covariance_draws(model, data, draws=40_000, seed=11)
    order = ("y3", "y0", "y4", "y2", "y1")
    permuted = prior_covariance_draws(
        permute_model(model, order, variables=data.variables), data.select(order), 40_000, 12
    )
    return np.array(psi), draws, permuted, data


def test_iw_equivalent_marginals_are_ig_three_halves_at_every_position(iw_equivalent_prior):
    """N08 §B5 item 3 (corrected): ``Σ_ii ~ IG((ν₀ − n + 1)/2, ψ_i/2) = IG(3/2, ψ_i/2)`` for every
    variable whatever its position (KS, not the infinite-variance mean)."""
    psi, draws, permuted, data = iw_equivalent_prior
    for pdraws in (draws, permuted):
        for i, v in enumerate(pdraws.variables):
            j = data.variables.index(v)
            x = pdraws.sigma[:, i, i]
            p = stats.kstest(x, stats.invgamma(a=1.5, scale=psi[j] / 2.0).cdf).pvalue
            assert p > 1e-3, (v, i, p)


def test_iw_equivalent_prior_is_inverse_wishart(iw_equivalent_prior):
    """N08 §B5 item 4 (Chan 2022, Prop. 1): ``(A, λ)`` draws vs direct ``IW(n + 2, diag ψ)``."""
    psi, draws, _, _ = iw_equivalent_prior
    n = psi.size
    direct = stats.invwishart(df=n + 2, scale=np.diag(psi)).rvs(40_000, random_state=13)
    a, b = draws.sigma, direct

    def corr(s, i, j):
        return s[:, i, j] / np.sqrt(s[:, i, i] * s[:, j, j])

    checks = {
        "logdet": (np.linalg.slogdet(a)[1], np.linalg.slogdet(b)[1]),
        "log S11": (np.log(a[:, 0, 0]), np.log(b[:, 0, 0])),
        "log S55": (np.log(a[:, 4, 4]), np.log(b[:, 4, 4])),
        "corr51": (corr(a, 4, 0), corr(b, 4, 0)),
        "corr32": (corr(a, 2, 1), corr(b, 2, 1)),
    }
    for name, (x, y) in checks.items():
        assert stats.ks_2samp(x, y).pvalue > 1e-3, name


def test_prior_mode_shows_mode_one_sv_position_effects_and_no_iw_equivalent_ones(data3):
    inv = ordering_sensitivity(mcmc_model(), data3, n_perm=6, mode="prior", draws=8000)
    assert not inv.material
    assert set(inv.dataset["value"].dims) == {"ordering", "variable", "stat"}
    med = inv.dataset["value"].sel(stat="var_q50").values
    np.testing.assert_allclose(med, np.broadcast_to(med.mean(axis=0), med.shape), rtol=0.1)
    sv = ordering_sensitivity(
        mcmc_model("structural", SV()), data3, n_perm=6, mode="prior", draws=8000
    )
    assert sv.material  # Mode I centres log λ_i at log(ψ_i/ι): the variance falls with position
    pos = sv.by_position().sel(stat="var_q50")
    assert pos.dims == ("ordering", "position")
    assert (pos.variable_at.values[:, 0] == np.array([o[0] for o in sv.orderings])).all()
    table = sv.summary()
    assert table.index.name == "variable" and table["material"].any()
    corr = sv.dataset["corr"]
    assert corr.dims == ("ordering", "variable", "variable_", "cstat")
    np.testing.assert_allclose(corr.sel(cstat="q50").values.diagonal(axis1=1, axis2=2), 1.0)


def test_analytic_prior_mode_is_inverse_wishart(scaled3):
    psi = (1.0, 4.0, 9.0)
    res = ordering_sensitivity(analytic(psi=psi), scaled3, n_perm=3, mode="prior", draws=20_000)
    med = res.dataset["value"].sel(stat="var_q50").values
    target = np.array(psi) / 2.0 / stats.gamma(1.5).median()  # median of IG(3/2, ψ/2)
    np.testing.assert_allclose(med, np.broadcast_to(target, med.shape), rtol=0.05)
    assert not res.material


# =========================================================================== backtest mode


def test_backtest_mode_unpermutes_origin_scores(scaled3):
    origins = [str(p) for p in scaled3.index[-12:-2:4]]
    model = analytic()
    res = ordering_sensitivity(
        model, scaled3, n_perm=3, mode="backtest", horizons=(1, 2), origins=origins,
        noise_reps=1, draws=150,
    )  # fmt: skip
    ds = res.dataset
    assert ds["origin_scores"].dims == ("ordering", "origin", "variable", "horizon", "score")
    fit_seeds = json.loads(ds.attrs["fit_seeds"])  # the models' own sampler seeds
    assert fit_seeds[:3] == [0] * 3 and len(fit_seeds) == 4 and fit_seeds[3] != 0
    seeds = json.loads(ds.attrs["backtest_seeds"])
    j = 1  # the reversed ordering, scored by label
    order = res.orderings[j]
    direct = backtest(
        permute_model(model, order, variables=scaled3.variables), scaled3.select(order),
        origins=origins, horizon=2, draws=150, seed=seeds[j],
    ).scores  # fmt: skip
    for v in scaled3.variables:
        np.testing.assert_array_equal(
            ds["origin_scores"].sel(ordering=j, variable=v, score="lpd").values,
            direct["lpd_marginal"].sel(variable=v).transpose("origin", "horizon").values,
        )
    np.testing.assert_array_equal(
        ds["origin_joint_lpd"].sel(ordering=j).values,
        direct["lpd"].transpose("origin", "horizon").values,
    )
    lpd = ds["value"].sel(stat="lpd").values
    np.testing.assert_allclose(
        lpd, ds["origin_scores"].sel(score="lpd").mean("origin").values, rtol=1e-12
    )
    rc = ds["rank_corr"].values
    assert rc.shape == (8, 8) and np.allclose(rc, rc.T, equal_nan=True)
    p = ds["mcs_pvalue"].values
    assert ((p >= 0) & (p <= 1)).all() and (p.max(axis=0) == 1).all()
    assert list(res.summary().columns[:3]) == ["lpd_range", "lpd_range_snr", "lpd_range_flag"]


# =========================================================================== metrics and flags


def _toy(values, noise, mcse, joint=None, stats_=("mean", "sd", "q0.05", "lpd")):
    K, V, H, _ = values.shape
    joint = np.zeros((K, H, 1)) if joint is None else joint
    ds = xr.Dataset(
        {
            "value": (("ordering", "variable", "horizon", "stat"), values),
            "joint": (("ordering", "horizon", "jstat"), joint),
            "noise": (("rep", "variable", "horizon", "stat"), noise),
            "noise_joint": (("rep", "horizon", "jstat"), np.zeros((noise.shape[0], H, 1))),
            "mcse": (("variable", "horizon", "stat"), mcse),
            "mcse_joint": (("horizon", "jstat"), np.full((H, 1), 0.01)),
            "position": (("ordering", "variable"), np.tile(np.arange(V), (K, 1))),
        },
        coords={
            "ordering": np.arange(K),
            "variable": [f"v{i}" for i in range(V)],
            "horizon": np.arange(1, H + 1),
            "stat": list(stats_),
            "jstat": ["lpd"],
            "rep": np.arange(noise.shape[0]),
        },
        attrs={"mode": "forecast", "orderings": json.dumps([[f"v{i}" for i in range(V)]] * K)},
    )
    return OrderingSensitivityResult(ds)


def test_flags_need_the_threshold_and_a_signal_above_the_noise():
    K, V, H = 4, 2, 1
    base = np.zeros((K, V, H, 4))
    base[..., 1] = 1.0  # sd = 1
    values = base.copy()
    values[:, 0, 0, 1] = [1.0, 1.0, 1.3, 1.3]  # v0: sd ratio 1.3, sd over orderings ≈ 0.17
    values[:, 1, 0, 0] = [0.0, 0.0, 0.05, 0.05]  # v1: mean shift 0.05 (< 0.10)
    values[:, 0, 0, 3] = [-1.0, -1.0, -1.2, -1.2]  # v0: lpd range 0.2
    noise = np.repeat(base[:1], 3, axis=0)
    noise[:, :, :, :] += np.array([0.0, 0.01, -0.01])[:, None, None, None]  # σ_MC = 0.01
    res = _toy(values, noise, np.full((V, H, 4), 0.005))
    m = res.metrics()
    assert m["sd_ratio"].values[0, 0] == pytest.approx(1.3)
    assert bool(m["sd_ratio_flag"].values[0, 0]) and not bool(m["sd_ratio_flag"].values[1, 0])
    assert m["mean_shift"].values[1, 0] == pytest.approx(0.05) and not m["mean_shift_flag"].any()
    assert bool(m["lpd_range_flag"].values[0, 0])
    assert m["q_spread"].values.tolist() == [[0.0], [0.0]]
    assert res.material
    # a noise floor as large as the dispersion silences every flag (signal-to-noise ≤ 3)
    loud = noise.copy()
    loud[:, :, :, :] += np.array([0.0, 0.5, -0.5])[:, None, None, None]
    assert not _toy(values, loud, np.full((V, H, 4), 0.005)).material
    # thresholds are overridable; unknown names are rejected
    assert not res.is_material({"sd_ratio": 2.0, "lpd_range": 1.0})
    with pytest.raises(KeyError, match="unknown threshold"):
        res.metrics({"nope": 1.0})
    table = res.summary()
    assert table.loc[("v0", 1), "material"] and not table.loc[("v1", 1), "material"]
    assert table.loc[("joint", 1), "lpd_range"] == 0.0
    assert table.attrs["thresholds"] == THRESHOLDS and table.attrs["material"]


def test_a_joint_lpd_range_alone_makes_the_result_material():
    """Only the joint LPD moves (range 0.2 > 0.10, signal-to-noise ≈ 11.5): flagged, material."""
    K, V, H = 4, 2, 1
    values = np.zeros((K, V, H, 4))
    values[..., 1] = 1.0  # sd = 1 everywhere: no per-variable metric moves
    joint = np.array([-2.0, -2.0, -2.2, -2.2]).reshape(K, H, 1)
    res = _toy(values, np.repeat(values[:1], 3, axis=0), np.full((V, H, 4), 0.005), joint)
    m = res.metrics()
    assert m["lpd_range_joint"].values.tolist() == pytest.approx([0.2])
    assert m["lpd_range_joint_snr"].values[0] == pytest.approx(np.std(joint, ddof=1) / 0.01)
    flagged = [k for k in m.data_vars if k.endswith("_flag") and bool(m[k].values.any())]
    assert flagged == ["lpd_range_joint_flag"]
    assert res.material and res.is_material()
    table = res.summary()
    assert table.loc[("joint", 1), "material"] and not table.loc[("v0", 1), "material"]
    assert table.attrs["material"]
    assert not res.is_material({"lpd_range_joint": 0.3})  # below the joint threshold
    assert not res.is_material({"signal_to_noise": 20.0})  # under the noise floor


def test_sigma_mc_takes_the_larger_of_replicate_sd_and_mcse():
    values = np.zeros((3, 1, 1, 4))
    values[..., 1] = [[[1.0]], [[1.1]], [[1.2]]]
    noise = np.zeros((2, 1, 1, 4))
    noise[..., 1] = [[[1.0]], [[1.02]]]  # sd ≈ 0.0141
    mcse = np.array([[[np.nan, 0.05, np.nan, np.nan]]])
    res = _toy(values, noise, mcse)
    sig = res.sigma_mc()[0, 0]
    assert sig[1] == pytest.approx(0.05) and sig[0] == 0.0
    assert res.signal_to_noise()[0, 0, 1] == pytest.approx(0.1 / 0.05)
    one = _toy(values, noise[:1], np.full((1, 1, 4), np.nan))  # one replicate, no MCSE
    assert np.isnan(one.sigma_mc()).all() and not one.material


def test_plot_and_repr(scaled3):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = ordering_sensitivity(analytic(), scaled3, n_perm=3, noise_reps=1, draws=100)
    fig = res.plot()
    assert len(fig.axes) == 3
    plt.close(fig)
    fig = res.plot("mean", relative=False)
    plt.close(fig)
    with pytest.raises(ValueError, match="stat must be one of"):
        res.plot("nope")
    assert repr(res).startswith("OrderingSensitivityResult(mode='forecast', orderings=3")


# =========================================================================== validation


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"mode": "nowcast"}, ValueError, "mode must be"),
        ({"pool": True, "mode": "prior"}, ValueError, "pool=True"),
        ({"batch": True}, ValueError, "MCMC engine"),
        ({"batch": True, "mode": "backtest"}, ValueError, "batch=True"),
        ({"horizons": (0,)}, ValueError, "positive int"),
        ({"horizons": (1, 1)}, ValueError, "repeat"),
        ({"quantiles": (0.5, 1.0)}, ValueError, r"\(0, 1\)"),
        ({"noise_reps": -1}, ValueError, "non-negative"),
        ({"noise_reps": 0, "mode": "backtest"}, ValueError, "noise_reps >= 1"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1.5}, TypeError, "seed"),
        ({"pool": 1}, TypeError, "pool must be a bool"),
        ({"cache": "x"}, ValueError, "mode='backtest'"),
        ({"origins": ["2000Q1", "2001Q1"]}, ValueError, "one origin"),
        ({"origins": "1900Q1"}, ValueError, "outside the sample"),
        ({"pinned": ["gdp", "infl"]}, ValueError, "two free"),
    ],
)
def test_ordering_sensitivity_validation(data3, kwargs, error, match):
    with pytest.raises(error, match=match):
        ordering_sensitivity(analytic(), data3, **kwargs)


def test_prior_mode_needs_a_bvar(data3):
    with pytest.raises(TypeError, match="needs a BVAR"):
        ordering_sensitivity(OLSVAR(1), data3, mode="prior")


def test_generic_models_run_in_forecast_mode(data3):
    """Models without posterior summaries (here ``OLSVAR``) still get the forecast statistics."""
    res = ordering_sensitivity(OLSVAR(1), data3, n_perm=3, noise_reps=1, draws=400)
    assert "B_post" not in res.dataset and res.dataset["value"].sizes["ordering"] == 3
    mean = res.dataset["value"].sel(stat="mean").values
    np.testing.assert_allclose(mean[1], mean[0], atol=0.3)


def test_lazy_export():
    assert rb.ordering_sensitivity is so.ordering_sensitivity


# =========================================================================== item 2 (nightly)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reduced_iw_equivalent_is_order_invariant_within_mcse():
    """N08 §B5 item 2: ``Linear(form="reduced") × Constant("iw_equivalent")`` (Chan 2022) is
    order-invariant, so posterior means of ``B`` and ``Σ`` and forecast quantiles of the user
    ordering and two permutations agree within MCSE (z-tests, Holm-corrected, 1%)."""
    from rai_bvar.diagnostics import mcse_mean_many, mcse_quantile_many

    data = var_data(3, 160, seed=21, scales=[1.0, 3.0, 0.5], names=["x", "y", "z"])
    model = BVAR(
        mean=Linear(prior=Minnesota(form="reduced", hierarchy="fixed")),
        vol=Constant("iw_equivalent"), lags=2,
        mcmc=MCMCSettings(chains=4, draws=2000, burn=500),
    )  # fmt: skip
    summaries = []
    for order in (("x", "y", "z"), ("z", "y", "x"), ("y", "z", "x")):
        fit = permute_model(model, order, variables=data.variables).fit(data.select(order))
        B = fit.reduced_form_draws()
        B = B.sel(variable=list(data.variables),
                  regressor=regressor_names(list(data.variables), 2, "c")).values  # fmt: skip
        C, D = B.shape[:2]
        _, chol = fit.draw_parameters(C * D)
        idx = [order.index(v) for v in data.variables]
        sigma = (chol @ np.swapaxes(chol, -1, -2))[:, idx][:, :, idx].reshape(C, D, 3, 3)
        fc = fit.forecast(4, draws=C * D, seed=5)
        paths = fc.paths.sel(variable=list(data.variables)).values  # (C, D', H, n)
        est, se = [B.mean((0, 1)).ravel(), sigma.mean((0, 1)).ravel()], [
            mcse_mean_many(B).ravel(), mcse_mean_many(sigma).ravel()]  # fmt: skip
        for q in (0.05, 0.5, 0.95):
            est.append(np.quantile(paths.reshape(-1, *paths.shape[2:]), q, axis=0).ravel())
            se.append(mcse_quantile_many(paths, q).ravel())
        summaries.append((np.concatenate(est), np.concatenate(se)))
    pvals = []
    for (e0, s0), (e1, s1) in itertools.combinations(summaries, 2):
        z = (e1 - e0) / np.hypot(s0, s1)
        pvals.append(2 * stats.norm.sf(np.abs(z[np.isfinite(z)])))
    p = np.sort(np.concatenate(pvals))
    holm = np.maximum.accumulate(p * (p.size - np.arange(p.size)))
    assert holm.min() > 0.01, holm[:5]


# =========================================================================== item 5 (weekly)


def cky_dgp(T: int = 500, seed: int = 0, sv: bool = True, common: bool = False) -> VARData:
    """CKY (2024) §6.1: ``B₀ y_t = c + B₁ y_{t−1} + ε_t``, non-triangular ``B₀ =
    [[1, −.8, −.8], [.8, 1, −.8], [.8, .8, 1]]``, ``ε_i,t ~ N(0, e^{h_i,t})``, ``h_i,t =
    0.95 h_i,t−1 + √0.05 η``; ``sv=False`` keeps ``h ≡ 0`` (homoskedastic control); ``common=True``
    draws one ``h_t`` shared by every shock (the CommonSV control, P3 §0.2 row 26)."""
    rng = np.random.default_rng(seed)
    B0 = np.array([[1.0, -0.8, -0.8], [0.8, 1.0, -0.8], [0.8, 0.8, 1.0]])
    A1 = 0.5 * np.eye(3)
    h = np.zeros(3)
    y = np.zeros((T + 100, 3))
    for t in range(1, T + 100):
        if sv and common:
            h = np.full(3, 0.95 * h[0] + math.sqrt(0.05) * rng.standard_normal())
        elif sv:
            h = 0.95 * h + math.sqrt(0.05) * rng.standard_normal(3)
        eps = np.exp(0.5 * h) * rng.standard_normal(3)
        y[t] = A1 @ y[t - 1] + np.linalg.solve(B0, eps)
    frame = pd.DataFrame(
        y[100:], index=pd.period_range("1960-01", periods=T, freq="M"), columns=["a", "b", "c"]
    )
    return VARData.from_frame(frame)


@pytest.mark.slow
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detection_power_and_false_alarm_control():
    """N08 §B5 item 5: ``Linear × SV`` on CKY's SV DGP is flagged; the order-invariant
    ``CommonSV`` composition, ``Minnesota(form="conjugate") × CommonSV()`` (the NIW prior with a
    common scalar volatility, exactly order-invariant, P3 §5.3), on CKY's DGP with one common
    ``h_t`` is not. This control replaces the Phase 2 stand-in (``reduced ×
    Constant("iw_equivalent")`` on the homoskedastic DGP; P3 §0.2 row 26, E140, E159).

    Backtest mode over 10 origins: with ``n = 3`` and constant ``A`` the predictive sd moves by
    only ~5% across orderings at one origin (below the 1.10 threshold, as N08 §B3 expects for
    constant ``A``), but the joint log score moves by ~0.14 nats per origin, far above the noise
    floor (calibration run: signal-to-noise ≈ 10)."""
    settings_ = MCMCSettings(chains=4, draws=1000, burn=500)
    kw = {"n_perm": 6, "noise_reps": 3, "draws": 1000, "horizons": (1,), "mode": "backtest"}
    data = cky_dgp(sv=True)
    origins = [str(p) for p in data.index[-60:-1:6]]
    res = ordering_sensitivity(BVAR(vol=SV(), lags=1, mcmc=settings_), data, origins=origins, **kw)
    assert res.material, res.summary()
    assert bool(res.metrics()["lpd_range_joint_flag"].values.any())
    inv = BVAR(mean=Linear(prior=Minnesota(form="conjugate")), vol=rb.CommonSV(), lags=1,
               mcmc=settings_)  # fmt: skip
    ctrl_data = cky_dgp(sv=True, common=True, seed=1)
    ctrl_origins = [str(p) for p in ctrl_data.index[-60:-1:6]]
    ctrl = ordering_sensitivity(inv, ctrl_data, origins=ctrl_origins, **kw)
    assert not ctrl.material, ctrl.summary()
