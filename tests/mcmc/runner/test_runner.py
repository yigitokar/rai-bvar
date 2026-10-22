"""Host driver and fits (DESIGN §5.1–§5.4, §11.3): toy posteriors, fit_mcmc, the posterior
DataTree, the production composition path, the fit axis and the WP10–WP13 smoke test."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pandas as pd
import pytest

from rai_bvar import diagnostics
from rai_bvar.engines.mcmc import monitor
from rai_bvar.engines.mcmc.config import StoreStatic
from rai_bvar.engines.mcmc.results import FitMeta, MCMCFit, fit_mcmc
from rai_bvar.engines.mcmc.runner import make_runner, run_chains
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _meta(static, T_eff, **kw):
    n = static.shapes.n
    names = tuple(f"v{i}" for i in range(n))
    return FitMeta(
        variables=names, periods=toys.periods(T_eff), hyper={"psi": np.ones(n), "lambda": 0.2}, **kw
    )


def _fit(static, settings, T_eff=20, seed=0, salt=1, overrides=None, **kw):
    ov = toys.toy_overrides(static) if overrides is None else overrides
    return fit_mcmc(
        static, toys.toy_data(static, T_eff, 4), toys.toy_prior(static), settings,
        Runtime(chunk=static.chunk), seed=seed, salt=salt, meta=_meta(static, T_eff), _overrides=ov, **kw,
    )  # fmt: skip


def _mcse(x):
    """MCSE of the mean of a (chains, draws) array."""
    return np.std(x) / np.sqrt(diagnostics.ess_bulk(x))


@pytest.mark.parametrize("correlated", [False, True])
def test_toy_posterior_recovered_within_mcse(correlated):
    """Mode II × Constant toy = Normal–inverse-gamma Gibbs sampler: E[λ_i | y] and E[θ_i | y]
    within 4 MCSE of the closed form, posterior s.d. within 10%. ``correlated``: data with a
    strong contemporaneous dependence (``a_10 ≈ −0.8``), so the volatility step sees the wrong
    variance unless the residual is formed from the *new* ``θ_i`` (DESIGN §4.2, S2): a residual
    without ``a_i``, or with the previous sweep's, moves ``E[λ_1]`` by many MCSE."""
    static = toys.toy_static("linear_structural", "constant", mode="II", n=2, T_eff=60, chunk=500)
    y = toys.simulate_correlated_y(60 + static.shapes.lags, 7, scale=10.0) if correlated else None
    data = toys.toy_data(static, 60, 7, y=y)
    prior = toys.toy_prior(static)
    fns = toys.cached_runner(static, toys.toy_overrides(static))
    out = run_chains(
        fns, static, data, prior, MCMCSettings(chains=4, draws=1500, burn=100), Runtime(chunk=500),
        seed=2, salt=3,
    )  # fmt: skip
    p = out.store.stream("params")
    lam = np.exp(np.asarray(p.vol.core.log_lam))  # (C, D, n)
    a = np.asarray(p.struct.a)
    delta = np.asarray(p.mean.gamma) - np.einsum("kj,cdij->cdik", prior.shared.Mc, a)
    exact = toys.nig_posterior(static, data, prior)
    k = static.shapes.k
    for i, ex in enumerate(exact):
        x = lam[..., i]
        assert abs(x.mean() - ex["E_lam"]) < 4 * _mcse(x), (i, x.mean(), ex["E_lam"])
        theta = np.concatenate([delta[..., i, :], a[..., i, :i]], axis=-1)
        assert theta.shape[-1] == ex["q"] == k + i
        for j in range(ex["q"]):
            t = theta[..., j]
            assert abs(t.mean() - ex["E_theta"][j]) < 4 * _mcse(t), (i, j)
            np.testing.assert_allclose(t.std(), ex["sd_theta"][j], rtol=0.1)
        np.testing.assert_array_equal(a[..., i, i:], 0.0)  # inactive slots exactly zero
    if correlated:
        assert exact[1]["E_theta"][-1] < -0.6  # the contemporaneous coefficient a_10


def test_fit_mcmc_with_toys():
    static = toys.toy_static(
        "linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5, paths=True
    )
    settings = MCMCSettings(
        chains=2, draws=9, burn=4, thin=2, store_max=4, path_thin=2, store="paths"
    )
    fit = _fit(static, settings)
    assert isinstance(fit, MCMCFit) and fit.engine == "mcmc"
    assert fit.plan.names == ("equation", "residual", "vol")
    assert fit.ordering == ("v0", "v1") and fit.psi is not None
    assert isinstance(fit.converged, bool) and isinstance(fit.diagnostics, pd.DataFrame)
    assert set(fit.timing) >= {"compile", "burn", "sample", "host", "ms_per_sweep"}
    assert fit.final_state.static == static and len(fit.final_state.periods) == 20
    tree = fit.posterior
    assert set(tree.children) == {
        "posterior", "monitor", "paths", "sample_stats", "latent_moments", "constant_data"
    }  # fmt: skip
    post = tree["posterior"].to_dataset()
    # store_thin = ⌈9/4⌉ = 3: kept draws 0, 3, 6
    np.testing.assert_array_equal(post["draw"].values, [0, 3, 6])
    assert post["gamma"].dims == ("chain", "draw", "variable", "regressor")
    assert list(post["regressor"].values) == ["const", "v0.L1", "v1.L1"]
    assert post["a"].dims == ("chain", "draw", "pair") and list(post["pair"].values) == ["v1~v0"]
    for name in ("mu", "phi", "sigma", "h_T", "p_outlier"):
        assert post[name].dims == ("chain", "draw", "variable")
    paths = tree["paths"].to_dataset()
    np.testing.assert_array_equal(paths["draw"].values, [0, 6])  # every 3·2-th kept draw
    assert paths["h"].shape == (2, 2, 2, 21) and paths["log_scale2"].shape == (2, 2, 2, 20)
    assert paths["time"].values[0] == "1990-01" and paths["h_index"].values[0] == "h_0"
    lat = tree["latent_moments"].to_dataset()
    assert lat["h_mean"].dims == ("chain", "variable", "h_index") and lat["p_outlier"].shape == (
        2,
        2,
        20,
    )
    stats = tree["sample_stats"].to_dataset()
    assert stats["acc_h"].dims == ("chain", "variable") and stats["min_pivot"].dims == ("chain",)
    const = tree["constant_data"].to_dataset()
    assert (
        "prior_eq_coef_mean0" in const
        and "hyper_psi" in const
        and const.attrs["hyper_lambda"] == 0.2
    )
    attrs = tree.attrs
    for key in ("jax_version", "prng_impl", "seed", "salt", "T_pad", "gram", "const_path", "mixture",
                "exact", "ordering", "chain_method", "it_last", "package_version"):  # fmt: skip
        assert key in attrs, key
    assert attrs["it_last"] == 4 + 9 * 2 - 1 and attrs["prng_impl"] == "threefry2x32"
    # B = (A⁻¹Γ)' per stored draw
    B = fit.reduced_form_draws()
    assert B.dims == ("chain", "draw", "regressor", "variable") and B.shape == (2, 3, 3, 2)
    g = np.asarray(fit.param_draws.mean.gamma)[1, 2]
    a = np.asarray(fit.param_draws.struct.a)[1, 2]
    A = np.eye(2)
    A[1, 0] = a[1, 0]
    np.testing.assert_allclose(B.values[1, 2], np.linalg.solve(A, g).T, rtol=1e-12)
    assert "converged" in repr(fit) and fit.summary().attrs["prior_mode"] == "Mode I"


def test_fit_mcmc_reduced_form_with_toys():
    static = toys.toy_static("linear_reduced", "constant", n=3, T_eff=20, chunk=5)
    fit = _fit(static, MCMCSettings(chains=2, draws=5, burn=2))
    post = fit.posterior["posterior"].to_dataset()
    assert post["Pi"].dims == ("chain", "draw", "regressor", "variable")
    np.testing.assert_array_equal(fit.reduced_form_draws().values, post["Pi"].values)
    assert fit.plan.names == ("cccm", "contemp", "residual", "vol")


def test_delegation_to_the_forecast_module():
    static = toys.toy_static("linear_structural", "constant", n=2, T_eff=20, chunk=5)
    fit = _fit(static, MCMCSettings(chains=2, draws=3, burn=1))
    from rai_bvar.engines.mcmc import forecast

    if getattr(forecast, "forecast", None) is None:
        with pytest.raises(NotImplementedError, match="WP17"):
            fit.forecast(4)
    calls = {}

    def fake(fit_, horizon, draws, seed):
        calls["forecast"] = (fit_, horizon, draws, seed)
        return "fc"

    def fake_irf(fit_, **kw):
        calls["irf"] = kw
        return "irf"

    def fake_draws(fit_, size, rng, *, date, which):
        calls["draws"] = (size, rng, date, which)
        return "draws"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(forecast, "forecast", fake, raising=False)
        mp.setattr(forecast, "irf", fake_irf, raising=False)
        mp.setattr(forecast, "draw_parameters", fake_draws, raising=False)
        assert fit.forecast(4, 100, 1) == "fc" and calls["forecast"] == (fit, 4, 100, 1)
        assert fit.irf(8, shock="unit") == "irf" and calls["irf"]["shock"] == "unit"
        assert fit.draw_parameters(5, 2, which="effective") == "draws"
        assert calls["draws"] == (5, 2, None, "effective")


def test_production_path_with_registry_toys(toy_registry):
    """No overrides: compose builds the registry classes, make_runner and the storage plan use
    the lru-cached production path."""
    static = toys.toy_static("linear_structural", "constant", n=2, T_eff=20, chunk=5)
    fit = fit_mcmc(
        static, toys.toy_data(static, 20, 4), toys.toy_prior(static), MCMCSettings(chains=2, draws=4, burn=2),
        Runtime(chunk=5), seed=0, salt=1, meta=_meta(static, 20),
    )  # fmt: skip
    assert isinstance(fit.program.mean, toys.ToyLinear)
    assert make_runner(static).program is fit.program


def test_runtime_and_static_consistency_checks():
    static = toys.toy_static("linear_structural", "constant", n=2, T_eff=20, chunk=5)
    fns = toys.cached_runner(static, toys.toy_overrides(static))
    data, prior = toys.toy_data(static, 20, 4), toys.toy_prior(static)
    with pytest.raises(ValueError, match=r"Runtime\.chunk"):
        run_chains(fns, static, data, prior, MCMCSettings(chains=2, draws=2, burn=1), Runtime(chunk=7),
                   seed=0, salt=0)  # fmt: skip
    other = dataclasses.replace(static, chunk=7)
    with pytest.raises(ValueError, match="does not match"):
        run_chains(fns, other, data, prior, MCMCSettings(chains=2, draws=2, burn=1), Runtime(chunk=7),
                   seed=0, salt=0)  # fmt: skip
    with pytest.raises(ValueError, match=r"store='paths'"):
        run_chains(fns, static, data, prior, MCMCSettings(chains=2, draws=2, store="paths"),
                   Runtime(chunk=5), seed=0, salt=0)  # fmt: skip
    looped = dataclasses.replace(static, store=StoreStatic(in_loop_forecast=4))
    from rai_bvar.engines.mcmc import forecast

    with pytest.MonkeyPatch.context() as mp:  # the in-loop hook needs WP17's forecast core
        mp.delattr(forecast, "forecast_core", raising=False)
        with pytest.raises(NotImplementedError, match="forecast_core"):
            make_runner(looped, overrides=toys.toy_overrides(looped))


def test_seed_and_salt_select_the_streams():
    static = toys.toy_static("linear_structural", "constant", n=2, T_eff=20, chunk=5)
    s = MCMCSettings(chains=2, draws=4, burn=2)
    a, b, c = (
        _fit(static, s, seed=sd, salt=sl).monitor["delta0"] for sd, sl in ((0, 1), (0, 1), (0, 2))
    )
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fit_axis_equals_looped_fits():
    """A leading fit axis (one more vmap) equals separate fits to 1e-10 (DESIGN §5.2)."""
    static = toys.toy_static(
        "linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5
    )
    ov = toys.toy_overrides(static)
    prior = toys.toy_prior(static)
    datas = [toys.toy_data(static, T, sd) for T, sd in ((20, 1), (17, 2))]
    settings = MCMCSettings(chains=2, draws=6, burn=3)
    fns_f = toys.cached_runner(static, ov, fit_axis=True)
    stack = lambda *xs: np.stack(xs)  # noqa: E731
    out_f = run_chains(
        fns_f, static, jax.tree.map(stack, *datas), jax.tree.map(stack, prior, prior), settings,
        Runtime(chunk=5), seed=[3, 4], salt=[5, 6],
    )  # fmt: skip
    fns = toys.cached_runner(static, ov)
    for f, (data, seed, salt) in enumerate(zip(datas, (3, 4), (5, 6), strict=True)):
        out = run_chains(fns, static, data, prior, settings, Runtime(chunk=5), seed=seed, salt=salt)
        for x, y in zip(jax.tree.leaves(out_f.store.stream("params")), jax.tree.leaves(out.store.stream("params")),
                        strict=True):  # fmt: skip
            np.testing.assert_allclose(np.asarray(x)[f], y, rtol=1e-10, atol=1e-10)
        for x, y in zip(
            jax.tree.leaves(out_f.final_state), jax.tree.leaves(out.final_state), strict=True
        ):
            np.testing.assert_allclose(np.asarray(x)[f], y, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("transitory", ["none", "outlier"])
def test_real_monitor_with_toy_kernels(monkeypatch, transitory):
    """The production monitors (WP15) record and judge a toy fit; with outliers the per-equation
    table carries the Rao-Blackwellised outlier probabilities over the real rows (DESIGN §8.4)."""
    real = {name: toys.REAL_MONITOR.get(name) for name in toys.MONITOR_DOUBLES}
    if any(fn is None for fn in real.values()):
        pytest.skip("engines.mcmc.monitor (WP15) not available")
    for name, fn in real.items():
        monkeypatch.setattr(monitor, name, fn)
    static = toys.toy_static(
        "linear_structural", "sv", transitory=transitory, n=2, T_eff=20, chunk=5
    )
    ov = toys.toy_overrides(static)  # fresh runner: traced with the real monitor_values
    fit = fit_mcmc(
        static, toys.toy_data(static, 20, 4), toys.toy_prior(static), MCMCSettings(chains=2, draws=6, burn=2),
        Runtime(chunk=5), seed=0, salt=1, meta=_meta(static, 20), _overrides=ov,
    )  # fmt: skip
    assert "loglik" in fit.monitor and fit.monitor["B_own1"].shape == (2, 6, 2)
    assert fit.converged is False and fit.warnings  # 6 draws never converge
    assert "loglik" in fit.diagnostics.index and fit.posterior["sample_stats"]["loglik"].shape == (
        2,
        6,
    )
    equations = fit.diagnostics.attrs["equations"]
    assert "h" in equations and "min_pivot" in fit.diagnostics.attrs
    if transitory == "outlier":  # the toy adds 0.1 per real row and sweep, 0 on padded rows
        np.testing.assert_allclose(list(equations["p_outlier"].values()), 0.1, rtol=1e-12)
    else:
        assert "p_outlier" not in equations


def test_summary_reports_implied_reduced_form_moments():
    """``fit.summary()`` carries the implied reduced-form prior moments (R8; DESIGN §8.4) of the
    fit's prior arrays at the composition's mode, with E[λ_i] the prior mean of λ_i."""
    from rai_bvar.priors.structural import implied_reduced_form_moments

    for mode, persistent in (("I", "constant"), ("II", "constant"), ("I", "sv")):
        static = toys.toy_static("linear_structural", persistent, mode=mode, n=3, T_eff=20, chunk=5)
        fit = _fit(static, MCMCSettings(chains=2, draws=3, burn=1))
        prior = toys.toy_prior(static)
        vol = prior.eq.vol_core
        lam = vol.scale / (vol.shape - 1.0) if persistent == "constant" else np.exp(vol.b_mu)
        np.testing.assert_allclose(fit.prior_vol_mean(), lam, rtol=1e-15)
        exp = implied_reduced_form_moments(
            prior.eq.coef, prior.eq.contemp, prior.shared, lam, mode=mode
        )
        im = fit.summary().attrs["implied_reduced_form"]
        assert im["method"] == "closed_form"
        regs = ["const", "v0.L1", "v1.L1", "v2.L1"]
        for c, v in enumerate(fit.ordering):
            for r, reg in enumerate(regs):
                assert im["mean"][v][reg] == exp.mean[r, c] and im["var"][v][reg] == exp.var[r, c]
            for r, w in enumerate(fit.ordering):
                assert im["sigma_mean"][v][w] == exp.sigma_mean[r, c]
                assert im["shares"][v][w] == exp.shares[r, c]
        assert fit.summary().attrs["prior_form"] == "structural"
    reduced = toys.toy_static("linear_reduced", "constant", n=3, T_eff=20, chunk=5)
    fit = _fit(reduced, MCMCSettings(chains=2, draws=3, burn=1))
    assert fit.implied_prior_moments() is None
    assert fit.summary().attrs["implied_reduced_form"].startswith("not applicable")


# --------------------------------------------------------------------------- production kernels


def _production_or_skip(static):
    from rai_bvar.engines.mcmc.compose import compose

    try:
        compose(static)
    except NotImplementedError as err:
        pytest.skip(f"production kernels not available yet: {err}")
    if any(toys.REAL_MONITOR.get(name) is None for name in toys.MONITOR_DOUBLES):
        pytest.skip("engines.mcmc.monitor (WP15) not available")


def _smoke(name, monkeypatch):
    from rai_bvar.validation.joint import joint_spec

    spec = joint_spec(name)
    static = dataclasses.replace(spec.static, chunk=25)
    _production_or_skip(static)
    for fn_name, fn in toys.REAL_MONITOR.items():
        monkeypatch.setattr(monitor, fn_name, fn)
    key = jax.random.key(7)
    st = spec.prior_sample(key)
    data = jax.tree.map(np.asarray, spec.simulate(jax.random.fold_in(key, 1), st))
    fit = fit_mcmc(
        static, data, spec.prior, MCMCSettings(chains=2, draws=50, burn=25), Runtime(chunk=25),
        seed=1, salt=2, meta=FitMeta(variables=tuple(f"y{i}" for i in range(static.shapes.n))),
    )  # fmt: skip
    assert np.isfinite(fit.monitor["loglik"]).all()
    equations = fit.diagnostics.attrs["equations"]
    if static.vol.has_outliers:  # Rao-Blackwellised P(o_t > 1 | y) in the §8.4 table
        p_out = np.array(list(equations["p_outlier"].values()))
        assert ((p_out >= 0) & (p_out <= 1)).all()
    if static.mean.kind == "linear_structural":
        assert isinstance(fit.summary().attrs["implied_reduced_form"], dict)
    return fit


def test_production_smoke_structural_sv(monkeypatch):
    """One production composition per PR: 2 chains × 50 draws run without error."""
    _smoke("structural_sv", monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("name", [
    "structural_constant", "structural_sv_outliers", "structural_sv_t", "structural_mode2_constant",
    "structural_iw_equivalent", "reduced_constant", "reduced_sv", "reduced_sv_outliers", "reduced_sv_t",
    "reduced_iw_equivalent", "structural_sv_outliers_t", "reduced_sv_outliers_t", "structural_sv_rw",
    "structural_sv_invgamma", "structural_sv_fixed_b0", "zero_sv",
])  # fmt: skip
def test_production_smoke_every_composition(name, monkeypatch):
    """DESIGN §13 WP14 acceptance: every composition of §9.5 runs (2 chains × 50 draws)."""
    _smoke(name, monkeypatch)
