"""``BVAR.fit`` on the MCMC engine: resolution and fit assembly (DESIGN §6, §10.5, §11.3; WP18).

``engines.mcmc.resolve`` turns a ``BVAR`` and a ``VARData`` window into ``(StaticConfig,
EngineData, PriorArrays, hyper, notes)`` and ``fit_bvar`` runs ``fit_mcmc`` on them. Most tests
here check the resolution (no compilation); a few run real fits with tiny settings, sharing one
compiled program per composition (``StaticConfig`` is the cache key, seeds and convergence
thresholds are not part of it).
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import rai_bvar as rb
from rai_bvar.api import static_parts
from rai_bvar.engines.mcmc import eb, keys, monitor, resolve
from rai_bvar.engines.mcmc.config import StoreStatic, bucket
from rai_bvar.engines.mcmc.data import engine_data_from_vardata
from rai_bvar.engines.mcmc.results import MCMCFit
from rai_bvar.engines.mcmc.state import Empty, prior_template
from rai_bvar.engines.mcmc.warmstart import FinalState, warm_state
from rai_bvar.priors.reduced import reduced_arrays
from rai_bvar.priors.scales import window_scales
from rai_bvar.priors.structural import contemp_arrays, structural_arrays
from rai_bvar.priors.sv_arrays import (
    C_REL,
    constant_prior,
    sv_prior_arrays,
    transitory_arrays,
)

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

SETTINGS = rb.MCMCSettings(chains=2, draws=30, burn=20)
RUNTIME = rb.Runtime(chunk=15)
LAGS = 2


def bvar(vol=None, form="auto", mcmc=SETTINGS, **prior):
    return rb.BVAR(
        mean=rb.Linear(prior=rb.Minnesota(form=form, **prior)),
        vol=rb.SV() if vol is None else vol,
        lags=LAGS,
        mcmc=mcmc,
    )


@pytest.fixture(scope="module")
def data(sim_data_stationary):
    return sim_data_stationary.tail(80)


@pytest.fixture(scope="module")
def covid_data(simulate_var):
    """Quarterly data 2000Q1–2021Q4 (covers 2020), tcode 1."""
    index = pd.period_range("2000Q1", periods=88, freq="Q")
    frame = pd.DataFrame(simulate_var(88, 11), index=index, columns=["gdp", "prices", "rate"])
    return rb.VARData.from_frame(frame, 1)


def flat(tree):
    """``{leaf path: NumPy array}`` of a pytree (``Empty`` parts have no leaves)."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        out[jax.tree_util.keystr(path, simple=True, separator=".")] = np.asarray(leaf)
    return out


def assert_tree_equal(a, b):
    fa, fb = flat(a), flat(b)
    assert fa.keys() == fb.keys()
    for name in fa:
        assert fa[name].dtype == fb[name].dtype, name
        np.testing.assert_array_equal(fa[name], fb[name], err_msg=name)


def assert_matches_template(resolved):
    """Structure, shapes and dtypes are those the ``StaticConfig`` implies (DESIGN §2.6)."""
    tmpl = flat(prior_template(resolved.static))
    got = flat(resolved.prior)
    assert got.keys() == tmpl.keys()
    for name, value in got.items():
        assert value.shape == tmpl[name].shape and value.dtype == tmpl[name].dtype, name


# =========================================================================== end to end


@pytest.fixture(scope="module")
def sv_model():
    return bvar()


@pytest.fixture(scope="module")
def sv_fit(sv_model, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # not converged at 2 × 30 draws
        return sv_model.fit(data, runtime=RUNTIME)


def test_sv_fit_end_to_end(sv_model, sv_fit, data):
    """``BVAR(vol=SV()).fit(data)``: structural Mode I × SV, resolved and sampled (DESIGN §11.1
    row 6); the fit carries the model, the window, the hyperparameters and the metadata."""
    fit = sv_fit
    assert isinstance(fit, MCMCFit) and fit.engine == "mcmc"
    assert fit.model is sv_model and fit.data is data
    s = fit.static
    T_eff = len(data) - LAGS
    assert (s.shapes.n, s.shapes.lags, s.shapes.n_det) == (3, LAGS, 1)
    assert s.shapes.T_pad == bucket(T_eff, 32) == 96
    assert s.struct.contemp_scaling == "none"
    assert (s.mean.kind, s.mean.coef_scaling, s.mean.const_path) == (
        "linear_structural",
        "none",
        "weighted",
    )
    assert (s.mean.gram, s.mean.gram_tile) == ("packed", 0)
    assert s.vol.persistent == "sv" and s.vol.transitory == "none" and not s.vol.coupled
    assert s.store == StoreStatic() and s.chunk == RUNTIME.chunk
    assert fit.ordering == data.variables
    pd.testing.assert_index_equal(pd.Index(fit.periods), pd.Index(data.index[LAGS:]))
    assert (fit.seed, fit.salt) == (0, keys.data_salt(data.fingerprint()))
    # the hyperparameters in use: the EB plug-in (DESIGN §6.5) and the SV level centres
    outcome = eb.eb_hyper(sv_model, data)
    assert fit.hyper["lambda"] == outcome.hyper["lambda"] and fit.hyper["eb_method"] == (
        "conjugate_twin"
    )
    np.testing.assert_array_equal(fit.psi, outcome.hyper["psi"])
    iota = np.arange(1.0, 4.0)
    np.testing.assert_allclose(fit.hyper["sv_level"], np.log(fit.psi / iota), rtol=1e-15)
    np.testing.assert_allclose(fit.hyper["offsets"], C_REL * fit.psi, rtol=1e-12)
    assert fit.hyper["offset_rule"] == "relative" and fit.hyper["c_rel"] == C_REL
    # prior arrays equal a fresh resolution, bit for bit (they are pure functions of the window)
    again = resolve.resolve(sv_model, data, RUNTIME)
    assert again.static == s
    assert_tree_equal(fit.prior, again.prior)
    assert_tree_equal(fit.engine_data, again.data)
    # draws, monitors, metadata
    post = fit.posterior["posterior"].to_dataset()
    assert post.sizes["chain"] == 2 and post.sizes["draw"] == 30
    assert np.isfinite(fit.monitor["loglik"]).all()
    for key, value in (("T_pad", 96), ("gram", "packed"), ("const_path", "weighted")):
        assert fit.metadata[key] == value
    assert fit.metadata["prior_mode"] == "I"
    assert isinstance(fit.converged, bool)
    assert fit.summary().attrs["prior_mode"] == "Mode I"


def test_seed_override(sv_model, sv_fit, data):
    """``fit(seed=s)`` overrides ``MCMCSettings.seed`` (DESIGN §11.3): it equals a model whose
    settings carry ``seed=s`` bit for bit, and differs from the default seed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = sv_model.fit(data, seed=5, runtime=RUNTIME)
        b = bvar(mcmc=dataclasses.replace(SETTINGS, seed=5)).fit(data, runtime=RUNTIME)
    assert a.seed == b.seed == 5
    assert_tree_equal(a.param_draws, b.param_draws)
    gamma_a = np.asarray(a.param_draws.mean.gamma)
    assert not np.array_equal(gamma_a, np.asarray(sv_fit.param_draws.mean.gamma))


def test_converged_includes_the_eb_flag(sv_model, data, monkeypatch):
    """With the sampler gates opened (``ess_min = 0``), ``converged`` is exactly the EB flag
    (DESIGN §8.2), and the EB message reaches ``fit.warnings``."""
    open_gates = dataclasses.replace(SETTINGS, ess_min=0.0, rhat_max=1e9)
    model = bvar(mcmc=open_gates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert model.fit(data, runtime=RUNTIME).converged
    real = eb.eb_hyper
    message = "hyperparameter optimisation did not converge: (test double)"

    def failing(model, data):
        out = real(model, data)
        return dataclasses.replace(out, converged=False, warnings=(*out.warnings, message))

    monkeypatch.setattr(eb, "eb_hyper", failing)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        fit = model.fit(data, runtime=RUNTIME)
    assert message in [str(w.message) for w in caught]
    assert not fit.converged
    assert message in fit.warnings
    assert fit.hyper["eb_converged"] is False and fit.metadata["converged"] == 0


def test_warm_start_from_a_fit_or_its_final_state(sv_model, sv_fit, data, sim_data_stationary):
    """``init`` takes an ``MCMCFit`` or its ``FinalState`` (DESIGN §5.5): the chains start from
    the remapped state and burn ``warm_burn`` sweeps; an incompatible state cold-starts with a
    warning."""
    settings = dataclasses.replace(SETTINGS, warm_burn=3)
    model = bvar(mcmc=settings)
    longer = sim_data_stationary.tail(82)  # two more periods, same T_pad bucket
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warm = model.fit(longer, init=sv_fit, runtime=RUNTIME)
        again = model.fit(longer, init=sv_fit.final_state, runtime=RUNTIME)
    assert isinstance(sv_fit.final_state, FinalState)
    assert warm.final_state.seed_meta["warm_start"] and warm.final_state.seed_meta["burn"] == 3
    assert_tree_equal(warm.param_draws, again.param_draws)
    assert np.isfinite(warm.monitor["loglik"]).all()
    # an incompatible state (other lags) or too few chains: cold start with a warning
    other = resolve.resolve(rb.BVAR(vol=rb.SV(), lags=1, mcmc=settings), longer, RUNTIME)
    same = resolve.resolve(model, longer, RUNTIME)
    with pytest.warns(UserWarning, match=r"warm start ignored \(lags differs"):
        assert resolve._warm_state(sv_fit, other, 0, 2) is None
    with pytest.warns(UserWarning, match="it has 2 chains, the settings ask for 4"):
        assert resolve._warm_state(sv_fit, same, 0, 4) is None
    assert resolve._warm_state(None, same, 0, 2) is None
    # the remap key follows the fit's seed: the two new periods' h are forward-simulated under it
    s0, s5 = (resolve._warm_state(sv_fit, same, seed, 2) for seed in (0, 5))
    key5 = explicit_warm_key(5, same.salt)
    assert_tree_equal(
        s5, warm_state(sv_fit.final_state, same.data, same.periods, same.static, key5)
    )
    assert not np.array_equal(flat(s0)["eq.vol.core.h"], flat(s5)["eq.vol.core.h"])
    with pytest.raises(TypeError, match="init must be"):
        model.fit(longer, init=object(), runtime=RUNTIME)


def explicit_warm_key(seed, salt):
    """``key(seed) → Stream.INIT → salt → 2**32 − 1 → Block.EQ_VOL`` spelled out (DESIGN §5.5)."""
    k = jax.random.key(seed, impl="threefry2x32")
    for value in (1, salt, 2**32 - 1, 2):
        k = jax.random.fold_in(k, np.uint32(value))
    return k


def test_warm_key_never_coincides_with_a_cold_start_key():
    """The warm-start key folds ``2**32 − 1`` (never a chain index) after the salt, then
    ``Block.EQ_VOL``: neither it nor the keys the remap derives from it (``→ c → i``) equals a
    cold-start init key (``salt → c → Block → i → Sub``) of any of 8 chains. The previous
    derivation (``salt → Block.EQ_VOL``) was the cold init key of chain 2."""
    data_of = jax.random.key_data
    seed, salt, C, n = 3, keys.data_salt("window"), 8, 3
    warm = resolve.warm_key(seed, salt)
    assert np.array_equal(data_of(warm), data_of(explicit_warm_key(seed, salt)))
    init = keys.init_keys(seed, salt, C)  # (C,)
    root = jax.random.key(seed, impl="threefry2x32")
    base = jax.random.fold_in(jax.random.fold_in(root, np.uint32(1)), np.uint32(salt))
    old = keys.block_key(base, keys.Block.EQ_VOL)
    assert np.array_equal(data_of(old), data_of(init[2]))  # the collision this derivation fixes

    fold = jax.vmap(jax.random.fold_in, in_axes=(0, None))  # every key × one value

    def tails(ks, values):
        return jnp.concatenate([fold(ks, np.uint32(v)) for v in values])

    blocks = [int(b) for b in keys.Block]
    cold_block = tails(init, blocks)  # c → Block
    cold_eq = tails(cold_block, range(n))  # c → Block → i
    cold_sub = tails(cold_eq, sorted({int(s) for s in keys.Sub}))  # c → Block → i → Sub
    cold = [init, cold_block, cold_eq, cold_sub]
    remap_c = tails(warm[None], range(C))  # the remap's keys: warm → c → i
    remap_i = tails(remap_c, range(n))
    warm_keys = [warm[None], remap_c, remap_i]
    as_rows = lambda ks: {tuple(r) for r in np.asarray(data_of(ks)).tolist()}  # noqa: E731
    cold_rows = set().union(*(as_rows(k) for k in cold))
    warm_rows = set().union(*(as_rows(k) for k in warm_keys))
    assert len(cold_rows) == sum(k.shape[0] for k in cold)  # the cold keys themselves are distinct
    assert len(warm_rows) == 1 + C + C * n
    assert not warm_rows & cold_rows


def test_warm_start_from_a_fit_with_more_chains(sv_fit, data, sim_data_stationary):
    """A final state with more chains than the settings ask for warm-starts the fit without a
    warning (``fit_mcmc`` keeps the first ``C``): with 4 stacked chains and ``chains=2`` the remap
    is not ``None`` and its first two chains equal the remap of the 2-chain state (the remap folds
    the chain index, so chain prefixes agree); the fit equals the warm fit from the 2-chain state."""
    model = bvar(mcmc=dataclasses.replace(SETTINGS, warm_burn=3))
    longer = sim_data_stationary.tail(82)
    prev = sv_fit.final_state
    four = dataclasses.replace(
        prev, state=jax.tree.map(lambda x: np.concatenate([x, x]), prev.state)
    )
    assert four.chains == 4
    same = resolve.resolve(model, longer, RUNTIME)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "warm start ignored" warning
        s4 = resolve._warm_state(four, same, 0, 2)
    assert s4 is not None
    s2 = resolve._warm_state(prev, same, 0, 2)
    assert_tree_equal(jax.tree.map(lambda x: np.asarray(x)[:2], s4), s2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # not converged at 2 × 30 draws
        a = model.fit(longer, init=four, runtime=RUNTIME)
        b = model.fit(longer, init=prev, runtime=RUNTIME)
    assert a.final_state.seed_meta["warm_start"] and a.final_state.chains == 2
    assert_tree_equal(a.param_draws, b.param_draws)


# =========================================================================== static configs


def test_mode_ii_static_config_and_arrays(data):
    """Structural × ``Constant("iw")`` (DESIGN §11.1 row 4): Mode II, both blocks λ-scaled,
    coupled Constant core, the fixed-factor path; the arrays are the WP6 Mode II arrays."""
    model = bvar(rb.Constant("iw"), form="structural", hierarchy="fixed")
    r = resolve.resolve(model, data, RUNTIME)
    s = r.static
    assert s.struct.contemp_scaling == "lambda" and s.mean.coef_scaling == "lambda"
    assert s.vol.persistent == "constant" and s.vol.coupled
    assert s.mean.const_path == "fixed_factor"
    assert r.hyper["lambda"] == 0.2 and r.eb_converged is None and "eb_method" not in r.hyper
    scales = window_scales(data, model.mean.prior, LAGS, "c")
    coef, shared = structural_arrays(
        scales, model.mean.prior, {"lambda": 0.2, "alpha": 2.0}, n=3, lags=LAGS, n_det=1, mode="II"
    )
    assert_tree_equal(r.prior.eq.coef, coef)
    assert_tree_equal(r.prior.shared, shared)
    assert_tree_equal(
        r.prior.eq.contemp, contemp_arrays(scales, model.mean.prior, 3, scaling="lambda")
    )
    assert_tree_equal(r.prior.eq.vol_core, constant_prior(scales, 3))
    assert isinstance(r.prior.eq.transitory, Empty)
    assert_matches_template(r)


def test_iw_equivalent_static_config_and_arrays(data):
    """Structural × ``Constant("iw_equivalent")`` (row 5): Mode I δ prior, λ-scaled ``a`` only
    (Chan 2022), coupled Constant core, the Cholesky constant path."""
    model = bvar(rb.Constant("iw_equivalent"), hierarchy="fixed")
    r = resolve.resolve(model, data, RUNTIME)
    s = r.static
    assert s.struct.contemp_scaling == "lambda" and s.mean.coef_scaling == "none"
    assert s.vol.coupled and s.mean.const_path == "cholesky_const"
    scales = window_scales(data, model.mean.prior, LAGS, "c")
    coef, _ = structural_arrays(scales, model.mean.prior, {}, n=3, lags=LAGS, n_det=1, mode="I")
    assert_tree_equal(r.prior.eq.coef, coef)
    assert_tree_equal(
        r.prior.eq.contemp, contemp_arrays(scales, model.mean.prior, 3, scaling="lambda")
    )
    assert_matches_template(r)


def test_mode_i_constant_takes_the_eigen_path(data):
    """Structural × ``Constant("ig")`` (row 3): Mode I, nothing scaled, the eigen path."""
    r = resolve.resolve(bvar(rb.Constant("ig"), hierarchy="fixed"), data, RUNTIME)
    s = r.static
    assert s.struct.contemp_scaling == "none" and s.mean.coef_scaling == "none"
    assert not s.vol.coupled and s.mean.const_path == "eigen"
    assert_matches_template(r)


@pytest.mark.parametrize(
    "vol",
    [rb.SV(), rb.Constant("ig"), rb.Constant("iw_equivalent")],
    ids=["SV", "ig", "iw_equivalent"],
)
def test_reduced_form_resolution(data, vol):
    """Reduced form (rows 7–8): the CCCM mean with the moment-matched independent Minnesota
    arrays, no D3 centring (``M_c = 0``), the weighted path, the packed-X Gram."""
    model = bvar(vol, form="reduced", hierarchy="fixed", cross=0.5)
    r = resolve.resolve(model, data, RUNTIME)
    s = r.static
    assert s.mean.kind == "linear_reduced" and s.mean.coef_scaling == "none"
    assert s.mean.const_path == "weighted" and s.mean.gram == "packed" and s.mean.cccm_precompute
    scaling = "lambda" if isinstance(vol, rb.Constant) and vol.prior == "iw_equivalent" else "none"
    assert s.struct.contemp_scaling == scaling
    scales = window_scales(data, model.mean.prior, LAGS, "c")
    coef, _ = reduced_arrays(scales, model.mean.prior, {}, n=3, lags=LAGS, n_det=1)
    assert_tree_equal(r.prior.eq.coef, coef)
    np.testing.assert_array_equal(r.prior.shared.Mc, np.zeros((7, 2)))
    # the a_i arrays at the composition's scaling: λ-free for reduced × iw_equivalent (row 8)
    assert_tree_equal(
        r.prior.eq.contemp, contemp_arrays(scales, model.mean.prior, 3, scaling=scaling)
    )
    assert_matches_template(r)


def test_sv_variants_resolve_their_volatility_arrays(data):
    """SV(outliers, t), random walk and fixed-B0 variants: the WP6 SV and transitory arrays, the
    outlier ``Beta`` from the quarterly frequency (D-9) and the offsets in ``hyper``."""
    sv = rb.SV(outliers=True, tails="t", offset=1e-6)
    r = resolve.resolve(bvar(sv, hierarchy="fixed"), data, RUNTIME)
    scales = window_scales(data, rb.Minnesota(), LAGS, "c")
    assert r.static.vol.transitory == "outlier_t"
    assert_tree_equal(r.prior.eq.vol_core, sv_prior_arrays(sv, scales))
    assert_tree_equal(r.prior.eq.transitory, transitory_arrays(sv, scales, r.static.vol.nu_grid))
    assert (r.hyper["outlier_beta_a"], r.hyper["outlier_beta_b"]) == (1.0, 39.0)
    assert r.hyper["offset_rule"] == "absolute" and r.hyper["offset"] == 1e-6
    rw = resolve.resolve(bvar(rb.SV(rw=True), hierarchy="fixed"), data, RUNTIME)
    assert rw.static.vol.rw
    np.testing.assert_array_equal(rw.hyper["sv_level"], np.log(scales.sigma2_bar))
    zero = resolve.resolve(bvar(rb.SV(offset=0.0), hierarchy="fixed"), data, RUNTIME)
    assert zero.static.vol.zero_offset and zero.hyper["offset_rule"] == "zero"
    for res in (r, rw, zero):
        assert_matches_template(res)


# =========================================================================== R9


def test_prior_arrays_depend_only_on_the_window(sim_data_stationary):
    """R9 (DESIGN §6.6): the window ``data.upto(origin)`` resolves to the same arrays, EB values
    and padded data whatever happens after ``origin``; the full sample resolves differently."""
    origin = sim_data_stationary.index[150]
    raw = sim_data_stationary.raw.copy()
    raw.iloc[160:] = raw.iloc[160:] * 3.0 + 1.0  # rewrite the future
    edited = rb.VARData(raw=raw, tcodes=sim_data_stationary.tcodes)
    model = bvar(rb.SV(outliers=True))
    a = resolve.resolve(model, sim_data_stationary.upto(origin), RUNTIME)
    b = resolve.resolve(model, edited.upto(origin), RUNTIME)
    assert a.static == b.static and a.salt == b.salt
    assert_tree_equal(a.prior, b.prior)
    assert_tree_equal(a.data, b.data)
    assert a.hyper.keys() == b.hyper.keys()
    for key in a.hyper:
        np.testing.assert_array_equal(np.asarray(a.hyper[key]), np.asarray(b.hyper[key]))
    full = resolve.resolve(model, sim_data_stationary, RUNTIME)
    assert full.hyper["lambda"] != a.hyper["lambda"]
    assert not np.array_equal(full.prior.eq.coef.prec0, a.prior.eq.coef.prec0)


# =========================================================================== size rules


def test_gram_rule_examples():
    """DESIGN §10.5: packed up to 256 MiB of products, else the largest 512-multiple tile within
    64 MiB; explicit choices are kept (a tiled choice gets the rule's tile)."""
    gram = resolve.resolve_gram
    assert gram("auto", 320, 5050) == ("packed", 0)  # n = 20, p = 4: 12.9 MB
    assert gram("auto", 320, 31375) == ("packed", 0)  # n = 50, p = 4: 80 MB
    q = 30 * 13 + 1 + 29  # n = 30, p = 13, T = 600
    kind, tile = gram("auto", 608, q * (q + 1) // 2)
    assert kind == "packed_tiled" and tile % 512 == 0
    assert 608 * tile * 8 <= 64 * 2**20 < 608 * (tile + 512) * 8
    assert gram("auto", 2**15, 10**6) == ("packed_tiled", 512)  # never below one unit
    assert gram("batched", 320, 5050) == ("batched", 0)
    assert gram("packed", 608, 10**6) == ("packed", 0)
    assert gram("packed_tiled", 320, 5050)[0] == "packed_tiled"
    with pytest.raises(ValueError, match="unknown gram"):
        gram("dense", 320, 5050)


def test_const_path_rule():
    """DESIGN §3.7.2: eigen (Mode I), fixed-factor (Mode II), Cholesky (iw_equivalent) for
    structural Linear × Constant; weighted for SV, transitory scales and the reduced form;
    ``const_path="cholesky"`` turns the fast paths off."""
    path = resolve.resolve_const_path
    cases = [
        (bvar(rb.Constant("ig")), "eigen", "cholesky_const"),
        (bvar(rb.Constant("iw"), form="structural"), "fixed_factor", "cholesky_const"),
        (bvar(rb.Constant("iw_equivalent")), "cholesky_const", "cholesky_const"),
        (bvar(rb.SV()), "weighted", "weighted"),
        (bvar(rb.SV(outliers=True)), "weighted", "weighted"),
        (bvar(rb.Constant("ig"), form="reduced"), "weighted", "weighted"),
    ]
    for model, auto, cholesky in cases:
        parts = static_parts(model)
        assert (path("auto", parts), path("cholesky", parts)) == (auto, cholesky), model.name
    with pytest.raises(ValueError, match="unknown const_path"):
        path("eigen", static_parts(cases[0][0]))


def test_cccm_precompute_rule():
    assert resolve.resolve_cccm_precompute(20, 81, 4)  # 2 MB
    assert resolve.resolve_cccm_precompute(1, 8192, 1)  # exactly 512 MiB
    assert not resolve.resolve_cccm_precompute(1, 8192, 2)


def test_size_rules_reach_the_static_config(data):
    """``t_bucket``, ``gram`` and ``const_path`` settings resolve into ``StaticConfig`` (spec fields,
    so a cache key covers them; D-23, E14)."""
    base = dict(chains=2, draws=10, burn=0)
    r = resolve.resolve(
        bvar(rb.Constant("ig"), mcmc=rb.MCMCSettings(**base, t_bucket=16, const_path="cholesky")),
        data,
        RUNTIME,
    )
    assert r.static.shapes.T_pad == 80 and r.static.mean.const_path == "cholesky_const"
    np.testing.assert_array_equal(r.data.tmask, (np.arange(80) < 78).astype(float))
    r = resolve.resolve(bvar(mcmc=rb.MCMCSettings(**base, gram="batched")), data, RUNTIME)
    assert (r.static.mean.gram, r.static.shapes.T_pad) == ("batched", 96)
    r = resolve.resolve(bvar(mcmc=rb.MCMCSettings(**base, gram="packed_tiled")), data, RUNTIME)
    assert r.static.mean.gram == "packed_tiled" and r.static.mean.gram_tile >= 512
    expected, _ = engine_data_from_vardata(data, LAGS, "c", 16)
    r16 = resolve.resolve(bvar(mcmc=rb.MCMCSettings(**base, t_bucket=16)), data, RUNTIME)
    assert_tree_equal(r16.data, expected)
    assert r16.static.chunk == RUNTIME.chunk


def test_size_rules_read_the_form_shapes_and_the_chain_count(data, monkeypatch):
    """``resolve`` feeds the Gram rule the structural ``P = q(q+1)/2`` and the reduced
    ``P_x = k(k+1)/2``, and the CCCM rule ``MCMCSettings.chains`` (DESIGN §10.5): with the bounds
    set between the two sizes, the forms and the chain counts resolve differently."""
    reduced = dict(form="reduced", hierarchy="fixed", cross=0.5)
    shapes = resolve.resolve(bvar(hierarchy="fixed"), data, RUNTIME).static.shapes
    assert (shapes.P_x, shapes.P) == (28, 45)  # k = 7, q = 9
    monkeypatch.setattr(resolve, "PACKED_MAX_BYTES", shapes.T_pad * shapes.P_x * 8)
    s = resolve.resolve(bvar(hierarchy="fixed"), data, RUNTIME).static.mean
    r = resolve.resolve(bvar(**reduced), data, RUNTIME).static.mean
    assert (
        s.gram == "packed_tiled"
        and s.gram_tile == resolve.resolve_gram("packed_tiled", shapes.T_pad, 1)[1]
    )
    assert (r.gram, r.gram_tile) == ("packed", 0)
    monkeypatch.setattr(resolve, "CCCM_MAX_BYTES", 2 * shapes.n * shapes.k**2 * 8)

    def precompute(chains):
        model = bvar(**reduced, mcmc=dataclasses.replace(SETTINGS, chains=chains))
        return resolve.resolve(model, data, RUNTIME).static.mean.cccm_precompute

    assert precompute(2) and not precompute(3)


def test_plan_and_storage_reach_the_static_config(data):
    """A ``FitPlan`` sets ``store.in_loop_forecast`` (DESIGN §5.8); ``store="paths"`` stores the
    latent paths (§5.3)."""
    model = bvar(mcmc=dataclasses.replace(SETTINGS, store="paths"))
    r = resolve.resolve(model, data, RUNTIME, plan=rb.FitPlan(horizon=4))
    assert r.static.store.in_loop_forecast == 4 and r.static.store.paths
    r = resolve.resolve(bvar(), data, RUNTIME)
    assert r.static.store.in_loop_forecast == 0 and not r.static.store.paths
    with pytest.raises(TypeError, match="plan"):
        resolve.resolve(bvar(), data, RUNTIME, plan=4)


# =========================================================================== salts and seeds


def test_salts_differ_across_windows_and_orderings(sim_data_stationary):
    """The data salt is ``crc32(data.fingerprint())`` (DESIGN §5.6): one value per window and per
    ordering, so chain noise is independent across them even under one seed."""
    model = bvar(hierarchy="fixed")
    d = sim_data_stationary
    salt = lambda x: resolve.resolve(model, x, RUNTIME).salt  # noqa: E731
    windows = [d.upto(d.index[100]), d.upto(d.index[101]), d.tail(150), d]
    orderings = [
        d.select(order) for order in (("rate", "gdp", "prices"), ("prices", "gdp", "rate"))
    ]
    salts = [salt(x) for x in windows + orderings]
    assert len(set(salts)) == len(salts)
    assert salts[3] == keys.data_salt(d.fingerprint()) == salt(d)
    assert all(0 <= s < 2**32 for s in salts)


# =========================================================================== warnings


def test_setting_warnings_are_recomputed_for_the_mcmc_fit(covid_data):
    """The 2020 check (SPEC §5.6) is recomputed by the MCMC fit: ``SV(outliers=True)`` counts as an
    outlier treatment, so it silences the warning; the twin's own 2020 warning never reaches the
    MCMC fit (DESIGN §6.5 step 4, §8.3)."""
    for vol, expected in ((rb.SV(), True), (rb.SV(outliers=True), False)):
        r = resolve.resolve(bvar(vol), covid_data, RUNTIME)
        assert r.warning_inputs["covers_2020"] is True
        assert not [w for w in r.warning_inputs["eb_warnings"] if "2020" in w]
        messages = monitor.warning_catalogue(
            monitor.DiagnosticsInput(
                static=r.static, settings=SETTINGS, variables=covid_data.variables,
                **r.warning_inputs,
            )
        )  # fmt: skip
        assert any("covers 2020" in m for m in messages) is expected
    pre = resolve.resolve(bvar(), covid_data.upto("2019Q4"), RUNTIME)
    assert pre.warning_inputs["covers_2020"] is False


@pytest.mark.parametrize(("start", "covers"), [("2020Q3", False), ("2020Q2", True)])
def test_covers_2020_reads_the_estimation_periods(simulate_var, start, covers):
    """The 2020 check reads the estimation periods, as Phase 1 does, not the pre-sample lag rows:
    a sample from 2020Q3 (lags 2020Q3–Q4, estimation from 2021Q1) does not cover 2020; one from
    2020Q2 (estimation from 2020Q4) does."""
    index = pd.period_range(start, periods=40, freq="Q")
    frame = pd.DataFrame(simulate_var(40, 12), index=index, columns=["gdp", "prices", "rate"])
    r = resolve.resolve(bvar(hierarchy="fixed"), rb.VARData.from_frame(frame, 1), RUNTIME)
    assert r.periods[0] == index[LAGS]
    assert r.warning_inputs["covers_2020"] is covers


def test_tcode_clash_is_recomputed(data):
    """An explicit own-lag mean of 0 on level series (tcode 1) clashes (SPEC §5.3)."""
    r = resolve.resolve(bvar(mean=0.0, hierarchy="fixed"), data, RUNTIME)
    assert r.warning_inputs["tcode_clash"] == data.variables
    messages = monitor.warning_catalogue(
        monitor.DiagnosticsInput(static=r.static, settings=SETTINGS, **r.warning_inputs)
    )
    assert any("contradict the transformation codes" in m for m in messages)
    assert (
        resolve.resolve(bvar(hierarchy="fixed"), data, RUNTIME).warning_inputs["tcode_clash"] == ()
    )


# =========================================================================== errors


def test_resolve_rejects_analytic_models_and_bad_arguments(data):
    with pytest.raises(ValueError, match="MCMC compositions"):
        resolve.resolve(rb.BVAR(), data, RUNTIME)
    with pytest.raises(TypeError, match="VARData"):
        resolve.resolve(bvar(), data.values, RUNTIME)
    with pytest.raises(TypeError, match="Runtime"):
        resolve.resolve(bvar(), data, {"chunk": 10})


def test_bvar_fit_dispatches_to_the_resolver():
    """The lazy entry point of ``BVAR.fit`` is this module's ``fit_bvar`` (DESIGN §1.1 rule 2)."""
    from rai_bvar import api

    assert api._mcmc_entry_point() is resolve.fit_bvar


# =========================================================================== every composition


COMPOSITIONS = [
    ("structural", rb.Constant("ig")),
    ("structural", rb.Constant("iw")),
    ("structural", rb.Constant("iw_equivalent")),
    ("structural", rb.SV()),
    ("structural", rb.SV(outliers=True)),
    ("structural", rb.SV(tails="t")),
    ("structural", rb.SV(outliers=True, tails="t")),
    ("structural", rb.SV(rw=True)),
    ("structural", rb.SV(h0=2.0)),
    ("structural", rb.SV(sigma2_prior=rb.sv_priors.InvGamma(2.0, 0.5))),
    ("reduced", rb.Constant("ig")),
    ("reduced", rb.Constant("iw_equivalent")),
    ("reduced", rb.SV()),
    ("reduced", rb.SV(rw=True)),  # the Clark–Ravazzolo primary SV arm (DESIGN §9.5 row 18)
    ("reduced", rb.SV(outliers=True, tails="t")),
]


@pytest.mark.slow
@pytest.mark.parametrize(("form", "vol"), COMPOSITIONS, ids=lambda x: getattr(x, "label", x))
def test_every_composition_fits(form, vol, data):
    """DESIGN §13 WP18 acceptance: ``fit()`` works for every composition of §11.1 (rows 3–8) and
    the SV variants of §9.5 (2 chains × 30 draws)."""
    model = bvar(vol, form=form, cross=0.5 if form == "reduced" else 1.0)
    assert model.engine == "mcmc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = model.fit(data, runtime=RUNTIME)
    assert isinstance(fit, MCMCFit)
    assert np.isfinite(fit.monitor["loglik"]).all()
    assert fit.posterior["posterior"].to_dataset().sizes["draw"] == 30
    assert fit.static == resolve.resolve(model, data, RUNTIME).static


@pytest.mark.slow
def test_covid_warning_end_to_end(covid_data):
    """The 2020 warning in ``fit.warnings``: present with ``SV()``, absent with
    ``SV(outliers=True)`` (DESIGN §8.3)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plain = bvar(rb.SV()).fit(covid_data, runtime=RUNTIME)
        robust = bvar(rb.SV(outliers=True)).fit(covid_data, runtime=RUNTIME)
    assert any("covers 2020" in w for w in plain.warnings)
    assert not any("covers 2020" in w for w in robust.warnings)


@pytest.mark.slow
def test_fit_plan_end_to_end(data):
    """``BVAR.fit(data, plan=FitPlan(...))`` compiles the in-loop forecast hook (DESIGN §5.8) and
    the fit carries the plan's forecasts."""
    from rai_bvar.engines.mcmc import forecast as forecast_mod

    if getattr(forecast_mod, "forecast_core", None) is None:
        pytest.skip("engines.mcmc.forecast.forecast_core (WP17) not available yet")
    plan = rb.FitPlan(horizon=2, draws=6, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = bvar(hierarchy="fixed").fit(data, runtime=RUNTIME, plan=plan)
    assert fit.static.store.in_loop_forecast == 2
    assert fit.in_loop_forecast is not None and fit.in_loop_forecast.matches(2, 6, 1)
