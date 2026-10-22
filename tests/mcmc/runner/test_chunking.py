"""Invariances of the runner (DESIGN §5.1, §5.6, §9.3 item 4). Bitwise: chunk size and burn piece
size, thinning with aligned sweeps, the exact final state, and chain-count prefixes with
``chains="map"``. To rounding (Stage 2 amendment of §5.6): chain-count prefixes under
``chains="vmap"``, where the batch size changes how XLA vectorises reductions (``rtol 1e-12``
with the toy kernels, which differ in the last bit on x86 Linux; ``rtol 1e-9`` with the
production kernels on short runs)."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.runner import run_chains
from rai_bvar.engines.mcmc.state import zero_accumulators
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

BASE = toys.toy_static("linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5)


def _run(static, settings, seed=3, salt=9, data=None, chains="vmap"):
    fns = toys.cached_runner(static, toys.toy_overrides(static), chains=chains)
    prior = toys.toy_prior(static)
    data = toys.toy_data(static, 20, 1) if data is None else data
    out = run_chains(
        fns, static, data, prior, settings, Runtime(chunk=static.chunk), seed=seed, salt=salt
    )
    return out, fns, data, prior


def _same(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _close(a, b):
    """Floats to rounding, integers (indicators, counters) exactly."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if np.issubdtype(x.dtype, np.floating):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
        else:
            np.testing.assert_array_equal(x, y)


def _streams(out):
    return out.store.stream("monitor"), out.store.stream("params")


def test_chunk_and_burn_piece_size_invariance():
    """chunk 5 (burn pieces 5+2, last chunk partial) and chunk 8 (one burn piece): identical."""
    settings = MCMCSettings(chains=2, draws=13, burn=7, thin=1)
    a, *_ = _run(BASE, settings)
    b, *_ = _run(dataclasses.replace(BASE, chunk=8), settings)
    _same(_streams(a), _streams(b))
    _same(a.final_state, b.final_state)
    _same(a.acceptance, b.acceptance)
    _same(a.latent, b.latent)


def test_thinning_with_aligned_sweeps():
    """thin = 3 keeps the states after sweeps burn + 3(j+1): kept draw 3j+2 of a thin = 1 run."""
    a, *_ = _run(BASE, MCMCSettings(chains=2, draws=12, burn=4, thin=1))
    b, *_ = _run(BASE, MCMCSettings(chains=2, draws=4, burn=4, thin=3))
    ma, pa = _streams(a)
    mb, pb = _streams(b)
    _same(jax.tree.map(lambda x: x[:, 2::3], ma), mb)
    _same(jax.tree.map(lambda x: x[:, 2::3], pa), pb)
    _same(a.final_state, b.final_state)  # 4 + 12 sweeps either way


def test_chain_prefix_invariance():
    """The first 3 of 5 chains equal a 3-chain run with the toy kernels (prefix-stable keys;
    DESIGN §9.3 item 4): bitwise on macOS, to rounding on x86 Linux (another vmap batch size)."""
    settings = MCMCSettings(chains=5, draws=6, burn=3)
    a, *_ = _run(BASE, settings)
    b, *_ = _run(BASE, dataclasses.replace(settings, chains=3))
    _close(jax.tree.map(lambda x: x[:3], _streams(a)), _streams(b))
    _close(jax.tree.map(lambda x: x[:3], a.final_state), b.final_state)


def test_chain_prefix_invariance_with_mapped_chains():
    """``chains="map"`` runs one identical program per chain: any prefix, one chain included."""
    settings = MCMCSettings(chains=4, draws=6, burn=3)
    a, *_ = _run(BASE, settings, chains="map")
    for C in (1, 3):
        b, *_ = _run(BASE, dataclasses.replace(settings, chains=C), chains="map")
        _same(jax.tree.map(lambda x, C=C: x[:C], _streams(a)), _streams(b))
        _same(jax.tree.map(lambda x, C=C: x[:C], a.final_state), b.final_state)


def test_mapped_chains_production_prefix():
    """The same with the production kernels (one SV composition), once they exist. (With
    ``chains="vmap"`` the production SV and CCCM kernels are not bitwise batch-size invariant.)"""
    from rai_bvar.engines.mcmc import keys
    from rai_bvar.engines.mcmc.compose import compose
    from rai_bvar.engines.mcmc.runner import make_runner
    from rai_bvar.validation.joint import joint_spec

    spec = joint_spec("structural_sv")
    static = dataclasses.replace(spec.static, chunk=5)
    try:
        compose(static)
    except NotImplementedError as err:
        pytest.skip(f"production kernels not available yet: {err}")
    key = jax.random.key(7)
    data = jax.tree.map(
        np.asarray, spec.simulate(jax.random.fold_in(key, 1), spec.prior_sample(key))
    )
    fns = make_runner(static, chains="map")
    cache = fns.prepare(data, spec.prior)
    out = {}
    for C in (1, 3):
        st = fns.init(keys.init_keys(3, 4, C), data, spec.prior, cache)
        ck = keys.chain_keys(keys.fit_root(3, 4), C)
        out[C] = fns.burn(ck, st, data, spec.prior, cache, np.uint32(0), np.int32(4))
    _same(jax.tree.map(lambda x: x[:1], out[3]), out[1])


def test_final_state_is_state_after_exactly_burn_plus_draws_thin_sweeps():
    """``snap`` of the last (partial) chunk: init, ``burn`` sweeps, zeroed accumulators, then
    ``draws·thin`` sweeps — not the surplus sweeps of the last chunk (D-27)."""
    settings = MCMCSettings(chains=2, draws=7, burn=6, thin=2)
    out, fns, data, prior = _run(BASE, settings)
    cache = fns.prepare(data, prior)
    st = fns.init(keys.init_keys(3, 9, 2), data, prior, cache)
    ckeys = keys.chain_keys(keys.fit_root(3, 9), 2)
    st = fns.burn(ckeys, st, data, prior, cache, np.uint32(0), np.int32(6))
    st = zero_accumulators(st)
    st = fns.burn(ckeys, st, data, prior, cache, np.uint32(6), np.int32(14))
    _same(jax.device_get(st), out.final_state)
    assert out.sweeps == 6 + 7 * 2


def test_accumulators_cover_exactly_the_sampling_sweeps():
    """Accumulators are zeroed at the burn-in → sampling boundary (D-27, E8): a counter the toy
    increments every sweep reads exactly 1 per sampling sweep, and the latent moments equal the
    moments of the h paths of every sampling sweep."""
    static = dataclasses.replace(BASE, store=dataclasses.replace(BASE.store, paths=True))
    out, *_ = _run(
        static, MCMCSettings(chains=2, draws=11, burn=9, thin=1, path_thin=1, store="paths")
    )
    np.testing.assert_array_equal(out.acceptance["acc_cgp"], 1.0)
    np.testing.assert_array_equal(out.acceptance["acc_cs2"], 0.5)
    h = out.store.stream("paths").h  # (C, D, n, T_pad+1), every sweep (thin = 1)
    np.testing.assert_allclose(out.latent["h_mean"], h.mean(axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.latent["h_sd"], h.std(axis=1), rtol=1e-6, atol=1e-7)
    T = 20
    p_out = out.latent["p_outlier"]
    np.testing.assert_allclose(p_out[..., :T], 0.1, rtol=1e-12)
    np.testing.assert_array_equal(p_out[..., T:], 0.0)
    assert out.acceptance["acc_rej_run_max"].dtype == np.int32


def test_accumulators_with_thinning_cover_every_sampling_sweep():
    """With ``thin = 3`` the accumulators run over all ``draws·thin`` sampling sweeps, not only the
    kept ones: rates are per sweep (``acc_cgp`` = 1, ``acc_cs2`` = 0.5) and the latent moments
    equal the per-sweep moments of the h paths of a ``thin = 1`` run of the same sweeps (keys
    depend only on the absolute sweep index)."""
    static = dataclasses.replace(BASE, store=dataclasses.replace(BASE.store, paths=True))
    thinned, *_ = _run(
        static, MCMCSettings(chains=2, draws=5, burn=4, thin=3, path_thin=1, store="paths")
    )
    every, *_ = _run(
        static, MCMCSettings(chains=2, draws=15, burn=4, thin=1, path_thin=1, store="paths")
    )
    np.testing.assert_array_equal(thinned.acceptance["acc_cgp"], 1.0)
    np.testing.assert_array_equal(thinned.acceptance["acc_cs2"], 0.5)
    h = every.store.stream("paths").h  # every sampling sweep 4..18
    np.testing.assert_allclose(thinned.latent["h_mean"], h.mean(axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(thinned.latent["h_sd"], h.std(axis=1), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(thinned.latent["p_outlier"][..., :20], 0.1, rtol=1e-12)
    _same(thinned.latent, every.latent)
    _same(thinned.final_state, every.final_state)


def _close(a, b, rtol=1e-9, atol=1e-12):
    """Equal to rounding: floating leaves to ``rtol``/``atol``, integer leaves exactly."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if np.issubdtype(x.dtype, np.floating):
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(x, y)


def _production_prefix(name, chains, C, C_ref, *, bitwise):
    """Burn 4 sweeps with ``C`` and ``C_ref`` chains of a production composition and compare the
    first ``C`` chains: bit for bit, or to rounding (DESIGN §5.6 Stage 2 amendment)."""
    from rai_bvar.engines.mcmc.compose import compose
    from rai_bvar.engines.mcmc.runner import make_runner
    from rai_bvar.validation.joint import joint_spec

    spec = joint_spec(name)
    static = dataclasses.replace(spec.static, chunk=5)
    try:
        compose(static)
    except NotImplementedError as err:
        pytest.skip(f"production kernels not available yet: {err}")
    key = jax.random.key(7)
    data = jax.tree.map(
        np.asarray, spec.simulate(jax.random.fold_in(key, 1), spec.prior_sample(key))
    )
    fns = make_runner(static, chains=chains)
    cache = fns.prepare(data, spec.prior)
    out = {}
    for c in (C, C_ref):
        st = fns.init(keys.init_keys(3, 4, c), data, spec.prior, cache)
        ck = keys.chain_keys(keys.fit_root(3, 4), c)
        out[c] = jax.device_get(
            fns.burn(ck, st, data, spec.prior, cache, np.uint32(0), np.int32(4))
        )
    (_same if bitwise else _close)(jax.tree.map(lambda x: x[:C], out[C_ref]), out[C])


def test_production_chain_prefix_with_vmapped_chains_to_rounding():
    """DESIGN §5.6 (Stage 2 amendment): with ``chains="vmap"`` the production SV kernels give the
    first chain of a 3-chain run to rounding (XLA vectorises their reductions differently for
    different batch sizes), not necessarily bit for bit."""
    _production_prefix("structural_sv", "vmap", 1, 3, bitwise=False)


def test_production_chain_prefix_with_vmapped_reduced_form_to_rounding():
    """The same for the CCCM (reduced-form) kernels."""
    _production_prefix("reduced_sv", "vmap", 1, 3, bitwise=False)


@pytest.mark.slow
def test_production_chain_prefix_3_of_5_with_vmapped_chains_to_rounding():
    """The DESIGN §9.3 item 4 case (first 3 of 5 chains) with the SV + outlier + Student-t
    kernels: to rounding with ``chains="vmap"``, bit for bit with ``chains="map"``."""
    _production_prefix("structural_sv_outliers_t", "vmap", 3, 5, bitwise=False)
    _production_prefix("structural_sv_outliers_t", "map", 3, 5, bitwise=True)
