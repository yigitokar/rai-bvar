"""The sweep skeleton (DESIGN §4.5, §5.6, §5.7): plan execution, keys, status codes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.keys import Block
from rai_bvar.engines.mcmc.plan import Step
from rai_bvar.engines.mcmc.state import Empty, StepInfo, SweepCtx, decode_error, encode_error
from rai_bvar.engines.mcmc.sweep import make_sweep, record_eq_status

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def _setup(static, seed=0, chains=3, mean=None):
    ov = toys.toy_overrides(static, mean=mean)
    fns = toys.cached_runner(static, ov)
    prior = toys.toy_prior(static)
    data = toys.toy_data(static, 20, seed)
    cache = fns.prepare(data, prior)
    st = fns.init(keys.init_keys(seed, 5, chains), data, prior, cache)
    ckeys = keys.chain_keys(keys.fit_root(seed, 5), chains)
    return fns, data, prior, cache, st, ckeys


def _equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


CONFIGS = {
    "structural_sv_outlier": dict(
        mean="linear_structural", persistent="sv", transitory="outlier", n=3
    ),
    "mode2_constant": dict(mean="linear_structural", persistent="constant", mode="II", n=3),
    "reduced_sv": dict(mean="linear_reduced", persistent="sv", n=3),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fused_equals_unfused_bitwise(name):
    """Running every equation step in its own eq_map gives the same draws bit for bit."""
    static = toys.toy_static(**CONFIGS[name], T_eff=20, chunk=5)
    fns, data, prior, cache, st, ckeys = _setup(static)
    program = fns.program
    fused = make_sweep(program)
    unfused = make_sweep(program, fused=False)

    def run(sweep):
        def many(ck, s):
            return jax.lax.fori_loop(
                0,
                4,
                lambda r, x: sweep(
                    ck, jnp.uint32(10) + r.astype(jnp.uint32), x, data, prior, cache
                ),
                s,
            )

        return jax.jit(jax.vmap(many))(ckeys, st)

    _equal(run(fused), run(unfused))


def test_vmap_over_chains_equals_sequential_loop():
    static = toys.toy_static(**CONFIGS["structural_sv_outlier"], T_eff=20, chunk=5)
    fns, data, prior, cache, st, ckeys = _setup(static)
    sweep = make_sweep(fns.program)
    one = jax.jit(lambda ck, s, it: sweep(ck, it, s, data, prior, cache))
    batched = jax.jit(
        jax.vmap(lambda ck, s, it: sweep(ck, it, s, data, prior, cache), in_axes=(0, 0, None))
    )
    it = np.uint32(17)
    loop = [one(ckeys[c], jax.tree.map(lambda x, c=c: x[c], st), it) for c in range(3)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *loop)
    _equal(batched(ckeys, st, it), stacked)


def test_counter_must_be_uint32():
    static = toys.toy_static(**CONFIGS["mode2_constant"], T_eff=20, chunk=5)
    fns, data, prior, cache, st, ckeys = _setup(static, chains=1)
    sweep = make_sweep(fns.program)
    one = jax.tree.map(lambda x: x[0], st)
    with pytest.raises(TypeError, match="uint32"):
        sweep(ckeys[0], jnp.int32(3), one, data, prior, cache)
    out = sweep(ckeys[0], jnp.uint32(3), one, data, prior, cache)
    assert out.status.it.dtype == jnp.uint32 and out.status.code.dtype == jnp.int32


def test_key_reuse_checker_passes_a_full_sweep():
    """One consumer per key (DESIGN §5.6; N07 §4.1)."""
    static = toys.toy_static(**CONFIGS["reduced_sv"], T_eff=20, chunk=5)
    fns, data, prior, cache, st, ckeys = _setup(static, chains=1)
    sweep = make_sweep(fns.program)
    one = jax.tree.map(lambda x: x[0], st)
    with jax.debug_key_reuse(True):
        out = jax.jit(lambda ck, s: sweep(ck, jnp.uint32(2), s, data, prior, cache))(ckeys[0], one)
    assert np.isfinite(np.asarray(out.glob.mean.Pi)).all()


def test_global_step_key_follows_the_sweep_index():
    """The pre-phase CCCM step draws with ``step_key(ckey, it, G_CCCM)`` (DESIGN §5.6): the Π of a
    sweep equals the step run on its own with that key, and two sweep indices give different Π
    from the same state (a key without ``it`` would reuse the CCCM noise every sweep)."""
    static = toys.toy_static(**CONFIGS["reduced_sv"], T_eff=20, chunk=5)
    fns, data, prior, cache, st, ckeys = _setup(static, chains=1)
    program = fns.program
    sweep = make_sweep(program)
    one = jax.tree.map(lambda x: x[0], st)
    (cccm,) = program.global_pre
    ctx = SweepCtx(data, prior, cache, Empty())
    run = jax.jit(lambda it: sweep(ckeys[0], it, one, data, prior, cache).glob.mean.Pi)
    step = jax.jit(
        lambda it: (
            cccm.sample(keys.step_key(ckeys[0], it, Block.G_CCCM), one, ctx, program.eq_map)[
                0
            ].glob.mean.Pi
        )
    )
    pi10, pi11 = run(jnp.uint32(10)), run(jnp.uint32(11))
    np.testing.assert_array_equal(pi10, step(jnp.uint32(10)))
    np.testing.assert_array_equal(pi11, step(jnp.uint32(11)))
    assert not np.any(np.asarray(pi10) == np.asarray(pi11))


def test_status_records_first_failure():
    """The DriftMean fails for equation 1 at sweep 7: the status keeps (code, it) of that first
    failure and the smallest pivot, and later sweeps do not overwrite it."""
    static = toys.toy_static("linear_structural", "constant", n=3, T_eff=20, chunk=5)
    drift = toys.DriftMean(static.mean, static.shapes)
    fns, data, prior, cache, st, ckeys = _setup(static, chains=2, mean=drift)
    st = fns.burn(ckeys, st, data, prior, cache, np.uint32(0), np.int32(12))
    code = np.asarray(st.status.code)
    assert (code == int(encode_error(Block.EQ_MEAN, 1))).all()
    assert (np.asarray(st.status.it) == 7).all()
    assert decode_error(int(code[0])) == (Block.EQ_MEAN, 1)
    assert (np.asarray(st.status.min_pivot) <= 0).all()


def test_record_eq_status_order():
    """First failing step in plan order, then the lowest failing equation."""
    from rai_bvar.engines.mcmc.state import ChainState, Status

    st = ChainState(eq=None, glob=None, status=Status.initial())  # only status is read
    steps = (
        Step("equation", Block.EQ_MEAN, "equation", frozenset(), frozenset()),
        Step("vol", Block.EQ_VOL, "equation", frozenset(), frozenset()),
    )
    infos = {
        "equation": StepInfo(
            jnp.array([True, True, False, False]), jnp.array([1.0, 0.5, 0.0, 0.0])
        ),
        "vol": StepInfo(jnp.array([False, True, True, True]), jnp.full(4, jnp.inf)),
    }
    out = record_eq_status(st, infos, steps, jnp.uint32(9))
    assert int(out.status.code) == int(encode_error(Block.EQ_MEAN, 2))
    assert int(out.status.it) == 9 and float(out.status.min_pivot) == 0.0
    later = record_eq_status(out, {"vol": infos["vol"]}, steps, jnp.uint32(10))
    assert int(later.status.code) == int(out.status.code) and int(later.status.it) == 9
