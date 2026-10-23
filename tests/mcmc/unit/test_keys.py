"""PRNG key derivation (DESIGN §5.6): enum stability, golden values, determinism, uint32 counters."""

from __future__ import annotations

import inspect
import itertools
import json
import os
import subprocess
import sys
import textwrap
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from rai_bvar.engines.mcmc import keys as K

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

PINNED_JAX = "0.12.0.1"  # golden key values are tied to the JAX version (DESIGN R11)


def kd(key) -> list[int]:
    return [int(x) for x in np.asarray(random.key_data(key))]


def test_enum_values_are_frozen():
    # Append-only enums: renumbering any member changes every stream (DESIGN §5.6).
    assert {m.name: int(m) for m in K.Stream} == {
        "MCMC": 0, "INIT": 1, "FORECAST": 2, "VALIDATION": 3, "IRF": 4,
    }  # fmt: skip
    assert {m.name: int(m) for m in K.Block} == {
        "EQ_MEAN": 0, "EQ_CONTEMP": 1, "EQ_VOL": 2, "EQ_COEF_LOCAL": 3, "G_CCCM": 16,
        "G_COEF_GLOBAL": 17, "G_COMMON_SV": 18, "G_LP_SCALE": 19, "G_REGIME": 32,
        "G_FACTORS": 33, "G_LOADINGS": 34, "G_FACTOR_SV": 35, "G_B0_ROWS": 36,
    }  # fmt: skip
    assert {m.name: int(m) for m in K.Sub} == {
        "S": 0, "H": 1, "H_ACC": 2, "C_GP": 3, "C_GP_ACC": 4, "C_S2": 5, "C_S2_ACC": 6,
        "NC_PHI": 7, "NC_PHI_ACC": 8, "NC_MS": 9, "NC_MS_ACC": 10, "O": 11, "P": 12, "NU": 13,
        "TAU": 14,
    }  # fmt: skip
    assert int(K.Sub.Z) == 0  # mean, contemp and CCCM sub-draw
    assert {m.name: int(m) for m in K.Ahead} == {"VOL": 0, "SHOCK": 1, "MEAN": 2, "GLOBAL": 3}
    assert {m.name: int(m) for m in K.Tier} == {
        "U": 0, "B": 1, "G": 2, "S": 3, "M": 4, "A": 5, "P": 6,
    }  # fmt: skip
    # Phase 3–5 reservations stay inside their ranges; error codes need block < 1000
    assert all(
        17 <= b <= 31 for b in (K.Block.G_COEF_GLOBAL, K.Block.G_COMMON_SV, K.Block.G_LP_SCALE)
    )
    assert all(32 <= b <= 63 for b in K.Block if b.name in {"G_REGIME", "G_B0_ROWS"})


def test_derivation_is_the_documented_fold_in_chain():
    seed, salt, c, it, i = 7, 123456789, 2, 9876, 3
    root = random.key(seed, impl="threefry2x32")
    fit = random.fold_in(random.fold_in(root, 0), salt)
    assert kd(K.fit_root(seed, salt)) == kd(fit)
    ck = random.fold_in(fit, c)
    assert kd(K.chain_keys(K.fit_root(seed, salt), 4)[c]) == kd(ck)
    sk = random.fold_in(random.fold_in(ck, it), 16)
    assert kd(K.step_key(ck, it, K.Block.G_CCCM)) == kd(sk)
    ek = random.fold_in(sk, i)
    assert kd(K.eq_key(sk, i)) == kd(ek)
    assert kd(K.sub_key(ek, K.Sub.NU)) == kd(random.fold_in(ek, 13))
    init = random.fold_in(random.fold_in(random.fold_in(root, 1), salt), c)
    assert kd(K.init_keys(seed, salt, 3)[c]) == kd(init)
    fc = random.fold_in(random.fold_in(random.key(11, impl="threefry2x32"), 2), c)
    fc = random.fold_in(random.fold_in(random.fold_in(fc, 5), 1), i)
    assert kd(K.forecast_key(11, c, 5, K.Ahead.SHOCK, i)) == kd(fc)
    val = random.fold_in(random.fold_in(random.fold_in(random.fold_in(root, 3), 2), 4), 8)
    assert kd(K.validation_key(seed, K.Tier.G, 4, 8)) == kd(val)


def derive_golden(K, np) -> dict:
    """The golden derivations. Self-contained: its source also runs in a subprocess."""
    salt = K.data_salt("abc")
    root = K.fit_root(20261022, salt)
    ck = K.chain_keys(root, 3)
    ek = K.eq_key(K.step_key(ck[0], 7, K.Block.EQ_VOL), 4)
    mk = K.eq_key(K.step_key(ck[0], 7, K.Block.EQ_MEAN), 4)
    return {
        "fit_root": root,
        "chain_2": ck[2],
        "step_cccm": K.step_key(ck[1], np.uint32(1234), K.Block.G_CCCM),
        "eq_vol": ek,
        "sub_tau": K.sub_key(ek, K.Sub.TAU),
        "sub_z": K.sub_key(mk, K.Sub.Z),
        "init_1": K.init_keys(5, 99, 2)[1],
        "forecast": K.forecast_key(42, 1, 17, K.Ahead.SHOCK, 2),
        "validation": K.validation_key(0, K.Tier.G, 3, 11),
    }


GOLDEN = {
    "fit_root": [4125446508, 988745234],
    "chain_2": [1918026647, 1773630621],
    "step_cccm": [3704251180, 2840794690],
    "eq_vol": [1614763699, 297140592],
    "sub_tau": [660446957, 3968974166],
    "sub_z": [1167686374, 2297100866],
    "init_1": [624843028, 2822562547],
    "forecast": [2923604062, 1640196588],
    "validation": [1776618869, 2296093282],
}


@pytest.mark.skipif(jax.__version__ != PINNED_JAX, reason=f"golden keys pinned to jax {PINNED_JAX}")
def test_golden_key_values():
    assert {name: kd(k) for name, k in derive_golden(K, np).items()} == GOLDEN


def test_data_salt_is_crc32():
    assert K.data_salt("abc") == 891568578 == zlib.crc32(b"abc")
    fp = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    salt = K.data_salt(fp)
    assert isinstance(salt, int) and 0 <= salt < 2**32
    assert K.data_salt(fp) == salt and K.data_salt(fp[::-1]) != salt
    with pytest.raises(TypeError):
        K.data_salt(b"abc")  # type: ignore[arg-type]


def test_chain_prefixes_are_stable():
    root = K.fit_root(3, 17)
    five, three = K.chain_keys(root, 5), K.chain_keys(root, 3)
    assert five.shape == (5,) and three.shape == (3,)
    assert [kd(k) for k in three] == [kd(k) for k in five[:3]]
    assert [kd(k) for k in K.init_keys(3, 17, 2)] == [kd(k) for k in K.init_keys(3, 17, 4)[:2]]


def test_streams_are_separate():
    mcmc = K.chain_keys(K.fit_root(1, 2), 2)
    init = K.init_keys(1, 2, 2)
    other_salt = K.chain_keys(K.fit_root(1, 3), 2)
    all_keys = [kd(k) for k in (*mcmc, *init, *other_salt)]
    all_keys.append(kd(K.forecast_key(1, 0, 0, K.Ahead.VOL, 0)))
    all_keys.append(kd(K.validation_key(1, 0, 0, 0)))
    assert len({tuple(k) for k in all_keys}) == len(all_keys)


def test_c_it_block_i_determinism_vmap_jit_and_loop():
    root = K.fit_root(5, 6)
    C, n = 3, 4
    its = np.array([0, 1, 250, 2**32 - 1], dtype=np.uint32)
    blocks = (K.Block.EQ_MEAN, K.Block.EQ_VOL, K.Block.G_CCCM)

    def grid(ckeys, it, block):
        def one_chain(ck):
            sk = K.step_key(ck, it, block)
            return jax.vmap(lambda i: K.eq_key(sk, i))(jnp.arange(n, dtype=jnp.int32))

        return jax.vmap(one_chain)(ckeys)

    jitted = jax.jit(grid, static_argnums=2)
    ckeys = K.chain_keys(root, C)
    seen = set()
    for it, block in itertools.product(its, blocks):
        batched = random.key_data(jitted(ckeys, jnp.uint32(it), block))
        for c, i in itertools.product(range(C), range(n)):
            loop = kd(K.eq_key(K.step_key(K.chain_keys(root, C)[c], int(it), block), i))
            assert [int(x) for x in batched[c, i]] == loop
            seen.add(tuple(loop))
    assert len(seen) == C * n * len(its) * len(blocks)  # no collisions on the grid


def test_sweep_counter_must_be_uint32():
    ck = K.chain_keys(K.fit_root(0, 0), 1)[0]
    ref = kd(K.step_key(ck, 5, K.Block.EQ_VOL))
    assert kd(K.step_key(ck, np.uint32(5), K.Block.EQ_VOL)) == ref
    assert kd(K.step_key(ck, jnp.uint32(5), K.Block.EQ_VOL)) == ref
    # uint32 arithmetic stays uint32 under x64 (it0 + j·thin + r) ...
    it0, j, thin, r = jnp.uint32(2), jnp.uint32(1), jnp.uint32(3), jnp.int32(0).astype(jnp.uint32)
    it = it0 + j * thin + r
    assert it.dtype == jnp.uint32
    assert kd(K.step_key(ck, it, K.Block.EQ_VOL)) == ref
    # ... but mixing in a signed counter promotes to int64, which must fail loudly (E7)
    promoted = it0 + jnp.int32(3)
    assert promoted.dtype == jnp.int64
    for bad in (promoted, jnp.int32(5), np.int64(5), jnp.float64(5.0)):
        with pytest.raises(TypeError, match="uint32"):
            K.step_key(ck, bad, K.Block.EQ_VOL)
    for out_of_range in (-1, 2**32):
        with pytest.raises(ValueError):
            K.step_key(ck, out_of_range, K.Block.EQ_VOL)
    with pytest.raises(ValueError, match="scalar"):
        K.step_key(ck, jnp.zeros(2, jnp.uint32), K.Block.EQ_VOL)
    with pytest.raises(TypeError):
        jax.jit(lambda t: K.step_key(ck, t, K.Block.EQ_VOL))(jnp.int32(5))
    traced = jax.jit(lambda t: K.step_key(ck, t, K.Block.EQ_VOL))(jnp.uint32(5))
    assert kd(traced) == ref


def test_index_arguments_are_cast_to_uint32():
    sk = K.step_key(K.chain_keys(K.fit_root(0, 0), 1)[0], 0, K.Block.EQ_MEAN)
    ref = kd(K.eq_key(sk, 3))
    for i in (np.uint32(3), np.int64(3), jnp.int32(3), jnp.int64(3), jnp.uint32(3)):
        assert kd(K.eq_key(sk, i)) == ref
    assert kd(jax.jit(lambda i: K.eq_key(sk, i))(jnp.int32(3))) == ref
    with pytest.raises(TypeError):
        K.eq_key(sk, 3.0)
    with pytest.raises(ValueError):
        K.eq_key(sk, -1)
    with pytest.raises(ValueError, match="vmap"):
        K.forecast_key(1, jnp.arange(2), 0, K.Ahead.VOL, 0)


def test_enum_arguments_are_checked():
    ck = K.chain_keys(K.fit_root(0, 0), 1)[0]
    assert kd(K.step_key(ck, 0, 16)) == kd(K.step_key(ck, 0, K.Block.G_CCCM))
    with pytest.raises(TypeError, match="Block"):
        K.step_key(ck, 0, K.Sub.H)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        K.step_key(ck, 0, 5)  # not a Block value
    ek = K.eq_key(K.step_key(ck, 0, K.Block.EQ_VOL), 0)
    with pytest.raises(TypeError, match="Sub"):
        K.sub_key(ek, K.Block.EQ_VOL)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Ahead"):
        K.forecast_key(0, 0, 0, K.Sub.S, 0)  # type: ignore[arg-type]


def test_typed_keys_and_seed_range():
    with pytest.raises(TypeError, match="typed key"):
        K.chain_keys(random.PRNGKey(0), 2)
    with pytest.raises(TypeError, match="typed key"):
        K.step_key(random.PRNGKey(0), 0, K.Block.EQ_MEAN)
    for bad in (-1, 2**63):
        with pytest.raises(ValueError):
            K.fit_root(bad, 0)
    with pytest.raises(TypeError):
        K.fit_root(True, 0)
    with pytest.raises(ValueError):
        K.chain_keys(K.fit_root(0, 0), 0)
    assert kd(K.fit_root(2**63 - 1, 2**32 - 1)) == kd(K.fit_root(2**63 - 1, 2**32 - 1))
    with pytest.raises(ValueError):
        K.fit_root(0, 2**32)  # salts are uint32


@pytest.mark.skipif(not hasattr(jax, "debug_key_reuse"), reason="key-reuse checker unavailable")
def test_one_key_per_consumer_passes_the_reuse_checker():
    with jax.debug_key_reuse(True):

        def body(ck):
            ek = K.eq_key(K.step_key(ck, jnp.uint32(3), K.Block.EQ_VOL), jnp.int32(1))
            return jnp.stack([random.normal(K.sub_key(ek, s)) for s in K.Sub])

        out = jax.jit(jax.vmap(body))(K.chain_keys(K.fit_root(0, 0), 2))
        assert out.shape == (2, len(K.Sub)) and out.dtype == jnp.float64
        assert np.unique(np.asarray(out)).size == out.size


def test_prng_implementation_is_threefry():
    assert K.PRNG_IMPL == "threefry2x32"
    for name, k in derive_golden(K, np).items():
        assert str(random.key_impl(k)) == "threefry2x32", name
        assert k.dtype == random.key(0, impl="threefry2x32").dtype, name
    for impl in ("rbg", "unsafe_rbg"):  # D-16: forbidden, on the host and under jit
        other = random.key(0, impl=impl)
        with pytest.raises(TypeError, match=impl):
            K.chain_keys(other, 2)
        with pytest.raises(TypeError, match=impl):
            K.step_key(other, 0, K.Block.EQ_MEAN)
        with pytest.raises(TypeError, match=impl):
            jax.jit(lambda k: K.sub_key(K.eq_key(k, 0), K.Sub.H))(other)


def test_default_prng_setting_cannot_switch_the_engine_to_rbg():
    # JAX_DEFAULT_PRNG_IMPL (and a later jax.config update) must not change any stream (D-16).
    code = textwrap.dedent(inspect.getsource(derive_golden)) + textwrap.dedent(
        """
        import json
        import jax
        import numpy as np
        from jax import random
        from rai_bvar.engines.mcmc import keys as K

        assert str(random.key_impl(random.key(0))) == "rbg", "the environment default is unused"
        out = {}
        for default in ("rbg", "unsafe_rbg"):
            jax.config.update("jax_default_prng_impl", default)
            keys = derive_golden(K, np)
            impls = {str(random.key_impl(k)) for k in keys.values()}
            assert impls == {"threefry2x32"}, (default, impls)
            for name, k in keys.items():
                out.setdefault(name, []).append([int(x) for x in np.asarray(random.key_data(k))])
            try:
                K.step_key(random.key(1), 0, K.Block.EQ_VOL)  # a default-impl key
            except TypeError as e:
                assert default in str(e), e
            else:
                raise AssertionError(f"a {default} key was accepted")
        print(json.dumps(out))
        """
    )
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_", "XLA_"))}
    environ.update(JAX_PLATFORMS="cpu", JAX_DEFAULT_PRNG_IMPL="rbg")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environ, timeout=300
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    here = {name: kd(k) for name, k in derive_golden(K, np).items()}
    assert got == {name: [v, v] for name, v in here.items()}
    if jax.__version__ == PINNED_JAX:
        assert here == GOLDEN
