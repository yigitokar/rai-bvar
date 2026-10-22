"""PRNG key derivation (DESIGN §5.6, D-16; N07 §4).

Typed threefry2x32 keys and a ``fold_in`` chain, so every draw is a pure function of
``(seed, data salt, chain, absolute sweep, block, equation, sub-draw)``::

    root      = key(seed)
    fit_root  = fold_in(fold_in(root, Stream.MCMC), salt)          salt = crc32(data fingerprint)
    chain_key = fold_in(fit_root, c)                                c = 0..C−1 (prefix-stable)
    step_key  = fold_in(fold_in(chain_key, it), Block.X)           it: absolute sweep, burn-in included
    eq_key    = fold_in(step_key, i)                                i: position in the ordering
    sub_key   = fold_in(eq_key, Sub.Y)
    init      : fold_in(fold_in(fold_in(root, Stream.INIT), salt), c) → Block → i → Sub
    forecast  : fold_in(fold_in(key(fseed), Stream.FORECAST), c) → d → Ahead.X → i
                                                                    d: kept-draw index
    validation: fold_in(key(seed), Stream.VALIDATION) → tier → composition → chain or replication

Keys therefore do not depend on chunk size, burn piece size, thinning alignment, chain-count
prefixes or vmap versus loops (N07 §4.1): draws are bitwise independent of the first three, and
equal across chain-count prefixes and ``vmap`` vs ``shard_map`` to rounding (XLA's batch-size
dependent vectorisation; DESIGN §5.6, Stage 2 amendment). No key is ever carried in a loop state.

The implementation is pinned to :data:`PRNG_IMPL` (D-16): roots are built with
``impl="threefry2x32"`` whatever ``jax_default_prng_impl`` (or ``JAX_DEFAULT_PRNG_IMPL``) says,
and every function that takes a key rejects other implementations (``rbg``/``unsafe_rbg``).

The enums are **append-only**: renumbering a member changes every stream. Counters fed to
``fold_in`` are uint32: the absolute sweep index must arrive as a uint32 array (an int64 result
of ``it0 + j·thin + r`` raises here instead of wrapping; E7), and index-like arguments are cast
explicitly. Python ints are range-checked on the host.
"""

from __future__ import annotations

import numbers
import zlib
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (enables float64 before any computation)

Key = jax.Array  # typed key array (dtype key<fry>), from jax.random.key

PRNG_IMPL = "threefry2x32"  # D-16, §5.6: the only implementation the engine accepts

_U32 = 2**32


class Stream(IntEnum):
    """Top-level streams: adding forecast or validation draws never perturbs posterior draws."""

    MCMC = 0
    INIT = 1
    FORECAST = 2
    VALIDATION = 3
    IRF = 4


class Block(IntEnum):
    """Sweep steps (key derivation and error codes). 17–31: Phase 3 global steps; 32–63: Phases 4–5."""

    EQ_MEAN = 0
    EQ_CONTEMP = 1
    EQ_VOL = 2
    EQ_COEF_LOCAL = 3
    G_CCCM = 16
    G_COEF_GLOBAL = 17
    G_COMMON_SV = 18
    G_LP_SCALE = 19
    G_REGIME = 32
    G_FACTORS = 33
    G_LOADINGS = 34
    G_FACTOR_SV = 35
    G_B0_ROWS = 36


class Sub(IntEnum):
    """Sub-draws within one equation's step. Vol block: S … TAU; mean, contemp, CCCM: Z."""

    S = 0
    H = 1
    H_ACC = 2
    C_GP = 3
    C_GP_ACC = 4
    C_S2 = 5
    C_S2_ACC = 6
    NC_PHI = 7
    NC_PHI_ACC = 8
    NC_MS = 9
    NC_MS_ACC = 10
    O = 11  # noqa: E741
    P = 12
    NU = 13
    TAU = 14
    Z = 0  # alias of S: the mean, contemp and CCCM blocks use only this one (CCCM folds j first)


class Ahead(IntEnum):
    """Forecast randomness per selected kept draw (DESIGN §5.6, §7.2)."""

    VOL = 0
    SHOCK = 1
    MEAN = 2
    GLOBAL = 3


class Tier(IntEnum):
    """Validation tiers (DESIGN §9) for :func:`validation_key`."""

    U = 0
    B = 1
    G = 2
    S = 3
    M = 4
    A = 5
    P = 6


def _host_u32(value: int, name: str) -> np.uint32:
    if not 0 <= value < _U32:
        raise ValueError(f"{name} must lie in [0, 2**32), got {value}")
    return np.uint32(value)


def _counter(it: object) -> object:
    """The absolute sweep index: a uint32 scalar array, or a Python int in [0, 2**32)."""
    if isinstance(it, numbers.Integral) and not isinstance(it, bool | np.generic):
        return _host_u32(int(it), "it")
    dtype = getattr(it, "dtype", None)
    if dtype != np.uint32:
        raise TypeError(
            f"the sweep counter must be uint32 (got dtype {dtype}); compute it from uint32 "
            "operands and cast explicitly (DESIGN §5.6, E7)"
        )
    if np.ndim(it) != 0:
        raise ValueError("the sweep counter must be a scalar (vmap over chains, not over it)")
    return it


def _index(value: object, name: str) -> object:
    """An index (chain, equation, draw, replication) as uint32; integer dtypes are cast."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool | np.generic):
        return _host_u32(int(value), name)
    dtype = getattr(value, "dtype", None)
    if dtype is None or not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"{name} must be an integer scalar, got {value!r}")
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches")
    if isinstance(value, np.generic | np.ndarray):
        return _host_u32(int(value), name)
    return value.astype(jnp.uint32)


def _member(value: object, enum: type[IntEnum]) -> int:
    if isinstance(value, IntEnum) and not isinstance(value, enum):
        raise TypeError(
            f"expected a {enum.__name__} member, got {type(value).__name__}.{value.name}"
        )
    return int(enum(value))


def _typed(key: object, name: str = "key") -> Key:
    """``key`` itself, after checking it is a typed threefry2x32 key (traced keys included)."""
    dtype = getattr(key, "dtype", None)
    if dtype is None or not jnp.issubdtype(dtype, jax.dtypes.prng_key):
        raise TypeError(f"{name} must be a typed key from jax.random.key (not a raw uint32 key)")
    impl = str(random.key_impl(key))
    if impl != PRNG_IMPL:
        raise TypeError(
            f"{name} uses the {impl!r} PRNG implementation; the engine requires {PRNG_IMPL!r} "
            "(D-16: rbg and unsafe_rbg are forbidden)"
        )
    return key  # type: ignore[return-value]


def _root(seed: int) -> Key:
    """``key(seed)`` with the implementation pinned, so ``jax_default_prng_impl`` cannot change
    any stream."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    if not 0 <= int(seed) < 2**63:
        raise ValueError(f"seed must lie in [0, 2**63), got {seed}")
    return random.key(int(seed), impl=PRNG_IMPL)


def data_salt(fingerprint: str) -> int:
    """``crc32(fingerprint)`` as a uint32 value: keeps windows and orderings on distinct streams."""
    if not isinstance(fingerprint, str):
        raise TypeError("fingerprint must be a str (VARData.fingerprint())")
    return zlib.crc32(fingerprint.encode("utf-8")) & 0xFFFFFFFF


def fit_root(seed: int, salt: int) -> Key:
    """``fold_in(fold_in(key(seed), Stream.MCMC), salt)``."""
    return random.fold_in(random.fold_in(_root(seed), int(Stream.MCMC)), _index(salt, "salt"))


def chain_keys(root: Key, C: int) -> Key:
    """``fold_in(root, c)`` for ``c = 0..C−1``; shape ``(C,)``. The first ``m`` keys do not depend
    on ``C`` (chain prefixes; the draws then agree across ``C`` to rounding, DESIGN §5.6)."""
    _typed(root, "root")
    if np.ndim(root) != 0:
        raise ValueError("root must be a scalar key")
    if isinstance(C, bool) or not isinstance(C, numbers.Integral) or C < 1:
        raise ValueError(f"C must be a positive integer, got {C!r}")
    return jax.vmap(lambda c: random.fold_in(root, c))(jnp.arange(int(C), dtype=jnp.uint32))


def init_keys(seed: int, salt: int, C: int) -> Key:
    """Chain keys of the initialisation stream; shape ``(C,)``. Derive ``Block → i → Sub`` below."""
    base = random.fold_in(random.fold_in(_root(seed), int(Stream.INIT)), _index(salt, "salt"))
    return chain_keys(base, C)


def block_key(key: Key, block: Block) -> Key:
    """``fold_in(key, block)``. Also the entry point of the init stream (§5.6:
    ``init_key → Block → i → Sub``, no sweep index)."""
    return random.fold_in(_typed(key, "key"), _member(block, Block))


def step_key(chain_key: Key, it: object, block: Block) -> Key:
    """``fold_in(fold_in(chain_key, it), block)``; ``it`` is the uint32 absolute sweep index."""
    _typed(chain_key, "chain_key")
    return block_key(random.fold_in(chain_key, _counter(it)), block)


def eq_key(step_key: Key, i: object) -> Key:
    """``fold_in(step_key, i)`` with ``i`` the equation's position in the ordering."""
    return random.fold_in(_typed(step_key, "step_key"), _index(i, "i"))


def sub_key(eq_key: Key, sub: Sub) -> Key:
    """``fold_in(eq_key, sub)``: one key per consumer (DESIGN §3.1 rule 5)."""
    return random.fold_in(_typed(eq_key, "eq_key"), _member(sub, Sub))


def forecast_key(fseed: int, c: object, d: object, kind: Ahead, i: object) -> Key:
    """Forecast randomness of chain ``c``, kept draw ``d``, kind ``Ahead.X``, equation ``i``
    (``i = 0`` for draw-level randomness). Scalars only; vmap over ``(c, d, i)`` for batches.

    ``d`` is the **kept-draw index** (DESIGN §5.6, §7.1; Stage 2 amendment): the ``draw``
    coordinate of ``fit.posterior``, not the draw's position in the store, so in-loop (§5.8) and
    post-hoc forecasts share keys whatever ``store_thin`` is."""
    k = random.fold_in(random.fold_in(_root(fseed), int(Stream.FORECAST)), _index(c, "c"))
    k = random.fold_in(random.fold_in(k, _index(d, "d")), _member(kind, Ahead))
    return random.fold_in(k, _index(i, "i"))


def validation_key(seed: int, tier: int, composition: int, rep: object) -> Key:
    """``fold_in(key(seed), Stream.VALIDATION) → tier → composition id → chain or replication``."""
    k = random.fold_in(_root(seed), int(Stream.VALIDATION))
    k = random.fold_in(random.fold_in(k, _index(tier, "tier")), _index(composition, "composition"))
    return random.fold_in(k, _index(rep, "rep"))
