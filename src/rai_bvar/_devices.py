"""Process-level JAX settings that must be chosen early (DESIGN §1.1 rule 4, §4.9, §5.2; N07 §6–7).

JAX is imported only inside the functions, so ``import rai_bvar`` stays JAX-free and
:func:`set_cpu_devices` can run before anything initialises the backend.
"""

from __future__ import annotations

import math
import numbers
import os
from pathlib import Path


def set_cpu_devices(k: int) -> None:
    """Expose ``k`` CPU devices so MCMC chains can run one per device (``shard_map``).

    Must run before JAX initialises its backend, i.e. right after ``import rai_bvar`` and before
    any fit or JAX computation. Afterwards it raises ``RuntimeError`` unless the backend already
    has exactly ``k`` CPU devices (then it is a no-op). Draws do not depend on the device count
    (DESIGN §5.2).
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    import jax

    try:
        jax.config.update("jax_num_cpu_devices", k)
    except RuntimeError as err:
        if len(jax.devices("cpu")) == k:
            return
        raise RuntimeError(
            f"rai_bvar.set_cpu_devices({k}) must run before JAX initialises its backend, i.e. "
            "right after `import rai_bvar` and before any fit or JAX computation; this process "
            f"already runs on {len(jax.devices('cpu'))} CPU device(s). Restart the process and "
            "call it first (MCMC fits still run, on the existing devices)."
        ) from err


def enable_compilation_cache(path: str | os.PathLike[str], min_compile_secs: float = 1.0) -> None:
    """Opt in to JAX's persistent compilation cache in ``path`` (created if missing).

    Executables that took at least ``min_compile_secs`` to compile are stored and reused by later
    processes (a full sweep: about 1.5–2.8 s cold, 0.37 s cached; N07 §7.6). CI uses
    ``min_compile_secs=0`` so that small validation programs are cached too. The cache is
    trusted: use a directory that only you can write to.
    """
    secs = float(min_compile_secs)
    if not math.isfinite(secs) or secs < 0:
        raise ValueError(f"min_compile_secs must be finite and >= 0, got {min_compile_secs!r}")
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    previous = jax.config.jax_compilation_cache_dir
    if previous and Path(previous).expanduser().resolve() != directory:
        compilation_cache.reset_cache()  # an initialised cache keeps its old directory otherwise
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", secs)
