"""float64 for the MCMC engine (SPEC §8, R4; DESIGN §1.1 rule 2; N07 §3).

Every JAX-using engine module imports this module first. Importing it enables
``jax_enable_x64`` process-wide, once (a documented global side effect, SPEC §8); every public
engine entry point additionally runs inside :func:`x64`, so float64 holds even if the user turns
the global flag off after import. With x64 off JAX does not raise on a requested float64: it
warns ("Explicitly requested dtype float64 ... will be truncated") and computes in float32, which
is why the engine's test modules turn that warning into an error.

Nothing here creates a ``jnp`` array: a module-level array would initialise the backend and make
``rai_bvar.set_cpu_devices`` impossible (DESIGN §1.1 rule 3).
"""

from __future__ import annotations

import contextlib

import jax

jax.config.update("jax_enable_x64", True)


def x64() -> contextlib.AbstractContextManager[None]:
    """Context manager enforcing float64 (``jax.enable_x64(True)``) for the enclosed block."""
    return jax.enable_x64(True)


def x64_enabled() -> bool:
    """Whether float64 is in effect here (the global flag or an enclosing :func:`x64`)."""
    return bool(jax.config.jax_enable_x64)
