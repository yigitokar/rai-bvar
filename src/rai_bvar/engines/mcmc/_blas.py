"""One BLAS thread while the MCMC executables run (DESIGN §10.4 lever 8 class; WP24 finding).

jaxlib lowers ``cholesky`` and ``triangular_solve`` on CPU to LAPACK/BLAS custom calls into the
library SciPy ships (OpenBLAS in the Linux and Windows wheels; Accelerate on macOS). XLA already
runs those calls on its own worker threads, and OpenBLAS additionally threads a level-3 call over
its right-hand-side columns, with busy-waiting worker threads. On a machine whose cores are all
in use (XLA's pool, or any other load) the hand-off costs whole scheduler time slices: in the
BENCH-P2 ``sample`` executable the monitor's ``B = (A⁻¹Γ)ᵀ`` (four ``20 × 20`` solves with 81
right-hand sides per kept draw) took 9 ms per draw on an idle 4-core x86 machine and 100 ms per
draw at load 12, against 0.7 ms single-threaded, while the sweep itself costs ≈ 16 ms.

:func:`single_thread` sets the OpenBLAS thread count to 1 for the duration of a run and restores
it afterwards. It changes no result: a column of a triangular solve and a ``≤ 100 × 100``
Cholesky are computed by the same kernel sequence whatever the thread count (checked bitwise on
BENCH-P2-shaped fits: every monitor value, stored draw and the final state; and by the engine's
bitwise-invariance tests). Where no OpenBLAS with a thread-count API is loaded (macOS
Accelerate, MKL, a NumPy-only process) it is a no-op. Re-entrant and thread-safe: the outermost
context sets and restores the count.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import os
import threading
from collections.abc import Iterator
from typing import Any

__all__ = ["openblas", "single_thread"]

_SYMBOLS = (  # (set, get): SciPy's bundled build first, then a system OpenBLAS
    ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
    ("openblas_set_num_threads", "openblas_get_num_threads"),
)
_lock = threading.Lock()
_depth = 0
_saved: int | None = None
_handle: tuple[Any, Any] | bool | None = False  # False: not looked up yet


def _scipy_libraries() -> list[str]:
    """Shared libraries SciPy's LAPACK may come from: its bundled ``scipy.libs`` OpenBLAS and,
    as a fallback, any OpenBLAS already mapped into this process (Linux)."""
    paths: list[str] = []
    try:
        import scipy

        root = os.path.dirname(os.path.dirname(scipy.__file__))
        paths += sorted(glob.glob(os.path.join(root, "scipy.libs", "*openblas*")))
    except ImportError:
        pass
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.rsplit(" ", 1)[-1].strip()
                if "openblas" in os.path.basename(path).lower() and path not in paths:
                    paths.append(path)
    except OSError:
        pass
    return paths


def openblas() -> tuple[Any, Any] | None:
    """``(set_num_threads, get_num_threads)`` of the OpenBLAS jaxlib's LAPACK calls use, or
    ``None``. Looked up once (``scipy.linalg``, which jaxlib's LAPACK kernels are initialised
    from, is imported first so the library is loaded)."""
    global _handle
    if _handle is not False:
        return _handle  # type: ignore[return-value]
    found = None
    try:
        import scipy.linalg  # noqa: F401  (loads the BLAS jaxlib's LAPACK kernels use)
    except ImportError:
        pass
    for path in _scipy_libraries():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for set_name, get_name in _SYMBOLS:
            setter, getter = getattr(lib, set_name, None), getattr(lib, get_name, None)
            if setter is not None and getter is not None:
                setter.argtypes, setter.restype = [ctypes.c_int], None
                getter.argtypes, getter.restype = [], ctypes.c_int
                found = (setter, getter)
                break
        if found is not None:
            break
    _handle = found
    return found


@contextlib.contextmanager
def single_thread() -> Iterator[None]:
    """Run the enclosed block with one OpenBLAS thread (module docstring)."""
    global _depth, _saved
    api = openblas()
    if api is None:
        yield
        return
    setter, getter = api
    with _lock:
        if _depth == 0:
            _saved = int(getter())
            if _saved != 1:
                setter(1)
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _saved is not None and _saved != 1:
                setter(_saved)
