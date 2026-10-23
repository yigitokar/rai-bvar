"""``engines.mcmc._blas.single_thread``: one OpenBLAS thread while the chains run (WP24).

The lever changes no draw (the chunking, device and fused/unfused bitwise tests run with it
active, through ``run_chains``); here: the count is 1 inside, restored outside, re-entrant, and
the context is a no-op where no OpenBLAS thread API is found (macOS Accelerate).
"""

from __future__ import annotations

import threading

import pytest

from rai_bvar.engines.mcmc import _blas


def test_single_thread_sets_and_restores():
    api = _blas.openblas()
    if api is None:
        with _blas.single_thread():  # no OpenBLAS: a no-op
            pass
        pytest.skip("no OpenBLAS thread API in this process (e.g. macOS Accelerate)")
    setter, getter = api
    before = getter()
    setter(2)
    try:
        with _blas.single_thread():
            assert getter() == 1
            with _blas.single_thread():  # re-entrant: the inner exit does not restore
                assert getter() == 1
            assert getter() == 1
        assert getter() == 2
    finally:
        setter(before)


def test_single_thread_restores_on_error():
    api = _blas.openblas()
    if api is None:
        pytest.skip("no OpenBLAS thread API in this process")
    setter, getter = api
    before = getter()
    setter(2)
    try:
        with pytest.raises(RuntimeError), _blas.single_thread():
            raise RuntimeError
        assert getter() == 2
    finally:
        setter(before)


def test_single_thread_concurrent_contexts():
    """Two threads inside at once: the count is restored only after the last one leaves."""
    api = _blas.openblas()
    if api is None:
        pytest.skip("no OpenBLAS thread API in this process")
    setter, getter = api
    before = getter()
    setter(2)
    inside, release = threading.Barrier(2), threading.Event()
    seen = []

    def worker():
        with _blas.single_thread():
            inside.wait()
            release.wait()
            seen.append(getter())

    try:
        t = threading.Thread(target=worker)
        t.start()
        with _blas.single_thread():
            inside.wait()
            release.set()
            t.join()
            assert getter() == 1
        assert seen == [1] and getter() == 2
    finally:
        setter(before)
