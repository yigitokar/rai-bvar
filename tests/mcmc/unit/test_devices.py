"""``set_cpu_devices`` and ``enable_compilation_cache`` (DESIGN §1.1 rule 4, §4.9, §5.2).

Process-level JAX settings: every check that touches JAX runs in a fresh interpreter, so the
test process's own backend and cache configuration are never changed."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import pytest

from rai_bvar._devices import enable_compilation_cache, set_cpu_devices

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


def run_python(code: str, **env: str) -> str:
    environ = {
        k: v
        for k, v in os.environ.items()
        if k not in {"JAX_ENABLE_X64", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES"}
    }
    environ.update({"JAX_PLATFORMS": "cpu", **env})
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=environ,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.mark.parametrize("k", [0, -1, 2.0, True, "4"])
def test_set_cpu_devices_validates_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        set_cpu_devices(k)  # type: ignore[arg-type]


@pytest.mark.parametrize("secs", [-1.0, math.nan, math.inf])
def test_enable_compilation_cache_validates_threshold(secs, tmp_path):
    with pytest.raises(ValueError, match="min_compile_secs"):
        enable_compilation_cache(tmp_path / "cache", secs)
    assert not (tmp_path / "cache").exists()


def test_set_cpu_devices_before_backend_initialisation():
    out = run_python(
        """
        import rai_bvar
        from rai_bvar._devices import set_cpu_devices
        set_cpu_devices(3)
        import jax, jax.numpy as jnp
        assert jax.local_device_count() == 3 and len(jax.devices("cpu")) == 3
        jnp.zeros(3).block_until_ready()
        set_cpu_devices(3)                       # same count after initialisation: a no-op
        try:
            set_cpu_devices(2)
        except RuntimeError as err:
            assert "before JAX initialises" in str(err) and "3 CPU device" in str(err), err
        else:
            raise AssertionError("set_cpu_devices(2) after initialisation did not raise")
        print("ok")
        """
    )
    assert out.strip() == "ok"


def test_set_cpu_devices_raises_after_backend_initialisation():
    out = run_python(
        """
        import jax.numpy as jnp
        from rai_bvar._devices import set_cpu_devices
        jnp.ones(2).sum().block_until_ready()    # the user touched JAX first
        set_cpu_devices(1)                       # already true: no-op
        try:
            set_cpu_devices(4)
        except RuntimeError as err:
            assert "right after `import rai_bvar`" in str(err), err
        else:
            raise AssertionError("set_cpu_devices(4) did not raise")
        import jax
        assert jax.local_device_count() == 1
        print("ok")
        """
    )
    assert out.strip() == "ok"


def test_enable_compilation_cache_writes_entries(tmp_path):
    first, second = tmp_path / "cache_a", tmp_path / "nested" / "cache_b"
    out = run_python(
        f"""
        import os
        import jax, jax.numpy as jnp
        from rai_bvar._devices import enable_compilation_cache
        enable_compilation_cache({str(first)!r}, min_compile_secs=0.0)
        assert jax.config.jax_compilation_cache_dir == {str(first.resolve())!r}
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
        jax.jit(lambda x: jnp.sin(x) * 2.0)(jnp.arange(3.0)).block_until_ready()
        assert os.listdir({str(first)!r}), "no cache entry written"
        enable_compilation_cache({str(second)!r}, min_compile_secs=0)   # switching directories
        jax.jit(lambda x: jnp.cos(x) + 1.0)(jnp.arange(4.0)).block_until_ready()
        assert os.listdir({str(second)!r}), "no cache entry written after switching"
        print("ok")
        """
    )
    assert out.strip() == "ok"
