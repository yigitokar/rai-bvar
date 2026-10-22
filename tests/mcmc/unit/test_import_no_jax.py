"""Import-time behaviour (DESIGN §1.1, D-31): no JAX for ``import rai_bvar`` and the pure-Python
engine modules, no backend initialisation on importing the JAX modules, and enforced float64.
Each check runs in a fresh interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

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


def test_pure_python_imports_load_no_jax():
    # One fresh interpreter, checked after every step so a failure names the culprit.
    out = run_python(
        """
        import sys

        def check(step):
            loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("jax", "jaxlib"))
            assert not loaded, f"{step} imported {loaded[:5]}"

        import rai_bvar
        check("import rai_bvar")
        import rai_bvar.engines.mcmc as m
        check("import rai_bvar.engines.mcmc")
        assert set(m.__all__) == {"fit_mcmc", "MCMCFit", "MCMCNumericalError"}
        assert "fit_mcmc" in dir(m)
        from rai_bvar.engines.mcmc import config
        check("import rai_bvar.engines.mcmc.config")
        from rai_bvar.engines.mcmc import registry
        check("import rai_bvar.engines.mcmc.registry")
        from rai_bvar import _devices
        check("import rai_bvar._devices")
        st = config.StaticConfig(
            config.Shapes.of(3, 2, 1, config.bucket(50, 32)),
            config.StructStatic(contemp_scaling="lambda"),
            config.MeanStatic("linear_structural", coef_scaling="lambda", const_path="fixed_factor"),
            config.CoefPriorStatic(),
            config.VolStatic("constant", coupled=True),
        )
        config.validate_composition(st.struct, st.mean, st.coef, st.vol)
        assert registry.slot_kinds(st)["core"] == "constant"
        check("building and validating a StaticConfig")
        print("ok")
        """
    )
    assert out.strip() == "ok"


def test_engine_modules_leave_the_backend_uninitialised():
    # A module-level jnp array would initialise the backend and make set_cpu_devices impossible.
    out = run_python(
        """
        from rai_bvar.engines.mcmc import _jax, config, keys, protocols, registry, state
        from rai_bvar._devices import set_cpu_devices
        set_cpu_devices(2)
        import jax
        assert jax.local_device_count() == 2, jax.devices()
        assert keys.chain_keys(keys.fit_root(0, 1), 2).shape == (2,)
        print("ok")
        """
    )
    assert out.strip() == "ok"


def test_float64_is_enforced_even_with_jax_enable_x64_off():
    out = run_python(
        """
        import warnings
        import jax
        assert not jax.config.jax_enable_x64          # JAX_ENABLE_X64=0 in the environment
        import jax.numpy as jnp
        from rai_bvar.engines.mcmc import _jax, keys, state
        assert jax.config.jax_enable_x64 and _jax.x64_enabled()
        assert jnp.zeros(1).dtype == jnp.float64
        jax.config.update("jax_enable_x64", False)    # the user turns it off after import
        assert jnp.zeros(1).dtype == jnp.float32
        with warnings.catch_warnings():
            warnings.simplefilter("error")             # truncation warnings must not occur
            with _jax.x64():
                assert _jax.x64_enabled()
                ck = keys.chain_keys(keys.fit_root(1, 2), 2)[0]
                ek = keys.eq_key(keys.step_key(ck, 3, keys.Block.EQ_VOL), 0)
                x = jax.jit(lambda k: jax.random.normal(k, (3,)))(keys.sub_key(ek, keys.Sub.H))
                assert x.dtype == jnp.float64
                st = state.Status.initial()
                assert st.min_pivot.dtype == jnp.float64 and st.it.dtype == jnp.uint32
                state.assert_float64(st)
                assert state.encode_error(keys.Block.EQ_VOL, jnp.int32(3)).dtype == jnp.int32
        assert not _jax.x64_enabled()
        print("ok")
        """,
        JAX_ENABLE_X64="0",
    )
    assert out.strip() == "ok"
