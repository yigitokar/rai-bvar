"""No recompilation across windows of one ``T_pad`` bucket (D-23; DESIGN §4.9, N07 §7.7)."""

from __future__ import annotations

import logging

import jax
import pytest

from rai_bvar.engines.mcmc.runner import make_runner, run_chains
from rai_bvar.settings import MCMCSettings, Runtime

from . import toys

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")


class _Compiles:
    """Backend compilations of the runner's executables (``jax.monitoring`` events)."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def __call__(self, event: str, duration: float, **kw) -> None:
        if event.endswith("backend_compile_duration"):
            self.names.append(str(kw.get("fun_name", "")))

    @property
    def runner(self) -> list[str]:
        return [n for n in self.names if "mcmc_" in n]


def test_zero_recompiles_across_windows_of_one_bucket(caplog):
    static = toys.toy_static(
        "linear_structural", "sv", transitory="outlier", n=2, T_eff=20, chunk=5
    )
    fns = make_runner(static, overrides=toys.toy_overrides(static))  # fresh executables
    prior = toys.toy_prior(static)
    counter = _Compiles()
    jax.monitoring.register_event_duration_secs_listener(counter)
    try:
        run_chains(fns, static, toys.toy_data(static, 20, 1), prior,
                   MCMCSettings(chains=2, draws=7, burn=4, thin=1), Runtime(chunk=5), seed=1, salt=1)  # fmt: skip
        first = list(counter.runner)
        assert {
            "jit(mcmc_prepare)",
            "jit(mcmc_init)",
            "jit(mcmc_burn)",
            "jit(mcmc_sample_chunk)",
        } <= set(first)
        counter.names.clear()
        jax.config.update("jax_log_compiles", True)
        with caplog.at_level(logging.WARNING, logger="jax._src.interpreters.pxla"):
            # another window of the same bucket (T_eff 27 ≤ 32), new data, burn, thin, draws, seeds
            run_chains(fns, static, toys.toy_data(static, 27, 2), prior,
                       MCMCSettings(chains=2, draws=12, burn=9, thin=3), Runtime(chunk=5), seed=2, salt=8)  # fmt: skip
    finally:
        jax.config.update("jax_log_compiles", False)
        jax.monitoring.unregister_event_duration_listener(counter)
    assert counter.runner == []
    logged = [
        r.getMessage()
        for r in caplog.records
        if "Compiling" in r.getMessage() and "mcmc_" in r.getMessage()
    ]
    assert logged == []
