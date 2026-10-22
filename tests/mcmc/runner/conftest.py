"""Fixtures of the runner tests: monitor doubles (WP15 stand-ins) and cache hygiene."""

from __future__ import annotations

import functools

import pytest

from rai_bvar.engines.mcmc import runner

from . import toys


@pytest.fixture(autouse=True)
def monitor_doubles(monkeypatch):
    """Every runner test records monitors through the doubles of ``toys`` (DESIGN §8 lives in
    WP15); ``raising=False`` because ``monitor.py`` may still be a stub."""
    toys.install_monitor_doubles(functools.partial(monkeypatch.setattr, raising=False))


@pytest.fixture
def toy_registry(monkeypatch):
    """``registry.load`` returns the toy classes: exercises the production (cached, override-free)
    composition path; the caches are cleared before and after so no toy program survives."""
    from rai_bvar.engines.mcmc import registry

    runner.clear_caches()
    monkeypatch.setattr(registry, "load", lambda kind: toys.TOY_KINDS[kind])
    yield
    runner.clear_caches()
