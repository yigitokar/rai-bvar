"""Fixtures of the host prior-builder tests (WP6; DESIGN §13)."""

from __future__ import annotations

import pytest

from rai_bvar.data import VARData

from .helpers import make_data


@pytest.fixture(scope="module")
def data4() -> VARData:
    """Quarterly, four variables (tcodes 1, 5, 2, 4), 119 transformed observations."""
    return make_data(4, 120, "Q", seed=1)


@pytest.fixture(scope="module")
def data4_monthly() -> VARData:
    """Monthly, four variables, 199 transformed observations."""
    return make_data(4, 200, "M", seed=2)
