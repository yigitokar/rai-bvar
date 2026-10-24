"""Analytic forecasts record the variable ordering (SPEC R8, §7 attrs; audit P1 item 10).

The MCMC engine writes ``attrs["ordering"]`` (``engines/mcmc/forecast.py`` ``_attrs``; asserted in
``tests/contract/test_forecast_contract_mcmc.py``) and the analytic IRF writes it too
(``structural/irf.py``). The analytic forecast now carries the same attribute in the same format:
the JSON list of the data's column order, which is the recursive (Cholesky) ordering.
"""

from __future__ import annotations

import json

import pytest

from rai_bvar import BVAR, Linear, Minnesota
from rai_bvar.data import VARData


@pytest.mark.parametrize("hierarchy", ["fixed", "eb"])
def test_analytic_forecast_records_the_ordering(sim_data: VARData, hierarchy: str):
    fit = BVAR(mean=Linear(prior=Minnesota(hierarchy=hierarchy)), lags=1).fit(sim_data)
    attrs = fit.forecast(horizon=2, draws=20, seed=0).dataset.attrs
    assert attrs["engine"] == "analytic"
    assert attrs["ordering"] == json.dumps(list(sim_data.variables))
    assert attrs["ordering"] == '["gdp", "prices", "rate"]'  # the MCMC contract test's string


def test_analytic_ordering_follows_the_column_order(sim_raw):
    order = ["rate", "gdp", "prices"]
    data = VARData.from_frame(sim_raw[order], {"gdp": 5, "prices": 5, "rate": 1})
    fit = BVAR(mean=Linear(prior=Minnesota(hierarchy="fixed")), lags=1).fit(data)
    attrs = fit.forecast(horizon=1, draws=10, seed=0).dataset.attrs
    assert json.loads(attrs["ordering"]) == order
