"""Shared fixtures: deterministic simulated data from a known stationary VAR(1).

The transformed series ``y_t = (gdp growth, inflation, rate)`` follow

    y_t = c + A y_{t-1} + e_t,   e_t ~ N(0, Σ),

with the parameters below. ``sim_data`` stores *raw levels*: the first two variables are
log-level series integrated from the growth rates (tcode 5, Δ log x) and the rate is kept as
is (tcode 1), so its transformed sample reproduces ``y_t`` exactly (up to rounding).
``sim_data_stationary`` stores ``y_t`` directly with tcode 1 for every variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from rai_bvar.data import VARData

SIM_SEED = 20261021
SIM_T = 240  # transformed observations, 1960Q1..2019Q4 (stays clear of the 2020 outliers)
SIM_START = pd.Period("1960Q1", freq="Q")
SIM_VARIABLES = ("gdp", "prices", "rate")
SIM_TCODES = {"gdp": 5, "prices": 5, "rate": 1}

TRUE_MEAN = np.array([0.0075, 0.006, 4.0])  # unconditional mean of y_t
TRUE_A = np.array(
    [
        [0.4, 0.0, -0.001],
        [0.1, 0.6, 0.0],
        [15.0, 25.0, 0.85],
    ]
)
TRUE_CONST = (np.eye(3) - TRUE_A) @ TRUE_MEAN
_SD = np.array([0.007, 0.003, 0.25])
_CORR = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.2], [0.3, 0.2, 1.0]])
TRUE_SIGMA = _CORR * np.outer(_SD, _SD)
INITIAL_LEVELS = {"gdp": 100.0, "prices": 50.0}  # raw levels one period before the sample


def pytest_collection_modifyitems(config, items):
    """Plain ``pytest`` is the fast suite: ``slow`` and ``network`` tests run only when selected
    with a marker expression, e.g. ``pytest -m "slow or network"`` (the nightly suite)."""
    if config.getoption("markexpr"):
        return
    for item in items:
        for name in ("slow", "network"):
            if item.get_closest_marker(name) is not None:
                item.add_marker(pytest.mark.skip(reason=f"{name} test: select with -m {name}"))


#: the Phase 3 validation records (``record_property("p3_record", json)``, P3 DESIGN §9.4–§9.6;
#: WP35) of this session: each is written to the terminal as soon as its test ends (so a job cut
#: short by its timeout still logs every finished test) and again in the terminal summary, so the
#: job log carries the evidence beside the validation-record artifact
_P3_RECORDS: list[tuple[str, str]] = []
_CONFIG: list[pytest.Config] = []


def pytest_configure(config):
    _CONFIG[:] = [config]


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    new = [(report.nodeid, v) for k, v in report.user_properties if k == "p3_record"]
    _P3_RECORDS.extend(new)
    reporter = _CONFIG[0].pluginmanager.get_plugin("terminalreporter") if _CONFIG else None
    if reporter is not None:
        for nodeid, value in new:
            reporter.write_line(f"\n[p3_record] {nodeid} {value}")


def pytest_terminal_summary(terminalreporter):
    if _P3_RECORDS:
        terminalreporter.section("Phase 3 validation records")
        for nodeid, value in _P3_RECORDS:
            terminalreporter.write_line(f"{nodeid} {value}")


@dataclass(frozen=True)
class VARTruth:
    """Parameters of the simulated VAR(1) on the transformed scale."""

    const: np.ndarray  # (n,)
    coefs: np.ndarray  # A, (n, n): y_t = const + A y_{t-1} + e_t
    sigma: np.ndarray  # (n, n)
    mean: np.ndarray  # unconditional mean, (n,)

    @property
    def chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.sigma)

    @property
    def B(self) -> np.ndarray:
        """Coefficients in the ``lagged_design`` layout with ``deterministic="c"``: ``(1 + n, n)``."""
        return np.vstack([self.const[None, :], self.coefs.T])


def simulate_transformed(T: int = SIM_T, seed: int = SIM_SEED, burn: int = 200) -> np.ndarray:
    """Draw ``T`` observations of the true VAR(1) after a burn-in started at the mean."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(TRUE_SIGMA)
    y = TRUE_MEAN.copy()
    out = np.empty((T, 3))
    for t in range(burn + T):
        y = TRUE_CONST + TRUE_A @ y + chol @ rng.standard_normal(3)
        if t >= burn:
            out[t - burn] = y
    return out


@pytest.fixture(scope="session")
def simulate_var():
    """``simulate_var(T, seed) -> (T, 3)`` array of fresh draws from the true VAR (replications)."""
    return simulate_transformed


@pytest.fixture(scope="session")
def sim_truth() -> VARTruth:
    return VARTruth(
        const=TRUE_CONST.copy(), coefs=TRUE_A.copy(), sigma=TRUE_SIGMA.copy(), mean=TRUE_MEAN.copy()
    )


@pytest.fixture(scope="session")
def sim_transformed() -> pd.DataFrame:
    """The transformed sample ``y_t`` (1960Q1..2019Q4) that both simulated datasets reproduce."""
    index = pd.period_range(SIM_START, periods=SIM_T, freq="Q")
    return pd.DataFrame(simulate_transformed(), index=index, columns=list(SIM_VARIABLES))


@pytest.fixture(scope="session")
def sim_raw(sim_transformed: pd.DataFrame) -> pd.DataFrame:
    """Raw levels behind ``sim_data``: one extra leading period (1959Q4) for the log differences."""
    y = sim_transformed
    index = pd.period_range(SIM_START - 1, periods=len(y) + 1, freq="Q")
    raw = pd.DataFrame(index=index, columns=list(SIM_VARIABLES), dtype=np.float64)
    for var, x0 in INITIAL_LEVELS.items():
        raw[var] = x0 * np.exp(np.concatenate([[0.0], np.cumsum(y[var].to_numpy())]))
    raw["rate"] = np.concatenate([[TRUE_MEAN[2]], y["rate"].to_numpy()])
    return raw


@pytest.fixture(scope="session")
def sim_data(sim_raw: pd.DataFrame) -> VARData:
    """Quarterly raw levels (gdp, prices: tcode 5; rate: tcode 1), 240 transformed observations."""
    return VARData.from_frame(sim_raw, SIM_TCODES)


@pytest.fixture(scope="session")
def sim_data_stationary(sim_transformed: pd.DataFrame) -> VARData:
    """The same transformed sample stored directly, tcode 1 for every variable."""
    return VARData.from_frame(sim_transformed, 1)
