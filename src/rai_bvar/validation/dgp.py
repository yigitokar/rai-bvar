"""DGP tests of the verdict engine (SPEC §10.5): size and power of ``compare().verdict()``.

Each replication simulates one sample from a known data-generating process, runs the full
``compare()`` pipeline on it (a rolling-window backtest of every arm, then the SPEC §9.5 verdict)
and records the verdict of every pair of arms. Over replications the verdicts give the size of the
engine (the rate of a false ``better`` on a DGP where the model cannot improve on the reference)
and its power (the rate of ``better`` where it should), each with its binomial Monte Carlo error.
Nothing here changes the engine: the arms are built with library defaults, ``compare()`` and
``verdict()`` run with their defaults, and the summaries only count what ``verdict()`` returned
(the secondary "gate off" verdict is ``verdict(min_converged=0)``, reported as such).

**DGPs** (:class:`LinearVARDGP`). The model's own recursive structural form (SPEC §4.1),

    y_t = c + Σ_l B_l y_{t−l} + u_t,   u_t = A⁻¹ e_t,   e_{i,t} = o_{i,t} exp(h_{i,t}/2) ε_{i,t},

with fixed coefficients (``A`` unit lower triangular), ``ε ~ N(0, I)``, and either constant log
variances ``h_{i,t} = μ_i`` or stationary AR(1) stochastic volatility ``h_{i,t} = μ_i + φ(h_{i,t−1}
− μ_i) + σ η_{i,t}``; with outliers, ``o_{i,t}`` is 1 with probability ``1 − p`` and uniform on
``{2, …, K}`` otherwise (the mechanism of ``SV(outliers=True)``, DESIGN D-9), else 1. The sample
is returned as the first differences of cumulated levels (tcode 2), so the estimation sample is the
simulated ``y`` and the default Minnesota prior mean of the own first lag is 0 (SPEC §5.3), as for
growth rates.

**Rows** (:data:`ROWS`). A row is one DGP, one ``compare()`` over its arms with a baseline, and the
pairs whose verdicts it scores, each with an expectation:

* ``"better"``: the verdict should be ``better`` (power: the rate of ``better``);
* ``"not_better"``: the model cannot improve on the reference (size: the rate of ``better``);
* ``"not_worse"``: the model should not lose to the reference (the rate of ``worse``).

The MCS route of the verdict rule depends on every arm in the ``compare()``, so a pair's verdict
depends on its row's arm set. A new model therefore gets a row of its own on an existing DGP
instead of an extra arm in an existing row: rows on one DGP share its simulated data (the data
seed depends on the DGP's name and the replication only) and the backtest seed, so a shared arm is
served from the score cache and earlier rows' verdicts do not move.

**Plug-in point for Phases 4–5.** The rows of SPEC §10.5 that need models the library does not have
yet are declared below with ``requires`` (the ``rai_bvar`` attributes their arms need) and, for
the DGPs not written yet, ``DGPS[name] = None``. :func:`row_status` reports them as pending, and
the study script skips them. To activate one: add the model (e.g. ``rai_bvar.BART``), add its arm
factory to :data:`ARMS` if its constructor differs from the sketch there, write the DGP simulator
(a class with ``simulate(T, rng) -> VARData``, like :class:`LinearVARDGP`) and register it in
:data:`DGPS`, then pre-register the row's settings and its reading (a power threshold, if any)
before its first run, as ``benchmarks/reports/dgp_study_preregistration.md`` did for the Phase 2
rows.

Importing this module imports no JAX (DESIGN D-31); fitting an MCMC arm does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import re
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

import rai_bvar
from rai_bvar.api import BVAR
from rai_bvar.components import SV, Constant
from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation.backtest import Cache, Origins, Rolling
from rai_bvar.evaluation.compare import CompareResult, compare
from rai_bvar.models.base import Model, spec_hash
from rai_bvar.settings import MCMCSettings

Expectation = Literal["better", "not_better", "not_worse"]
VERDICTS = ("better", "worse", "inconclusive")
Z95 = 1.959963984540054  # standard normal 0.975 quantile (Wilson intervals)

# the verdict rule's defaults (SPEC §9.5), read from the engine so the summaries follow it
VERDICT_DEFAULTS: dict[str, float] = {
    name: p.default
    for name, p in inspect.signature(CompareResult.verdict).parameters.items()
    if p.kind is inspect.Parameter.KEYWORD_ONLY
}
NOMINAL = float(VERDICT_DEFAULTS["test_level"])  # the one-sided DM level of the rule (0.10)


# --------------------------------------------------------------------------- DGPs


@dataclass(frozen=True)
class LinearVARDGP:
    """A linear VAR(p) in recursive structural form with constant or stochastic volatility.

    ``lag_coefs[l][i][j]`` is the reduced-form coefficient of ``y_{j,t−l−1}`` in equation ``i``;
    ``contemp`` is ``A`` (unit lower triangular); ``log_var`` holds ``μ_i``. ``vol="sv"`` uses
    ``phi`` and ``sigma`` with a stationary start ``h_{i,0} ~ N(μ_i, σ²/(1 − φ²))``;
    ``outlier_prob > 0`` adds the outlier scale ``o_{i,t}``. The first ``burn`` periods (started
    at the unconditional mean) are discarded.
    """

    name: str
    intercept: tuple[float, ...]
    lag_coefs: tuple[tuple[tuple[float, ...], ...], ...]
    contemp: tuple[tuple[float, ...], ...]
    log_var: tuple[float, ...]
    vol: Literal["constant", "sv"] = "constant"
    phi: float = 0.95
    sigma: float = 0.2
    outlier_prob: float = 0.0
    outlier_max: int = 20
    burn: int = 100
    start: str = "1950Q1"  # first raw (level) period; quarterly
    variables: tuple[str, ...] = ("y1", "y2", "y3")

    def __post_init__(self) -> None:
        n = len(self.intercept)
        A = np.asarray(self.contemp, dtype=np.float64)
        if A.shape != (n, n) or not np.allclose(np.triu(A, 1), 0.0) or not np.all(np.diag(A) == 1):
            raise ValueError("contemp must be an n × n unit lower triangular matrix")
        if np.asarray(self.lag_coefs).shape[1:] != (n, n) or len(self.log_var) != n:
            raise ValueError("lag_coefs must be p × n × n and log_var of length n")
        if len(self.variables) != n:
            raise ValueError(f"{n} variables need {n} names")
        if self.vol not in ("constant", "sv"):
            raise ValueError(f"vol must be 'constant' or 'sv', got {self.vol!r}")
        if self.vol == "sv" and not (abs(self.phi) < 1 and self.sigma > 0):
            raise ValueError("stochastic volatility needs |phi| < 1 and sigma > 0")
        if not 0 <= self.outlier_prob < 1 or self.outlier_max < 2:
            raise ValueError("outlier_prob must be in [0, 1) and outlier_max at least 2")
        if self.spectral_radius() >= 1:
            raise ValueError(
                f"the VAR is not stationary (spectral radius {self.spectral_radius()})"
            )

    @property
    def n(self) -> int:
        return len(self.intercept)

    @property
    def lags(self) -> int:
        return len(self.lag_coefs)

    def companion(self) -> np.ndarray:
        n, p = self.n, self.lags
        C = np.zeros((n * p, n * p))
        C[:n] = np.hstack([np.asarray(b, dtype=np.float64) for b in self.lag_coefs])
        C[n:, : n * (p - 1)] = np.eye(n * (p - 1))
        return C

    def spectral_radius(self) -> float:
        return float(np.abs(np.linalg.eigvals(self.companion())).max())

    def mean(self) -> np.ndarray:
        """Unconditional mean ``(I − Σ B_l)⁻¹ c``."""
        B = sum(np.asarray(b, dtype=np.float64) for b in self.lag_coefs)
        return np.linalg.solve(np.eye(self.n) - B, np.asarray(self.intercept, dtype=np.float64))

    def simulate_arrays(self, T: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """``y`` (``T × n``), the reduced-form errors ``u``, the log variances ``h`` and the
        outlier scales ``o`` of one draw.

        The draws are taken in a fixed order: ``ε``, then (SV) ``h_0`` and ``η``, then (outliers)
        the outlier indicators and scales; so a seed gives the same sample on every platform.
        """
        n, p, H = self.n, self.lags, self.burn + T
        eps = rng.standard_normal((H, n))
        mu = np.asarray(self.log_var, dtype=np.float64)
        if self.vol == "sv":
            z0, eta = rng.standard_normal(n), rng.standard_normal((H, n))
            h = np.empty((H, n))
            prev = mu + self.sigma / math.sqrt(1.0 - self.phi**2) * z0
            for t in range(H):
                prev = mu + self.phi * (prev - mu) + self.sigma * eta[t]
                h[t] = prev
        else:
            h = np.broadcast_to(mu, (H, n)).copy()
        o = np.ones((H, n))
        if self.outlier_prob > 0:
            hit = rng.random((H, n)) < self.outlier_prob
            scale = rng.integers(2, self.outlier_max + 1, size=(H, n))
            o = np.where(hit, scale, 1).astype(np.float64)
        e = o * np.exp(0.5 * h) * eps
        A = np.asarray(self.contemp, dtype=np.float64)
        u = np.linalg.solve(A, e.T).T  # A is unit lower triangular: exact forward substitution
        c = np.asarray(self.intercept, dtype=np.float64)
        B = [np.asarray(b, dtype=np.float64) for b in self.lag_coefs]
        y = np.empty((H + p, n))
        y[:p] = self.mean()
        for t in range(H):
            y[p + t] = c + sum(B[j] @ y[p + t - j - 1] for j in range(p)) + u[t]
        keep = slice(self.burn, H)
        return {"y": y[p:][keep], "u": u[keep], "h": h[keep], "o": o[keep]}

    def simulate(self, T: int, rng: np.random.Generator) -> VARData:
        """``T`` observations as a quarterly ``VARData`` of tcode-2 series (the sample is ``y``)."""
        y = self.simulate_arrays(T, rng)["y"]
        levels = np.vstack([np.zeros((1, self.n)), np.cumsum(y, axis=0)])
        index = pd.period_range(self.start, periods=T + 1, freq="Q")
        frame = pd.DataFrame(levels, index=index, columns=list(self.variables))
        return VARData.from_frame(frame, 2, vintage=f"dgp:{self.name}")

    def describe(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# One mean and contemporaneous structure for every Phase 2 DGP (fixed before any run; spectral
# radius 0.82; reduced-form error correlations 0.49, −0.15 and 0.26 under constant volatility)
_MEAN = {
    "intercept": (0.3, 0.2, 0.1),
    "lag_coefs": (
        ((0.5, 0.1, 0.0), (0.2, 0.4, 0.1), (0.0, 0.2, 0.3)),
        ((0.2, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
    ),
    "contemp": ((1.0, 0.0, 0.0), (-0.4, 1.0, 0.0), (0.2, -0.3, 1.0)),
    "log_var": (0.0, math.log(0.5), math.log(0.25)),
}

DGPS: dict[str, LinearVARDGP | None] = {
    "linear-homoskedastic": LinearVARDGP("linear-homoskedastic", **_MEAN),
    "linear-sv": LinearVARDGP("linear-sv", **_MEAN, vol="sv", phi=0.95, sigma=0.2),
    # outlier rate 1/16: the prior mean of SV(outliers=True) on quarterly data (one every 4 years)
    "linear-sv-outliers": LinearVARDGP(
        "linear-sv-outliers", **_MEAN, vol="sv", phi=0.95, sigma=0.2, outlier_prob=1 / 16
    ),
    # Phase 4–5 (pending): to be written and pre-registered with their rows
    "threshold-sv": None,
    "tvp-sv": None,
}


# --------------------------------------------------------------------------- arms and rows


def _component(name: str) -> Any:
    """A ``rai_bvar`` component that may not exist yet (Phase 4–5 arms); resolved when built."""
    try:
        return getattr(rai_bvar, name)
    except AttributeError:
        raise NotImplementedError(f"rai_bvar.{name} is not part of this build") from None


def _bvar(vol: Any, mean: Any = None) -> Callable[[int, MCMCSettings | None], Model]:
    def build(lags: int, mcmc: MCMCSettings | None) -> Model:
        kwargs: dict[str, Any] = {"vol": vol, "lags": lags, "mcmc": mcmc}
        if mean is not None:
            kwargs["mean"] = mean() if callable(mean) else mean
        return BVAR(**kwargs)

    return build


ARMS: dict[str, Callable[[int, MCMCSettings | None], Model]] = {
    # library defaults throughout: Minnesota(form="auto") resolves to the structural form on the
    # MCMC engine and Constant("ig") keeps the Constant arm there (SPEC §4.5), so every pair
    # differs along one axis; mcmc=None is the per-composition default budget (D-35)
    "Linear × Constant": _bvar(Constant("ig")),
    "Linear × SV": _bvar(SV()),
    "Linear × SV(outliers)": _bvar(SV(outliers=True)),
    # Phase 4–5 sketches (their constructors are not settled yet)
    "BART × SV": _bvar(SV(), mean=lambda: _component("BART")()),
    "Threshold × SV": _bvar(SV(), mean=lambda: _component("Threshold")()),
    "TVP × SV": _bvar(SV(), mean=lambda: _component("TVP")()),
}


@dataclass(frozen=True)
class Pair:
    """A scored verdict: ``model`` against ``reference`` along ``axis``, with its expectation."""

    model: str
    reference: str
    axis: str
    expect: Expectation


@dataclass(frozen=True)
class Row:
    name: str
    dgp: str
    arms: tuple[str, ...]
    baseline: str
    pairs: tuple[Pair, ...]
    spec: str  # the SPEC §10.5 line this row tests
    requires: tuple[str, ...] = ()  # rai_bvar attributes the arms need
    phase: int = 2


_C, _SV, _SVO = "Linear × Constant", "Linear × SV", "Linear × SV(outliers)"

ROWS: tuple[Row, ...] = (
    Row(
        "linear-homoskedastic",
        "linear-homoskedastic",
        (_C, _SV, _SVO),
        _C,
        (Pair(_SV, _C, "vol", "not_better"), Pair(_SVO, _C, "vol", "not_better")),
        "Linear, homoskedastic: nothing beats Linear × Constant",
    ),
    Row(
        "linear-sv",
        "linear-sv",
        (_C, _SV),
        _C,
        (Pair(_SV, _C, "vol", "better"),),
        "Linear, SV: SV beats Constant",
    ),
    Row(
        "linear-sv-outliers",
        "linear-sv-outliers",
        (_SV, _SVO),
        _SV,
        (Pair(_SVO, _SV, "vol", "not_worse"),),
        "Linear, SV + outliers: SV(outliers=True) ≥ SV",
    ),
    # ---- pending (Phases 4–5): the plug-in point of the module docstring
    Row(
        "linear-sv-bart",
        "linear-sv",
        (_SV, "BART × SV"),
        _SV,
        (Pair("BART × SV", _SV, "mean", "not_better"),),
        "Linear, SV: BART × SV does not beat Linear × SV (false-positive rate ≤ nominal)",
        requires=("BART",),
        phase=4,
    ),
    Row(
        "threshold-sv-bart",
        "threshold-sv",
        (_SV, "BART × SV"),
        _SV,
        (Pair("BART × SV", _SV, "mean", "better"),),
        "Threshold, SV: BART × SV beats Linear × SV (power)",
        requires=("BART",),
        phase=4,
    ),
    Row(
        "threshold-sv-threshold",
        "threshold-sv",
        (_SV, "Threshold × SV"),
        _SV,
        (Pair("Threshold × SV", _SV, "mean", "better"),),
        "Threshold, SV: Threshold × SV beats Linear × SV (power)",
        requires=("Threshold",),
        phase=5,
    ),
    Row(
        "tvp-sv",
        "tvp-sv",
        (_SV, "TVP × SV"),
        _SV,
        (Pair("TVP × SV", _SV, "mean", "better"),),
        "TVP with sizeable drift, SV: TVP × SV beats Linear × SV",
        requires=("TVP",),
        phase=5,
    ),
)


def get_row(name: str) -> Row:
    for row in ROWS:
        if row.name == name:
            return row
    raise KeyError(f"no DGP row {name!r}; rows: {[r.name for r in ROWS]}")


def row_status(row: Row) -> str | None:
    """``None`` when the row can run in this build, else why it is pending."""
    missing = [r for r in row.requires if not hasattr(rai_bvar, r)]
    if DGPS.get(row.dgp) is None:
        missing.append(f"the {row.dgp!r} DGP simulator")
    if missing:
        return f"pending (Phase {row.phase}): needs {', '.join(missing)}"
    return None


# --------------------------------------------------------------------------- one replication


@dataclass(frozen=True)
class Design:
    """The settings of one study cell, shared by all its replications (and by pilot and full).

    ``T`` simulated observations; a rolling window of ``window`` observations, with forecast
    origins every ``origin_step`` periods from the first full window to the second-to-last
    period; ``mcmc=None`` keeps every arm's library default budget.
    """

    T: int = 200
    window: int = 120
    lags: int = 2
    horizons: tuple[int, ...] = (1, 4)
    draws: int = 1000  # forecast draws per origin: compare()'s default
    origin_step: int = 1
    seed: int = 20261025
    mcmc: MCMCSettings | None = None

    def describe(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["horizons"] = list(self.horizons)
        out["mcmc"] = None if self.mcmc is None else dataclasses.asdict(self.mcmc)
        return out

    def key(self) -> str:
        """Hash of the design: records of one design can be pooled, others never are."""
        blob = json.dumps(self.describe(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def origins(self, data: VARData) -> Origins:
        idx = data.index
        return Origins(idx[self.window - 1], idx[-2], self.origin_step)


def data_rng(design: Design, dgp: str, rep: int) -> np.random.Generator:
    """The DGP noise of replication ``rep``: depends on the DGP's name, not on the row."""
    return np.random.default_rng([design.seed, zlib.crc32(dgp.encode()), rep])


def compare_seed(design: Design, dgp: str, rep: int) -> int:
    """The backtest seed of replication ``rep`` (per-origin fit seeds derive from it)."""
    rng = np.random.default_rng([design.seed, zlib.crc32(dgp.encode()), rep, 1])
    return int(rng.integers(2**31))


def models(row: Row, design: Design) -> dict[str, Model]:
    return {name: ARMS[name](design.lags, design.mcmc) for name in row.arms}


def run_replication(
    row: Row, rep: int, design: Design, *, cache: Cache | str | None = None
) -> dict[str, Any]:
    """Simulate replication ``rep`` of ``row``, run ``compare()`` and return its record.

    The record holds, per pair and horizon, the verdict (``verdict``), the verdict with the
    convergence gate off (``verdict_gate_off``, secondary) and the quantities the rule used, and
    per arm and horizon the converged fraction of origins.
    """
    status = row_status(row)
    if status is not None:
        raise NotImplementedError(f"row {row.name!r} is {status}")
    dgp = DGPS[row.dgp]
    assert dgp is not None
    data = dgp.simulate(design.T, data_rng(design, row.dgp, rep))
    seed = compare_seed(design, row.dgp, rep)
    arms = models(row, design)
    start = time.perf_counter()
    res = compare(
        arms,
        data,
        origins=design.origins(data),
        window=Rolling(design.window),
        horizons=list(design.horizons),
        draws=design.draws,
        seed=seed,
        baseline=row.baseline,
        cache=cache,
    )
    table = res.table()
    pairs = []
    for axis in sorted({p.axis for p in row.pairs}):
        on, off = res.verdict(axis=axis), res.verdict(axis=axis, min_converged=0.0)
        for pair in (p for p in row.pairs if p.axis == axis):
            for h in design.horizons:
                pairs.append(_pair_record(pair, h, on, off, table))
    seconds = time.perf_counter() - start
    return {
        "row": row.name,
        "rep": rep,
        "design": design.key(),
        "dgp": row.dgp,
        "data_fingerprint": data.fingerprint(),
        "compare_seed": seed,
        "seconds": seconds,
        "arms": {
            name: {
                "spec_hash": spec_hash(arms[name]),
                "seconds": r.seconds,
                "cached": r.cached,
                "origins": len(r.origins),
                "converged": {
                    str(h): float(table.loc[(name, h), "converged"]) for h in design.horizons
                },
                **warning_counts(r.warnings),
            }
            for name, r in res.results.items()
        },
        "pairs": pairs,
    }


_WORST = re.compile(r"worst: ([^ ]+) R-hat")


def warning_counts(by_message: Mapping[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    """A backtest's warnings (message → origins) as origins per kind of message (the text before
    the first colon: the messages embed per-origin numbers) and, for ``MCMC not converged``, per
    first-named failing quantity (the worst one, DESIGN §8.3)."""
    kinds: dict[str, int] = {}
    worst: dict[str, int] = {}
    for message, origins in by_message.items():
        kind = message.split(":")[0][:80]
        kinds[kind] = kinds.get(kind, 0) + len(origins)
        match = _WORST.search(message) if kind == "MCMC not converged" else None
        if match:
            worst[match.group(1)] = worst.get(match.group(1), 0) + len(origins)
    return {"warnings": dict(sorted(kinds.items())), "worst": dict(sorted(worst.items()))}


def _pair_record(
    pair: Pair, h: int, on: pd.DataFrame, off: pd.DataFrame, table: pd.DataFrame
) -> dict[str, Any]:
    def pick(frame: pd.DataFrame) -> pd.Series:
        sel = frame[
            (frame["model"] == pair.model)
            & (frame["reference"] == pair.reference)
            & (frame["horizon"] == h)
        ]
        if len(sel) != 1:
            raise RuntimeError(f"verdict has {len(sel)} rows for {pair} at h={h}")
        out = sel.iloc[0]
        if out["verdict"] not in VERDICTS:
            raise RuntimeError(f"{pair} at h={h}: {out['verdict']} ({out['reasons']})")
        return out

    v, g = pick(on), pick(off)
    return {
        "model": pair.model,
        "reference": pair.reference,
        "axis": pair.axis,
        "expect": pair.expect,
        "horizon": int(h),
        "verdict": str(v["verdict"]),
        "verdict_gate_off": str(g["verdict"]),
        "dlpd": float(v["dlpd"]),
        "dlpd_mcse": float(v["dlpd_mcse"]),
        "dm_pvalue_better": float(v["dm_pvalue_better"]),
        "dm_pvalue_worse": float(v["dm_pvalue_worse"]),
        "tests_valid": bool(v["tests_valid"]),
        "model_in_mcs": bool(v["model_in_mcs"]),
        "reference_in_mcs": bool(v["reference_in_mcs"]),
        "origins": int(table.loc[(pair.model, h), "origins"]),
        "converged_model": float(table.loc[(pair.model, h), "converged"]),
        "converged_reference": float(table.loc[(pair.reference, h), "converged"]),
        "reasons": str(v["reasons"]),
    }


# --------------------------------------------------------------------------- summaries


def mc_se(k: int, n: int) -> float:
    """Binomial Monte Carlo standard error of a rate ``k/n`` over independent replications."""
    if n == 0:
        return math.nan
    p = k / n
    return math.sqrt(p * (1.0 - p) / n)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (95% by default)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(pair: Mapping[str, Any]) -> str:
    """Why a pair got its verdict, from the quantities the rule used (SPEC §9.5).

    ``better:test``, ``better:mcs``, ``better:both`` (and the same for ``worse``): which route of
    condition 3 held. For ``inconclusive``: ``gate`` (the convergence gate alone, since the gate-off
    verdict is decisive), ``imprecise`` (|ΔLPD| ≤ 3 MCSE) or ``no_route`` (precise, but neither the
    DM test nor the MCS separates the pair in the direction of ΔLPD). Raises if the derivation
    disagrees with the engine's gate-off verdict, so a harness that misreads the rule fails loudly.
    """
    level, mult = VERDICT_DEFAULTS["test_level"], VERDICT_DEFAULTS["mc_multiple"]
    dlpd, mcse = pair["dlpd"], pair["dlpd_mcse"]
    precise = abs(dlpd) > mult * mcse
    up = dlpd > 0
    p = pair["dm_pvalue_better"] if up else pair["dm_pvalue_worse"]
    test = bool(pair["tests_valid"]) and p < level
    m_in, r_in = pair["model_in_mcs"], pair["reference_in_mcs"]
    mcs = (m_in and not r_in) if up else (r_in and not m_in)
    decisive = dlpd != 0 and precise and (test or mcs)
    derived = ("better" if up else "worse") if decisive else "inconclusive"
    if derived != pair["verdict_gate_off"]:
        raise RuntimeError(
            f"derived gate-off verdict {derived!r} != engine's {pair['verdict_gate_off']!r}: {pair}"
        )
    if pair["verdict"] != "inconclusive":
        route = "both" if test and mcs else "test" if test else "mcs"
        return f"{pair['verdict']}:{route}"
    if decisive:
        return "gate"
    return "imprecise" if not precise else "no_route"


def _rates(verdicts: Sequence[str]) -> dict[str, Any]:
    n = len(verdicts)
    out: dict[str, Any] = {"n": n}
    for v in VERDICTS:
        k = sum(x == v for x in verdicts)
        lo, hi = wilson(k, n)
        out[v] = {"k": k, "rate": k / n if n else math.nan, "se": mc_se(k, n), "lo": lo, "hi": hi}
    return out


TARGET = {"better": "better", "not_better": "better", "not_worse": "worse"}
"""The verdict whose rate each expectation scores: power for ``better``, the false-``better``
rate (size) for ``not_better``, the ``worse`` rate for ``not_worse``."""


def reading(expect: Expectation, k: int, n: int) -> str:
    """The pre-registered reading of a scored rate ``k/n`` (``benchmarks/reports/
    dgp_study_preregistration.md``): a null-type rate is "above nominal" only when its Wilson 95%
    lower bound exceeds the rule's DM level (:data:`NOMINAL`); power is descriptive."""
    if n == 0:
        return "no replications"
    if expect == "better":
        return "descriptive (power)"
    return "above nominal" if wilson(k, n)[0] > NOMINAL else "not above nominal"


def summarise(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One summary per (row, pair, horizon) over the records' replications."""
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    expects: dict[tuple[str, str, str, int], str] = {}
    for rec in records:
        for p in rec["pairs"]:
            key = (rec["row"], p["model"], p["reference"], int(p["horizon"]))
            groups.setdefault(key, []).append(p)
            expects[key] = p["expect"]
    out = []
    for key in sorted(groups, key=lambda k: (_row_order(k[0]), k[1], k[2], k[3])):
        ps = groups[key]
        on = _rates([p["verdict"] for p in ps])
        off = _rates([p["verdict_gate_off"] for p in ps])
        expect = expects[key]
        target = on[TARGET[expect]]
        gate = VERDICT_DEFAULTS["min_converged"]
        conv = np.array([min(p["converged_model"], p["converged_reference"]) for p in ps])
        dlpd = np.array([p["dlpd"] for p in ps])
        out.append(
            {
                "row": key[0],
                "model": key[1],
                "reference": key[2],
                "horizon": key[3],
                "expect": expect,
                "target": TARGET[expect],
                "n": len(ps),
                "rate": target["rate"],
                "se": target["se"],
                "lo": target["lo"],
                "hi": target["hi"],
                "reading": reading(expect, target["k"], len(ps)),  # type: ignore[arg-type]
                "gate_on": on,
                "gate_off": off,
                "classes": _counts(classify(p) for p in ps),
                "dlpd_mean": float(dlpd.mean()),
                "dlpd_median": float(np.median(dlpd)),
                "dlpd_mcse_mean": float(np.mean([p["dlpd_mcse"] for p in ps])),
                "origins_mean": float(np.mean([p["origins"] for p in ps])),
                "converged_mean": float(conv.mean()),
                "gate_pass": float(np.mean(conv >= gate)),
            }
        )
    return out


def family(records: Sequence[Mapping[str, Any]], row: str) -> list[dict[str, Any]]:
    """Per horizon, the rate of replications in which any ``not_better`` pair of ``row`` was
    ``better`` (gate on and gate off): the family-wise false-``better`` rate."""
    out = []
    reps = [r for r in records if r["row"] == row]
    horizons = sorted({int(p["horizon"]) for r in reps for p in r["pairs"]})
    for h in horizons:
        res = {}
        for col in ("verdict", "verdict_gate_off"):
            hits = [
                any(
                    p[col] == "better"
                    for p in r["pairs"]
                    if p["expect"] == "not_better" and int(p["horizon"]) == h
                )
                for r in reps
            ]
            k, n = sum(hits), len(hits)
            lo, hi = wilson(k, n)
            res[col] = {
                "k": k,
                "n": n,
                "rate": k / n if n else math.nan,
                "se": mc_se(k, n),
                "lo": lo,
                "hi": hi,
            }
        out.append({"row": row, "horizon": h, **res})
    return out


def convergence(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per (row, arm, horizon): the mean converged fraction of origins, the share of
    replications at or above the gate, the mean backtest time per fitted origin, and the
    quantities most often named worst when a fit missed the gate (all horizons share a fit)."""
    gate = VERDICT_DEFAULTS["min_converged"]
    groups: dict[tuple[str, str, str], list[float]] = {}
    seconds: dict[tuple[str, str], list[float]] = {}
    worst: dict[tuple[str, str], dict[str, int]] = {}
    for rec in records:
        for arm, a in rec["arms"].items():
            for h, c in a["converged"].items():
                groups.setdefault((rec["row"], arm, h), []).append(c)
            fitted = a["origins"] - a["cached"]
            if fitted > 0:
                seconds.setdefault((rec["row"], arm), []).append(a["seconds"] / fitted)
            tally = worst.setdefault((rec["row"], arm), {})
            for q, k in a.get("worst", {}).items():
                tally[q] = tally.get(q, 0) + k
    out = []
    order = sorted(groups, key=lambda k: (_row_order(k[0]), k[1], int(k[2])))
    for row, arm, h in order:
        c = np.array(groups[(row, arm, h)])
        s = seconds.get((row, arm), [])
        top = sorted(worst.get((row, arm), {}).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        out.append(
            {
                "row": row,
                "arm": arm,
                "horizon": int(h),
                "n": len(c),
                "converged_mean": float(c.mean()),
                "converged_min": float(c.min()),
                "gate_pass": float(np.mean(c >= gate)),
                "seconds_per_origin": float(np.mean(s)) if s else math.nan,
                "worst": dict(top),
            }
        )
    return out


def _counts(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _row_order(name: str) -> int:
    for i, row in enumerate(ROWS):
        if row.name == name:
            return i
    return len(ROWS)
