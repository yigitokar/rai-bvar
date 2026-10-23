"""Clark & Ravazzolo (2015) replication on FRED-QD (DESIGN §13 WP23; N08 §C1–C6, §B4).

**The acceptance run** (WP23: all models, both samples, all 24 orderings), from the repository
root, as one command::

    uv run python benchmarks/clark_ravazzolo.py benchmarks/reports/clark_ravazzolo \\
        --config full --n-jobs 4

It writes ``benchmarks/reports/clark_ravazzolo.md`` and ``.json``. **Estimated runtime:** about
12 hours with ``--n-jobs 1``, extrapolated from fits measured on a 4-core x86-64 VM (5000-draw
SV fits took 6–12 s each, Constant fits about 3.5 s, for 742 main-stage fits and 2 × 25 × 106 =
5300 ordering-stage fits). On a 10-core machine ``--n-jobs 4`` should bring that to roughly 4–6
hours, an *estimate*, not a measurement. Every origin is cached as it finishes
(``--cache``, default ``~/.cache/rai-bvar-cr``), so rerunning the same command after an
interruption resumes where it stopped. The report of the main stage is written before the
24-ordering stage starts.

Usage::

    python benchmarks/clark_ravazzolo.py OUT [--config full|quick|smoke] [--cache DIR]
        [--n-jobs N] [--qd-url URL] [--md-url URL] [--skip-orderings] [--skip-structural]
        [--only-ordering MODEL] [--seed 0]

writes ``OUT.md`` (the report) and ``OUT.json`` (every number in it). ``--n-jobs N`` fits N
origins at a time in ``spawn``-started worker processes (DESIGN §11.6; each worker's XLA is
multi-threaded too, so use about a third of the cores). Results do not depend on it: the per-origin
seeds and cache keys are the same. ``--qd-url`` / ``--md-url`` point the loader at a mirror or a
``file://`` copy of the pinned vintage files (the SHA-256 checksum of the registry is still
verified; the St. Louis Fed server has rejected the library's User-Agent through some proxies,
in which case ``curl`` the two CSVs and pass ``file://`` URLs).

What it does (N08 §C):

* **Data** (§C2): FRED-QD vintage 2026-09, GDP growth ``100·Δlog GDPC1``, inflation
  ``100·Δlog GDPCTPI`` (FRED-QD has no ``GDPDEF``), the unemployment rate ``UNRATE`` and the
  3-month bill ``TB3MS`` in levels; non-annualised quarterly rates. Estimation starts in 1965Q1
  (FRED-QD starts in 1959; §C2 recommendation). The script **asserts** that FRED-QD's ``UNRATE``
  and ``TB3MS`` are within-quarter averages of the FRED-MD monthly series of the same vintage
  (§C6) and that every prior quantity is computed from the estimation window alone (R9; §C6: CR's
  last-vintage training sample is *not* replicated).
* **Evaluation** (§C2, §C4): recursive (expanding) estimation, origins 1985Q1–2011Q2 (full) and
  1985Q1–2007Q4 (pre-crisis, a subset of the same origins), h = 1, 2, 4, 8 (quarterly rates, h = 1
  is the first quarter after the last observation). Pseudo real time on the final 2026-09 data
  (D4): CR's real-time vintages and second-release actuals are not used.
* **Models** (§C3): ``Linear(Minnesota(form="reduced", λ=0.2, α=2, cross=1, mean=0))``, p = 4,
  with ``Constant("iw_equivalent")`` (the baseline: Chan's order-invariant stand-in for CR's
  normal-diffuse VAR), ``SV(rw=True)`` (CR's VAR-SV; the *primary* SV model of criteria 1–4 and
  6), ``SV()`` (the library's AR default), ``SV(rw=True, tails="t", ν = 10 fixed, unit scale)``
  (CR's VAR-SVt, d = 10) and an AR-SV with CR's tight ``a₁ ~ N(0.9, 0.05²)`` mapped to
  ``(φ+1)/2 ~ Beta(71.25, 3.75)`` (same mean and sd). The structural-form pair
  (``Constant("ig")`` vs ``SV(rw=True)``) is reported as a robustness check (§C3 R2). Every other
  setting is the library default: no prior was tuned.
* **Scores** (§C4): the Rao–Blackwellised LPD (primary), the Gaussian-approximation LPS of
  Adolfson, Lindé & Villani (2007) from the predictive draws' mean and covariance (CR's score;
  a parity metric), CRPS and RMSE of the predictive mean. Relative measures are against the
  baseline over common origins. ``compare()`` verdicts (SPEC §9.5) and one-sided DM/HLN tests per
  variable (descriptive: expanding window).
* **Orderings** (§B4, §C5 criterion 6): ``rb.ordering_sensitivity(mode="backtest", design="all")``
  over all 24 orderings for the primary SV model and for the baseline, then a ``compare()``
  verdict per ordering.

The six acceptance criteria of N08 §C5 are evaluated with the operational definitions printed in
the report. ``--config full`` uses 4 chains × 5000 kept draws (burn 1000); ``--config quick``
the library's default ``MCMCSettings()`` (4 × 2000, burn 1000); both use every origin and 1000
forecast draws. ``benchmarks/reports/clark_ravazzolo_quick.md`` is a ``quick`` main-stage report
(no ordering stage), labelled as not the acceptance run. ``--config smoke`` only checks that the
script runs (tiny chains, every 8th origin; its numbers mean nothing). Everything is cached (``--cache``), so an interrupted run
resumes, and ``--only-ordering MODEL`` fills the cache for one model's 24-ordering stage from a
parallel process.

Notes on deviations from N08 (also listed in the report):

* MCMC convergence: ``compare().verdict`` returns "inconclusive" when fewer than 95% of origins
  have ``converged`` fits (SPEC §9.5). With the chain lengths affordable for 24 × 106 SV fits the
  SV hyperparameters (σ, φ) often miss the R-hat/ESS targets, so the report gives the verdict as
  ``compare()`` returns it *and* with the convergence gate off (``min_converged=0``), labelled.
* Gaussian-approximation LPS: computed from the very forecast draws the backtest scores (a probe
  wraps the model; its spec carries a ``"probe"`` entry, so its cache entries are separate).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import rai_bvar as rb
from rai_bvar import BVAR, SV, Constant, Linear, MCMCSettings, Minnesota, VARData
from rai_bvar.datasets import load
from rai_bvar.evaluation.backtest import (
    BacktestResult,
    Cache,
    Expanding,
    Origins,
    backtest,
    fit_seed,
)
from rai_bvar.evaluation.compare import CompareResult, Entry
from rai_bvar.evaluation.significance import diebold_mariano
from rai_bvar.priors.sv import Beta, DiscreteUniform
from rai_bvar.structural.ordering import permute_model

VINTAGE = "2026-09"
SERIES = {"GDPC1": "GDP", "GDPCTPI": "INFL", "UNRATE": "UR", "TB3MS": "TBILL"}
VARS = list(SERIES.values())
ESTIMATION_START = "1965Q1"
FULL = ("1985Q1", "2011Q2")
PRE_CRISIS_END = "2007Q4"
HORIZONS = (1, 2, 4, 8)
LAGS = 4

# CR WP Tables 1–3 (full sample) and the pre-crisis ΔLPS (N08 §C5), for side-by-side display
CR_FULL = {
    "dlps": {
        "GDP": (0.193, 0.133, 0.090, 0.060),
        "INFL": (0.075, 0.083, 0.155, 0.138),
        "UR": (0.162, 0.098, -0.182, -0.707),
        "TBILL": (0.284, 0.043, -0.304, -0.403),
    },
    "crps_ratio": {
        "GDP": (0.864, 0.856, 0.890, 0.927),
        "INFL": (0.954, 0.944, 0.895, 0.887),
        "UR": (0.922, 0.897, 0.906, 1.001),
        "TBILL": (0.822, 0.915, 1.010, 1.059),
    },
    "rmse_ratio": {
        "GDP": (0.896, 0.878, 0.932, 1.007),
        "INFL": (0.966, 0.949, 0.915, 0.883),
        "UR": (0.967, 0.930, 0.914, 0.971),
        "TBILL": (0.933, 0.944, 0.974, 0.990),
    },
}
CR_PRE_DLPS = {
    "GDP": (0.222, 0.214),
    "INFL": (0.077, 0.152),
    "UR": (0.187, -0.065),
    "TBILL": (0.316, -0.261),
}  # (h = 1, h = 8)


@dataclass(frozen=True)
class Config:
    name: str
    mcmc: MCMCSettings
    draws: int  # forecast draws per origin
    ordering_step: int  # every k-th origin in the ordering stage
    noise_reps: int = 1
    origin_step: int = 1  # every k-th origin in the main stage (smoke tests only)


CONFIGS = {
    # 5000 kept draws: the SV hyperparameters come close to the R-hat < 1.01 / ESS > 400 targets
    "full": Config("full", MCMCSettings(chains=4, draws=5000, burn=1000), 1000, 1),
    # the library's default MCMCSettings (4 × 2000, burn 1000)
    "quick": Config("quick", MCMCSettings(), 1000, 1),
    # end-to-end check of the script only (meaningless numbers)
    "smoke": Config("smoke", MCMCSettings(chains=2, draws=100, burn=100), 100, 8, 1, 8),
}


# --------------------------------------------------------------------------- data


def load_data(qd_url: str | None, md_url: str | None) -> tuple[VARData, dict[str, Any]]:
    """The four CR series as a ``VARData`` (percent units) and the aggregation check (§C6)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # ragged-edge trimming notes
        qd = load(
            "fred-qd", VINTAGE, variables=list(SERIES), tcodes={k: 1 for k in SERIES}, url=qd_url
        )
        md = load(
            "fred-md",
            VINTAGE,
            variables=["UNRATE", "TB3MS"],
            tcodes={"UNRATE": 1, "TB3MS": 1},
            url=md_url,
        )
    check = assert_quarterly_averages(qd.raw, md.raw)
    raw = qd.raw.rename(columns=SERIES)
    frame = pd.DataFrame(
        {
            "GDP": 100.0 * np.log(raw["GDP"]),
            "INFL": 100.0 * np.log(raw["INFL"]),
            "UR": raw["UR"],
            "TBILL": raw["TBILL"],
        }
    )
    start = pd.Period(ESTIMATION_START, freq="Q")
    frame = frame.loc[start - 1 :]  # one raw quarter before 1965Q1 for the difference
    data = VARData.from_frame(
        frame, {"GDP": 2, "INFL": 2, "UR": 1, "TBILL": 1}, vintage=f"fred-qd:{VINTAGE}"
    )
    assert str(data.first) == ESTIMATION_START, data.first
    return data, check


def assert_quarterly_averages(qd_raw: pd.DataFrame, md_raw: pd.DataFrame) -> dict[str, Any]:
    """FRED-QD ``UNRATE``/``TB3MS`` must be within-quarter means of the monthly FRED-MD values of
    the same vintage (N08 §C6: the loader should assert this for every vintage)."""
    out: dict[str, Any] = {}
    monthly = md_raw.copy()
    monthly.index = monthly.index.asfreq("M") if hasattr(monthly.index, "asfreq") else monthly.index
    quarter = monthly.index.asfreq("Q")
    counts = monthly.groupby(quarter).count()
    means = monthly.groupby(quarter).mean()
    for name in ("UNRATE", "TB3MS"):
        full = counts[name] == 3
        q = means.loc[full, name]
        common = q.index.intersection(qd_raw.index)
        diff = (qd_raw.loc[common, name] - q.loc[common]).abs()
        diff = diff[np.isfinite(diff)]
        out[name] = {
            "quarters": int(diff.size),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "first": str(diff.index[0]),
            "last": str(diff.index[-1]),
        }
        # FRED-QD rounds to the published precision: allow a rounding tolerance
        assert diff.size > 200 and float(diff.max()) < 0.01, (name, out[name])
    return out


def assert_in_window(model: BVAR, data: VARData, origins: Sequence[str]) -> dict[str, Any]:
    """R9 / §C6: the resolved prior and hyperparameters at an origin are identical when every
    observation after the origin is removed from the source (no look-ahead from the vintage)."""
    import jax

    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.settings import Runtime

    checked = []
    for o in origins:
        window = data.upto(o)
        cut = VARData.from_frame(data.raw.loc[: pd.Period(o, freq="Q")], data.tcodes)
        cut = cut.upto(o)
        assert str(window.last) == str(o) == str(cut.last)
        a, b = resolve(model, window, Runtime()), resolve(model, cut, Runtime())
        for x, y in zip(jax.tree.leaves(a.prior), jax.tree.leaves(b.prior), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(a.data), jax.tree.leaves(b.data), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert json.dumps(a.hyper, default=str, sort_keys=True) == json.dumps(
            b.hyper, default=str, sort_keys=True
        )
        checked.append(str(o))
    return {"origins": checked, "identical": True}


# --------------------------------------------------------------------------- models


def minnesota(form: str) -> Minnesota:
    """CR's Minnesota (WP App. 6.1): θ = 0.2, 1/l² lag decay, no cross shrinkage, zero means."""
    return Minnesota(form=form, hierarchy="fixed", lambda_=0.2, alpha=2.0, cross=1.0, mean=0.0)


def models(mcmc: MCMCSettings) -> dict[str, BVAR]:
    def bvar(vol, form="reduced"):
        return BVAR(mean=Linear(prior=minnesota(form)), vol=vol, lags=LAGS, mcmc=mcmc)

    return {
        "Constant": bvar(Constant("iw_equivalent")),
        "SV(rw)": bvar(SV(rw=True)),
        "SV": bvar(SV()),
        "SV(rw,t10)": bvar(
            SV(rw=True, tails="t", nu_prior=DiscreteUniform(10, 10), t_scale="unit")
        ),
        "AR-SV(0.9)": bvar(SV(phi_prior=Beta(71.25, 3.75))),
    }


def structural_models(mcmc: MCMCSettings) -> dict[str, BVAR]:
    def bvar(vol):
        return BVAR(mean=Linear(prior=minnesota("structural")), vol=vol, lags=LAGS, mcmc=mcmc)

    return {"Constant [structural]": bvar(Constant("ig")), "SV(rw) [structural]": bvar(SV(rw=True))}


PRIMARY_SV = "SV(rw)"
BASELINE = "Constant"


@dataclass(frozen=True)
class MomentsProbe:
    """Wraps a model; records the mean and covariance of every forecast's draws (the Gaussian
    approximation LPS of CR, §C4). Its spec adds ``"probe"`` so its cache entries are its own."""

    inner: Any
    store: str
    seed: int
    draws: int

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def axes(self) -> dict[str, str]:
        return self.inner.axes

    @property
    def uses_fit_seed(self) -> bool:
        return self.inner.uses_fit_seed

    @property
    def supports_warm_start(self) -> bool:
        return self.inner.supports_warm_start

    def spec(self) -> dict[str, Any]:
        return {**self.inner.spec(), "probe": "gaussian-moments-v1"}

    def directory(self) -> Path:
        tag = json.dumps([self.spec(), self.seed, self.draws], sort_keys=True, default=str)
        return Path(self.store) / hashlib.sha256(tag.encode()).hexdigest()[:24]

    def fit(self, data: VARData, **kwargs: Any) -> _ProbeFit:
        return _ProbeFit(self, self.inner.fit(data, **kwargs), str(data.last))

    def moments(self, origin: str) -> dict[str, np.ndarray] | None:
        path = self.directory() / f"{origin}.npz"
        if not path.exists():
            return None
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}


class _ProbeFit:
    def __init__(self, probe: MomentsProbe, fit: Any, origin: str):
        self._probe, self._fit, self._origin = probe, fit, origin

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fit, name)

    def forecast(self, horizon: int, draws: int, seed: Any = None) -> Any:
        fr = self._fit.forecast(horizon, draws, seed=seed)
        x = fr.stacked()  # (S, H, n)
        mean = x.mean(axis=0)
        cov = np.stack([np.cov(x[:, h, :], rowvar=False) for h in range(x.shape[1])])
        directory = self._probe.directory()
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / f"{self._origin}.npz", mean=mean, cov=cov, variables=np.array(fr.variables)
        )
        return fr


def gaussian_lps(probe: MomentsProbe, data: VARData, origins: Sequence[str]) -> dict[str, Any]:
    """Per origin and horizon: the joint and marginal Gaussian-approximation log scores."""
    H = max(HORIZONS)
    joint = np.full((len(origins), H), np.nan)
    marg = np.full((len(origins), H, len(VARS)), np.nan)
    for i, o in enumerate(origins):
        m = probe.moments(o)
        if m is None:
            raise RuntimeError(f"no forecast moments for {probe.name} at {o}")
        assert list(m["variables"]) == VARS
        truth = data.realized(o, H)
        for h in range(H):
            y, mu, V = truth[h], m["mean"][h], m["cov"][h]
            e = y - mu
            if np.isfinite(y).all():
                _, logdet = np.linalg.slogdet(V)
                joint[i, h] = -0.5 * (
                    len(y) * math.log(2 * math.pi) + logdet + e @ np.linalg.solve(V, e)
                )
            for j in range(len(VARS)):
                if np.isfinite(y[j]):
                    marg[i, h, j] = -0.5 * (math.log(2 * math.pi * V[j, j]) + e[j] ** 2 / V[j, j])
    return {"joint": joint, "marginal": marg}


# --------------------------------------------------------------------------- tables


def sliced(result: BacktestResult, origins: Sequence[str]) -> BacktestResult:
    return dataclasses.replace(result, scores=result.scores.sel(origin=list(origins)))


def relative_table(
    results: dict[str, BacktestResult],
    model: str,
    base: str,
    origins: Sequence[str],
    glps: dict[str, dict[str, Any]] | None = None,
    all_origins: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per variable × horizon: ΔLPS (mean), CRPS and RMSE ratios, one-sided DM/HLN p-values
    (log score), Gaussian-approx ΔLPS; joint ΔLPD; over ``origins``."""
    m = results[model].scores.sel(origin=list(origins))
    b = results[base].scores.sel(origin=list(origins))
    idx = None
    if glps is not None and all_origins is not None:
        pos = {o: i for i, o in enumerate(all_origins)}
        idx = [pos[o] for o in origins]
    out: dict[str, Any] = {"variables": {}, "joint": {}}
    for v_i, v in enumerate(VARS):
        rows = {}
        for h in HORIZONS:
            lm = m["lpd_marginal"].sel(variable=v, horizon=h).values
            lb = b["lpd_marginal"].sel(variable=v, horizon=h).values
            ok = np.isfinite(lm) & np.isfinite(lb)
            cm = m["crps"].sel(variable=v, horizon=h).values[ok]
            cb = b["crps"].sel(variable=v, horizon=h).values[ok]
            sm = m["sq_err"].sel(variable=v, horizon=h).values[ok]
            sb = b["sq_err"].sel(variable=v, horizon=h).values[ok]
            dm = diebold_mariano(-lm[ok], -lb[ok], h, alternative="less")
            d = lm[ok] - lb[ok]
            labels = np.asarray(list(origins))[ok]
            top = np.argsort(-np.abs(d))[:3]
            row = {
                "n": int(ok.sum()),
                "dlps": float(np.mean(d)),
                "dlps_median": float(np.median(d)),
                "dlps_top": [[str(labels[i]), float(d[i])] for i in top],
                "crps_ratio": float(cm.mean() / cb.mean()),
                "rmse_ratio": float(math.sqrt(sm.mean()) / math.sqrt(sb.mean())),
                "dm_p_better": float(dm.pvalue),
                "lps_model": float(np.mean(lm[ok])),
                "lps_base": float(np.mean(lb[ok])),
                "rmse_base": float(math.sqrt(sb.mean())),
            }
            if idx is not None:
                gm = glps[model]["marginal"][idx, h - 1, v_i]
                gb = glps[base]["marginal"][idx, h - 1, v_i]
                okg = np.isfinite(gm) & np.isfinite(gb)
                row["dlps_gauss"] = float(np.mean(gm[okg] - gb[okg]))
            rows[h] = row
        out["variables"][v] = rows
    for h in HORIZONS:
        jm = m["lpd"].sel(horizon=h).values
        jb = b["lpd"].sel(horizon=h).values
        ok = np.isfinite(jm) & np.isfinite(jb)
        dm = diebold_mariano(-jm[ok], -jb[ok], h, alternative="less")
        row = {
            "n": int(ok.sum()),
            "dlpd": float(np.mean(jm[ok] - jb[ok])),
            "dm_p_better": float(dm.pvalue),
        }
        if idx is not None:
            gm = glps[model]["joint"][idx, h - 1]
            gb = glps[base]["joint"][idx, h - 1]
            okg = np.isfinite(gm) & np.isfinite(gb)
            row["dlpd_gauss"] = float(np.mean(gm[okg] - gb[okg]))
        out["joint"][h] = row
    return out


def verdicts(
    results: dict[str, BacktestResult],
    mods: dict[str, Any],
    names: Sequence[str],
    base: str,
    origins: Sequence[str],
) -> list[dict[str, Any]]:
    """``compare()`` verdicts of every model in ``names`` against ``base`` (SPEC §9.5), as
    returned (convergence gate on) and with the gate off. The ``CompareResult`` is built from the
    backtests already run (``compare()`` itself would call the same ``backtest``s)."""
    res = {n: sliced(results[n], origins) for n in [base, *names]}
    entries = [Entry(n, mods[n], dict(res[n].axes)) for n in res]
    cr = CompareResult(entries, res, list(HORIZONS), base, Expanding(), "Q-DEC")
    rows = []
    for gated in (True, False):
        for rec in _verdict_table(cr, min_converged=0.95 if gated else 0.0):
            rec["convergence_gate"] = gated
            rows.append(rec)
    return rows


def _verdict_table(cr: CompareResult, min_converged: float) -> list[dict]:
    frame = cr.verdict(min_converged=min_converged)
    cols = [
        "model",
        "reference",
        "axis",
        "horizon",
        "verdict",
        "dlpd",
        "dlpd_mcse",
        "dm_pvalue_better",
        "model_in_mcs",
        "reference_in_mcs",
        "reasons",
    ]
    return [{c: _plain(r.get(c)) for c in cols} for r in frame.to_dict("records")]


def _plain(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def converged_share(result: BacktestResult, origins: Sequence[str]) -> float:
    return float(result.scores["converged"].sel(origin=list(origins)).mean())


# --------------------------------------------------------------------------- criteria


def _c4_secondary(full: dict[str, Any], pre: dict[str, Any]) -> str:
    """Criterion 4 seen through the median per-origin ΔLPS, the dominant origins and CR's
    Gaussian-approximation score (information only: the verdict uses the primary RB LPD)."""
    parts = []
    for v in ("UR", "TBILL"):
        r1, r8 = full["variables"][v][1], full["variables"][v][8]
        top = ", ".join(f"{o} {d:+.1f}" for o, d in r8["dlps_top"])
        parts.append(
            f"full {v} h8: median {r8['dlps_median']:+.3f} (largest origins: {top})"
            + (
                f", Gaussian-approx h1 {r1['dlps_gauss']:+.3f} → h8 {r8['dlps_gauss']:+.3f}"
                if "dlps_gauss" in r8
                else ""
            )
        )
    for v in ("GDP", "INFL"):
        r8 = pre["variables"][v][8]
        if "dlps_gauss" in r8:
            parts.append(f"pre-crisis {v} h8 Gaussian-approx {r8['dlps_gauss']:+.3f}")
    return ". Secondary (not part of the rule): " + "; ".join(parts)


def criteria(rep: dict[str, Any]) -> list[dict[str, Any]]:
    full = rep["relative"]["full"][PRIMARY_SV]
    pre = rep["relative"]["pre_crisis"][PRIMARY_SV]
    out = []

    # 1. ΔLPS > 0 at h = 1, 2 for all four variables in both samples; joint gain at h = 1 clearly
    # positive (operational: mean joint ΔLPD > 0 with one-sided DM/HLN p < 0.10)
    cells = {
        (s, v, h): t["variables"][v][h]["dlps"]
        for s, t in (("full", full), ("pre", pre))
        for v in VARS
        for h in (1, 2)
    }
    fails = [f"{s}/{v}/h{h}: {d:+.3f}" for (s, v, h), d in cells.items() if not d > 0]
    joint = {s: t["joint"][1] for s, t in (("full", full), ("pre", pre))}
    joint_ok = all(j["dlpd"] > 0 and j["dm_p_better"] < 0.10 for j in joint.values())
    out.append(
        {
            "id": 1,
            "met": not fails and joint_ok,
            "rule": "ΔLPS(SV(rw) − Constant) > 0 at h = 1, 2 for all four variables in the full and "
            "pre-crisis samples, and the joint ΔLPD at h = 1 > 0 with one-sided DM/HLN "
            "p < 0.10 in both samples",
            "detail": (
                f"{len(cells) - len(fails)} of {len(cells)} cells positive"
                + (f"; failing: {', '.join(fails)}" if fails else "")
                + "; joint h=1: "
                + ", ".join(
                    f"{s} {j['dlpd']:+.3f} nats/quarter (p={j['dm_p_better']:.3f})"
                    for s, j in joint.items()
                )
            ),
        }
    )

    # 2. CRPS ratio < 1 at h ≤ 2 for all four variables (full sample; pre-crisis reported)
    c2 = {(v, h): full["variables"][v][h]["crps_ratio"] for v in VARS for h in (1, 2)}
    fails2 = [f"{v}/h{h}: {r:.3f}" for (v, h), r in c2.items() if not r < 1]
    pre2 = [
        f"{v}/h{h}: {pre['variables'][v][h]['crps_ratio']:.3f}"
        for v in VARS
        for h in (1, 2)
        if not pre["variables"][v][h]["crps_ratio"] < 1
    ]
    out.append(
        {
            "id": 2,
            "met": not fails2,
            "rule": "CRPS ratio SV(rw)/Constant < 1 at h = 1, 2 for all four variables (full sample)",
            "detail": f"{8 - len(fails2)} of 8 cells < 1"
            + (f"; failing: {', '.join(fails2)}" if fails2 else "")
            + (f"; pre-crisis cells ≥ 1: {', '.join(pre2)}" if pre2 else "; pre-crisis: all 8 < 1"),
        }
    )

    # 3. RMSE ratios ≤ about 1 at short horizons (operational: ≤ 1.02 at h = 1, 2)
    c3 = {(v, h): full["variables"][v][h]["rmse_ratio"] for v in VARS for h in (1, 2)}
    fails3 = [f"{v}/h{h}: {r:.3f}" for (v, h), r in c3.items() if not r <= 1.02]
    out.append(
        {
            "id": 3,
            "met": not fails3,
            "rule": "RMSE ratio SV(rw)/Constant ≤ 1.02 ('about 1') at h = 1, 2 for all four "
            "variables (full sample); CR's gains for GDP and the bill are 5–10%",
            "detail": ", ".join(f"{v}/h{h} {r:.3f}" for (v, h), r in c3.items())
            + (f"; failing: {', '.join(fails3)}" if fails3 else ""),
        }
    )

    # 4. full sample: h = 8 gain shrinks (below its h = 1 value) or turns negative for UR and the
    # bill; pre-crisis: the h = 8 gain stays positive for GDP and inflation
    shrink = {
        v: (full["variables"][v][1]["dlps"], full["variables"][v][8]["dlps"])
        for v in ("UR", "TBILL")
    }
    keep = {v: pre["variables"][v][8]["dlps"] for v in ("GDP", "INFL")}
    ok4 = all(d8 < d1 for d1, d8 in shrink.values()) and all(d > 0 for d in keep.values())
    out.append(
        {
            "id": 4,
            "met": ok4,
            "rule": "full sample: ΔLPS(h=8) < ΔLPS(h=1) for UR and TBILL; pre-crisis: ΔLPS(h=8) > 0 "
            "for GDP and INFL",
            "detail": "; ".join(
                f"full {v}: h1 {a:+.3f} → h8 {b:+.3f}" for v, (a, b) in shrink.items()
            )
            + "; "
            + "; ".join(f"pre-crisis {v} h8 {d:+.3f}" for v, d in keep.items())
            + _c4_secondary(full, pre),
        }
    )

    # 5. neither SV(t) nor AR-SV systematically beats RW/AR SV (operational: the variant is not
    # better than BOTH SV(rw) and SV in a majority (≥ 3 of 4) of joint-LPD horizons with
    # one-sided DM p < 0.10, and not in ≥ 8 of the 16 marginal cells)
    det5, ok5 = [], True
    for variant in ("SV(rw,t10)", "AR-SV(0.9)"):
        joint_wins, cell_wins = 0, 0
        for h in HORIZONS:
            if all(
                rep["variants"][variant][ref]["joint"][h]["dm_p_better"] < 0.10
                and rep["variants"][variant][ref]["joint"][h]["dlpd"] > 0
                for ref in ("SV(rw)", "SV")
            ):
                joint_wins += 1
            for v in VARS:
                if all(
                    rep["variants"][variant][ref]["variables"][v][h]["dm_p_better"] < 0.10
                    and rep["variants"][variant][ref]["variables"][v][h]["dlps"] > 0
                    for ref in ("SV(rw)", "SV")
                ):
                    cell_wins += 1
        systematic = joint_wins >= 3 or cell_wins >= 8
        ok5 &= not systematic
        det5.append(
            f"{variant}: significantly better than both SV(rw) and SV at {joint_wins}/4 "
            f"joint horizons and {cell_wins}/16 marginal cells"
        )
    out.append(
        {
            "id": 5,
            "met": ok5,
            "rule": "neither SV(rw,t10) nor AR-SV(0.9) is better than both SV(rw) and SV (one-sided "
            "DM/HLN p < 0.10, positive ΔLPD) at ≥ 3 of 4 joint horizons or ≥ 8 of 16 "
            "marginal cells (full sample)",
            "detail": "; ".join(det5),
        }
    )

    # 6. compare() verdict "better" at h = 1 and one-sided DM p < 0.10 for GDP and TBILL at h = 1,
    # in all 24 orderings
    ords = rep.get("orderings")
    if ords is None:
        out.append(
            {"id": 6, "met": None, "rule": "24 orderings", "detail": "ordering stage skipped"}
        )
    else:
        rows = ords["per_ordering"]
        gated = [r for r in rows if not r["verdict_h1"] == "better"]
        ungated = [r for r in rows if not r["verdict_h1_ungated"] == "better"]
        dm_fail = [r for r in rows if not (r["dm_p"]["GDP"] < 0.10 and r["dm_p"]["TBILL"] < 0.10)]
        met = not ungated and not dm_fail and not gated
        out.append(
            {
                "id": 6,
                "met": met,
                "rule": "in every one of the 24 orderings: compare() verdict SV(rw) vs Constant at h = 1 "
                "is 'better', and the one-sided DM/HLN p-value of the marginal log score is "
                "< 0.10 for GDP and TBILL at h = 1",
                "detail": (
                    f"verdict 'better' (convergence gate on, as compare() returns it) in "
                    f"{len(rows) - len(gated)}/{len(rows)} orderings; with the gate off in "
                    f"{len(rows) - len(ungated)}/{len(rows)}; DM GDP and TBILL < 0.10 in "
                    f"{len(rows) - len(dm_fail)}/{len(rows)}"
                    + (
                        f"; DM failures: {', '.join(r['label'] for r in dm_fail)}"
                        if dm_fail
                        else ""
                    )
                    + (
                        f"; ungated verdict failures: "
                        f"{', '.join(r['label'] + ' (' + r['verdict_h1_ungated'] + ')' for r in ungated)}"
                        if ungated
                        else ""
                    )
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- ordering stage


def ordering_stage(
    mods: dict[str, BVAR],
    data: VARData,
    origins: list[str],
    cfg: Config,
    seed: int,
    cache: Cache,
    log,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """All 24 orderings for the primary SV model and the baseline (N08 §B4, §C5 criterion 6)."""
    out: dict[str, Any] = {"origins": origins, "results": {}}
    sens = {}
    for name in (BASELINE, PRIMARY_SV):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = rb.ordering_sensitivity(
                mods[name],
                data,
                n_perm=24,
                design="all",
                mode="backtest",
                horizons=HORIZONS,
                origins=origins,
                noise_reps=cfg.noise_reps,
                draws=cfg.draws,
                seed=seed,
                cache=cache,
                n_jobs=n_jobs,
            )
        sens[name] = res
        log(f"ordering stage {name}: {time.time() - t0:.0f}s, material={res.material}")
        summary = res.summary()
        out["results"][name] = {
            "material": bool(res.material),
            "summary": json.loads(summary.reset_index().to_json(orient="records")),
            "joint_lpd_range": {
                int(h): float(np.ptp(res.dataset["joint"].sel(horizon=h, jstat="lpd").values))
                for h in HORIZONS
            },
            "converged_orderings": int(res.dataset["converged"].sum()),
        }
    # per ordering: rebuild both backtests from the cache (identical calls) and compare
    seeds = json.loads(sens[PRIMARY_SV].dataset.attrs["backtest_seeds"])
    rows = []
    for j, order in enumerate(sens[PRIMARY_SV].orderings):
        results, used = {}, {}
        for name in (BASELINE, PRIMARY_SV):
            m = permute_model(mods[name], order, variables=data.variables)
            m = dataclasses.replace(m, mcmc=dataclasses.replace(m.mcmc, seed=seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bt = backtest(
                    m,
                    data.select(order),
                    origins=origins,
                    horizon=max(HORIZONS),
                    draws=cfg.draws,
                    seed=seeds[j],
                    cache=cache,
                    n_jobs=1,  # every origin is cached by now: no worker pool
                    name=name,
                )
            if bt.cached != len(origins):
                log(f"warning: ordering {j} {name}: {len(origins) - bt.cached} origins not cached")
            results[name], used[name] = bt, m
        v = verdicts(results, used, [PRIMARY_SV], BASELINE, origins)
        h1 = {r["convergence_gate"]: r for r in v if r["horizon"] == 1}
        rel = relative_table(results, PRIMARY_SV, BASELINE, origins)
        rows.append(
            {
                "ordering": j,
                "label": " > ".join(order),
                "verdict_h1": h1[True]["verdict"],
                "verdict_h1_ungated": h1[False]["verdict"],
                "dlpd_h1": rel["joint"][1]["dlpd"],
                "dm_p_joint_h1": rel["joint"][1]["dm_p_better"],
                "dm_p": {var: rel["variables"][var][1]["dm_p_better"] for var in VARS},
                "dlps_h1": {var: rel["variables"][var][1]["dlps"] for var in VARS},
                "converged_sv": converged_share(results[PRIMARY_SV], origins),
                "converged_constant": converged_share(results[BASELINE], origins),
                "verdicts": v,
            }
        )
    out["per_ordering"] = rows
    return out


# --------------------------------------------------------------------------- report


def write_report(rep: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=_json_default))
    out.with_suffix(".md").write_text(markdown(rep))


def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _f(x: Any, fmt: str = "+.3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def markdown(rep: dict[str, Any]) -> str:
    cfg = rep["config"]
    L: list[str] = []
    add = L.append
    title = "# Clark–Ravazzolo (2015) replication on FRED-QD (WP23)"
    if cfg["name"] != "full":
        title += f" — {cfg['name']}, NOT the acceptance run"
    add(title + "\n")
    if cfg["name"] != "full":
        add(
            f"> **Config `{cfg['name']}`: this is not the acceptance run.** The acceptance run is "
            "`--config full` (4 chains × 5000 draws, all 24 orderings); see the module docstring "
            "of `benchmarks/clark_ravazzolo.py` for the command. "
            + (
                "This report has no 24-ordering stage, so criterion 6 is not evaluated. "
                if not rep.get("orderings")
                else ""
            )
            + "".join(f"{note} " for note in rep.get("notes", []))
            + "\n"
        )
    add(
        f"Generated {rep['generated']} by `benchmarks/clark_ravazzolo.py` (config "
        f"`{cfg['name']}`, rai-bvar {rep['versions']['rai_bvar']}, jax {rep['versions']['jax']}, "
        f"{rep['versions']['platform']}). Run time of this invocation {rep['seconds'] / 3600:.2f} h "
        "(cached origins are read, not refitted).\n"
    )
    add("## Acceptance criteria (N08 §C5)\n")
    add("| # | Met | Operational rule | Result |")
    add("| :- | :- | :- | :- |")
    for c in rep["criteria"]:
        met = {True: "**met**", False: "**not met**", None: "not run"}[c["met"]]
        add(f"| {c['id']} | {met} | {c['rule']} | {c['detail']} |")
    add("")
    add("## Design\n")
    add(
        f"* Data: FRED-QD {VINTAGE} (SHA-256 verified), GDP = 100·Δlog GDPC1, INFL = 100·Δlog "
        "GDPCTPI, UR = UNRATE, TBILL = TB3MS (levels); non-annualised quarterly rates; estimation "
        f"sample starts {ESTIMATION_START}; pseudo real time on the final vintage (D4)."
    )
    agg = rep["data_checks"]["aggregation"]
    add(
        "* Asserted (§C6): FRED-QD UNRATE/TB3MS equal within-quarter means of FRED-MD monthly "
        "values of the same vintage — "
        + "; ".join(
            f"{k}: max |diff| {v['max_abs_diff']:.4f}, mean {v['mean_abs_diff']:.5f} over "
            f"{v['quarters']} quarters ({v['first']}–{v['last']})"
            for k, v in agg.items()
        )
        + "."
    )
    add(
        "* Asserted (R9, §C6): at origins "
        f"{', '.join(rep['data_checks']['in_window']['origins'])} the resolved priors, "
        "hyperparameters and padded data are bitwise identical when the vintage is cut at the "
        "origin (no look-ahead; CR's last-vintage SV training sample is not replicated)."
    )
    add(
        f"* Origins: {FULL[0]}–{FULL[1]} ({rep['n_origins']['full']}, full) and "
        f"{FULL[0]}–{PRE_CRISIS_END} ({rep['n_origins']['pre_crisis']}, pre-crisis subset); "
        "expanding window; h = 1, 2, 4, 8."
    )
    m = cfg["mcmc"]
    add(
        f"* MCMC: {m['chains']} chains × {m['draws']} kept draws, burn {m['burn']}, thin "
        f"{m['thin']}; {cfg['draws']} forecast draws per origin; backtest seed {cfg['seed']}; "
        "per-origin fit seeds (DESIGN §11.6); cold starts."
    )
    add(
        "* Mean prior (all models): `Minnesota(form='reduced', hierarchy='fixed', lambda_=0.2, "
        "alpha=2, cross=1, mean=0)`, p = 4. Volatility arms: "
        + ", ".join(f"`{k}` = `{v}`" for k, v in rep["models"].items())
        + "."
    )
    add(
        "* LPD is the library's Rao–Blackwellised predictive density (primary); `ΔLPS gauss` is "
        "CR's Gaussian approximation (Adolfson et al. 2007) from the same draws.\n"
    )
    for sample in ("full", "pre_crisis"):
        add(
            f"## {'Full sample' if sample == 'full' else 'Pre-crisis'}: "
            f"each SV arm vs `{BASELINE}`\n"
        )
        for model, t in rep["relative"][sample].items():
            conv = rep["converged"][sample]
            add(
                f"### {model} (converged at {conv[model]:.0%} of origins; baseline "
                f"{conv[BASELINE]:.0%})\n"
            )
            add(
                "| variable | h | ΔLPS | ΔLPS gauss | DM p (1-sided) | CRPS ratio | RMSE ratio |"
                " CR ΔLPS | CR CRPS | CR RMSE |"
            )
            add("| :- | -: | -: | -: | -: | -: | -: | -: | -: | -: |")
            for v in VARS:
                for k, h in enumerate(HORIZONS):
                    r = t["variables"][v][h]
                    if model == PRIMARY_SV and sample == "full":
                        cr = (
                            f"{CR_FULL['dlps'][v][k]:+.3f} | {CR_FULL['crps_ratio'][v][k]:.3f} | "
                            f"{CR_FULL['rmse_ratio'][v][k]:.3f}"
                        )
                    elif model == PRIMARY_SV and h in (1, 8):
                        cr = f"{CR_PRE_DLPS[v][0 if h == 1 else 1]:+.3f} | | "
                    else:
                        cr = " | | "
                    add(
                        f"| {v} | {h} | {_f(r['dlps'])} | {_f(r.get('dlps_gauss'))} | "
                        f"{_f(r['dm_p_better'], '.3f')} | {_f(r['crps_ratio'], '.3f')} | "
                        f"{_f(r['rmse_ratio'], '.3f')} | {cr} |"
                    )
            add("")
            add("| joint | h | ΔLPD | ΔLPD gauss | DM p (1-sided) |")
            add("| :- | -: | -: | -: | -: |")
            for h in HORIZONS:
                r = t["joint"][h]
                add(
                    f"| joint | {h} | {_f(r['dlpd'])} | {_f(r.get('dlpd_gauss'))} | "
                    f"{_f(r['dm_p_better'], '.3f')} |"
                )
            add("")
        add(f"### `compare()` verdicts vs `{BASELINE}` ({sample.replace('_', '-')})\n")
        add(
            "| model | h | verdict (gate on) | verdict (gate off) | ΔLPD total | MC s.e. | "
            "DM p | model in MCS25 | baseline in MCS25 |"
        )
        add("| :- | -: | :- | :- | -: | -: | -: | :- | :- |")
        vs = rep["verdicts"][sample]
        for r in [x for x in vs if x["convergence_gate"]]:
            off = next(
                x
                for x in vs
                if not x["convergence_gate"]
                and x["model"] == r["model"]
                and x["horizon"] == r["horizon"]
            )
            add(
                f"| {r['model']} | {r['horizon']} | {r['verdict']} | {off['verdict']} | "
                f"{_f(r['dlpd'], '+.1f')} | {_f(r['dlpd_mcse'], '.2f')} | "
                f"{_f(r['dm_pvalue_better'], '.3f')} | {r['model_in_mcs']} | "
                f"{r['reference_in_mcs']} |"
            )
        add("")
    add("## Variants vs RW and AR SV (criterion 5, full sample)\n")
    add("| variant | reference | h | joint ΔLPD | DM p (variant better) |")
    add("| :- | :- | -: | -: | -: |")
    for variant, refs in rep["variants"].items():
        for ref, t in refs.items():
            for h in HORIZONS:
                add(
                    f"| {variant} | {ref} | {h} | {_f(t['joint'][h]['dlpd'])} | "
                    f"{_f(t['joint'][h]['dm_p_better'], '.3f')} |"
                )
    add("")
    if rep.get("structural"):
        add("## Structural-form pair (robustness, full sample)\n")
        t = rep["structural"]
        add("| variable | h=1 ΔLPS | h=2 ΔLPS | h=8 ΔLPS |")
        add("| :- | -: | -: | -: |")
        for v in VARS:
            add(
                f"| {v} | {_f(t['variables'][v][1]['dlps'])} | "
                f"{_f(t['variables'][v][2]['dlps'])} | {_f(t['variables'][v][8]['dlps'])} |"
            )
        add(
            f"| joint | {_f(t['joint'][1]['dlpd'])} | {_f(t['joint'][2]['dlpd'])} | "
            f"{_f(t['joint'][8]['dlpd'])} |\n"
        )
    ords = rep.get("orderings")
    if ords:
        add("## All 24 orderings (criterion 6)\n")
        add(
            f"Origins in this stage: {len(ords['origins'])} ({ords['origins'][0]}–"
            f"{ords['origins'][-1]}{', every ' + str(cfg['ordering_step']) + ' quarters' if cfg['ordering_step'] > 1 else ''}). "
            "Each ordering's SV(rw) and Constant backtests are compared with `compare()`; the "
            "Constant arm is order-invariant in theory (Chan 2022), so its spread is Monte "
            "Carlo noise.\n"
        )
        add(
            "| # | ordering | verdict h=1 (gate on) | gate off | joint ΔLPD h=1 | DM p joint | "
            "DM p GDP | DM p TBILL | SV converged | Const converged |"
        )
        add("| -: | :- | :- | :- | -: | -: | -: | -: | -: | -: |")
        for r in ords["per_ordering"]:
            add(
                f"| {r['ordering']} | {r['label']} | {r['verdict_h1']} | {r['verdict_h1_ungated']} "
                f"| {_f(r['dlpd_h1'])} | {_f(r['dm_p_joint_h1'], '.3f')} | "
                f"{_f(r['dm_p']['GDP'], '.3f')} | {_f(r['dm_p']['TBILL'], '.3f')} | "
                f"{r['converged_sv']:.0%} | {r['converged_constant']:.0%} |"
            )
        add("")
        for name, res in ords["results"].items():
            add(
                f"`ordering_sensitivity` for `{name}`: material = {res['material']}; joint "
                "LPD range over orderings (nats per origin): "
                + ", ".join(f"h={h}: {v:.3f}" for h, v in res["joint_lpd_range"].items())
                + ".\n"
            )
    add("## Deviations and caveats\n")
    for d in rep["deviations"]:
        add(f"* {d}")
    add("")
    add("## Warnings raised during the run (aggregated)\n")
    add(
        "Messages that differ only in numbers or variable lists are grouped; one example is shown.\n"
    )
    for name, groups in rep["warnings"].items():
        for g in groups.values():
            add(
                f"* `{name}`, {g['origin_messages']} origin-messages ({g['variants']} variants), "
                f"e.g.: {g['example'][:300]}"
            )
    add("")
    return "\n".join(L)


def group_warnings(by_message: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Backtest warnings (message → origins) grouped by the message with numbers stripped."""
    groups: dict[str, dict[str, Any]] = {}
    for msg, origins in by_message.items():
        key = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", msg)
        key = re.sub(r"\[[^\]]*\]", "[…]", key).split(":")[0]
        g = groups.setdefault(key, {"origin_messages": 0, "variants": 0, "example": msg})
        g["origin_messages"] += len(origins)
        g["variants"] += 1
    return groups


DEVIATIONS = [
    "Not a deviation from CR, but from the library default: every own-lag prior mean is 0, as in "
    "CR (WP App. 6.1), including the UR and TBILL levels; the library warns that this contradicts "
    "their level transformation codes (it would centre them on 1).",
    "Pseudo real time on the final FRED-QD 2026-09 vintage instead of CR's RTDSM real-time "
    "vintages and second-release actuals (D4; §C2). Score levels differ; rankings are expected to "
    "be robust (§C6).",
    "Estimation starts in 1965Q1, not 1955Q1 (FRED-QD starts in 1959; §C2).",
    "Inflation uses the chain-type GDPCTPI (FRED-QD has no GDPDEF; §C6).",
    "Baseline: Constant('iw_equivalent') (Chan 2022) stands in for CR's normal-diffuse VAR "
    "(§C3); the SV arms use the library's default SV priors except the two CR variants (no prior "
    "was tuned). CR's RW-SV σ² prior IG(1/2, 0.0175) and its 36-observation training prior for "
    "h₀ are not used (the latter would violate R9, §C6).",
    "AR-SV variant: CR's a₁ ~ N(0.9, 0.05²) is mapped to (φ+1)/2 ~ Beta(71.25, 3.75) (same mean "
    "and sd on φ); the SV level μ keeps the library's in-window data-centred prior instead of CR's "
    "a₀ ~ N(0, 0.4²).",
    "LPD: the primary score is the Rao–Blackwellised predictive density; CR's Gaussian "
    "approximation is reported alongside (§C4, §C6).",
    "Significance: one-sided DM/HLN p-values are descriptive under an expanding window; "
    "compare() therefore decides 'better' through the Model Confidence Set (SPEC §9.5).",
    "compare() turns a verdict 'inconclusive' when fewer than 95% of origins converged (R-hat/ESS "
    "targets of DESIGN §8.2); verdicts are reported with that gate on (as returned) and off.",
]


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path, help="report path without suffix (writes .md and .json)")
    p.add_argument("--config", choices=sorted(CONFIGS), default="full")
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "rai-bvar-cr")
    p.add_argument("--qd-url", default=None)
    p.add_argument("--md-url", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="origins fitted at a time in spawn-started worker processes (DESIGN §11.6)",
    )
    p.add_argument("--skip-orderings", action="store_true")
    p.add_argument("--note", action="append", default=[], help="a sentence added to the report")
    p.add_argument("--skip-structural", action="store_true")
    p.add_argument(
        "--only-ordering",
        choices=[BASELINE, PRIMARY_SV],
        default=None,
        help="only run the 24-ordering backtests of this model (fills the cache; run it in a "
        "parallel process, then the full command assembles the report from the cache)",
    )
    args = p.parse_args(argv)
    cfg = CONFIGS[args.config]
    t_start = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t_start:8.0f}s] {msg}", flush=True)

    if args.n_jobs == 0 or args.n_jobs < -1:
        p.error("--n-jobs must be a positive int or -1 (all cores)")
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        # set before JAX is imported, so spawned workers (which inherit the environment) share
        # the persistent compilation cache with this process
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.cache / "xla")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        rb.enable_compilation_cache(args.cache / "xla", min_compile_secs=0)
    import jax

    n_jobs = args.n_jobs
    data, aggregation = load_data(args.qd_url, args.md_url)
    log(f"data {data}")
    all_origins = [str(o) for o in Origins(*FULL).resolve(data.index)][:: cfg.origin_step]
    pre = [o for o in all_origins if pd.Period(o, freq="Q") <= pd.Period(PRE_CRISIS_END, freq="Q")]
    mods = models(cfg.mcmc)
    cache = Cache(args.cache)
    if args.only_ordering:
        ord_origins = all_origins[:: cfg.ordering_step]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rb.ordering_sensitivity(
                mods[args.only_ordering],
                data,
                n_perm=24,
                design="all",
                mode="backtest",
                horizons=HORIZONS,
                origins=ord_origins,
                noise_reps=cfg.noise_reps,
                draws=cfg.draws,
                seed=args.seed,
                cache=cache,
                n_jobs=n_jobs,
            )
        log(f"ordering backtests of {args.only_ordering} cached")
        return 0
    in_window = assert_in_window(mods[PRIMARY_SV], data, ["1985Q1", "2007Q4", "2011Q2"])
    log("data checks passed")
    store = str(args.cache / "moments")

    results: dict[str, BacktestResult] = {}
    probes: dict[str, MomentsProbe] = {}
    warn_counts: dict[str, dict[str, dict[str, Any]]] = {}
    for name, model in mods.items():
        probe = MomentsProbe(model, store, args.seed, cfg.draws)
        probes[name] = probe
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = backtest(
                probe,
                data,
                origins=all_origins,
                horizon=max(HORIZONS),
                draws=cfg.draws,
                seed=args.seed,
                cache=cache,
                n_jobs=n_jobs,
                name=name,
            )
        results[name] = res
        warn_counts[name] = group_warnings(res.warnings)
        log(
            f"{name}: {time.time() - t0:.0f}s ({res.cached} cached), "
            f"converged {converged_share(res, all_origins):.0%}"
        )
    glps = {n: gaussian_lps(p_, data, all_origins) for n, p_ in probes.items()}

    rep: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "config": {
            "name": cfg.name,
            "mcmc": dataclasses.asdict(cfg.mcmc),
            "draws": cfg.draws,
            "ordering_step": cfg.ordering_step,
            "noise_reps": cfg.noise_reps,
            "seed": args.seed,
            "n_jobs": n_jobs,
        },
        "versions": {
            "rai_bvar": rb.__version__,
            "jax": jax.__version__,
            "platform": f"{platform.system()} {platform.machine()}, "
            f"python {sys.version.split()[0]}",
        },
        "models": {
            k: v.vol.label if hasattr(v.vol, "label") else str(v.vol) for k, v in mods.items()
        },
        "data_checks": {"aggregation": aggregation, "in_window": in_window},
        "n_origins": {"full": len(all_origins), "pre_crisis": len(pre)},
        "relative": {},
        "verdicts": {},
        "converged": {},
        "variants": {},
        "fit_seed_example": {o: fit_seed(args.seed, o) for o in all_origins[:2]},
        "deviations": list(DEVIATIONS),
        "notes": list(args.note),
        "warnings": warn_counts,
    }
    sv_names = [n for n in mods if n != BASELINE]
    for sample, origins in (("full", all_origins), ("pre_crisis", pre)):
        rep["relative"][sample] = {
            n: relative_table(results, n, BASELINE, origins, glps, all_origins) for n in sv_names
        }
        rep["verdicts"][sample] = verdicts(results, probes, sv_names, BASELINE, origins)
        rep["converged"][sample] = {n: converged_share(r, origins) for n, r in results.items()}
    for variant in ("SV(rw,t10)", "AR-SV(0.9)"):
        rep["variants"][variant] = {
            ref: relative_table(results, variant, ref, all_origins) for ref in ("SV(rw)", "SV")
        }
    if not args.skip_structural:
        smods = structural_models(cfg.mcmc)
        sres = {}
        for name, model in smods.items():
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sres[name] = backtest(
                    model,
                    data,
                    origins=all_origins,
                    horizon=max(HORIZONS),
                    draws=cfg.draws,
                    seed=args.seed,
                    cache=cache,
                    n_jobs=n_jobs,
                    name=name,
                )
            log(f"{name}: {time.time() - t0:.0f}s")
        rep["structural"] = relative_table(
            sres, "SV(rw) [structural]", "Constant [structural]", all_origins
        )
    write_report_partial = dict(rep)
    write_report_partial["criteria"] = criteria(rep)
    write_report_partial["seconds"] = time.time() - t_start
    write_report(write_report_partial, args.out)  # a usable report before the long stage
    log(f"main stage written to {args.out}.md")
    if not args.skip_orderings:
        ord_origins = all_origins[:: cfg.ordering_step]
        if cfg.ordering_step > 1:
            rep["deviations"].append(
                f"Config '{cfg.name}': the 24-ordering stage uses every {cfg.ordering_step}-th "
                f"origin ({len(ord_origins)} of {len(all_origins)})."
            )
        rep["orderings"] = ordering_stage(
            mods, data, ord_origins, cfg, args.seed, cache, log, n_jobs
        )
    rep["criteria"] = criteria(rep)
    rep["seconds"] = time.time() - t_start
    write_report(rep, args.out)
    log(f"done: {args.out}.md")
    for c in rep["criteria"]:
        log(f"criterion {c['id']}: {c['met']} — {c['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
