"""Model comparison and the verdict engine (SPEC §9.5).

``compare`` backtests every model on the same origins, window and outcomes. ``table`` reports
the ablation table; ``verdict`` applies the rule of SPEC §9.5: a model beats a reference only if
the two differ along exactly one axis (R2), the LPD gain is large relative to its Monte Carlo
error, and either a valid (rolling-window) DM/AG-type test rejects or the Model Confidence Set
separates them.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from rai_bvar.data.dataset import VARData
from rai_bvar.evaluation.backtest import (
    BacktestResult,
    Cache,
    DataSource,
    Expanding,
    Origins,
    Window,
    backtest,
)
from rai_bvar.evaluation.significance import diebold_mariano, model_confidence_set
from rai_bvar.models.base import Model


@dataclass(frozen=True)
class Entry:
    """A model in a comparison, with its name and its labels along the ablation axes."""

    name: str
    model: Model
    axes: Mapping[str, str] = field(default_factory=dict)


def grid(factory: Callable[..., Model], **axes: Sequence[Any]) -> list[Entry]:
    """Cartesian product of component choices, e.g. ``grid(BVAR, mean=[...], vol=[...])``."""
    keys = list(axes)
    entries = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        kwargs = dict(zip(keys, combo, strict=True))
        labels = {k: _label(v) for k, v in kwargs.items()}
        entries.append(Entry(" × ".join(labels.values()), factory(**kwargs), labels))
    return entries


def compare(
    models: Mapping[str, Model] | Sequence[Model | Entry],
    data: DataSource,
    *,
    origins: Origins | Sequence[str | pd.Period],
    window: Window = Expanding(),
    horizons: Sequence[int] = (1,),
    draws: int = 1000,
    seed: int = 0,
    baseline: str | None = None,
    cache: Cache | str | os.PathLike | None = None,
    n_jobs: int = 1,
    realized: VARData | None = None,
) -> CompareResult:
    """Backtest every model on identical origins, windows and outcomes."""
    entries = _entries(models)
    names = [e.name for e in entries]
    baseline = baseline or names[0]
    if baseline not in names:
        raise ValueError(f"baseline {baseline!r} is not one of {names}")
    horizons = sorted({int(h) for h in horizons})
    results = {
        e.name: backtest(
            e.model,
            data,
            origins=origins,
            window=window,
            horizon=max(horizons),
            draws=draws,
            seed=seed,
            cache=cache,
            n_jobs=n_jobs,
            realized=realized,
            name=e.name,
            axes=dict(e.axes),
        )
        for e in entries
    }
    freq = (realized if realized is not None else data).index.freq
    return CompareResult(entries, results, horizons, baseline, window, freq)


class CompareResult:
    def __init__(
        self,
        entries: list[Entry],
        results: dict[str, BacktestResult],
        horizons: list[int],
        baseline: str,
        window: Window,
        freq,
    ):
        self.entries, self.results, self.horizons = entries, results, horizons
        self.baseline, self.window, self.freq = baseline, window, freq

    def __repr__(self) -> str:
        return f"CompareResult(models={list(self.results)}, baseline={self.baseline!r}, horizons={self.horizons})"

    def scores(self, name: str) -> xr.Dataset:
        return self.results[name].scores

    # ------------------------------------------------------------------ building blocks

    def common(self, horizon: int) -> dict[str, xr.Dataset]:
        """Scores at ``horizon`` on the origins where every model has a joint LPD."""
        sel = {name: r.scores.sel(horizon=horizon) for name, r in self.results.items()}
        labels = None
        for ds in sel.values():
            ok = {
                str(o)
                for o, v in zip(ds["origin"].values, ds["lpd"].values, strict=True)
                if np.isfinite(v)
            }
            labels = ok if labels is None else labels & ok
        order = [str(o) for o in sel[self.baseline]["origin"].values if str(o) in labels]
        return {name: ds.sel(origin=order) for name, ds in sel.items()}

    def mcs(self, horizon: int, *, n_boot: int = 2000, seed: int = 0) -> pd.Series:
        """MCS p-values (log score losses) of every model at ``horizon``."""
        common = self.common(horizon)
        losses = np.column_stack([-ds["lpd"].values for ds in common.values()])
        result = model_confidence_set(losses, n_boot=n_boot, seed=seed)
        return pd.Series(result.pvalues, index=list(common), name=f"mcs_pvalue_h{horizon}")

    # ------------------------------------------------------------------ outputs

    def table(self) -> pd.DataFrame:
        """Ablation table: one row per (model, horizon), relative measures against the baseline."""
        rows = []
        for h in self.horizons:
            common = self.common(h)
            base = common[self.baseline]
            mcs = self.mcs(h)
            for entry in self.entries:
                ds = common[entry.name]
                is_base = entry.name == self.baseline
                diff = ds["lpd"].values - base["lpd"].values
                dm = None if is_base else diebold_mariano(-ds["lpd"].values, -base["lpd"].values, h)
                rows.append(
                    {
                        "model": entry.name,
                        "horizon": h,
                        **{f"axis:{k}": v for k, v in entry.axes.items()},
                        "origins": ds.sizes["origin"],
                        "lpd": float(ds["lpd"].sum()),
                        "dlpd": float(diff.sum()),
                        "dlpd_mean": float(diff.mean()) if diff.size else np.nan,
                        "dlpd_mcse": 0.0 if is_base else _mcse_total(ds, base),
                        "crps": float(ds["crps"].mean()),
                        "crps_rel": float(ds["crps"].mean() / base["crps"].mean()),
                        "qwcrps_left": float(ds["qwcrps_left"].mean()),
                        "qwcrps_right": float(ds["qwcrps_right"].mean()),
                        "energy": float(ds["energy"].mean()),
                        "variogram": float(ds["variogram"].mean()),
                        "rmse_rel": _rmse(ds) / _rmse(base),
                        "mae_rel": _mae(ds) / _mae(base),
                        "dm_stat": np.nan if dm is None else dm.statistic,
                        "dm_pvalue": np.nan if dm is None else dm.pvalue,
                        "tests_valid": self.window.valid_tests,
                        "mcs_pvalue": float(mcs[entry.name]),
                        "in_mcs_10": bool(mcs[entry.name] >= 0.10),
                        "in_mcs_25": bool(mcs[entry.name] >= 0.25),
                        "converged": _converged(ds),
                    }
                )
        return pd.DataFrame(rows).set_index(["model", "horizon"])

    def verdict(
        self,
        axis: str | None = None,
        horizons: Sequence[int] | None = None,
        *,
        test_level: float = 0.10,
        mcs_alpha: float = 0.25,
        mc_multiple: float = 3.0,
        min_converged: float = 0.95,
    ) -> pd.DataFrame:
        """Apply the verdict rule of SPEC §9.5.

        Without ``axis`` every model is compared with the baseline. With ``axis`` every model is
        compared with its neighbour that takes the baseline's value on that axis and agrees on
        all other axes (e.g. ``axis="mean"``: the mean effect holding each volatility fixed).
        """
        entries = {e.name: e for e in self.entries}
        base_axes = dict(entries[self.baseline].axes)
        pairs = []
        for e in self.entries:
            if axis is None:
                if e.name != self.baseline:
                    pairs.append((e.name, self.baseline, _differing_axes(e.axes, base_axes)))
            elif e.axes.get(axis) != base_axes.get(axis):
                target = {**e.axes, axis: base_axes.get(axis)}
                ref = next((f.name for f in self.entries if dict(f.axes) == target), None)
                if ref is not None:
                    pairs.append((e.name, ref, [axis]))

        rows = []
        for h in horizons or self.horizons:
            common, mcs = self.common(h), self.mcs(h)
            for name, ref, diff in pairs:
                row = {
                    "model": name,
                    "reference": ref,
                    "axis": ",".join(diff) or "(none)",
                    "horizon": h,
                }
                if len(diff) > 1:
                    row |= {
                        "verdict": "not comparable",
                        "reasons": f"differs along {diff}; a verdict needs exactly one axis (SPEC R2)",
                    }
                    rows.append(row)
                    continue
                m, r = common[name], common[ref]
                total = float((m["lpd"] - r["lpd"]).sum())
                mcse = _mcse_total(m, r)
                precise = abs(total) > mc_multiple * mcse
                lm, lr = -m["lpd"].values, -r["lpd"].values
                p_better = diebold_mariano(lm, lr, h, alternative="less").pvalue
                p_worse = diebold_mariano(lm, lr, h, alternative="greater").pvalue
                valid = self.window.valid_tests
                m_in, r_in = mcs[name] >= mcs_alpha, mcs[ref] >= mcs_alpha
                converged = min(_converged(m), _converged(r))
                better = (
                    total > 0
                    and precise
                    and ((valid and p_better < test_level) or (m_in and not r_in))
                )
                worse = (
                    total < 0
                    and precise
                    and ((valid and p_worse < test_level) or (r_in and not m_in))
                )
                verdict = "better" if better else "worse" if worse else "inconclusive"
                if converged < min_converged:
                    verdict = "inconclusive"
                reasons = [
                    f"ΔLPD={total:+.2f} (MC s.e. {mcse:.2f}{'' if precise else ', not precise enough'})",
                    f"DM p={min(p_better, p_worse):.3f} "
                    f"({'valid: rolling window' if valid else 'descriptive only: expanding window'})",
                    f"MCS{int(mcs_alpha * 100)}: model {'in' if m_in else 'out'}, reference {'in' if r_in else 'out'}",
                ]
                if converged < min_converged:
                    reasons.append(f"converged at only {converged:.0%} of origins")
                row |= {
                    "verdict": verdict,
                    "dlpd": total,
                    "dlpd_mcse": mcse,
                    "dm_pvalue_better": p_better,
                    "dm_pvalue_worse": p_worse,
                    "tests_valid": valid,
                    "model_in_mcs": bool(m_in),
                    "reference_in_mcs": bool(r_in),
                    "reasons": "; ".join(reasons),
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def plot_cumulative_lpd(self, horizon: int = 1, ax=None):
        """Cumulative LPD difference of every model against the baseline over origins."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        common = self.common(horizon)
        base = common[self.baseline]["lpd"].values
        x = pd.PeriodIndex(
            common[self.baseline]["origin"].values.astype(str), freq=self.freq
        ).to_timestamp()
        for name, ds in common.items():
            if name != self.baseline:
                ax.plot(x, np.cumsum(ds["lpd"].values - base), label=name)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_ylabel(f"cumulative LPD vs {self.baseline} (h={horizon})")
        ax.legend(frameon=False)
        return ax


def _entries(models) -> list[Entry]:
    if isinstance(models, Mapping):
        entries = [Entry(str(name), m, dict(m.axes)) for name, m in models.items()]
    else:
        entries = [m if isinstance(m, Entry) else Entry(m.name, m, dict(m.axes)) for m in models]
    names = [e.name for e in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    if not entries:
        raise ValueError("no models to compare")
    return entries


def _label(value: Any) -> str:
    return str(getattr(value, "name", None) or value)


def _differing_axes(a: Mapping[str, str], b: Mapping[str, str]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _mcse_total(a: xr.Dataset, b: xr.Dataset) -> float:
    return float(np.sqrt(np.nansum(a["lpd_mcse"].values ** 2 + b["lpd_mcse"].values ** 2)))


def _converged(ds: xr.Dataset) -> float:
    return float(ds["converged"].mean()) if "converged" in ds else 1.0


def _rmse(ds: xr.Dataset) -> float:
    return float(np.sqrt(ds["sq_err"].mean(dim="origin")).mean())


def _mae(ds: xr.Dataset) -> float:
    return float(ds["abs_err"].mean())
