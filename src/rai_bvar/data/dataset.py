"""``VARData``: raw levels plus transformation codes → the transformed estimation sample.

The raw levels are kept so forecasts can be mapped back pathwise (SPEC §5.2). Column order is
the variable ordering used by recursive (triangular) models (SPEC §3 R8).
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rai_bvar.data import transforms


@dataclass(frozen=True, eq=False)
class VARData:
    """Raw data, transformation codes and the resulting estimation sample.

    Construct with :meth:`from_frame`. ``start`` trims the beginning of the transformed sample
    (used by rolling windows) while keeping earlier raw values for pathwise inversion.
    """

    raw: pd.DataFrame
    tcodes: Mapping[str, int]
    vintage: str | None = None
    start: pd.Period | None = None
    quiet: bool = field(default=False, repr=False)
    sample: pd.DataFrame = field(init=False, repr=False)
    _transformed: pd.DataFrame = field(init=False, repr=False)  # all periods, before trimming

    def __post_init__(self) -> None:
        raw = self.raw
        if not isinstance(raw.index, pd.PeriodIndex):
            raise TypeError("VARData.raw must have a PeriodIndex; use VARData.from_frame")
        missing = [c for c in raw.columns if c not in self.tcodes]
        if missing:
            raise ValueError(f"no transformation code for {missing}")
        if np.isinf(raw.to_numpy(dtype=np.float64)).any():
            raise ValueError("raw data contains infinite values")
        transformed = pd.DataFrame(
            {c: transforms.forward(raw[c].to_numpy(), self.tcodes[c]) for c in raw.columns},
            index=raw.index,
        )
        object.__setattr__(self, "_transformed", transformed)
        object.__setattr__(self, "sample", self._trim(transformed))
        object.__setattr__(self, "tcodes", {c: int(self.tcodes[c]) for c in raw.columns})

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        tcodes: Mapping[str, int] | Sequence[int] | int | None = None,
        *,
        vintage: str | None = None,
    ) -> VARData:
        """Build from a DataFrame of raw levels indexed by dates or periods.

        ``tcodes`` defaults to 1 (no transformation) for every column.
        """
        if frame.shape[1] == 0:
            raise ValueError("frame has no columns")
        raw = frame.copy()
        raw.columns = [str(c) for c in raw.columns]
        if len(set(raw.columns)) != len(raw.columns):
            raise ValueError("duplicate column names")
        raw.index = _to_period_index(raw.index)
        raw = raw.astype(np.float64)
        return cls(raw=raw, tcodes=_normalise_tcodes(tcodes, list(raw.columns)), vintage=vintage)

    # ------------------------------------------------------------------ basic properties

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.raw.columns)

    @property
    def n_vars(self) -> int:
        return self.raw.shape[1]

    @property
    def index(self) -> pd.PeriodIndex:
        return self.sample.index

    @property
    def freq(self) -> str:
        return self.raw.index.freqstr

    @property
    def first(self) -> pd.Period:
        return self.sample.index[0]

    @property
    def last(self) -> pd.Period:
        return self.sample.index[-1]

    @property
    def values(self) -> np.ndarray:
        """Transformed estimation sample, shape ``(T, n)``."""
        return self.sample.to_numpy(dtype=np.float64, copy=True)

    def __len__(self) -> int:
        return len(self.sample)

    def __repr__(self) -> str:
        return (
            f"VARData(variables={list(self.variables)}, T={len(self)}, "
            f"{self.first}..{self.last}, freq={self.freq!r}, vintage={self.vintage!r})"
        )

    def prior_means(self) -> dict[str, float]:
        """Transform-aware Minnesota prior mean of each variable's own first lag (SPEC §5.3)."""
        return {v: transforms.prior_mean_own_lag(self.tcodes[v]) for v in self.variables}

    # ------------------------------------------------------------------ sub-samples

    def upto(self, origin: pd.Period | str) -> VARData:
        """Data available at ``origin`` (inclusive): nothing after it is visible (SPEC §3 R9)."""
        origin = self._period(origin)
        if origin < self.first or origin > self.last:
            raise ValueError(f"origin {origin} outside the sample {self.first}..{self.last}")
        return VARData(
            raw=self.raw.loc[:origin],
            tcodes=self.tcodes,
            vintage=self.vintage,
            start=self.start,
            quiet=True,
        )

    def tail(self, n_obs: int) -> VARData:
        """Keep the last ``n_obs`` transformed observations (rolling windows)."""
        if not 0 < n_obs <= len(self):
            raise ValueError(f"cannot keep {n_obs} of {len(self)} observations")
        return VARData(
            raw=self.raw,
            tcodes=self.tcodes,
            vintage=self.vintage,
            start=self.sample.index[-n_obs],
            quiet=True,
        )

    def select(self, variables: Sequence[str]) -> VARData:
        """Subset and/or reorder variables."""
        variables = [str(v) for v in variables]
        unknown = [v for v in variables if v not in self.raw.columns]
        if unknown:
            raise KeyError(f"unknown variables {unknown}")
        return VARData(
            raw=self.raw[variables],
            tcodes={v: self.tcodes[v] for v in variables},
            vintage=self.vintage,
            start=self.start,
            quiet=True,
        )

    def realized(self, origin: pd.Period | str, horizon: int) -> np.ndarray:
        """Transformed outcomes for ``origin+1 … origin+horizon``; NaN where not yet observed.

        Values in a trailing ragged edge (periods dropped from the sample because some variable
        is missing) are returned for the variables that are observed.
        """
        origin = self._period(origin)
        future = [origin + h for h in range(1, horizon + 1)]
        future_values = self._transformed.reindex(pd.PeriodIndex(future))
        return future_values.to_numpy(dtype=np.float64, copy=True)

    # ------------------------------------------------------------------ inversion

    def raw_history(self, n_obs: int) -> np.ndarray:
        """Last ``n_obs`` raw observations up to the end of the sample, shape ``(n_obs, n)``."""
        raw = self.raw.loc[: self.last].to_numpy(dtype=np.float64)
        if n_obs > raw.shape[0]:
            raise ValueError("not enough raw history")
        return raw[raw.shape[0] - n_obs :]

    def inverse_paths(self, paths: np.ndarray) -> np.ndarray:
        """Map transformed paths ``(..., H, n)`` starting after the sample end to raw levels."""
        paths = np.asarray(paths, dtype=np.float64)
        if paths.shape[-1] != self.n_vars:
            raise ValueError("last axis of paths must match the number of variables")
        need = max(transforms.history_length(t) for t in self.tcodes.values())
        history = self.raw_history(max(need, 1))
        out = np.empty_like(paths)
        for j, var in enumerate(self.variables):
            out[..., j] = transforms.inverse_paths(paths[..., j], history[:, j], self.tcodes[var])
        return out

    # ------------------------------------------------------------------ identity

    def fingerprint(self) -> str:
        """Content hash of everything that determines the estimation sample and its inversion."""
        need = max(transforms.history_length(t) for t in self.tcodes.values())
        pos = self.raw.index.get_loc(self.first)
        raw = self.raw.iloc[max(pos - need, 0) : self.raw.index.get_loc(self.last) + 1]
        h = hashlib.sha256()
        h.update(
            repr((self.variables, self.tcodes, self.freq, str(self.first), str(self.last))).encode()
        )
        h.update(
            np.ascontiguousarray(np.nan_to_num(raw.to_numpy(np.float64), nan=-1.2345e300)).tobytes()
        )
        return h.hexdigest()

    # ------------------------------------------------------------------ helpers

    def _period(self, value: pd.Period | str) -> pd.Period:
        return pd.Period(value, freq=self.raw.index.freq)

    def _trim(self, transformed: pd.DataFrame) -> pd.DataFrame:
        complete = transformed.notna().all(axis=1).to_numpy()
        if not complete.any():
            raise ValueError("no period has all variables observed after transformation")
        first, last = int(np.argmax(complete)), len(complete) - 1 - int(np.argmax(complete[::-1]))
        need = max(transforms.history_length(t) for t in self.tcodes.values())
        if first > need and not self.quiet:
            warnings.warn(
                f"dropped {first} leading periods with missing values (sample starts {transformed.index[first]})",
                stacklevel=4,
            )
        if last < len(complete) - 1 and not self.quiet:
            warnings.warn(
                f"dropped {len(complete) - 1 - last} trailing periods with missing values "
                f"(ragged edge; sample ends {transformed.index[last]})",
                stacklevel=4,
            )
        sample = transformed.iloc[first : last + 1]
        if not complete[first : last + 1].all():
            bad = sample.index[~sample.notna().all(axis=1)]
            cols = sample.columns[sample.loc[bad].isna().any(axis=0)].tolist()
            raise ValueError(
                f"missing values inside the sample at {list(map(str, bad[:5]))} in {cols}"
            )
        if self.start is not None:
            sample = sample.loc[self.start :]
            if sample.empty:
                raise ValueError(f"start {self.start} is after the end of the sample")
        return sample


def _normalise_tcodes(
    tcodes: Mapping[str, int] | Sequence[int] | int | None, columns: list[str]
) -> dict[str, int]:
    if tcodes is None:
        tcodes = 1
    if isinstance(tcodes, int | np.integer):
        out = dict.fromkeys(columns, int(tcodes))
    elif isinstance(tcodes, Mapping):
        out = {c: int(tcodes[c]) for c in columns if c in tcodes}
        if len(out) != len(columns):
            raise ValueError(f"tcodes missing for {[c for c in columns if c not in out]}")
    else:
        tcodes = list(tcodes)
        if len(tcodes) != len(columns):
            raise ValueError("tcodes sequence must have one entry per column")
        out = dict(zip(columns, map(int, tcodes), strict=True))
    for code in out.values():
        transforms.validate_tcode(code)
    return out


def _to_period_index(index: pd.Index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        period = index
    elif isinstance(index, pd.DatetimeIndex):
        freq = index.freqstr or pd.infer_freq(index)
        if freq is None:
            raise ValueError("cannot infer a regular frequency from the DatetimeIndex")
        base = freq.split("-")[0].rstrip("SE")  # MS/ME → M, QS-DEC → Q, YS → Y
        period = index.to_period({"A": "Y"}.get(base, base))
    else:
        raise TypeError("index must be a PeriodIndex or DatetimeIndex")
    if period.freq is None:
        raise ValueError("index has no frequency")
    if period.has_duplicates or not period.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing")
    expected = pd.period_range(period[0], period[-1], freq=period.freq)
    if len(expected) != len(period):
        raise ValueError("index has gaps; a regular frequency is required")
    return period
