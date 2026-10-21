"""FRED-MD and FRED-QD reference datasets (McCracken & Ng 2016, 2020), downloaded on demand.

The files are **not bundled** with the package: FRED-MD/QD mix series from U.S. federal agencies
with series from third parties, and some of the latter are copyrighted (e.g. S&P stock
indices, the University of Michigan sentiment index). :func:`load` downloads a pinned vintage
from the St. Louis Fed, checks its SHA-256 checksum against :data:`REGISTRY` and caches it under
``<cache>/<name>/<vintage>.csv``, where ``<cache>`` is ``cache_dir``, else the ``RAI_BVAR_DATA``
environment variable, else ``~/.cache/rai-bvar`` (SPEC §5.8, decision D4).

The small / medium / large sets in :data:`VARIABLE_SETS` contain only series published by U.S.
federal agencies (BEA, BLS, Census, Federal Reserve). Test fixtures committed to the repository
must likewise use federal-agency series only, with synthetic values.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import io
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import IO, Literal

import numpy as np
import pandas as pd

from rai_bvar._version import __version__
from rai_bvar.data.dataset import VARData

__all__ = [
    "BASE_URL",
    "DEFAULT_VINTAGE",
    "REGISTRY",
    "VARIABLE_SETS",
    "dataset_url",
    "default_cache_dir",
    "fetch",
    "load",
    "parse_fred_csv",
]

DatasetName = Literal["fred-md", "fred-qd"]
Kind = Literal["md", "qd"]

BASE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md"
LANDING_PAGE = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
DEFAULT_VINTAGE = "2026-09"
CACHE_ENV = "RAI_BVAR_DATA"
USER_AGENT = f"rai-bvar/{__version__} (Python urllib)"
DOWNLOAD_TIMEOUT = 60.0  # seconds per blocking socket operation
DOWNLOAD_ATTEMPTS = 3  # transient network failures are retried; HTTP errors are not

_KINDS: dict[str, Kind] = {"fred-md": "md", "fred-qd": "qd"}
_FOLDERS = {"md": "monthly", "qd": "quarterly"}
_FREQS = {"md": "M", "qd": "Q"}
_VINTAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_NA_TOKENS = frozenset({"", "NA", "NaN", "nan", "."})
_LOGGED_TCODES = frozenset({4, 5, 6})

#: Verified SHA-256 checksums of published vintages, keyed by ``(name, vintage)``.
#: Add an entry to verify a vintage that is not listed here.
REGISTRY: dict[tuple[str, str], str] = {
    ("fred-md", "2026-09"): "d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d",
    ("fred-qd", "2026-09"): "a144b1547ddbad5c68fc934c90c0c0f9837a5139571ba8e64afcc2637aaa3feb",
}

_QD_SETS = {
    "small": {
        "GDPC1": 5,  # real GDP (BEA)
        "GDPCTPI": 5,  # GDP chain-type price index (BEA)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
    },
    "medium": {
        "GDPC1": 5,  # real GDP (BEA)
        "PCECC96": 5,  # real personal consumption expenditures (BEA)
        "GPDIC1": 5,  # real gross private domestic investment (BEA)
        "HOANBS": 5,  # hours, nonfarm business sector (BLS)
        "COMPRNFB": 5,  # real compensation per hour, nonfarm business (BLS)
        "GDPCTPI": 5,  # GDP chain-type price index (BEA)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
    },
    "large": {
        "GDPC1": 5,  # real GDP (BEA)
        "PCECC96": 5,  # real personal consumption expenditures (BEA)
        "GPDIC1": 5,  # real gross private domestic investment (BEA)
        "DPIC96": 5,  # real disposable personal income (BEA)
        "EXPGSC1": 5,  # real exports (BEA)
        "IMPGSC1": 5,  # real imports (BEA)
        "INDPRO": 5,  # industrial production (Fed)
        "TCU": 1,  # capacity utilization, total industry (Fed)
        "PAYEMS": 5,  # nonfarm payrolls (BLS)
        "HOANBS": 5,  # hours, nonfarm business sector (BLS)
        "UNRATE": 1,  # unemployment rate (BLS)
        "HOUST": 4,  # housing starts (Census)
        "COMPRNFB": 5,  # real compensation per hour, nonfarm business (BLS)
        "GDPCTPI": 5,  # GDP chain-type price index (BEA)
        "PCECTPI": 5,  # PCE chain-type price index (BEA)
        "CPIAUCSL": 5,  # CPI, all items (BLS)
        "M2REAL": 5,  # real M2 money stock (Fed)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
        "GS1": 1,  # 1-year Treasury yield (Fed)
        "GS10": 1,  # 10-year Treasury yield (Fed)
    },
}

_MD_SETS = {
    "small": {
        "INDPRO": 5,  # industrial production (Fed)
        "CPIAUCSL": 5,  # CPI, all items (BLS)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
    },
    "medium": {
        "INDPRO": 5,  # industrial production (Fed)
        "PAYEMS": 5,  # nonfarm payrolls (BLS)
        "UNRATE": 1,  # unemployment rate (BLS)
        "DPCERA3M086SBEA": 5,  # real personal consumption expenditures (BEA)
        "CPIAUCSL": 5,  # CPI, all items (BLS)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
        "GS10": 1,  # 10-year Treasury yield (Fed)
    },
    "large": {
        "RPI": 5,  # real personal income (BEA)
        "W875RX1": 5,  # real personal income ex transfer receipts (BEA)
        "DPCERA3M086SBEA": 5,  # real personal consumption expenditures (BEA)
        "INDPRO": 5,  # industrial production (Fed)
        "CUMFNS": 1,  # capacity utilization, manufacturing (Fed)
        "PAYEMS": 5,  # nonfarm payrolls (BLS)
        "CE16OV": 5,  # civilian employment (BLS)
        "UNRATE": 1,  # unemployment rate (BLS)
        "AWHMAN": 1,  # average weekly hours, manufacturing (BLS)
        "HOUST": 4,  # housing starts (Census)
        "CES0600000008": 5,  # average hourly earnings, goods-producing (BLS)
        "WPSFD49207": 5,  # PPI, finished goods (BLS)
        "PCEPI": 5,  # PCE price index (BEA)
        "CPIAUCSL": 5,  # CPI, all items (BLS)
        "M2REAL": 5,  # real M2 money stock (Fed)
        "FEDFUNDS": 1,  # effective federal funds rate (Fed)
        "TB3MS": 1,  # 3-month Treasury bill rate (Fed)
        "GS1": 1,  # 1-year Treasury yield (Fed)
        "GS5": 1,  # 5-year Treasury yield (Fed)
        "GS10": 1,  # 10-year Treasury yield (Fed)
    },
}

#: Ordered variable sets ``VARIABLE_SETS[name][size] -> {series: rai tcode}`` (read-only).
#: The order is the recursive ordering (SPEC §3 R8); the codes are the ``tcodes="rai"`` defaults.
VARIABLE_SETS: Mapping[str, Mapping[str, Mapping[str, int]]] = MappingProxyType(
    {
        name: MappingProxyType(
            {size: MappingProxyType(dict(codes)) for size, codes in sets.items()}
        )
        for name, sets in (("fred-md", _MD_SETS), ("fred-qd", _QD_SETS))
    }
)


# ---------------------------------------------------------------------------- public API


def load(
    name: DatasetName,
    vintage: str = DEFAULT_VINTAGE,
    variables: Literal["small", "medium", "large"] | Sequence[str] = "small",
    tcodes: Literal["rai", "official"] | Mapping[str, int] = "rai",
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    url: str | None = None,
    verify: bool = True,
    fill_gaps: bool = True,
) -> VARData:
    """Load a FRED-MD / FRED-QD vintage as :class:`~rai_bvar.data.VARData`.

    Parameters
    ----------
    name
        ``"fred-md"`` (monthly) or ``"fred-qd"`` (quarterly).
    vintage
        Vintage label as published, e.g. ``"2026-09"`` or a revised ``"2026-rev-08"``.
    variables
        ``"small"``, ``"medium"``, ``"large"`` (see :data:`VARIABLE_SETS`) or a sequence of
        series names; the order given is the variable ordering.
    tcodes
        ``"rai"``: the codes of the named set (a custom list falls back to the file's official
        codes); ``"official"``: the file's codes; a mapping overrides individual variables on
        top of the ``"rai"`` default.
    cache_dir
        Cache root; defaults to ``$RAI_BVAR_DATA``, else ``~/.cache/rai-bvar``.
    url
        Download from this URL instead of the St. Louis Fed (mirrors, ``file://`` copies).
    verify
        Check the SHA-256 checksum against :data:`REGISTRY`. A mismatch raises; a vintage
        without a registered checksum warns and reports the computed hash.
    fill_gaps
        Interpolate missing values strictly *inside* a series' observed span (linearly, in logs
        for log-transformed series) with a warning. Publication gaps occur in the real files,
        e.g. October 2025 for BLS series after the federal shutdown. With ``False`` they are
        kept and :class:`VARData` raises if they fall inside the estimation sample.

    The latest periods are usually ragged (some series not yet published); ``VARData`` trims
    those trailing incomplete periods with a warning.
    """
    key = _check_name(name)
    path = fetch(key, vintage, cache_dir=cache_dir, url=url, verify=verify, _stacklevel=3)
    raw, official = parse_fred_csv(path, _KINDS[key])

    if isinstance(variables, str):
        sets = VARIABLE_SETS[key]
        if variables not in sets:
            raise ValueError(
                f"unknown variable set {variables!r}; expected one of {list(sets)} "
                "or a sequence of series names"
            )
        preset: Mapping[str, int] | None = sets[variables]
        names = list(preset)
    else:
        preset = None
        names = [str(v) for v in variables]
        if not names:
            raise ValueError("variables must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variables in {names}")
    missing = [v for v in names if v not in raw.columns]
    if missing:
        raise KeyError(f"{key} vintage {vintage!r} has no series {missing}")

    codes = _resolve_tcodes(names, preset, tcodes, official)
    frame = raw[names]
    if fill_gaps:
        frame, filled = _fill_interior_gaps(frame, codes)
        if filled:
            detail = "; ".join(f"{v}: {', '.join(p)}" for v, p in filled.items())
            warnings.warn(
                f"interpolated missing values inside the observed span of {list(filled)} "
                f"({detail}); pass fill_gaps=False to keep them missing",
                UserWarning,
                stacklevel=2,
            )
    return VARData.from_frame(frame, codes, vintage=f"{key}:{vintage}")


def parse_fred_csv(
    source: str | os.PathLike[str] | bytes | IO[str] | IO[bytes], kind: Kind
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse a FRED-MD (``kind="md"``) or FRED-QD (``kind="qd"``) CSV file.

    ``source`` is a path, the file's bytes or an open file. Returns the raw levels as float64
    with a monthly or quarterly ``PeriodIndex`` (the calendar period of each ``m/d/yyyy``
    date; empty cells are NaN) and the official transformation codes ``{series: tcode}``.

    Special rows are recognised by their first cell, case-insensitively: ``Transform:``
    (FRED-MD) / ``transform`` (FRED-QD) holds the codes; FRED-QD's ``factors`` row is skipped.
    Rows with an empty date are dropped.
    """
    if kind not in _FREQS:
        raise ValueError(f"kind must be 'md' or 'qd', not {kind!r}")
    rows = list(csv.reader(io.StringIO(_read_text(source))))
    if not rows:
        raise ValueError("empty FRED CSV")
    header = [cell.strip() for cell in rows[0]]
    while len(header) > 1 and not header[-1]:  # trailing commas
        header.pop()
    names = header[1:]
    if not names:
        raise ValueError("FRED CSV has no series columns")
    if any(not n for n in names):
        raise ValueError("FRED CSV header has an empty series name")
    if len(set(names)) != len(names):
        raise ValueError("FRED CSV header has duplicate series names")
    width = len(header)

    tcode_cells: list[str] | None = None
    dates: list[str] = []
    values: list[list[float]] = []
    for lineno, row in enumerate(rows[1:], start=2):
        cells = [cell.strip() for cell in row]
        if any(cells[width:]):
            raise ValueError(f"line {lineno} has more cells than the header")
        cells = (cells + [""] * width)[:width]
        label = cells[0].lower().rstrip(":").strip()
        if label == "transform":
            if tcode_cells is not None:
                raise ValueError("FRED CSV has more than one transform row")
            tcode_cells = cells[1:]
        elif label == "factors":
            continue
        elif cells[0]:
            dates.append(cells[0])
            values.append([_to_float(cell, lineno) for cell in cells[1:]])
    if tcode_cells is None:
        raise ValueError("FRED CSV has no 'Transform:'/'transform' row")
    if not dates:
        raise ValueError("FRED CSV has no data rows")

    tcodes = {}
    for series, cell in zip(names, tcode_cells, strict=True):
        try:
            tcodes[series] = int(float(cell))
        except ValueError:
            raise ValueError(f"invalid transformation code {cell!r} for {series}") from None
    index = _period_index(dates, _FREQS[kind])
    frame = pd.DataFrame(np.array(values, dtype=np.float64), index=index, columns=names)
    return frame, tcodes


def fetch(
    name: DatasetName,
    vintage: str = DEFAULT_VINTAGE,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    url: str | None = None,
    verify: bool = True,
    _stacklevel: int = 2,
) -> Path:
    """Return the local path of a vintage, downloading and verifying it if it is not cached."""
    key = _check_name(name)
    vintage = _check_vintage(vintage)
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    path = root / key / f"{vintage}.csv"
    if path.is_file():
        if verify:
            try:
                _verify(path.read_bytes(), key, vintage, str(path), _stacklevel + 1)
            except ValueError as exc:
                path.unlink(missing_ok=True)
                raise ValueError(
                    f"{exc}; the cached file was removed, load again to re-download"
                ) from None
        return path
    source = url if url is not None else dataset_url(key, vintage)
    payload = _download(source, timeout=DOWNLOAD_TIMEOUT)
    if verify:
        _verify(payload, key, vintage, source, _stacklevel + 1)  # raises before anything is cached
    _write_atomic(path, payload)
    return path


def dataset_url(name: DatasetName, vintage: str) -> str:
    """Official download URL of a vintage."""
    kind = _KINDS[_check_name(name)]
    return f"{BASE_URL}/{_FOLDERS[kind]}/{_check_vintage(vintage)}-{kind}.csv"


def default_cache_dir() -> Path:
    """``$RAI_BVAR_DATA`` if set, else ``~/.cache/rai-bvar``."""
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "rai-bvar"


# ---------------------------------------------------------------------------- download & cache


def _download(url: str, *, timeout: float = DOWNLOAD_TIMEOUT) -> bytes:
    """Fetch ``url`` and return its content (tests replace this function)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise OSError(
                f"downloading {url} failed: HTTP {exc.code} {exc.reason}; check that the vintage "
                f"exists ({LANDING_PAGE})"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, resets
            if attempt == DOWNLOAD_ATTEMPTS or not url.lower().startswith(("http:", "https:")):
                raise OSError(
                    f"downloading {url} failed after {attempt} attempt(s): {exc}"
                ) from exc
            time.sleep(attempt)
    raise AssertionError("unreachable")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _verify(payload: bytes, name: str, vintage: str, origin: str, stacklevel: int) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    expected = REGISTRY.get((name, vintage))
    if expected is None:
        warnings.warn(
            f"no registered checksum for {name} vintage {vintage!r}, so {origin} was not verified "
            f"(sha256={digest}); add it to rai_bvar.datasets.REGISTRY or pass verify=False",
            UserWarning,
            stacklevel=stacklevel,
        )
    elif digest != expected.lower():
        raise ValueError(
            f"SHA-256 checksum mismatch for {name} vintage {vintage!r} from {origin}: "
            f"expected {expected}, got {digest}"
        )


# ---------------------------------------------------------------------------- helpers


def _check_name(name: str) -> str:
    key = str(name).lower()
    if key not in _KINDS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(_KINDS)}")
    return key


def _check_vintage(vintage: str) -> str:
    if not isinstance(vintage, str) or not _VINTAGE_RE.fullmatch(vintage):
        raise ValueError(f"invalid vintage {vintage!r}; expected a label such as '2026-09'")
    return vintage


def _resolve_tcodes(
    names: list[str],
    preset: Mapping[str, int] | None,
    tcodes: str | Mapping[str, int],
    official: Mapping[str, int],
) -> dict[str, int]:
    if isinstance(tcodes, str):
        if tcodes not in ("rai", "official"):
            raise ValueError(f"tcodes must be 'rai', 'official' or a mapping, not {tcodes!r}")
        if tcodes == "official" or preset is None:
            return {v: official[v] for v in names}
        return dict(preset)
    if not isinstance(tcodes, Mapping):
        raise TypeError("tcodes must be 'rai', 'official' or a mapping {series: tcode}")
    unknown = [v for v in tcodes if v not in names]
    if unknown:
        raise KeyError(f"tcode overrides for unselected variables {unknown}")
    codes = _resolve_tcodes(names, preset, "rai", official)
    codes.update({v: int(code) for v, code in tcodes.items()})
    return codes


def _fill_interior_gaps(
    frame: pd.DataFrame, codes: Mapping[str, int]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Interpolate NaNs strictly between a series' first and last observations."""
    out = frame.copy()
    filled: dict[str, list[str]] = {}
    pos = np.arange(len(frame))
    for col in frame.columns:
        x = frame[col].to_numpy(dtype=np.float64)
        valid = np.isfinite(x)
        if not valid.any():
            continue
        first, last = int(np.argmax(valid)), len(x) - 1 - int(np.argmax(valid[::-1]))
        gaps = ~valid & (pos > first) & (pos < last)
        if not gaps.any():
            continue
        logged = codes[col] in _LOGGED_TCODES and bool(np.all(x[valid] > 0))
        y = np.log(x[valid]) if logged else x[valid]
        fill = np.interp(pos[gaps], pos[valid], y)
        x = x.copy()
        x[gaps] = np.exp(fill) if logged else fill
        out[col] = x
        filled[col] = [str(p) for p in frame.index[gaps]]
    return out, filled


def _read_text(source: str | os.PathLike[str] | bytes | IO[str] | IO[bytes]) -> str:
    if isinstance(source, bytes | bytearray):
        return bytes(source).decode("utf-8-sig")
    if isinstance(source, str | os.PathLike):
        return Path(source).read_text(encoding="utf-8-sig")
    content = source.read()
    return content.decode("utf-8-sig") if isinstance(content, bytes) else content.lstrip("﻿")


def _to_float(cell: str, lineno: int) -> float:
    if cell in _NA_TOKENS:
        return np.nan
    try:
        return float(cell)
    except ValueError:
        raise ValueError(f"line {lineno}: cannot parse {cell!r} as a number") from None


def _period_index(dates: list[str], freq: str) -> pd.PeriodIndex:
    try:
        stamps = pd.to_datetime(pd.Index(dates), format="%m/%d/%Y")
    except ValueError:
        try:
            stamps = pd.to_datetime(pd.Index(dates), format="ISO8601")
        except ValueError as exc:
            raise ValueError(f"cannot parse dates as m/d/yyyy: {exc}") from None
    index = stamps.to_period(freq)
    index.name = "date"
    if index.has_duplicates:
        dup = sorted({str(p) for p in index[index.duplicated()]})[:5]
        raise ValueError(f"duplicate periods {dup}; is the file kind ('md'/'qd') right?")
    if not index.is_monotonic_increasing:
        raise ValueError("dates are not increasing")
    if len(index) != len(pd.period_range(index[0], index[-1], freq=freq)):
        raise ValueError("dates have gaps; a regular frequency is required")
    return index
