"""Reference dataset loaders (SPEC §5.8): parsing, variable sets, tcodes, checksums and caching.

The committed fixtures are synthetic files in the real FRED-MD/QD layouts and contain only
federal-agency series names. Only the ``network`` test touches the internet.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rai_bvar.datasets as datasets
from rai_bvar.data import VARData
from rai_bvar.datasets import REGISTRY, VARIABLE_SETS, fred, load, parse_fred_csv

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MD_FILE = FIXTURES / "fred_md_sample.csv"
QD_FILE = FIXTURES / "fred_qd_sample.csv"
FILES = {"fred-md": MD_FILE, "fred-qd": QD_FILE}
TEST_VINTAGE = "2000-test"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the fixtures instead of downloading; returns the list of requested URLs."""
    calls: list[str] = []

    def download(url: str, *, timeout: float = 0.0) -> bytes:
        calls.append(url)
        kind = "fred-md" if "/monthly/" in url else "fred-qd"
        return FILES[kind].read_bytes()

    monkeypatch.setattr(fred, "_download", download)
    return calls


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the fixtures' checksums under ``TEST_VINTAGE``."""
    for name, path in FILES.items():
        monkeypatch.setitem(REGISTRY, (name, TEST_VINTAGE), _sha256(path))


def _load(name: str, tmp_path: Path, **kwargs) -> VARData:
    """``load`` a fixture vintage; the ragged last row is trimmed with a warning."""
    kwargs.setdefault("vintage", TEST_VINTAGE)
    with pytest.warns(UserWarning, match="ragged edge"):
        return load(name, cache_dir=tmp_path, **kwargs)


# ---------------------------------------------------------------------------- parsing


def test_parse_md_layout():
    frame, tcodes = parse_fred_csv(MD_FILE, "md")
    assert isinstance(frame.index, pd.PeriodIndex)
    assert frame.index.freqstr == "M"
    assert frame.index[0] == pd.Period("2000-01", freq="M")
    assert frame.index[-1] == pd.Period("2001-04", freq="M")
    assert len(frame) == 16
    assert list(frame.columns) == ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE"]
    assert tcodes == {"INDPRO": 5, "CPIAUCSL": 6, "FEDFUNDS": 2, "UNRATE": 2}
    assert (frame.dtypes == np.float64).all()
    assert frame.loc["2000-01", "INDPRO"] == 100.0
    assert np.isnan(frame.loc["2001-04", "INDPRO"])  # ragged last row
    assert np.isnan(frame.loc["2000-06", "UNRATE"])  # interior gap
    assert frame.isna().sum().sum() == 2


def test_parse_qd_layout():
    frame, tcodes = parse_fred_csv(QD_FILE, "qd")
    assert isinstance(frame.index, pd.PeriodIndex)
    assert frame.index.freqstr == "Q-DEC"
    assert frame.index[0] == pd.Period("2000Q1", freq="Q")
    assert frame.index[-1] == pd.Period("2003Q4", freq="Q")  # the empty-date row is dropped
    assert len(frame) == 16  # the factors and transform rows are not data
    assert tcodes == {"GDPC1": 5, "GDPCTPI": 6, "FEDFUNDS": 2, "PCECC96": 5}
    assert (frame.dtypes == np.float64).all()
    assert frame["PCECC96"].isna().tolist()[:3] == [True, True, False]
    assert frame.loc["2003Q4"].isna().tolist() == [True, True, False, True]


@pytest.mark.parametrize(
    ("date", "kind", "period"),
    [
        ("1/1/1959", "md", "1959-01"),
        ("12/1/2025", "md", "2025-12"),
        ("3/1/1959", "qd", "1959Q1"),
        ("6/1/1959", "qd", "1959Q2"),
        ("12/1/2025", "qd", "2025Q4"),
    ],
)
def test_parse_dates_to_calendar_periods(date: str, kind: str, period: str):
    text = f"sasdate,X\nTRANSFORM:,1\n{date},1.5\n"
    frame, tcodes = parse_fred_csv(text.encode(), kind)
    assert frame.index[0] == pd.Period(period, freq=frame.index.freq)
    assert tcodes == {"X": 1}


def test_parse_accepts_file_objects_and_missing_tokens():
    text = "sasdate,A,B\nfactors,1,0\nTransform,4,1\n3/1/2000,1, \n6/1/2000,NA,2\n,,\n"
    for source in (io.StringIO(text), io.BytesIO(("﻿" + text).encode())):
        frame, tcodes = parse_fred_csv(source, "qd")
        assert tcodes == {"A": 4, "B": 1}
        assert list(frame.index.astype(str)) == ["2000Q1", "2000Q2"]
        np.testing.assert_array_equal(frame.to_numpy(), [[1.0, np.nan], [np.nan, 2.0]])


@pytest.mark.parametrize(
    ("text", "kind", "match"),
    [
        ("sasdate,A\n1/1/2000,1\n", "md", "transform"),
        ("sasdate,A\nTransform:,x\n1/1/2000,1\n", "md", "transformation code"),
        ("sasdate,A\nTransform:,1\n2000/13/45,1\n", "md", "dates"),
        ("sasdate,A\nTransform:,1\n1/1/2000,1\n2/1/2000,2\n", "qd", "duplicate periods"),
        ("sasdate,A\nTransform:,1\n1/1/2000,1\n3/1/2000,2\n", "md", "gaps"),
        ("sasdate,A\nTransform:,1\n1/1/2000,abc\n", "md", "cannot parse"),
        ("sasdate,A,A\nTransform:,1,1\n1/1/2000,1,2\n", "md", "duplicate series"),
        ("sasdate,A\nTransform:,1\n", "md", "no data rows"),
    ],
)
def test_parse_errors(text: str, kind: str, match: str):
    with pytest.raises(ValueError, match=match):
        parse_fred_csv(text.encode(), kind)


def test_parse_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        parse_fred_csv(MD_FILE, "monthly")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- variable sets


def test_public_exports():
    assert set(datasets.__all__) == {"REGISTRY", "VARIABLE_SETS", "load", "parse_fred_csv"}


def test_variable_sets_structure():
    assert set(VARIABLE_SETS) == {"fred-md", "fred-qd"}
    for sets in VARIABLE_SETS.values():
        assert list(sets) == ["small", "medium", "large"]
        assert [len(sets[size]) for size in sets] == [3, 7, 20]
        for codes in sets.values():
            assert set(codes.values()) <= {1, 4, 5}
            assert list(codes)[-1] in {"FEDFUNDS", "GS10"}  # rates come last (recursive ordering)
        assert set(sets["small"]) <= set(sets["medium"]) <= set(sets["large"])
    assert dict(VARIABLE_SETS["fred-qd"]["small"]) == {"GDPC1": 5, "GDPCTPI": 5, "FEDFUNDS": 1}
    assert dict(VARIABLE_SETS["fred-md"]["small"]) == {"INDPRO": 5, "CPIAUCSL": 5, "FEDFUNDS": 1}
    qd_medium = "GDPC1 PCECC96 GPDIC1 HOANBS COMPRNFB GDPCTPI FEDFUNDS"
    md_medium = "INDPRO PAYEMS UNRATE DPCERA3M086SBEA CPIAUCSL FEDFUNDS GS10"
    assert list(VARIABLE_SETS["fred-qd"]["medium"]) == qd_medium.split()
    assert list(VARIABLE_SETS["fred-md"]["medium"]) == md_medium.split()
    assert list(VARIABLE_SETS["fred-qd"]["large"])[:3] == ["GDPC1", "PCECC96", "GPDIC1"]
    assert list(VARIABLE_SETS["fred-md"]["large"])[:3] == ["RPI", "W875RX1", "DPCERA3M086SBEA"]
    with pytest.raises(TypeError):
        VARIABLE_SETS["fred-qd"]["small"]["GDPC1"] = 1  # type: ignore[index]


@pytest.mark.usefixtures("registered", "fake_download")
@pytest.mark.parametrize("name", ["fred-md", "fred-qd"])
def test_load_small_set(name: str, tmp_path: Path):
    data = _load(name, tmp_path)
    preset = VARIABLE_SETS[name]["small"]
    assert data.variables == tuple(preset)
    assert data.tcodes == dict(preset)
    assert data.vintage == f"{name}:{TEST_VINTAGE}"
    raw, _ = parse_fred_csv(FILES[name], name[-2:])
    assert data.last == raw.index[-2]  # the ragged last period is trimmed
    assert data.first == raw.index[1]  # one period lost to the log differences
    np.testing.assert_allclose(data.sample["FEDFUNDS"], raw["FEDFUNDS"].iloc[1:-1])


@pytest.mark.usefixtures("registered", "fake_download")
def test_load_custom_variables_keep_order_and_use_official_codes(tmp_path: Path):
    with pytest.warns(UserWarning, match="ragged edge"):
        data = load("fred-qd", TEST_VINTAGE, ["FEDFUNDS", "GDPC1"], cache_dir=tmp_path)
    assert data.variables == ("FEDFUNDS", "GDPC1")
    assert data.tcodes == {"FEDFUNDS": 2, "GDPC1": 5}  # no preset: the file's codes


@pytest.mark.usefixtures("registered", "fake_download")
def test_tcode_modes(tmp_path: Path):
    official = _load("fred-md", tmp_path, tcodes="official")
    assert official.tcodes == {"INDPRO": 5, "CPIAUCSL": 6, "FEDFUNDS": 2}
    override = _load("fred-md", tmp_path, tcodes={"FEDFUNDS": 2})
    assert override.tcodes == {"INDPRO": 5, "CPIAUCSL": 5, "FEDFUNDS": 2}
    # Custom list (both observed in the last row, so no ragged edge): official codes + override.
    custom = load(
        "fred-md", TEST_VINTAGE, ["CPIAUCSL", "FEDFUNDS"], {"CPIAUCSL": 5}, cache_dir=tmp_path
    )
    assert custom.tcodes == {"CPIAUCSL": 5, "FEDFUNDS": 2}


@pytest.mark.usefixtures("registered", "fake_download")
def test_load_argument_errors(tmp_path: Path):
    with pytest.raises(KeyError, match="NOTASERIES"):
        load("fred-qd", TEST_VINTAGE, ["GDPC1", "NOTASERIES"], cache_dir=tmp_path)
    with pytest.raises(KeyError, match="GPDIC1"):  # not in the fixture: the medium set fails
        load("fred-qd", TEST_VINTAGE, "medium", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="variable set"):
        load("fred-qd", TEST_VINTAGE, "huge", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        load("fred-qd", TEST_VINTAGE, ["GDPC1", "GDPC1"], cache_dir=tmp_path)
    with pytest.raises(ValueError, match="tcodes"):
        load("fred-qd", TEST_VINTAGE, tcodes="mine", cache_dir=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="PCECC96"):
        load("fred-qd", TEST_VINTAGE, tcodes={"PCECC96": 5}, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        load("fred-xx", TEST_VINTAGE, cache_dir=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid vintage"):
        load("fred-qd", "../2026-09", cache_dir=tmp_path)


@pytest.mark.usefixtures("registered", "fake_download")
def test_interior_gaps_are_interpolated_with_a_warning(tmp_path: Path):
    with pytest.warns(UserWarning, match=r"interpolated .*UNRATE: 2000-06"):
        data = load(
            "fred-md", TEST_VINTAGE, ["UNRATE", "FEDFUNDS"], {"UNRATE": 1}, cache_dir=tmp_path
        )
    assert data.raw.loc["2000-06", "UNRATE"] == pytest.approx((4.0 + 4.2) / 2)
    assert data.raw.loc["2000-05", "UNRATE"] == 4.0  # observed values are untouched
    assert data.raw["UNRATE"].notna().all()
    assert data.last == pd.Period("2001-04", freq="M")  # both observed in the last row

    with pytest.raises(ValueError, match="missing values inside the sample"):
        load(
            "fred-md", TEST_VINTAGE, ["UNRATE"], {"UNRATE": 1}, cache_dir=tmp_path, fill_gaps=False
        )


def test_log_interpolation_for_log_transformed_series():
    frame = pd.DataFrame(
        {"A": [1.0, np.nan, 4.0], "B": [1.0, np.nan, 4.0], "C": [np.nan, 1.0, np.nan]},
        index=pd.period_range("2000-01", periods=3, freq="M"),
    )
    out, filled = fred._fill_interior_gaps(frame, {"A": 5, "B": 1, "C": 1})
    assert out["A"].iloc[1] == pytest.approx(2.0)  # geometric
    assert out["B"].iloc[1] == pytest.approx(2.5)  # linear
    assert out["C"].isna().tolist() == [True, False, True]  # leading/trailing NaNs are kept
    assert filled == {"A": ["2000-02"], "B": ["2000-02"]}


# ---------------------------------------------------------------------------- download, cache, checksums


def test_dataset_url_and_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md"
    assert fred.dataset_url("fred-md", "2026-09") == f"{base}/monthly/2026-09-md.csv"
    assert fred.dataset_url("fred-qd", "2026-rev-08") == f"{base}/quarterly/2026-rev-08-qd.csv"
    monkeypatch.setenv("RAI_BVAR_DATA", str(tmp_path / "env"))
    assert fred.default_cache_dir() == tmp_path / "env"
    monkeypatch.delenv("RAI_BVAR_DATA")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert fred.default_cache_dir() == tmp_path / ".cache" / "rai-bvar"


def test_registry_has_the_pinned_vintages():
    for key in [("fred-md", "2026-09"), ("fred-qd", "2026-09")]:
        assert len(REGISTRY[key]) == 64
        int(REGISTRY[key], 16)


@pytest.mark.usefixtures("registered")
def test_verified_download_is_cached_and_reused(fake_download: list[str], tmp_path: Path):
    first = _load("fred-qd", tmp_path)
    second = _load("fred-qd", tmp_path, tcodes="official")
    assert fake_download == [fred.dataset_url("fred-qd", TEST_VINTAGE)]  # downloaded once
    cached = tmp_path / "fred-qd" / f"{TEST_VINTAGE}.csv"
    assert cached.read_bytes() == QD_FILE.read_bytes()
    assert [p.name for p in cached.parent.iterdir()] == [cached.name]  # no temp files left
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.usefixtures("fake_download")
def test_cache_dir_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RAI_BVAR_DATA", str(tmp_path))
    monkeypatch.setitem(REGISTRY, ("fred-md", TEST_VINTAGE), _sha256(MD_FILE))
    with pytest.warns(UserWarning, match="ragged edge"):
        load("fred-md", TEST_VINTAGE)
    assert (tmp_path / "fred-md" / f"{TEST_VINTAGE}.csv").is_file()


def test_checksum_mismatch_raises_and_caches_nothing(
    monkeypatch: pytest.MonkeyPatch, fake_download: list[str], tmp_path: Path
):
    monkeypatch.setitem(REGISTRY, ("fred-md", TEST_VINTAGE), "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load("fred-md", TEST_VINTAGE, cache_dir=tmp_path)
    assert not (tmp_path / "fred-md" / f"{TEST_VINTAGE}.csv").exists()
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_corrupt_cached_file_is_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setitem(REGISTRY, ("fred-md", TEST_VINTAGE), _sha256(MD_FILE))
    cached = tmp_path / "fred-md" / f"{TEST_VINTAGE}.csv"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(MD_FILE.read_bytes().replace(b"100.40", b"100.41"))
    with pytest.raises(ValueError, match=r"checksum mismatch.*removed"):
        load("fred-md", TEST_VINTAGE, cache_dir=tmp_path)
    assert not cached.exists()


def test_file_url_with_fake_registry_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setitem(REGISTRY, ("fred-qd", TEST_VINTAGE), _sha256(QD_FILE))
    data = _load("fred-qd", tmp_path / "cache", url=QD_FILE.as_uri())  # real urllib, file://
    assert data.variables == ("GDPC1", "GDPCTPI", "FEDFUNDS")

    monkeypatch.setitem(REGISTRY, ("fred-md", TEST_VINTAGE), _sha256(QD_FILE))  # wrong file
    with pytest.raises(ValueError, match="checksum mismatch"):
        load("fred-md", TEST_VINTAGE, cache_dir=tmp_path / "cache", url=MD_FILE.as_uri())
    assert not (tmp_path / "cache" / "fred-md").exists()


def test_missing_file_url_raises_oserror(tmp_path: Path):
    with pytest.raises(OSError, match="failed"):
        load("fred-md", TEST_VINTAGE, cache_dir=tmp_path, url=(tmp_path / "nope.csv").as_uri())


@pytest.mark.usefixtures("fake_download")
def test_unregistered_vintage_warns_with_hash(tmp_path: Path):
    digest = _sha256(MD_FILE)
    with pytest.warns(UserWarning, match=f"no registered checksum.*sha256={digest}"):
        _load("fred-md", tmp_path, vintage="2000-unregistered")
    assert (tmp_path / "fred-md" / "2000-unregistered.csv").is_file()
    with pytest.warns(UserWarning, match="ragged edge") as record:  # verify=False: no warning
        load("fred-md", "2000-unregistered", cache_dir=tmp_path, verify=False)
    assert not any("checksum" in str(w.message) for w in record)


# ---------------------------------------------------------------------------- network


@pytest.mark.network
@pytest.mark.parametrize("name", ["fred-md", "fred-qd"])
def test_download_pinned_vintages(name: str, tmp_path: Path):
    path = fred.fetch(name, "2026-09", cache_dir=tmp_path)
    assert path == tmp_path / name / "2026-09.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRY[(name, "2026-09")]
    expected = ("interpolated", "leading periods", "ragged edge")
    for size, codes in VARIABLE_SETS[name].items():
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            data = load(name, "2026-09", size, cache_dir=tmp_path)
        messages = [str(w.message) for w in record]
        assert not [m for m in messages if not any(s in m for s in expected)]
        if name == "fred-md":  # BLS published no October 2025 CPI (federal shutdown)
            assert any("CPIAUCSL: 2025-10" in m for m in messages)
        assert data.variables == tuple(codes)
        assert data.tcodes == dict(codes)
        assert data.vintage == f"{name}:2026-09"
        assert len(data) > 200
        assert np.isfinite(data.values).all()
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # downloaded once, reused
