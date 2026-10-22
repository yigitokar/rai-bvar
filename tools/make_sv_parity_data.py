"""Parity datasets and NetCDF packing for the Phase 2 R fixtures (DESIGN §9.9, work package WP8).

Run offline, never in CI. Three steps, each a sub-command (``python -I`` keeps the repository off
``sys.path``)::

    python -I tools/make_sv_parity_data.py data            # tests/parity/fixtures/data/*.csv
    Rscript tools/make_r_fixtures.R --sections=stochvol,bvars --staging=DIR
    python -I tools/make_sv_parity_data.py pack DIR        # tests/parity/fixtures/*.nc
    python -I tools/make_sv_parity_data.py all [--quick]   # the three steps in order
    python -I tools/make_sv_parity_data.py check           # regenerate the data, compare to the CSVs

``data`` writes the datasets with NumPy, so the data-generating process lives in our code and the
R side only reads committed numbers (N02 §13.6, N03 §7.4):

* the stochvol datasets of DESIGN §9.9: (i) ``T = 300``, ``μ = −1``, ``φ = 0.95``, ``σ = 0.2``;
  (ii) ``φ = 0.5``, ``σ = 0.5``; (iii) ``φ = 0.99``, ``σ = 0.1``; (iv) standardised AR(4) OLS
  residuals of real-GDP growth (``GDPC1`` of the Phase 1 parity file ``qd_small_levels.csv``);
  (v) the ``h`` and ``ε`` paths of (i) with standardized Student-t errors, ``ν = 5``
  (``τ_t ~ IG(ν/2, (ν−2)/2)``, D-8). Model: ``h_0 ~ N(μ, σ²/(1−φ²))``,
  ``h_t = μ + φ(h_{t−1} − μ) + σ η_t``, ``y_t = e^{h_t/2} √τ_t ε_t`` (``τ ≡ 1`` except (v));
* reduced-form VAR-SV datasets for the system-wide Gibbs oracle that replaces the bayesianVARs
  runs (bayesianVARs could not be built here, see the ``bvars`` section of the R script):
  ``y_t = c + Σ_l B_l y_{t−l} + A⁻¹ diag(e^{h_t/2}) ε_t`` with ``A`` unit lower triangular
  (``A u_t = e_t``, N04 §2) and independent AR(1) log-volatilities;
* the bayesianVARs probe datasets of DESIGN §9.9 (P1–P5, N04 §9.8, §10 item 4).

Every CSV has a ``date`` column (quarterly periods ending before 2020, so no COVID-sample warning
fires) and float columns written with ``%.17g`` (R's ``read.csv`` and
``pandas.read_csv(float_precision="round_trip")`` recover the doubles exactly). The SV data never
contain zeros (stochvol would otherwise add an offset).

``pack`` turns the JSON files that ``tools/make_r_fixtures.R`` stages (R has no NetCDF writer in
this environment) into NetCDF-3 files (``xarray``, scipy engine) and adds the data-generating
parameters of :data:`DATASETS` as ``dgp_*`` attributes. The variables and attributes of each
fixture kind are described in ``tests/parity/test_fixture_schema.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "parity" / "fixtures"
DATA_DIR = FIXTURES / "data"
QD_FILE = FIXTURES / "qd_small_levels.csv"
R_SCRIPT = ROOT / "tools" / "make_r_fixtures.R"


# ------------------------------------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """One committed CSV: how it is generated and what its data-generating process is."""

    name: str
    kind: str  # "sv" (univariate), "var" (reduced-form VAR), "probe" (bayesianVARs probes)
    start: str  # first quarterly period of the ``date`` column
    seed: int
    dgp: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def file(self) -> str:
        return f"{self.name}.csv"


SV_T = 300
SV_MU = -1.0
NU_T = 5

DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset(
            "sv_i",
            "sv",
            "1940Q1",
            20261022,
            {"T": SV_T, "mu": SV_MU, "phi": 0.95, "sigma": 0.2, "tails": "normal"},
            "stochvol dataset (i): Gaussian SV, T = 300, mu = -1, phi = 0.95, sigma = 0.2",
        ),
        Dataset(
            "sv_ii",
            "sv",
            "1940Q1",
            20261023,
            {"T": SV_T, "mu": SV_MU, "phi": 0.5, "sigma": 0.5, "tails": "normal"},
            "stochvol dataset (ii): Gaussian SV, T = 300, mu = -1, phi = 0.5, sigma = 0.5",
        ),
        Dataset(
            "sv_iii",
            "sv",
            "1940Q1",
            20261024,
            {"T": SV_T, "mu": SV_MU, "phi": 0.99, "sigma": 0.1, "tails": "normal"},
            "stochvol dataset (iii): Gaussian SV, T = 300, mu = -1, phi = 0.99, sigma = 0.1",
        ),
        Dataset(
            "sv_iv",
            "sv",
            "",  # dates come from qd_small_levels.csv
            0,
            {"source": "qd_small_levels.csv:GDPC1", "transform": "400*diff(log)", "ar_lags": 4},
            "stochvol dataset (iv): standardised AR(4) OLS residuals (with intercept) of "
            "annualised real-GDP growth 400*diff(log GDPC1), divided by their sample s.d.",
        ),
        Dataset(
            "sv_v",
            "sv",
            "1940Q1",
            20261025,
            {
                "T": SV_T,
                "mu": SV_MU,
                "phi": 0.95,
                "sigma": 0.2,
                "tails": "t",
                "nu": NU_T,
                "tau": "IG(nu/2, (nu-2)/2)",
                "base": "sv_i",
            },
            "stochvol dataset (v): the h and eps paths of (i) with standardized Student-t "
            "errors, nu = 5, tau_t ~ IG(nu/2, (nu-2)/2)",
        ),
        Dataset(
            "var_sv_n3",
            "var",
            "1960Q1",
            20261026,
            {"n": 3, "lags": 2, "T_raw": 202, "burn": 500},
            "reduced-form VAR(2)-SV, n = 3 (system-wide Gibbs oracle data)",
        ),
        Dataset(
            "var_sv_n7",
            "var",
            "1960Q1",
            20261027,
            {"n": 7, "lags": 4, "T_raw": 204, "burn": 500},
            "reduced-form VAR(4)-SV, n = 7 (system-wide Gibbs oracle data)",
        ),
        Dataset(
            "probe_tiny",
            "probe",
            "1990Q1",
            20261028,
            {"n": 2, "T_raw": 5},
            "bayesianVARs probes P1 (normal_sds: variance or s.d.) and P5 "
            "(cholesky_priorhomoscedastic scale): T_eff = 4 at p = 1, prior-dominated",
        ),
        Dataset(
            "probe_corr",
            "probe",
            "1860Q1",
            20261029,
            {
                "n": 2,
                "T_raw": 601,
                "intercept": [5.0, -3.0],
                "own_lag": 0.3,
                "sd": [1.0, 2.0],
                "corr": 0.8,
            },
            "bayesianVARs probes P2 (U vs U^-1 and sign: corr(u1, u2) = +0.8) and P3 "
            "(intercept row of PHI: intercepts 5 and -3), VAR(1), T = 600",
        ),
        Dataset(
            "probe_sv",
            "probe",
            "1900Q1",
            20261030,
            {
                "n": 3,
                "T_raw": 401,
                "own_lag": 0.2,
                "mu": [-3.0, 0.0, 2.0],
                "phi": [0.8, 0.9, 0.97],
                "sigma": [0.4, 0.3, 0.15],
            },
            "bayesianVARs probe P4 (sv_para layout): VAR(1) with diagonal A and SV "
            "parameters that differ across equations, T = 400",
        ),
    )
}


# ------------------------------------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------------------------------------


def _quarters(start: str, T: int) -> list[str]:
    return [str(p) for p in pd.period_range(start=start, periods=T, freq="Q")]


def _sv_paths(
    rng: np.random.Generator, T: int, mu: float, phi: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary AR(1) log-variance ``h_{0:T}`` and the standard-normal ``ε_{1:T}``."""
    h = np.empty(T + 1)
    h[0] = mu + sigma / np.sqrt(1.0 - phi**2) * rng.standard_normal()
    eta = rng.standard_normal(T)
    for t in range(1, T + 1):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma * eta[t - 1]
    eps = rng.standard_normal(T)
    return h, eps


def _sv_frame(ds: Dataset) -> pd.DataFrame:
    g = ds.dgp
    if ds.name == "sv_iv":
        return _gdp_residuals()
    base = DATASETS[g.get("base", ds.name)]
    rng = np.random.default_rng(base.seed)
    h, eps = _sv_paths(rng, g["T"], g["mu"], g["phi"], g["sigma"])
    frame = pd.DataFrame({"date": _quarters(ds.start, g["T"])})
    if g["tails"] == "t":
        nu = g["nu"]
        tau = 0.5 * (nu - 2.0) / np.random.default_rng(ds.seed).gamma(0.5 * nu, size=g["T"])
        frame["y"] = np.exp(0.5 * h[1:]) * np.sqrt(tau) * eps
        frame["h"] = h[1:]
        frame["tau"] = tau
    else:
        frame["y"] = np.exp(0.5 * h[1:]) * eps
        frame["h"] = h[1:]
    return frame


def _gdp_residuals() -> pd.DataFrame:
    """Dataset (iv): ``y = (e − ē)/sd(e)``, ``e`` the AR(4) OLS residuals of GDP growth."""
    raw = pd.read_csv(QD_FILE, float_precision="round_trip")
    level = raw["GDPC1"].to_numpy(np.float64)
    growth = 400.0 * np.diff(np.log(level))
    p = DATASETS["sv_iv"].dgp["ar_lags"]
    Y = growth[p:]
    X = np.column_stack([np.ones(Y.size)] + [growth[p - lag : -lag] for lag in range(1, p + 1)])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ coef
    y = (e - e.mean()) / e.std(ddof=1)
    dates = raw["date"].to_numpy()[1 + p :]
    return pd.DataFrame({"date": dates, "y": y})


def _companion_radius(B: list[np.ndarray]) -> float:
    n, p = B[0].shape[0], len(B)
    C = np.zeros((n * p, n * p))
    C[:n, :] = np.hstack(B)
    C[n:, :-n] = np.eye(n * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def var_parameters(ds: Dataset) -> dict[str, np.ndarray]:
    """True parameters of a VAR-SV dataset (``B_l`` is ``n × n`` with ``y_t = c + Σ B_l y_{t−l}``)."""
    n, p = ds.dgp["n"], ds.dgp["lags"]
    rng = np.random.default_rng(ds.seed + 1_000_000)
    if n == 3:
        c = np.array([0.5, -0.3, 0.2])
        B = [
            np.array([[0.5, 0.1, 0.0], [0.05, 0.4, 0.1], [0.0, -0.1, 0.6]]),
            np.array([[0.15, 0.0, 0.0], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]),
        ]
        A = np.array([[1.0, 0.0, 0.0], [-0.5, 1.0, 0.0], [0.3, -0.4, 1.0]])
        mu = np.array([-0.5, 0.0, 0.5])
        phi = np.array([0.95, 0.9, 0.97])
        sigma = np.array([0.25, 0.3, 0.2])
    else:
        c = rng.uniform(-0.5, 0.5, n)
        own = [0.45, 0.1, 0.05, 0.02]
        B = [np.diag(np.full(n, own[lag])) for lag in range(p)]
        B[0] = B[0] + np.where(np.eye(n) == 1, 0.0, rng.uniform(-0.1, 0.1, (n, n)))
        A = np.eye(n) + np.tril(rng.normal(0.0, 0.3, (n, n)), -1)
        mu = np.linspace(-1.0, 1.0, n)
        phi = np.linspace(0.9, 0.98, n)
        sigma = np.linspace(0.3, 0.15, n)
    radius = _companion_radius(B)
    if radius >= 0.95:
        raise ValueError(f"{ds.name}: companion spectral radius {radius:.3f} >= 0.95")
    return {"c": c, "B": np.stack(B), "A": A, "mu": mu, "phi": phi, "sigma": sigma}


def _var_frame(ds: Dataset) -> pd.DataFrame:
    n, p, T_raw, burn = ds.dgp["n"], ds.dgp["lags"], ds.dgp["T_raw"], ds.dgp["burn"]
    par = var_parameters(ds)
    rng = np.random.default_rng(ds.seed)
    total = burn + T_raw
    h = par["mu"] + par["sigma"] / np.sqrt(1.0 - par["phi"] ** 2) * rng.standard_normal(n)
    y = np.zeros((total + p, n))
    for t in range(p, total + p):
        h = par["mu"] + par["phi"] * (h - par["mu"]) + par["sigma"] * rng.standard_normal(n)
        e = np.exp(0.5 * h) * rng.standard_normal(n)
        u = np.linalg.solve(par["A"], e)  # A u_t = e_t (A unit lower triangular)
        y[t] = par["c"] + sum(par["B"][lag] @ y[t - 1 - lag] for lag in range(p)) + u
    y = y[-T_raw:]
    frame = pd.DataFrame(y, columns=[f"y{i + 1}" for i in range(n)])
    frame.insert(0, "date", _quarters(ds.start, T_raw))
    return frame


def _probe_frame(ds: Dataset) -> pd.DataFrame:
    g, rng = ds.dgp, np.random.default_rng(ds.seed)
    n, T_raw = g["n"], g["T_raw"]
    if ds.name == "probe_tiny":
        y = rng.standard_normal((T_raw, n))
    elif ds.name == "probe_corr":
        sd, corr = np.asarray(g["sd"]), g["corr"]
        cov = np.outer(sd, sd) * np.array([[1.0, corr], [corr, 1.0]])
        chol = np.linalg.cholesky(cov)
        c = np.asarray(g["intercept"])
        y = np.zeros((T_raw + 200, n))
        y[0] = c / (1.0 - g["own_lag"])
        for t in range(1, y.shape[0]):
            y[t] = c + g["own_lag"] * y[t - 1] + chol @ rng.standard_normal(n)
        y = y[-T_raw:]
    else:  # probe_sv
        mu, phi, sigma = (np.asarray(g[key]) for key in ("mu", "phi", "sigma"))
        h = mu + sigma / np.sqrt(1.0 - phi**2) * rng.standard_normal(n)
        y = np.zeros((T_raw + 200, n))
        for t in range(1, y.shape[0]):
            h = mu + phi * (h - mu) + sigma * rng.standard_normal(n)
            y[t] = g["own_lag"] * y[t - 1] + np.exp(0.5 * h) * rng.standard_normal(n)
        y = y[-T_raw:]
    frame = pd.DataFrame(y, columns=[f"y{i + 1}" for i in range(n)])
    frame.insert(0, "date", _quarters(ds.start, T_raw))
    return frame


_GENERATORS: dict[str, Callable[[Dataset], pd.DataFrame]] = {
    "sv": _sv_frame,
    "var": _var_frame,
    "probe": _probe_frame,
}


def generate(name: str) -> pd.DataFrame:
    """The dataset ``name`` as a DataFrame (``date`` plus float columns); validated."""
    ds = DATASETS[name]
    frame = _GENERATORS[ds.kind](ds)
    values = frame.drop(columns="date").to_numpy(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite values")
    if ds.kind == "sv" and not np.all(np.abs(frame["y"].to_numpy()) > 0.0):
        raise ValueError(f"{name}: SV data must not contain zeros (stochvol would add an offset)")
    if pd.PeriodIndex(frame["date"], freq="Q")[-1] >= pd.Period("2020Q1", freq="Q"):
        raise ValueError(f"{name}: the sample must end before 2020")
    return frame


def to_csv_text(frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, float_format="%.17g", lineterminator="\n")
    return buf.getvalue()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_data(out: Path = DATA_DIR) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, ds in DATASETS.items():
        path = out / ds.file
        path.write_text(to_csv_text(generate(name)))
        paths.append(path)
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return paths


def check_data(out: Path = DATA_DIR) -> int:
    """0 if every committed CSV equals a fresh regeneration (NumPy stream changes break this)."""
    bad = 0
    for name, ds in DATASETS.items():
        committed = pd.read_csv(out / ds.file, float_precision="round_trip")
        fresh = generate(name)
        same = list(committed.columns) == list(fresh.columns) and np.array_equal(
            committed.drop(columns="date").to_numpy(), fresh.drop(columns="date").to_numpy()
        )
        print(f"{ds.file}: {'identical' if same else 'DIFFERS'}")
        bad += not same
    return int(bad > 0)


# ------------------------------------------------------------------------------------------------
# Packing staged R output into NetCDF
# ------------------------------------------------------------------------------------------------


def _array(spec: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(s) for s in spec["shape"])
    data, dtype = spec["data"], spec.get("dtype")
    if dtype == "float64":  # "%.17g" strings: exact doubles ("NA" is R's missing value)
        values = [np.nan if v is None or v == "NA" else float(v) for v in data]
        return np.asarray(values, dtype=np.float64).reshape(shape)
    if dtype == "int32":
        return np.asarray(data, dtype=np.int32).reshape(shape)
    if data and any(isinstance(v, str) for v in data):
        return np.asarray(data, dtype=str).reshape(shape)
    return np.asarray([np.nan if v is None else v for v in data], dtype=np.float64).reshape(shape)


def _attr_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, list):
        if all(isinstance(v, bool) for v in value):
            return np.asarray(value, dtype=np.int32)
        if all(isinstance(v, int | float) for v in value):
            return np.asarray(value, dtype=np.float64)
        return "\n".join(str(v) for v in value)
    return value


def _flat_attrs(attrs: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        name = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            out.update(_flat_attrs(value, f"{name}_"))
        else:
            out[name] = _attr_value(value)
    return out


def _dgp_attrs(data_file: str) -> dict[str, Any]:
    ds = next((d for d in DATASETS.values() if d.file == data_file), None)
    if ds is None:
        return {}
    out: dict[str, Any] = {"dgp_description": ds.description, "dgp_seed": ds.seed}
    out.update(_flat_attrs(ds.dgp, "dgp_"))
    if ds.kind == "var":
        par = var_parameters(ds)
        for key, value in par.items():
            out[f"dgp_{key}"] = np.asarray(value, dtype=np.float64).ravel()
        out["dgp_B_layout"] = "dgp_B is B[l, i, j] (lag l, equation i, variable j), C order"
        out["dgp_A_layout"] = "dgp_A is A[i, j] row-major; A u_t = e_t"
    return out


def staged_to_dataset(staged: dict[str, Any]):
    """xarray Dataset from one staged JSON object written by ``tools/make_r_fixtures.R``."""
    import xarray as xr

    coords = {name: (spec["dims"], _array(spec)) for name, spec in staged["coords"].items()}
    data_vars = {}
    for name, spec in staged["vars"].items():
        attrs = {k: v for k, v in (spec.get("attrs") or {}).items() if v is not None}
        data_vars[name] = xr.Variable(spec["dims"], _array(spec), attrs=attrs)
    ds = xr.Dataset(data_vars, coords=coords)
    attrs = _flat_attrs(staged["attrs"])
    if "data_file" in attrs:
        attrs.update(_dgp_attrs(Path(str(attrs["data_file"])).name))
    attrs["packed_by"] = "tools/make_sv_parity_data.py pack"
    ds.attrs = attrs
    return ds


def pack(staging: Path, out: Path = FIXTURES) -> list[Path]:
    files = sorted(staging.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no staged JSON files in {staging}")
    written = []
    for path in files:
        staged = json.loads(path.read_text())
        ds = staged_to_dataset(staged)
        target = out / f"{staged['name']}.nc"
        ds.to_netcdf(target, engine="scipy", format="NETCDF3_64BIT")
        written.append(target)
        print(f"wrote {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")
    return written


def run_r(staging: Path, quick: bool) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript not found on PATH")
    cmd = [rscript, str(R_SCRIPT), "--sections=stochvol,bvars", f"--staging={staging}"]
    if quick:
        cmd.append("--quick")
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="write tests/parity/fixtures/data/*.csv")
    sub.add_parser("check", help="compare a fresh regeneration with the committed CSVs")
    p_pack = sub.add_parser("pack", help="convert staged R JSON into NetCDF fixtures")
    p_pack.add_argument("staging", type=Path)
    p_pack.add_argument("--out", type=Path, default=FIXTURES)
    p_all = sub.add_parser("all", help="data, then the R sections, then pack")
    p_all.add_argument(
        "--quick",
        action="store_true",
        help="short chains (smoke test: mixing checks only warn); packs into the staging dir",
    )
    p_all.add_argument("--staging", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "data":
        write_data()
    elif args.command == "check":
        return check_data()
    elif args.command == "pack":
        pack(args.staging, args.out)
    else:
        staging = args.staging or Path(tempfile.mkdtemp(prefix="rai_bvar_r_fixtures_"))
        staging.mkdir(parents=True, exist_ok=True)
        write_data()
        run_r(staging, args.quick)
        pack(staging, staging if args.quick else FIXTURES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
