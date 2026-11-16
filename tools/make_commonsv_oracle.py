"""The exact ``CommonSV`` oracle's dataset and fixtures (docs/phase3/DESIGN.md §9.8, WP38a).

Run offline, never in CI (``python -I`` keeps the repository off ``sys.path``)::

    python -I tools/make_commonsv_oracle.py data              # fixtures/data/commonsv_n3.csv (C2)
    python -I tools/make_commonsv_oracle.py check             # regenerate C2's data, compare
    python -I tools/make_commonsv_oracle.py run C1 0 DIR      # one run -> DIR/C1_run0.npz
    python -I tools/make_commonsv_oracle.py pack C1 DIR       # fixtures/commonsv_oracle_c1.nc
    python -I tools/make_commonsv_oracle.py all [--cases C1 C2] [--staging DIR]
    python -I tools/make_commonsv_oracle.py all --quick       # smoke test, packed into DIR only

``all`` starts the ``4`` runs of every case as parallel processes (one core each), then packs.

**Runs** (§9.8, fixed before any output existed and never lengthened): every case is 4
independent runs (single chains, :func:`commonsv_exact.run`) × 200,000 kept sweeps after 10,000,
each with its own recorded seed; the predictive (one path per kept sweep, 4 steps) comes from a
separate recorded seed per run. **C1**: ``qd3_growth.csv`` (``n = 3, p = 4``, WP29's data, E146),
no dummy rows, ``ψ`` the sample variances of the series and own-lag means 0. **C2**:
``commonsv_n3.csv`` (``n = 3, p = 2, T = 200``) simulated here from the model with
``default_rng(20261026)``, SOC and SUR rows with ``μ = δ = 1``, ``ψ = 1``, own-lag means 0.5.
Both: ``λ = 0.2``, ``α = 2``, ``var_const = 1e7``, ``d = n + 2``, CommonSV's defaults.

**Fixtures** ``tests/parity/fixtures/commonsv_oracle_{c1,c2}.nc`` (NetCDF-3, ``xarray`` with the
scipy engine, as Phase 2's). For every compared quantity of §9.8's Comparison (every ``B`` entry
over ``(regressor, equation)``; the unique ``Σ`` entries over ``sigma_entry``; ``h_t`` at
``t_sel = {0, ⌊T/4⌋, ⌊T/2⌋, T − 1}``; ``rho`` and ``sigma_h`` over ``sv_param``; the 1- and
4-step predictive over ``(horizon, equation)``) it stores Phase 2's summary block (``tests/parity/
test_fixture_schema.py``): ``*_q`` (type-7 quantiles 5/25/50/75/95% of the pooled runs),
``*_mcse`` = max(``*_mcse_vehtari``, Vehtari et al. 2021 §4.3; ``*_mcse_batch``, 20 batches per
run), ``*_mean``, ``*_sd``, ``*_ess`` (bulk), ``*_mcse_mean``, ``*_rhat`` (rank-normalised split
R-hat), plus ``*_ess_tail``, ``*_ess_run`` (the bulk ESS of each run alone) and ``*_mixing_ok``
(``rhat ≤ 1.01`` and ``ess ≥ 400``: Q39(b)'s gate, a failure is recorded as not met, never
rerun). The diagnostics are ``rai_bvar.diagnostics``', so both sides of WP38's comparison use one
estimator; the oracle module itself imports nothing from ``rai_bvar``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "parity" / "fixtures"
DATA_DIR = FIXTURES / "data"
ORACLE = ROOT / "tests" / "parity" / "oracles" / "commonsv_exact.py"

CHAINS = 4
DRAWS = 200_000
BURNIN = 10_000
QUICK_DRAWS, QUICK_BURNIN = 2_000, 500
HORIZON = 4
COMPARED_HORIZONS = (1, 4)  # §9.8 Comparison: the 1- and 4-step predictive quantiles
PROBS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
MCSE_BATCHES = 20  # batches per run of the batch-quantile MCSE (Phase 2's fixtures' method)
RHAT_MAX = 1.01  # §9.8 Comparison / Q39(b): a compared quantity with split R-hat above this ...
ESS_MIN = 400.0  # ... or bulk ESS below this is not met

# C2's data-generating process (§9.8 Runs; the draw order is this tool's, the CSV the record).
C2_DGP: dict[str, Any] = {
    "seed": 20261026,
    "n": 3,
    "lags": 2,
    "T": 200,
    "burn": 50,
    "start": "1960Q1",
    "c": [0.2, 0.1, 0.3],
    "A1": 0.5,
    "A2": 0.1,
    "corr_12": 0.3,
    "corr_13": 0.1,
    "corr_23": 0.2,
    "rho": 0.9,
    "sigma_h": 0.15,
}


@cache
def oracle() -> ModuleType:
    """``tests/parity/oracles/commonsv_exact.py`` imported from its path."""
    name = "_rai_bvar_commonsv_exact"
    spec = importlib.util.spec_from_file_location(name, ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------
# Cases
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One oracle case of §9.8: data, lags, prior rule and the recorded seeds of its runs."""

    name: str
    data_file: str
    lags: int
    psi_rule: str  # "sample_variance" (C1) or "one" (C2)
    mean: float  # own-lag prior mean of every variable
    soc: float | None
    sur: float | None
    seeds: tuple[int, ...]
    predictive_seeds: tuple[int, ...]

    @property
    def fixture(self) -> str:
        return f"commonsv_oracle_{self.name.lower()}"

    def frame(self) -> pd.DataFrame:
        return pd.read_csv(DATA_DIR / self.data_file, float_precision="round_trip")

    def values(self) -> np.ndarray:
        return self.frame().drop(columns="date").to_numpy(np.float64)

    def psi(self, values: np.ndarray) -> tuple[float, ...]:
        if self.psi_rule == "sample_variance":
            # the sample variance (divisor N − 1) of each series over every row of the file
            return tuple(float(v) for v in values.var(axis=0, ddof=1))
        return (1.0,) * values.shape[1]

    def prior(self, values: np.ndarray | None = None):
        values = self.values() if values is None else values
        n = values.shape[1]
        return oracle().Prior(
            psi=self.psi(values),
            means=(self.mean,) * n,
            lam=0.2,
            alpha=2.0,
            var_const=1e7,
            soc=self.soc,
            sur=self.sur,
        )

    def problem(self):
        values = self.values()
        return oracle().build_problem(values, self.lags, self.prior(values))


CASES: dict[str, Case] = {
    "C1": Case(
        "C1",
        "qd3_growth.csv",
        4,
        "sample_variance",
        0.0,
        None,
        None,
        seeds=(20261111, 20261112, 20261113, 20261114),
        predictive_seeds=(20261121, 20261122, 20261123, 20261124),
    ),
    "C2": Case(
        "C2",
        "commonsv_n3.csv",
        2,
        "one",
        0.5,
        1.0,
        1.0,
        seeds=(20261211, 20261212, 20261213, 20261214),
        predictive_seeds=(20261221, 20261222, 20261223, 20261224),
    ),
}


# ------------------------------------------------------------------------------------------------
# C2's data
# ------------------------------------------------------------------------------------------------


def c2_sigma() -> np.ndarray:
    g = C2_DGP
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = g["corr_12"]
    corr[0, 2] = corr[2, 0] = g["corr_13"]
    corr[1, 2] = corr[2, 1] = g["corr_23"]
    return corr  # unit variances


def simulate_c2() -> tuple[pd.DataFrame, np.ndarray]:
    """C2's ``p + T`` rows and the true ``h`` of its ``T`` data rows.

    ``y_t = c + A₁y_{t−1} + A₂y_{t−2} + e^{h_t/2}Lε_t`` with ``Σ = LLᵀ`` (unit variances,
    correlations 0.3, 0.1, 0.2), ``h_t = 0.9h_{t−1} + 0.15η_t`` from its stationary law, ``y``
    started at 0; ``burn + p + T`` rows are simulated and the first ``burn = 50`` discarded. Draw
    order, all from ``default_rng(20261026)``: the ``burn + p + T`` standard normals of the ``h``
    path (``h_0``'s first), then the ``(burn + p + T, n)`` standard normals ``ε``.
    """
    g = C2_DGP
    n, lags, T, burn = g["n"], g["lags"], g["T"], g["burn"]
    total = burn + lags + T
    rng = np.random.default_rng(g["seed"])
    z = rng.standard_normal(total)
    eps = rng.standard_normal((total, n))
    rho, sigma_h = g["rho"], g["sigma_h"]
    h = np.empty(total)
    h[0] = sigma_h / math.sqrt(1.0 - rho**2) * z[0]
    for t in range(1, total):
        h[t] = rho * h[t - 1] + sigma_h * z[t]
    chol = np.linalg.cholesky(c2_sigma())
    c = np.asarray(g["c"])
    y = np.zeros((total + lags, n))  # two zero rows before the first simulated row
    for t in range(total):
        lag1, lag2 = y[lags + t - 1], y[lags + t - 2]
        y[lags + t] = c + g["A1"] * lag1 + g["A2"] * lag2 + math.exp(0.5 * h[t]) * chol @ eps[t]
    y, h = y[lags + burn :], h[burn:]
    frame = pd.DataFrame(y, columns=[f"y{i + 1}" for i in range(n)])
    dates = pd.period_range(start=g["start"], periods=lags + T, freq="Q")
    frame.insert(0, "date", [str(p) for p in dates])
    return frame, h[lags:]


def to_csv_text(frame: pd.DataFrame) -> str:
    """``%.17g`` floats (exact doubles under ``read_csv(float_precision="round_trip")``)."""
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_data() -> Path:
    path = DATA_DIR / CASES["C2"].data_file
    path.write_text(to_csv_text(simulate_c2()[0]))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def check_data() -> int:
    """The committed C2 CSV against a fresh simulation, by the rule of the test
    ``test_c2_data_regenerate_from_the_stated_process`` (``tests/parity/oracles/
    test_commonsv_exact.py``): columns and dates exactly, values with ``rtol = atol = 1e-12`` (the
    same draws; floating-point rounding differs across CPU models and BLAS builds, ≈ 4e-16 between
    arm64 and x86). Prints the max |difference|; returns 1 when the files differ."""
    name = CASES["C2"].data_file
    committed = pd.read_csv(DATA_DIR / name, float_precision="round_trip")
    fresh = simulate_c2()[0]
    if list(committed.columns) != list(fresh.columns) or committed.shape != fresh.shape:
        print(f"{name}: DIFFERS (columns {list(committed.columns)} and shape {committed.shape}; "
              f"fresh {list(fresh.columns)} and {fresh.shape})")  # fmt: skip
        return 1
    dates = np.array_equal(committed["date"].to_numpy(), fresh["date"].to_numpy())
    a = committed.drop(columns="date").to_numpy()
    b = fresh.drop(columns="date").to_numpy()
    diff = float(np.max(np.abs(a - b)))
    close = np.allclose(a, b, rtol=1e-12, atol=1e-12)
    if not dates:
        verdict = "DIFFERS (dates)"
    elif not close:
        verdict = "DIFFERS (values beyond rtol = atol = 1e-12)"
    else:
        verdict = "identical" if diff == 0.0 else "equal to rounding (rtol = atol = 1e-12)"
    print(f"{name}: {verdict}; max |diff| = {diff:.3g}")
    return int(not (dates and close))


# ------------------------------------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------------------------------------


def run_one(case: Case, index: int, out: Path, quick: bool = False) -> Path:
    """Run ``index`` of ``case``: the chain, then its predictive; saved as ``.npz``."""
    cx = oracle()
    problem = case.problem()
    keep, burn = (QUICK_DRAWS, QUICK_BURNIN) if quick else (DRAWS, BURNIN)
    log = out / f"{case.name}_run{index}.log"

    def progress(done: int, seconds: float) -> None:
        with log.open("a") as fh:
            fh.write(f"{done} sweeps, {seconds:.1f} s\n")

    draws = cx.run(problem, burn, keep, case.seeds[index], progress=progress)
    start = time.perf_counter()
    forecast = cx.predictive(
        problem,
        draws["B"],
        draws["Sigma"],
        draws["h_last"],
        draws["rho"],
        draws["sigma_h"],
        np.random.default_rng(case.predictive_seeds[index]),
        horizon=HORIZON,
    )
    path = out / f"{case.name}_run{index}.npz"
    np.savez(
        path,
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=int(bool(git("status", "--porcelain", "--untracked-files=no"))),
        oracle_md5=md5(ORACLE),
        data_md5=md5(DATA_DIR / case.data_file),
        cpu=cpu_name(),
        forecast=forecast,
        predictive_seconds=time.perf_counter() - start,
        seed=case.seeds[index],
        predictive_seed=case.predictive_seeds[index],
        keep=keep,
        burn=burn,
        **{key: np.asarray(value) for key, value in draws.items()},
    )
    print(f"wrote {path} ({draws['wall_time']:.1f} s)")
    return path


def run_parallel(cases: list[Case], staging: Path, quick: bool) -> None:
    """Every run of ``cases`` as its own process (single-threaded BLAS), then wait for all."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):  # fmt: skip
        env[var] = "1"
    procs = []
    for case in cases:
        for index in range(CHAINS):
            cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "run", case.name,
                   str(index), str(staging)] + (["--quick"] if quick else [])  # fmt: skip
            procs.append((case.name, index, subprocess.Popen(cmd, env=env, cwd=ROOT)))
    failed = [(name, i) for name, i, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit(f"runs failed: {failed}")


# ------------------------------------------------------------------------------------------------
# Summaries and packing
# ------------------------------------------------------------------------------------------------


def batch_quantile_mcse(x: np.ndarray, batches: int = MCSE_BATCHES) -> np.ndarray:
    """``sd(q̂_b)/√(C·B)`` over the ``B`` consecutive batches of every run (``x``: ``(C, D, *S)``;
    returns ``(*S, 5)``), as Phase 2's fixtures and ``tests/parity/test_stochvol.py``."""
    parts = np.array_split(np.arange(x.shape[1]), batches)
    bq = np.stack([np.quantile(x[:, idx], PROBS, axis=1) for idx in parts])  # (B, 5, C, *S)
    bq = np.moveaxis(bq, 2, 1).reshape(-1, len(PROBS), *x.shape[2:])  # (B·C, 5, *S)
    return np.moveaxis(bq.std(axis=0, ddof=1) / math.sqrt(bq.shape[0]), 0, -1)


def summarise(x: np.ndarray) -> dict[str, np.ndarray]:
    """Phase 2's summary block of draws ``x`` (``(C, D, *S)``) plus the per-run ESS and the gate."""
    from rai_bvar import diagnostics as dg

    pooled = x.reshape(-1, *x.shape[2:])
    q = np.moveaxis(np.quantile(pooled, PROBS, axis=0), 0, -1)
    vehtari = np.stack([dg.mcse_quantile_many(x, p) for p in PROBS], axis=-1)
    batch = batch_quantile_mcse(x)
    rhat, ess = dg.rhat_ess_bulk_many(x)
    out = {
        "_q": q,
        "_mcse": np.maximum(vehtari, batch),
        "_mcse_vehtari": vehtari,
        "_mcse_batch": batch,
        "_mean": pooled.mean(axis=0),
        "_sd": pooled.std(axis=0, ddof=1),
        "_ess": np.asarray(ess, dtype=np.float64),
        "_ess_tail": np.asarray(dg.ess_tail_many(x), dtype=np.float64),
        "_mcse_mean": np.asarray(dg.mcse_mean_many(x), dtype=np.float64),
        "_rhat": np.asarray(rhat, dtype=np.float64),
        "_ess_run": np.stack([dg.ess_bulk_many(x[c : c + 1]) for c in range(x.shape[0])]),
    }
    ok = (out["_rhat"] <= RHAT_MAX) & (out["_ess"] >= ESS_MIN)
    out["_mixing_ok"] = ok.astype(np.int32)  # NaN (a chain that never moves) fails
    return out


def cpu_name() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True)  # fmt: skip
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def engine_spec(case: Case, prior) -> str:
    """The composition both sides target (§9.8 Model), as a rai-bvar spec string."""
    dummies = ""
    if prior.soc is not None:
        dummies += f", soc={prior.soc!r}"
    if prior.sur is not None:
        dummies += f", sur={prior.sur!r}"
    psi = tuple(float(v) for v in prior.psi)
    return (
        'BVAR(mean=Linear(prior=Minnesota(form="conjugate", hierarchy="fixed", '
        f"lambda_={prior.lam!r}, alpha={prior.alpha!r}, psi={psi!r}, mean={case.mean!r}, "
        f'var_const={prior.var_const!r}{dummies}, soc_scope="persistent")), '
        f"vol=CommonSV(), lags={case.lags})"
    )


def load_runs(case: Case, staging: Path) -> list[dict[str, np.ndarray]]:
    runs = []
    for index in range(CHAINS):
        with np.load(staging / f"{case.name}_run{index}.npz") as npz:
            runs.append({key: npz[key] for key in npz.files})
    return runs


def pack(case: Case, staging: Path, out: Path = FIXTURES, quick: bool = False) -> Path:
    import scipy
    import xarray as xr

    cx = oracle()
    problem = case.problem()
    prior = problem.prior
    frame = case.frame()
    variables = [c for c in frame.columns if c != "date"]
    n, lags, T, k, m = problem.n, problem.lags, problem.T, problem.k, problem.m
    runs = load_runs(case, staging)
    for index, r in enumerate(runs):
        if (
            int(r["seed"]) != case.seeds[index]
            or int(r["predictive_seed"]) != (case.predictive_seeds[index])
        ):
            raise ValueError(f"{case.name} run {index}: seeds differ from the registry")
    keep = int(runs[0]["keep"])
    if not quick and (keep != DRAWS or int(runs[0]["burn"]) != BURNIN):
        raise ValueError(f"{case.name}: a full fixture needs {CHAINS} × {DRAWS} after {BURNIN}")

    def stack(key: str) -> np.ndarray:
        return np.stack([r[key] for r in runs])

    def run_attr(key: str):
        """A provenance field the runs recorded when they ran; the same in every run."""
        values = {str(r[key]) for r in runs}
        if len(values) != 1:
            raise ValueError(f"{case.name}: the runs differ in {key}: {sorted(values)}")
        return runs[0][key].item()

    if run_attr("data_md5") != md5(DATA_DIR / case.data_file):
        raise ValueError(f"{case.name}: the data changed after the runs")
    if run_attr("oracle_md5") != md5(ORACLE) and not quick:
        raise ValueError(f"{case.name}: the oracle changed after the runs")

    rows, cols = np.tril_indices(n)
    groups = {
        "B": (stack("B"), ("regressor", "equation")),
        "Sigma": (stack("Sigma")[:, :, rows, cols], ("sigma_entry",)),
        "h": (stack("h"), ("t_sel",)),
        "sv": (np.stack([stack("rho"), stack("sigma_h")], axis=-1), ("sv_param",)),
        "forecast": (
            stack("forecast")[:, :, [h - 1 for h in COMPARED_HORIZONS]],
            ("horizon", "equation"),
        ),
    }
    regressors = ["const"] + [f"{v}.L{lag}" for lag in range(1, lags + 1) for v in variables]
    coords: dict[str, Any] = {
        "quantile": ("quantile", PROBS),
        "run": ("run", np.arange(CHAINS, dtype=np.int32)),
        "regressor": ("regressor", regressors),
        "equation": ("equation", variables),
        "sigma_entry": (
            "sigma_entry",
            [f"{variables[i]},{variables[j]}" for i, j in zip(rows, cols, strict=True)],
        ),
        "sigma_row": ("sigma_entry", rows.astype(np.int32)),
        "sigma_col": ("sigma_entry", cols.astype(np.int32)),
        "t_sel": ("t_sel", cx.compared_dates(T).astype(np.int32)),
        "sv_param": ("sv_param", ["rho", "sigma_h"]),
        "horizon": ("horizon", np.asarray(COMPARED_HORIZONS, dtype=np.int32)),
        "t": ("t", np.arange(T, dtype=np.int32)),
        "row": ("row", np.arange(T + lags, dtype=np.int32)),
    }
    data_vars: dict[str, Any] = {}
    gate_rows, not_met = [], []
    for prefix, (draws, dims) in groups.items():
        summary = summarise(draws)
        for suffix, value in summary.items():
            if suffix in ("_q", "_mcse", "_mcse_vehtari", "_mcse_batch"):
                vdims = (*dims, "quantile")
            elif suffix == "_ess_run":
                vdims = ("run", *dims)
            else:
                vdims = dims
            data_vars[f"{prefix}{suffix}"] = (vdims, value)
        labels = _labels(prefix, dims, coords)
        for label, rhat, ess, ok in zip(
            labels,
            summary["_rhat"].ravel(),
            summary["_ess"].ravel(),
            summary["_mixing_ok"].ravel(),
            strict=True,
        ):
            gate_rows.append(
                f"{label}: rhat {rhat:.4f}, ess {ess:.0f}, {'met' if ok else 'NOT MET'}"
            )
            if not ok:
                not_met.append(label)
    data_vars["h_path_mean"] = (("t",), stack("h_path_mean").mean(axis=0))
    data_vars["h_path_mean_run"] = (("run", "t"), stack("h_path_mean"))
    data_vars["y"] = (("row", "equation"), case.values())
    data_vars["prior_omega"] = (("regressor",), problem.omega)
    data_vars["prior_b"] = (("regressor", "equation"), problem.b)
    data_vars["prior_psi"] = (("equation",), np.asarray(prior.psi))
    if m:
        coords["dummy"] = ("dummy", np.arange(m, dtype=np.int32))
        data_vars["dummy_y"] = (("dummy", "equation"), problem.Yd)
        data_vars["dummy_x"] = (("dummy", "regressor"), problem.Xd)
    attrs: dict[str, Any] = {
        "kind": "commonsv_oracle",
        "case": case.name,
        "oracle": "exact independent CommonSV oracle (docs/phase3/DESIGN.md §9.8, WP38a)",
        "generator": "tools/make_commonsv_oracle.py",
        "oracle_file": "tests/parity/oracles/commonsv_exact.py",
        "oracle_md5": run_attr("oracle_md5"),
        "git_commit": run_attr("git_commit"),
        "git_dirty": int(run_attr("git_dirty")),
        "packed_at_commit": git("rev-parse", "HEAD"),
        "cpu": run_attr("cpu"),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quick": int(quick),
        "data_file": f"data/{case.data_file}",
        "data_md5": md5(DATA_DIR / case.data_file),
        "variable_names": "\n".join(variables),
        "n": n,
        "lags": lags,
        "T": T,
        "k": k,
        "m": m,
        "chains": CHAINS,
        "draws": keep,
        "burnin": int(runs[0]["burn"]),
        "thin": 1,
        "seeds": np.asarray(case.seeds, dtype=np.int64),
        "predictive_seeds": np.asarray(case.predictive_seeds, dtype=np.int64),
        "rng": "numpy.random.default_rng(seed) (PCG64), one generator per run",
        "horizon": HORIZON,
        "regressor_order": "x_t = (1, y_{t-1}', ..., y_{t-p}')'",
        "prior_lambda": prior.lam,
        "prior_alpha": prior.alpha,
        "prior_var_const": prior.var_const,
        "prior_d": prior.d,
        "prior_psi": np.asarray(prior.psi, dtype=np.float64),
        "prior_psi_rule": (
            "sample variance (divisor N - 1) of each series over every row of the data file"
            if case.psi_rule == "sample_variance"
            else "1 for every variable"
        ),
        "prior_own_lag_means": np.asarray(prior.means, dtype=np.float64),
        "prior_soc": int(prior.soc is not None),
        "prior_soc_mu": float("nan") if prior.soc is None else prior.soc,
        "prior_sur": int(prior.sur is not None),
        "prior_sur_delta": float("nan") if prior.sur is None else prior.sur,
        "prior_soc_scope": "persistent",
        "prior_dummy_weight": "1 (dummy rows are never scaled by exp(h))",
        "prior_rho": "N(0.9, 0.2^2) 1(|rho| < 1) (TruncatedNormal(0.9, 0.04))",
        "prior_rho_mean": prior.rho_mean,
        "prior_rho_sd": prior.rho_sd,
        "prior_sigma2": "sigma_h^2 ~ IG(5, 0.04) (shape, scale; InvGamma(5.0, 0.04))",
        "prior_sigma2_shape": prior.sigma2_shape,
        "prior_sigma2_scale": prior.sigma2_scale,
        "prior_h0": "h_0 ~ N(0, sigma_h^2/(1 - rho^2)) at the first data row (0-based); E h = 0",
        "rai_bvar_spec": engine_spec(case, prior),
        "sampler": (
            "block 1: joint MN-IW (B, Sigma) | h (Bartlett), weights exp(-h_t) on data rows, 1 on "
            "dummy rows, dbar = d + T + m; block 2: 5 elliptical slice updates of h; block 3: "
            "slice rho (width 0.1, <= 50 steps), centred sigma_h^2 | h, rho ~ IG(5 + T/2, 0.04 + "
            "S/2), non-centred slice of log sigma_h with x = h/sigma_h fixed (width 0.5, <= 50 "
            "steps)"
        ),
        "ess_updates": cx.ESS_UPDATES,
        "rho_slice_width": cx.RHO_WIDTH,
        "rho_slice_max_steps": cx.RHO_MAX_STEPS,
        "s_slice_width": cx.S_WIDTH,
        "s_slice_max_steps": cx.S_MAX_STEPS,
        "initial_state": "(h, rho, sigma_h) from the prior, (B, Sigma) from block 1 given h",
        "predictive": (
            "one path per kept sweep: h_T..h_{T+3} by the AR(1) from h_{T-1}, u ~ N(0, e^h "
            "Sigma), regressors from the last p data rows; one (S, 4, 1 + n) standard normal "
            "array per run from default_rng(predictive_seed)"
        ),
        "quantile_type": "type 7 (numpy linear) of the pooled runs",
        "mcse_method": (
            "*_mcse = max(*_mcse_vehtari: rai_bvar.diagnostics.mcse_quantile_many, Vehtari et al. "
            f"2021 §4.3; *_mcse_batch: sd of the quantiles of {MCSE_BATCHES} consecutive batches "
            "per run / sqrt(runs x batches)); *_ess bulk ESS and *_rhat rank-normalised split "
            "R-hat over the 4 runs (rai_bvar.diagnostics); *_ess_run the bulk ESS of each run"
        ),
        "gate": f"split R-hat <= {RHAT_MAX} and bulk ESS >= {ESS_MIN:.0f} (Q39(b), §9.8)",
        "gate_rhat_max": RHAT_MAX,
        "gate_ess_min": ESS_MIN,
        "mixing_ok": int(not not_met),
        "n_not_met": len(not_met),
        "not_met": "\n".join(not_met) if not_met else "none",
        "gate_table": "\n".join(gate_rows),
        "wall_time_seconds": np.asarray([float(r["wall_time"]) for r in runs]),
        "predictive_seconds": np.asarray([float(r["predictive_seconds"]) for r in runs]),
        "ms_per_sweep": np.asarray(
            [1e3 * float(r["wall_time"]) / (keep + int(r["burn"])) for r in runs]
        ),
    }
    if case.name == "C2":
        _, h_true = simulate_c2()
        data_vars["dgp_h"] = (("t",), h_true)
        for key, value in C2_DGP.items():
            attrs[f"dgp_{key}"] = np.asarray(value, dtype=np.float64) if isinstance(
                value, list) else value  # fmt: skip
        attrs["dgp_description"] = (
            "y_t = c + A1 y_{t-1} + A2 y_{t-2} + exp(h_t/2) L eps_t, Sigma = LL' with unit "
            "variances and correlations 0.3, 0.1, 0.2 for (1, 2), (1, 3), (2, 3); h_t = 0.9 "
            "h_{t-1} + 0.15 eta_t from its stationary law; y started at 0, the first 50 rows "
            "discarded"
        )
    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    target = out / f"{case.fixture}.nc"
    ds.to_netcdf(target, engine="scipy", format="NETCDF3_64BIT")
    print(f"wrote {target}")
    print(f"{case.name}: {len(not_met)} of {len(gate_rows)} compared quantities not met")
    return target


def _labels(prefix: str, dims: tuple[str, ...], coords: dict[str, Any]) -> list[str]:
    values = [list(coords[d][1]) for d in dims]
    grid = np.array(np.meshgrid(*[np.arange(len(v)) for v in values], indexing="ij"))
    flat = grid.reshape(len(dims), -1).T
    return [
        f"{prefix}["
        + ", ".join(f"{d}={values[i][j]}" for i, (d, j) in enumerate(zip(dims, row, strict=True)))
        + "]"
        for row in flat
    ]


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="write tests/parity/fixtures/data/commonsv_n3.csv (C2)")
    sub.add_parser(
        "check", help="compare a fresh C2 regeneration with the committed CSV (rtol = atol = 1e-12)"
    )
    p_run = sub.add_parser("run", help="one run of a case into a staging directory")
    p_run.add_argument("case", choices=sorted(CASES))
    p_run.add_argument("index", type=int, choices=range(CHAINS))
    p_run.add_argument("staging", type=Path)
    p_run.add_argument("--quick", action="store_true")
    p_pack = sub.add_parser("pack", help="summarise a case's staged runs into its fixture")
    p_pack.add_argument("case", choices=sorted(CASES))
    p_pack.add_argument("staging", type=Path)
    p_pack.add_argument("--quick", action="store_true")
    p_all = sub.add_parser("all", help="run every run of the cases in parallel, then pack")
    p_all.add_argument("--cases", nargs="+", default=sorted(CASES), choices=sorted(CASES))
    p_all.add_argument("--staging", type=Path, default=None)
    p_all.add_argument("--quick", action="store_true",
                       help="short runs (smoke test), packed into the staging directory")  # fmt: skip
    args = parser.parse_args(argv)

    if args.command == "data":
        write_data()
    elif args.command == "check":
        return check_data()
    elif args.command == "run":
        args.staging.mkdir(parents=True, exist_ok=True)
        run_one(CASES[args.case], args.index, args.staging, args.quick)
    elif args.command == "pack":
        out = args.staging if args.quick else FIXTURES
        pack(CASES[args.case], args.staging, out, args.quick)
    else:
        staging = args.staging or Path(tempfile.mkdtemp(prefix="rai_bvar_commonsv_oracle_"))
        staging.mkdir(parents=True, exist_ok=True)
        cases = [CASES[name] for name in args.cases]
        start = time.perf_counter()
        run_parallel(cases, staging, args.quick)
        elapsed = time.perf_counter() - start
        (staging / "wall_time.json").write_text(json.dumps({"seconds": elapsed}) + "\n")
        for case in cases:
            pack(case, staging, staging if args.quick else FIXTURES, args.quick)
        print(f"all runs: {elapsed:.0f} s wall time; staging {staging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
