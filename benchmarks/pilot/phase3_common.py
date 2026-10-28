"""Shared helpers of the Phase 3 pilot (P3 DESIGN §8, §10; WP36): data, models, the chain
harness, the gate on prefixes, timing hygiene and the machine fingerprint.

Unlike Phase 2's pilot (``pilot_common.py``: clean-room prototypes), the Phase 3 pilot measures
the **validated production kernels** of WP26–WP35. Nothing here re-implements a kernel:

* :func:`run_chain` resolves the model with ``engines.mcmc.resolve``, builds the runner with
  ``make_runner``, initialises and burns in with the runner's own ``init`` and ``burn``
  executables (as ``run_chains`` does), zeroes the accumulators at the boundary, and then scans
  the production sweep (``make_sweep(program)``, keyed by the absolute sweep index) and records,
  after every sweep, the production ``monitor_values`` plus a few extra quantities that
  production does not monitor (:func:`extra_quantities`). With ``thin = 1`` the recorded monitor
  stream is the monitor stream of ``BVAR.fit`` with the same settings; ``phase3_mixing.py
  --check`` verifies that against a public fit and that a shorter fit is a prefix of a longer
  one (the keys depend only on the sweep index), which is what lets one long run stand for fits
  at every shorter budget.
* :func:`gate_ladder` applies the production gate ``engines.mcmc.monitor.converged`` (P2 §8.2,
  unchanged: every primary quantity with split R-hat < 1.01 and bulk ESS > 400) to the first
  ``D`` kept draws of every chain, for each ``D`` of a ladder.
* :func:`load_snapshot`, :func:`wait_idle` and :func:`fingerprint` record the timing hygiene:
  the load average and the top CPU processes before and after every timed block, a gate that
  waits for the 1-minute load average to fall below a threshold, and the machine fingerprint
  (CPU, cores, OS, Python, NumPy/JAX/jaxlib versions, BLAS).

Scripts in this directory run with ``python -I``; each inserts its own directory into
``sys.path`` before importing this module (as the Phase 2 pilot scripts do).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# --------------------------------------------------------------------------- sizes (P3 §8)

#: FRED-QD medium set (``rai_bvar.datasets``, vintage 2026-09): 204 quarters 1969Q1–2019Q4,
#: so ``T_eff = 200`` at ``p = 4`` (the sample ends before 2020, as the GK evaluation does).
QD7_END, QD7_ROWS = "2019Q4", 204
LAGS = 4
#: BENCH-P2's shape and data (P2 §10.1: ``simulate_var_sv(20, 4, 300, seed=0)``).
BP2_N, BP2_T = 20, 300

#: The gate's budget ladder (kept draws per chain, 4 chains; P3 §8, D-58).
LADDER = (2_000, 5_000, 10_000, 20_000, 50_000)


def _load_module(name: str, path: Path) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules[name]


def phase2_perf() -> Any:
    """``benchmarks/phase2_perf.py`` (BENCH-P2's data, model and ``machine_info``)."""
    return _load_module("wp36_phase2_perf", ROOT / "benchmarks" / "phase2_perf.py")


# --------------------------------------------------------------------------- timing hygiene


def top_processes(k: int = 6) -> list[dict[str, Any]]:
    """The ``k`` processes with the highest CPU share (``ps -Ao pid,pcpu,comm -r``)."""
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            try:
                rows.append({"pid": int(parts[0]), "pcpu": float(parts[1]), "comm": parts[2]})
            except ValueError:
                continue
    rows.sort(key=lambda r: -r["pcpu"])
    for r in rows:
        r["comm"] = r["comm"][-80:]
    return rows[:k]


def load_snapshot() -> dict[str, Any]:
    """Load average, ``uptime`` and the top CPU processes, now."""
    try:
        up = subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        up = ""
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "loadavg": list(os.getloadavg()),
        "uptime": up,
        "top": top_processes(),
    }


def wait_idle(threshold: float = 2.0, timeout: float = 900.0, poll: float = 15.0) -> dict:
    """Wait until the 1-minute load average is below ``threshold`` (at most ``timeout`` s).
    Returns ``{"gate_open": bool, "waited_s": float, "load": snapshot}``."""
    t0 = time.time()
    while True:
        snap = load_snapshot()
        if snap["loadavg"][0] < threshold:
            return {"gate_open": True, "waited_s": time.time() - t0, "load": snap}
        if time.time() - t0 > timeout:
            return {"gate_open": False, "waited_s": time.time() - t0, "load": snap}
        time.sleep(poll)


def contaminated(before: dict, after: dict, threshold: float = 2.0) -> bool:
    """A timed block is contaminated when the 1-minute load average before or after it is above
    ``threshold`` (the brief's rule: about 2)."""
    return max(before["loadavg"][0], after["loadavg"][0]) > threshold


def _blas_info() -> dict[str, Any]:
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return {k: deps.get(k, {}).get("name", "") for k in ("blas", "lapack")}
    except Exception:  # older NumPy
        return {}


def fingerprint() -> dict[str, Any]:
    """CPU model, cores, memory, OS, Python, NumPy/SciPy/JAX/jaxlib versions, BLAS, devices, load."""
    info = dict(phase2_perf().machine_info(with_jax=True))
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = ""
    try:
        info["mem_bytes"] = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        info["efficiency_cores"] = subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.physicalcpu"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()  # fmt: skip
        info["os"] = subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    info["numpy_blas"] = _blas_info()
    info["jax_x64_note"] = "jax_enable_x64 on inside the engine (engines.mcmc._jax.x64)"
    info["load"] = load_snapshot()
    return info


def save_json(out_dir: str | os.PathLike, name: str, payload: dict[str, Any]) -> Path:
    path = Path(out_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- data


def fredqd_medium(vintage: str = "2026-09") -> Any:
    """The FRED-QD medium set (seven federal series, ``tcodes="rai"``) from the verified local
    cache (or a download with checksum, SPEC §5.8): 1969Q1–2019Q4, ``T_eff = 200`` at ``p = 4``."""
    import rai_bvar as rb

    d = rb.datasets.load("fred-qd", vintage, variables="medium")
    w = d.upto(QD7_END).tail(QD7_ROWS)
    if len(w) != QD7_ROWS or np.isnan(w.values).any():
        raise RuntimeError(f"FRED-QD medium window has {len(w)} rows or missing values")
    return w


def sim_data(n: int, T: int, seed: int = 0, lags: int = LAGS) -> Any:
    """``pilot_common.simulate_var_sv(n, lags, T, seed)`` as ``VARData`` (BENCH-P2's DGP)."""
    return phase2_perf().bench_data(n, lags, T, seed)


def size_data(size: str) -> Any:
    if size == "qd7":
        return fredqd_medium()
    if size == "bp2":
        return sim_data(BP2_N, BP2_T, 0)
    if size.startswith("sim"):  # "sim<n>x<T>"
        n, T = size[3:].split("x")
        return sim_data(int(n), int(T), 0)
    raise ValueError(f"unknown size {size!r}")


# --------------------------------------------------------------------------- models

FAMILIES = ("hs", "ng", "r2d2", "dl", "ssvs", "ssvs_beta")
#: Row 325's CommonSV priors (P3 §9.4 test values: ``(ρ+1)/2 ~ Beta(10, 2)``, ``σ_h² ~ 0.1·χ²₁``).
ROW325 = {"rho_a": 10.0, "rho_b": 2.0, "B": 0.1}


def coef_prior(family: str, *, reduced: bool = False, gk: bool = False) -> Any:
    """The coefficient prior of a pilot configuration. ``mn``: ``Minnesota(form="structural",
    hierarchy="fixed")`` (BENCH-P2's); GL families take their defaults (P3 §3.2; ``form="auto"``
    resolves to structural, D-43) except ``ssvs_beta`` (``p ~ Beta(1, 1)``, the GK arm and the
    only SSVS with a primary GL monitor, D-52). ``reduced`` / ``gk``: the GK replication's form
    (``form="reduced"``, ``scale="none"``, ``mean=0.0``; §9.10). ``*_global``: ``groups="global"``."""
    import rai_bvar as rb
    from rai_bvar.priors.sv import Beta

    base, _, groups = family.partition("@")
    kw: dict[str, Any] = {}
    if reduced:
        kw["form"] = "reduced"
    if gk:
        kw.update(scale="none", mean=0.0)
    if groups:
        kw["groups"] = groups
    if base == "mn":
        if reduced:
            return rb.Minnesota(form="reduced", hierarchy="fixed")
        return rb.Minnesota(form="structural", hierarchy="fixed")
    if base == "mn_eb":
        return rb.Minnesota(form="reduced" if reduced else "structural", hierarchy="eb")
    if base == "mn_conj":
        return rb.Minnesota(form="conjugate", hierarchy="fixed")
    cls = {
        "hs": rb.Horseshoe, "ng": rb.NormalGamma, "r2d2": rb.R2D2, "dl": rb.DirichletLaplace,
        "ssvs": rb.SSVS, "ssvs_beta": rb.SSVS,
    }[base]  # fmt: skip
    if base == "ssvs_beta":
        kw["p"] = Beta(1.0, 1.0)
    return cls(**kw)


def build_model(config: str, *, draws: int, burn: int = 1000, seed: int = 0, chains: int = 4,
                const_path: str = "auto") -> Any:  # fmt: skip
    """A pilot model by name.

    * ``"<family>_const"`` / ``"<family>_sv"``: ``Linear(<family prior>) × Constant()`` / ``SV()``
      (``family`` in ``mn`` and :data:`FAMILIES`; ``"<family>@<groups>"`` sets ``groups``);
    * ``"csv_conj"``: ``Minnesota(form="conjugate", hierarchy="fixed") × CommonSV()``, the
      shared-factor path (P3 §5.3); ``"csv_struct"``: the structural Mode I Minnesota ×
      ``CommonSV()``, the per-equation program (§5.2);
    * ``"csv_conj325"`` / ``"csv_struct325"``: the same with row 325's priors (:data:`ROW325`);
    * ``"red_<family>_sv"``: the GK form (reduced, ``scale="none"``, ``mean=0``) × ``SV()``.
    """
    import rai_bvar as rb
    from rai_bvar.priors.sv import Beta, ScaledChi2

    settings = rb.MCMCSettings(
        chains=chains, burn=burn, draws=draws, seed=seed, const_path=const_path
    )
    if config.startswith("csv_"):
        kind = config[4:]
        p325 = kind.endswith("325")
        kind = kind.removesuffix("325")
        prior = coef_prior({"conj": "mn_conj", "struct": "mn"}[kind])
        vol = rb.CommonSV()
        if p325:
            vol = rb.CommonSV(
                rho_prior=Beta(ROW325["rho_a"], ROW325["rho_b"]),
                sigma2_prior=ScaledChi2(ROW325["B"]),
            )
        return rb.BVAR(mean=rb.Linear(prior=prior), vol=vol, lags=LAGS, mcmc=settings)
    reduced = config.startswith("red_")
    family, _, vol_name = config.removeprefix("red_").rpartition("_")
    vol = {"const": rb.Constant(), "sv": rb.SV()}[vol_name]
    prior = coef_prior(family, reduced=reduced, gk=reduced)
    return rb.BVAR(mean=rb.Linear(prior=prior), vol=vol, lags=LAGS, mcmc=settings)


# --------------------------------------------------------------------------- extra quantities


def extra_quantities(static: Any) -> Callable[..., dict[str, Any]]:
    """Per-sweep quantities production does not monitor (P3 §8 findings; E179, R18, R31):

    * GL priors: the reference group functions of WP30 (``validation.reference``;
      ``mean_log_v[g]`` for every family, HS ``log_zeta[g]``/``log_tau[g]``, NG/R2D2
      ``log_xi[g]``, SSVS ``N1[g]``/``log_p[g]``, grid indices), as one ``(G,)`` array per name;
    * CommonSV: ``mean h`` and ``mean |h|`` over the data rows (R31's level, E179's row 325
      functions), ``log σ_h``.
    """
    import jax.numpy as jnp

    from rai_bvar.validation.reference import references_for

    refs = references_for(static)
    G = int(getattr(static.coef, "n_groups", 0) or 0)

    def fn(st: Any, data: Any, prior: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if static.coef.varies:
            arr, gl = prior.eq.coef.gl, prior.shared.gl
            fns = refs.coef.group_functions(st.eq.coef, st.glob.coef, arr, gl, static)
            names: dict[str, list[Any]] = {}
            for key, (val, _) in fns.items():
                name, _, rest = key.partition("[")
                names.setdefault(name, []).append((int(rest.rstrip("]")), val))
            for name, items in names.items():
                items.sort(key=lambda t: t[0])
                out[f"gl.{name}"] = jnp.stack([jnp.asarray(v, jnp.float64) for _, v in items])
            assert all(v.shape == (G,) for v in out.values())
        if static.vol.common:
            m = data.tmask
            h = st.glob.vol.h
            T = jnp.sum(m)
            out["csv.mean_h"] = jnp.sum(m * h) / T
            out["csv.mean_abs_h"] = jnp.sum(m * jnp.abs(h)) / T
            out["csv.log_sigma"] = st.glob.vol.log_sigma
        return out

    return fn


# --------------------------------------------------------------------------- chain harness


@dataclass
class ChainRun:
    """The recorded streams of one run: ``mon`` (production monitor fields, ``(C, D, …)``),
    ``extra`` (``(C, D, …)``), the monitor spec, the resolved model's settings and timings."""

    config: str
    size: str
    seed: int
    draws: int
    burn: int
    mon: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]
    spec: Any
    settings: Any
    static: Any
    acceptance: dict[str, np.ndarray] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


def _status_ok(st: Any) -> None:
    code = np.asarray(st.status.code)
    if np.any(code != 0):
        raise RuntimeError(f"a chain reported a numerical failure: status codes {code.tolist()}")


def run_chain(model: Any, data: Any, *, config: str = "", size: str = "", chunk: int = 1000,
              extras: bool = True, progress: Callable[[str], None] | None = None) -> ChainRun:  # fmt: skip
    """Run ``model`` on ``data`` with the production executables and sweep (module docstring):
    the fit's ``init`` and ``burn`` and then ``model.mcmc.draws`` recorded sweeps (``thin = 1``)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    import rai_bvar as rb
    from rai_bvar.engines.mcmc import _blas, _jax, keys
    from rai_bvar.engines.mcmc.monitor import monitor_spec, monitor_values
    from rai_bvar.engines.mcmc.resolve import resolve
    from rai_bvar.engines.mcmc.runner import make_runner
    from rai_bvar.engines.mcmc.state import zero_accumulators
    from rai_bvar.engines.mcmc.sweep import make_sweep

    settings = model.mcmc
    if settings.thin != 1:
        raise ValueError("the harness records every sweep (thin = 1)")
    C, draws, burn, seed = settings.chains, int(settings.draws), settings.burn, settings.seed
    t_start = time.perf_counter()
    with _jax.x64(), _blas.single_thread():
        res = resolve(model, data, rb.Runtime())
        static = res.static
        fns = make_runner(static)
        program = fns.program
        sweep = make_sweep(program)
        d = jax.tree.map(jnp.asarray, res.data)
        p = jax.tree.map(jnp.asarray, res.prior)
        cache = fns.prepare(d, p)
        state = fns.init(keys.init_keys(seed, res.salt, C), d, p, cache)
        ckeys = keys.chain_keys(keys.fit_root(seed, res.salt), C)
        t0 = time.perf_counter()
        it = 0
        while it < burn:  # the runner's pieces (``run_chains``); keys depend on ``it`` only
            m = min(burn - it, static.chunk)
            state = fns.burn(
                ckeys, state, d, p, cache, np.asarray(it, np.uint32), np.asarray(m, np.int32)
            )
            it += m
        state = jax.block_until_ready(state)
        _status_ok(state)
        t_burn = time.perf_counter() - t0
        state = zero_accumulators(state)
        xfn = extra_quantities(static) if extras else (lambda st, data, prior: {})

        def one_chain(ckey, st, it0, data, prior, cache):
            def body(s, j):
                s = sweep(ckey, it0 + j, s, data, prior, cache)
                return s, (monitor_values(s, data, prior, cache, program), xfn(s, data, prior))

            return lax.scan(body, st, jnp.arange(chunk, dtype=jnp.uint32))

        rec = jax.jit(
            jax.vmap(one_chain, in_axes=(0, 0, None, None, None, None)), donate_argnums=(1,)
        )
        mons: list[Any] = []
        extra: list[Any] = []
        t0 = time.perf_counter()
        n_chunks = -(-draws // chunk)
        for c in range(n_chunks):
            it0 = np.asarray(burn + c * chunk, np.uint32)
            state, (mon, ex) = rec(ckeys, state, it0, d, p, cache)
            mon, ex = jax.device_get((mon, ex))
            mons.append(mon)
            extra.append(ex)
            if progress is not None and (c + 1) % 10 == 0:
                progress(f"{config} {size} seed {seed}: {(c + 1) * chunk} of {draws}")
        state = jax.block_until_ready(state)
        _status_ok(state)
        t_sample = time.perf_counter() - t0
        host = jax.device_get(state)
    mon_all = {k: np.concatenate([m[k] for m in mons], axis=1)[:, :draws] for k in mons[0]}
    ex_all = {k: np.concatenate([e[k] for e in extra], axis=1)[:, :draws] for k in extra[0]}
    acc = _acceptance(host, n_chunks * chunk)
    spec = monitor_spec(static, tuple(data.variables))
    return ChainRun(
        config=config, size=size, seed=seed, draws=draws, burn=burn, mon=mon_all, extra=ex_all,
        spec=spec, settings=settings, static=static, acceptance=acc,
        timing={"burn_s": t_burn, "sample_s": t_sample, "total_s": time.perf_counter() - t_start,
                "ms_per_sweep_scan": 1000.0 * t_sample / (n_chunks * chunk)},
    )  # fmt: skip


def _acceptance(host: Any, sweeps: int) -> dict[str, np.ndarray]:
    """Every ``acc_*`` leaf of the final state divided by the sweeps since the boundary."""
    import jax

    out: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(host)[0]:
        name = getattr(path[-1], "name", None)
        if isinstance(name, str) and name.startswith("acc_"):
            where = ".".join(str(getattr(q, "name", getattr(q, "idx", q))) for q in path)
            out[where] = np.asarray(leaf, dtype=np.float64) / sweeps
    return out


# --------------------------------------------------------------------------- gate and ESS


def gate(mon: dict[str, np.ndarray], spec: Any, settings: Any, D: int) -> dict[str, Any]:
    """The production gate (``monitor.converged``, P2 §8.2 unchanged) on the first ``D`` kept
    draws of every chain, with the primary R-hat / bulk ESS behind it."""
    from rai_bvar.engines.mcmc import monitor

    sub = {k: v[:, :D] for k, v in mon.items()}
    ok, msgs = monitor.converged(sub, spec, settings, error=False, eb_converged=None)
    rhat, ess = monitor.rhat_ess(sub, spec, spec.primary)
    worst = int(np.argmax(np.maximum((rhat - 1.0) / 0.01, 400.0 / ess)))
    return {
        "D": D,
        "pass": bool(ok),
        "n_primary": len(spec.primary),
        "n_fail": int(np.sum((rhat >= settings.rhat_max) | (ess <= settings.ess_min))),
        "rhat_max": float(np.max(rhat)),
        "ess_min": float(np.min(ess)),
        "worst": spec.primary[worst],
        "messages": [m.splitlines()[0][:300] for m in msgs],
    }


def gate_ladder(run: ChainRun, ladder: Sequence[int] = LADDER) -> list[dict[str, Any]]:
    return [gate(run.mon, run.spec, run.settings, D) for D in ladder if D <= run.draws]


def ess_table(run: ChainRun, D: int | None = None) -> dict[str, dict[str, float]]:
    """Bulk ESS, split R-hat and ESS per 1,000 sweeps (bulk ESS of the 4 chains ÷ (C·D/1000),
    i.e. per 1,000 sweeps of one chain) of every monitored quantity (primary and secondary) and
    every extra quantity, on the first ``D`` draws; plus the lag-1 autocorrelation (pooled over
    chains) of each."""
    from rai_bvar.diagnostics import rhat_ess_bulk_many

    D = run.draws if D is None else D
    C = next(iter(run.mon.values())).shape[0]
    labels = list(run.spec.primary + run.spec.secondary)
    x = np.asarray(run.spec.stack({k: v[:, :D] for k, v in run.mon.items()}, labels))  # (C, D, Q)
    cols: list[np.ndarray] = []
    names: list[str] = []
    prim = set(run.spec.primary)
    for k, v in run.extra.items():
        v = np.asarray(v, dtype=np.float64)[:, :D]
        if v.ndim == 2:
            cols.append(v[..., None])
            names.append(k)
        else:
            cols.append(v.reshape(C, D, -1))
            names.extend(f"{k}[{g}]" for g in range(cols[-1].shape[2]))
    if x.shape[:2] != (C, D):
        raise AssertionError(f"unexpected monitor stack shape {x.shape}")
    allx = np.concatenate([x, *cols], axis=2) if cols else x
    allnames = labels + names
    rh, es = rhat_ess_bulk_many(allx)
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(allnames):
        z = allx[:, :, j]
        zc = z - z.mean(axis=1, keepdims=True)
        den = np.sum(zc * zc)
        acf1 = float(np.sum(zc[:, 1:] * zc[:, :-1]) / den) if den > 0 else float("nan")
        out[name] = {
            "primary": name in prim,
            "extra": j >= len(labels),
            "ess_bulk": float(es[j]),
            "rhat": float(rh[j]),
            "ess_per_1k": float(es[j]) / (C * D / 1000.0),
            "acf1": acf1,
            "mean": float(np.mean(z)),
            "sd": float(np.std(z)),
        }
    return out
