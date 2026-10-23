"""stochvol parity of the SV block at block level (DESIGN §9.9 Tier P, §6.3, D-7; work package WP21).

**Oracle.** R ``stochvol`` 3.2.1 ``svsample`` runs committed by WP8 as
``fixtures/stochvol_<dataset>_<run>.nc`` (4 chains × 50,000 draws; layout and provenance in
``test_fixture_schema.py``; conventions and findings in ``SV_PARAMETRIZATION.md``). Datasets
(i)–(iii) are simulated Gaussian SV series, (iv) the standardised AR(4) residuals of GDP growth,
(v) the paths of (i) with standardized Student-t errors, ``ν = 5``.

**Ours.** The production runner (``fit_mcmc``) on the composition ``ZeroMean × SV`` with ``n = 1``
(``g ≡ 0`` and no contemporaneous slots, so the volatility block sees ``e = y`` exactly as
``svsample`` does). The volatility part of the ``StaticConfig`` comes from ``api.static_parts`` and
the prior arrays from ``priors.sv_arrays``: the production mapping of the ``SV`` spec.

**One parameter set, both sides derived** (DESIGN §6.3, §9.9, E11). Each run is one stochvol
parametrisation ``μ ~ N(b_μ, B_μ)``, ``(φ+1)/2 ~ Beta(a_φ, b_φ)``, ``σ² ~ B_σ·χ²₁``, stationary
``h_0``, offset ``c = 0``; the rai-bvar ``SV`` spec and the R arguments are both derived from it::

    priormu = c(b_μ, √B_μ)   priorphi = c(a_φ, b_φ)   priorsigma = B_σ       (svsample)
    sv_normal(b_μ, √B_μ)     sv_beta(a_φ, b_φ)        sv_gamma(0.5, 0.5/B_σ) (specify_priors)

and the fixture's recorded call, prior summary and spec are checked against that derivation.

**Comparison.** The 5/25/50/75/95% quantiles of ``μ, φ, σ``, of ``h_t`` at ``t ∈ {0, 1, T/4,
T/2, T}`` (``t = 0`` is ``h_0``; ``h_t`` belongs to ``y_t``) and, for SV-t, of ``τ_t`` at the three
largest ``|y_t|``::

    z = (q̂_ours − q̂_R)/√(mcse_R² + mcse_ours²),    |z| ≤ max(4, z_{1 − α/(2m)}),  α = 1%

with ``mcse = max(Vehtari et al. 2021 §4.3 quantile MCSE, batch-quantile MCSE with 20 batches per
chain)`` on both sides (the fixtures' method) and type-7 quantiles of the pooled chains. The bound
is 4 quantile-MCSE (DESIGN §9.9) for every family here (``m ≤ 157`` comparisons); the Bonferroni
term only matters for the larger families of ``test_bayesianvars.py``.

**Exclusion** (D-7): stochvol's non-centred step takes ``|σ|`` without flipping ``h̃``. A fixture
on which our fold fraction (sweeps with an accepted non-centred ``σ' < 0``) exceeds 0.5% is
excluded: it is still compared, and a failure there is reported as a skip, not an error.

**SV-t** (``stochvol_v_t5``). stochvol 3.2.1's Student-t sampler does not target the
standardized-t SV posterior that its own simulator ``svsim`` and the fixture's DGP use: on a
four-observation problem whose posterior is computed exactly by importance sampling, stochvol's
``τ`` and ``h`` quantiles miss the exact ones (with and without ASIS), while ours match them
(``SV_PARAMETRIZATION.md`` §3.2). SV-t is therefore checked against that exact importance-sampling
oracle (:func:`test_sv_t_matches_the_exact_importance_sampling_posterior`), and the fixture row is
a strict ``xfail`` narrowed to :class:`OracleDeviation`: our ``T = 300`` run must still mix (R-hat),
give finite ``z`` and recover the DGP (true ``μ, φ, σ`` inside our central 99.8% interval, 90%
bands covering ≥ 70% of the true ``h_t`` and ``log τ_t``) before the stochvol mismatch is accepted.

**Tiers.** PR (exit criterion): the ``main`` and ``defaults`` runs of (i)–(iv) at 4 × 6,000 kept
draws, the SV-t exact oracle and the SV-t row (``xfail`` as above). Nightly (``slow``,
``parity_long``): every run, including SV-t and the optional ``exact`` rows (our ``exact=True``
against stochvol's importance-weighted correction), at 4 × 40,000 kept draws.
"""

from __future__ import annotations

import ast
import math
import zlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rai_bvar.api import BVAR, static_parts
from rai_bvar.components.specs import SV, Linear
from rai_bvar.diagnostics import mcse_quantile_many, rhat_many
from rai_bvar.priors.minnesota import Minnesota
from rai_bvar.priors.sv import Beta, DiscreteUniform, InvGamma, Normal, ScaledChi2

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

FIXTURES = Path(__file__).parent / "fixtures"
PROBS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
MCSE_BATCHES = 20  # batches per chain of the batch-quantile MCSE (the fixtures' method)
Z_BOUND = 4.0  # DESIGN §9.9: within 4 quantile-MCSE
FAMILY_ALPHA = 0.01
FOLD_MAX = 0.005  # D-7 exclusion threshold (DESIGN §9.9)
RHAT_MAX = 1.05  # gross non-mixing guard on our compared parameters (MCSE covers the rest)
TRUTH_TAIL = 0.001  # SV-t DGP guard: the true μ, φ, σ inside our central 99.8% interval
TRUTH_COVER_MIN = 0.7  # SV-t DGP guard: 90% bands cover ≥ 70% of the true h_t and log τ_t

PR_DRAWS, PR_BURN = 6_000, 1_000
LONG_DRAWS, LONG_BURN = 40_000, 2_000
SEED = 20261022
FIXTURE_T_PAD = 320  # every stochvol dataset (T = 300 and 235) in one compiled program


# --------------------------------------------------------------------------- the parameter sets


@dataclass(frozen=True)
class ParameterSet:
    """One stochvol parametrisation (DESIGN §6.3): ``μ ~ N(b_mu, B_mu)``, ``(φ+1)/2 ~ Beta(a_phi,
    b_phi)``, ``σ² ~ B_sigma·χ²₁``, stationary ``h_0``, offset 0; ``nu`` fixes Student-t tails."""

    b_mu: float
    B_mu: float
    a_phi: float
    b_phi: float
    B_sigma: float
    nu: int | None = None
    exact: bool = False

    def sv(self) -> SV:
        """The rai-bvar spec: ``SV(mu_prior=Normal(b_μ, B_μ), phi_prior=Beta(a_φ, b_φ),
        sigma2_prior=ScaledChi2(B_σ), offset=0.0, exact=…)``, plus ``tails="t"`` and
        ``nu_prior=DiscreteUniform(ν, ν)`` for a fixed ν."""
        tails: dict[str, Any] = {}
        if self.nu is not None:
            tails = {"tails": "t", "nu_prior": DiscreteUniform(self.nu, self.nu)}
        return SV(
            mu_prior=Normal(self.b_mu, self.B_mu),
            phi_prior=Beta(self.a_phi, self.b_phi),
            sigma2_prior=ScaledChi2(self.B_sigma),
            h0="stationary",
            offset=0.0,
            exact=self.exact,
            **tails,
        )

    def svsample_args(self) -> str:
        """``svsample`` prior arguments: ``priormu = c(b_μ, √B_μ)`` (mean, s.d.), ``priorphi =
        c(a_φ, b_φ)``, ``priorsigma = B_σ`` (not its square root)."""
        return (
            f"priormu = c({_r(self.b_mu)}, {_r(math.sqrt(self.B_mu))}), "
            f"priorphi = c({_r(self.a_phi)}, {_r(self.b_phi)}), priorsigma = {_r(self.B_sigma)}, "
            'priorlatent0 = "stationary"'
        )

    def specify_priors(self) -> str:
        """``specify_priors`` form: ``sv_normal(b_μ, √B_μ)``, ``sv_beta(a_φ, b_φ)``,
        ``sv_gamma(0.5, 0.5/B_σ)`` (shape ½, rate ½/B_σ ⇔ ``B_σ·χ²₁``), ``sv_constant(ν)``."""
        nu = "" if self.nu is None else f", nu = sv_constant({self.nu})"
        return (
            f"specify_priors(mu = sv_normal({_r(self.b_mu)}, {_r(math.sqrt(self.B_mu))}), "
            f"phi = sv_beta({_r(self.a_phi)}, {_r(self.b_phi)}), "
            f"sigma2 = sv_gamma(0.5, 0.5/{_r(self.B_sigma)}){nu})"
        )


def _r(x: float) -> str:
    """A number as R prints it in the recorded calls (``10``, ``1.5``, ``0.1``)."""
    return f"{float(x):g}"


MAIN = ParameterSet(b_mu=0.0, B_mu=100.0, a_phi=20.0, b_phi=1.5, B_sigma=0.1)
# svsample's own defaults (stochvol manual: priormu = c(0, 100), priorphi = c(5, 1.5),
# priorsigma = 1, priorlatent0 = "stationary"); the "defaults" fixtures pass no prior argument.
STOCHVOL_DEFAULTS = ParameterSet(b_mu=0.0, B_mu=1e4, a_phi=5.0, b_phi=1.5, B_sigma=1.0)
RUNS: dict[str, ParameterSet] = {
    "main": MAIN,
    "defaults": STOCHVOL_DEFAULTS,
    "t5": ParameterSet(**{**MAIN.__dict__, "nu": 5}),
    "exact": ParameterSet(**{**MAIN.__dict__, "exact": True}),
}
FIXTURE_RUNS = {  # fixture: (dataset, run)
    "stochvol_i_main": ("i", "main"),
    "stochvol_i_defaults": ("i", "defaults"),
    "stochvol_ii_main": ("ii", "main"),
    "stochvol_ii_defaults": ("ii", "defaults"),
    "stochvol_iii_main": ("iii", "main"),
    "stochvol_iii_defaults": ("iii", "defaults"),
    "stochvol_iv_main": ("iv", "main"),
    "stochvol_iv_defaults": ("iv", "defaults"),
    "stochvol_v_t5": ("v", "t5"),
    "stochvol_i_exact": ("i", "exact"),
    "stochvol_iv_exact": ("iv", "exact"),
}
T_DEVIATION = (
    "stochvol 3.2.1's Student-t sampler misses the exact standardized-t SV posterior "
    "(SV_PARAMETRIZATION.md §3.2); replaced by the exact importance-sampling oracle"
)


def fixture_params(names: list[str]) -> list[Any]:
    """Test parameters; the SV-t fixture as a strict ``xfail`` that accepts only an
    :class:`OracleDeviation` (any other exception, and an unexpected pass, fail the test)."""
    xfail = pytest.mark.xfail(strict=True, raises=OracleDeviation, reason=T_DEVIATION)
    return [pytest.param(n, marks=xfail) if FIXTURE_RUNS[n][1] == "t5" else n for n in names]


PR_FIXTURES = [n for n, (_, run) in FIXTURE_RUNS.items() if run in ("main", "defaults", "t5")]


# --------------------------------------------------------------------------- fixture access


@cache
def fixture(name: str):
    import xarray as xr

    path = FIXTURES / f"{name}.nc"
    if not path.exists():
        pytest.skip(f"stochvol fixture {path.name} is missing (tools/make_r_fixtures.R)")
    return xr.load_dataset(path, engine="scipy")


_SPEC_NAMES = {
    cls.__name__: cls for cls in (SV, Normal, Beta, ScaledChi2, InvGamma, DiscreteUniform)
}


def parse_spec(text: str) -> Any:
    """The recorded ``rai_bvar_spec`` string as an object: a literal evaluator that accepts only
    calls of the SV spec and prior classes with literal arguments (never ``eval``)."""

    def value(node: ast.AST) -> Any:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            cls = _SPEC_NAMES.get(node.func.id)
            if cls is None:
                raise ValueError(f"unexpected constructor {node.func.id!r} in {text!r}")
            kwargs = {kw.arg: value(kw.value) for kw in node.keywords if kw.arg is not None}
            return cls(*(value(a) for a in node.args), **kwargs)
        return ast.literal_eval(node)

    return value(ast.parse(text, mode="eval").body)


# --------------------------------------------------------------------------- comparison helpers


def batch_quantile_mcse(x: np.ndarray, batches: int = MCSE_BATCHES) -> np.ndarray:
    """``sd(q̂_b)/√(C·B)`` over the ``B`` consecutive batches of every chain (``x``: ``(C, D, *S)``;
    returns ``(*S, 5)``), the fixtures' batch-quantile MCSE."""
    parts = np.array_split(np.arange(x.shape[1]), batches)
    bq = np.stack([np.quantile(x[:, idx], PROBS, axis=1) for idx in parts])  # (B, 5, C, *S)
    bq = np.moveaxis(bq, 2, 1).reshape(-1, len(PROBS), *x.shape[2:])  # (B·C, 5, *S)
    return np.moveaxis(bq.std(axis=0, ddof=1) / math.sqrt(bq.shape[0]), 0, -1)


def summarise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Type-7 quantiles of the pooled chains and their MCSE (max of Vehtari and batch), each
    ``(*S, 5)``, of draws ``x`` with shape ``(C, D, *S)``."""
    q = np.moveaxis(np.quantile(x.reshape(-1, *x.shape[2:]), PROBS, axis=0), 0, -1)
    vehtari = np.stack([mcse_quantile_many(x, p) for p in PROBS], axis=-1)
    return q, np.maximum(vehtari, batch_quantile_mcse(x))


def z_bound(m: int, alpha: float = FAMILY_ALPHA) -> float:
    """``max(4, z_{1 − α/(2m)})``: 4 quantile-MCSE, widened only when a family of ``m``
    comparisons would make a 4-MCSE excursion likely under the null (Bonferroni at ``α``)."""
    return max(Z_BOUND, NormalDist().inv_cdf(1.0 - alpha / (2.0 * m)))


@dataclass
class Comparison:
    """``z = (q̂_ours − q̂_ref)/√(mcse_ours² + mcse_ref²)`` of one group, with labels."""

    group: str
    labels: list[str]
    z: np.ndarray  # (len(labels), 5)

    @classmethod
    def of(cls, group: str, labels: list[str], ours: np.ndarray, q_ref, mcse_ref) -> Comparison:
        q, m = summarise(ours)
        z = (q - np.asarray(q_ref)) / np.sqrt(m**2 + np.asarray(mcse_ref) ** 2)
        assert z.shape == np.shape(q_ref) and np.all(np.isfinite(z)), group
        return cls(group, labels, z.reshape(len(labels), len(PROBS)))


def parity_failures(comparisons: list[Comparison], context: str) -> tuple[float, str | None]:
    """``(max |z|, message)``; ``message`` is ``None`` when every ``|z| ≤ z_bound(m)`` and lists
    the worst comparisons otherwise."""
    m = sum(c.z.size for c in comparisons)
    bound = z_bound(m)
    rows = sorted(
        (
            (abs(z), f"{c.group}[{label}] q{round(100 * p)}: z = {z:+.2f}")
            for c in comparisons
            for label, zs in zip(c.labels, c.z, strict=True)
            for p, z in zip(PROBS, zs, strict=True)
        ),
        key=lambda r: -r[0],
    )
    worst = rows[0][0]
    if worst <= bound:
        return worst, None
    bad = sum(r[0] > bound for r in rows)
    return worst, (
        f"{context}: {bad} of {m} quantiles differ by more than {bound:.2f} MCSE; worst: "
        + "; ".join(r[1] for r in rows[:6])
    )


def assert_parity(comparisons: list[Comparison], context: str) -> float:
    """Every ``|z| ≤ z_bound(m)``; returns max ``|z|``."""
    worst, message = parity_failures(comparisons, context)
    assert message is None, message
    return worst


# --------------------------------------------------------------------------- the block-level run


@dataclass
class SVRun:
    fit: Any
    fold: float  # fraction of sweeps with a non-centred fold, mean over chains


def run_block(
    y: np.ndarray, pset: ParameterSet, draws: int, burn: int, salt: int, T_pad: int | None = None
) -> SVRun:
    """``fit_mcmc`` with ``ZeroMean × SV`` (``n = 1``) on ``y`` (DESIGN §9.9): 4 chains, every
    kept draw's ``h`` path (and ``log τ`` for t) stored. ``T_pad`` (default ``bucket(T, 32)``)
    lets the datasets share one compiled program (padding is exact, D-15)."""
    import jax

    from rai_bvar.engines.mcmc import data as data_mod
    from rai_bvar.engines.mcmc.config import (
        CoefPriorStatic,
        MeanStatic,
        Shapes,
        StaticConfig,
        StoreStatic,
        StructStatic,
        bucket,
    )
    from rai_bvar.engines.mcmc.results import FitMeta, fit_mcmc
    from rai_bvar.engines.mcmc.state import (
        CoefPriorArrays,
        ContempPriorArrays,
        EqPrior,
        PriorArrays,
        SharedPrior,
    )
    from rai_bvar.priors.scales import WindowScales
    from rai_bvar.priors.sv_arrays import sv_prior_arrays, transitory_arrays
    from rai_bvar.settings import MCMCSettings, Runtime

    sv = pset.sv()
    T = y.size
    # production mapping SV spec → static part and prior arrays (DESIGN §6.3)
    vstatic = static_parts(BVAR(mean=Linear(prior=Minnesota(form="reduced")), vol=sv)).vol
    assert vstatic.zero_offset and not vstatic.rw and vstatic.h0 == "stationary"
    assert vstatic.exact == pset.exact and vstatic.sigma2_prior == "scaled_chi2"
    assert vstatic.transitory == ("none" if pset.nu is None else "t")
    lags, n_det = 1, 1  # ZeroMean ignores the regressors; Shapes needs k ≥ 1
    T_pad = bucket(T, 32) if T_pad is None else T_pad
    assert T_pad >= T
    shapes = Shapes.of(1, lags, n_det, T_pad)
    chunk = min(draws, 2_000)
    static = StaticConfig(
        shapes=shapes,
        struct=StructStatic(),
        mean=MeanStatic(kind="zero"),
        coef=CoefPriorStatic(),
        vol=vstatic,
        store=StoreStatic(paths=True),
        chunk=chunk,
    )
    scales = WindowScales(  # ψ is not read: explicit μ prior, absolute offset 0
        psi=np.array([np.mean(y * y)]), own_means=np.zeros(1), freq="Q-DEC"
    )
    core = sv_prior_arrays(sv, scales)
    expected = {"b_mu": pset.b_mu, "B_mu": pset.B_mu, "a_phi": pset.a_phi, "b_phi": pset.b_phi,
                "B_sigma": pset.B_sigma, "log_c": -np.inf}  # fmt: skip
    for field_name, value in expected.items():
        np.testing.assert_array_equal(getattr(core, field_name), [value], err_msg=field_name)
    k = shapes.k
    prior = PriorArrays(
        eq=EqPrior(
            coef=CoefPriorArrays(  # inert: ZeroMean has no coefficients
                mean0=np.zeros((1, k)), prec0=np.ones((1, k)), group=np.zeros((1, k), np.int32)
            ),
            contemp=ContempPriorArrays(
                mean0=np.zeros((1, 0)), prec0=np.ones((1, 0)), cmask=np.zeros((1, 0))
            ),
            vol_core=core,
            transitory=transitory_arrays(sv, scales, vstatic.nu_grid),
        ),
        shared=SharedPrior(Mc=np.zeros((k, 0))),
    )
    y_pad = np.zeros((shapes.T_pad, 1))
    y_pad[:T, 0] = y
    data = jax.tree.map(
        np.asarray,
        data_mod.build_engine_data(
            y_pad, np.zeros((lags, 1)), np.int32(T), n_det=n_det, t0=float(lags + 1)
        ),
    )
    settings = MCMCSettings(
        chains=4, draws=draws, burn=burn, seed=SEED, store="paths", path_thin=1, store_max=draws
    )
    fit = fit_mcmc(
        static, data, prior, settings, Runtime(chunk=chunk), seed=SEED, salt=salt,
        meta=FitMeta(variables=("y",)),
    )  # fmt: skip
    return SVRun(fit=fit, fold=float(np.mean(fit.acceptance["acc_fold"])))


def our_draws(fit: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(params (C, D, 3) = μ, φ, σ;  h (C, D, T_pad + 1) with h[..., 0] = h_0;  log τ (C, D,
    T_pad) or None)`` of a ``ZeroMean × SV`` fit."""
    core = fit.param_draws.vol.core
    params = np.stack([np.asarray(getattr(core, p))[..., 0] for p in ("mu", "phi", "sigma")], -1)
    h = np.asarray(fit.path_draws.h)[:, :, 0, :]
    log_tau = None
    if fit.static.vol.has_t:  # PathDraw.log_scale2 = 2 log o + log τ = log τ without outliers
        log_tau = np.asarray(fit.path_draws.log_scale2)[:, :, 0, :]
    return params, h, log_tau


def fixture_comparisons(name: str, fit: Any) -> list[Comparison]:
    """The DESIGN §9.9 comparisons of one fixture."""
    ds = fixture(name)
    params, h, log_tau = our_draws(fit)
    t_sel = ds["t_sel"].values
    comparisons = [
        Comparison.of("param", list(ds["param"].values), params, ds["param_q"], ds["param_mcse"]),
        Comparison.of("h", [f"t={t}" for t in t_sel], h[..., t_sel], ds["h_q"], ds["h_mcse"]),
    ]
    if "tau_q" in ds:  # SV-t: τ_t at the three largest |y_t| (obs t ↔ data row t − 1)
        obs = ds["tau_t"].values
        comparisons.append(
            Comparison.of("tau", [f"t={t}" for t in obs], np.exp(log_tau[..., obs - 1]),
                          ds["tau_q"], ds["tau_mcse"])
        )  # fmt: skip
    return comparisons


def fixture_y(name: str) -> np.ndarray:
    ds = fixture(name)
    frame = pd.read_csv(FIXTURES / ds.attrs["data_file"], float_precision="round_trip")
    y = frame["y"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(y, ds["y"].values)  # the oracle ran on exactly these numbers
    return y


class OracleDeviation(Exception):
    """The SV-t parity failure caused by the faulty stochvol oracle (``SV_PARAMETRIZATION.md``
    §3.2). Raised only after our own run passed every sanity check, and the only exception the
    SV-t ``xfail`` accepts, so a crash, a NaN or non-mixing chains still fail the test."""


def check_recovers_dgp(name: str, fit: Any) -> None:
    """Gross-error guard where the oracle cannot judge us (SV-t): the DGP's ``μ, φ, σ`` (fixture
    attributes) lie inside our central ``1 − 2·TRUTH_TAIL`` posterior interval, and our pointwise
    90% bands cover at least ``TRUTH_COVER_MIN`` of the true ``h_t`` and ``log τ_t`` (the data
    file's ``h`` and ``tau`` columns; nominal coverage 90%, misses dependent along the path)."""
    ds = fixture(name)
    truth = pd.read_csv(FIXTURES / ds.attrs["data_file"], float_precision="round_trip")
    params, h, log_tau = our_draws(fit)
    T = len(truth)
    flat = params.reshape(-1, params.shape[-1])
    for i, p in enumerate(("mu", "phi", "sigma")):
        rank = float(np.mean(flat[:, i] <= ds.attrs[f"dgp_{p}"]))
        assert TRUTH_TAIL < rank < 1.0 - TRUTH_TAIL, f"{name}: DGP {p} at posterior rank {rank}"
    paths = {"h": h[..., 1 : T + 1], "tau": log_tau[..., :T]}  # obs t ↔ data row t − 1
    for col, draws in paths.items():
        lo, hi = np.quantile(draws.reshape(-1, T), [0.05, 0.95], axis=0)
        true = truth[col].to_numpy() if col == "h" else np.log(truth[col].to_numpy())
        cover = float(np.mean((lo <= true) & (true <= hi)))
        assert cover >= TRUTH_COVER_MIN, f"{name}: 90% bands cover {cover:.2f} of the true {col}"


def check_fixture(name: str, draws: int, burn: int) -> None:
    """Run, check our chains (R-hat; finite ``z``), compare, and apply the D-7 exclusion (a
    failure on a fixture whose fold fraction exceeds 0.5% is a skip). SV-t is not D-7 (its
    deviation persists without the non-centred step): its run must also recover the DGP
    (:func:`check_recovers_dgp`), and only then is a parity failure an :class:`OracleDeviation`."""
    pset = RUNS[FIXTURE_RUNS[name][1]]
    y = fixture_y(name)
    run = run_block(y, pset, draws, burn, salt=zlib.crc32(name.encode()), T_pad=FIXTURE_T_PAD)
    params = our_draws(run.fit)[0]
    rhat = rhat_many(params)
    assert np.all(rhat < RHAT_MAX), f"{name}: our chains do not mix (R-hat of μ, φ, σ: {rhat})"
    _, message = parity_failures(fixture_comparisons(name, run.fit), name)
    if FIXTURE_RUNS[name][1] == "t5":
        check_recovers_dgp(name, run.fit)
        if message is not None:
            raise OracleDeviation(f"{message} (fold fraction {100 * run.fold:.3f}%)")
        return
    if message is not None and run.fold > FOLD_MAX:
        pytest.skip(
            f"excluded (DESIGN §9.9, D-7): our fold fraction {100 * run.fold:.2f}% exceeds "
            f"{100 * FOLD_MAX:.1f}%; {message}"
        )
    assert message is None, f"{message} (fold fraction {100 * run.fold:.3f}%)"


# --------------------------------------------------------------------------- exact SV-t oracle


TINY_Y = np.array([0.4, -2.5, 0.3, 0.6])  # one outlier: its τ is far from its prior
TINY = ParameterSet(b_mu=-1.0, B_mu=0.25, a_phi=20.0, b_phi=1.5, B_sigma=0.1, nu=5)
TINY_TAU_OBS = 2  # 1-based observation whose τ is compared


def weighted_quantiles(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Self-normalised IS quantiles of ``x`` (``w`` sums to 1) and their MCSE: the linearised
    ``s_α = √(Σ w_i²(1{x_i ≤ q_α} − α)²)`` mapped through the weighted quantile function,
    ``(q(α + s_α) − q(α − s_α))/2``."""
    order = np.argsort(x)
    xs, cw = x[order], np.cumsum(w[order])

    def quantile(p):
        return np.interp(p, cw, xs)

    q = quantile(PROBS)
    s = np.array(
        [math.sqrt(np.sum((w * ((x <= qa) - a)) ** 2)) for qa, a in zip(q, PROBS, strict=True)]
    )
    return q, 0.5 * (quantile(PROBS + s) - quantile(PROBS - s))


def is_oracle(
    y: np.ndarray, pset: ParameterSet, tau_obs: int, size: int, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Exact posterior quantiles of the standardized-t SV model by importance sampling from the
    prior, with ``τ`` integrated out (D-8; N03 §4.3a)::

        y_t | h_t ~ t_ν(0, s_t²),  s_t² = e^{h_t}(ν − 2)/ν     (Var = e^{h_t})
        log w = Σ_t [log t_ν(y_t/s_t) − log s_t]

    and ``τ`` of the 1-based observation ``tau_obs`` drawn per sample from its exact conditional
    ``IG((ν+1)/2, (ν − 2 + y² e^{−h})/2)``. Returns ``{name: (q (…, 5), mcse (…, 5))}`` for
    ``param`` (μ, φ, σ), ``h`` (``h_0..h_T``) and ``tau``."""
    rng = np.random.default_rng(seed)
    nu, T = pset.nu, y.size
    mu = pset.b_mu + math.sqrt(pset.B_mu) * rng.standard_normal(size)
    phi = 2.0 * rng.beta(pset.a_phi, pset.b_phi, size) - 1.0
    sigma = math.sqrt(pset.B_sigma) * np.abs(rng.standard_normal(size))
    h = np.empty((T + 1, size))
    h[0] = mu + sigma / np.sqrt(1.0 - phi**2) * rng.standard_normal(size)
    for t in range(1, T + 1):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma * rng.standard_normal(size)
    s = np.sqrt(np.exp(h[1:]) * (nu - 2.0) / nu)
    logw = np.sum(stats.t.logpdf(y[:, None] / s, df=nu) - np.log(s), axis=0)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    assert 1.0 / np.sum(w**2) > 0.2 * size  # the prior is a good proposal for T = 4
    scale = (nu - 2.0 + y[tau_obs - 1] ** 2 * np.exp(-h[tau_obs])) / 2.0
    tau = scale / rng.gamma((nu + 1.0) / 2.0, size=size)
    return {
        name: tuple(np.stack(v) for v in zip(*(weighted_quantiles(x, w) for x in xs), strict=True))
        for name, xs in (("param", (mu, phi, sigma)), ("h", tuple(h)), ("tau", (tau,)))
    }


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("name", list(FIXTURE_RUNS))
def test_fixture_and_spec_derive_from_one_parameter_set(name):
    """E11: the fixture's ``rai_bvar_spec`` is our ``SV`` spec and its R call carries the stochvol
    arguments derived from the same parameter set through the DESIGN §6.3 formulas."""
    dataset, run = FIXTURE_RUNS[name]
    a = fixture(name).attrs
    pset = RUNS[run]
    assert (a["dataset"], a["run"]) == (dataset, run)
    assert parse_spec(a["rai_bvar_spec"]) == pset.sv()
    if run in ("main", "exact"):
        call = a["equivalent_svsample_call"] if run == "exact" else a["call"]
        assert pset.svsample_args() in call, (pset.svsample_args(), call)
    if run == "exact":
        assert "correct_model_misspecification = TRUE" in a["equivalent_svsample_call"]
        assert int(a["importance_weighted"]) == 1
    if run == "defaults":
        assert "prior" not in a["call"]  # stochvol's defaults: the STOCHVOL_DEFAULTS set
    if run == "t5":
        assert pset.specify_priors() in a["call"], (pset.specify_priors(), a["call"])
        assert "nu        ~ Constant(value = 5)" in a["priors"]
    # stochvol's own print-out of the prior it used
    assert f"Normal(mean = {_r(pset.b_mu)}, sd = {_r(math.sqrt(pset.B_mu))})" in a["priors"]
    assert f"Beta(a = {_r(pset.a_phi)}, b = {_r(pset.b_phi)})" in a["priors"]
    assert f"Gamma(shape = 0.5, rate = {_r(0.5 / pset.B_sigma)})" in a["priors"]


def test_parse_spec_rejects_anything_but_spec_constructors():
    with pytest.raises(ValueError, match="unexpected constructor"):
        parse_spec("print(1)")
    with pytest.raises(ValueError):
        parse_spec("SV(mu_prior=__import__('os'))")
    assert parse_spec("Normal(0.0, 1e4)") == Normal(0.0, 1e4)


def test_comparison_helpers_against_known_cases():
    """The MCSE of iid draws is ≈ the iid quantile s.e.; identical summaries give z = 0; the bound
    is 4 up to m = 157 comparisons and grows slowly after (α = 1%); the message names the worst."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5_000))
    q, m = summarise(x)
    se = np.sqrt(PROBS * (1 - PROBS) / x.size) * math.sqrt(2 * math.pi) * np.exp(0.5 * q**2)
    np.testing.assert_allclose(m, se, rtol=0.35)
    np.testing.assert_array_equal(Comparison.of("x", ["x"], x, q, m).z, 0.0)
    assert z_bound(55) == z_bound(157) == 4.0 and 4.0 < z_bound(159) < z_bound(2_000) < 4.6
    shifted = Comparison("x", ["x"], np.array([[0.0, 0.0, 4.5, 0.0, 0.0]]))
    with pytest.raises(AssertionError, match=r"x\[x\] q50: z = \+4.50"):
        assert_parity([shifted], "case")
    w = np.full(20_000, 1 / 20_000)
    xs = rng.standard_normal(20_000)
    qw, mw = weighted_quantiles(xs, w)
    np.testing.assert_allclose(qw, np.quantile(xs, PROBS), atol=2e-3)
    np.testing.assert_allclose(mw, se * math.sqrt(x.size / xs.size), rtol=0.1)


@pytest.mark.parametrize("name", fixture_params(PR_FIXTURES))
def test_stochvol_parity(name):
    """PR tier (exit criterion): ``main`` and ``defaults`` runs within 4 quantile-MCSE; the SV-t
    run must mix and recover its DGP, and its stochvol comparison is a strict ``xfail``
    (:class:`OracleDeviation` only)."""
    check_fixture(name, PR_DRAWS, PR_BURN)


def test_sv_t_matches_the_exact_importance_sampling_posterior():
    """PR tier, replacing the SV-t stochvol row: our SV-t (ν = 5 fixed, standardized,
    ``exact=False``) on four observations against the exact posterior by importance sampling:
    quantiles of ``μ, φ, σ``, ``h_0..h_4`` and ``τ_2`` (the outlier) within 4 MCSE. Run at the
    fixtures' ``T_pad = 320`` and PR draws, so it shares the compiled program of the SV-t row
    (and checks that 316 padded steps are inert, D-15)."""
    run = run_block(TINY_Y, TINY, PR_DRAWS, PR_BURN, salt=7, T_pad=FIXTURE_T_PAD)
    ref = is_oracle(TINY_Y, TINY, TINY_TAU_OBS, size=1_000_000, seed=11)
    params, h, log_tau = our_draws(run.fit)
    T = TINY_Y.size
    assert_parity(
        [
            Comparison.of("param", ["mu", "phi", "sigma"], params, *ref["param"]),
            Comparison.of("h", [f"t={t}" for t in range(T + 1)], h[..., : T + 1], *ref["h"]),
            Comparison.of(
                "tau",
                [f"t={TINY_TAU_OBS}"],
                np.exp(log_tau[..., TINY_TAU_OBS - 1 : TINY_TAU_OBS]),
                *ref["tau"],
            ),
        ],
        "SV-t vs importance sampling",
    )


@pytest.mark.slow
@pytest.mark.parity_long
@pytest.mark.parametrize("name", fixture_params(list(FIXTURE_RUNS)))
def test_stochvol_parity_long(name):
    """Nightly: every run (SV-t as in the PR tier, the optional exact rows too) at 4 × 40,000
    kept draws."""
    check_fixture(name, LONG_DRAWS, LONG_BURN)
