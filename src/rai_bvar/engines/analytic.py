"""Conjugate Normal-Inverse-Wishart engine with the GLP (2015) hierarchy (SPEC §4.6).

Prior: ``Σ ~ IW(Ψ, d)`` with ``Ψ = diag(ψ)``, ``d = n + 2``, and ``vec(B) | Σ ~ N(vec(b), Σ ⊗ Ω)``
with Ω diagonal. The Minnesota part enters as ``k`` dummy rows ``Ω^{−1/2}(B − b)``; SOC/SUR dummy
observations are further rows. The posterior comes from one QR of the stacked ``[X* | Y*]``,
never from explicit inverses (SPEC §8).

Marginal likelihood (GLP 2015, appendix) of ``T`` rows of data:

    log p(Y | h) = −(nT/2) log π + log Γ_n((T+d)/2) − log Γ_n(d/2) − (n/2) log|Ω|
                   − (n/2) log|X'X + Ω⁻¹| + (d/2) log|Ψ| − ((T+d)/2) log|S̄|,

with ``S̄ = Ψ + ε̂'ε̂ + (B̂ − b)'Ω⁻¹(B̂ − b)``. With SOC/SUR dummies the marginal likelihood of
the data is ``ML(dummies + data) − ML(dummies)``; the dummies-only term uses the determinant
lemma on its ``m ≤ n + 1`` rows. With the Lenza-Primiceri COVID scaling, rows are divided by
``s_t`` and the Jacobian ``−n Σ log s_t`` is added.

Conventions (documented choices):

* ψ default (``psi="ar"``): residual variance of a univariate AR(p) with constant by least
  squares; with COVID scaling active it is estimated on the periods before ``t*`` only, where
  ``s_t = 1`` (as Lenza & Primiceri do). If fewer than ``2p + 2`` such periods exist (rolling
  windows that start near or after ``t*``), the AR regression uses the whole window without
  the rows whose dependent period or lags fall in ``t* … t*+2`` (with a warning).
* SOC dummies are built only for variables whose own-lag prior mean is non-zero
  (``soc_scope="persistent"``, default): for a differenced variable a sum-of-coefficients pull
  towards a unit root would contradict the transform-aware prior. ``soc_scope="all"`` keeps
  GLP's rows for every variable. SUR is unchanged (it only centres the intercepts on ``ȳ0``).
* ``ȳ0`` is the mean of the first ``p`` observations (GLP).
* Hyperparameters are optimised and sampled on unconstrained scales: ``log`` for positive
  ones, ``logit`` for the COVID decay ρ ∈ (0, 1).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import expit, logit, multigammaln

from rai_bvar._version import __version__
from rai_bvar.data import transforms
from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import lagged_design, n_deterministic
from rai_bvar.data.outliers import COVID_NAMES, LenzaPrimiceri
from rai_bvar.diagnostics import ess_bulk, rhat
from rai_bvar.forecast.result import ForecastResult
from rai_bvar.forecast.simulate import simulate_linear_var
from rai_bvar.linalg import (
    gram_cholesky,
    qr_least_squares,
    sample_inverse_wishart_chol,
    sample_matrix_normal,
)
from rai_bvar.models.base import SeedLike, seed_label, spec_hash_of
from rai_bvar.priors.minnesota import Minnesota

_LOG_PI = float(np.log(np.pi))
_SCALARS = frozenset({"lambda", "alpha", "soc", "sur"})
_CHUNK_ELEMENTS = 20_000_000  # coefficient-draw elements simulated at a time (~160 MB)
_DRAW_BLOCK = 16  # parameter draws per random stream (chunk-size invariance)
_STATIONARITY_TOL = 1e-6  # accepted predicted Newton gain at an EB mode (log-posterior units)
_COVID_WINDOW = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"))


@dataclass(frozen=True)
class NIWPosterior:
    """``Σ | Y ~ IW(S̄, ν̄)`` and ``B | Σ, Y ~ MN(B̄, (RᵀR)⁻¹, Σ)``."""

    mean: np.ndarray  # B̄, (k, n)
    row_upper: np.ndarray  # R, (k, k): X*'X* = RᵀR
    scale_chol: np.ndarray  # lower Cholesky factor of S̄, (n, n)
    df: float  # ν̄

    @property
    def scale(self) -> np.ndarray:
        return self.scale_chol @ self.scale_chol.T

    @property
    def sigma_mean(self) -> np.ndarray:
        """Posterior mean of Σ: ``S̄ / (ν̄ − n − 1)``."""
        return self.scale / (self.df - self.scale.shape[0] - 1.0)

    def omega(self) -> np.ndarray:
        """Posterior row covariance ``(X*'X*)⁻¹``, dense. For inspection and tests only; the
        engine itself never forms it."""
        k = self.row_upper.shape[0]
        return cho_solve((self.row_upper, False), np.eye(k))

    def draw(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        chols = sample_inverse_wishart_chol(self.scale_chol, self.df, size, rng)
        return sample_matrix_normal(self.mean, self.row_upper, chols, rng), chols


@dataclass
class MHResult:
    """Random-walk Metropolis-Hastings draws of the free hyperparameters (natural scale)."""

    names: tuple[str, ...]
    chains: np.ndarray  # (chains, draws, m)
    acceptance: np.ndarray  # (chains,)
    rhat: dict[str, float]
    ess: dict[str, float]
    converged: bool
    seed: int

    @property
    def draws(self) -> dict[str, np.ndarray]:
        pooled = self.chains.reshape(-1, self.chains.shape[-1])
        return {name: pooled[:, i] for i, name in enumerate(self.names)}


@dataclass
class _Free:
    """The free hyperparameters as one flat vector with bounds and unconstrained transforms.

    ``blocks`` holds ``(key, size, index)``: scalars and ψ use ``index = -1``; COVID parameters
    are single components ``index`` of the ``covid`` vector (only those the data inform).
    ``logit`` marks coordinates in (0, 1) that are mapped with the logit instead of the log.
    """

    blocks: list[tuple[str, int, int]]
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    logit: np.ndarray

    def vector(self, hyper: dict[str, Any]) -> np.ndarray:
        parts = []
        for key, _, index in self.blocks:
            value = np.atleast_1d(np.asarray(hyper[key], dtype=np.float64))
            parts.append(value if index < 0 else value[index : index + 1])
        return np.concatenate(parts) if parts else np.zeros(0)

    def merge(self, base: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        pos = 0
        for key, size, index in self.blocks:
            if key in _SCALARS:
                out[key] = float(x[pos])
            elif index < 0:
                out[key] = np.array(x[pos : pos + size], dtype=np.float64)
            else:
                out[key][index] = x[pos]
            pos += size
        return out

    def to_z(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = np.empty_like(x)
        z[~self.logit] = np.log(x[~self.logit])
        z[self.logit] = logit(np.clip(x[self.logit], 1e-300, 1.0 - 1e-16))
        return z

    def from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        x = np.empty_like(z)
        x[~self.logit] = np.exp(z[~self.logit])
        x[self.logit] = expit(z[self.logit])
        return x

    def log_jacobian(self, z: np.ndarray) -> float:
        """``log |dx/dz|`` summed over coordinates (``z`` for log, ``log x(1−x)`` for logit)."""
        z = np.asarray(z, dtype=np.float64)
        u = z[self.logit]
        return float(
            z[~self.logit].sum() - np.logaddexp(0.0, u).sum() - np.logaddexp(0.0, -u).sum()
        )

    def z_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.to_z(self.lower), self.to_z(self.upper)


class ConjugateProblem:
    """Everything about a conjugate Minnesota BVAR that does not depend on hyperparameters."""

    def __init__(
        self,
        data: VARData,
        prior: Minnesota,
        lags: int,
        deterministic: str,
        outliers: LenzaPrimiceri | None,
    ):
        self.data, self.prior, self.lags, self.deterministic = data, prior, lags, deterministic
        self.notes: list[str] = []  # warnings raised while setting up (emitted by AnalyticFit)
        values = data.values
        if len(values) <= lags:
            raise ValueError(
                f"need more than {lags} observations for a VAR({lags}), got {len(values)}"
            )
        self.Y, self.X = lagged_design(values, lags, deterministic)
        self.T, self.n = self.Y.shape
        self.k = self.X.shape[1]
        self.n_det = n_deterministic(deterministic)
        self.d = self.n + 2.0
        self.periods = data.index[lags:]
        self.ybar0 = values[:lags].mean(axis=0)
        self.means = np.array(prior.own_lag_means(data.variables, data.prior_means()))
        self.b = np.zeros((self.k, self.n))
        self.b[self.n_det + np.arange(self.n), np.arange(self.n)] = self.means
        scope_all = prior.soc_scope == "all"
        self.soc_rows = np.arange(self.n) if scope_all else np.flatnonzero(self.means != 0.0)
        if prior.soc is not None and self.soc_rows.size == 0:
            raise ValueError(
                "the SOC dummy needs at least one variable with a non-zero own-lag prior mean"
            )
        lag_index = np.repeat(np.arange(1, lags + 1), self.n)
        self.lag_of = np.concatenate([np.zeros(self.n_det), lag_index]).astype(np.float64)
        self.var_of = np.concatenate([np.full(self.n_det, -1), np.tile(np.arange(self.n), lags)])
        self.outliers = outliers if outliers is not None and outliers.active(self.periods) else None
        self.covid_informed = (
            self.outliers.informed(self.periods)
            if self.outliers is not None
            else np.zeros(4, dtype=bool)
        )
        self.default = self._defaults(values)
        self.free = self._free_space()

    # ------------------------------------------------------------------ hyperparameters

    def _defaults(self, values: np.ndarray) -> dict[str, Any]:
        p = self.prior
        if p.psi == "ar":
            psi = self._ar_psi(values)
        else:
            psi = np.array(p.psi, dtype=np.float64)
            if psi.shape != (self.n,):
                raise ValueError(f"psi needs {self.n} values, got {psi.size}")
        covid = None
        if self.outliers is not None:
            if self.outliers.fixed is not None:
                covid = np.array(self.outliers.fixed)
            else:  # informed components start at a COVID-sized scale, the rest sit at the prior mode
                covid = np.where(
                    self.covid_informed, np.array([2.0, 2.0, 2.0, 0.8]), self.outliers.mode()
                )
        return {
            "lambda": p.lambda_,
            "alpha": p.alpha,
            "soc": p.soc,
            "sur": p.sur,
            "psi": psi,
            "covid": covid,
        }

    def _ar_psi(self, values: np.ndarray) -> np.ndarray:
        names = self.data.variables
        if self.outliers is None:
            return _ar_residual_variances(values, self.lags, names, "sample", self.notes)
        t_star = self.outliers.start_period(self.data.index.freq)
        pre = values[np.asarray(self.data.index < t_star)]
        if len(pre) >= 2 * self.lags + 2:  # residual variances where s_t = 1
            return _ar_residual_variances(
                pre, self.lags, names, f"periods before t* = {t_star}", self.notes
            )
        # too few pre-t* periods: drop the regression rows touching the free-scale burst
        j = self.outliers.offsets(self.data.index)[self.lags :]
        keep = (j < 0) | (j > 2 + self.lags)
        window = f"window without the rows touching {t_star}..{t_star + 2}"
        self.notes.append(
            f"psi='ar': fewer than {2 * self.lags + 2} observations before t* = {t_star}; "
            f"AR({self.lags}) residual variances use the {window}"
        )
        return _ar_residual_variances(values, self.lags, names, window, self.notes, row_mask=keep)

    def _free_space(self) -> _Free:
        hp = self.prior.hyperpriors
        blocks, names, lower, upper, is_logit = [], [], [], [], []
        bounded = {"lambda": hp.lambda_, "alpha": hp.alpha, "soc": hp.soc, "sur": hp.sur}
        for name in self.prior.free:
            if name in bounded:
                blocks.append((name, 1, -1))
                names.append(name)
                lower.append(bounded[name].lower)
                upper.append(bounded[name].upper)
                is_logit.append(False)
            else:  # psi
                mode = self.default["psi"]
                lo = mode / 100.0 if hp.psi.lower is None else np.full(self.n, hp.psi.lower)
                hi = mode * 100.0 if hp.psi.upper is None else np.full(self.n, hp.psi.upper)
                inverted = [
                    v for v, a, b in zip(self.data.variables, lo, hi, strict=True) if not a < b
                ]
                if inverted:
                    raise ValueError(
                        f"psi bounds are empty for {inverted}: set both InvGammaPrior bounds "
                        "(a missing bound defaults to the AR variance /100 or ×100)"
                    )
                blocks.append(("psi", self.n, -1))
                names += [f"psi_{v}" for v in self.data.variables]
                lower += list(lo)
                upper += list(hi)
                is_logit += [False] * self.n
        if self.outliers is not None and self.outliers.fixed is None:
            if self.prior.hierarchy == "fixed":
                raise ValueError(
                    "Lenza-Primiceri scaling needs fixed=(s0, s1, s2, rho) with hierarchy='fixed'"
                )
            for i, (lo, hi) in enumerate(self.outliers.bounds()):
                if self.covid_informed[i]:
                    blocks.append(("covid", 1, i))
                    names.append(COVID_NAMES[i])
                    lower.append(lo)
                    upper.append(hi)
                    is_logit.append(i == 3)  # ρ ∈ (0, 1)
        return _Free(
            blocks,
            tuple(names),
            np.array(lower, dtype=float),
            np.array(upper, dtype=float),
            np.array(is_logit, dtype=bool),
        )

    @property
    def covid_prior_only(self) -> np.ndarray:
        """COVID components that are free but uninformed by the sample (drawn from the prior)."""
        if self.outliers is None or self.outliers.fixed is not None:
            return np.zeros(4, dtype=bool)
        return ~self.covid_informed

    def complete(self, hyper: dict[str, Any] | None) -> dict[str, Any]:
        """Defaults overridden by ``hyper``, validated."""
        out = dict(self.default)
        for key, value in (hyper or {}).items():
            if key not in out:
                raise KeyError(f"unknown hyperparameter {key!r}; expected one of {sorted(out)}")
            out[key] = value
        free_scalars = {key for key, _, _ in self.free.blocks if key in _SCALARS}
        for key in ("lambda", "alpha"):
            if not (
                isinstance(out[key], int | float | np.number)
                and np.isfinite(out[key])
                and out[key] > 0
            ):
                raise ValueError(f"{key} must be positive and finite, got {out[key]!r}")
            out[key] = float(out[key])
        for key in ("soc", "sur"):
            if out[key] is None:
                if key in free_scalars:
                    raise ValueError(f"{key} is a free hyperparameter here and cannot be None")
                continue
            if not (
                isinstance(out[key], int | float | np.number)
                and np.isfinite(out[key])
                and out[key] > 0
            ):
                raise ValueError(f"{key} must be None or positive and finite, got {out[key]!r}")
            out[key] = float(out[key])
        if out["soc"] is not None and self.soc_rows.size == 0:
            raise ValueError(
                "the SOC dummy needs at least one variable with a non-zero own-lag prior mean"
            )
        psi = np.asarray(out["psi"], dtype=np.float64)
        if psi.shape != (self.n,) or not np.all(np.isfinite(psi) & (psi > 0)):
            raise ValueError(f"psi must be {self.n} positive finite values")
        out["psi"] = psi
        if self.outliers is None:
            if out["covid"] is not None:
                raise ValueError("covid parameters given but no Lenza-Primiceri scaling is active")
        else:
            covid = np.asarray(out["covid"], dtype=np.float64)
            if covid.shape != (4,) or not np.all(np.isfinite(covid)) or np.any(covid[:3] <= 0):
                raise ValueError("covid must be four finite values with s0, s1, s2 > 0")
            if not 0.0 <= covid[3] < 1.0:
                raise ValueError("covid rho must lie in [0, 1)")
            out["covid"] = covid
        return out

    # ------------------------------------------------------------------ prior pieces

    def omega(self, h: dict[str, Any]) -> np.ndarray:
        scale = h["psi"][np.maximum(self.var_of, 0)] / (self.d - self.n - 1.0)
        lags = np.maximum(self.lag_of, 1.0)  # deterministic rows are overwritten below
        lagged = h["lambda"] ** 2 / (lags ** h["alpha"] * scale)
        return np.where(self.var_of < 0, self.prior.var_const, lagged)

    def dummies(self, h: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        rows_y, rows_x = [], []
        if h["soc"] is not None:
            yd = (np.diag(self.ybar0) / h["soc"])[self.soc_rows]
            rows_y.append(yd)
            rows_x.append(np.hstack([np.zeros((len(yd), self.n_det)), np.tile(yd, self.lags)]))
        if h["sur"] is not None:
            yd = self.ybar0[None, :] / h["sur"]
            det = np.zeros((1, self.n_det))
            if self.n_det:
                det[0, 0] = 1.0 / h["sur"]  # constant only; a trend gets no SUR weight
            rows_y.append(yd)
            rows_x.append(np.hstack([det, np.tile(yd, self.lags)]))
        if not rows_y:
            return np.zeros((0, self.n)), np.zeros((0, self.k))
        return np.vstack(rows_y), np.vstack(rows_x)

    def row_scales(self, h: dict[str, Any]) -> np.ndarray:
        if self.outliers is None:
            return np.ones(self.T)
        return self.outliers.scales(self.periods, h["covid"])

    def log_prior(self, h: dict[str, Any]) -> float:
        """Log hyperprior density of the free hyperparameters (natural scale)."""
        hp, total = self.prior.hyperpriors, 0.0
        bounded = {"lambda": hp.lambda_, "alpha": hp.alpha, "soc": hp.soc, "sur": hp.sur}
        covid_terms = None
        for key, _, index in self.free.blocks:
            if key in bounded:
                total += bounded[key].logpdf(h[key])
            elif key == "psi":
                c = self.d - self.n - 1.0
                total += sum(hp.psi.logpdf(v / c) for v in h["psi"]) - self.n * np.log(c)
            else:
                if covid_terms is None:
                    covid_terms = self.outliers.log_prior_components(h["covid"])
                total += covid_terms[index]
        return float(total)

    # ------------------------------------------------------------------ likelihood

    def _core(self, Yd: np.ndarray, Xd: np.ndarray, omega: np.ndarray, psi: np.ndarray):
        """Log ML and posterior from the QR of ``[X* | Y*]`` (prior rows stacked on the data)."""
        rows, k, n, d = Yd.shape[0], self.k, self.n, self.d
        w = 1.0 / np.sqrt(omega)
        Xs = np.vstack([np.diag(w), Xd])
        Ys = np.vstack([w[:, None] * self.b, Yd])
        R = np.linalg.qr(np.hstack([Xs, Ys]), mode="r")
        signs = np.sign(np.diag(R)[:k])
        signs[signs == 0] = 1.0
        R[:k] *= signs[:, None]
        R_xx, R_xy, R_yy = R[:k, :k], R[:k, k:], R[k:, k:]
        if np.any(np.diag(R_xx) <= 1e-10 * np.linalg.norm(Xs, axis=0)):
            raise np.linalg.LinAlgError("stacked regressors are (numerically) rank deficient")
        B = solve_triangular(R_xx, R_xy, lower=False)
        S_chol = gram_cholesky(np.vstack([np.diag(np.sqrt(psi)), R_yy]))
        log_ml = (
            -0.5 * n * rows * _LOG_PI
            + multigammaln(0.5 * (rows + d), n)
            - multigammaln(0.5 * d, n)
            - 0.5 * n * np.log(omega).sum()
            - n * np.log(np.diag(R_xx)).sum()
            + 0.5 * d * np.log(psi).sum()
            - (rows + d) * np.log(np.diag(S_chol)).sum()
        )
        return float(log_ml), NIWPosterior(B, R_xx, S_chol, d + rows)

    def _log_ml_small(
        self, Yd: np.ndarray, Xd: np.ndarray, omega: np.ndarray, psi: np.ndarray
    ) -> float:
        """Log ML of a few rows via ``|Ω||Ω⁻¹ + X'X| = |I + XΩX'|`` (no k × k factorisation)."""
        rows, n, d = Yd.shape[0], self.n, self.d
        G = np.eye(rows) + (Xd * omega) @ Xd.T
        L = np.linalg.cholesky(G)
        Z = solve_triangular(L, Yd - Xd @ self.b, lower=True)
        S_chol = gram_cholesky(np.vstack([np.diag(np.sqrt(psi)), Z]))
        return float(
            -0.5 * n * rows * _LOG_PI
            + multigammaln(0.5 * (rows + d), n)
            - multigammaln(0.5 * d, n)
            - n * np.log(np.diag(L)).sum()
            + 0.5 * d * np.log(psi).sum()
            - (rows + d) * np.log(np.diag(S_chol)).sum()
        )

    def _stacked(self, h: dict[str, Any]):
        omega = self.omega(h)
        s = self.row_scales(h)
        Yd, Xd = self.dummies(h)
        Y = np.vstack([Yd, self.Y / s[:, None]])
        X = np.vstack([Xd, self.X / s[:, None]])
        return omega, s, Yd, Xd, Y, X

    def evaluate(self, h: dict[str, Any]) -> tuple[float, NIWPosterior]:
        """``(log p(Y | h), posterior)`` for complete hyperparameters ``h``."""
        omega, s, Yd, Xd, Y, X = self._stacked(h)
        log_ml, post = self._core(Y, X, omega, h["psi"])
        if Yd.shape[0]:
            log_ml -= self._log_ml_small(Yd, Xd, omega, h["psi"])
        return log_ml - self.n * np.log(s).sum(), post

    def posterior(self, h: dict[str, Any]) -> NIWPosterior:
        """The NIW posterior at ``h`` without the marginal-likelihood bookkeeping."""
        omega, _, _, _, Y, X = self._stacked(h)
        return self._core(Y, X, omega, h["psi"])[1]

    def log_posterior_vector(self, x: np.ndarray) -> float:
        """Log ML + log hyperprior at free values ``x`` (natural scale; clipped into the box)."""
        if not np.all(np.isfinite(x)):
            return -np.inf
        h = self.free.merge(self.default, np.clip(x, self.free.lower, self.free.upper))
        try:
            value = self.evaluate(h)[0] + self.log_prior(h)
        except np.linalg.LinAlgError:
            return -np.inf
        return value if np.isfinite(value) else -np.inf


class AnalyticFit:
    """A fitted conjugate BVAR: hyperparameters, posterior, forecasts and impulse responses.

    ``warnings`` collects every warning the fit raised (also recorded in forecast attrs, so
    backtests can aggregate them across origins).
    """

    engine = "analytic"

    def __init__(self, model, data: VARData, seed: int | None):
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0
        ):
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        self.model, self.data = model, data
        self.warnings: list[str] = []
        prior: Minnesota = model.mean.prior
        self.problem = ConjugateProblem(
            data, prior, model.lags, model.deterministic, model.outliers
        )
        for note in self.problem.notes + _setting_warnings(self.problem, model, data):
            self._warn(note)
        self.optimization: dict[str, Any] | None = None
        self.mh: MHResult | None = None
        self.mh_seed = prior.mh.seed if seed is None else int(seed)
        self.hyper = self.problem.default
        if self.problem.free.names:
            self.hyper = self._optimise()
            if prior.hierarchy == "mh":
                self.mh = _metropolis(
                    self.problem, self.problem.free.vector(self.hyper), prior, self.mh_seed
                )
                if not self.mh.converged:
                    self._warn(_mh_warning(self.mh, prior))
        self._posterior = self.problem.posterior(self.hyper)

    def _warn(self, message: str, stacklevel: int = 4) -> None:
        self.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=stacklevel)

    # ------------------------------------------------------------------ inspection

    @property
    def psi(self) -> np.ndarray:
        return np.asarray(self.hyper["psi"])

    @property
    def converged(self) -> bool:
        """EB optimisation succeeded and, with MH, the chains passed R-hat and ESS checks."""
        optimised = self.optimization is None or self.optimization["success"]
        return optimised and (self.mh is None or self.mh.converged)

    def log_ml(self, hyper: dict[str, Any] | None = None) -> float:
        """``log p(Y | h)``: the data marginal likelihood, without hyperprior terms."""
        return self.problem.evaluate(self._merge(hyper))[0]

    def log_posterior(self, hyper: dict[str, Any] | None = None) -> float:
        """``log p(Y | h) + log p(h)`` over the free hyperparameters (natural-scale densities)."""
        h = self._merge(hyper)
        return self.problem.evaluate(h)[0] + self.problem.log_prior(h)

    def posterior(self, hyper: dict[str, Any] | None = None) -> NIWPosterior:
        return self._posterior if hyper is None else self.problem.posterior(self._merge(hyper))

    def summary(self) -> pd.DataFrame:
        """Hyperparameter values in use and, with MH, posterior quantiles and diagnostics."""
        x = self.problem.free.vector(self.hyper)
        frame = pd.DataFrame({"mode": x}, index=list(self.problem.free.names))
        if self.mh is not None:
            q = np.quantile(
                self.mh.chains.reshape(-1, len(self.mh.names)), [0.05, 0.5, 0.95], axis=0
            )
            frame["q05"], frame["q50"], frame["q95"] = q
            frame["rhat"] = [self.mh.rhat[nm] for nm in frame.index]
            frame["ess"] = [self.mh.ess[nm] for nm in frame.index]
        return frame

    # ------------------------------------------------------------------ draws

    def draw_parameters(self, size: int, rng: SeedLike = None) -> tuple[np.ndarray, np.ndarray]:
        """Posterior draws ``(coefs (S, k, n), chol (S, n, n))``, integrating over MH hyperparameters."""
        coefs = np.empty((size, self.problem.k, self.problem.n))
        chol = np.empty((size, self.problem.n, self.problem.n))
        for sl, c, ch, _ in self._iter_draws(size, np.random.default_rng(rng)):
            coefs[sl], chol[sl] = c, ch
        return coefs, chol

    def _iter_draws(
        self, size: int, rng: np.random.Generator
    ) -> Iterator[tuple[slice, Any, Any, Any]]:
        """Parameter draws in chunks of bounded memory: ``(slice, coefs, chol, covid or None)``.

        With MH, ``size`` hyperparameter draws are taken evenly from the pooled chains and one
        ``(B, Σ)`` draw is made from the conditional NIW posterior of each. COVID parameters the
        sample does not inform are drawn from their prior. Draws come in fixed blocks of
        ``_DRAW_BLOCK`` with their own random streams, so results do not depend on the chunk
        size (memory setting); one chunk holds at most ``max(chunk, _DRAW_BLOCK)`` draws.
        """
        if size < 1:
            raise ValueError("draws must be positive")
        prob = self.problem
        chunk = max(1, _CHUNK_ELEMENTS // (prob.k * prob.n))
        if self.mh is None:
            groups = [(self._posterior, self.hyper, size)]
        else:
            pooled = self.mh.chains.reshape(-1, self.mh.chains.shape[-1])
            picks = np.round(np.linspace(0, len(pooled) - 1, size)).astype(int)
            unique, counts = np.unique(pooled[picks], axis=0, return_counts=True)
            groups = [
                (None, prob.free.merge(self.hyper, x), int(c))
                for x, c in zip(unique, counts, strict=True)
            ]
        blocks = []
        for gid, (post, h, count) in enumerate(groups):
            while count > 0:
                take = min(count, _DRAW_BLOCK)
                blocks.append((gid, post, h, take))
                count -= take
        # derive the streams with draws from ``rng`` so that it advances (seed labels stay unique)
        param_seed, covid_seed = rng.integers(0, 2**63, size=2)
        streams = np.random.default_rng(param_seed).spawn(len(blocks))
        covid_rng = np.random.default_rng(covid_seed)
        prior_only = prob.covid_prior_only if self.mh is not None else np.zeros(4, dtype=bool)
        prior_draws = prob.outliers.sample_prior(size, covid_rng) if prior_only.any() else None
        batches, batch, filled = [], [], 0
        for b, block in enumerate(blocks):
            if batch and filled + block[3] > chunk:
                batches.append(batch)
                batch, filled = [], 0
            batch.append((b, *block))
            filled += block[3]
        batches.append(batch)

        cached: tuple[int, NIWPosterior] | None = None
        start = 0
        for batch in batches:
            total = sum(block[4] for block in batch)
            coefs = np.empty((total, prob.k, prob.n))
            chol = np.empty((total, prob.n, prob.n))
            covid = None if prob.outliers is None else np.empty((total, 4))
            pos = 0
            for b, gid, post, h, take in batch:
                if post is None:
                    if cached is None or cached[0] != gid:
                        cached = (gid, prob.posterior(h))
                    post = cached[1]
                coefs[pos : pos + take], chol[pos : pos + take] = post.draw(take, streams[b])
                if covid is not None:
                    covid[pos : pos + take] = h["covid"]
                pos += take
            if prior_draws is not None:
                covid[:, prior_only] = prior_draws[start : start + total][:, prior_only]
            yield slice(start, start + total), coefs, chol, covid
            start += total

    def forecast(self, horizon: int, draws: int = 1000, seed: SeedLike = None) -> ForecastResult:
        if horizon < 1 or draws < 1:
            raise ValueError("horizon and draws must be positive")
        rng = np.random.default_rng(seed)
        label = seed_label(seed, rng)
        prob, values = self.problem, self.data.values
        n = prob.n
        paths, means = np.empty((draws, horizon, n)), np.empty((draws, horizon, n))
        scaled = prob.outliers is not None
        # without COVID scaling the conditional factor does not depend on h: keep (S, n, n) and
        # expose a zero-stride view instead of materialising (S, H, n, n)
        store = np.empty((draws, horizon, n, n)) if scaled else np.empty((draws, n, n))
        future = pd.period_range(self.data.last + 1, periods=horizon, freq=self.data.index.freq)
        shock_rng = rng.spawn(1)[0]  # leaves rng's own stream to the parameter draws
        shocks = shock_rng.standard_normal((draws, horizon, n))
        for sl, coefs, chol, covid in self._iter_draws(draws, rng):
            scale = None if covid is None else prob.outliers.scales(future, covid)
            p, m, c = simulate_linear_var(
                coefs,
                chol,
                values[-self.model.lags :],
                horizon,
                self.model.deterministic,
                len(values),
                shock_rng,
                scale,
                shocks=shocks[sl],
            )
            paths[sl], means[sl] = p, m
            store[sl] = c if scaled else chol
        cond_chol = store if scaled else np.broadcast_to(store[:, None], (draws, horizon, n, n))
        spec = self.model.spec()
        return ForecastResult.from_arrays(
            paths=paths,
            cond_mean=means,
            cond_chol=cond_chol,
            paths_level=self.data.inverse_paths(paths),
            variables=self.data.variables,
            origin=self.data.last,
            attrs={
                "model": self.model.name,
                "engine": self.engine,
                "spec": spec,
                "spec_hash": spec_hash_of(spec),
                "tcodes": self.data.tcodes,
                "seed": label,
                "mh_seed": self.mh_seed if self.mh is not None else -1,
                "converged": int(self.converged),
                "warnings": "\n".join(self.warnings),
                "rai_bvar_version": __version__,
            },
        )

    def irf(self, horizon: int = 20, draws: int = 1000, seed: SeedLike = None, shock: str = "sd"):
        """Recursive (Cholesky) impulse responses to pre-COVID one-s.d. (or unit) shocks."""
        from rai_bvar.structural import IRFResult, recursive_irf

        if horizon < 0 or draws < 1:
            raise ValueError("horizon must be ≥ 0 and draws positive")
        rng = np.random.default_rng(seed)
        label = seed_label(seed, rng)
        n = self.problem.n
        responses = np.empty((draws, horizon + 1, n, n))
        for sl, coefs, chol, _ in self._iter_draws(draws, rng):
            responses[sl] = recursive_irf(
                coefs, chol, self.model.lags, self.problem.n_det, horizon, shock=shock
            )
        spec = self.model.spec()
        return IRFResult.from_arrays(
            responses,
            self.data.variables,
            shock=shock,
            attrs={
                "model": self.model.name,
                "spec_hash": spec_hash_of(spec),
                "seed": label,
                "mh_seed": self.mh_seed if self.mh is not None else -1,
            },
        )

    # ------------------------------------------------------------------ estimation

    def _merge(self, hyper: dict[str, Any] | None) -> dict[str, Any]:
        return self.problem.complete({**self.hyper, **(hyper or {})})

    def _optimise(self) -> dict[str, Any]:
        """Posterior mode of the free hyperparameters (L-BFGS-B on unconstrained values).

        Gradients are central differences; convergence is judged by the predicted Newton gain
        ``½ Σ g_i² / H_ii`` (scale free), not by an absolute gradient threshold.
        """
        prob, free = self.problem, self.problem.free
        lo, hi = free.z_bounds()
        z0 = np.clip(free.to_z(free.vector(prob.default)), lo, hi)
        state = {"z": z0, "f": None, "evaluations": 0}

        def objective(z: np.ndarray) -> float:
            state["evaluations"] += 1
            value = prob.log_posterior_vector(free.from_z(z))
            if np.isfinite(value):
                state["z"], state["f"] = z.copy(), -value
                return -value
            # finite penalty that grows away from the last finite point (no flat sentinel)
            anchor = 0.0 if state["f"] is None else state["f"]
            return anchor + 1e4 * (1.0 + float(np.sum((z - state["z"]) ** 2)))

        def run(start: np.ndarray):
            res = optimize.minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac="3-point",
                bounds=list(zip(lo, hi, strict=True)),
                options={"ftol": 1e-12, "maxiter": 2000},
            )
            state["evaluations"] += 1
            value = prob.log_posterior_vector(free.from_z(res.x))
            gain, grad = _stationarity(objective, res.x, lo, hi)
            return res, value, gain, grad

        def passes(candidate) -> bool:
            return bool(np.isfinite(candidate[1]) and candidate[2] < _STATIONARITY_TOL)

        def better(candidate, incumbent) -> bool:
            """Higher log posterior; at a tie, a candidate that passes the stationarity test."""
            if not np.isfinite(candidate[1]):
                return False
            if not np.isfinite(incumbent[1]) or candidate[1] > incumbent[1] + 1e-7:
                return True
            return (
                candidate[1] > incumbent[1] - 1e-7 and passes(candidate) and not passes(incumbent)
            )

        # deterministic multistart along the axes with known secondary modes (dummy weights,
        # COVID decay); each start moves one axis away from the default
        x0 = free.from_z(z0)
        starts = [z0]
        for name, values in (("soc", (0.3, 5.0)), ("sur", (0.3, 5.0)), ("covid_rho", (0.95, 0.99))):
            if name in free.names:
                for v in values:
                    x = x0.copy()
                    x[free.names.index(name)] = v
                    if name == "covid_rho" and "covid_s2" in free.names:
                        x[free.names.index("covid_s2")] = 2.0
                    starts.append(np.clip(free.to_z(np.clip(x, free.lower, free.upper)), lo, hi))
        best = run(starts[0])
        for start in starts[1:]:
            candidate = run(start)
            if better(candidate, best):
                best = candidate
        if not passes(best):  # one restart from a perturbed start
            candidate = run(np.clip(z0 + 0.25, lo, hi))
            if better(candidate, best):
                best = candidate
        result, value, gain, grad = best
        ok = passes(best)
        if not ok:
            self._warn(
                f"hyperparameter optimisation did not converge: {result.message}; predicted gain "
                f"{gain:.2g}, projected gradient {grad:.2g}; fit.converged is False",
                stacklevel=5,
            )
        self.optimization = {
            "success": ok,
            "message": str(result.message),
            "evaluations": int(state["evaluations"]),
            "log_posterior": float(value),
            "projected_gradient": float(grad),
            "predicted_gain": float(gain),
        }
        return free.merge(prob.default, np.clip(free.from_z(result.x), free.lower, free.upper))


def _stationarity(objective, z: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: float = 1e-4):
    """``(½ Σ g_i²/H_ii, max |g_i|)`` over coordinates not pinned at a bound.

    Central (or one-sided at a bound) finite differences of the objective. A coordinate at a
    bound whose gradient points out of the box is pinned and ignored; a non-positive curvature
    on an interior coordinate means no local minimum was reached (infinite gain).
    """
    f0 = objective(z)
    gain, max_grad = 0.0, 0.0
    for i in range(z.size):
        e = np.zeros(z.size)
        e[i] = h
        at_lower, at_upper = z[i] - h < lo[i], z[i] + h > hi[i]
        if at_lower and at_upper:
            continue
        if at_lower:
            f1, f2 = objective(z + e), objective(z + 2 * e)
            g, curv = (-3 * f0 + 4 * f1 - f2) / (2 * h), (f0 - 2 * f1 + f2) / h**2
            if g > 0:  # descent would leave the box: pinned
                continue
        elif at_upper:
            f1, f2 = objective(z - e), objective(z - 2 * e)
            g, curv = (3 * f0 - 4 * f1 + f2) / (2 * h), (f0 - 2 * f1 + f2) / h**2
            if g < 0:
                continue
        else:
            fp, fm = objective(z + e), objective(z - e)
            g, curv = (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2
        max_grad = max(max_grad, abs(g))
        if not curv > 0:
            return np.inf, max_grad
        gain += 0.5 * g * g / curv
    return float(gain), float(max_grad)


def _metropolis(
    problem: ConjugateProblem, mode: np.ndarray, prior: Minnesota, seed: int
) -> MHResult:
    """Random-walk MH on unconstrained hyperparameters; the target includes the log-Jacobian.

    The proposal starts from the inverse Hessian at the mode; at ½ and ¾ of burn-in it is
    re-estimated from the chain's recent burn-in draws (shrunk towards the current proposal)
    and the step is re-tuned for the rest of burn-in. Nothing adapts after burn-in.
    """
    settings, free = prior.mh, problem.free
    rng = np.random.default_rng(seed)
    lo, hi = free.z_bounds()
    z_mode = np.clip(free.to_z(mode), lo, hi)
    m = z_mode.size

    def target(z: np.ndarray) -> float:
        return problem.log_posterior_vector(free.from_z(z)) + free.log_jacobian(z)

    base_factor = _proposal_factor(target, z_mode, lo, hi)
    total = settings.burn + settings.draws * settings.thin
    # re-estimate the proposal from the chain's recent burn-in draws at ½ and ¾ of burn-in
    refits = {settings.burn // 2: settings.burn // 4, settings.burn * 3 // 4: settings.burn // 2}
    refits = refits if settings.burn >= 200 else {}
    chains = np.full((settings.chains, settings.draws, m), np.nan)
    accepted = np.zeros(settings.chains)
    for c in range(settings.chains):
        factor = base_factor
        z = np.clip(z_mode + 2.0 * factor @ rng.standard_normal(m), lo, hi)
        current = target(z)
        if not np.isfinite(current):
            z, current = z_mode.copy(), target(z_mode)
        step, window_acc, kept = 2.38 / np.sqrt(m), 0, 0
        burn_draws = np.empty((max(settings.burn, 1), m))
        for it in range(total):
            proposal = z + step * (factor @ rng.standard_normal(m))
            inside = np.all((proposal >= lo) & (proposal <= hi))
            value = target(proposal) if inside else -np.inf
            if np.log(rng.uniform()) < value - current:
                z, current = proposal, value
                window_acc += 1
                if it >= settings.burn:
                    accepted[c] += 1
            if it < settings.burn:  # adaptation during burn-in only
                burn_draws[it] = z
                if (it + 1) % 50 == 0:
                    step *= np.exp(2.0 * (window_acc / 50 - settings.target_acceptance))
                    window_acc = 0
                if it + 1 in refits:
                    window = burn_draws[refits[it + 1] : it + 1]
                    factor, step = _empirical_factor(window, factor), 2.38 / np.sqrt(m)
            elif (it - settings.burn) % settings.thin == 0:
                chains[c, kept] = free.from_z(z)
                kept += 1
        if kept != settings.draws:
            raise AssertionError(f"MH bookkeeping: kept {kept} of {settings.draws} draws")
    names = free.names
    rhats = {nm: rhat(chains[:, :, i]) for i, nm in enumerate(names)}
    ess = {nm: ess_bulk(chains[:, :, i]) for i, nm in enumerate(names)}
    converged = settings.chains > 1 and all(
        np.isfinite(rhats[nm])
        and rhats[nm] < settings.rhat_max
        and np.isfinite(ess[nm])
        and ess[nm] > settings.ess_min
        for nm in names
    )
    acceptance = accepted / (settings.draws * settings.thin)
    return MHResult(names, chains, acceptance, rhats, ess, converged, seed)


def _empirical_factor(draws: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Cholesky factor of the burn-in draws' covariance, shrunk towards the current proposal.

    The window's covariance is blended with the current proposal covariance (rescaled to the
    same trace) with weight ``10m / (10m + distinct rows)``, so a short or sticky window cannot
    produce an ill-conditioned proposal (Haario-style adaptation with regularisation).
    """
    m = draws.shape[1]
    distinct = len(np.unique(draws, axis=0))
    if distinct < m + 2:
        return fallback
    cov = np.atleast_2d(np.cov(draws, rowvar=False))
    if not np.all(np.isfinite(cov)) or not np.trace(cov) > 0:
        return fallback
    base = fallback @ fallback.T
    base = base * (np.trace(cov) / np.trace(base))
    w = 10.0 * m / (10.0 * m + distinct)
    try:
        return np.linalg.cholesky(w * base + (1.0 - w) * cov)
    except np.linalg.LinAlgError:
        return fallback


def _proposal_factor(
    target, z: np.ndarray, lo: np.ndarray, hi: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    """``F`` with ``FFᵀ`` = inverse Hessian of ``−target`` at ``z``.

    Eigenvalues are floored at 1, so no direction gets a proposal s.d. above one unit on the
    unconstrained scale: a direction the data do not inform would otherwise dominate the
    proposal. With more than 20 free hyperparameters only the diagonal curvature is used
    (``2m + 1`` evaluations instead of ``O(m²)``).
    """
    m = z.size
    z = np.clip(z, lo + 2 * step, hi - 2 * step)
    eye = np.eye(m) * step
    if m > 20:
        f0 = target(z)
        diag = np.array(
            [-(target(z + eye[i]) - 2 * f0 + target(z - eye[i])) / step**2 for i in range(m)]
        )
        if not np.all(np.isfinite(diag)):
            return np.eye(m) * 0.1
        return np.diag(1.0 / np.sqrt(np.maximum(np.abs(diag), 1.0)))
    hess = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            ei, ej = eye[i], eye[j]
            value = (
                target(z + ei + ej)
                - target(z + ei - ej)
                - target(z - ei + ej)
                + target(z - ei - ej)
            ) / (4 * step**2)
            hess[i, j] = hess[j, i] = -value
    if not np.all(np.isfinite(hess)):
        return np.eye(m) * 0.1
    eigval, eigvec = np.linalg.eigh(hess)
    eigval = np.maximum(np.abs(eigval), 1.0)
    return eigvec / np.sqrt(eigval)


def _ar_residual_variances(
    values: np.ndarray,
    lags: int,
    names: tuple[str, ...],
    window: str,
    notes: list[str],
    row_mask: np.ndarray | None = None,
) -> np.ndarray:
    """ψ default: residual variance of a univariate AR(p) with constant, by least squares."""
    out = np.empty(values.shape[1])
    small = []
    for j, name in enumerate(names):
        Y, X = lagged_design(values[:, j : j + 1], lags, "c")
        if row_mask is not None:
            Y, X = Y[row_mask], X[row_mask]
        df = Y.shape[0] - X.shape[1]
        if df < 1:
            raise ValueError(
                f"psi='ar' needs at least {2 * lags + 2} usable observations of {name!r} over the "
                f"{window} for AR({lags}) residual variances, got {Y.shape[0] + lags}; pass psi=(...) explicitly"
            )
        try:
            _, _, resid = qr_least_squares(X, Y)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"psi='ar': {name!r} is constant (or collinear with its lags) over the {window} used "
                f"for the AR({lags}) residual variance; pass psi=(...) explicitly"
            ) from None
        rss = float(resid[:, 0] @ resid[:, 0])
        if not np.isfinite(rss) or rss <= (100 * np.finfo(float).eps) ** 2 * float(
            Y[:, 0] @ Y[:, 0]
        ):
            raise ValueError(
                f"psi='ar': {name!r} is fitted (numerically) exactly by an AR({lags}) with constant "
                f"over the {window}; pass psi=(...) explicitly"
            )
        out[j] = rss / df
        if df < 10:
            small.append(name)
    if small:
        notes.append(
            f"psi='ar': AR({lags}) residual variances of {small} rest on fewer than 10 degrees of "
            f"freedom over the {window}; consider passing psi=(...)"
        )
    return out


def _setting_warnings(problem: ConjugateProblem, model, data: VARData) -> list[str]:
    """Messages for the warnings SPEC §5.3 and §5.6 ask fit() to give."""
    out = []
    prior = model.mean.prior
    if prior.mean is not None:
        clash = [
            v
            for v, m in zip(data.variables, problem.means, strict=True)
            if (data.tcodes[v] in transforms.PERSISTENT_TCODES and m < 0.5)
            or (data.tcodes[v] not in transforms.PERSISTENT_TCODES and m >= 0.5)
        ]
        if clash:
            out.append(
                f"own-lag prior means contradict the transformation codes for {clash} "
                "(levels with mean < 0.5 or differences with mean ≥ 0.5)"
            )
    if model.outliers is None:
        start, end = _COVID_WINDOW
        if any(p.end_time >= start and p.start_time <= end for p in problem.periods):
            try:
                LenzaPrimiceri().start_period(data.index.freq)
                advice = "consider outliers='lenza-primiceri'"
            except ValueError:
                advice = "Lenza-Primiceri scaling supports monthly and quarterly data only; consider excluding or dummying 2020"
            out.append(
                f"the estimation sample covers 2020 but no outlier treatment is set; {advice} (SPEC §5.6)"
            )
    return out


def _mh_warning(mh: MHResult, prior: Minnesota) -> str:
    settings = prior.mh
    failing = {
        nm: (round(mh.rhat[nm], 4), round(mh.ess[nm], 1))
        for nm in mh.names
        if not (mh.rhat[nm] < settings.rhat_max and mh.ess[nm] > settings.ess_min)
    }
    parts = []
    if settings.chains == 1:
        parts.append(
            "single chain, so split R-hat cannot detect between-chain disagreement (use chains ≥ 2)"
        )
    if failing:
        parts.append(f"failing (R-hat, bulk ESS): {failing}")
    return f"hyperparameter MH did not converge: {'; '.join(parts)}; acceptance {np.round(mh.acceptance, 3).tolist()}"


__all__ = ["AnalyticFit", "ConjugateProblem", "MHResult", "NIWPosterior"]
