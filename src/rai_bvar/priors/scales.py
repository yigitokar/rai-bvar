"""Per-window prior scales of the MCMC engine (DESIGN §6, §6.6; R9). Pure NumPy, no JAX.

Everything data-dependent in the Phase 2 priors is computed here, per estimation window, from
the window alone, and passed to the engine as arrays (R9; N05 §7.6, §12):

* ``ψ`` (variance units): the Phase 1 default ``psi="ar"``, i.e. the residual variance of a
  univariate AR(p) with constant by least squares over the window, computed by the Phase 1
  helper ``engines.analytic._ar_residual_variances`` (read-only reuse, so the MCMC prior and the
  conjugate EB twin see the same ψ); explicit values from ``Minnesota(psi=(...))``; or an
  override (the EB value, DESIGN §6.5 step 4). Lenza–Primiceri scaling is not available on the
  MCMC engine (D-21), so the COVID ψ rules of Phase 1 never apply here.
* own-lag prior means ``m_i`` from the transformation codes (SPEC §5.3,
  ``Minnesota.own_lag_means``);
* the data frequency, which resolves the outlier prior (D-9).

**The ``AsymmetricConjugate`` adapter** (P3 DESIGN §4.2, WP33). The structural NIG closed form
(``engines.structural_nig``) reads a ``Minnesota`` prior. :func:`structural_minnesota` maps an
``AsymmetricConjugate`` onto its internal structural ``Minnesota`` (``AsymmetricConjugate.
minnesota()``: the same ``lambda_``, ``alpha``, ``psi``, ``mean``, ``cross``, ``contemp``,
``var_const``, ``soc``, ``sur``, ``soc_scope``, at ``form="structural"``, ``hierarchy="fixed"``),
which never reaches the user or a spec; :func:`window_scales` and :func:`conjugate_omega` accept
either prior through it. Every other prior is rejected, as before.

Notation (DESIGN §6): 0-based equation ``i``, 1-based position ``ι = i + 1``, ``d = n + 2``;
the Mode I fixed structural scale is ``σ̄_i² = ψ_i/(d − n + ι − 2) = ψ_i/ι`` (N05 §4.4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from rai_bvar.data.dataset import VARData
from rai_bvar.data.design import n_deterministic
from rai_bvar.engines.analytic import _ar_residual_variances
from rai_bvar.priors.asymmetric import AsymmetricConjugate
from rai_bvar.priors.minnesota import Minnesota

# Keys of a Phase 1 hyperparameter dict (AnalyticFit.hyper). Only lambda and alpha enter the
# coefficient arrays; psi enters through WindowScales; the SOC/SUR dummy rows are built by
# priors.dummies (WP25; N05 §7.7) and the COVID scales are not available on the MCMC engine (D-21).
# A structural NIG dict (P3 §4.3) also carries cross and contemp, which enter through the prior.
_HYPER_KEYS = frozenset({"lambda", "alpha", "psi", "soc", "sur", "covid", "cross", "contemp"})


@dataclass(frozen=True)
class WindowScales:
    """Data-dependent scales of one estimation window (R9).

    ``psi`` (n,) Minnesota variance scales ψ_i; ``own_means`` (n,) own-first-lag prior means;
    ``freq`` the pandas frequency string of the data (``VARData.freq``); ``notes`` messages raised
    while computing ψ (e.g. few AR degrees of freedom), for ``fit.warnings``. Arrays are float64
    and read-only.
    """

    psi: np.ndarray
    own_means: np.ndarray
    freq: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        psi = _readonly(self.psi)
        own = _readonly(self.own_means)
        if psi.ndim != 1 or psi.size < 1 or not np.all(np.isfinite(psi) & (psi > 0)):
            raise ValueError(f"psi must be a non-empty vector of positive finite values, got {psi}")
        if own.shape != psi.shape or not np.all(np.isfinite(own)):
            raise ValueError(f"own_means must be {psi.size} finite values, got {own}")
        object.__setattr__(self, "psi", psi)
        object.__setattr__(self, "own_means", own)
        object.__setattr__(self, "freq", str(self.freq))
        object.__setattr__(self, "notes", tuple(str(m) for m in self.notes))

    @property
    def n(self) -> int:
        return int(self.psi.size)

    @property
    def iota(self) -> np.ndarray:
        """1-based equation positions ``ι = 1..n`` (float64)."""
        return np.arange(1.0, self.n + 1.0)

    @property
    def sigma2_bar(self) -> np.ndarray:
        """Mode I fixed structural scale ``σ̄_i² = ψ_i/ι`` (N05 §4.4, ``d = n + 2``)."""
        return self.psi / self.iota


def structural_minnesota(prior: Minnesota | AsymmetricConjugate) -> Minnesota:
    """The ``Minnesota`` the per-window helpers and the structural NIG closed form read (P3 DESIGN
    §4.2): ``prior`` itself, or the internal structural ``Minnesota`` of an
    ``AsymmetricConjugate`` (its ``minnesota()``; the adapter). ``TypeError`` for any other prior."""
    if isinstance(prior, AsymmetricConjugate):
        return prior.minnesota()
    if isinstance(prior, Minnesota):
        return prior
    raise TypeError(f"prior must be a Minnesota prior, got {type(prior).__name__}")


def window_scales(
    data: VARData,
    prior: Minnesota | AsymmetricConjugate,
    lags: int,
    deterministic: str,
    psi: np.ndarray | None = None,
) -> WindowScales:
    """ψ, own-lag means and frequency of the window ``data`` (DESIGN §6, §6.6).

    ``psi`` (an EB value, DESIGN §6.5) overrides ``prior.psi``; otherwise explicit
    ``Minnesota(psi=(...))`` values are used, or, for ``psi="ar"``, the Phase 1 AR(p)-with-constant
    residual variances over the whole window (identical to ``ConjugateProblem.default["psi"]``
    without Lenza–Primiceri scaling). Only ``data`` is read, so a window ``data.upto(origin)``
    ignores later observations (R9). An ``AsymmetricConjugate`` is read through its internal
    ``Minnesota`` (:func:`structural_minnesota`).
    """
    prior = structural_minnesota(prior)
    if isinstance(lags, bool) or not isinstance(lags, int | np.integer) or lags < 1:
        raise ValueError(f"lags must be a positive integer, got {lags!r}")
    n_deterministic(deterministic)  # validates the kind
    n, T = data.n_vars, len(data)
    if T <= lags:
        raise ValueError(f"need more than {lags} observations for a VAR({lags}), got {T}")
    notes: list[str] = []
    if psi is not None:
        values = np.array(psi, dtype=np.float64)
        source = "psi"
    elif not isinstance(prior.psi, str):
        values = np.array(prior.psi, dtype=np.float64)
        source = "Minnesota.psi"
    else:
        values = _ar_residual_variances(data.values, int(lags), data.variables, "sample", notes)
        source = "psi='ar'"
    if values.shape != (n,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"{source} needs {n} positive finite values, got {values}")
    own = np.array(prior.own_lag_means(data.variables, data.prior_means()), dtype=np.float64)
    return WindowScales(psi=values, own_means=own, freq=data.freq, notes=tuple(notes))


# --------------------------------------------------------------------------- shared helpers


def regressor_layout(n: int, lags: int, n_det: int) -> tuple[np.ndarray, np.ndarray]:
    """``(lag_of, var_of)`` of the ``k = n_det + n·lags`` regressors in the ``data/design.py``
    layout ``x_t = (d_t', y_{t−1}', …, y_{t−p}')'``: lag 0 and variable −1 mark deterministic
    terms; lag ``l`` of variable ``m`` is row ``n_det + (l−1)n + m`` (N05 §2)."""
    _check_dims(n, lags, n_det)
    lag_of = np.concatenate([np.zeros(n_det), np.repeat(np.arange(1.0, lags + 1.0), n)])
    var_of = np.concatenate([np.full(n_det, -1), np.tile(np.arange(n), lags)]).astype(np.int64)
    return lag_of, var_of


def coefficient_groups(n: int, lags: int, n_det: int) -> np.ndarray:
    """``group`` (n, k) int32 of ``CoefPriorArrays``: 0 deterministic, 1 own lag, 2 cross lag."""
    _, var_of = regressor_layout(n, lags, n_det)
    eq = np.arange(n)[:, None]
    groups = np.where(var_of[None, :] < 0, 0, np.where(var_of[None, :] == eq, 1, 2))
    return groups.astype(np.int32)


def prior_mean_matrix(scales: WindowScales, n: int, lags: int, n_det: int) -> np.ndarray:
    """``M`` (k, n): the Minnesota prior mean, zero except the own first lag ``M[n_det + i, i] =
    m_i`` (deterministic terms and longer lags have mean 0; N05 §4.4)."""
    _check_scales(scales, n)
    _check_dims(n, lags, n_det)
    M = np.zeros((n_det + n * lags, n))
    M[n_det + np.arange(n), np.arange(n)] = scales.own_means
    return M


def conjugate_omega(
    scales: WindowScales,
    prior: Minnesota | AsymmetricConjugate,
    lam: float,
    alpha: float,
    n: int,
    lags: int,
    n_det: int,
    *,
    cross: float | None = None,
) -> np.ndarray:
    """λ-free conjugate pattern ``Ω^{(i)}`` (n, k) of every equation (N05 §4.4; DESIGN §6.1):

        Ω^{(i)}_{(l,m)} = κ² c_im (d − n − 1)/(l^α ψ_m),   c_ii = 1, c_im = κ_c² (m ≠ i),
        Ω^{(i)}_det     = κ_d,                              d = n + 2,

    computed in the operation order of ``ConjugateProblem.omega``, so at ``cross = 1`` every row
    equals the Phase 1 ``Ω`` bit for bit. ``κ_c`` is ``cross`` when given (the structural NIG
    closed form passes it per evaluation, P3 §4.3), else ``prior.cross``; an ``AsymmetricConjugate``
    is read through its internal ``Minnesota`` (:func:`structural_minnesota`).
    """
    if isinstance(prior, AsymmetricConjugate):
        prior = prior.minnesota()
    _check_scales(scales, n)
    kappa_c = prior.cross if cross is None else cross
    lag_of, var_of = regressor_layout(n, lags, n_det)
    d = n + 2.0
    scale = scales.psi[np.maximum(var_of, 0)] / (d - n - 1.0)
    lagged = lam**2 / (np.maximum(lag_of, 1.0) ** alpha * scale)  # (k,)
    own = var_of[None, :] == np.arange(n)[:, None]  # (n, k)
    pattern = np.where(own, 1.0, kappa_c**2)
    return np.where(var_of[None, :] < 0, prior.var_const, lagged[None, :] * pattern)


def lambda_alpha(
    prior: Minnesota, hyper: Mapping[str, object], scales: WindowScales
) -> tuple[float, float]:
    """``(κ, α)`` in use: ``hyper`` values (EB, DESIGN §6.5) override the prior's.

    ``hyper`` may be a Phase 1 hyperparameter dict; a ``psi`` entry must equal ``scales.psi``
    (ψ enters through :func:`window_scales`), ``cross``/``contemp`` entries (a structural NIG
    dict, P3 §4.3) must equal the prior's (they enter through ``prior``), and
    ``soc``/``sur``/``covid`` do not enter the diagonal prior arrays. Unknown keys raise
    ``KeyError``.
    """
    unknown = sorted(set(hyper) - _HYPER_KEYS)
    if unknown:
        raise KeyError(f"unknown hyperparameters {unknown}; expected some of {sorted(_HYPER_KEYS)}")
    out = []
    for key, default in (("lambda", prior.lambda_), ("alpha", prior.alpha)):
        value = hyper.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            raise TypeError(f"hyperparameter {key!r} must be a number, got {value!r}")
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"hyperparameter {key!r} must be positive and finite, got {value!r}")
        out.append(float(value))
    psi = hyper.get("psi")
    if psi is not None and not np.array_equal(np.asarray(psi, dtype=np.float64), scales.psi):
        raise ValueError(
            "hyper['psi'] differs from the window's psi: pass it to window_scales(..., psi=...)"
        )
    for key in ("cross", "contemp"):
        if key in hyper and hyper[key] != getattr(prior, key):
            raise ValueError(
                f"hyper[{key!r}] = {hyper[key]!r} differs from the prior's {key} = "
                f"{getattr(prior, key)!r}: these arrays read {key} from the prior"
            )
    return out[0], out[1]


def _check_dims(n: int, lags: int, n_det: int) -> None:
    for name, value, low in (("n", n, 1), ("lags", lags, 1), ("n_det", n_det, 0)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < low:
            raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")


def _check_scales(scales: WindowScales, n: int) -> None:
    if not isinstance(scales, WindowScales):
        raise TypeError(f"scales must be a WindowScales, got {type(scales).__name__}")
    if scales.n != n:
        raise ValueError(f"the window scales describe {scales.n} variables, not n = {n}")


def _readonly(values: object) -> np.ndarray:
    out = np.array(values, dtype=np.float64)  # a copy: the caller's array stays writable
    out.flags.writeable = False
    return out


__all__ = [
    "WindowScales",
    "coefficient_groups",
    "conjugate_omega",
    "lambda_alpha",
    "prior_mean_matrix",
    "regressor_layout",
    "structural_minnesota",
    "window_scales",
]
