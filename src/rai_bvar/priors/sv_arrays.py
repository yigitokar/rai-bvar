"""Volatility prior arrays per window: ``ConstantPrior``, ``SVPrior``, ``TransitoryPrior``
(DESIGN §2.6, §6.1, §6.3–6.4; D-2, D-8, D-9, D-13). NumPy leaves on the host.

* ``Constant``: ``λ_i ~ IG(1 + ι/2, ψ_i/2)`` (shape, scale), mean ``ψ_i/ι`` (N05 §4.2, §4.4;
  Chan 2022 ``ν_i = 1 + i/2``, ``S_i = s_i²/2``). The same law serves Mode I (``"ig"``), Mode II
  (``"iw"``) and ``"iw_equivalent"``: the modes differ only in which coefficient blocks are
  λ-scaled (DESIGN §4.7).
* ``SV`` (stochvol parametrisation, N02 §1, §10.2): ``μ ~ N(b_μ, B_μ)`` with ``B_μ`` a
  variance, data-centred by default at ``b_μ = log(ψ_i/ι)`` (the Mode I scale, DESIGN §6.3;
  open question Q3); ``(φ+1)/2 ~ Beta(a_φ, b_φ)``; ``σ² ~ B_σ·χ²₁`` or ``IG(c₀, C₀)``;
  ``h_0 ~ N(μ, B₀σ²)`` for ``SV(h0=B₀)``; RW ``h_0 ~ N(m₀, V₀)`` with ``m₀ = log(ψ_i/ι)``.
  The offset of ``log(ẽ² + c)`` is ``c_i = c_rel·ψ_i`` with ``c_rel = 1e-12`` (D-2;
  N01 §7.2), or an absolute ``SV(offset=c)``; ``c = 0`` gives ``log_c = −inf`` (zero-offset
  parity mode, N02 §2.4).
* transitory (N03 §0, §3): outlier probability ``p ~ Beta(a_p, b_p)`` resolved from the window
  frequency (``OutlierPrior``, D-9) unless given explicitly; log prior weights of ν on the
  static grid (``DiscreteUniform``: ``−log G`` on its support).
* ``CommonSV`` (P3 §5.1, D-46; :func:`common_sv_prior`): ``ρ ~ N(ρ₀, V_ρ)·1(|ρ| < 1)`` or
  ``(ρ + 1)/2 ~ Beta(a, b)``; ``σ_h² ~ IG(c₀, C₀)`` or ``B·χ²₁``; ``n_eq`` the number of
  equations that share ``h``. Every field of both options is filled (the unused ones with the
  inert placeholders above); ``VolStatic.csv_rho_prior`` and ``csv_sigma2_prior`` select which
  apply. No field depends on the data (R9).

Fields a variant does not use hold inert finite placeholders (0 for locations, 1 for shapes,
scales and variances), so every leaf is finite except ``log_c`` in zero-offset mode and the
ν weights of grid points outside the prior support (``−inf``, weight 0). Every leaf is its own
freshly allocated buffer (no two fields alias, none aliases ``WindowScales``), so perturbing one
leaf in place leaves the others unchanged.
"""

from __future__ import annotations

import numpy as np

from rai_bvar.components.specs import SV, CommonSV
from rai_bvar.engines.mcmc.state import (
    CommonSVPrior,
    ConstantPrior,
    Empty,
    SVPrior,
    TransitoryPrior,
)
from rai_bvar.priors.scales import WindowScales, _check_scales
from rai_bvar.priors.sv import Beta, InvGamma, OutlierPrior, ScaledChi2, TruncatedNormal

C_REL = 1e-12  # relative offset c_i = C_REL·ψ_i (D-2)


def constant_prior(scales: WindowScales, n: int) -> ConstantPrior:
    """``λ_i ~ IG(1 + ι/2, ψ_i/2)``: shape ``(d − n + ι)/2``, scale ``ψ_i/2`` (``d = n + 2``)."""
    _check_scales(scales, n)
    return ConstantPrior(shape=1.0 + scales.iota / 2.0, scale=scales.psi / 2.0)


def sv_prior_arrays(sv: SV, scales: WindowScales) -> SVPrior:
    """``SVPrior`` arrays of ``sv`` for the window (DESIGN §2.6, §6.3).

    ``mu_prior="data"``: ``b_μ = log(ψ_i/ι)``, ``B_μ = mu_var``; a ``Normal(mean, var)`` gives
    the same values for every equation (stochvol parity). ``sigma2_prior``: ``ScaledChi2(B)`` →
    ``B_σ = B``; ``InvGamma(c₀, C₀)`` → ``(c0, C0)``. ``h0=B₀`` → ``B0``. RW: ``m0 = log(ψ_i/ι)``,
    ``V0 = rw_h0_var``. ``log_c = log(1e-12·ψ_i)`` (``offset="relative"``), ``log c`` for an
    absolute offset, ``−inf`` for ``offset=0.0``.
    """
    if not isinstance(sv, SV):
        raise TypeError(f"sv must be an SV specification, got {type(sv).__name__}")
    if not isinstance(scales, WindowScales):
        raise TypeError(f"scales must be a WindowScales, got {type(scales).__name__}")
    n = scales.n
    zeros, ones = np.zeros(n), np.ones(n)
    level = np.log(scales.sigma2_bar)  # log(ψ_i/ι): the Mode I variance scale (D-13)
    if sv.rw:  # h_0 ~ N(m0, V0); no μ, φ (N02 §8.1)
        b_mu, B_mu, a_phi, b_phi = zeros, ones, ones, ones
        m0, V0 = level, np.full(n, sv.rw_h0_var)
    else:
        if sv.mu_prior == "data":
            b_mu, B_mu = level, np.full(n, sv.mu_var)
        else:
            b_mu, B_mu = np.full(n, sv.mu_prior.mean), np.full(n, sv.mu_prior.var)
        a_phi, b_phi = np.full(n, sv.phi_prior.a), np.full(n, sv.phi_prior.b)
        m0, V0 = zeros, ones
    sigma2 = sv.resolved_sigma2_prior()
    if isinstance(sigma2, ScaledChi2):
        B_sigma, c0, C0 = np.full(n, sigma2.B), ones, ones
    elif isinstance(sigma2, InvGamma):
        B_sigma, c0, C0 = ones, np.full(n, sigma2.shape), np.full(n, sigma2.scale)
    else:  # pragma: no cover - SV validates its sigma2_prior
        raise TypeError(f"unsupported sigma2 prior {sigma2!r}")
    B0 = np.full(n, sv.h0) if not sv.rw and sv.h0 != "stationary" else ones
    if sv.offset == "relative":
        log_c = np.log(C_REL) + np.log(scales.psi)
    elif sv.offset == 0.0:
        log_c = np.full(n, -np.inf)
    else:
        log_c = np.full(n, np.log(sv.offset))
    return SVPrior(
        b_mu=_f64(b_mu), B_mu=_f64(B_mu), a_phi=_f64(a_phi), b_phi=_f64(b_phi),
        B_sigma=_f64(B_sigma), c0=_f64(c0), C0=_f64(C0), B0=_f64(B0), m0=_f64(m0), V0=_f64(V0),
        log_c=_f64(log_c),
    )  # fmt: skip


def outlier_beta(sv: SV, freq: str) -> Beta:
    """Prior of the outlier probability ``p`` (D-9; N03 §3.1): an explicit ``Beta``, or
    ``OutlierPrior(every_years, prior_years)`` resolved with ``f`` periods per year to
    ``Beta(a, f·prior_years − a)``, ``a = prior_years/every_years``. Defaults: SVO ``Beta(2.5,
    10f − 2.5)``, SVO-t ``Beta(1, 10f − 1)``. Frequencies other than quarterly and monthly raise
    an actionable ``ValueError`` unless the ``Beta`` is explicit."""
    prior = sv.resolved_outlier_prior()
    if prior is None:
        raise ValueError("this SV specification has no outlier component (outliers=False)")
    if isinstance(prior, OutlierPrior):
        return prior.resolve(freq)
    return prior


def transitory_arrays(
    sv: SV, scales: WindowScales, nu_grid: tuple[int, ...]
) -> TransitoryPrior | Empty:
    """``TransitoryPrior`` of ``sv`` on the static ν grid, or ``Empty()`` without outliers and t.

    ``(a_p, b_p)`` from :func:`outlier_beta` (placeholders 1 without outliers); ``log_nu_w``
    (n, G) the log prior weights of ``ν = nu_grid[g]``: ``−log |support|`` on the support of
    ``nu_prior`` and ``−inf`` elsewhere (every grid point ``−log G`` without t). The support must
    lie on the grid. None of these depends on the data values (R9), only on the frequency.
    """
    if not isinstance(sv, SV):
        raise TypeError(f"sv must be an SV specification, got {type(sv).__name__}")
    if not isinstance(scales, WindowScales):
        raise TypeError(f"scales must be a WindowScales, got {type(scales).__name__}")
    has_t = sv.tails == "t"
    if not sv.outliers and not has_t:
        return Empty()
    grid = tuple(int(v) for v in nu_grid)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError(f"nu_grid must be a non-empty tuple of distinct integers, got {nu_grid!r}")
    n, G = scales.n, len(grid)
    if sv.outliers:
        beta = outlier_beta(sv, scales.freq)
        a_p, b_p = np.full(n, beta.a), np.full(n, beta.b)
    else:
        a_p, b_p = np.ones(n), np.ones(n)
    if has_t:
        support = sv.nu_prior.support
        missing = sorted(set(support) - set(grid))
        if missing:
            raise ValueError(
                f"nu_prior puts mass on ν = {missing}, which the static grid {grid} lacks"
            )
        on = np.isin(np.array(grid), np.array(support))
        row = np.where(on, -np.log(len(support)), -np.inf)
    else:
        row = np.full(G, -np.log(G))
    log_nu_w = _f64(np.broadcast_to(row, (n, G)))
    return TransitoryPrior(a_p=_f64(a_p), b_p=_f64(b_p), log_nu_w=log_nu_w)


def common_sv_prior(vol: CommonSV, n: int) -> CommonSVPrior:
    """``CommonSVPrior`` of ``vol`` for ``n`` equations (P3 §2.3, §5.1, D-46; NumPy scalars).

    ``rho_prior``: a ``TruncatedNormal(mean, var)`` gives ``rho_mean``, ``rho_var`` (``var`` is
    the variance before truncation to ``(−1, 1)``); a ``Beta(a, b)`` on ``(ρ + 1)/2`` gives
    ``rho_a``, ``rho_b``. ``sigma2_prior``: ``InvGamma(shape, scale)`` gives ``s2_shape``,
    ``s2_scale``; ``ScaledChi2(B)`` (``σ_h² ~ B·χ²₁``) gives ``s2_B``. The fields of the option
    not chosen hold inert placeholders (0 for a location, 1 otherwise); ``n_eq = n``."""
    if not isinstance(vol, CommonSV):
        raise TypeError(f"vol must be a CommonSV specification, got {type(vol).__name__}")
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rho, s2 = vol.rho_prior, vol.sigma2_prior
    if isinstance(rho, TruncatedNormal):
        rho_mean, rho_var, rho_a, rho_b = rho.mean, rho.var, 1.0, 1.0
    elif isinstance(rho, Beta):
        rho_mean, rho_var, rho_a, rho_b = 0.0, 1.0, rho.a, rho.b
    else:  # pragma: no cover - CommonSV validates its rho_prior
        raise TypeError(f"unsupported rho prior {rho!r}")
    if isinstance(s2, InvGamma):
        s2_shape, s2_scale, s2_B = s2.shape, s2.scale, 1.0
    elif isinstance(s2, ScaledChi2):
        s2_shape, s2_scale, s2_B = 1.0, 1.0, s2.B
    else:  # pragma: no cover - CommonSV validates its sigma2_prior
        raise TypeError(f"unsupported sigma2 prior {s2!r}")
    values = (rho_mean, rho_var, rho_a, rho_b, s2_shape, s2_scale, s2_B, float(n))
    return CommonSVPrior(*(np.array(v, dtype=np.float64) for v in values))


def _f64(x: np.ndarray) -> np.ndarray:
    """A fresh C-contiguous float64 copy: leaves never share a buffer (placeholders reuse one)."""
    return np.array(x, dtype=np.float64, order="C", copy=True)


__all__ = [
    "C_REL",
    "common_sv_prior",
    "constant_prior",
    "outlier_beta",
    "sv_prior_arrays",
    "transitory_arrays",
]
