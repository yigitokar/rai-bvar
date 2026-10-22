"""Clean-room kernels and helpers shared by the Phase 2 pilot scripts (WP3; DESIGN §10.6).

Standalone prototypes for timing and acceptance measurements only: nothing in ``src/rai_bvar``
imports this directory, and the production kernels are written separately by WP4/WP10/WP12/WP13.
Every algorithm follows the cross-checked notes (no GPL source was consulted):

* OCSN-10 / KSC-7 tables and log-space indicator weights: N01 §5.3, §8.2 (D-3);
* inverse-CDF categorical with one uniform per row and a ``≤`` comparison: N07 §9.2 (D-4);
* tridiagonal precision sampler as two ``lax.scan`` passes: N07 §8.3, N02 §5.5;
* SV core in non-centred AWOL form, exact mixture correction, 2-block centred step with centred
  cross-products, NC ``φ`` and ``(μ, σ)`` steps, measure-preserving fold: DESIGN §3.7.4;
  N02 §5.3, §6.1, §7; N01 §10.2–10.3 (D-1, D-2, D-5, D-6, D-7);
* collapsed Student-t ``ν``-grid step and ``τ`` draw: N03 §4.3a, §4.4 (D-8);
* outlier-scale logits: N03 §4.1 (D-9);
* structural mean draw with the packed Khatri–Rao Gram and the Jacobi-scaled two-solve precision
  draw: DESIGN §3.7.2 (D-14, D-25).

Scripts in this directory run with ``python -I``, which does not put the script directory on
``sys.path``; each script therefore inserts it explicitly before importing this module. No
module-level ``jnp`` array is created here (N07 §11), so ``setup_jax(devices=k)`` still works
after import.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

# --------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------


def setup_jax(devices: int = 1) -> Any:
    """Enable float64 and (optionally) ``devices`` CPU devices; must run before any JAX op."""
    import jax

    if devices > 1:
        jax.config.update("jax_num_cpu_devices", devices)
    jax.config.update("jax_enable_x64", True)
    return jax


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def fingerprint() -> dict[str, Any]:
    """Machine fingerprint, load average and library versions (recorded with every result)."""
    import jax
    import jaxlib
    import scipy

    return {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "perf_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "eff_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "mem_bytes": _sysctl("hw.memsize"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "devices": jax.local_device_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "loadavg": os.getloadavg(),
    }


def save_result(out_dir: str | os.PathLike, name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` (plus fingerprint) as ``<out_dir>/<name>.json`` and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    payload = {"name": name, "fingerprint": fingerprint(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def block(x: Any) -> Any:
    import jax

    return jax.block_until_ready(x)


def time_call(fn: Callable[[], Any], reps: int = 5, warmup: int = 1) -> dict[str, float]:
    """Minimum and median wall time (seconds) of ``fn()`` with ``block_until_ready``."""
    for _ in range(warmup):
        block(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        block(fn())
        ts.append(time.perf_counter() - t0)
    return {"min": float(np.min(ts)), "median": float(np.median(ts)), "reps": reps}


def aot(fn: Callable, *args: Any, **jit_kw: Any) -> tuple[Any, float]:
    """``jax.jit(fn).lower(*args).compile()`` and its wall time (compile separated from run)."""
    import jax

    t0 = time.perf_counter()
    compiled = jax.jit(fn, **jit_kw).lower(*args).compile()
    return compiled, time.perf_counter() - t0


# --------------------------------------------------------------------------------------------
# Key derivation (DESIGN §5.6; enum values copied from the frozen design)
# --------------------------------------------------------------------------------------------

STREAM_MCMC, STREAM_INIT = 0, 1
BLOCK_EQ_MEAN, BLOCK_EQ_VOL = 0, 2
SUB_S, SUB_H, SUB_H_ACC = 0, 1, 2
SUB_C_GP, SUB_C_GP_ACC, SUB_C_S2, SUB_C_S2_ACC = 3, 4, 5, 6
SUB_NC_PHI, SUB_NC_PHI_ACC, SUB_NC_MS, SUB_NC_MS_ACC = 7, 8, 9, 10
SUB_O, SUB_P, SUB_NU, SUB_TAU = 11, 12, 13, 14
SUB_Z = 0


def fold(key: Any, data: Any) -> Any:
    import jax.numpy as jnp
    from jax import random

    return random.fold_in(key, jnp.asarray(data).astype(jnp.uint32))


def chain_keys(seed: int, C: int, impl: str = "threefry2x32", stream: int = STREAM_MCMC) -> Any:
    """``fold_in(fold_in(key(seed), stream), c)`` for ``c = 0..C−1`` (data salt omitted here)."""
    import jax
    import jax.numpy as jnp
    from jax import random

    root = random.fold_in(random.key(seed, impl=impl), stream)
    return jax.vmap(lambda c: random.fold_in(root, c))(jnp.arange(C, dtype=jnp.uint32))


def uniform_tiny(key: Any, shape: tuple[int, ...] = ()) -> Any:
    import jax.numpy as jnp
    from jax import random

    return random.uniform(key, shape, dtype=jnp.float64, minval=np.finfo(np.float64).tiny)


# --------------------------------------------------------------------------------------------
# Mixture tables (N01 §5.2–5.3; N02 §2.3): weights, means, variances of log χ²₁ components
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureTable:
    name: str
    p: tuple[float, ...]
    m: tuple[float, ...]
    v: tuple[float, ...]

    @property
    def J(self) -> int:
        return len(self.p)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(m, v, logw, half_inv_v)`` with ``logw = log p − ½ log v`` (NumPy, float64)."""
        p, m, v = (np.asarray(a, dtype=np.float64) for a in (self.p, self.m, self.v))
        return m, v, np.log(p) - 0.5 * np.log(v), 0.5 / v


OCSN10 = MixtureTable(
    "ocsn10",
    (0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115),
    (1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65),
    (0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342),
)
KSC7 = MixtureTable(
    "ksc7",
    (0.04395, 0.24566, 0.34001, 0.25750, 0.10556, 0.00002, 0.00730),
    (1.50746, 0.52478, -0.65098, -2.35859, -5.24321, -9.83726, -11.40039),
    (0.16735, 0.34023, 0.64009, 1.26261, 2.61369, 5.17950, 5.79596),
)
TABLES = {"ocsn10": OCSN10, "ksc7": KSC7}
LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------------------------
# Generic JAX kernels
# --------------------------------------------------------------------------------------------


def categorical_inverse_cdf(u: Any, logw: Any) -> tuple[Any, Any]:
    """Inverse-CDF draw from unnormalised log weights (D-4; N07 §9.2) plus ``logsumexp``.

    ``w = exp(logw − max)``, ``s = min(Σ_j 1{cw_j ≤ u·cw_J}, J−1)``; the row ``logsumexp`` is
    ``max + log cw_J`` and is returned so the exact correction can reuse it (DESIGN §3.7.4).
    """
    import jax.numpy as jnp

    J = logw.shape[-1]
    mx = jnp.max(logw, axis=-1, keepdims=True)
    cw = jnp.cumsum(jnp.exp(logw - mx), axis=-1)
    tot = cw[..., -1]
    s = jnp.minimum(jnp.sum(cw <= (u * tot)[..., None], axis=-1), J - 1).astype(jnp.int32)
    return s, mx[..., 0] + jnp.log(tot)


def tridiag_precision_sample(diag: Any, off: Any, b: Any, z: Any, unroll: int = 1) -> tuple:
    """``x ~ N(Q⁻¹b, Q⁻¹)`` for tridiagonal SPD ``Q`` (N07 §8.3; N02 §5.5).

    ``diag`` (m,), ``off`` (m−1,) with ``off[t] = Q[t+1, t]``. Forward scan: bidiagonal Cholesky
    fused with ``Lv = b``; reverse scan: ``Lᵀx = v + z``. Returns ``(x, log|Q|, min pivot)``.
    """
    import jax.numpy as jnp
    from jax import lax

    l0 = jnp.sqrt(diag[0])
    v0 = b[0] / l0

    def fwd(carry, inp):
        l_prev, v_prev = carry
        d_t, e_t, b_t = inp
        c_t = e_t / l_prev
        piv = d_t - c_t * c_t
        l_t = jnp.sqrt(piv)
        v_t = (b_t - c_t * v_prev) / l_t
        return (l_t, v_t), (l_t, c_t, v_t)

    _, (ls, cs, vs) = lax.scan(fwd, (l0, v0), (diag[1:], off, b[1:]), unroll=unroll)
    ell = jnp.concatenate([l0[None], ls])
    c = jnp.concatenate([jnp.zeros((1,), diag.dtype), cs])
    w = jnp.concatenate([v0[None], vs]) + z
    x_last = w[-1] / ell[-1]

    def bwd(x_next, inp):
        w_t, l_t, c_next = inp
        x_t = (w_t - c_next * x_next) / l_t
        return x_t, x_t

    _, xs = lax.scan(bwd, x_last, (w[:-1], ell[:-1], c[1:]), reverse=True, unroll=unroll)
    x = jnp.concatenate([xs, x_last[None]])
    return x, 2.0 * jnp.sum(jnp.log(ell)), jnp.min(ell * ell)


# --------------------------------------------------------------------------------------------
# SV core (one series; batched by vmap). DESIGN §3.7.4
# --------------------------------------------------------------------------------------------


class SVPrior(NamedTuple):
    b_mu: Any  # μ ~ N(b_μ, B_μ)
    B_mu: Any
    a_phi: Any  # (φ+1)/2 ~ Beta(a_φ, b_φ)
    b_phi: Any
    B_sigma: Any  # σ² ~ B_σ·χ²₁
    log_c: Any  # log offset c_i = c_rel·ψ_i (D-2)


class SVState(NamedTuple):
    h: Any  # (T_pad+1,): index 0 = h_0, index t+1 = data row t
    mu: Any
    phi: Any
    sigma: Any
    acc_h: Any
    acc_cgp: Any
    acc_cs2: Any
    acc_ncphi: Any
    acc_ncms: Any
    acc_fold: Any
    rej_run: Any
    rej_run_max: Any
    rb_h_sum: Any
    rb_h_sq: Any


def sv_state_init(h0: Any, mu: Any, phi: Any, sigma: Any) -> SVState:
    import jax.numpy as jnp

    z = jnp.zeros_like(mu)
    zi = jnp.zeros(jnp.shape(mu), jnp.int32)
    return SVState(
        h0, mu, phi, sigma, z, z, z, z, z, z, zi, zi, jnp.zeros_like(h0), jnp.zeros_like(h0)
    )


_ACC_FIELDS = (
    "acc_h", "acc_cgp", "acc_cs2", "acc_ncphi", "acc_ncms", "acc_fold",
    "rej_run", "rej_run_max", "rb_h_sum", "rb_h_sq",
)  # fmt: skip


def zero_acc(st: SVState) -> SVState:
    """Every ``acc_*``/``rb_*`` leaf to 0 (burn-in → sampling boundary, DESIGN §5.1).

    Each leaf gets its own buffer, so the result can be donated to an executable.
    """
    import jax.numpy as jnp

    return st._replace(**{f: jnp.zeros_like(getattr(st, f)) for f in _ACC_FIELDS})


def _log_g(r: Any, tab: tuple) -> Any:
    """``log g(r) = logsumexp_j(logw_j − (r − m_j)²/(2v_j)) − ½ log 2π`` (mixture density)."""
    import jax
    import jax.numpy as jnp

    m, _, logw, hiv = (jnp.asarray(a) for a in tab)
    return jax.nn.logsumexp(logw - (r[:, None] - m) ** 2 * hiv, axis=-1) - 0.5 * LOG_2PI


def _w_weight(hd: Any, log_e2c: Any, obs: Any, log_g: Any) -> Any:
    """``w(h) = Σ_t m_t[−½(log 2π + h_{t+1} + ẽ_t² e^{−h_{t+1}}) − log g(y*_t − h_{t+1})]``."""
    import jax.numpy as jnp

    val = -0.5 * (LOG_2PI + hd + jnp.exp(log_e2c - hd)) - log_g
    return jnp.sum(jnp.where(obs, val, 0.0))


def _log_p_phi(phi: Any, a: Any, b: Any) -> Any:
    import jax.numpy as jnp

    return (a - 1.0) * jnp.log1p(phi) + (b - 1.0) * jnp.log1p(-phi)


_B011, _B022 = 1e12, 1e8  # KFS auxiliary prior variances of (γ, φ) (D-6; N02 §6)


def _cgp_log_target(gam: Any, phi: Any, h0: Any, s2: Any, pr: SVPrior) -> Any:
    """Centred (γ, φ) MH ratio pieces (N02 §6.1): ℓ₀ + log p(γ|φ) + log p_φ − log p_aux."""
    import jax.numpy as jnp

    mu = gam / (1.0 - phi)
    om = 1.0 - phi * phi
    l0 = 0.5 * jnp.log(om / s2) - 0.5 * om * (h0 - mu) ** 2 / s2
    one_m = 1.0 - phi
    lgam = -jnp.log(jnp.abs(one_m)) - (gam - pr.b_mu * one_m) ** 2 / (2.0 * pr.B_mu * one_m**2)
    aux = (gam * gam / _B011 + phi * phi / _B022) / (2.0 * s2)
    return l0 + lgam + _log_p_phi(phi, pr.a_phi, pr.b_phi) + aux


def sv_step(
    key: Any,
    st: SVState,
    log_e2c: Any,
    tmask: Any,
    T_eff: Any,
    pr: SVPrior,
    *,
    table: MixtureTable = OCSN10,
    exact: bool = True,
    cstep: bool = True,
    asis: bool = True,
    unroll: int = 1,
) -> tuple[SVState, Any]:
    """One SV-core update (c) s → (a) h → (b) centred θ → (b**) NC θ (DESIGN §3.7.4).

    Stationary AR ``h_0``, ``σ² ~ B_σχ²₁``. ``log_e2c`` (T_pad,) holds ``log ẽ²`` on real rows and
    0 on padded rows. Returns the new state and the minimum tridiagonal pivot.
    """
    import jax.numpy as jnp
    from jax import random

    tab = table.arrays()
    t_m, t_v, t_logw, t_hiv = (jnp.asarray(a) for a in tab)
    m = tmask
    obs = m > 0
    T_pad = log_e2c.shape[0]
    fT = jnp.sum(m)
    ystar = jnp.where(obs, jnp.logaddexp(log_e2c, pr.log_c), 0.0)  # D-2
    h, mu, phi, sigma = st.h, st.mu, st.phi, st.sigma

    # (c) indicators | y*, h  (N01 §8.2; D-4)
    r = ystar - h[1:]
    logits = t_logw - (r[:, None] - t_m) ** 2 * t_hiv
    s, lse = categorical_inverse_cdf(random.uniform(fold(key, SUB_S), (T_pad,)), logits)
    log_g_cur = lse - 0.5 * LOG_2PI
    ms_, vs_ = t_m[s], t_v[s]
    D = m / vs_

    # (a) h | y*, s, θ in non-centred AWOL form (N02 §5.3; DESIGN §3.7.4)
    idx = jnp.arange(1, T_pad + 1)
    s2 = sigma * sigma
    diag_rows = jnp.where(
        idx < T_eff, s2 * D + 1.0 + phi * phi, jnp.where(idx == T_eff, s2 * D + 1.0, 1.0)
    )
    diag = jnp.concatenate([jnp.ones((1,)), diag_rows])  # stationary: ω₀ + φ² = 1
    off = jnp.where(idx <= T_eff, -phi, 0.0)
    btil = jnp.concatenate([jnp.zeros((1,)), sigma * D * (ystar - ms_ - mu)])
    x, _, min_piv = tridiag_precision_sample(
        diag, off, btil, random.normal(fold(key, SUB_H), (T_pad + 1,)), unroll
    )
    h_prop = mu + sigma * x
    if exact:  # DNP Alg. 3 in the Gaussian-ẽ form (N01 §10.2; D-1)
        w_cur = _w_weight(h[1:], log_e2c, obs, log_g_cur)
        w_prop = _w_weight(h_prop[1:], log_e2c, obs, _log_g(ystar - h_prop[1:], tab))
        acc_h = jnp.log(uniform_tiny(fold(key, SUB_H_ACC))) < w_prop - w_cur
        h = jnp.where(acc_h, h_prop, h)
        w_h = jnp.where(acc_h, w_prop, w_cur)
    else:
        acc_h = jnp.asarray(True)
        h = h_prop
        w_h = jnp.zeros(())
    acc_cgp = acc_cs2 = acc_ncphi = acc_ncms = fold_ = jnp.asarray(False)

    # (b) centred (γ, φ) then σ² (2-block independence MH; N02 §6.1; D-6)
    if cstep:
        hx, hz = h[:-1], h[1:]
        xbar = jnp.sum(m * hx) / fT
        zbar = jnp.sum(m * hz) / fT
        dx, dz = hx - xbar, hz - zbar
        Sc = jnp.sum(m * dx * dx)
        Sxz = jnp.sum(m * dx * dz)
        e1, e2 = 1.0 / _B011, 1.0 / _B022
        A12 = fT * xbar
        A22 = Sc + fT * xbar * xbar + e2
        det = fT * Sc + fT * e2 + e1 * A22
        phi_hat = (fT * Sxz + e1 * (Sxz + fT * xbar * zbar)) / det
        gam_hat = (fT * zbar * (Sc + e2) - fT * xbar * Sxz) / det
        L11 = jnp.sqrt(A22 / det)
        L21 = -A12 / jnp.sqrt(det * A22)
        L22 = 1.0 / jnp.sqrt(A22)
        zz = random.normal(fold(key, SUB_C_GP), (2,))
        gam_p = gam_hat + sigma * L11 * zz[0]
        phi_p = phi_hat + sigma * (L21 * zz[0] + L22 * zz[1])
        valid = jnp.abs(phi_p) < 1.0
        phi_ps = jnp.where(valid, phi_p, 0.0)
        log_r = _cgp_log_target(gam_p, phi_ps, h[0], s2, pr) - _cgp_log_target(
            mu * (1.0 - phi), phi, h[0], s2, pr
        )
        log_r = jnp.where(valid, log_r, -jnp.inf)
        acc_cgp = jnp.log(uniform_tiny(fold(key, SUB_C_GP_ACC))) < log_r
        mu = jnp.where(acc_cgp, gam_p / (1.0 - phi_ps), mu)
        phi = jnp.where(acc_cgp, phi_ps, phi)
        res = hz - mu - phi * (hx - mu)
        CT = 0.5 * (jnp.sum(m * res * res) + (1.0 - phi * phi) * (h[0] - mu) ** 2)
        log_s2p = jnp.log(CT) - random.loggamma(fold(key, SUB_C_S2), 0.5 * fT)
        s2p = jnp.exp(log_s2p)
        acc_cs2 = jnp.log(uniform_tiny(fold(key, SUB_C_S2_ACC))) < (s2 - s2p) / (2.0 * pr.B_sigma)
        sigma = jnp.where(acc_cs2, jnp.sqrt(s2p), sigma)

    # (b**) non-centred φ | h̃, then (μ, σ) | h̃, s, y* with the exact w-correction (N02 §7; N01 §10.3)
    if asis:
        ht = (h - mu) / sigma
        hx, hz = ht[:-1], ht[1:]
        sxx = jnp.sum(m * hx * hx)
        sxy = jnp.sum(m * hx * hz)
        phi_p = sxy / sxx + random.normal(fold(key, SUB_NC_PHI)) / jnp.sqrt(sxx)
        valid = jnp.abs(phi_p) < 1.0
        phi_ps = jnp.where(valid, phi_p, 0.0)

        def lt(ph):
            om = 1.0 - ph * ph
            return 0.5 * jnp.log(om) - 0.5 * om * ht[0] ** 2 + _log_p_phi(ph, pr.a_phi, pr.b_phi)

        log_r = jnp.where(valid, lt(phi_ps) - lt(phi), -jnp.inf)
        acc_ncphi = jnp.log(uniform_tiny(fold(key, SUB_NC_PHI_ACC))) < log_r
        phi = jnp.where(acc_ncphi, phi_ps, phi)
        wts = m / vs_
        yt = ystar - ms_
        P11 = 1.0 / pr.B_mu + jnp.sum(wts)
        P12 = jnp.sum(wts * hz)
        P22 = 1.0 / pr.B_sigma + jnp.sum(wts * hz * hz)
        r1 = pr.b_mu / pr.B_mu + jnp.sum(wts * yt)
        r2 = jnp.sum(wts * hz * yt)
        L11 = jnp.sqrt(P11)
        L21 = P12 / L11
        L22 = jnp.sqrt(P22 - L21 * L21)
        v1 = r1 / L11
        v2 = (r2 - L21 * v1) / L22
        zz = random.normal(fold(key, SUB_NC_MS), (2,))
        sig_p = (v2 + zz[1]) / L22
        mu_p = (v1 + zz[0] - L21 * sig_p) / L11
        h_nc = mu_p + sig_p * ht
        if exact:
            w_nc = _w_weight(h_nc[1:], log_e2c, obs, _log_g(ystar - h_nc[1:], tab))
            acc_ncms = jnp.log(uniform_tiny(fold(key, SUB_NC_MS_ACC))) < w_nc - w_h
        else:
            acc_ncms = jnp.asarray(True)
        fold_ = acc_ncms & (sig_p < 0.0)  # (σ, h̃) ↦ (−σ, −h̃) leaves h unchanged (D-7)
        mu = jnp.where(acc_ncms, mu_p, mu)
        sigma = jnp.where(acc_ncms, jnp.abs(sig_p), sigma)
        h = jnp.where(acc_ncms, h_nc, h)

    f = jnp.float64
    rej = (st.rej_run + 1) * (1 - acc_h.astype(jnp.int32))
    new = SVState(
        h,
        mu,
        phi,
        sigma,
        st.acc_h + acc_h.astype(f),
        st.acc_cgp + acc_cgp.astype(f),
        st.acc_cs2 + acc_cs2.astype(f),
        st.acc_ncphi + acc_ncphi.astype(f),
        st.acc_ncms + acc_ncms.astype(f),
        st.acc_fold + fold_.astype(f),
        rej,
        jnp.maximum(st.rej_run_max, rej),
        st.rb_h_sum + h,
        st.rb_h_sq + h * h,
    )
    return new, min_piv


def sv_loglik(h: Any, log_e2: Any, tmask: Any) -> Any:
    """Gaussian log-likelihood ``Σ_t m_t log N(e_t; 0, e^{h_{t+1}})`` (monitor)."""
    import jax.numpy as jnp

    hd = h[1:]
    return jnp.sum(jnp.where(tmask > 0, -0.5 * (LOG_2PI + hd + jnp.exp(log_e2 - hd)), 0.0))


def masked_log_e2(e: Any, tmask: Any) -> Any:
    """``log e²`` on real rows, 0 on padded rows; the mask is applied before the log (E23)."""
    import jax.numpy as jnp

    obs = tmask > 0
    return jnp.where(obs, jnp.log(jnp.where(obs, e * e, 1.0)), 0.0)


# --------------------------------------------------------------------------------------------
# Transitory scales (N03 §4.1–4.4)
# --------------------------------------------------------------------------------------------

NU_GRID = np.arange(3, 41, dtype=np.float64)  # G = 38 (D-8)


def nu_constants(t_scale: str = "standardized") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ν_g, log c(ν_g), A_g)`` with ``A_g = lnΓ((ν+1)/2) − lnΓ(ν/2) − ½ log(π c)`` (NumPy)."""
    from scipy.special import gammaln

    nu = NU_GRID
    c = nu - 2.0 if t_scale == "standardized" else nu
    A = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * c)
    return nu, np.log(c), A


def nu_grid_sums(log_r2: Any, obs: Any, log_c: Any, form: str) -> Any:
    """``S_g = Σ_t m_t log1p(r_t²/c_g)`` for every grid point (the O(T·G) part of N03 §4.3a).

    ``form``: ``"logaddexp"`` = ``logaddexp(log r², log c) − log c`` (two transcendentals,
    overflow-free); ``"log1p_exp"`` = ``log1p(exp(log r² − log c))``; ``"log_sum"`` =
    ``log(c + r²) − log c`` with ``r² = exp(log r²)`` computed once per row (one log per
    element). Rows with ``log r² > 700`` use ``log r²`` itself, which equals ``log(c + r²)`` in
    float64 because ``c/r² < 1e-300`` (no overflow, no approximation beyond rounding).
    """
    import jax.numpy as jnp

    if form == "logaddexp":
        val = jnp.logaddexp(log_r2[:, None], log_c[None, :]) - log_c[None, :]
    elif form == "log1p_exp":
        val = jnp.log1p(jnp.exp(log_r2[:, None] - log_c[None, :]))
    elif form == "log_sum":
        big = log_r2 > 700.0
        r2 = jnp.exp(jnp.where(big, 0.0, log_r2))
        lsum = jnp.log(jnp.exp(log_c)[None, :] + r2[:, None])
        val = jnp.where(big[:, None], log_r2[:, None], lsum) - log_c[None, :]
    else:
        raise ValueError(form)
    return jnp.sum(jnp.where(obs[:, None], val, 0.0), axis=0)


def loggamma_mt(key: Any, a: Any, shape: tuple[int, ...], M: int = 3) -> Any:
    """``log G``, ``G ~ Gamma(a, 1)``, ``a ≥ 1``: Marsaglia & Tsang (2000) with ``M`` pre-drawn
    candidates per element and a ``while_loop`` fallback for elements that rejected all ``M``.

    Exact (taking the first accepted candidate of an i.i.d. sequence is the rejection sampler);
    with ``d = a − 1/3``, ``c = 1/√(9d)``, ``v = (1 + cx)³``: accept iff
    ``v > 0`` and ``log u < x²/2 + d − dv + d log v``; ``log G = log d + log v``. Avoids the
    per-element loop of ``random.gamma`` under ``vmap`` (pilot finding, see 09_pilot.md §5).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax, random

    a = jnp.asarray(a, jnp.float64)
    d = a - 1.0 / 3.0
    c = 1.0 / jnp.sqrt(9.0 * d)

    def cand(k):
        x = random.normal(fold(k, 0), shape)
        u = uniform_tiny(fold(k, 1), shape)
        t = 1.0 + c * x
        pos = t > 0.0
        logv = 3.0 * jnp.log(jnp.where(pos, t, 1.0))
        ok = pos & (jnp.log(u) < 0.5 * x * x + d - d * jnp.exp(logv) + d * logv)
        return ok, logv

    oks, logvs = jax.vmap(cand)(jax.vmap(lambda m: fold(key, m))(jnp.arange(M, dtype=jnp.uint32)))
    first = jnp.argmax(oks, axis=0)
    done = jnp.any(oks, axis=0)
    logv = jnp.take_along_axis(logvs, first[None], axis=0)[0]

    def cond(s):
        return ~jnp.all(s[1])

    def body(s):
        it, dn, lv = s
        ok, lv_new = cand(fold(key, M + it))
        take = ~dn & ok
        return it + 1, dn | ok, jnp.where(take, lv_new, lv)

    _, _, logv = lax.while_loop(cond, body, (jnp.uint32(0), done, logv))
    return jnp.log(d) + logv


def t_step(
    key: Any,
    log_z2: Any,
    tmask: Any,
    T_eff: Any,
    log_nu_w: Any,
    *,
    form: str = "logaddexp",
    t_scale: str = "standardized",
    tau_sampler: str = "mt3",
) -> tuple[Any, Any]:
    """Collapsed ``ν | e, h`` on the grid, then ``τ | e, h, ν`` (N03 §4.3a, §4.4; D-8).

    ``log_z2`` = ``log e² − log λ^{pers}`` on real rows. Returns ``(ν index, log τ)`` with
    ``log τ = 0`` on padded rows. ``tau_sampler``: ``"jax"`` (``random.loggamma``) or
    ``"mt<M>"`` (:func:`loggamma_mt` with ``M`` candidates).
    """
    import jax.numpy as jnp
    from jax import random

    nu_np, log_c_np, A_np = nu_constants(t_scale)
    nu, log_c, A = jnp.asarray(nu_np), jnp.asarray(log_c_np), jnp.asarray(A_np)
    obs = tmask > 0
    S = nu_grid_sums(log_z2, obs, log_c, form)
    logits = log_nu_w + T_eff * A - 0.5 * (nu + 1.0) * S
    g, _ = categorical_inverse_cdf(random.uniform(fold(key, SUB_NU)), logits)
    nu_s, lc = nu[g], log_c[g]
    shape_tau = 0.5 * (nu_s + 1.0)
    if tau_sampler == "jax":
        lg = random.loggamma(fold(key, SUB_TAU), shape_tau, log_z2.shape)
    else:
        lg = loggamma_mt(fold(key, SUB_TAU), shape_tau, log_z2.shape, M=int(tau_sampler[2:]))
    log_tau = jnp.logaddexp(lc, log_z2) - np.log(2.0) - lg  # IG((ν+1)/2, (c(ν) + z²)/2)
    return g, jnp.where(obs, log_tau, 0.0)


def t_step_uncollapsed(key: Any, log_tau: Any, tmask: Any, T_eff: Any, log_nu_w: Any) -> Any:
    """Uncollapsed ``ν | τ`` on the grid (standardized; N03 §4.3b derived form): O(T + G)."""
    import jax.numpy as jnp
    from jax import random
    from jax.scipy.special import gammaln

    nu = jnp.asarray(NU_GRID)
    obs = tmask > 0
    sl = jnp.sum(jnp.where(obs, log_tau, 0.0))
    si = jnp.sum(jnp.where(obs, jnp.exp(-log_tau), 0.0))
    ll = T_eff * (0.5 * nu * jnp.log(0.5 * (nu - 2.0)) - gammaln(0.5 * nu))
    logits = log_nu_w + ll - 0.5 * nu * sl - 0.5 * (nu - 2.0) * si
    g, _ = categorical_inverse_cdf(random.uniform(fold(key, SUB_NU)), logits)
    return g


def outlier_step(
    key: Any, log_z2: Any, tmask: Any, log_p: Any, log1m_p: Any, K: int = 20, fast: bool = False
):
    """``o | e, h, p`` on ``{1..K}`` then ``p | o ~ Beta(2.5 + N₁, 37.5 + T − N₁)`` (N03 §4.1–4.2;
    D-9; the quarterly SVO default prior). Returns ``(o index, log p)``.

    ``fast``: the quadratic term ``exp(log z² − 2 log k)/2`` is computed as ``z²·(½k⁻²)`` with
    ``z² = exp(min(log z², 700))`` once per row (T instead of T·K exponentials; equal up to
    rounding). The clamp cannot change a draw: at ``log z² ≥ 700`` adjacent logits differ by
    more than 1e290, so ``o = K`` with probability one in both forms.
    """
    import jax.numpy as jnp
    from jax import random

    k = jnp.arange(1, K + 1, dtype=jnp.float64)
    logk = jnp.log(k)
    log_w = jnp.concatenate([log1m_p[None], jnp.full((K - 1,), log_p - np.log(K - 1.0))])
    obs = tmask > 0
    if fast:
        z2 = jnp.exp(jnp.minimum(log_z2, 700.0))
        quad = z2[:, None] * (0.5 / k**2)
    else:
        quad = 0.5 * jnp.exp(log_z2[:, None] - 2.0 * logk)
    logits = log_w - logk - quad
    o, _ = categorical_inverse_cdf(random.uniform(fold(key, SUB_O), log_z2.shape), logits)
    o = jnp.where(obs, o, 0)
    n1 = jnp.sum(jnp.where(obs & (o > 0), 1.0, 0.0))
    fT = jnp.sum(tmask)
    kp = fold(key, SUB_P)
    lg1 = random.loggamma(fold(kp, 0), 2.5 + n1)
    lg2 = random.loggamma(fold(kp, 1), 37.5 + fT - n1)
    mx = jnp.maximum(lg1, lg2)
    lse = mx + jnp.log(jnp.exp(lg1 - mx) + jnp.exp(lg2 - mx))
    return o, lg1 - lse


# --------------------------------------------------------------------------------------------
# Structural mean block (DESIGN §3.7.2, weighted path)
# --------------------------------------------------------------------------------------------


def packed_index(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-major upper-triangle order: pairs ``(r, s)``, ``r ≤ s``, sorted by ``s`` then ``r``.

    The first ``q'(q'+1)/2`` packed columns cover exactly the leading ``q' × q'`` block, so a
    second equation bucket can use a prefix of ``FF`` (pilot recommendation for WP5).
    Returns ``(iu, ju, pos)`` with ``pos[r, s] = pos[s, r]`` = packed column.
    """
    iu = np.concatenate([np.arange(s + 1) for s in range(q)]).astype(np.int32)
    ju = np.concatenate([np.full(s + 1, s) for s in range(q)]).astype(np.int32)
    pos = np.zeros((q, q), dtype=np.int32)
    pos[iu, ju] = np.arange(iu.size, dtype=np.int32)
    pos[ju, iu] = np.arange(iu.size, dtype=np.int32)
    return iu, ju, pos


def precision_draw(K: Any, b: Any, z: Any) -> tuple[Any, Any]:
    """``θ = s ⊙ L⁻ᵀ(L⁻¹(s⊙b) + z)``, ``s = diag(K)^{−½}``, ``L = chol(sKs)`` (D-14; one problem)."""
    import jax.numpy as jnp
    from jax import lax

    s = 1.0 / jnp.sqrt(jnp.diagonal(K))
    L = lax.linalg.cholesky(s[:, None] * K * s[None, :], symmetrize_input=False)
    v = lax.linalg.triangular_solve(L, (s * b)[:, None], left_side=True, lower=True)
    th = lax.linalg.triangular_solve(
        L, v + z[:, None], left_side=True, lower=True, transpose_a=True
    )[:, 0]
    return s * th, jnp.min(jnp.diagonal(L))


def mean_draw(
    key: Any, w: Any, y: Any, F: Any, FF: Any, pos: np.ndarray, cmask: Any, prec0: Any, mean0: Any
) -> tuple[Any, Any, Any]:
    """Weighted structural draw of ``θ_i = [δ_i | a_i]`` with the packed Gram (DESIGN §3.7.2).

    ``G = (w @ FF)[pos]``; ``K = cmask⊗cmask ⊙ G + diag(prec0)``; ``r = prec0⊙mean0 +
    cmask⊙Fᵀ(w⊙y)``. Returns ``(θ ⊙ cmask, e = y − Fθ, min diag L)``.
    """
    from jax import random

    q = F.shape[1]
    G = (w @ FF)[pos]
    K = G * cmask[:, None] * cmask[None, :] + _diag(prec0)
    r = prec0 * mean0 + cmask * (F.T @ (w * y))
    th, mind = precision_draw(K, r, random.normal(fold(key, SUB_Z), (q,)))
    th = th * cmask
    return th, y - F @ th, mind


def _diag(v: Any) -> Any:
    import jax.numpy as jnp

    return jnp.diag(v)


# --------------------------------------------------------------------------------------------
# Data generators (simulated DGPs; seeds fixed)
# --------------------------------------------------------------------------------------------


def simulate_sv_series(
    rng: np.random.Generator, T: int, mu: float, phi: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """``e_t = exp(h_t/2)ε_t``, ``h_t = μ + φ(h_{t−1} − μ) + ση_t``, stationary ``h_0``."""
    h = np.empty(T + 1)
    h[0] = mu + sigma / np.sqrt(1.0 - phi * phi) * rng.normal()
    for t in range(1, T + 1):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma * rng.normal()
    return np.exp(h[1:] / 2.0) * rng.normal(size=T), h


@dataclass(frozen=True)
class VARSVData:
    Y: np.ndarray  # (T, n) dependent rows
    X: np.ndarray  # (T, k) [1, y_{t−1}', …, y_{t−p}']
    psi: np.ndarray  # (n,) AR(p) residual variances
    h_true: np.ndarray  # (T, n)


def simulate_var_sv(n: int, p: int, T: int, seed: int = 0, burn: int = 100) -> VARSVData:
    """Stationary VAR(p)-SV DGP (DESIGN §10.1 reference data; committed generator, seed 0).

    Own lag ``l`` coefficient ``0.5/l``, cross lags ``N(0, 0.02²)/l``, rescaled until the companion
    matrix has spectral radius ≤ 0.9; unit lower-triangular ``A`` with ``a_ij ~ N(0, 0.3²)``;
    structural log variances AR(1) with ``μ_i ~ U(−1, 1)``, ``φ = 0.95``, ``σ = 0.2``.
    """
    rng = np.random.default_rng(seed)
    B = np.zeros((p, n, n))
    for lag in range(p):
        B[lag] = rng.normal(scale=0.02, size=(n, n)) / (lag + 1)
        B[lag][np.diag_indices(n)] = 0.5 / (lag + 1)
    comp = np.zeros((n * p, n * p))
    for _ in range(50):
        comp[:n] = np.concatenate(list(B), axis=1)
        comp[n:, :-n] = np.eye(n * (p - 1))
        rho = np.max(np.abs(np.linalg.eigvals(comp)))
        if rho <= 0.9:
            break
        B *= 0.9 / rho
    A = np.eye(n) + np.tril(rng.normal(scale=0.3, size=(n, n)), -1)
    mu = rng.uniform(-1.0, 1.0, size=n)
    Tt = T + p + burn
    h = np.empty((Tt, n))
    h[0] = mu + 0.2 / np.sqrt(1 - 0.95**2) * rng.normal(size=n)
    for t in range(1, Tt):
        h[t] = mu + 0.95 * (h[t - 1] - mu) + 0.2 * rng.normal(size=n)
    y = np.zeros((Tt, n))
    for t in range(p, Tt):
        u = np.exp(h[t] / 2.0) * rng.normal(size=n)
        y[t] = sum(B[lag] @ y[t - 1 - lag] for lag in range(p)) + np.linalg.solve(A, u) + 0.1
    y = y[burn:]
    h = h[burn:]
    Y = y[p:]
    X = np.concatenate([np.ones((T, 1))] + [y[p - lag : T + p - lag] for lag in range(1, p + 1)], 1)
    psi = np.empty(n)
    for i in range(n):
        Z = np.concatenate(
            [np.ones((T, 1))] + [y[p - lag : T + p - lag, [i]] for lag in range(1, p + 1)], 1
        )
        beta, *_ = np.linalg.lstsq(Z, Y[:, i], rcond=None)
        psi[i] = np.sum((Y[:, i] - Z @ beta) ** 2) / (T - p - 1)
    return VARSVData(Y, X, psi, h[p:])


# --------------------------------------------------------------------------------------------
# MCMC summaries (NumPy)
# --------------------------------------------------------------------------------------------


def ess(x: np.ndarray) -> float:
    """Single-chain ESS with Geyer's initial monotone positive sequence (FFT autocorrelation)."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    xc = x - x.mean()
    var = xc @ xc / N
    if var <= 0.0 or not np.isfinite(var):
        return float("nan")
    f = np.fft.rfft(xc, 2 * N)
    acov = np.fft.irfft(f * np.conj(f))[:N] / N
    rho = acov / var
    pairs = rho[: N - N % 2].reshape(-1, 2).sum(axis=1)
    pos = np.argmax(pairs <= 0.0) if np.any(pairs <= 0.0) else pairs.size
    gam = np.minimum.accumulate(pairs[:pos]) if pos > 0 else np.array([1.0])
    tau = -1.0 + 2.0 * np.sum(gam)
    return float(N / max(tau, 1.0 / np.log10(max(N, 10))))


def split_rhat(chains: np.ndarray) -> float:
    """Split-R̂ (Gelman et al. 2013) of ``chains`` with shape (C, N)."""
    N = chains.shape[1]
    h = N // 2
    sp = np.concatenate([chains[:, :h], chains[:, h : 2 * h]], axis=0)
    m = sp.mean(axis=1)
    W = sp.var(axis=1, ddof=1).mean()
    Bv = h * m.var(ddof=1)
    var = (h - 1) / h * W + Bv / h
    return float(np.sqrt(var / W)) if W > 0 else float("nan")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
