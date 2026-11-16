"""Geweke test functions and SBC test quantities (DESIGN §9.5, §9.6; N06 §3.4, §4.4; N03 §7.2).

:func:`test_functions` returns an ordered ``{name: (value, finite)}`` dict. ``finite`` is a static
flag: the function has **finite variance under p(θ, y)** (Geweke 2004 eq. 2), which a z-test
needs (N06 §3.4). It depends on the prior's tails (``reference.PriorTails``, computed from the
concrete prior unless ``tails`` is given): e.g. ``h²`` under a ``Beta(·, 2)`` prior on
``(φ+1)/2``, ``σ²`` under ``σ² ~ IG(c₀ ≤ 2)``, or ``δ²`` under Mode II with IG shape ≤ 2.
Infinite-variance functions are reported with ``finite=False`` and are dropped from Geweke sets;
SBC ranks need no moments and use them all.

Contents, by source:

* every scalar parameter and the square where finite; latent states at fixed dates; within-block
  cross-products (``μφ``, ``φσ``, ``p·N_1``, ``ν·mean log τ``) — the per-kind references;
* the D3 centring cross-moment ``a_ij·γ_{i,(lag1,j)}`` and ``γ_{i,(lag1,j)}`` (N05 §13.3), the
  reduced-form own first lag ``B_{(lag1,i),i}`` of ``B = (A⁻¹Γ)'`` and ``‖Γ − AM'‖²_F``
  (DESIGN §9.6; ``M'`` = ``prior.eq.coef.mean0``, so under D3 it equals ``Σ_i ‖δ_i − m_i‖²``;
  structural means);
* cross-equation products ``π_{l,j1}π_{l,j2}`` (reduced form; in the ``linear_reduced`` reference);
* with SOC/SUR dummy rows (WP25), their squared residuals ``Σ_d (y_d − F_dθ_i)²`` per structural
  equation (``eq{i}.joint.dummy_ss``) or per column of ``Π`` (``glob.mean.dummy_ss[j]``);
* the exact log-likelihood, ``log p(θ)``, per-equation ``Σ_t e²/λ_eff`` and, with a transitory
  scale, the transitory-marginal log-likelihood ``Σ_t log p(e_t | h_t, p, ν)`` (N03 §2.2, §7.2);
* data functions on the log scale or bounded only (``mean_t log y²``, first autocorrelations of
  ``y`` and ``log y²``, ``mean tanh y``, sample correlations): raw data moments have infinite
  mean under SV priors (N06 §3.4). ``raw_moments=True`` adds ``mean y²``, cross moments and
  first autocovariances, flagged finite only for an uncoupled Constant core without transitory
  scale and ``E[λ²] < ∞`` (IG shape > 2; N06 §3.4).

Names: ``"eq{i}.{slot}.{name}"`` (slots ``struct``, ``mean``, ``core``, ``trans``, ``joint``),
``"glob.mean.{name}"``, ``"joint.{name}"`` (cross-equation), ``"loglik"``, ``"log_prior"`` and
``"data.{name}"``.

Phase 3 (``docs/phase3/DESIGN.md`` §9.4–§9.5; WP30), appended after the Phase 2 functions so the
Phase 2 order is unchanged:

* global-local and SSVS priors: at every GL coordinate ``(i, r)`` the bounded transforms
  ``eq{i}.coef.atan_dev[r] = arctan(d/√r_j)`` (``d = coefficient − m``) and
  ``eq{i}.coef.small_dev[r] = 1{|d| < 0.1√r_j}`` (from ``dev_log``); the family's local log
  scales at two fixed coordinates (own lag 1 of equation 0, and lag 1 of variable 0 in the last
  equation); group functions ``glob.coef.*`` (group log scales, ``a``-grid indices, the joint
  ``a_g·log ξ_g`` of M35, SSVS ``N1_g`` and ``log p_g``, and ``mean_log_v[g]``). Raw coefficient
  functions carry the deviations' tails (``PriorTails.coef``): e.g. under the horseshoe ``δ`` has
  no finite mean, so ``δ``, ``δ²``, ``B_own1`` … are SBC-only;
* CommonSV: ``glob.vol.*`` (``h`` at three dates and its mean, their arctan, ``ρ``, ``σ_h``,
  ``log σ_h``) and ``joint.W = Σ_t q_t e^{−h_t}``, ``q_t = Σ_i e²_{it}/λ_i`` (``χ²_{nT}`` under
  the joint, N08 §A11); ``log λ_i`` comes from the Constant core. Under the truncated-normal
  ``ρ`` prior ``Var(h_t) = ∞`` (``prior_tails``), so the ``h`` dates, ``mean_h``, the
  log-likelihoods and the log-scale data functions are flagged infinite (SBC-only), while their
  bounded transforms stay in the Geweke set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from rai_bvar.engines.mcmc import _jax  # noqa: F401  (float64 before any computation)
from rai_bvar.engines.mcmc.config import StaticConfig
from rai_bvar.engines.mcmc.state import ChainState, EngineData, PriorArrays
from rai_bvar.validation.dummies import has_dummies, structural_rows_eq
from rai_bvar.validation.log_joint import gl_coefficients, log_prior, loglik_terms, residuals
from rai_bvar.validation.reference import (
    PriorTails,
    SlotReferences,
    prior_tails,
    references_for,
    state_log_var,
    transitory_log_density,
)
from rai_bvar.validation.simulate import structural_gamma, unit_lower

Array = jax.Array
_F64 = jnp.float64
TestFns = dict[str, tuple[Array, bool]]


def _slice(tree: Any, i: int) -> Any:
    return jax.tree.map(lambda x: x[i], tree)


def _masked_mean(x: Array, m: Array) -> Array:
    return jnp.sum(jnp.where(m > 0, x, 0.0)) / jnp.sum(m)


def _acf1(x: Array, m: Array) -> Array:
    """First autocorrelation over the real rows ``t < T_eff`` (pairs with ``m_t m_{t−1} = 1``)."""
    xc = jnp.where(m > 0, x - _masked_mean(x, m), 0.0)
    num = jnp.sum(xc[1:] * xc[:-1] * m[1:] * m[:-1])
    return num / jnp.sum(xc**2)


def _data_functions(data: EngineData, raw_moments: bool, log_ok: bool, raw_ok: bool) -> TestFns:
    """``log_ok``: ``E[(log y²)²] < ∞`` (needs ``E h² < ∞`` under SV); ``raw_ok``: ``E y⁴ < ∞``."""
    Y, m = data.Y, data.tmask
    n = Y.shape[1]
    out: TestFns = {}
    log_y2 = jnp.where(m[:, None] > 0, jnp.log(jnp.where(m[:, None] > 0, Y**2, 1.0)), 0.0)
    for i in range(n):
        out[f"data.mean_log_y2[{i}]"] = (_masked_mean(log_y2[:, i], m), log_ok)
        out[f"data.mean_tanh_y[{i}]"] = (_masked_mean(jnp.tanh(Y[:, i]), m), True)
        out[f"data.acf1_y[{i}]"] = (_acf1(Y[:, i], m), True)
        out[f"data.acf1_log_y2[{i}]"] = (_acf1(log_y2[:, i], m), True)
    mean = jnp.sum(Y * m[:, None], axis=0) / jnp.sum(m)
    Yc = (Y - mean) * m[:, None]
    sd = jnp.sqrt(jnp.sum(Yc**2, axis=0))
    for i in range(n):
        for j in range(i + 1, n):
            out[f"data.corr_y[{i},{j}]"] = (jnp.sum(Yc[:, i] * Yc[:, j]) / (sd[i] * sd[j]), True)
    if raw_moments:
        T = jnp.sum(m)
        for i in range(n):
            out[f"data.mean_y2[{i}]"] = (jnp.sum(Y[:, i] ** 2) / T, raw_ok)
            out[f"data.acov1_y[{i}]"] = (jnp.sum(Y[1:, i] * Y[:-1, i]) / T, raw_ok)
            for j in range(i + 1, n):
                out[f"data.mean_yy[{i},{j}]"] = (jnp.sum(Y[:, i] * Y[:, j]) / T, raw_ok)
    return out


def _dummy_functions(
    st: ChainState, prior: PriorArrays, static: StaticConfig, tails: PriorTails
) -> TestFns:
    """Squared residuals of the SOC/SUR dummy rows (WP25): the rows' own statistic, sensitive to a
    dropped or mis-weighted row. Structural: ``Σ_d (y_{d,i} − F_dθ_i)²`` with ``θ_i = [δ_i | a_i]``
    (a λ-scaled block puts ``√λ_i`` in it: finite variance needs ``E λ² < ∞``); reduced form:
    ``Σ_d (Y_d[d, j] − x_d'π_j)²`` (Gaussian, every moment finite)."""
    n = static.shapes.n
    out: TestFns = {}
    if static.mean.kind == "linear_structural":
        scaled = static.mean.coef_scaling == "lambda" or static.struct.contemp_scaling == "lambda"
        for i in range(n):
            p = structural_rows_eq(jnp.int32(i), _slice(prior.eq, i), prior.shared, static, 0.0)
            theta = jnp.concatenate([st.eq.mean.delta[i], st.eq.struct.a[i]])
            r = p["y"] - p["F"] @ theta
            out[f"eq{i}.joint.dummy_ss"] = (jnp.sum(r * r), not scaled or tails.lam > 2.0)
    else:
        d = prior.shared.dummies
        R = d.Y - d.X @ st.glob.mean.Pi
        for j in range(n):
            out[f"glob.mean.dummy_ss[{j}]"] = (jnp.sum(R[:, j] ** 2), True)
    return out


def raw_moments_finite(static: StaticConfig, tails: PriorTails) -> bool:
    """Raw data moments have finite variance (``E y⁴ < ∞``; N06 §3.4): an uncoupled Constant core
    without transitory scale (``y`` is linear in ``√λ_i ε``) and ``E[λ²] < ∞``. Under coupling the
    lag coefficients carry ``√λ``, so ``y_T`` involves ``λ^{T/2}`` and high moments diverge. P3: a
    varying coefficient prior also needs ``E d⁴ < ∞`` (``tails.coef > 4``)."""
    gl_ok = not static.coef.varies or tails.coef > 4.0
    return static.vol.constant_weights and not static.vol.coupled and tails.lam > 2.0 and gl_ok


def test_functions(
    st: ChainState,
    data: EngineData,
    prior: PriorArrays,
    static: StaticConfig,
    *,
    raw_moments: bool = False,
    tails: PriorTails | None = None,
) -> TestFns:
    """Every test function of one chain state and dataset (module docstring); ordered dict.
    ``tails`` defaults to ``prior_tails(static, prior)`` (needs a concrete ``prior``)."""
    tails = prior_tails(static, prior) if tails is None else tails
    refs = references_for(static)
    n, n_det = static.shapes.n, static.shapes.n_det
    m = data.tmask
    h_ok = tails.h2 > 1.0  # E h² < ∞: log-likelihoods and log-scale data functions
    out: TestFns = {}

    def add(prefix: str, fns: TestFns) -> None:
        for name, (val, fin) in fns.items():
            out[f"{prefix}.{name}"] = (jnp.asarray(val, _F64), bool(fin))

    for i in range(n):
        eq_i = _slice(st.eq, i)
        add(f"eq{i}.struct", refs.structure.test_functions_eq(eq_i.struct, m, i=i, tails=tails))
        add(f"eq{i}.mean", refs.mean.test_functions_eq(eq_i.mean, m, i=i, tails=tails))
        add(f"eq{i}.coef", refs.coef.test_functions_eq(eq_i.coef, m, i=i, tails=tails))
        add(f"eq{i}.core", refs.core.test_functions_eq(eq_i.vol.core, m, i=i, tails=tails))
        add(f"eq{i}.trans", refs.trans.test_functions_eq(eq_i.vol.trans, m, i=i, tails=tails))
    add("glob.mean", refs.mean.test_functions_global(st.glob.mean, tails=tails))

    if static.mean.kind == "linear_structural":
        # λ-scaling (Mode II: a and δ; iw_equivalent: a) puts √λ_i in a_i and γ_i. A path product
        # of A⁻¹ never uses row j's own a_j next to γ_j, so B is linear in each √λ_j: Var(B) and
        # Var(γ) need E λ; a·γ = a·δ + M_jj a² needs E λ² when a is scaled.
        sa = static.struct.contemp_scaling == "lambda"
        sd = static.mean.coef_scaling == "lambda"
        lam1 = not (sa or sd) or tails.lam > 1.0
        cross = tails.lam > 2.0 if sa else (tails.lam > 1.0 if sd else True)
        # P3: γ, a·γ and B read GL deviations (E d² < ∞ needed); ‖Γ − AM'‖² needs E d⁴ < ∞
        gl2 = not static.coef.varies or tails.coef > 2.0
        gl4 = not static.coef.varies or tails.coef > 4.0
        lam1, cross = lam1 and gl2, cross and gl2
        gamma = structural_gamma(st, prior)
        A = unit_lower(st.eq.struct.a)
        B = jax.scipy.linalg.solve_triangular(A, gamma, lower=True, unit_diagonal=True).T
        for i in range(n):
            for j in range(i):
                g = gamma[i, n_det + j]  # γ_{i,(lag 1, variable j)}
                out[f"eq{i}.joint.gamma_lag1_{j}"] = (g, lam1)
                out[f"eq{i}.joint.a{j}*gamma_lag1_{j}"] = (st.eq.struct.a[i, j] * g, cross)
            out[f"eq{i}.joint.B_own1"] = (B[n_det + i, i], lam1)
        dev = gamma - A @ prior.eq.coef.mean0  # Γ − AM' (D3: row i is δ_i − m_i)
        out["joint.gamma_minus_AM_sq"] = (jnp.sum(dev**2), (not sd or tails.lam > 2.0) and gl4)

    if has_dummies(static):
        out.update(_dummy_functions(st, prior, static, tails))

    terms = loglik_terms(st, data, prior, static)
    out["loglik"] = (jnp.sum(terms), h_ok)
    out["log_prior"] = (log_prior(st, prior, static, m), True)
    e = residuals(st, data, prior, static)
    lv = state_log_var(st, static, refs=refs)
    live = m[None, :] > 0
    rss = jnp.sum(jnp.where(live, e**2 * jnp.exp(-jnp.where(live, lv, 0.0)), 0.0), axis=1)
    for i in range(n):
        out[f"eq{i}.joint.rss_std"] = (rss[i], True)  # χ²_T under the joint
        if n > 1:  # with one equation it is "loglik" itself
            out[f"eq{i}.joint.loglik"] = (jnp.sum(terms[i]), h_ok)
    if static.vol.transitory != "none":
        lp = jnp.where(live, state_log_var(st, static, "persistent", refs), 0.0)
        log_z2 = jnp.where(live, jnp.log(jnp.where(live, e**2, 1.0)) - lp, 0.0)
        dens = jax.vmap(lambda lz, tr: transitory_log_density(lz, tr, static))(
            log_z2, st.eq.vol.trans
        )
        marg = jnp.sum(jnp.where(live, dens - 0.5 * lp, 0.0), axis=1)  # log f(e/√λ) − ½ log λ
        for i in range(n):
            out[f"eq{i}.joint.loglik_marginal"] = (marg[i], h_ok)
    out.update(_data_functions(data, raw_moments, h_ok, raw_moments_finite(static, tails)))
    if static.coef.varies:  # P3 global-local and SSVS priors
        out.update(_gl_functions(st, prior, static, refs))
    if static.vol.common:  # P3 CommonSV
        add("glob.vol", refs.gvol.test_functions_global(st.glob.vol, tails=tails, tmask=m))
        out["joint.W"] = (jnp.sum(rss), True)  # Σ_t q_t e^{−h_t} ~ χ²_{nT} under the joint
    return out


def gl_coordinates(static: StaticConfig) -> tuple[tuple[int, int], ...]:
    """The two fixed GL coordinates ``(i, r)`` of the local-scale test functions: own lag 1 of
    equation 0 and lag 1 of variable 0 in the last equation (one own and one cross lag when
    ``n > 1``)."""
    n, nd = static.shapes.n, static.shapes.n_det
    return tuple(dict.fromkeys(((0, nd), (n - 1, nd))))


def _gl_functions(
    st: ChainState, prior: PriorArrays, static: StaticConfig, refs: SlotReferences
) -> TestFns:
    """The GL test functions of P3 §9.4 (module docstring); every one is bounded or a log of a
    gamma / inverse-gamma variable, so all are finite under ``p(θ, y)``."""
    gref, arr, gl = refs.coef, prior.eq.coef.gl, prior.shared.gl
    n, k, nd = static.shapes.n, static.shapes.k, static.shapes.n_det
    coef, dev_log = gl_coefficients(st, static)
    half_log_r = 0.5 * arr.log_r
    atan = jnp.arctan((coef - prior.eq.coef.mean0) * jnp.exp(-half_log_r))
    small = (dev_log < jnp.log(0.1) + half_log_r).astype(_F64)
    out: TestFns = {}
    for i in range(n):
        for r in range(nd, k):  # the GL slots are the lag slots (D-36)
            out[f"eq{i}.coef.atan_dev[{r}]"] = (atan[i, r], True)
            out[f"eq{i}.coef.small_dev[{r}]"] = (small[i, r], True)
    logs = gref.local_logs(st.eq.coef)
    for i, r in gl_coordinates(static):
        for name, val in logs.items():
            out[f"eq{i}.coef.{name}[{r}]"] = (jnp.asarray(val[i, r], _F64), True)
    for name, (val, fin) in gref.group_functions(st.eq.coef, st.glob.coef, arr, gl, static).items():
        out[f"glob.coef.{name}"] = (jnp.asarray(val, _F64), bool(fin))
    return out


def engine_data_template(static: StaticConfig, T_eff: int) -> EngineData:
    """Zero-filled :class:`EngineData` with the shapes of ``static`` (for shape-only tracing)."""
    s = static.shapes
    z = jnp.zeros
    return EngineData(
        Y=z((s.T_pad, s.n), _F64),
        X=z((s.T_pad, s.k), _F64),
        tmask=(jnp.arange(s.T_pad) < T_eff).astype(_F64),
        T_eff=jnp.asarray(T_eff, jnp.int32),
        hist=z((s.lags, s.n), _F64),
        t_last=jnp.zeros((), _F64),
    )


def test_function_spec(
    static: StaticConfig,
    prior: PriorArrays,
    T_eff: int,
    *,
    raw_moments: bool = False,
    tails: PriorTails | None = None,
    fn: Callable[..., TestFns] = test_functions,
) -> dict[str, bool]:
    """Ordered ``{name: finite}`` of :func:`test_functions`, found by tracing shapes only."""
    from rai_bvar.engines.mcmc.state import state_template

    tails = prior_tails(static, prior) if tails is None else tails
    flags: dict[str, bool] = {}

    def run(st: ChainState, data: EngineData) -> dict[str, Array]:
        d = fn(st, data, prior, static, raw_moments=raw_moments, tails=tails)
        flags.update({k: f for k, (_, f) in d.items()})
        return {k: v for k, (v, _) in d.items()}

    tmpl = jax.tree.map(jnp.asarray, state_template(static))
    jax.eval_shape(run, tmpl, engine_data_template(static, T_eff))
    return flags


__all__ = [
    "engine_data_template",
    "gl_coordinates",
    "raw_moments_finite",
    "test_function_spec",
    "test_functions",
]

# pytest collects module-level names starting with "test" from test modules that import them.
test_functions.__test__ = False  # type: ignore[attr-defined]
test_function_spec.__test__ = False  # type: ignore[attr-defined]
