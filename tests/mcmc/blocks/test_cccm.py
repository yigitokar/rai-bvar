"""Reduced-form CCCM kernel: ``cccm_core``, ``CCCMStep``, ``ReducedEq`` (DESIGN §13 WP11; §3.7.3,
§4.3, §9.3–§9.5, §9.8 A7; N04 §5.3, §6, §9; N06 §5).

The truth is always written by someone else: the NumPy SUR/SWA oracle in ``swa_oracle.py`` (test
code, no JAX), the printed numbers of N04 §9.1, and the independent ``validation`` references
(``log_joint``, prior sampler and simulator; WP9).

Tier U (deterministic):

1. N04 §9.1 toy table: ``K_j`` and conditional means; the oracle's joint moments.
2. 20 random problems (``n ∈ {1, 2, 3, 5}``, padded rows, time-varying ``λ``) vs the SUR oracle:
   the exposed conditional and one injected-noise sweep (rtol 1e-10).
3. Kernel invariance (N04 §9.2): ``vec Π′ = M vec Π + c + N vec Z`` extracted from the kernel;
   ``(I − M)⁻¹c = Q⁻¹b``, the Lyapunov solution ``S = MSMᵀ + NNᵀ`` equals ``Q⁻¹``, ``M`` equals the
   block Gauss–Seidel matrix, ``ρ(M) = 0.5037`` on the toy; noiseless sweeps converge to ``Q⁻¹b``.
4. CCM-2019 regression guard: the 2019 conditional differs for ``j < n`` and equals the CTA for
   ``j = n``; the mutant injected through the ``cccm_core`` override has ``M = 0`` and the wrong
   stationary law (N04 §4, §9.2), and fails the density-ratio identity.
5. ``A = I`` decouples (``M = 0``, per-equation regressions); ``n = 1``.
6. Packed, tiled and batched ``K_j``; precompute vs in-loop (1e-12); padded rows inert;
   ``jit``/``vmap`` over chains; failures are values.
7. Density-ratio identities of every ``π_j`` step against ``validation.log_joint`` (Constant and
   SV, 20 states each; relative 1e-8).
8. ``CCCMStep`` and ``ReducedEq`` plumbing (keys, ``eq_map``, write set, init, log prior,
   overrides, protocols, registry) with minimal local doubles of the vol and coefficient kernels,
   and the ``a_j | Π`` step of ``Recursive`` (WP10) on ``ỹ = U`` against N04 §6.2.

Cold start (DESIGN §5.4): priors whose λ centre is not 1 (Constant, IW-equivalent, SV AR and RW
with ``m₀ ≠ b_μ``), through the kernel, the ``CCCMStep`` that ``compose`` builds and the runner's
``init``; the centre equals ``runner.initial_log_variance`` (one convention for every cold start).

Tier B: block-conditional SBC of every ``π_j`` step (N06 §5.2), including the first update of the
production sweep, with the CCM-2019 step as a negative control; block Geweke ``cccm_only``
(G-small) through the production ``CCCMStep`` and its CCM-2019 power check, and the same test
through the WP14 ``JointSpec.step``.

Tier A7 (N04 §9.5; DESIGN §9.8), reduced × Constant, ``T = 60``, means, second moments and
5/50/95% quantile indicators within MCSE (Bonferroni):

* Π step: SWA-Gibbs vs CTA-Gibbs on shared NumPy ``a`` and ``λ`` steps, ``n ∈ {2, 3, 4}``,
  ``p ∈ {1, 2}`` (nightly; a PR-sized version every PR).
* Engine: the production engine (runner ``init``, production sweep, padded rows) vs the NumPy
  SWA Gibbs, and vs the engine with SWA swapped in for the Π step (nightly, same cases; a
  PR-sized version every PR); power checks with the CCM-2019 Π step and an ``a``-step wiring
  mutant.
"""

from __future__ import annotations

import dataclasses
import functools
import traceback
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from jax import lax, random
from scipy import stats

from rai_bvar.components.mean import linear_reduced as LR
from rai_bvar.components.mean.linear_reduced import CCCMStep, ReducedEq
from rai_bvar.engines.mcmc import keys, registry
from rai_bvar.engines.mcmc.data import base_cache, build_engine_data, packed_products
from rai_bvar.engines.mcmc.keys import Block, Sub
from rai_bvar.engines.mcmc.linalg import precision_draw
from rai_bvar.engines.mcmc.protocols import Batching, GlobalStep, MeanBlock
from rai_bvar.engines.mcmc.state import (
    CCCMDraw,
    CCCMGlobal,
    ConstantPrior,
    Coupling,
    Empty,
    EqContext,
    GaussianDiag,
    GaussianPrec,
    GlobDraw,
    StepInfo,
    SweepCtx,
    leaf_path,
)
from rai_bvar.validation import joint as J
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.identities import density_ratio_error
from rai_bvar.validation.reference import effective_log_var, references_for
from rai_bvar.validation.sbc import block_sbc

try:  # package layout (tests/mcmc/blocks/__init__.py) or rootdir-inserted directory
    from . import swa_oracle as O
except ImportError:
    import swa_oracle as O

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

KEY = random.key(20261022, impl="threefry2x32")
CORE = jax.jit(LR.cccm_core, static_argnames=("gram", "precompute", "tile"))


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def close(a: Any, b: Any, rtol: float, what: str = "") -> None:
    """Relative closeness against the scale of ``b``."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(1.0, float(np.max(np.abs(b)))) if b.size else 1.0
    err = float(np.max(np.abs(a - b))) if b.size else 0.0
    assert err <= rtol * scale, f"{what}: max error {err:.3e} > {rtol:.1e}·{scale:.3g}"


# --------------------------------------------------------------------------- problems


@dataclass(frozen=True)
class Problem:
    """One Π-block problem in engine layout; rows ``t ≥ T_eff`` are padding (garbage ``log λ``)."""

    Pi: np.ndarray  # (k, n)
    a: np.ndarray  # (n, n−1)
    log_lam: np.ndarray  # (n, T_pad)
    X: np.ndarray  # (T_pad, k)
    Y: np.ndarray  # (T_pad, n)
    mean0: np.ndarray  # (n, k)
    prec0: np.ndarray  # (n, k)
    tmask: np.ndarray  # (T_pad,)

    @property
    def k(self) -> int:
        return self.X.shape[1]

    @property
    def n(self) -> int:
        return self.Y.shape[1]

    @property
    def inv_lam(self) -> jax.Array:
        return LR.masked_inv_lam(jnp.asarray(self.log_lam), jnp.asarray(self.tmask))

    @property
    def lam(self) -> np.ndarray:
        """Variances for the oracle (1 on padded rows, which it skips)."""
        live = self.tmask > 0
        return np.where(live[None], np.exp(np.where(live[None], self.log_lam, 0.0)), 1.0)

    @property
    def XX(self) -> jax.Array:
        return packed_products(jnp.asarray(self.X))

    def sur(self) -> tuple[np.ndarray, np.ndarray]:
        return O.sur_system(self.a, self.lam, self.X, self.Y, self.mean0, self.prec0, self.tmask)

    def core(
        self, Pi: Any, Z: Any, *, gram: str = "packed", precompute: bool = True, tile: int = 0
    ):
        XX = self.XX if gram == "packed" else Empty()
        kw = {"tile": tile} if gram == "packed_tiled" else {}
        return CORE(
            jnp.asarray(Pi), jnp.asarray(self.a), self.inv_lam, jnp.asarray(self.X),
            jnp.asarray(self.Y), XX, jnp.asarray(self.mean0), jnp.asarray(self.prec0),
            jnp.asarray(Z), gram=gram, precompute=precompute, **kw,
        )  # fmt: skip


def toy_problem() -> Problem:
    T = O.TOY_X.shape[0]
    return Problem(
        Pi=O.TOY_PI, a=O.toy_state_a(), log_lam=np.log(O.TOY_LAM), X=O.TOY_X, Y=O.TOY_Y,
        mean0=O.TOY_M, prec0=1.0 / O.TOY_V, tmask=np.ones(T),
    )  # fmt: skip


def random_problem(
    seed: int, n: int, k: int, T_eff: int, T_pad: int | None = None, *, tv: bool = True,
    a_scale: float = 0.8, identity_a: bool = False,
) -> Problem:  # fmt: skip
    r = rng(seed)
    T_pad = T_eff if T_pad is None else T_pad
    X = np.zeros((T_pad, k))
    X[:T_eff] = r.normal(size=(T_eff, k))
    X[:T_eff, 0] = 1.0
    Y = np.zeros((T_pad, n))
    Y[:T_eff] = r.normal(size=(T_eff, n))
    cm = (np.arange(n - 1)[None, :] < np.arange(n)[:, None]).astype(float)
    a = 0.0 * cm if identity_a else a_scale * r.normal(size=(n, n - 1)) * cm
    if tv:
        log_lam = 0.7 * r.normal(size=(n, T_pad))
    else:
        log_lam = np.repeat(0.5 * r.normal(size=(n, 1)), T_pad, axis=1)
    # garbage on padded rows: never read (masked before the exp)
    garbage = np.array([800.0, -800.0, np.nan, np.inf])
    log_lam[:, T_eff:] = garbage[np.arange(T_pad - T_eff) % 4][None, :]
    tmask = (np.arange(T_pad) < T_eff).astype(float)
    return Problem(
        Pi=0.5 * r.normal(size=(k, n)), a=a, log_lam=log_lam, X=X, Y=Y,
        mean0=0.3 * r.normal(size=(n, k)), prec0=np.exp(r.normal(size=(n, k))), tmask=tmask,
    )  # fmt: skip


def affine_map(core_fn, prob: Problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(M, c, N)`` of ``vec Π′ = M vec Π + c + N vec Z`` extracted from an implementation
    (N04 §9.2); ``vec Z`` stacks the rows ``z_j``."""
    k, n = prob.k, prob.n
    args = (
        jnp.asarray(prob.a), prob.inv_lam, jnp.asarray(prob.X), jnp.asarray(prob.Y), prob.XX,
        jnp.asarray(prob.mean0), jnp.asarray(prob.prec0),
    )  # fmt: skip

    def f(vpi: jax.Array, vz: jax.Array) -> jax.Array:
        Pi = vpi.reshape(n, k).T
        out, _ = core_fn(Pi, *args, vz.reshape(n, k), gram="packed", precompute=True)
        return out.T.reshape(-1)

    zero = jnp.zeros(n * k)
    eye = jnp.eye(n * k)
    c = jax.jit(f)(zero, zero)
    M = jax.jit(jax.vmap(lambda e: f(e, zero)))(eye).T - c[:, None]
    N = jax.jit(jax.vmap(lambda e: f(zero, e)))(eye).T - c[:, None]
    return np.asarray(M), np.asarray(c), np.asarray(N)


def stationary_law(M: np.ndarray, c: np.ndarray, N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``(I − M)⁻¹c`` and covariance ``S = MSMᵀ + NNᵀ`` of the affine Gaussian chain."""
    mean = np.linalg.solve(np.eye(M.shape[0]) - M, c)
    S = sla.solve_discrete_lyapunov(M, N @ N.T)
    return mean, 0.5 * (S + S.T)


def ccm2019_core(Pi, a, inv_lam, X, Y, XX, mean0, prec0, Z, *, gram, precompute, tile=0):
    """Mutant M1 (CCM 2019 triangular step; N04 §4) with the ``cccm_core`` signature: ``π_j``
    from equation ``j`` alone, ``y*_j = y_j + Σ_{i<j} a_ji u_i`` with the updated ``π_{<j}``."""
    A = LR.unit_lower(a)
    ok = jnp.ones((), bool)
    for j in range(Pi.shape[1]):
        U = Y - X @ Pi
        ystar = Y[:, j] + U[:, :j] @ A[j, :j]
        w = inv_lam[j]
        K = X.T @ (w[:, None] * X) + jnp.diag(prec0[j])
        b = prec0[j] * mean0[j] + X.T @ (w * ystar)
        pj, info = precision_draw(K, b, Z[j])
        ok = ok & info.ok
        Pi = Pi.at[:, j].set(pj)
    return Pi, StepInfo(ok, jnp.full((), jnp.inf))


# --------------------------------------------------------------------------- local doubles


@dataclass(frozen=True)
class VolDouble:
    """Minimal ``VolBlock`` stand-in: ``log_var_eq`` from the independent validation references
    (the kernels of WP12/WP13 are not needed to test the Π step)."""

    static: Any

    def log_var_eq(self, vs: Any, gvol: Any, which: str = "effective") -> jax.Array:
        refs = references_for(self.static)
        T = self.static.shapes.T_pad
        return refs.core.latent_log_var_eq(vs.core, which, T_pad=T) + refs.trans.latent_log_var_eq(
            vs.trans, which, T_pad=T
        )


@dataclass(frozen=True)
class CoefDouble:
    """Minimal ``CoefPrior`` stand-in with the ``FixedDiagonal`` moments (DESIGN §3.7.7)."""

    def moments_eq(self, cs_i: Any, i: Any, prior_i: Any, glob_coef: Any) -> GaussianDiag:
        c = prior_i.coef
        return GaussianDiag(mean=c.mean0, prec=c.prec0, cmask=jnp.ones_like(c.prec0))


def lax_map(f):
    """The ``Batching.MAP`` ``eq_map`` of DESIGN §4.5."""
    return lambda *a: lax.map(lambda t: f(*t), a)


def spec_state(name: str, seed: int, **overrides: Any):
    spec = J.joint_spec(name, **overrides)
    st = spec.prior_sample(random.fold_in(KEY, seed))
    data = spec.simulate(random.fold_in(KEY, 10_000 + seed), st)
    cache = ReducedEq(spec.static.mean).prepare(
        data, spec.prior, base_cache(data, None, spec.static)
    )
    return spec, st, data, cache


def reference_inv_lam(spec, st, data) -> jax.Array:
    return LR.masked_inv_lam(effective_log_var(st.eq.vol, spec.static), data.tmask)


def step_of(spec, **kw) -> CCCMStep:
    return CCCMStep(spec.static.mean, vol=VolDouble(spec.static), **kw)


_OWN = ("CCCMStep", "ReducedEq", "linear_reduced")


def production_or_skip(fn, *args: Any, **kw: Any) -> Any:
    """Call an engine-assembly entry point (WP14: ``compose``, ``make_runner``,
    ``production_sweep``); skip while the assembly or a sibling kernel does not build. An error
    raised from this module's kernels (``CCCMStep``, ``ReducedEq``) is never skipped."""
    try:
        return fn(*args, **kw)
    except Exception as err:  # pragma: no cover - engine assembly or a sibling kernel not ready
        frames = traceback.extract_tb(err.__traceback__)
        if any(f.filename.endswith("linear_reduced.py") for f in frames) or any(
            name in str(err) for name in _OWN
        ):
            raise
        pytest.skip(f"production engine not available yet: {type(err).__name__}: {err}")


# =========================================================================== tier U


def test_toy_table_conditionals():
    """N04 §9.1: K_j and μ̄_j at Π_cur equal the printed table and the SUR oracle."""
    p = toy_problem()
    Q, b = p.sur()
    for j in range(3):
        g = LR.cccm_conditional_j(p.Pi, p.a, p.inv_lam, p.X, p.Y, p.mean0, p.prec0, j)
        K = np.asarray(g.K)
        mu = np.linalg.solve(K, np.asarray(g.b))
        np.testing.assert_allclose(K, O.TOY_K[j], rtol=1e-10, atol=1e-10)  # printed to 1e-10
        np.testing.assert_allclose(mu, O.TOY_MEAN[j], rtol=1e-10, atol=1e-10)
        Ko, muo = O.conditional_j(Q, b, p.Pi, j)
        close(K, Ko, 1e-12, f"K_{j} vs SUR")
        close(mu, muo, 1e-12, f"mean_{j} vs SUR")
        assert g.K.dtype == g.b.dtype == jnp.float64
    # traced j gives the same conditional
    g1 = jax.jit(LR.cccm_conditional_j)(p.Pi, p.a, p.inv_lam, p.X, p.Y, p.mean0, p.prec0, 1)
    close(g1.K, O.TOY_K[1], 1e-10)


def test_toy_joint_posterior_of_the_oracle():
    """The oracle reproduces N04 §9.1's joint posterior of vec Π (printed to 1e-10)."""
    p = toy_problem()
    mean, cov = O.joint_moments(*p.sur())
    np.testing.assert_allclose(mean, O.TOY_JOINT_MEAN, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), O.TOY_JOINT_SD, atol=1e-10)


def test_toy_sweep_matches_reference():
    p = toy_problem()
    Z = rng(1).normal(size=(3, 2))
    out, info = p.core(p.Pi, Z)
    ref = O.cta_sweep_reference(p.Pi, p.a, p.lam, p.X, p.Y, p.mean0, p.prec0, Z)
    close(out, ref, 1e-12, "toy sweep")
    assert bool(info.ok) and 0.0 < float(info.min_pivot) <= 1.0
    assert out.dtype == jnp.float64


RANDOM_CASES = [(s, (1, 2, 3, 5)[s % 4], 1 + s % 5, 7 + 3 * s) for s in range(20)]


@pytest.mark.parametrize(("seed", "n", "k", "T_eff"), RANDOM_CASES)
def test_random_problems_vs_sur_oracle(seed, n, k, T_eff):
    """20 random problems: every conditional and one injected-noise sweep equal the SUR oracle
    (N04 §9.1 assertion 1; padded rows and time-varying λ included)."""
    T_pad = 32 * (-(-T_eff // 32))
    p = random_problem(seed, n, k, T_eff, T_pad, tv=seed % 3 != 0)
    Q, b = p.sur()
    for j in range(n):
        g = LR.cccm_conditional_j(p.Pi, p.a, p.inv_lam, p.X, p.Y, p.mean0, p.prec0, j)
        Ko, muo = O.conditional_j(Q, b, p.Pi, j)
        close(g.K, Ko, 1e-10, f"K_{j}")
        close(np.linalg.solve(np.asarray(g.K), np.asarray(g.b)), muo, 1e-10, f"mean_{j}")
    Z = rng(100 + seed).normal(size=(n, k))
    out, info = p.core(p.Pi, Z)
    ref = O.cta_sweep_reference(p.Pi, p.a, p.lam, p.X, p.Y, p.mean0, p.prec0, Z, p.tmask)
    assert bool(info.ok)
    close(out, ref, 1e-10, "sweep")


def test_kernel_invariance_toy():
    """N04 §9.2: the stationary law of the extracted affine map is the joint SUR Gaussian, M is
    the block Gauss–Seidel matrix of Q and ρ(M) = 0.5037."""
    p = toy_problem()
    Q, b = p.sur()
    mu, Sigma = O.joint_moments(Q, b)
    M, c, N = affine_map(LR.cccm_core, p)
    close(M, O.block_gauss_seidel(Q, p.k, p.n), 1e-12, "M vs Gauss-Seidel")
    assert abs(max(abs(np.linalg.eigvals(M))) - 0.5037) < 5e-5
    mean, S = stationary_law(M, c, N)
    close(mean, mu, 1e-10, "stationary mean")
    close(S, Sigma, 1e-10, "stationary covariance")
    # zero-noise iteration = block Gauss–Seidel converges to the joint mean (N04 §9.2)
    sweep = jax.jit(lambda Pi: p.core(Pi, jnp.zeros((p.n, p.k)))[0])
    Pi = jnp.asarray(p.Pi)
    for _ in range(200):
        Pi = sweep(Pi)
    close(O.vec(np.asarray(Pi)), mu, 1e-12, "noiseless limit")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_invariance_random(seed):
    """As above on n = 4, k = 3 with padding and time-varying λ."""
    p = random_problem(500 + seed, 4, 3, 21, 32)
    Q, b = p.sur()
    mu, Sigma = O.joint_moments(Q, b)
    M, c, N = affine_map(LR.cccm_core, p)
    close(M, O.block_gauss_seidel(Q, p.k, p.n), 1e-11, "M vs Gauss-Seidel")
    assert max(abs(np.linalg.eigvals(M))) < 1.0
    mean, S = stationary_law(M, c, N)
    close(mean, mu, 1e-10, "stationary mean")
    close(S, Sigma, 1e-10, "stationary covariance")


def test_ccm2019_regression_guard():
    """CCM 2019 (N04 §4, §9.1 assertion 2): its conditional differs for j < n (max |ΔK| = 2.82,
    5.98) and equals the CTA for j = n; the mutant injected through the ``cccm_core`` override
    has M = 0 and a stationary law off by 0.040 (mean) and 0.028 (covariance), s.d. 2.7–5.6% wide."""
    p = toy_problem()
    Q, b = p.sur()
    mu, Sigma = O.joint_moments(Q, b)
    dK = []
    for j in range(3):
        Kc, muc = O.ccm2019_conditional_j(p.Pi, p.a, p.lam, p.X, p.Y, p.mean0, p.prec0, j)
        K, m = O.conditional_j(Q, b, p.Pi, j)
        if j < 2:
            np.testing.assert_allclose(Kc, O.TOY_CCM_K[j], atol=1e-10)
            np.testing.assert_allclose(muc, O.TOY_CCM_MEAN[j], atol=1e-10)
        dK.append(np.max(np.abs(Kc - K)))
        if j == 2:
            close(Kc, K, 1e-12, "last equation K")
            close(muc, m, 1e-12, "last equation mean")
    np.testing.assert_allclose(dK[:2], [2.82, 5.98], atol=0.01)

    # the mutant (JAX) is the oracle's CCM step, injected through the override seam
    Z = rng(2).normal(size=(3, 2))
    out, _ = jax.jit(ccm2019_core, static_argnames=("gram", "precompute"))(
        p.Pi, p.a, p.inv_lam, p.X, p.Y, p.XX, p.mean0, p.prec0, Z, gram="packed",
        precompute=True,
    )  # fmt: skip
    close(out, O.ccm2019_sweep(p.Pi, p.a, p.lam, p.X, p.Y, p.mean0, p.prec0, Z), 1e-12, "mutant")
    spec = J.joint_spec("cccm_only")
    mutant = step_of(spec).with_overrides({"cccm_core": ccm2019_core})
    assert mutant.impl("cccm_core") is ccm2019_core
    M, c, N = affine_map(mutant.impl("cccm_core"), p)
    assert np.max(np.abs(M)) < 1e-14  # ignores Π_old (ρ = 0)
    mean, S = stationary_law(M, c, N)
    assert 0.035 < np.max(np.abs(mean - mu)) < 0.045
    assert 0.022 < np.max(np.abs(S - Sigma)) < 0.034
    ratio = np.sqrt(np.diag(S) / np.diag(Sigma))
    assert 1.02 < ratio.max() < 1.06 and ratio.min() > 1.0 - 1e-12
    # while the production map has the right law (test_kernel_invariance_toy) and ρ > 0
    M0, _, _ = affine_map(step_of(spec).impl("cccm_core"), p)
    assert max(abs(np.linalg.eigvals(M0))) > 0.5


@pytest.mark.parametrize("n", [1, 2, 4])
def test_identity_A_decouples(n):
    """A = I (N04 §8): M = 0 and every conditional is the per-equation weighted regression."""
    p = random_problem(40 + n, n, 3, 15, 32, identity_a=True)
    M, _, _ = affine_map(LR.cccm_core, p)
    assert np.max(np.abs(M)) < 1e-13  # zero up to the rounding of G_j + W_·j ⊙ Xπ_j
    live = p.tmask > 0
    for j in range(n):
        g = LR.cccm_conditional_j(p.Pi, p.a, p.inv_lam, p.X, p.Y, p.mean0, p.prec0, j)
        w = np.where(live, 1.0 / p.lam[j], 0.0)
        K = np.diag(p.prec0[j]) + p.X.T @ (w[:, None] * p.X)
        bb = p.prec0[j] * p.mean0[j] + p.X.T @ (w * p.Y[:, j])
        close(g.K, K, 1e-12, "K")
        close(g.b, bb, 1e-12, "b")


def test_single_equation_is_one_regression():
    """n = 1: A = [1], W = 1/λ; one sweep is an exact draw from the heteroskedastic regression."""
    p = random_problem(77, 1, 4, 25, 32)
    assert p.a.shape == (1, 0)
    Z = rng(3).normal(size=(1, 4))
    out, info = p.core(p.Pi, Z)
    w = np.where(p.tmask > 0, 1.0 / p.lam[0], 0.0)
    K = np.diag(p.prec0[0]) + p.X.T @ (w[:, None] * p.X)
    bb = p.prec0[0] * p.mean0[0] + p.X.T @ (w * p.Y[:, 0])
    close(out[:, 0], O.scaled_draw(K, bb, Z[0]), 1e-12, "n = 1")
    assert bool(info.ok)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gram_modes_agree(seed):
    """Packed, tiled (several tile widths) and batched K_j equal the dense NumPy Gram (1e-13
    relative) and give the same sweep (1e-12)."""
    p = random_problem(900 + seed, 3, 2 + 2 * seed, 19, 32)
    W = LR.cccm_weights(jnp.asarray(p.a), p.inv_lam)
    A = O.unit_lower(p.a)
    inv = np.where(p.tmask[None] > 0, 1.0 / p.lam, 0.0)
    np.testing.assert_allclose(np.asarray(W), inv.T @ (A * A), rtol=1e-14, atol=1e-14)
    dense = np.stack(
        [p.X.T @ (np.asarray(W)[:, j, None] * p.X) + np.diag(p.prec0[j]) for j in range(3)]
    )
    P_x = p.k * (p.k + 1) // 2
    modes = [("packed", 0), ("batched", 0)] + [("packed_tiled", t) for t in (1, 3, P_x)]
    Z = rng(seed).normal(size=(3, p.k))
    ref, _ = p.core(p.Pi, Z, gram="batched")
    for gram, tile in modes:
        XX = p.XX if gram == "packed" else Empty()
        K = LR.cccm_precisions(W, jnp.asarray(p.X), XX, jnp.asarray(p.prec0), gram=gram, tile=tile)
        close(K, dense, 1e-13, f"K {gram}/{tile}")
        assert np.array_equal(np.asarray(K), np.swapaxes(np.asarray(K), 1, 2))
        out, info = p.core(p.Pi, Z, gram=gram, tile=tile)
        close(out, ref, 1e-12, f"sweep {gram}/{tile}")
        assert bool(info.ok)
    # gram="packed" without cached products forms them on the fly (same products, same draw)
    out_nc, _ = CORE(
        jnp.asarray(p.Pi), jnp.asarray(p.a), p.inv_lam, jnp.asarray(p.X), jnp.asarray(p.Y),
        Empty(), jnp.asarray(p.mean0), jnp.asarray(p.prec0), jnp.asarray(Z), gram="packed",
        precompute=True,
    )  # fmt: skip
    out_c, _ = p.core(p.Pi, Z, gram="packed")
    np.testing.assert_array_equal(np.asarray(out_nc), np.asarray(out_c))


@pytest.mark.parametrize("gram", ["packed", "batched"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_precompute_vs_in_loop(gram, seed):
    """All (L_j, s_j) before the loop or inside it: same draw (1e-12) and same pivots."""
    p = random_problem(300 + seed, 4, 5, 26, 32)
    Z = rng(seed).normal(size=(4, 5))
    a, ia = p.core(p.Pi, Z, gram=gram, precompute=True)
    b, ib = p.core(p.Pi, Z, gram=gram, precompute=False)
    close(a, b, 1e-12, "precompute vs in-loop")
    assert bool(ia.ok) and bool(ib.ok)
    np.testing.assert_allclose(float(ia.min_pivot), float(ib.min_pivot), rtol=1e-12)


def test_padding_is_inert():
    """Padded rows (zero data, garbage log λ: ±800, NaN, ∞) change nothing: the padded problem
    equals the unpadded one (1e-12), for every Gram mode and both precompute settings."""
    padded = random_problem(61, 3, 4, 13, 32)
    T = 13
    unpadded = dataclasses.replace(
        padded, log_lam=padded.log_lam[:, :T], X=padded.X[:T], Y=padded.Y[:T], tmask=np.ones(T)
    )
    assert not np.all(np.isfinite(padded.log_lam))
    Z = rng(5).normal(size=(3, 4))
    for gram, pre in (("packed", True), ("batched", False), ("packed_tiled", True)):
        tile = 4 if gram == "packed_tiled" else 0
        a, ia = padded.core(padded.Pi, Z, gram=gram, precompute=pre, tile=tile)
        b, _ = unpadded.core(unpadded.Pi, Z, gram=gram, precompute=pre, tile=tile)
        close(a, b, 1e-12, f"padded vs unpadded ({gram})")
        assert bool(ia.ok) and np.all(np.isfinite(np.asarray(a)))
    inv = np.asarray(padded.inv_lam)
    assert np.all(inv[:, T:] == 0.0) and np.all(np.isfinite(inv))


def test_vmap_over_chains_equals_loop():
    """vmap over chains (DESIGN §5.2) equals per-chain calls (1e-14); float64 throughout."""
    p = random_problem(88, 3, 4, 20, 32)
    C = 3
    r = rng(9)
    Pis = r.normal(size=(C, p.k, p.n))
    As = 0.5 * r.normal(size=(C, p.n, p.n - 1)) * (np.arange(2)[None] < np.arange(3)[:, None])
    Zs = r.normal(size=(C, p.n, p.k))
    inv = jnp.stack([p.inv_lam * s for s in (1.0, 0.5, 2.0)])
    f = functools.partial(LR.cccm_core, gram="packed", precompute=True)
    batched, info = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, None, None, None, None, None, 0)))(
        jnp.asarray(Pis), jnp.asarray(As), inv, jnp.asarray(p.X), jnp.asarray(p.Y), p.XX,
        jnp.asarray(p.mean0), jnp.asarray(p.prec0), jnp.asarray(Zs),
    )  # fmt: skip
    assert batched.dtype == jnp.float64 and info.ok.shape == (C,)
    for c in range(C):
        one, _ = CORE(
            jnp.asarray(Pis[c]), jnp.asarray(As[c]), inv[c], jnp.asarray(p.X), jnp.asarray(p.Y),
            p.XX, jnp.asarray(p.mean0), jnp.asarray(p.prec0), jnp.asarray(Zs[c]), gram="packed",
            precompute=True,
        )  # fmt: skip
        close(batched[c], one, 1e-14, f"chain {c}")


def test_failures_are_values():
    """A non-PD K_j or a NaN weight sets ``ok = False`` (min pivot ≤ 0 for the factorisation);
    nothing raises inside jit and no jitter is added (D-18)."""
    p = random_problem(5, 3, 3, 20, 32)
    bad = p.prec0.copy()
    bad[1, 0] = -1e6  # K_1 indefinite
    q = dataclasses.replace(p, prec0=bad)
    Z = rng(0).normal(size=(3, 3))
    for pre in (True, False):
        _, info = q.core(q.Pi, Z, precompute=pre)
        assert not bool(info.ok) and float(info.min_pivot) <= 0.0
        assert np.isfinite(float(info.min_pivot))
    log_lam = p.log_lam.copy()
    log_lam[2, 3] = np.nan  # a real row
    _, info = dataclasses.replace(p, log_lam=log_lam).core(p.Pi, Z)
    assert not bool(info.ok)


@pytest.mark.parametrize("name", ["cccm_only", "reduced_sv"])
def test_density_ratio_identities(name):
    """Every π_j step at 20 random states: log q(π′_j) − log q(π_j) = log p(π′_j, ·, y) −
    log p(π_j, ·, y) with the independent ``validation.log_joint`` (relative 1e-8; DESIGN §9.3
    item 2). The CCM-2019 π_0 conditional fails the same identity (N06 §5.1)."""
    worst, ccm_errors = 0.0, []
    for s in range(20):
        spec, st, data, _ = spec_state(name, s)
        n, k = spec.static.shapes.n, spec.static.shapes.k
        j = s % n
        inv = reference_inv_lam(spec, st, data)
        c = spec.prior.eq.coef
        Pi = st.glob.mean.Pi
        g = LR.cccm_conditional_j(Pi, st.eq.struct.a, inv, data.X, data.Y, c.mean0, c.prec0, j)
        new = Pi.at[:, j].set(Pi[:, j] + 0.3 * random.normal(random.fold_in(KEY, 7 + s), (k,)))
        x0, x1 = (
            st,
            dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(new))),
        )

        def log_q(x, K=g.K, b=g.b, j=j):
            pj = x.glob.mean.Pi[:, j]
            return -0.5 * pj @ K @ pj + b @ pj

        def log_joint(x, data=data, spec=spec):
            return LJ.log_joint(x, data, spec.prior, spec.static)

        worst = max(worst, density_ratio_error(log_q, log_joint, x0, x1))
        if j == 0:
            live = np.asarray(data.tmask) > 0
            lam = np.where(live[None], 1.0 / np.where(live[None], np.asarray(inv), 1.0), 1.0)
            Kc, mc = O.ccm2019_conditional_j(
                np.asarray(Pi), np.asarray(st.eq.struct.a), lam, np.asarray(data.X),
                np.asarray(data.Y), np.asarray(c.mean0), np.asarray(c.prec0), 0, live,
            )  # fmt: skip

            def log_q_ccm(x, Kc=Kc, bc=Kc @ mc):
                pj = np.asarray(x.glob.mean.Pi[:, 0])
                return -0.5 * pj @ Kc @ pj + bc @ pj

            ccm_errors.append(density_ratio_error(log_q_ccm, log_joint, x0, x1))
    assert worst <= 1e-8, worst
    assert min(ccm_errors) > 1e-4, ccm_errors


# --------------------------------------------------------------------------- CCCMStep


@pytest.mark.parametrize("name", ["cccm_only", "reduced_sv", "reduced_sv_t"])
def test_step_sample_matches_core(name):
    """``CCCMStep.sample`` = ``cccm_core`` on (Π, a, m_t·e^{−log λ_eff}, X, Y, XX, m, p) with
    ``z_j`` from ``fold_in(fold_in(step_key, j), Sub.Z)`` (DESIGN §5.6); identical under the
    ``vmap`` and ``lax.map`` eq_maps and with or without a CoefPrior kernel."""
    spec, st, data, cache = spec_state(name, 3)
    n, k = spec.static.shapes.n, spec.static.shapes.k
    key = random.fold_in(random.fold_in(KEY, 41), int(Block.G_CCCM))
    Z = jnp.stack(
        [random.normal(random.fold_in(random.fold_in(key, j), int(Sub.Z)), (k,)) for j in range(n)]
    )
    c = spec.prior.eq.coef
    expected, einfo = CORE(
        st.glob.mean.Pi, st.eq.struct.a, reference_inv_lam(spec, st, data), data.X, data.Y,
        cache.XX, c.mean0, c.prec0, Z, gram="packed", precompute=True,
    )  # fmt: skip
    ctx = SweepCtx(data, spec.prior, cache, Empty())
    outs = []
    for kernel, eq_map in (
        (step_of(spec), jax.vmap),
        (step_of(spec, coef=CoefDouble()), jax.vmap),
        (step_of(spec, coef=CoefDouble()), lax_map),
    ):
        new, info = jax.jit(lambda k_, s_, kernel=kernel, eq_map=eq_map: kernel.sample(
            k_, s_, ctx, eq_map))(key, st)  # fmt: skip
        close(new.glob.mean.Pi, expected, 1e-13, f"{name}: sample vs core")
        assert bool(info.ok) and bool(einfo.ok)
        outs.append(np.asarray(new.glob.mean.Pi))
    np.testing.assert_allclose(outs[0], outs[2], rtol=0, atol=1e-13)


def test_step_writes_only_declared_paths():
    """Tree diff (DESIGN §4.8 item 4): only ``glob.mean.Pi`` changes, everything else is bitwise
    identical; the declared reads/writes and class variables of the GlobalStep protocol."""
    spec, st, data, cache = spec_state("reduced_sv_outliers", 5)
    kernel = step_of(spec)
    new, _ = jax.jit(lambda s: kernel.sample(KEY, s, SweepCtx(data, spec.prior, cache, Empty()),
                                             jax.vmap))(st)  # fmt: skip
    changed = set()
    for (path, a), b in zip(
        jax.tree_util.tree_flatten_with_path(st)[0], jax.tree.leaves(new), strict=True
    ):
        if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
            changed.add(leaf_path(path))
    assert changed == {"glob.mean.Pi"}
    assert kernel.block is Block.G_CCCM and kernel.phase == "pre"
    assert kernel.writes == frozenset({"glob.mean.Pi"})
    assert kernel.reads == frozenset({"glob.mean.Pi", "eq.struct.a", "eq.vol"})
    assert kernel.marginal_of_indicators is True
    assert isinstance(kernel, GlobalStep)


# Priors whose λ centre is not 1 and differs across equations (a cold start that ignored the
# prior, λ ≡ 1, or took the wrong SV branch would pass with the reference priors, whose centre
# is exactly 1); the RW case has m₀ ≠ b_μ.
_SV_CENTRES = {"eq.vol_core.b_mu": [-1.2, 0.4, 2.0], "eq.vol_core.m0": [0.8, -1.5, 1.1]}
INIT_CASES = {
    "constant": ("cccm_only", {"eq.vol_core.shape": [5.0, 3.0, 1.5],
                               "eq.vol_core.scale": [0.3, 5.0, 7.0]}),
    "iw_equivalent": ("reduced_iw_equivalent", {"eq.vol_core.scale": [0.5, 9.0, 2.5]}),
    "sv": ("reduced_sv", _SV_CENTRES),
    "sv_rw": ("reduced_sv", _SV_CENTRES),
}  # fmt: skip


def init_case(case: str):
    name, updates = INIT_CASES[case]
    kw = {"rw": True} if case == "sv_rw" else {}
    return spec_state(name, 8, prior_updates=updates, **kw)


def prior_centre(spec) -> np.ndarray:
    """DESIGN §5.4 ``λ_j = ψ_ι/ι`` written out: Constant IG mean ``scale/(shape − 1)``; SV
    ``b_μ`` (AR) or ``m₀`` (RW)."""
    core = spec.prior.eq.vol_core
    if isinstance(core, ConstantPrior):
        return np.log(np.asarray(core.scale) / (np.asarray(core.shape) - 1.0))
    return np.asarray(core.m0 if spec.static.vol.rw else core.b_mu)


def init_expected(spec, data, cache, key, log_lam0: np.ndarray) -> jax.Array:
    """One ``cccm_core`` draw at ``Π = 0``, ``A = I``, ``λ_j = exp(log_lam0_j)`` with the step's
    noise rows."""
    n, k, T_pad = spec.static.shapes.n, spec.static.shapes.k, spec.static.shapes.T_pad
    inv = LR.masked_inv_lam(jnp.asarray(np.repeat(log_lam0[:, None], T_pad, 1)), data.tmask)
    c = spec.prior.eq.coef
    out, _ = CORE(
        jnp.zeros((k, n)), jnp.zeros((n, n - 1)), inv, data.X, data.Y, cache.XX, c.mean0,
        c.prec0, LR.cccm_noise(key, n, k), gram="packed", precompute=True,
    )  # fmt: skip
    return out


@pytest.mark.parametrize("case", list(INIT_CASES))
def test_step_init(case):
    """Cold start (DESIGN §5.4): one draw with A = I and λ_j at its prior centre (Constant
    IG mean scale/(shape − 1); SV b_μ, or m₀ under RW), independent of the old Π and a."""
    spec, st, data, cache = init_case(case)
    log_lam0 = prior_centre(spec)
    assert np.min(np.abs(log_lam0)) > 0.3 and np.ptp(log_lam0) > 1.0  # the centre is informative
    if case == "sv_rw":
        core = spec.prior.eq.vol_core
        assert spec.static.vol.rw and np.all(np.abs(np.asarray(core.m0 - core.b_mu)) > 0.5)
    key = random.fold_in(KEY, 99)
    kernel = step_of(spec)
    out = jax.jit(lambda s: kernel.init(key, s, data, spec.prior, cache))(st)
    expected = init_expected(spec, data, cache, key, log_lam0)
    close(out.glob.mean.Pi, expected, 1e-13, "init")
    # the centre matters: λ ≡ 1, or the other SV branch, gives a different draw
    core = spec.prior.eq.vol_core
    others = [np.zeros_like(log_lam0)]
    if not isinstance(core, ConstantPrior):
        others.append(np.asarray(core.b_mu if spec.static.vol.rw else core.m0))
    for other in others:
        wrong = init_expected(spec, data, cache, key, other)
        assert float(jnp.max(jnp.abs(out.glob.mean.Pi - wrong))) > 1e-3
    # A = I: the draw ignores the old Π and the a currently in the state
    st2 = dataclasses.replace(
        st,
        glob=dataclasses.replace(st.glob, mean=CCCMGlobal(st.glob.mean.Pi + 5.0)),
        eq=dataclasses.replace(
            st.eq, struct=dataclasses.replace(st.eq.struct, a=st.eq.struct.a + 1)
        ),
    )
    out2 = kernel.init(key, st2, data, spec.prior, cache)
    close(out2.glob.mean.Pi, expected, 1e-13, "init independent of the old state")
    assert jax.tree.structure(out) == jax.tree.structure(st)


def _engine():
    """The WP14 modules ``(compose, runner)`` (skips while they do not import)."""
    import importlib

    return production_or_skip(
        lambda: tuple(
            importlib.import_module(f"rai_bvar.engines.mcmc.{m}") for m in ("compose", "runner")
        )
    )


@pytest.mark.parametrize("case", list(INIT_CASES))
def test_step_init_production(case):
    """The same cold start through the production assembly: the ``CCCMStep`` that ``compose``
    builds (its ``vol`` is the real ``VolBlockImpl``, which carries ``rw``) and the runner's
    ``init`` executable, whose Π is that draw with the key ``block_key(init_key, G_CCCM)``."""
    spec, st, data, cache = init_case(case)
    compose_mod, runner_mod = _engine()
    program = production_or_skip(compose_mod.compose, spec.static)
    (step,) = [g for g in program.global_pre if isinstance(g, CCCMStep)]
    log_lam0 = prior_centre(spec)
    key = random.fold_in(KEY, 98)
    out = step.init(key, st, data, spec.prior, cache)
    close(out.glob.mean.Pi, init_expected(spec, data, cache, key, log_lam0), 1e-13, "composed")
    fns = production_or_skip(runner_mod.make_runner, spec.static)
    ikeys = keys.init_keys(5, 0, 2)
    st0 = fns.init(ikeys, data, spec.prior, fns.prepare(data, spec.prior))
    for c in range(2):
        exp_c = init_expected(spec, data, cache, keys.block_key(ikeys[c], Block.G_CCCM), log_lam0)
        close(st0.glob.mean.Pi[c], exp_c, 1e-12, f"runner init, chain {c}")


def test_prior_centre_matches_runner():
    """One λ centre for every cold start (DESIGN §5.4): the reduced form's equals the runner's
    ``initial_log_variance`` (structural cold start and provisional context), including shape
    ``≤ 1`` (``C/c``) and both SV branches."""
    _, runner_mod = _engine()
    constant = J.joint_spec(
        "cccm_only",
        prior_updates={"eq.vol_core.shape": [0.5, 1.0, 3.0], "eq.vol_core.scale": [0.3, 2.0, 7.0]},
    )
    sv = J.joint_spec("reduced_sv", prior_updates=_SV_CENTRES)
    for spec in (constant, sv):
        for rw in (False, True):
            ours = LR._prior_log_lam(spec.prior.eq.vol_core, rw)
            ref = jax.vmap(lambda p, rw=rw: runner_mod.initial_log_variance(p, rw))(spec.prior.eq)
            close(ours, ref, 1e-15, f"{spec.name} rw={rw}")
    close(
        LR._prior_log_lam(constant.prior.eq.vol_core, False),
        np.log([0.3 / 0.5, 2.0 / 1.0, 7.0 / 2.0]), 1e-15, "Constant centre",
    )  # fmt: skip


def test_vol_rw_detection():
    """``rw`` is read from the ``VolBlock``'s ``VolStatic`` (``VolBlockImpl.static``), from a
    ``StaticConfig`` (``.vol``) or from a bare core's ``rw``; ``None`` means AR."""
    rw_spec = J.joint_spec("reduced_sv", rw=True)
    ar_spec = J.joint_spec("reduced_sv")
    assert LR._vol_rw(VolDouble(rw_spec.static)) is True
    assert LR._vol_rw(VolDouble(ar_spec.static)) is False
    assert LR._vol_rw(VolDouble(rw_spec.static.vol)) is True
    assert LR._vol_rw(dataclasses.make_dataclass("Core", [("rw", bool)])(True)) is True
    assert LR._vol_rw(None) is False


def test_noise_keys():
    """``cccm_noise``: row j from ``sub_key(eq_key(key, j), Sub.Z)``; distinct rows; float64."""
    key = random.fold_in(KEY, 3)
    Z = LR.cccm_noise(key, 4, 5)
    assert Z.shape == (4, 5) and Z.dtype == jnp.float64
    for j in range(4):
        ref = random.normal(random.fold_in(random.fold_in(key, j), 0), (5,), dtype=jnp.float64)
        np.testing.assert_array_equal(np.asarray(Z[j]), np.asarray(ref))
    assert len({tuple(np.asarray(z)) for z in Z}) == 4


def test_step_log_prior_matches_reference():
    """``CCCMStep.log_prior`` equals the independent reference ``log p(Π)`` (WP9)."""
    for s in range(3):
        spec, st, _, _ = spec_state("cccm_only", 20 + s)
        ref = references_for(spec.static).mean.log_prior_global(
            st.glob.mean, spec.prior, spec.static
        )
        close(step_of(spec).log_prior(st.glob, spec.prior), ref, 1e-12, "log prior")


def test_step_overrides_and_identity():
    """The ``cccm_core`` override seam (DESIGN §4.9): the validation identity override leaves Π
    unchanged (``contemp_only`` configuration); overridden kernels never compare equal to the
    production kernel; unknown names are rejected."""
    spec, st, data, cache = spec_state("cccm_only", 2)
    base = step_of(spec)
    held = base.with_overrides(J.IDENTITY_INTERNALS)
    assert held != base and hash(held) != hash(base)
    assert step_of(spec) == base and hash(step_of(spec)) == hash(base)
    new, info = held.sample(KEY, st, SweepCtx(data, spec.prior, cache, Empty()), jax.vmap)
    np.testing.assert_array_equal(np.asarray(new.glob.mean.Pi), np.asarray(st.glob.mean.Pi))
    assert bool(info.ok)
    with pytest.raises(ValueError, match="not overridable"):
        base.with_overrides({"cccm": ccm2019_core})


def test_step_construction_rules():
    spec = J.joint_spec("cccm_only")
    with pytest.raises(ValueError, match="linear_reduced"):
        CCCMStep(dataclasses.replace(spec.static.mean, kind="linear_structural"))
    with pytest.raises(TypeError, match="MeanStatic"):
        ReducedEq(spec.static)
    assert CCCMStep.from_static(spec.static, vol=VolDouble(spec.static)) == step_of(spec)
    _, st, data, cache = spec_state("cccm_only", 0)
    no_vol = CCCMStep(spec.static.mean)
    with pytest.raises(ValueError, match="VolBlock"):
        no_vol.sample(KEY, st, SweepCtx(data, spec.prior, cache, Empty()), jax.vmap)


@pytest.mark.parametrize(("gram", "tile"), [("packed_tiled", 3), ("batched", 0)])
def test_step_follows_static_gram(gram, tile):
    """The kernel passes ``MeanStatic.gram`` (and ``gram_tile``) and ``cccm_precompute`` to the
    core; every setting draws the same Π to rounding."""
    spec, st, data, cache = spec_state("reduced_sv", 6)
    ctx = SweepCtx(data, spec.prior, cache, Empty())
    ref, _ = step_of(spec).sample(KEY, st, ctx, jax.vmap)
    static = dataclasses.replace(spec.static.mean, gram=gram, gram_tile=tile, cccm_precompute=False)
    kernel = CCCMStep(static, vol=VolDouble(spec.static))
    assert kernel._core_kwargs() == (
        {"gram": gram, "precompute": False} | ({"tile": tile} if tile else {})
    )
    out, info = kernel.sample(KEY, st, ctx, jax.vmap)
    close(out.glob.mean.Pi, ref.glob.mean.Pi, 1e-12, gram)
    assert bool(info.ok)


# --------------------------------------------------------------------------- ReducedEq


def test_reduced_eq_protocol():
    """``ReducedEq`` (DESIGN §3.3, §3.7.3): no state, g ≡ 0, fitted_reduced = XΠ (zero on padded
    rows), predict_reduced = Π'x, the draw's Π in ``GlobDraw.mean``."""
    spec, st, data, cache = spec_state("reduced_sv", 1)
    n, k, T_pad = spec.static.shapes.n, spec.static.shapes.k, spec.static.shapes.T_pad
    mean = ReducedEq.from_static(spec.static)
    assert isinstance(mean, MeanBlock)
    assert mean.fuses_contemp is False and mean.batching is Batching.VMAP
    assert mean.float32_leaves == frozenset()
    Pi = st.glob.mean.Pi
    f = mean.fitted_reduced(st.glob, data)
    close(f, np.asarray(data.X) @ np.asarray(Pi), 1e-14, "XΠ")
    T_eff = int(data.T_eff)
    assert np.all(np.asarray(f)[T_eff:] == 0.0)
    a_i = st.eq.struct.a[2]
    ctx = EqContext(
        i=jnp.int32(2), target=data.Y[:, 2], cols=-data.Y[:, : n - 1], cmask=jnp.ones(n - 1),
        weights=data.tmask, log_scale=jnp.zeros(()), tmask=data.tmask,
    )  # fmt: skip
    pm = GaussianDiag(spec.prior.eq.coef.mean0[2], spec.prior.eq.coef.prec0[2], jnp.ones(k))
    ms, a_out, cpl, info = mean.sample_eq(KEY, Empty(), a_i, ctx, pm, None, Empty(), cache,
                                          spec.prior.shared)  # fmt: skip
    assert isinstance(ms, Empty) and a_out is a_i and bool(info.ok)
    assert float(cpl.n_active) == 0.0 and float(cpl.quad) == 0.0
    assert isinstance(cpl, Coupling)
    ms0, a0 = mean.init_eq(KEY, a_i, ctx, pm, pm, Empty(), cache, spec.prior.shared)
    assert isinstance(ms0, Empty) and a0 is a_i
    g = mean.fitted_eq(ms, a_i, ctx, cache, spec.prior.shared)
    assert g.shape == (T_pad,) and np.all(np.asarray(g) == 0.0)
    coefs = mean.coefficients_eq(ms, a_i, spec.prior.shared)
    assert coefs.shape == (k,) and np.all(np.asarray(coefs) == 0.0)
    assert float(mean.log_prior_eq(ms, a_i, pm, None, spec.prior.shared)) == 0.0
    cond = mean.conditional_eq(ms, a_i, ctx, pm, None, Empty(), cache, spec.prior.shared)
    assert cond.K.shape == (0, 0) and cond.b.shape == (0,)
    assert isinstance(mean.extract_eq(ms, a_i, spec.prior.shared), Empty)
    gd = mean.extract_global(st.glob)
    assert isinstance(gd, GlobDraw) and isinstance(gd.mean, CCCMDraw)
    assert all(isinstance(x, Empty) for x in (gd.coef, gd.vol, gd.struct))
    np.testing.assert_array_equal(np.asarray(gd.mean.Pi), np.asarray(Pi))
    x = jnp.asarray(rng(0).normal(size=k))
    close(mean.predict_reduced(gd, x, jnp.int32(1), Empty()), np.asarray(Pi).T @ np.asarray(x),
          1e-14, "Π'x")  # fmt: skip
    assert float(mean.predict_eq(Empty(), Empty(), x, jnp.int32(1), Empty())) == 0.0
    assert isinstance(mean.future_eq(KEY, Empty(), 4), Empty)
    assert isinstance(step_of(spec).future(KEY, gd, 4), Empty)


def test_reduced_eq_prepare():
    """``prepare`` fills the packed products of X for ``gram="packed"`` (when the base cache lacks
    them) and leaves other caches alone."""
    spec, _, data, cache = spec_state("cccm_only", 0)
    mean = ReducedEq(spec.static.mean)
    empty = dataclasses.replace(cache, XX=Empty())
    filled = mean.prepare(data, spec.prior, empty)
    np.testing.assert_array_equal(np.asarray(filled.XX), np.asarray(packed_products(data.X)))
    assert mean.prepare(data, spec.prior, cache) is cache
    batched = ReducedEq(dataclasses.replace(spec.static.mean, gram="batched"))
    assert isinstance(batched.prepare(data, spec.prior, empty).XX, Empty)


def test_registry_paths():
    assert registry.load("linear_reduced") is ReducedEq
    assert registry.load("cccm") is CCCMStep
    spec = J.joint_spec("cccm_only")
    assert registry.slot_kinds(spec.static)["global_pre"] == ("cccm",)


# --------------------------------------------------------------------------- with WP10


def _recursive():
    try:
        cls = registry.load("recursive")
        return cls.from_static
    except (NotImplementedError, AttributeError) as err:  # pragma: no cover - before WP10
        pytest.skip(f"Recursive (WP10) not available: {err}")


@pytest.mark.parametrize("name", ["cccm_only", "reduced_sv"])
def test_contemp_step_on_reduced_residuals(name):
    """Integration (WP10): ``Recursive.contemp_*`` on ``ỹ = Y − XΠ`` with ``ReducedEq``'s
    ``g ≡ 0`` is the regression of ``u_j`` on ``−u_{<j}`` (N04 §6.2, minus sign), and the vol
    blocks receive ``e = UAᵀ``."""
    make = _recursive()
    spec, st, data, cache = spec_state(name, 4)
    rec = make(spec.static)
    mean = ReducedEq(spec.static.mean)
    vol = VolDouble(spec.static)
    ytil = rec.pre_residual(st.glob, data, mean.fitted_reduced(st.glob, data))
    U = np.asarray(data.Y) - np.asarray(data.X) @ np.asarray(st.glob.mean.Pi)
    live = np.asarray(data.tmask) > 0
    U[~live] = 0.0
    close(ytil, U, 1e-13, "ỹ = U")
    inv = np.asarray(reference_inv_lam(spec, st, data))
    A = O.unit_lower(np.asarray(st.eq.struct.a))
    for i in range(spec.static.shapes.n):
        eq_i = jax.tree.map(lambda x, i=i: x[i], st.eq)
        prior_i = jax.tree.map(lambda x, i=i: x[i], spec.prior.eq)
        ctx = rec.context(jnp.int32(i), ytil, eq_i, vol, data, st.glob.vol)
        cm = rec.contemp_moments(ctx, prior_i)
        g = mean.fitted_eq(Empty(), eq_i.struct.a, ctx, cache, spec.prior.shared)
        cond = rec.contemp_conditional(ctx, g, cm)
        e = rec.residual(eq_i.struct.a, ctx, g)
        close(e, (U @ A.T)[:, i], 1e-12, f"e_{i}")
        if i == 0:
            continue
        R = -U[:, :i]
        w = inv[i]
        cp = spec.prior.eq.contemp
        Pa = np.asarray(cp.prec0[i, :i])
        K = np.diag(Pa) + R.T @ (w[:, None] * R)
        b = Pa * np.asarray(cp.mean0[i, :i]) + R.T @ (w * U[:, i])
        close(np.asarray(cond.K)[:i, :i], K, 1e-12, f"K_a{i}")
        close(np.asarray(cond.b)[:i], b, 1e-12, f"b_a{i}")


# =========================================================================== tier B


def _block_sbc_model(name: str, j: int, source: str):
    spec = J.joint_spec(name)
    static, prior = spec.static, spec.prior
    n, k = static.shapes.n, static.shapes.k
    c = prior.eq.coef

    def joint_sample(key):
        st = spec.prior_sample(random.fold_in(key, 0))
        return st, spec.simulate(random.fold_in(key, 1), st)

    def conditional_draws(key, st, data):
        inv = reference_inv_lam(spec, st, data)
        Pi = st.glob.mean.Pi
        if source in ("core", "ccm2019"):  # first update of a sweep: π_0 | π_{−0}
            Z = random.normal(key, (n, k), dtype=jnp.float64)
            XX = packed_products(data.X)
            core = LR.cccm_core if source == "core" else ccm2019_core
            out, _ = core(Pi, st.eq.struct.a, inv, data.X, data.Y, XX, c.mean0, c.prec0, Z,
                          gram="packed", precompute=True)  # fmt: skip
            pj = out[:, j]
        else:
            g = LR.cccm_conditional_j(Pi, st.eq.struct.a, inv, data.X, data.Y, c.mean0,
                                      c.prec0, j)  # fmt: skip
            pj, _ = precision_draw(g.K, g.b, random.normal(key, (k,), dtype=jnp.float64))
        return dataclasses.replace(
            st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi.at[:, j].set(pj)))
        )

    def stat(st, data):
        return jnp.concatenate([st.glob.mean.Pi[:, j], LJ.loglik(st, data, prior, static)[None]])

    names = (*(f"Pi[{r},{j}]" for r in range(k)), "loglik")
    return joint_sample, conditional_draws, stat, names


@pytest.mark.parametrize(
    ("name", "j", "source"),
    [("cccm_only", 0, "core"), ("cccm_only", 0, "conditional"), ("cccm_only", 1, "conditional"),
     ("cccm_only", 2, "conditional"), ("reduced_sv", 1, "conditional")],
)  # fmt: skip
def test_block_sbc_pi_steps(name, j, source):
    """Block-conditional SBC (N06 §5.2; DESIGN §9.4): N = 200, L = 99, ranks of π_j and of the
    log-likelihood, γ test with Bonferroni at α = 1e-3."""
    joint_sample, conditional_draws, stat, names = _block_sbc_model(name, j, source)
    res = block_sbc(joint_sample, conditional_draws, stat, N=200, L=99, seed=17 + j,
                    names=names, alpha=1e-3)  # fmt: skip
    assert res.passed, (res.reasons, res.p)


@pytest.mark.parametrize("name", ["cccm_only", "reduced_sv"])
def test_block_sbc_catches_ccm2019(name):
    """Negative control (N06 §5.2, E1b): the CCM-2019 π_0 step fails the same block SBC through
    the rank of the log-likelihood."""
    joint_sample, conditional_draws, stat, names = _block_sbc_model(name, 0, "ccm2019")
    res = block_sbc(joint_sample, conditional_draws, stat, N=200, L=99, seed=17,
                    names=names, alpha=1e-3)  # fmt: skip
    assert not res.passed and "loglik" in res.failing, (res.failing, res.p)


def _local_cccm_model(spec: J.JointSpec, kernel: CCCMStep, name: str):
    """``spec`` with its kernel replaced by one ``CCCMStep`` sweep (``cccm_only`` holds every other
    block, and CCCM is the first step of the reduced-form sweep, so this equals the production
    sweep followed by ``hold_frozen``; DESIGN §9.2)."""

    @dataclass(frozen=True, eq=False)
    class Local:
        name: str
        names: tuple[str, ...]

        def prior_sample(self, key):
            return spec.prior_sample(key)

        def simulate(self, key, st):
            return spec.simulate(key, st)

        def test_functions(self, st, data):
            return spec.test_functions(st, data)

        def step(self, key, it, st, data):
            cache = ReducedEq(spec.static.mean).prepare(
                data, spec.prior, base_cache(data, None, spec.static)
            )
            kit = random.fold_in(key, it)
            ctx = SweepCtx(data, spec.prior, cache, Empty())
            new, _ = kernel.sample(random.fold_in(kit, int(Block.G_CCCM)), st, ctx, jax.vmap)
            return J.hold_frozen(new, st, spec.live)

    return Local(name=name, names=spec.names)


G_SMALL = GewekeConfig(C=1000, M_c=100, M1=200_000, alpha=1e-3, seed=11)


@pytest.mark.geweke
def test_block_geweke_cccm_only():
    """Block Geweke ``cccm_only`` (DESIGN §9.2, §9.5 'blocks'; G-small): Π live, (a, λ) held at
    prior draws, the production ``CCCMStep`` as the kernel."""
    spec = J.joint_spec("cccm_only")
    res = geweke(_local_cccm_model(spec, step_of(spec), "cccm_only"), G_SMALL)
    assert res.passed, res.table().head(10)


@pytest.mark.geweke
def test_block_geweke_cccm_only_catches_ccm2019():
    """Power check (N06 §7): the CCM-2019 mutant through the ``cccm_core`` override fails the
    same block Geweke test (observed |z| ≈ 14)."""
    spec = J.joint_spec("cccm_only")
    mutant = step_of(spec).with_overrides({"cccm_core": ccm2019_core})
    res = geweke(_local_cccm_model(spec, mutant, "cccm_only_ccm2019"), G_SMALL)
    assert not res.passed
    assert np.max(np.abs(res.z)) > 8.0


@pytest.mark.geweke
def test_block_geweke_cccm_only_production_sweep():
    """The same configuration through ``JointSpec.step`` (production compose/sweep, WP14) once
    the engine assembly builds; skipped until then. A build error raised from this module's
    kernels (``CCCMStep``, ``ReducedEq``) is never skipped."""
    spec = J.joint_spec("cccm_only")
    production_or_skip(J.production_sweep, spec.static, spec.overrides)
    res = geweke(spec, G_SMALL)
    assert res.passed, res.table().head(10)


# =========================================================================== A7: SWA vs CTA


def a7_problem(n: int, lags: int, T: int = 60, seed: int = 0) -> O.ReducedConstant:
    """A VAR(p) dataset from the model (intercept), with an informative, equation-asymmetric
    reduced-form prior (N04 §9.6: the regime where a CCM-type error shows)."""
    r = rng(1000 * n + 10 * lags + seed)
    k = 1 + n * lags
    Pi = np.zeros((k, n))
    Pi[0] = 0.2 * r.normal(size=n)
    for lag in range(lags):
        B = 0.1 * r.normal(size=(n, n)) + np.eye(n) * (0.5 if lag == 0 else 0.1)
        Pi[1 + lag * n : 1 + (lag + 1) * n] = B.T
    cm = (np.arange(n - 1)[None] < np.arange(n)[:, None]).astype(float)
    A = O.unit_lower(0.6 * r.normal(size=(n, n - 1)) * cm)
    lam = np.exp(0.5 * r.normal(size=n))
    burn = 50
    y = np.zeros((T + burn + lags, n))
    for t in range(lags, len(y)):
        x = np.concatenate([[1.0], y[t - lags : t][::-1].reshape(-1)])
        y[t] = Pi.T @ x + np.linalg.solve(A, np.sqrt(lam) * r.normal(size=n))
    y = y[burn:]
    X = np.stack(
        [np.concatenate([[1.0], y[t - lags : t][::-1].reshape(-1)]) for t in range(lags, len(y))]
    )
    Y = y[lags:]
    sd = np.ones((n, k))
    mean0 = np.zeros((n, k))
    for j in range(n):
        for lag in range(lags):
            for v in range(n):
                sd[j, 1 + lag * n + v] = (0.3 if v == j else 0.15) / (lag + 1) * (1.0 + 0.3 * j)
        mean0[j, 1 + j] = 0.4
    a_prec = np.where(cm > 0, 1.0 / 0.7**2, 1.0)
    return O.ReducedConstant(
        X=X, Y=Y, mean0=mean0, prec0=1.0 / sd**2, a_mean0=np.zeros((n, n - 1)), a_prec0=a_prec,
        shape=np.full(n, 3.0), scale=np.full(n, 2.0),
    )  # fmt: skip


def a7_init(prob: O.ReducedConstant, C: int, seed: int):
    """Dispersed starts from the prior (shared by both samplers)."""
    r = rng(seed)
    n, k = prob.n, prob.k
    Pi = np.swapaxes(prob.mean0 + r.normal(size=(C, n, k)) / np.sqrt(prob.prec0), 1, 2)
    cm = (np.arange(n - 1)[None] < np.arange(n)[:, None]).astype(float)
    a = 0.7 * r.normal(size=(C, n, n - 1)) * cm
    log_lam = np.log(prob.scale / (prob.shape - 1.0)) + 0.5 * r.normal(size=(C, n))
    return Pi, a, log_lam


def cta_pi_step(prob: O.ReducedConstant, core=LR.cccm_core):
    """The production Π step (``cccm_core``, packed Gram, precompute) vmapped over chains."""
    X, Y = jnp.asarray(prob.X), jnp.asarray(prob.Y)
    XX = packed_products(X)
    mean0, prec0 = jnp.asarray(prob.mean0), jnp.asarray(prob.prec0)
    T = prob.X.shape[0]
    f = functools.partial(core, gram="packed", precompute=True)
    batched = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, None, None, None, None, None, 0)))

    def step(Pi, a, log_lam, Z):
        inv = jnp.broadcast_to(jnp.exp(-jnp.asarray(log_lam))[:, :, None], (*log_lam.shape, T))
        out, info = batched(jnp.asarray(Pi), jnp.asarray(a), inv, X, Y, XX, mean0, prec0,
                            jnp.asarray(Z))  # fmt: skip
        assert bool(np.all(np.asarray(info.ok)))
        return np.asarray(out)

    return step


def a7_functions(d: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, list[str]]:
    """Base functions per draw (C, D, G): every Π entry, every active a_ij, every log λ_i."""
    C, D, k, _ = d["Pi"].shape
    parts = [d["Pi"].reshape(C, D, k * n)]
    names = [f"Pi[{r},{j}]" for r in range(k) for j in range(n)]
    for i in range(1, n):
        parts.append(d["a"][:, :, i, :i])
        names += [f"a[{i},{j}]" for j in range(i)]
    parts.append(d["log_lam"])
    names += [f"log_lam[{i}]" for i in range(n)]
    return np.concatenate(parts, axis=2), names


def a7_compare(d1, d2, n: int, alpha: float = 1e-3) -> tuple[float, float, list[str]]:
    """Means, second moments and 5/50/95% quantile indicators (at the pooled quantiles) of every
    base function: chain-mean z statistics (independent chains), Bonferroni at ``alpha``.
    Returns ``(max |z|, critical value, failing names)``."""
    g1, names = a7_functions(d1, n)
    g2, _ = a7_functions(d2, n)
    pooled = np.concatenate([g1.reshape(-1, g1.shape[2]), g2.reshape(-1, g2.shape[2])])
    qs = np.quantile(pooled, [0.05, 0.5, 0.95], axis=0)

    def feats(g):
        out = [g, g**2] + [(g <= q).astype(float) for q in qs]
        return np.concatenate(out, axis=2).mean(axis=1)  # chain means (C, 5G)

    labels = [f"{t}({nm})" for t in ("mean", "sq", "q05", "q50", "q95") for nm in names]
    f1, f2 = feats(g1), feats(g2)
    se2 = f1.var(axis=0, ddof=1) / f1.shape[0] + f2.var(axis=0, ddof=1) / f2.shape[0]
    z = (f1.mean(axis=0) - f2.mean(axis=0)) / np.sqrt(se2)
    crit = float(stats.norm.isf(alpha / (2 * len(labels))))
    failing = [lab for lab, v in zip(labels, z, strict=True) if abs(v) > crit]
    return float(np.max(np.abs(z))), crit, failing


def run_a7(n: int, lags: int, *, C: int, burn: int, draws: int, seed: int = 0):
    prob = a7_problem(n, lags, seed=seed)
    init = a7_init(prob, C, seed + 1)
    swa = O.gibbs(prob, functools.partial(O.swa_pi_step, prob), init, burn=burn, draws=draws,
                  seed=seed + 2)  # fmt: skip
    cta = O.gibbs(prob, cta_pi_step(prob), init, burn=burn, draws=draws, seed=seed + 3)
    return prob, swa, cta


def test_a7_swa_vs_cta_small():
    """PR-sized A7: n = 3, p = 1, T = 60, 64 chains."""
    _, swa, cta = run_a7(3, 1, C=64, burn=100, draws=300)
    zmax, crit, failing = a7_compare(swa, cta, 3)
    assert not failing, (zmax, crit, failing)


@pytest.mark.slow
@pytest.mark.parametrize(("n", "lags"), [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)])
def test_a7_swa_vs_cta(n, lags):
    """A7 (DESIGN §9.8, N04 §9.5; nightly): reduced × Constant, SWA joint draw (NumPy oracle)
    vs the production CTA, sharing the a and λ steps; 128 independent chains × 1000 draws."""
    _, swa, cta = run_a7(n, lags, C=128, burn=200, draws=1000, seed=n + 10 * lags)
    zmax, crit, failing = a7_compare(swa, cta, n)
    assert not failing, (zmax, crit, failing)


@pytest.mark.slow
def test_a7_detects_ccm2019():
    """Power check of the A7 harness: CCM-2019 Gibbs differs from SWA-Gibbs (n = 3, p = 1)."""
    prob = a7_problem(3, 1, seed=5)
    init = a7_init(prob, 128, 6)
    swa = O.gibbs(prob, functools.partial(O.swa_pi_step, prob), init, burn=200, draws=1000, seed=7)
    ccm = O.gibbs(prob, cta_pi_step(prob, ccm2019_core), init, burn=200, draws=1000, seed=8)
    zmax, crit, failing = a7_compare(swa, ccm, 3)
    assert failing, (zmax, crit)


# --------------------------------------------------------------------------- A7 through the engine


def swa_joint_core(Pi, a, inv_lam, X, Y, XX, mean0, prec0, Z, *, gram, precompute, tile=0):
    """SWA with the ``cccm_core`` signature (N04 §3, §9.5): one joint draw of ``vec Π`` from
    ``N(Q⁻¹b, Q⁻¹)``, ``Q = blkdiag(diag p_j) + Σ_t Ω_t ⊗ x_t x_t'``,
    ``b = vec_j(p_j ⊙ m_j) + Σ_t (Ω_t y_t) ⊗ x_t`` with ``Ω_t = A'Λ_t⁻¹A`` formed explicitly
    (time-varying ``λ``; padded rows carry ``inv_lam = 0``). Ignores ``Π``. Test code: the JAX
    twin of ``swa_oracle.swa_pi_step``, swapped into the production sweep as an override."""
    k, n = Pi.shape
    A = LR.unit_lower(a)
    Om = jnp.einsum("ij,it,il->tjl", A, inv_lam, A)
    Q = jnp.einsum("tjl,tr,ts->jrls", Om, X, X).reshape(n * k, n * k)
    Q = 0.5 * (Q + Q.T) + jnp.diag(prec0.reshape(-1))
    b = (prec0 * mean0).reshape(-1) + jnp.einsum("tjl,tl,tr->jr", Om, Y, X).reshape(-1)
    theta, info = precision_draw(Q, b, Z.reshape(-1))
    return theta.reshape(n, k).T, info


def a7_engine(prob: O.ReducedConstant) -> tuple[J.JointSpec, Any]:
    """The production reduced × Constant composition (``joint_spec("reduced_constant")``) with
    the prior of ``prob`` and its data as ``EngineData``: ``T_pad = bucket(T, 32) > T``, so padded
    rows (NaN targets before masking) are in the loop; ``X`` is rebuilt by the engine from the
    series and the presample and must equal ``prob.X``."""
    n, T = prob.n, prob.Y.shape[0]
    lags = (prob.k - 1) // n
    spec = J.joint_spec(
        "reduced_constant", n=n, lags=lags, T_eff=T,
        prior_updates={
            "eq.coef.mean0": prob.mean0, "eq.coef.prec0": prob.prec0,
            "eq.contemp.mean0": prob.a_mean0, "eq.contemp.prec0": prob.a_prec0,
            "eq.vol_core.shape": prob.shape, "eq.vol_core.scale": prob.scale,
        },
    )  # fmt: skip
    T_pad = spec.static.shapes.T_pad
    assert T_pad > T
    y = np.full((T_pad, n), np.nan)
    y[:T] = prob.Y
    presample = prob.X[0, 1:].reshape(lags, n)[::-1]  # x_0 = (1, y_{−1}', …, y_{−p}')'
    data = build_engine_data(y, presample, np.int32(T), n_det=1, t0=float(lags + 1))
    np.testing.assert_array_equal(np.asarray(data.X)[:T], prob.X)
    np.testing.assert_array_equal(np.asarray(data.Y)[:T], prob.Y)
    return spec, data


def production_chains(
    spec: J.JointSpec, data: Any, *, C: int, burn: int, draws: int, seed: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:  # fmt: skip
    """The production reduced-form engine end to end (WP14): ``make_runner``'s ``prepare`` and
    ``init`` (the cold start of DESIGN §5.4, ``CCCMStep.init`` included), then ``burn + draws``
    production sweeps (``compose`` → ``make_sweep``: ``CCCMStep``, ``Recursive.contemp_step`` on
    ``ỹ = U``, the Constant core on ``e = UAᵀ``; DESIGN §4.3), vmapped over ``C`` chains with the
    runner's chain keys. ``overrides`` go to ``compose`` (e.g. ``{"cccm_core": swa_joint_core}``).
    Returns the A7 layout of ``swa_oracle.gibbs``; fails if any chain reports a failure."""
    ov = dict(overrides or {})
    _, runner_mod = _engine()
    fns = production_or_skip(runner_mod.make_runner, spec.static, overrides=ov or None)
    _, sweep = production_or_skip(J.production_sweep, spec.static, tuple(sorted(ov.items())))
    prior = spec.prior
    cache = fns.prepare(data, prior)
    st = fns.init(keys.init_keys(seed, 0, C), data, prior, cache)
    ckeys = keys.chain_keys(keys.fit_root(seed, 0), C)

    def chain(ckey, s):
        def burn_body(r, s):
            return sweep(ckey, r.astype(jnp.uint32), s, data, prior, cache)

        def body(s, it):
            s = sweep(ckey, it, s, data, prior, cache)
            return s, (s.glob.mean.Pi, s.eq.struct.a, s.eq.vol.core.log_lam)

        s = lax.fori_loop(0, burn, burn_body, s)
        s, out = lax.scan(body, s, jnp.arange(burn, burn + draws, dtype=jnp.uint32))
        return s.status.code, out

    code, (Pi, a, log_lam) = jax.jit(jax.vmap(chain))(ckeys, st)
    assert not np.any(np.asarray(code)), np.unique(np.asarray(code))
    return {"Pi": np.asarray(Pi), "a": np.asarray(a), "log_lam": np.asarray(log_lam)}


def numpy_swa_gibbs(prob: O.ReducedConstant, *, C: int, burn: int, draws: int, seed: int):
    """The NumPy SWA Gibbs (``swa_oracle.gibbs``): shares no code with the engine."""
    init = a7_init(prob, C, seed)
    step = functools.partial(O.swa_pi_step, prob)
    return O.gibbs(prob, step, init, burn=burn, draws=draws, seed=seed + 1)


def test_a7_production_small():
    """PR-sized A7 through the engine (n = 3, p = 1, T = 60, 64 chains): the production
    reduced × Constant engine (runner init, production sweep, padded rows) vs the NumPy SWA
    Gibbs, so a wiring regression anywhere in the reduced-form sweep fails every PR."""
    prob = a7_problem(3, 1)
    spec, data = a7_engine(prob)
    prod = production_chains(spec, data, C=64, burn=100, draws=300, seed=21)
    swa = numpy_swa_gibbs(prob, C=64, burn=100, draws=300, seed=22)
    zmax, crit, failing = a7_compare(swa, prod, 3)
    assert not failing, (zmax, crit, failing)


@pytest.mark.slow
@pytest.mark.parametrize(("n", "lags"), [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)])
def test_a7_production(n, lags):
    """A7 with the production sweep in the loop (DESIGN §9.8; N04 §9.5 "swap [SWA] into the full
    Gibbs sampler"; nightly). Three samplers on the same data and prior, 128 chains × 1000
    draws: (i) the engine as shipped (CTA); (ii) the same engine with SWA swapped in for the Π
    step (``cccm_core`` override :func:`swa_joint_core`); (iii) the NumPy SWA Gibbs. (i) vs (iii)
    checks the whole reduced sweep (``a | Π`` on ``ỹ = U``, ``e = UAᵀ`` to the Constant core,
    runner, padding) against code that shares nothing with it; (i) vs (ii) isolates the Π step
    inside the production sweep. Bonferroni α = 5e-4 per comparison (1e-3 per test)."""
    prob = a7_problem(n, lags, seed=n + 10 * lags)
    spec, data = a7_engine(prob)
    kw = {"C": 128, "burn": 200, "draws": 1000}
    cta = production_chains(spec, data, seed=100 + n + 10 * lags, **kw)
    swapped = production_chains(
        spec, data, seed=200 + n + 10 * lags, overrides={"cccm_core": swa_joint_core}, **kw
    )
    swa = numpy_swa_gibbs(prob, seed=300 + n + 10 * lags, **kw)
    for label, other in (("NumPy SWA Gibbs", swa), ("engine with SWA swapped in", swapped)):
        zmax, crit, failing = a7_compare(other, cta, n, alpha=5e-4)
        assert not failing, (label, zmax, crit, failing)


@pytest.mark.slow
def test_a7_production_detects_ccm2019():
    """Power of the engine-level A7: the engine with the CCM-2019 Π step (``cccm_core``
    override) differs from both SWA samplers (n = 3, p = 1, the PR-sized dataset; observed max |z|
    ≈ 100 against a critical value of 4.5)."""
    prob = a7_problem(3, 1)
    spec, data = a7_engine(prob)
    kw = {"C": 128, "burn": 200, "draws": 1000}
    ccm = production_chains(spec, data, seed=401, overrides={"cccm_core": ccm2019_core}, **kw)
    swapped = production_chains(spec, data, seed=402, overrides={"cccm_core": swa_joint_core}, **kw)
    swa = numpy_swa_gibbs(prob, seed=403, **kw)
    for label, other in (("NumPy SWA Gibbs", swa), ("engine with SWA swapped in", swapped)):
        zmax, crit, failing = a7_compare(other, ccm, 3, alpha=5e-4)
        assert failing and zmax > 2 * crit, (label, zmax, crit)


def _contemp_unweighted(ctx, g_i, cm):
    """Wiring mutant of the reduced sweep's ``a`` step (``contemp_conditional`` override): the
    ``1/λ_j`` weights dropped (N04 §6.2), invisible to the Π-step A7."""
    Z = ctx.cols * cm.cmask[None, :]
    Zw = Z * (ctx.weights > 0).astype(Z.dtype)[:, None]
    K = Zw.T @ Z + jnp.diag(cm.prec)
    K = jnp.tril(K) + jnp.tril(K, -1).T
    return GaussianPrec(K=K, b=cm.prec * cm.mean + Zw.T @ (ctx.target - g_i))


@pytest.mark.slow
def test_a7_production_detects_sweep_wiring():
    """Power of the PR-sized engine A7 against an error outside the Π step: the engine with the
    unweighted ``a`` step differs from the NumPy SWA Gibbs (observed max |z| ≈ 22)."""
    prob = a7_problem(3, 1)
    spec, data = a7_engine(prob)
    swa = numpy_swa_gibbs(prob, C=64, burn=100, draws=300, seed=22)
    bad = production_chains(
        spec, data, C=64, burn=100, draws=300, seed=21,
        overrides={"contemp_conditional": _contemp_unweighted},
    )  # fmt: skip
    zmax, crit, failing = a7_compare(swa, bad, 3)
    assert failing and zmax > 2 * crit, (zmax, crit)
