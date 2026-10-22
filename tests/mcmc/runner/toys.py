"""Toy kernels, priors, data and monitor doubles for the runner tests (DESIGN §13 WP14).

The runner, sweep, storage and warm-start code is tested against these toys so that it does not
depend on the production kernels (WP10–WP13) or monitors (WP15). The toys implement the block
protocols of ``engines/mcmc/protocols.py`` with the Phase 2 state classes:

* :class:`ToyRecursive`: the covariance structure (``a_i`` prior, contemporaneous step,
  residual algebra, ``A⁻¹`` maps), dense and simple;
* :class:`ToyLinear`: a fusing structural mean drawing ``θ_i = [δ_i | a_i]`` from its exact
  Gaussian conditional, with the Mode II / ``iw_equivalent`` coupling; with :class:`ToyVol`'s
  Constant core this is an exact Gibbs sampler of the Normal–inverse-gamma model, whose posterior
  is available in closed form (:func:`nig_posterior`);
* :class:`ToyReducedMean` + :class:`ToyCCCM`: a non-fusing mean and a pre-phase global step
  (column-wise ``Π`` draw; mechanics only, not an exact sampler);
* :class:`ToyZeroMean`; :class:`ToyFixedDiagonal`;
* :class:`ToyVol`: Constant core (exact IG conditional with coupling) or an "SV-like" core (AR
  or random walk) that exercises ``h`` paths, ``acc_*`` and ``rb_*`` accumulators and outlier
  indices (not a valid sampler of anything; mechanics only);
* :class:`DriftMean`: a deterministic mean whose prior precision turns negative at sweep 7
  (failure injection), and invariant-violating global steps.

:func:`install_monitor_doubles` stands in for ``engines.mcmc.monitor`` (WP15).
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

from rai_bvar import diagnostics
from rai_bvar.engines.mcmc import data as data_mod
from rai_bvar.engines.mcmc import keys
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StoreStatic,
    StructStatic,
    VolStatic,
    bucket,
)
from rai_bvar.engines.mcmc.keys import Block, Sub
from rai_bvar.engines.mcmc.linalg import precision_draw, solve_unit_lower
from rai_bvar.engines.mcmc.protocols import Batching, KernelBase
from rai_bvar.engines.mcmc.random import log_inv_gamma
from rai_bvar.engines.mcmc.state import (
    CCCMDraw,
    CCCMGlobal,
    CoefPriorArrays,
    ConstantDraw,
    ConstantEq,
    ConstantPrior,
    ContempPriorArrays,
    Coupling,
    Empty,
    EqContext,
    EqPrior,
    GaussianDiag,
    GaussianPrec,
    GlobDraw,
    LinearDraw,
    LinearEq,
    OutlierDraw,
    OutlierEq,
    PriorArrays,
    RecursiveEq,
    SharedPrior,
    StepInfo,
    StructDraw,
    SVDraw,
    SVEq,
    SVPrior,
    SVRWDraw,
    SVRWEq,
    TransitoryPrior,
    VolDraw,
    VolEq,
)

F64 = jnp.float64
_LOG2PI = float(np.log(2.0 * np.pi))


def _strict_lower(n: int) -> np.ndarray:
    return np.arange(n - 1)[None, :] < np.arange(n)[:, None]  # (n, n−1): j < i


def _A(a: jax.Array, n: int) -> jax.Array:
    """``A = I + strict-lower(a)`` from the padded rows ``a`` (n, n−1)."""
    full = jnp.concatenate([a, jnp.zeros((n, 1), a.dtype)], axis=1)
    return jnp.eye(n) + full * np.tril(np.ones((n, n)), -1)


# --------------------------------------------------------------------------- structure


@dataclass(frozen=True)
class ToyRecursive(KernelBase):
    static: StructStatic
    shapes: Shapes
    has_contemp: ClassVar[bool] = True
    supports_reduced_mean: ClassVar[bool] = True

    @property
    def scaled(self) -> bool:
        return self.static.contemp_scaling == "lambda"

    def init_eq(self, key, i, prior_i):
        return RecursiveEq(a=jnp.zeros(self.shapes.n - 1))

    def pre_residual(self, glob, data, f_reduced):
        ytil = data.Y if f_reduced is None else data.Y - f_reduced
        return ytil * data.tmask[:, None]

    def context(self, i, ytil, eq_i, vol, data, gvol):
        n = self.shapes.n
        lv = vol.log_var_eq(eq_i.vol, gvol)
        return EqContext(
            i=i,
            target=jax.lax.dynamic_index_in_dim(ytil, i, axis=1, keepdims=False),
            cols=-ytil[:, : n - 1],
            cmask=(jnp.arange(n - 1) < i).astype(F64),
            weights=data.tmask * jnp.exp(-lv),
            log_scale=vol.log_scale_eq(eq_i.vol) if self.scaled else jnp.zeros(()),
            tmask=data.tmask,
        )

    def contemp_moments(self, ctx, prior_i):
        c = prior_i.contemp
        prec = c.prec0
        if self.scaled:
            prec = jnp.where(c.cmask > 0, prec * jnp.exp(-ctx.log_scale), prec)
        return GaussianDiag(c.mean0, prec, c.cmask)

    def contemp_conditional(self, ctx, g_i, cm):
        Z = ctx.cols * ctx.cmask
        K = jnp.diag(cm.prec) + Z.T @ (ctx.weights[:, None] * Z)
        b = cm.prec * cm.mean + Z.T @ (ctx.weights * (ctx.target - g_i))
        return GaussianPrec(K, b)

    def contemp_step(self, key, s_i, ctx, g_i, cm):
        if self.shapes.n == 1:  # no contemporaneous coefficients
            return s_i, Coupling.zeros(), StepInfo.success()
        gp = self.impl("contemp_conditional")(ctx, g_i, cm)
        z = random.normal(keys.sub_key(key, Sub.Z), cm.mean.shape)
        a, info = precision_draw(gp.K, gp.b, z)
        a = a * cm.cmask
        if self.scaled:
            free = cm.prec * jnp.exp(ctx.log_scale)
            cpl = Coupling(jnp.sum(cm.cmask), 0.5 * jnp.sum(cm.cmask * free * (a - cm.mean) ** 2))
        else:
            cpl = Coupling.zeros()
        return RecursiveEq(a=a), cpl, info

    def residual(self, a_i, ctx, g_i):
        return (ctx.target - (ctx.cols * ctx.cmask) @ a_i - g_i) * ctx.tmask

    def log_jacobian(self, glob, T_eff):
        return jnp.zeros(())

    def extract(self, eq_struct, glob):
        return StructDraw(a=eq_struct.a)

    def mean_map(self, sd, g, f):
        return solve_unit_lower(_A(sd.a, self.shapes.n), g) + f

    def chol(self, sd, log_lam):
        return solve_unit_lower(_A(sd.a, self.shapes.n), jnp.diag(jnp.exp(0.5 * log_lam)))

    def reduced_form(self, sd, gamma):
        return solve_unit_lower(_A(sd.a, self.shapes.n), gamma).T

    def realign(self, s_host, drop_front, append):
        return s_host


# --------------------------------------------------------------------------- means


@dataclass(frozen=True)
class ToyLinear(KernelBase):
    """Fusing structural mean: exact Gaussian draw of ``θ_i = [δ_i | a_i]`` (dense)."""

    static: MeanStatic
    shapes: Shapes
    contemp_scaling: str = "none"
    fuses_contemp: ClassVar[bool] = True
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()

    def prepare(self, data, prior, base):
        return base

    def fitted_reduced(self, glob, data):
        return None

    def _theta_prior(self, ctx, pm, cm):
        k = self.shapes.k
        coef_prec = (
            pm.prec * jnp.exp(-ctx.log_scale) if self.static.coef_scaling == "lambda" else pm.prec
        )
        mean = jnp.concatenate([pm.mean, cm.mean])
        prec = jnp.concatenate([coef_prec, cm.prec])
        mask = jnp.concatenate([jnp.ones(k), cm.cmask])
        return mean, prec, mask

    def linear_conditional(self, ctx, pm, cm, cache):
        mean, prec, mask = self._theta_prior(ctx, pm, cm)
        F = cache.F
        G = F.T @ (ctx.weights[:, None] * F)
        K = jnp.outer(mask, mask) * G + jnp.diag(prec)
        b = prec * mean + mask * (F.T @ (ctx.weights * ctx.target))
        return GaussianPrec(K, b)

    def conditional_eq(self, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        return self.impl("linear_conditional")(ctx, pm, cm, cache)

    def _coupling(self, theta, ctx, pm, cm):
        k = self.shapes.k
        mean, _, mask = self._theta_prior(ctx, pm, cm)
        free = jnp.concatenate([pm.prec, cm.prec * jnp.exp(ctx.log_scale)])
        dev = mask * free * (theta - mean) ** 2
        if self.static.coef_scaling == "lambda":
            return Coupling(jnp.sum(mask), 0.5 * jnp.sum(dev))
        if self.contemp_scaling == "lambda":
            return Coupling(jnp.sum(cm.cmask), 0.5 * jnp.sum(dev[k:]))
        return Coupling.zeros()

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        k, q = self.shapes.k, self.shapes.q
        gp = self.conditional_eq(ms, a_i, ctx, pm, cm, cache_i, cache, shared)
        z = random.normal(keys.sub_key(key, Sub.Z), (q,))
        theta, info = precision_draw(gp.K, gp.b, z)
        _, _, mask = self._theta_prior(ctx, pm, cm)
        theta = theta * mask
        return LinearEq(delta=theta[:k]), theta[k:], self._coupling(theta, ctx, pm, cm), info

    def init_eq(self, key, a_i, ctx, pm, cm, cache_i, cache, shared):
        ms0 = LinearEq(delta=jnp.zeros(self.shapes.k))
        ms, a, _, _ = self.sample_eq(key, ms0, a_i, ctx, pm, cm, cache_i, cache, shared)
        return ms, a

    def fitted_eq(self, ms, a_i, ctx, cache, shared):
        X = cache.F[:, : self.shapes.k]
        return X @ (ms.delta + shared.Mc @ a_i)

    def coefficients_eq(self, ms, a_i, shared):
        return ms.delta

    def log_prior_eq(self, ms, a_i, pm, cm, shared):
        lp = -0.5 * jnp.sum(pm.prec * (ms.delta - pm.mean) ** 2 - jnp.log(pm.prec) + _LOG2PI)
        if cm is not None:
            lp += -0.5 * jnp.sum(
                cm.cmask * (cm.prec * (a_i - cm.mean) ** 2 - jnp.log(cm.prec) + _LOG2PI)
            )
        return lp

    def extract_eq(self, ms, a_i, shared):
        return LinearDraw(gamma=ms.delta + shared.Mc @ a_i)

    def extract_global(self, glob):
        return GlobDraw(Empty(), Empty(), Empty(), Empty())

    def future_eq(self, key, md, H):
        return Empty()

    def predict_eq(self, md, fut, x, h, gfut):
        return x @ md.gamma

    def predict_reduced(self, gd, x, h, gfut):
        return jnp.zeros(self.shapes.n)


@dataclass(frozen=True)
class DriftMean(ToyLinear):
    """Deterministic fusing mean (noise 0; ``δ_0`` grows by one per sweep). For equation
    ``fail_eq`` the prior precision turns negative once ``δ_0 ≥ threshold``: with the default
    threshold the conditional is not positive definite at sweep 7 exactly (failure injection)."""

    threshold: float = 7.5
    fail_eq: int = 1

    def conditional_eq(self, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        bad = (ctx.i == self.fail_eq) & (ms.delta[0] >= self.threshold)
        pm = GaussianDiag(pm.mean, jnp.where(bad, -pm.prec, pm.prec), pm.cmask)
        return super().conditional_eq(ms, a_i, ctx, pm, cm, cache_i, cache, shared)

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        k, q = self.shapes.k, self.shapes.q
        gp = self.conditional_eq(ms, a_i, ctx, pm, cm, cache_i, cache, shared)
        theta, info = precision_draw(gp.K, gp.b, jnp.zeros(q))
        _, _, mask = self._theta_prior(ctx, pm, cm)
        theta = theta * mask
        delta = theta[:k].at[0].set(ms.delta[0] + 1.0)
        return LinearEq(delta=delta), theta[k:], Coupling.zeros(), info


@dataclass(frozen=True)
class ToyZeroMean(KernelBase):
    static: MeanStatic
    shapes: Shapes
    fuses_contemp: ClassVar[bool] = False
    batching: ClassVar[Batching] = Batching.VMAP
    float32_leaves: ClassVar[frozenset[str]] = frozenset()

    def prepare(self, data, prior, base):
        return base

    def fitted_reduced(self, glob, data):
        return None

    def init_eq(self, key, a_i, ctx, pm, cm, cache_i, cache, shared):
        return Empty(), a_i

    def sample_eq(self, key, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        return Empty(), a_i, Coupling.zeros(), StepInfo.success()

    def conditional_eq(self, ms, a_i, ctx, pm, cm, cache_i, cache, shared):
        return GaussianPrec(jnp.eye(1), jnp.zeros(1))

    def fitted_eq(self, ms, a_i, ctx, cache, shared):
        return jnp.zeros(self.shapes.T_pad)

    def coefficients_eq(self, ms, a_i, shared):
        return jnp.zeros(self.shapes.k)

    def log_prior_eq(self, ms, a_i, pm, cm, shared):
        return jnp.zeros(())

    def extract_eq(self, ms, a_i, shared):
        return Empty()

    def extract_global(self, glob):
        return GlobDraw(Empty(), Empty(), Empty(), Empty())

    def future_eq(self, key, md, H):
        return Empty()

    def predict_eq(self, md, fut, x, h, gfut):
        return jnp.zeros(())

    def predict_reduced(self, gd, x, h, gfut):
        return jnp.zeros(self.shapes.n)


@dataclass(frozen=True)
class ToyReducedMean(ToyZeroMean):
    """Reduced-form per-equation mean: ``g_i = 0``, ``ỹ = Y − XΠ``."""

    def fitted_reduced(self, glob, data):
        return data.X @ glob.mean.Pi

    def extract_global(self, glob):
        return GlobDraw(CCCMDraw(Pi=glob.mean.Pi), Empty(), Empty(), Empty())

    def predict_reduced(self, gd, x, h, gfut):
        return x @ gd.mean.Pi


@dataclass(frozen=True)
class ToyCCCM(KernelBase):
    """Pre-phase ``Π`` step: column ``j`` from ``N`` given ``λ_j`` with ``A = I`` (mechanics)."""

    static: MeanStatic
    shapes: Shapes
    vol: Any
    block: ClassVar[Block] = Block.G_CCCM
    phase: ClassVar[str] = "pre"
    reads: ClassVar[frozenset[str]] = frozenset({"glob.mean.Pi", "eq.struct.a", "eq.vol"})
    writes: ClassVar[frozenset[str]] = frozenset({"glob.mean.Pi"})
    marginal_of_indicators: ClassVar[bool] = True

    def cccm_core(self, Pi, inv_lam, X, Y, mean0, prec0, Z):
        def column(w, y, m, p, z):
            K = X.T @ (w[:, None] * X) + jnp.diag(p)
            b = p * m + X.T @ (w * y)
            return precision_draw(K, b, z)

        cols, info = jax.vmap(column)(inv_lam, Y.T, mean0, prec0, Z)
        return cols.T, StepInfo(jnp.all(info.ok), jnp.min(info.min_pivot))

    def prior_prec(self, Pi, prior):
        """Prior precisions of the columns ``π_j`` (n, k) at the current ``Π`` (hook)."""
        return prior.eq.coef.prec0

    def moments(self, st, prior, eq_map=None):
        """``(m_j, p_j)`` of every column, as ``CCCMStep.moments``."""
        c = prior.eq.coef
        prec = self.prior_prec(st.glob.mean.Pi, prior)
        return GaussianDiag(c.mean0, prec, jnp.ones_like(c.prec0))

    def inv_lam(self, st, tmask, eq_map=None):
        """``m_t exp(−log λ_eff,j,t)`` (n, T_pad) from the current volatility state."""
        eq_map = eq_map or jax.vmap
        T = self.shapes.T_pad
        lv = eq_map(lambda vs: jnp.broadcast_to(self.vol.log_var_eq(vs, st.glob.vol), (T,)))(
            st.eq.vol
        )
        return tmask * jnp.exp(-lv)

    def conditional_j(self, st, ctx, j, eq_map=None):
        """The exact conditional of column ``j`` (``A = I``), as ``CCCMStep.conditional_j``."""
        X = ctx.data.X
        w = self.inv_lam(st, ctx.data.tmask, eq_map)[j]
        p = self.prior_prec(st.glob.mean.Pi, ctx.prior)[j]
        K = X.T @ (w[:, None] * X) + jnp.diag(p)
        b = p * ctx.prior.eq.coef.mean0[j] + X.T @ (w * ctx.data.Y[:, j])
        return GaussianPrec(K, b)

    def _noise(self, key):
        n, k = self.shapes.n, self.shapes.k
        return jax.vmap(lambda j: random.normal(keys.sub_key(keys.eq_key(key, j), Sub.Z), (k,)))(
            jnp.arange(n, dtype=jnp.int32)
        )

    def _draw(self, key, Pi, inv_lam, data, prior):
        return self.impl("cccm_core")(
            Pi, inv_lam, data.X, data.Y, prior.eq.coef.mean0, self.prior_prec(Pi, prior),
            self._noise(key),
        )  # fmt: skip

    def sample(self, key, st, ctx, eq_map):
        inv_lam = self.inv_lam(st, ctx.data.tmask, eq_map)
        Pi, info = self._draw(key, st.glob.mean.Pi, inv_lam, ctx.data, ctx.prior)
        return dataclasses.replace(
            st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=Pi))
        ), info

    def init(self, key, st, data, prior, cache):
        inv_lam = jnp.broadcast_to(data.tmask, (self.shapes.n, self.shapes.T_pad))
        Pi, _ = self._draw(key, st.glob.mean.Pi, inv_lam, data, prior)
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi=Pi)))

    def log_prior(self, glob, prior):
        return jnp.zeros(())

    def future(self, key, gd, H):
        return Empty()


@dataclass(frozen=True)
class DriftCCCM(ToyCCCM):
    """Deterministic CCCM step (noise 0; ``Π_00`` grows by one per sweep, starting at 1 after the
    cold start). Once ``Π_00 ≥ threshold`` the prior precision of column ``fail_j`` becomes
    ``−1e6·p``, so ``K_j`` is not positive definite: with the default threshold the step fails at
    sweep 7 exactly, in column 1 (failure injection for a global step)."""

    threshold: float = 7.5
    fail_j: int = 1

    def prior_prec(self, Pi, prior):
        p = prior.eq.coef.prec0
        bad = (Pi[0, 0] >= self.threshold) & (jnp.arange(p.shape[0]) == self.fail_j)
        return jnp.where(bad[:, None], -1e6 * p, p)

    def _noise(self, key):
        return jnp.zeros((self.shapes.n, self.shapes.k))

    def sample(self, key, st, ctx, eq_map):
        Pi0 = st.glob.mean.Pi
        st, info = super().sample(key, st, ctx, eq_map)
        Pi = st.glob.mean.Pi.at[0, 0].set(Pi0[0, 0] + 1.0)
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi))), info

    def init(self, key, st, data, prior, cache):
        st = super().init(key, st, data, prior, cache)
        Pi = st.glob.mean.Pi.at[0, 0].set(1.0)
        return dataclasses.replace(st, glob=dataclasses.replace(st.glob, mean=CCCMGlobal(Pi)))


@dataclass(frozen=True)
class ToyFixedDiagonal(KernelBase):
    static: CoefPriorStatic
    shapes: Shapes
    varies: ClassVar[bool] = False

    def init_eq(self, key, i, prior_i):
        return Empty()

    def moments_eq(self, cs_i, i, prior_i, glob_coef):
        c = prior_i.coef
        return GaussianDiag(c.mean0, c.prec0, jnp.ones_like(c.mean0))

    def sample_local_eq(self, key, cs_i, coef_i, prior_i, glob_coef):
        return cs_i

    def log_prior_eq(self, cs_i, coef_i, prior_i, glob_coef):
        c = prior_i.coef
        return -0.5 * jnp.sum(c.prec0 * (coef_i - c.mean0) ** 2 - jnp.log(c.prec0) + _LOG2PI)

    def global_steps(self):
        return ()


# --------------------------------------------------------------------------- volatility


@dataclass(frozen=True)
class ToyVol(KernelBase):
    """Constant core (exact IG conditional with the coupling) or an SV-like mechanics core,
    optionally with outlier indices."""

    static: VolStatic
    shapes: Shapes

    @property
    def sv(self) -> bool:
        return self.static.persistent == "sv"

    @property
    def rw(self) -> bool:
        return self.sv and self.static.rw

    @property
    def outliers(self) -> bool:
        return self.static.transitory == "outlier"

    def constant_weights(self):
        return self.static.constant_weights

    @staticmethod
    def _log_e2(resid, tmask):
        return jnp.where(tmask > 0, jnp.log(jnp.where(tmask > 0, resid**2, 1.0)), 0.0)

    def init_eq(self, key, resid0, tmask, T_eff, prior_i):
        T = self.shapes.T_pad
        lv = jnp.log(jnp.sum(tmask * resid0**2) / T_eff)
        if self.rw:
            z, iz = jnp.zeros(()), jnp.zeros((), jnp.int32)
            core = SVRWEq(
                h=jnp.full(T + 1, lv), sigma=jnp.asarray(0.3), acc_h=z, acc_cs2=z, acc_ncms=z,
                acc_fold=z, acc_floor=z, acc_rej_run=iz, acc_rej_run_max=iz,
                rb_h_sum=jnp.zeros(T + 1), rb_h_sq=jnp.zeros(T + 1),
            )  # fmt: skip
        elif self.sv:
            z = jnp.zeros(())
            core = SVEq(
                h=jnp.full(T + 1, lv), mu=lv, phi=jnp.asarray(0.5), sigma=jnp.asarray(0.3),
                acc_h=z, acc_cgp=z, acc_cs2=z, acc_ncphi=z, acc_ncms=z, acc_fold=z, acc_floor=z,
                acc_rej_run=jnp.zeros((), jnp.int32), acc_rej_run_max=jnp.zeros((), jnp.int32),
                rb_h_sum=jnp.zeros(T + 1), rb_h_sq=jnp.zeros(T + 1),
            )  # fmt: skip
        else:
            core = ConstantEq(log_lam=lv)
        trans = (
            OutlierEq(
                o_idx=jnp.zeros(T, jnp.int32),
                log_p=jnp.log(0.1),
                log1m_p=jnp.log(0.9),
                rb_out=jnp.zeros(T),
            )
            if self.outliers
            else Empty()
        )
        return VolEq(core=core, trans=trans)

    def constant_conditional(self, resid, tmask, T_eff, prior_i, cpl):
        cp = prior_i.vol_core
        shape = cp.shape + 0.5 * T_eff + 0.5 * cpl.n_active
        scale = cp.scale + 0.5 * jnp.sum(tmask * resid**2) + cpl.quad
        return shape, scale

    def sample_eq(self, key, vs, resid, tmask, T_eff, prior_i, cpl, gvol):
        T = self.shapes.T_pad
        trans = vs.trans
        log_s2 = jnp.zeros(T)
        if self.outliers:
            u = random.uniform(keys.sub_key(key, Sub.O), (T,))
            o_idx = jnp.where(tmask > 0, (u < 0.1).astype(jnp.int32), 0)
            trans = OutlierEq(o_idx, trans.log_p, trans.log1m_p, trans.rb_out + 0.1 * tmask)
            log_s2 = 2.0 * jnp.log1p(o_idx.astype(F64))
        if not self.sv:
            shape, scale = self.impl("constant_conditional")(resid, tmask, T_eff, prior_i, cpl)
            log_lam = log_inv_gamma(keys.sub_key(key, Sub.S), shape, scale)
            return VolEq(core=ConstantEq(log_lam=log_lam), trans=trans), StepInfo.success()
        c = vs.core
        log_e2 = self._log_e2(resid, tmask) - log_s2
        level = jnp.sum(tmask * log_e2) / T_eff
        z = random.normal(keys.sub_key(key, Sub.H), (T + 1,))
        accept = random.uniform(keys.sub_key(key, Sub.H_ACC), ()) < 0.8
        if self.rw:  # random walk: no level, no persistence
            prop = jnp.where(jnp.arange(T + 1) <= T_eff, c.h + c.sigma * z, z)
            h = jnp.where(accept, prop, c.h)
            acc = accept.astype(F64)
            rej = (c.acc_rej_run + 1) * (1 - accept.astype(jnp.int32))
            core = dataclasses.replace(
                c, h=h, acc_h=c.acc_h + acc, acc_cs2=c.acc_cs2 + 0.5, acc_rej_run=rej,
                acc_rej_run_max=jnp.maximum(c.acc_rej_run_max, rej),
                rb_h_sum=c.rb_h_sum + h, rb_h_sq=c.rb_h_sq + h**2,
            )  # fmt: skip
            return VolEq(core=core, trans=trans), StepInfo.success()
        prop = c.mu + c.phi * (c.h - c.mu) + c.sigma * z
        prop = jnp.where(jnp.arange(T + 1) <= T_eff, prop, z)  # padded states: N(0, 1) dummies
        h = jnp.where(accept, prop, c.h)
        mu = 0.8 * c.mu + 0.2 * level + 0.05 * random.normal(keys.sub_key(key, Sub.NC_MS), ())
        acc = accept.astype(F64)
        rej = (c.acc_rej_run + 1) * (1 - accept.astype(jnp.int32))
        core = dataclasses.replace(
            c, h=h, mu=mu, acc_h=c.acc_h + acc, acc_cgp=c.acc_cgp + 1.0, acc_cs2=c.acc_cs2 + 0.5,
            acc_rej_run=rej, acc_rej_run_max=jnp.maximum(c.acc_rej_run_max, rej),
            rb_h_sum=c.rb_h_sum + h, rb_h_sq=c.rb_h_sq + h**2,
        )  # fmt: skip
        return VolEq(core=core, trans=trans), StepInfo.success()

    def _log_s2(self, vs):
        if not self.outliers:
            return jnp.zeros(self.shapes.T_pad)
        return 2.0 * jnp.log1p(vs.trans.o_idx.astype(F64))

    def log_var_eq(self, vs, gvol, which="effective"):
        T = self.shapes.T_pad
        pers = vs.core.h[1:] if self.sv else jnp.broadcast_to(vs.core.log_lam, (T,))
        return pers if which == "persistent" else pers + self._log_s2(vs)

    def log_scale_eq(self, vs):
        if self.sv or not self.static.coupled:
            return jnp.zeros(())
        return vs.core.log_lam

    def loglik_eq(self, vs, resid, tmask, gvol):
        lv = self.log_var_eq(vs, gvol)
        return -0.5 * jnp.sum(tmask * (_LOG2PI + lv + resid**2 * jnp.exp(-lv)))

    def obs_logdensity_eq(self, e, log_lam_pers, vd):
        return -0.5 * (_LOG2PI + log_lam_pers + e**2 * jnp.exp(-log_lam_pers))

    def transition_eq(self, key, log_lam_prev, vd):
        return log_lam_prev

    def extract_eq(self, vs, T_eff):
        if self.rw:
            c = vs.core
            h_T = jax.lax.dynamic_index_in_dim(c.h, T_eff, keepdims=False)
            core = SVRWDraw(h_T=h_T, sigma=c.sigma)
        elif self.sv:
            c = vs.core
            core = SVDraw(h_T=jax.lax.dynamic_index_in_dim(c.h, T_eff, keepdims=False), mu=c.mu, phi=c.phi,
                          sigma=c.sigma)  # fmt: skip
        else:
            core = ConstantDraw(log_lam=vs.core.log_lam)
        trans = OutlierDraw(vs.trans.log_p, vs.trans.log1m_p) if self.outliers else Empty()
        return VolDraw(core=core, trans=trans)

    def draw_ahead_eq(self, key, vd, prior_i, H, gfut):
        lv = jnp.full(H, vd.core.h_T if self.sv else vd.core.log_lam)
        return lv, lv

    def log_prior_eq(self, vs, prior_i, tmask):
        return jnp.zeros(())


@dataclass(frozen=True)
class ToyCore(KernelBase):
    """Placeholder for the persistent-core and transitory registry kinds (``ToyVol`` is monolithic)."""

    static: VolStatic


# --------------------------------------------------------------------------- bad global steps


@dataclass(frozen=True)
class LeakyPreStep(KernelBase):
    """Declares ``writes = {glob.mean}`` but also changes ``eq.struct.a`` (tree-diff violation)."""

    block: ClassVar[Block] = Block.G_COEF_GLOBAL
    phase: ClassVar[str] = "pre"
    reads: ClassVar[frozenset[str]] = frozenset({"eq.struct.a"})
    writes: ClassVar[frozenset[str]] = frozenset({"glob.mean"})
    marginal_of_indicators: ClassVar[bool] = True

    def sample(self, key, st, ctx, eq_map):
        struct = dataclasses.replace(st.eq.struct, a=st.eq.struct.a * 0.5)
        return dataclasses.replace(
            st, eq=dataclasses.replace(st.eq, struct=struct)
        ), StepInfo.success()

    def init(self, key, st, data, prior, cache):
        return st

    def log_prior(self, glob, prior):
        return jnp.zeros(())

    def future(self, key, gd, H):
        return Empty()


@dataclass(frozen=True)
class PostMeanStep(LeakyPreStep):
    """A post step that declares a write of a residual input (static violation, §4.8 item 1)."""

    block: ClassVar[Block] = Block.G_LP_SCALE
    phase: ClassVar[str] = "post"
    writes: ClassVar[frozenset[str]] = frozenset({"eq.mean"})

    def sample(self, key, st, ctx, eq_map):
        return st, StepInfo.success()


@dataclass(frozen=True)
class NotMarginalStep(LeakyPreStep):
    marginal_of_indicators: ClassVar[bool] = False


# --------------------------------------------------------------------------- configurations


TOY_KINDS = {
    "recursive": ToyRecursive,
    "linear_structural": ToyLinear,
    "linear_reduced": ToyReducedMean,
    "cccm": ToyCCCM,
    "zero": ToyZeroMean,
    "fixed_diagonal": ToyFixedDiagonal,
    "constant": ToyCore,
    "sv": ToyCore,
    "none": ToyCore,
    "outlier": ToyCore,
    "t": ToyCore,
    "outlier_t": ToyCore,
    "vol_block": ToyVol,
}
"""Registry kinds → toy classes (for tests that monkeypatch ``registry.load``)."""


def toy_static(
    mean: str = "linear_structural",
    persistent: str = "constant",
    *,
    transitory: str = "none",
    mode: str = "I",
    n: int = 2,
    lags: int = 1,
    n_det: int = 1,
    T_eff: int = 40,
    chunk: int = 50,
    paths: bool = False,
    batching: str = "vmap",
    rw: bool = False,
) -> StaticConfig:
    vol = VolStatic(persistent=persistent, rw=rw, transitory=transitory, coupled=mode != "I")
    return StaticConfig(
        shapes=Shapes.of(n, lags, n_det, bucket(T_eff, 32)),
        struct=StructStatic(contemp_scaling="none" if mode == "I" else "lambda"),
        mean=MeanStatic(kind=mean, coef_scaling="lambda" if mode == "II" else "none"),
        coef=CoefPriorStatic(),
        vol=vol,
        store=StoreStatic(paths=paths),
        batching=batching,
        chunk=chunk,
    )


def toy_overrides(static: StaticConfig, *, mean: Any = None) -> dict[str, Any]:
    """Slot overrides that replace every production kernel of ``static`` by a toy."""
    s = static.shapes
    vol = ToyVol(static.vol, s)
    kind = static.mean.kind
    if mean is None:
        mean = {
            "linear_structural": ToyLinear(static.mean, s, static.struct.contemp_scaling),
            "linear_reduced": ToyReducedMean(static.mean, s),
            "zero": ToyZeroMean(static.mean, s),
        }[kind]
    out = {
        "structure": ToyRecursive(static.struct, s),
        "mean": mean,
        "coef": ToyFixedDiagonal(static.coef, s),
        "vol": vol,
        "global_pre": (ToyCCCM(static.mean, s, vol),) if kind == "linear_reduced" else (),
    }
    return out


_RUNNERS: dict[Any, Any] = {}


def cached_runner(
    static: StaticConfig, overrides: dict[str, Any], *, fit_axis: bool = False, chains: str = "vmap"
) -> Any:
    """``make_runner(static, overrides=…)`` memoised by value across tests (overrides bypass the
    production cache; compiling every toy program once keeps the suite fast). Programs traced
    here record monitors through the doubles (the conftest installs them for every test)."""
    from rai_bvar.engines.mcmc.runner import make_runner

    key = (static, fit_axis, chains, tuple(sorted(overrides.items(), key=lambda kv: kv[0])))
    if key not in _RUNNERS:
        _RUNNERS[key] = make_runner(static, fit_axis=fit_axis, overrides=overrides, chains=chains)
    return _RUNNERS[key]


def toy_prior(static: StaticConfig, *, tight: bool = False) -> PriorArrays:
    s, vol = static.shapes, static.vol
    n, k, lags, n_det = s.n, s.k, s.lags, s.n_det
    mean0 = np.zeros((n, k))
    for i in range(n):
        mean0[i, n_det + i] = 0.3  # own first lag
    prec0 = np.full((n, k), 4.0 if tight else 1.0)
    prec0[:, :n_det] = 0.25
    cm = _strict_lower(n).astype(np.float64)
    contemp = ContempPriorArrays(
        mean0=np.zeros((n, n - 1)), prec0=np.where(cm > 0, 2.0, 1.0), cmask=cm
    )
    if vol.persistent == "constant":
        core = ConstantPrior(shape=np.full(n, 3.0), scale=np.full(n, 2.0))
    else:
        f = np.full
        core = SVPrior(
            b_mu=f(n, 0.0), B_mu=f(n, 1.0), a_phi=f(n, 10.0), b_phi=f(n, 2.0), B_sigma=f(n, 0.1),
            c0=f(n, 2.5), C0=f(n, 0.2), B0=f(n, 1.0), m0=f(n, 0.0), V0=f(n, 1.0),
            log_c=f(n, np.log(1e-12)),
        )  # fmt: skip
    trans = (
        Empty()
        if vol.transitory == "none"
        else TransitoryPrior(
            a_p=np.full(n, 2.0),
            b_p=np.full(n, 18.0),
            log_nu_w=np.full((n, len(vol.nu_grid)), -np.log(len(vol.nu_grid))),
        )
    )
    Mc = np.zeros((k, n - 1))
    if static.mean.kind == "linear_structural":
        for j in range(n - 1):
            Mc[n_det + j, j] = 0.3
    del lags
    return PriorArrays(
        eq=EqPrior(
            coef=CoefPriorArrays(mean0=mean0, prec0=prec0, group=np.zeros((n, k), np.int32)),
            contemp=contemp,
            vol_core=core,
            transitory=trans,
        ),
        shared=SharedPrior(Mc=Mc),
    )


def simulate_y(n: int, T: int, seed: int, *, scale: float = 0.5) -> np.ndarray:
    """``T + 1`` observations of the stable VAR(1) ``y_t = 0.3 y_{t−1} + scale·ε_t``."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 1, n))
    for t in range(1, T + 1):
        y[t] = 0.3 * y[t - 1] + scale * rng.standard_normal(n)
    return y


def toy_data(
    static: StaticConfig, T_eff: int, seed: int = 0, *, y: np.ndarray | None = None
) -> Any:
    """``EngineData`` (host NumPy leaves) of a simulated window with ``T_eff`` rows."""
    s = static.shapes
    if y is None:
        y = simulate_y(s.n, T_eff + s.lags, seed)
    presample, rows = y[: s.lags], y[s.lags : s.lags + T_eff]
    y_pad = np.zeros((s.T_pad, s.n))
    y_pad[:T_eff] = rows
    d = data_mod.build_engine_data(
        y_pad, presample, np.int32(T_eff), n_det=s.n_det, t0=float(s.lags + 1)
    )
    return jax.tree.map(np.asarray, d)


def simulate_correlated_y(
    T: int, seed: int, *, a10: float = -0.8, scale: float = 1.0
) -> np.ndarray:
    """``T + 1`` observations of a bivariate VAR(1) ``y_t = 0.3 y_{t−1} + A⁻¹ε_t`` with the
    recursive structure ``A = [[1, 0], [a10, 1]]``, ``ε ~ N(0, scale²·diag(0.25, 0.04))``: equation
    1 has ``u_1 = −a10·u_0 + ε_1``, a strong contemporaneous dependence (corr(u_0, u_1) ≈ 0.89)."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T + 1, 2))
    for t in range(1, T + 1):
        e = scale * np.array([0.5, 0.2]) * rng.standard_normal(2)
        u = np.array([e[0], e[1] - a10 * e[0]])  # A u = ε, A unit lower triangular
        y[t] = 0.3 * y[t - 1] + u
    return y


def periods(T: int, start: str = "1990-01", freq: str = "M") -> pd.PeriodIndex:
    return pd.period_range(start, periods=T, freq=freq)


# --------------------------------------------------------------------------- forecast core double (WP17)


def toy_forecast_core(draw, hist, t_last, shocks, program):
    """Test double of ``engines.mcmc.forecast.forecast_core`` (WP17; DESIGN §7.2): the documented
    recursion ``y = mean_map(sd, Γx, Π'x) + chol(sd, log λ) ε`` with the program's kernels, and a
    toy volatility path ``log λ_{T+h} = base + 0.1 Σ_{j≤h} ν_j`` whose ``ν`` comes from the
    injected per-equation volatility keys (so both key kinds are exercised)."""
    s = program.static.shapes
    n, H = s.n, shocks.eps.shape[0]
    core = draw.vol.core
    base = core.h_T if hasattr(core, "h_T") else core.log_lam  # (n,)
    nu = jax.vmap(lambda k: random.normal(k, (H,), dtype=F64))(shocks.vol_keys).T  # (H, n)
    log_lam = base[None, :] + 0.1 * jnp.cumsum(nu, axis=0)
    gamma = None if isinstance(draw.mean, Empty) else draw.mean.gamma
    Pi = None if isinstance(draw.glob.mean, Empty) else draw.glob.mean.Pi

    def step(window, h):
        x = data_mod.regressors_from_window(window, t_last + h.astype(F64), s.n_det)
        g = jnp.zeros(n) if gamma is None else gamma @ x
        f = jnp.zeros(n) if Pi is None else x @ Pi
        m = program.structure.mean_map(draw.struct, g, f)
        L = program.structure.chol(draw.struct, log_lam[h - 1])
        y = m + L @ shocks.eps[h - 1]
        return jnp.concatenate([window[1:], y[None, :]], axis=0), (y, m, L)

    _, (paths, cond_mean, cond_chol) = jax.lax.scan(step, hist, jnp.arange(1, H + 1))
    return {"paths": paths, "cond_mean": cond_mean, "cond_chol": cond_chol, "log_lam": log_lam}


# --------------------------------------------------------------------------- monitor doubles (WP15)


def toy_monitor_values(st, data, prior, cache, program):
    """Test double of ``monitor.monitor_values``: a few per-equation quantities."""
    T = program.static.shapes.T_pad
    lv = program.eq_map(lambda vs: jnp.broadcast_to(program.vol.log_var_eq(vs, st.glob.vol), (T,)))(
        st.eq.vol
    )
    m = data.tmask
    out = {"log_var_mean": (lv * m).sum(axis=1) / m.sum(), "a": st.eq.struct.a}
    if isinstance(st.eq.mean, LinearEq):
        out["delta0"] = st.eq.mean.delta[:, 0]
    if isinstance(st.glob.mean, CCCMGlobal):
        out["Pi_row0"] = st.glob.mean.Pi[0, :]
    return out


def toy_monitor_spec(static, variables):
    return types.SimpleNamespace(primary=("log_var_mean",), secondary=("a",))


def _columns(mon, names):
    for name in names:
        arr = np.asarray(mon[name])
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for j in range(flat.shape[-1]):
            yield f"{name}[{j}]", flat[..., j]


def toy_converged(mon, spec, settings, *, error, eb_converged):
    msgs = []
    if error:
        return False, ["a chain failed"]
    first = next(iter(mon.values()))
    if np.shape(first)[0] < 2:
        return False, ["one chain: split R-hat cannot detect between-chain disagreement"]
    ok = True
    for name, x in _columns(mon, spec.primary):
        r, e = diagnostics.rhat(x), diagnostics.ess_bulk(x)
        if not (r < settings.rhat_max and e > settings.ess_min):
            ok = False
            msgs.append(f"{name}: R-hat {r:.3f}, ESS {e:.0f}")
    if eb_converged is False:
        ok = False
    return ok, msgs


def toy_diagnostics_table(mon, spec, acc):
    rows = []
    for primary, names in ((True, spec.primary), (False, spec.secondary)):
        for name, x in _columns(mon, names):
            rows.append(
                {
                    "quantity": name,
                    "mean": x.mean(),
                    "sd": x.std(),
                    "rhat": diagnostics.rhat(x),
                    "ess_bulk": diagnostics.ess_bulk(x),
                    "primary": primary,
                }
            )
    return pd.DataFrame(rows).set_index("quantity")


def toy_warning_catalogue(inputs):
    return toy_converged(
        inputs.mon,
        inputs.spec,
        inputs.settings,
        error=inputs.error,
        eb_converged=inputs.eb_converged,
    )[1]


def _real_monitor() -> dict[str, Any]:
    from rai_bvar.engines.mcmc import monitor

    names = (
        "monitor_values",
        "monitor_spec",
        "converged",
        "diagnostics_table",
        "warning_catalogue",
    )
    return {name: getattr(monitor, name, None) for name in names}


REAL_MONITOR: dict[str, Any] = _real_monitor()
"""The production monitor functions (WP15) as imported, before any test installs the doubles."""

MONITOR_DOUBLES = {
    "monitor_values": toy_monitor_values,
    "monitor_spec": toy_monitor_spec,
    "converged": toy_converged,
    "diagnostics_table": toy_diagnostics_table,
    "warning_catalogue": toy_warning_catalogue,
}


def install_monitor_doubles(setattr_fn=None) -> None:
    """Install the doubles on ``engines.mcmc.monitor`` (``setattr_fn``: e.g. ``monkeypatch.setattr``
    bound with ``raising=False``; default: plain ``setattr``, for subprocesses)."""
    from rai_bvar.engines.mcmc import monitor

    for name, fn in MONITOR_DOUBLES.items():
        if setattr_fn is None:
            setattr(monitor, name, fn)
        else:
            setattr_fn(monitor, name, fn)


# --------------------------------------------------------------------------- closed-form posterior


def nig_posterior(
    static: StaticConfig, data: Any, prior: PriorArrays
) -> list[dict[str, np.ndarray]]:
    """Exact posterior of the Mode II toy (Normal–inverse-gamma per equation, DESIGN §4.7):
    ``θ_i | λ_i, y ~ N(m̂, λ_i K̃⁻¹)``, ``λ_i | y ~ IG(c + T/2, C + S/2)``,
    ``S = y'y + m'P̃m − m̂'K̃m̂``. Returns per equation ``E λ``, ``E θ`` and ``sd θ`` (active)."""
    s = static.shapes
    n, k = s.n, s.k
    T = int(data.T_eff)
    Y, X = np.asarray(data.Y)[:T], np.asarray(data.X)[:T]
    Mc = np.asarray(prior.shared.Mc)
    out = []
    for i in range(n):
        F = np.concatenate([X, X @ Mc[:, :i] - Y[:, :i]], axis=1)
        m = np.concatenate([prior.eq.coef.mean0[i], prior.eq.contemp.mean0[i, :i]])
        P = np.concatenate([prior.eq.coef.prec0[i], prior.eq.contemp.prec0[i, :i]])
        y = Y[:, i]
        K = np.diag(P) + F.T @ F
        mhat = np.linalg.solve(K, P * m + F.T @ y)
        S = y @ y + m @ (P * m) - mhat @ K @ mhat
        a_post = prior.eq.vol_core.shape[i] + T / 2
        b_post = prior.eq.vol_core.scale[i] + S / 2
        e_lam = b_post / (a_post - 1)
        cov = e_lam * np.linalg.inv(K)  # Student-t covariance: (b/(a−1)) K⁻¹
        out.append({"E_lam": e_lam, "E_theta": mhat, "sd_theta": np.sqrt(np.diag(cov)), "q": k + i})
    return out
