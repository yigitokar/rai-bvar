"""Monitored sets, ``monitor_values``, the ``converged`` flag, the warning catalogue and the
diagnostics table (DESIGN §8; WP15).

The Phase 2 kernels (WP10–WP13) and ``compose`` (WP14) are written in parallel, so
``monitor_values`` is driven here by small test-double kernels that implement exactly the protocol
methods it calls (DESIGN §3.2–§3.6) with the documented algebra, and is checked against a dense
NumPy reference computed from the raw state arrays: ``e_t = A y_t − Γ x_t`` (structural),
``A(y_t − Π'x_t)`` (reduced) or ``A y_t`` (zero mean), ``B = (A⁻¹Γ)'`` or ``Π``, and the log
variances ``h_{t+1}`` or ``log λ`` plus ``2 log o_t + log τ_t``.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import lax

from rai_bvar import diagnostics as dg
from rai_bvar.engines.mcmc import monitor as M
from rai_bvar.engines.mcmc import state as S
from rai_bvar.engines.mcmc.config import (
    CoefPriorStatic,
    MeanStatic,
    Shapes,
    StaticConfig,
    StoreStatic,
    StructStatic,
    VolStatic,
)
from rai_bvar.engines.mcmc.data import base_cache, build_engine_data
from rai_bvar.settings import MCMCSettings

pytestmark = pytest.mark.filterwarnings("error:Explicitly requested dtype")

LOG_2PI = np.log(2.0 * np.pi)
T_PAD, T_EFF, LAGS = 64, 37, 2  # T_eff deliberately not a multiple of anything (R8)


# --------------------------------------------------------------------------- test-double kernels


def _gauss(x, mean, prec, mask):
    terms = 0.5 * (jnp.log(prec) - LOG_2PI) - 0.5 * prec * (x - mean) ** 2
    return jnp.sum(jnp.where(mask > 0, terms, 0.0))


@dataclass(frozen=True)
class ToyVol:
    """VolBlock double: Constant or SV core, optional outlier/t scales (DESIGN §3.5)."""

    static: VolStatic
    T_pad: int

    def _core_lv(self, cs):
        if self.static.persistent == "constant":
            return jnp.broadcast_to(cs.log_lam, (self.T_pad,))
        return cs.h[1:]

    def _scale2(self, ts):
        out = jnp.zeros(self.T_pad)
        if self.static.has_outliers:
            out = out + 2.0 * jnp.log(ts.o_idx.astype(jnp.float64) + 1.0)
        if self.static.has_t:
            out = out + ts.log_tau
        return out

    def log_var_eq(self, vs, gvol, which="effective"):
        lv = self._core_lv(vs.core)
        return lv + self._scale2(vs.trans) if which == "effective" else lv

    def log_scale_eq(self, vs):
        return vs.core.log_lam if self.static.coupled else jnp.zeros(())

    def loglik_eq(self, vs, resid, tmask, gvol):
        lv = self.log_var_eq(vs, gvol)
        terms = -0.5 * (LOG_2PI + lv + resid**2 * jnp.exp(-lv))
        return jnp.sum(jnp.where(tmask > 0, terms, 0.0))

    def extract_eq(self, vs, T_eff):
        cs, ts, st = vs.core, vs.trans, self.static
        if st.persistent == "constant":
            core = S.ConstantDraw(log_lam=cs.log_lam)
        else:
            h_T = lax.dynamic_index_in_dim(cs.h, T_eff, keepdims=False)
            core = (
                S.SVRWDraw(h_T=h_T, sigma=cs.sigma)
                if st.rw
                else S.SVDraw(h_T=h_T, mu=cs.mu, phi=cs.phi, sigma=cs.sigma)
            )
        grid = np.asarray(st.nu_grid, dtype=np.float64)
        if st.transitory == "outlier":
            trans = S.OutlierDraw(log_p=ts.log_p, log1m_p=ts.log1m_p)
        elif st.transitory == "t":
            trans = S.TDraw(nu=jnp.asarray(grid)[ts.nu_idx])
        elif st.transitory == "outlier_t":
            trans = S.OutlierTDraw(
                log_p=ts.log_p, log1m_p=ts.log1m_p, nu=jnp.asarray(grid)[ts.nu_idx]
            )
        else:
            trans = S.Empty()
        return S.VolDraw(core=core, trans=trans)

    def log_prior_eq(self, vs, prior_i, tmask):
        cs, ts, st = vs.core, vs.trans, self.static
        if st.persistent == "constant":
            lp = -2.0 * cs.log_lam - jnp.exp(-cs.log_lam)
        else:
            live = jnp.concatenate([jnp.ones(1), tmask])
            lp = -cs.sigma - 0.5 * jnp.sum(jnp.where(live > 0, cs.h**2, 0.0))
            if not st.rw:
                lp = lp - 0.5 * cs.mu**2 - cs.phi**2
        if st.has_outliers:
            lp = lp + 3.0 * ts.log_p
        if st.has_t:
            lp = lp - 0.1 * ts.nu_idx
        return lp


@dataclass(frozen=True)
class ToyStructure:
    """Recursive-structure double: ``e_i = ỹ_i + Σ_{j<i} a_ij ỹ_j − g_i`` (DESIGN §3.2, §3.7.1)."""

    static: StructStatic
    scaled: bool  # some prior block is λ-scaled (ctx.log_scale = log λ_i)

    def pre_residual(self, glob, data, f_reduced):
        y = data.Y if f_reduced is None else data.Y - f_reduced
        return y * data.tmask[:, None]

    def context(self, i, ytil, eq_i, vol, data, gvol):
        n = ytil.shape[1]
        return S.EqContext(
            i=i,
            target=ytil[:, i],
            cols=-ytil[:, : n - 1],
            cmask=(jnp.arange(n - 1) < i).astype(jnp.float64),
            weights=data.tmask * jnp.exp(-vol.log_var_eq(eq_i.vol, gvol)),
            log_scale=vol.log_scale_eq(eq_i.vol) if self.scaled else jnp.zeros(()),
            tmask=data.tmask,
        )

    def contemp_moments(self, ctx, prior_i):
        c = prior_i.contemp
        factor = jnp.exp(-ctx.log_scale) if self.static.contemp_scaling == "lambda" else 1.0
        return S.GaussianDiag(c.mean0, jnp.where(c.cmask > 0, c.prec0 * factor, 1.0), c.cmask)

    def residual(self, a_i, ctx, g_i):
        return (ctx.target - (ctx.cols * ctx.cmask) @ a_i - g_i) * ctx.tmask

    def log_jacobian(self, glob, T_eff):
        return jnp.zeros(())

    def extract(self, eq_struct, glob):
        return S.StructDraw(a=eq_struct.a)

    def reduced_form(self, sd, gamma):
        n = gamma.shape[0]
        A = jnp.eye(n) + jnp.tril(jnp.pad(sd.a, ((0, 0), (0, 1))), -1)
        return lax.linalg.triangular_solve(A, gamma, left_side=True, lower=True).T


@dataclass(frozen=True)
class ToyLinear:
    """Structural Linear double: fuses ``a_i``; ``g_i = X(δ_i + M_c a_i)`` read from ``cache.F``."""

    static: MeanStatic
    fuses_contemp = True

    def fitted_reduced(self, glob, data):
        return None

    def fitted_eq(self, ms, a_i, ctx, cache, shared):
        k = ms.delta.shape[0]
        return cache.F[:, :k] @ (ms.delta + shared.Mc @ a_i)

    def coefficients_eq(self, ms, a_i, shared):
        return ms.delta

    def log_prior_eq(self, ms, a_i, pm, cm, shared):
        return _gauss(ms.delta, pm.mean, pm.prec, pm.cmask) + _gauss(
            a_i, cm.mean, cm.prec, cm.cmask
        )

    def extract_eq(self, ms, a_i, shared):
        return S.LinearDraw(gamma=ms.delta + shared.Mc @ a_i)

    def extract_global(self, glob):
        return S.GlobDraw(S.Empty(), S.Empty(), S.Empty(), S.Empty())


@dataclass(frozen=True)
class ToyReduced:
    """Reduced-form double: ``ỹ = Y − XΠ``, ``g_i = 0``, ``Π`` global (DESIGN §3.7.3)."""

    static: MeanStatic
    k: int
    fuses_contemp = False

    def fitted_reduced(self, glob, data):
        return data.X @ glob.mean.Pi

    def fitted_eq(self, ms, a_i, ctx, cache, shared):
        return jnp.zeros_like(ctx.target)

    def coefficients_eq(self, ms, a_i, shared):
        return jnp.zeros(self.k)

    def log_prior_eq(self, ms, a_i, pm, cm, shared):
        assert cm is None  # a non-fusing mean never gets the a_i prior
        return jnp.zeros(())

    def extract_eq(self, ms, a_i, shared):
        return S.Empty()

    def extract_global(self, glob):
        return S.GlobDraw(S.CCCMDraw(Pi=glob.mean.Pi), S.Empty(), S.Empty(), S.Empty())


@dataclass(frozen=True)
class ToyZero(ToyReduced):
    def fitted_reduced(self, glob, data):
        return None

    def extract_global(self, glob):
        return S.GlobDraw(S.Empty(), S.Empty(), S.Empty(), S.Empty())


@dataclass(frozen=True)
class ToyCoef:
    def moments_eq(self, cs_i, i, prior_i, glob_coef):
        c = prior_i.coef
        return S.GaussianDiag(c.mean0, c.prec0, jnp.ones_like(c.mean0))

    def log_prior_eq(self, cs_i, coef_i, prior_i, glob_coef):
        return jnp.zeros(())  # FixedDiagonal has no state of its own


@dataclass(frozen=True)
class ToyCCCM:
    def log_prior(self, glob, prior):
        return -0.5 * jnp.sum(glob.mean.Pi**2)


def _vmap_eq(f):
    return jax.vmap(f, in_axes=(0, 0, 0, 0))


def _map_eq(f):
    return lambda *args: lax.map(lambda t: f(*t), args)


@dataclass(frozen=True)
class ToyProgram:
    structure: Any
    mean: Any
    coef: Any
    vol: Any
    global_pre: tuple = ()
    global_post: tuple = ()
    eq_map: Any = _vmap_eq


def jitted(program: ToyProgram):
    """``monitor_values`` as the runner calls it: inside ``jit`` (one compile, no op-by-op)."""
    return jax.jit(lambda st, data, prior, cache: M.monitor_values(st, data, prior, cache, program))


def toy_program(static: StaticConfig, eq_map=_vmap_eq) -> ToyProgram:
    kind = static.mean.kind
    scaled = static.struct.contemp_scaling == "lambda" or static.mean.coef_scaling == "lambda"
    mean = {
        "linear_structural": lambda: ToyLinear(static.mean),
        "linear_reduced": lambda: ToyReduced(static.mean, static.shapes.k),
        "zero": lambda: ToyZero(static.mean, static.shapes.k),
    }[kind]()
    return ToyProgram(
        structure=ToyStructure(static.struct, scaled),
        mean=mean,
        coef=ToyCoef(),
        vol=ToyVol(static.vol, static.shapes.T_pad),
        global_pre=(ToyCCCM(),) if kind == "linear_reduced" else (),
        eq_map=eq_map,
    )


# --------------------------------------------------------------------------- compositions

COMPOSITIONS = {
    "structural-constant": (
        3,
        1,
        StructStatic(),
        MeanStatic("linear_structural"),
        VolStatic("constant"),
    ),
    "structural-mode2": (
        3,
        1,
        StructStatic(contemp_scaling="lambda"),
        MeanStatic("linear_structural", coef_scaling="lambda"),
        VolStatic("constant", coupled=True),
    ),
    "structural-sv": (3, 2, StructStatic(), MeanStatic("linear_structural"), VolStatic("sv")),
    "structural-sv-rw": (
        3,
        0,
        StructStatic(),
        MeanStatic("linear_structural"),
        VolStatic("sv", rw=True),
    ),
    "structural-sv-ot": (
        4,
        1,
        StructStatic(),
        MeanStatic("linear_structural"),
        VolStatic("sv", transitory="outlier_t"),
    ),
    "structural-sv-t": (
        2,
        1,
        StructStatic(),
        MeanStatic("linear_structural"),
        VolStatic("sv", transitory="t"),
    ),
    "structural-constant-o": (
        3,
        1,
        StructStatic(),
        MeanStatic("linear_structural"),
        VolStatic("constant", transitory="outlier"),
    ),
    "reduced-sv": (3, 1, StructStatic(), MeanStatic("linear_reduced"), VolStatic("sv")),
    "reduced-iw-equivalent": (
        4,
        1,
        StructStatic(contemp_scaling="lambda"),
        MeanStatic("linear_reduced"),
        VolStatic("constant", coupled=True),
    ),
    "zero-sv-n1": (1, 0, StructStatic(), MeanStatic("zero"), VolStatic("sv")),
    "zero-sv-o": (2, 1, StructStatic(), MeanStatic("zero"), VolStatic("sv", transitory="outlier")),
}


def make_static(name: str) -> StaticConfig:
    n, n_det, struct, mean, vol = COMPOSITIONS[name]
    return StaticConfig(
        Shapes.of(n, LAGS, n_det, T_PAD), struct, mean, CoefPriorStatic(), vol, StoreStatic()
    )


def variables(n: int) -> tuple[str, ...]:
    return tuple(f"y{i}" for i in range(n))


def random_problem(static: StaticConfig, seed: int):
    """A random valid ``(state, data, prior, cache)`` (NumPy leaves) for ``static``."""
    rng = np.random.default_rng(seed)
    sh, vol = static.shapes, static.vol
    n, k, T = sh.n, sh.k, sh.T_pad
    lower = np.arange(n - 1)[None, :] < np.arange(n)[:, None]  # active a-slots j < i

    def fill(path, leaf):
        name = S.leaf_path(path)
        shape = np.shape(leaf)
        if name == "eq.struct.a":
            return np.where(lower, 0.3 * rng.standard_normal(shape), 0.0)
        if name in ("eq.vol.core.phi",):
            return rng.uniform(-0.9, 0.95, shape)
        if name == "eq.vol.core.sigma":
            return rng.uniform(0.1, 0.5, shape)
        if name == "eq.vol.core.h":
            return 0.5 * rng.standard_normal(shape)
        if name == "eq.vol.trans.o_idx":
            o = rng.integers(0, vol.outlier_K, shape).astype(np.int32)
            o[:, T_EFF:] = 0
            return o
        if name == "eq.vol.trans.log_tau":
            t = 0.3 * rng.standard_normal(shape)
            t[:, T_EFF:] = 0.0
            return t
        if name == "eq.vol.trans.nu_idx":
            return rng.integers(0, len(vol.nu_grid), shape).astype(np.int32)
        if name == "eq.vol.trans.log_p":
            return np.log(rng.uniform(0.01, 0.2, shape))
        if name.startswith("status") or name.split(".")[-1].startswith(("acc_", "rb_")):
            return leaf
        return 0.3 * rng.standard_normal(shape) if leaf.dtype == np.float64 else leaf

    st = jax.tree_util.tree_map_with_path(fill, S.state_template(static))
    if vol.has_outliers:
        trans = st.eq.vol.trans
        st = dataclasses.replace(
            st,
            eq=dataclasses.replace(
                st.eq,
                vol=dataclasses.replace(
                    st.eq.vol,
                    trans=dataclasses.replace(trans, log1m_p=np.log1p(-np.exp(trans.log_p))),
                ),
            ),
        )
    tmpl = S.prior_template(static)
    cmask = lower.astype(np.float64)
    coef = S.CoefPriorArrays(
        mean0=0.2 * rng.standard_normal((n, k)), prec0=rng.uniform(0.5, 4.0, (n, k)),
        group=tmpl.eq.coef.group,
    )  # fmt: skip
    contemp = S.ContempPriorArrays(
        mean0=np.where(lower, 0.1 * rng.standard_normal((n, n - 1)), 0.0),
        prec0=np.where(lower, rng.uniform(0.5, 4.0, (n, n - 1)), 1.0),
        cmask=cmask,
    )
    Mc = (
        0.2 * rng.standard_normal((k, n - 1))
        if static.mean.kind == "linear_structural"
        else np.zeros((k, n - 1))
    )
    prior = S.PriorArrays(
        eq=S.EqPrior(coef, contemp, tmpl.eq.vol_core, tmpl.eq.transitory),
        shared=S.SharedPrior(Mc=Mc),
    )
    y = rng.standard_normal((T, n))
    data = build_engine_data(
        y, rng.standard_normal((LAGS, n)), np.int32(T_EFF), n_det=sh.n_det, t0=LAGS + 1.0
    )
    data = jax.tree.map(np.asarray, data)
    if static.mean.kind == "linear_structural":
        F = np.concatenate([data.X, data.X @ Mc - data.Y[:, : n - 1]], axis=1)
        cache = dataclasses.replace(S.FitCache.empty(), F=F)
    else:
        cache = S.FitCache.empty()
    return st, data, prior, cache


def reference_monitors(static: StaticConfig, st, data, prior) -> dict[str, Any]:
    """Dense NumPy monitors from the raw state arrays (independent of ``monitor_values``)."""
    sh, vol = static.shapes, static.vol
    n, n_det = sh.n, sh.n_det
    Y, X, m = data.Y, data.X, data.tmask
    a = st.eq.struct.a
    A = np.eye(n) + np.tril(np.pad(a, ((0, 0), (0, 1))), -1)
    kind = static.mean.kind
    if kind == "linear_structural":
        Gamma = st.eq.mean.delta + a @ prior.shared.Mc.T  # γ_i = δ_i + M_c a_i
        E = Y @ A.T - X @ Gamma.T
        B = np.linalg.solve(A, Gamma).T
    elif kind == "linear_reduced":
        Pi = st.glob.mean.Pi
        E = (Y - X @ Pi) @ A.T
        B = Pi
    else:
        E = Y @ A.T
        B = None
    core, trans = st.eq.vol.core, st.eq.vol.trans
    lv = (
        np.repeat(core.log_lam[:, None], sh.T_pad, axis=1)
        if vol.persistent == "constant"
        else core.h[:, 1:].copy()
    )
    lv_eff = lv.copy()
    if vol.has_outliers:
        lv_eff += 2.0 * np.log(trans.o_idx + 1.0)
    if vol.has_t:
        lv_eff += trans.log_tau
    live = m > 0
    terms = -0.5 * (LOG_2PI + lv_eff.T + E**2 * np.exp(-lv_eff.T))
    out: dict[str, Any] = {"loglik": float(np.sum(terms[live]))}
    # log prior: coefficient blocks (λ-scaled where static), vol blocks, global steps
    scale = np.exp(core.log_lam) if vol.coupled else np.ones(n)
    lower = np.arange(n - 1)[None, :] < np.arange(n)[:, None]

    def gauss(x, mean, prec):
        return 0.5 * (np.log(prec) - LOG_2PI) - 0.5 * prec * (x - mean) ** 2

    c = prior.eq.contemp
    cprec = c.prec0 / (scale[:, None] if static.struct.contemp_scaling == "lambda" else 1.0)
    lp = float(np.sum(gauss(a, c.mean0, cprec)[lower]))
    if kind == "linear_structural":
        p = prior.eq.coef
        pprec = p.prec0 / (scale[:, None] if static.mean.coef_scaling == "lambda" else 1.0)
        lp += float(np.sum(gauss(st.eq.mean.delta, p.mean0, pprec)))
    if kind == "linear_reduced":
        lp += -0.5 * float(np.sum(st.glob.mean.Pi**2))
    if vol.persistent == "constant":
        lp += float(np.sum(-2.0 * core.log_lam - np.exp(-core.log_lam)))
    else:
        lp += float(np.sum(-core.sigma - 0.5 * np.sum(core.h[:, : T_EFF + 1] ** 2, axis=1)))
        if not vol.rw:
            lp += float(np.sum(-0.5 * core.mu**2 - core.phi**2))
    if vol.has_outliers:
        lp += float(np.sum(3.0 * trans.log_p))
    if vol.has_t:
        lp += float(np.sum(-0.1 * trans.nu_idx))
    out["log_prior"] = lp
    if B is not None:
        out["B_own1"] = B[n_det + np.arange(n), np.arange(n)]
        if n_det:
            out["intercept"] = B[0]
    if vol.persistent == "constant":
        out["log_lam"] = core.log_lam
    else:
        out["h_T"] = core.h[:, T_EFF]
        out["sigma"] = core.sigma
        out["h_1"] = core.h[:, 1]
        out["h_mid"] = core.h[:, -(-T_EFF // 2)]  # h_{⌈T/2⌉}
        if not vol.rw:
            out["mu"], out["phi"] = core.mu, core.phi
    if n >= 2:
        rows, cols = np.tril_indices(n, -1)
        out["a"] = a[rows, cols]
    if vol.has_outliers:
        out["log_p"] = trans.log_p
        o = trans.o_idx[:, :T_EFF] + 1  # real rows only
        out["outlier_share"] = (o > 1).mean(axis=1)
        out["outlier_at_max"] = (o == vol.outlier_K).mean(axis=1)
    if vol.has_t:
        out["nu"] = np.asarray(vol.nu_grid, dtype=np.float64)[trans.nu_idx]
    return out


# --------------------------------------------------------------------------- monitored sets


@pytest.mark.parametrize(
    ("vol", "primary", "secondary"),
    [
        (VolStatic("constant"), 41, 190 + 20 + 1),  # DESIGN §8.1: 41 with Constant
        (VolStatic("sv"), 101, 190 + 40 + 20 + 1),  # 101 with SV
        (VolStatic("sv", rw=True), 61, 190 + 40 + 20 + 1),
        (VolStatic("sv", transitory="outlier_t"), 101, 291),  # ≈ 290 secondary
    ],
)
def test_monitored_set_sizes_at_n20(vol, primary, secondary):
    static = StaticConfig(
        Shapes.of(20, 4, 1, 320), StructStatic(), MeanStatic("linear_structural"),
        CoefPriorStatic(), vol,
    )  # fmt: skip
    spec = M.monitor_spec(static, variables(20))
    assert len(spec.primary) == primary
    assert len(spec.secondary) == secondary


def test_primary_and_secondary_membership():
    static = make_static("structural-sv-ot")
    spec = M.monitor_spec(static, ("gdp", "cpi", "ff", "ur"))
    assert spec.primary[0] == "loglik"
    primary_fields = {spec.slot(x).field for x in spec.primary}
    secondary_fields = {spec.slot(x).field for x in spec.secondary}
    assert primary_fields == {"loglik", "B_own1", "mu", "phi", "sigma", "h_T"}
    assert secondary_fields == {"a", "h_1", "h_mid", "intercept", "log_p", "nu", "log_prior"}
    assert "a[cpi,gdp]" in spec.secondary and "a[ur,ff]" in spec.secondary
    assert spec.slot("a[ur,ff]").index == 5 and spec.slot("a[ur,ff]").variable == "ur,ff"
    assert spec.slot("mu[ff]") == M.MonitorSlot("mu[ff]", "mu", 2, "ff")
    assert spec.variables == ("gdp", "cpi", "ff", "ur")


@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_fields_follow_the_composition(name):
    static = make_static(name)
    spec = M.monitor_spec(static, variables(static.shapes.n))
    shapes = spec.field_shapes()
    n, vol = static.shapes.n, static.vol
    linear = static.mean.kind != "zero"
    assert ("B_own1" in shapes) == linear
    assert ("intercept" in shapes) == (linear and static.shapes.n_det > 0)
    assert ("log_lam" in shapes) == (vol.persistent == "constant")
    assert ("mu" in shapes) == ("phi" in shapes) == (vol.persistent == "sv" and not vol.rw)
    assert ("h_mid" in shapes) == (vol.persistent == "sv")
    assert ("a" in shapes) == (n >= 2)
    if n >= 2:
        assert shapes["a"] == (n * (n - 1) // 2,)
    assert ("log_p" in shapes) == vol.has_outliers and ("nu" in shapes) == vol.has_t
    aux = M.auxiliary_fields(vol)
    assert aux == (("outlier_share", "outlier_at_max") if vol.has_outliers else ())
    assert not set(aux) & set(shapes)  # auxiliary fields are no monitored quantity: never gate


def test_spec_validation():
    static = make_static("structural-sv")
    with pytest.raises(ValueError, match="3 variable names"):
        M.monitor_spec(static, ("a", "b"))
    with pytest.raises(ValueError, match="unique"):
        M.monitor_spec(static, ("a", "a", "b"))
    with pytest.raises(ValueError, match="unique"):
        M.MonitorSpec(("x", "y"), ("x",))
    with pytest.raises(ValueError, match="not monitored"):
        M.MonitorSpec(("x",), (), (M.MonitorSlot("z", "z"),))


# --------------------------------------------------------------------------- monitor_values


@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_monitor_values_match_the_dense_reference(name):
    static = make_static(name)
    st, data, prior, cache = random_problem(static, seed=sorted(COMPOSITIONS).index(name))
    got = jitted(toy_program(static))(st, data, prior, cache)
    spec = M.monitor_spec(static, variables(static.shapes.n))
    shapes = {
        **spec.field_shapes(),
        **{f: (static.shapes.n,) for f in M.auxiliary_fields(static.vol)},
    }
    assert set(got) == set(shapes)  # the same fields (jit sorts dict keys)
    for key, value in got.items():
        assert value.dtype == jnp.float64
        assert value.shape == shapes[key], key
    ref = reference_monitors(static, st, data, prior)
    assert set(ref) == set(got)
    for key in got:
        np.testing.assert_allclose(
            np.asarray(got[key]), ref[key], rtol=1e-12, atol=1e-12, err_msg=key
        )


@pytest.mark.parametrize("name", ["structural-sv-ot", "reduced-iw-equivalent", "zero-sv-o"])
def test_monitor_values_jit_vmap_over_chains_and_lax_map(name):
    """The runner's ``extract`` jits ``monitor_values`` and vmaps it over chains; a
    ``Batching.MAP`` program batches equations with ``lax.map`` (DESIGN §4.5, §5.1)."""
    static = make_static(name)
    problems = [random_problem(static, seed) for seed in (11, 12, 13)]
    _, data, prior, cache = problems[0]
    states = jax.tree.map(lambda *xs: np.stack(xs), *(p[0] for p in problems))
    program = toy_program(static)
    batched = jax.jit(jax.vmap(lambda s: M.monitor_values(s, data, prior, cache, program)))(states)
    mapped = jitted(toy_program(static, _map_eq))(problems[1][0], data, prior, cache)
    one = jitted(program)
    for c in range(3):
        eager = one(problems[c][0], data, prior, cache)
        for key in eager:
            np.testing.assert_allclose(batched[key][c], eager[key], rtol=1e-13, atol=1e-13)
    eager = one(problems[1][0], data, prior, cache)
    for key in eager:
        np.testing.assert_allclose(mapped[key], eager[key], rtol=1e-13, atol=1e-13)


def test_monitor_values_ignore_padded_rows():
    """Values on rows ``t ≥ T_eff`` (and SV states beyond ``T_eff``) never reach a monitor."""
    static = make_static("structural-sv-ot")
    st, data, prior, cache = random_problem(static, 5)
    fn = jitted(toy_program(static))
    base = fn(st, data, prior, cache)
    core, trans = st.eq.vol.core, st.eq.vol.trans
    h = core.h.copy()
    h[:, T_EFF + 1 :] = 1e6  # padded SV states
    o_idx = trans.o_idx.copy()
    o_idx[:, T_EFF:] = static.vol.outlier_K - 1  # padded outlier scales at K
    vol = dataclasses.replace(
        st.eq.vol,
        core=dataclasses.replace(core, h=h),
        trans=dataclasses.replace(trans, o_idx=o_idx),
    )
    st2 = dataclasses.replace(st, eq=dataclasses.replace(st.eq, vol=vol))
    moved = fn(st2, data, prior, cache)
    for key in base:
        np.testing.assert_array_equal(np.asarray(moved[key]), np.asarray(base[key]), err_msg=key)


def proper_vol_prior(static: StaticConfig, prior: S.PriorArrays) -> S.PriorArrays:
    """Replace the zero vol-prior template of :func:`random_problem` by proper priors."""
    n, vol = static.shapes.n, static.vol
    full = lambda v: np.full(n, v)  # noqa: E731
    if vol.persistent == "constant":
        core: Any = S.ConstantPrior(shape=full(3.0), scale=full(2.0))
    else:
        core = S.SVPrior(
            b_mu=full(0.0), B_mu=full(10.0), a_phi=full(20.0), b_phi=full(1.5),
            B_sigma=full(1.0), c0=full(2.5), C0=full(0.5), B0=full(1.0), m0=full(0.0),
            V0=full(4.0), log_c=full(np.log(1e-12)),
        )  # fmt: skip
    G = len(vol.nu_grid)
    trans: Any = (
        S.Empty()
        if vol.transitory == "none"
        else S.TransitoryPrior(a_p=full(2.0), b_p=full(30.0), log_nu_w=np.full((n, G), -np.log(G)))
    )
    eq = dataclasses.replace(prior.eq, vol_core=core, transitory=trans)
    return dataclasses.replace(prior, eq=eq)


@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_monitor_values_on_the_composed_kernels(name):
    """Wired to the production kernels through ``compose`` (WP10–WP14) when they exist: the
    monitored log-likelihood and log prior equal the independent validation references
    (``validation.log_joint``, WP9; D-22), and every other field the dense reference."""
    compose = pytest.importorskip("rai_bvar.engines.mcmc.compose")
    log_joint = pytest.importorskip("rai_bvar.validation.log_joint")
    static = make_static(name)
    try:
        program = compose.compose(static)
    except NotImplementedError as err:  # a kernel of this composition is not written yet
        pytest.skip(str(err))
    st, data, prior, _ = random_problem(static, 3)
    prior = proper_vol_prior(static, prior)
    cache = jax.jit(lambda d, p: program.mean.prepare(d, p, base_cache(d, p.shared.Mc, static)))(
        data, prior
    )
    got = jitted(program)(st, data, prior, cache)
    jst, jdata, jprior = (jax.tree.map(jnp.asarray, t) for t in (st, data, prior))
    np.testing.assert_allclose(
        got["loglik"], log_joint.loglik(jst, jdata, jprior, static), rtol=1e-10
    )
    np.testing.assert_allclose(
        got["log_prior"], log_joint.log_prior(jst, jprior, static, jdata.tmask), rtol=1e-10
    )
    ref = reference_monitors(static, st, data, prior)  # its log prior is the test doubles'
    for key in set(got) - {"log_prior"}:
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-10, atol=1e-12, err_msg=key)


# --------------------------------------------------------------------------- the converged flag


def ar1(rng, rho, shape):
    """Stationary AR(1) draws along axis 1 of ``shape = (chains, draws, ...)``."""
    e = rng.standard_normal(shape)
    x = np.empty(shape)
    x[:, 0] = e[:, 0] / np.sqrt(1.0 - rho**2)
    for t in range(1, shape[1]):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def iid_store(spec: M.MonitorSpec, chains=4, draws=1000, seed=0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal((chains, draws, *s)) for k, s in spec.field_shapes().items()}


@pytest.fixture
def spec3() -> M.MonitorSpec:
    return M.monitor_spec(make_static("structural-sv"), ("gdp", "cpi", "ff"))


SETTINGS = MCMCSettings()


def test_well_mixed_chains_converge(spec3):
    ok, warnings = M.converged(iid_store(spec3), spec3, SETTINGS, error=False, eb_converged=None)
    assert ok and warnings == []
    ok, _ = M.converged(iid_store(spec3), spec3, SETTINGS, error=False, eb_converged=True)
    assert ok


def test_failure_in_a_primary_quantity_is_not_converged(spec3):
    mon = iid_store(spec3)
    mon["phi"][2, :, 1] += 1.5  # one chain of phi[cpi] disagrees
    ok, warnings = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok
    (msg,) = warnings
    assert msg.startswith("MCMC not converged: 1 of 16 primary quantities fail R-hat < 1.01")
    assert "phi[cpi] R-hat" in msg and msg.endswith("increase draws, burn or thin")


def test_low_primary_ess_is_not_converged(spec3):
    mon = iid_store(spec3)
    mon["loglik"] = ar1(np.random.default_rng(3), 0.97, (4, 1000))  # R-hat fine, ESS ≈ 60
    ok, (msg,) = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok and "loglik R-hat" in msg


def test_primary_warning_names_the_five_worst_in_order(spec3):
    mon = iid_store(spec3)
    shifts = {("mu", 0): 0.5, ("mu", 1): 0.7, ("sigma", 2): 0.9, ("h_T", 0): 1.2,
              ("phi", 1): 1.5, ("B_own1", 2): 1.8, ("h_T", 2): 2.1}  # fmt: skip
    for (name, i), shift in shifts.items():
        mon[name][0, :, i] += shift
    ok, (msg,) = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok and "7 of 16" in msg
    worst = msg.split("(worst: ")[1].split("); increase")[0]
    named = [part.split(" R-hat")[0] for part in worst.split("; ")]
    assert named == ["h_T[ff]", "B_own1[ff]", "phi[cpi]", "h_T[gdp]", "sigma[ff]"]


def test_secondary_only_failure_converges_with_a_warning(spec3):
    mon = iid_store(spec3)
    mon["a"][1] += 1.0  # every a_ij disagrees across chains; a_ij are secondary
    mon["h_mid"][1] += 1.0
    ok, warnings = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert ok
    (msg,) = warnings
    assert msg.startswith("secondary (informational; does not gate converged): 100% of the 3")
    assert "contemporaneous coefficients a_ij have R-hat ≥ 1.01" in msg and "a[cpi,gdp]" in msg


def test_few_secondary_failures_stay_silent():
    n = 15  # 105 a_ij: the warning needs more than 5.25 of them
    static = StaticConfig(
        Shapes.of(n, 1, 1, 64), StructStatic(), MeanStatic("linear_structural"),
        CoefPriorStatic(), VolStatic("constant"),
    )  # fmt: skip
    spec = M.monitor_spec(static, variables(n))
    mon = iid_store(spec, draws=800, seed=4)
    mon["a"][0, :, :5] += 1.0  # 5 of 105 = 4.8%
    assert M.converged(mon, spec, SETTINGS, error=False, eb_converged=None) == (True, [])
    mon["a"][0, :, :7] += 1.0  # 7 of 105 = 6.7%
    ok, (msg,) = M.converged(mon, spec, SETTINGS, error=False, eb_converged=None)
    assert ok and "7% of the 105 contemporaneous coefficients" in msg


def test_one_chain_is_never_converged(spec3):
    mon = {k: v[:1] for k, v in iid_store(spec3, draws=4000).items()}
    ok, warnings = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok
    assert warnings == [
        "single chain: split R-hat cannot detect between-chain disagreement, so fit.converged "
        "is False; use chains ≥ 2"
    ]


def test_an_error_is_never_converged(spec3):
    ok, warnings = M.converged(iid_store(spec3), spec3, SETTINGS, error=True, eb_converged=None)
    assert not ok
    assert warnings == [
        "a chain reported a numerical failure (MCMCNumericalError), so fit.converged is False"
    ]


def test_eb_failure_is_never_converged(spec3):
    ok, warnings = M.converged(iid_store(spec3), spec3, SETTINGS, error=False, eb_converged=False)
    assert not ok
    assert warnings == [
        "the empirical-Bayes optimisation of the Minnesota hyperparameters did not converge, so "
        "fit.converged is False"
    ]


def test_stuck_or_constant_primary_is_not_converged(spec3):
    mon = iid_store(spec3)
    mon["sigma"][:, :, 0] = 0.3  # never moves: R-hat and ESS are NaN
    ok, (msg,) = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok and "sigma[gdp] R-hat nan, ESS nan" in msg


def test_thresholds_come_from_the_settings(spec3):
    mon = iid_store(spec3)  # bulk ESS ≈ 4000 per quantity
    assert M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)[0]
    strict = MCMCSettings(ess_min=10_000.0)
    ok, (msg,) = M.converged(mon, spec3, strict, error=False, eb_converged=None)
    assert not ok and "bulk ESS > 10000" in msg and "16 of 16" in msg
    lax_rhat = MCMCSettings(rhat_max=1.5, ess_min=10.0)  # a shifted chain also cuts the ESS
    mon["phi"][2, :, 1] += 1.5
    assert not M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)[0]
    assert M.converged(mon, spec3, lax_rhat, error=False, eb_converged=None)[0]


def test_label_only_specs_read_monitor_arrays_directly():
    rng = np.random.default_rng(9)
    spec = M.MonitorSpec(("x",), ("a[2,1]",))
    mon = {
        "x": rng.standard_normal((4, 500)),
        "a[2,1]": rng.standard_normal((4, 500)) + np.array([[0.0], [0.0], [0.0], [2.0]]),
    }
    ok, (msg,) = M.converged(mon, spec, SETTINGS, error=False, eb_converged=None)
    assert ok and "100% of the 1 contemporaneous" in msg


def test_inputs_are_not_modified(spec3):
    mon = iid_store(spec3)
    copy = {k: v.copy() for k, v in mon.items()}
    M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    M.diagnostics_table(mon, spec3, {})
    for k in mon:
        np.testing.assert_array_equal(mon[k], copy[k])


# --------------------------------------------------------------------------- one evaluation per fit


@pytest.fixture
def counted(monkeypatch):
    """An empty R-hat/ESS memo and a record of the quantities ``rhat_ess_bulk_many`` evaluates."""
    monkeypatch.setattr(M, "_stats_cache", type(M._stats_cache)())
    seen: list[int] = []
    real = dg.rhat_ess_bulk_many

    def counting(x):
        seen.append(x.shape[-1])
        return real(x)

    monkeypatch.setattr(dg, "rhat_ess_bulk_many", counting)
    return seen


@pytest.mark.parametrize(
    ("n", "draws"),
    [(4, 400), (20, 2000)],  # n = 20 SV-O-t at 4 × 2000 (pre-D-35 default): 392 quantities, §8.2
)
def test_flag_table_and_warnings_evaluate_each_quantity_once(counted, n, draws):
    """The fit's sequence (results.py: ``converged``, ``diagnostics_table``, then
    ``warning_catalogue``, which runs ``converged`` again) ranks and FFTs every quantity once.

    A memoised value equals a recomputation to rounding, not bitwise: it is the value of the
    batch in which the quantity was first evaluated, and the FFT's last bits depend on the batch
    (at ``n = 20`` a few bulk ESS differ from the scalar function's in the last digits)."""
    static = make_static("structural-sv-ot")
    static = dataclasses.replace(static, shapes=Shapes.of(n, LAGS, 1, T_PAD))
    spec = M.monitor_spec(static, variables(n))
    mon = iid_store(spec, draws=draws, seed=31)
    mon["phi"][1, :, 2] += 1.0  # one primary failure, so every message path runs
    flag, first = M.converged(mon, spec, SETTINGS, error=False, eb_converged=None)
    primary_and_a = len(spec.primary) + len(spec.field_labels("a"))
    assert counted == [len(spec.primary), len(spec.field_labels("a"))]
    table = M.diagnostics_table(mon, spec, {})
    assert sum(counted) == len(spec.labels)  # the table only adds the other secondaries
    assert counted[-1] == len(spec.labels) - primary_and_a
    out = M.warning_catalogue(
        M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec)
    )
    assert sum(counted) == len(spec.labels) and not flag and out[0] == first[0]
    assert len(spec.labels) == (392 if n == 20 else 48)
    # every memoised value against one batch of all quantities, and some against the scalar
    # functions (Phase 1), to rounding
    rhat, ess = dg.rhat_ess_bulk_many(spec.stack(mon))
    np.testing.assert_allclose(table["rhat"], rhat, rtol=1e-13, atol=0)
    np.testing.assert_allclose(table["ess_bulk"], ess, rtol=1e-13, atol=0)
    for label in ("loglik", "phi[y2]", "a[y3,y1]", "nu[y0]", "log_prior", f"B_own1[y{n // 2}]"):
        x = spec.select(mon, [label])[label]
        np.testing.assert_allclose(table.loc[label, "rhat"], dg.rhat(x), rtol=1e-13, atol=0)
        np.testing.assert_allclose(table.loc[label, "ess_bulk"], dg.ess_bulk(x), rtol=1e-13, atol=0)


def test_memoised_statistics_follow_in_place_changes(spec3, counted):
    mon = iid_store(spec3, seed=33)
    assert M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)[0]
    mon["phi"][2, :, 1] += 1.5  # the same arrays, new content: a new key, a new evaluation
    ok, (msg,) = M.converged(mon, spec3, SETTINGS, error=False, eb_converged=None)
    assert not ok and "phi[cpi] R-hat" in msg
    rhat, ess = M.rhat_ess(mon, spec3, ["phi[cpi]", "phi[gdp]"])
    close = functools.partial(np.testing.assert_allclose, rtol=1e-13, atol=0)  # to rounding
    close(rhat[0], dg.rhat(mon["phi"][:, :, 1]))
    close(ess[0], dg.ess_bulk(mon["phi"][:, :, 1]))
    close(rhat[1], dg.rhat(mon["phi"][:, :, 0]))
    # (chains, draws) is part of the key: the same values split into other chains are new
    x = mon["phi"][:, :, 0].reshape(2, -1)
    spec = M.MonitorSpec(("x",), ())
    close(M.rhat_ess({"x": x}, spec, ["x"])[0][0], dg.rhat(x))
    # primary and a_ij once, then the changed column, then the re-split one: nothing else
    assert counted == [len(spec3.primary), len(spec3.field_labels("a")), 1, 1]


def test_memo_is_bounded(spec3, counted, monkeypatch):
    """At most ``_STATS_CACHE_SIZE`` pairs are kept and the least recently *used* go first: a hit
    refreshes its entry, so it outlives an older untouched one (a FIFO memo would drop it)."""
    monkeypatch.setattr(M, "_STATS_CACHE_SIZE", 5)
    mon = iid_store(spec3, seed=35)
    primary = spec3.primary
    M.rhat_ess(mon, spec3, primary)
    assert len(M._stats_cache) == 5
    M.rhat_ess(mon, spec3, primary[-5:])  # the most recent five are kept
    assert counted == [len(primary)]
    touched, untouched = primary[-5], primary[-4]  # the two oldest entries, in this order
    M.rhat_ess(mon, spec3, [touched])  # a hit: now the most recently used
    assert counted == [len(primary)]
    M.rhat_ess(mon, spec3, primary[:2])  # two new pairs evict the two least recently used
    assert counted == [len(primary), 2] and len(M._stats_cache) == 5
    M.rhat_ess(mon, spec3, [touched])  # still a hit (FIFO would have evicted it)
    assert counted == [len(primary), 2]
    M.rhat_ess(mon, spec3, [untouched])  # evicted: evaluated again
    assert counted == [len(primary), 2, 1] and len(M._stats_cache) == 5


# --------------------------------------------------------------------------- warning catalogue


def catalogue(name="structural-sv", n=None, **kw) -> list[str]:
    static = make_static(name)
    if n is not None:
        static = dataclasses.replace(static, shapes=Shapes.of(n, LAGS, 1, T_PAD))
    return M.warning_catalogue(
        M.DiagnosticsInput(
            static=static, settings=SETTINGS, variables=variables(static.shapes.n), **kw
        )
    )


def test_quiet_run_has_no_warnings():
    spec = M.monitor_spec(make_static("structural-sv"), variables(3))
    acc = {
        "acc_h": np.full((4, 3), 0.97),
        "acc_rej_run_max": np.full((4, 3), 4),
        "acc_fold": np.zeros((4, 3)),
    }
    assert catalogue(mon=iid_store(spec), spec=spec, acc=acc) == []


def test_catalogue_includes_the_flag_reasons():
    spec = M.monitor_spec(make_static("structural-sv"), variables(3))
    mon = {k: v[:1] for k, v in iid_store(spec).items()}
    out = catalogue(
        mon=mon, spec=spec, error=True, eb_converged=False, eb_warnings=("EB: stopped at a bound",)
    )
    assert out[0].startswith("a chain reported a numerical failure")
    assert out[1].startswith("single chain")
    assert out[-2].startswith("the empirical-Bayes optimisation")
    assert out[-1] == "EB: stopped at a bound"
    without_monitors = catalogue(chains=1, eb_converged=False)
    assert without_monitors == [out[1], out[-2]]
    assert catalogue(error=True)[0] == out[0]  # chains from settings: 4


def test_exact_h_acceptance_and_rejection_runs():
    acc = {
        "h": np.array([[0.95, 0.42, 0.97], [0.96, 0.6, 0.97]]),
        "rej_run_max": np.array([[3, 9, 51], [2, 8, 7]]),
    }
    (msg,) = catalogue(acc=acc)
    assert msg.startswith(
        "exact-h acceptance below 0.5 or a run of more than 50 rejected sweeps for ['y1', 'y2']"
    )
    assert "acceptance [0.42, 0.97], longest run [9.0, 51.0]" in msg
    assert "mixture correction mixes poorly there (N01 §10.2)" in msg
    exact_off = dataclasses.replace(make_static("structural-sv").vol, exact=False)
    static = dataclasses.replace(make_static("structural-sv"), vol=exact_off)
    assert M.warning_catalogue(M.DiagnosticsInput(static=static, settings=SETTINGS, acc=acc)) == []


def test_nc_fold_fraction():
    (msg,) = catalogue(acc={"acc_fold": np.array([[0.0, 0.02, 0.009], [0.0, 0.03, 0.0]])})
    assert msg.startswith(
        "non-centred sign fold in more than 1% of sweeps for ['y1'] (fractions [0.025])"
    )
    assert msg.endswith("where stochvol and bayesianVARs differ from rai-bvar (D-7)")
    # "> 1%": exactly 1% in each of 32 chains is quiet (a plain float mean rounds above), the
    # next float above warns
    fold = np.full((32, 3), 0.01)
    assert fold.mean(axis=0)[0] > 0.01
    assert catalogue(acc={"fold": fold}) == []
    fold[:, 2] = np.nextafter(0.01, 1.0)
    (msg,) = catalogue(acc={"fold": fold})
    assert "for ['y2'] (fractions [0.01])" in msg


def test_nu_mass_at_the_grid_bound():
    static = make_static("structural-sv-t")
    spec = M.monitor_spec(static, variables(2))
    mon = iid_store(spec)
    nu = np.full((4, 1000, 2), 3.0)
    nu[:, :700, 0] = 40.0  # mass 0.7 at the bound
    nu[:, :499, 1] = 40.0  # mass 0.499
    mon["nu"] = nu
    out = M.warning_catalogue(
        M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec)
    )
    nu_warnings = [w for w in out if w.startswith("Student-t")]
    assert nu_warnings == [
        "Student-t ν has posterior mass ≥ 0.5 at the grid bound 40 for ['y0'] (mass [0.7]): "
        "near-Gaussian tails, SV may suffice (report only; mass at ν = 3 is normal, N03 §6)"
    ]


def test_nu_mass_threshold_is_inclusive_at_one_half():
    """``mass ≥ 0.5`` (DESIGN §8.3): exactly one half warns, just below does not."""
    static = make_static("structural-sv-ot")
    spec = M.monitor_spec(static, variables(4))
    mon = iid_store(spec)
    nu = np.full((4, 1000, 4), 3.0)
    for i, kept in enumerate((700, 499, 500, 550)):
        nu[:, :kept, i] = 40.0
    mon["nu"] = nu
    out = M.warning_catalogue(
        M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec)
    )
    (msg,) = [w for w in out if w.startswith("Student-t")]
    assert "for ['y0', 'y2', 'y3'] (mass [0.7, 0.5, 0.55])" in msg


def test_outliers_at_the_grid_maximum():
    (msg,) = catalogue("zero-sv-o", outlier_at_max=np.array([0.002, 0.04]))
    assert msg == (
        "outlier scale at its maximum K = 20 for more than 1% of rows in ['y1'] (fractions "
        "[0.04]): consider a larger outlier_max"
    )
    assert catalogue("structural-sv", outlier_at_max=np.array([0.5, 0.5, 0.5])) == []


def test_outliers_at_the_grid_maximum_from_the_monitor():
    """Without an explicit input the warning reads the monitor's auxiliary ``outlier_at_max``
    (posterior mean over chains and draws): the production path, where no caller supplies it."""
    static = make_static("zero-sv-o")
    spec = M.monitor_spec(static, variables(2))
    mon = iid_store(spec)
    at_max = np.zeros((4, 1000, 2))
    at_max[:, :, 1] = 0.03
    at_max[:2, :, 0] = 0.019  # mean 0.0095: not more than 1%
    mon["outlier_at_max"] = at_max

    def outlier_warnings(**kw):
        inputs = M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec, **kw)
        return [w for w in M.warning_catalogue(inputs) if w.startswith("outlier scale")]

    (msg,) = outlier_warnings()
    assert "in ['y1'] (fractions [0.03]): consider a larger outlier_max" in msg
    assert outlier_warnings(outlier_at_max=np.zeros(2)) == []  # an explicit input wins
    del mon["outlier_at_max"]
    assert outlier_warnings() == []


def test_outliers_at_the_grid_maximum_threshold_is_strict():
    """``> 1% of rows`` (DESIGN §8.3): a posterior mean of exactly 1% at ``K`` is quiet, the next
    float above it warns. At 4 × 2000 draws (the pre-D-35 default) a plain float mean over chains
    and draws of ``0.01`` (one of 100 rows at ``K`` in every draw) rounds to
    ``0.010000000000000503``, so the decision must not rest on it."""
    static = make_static("zero-sv-o")
    spec = M.monitor_spec(static, variables(2))
    above = np.nextafter(0.01, 1.0)
    trap = np.full((4, 2000, 2), 1 / 100).reshape(-1, 2).mean(axis=0)  # strided: no pairwise sum
    assert (trap > 0.01).all()  # the trap this test pins

    def outlier_warnings(mon=None, **kw):
        inputs = M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec, **kw)
        return [w for w in M.warning_catalogue(inputs) if w.startswith("outlier scale")]

    mon = iid_store(spec, draws=2000)
    mon["outlier_at_max"] = np.full((4, 2000, 2), 1 / 100)
    assert outlier_warnings(mon) == []
    mon["outlier_at_max"][..., 1] = above  # just above, in every draw
    (msg,) = outlier_warnings(mon)
    assert "for more than 1% of rows in ['y1'] (fractions [0.01])" in msg
    mixed = np.zeros((4, 2000, 2))
    mixed[:, ::2, :] = 2 / 100  # half the draws at 2%, half at 0: mean exactly 1%
    mixed[:, 0, 1] = 3 / 100  # one more row at K in one draw: above
    mon["outlier_at_max"] = mixed
    (msg,) = outlier_warnings(mon)
    assert "in ['y1']" in msg
    # explicit inputs, (n,) and (chains, n)
    assert outlier_warnings(outlier_at_max=np.array([0.01, 0.01])) == []
    assert outlier_warnings(outlier_at_max=np.full((32, 2), 0.01)) == []
    (msg,) = outlier_warnings(outlier_at_max=np.array([above, 0.01]))
    assert "in ['y0']" in msg


def test_outlier_monitors_end_to_end():
    """``monitor_values`` → stacked draws → the warning and the per-equation table."""
    static = make_static("zero-sv-o")
    K = static.vol.outlier_K
    st, data, prior, cache = random_problem(static, 8)
    o_idx = np.zeros_like(st.eq.vol.trans.o_idx)
    o_idx[1, [3, 10]] = K - 1  # 2 of 37 real rows at K
    o_idx[0, 5] = K - 2  # an outlier below K
    o_idx[0, T_EFF:] = K - 1  # padded rows never count
    trans = dataclasses.replace(st.eq.vol.trans, o_idx=o_idx)
    st = dataclasses.replace(
        st, eq=dataclasses.replace(st.eq, vol=dataclasses.replace(st.eq.vol, trans=trans))
    )
    one = jitted(toy_program(static))(st, data, prior, cache)
    np.testing.assert_allclose(one["outlier_share"], [1 / T_EFF, 2 / T_EFF], rtol=1e-15)
    np.testing.assert_allclose(one["outlier_at_max"], [0.0, 2 / T_EFF], rtol=1e-15)
    mon = {k: np.broadcast_to(np.asarray(v), (2, 3, *np.shape(v))) for k, v in one.items()}
    spec = M.monitor_spec(static, variables(2))
    out = M.warning_catalogue(
        M.DiagnosticsInput(static=static, settings=SETTINGS, mon=mon, spec=spec)
    )
    assert [w for w in out if w.startswith("outlier scale")] == [
        "outlier scale at its maximum K = 20 for more than 1% of rows in ['y1'] (fractions "
        "[0.0541]): consider a larger outlier_max"
    ]
    table = M.diagnostics_table(mon, spec, {})
    assert not {"outlier_share", "outlier_at_max"} & set(table["field"])  # no quantity rows
    eq = table.attrs["equations"]
    assert eq["outlier_share"] == pytest.approx({"y0": 1 / T_EFF, "y1": 2 / T_EFF}, rel=1e-15)
    assert eq["outlier_at_max"] == pytest.approx({"y0": 0.0, "y1": 2 / T_EFF}, rel=1e-15)


def test_floor_hits_in_zero_offset_mode():
    static = make_static("structural-sv")
    static = dataclasses.replace(static, vol=dataclasses.replace(static.vol, zero_offset=True))
    acc = {"floor": np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]])}
    (msg,) = M.warning_catalogue(
        M.DiagnosticsInput(static=static, settings=SETTINGS, acc=acc, variables=("a", "b", "c"))
    )
    assert msg.startswith(
        "zero_offset parity mode: exact zero residuals were masked from the SV step in ['b']"
    )
    assert catalogue(acc=acc) == []  # c > 0 offsets never floor


def test_ordering_note_for_non_invariant_compositions():
    (msg,) = catalogue("structural-sv-ot")  # n = 4
    assert msg.startswith("this composition is not order-invariant: with 4 variables")
    assert "rai_bvar.ordering_sensitivity(model, data, mode='prior') (N08 §B4)" in msg
    assert catalogue("structural-sv") == []  # n = 3
    assert catalogue("reduced-iw-equivalent") == []  # n = 4, order-invariant (N08 §B2)
    assert M.is_order_invariant(make_static("reduced-iw-equivalent"))
    assert not M.is_order_invariant(make_static("reduced-sv"))
    assert not M.is_order_invariant(make_static("structural-mode2"))
    assert catalogue("structural-sv-ot", order_invariant=True) == []


def test_covid_and_tcode_messages():
    out = catalogue(covers_2020=True, tcode_clash=("cpi", "ur"))
    assert out == [
        "the estimation sample covers 2020 but no outlier treatment is set; consider "
        "SV(outliers=True) (SPEC §5.6)",
        "own-lag prior means contradict the transformation codes for ['cpi', 'ur'] (levels with "
        "mean < 0.5 or differences with mean ≥ 0.5)",
    ]
    assert catalogue("zero-sv-o", covers_2020=True) == []  # SV(outliers=True) suppresses it


# --------------------------------------------------------------------------- diagnostics table


def test_diagnostics_table(spec3):
    mon = iid_store(spec3, seed=21)
    acc = {
        "acc_h": np.full((4, 3), 0.9),
        "acc_rej_run_max": np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1], [0, 0, 9]]),
        "min_pivot": np.array([1e-3, 2e-3, 5e-4, 1e-2]),
    }
    table = M.diagnostics_table(mon, spec3, acc)
    assert list(table.index) == list(spec3.labels)
    assert table.index.name == "quantity"
    assert list(table.columns) == [
        "field", "variable", "primary", "mean", "sd", "mcse_mean", "q5", "q50", "q95",
        "mcse_q5", "mcse_q50", "mcse_q95", "rhat", "ess_bulk", "ess_tail", "escape",
    ]  # fmt: skip
    assert table["primary"].sum() == len(spec3.primary)
    assert table.loc["mu[cpi]", "variable"] == "cpi" and table.loc["a[ff,gdp]", "field"] == "a"
    x = mon["mu"][:, :, 1]
    row = table.loc["mu[cpi]"]
    np.testing.assert_allclose(row["mean"], x.mean(), rtol=1e-13)
    np.testing.assert_allclose(row["sd"], x.std(ddof=1), rtol=1e-13)
    np.testing.assert_allclose(
        row[["q5", "q50", "q95"]].to_numpy(float), np.quantile(x, [0.05, 0.5, 0.95]), rtol=1e-13
    )
    assert row["rhat"] == dg.rhat(x) and row["ess_bulk"] == dg.ess_bulk(x)
    assert row["ess_tail"] == dg.ess_tail(x) and row["mcse_mean"] == dg.mcse_mean(x)
    assert row["mcse_q50"] == dg.mcse_quantile(x, 0.5)
    assert table["escape"].isna().all()
    eq = table.attrs["equations"]
    assert eq["h"] == {"gdp": 0.9, "cpi": 0.9, "ff": 0.9}
    assert eq["rej_run_max"] == {"gdp": 4.0, "cpi": 5.0, "ff": 9.0}
    assert table.attrs["min_pivot"] == 5e-4  # the smallest pivot over chains (DESIGN §8.4)
    assert table.attrs["chains"] == 4 and table.attrs["draws"] == 1000
    assert set(table.attrs) == {"equations", "min_pivot", "chains", "draws"}
    pd.concat([table, table])  # attrs hold plain values: concatenation must not choke
    label_only = M.MonitorSpec(("x",), ())
    bare = M.diagnostics_table({"x": mon["mu"][:, :, 0]}, label_only, {"min_pivot": [0.3, 0.2]})
    assert bare.attrs == {"min_pivot": 0.2, "chains": 4, "draws": 1000}


def test_escape_probability_for_nu():
    static = make_static("structural-sv-t")
    spec = M.monitor_spec(static, variables(2))
    mon = iid_store(spec)
    nu = np.full((4, 1000, 2), 7.0)
    nu[:, 1::4, 0] = 8.0  # moves at 2 of every 4 consecutive pairs
    mon["nu"] = nu
    table = M.diagnostics_table(mon, spec, {})
    assert table.loc["nu[y0]", "escape"] == pytest.approx(2000 / 3996)
    assert table.loc["nu[y1]", "escape"] == 0.0
    assert np.isnan(table.loc["nu[y1]", "rhat"])  # never moves
    assert table["escape"].notna().sum() == 2


def test_escape_probability_is_per_sweep():
    """With ``acc_nu_move`` (sweeps with ``ν′ ≠ ν`` per sweep, the runner's rate) the escape
    column is that per-sweep probability averaged over chains, not the kept-draw share, which
    with ``thin > 1`` is ``P(ν^{(m+thin)} ≠ ν^{(m)})`` (N03 §6)."""
    static = make_static("structural-sv-t")
    spec = M.monitor_spec(static, variables(2))
    mon = iid_store(spec)
    nu = np.full((4, 1000, 2), 7.0)
    nu[:, 1::2, :] = 8.0  # thinned: every kept draw differs from the previous one
    mon["nu"] = nu
    acc = {"acc_nu_move": np.array([[0.2, 0.0], [0.4, 0.1], [0.3, 0.1], [0.3, 0.0]])}
    table = M.diagnostics_table(mon, spec, acc)
    assert table.loc["nu[y0]", "escape"] == pytest.approx(0.3, rel=1e-15)
    assert table.loc["nu[y1]", "escape"] == pytest.approx(0.05, rel=1e-15)
    assert table.attrs["equations"]["nu_move"] == pytest.approx({"y0": 0.3, "y1": 0.05})
    assert M.diagnostics_table(mon, spec, {}).loc["nu[y0]", "escape"] == 1.0  # kept draws


def test_equation_table_outlier_probabilities():
    """Rao-Blackwellised ``P(o_t > 1 | y)`` per row (the runner's ``latent["p_outlier"]``)
    averaged over chains and the ``T_eff`` real rows; the monitored shares beside them."""
    rng = np.random.default_rng(4)
    rb = rng.uniform(0.0, 0.2, (3, 2, 8))
    rb[..., 5:] = 0.0  # padded rows
    mon = {
        "outlier_share": rng.uniform(0.0, 0.1, (3, 50, 2)),
        "outlier_at_max": np.zeros((3, 50, 2)),
    }
    table = M.acceptance_table({"p_outlier": rb, "T_eff": 5}, ("x", "y"), mon=mon)
    assert list(table.columns) == ["p_outlier", "outlier_share", "outlier_at_max"]
    np.testing.assert_allclose(table["p_outlier"], rb[..., :5].mean(axis=(0, 2)), rtol=1e-14)
    np.testing.assert_allclose(table["outlier_share"], mon["outlier_share"].mean(axis=(0, 1)))
    assert table.attrs == {}  # T_eff is no sampler statistic
    trimmed = M.acceptance_table({"p_outlier": rb[..., :5]}, ("x", "y"))
    np.testing.assert_allclose(trimmed["p_outlier"], table["p_outlier"], rtol=1e-14)


def test_acceptance_table():
    acc = {
        "acc_h": np.array([[0.9, 0.8], [0.7, 0.6]]),
        "acc_rej_run_max": np.array([[3, 1], [2, 8]]),
        "floor": np.array([[1.0, 0.0], [2.0, 0.0]]),
        "min_pivot": np.array([0.5, 0.1]),
        "rb_out": np.zeros((2, 2, 5)),
    }
    table = M.acceptance_table(acc, ("x", "y"))
    assert list(table.columns) == ["h", "rej_run_max", "floor"]
    np.testing.assert_allclose(table["h"], [0.8, 0.7])
    np.testing.assert_allclose(table["rej_run_max"], [3.0, 8.0])
    np.testing.assert_allclose(table["floor"], [3.0, 0.0])
    assert table.attrs == {"min_pivot": 0.1}
