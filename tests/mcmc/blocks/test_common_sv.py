"""``CommonSV``: the ``h, ρ, σ_h`` kernel, the common cores and the shared-factor path (P3 DESIGN §5;
WP32).

Tier U (P3 §9.2), against WP30's ``validation.log_joint`` (natural coordinates: ``h``, ``ρ``,
``σ_h > 0``, ``λ``; the conditional densities below are written from the distributions the
kernels' parameters name, never from the kernels):

* item 1: the tridiagonal **Newton** of every ``h`` block (the gradient of the full ``ℓ`` below
  1e-8 at the returned mode, sequential layout); the **proposal's state independence** (the block
  proposal is bitwise unchanged when the current block values change and moves with a neighbour;
  mutant M28's required catcher, demonstrated here with a local M28); the **shared-factor draw**
  against the weighted path on the same noise (P2 §9.3 item 1's conditioning-aware rule
  ``2e-12 + 5εκ(1 + t)`` in posterior s.d.), and its factor computed once per chain and sweep
  (unbatched inside the equation body, E87);
* item 2: **density-ratio identities** (relative 1e-8, 20 states) of every ``h`` block, the
  ``σ_h²`` step (IG and GIG) and the common core (Mode I, and coupled with SOC/SUR rows), and the
  **MH identities** (1e-10) of the ``h`` blocks and the ``ρ`` step (both priors), with the two
  pinned ``ρ`` cases of §9.2 item 2 (E117, E158).

Tier B (P3 §9.3): the per-PR block Geweke of 407 (``common_sv_only``, ``kernel="global_step"``)
and 408 (``conjugate_niw_only``, ``kernel="sweep"``) through ``validation.joint`` at G-small.

Fit level: monitors (D-52), storage, the warnings of P3 §8, forecasts with the common ``h``
(futures by block id), parameter draws at a date, warm starts, the resolution rules (§2.1, rule
(14)), the R8 ordering note (E99, E130) and the ``lag_coefs`` keyword (E162(a)).
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import random
from jax.scipy.special import gammaln

import rai_bvar as rb
from rai_bvar.components.vol import common as CS
from rai_bvar.components.vol.robust import log_e2
from rai_bvar.engines.mcmc import keys as K
from rai_bvar.engines.mcmc import linalg as LA
from rai_bvar.engines.mcmc.compose import compose
from rai_bvar.engines.mcmc.keys import Block, Sub
from rai_bvar.engines.mcmc.plan import build_plan, check_invariants
from rai_bvar.engines.mcmc.protocols import GlobalStep, SupportsGlobalSteps, SupportsGlobalVol
from rai_bvar.engines.mcmc.runner import _prepare_fn
from rai_bvar.engines.mcmc.state import (
    CommonSVGlobal,
    Coupling,
    EqOut,
    SweepCtx,
    check_state_invariants,
    draw_template,
)
from rai_bvar.validation import log_joint as LJ
from rai_bvar.validation.geweke import GewekeConfig, geweke
from rai_bvar.validation.identities import density_ratio_error, mh_ratio_error
from rai_bvar.validation.joint import joint_spec

KEY = random.key(3232)
N_STATES = 20
EPS = float(np.finfo(np.float64).eps)
LOG2 = math.log(2.0)
G_SMALL = {"C": 1000, "M_c": 100, "M1": 200_000, "alpha": 1e-3}

_SPECS: dict[tuple, object] = {}


def spec_of(name: str, size: str = "full", **kw):
    key = (name, size, tuple(sorted(kw.items())))
    if key not in _SPECS:
        _SPECS[key] = joint_spec(name, size, **kw)
    return _SPECS[key]


def program_of(spec):
    return (
        compose(spec.static, overrides=dict(spec.overrides))
        if spec.overrides
        else compose(spec.static)
    )


def step_of(spec) -> CS.CommonSVStep:
    """The production ``CommonSVStep`` of ``spec``'s program (internals through ``impl``, so a
    mutant injected at the seam of P3 §9.6 reaches them)."""
    (step,) = [g for g in program_of(spec).global_post if g.block == Block.G_COMMON_SV]
    return step


_LOG_JOINT: dict[int, object] = {}


def log_joint_of(spec):
    """``spec.log_joint`` (WP30's reference joint) under ``jit`` (one trace per spec)."""
    if id(spec) not in _LOG_JOINT:
        _LOG_JOINT[id(spec)] = jax.jit(lambda st, data: spec.log_joint(st, data))
    return _LOG_JOINT[id(spec)]


def draw_state(spec, rep: int):
    k = random.fold_in(KEY, 1000 + rep)
    st = spec.prior_sample(random.fold_in(k, 0))
    data = spec.simulate(random.fold_in(k, 1), st)
    return st, data


def with_vol(st, **leaves):
    gv = dataclasses.replace(st.glob.vol, **{k: jnp.asarray(v) for k, v in leaves.items()})
    return dataclasses.replace(st, glob=dataclasses.replace(st.glob, vol=gv))


def log_q_of(spec, st, data) -> jax.Array:
    """``log q_t`` from WP30's reference residuals and the state's ``λ_i`` (the step's
    ``csv_q`` internal)."""
    e = LJ.residuals(st, data, spec.prior, spec.static)
    tm = jnp.asarray(data.tmask)
    return step_of(spec).impl("csv_q")(log_e2(e, tm[None, :]), st.eq.vol.core.log_lam, tm)


def layout(step: CS.CommonSVStep, o: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in step.block_bounds(b, o)) for b in range(step.n_blocks)]


def dense_kh(rho: float, s2: float, T: int) -> np.ndarray:
    """``K_h`` (``T × T``) from the AR(1) definition: ``H_ρᵀ S⁻¹ H_ρ`` (N08 §A5.2)."""
    H = np.eye(T) - rho * np.eye(T, k=-1)
    S = np.full(T, s2)
    S[0] = s2 / (1.0 - rho * rho)
    return H.T @ np.diag(1.0 / S) @ H


def log_normal_kernel(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> float:
    """``log N(x; mean, prec⁻¹)`` up to its normalising constant (it cancels in an independence
    MH identity, where both directions use the same proposal)."""
    d = x - mean
    return float(-0.5 * d @ prec @ d)


_JIT: dict[int, tuple] = {}


def jitted(step: CS.CommonSVStep):
    """``(block_of, proposal, draw_block)`` of ``step`` under ``jit`` (the tests call them per
    block; eager dispatch of the Newton ``while_loop`` would dominate the run time)."""
    key = id(step)
    if key not in _JIT:
        _JIT[key] = (step, jax.jit(step.block_of), jax.jit(step.proposal),
                     jax.jit(step.draw_block))  # fmt: skip
    return _JIT[key][1:]


def tridiag_dense(diag: np.ndarray, off: np.ndarray) -> np.ndarray:
    return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)


# --------------------------------------------------------------------------- protocol, plan, keys


def test_step_protocol_registry_and_compose_guard():
    """The step is a post ``GlobalStep`` at ``G_COMMON_SV`` reading ``eq.vol``, ``glob.vol`` and
    writing ``glob.vol`` (P3 §5.2); the vol block brings it (``global_steps``), so ``compose``'s
    E149 guard passes, and the Constant core reads the common ``h`` (``uses_global_vol``)."""
    from rai_bvar.engines.mcmc import registry

    assert registry.load("common_sv") is CS.CommonSVStep
    spec = spec_of("structural_commonsv")
    program = program_of(spec)
    step = step_of(spec)
    assert isinstance(step, GlobalStep) and step.phase == "post"
    assert step.reads == frozenset({"eq.vol", "glob.vol"}) and step.writes == {"glob.vol"}
    assert isinstance(program.vol, SupportsGlobalSteps)
    assert isinstance(program.vol.core, SupportsGlobalVol) and program.vol.core.uses_global_vol
    assert not program.vol.constant_weights()
    plan = build_plan(program)
    assert plan.names[-1] == "common_sv"
    phase2 = compose(spec_of("structural_constant", "small").static)
    assert phase2.vol.global_steps() == () and not phase2.vol.core.uses_global_vol


@pytest.mark.parametrize(
    "name", ["structural_commonsv", "conjugate_commonsv_dummies", "reduced_commonsv"]
)
def test_plan_invariants_and_write_set(name):
    """Plan invariants with the write-set tree diff (P3 §9.2 item 4): the CommonSV step writes
    ``glob.vol`` only, the equation steps leave it alone."""
    spec = spec_of(name, "small")
    program = program_of(spec)
    st, data = draw_state(spec, 0)
    cache = _prepare_fn(program)(data, spec.prior)
    check_invariants(build_plan(program), program, probe=(KEY, jnp.uint32(3), st, data,
                                                           spec.prior, cache))  # fmt: skip


@pytest.mark.parametrize("block,T_eff,T_pad", [(100, 50, 64), (7, 50, 64), (7, 9, 16), (2, 1, 8),
                                               (5, 20, 32)])  # fmt: skip
def test_block_layout_covers_every_real_date_once(block, T_eff, T_pad):
    """§5.4's 0-based layout: ``B_0 = [0, o)``, ``B_b = [o + (b−1)·block, o + b·block)`` for
    ``b = 1 … ⌈T_pad/block⌉``, intersected with ``[0, T)``: every real date lies in exactly one
    block for every offset, and the windows of length ``min(block, T_pad)`` hold them."""
    from rai_bvar.engines.mcmc.config import VolStatic

    vol = VolStatic(persistent="constant", common=True, csv_block=block)
    step = CS.CommonSVStep(vol, T_pad)
    assert step.n_blocks == 1 + -(-T_pad // block)
    for o in range(block):
        hits = np.zeros(T_eff, int)
        for lo, hi in layout(step, o):
            lo_, hi_ = min(lo, T_eff), min(hi, T_eff)
            assert hi_ - lo_ <= step.window
            hits[lo_:hi_] += 1
        assert np.all(hits == 1), (o, hits)


def test_h_step_keeps_padding_and_invariants():
    """One production sweep keeps ``h[T_eff:] == 0`` exactly and ``|ρ| < 1`` (P3 §2.2 invariants)."""
    spec = spec_of("structural_commonsv", "small")
    for rep in range(3):
        st, data = draw_state(spec, rep)
        new = spec.step(random.fold_in(KEY, rep), jnp.uint32(rep), st, data)
        assert check_state_invariants(new, spec.static, int(data.T_eff)) == []
        assert np.all(np.asarray(new.glob.vol.h)[int(data.T_eff) :] == 0.0)


# --------------------------------------------------------------------------- Tier U item 1: Newton


@pytest.mark.parametrize("block", [100, 7])
def test_newton_gradient_below_1e_8_on_every_block(block):
    """For every block of the §5.4 layout, given the current ``h`` outside it, the gradient of the
    full ``ℓ(h)`` (dense ``K_h`` from the AR(1) definition) on the block is below 1e-8 at the mode
    the step's Newton returns, and Newton converged within the cap (P3 §9.2 item 1)."""
    spec = spec_of("common_sv_only", csv_block=block)
    step = step_of(spec)
    n = float(spec.static.shapes.n)
    worst, rng = 0.0, np.random.default_rng(block)
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        T = int(data.T_eff)
        lq = np.asarray(log_q_of(spec, st, data))
        gv = st.glob.vol
        rho, s2 = float(gv.rho), float(np.exp(2 * gv.log_sigma))
        Kd = dense_kh(rho, s2, T)
        h = np.asarray(gv.h).copy()
        block_of, proposal, _ = jitted(step)
        for lo, hi in layout(step, int(rng.integers(block))):
            blk = block_of(h, lq, gv.rho, gv.log_sigma, data.T_eff, lo, hi)
            prop = proposal(blk, jnp.float64(n))
            m = np.asarray(blk.mask) > 0
            if not m.any():
                continue
            assert bool(prop.converged) and bool(prop.info.ok)
            idx = int(blk.start) + np.nonzero(m)[0]
            h[idx] = np.asarray(prop.mode)[m]
            g = -0.5 * n + 0.5 * np.exp(lq[:T] - h[:T]) - Kd @ h[:T]
            worst = max(worst, float(np.max(np.abs(g[idx]))))
    print(f"Newton: worst |gradient| at the mode {worst:.2e}")
    assert worst < 1e-8


def _m28_start(log_q, h, mask, n):
    """Mutant M28 (local, never committed as a mutant): Newton started at the current block, one
    iteration (a state-dependent proposal)."""
    return jnp.where(mask > 0, h, 0.0), 1


def test_proposal_is_independent_of_the_current_block():
    """M28's required catcher (P3 §9.6, §9.2 item 1; deterministic): the block proposal
    ``N(ĥ, G⁻¹)`` is bitwise unchanged when the current values inside the block change, and it
    changes with a neighbour. The local M28 seam (start at the current block, one iteration)
    fails the first property."""
    spec = spec_of("common_sv_only", csv_block=7)
    step = step_of(spec)
    m28 = step.with_overrides({"csv_newton_start": _m28_start})
    n = jnp.float64(spec.static.shapes.n)
    st, data = draw_state(spec, 0)
    lq = log_q_of(spec, st, data)
    gv = st.glob.vol
    h0 = np.asarray(gv.h)
    for lo, hi in [(3, 10), (10, 17), (0, 3)]:
        moved = h0.copy()
        moved[lo:hi] += np.linspace(-1.5, 2.0, hi - lo)
        neighbour = h0.copy()
        neighbour[hi] += 0.7
        props = {}
        for key, s in (("prod", step), ("m28", m28)):
            for label, h in (("a", h0), ("b", moved), ("c", neighbour)):
                blk = s.block_of(h, lq, gv.rho, gv.log_sigma, data.T_eff, lo, hi)
                props[key, label] = s.proposal(blk, n)
        for f in ("mode", "diag", "off"):
            a, b = (
                np.asarray(getattr(props["prod", "a"], f)),
                np.asarray(getattr(props["prod", "b"], f)),
            )
            assert np.array_equal(a, b), (lo, hi, f)
        assert not np.array_equal(np.asarray(props["prod", "a"].mode),
                                  np.asarray(props["prod", "c"].mode))  # fmt: skip
        assert not np.array_equal(np.asarray(props["m28", "a"].mode),
                                  np.asarray(props["m28", "b"].mode))  # fmt: skip


def test_a_single_data_row():
    """``T_eff = 1`` (E152): ``K_h`` is the single entry ``(1 − ρ²)/σ_h²``, the block target is the
    dense ``ℓ``, a full step keeps the padding exact and ``ρ`` (``S_xx = 0``; Beta: the uniform
    branch) and ``σ_h²`` (``S = (1 − ρ²)h_0²``) draw finitely under both priors."""
    from rai_bvar.engines.mcmc.config import VolStatic
    from rai_bvar.priors.sv_arrays import common_sv_prior

    rho, s2, n, T_pad = 0.7, 0.2, 3.0, 8
    ls = 0.5 * math.log(s2)
    kd, ko = CS.ar1_precision(jnp.float64(rho), jnp.float64(ls), 1, T_pad)
    assert float(kd[0]) == pytest.approx((1 - rho**2) / s2, rel=1e-15)
    assert np.all(np.asarray(kd[1:]) == 1.0) and not np.any(np.asarray(ko))
    lq = jnp.asarray([0.4] + [0.0] * (T_pad - 1))
    for rho_kind in ("truncated_normal", "beta"):
        vol = VolStatic(persistent="constant", common=True, csv_block=4, csv_rho_prior=rho_kind)
        step = CS.CommonSVStep(vol, T_pad)
        spec = rb.CommonSV(rho_prior=rb.sv_priors.Beta(10.0, 2.0)) if rho_kind == "beta" else (
            rb.CommonSV()
        )  # fmt: skip
        csv = jax.tree.map(jnp.asarray, common_sv_prior(spec, 3))
        blk = step.block_of(jnp.zeros(T_pad), lq, rho, ls, 1, 0, 4)
        assert np.asarray(blk.mask).tolist() == [1.0, 0.0, 0.0, 0.0]
        for x in (-1.3, 0.0, 0.8):
            xw = jnp.asarray([x, 0.0, 0.0, 0.0])
            got = float(CS.h_block_log_target(xw, blk.log_q, n, blk.diag, blk.off, blk.lin,
                                              blk.mask))  # fmt: skip
            ref = -0.5 * n * x - 0.5 * math.exp(0.4 - x) - 0.5 * (1 - rho**2) / s2 * x * x
            assert got == pytest.approx(ref, rel=1e-14, abs=1e-14)
        h, frac, cap, info = step.h_step(random.key(1), jnp.zeros(T_pad), lq, rho, ls, n, 1)
        assert bool(info.ok) and float(frac) in (0.0, 1.0) and float(cap) == 0.0
        assert np.all(np.asarray(h)[1:] == 0.0) and np.isfinite(float(h[0]))
        prop = CS.rho_proposal(*CS.rho_sufficient(h, 1)[:2], jnp.float64(ls), csv, rho_kind)
        assert bool(prop.uniform) == (rho_kind == "beta")
        r_new, _ = step.rho_step(random.key(2), h, jnp.float64(rho), jnp.float64(ls), 1, csv)
        ls_new, ok = step.sigma_step(random.key(3), h, r_new, 1, csv)
        assert abs(float(r_new)) < 1 and bool(ok) and np.isfinite(float(ls_new))


# --------------------------------------------------------------------------- Tier U item 2: h


def _h_cases():
    return [("common_sv_only", 100), ("common_sv_only", 7), ("structural_commonsv", 7)]


@pytest.mark.parametrize("name,block", _h_cases())
def test_density_ratio_h_blocks(name, block):
    """Each ``h`` block's target ``φ_B`` (§5.4) against the reference joint: moving ``h_B`` by
    a random amount changes both by the same (20 states, every block of a random layout; 1e-8)."""
    spec = spec_of(name, csv_block=block)
    step = step_of(spec)
    n = jnp.float64(spec.static.shapes.n)
    worst, rng = 0.0, np.random.default_rng(7 + block)
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        lq = log_q_of(spec, st, data)
        gv = st.glob.vol
        block_of = jitted(step)[0]
        for lo, hi in layout(step, int(rng.integers(block))):
            blk = block_of(gv.h, lq, gv.rho, gv.log_sigma, data.T_eff, lo, hi)
            m = np.asarray(blk.mask) > 0
            if not m.any():
                continue
            idx = int(blk.start) + np.nonzero(m)[0]
            x0 = np.asarray(blk.h)
            x1 = x0 + np.where(m, 0.4 * rng.standard_normal(x0.shape), 0.0)

            def log_q(x, blk=blk):
                return CS.h_block_log_target(jnp.asarray(x), blk.log_q, n, blk.diag, blk.off,
                                             blk.lin, blk.mask)  # fmt: skip

            def joint(x, st=st, data=data, idx=idx, m=m):
                h = np.asarray(st.glob.vol.h).copy()
                h[idx] = np.asarray(x)[m]
                return float(log_joint_of(spec)(with_vol(st, h=h), data))

            worst = max(worst, density_ratio_error(log_q, joint, x0, x1))
    print(f"h block density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8


@pytest.mark.parametrize("name,block", _h_cases())
def test_mh_log_ratio_h_blocks(name, block):
    """The computed ``log α`` of every block move equals ``log π(h*) − log π(h) + log q(h) −
    log q(h*)`` with ``π`` the reference joint and ``q = N(ĥ, G⁻¹)`` (dense) of the step's
    proposal (20 states; 1e-10)."""
    spec = spec_of(name, csv_block=block)
    step = step_of(spec)
    n = jnp.float64(spec.static.shapes.n)
    worst, rng = 0.0, np.random.default_rng(17 + block)
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        lq = log_q_of(spec, st, data)
        gv = st.glob.vol
        block_of, proposal, draw_block = jitted(step)
        for lo, hi in layout(step, int(rng.integers(block))):
            blk = block_of(gv.h, lq, gv.rho, gv.log_sigma, data.T_eff, lo, hi)
            m = np.asarray(blk.mask) > 0
            if not m.any():
                continue
            prop = proposal(blk, n)
            z = jnp.asarray(rng.standard_normal(step.window))
            _, _, log_alpha, ok = draw_block(prop, blk, z, jnp.float64(0.5), n)
            assert bool(ok)
            idx = int(blk.start) + np.nonzero(m)[0]
            mode = np.asarray(prop.mode)[m]
            G = tridiag_dense(np.asarray(prop.diag), np.asarray(prop.off))[np.ix_(m, m)]
            # the proposed block: ĥ + L⁻ᵀz with G = LLᵀ (the draw the step evaluates)
            dev, _, _ = LA.tridiag_precision_sample(prop.diag, prop.off, jnp.zeros_like(z), z)
            hstar = mode + np.asarray(dev)[m]
            x_old = np.asarray(blk.h)[m]

            def target(x, st=st, data=data, idx=idx):
                h = np.asarray(st.glob.vol.h).copy()
                h[idx] = x
                return float(log_joint_of(spec)(with_vol(st, h=h), data))

            def prop_density(to, frm, mode=mode, G=G):
                return log_normal_kernel(to, mode, G)

            err = mh_ratio_error(lambda x, xp, la=float(log_alpha): la, target, prop_density,
                                 x_old, hstar)  # fmt: skip
            worst = max(worst, err)
    print(f"h block MH identity: worst scaled error {worst:.2e}")
    assert worst <= 1e-10


# --------------------------------------------------------------------------- Tier U item 2: ρ


def _rho_target(spec, st, data):
    def target(r):
        return float(log_joint_of(spec)(with_vol(st, rho=jnp.float64(r)), data))

    return target


def _rho_prop_density(prop):
    prec, mean, uniform = float(prop.prec), float(prop.mean), bool(prop.uniform)

    def log_prop(to, frm):  # an independence proposal on (−1, 1); constants cancel
        return 0.0 if uniform else -0.5 * prec * (float(to) - mean) ** 2

    return log_prop


def _rho_alpha(step, spec, st, kind):
    gv = st.glob.vol
    csv = spec.prior.shared.csv
    T = int(np.sum(np.asarray(spec.tmask)))
    s_xy, s_xx, h0 = CS.rho_sufficient(gv.h, T)
    prop = CS.rho_proposal(s_xy, s_xx, gv.log_sigma, csv, kind)

    def log_alpha(r_old, r_new):
        return float(step.impl("csv_rho_accept")(jnp.float64(r_new), jnp.float64(r_old), h0,
                                                  gv.log_sigma, s_xy, s_xx, csv, prop.uniform,
                                                  kind))  # fmt: skip

    return prop, log_alpha


@pytest.mark.parametrize("kind", ["truncated_normal", "beta"])
def test_mh_log_ratio_rho(kind):
    """``log α`` of the ρ move equals the target and proposal densities (20 states, three
    proposals each; 1e-10): the TN proposal ``TN(ρ̂, K_ρ⁻¹; −1, 1)`` under both priors, with the
    stationary-start factor ``g`` and, under Beta, the Beta log-density ratio."""
    spec = spec_of("common_sv_only", csv_rho_prior=kind)
    step = step_of(spec)
    worst, rng = 0.0, np.random.default_rng(23)
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        prop, log_alpha = _rho_alpha(step, spec, st, kind)
        assert not bool(prop.uniform)
        for _ in range(3):
            r_new = float(step.rho_draw(prop, jnp.float64(rng.uniform(0.01, 0.99))))
            err = mh_ratio_error(log_alpha, _rho_target(spec, st, data), _rho_prop_density(prop),
                                 float(st.glob.vol.rho), r_new)  # fmt: skip
            worst = max(worst, err)
    print(f"rho MH identity ({kind}): worst scaled error {worst:.2e}")
    assert worst <= 1e-10


def _pinned_state(spec, st, h_tail):
    h = np.zeros_like(np.asarray(st.glob.vol.h))
    T = int(np.sum(np.asarray(spec.tmask)))
    for j, v in enumerate(h_tail):
        h[T - len(h_tail) + j] = v
    return with_vol(st, h=h)


@pytest.mark.parametrize("kind", ["truncated_normal", "beta"])
def test_rho_pinned_case_zero_sxx(kind):
    """§9.2 item 2, first pinned case (E117, E158): ``h_0 … h_{T−2} = 0`` (``S_xx = S_xy = 0``).
    TruncatedNormal keeps the TN proposal (``K_ρ = 1/V_ρ``), Beta takes the uniform branch; the
    MH identity holds on either (1e-10)."""
    spec = spec_of("common_sv_only", csv_rho_prior=kind)
    step = step_of(spec)
    st0, data = draw_state(spec, 0)
    sig = float(np.exp(st0.glob.vol.log_sigma))
    st = _pinned_state(spec, st0, [0.0, 0.8 * sig])
    prop, log_alpha = _rho_alpha(step, spec, st, kind)
    assert bool(prop.uniform) == (kind == "beta")
    if kind == "truncated_normal":
        assert float(prop.prec) == pytest.approx(1.0 / float(spec.prior.shared.csv.rho_var))
    worst = 0.0
    for u in (0.03, 0.4, 0.77, 0.98):
        r_new = float(step.rho_draw(prop, jnp.float64(u)))
        worst = max(worst, mh_ratio_error(log_alpha, _rho_target(spec, st, data),
                                          _rho_prop_density(prop), float(st.glob.vol.rho), r_new))  # fmt: skip
    print(f"rho pinned case S_xx = 0 ({kind}): worst scaled error {worst:.2e}")
    assert worst <= 1e-10


def test_rho_pinned_case_uniform_branch_with_a_data_term():
    """§9.2 item 2, second pinned case (E158), under Beta: ``h_0 … h_{T−3} = 0``, ``h_{T−2} =
    10⁻⁵σ_h``, ``h_{T−1} = 10³σ_h`` give ``S_xx/σ_h² = 10⁻¹⁰ < 10⁻⁸`` and ``S_xy/σ_h² = 10⁻²``:
    the uniform proposal with a non-zero data term, so a missing prior or data term fails."""
    spec = spec_of("common_sv_only", csv_rho_prior="beta")
    step = step_of(spec)
    st0, data = draw_state(spec, 1)
    sig = float(np.exp(st0.glob.vol.log_sigma))
    st = _pinned_state(spec, st0, [1e-5 * sig, 1e3 * sig])
    prop, log_alpha = _rho_alpha(step, spec, st, "beta")
    T = int(np.sum(np.asarray(spec.tmask)))
    s_xy, s_xx, _ = CS.rho_sufficient(st.glob.vol.h, T)
    assert float(s_xx) / sig**2 == pytest.approx(1e-10, rel=1e-9)
    assert float(s_xy) / sig**2 == pytest.approx(1e-2, rel=1e-9)
    assert bool(prop.uniform)
    worst = 0.0
    target = _rho_target(spec, st, data)
    for u in (0.03, 0.4, 0.77, 0.98):
        r_new = float(step.rho_draw(prop, jnp.float64(u)))
        assert r_new == pytest.approx(-1.0 + 2.0 * u)
        worst = max(worst, mh_ratio_error(log_alpha, target, _rho_prop_density(prop),
                                          float(st.glob.vol.rho), r_new))  # fmt: skip
        # a missing data term would be off by (S_xy Δρ − ½S_xx Δρ²)/σ² ≈ 1e-2·Δρ ≫ 1e-10
    print(f"rho pinned uniform case: worst scaled error {worst:.2e}")
    assert worst <= 1e-10


def test_rho_proposal_rejects_outside_the_stationary_region():
    """A draw with ``|ρ*| ≥ 1`` rejects (``log α = −∞``)."""
    spec = spec_of("common_sv_only")
    st, _ = draw_state(spec, 0)
    gv, csv = st.glob.vol, spec.prior.shared.csv
    la = CS.csv_rho_accept(jnp.float64(1.0), gv.rho, gv.h[0], gv.log_sigma, 0.0, 0.0, csv,
                           jnp.bool_(False), "truncated_normal")  # fmt: skip
    assert float(la) == -np.inf


# --------------------------------------------------------------------------- Tier U item 2: σ_h²


@pytest.mark.parametrize("kind", ["inv_gamma", "scaled_chi2"])
def test_density_ratio_sigma2(kind):
    """``σ_h² | h, ρ`` (IG, or GIG under ScaledChi2) as a density of ``σ_h`` (Jacobian ``2σ_h``)
    against the reference joint (20 states; 1e-8)."""
    spec = spec_of("common_sv_only", csv_sigma2_prior=kind)
    worst, rng = 0.0, np.random.default_rng(31)
    T = int(np.sum(np.asarray(spec.tmask)))
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        gv = st.glob.vol
        c = CS.sigma2_conditional(gv.h, gv.rho, T, spec.prior.shared.csv, kind)

        def log_q(ls, c=c):
            s2 = math.exp(2.0 * ls)
            if kind == "inv_gamma":
                a, b = float(c.shape), math.exp(float(c.log_scale))
                lp = a * math.log(b) - float(gammaln(a)) - (a + 1.0) * math.log(s2) - b / s2
            else:
                lam, psi, chi = float(c.lam), math.exp(float(c.log_psi)), math.exp(float(c.log_chi))
                lp = (lam - 1.0) * math.log(s2) - 0.5 * (psi * s2 + chi / s2)
            return lp + LOG2 + ls  # density of σ_h: 2σ_h p(σ_h²), in logs

        def joint(ls, st=st, data=data):
            return float(log_joint_of(spec)(with_vol(st, log_sigma=jnp.float64(ls)), data))

        x0 = float(gv.log_sigma)
        worst = max(worst, density_ratio_error(log_q, joint, x0, x0 + 0.3 * rng.standard_normal()))
    print(f"sigma2 ({kind}) density ratio: worst scaled error {worst:.2e}")
    assert worst <= 1e-8


# --------------------------------------------------------------------------- the common cores


def eq_views(program, st, data, prior, cache, i):
    """One equation's ``(ctx, pm, cm, eq_i, prior_i, cache_i, e)`` as the sweep builds them."""
    s, mean, coef, vol = program.structure, program.mean, program.coef, program.vol
    take = lambda t: jax.tree.map(lambda x: x[i], t)  # noqa: E731
    eq_i, prior_i, cache_i = take(st.eq), take(prior.eq), take(cache.eq)
    ytil = s.pre_residual(st.glob, data, mean.fitted_reduced(st.glob, data))
    ctx = s.context(jnp.int32(i), ytil, eq_i, vol, data, st.glob.vol)
    cm = s.contemp_moments(ctx, prior_i)
    pm = coef.moments_eq(eq_i.coef, jnp.int32(i), prior_i, st.glob.coef)
    a = eq_i.struct.a
    g = mean.fitted_eq(eq_i.mean, a, ctx, cache, prior.shared)
    e = s.residual(a, ctx, g)
    return ctx, pm, cm, eq_i, prior_i, cache_i, e


def core_conditional(program, st, data, prior, cache, i):
    """``(shape, scale)`` of ``λ_i | ·``: the mean's coupling and the core's conditional with the
    common ``h`` (what the vol step draws from)."""
    ctx, pm, cm, eq_i, prior_i, _, e = eq_views(program, st, data, prior, cache, i)
    a = eq_i.struct.a
    if program.static.vol.coupled:
        cpl = program.mean.coupling(eq_i.mean.delta, a, ctx, pm, cm, cache=cache,
                                    shared=prior.shared)  # fmt: skip
    else:
        cpl = Coupling.zeros()
    tm = jnp.asarray(data.tmask)
    return program.vol.core.conditional(log_e2(e, tm), tm, data.T_eff, prior_i, cpl, st.glob.vol)


@pytest.mark.parametrize("name", ["structural_commonsv", "conjugate_commonsv_dummies",
                                  "conjugate_niw_only", "reduced_commonsv"])  # fmt: skip
def test_density_ratio_common_core(name):
    """``λ_i | e_i, h (, θ_i, rows)`` (P3 §5.2–§5.3): the IG of the common core against the
    reference joint, Mode I, reduced, and coupled with SOC/SUR rows (``m > 0``: mutant M36's
    catcher) (20 states, every equation; 1e-8)."""
    spec = spec_of(name)
    program = program_of(spec)
    worst, rng = 0.0, np.random.default_rng(41)
    for rep in range(N_STATES):
        st, data = draw_state(spec, rep)
        cache = _prepare_fn(program)(data, spec.prior)
        for i in range(spec.static.shapes.n):
            shape, scale = core_conditional(program, st, data, spec.prior, cache, i)
            a, b = float(shape), float(scale)

            def log_q(ll, a=a, b=b):
                return a * math.log(b) - float(gammaln(a)) - (a + 1.0) * ll - b * math.exp(-ll)

            def joint(ll, st=st, data=data, i=i):
                lam = np.asarray(st.eq.vol.core.log_lam).copy()
                lam[i] = ll
                core = dataclasses.replace(st.eq.vol.core, log_lam=jnp.asarray(lam))
                eq = dataclasses.replace(st.eq, vol=dataclasses.replace(st.eq.vol, core=core))
                return float(log_joint_of(spec)(dataclasses.replace(st, eq=eq), data))

            x0 = float(st.eq.vol.core.log_lam[i])
            worst = max(
                worst, density_ratio_error(log_q, joint, x0, x0 + 0.3 * rng.standard_normal())
            )
    print(f"common core density ratio ({name}): worst scaled error {worst:.2e}")
    assert worst <= 1e-8


def test_coupled_common_core_seam_reaches_the_coupling():
    """``csv_lambda_conditional`` (P3 §2.4; mutants M31, M36) forms the coupled common core's
    coefficient terms: its production value is the θ part plus the rows' part (``n_active = q_i
    + m``), and an override reaches the program's coupling."""
    spec = spec_of("conjugate_commonsv_dummies")
    program = program_of(spec)
    st, data = draw_state(spec, 0)
    cache = _prepare_fn(program)(data, spec.prior)
    i = 2
    ctx, pm, cm, eq_i, *_ = eq_views(program, st, data, spec.prior, cache, i)
    cpl = program.mean.coupling(eq_i.mean.delta, eq_i.struct.a, ctx, pm, cm, cache=cache,
                                shared=spec.prior.shared)  # fmt: skip
    k, m = spec.static.shapes.k, spec.static.mean.n_dummy
    assert float(cpl.n_active) == k + i + m
    no_rows = program.mean.with_overrides({"csv_lambda_conditional": lambda theta, rows: theta})
    cpl0 = no_rows.coupling(eq_i.mean.delta, eq_i.struct.a, ctx, pm, cm, cache=cache,
                            shared=spec.prior.shared)  # fmt: skip
    assert float(cpl0.n_active) == k + i and float(cpl0.quad) < float(cpl.quad)


# --------------------------------------------------------------------------- the shared factor


def test_shared_factor_equals_the_weighted_path_on_the_same_noise():
    """P3 §9.2 item 1: the shared-factor draw (§5.3) against the weighted path (Mode II, weights
    ``m_t e^{−h_t}/λ_i``, rows ``1/λ_i``) on the same ``z``, every equation of 10 prior states
    with SOC/SUR rows, within ``2e-12 + 5εκ(1 + t)`` posterior s.d. (``κ = cond(sKs)``)."""
    spec = spec_of("conjugate_commonsv_dummies")
    program = program_of(spec)
    assert spec.static.mean.const_path == "shared_factor"
    w_static = dataclasses.replace(
        spec.static, mean=dataclasses.replace(spec.static.mean, const_path="weighted")
    )
    w_program = compose(w_static)
    rng = np.random.default_rng(51)
    worst = 0.0
    for rep in range(10):
        st, data = draw_state(spec, rep)
        cache = _prepare_fn(program)(data, spec.prior)
        w_cache = _prepare_fn(w_program)(data, spec.prior)
        for i in range(spec.static.shapes.n):
            ctx, pm, cm, _, _, cache_i, _ = eq_views(program, st, data, spec.prior, cache, i)
            z = jnp.asarray(rng.standard_normal(spec.static.shapes.q))
            th_s, info = program.mean.draw_core(z, ctx, pm, cm, cache_i, cache, spec.prior.shared)
            th_w, _ = w_program.mean.draw_core(z, ctx, pm, cm, cache_i, w_cache, spec.prior.shared)
            cond = w_program.mean.linear_conditional(ctx, pm, cm, w_cache, spec.prior.shared)
            act = np.asarray(jnp.concatenate([pm.cmask, cm.cmask])) > 0
            Kd, bd = np.asarray(cond.K), np.asarray(cond.b)
            Ka = Kd[np.ix_(act, act)]
            C = np.linalg.inv(Ka)
            sd = np.sqrt(np.diag(C))
            sc = 1.0 / np.sqrt(np.diag(Ka))
            kappa = float(np.linalg.cond(sc[:, None] * Ka * sc[None, :]))
            t = float(np.max(np.abs(C @ bd[act]) / sd))
            err = np.max(np.abs(np.asarray(th_s)[act] - np.asarray(th_w)[act]) / sd)
            tol = 2e-12 + 5.0 * EPS * kappa * (1.0 + t)
            assert bool(info.ok) and np.all(np.asarray(th_s)[~act] == 0.0)
            worst = max(worst, err / tol)
            assert err <= tol, (rep, i, err, tol)
    print(f"shared factor vs weighted: worst error/tolerance {worst:.3f}")


def _cholesky_shapes(jaxpr) -> list[tuple[int, ...]]:
    out = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "cholesky":
            out.append(tuple(eqn.invars[0].aval.shape))
        for sub in jax.core.subjaxprs(eqn) if hasattr(jax.core, "subjaxprs") else ():
            out += _cholesky_shapes(sub)
        for v in eqn.params.values():
            inner = getattr(v, "jaxpr", None)
            if inner is not None and hasattr(inner, "eqns"):
                out += _cholesky_shapes(inner)
            elif hasattr(v, "eqns"):
                out += _cholesky_shapes(v)
    return out


def test_shared_factor_is_factored_once_per_sweep():
    """E87: the shared factor is built from closed-over, unbatched inputs, so inside the vmapped
    equation body its Cholesky is one ``q × q`` factorisation (not ``n`` of them); the weighted
    path factors ``n × q × q``."""
    from rai_bvar.engines.mcmc.sweep import equation_phase, sweep_values

    def shapes(spec_static):
        program = compose(spec_static)
        spec = spec_of("conjugate_commonsv_dummies")
        st, data = draw_state(spec, 0)
        cache = _prepare_fn(program)(data, spec.prior)
        plan = build_plan(program)

        def run(st):
            sh = sweep_values(program, KEY, jnp.uint32(0), st, data, spec.prior, cache)
            return equation_phase(program, plan.equation, sh, st, spec.prior, cache, True)[0]

        return _cholesky_shapes(jax.make_jaxpr(run)(st).jaxpr)

    spec = spec_of("conjugate_commonsv_dummies")
    q, n = spec.static.shapes.q, spec.static.shapes.n
    assert shapes(spec.static) == [(q, q)]
    w_static = dataclasses.replace(
        spec.static, mean=dataclasses.replace(spec.static.mean, const_path="weighted")
    )
    assert shapes(w_static) == [(n, q, q)]


def test_shared_factor_rows_seam_moves_the_draw():
    """The ``shared_factor_rows`` internal (mutant M32: rows weighted by ``e^{−h_0}``) reaches the
    shared Gram: with ``h_0 ≠ 0`` the overridden draw differs from production by far more than the
    weighted-path tolerance."""
    spec = spec_of("conjugate_commonsv_dummies")
    program = program_of(spec)
    m32 = program.mean.with_overrides(
        {"shared_factor_rows": lambda w, Fd, Yd: (jnp.sqrt(w[0]) * Fd, jnp.sqrt(w[0]) * Yd)}
    )
    st, data = draw_state(spec, 3)
    st = with_vol(st, h=np.where(np.arange(st.glob.vol.h.shape[0]) < int(data.T_eff), 1.5, 0.0))
    cache = _prepare_fn(program)(data, spec.prior)
    ctx, pm, cm, _, _, cache_i, _ = eq_views(program, st, data, spec.prior, cache, 2)
    z = jnp.ones(spec.static.shapes.q)
    a, _ = program.mean.draw_core(z, ctx, pm, cm, cache_i, cache, spec.prior.shared)
    b, _ = m32.draw_core(z, ctx, pm, cm, cache_i, cache, spec.prior.shared)
    assert float(jnp.max(jnp.abs(a - b))) > 1e-3


# --------------------------------------------------------------------------- init and E163


def test_cold_start_of_the_common_volatility():
    """P3 §5.5, E133: ``h = 0``, ``ρ`` the prior centre clipped to ``(−0.99, 0.99)``, ``log σ_h``
    the prior mean's log jittered per chain, accumulators zero; the state passes the invariants."""
    from rai_bvar.engines.mcmc.runner import make_runner

    for name, kw in (("structural_commonsv", {}), ("conjugate_commonsv_parity", {})):
        spec = spec_of(name, "small", **kw)
        fns = make_runner(spec.static)
        _, data = draw_state(spec, 0)
        cache = fns.prepare(data, spec.prior)
        st0 = fns.init(K.init_keys(1, 2, 3), data, spec.prior, cache)
        gv, csv = st0.glob.vol, spec.prior.shared.csv
        assert np.all(np.asarray(gv.h) == 0.0)
        if spec.static.vol.csv_rho_prior == "truncated_normal":
            centre = float(csv.rho_mean)
            s2 = float(csv.s2_scale) / (float(csv.s2_shape) - 1.0)
        else:
            centre = 2.0 * float(csv.rho_a) / float(csv.rho_a + csv.rho_b) - 1.0
            s2 = float(csv.s2_B)
        assert np.allclose(np.asarray(gv.rho), np.clip(centre, -0.99, 0.99))
        ls = np.asarray(gv.log_sigma)
        assert len(set(ls.tolist())) == 3 and abs(np.mean(ls) - 0.5 * math.log(s2)) < 1.0
        for leaf in ("acc_csv_h", "acc_csv_rho", "acc_csv_newton", "rb_ch_sum", "rb_ch_sq"):
            assert not np.any(np.asarray(getattr(gv, leaf)))
        assert check_state_invariants(jax.device_get(st0), spec.static, int(data.T_eff)) == []


def test_kernel_module_imports_inside_a_trace():
    """E163: a fresh interpreter whose first import of the kernel module happens inside a traced
    step (``validation.joint`` composes lazily) runs block 407."""
    import subprocess
    import sys

    code = (
        "import jax, jax.numpy as jnp\n"
        "from rai_bvar.validation.joint import joint_spec\n"
        "s = joint_spec('common_sv_only')\n"
        "st = s.prior_sample(jax.random.key(0)); d = s.simulate(jax.random.key(1), st)\n"
        "out = jax.jit(lambda a, b: s.step(jax.random.key(2), jnp.uint32(0), a, b))(st, d)\n"
        "assert int(out.status.code) == 0\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]


# --------------------------------------------------------------------------- block Geweke (Tier B)

BLOCKS = ["common_sv_only", "conjugate_niw_only"]


@pytest.mark.geweke
@pytest.mark.parametrize("name", BLOCKS)
def test_block_geweke(name):
    """Block Geweke 407 and 408 (P3 §9.3) at G-small (``C = 1000, M_c = 100, M₁ = 2·10⁵,
    α = 1e-3``; Bonferroni over the configuration's test functions): 407 runs the production
    CommonSV step alone (``kernel="global_step"``) with the coefficients and ``λ`` held, 408 the
    production sweep (shared-factor path and coupled common core) with ``h`` held."""
    spec = joint_spec(name)
    assert spec.kernel == ("global_step" if name == "common_sv_only" else "sweep")
    res = geweke(spec, GewekeConfig(seed=407 + BLOCKS.index(name), **G_SMALL))
    print(f"{name}: {len(res.names)} functions, max |z| = {np.max(np.abs(res.z)):.3f}, "
          f"min p = {np.min(res.p):.2e} (Bonferroni: p > {G_SMALL['alpha']}/{len(res.names)})")  # fmt: skip
    assert res.passed, res.table().head(8)


# --------------------------------------------------------------------------- fit level


def _data(T: int = 90, n: int = 3, seed: int = 5, start: str = "1990Q1"):
    from rai_bvar.data import VARData

    rng = np.random.default_rng(seed)
    y = np.zeros((T, n))
    h = 0.0
    for t in range(1, T):
        h = 0.9 * h + 0.3 * rng.standard_normal()
        y[t] = 0.5 * y[t - 1] + np.exp(0.5 * h) * rng.standard_normal(n) * np.linspace(0.5, 2.0, n)
    df = pd.DataFrame(y, index=pd.period_range(start, periods=T, freq="Q"),
                      columns=[f"y{j}" for j in range(n)])  # fmt: skip
    return VARData.from_frame(df, dict.fromkeys(df.columns, 1))


@pytest.fixture(scope="module")
def fits():
    data = _data()
    settings = rb.MCMCSettings(chains=2, draws=40, burn=40, store="paths")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mode1 = rb.BVAR(vol=rb.CommonSV(), lags=2, mcmc=settings).fit(data)
        conj = rb.BVAR(mean=rb.Linear(rb.Minnesota(form="conjugate")), vol=rb.CommonSV(),
                       lags=2, mcmc=settings).fit(data)  # fmt: skip
    return {"data": data, "mode1": mode1, "conjugate": conj}


def test_fit_paths_monitors_and_storage(fits):
    """D-52 monitors (``common_h_T``, ``common_rho``, ``common_sigma`` primary with every
    ``log λ_i``; ``common_h_0``, ``common_h_mid`` secondary), storage (``posterior/common_h_T``,
    ``rho``, ``sigma``; ``latent_moments/common_h_mean``, ``common_h_sd`` on ``time``), the
    per-chain ``acc_csv_*`` (E132) and the resolved paths of both programs."""
    mode1, conj = fits["mode1"], fits["conjugate"]
    assert mode1.static.mean.const_path == "weighted" and mode1.static.mean.coef_scaling == "none"
    assert conj.static.mean.const_path == "shared_factor" and conj.static.vol.coupled
    for fit in (mode1, conj):
        spec = fit.monitor_spec
        assert {"common_h_T", "common_rho", "common_sigma"} <= set(spec.primary)
        assert {"common_h_0", "common_h_mid"} <= set(spec.secondary)
        assert {"log_lam[y0]", "log_lam[y1]", "log_lam[y2]"} <= set(spec.primary)
        post = fit.posterior["posterior"]
        for name in ("common_h_T", "rho", "sigma"):
            assert post[name].dims == ("chain", "draw"), name
        assert np.all(np.abs(post["rho"].values) < 1) and np.all(post["sigma"].values > 0)
        lat = fit.posterior["latent_moments"]
        T = int(np.asarray(fit.engine_data.T_eff))
        for name in ("common_h_mean", "common_h_sd"):
            assert lat[name].dims == ("chain", "time") and lat.sizes["time"] == T
        assert list(lat["time"].values) == [str(p) for p in fit.periods]
        h_last = lat["common_h_mean"].values[:, -1]
        assert np.all(np.isfinite(h_last))
        stats = fit.posterior["sample_stats"]
        for name in ("acc_csv_h", "acc_csv_rho", "acc_csv_newton"):
            assert stats[name].dims == ("chain",), name
        assert np.all((stats["acc_csv_h"].values > 0) & (stats["acc_csv_h"].values <= 1))
        attrs = fit.diagnostics.attrs
        assert {"csv_h", "csv_rho", "csv_newton"} <= set(attrs)
        assert "csv_h" not in attrs["equations"]
        mon = fit.posterior["monitor"]
        np.testing.assert_array_equal(mon["common_rho"].values[:, ::1][:, fit.draw_index],
                                      post["rho"].values)  # fmt: skip


def test_forecast_adds_the_common_future(fits):
    """Forecasts (P3 §5.5): ``cond_scale² = λ_i e^{h_{T+j}}`` with one common ``h`` path per draw
    (futures looked up by block id), recomputed exactly from the draw and its forecast keys."""
    from rai_bvar.engines.mcmc.runner import forecast_chain_keys, forecast_shocks

    fit = fits["conjugate"]
    fc = fit.forecast(3, draws=8, seed=11)
    scale = np.asarray(fc.dataset["cond_scale"].values)  # (C, D', H, n)
    lam = np.asarray(fit.param_draws.vol.core.log_lam)  # (C, D, n)
    kept = np.asarray(fit.draw_index)
    common = 2.0 * np.log(scale)  # log λ_i + h_{T+j}
    C = scale.shape[0]
    step = next(g for g in fit.program.global_post if g.block == Block.G_COMMON_SV)
    ckeys = forecast_chain_keys(int(fc.dataset.attrs["fseed"]), C)
    from rai_bvar.engines.mcmc.forecast import select_draws

    pos, _ = select_draws(kept.size, C, 8)
    for c in range(C):
        for j, p in enumerate(pos):
            hf = common[c, j] - lam[c, p][None, :]
            np.testing.assert_allclose(hf, hf[:, :1] * np.ones((1, hf.shape[1])), atol=1e-12)
            sh = forecast_shocks(ckeys[c], jnp.uint32(kept[p]), 3, fit.static.shapes.n)
            gd = jax.tree.map(lambda x, c=c, p=p: x[c, p], fit.param_draws.glob)
            fut = step.future(random.fold_in(sh.glob_key, int(Block.G_COMMON_SV)), gd, 3)
            np.testing.assert_allclose(hf[:, 0], np.asarray(fut), atol=1e-12)
    assert "lag_coefs" in fc.dataset.data_vars


def test_in_loop_forecast_matches_the_post_hoc_forecast():
    """In-loop forecasts (P2 §5.8) of a CommonSV fit draw the common future inside the sample
    executable (global futures by block id) and equal the post-hoc forecast bitwise."""
    from rai_bvar.settings import FitPlan

    model = rb.BVAR(mean=rb.Linear(rb.Minnesota(form="conjugate", hierarchy="fixed")),
                    vol=rb.CommonSV(), lags=2,
                    mcmc=rb.MCMCSettings(chains=2, draws=30, burn=30))  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = model.fit(_data(), plan=FitPlan(horizon=2, draws=10, seed=3))
    a = fit.forecast(2, draws=10, seed=3)
    assert int(a.dataset.attrs["in_loop"]) == 1
    fit.in_loop_forecast = None
    b = fit.forecast(2, draws=10, seed=3)
    np.testing.assert_array_equal(a.paths.values, b.paths.values)
    np.testing.assert_array_equal(a.dataset["cond_scale"].values, b.dataset["cond_scale"].values)


def test_parameter_draws_and_irfs_read_the_common_h(fits):
    """``draw_parameters`` and IRFs (P3 §5.5): at the last period ``chol`` has diagonal
    ``exp(½(log λ_i + h_T))``; at another date (stored paths) ``exp(½(log λ_i + h_t))``."""
    fit = fits["mode1"]
    _, chol = fit.draw_parameters(10)
    p = fit.param_draws
    lam = np.asarray(p.vol.core.log_lam).reshape(-1, 3)
    hT = np.asarray(p.glob.vol.h_T).reshape(-1)
    flat = np.round(np.linspace(0, lam.shape[0] - 1, 10)).astype(int)
    np.testing.assert_allclose(np.log(np.diagonal(chol, axis1=1, axis2=2) ** 2),
                               lam[flat] + hT[flat, None], rtol=1e-12, atol=1e-12)  # fmt: skip
    date = fit.periods[10]
    _, chol_t = fit.draw_parameters(4, date=date)
    paths = fit.path_draws
    assert chol_t.shape == (4, 3, 3) and np.all(np.isfinite(chol_t))
    h_paths = np.asarray(paths.h)  # (C, Dp, n, T + 1): log λ_i + h_t, state t + 1 ↔ row t
    kept = fit.store.plan.streams["paths"].draw_index()
    where = np.searchsorted(fit.draw_index, kept)
    lam_p = np.asarray(fit.param_draws.vol.core.log_lam)[:, where]  # (C, Dp, n)
    common = h_paths[..., 11] - lam_p  # h_10, the same for every equation
    np.testing.assert_allclose(common, common[..., :1] * np.ones((1, 1, 3)), atol=1e-10)
    res = fit.irf(4, draws=4, date=date)
    assert np.all(np.isfinite(np.asarray(res.data.values))) and np.all(
        np.isfinite(res.fevd().values)
    )


def test_warm_start_remaps_the_common_h_by_period(fits):
    """Warm starts (P3 §5.5, E109): ``glob.vol.h`` is aligned by period (``TIME_INDEXED``), new
    trailing periods simulated forward with the copied ``(ρ, σ_h)``, padded states 0; ``ρ`` and
    ``log σ_h`` copied."""
    from rai_bvar.engines.mcmc import warmstart as W
    from rai_bvar.engines.mcmc.resolve import resolve, warm_key
    from rai_bvar.settings import Runtime

    assert "glob.vol.h" in W.TIME_INDEXED
    fit = fits["mode1"]
    longer = _data(T=94)
    model = fit.model
    r = resolve(model, longer, Runtime())
    new = W.remap(fit.final_state, r.data, r.periods, r.static, warm_key(0, r.salt))
    old_h = np.asarray(fit.final_state.state.glob.vol.h)
    T_old = int(np.asarray(fit.engine_data.T_eff))
    T_new = int(np.asarray(r.data.T_eff))
    new_h = np.asarray(new.glob.vol.h)
    shift = list(r.periods).index(fit.periods[0])
    np.testing.assert_array_equal(new_h[:, shift : shift + T_old], old_h[:, :T_old])
    assert np.all(new_h[:, T_new:] == 0.0) and np.all(np.isfinite(new_h[:, :T_new]))
    np.testing.assert_array_equal(np.asarray(new.glob.vol.rho),
                                  np.asarray(fit.final_state.state.glob.vol.rho))  # fmt: skip
    assert not np.any(np.asarray(new.glob.vol.acc_csv_h))
    assert check_state_invariants(new, r.static, T_new) == []


def test_monitored_log_prior_equals_the_reference():
    """The monitored ``log p(θ)`` (the cores, means and SOC/SUR rows per equation plus the
    step's ``log p(h, ρ, σ_h)``) equals WP30's ``log_joint.log_prior`` at prior states (1e-10
    relative), under both ρ and both σ_h² priors and under global-local priors (rows 324, 328: the
    scale and CCCM steps' ``log_prior_state`` beside the CommonSV step's
    ``log_prior_data``); with SOC/SUR rows up to the rows' normalising
    constant ``log p(d)``, which the monitor omits (P2 WP25), so the difference is the same at
    every state."""
    from rai_bvar.engines.mcmc.monitor import monitor_values

    offsets: dict[str, float] = {}  # with SOC/SUR rows: equal up to a state-independent constant
    for name, kw in (("structural_commonsv", {}), ("conjugate_commonsv_dummies", {}),
                     ("conjugate_commonsv_parity", {}), ("reduced_commonsv", {}),
                     ("structural_commonsv_hs", {}), ("reduced_commonsv_ng", {})):  # fmt: skip
        spec = spec_of(name, "full", **kw)
        program = program_of(spec)
        for rep in range(3):
            st, data = draw_state(spec, rep)
            cache = _prepare_fn(program)(data, spec.prior)
            got = float(monitor_values(st, data, spec.prior, cache, program)["log_prior"])
            ref = float(LJ.log_prior(st, spec.prior, spec.static, data.tmask))
            if spec.static.mean.n_dummy:  # P2 WP25: the monitor omits the rows' log p(d)
                offsets.setdefault(name, got - ref)
                ref = ref + offsets[name]
            assert abs(got - ref) <= 1e-10 * (1.0 + abs(ref)), (name, got, ref)


def test_resolve_rules_for_common_sv():
    """§2.1: the conjugate form × CommonSV resolves to ``shared_factor`` (``"cholesky"``: the
    weighted path, row 323), Mode I CommonSV to ``weighted``; rule (14)'s array check builds
    ``shared_prec`` and rejects equation-dependent precisions; the CommonSV prior arrays; the
    D-35 budget counts CommonSV (D-51)."""
    from rai_bvar.engines.mcmc.resolve import check_shared_precision, resolve
    from rai_bvar.settings import Runtime

    data = _data()
    settings = rb.MCMCSettings(chains=2, draws=20, burn=10)
    conj = rb.BVAR(mean=rb.Linear(rb.Minnesota(form="conjugate", hierarchy="fixed")),
                   vol=rb.CommonSV(), lags=2, mcmc=settings)  # fmt: skip
    r = resolve(conj, data, Runtime())
    assert r.static.mean.const_path == "shared_factor"
    prec = check_shared_precision(r.prior.eq.coef, r.prior.eq.contemp)
    np.testing.assert_array_equal(np.asarray(r.prior.shared.shared_prec), prec)
    assert prec.shape == (r.static.shapes.q,)
    assert r.warning_inputs["sv_budget"] == (2, 20)
    assert r.warning_inputs.get("order_invariant") is True
    csv = r.prior.shared.csv
    assert float(csv.rho_mean) == 0.9 and float(csv.rho_var) == 0.04
    assert float(csv.s2_shape) == 5.0 and float(csv.s2_scale) == 0.04 and float(csv.n_eq) == 3
    chol = dataclasses.replace(
        conj, mcmc=rb.MCMCSettings(chains=2, draws=20, const_path="cholesky")
    )
    assert resolve(chol, data, Runtime()).static.mean.const_path == "weighted"
    mode1 = rb.BVAR(vol=rb.CommonSV(rho_prior=rb.sv_priors.Beta(10.0, 2.0),
                                    sigma2_prior=rb.sv_priors.ScaledChi2(0.1)), lags=2,
                    mcmc=settings)  # fmt: skip
    r1 = resolve(mode1, data, Runtime())
    assert r1.static.mean.const_path == "weighted" and r1.static.vol.csv_rho_prior == "beta"
    assert isinstance(r1.prior.shared.shared_prec, type(r1.prior.shared.gl))  # both Empty
    assert "order_invariant" not in r1.warning_inputs
    bad = dataclasses.replace(r.prior.eq.coef, prec0=np.asarray(r.prior.eq.coef.prec0) *
                              np.array([[1.0], [1.0], [1.5]]))  # fmt: skip
    with pytest.raises(ValueError, match="cross = 1 and contemp = 1"):
        check_shared_precision(bad, r.prior.eq.contemp)
    with pytest.raises(ValueError, match="weight"):
        check_shared_precision(r.prior.eq.coef, r.prior.eq.contemp, np.array([1.0, 2.0, 1.0]))


def test_r8_note_is_gone_at_cross_one_and_stays_otherwise():
    """E99, E130 (an intended Phase 2 behaviour change): a structural Mode II × ``Constant("iw")``
    fit at ``cross = contemp = 1`` with ``n = 4`` no longer carries the R8 ordering note, and at
    ``cross ≠ 1`` it still does; the conjugate × CommonSV composition has none either (static rule
    on ``shared_factor``)."""
    data = _data(T=60, n=4, seed=8)
    settings = rb.MCMCSettings(chains=2, draws=20, burn=10)
    note = "this composition is not order-invariant"

    def notes(prior, vol):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = rb.BVAR(mean=rb.Linear(prior), vol=vol, lags=1, mcmc=settings).fit(data)
        return [w for w in fit.warnings if w.startswith(note)]

    assert notes(rb.Minnesota(form="structural", hierarchy="fixed"), rb.Constant("iw")) == []
    assert len(notes(rb.Minnesota(form="structural", hierarchy="fixed", cross=0.5),
                     rb.Constant("iw"))) == 1  # fmt: skip
    assert notes(rb.Minnesota(form="conjugate", hierarchy="fixed"), rb.CommonSV()) == []


def test_common_sv_warnings():
    """P3 §8: an ``h`` block acceptance below 0.1 in some chain says to decrease ``block``; any
    Newton cap hit is reported (informational); neither fires on healthy rates."""
    from rai_bvar.engines.mcmc import monitor as M

    spec = spec_of("structural_commonsv", "small")
    settings = rb.MCMCSettings(chains=2, draws=20)

    def catalogue(acc):
        return M.warning_catalogue(M.DiagnosticsInput(static=spec.static, settings=settings,
                                                      acc=acc, chains=2))  # fmt: skip

    good = {"acc_csv_h": np.array([0.8, 0.7]), "acc_csv_rho": np.array([0.9, 0.9]),
            "acc_csv_newton": np.zeros(2)}  # fmt: skip
    assert not [w for w in catalogue(good) if "CommonSV" in w]
    bad = {**good, "acc_csv_h": np.array([0.8, 0.05]), "acc_csv_newton": np.array([0.0, 0.01])}
    msgs = [w for w in catalogue(bad) if "CommonSV" in w]
    assert any("decrease CommonSV(block=" in w for w in msgs)
    assert any("Newton" in w and "informational" in w for w in msgs)
    assert "CommonSV" in M.sv_budget_message(2, 20, common=True)


def test_forecast_lag_coefs_keyword(fits):
    """E162(a): ``MCMCFit.forecast`` takes the keyword ``lag_coefs`` for protocol symmetry and
    always attaches the accelerator variable; a non-bool raises."""
    fit = fits["mode1"]
    a = fit.forecast(2, draws=4, seed=3, lag_coefs=True)
    b = fit.forecast(2, draws=4, seed=3, lag_coefs=False)
    assert "lag_coefs" in a.dataset.data_vars and "lag_coefs" in b.dataset.data_vars
    np.testing.assert_array_equal(a.paths.values, b.paths.values)
    with pytest.raises(TypeError, match="lag_coefs"):
        fit.forecast(2, draws=4, lag_coefs="yes")


def test_draw_template_shows_the_ssvs_indicators():
    """E164(c): ``state.draw_template`` gains ``SSVSDraw(incl int8[n, k])`` in ``GlobDraw.coef``
    for the SSVS kind, the structure ``extract_params`` produces; other kinds keep ``Empty``."""
    from rai_bvar.components.coef.global_local import SSVSDraw
    from rai_bvar.engines.mcmc.runner import extract_params

    spec = spec_of("structural_constant_ssvs", "small")
    tmpl = draw_template(spec.static)
    assert isinstance(tmpl.glob.coef, SSVSDraw)
    assert tmpl.glob.coef.incl.dtype == np.int8
    assert tmpl.glob.coef.incl.shape == (spec.static.shapes.n, spec.static.shapes.k)
    program = program_of(spec)
    st, data = draw_state(spec, 0)
    got = jax.eval_shape(lambda s: extract_params(program, s, data, spec.prior), st)
    assert jax.tree.structure(got) == jax.tree.structure(tmpl)
    assert [(x.shape, x.dtype) for x in jax.tree.leaves(got)] == [
        (np.shape(x), np.asarray(x).dtype) for x in jax.tree.leaves(tmpl)
    ]
    csv = spec_of("structural_commonsv", "small")
    tmpl_c = draw_template(csv.static)
    got_c = jax.eval_shape(
        lambda s: extract_params(program_of(csv), s, draw_state(csv, 0)[1], csv.prior),
        draw_state(csv, 0)[0],
    )
    assert jax.tree.structure(got_c) == jax.tree.structure(tmpl_c)
    assert not isinstance(draw_template(spec_of("structural_commonsv", "small").static).glob.coef,
                          SSVSDraw)  # fmt: skip


def test_step_sweep_context_and_keys():
    """The step reads ``q`` from ``SweepCtx.eq_out.e`` and the cores' ``log λ_i`` (not from
    ``eq_out.log_lam``), consumes its keys through ``Sub.CSV_*`` and is deterministic by key."""
    spec = spec_of("structural_commonsv", "small")
    step = step_of(spec)
    st, data = draw_state(spec, 2)
    e = LJ.residuals(st, data, spec.prior, spec.static)
    junk = EqOut(e, jnp.full_like(e, 123.0))  # log_lam of eq_out is not read
    cache = _prepare_fn(program_of(spec))(data, spec.prior)
    ctx = SweepCtx(data, spec.prior, cache, junk)
    ctx2 = SweepCtx(data, spec.prior, cache, EqOut(e, jnp.zeros_like(e)))
    k = random.key(5)
    a, ia = step.sample(k, st, ctx, None)
    b, _ = step.sample(k, st, ctx2, None)
    c, _ = step.sample(random.key(6), st, ctx, None)
    assert bool(ia.ok)
    np.testing.assert_array_equal(np.asarray(a.glob.vol.h), np.asarray(b.glob.vol.h))
    assert not np.array_equal(np.asarray(a.glob.vol.h), np.asarray(c.glob.vol.h))
    assert isinstance(a.glob.vol, CommonSVGlobal)
    assert jax.tree.structure(a.eq) == jax.tree.structure(st.eq)
    np.testing.assert_array_equal(np.asarray(a.eq.vol.core.log_lam),
                                  np.asarray(st.eq.vol.core.log_lam))  # fmt: skip
    assert int(Sub.CSV_OFFSET) == 20 and int(Sub.CSV_S2) == 25
