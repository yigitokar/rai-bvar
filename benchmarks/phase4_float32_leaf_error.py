"""Phase 4 week-1 spike: error of bartz's float32 leaf conditional (``docs/phase4/SPIKE.md``, Q2).

One leaf holding ``n`` observations with weights ``w_t = exp(u_t)``, ``u_t ~ U(−R, R)`` (a
volatility ratio of ``e^{2R}`` within the sample), targets ``y_t = c·σ̄ + ε_t/√w_t`` with
``σ̄ = (mean w)^{−½}`` (``c``: the level the trees have not yet fitted, in noise units) and leaf
prior precision ``p₀``. The exact conditional is ``N(m, 1/P)``, ``P = p₀ + Σw``,
``m = Σw·y/P``. bartz's value is recomputed with its own internals on the state the adapter
builds: the per-leaf reduction ``sum_resid`` of ``resid·prec_scale``, the cached ``prec_tree``,
``scaled_error_cov_inv`` and ``_precompute_leaf_terms_uv`` (the float32 ``mean_factor`` and
posterior variance), exactly as ``accept_move_and_sample_leaves`` combines them. Reported: the
largest ``|m_bartz − m|·√P`` (posterior s.d. units) and ``|v_bartz·P − 1|`` over replications,
for path (a) (all bartz storage float32) and path (b) (float64 residuals and precision scales).

    uv run --extra bart python benchmarks/phase4_float32_leaf_error.py
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from rai_bvar.components.mean import _bart_spike as bs
from rai_bvar.validation._bart_spike import bartz_x64_patch


def leaf_conditional(st, x64: bool) -> tuple[float, float]:
    from bartz.mcmcstep._state import scaled_error_cov_inv
    from bartz.mcmcstep._step import _precompute_leaf_terms_uv, sum_resid

    with jax.enable_x64(x64):
        f = st.forest
        size = f.leaf_tree.shape[-1]
        scaled = st.resid * st.prec_scale
        rt = sum_resid(scaled, f.leaf_indices[0], size, st.config.resid_reduction_config, False)
        rt = rt * st.resid_unit + f.prec_tree[0] * (f.leaf_unit * f.leaf_tree[0])
        eci = scaled_error_cov_inv(st)
        pre = _precompute_leaf_terms_uv(
            random.key(0), f.prec_tree[:1], eci, f.leaf_prior_cov_inv.value
        )
        mean = rt[1] * pre.mean_factor[0, 1]
        var = pre.mean_factor[0, 1] / eci
    return float(mean), float(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = a.n
    rows = []
    with bartz_x64_patch():
        for x64 in (False, True):
            prior = bs.BartPrior(num_trees=200, max_depth=2, x64=x64)
            p0 = 1.0 / prior.leaf_sd**2
            tmpl = bs.template_state(prior, 1, n)
            for R in (0.0, 2.0, 4.0, 6.0, 8.0):
                for c in (0.0, 1e1, 1e3, 1e5):
                    worst_m = worst_v = 0.0
                    for r in range(a.reps):
                        rng = np.random.default_rng(1000 * r + 7)
                        w = np.exp(rng.uniform(-R, R, n))
                        sig = 1.0 / np.sqrt(w.mean())
                        y = c * sig + rng.normal(size=n) / np.sqrt(w)
                        st = jax.jit(bs.set_data)(
                            jax.tree.map(jnp.asarray, tmpl), jnp.zeros((1, n), jnp.uint8),
                            jnp.asarray(y), jnp.asarray(w),
                        )  # fmt: skip
                        mean, var = leaf_conditional(st, x64)
                        P = p0 + w.sum()
                        m = (w * y).sum() / P
                        worst_m = max(worst_m, abs(mean - m) * np.sqrt(P))
                        worst_v = max(worst_v, abs(var * P - 1.0))
                    row = {"path": "b" if x64 else "a", "R": R, "ratio": float(np.exp(2 * R)),
                           "c": c, "max_mean_err_sd": worst_m, "max_var_rel_err": worst_v}  # fmt: skip
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"n": n, "reps": a.reps, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
