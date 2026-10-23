"""TabE1, lower panel (Appendix E): the random-walk AWOL step at sigma = 5e-9, recomputed.

For the scaled RW AWOL system Q~ = sigma^2 Q = Lap_w + diag(delta) of one equation at the synthetic
state of ``test_rw_awol_at_a_tiny_sigma`` (``problem(4, T_eff=88, T_pad=96, rw=True,
sigma=5e-9)`` with ``prior()`` and noise ``rng(41)``, ``tests/mcmc/blocks/test_sv_core.py``), the
panel gives the smallest pivot, log|Q~|, the drawn level h_0 = sigma x_0, the largest error of the
drawn levels h_t over the 89 live states, and whether the engine's failure status would be raised
(a pivot that is not positive or a non-finite draw), for

  * the assembled tridiagonal: ``tridiag_precision_sample`` on ``SV.awol_system``'s (diag, off, b),
    the RW draw before f60d736;
  * the excess-pivot recursion: ``laplacian_precision_sample`` on (delta, w, b), the draw since
    f60d736;
  * the 60-digit reference ``_decimal_laplacian_sample`` of the test module (plain recursion on the
    assembled matrix in 60-digit decimal arithmetic from the same float64 inputs and noise).

Computed on the fly from committed code of the main repository (the same computation as
``make_fig_laplacian_pivots.py`` at one sigma); evidence tag [V] for the recursion and the
reference (asserted by ``test_rw_awol_at_a_tiny_sigma``), recomputation for the assembled path (no
committed test asserts its values yet, OUTLINE TODO 23). The recorded values of commit f60d736 and
dossier 05 (minimum pivot 2.22e-16 against an exact 5.308e-15; log-determinant -36.04 against
-32.8695; h_0 = -73.90 against -3.024) are asserted. JAX runs single-threaded on the CPU.

Usage: tab_rw_pivots.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import tab_common as tc  # noqa: E402

NAME = "tab_rw_pivots"
SIGMA = 5e-9


def compute() -> dict:
    import sys

    sys.dont_write_bytecode = True
    for sub in ("src", "tests/mcmc/blocks"):
        p = str(tc.MAIN_REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import test_sv_core as T
    from rai_bvar.engines.mcmc.linalg import laplacian_precision_sample, tridiag_precision_sample

    sp = T.prior()
    P = T.problem(4, T_eff=88, T_pad=96, rw=True, sigma=SIGMA)
    live, sigma = P["T_eff"] + 1, P["sigma"]
    diag, off, b = (np.asarray(a) for a in T.awol(P, sp, "rw"))
    delta, w, b2 = (np.asarray(a) for a in T.lap(P, sp))
    z = T.rng(41).normal(size=delta.size)
    x0, ld0, p0 = tridiag_precision_sample(*(jnp.asarray(a) for a in (diag, off, b, z)))
    x1, ld1, p1 = laplacian_precision_sample(*(jnp.asarray(a) for a in (delta, w, b2, z)))
    xr, ldr, pr = T._decimal_laplacian_sample(delta, w, b2, z)
    href = sigma * xr[:live]
    out = {"live": live, "T_eff": P["T_eff"], "T_pad": P["T_pad"], "V0": float(sp.V0),
           "m0": float(sp.m0), "ref": {"piv": float(pr[:live].min()), "ld": float(ldr),
                                       "h0": float(href[0]), "argmin": int(np.argmin(pr[:live]))}}  # fmt: skip
    for tag, x, ld, p in (("assembled", x0, ld0, p0), ("laplacian", x1, ld1, p1)):
        x = np.asarray(x)
        p = float(p)
        ok = p > 0 and bool(np.isfinite(x[:live]).all())
        h = sigma * x[:live]
        out[tag] = {"piv": p, "ld": float(ld), "h0": float(h[0]), "ok": ok,
                    "h_err": float(np.max(np.abs(h - href))) if ok else float("inf")}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    d = compute()
    ref, old, new = d["ref"], d["assembled"], d["laplacian"]

    # the recorded diagnostics (commit f60d736; dossier 05 sections 2.2-2.3), to the digits quoted
    assert f"{old['piv']:.3g}" == "2.22e-16" and f"{ref['piv']:.4g}" == "5.308e-15", (old, ref)
    assert f"{old['ld']:.2f}" == "-36.04" and f"{ref['ld']:.4f}" == "-32.8695", (old, ref)
    assert f"{old['h0']:.2f}" == "-73.90" and f"{ref['h0']:.3f}" == "-3.024", (old, ref)
    assert old["ok"] and new["ok"], "the assembled path reports no failure at this state"
    assert ref["argmin"] == d["live"] - 1  # the level direction carries the smallest pivot

    def rel(a: float) -> str:
        return tc.sci(abs(a - ref["piv"]) / ref["piv"], 2)

    rows = [
        (r"Assembled tridiagonal (before \code{f60d736})", old),
        (r"Excess-pivot recursion (since \code{f60d736})", new),
        (r"60-digit reference", None),
    ]
    lines = [r"\begin{tabular}{@{}lrrrrrc@{}}", r"\toprule",
             r"Factorisation of $\tilde Q$ & Min.\ pivot & Rel.\ error & $\log|\tilde Q|$ & "
             r"$h_0 = \sigma x_0$ & $\max_t |h_t - h_t^{\mathrm{ref}}|$ & Failure reported \\",
             r"\midrule"]  # fmt: skip
    for label, r in rows:
        if r is None:
            lines.append(f"{label} & {tc.sci(ref['piv'], 4)} & --- & ${ref['ld']:.4f}$ & "
                         f"${ref['h0']:.3f}$ & --- & --- \\\\")  # fmt: skip
        else:
            lines.append(f"{label} & {tc.sci(r['piv'], 4)} & {rel(r['piv'])} & ${r['ld']:.4f}$ & "
                         f"${r['h0']:.3f}$ & {tc.sci(r['h_err'], 2)} & "
                         f"{'no' if r['ok'] else 'yes'} \\\\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}"]
    foot = ("Recomputed from committed code at the synthetic state of "
            r"\code{tests/mcmc/blocks/test_sv_core.py::test_rw_awol_at_a_tiny_sigma} "
            f"($\\sigma = 5\\times10^{{-9}}$, $T_{{\\mathrm{{eff}}}} = {d['T_eff']}$, "
            f"$T_{{\\mathrm{{pad}}}} = {d['T_pad']}$, {d['live']} live states, $V_0 = {d['V0']:g}$, "
            f"$m_0 = {d['m0']:g}$, noise seed 41), not at the production state of F2. The smallest "
            "pivot is that of the level direction (the last live state). ``Failure reported'': the "
            "engine's status would flag a pivot that is not positive or a non-finite draw. At this "
            "state the assembled factorisation returns a positive pivot and finite draws, so no "
            "failure is reported, while $h_0$ is off by "
            f"{abs(old['h0'] - ref['h0']):.1f}.")  # fmt: skip
    body = tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    tc.write(NAME, body, generator="tab_rw_pivots.py",
             status="recomputed from committed code ([V] recursion and reference; assembled path: "
             "recomputation, TODO 23)",
             sources=["main repo tests/mcmc/blocks/test_sv_core.py (problem, prior, awol, lap, rng, "
                      "_decimal_laplacian_sample)",
                      "main repo src/rai_bvar/engines/mcmc/linalg.py (tridiag_precision_sample, "
                      "laplacian_precision_sample)",
                      f"main repo HEAD {tc.git('rev-parse', '--short', 'HEAD')}"])  # fmt: skip


if __name__ == "__main__":
    main()
