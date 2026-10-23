"""TabD4 (Appendix D): analytic twins and exact oracles A1--A7.

One row per twin: the oracle, the statistic and tolerance, the negative control (what it perturbs,
and that it must fail), the cadence, the tests, and the result on the release commit.

Data. The descriptions follow dossier 04 section 9 and the test modules' docstrings; each number a
row quotes is checked to occur in the source text of the cited module (``EVIDENCE`` below), and
every cited test function must exist in that module at the main repository's HEAD (``ast``). The
cadence is derived from the tests' ``pytest.mark.slow`` decorators: PR for the tests without it,
weekly for those with it (the CI job named "nightly" runs weekly; ``.github/workflows/ci.yml``).
The result column reads pytest's JUnit XML of the release run, ``docs/paper/data/release/
twins.xml`` (OUTLINE TODO 7: the twins have no record hook); until it exists it is a red [REL]
mark.

Usage: tab_twins.py [--outdir DIR] [--junit PATH]
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import tab_common as tc

NAME = "tab_twins"
M_TWIN = "tests/analytic/test_mcmc_twin.py"
M_QUAD = "tests/analytic/test_mode1_quadrature.py"
M_NIG = "tests/analytic/test_structural_nig.py"
M_TOY = "tests/mcmc/blocks/test_sv_toy_quadrature.py"
M_CCCM = "tests/mcmc/blocks/test_cccm.py"

# (id, oracle, statistic and tolerance, negative control, tests [(module, function)],
#  evidence [(module, text that must occur in its source)])
TWINS = [
    ("A1",
     r"Structural NIG log ML \citep{chan2022asymmetric} $=$ Phase~1 analytic log ML (Mode~II, "
     r"$\kappa_c = \kappa_a = 1$), with and without SOC/SUR; EB $\lambda$ at \code{cross=1}; MCMC "
     r'Mode~II $\times$ \code{Constant("iw")} with dummy rows against the Phase~1 NIW posterior',
     r"log ML abs.\ $10^{-10}$; EB modes abs.\ $10^{-6}$; dummy-row extension: CDF-form quantiles, "
     r"$|z| \leq \max(4, z_{\mathrm{Bonf}})$",
     r"the same draws against the posterior without the rows must fail",
     [(M_NIG, "test_a1_log_ml_equals_phase1"), (M_NIG, "test_a1_eb_equals_phase1_eb_at_cross_one"),
      (M_NIG, "test_a1_holds_when_the_level_dwarfs_the_noise"),
      (M_TWIN, "test_a1_dummies_mode2_iw_matches_the_phase1_posterior_with_dummies"),
      (M_TWIN, "test_a1_dummies_comparison_sees_the_rows"),
      (M_TWIN, "test_a1_dummies_other_constant_weight_paths"), (M_TWIN, "test_a1_dummies_realistic_size")],
     [(M_NIG, "abs=1e-10"), (M_NIG, "abs=1e-6"), (M_TWIN, "*without* the rows")]),
    ("A2",
     r'MCMC Mode~II $\times$ \code{Constant("iw")} against the Phase~1 NIW posterior ($B$, $\Sigma$, '
     r"$A$, $\Lambda$), $n = 3$, $p = 2$, $T = 60$",
     r"CDF-form quantiles (5--95\%), $|z| \leq \max(4, z_{\mathrm{Bonf}})$ at family level $10^{-2}$",
     r"NIW posterior with one extra degree of freedom (about 1.5\% on $\Sigma$) must fail",
     [(M_TWIN, "test_a2_mode2_iw_matches_the_phase1_niw_posterior"),
      (M_TWIN, "test_a2_rejects_a_slightly_wrong_posterior"), (M_TWIN, "test_a2_other_constant_weight_paths")],
     [(M_TWIN, "~1.5% on the scale of ``Σ``"), (M_TWIN, "``n = 3, p = 2, T = 60``"),
      (M_TWIN, "family level 1e-2")]),
    ("A3", r"as A2 at $n = 20$, $p = 4$, $T = 300$ (the realistic-size oracle)",
     r"as A2; about $10^{4}$ quantiles at family level $10^{-2}$", "---",
     [(M_TWIN, "test_a3_realistic_size")],
     [(M_TWIN, "``n = 20, p = 4, T = 300``"), (M_TWIN, "~10⁴ of A3")]),
    ("A4",
     r"Mode~II with $\kappa_c = 0.5$ \citep[asymmetric conjugate prior, not an NIW;][]{chan2022asymmetric} "
     r"against the closed form: $t_{2\hat\nu}$, $\mathrm{IG}(\hat\nu, \hat S)$ and iid draws",
     "as A2", "---",
     [(M_TWIN, "test_a4_chan_asymmetric_prior")],
     [(M_TWIN, "``κ_c = 0.5``"), (M_TWIN, "``t_{2ν̂}``")]),
    ("A5",
     r"Mode~I $\times$ Constant: per-equation 1-D quadrature of $p(\lambda_i \mid y)$, exact "
     r"$\mathrm{E}[\lambda_i]$, $\mathrm{E}[\theta_i]$, $\operatorname{Var}(\theta_i)$; the quadrature "
     r"checks itself on every call",
     r"$|z| \leq \max(4, z_{\mathrm{Bonf}})$ with the MCSE of the mean",
     r"inverse-gamma shape off by $\frac12$ (about 1.6\% in $\mathrm{E}[\lambda_i]$) must fail",
     [(M_QUAD, "test_a5_mode1_constant_matches_quadrature"), (M_QUAD, "test_a5_rejects_a_slightly_wrong_posterior"),
      (M_QUAD, "test_a5_mode1_constant_other_paths"), (M_QUAD, "test_a5_mode1_constant_n20"),
      (M_QUAD, "test_a5_extension_iw_equivalent"), (M_QUAD, "test_quadrature_reproduces_the_structural_nig_closed_form")],
     [(M_QUAD, "a ~1.6% shift of ``E[λ_i]``"), (M_QUAD, "off by\n    ½")]),
    ("A6",
     r"toy SV posteriors by 2-D quadrature or SIR: exact-mode invariance of the $h$ step, the ASIS "
     r"correction, the step order, T13",
     r"moments at the exact values, $|z| < 4.5$",
     r"wrong kernels ($\code{exact=False}$ with a one-component table, uncorrected ASIS (M7), "
     r"Primiceri's Algorithm~1 order) must reach $\max|z| > 8$",
     [(M_TOY, "test_exact_h_step_leaves_the_exact_posterior_invariant"),
      (M_TOY, "test_approximate_h_step_is_detected_with_a_coarse_table"),
      (M_TOY, "test_asis_move_with_w_correction_is_exact"), (M_TOY, "test_uncorrected_asis_move_m7_is_detected"),
      (M_TOY, "test_production_step_order_targets_the_posterior"), (M_TOY, "test_stale_indicator_order_algorithm_1_fails"),
      (M_TOY, "test_t13_stale_order_collapses_h_production_order_does_not")],
     [(M_TOY, "``|z| < 4.5``"), (M_TOY, "``max |z| > 8``")]),
    ("A7",
     r"reduced $\times$ Constant: exact system-wide joint draw of $\operatorname{vec}\Pi$ (NumPy test "
     r"code, $O((nk)^3)$) against the production CTA Gibbs, $n \in \{2, 3, 4\}$, also inside the "
     r"production sweep",
     r"MCSE $z$, Bonferroni (engine-level: $\alpha = 5\times10^{-4}$ per comparison)",
     r"the CCM-2019 $\Pi$ step and an unweighted $a$ step must fail",
     [(M_CCCM, "test_a7_swa_vs_cta_small"), (M_CCCM, "test_a7_swa_vs_cta"), (M_CCCM, "test_a7_detects_ccm2019"),
      (M_CCCM, "test_a7_production_small"), (M_CCCM, "test_a7_production"),
      (M_CCCM, "test_a7_production_detects_ccm2019"), (M_CCCM, "test_a7_production_detects_sweep_wiring")],
     [(M_CCCM, "Bonferroni α = 5e-4 per comparison"), (M_CCCM, "n = 3, p = 1, T = 60, 64 chains")]),
]  # fmt: skip


def junit_results(path: Path) -> dict[str, str]:
    """``{test function: 'passed' | 'failed' | 'skipped'}`` over all its parametrisations."""
    out: dict[str, set[str]] = {}
    for tcase in ET.parse(path).getroot().iter("testcase"):
        fn = tcase.get("name", "").split("[")[0]
        st = "failed" if tcase.find("failure") is not None or tcase.find("error") is not None else (
            "skipped" if tcase.find("skipped") is not None else "passed")
        out.setdefault(fn, set()).add(st)
    return {k: ("failed" if "failed" in v else "passed" if "passed" in v else "skipped") for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    ap.add_argument("--junit", type=Path, default=tc.RELEASE / "twins.xml")
    args = ap.parse_args()
    tc.apply_outdir(args)
    head = tc.git("rev-parse", "--short", "HEAD")
    src = {m: tc.git_show("HEAD", m) for m in (M_TWIN, M_QUAD, M_NIG, M_TOY, M_CCCM)}
    fns = {m: tc.functions(s) for m, s in src.items()}
    results = junit_results(args.junit) if args.junit.exists() else None

    lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.27\linewidth}"
             r">{\raggedright\arraybackslash}p{0.19\linewidth}>{\raggedright\arraybackslash}p{0.19\linewidth}"
             r">{\raggedright\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}p{0.06\linewidth}l@{}}",
             r"\toprule",
             r"& Oracle & Statistic, tolerance & Negative control & Cadence & Tests & Result \\",
             r"\midrule"]  # fmt: skip
    for tid, oracle, stat, neg, tests, evidence in TWINS:
        for m, text in evidence:
            assert text in src[m], f"{tid}: {text!r} not in {m}"
        pr, weekly = [], []
        for m, f in tests:
            assert f in fns[m], f"{tid}: {m}::{f} does not exist at HEAD"
            (weekly if "slow" in fns[m][f] else pr).append(f)
        cad = ", ".join(x for x, ok in (("PR", pr), ("weekly", weekly)) if ok)
        if results is None:
            res = tc.mark("REL")
        else:
            got = [results.get(f, "missing") for _, f in tests]
            res = "passed" if all(g == "passed" for g in got) else "; ".join(
                f"{f}: {g}" for (_, f), g in zip(tests, got, strict=True) if g != "passed")
        lines.append(f"{tid} & {oracle} & {stat} & {neg} & {cad} & {len(tests)} & {res} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    mods = sorted({m for t in TWINS for m, _ in t[4]})
    foot = ("Tests: the number of test functions (each may be parametrised) in "
            + ", ".join(tc.code(m) for m in mods)
            + f" at commit {head}; cadence from their \\code{{slow}} markers (PR: every pull request; "
            "weekly: the CI job named ``nightly''). CDF form: each exact quantile $x_q$ is tested "
            r"by $z = (\hat F_{\mathrm{MCMC}}(x_q) - q)/\sqrt{\mathrm{MCSE}^2 + q(1-q)/N_{\mathrm{ref}}}$ "
            "with the indicator's own ESS. A negative control passes when the comparison rejects "
            "the perturbed oracle.")  # fmt: skip
    body = tc.preamble(marks=results is None) + tc.fit("\n".join(lines)) + "\n" + tc.note(foot)
    if results is None:
        body += "\n" + tc.note(r"Placeholder: the Result column waits for the JUnit XML of the "
                               r"release run, \code{docs/paper/data/release/twins.xml} (OUTLINE TODO 7).",
                               red=True)  # fmt: skip
    tc.write(NAME, body, generator="tab_twins.py",
             status="static rows [V] (tests and quoted numbers checked at HEAD); Result column "
             + ("from " + str(args.junit) if results else "[REL] placeholder"),
             sources=[f"main repo {m} at {head}" for m in mods] + ["docs/paper/dossiers/04_validation.md section 9"])


if __name__ == "__main__":
    main()
