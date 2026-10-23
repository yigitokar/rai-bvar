"""TabD2 (Appendix D): parity summary, in neutral wording.

Upper panel, the analytic engine against R BVAR 1.0.5 and the Bank of England's bvar 0.3.5:
quantity, tolerance asserted by the tests, and the deviation the parity notes record. Lower
panel, the MCMC engine: the stochvol 3.2.1 rows (main and default priors on four datasets, and the
two exact-mode rows), the exact importance-sampling oracle of the Student-t SV sampler, and the
in-house system-wide Gibbs oracle that replaces bayesianVARs: number of compared quantiles m, the
bound max(4, z_{1-0.01/(2m)}), and the largest |z| at the per-PR and at the weekly size.

Sources ([R]; read from the committed notes of the main repository at HEAD, never typed here):
``tests/parity/BVAR_PARAMETRIZATION.md`` section 12, ``tests/parity/BOE_BVAR_NOTES.md`` section 14
and the results table of ``tests/parity/SV_PARAMETRIZATION.md`` section 6. Every number of the
upper panel is taken from the notes' sentences by a regular expression that must match exactly
once. Neutral wording (OUTLINE TODO 13): the table states what our tests compare and what they
measured, nothing about another package's sampler; the note's Student-t stochvol fixture row and
the fold-fraction column (the sign-fold observation) are left out until TODO 13 is resolved. The
release record (``pytest -m parity_long`` on the release commit, OUTLINE TODO 7) has no record hook
yet; until it exists the notes are the source, with their machine label (TODO 8).

Usage: tab_parity.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import re

import tab_common as tc

NAME = "tab_parity"
BVAR = "tests/parity/BVAR_PARAMETRIZATION.md"
BOE = "tests/parity/BOE_BVAR_NOTES.md"
SV = "tests/parity/SV_PARAMETRIZATION.md"
EXCLUDE = ("stochvol_v_t5",)  # OUTLINE TODO 13


def sci(s: str) -> str:
    """'1.5e-11' -> $1.5\\times10^{-11}$."""
    m = re.fullmatch(r"([\d.]+)e([+-]?\d+)", s)
    if not m:
        return s
    a, e = m.groups()
    return f"${'' if a == '1' else a + r'\times'}10^{{{int(e)}}}$"


def analytic() -> tuple[list[str], dict]:
    bv = tc.git_show("HEAD", BVAR)
    be = tc.git_show("HEAD", BOE)
    v_bvar = tc.grab(bv, r"`BVAR` (\d+\.\d+\.\d+) \(Kuschnig")
    v_boe = tc.grab(be, r"\*\*`bvar` (\d+\.\d+\.\d+)\*\*")
    s12 = re.sub(r"\s+", " ", tc.section(bv, r"^## 12\. "))
    s14 = re.sub(r"\s+", " ", tc.section(be, r"^## 14\. "))
    b_ml = tc.grab(s12, r"log ML at fixed `h` (1e-\d+) relative \(BVAR and the exact values agree to < (1e-\d+)\)", 0)
    b_ml_tol, b_ml_got = re.findall(r"1e-\d+", b_ml)
    b_niw = tc.grab(s12, r"NIW parameters (1e-\d+) relative to the largest entry of each matrix \(BVAR's own "
                         r"normal-equation numbers deviate by up to ([\d.]+e-\d+) on that scale\)", 0)  # fmt: skip
    b_niw_tol, b_niw_got = re.findall(r"[\d.]*\de-\d+", b_niw)
    b_mode = tc.grab(s12, r"hyperparameter modes (1e-\d+) relative \(targets accurate to ~(1e-\d+)\)", 0)
    b_mode_tol = re.findall(r"1e-\d+", b_mode)[0]
    b_mc = tc.grab(s12, r"Monte Carlo quantiles within `(\d) · sqrt\(MCSE_R² \+ MCSE_ours²\)`")
    e_ml = tc.grab(s14, r"Log ML at fixed h with explicit ψ: rtol (1e-\d+) \(measured ≤ ([\d.]+e-\d+)\)", 0)
    e_ml_tol, e_ml_got = re.findall(r"[\d.]*\de-\d+", e_ml)
    e_b = tc.grab(s14, r"B̄ on level data \(measured ([\d.]+e-\d+) against a (1e-\d+) column-scaled tolerance", 0)
    e_b_got, e_b_tol = re.findall(r"[\d.]*\de-\d+", e_b)
    e_draws = tc.grab(s14, r"the means of (\d+) of our iid draws against BoE's B̄ and E\[Σ\] within (\d) MCSE", 0)
    n_draws, k_mcse = re.findall(r"\d+", e_draws)
    e_mode = tc.grab(s14, r"within (1e-\d+) relative of the mode of BoE's objective")
    assert "Forecast quantiles within Monte Carlo tolerance" in s14
    rows = [
        (f"R \\pkg{{BVAR}} {v_bvar}", "log marginal likelihood at fixed hyperparameters",
         f"rel.\\ {sci(b_ml_tol)}", f"$< {sci(b_ml_got)[1:-1]}$"),
        ("", "NIW posterior parameters (relative to the largest entry)", f"rel.\\ {sci(b_niw_tol)}",
         f"up to {sci(b_niw_got)}"),
        ("", "empirical-Bayes modes of the hyperparameters", f"rel.\\ {sci(b_mode_tol)}", "within tolerance"),
        ("", "Metropolis--Hastings and forecast quantiles", f"{b_mc} combined MCSE", "within tolerance"),
        (f"\\pkg{{bvar}} {v_boe} (Bank of England)", "log marginal likelihood at fixed hyperparameters",
         f"rel.\\ {sci(e_ml_tol)}", f"$\\leq {sci(e_ml_got)[1:-1]}$"),
        ("", r"posterior mean $\bar B$ on level data (column-scaled)", f"{sci(e_b_tol)}", f"{sci(e_b_got)}"),
        ("", f"means of {n_draws} iid posterior draws", f"{k_mcse} MCSE", "within tolerance"),
        ("", "hyperparameter modes", f"rel.\\ {sci(e_mode)}", "within tolerance"),
        ("", "forecast quantiles", "Monte Carlo tolerance", "within tolerance"),
    ]  # fmt: skip
    lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.40\linewidth}ll@{}}", r"\toprule",
             r"Oracle & Quantity & Tolerance & Recorded deviation \\", r"\midrule"]  # fmt: skip
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines, {"bvar": v_bvar, "boe": v_boe}


def fixture_label(cell: str) -> tuple[str, str]:
    """(oracle group, fixture text) of a row of the SV note's results table."""
    name = re.split(r"\s+\(", cell, maxsplit=1)[0]
    rest = cell[len(name):].strip()
    parts = re.split(r"`([^`]+)`", rest)
    rest = "".join(tc.code(p.replace(" ", "")) if i % 2 else tc.tex(p) for i, p in enumerate(parts))
    if name.startswith("stochvol_"):
        group = "stochvol"
    elif name.startswith("bvars_swa_"):
        group = "swa"
    elif name.startswith("SV-t vs exact importance sampling"):
        return "is", "four observations, " + tc.code("T_pad=320")
    else:
        raise ValueError(f"unknown parity fixture {cell!r}")
    return group, tc.code(name) + (" " + rest if rest else "")


def mcmc() -> tuple[list[str], dict]:
    sv = tc.git_show("HEAD", SV)
    v_sv = tc.grab(sv, r"stochvol (\d+\.\d+\.\d+) \(DESIGN §9\.9 names")
    s6 = tc.section(sv, r"^## 6\. Results")
    machine = tc.grab(s6, r"Measured on the reference laptop \((.*?), jax")
    table = tc.md_table(s6)
    groups = {"stochvol": [], "is": [], "swa": []}
    excluded = []
    for r in table:
        fx = r["fixture"]
        if fx.split()[0] in EXCLUDE:
            excluded.append(fx.split()[0])
            continue
        g, label = fixture_label(fx)
        pr = r["PR: max \\|z\\| (4 × 6,000)"] if "PR: max \\|z\\| (4 × 6,000)" in r else r["PR: max |z| (4 × 6,000)"]
        nightly = r.get("nightly: max \\|z\\|", r.get("nightly: max |z|"))
        for bad in ("**",):
            assert bad not in pr and bad not in nightly, (fx, pr, nightly)
        pr = pr.replace("nightly only", "---").replace("×", r"$\times$").replace(",", "{,}")
        nightly = nightly.replace("—", "---")
        groups[g].append((label, r["m"].replace(",", "{,}"), r["bound"], pr, nightly))
    assert excluded == list(EXCLUDE), excluded
    titles = {"stochvol": f"\\pkg{{stochvol}} {v_sv}, \\code{{ZeroMean}} $\\times$ SV, $n = 1$",
              "is": "exact posterior by self-normalised importance sampling (Student-$t$ SV)",
              "swa": r"exact system-wide Gibbs sampler in R (reduced form; replaces \pkg{bayesianVARs})"}  # fmt: skip
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{max $|z|$} \\ \cmidrule(l){4-5}",
             r"Oracle and fixture & $m$ & Bound & per-PR size & weekly size \\", r"\midrule"]  # fmt: skip
    for g in ("stochvol", "is", "swa"):
        lines.append(r"\multicolumn{5}{@{}l}{\emph{" + titles[g] + r"}} \\")
        for label, m, bound, pr, nightly in groups[g]:
            lines.append(f"\\quad {label} & {m} & {bound} & {pr} & {nightly} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines, {"stochvol": v_sv, "machine": machine, "n_stochvol": len(groups["stochvol"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap)
    args = ap.parse_args()
    tc.apply_outdir(args)
    head = tc.git("rev-parse", "--short", "HEAD")
    up, va = analytic()
    lo, vm = mcmc()
    foot = (r"Upper panel: \code{tests/parity/test_r_bvar.py} and \code{test_boe_bvar.py}; tolerances "
            r"and recorded deviations from \code{tests/parity/BVAR_PARAMETRIZATION.md} \S12 and "
            r"\code{BOE_BVAR_NOTES.md} \S14. Lower panel: \code{tests/parity/test_stochvol.py} and "
            r"\code{test_bayesianvars.py}, results table of \code{SV_PARAMETRIZATION.md} \S6; "
            r"$z = (\hat q_{\mathrm{ours}} - \hat q_{\mathrm{ref}})/\sqrt{\mathrm{mcse}^2_{\mathrm{ours}} "
            r"+ \mathrm{mcse}^2_{\mathrm{ref}}}$ over the 5, 25, 50, 75 and 95\% quantiles, bound "
            r"$\max(4, z_{1-0.01/(2m)})$; per-PR size $4\times6000$ kept draws ($4\times3000$ where "
            r"stated), weekly size $4\times40{,}000$ (\pkg{stochvol}) or $4\times20{,}000$ (VAR); "
            r"--- not run at that size. Recorded in the notes (machine labelled "
            f"``{tc.tex(vm['machine'])}'' there, under review, OUTLINE TODO 8), main repository "
            f"at {head}; to be re-recorded on the release commit (\\code{{-m parity_long}}, TODO 7). "
            "The notes' Student-$t$ \\pkg{stochvol} fixture and fold fractions are not shown (TODO 13).")  # fmt: skip
    body = (tc.fit("\n".join(up)) + "\n\n\\medskip\n" + tc.fit("\n".join(lo)) + "\n" + tc.note(foot))
    tc.write(NAME, body, generator="tab_parity.py",
             status="[R] parity notes (neutral wording; stochvol_v_t5 and fold column omitted, TODO 13)",
             sources=[f"main repo {BVAR} section 12 at {head}", f"main repo {BOE} section 14 at {head}",
                      f"main repo {SV} section 6 at {head}"])  # fmt: skip


if __name__ == "__main__":
    main()
