"""TabF5 (Appendix F): every SV arm against Constant, and the variant comparisons of criterion 5.

Every acceptance number is an ``\\acc{}`` placeholder (OUTLINE section 0.5 [ACC]); the script never
reads a value from the acceptance report. Keys (report of ``benchmarks/clark_ravazzolo.py`` at
c823d20; grammar of ``sections/11_replication.tex``):

* ``relative.<sample>['<arm>'].joint['<h>'].dlpd`` (joint dLPD, path-integrated RB);
* ``converged.<sample>['<arm>']`` (share of origins whose fit met the convergence rule);
* ``verdicts.<sample>[model='<arm>',horizon=<h>,convergence_gate=<true|false>].verdict`` and, at
  the gate-on entry, ``.model_in_mcs`` and ``.reference_in_mcs`` (``compare()`` with the 95%
  gate on and off; the MCS membership does not depend on the gate);
* ``variants['<variant>']['<ref>'].joint['<h>'].dlpd`` and ``.joint['1'].dm_p_better`` (criterion
  5; the report computes the variants on the full sample only, ``main()`` of the script).

The arms, samples, horizons and variant pairs are read from the committed script (``models``,
``HORIZONS``, the variant loop of ``main()``) with ``ast``/regular expressions, not typed here.
``--check-acc [REPORT]`` checks that every key resolves (no value is printed).

Usage: tab_cr_arms.py [--outdir DIR] [--check-acc [REPORT]]
"""

from __future__ import annotations

import argparse
import re

import tab_common as tc

NAME = "tab_cr_arms"
SCRIPT = ("c823d20", "benchmarks/clark_ravazzolo.py")
SAMPLES = (("full", "Full"), ("pre_crisis", "Pre-crisis"))


def script_facts() -> dict:
    src = tc.git_show(*SCRIPT)
    horizons = [str(h) for h in tc.literal(src, "HORIZONS")]
    baseline = tc.literal(src, "BASELINE")
    models_src = tc.grab(src, r"\ndef models\(.*?\n    return \{(.*?)\n    \}")
    arms = re.findall(r'^\s*"([^"]+)":', models_src, re.M)
    vline = tc.grab(src, r'^\s*for variant in \((.*)\):\s*\n\s*rep\["variants"\]\[variant\] = \{', flags=re.M)
    rline = tc.grab(src, r"relative_table\(results, variant, ref, all_origins\) for ref in \((.*)\)\s*$",
                    flags=re.M)  # fmt: skip
    variants = re.findall(r'"([^"]+)"', vline)
    refs = re.findall(r'"([^"]+)"', rline)
    assert baseline in arms, (baseline, arms)
    return {"horizons": horizons, "baseline": baseline,
            "arms": [a for a in arms if a != baseline], "variants": variants, "refs": refs}  # fmt: skip


def q(name: str) -> str:
    return f"['{name}']"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc.add_common_args(ap, acc_keys=True)
    args = ap.parse_args()
    tc.apply_outdir(args)
    f = script_facts()
    H, arms, base = f["horizons"], f["arms"], f["baseline"]
    acc = tc.acc

    # panel A: joint dLPD by horizon and converged shares
    a = [r"\begin{tabular}{@{}ll" + "r" * len(H) + "r@{}}", r"\toprule",
         r" & & \multicolumn{" + str(len(H)) + r"}{c}{Joint $\Delta$LPD against " + tc.tex(base)
         + r"} & \\", r"\cmidrule(lr){3-" + str(2 + len(H)) + "}",
         "Sample & Arm & " + " & ".join(f"$h = {h}$" for h in H) + r" & Converged \\", r"\midrule"]  # fmt: skip
    for s, label in SAMPLES:
        for i, arm in enumerate(arms):
            cells = [acc(f"relative.{s}{q(arm)}.joint['{h}'].dlpd") for h in H]
            a.append(f"{label if i == 0 else ''} & {tc.tex(arm)} & " + " & ".join(cells)
                     + f" & {acc(f'converged.{s}{q(arm)}')} \\\\")  # fmt: skip
        a.append(f" & {tc.tex(base)} & " + " & ".join(["---"] * len(H))
                 + f" & {acc(f'converged.{s}.{base}')} \\\\")  # fmt: skip
        a.append(r"\addlinespace")
    a[-1] = r"\bottomrule"
    a.append(r"\end{tabular}")

    # panel B: compare() verdicts with the gate on and off, and MCS membership
    b = [r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{Full sample} & \multicolumn{4}{c}{Pre-crisis} \\",
         r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
         r" & & \multicolumn{2}{c}{Verdict} & \multicolumn{2}{c}{In MCS} & "
         r"\multicolumn{2}{c}{Verdict} & \multicolumn{2}{c}{In MCS} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
         r"Arm & $h$ & gate on & off & model & ref. & gate on & off & model & ref. \\", r"\midrule"]  # fmt: skip
    for arm in arms:
        for i, h in enumerate(H):
            cells = []
            for s, _ in SAMPLES:
                sel = f"verdicts.{s}[model='{arm}',horizon={h},convergence_gate="
                cells += [acc(sel + "true].verdict"), acc(sel + "false].verdict"),
                          acc(sel + "true].model_in_mcs"), acc(sel + "true].reference_in_mcs")]  # fmt: skip
            b.append(f"{tc.tex(arm) if i == 0 else ''} & {h} & " + " & ".join(cells) + r" \\")
        b.append(r"\addlinespace")
    b[-1] = r"\bottomrule"
    b.append(r"\end{tabular}")

    # panel C: the variant comparisons of criterion 5 (full sample)
    c = [r"\begin{tabular}{@{}ll" + "r" * len(H) + "r@{}}", r"\toprule",
         r"\multicolumn{" + str(3 + len(H)) + r"}{@{}l}{\emph{Variants (criterion 5), full sample, "
         r"joint $\Delta$LPD}} \\",
         "Variant & against & " + " & ".join(f"$h = {h}$" for h in H) + r" & DM $p$, $h = 1$ \\",
         r"\midrule"]  # fmt: skip
    for v in f["variants"]:
        for ref in f["refs"]:
            p = f"variants{q(v)}{q(ref)}.joint"
            dm = acc(f"{p}['1'].dm_p_better")
            c.append(f"{tc.tex(v)} & {tc.tex(ref)} & " + " & ".join(acc(f"{p}['{h}'].dlpd") for h in H)
                     + f" & {dm} \\\\")  # fmt: skip
    c += [r"\bottomrule", r"\end{tabular}"]

    foot = ("Joint $\\Delta$LPD: path-integrated RB estimator, mean over origins, model minus "
            f"reference. Converged: share of origins at which the fit met the convergence rule "
            "(R-hat $< 1.01$, bulk ESS $> 400$ on every primary quantity). Verdict: "
            r"\code{compare()} on the volatility axis, with the 95\% convergence gate on and off. In "
            "MCS: whether the model and the reference are in the model confidence set. Under the "
            "expanding window the DM test is descriptive only, so a ``better'' verdict can come "
            "only through the model confidence set (model in, reference out); each verdict's route "
            r"is stated in the report's \code{reasons}. The variant comparisons exist for the full "
            "sample only.")  # fmt: skip
    body = ("\\footnotesize\n" + tc.preamble(acc_cells=True) + tc.fit("\n".join(a)) + "\n\n\\medskip\n"
            + tc.fit("\n".join(b)) + "\n\n\\medskip\n" + tc.fit("\n".join(c)) + "\n"
            + tc.note(foot) + "\n" + tc.note(tc.acc_placeholder_note("relative, converged, verdicts, variants"), red=True))  # fmt: skip
    path = tc.write(NAME, body, generator="tab_cr_arms.py",
                    status="[ACC] placeholders (no acceptance number written)",
                    sources=["acceptance report keys relative.{full,pre_crisis}[arm].joint[h].dlpd, "
                             "converged.{full,pre_crisis}[arm], verdicts.{full,pre_crisis}[...]."
                             "{verdict,model_in_mcs,reference_in_mcs}, variants[v][ref].joint",
                             f"main repo {SCRIPT[1]} at {SCRIPT[0]}: models(), BASELINE, HORIZONS, "
                             "the variant loop of main()"])  # fmt: skip
    tc.check_acc(path, args.check_acc)


if __name__ == "__main__":
    main()
