#!/usr/bin/env python3
r"""Fill the paper's \acc{...} placeholders from the Clark--Ravazzolo acceptance report.

The sources keep their placeholders: this script writes ``acceptance_values.tex``, one line
``\accdefine{<key>}{<value>}`` per distinct key used in ``sections/*.tex``, which ``main.tex`` reads
if it exists. ``\acc{<key>}`` then prints the value (green in a draft build, plain with
``\draftmarksfalse``) instead of the red placeholder. Delete the file to go back to placeholders.

    python3 fill_acceptance.py                       # default report of the main repository
    python3 fill_acceptance.py path/to/clark_ravazzolo_acceptance.json
    python3 fill_acceptance.py --list                # every key with the value it would print
    python3 fill_acceptance.py REPORT --allow-non-acceptance --partial --out /tmp/x/acceptance_values.tex
                                                     # dry run on another report (never into the paper)

Before anything is written, the report's header is checked against the pre-registration
(``figures/acceptance_check.py``, the same test the acceptance figures use): configuration
``acceptance``, path-integrated LPD, 8000 forecast draws, 4 x 20,000 kept draws with burn-in 1000,
seed 0, commit c823d20, a clean tree, no ``pending`` key, every stage complete. A report that fails
is refused unless ``--allow-non-acceptance`` is given, and then only with ``--out`` outside
``docs/paper``, so first-run numbers can never be typeset as acceptance values. A key that does not
resolve is an error unless ``--partial`` (then it keeps its placeholder).

Key grammar (header of ``sections/11_replication.tex``)::

    a.b.c             object keys
    ['k']             a key with other characters (model names, horizon keys '1','2','4','8')
    [f=v,g=w]         the list elements whose fields match (strings in single quotes, true/false,
                      numbers); one match gives that element, otherwise the list of matches
    [*]               every element of a list or every value of an object
    [N]               list index
    min(P), max(P)    over the values selected by P
    count(P = v)      how many values selected by P equal v
    absent(k)         whether the top-level key k is missing

Formatting is fixed by the last key of the path (see ``FORMATS`` below): log-score differences
with an explicit sign and three decimals, p-values with three decimals (``<0.001`` below that),
ratios and ranges with three decimals, converged shares as percentages, counts as integers, the
outcome of ``criteria[*].met`` as one phrase ("criteria 1, 2, 3 and 5 met; 4 and 6 not met"), and
strings escaped for LaTeX. Sentences built around a filled value still need the author's reading
(``\todo`` marks say where).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent  # docs/paper
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "figures"))
from acceptance_check import acceptance_problems  # noqa: E402
from texmarks import find_marks  # noqa: E402

MAIN_REPO = Path(os.environ.get("RAI_BVAR_REPO", "/Users/yigitokar/rai-bvar"))
DEFAULT_JSON = MAIN_REPO / "benchmarks" / "reports" / "clark_ravazzolo_acceptance.json"
DEFAULT_OUT = HERE / "acceptance_values.tex"


class Unresolved(Exception):
    pass


# ---------------------------------------------------------------------------------- parsing
def _split_top(s: str, sep: str) -> list[str]:
    """Split at ``sep`` outside brackets, parentheses and single quotes."""
    out, depth, quote, cur = [], 0, False, []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "'":
            quote = not quote
        elif not quote and c in "[(":
            depth += 1
        elif not quote and c in "])":
            depth -= 1
        if not quote and depth == 0 and s.startswith(sep, i):
            out.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def _literal(tok: str):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] == "'":
        return tok[1:-1]
    if tok == "true":
        return True
    if tok == "false":
        return False
    if tok == "null":
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        raise Unresolved(f"cannot read the literal {tok!r}") from None


def parse_path(s: str) -> list[tuple]:
    """``relative.full['SV(rw)'].joint['1'].dlpd`` -> [('key','relative'), ...]."""
    steps: list[tuple] = []
    i, s = 0, s.strip()
    while i < len(s):
        c = s[i]
        if c == ".":
            i += 1
            continue
        if c == "[":
            depth, quote, j = 0, False, i
            while j < len(s):
                if s[j] == "'":
                    quote = not quote
                elif not quote and s[j] == "[":
                    depth += 1
                elif not quote and s[j] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = s[i + 1 : j].strip()
            i = j + 1
            if inner == "*":
                steps.append(("all",))
            elif len(inner) >= 2 and inner[0] == inner[-1] == "'":
                steps.append(("key", inner[1:-1]))
            elif re.fullmatch(r"-?\d+", inner):
                steps.append(("index", int(inner)))
            else:
                conds = []
                for part in _split_top(inner, ","):
                    if "=" not in part:
                        raise Unresolved(f"bad selector [{inner}]")
                    f, v = part.split("=", 1)
                    conds.append((f.strip(), _literal(v)))
                steps.append(("filter", tuple(conds)))
            continue
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
        if not m:
            raise Unresolved(f"cannot parse {s[i:]!r}")
        steps.append(("key", m.group(0)))
        i += m.end()
    return steps


def parse_expr(expr: str):
    e = " ".join(expr.split())
    m = re.fullmatch(r"(min|max)\((.*)\)", e)
    if m:
        return (m.group(1), parse_path(m.group(2)))
    m = re.fullmatch(r"count\((.*)\)", e)
    if m:
        parts = _split_top(m.group(1), "=")
        if len(parts) != 2:
            raise Unresolved(f"count() needs 'P = v': {e!r}")
        return ("count", parse_path(parts[0]), _literal(parts[1]))
    m = re.fullmatch(r"absent\(([A-Za-z_][A-Za-z0-9_]*)\)", e)
    if m:
        return ("absent", m.group(1))
    return ("path", parse_path(e))


# ------------------------------------------------------------------------------- evaluation
def _match(el, conds) -> bool:
    if not isinstance(el, dict):
        return False
    for f, v in conds:
        if f not in el:
            return False
        x = el[f]
        if isinstance(v, bool) or isinstance(x, bool):
            if x is not v:
                return False
        elif x != v:
            return False
    return True


def walk(root, steps, where: str):
    """Returns (value, fanned_out)."""
    nodes, fan = [root], False
    for st in steps:
        nxt = []
        for node in nodes:
            if st[0] == "key":
                if not isinstance(node, dict) or st[1] not in node:
                    raise Unresolved(f"{where}: no key {st[1]!r}")
                nxt.append(node[st[1]])
            elif st[0] == "index":
                if not isinstance(node, list) or not -len(node) <= st[1] < len(node):
                    raise Unresolved(f"{where}: no index {st[1]}")
                nxt.append(node[st[1]])
            elif st[0] == "all":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
                else:
                    raise Unresolved(f"{where}: [*] on a scalar")
                fan = True
            elif st[0] == "filter":
                if not isinstance(node, list):
                    raise Unresolved(f"{where}: selector on a non-list")
                hits = [el for el in node if _match(el, st[1])]
                if len(hits) != 1:
                    fan = True
                nxt.extend(hits)
        nodes = nxt
    if fan:
        return nodes, True
    return nodes[0], False


def evaluate(rep: dict, expr: str):
    kind = parse_expr(expr)
    if kind[0] == "path":
        return walk(rep, kind[1], expr)
    if kind[0] in ("min", "max"):
        vals, _ = walk(rep, kind[1], expr)
        vals = [v for v in (vals if isinstance(vals, list) else [vals]) if isinstance(v, (int, float))]
        if not vals:
            raise Unresolved(f"{expr}: no numeric values")
        return (min if kind[0] == "min" else max)(vals), False
    if kind[0] == "count":
        vals, _ = walk(rep, kind[1], expr)
        vals = vals if isinstance(vals, list) else [vals]
        v = kind[2]
        return sum(1 for x in vals if (x is v if isinstance(v, bool) else x == v)), False
    if kind[0] == "absent":
        return kind[1] not in rep, False
    raise Unresolved(expr)


# ------------------------------------------------------------------------------- formatting
_UNICODE = {
    "Δ": r"$\Delta$", "δ": r"$\delta$", "σ": r"$\sigma$", "φ": r"$\phi$", "ν": r"$\nu$",
    "λ": r"$\lambda$", "μ": r"$\mu$", "τ": r"$\tau$", "θ": r"$\theta$", "κ": r"$\kappa$",
    "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\to$", "←": r"$\leftarrow$", "×": r"$\times$",
    "±": r"$\pm$", "−": r"$-$", "≈": r"$\approx$", "≠": r"$\neq$", "…": r"\ldots{}",
    "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "·": r"$\cdot$",
    "√": r"$\surd$", "∞": r"$\infty$", "§": r"\S{}",
}
_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_text(s: str) -> str:
    out = []
    for ch in str(s):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def tex_code(s: str) -> str:
    return r"\code{" + str(s).replace("{", "").replace("}", "").replace("%", "") + "}"


def _num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def f_signed(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return tex_text(x)
    s = _num(x)
    if s.lstrip("-") == "0.000":
        return r"\ensuremath{0.000}"
    return r"\ensuremath{" + ("+" if x > 0 else "") + s + "}"


def f_p(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return tex_text(x)
    return r"\ensuremath{<0.001}" if x < 0.001 else r"\ensuremath{" + _num(x) + "}"


def f_dec(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return tex_text(x)
    return r"\ensuremath{" + _num(x) + "}"


def f_pct(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return tex_text(x)
    return f"{100 * x:.0f}\\%"


def f_int(x) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return tex_text(x)
    x = int(round(x))
    return f"{x:,}".replace(",", "{,}") if abs(x) >= 10000 else str(x)


def f_sci(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or x == 0:
        return tex_text(x)
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    if -3 <= e <= 3:
        return r"\ensuremath{" + f"{x:.2g}" + "}"
    return r"\ensuremath{" + f"{m:.2g}" + r"\times10^{" + str(e) + "}}"


def f_bool(x) -> str:
    return r"\code{true}" if x is True else r"\code{false}" if x is False else tex_text(x)


def f_yesno(x) -> str:
    return "yes" if x is True else "no" if x is False else tex_text(x)


def f_met(x) -> str:
    return "met" if x is True else "not met" if x is False else tex_text(x)


def f_verdict(x) -> str:
    return "``" + tex_text(x) + "''"


def f_commit(x) -> str:
    return tex_code(str(x)[:7]) if x else tex_text(x)


def f_seconds(x) -> str:
    if isinstance(x, dict):
        return "; ".join(f"{tex_text(k)} {f_seconds(v)}" for k, v in x.items())
    if not isinstance(x, (int, float)):
        return tex_text(x)
    h, rem = divmod(int(round(x)), 3600)
    return f"{f_int(round(x))}\\,s ({h}\\,h\\,{rem // 60:02d}\\,min)" if h else f"{f_int(round(x))}\\,s"


def f_mcmc(x) -> str:
    if not isinstance(x, dict):
        return tex_text(x)
    return (f"{x.get('chains')} chains $\\times$ {f_int(x.get('draws'))} kept draws, burn-in "
            f"{f_int(x.get('burn'))}")


def f_versions(x) -> str:
    if not isinstance(x, dict):
        return tex_text(x)
    parts = []
    for k, v in x.items():
        if v is None:  # e.g. git_diff_sha256 of a clean tree: nothing to report
            continue
        if k == "git_commit":
            parts.append("commit " + f_commit(v))
        elif k == "git_dirty":
            parts.append("tracked files " + ("modified" if v else "unchanged"))
        else:
            parts.append(f"{tex_text(k)} {tex_text(v)}")
    return "; ".join(parts)


def f_aggregation(x) -> str:
    if not isinstance(x, dict):
        return tex_text(x)
    return "; ".join(
        f"\\code{{{k}}}: maximum absolute difference {f_sci(v.get('max_abs_diff'))} over "
        f"{v.get('quarters')} quarters" for k, v in x.items())


def f_in_window(x) -> str:
    if not isinstance(x, dict):
        return tex_text(x)
    o = x.get("origins", [])
    where = ", ".join(o[:-1]) + (" and " if len(o) > 1 else "") + (o[-1] if o else "")
    return ("bitwise identical" if x.get("identical") else "not identical") + f" at {where}"


def f_summary_rows(x) -> str:
    rows = x if isinstance(x, list) else [x]
    if not rows:
        return "none"
    return "; ".join(
        f"{tex_text(r.get('variable'))} $h = {r.get('horizon')}$ (range {f_dec(r.get('lpd_range'))}, "
        f"SNR {f_dec(r.get('lpd_range_snr'))})" for r in rows)


def f_warnings(x) -> str:
    if not isinstance(x, dict):
        return tex_text(x)
    parts = []
    for arm, kinds in x.items():
        n = sum(int(v.get("origin_messages", 0)) for v in kinds.values()) if isinstance(kinds, dict) else 0
        k = len(kinds) if isinstance(kinds, dict) else 0
        parts.append(f"{tex_text(arm)}: {k} kind{'s' if k != 1 else ''} in {n} origin messages")
    return "; ".join(parts)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def f_criteria_met(rep: dict) -> str:
    crit = rep.get("criteria")
    if not isinstance(crit, list) or not crit:
        raise Unresolved("criteria[*].met: no criteria list")
    met = [str(c["id"]) for c in crit if c.get("met") is True]
    miss = [str(c["id"]) for c in crit if c.get("met") is not True]
    if not miss:
        return f"all {len(crit)} criteria met"
    if not met:
        return f"none of the {len(crit)} criteria met"
    return f"criteria {_and(met)} met; {_and(miss)} not met"


def f_details(x) -> str:
    rows = x if isinstance(x, list) else [x]
    return "; ".join(tex_text(r) for r in rows)


# last key of the path -> formatter
FORMATS = {
    **{k: f_signed for k in ("dlps", "dlpd", "dlps_conditioned", "dlpd_conditioned", "dlps_gauss",
                             "dlpd_gauss", "dlps_median", "dlpd_h1", "dlps_h1",
                             # dlps_top[i][1] (a per-origin difference) signed; dlps_top[i][0]
                             # (its origin, a string) passes through tex_text
                             "dlps_top")},
    **{k: f_p for k in ("dm_p_better", "dm_p", "dm_p_joint_h1", "dm_pvalue_better", "p")},
    **{k: f_dec for k in ("crps_ratio", "rmse_ratio", "lpd_range", "lpd_range_snr", "dlpd_mcse",
                          "lps_base", "rmse_base", "lps_model")},
    **{k: f_pct for k in ("converged_sv", "converged_constant")},
    **{k: f_int for k in ("rb_degenerate", "n", "draws", "seed", "quarters", "ordering")},
    "rb_gauss_disagree": f_yesno,
    "material": f_bool,
    "identical": f_bool,
    "git_dirty": f_bool,
    "met": f_met,
    "verdict": f_verdict,
    "verdict_h1": f_verdict,
    "verdict_h1_ungated": f_verdict,
    "git_commit": f_commit,
    "name": tex_code,
    "lpd_method": tex_code,
    "max_abs_diff": f_sci,
    "seconds": f_seconds,
    "mcmc": f_mcmc,
    "versions": f_versions,
    "aggregation": f_aggregation,
    "in_window": f_in_window,
    "summary": f_summary_rows,
    "warnings": f_warnings,
    "detail": f_details,
}


def last_key(steps) -> str | None:
    keys = [s[1] for s in steps if s[0] == "key"]
    for k in reversed(keys):
        if not re.fullmatch(r"\d+", k):
            return k
    return None


def generic(v) -> str:
    if isinstance(v, bool):
        return f_bool(v)
    if isinstance(v, int):
        return f_int(v)
    if isinstance(v, float):
        return f_dec(v) if math.isfinite(v) else tex_text(v)
    if isinstance(v, list):
        return ", ".join(generic(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return "; ".join(f"{tex_text(k)} = {generic(x)}" for k, x in v.items())
    if v is None:
        return r"\code{null}"
    return tex_text(v)


def render(rep: dict, expr: str) -> str:
    e = " ".join(expr.split())
    if e == "criteria[*].met":
        return f_criteria_met(rep)
    kind = parse_expr(e)
    value, fanned = evaluate(rep, e)
    if kind[0] == "count":
        return f_int(value)
    if kind[0] == "absent":
        return "absent" if value else "present"
    steps = kind[1]
    key = last_key(steps)
    if key in ("converged",) or (steps and steps[0] == ("key", "converged")):
        fmt = f_pct
    elif e.startswith("verdicts.") and key == "verdict":
        fmt = f_verdict
    else:
        fmt = FORMATS.get(key)
    if key == "summary" or (len(steps) and steps[-1][0] == "filter" and key == "summary"):
        return f_summary_rows(value)
    if fmt is None:
        return generic(value)
    if fanned and fmt not in (f_details, f_summary_rows):
        vals = value if isinstance(value, list) else [value]
        return ", ".join(fmt(v) for v in vals) if vals else "none"
    return fmt(value)


# ------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", nargs="?", type=Path, default=DEFAULT_JSON,
                    help=f"the acceptance report JSON (default {DEFAULT_JSON})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"macro file to write (default {DEFAULT_OUT.name} next to main.tex)")
    ap.add_argument("--allow-non-acceptance", action="store_true",
                    help="dry run on a report that is not the registered acceptance run; needs "
                    "--out outside docs/paper")
    ap.add_argument("--partial", action="store_true",
                    help="write even if some keys do not resolve (they keep their placeholder)")
    ap.add_argument("--list", action="store_true", help="print every key and value; write nothing")
    args = ap.parse_args()

    if not args.report.exists():
        print(f"no report at {args.report} (the acceptance JSON is committed under OUTLINE TODO 2)",
              file=sys.stderr)
        return 2
    raw = args.report.read_bytes()
    rep = json.loads(raw)
    probs = acceptance_problems(rep)
    if probs and not args.allow_non_acceptance:
        print(f"{args.report} is not the registered acceptance report:\n  " + "\n  ".join(probs),
              file=sys.stderr)
        return 3
    out = args.out.resolve()
    if probs and not args.list and HERE in out.parents:
        print("--allow-non-acceptance writes only outside docs/paper (a dry run must never fill the "
              "paper with another run's numbers)", file=sys.stderr)
        return 3

    marks = find_marks("acc")
    keys: dict[str, list[str]] = {}
    for m in marks:
        keys.setdefault(m.flat, []).append(f"{m.rel}:{m.line}")

    values, missing = {}, {}
    for k in keys:
        try:
            values[k] = render(rep, k)
        except (Unresolved, KeyError, TypeError, ValueError) as exc:
            missing[k] = str(exc)

    if args.list:
        for k in keys:
            print(f"{k}\n    -> {values.get(k, '[UNRESOLVED: ' + missing.get(k, '') + ']')}")
        print(f"\n{len(values)} of {len(keys)} keys resolve" + (f"; header problems: {probs}" if probs else ""))
        return 0 if not missing else 1

    if missing:
        print(f"{len(missing)} of {len(keys)} keys do not resolve:", file=sys.stderr)
        for k, why in missing.items():
            print(f"  {k}  ({', '.join(keys[k])}): {why}", file=sys.stderr)
        if not args.partial:
            print("nothing written (use --partial to write the rest)", file=sys.stderr)
            return 1

    sha = hashlib.sha256(raw).hexdigest()
    ver = rep.get("versions", {})
    lines = [
        "% Generated by docs/paper/fill_acceptance.py -- do not edit; delete to restore the placeholders.",
        f"% report:    {args.report}",
        f"% sha256:    {sha}",
        f"% generated: {rep.get('generated')} (the report's field: set when the report is first assembled, "
        "after the five reduced-form main-stage arms and before the structural and ordering stages; "
        "not the end of the run, which is the last line of the committed run log, 2026-10-23T19:46:21Z "
        "for the acceptance run)",
        f"% written:   {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} (this file)",
        f"% commit:    {ver.get('git_commit')}  dirty: {ver.get('git_dirty')}",
        f"% header:    {'acceptance run' if not probs else 'NOT the acceptance run: ' + '; '.join(probs)}",
        f"% keys:      {len(values)} of {len(keys)} resolved" + (f"; unresolved: {', '.join(missing)}" if missing else ""),
        r"\accsource{" + tex_text(args.report.name) + " (sha256 " + sha[:12] + ")}",
    ]
    for k, v in values.items():
        if "\n" in v or "%" in v.replace(r"\%", ""):
            raise SystemExit(f"value of {k} would break the macro file: {v!r}")
        lines.append(r"\accdefine{" + k + "}{" + v + "}% " + ", ".join(keys[k]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(values)} of {len(keys)} keys")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
