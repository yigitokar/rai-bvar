#!/usr/bin/env python3
r"""Mechanical check of the paper's acceptance-run numbers against the committed report JSON.

Layer A: every \accdefine{key}{value} in acceptance_values.tex is re-resolved against the JSON by an
          evaluator written here (independent of fill_acceptance.py), and the printed number(s) are
          compared with the raw value at the printed precision (sign, decimals, <0.001, %, counts).
Layer B: every number typed in the rendered text (comments stripped, \acc marks removed) of the
          in-scope regions is matched against an index of known sources: the acceptance JSON (with
          simple derived values), the committed run log and report .md, the pre-registration, the
          working-paper numbers CR_FULL / CR_PRE_DLPS, the first full run's JSON (for \fullrun{}),
          and fixed thresholds. Tokens matching nothing are printed for manual review.
Layer C: explicit assertions for typed claims about the run (recomputations, counts, orderings).

    python3 check_acceptance_numbers.py [--all]   (--all also lists matched tokens; exit 1 on a mismatch)

Layers C and E hold the typed claims of the text (72%, +0.079, the tab:cr-c4 cells, ...) as
assertions: when a sentence with a typed number changes, change its assertion here too.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent  # docs/paper
REPO = PAPER.parent.parent  # the checkout that holds benchmarks/reports/
sys.path.insert(0, str(PAPER))
from texmarks import strip_comments  # noqa: E402  (comment stripping only)

ACCJ = REPO / "benchmarks/reports/clark_ravazzolo_acceptance.json"
FIRSTJ = REPO / "benchmarks/reports/clark_ravazzolo.json"
LOG = REPO / "benchmarks/reports/clark_ravazzolo_acceptance.log"
MD = REPO / "benchmarks/reports/clark_ravazzolo_acceptance.md"
PREREG = REPO / "benchmarks/reports/clark_ravazzolo_preregistration.md"
VALUES = PAPER / "acceptance_values.tex"

A = json.loads(ACCJ.read_text())
F = json.loads(FIRSTJ.read_text())
VERBOSE = "--all" in sys.argv

problems: list[str] = []


def bad(msg: str) -> None:
    problems.append(msg)
    print("  MISMATCH:", msg)


# =========================================================================== independent evaluator
TOK = re.compile(r"""\s*(?:
      \.?(?P<key>[A-Za-z_][A-Za-z0-9_]*)
    | \[\s*'(?P<qkey>[^']*)'\s*\]
    | \[\s*(?P<idx>-?\d+)\s*\]
    | \[\s*(?P<star>\*)\s*\]
    | \[(?P<filt>[^\]]*=[^\]]*)\]
)""", re.X)


def lit(v: str):
    v = v.strip()
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(v)
    except ValueError:
        return float(v)


def walk(root, path: str):
    nodes, fan, i = [root], False, 0
    while i < len(path):
        m = TOK.match(path, i)
        if not m or m.end() == i:
            raise ValueError(f"cannot parse {path[i:]!r} in {path!r}")
        i = m.end()
        nxt = []
        for n in nodes:
            if m.group("key") is not None:
                nxt.append(n[m.group("key")])
            elif m.group("qkey") is not None:
                nxt.append(n[m.group("qkey")])
            elif m.group("idx") is not None:
                nxt.append(n[int(m.group("idx"))])
            elif m.group("star"):
                nxt.extend(n if isinstance(n, list) else list(n.values()))
                fan = True
            else:
                conds = [c.split("=", 1) for c in re.findall(r"(?:'[^']*'|[^,])+", m.group("filt"))]
                hits = [e for e in n if all(
                    k.strip() in e and (e[k.strip()] is lit(v) if isinstance(lit(v), bool) else e[k.strip()] == lit(v))
                    for k, v in conds)]
                if len(hits) != 1:
                    fan = True
                nxt.extend(hits)
        nodes = nxt
    return (nodes, True) if fan else (nodes[0], False)


def resolve(expr: str):
    e = " ".join(expr.split())
    m = re.fullmatch(r"(min|max)\((.*)\)", e)
    if m:
        vals, _ = walk(A, m.group(2))
        vals = vals if isinstance(vals, list) else [vals]
        vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return (min if m.group(1) == "min" else max)(vals), "minmax"
    m = re.fullmatch(r"count\((.*)=(.*)\)", e)
    if m:
        vals, _ = walk(A, m.group(1).strip())
        vals = vals if isinstance(vals, list) else [vals]
        v = lit(m.group(2))
        return sum(1 for x in vals if (x is v if isinstance(v, bool) else x == v)), "count"
    m = re.fullmatch(r"absent\((\w+)\)", e)
    if m:
        return (m.group(1) not in A), "absent"
    val, fan = walk(A, e)
    return val, ("fan" if fan else "path")


# =========================================================================== layer A
def brace_arg(s: str, i: int) -> tuple[str, int]:
    assert s[i] == "{", (s[i:i + 20])
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


NUMRX = re.compile(r"([<+\-]?)(\d[\d,]*(?:\.\d+)?)")


def printed_numbers(v: str):
    t = v.replace(r"\ensuremath", "").replace("{,}", ",").replace("$-$", "-").replace(r"\,", " ")
    t = t.replace("{", "").replace("}", "")
    return [(m.group(1), m.group(2).replace(",", "")) for m in NUMRX.finditer(t)]


def num_ok(raw: float, sign: str, txt: str, pct: bool) -> bool:
    d = len(txt.split(".")[1]) if "." in txt else 0
    val = float(txt)
    x = raw * 100 if pct else raw
    if sign == "<":
        return x < val
    if sign == "-":
        val = -val
    elif sign == "+":
        if not x > 0:
            return False
    elif x < 0 and d > 0:
        return False
    if sign == "" and d > 0 and x > 0 and not pct:
        # an unsigned positive decimal: fine for ratios and p-values; a signed key would be '+'
        pass
    return abs(x - val) <= 0.5 * 10 ** -d + 1e-9


SIGNED = ("dlps", "dlpd", "dlps_conditioned", "dlpd_conditioned", "dlps_gauss", "dlpd_gauss",
          "dlps_median", "dlpd_h1", "dlps_h1")


def check_value(key: str, printed: str) -> None:
    if " ".join(key.split()) == "criteria[*].met":
        met = [c["id"] for c in A["criteria"] if c["met"] is True]
        miss = [c["id"] for c in A["criteria"] if c["met"] is not True]
        exp = "criteria " + ", ".join(map(str, met[:-1])) + f" and {met[-1]} met; " + \
              " and ".join(map(str, miss)) + " not met"
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    raw, kind = resolve(key)
    if kind == "absent":
        exp = "absent" if raw else "present"
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    if isinstance(raw, dict) or (kind == "fan" and any(isinstance(v, (dict, str)) for v in raw) and key.endswith(".detail")):
        COMPOSITE.append((key, raw, printed))
        return
    last = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", key)[-1] if key else ""
    if kind == "fan":
        vals = raw
    else:
        vals = [raw]
    # strings / bools
    if all(isinstance(v, bool) for v in vals):
        maps = {"rb_gauss_disagree": ("yes", "no"), "met": ("met", "not met"),
                "material": (r"\code{true}", r"\code{false}"), "identical": (r"\code{true}", r"\code{false}"),
                "git_dirty": (r"\code{true}", r"\code{false}")}
        t, f = maps.get(last, (r"\code{true}", r"\code{false}"))
        exp = ", ".join(t if v else f for v in vals)
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    if all(isinstance(v, str) for v in vals):
        if last == "git_commit":
            exp = ", ".join(r"\code{" + v[:7] + "}" for v in vals)
        elif last in ("verdict", "verdict_h1", "verdict_h1_ungated"):
            exp = ", ".join("``" + v + "''" for v in vals)
        elif last in ("name", "lpd_method"):
            exp = ", ".join(r"\code{" + v + "}" for v in vals)
        else:
            exp = ", ".join(vals)
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    if last == "seconds" and kind == "path":
        s = int(round(raw))
        h, rem = divmod(s, 3600)
        exp = f"{s:,}".replace(",", "{,}") + rf"\,s ({h}\,h\,{rem // 60:02d}\,min)"
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    if isinstance(raw, list) and raw and all(isinstance(v, str) for v in raw):
        exp = ", ".join(raw)
        if printed != exp:
            bad(f"{key}: printed {printed!r}, expected {exp!r}")
        return
    nums = printed_numbers(printed)
    flat = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) != len(flat):
        bad(f"{key}: printed {printed!r} has {len(nums)} numbers, the JSON gives {flat!r}")
        return
    pct = r"\%" in printed
    is_conv = key.lstrip("minax(").startswith("converged") or last in ("converged_sv", "converged_constant")
    if is_conv and not pct:
        bad(f"{key}: a share printed without %: {printed!r}")
    for (sg, txt), v in zip(nums, flat):
        if not num_ok(float(v), sg, txt, pct):
            bad(f"{key}: printed {printed!r}, JSON {v!r}")
        if last in SIGNED and kind != "count" and sg == "" and txt.strip("0.") != "":
            bad(f"{key}: signed quantity printed without sign: {printed!r}")


def layer_a() -> int:
    print("== Layer A: acceptance_values.tex against the JSON (independent evaluator)")
    src = VALUES.read_text()
    n = 0
    for m in re.finditer(r"\\accdefine\{", src):
        key, j = brace_arg(src, m.end() - 1)
        val, _ = brace_arg(src, j)
        n += 1
        try:
            check_value(key, val)
        except Exception as exc:  # noqa: BLE001
            bad(f"{key}: could not evaluate ({type(exc).__name__}: {exc})")
    print(f"  {n} values checked")
    return n


# =========================================================================== source index
def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


INDEX: list[tuple[float, str, str]] = []  # (value, source class, label)


def add(v, cls, label):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return
    INDEX.append((float(v), cls, label))


def build_index():
    for p, v in leaves(A):
        add(v, "ACC", p)
        if isinstance(v, float) and 0 <= abs(v) <= 1:
            add(100 * v, "ACC", p + "*100")
    # derived values of the acceptance JSON
    add(A["seconds"] / 3600, "ACC", "seconds/3600")
    add(A["seconds"] / 60, "ACC", "seconds/60")
    for p, o in [(p, o) for p, o in _dicts(A) if isinstance(o, dict) and "n" in o and "dlps" in o]:
        tot = o["dlps"] * o["n"]
        add(tot, "ACC", p + ".dlps*n")
        top = [x[1] for x in o.get("dlps_top", [])]
        if top and tot:
            add(100 * sum(top) / tot, "ACC", p + " top3 share %")
            add((tot - sum(top)) / (o["n"] - len(top)), "ACC", p + " mean w/o top3")
            add(sum(top), "ACC", p + " top3 sum")
    for p, o in _dicts(A):
        if isinstance(o, list):
            add(len(o), "ACC", p + "#len")
        if isinstance(o, dict):
            add(len(o), "ACC", p + "#keys")
    # report md, run log, prereg, first run, CR working paper, thresholds
    for path, cls in [(MD, "MD"), (LOG, "LOG"), (PREREG, "PREREG")]:
        t = path.read_text().replace("−", "-").replace(",000", "000")
        for m in re.finditer(r"(?<![\w.])([+-]?)(\d+(?:\.\d+)?)", t):
            add(float(m.group(1) + m.group(2)), cls, path.name)
            add(float(m.group(2)), cls, path.name)
    for p, v in leaves(F):
        add(v, "FULL", p)
        if isinstance(v, float) and 0 <= abs(v) <= 1:
            add(100 * v, "FULL", p + "*100")
    for p, o in [(p, o) for p, o in _dicts(F) if isinstance(o, dict) and "n" in o and "dlps" in o]:
        pass
    cr_full = {
        "dlps": {"GDP": (0.193, 0.133, 0.090, 0.060), "INFL": (0.075, 0.083, 0.155, 0.138),
                 "UR": (0.162, 0.098, -0.182, -0.707), "TBILL": (0.284, 0.043, -0.304, -0.403)},
        "crps_ratio": {"GDP": (0.864, 0.856, 0.890, 0.927), "INFL": (0.954, 0.944, 0.895, 0.887),
                       "UR": (0.922, 0.897, 0.906, 1.001), "TBILL": (0.822, 0.915, 1.010, 1.059)},
        "rmse_ratio": {"GDP": (0.896, 0.878, 0.932, 1.007), "INFL": (0.966, 0.949, 0.915, 0.883),
                       "UR": (0.967, 0.930, 0.914, 0.971), "TBILL": (0.933, 0.944, 0.974, 0.990)},
    }
    for k, d in cr_full.items():
        for var, t in d.items():
            for h, v in zip((1, 2, 4, 8), t):
                add(v, "CR", f"CR_FULL.{k}.{var}.h{h}")
    for var, t in {"GDP": (0.222, 0.214), "INFL": (0.077, 0.152), "UR": (0.187, -0.065),
                   "TBILL": (0.316, -0.261)}.items():
        for h, v in zip((1, 8), t):
            add(v, "CR", f"CR_PRE_DLPS.{var}.h{h}")
    for v in (0.9, 0.10, 0.95, 95, 75, 25, 1.01, 400, 3, 1, 2, 4, 8, 6, 24, 20, 0.5, 10, 1.03):
        add(v, "CONST", "threshold/design constant")


def _dicts(o, p=""):
    yield p, o
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _dicts(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _dicts(v, f"{p}[{i}]")


def lookup(val: float, d: int, sign: str):
    tol = 0.5 * 10 ** -d + 1e-9
    out = []
    for v, cls, lab in INDEX:
        if sign == "+" and not v > 0:
            continue
        if sign == "-" and not v < 0:
            continue
        x = val if sign != "-" else -val
        if sign == "" and v < 0:
            x = -val  # an unsigned token may be quoted as a magnitude
        if abs(v - x) <= tol:
            out.append((cls, lab))
    return out


# =========================================================================== layer B
SCOPE = [
    ("sections/00_abstract.tex", None),
    ("sections/01_introduction.tex", None),
    ("sections/06_evaluation.tex", (75, 135)),
    ("sections/08_validation.tex", (186, 215)),
    ("sections/08_validation.tex", (340, 346)),
    ("sections/09_robustness.tex", None),
    ("sections/11_replication.tex", None),
    ("sections/12_limitations.tex", None),
    ("sections/13_conclusion.tex", None),
    ("sections/appendix.tex", "app:cr"),
    ("sections/appendix.tex", "app:todo"),
    *[(str(p.relative_to(PAPER)), None) for p in sorted((PAPER / "figures/tables").glob("tab_cr_*.tex"))],
    ("figures/tables/tab_runs.tex", None),
]

DROP_ARGS = {  # command -> number of brace arguments to blank (optional [..] args are blanked too)
    "cref": 1, "Cref": 1, "label": 1, "ref": 1, "eqref": 1, "pageref": 1, "citep": 1, "citet": 1,
    "cite": 1, "citealp": 1, "citeauthor": 1, "includegraphics": 1, "input": 1, "lstinputlisting": 1,
    "setlength": 2, "hspace": 1, "vspace": 1, "cmidrule": 1, "multicolumn": 2, "resizebox": 2,
    "newsavebox": 1, "sbox": 1, "usebox": 1, "begin": 1, "end": 1, "hyperref": 0, "url": 1,
    "addlinespace": 0, "rule": 2, "arraybackslash": 0, "linewidth": 0, "columnwidth": 0,
    "todo": 1,  # author notes are not rendered in a submission build; listed separately
}


def blank(s: str, a: int, b: int) -> str:
    seg = s[a:b]
    return s[:a] + re.sub(r"[^\n]", " ", seg) + s[b:]


def preprocess(src: str) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Blank non-rendered arguments; return text and the spans of \\acc, \\fullrun and \\code args."""
    s = strip_comments(src)
    acc, full, code = [], [], []
    i = 0
    while True:
        m = re.compile(r"\\([A-Za-z]+)\*?").search(s, i)
        if not m:
            break
        name, j = m.group(1), m.end()
        if name == "acc":
            while j < len(s) and s[j] in " \n":
                j += 1
            arg, k = brace_arg(s, j)
            acc.append((m.start(), k))
            s = blank(s, m.start(), k)
            i = k
            continue
        if name in ("fullrun", "code", "pkg"):
            if j < len(s) and s[j] == "{":
                _, k = brace_arg(s, j)
                (full if name == "fullrun" else code).append((j, k))
            i = j
            continue
        if name in DROP_ARGS:
            k = j
            n = DROP_ARGS[name]
            # optional args and parenthesised args (cmidrule(lr))
            while k < len(s) and s[k] in "[(":
                close = "]" if s[k] == "[" else ")"
                k = s.index(close, k) + 1
            for _ in range(n):
                while k < len(s) and s[k] in " ":
                    k += 1
                if k < len(s) and s[k] == "{":
                    _, k = brace_arg(s, k)
            if name == "begin" and re.match(r"\{(tabular|tabularx|longtable)\}", s[j:]):
                while k < len(s) and s[k] in " [":
                    if s[k] == "[":
                        k = s.index("]", k) + 1
                    else:
                        k += 1
                if k < len(s) and s[k] == "{":
                    _, k = brace_arg(s, k)
            s = blank(s, m.start(), k)
            i = k
            continue
        # any other control word: blank the name itself (so \footnotesize etc. carry no digits)
        s = blank(s, m.start(), m.end())
        i = m.end()
    return s, acc, full, code


NUM_TOKEN = re.compile(r"""
    (?P<date>(?<![\w])(?:19|20)\d\dQ[1-4](?![\w]))
  | (?P<time>\b\d\d:\d\d(?::\d\d)?Z?)
  | (?P<hash>(?<![\w])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![\w]))
  | (?P<num>(?<![\w.])(?P<sign>[+\-−])?\d+(?:\{,\}\d{3}|,\d{3})*(?:\.\d+)?(?![\w]))
""", re.X)


def region(path: str, sel):
    lines = (PAPER / path).read_text().split("\n")
    if sel is None:
        return 1, len(lines)
    if isinstance(sel, tuple):
        return sel
    # a \section label: from it to the next \section
    start = next(i for i, l in enumerate(lines, 1) if f"\\label{{{sel}}}" in l and l.lstrip().startswith("\\section"))
    end = next((i for i, l in enumerate(lines, 1) if i > start and l.lstrip().startswith("\\section")), len(lines) + 1) - 1
    return start, end


def layer_b():
    print("\n== Layer B: typed numbers in the rendered text of the in-scope regions")
    unmatched_total = 0
    counted = 0
    for path, sel in SCOPE:
        a, b = region(path, sel)
        src = (PAPER / path).read_text()
        text, acc, full, code = preprocess(src)
        lines = text.split("\n")
        offs = [0]
        for l in lines:
            offs.append(offs[-1] + len(l) + 1)
        print(f"-- {path} lines {a}-{b} ({len(acc)} \\acc marks)")
        for m in NUM_TOKEN.finditer(text):
            pos = m.start()
            line = next(i for i in range(len(offs)) if offs[i + 1] > pos) + 1
            if not a <= line <= b:
                continue
            ctx = " ".join(text[max(0, pos - 70):pos + 50].split())
            infull = any(x <= pos < y for x, y in full)
            incode = any(x <= pos < y for x, y in code)
            tag = "FULL" if infull else "CODE" if incode else ""
            counted += 1
            if m.group("date"):
                status = "date"
                if VERBOSE:
                    print(f"   {line:5d} DATE {m.group(0):8s} {ctx}")
                DATES.append((path, line, m.group(0), ctx))
                continue
            if m.group("time") or m.group("hash"):
                OTHER.append((path, line, m.group(0), ctx))
                if VERBOSE:
                    print(f"   {line:5d} {'TIME' if m.group('time') else 'HASH'} {m.group(0):8s} {ctx}")
                continue
            tok = m.group("num")
            sign = (m.group("sign") or "").replace("−", "-")
            # a "--" range or a subtraction is not a sign
            if sign == "-" and pos > 0 and text[pos - 1] in "-0123456789":
                sign = ""
            # $-0.707$ style sign inside math: "$-$" handled by the raw '-' already
            body = tok.lstrip("+-−").replace("{,}", "").replace(",", "")
            val = float(body)
            d = len(body.split(".")[1]) if "." in body else 0
            hits = lookup(val, d, sign)
            pct = text[m.end():m.end() + 2] == "\\%"
            classes = sorted({c for c, _ in hits})
            want = "FULL" if infull else "ACC"
            strong = (d >= 2) or (d == 0 and val >= 25 and not val.is_integer() is False and val not in (1985, 2007, 2011))
            if infull:
                ok = "FULL" in classes
            else:
                ok = bool({"ACC", "MD", "LOG"} & set(classes)) or bool(classes)
            label = ",".join(classes) or "UNMATCHED"
            row = (path, line, (sign + body) + ("%" if pct else ""), tag, label,
                   [l for c, l in hits if c in ("ACC", want)][:3], ctx)
            if not ok:
                unmatched_total += 1
                UNMATCHED.append(row)
                print(f"   {line:5d} {tag:4s} {row[2]:>10s}  UNMATCHED  | {ctx}")
            else:
                MATCHED.append(row)
                if VERBOSE:
                    print(f"   {line:5d} {tag:4s} {row[2]:>10s}  {label:22s} {row[5][:2]} | {ctx}")
    print(f"  {counted} tokens in scope; {unmatched_total} match no known source (listed above)")
    return counted


DATES: list = []
COMPOSITE: list = []
OTHER: list = []
UNMATCHED: list = []
MATCHED: list = []


# =========================================================================== layer C
def layer_c():
    print("\n== Layer C: typed claims about the run, recomputed from the JSON")
    R = A["relative"]
    ur8 = R["full"]["SV(rw)"]["variables"]["UR"]["8"]
    ur1 = R["full"]["SV(rw)"]["variables"]["UR"]["1"]

    def claim(name, cond, detail=""):
        print(f"  {'ok ' if cond else 'BAD'} {name}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            bad(f"claim: {name} {detail}")

    tot8 = ur8["dlps"] * ur8["n"]
    top8 = sum(x[1] for x in ur8["dlps_top"])
    claim("72% of the 106-origin UR h8 sum from three origins", round(100 * top8 / tot8) == 72,
          f"{100 * top8 / tot8:.2f}%")
    srt = lambda t: sorted(t, key=lambda x: -abs(x[1]))  # noqa: E731
    claim("dlps_top sorted by |d| (h1 and h8)", srt(ur1["dlps_top"]) == ur1["dlps_top"] and srt(ur8["dlps_top"]) == ur8["dlps_top"])
    m8 = (tot8 - top8) / (ur8["n"] - 3)
    tot1 = ur1["dlps"] * ur1["n"]
    m1 = (tot1 - sum(x[1] for x in ur1["dlps_top"])) / (ur1["n"] - 3)
    claim("mean w/o three largest |d|: +0.079 at h8", f"{m8:+.3f}" == "+0.079", f"{m8:+.4f}")
    claim("mean w/o three largest |d|: +0.151 at h1", f"{m1:+.3f}" == "+0.151", f"{m1:+.4f}")
    claim("top origins 2007Q4 +7.6, 2008Q1 +6.6, 2008Q2 +6.5",
          [(o, f"{v:+.1f}") for o, v in ur8["dlps_top"]] == [("2007Q4", "+7.6"), ("2008Q1", "+6.6"), ("2008Q2", "+6.5")])
    claim("UR median falls h1 -> h8", ur8["dlps_median"] < ur1["dlps_median"],
          f"{ur1['dlps_median']:+.3f} -> {ur8['dlps_median']:+.3f}")
    p = R["pre_crisis"]["SV(rw)"]["variables"]["UR"]
    claim("pre-crisis UR mean falls h1 -> h8", p["8"]["dlps"] < p["1"]["dlps"], f"{p['1']['dlps']:+.3f} -> {p['8']['dlps']:+.3f}")
    claim("h=1: path-conditioned == path-integrated in every relative cell",
          all(c["1"]["dlps"] == c["1"]["dlps_conditioned"] for s in R.values() for arm in s.values() for c in arm["variables"].values())
          and all(arm["joint"]["1"]["dlpd"] == arm["joint"]["1"]["dlpd_conditioned"] for s in R.values() for arm in s.values()))
    claim("3.58 h run time (seconds/3600)", f"{A['seconds'] / 3600:.2f}" == "3.58", f"{A['seconds'] / 3600:.4f}")
    claim("every reduced-form arm converged at 100% in both samples", all(v == 1.0 for s in A["converged"].values() for v in s.values()))
    claim("criteria 1,2,3,5,6 met, 4 not", [c["met"] for c in A["criteria"]] == [True, True, True, False, True, True])
    claim("criterion 1: SV(rw) dlps > 0 at h=1,2 all vars both samples",
          all(R[s]["SV(rw)"]["variables"][v][h]["dlps"] > 0 for s in R for v in ("GDP", "INFL", "UR", "TBILL") for h in ("1", "2")))
    claim("full-sample CRPS and RMSE ratios < 1 at h <= 2 (SV(rw))",
          all(R["full"]["SV(rw)"]["variables"][v][h][k] < 1 for v in ("GDP", "INFL", "UR", "TBILL") for h in ("1", "2") for k in ("crps_ratio", "rmse_ratio")),
          "largest " + ", ".join(f"{max(R['full']['SV(rw)']['variables'][v][h][k] for v in ('GDP', 'INFL', 'UR', 'TBILL') for h in ('1', '2')):.3f}" for k in ("crps_ratio", "rmse_ratio")))
    claim("pre-crisis CRPS and RMSE ratios < 1 at h <= 2 (SV(rw)) [if the text says 'in both samples']",
          all(R["pre_crisis"]["SV(rw)"]["variables"][v][h][k] < 1 for v in ("GDP", "INFL", "UR", "TBILL") for h in ("1", "2") for k in ("crps_ratio", "rmse_ratio")),
          "largest " + ", ".join(f"{max(R['pre_crisis']['SV(rw)']['variables'][v][h][k] for v in ('GDP', 'INFL', 'UR', 'TBILL') for h in ('1', '2')):.3f}" for k in ("crps_ratio", "rmse_ratio")))
    claim("GDP has the smallest h=1 gain (full, SV(rw))",
          min(("GDP", "INFL", "UR", "TBILL"), key=lambda v: R["full"]["SV(rw)"]["variables"][v]["1"]["dlps"]) == "GDP")
    V = A["verdicts"]
    pairs = 0
    eq = True
    for s, rows in V.items():
        for r in rows:
            if r["convergence_gate"]:
                twin = [x for x in rows if x["model"] == r["model"] and x["reference"] == r["reference"]
                        and x["horizon"] == r["horizon"] and x.get("axis") == r.get("axis") and not x["convergence_gate"]]
                pairs += 1
                eq &= len(twin) == 1 and twin[0]["verdict"] == r["verdict"]
    claim("gate on and off give the same verdict in every pair", eq, f"{pairs} pairs ({', '.join(f'{s}: {len(r) // 2}' for s, r in V.items())})")
    better = [r for rows in V.values() for r in rows if r["verdict"] == "better"]
    claim("every 'better' verdict via MCS (model in, reference out)",
          all(r["model_in_mcs"] and not r["reference_in_mcs"] and "MCS" in r["reasons"] for r in better),
          f"{sum(1 for r in V['full'] if r['verdict'] == 'better')} + {sum(1 for r in V['pre_crisis'] if r['verdict'] == 'better')} rows")
    sv = [r for r in V["full"] if r["model"] == "SV(rw)" and r["reference"] == "Constant"]
    claim("SV(rw) full: better at h=1, inconclusive at h=2,4,8, neither in MCS at h>=2",
          all((r["verdict"] == "better") if r["horizon"] == 1 else (r["verdict"] == "inconclusive" and not r["model_in_mcs"] and not r["reference_in_mcs"]) for r in sv),
          "; ".join(f"h{r['horizon']} {r['verdict']} gate={r['convergence_gate']}" for r in sv))
    O = A["orderings"]
    po = O["per_ordering"]
    claim("ordering stage: 24 orderings", len(po) == 24)
    claim("ordering stage: 'better' at every horizon, gate on and off, 24/24",
          all(v["verdict"] == "better" for o in po for v in o["verdicts"]),
          f"{sum(1 for o in po for v in o['verdicts'] if v['verdict'] == 'better')} of {sum(len(o['verdicts']) for o in po)} rows")
    p8 = [v["dm_pvalue_better"] for o in po for v in o["verdicts"] if v["horizon"] == 8 and v["convergence_gate"]]
    claim("h=8 ordering DM p mostly above 0.10", sum(x > 0.10 for x in p8) > len(p8) / 2,
          f"{sum(x > 0.10 for x in p8)} of {len(p8)} above 0.10; range {min(p8):.3f}-{max(p8):.3f}")
    claim("criterion 6: verdict_h1 'better' in 24/24, gate on and off",
          sum(o["verdict_h1"] == "better" for o in po) == 24 and sum(o["verdict_h1_ungated"] == "better" for o in po) == 24)
    claim("SV(rw) converged at 100% in every ordering", min(o["converged_sv"] for o in po) == 1.0)
    claim("Constant ordering material = False", O["results"]["Constant"]["material"] is False)
    nm = sum(1 for r in O["results"]["SV(rw)"]["summary"] if r["material"])
    claim("SV(rw) flagged at 14 of its 20 cells", nm == 14 and len(O["results"]["SV(rw)"]["summary"]) == 20, f"{nm} of {len(O['results']['SV(rw)']['summary'])}")
    claim("Constant's 20 variable-horizon cells", len(O["results"]["Constant"]["summary"]) == 20)
    claim("no 'MCMC not converged' warning kind",
          not any("not converged" in k for arm in A["warnings"].values() for k in arm))
    claim("no reduced-form origin RB-degenerate (joint, every horizon, both samples)",
          all(arm["joint"][h]["rb_degenerate"] == 0 for s in R.values() for arm in s.values() for h in arm["joint"]))
    claim("no reduced-form origin RB-degenerate (marginal cells)",
          all(c[h]["rb_degenerate"] == 0 for s in R.values() for arm in s.values() for c in arm["variables"].values() for h in c))
    claim("structural pair: one RB-degenerate origin, 2008Q2, joint h=4 only",
          A["structural"]["joint"]["4"]["rb_degenerate"] == 1 and A["structural"]["joint"]["4"]["rb_degenerate_origins"] == ["2008Q2"]
          and all(A["structural"]["joint"][h]["rb_degenerate"] == 0 for h in ("1", "2", "8")),
          f"{A['structural']['joint']['4']['rb_degenerate_origins']}")
    claim("criterion 5 detail: 0/4 joint and 0/16 cells for each variant",
          A["criteria"][4]["detail"].count("0/4 joint horizons and 0/16 marginal cells") == 2)
    claim("n_origins 106 / 92", A["n_origins"] == {"full": 106, "pre_crisis": 92})
    claim("config: 4 x 20000, burn 1000, 8000 forecast draws, seed 0, integrated",
          (A["config"]["mcmc"]["chains"], A["config"]["mcmc"]["draws"], A["config"]["mcmc"]["burn"], A["config"]["draws"],
           A["config"]["seed"], A["config"]["lpd_method"]) == (4, 20000, 1000, 8000, 0, "integrated"))
    claim("commit c823d20, clean", A["versions"]["git_commit"].startswith("c823d20") and A["versions"]["git_dirty"] is False)
    claim("no origin read from the cache (0 of 742 = 7 x 106)",
          all(v["cached"] == 0 for v in A["cached"].values()) and sum(v["origins"] for v in A["cached"].values()) == 742)
    # first-run comparison of the h = 1 gains (reading rule B yardstick)
    mv_cell = mv_joint = 0.0
    arg = None
    for s in A["relative"]:
        for arm in A["relative"][s]:
            fa = F["relative"][s][arm]
            aa = A["relative"][s][arm]
            for v in aa["variables"]:
                dlt = abs(aa["variables"][v]["1"]["dlps"] - fa["variables"][v]["1"]["dlps"])
                if dlt > mv_cell:
                    mv_cell, arg = dlt, (s, arm, v)
            mv_joint = max(mv_joint, abs(aa["joint"]["1"]["dlpd"] - fa["joint"]["1"]["dlpd"]))
    claim("h=1 moves vs first run below the yardstick (0.0044 per cell, 0.0122 joint)",
          mv_cell < 0.0044 and mv_joint < 0.0122, f"largest cell {mv_cell:.4f} at {arg}; joint {mv_joint:.4f}")
    fj = F["relative"]["full"]["SV(rw)"]["joint"]["1"]["dlpd"]
    claim("first run SV(rw) joint h1 = +0.809", f"{fj:+.3f}" == "+0.809", f"{fj:+.4f}")
    fu = F["relative"]["full"]["SV(rw)"]["variables"]["UR"]
    claim("first run UR h8 +1.954 vs h1 +0.159", f"{fu['8']['dlps']:+.3f}" == "+1.954" and f"{fu['1']['dlps']:+.3f}" == "+0.159",
          f"{fu['1']['dlps']:+.3f} -> {fu['8']['dlps']:+.3f}")
    ft = fu["8"]["dlps_top"]
    claim("first run UR h8 top three +40.4 .. +41.6", sorted(f"{x[1]:+.1f}" for x in ft) == sorted(["+40.4", "+41.6", f"{ft[1][1]:+.1f}"])
          and min(x[1] for x in ft) >= 40.35 and max(x[1] for x in ft) < 41.65, f"{[(o, round(v, 2)) for o, v in ft]}")


def composites():
    print("\n== composite \\acc values (dicts and detail strings), shown for a check by eye")
    for key, raw, printed in COMPOSITE:
        print(f"  {key}\n     JSON:    {json.dumps(raw, ensure_ascii=False)[:600]}\n     printed: {printed[:600]}")


# =========================================================================== layer D: table rows
VARS = ("GDP", "INFL", "UR", "TBILL")


def key_facets(k: str) -> dict:
    f = {}
    m = re.search(r"\b(full|pre_crisis)\b", k)
    if m:
        f["sample"] = m.group(1)
    m = re.search(r"(?:relative|converged)\.(?:full|pre_crisis)(?:\['([^']+)'\]|\.(\w+))", k)
    if m:
        f["arm"] = m.group(1) or m.group(2)
    m = re.search(r"variables\.(\w+)", k)
    if m:
        f["var"] = m.group(1)
    elif ".joint" in k:
        f["var"] = "joint"
    m = re.search(r"(?:joint|variables\.\w+)\['(\d)'\]", k)
    if m:
        f["h"] = m.group(1)
    return f


def layer_d():
    print("\n== Layer D: table rows -- each \\acc key matches its row label and column header")
    files = [PAPER / "sections/11_replication.tex", PAPER / "sections/appendix.tex",
             *sorted((PAPER / "figures/tables").glob("tab_*.tex"))]
    nrows = nkeys = 0
    for fp in files:
        src = strip_comments(fp.read_text())
        for tab in re.finditer(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", src, re.S):
            body = tab.group(1)
            if "\\acc{" not in body:
                continue
            rows = re.split(r"\\\\", body)
            header = None
            carry = {}
            for row in rows:
                cells = row.split("&")
                if any(re.search(r"\$h\s*=\s*\d\$", c) for c in cells) and "\\acc{" not in row:
                    header = [re.search(r"\$h\s*=\s*(\d)\$", c).group(1) if re.search(r"\$h\s*=\s*(\d)\$", c) else None for c in cells]
                if "\\acc{" not in row:
                    continue
                nrows += 1
                lab = {}
                for ci, c in enumerate(cells):
                    c0 = re.sub(r"\\(midrule|addlinespace|toprule|cmidrule\S*|bottomrule)", "", c).strip()
                    if "\\acc{" in c0:
                        break
                    if c0 in VARS or c0 == "joint":
                        lab["var"] = c0
                    elif re.fullmatch(r"\d", c0):
                        lab["h"] = c0
                    elif c0.lower().startswith("full"):
                        lab["sample"] = "full"
                    elif c0.lower().startswith("pre"):
                        lab["sample"] = "pre_crisis"
                    elif c0 in ("SV(rw)", "SV", "SV(rw,t10)", "AR-SV(0.9)", "Constant"):
                        lab["arm"] = c0
                for k2 in ("sample", "arm", "var"):
                    if k2 in lab:
                        carry[k2] = lab[k2]
                        if k2 == "var":
                            carry.pop("h", None)
                    elif k2 in carry:
                        lab[k2] = carry[k2]
                for ci, c in enumerate(cells):
                    for km in re.finditer(r"\\acc\{", c):
                        key, _ = brace_arg(c, km.end() - 1)
                        nkeys += 1
                        f = key_facets(key)
                        exp = dict(lab)
                        if header and ci < len(header) and header[ci]:
                            exp["h"] = header[ci]
                        for k2, v in exp.items():
                            if k2 in f and f[k2] != v:
                                bad(f"{fp.name}: row {' '.join(row.split())[:60]!r}: key {key} has {k2}={f[k2]}, row/column says {v}")
    print(f"  {nrows} table rows, {nkeys} keys checked")


# =========================================================================== layer E: tab:cr-c4, rule B, first-run prose
def c4_cell(sample, var, h):
    c = A["relative"][sample]["SV(rw)"]["variables"][var][h]
    tot = c["dlps"] * c["n"]
    top = [x[1] for x in c["dlps_top"]]
    same = all((t > 0) == (tot > 0) for t in top)
    share = 100 * sum(top) / tot if same else None
    mean = (tot - sum(top)) / (c["n"] - len(top))
    return share, mean


def layer_e():
    print("\n== Layer E: typed cells of tab:cr-c4, rule B, SV(rw) joint move, first-run prose")
    rows = {  # (sample, var, h): (share as printed, mean w/o top 3 as printed)
        ("full", "UR", "1"): ("---", "+0.151"), ("full", "UR", "8"): ("72", "+0.079"),
        ("full", "TBILL", "1"): ("---", "+0.449"), ("full", "TBILL", "8"): ("32", "-0.136"),
        ("pre_crisis", "GDP", "8"): ("---", "+0.152"), ("pre_crisis", "INFL", "8"): ("12", "+0.280"),
        ("pre_crisis", "UR", "8"): (">100", "-0.166"),
    }
    # the typed cells as they stand in the appendix source (re-read, not trusted from above)
    src = strip_comments((PAPER / "sections/appendix.tex").read_text())
    tab = src[src.index(r"\label{tab:cr-c4}"):]
    tab = tab[:tab.index(r"\end{tabular}")]
    typed = re.findall(r"& (---|\$>\$100\\%|\d+\\%) & \$([+-]\d\.\d{3})\$ \\\\", tab)
    print(f"  {len(typed)} typed rows in tab:cr-c4 source: {typed}")
    if len(typed) != len(rows):
        bad(f"tab:cr-c4: expected {len(rows)} typed rows, found {len(typed)}")
    for (key, (sh, mn)), (tsh, tmn) in zip(rows.items(), typed):
        share, mean = c4_cell(*key)
        tsh2 = tsh.replace("\\%", "").replace("$>$", ">")
        exp_sh = "---" if share is None else (">100" if share > 100 else f"{share:.0f}")
        ok = tsh2 == exp_sh and tmn == f"{mean:+.3f}"
        print(f"  {'ok ' if ok else 'BAD'} {key}: printed {tsh2:>5s} {tmn}, recomputed {exp_sh:>5s} {mean:+.4f}"
              + (f" (share {share:.1f}%)" if share is not None else ""))
        if not ok:
            bad(f"tab:cr-c4 {key}: printed {tsh2} {tmn}, recomputed {exp_sh} {mean:+.4f}")
    # SV(rw)'s own joint h=1 move between first and acceptance run (appendix: 0.0106)
    mv = abs(A["relative"]["full"]["SV(rw)"]["joint"]["1"]["dlpd"] - F["relative"]["full"]["SV(rw)"]["joint"]["1"]["dlpd"])
    print(f"  {'ok ' if f'{mv:.4f}' == '0.0106' else 'BAD'} SV(rw) full joint h1 move first->acceptance = {mv:.4f} (text 0.0106)")
    if f"{mv:.4f}" != "0.0106":
        bad(f"SV(rw) joint move {mv:.4f} != 0.0106")
    # rule B yardstick: quick vs first full run
    Q = json.loads((REPO / "benchmarks/reports/clark_ravazzolo_quick.json").read_text())
    cell = joint = 0.0
    svc = svj = 0.0
    for smp in Q["relative"]:
        for arm in Q["relative"][smp]:
            if arm not in F["relative"][smp]:
                continue
            q, f = Q["relative"][smp][arm], F["relative"][smp][arm]
            for v in q["variables"]:
                d = abs(q["variables"][v]["1"]["dlps"] - f["variables"][v]["1"]["dlps"])
                cell = max(cell, d)
                if smp == "full" and arm == "SV(rw)":
                    svc = max(svc, d)
            d = abs(q["joint"]["1"]["dlpd"] - f["joint"]["1"]["dlpd"])
            joint = max(joint, d)
            if smp == "full" and arm == "SV(rw)":
                svj = d
    ok = f"{svc:.3f}" == "0.003" and f"{svj:.3f}" == "0.002" and f"{cell:.4f}" == "0.0044" and f"{joint:.4f}" == "0.0122"
    print(f"  {'ok ' if ok else 'BAD'} rule B (quick vs first): SV(rw) {svc:.4f}/{svj:.4f}, all arms {cell:.4f}/{joint:.4f} (text 0.003/0.002, 0.0044/0.0122)")
    if not ok:
        bad("rule B yardstick numbers")
    # first-run prose of app:cr-first-run
    fu8 = F["relative"]["full"]["SV(rw)"]["variables"]["UR"]["8"]
    chk = [("median UR h8 +0.099", f"{fu8['dlps_median']:+.3f}" == "+0.099", f"{fu8['dlps_median']:+.4f}"),
           ("Gaussian UR h8 -0.499", f"{fu8['dlps_gauss']:+.3f}" == "-0.499", f"{fu8['dlps_gauss']:+.4f}")]
    nm = sum(1 for smp in F["relative"].values() for arm in smp.values() for c in arm["variables"].values() for h, x in c.items() if x["rb_gauss_disagree"])
    nj = sum(1 for smp in F["relative"].values() for arm in smp.values() for h, x in arm["joint"].items() if x["rb_gauss_disagree"])
    h1 = sum(1 for smp in F["relative"].values() for arm in smp.values() for c in arm["variables"].values() for h, x in c.items() if x["rb_gauss_disagree"] and h == "1") + \
        sum(1 for smp in F["relative"].values() for arm in smp.values() for h, x in arm["joint"].items() if x["rb_gauss_disagree"] and h == "1")
    chk.append(("28 marginal and 8 joint sign disagreements, all at h >= 2", (nm, nj, h1) == (28, 8, 0), f"{nm}, {nj}, h=1: {h1}"))
    vf = [r for r in F["verdicts"]["full"] if r["horizon"] == 1 and not r["convergence_gate"] and r["reference"] == "Constant"]
    tot = [r["dlpd"] for r in vf]
    mc = [r["dlpd_mcse"] for r in vf]
    chk.append(("gate off h1 full: all four SV arms better via MCS", len(vf) == 4 and all(r["verdict"] == "better" and r["model_in_mcs"] and not r["reference_in_mcs"] for r in vf), f"{[r['model'] for r in vf]}"))
    chk.append(("total dLPD +85.7 to +87.5, MCSE 1.00-1.06", f"{min(tot):+.1f}" == "+85.7" and f"{max(tot):+.1f}" == "+87.5" and f"{min(mc):.2f}" == "1.00" and f"{max(mc):.2f}" == "1.06",
                f"{min(tot):+.2f}..{max(tot):+.2f}; {min(mc):.3f}..{max(mc):.3f}"))
    FO = F["orderings"]["results"]["SV(rw)"]
    jr = FO["joint_lpd_range"]["1"]
    chk.append(("first-run joint h1 ordering range 0.083", f"{jr:.3f}" == "0.083", f"{jr:.4f}"))
    ratios = [F["relative"]["full"]["SV(rw)"]["variables"][v]["1"]["dlps"] / next(r["lpd_range"] for r in FO["summary"] if r["variable"] == v and r["horizon"] == 1)
              for v in ("GDP", "INFL", "UR", "TBILL")]
    chk.append(("per-variable h1 gains 2.2-7.8 times their ranges", f"{min(ratios):.1f}" == "2.2" and f"{max(ratios):.1f}" == "7.8", f"{[round(x, 2) for x in ratios]}"))
    cr = {"1": 0.075, "2": 0.083, "4": 0.155, "8": 0.138}
    infl = [F["relative"]["full"]["SV(rw)"]["variables"]["INFL"][h]["dlps"] / cr[h] for h in cr]
    chk.append(("inflation gains 1.6-3.2 times CR's", f"{min(infl):.1f}" == "1.6" and f"{max(infl):.1f}" == "3.2", f"{[round(x, 2) for x in infl]}"))
    fc = F["orderings"]["results"]["Constant"]
    flagged = [(r["variable"], r["horizon"], round(r["lpd_range"], 3), round(r["lpd_range_snr"], 1)) for r in fc["summary"] if r["material"]]
    chk.append(("first-run Constant flagged at joint h2 and UR h8", fc["material"] is True and sorted((v, h) for v, h, *_ in flagged) == sorted([("joint", 2), ("UR", 8)]), f"{flagged}"))
    pg = F["relative"]["pre_crisis"]["SV(rw)"]["variables"]["GDP"]["8"]
    chk.append(("first-run pre-crisis GDP h8 +0.108 conditioned, -0.004 Gaussian", f"{pg['dlps']:+.3f}" == "+0.108" and f"{pg['dlps_gauss']:+.3f}" == "-0.004", f"{pg['dlps']:+.4f}, {pg['dlps_gauss']:+.4f}"))
    conv = [o["converged_sv"] for o in F["orderings"]["per_ordering"]]
    chk.append(("first-run SV(rw) converged 57-82% across orderings", f"{100 * min(conv):.0f}" == "57" and f"{100 * max(conv):.0f}" == "82", f"{100 * min(conv):.1f}..{100 * max(conv):.1f}"))
    chk.append(("first-run gate on 0/24, off 24/24", sum(o["verdict_h1"] == "better" for o in F["orderings"]["per_ordering"]) == 0 and sum(o["verdict_h1_ungated"] == "better" for o in F["orderings"]["per_ordering"]) == 24, ""))
    o0 = next(o for o in F["orderings"]["per_ordering"] if o["ordering"] == 0)
    chk.append(("first-run identity ordering: 69/106 converged, joint +0.838", round(o0["converged_sv"] * 106) == 69 and f"{o0['dlpd_h1']:+.3f}" == "+0.838", f"{o0['converged_sv'] * 106:.1f}, {o0['dlpd_h1']:+.4f}"))
    chk.append(("first-run main stage SV(rw) converged 72/106", round(F["converged"]["full"]["SV(rw)"] * 106) == 72, f"{F['converged']['full']['SV(rw)'] * 106:.1f}"))
    chk.append(("first-run SV arms converged 37-68% (main stage)",
                f"{100 * min(v for k, v in F['converged']['full'].items() if k != 'Constant'):.0f}" == "37" and f"{100 * max(v for k, v in F['converged']['full'].items() if k != 'Constant'):.0f}" == "68",
                f"{sorted((k, round(100 * v, 1)) for k, v in F['converged']['full'].items())}; pre {sorted((k, round(100 * v, 1)) for k, v in F['converged']['pre_crisis'].items())}"))
    chk.append(("first-run seconds 3950 (1.10 h)", round(F["seconds"]) == 3950 and f"{F['seconds'] / 3600:.2f}" == "1.10", f"{F['seconds']:.1f}"))
    for name, cond, det in chk:
        print(f"  {'ok ' if cond else 'BAD'} {name}  [{det}]")
        if not cond:
            bad(f"first run: {name} [{det}]")


def layer_d2():
    print("\n== Layer D2: verdict and variant tables of tab_cr_arms.tex (row arm/h, column sample/gate/field)")
    src = strip_comments((PAPER / "figures/tables/tab_cr_arms.tex").read_text())
    tabs = re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", src, re.S)
    n = 0
    # verdict table
    vt = next(t for t in tabs if "gate on" in t)
    arm = None
    cols = {2: ("full", "true", "verdict"), 3: ("full", "false", "verdict"), 4: ("full", "true", "model_in_mcs"),
            5: ("full", "true", "reference_in_mcs"), 6: ("pre_crisis", "true", "verdict"), 7: ("pre_crisis", "false", "verdict"),
            8: ("pre_crisis", "true", "model_in_mcs"), 9: ("pre_crisis", "true", "reference_in_mcs")}
    for row in re.split(r"\\\\", vt):
        if "\\acc{" not in row:
            continue
        cells = [c.strip() for c in row.split("&")]
        c0 = re.sub(r"\\addlinespace|\\midrule", "", cells[0]).strip()
        arm = c0 or arm
        h = cells[1]
        for ci, c in enumerate(cells):
            for km in re.finditer(r"\\acc\{", c):
                key, _ = brace_arg(c, km.end() - 1)
                n += 1
                smp, gate, field = cols[ci]
                exp = f"verdicts.{smp}[model='{arm}',horizon={h},convergence_gate={gate}].{field}"
                if key != exp:
                    bad(f"tab_cr_arms verdict cell ({arm}, h={h}, col {ci}): {key} != {exp}")
    # variants table
    vr = next(t for t in tabs if "Variant" in t)
    hs = {2: "1", 3: "2", 4: "4", 5: "8"}
    for row in re.split(r"\\\\", vr):
        if "\\acc{" not in row:
            continue
        cells = [c.strip() for c in row.split("&")]
        var, ref = re.sub(r"\\midrule", "", cells[0]).strip(), cells[1]
        for ci, c in enumerate(cells):
            for km in re.finditer(r"\\acc\{", c):
                key, _ = brace_arg(c, km.end() - 1)
                n += 1
                exp = (f"variants['{var}']['{ref}'].joint['{hs[ci]}'].dlpd" if ci in hs
                       else f"variants['{var}']['{ref}'].joint['1'].dm_p_better")
                if key != exp:
                    bad(f"tab_cr_arms variant cell ({var} vs {ref}, col {ci}): {key} != {exp}")
    print(f"  {n} keys checked")


if __name__ == "__main__":
    build_index()
    na = layer_a(); composites()
    nb = layer_b()
    layer_c()
    layer_d()
    layer_d2()
    layer_e()
    print("\n== dates in scope (checked by eye against the JSON origins and the design)")
    origins = set(A["orderings"]["origins"])
    known = {"1965Q1", "1985Q1", "2011Q2", "2007Q4", "2009Q4", "2013Q2", "1955Q1", "1959Q1", "2026Q2", "1960Q2"}
    for path, line, d, ctx in DATES:
        flag = "" if d in origins or d in known else "  (not an origin of the run)"
        if VERBOSE or flag:
            print(f"   {path}:{line} {d}{flag} | {ctx}")
    print(f"\n{len(problems)} mismatches; layer A {na} values, layer B {nb} tokens")
    for p in problems:
        print("  -", p)
    raise SystemExit(1 if problems else 0)

