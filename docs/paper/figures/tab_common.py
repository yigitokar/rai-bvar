"""Shared helpers for the table generators ``tab_*.py`` (not a table generator itself).

Every generator writes one LaTeX file, ``docs/paper/figures/tables/<stem>.tex``, whose ``<stem>`` is
the script stem named in OUTLINE.md section 14.2 and in the appendix captions (``tab_geweke``,
``tab_bench``, ...). The file holds only the *body* of a float: one or more ``tabular``
environments and, where needed, a short note. The caption, the label and the evidence tags of the
float stay in ``sections/*.tex``, which reads the file with ``\\input``. The first lines
of every file are ``%`` comments that name the generator, its data sources and their evidence tags,
so the provenance travels with the table.

Three kinds of cell need care (OUTLINE section 0.5):

* acceptance-run numbers are never written: such a cell is ``\\acc{<json key>}`` with a key in the
  grammar of ``sections/11_replication.tex``, which prints a red placeholder until
  ``acceptance_values.tex`` (``fill_acceptance.py``) defines the key. In these tables the
  placeholder prints as a short red ``[ACC]`` (:data:`SHORT_ACC`); the keys are in the ``.tex``
  file. ``--check-acc REPORT`` verifies that every emitted key resolves in a report without
  printing a value.
* first-full-run numbers (report 551f181) are wrapped in ``\\fullrun{}``.
* numbers that must come from the release validation records (``docs/paper/data/release/``,
  OUTLINE TODO 7) are a red ``[REL]`` mark (:func:`mark`) until those records exist. A mark left in
  a build with ``\\draftmarksfalse`` is a LaTeX error, like ``\\acc`` and ``\\todo``.

Run the generators with the main repository's environment, without writing byte code::

    PYTHONDONTWRITEBYTECODE=1 /Users/yigitokar/rai-bvar/.venv/bin/python tab_<name>.py
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent  # docs/paper/figures
PAPER = HERE.parent  # docs/paper
DATA = HERE / "data"
TABLES = HERE / "tables"
MAIN_REPO = Path(os.environ.get("RAI_BVAR_REPO", "/Users/yigitokar/rai-bvar"))
RELEASE = PAPER / "data" / "release"  # OUTLINE section 0.5 [REL]
ACCEPTANCE_JSON = MAIN_REPO / "benchmarks" / "reports" / "clark_ravazzolo_acceptance.json"

OUTDIR = TABLES  # --outdir (dry runs write elsewhere, never over the paper's tables)

# The short placeholder of \acc inside these tables (local to the float's group; main.tex keeps the
# long form elsewhere). \rb@acc is main.tex's placeholder printer; if it is ever renamed, the cells
# fall back to the long placeholder and nothing else changes.
SHORT_ACC = (
    r"\makeatletter\ifdefined\rb@acc\def\rb@acc#1{{\color{accmark}\scriptsize[ACC]}}\fi\makeatother"
)

# [REL] / [TC] marks: red in a draft, a LaTeX error in a submission build (\draftmarksfalse)
MARK_MACRO = (
    r"\providecommand{\rbtabmark}[1]{\ifdraftmarks{\color{accmark}\scriptsize[#1]}\else"
    r"\PackageError{rai-bvar}{placeholder [#1] left in a submission build}"
    r"{Regenerate the table from its data (docs/paper/figures/tables/)}\fi}"
)

# a tabular that is wider than the float is shrunk to \linewidth; otherwise it keeps its size
FIT_BOX = r"\ifdefined\rbtabbox\else\newsavebox{\rbtabbox}\fi"


def mark(tag: str = "REL") -> str:
    return r"\rbtabmark{" + tag + "}"


def acc(key: str) -> str:
    return r"\acc{" + key + "}"


def fullrun(s: str) -> str:
    return r"\fullrun{" + s + "}"


# --------------------------------------------------------------------------------- text and numbers
_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}  # fmt: skip
_UNICODE = {"σ": r"$\sigma$", "φ": r"$\phi$", "μ": r"$\mu$", "ν": r"$\nu$", "λ": r"$\lambda$",
            "Δ": r"$\Delta$", "×": r"$\times$", "≤": r"$\leq$", "≥": r"$\geq$", "−": "-",
            "–": r"\textendash{}", "—": r"\textemdash{}", "…": r"\ldots{}"}  # fmt: skip


def tex(s) -> str:
    """Plain text for LaTeX: special characters escaped, a few Unicode symbols mapped."""
    out = []
    for ch in str(s):
        out.append(_SPECIAL.get(ch) or _UNICODE.get(ch) or ch)
    # ASCII "--" in quoted text (command-line flags) stays two hyphens, not an en dash
    return "".join(out).replace("--", "-{}-").replace("-{}--", "-{}-{}-")


def code(s) -> str:
    r"""``\code{s}``: identifiers and paths as typed (main.tex's \code un-escapes \_ itself)."""
    s = str(s)
    if any(c in s for c in "{}%"):
        raise ValueError(f"\\code cannot hold {s!r}")
    return r"\code{" + s + "}"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def signed(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
        return "$" + s + "$"
    return "$" + ("+" if x > 0 else "") + s + "$"


def sci(x: float, digits: int = 3) -> str:
    """``$a\\times10^{b}$`` with ``digits`` significant digits (plain decimals for 1e-3 <= |x| < 1e4)."""
    if x == 0 or not math.isfinite(x):
        return f"${x:g}$"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e <= 3:
        return f"${x:.{max(digits - 1 - e, 0)}f}$"
    m = x / 10**e
    ms = f"{m:.{digits - 1}f}"
    if float(ms) >= 10:  # rounding carried into the next decade
        e += 1
        ms = f"{m / 10:.{digits - 1}f}"
    return f"${ms}\\times10^{{{e}}}$"


def thousands(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "{,}")


# ----------------------------------------------------------------------------------------- sources
def git(*args: str, repo: Path = MAIN_REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def git_show(commit: str, path: str, repo: Path = MAIN_REPO) -> str:
    """The text of ``path`` at ``commit`` (the committed version, whatever the working tree holds)."""
    return git("show", f"{commit}:{path}", repo=repo)


def git_show_json(commit: str, path: str, repo: Path = MAIN_REPO) -> dict:
    return json.loads(git_show(commit, path, repo=repo))


def is_ancestor(a: str, b: str, repo: Path = MAIN_REPO) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", a, b],
                       capture_output=True)  # fmt: skip
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git merge-base --is-ancestor {a} {b}: {r.stderr.decode().strip()}")
    return r.returncode == 0


def literal(source: str, name: str):
    """The value of the top-level assignment ``name = <literal>`` in Python ``source`` (``ast``; the
    module is never imported)."""
    tree = ast.parse(source)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        if name in targets:
            return ast.literal_eval(value)
    raise KeyError(f"no top-level literal {name!r}")


def functions(source: str) -> dict[str, list[str]]:
    """Top-level test functions of a module and the ``pytest.mark.<x>`` names on each (``ast``)."""
    tree = ast.parse(source)
    module_marks: list[str] = []
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark"
                                                for t in node.targets):  # fmt: skip
            module_marks += re.findall(r"pytest\.mark\.(\w+)", ast.unparse(node.value))
        if isinstance(node, ast.FunctionDef):
            marks = []
            for d in node.decorator_list:
                marks += re.findall(r"pytest\.mark\.(\w+)", ast.unparse(d))
            out[node.name] = [m for m in module_marks + marks if m not in ("parametrize",
                                                                            "filterwarnings")]  # fmt: skip
    return out


def grab(text: str, pattern: str, group: int | str = 1, flags: int = re.S) -> str:
    """The one match of ``pattern`` in ``text`` (an error if there is none or more than one): numbers
    quoted from prose are taken from the source at generation time, never typed into a script."""
    ms = list(re.finditer(pattern, text, flags))
    if len(ms) != 1:
        raise ValueError(f"pattern {pattern!r}: {len(ms)} matches (expected 1)")
    return ms[0].group(group)


def section(text: str, heading_regex: str) -> str:
    """The Markdown section whose heading matches ``heading_regex``, up to the next heading of the
    same or a higher level."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"(#+)\s", line)
        if m and re.search(heading_regex, line):
            level = len(m.group(1))
            j = i + 1
            while j < len(lines):
                n = re.match(r"(#+)\s", lines[j])
                if n and len(n.group(1)) <= level:
                    break
                j += 1
            return "\n".join(lines[i:j])
    raise KeyError(f"no heading matching {heading_regex!r}")


def md_table(text: str) -> list[dict[str, str]]:
    """The first Markdown pipe table in ``text`` as a list of rows keyed by the header cells."""
    rows = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
    if len(rows) < 3:
        raise ValueError("no Markdown table")

    def cells(ln: str) -> list[str]:
        ln = ln.strip().strip("|").replace(r"\|", "\x00")
        return [c.strip().replace("\x00", "|") for c in ln.split("|")]

    head = cells(rows[0])
    out = []
    for ln in rows[2:]:
        c = cells(ln)
        if len(c) != len(head):
            raise ValueError(f"row with {len(c)} cells under a {len(head)}-cell header: {ln!r}")
        out.append(dict(zip(head, c, strict=True)))
    return out


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -------------------------------------------------------------------------------------------- output
def add_common_args(ap: argparse.ArgumentParser, *, acc_keys: bool = False) -> None:
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write the table here instead of docs/paper/figures/tables/")  # fmt: skip
    if acc_keys:
        ap.add_argument("--check-acc", type=Path, nargs="?", const=ACCEPTANCE_JSON, default=None,
                        metavar="REPORT",
                        help="check that every \\acc key of the written table resolves in REPORT "
                        "(default: the acceptance report of the main repository); prints counts "
                        "and unresolved keys only, never a value")  # fmt: skip


def apply_outdir(args) -> None:
    global OUTDIR
    if getattr(args, "outdir", None):
        args.outdir.mkdir(parents=True, exist_ok=True)
        OUTDIR = args.outdir.resolve()


def fit(tabular: str) -> str:
    """Box ``tabular`` and shrink it to ``\\linewidth`` only if it is wider."""
    return (f"{FIT_BOX}%\n\\sbox{{\\rbtabbox}}{{%\n{tabular}}}%\n"
            "\\ifdim\\wd\\rbtabbox>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox{\\rbtabbox}}%\n"
            "\\else\\usebox{\\rbtabbox}\\fi")  # fmt: skip


def note(text: str, *, red: bool = False) -> str:
    """A note under the tabular, in the float's width."""
    col = r"\color{accmark}" if red else ""
    return ("\\par\\smallskip\n{\\footnotesize\\raggedright" + col + " " + text + "\\par}")


def write(name: str, body: str, *, sources: Iterable[str], status: str, generator: str) -> Path:
    """Write ``tables/<name>.tex``: header comments (generator, sources, status), then ``body``."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.tex"
    head = [f"% {name}.tex -- generated by docs/paper/figures/{generator}; do not edit by hand.",
            f"% Status: {status}"]  # fmt: skip
    head += [f"% Source: {s}" for s in sources]
    head += ["% Body only: the caption, label and evidence tags are in sections/*.tex.", ""]
    text = "\n".join(head) + _acc_note_unless_filled(body.rstrip()) + "\n"
    path.write_text(text, encoding="utf-8")
    rel = path.relative_to(PAPER) if path.is_relative_to(PAPER) else path
    print(f"wrote {rel}")
    return path


def check_acc(path: Path, report: Path | None) -> None:
    """Every ``\\acc`` key of ``path`` must resolve in ``report`` (fill_acceptance's grammar).
    Prints the number of keys and the unresolved ones; no value is printed."""
    if report is None:
        return
    sys.path.insert(0, str(PAPER))
    from fill_acceptance import Unresolved, render  # noqa: PLC0415
    from texmarks import find_marks  # noqa: PLC0415

    rep = json.loads(Path(report).read_text())
    keys = sorted({m.flat for m in find_marks("acc", [path])})
    bad = []
    for k in keys:
        try:
            render(rep, k)
        except (Unresolved, KeyError, IndexError, TypeError, ValueError) as err:
            bad.append(f"{k}: {type(err).__name__}")
    print(f"{path.name}: {len(keys)} distinct \\acc keys, {len(keys) - len(bad)} resolve in "
          f"{report}")  # fmt: skip
    for b in bad:
        print(f"  UNRESOLVED {b}")
    if bad:
        raise SystemExit(1)


ACC_NOTE_TEXT = "Placeholder: each [ACC] cell holds a key of the acceptance report"


def acc_placeholder_note(prefix: str) -> str:
    """The red note under a table whose cells are acceptance-report keys. :func:`write` makes it
    conditional: it prints only while some ``\\acc`` key of the table has no value, i.e. until
    ``acceptance_values.tex`` defines every key (then each cell prints its number instead)."""
    where = tex(prefix) if "," in prefix else code(prefix)
    return (f"{ACC_NOTE_TEXT} (under "
            f"{where}); the number is never written here. The cell prints the value once "
            r"\code{acceptance_values.tex} (\code{fill_acceptance.py}) defines its key.")


def acc_keys(text: str) -> list[str]:
    """The arguments of every ``\\acc{...}`` in ``text`` (balanced braces), in order, unique."""
    out: list[str] = []
    for m in re.finditer(r"\\acc(?![A-Za-z@])\s*\{", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        key = text[m.end() : j - 1]
        if key not in out:
            out.append(key)
    return out


def _acc_note_unless_filled(body: str) -> str:
    """Wrap the red [ACC] note (:func:`acc_placeholder_note` through :func:`note`) so that it is
    typeset only while at least one ``\\acc`` key of the table is undefined. The test uses
    ``\\rb@accifdef`` of main.tex, the lookup ``\\acc`` itself uses, on the same key text; when
    main.tex's macros are absent the note is always printed."""
    i = body.find(ACC_NOTE_TEXT)
    if i < 0:
        return body
    start = body.rfind("\\par\\smallskip\n", 0, i)
    end = body.find("\\par}", i)
    if start < 0 or end < 0:
        return body
    end += len("\\par}")
    keys = acc_keys(body[:start] + body[end:])
    tests = "".join(r"\rb@accifdef{" + k + r"}{\@gobble}{\gdef\rbaccunfilled{1}}%" + "\n"
                    for k in keys)  # fmt: skip
    cond = ("% the red note below prints only while a key of this table has no value "
            "(acceptance_values.tex, fill_acceptance.py)\n"
            r"\makeatletter\gdef\rbaccunfilled{}\ifdefined\rb@accifdef" + "%\n" + tests
            + r"\else\gdef\rbaccunfilled{1}\fi\makeatother" + "%\n"
            + r"\ifx\rbaccunfilled\empty\else" + "%\n" + body[start:end] + "%\n" + r"\fi")
    return body[:start] + cond + body[end:]


def preamble(*, acc_cells: bool = False, marks: bool = False) -> str:
    parts = []
    if acc_cells:
        parts.append(SHORT_ACC + "%")
    if marks:
        parts.append(MARK_MACRO + "%")
    return "\n".join(parts) + ("\n" if parts else "")


if __name__ == "__main__":  # a `for f in tab_*.py` loop may run this file: do nothing, successfully
    print("tab_common.py holds helpers for the tab_*.py generators; nothing to generate")
