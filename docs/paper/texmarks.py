"""Find the draft marks \\acc{...}, \\todo{...} and \\fullrun{...} in the paper's LaTeX sources.

Shared by fill_acceptance.py and collect_todos.py. Comments are stripped first (an escaped \\% is
not a comment), and arguments are read with balanced braces, so a mark may span several lines.

    from texmarks import find_marks
    for m in find_marks("todo"):           # every \\todo in sections/*.tex (and main.tex)
        print(m.path, m.line, m.text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = [*sorted((HERE / "sections").glob("*.tex"))]


@dataclass(frozen=True)
class Mark:
    path: Path  # source file
    line: int  # 1-based line of the backslash
    text: str  # the argument, verbatim (comments removed, newlines kept)

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(HERE))

    @property
    def flat(self) -> str:
        """The argument on one line (runs of whitespace collapsed)."""
        return " ".join(self.text.split())


def strip_comments(src: str) -> str:
    """Blank out %-comments (keep \\%, keep the newline, keep offsets on each line's start)."""
    out = []
    for line in src.split("\n"):
        i = 0
        cut = len(line)
        while i < len(line):
            c = line[i]
            if c == "\\":
                i += 2  # skip the escaped character (\%, \\, \{, ...)
                continue
            if c == "%":
                cut = i
                break
            i += 1
        out.append(line[:cut] + " " * (len(line) - cut))
    return "\n".join(out)


def _arg(src: str, start: int) -> tuple[str, int] | None:
    """The brace group that opens at src[start] == '{'; returns (content, index after '}')."""
    if start >= len(src) or src[start] != "{":
        return None
    depth = 0
    i = start
    while i < len(src):
        c = src[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1 : i], i + 1
        i += 1
    raise ValueError(f"unbalanced braces from offset {start}")


def find_marks(name: str, paths: list[Path] | None = None) -> list[Mark]:
    """Every \\<name>{...} in the given files (default: sections/*.tex), in file order.

    \\string\\<name>{} (main.tex's error messages) and the macro definitions themselves are skipped.
    """
    pat = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z@])")
    marks: list[Mark] = []
    for path in paths if paths is not None else SOURCES:
        raw = path.read_text(encoding="utf-8")
        src = strip_comments(raw)
        for m in pat.finditer(src):
            before = src[max(0, m.start() - 8) : m.start()]
            if before.endswith("\\string") or before.endswith("Command{") or before.endswith("command{"):
                continue
            j = m.end()
            while j < len(src) and src[j] in " \t":
                j += 1
            got = _arg(src, j)
            if got is None:
                continue
            text, _ = got
            line = src.count("\n", 0, m.start()) + 1
            marks.append(Mark(path, line, text))
    return marks


if __name__ == "__main__":
    for kind in ("acc", "todo", "fullrun"):
        ms = find_marks(kind)
        print(f"{kind}: {len(ms)} marks, {len({m.flat for m in ms})} distinct")
