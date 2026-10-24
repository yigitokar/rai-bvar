"""Copy a LaTeX source without its comment text, for the arXiv tarball (make_arxiv.sh).

arXiv publishes the source files of a submission. The annotated sources (the evidence notes and
TODO tags in comments) stay on the branch paper-draft; the tarball's .tex files (main.tex,
sections/*.tex, figures/tables/*.tex, acceptance_values.tex) carry no comment text. The rules
keep the typeset output identical:

- a line holding only a comment (optional spaces or tabs, then %) is dropped: TeX skips the
  leading spaces and the comment eats the line end, so the line contributes nothing;
- elsewhere the text after the first unescaped % is dropped and everything up to and including
  the % is kept, since a % at the line end suppresses the space the line end would give and a
  space before the % is part of the text;
- a % after an odd number of backslashes (\\%) is text, not a comment.

Verbatim-like material (verbatim, lstlisting and similar environments, \\verb, \\lstinline) reads
% as text; the script refuses a file that has any, rather than guess. \\url, \\href and \\code are
not parsed, so a % inside their arguments would be cut: the sources have none, and the pdfLaTeX
test of make_arxiv.sh and the text comparison in ARXIV_SUBMISSION.md would show one.

Usage: python3 strip_comments.py SRC DST
"""

from __future__ import annotations

import re
import sys

VERBATIM = re.compile(
    r"\\begin\{(verbatim|Verbatim|lstlisting|comment|minted|filecontents)\*?\}|\\verb\W|\\lstinline\W"
)


def comment_start(line: str) -> int:
    """Index of the first % that starts a comment, or -1."""
    for i, c in enumerate(line):
        if c != "%":
            continue
        k = i
        while k > 0 and line[k - 1] == "\\":
            k -= 1
        if (i - k) % 2 == 0:
            return i
    return -1


def strip(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = line[len(body):]
        i = comment_start(body)
        if i < 0:
            out.append(line)
        elif body[:i].strip(" \t") == "":
            continue
        else:
            out.append(body[: i + 1] + end)
    return "".join(out)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    m = VERBATIM.search(text)
    if m:
        sys.exit(f"strip_comments.py: {src} has verbatim-like material ({m.group(0)!r}); not stripping")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(strip(text))


if __name__ == "__main__":
    main()
