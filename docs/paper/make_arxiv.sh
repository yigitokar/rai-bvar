#!/usr/bin/env bash
# Package the paper for arXiv as docs/paper/arxiv.tar.gz.
#
#   make arxiv               build the PDF (so main.bbl is current), then run this script
#   ./make_arxiv.sh          package the current sources and main.bbl (no build)
#   ./make_arxiv.sh --test   also unpack the tarball into a clean temporary directory and compile
#                            it there with tectonic (the test does not touch this directory)
#
# The tarball holds what the LaTeX build reads and nothing else:
#   main.tex, main.bbl (arXiv does not run BibTeX), refs.bib,
#   sections/*.tex (not the staging_*.tex files of the page-budget pass, if any are left),
#   the figure PDFs that the sections \includegraphics,
#   the generated table bodies figures/tables/*.tex,
#   the listing inputs that the sections \lstinputlisting (code/ex_a.py, code/ex_b.py and their
#     captured output in code/out/: the paper prints them, so they are content, not build scripts),
#   acceptance_values.tex if `make fill` has written it.
# No figure or table scripts, no data, no intermediates, no main.pdf.
#
# arXiv compiles with pdfLaTeX; main.tex has a pdfTeX branch (T1, lmodern, listings `literate`),
# which tectonic (XeTeX) does not exercise. The --test build checks the XeTeX branch only.
set -euo pipefail
cd "$(dirname "$0")"

OUT=arxiv.tar.gz
TEST=0
[[ "${1:-}" == "--test" ]] && TEST=1

[[ -f main.bbl ]] || { echo "no main.bbl: run ./build.sh (or make pdf) first" >&2; exit 2; }
for f in refs.bib main.tex; do
  if [[ "$f" -nt main.bbl ]]; then
    echo "warning: $f is newer than main.bbl; run ./build.sh first so the .bbl is current" >&2
  fi
done

files=(main.tex main.bbl refs.bib)
[[ -f acceptance_values.tex ]] && files+=(acceptance_values.tex)

# section files, except leftover staging fragments
for f in sections/*.tex; do
  case "$(basename "$f")" in staging_*) echo "skipping $f (staging)" >&2; continue;; esac
  files+=("$f")
done

# generated table bodies
for f in figures/tables/*.tex; do files+=("$f"); done

# figures and listing inputs named in the sources (comments stripped first)
src=$(cat main.tex sections/*.tex | sed -e 's/\\%//g' -e 's/%.*$//')
while IFS= read -r g; do
  [[ -z "$g" ]] && continue
  g="${g#figures/}"; g="figures/${g}"
  [[ "$g" == *.pdf ]] || g="${g}.pdf"
  [[ -f "$g" ]] || { echo "missing figure $g" >&2; exit 1; }
  files+=("$g")
done < <(printf '%s' "$src" | tr '\n' ' ' | grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' \
           | sed 's/.*{\([^}]*\)}/\1/' | sort -u)
while IFS= read -r l; do
  [[ -z "$l" ]] && continue
  [[ -f "$l" ]] || { echo "missing listing input $l" >&2; exit 1; }
  files+=("$l")
done < <(printf '%s' "$src" | tr '\n' ' ' | grep -oE '\\lstinputlisting(\[[^]]*\])?\{[^}]*\}' \
           | sed 's/.*{\([^}]*\)}/\1/' | sort -u)

# de-duplicate, keep order
uniq_files=()
for f in "${files[@]}"; do
  skip=0
  for u in "${uniq_files[@]:-}"; do [[ "$u" == "$f" ]] && { skip=1; break; }; done
  [[ $skip == 0 ]] && uniq_files+=("$f")
done

# draft marks: a draft is packaged as is, but say so
if grep -q '^\\draftmarkstrue' main.tex; then
  echo "note: main.tex has \\draftmarkstrue (draft marks print); set \\draftmarksfalse for the" \
       "submission build" >&2
fi

rm -f "$OUT"
COPYFILE_DISABLE=1 tar -czf "$OUT" "${uniq_files[@]}"
echo "wrote $OUT: ${#uniq_files[@]} files, $(du -h "$OUT" | cut -f1 | tr -d ' ')"
tar -tzf "$OUT" | sed 's/^/  /'

if [[ $TEST == 1 ]]; then
  command -v tectonic >/dev/null || { echo "tectonic not found" >&2; exit 2; }
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/rai-bvar-arxiv.XXXXXX")
  tar -xzf "$OUT" -C "$tmp"
  echo "== test build in $tmp"
  ( cd "$tmp" && tectonic main.tex --keep-logs > build.txt 2>&1 ) \
    || { echo "test build FAILED; see $tmp/build.txt" >&2; tail -30 "$tmp/build.txt" >&2; exit 1; }
  pages=$(grep -o 'Output written on main.xdv ([0-9]* pages' "$tmp/main.log" | grep -o '[0-9]* pages' || true)
  undef=$(grep -c "undefined" "$tmp/main.log" || true)
  miss=$(grep -c 'Missing character' "$tmp/main.log" || true)
  echo "test build OK: main.pdf ${pages:-? pages}; log lines with 'undefined': ${undef}; missing characters: ${miss}"
  echo "(left in $tmp)"
fi
