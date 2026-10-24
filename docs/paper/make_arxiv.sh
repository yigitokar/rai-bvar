#!/usr/bin/env bash
# Package the paper for arXiv as docs/paper/arxiv.tar.gz.
#
#   make arxiv               build the PDF (so main.bbl is current), then run this script
#   ./make_arxiv.sh          package the current sources and main.bbl (no build)
#   ./make_arxiv.sh --test   also unpack the tarball into a clean temporary directory and compile
#                            it there with tectonic (the test does not touch this directory)
#   ./make_arxiv.sh --pdflatex
#                            also compile the tarball with pdfLaTeX in the TeX Live docker image
#                            (ARXIV_IMAGE, default texlive/texlive:latest), twice, each in its own
#                            clean temporary directory: (a) as arXiv does, pdflatex three times on
#                            the shipped main.bbl, no BibTeX; (b) pdflatex, bibtex, pdflatex,
#                            pdflatex, and the regenerated main.bbl compared with the shipped one
#   (--test and --pdflatex may be given together)
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
# arXiv compiles with pdfLaTeX; main.tex has a pdfTeX branch (T1, lmodern, listings `literate`,
# straight quotes in \code), which tectonic (XeTeX) does not exercise. --test checks the XeTeX
# branch, --pdflatex the pdfTeX branch (ARXIV_SUBMISSION.md records the result).
set -euo pipefail
cd "$(dirname "$0")"

OUT=arxiv.tar.gz
TEST=0
PDFLATEX=0
for a in "$@"; do
  case "$a" in
    --test) TEST=1;;
    --pdflatex) PDFLATEX=1;;
    *) echo "unknown option $a" >&2; exit 2;;
  esac
done

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

if [[ $PDFLATEX == 1 ]]; then
  IMAGE="${ARXIV_IMAGE:-texlive/texlive:latest}"
  command -v docker >/dev/null || { echo "docker not found" >&2; exit 2; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || { echo "docker image $IMAGE not present (docker pull $IMAGE)" >&2; exit 2; }
  # report one pdfLaTeX build: exit codes, TeX errors, undefined citations/references, rerun
  # requests, missing characters, overfull boxes and the page count of the last run
  report() {
    local d="$1" log="$1/main.log" bad=0
    local errs cites refs rerun miss over pages
    errs=$(grep -c '^!' "$log" || true)
    cites=$(grep -c "Citation \`[^']*' on page [0-9]* undefined" "$log" || true)
    refs=$(grep -c "Reference \`[^']*' on page [0-9]* undefined" "$log" || true)
    rerun=$(grep -c 'Rerun to get\|There were undefined references\|Label(s) may have changed' "$log" || true)
    miss=$(grep -a -c 'Missing character' "$log" || true)
    over=$(grep -c 'Overfull \\[hv]box' "$log" || true)
    pages=$(grep -o 'Output written on main.pdf ([0-9]* pages' "$log" | grep -o '[0-9]* pages' || true)
    echo "  $(head -1 "$log" | cut -c1-80)"
    echo "  exit codes: $(cat "$d/exit.txt" | tr '\n' ' ')"
    echo "  TeX errors: $errs; undefined citations: $cites; undefined references: $refs;" \
         "rerun requests: $rerun; missing characters: $miss; overfull boxes: $over;" \
         "main.pdf: ${pages:-? pages}"
    grep -q -v '^0$' "$d/exit.txt" && bad=1
    [[ "$errs$cites$refs$rerun$miss" == "00000" ]] || bad=1
    return $bad
  }
  status=0
  tmpa=$(mktemp -d "${TMPDIR:-/tmp}/rai-bvar-arxiv-pdflatex.XXXXXX")
  tar -xzf "$OUT" -C "$tmpa"
  echo "== pdfLaTeX (a), shipped main.bbl, no BibTeX, in $tmpa ($IMAGE)"
  docker run --rm -v "$tmpa":/w -w /w "$IMAGE" sh -c \
    'for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex > pdflatex$i.txt 2>&1; echo $?; done > exit.txt'
  report "$tmpa" || status=1
  tmpb=$(mktemp -d "${TMPDIR:-/tmp}/rai-bvar-arxiv-bibtex.XXXXXX")
  tar -xzf "$OUT" -C "$tmpb"
  cp "$tmpb/main.bbl" "$tmpb/main.bbl.shipped"
  echo "== pdfLaTeX (b), pdflatex, bibtex, pdflatex, pdflatex, in $tmpb"
  docker run --rm -v "$tmpb":/w -w /w "$IMAGE" sh -c \
    '{ pdflatex -interaction=nonstopmode main.tex > pdflatex1.txt 2>&1; echo $?;
       bibtex main > bibtex.txt 2>&1; echo $?;
       pdflatex -interaction=nonstopmode main.tex > pdflatex2.txt 2>&1; echo $?;
       pdflatex -interaction=nonstopmode main.tex > pdflatex3.txt 2>&1; echo $?; } > exit.txt'
  report "$tmpb" || status=1
  echo "  $(head -1 "$tmpb/main.blg"); BibTeX warnings: $(grep -c '^Warning' "$tmpb/main.blg" || true)"
  if cmp -s "$tmpb/main.bbl" "$tmpb/main.bbl.shipped"; then
    echo "  regenerated main.bbl is identical to the shipped one"
  else
    echo "  regenerated main.bbl DIFFERS from the shipped one:"; diff "$tmpb/main.bbl.shipped" "$tmpb/main.bbl" | head -20
    status=1
  fi
  echo "(left in $tmpa and $tmpb)"
  [[ $status == 0 ]] && echo "pdfLaTeX test OK" || { echo "pdfLaTeX test FAILED" >&2; exit 1; }
fi
