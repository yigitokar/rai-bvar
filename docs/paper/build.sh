#!/usr/bin/env bash
# Build the rai-bvar paper and check the log.
#
#   ./build.sh            compile main.tex with tectonic, then run the checks below
#   ./build.sh --check    only run the checks on an existing main.log
#   STRICT=1 ./build.sh   also fail on undefined references and multiply-defined labels
#
# Checks (on main.log):
#   * which sections/ file each \rbinput slot of main.tex took, and any slot that is MISSING
#     (error if STRICT=1);
#   * section files that nothing \input (a name that no candidate list of main.tex knows);
#   * "Missing character" (XeTeX drops glyphs the font lacks, e.g. Greek pasted into text): error;
#   * undefined citations: error; undefined references, multiply-defined labels: error if STRICT=1;
#   * overfull boxes wider than 10pt (listed);
#   * the number of \acc{} marks still in the sources and generated tables (figures/tables/), and
#     of \todo{} marks in the sources (information).
set -euo pipefail
cd "$(dirname "$0")"

STRICT="${STRICT:-0}"
if [[ "${1:-}" != "--check" ]]; then
  command -v tectonic >/dev/null || { echo "tectonic not found (brew install tectonic)" >&2; exit 2; }
  tectonic main.tex --keep-intermediates --keep-logs
fi
[[ -f main.log ]] || { echo "no main.log" >&2; exit 2; }

status=0
echo "== section files"
grep -o 'rai-bvar: [A-Za-z]* sections/[^ ]*' main.log | sed 's/^rai-bvar: /  /' || true
if grep -q 'rai-bvar: MISSING' main.log && [[ "$STRICT" == 1 ]]; then status=1; fi

echo "== section files not \\input anywhere"
orph=0
for f in sections/*.tex; do
  b="${f%.tex}"
  if ! grep -q "(${b}\b\|(./${b}\b\|${b}.tex" main.log; then
    echo "  ORPHAN $f"; orph=1
  fi
done
[[ $orph == 0 ]] && echo "  none"

missing_chars=$(grep -c 'Missing character' main.log || true)
echo "== missing characters: ${missing_chars}"
if [[ "$missing_chars" != "0" ]]; then
  grep 'Missing character' main.log | sort | uniq -c | sed 's/^/  /'
  status=1
fi

cites=$(grep -o "Citation \`[^']*' on page [0-9]* undefined" main.log | sed "s/Citation \`\([^']*\)'.*/\1/" | sort -u || true)
echo "== undefined citations: $(printf '%s' "$cites" | grep -c . || true)"
if [[ -n "$cites" ]]; then printf '  %s\n' $cites; status=1; fi

refs=$(grep -o "Reference \`[^']*' on page [0-9]* undefined" main.log | sed "s/Reference \`\([^']*\)'.*/\1/" | sort -u || true)
echo "== undefined references: $(printf '%s' "$refs" | grep -c . || true)"
if [[ -n "$refs" ]]; then printf '  %s\n' $refs; [[ "$STRICT" == 1 ]] && status=1; fi

multi=$(grep -o "Label \`[^']*' multiply defined" main.log | sed "s/Label \`\([^']*\)'.*/\1/" | grep -v '@cref$' | sort -u || true)
echo "== multiply-defined labels: $(printf '%s' "$multi" | grep -c . || true)"
if [[ -n "$multi" ]]; then printf '  %s\n' $multi; [[ "$STRICT" == 1 ]] && status=1; fi

echo "== overfull boxes wider than 10pt"
grep -o 'Overfull \\hbox ([0-9.]*pt too wide).*' main.log \
  | awk -F'[(p]' '{ if ($2 + 0 > 10) print "  " $0 }' || true

# comments are stripped first (an escaped \% is not a comment); main.tex's own \string\todo{} is
# not a mark
acc=$(cat sections/*.tex figures/tables/*.tex 2>/dev/null | sed -e 's/\\%//g' -e 's/%.*$//' | grep -o '\\acc{' | wc -l | tr -d ' ')
todo=$(cat sections/*.tex main.tex | sed -e 's/\\%//g' -e 's/%.*$//' -e 's/\\string\\todo{//g' -e 's/\\todo{abstract not written yet//' \
  | grep -o '\\todo{' | wc -l | tr -d ' ')
echo "== draft marks in the sources: ${acc} \\acc{} (sections and figures/tables), ${todo} \\todo{}"

pages=$(grep -o 'Output written on main.xdv ([0-9]* pages' main.log | grep -o '[0-9]* pages' || true)
echo "== main.pdf: ${pages:-? pages}"
exit $status
