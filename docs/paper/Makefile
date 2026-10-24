# rai-bvar paper: figures and PDF.
#
#   make              figures that need no acceptance data, then the PDF and the log checks
#   make pdf          the PDF and the checks only (build.sh)
#   make figures      every figure script (placeholders where the data do not exist yet)
#   make acc-figures  Fig6 and FigF2 from the acceptance report (ACC=...), FigF1/FigF3 from the
#                     per-origin export (run `make per-origin` first)
#   make per-origin   export the acceptance run's per-origin scores from its score cache
#                     (read-only; run once, after the acceptance run has finished; the export
#                     script asks for a clean checkout of c823d20 and only warns otherwise, so
#                     pass EXPORT_REPO=<that checkout> when the main repository is on another
#                     commit or has uncommitted changes)
#   make fill         write acceptance_values.tex from the acceptance report (ACC=...), so that
#                     \acc{} prints numbers (refuses a report that is not the registered run)
#   make tables       every table generator figures/tab_*.py -> figures/tables/tab_*.tex (the
#                     appendix \input's them; [REL]/[ACC] cells stay marks until the data exist)
#   make todos        regenerate AUTHOR_TODO.md from the \todo marks and OUTLINE.md section 16
#   make arxiv        build the PDF, then write arxiv.tar.gz (make_arxiv.sh; ARXIV_TEST=1 also
#                     compiles the tarball in a clean temporary directory)
#   make clean        remove LaTeX intermediates
#
# Figure scripts run with the main repository's environment and import its code read-only
# (no byte code is written). Override REPO/PY for another checkout.

REPO  ?= /Users/yigitokar/rai-bvar
PY    ?= $(REPO)/.venv/bin/python
ACC   ?= $(REPO)/benchmarks/reports/clark_ravazzolo_acceptance.json
CACHE ?= $(HOME)/.cache/rai-bvar-cr-c823d20
EXPORT_REPO ?= $(REPO)

export PYTHONDONTWRITEBYTECODE := 1
export MPLBACKEND := pdf
export RAI_BVAR_REPO := $(REPO)

# figures whose data exist now: static, computed from committed code, or drawn from the release
# validation record at 7f880e1 (benchmarks/reports/validation/7f880e1_apple-m5-max/record/, main
# 3d5a407; FigD1 and FigD2 name it in a footer)
NOW := fig_architecture fig_verdict_flow fig_mutation_heatmap fig_geweke_dots \
       fig_lpd_estimators fig_laplacian_pivots fig_plan
# figures that need the acceptance run (placeholders until then)
ACC_REPORT := fig_cr_orderings fig_cr_convergence
ACC_EXPORT := fig_cr_per_origin fig_cr_cumulative

.PHONY: all pdf check figures tables acc-figures per-origin fill todos arxiv clean

all: figures pdf

pdf:
	./build.sh

check:
	./build.sh --check

figures:
	cd figures && for f in $(NOW); do $(PY) make_$$f.py || exit 1; done
	cd figures && for f in $(ACC_REPORT); do \
	  if [ -f "$(ACC)" ]; then $(PY) make_$$f.py --acc "$(ACC)" || exit 1; \
	  else $(PY) make_$$f.py || exit 1; fi; done
	cd figures && for f in $(ACC_EXPORT); do $(PY) make_$$f.py || exit 1; done

# table generators (tab_common.py is the shared helper, not a table)
TABLES := $(filter-out tab_common,$(basename $(notdir $(wildcard figures/tab_*.py))))

tables:
	cd figures && for f in $(TABLES); do $(PY) $$f.py || exit 1; done

acc-figures:
	@test -f "$(ACC)" || { echo "no acceptance report at $(ACC)"; exit 1; }
	cd figures && for f in $(ACC_REPORT); do $(PY) make_$$f.py --acc "$(ACC)" || exit 1; done
	cd figures && for f in $(ACC_EXPORT); do $(PY) make_$$f.py || exit 1; done

per-origin:
	$(PY) figures/make_data_cr_per_origin.py --cache "$(CACHE)" --repo "$(EXPORT_REPO)"

fill:
	python3 fill_acceptance.py "$(ACC)"

todos:
	python3 collect_todos.py

arxiv: pdf
	./make_arxiv.sh $(if $(ARXIV_TEST),--test,)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.xdv main.lof main.lot arxiv.tar.gz
