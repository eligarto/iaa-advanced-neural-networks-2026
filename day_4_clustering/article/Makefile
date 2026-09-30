# Companion workbook: build rules
#
#   make            vendor the deck figures, then build workbook.pdf
#   make figures    vendor figures only (deck PNGs, GIF filmstrips, QR codes)
#   make pdf        LaTeX only (assumes figures/ is already populated)
#   make clean      drop LaTeX intermediates
#   make distclean  also drop workbook.pdf and the vendored figures
#
# The deck lives in the website repo; point DECK_ASSETS elsewhere to
# vendor from a different checkout or a copy of the deck's asset directory.

DECK_ASSETS ?= $(shell if [ -d ../garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026 ]; then \
    echo ../garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026; \
  elif [ -d /home/rd24/git/garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026 ]; then \
    echo /home/rd24/git/garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026; \
  else echo ""; fi)

LATEXMK  ?= latexmk
MAIN      = workbook
# sync_figures.py needs Pillow (filmstrips, DPI normalisation) and segno (QR
# codes); neither is a dependency of the analysis package, so they are pulled in
# per-invocation rather than added to pyproject.toml.
PY       ?= uv run --with pillow --with segno python

.PHONY: all figures pdf clean distclean check

all: pdf

figures:
	$(PY) scripts/sync_figures.py --deck-assets "$(DECK_ASSETS)" --out figures

pdf: check workbook.tex chapters/*.tex references.bib ar-1col.cls
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -use-make $(MAIN).tex
	@echo "---" && pdfinfo $(MAIN).pdf | grep -E "Pages|File size"

check:
	uv run python scripts/check_workbook.py

clean:
	$(LATEXMK) -c $(MAIN).tex || true
	# -c keeps latexmk's own database and the .fls file; a stale database makes
	# the next build run bibtex against an empty .aux, so drop them too.
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).run.xml $(MAIN).out $(MAIN).fdb_latexmk $(MAIN).fls

distclean: clean
	rm -f $(MAIN).pdf
	rm -rf figures
