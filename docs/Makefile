# Minimal Makefile for the BetterRobot Sphinx docs.
#
# Usage:
#   make html          # build the HTML site → _build/html/
#   make clean         # remove the build directory
#   make linkcheck     # validate external links
#   make strict        # build with warnings-as-errors (nitpicky)
#   make serve         # run a local server at http://localhost:8000

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean linkcheck strict serve

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" reference/api

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

strict:
	BR_DOCS_NITPICKY=1 $(SPHINXBUILD) -W --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

serve: html
	@echo "Serving docs at http://localhost:8000  (Ctrl-C to stop)"
	@cd "$(BUILDDIR)/html" && python -m http.server 8000
