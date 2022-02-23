SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

TEXFILES= $(shell find . -type f -name '*.tex')
FIGUREFOLDER=figures
IMAGES=$(wildcard images/**)

TIKZFOLDER=tikz
TIKZFILES= $(wildcard tikz/*.tikz)
TIKZ=$(subst $(TIKZFOLDER)/, $(FIGUREFOLDER)/tikz-, $(TIKZFILES:.tikz=.pdf))

JOBNAME=paper

all: out/$(JOBNAME).pdf

out/$(JOBNAME).pdf: $(TEXFILES) $(IMAGES) $(TIKZ) bibliography.bib
	latexmk -jobname=$(JOBNAME)

$(FIGUREFOLDER)/tikz-%.pdf: $(TIKZFOLDER)/%.tikz | $(FIGUREFOLDER)
	latexmk -cd $<
	mv $(subst $(TIKZFOLDER), $(TIKZFOLDER)/out, $(<:.tikz=.pdf)) $@

$(FIGUREFOLDER):
	mkdir -p $(FIGUREFOLDER)

bib:
	chmod +rx ./scripts/format.sh
	./scripts/format.sh --file bibliography.bib

clean:
	rm -rf out
	rm -rf $(TIKZFOLDER)/out
	rm -rf $(FIGUREFOLDER)/tikz-*.pdf

.PHONY: all clean bib