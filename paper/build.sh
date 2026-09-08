#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker run --rm -v "$(pwd)":/work -w /work texlive/texlive:latest bash -c '
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
'
echo "Built: $(pwd)/main.pdf"
