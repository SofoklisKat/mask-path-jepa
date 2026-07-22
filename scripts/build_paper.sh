#!/usr/bin/env bash
# Build paper PDF when a LaTeX engine is available; always build results.pdf from checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Building results.pdf from checkpoints..."
PYTHONPATH=. python scripts/build_results_pdf.py

if command -v pdflatex >/dev/null 2>&1; then
  echo "==> Compiling paper/main.tex..."
  cd paper
  pdflatex -interaction=nonstopmode main.tex
  bibtex main || true
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  echo "==> Wrote paper/main.pdf"
else
  echo "==> pdflatex not found; skipped paper/main.pdf"
  echo "    Install: sudo apt install texlive-latex-recommended texlive-latex-extra"
  echo "    Results report: paper/results.pdf"
fi
