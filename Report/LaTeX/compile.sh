#!/bin/bash

# Compile the main LaTeX document
# Usage: bash compile.sh

# Change to the script directory
cd "$(dirname "$0")"

# Compile with pdflatex + bibtex + pdflatex x3 (standard for natbib/hyperref)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex

# Clean up auxiliary files (optional)
rm -f main.aux main.bbl main.blg main.log main.out main.toc

echo "Compilation complete. Output: main.pdf"
