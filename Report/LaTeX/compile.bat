@echo off
REM Compile the main LaTeX document on Windows
REM Usage: compile.bat

REM Compile with pdflatex + bibtex + pdflatex (standard for natbib)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

REM Clean up auxiliary files (optional)
del main.aux main.bbl main.blg main.log main.out main.toc 2>nul

echo Compilation complete. Output: main.pdf
