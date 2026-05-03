# PubMedQA QLoRA Assistant Black Book

This folder contains a LaTeX project black book for the `PubMedQA QLoRA Assistant` project.

## Files

- `main.tex` - complete thesis-style LaTeX report
- `references.bib` - BibTeX references

## Compile on Overleaf

1. Upload this folder to Overleaf.
2. Set `main.tex` as the main document.
3. Compile with `pdfLaTeX`.
4. If references do not appear on the first pass, recompile.

## Compile Locally

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before Submission

Replace placeholders in `main.tex`:

- `[Student Name]`
- `[Roll Number]`
- `[Department Name]`
- `[Institution Name]`
- `[University Name]`
- `[Guide/Supervisor Name]`
- `[Degree Name]`
- `[Submission Date]`

You can also replace the boxed logo placeholder on the title page with an actual institute logo.
