# Template: Quantum (quantumarticle class)

- Source: https://github.com/quantum-journal/quantum-journal (official class repo
  of Quantum — the open journal for quantum science, quantum-journal.org)
- Files: quantumarticle.cls, quantum-template.tex
- Accessed: 2026-07-23
- Journal: Quantum; article type: regular article (no hard length/figure cap;
  clarity valued over brevity per journal policy)
- Key constraints: LaTeX (pdflatex/lualatex), quantumarticle class, title +
  abstract on first page, numbered sections, bibliography via bibtex/biblatex.

## Draft-compilation note (2026-07-23)
quantumarticle.cls (GitHub master AND TeX Live 2025's copy) fails on the
2025 LaTeX kernel here ("Sorting rule for 'begindocument' hook applied too
late", cascading into DVI-mode errors). The draft compiles with the plain
article class + geometry; swap the documentclass back to quantumarticle at
submission on a toolchain with TeX Live <= 2023 (e.g., Overleaf legacy).
