# Template: SciPost Physics (SciPost.cls)

- Source: https://scipost.org/media/uploads/templates/SciPostPhys/2024/SciPostPhys_LaTeX_Template_9RNCEYb.tgz
  (linked from the official author guidelines at https://scipost.org/SciPostPhys/authoring)
- Files: SciPost.cls, SciPost_bibstyle.bst, SciPostPhys.tex (example),
  SciPost_Example_BiBTeX_File.bib — extracted under template/scipost/;
  SciPost.cls and SciPost_bibstyle.bst are copied next to main.tex.
- Template version: 2024-07. Accessed: 2026-07-25.
- Journal: SciPost Physics; article type: regular submission.
- Key constraints: bold abstract that should fit in 8 lines; table of
  contents required for papers over 6 pages; centered title/author/
  affiliation blocks (numbered affiliations, starred corresponding-author
  email); copyright placeholder block after the abstract; \linenumbers on
  for refereeing; a Conclusion section is mandatory; Acknowledgements
  immediately after it, then Funding information (required — grant numbers,
  Fundref-linkable); appendices inside \begin{appendix}; references must
  carry DOI links (SciPost_bibstyle does this from the doi field) and
  preprints an arXiv link via the eprint field. No strict length limit,
  but content should be the minimum needed for reproducibility.
- Acceptance expectations (SciPost Physics, at least one must be met): a
  groundbreaking discovery / a breakthrough on a previously-identified
  stumbling block / a novel synergetic link between fields / opening a
  new research pathway. This paper argues the stumbling-block route (the
  late-schedule freeze of annealing tree refiners).
- Compiles cleanly on TeX Live 2025 with pdflatex + bibtex (unlike the
  previously targeted quantumarticle.cls — see git history for that
  README).

## Previous target (superseded 2026-07-25)

The draft originally targeted Quantum (quantumarticle.cls), which is
incompatible with the 2025 LaTeX kernel on this machine. The class files
remain in template/ for reference; the manuscript now uses SciPost.cls.
