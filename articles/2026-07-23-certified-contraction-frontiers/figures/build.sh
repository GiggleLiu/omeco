#!/bin/bash
# Figure build pipeline: data rendered to JSON, plotted with Typst.
#   1. export_data.py reduces raw provenance (jsonl/csv/logs/score dirs)
#      to one plot-ready JSON per figure under ../data/fig/.
#   2. Each figN_*.typ reads only its JSON and compiles to figN_*.pdf,
#      which main.tex includes directly.
# The legacy matplotlib scripts (make_figures*.py) are kept for history
# only; the .typ sources are canonical.
set -e
cd "$(dirname "$0")"

python3 export_data.py

for t in fig*.typ; do
  out="${t%.typ}.pdf"
  echo "typst: $t -> $out"
  typst compile --root .. "$t" "$out"
  pdftoppm -png -r 150 -singlefile "$out" "${t%.typ}"
done
echo "all figures built"
