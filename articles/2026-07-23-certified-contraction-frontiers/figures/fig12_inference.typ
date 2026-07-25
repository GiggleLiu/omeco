// F12: UAI-2014 inference batch, tc above the per-instance frontier.
// Two regimes (dense DBN vs pedigree linkage) shown as shaded background bands.
// Data: ../data/fig/fig12_inference.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig12_inference.json")
#let rows = data.rows

// multi-line tick label from a "line1\nline2" string
#let ml(s) = s.split("\n").map(p => box[#text(size: 8pt)[#p]]).join(linebreak())
#let ticks = rows.enumerate().map(p => (p.at(0), ml(p.at(1).label)))

#let series(key) = rows.enumerate().map(p => (p.at(0), p.at(1).at(key)))

#canvas(length: 1cm, {
  plot.plot(
    size: (12, 5),
    x-label: none,
    y-label: [tc above frontier  [log#sub[2] flops]],
    x-min: -0.5, x-max: 9.5, y-min: -1.2, y-max: 36,
    x-tick-step: none, x-ticks: ticks,
    y-tick-step: 5,
    legend: "inner-north-east",
    legend-style: (fill: white, stroke: black + 0.6pt, padding: 0.15, spacing: 0.14, item: (spacing: 0.08)),
    {
      // shaded regime bands (background -> behind the data)
      plot.annotate(background: true, {
        draw.rect((-0.5, -1.2), (2.5, 36), fill: luma(237), stroke: none)
        draw.rect((2.5, -1.2), (6.5, 36), fill: luma(251), stroke: none)
      })
      // thick per-instance frontier line at 0
      plot.add(((-0.5, 0), (9.5, 0)), mark: none, style: (stroke: black + 2pt))
      // series
      plot.add(series("default"), style: (stroke: none), mark: "triangle",
        mark-size: 0.18, mark-style: (fill: black, stroke: black + 1pt),
        label: text(size: 7pt)[TensorInference default (GreedyMethod)])
      plot.add(series("treesa"), style: (stroke: none), mark: "square",
        mark-size: 0.15, mark-style: (fill: white, stroke: rgb("#0033cc") + 2pt),
        label: text(size: 7pt)[best tuned TreeSA (Julia ladder)])
      plot.add(series("elim"), style: (stroke: none), mark: "triangle",
        mark-size: 0.18, mark-style: (fill: white, stroke: rgb("#cc6600") + 2pt),
        label: text(size: 7pt)[best elimination (HyperND / Treewidth-MF)])
      plot.add(series("ours"), style: (stroke: none), mark: "x",
        mark-size: 0.22, mark-style: (stroke: rgb("#cc0000") + 2.4pt),
        label: text(size: 7pt)[this work (best method, median of 5)])
      // foreground text: regime labels + frontier annotation
      plot.annotate({
        draw.content((1.0, 33.5), align(center, text(size: 8pt)[dense DBN:\ elimination wins]))
        draw.content((4.5, 33.5), align(center, text(size: 8pt)[pedigree linkage:\ annealing wins]))
        draw.content((6.65, 1.5), anchor: "west", text(size: 8pt)[per-instance frontier])
      })
    },
  )
})
