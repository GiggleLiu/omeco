// F8: record board — per-instance record improvement over tuned TreeSA.
// Data: ../data/fig/fig8_board.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let rows = json("../data/fig/fig8_board.json").rows

// (mechanism key, legend label, mark, color)
#let mechs = (
  ("simplify-then-anneal", "simplify-then-anneal", "o", blue),
  ("composite", "composite", "diamond", purple),
  ("waist surgery", "waist surgery", "square", red),
  ("VE seed", "VE seed", "triangle", orange),
)

#let colorof(m) = {
  if m == "simplify-then-anneal" { blue }
  else if m == "composite" { purple }
  else if m == "waist surgery" { red }
  else if m == "VE seed" { orange }
  else { black }
}

// multi-line tick label from a "\n"-joined string
#let mklabel(s) = align(center, text(size: 7pt,
  s.split("\n").map(p => [#p]).join(linebreak())))

#canvas(length: 1cm, {
  import draw: content, line
  let width = 13
  let height = 5
  let xmin = -0.6
  let xmax = 7.6
  let ymin = -0.08
  let ymax = 3.2
  let px(dx) = (dx - xmin) / (xmax - xmin) * width
  let py(dy) = (dy - ymin) / (ymax - ymin) * height

  let idxrows = rows.enumerate()

  plot.plot(
    size: (width, height),
    x-label: none,
    y-label: [record improvement  $Delta$tc  [log#sub[2] flops]],
    x-min: xmin, x-max: xmax, y-min: ymin, y-max: ymax,
    y-tick-step: 0.5,
    x-tick-step: none,
    x-ticks: idxrows.map(((i, r)) => (i, mklabel(r.label))),
    legend: "inner-north-east",
    {
      // solid black reference line at y = 0
      plot.add(((xmin, 0), (xmax, 0)), style: (stroke: black + 2pt), mark: none)
      // dotted vertical stems
      for (i, r) in idxrows {
        if r.mechanism != "reference" {
          plot.add(((i, 0), (i, r.delta)),
            style: (stroke: (paint: colorof(r.mechanism), dash: "dotted", thickness: 1.3pt)),
            mark: none)
        }
      }
      // markers grouped by mechanism (one legend entry each)
      for (key, lab, mk, col) in mechs {
        let pts = idxrows.filter(((i, r)) => r.mechanism == key).map(((i, r)) => (i, r.delta))
        if pts.len() == 0 { continue }
        plot.add(pts,
          style: (stroke: none),
          mark: mk,
          mark-size: 0.24,
          mark-style: (fill: white, stroke: col + 1.6pt),
          label: lab)
      }
    },
  )

  // thick black dash markers for the reference instances
  for (i, r) in idxrows {
    if r.mechanism == "reference" {
      line((px(i - 0.2), py(0)), (px(i + 0.2), py(0)), stroke: black + 3.5pt)
    }
  }
  // reference annotation near the y = 0 line
  content((px(7.55), py(0.2)),
    align(right, text(size: 8pt)[tuned TreeSA\ (reference)]),
    anchor: "east")
})
