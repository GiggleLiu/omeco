// F2: frontier convergence. Two panels (reg3_250, sycamore_m20).
// x = tc - frontier (per instance); all points at y=1, y-axis hidden.
// 13 attempts as open violet triangles; reference rows as distinct filled
// markers; dashed blue vertical line at 0. Legend only on left panel.
// Data: ../data/fig/fig2_frontier.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig2_frontier.json")
#let violet = rgb("#660099")

// (label, key, color, mark, filled?)
#let refs = (
  ("TreeSA default sc_target=20", "TreeSA default sc_target=20", black, "x", true),
  ("cotengra hyper", "cotengra hyper", red, "square", true),
  ("cotengra SA", "cotengra SA", red, "triangle", true),
  ("TreeSA sc_target=∞ (frontier ref)", "TreeSA sc_target=inf (frontier ref)", blue, "o", true),
)

#let panel(inst, legend: none) = {
  let frontier = data.frontier.at(inst)
  let py = 0.55
  let attempts = data.attempts.at(inst).map(v => (v - frontier, py))
  plot.plot(
    size: (8, 5.4),
    x-label: [$"tc" - "tc"_"frontier"$  [$log_2$ flops]],
    y-label: none,
    x-min: -0.6, x-max: 12,
    y-min: 0, y-max: 2,
    y-tick-step: none, x-tick-step: 2,
    legend: legend, legend-style: (stroke: none, padding: 0.15),
    {
      // dashed blue vertical line at x = 0
      plot.add(
        ((0, 0), (0, 2)),
        style: (stroke: (dash: "dashed", paint: blue, thickness: 1.8pt)),
        mark: none,
      )
      // 13 search mechanisms: open violet triangles
      plot.add(
        attempts,
        style: (stroke: none),
        mark: "triangle", mark-size: 0.24,
        mark-style: (fill: white, stroke: violet + 1.4pt),
        label: if legend != none { [13 search mechanisms (this work)] } else { none },
      )
      // reference rows
      for (lab, key, color, mark, filled) in refs {
        let entry = data.references.find(r => r.label == key)
        plot.add(
          ((entry.at(inst) - frontier, py),),
          style: (stroke: none),
          mark: mark, mark-size: if mark == "x" { 0.28 } else { 0.22 },
          mark-style: (
            fill: if filled { color } else { white },
            stroke: color + (if mark == "x" { 2pt } else { 1pt }),
          ),
          label: if legend != none { [#lab] } else { none },
        )
      }
    },
  )
}

#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    panel("reg3_250", legend: "inner-north-west")
    content((4, 5.75), [*reg3_250*])
  }),
  canvas(length: 1cm, {
    import draw: content
    panel("sycamore_m20")
    content((4, 5.75), [*sycamore_m20*])
  }),
)
