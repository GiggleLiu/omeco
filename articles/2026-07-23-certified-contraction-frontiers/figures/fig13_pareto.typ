// F13: JOSS-style time-vs-quality Pareto scatter (mimics the presentation
// of the OMEinsumContractionOrders benchmark figures: cetz-plot scatter,
// log10 wall time vs log2 flops, dashed Pareto front).
// Data: ../data/fig/fig13_pareto.json (rendered by figures/export_data.py).
#import "@preview/cetz:0.4.0" as cetz: canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig13_pareto.json")

// (family key, legend label, mark, color, filled?)
#let fams = (
  ("julia-greedy", "GreedyMethod (Julia)", "o", red, true),
  ("treesa", "TreeSA (all configs, both impls.)", "o", purple, true),
  ("hypernd", "HyperND (Julia)", "square", blue, true),
  ("treewidth", "Treewidth (Julia, 9 back-ends)", "triangle", orange, true),
  ("cotengra", "cotengra hyper+SA", "+", black, true),
  ("ours-simplify", "simplify+anneal (this work)", "triangle", blue, false),
  ("ours-surgery", "waist surgery (this work)", "x", red, true),
)

#let panel(inst, x-min, x-max, legend: none) = {
  let entry = data.instances.at(inst)
  plot.plot(
    size: (8, 7),
    x-label: [Log2 FLOPs (Time Complexity)],
    y-label: [Log10 Wall Clock Time (seconds)],
    x-min: x-min, x-max: x-max, y-min: -2, y-max: 3.6,
    legend: legend,
    {
      // dashed Pareto front through the non-dominated points
      plot.add(
        entry.front.filter(p => p.tc >= x-min and p.tc <= x-max)
          .map(p => (p.tc, calc.log(p.t, base: 10))),
        style: (stroke: (dash: "dashed", paint: black, thickness: 1pt)),
        mark: none,
      )
      for (fam, lab, mark, color, filled) in fams {
        // clip off-scale orderings (BFS/MCS-style treewidth, tc up to 443)
        let pts = entry.points.filter(p =>
          p.family == fam and p.tc >= x-min and p.tc <= x-max)
        if pts.len() == 0 { continue }
        plot.add(
          pts.map(p => (p.tc, calc.log(calc.max(p.t, 0.02), base: 10))),
          style: (stroke: none),
          mark: mark,
          mark-size: if mark == "x" { 0.24 } else { 0.16 },
          mark-style: (
            fill: if filled { color } else { white },
            stroke: color + (if mark == "x" { 1.8pt } else { 1.2pt }),
          ),
          label: lab,
        )
      }
    },
  )
}


// Inset zoom of the high-quality wall: same series, restricted ranges,
// drawn as a small plot translated inside the main canvas. (cx, cy) is the
// inset's lower-left corner in canvas cm; the dashed rectangle marks the
// zoomed window on the main axes.
#let wall-inset(inst, x-min, x-max, zx-min, zx-max, cx, cy, w, h,
                show-title: true) = {
  import cetz.draw: group, translate, rect, content
  let entry = data.instances.at(inst)
  // zoom window on the main axes (y in [0.6, 3.2] = roughly 4 s .. 1600 s)
  let px(v) = (v - x-min) / (x-max - x-min) * 8
  let py(v) = (v + 2) / 5.6 * 7
  rect((px(zx-min), py(0.6)), (px(zx-max), py(3.2)),
       stroke: (paint: gray, dash: "dotted", thickness: 0.8pt))
  // opaque backing so main-panel points do not bleed through the inset
  rect((cx - 0.8, cy - 0.6),
       (cx + w + 0.25, cy + h + (if show-title { 0.45 } else { 0.15 })),
       fill: white, stroke: none)
  if show-title {
    content((cx + w / 2, cy + h + 0.22), text(size: 7pt)[wall region (zoom)])
  }
  group({
    translate((cx, cy))
    plot.plot(
      size: (w, h),
      x-min: zx-min, x-max: zx-max, y-min: 0.6, y-max: 3.2,
      x-tick-step: 1, y-tick-step: 1,
      x-label: none, y-label: none,
      legend: none,
      {
        for (fam, lab, mark, color, filled) in fams {
          let pts = entry.points.filter(p =>
            p.family == fam and p.tc >= zx-min and p.tc <= zx-max)
          if pts.len() == 0 { continue }
          plot.add(
            pts.map(p => (p.tc, calc.log(calc.max(p.t, 0.02), base: 10))),
            style: (stroke: none),
            mark: mark,
            mark-size: if mark == "x" { 0.18 } else { 0.12 },
            mark-style: (
              fill: if filled { color } else { white },
              stroke: color + (if mark == "x" { 1.4pt } else { 1.0pt }),
            ),
          )
        }
      },
    )
  })
}

#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    panel("sycamore_53_20_0", 56, 110, legend: "inner-north-east")
    content((4, 7.45), [*Sycamore 53q, m=20 (3369 tensors)*])
    content((1.6, 2.95), [Pareto Front], align: center,
            fill: white.transparentize(50%), frame: "rect", padding: 0.1,
            stroke: none)
    wall-inset("sycamore_53_20_0", 56, 110, 59.5, 63.0, 0.95, 0.5, 2.9, 1.85,
                show-title: false)
  }),
  canvas(length: 1cm, {
    import draw: content
    panel("surfacecode_d21", 46, 84)
    content((4, 7.45), [*surface code d=21 (2203 tensors)*])
    content((2.6, 4.0), [Pareto Front], align: center,
            fill: white.transparentize(50%), frame: "rect", padding: 0.1,
            stroke: none)
    wall-inset("surfacecode_d21", 46, 84, 47.1, 49.4, 4.45, 0.75, 3.1, 2.2)
  }),
)
