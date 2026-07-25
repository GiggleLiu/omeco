// F9: simplification — (a) free-contraction shrink fractions, (b) Sycamore record march.
// Data: ../data/fig/fig9_simplify.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig9_simplify.json")
#let shrink = data.shrink
#let march = data.march

#let mklabel(s) = align(center, text(size: 8pt,
  s.split("\n").map(p => [#p]).join(linebreak())))

#let bymap(s) = {
  if s == "ref:treesa-inf" { "tuned TreeSA" }
  else { s.replace("attempt-", "attempt ") }
}

// format with one decimal place
#let fmt1(x) = {
  let r = calc.round(x * 10)
  str(calc.div-euclid(r, 10)) + "." + str(calc.rem(r, 10))
}

#grid(columns: 2, gutter: 18pt,
  // ---- panel (a): bar chart of shrink fractions ----
  canvas(length: 1cm, {
    import draw: content
    let width = 7
    let height = 6
    let xmin = -0.6
    let xmax = 5.6
    let ymin = 0
    let ymax = 1.02
    let px(dx) = (dx - xmin) / (xmax - xmin) * width
    let py(dy) = (dy - ymin) / (ymax - ymin) * height
    let w = 0.38

    plot.plot(
      size: (width, height),
      x-label: none,
      y-label: [tensors removed by free contractions  [fraction]],
      x-min: xmin, x-max: xmax, y-min: ymin, y-max: ymax,
      y-tick-step: 0.2,
      x-tick-step: none,
      x-ticks: shrink.enumerate().map(((i, r)) => (i, mklabel(r.label))),
      {
        for (i, r) in shrink.enumerate() {
          plot.add(
            ((i - w, 0), (i - w, r.frac), (i + w, r.frac), (i + w, 0)),
            style: (stroke: blue + 1.3pt),
            mark: none,
          )
        }
      },
    )
    // value labels above bars
    for (i, r) in shrink.enumerate() {
      content((px(i), py(r.frac + 0.035)),
        text(size: 8pt)[#{ str(calc.round(r.frac * 100)) }%])
    }
    content((px(5.3), py(0.97)), text(size: 11pt)[(a)])
  }),

  // ---- panel (b): Sycamore record march ----
  canvas(length: 1cm, {
    import draw: content
    let width = 7
    let height = 6
    let xmin = -0.35
    let xmax = 3.72
    let ymin = 59.4
    let ymax = 61.15
    let px(dx) = (dx - xmin) / (xmax - xmin) * width
    let py(dy) = (dy - ymin) / (ymax - ymin) * height
    let ticks = ("reference", "cycle 8", "cycle 9a", "cycle 9b")

    plot.plot(
      size: (width, height),
      x-label: none,
      y-label: [tc  [log#sub[2] flops]],
      x-min: xmin, x-max: xmax, y-min: ymin, y-max: ymax,
      y-tick-step: 0.25,
      x-tick-step: none,
      x-ticks: ticks.enumerate().map(((i, t)) => (i, text(size: 9pt)[#t])),
      {
        plot.add(
          march.enumerate().map(((i, r)) => (i, r.tc)),
          style: (stroke: blue + 2pt),
          mark: "o",
          mark-size: 0.22,
          mark-style: (fill: white, stroke: blue + 1.6pt),
        )
      },
    )
    // per-point "by" annotations
    for (i, r) in march.enumerate() {
      content((px(i + 0.07), py(r.tc + 0.12)),
        text(size: 8pt)[#bymap(r.by)], anchor: "west")
    }
    // off-scale red annotation
    content((px(0.0), py(61.05)),
      text(size: 8pt, fill: red)[same anneal without simplification: #fmt1(data.ab_without_simplify) (off scale #sym.arrow.t)],
      anchor: "west")
    // boxed instance label
    content((px(1.5), py(59.72)), text(size: 9pt)[Sycamore 53q, m=20],
      frame: "rect", padding: 0.12, stroke: black + 0.6pt, fill: white)
    content((px(3.35), py(59.55)), text(size: 11pt)[(b)])
  }),
)
