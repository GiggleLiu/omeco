// F14: advantage summary — (a) per-instance record board (confirmed
// improvement over the tuned TreeSA reference), (b) surface-code family:
// the surgery advantage grows with code distance.
// Data: ../data/fig/fig8_board.json + ../data/fig/fig11_campaign.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let rows = json("../data/fig/fig8_board.json").rows
#let fam = json("../data/fig/fig11_campaign.json").family

#let mechs = (
  ("simplify-then-anneal", "simplify-then-anneal", "o", blue),
  ("waist surgery", "waist surgery", "square", red),
  ("VE seed", "VE seed", "triangle", orange),
)

#let colorof(m) = {
  if m == "simplify-then-anneal" { blue }
  else if m == "waist surgery" { red }
  else if m == "VE seed" { orange }
  else { black }
}

#let mklabel(s) = align(center, text(size: 6pt,
  s.split("\n").map(p => [#p]).join(linebreak())))

#grid(columns: 2, gutter: 16pt,
  // ------------------------------------------------------- (a) record board
  canvas(length: 1cm, {
    import draw: content, line
    let width = 9.5
    let height = 5.2
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
        plot.add(((xmin, 0), (xmax, 0)), style: (stroke: black + 2pt), mark: none)
        for (i, r) in idxrows {
          if r.mechanism != "reference" {
            plot.add(((i, 0), (i, r.delta)),
              style: (stroke: (paint: colorof(r.mechanism), dash: "dotted", thickness: 1.3pt)),
              mark: none)
          }
        }
        for (key, lab, mk, col) in mechs {
          let pts = idxrows.filter(((i, r)) => r.mechanism == key).map(((i, r)) => (i, r.delta))
          if pts.len() == 0 { continue }
          plot.add(pts,
            style: (stroke: none),
            mark: mk,
            mark-size: 0.22,
            mark-style: (fill: white, stroke: col + 1.6pt),
            label: lab)
        }
      },
    )
    for (i, r) in idxrows {
      if r.mechanism == "reference" {
        line((px(i - 0.2), py(0)), (px(i + 0.2), py(0)), stroke: black + 3.5pt)
      }
    }
    content((px(7.5), py(0.22)),
      align(right, text(size: 8pt)[tuned TreeSA\ (reference)]),
      anchor: "east")
    content((px(-0.35), py(2.98)), [(a)])
  }),
  // ------------------------------------------- (b) surface-code family trend
  canvas(length: 1cm, {
    import draw: content
    let ticks-b = fam.enumerate().map(p => (p.at(0), [d=#p.at(1).d]))
    let ref-b = {
      let a = ()
      for (j, e) in fam.enumerate() { for v in e.ref { a.push((j - 0.1, v)) } }
      a
    }
    let surg-b = {
      let a = ()
      for (j, e) in fam.enumerate() { for v in e.surgery { a.push((j + 0.1, v)) } }
      a
    }
    plot.plot(
      size: (4.8, 5.2),
      x-label: [surface-code distance],
      y-label: [tc $-$ surgery median  [log#sub[2]]],
      x-min: -0.5, x-max: 3.5, y-min: -0.3, y-max: 1.5,
      x-tick-step: none, x-ticks: ticks-b,
      y-tick-step: 0.2,
      {
        plot.add(((-0.5, 0), (3.5, 0)), mark: none,
          style: (stroke: (dash: "dotted", paint: black, thickness: 1pt)))
        plot.add(ref-b, style: (stroke: none), mark: "o", mark-size: 0.14,
          mark-style: (fill: white, stroke: black + 1pt))
        plot.add(surg-b, style: (stroke: none), mark: "square", mark-size: 0.14,
          mark-style: (fill: white, stroke: red + 1pt))
        plot.annotate({
          draw.content((1.05, 1.36),
            text(size: 7.5pt)[circles: tuned TreeSA\ squares: waist surgery])
          draw.content((-0.28, -0.2), [(b)])
        })
      },
    )
  }),
)
