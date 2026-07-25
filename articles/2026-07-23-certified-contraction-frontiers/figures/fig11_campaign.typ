// F11: matched-budget same-machine campaign (JOSS-style cetz-plot).
// (a) per-instance 15-rep distributions, centered on the tuned-TreeSA median.
// (b) surface-code family, both methods vs distance, anchored on surgery median.
// Data: ../data/fig/fig11_campaign.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig11_campaign.json")

// multi-line tick label from a "line1\nline2" string
#let ml(s) = s.split("\n").map(p => box[#text(size: 7pt)[#p]]).join(linebreak())

// ---------------------------------------------------------------- panel (a)
#let dists = data.distributions
#let ticks-a = dists.enumerate().map(p => (p.at(0), ml(p.at(1).label)))
#let ref-a = {
  let a = ()
  for (i, d) in dists.enumerate() {
    for (k, v) in d.ref.enumerate() { a.push((i - 0.16 + 0.02 * k, v)) }
  }
  a
}
#let best-a = {
  let a = ()
  for (i, d) in dists.enumerate() {
    for (k, v) in d.best.enumerate() { a.push((i + 0.16 - 0.02 * k, v)) }
  }
  a
}

// ---------------------------------------------------------------- panel (b)
#let fam = data.family
#let ticks-b = fam.enumerate().map(p => (p.at(0), [d=#p.at(1).d]))
#let ref-b = {
  let a = ()
  for (j, e) in fam.enumerate() { for v in e.ref { a.push((j - 0.1, v)) } }
  a
}
#let surg-b = {
  let a = ()
  for (j, e) in fam.enumerate() { for v in e.surgery { a.push((j + 0.1, v)) } }
  a
}

#grid(columns: 2, gutter: 18pt,
  canvas(length: 1cm, {
    plot.plot(
      size: (7, 5.2),
      x-label: none,
      y-label: [tc $-$ tuned TreeSA median  [log#sub[2]]],
      x-min: -0.5, x-max: 5.5, y-min: -10.5, y-max: 9.5,
      x-tick-step: none, x-ticks: ticks-a,
      y-tick-step: 2.5,
      legend: "inner-north-west",
      legend-style: (fill: white, stroke: none, spacing: 0.15, item: (spacing: 0.1)),
      {
        plot.add(((-0.5, 0), (5.5, 0)), mark: none,
          style: (stroke: (dash: "dotted", paint: black, thickness: 1pt)))
        plot.add(ref-a, style: (stroke: none), mark: "o", mark-size: 0.13,
          mark-style: (fill: white, stroke: black + 1pt),
          label: "tuned TreeSA (15 reps)")
        plot.add(best-a, style: (stroke: none), mark: "square", mark-size: 0.13,
          mark-style: (fill: white, stroke: red + 1pt),
          label: "best attempt (15 reps)")
        plot.annotate({ draw.content((-0.32, -9.6), [(a)]) })
      },
    )
  }),
  canvas(length: 1cm, {
    plot.plot(
      size: (7, 5.2),
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
          draw.content((0.55, 1.31),
            text(size: 7.5pt)[circles: tuned TreeSA\ squares: waist surgery])
          draw.content((-0.32, -0.22), [(b)])
        })
      },
    )
  }),
)
