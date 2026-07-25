// F11 (appendix): per-instance 15-rep distributions at 90 s, centered on
// the tuned-TreeSA median. The family panel lives in fig14_advantage.
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

#canvas(length: 1cm, {
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
      },
    )
  })
