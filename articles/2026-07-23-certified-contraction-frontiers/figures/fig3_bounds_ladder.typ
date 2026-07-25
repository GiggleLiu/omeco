// F3: bounds ladder. Two panels (reg3_250, sycamore_m20).
// Horizontal stem chart: 7 rows (y = 0..6), line from x=0 to the bound value
// ending in a marker. Certified rows solid blue + circle; empirical rows
// dashed orange + square. Row labels on left panel only. Solid black vertical
// line at frontier; red dotted line at profile-bound cap.
// Data: ../data/fig/fig3_bounds.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig3_bounds.json")
#let orange = rgb("#cc6600")
#let rows = data.rows

#let panel(inst, ylabels: false) = {
  let frontier = data.frontier.at(inst)
  let cap = data.caps.at(inst)
  let xmax = frontier + 4
  let n = rows.len()
  // y ticks
  let yticks = ()
  if ylabels {
    for (i, r) in rows.enumerate() {
      yticks.push((i, [#text(size: 8pt)[#r.label]]))
    }
  }
  plot.plot(
    size: (7.2, 5.4),
    x-label: [lower bound on tc  [$log_2$ flops]],
    y-label: none,
    x-min: 0, x-max: xmax,
    y-min: -0.6, y-max: 6.6,
    y-tick-step: none, y-ticks: yticks,
    x-tick-step: 10,
    {
      // stems + markers
      for (i, r) in rows.enumerate() {
        let v = r.at(inst)
        let cert = r.certified
        plot.add(
          ((0, i), (v, i)),
          style: (stroke: (
            paint: if cert { blue } else { orange },
            thickness: 2pt,
            dash: if cert { "solid" } else { "dashed" },
          )),
          mark: none,
        )
        plot.add(
          ((v, i),),
          style: (stroke: none),
          mark: if cert { "o" } else { "square" },
          mark-size: 0.2,
          mark-style: (
            fill: if cert { blue } else { orange },
            stroke: if cert { blue } else { orange },
          ),
        )
      }
      // frontier: solid black vertical line
      plot.add(
        ((frontier, -0.6), (frontier, 6.6)),
        style: (stroke: (paint: black, thickness: 2.5pt)),
        mark: none,
      )
      // profile-bound cap: red dotted vertical line
      plot.add(
        ((cap, -0.6), (cap, 6.6)),
        style: (stroke: (dash: "dotted", paint: red, thickness: 1.8pt)),
        mark: none,
      )
      plot.annotate({
        import draw: content
        content((frontier, 6.45), [#text(size: 8pt)[frontier]],
          angle: 90deg, anchor: "east", padding: 0.08)
        content((cap, -0.45), [#text(size: 7pt, fill: red)[profile-bound cap \ (bisection width)]],
          anchor: "north-west", padding: 0.06)
      })
    },
  )
}

#align(center)[#text(size: 9pt)[solid/circle = certified,  dashed/square = high-confidence empirical]]
#v(2pt)
#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    panel("reg3_250", ylabels: true)
    content((3.6, 5.75), [*reg3_250*])
  }),
  canvas(length: 1cm, {
    import draw: content
    panel("sycamore_m20")
    content((3.6, 5.75), [*sycamore_m20*])
  }),
)
