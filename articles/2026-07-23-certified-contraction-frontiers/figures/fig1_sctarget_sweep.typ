// F1: sc_target sweep. Two panels (reg3_250, sycamore_m20).
// x = ordered sc_target categories (last = inf, rendered as infinity),
// y = best tc at 90 s. Median filled blue points with min-max whiskers,
// dashed black frontier line, dotted red "natural sc" line.
// Data: ../data/fig/fig1_sweep.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig1_sweep.json")

#let panel(inst, y-min, y-max, ylabel: none, natlabel-y: 0) = {
  let rows = data.instances.at(inst)
  let n = rows.len()
  let frontier = data.frontier.at(inst)
  let natural = data.natural.at(inst)
  // x category ticks
  let xticks = ()
  let nat-idx = 0
  for (i, r) in rows.enumerate() {
    let lab = if r.sc_target == "inf" { $infinity$ } else { [#r.sc_target] }
    xticks.push((i, lab))
    if r.sc_target == natural { nat-idx = i }
  }
  plot.plot(
    size: (8, 6.2),
    x-label: [$"sc"_"target"$  [$log_2$ memory]],
    y-label: ylabel,
    x-min: -0.6, x-max: n - 0.4,
    y-min: y-min, y-max: y-max,
    x-tick-step: none, x-ticks: xticks,
    y-grid: false,
    {
      // frontier dashed line
      plot.add(
        ((-0.6, frontier), (n - 0.4, frontier)),
        style: (stroke: (dash: "dashed", paint: black, thickness: 1.2pt)),
        mark: none,
      )
      // natural sc dotted vertical line
      plot.add(
        ((nat-idx, y-min), (nat-idx, y-max)),
        style: (stroke: (dash: "dotted", paint: red, thickness: 1.4pt)),
        mark: none,
      )
      // whiskers (min-max range)
      for (i, r) in rows.enumerate() {
        plot.add(
          ((i, r.lo), (i, r.hi)),
          style: (stroke: (paint: blue, thickness: 1.2pt)),
          mark: none,
        )
      }
      // median points
      plot.add(
        rows.enumerate().map(((i, r)) => (i, r.median)),
        style: (stroke: none),
        mark: "o", mark-size: 0.18,
        mark-style: (fill: blue, stroke: blue),
      )
      // annotations in data coordinates
      plot.annotate({
        import draw: content
        content((0.0, frontier + (y-max - y-min) * 0.018), [frontier],
          anchor: "west", padding: 0.05)
        content((nat-idx + 0.15, y-max), [#text(fill: red)[natural sc]],
          anchor: "north-west", padding: 0.05)
        content((nat-idx * 0 + 0.7, natlabel-y), [median of 3 (range)],
          anchor: "west", padding: 0.05)
      })
    },
  )
}

#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    panel("reg3_250", 39.3, 48.6, ylabel: [best tc at 90 s  [$log_2$ flops]],
      natlabel-y: 47.85)
    content((4, 6.6), [*reg3_250*])
  }),
  canvas(length: 1cm, {
    import draw: content
    panel("sycamore_m20", 60.8, 77.8, natlabel-y: 69.2)
    content((4, 6.6), [*sycamore_m20*])
  }),
)
