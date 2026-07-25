// F7: methodology timeline (JOSS cetz-plot house style).
// Single wide panel: two post-step record curves (reg3 solid blue,
// sycamore dashed red), a dotted red certified-LB line, and per-event
// two-line annotations placed at alternating heights with thin leaders.
// Data: ../data/fig/fig7_timeline.json (rendered by figures/export_data.py).
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig7_timeline.json")
#let events = data.events
#let lb = data.certified_lb_sycamore

// plot geometry (canvas cm) and data ranges
#let W = 13.0
#let H = 5.0
#let x0 = 0.7
#let x1 = 6.4
#let y0 = 35.0
#let y1 = 80.0
#let cx(x) = (x - x0) / (x1 - x0) * W
#let cy(y) = (y - y0) / (y1 - y0) * H

// build post-step polyline from (x, y) points
#let stepify(pts) = {
  let out = ()
  for (i, p) in pts.enumerate() {
    if i == 0 { out.push(p) } else {
      out.push((p.at(0), pts.at(i - 1).at(1)))
      out.push(p)
    }
  }
  out
}

// render a "\n"-containing string as multi-line content
#let ml(s) = s.split("\n").map(p => [#p]).join(linebreak())

// alternating annotation heights (data-y), first event sits high above its point
#let note-y = (77.0, 66.0, 71.5, 66.0, 71.5, 66.0, 71.5)
#let note-anchor = ("west", "center", "center", "center", "center", "center", "east")

#canvas(length: 1cm, {
  import draw: content, line
  plot.plot(
    size: (W, H),
    x-label: [research-loop cycle],
    y-label: [best verified tc  [$log_2$ flops]],
    x-min: x0, x-max: x1,
    y-min: y0, y-max: y1,
    x-tick-step: none,
    x-ticks: (1, 2, 3, 4, 5, 6).map(v => (v, str(v))),
    y-tick-step: 5,
    legend: "inner-east",
    legend-style: (stroke: none, fill: white.transparentize(10%)),
    {
      // dotted red certified-LB line
      plot.add(
        ((x0, lb), (x1, lb)),
        style: (stroke: (dash: "dotted", paint: red, thickness: 1.4pt)),
        mark: none,
      )
      // reg3 record (solid blue step)
      plot.add(
        stepify(events.map(e => (e.cycle, e.reg3))),
        style: (stroke: (paint: blue, thickness: 2.2pt)),
        mark: none,
        label: [reg3_250 best tc],
      )
      // sycamore record (dashed red step)
      plot.add(
        stepify(events.map(e => (e.cycle, e.syc))),
        style: (stroke: (dash: "dashed", paint: red, thickness: 2.2pt)),
        mark: none,
        label: [sycamore_m20 best tc],
      )
    },
  )
  // certified-LB label (above the dotted line, left band — keeps clear of the legend)
  content((cx(1.05), cy(lb) + 0.28), text(size: 8pt, fill: red)[certified LB (sycamore)], anchor: "west")
  // per-event annotations with thin gray leaders
  for (i, e) in events.enumerate() {
    let px = cx(e.cycle)
    let py = cy(e.syc)
    let ny = cy(note-y.at(i))
    line((px, py + 0.05), (px, ny - 0.42), stroke: gray + 0.4pt)
    content((px, ny), text(size: 6.5pt, ml(e.note)), anchor: note-anchor.at(i))
  }
})
