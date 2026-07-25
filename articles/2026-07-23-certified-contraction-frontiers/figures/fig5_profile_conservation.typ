// F5: profile-conservation grouped bar chart (JOSS cetz-plot house style).
// Two panels; three optimizers per panel; two bars each ("within 1.0" solid
// blue fill, "within 2.0" white fill + orange stroke). Peak/tc label above
// each group. Axes come from an (invisible) plot; bars/labels are drawn in
// canvas coordinates on top of the plot area (which maps to (0,0)-(8,6)).
// Data: ../data/fig/fig5_conservation.json (rendered by figures/export_data.py).
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig5_conservation.json")

// pretty labels for the three optimizers (in file order)
#let nice = (
  "TreeSA-inf": [TreeSA-$oo$],
  "plain-tc SA": [plain-tc SA],
  "profile-aware SA": [profile-aware SA],
)

#let W = 8      // plot area width  (canvas cm)
#let H = 6      // plot area height (canvas cm)
#let bw = 0.38  // half bar width in data units

// data x-range so three groups sit at x=0,1,2 with padding
#let xmin = -0.6

// draw the plot frame + ticks only (transparent data anchors)
#let frame(inst, y-max, yticks, show-ylabel: false) = {
  let rows = data.at(inst)
  let xmax = rows.len() - 0.4
  plot.plot(
    size: (W, H),
    x-label: [],
    y-label: if show-ylabel [count of contractions] else [],
    x-min: xmin, x-max: xmax,
    y-min: 0, y-max: y-max,
    x-tick-step: none,
    x-ticks: rows.enumerate().map(((i, r)) => (i, nice.at(r.label))),
    y-tick-step: none,
    y-ticks: yticks.map(v => (v, str(v))),
    axis-style: "scientific",
    {
      plot.add(((xmin, 0), (xmax, 0)), style: (stroke: none), mark: none)
    },
  )
}

// map data coords -> canvas coords within the plot area
#let cx(inst, x) = {
  let xmax = data.at(inst).len() - 0.4
  (x - xmin) / (xmax - xmin) * W
}
#let cy(y, y-max) = y / y-max * H

#let bars(inst, y-max) = {
  import draw: rect, content
  let rows = data.at(inst)
  for (i, r) in rows.enumerate() {
    // solid blue: within 1.0
    rect((cx(inst, i - bw), 0), (cx(inst, i), cy(r.near1, y-max)),
         fill: blue, stroke: blue)
    // white fill + orange stroke: within 2.0
    rect((cx(inst, i), 0), (cx(inst, i + bw), cy(r.near2, y-max)),
         fill: white, stroke: orange + 1.2pt)
    // peak/tc note above the taller bar
    let top = calc.max(r.near1, r.near2)
    content((cx(inst, i), cy(top, y-max) + 0.42),
            text(size: 8pt)[peak #r.peak \ tc #calc.round(r.tc, digits: 2)],
            anchor: "center")
  }
}

// manual legend swatches (top-left of first panel)
#let legend = {
  import draw: rect, content
  rect((0.3, 5.5), (0.75, 5.82), fill: blue, stroke: blue)
  content((0.85, 5.66), text(size: 8pt)[nodes within 1.0 of max], anchor: "west")
  rect((0.3, 5.0), (0.75, 5.32), fill: white, stroke: orange + 1.2pt)
  content((0.85, 5.16), text(size: 8pt)[nodes within 2.0 of max], anchor: "west")
}

#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    frame("reg3_250", 35, (0, 5, 10, 15, 20, 25, 30, 35), show-ylabel: true)
    bars("reg3_250", 35)
    legend
    content((4, 6.5), [*reg3_250*])
  }),
  canvas(length: 1cm, {
    import draw: content
    frame("sycamore_m20", 88, (0, 20, 40, 60, 80))
    bars("sycamore_m20", 88)
    content((4, 6.5), [*sycamore_m20*])
  }),
)
