// One shared K4 example for structural simplification and waist surgery.
// Standalone Typst 0.14+ source; no external packages.

#set page(width: 178mm, height: 98mm, margin: 0mm, fill: white)
#set text(font: "Avenir", size: 7.1pt, fill: rgb("#172033"))

#let ink = rgb("#172033")
#let muted = rgb("#69768A")
#let edge = rgb("#A1ADBC")
#let blue = rgb("#2863B6")
#let blue-pale = rgb("#E8F1FE")
#let amber = rgb("#D8890B")
#let amber-pale = rgb("#FFF2D5")
#let red = rgb("#D94255")
#let red-pale = rgb("#FDE8EC")
#let teal = rgb("#078C79")
#let teal-pale = rgb("#E1F6F1")
#let neutral-pale = rgb("#F3F6FA")

#let at(x, y, body) = place(top + left, dx: x, dy: y, body)
#let rule(x, y, dx, dy, paint: edge, thick: 0.82pt, dash: none) = at(
  x, y,
  line(
    start: (0mm, 0mm),
    end: (dx, dy),
    stroke: (paint: paint, thickness: thick, dash: dash),
  ),
)
#let arrow(x, y, len, paint: teal) = {
  rule(x, y, len, 0mm, paint: paint, thick: 1.35pt)
  at(x + len - 0.2mm, y - 1.55mm, polygon(
    (0mm, 0mm), (3.1mm, 1.55mm), (0mm, 3.1mm),
    fill: paint,
  ))
}
#let node(label, side: "neutral", super: false) = {
  let fill = if side == "A" {
    blue-pale
  } else if side == "B" {
    amber-pale
  } else if super {
    teal-pale
  } else {
    neutral-pale
  }
  let normal-stroke = if side == "A" {
    blue
  } else if side == "B" {
    amber
  } else {
    blue
  }
  let stroke = if super { teal } else { normal-stroke }
  let label-color = if side == "A" {
    blue
  } else if side == "B" {
    amber
  } else {
    stroke
  }
  circle(
    radius: if super { 3.35mm } else { 2.85mm },
    fill: fill,
    stroke: (paint: stroke, thickness: if super { 1.35pt } else { 0.9pt }),
  )[
    #if label != none [
      #align(center + horizon)[
        #set text(size: if super { 4.4pt } else { 4.7pt }, weight: 800, fill: label-color)
        #label
      ]
    ]
  ]
}
#let tail-node() = circle(
  radius: 1.65mm,
  fill: amber-pale,
  stroke: 0.85pt + amber,
)
#let chip(body, fill, fg) = rect(
  radius: 2.2mm,
  fill: fill,
  inset: (x: 1.8mm, y: 0.72mm),
)[
  #set text(size: 5.85pt, weight: 700, fill: fg)
  #body
]
#let keep(x, y, dx, dy) = rule(x, y, dx, dy)
#let cut(x, y, dx, dy) = rule(x, y, dx, dy, paint: red, thick: 1.65pt)

// Draw the two K4 modules and their two sparse bridge edges.
#let graph-edges(x1, x2, x3, x4, y1, y2,
                 bad-cut: false, good-cut: false) = {
  let left-internal = if bad-cut { red } else { edge }
  let right-internal = if bad-cut { red } else { edge }
  let internal-thick = if bad-cut { 1.65pt } else { 0.82pt }
  let bridge-paint = if good-cut { red } else { edge }
  let bridge-thick = if good-cut { 1.65pt } else { 0.82pt }

  // K4 L.
  rule(x1, y1, x2 - x1, 0mm)
  rule(x1, y2, x2 - x1, 0mm)
  rule(x1, y1, 0mm, y2 - y1, paint: left-internal, thick: internal-thick)
  rule(x2, y1, 0mm, y2 - y1, paint: left-internal, thick: internal-thick)
  rule(x1, y1, x2 - x1, y2 - y1, paint: left-internal, thick: internal-thick)
  rule(x2, y1, x1 - x2, y2 - y1, paint: left-internal, thick: internal-thick)

  // K4 R.
  rule(x3, y1, x4 - x3, 0mm)
  rule(x3, y2, x4 - x3, 0mm)
  rule(x3, y1, 0mm, y2 - y1, paint: right-internal, thick: internal-thick)
  rule(x4, y1, 0mm, y2 - y1, paint: right-internal, thick: internal-thick)
  rule(x3, y1, x4 - x3, y2 - y1, paint: right-internal, thick: internal-thick)
  rule(x4, y1, x3 - x4, y2 - y1, paint: right-internal, thick: internal-thick)

  // Sparse interface.
  rule(x2, y1, x3 - x2, 0mm, paint: bridge-paint, thick: bridge-thick)
  rule(x2, y2, x3 - x2, 0mm, paint: bridge-paint, thick: bridge-thick)
}

#let graph-nodes(x1, x2, x3, x4, y1, y2,
                 left-side: "neutral", right-side: "neutral",
                 row-partition: false, supers: false) = {
  let lt = if row-partition { "A" } else { left-side }
  let lb = if row-partition { "B" } else { left-side }
  let rt = if row-partition { "A" } else { right-side }
  let rb = if row-partition { "B" } else { right-side }

  at(x1 - 2.85mm, y1 - 2.85mm, node("L1", side: lt))
  at(x2 - 2.85mm, y1 - 2.85mm, node(if supers { "L2*" } else { "L2" }, side: lt, super: supers))
  at(x1 - 2.85mm, y2 - 2.85mm, node("L3", side: lb))
  at(x2 - 2.85mm, y2 - 2.85mm, node(if supers { "L4*" } else { "L4" }, side: lb, super: supers))
  at(x3 - 2.85mm, y1 - 2.85mm, node("R1", side: rt))
  at(x4 - 2.85mm, y1 - 2.85mm, node("R2", side: rt))
  at(x3 - 2.85mm, y2 - 2.85mm, node("R3", side: rb))
  at(x4 - 2.85mm, y2 - 2.85mm, node("R4", side: rb))
}

#box(width: 178mm, height: 98mm)[
  // =============================================================== top row
  #at(4mm, 3mm)[#set text(size: 8pt, weight: 800); (a)]
  #at(103mm, 3mm)[#set text(size: 8pt, weight: 800); (b)]

  // Raw graph: two K4 communities joined by two subdivided interface paths.
  #graph-edges(10mm, 28mm, 54mm, 72mm, 19mm, 35mm)
  // Replace each direct bridge visually by a natural degree-2 series path.
  #rule(28mm, 19mm, 26mm, 0mm, paint: amber, thick: 1.05pt)
  #rule(28mm, 35mm, 26mm, 0mm, paint: amber, thick: 1.05pt)
  #at(34.85mm, 17.35mm, tail-node())
  #at(43.85mm, 17.35mm, tail-node())
  #at(34.85mm, 33.35mm, tail-node())
  #at(43.85mm, 33.35mm, tail-node())
  // The reducible motifs are existing interface paths, not appended leaves.
  #at(23.8mm, 13.6mm, rect(
    width: 25.7mm, height: 10.8mm, radius: 5mm,
    fill: none,
    stroke: (paint: teal, thickness: 0.95pt, dash: "dashed"),
  ))
  #at(23.8mm, 29.6mm, rect(
    width: 25.7mm, height: 10.8mm, radius: 5mm,
    fill: none,
    stroke: (paint: teal, thickness: 0.95pt, dash: "dashed"),
  ))
  #graph-nodes(10mm, 28mm, 54mm, 72mm, 19mm, 35mm)

  // Exact simplification arrow.
  #arrow(88.5mm, 26mm, 9mm)

  // Simplified K4 graph.
  #graph-edges(110mm, 128mm, 148mm, 166mm, 19mm, 35mm)
  #graph-nodes(110mm, 128mm, 148mm, 166mm, 19mm, 35mm, supers: true)

  #rule(3mm, 49mm, 172mm, 0mm, paint: rgb("#D7DEE8"), thick: 0.6pt)

  // ============================================================ bottom row
  #at(4mm, 52mm)[#set text(size: 8pt, weight: 800); (c)]
  #at(103mm, 52mm)[#set text(size: 8pt, weight: 800); (d)]

  // Bad horizontal 4|4 partition on the simplified graph.
  #graph-edges(12mm, 30mm, 50mm, 68mm, 68mm, 84mm, bad-cut: true)
  #graph-nodes(12mm, 30mm, 50mm, 68mm, 68mm, 84mm, row-partition: true, supers: true)
  #rule(3mm, 76mm, 76mm, 0mm, paint: red, thick: 1.05pt, dash: "dashed")

  // Waist-surgery arrow.
  #arrow(88.5mm, 76mm, 9mm)

  // Good vertical 4|4 partition on the same simplified graph.
  #graph-edges(110mm, 128mm, 148mm, 166mm, 68mm, 84mm, good-cut: true)
  #graph-nodes(
    110mm, 128mm, 148mm, 166mm, 68mm, 84mm,
    left-side: "A", right-side: "B", supers: true,
  )
  #rule(138mm, 59mm, 0mm, 32mm, paint: teal, thick: 1.05pt, dash: "dashed")
]
