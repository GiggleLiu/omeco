// F15: why TreeSA freezes and surgery escapes — schematic landscape over
// tree configurations connected by local rewrites. Double well: incumbent
// bipartition basin (left), cheaper separator basin (right), barrier of
// mixed bipartitions between them. Pure schematic, no data file.
#import "@preview/cetz:0.4.0": canvas, draw
#set page(width: auto, height: auto, margin: 5pt)

#canvas(length: 1cm, {
  import draw: line, content, circle, bezier, arc, mark

  // ---- landscape curve: smoothstep interpolation through knots ----
  // knots: (x, y) with zero slope; two wells and one barrier
  let knots = ((0.0, 4.6), (2.4, 3.1), (6.2, 5.3), (10.2, 1.5), (12.4, 2.8))
  let pts = ()
  for k in range(knots.len() - 1) {
    let (x0, y0) = knots.at(k)
    let (x1, y1) = knots.at(k + 1)
    let n = 24
    for i in range(n) {
      let t = i / n
      let s = 3 * t * t - 2 * t * t * t
      pts.push((x0 + (x1 - x0) * t, y0 + (y1 - y0) * s))
    }
  }
  pts.push(knots.last())
  line(..pts, stroke: black + 2pt)

  // ---- axes ----
  line((-0.4, 0.6), (-0.4, 5.9), stroke: 1pt, mark: (end: "straight"))
  content((-0.75, 3.2), rotate(-90deg)[waist cut weight  $w$  [log#sub[2]]])
  line((-0.4, 0.6), (12.8, 0.6), stroke: 1pt, mark: (end: "straight"))
  content((6.2, 0.2), [contraction trees connected by local rewrites])

  // ---- left well: frozen incumbent ----
  circle((2.4, 3.28), radius: 0.16, fill: rgb("#3b6fc4"), stroke: black + 0.8pt)
  content((2.4, 2.3), align(center)[frozen incumbent\ waist $(A, B)$])
  // thermal jiggle: short double arrow inside the well
  line((1.7, 3.55), (3.1, 3.55), stroke: (paint: rgb("#3b6fc4"), thickness: 1.2pt),
       mark: (start: "straight", end: "straight"))
  content((1.15, 5.25),
    align(center, text(size: 8.5pt)[local rewrites:\ one subtree at a time]))
  line((1.35, 4.85), (2.05, 3.75), stroke: (paint: gray, thickness: 0.6pt))

  // ---- barrier annotation ----
  content((6.35, 3.65), align(center, text(size: 8.5pt)[mixed bipartitions:\ every local path\ climbs here]))
  // rejected uphill attempt: dotted trajectory partway up, crossed out
  let attempt = pts.filter(p => p.at(0) >= 2.4 and p.at(0) <= 4.4).map(p => (p.at(0), p.at(1) + 0.12))
  line(..attempt, stroke: (paint: rgb("#3b6fc4"), thickness: 1.2pt, dash: "dotted"))
  content((4.72, 4.5), text(size: 12pt, fill: rgb("#3b6fc4"))[$times$])
  content((5.05, 2.45), align(center, text(size: 8.5pt, fill: rgb("#3b6fc4"))[cooled annealer:\ uphill sequence never accepted]))

  // ---- right well: cheaper separator ----
  circle((10.2, 1.68), radius: 0.16, fill: white, stroke: rgb("#c43b3b") + 1.4pt)
  content((10.2, 0.95), align(center)[cheaper separator $(A', B')$])
  // measured gap between well depths
  line((11.6, 3.1), (11.6, 1.5), stroke: (paint: gray, thickness: 0.9pt),
       mark: (start: "straight", end: "straight"))
  line((2.4, 3.1), (11.85, 3.1), stroke: (paint: gray, thickness: 0.6pt, dash: "dotted"))
  line((10.2, 1.5), (11.85, 1.5), stroke: (paint: gray, thickness: 0.6pt, dash: "dotted"))
  content((12.15, 2.3), text(size: 8.5pt)[3--12 bits\ (835/835 calls)], anchor: "west")

  // ---- surgery arc over the barrier ----
  bezier((2.4, 3.45), (10.2, 1.95), (4.6, 7.3), (8.6, 6.9),
         stroke: (paint: rgb("#c43b3b"), thickness: 2pt, dash: "dashed"),
         mark: (end: "straight"))
  content((6.4, 6.85), align(center, text(fill: rgb("#c43b3b"))[
    global cut surgery: bounded FM improves the cut directly,\
    both sides rebuilt cold, accepted on measured tc]))
})
