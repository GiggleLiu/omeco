#set page(width: 210mm, height: 56mm, margin: 0mm, fill: white)
#set text(font: "Avenir", size: 7pt, fill: rgb("#172033"))

#let edge = rgb("#A1ADBC")
#let node = rgb("#344256")
#let red = rgb("#D94255")
#let teal = rgb("#078C79")
#let muted = rgb("#7C8795")

#let at(x, y, body) = place(top + left, dx: x, dy: y, body)
#let rule(x, y, dx, dy, paint: edge, thick: 0.82pt, dash: none) = at(
  x, y,
  line(
    start: (0mm, 0mm),
    end: (dx, dy),
    stroke: (paint: paint, thickness: thick, dash: dash),
  ),
)

#let stage-arrow(x, label) = {
  rule(x, 28mm, 8mm, 0mm, paint: teal, thick: 1.25pt)
  at(x + 7.9mm, 26.7mm, polygon(
    (0mm, 0mm), (2.6mm, 1.3mm), (0mm, 2.6mm),
    fill: teal,
  ))
  at(x - 1mm, 20mm, box(width: 12mm)[
    #align(center)[
      #set text(size: 4.6pt, weight: 800, fill: teal)
      #label
    ]
  ])
}

#let tensor-node(label) = circle(
  radius: 2.25mm,
  fill: white,
  stroke: (paint: node, thickness: 0.85pt),
)[
  #align(center + horizon)[
    #set text(size: 3.9pt, weight: 800, fill: node)
    #label
  ]
]

#let tree-junction(paint: node) = circle(radius: 0.72mm, fill: paint, stroke: none)

#let draw-tree(x0, panel, left-labels, right-labels, mirror-right: false) = {
  let root = (x0 + 22mm, 14mm)
  let a = (x0 + 12mm, 23mm)
  let b = (x0 + 32mm, 23mm)
  let a2 = (x0 + 13mm, 33mm)
  let b2 = if mirror-right {
    (x0 + 35mm, 33mm)
  } else {
    (x0 + 31mm, 33mm)
  }
  let leaves = (
    (x0 + 3mm, 42mm),
    (x0 + 9mm, 48mm),
    (x0 + 19mm, 48mm),
    if mirror-right { (x0 + 27mm, 42mm) } else { (x0 + 41mm, 42mm) },
    if mirror-right { (x0 + 32mm, 48mm) } else { (x0 + 26mm, 48mm) },
    if mirror-right { (x0 + 38mm, 48mm) } else { (x0 + 36mm, 48mm) },
  )

  // The highlighted root contraction combines the two three-leaf subtrees.
  rule(
    root.at(0), root.at(1),
    a.at(0) - root.at(0), a.at(1) - root.at(1),
    paint: red,
    thick: 1.55pt,
  )
  rule(
    root.at(0), root.at(1),
    b.at(0) - root.at(0), b.at(1) - root.at(1),
    paint: red,
    thick: 1.55pt,
  )

  rule(a.at(0), a.at(1), leaves.at(0).at(0) - a.at(0), leaves.at(0).at(1) - a.at(1))
  rule(a.at(0), a.at(1), a2.at(0) - a.at(0), a2.at(1) - a.at(1))
  rule(a2.at(0), a2.at(1), leaves.at(1).at(0) - a2.at(0), leaves.at(1).at(1) - a2.at(1))
  rule(a2.at(0), a2.at(1), leaves.at(2).at(0) - a2.at(0), leaves.at(2).at(1) - a2.at(1))

  rule(b.at(0), b.at(1), leaves.at(3).at(0) - b.at(0), leaves.at(3).at(1) - b.at(1))
  rule(b.at(0), b.at(1), b2.at(0) - b.at(0), b2.at(1) - b.at(1))
  rule(b2.at(0), b2.at(1), leaves.at(4).at(0) - b2.at(0), leaves.at(4).at(1) - b2.at(1))
  rule(b2.at(0), b2.at(1), leaves.at(5).at(0) - b2.at(0), leaves.at(5).at(1) - b2.at(1))

  for p in (a, b, a2, b2) {
    at(p.at(0) - 0.72mm, p.at(1) - 0.72mm, tree-junction())
  }
  at(root.at(0) - 0.72mm, root.at(1) - 0.72mm, tree-junction(paint: red))

  let all-labels = (
    left-labels.at(0),
    left-labels.at(1),
    left-labels.at(2),
    right-labels.at(0),
    right-labels.at(1),
    right-labels.at(2),
  )
  for i in range(6) {
    let p = leaves.at(i)
    at(p.at(0) - 2.25mm, p.at(1) - 2.25mm, tensor-node(all-labels.at(i)))
  }

  at(x0, 2mm)[
    #set text(size: 7.5pt, weight: 800, fill: node)
    (#panel)
  ]
}

#let draw-cut(x0, panel, left-nodes) = {
  let labels = ("L1", "L2", "L3", "R1", "R2", "R3")
  let right-nodes = range(6).filter(i => not left-nodes.contains(i))
  let left-slots = (
    (x0 + 3mm, 24mm),
    (x0 + 11mm, 15mm),
    (x0 + 15mm, 36mm),
  )
  let right-slots = (
    (x0 + 25mm, 27mm),
    (x0 + 34mm, 18mm),
    (x0 + 31mm, 38mm),
  )

  let position(i) = {
    if left-nodes.contains(i) {
      left-slots.at(left-nodes.position(v => v == i))
    } else {
      right-slots.at(right-nodes.position(v => v == i))
    }
  }

  let draw-edge(i, j) = {
    let pi = position(i)
    let pj = position(j)
    let crosses = left-nodes.contains(i) != left-nodes.contains(j)
    if crosses {
      rule(
        pi.at(0), pi.at(1),
        pj.at(0) - pi.at(0), pj.at(1) - pi.at(1),
        paint: white,
        thick: 2.65pt,
      )
      rule(
        pi.at(0), pi.at(1),
        pj.at(0) - pi.at(0), pj.at(1) - pi.at(1),
        paint: red,
        thick: 1.3pt,
      )
    } else {
      rule(
        pi.at(0), pi.at(1),
        pj.at(0) - pi.at(0), pj.at(1) - pi.at(1),
        paint: edge,
        thick: 0.8pt,
      )
    }
  }

  // Two triangles joined by the bridge L2--R1.
  draw-edge(0, 1)
  draw-edge(1, 2)
  draw-edge(2, 0)
  draw-edge(3, 4)
  draw-edge(4, 5)
  draw-edge(5, 3)
  draw-edge(1, 3)

  rule(x0 + 20mm, 11mm, 0mm, 31mm, paint: teal, thick: 0.82pt, dash: "dashed")

  for i in range(6) {
    let p = position(i)
    at(p.at(0) - 2.25mm, p.at(1) - 2.25mm, tensor-node(labels.at(i)))
  }

  at(x0, 2mm)[
    #set text(size: 7.5pt, weight: 800, fill: node)
    (#panel)
  ]
}

#box(width: 210mm, height: 56mm)[
  // Incumbent tree: its waist mixes the two triangles.
  #draw-tree(
    3mm,
    [a],
    ("L1", "L2", "R1"),
    ("L3", "R2", "R3"),
  )

  #stage-arrow(49mm, [extract])

  // The same mixed partition, now viewed as a graph cut.
  #draw-cut(61mm, [b], (0, 1, 3))

  #stage-arrow(100mm, [FM pass])

  // Best balanced prefix of the FM pass.
  #draw-cut(112mm, [c], (0, 1, 2))

  #stage-arrow(151mm, [rebuild])

  // Both sides are rebuilt around the improved partition.
  #draw-tree(
    163mm,
    [d],
    ("L1", "L2", "L3"),
    ("R1", "R2", "R3"),
    mirror-right: true,
  )

  #at(168mm, 52mm, box(width: 37mm)[
    #align(center)[
      #set text(size: 4.7pt, weight: 800, fill: teal)
      accept if measured cost decreases
    ]
  ])
]
