#set page(width: 302mm, height: 42mm, margin: 0mm, fill: white)
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

#let time-arrow(x, label) = {
  rule(x, 28mm, 5.4mm, 0mm, paint: teal, thick: 1.05pt)
  at(x + 5.3mm, 26.9mm, polygon(
    (0mm, 0mm), (2.2mm, 1.1mm), (0mm, 2.2mm),
    fill: teal,
  ))
  at(x - 3.3mm, 24.8mm, box(width: 12mm)[
    #align(center)[
      #set text(size: 4.4pt, weight: 800, fill: node)
      #label
    ]
  ])
}

#let tensor-node(label) = circle(
  radius: 2.15mm,
  fill: white,
  stroke: (paint: node, thickness: 0.82pt),
)[
  #align(center + horizon)[
    #set text(size: 3.9pt, weight: 800, fill: node)
    #label
  ]
]

#let draw-state(x0, step, left-nodes) = {
  let labels = ("L1", "L2", "L3", "R1", "R2", "R3")
  let right-nodes = range(6).filter(i => not left-nodes.contains(i))

  let left-slots = if left-nodes.len() == 4 {
    (
      (x0 + 3mm, 22mm),
      (x0 + 10mm, 15mm),
      (x0 + 15mm, 24mm),
      (x0 + 10mm, 37mm),
    )
  } else if left-nodes.len() == 3 {
    (
      (x0 + 3mm, 23mm),
      (x0 + 10mm, 15mm),
      (x0 + 14mm, 35mm),
    )
  } else {
    (
      (x0 + 5mm, 19mm),
      (x0 + 14mm, 35mm),
    )
  }

  let right-slots = if right-nodes.len() == 4 {
    (
      (x0 + 22mm, 24mm),
      (x0 + 28mm, 16mm),
      (x0 + 29mm, 34mm),
      (x0 + 22mm, 38mm),
    )
  } else if right-nodes.len() == 3 {
    (
      if step == 0 or step == 4 {
        (x0 + 23mm, 30mm)
      } else {
        (x0 + 22mm, 26mm)
      },
      (x0 + 29mm, 18mm),
      (x0 + 27mm, 37mm),
    )
  } else {
    (
      (x0 + 24mm, 20mm),
      (x0 + 29mm, 35mm),
    )
  }

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
        thick: 2.55pt,
      )
      rule(
        pi.at(0), pi.at(1),
        pj.at(0) - pi.at(0), pj.at(1) - pi.at(1),
        paint: red,
        thick: 1.25pt,
      )
    } else {
      rule(
        pi.at(0), pi.at(1),
        pj.at(0) - pi.at(0), pj.at(1) - pi.at(1),
        paint: edge,
        thick: 0.78pt,
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

  rule(x0 + 18.5mm, 12mm, 0mm, 28mm, paint: teal, thick: 0.78pt, dash: "dashed")

  for i in range(6) {
    let p = position(i)
    at(p.at(0) - 2.15mm, p.at(1) - 2.15mm, tensor-node(labels.at(i)))
  }

  at(x0 + 13mm, 1.3mm, box(width: 7mm)[
    #align(center)[
      #set text(size: 4.6pt, weight: 800, fill: muted)
      #step
    ]
  ])
}

#box(width: 302mm, height: 42mm)[
  #draw-state(2mm, 0, (0, 1, 3))
  #time-arrow(36.5mm, [move L3])

  #draw-state(46mm, 1, (0, 1, 2, 3))
  #time-arrow(80.5mm, [move R1])

  #draw-state(90mm, 2, (0, 1, 2))
  #time-arrow(124.5mm, [move L2])

  #draw-state(134mm, 3, (0, 2))
  #time-arrow(168.5mm, [move R2])

  #draw-state(178mm, 4, (0, 2, 4))
  #time-arrow(212.5mm, [move L1])

  #draw-state(222mm, 5, (2, 4))
  #time-arrow(256.5mm, [move R3])

  #draw-state(266mm, 6, (2, 4, 5))
]
