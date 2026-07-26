#set page(width: 188mm, height: 50mm, margin: 0mm, fill: white)
#set text(font: "Avenir", size: 7pt, fill: rgb("#172033"))

#let edge = rgb("#A1ADBC")
#let node = rgb("#344256")
#let teal = rgb("#078C79")
#let teal-pale = rgb("#E8F7F3")

#let at(x, y, body) = place(top + left, dx: x, dy: y, body)
#let rule(x, y, dx, dy, paint: edge, thick: 0.82pt) = at(
  x, y,
  line(
    start: (0mm, 0mm),
    end: (dx, dy),
    stroke: (paint: paint, thickness: thick),
  ),
)

#let arrow(x, y, len) = {
  rule(x, y, len, 0mm, paint: teal, thick: 1.3pt)
  at(x + len - 0.1mm, y - 1.3mm, polygon(
    (0mm, 0mm), (2.6mm, 1.3mm), (0mm, 2.6mm),
    fill: teal,
  ))
}

#let tensor-node(label, auxiliary: false, super: false) = circle(
  radius: if super { 2.75mm } else if auxiliary { 1.9mm } else { 2.3mm },
  fill: white,
  stroke: (
    paint: node,
    thickness: if super { 1.15pt } else { 0.85pt },
  ),
)[
  #align(center + horizon)[
    #set text(
      size: if auxiliary { 3.5pt } else if super { 3.8pt } else { 4pt },
      weight: 800,
      fill: node,
    )
    #label
  ]
]

#let connect(p, q, paint: edge, thick: 0.82pt) = rule(
  p.at(0), p.at(1),
  q.at(0) - p.at(0), q.at(1) - p.at(1),
  paint: paint,
  thick: thick,
)

#let place-node(p, label, auxiliary: false, super: false) = {
  let r = if super { 2.75mm } else if auxiliary { 1.9mm } else { 2.3mm }
  at(
    p.at(0) - r,
    p.at(1) - r,
    tensor-node(label, auxiliary: auxiliary, super: super),
  )
}

#let draw-raw() = {
  let l1 = (12mm, 32mm)
  let x = (25mm, 23mm)
  let l2 = (37mm, 15mm)
  let y = (41mm, 30mm)
  let l3 = (39mm, 45mm)
  let r1 = (67mm, 32mm)
  let z = (75mm, 24mm)
  let r2 = (84mm, 16mm)
  let r3 = (82mm, 45mm)

  // Solid regions mark rank-non-increasing fragments.
  at(22mm, 10mm, ellipse(
    width: 23mm,
    height: 25mm,
    fill: teal-pale,
    stroke: (paint: teal, thickness: 0.75pt),
  ))
  at(71mm, 11mm, ellipse(
    width: 17mm,
    height: 18mm,
    fill: teal-pale,
    stroke: (paint: teal, thickness: 0.75pt),
  ))

  // Two triangles joined by L2--R1. The degree-2 series tensors x,y,z
  // can be absorbed without increasing tensor rank.
  connect(l1, x)
  connect(x, l2, paint: teal, thick: 1.2pt)
  connect(l2, y, paint: teal, thick: 1.2pt)
  connect(y, l3)
  connect(l3, l1)
  connect(l2, r1)
  connect(r1, z)
  connect(z, r2, paint: teal, thick: 1.2pt)
  connect(r2, r3)
  connect(r3, r1)

  place-node(l1, [L1])
  place-node(x, [x], auxiliary: true)
  place-node(l2, [L2])
  place-node(y, [y], auxiliary: true)
  place-node(l3, [L3])
  place-node(r1, [R1])
  place-node(z, [z], auxiliary: true)
  place-node(r2, [R2])
  place-node(r3, [R3])
}

#let draw-simplified() = {
  let l1 = (112mm, 32mm)
  let l2 = (132mm, 16mm)
  let l3 = (134mm, 45mm)
  let r1 = (158mm, 32mm)
  let r2 = (178mm, 16mm)
  let r3 = (176mm, 45mm)

  connect(l1, l2)
  connect(l2, l3)
  connect(l3, l1)
  connect(l2, r1)
  connect(r1, r2)
  connect(r2, r3)
  connect(r3, r1)

  place-node(l1, [L1])
  place-node(l2, [L2\*], super: true)
  place-node(l3, [L3])
  place-node(r1, [R1])
  place-node(r2, [R2\*], super: true)
  place-node(r3, [R3])
}

#box(width: 188mm, height: 50mm)[
  #at(3mm, 2mm)[
    #set text(size: 8pt, weight: 800, fill: node)
    (a)
  ]
  #at(106mm, 2mm)[
    #set text(size: 8pt, weight: 800, fill: node)
    (b)
  ]

  #draw-raw()
  #arrow(94mm, 29mm, 9mm)
  #draw-simplified()
]
