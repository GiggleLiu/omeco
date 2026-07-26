#set page(width: 302mm, height: 43mm, margin: 0mm, fill: white)
#set text(font: "Avenir", size: 7pt, fill: rgb("#172033"))

#let edge = rgb("#A1ADBC")
#let node = rgb("#344256")
#let red = rgb("#D94255")
#let teal = rgb("#078C79")
#let muted = rgb("#7C8795")

#let at(x, y, body) = place(top + left, dx: x, dy: y, body)
#let rule(x, y, dx, dy, paint: edge, thick: 0.82pt) = at(
  x, y,
  line(
    start: (0mm, 0mm),
    end: (dx, dy),
    stroke: (paint: paint, thickness: thick),
  ),
)

#let rewrite-arrow(x) = {
  rule(x, 23mm, 5.4mm, 0mm, paint: teal, thick: 1.05pt)
  at(x + 5.3mm, 21.9mm, polygon(
    (0mm, 0mm), (2.2mm, 1.1mm), (0mm, 2.2mm),
    fill: teal,
  ))
}

#let leaf(label) = circle(
  radius: 2.05mm,
  fill: white,
  stroke: (paint: node, thickness: 0.82pt),
)[
  #align(center + horizon)[
    #set text(size: 3.7pt, weight: 800, fill: node)
    #label
  ]
]

#let junction(paint: node) = circle(radius: 0.66mm, fill: paint, stroke: none)

#let connect(p, q, paint: edge, thick: 0.82pt) = rule(
  p.at(0), p.at(1),
  q.at(0) - p.at(0), q.at(1) - p.at(1),
  paint: paint,
  thick: thick,
)

#let place-junction(p, paint: node) = at(
  p.at(0) - 0.66mm,
  p.at(1) - 0.66mm,
  junction(paint: paint),
)

#let place-leaf(p, label) = at(
  p.at(0) - 2.05mm,
  p.at(1) - 2.05mm,
  leaf(label),
)

#let panel-meta(x0, step) = {
  at(x0 + 14mm, 1.2mm, box(width: 7mm)[
    #align(center)[
      #set text(size: 4.6pt, weight: 800, fill: muted)
      #step
    ]
  ])
}

#let root-split(root, left, right) = {
  connect(root, left, paint: red, thick: 1.45pt)
  connect(root, right, paint: red, thick: 1.45pt)
  place-junction(root, paint: red)
  place-junction(left)
  place-junction(right)
}

#let draw-state-0(x0) = {
  let r = (x0 + 17mm, 10mm)
  let a = (x0 + 8mm, 19mm)
  let b = (x0 + 26mm, 19mm)
  let ai = (x0 + 11mm, 27mm)
  let bi = (x0 + 23mm, 27mm)
  let l1 = (x0 + 3mm, 34mm)
  let l2 = (x0 + 8mm, 39mm)
  let r1 = (x0 + 14mm, 39mm)
  let l3 = (x0 + 31mm, 34mm)
  let r2 = (x0 + 20mm, 39mm)
  let r3 = (x0 + 26mm, 39mm)

  root-split(r, a, b)
  connect(a, l1)
  connect(a, ai)
  connect(ai, l2)
  connect(ai, r1)
  connect(b, l3)
  connect(b, bi)
  connect(bi, r2)
  connect(bi, r3)
  place-junction(ai)
  place-junction(bi)
  place-leaf(l1, [L1])
  place-leaf(l2, [L2])
  place-leaf(r1, [R1])
  place-leaf(l3, [L3])
  place-leaf(r2, [R2])
  place-leaf(r3, [R3])
  panel-meta(x0, 0)
}

#let draw-state-1(x0) = {
  let r = (x0 + 17mm, 10mm)
  let a = (x0 + 8mm, 19mm)
  let b = (x0 + 26mm, 19mm)
  let ai = (x0 + 5mm, 27mm)
  let bi = (x0 + 23mm, 27mm)
  let l1 = (x0 + 2mm, 39mm)
  let l2 = (x0 + 8mm, 39mm)
  let r1 = (x0 + 14mm, 34mm)
  let l3 = (x0 + 31mm, 34mm)
  let r2 = (x0 + 20mm, 39mm)
  let r3 = (x0 + 26mm, 39mm)

  root-split(r, a, b)
  connect(a, ai)
  connect(a, r1)
  connect(ai, l1)
  connect(ai, l2)
  connect(b, l3)
  connect(b, bi)
  connect(bi, r2)
  connect(bi, r3)
  place-junction(ai)
  place-junction(bi)
  place-leaf(l1, [L1])
  place-leaf(l2, [L2])
  place-leaf(r1, [R1])
  place-leaf(l3, [L3])
  place-leaf(r2, [R2])
  place-leaf(r3, [R3])
  panel-meta(x0, 1)
}

#let draw-state-2(x0, step: 2, l3-first: false) = {
  let r = (x0 + 17mm, 10mm)
  let x = (x0 + 6mm, 19mm)
  let y = (x0 + 25mm, 17mm)
  let c = (x0 + 29mm, 24mm)
  let z = (x0 + 31mm, 31mm)
  let l1 = (x0 + 2mm, 34mm)
  let l2 = (x0 + 9mm, 34mm)
  let first = (x0 + 19mm, 28mm)
  let second = (x0 + 24mm, 35mm)
  let r2 = (x0 + 28mm, 39mm)
  let r3 = (x0 + 34mm, 39mm)

  root-split(r, x, y)
  connect(x, l1)
  connect(x, l2)
  connect(y, first)
  connect(y, c)
  connect(c, second)
  connect(c, z)
  connect(z, r2)
  connect(z, r3)
  place-junction(c)
  place-junction(z)
  place-leaf(l1, [L1])
  place-leaf(l2, [L2])
  place-leaf(first, if l3-first { [L3] } else { [R1] })
  place-leaf(second, if l3-first { [R1] } else { [L3] })
  place-leaf(r2, [R2])
  place-leaf(r3, [R3])
  panel-meta(x0, step)
}

#let draw-state-4(x0, final: false) = {
  let r = (x0 + 17mm, 10mm)
  let a = (x0 + 8mm, 19mm)
  let b = (x0 + 26mm, 19mm)
  let ai = if final { (x0 + 11mm, 27mm) } else { (x0 + 5mm, 27mm) }
  let bi = (x0 + 29mm, 27mm)
  let l1 = if final { (x0 + 3mm, 34mm) } else { (x0 + 2mm, 39mm) }
  let l2 = if final { (x0 + 8mm, 39mm) } else { (x0 + 8mm, 39mm) }
  let l3 = if final { (x0 + 14mm, 39mm) } else { (x0 + 14mm, 34mm) }
  let r1 = (x0 + 21mm, 34mm)
  let r2 = (x0 + 26mm, 39mm)
  let r3 = (x0 + 32mm, 39mm)

  root-split(r, a, b)
  if final {
    connect(a, l1)
    connect(a, ai)
    connect(ai, l2)
    connect(ai, l3)
  } else {
    connect(a, ai)
    connect(a, l3)
    connect(ai, l1)
    connect(ai, l2)
  }
  connect(b, r1)
  connect(b, bi)
  connect(bi, r2)
  connect(bi, r3)
  place-junction(ai)
  place-junction(bi)
  place-leaf(l1, [L1])
  place-leaf(l2, [L2])
  place-leaf(l3, [L3])
  place-leaf(r1, [R1])
  place-leaf(r2, [R2])
  place-leaf(r3, [R3])
  panel-meta(x0, if final { 5 } else { 4 })
}

#box(width: 302mm, height: 43mm)[
  #draw-state-0(2mm)
  #rewrite-arrow(42.5mm)

  #draw-state-1(54mm)
  #rewrite-arrow(94.5mm)

  #draw-state-2(106mm)
  #rewrite-arrow(146.5mm)

  #draw-state-2(158mm, step: 3, l3-first: true)
  #rewrite-arrow(198.5mm)

  #draw-state-4(210mm)
  #rewrite-arrow(250.5mm)

  #draw-state-4(262mm, final: true)
]
