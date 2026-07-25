// F10: waist surgery — (a) FM-call scatter vs incumbent cut, (b) surgery trajectory.
// Data: ../data/fig/fig10_waist.json
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig10_waist.json")
#let sc = data.calls_surfacecode
#let ksg = data.calls_ksg
#let traj = data.trajectory_surfacecode_run2

#grid(columns: 2, gutter: 18pt,
  // ---- panel (a): FM-call scatter ----
  canvas(length: 1cm, {
    import draw: content
    let width = 7
    let height = 6
    let xmin = 18
    let xmax = 56
    let ymin = 18
    let ymax = 57
    let px(dx) = (dx - xmin) / (xmax - xmin) * width
    let py(dy) = (dy - ymin) / (ymax - ymin) * height

    plot.plot(
      size: (width, height),
      x-label: [incumbent waist cut weight  [log#sub[2]]],
      y-label: [best cut found by FM  [log#sub[2]]],
      x-min: xmin, x-max: xmax, y-min: ymin, y-max: ymax,
      x-tick-step: 10, y-tick-step: 5,
      legend: "inner-north-west",
      {
        // diagonal y = x
        plot.add(((20, 20), (55, 55)),
          style: (stroke: (dash: "dashed", paint: black, thickness: 1.2pt)),
          mark: none)
        plot.add(
          sc.map(p => (p.cost, p.alt)),
          style: (stroke: none),
          mark: "o",
          mark-size: 0.15,
          mark-style: (fill: white, stroke: blue + 1.1pt),
          label: "surface code d=21 (" + str(sc.len()) + " calls)")
        plot.add(
          ksg.map(p => (p.cost, p.alt)),
          style: (stroke: none),
          mark: "square",
          mark-size: 0.15,
          mark-style: (fill: white, stroke: red + 1.1pt),
          label: "king graph IS (" + str(ksg.len()) + " calls)")
      },
    )
    content((px(47), py(50.5)), align(center, text(size: 8pt)[waist not\ improvable]))
    content((px(53), py(20.5)), text(size: 11pt)[(a)])
  }),

  // ---- panel (b): surgery trajectory ----
  canvas(length: 1cm, {
    import draw: content
    let width = 7
    let height = 6
    let xmin = -2
    let xmax = 89
    let ymin = 47.1
    let ymax = 53.4
    let px(dx) = (dx - xmin) / (xmax - xmin) * width
    let py(dy) = (dy - ymin) / (ymax - ymin) * height

    let incumbent = traj.filter(p => p.accept == none).map(p => (p.t, p.tc))
    let accepted = traj.filter(p => p.accept == true).map(p => (p.t, p.tc))

    plot.plot(
      size: (width, height),
      x-label: [wall-clock time  [s]],
      y-label: [surface code d=21   tc  [log#sub[2] flops]],
      x-min: xmin, x-max: xmax, y-min: ymin, y-max: ymax,
      x-tick-step: 20, y-tick-step: 1,
      legend: "inner-north-east",
      {
        // pre-surgery record (black dotted) and official median (red dash-dotted)
        plot.add(((0, data.pre_surgery_record), (88, data.pre_surgery_record)),
          style: (stroke: (dash: "dotted", paint: black, thickness: 1.4pt)), mark: none)
        plot.add(((0, data.official_median), (88, data.official_median)),
          style: (stroke: (dash: "dash-dotted", paint: red, thickness: 1.4pt)), mark: none)
        // incumbent tc line
        plot.add(incumbent,
          style: (stroke: blue + 1.6pt), mark: none,
          label: "incumbent tc")
        // accepted waist rebuilds
        plot.add(accepted,
          style: (stroke: none),
          mark: "x", mark-size: 0.32,
          mark-style: (stroke: red + 2.2pt),
          label: "accepted waist rebuild")
      },
    )
    content((px(44), py(47.95)),
      text(size: 8pt)[pre-surgery record 47.824], anchor: "west")
    content((px(1), py(47.28)),
      text(size: 8pt, fill: red)[official median with surgery 47.377], anchor: "west")
    content((px(86), py(47.42)), text(size: 11pt)[(b)], anchor: "east")
  }),
)
