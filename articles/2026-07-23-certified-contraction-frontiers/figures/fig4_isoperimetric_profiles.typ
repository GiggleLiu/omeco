// F4: isoperimetric profiles (JOSS cetz-plot house style).
// Two panels: empirical boundary profile vs spectral bound, with the
// bisection-width cap and (sycamore only) the certified cut marker.
// Data: ../data/fig/fig4_profiles.json (rendered by figures/export_data.py).
#import "@preview/cetz:0.4.0": canvas, draw
#import "@preview/cetz-plot:0.1.2": plot
#set page(width: auto, height: auto, margin: 5pt)

#let data = json("../data/fig/fig4_profiles.json")

#let panel(inst, y-max, legend: none) = {
  let prof = data.profiles.at(inst)
  let meta = data.meta.at(inst)
  let n = meta.n
  let bis = meta.bisection
  plot.plot(
    size: (8, 6),
    x-label: [subset size $k$  [tensors]],
    y-label: if inst == "reg3_250" [boundary $|diff S|$  [$log_2$ size]] else [],
    x-min: 0, x-max: n,
    y-min: 0, y-max: y-max,
    legend: legend,
    legend-style: (stroke: none, fill: white.transparentize(15%)),
    {
      // dotted bisection-width cap
      plot.add(
        ((0, bis), (n, bis)),
        style: (stroke: (dash: "dotted", paint: black, thickness: 1.6pt)),
        mark: none,
      )
      // solid blue empirical profile
      plot.add(
        prof.map(r => (r.k, r.emp)),
        style: (stroke: (paint: blue, thickness: 1.6pt)),
        mark: none,
        label: [empirical profile $b(k)$ (upper est.)],
      )
      // dashed red spectral bound
      plot.add(
        prof.map(r => (r.k, r.spec)),
        style: (stroke: (dash: "dashed", paint: red, thickness: 1.6pt)),
        mark: none,
        label: [spectral bound $lambda_2 k(n-k)/n$],
      )
      // certified cut marker (sycamore only)
      if meta.cert != none {
        plot.add(
          ((meta.cert.k, meta.cert.b),),
          style: (stroke: none),
          mark: "x",
          mark-size: 0.42,
          mark-style: (stroke: orange + 3pt),
          label: [certified cut: $|S|=141$, $|diff S|=40$],
        )
      }
    },
  )
}

#grid(columns: 2, gutter: 15pt,
  canvas(length: 1cm, {
    import draw: content
    panel("reg3_250", 36, legend: "inner-south-east")
    content((4, 6.4), [*reg3_250*])
    content((0.3, 5.35), [bisection width], anchor: "west")
  }),
  canvas(length: 1cm, {
    import draw: content
    panel("sycamore_m20", 56, legend: "inner-south")
    content((4, 6.4), [*sycamore_m20*])
    content((1.0, 5.65), [bisection width], anchor: "west")
  }),
)
