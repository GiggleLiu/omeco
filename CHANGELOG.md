# Changelog

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_iters` (default `0`/off): number of interleaved
  anneal–surgery rounds (Algorithm 1 of the companion paper) run on the
  pipeline's tree. Each round is one waist-surgery iteration followed by a
  full warm-started annealing pass, so a round costs about one extra anneal;
  the best tree seen is returned, so more rounds are never worse. Fully
  deterministic — rounds are counted, never timed — and auto-skipped for
  `DecompositionType::Path` configs, which the loop cannot preserve. `TreeSA`
  has no wall-clock knob; the low-level `waist_surgery::refine`/
  `refine_capped` APIs are unchanged and keep their `Duration` budget for
  power users.
- New `treesa::anneal_surgery_rounds` + `RoundsReport`: the same interleaved
  loop as a standalone function, with a per-round score trace.
- Python: `TreeSA(preprocess=..., surgery_iters=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.
