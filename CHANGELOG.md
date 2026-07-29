# Changelog

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_budget` (seconds, default 0.0/off): opt-in
  waist-surgery post-pass; never worse than the unrefined tree, wall-clock
  dependent.
- Python: `TreeSA(preprocess=..., surgery_budget=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.
