# Changelog

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_iters` (default `0`/off): deterministic iteration cap
  for a waist-surgery post-pass; never worse than the unrefined tree, and
  fully reproducible across machines for any fixed config. `TreeSA` has no
  wall-clock knob — the low-level `waist_surgery::refine`/`refine_capped`
  APIs keep their `Duration` budget for power users.
- Python: `TreeSA(preprocess=..., surgery_iters=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.
