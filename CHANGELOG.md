# Changelog

## 0.2.7 — 2026-07-29

**Default behavior change:** `TreeSA` now runs the structural
simplification front-end by default (`preprocess: true`) — output trees
may differ from 0.2.6 (same contraction result, typically equal or better
cost). Restore the old behavior with `preprocess: false`.

- New `TreeSA.surgery_iters` (default `0`/off): number of interleaved
  anneal–surgery rounds (Algorithm 1 of the companion paper) run on the
  reduced-network tree. Each round is one waist-surgery iteration followed by
  a cold span-gated fine-tuning pass with an incumbent ratchet. After
  splice-back, the candidate is guarded against the rounds-off baseline; the
  standalone reduced-network loop is monotone in its round count. Fully
  deterministic — rounds are counted, never timed — and auto-skipped for
  `DecompositionType::Path` configs, which the loop cannot preserve. `TreeSA`
  has no wall-clock knob; the low-level `waist_surgery::refine`/
  `refine_capped` APIs are unchanged and keep their `Duration` budget for
  power users.
- New `TreeSA.surgery_probability` (default `0.0`/off): mixes waist surgery
  directly into TreeSA. At each sweep, the configured probability replaces
  the local sweep with one waist-guided leaf prune-and-regraft move, accepted
  at the current inverse temperature. The rule never restarts cooling, invokes
  a timer, or launches a nested anneal; Rust and Python expose the same setting.
- New `treesa::anneal_surgery_rounds` + `RoundsReport`: the same interleaved
  loop as a standalone function, with separate per-round surgery candidate,
  raw fine-tuning endpoint, and retained-incumbent traces.
- Python: `TreeSA(preprocess=..., surgery_iters=..., surgery_probability=...)`,
  `simplify_then_optimize`, `waist_refine`, `SimplifyReport`, `WaistReport`.
- Fixed Python `TreeSA()` to delegate its default β schedule to Rust
  `TreeSA::default()` (`0..300`); the previous duplicate used `1..=300` and
  could shift benchmark quality by several bits.
- Committed paper-benchmark artifact under `benchmarks/paper/` (manifest,
  instances, deterministic runner, checker), plus a CI job that re-derives the
  small `ci` set on every PR and fails on a single changed field. A dedicated
  `figure2b` set checks the paper mechanism and renders a static SVG.
- `Label` now requires `Ord` (all built-in label types already qualify).
- The Julia behavioral-alignment rule is retired; JSON interop is unchanged.
