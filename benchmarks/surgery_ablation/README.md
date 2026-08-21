# Surgery v2 and work-matched ablation

This benchmark compares the matched cold-only control, four opt-in surgery
variants, and longer TreeSA runs. The default protocol uses five deterministic
labels (`r0`–`r4`), 140 million planned baseline node visits, rounds 8 and 32,
pure-time scoring, preprocessing, and serial execution.

Each JSONL row is appended and flushed independently. Re-running the same
command skips existing keys, so an interrupted campaign resumes in place.
`wall_seconds` is machine dependent; `total_node_visits` is the primary
machine-independent work coordinate. Surgery/FM work is not represented by the
node-visit coordinate and must be interpreted alongside wall time.

For every rounds arm, the driver compares the spliced rounds result with the
spliced baseline using pure time complexity on the original relabeled network.
The reported tree is never worse than that baseline; the JSONL field
`post_splice_guard_triggered` records when the guard selected the baseline.

## Full campaign

```bash
INSTANCE_DIR=/Users/liujinguo/rcode/contraction-order-frontiers/benchmarks/omeco/instances

RAYON_NUM_THREADS=1 cargo run --release -p omeco --example surgery_ablation -- \
  --instances "$INSTANCE_DIR" \
  --out benchmarks/surgery_ablation/results/full-preprocessed.jsonl

RAYON_NUM_THREADS=1 cargo run --release -p omeco --example surgery_ablation -- \
  --instances "$INSTANCE_DIR" \
  --out benchmarks/surgery_ablation/results/full-raw.jsonl \
  --raw

python3 benchmarks/surgery_ablation/summarize.py \
  benchmarks/surgery_ablation/results/full-preprocessed.jsonl \
  --out benchmarks/surgery_ablation/results/full-preprocessed.md

python3 benchmarks/surgery_ablation/summarize.py \
  benchmarks/surgery_ablation/results/full-raw.jsonl \
  --out benchmarks/surgery_ablation/results/full-raw.md
```

Use `--jobs N` only on a machine where process-level parallelism and timing
contention are acceptable. Each process remains single-trial and deterministic;
the default is serial. `--only name,name`, `--set a|b`, `--labels N`, and
`--rounds list` select subsets without changing the default full protocol.

## Smoke run

The smoke run checks all set-B arms at R=8 on four instances and two
labels. It intentionally lowers the visit target to keep a clean run below 15
minutes; its rows record that override and are functional evidence, not paper
quality data. All generated rows land in `results/`, which is gitignored;
completed campaign artifacts are archived in the companion
`contraction-order-frontiers` repository under `data/surgery_ablation/`.

```bash
RAYON_NUM_THREADS=1 cargo run --release -p omeco --example surgery_ablation -- \
  --instances /Users/liujinguo/rcode/contraction-order-frontiers/benchmarks/omeco/instances \
  --only surfacecode_d9,dbn_13,qft_27,petersen \
  --set b --rounds 8 --labels 2 --visits 200000 \
  --out benchmarks/surgery_ablation/results/smoke.jsonl

python3 benchmarks/surgery_ablation/summarize.py \
  benchmarks/surgery_ablation/results/smoke.jsonl \
  --out benchmarks/surgery_ablation/results/smoke.md
```
