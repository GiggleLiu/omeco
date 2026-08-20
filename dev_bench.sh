#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: dev_bench.sh <instances_dir> <out.jsonl>" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")" && pwd)
INSTANCES_DIR=$1
OUT=$2
SUMMARIZER="$ROOT/summarize_attempt.py"
WALL_CAP_S=600
BUILD_CAP_S=120
PREP_ALLOWANCE_S=5
RUN_BUDGET_MS=30000
RUN_GUARD_S=5
SUMMARY_ALLOWANCE_S=10
INSTANCES=2
RELABELINGS=2
ARMS=2
RUNS=$(( INSTANCES * RELABELINGS * ARMS ))
RUN_CAP_S=$(( (RUN_BUDGET_MS + 999) / 1000 + RUN_GUARD_S ))
PLANNED_WALL_S=$(( BUILD_CAP_S + PREP_ALLOWANCE_S + RUNS * RUN_CAP_S + SUMMARY_ALLOWANCE_S ))

if (( PLANNED_WALL_S > WALL_CAP_S )); then
  echo "refusing plan: ${PLANNED_WALL_S}s exceeds hard ${WALL_CAP_S}s cap" >&2
  exit 2
fi
if [[ ! -f "$INSTANCES_DIR/sycamore_m20.json" || ! -f "$INSTANCES_DIR/reg3_250.json" ]]; then
  echo "instances_dir must contain sycamore_m20.json and reg3_250.json" >&2
  exit 2
fi

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  echo "budget plan: build<=${BUILD_CAP_S}s + prep<=${PREP_ALLOWANCE_S}s + ${RUNS} runs*(budget=${RUN_BUDGET_MS}ms+guard=${RUN_GUARD_S}s) + summarize<=${SUMMARY_ALLOWANCE_S}s = ${PLANNED_WALL_S}s <= hard ${WALL_CAP_S}s" >&2
  echo "evidence plan: sycamore_m20 + reg3_250; original + deterministic relabel; robust + ATT_PARENT=1; snapshots at 1,3,10,30s" >&2
  exec env ATT_BENCH_GUARDED=1 python3 "$SUMMARIZER" supervise "$WALL_CAP_S" -- "$0" "$@"
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/omeco-attempt063.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
: > "$OUT"

python3 "$SUMMARIZER" exec "$BUILD_CAP_S" "$TMP_DIR/build.stdout" "$TMP_DIR/build.stderr" -- \
  cargo build --manifest-path "$ROOT/Cargo.toml" --jobs 2 --release --offline --example attempt -p omeco

BIN="$ROOT/target/release/examples/attempt"
for instance in sycamore_m20 reg3_250; do
  source_graph="$INSTANCES_DIR/$instance.json"
  original_graph="$TMP_DIR/${instance}-r0.json"
  relabeled_graph="$TMP_DIR/${instance}-r1.json"
  cp "$source_graph" "$original_graph"
  python3 "$SUMMARIZER" relabel "$source_graph" "$relabeled_graph" 6301

  for relabeling in r0 r1; do
    graph="$TMP_DIR/${instance}-${relabeling}.json"
    for arm in parent robust; do
      log="$TMP_DIR/${instance}-${relabeling}-${arm}.stderr"
      tree="$TMP_DIR/${instance}-${relabeling}-${arm}.json"
      toggles=(ATT_PARENT=0)
      if [[ $arm == parent ]]; then
        toggles=(ATT_PARENT=1)
      fi
      python3 "$SUMMARIZER" run "$RUN_CAP_S" "$instance" "$relabeling" "$arm" \
        "$graph" "$TMP_DIR/${instance}-${relabeling}-${arm}.stdout" "$log" "$tree" -- \
        env -u ATT_PARENT -u ATT_DIAG -u ATT_BAND_BITS -u ATT_EPOCH_SWEEPS \
        -u ATT_BAND_BLO -u ATT_BAND_BHI -u ATT_MAX_SWEEPS -u ATT_BHI \
        -u ATT_BLO_COLD -u ATT_BKICK -u ATT_SW_COLD -u ATT_SW_KICK -u ATT_STAG \
        RAYON_NUM_THREADS=1 "${toggles[@]}" \
        "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree" >> "$OUT"
    done
  done
done

python3 "$SUMMARIZER" report "$OUT"
