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
BUILD_CAP_S=60
RUN_BUDGET_MS=${ATT_DEV_BUDGET_MS:-50000}
SWEEP_CAP=${ATT_DEV_SWEEPS:-2048}
RUN_GUARD_S=3
SUMMARY_ALLOWANCE_S=10
RUNS=8
RUN_CAP_S=$(( (RUN_BUDGET_MS + 999) / 1000 + RUN_GUARD_S ))
PLANNED_WALL_S=$(( BUILD_CAP_S + RUNS * RUN_CAP_S + SUMMARY_ALLOWANCE_S ))

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  exec env ATT_BENCH_GUARDED=1 python3 "$SUMMARIZER" supervise "$WALL_CAP_S" -- "$0" "$@"
fi

if (( PLANNED_WALL_S > WALL_CAP_S )); then
  echo "refusing plan: ${PLANNED_WALL_S}s exceeds hard ${WALL_CAP_S}s cap" >&2
  exit 2
fi
if [[ ! -f "$INSTANCES_DIR/ksg.json" || ! -f "$INSTANCES_DIR/sycamore_m20.json" ]]; then
  echo "instances_dir must contain ksg.json and sycamore_m20.json" >&2
  exit 2
fi

echo "budget plan: build<=${BUILD_CAP_S}s + ${RUNS} runs*(budget=${RUN_BUDGET_MS}ms+guard=${RUN_GUARD_S}s) + summarize<=${SUMMARY_ALLOWANCE_S}s = ${PLANNED_WALL_S}s <= ${WALL_CAP_S}s" >&2
echo "work match: ATT_MAX_SWEEPS=${SWEEP_CAP}; arms per instance: parent,c=1,c=2,c=4" >&2

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/omeco-attempt061.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
: > "$OUT"

python3 "$SUMMARIZER" exec "$BUILD_CAP_S" "$TMP_DIR/build.stdout" "$TMP_DIR/build.stderr" -- \
  cargo build --release --offline --example attempt -p omeco

BIN="$ROOT/target/release/examples/attempt"
for instance in ksg sycamore_m20; do
  graph="$INSTANCES_DIR/$instance.json"
  for arm in parent c1 c2 c4; do
    log="$TMP_DIR/${instance}-${arm}.stderr"
    tree="$TMP_DIR/${instance}-${arm}.json"
    if [[ $arm == parent ]]; then
      mode=parent
      c=parent
      toggles=(ATT_PARENT=1)
    else
      mode=band
      c=${arm#c}
      toggles=("ATT_BAND_BITS=$c")
    fi
    python3 "$SUMMARIZER" exec "$RUN_CAP_S" "$TMP_DIR/${instance}-${arm}.stdout" "$log" -- \
      env RAYON_NUM_THREADS=1 ATT_DIAG=1 ATT_MAX_SWEEPS="$SWEEP_CAP" "${toggles[@]}" \
      "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree"
    python3 "$SUMMARIZER" row "$instance" "$mode" "$c" "$SWEEP_CAP" "$log" "$tree" >> "$OUT"
  done
done

python3 "$SUMMARIZER" report "$OUT"
