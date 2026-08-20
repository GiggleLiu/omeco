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
RUN_BUDGET_MS=${ATT_DEV_BUDGET_MS:-30000}
SWEEP_CAP=${ATT_DEV_SWEEPS:-1024}
RUN_GUARD_S=3
SUMMARY_ALLOWANCE_S=10
RUNS=15
RUN_CAP_S=$(( (RUN_BUDGET_MS + 999) / 1000 + RUN_GUARD_S ))
PLANNED_WALL_S=$(( BUILD_CAP_S + RUNS * RUN_CAP_S + SUMMARY_ALLOWANCE_S ))

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  echo "budget plan: build<=${BUILD_CAP_S}s + ${RUNS} runs*(budget=${RUN_BUDGET_MS}ms+guard=${RUN_GUARD_S}s) + summarize<=${SUMMARY_ALLOWANCE_S}s = ${PLANNED_WALL_S}s (hard cap ${WALL_CAP_S}s)" >&2
  echo "work match: ATT_MAX_SWEEPS=${SWEEP_CAP}; serial 1-core runs; 3 instances; arms: event-W,event-2W,fixed-{15,25,40}%" >&2
fi

if (( PLANNED_WALL_S > WALL_CAP_S )); then
  echo "refusing plan: ${PLANNED_WALL_S}s exceeds hard ${WALL_CAP_S}s cap" >&2
  exit 2
fi
for instance in ksg reg3_250 surfacecode_d13; do
  if [[ ! -f "$INSTANCES_DIR/$instance.json" ]]; then
    echo "instances_dir must contain $instance.json" >&2
    exit 2
  fi
done

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  exec env ATT_BENCH_GUARDED=1 python3 "$SUMMARIZER" supervise "$WALL_CAP_S" -- "$0" "$@"
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/omeco-attempt064.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
: > "$OUT"

python3 "$SUMMARIZER" exec "$BUILD_CAP_S" "$TMP_DIR/build.stdout" "$TMP_DIR/build.stderr" -- \
  cargo build --manifest-path "$ROOT/Cargo.toml" --release --offline --example attempt -p omeco

BIN="$ROOT/target/release/examples/attempt"
for instance in ksg reg3_250 surfacecode_d13; do
  graph="$INSTANCES_DIR/$instance.json"
  for arm in event-w event-2w fixed15 fixed25 fixed40; do
    log="$TMP_DIR/${instance}-${arm}.stderr"
    tree="$TMP_DIR/${instance}-${arm}.json"
    case "$arm" in
      event-w)
        parameter=1
        toggles=(ATT_STALL_WINDOW_MULT=1)
        ;;
      event-2w)
        parameter=2
        toggles=(ATT_STALL_WINDOW_MULT=2)
        ;;
      fixed*)
        percent=${arm#fixed}
        parameter=$(python3 -c 'import sys; print(int(sys.argv[1]) / 100)' "$percent")
        toggles=("ATT_FIXED_SWITCH=$parameter")
        ;;
    esac
    python3 "$SUMMARIZER" exec "$RUN_CAP_S" "$TMP_DIR/${instance}-${arm}.stdout" "$log" -- \
      env RAYON_NUM_THREADS=1 ATT_DIAG=1 ATT_MAX_SWEEPS="$SWEEP_CAP" "${toggles[@]}" \
      "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree"
    python3 "$SUMMARIZER" row "$instance" "$arm" "$parameter" "$SWEEP_CAP" "$log" "$tree" >> "$OUT"
  done
done

python3 "$SUMMARIZER" report "$OUT"
