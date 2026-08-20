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
RUN_BUDGET_MS=${ATT_DEV_BUDGET_MS:-15000}
RUN_GUARD_S=5
SUMMARY_ALLOWANCE_S=20
INSTANCES=(reg3_250 surfacecode_d13)
ARMS=(band parent)
RELABEL_SEEDS=(65000 65001 65002)
RUNS=$(( ${#INSTANCES[@]} * ${#ARMS[@]} * ${#RELABEL_SEEDS[@]} ))
RUN_CAP_S=$(( (RUN_BUDGET_MS + 999) / 1000 + RUN_GUARD_S ))
PLANNED_WALL_S=$(( BUILD_CAP_S + RUNS * RUN_CAP_S + SUMMARY_ALLOWANCE_S ))

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  echo "budget plan: build<=${BUILD_CAP_S}s + ${RUNS} serial runs*(budget=${RUN_BUDGET_MS}ms+guard=${RUN_GUARD_S}s) + summarize<=${SUMMARY_ALLOWANCE_S}s = ${PLANNED_WALL_S}s (hard cap ${WALL_CAP_S}s)" >&2
  echo "evidence plan: reg3_250,surfacecode_d13 x band,parent x 3 fixed relabelings; per-sweep trace; RAYON_NUM_THREADS=1; CARGO_BUILD_JOBS=2" >&2
  if (( PLANNED_WALL_S > WALL_CAP_S )); then
    echo "refusing plan: ${PLANNED_WALL_S}s exceeds hard ${WALL_CAP_S}s cap" >&2
    exit 2
  fi
  exec env ATT_BENCH_GUARDED=1 python3 "$SUMMARIZER" supervise "$WALL_CAP_S" -- "$0" "$@"
fi
if (( PLANNED_WALL_S > WALL_CAP_S )); then
  echo "refusing plan: ${PLANNED_WALL_S}s exceeds hard ${WALL_CAP_S}s cap" >&2
  exit 2
fi

for instance in "${INSTANCES[@]}"; do
  if [[ ! -f "$INSTANCES_DIR/$instance.json" ]]; then
    echo "instances_dir must contain $instance.json" >&2
    exit 2
  fi
done

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/omeco-attempt065.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
: > "$OUT"

cd "$ROOT"
python3 "$SUMMARIZER" exec "$BUILD_CAP_S" "$TMP_DIR/build.stdout" "$TMP_DIR/build.stderr" -- \
  env CARGO_BUILD_JOBS=2 cargo build --release --offline --example attempt -p omeco

BIN="$ROOT/target/release/examples/attempt"
for instance in "${INSTANCES[@]}"; do
  source_graph="$INSTANCES_DIR/$instance.json"
  for relabel_index in "${!RELABEL_SEEDS[@]}"; do
    relabel_seed=${RELABEL_SEEDS[$relabel_index]}
    graph="$TMP_DIR/${instance}-r${relabel_index}.json"
    python3 "$SUMMARIZER" relabel "$source_graph" "$graph" "$relabel_seed"
    for arm in "${ARMS[@]}"; do
      trace="$TMP_DIR/${instance}-r${relabel_index}-${arm}.trace.jsonl"
      tree="$TMP_DIR/${instance}-r${relabel_index}-${arm}.tree.json"
      log="$TMP_DIR/${instance}-r${relabel_index}-${arm}.stderr"
      if [[ $arm == parent ]]; then
        python3 "$SUMMARIZER" exec "$RUN_CAP_S" "$TMP_DIR/${instance}-r${relabel_index}-${arm}.stdout" "$log" -- \
          env RAYON_NUM_THREADS=1 ATT_TRACE_PATH="$trace" ATT_PARENT=1 \
          "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree"
      else
        python3 "$SUMMARIZER" exec "$RUN_CAP_S" "$TMP_DIR/${instance}-r${relabel_index}-${arm}.stdout" "$log" -- \
          env RAYON_NUM_THREADS=1 ATT_TRACE_PATH="$trace" \
          "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree"
      fi
      python3 "$SUMMARIZER" annotate \
        "$instance" "$arm" "$relabel_index" "$relabel_seed" "$trace" >> "$OUT"
    done
  done
done

python3 "$SUMMARIZER" report "$OUT"
