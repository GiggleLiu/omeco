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
RUN_BUDGET_MS=${ATT_DEV_BUDGET_MS:-60000}
SWEEP_CAP=${ATT_DEV_SWEEPS:-}
RUN_GUARD_S=3
SUMMARY_ALLOWANCE_S=10
INSTANCES=(reg3_250 ksg)
FRACTIONS=(0.25 0.40)
RELABEL_SEEDS=(66000 66001)
RUNS=$(( ${#INSTANCES[@]} * ${#FRACTIONS[@]} * ${#RELABEL_SEEDS[@]} ))
RUN_CAP_S=$(( (RUN_BUDGET_MS + 999) / 1000 + RUN_GUARD_S ))
PLANNED_WALL_S=$(( BUILD_CAP_S + RUNS * RUN_CAP_S + SUMMARY_ALLOWANCE_S ))

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  echo "budget plan: build<=${BUILD_CAP_S}s + ${RUNS} serial runs*(budget=${RUN_BUDGET_MS}ms+guard=${RUN_GUARD_S}s) + summarize<=${SUMMARY_ALLOWANCE_S}s = ${PLANNED_WALL_S}s (hard cap ${WALL_CAP_S}s)" >&2
  echo "evidence plan: reg3_250,ksg x q={0.25,0.40} x 2 fixed relabelings; wall-matched tc(t) every 40 sweeps; reg3 record-eps=39.883325463011175+0.05=39.933325463011175; ATT_MAX_SWEEPS=${SWEEP_CAP:-unlimited}; RAYON_NUM_THREADS=1; CARGO_BUILD_JOBS=2" >&2
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

if [[ ${ATT_BENCH_GUARDED:-0} != 1 ]]; then
  exec env ATT_BENCH_GUARDED=1 python3 "$SUMMARIZER" supervise "$WALL_CAP_S" -- "$0" "$@"
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/omeco-attempt066.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
: > "$OUT"

python3 "$SUMMARIZER" exec "$BUILD_CAP_S" "$TMP_DIR/build.stdout" "$TMP_DIR/build.stderr" -- \
  env CARGO_BUILD_JOBS=2 cargo build --manifest-path "$ROOT/Cargo.toml" --release --offline --example attempt -p omeco

BIN="$ROOT/target/release/examples/attempt"
for instance in "${INSTANCES[@]}"; do
  source_graph="$INSTANCES_DIR/$instance.json"
  for relabel_index in "${!RELABEL_SEEDS[@]}"; do
    relabel_seed=${RELABEL_SEEDS[$relabel_index]}
    graph="$TMP_DIR/${instance}-r${relabel_index}.json"
    python3 "$SUMMARIZER" relabel "$source_graph" "$graph" "$relabel_seed"
    for fraction in "${FRACTIONS[@]}"; do
      sweep_toggle=()
      if [[ -n $SWEEP_CAP ]]; then
        sweep_toggle=("ATT_MAX_SWEEPS=$SWEEP_CAP")
      fi
      arm="q${fraction/./}"
      log="$TMP_DIR/${instance}-r${relabel_index}-${arm}.stderr"
      tree="$TMP_DIR/${instance}-r${relabel_index}-${arm}.json"
      python3 "$SUMMARIZER" exec "$RUN_CAP_S" "$TMP_DIR/${instance}-r${relabel_index}-${arm}.stdout" "$log" -- \
        env RAYON_NUM_THREADS=1 ATT_DIAG=1 ATT_FIXED_SWITCH="$fraction" "${sweep_toggle[@]}" \
        "$BIN" "$graph" "$RUN_BUDGET_MS" "$tree"
      python3 "$SUMMARIZER" row \
        "$instance" "$relabel_index" "$relabel_seed" "$arm" "$fraction" "$SWEEP_CAP" "$log" "$tree" >> "$OUT"
    done
  done
done

python3 "$SUMMARIZER" report "$OUT"
