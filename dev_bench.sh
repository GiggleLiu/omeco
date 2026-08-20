#!/usr/bin/env bash
set -euo pipefail

# Enforce the total wall cap from a parent process. The 594 s run timeout plus
# a bounded TERM/KILL grace remains strictly below the 600 s contract.
if [[ "${ATT_BENCH_INNER:-0}" != "1" ]]; then
    exec python3 - "$0" "$@" <<'PY'
import os
import signal
import subprocess
import sys

env = os.environ.copy()
env["ATT_BENCH_INNER"] = "1"
proc = subprocess.Popen(
    [sys.argv[1], *sys.argv[2:]],
    env=env,
    start_new_session=True,
)
try:
    raise SystemExit(proc.wait(timeout=594))
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=1)
    print("dev_bench: HARD 600 s wall cap reached", file=sys.stderr)
    raise SystemExit(124)
PY
fi

if [[ "$#" -ne 2 ]]; then
    echo "usage: dev_bench.sh <instances_dir> <out.jsonl>" >&2
    exit 2
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
instances_dir="$1"
out_jsonl="$2"
instances=("ksg.json" "surfacecode_d13.json")
budgets_ms=(30000 75000)
modes=("active" "parent")

candidate_seconds=0
for _instance in "${instances[@]}"; do
    for budget_ms in "${budgets_ms[@]}"; do
        for _mode in "${modes[@]}"; do
            candidate_seconds=$((candidate_seconds + budget_ms / 1000))
        done
    done
done
build_allowance=120
postprocess_allowance=30
safety_reserve=30
planned_seconds=$((candidate_seconds + build_allowance + postprocess_allowance + safety_reserve))

echo "attempt-060 dev benchmark budget plan:" >&2
echo "  instances: ${instances[*]}" >&2
echo "  budgets: 30 s, 75 s; modes: active, parent" >&2
echo "  candidate wall: ${candidate_seconds} s" >&2
echo "  build allowance: ${build_allowance} s" >&2
echo "  postprocess allowance: ${postprocess_allowance} s" >&2
echo "  safety reserve: ${safety_reserve} s" >&2
echo "  planned total: ${planned_seconds} s / hard cap 600 s (outer kill at 594 s)" >&2
if (( planned_seconds > 600 )); then
    echo "dev_bench: refusing plan above 600 s" >&2
    exit 2
fi

if [[ ! -d "$instances_dir" ]]; then
    echo "dev_bench: not a directory: $instances_dir" >&2
    exit 2
fi
instances_dir="$(CDPATH= cd -- "$instances_dir" && pwd)"
out_dir="$(dirname -- "$out_jsonl")"
if [[ ! -d "$out_dir" ]]; then
    echo "dev_bench: output directory does not exist: $out_dir" >&2
    exit 2
fi
out_jsonl="$(CDPATH= cd -- "$out_dir" && pwd)/$(basename -- "$out_jsonl")"

for instance in "${instances[@]}"; do
    if [[ ! -f "$instances_dir/$instance" ]]; then
        echo "dev_bench: missing $instances_dir/$instance" >&2
        exit 2
    fi
done

cd "$script_dir"
cargo build --release --offline --example attempt -p omeco
binary="$script_dir/target/release/examples/attempt"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/attempt060.XXXXXX")"
trap 'rm -rf -- "$tmp_dir"' EXIT
: > "$out_jsonl"

for instance in "${instances[@]}"; do
    for budget_ms in "${budgets_ms[@]}"; do
        for mode in "${modes[@]}"; do
            stem="${instance%.json}-${budget_ms}-${mode}"
            tree="$tmp_dir/$stem.json"
            stderr="$tmp_dir/$stem.stderr"
            run_limit=$((budget_ms / 1000 + 8))
            echo "run instance=$instance budget_ms=$budget_ms mode=$mode limit=${run_limit}s" >&2
            if [[ "$mode" == "parent" ]]; then
                parent=1
            else
                parent=0
            fi
            if ! timeout --foreground --kill-after=2s "${run_limit}s" \
                env ATT_PARENT="$parent" RAYON_NUM_THREADS=1 \
                "$binary" "$instances_dir/$instance" "$budget_ms" "$tree" \
                2> "$stderr"; then
                echo "dev_bench: candidate failed for $stem" >&2
                tail -40 "$stderr" >&2
                exit 1
            fi
            python3 "$script_dir/summarize.py" \
                --instance "$instance" \
                --mode "$mode" \
                --budget-ms "$budget_ms" \
                --stderr "$stderr" \
                --tree "$tree" >> "$out_jsonl"
        done
    done
done

python3 "$script_dir/summarize.py" --report "$out_jsonl"
