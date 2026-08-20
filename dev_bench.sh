#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: dev_bench.sh <instances_dir> <out.jsonl>" >&2
    exit 2
fi

instances_dir=$1
out_jsonl=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Put the entire script under an independent hard wall guard. The target host is
# Linux; refusing to run without GNU timeout is safer than claiming a cap that
# cannot be enforced.
if [[ ${ATT_BENCH_INNER:-0} != 1 ]]; then
    timeout_bin=$(command -v timeout || command -v gtimeout || true)
    if [[ -z $timeout_bin ]]; then
        echo "error: GNU timeout is required to enforce the 600 s hard cap" >&2
        exit 2
    fi
    exec "$timeout_bin" --signal=TERM --kill-after=5s 600s \
        env ATT_BENCH_INNER=1 ATT_TIMEOUT_BIN="$timeout_bin" "$0" "$@"
fi

timeout_bin=${ATT_TIMEOUT_BIN:-timeout}

instances=(surfacecode_d13.json ksg.json)
sweep_budgets=(380 620 940)
arms=(continuous parent)
build_cap_s=120
run_cap_s=35
run_budget_ms=32000
overhead_cap_s=20
run_count=$((${#instances[@]} * ${#sweep_budgets[@]} * ${#arms[@]}))
planned_s=$((build_cap_s + run_count * run_cap_s + overhead_cap_s))

echo "budget plan: build<=${build_cap_s}s + ${run_count} runs*${run_cap_s}s + overhead<=${overhead_cap_s}s = ${planned_s}s (hard cap 600s)" >&2
if ((planned_s > 600)); then
    echo "error: planned wall budget exceeds 600 s" >&2
    exit 2
fi

for instance in "${instances[@]}"; do
    if [[ ! -f "$instances_dir/$instance" ]]; then
        echo "error: missing instance $instances_dir/$instance" >&2
        exit 2
    fi
done

"$timeout_bin" "$build_cap_s" cargo build \
    --release --offline --example attempt -p omeco \
    --manifest-path "$repo_root/Cargo.toml"

attempt_bin="$repo_root/target/release/examples/attempt"
bench_tmp=$(mktemp -d "${TMPDIR:-/tmp}/attempt-059.XXXXXX")
trap 'rm -rf -- "$bench_tmp"' EXIT
: >"$out_jsonl"

for instance in "${instances[@]}"; do
    instance_name=${instance%.json}
    for sweep_budget in "${sweep_budgets[@]}"; do
        for arm in "${arms[@]}"; do
            result="$bench_tmp/${instance_name}-${sweep_budget}-${arm}.json"
            stderr_log="$bench_tmp/${instance_name}-${sweep_budget}-${arm}.stderr"
            parent_env=()
            if [[ $arm == parent ]]; then
                parent_env=(ATT_PARENT=1)
            fi
            echo "run instance=$instance_name arm=$arm max_sweeps=$sweep_budget wall<=${run_cap_s}s" >&2
            "$timeout_bin" "$run_cap_s" env \
                RAYON_NUM_THREADS=1 \
                ATT_SEED_MS=0 \
                ATT_MAX_SWEEPS="$sweep_budget" \
                "${parent_env[@]}" \
                "$attempt_bin" "$instances_dir/$instance" "$run_budget_ms" "$result" \
                2>"$stderr_log"
            python3 "$repo_root/summarize.py" --record \
                "$instance_name" "$arm" "$sweep_budget" \
                "$stderr_log" "$result" "$out_jsonl"
        done
    done
done

python3 "$repo_root/summarize.py" "$out_jsonl"
