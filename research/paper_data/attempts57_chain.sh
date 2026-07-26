#!/bin/bash
# Build attempts 057/058 with the pinned toolchain, then run their official
# campaign (relational family + UAI hard ten). Assumes the relational chain
# has finished so the machine is otherwise idle.
cd ~/omeco_campaign
export PATH=/usr/lib/rust-1.91/bin:$PATH

for a in attempt-057 attempt-058; do
  echo "[build] $a" >> attempts57.log
  (cd worktrees/$a && cargo build --release -p omeco --example attempt \
     >> ~/omeco_campaign/attempts57.log 2>&1) || {
    echo "[build] $a FAILED" >> attempts57.log; exit 1; }
done
echo "[build] done" >> attempts57.log

python3 campaign_attempts57.py --lanes 2 >> attempts57.log 2>&1
echo "ATTEMPTS57-CHAIN-DONE" >> attempts57.log
