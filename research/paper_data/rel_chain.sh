#!/bin/bash
# Beyond-the-benchmark chain: our 140-job relational batch first, then the
# Julia baselines under the crash-retry pattern (KaHyPar segfaults, plus a
# 2 h per-launch timeout so a runaway TreeSA rung on a 63k-tensor instance
# is marked failed instead of stalling the chain).
cd ~/omeco_campaign
export JULIA_PKG_SERVER=https://mirror.nju.edu.cn/julia

python3 campaign_relational.py --lanes 2 >> relational.log 2>&1

for i in $(seq 1 40); do
  timeout 7200 julia --project=/root/OMEinsumContractionOrdersBenchmark \
    rel_julia_baselines.jl ~/omeco_campaign/uai_mar \
    ~/omeco_campaign/rel_julia.jsonl >> rel_julia.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "[retry] complete after $i launches" >> rel_julia.log; break; fi
  pair=$(grep -E "Running:" rel_julia.log | tail -1 | sed "s/.*Running: //")
  inst=$(echo "$pair" | awk "{print \$1}")
  opt=$(echo "$pair" | awk "{print \$3}")
  echo "{\"instance\": \"$inst\", \"optimizer\": \"$opt\", \"error\": \"crash-or-timeout rc=$rc\", \"host\": \"huawei-ecs-2core\"}" >> rel_julia.jsonl
  echo "[retry] julia died rc=$rc on $inst/$opt - marked failed, relaunching" >> rel_julia.log
done
echo "RELATIONAL-CHAIN-DONE" >> relational.log
