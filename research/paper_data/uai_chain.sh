#!/bin/bash
# Chain: wait for the matched-budget ladder to finish, then run the UAI batch,
# then the UAI Julia baselines. All resumable.
cd ~/omeco_campaign
while pgrep -f "julia_matched_budget_v2" > /dev/null; do sleep 60; done
echo "[chain] ladder done, starting UAI campaign at $(date)"
python3 campaign_uai.py --lanes 2 >> uai_campaign.log 2>&1
echo "[chain] UAI campaign done, starting Julia baselines at $(date)"
export JULIA_PKG_SERVER=https://mirror.nju.edu.cn/julia
julia --project=/root/OMEinsumContractionOrdersBenchmark uai_julia_baselines.jl ~/omeco_campaign/uai_mar ~/omeco_campaign/uai_julia.jsonl >> uai_julia.log 2>&1
echo "[chain] ALL DONE at $(date)"
