#!/bin/bash
# Quick evaluations of the three beta-sweep models (beta 0.1 is pilot_1e-5).
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for r in pilot_1e-5 beta_0.05 beta_0.5; do
  $LOCK $S/gpu_run.sh eval_$r $S/evaluate.py $R/$r/model_final.pt $r --quick > $LG/eval_$r.log 2>&1
done
