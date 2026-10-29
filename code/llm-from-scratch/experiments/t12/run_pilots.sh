#!/bin/bash
# GRPO learning-rate pilots: 40 steps each from the warm-start, beta 0.04, G 8, 32 prompts per step.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for lr in "$@"; do
  timeout 600 $LOCK $S/gpu_run.sh pilot_$lr grpo.py train --init $R/warm4000/model_final.pt --out $R/pilot_$lr --lr $lr \
      --steps 40 --eval-every 10 --show-every 20 --ckpt-every 100 --time-cap 420 --fused > $LG/pilot_$lr.log 2>&1
done
