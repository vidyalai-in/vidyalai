#!/bin/bash
# Ablations, 100 steps each, otherwise the main run's settings. usage: run_ablations.sh <name> ...
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
declare -A EXTRA=( [kl0]="--kl 0" [drgrpo]="--no-std --agg token-sum" [updates4]="--updates 4" [contains]="--reward contains"
                   [g16]="--group 16 --prompts 16" )
for name in "$@"; do
  timeout 1200 $LOCK $S/gpu_run.sh abl_$name grpo.py train --init $R/warm4000/model_final.pt --out $R/abl_$name \
      --lr 1e-5 --kl 0.04 --steps 100 --eval-every 25 --show-every 25 --ckpt-every 25 --time-cap 900 --fused ${EXTRA[$name]} \
      > $LG/abl_$name.log 2>&1
done
