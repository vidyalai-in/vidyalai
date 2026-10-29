#!/bin/bash
# The main GRPO run: lr 1e-5, beta 0.04, G 8, 32 problems per step, 250 steps, from the warm start.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
timeout 2400 $LOCK $S/gpu_run.sh grpo_main grpo.py train --init $R/warm4000/model_final.pt --out $R/grpo_main \
    --lr 1e-5 --kl 0.04 --steps 250 --eval-every 25 --show-every 25 --ckpt-every 25 --time-cap 1800 --fused $@ > $LG/grpo_main.log 2>&1
