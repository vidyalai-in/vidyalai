#!/bin/bash
# GRPO from the mixed warm-start with the main run's settings, then its evaluation.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
timeout 2000 $LOCK $S/gpu_run.sh grpo_mixed grpo.py train --init $R/warm_mixed/model_final.pt --out $R/grpo_mixed \
    --lr 1e-5 --kl 0.04 --steps 250 --eval-every 25 --show-every 25 --ckpt-every 25 --time-cap 1500 --fused > $LG/grpo_mixed.log 2>&1 || exit 1
timeout 900 $LOCK $S/gpu_run.sh eval_grpo_mixed $S/evaluate.py $R/grpo_mixed/model_final.pt grpo_mixed --n 64 --chat --bench > $LG/eval_grpo_mixed.log 2>&1
