#!/bin/bash
# The fix: warm-start on 4,000 sums plus 4,000 of Part 10's smol-smoltalk conversations, evaluate, GRPO, evaluate.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
timeout 900 $LOCK $S/gpu_run.sh warm_mixed grpo.py warmstart --out $R/warm_mixed --n 4000 --chat 4000 --batch 128 \
    --lr 3e-4 --warmup 5 --fused > $LG/warm_mixed.log 2>&1 || exit 1
timeout 900 $LOCK $S/gpu_run.sh eval_warm_mixed $S/evaluate.py $R/warm_mixed/model_final.pt warm_mixed --n 64 --chat --bench > $LG/eval_warm_mixed.log 2>&1
