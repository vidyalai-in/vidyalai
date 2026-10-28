#!/bin/bash
# The main DPO run (beta 0.1, lr 1e-5, 16,000 pairs, one epoch), then its full evaluation with benchmarks.
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh dpo88 dpo.py train --out $R/dpo88 --lr 1e-5 --beta 0.1 --limit 16000 --epochs 1 --budget 4096 \
    --accum 13 --warmup 10 --val-every 25 --ckpt-every 50 --fused > $LG/dpo88.log 2>&1
$LOCK $S/gpu_run.sh eval_dpo88 $S/evaluate.py $R/dpo88/model_final.pt dpo88 --bench > $LG/eval_dpo88.log 2>&1
