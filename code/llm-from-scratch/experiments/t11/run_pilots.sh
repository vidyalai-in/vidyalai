#!/bin/bash
# Off-policy measurement, then four learning-rate pilots (beta 0.1, first 2,000 pairs, one epoch).
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh offpolicy $S/offpolicy.py > /home/amit/.cache/vidyalai-scratch/t11/logs/offpolicy.log 2>&1
for lr in 1e-6 3e-6 1e-5 3e-5; do
  $LOCK $S/gpu_run.sh pilot_$lr dpo.py train --out $R/pilot_$lr --lr $lr --beta 0.1 --limit 2000 --accum 13 \
      --warmup 3 --val-every 0 --ckpt-every 1000 --fused > /home/amit/.cache/vidyalai-scratch/t11/logs/pilot_$lr.log 2>&1
done
