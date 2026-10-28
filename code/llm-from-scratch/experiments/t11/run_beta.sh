#!/bin/bash
# Beta sweep: lr 1e-5, first 2,000 pairs, one epoch (beta 0.1 is pilot_1e-5), then quick evaluations.
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for b in 0.05 0.5; do
  $LOCK $S/gpu_run.sh beta_$b dpo.py train --out $R/beta_$b --lr 1e-5 --beta $b --limit 2000 --accum 13 \
      --warmup 3 --val-every 0 --ckpt-every 1000 --fused > $LG/beta_$b.log 2>&1
done
for r in pilot_1e-5 beta_0.05 beta_0.5; do
  $LOCK $S/gpu_run.sh eval_$r $S/evaluate.py $R/$r/model_final.pt $r --quick > $LG/eval_$r.log 2>&1
done
