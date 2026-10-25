#!/bin/bash
# LR sweep on the 22M model: 50 steps of 524,288 tokens (26.2M tokens), warm-up 5, cosine to 10%.
R=/home/amit/.cache/vidyalai-scratch/t05/runs; S=/home/amit/.cache/vidyalai-scratch/t05/scripts
for lr in 3e-3 1.2e-2 7.5e-4 6e-3 1.5e-3; do
  rm -rf $R/lr_$lr
  $S/gpu.sh lr_$lr train.py --out $R/lr_$lr --steps 50 --warmup 5 --lr $lr --val-every 25 --ckpt-every 1000 --compile --fused > /dev/null 2>&1
done
echo SWEEP DONE
