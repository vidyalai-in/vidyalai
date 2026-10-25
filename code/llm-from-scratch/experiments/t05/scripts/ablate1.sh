#!/bin/bash
# Ablations on the 22M model at 65,536 tokens per step (16 x 1,024 x 4): an LR sweep, then no warm-up / no clipping.
R=/home/amit/.cache/vidyalai-scratch/t05/runs; S=/home/amit/.cache/vidyalai-scratch/t05/scripts
until grep -q '"event": "end"' $R/main/log.jsonl 2>/dev/null; do sleep 15; done
C="--accum 4 --steps 150 --val-every 75 --ckpt-every 100000 --compile --fused"
for lr in 3e-3 1.2e-2 7.5e-4 6e-3 1.5e-3; do
  rm -rf $R/b64_lr_$lr; $S/gpu.sh b64_lr_$lr train.py --out $R/b64_lr_$lr $C --warmup 15 --lr $lr > /dev/null 2>&1
done
rm -rf $R/b64_nowarm; $S/gpu.sh b64_nowarm train.py --out $R/b64_nowarm $C --warmup 0 --lr 1.2e-2 > /dev/null 2>&1
rm -rf $R/b64_noclip; $S/gpu.sh b64_noclip train.py --out $R/b64_noclip $C --warmup 15 --grad-clip 0 --lr 1.2e-2 > /dev/null 2>&1
rm -rf $R/b64_neither; $S/gpu.sh b64_neither train.py --out $R/b64_neither $C --warmup 0 --grad-clip 0 --lr 1.2e-2 > /dev/null 2>&1
echo ABLATE1 DONE
