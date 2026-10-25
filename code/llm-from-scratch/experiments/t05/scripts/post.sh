#!/bin/bash
R=/home/amit/.cache/vidyalai-scratch/t05/runs; S=/home/amit/.cache/vidyalai-scratch/t05/scripts
$S/gpu.sh post_main $S/post_main.py $R/main/ckpt.pt > /dev/null 2>&1
$S/gpu.sh bf16_final $S/bf16_grads.py $R/main/ckpt.pt > /dev/null 2>&1
rm -rf $R/main88
$S/gpu.sh main88 train.py --model main --out $R/main88 --steps 3360 --warmup 100 --lr 1e-3 --stop-at 4 --val-every 1000 --val-batches 8 --ckpt-every 100000 --compile --fused > /dev/null 2>&1
echo POST DONE
