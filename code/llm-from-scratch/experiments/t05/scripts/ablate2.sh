#!/bin/bash
R=/home/amit/.cache/vidyalai-scratch/t05/runs; S=/home/amit/.cache/vidyalai-scratch/t05/scripts
C="--accum 4 --steps 150 --val-every 75 --ckpt-every 100000 --compile --fused"
rm -rf $R/b64_lr_3.75e-4; $S/gpu.sh b64_lr_3.75e-4 train.py --out $R/b64_lr_3.75e-4 $C --warmup 15 --lr 3.75e-4 > /dev/null 2>&1
rm -rf $R/b64_lr_1.2e-2_rep; $S/gpu.sh b64_lr_1.2e-2_rep train.py --out $R/b64_lr_1.2e-2_rep $C --warmup 15 --lr 1.2e-2 > /dev/null 2>&1
rm -rf $R/b64_400; $S/gpu.sh b64_400 train.py --out $R/b64_400 --accum 4 --steps 400 --warmup 40 --lr 7.5e-4 --val-every 100 --ckpt-every 100000 --compile --fused > /dev/null 2>&1
echo ABLATE2 DONE
