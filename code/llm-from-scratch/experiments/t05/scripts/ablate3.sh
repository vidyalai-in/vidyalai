#!/bin/bash
R=/home/amit/.cache/vidyalai-scratch/t05/runs; S=/home/amit/.cache/vidyalai-scratch/t05/scripts
until grep -q "ABLATE2 DONE" /home/amit/.cache/vidyalai-scratch/t05/logs/ablate2.nohup; do sleep 10; done
C="--accum 4 --steps 150 --val-every 75 --ckpt-every 100000 --compile --fused --lr 1.5e-3"
rm -rf $R/b64_15_nowarm; $S/gpu.sh b64_15_nowarm train.py --out $R/b64_15_nowarm $C --warmup 0 > /dev/null 2>&1
rm -rf $R/b64_15_noclip; $S/gpu.sh b64_15_noclip train.py --out $R/b64_15_noclip $C --warmup 15 --grad-clip 0 > /dev/null 2>&1
rm -rf $R/b64_lr_1.5e-3_rep; $S/gpu.sh b64_lr_1.5e-3_rep train.py --out $R/b64_lr_1.5e-3_rep $C --warmup 15 > /dev/null 2>&1
echo ABLATE3 DONE
