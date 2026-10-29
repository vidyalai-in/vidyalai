#!/bin/bash
# After the main run: evaluate the starting points, then the ablations.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
until grep -q "grpo_main wall" $LG/gpu_wall.log; do sleep 10; done
timeout 1500 $LOCK $S/gpu_run.sh eval_warm $S/evaluate.py $R/warm4000/model_final.pt warm --n 64 --chat --bench > $LG/eval_warm.log 2>&1
timeout 1500 $LOCK $S/gpu_run.sh eval_main $S/evaluate.py $R/grpo_main/model_final.pt main --n 64 --chat --bench > $LG/eval_main.log 2>&1
$S/run_ablations.sh kl0 drgrpo updates4 contains
