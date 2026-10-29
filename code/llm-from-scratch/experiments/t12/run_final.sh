#!/bin/bash
# After the ablations: the Part 10 SFT model's evaluation, the working audit, quick evaluations of the ablations.
S=/home/amit/.cache/vidyalai-scratch/t12/scripts; R=/home/amit/.cache/vidyalai-scratch/t12/runs; LG=/home/amit/.cache/vidyalai-scratch/t12/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
until grep -q "abl_contains wall" $LG/gpu_wall.log; do sleep 10; done
timeout 900 $LOCK $S/gpu_run.sh eval_sft $S/evaluate.py /home/amit/.cache/vidyalai-scratch/t10/runs/sft88/model_final.pt sft --n 16 --chat > $LG/eval_sft.log 2>&1
for t in warm4000:warm grpo_main:main abl_contains:contains; do
  timeout 300 $LOCK $S/gpu_run.sh steps_${t#*:} $S/check_steps.py $R/${t%:*}/model_final.pt ${t#*:} 8 > $LG/steps_${t#*:}.log 2>&1
done
for a in kl0 drgrpo updates4 contains; do
  timeout 400 $LOCK $S/gpu_run.sh eval_$a $S/evaluate.py $R/abl_$a/model_final.pt $a --n 16 > $LG/eval_$a.log 2>&1
done
