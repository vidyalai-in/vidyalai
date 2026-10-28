#!/bin/bash
# On-policy arm (approved extra budget): sample 4 replies per prompt at 64 rows per batch for as many of the
# first 2,000 training prompts as fit in 750 s, rank with the reward model, DPO with the pilot settings,
# full evaluation, reward-model judge.
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh onpolicy_gen2 $S/onpolicy.py 2000 4 384 750 > $LG/onpolicy_gen2.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh onpolicy_ref dpo.py reference --pairs onpolicy > $LG/onpolicy_ref.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh onpolicy_train dpo.py train --pairs onpolicy --out $R/onpolicy --lr 1e-5 --beta 0.1 \
    --accum 13 --warmup 3 --val-every 0 --ckpt-every 1000 --fused > $LG/onpolicy_train.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh eval_onpolicy $S/evaluate.py $R/onpolicy/model_final.pt onpolicy --bench > $LG/eval_onpolicy.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh rm_eval_onpolicy $S/rm_eval.py sft onpolicy --no-calib > $LG/rm_eval_onpolicy.log 2>&1
