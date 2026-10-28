#!/bin/bash
# Budget trim: after the on-policy full evaluation (PID 2491980), a lite evaluation of the off-policy control
# (held-out pairs and KL) and the reward-model judge for the on-policy model.
while kill -0 2491980 2>/dev/null; do sleep 5; done
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh eval_offpolicy704 $S/evaluate.py $R/offpolicy704/model_final.pt offpolicy704 --lite > $LG/eval_offpolicy_ctrl.log 2>&1
$LOCK $S/gpu_run.sh rm_eval_onpolicy $S/rm_eval.py sft onpolicy --no-calib > $LG/rm_eval_onpolicy.log 2>&1
rm -f $R/onpolicy/ckpt.pt $R/offpolicy704/ckpt.pt
