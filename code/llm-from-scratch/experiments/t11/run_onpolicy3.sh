#!/bin/bash
# After generation (PID 2479158): reference, on-policy DPO (704 pairs x 3 epochs, pilot lr/beta/warm-up), an
# off-policy control on the same 704 prompts with identical settings, evaluations, reward-model judge.
while kill -0 2479158 2>/dev/null; do sleep 5; done
grep -q "^onpolicy_gen2 .* rc 0" /home/amit/.cache/vidyalai-scratch/t11/logs/gpu_wall.log || exit 1
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh onpolicy_ref dpo.py reference --pairs onpolicy > $LG/onpolicy_ref.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh onpolicy_train dpo.py train --pairs onpolicy --out $R/onpolicy --lr 1e-5 --beta 0.1 --epochs 3 \
    --accum 13 --warmup 3 --val-every 0 --ckpt-every 1000 --fused > $LG/onpolicy_train.log 2>&1 || exit 1
N=$(python3 -c "import json;print(json.load(open('/home/amit/.cache/vidyalai-scratch/t11/post/onpolicy_data.json'))['prompts'])")
$LOCK $S/gpu_run.sh offpolicy_ctrl dpo.py train --out $R/offpolicy$N --limit $N --lr 1e-5 --beta 0.1 --epochs 3 \
    --accum 13 --warmup 3 --val-every 0 --ckpt-every 1000 --fused > $LG/offpolicy_ctrl.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh eval_onpolicy $S/evaluate.py $R/onpolicy/model_final.pt onpolicy --bench > $LG/eval_onpolicy.log 2>&1
$LOCK $S/gpu_run.sh eval_offpolicy$N $S/evaluate.py $R/offpolicy$N/model_final.pt offpolicy$N --quick > $LG/eval_offpolicy_ctrl.log 2>&1
$LOCK $S/gpu_run.sh rm_eval_onpolicy $S/rm_eval.py sft onpolicy offpolicy$N --no-calib > $LG/rm_eval_onpolicy.log 2>&1
rm -f $R/onpolicy/ckpt.pt $R/offpolicy$N/ckpt.pt
