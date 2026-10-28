#!/bin/bash
# On-policy arm: 2,000 prompts x 4 samples from the SFT model, ranked by the reward model; DPO with the
# same settings as pilot_1e-5 (lr 1e-5, beta 0.1, 2,000 pairs, accum 13); quick evaluation; RM evaluation.
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh onpolicy_gen $S/onpolicy.py 2000 4 384 > $LG/onpolicy_gen.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh onpolicy_ref dpo.py reference --pairs onpolicy > $LG/onpolicy_ref.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh onpolicy_train dpo.py train --pairs onpolicy --out $R/onpolicy --lr 1e-5 --beta 0.1 --limit 2000 \
    --accum 13 --warmup 3 --val-every 0 --ckpt-every 1000 --fused > $LG/onpolicy_train.log 2>&1 || exit 1
$LOCK $S/gpu_run.sh eval_onpolicy $S/evaluate.py $R/onpolicy/model_final.pt onpolicy --quick > $LG/eval_onpolicy.log 2>&1
