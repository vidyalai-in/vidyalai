#!/bin/bash
# Final measurements without the on-policy arm (cut for the GPU budget): <|end|> log-probs, reward-model
# evaluation of SFT vs DPO replies, then lite evaluations of the beta sweep (held-out pairs and KL).
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
$LOCK $S/gpu_run.sh end_token $S/end_token.py dpo88=$R/dpo88/model_final.pt > $LG/end_token.log 2>&1
$LOCK $S/gpu_run.sh rm_eval $S/rm_eval.py sft dpo88 > $LG/rm_eval.log 2>&1
for r in pilot_1e-5 beta_0.05 beta_0.5; do
  $LOCK $S/gpu_run.sh eval_$r $S/evaluate.py $R/$r/model_final.pt $r --lite > $LG/eval_$r.log 2>&1
done
