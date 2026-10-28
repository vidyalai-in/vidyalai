#!/bin/bash
# After the on-policy arm: lite evaluations of the beta sweep (held-out pairs, KL), the <|end|> measurement
# and the reward-model evaluation.
while kill -0 2445389 2>/dev/null; do sleep 5; done
S=/home/amit/.cache/vidyalai-scratch/t11/scripts
R=/home/amit/.cache/vidyalai-scratch/t11/runs
LG=/home/amit/.cache/vidyalai-scratch/t11/logs
LOCK="flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for r in pilot_1e-5 beta_0.05 beta_0.5; do
  $LOCK $S/gpu_run.sh eval_$r $S/evaluate.py $R/$r/model_final.pt $r --lite > $LG/eval_$r.log 2>&1
done
$LOCK $S/gpu_run.sh end_token $S/end_token.py dpo88=$R/dpo88/model_final.pt onpolicy=$R/onpolicy/model_final.pt > $LG/end_token.log 2>&1
$LOCK $S/gpu_run.sh rm_eval $S/rm_eval.py sft dpo88 onpolicy > $LG/rm_eval.log 2>&1
