#!/bin/bash
# Part 13, after the logit runs: three SFT runs on the sequence-distillation sets, then every evaluation.
S=/home/amit/.cache/vidyalai-scratch; R=$S/t13/runs; G=$S/t13/scripts/gpu_run.sh; D=$S/t13/data; P=$S/t13/scripts
lock() { flock $S/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 "$@"; }
for set in orig qwen_filt qwen_raw; do
  lock bash $G sft_$set -c "from distill import sft_train; sft_train('$D/seq/$set', ['--out', '$R/sft_$set', '--lr', '3e-4', '--epochs', '3', '--budget', '4096', '--accum', '8', '--warmup', '10', '--val-every', '25', '--ckpt-every', '50', '--fused'])"
done
for run in ce_s15 kd_s15_t1 kd_s15_t2a09; do
  lock bash $G eval_$run $P/eval_students.py $run $R/$run/model_final.pt
done
lock bash $G eval_sft88 $P/eval_sft.py sft88 $S/t10/runs/sft88/model_final.pt
lock bash $G eval_sft_orig $P/eval_sft.py orig $R/sft_orig/model_final.pt --bench
lock bash $G eval_sft_qwen_filt $P/eval_sft.py qwen_filt $R/sft_qwen_filt/model_final.pt --bench
lock bash $G eval_sft_qwen_raw $P/eval_sft.py qwen_raw $R/sft_qwen_raw/model_final.pt
