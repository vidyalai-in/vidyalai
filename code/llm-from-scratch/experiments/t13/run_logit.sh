#!/bin/bash
# Part 13 logit-distillation runs: a plain-CE S15 baseline (Part 5's train.py, unchanged) and KD students.
# Same token stream (first 65.5M training tokens), schedule and seed; only the loss differs.
# usage: bash run_logit.sh ce|kd1|kd2
S=/home/amit/.cache/vidyalai-scratch; R=$S/t13/runs; G=$S/t13/scripts/gpu_run.sh
lock() { flock $S/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 "$@"; }
COMMON="--model s15 --steps 1000 --lr 2.08e-3 --warmup 45 --val-every 50 --ckpt-every 100 --seed 1337 --compile --fused --resume"
case $1 in
  ce)  lock bash $G ce_s15 train.py --out $R/ce_s15 --micro-batch 16 --accum 4 --val-batches 40 $COMMON ;;
  kd1) lock bash $G kd_s15_t1 distill.py --out $R/kd_s15_t1 --temperature 1 --alpha 1 --micro-batch 8 --accum 8 --val-batches 80 $COMMON ;;
  kd2) lock bash $G kd_s15_t2 distill.py --out $R/kd_s15_t2a09 --temperature 2 --alpha 0.9 --micro-batch 8 --accum 8 --val-batches 80 $COMMON --val-every 100 ;;
esac
