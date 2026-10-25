#!/bin/bash
# micro-batch search: one process per (size, mb); GPU lock taken per run
D=/home/amit/.cache/vidyalai-scratch/t01
for size in t20 t50 t85 t125; do
  for mb in 4 8 16 32 64; do
    timeout 600 flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 $D/run.sh mb_${size}_${mb} --size $size --mb $mb --micro_only 14
    if ! grep -q RESULT $D/logs/mb_${size}_${mb}.out; then echo "$size mb=$mb failed"; break; fi
    grep RESULT $D/logs/mb_${size}_${mb}.out
  done
done
