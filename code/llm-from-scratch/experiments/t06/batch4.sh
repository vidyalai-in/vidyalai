#!/bin/bash
D=/home/amit/.cache/vidyalai-scratch/t06
$D/ladder.sh "C_ro --mb 16 --compile reduce-overhead --opt fused" "C_ma --mb 16 --compile max-autotune --opt fused"
# warm-cache compile: reuse L6_fused's cache directory
if ! grep -q '^RESULT' $D/logs/W_warm.out 2>/dev/null; then
  TORCHINDUCTOR_CACHE_DIR=$D/cache/L6_fused TRITON_CACHE_DIR=$D/cache/L6_fused/triton $D/run.sh W_warm bench.py --name W_warm --mb 16 --compile default --opt fused --steps 2
  echo "W_warm $(grep -o '"first_micro_s": [0-9.]*' $D/logs/W_warm.out) <- W_warm"
fi
$D/run.sh explain explain.py; echo "explain <- $(tail -2 $D/logs/explain.out | tr '\n' ' ')"
TORCHINDUCTOR_CACHE_DIR=$D/cache/prof TRITON_CACHE_DIR=$D/cache/prof/triton $D/run.sh prof_cudnn_e prof.py --name prof_cudnn_e --mb 2 --attn cudnn --n 1; echo "prof done <- prof_cudnn_e"
echo BATCH4_DONE
