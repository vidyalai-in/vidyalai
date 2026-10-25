#!/bin/bash
# profiles, then compile modes, optimizer, micro-batch sweep, accumulation, host copies, repeats
D=/home/amit/.cache/vidyalai-scratch/t06
prof() { name=$1; shift; grep -q '^PROF' $D/logs/$name.out 2>/dev/null && return
  export TORCHINDUCTOR_CACHE_DIR=$D/cache/prof TRITON_CACHE_DIR=$D/cache/prof/triton
  $D/run.sh $name prof.py --name $name "$@"; echo "prof done <- $name"; }
prof prof_c4 --compile default --mb 4 --n 2
prof prof_final --compile default --mb 16 --n 2 --opt fused --pin 1 --nb 1
prof prof_final_blocking --compile default --mb 16 --n 2 --opt fused
prof prof_tf32_eager --precision tf32 --mb 4 --n 1
prof prof_final_pageable_nb --compile default --mb 16 --n 2 --opt fused --nb 1
$D/ladder.sh \
 "C_ro --mb 16 --compile reduce-overhead --opt fused" \
 "C_mat --mb 16 --compile max-autotune-no-cudagraphs --opt fused" \
 "C_ma --mb 16 --compile max-autotune --opt fused" \
 "O_forloop --mb 16 --compile default --opt forloop" \
 "M_mb1 --mb 1 --compile default --opt fused --steps 2" \
 "M_mb2 --mb 2 --compile default --opt fused" \
 "M_mb4 --mb 4 --compile default --opt fused" \
 "M_mb8 --mb 8 --compile default --opt fused" \
 "G_acc1 --mb 16 --compile default --opt fused --tokens_per_step 16384 --steps 40" \
 "G_acc4 --mb 16 --compile default --opt fused --tokens_per_step 65536 --steps 12" \
 "D_nb --mb 16 --compile default --opt fused --nb 1" \
 "R_final1 --mb 16 --compile default --opt fused --pin 1 --nb 1 --steps 5"
echo BATCH3_DONE
