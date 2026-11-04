#!/bin/bash
# usage: gpu_multi.sh <label> "<shell commands>"  -- runs under the shared GPU lock, logs locked seconds
cd /home/amit/.cache/vidyalai-scratch/v05
label=$1; cmd=$2
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c '
t0=$(date +%s); echo "start $(date +%T) '"$label"'"
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -E LD_LIBRARY_PATH=/usr/lib/wsl/lib bash -c "$0"
rc=$?; t1=$(date +%s); echo "'"$label"' $t0 $t1 $(( (t1-t0) )) rc=$rc" >> gpu_minutes.log; echo "end $(date +%T) rc=$rc"' "$cmd"
