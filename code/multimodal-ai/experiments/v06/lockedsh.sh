#!/bin/bash
# Usage: lockedsh.sh <label> <mem> "<shell commands>"; logs locked seconds to gpu_minutes.log
label=$1; mem=$2; cmd=$3
cd /home/amit/.cache/vidyalai-scratch/v06
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
t0=$(date +%s)
flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 -E LD_LIBRARY_PATH=/usr/lib/wsl/lib -E HF_HOME=/home/amit/.cache/vidyalai-scratch/hf bash -c "cd /home/amit/.cache/vidyalai-scratch/v06; s=\$(date +%s); $cmd; rc=\$?; e=\$(date +%s); echo \"$label locked_s=\$((e-s)) waited_s=\$((s-$t0)) rc=\$rc\" >> gpu_minutes.log"
tail -1 /home/amit/.cache/vidyalai-scratch/v06/gpu_minutes.log
