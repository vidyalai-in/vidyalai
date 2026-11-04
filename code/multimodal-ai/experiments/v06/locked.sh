#!/bin/bash
# Usage: locked.sh <label> <mem e.g. 4G> <timeout s> <python args...>; logs locked seconds to gpu_minutes.log
label=$1; mem=$2; to=$3; shift 3
cd /home/amit/.cache/vidyalai-scratch/v06
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
t0=$(date +%s)
flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 bash -c "s=\$(date +%s); timeout $to /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python $*; rc=\$?; e=\$(date +%s); echo \"$label locked_s=\$((e-s)) waited_s=\$((s-$t0)) rc=\$rc\" >> /home/amit/.cache/vidyalai-scratch/v06/gpu_minutes.log"
tail -1 /home/amit/.cache/vidyalai-scratch/v06/gpu_minutes.log
