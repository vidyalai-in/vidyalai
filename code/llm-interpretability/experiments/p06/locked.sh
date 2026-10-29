#!/bin/bash
# usage: locked.sh <logfile> <timeout_s> <mem> <python args...>  (waits for the GPU lock, runs with a time and memory limit)
log=$1; to=$2; mem=$3; shift 3
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/amit/.cache/vidyalai-scratch/p06
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "s=\$(date +%s); echo LOCK_ACQUIRED \$(date +%T) >> $log; /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $log; timeout $to systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python $* >> $log 2>&1; rc=\$?; e=\$(date +%s); echo EXIT \$rc \$(date +%T) >> $log; echo \"\$(date -Iseconds) $* rc=\$rc locked=\$((e-s))s\" >> notes/gpu_log.txt"
