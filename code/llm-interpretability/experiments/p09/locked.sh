#!/bin/bash
# usage: locked.sh <logfile> <timeout_s> <memcap> <python args...>  (waits for the GPU lock, then runs with a time limit inside it)
log=$1; to=$2; mem=$3; shift 3
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/amit/.cache/vidyalai-scratch/p09
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "echo LOCK_ACQUIRED \$(date +%T) >> $log; timeout $to systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python $* >> $log 2>&1; echo EXIT \$? \$(date +%T) >> $log"
