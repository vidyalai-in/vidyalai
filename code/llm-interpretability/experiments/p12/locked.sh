#!/bin/bash
# usage: locked.sh <memcap> <timeout_s> <logfile> <cmd...>   (timeout inside the lock)
CAP=$1; TO=$2; LOG=$3; shift 3
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
cd /home/amit/.cache/vidyalai-scratch/p12
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "echo LOCK_ACQ \$(date +%s) >> $LOG; systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 timeout $TO /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python $* >> $LOG 2>&1; echo EXIT \$? LOCK_REL \$(date +%s) >> $LOG"
