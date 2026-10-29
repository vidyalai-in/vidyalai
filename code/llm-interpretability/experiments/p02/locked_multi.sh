#!/bin/bash
# usage: locked_multi.sh <logfile> <timeout_s_each> "<python args 1>" "<python args 2>" ...
# Waits for the GPU lock once, then runs each python job in turn (each with its own time limit).
log=$1; to=$2; shift 2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/amit/.cache/vidyalai-scratch/p02
exec 9>/home/amit/.cache/vidyalai-scratch/gpu.lock
flock 9
echo LOCK_ACQUIRED $(date +%T) >> $log
for job in "$@"; do
  echo JOB $job $(date +%T) >> $log
  timeout $to systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python $job >> $log 2>&1
  echo EXIT $? $(date +%T) >> $log
done
echo ALL_DONE $(date +%T) >> $log
