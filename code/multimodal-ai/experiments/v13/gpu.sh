#!/bin/bash
# Run one GPU job under the lock with a 4 GB memory cap; log locked seconds.
cd /home/amit/.cache/vidyalai-scratch/v13
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
free -m | awk 'NR==2{print "available MB:", $7}'
t0=$(date +%s)
flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
  -E LD_LIBRARY_PATH -E HF_HOME -E HF_HUB_OFFLINE timeout 840 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python "$@"
rc=$?
t1=$(date +%s)
echo "$(date -Is) $* locked $((t1-t0)) s rc=$rc" >> runs/gpu_minutes.log
echo "locked $((t1-t0)) s rc=$rc"
