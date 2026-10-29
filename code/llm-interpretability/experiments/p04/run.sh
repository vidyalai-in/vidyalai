#!/bin/bash
# run.sh <memcap> <script> [args]: run a GPU script under the shared lock, log wall time
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cap=$1; shift; tmo=${TMO:-900}
t0=$(date +%s)
flock /home/amit/.cache/vidyalai-scratch/gpu.lock timeout $tmo systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python "$@"
rc=$?
t1=$(date +%s)
echo "$(date -Is) $* rc=$rc wall=$((t1-t0))s" >> /home/amit/.cache/vidyalai-scratch/p04/notes/gpu_log.txt
