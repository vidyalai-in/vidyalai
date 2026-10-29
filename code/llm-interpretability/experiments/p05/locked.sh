#!/bin/bash
# Usage: locked.sh <memcap> <logfile> <python args...>; records locked wall time in ../notes/gpu_time.txt
cap=$1; log=$2; shift 2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
cd /home/amit/.cache/vidyalai-scratch/p05/scripts
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "s=\$(date +%s); timeout ${PT:-840} systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python $* > $log 2>&1; rc=\$?; e=\$(date +%s); echo \"\$(date -Is) \$((e-s))s rc=\$rc $*\" >> ../notes/gpu_time.txt; exit \$rc"
