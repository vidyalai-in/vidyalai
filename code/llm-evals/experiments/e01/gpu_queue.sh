#!/bin/bash
# Runs run_open.py for each model, one at a time, under the GPU lock. Memory rule (editor update):
# models up to ~1.2B params load once host RAM "available" >= 2800 MB, capped at MemoryMax=2500M;
# bigger models need >= 4500 MB and a 4G cap. Usage: gpu_queue.sh small|big <model>...
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
PY=/home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python
S=/home/amit/.cache/vidyalai-scratch/e01
size=$1; shift
if [ "$size" = small ]; then NEED=2800; CAP=2500M; TRIES=240; else NEED=4500; CAP=4G; TRIES=60; fi
for m in "$@"; do
  for t in $(seq 1 $TRIES); do
    avail=$(free -m | awk '/^Mem:/{print $7}')
    [ "$avail" -ge $NEED ] && break
    sleep 30
  done
  echo "$(date +%T) $m avail=${avail}MB need=$NEED cap=$CAP"
  [ "$avail" -ge $NEED ] || { echo "$(date +%T) $m skipped: memory never reached $NEED MB"; continue; }
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 \
    timeout 900 $PY $S/scripts/run_open.py "$m"
  echo "$(date +%T) $m exit=$?"
done
