#!/bin/bash
# Score one open model under the GPU lock and a 4 GB memory cap, only if >= 4,500 MB is available.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
avail=$(free -m | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 4500 ]; then echo "only $avail MB available, not loading $1"; exit 3; fi
cd /home/amit/.cache/vidyalai-scratch/e02
flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
  timeout 900 /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python scripts/score_open.py "$1" "${2:-16}"
