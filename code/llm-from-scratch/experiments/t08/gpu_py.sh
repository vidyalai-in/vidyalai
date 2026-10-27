#!/bin/bash
# usage: gpu_py.sh <name> <script.py> [args...]  -- a short locked GPU Python job, 6G cap, nvidia-smi logged
export LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
D=/home/amit/.cache/vidyalai-scratch/t08; NAME=$1; shift
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/fromscratch
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c '
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used --format=csv,noheader -lms 1000 >> "$0/post/$1.smi.csv" & SMI=$!
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 "$2" "${@:3}"; rc=$?
  kill $SMI; exit $rc' "$D" "$NAME" "$PY" "$@"
