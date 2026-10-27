#!/bin/bash
# usage: gpu.sh <name> <train.py args...>
# One short locked GPU job: GPU lock, 6G memory cap, nvidia-smi every second to ../logs/<name>.smi.csv.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
D=/home/amit/.cache/vidyalai-scratch/t08; NAME=$1; shift
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/fromscratch
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c '
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv,noheader -lms 1000 >> "$0/logs/$1.smi.csv" & SMI=$!
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 "$2" train.py "${@:3}" > "$0/logs/$1.out" 2>&1; rc=$?
  kill $SMI; echo "exit $rc" >> "$0/logs/$1.out"; exit $rc' "$D" "$NAME" "$PY" "$@"
