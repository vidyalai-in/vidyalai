#!/bin/bash
# usage (always under the lock): flock gpu.lock systemd-run ... bash gpu_run.sh <log-tag> <run_eval.py args...>
# Logs clocks and power every second while the evaluation runs.
tag=$1; shift
cd /home/amit/.cache/vidyalai-scratch/t09
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu \
    --format=csv,noheader,nounits -lms 1000 > logs/smi_$tag.csv &
SMI=$!
start=$(date +%s)
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python -u scripts/run_eval.py "$@"; rc=$?
echo "$tag wall $(( $(date +%s) - start )) s" >> logs/gpu_wall.log
kill $SMI; wait $SMI 2>/dev/null
exit $rc
