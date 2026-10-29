#!/bin/bash
# usage (always under the lock): flock gpu.lock systemd-run ... bash gpu_run.sh <log-tag> <python args...>
# Runs python from the shared code folder while nvidia-smi logs clocks and power every second.
tag=$1; shift
cd /home/amit/.cache/vidyalai-scratch/fromscratch
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/amit/.cache/vidyalai-scratch/t12/logs
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used \
    --format=csv,noheader,nounits -lms 1000 > $L/smi_$tag.csv &
SMI=$!
start=$(date +%s)
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python -u "$@"; rc=$?
echo "$tag wall $(( $(date +%s) - start )) s rc $rc" >> $L/gpu_wall.log
kill $SMI; wait $SMI 2>/dev/null
exit $rc
