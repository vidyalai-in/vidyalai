#!/bin/bash
# usage: gpu.sh <tag> <cmd...>   (run inside flock + systemd-run). Logs SM clock/power every second and locked time.
cd /home/amit/.cache/vidyalai-scratch/v07
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
tag=$1; shift
s=$(date +%s)
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > runs/smi-$tag.csv 2>&1 &
SMI=$!
"$@"; rc=$?
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
e=$(date +%s)
echo "$tag start=$s end=$e minutes=$(( (e-s) ))s rc=$rc" >> gpu_minutes.log
exit $rc
