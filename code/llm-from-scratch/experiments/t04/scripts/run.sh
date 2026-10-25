#!/bin/bash
# usage: run.sh <logname> <script.py> <args...>   (logs nvidia-smi every 1 s alongside)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
D=/home/amit/.cache/vidyalai-scratch/t04; name=$1; shift
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader -lms 1000 > $D/logs/$name.smi.csv &
SMI=$!
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python "$@" 2>&1 | tee $D/logs/$name.out
rc=${PIPESTATUS[0]}
kill $SMI
exit $rc
