#!/bin/bash
# usage: run.sh <logname> <python args...>  (from fromscratch/; logs nvidia-smi every 1 s alongside)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
D=/home/amit/.cache/vidyalai-scratch/t07/post/logs; name=$1; shift
cd /home/amit/.cache/vidyalai-scratch/fromscratch
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used --format=csv,noheader -lms 1000 > $D/$name.smi.csv &
SMI=$!
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python "$@" > $D/$name.out 2>&1
rc=$?
kill $SMI
exit $rc
