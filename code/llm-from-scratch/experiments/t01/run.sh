#!/bin/bash
# usage: run.sh <logname> <python args...>   (logs nvidia-smi every 1 s alongside)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
D=/home/amit/.cache/vidyalai-scratch/t01; name=$1; shift
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,enforced.power.limit,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv,noheader -lms 1000 > $D/logs/$name.smi.csv &
SMI=$!
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python $D/proto.py "$@" > $D/logs/$name.out 2>&1
rc=$?
kill $SMI
exit $rc
