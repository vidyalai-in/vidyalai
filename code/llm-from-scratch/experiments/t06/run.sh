#!/bin/bash
# usage: run.sh <logname> <script.py> <args...>
# Takes the GPU lock, caps memory at 4G, logs nvidia-smi every second next to the output.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
D=/home/amit/.cache/vidyalai-scratch/t06; name=$1; script=$2; shift 2
mkdir -p $D/logs
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
  bash -c "
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv,noheader -lms 1000 > $D/logs/$name.smi.csv &
SMI=\$!
timeout ${TMO:-900} /home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python $D/$script $* > $D/logs/$name.out 2>&1
rc=\$?
kill \$SMI
exit \$rc"
