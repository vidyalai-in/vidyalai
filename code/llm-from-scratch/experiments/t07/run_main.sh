#!/bin/bash
# Part 7: the 88M pretraining run. Run under the GPU lock:
#   flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash run_main.sh [extra train.py args]
# Logs nvidia-smi every second; after a crash it waits 2 minutes and resumes from ckpt.pt (up to 6 tries).
export LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
D=/home/amit/.cache/vidyalai-scratch/t07; OUT=${OUT:-$D/runs/main88}; NAME=$(basename $OUT)
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/fromscratch
mkdir -p $D/logs $OUT
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,enforced.power.limit,temperature.gpu,memory.used,clocks_event_reasons.active --format=csv,noheader -lms 1000 >> $D/logs/$NAME.smi.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
for try in 1 2 3 4 5 6; do
  echo "=== try $try $(date -Is)" >> $D/logs/$NAME.out
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
    $PY train.py --model main --out $OUT --steps 3360 --warmup 150 --lr 6e-4 \
      --val-every 50 --ckpt-every 100 --keep-ckpts \
      --snapshot-at 0,5,10,15,20,25,30,40,50,60,75,90,125,150,250 \
      --compile --fused --resume "$@" >> $D/logs/$NAME.out 2>&1
  echo "=== exit $? $(date -Is)" >> $D/logs/$NAME.out
  grep -q '"event": "end"' $OUT/log.jsonl 2>/dev/null && break
  sleep 120
done
