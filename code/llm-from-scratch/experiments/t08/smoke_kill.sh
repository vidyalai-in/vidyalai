#!/bin/bash
# Smoke-test helper: once smoke/s15 has logged step 15, kill its train.py once (simulated crash).
L=/home/amit/.cache/vidyalai-scratch/t08/smoke/s15/log.jsonl
for i in $(seq 1 600); do
  grep -q '"step": 15, "loss"' $L 2>/dev/null && break; sleep 1
done
PID=$(pgrep -f "[t]rain.py --model s15 --out /home/amit/.cache/vidyalai-scratch/t08/smoke" | head -1)
echo "killing $PID at $(date -Is)"; kill -9 $PID
