#!/bin/bash
# Interleave arms repeat by repeat so time-of-day latency drift spreads over all arms.
cd /home/amit/.cache/vidyalai-scratch/a15
PY=/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python
run() { systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 $PY run.py "$@"; }
for k in 1 2 3; do
  for arm in h-off h-low h-med h-high h-med-nocache; do run $arm 1 --first-repeat $k; done
  if [ $k -le 2 ]; then
    run s-med 1 --first-repeat $k
    run team-s 1 --first-repeat $k --jobs 1
  fi
done
