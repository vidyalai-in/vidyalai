#!/bin/bash
# Full Haiku sweep: 36 tasks x 3 repeats per arm.
cd /home/amit/.cache/vidyalai-scratch/a03
for arm in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 ../agents/.venv/bin/python run.py $arm 3 > logs/$arm.log 2>&1
  tail -6 logs/$arm.log
done
