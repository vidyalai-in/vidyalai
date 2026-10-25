#!/bin/bash
# usage: main.sh FIRST_REPEAT REPEATS arm...
F=$1; N=$2; shift 2
for arm in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python run.py $arm $N --first-repeat $F --workers 4
done
