#!/bin/bash
# Exp 1b full run: 40 rounds per size, three arms each (360 calls + 3 priming calls).
cd /home/amit/.cache/vidyalai-scratch/v13
PY=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
for s in S L XL; do
  timeout 1500 systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 $PY scripts/s02_cache.py $s 40 r
done
