#!/bin/bash
# Runs the CPU stages of the Part 3 pipeline after the download finishes. Each stage resumes per part.
cd /home/amit/.cache/vidyalai-scratch/fromscratch
PY=/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python
L=/home/amit/.cache/vidyalai-scratch/t03/logs
until grep -q "download done" $L/sample_download.log; do sleep 10; done
for stage in minhash tokenize dedup; do
  echo "== $stage $(date)" >> $L/pipeline.log
  /usr/bin/time -v systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 $PY data.py $stage > $L/$stage.log 2>&1
  echo "== $stage exit $? $(date)" >> $L/pipeline.log
done
