#!/bin/bash
# GPU chunk D: aMUSEd-256 generation for the CLIP-score sweep, then timing.
cd /home/amit/.cache/vidyalai-scratch/v08
P="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 timeout 700 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
$P amused_run.py gen 2>&1 | grep -v -i warn
$P amused_run.py time 2>&1 | grep -v -i warn
