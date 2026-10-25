#!/bin/bash
P="systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python"
cd /home/amit/.cache/vidyalai-scratch/a07
$P run.py single-full 2 --jobs 3
$P run.py team-full 2 --jobs 1
$P run.py single-sized 2 --jobs 3
$P run.py team-sized 2 --jobs 1
