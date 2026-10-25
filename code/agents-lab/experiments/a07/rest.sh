#!/bin/bash
P="systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python"
cd /home/amit/.cache/vidyalai-scratch/a07
$P run.py single-sized 2 --set stress --jobs 3
$P run.py team-sized 2 --set stress --jobs 1
$P run.py sonnet-team-full 1 w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 --jobs 1
