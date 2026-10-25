#!/bin/bash
# Part 11 main runs: k repeats per task per configuration, one configuration at a time (3 workers).
cd /home/amit/.cache/vidyalai-scratch/a11
P="systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python"
$P run.py code-haiku 10
$P run.py code-sonnet 5
$P run.py salt-off 10
$P run.py salt-low 10
