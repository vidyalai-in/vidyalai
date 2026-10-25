#!/bin/bash
# Stress catalogs (145, 300, 486 tools; wide system prompt), Haiku 5.5, after the main sweep.
PY=/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/a06
until grep -q TOTAL runs/sweep_haiku.log; do sleep 10; done
run() { systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 $PY run.py "$@" --tag stress; }
for n in 145 300 486; do run all $n --catalog $n --reps 2; run search_bm25 $n --catalog $n --reps 2; done
$PY /home/amit/.cache/vidyalai-scratch/agents/lab/ledger.py
