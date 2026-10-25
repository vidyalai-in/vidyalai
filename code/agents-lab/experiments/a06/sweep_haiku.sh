#!/bin/bash
# Haiku 5.5 sweep, resumable (re-running skips finished rows).
PY=/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python
cd /home/amit/.cache/vidyalai-scratch/a06
run() { systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 $PY run.py "$@" --tag main; }
for n in 5 20 50 100 145; do run all $n --reps 3; done
for n in 20 50 100 145; do run search_bm25 $n --reps 3; done
run search_regex 145 --reps 2
for n in 20 50 100 145; do run client_bm25 $n --reps 2; done
$PY /home/amit/.cache/vidyalai-scratch/agents/lab/ledger.py
