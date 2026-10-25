#!/bin/bash
# Experiment 1: three single-threaded workers (one request each in flight), seeds split 1-10, 11-20, 21-30.
cd /home/amit/.cache/vidyalai-scratch/a12
PY=/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python
worker() {
  for n in 10 20 40 80; do
    CONC=1 systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 $PY run_chain.py off xhard $n $1
  done
}
worker 1-10 > logs/w1.log 2>&1 &
worker 11-20 > logs/w2.log 2>&1 &
worker 21-30 > logs/w3.log 2>&1 &
wait
