#!/bin/bash
# Three workers pull jobs (one line each: script and arguments) from jobs.txt; one request in flight per worker.
cd /home/amit/.cache/vidyalai-scratch/a12
PY=/home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python
worker() {
  while true; do
    job=$(flock jobs.lock bash -c 'j=$(head -1 jobs.txt); [ -n "$j" ] && sed -i 1d jobs.txt; echo "$j"')
    [ -z "$job" ] && break
    echo "== $(date +%T) $job"
    CONC=1 systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 $PY $job
  done
}
worker > logs/q1.log 2>&1 &
worker > logs/q2.log 2>&1 &
worker > logs/q3.log 2>&1 &
wait
echo QUEUE-DONE
