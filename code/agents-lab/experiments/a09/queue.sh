#!/bin/bash
# usage: queue.sh <pid-to-wait-for> <run.py args...> ; waits, then runs one job
cd /home/amit/.cache/vidyalai-scratch/a09
while kill -0 "$1" 2>/dev/null; do sleep 10; done
shift
systemd-run --user --scope -q -p MemoryMax=1G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python run.py "$@"
