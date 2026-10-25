#!/bin/bash
# usage: gpu.sh <logname> <python args...>  -- GPU lock + 4G memory cap
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/t05/scripts/run.sh "$@"
