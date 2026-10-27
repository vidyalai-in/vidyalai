#!/bin/bash
# usage: gpu.sh <logname> <python args...>  -- GPU lock + 6G memory cap
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/t07/scripts/run.sh "$@"
