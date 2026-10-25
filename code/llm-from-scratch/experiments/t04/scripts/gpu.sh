#!/bin/bash
# usage: gpu.sh <logname> <script.py> <args...>  -- GPU lock + 3G memory cap
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/t04/scripts/run.sh "$@"
