#!/bin/bash
# usage: run.sh <tag> <mem cap> <python args...>  -- takes the GPU lock, caps host memory, logs nvidia-smi
tag=$1; cap=$2; shift 2
exec flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 \
    bash /home/amit/.cache/vidyalai-scratch/t14/scripts/gpu_run.sh "$tag" "$@"
