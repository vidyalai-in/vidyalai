#!/bin/bash
# usage: gpu.sh <label> <python args...>
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
cd /home/amit/.cache/vidyalai-scratch/v04
flock /home/amit/.cache/vidyalai-scratch/gpu.lock ./gpu_inner.sh "$@"
