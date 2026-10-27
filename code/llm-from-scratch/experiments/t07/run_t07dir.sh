#!/bin/bash
# like run.sh, but runs from the t07 folder (where runs/main88 is), for the article's snippet
export LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONPATH=/home/amit/.cache/vidyalai-scratch/fromscratch
cd /home/amit/.cache/vidyalai-scratch/t07
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python "$@"
