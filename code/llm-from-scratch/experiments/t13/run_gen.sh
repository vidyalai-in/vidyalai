#!/bin/bash
# Teacher generation for Part 13: 2,800 more training prompts (after the 512 of the pilot) and 300 validation prompts.
S=/home/amit/.cache/vidyalai-scratch; G=$S/t13/scripts/gpu_run.sh; D=$S/t13/data
lock() { flock $S/gpu.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 "$@"; }
lock bash $G gen_train $S/t13/scripts/gen_teacher.py train 512 2800 64 512 $D/gen_train.jsonl
lock bash $G gen_val $S/t13/scripts/gen_teacher.py val 0 300 64 512 $D/gen_val.jsonl
