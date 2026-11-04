#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v04
until grep -q "QUEUE DONE" queue_local.log; do sleep 10; done
S=HuggingFaceTB/SmolVLM-256M-Instruct; Q=Qwen/Qwen2-VL-2B-Instruct
./gpu.sh qwen-time run_local.py $Q timing.jsonl --time --sizes 224 448 896 1344
./gpu.sh smol-time run_local.py $S timing.jsonl --time --sizes 224 896
./gpu.sh smol-time-512 run_local.py $S timing.jsonl --time --sizes 896 --longest-edge 512
./gpu.sh smol-time-1024 run_local.py $S timing.jsonl --time --sizes 896 --longest-edge 1024
echo TIME DONE
