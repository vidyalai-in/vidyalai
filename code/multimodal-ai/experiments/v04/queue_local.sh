#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v04
S=HuggingFaceTB/SmolVLM-256M-Instruct; Q=Qwen/Qwen2-VL-2B-Instruct
for pass in 1 2; do
  ./gpu.sh smol-acc-$pass run_local.py $S local_smol.jsonl --sizes 224 448 896
  ./gpu.sh qwen-acc-$pass run_local.py $Q local_qwen.jsonl --sizes 224 448 896
done
./gpu.sh smol-le512 run_local.py $S local_smol.jsonl --sizes 896 --longest-edge 512
echo QUEUE DONE
