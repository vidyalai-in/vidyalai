#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/p05/scripts
./locked.sh 6G ../runs/rome.log rome.py; echo "rome rc=$?"
./locked.sh 7G ../runs/trace_gpt2-medium.log trace.py gpt2-medium; echo "medium rc=$?"
./locked.sh 7G ../runs/trace_qwen.log trace.py qwen; echo "qwen rc=$?"
