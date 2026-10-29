#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/p05/scripts
./locked.sh 7G ../runs/trace_qwen.log trace.py qwen; echo "qwen rc=$?"
./locked.sh 6G ../runs/rome.log rome.py; echo "rome rc=$?"
