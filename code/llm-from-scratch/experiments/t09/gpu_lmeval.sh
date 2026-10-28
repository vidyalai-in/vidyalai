#!/bin/bash
# lm-evaluation-harness 0.4.13 on GPT-2 small, FP32, zero-shot (run under the GPU lock).
cd /home/amit/.cache/vidyalai-scratch/t09
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=/home/amit/.cache/vidyalai-scratch/t09/lmeval-pkgs
/usr/lib/wsl/lib/nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu \
    --format=csv,noheader,nounits -lms 1000 > logs/smi_lmeval.csv &
SMI=$!
start=$(date +%s)
/home/amit/.cache/vidyalai-scratch/gpu/.venv/bin/python -m lm_eval --model hf \
    --model_args pretrained=openai-community/gpt2,dtype=float32 \
    --tasks hellaswag,arc_easy,piqa --num_fewshot 0 --device cuda:0 --batch_size 32 \
    --output_path post/lmeval --log_samples; rc=$?
echo "lmeval wall $(( $(date +%s) - start )) s" >> logs/gpu_wall.log
kill $SMI; wait $SMI 2>/dev/null
exit $rc
