#!/bin/bash
# Run each open model when memory allows; retry every 60 s, give up after 45 min.
cd /home/amit/.cache/vidyalai-scratch/e02
start=$(date +%s)
for m in HuggingFaceTB/SmolLM2-135M EleutherAI/pythia-160m Qwen/Qwen2.5-0.5B unsloth/Llama-3.2-1B Qwen/Qwen2.5-1.5B-Instruct google/gemma-2-2b; do
  tag=${m##*/}
  [ -f runs/open-$tag.json ] && continue
  while true; do
    scripts/run_open.sh "$m" 16; rc=$?
    [ $rc -ne 3 ] && break
    [ $(( $(date +%s) - start )) -gt 2700 ] && { echo "gave up waiting for memory"; exit 3; }
    sleep 60
  done
  echo "$m exit $rc"
done
