#!/bin/bash
# Runs the three main training jobs, each as its own locked command (lock released between them).
cd /home/amit/.cache/vidyalai-scratch/v07
P=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
for job in "ddpm-unet --model unet --obj ddpm" "rf-unet --model unet --obj flow" "rf-dit --model dit --obj flow"; do
  set -- $job; name=$1; shift
  until [ $(free -m | awk '/Mem:/{print $7}') -ge 4500 ]; do sleep 30; done
  echo "$(date +%T) waiting lock $name"
  flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
    bash gpu.sh train-$name timeout 840 $P train.py --name $name "$@" --steps 4000 --snap 500 --minutes 12
  echo "$(date +%T) done $name rc=$?"
done
