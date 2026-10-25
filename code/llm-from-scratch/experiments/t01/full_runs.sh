#!/bin/bash
# full optimizer steps at 524,288 tokens/step, largest micro-batch that fits; a few minutes per size
D=/home/amit/.cache/vidyalai-scratch/t01
for spec in "t20 32 30" "t50 16 14" "t85 16 9" "t125 8 7"; do
  set -- $spec
  timeout 900 flock /home/amit/.cache/vidyalai-scratch/gpu.lock systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 $D/run.sh full_$1 --size $1 --mb $2 --steps $3
  grep RESULT $D/logs/full_$1.out || echo "$1 failed"
done
echo ALLDONE
