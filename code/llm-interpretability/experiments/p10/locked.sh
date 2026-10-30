#!/bin/bash
# Usage: locked.sh LOGNAME MEM TIMEOUT cmd...  -- runs cmd under the GPU lock, logging lock-held seconds.
cd /home/amit/.cache/vidyalai-scratch/p10
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
LOG=runs/$1.log; MEM=$2; TO=$3; shift 3
flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "a=\$(date +%s); systemd-run --user --scope -q -p MemoryMax=$MEM -p MemorySwapMax=0 timeout $TO $* >> $LOG 2>&1; e=\$?; b=\$(date +%s); echo \"$1 \$a \$b \$((b-a)) exit \$e\" >> runs/gpu_seconds.txt"
