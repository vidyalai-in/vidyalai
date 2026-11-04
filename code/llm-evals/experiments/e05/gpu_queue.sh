#!/bin/bash
# Runs the remaining GPU jobs one at a time, each under the GPU lock, after checking free memory.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1
cd /home/amit/.cache/vidyalai-scratch/e05/scripts
PY=/home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python
run() {  # run <log> <args...>
  for i in $(seq 1 30); do
    avail=$(free -m | awk '/Mem:/{print $7}')
    [ "$avail" -ge 2800 ] && break
    echo "waiting for memory ($avail MB)"; sleep 60
  done
  echo "$(date +%T) start $*" >> ../runs/gpu-times.log
  flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "echo \$(date +%T) lock $1 >> ../runs/gpu-times.log; systemd-run --user --scope -q -p MemoryMax=2500M -p MemorySwapMax=0 timeout 840 $PY ${*:2} > ../runs/$1.log 2>&1; echo \$(date +%T) end $1 \$? >> ../runs/gpu-times.log"
}
run eval-contam 04_eval.py contam ../runs/ckpt-contam 100
for m in qwen05 smol135 pythia160 ours88; do run items-$m 05_offshelf.py items $m; done
for b in arc mmlu gsm8k; do run shard-qwen05-$b 05_offshelf.py shard qwen05 $b 50; done
