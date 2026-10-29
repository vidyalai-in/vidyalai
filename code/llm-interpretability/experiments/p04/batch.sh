#!/bin/bash
# batch.sh <step>...: runs several short measurement steps inside ONE lock hold (each step has its own timeout)
cd /home/amit/.cache/vidyalai-scratch/p04
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python
echo "lock acquired $(date -Is)"
s0=$(date +%s)
while [ $# -gt 0 ]; do
  EXTRA=""; case $1 in *=*) EXTRA=$1; shift;; esac; log=$1; shift; IFS=' ' read -r -a args <<< "$1"; shift
  t=$(date +%s)
  env $EXTRA timeout 300 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 $PY "${args[@]}" > runs/$log 2>&1
  echo "$log rc=$? $(( $(date +%s)-t ))s $(grep -E '^done|Error|Killed' runs/$log | tail -1)"
done
echo "lock held $(( $(date +%s)-s0 ))s"
