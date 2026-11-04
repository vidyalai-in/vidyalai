#!/bin/bash
# runs while holding the GPU lock: memory check, then the job in a 4G scope; logs locked time
label=$1; shift
avail=$(free -m | awk '/Mem:/{print $7}')
echo "$(date +%s) start $label avail=$avail" >> gpu_minutes.log
if [ "$avail" -lt 4500 ]; then echo "$(date +%s) skip $label avail=$avail" >> gpu_minutes.log; exit 99; fi
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python "$@"
rc=$?
echo "$(date +%s) end $label rc=$rc" >> gpu_minutes.log
exit $rc
