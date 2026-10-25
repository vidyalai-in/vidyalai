P="systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python run.py"
$P naive 1 c13 --workers 1 &
$P edit 1 f06 --workers 1 &
$P compact 1 f06 --workers 1 &
$P trunc 1 c01 --workers 1 &
wait
