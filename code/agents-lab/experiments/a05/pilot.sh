P="systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/agents/.venv/bin/python run.py"
$P naive 1 c10 --workers 1 &
$P edit 1 c05 c10 --workers 2 &
wait
$P compact 1 c05 c10 --workers 2 &
$P memory 1 c05 c10 --workers 2 &
wait
