#!/bin/bash
# GPU chunk A: VQ-VAE codebook sweep on the 7x7 grid, 3,000 steps, with and without dead-code restarts.
cd /home/amit/.cache/vidyalai-scratch/v08
P="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 timeout 800 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
$P vq_sweep.py 4 3000 16,64,512,4096 0
$P vq_sweep.py 4 3000 16,64,512,4096 200
