#!/bin/bash
# GPU chunk B: 14x14 tokenizer (K=512, restarts) for the priors, encode codes, prior speed test.
cd /home/amit/.cache/vidyalai-scratch/v08
P="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 timeout 800 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
$P vq_sweep.py 2 4000 512 200
$P prior_run.py codes /home/amit/.cache/vidyalai-scratch/v08/ckpt/vq_d2_K512_r200_s0_4000.pt 512 2
$P prior_speed.py
