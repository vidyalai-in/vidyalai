#!/bin/bash
# GPU chunk C: codes from the 7x7 K=512 tokenizer, AR and masked priors (2,500 steps each), FD sampling, timing.
cd /home/amit/.cache/vidyalai-scratch/v08
P="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 timeout 600 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
$P prior_run.py codes /home/amit/.cache/vidyalai-scratch/v08/ckpt/vq_d4_K512_r0_s0_3000.pt 512 4
$P prior_run.py train ar 2500 2500
$P prior_run.py train mask 2500 2500
$P prior_run.py sample
$P prior_run.py time
