#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v05
until grep -q "^lin1 " gpu_minutes.log 2>/dev/null; do sleep 10; done
cp runs/fm_linear/ckpt.pt runs/fm_linear/ckpt_c1.pt
until [ $(free -m | awk "/Mem/{print \$7}") -ge 4500 ]; do sleep 30; done
./gpu.sh cos1 train_fm.py --sched cosine --ch 32 --nres 2 --cl 0 --compile 1 --minutes 10.5 --total 20000 > runs/cos1.log 2>&1
until [ $(free -m | awk "/Mem/{print \$7}") -ge 4500 ]; do sleep 30; done
./gpu.sh lin2 train_fm.py --sched linear --ch 32 --nres 2 --cl 0 --compile 1 --minutes 10.5 --total 20000 > runs/lin2.log 2>&1
echo chain1 done
