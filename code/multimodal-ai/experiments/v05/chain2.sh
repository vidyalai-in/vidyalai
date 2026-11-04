#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v05
until grep -q "chain1 done" runs/chain1.log 2>/dev/null; do sleep 10; done
until [ $(free -m | awk "/Mem/{print \$7}") -ge 4500 ]; do sleep 30; done
./gpu.sh sweep1 sweep_all.py --minutes 11 > runs/sweep1.log 2>&1
if grep -q "skip (time)" runs/sweep1.log; then
  until [ $(free -m | awk "/Mem/{print \$7}") -ge 4500 ]; do sleep 30; done
  ./gpu.sh sweep2 sweep_all.py --minutes 8 > runs/sweep2.log 2>&1
fi
echo chain2 done
