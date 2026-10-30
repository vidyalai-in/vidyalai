#!/bin/bash
for r in s11 s15 s22 s33 s47 iso_s11 iso_s22; do
  /home/amit/.cache/vidyalai-scratch/interp/.venv/bin/python scripts/track.py $r 150
done
