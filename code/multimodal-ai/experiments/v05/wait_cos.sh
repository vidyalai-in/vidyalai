#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v05
until grep -q "^cos1 " gpu_minutes.log; do sleep 10; done
python3 -c "import json; s=json.load(open('runs/fm_cosine/state.json')); print(s['step'])" > runs/fm_linear/total_override
cat runs/fm_linear/total_override
