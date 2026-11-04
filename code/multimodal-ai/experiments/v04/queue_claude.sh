#!/bin/bash
cd /home/amit/.cache/vidyalai-scratch/v04
PY="systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 /home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python"
LIMIT=100 TERSE=1 TASKS=count,dense $PY claude_run.py batch claude_terse.jsonl 448 896
LIMIT=100 THINK=default MAX_TOKENS=4000 TASKS=count,dense,read $PY claude_run.py batch claude_haiku_think.jsonl 224 448 896
for s in 224 448 896; do
  LIMIT=100 MODEL=claude-sonnet-5-5 THINK=between_tools MAX_TOKENS=1000 TASKS=read,count,dense $PY claude_run.py batch claude_sonnet.jsonl $s
done
echo CLAUDE DONE
