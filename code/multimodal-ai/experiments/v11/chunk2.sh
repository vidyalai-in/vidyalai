#!/bin/bash
# One locked chunk for the remaining GPU work of Part 11 (each step under its own timeout).
cd /home/amit/.cache/vidyalai-scratch/v11
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
P=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
step() { name=$1; shift; s=$(date +%s); timeout 300 $P "$@" > runs/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s) - s )) s" >> runs/chunk2.log; }
date +%s > runs/lock_chunk2.start
step s06 scripts/s06_asr.py
step s05c scripts/s05_tts.py runs/claude_stream.jsonl
step s07 scripts/s07_qwen.py
step s05q scripts/s05_tts.py runs/qwen_stream.jsonl
cd snip; s=$(date +%s); timeout 120 $P mimi_snip.py > ../runs/snip.log 2>&1; echo "snip rc=$? $(( $(date +%s) - s )) s" >> ../runs/chunk2.log; cd ..
step s02c scripts/s02_codec.py --libri mimi dac
date +%s > runs/lock_chunk2.end
