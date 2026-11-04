#!/bin/bash
# Remaining locked GPU jobs for Part 11, each its own short lock. Times logged per chunk.
cd /home/amit/.cache/vidyalai-scratch/v11
export LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HOME=/home/amit/.cache/vidyalai-scratch/hf
P=/home/amit/.cache/vidyalai-scratch/mm/.venv/bin/python
until [ -f runs/lock_s02b.end ]; do sleep 3; done
run() { # name, args...
  name=$1; shift
  while [ $(free -m | awk '/Mem:/{print $7}') -lt 4500 ]; do sleep 20; done
  flock /home/amit/.cache/vidyalai-scratch/gpu.lock bash -c "date +%s > runs/lock_$name.start; systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 timeout 840 $P $* > runs/$name.log 2>&1; echo rc=\$? >> runs/$name.log; date +%s > runs/lock_$name.end"
  echo "$name locked $(( $(cat runs/lock_$name.end) - $(cat runs/lock_$name.start) )) s"
}
run s02c scripts/s02_codec.py --libri mimi dac
run s06 scripts/s06_asr.py
until [ $(grep -c '"ttft"' runs/claude_stream.jsonl) -ge 120 ] || ! pgrep -f s04_claude_stream >/dev/null; do sleep 5; done
run s05c scripts/s05_tts.py runs/claude_stream.jsonl
run s07 scripts/s07_qwen.py
run s05q scripts/s05_tts.py runs/qwen_stream.jsonl
echo all-done
