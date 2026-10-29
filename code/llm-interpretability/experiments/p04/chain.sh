#!/bin/bash
# runs the remaining measurements one locked command at a time
cd /home/amit/.cache/vidyalai-scratch/p04
r() { local log=$1; shift; TMO=840 scripts/run.sh 4G "$@" > runs/$log 2>&1; echo "$log: $(grep -c . runs/$log) lines, $(grep -E '^done|Error|Terminated' runs/$log | tail -1)"; }
r log_listing.txt scripts/listing.py
r log_s2_gpt2_dla.txt scripts/s2_ablate.py gpt2 dla
r log_s2_gpt2_kpath.txt scripts/s2_ablate.py gpt2 kpath
r log_s3_gpt2_rep.txt scripts/s3_lite.py gpt2 rep
r log_s1_pythia.txt scripts/s1_scores.py pythia
r log_s3_pythia_rep.txt scripts/s3_lite.py pythia rep
r log_s1_qwen.txt scripts/s1_scores.py qwen
r log_s3_qwen_rep.txt scripts/s3_lite.py qwen rep
r log_s3_pythia_nat.txt scripts/s3_lite.py pythia nat
export NWIN=400 NRAND=1; r log_s3_qwen_nat.txt scripts/s3_lite.py qwen nat
echo chain-done
